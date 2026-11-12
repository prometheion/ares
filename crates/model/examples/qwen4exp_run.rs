//! Runs a `qwen4exp` model file on the GPUs and measures it: a tool for
//! development, not a part of the server.
//!
//! ```text
//! qwen4exp_run --model FIRST_PART.gguf [--devices 0,1] [--cells 4096]
//!     [--ram-gib 8] [--headroom-mib 256] [--bounds 0,24,48] [--keys f32|f16]
//!     [--spread auto|layers|tensor] [--mixer 0] [--core-share 0.6] [--shared mixer|worker]
//!     [--owners rule|routing] [--worker-picks 0.35]
//!     [--placement] [--read-time] [--profile A.q4route,B.q4route]
//!     [--check DIR/NAME,...] [--chat] [--generate 64]
//!     [--text TOKENS.txt] [--text-file A.md,B.md] [--decode-at 1000,8000]
//!     [--decode-steps 64]
//!     [--prefill 2048] [--breakdown] [--serve model|stream|cpu] [--transit 32]
//!     [--read-ahead on|off]
//! ```
//!
//! - `--spread`: how two devices share the model: a range of layers each
//!   (`layers`), or both on every layer (`tensor`; `--mixer` is the index of
//!   the device that holds the streams, `--core-share` its share of the
//!   recurrent heads, `--shared` who computes the shared expert; what is not
//!   given is measured at load). `auto`, the default, takes `tensor` when
//!   the devices can write each other's memory. `--owners routing` divides
//!   the experts between the two by the routing `--profile` gives instead of
//!   the fixed rule; `--worker-picks` is the share of the picks the worker's
//!   experts are to get then.
//! - `--read-time`: times a read of a stream mixer on every device, as the
//!   session does at load, and ends.
//! - `--placement`: prints where the layers would go and what every device
//!   would hold, and ends without loading.
//! - `--profile`: recorded routing (`q4trace` files); the tiers are filled
//!   with the experts it used, most used first, before anything runs.
//! - `--check DIR/NAME`: the tokens of `DIR/NAME.tokens.txt`, of which
//!   `DIR/NAME.q4route` says how many were given: the logits of the last
//!   given token against `DIR/NAME.prompt-logits.f32` (the CPU reference's),
//!   then greedy tokens against the rest of the file.
//! - `--chat`: a German and an English question through the chat template,
//!   greedy, decoded with the tokenizer.
//! - `--text`, `--text-file`: the text that contexts and prompts are taken
//!   from: a file of token ids, then plain text files, which are tokenized.
//! - `--decode-at N`: milliseconds per decode step with `N` tokens of the
//!   text as the context; `--breakdown` adds device time by kind (the
//!   timed events cost a part of a step themselves, so steps are measured
//!   both ways).
//! - `--prefill N`: reads `N` tokens of `--text` from an empty sequence and
//!   reports tokens per second, in steps of `--rows` rows (default 8, the decode
//!   kernels' limit; more runs the many-row kernels and reports what the wide steps
//!   found missing and waited for); `--repeat K` reads it `K` times, the later ones
//!   with the experts the first one fetched.
//! - `--transit N`: transit slots per device for the experts a wide step misses, to
//!   measure what they are worth; 0 brings them into the tier's slots, and without the
//!   option the plan of the store decides. `--read-ahead on` starts the reads of the
//!   layers ahead on a guess (what the wide step before missed there), to measure what the
//!   guesses are worth.
//! - `--serve`: how an expert that is in no slot is served: `model` lets the
//!   store's measured costs decide between an upload and the CPU (what the
//!   engine does), `stream` uploads every one, `cpu` computes every one.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_cuda::{Device, KeyFormat, Wait};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    DeviceMemory, Options, PHASES, Placement, Profile, Rows, Session, SessionOptions, Spread,
    TensorChoice, Wanted, read_time,
};
use ares_model::qwen4exp::{Config, Weights};
use ares_runtime::expert_store::{Serve, StoreStats};
use ares_tokenizer::{ChatOptions, EncodeOptions, Message, Tokenizer};

type AnyError = Box<dyn std::error::Error>;

#[derive(Debug)]
struct Args {
    model: String,
    devices: Vec<u32>,
    cells: u32,
    ram_gib: Option<f64>,
    headroom_mib: u64,
    bounds: Option<Vec<usize>>,
    spread: String,
    mixer: Option<usize>,
    core_share: Option<f64>,
    shared: Option<String>,
    owners: String,
    worker_picks: Option<f64>,
    keys: KeyFormat,
    placement: bool,
    read_time: bool,
    profile: Vec<String>,
    check: Vec<String>,
    chat: bool,
    generate: usize,
    text: Option<String>,
    text_file: Vec<String>,
    decode_at: Vec<usize>,
    decode_steps: usize,
    prefill: usize,
    repeat: usize,
    rows: u32,
    breakdown: bool,
    serve: Serve,
    stall_s: u64,
    transit: Option<u32>,
    read_ahead: bool,
}

fn parse() -> Result<Args, AnyError> {
    let mut args = Args {
        model: String::new(),
        devices: vec![0],
        cells: 4096,
        ram_gib: None,
        headroom_mib: 256,
        bounds: None,
        spread: "auto".into(),
        mixer: None,
        core_share: None,
        shared: None,
        owners: "rule".into(),
        worker_picks: None,
        keys: KeyFormat::F32,
        placement: false,
        read_time: false,
        profile: Vec::new(),
        check: Vec::new(),
        chat: false,
        generate: 64,
        text: None,
        text_file: Vec::new(),
        decode_at: Vec::new(),
        decode_steps: 64,
        prefill: 0,
        repeat: 1,
        rows: 8,
        breakdown: false,
        serve: Serve::Measured,
        stall_s: 20,
        transit: None,
        read_ahead: false,
    };
    let mut it = std::env::args().skip(1);
    let list = |s: String| -> Vec<String> { s.split(',').map(str::to_owned).collect() };
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => args.model = value()?,
            "--devices" => {
                args.devices = list(value()?).iter().map(|d| d.parse()).collect::<Result<_, _>>()?
            }
            "--cells" => args.cells = value()?.parse()?,
            "--ram-gib" => args.ram_gib = Some(value()?.parse()?),
            "--headroom-mib" => args.headroom_mib = value()?.parse()?,
            "--bounds" => {
                args.bounds =
                    Some(list(value()?).iter().map(|d| d.parse()).collect::<Result<_, _>>()?)
            }
            "--keys" => {
                args.keys = match value()?.as_str() {
                    "f32" => KeyFormat::F32,
                    "f16" => KeyFormat::F16,
                    other => return Err(format!("--keys {other}: f32 or f16").into()),
                }
            }
            "--spread" => args.spread = value()?,
            "--mixer" => args.mixer = Some(value()?.parse()?),
            "--core-share" => args.core_share = Some(value()?.parse()?),
            "--shared" => args.shared = Some(value()?),
            "--owners" => args.owners = value()?,
            "--worker-picks" => args.worker_picks = Some(value()?.parse()?),
            "--placement" => args.placement = true,
            "--read-time" => args.read_time = true,
            "--profile" => args.profile = list(value()?),
            "--check" => args.check = list(value()?),
            "--chat" => args.chat = true,
            "--generate" => args.generate = value()?.parse()?,
            "--text" => args.text = Some(value()?),
            "--text-file" => args.text_file = list(value()?),
            "--decode-at" => {
                args.decode_at =
                    list(value()?).iter().map(|d| d.parse()).collect::<Result<_, _>>()?
            }
            "--decode-steps" => args.decode_steps = value()?.parse()?,
            "--prefill" => args.prefill = value()?.parse()?,
            "--rows" => args.rows = value()?.parse()?,
            "--repeat" => args.repeat = value()?.parse()?,
            "--breakdown" => args.breakdown = true,
            "--stall-s" => args.stall_s = value()?.parse()?,
            "--transit" => args.transit = Some(value()?.parse()?),
            "--read-ahead" => {
                args.read_ahead = match value()?.as_str() {
                    "on" => true,
                    "off" => false,
                    other => return Err(format!("--read-ahead {other}: on or off").into()),
                }
            }
            "--serve" => {
                args.serve = match value()?.as_str() {
                    "stream" => Serve::Stream,
                    "cpu" => Serve::Cpu,
                    "model" => Serve::Measured,
                    other => return Err(format!("--serve {other}: stream, cpu or model").into()),
                }
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if args.model.is_empty() {
        return Err("--model FILE is needed".into());
    }
    Ok(args)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn load_average() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "?".into())
}

fn tokens_of(path: &str) -> Result<Vec<u32>, AnyError> {
    Ok(std::fs::read_to_string(path)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?)
}

/// A recorded routing: how many tokens were given, and per layer how often
/// every expert was picked.
fn routing(path: &str, counts: &mut [Vec<u32>]) -> Result<usize, AnyError> {
    let bytes = std::fs::read(path)?;
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    if bytes.len() < 64 || &bytes[..8] != b"Q4ROUTE1" {
        return Err(format!("{path} is not a routing trace").into());
    }
    let (layers, k, tokens, given, size) = (word(12), word(20), word(24), word(28), word(32));
    if layers != counts.len() || bytes.len() < 64 + 4 * tokens + tokens * layers * size {
        return Err(
            format!("{path}: {layers} layers, {tokens} tokens, {} bytes", bytes.len()).into()
        );
    }
    for token in 0..tokens {
        for (layer, counts) in counts.iter_mut().enumerate() {
            let at = 64 + 4 * tokens + (token * layers + layer) * size;
            for pick in 0..k {
                let id = u16::from_le_bytes([bytes[at + 2 * pick], bytes[at + 2 * pick + 1]]);
                if let Some(count) = counts.get_mut(id as usize) {
                    *count += 1;
                }
            }
        }
    }
    Ok(given)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
}

fn percentile(values: &[f32], p: f64) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted.get(((sorted.len() as f64 - 1.0) * p).round() as usize).copied().unwrap_or(f32::NAN)
}

/// What the store counted over `steps` steps: how the picks were served.
fn store_line(stats: &StoreStats, steps: usize) -> String {
    let per = |n: u64| n as f64 / steps.max(1) as f64;
    let misses = stats.picks.saturating_sub(stats.hits);
    format!(
        "{:.2} % of picks in a slot; per step {:.1} experts outside ({:.1} of them computed on \
         the CPU, {:.1} uploaded for the step); where their bytes were: {:.1} in RAM, {:.1} in the \
         file ({:.1} MB read), {:.1} on their way; waiting per step {:.2} ms summed over layers; \
         a miss served from RAM {:.2} ms (median) / {:.2} (99 %), from the file {:.2} / {:.2} ms, a \
         layer's CPU share {:.2} / {:.2} ms; between steps {:.1} experts promoted to VRAM per step",
        100.0 * stats.hits as f64 / stats.picks.max(1) as f64,
        per(misses),
        per(stats.cpu_experts),
        per(misses.saturating_sub(stats.cpu_experts)),
        per(stats.miss_from_ram),
        per(stats.miss_from_disk),
        per(stats.demand_read_bytes) / 1e6,
        per(stats.miss_on_its_way),
        stats.wait_ms / steps.max(1) as f64,
        percentile(&stats.ram_miss_ms, 0.5),
        percentile(&stats.ram_miss_ms, 0.99),
        percentile(&stats.disk_miss_ms, 0.5),
        percentile(&stats.disk_miss_ms, 0.99),
        percentile(&stats.cpu_ms, 0.5),
        percentile(&stats.cpu_ms, 0.99),
        per(stats.promotions),
    )
}

/// What the tiers moved: experts uploaded into slots, evicted, copied down to RAM and read
/// from the file, and what one upload takes on each device by the store's cost model.
fn churn_line(stats: &StoreStats, session: &Session<'_>) -> String {
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    let link: Vec<String> = (0..session.devices().len())
        .map(|device| {
            match session.store().cost_model().and_then(|model| model.upload_time(device)) {
                Some(seconds) => format!("{:.2}", seconds * 1e3),
                None => "?".to_owned(),
            }
        })
        .collect();
    format!(
        "churn: {} uploads into slots ({:.2} GiB), {} evictions, {} copied down to RAM ({:.2} \
         GiB), {} reads from the file for a step ({:.2} GiB), {} guessed reads ({:.2} GiB), {} \
         promotions; an upload takes {} ms per device by the cost model",
        stats.uploads,
        gib(stats.upload_bytes),
        stats.evictions,
        stats.demotions,
        gib(stats.demote_bytes),
        stats.demand_reads,
        gib(stats.demand_read_bytes),
        stats.guess_reads,
        gib(stats.guess_read_bytes),
        stats.promotions,
        link.join(" | "),
    )
}

/// What one decode step cost and missed.
struct StepKind {
    /// The step's milliseconds.
    ms: f64,
    /// The experts it picked outside a slot, and of them those read from the
    /// file.
    misses: u64,
    from_file: u64,
    /// The milliseconds the n-gram rows kept it waiting.
    rows_ms: f64,
}

/// What the decode steps cost by what they missed: steps that picked no
/// expert outside a slot, steps whose misses all came from RAM (or were
/// computed by the CPU), and steps with at least one expert read from the file.
fn step_classes(steps: &[StepKind]) -> String {
    let class = |keep: &dyn Fn(&StepKind) -> bool| {
        let picked: Vec<_> = steps.iter().filter(|s| keep(s)).collect();
        if picked.is_empty() {
            return "none".to_owned();
        }
        let n = picked.len() as f64;
        let mut ms: Vec<f64> = picked.iter().map(|s| s.ms).collect();
        format!(
            "{} steps ({:.0} %), median {:.2} ms, {:.1} misses and {:.2} ms of n-gram rows each",
            picked.len(),
            100.0 * n / steps.len() as f64,
            median(&mut ms),
            picked.iter().map(|s| s.misses as f64).sum::<f64>() / n,
            picked.iter().map(|s| s.rows_ms).sum::<f64>() / n,
        )
    };
    format!(
        "  steps by what they missed: no expert outside a slot: {}; misses served from RAM or the \
         CPU only: {}; at least one read from the file: {}",
        class(&|s| s.misses == 0),
        class(&|s| s.misses > 0 && s.from_file == 0),
        class(&|s| s.from_file > 0),
    )
}

/// Reads `tokens` as certain rows, eight to a step, and returns the pick
/// after the last one when `want_last`.
fn read(
    session: &mut Session<'_>,
    tokens: &[u32],
    want_last: bool,
) -> Result<Option<u32>, AnyError> {
    let mut pick = None;
    let rows = session.options().part.max_rows as usize;
    let chunks = tokens.chunks(rows).count();
    for (i, chunk) in tokens.chunks(rows).enumerate() {
        let wanted = if want_last && i + 1 == chunks { Wanted::Last } else { Wanted::None };
        let picks = session.step(&[Rows { slot: 0, tokens: chunk, direct: true, wanted }])?;
        pick = picks.first().copied();
    }
    Ok(pick)
}

/// Generates up to `n` tokens greedily from `first`, the pick after the
/// context, and returns them with the time of every step.
fn generate(
    session: &mut Session<'_>,
    first: u32,
    n: usize,
    ends: &dyn Fn(u32) -> bool,
) -> Result<(Vec<u32>, Vec<f64>), AnyError> {
    let mut tokens = vec![first];
    let mut ms = Vec::new();
    while tokens.len() < n && !ends(tokens[tokens.len() - 1]) {
        let token = [tokens[tokens.len() - 1]];
        let picks = session.step(&[Rows {
            slot: 0,
            tokens: &token,
            direct: true,
            wanted: Wanted::Last,
        }])?;
        let pick = picks[0];
        ms.push(session.last_time().wall.as_secs_f64() * 1e3);
        tokens.push(pick);
    }
    Ok((tokens, ms))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen4exp_run: {error}");
        // Without unwinding: after a step that stood still a device may
        // still wait for the host, and freeing its memory would wait too.
        std::process::exit(1);
    }
}

fn run() -> Result<(), AnyError> {
    let args = parse()?;
    let gguf = Gguf::open(&args.model)?;
    let c = Config::from_gguf(&gguf)?;
    let w = Weights::load(&gguf, &c)?;
    let profile = args.breakdown.then(|| Arc::new(Profile::new()));
    if let Some(profile) = &profile {
        // Only the decode steps that are measured for it are timed.
        profile.pause(true);
    }
    // The recorded routing: the tiers are filled from it, and two devices
    // that share the layers divide the experts by it.
    let mut counts = vec![vec![0u32; c.experts.count]; c.n_layer];
    for path in &args.profile {
        routing(path, &mut counts)?;
    }
    let opts = SessionOptions {
        routing: (!args.profile.is_empty() && args.owners == "routing").then(|| counts.clone()),
        worker_picks: args.worker_picks,
        part: Options {
            cells: args.cells,
            keys: args.keys,
            max_rows: args.rows,
            profile: profile.clone(),
            ..Options::default()
        },
        headroom: args.headroom_mib << 20,
        ram_tier: args.ram_gib.map(|g| (g * (1u64 << 30) as f64) as u64),
        spread: match args.spread.as_str() {
            "auto" => Spread::Auto,
            "layers" => Spread::Layers,
            "tensor" => Spread::Tensor {
                mixer: args.mixer,
                core_share: args.core_share,
                shared_on_mixer: args.shared.as_deref().map(|s| s == "mixer"),
            },
            other => return Err(format!("--spread {other}: auto, layers or tensor").into()),
        },
        bounds: args.bounds.clone(),
        serve: args.serve,
        stall: Duration::from_secs(args.stall_s),
        transit: args.transit,
        read_ahead: args.read_ahead,
        ..SessionOptions::default()
    };
    let devices: Vec<Device> =
        args.devices.iter().map(|d| Device::open(*d, Wait::Spin)).collect::<Result<_, _>>()?;

    if args.read_time {
        // What the session measures at load to choose the mixer.
        for device in &devices {
            let mut us: Vec<f64> = (0..5)
                .map(|_| read_time(device, &c, &w).map(|t| t.as_secs_f64() * 1e6))
                .collect::<Result<_, _>>()?;
            us.sort_by(f64::total_cmp);
            println!(
                "a read of a stream mixer on {}: {:.1} us (fastest of 5 times 64, slowest {:.1})",
                device.info().name,
                us[0],
                us[4]
            );
        }
        return Ok(());
    }
    if args.placement {
        let memory: Vec<DeviceMemory> = devices
            .iter()
            .map(|d| {
                Ok(DeviceMemory {
                    budget: d.mem_info()?.free.saturating_sub(opts.headroom),
                    multiprocessors: d.info().multiprocessors,
                })
            })
            .collect::<Result<_, ares_cuda::Error>>()?;
        let placement = match (&opts.spread, &opts.bounds) {
            // Without a session nothing is measured: what is not given is
            // the first device and an even share.
            (Spread::Tensor { mixer, core_share, shared_on_mixer }, _) => Placement::tensor(
                &c,
                &w,
                &opts.part,
                &memory,
                &TensorChoice {
                    mixer: mixer.unwrap_or(0),
                    core_share: core_share.unwrap_or(0.5),
                    shared_on_mixer: shared_on_mixer.unwrap_or(false),
                },
            )?,
            (_, Some(bounds)) => Placement::at(&c, &w, &opts.part, &memory, bounds)?,
            _ => Placement::plan(&c, &w, &opts.part, &memory)?,
        };
        println!("placement for {} cells:", args.cells);
        for (device, plan) in devices.iter().zip(&placement.devices) {
            println!(
                "  {} ({:.0} MiB to give): layers {:?}, dense weights {:.0} MiB, recurrent state \
                 {:.1} MiB, K/V and keys {:.0} MiB, scratch {:.1} MiB, expert slots {:.0} MiB = \
                 {:.1} % of its layers' experts",
                device.info().name,
                mib(memory[placement.devices.iter().position(|p| p == plan).unwrap_or(0)].budget),
                plan.layers,
                mib(plan.weights),
                mib(plan.state),
                mib(plan.pools),
                mib(plan.step),
                mib(plan.slots),
                100.0 * plan.slot_share(),
            );
        }
        return Ok(());
    }

    let opening = Instant::now();
    let mut session = Session::open(&gguf, &c, &w, devices, opts)?;
    println!(
        "opened in {:.1} s (load average {}):",
        opening.elapsed().as_secs_f64(),
        load_average(),
    );
    let ram_slots = session.ram_slots();
    if let Some(tensor) = &session.placement().tensor {
        let heads = |device: usize| -> u32 {
            tensor.shares[device].iter().map(|share| share.heads.len() as u32).sum()
        };
        let shared = tensor.shares[tensor.mixer].first().is_some_and(|share| share.shared);
        println!(
            "  both devices on every layer: device {} holds the streams and the mixers; heads \
             over all layers {} | {}, shares of the slots {:.3} | {:.3}, the shared expert on \
             the {}, experts divided by {}",
            tensor.mixer,
            heads(0),
            heads(1),
            tensor.expert_share[0],
            tensor.expert_share[1],
            if shared { "mixer" } else { "worker" },
            if tensor.owners.is_some() { "the recorded routing" } else { "the rule" },
        );
    }
    for part in session.report() {
        println!(
            "  {} ({:?}): layers {:?}; dense weights {:.0} MiB (uploaded in {:.1} s), {} expert slots in \
             {:.0} MiB, recurrent state {:.1} MiB per sequence, K/V and keys {:.2} MiB per 1000 \
             cells (keys {:.2}), scratch {:.1} MiB; reserved {:.0} MiB",
            part.device,
            part.role,
            part.layers,
            mib(part.weights),
            part.upload.as_secs_f64(),
            part.n_slots,
            mib(part.slots),
            mib(part.state_per_slot),
            mib(part.kv_per_1000),
            mib(part.keys_per_1000),
            mib(part.scratch),
            mib(part.reserved),
        );
    }
    println!("  RAM tier: {ram_slots} slots");

    if !args.profile.is_empty() {
        let mut counts = vec![vec![0u32; c.experts.count]; c.n_layer];
        for path in &args.profile {
            routing(path, &mut counts)?;
        }
        let fill = session.fill(&mut |layer| {
            let mut order: Vec<u32> =
                (0..c.experts.count as u32).filter(|e| counts[layer][*e as usize] > 0).collect();
            order.sort_by_key(|e| std::cmp::Reverse(counts[layer][*e as usize]));
            order
        })?;
        println!(
            "tiers filled from the profile in {:.1} s: {:.2} GiB read ({:.2} GiB/s), {} experts \
             uploaded",
            fill.time.as_secs_f64(),
            fill.read_bytes as f64 / (1u64 << 30) as f64,
            fill.read_bytes as f64 / (1u64 << 30) as f64 / fill.time.as_secs_f64(),
            fill.uploads,
        );
    }

    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let ends = |token: u32| tokenizer.is_end_of_generation(token);

    for name in &args.check {
        let tokens = tokens_of(&format!("{name}.tokens.txt"))?;
        let mut counts = vec![vec![0u32; c.experts.count]; c.n_layer];
        let given = routing(&format!("{name}.q4route"), &mut counts)?;
        let want: Vec<f32> = std::fs::read(format!("{name}.prompt-logits.f32"))?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        session.restart(0)?;
        session.take_store_stats();
        let started = Instant::now();
        let pick = read(&mut session, &tokens[..given], true)?.ok_or("no pick")?;
        let prompt_s = started.elapsed().as_secs_f64();
        let got = session.logits(0).ok_or("no logits")?.to_vec();
        // The logits of the last given token against the reference's.
        let (mut diff, mut size) = (0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(&want) {
            diff += (f64::from(*g) - f64::from(*w)).powi(2);
            size += f64::from(*w).powi(2);
        }
        let top = |logits: &[f32], n: usize| -> Vec<usize> {
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
            order.truncate(n);
            order
        };
        let (ours, theirs) = (top(&got, 10), top(&want, 10));
        let log_softmax = |logits: &[f32]| -> Vec<f64> {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = logits.iter().map(|l| f64::from(l - max).exp()).sum();
            logits.iter().map(|l| f64::from(l - max) - sum.ln()).collect()
        };
        let (lp, lq) = (log_softmax(&want), log_softmax(&got));
        let kl: f64 = lp.iter().zip(&lq).map(|(p, q)| p.exp() * (p - q)).sum();
        // The reference's own tokens are fed one at a time: the pick after
        // each is compared with the reference's next token, so one near tie
        // does not hide what comes after it.
        let mut picks = vec![pick];
        let mut ms = Vec::new();
        // At a difference: how far our logit of the reference's token is
        // below the one we picked.
        let mut margins = Vec::new();
        for at in given..tokens.len() {
            let last = picks[picks.len() - 1];
            if last != tokens[at] {
                // The logits of the step before are still the session's.
                let logits = session.logits(0).ok_or("no logits")?;
                margins.push((at, logits[last as usize] - logits[tokens[at] as usize]));
            }
            if at + 1 == tokens.len() {
                break;
            }
            let token = [tokens[at]];
            let step = session.step(&[Rows {
                slot: 0,
                tokens: &token,
                direct: true,
                wanted: Wanted::Last,
            }])?;
            picks.push(step[0]);
            ms.push(session.last_time().wall.as_secs_f64() * 1e3);
        }
        let n_new = tokens.len() - given;
        let equal = picks.iter().zip(&tokens[given..]).take_while(|(a, b)| a == b).count();
        let same = picks.iter().zip(&tokens[given..]).filter(|(a, b)| a == b).count();
        let stats = session.take_store_stats();
        println!(
            "check {name}: {given} given tokens in {prompt_s:.2} s; logits of the last one: relative \
             RMS {:.3} from the reference's, KL {:.2e} nats, largest logit {} (reference {}), {} of \
             the 10 largest in common; fed with the reference's tokens, the pick is the \
             reference's next token in {same} of {n_new} rows, the first {equal} in a row (what \
             greedy generation would share); {:.1} ms per step (median)",
            (diff / size).sqrt(),
            kl,
            ours[0],
            theirs[0],
            ours.iter().filter(|t| theirs.contains(t)).count(),
            median(&mut ms.clone()),
        );
        println!("  store: {}", store_line(&stats, given.div_ceil(8) + ms.len()));
        for (at, margin) in margins.iter().take(6) {
            let piece = |t: u32| {
                String::from_utf8_lossy(&tokenizer.decode(&[t], true).unwrap_or_default())
                    .into_owned()
            };
            println!(
                "  token {at}: ours {:?}, reference {:?}; our logit of the reference's token is \
                 {margin:.3} below our pick's",
                piece(picks[at - given]),
                piece(tokens[*at]),
            );
        }
    }

    if args.chat {
        let questions = [
            "Erkläre in drei Sätzen, warum der Himmel blau ist.",
            "Explain in three sentences how a hash table works.",
        ];
        for question in questions {
            let chat = ChatOptions { enable_thinking: false, ..ChatOptions::default() };
            let prompt = tokenizer.render_chat(&[Message::new("user", question)], &[], &chat)?;
            let tokens =
                tokenizer.encode(&prompt, EncodeOptions { add_special: true, parse_special: true });
            session.restart(0)?;
            session.take_store_stats();
            let started = Instant::now();
            let pick = read(&mut session, &tokens, true)?.ok_or("no pick")?;
            let prompt_s = started.elapsed().as_secs_f64();
            let (generated, mut ms) = generate(&mut session, pick, args.generate, &ends)?;
            let end = generated.iter().position(|t| ends(*t)).unwrap_or(generated.len());
            let text = tokenizer.decode(&generated[..end], false)?;
            let stats = session.take_store_stats();
            println!(
                "chat: {question:?} ({} prompt tokens in {prompt_s:.2} s, {} generated, {:.1} ms \
                 per token median, {:.1} tokens/s over all)\n  -> {:?}",
                tokens.len(),
                generated.len(),
                median(&mut ms.clone()),
                ms.len() as f64 * 1e3 / ms.iter().sum::<f64>().max(1e-9),
                String::from_utf8_lossy(&text),
            );
            ms.clear();
            println!("  store: {}", store_line(&stats, tokens.len().div_ceil(8) + generated.len()));
        }
    }

    let mut text = match &args.text {
        Some(path) => tokens_of(path)?,
        None => Vec::new(),
    };
    for path in &args.text_file {
        let plain = std::fs::read_to_string(path)?;
        let options = EncodeOptions { add_special: false, parse_special: false };
        text.extend(tokenizer.encode(&plain, options));
    }
    // The pick after the last token that was read, for a decode run that
    // starts right there.
    let mut after_prefill = None;
    if args.prefill > 0 {
        if text.len() < args.prefill {
            return Err(format!("--text has {} tokens", text.len()).into());
        }
        for round in 0..args.repeat {
            session.restart(0)?;
            session.take_store_stats();
            if let Some(profile) = &profile {
                // `--breakdown` times the last reading.
                profile.reset();
                profile.pause(round + 1 < args.repeat);
            }
            let started = Instant::now();
            after_prefill = read(&mut session, &text[..args.prefill], true)?;
            let seconds = started.elapsed().as_secs_f64();
            let stats = session.take_store_stats();
            println!(
                "prefill: {} tokens in {seconds:.2} s = {:.1} tokens/s in steps of {} rows (load \
             average {})",
                args.prefill,
                args.prefill as f64 / seconds,
                session.options().part.max_rows,
                load_average(),
            );
            println!("  store: {}", store_line(&stats, args.prefill.div_ceil(args.rows as usize)));
            println!("  {}", churn_line(&stats, &session));
            if stats.transit_uploads > 0 {
                println!(
                    "  transit: {} experts ({} from RAM, {:.2} GiB up), {:.2} ms each from asked \
                     to usable; read from the file: {} asked, {} ahead of which {} used and {} \
                     dropped ({:.2} GiB); afterwards {} took a VRAM slot and {} a RAM slot",
                    stats.transit_uploads,
                    stats.transit_from_ram,
                    stats.transit_upload_bytes as f64 / (1u64 << 30) as f64,
                    stats.transit_ms / stats.transit_uploads as f64,
                    stats.transit_reads,
                    stats.transit_ahead_reads,
                    stats.transit_ahead_used,
                    stats.transit_ahead_unused,
                    stats.transit_read_bytes as f64 / (1u64 << 30) as f64,
                    stats.transit_adopted,
                    stats.transit_admitted,
                );
            }
            if round + 1 == args.repeat {
                breakdown(profile.as_deref(), &session);
            }
            let wide = session.take_wide_stats();
            if wide.layers > 0 {
                let layers = wide.layers as f64;
                println!(
                    "  wide steps: {:.1} experts picked per layer, {:.1} of them without a slot, {:.2} \
                 more rounds; per layer the host waited {:.2} ms for the picks, {:.2} ms for \
                 the first round to end, {:.2} ms for the misses to arrive, {:.2} ms for later \
                 rounds",
                    wide.picked as f64 / layers,
                    wide.missing as f64 / layers,
                    wide.extra_rounds as f64 / layers,
                    wide.record_wait.as_secs_f64() * 1e3 / layers,
                    wide.compute_wait.as_secs_f64() * 1e3 / layers,
                    wide.bring_wait.as_secs_f64() * 1e3 / layers,
                    wide.round_wait.as_secs_f64() * 1e3 / layers,
                );
            }
        }
    }
    let mut at = if args.prefill > 0 { args.prefill } else { 0 };
    for &context in &args.decode_at {
        if text.len() < context {
            return Err(format!("--text has {} tokens, --decode-at {context}", text.len()).into());
        }
        if context < at || at == 0 {
            session.restart(0)?;
            at = 0;
        }
        // The context, with the pick after its last token.
        let started = Instant::now();
        let pick = match after_prefill.take().filter(|_| context == at) {
            Some(pick) => pick,
            None => {
                let split = context.saturating_sub(1).max(at);
                read(&mut session, &text[at..split], false)?;
                read(&mut session, &text[split..context], true)?.ok_or("no pick")?
            }
        };
        let reading = started.elapsed();
        // Twice with `--breakdown`: untimed steps for the speed, then
        // steps with timed events for where the time goes.
        let mut tokens = vec![pick];
        for timed in [false, true] {
            if timed && profile.is_none() {
                break;
            }
            if let Some(profile) = &profile {
                profile.reset();
                profile.pause(!timed);
            }
            session.take_store_stats();
            let mut host = [Duration::ZERO; 4];
            let mut ms = Vec::new();
            let mut kinds: Vec<StepKind> = Vec::new();
            let first = tokens.len() - 1;
            for _ in 0..args.decode_steps {
                let before = session.store().stats().clone();
                let token = [tokens[tokens.len() - 1]];
                let picks = session.step(&[Rows {
                    slot: 0,
                    tokens: &token,
                    direct: true,
                    wanted: Wanted::Last,
                }])?;
                tokens.push(picks[0]);
                let time = session.last_time();
                ms.push(time.wall.as_secs_f64() * 1e3);
                let after = session.store().stats();
                kinds.push(StepKind {
                    ms: time.wall.as_secs_f64() * 1e3,
                    misses: (after.picks - after.hits) - (before.picks - before.hits),
                    from_file: after.miss_from_disk - before.miss_from_disk,
                    rows_ms: time.rows.as_secs_f64() * 1e3,
                });
                for (total, part) in
                    host.iter_mut().zip([time.prepare, time.rows, time.enqueue, time.wait])
                {
                    *total += part;
                }
            }
            let stats = session.take_store_stats();
            let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
            let fastest = ms.iter().copied().fold(f64::INFINITY, f64::min);
            let steps = ms.len().max(1) as f64;
            let host_ms = host.map(|d| d.as_secs_f64() * 1e3 / steps);
            let mut sorted = ms.clone();
            sorted.sort_by(f64::total_cmp);
            println!(
                "decode at {} tokens of context ({:.1} s to read the context): {:.2} ms per step \
                 (median of {}), mean {mean:.2} ms = {:.1} tokens/s, fastest {fastest:.2} ms, 90 % \
                 below {:.2} ms; host per step: prepare {:.3} ms, n-gram rows {:.3} ms, enqueue \
                 {:.3} ms, wait {:.3} ms (load average {}){}",
                context + first,
                reading.as_secs_f64(),
                median(&mut ms.clone()),
                ms.len(),
                1e3 / mean,
                sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)],
                host_ms[0],
                host_ms[1],
                host_ms[2],
                host_ms[3],
                load_average(),
                if timed { ", with timed events" } else { "" },
            );
            println!("  store: {}", store_line(&stats, ms.len()));
            println!("{}", step_classes(&kinds));
            if timed {
                breakdown(profile.as_deref(), &session);
            }
        }
        at = context + tokens.len() - 1;
        let text = tokenizer.decode(&tokens, true).unwrap_or_default();
        let text = String::from_utf8_lossy(&text);
        println!("  text: {:?}", text.chars().take(200).collect::<String>());
    }
    Ok(())
}

/// Prints the device time by kind since the profile was reset.
fn breakdown(profile: Option<&Profile>, session: &Session<'_>) {
    let Some(profile) = profile else { return };
    for (device, (steps, ms)) in profile.totals().iter().enumerate() {
        let steps = (*steps).max(1) as f64;
        let name = session.devices().get(device).map_or("?", |d| d.info().name.as_str());
        let parts: Vec<String> =
            PHASES.iter().map(|p| format!("{p:?} {:.2}", ms[*p as usize] / steps)).collect();
        println!(
            "  device time per step on {name}: {} (sum {:.2} ms)",
            parts.join(", "),
            ms.iter().sum::<f64>() / steps
        );
    }
}
