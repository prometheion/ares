//! Measures what the routers' pruning of experts only the file holds costs
//! and gives (`ares_cuda::Prune`): a tool for development, not a part of the
//! server.
//!
//! ```text
//! qwen4exp_prune measure --model FIRST_PART.gguf --texts german.txt,english.txt,code.txt
//!     --ref DIR [--write-ref] [--devices 0,1] [--ram-gib 8] [--usage DIR]
//!     [--prompt 256] [--tokens 2000] [--rule off|skip|router|similar] [--from-rank 8]
//!     [--below-weight 0.05] [--weights keep|renorm|scaled] [--min-closeness 0.2]
//!     [--near DIR] [--label NAME] [--out results.tsv] [--serve model|stream]
//!     [--mixer 0 --core-share 0.76 --shared worker | --spread layers]
//! qwen4exp_prune calibrate --model FIRST_PART.gguf --texts a.txt,b.txt --tokens 1024
//!     --out FILE [--devices 0,1] [--ram-gib 8]
//! qwen4exp_prune near --model FIRST_PART.gguf --calibration FILE --out DIR [--count 16]
//!     [--candidates 48] [--rows 16] [--threads 12] [--study 0,16,32,47] [--layers 0..48]
//! ```
//!
//! - `measure` opens the model as the server does (rows of a step as the
//!   server takes them, the tiers filled from the usage profile in `--usage`,
//!   by default the user's cache directory, as a start of the server fills
//!   them), and for every text reads its first `--prompt` tokens as a prompt
//!   and then the next `--tokens` one token a step, as generation does, with
//!   the text's tokens as the tokens generated. With `--write-ref` it writes
//!   the 256 most likely tokens of every step and their log-probabilities
//!   into `--ref`; otherwise it compares every step with those: the KL
//!   divergence of the step's distribution from the reference's (over those
//!   tokens and the rest as one), whether the most likely token is the same,
//!   reads from the file per token, and tokens per second. Each text gives a
//!   line in `--out`, tab-separated.
//! - `calibrate` reads texts in steps of eight rows and writes, for every row
//!   and layer, the input of the layer's experts as the kernels read it (int8
//!   codes and scales) and the router's picks with their weights.
//! - `near` computes from a calibration the table of the experts most like
//!   each expert ([`ExpertNeighbours`]) on the CPU and writes it into `--out`
//!   under the model's key. With `--study` it compares, on those layers, which
//!   measure finds the expert whose output is nearest: the outputs on the
//!   expert's own rows (what the table keeps), the outputs on rows shared by
//!   all experts, the router's rows, and the experts' weights.

use std::io::{BufReader, BufWriter, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_cuda::{Device, Prune, PruneRule, PruneWeights, RouteRecord, Wait};
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    Options, PruneSetup, Role, Rows, Session, SessionOptions, Spread, Wanted, profile_key,
    step_rows,
};
use ares_model::qwen4exp::{Config, Weights};
use ares_runtime::expert_store::{ExpertNeighbours, ExpertProfile, Serve, StoreStats};
use ares_tokenizer::{EncodeOptions, Tokenizer};

mod near;

type AnyError = Box<dyn std::error::Error>;

/// The most likely tokens of a step that the reference keeps.
const KEPT: usize = 256;

#[derive(Debug, Clone)]
struct Args {
    command: String,
    model: String,
    devices: Vec<u32>,
    ram_gib: f64,
    usage: Option<PathBuf>,
    texts: Vec<String>,
    prompt: usize,
    tokens: usize,
    reference: Option<PathBuf>,
    write_ref: bool,
    rule: String,
    from_rank: u32,
    below_weight: Option<f32>,
    weights: Option<String>,
    min_closeness: f32,
    near: Option<PathBuf>,
    label: String,
    out: Option<PathBuf>,
    serve: Serve,
    mixer: Option<usize>,
    core_share: Option<f64>,
    shared: Option<String>,
    layers_split: bool,
    calibration: Option<PathBuf>,
    count: usize,
    candidates: usize,
    rows: usize,
    threads: usize,
    study: Vec<usize>,
    layers: Option<(usize, usize)>,
}

fn parse() -> Result<Args, AnyError> {
    let mut it = std::env::args().skip(1);
    let command = it.next().ok_or("a command: measure, calibrate or near")?;
    let mut args = Args {
        command,
        model: String::new(),
        devices: vec![0, 1],
        ram_gib: 8.0,
        usage: None,
        texts: Vec::new(),
        prompt: 256,
        tokens: 2000,
        reference: None,
        write_ref: false,
        rule: "off".into(),
        from_rank: 8,
        below_weight: None,
        weights: None,
        min_closeness: 0.0,
        near: None,
        label: String::new(),
        out: None,
        serve: Serve::Measured,
        mixer: None,
        core_share: None,
        shared: None,
        layers_split: false,
        calibration: None,
        count: 16,
        candidates: 48,
        rows: 16,
        threads: 12,
        study: Vec::new(),
        layers: None,
    };
    let list = |s: String| -> Vec<String> { s.split(',').map(str::to_owned).collect() };
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => args.model = value()?,
            "--devices" => {
                args.devices = list(value()?).iter().map(|d| d.parse()).collect::<Result<_, _>>()?
            }
            "--ram-gib" => args.ram_gib = value()?.parse()?,
            "--usage" => args.usage = Some(value()?.into()),
            "--texts" => args.texts = list(value()?),
            "--prompt" => args.prompt = value()?.parse()?,
            "--tokens" => args.tokens = value()?.parse()?,
            "--ref" => args.reference = Some(value()?.into()),
            "--write-ref" => args.write_ref = true,
            "--rule" => args.rule = value()?,
            "--from-rank" => args.from_rank = value()?.parse()?,
            "--below-weight" => args.below_weight = Some(value()?.parse()?),
            "--weights" => args.weights = Some(value()?),
            "--min-closeness" => args.min_closeness = value()?.parse()?,
            "--near" => args.near = Some(value()?.into()),
            "--label" => args.label = value()?,
            "--out" => args.out = Some(value()?.into()),
            "--serve" => {
                args.serve = match value()?.as_str() {
                    "model" => Serve::Measured,
                    "stream" => Serve::Stream,
                    other => return Err(format!("--serve {other}: model or stream").into()),
                }
            }
            "--mixer" => args.mixer = Some(value()?.parse()?),
            "--core-share" => args.core_share = Some(value()?.parse()?),
            "--shared" => args.shared = Some(value()?),
            "--spread" => {
                args.layers_split = match value()?.as_str() {
                    "layers" => true,
                    "auto" => false,
                    other => return Err(format!("--spread {other}: auto or layers").into()),
                }
            }
            "--calibration" => args.calibration = Some(value()?.into()),
            "--count" => args.count = value()?.parse()?,
            "--candidates" => args.candidates = value()?.parse()?,
            "--rows" => args.rows = value()?.parse()?,
            "--threads" => args.threads = value()?.parse()?,
            "--study" => {
                args.study = list(value()?).iter().map(|d| d.parse()).collect::<Result<_, _>>()?
            }
            "--layers" => {
                let v = value()?;
                let (a, b) = v.split_once("..").ok_or("--layers FIRST..END")?;
                args.layers = Some((a.parse()?, b.parse()?));
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if args.model.is_empty() {
        return Err("--model FILE is needed".into());
    }
    Ok(args)
}

fn load_average() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| "?".into())
}

fn mem_available_gib() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        })
        .map_or(f64::NAN, |kib| kib / (1u64 << 20) as f64)
}

/// The cache directory the server keeps its profiles in.
fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("ares");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/ares")
}

/// The dial the arguments describe; `None` for `--rule off`.
fn prune_of(args: &Args) -> Result<Option<Prune>, AnyError> {
    let rule = match args.rule.as_str() {
        "off" => return Ok(None),
        "skip" => PruneRule::Skip,
        "router" => PruneRule::Router,
        "similar" => PruneRule::Similar,
        other => return Err(format!("--rule {other}: off, skip, router or similar").into()),
    };
    if args.from_rank == 0 {
        return Err("--from-rank counts from 1".into());
    }
    let mut prune = Prune::new(rule, args.from_rank - 1);
    if let Some(weights) = &args.weights {
        prune.weights = match weights.as_str() {
            "keep" => PruneWeights::Keep,
            "renorm" => PruneWeights::Renormalise,
            "scaled" => PruneWeights::Scaled,
            other => return Err(format!("--weights {other}: keep, renorm or scaled").into()),
        };
    }
    if let Some(w) = args.below_weight {
        prune.below_weight = w;
    }
    prune.min_closeness = args.min_closeness;
    Ok(Some(prune))
}

fn session_options(args: &Args, c: &Config, rows: u32) -> SessionOptions {
    let spread = if args.layers_split {
        Spread::Layers
    } else if args.mixer.is_some() || args.core_share.is_some() || args.shared.is_some() {
        Spread::Tensor {
            mixer: args.mixer,
            core_share: args.core_share,
            shared_on_mixer: args.shared.as_deref().map(|s| s == "mixer"),
        }
    } else {
        Spread::Auto
    };
    let _ = c;
    SessionOptions {
        part: Options { max_rows: rows, cells: 8192, logit_rows: 8, ..Options::default() },
        ram_tier: Some((args.ram_gib * (1u64 << 30) as f64) as u64),
        spread,
        serve: args.serve,
        ..SessionOptions::default()
    }
}

fn tokens_of_texts(tokenizer: &Tokenizer, texts: &[String]) -> Result<Vec<Vec<u32>>, AnyError> {
    texts
        .iter()
        .map(|path| {
            let plain = std::fs::read_to_string(path)?;
            let options = EncodeOptions { add_special: false, parse_special: false };
            Ok(tokenizer.encode(&plain, options))
        })
        .collect()
}

fn text_name(path: &str) -> String {
    Path::new(path).file_stem().map_or_else(|| path.to_owned(), |s| s.to_string_lossy().into())
}

/// The log-probabilities of a row of logits, in f64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let largest = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let total: f64 = logits.iter().map(|l| (*l as f64 - largest).exp()).sum();
    let log_total = largest + total.ln();
    logits.iter().map(|l| *l as f64 - log_total).collect()
}

/// The `KEPT` most likely tokens of a row: (id, log-probability), most
/// likely first.
fn kept(logp: &[f64]) -> Vec<(u32, f32)> {
    let mut order: Vec<u32> = (0..logp.len() as u32).collect();
    let k = KEPT.min(order.len());
    order.select_nth_unstable_by(k - 1, |a, b| logp[*b as usize].total_cmp(&logp[*a as usize]));
    order.truncate(k);
    order.sort_by(|a, b| logp[*b as usize].total_cmp(&logp[*a as usize]).then(a.cmp(b)));
    order.into_iter().map(|id| (id, logp[id as usize] as f32)).collect()
}

/// KL(reference || step) over the reference's kept tokens and the rest as
/// one, and whether the most likely token is the same.
fn compare(reference: &[(u32, f32)], logp: &[f64]) -> (f64, bool) {
    let mut kl = 0.0;
    let (mut p_kept, mut q_kept) = (0.0f64, 0.0f64);
    for (id, lp) in reference {
        let p = f64::from(*lp).exp();
        let lq = logp[*id as usize];
        kl += p * (f64::from(*lp) - lq);
        p_kept += p;
        q_kept += lq.exp();
    }
    let (p_rest, q_rest) = ((1.0 - p_kept).max(0.0), (1.0 - q_kept).max(1e-12));
    if p_rest > 1e-12 {
        kl += p_rest * (p_rest.ln() - q_rest.ln());
    }
    let top = logp
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .map_or(0, |(i, _)| i as u32);
    (kl.max(0.0), reference.first().is_some_and(|r| r.0 == top))
}

fn write_reference(path: &Path, rows: &[Vec<(u32, f32)>]) -> Result<(), AnyError> {
    let mut out = BufWriter::new(std::fs::File::create(path)?);
    out.write_all(b"ARESPREF")?;
    out.write_all(&(rows.len() as u32).to_le_bytes())?;
    out.write_all(&(KEPT as u32).to_le_bytes())?;
    for row in rows {
        for i in 0..KEPT {
            let (id, lp) = row.get(i).copied().unwrap_or((0, f32::NEG_INFINITY));
            out.write_all(&id.to_le_bytes())?;
            out.write_all(&lp.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_reference(path: &Path) -> Result<Vec<Vec<(u32, f32)>>, AnyError> {
    let mut bytes = Vec::new();
    BufReader::new(std::fs::File::open(path)?).read_to_end(&mut bytes)?;
    if bytes.len() < 16 || &bytes[..8] != b"ARESPREF" {
        return Err(format!("{} is no reference", path.display()).into());
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let (n, k) = (word(8), word(12));
    let mut rows = Vec::with_capacity(n);
    for r in 0..n {
        let at = 16 + r * k * 8;
        let row: Vec<(u32, f32)> = (0..k)
            .map(|i| {
                let e = at + i * 8;
                (word(e) as u32, f32::from_le_bytes(bytes[e + 4..e + 8].try_into().unwrap()))
            })
            .filter(|(_, lp)| lp.is_finite())
            .collect();
        rows.push(row);
    }
    Ok(rows)
}

/// Opens the model on the devices as the server does (with steps of `rows`
/// rows, or as many as the server takes), the tiers filled from the usage
/// profile.
fn open<'g>(
    args: &Args,
    gguf: &'g Gguf,
    c: &Config,
    w: &'g Weights<'g>,
    prune: Option<PruneSetup>,
    rows: Option<u32>,
) -> Result<Session<'g>, AnyError> {
    let devices: Vec<Device> =
        args.devices.iter().map(|d| Device::open(*d, Wait::Spin)).collect::<Result<_, _>>()?;
    let rows = match rows {
        Some(rows) => rows,
        None => step_rows(gguf, &devices, &Options::default())?,
    };
    let mut opts = session_options(args, c, rows);
    opts.part.prune = prune;
    let begun = Instant::now();
    let mut session = Session::open(gguf, c, w, devices, opts)?;
    let opened = begun.elapsed();
    let dir = args.usage.clone().unwrap_or_else(cache_dir);
    let key = profile_key(gguf, c);
    let profile = ExpertProfile::read(&dir, &key)
        .map_err(|e| format!("the usage profile in {}: {e}", dir.display()))?
        .ok_or_else(|| format!("no usage profile {} in {}", key.file_name(), dir.display()))?;
    let (filled, took) = session.fill_from_profile(&profile)?;
    let ram_slots = session.ram_slots();
    println!(
        "opened in {:.1} s, steps of {rows} rows; filled from the profile in {:.1} s: {} into \
         VRAM, {} into RAM ({:.1} GiB read); RAM slots {ram_slots}; load {}, MemAvailable {:.1} GiB",
        opened.as_secs_f64(),
        took.as_secs_f64(),
        filled.vram,
        filled.ram,
        filled.read_bytes as f64 / (1u64 << 30) as f64,
        load_average(),
        mem_available_gib(),
    );
    Ok(session)
}

/// Reads `tokens` as a prompt, `rows` to a step, wanting no logits.
fn prompt(session: &mut Session<'_>, tokens: &[u32], rows: usize) -> Result<(), AnyError> {
    for chunk in tokens.chunks(rows) {
        session.step(&[Rows { slot: 0, tokens: chunk, direct: true, wanted: Wanted::None }])?;
    }
    Ok(())
}

#[derive(Default)]
struct Totals {
    positions: usize,
    kl: f64,
    kl_max: f64,
    top1: usize,
    seconds: f64,
    disk: u64,
    demand_reads: u64,
    guess_reads: u64,
    wait_ms: f64,
    picks: u64,
    skipped: u64,
    stood_in: u64,
}

fn measure(args: &Args) -> Result<(), AnyError> {
    let gguf = Gguf::open(&args.model)?;
    let c = Config::from_gguf(&gguf)?;
    let w = Weights::load(&gguf, &c)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let prune = prune_of(args)?;
    let near = match (&prune, &args.near) {
        (Some(p), Some(dir)) if p.rule == PruneRule::Similar => {
            let key = profile_key(&gguf, &c);
            let table = ExpertNeighbours::read(dir, &key)
                .map_err(|e| format!("the table of similar experts in {}: {e}", dir.display()))?
                .ok_or_else(|| format!("no table of similar experts in {}", dir.display()))?;
            Some(Arc::new(table))
        }
        (Some(p), None) if p.rule == PruneRule::Similar => {
            return Err("--rule similar needs --near DIR".into());
        }
        _ => None,
    };
    let setup = prune.map(|prune| PruneSetup { prune, near });
    let reference = args.reference.clone().ok_or("--ref DIR is needed")?;
    std::fs::create_dir_all(&reference)?;
    let texts = tokens_of_texts(&tokenizer, &args.texts)?;
    let mut session = open(args, &gguf, &c, &w, setup, None)?;
    let mut all = Totals::default();
    let mut lines = Vec::new();
    for (path, tokens) in args.texts.iter().zip(&texts) {
        let name = text_name(path);
        let need = args.prompt + args.tokens + 1;
        if tokens.len() < need {
            return Err(format!("{path} has {} tokens, {need} are needed", tokens.len()).into());
        }
        session.restart(0)?;
        // The prompt the same way for every rule: unpruned, in steps of
        // eight rows, whose sums do not depend on when an expert arrives (a
        // wide step adds the rounds of experts that came in the order they came).
        session.set_pruning(false);
        prompt(&mut session, &tokens[..args.prompt], 8)?;
        session.set_pruning(true);
        session.take_store_stats();
        session.take_prune_stats();
        let ref_path = reference.join(format!("{name}.ref"));
        let wanted = if args.write_ref { None } else { Some(read_reference(&ref_path)?) };
        let mut written = Vec::new();
        let mut t = Totals::default();
        for (i, token) in tokens[args.prompt..args.prompt + args.tokens].iter().enumerate() {
            let begun = Instant::now();
            session.step(&[Rows {
                slot: 0,
                tokens: std::slice::from_ref(token),
                direct: true,
                wanted: Wanted::Last,
            }])?;
            t.seconds += begun.elapsed().as_secs_f64();
            let logits = session.logits(0).ok_or("no logits")?;
            let logp = log_softmax(logits);
            match &wanted {
                None => written.push(kept(&logp)),
                Some(rows) => {
                    let (kl, same) = compare(&rows[i], &logp);
                    t.kl += kl;
                    t.kl_max = t.kl_max.max(kl);
                    t.top1 += usize::from(same);
                }
            }
            t.positions += 1;
        }
        let stats: StoreStats = session.take_store_stats();
        let pruned = session.take_prune_stats();
        t.disk = stats.miss_from_disk;
        t.demand_reads = stats.demand_reads;
        t.guess_reads = stats.guess_reads;
        t.wait_ms = stats.wait_ms;
        t.picks = stats.picks;
        t.skipped = pruned.left_out();
        t.stood_in = pruned.stood_in;
        if args.write_ref {
            write_reference(&ref_path, &written)?;
        }
        lines.push(line(args, &name, &t));
        println!("{}", lines.last().unwrap());
        all.positions += t.positions;
        all.kl += t.kl;
        all.kl_max = all.kl_max.max(t.kl_max);
        all.top1 += t.top1;
        all.seconds += t.seconds;
        all.disk += t.disk;
        all.demand_reads += t.demand_reads;
        all.guess_reads += t.guess_reads;
        all.wait_ms += t.wait_ms;
        all.picks += t.picks;
        all.skipped += t.skipped;
        all.stood_in += t.stood_in;
    }
    lines.push(line(args, "all", &all));
    println!("{}", lines.last().unwrap());
    if let Some(out) = &args.out {
        let new = !out.exists();
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(out)?;
        if new {
            writeln!(
                file,
                "label\ttext\trule\tfrom_rank\tbelow_weight\tweights\tmin_closeness\tram_gib\t\
                 positions\tkl_mean\tkl_max\ttop1\tdisk_per_token\tdemand_reads_per_token\t\
                 guess_reads_per_token\twait_ms_per_token\ttokens_per_s\tskipped_per_token\t\
                 stood_in_per_token\tload"
            )?;
        }
        for line in &lines {
            writeln!(file, "{line}")?;
        }
    }
    Ok(())
}

fn line(args: &Args, text: &str, t: &Totals) -> String {
    let n = t.positions.max(1) as f64;
    let prune = prune_of(args).ok().flatten();
    format!(
        "{}\t{text}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.5}\t{:.3}\t{:.4}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{}",
        args.label,
        args.rule,
        if prune.is_some() { args.from_rank.to_string() } else { "-".into() },
        prune.map_or("-".into(), |p| {
            if p.below_weight.is_finite() { p.below_weight.to_string() } else { "-".into() }
        }),
        prune.map_or("-".into(), |p| format!("{:?}", p.weights).to_lowercase()),
        prune.map_or("-".into(), |p| p.min_closeness.to_string()),
        args.ram_gib,
        t.positions,
        if args.write_ref { f64::NAN } else { t.kl / n },
        if args.write_ref { f64::NAN } else { t.kl_max },
        if args.write_ref { f64::NAN } else { t.top1 as f64 / n },
        t.disk as f64 / n,
        t.demand_reads as f64 / n,
        t.guess_reads as f64 / n,
        t.wait_ms / n,
        n / t.seconds.max(1e-9),
        t.skipped as f64 / n,
        t.stood_in as f64 / n,
        load_average(),
    )
}

/// Reads texts in steps of eight rows and writes, for every row and layer,
/// the input of the layer's experts as the kernels read them and the
/// router's picks: `near` computes the table of similar experts from it.
fn calibrate(args: &Args) -> Result<(), AnyError> {
    let gguf = Gguf::open(&args.model)?;
    let c = Config::from_gguf(&gguf)?;
    let w = Weights::load(&gguf, &c)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let texts = tokens_of_texts(&tokenizer, &args.texts)?;
    let out = args.out.clone().ok_or("--out FILE is needed")?;
    let mut session = open(args, &gguf, &c, &w, None, Some(8))?;
    let mut writer = near::CalibrationWriter::create(&out, &c)?;
    let k = c.experts.used;
    for tokens in &texts {
        session.restart(0)?;
        let take = tokens.len().min(args.tokens);
        for chunk in tokens[..take].chunks(8) {
            session.step(&[Rows { slot: 0, tokens: chunk, direct: true, wanted: Wanted::None }])?;
            // Every layer's record and rows are on the host once the step
            // has ended; the part that routes a layer has its whole record.
            for layer in 0..c.n_layer {
                let part = session
                    .parts()
                    .iter()
                    .find(|p| p.layer_range().contains(&layer) && p.role() != Role::Worker)
                    .ok_or("no part routes the layer")?;
                let i = layer - part.layer_range().start;
                let board = part.board();
                let record = RouteRecord::parse(board.record(i), k as u32, chunk.len() as u32)
                    .ok_or("a route record that is none")?;
                let act = board.rows(i, chunk.len());
                writer.layer(layer, chunk.len(), act, &record)?;
            }
            writer.rows_done(chunk.len())?;
        }
        println!("calibrated {take} tokens (load {})", load_average());
    }
    writer.finish()?;
    Ok(())
}

fn main() {
    let args = match parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    };
    let begun = Instant::now();
    let result = match args.command.as_str() {
        "measure" => measure(&args),
        "calibrate" => calibrate(&args),
        "near" => near::near(&args),
        other => Err(format!("unknown command {other}: measure, calibrate or near").into()),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    let _ = Duration::ZERO;
    println!("done in {:.1} s", begun.elapsed().as_secs_f64());
}
