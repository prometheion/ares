//! Measures the step loop on the synthetic model: tokens per second and where
//! each step goes, for one device and for both placements over two devices.
//!
//! ```sh
//! cargo run --release -p ares-runtime --example runtime_bench -- --help
//! ```
//!
//! Each configuration starts N requests at once (a short prompt and a fixed
//! number of tokens each) and waits for all of them. That is repeated, and the
//! run with the median speed is reported with its per-step breakdown. The
//! GPUs are shared with whatever else runs on the machine: check
//! `nvidia-smi` first.
//!
//! The two-device placements use peer access when the driver allows it; point
//! `ARES_LIBCUDA` at the patched library to get it.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Sampling};
use ares_core::GgmlType;
use ares_cuda::{Device, Wait, devices, peer_access};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{
    DevicePhase, GpuConfig, Phase, Runtime, RuntimeConfig, Snapshot, Split, SyntheticConfig,
    SyntheticModel,
};

const HELP: &str = "\
runtime_bench: tokens per second and per-step breakdown of the step loop

OPTIONS
  --placement LIST      single,tensor,layer (default: all the devices allow)
  --requests LIST       concurrent requests (default 1,2,4,8,16,32)
  --tokens N            tokens per request (default 128)
  --prompt N            prompt characters (default 24)
  --repeat N            runs per configuration (default 5)
  --layers N            layers (default 64)
  --width N             residual width (default 5120)
  --ffn N               feed-forward inner width (default 17408)
  --type NAME           weight type: q4_K, q5_K, q6_K, q8_0, iq4_xs ... (default q4_K)
  --distinct N          distinct layers of weights (default 4)
  --speculation on|off  drafts from the model's own draft block and the lookup in earlier
                        text (default off: the text does not repeat)
  --repeating           make the text loop (window 1, epoch 40) so drafts are accepted
  --device-spin US      device threads spin this long for work before parking (default 2000)
  --conductor-spin US   conductor spins this long for the devices before parking (default 100000)
  --device-flag MODE    spin|yield|block: how the driver waits (default spin)
  --no-guard            do not look for other GPU users (for profiling under nsys)
  --noise-ok            keep runs during which other processes used the CPUs (to measure
                        the engine under a deliberate competing load)
  --fake                no GPU: the conductor against a host executor with zero step time
  --detail              print the full per-step table of the last run of each configuration
  --max-rows N          rows per step (default 64)
";

#[derive(Clone)]
struct Options {
    placements: Vec<String>,
    requests: Vec<usize>,
    tokens: u32,
    prompt: usize,
    repeat: usize,
    layers: u32,
    width: u32,
    ffn: u32,
    ty: GgmlType,
    distinct: u32,
    speculation: bool,
    repeating: bool,
    device_spin: u64,
    conductor_spin: u64,
    device_flag: Wait,
    fake: bool,
    guard: bool,
    noise_ok: bool,
    detail: bool,
    max_rows: u32,
}

fn parse() -> Result<Options, String> {
    let mut o = Options {
        placements: Vec::new(),
        requests: vec![1, 2, 4, 8, 16, 32],
        tokens: 128,
        prompt: 24,
        repeat: 5,
        layers: 64,
        width: 5120,
        ffn: 17408,
        ty: GgmlType::Q4_K,
        distinct: 4,
        speculation: false,
        repeating: false,
        device_spin: 2000,
        conductor_spin: 100_000,
        device_flag: Wait::Spin,
        fake: false,
        guard: true,
        noise_ok: false,
        detail: false,
        max_rows: 64,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        let number = |text: String, name: &str| {
            text.parse::<u64>().map_err(|_| format!("{name}: '{text}' is not a number"))
        };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--placement" => o.placements = value(&flag)?.split(',').map(str::to_owned).collect(),
            "--requests" => {
                o.requests = value(&flag)?
                    .split(',')
                    .map(|s| s.parse().map_err(|_| format!("bad request count '{s}'")))
                    .collect::<Result<_, _>>()?
            }
            "--tokens" => o.tokens = number(value(&flag)?, &flag)? as u32,
            "--prompt" => o.prompt = number(value(&flag)?, &flag)? as usize,
            "--repeat" => o.repeat = number(value(&flag)?, &flag)? as usize,
            "--layers" => o.layers = number(value(&flag)?, &flag)? as u32,
            "--width" => o.width = number(value(&flag)?, &flag)? as u32,
            "--ffn" => o.ffn = number(value(&flag)?, &flag)? as u32,
            "--distinct" => o.distinct = number(value(&flag)?, &flag)? as u32,
            "--max-rows" => o.max_rows = number(value(&flag)?, &flag)? as u32,
            "--type" => {
                let name = value(&flag)?;
                o.ty = [
                    GgmlType::Q4_K,
                    GgmlType::Q5_K,
                    GgmlType::Q6_K,
                    GgmlType::Q3_K,
                    GgmlType::Q8_0,
                    GgmlType::IQ4_XS,
                    GgmlType::IQ4_NL,
                    GgmlType::IQ3_S,
                    GgmlType::IQ3_XXS,
                    GgmlType::IQ2_S,
                    GgmlType::IQ2_XS,
                ]
                .into_iter()
                .find(|t| t.name().eq_ignore_ascii_case(&name))
                .ok_or(format!("no decode kernel for type '{name}'"))?
            }
            "--speculation" => o.speculation = value(&flag)? == "on",
            "--repeating" => o.repeating = true,
            "--device-spin" => o.device_spin = number(value(&flag)?, &flag)?,
            "--conductor-spin" => o.conductor_spin = number(value(&flag)?, &flag)?,
            "--device-flag" => {
                o.device_flag = match value(&flag)?.as_str() {
                    "yield" => Wait::Yield,
                    "block" => Wait::Block,
                    _ => Wait::Spin,
                }
            }
            "--fake" => o.fake = true,
            "--no-guard" => o.guard = false,
            "--noise-ok" => o.noise_ok = true,
            "--detail" => o.detail = true,
            other => return Err(format!("unknown option '{other}' (see --help)")),
        }
    }
    Ok(o)
}

/// One run: N requests started together, waited for.
struct Run {
    wall: Duration,
    tokens: u64,
    snapshot: Snapshot,
    /// The CPU time other processes used during the run, as a share of all
    /// CPUs.
    others_cpu: f64,
}

/// Busy and total jiffies of all CPUs, and this process's own.
fn cpu_jiffies() -> (u64, u64, u64) {
    let stat = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let fields: Vec<u64> = stat
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    // user nice system idle iowait irq softirq steal
    let idle = fields.get(3).copied().unwrap_or(0) + fields.get(4).copied().unwrap_or(0);
    let total: u64 = fields.iter().take(8).sum();
    let own = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            // The command name is in parentheses and may hold spaces.
            let rest = s.rsplit_once(')')?.1.to_owned();
            let f: Vec<&str> = rest.split_whitespace().collect();
            Some(f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?)
        })
        .unwrap_or(0);
    (total.saturating_sub(idle), total, own)
}

/// What the kernel's scheduler did to one thread so far.
#[derive(Clone, Copy, Debug, Default)]
struct ThreadUse {
    /// Nanoseconds on a CPU.
    running_ns: u64,
    /// Nanoseconds runnable but waiting for a CPU.
    queued_ns: u64,
    /// Times the thread went to sleep (voluntary) and was taken off a CPU (not).
    voluntary: u64,
    involuntary: u64,
    /// CPU migrations are not exposed per thread; `slices` counts times it was
    /// put on a CPU.
    slices: u64,
}

/// The engine's threads (by name) and what the scheduler did to them,
/// from `/proc/self/task`.
fn thread_use() -> BTreeMap<String, ThreadUse> {
    let mut all = BTreeMap::new();
    for entry in std::fs::read_dir("/proc/self/task").into_iter().flatten().flatten() {
        let dir = entry.path();
        let Ok(name) = std::fs::read_to_string(dir.join("comm")) else { continue };
        let name = name.trim().to_owned();
        if !name.starts_with("engine-") {
            continue;
        }
        let mut u = ThreadUse::default();
        if let Ok(text) = std::fs::read_to_string(dir.join("schedstat")) {
            let n: Vec<u64> = text.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            if n.len() == 3 {
                (u.running_ns, u.queued_ns, u.slices) = (n[0], n[1], n[2]);
            }
        }
        if let Ok(text) = std::fs::read_to_string(dir.join("status")) {
            for line in text.lines() {
                let value =
                    |key: &str| line.strip_prefix(key).and_then(|v| v.trim().parse::<u64>().ok());
                if let Some(v) = value("voluntary_ctxt_switches:") {
                    u.voluntary = v;
                }
                if let Some(v) = value("nonvoluntary_ctxt_switches:") {
                    u.involuntary = v;
                }
            }
        }
        all.insert(name, u);
    }
    all
}

/// The processes other than this one that use a GPU right now. The GPUs are
/// shared with other work on this machine, and a run beside it measures that
/// work as much as this one.
fn foreign_gpu_users() -> Vec<u32> {
    let me = std::process::id();
    let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader"])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .filter(|pid| *pid != me)
        .collect()
}

/// Waits until no other process uses a GPU, three looks in a row.
fn wait_for_idle_gpus() {
    let (mut quiet, mut waited) = (0, 0);
    while quiet < 3 {
        if foreign_gpu_users().is_empty() {
            quiet += 1;
        } else {
            quiet = 0;
            waited += 1;
            if waited % 30 == 1 {
                eprintln!("waiting for another process to leave the GPUs");
            }
        }
        std::thread::sleep(Duration::from_millis(700));
    }
}

/// Like [`run_once`], retried until no other process touched the GPUs during
/// it (looked at twice a second).
fn run_clean(rt: &Runtime, o: &Options, n: usize, round: usize) -> Run {
    use std::sync::atomic::{AtomicBool, Ordering};
    if o.fake || !o.guard {
        return run_once(rt, o, n, round);
    }
    loop {
        wait_for_idle_gpus();
        let watching = Arc::new(AtomicBool::new(true));
        let dirty = Arc::new(AtomicBool::new(false));
        let watcher = {
            let (watching, dirty) = (watching.clone(), dirty.clone());
            std::thread::spawn(move || {
                while watching.load(Ordering::Relaxed) {
                    if !foreign_gpu_users().is_empty() {
                        dirty.store(true, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(400));
                }
            })
        };
        let run = run_once(rt, o, n, round);
        watching.store(false, Ordering::Relaxed);
        watcher.join().expect("watcher");
        if run.others_cpu > 0.08 && !o.noise_ok {
            eprintln!(
                "run discarded: other processes used {:.0} % of the CPUs during it",
                100.0 * run.others_cpu
            );
            continue;
        }
        if !dirty.load(Ordering::Relaxed) && foreign_gpu_users().is_empty() {
            return run;
        }
        eprintln!("run discarded: another process used the GPUs during it");
    }
}

fn run_once(rt: &Runtime, o: &Options, n: usize, round: usize) -> Run {
    rt.take_snapshot().expect("conductor");
    let (busy0, total0, own0) = cpu_jiffies();
    let (tx, rx) = mpsc::channel();
    let started = Instant::now();
    for i in 0..n {
        let tx = tx.clone();
        let prompt: String = (0..o.prompt)
            .map(|k| char::from(b'a' + ((i * 7 + k * 3 + round) % 26) as u8))
            .collect();
        let request = GenerateRequest {
            model: "synthetic".into(),
            prompt: Prompt::Text(prompt),
            sampling: Sampling::default(),
            max_tokens: Some(o.tokens),
            stop: Vec::new(),
            draft: true,
            structure: None,
            id: RequestId((round * 1000 + i) as u64 + 1),
        };
        rt.submit(
            request,
            EventSink::new(move |e| {
                let terminal = e.is_terminal();
                let tokens = if let Event::Done { usage, .. } = &e {
                    u64::from(usage.completion_tokens)
                } else {
                    0
                };
                if terminal {
                    let _ = tx.send(tokens);
                }
                true
            }),
        )
        .expect("accepted");
    }
    drop(tx);
    let mut tokens = 0;
    for _ in 0..n {
        tokens += rx.recv_timeout(Duration::from_secs(300)).expect("a request ended");
    }
    let wall = started.elapsed();
    let (busy1, total1, own1) = cpu_jiffies();
    let others =
        (busy1 - busy0).saturating_sub(own1 - own0) as f64 / (total1 - total0).max(1) as f64;
    Run { wall, tokens, snapshot: rt.snapshot().expect("conductor"), others_cpu: others }
}

fn median<T: Copy>(mut values: Vec<(f64, T)>) -> (f64, f64, f64, T) {
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mid = values[values.len() / 2];
    (values[0].0, mid.0, values[values.len() - 1].0, mid.1)
}

fn main() {
    let o = match parse() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let count = if o.fake { 2 } else { devices().expect("driver").len() };
    println!(
        "{} CUDA device(s); ARES_LIBCUDA={}",
        count,
        std::env::var("ARES_LIBCUDA").unwrap_or_else(|_| "(system library)".into())
    );
    if !o.fake && count >= 2 {
        let (peer, rank) = peer_access(0, 1).expect("peer query");
        println!(
            "peer access 0 -> 1: {peer} (rank {rank}): two-device sums go {}",
            if peer { "directly" } else { "through host memory" }
        );
    }
    let mut placements = o.placements.clone();
    if placements.is_empty() {
        placements = vec!["single".into()];
        if count >= 2 {
            placements.extend(["tensor".into(), "layer".into()]);
        }
    }
    let open: Vec<Device> = if o.fake {
        Vec::new()
    } else {
        (0..count as u32).map(|i| Device::open(i, o.device_flag).expect("open device")).collect()
    };

    let mut summary: BTreeMap<(String, usize), String> = BTreeMap::new();
    println!(
        "\nlayers {}, width {}, ffn {}, {} weights ({} distinct), {} tokens x N requests, prompt {}, speculation {}, repeat {}",
        o.layers,
        o.width,
        o.ffn,
        o.ty,
        o.distinct,
        o.tokens,
        o.prompt,
        if o.speculation { "on" } else { "off" },
        o.repeat
    );
    println!(
        "device spin {} us, conductor spin {} us, driver wait {:?}\n",
        o.device_spin, o.conductor_spin, o.device_flag
    );
    println!(
        "{:<7} {:>3} | {:>8} {:>8} {:>8} | {:>7} {:>7} {:>8} | {:>5} {:>5} {:>5} {:>5} {:>8} {:>5} {:>5} | {:>5} {:>6} {:>8} {:>6} | {:>6} {:>6} | {:>5} {:>5} {:>5}",
        "place",
        "N",
        "tok/s",
        "min",
        "max",
        "per req",
        "steps/s",
        "step ms",
        "plan",
        "write",
        "disp",
        "turn",
        "wait",
        "comm",
        "evnt",
        "wake",
        "enq",
        "dev",
        "gap",
        "gap %",
        "host %",
        "t p50",
        "t p99",
        "oth %"
    );
    println!(
        "{:<12} | {:^26} | {:^26} | {:^50} | {:^27} | idle share (gap: dev events, worst device; host: all devices)",
        "", "tokens per second", "", "conductor, mean us per step", "device 0, mean us per step"
    );

    for name in &placements {
        let (split, devices_used) = match name.as_str() {
            "single" => (Split::Single, 1),
            "tensor" => (Split::Tensor, 2),
            "layer" => (Split::Layer, 2),
            other => panic!("unknown placement '{other}'"),
        };
        if !o.fake && devices_used > count {
            println!("{name}: needs {devices_used} devices, skipped");
            continue;
        }
        let mut sc = SyntheticConfig::realistic(split, devices_used);
        sc.layers = o.layers;
        sc.width = o.width;
        sc.attn_inner = o.width;
        sc.ffn_inner = o.ffn;
        sc.weight_type = o.ty;
        sc.distinct_layers = o.distinct;
        sc.max_rows = o.max_rows;
        if o.repeating {
            sc = sc.repetitive(1, 40);
        }
        if o.speculation {
            sc = sc.with_own_drafts(7, &[0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]);
        }
        if split == Split::Tensor && !o.fake {
            // Slice the rows by memory bandwidth, as a split planner would.
            sc.tensor_shares = (0..devices_used)
                .map(|d| {
                    let i = open[d].info();
                    f64::from(i.memory_bus_width_bits) * f64::from(i.memory_clock_khz)
                })
                .collect();
        }
        if split == Split::Layer && !o.fake {
            // Split the layers by memory bandwidth, as a split planner would.
            let bw: Vec<f64> = (0..devices_used)
                .map(|d| {
                    let i = open[d].info();
                    f64::from(i.memory_bus_width_bits) * f64::from(i.memory_clock_khz)
                })
                .collect();
            let total: f64 = bw.iter().sum();
            let mut left = o.layers;
            let mut per: Vec<u32> =
                bw.iter().map(|b| ((b / total) * f64::from(o.layers)).round() as u32).collect();
            let last = per.len() - 1;
            per[last] = 0;
            let used: u32 = per.iter().sum();
            per[last] = left.saturating_sub(used).max(1);
            left = per.iter().sum();
            assert_eq!(left, o.layers, "layers per device {per:?}");
            sc.layers_per_device = per;
        }
        let model = Arc::new(SyntheticModel::new(sc.clone()).expect("a valid model"));
        let config = RuntimeConfig {
            window_rows: if o.speculation { 8 } else { 1 },
            conductor_spin: Duration::from_micros(o.conductor_spin),
            pages: 8192,
            ..RuntimeConfig::default()
        };
        let started = Instant::now();
        let rt = if o.fake {
            let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
                .with_devices(devices_used);
            Runtime::start(model.clone(), Box::new(host), config).expect("starts")
        } else {
            let mut gpu = GpuConfig::new(&config, &*model);
            gpu.spin = Duration::from_micros(o.device_spin);
            Runtime::start_gpu(model.clone(), open[..devices_used].to_vec(), config, gpu)
                .expect("starts")
        };
        let weights: Vec<String> = (0..devices_used)
            .map(|d| format!("{:.2} GiB", sc.weight_bytes(d) as f64 / (1u64 << 30) as f64))
            .collect();
        println!(
            "-- {name}: started in {:.1} s, weights per device {}",
            started.elapsed().as_secs_f64(),
            weights.join(", ")
        );

        for &n in &o.requests {
            let threads_before = thread_use();
            let mut runs = Vec::new();
            for round in 0..o.repeat + 1 {
                let run = run_clean(&rt, &o, n, round);
                // The first run of each configuration is a warm-up.
                if round > 0 {
                    runs.push(run);
                }
            }
            // Each metric is the median over the runs of that run's mean, so one
            // run that met a disturbance does not move it.
            let med = |f: &dyn Fn(&Run) -> f64| median(runs.iter().map(|r| (f(r), ())).collect()).1;
            let tps = |r: &Run| r.tokens as f64 / r.wall.as_secs_f64();
            let (min, mid, max, index) =
                median(runs.iter().enumerate().map(|(i, r)| (tps(r), i)).collect());
            let run = &runs[index];
            let s = &run.snapshot;
            let ph = |p: Phase| med(&|r: &Run| r.snapshot.phase(p).mean_us);
            let dev = |p: DevicePhase| med(&|r: &Run| r.snapshot.device_phase(0, p).mean_us);
            let line = format!(
                "{:<7} {:>3} | {:>8.1} {:>8.1} {:>8.1} | {:>7.1} {:>7.1} {:>8.2} | {:>5.0} {:>5.0} {:>5.0} {:>5.0} {:>8.0} {:>5.0} {:>5.0} | {:>5.0} {:>6.0} {:>8.0} {:>6.0} | {:>6.2} {:>6.2} | {:>5.1} {:>5.1} {:>5.1}",
                name,
                n,
                mid,
                min,
                max,
                mid / n as f64,
                med(&|r: &Run| r.snapshot.counters.steps as f64 / r.wall.as_secs_f64()),
                ph(Phase::Step) / 1e3,
                ph(Phase::Plan),
                ph(Phase::Write),
                ph(Phase::Dispatch),
                ph(Phase::Turnaround),
                ph(Phase::Wait),
                ph(Phase::Commit),
                ph(Phase::Events),
                dev(DevicePhase::Wake),
                dev(DevicePhase::Enqueue),
                dev(DevicePhase::Device),
                dev(DevicePhase::Gap),
                100.0
                    * med(&|r: &Run| (0..devices_used)
                        .map(|d| r.snapshot.gpu_wait_share(d))
                        .fold(0.0, f64::max)),
                100.0 * med(&|r: &Run| r.snapshot.host_wait_share()),
                med(&|r: &Run| r.snapshot.phase(Phase::Turnaround).p50_us),
                med(&|r: &Run| r.snapshot.phase(Phase::Turnaround).p99_us),
                100.0 * med(&|r: &Run| r.others_cpu)
            );
            println!("{line}");
            summary.insert((name.clone(), n), line);
            if o.detail {
                println!("{s}");
                let after = thread_use();
                let wall: f64 = runs.iter().map(|r| r.wall.as_secs_f64()).sum();
                println!(
                    "{:<18} {:>9} {:>11} {:>9} {:>9} {:>10}  (over {:.1} s of measured runs and the warm-up)",
                    "thread", "cpu %", "queued ms", "sleeps", "preempted", "slices", wall
                );
                for (name, a) in &after {
                    let b = threads_before.get(name).copied().unwrap_or_default();
                    println!(
                        "{:<18} {:>9.1} {:>11.2} {:>9} {:>9} {:>10}",
                        name,
                        100.0 * (a.running_ns - b.running_ns) as f64 / 1e9 / wall,
                        (a.queued_ns - b.queued_ns) as f64 / 1e6,
                        a.voluntary - b.voluntary,
                        a.involuntary - b.involuntary,
                        a.slices - b.slices
                    );
                }
                println!();
            }
        }
        rt.shutdown();
    }
}
