//! Flash-Next in this process on a machine that is made smaller than it is:
//! the engine follows the memory this machine reports less a number of bytes
//! the test hides, so that the engine gives memory back while the machine
//! itself still has plenty. A helper program then takes memory, and the test
//! reports how fast the tier of experts followed, what the kernel counted for
//! this process, whether generation went on, and when the tier grew back.
//!
//! Skipped unless `ARES_TEST_CUDA=1` and the model are set:
//!
//! ```text
//! ARES_TEST_CUDA=1 ARES_TEST_FLASH_NEXT=<file.gguf> \
//!   ARES_TEST_HOG="<helper> <total MiB> <step MiB> <interval ms> <hold s> <guard MiB>" \
//!   [ARES_TEST_DEVICES=0,1] [ARES_TEST_HIDE=<bytes hidden, default 4 GiB>] \
//!   [ARES_TEST_RAM=<the engine's limit in bytes>] \
//!   cargo test --release -p ares-server --test real_elastic -- --nocapture
//! ```
//!
//! The helper is any program that takes the five numbers, allocates and
//! touches memory in steps, holds it and frees it. Without it the test only
//! loads the model and checks that the tier took what the hidden machine has.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Sampling};
use ares_runtime::{Machine, Meminfo, MemorySource, Rule};
use ares_server::load;
use ares_server::plan::LoadOptions;

mod common;

const GIB: f64 = (1u64 << 30) as f64;

/// The least this machine may have available while the test runs: below it
/// the test stops what it started.
const SAFE: u64 = 3 << 30;

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok()
}

fn resident() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let line = status.lines().find_map(|line| line.strip_prefix("VmRSS:")).unwrap_or("0 kB");
    line.trim().trim_end_matches("kB").trim().parse::<u64>().unwrap_or(0) * 1024
}

/// This machine with `0` bytes less available: what the engine follows.
struct Smaller(u64);

impl MemorySource for Smaller {
    fn available(&self) -> Option<u64> {
        Meminfo.available().map(|bytes| bytes.saturating_sub(self.0))
    }

    fn own(&self) -> Option<u64> {
        Meminfo.own()
    }
}

fn name_of(loaded: &load::Loaded) -> String {
    loaded.name().to_owned()
}

fn load_average() -> String {
    let text = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    text.split_whitespace().next().unwrap_or("?").to_owned()
}

/// Asks for `max_tokens` tokens after `text` and counts them as they come.
fn generate(
    engine: &Arc<dyn Engine>,
    model: &str,
    id: u64,
    text: &str,
    max_tokens: u32,
    pieces: Arc<AtomicU64>,
) -> mpsc::Receiver<Result<(), String>> {
    let (done, wait) = mpsc::channel();
    let sink = EventSink::new(move |event| {
        match event {
            Event::Text(_) => {
                pieces.fetch_add(1, Ordering::Relaxed);
            }
            Event::Done { .. } => {
                let _ = done.send(Ok(()));
            }
            Event::Error(error) => {
                let _ = done.send(Err(error.to_string()));
            }
            Event::Logprobs(_) => {}
        }
        true
    });
    let request = GenerateRequest {
        model: model.to_owned(),
        prompt: Prompt::Text(text.to_owned()),
        sampling: Sampling::default(),
        max_tokens: Some(max_tokens),
        stop: Vec::new(),
        draft: true,
        structure: None,
        priority: 0,
        id: RequestId(id),
    };
    engine.submit(request, sink).expect("the engine takes the request");
    wait
}

#[test]
fn the_tier_of_flash_next_follows_the_memory_of_a_smaller_machine() {
    let Some(model) = common::model_file("ARES_TEST_FLASH_NEXT", "the first part of Flash-Next")
    else {
        return;
    };
    let hidden = env_u64("ARES_TEST_HIDE").unwrap_or(4 << 30);
    let real = || Meminfo.available().unwrap_or(0);
    assert!(real() >= SAFE + (1 << 30), "this machine has too little available to test on");

    // The machine the engine sees has `hidden` bytes less than this one.
    let started = Instant::now();
    let machine = Machine::new(Rule::of_this_machine(), Smaller(hidden), move || started.elapsed());
    machine.watch();
    assert!(load::follow(machine.clone()), "no watch was in use");

    let mut options = LoadOptions::new(model);
    if let Ok(devices) = std::env::var("ARES_TEST_DEVICES") {
        options.devices = Some(devices.split(',').map(|d| d.parse().expect("a device")).collect());
    }
    options.ram = env_u64("ARES_TEST_RAM");
    // The usage profile of the experts stays with the tests (for this model
    // the directory of the prompt cache holds nothing else).
    options.cache = Some(PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("elastic-profile"));
    let before_load = (real(), resident());
    let loaded = load::load(&options).expect("the model loads");
    let engine = loaded.engine();
    let name = loaded.name().to_owned();
    let tier =
        |engine: &Arc<dyn Engine>| engine.stats().map_or(0, |stats| stats.experts.ram_tier_bytes);
    let (speed, latency, wait) = machine.seen();
    eprintln!(
        "RES loaded: this machine had {:.2} GiB available and has {:.2} now, {:.2} of them hidden \
         from the engine; headroom {:.0} MiB (growth of others seen {:.2} GiB/s, release {:.0} \
         ms, wait before taking {:.0} s); the tier holds {:.2} GiB, the process {:.2} GiB \
         resident (was {:.2}); load average {}",
        before_load.0 as f64 / GIB,
        real() as f64 / GIB,
        hidden as f64 / GIB,
        machine.headroom() as f64 / (1 << 20) as f64,
        speed as f64 / GIB,
        latency.as_secs_f64() * 1e3,
        wait.as_secs_f64(),
        tier(&engine) as f64 / GIB,
        resident() as f64 / GIB,
        before_load.1 as f64 / GIB,
        load_average(),
    );
    assert!(real() >= SAFE, "the engine left this machine {} bytes", real());
    // What the load itself took for a while is settled: the engine leaves
    // what it sees as available at its headroom or above, or is still
    // giving back.
    let settled = Instant::now();
    while machine.look().is_some_and(|look| look.owed > 0)
        && settled.elapsed() < Duration::from_secs(60)
    {
        std::thread::sleep(Duration::from_millis(50));
    }
    let seen = real().saturating_sub(hidden);
    eprintln!(
        "RES settled after {:.1} s: the engine sees {:.2} GiB available, its headroom is {:.0} \
         MiB, the tier holds {:.2} GiB",
        settled.elapsed().as_secs_f64(),
        seen as f64 / GIB,
        machine.headroom() as f64 / (1 << 20) as f64,
        tier(&engine) as f64 / GIB,
    );
    assert!(seen + (256 << 20) >= machine.headroom(), "{seen} of {} left", machine.headroom());

    // What the tier that the machine allows is worth: three texts, each
    // asked twice, with the reads from the file a step needed.
    let texts = [
        (
            "German",
            "Erkläre in einfachen Worten, wie ein Wärmetauscher in einer Heizungsanlage \
                    funktioniert und warum der Rücklauf kälter ist als der Vorlauf.\n\n",
        ),
        (
            "English",
            "Explain in plain words how a hash table stores and finds its entries, and \
                     what happens when two keys collide.\n\n",
        ),
        (
            "code",
            "Write a Python function that merges two sorted lists into one sorted list \
                  without using sort(), with a short docstring.\n\n```python\n",
        ),
    ];
    let mut id = 10;
    for (name, text) in texts {
        for turn in ["first", "again"] {
            let before = engine.stats().expect("the engine counts");
            let begun = Instant::now();
            id += 1;
            let done = generate(&engine, &name_of(&loaded), id, text, 96, Arc::default());
            done.recv_timeout(Duration::from_secs(600)).expect("an answer").expect("no error");
            let took = begun.elapsed().as_secs_f64();
            let after = engine.stats().expect("the engine counts");
            let steps = (after.steps - before.steps).max(1);
            let tokens = after.generated_tokens - before.generated_tokens;
            let (a, b) = (after.experts, before.experts);
            eprintln!(
                "RES {name} {turn}: {} prompt + {tokens} tokens in {took:.2} s = {:.1} tok/s \
                 overall; steps {steps}, picks {}, in VRAM {}, from RAM {}, from the file {} ({:.2} \
                 per step), read {} MiB, waited {:.2} s; tier {:.2} GiB; this machine has {:.1} \
                 GiB available, load average {}",
                after.prompt_tokens - before.prompt_tokens,
                tokens as f64 / took,
                a.picks - b.picks,
                a.hits - b.hits,
                a.from_ram - b.from_ram,
                a.from_disk - b.from_disk,
                (a.from_disk - b.from_disk) as f64 / steps as f64,
                (a.read_bytes - b.read_bytes) >> 20,
                (a.wait_us - b.wait_us) as f64 / 1e6,
                a.ram_tier_bytes as f64 / GIB,
                real() as f64 / GIB,
                load_average(),
            );
            assert!(real() >= SAFE, "the engine left this machine {} bytes", real());
        }
    }

    let Ok(hog) = std::env::var("ARES_TEST_HOG") else {
        eprintln!("no ARES_TEST_HOG: the helper that takes memory was not run");
        drop(engine);
        loaded.unload();
        return;
    };

    // A long generation runs while the helper takes memory.
    let pieces = Arc::new(AtomicU64::new(0));
    let text = "Write a long, detailed essay about the history of the steam engine, from the \
                first atmospheric engines to the locomotives of the nineteenth century.\n\n";
    let done = generate(&engine, &name, 1, text, 1500, pieces.clone());
    std::thread::sleep(Duration::from_secs(5));
    let tier_before = tier(&engine);
    let rss_before = resident();
    let rate_before = {
        let (t, n) = (Instant::now(), pieces.load(Ordering::Relaxed));
        std::thread::sleep(Duration::from_secs(3));
        (pieces.load(Ordering::Relaxed) - n) as f64 / t.elapsed().as_secs_f64()
    };

    let mut words = hog.split_whitespace();
    let mut child = Command::new(words.next().expect("a helper"))
        .args(words)
        .stdout(Stdio::inherit())
        .spawn()
        .expect("the helper starts");
    let begun = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let mut least_real = real();
    let mut least_tier = tier_before;
    let mut first_release = None;
    let mut released_gib_at = Vec::new();
    let mut helper_gone = None;
    let mut regrown = None;
    let mut pieces_at_start = pieces.load(Ordering::Relaxed);
    let mut rate_during = 0.0;
    let mut least_rss = rss_before;
    let deadline = Duration::from_secs(env_u64("ARES_TEST_WATCH_S").unwrap_or(120));
    while begun.elapsed() < deadline {
        std::thread::sleep(Duration::from_millis(10));
        let (now, available, held) = (begun.elapsed(), real(), tier(&engine));
        least_real = least_real.min(available);
        least_rss = least_rss.min(resident());
        if available < SAFE && helper_gone.is_none() {
            eprintln!("RES STOP: {available} bytes available, under the safe line: helper killed");
            let _ = child.kill();
            stop.store(true, Ordering::SeqCst);
        }
        if held < tier_before && first_release.is_none() {
            first_release = Some(now);
        }
        let gone = tier_before.saturating_sub(held) / (1 << 30);
        while (released_gib_at.len() as u64) < gone {
            released_gib_at.push(now);
        }
        least_tier = least_tier.min(held);
        if helper_gone.is_none() && child.try_wait().ok().flatten().is_some() {
            helper_gone = Some(now);
            rate_during =
                (pieces.load(Ordering::Relaxed) - pieces_at_start) as f64 / now.as_secs_f64();
            pieces_at_start = pieces.load(Ordering::Relaxed);
        }
        if helper_gone.is_some() && regrown.is_none() && held + (256 << 20) >= tier_before {
            regrown = Some(now);
            break;
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    let after = helper_gone.map_or(0.0, |gone| {
        (pieces.load(Ordering::Relaxed) - pieces_at_start) as f64
            / (begun.elapsed() - gone).as_secs_f64().max(1e-3)
    });
    let (speed, latency, wait) = machine.seen();
    eprintln!(
        "RES give-back: the tier held {:.2} GiB and fell to {:.2}; first piece gone {} after the \
         helper started, whole GiB gone after {:?} s; this machine's available memory never \
         under {:.2} GiB ({:.2} as the engine sees it); resident {:.2} -> least {:.2} GiB; \
         helper gone after {:.1} s; tier back within 256 MiB of what it held {}; generation \
         {rate_before:.1} pieces/s before, {rate_during:.1} while the helper ran, {after:.1} \
         after; headroom now {:.0} MiB (growth seen {:.2} GiB/s, release {:.0} ms, wait {:.0} s); \
         load average {}",
        tier_before as f64 / GIB,
        least_tier as f64 / GIB,
        first_release.map_or("never".to_owned(), |t| format!("{:.0} ms", t.as_secs_f64() * 1e3)),
        released_gib_at
            .iter()
            .map(|t| (t.as_secs_f64() * 100.0).round() / 100.0)
            .collect::<Vec<_>>(),
        least_real as f64 / GIB,
        least_real.saturating_sub(hidden) as f64 / GIB,
        rss_before as f64 / GIB,
        least_rss as f64 / GIB,
        helper_gone.map_or(f64::NAN, |t| t.as_secs_f64()),
        match (regrown, helper_gone) {
            (Some(back), Some(gone)) => format!("{:.1} s after it", (back - gone).as_secs_f64()),
            _ => "not within the time watched".to_owned(),
        },
        machine.headroom() as f64 / (1 << 20) as f64,
        speed as f64 / GIB,
        latency.as_secs_f64() * 1e3,
        wait.as_secs_f64(),
        load_average(),
    );
    assert!(
        !stop.load(Ordering::SeqCst),
        "the machine's available memory fell under the safe line"
    );
    assert!(least_tier < tier_before, "the tier gave nothing back");
    assert!(
        rss_before.saturating_sub(least_rss) + (128 << 20) >= tier_before - least_tier,
        "what the tier gave did not leave the process: {rss_before} -> {least_rss}"
    );
    assert!(regrown.is_some(), "the tier did not grow back");
    // The generation ended as a generation does, not with an error.
    match done.recv_timeout(Duration::from_secs(300)) {
        Ok(result) => result.expect("the generation went on to its end"),
        Err(_) => panic!("the generation did not end"),
    }
    drop(engine);
    loaded.unload();
}
