//! The prefix cache in the conductor, against the host executor as an oracle.
//!
//! The executor keeps the cells of every page and the recurrent state of every
//! slot and checkpoint place ([`ares_runtime::testing`]), carries out the step
//! record's copies and the transfers of the levels below the devices (a
//! page's cells and a place's state go to the host as bytes and come back),
//! and makes a row that reads anything wrong pick another token. So every
//! answer here that equals the answer of an engine without a cache says that
//! the pages, tails and checkpoints the conductor shared, restored, replayed,
//! evicted, kept in RAM, wrote to the drive and read back were the right ones;
//! and the counters and the per-request reports say how much it computed.
//!
//! - [`prefix`]: the index on the devices: pages, tails, checkpoints, replays.
//! - [`ram`]: the level in RAM, and the books that follow the machine.
//! - [`ssd`]: the level on SSD, across engines.
//! - [`ssd_ram`]: the two together, while the machine's memory comes and goes.

#[path = "../common/mod.rs"]
mod common;
mod decisions;
mod prefix;
mod ram;
mod ssd;
mod ssd_ram;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ares_runtime::testing::{HostExecutor, Problems};
use ares_runtime::{
    RequestReport, Runtime, RuntimeConfig, Snapshot, Split, SsdConfig, SsdCounters,
    SyntheticConfig, SyntheticModel,
};
use common::*;

/// Cells per page, and the bytes one checkpoint takes in the model that
/// pretends to recurrent state.
const PAGE: u32 = 16;
const CHECKPOINT: u64 = 4096;
/// The bytes of a page in RAM (one block of direct I/O), and the pages that
/// pass through RAM on their way to the drive and back, which a budget for
/// the prompt cache's RAM holds besides its checkpoints with a level on SSD.
const PAGE_BYTES: u64 = 4096;
const FLOOR: u64 = 16 * PAGE_BYTES;
/// The settled cells below which nothing of a text is written to the drive.
const MIN_CELLS: u32 = 48;
/// A machine with memory to spare.
const PLENTY: u64 = 64 << 30;

/// A fresh directory under the system temp dir, removed when dropped. The
/// suites run in one process, so each directory has a number of its own.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::SeqCst);
        let name = format!("ares-runtime-cache-{tag}-{}-{n}", std::process::id());
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A level on SSD in `dir` that writes texts of [`MIN_CELLS`] and more, in
/// small segments.
fn store(dir: &Path) -> SsdConfig {
    SsdConfig { min_cells: MIN_CELLS, segment_bytes: 64 << 10, ..SsdConfig::new(dir, 1 << 30) }
}

/// An engine on the host executor with the pool of `config`, and what its
/// oracle finds.
struct Rig {
    rt: Runtime,
    problems: Problems,
}

/// An engine whose executor `adjust` shapes further.
fn engine_with(
    model: &Arc<SyntheticModel>,
    config: RuntimeConfig,
    adjust: impl FnOnce(HostExecutor) -> HostExecutor,
) -> Rig {
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_pool(config.pages, config.cells_per_page);
    let host = adjust(host);
    let problems = host.problems();
    Rig { rt: Runtime::start(model.clone(), Box::new(host), config).unwrap(), problems }
}

fn engine(model: &Arc<SyntheticModel>, config: RuntimeConfig) -> Rig {
    engine_with(model, config, |host| host)
}

/// What a request said and what it took from the cache and computed.
struct Answer {
    run: Run,
    report: RequestReport,
}

impl Rig {
    fn ask(&self, prompt: &str, max_tokens: u32) -> Answer {
        let request = request(prompt, Some(max_tokens));
        let id = request.id;
        let run = run_one(&self.rt, request);
        let report = *snapshot(&self.rt).request(id).expect("the request ended");
        Answer { run, report }
    }

    fn snapshot(&self) -> Snapshot {
        snapshot(&self.rt)
    }

    /// The counters since `before`.
    fn since(&self, before: &Snapshot) -> HashMap<&'static str, u64> {
        since(&self.snapshot(), before)
    }

    /// The snapshot once the level on SSD has nothing left to write.
    fn written(&self) -> Snapshot {
        let deadline = Instant::now() + LIMIT;
        loop {
            let s = self.snapshot();
            if s.counters.ssd.pending == 0 {
                return s;
            }
            assert!(Instant::now() < deadline, "the writer is stuck: {s}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The oracle found nothing wrong, and the accounting counted the pages a
    /// step needs right: none had to be found after a claim was granted.
    fn consistent(&self) {
        let problems = self.problems.lock().unwrap();
        assert!(problems.is_empty(), "the oracle found: {problems:#?}");
        let s = self.snapshot();
        assert_eq!(s.counters.cache_pages_evicted_late, 0, "{s}");
    }
}

/// The run of an engine without any cache on the small model of 2048 tokens
/// as `shape` changes it, which is what the cache must not change.
fn fresh_run(shape: impl FnOnce(&mut SyntheticConfig), prompt: &str, max_tokens: u32) -> Run {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    shape(&mut config);
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let rt = Runtime::start(
        model.clone(),
        Box::new(HostExecutor::new(model.clone(), model.config().max_rows, 16)),
        RuntimeConfig {
            pages: 4096,
            cells_per_page: PAGE,
            max_slots: 16,
            window_rows: 1,
            warm_up: false,
            ..RuntimeConfig::default()
        },
    )
    .unwrap();
    let mut run = alone(&rt, prompt, Some(max_tokens));
    // A cold engine cached nothing.
    run.usage.cached_prompt_tokens = 0;
    run
}

/// What an engine without any cache says.
fn fresh(prompt: &str, max_tokens: u32) -> String {
    fresh_run(|_| {}, prompt, max_tokens).text
}

fn text_of(seed: u64, len: usize) -> String {
    Rng(seed).prompt(len)
}

/// What the store in `dir` holds, as an engine of `model` that opens it sees
/// it.
fn stored(model: &Arc<SyntheticModel>, dir: &Path) -> SsdCounters {
    let config = RuntimeConfig {
        pages: 256,
        cells_per_page: PAGE,
        max_slots: 8,
        window_rows: 1,
        warm_up: false,
        cache_ram: 8 * CHECKPOINT + FLOOR,
        checkpoint_prompt_interval: 2 * PAGE,
        checkpoint_generation_interval: 2 * PAGE,
        ssd: Some(store(dir)),
        ..RuntimeConfig::default()
    };
    engine(model, config).snapshot().counters.ssd
}
