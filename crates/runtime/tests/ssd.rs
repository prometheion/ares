//! The level of the prefix cache that is on SSD, in the conductor, against
//! the host executor as an oracle.
//!
//! The executor keeps the cells of every page and the recurrent state of
//! every slot and checkpoint place ([`ares_runtime::testing`]), and it carries
//! out the transfers of the level on SSD: a page's cells and a place's state
//! go to the host as bytes and come back. A row that reads a cell or a state
//! that is not the one its history made picks another token, so an answer that
//! equals the answer of an engine without a cache says that what was written
//! to the store, read back into another engine and restored there was the
//! right state.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ares_runtime::testing::{HostExecutor, Problems};
use ares_runtime::{
    RequestReport, Runtime, RuntimeConfig, Snapshot, Split, SsdConfig, SyntheticConfig,
    SyntheticModel,
};
use common::*;

/// Cells per page, and the bytes one checkpoint takes in the model that
/// pretends to recurrent state.
const PAGE: u32 = 16;
const CHECKPOINT: u64 = 4096;
/// The settled cells below which nothing of a text is written here.
const MIN_CELLS: u32 = 48;

/// A fresh directory under the system temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ares-runtime-ssd-{tag}-{}", std::process::id()));
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

/// A model with recurrent state whose device side moves state to the host;
/// `weights` stands for its file in the store's fingerprint.
fn model(weights: u64, configure: impl FnOnce(&mut SyntheticConfig)) -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.prefix_cache = Some(CHECKPOINT);
    config.durable = Some(weights);
    configure(&mut config);
    Arc::new(SyntheticModel::new(config).unwrap())
}

fn store(dir: &Path) -> SsdConfig {
    SsdConfig { min_cells: MIN_CELLS, segment_bytes: 64 << 10, ..SsdConfig::new(dir, 1 << 30) }
}

/// Checkpoints at every second page boundary of a prompt and while
/// generating, `places` of them in RAM.
fn config(pages: u32, places: u64, ssd: Option<SsdConfig>) -> RuntimeConfig {
    RuntimeConfig {
        pages,
        cells_per_page: PAGE,
        max_slots: 8,
        window_rows: 1,
        warm_up: false,
        cache_ram: places * CHECKPOINT,
        checkpoint_prompt_interval: 2 * PAGE,
        checkpoint_generation_interval: 2 * PAGE,
        ssd,
        ..RuntimeConfig::default()
    }
}

struct Rig {
    rt: Runtime,
    problems: Problems,
}

fn engine(model: &Arc<SyntheticModel>, config: RuntimeConfig) -> Rig {
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_pool(config.pages, config.cells_per_page);
    let problems = host.problems();
    Rig { rt: Runtime::start(model.clone(), Box::new(host), config).unwrap(), problems }
}

struct Answer {
    text: String,
    report: RequestReport,
}

impl Rig {
    fn ask(&self, prompt: &str, max_tokens: u32) -> Answer {
        let request = request(prompt, Some(max_tokens));
        let id = request.id;
        let run = run_one(&self.rt, request);
        let report = *snapshot(&self.rt).request(id).expect("the request ended");
        Answer { text: run.text, report }
    }

    fn snapshot(&self) -> Snapshot {
        snapshot(&self.rt)
    }

    /// Waits until the level on SSD has nothing left to write.
    fn written(&self, pages: u64) -> Snapshot {
        let deadline = std::time::Instant::now() + LIMIT;
        loop {
            let s = self.snapshot();
            if s.counters.ssd.pages_written >= pages && s.counters.ssd.pending == 0 {
                return s;
            }
            assert!(std::time::Instant::now() < deadline, "{pages} pages were not written: {s}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// The oracle found nothing wrong.
    fn consistent(&self) {
        let problems = self.problems.lock().unwrap();
        assert!(problems.is_empty(), "the oracle found: {problems:#?}");
    }
}

/// What an engine without any cache says.
fn fresh(prompt: &str, max_tokens: u32) -> String {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let rt = Runtime::start(
        model.clone(),
        Box::new(HostExecutor::new(model.clone(), model.config().max_rows, 8)),
        RuntimeConfig { ssd: None, ..self::config(4096, 0, None) },
    )
    .unwrap();
    alone(&rt, prompt, Some(max_tokens)).text
}

fn text_of(seed: u64, len: usize) -> String {
    Rng(seed).prompt(len)
}

#[test]
fn a_text_resumes_from_ssd_in_another_engine_with_the_same_tokens() {
    let dir = TempDir::new("restart");
    let model = model(1, |_| {});
    let first = text_of(1, 200);

    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let a = e.ask(&first, 20);
    assert_eq!(a.text, fresh(&first, 20));
    assert_eq!(a.report.cached_tokens, 0);
    // 220 cells are 13 full pages; the checkpoints at the page boundaries of
    // the prompt that lie deep enough went along.
    let s = e.written(13);
    assert_eq!(s.counters.ssd.pages_written, 13, "{s}");
    assert!(s.counters.ssd.states_written >= 4, "{s}");
    assert!(s.ssd_problem.is_none());
    e.consistent();
    drop(e);

    // Another engine, whose devices hold nothing: the second turn of the
    // conversation finds the first on SSD.
    let second = format!("{first}{}{}", a.text, text_of(2, 40));
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    assert_eq!(e.snapshot().counters.ssd.store_pages, 13);
    let b = e.ask(&second, 20);
    assert_eq!(b.text, fresh(&second, 20));
    let s = e.snapshot();
    // The run ends at the deepest checkpoint, which is restored; nothing is
    // replayed, and the rest is computed.
    assert!(b.report.cached_tokens >= 192, "{:?}", b.report);
    assert_eq!(b.report.replayed_rows, 0, "{:?}", b.report);
    assert_eq!(b.report.computed_rows + b.report.cached_tokens, b.report.prompt_tokens);
    assert_eq!(u64::from(b.report.cached_tokens), s.counters.ssd.pages_read * u64::from(PAGE));
    assert_eq!((s.counters.ssd.states_read, s.counters.checkpoints_restored), (1, 1), "{s}");
    assert_eq!(s.counters.ssd.imports, 1);
    e.consistent();

    // The same prompt again is in the devices now: nothing is read.
    let c = e.ask(&second, 20);
    assert_eq!(c.text, b.text);
    assert_eq!(e.snapshot().counters.ssd.pages_read, s.counters.ssd.pages_read);
    e.consistent();
}

#[test]
fn what_the_devices_gave_up_comes_back_from_ssd() {
    let dir = TempDir::new("evicted");
    let model = model(1, |_| {});
    // A pool of 32 pages: a text of 300 cells and another of 300 do not fit
    // together.
    let e = engine(&model, config(32, 4, Some(store(&dir.0))));
    let (one, two) = (text_of(3, 300), text_of(4, 300));
    let a = e.ask(&one, 10);
    e.written(19);
    let b = e.ask(&two, 10);
    assert_eq!((a.report.cached_tokens, b.report.cached_tokens), (0, 0));
    let s = e.written(2 * 19);
    assert!(s.counters.cache_pages_evicted > 0, "the second text made room: {s}");

    // The first text again: its pages are gone from the pool and come back.
    let again = e.ask(&one, 10);
    assert_eq!(again.text, a.text);
    assert_eq!(again.text, fresh(&one, 10));
    let s = e.snapshot();
    assert!(again.report.cached_tokens >= 256, "{:?}", again.report);
    assert!(s.counters.ssd.pages_read > 0, "{s}");
    assert!(s.counters.ssd.pages_read < 18, "what the pool still held was not read: {s}");
    // Nothing is written twice.
    assert_eq!(s.counters.ssd.pages_written, 2 * 19, "{s}");
    e.consistent();
}

#[test]
fn a_short_text_is_not_written() {
    let dir = TempDir::new("short");
    let model = model(1, |_| {});
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    // 30 cells of prompt and 10 of reply stay below the 48 of the rule.
    let a = e.ask(&text_of(5, 30), 10);
    assert_eq!(a.text, fresh(&text_of(5, 30), 10));
    drop(e);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let s = e.snapshot();
    assert_eq!((s.counters.ssd.store_pages, s.counters.ssd.store_states), (0, 0), "{s}");
    // A text that grows past the rule is written from its first page.
    let long = text_of(5, 30) + &a.text + &text_of(6, 60);
    e.ask(&long, 10);
    let s = e.written(6);
    assert_eq!(s.counters.ssd.pages_written, 6, "110 cells are 6 full pages: {s}");
    e.consistent();
}

#[test]
fn state_of_another_fingerprint_is_never_used_and_makes_room_first() {
    let dir = TempDir::new("fingerprint");
    let prompt = text_of(7, 400);
    let at = |max_bytes| SsdConfig { max_bytes, ..store(&dir.0) };
    let e = engine(&model(1, |_| {}), config(256, 8, Some(at(1 << 30))));
    let a = e.ask(&prompt, 10);
    let s = e.written(25);
    let old_bytes = s.counters.ssd.store_bytes;
    drop(e);
    let stores = || std::fs::read_dir(&dir.0).unwrap().count();
    assert_eq!(stores(), 1);

    // Other weights, the same text: nothing is taken from the store.
    let other = model(2, |_| {});
    let e = engine(&other, config(256, 8, Some(at(1 << 30))));
    let b = e.ask(&prompt, 10);
    assert_eq!(b.text, a.text, "the synthetic model's text does not depend on the number");
    assert_eq!(b.report.cached_tokens, 0);
    let s = e.written(25);
    assert_eq!(s.counters.ssd.pages_read, 0, "{s}");
    e.consistent();
    drop(e);
    assert_eq!(stores(), 2);

    // A limit that holds one store and not two: opening the second removes
    // the first, which nobody has open.
    let e = engine(&other, config(256, 8, Some(at(old_bytes + old_bytes / 2))));
    assert_eq!(stores(), 1);
    let c = e.ask(&prompt, 10);
    assert_eq!(c.text, a.text);
    assert!(c.report.cached_tokens > 0);
    e.consistent();
}

#[test]
fn the_size_limit_gives_up_the_texts_used_least_recently() {
    let dir = TempDir::new("limit");
    let model = model(1, |_| {});
    // A segment of 160 KiB holds the pages of one text of 300 cells (a page's
    // record is 8 KiB here); the limit holds about three texts with their
    // checkpoints.
    let at = SsdConfig { max_bytes: 700 << 10, segment_bytes: 160 << 10, ..store(&dir.0) };
    let texts: Vec<String> = (0..8).map(|i| text_of(20 + i, 300)).collect();
    let e = engine(&model, config(64, 4, Some(at.clone())));
    // Each text is written before the next one takes its pages of the pool:
    // the engine does not wait for the drive, and a page the pool gives up
    // before it was written is not written.
    let mut answers = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        answers.push(e.ask(text, 8).text);
        e.written((i as u64 + 1) * 19);
    }
    let s = e.written(8 * 19);
    assert!(s.counters.ssd.store_bytes <= at.max_bytes, "{s}");
    assert!(s.counters.ssd.store_pages < 8 * 19, "{s}");
    drop(e);

    let e = engine(&model, config(64, 4, Some(at)));
    // The newest text is there, the oldest is not: it is computed, and both
    // say what they said.
    let newest = e.ask(&texts[7], 8);
    assert!(newest.report.cached_tokens >= 256, "{:?}", newest.report);
    assert_eq!(newest.text, answers[7]);
    let oldest = e.ask(&texts[0], 8);
    assert_eq!(oldest.report.cached_tokens, 0, "{:?}", oldest.report);
    assert_eq!(oldest.text, answers[0]);
    e.consistent();
}

#[test]
fn damaged_and_missing_files_cost_only_time() {
    use std::os::unix::fs::FileExt;
    let dir = TempDir::new("damage");
    let model = model(1, |_| {});
    let (one, two) = (text_of(40, 500), text_of(41, 500));
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let a = e.ask(&one, 10);
    e.written(31);
    let b = e.ask(&two, 10);
    e.written(2 * 31);
    drop(e);

    let fingerprint = std::fs::read_dir(&dir.0).unwrap().next().unwrap().unwrap().path();
    let mut pages: Vec<PathBuf> = std::fs::read_dir(&fingerprint)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "pages"))
        .collect();
    pages.sort();
    assert!(pages.len() >= 4, "{pages:?}");
    // A flipped bit in the fourth page of the first segment, and the last
    // segment gone.
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&pages[0]).unwrap();
    let at = 3 * 8192 + 4096 + 100;
    let mut byte = [0u8; 1];
    file.read_exact_at(&mut byte, at).unwrap();
    file.write_all_at(&[byte[0] ^ 0x10], at).unwrap();
    drop(file);
    std::fs::remove_file(pages.last().unwrap()).unwrap();

    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let again = e.ask(&one, 10);
    assert_eq!(again.text, a.text);
    // The run was cut at the damaged page: what came before it is used.
    assert!(again.report.cached_tokens < 4 * PAGE, "{:?}", again.report);
    let s = e.snapshot();
    assert_eq!(s.counters.ssd.imports_cut, 1, "{s}");
    assert_eq!(e.ask(&two, 10).text, b.text);
    e.consistent();

    // The directory vanishes while the engine runs: it keeps serving, and
    // what it computes from then on is written to a new one.
    std::fs::remove_dir_all(&dir.0).unwrap();
    let three = text_of(42, 600);
    let c = e.ask(&three, 10);
    assert_eq!(c.text, fresh(&three, 10));
    let four = text_of(43, 600);
    let d = e.ask(&four, 10);
    assert_eq!(d.text, fresh(&four, 10));
    e.consistent();
    drop(e);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let s = e.snapshot();
    assert!(s.ssd_problem.is_none() && s.counters.ssd.enabled, "{s}");
    let d2 = e.ask(&four, 10);
    assert_eq!(d2.text, d.text);
    e.consistent();
}

#[test]
fn one_place_for_checkpoints_is_enough() {
    let dir = TempDir::new("one-place");
    let model = model(1, |_| {});
    let first = text_of(50, 330);
    let e = engine(&model, config(256, 1, Some(store(&dir.0))));
    let a = e.ask(&first, 30);
    let s = e.written(22);
    // Every checkpoint went through the one place and to the drive.
    assert!(s.counters.ssd.states_written >= 8, "{s}");
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    drop(e);

    let second = format!("{first}{}{}", a.text, text_of(51, 20));
    let e = engine(&model, config(256, 1, Some(store(&dir.0))));
    let b = e.ask(&second, 20);
    assert_eq!(b.text, fresh(&second, 20));
    assert!(b.report.cached_tokens >= 352, "{:?}", b.report);
    assert_eq!(b.report.replayed_rows, 0);
    e.consistent();
}

#[test]
fn pages_that_settle_late_are_written_late_and_at_the_end() {
    let dir = TempDir::new("settle");
    // The model says its last 40 cells have not reached the form they keep.
    let model = model(1, |c| c.settle_cells = 40);
    let first = text_of(60, 200);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let a = e.ask(&first, 20);
    let s = e.written(13);
    assert_eq!(s.counters.ssd.pages_written, 13, "the end writes the rest: {s}");
    drop(e);
    let second = format!("{first}{}{}", a.text, text_of(61, 40));
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let b = e.ask(&second, 20);
    assert_eq!(b.text, fresh(&second, 20));
    assert!(b.report.cached_tokens >= 192, "{:?}", b.report);
    e.consistent();
}

#[test]
fn a_second_engine_on_the_same_store_serves_without_it() {
    let dir = TempDir::new("two");
    let model = model(1, |_| {});
    let prompt = text_of(70, 200);
    let first = engine(&model, config(256, 8, Some(store(&dir.0))));
    let second = engine(&model, config(256, 8, Some(store(&dir.0))));
    let s = second.snapshot();
    assert!(!s.counters.ssd.enabled);
    assert!(s.ssd_problem.as_deref().is_some_and(|why| why.contains("in use")), "{s}");
    assert_eq!(second.ask(&prompt, 10).text, first.ask(&prompt, 10).text);
    // A model whose devices move no state says so, too.
    let plain = self::model(1, |c| c.durable = None);
    let third = engine(&plain, config(256, 8, Some(store(&dir.0))));
    assert!(third.snapshot().ssd_problem.is_some());
    assert_eq!(third.ask(&prompt, 10).text, fresh(&prompt, 10));
}

#[test]
fn requests_that_share_a_run_from_ssd_wait_for_it_together() {
    let dir = TempDir::new("together");
    let model = model(1, |_| {});
    let base = text_of(80, 600);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let a = e.ask(&base, 10);
    e.written(38);
    drop(e);

    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let prompts: Vec<String> =
        (0..4).map(|i| format!("{base}{}{}", a.text, text_of(90 + i, 20))).collect();
    let handles: Vec<Handle> =
        prompts.iter().map(|p| start(&e.rt, request(p, Some(10))).unwrap()).collect();
    for (handle, prompt) in handles.into_iter().zip(&prompts) {
        let run = handle.finish();
        assert_eq!(run.text, fresh(prompt, 10));
        assert!(run.usage.cached_prompt_tokens >= 592, "{:?}", run.usage);
    }
    let s = e.snapshot();
    assert_eq!(s.counters.ssd.imports, 1, "one run serves all four: {s}");
    assert_eq!(s.counters.ssd.pages_read, 37, "{s}");
    e.consistent();
}
