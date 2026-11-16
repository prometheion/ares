//! The level of the prefix cache in RAM, in the conductor, against the host
//! executor as an oracle.
//!
//! Pages that the pool gives up are copied into RAM (the executor plays the
//! device: a page's cells leave as bytes and come back), and a prompt that
//! finds its start there has them copied back. A row that reads a cell or a
//! state that is not the one its history made picks another token, so an
//! answer that equals the answer of an engine without a cache says that what
//! came back from RAM was the right state. Below RAM lies the level on SSD,
//! which is written to only when something leaves RAM and when an engine
//! ends.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ares_runtime::testing::{HostExecutor, Problems};
use ares_runtime::{
    Machine, PinnedRam, RequestReport, Rule, Runtime, RuntimeConfig, Snapshot, Split, SsdConfig,
    SyntheticConfig, SyntheticModel,
};
use common::*;

/// Cells per page, the bytes of one checkpoint of the model that pretends to
/// recurrent state, and the bytes of a page in RAM (a block of direct I/O).
const PAGE: u32 = 16;
const CHECKPOINT: u64 = 4096;
const PAGE_BYTES: u64 = 4096;
/// The pages a prompt cache with a level on SSD keeps whatever the machine
/// has.
const FLOOR: u64 = 16 * PAGE_BYTES;
/// The settled cells below which nothing of a text is written.
const MIN_CELLS: u32 = 48;
const PLENTY: u64 = 64 << 30;

/// A fresh directory under the system temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ares-runtime-ram-{tag}-{}", std::process::id()));
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

/// A model with recurrent state whose device side moves state to the host.
fn model() -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.prefix_cache = Some(CHECKPOINT);
    config.durable = Some(1);
    Arc::new(SyntheticModel::new(config).unwrap())
}

fn store(dir: &Path) -> SsdConfig {
    SsdConfig { min_cells: MIN_CELLS, segment_bytes: 64 << 10, ..SsdConfig::new(dir, 1 << 30) }
}

/// A pool of `pages` pages, which two texts of 300 cells do not fit together;
/// checkpoints at every second page boundary; eight places and `ram_pages`
/// pages in RAM (and the floor of pages with a level on SSD).
fn config(pages: u32, ram_pages: u64, ssd: Option<SsdConfig>) -> RuntimeConfig {
    let floor = if ssd.is_some() { FLOOR } else { 0 };
    RuntimeConfig {
        pages,
        cells_per_page: PAGE,
        max_slots: 8,
        window_rows: 1,
        warm_up: false,
        cache_ram: 8 * CHECKPOINT + ram_pages * PAGE_BYTES + floor,
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

/// An engine whose prompt cache keeps its pages in books of its own.
fn engine(model: &Arc<SyntheticModel>, config: RuntimeConfig) -> Rig {
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_pool(config.pages, config.cells_per_page);
    let problems = host.problems();
    Rig { rt: Runtime::start(model.clone(), Box::new(host), config).unwrap(), problems }
}

/// An engine whose checkpoints and pages in RAM lie in books that follow
/// `machine`, below a limit of `limit` bytes.
fn following(
    model: &Arc<SyntheticModel>,
    config: RuntimeConfig,
    machine: &Arc<Machine>,
    limit: u64,
) -> (Rig, PinnedRam) {
    let ram = PinnedRam::following(limit, machine);
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_pool(config.pages, config.cells_per_page)
        .with_ram(ram.clone());
    let problems = host.problems();
    let rt = Runtime::start(model.clone(), Box::new(host), config).unwrap();
    (Rig { rt, problems }, ram)
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

    /// The snapshot once the level on SSD has nothing left to write.
    fn written(&self) -> Snapshot {
        let deadline = std::time::Instant::now() + LIMIT;
        loop {
            let s = snapshot(&self.rt);
            if s.counters.ssd.pending == 0 {
                return s;
            }
            assert!(std::time::Instant::now() < deadline, "the writer is stuck: {s}");
            std::thread::sleep(Duration::from_millis(1));
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
        RuntimeConfig { pages: 4096, cache_ram: 0, ..self::config(4096, 0, None) },
    )
    .unwrap();
    alone(&rt, prompt, Some(max_tokens)).text
}

fn text_of(seed: u64, len: usize) -> String {
    Rng(seed).prompt(len)
}

/// What the store in `dir` holds, as an engine that opens it sees it.
fn stored(model: &Arc<SyntheticModel>, dir: &Path) -> ares_runtime::SsdCounters {
    snapshot(&engine(model, config(256, 0, Some(store(dir)))).rt).counters.ssd
}

/// The machine the engines of a test follow: what it has available is set by
/// the test, and it is looked at when the test says so.
struct Memory {
    available: Arc<AtomicU64>,
    ms: Arc<AtomicU64>,
    machine: Arc<Machine>,
}

impl Memory {
    fn new(available: u64) -> Self {
        let (bytes, ms) = (Arc::new(AtomicU64::new(available)), Arc::new(AtomicU64::new(0)));
        let (source, clock) = (bytes.clone(), ms.clone());
        let machine = Machine::new(
            Rule::default(),
            move || Some(source.load(Ordering::SeqCst)),
            move || Duration::from_millis(clock.load(Ordering::SeqCst)),
        );
        Self { available: bytes, ms, machine }
    }

    fn has(&self, available: u64) {
        self.available.store(available, Ordering::SeqCst);
        self.ms.fetch_add(50, Ordering::SeqCst);
        self.machine.look().expect("the machine says what it has");
    }
}

#[test]
fn a_conversation_resumes_from_ram_after_the_devices_gave_it_up() {
    let model = model();
    // No level on SSD: RAM is the only place below the devices.
    let e = engine(&model, config(32, 64, None));
    let (one, two) = (text_of(1, 300), text_of(2, 300));
    let a = e.ask(&one, 10);
    let b = e.ask(&two, 10);
    assert_eq!((a.text.clone(), b.text), (fresh(&one, 10), fresh(&two, 10)));
    let s = snapshot(&e.rt);
    assert!(s.counters.cache_pages_evicted > 0, "the second text made room: {s}");
    assert!(s.counters.ram.pages_in > 0, "what the pool gave up went into RAM: {s}");
    assert!(!s.counters.ssd.enabled);

    // The first conversation's next turn: its pages and the tail with the
    // state at its end come back from RAM, and nothing is replayed.
    let second = format!("{one}{}{}", a.text, text_of(3, 40));
    let c = e.ask(&second, 10);
    assert_eq!(c.text, fresh(&second, 10));
    let s = snapshot(&e.rt);
    assert!(c.report.cached_tokens >= 300, "{:?}", c.report);
    assert_eq!(c.report.replayed_rows, 0, "{:?}", c.report);
    assert!(s.counters.ram.pages_read > 0, "{s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");
    e.consistent();

    // The same with the second conversation, which left the devices for the
    // first one's turn.
    let again = format!("{two}{}{}", fresh(&two, 10), text_of(4, 40));
    let d = e.ask(&again, 10);
    assert_eq!(d.text, fresh(&again, 10));
    assert!(d.report.cached_tokens >= 300, "{:?}", d.report);
    assert_eq!(d.report.replayed_rows, 0, "{:?}", d.report);
    e.consistent();
}

#[test]
fn nothing_is_written_while_ram_holds_a_conversation_and_all_of_it_when_the_engine_ends() {
    let dir = TempDir::new("held");
    let model = model();
    let e = engine(&model, config(32, 64, Some(store(&dir.0))));
    // Two conversations take turns; each turn pushes the other out of the
    // devices, into RAM, and back.
    let mut texts = [text_of(10, 300), text_of(11, 300)];
    for turn in 0..3 {
        for (i, text) in texts.iter_mut().enumerate() {
            let answer = e.ask(text, 10);
            assert_eq!(answer.text, fresh(text, 10), "turn {turn} of conversation {i}");
            if turn > 0 {
                assert!(answer.report.cached_tokens >= 300, "{:?}", answer.report);
                assert_eq!(answer.report.replayed_rows, 0, "{:?}", answer.report);
            }
            text.push_str(&answer.text);
            text.push_str(&text_of(20 + turn * 2 + i as u64, 20));
        }
    }
    let s = e.written();
    assert!(s.counters.ram.pages_read > 0, "{s}");
    let ssd = s.counters.ssd;
    assert_eq!((ssd.pages_written, ssd.states_written, ssd.bytes_written), (0, 0, 0), "{s}");
    e.consistent();

    // The engine ends (an unload, a clean stop): what is worth keeping is
    // written, the pages of both conversations and the checkpoints that are
    // not left behind.
    drop(e);
    let stored = stored(&model, &dir.0);
    let pages: u64 = texts.iter().map(|text| text.len() as u64 / u64::from(PAGE)).sum();
    assert!(stored.store_pages + 4 >= pages, "{stored:?} of {pages}");
    assert!(stored.store_states >= 2, "{stored:?}");

    // And after a restart both conversations resume from the drive.
    let e = engine(&model, config(256, 64, Some(store(&dir.0))));
    for text in &texts {
        let next = format!("{text}{}", text_of(40, 20));
        let answer = e.ask(&next, 10);
        assert_eq!(answer.text, fresh(&next, 10));
        // From the last checkpoint at a page boundary: the end of the last
        // reply (a tail and the state after it) is not written.
        let report = answer.report;
        assert!(report.cached_tokens as usize >= text.len() - 4 * PAGE as usize, "{report:?}");
    }
    let s = snapshot(&e.rt);
    assert_eq!(s.counters.ssd.imports, 2, "{s}");
    e.consistent();
}

#[test]
fn a_conversation_that_ram_gave_up_comes_back_from_ssd() {
    let dir = TempDir::new("gone");
    let model = model();
    // RAM holds its floor of pages and nothing more.
    let e = engine(&model, config(32, 0, Some(store(&dir.0))));
    let (one, two) = (text_of(30, 300), text_of(31, 300));
    let a = e.ask(&one, 10);
    let b = e.ask(&two, 10);
    assert_eq!(b.text, fresh(&two, 10));
    let s = e.written();
    assert!(s.counters.ssd.pages_written > 0, "what left RAM was written: {s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");

    let second = format!("{one}{}{}", a.text, text_of(32, 40));
    let c = e.ask(&second, 10);
    assert_eq!(c.text, fresh(&second, 10));
    let s = snapshot(&e.rt);
    assert!(c.report.cached_tokens >= 256, "{:?}", c.report);
    assert!(s.counters.ssd.pages_read > 0, "{s}");
    e.consistent();
}

#[test]
fn without_a_level_on_ssd_nothing_is_written_and_ram_still_serves() {
    let dir = TempDir::new("none");
    let model = model();
    let e = engine(&model, config(32, 64, None));
    let (one, two) = (text_of(50, 300), text_of(51, 300));
    let a = e.ask(&one, 10);
    e.ask(&two, 10);
    let second = format!("{one}{}{}", a.text, text_of(52, 40));
    let c = e.ask(&second, 10);
    assert_eq!(c.text, fresh(&second, 10));
    assert!(c.report.cached_tokens >= 300, "{:?}", c.report);
    let s = snapshot(&e.rt);
    assert!(!s.counters.ssd.enabled && s.counters.ssd.bytes_written == 0, "{s}");
    e.consistent();
    drop(e);
    // Nothing reached any directory.
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
}

#[test]
fn a_shrink_writes_what_only_ram_holds_and_gives_the_memory_back() {
    let dir = TempDir::new("shrink");
    let model = model();
    let memory = Memory::new(PLENTY);
    let budget = config(32, 256, Some(store(&dir.0)));
    let (e, ram) = following(&model, budget, &memory.machine, 1 << 30);
    // Four conversations pass through a pool that holds one and a half: RAM
    // holds what the pool gave up, and nothing is written.
    let texts: Vec<String> = (0..4).map(|i| text_of(60 + i, 300)).collect();
    let answers: Vec<String> = texts.iter().map(|text| e.ask(text, 10).text).collect();
    let s = e.written();
    let held = ram.cache_pages();
    assert!(held >= 40 * PAGE_BYTES, "{held} {s}");
    assert_eq!(s.counters.ssd.pages_written, 0, "{s}");

    // The machine runs short: the conductor is woken, writes what only RAM
    // holds and gives the memory of the pages back, down to the floor.
    memory.has(0);
    let deadline = std::time::Instant::now() + LIMIT;
    while ram.cache_pages() > FLOOR {
        assert!(std::time::Instant::now() < deadline, "RAM kept {}", ram.cache_pages());
        std::thread::sleep(Duration::from_millis(1));
    }
    let s = e.written();
    assert!(s.counters.ssd.pages_written * PAGE_BYTES + FLOOR >= held, "{s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    e.consistent();

    // What went is on the drive: each conversation's next turn finds it, on
    // a machine that has no room for more than the floor. The pages that
    // leave the devices for them pass through it to the drive.
    for (text, said) in texts.iter().zip(&answers) {
        let next = format!("{text}{said}{}", text_of(70, 20));
        let answer = e.ask(&next, 10);
        assert_eq!(answer.text, fresh(&next, 10));
        assert!(answer.report.cached_tokens >= 256, "{:?} {}", answer.report, snapshot(&e.rt));
    }
    let s = e.written();
    assert!(s.counters.ssd.pages_read > 0, "{s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");
    e.consistent();
}

#[test]
fn the_anchor_that_later_turns_share_inside_its_page_reaches_the_drive() {
    let dir = TempDir::new("anchor");
    let model = model();
    // Turns that add a few tokens each, so that their prompts end in the page
    // after the first prompt's anchor: they leave no checkpoint at a page
    // boundary of their own, only the state at the ends of their tails, which
    // the drive does not keep. The anchor is where the next engine resumes.
    let e = engine(&model, config(256, 64, Some(store(&dir.0))));
    let mut text = text_of(80, 200);
    for turn in 0..3 {
        let answer = e.ask(&text, 2);
        assert_eq!(answer.text, fresh(&text, 2), "turn {turn}");
        text.push_str(&answer.text);
        text.push_str(&text_of(81 + turn, 2));
    }
    e.consistent();
    drop(e);
    assert!(stored(&model, &dir.0).store_states >= 1);

    let e = engine(&model, config(256, 64, Some(store(&dir.0))));
    let answer = e.ask(&text, 10);
    assert_eq!(answer.text, fresh(&text, 10));
    assert!(answer.report.cached_tokens >= 192, "{:?}", answer.report);
    e.consistent();
}

#[test]
fn an_engine_that_ends_writes_all_its_pages_through_the_floor_of_ram() {
    let dir = TempDir::new("floor");
    let model = model();
    // RAM holds its floor of sixteen pages and nothing more; the devices hold
    // three texts of 300 cells, 57 full pages, when the engine ends. They
    // reach the drive through the floor, sixteen at a time.
    let e = engine(&model, config(256, 0, Some(store(&dir.0))));
    let texts: Vec<String> = (0..3).map(|i| text_of(90 + i, 300)).collect();
    let answers: Vec<String> = texts.iter().map(|text| e.ask(text, 10).text).collect();
    assert_eq!(e.written().counters.ssd.pages_written, 0);
    drop(e);
    let stored = stored(&model, &dir.0);
    assert_eq!(stored.store_pages, 3 * 19, "{stored:?}");
    let e = engine(&model, config(256, 0, Some(store(&dir.0))));
    for (text, said) in texts.iter().zip(&answers) {
        let next = format!("{text}{said}{}", text_of(95, 20));
        let answer = e.ask(&next, 10);
        assert_eq!(answer.text, fresh(&next, 10));
        assert!(answer.report.cached_tokens >= 288, "{:?}", answer.report);
    }
    e.consistent();
}

#[test]
fn pages_in_ram_never_leave_a_checkpoint_without_memory_the_limit_can_lock() {
    let dir = TempDir::new("place");
    // Checkpoints of 32 pages each: more than the room RAM keeps for pages
    // that come. The books' limit holds the floor, four checkpoints and 48
    // pages; the budget is larger, so pages take what the limit leaves.
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.prefix_cache = Some(32 * PAGE_BYTES);
    config.durable = Some(1);
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let memory = Memory::new(PLENTY);
    let budget = RuntimeConfig {
        cache_ram: 64 * 32 * PAGE_BYTES + 256 * PAGE_BYTES,
        ..self::config(32, 0, Some(store(&dir.0)))
    };
    let limit = 32 * PAGE_BYTES + FLOOR + 4 * 32 * PAGE_BYTES + 48 * PAGE_BYTES;
    let (e, ram) = following(&model, budget, &memory.machine, limit);
    // Texts pass through a pool that holds one and a half of them: RAM fills
    // with what only it holds, while every text takes checkpoints. A place
    // whose memory the limit could not lock would be a problem the oracle
    // writes down.
    for i in 0..6 {
        let text = text_of(110 + i, 300);
        let answer = e.ask(&text, 10);
        assert_eq!(answer.text, fresh(&text, 10));
        e.consistent();
        assert!(ram.used() <= limit, "{} of {limit}", ram.used());
    }
    let s = e.written();
    assert!(s.counters.checkpoints_saved > 0, "{s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");
}
