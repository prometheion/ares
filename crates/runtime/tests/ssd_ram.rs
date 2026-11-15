//! Where the host memory that follows the machine meets the level of the
//! prefix cache on SSD, in the conductor, against the host executor as an
//! oracle.
//!
//! The executor is given the books of page-locked RAM
//! ([`HostExecutor::with_ram`]) and plays the strictest device they allow: a
//! checkpoint place loses its memory, and what it held, as soon as the
//! conductor has called it unused. A save into such a place and a checkpoint
//! that leaves such a place for the drive are problems the oracle writes
//! down, and a state that is not the one its history made changes the text.
//! The books follow a machine whose available memory and clock are the
//! test's, so "the machine has no room for checkpoints" is a line of a test.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use ares_runtime::testing::{HostExecutor, Problems};
use ares_runtime::{
    Machine, PinnedRam, RecordView, RequestReport, Rule, Runtime, RuntimeConfig, Snapshot, Split,
    SsdConfig, StateCopyKind, SyntheticConfig, SyntheticModel,
};
use common::*;

/// Cells per page, and the bytes one checkpoint takes in the model that
/// pretends to recurrent state.
const PAGE: u32 = 16;
const CHECKPOINT: u64 = 4096;
/// The settled cells below which nothing of a text is written here.
const MIN_CELLS: u32 = 48;
/// A machine with memory to spare, and the limit of the engine's books.
const PLENTY: u64 = 64 << 30;
const RAM_LIMIT: u64 = 1 << 20;

/// A fresh directory under the system temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ares-runtime-ssd-ram-{tag}-{}", std::process::id()));
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

/// Checkpoints at every second page boundary of a prompt and while
/// generating, a budget of eight of them in RAM.
fn config(ssd: Option<SsdConfig>) -> RuntimeConfig {
    RuntimeConfig {
        pages: 256,
        cells_per_page: PAGE,
        max_slots: 8,
        window_rows: 1,
        warm_up: false,
        checkpoint_budget: 8 * CHECKPOINT,
        checkpoint_prompt_interval: 2 * PAGE,
        checkpoint_generation_interval: 2 * PAGE,
        ssd,
        ..RuntimeConfig::default()
    }
}

/// The machine the engines of a test follow: what it has available and its
/// clock are set by the test, and it is looked at when the test says so.
struct Memory {
    available: Arc<AtomicU64>,
    ms: Arc<AtomicU64>,
    machine: Arc<Machine>,
}

impl Memory {
    fn new(available: u64) -> Arc<Self> {
        Self::with_rule(available, Rule::default())
    }

    fn with_rule(available: u64, rule: Rule) -> Arc<Self> {
        let (bytes, ms) = (Arc::new(AtomicU64::new(available)), Arc::new(AtomicU64::new(0)));
        let (source, clock) = (bytes.clone(), ms.clone());
        let machine = Machine::new(
            rule,
            move || Some(source.load(Ordering::SeqCst)),
            move || Duration::from_millis(clock.load(Ordering::SeqCst)),
        );
        Arc::new(Self { available: bytes, ms, machine })
    }

    /// The machine has `available` bytes from now on, and has been looked at:
    /// the engines' allowances are what that leaves them.
    fn has(&self, available: u64) {
        self.available.store(available, Ordering::SeqCst);
        self.ms.fetch_add(50, Ordering::SeqCst);
        self.machine.look().expect("the machine says what it has");
    }
}

struct Rig {
    rt: Runtime,
    problems: Problems,
    ram: PinnedRam,
    /// The most bytes the checkpoint places held at the start of a step.
    peak: Arc<AtomicU64>,
}

type Observer = Box<dyn FnMut(&RecordView<'_>) + Send>;

/// An engine whose checkpoint places lie in books that follow `memory`.
/// `observer` sees the record of every step before it runs, on the
/// conductor's thread.
fn engine(
    model: &Arc<SyntheticModel>,
    config: RuntimeConfig,
    memory: &Memory,
    mut observer: Option<Observer>,
) -> Rig {
    let ram = PinnedRam::following(RAM_LIMIT, &memory.machine);
    let peak = Arc::new(AtomicU64::new(0));
    let (books, most) = (ram.clone(), peak.clone());
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_pool(config.pages, config.cells_per_page)
        .with_ram(ram.clone())
        .with_observer(move |record| {
            most.fetch_max(books.checkpoints(), Ordering::SeqCst);
            if let Some(observer) = &mut observer {
                observer(record);
            }
        });
    let problems = host.problems();
    Rig { rt: Runtime::start(model.clone(), Box::new(host), config).unwrap(), problems, ram, peak }
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

    /// Waits until the level on SSD has written `pages` pages and has
    /// nothing left to write.
    fn written(&self, pages: u64) -> Snapshot {
        let deadline = std::time::Instant::now() + LIMIT;
        loop {
            let s = self.snapshot();
            if s.counters.ssd.pages_written >= pages && s.counters.ssd.pending == 0 {
                return s;
            }
            assert!(std::time::Instant::now() < deadline, "{pages} pages were not written: {s}");
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
        RuntimeConfig { pages: 4096, checkpoint_budget: 0, ..self::config(None) },
    )
    .unwrap();
    alone(&rt, prompt, Some(max_tokens)).text
}

fn text_of(seed: u64, len: usize) -> String {
    Rng(seed).prompt(len)
}

/// Whether the record carries the save of a checkpoint.
fn saves(record: &RecordView<'_>) -> bool {
    (0..record.state_copies()).any(|i| record.state_copy(i).kind == StateCopyKind::Save)
}

#[test]
fn a_checkpoint_leaves_for_the_drive_before_the_machine_gets_its_place_back() {
    let model = model();
    let first = text_of(1, 200);

    // What the level on SSD writes of this text while memory is free.
    let free = TempDir::new("free");
    let memory = Memory::new(PLENTY);
    let e = engine(&model, config(Some(store(&free.0))), &memory, None);
    let a = e.ask(&first, 20);
    let unhurried = e.written(13);
    assert!(unhurried.counters.ssd.states_written >= 4, "{unhurried}");
    e.consistent();
    drop(e);

    // The same text on a machine that runs out of memory while the step
    // with the third save is under way: the allowance of the checkpoints
    // falls to the one place of the floor, and the conductor is told to trim.
    let dir = TempDir::new("short");
    let memory = Memory::new(PLENTY);
    let (seen, short) = (Arc::new(AtomicUsize::new(0)), memory.clone());
    let e = engine(
        &model,
        config(Some(store(&dir.0))),
        &memory,
        Some(Box::new(move |record| {
            if saves(record) && seen.fetch_add(1, Ordering::SeqCst) == 2 {
                short.has(0);
            }
        })),
    );
    assert_eq!(e.ram.checkpoints_allowance(), 8 * CHECKPOINT);
    let b = e.ask(&first, 20);
    assert_eq!(b.text, a.text);
    assert_eq!(b.text, fresh(&first, 20));
    assert_eq!(e.ram.checkpoints_allowance(), CHECKPOINT, "the floor: one place");
    let s = e.written(13);
    // The checkpoints that were in RAM went, the coldest first, and their
    // places with them; every checkpoint that was due on the drive got
    // there all the same, the one whose step was under way included: it
    // left its place before the place was called unused.
    assert!(s.counters.checkpoints_dropped >= unhurried.counters.checkpoints_dropped + 2, "{s}");
    assert_eq!(s.counters.ssd.states_written, unhurried.counters.ssd.states_written, "{s}");
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    assert_eq!(s.counters.ssd.pages_written, 13, "{s}");
    e.consistent();
    drop(e);

    // What was written is the right state: another engine resumes from it.
    let second = format!("{first}{}{}", b.text, text_of(2, 40));
    let memory = Memory::new(PLENTY);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let c = e.ask(&second, 20);
    assert_eq!(c.text, fresh(&second, 20));
    assert!(c.report.cached_tokens >= 192, "{:?}", c.report);
    assert_eq!(c.report.replayed_rows, 0, "{:?}", c.report);
    e.consistent();
}

#[test]
fn a_prompt_resumes_from_ssd_while_the_machine_has_no_room_for_checkpoints() {
    let dir = TempDir::new("resume");
    let model = model();
    let first = text_of(11, 200);
    let memory = Memory::new(PLENTY);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let a = e.ask(&first, 20);
    e.written(13);
    drop(e);

    // Another engine on a machine that has nothing to spare from the start.
    let memory = Memory::new(0);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    assert_eq!(e.ram.checkpoints_allowance(), CHECKPOINT, "nothing but the floor");
    let second = format!("{first}{}{}", a.text, text_of(12, 40));
    let b = e.ask(&second, 20);
    assert_eq!(b.text, fresh(&second, 20));
    let s = e.snapshot();
    // The checkpoint came back from the drive into the one place, was
    // restored from it, and nothing was replayed.
    assert!(b.report.cached_tokens >= 192, "{:?}", b.report);
    assert_eq!(b.report.replayed_rows, 0, "{:?}", b.report);
    assert_eq!((s.counters.ssd.states_read, s.counters.checkpoints_restored), (1, 1), "{s}");
    assert_eq!(u64::from(b.report.cached_tokens), s.counters.ssd.pages_read * u64::from(PAGE));
    // The checkpoints of the rest of the text went through places too, one
    // at a time: a place whose restore is under way and the place of the
    // next save, never more.
    assert!(s.counters.checkpoints_saved > 0, "{s}");
    assert!(e.peak.load(Ordering::SeqCst) <= 2 * CHECKPOINT, "{}", e.peak.load(Ordering::SeqCst));
    e.consistent();
    // And they reached the drive (280 cells are 17 full pages, 13 of which
    // the store had): a third turn resumes behind them, on the same machine
    // without room.
    let s = e.written(4);
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    drop(e);
    let third = format!("{second}{}{}", b.text, text_of(13, 40));
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let c = e.ask(&third, 20);
    assert_eq!(c.text, fresh(&third, 20));
    assert!(c.report.cached_tokens > b.report.cached_tokens, "{:?}", c.report);
    assert_eq!(c.report.replayed_rows, 0, "{:?}", c.report);
    e.consistent();
    drop(e);

    // Without a level on SSD the same machine leaves the cache no place at
    // all: no checkpoint is taken.
    let e = engine(&model, config(None), &memory, None);
    assert_eq!(e.ram.checkpoints_allowance(), 0);
    assert_eq!(e.ask(&first, 20).text, a.text);
    let s = e.snapshot();
    assert_eq!((s.counters.checkpoints_saved, e.peak.load(Ordering::SeqCst)), (0, 0), "{s}");
    e.consistent();
}

#[test]
fn memory_that_comes_back_is_taken_again_and_the_level_on_ssd_never_notices() {
    let dir = TempDir::new("back");
    let model = model();
    let memory = Memory::new(PLENTY);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let first = text_of(21, 200);
    let a = e.ask(&first, 20);
    e.written(13);
    let held = e.snapshot().counters.checkpoints_saved;
    assert!(held >= 4);

    // Nothing runs. The machine runs short: the conductor is woken and
    // gives the checkpoints up, down to the one of the floor.
    memory.has(0);
    let deadline = std::time::Instant::now() + LIMIT;
    while e.snapshot().counters.checkpoints_dropped == 0 {
        assert!(std::time::Instant::now() < deadline, "nothing was given up");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(e.ram.checkpoints_allowance(), CHECKPOINT);

    // The second turn finds its pages in the devices and its checkpoints
    // gone from RAM but for one; it says the same as an engine that kept
    // everything. Then memory is free again, long enough to be taken.
    let second = format!("{first}{}{}", a.text, text_of(22, 40));
    let b = e.ask(&second, 20);
    assert_eq!(b.text, fresh(&second, 20));
    for _ in 0..100 {
        memory.has(PLENTY);
    }
    assert_eq!(e.ram.checkpoints_allowance(), 8 * CHECKPOINT, "all of the budget again");
    let third = format!("{second}{}{}", b.text, text_of(23, 40));
    let c = e.ask(&third, 20);
    assert_eq!(c.text, fresh(&third, 20));
    let s = e.written(17);
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    e.consistent();
}

#[test]
fn a_save_that_never_ran_is_never_read_for_the_drive() {
    let dir = TempDir::new("unsaved");
    let model = model();
    // Pages of four cells and a checkpoint at every page boundary: of six
    // texts that are generated at once, two reach a boundary in the same
    // step (more than four cannot all differ), and two saves wait for the
    // next record.
    let small = RuntimeConfig {
        pages: 1024,
        cells_per_page: 4,
        checkpoint_prompt_interval: 4,
        checkpoint_generation_interval: 4,
        ..config(Some(store(&dir.0)))
    };
    // The machine has memory to spare whenever a step begins (the rule
    // waits a second before it takes, and four pass here)...
    let rule =
        Rule { calm: Duration::from_secs(1), calm_max: Duration::from_secs(1), ..Rule::default() };
    let memory = Memory::with_rule(PLENTY, rule);
    let spare = memory.clone();
    let observer: Observer = Box::new(move |_| (0..80).for_each(|_| spare.has(PLENTY)));
    let e = engine(&model, small.clone(), &memory, Some(observer));
    // ...and none whenever a text gets a token: after the saves of that step
    // were queued and before the record that carries them is written. The
    // cache then keeps the one place of its floor, and of two saves that
    // wait, one is given up before it ran.
    let prompts: Vec<String> = (0..6).map(|i| text_of(100 + i, 90 + 7 * i as usize)).collect();
    let expected: Vec<String> = prompts.iter().map(|p| fresh(p, 60)).collect();
    let handles: Vec<Handle> = prompts
        .iter()
        .map(|prompt| {
            let short = memory.clone();
            start_seen(&e.rt, request(prompt, Some(60)), move |event| {
                if matches!(event, ares_api::Event::Text(_)) {
                    short.has(0);
                }
            })
            .unwrap()
        })
        .collect();
    for (handle, expected) in handles.into_iter().zip(&expected) {
        assert_eq!(&handle.finish().text, expected);
    }
    let s = e.snapshot();
    let s = e.written(s.counters.ssd.pages_written);
    // What was given up before its save ran is counted, was not saved (the
    // oracle would have seen a save into a place that is not in use) and was
    // not read for the drive (it would have seen that, too).
    assert!(s.counters.ssd.states_skipped >= 1, "{s}");
    assert!(s.counters.ssd.states_written >= 1, "{s}");
    e.consistent();
    drop(e);

    // What did reach the drive is the right state: an engine with memory to
    // spare resumes every text from it.
    let memory = Memory::new(PLENTY);
    let e = engine(&model, small, &memory, None);
    for (prompt, said) in prompts.iter().zip(&expected) {
        let longer = format!("{prompt}{said}{}", text_of(120, 10));
        let again = e.ask(&longer, 10);
        assert_eq!(again.text, fresh(&longer, 10));
        assert!(again.report.cached_tokens >= 48, "{:?}", again.report);
    }
    e.consistent();
}

#[test]
fn saves_and_checkpoints_from_ssd_pass_through_the_one_place_together() {
    let dir = TempDir::new("together");
    let model = model();
    let base = text_of(31, 600);
    let memory = Memory::new(PLENTY);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let a = e.ask(&base, 10);
    e.written(38);
    drop(e);

    // A machine without room: one place. Prompts that resume from SSD (a
    // checkpoint from the drive takes the place) run next to texts that
    // are new (their saves take it, and leave for the drive from it).
    let memory = Memory::new(0);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let mut prompts: Vec<String> = Vec::new();
    for i in 0..3 {
        prompts.push(format!("{base}{}{}", a.text, text_of(40 + i, 20 + 16 * i as usize)));
        prompts.push(text_of(50 + i, 150 + 40 * i as usize));
    }
    let expected: Vec<String> = prompts.iter().map(|p| fresh(p, 40)).collect();
    for round in 0..2 {
        let handles: Vec<Handle> =
            prompts.iter().map(|p| start(&e.rt, request(p, Some(40))).unwrap()).collect();
        for (handle, expected) in handles.into_iter().zip(&expected) {
            assert_eq!(&handle.finish().text, expected, "round {round}");
        }
        e.consistent();
    }
    let s = e.snapshot();
    assert!(s.counters.ssd.states_read >= 1, "{s}");
    assert!(s.counters.checkpoints_saved > 0, "{s}");
    let s = e.written(s.counters.ssd.pages_written);
    drop(e);

    // Whatever reached the drive under these conditions is the right state:
    // an engine with memory to spare resumes every text from it.
    let memory = Memory::new(PLENTY);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    for (prompt, said) in prompts.iter().zip(&expected) {
        let longer = format!("{prompt}{said}{}", text_of(60, 20));
        assert_eq!(e.ask(&longer, 10).text, fresh(&longer, 10));
    }
    assert!(e.snapshot().counters.ssd.states_read >= 1, "{s}");
    e.consistent();
}

#[test]
fn memory_that_comes_and_goes_while_state_moves_stops_nothing() {
    let dir = TempDir::new("flicker");
    let model = model();
    let base = text_of(71, 400);
    let memory = Memory::new(PLENTY);
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let a = e.ask(&base, 10);
    e.written(25);
    drop(e);

    // A machine whose memory comes and goes every few milliseconds, watched
    // by a thread as the server's is: the watch tells the conductor to trim
    // at any moment, while steps run, while transfers run and while nothing
    // runs. The memory is free for longer than the rule waits before it
    // takes, so the allowance rises and falls all the time.
    let started = std::time::Instant::now();
    let flips = Arc::new(AtomicU64::new(0));
    let counted = flips.clone();
    let machine = Machine::new(
        Rule {
            period: Duration::from_micros(200),
            calm: Duration::from_millis(1),
            calm_max: Duration::from_millis(1),
            ..Rule::default()
        },
        move || {
            let phase = started.elapsed().as_millis() / 3;
            counted.store(phase as u64, Ordering::SeqCst);
            Some(if phase.is_multiple_of(2) { PLENTY } else { 0 })
        },
        move || started.elapsed(),
    );
    machine.watch();
    let memory =
        Memory { available: Arc::new(AtomicU64::new(0)), ms: Arc::new(AtomicU64::new(0)), machine };
    let e = engine(&model, config(Some(store(&dir.0))), &memory, None);
    let mut prompts: Vec<String> = Vec::new();
    for i in 0..3 {
        prompts.push(format!("{base}{}{}", a.text, text_of(80 + i, 30 + 16 * i as usize)));
        prompts.push(text_of(90 + i, 200 + 40 * i as usize));
    }
    let expected: Vec<String> = prompts.iter().map(|p| fresh(p, 60)).collect();
    for round in 0..4 {
        let handles: Vec<Handle> =
            prompts.iter().map(|p| start(&e.rt, request(p, Some(60))).unwrap()).collect();
        for (handle, expected) in handles.into_iter().zip(&expected) {
            // `finish` gives a request a minute: an engine that stood still
            // would fail here.
            assert_eq!(&handle.finish().text, expected, "round {round}");
        }
        e.consistent();
    }
    let s = e.snapshot();
    assert!(flips.load(Ordering::SeqCst) >= 2, "the memory changed while the requests ran");
    assert!(s.counters.checkpoints_dropped > 0, "the conductor trimmed: {s}");
    assert!(s.counters.ssd.states_read >= 1, "{s}");
    e.written(s.counters.ssd.pages_written);
    e.consistent();
}
