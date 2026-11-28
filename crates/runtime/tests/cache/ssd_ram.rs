//! Where the host memory that follows the machine meets the level of the
//! prefix cache on SSD.
//!
//! The executor is given the books of page-locked RAM
//! ([`HostExecutor::with_ram`]): its checkpoint places lie in chunks of them,
//! booked and given back as a device part's are, and a place loses what it
//! held as soon as the conductor has called it unused. A save into such a
//! place and a checkpoint that leaves such a place for the drive are problems
//! the oracle writes down, and a state that is not the one its history made
//! changes the text.
//! The books follow a machine whose available memory and clock are the
//! test's, so "the machine has no room for checkpoints" is a line of a test.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use ares_runtime::testing::TestMachine;
use ares_runtime::{
    Machine, PinnedRam, RecordView, Rule, RuntimeConfig, SsdConfig, StateCopyKind, SyntheticConfig,
    SyntheticModel,
};

use super::*;

/// The limit of the engine's books.
const RAM_LIMIT: u64 = 1 << 20;

/// A model with recurrent state whose device side moves state to the host.
fn model() -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.prefix_cache = Some(CHECKPOINT);
    config.durable = Some(1);
    Arc::new(SyntheticModel::new(config).unwrap())
}

/// Checkpoints at every second page boundary of a prompt and while
/// generating, a budget of eight of them in RAM, and with a level on SSD the
/// floor of pages.
fn config(ssd: Option<SsdConfig>) -> RuntimeConfig {
    RuntimeConfig {
        pages: 256,
        cells_per_page: PAGE,
        max_slots: 8,
        window_rows: 1,
        warm_up: false,
        cache_ram: 8 * CHECKPOINT + if ssd.is_some() { FLOOR } else { 0 },
        checkpoint_prompt_interval: 2 * PAGE,
        checkpoint_generation_interval: 2 * PAGE,
        ssd,
        ..RuntimeConfig::default()
    }
}

/// An engine of the rig, the books it follows, and the most bytes the
/// checkpoint places held at the start of a step.
struct Following {
    rig: Rig,
    ram: PinnedRam,
    peak: Arc<AtomicU64>,
}

impl std::ops::Deref for Following {
    type Target = Rig;
    fn deref(&self) -> &Rig {
        &self.rig
    }
}

type Observer = Box<dyn FnMut(&RecordView<'_>) + Send>;

/// An engine whose checkpoint places lie in books that follow `machine`.
/// `observer` sees the record of every step before it runs, on the
/// conductor's thread.
fn engine(
    model: &Arc<SyntheticModel>,
    config: RuntimeConfig,
    machine: &Arc<Machine>,
    mut observer: Option<Observer>,
) -> Following {
    let ram = PinnedRam::following(RAM_LIMIT, machine);
    let peak = Arc::new(AtomicU64::new(0));
    let (books, most) = (ram.clone(), peak.clone());
    let rig = engine_with(model, config, |host| {
        host.with_ram(ram.clone()).with_observer(move |record| {
            most.fetch_max(books.checkpoints(), Ordering::SeqCst);
            if let Some(observer) = &mut observer {
                observer(record);
            }
        })
    });
    Following { rig, ram, peak }
}

/// Whether the record carries the save of a checkpoint.
fn saves(record: &RecordView<'_>) -> bool {
    (0..record.state_copies()).any(|i| record.state_copy(i).kind == StateCopyKind::Save)
}

#[test]
fn a_checkpoint_leaves_for_the_drive_before_the_machine_gets_its_place_back() {
    let model = model();
    let first = text_of(1, 200);

    // What the level on SSD holds of this text while memory is free: nothing
    // while the engine runs, and what is worth keeping once it ends.
    let free = TempDir::new("free");
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, config(Some(store(&free.0))), memory.machine(), None);
    let a = e.ask(&first, 20);
    let unhurried = e.written();
    assert_eq!(unhurried.counters.ssd.states_written, 0, "{unhurried}");
    e.consistent();
    drop(e);
    let kept = stored(&model, &free.0);
    assert!(kept.store_states >= 4, "{kept:?}");

    // The same text on a machine that runs out of memory while the step
    // with the third save is under way: the allowance of the prompt cache
    // falls to its floor, one place, and the conductor is told to trim.
    let dir = TempDir::new("short");
    let memory = Arc::new(TestMachine::new(PLENTY));
    let (seen, short) = (Arc::new(AtomicUsize::new(0)), memory.clone());
    let e = engine(
        &model,
        config(Some(store(&dir.0))),
        memory.machine(),
        Some(Box::new(move |record| {
            if saves(record) && seen.fetch_add(1, Ordering::SeqCst) == 2 {
                short.has(0);
            }
        })),
    );
    assert_eq!(e.ram.cache_allowance(), 8 * CHECKPOINT + FLOOR);
    let b = e.ask(&first, 20);
    assert_eq!(b.run.text, a.run.text);
    assert_eq!(b.run.text, fresh(&first, 20));
    assert_eq!(e.ram.cache_allowance(), CHECKPOINT + FLOOR, "the floor: one place and pages");
    let s = e.written();
    // The checkpoints that were in RAM went, the coldest first, and their
    // places with them; each went to the drive as it left, before its place
    // was called unused (the oracle would have seen a read of a place that
    // lost its memory).
    assert!(s.counters.checkpoints_dropped >= unhurried.counters.checkpoints_dropped + 2, "{s}");
    assert!(s.counters.ssd.states_written >= 2, "{s}");
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    assert_eq!(s.counters.ssd.pages_written, 0, "the devices hold the pages: {s}");
    e.consistent();
    drop(e);
    // 13 full pages and the tail of the prompt's anchor.
    assert_eq!(stored(&model, &dir.0).store_pages, 14);

    // What was written is the right state: another engine resumes from it.
    let second = format!("{first}{}{}", b.run.text, text_of(2, 40));
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    let c = e.ask(&second, 20);
    assert_eq!(c.run.text, fresh(&second, 20));
    assert!(c.report.cached_tokens >= 192, "{:?}", c.report);
    assert_eq!(c.report.replayed_rows, 0, "{:?}", c.report);
    e.consistent();
}

#[test]
fn a_prompt_resumes_from_ssd_while_the_machine_has_no_room_for_checkpoints() {
    let dir = TempDir::new("resume");
    let model = model();
    let first = text_of(11, 200);
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    let a = e.ask(&first, 20);
    drop(e);

    // Another engine on a machine that has nothing to spare from the start.
    let memory = Arc::new(TestMachine::new(0));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    assert_eq!(e.ram.cache_allowance(), CHECKPOINT + FLOOR, "nothing but the floor");
    let second = format!("{first}{}{}", a.run.text, text_of(12, 40));
    let b = e.ask(&second, 20);
    assert_eq!(b.run.text, fresh(&second, 20));
    let s = e.snapshot();
    // The checkpoint came back from the drive into the one place, was
    // restored from it, and nothing was replayed.
    assert!(b.report.cached_tokens >= 192, "{:?}", b.report);
    assert_eq!(b.report.replayed_rows, 0, "{:?}", b.report);
    assert_eq!((s.counters.ssd.states_read, s.counters.checkpoints_restored), (1, 1), "{s}");
    // The pages before the deepest state and, where it is a tail's, the tail.
    let read = u64::from(b.report.cached_tokens).div_ceil(u64::from(PAGE));
    assert_eq!(read, s.counters.ssd.pages_read, "{s}");
    // The checkpoints of the rest of the text went through places too, one
    // at a time: a place whose restore is under way and the place of the
    // next save, never more.
    assert!(s.counters.checkpoints_saved > 0, "{s}");
    assert!(e.peak.load(Ordering::SeqCst) <= 2 * CHECKPOINT, "{}", e.peak.load(Ordering::SeqCst));
    e.consistent();
    // And they reach the drive when they leave the place, or when the engine
    // ends (280 cells are 17 full pages, 13 of which the store had, and the
    // tails of both prompts' anchors): a third turn resumes behind them, on
    // the same machine without room.
    let s = e.written();
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    drop(e);
    assert_eq!(stored(&model, &dir.0).store_pages, 19);
    let third = format!("{second}{}{}", b.run.text, text_of(13, 40));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    let c = e.ask(&third, 20);
    assert_eq!(c.run.text, fresh(&third, 20));
    assert!(c.report.cached_tokens > b.report.cached_tokens, "{:?}", c.report);
    assert_eq!(c.report.replayed_rows, 0, "{:?}", c.report);
    e.consistent();
    drop(e);

    // Without a level on SSD the same machine leaves the cache its one
    // checkpoint place and no pages: checkpoints are still taken, one at a
    // time in that place.
    let e = engine(&model, config(None), memory.machine(), None);
    assert_eq!(e.ram.cache_allowance(), CHECKPOINT);
    assert_eq!(e.ask(&first, 20).run.text, a.run.text);
    let s = e.snapshot();
    assert!(s.counters.checkpoints_saved > 0, "{s}");
    assert!(e.peak.load(Ordering::SeqCst) <= CHECKPOINT, "{}", e.peak.load(Ordering::SeqCst));
    e.consistent();
}

#[test]
fn memory_that_comes_back_is_taken_again_and_the_level_on_ssd_never_notices() {
    let dir = TempDir::new("back");
    let model = model();
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    let first = text_of(21, 200);
    let a = e.ask(&first, 20);
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
    assert_eq!(e.ram.cache_allowance(), CHECKPOINT + FLOOR);
    // The checkpoints that went were written as they went.
    let s = e.written();
    assert!(s.counters.ssd.states_written >= 1, "{s}");
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");

    // The second turn finds its pages in the devices and its checkpoints
    // gone from RAM but for one; it says the same as an engine that kept
    // everything. Then memory is free again, long enough to be taken.
    let second = format!("{first}{}{}", a.run.text, text_of(22, 40));
    let b = e.ask(&second, 20);
    assert_eq!(b.run.text, fresh(&second, 20));
    for _ in 0..100 {
        memory.has(PLENTY);
    }
    assert_eq!(e.ram.cache_allowance(), 8 * CHECKPOINT + FLOOR, "all of the budget again");
    let third = format!("{second}{}{}", b.run.text, text_of(23, 40));
    let c = e.ask(&third, 20);
    assert_eq!(c.run.text, fresh(&third, 20));
    let s = e.written();
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
    let memory = Arc::new(TestMachine::with_rule(PLENTY, rule));
    let spare = memory.clone();
    let observer: Observer = Box::new(move |_| (0..80).for_each(|_| spare.has(PLENTY)));
    let e = engine(&model, small.clone(), memory.machine(), Some(observer));
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
    let s = e.written();
    // What was given up before its save ran is counted, was not saved (the
    // oracle would have seen a save into a place that is not in use) and was
    // not read for the drive (it would have seen that, too).
    assert!(s.counters.ssd.states_skipped >= 1, "{s}");
    assert!(s.counters.ssd.states_written >= 1, "{s}");
    e.consistent();
    drop(e);

    // What did reach the drive is the right state: an engine with memory to
    // spare resumes every text from it.
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, small, memory.machine(), None);
    for (prompt, said) in prompts.iter().zip(&expected) {
        let longer = format!("{prompt}{said}{}", text_of(120, 10));
        let again = e.ask(&longer, 10);
        assert_eq!(again.run.text, fresh(&longer, 10));
        assert!(again.report.cached_tokens >= 48, "{:?}", again.report);
    }
    e.consistent();
}

#[test]
fn saves_and_checkpoints_from_ssd_pass_through_the_one_place_together() {
    let dir = TempDir::new("together");
    let model = model();
    let base = text_of(31, 600);
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    let a = e.ask(&base, 10);
    drop(e);

    // A machine without room: one place. Prompts that resume from SSD (a
    // checkpoint from the drive takes the place) run next to texts that
    // are new (their saves take it, and leave for the drive from it).
    let memory = Arc::new(TestMachine::new(0));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    let mut prompts: Vec<String> = Vec::new();
    for i in 0..3 {
        prompts.push(format!("{base}{}{}", a.run.text, text_of(40 + i, 20 + 16 * i as usize)));
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
    let s = e.written();
    drop(e);

    // Whatever reached the drive under these conditions is the right state:
    // an engine with memory to spare resumes every text from it.
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    for (prompt, said) in prompts.iter().zip(&expected) {
        let longer = format!("{prompt}{said}{}", text_of(60, 20));
        assert_eq!(e.ask(&longer, 10).run.text, fresh(&longer, 10));
    }
    assert!(e.snapshot().counters.ssd.states_read >= 1, "{s}");
    e.consistent();
}

#[test]
fn memory_that_comes_and_goes_while_state_moves_stops_nothing() {
    let dir = TempDir::new("flicker");
    let model = model();
    let base = text_of(71, 400);
    let memory = Arc::new(TestMachine::new(PLENTY));
    let e = engine(&model, config(Some(store(&dir.0))), memory.machine(), None);
    let a = e.ask(&base, 10);
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
    let e = engine(&model, config(Some(store(&dir.0))), &machine, None);
    let mut prompts: Vec<String> = Vec::new();
    for i in 0..3 {
        prompts.push(format!("{base}{}{}", a.run.text, text_of(80 + i, 30 + 16 * i as usize)));
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
    e.written();
    e.consistent();
}
