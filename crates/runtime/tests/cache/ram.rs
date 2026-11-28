//! The level of the prefix cache in RAM.
//!
//! Pages that the pool gives up are copied into RAM, and a prompt that finds
//! its start there has them copied back: an answer that equals the answer of
//! an engine without a cache says that what came back from RAM was the right
//! state. Below RAM lies the level on SSD, which is written to only when
//! something leaves RAM and when an engine ends.

use std::sync::Arc;

use ares_runtime::testing::TestMachine;
use ares_runtime::{
    FarPages, Machine, PinnedRam, RuntimeConfig, SsdConfig, SyntheticConfig, SyntheticModel,
};

use super::*;

/// A model with recurrent state whose device side moves state to the host.
fn model() -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.prefix_cache = Some(CHECKPOINT);
    config.durable = Some(1);
    Arc::new(SyntheticModel::new(config).unwrap())
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

/// An engine whose checkpoints and pages in RAM lie in books that follow
/// `machine`, below a limit of `limit` bytes.
fn following(
    model: &Arc<SyntheticModel>,
    config: RuntimeConfig,
    machine: &Arc<Machine>,
    limit: u64,
) -> (Rig, PinnedRam) {
    let ram = PinnedRam::following(limit, machine);
    let rig = engine_with(model, config, |host| host.with_ram(ram.clone()));
    (rig, ram)
}

#[test]
fn a_conversation_resumes_from_ram_after_the_devices_gave_it_up() {
    let model = model();
    // No level on SSD: RAM is the only place below the devices.
    let e = engine(&model, config(32, 64, None));
    let (one, two) = (text_of(1, 300), text_of(2, 300));
    let a = e.ask(&one, 10);
    let b = e.ask(&two, 10);
    assert_eq!((a.run.text.clone(), b.run.text), (fresh(&one, 10), fresh(&two, 10)));
    let s = snapshot(&e.rt);
    assert!(s.counters.cache_pages_evicted > 0, "the second text made room: {s}");
    assert!(s.counters.ram.pages_in > 0, "what the pool gave up went into RAM: {s}");
    assert!(!s.counters.ssd.enabled);

    // The first conversation's next turn: its pages and the tail with the
    // state at its end come back from RAM, and nothing is replayed.
    let second = format!("{one}{}{}", a.run.text, text_of(3, 40));
    let c = e.ask(&second, 10);
    assert_eq!(c.run.text, fresh(&second, 10));
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
    assert_eq!(d.run.text, fresh(&again, 10));
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
            assert_eq!(answer.run.text, fresh(text, 10), "turn {turn} of conversation {i}");
            if turn > 0 {
                assert!(answer.report.cached_tokens >= 300, "{:?}", answer.report);
                assert_eq!(answer.report.replayed_rows, 0, "{:?}", answer.report);
            }
            text.push_str(&answer.run.text);
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
        assert_eq!(answer.run.text, fresh(&next, 10));
        // From the deepest state on the drive: at a page boundary or at the
        // end of a tail (a prompt's anchor, a request's end whose save ran).
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
    let (one, two, three) = (text_of(30, 300), text_of(31, 300), text_of(33, 300));
    let a = e.ask(&one, 10);
    // Two more texts take the pool: all 20 pages of the first and 8 of the
    // second leave the devices, and RAM, which has room for 24 at most (its
    // floor and the places no checkpoint uses), gives up what arrived first,
    // the end of the first text. With one text more only 8 pages left, and
    // whether RAM kept the ones the next turn resumes from depended on how
    // many places were in use as they arrived.
    for text in [&two, &three] {
        assert_eq!(e.ask(text, 10).run.text, fresh(text, 10));
    }
    let s = e.written();
    assert!(s.counters.ssd.pages_written > 0, "what left RAM was written: {s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");

    let second = format!("{one}{}{}", a.run.text, text_of(32, 40));
    let c = e.ask(&second, 10);
    assert_eq!(c.run.text, fresh(&second, 10));
    let s = snapshot(&e.rt);
    assert!(c.report.cached_tokens >= 256, "{:?}", c.report);
    assert!(s.counters.ssd.pages_read > 0, "{s}");
    e.consistent();
}

/// The same beside far pages: the model keeps the pages from page 4 on in host memory, one
/// request holds at most 20 of them (a context of 24 pages) and the pool has 28, and a tier of
/// experts that wants all the limit leaves the prompt cache its floor (one place, the pages the
/// drive moves through RAM and one request's far pages). The conversation comes back from the
/// drive, and its run is asked for once (#98 needed the cards to show its loop: the cache unit
/// tests `far_pages_beyond_the_floor_leave_the_floor_its_place` and
/// `a_run_whose_import_was_cut_is_not_asked_for_by_the_next_look` show its two causes). Where
/// the run's far pages were the only ones at the end of a chain, the trim after the run came
/// back took them, and the request asked for it again for ever: on CI's runners and now and then
/// on a quiet machine (#104, `a_run_that_came_back_whole_is_left_to_the_look_after_the_trim`).
#[test]
fn a_conversation_with_far_pages_comes_back_from_ssd_beside_a_hungry_tier() {
    let dir = TempDir::new("far-floor");
    // A checkpoint larger than a page in RAM, as a real model's (112.6 MiB against a page's
    // 0.8 MiB on Flash-Next): far pages beyond the floor's fit in the room of a place.
    let checkpoint = 16 * PAGE_BYTES;
    let mut shape = SyntheticConfig::small(Split::Single, 1);
    shape.context_length = 24 * PAGE;
    shape.prefix_cache = Some(checkpoint);
    shape.durable = Some(1);
    shape.far_pages = Some(FarPages { first: 4, page_bytes: 256 });
    let model = Arc::new(SyntheticModel::new(shape).unwrap());
    let ram = PinnedRam::new(1 << 30);
    ram.want_experts(1 << 40, 0);
    let books = ram.clone();
    let config =
        RuntimeConfig { cache_ram: 8 * checkpoint + FLOOR, ..config(32, 0, Some(store(&dir.0))) };
    let e = engine_with(&model, config, move |host| host.with_ram(books));
    let (one, two, three) = (text_of(30, 300), text_of(31, 300), text_of(33, 300));
    let a = e.ask(&one, 10);
    for text in [&two, &three] {
        assert_eq!(e.ask(text, 10).run.text, fresh(text, 10));
    }
    let s = e.written();
    assert!(s.counters.ssd.pages_written > 0, "what left RAM was written: {s}");

    let second = format!("{one}{}{}", a.run.text, text_of(32, 40));
    let c = e.ask(&second, 10);
    assert_eq!(c.run.text, fresh(&second, 10));
    let s = snapshot(&e.rt);
    assert!(s.counters.ssd.imports_cut <= 1, "the run was not asked for again and again: {s}");
    assert!(c.report.cached_tokens >= 256, "{:?}", c.report);
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
    let second = format!("{one}{}{}", a.run.text, text_of(52, 40));
    let c = e.ask(&second, 10);
    assert_eq!(c.run.text, fresh(&second, 10));
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
    let memory = TestMachine::new(PLENTY);
    let budget = config(32, 256, Some(store(&dir.0)));
    let (e, ram) = following(&model, budget, memory.machine(), 1 << 30);
    // Four conversations pass through a pool that holds one and a half: RAM
    // holds what the pool gave up, and nothing is written.
    let texts: Vec<String> = (0..4).map(|i| text_of(60 + i, 300)).collect();
    let answers: Vec<String> = texts.iter().map(|text| e.ask(text, 10).run.text).collect();
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
    // leave the devices for them pass through it to the drive, and the next
    // turn asks once the writer has them there: a page that is with the
    // writer is in neither place yet (docs/OPEN.md), which a slow drive
    // showed now and then.
    for (text, said) in texts.iter().zip(&answers) {
        let next = format!("{text}{said}{}", text_of(70, 20));
        e.written();
        let answer = e.ask(&next, 10);
        assert_eq!(answer.run.text, fresh(&next, 10));
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
        assert_eq!(answer.run.text, fresh(&text, 2), "turn {turn}");
        text.push_str(&answer.run.text);
        text.push_str(&text_of(81 + turn, 2));
    }
    e.consistent();
    drop(e);
    assert!(stored(&model, &dir.0).store_states >= 1);

    let e = engine(&model, config(256, 64, Some(store(&dir.0))));
    let answer = e.ask(&text, 10);
    assert_eq!(answer.run.text, fresh(&text, 10));
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
    let answers: Vec<String> = texts.iter().map(|text| e.ask(text, 10).run.text).collect();
    assert_eq!(e.written().counters.ssd.pages_written, 0);
    drop(e);
    let stored = stored(&model, &dir.0);
    // The 57 full pages, and the tails with the states at their ends that the
    // places still held (a prompt's anchor, a request's end; one whose save
    // waits for a record that does not come is not written): two a text at
    // most.
    assert!((3 * 19..=3 * 19 + 6).contains(&stored.store_pages), "{stored:?}");
    let e = engine(&model, config(256, 0, Some(store(&dir.0))));
    for (text, said) in texts.iter().zip(&answers) {
        let next = format!("{text}{said}{}", text_of(95, 20));
        let answer = e.ask(&next, 10);
        assert_eq!(answer.run.text, fresh(&next, 10));
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
    let memory = TestMachine::new(PLENTY);
    let budget = RuntimeConfig {
        cache_ram: 64 * 32 * PAGE_BYTES + 256 * PAGE_BYTES,
        ..self::config(32, 0, Some(store(&dir.0)))
    };
    let limit = 32 * PAGE_BYTES + FLOOR + 4 * 32 * PAGE_BYTES + 48 * PAGE_BYTES;
    let (e, ram) = following(&model, budget, memory.machine(), limit);
    // Texts pass through a pool that holds one and a half of them: RAM fills
    // with what only it holds, while every text takes checkpoints. A place
    // whose memory the limit could not lock would be a problem the oracle
    // writes down.
    for i in 0..6 {
        let text = text_of(110 + i, 300);
        let answer = e.ask(&text, 10);
        assert_eq!(answer.run.text, fresh(&text, 10));
        e.consistent();
        assert!(ram.used() <= limit, "{} of {limit}", ram.used());
    }
    let s = e.written();
    assert!(s.counters.checkpoints_saved > 0, "{s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");
}
