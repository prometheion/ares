//! The level of the prefix cache that is on SSD.
//!
//! What is written: what leaves RAM, and what is worth keeping when an engine
//! ends. The engines here keep little in RAM (their budget is a few
//! checkpoints), so that what the devices give up reaches the drive soon. An
//! answer that equals the answer of an engine without a cache says that what
//! was written to the store, read back into another engine and restored there
//! was the right state.

use std::path::PathBuf;
use std::sync::Arc;

use ares_runtime::{RuntimeConfig, SsdConfig, SyntheticConfig, SyntheticModel};

use super::*;

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

/// Checkpoints at every second page boundary of a prompt and while
/// generating, `places` of them in RAM, and with a level on SSD the floor of
/// pages.
fn config(pages: u32, places: u64, ssd: Option<SsdConfig>) -> RuntimeConfig {
    RuntimeConfig {
        pages,
        cells_per_page: PAGE,
        max_slots: 8,
        window_rows: 1,
        warm_up: false,
        cache_ram: places * CHECKPOINT + if ssd.is_some() { FLOOR } else { 0 },
        checkpoint_prompt_interval: 2 * PAGE,
        checkpoint_generation_interval: 2 * PAGE,
        ssd,
        ..RuntimeConfig::default()
    }
}

#[test]
fn a_text_resumes_from_ssd_in_another_engine_with_the_same_tokens() {
    let dir = TempDir::new("restart");
    let model = model(1, |_| {});
    let first = text_of(1, 200);

    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let a = e.ask(&first, 20);
    assert_eq!(a.run.text, fresh(&first, 20));
    assert_eq!(a.report.cached_tokens, 0);
    // Nothing is written while the devices hold the text.
    let s = e.written();
    assert_eq!((s.counters.ssd.pages_written, s.counters.ssd.states_written), (0, 0), "{s}");
    assert!(s.ssd_problem.is_none());
    e.consistent();
    // The engine ends: 220 cells are 13 full pages, and the checkpoints at
    // the page boundaries of the prompt that lie deep enough go along, with
    // the tail of the prompt's anchor (192 to 199) and the state at its end.
    // The request's end waits for a save with a record that does not come:
    // it is not written.
    drop(e);
    let s = stored(&model, &dir.0);
    assert_eq!(s.store_pages, 14, "{s:?}");
    assert!(s.store_states >= 4, "{s:?}");

    // Another engine, whose devices hold nothing: the second turn of the
    // conversation finds the first on SSD.
    let second = format!("{first}{}{}", a.run.text, text_of(2, 40));
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    assert_eq!(e.snapshot().counters.ssd.store_pages, 14);
    let b = e.ask(&second, 20);
    assert_eq!(b.run.text, fresh(&second, 20));
    let s = e.snapshot();
    // The run ends at the deepest checkpoint, the anchor's at 199: the pages
    // before it and its tail come back, the state is restored; nothing is
    // replayed, and the rest is computed.
    assert_eq!(b.report.cached_tokens, 199, "{:?}", b.report);
    assert_eq!(b.report.replayed_rows, 0, "{:?}", b.report);
    assert_eq!(b.report.computed_rows + b.report.cached_tokens, b.report.prompt_tokens);
    let read = u64::from(b.report.cached_tokens).div_ceil(u64::from(PAGE));
    assert_eq!(read, s.counters.ssd.pages_read, "12 pages and the tail: {s}");
    assert_eq!((s.counters.ssd.states_read, s.counters.checkpoints_restored), (1, 1), "{s}");
    assert_eq!(s.counters.ssd.imports, 1);
    e.consistent();

    // The same prompt again is in the devices now: nothing is read.
    let c = e.ask(&second, 20);
    assert_eq!(c.run.text, b.run.text);
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
    let b = e.ask(&two, 10);
    assert_eq!((a.report.cached_tokens, b.report.cached_tokens), (0, 0));
    let s = e.written();
    assert!(s.counters.cache_pages_evicted > 0, "the second text made room: {s}");
    // RAM keeps its floor of pages; what left the devices beyond that went to
    // the drive, the end of the text first.
    assert!(s.counters.ssd.pages_written > 0, "{s}");
    assert!(s.counters.ssd.pages_written < 19, "{s}");

    // The first text again: its pages are gone from the pool and come back,
    // from the drive (and from RAM, as far as its floor still holds them:
    // that depends on how fast the writer was).
    let again = e.ask(&one, 10);
    assert_eq!(again.run.text, a.run.text);
    assert_eq!(again.run.text, fresh(&one, 10));
    let s = e.snapshot();
    assert!(again.report.cached_tokens >= 256, "{:?}", again.report);
    assert!(s.counters.ssd.pages_read > 0, "{s}");
    assert!(s.counters.ssd.pages_read < 18, "what the pool still held was not read: {s}");
    e.consistent();
    // Nothing is written twice: the full pages of both texts, and the tails
    // with the states at their ends that the places still held, two a text
    // at most (the same text again ends where it ended).
    drop(e);
    let pages = stored(&model, &dir.0).store_pages;
    assert!((2 * 19..=2 * 19 + 4).contains(&pages), "{pages}");
}

#[test]
fn a_short_text_is_not_written() {
    let dir = TempDir::new("short");
    let model = model(1, |_| {});
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    // 30 cells of prompt and 10 of reply stay below the 48 of the rule.
    let a = e.ask(&text_of(5, 30), 10);
    assert_eq!(a.run.text, fresh(&text_of(5, 30), 10));
    drop(e);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let s = e.snapshot();
    assert_eq!((s.counters.ssd.store_pages, s.counters.ssd.store_states), (0, 0), "{s}");
    // A text that grows past the rule is written from its first page.
    let long = text_of(5, 30) + &a.run.text + &text_of(6, 60);
    e.ask(&long, 10);
    e.consistent();
    drop(e);
    let pages = stored(&model, &dir.0).store_pages;
    assert_eq!(pages, 7, "110 cells are 6 full pages, and the prompt's anchor (96 to 99) a tail");
}

#[test]
fn state_of_another_fingerprint_is_never_used_and_makes_room_first() {
    let dir = TempDir::new("fingerprint");
    let prompt = text_of(7, 400);
    let at = |max_bytes| SsdConfig { max_bytes, ..store(&dir.0) };
    let e = engine(&model(1, |_| {}), config(256, 8, Some(at(1 << 30))));
    let a = e.ask(&prompt, 10);
    drop(e);
    let old_bytes = stored(&model(1, |_| {}), &dir.0).store_bytes;
    let stores = || std::fs::read_dir(&dir.0).unwrap().count();
    assert_eq!(stores(), 1);

    // Other weights, the same text: nothing is taken from the store.
    let other = model(2, |_| {});
    let e = engine(&other, config(256, 8, Some(at(1 << 30))));
    let b = e.ask(&prompt, 10);
    assert_eq!(b.run.text, a.run.text, "the synthetic model's text does not depend on the number");
    assert_eq!(b.report.cached_tokens, 0);
    let s = e.written();
    assert_eq!(s.counters.ssd.pages_read, 0, "{s}");
    e.consistent();
    drop(e);
    assert_eq!(stores(), 2);

    // A limit that holds one store and not two: opening the second removes
    // the first, which nobody has open.
    let e = engine(&other, config(256, 8, Some(at(old_bytes + old_bytes / 2))));
    assert_eq!(stores(), 1);
    let c = e.ask(&prompt, 10);
    assert_eq!(c.run.text, a.run.text);
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
    // What the pool gives up goes to RAM, and from there to the drive.
    let mut answers = Vec::new();
    for text in &texts {
        answers.push(e.ask(text, 8).run.text);
        e.written();
    }
    let s = e.written();
    assert!(s.counters.ssd.store_bytes <= at.max_bytes, "{s}");
    assert!(s.counters.ssd.store_pages < 8 * 19, "{s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");
    drop(e);

    let e = engine(&model, config(64, 4, Some(at)));
    // The newest text is there, the oldest is not: it is computed, and both
    // say what they said.
    let newest = e.ask(&texts[7], 8);
    assert!(newest.report.cached_tokens >= 256, "{:?}", newest.report);
    assert_eq!(newest.run.text, answers[7]);
    let oldest = e.ask(&texts[0], 8);
    assert_eq!(oldest.report.cached_tokens, 0, "{:?}", oldest.report);
    assert_eq!(oldest.run.text, answers[0]);
    e.consistent();
}

/// A drive that runs full: the store lies in a tmpfs of 700 KiB, which the
/// test mounts in a user namespace of its own (it runs itself again in one).
/// The store's own limit (1 GiB) never comes near; the drive's refusals
/// (`ENOSPC`) make it give up its oldest segments and write on. Where user
/// namespaces are not allowed the test says so and passes.
#[test]
fn a_full_drive_gives_up_the_oldest_segments_and_the_store_writes_on() {
    const CHILD: &str = "ARES_TEST_FULL_DRIVE";
    const NAME: &str = "ssd::a_full_drive_gives_up_the_oldest_segments_and_the_store_writes_on";
    let Some(dir) = std::env::var_os(CHILD).map(PathBuf::from) else {
        let dir = TempDir::new("full");
        let script = format!(
            "mount -t tmpfs -o size=700k none '{}' || exit 77; exec \"$0\" --exact {NAME} \
             --test-threads=1",
            dir.0.display()
        );
        let exe = std::env::current_exe().unwrap();
        let run = std::process::Command::new("unshare")
            .args(["--user", "--map-root-user", "--mount", "sh", "-c", &script])
            .arg(exe)
            .env(CHILD, &dir.0)
            .status();
        match run.map(|status| status.code()) {
            Ok(Some(0)) => {}
            Ok(Some(77) | Some(1)) | Err(_) => {
                eprintln!("skipped: no user namespace with a tmpfs of its own here");
            }
            Ok(other) => panic!("the test in its own namespace failed: {other:?}"),
        }
        return;
    };
    let model = model(1, |_| {});
    // Segments of 160 KiB, which hold the pages of one text of 300 cells; the
    // drive holds about three of them, the store's limit far more.
    let at = SsdConfig { segment_bytes: 160 << 10, ..store(&dir) };
    let texts: Vec<String> = (0..8).map(|i| text_of(60 + i, 300)).collect();
    let e = engine(&model, config(64, 4, Some(at.clone())));
    let mut answers = Vec::new();
    for text in &texts {
        let answer = e.ask(text, 8);
        assert_eq!(answer.run.text, fresh(text, 8));
        answers.push(answer.run.text);
        e.written();
    }
    let s = e.written();
    assert!(s.ssd_problem.is_none(), "{s}");
    assert!(s.counters.ssd.store_bytes <= 700 << 10, "{s}");
    // More was written than the drive holds: it made room and wrote on.
    assert!(s.counters.ssd.bytes_written > 700 << 10, "{s}");
    assert_eq!(s.counters.ram.pages_dropped, 0, "{s}");
    e.consistent();
    drop(e);

    // The newest text resumes from the drive in another engine; the oldest
    // was given up and is computed. Both say what they said.
    let e = engine(&model, config(64, 4, Some(at)));
    let newest = e.ask(&texts[7], 8);
    assert!(newest.report.cached_tokens >= 256, "{:?}", newest.report);
    assert_eq!(newest.run.text, answers[7]);
    let oldest = e.ask(&texts[0], 8);
    assert_eq!(oldest.report.cached_tokens, 0, "{:?}", oldest.report);
    assert_eq!(oldest.run.text, answers[0]);
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
    let b = e.ask(&two, 10);
    // The engine ends: the pages of the devices are written, the least
    // recently used first, so the first text's come first.
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
    assert_eq!(again.run.text, a.run.text);
    // The run was cut at the damaged page: what came before it is used.
    assert!(again.report.cached_tokens < 4 * PAGE, "{:?}", again.report);
    let s = e.snapshot();
    assert_eq!(s.counters.ssd.imports_cut, 1, "{s}");
    assert_eq!(e.ask(&two, 10).run.text, b.run.text);
    e.consistent();

    // The directory vanishes while the engine runs: it keeps serving, and
    // what it computes from then on is written to a new one.
    std::fs::remove_dir_all(&dir.0).unwrap();
    let three = text_of(42, 600);
    let c = e.ask(&three, 10);
    assert_eq!(c.run.text, fresh(&three, 10));
    let four = text_of(43, 600);
    let d = e.ask(&four, 10);
    assert_eq!(d.run.text, fresh(&four, 10));
    e.consistent();
    drop(e);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let s = e.snapshot();
    assert!(s.ssd_problem.is_none() && s.counters.ssd.enabled, "{s}");
    let d2 = e.ask(&four, 10);
    assert_eq!(d2.run.text, d.run.text);
    e.consistent();
}

#[test]
fn one_place_for_checkpoints_is_enough() {
    let dir = TempDir::new("one-place");
    let model = model(1, |_| {});
    let first = text_of(50, 330);
    let e = engine(&model, config(256, 1, Some(store(&dir.0))));
    let a = e.ask(&first, 30);
    let s = e.written();
    // Every checkpoint went through the one place, and to the drive as the
    // next one took the place over.
    assert!(s.counters.ssd.states_written >= 8, "{s}");
    assert_eq!(s.counters.ssd.states_skipped, 0, "{s}");
    assert_eq!(s.counters.ssd.pages_written, 0, "the devices hold the pages: {s}");
    drop(e);

    let second = format!("{first}{}{}", a.run.text, text_of(51, 20));
    let e = engine(&model, config(256, 1, Some(store(&dir.0))));
    let b = e.ask(&second, 20);
    assert_eq!(b.run.text, fresh(&second, 20));
    assert!(b.report.cached_tokens >= 352, "{:?}", b.report);
    assert_eq!(b.report.replayed_rows, 0);
    e.consistent();
}

#[test]
fn pages_that_settle_late_are_written_in_the_form_they_have() {
    let dir = TempDir::new("settle");
    // The model says its last 40 cells have not reached the form they keep.
    let model = model(1, |c| c.settle_cells = 40);
    let first = text_of(60, 200);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let a = e.ask(&first, 20);
    drop(e);
    let pages = stored(&model, &dir.0).store_pages;
    assert_eq!(pages, 14, "the end writes them all, and the tail of the anchor");
    let second = format!("{first}{}{}", a.run.text, text_of(61, 40));
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let b = e.ask(&second, 20);
    assert_eq!(b.run.text, fresh(&second, 20));
    assert!(b.report.cached_tokens >= 192, "{:?}", b.report);
    e.consistent();
}

#[test]
fn a_store_that_the_engine_before_holds_a_moment_longer_is_waited_for() {
    // The engine before lets the store go 300 ms after the next one starts
    // (an unload and a load, a lock that a forked process holds until it
    // runs its program): the next one waits and has its level on SSD.
    let dir = TempDir::new("moment");
    let model = model(1, |_| {});
    let before = engine(&model, config(256, 8, Some(store(&dir.0))));
    let gone = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        drop(before);
    });
    let next = engine(&model, config(256, 8, Some(store(&dir.0))));
    gone.join().unwrap();
    let s = next.snapshot();
    assert!(s.counters.ssd.enabled, "{:?}", s.ssd_problem);
    let prompt = text_of(71, 100);
    assert_eq!(next.ask(&prompt, 10).run.text, fresh(&prompt, 10));
    next.consistent();
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
    assert_eq!(second.ask(&prompt, 10).run.text, first.ask(&prompt, 10).run.text);
    // A model whose devices move no state says so, too.
    let plain = self::model(1, |c| c.durable = None);
    let third = engine(&plain, config(256, 8, Some(store(&dir.0))));
    assert!(third.snapshot().ssd_problem.is_some());
    assert_eq!(third.ask(&prompt, 10).run.text, fresh(&prompt, 10));
}

#[test]
fn requests_that_share_a_run_from_ssd_wait_for_it_together() {
    let dir = TempDir::new("together");
    let model = model(1, |_| {});
    let base = text_of(80, 600);
    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let a = e.ask(&base, 10);
    drop(e);

    let e = engine(&model, config(256, 8, Some(store(&dir.0))));
    let prompts: Vec<String> =
        (0..4).map(|i| format!("{base}{}{}", a.run.text, text_of(90 + i, 20))).collect();
    let handles: Vec<Handle> =
        prompts.iter().map(|p| start(&e.rt, request(p, Some(10))).unwrap()).collect();
    for (handle, prompt) in handles.into_iter().zip(&prompts) {
        let run = handle.finish();
        assert_eq!(run.text, fresh(prompt, 10));
        assert!(run.usage.cached_prompt_tokens >= 592, "{:?}", run.usage);
    }
    let s = e.snapshot();
    assert_eq!(s.counters.ssd.imports, 1, "one run serves all four: {s}");
    // 37 pages and the tail of the prompt's anchor (592 to 599).
    assert_eq!(s.counters.ssd.pages_read, 38, "{s}");
    e.consistent();
}
