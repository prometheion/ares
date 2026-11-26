//! The index on the devices: the pages, tails and checkpoints the conductor
//! shares, restores, replays and evicts, and what the per-request reports say
//! it computed.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ares_api::Engine;
use ares_runtime::testing::{Gate, HostExecutor};
use ares_runtime::{
    FarPages, PinnedRam, RuntimeConfig, Split, StateCopyKind, SyntheticConfig, SyntheticModel,
};

use super::*;

fn model(
    prefix_cache: Option<u64>,
    configure: impl FnOnce(&mut SyntheticConfig),
) -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 512;
    config.prefix_cache = prefix_cache;
    configure(&mut config);
    Arc::new(SyntheticModel::new(config).unwrap())
}

/// How a test shapes the synthetic model.
type Shape = fn(&mut SyntheticConfig);

/// A text that loops and changes its loop every 40 positions, so that drafts
/// are accepted and rejected both.
fn repeating(c: &mut SyntheticConfig) {
    c.context.window = 1;
    c.context.epoch = 40;
    c.context.buckets = 512;
    c.mtp_links = 7;
    c.mtp_accuracy = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
}

/// The rows of a window: drafts or none.
fn window(speculation: bool) -> u32 {
    if speculation { 8 } else { 1 }
}

/// A recurrent model whose checkpoints are taken at every page boundary of a
/// prompt and every two pages while generating.
fn config(pages: u32, checkpoints: u64) -> RuntimeConfig {
    RuntimeConfig {
        pages,
        cells_per_page: PAGE,
        max_slots: 16,
        window_rows: 1,
        warm_up: false,
        cache_ram: checkpoints * CHECKPOINT,
        checkpoint_prompt_interval: PAGE,
        checkpoint_generation_interval: 2 * PAGE,
        ..RuntimeConfig::default()
    }
}

/// Waits until the conductor has dispatched `n` steps, which an observer
/// counts in `steps`. It is inside the last of them, or about to be: what is
/// submitted from here on it takes in after that step.
fn dispatched(steps: &AtomicU32, n: u32) {
    let deadline = Instant::now() + LIMIT;
    while steps.load(Ordering::SeqCst) < n {
        assert!(Instant::now() < deadline, "the conductor never dispatched step {n}");
        std::thread::sleep(Duration::from_micros(200));
    }
}

/// The first `n` tokens of a text: a token is a character here.
fn head(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

fn tokens_of(text: &str) -> u64 {
    text.chars().count() as u64
}

#[test]
fn the_same_prompt_twice_computes_only_what_the_cache_lacks_the_second_time() {
    let mut rng = Rng(7);
    let prompt = rng.prompt(100);
    let model = model(Some(CHECKPOINT), |_| {});
    let e = engine(&model, config(256, 16));
    let first = e.ask(&prompt, 40);
    assert_eq!(first.run.usage.cached_prompt_tokens, 0);
    assert_eq!((first.report.cached_tokens, first.report.replayed_rows), (0, 0));
    assert_eq!(first.report.computed_rows, 100);

    let before = e.snapshot();
    let second = e.ask(&prompt, 40);
    assert_eq!(second.run.text, first.run.text, "the same tokens");
    assert_eq!(second.run.text, fresh(&prompt, 40), "and as with no cache at all");
    // 99 cells are looked up (the last token is always computed): 6 pages of
    // 16. The checkpoint at the anchor, the last page boundary before the
    // prompt's end, is at 96: nothing is replayed and the last 4 rows run.
    assert_eq!(second.run.usage.cached_prompt_tokens, 96);
    assert_eq!(second.run.usage.prompt_tokens, 100);
    assert_eq!(
        (second.report.cached_tokens, second.report.replayed_rows, second.report.computed_rows),
        (96, 0, 4)
    );
    let d = e.since(&before);
    assert_eq!(d["tokens"], 40);
    let c = e.snapshot().counters;
    assert_eq!((c.prefix_hits, c.cached_cells, c.checkpoints_restored), (1, 96, 1));
    e.consistent();
}

#[test]
fn without_checkpoints_a_hit_replays_every_cached_row_and_writes_none_of_them() {
    let mut rng = Rng(8);
    let prompt = rng.prompt(100);
    let model = model(Some(CHECKPOINT), |_| {});
    // No budget: nothing to restore, so the 96 cached rows run again.
    let e = engine(&model, config(256, 0));
    let first = e.ask(&prompt, 20);
    let second = e.ask(&prompt, 20);
    assert_eq!(second.run.text, first.run.text);
    assert_eq!(second.run.text, fresh(&prompt, 20));
    assert_eq!(second.run.usage.cached_prompt_tokens, 96);
    assert_eq!(
        (second.report.cached_tokens, second.report.replayed_rows, second.report.computed_rows),
        (96, 96, 4)
    );
    let c = e.snapshot().counters;
    assert_eq!((c.checkpoints_saved, c.checkpoints_restored, c.replay_rows), (0, 0, 96));
    e.consistent();
}

#[test]
fn a_conversation_continued_turn_by_turn_resumes_at_the_tail_checkpoint() {
    let mut rng = Rng(11);
    let model = model(Some(CHECKPOINT), |_| {});
    let e = engine(&model, config(256, 16));
    let mut text = rng.prompt(70);
    let mut cells_left = 0u64;
    for turn in 0..5 {
        let before = e.snapshot();
        let answer = e.ask(&text, 30);
        assert_eq!(answer.run.text, fresh(&text, 30), "turn {turn}");
        let tokens = tokens_of(&text);
        if turn > 0 {
            // The last turn left the cells of its prompt and its answer but
            // the answer's last token, and the live state at their end: this
            // turn takes all of it and replays nothing.
            assert_eq!(u64::from(answer.run.usage.cached_prompt_tokens), cells_left, "turn {turn}");
            assert_eq!(answer.report.replayed_rows, 0, "turn {turn}");
            assert_eq!(u64::from(answer.report.computed_rows), tokens - cells_left, "turn {turn}");
            let d = e.since(&before);
            let copies = e.snapshot().counters.page_copies - before.counters.page_copies;
            // The tail is a partly used page that the turn copies before it writes.
            assert_eq!(
                copies,
                u64::from(!cells_left.is_multiple_of(u64::from(PAGE))),
                "turn {turn} {d:?}"
            );
        }
        cells_left = tokens + 30 - 1;
        text = format!("{text}{}{}", answer.run.text, rng.prompt(12));
    }
    e.consistent();
}

#[test]
fn a_prompt_that_diverges_inside_a_page_resumes_at_the_boundary_before_it() {
    let mut rng = Rng(12);
    let first = rng.prompt(100);
    // The same text up to 50, which is inside the fourth page, then another.
    let other = format!("{}{}", head(&first, 50), rng.prompt(40));
    for budget in [16, 0] {
        let model = model(Some(CHECKPOINT), |_| {});
        let e = engine(&model, config(256, budget));
        e.ask(&first, 30);
        let answer = e.ask(&other, 30);
        assert_eq!(answer.run.text, fresh(&other, 30), "budget {budget}");
        // Three whole pages are the same text. With checkpoints the one at
        // 48 is restored; without, the three pages are replayed.
        assert_eq!(answer.report.cached_tokens, 48, "budget {budget}");
        assert_eq!(answer.report.replayed_rows, if budget > 0 { 0 } else { 48 }, "budget {budget}");
        assert_eq!(answer.report.computed_rows, 90 - 48, "budget {budget}");
        e.consistent();
    }
}

#[test]
fn a_prompt_that_continues_into_the_middle_of_a_tail_takes_the_cells_it_shares() {
    let mut rng = Rng(13);
    let model = model(Some(CHECKPOINT), |_| {});
    let e = engine(&model, config(256, 16));
    // 40 tokens and 25 more: the cells end at 64 and the request leaves a
    // full page; 40 and 30 end at 69, a tail of 5 cells.
    let first = rng.prompt(40);
    let done = e.ask(&first, 30);
    // 69 cells: 4 pages and a tail of 5 cells. A prompt that follows the
    // answer for 3 of them and then goes its own way shares 4 pages and 3.
    let own = format!("{first}{}{}", head(&done.run.text, 64 - 40 + 3), rng.prompt(10));
    let answer = e.ask(&own, 20);
    assert_eq!(answer.run.text, fresh(&own, 20));
    assert_eq!(answer.report.cached_tokens, 67);
    // The tail's checkpoint is the state after 69 cells, not after 67: the
    // checkpoint at 64 is restored and 3 rows replayed.
    assert_eq!((answer.report.replayed_rows, answer.report.computed_rows), (3, 77 - 67));
    e.consistent();
}

#[test]
fn a_model_without_recurrent_state_shares_pages_and_tails_and_replays_nothing() {
    let mut rng = Rng(14);
    let prompt = rng.prompt(100);
    let model = model(Some(0), |_| {});
    let e = engine(&model, config(256, 0));
    let first = e.ask(&prompt, 40);
    let second = e.ask(&prompt, 40);
    assert_eq!(second.run.text, first.run.text);
    assert_eq!(second.run.text, fresh(&prompt, 40));
    assert_eq!(
        (second.report.cached_tokens, second.report.replayed_rows, second.report.computed_rows),
        (96, 0, 4)
    );
    // A conversation's next turn takes the tail as well.
    let next = format!("{prompt}{}more", first.run.text);
    let third = e.ask(&next, 10);
    assert_eq!(third.run.text, fresh(&next, 10));
    assert_eq!(third.report.cached_tokens, 100 + 40 - 1);
    assert_eq!(third.report.replayed_rows, 0);
    let c = e.snapshot().counters;
    assert_eq!((c.checkpoints_saved, c.checkpoints_restored, c.replay_rows), (0, 0, 0));
    e.consistent();
}

#[test]
fn a_model_that_does_not_take_part_has_no_cache() {
    let mut rng = Rng(15);
    let prompt = rng.prompt(100);
    let model = model(None, |_| {});
    let rt = Runtime::start(
        model.clone(),
        Box::new(HostExecutor::new(model.clone(), model.config().max_rows, 16)),
        config(256, 16),
    )
    .unwrap();
    let a = alone(&rt, &prompt, Some(20));
    let b = alone(&rt, &prompt, Some(20));
    assert_eq!(a, b);
    assert_eq!(b.usage.cached_prompt_tokens, 0);
    let s = snapshot(&rt);
    assert_eq!((s.cached_pages, s.pages_free), (0, s.pages_capacity), "nothing stays behind");
    assert_eq!(s.counters.prefix_hits + s.counters.replay_rows + s.counters.page_copies, 0);
}

#[test]
fn two_equal_prompts_submitted_together_share_from_the_first_full_page_and_one_is_held_back() {
    let mut rng = Rng(16);
    let prompt = rng.prompt(100);
    let model = model(Some(CHECKPOINT), |_| {});
    let gate = Gate::new();
    let steps = Arc::new(AtomicU32::new(0));
    let e = engine_with(&model, config(256, 16), |host| {
        let steps = steps.clone();
        host.with_gate(gate.clone()).with_observer(move |_| {
            steps.fetch_add(1, Ordering::SeqCst);
        })
    });
    let expected = fresh_run(|_| {}, &prompt, 30);

    // Keep the conductor inside a step while both are submitted, so that it
    // takes them in one turn and starts them in one pass.
    gate.close();
    let busy = start(&e.rt, request("busy", Some(1))).unwrap();
    dispatched(&steps, 1);
    let (a, b) = (request(&prompt, Some(30)), request(&prompt, Some(30)));
    let (a_id, b_id) = (a.id, b.id);
    let (a, b) = (start(&e.rt, a).unwrap(), start(&e.rt, b).unwrap());
    gate.open();
    busy.finish();
    let (a, b) = (a.finish(), b.finish());
    assert_eq!(
        (a.text.as_str(), b.text.as_str()),
        (expected.text.as_str(), expected.text.as_str())
    );

    let s = e.snapshot();
    let (first, second) = (s.request(a_id).unwrap(), s.request(b_id).unwrap());
    // One of them was held back until the other had published the pages of
    // their prompt; it took all 6 and the anchor's checkpoint.
    assert_eq!(s.counters.requests_held, 1, "{s}");
    assert_eq!((first.cached_tokens, first.computed_rows), (0, 100));
    assert_eq!((second.cached_tokens, second.replayed_rows, second.computed_rows), (96, 0, 4));
    // The rows of the prompt that ran: "busy", the prompt, and the last 4 rows.
    assert_eq!(s.counters.prefill_rows, 4 + 100 + 4);
    e.consistent();
}

#[test]
fn a_prompt_that_arrives_while_another_computes_its_pages_waits_for_them() {
    let mut rng = Rng(17);
    let prompt = rng.prompt(300);
    let model = model(Some(CHECKPOINT), |_| {});
    let e =
        engine_with(&model, config(256, 32), |host| host.with_delay(Duration::from_micros(300)));
    let expected = fresh_run(|_| {}, &prompt, 20);
    let first = start(&e.rt, request(&prompt, Some(20))).unwrap();
    // The first prompt takes 19 steps of one page each.
    std::thread::sleep(Duration::from_millis(1));
    let later = request(&prompt, Some(20));
    let later_id = later.id;
    let later = start(&e.rt, later).unwrap();
    assert_eq!(first.finish().text, expected.text);
    assert_eq!(later.finish().text, expected.text);
    let s = e.snapshot();
    let report = s.request(later_id).unwrap();
    // Whenever it came, it computed no row of the pages that the first had
    // computed or was computing.
    assert!(report.cached_tokens >= 16, "{report:?}");
    assert_eq!(
        u64::from(report.computed_rows)
            + u64::from(report.replayed_rows)
            + u64::from(report.cached_tokens),
        300 + u64::from(report.replayed_rows)
    );
    assert!(s.counters.prefill_rows < 300 + 300 - 16 + 1, "{s}");
    e.consistent();
}

#[test]
fn a_request_held_for_a_page_that_is_given_up_computes_it_itself() {
    let mut rng = Rng(18);
    let prompt = rng.prompt(300);
    let model = model(Some(CHECKPOINT), |_| {});
    let gate = Gate::new();
    // The conductor stops in the second step, which is the first of the
    // request that computes the prompt. The gate is closed before the step is
    // counted, so whoever sees the count finds it closed.
    let (steps, stop) = (Arc::new(AtomicU32::new(0)), gate.clone());
    let e = engine_with(&model, config(256, 32), |host| {
        let steps = steps.clone();
        host.with_gate(gate.clone()).with_observer(move |_| {
            if steps.load(Ordering::SeqCst) == 1 {
                stop.close();
            }
            steps.fetch_add(1, Ordering::SeqCst);
        })
    });
    let expected = fresh_run(|_| {}, &prompt, 20);
    gate.close();
    let busy = start(&e.rt, request("busy", Some(1))).unwrap();
    // The conductor is inside the step of "busy": the two prompts arrive
    // together, after it.
    dispatched(&steps, 1);
    let (owner, held) = (request(&prompt, Some(20)), request(&prompt, Some(20)));
    let (owner_id, held_id) = (owner.id, held.id);
    let (owner, held) = (start(&e.rt, owner).unwrap(), start(&e.rt, held).unwrap());
    gate.open();
    busy.finish();
    // The conductor is inside the second step, held at the gate: the first
    // request has read one page when it is cancelled, and the other has been
    // waiting for the second.
    dispatched(&steps, 2);
    e.rt.cancel(owner_id);
    gate.open();
    assert_eq!(owner.finish().finish, ares_api::FinishReason::Cancelled);
    assert_eq!(held.finish().text, expected.text);
    let s = e.snapshot();
    assert_eq!(s.counters.requests_held, 1);
    let report = s.request(held_id).unwrap();
    assert_eq!((report.cached_tokens, report.replayed_rows, report.computed_rows), (16, 0, 284));
    e.consistent();
}

#[test]
fn rejected_drafts_inside_a_tail_are_cut_and_the_next_turn_takes_exactly_what_was_accepted() {
    let mut rng = Rng(19);
    let model = model(Some(CHECKPOINT), repeating);
    let cfg = RuntimeConfig { window_rows: 8, ..config(512, 64) };
    let e = engine_with(&model, cfg, |host| host.with_delay(Duration::from_micros(100)));
    let mut ended_in_a_window = 0;
    for max_tokens in 20..70u32 {
        let len = 20 + rng.below(40) as usize;
        let first = rng.prompt(len);
        let before = e.snapshot();
        let said = e.ask(&first, max_tokens);
        assert_eq!(said.run.text, fresh_run(repeating, &first, max_tokens).text, "{max_tokens}");
        let d = e.since(&before);
        // The request ended with its last window: every cell but the last
        // token's, nothing of the drafts that were rejected.
        let cells = tokens_of(&first) + u64::from(max_tokens) - 1;
        ended_in_a_window += u64::from(d["accepted"] + d["rejected"] > 0);

        // The next turn continues the text and takes all those cells.
        let next = format!("{first}{}and on", said.run.text);
        let answer = e.ask(&next, 12);
        assert_eq!(answer.run.text, fresh_run(repeating, &next, 12).text, "{max_tokens}");
        assert_eq!(u64::from(answer.run.usage.cached_prompt_tokens), cells, "{max_tokens}");
        assert_eq!(answer.report.replayed_rows, 0, "{max_tokens}");
    }
    let c = e.snapshot().counters;
    assert!(c.drafts_accepted > 100 && c.drafts_rejected > 20, "{c:?}");
    assert!(ended_in_a_window > 10);

    // A stop string ends a request wherever it comes up, also in the middle
    // of the drafts that were accepted: what the request keeps ends there.
    for at in 8..50 {
        let len = 30 + rng.below(30) as usize;
        let first = rng.prompt(len);
        let long = fresh_run(repeating, &first, 80).text;
        let stop: String = long.chars().skip(at).take(2).collect();
        let mut stopped = request(&first, Some(80));
        stopped.stop = vec![stop.clone()];
        let said = run_one(&e.rt, stopped);
        assert_eq!(said.finish, ares_api::FinishReason::StopString(0), "{at}");
        let produced = u64::from(said.usage.completion_tokens);
        // The tokens of the stop string were generated and are cells too, but
        // the last token of the request has none.
        let cells = tokens_of(&first) + produced - 1;
        let written = head(&long, produced as usize);
        let next = format!("{first}{written}and on");
        let answer = e.ask(&next, 12);
        assert_eq!(answer.run.text, fresh_run(repeating, &next, 12).text, "{at}");
        assert_eq!(u64::from(answer.run.usage.cached_prompt_tokens), cells, "stop at {at}");
    }
    e.consistent();
}

#[test]
fn eviction_under_page_pressure_and_a_small_budget_of_ram_keeps_every_answer_equal() {
    let mut rng = Rng(21);
    let model = model(Some(CHECKPOINT), |c| c.context_length = 256);
    // 30 pages of 16 cells, and room for 3 checkpoints.
    let e = engine(&model, config(30, 3));
    // 12 prompts of 90 tokens and 20 more: 7 pages each, more than the pool.
    let prompts: Vec<String> = (0..12).map(|_| rng.prompt(90)).collect();
    let expected: Vec<Run> =
        prompts.iter().map(|p| fresh_run(|c| c.context_length = 256, p, 20)).collect();
    for round in 0..3 {
        for (i, prompt) in prompts.iter().enumerate() {
            let answer = e.ask(prompt, 20);
            assert_eq!(answer.run.text, expected[i].text, "round {round}, prompt {i}");
        }
    }
    // The prompt just asked is still there to be asked again.
    let again = e.ask(&prompts[11], 20);
    assert_eq!(again.run.text, expected[11].text);
    assert!(again.report.cached_tokens >= 80, "{:?}", again.report);

    let s = e.snapshot();
    let c = &s.counters;
    assert!(c.cache_pages_evicted > 20 && c.checkpoints_dropped > 5, "{s}");
    assert!(c.checkpoints_saved > 20 && c.checkpoints_restored > 0, "{s}");
    assert_eq!(s.pages_free + s.cached_pages, s.pages_capacity, "no page is lost");
    e.consistent();
}

#[test]
fn a_request_that_gives_way_goes_into_the_cache_and_comes_back_as_a_hit() {
    // 24 pages of 16 cells; a request of 26 prompt tokens and 180 of output
    // ends with 13 pages. The accounting is told to expect a twentieth of the
    // output, so it lets four in where two fit.
    for (name, speculation, shape) in [
        ("plain", false, (|_: &mut SyntheticConfig| {}) as Shape),
        ("repeating", true, repeating as Shape),
    ] {
        let shape_of = |c: &mut SyntheticConfig| {
            c.context_length = 256;
            shape(c);
        };
        let model = model(Some(CHECKPOINT), shape_of);
        let cfg = RuntimeConfig {
            output_ratio: 0.05,
            window_rows: window(speculation),
            ..config(24, 16)
        };
        let e = engine_with(&model, cfg, |host| host.with_delay(Duration::from_micros(100)));
        let jobs: Vec<String> = (0..4).map(|i| format!("writes more than expected {i}")).collect();
        let expected: Vec<Run> = jobs.iter().map(|p| fresh_run(shape_of, p, 180)).collect();
        // Requests that end at once teach the accounting to expect little.
        for i in 0..60 {
            let mut short = request(&format!("ends early {i}"), Some(200));
            short.stop = (0..=255u8).map(|b| ares_runtime::printable(b).to_string()).collect();
            run_one(&e.rt, short);
        }
        let before = e.snapshot();
        let handles: Vec<Handle> =
            jobs.iter().map(|p| start(&e.rt, request(p, Some(180))).unwrap()).collect();
        let ids: Vec<_> = handles.iter().map(|h| h.id).collect();
        for (i, handle) in handles.into_iter().enumerate() {
            let run = handle.finish();
            assert_eq!(run.text, expected[i].text, "{name}: request {i}");
            assert_eq!(run.usage.completion_tokens, 180);
        }
        let s = e.snapshot();
        let gave_way = s.counters.requests_suspended - before.counters.requests_suspended;
        assert!(gave_way >= 1, "{name}: the pool never ran short: {s}");
        // Each that gave way came back as a hit, not as a prompt of everything
        // it had written: it computed far fewer rows than it had, which
        // would be 26 + up to 180 for each time it came back.
        let hits = s.counters.prefix_hits - before.counters.prefix_hits;
        assert!(hits >= gave_way, "{name}: {hits} hits for {gave_way} that gave way");
        let computed: u32 = ids.iter().map(|id| s.request(*id).unwrap().computed_rows).sum();
        let rows_without_the_cache: u32 = ids.iter().map(|_| 26).sum::<u32>() + 180 * 4;
        assert!(
            computed < rows_without_the_cache / 2 + 4 * 26,
            "{name}: {computed} rows of the prompt and of what came back"
        );
        assert_eq!(s.pages_free + s.cached_pages, s.pages_capacity, "{name}");
        assert_eq!((s.active, s.waiting), (0, 0));
        e.consistent();
    }
}

#[test]
fn a_window_ends_where_a_checkpoint_is_due_whatever_it_drafts() {
    let mut rng = Rng(22);
    let model = model(Some(CHECKPOINT), repeating);
    for speculation in [false, true] {
        let e = engine_with(
            &model,
            RuntimeConfig { window_rows: window(speculation), ..config(256, 24) },
            |host| host.with_delay(Duration::from_micros(100)),
        );
        let prompt = rng.prompt(20);
        let before = e.snapshot();
        let said = e.ask(&prompt, 200);
        assert_eq!(said.run.text, fresh_run(repeating, &prompt, 200).text);
        let d = e.snapshot().counters;
        if speculation {
            assert!(d.drafts_accepted - before.counters.drafts_accepted > 50, "{d:?}");
        }
        // The anchor at 16, then every 32 cells from there: 48, 80, 112, 144,
        // 176 and 208 (the cells end at 219), and the end of the request.
        assert_eq!(
            d.checkpoints_saved - before.counters.checkpoints_saved,
            8,
            "speculation {speculation}: {d:?}"
        );
        e.consistent();
    }
}

/// What a record held, for the tests that look at what a device would be told.
#[derive(Clone, Debug)]
struct Seen {
    /// Slot, first position, rows, replay, direct.
    segments: Vec<(u32, u32, u32, bool, bool)>,
    states: Vec<(StateCopyKind, u32, u32)>,
    pages: Vec<(u32, u32, u32)>,
}

fn watched() -> (Arc<Mutex<Vec<Seen>>>, impl FnOnce(HostExecutor) -> HostExecutor) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    (seen, move |host: HostExecutor| {
        host.with_observer(move |record| {
            let segments = (0..record.segments())
                .map(|i| {
                    let s = record.segment(i);
                    (s.slot, s.pos0 as u32, s.n_rows, record.replay(i), s.direct)
                })
                .collect();
            let states = (0..record.state_copies())
                .map(|i| {
                    let c = record.state_copy(i);
                    (c.kind, c.slot, c.place)
                })
                .collect();
            let pages = (0..record.page_copies())
                .map(|i| {
                    let c = record.page_copy(i);
                    (c.from, c.to, c.cells)
                })
                .collect();
            log.lock().unwrap().push(Seen { segments, states, pages });
        })
    })
}

#[test]
fn a_prefill_step_ends_where_a_checkpoint_is_due_and_it_is_saved_with_the_next_record() {
    let mut rng = Rng(23);
    let prompt = rng.prompt(100);
    let model = model(Some(CHECKPOINT), |_| {});
    let (seen, watch) = watched();
    let e = engine_with(&model, config(256, 16), watch);
    e.ask(&prompt, 5);
    let cold: Vec<Seen> = std::mem::take(&mut *seen.lock().unwrap());
    // One step per page: the prompt's rows end at every boundary, which is
    // where a checkpoint is due, and then at the prompt's end.
    let ends: Vec<u32> = cold
        .iter()
        .flat_map(|r| r.segments.iter())
        .filter(|s| s.2 > 1 || s.1 < 100)
        .map(|s| s.1 + s.2)
        .collect();
    assert_eq!(ends, [16, 32, 48, 64, 80, 96, 100]);
    // A checkpoint taken at the commit of a step is saved with the next
    // record: the first record has none, each of the next 6 has one.
    let saves: Vec<usize> = cold
        .iter()
        .map(|r| r.states.iter().filter(|c| c.0 == StateCopyKind::Save).count())
        .collect();
    assert_eq!(saves[0], 0);
    assert!(saves[1..7].iter().all(|n| *n == 1), "{saves:?}");
    assert!(cold.iter().all(|r| r.segments.iter().all(|s| !s.3)), "nothing is replayed cold");
    let places: Vec<u32> = cold.iter().flat_map(|r| r.states.iter().map(|c| c.2)).collect();
    assert_eq!(places.len(), places.iter().collect::<std::collections::HashSet<_>>().len());

    // The same prompt again restores the checkpoint at its anchor, before its
    // first row. The first request left a tail of 8 cells after page 6, whose
    // first 3 cells are the prompt's last: they are cached, and replayed from
    // the anchor, and the last row runs.
    e.ask(&prompt, 5);
    let warm: Vec<Seen> = std::mem::take(&mut *seen.lock().unwrap());
    let first = &warm[0];
    let anchor = places[5];
    let restores: Vec<_> = first.states.iter().filter(|c| c.0 == StateCopyKind::Restore).collect();
    assert_eq!(restores.len(), 1);
    assert_eq!(restores[0].2, anchor, "the checkpoint taken at 96");
    assert_eq!((first.segments[0].1, first.segments[0].2, first.segments[0].3), (96, 3, true));
    let next = &warm[1];
    assert_eq!((next.segments[0].1, next.segments[0].2, next.segments[0].3), (99, 1, false));
    // The records carry the saves (of the last request's end) before the
    // restores, as the device runs them.
    let kinds: Vec<StateCopyKind> = first.states.iter().map(|c| c.0).collect();
    assert!(
        kinds.windows(2).all(|w| w[0] == StateCopyKind::Save || w[1] == StateCopyKind::Restore)
    );
    e.consistent();
}

#[test]
fn rows_that_rebuild_state_are_marked_replay_and_the_copy_of_a_shared_page_is_in_the_record() {
    let mut rng = Rng(24);
    let model = model(Some(CHECKPOINT), |_| {});
    let (seen, watch) = watched();
    // No checkpoints: a hit replays from the start.
    let e = engine_with(&model, config(256, 0), watch);
    let first = rng.prompt(40);
    let said = e.ask(&first, 30);
    seen.lock().unwrap().clear();
    // 69 cells were left: 4 pages and a tail of 5 cells.
    let next = format!("{first}{}more", said.run.text);
    e.ask(&next, 5);
    let records = std::mem::take(&mut *seen.lock().unwrap());
    let replay: Vec<(u32, u32)> = records
        .iter()
        .flat_map(|r| r.segments.iter())
        .filter(|s| s.3)
        .map(|s| (s.1, s.1 + s.2))
        .collect();
    // 64 rows fit a step: 0..64, then 64..69.
    assert_eq!(replay, [(0, 64), (64, 69)]);
    assert!(records.iter().flat_map(|r| r.segments.iter()).filter(|s| s.3).all(|s| s.4), "direct");
    // The rows that write follow the last replayed one and start at 69.
    let writing: Vec<u32> =
        records.iter().flat_map(|r| r.segments.iter()).filter(|s| !s.3).map(|s| s.1).collect();
    assert_eq!(writing.first(), Some(&69));
    // The first write goes into a copy of the tail's page: 5 cells.
    let copies: Vec<(u32, u32, u32)> = records.iter().flat_map(|r| r.pages.clone()).collect();
    assert_eq!(copies.len(), 1, "{copies:?}");
    assert_eq!(copies[0].2, 5);
    assert_ne!(copies[0].0, copies[0].1);
    e.consistent();
}

/// A request and what an engine with no cache says to it.
struct Job {
    prompt: String,
    max_tokens: u32,
    expected: Run,
}

/// Requests that start with each other's text: slices of a few documents with
/// edits, conversations that go on with the answers, the same prompt again,
/// and prompts of their own. What each says is taken from an engine that has
/// no cache, one request at a time.
fn jobs(shape: Shape, rng: &mut Rng, n: usize, longest: u64) -> Vec<Job> {
    let model = model(None, shape);
    let reference = Runtime::start(
        model.clone(),
        Box::new(HostExecutor::new(model.clone(), model.config().max_rows, 16)),
        RuntimeConfig { window_rows: 1, warm_up: false, ..config(4096, 0) },
    )
    .unwrap();
    let documents: Vec<String> = (0..4)
        .map(|_| {
            let len = 150 + rng.below(70) as usize;
            rng.prompt(len)
        })
        .collect();
    let mut jobs: Vec<Job> = Vec::new();
    while jobs.len() < n {
        let kind = rng.below(5);
        let prompt = match kind {
            0 => {
                let len = 20 + rng.below(100) as usize;
                rng.prompt(len)
            }
            1 | 2 => {
                // A slice of a document, edited in a few places, and more.
                let mut chars: Vec<char> = documents[rng.below(4) as usize].chars().collect();
                chars.truncate(10 + rng.below(chars.len() as u64 - 10) as usize);
                if kind == 2 {
                    for _ in 0..1 + rng.below(3) {
                        let at = rng.below(chars.len() as u64) as usize;
                        chars[at] = char::from(b' ' + rng.below(95) as u8);
                    }
                }
                let more = rng.below(20) as usize;
                format!("{}{}", chars.into_iter().collect::<String>(), rng.prompt(more))
            }
            3 if !jobs.is_empty() => {
                // The conversation goes on after an answer.
                let last = &jobs[rng.below(jobs.len() as u64) as usize];
                let more = 5 + rng.below(15) as usize;
                format!("{}{}{}", last.prompt, last.expected.text, rng.prompt(more))
            }
            _ if !jobs.is_empty() => jobs[rng.below(jobs.len() as u64) as usize].prompt.clone(),
            _ => rng.prompt(40),
        };
        if prompt.chars().count() > 300 {
            continue;
        }
        let max_tokens = 1 + rng.below(longest) as u32;
        let expected = alone(&reference, &prompt, Some(max_tokens));
        jobs.push(Job { prompt, max_tokens, expected });
    }
    jobs
}

#[test]
fn many_requests_with_shared_prefixes_edits_and_tight_memory_all_say_what_they_say_alone() {
    // Each of the configurations runs the same requests twice, in parallel,
    // arriving at random times or all at once: the second time nearly
    // everything is cached.
    let shapes: [(&str, Shape); 2] = [("plain", |_| {}), ("repeating", repeating)];
    let mut gave_way = 0;
    for (shape_name, shape) in shapes {
        // A seed with which requests give way in the long runs, every time.
        let mut rng = Rng(23_770);
        let short = jobs(shape, &mut rng, 48, 60);
        let long = jobs(shape, &mut rng, 24, 150);
        // Name, pages, checkpoint places, slots, the output the accounting
        // expects, whether the requests arrive all at once (the conductor is
        // held in a step while they are sent, so it takes them in one turn),
        // and the requests.
        for (name, pages, checkpoints, slots, ratio, burst, jobs) in [
            ("roomy", 1024, 64, 16, 0.7, false, &short),
            ("tight, 3 checkpoints", 64, 3, 6, 0.1, false, &short),
            ("tight, no checkpoints", 48, 0, 4, 0.1, false, &short),
            ("tight, 2 checkpoints", 64, 2, 8, 0.1, false, &short),
            // Prompts are read in pieces at full speed, so fewer are half read
            // at once than when they trickled in: it takes more slots than
            // the pool can feed for requests to give way.
            ("long answers, 2 checkpoints", 36, 2, 16, 0.02, true, &long),
            ("long answers, 5 checkpoints", 36, 5, 16, 0.02, true, &long),
        ] {
            let model = model(Some(CHECKPOINT), shape);
            let cfg = RuntimeConfig {
                window_rows: 8,
                max_slots: slots,
                output_ratio: ratio,
                ..config(pages, checkpoints)
            };
            let gate = Gate::new();
            let steps = Arc::new(AtomicU32::new(0));
            let e = engine_with(&model, cfg, |host| {
                let steps = steps.clone();
                host.with_delay(Duration::from_micros(80)).with_gate(gate.clone()).with_observer(
                    move |_| {
                        steps.fetch_add(1, Ordering::SeqCst);
                    },
                )
            });
            for round in 0..2 {
                let mut arrival = Rng(77 + round);
                let busy = burst.then(|| {
                    // Nothing runs, so the next step is the one of "busy", and
                    // the conductor is inside it when the requests are sent.
                    let idle = steps.load(Ordering::SeqCst);
                    gate.close();
                    let busy = start(&e.rt, request("busy", Some(1))).unwrap();
                    dispatched(&steps, idle + 1);
                    busy
                });
                let handles: Vec<Handle> = jobs
                    .iter()
                    .map(|job| {
                        if !burst {
                            std::thread::sleep(Duration::from_micros(arrival.below(300)));
                        }
                        start(&e.rt, request(&job.prompt, Some(job.max_tokens))).unwrap()
                    })
                    .collect();
                gate.open();
                busy.map(Handle::finish);
                for (i, handle) in handles.into_iter().enumerate() {
                    let run = handle.finish();
                    let want = &jobs[i].expected;
                    let label = format!("{shape_name}, {name}, round {round}, request {i}");
                    assert_eq!(run.text, want.text, "{label}");
                    assert_eq!(run.finish, want.finish, "{label}");
                    assert_eq!(
                        run.usage.completion_tokens, want.usage.completion_tokens,
                        "{label}"
                    );
                    assert_eq!(run.usage.prompt_tokens, want.usage.prompt_tokens, "{label}");
                }
            }
            let s = e.snapshot();
            assert_eq!((s.active, s.waiting), (0, 0));
            assert_eq!(s.pages_free + s.cached_pages, s.pages_capacity, "{shape_name}, {name}");
            assert!(s.counters.prefix_hits >= 10, "{shape_name}, {name}: {s}");
            if pages <= 64 {
                assert!(s.counters.cache_pages_evicted > 0, "{shape_name}, {name}: {s}");
            }
            gave_way += u32::from(s.counters.requests_suspended > 0);
            e.consistent();
        }
    }
    assert!(gave_way >= 3, "requests gave way in {gave_way} of the runs");
}

#[test]
fn the_scenarios_of_an_engine_without_a_cache_run_clean_under_the_oracle() {
    // The oracle checks pages, tables and cut drafts of the ordinary path as
    // well: a model without a cache is a model without recurrent state to it.
    for speculation in [false, true] {
        let model = model(None, repeating);
        let cfg =
            RuntimeConfig { window_rows: window(speculation), max_slots: 64, ..config(4096, 0) };
        let e = engine_with(&model, cfg, |host| host.with_delay(Duration::from_micros(300)));
        output_is_independent_of_the_batch(&e.rt, Duration::from_micros(400));
        e.consistent();
    }
    let small = model(None, |c| c.context_length = 256);
    let cfg = RuntimeConfig { window_rows: 8, max_slots: 64, ..config(4096, 0) };
    let e = engine_with(&small, cfg, |host| host.with_delay(Duration::from_micros(500)));
    cancel_frees_pages_and_spares_the_others(&e.rt);
    e.consistent();

    let looping = model(None, |c| {
        c.context_length = 256;
        repeating(c);
    });
    let cfg = RuntimeConfig { output_ratio: 0.05, window_rows: 8, ..config(24, 0) };
    let e = engine_with(&looping, cfg, |host| host.with_delay(Duration::from_micros(100)));
    a_request_that_gives_way_comes_back_and_says_the_same(&e.rt);
    e.consistent();
}

/// The pages the records name as freed, and the pages their updates table, as
/// a device that keeps something per page would see them.
#[derive(Default)]
struct Pages {
    freed: Vec<u32>,
    tabled: std::collections::HashSet<u32>,
    /// The pages tabled after the first batch of requests.
    later: std::collections::HashSet<u32>,
    after_first: bool,
}

#[test]
fn every_page_a_request_used_is_named_as_freed_once_it_ended() {
    let mut rng = Rng(77);
    // No prefix cache: a request's pages are free when it ends.
    let model = model(None, |_| {});
    let seen = Arc::new(Mutex::new(Pages::default()));
    let log = seen.clone();
    let e = engine_with(&model, config(256, 0), move |host| {
        host.with_observer(move |record| {
            let mut pages = log.lock().unwrap();
            for (_, entries) in record.table_updates() {
                for page in entries.map(|k| record.entry(k)) {
                    pages.tabled.insert(page);
                    if pages.after_first {
                        pages.later.insert(page);
                    }
                }
            }
            for i in 0..record.freed() {
                let page = record.freed_page(i);
                pages.freed.push(page);
            }
        })
    });
    let mut first_batch = std::collections::HashSet::new();
    for n in 0..6 {
        let prompt = rng.prompt(20 + 25 * n);
        e.ask(&prompt, 12);
        if n == 4 {
            let mut pages = seen.lock().unwrap();
            first_batch = pages.tabled.clone();
            pages.after_first = true;
        }
    }
    // One more request, so that a record follows the last of the others.
    e.ask("again", 2);
    let pages = seen.lock().unwrap();
    assert!(!pages.freed.is_empty());
    // The pool hands out its lowest free page first, so a later request may hold a page
    // again before a record names it as freed: whatever a device kept for it is then the new
    // holder's.
    for page in &first_batch {
        assert!(
            pages.freed.contains(page) || pages.later.contains(page),
            "page {page} was used, never named as freed and never used again"
        );
    }
    assert!(pages.freed.iter().all(|p| pages.tabled.contains(p)), "only pages that were used");
    e.consistent();
}

/// A conversation as a request: the synthetic model writes each message as
/// `role: text` and a line break, and the reply after `assistant: `. The
/// parts of a conversation end where the text of a message starts and ends;
/// the system prompt's and the first message's are the roots.
fn chat(messages: &[(ares_api::Role, String)], max_tokens: u32) -> ares_api::GenerateRequest {
    let mut request = request("", Some(max_tokens));
    request.prompt = ares_api::Prompt::Chat {
        messages: messages
            .iter()
            .map(|(role, text)| ares_api::Message::new(*role, text.as_str()))
            .collect(),
        tools: None,
        variables: Vec::new(),
        continue_final: false,
    };
    request
}

/// The text the synthetic model renders a conversation as, for a request of
/// plain text that has the same tokens and no boundaries.
fn rendered(messages: &[(ares_api::Role, String)]) -> String {
    let mut text: String =
        messages.iter().map(|(role, content)| format!("{}: {content}\n", role.as_str())).collect();
    text.push_str("assistant: ");
    text
}

impl Rig {
    fn ask_chat(&self, messages: &[(ares_api::Role, String)], max_tokens: u32) -> Answer {
        let request = chat(messages, max_tokens);
        let id = request.id;
        let run = run_one(&self.rt, request);
        let report = *snapshot(&self.rt).request(id).expect("the request ended");
        Answer { run, report }
    }
}

/// Checkpoints at message boundaries only: no fixed distance in a prompt or
/// while generating, a boundary one page past the last checkpoint gets one.
fn boundary_config() -> RuntimeConfig {
    RuntimeConfig {
        checkpoint_prompt_interval: 4096,
        checkpoint_generation_interval: 4096,
        checkpoint_boundary_interval: PAGE,
        ..config(256, 64)
    }
}

#[test]
fn a_new_conversation_with_the_same_system_prompt_starts_from_its_root() {
    use ares_api::Role::{System, User};
    let mut rng = Rng(31);
    let system = rng.prompt(100);
    let first = [(System, system.clone()), (User, rng.prompt(30))];
    let second = [(System, system.clone()), (User, rng.prompt(30))];
    let model = model(Some(CHECKPOINT), |_| {});
    // The system message is `system: ` and 100 characters: it ends at 108,
    // and the first message's text starts at 115 after `\nuser: `. The roots
    // are the page boundaries at 96 and 112.
    let e = engine(&model, boundary_config());
    let a = e.ask_chat(&first, 20);
    assert_eq!(a.report.replayed_rows, 0);
    let b = e.ask_chat(&second, 20);
    assert_eq!(b.run.text, fresh(&rendered(&second), 20));
    assert_eq!((b.report.cached_tokens, b.report.replayed_rows), (112, 0));
    e.consistent();
    // The same texts without boundaries (plain text of the same tokens):
    // nothing is kept at the system prompt's end, every shared row runs again.
    let e = engine(&model, boundary_config());
    e.ask(&rendered(&first), 20);
    let b = e.ask(&rendered(&second), 20);
    assert_eq!((b.report.cached_tokens, b.report.replayed_rows), (112, 112));
    e.consistent();
}

#[test]
fn an_edited_message_replays_only_the_rows_from_the_boundary_before_it() {
    use ares_api::Role::{Assistant, System, User};
    let mut rng = Rng(32);
    let model = model(Some(CHECKPOINT), |_| {});
    // A conversation of five turns that comes whole (a client that sends a
    // history, a conversation the cache had lost), then again with its
    // second answer changed.
    let mut messages = vec![(System, rng.prompt(40))];
    for turn in 0..5 {
        messages.push((User, rng.prompt(25)));
        if turn < 4 {
            messages.push((Assistant, rng.prompt(30)));
        }
    }
    let mut edited = messages.clone();
    edited[4].1 = rng.prompt(30);
    // The text of the changed answer starts here.
    let shared = (rendered(&messages[..4]).len() as u32) / PAGE * PAGE;

    let e = engine(&model, boundary_config());
    let first = e.ask_chat(&messages, 20);
    assert_eq!(first.run.text, fresh(&rendered(&messages), 20));
    let answer = e.ask_chat(&edited, 20);
    assert_eq!(answer.run.text, fresh(&rendered(&edited), 20));
    // It takes the pages before the change and replays from a checkpoint
    // less than a piece of prompt (64 rows) before it.
    assert_eq!(answer.report.cached_tokens, shared);
    assert!(answer.report.replayed_rows < 64, "{:?}", answer.report);
    e.consistent();

    // Without boundaries (the same tokens as plain text) the history has a
    // checkpoint only at its end: the change replays every shared row.
    let e = engine(&model, boundary_config());
    e.ask(&rendered(&messages), 20);
    let answer = e.ask(&rendered(&edited), 20);
    assert_eq!((answer.report.cached_tokens, answer.report.replayed_rows), (shared, shared));
    e.consistent();
}

#[test]
fn the_same_prompt_asked_again_resumes_at_its_anchor_while_the_roots_keep_their_places() {
    use ares_api::Role::{Assistant, System, Tool, User};
    // The agent suite of the Strata comparison: a system prompt, a task, a
    // tool call and a long result, asked twice, then a question about the
    // answer. The roots of the system prompt (the page boundaries at 64 and
    // 80) keep their places; the conversation has the place left over, which
    // held the end state of the first ask instead of its anchor, so the
    // second ask replayed every row after the roots.
    for places in [2, 3] {
        let mut rng = Rng(77);
        let long = |c: &mut SyntheticConfig| c.context_length = 4096;
        let model = model(Some(CHECKPOINT), long);
        let config = RuntimeConfig {
            checkpoint_prompt_interval: 1024,
            checkpoint_generation_interval: 256,
            checkpoint_boundary_interval: 128,
            ..config(1024, places)
        };
        let e = engine(&model, config);
        let mut messages = vec![(System, rng.prompt(70)), (User, rng.prompt(20))];
        let first = e.ask_chat(&messages, 12);
        messages.push((Assistant, first.run.text.clone()));
        messages.push((Tool, rng.prompt(600)));
        let once = e.ask_chat(&messages, 12);
        let again = e.ask_chat(&messages, 12);
        assert_eq!(again.run.text, once.run.text, "{places} places");
        assert_eq!(again.run.text, fresh_run(long, &rendered(&messages), 12).text);
        // 747 cells are looked up: 46 pages of 16, and the anchor at 736.
        let report = &again.report;
        assert_eq!((report.cached_tokens, report.replayed_rows), (736, 0), "{places} places");
        // The next turn continues the second ask's reply from its end state.
        messages.push((Assistant, again.run.text.clone()));
        messages.push((User, rng.prompt(20)));
        let next = e.ask_chat(&messages, 12);
        assert_eq!(next.run.text, fresh_run(long, &rendered(&messages), 12).text);
        assert_eq!(next.report.replayed_rows, 0, "{places} places: {:?}", next.report);
        e.consistent();
    }
}

/// The runtime's settings for the tests of paused steps: checkpoints of a prompt far apart,
/// so that a prompt read alone takes steps wider than the pieces of one read beside others.
fn paused_config() -> RuntimeConfig {
    RuntimeConfig { checkpoint_prompt_interval: 4 * PAGE * 4, ..config(256, 16) }
}

/// A model that reads a prompt alone in pieces may pause the step between two of them
/// (a long prompt read layer by layer, `StepOutput::paused`): the conductor runs the same
/// rows again, alone, until they are done; the answer and the cache are as without pauses.
#[test]
fn a_paused_step_runs_again_until_its_rows_are_done() {
    let mut rng = Rng(11);
    let prompt = rng.prompt(200);
    let shape = |c: &mut SyntheticConfig| c.own_rows = 16;
    let model = model(Some(CHECKPOINT), shape);
    let e = engine_with(&model, paused_config(), |host| host.with_pauses(16, 4));
    let first = e.ask(&prompt, 40);
    assert_eq!(first.run.text, fresh_run(shape, &prompt, 40).text, "as without pauses");
    let paused = e.snapshot().counters.steps_paused;
    assert!(paused >= 3, "{paused} steps paused");
    let second = e.ask(&prompt, 40);
    assert_eq!(second.run.text, first.run.text, "from the cache, as the first time");
    e.consistent();
}

/// A request that comes while a step of another prompt is paused runs before that step
/// is done: the paused step waits for its turn, the new request's prompt is read in pieces
/// of its own, and both answer as they do alone.
#[test]
fn a_request_runs_between_the_pieces_of_a_paused_step() {
    let mut rng = Rng(12);
    let long = rng.prompt(240);
    let short = rng.prompt(10);
    let shape = |c: &mut SyntheticConfig| c.own_rows = 16;
    let model = model(Some(CHECKPOINT), shape);
    let pieces = 40;
    // A floor of generation that the rounds cannot reach: the short request gets half of
    // the time, a round for each piece of the long prompt.
    let config = RuntimeConfig { generation_floor: 1e6, ..paused_config() };
    let e = engine_with(&model, config, |host| {
        host.with_pauses(16, pieces).with_delay(Duration::from_millis(2))
    });
    let long_run = start(&e.rt, request(&long, Some(20))).unwrap();
    let deadline = Instant::now() + LIMIT;
    while e.snapshot().counters.steps_paused < 1 {
        assert!(Instant::now() < deadline, "the step never paused");
        std::thread::sleep(Duration::from_micros(200));
    }
    let at_start = e.snapshot().counters.steps_paused;
    let short_run = start(&e.rt, request(&short, Some(5))).unwrap();
    let short_done = short_run.finish();
    // A step of the long prompt pauses `pieces - 1` times before its rows are done; the
    // short request ran between its pieces, a round of its own for a piece of the long one.
    let paused = e.snapshot().counters.steps_paused - at_start;
    let long_done = long_run.finish();
    assert!(paused < u64::from(pieces) / 2, "{paused} pauses while the short request ran");
    assert_eq!(short_done.text, fresh_run(shape, &short, 5).text);
    assert_eq!(long_done.text, fresh_run(shape, &long, 20).text);
    e.consistent();
}

/// A model that pauses its wide steps reads a long prompt that comes while another
/// sequence generates in such steps too: a step of the prompt's rows alone, as wide as a
/// prompt read alone, whose pieces the generating sequence's rounds run between; both
/// answer as they do alone.
#[test]
fn a_long_prompt_beside_generation_is_read_in_wide_steps_that_pause() {
    let mut rng = Rng(14);
    let first = rng.prompt(10);
    let long = rng.prompt(240);
    let shape = |c: &mut SyntheticConfig| {
        c.own_rows = 16;
        c.pauses_wide_steps = true;
    };
    let model = model(Some(CHECKPOINT), shape);
    // Per step: the rows of its widest segment and its segments.
    let seen: Arc<Mutex<Vec<(u32, u32)>>> = Arc::default();
    let observed = seen.clone();
    let e = engine_with(&model, paused_config(), |host| {
        host.with_pauses(16, 4).with_delay(Duration::from_millis(1)).with_observer(move |view| {
            let widest = (0..view.segments()).map(|i| view.segment(i).n_rows).max().unwrap_or(0);
            observed.lock().unwrap().push((widest, view.segments()));
        })
    });
    let generating = start(&e.rt, request(&first, Some(400))).unwrap();
    let head = generating.wait_for_text(20);
    let before = seen.lock().unwrap().len();
    let long_done = run_one(&e.rt, request(&long, Some(20)));
    let after = seen.lock().unwrap().len();
    let generated = generating.finish();
    let steps: Vec<(u32, u32)> = seen.lock().unwrap()[before..after].to_vec();
    let wide: Vec<usize> = (0..steps.len()).filter(|&i| steps[i].0 > 16).collect();
    assert!(wide.len() >= 4, "the prompt's wide step ran in pieces: {steps:?}");
    assert!(wide.iter().all(|&i| steps[i].1 == 1), "a wide step has the prompt's rows alone");
    assert!(
        wide.windows(2).any(|pair| pair[1] > pair[0] + 1),
        "rounds ran between the pieces: {steps:?}"
    );
    assert_eq!(long_done.text, fresh_run(shape, &long, 20).text);
    assert_eq!(format!("{head}{}", generated.text), fresh_run(shape, &first, 400).text);
    e.consistent();
}

/// A request whose step is paused can be cancelled: what it computed is not kept, the
/// model is told to give the step up, and the next request runs as it would alone.
#[test]
fn a_cancelled_request_gives_its_paused_step_up() {
    let mut rng = Rng(13);
    let long = rng.prompt(240);
    let shape = |c: &mut SyntheticConfig| c.own_rows = 16;
    let model = model(Some(CHECKPOINT), shape);
    let e = engine_with(&model, paused_config(), |host| {
        host.with_pauses(16, 1000).with_delay(Duration::from_millis(1))
    });
    let handle = start(&e.rt, request(&long, Some(20))).unwrap();
    let deadline = Instant::now() + LIMIT;
    while e.snapshot().counters.steps_paused < 3 {
        assert!(Instant::now() < deadline, "the step never paused");
        std::thread::sleep(Duration::from_millis(1));
    }
    e.rt.cancel(handle.id);
    let run = handle.finish();
    assert_eq!(run.finish, ares_api::FinishReason::Cancelled);
    // The same prompt again: nothing of the cancelled one is cached past what it had
    // before its paused step.
    let again = e.ask(&long, 20);
    assert_eq!(again.run.text, fresh_run(shape, &long, 20).text);
    e.consistent();
}

/// Two turns of a conversation: a prompt of 220 rows, then the prompt with the start of the
/// reply and more, which parts from the reply before its end (as a stop string cuts a reply
/// that the client then sends without it), so the end state lies behind the shared cells.
/// Returns the rows the second turn replayed, its answer and the engine.
fn two_turns(inside: bool, checkpoints: u64) -> (u64, Rig, String, Run) {
    let mut rng = Rng(29);
    let prompt = rng.prompt(220);
    let model = model(Some(CHECKPOINT), move |c: &mut SyntheticConfig| {
        c.max_rows = 256;
        c.own_rows = 16;
        c.inner_saves = inside;
    });
    let config = RuntimeConfig { checkpoint_prompt_interval: 4 * PAGE, ..config(256, checkpoints) };
    let e = engine(&model, config);
    let first = e.ask(&prompt, 6);
    let saved = e.snapshot().counters;
    let reply: String = first.run.text.chars().take(4).collect();
    let second = format!("{prompt}{reply}{}", rng.prompt(10));
    let again = e.ask(&second, 6);
    let after = e.snapshot().counters;
    assert_eq!(after.checkpoints_restored - saved.checkpoints_restored, 1);
    (after.replay_rows - saved.replay_rows, e, second, again.run)
}

/// A prompt read in one step that saves its checkpoints inside it leaves the next turn no
/// more to replay than steps cut at those checkpoints do, whatever number of checkpoints the
/// memory holds: the places are taken the latest first, and a later save pushes an earlier
/// one out as it would after a cut. (With the real model the places taken all at once kept
/// an earlier checkpoint and lost the anchor, which `ares-server`'s `real_cache` test guards:
/// this made-up model's budgets did not show it.)
#[test]
fn a_turn_after_saves_inside_a_step_replays_no_more_than_after_cut_steps() {
    for checkpoints in 1..=4 {
        let (cut, ..) = two_turns(false, checkpoints);
        let (inside, e, second, run) = two_turns(true, checkpoints);
        let shape = |c: &mut SyntheticConfig| {
            c.max_rows = 256;
            c.own_rows = 16;
            c.inner_saves = true;
        };
        assert_eq!(run.text, fresh_run(shape, &second, 6).text, "{checkpoints} checkpoints");
        assert!(
            inside <= cut,
            "{checkpoints} checkpoints: {inside} rows replayed after saves inside, {cut} after cuts"
        );
        e.consistent();
    }
}

/// A model that saves the state inside a step wider than its own rows (a prompt read layer
/// by layer) gets the checkpoints due inside the step's rows named in its record, and the
/// step is not cut there: they serve a later prompt as checkpoints at the end of steps do,
/// also when the step was paused between its pieces.
#[test]
fn checkpoints_inside_a_step_serve_a_later_prompt() {
    let mut rng = Rng(21);
    let prompt = rng.prompt(200);
    let later = format!("{}{}", head(&prompt, 150), rng.prompt(30));
    let shape = |c: &mut SyntheticConfig| {
        c.own_rows = 16;
        c.inner_saves = true;
    };
    for pieces in [1, 3] {
        let model = model(Some(CHECKPOINT), shape);
        let e = engine_with(&model, config(256, 16), |host| host.with_pauses(16, pieces));
        let first = e.ask(&prompt, 20);
        assert_eq!(first.run.text, fresh_run(shape, &prompt, 20).text, "pieces {pieces}");
        let c = e.snapshot().counters;
        // Steps of 64 rows with a checkpoint every 16 cells: three inside each.
        assert!(c.checkpoints_saved >= 9, "pieces {pieces}: {} checkpoints", c.checkpoints_saved);
        assert!(c.prefill_rows <= 200, "pieces {pieces}: {} rows", c.prefill_rows);
        let second = e.ask(&later, 20);
        assert_eq!(second.run.text, fresh_run(shape, &later, 20).text, "pieces {pieces}");
        assert!(second.report.cached_tokens >= 128, "{:?}", second.report);
        e.consistent();
    }
}

/// An agent's next turn right after the last (a short turn, then the whole conversation and a
/// long tool result): the end state of the short turn is saved with the next record, into the
/// one place there is, and the next turn restores it in that same record. Its rows are read in
/// one wide step whose checkpoints inside take places before the record is written, which
/// gives up the end state to make room: its save must still run before its restore. (Dropped
/// with the checkpoint, the restore read what the place held before, and on Flash-Next the
/// step failed: "the rows of slot 0 start at cell 650 and its sequence is at cell 512".)
#[test]
fn a_turn_that_restores_an_end_state_still_saved_with_its_record_gets_that_state() {
    let mut rng = Rng(31);
    let shape = |c: &mut SyntheticConfig| {
        c.max_rows = 256;
        c.own_rows = 16;
        c.inner_saves = true;
    };
    let model = model(Some(CHECKPOINT), shape);
    let e = engine(&model, config(256, 1));
    // Less than a page with the reply: no checkpoint but the end state's.
    let first = rng.prompt(10);
    let done = e.ask(&first, 6);
    assert_eq!(done.run.text, fresh_run(shape, &first, 6).text);
    let saved = e.snapshot().counters;
    assert_eq!(saved.checkpoints_saved, 1, "the end state is the one checkpoint");
    let second = format!("{first}{}{}", done.run.text, rng.prompt(150));
    let again = e.ask(&second, 6);
    let after = e.snapshot().counters;
    assert_eq!(after.checkpoints_restored - saved.checkpoints_restored, 1);
    assert_eq!(again.report.replayed_rows, 0, "{:?}", again.report);
    assert_eq!(again.run.text, fresh_run(shape, &second, 6).text);
    e.consistent();
}

/// Pages a model keeps in host memory (from page 4 on, [`FarPages`]: the K and V of a context
/// beyond VRAM) that only the prompt cache holds once their request ended rank as prompt cache,
/// the lowest user of the engine's RAM (the owner's rule of 2026-11-22). With a tier of experts
/// that wants more than the limit leaves, they leave the devices as soon as the request ended,
/// with nothing below the devices to keep them (no level on SSD here): the next request with
/// the same prompt finds only the pages in VRAM. With a tier that wants little, they stay. The
/// answers are the same either way.
#[test]
fn far_pages_of_a_finished_request_give_way_when_the_tier_of_experts_wants_the_memory() {
    let mut rng = Rng(31);
    let prompt = rng.prompt(10 * PAGE as usize + 3);
    let mut texts = Vec::new();
    for hungry in [true, false] {
        let model = model(Some(CHECKPOINT), |c| {
            c.far_pages = Some(FarPages { first: 4, page_bytes: 1 << 20 });
        });
        let ram = PinnedRam::new(64 << 20);
        ram.want_experts(if hungry { 1 << 40 } else { 0 }, 0);
        let books = ram.clone();
        let e = engine_with(&model, config(64, 4), move |host| host.with_ram(books));
        let first = e.ask(&prompt, 4);
        // One more turn of the conductor's loop, which trims between steps.
        let _ = e.ask("ab", 1);
        let cached_far = ram.context_cached();
        let again = e.ask(&prompt, 4);
        let found = again.run.usage.cached_prompt_tokens;
        if hungry {
            assert_eq!(cached_far, 0, "the far pages went");
            assert_eq!(found, 4 * PAGE, "only the pages in VRAM stayed cached");
        } else {
            assert!(cached_far >= 6 << 20, "the far pages stayed: {cached_far}");
            assert!(found >= 10 * PAGE, "the whole prompt stayed cached: {found}");
        }
        assert_eq!(again.run.text, first.run.text, "hungry {hungry}");
        texts.push(first.run.text);
        e.consistent();
    }
    assert_eq!(texts[0], texts[1]);
}

// --- the progress of a prompt ------------------------------------------------

/// A prompt read in many steps hears how far it has got after each of them, from
/// nothing to all of it, before its first text; a prompt the cache holds most of
/// begins at the cached tokens, and the replay that rebuilds the state counts as
/// work, so the count still rises step by step.
#[test]
fn a_prompt_hears_its_progress_step_by_step_from_what_the_cache_gave() {
    let mut rng = Rng(41);
    let prompt = rng.prompt(200);
    let model = model(Some(CHECKPOINT), |_| {});
    let e = engine(&model, config(256, 16));
    let (first, progress) = start(&e.rt, request(&prompt, Some(8))).unwrap().finish_with_progress();
    assert_eq!(first.usage.cached_prompt_tokens, 0);
    let said: Vec<u32> = progress.iter().map(|p| p.processed).collect();
    // The steps end at the checkpoints, every page: a report after each.
    assert!(said.len() >= 200 / PAGE as usize, "{said:?}");
    assert_eq!((said[0], *said.last().unwrap()), (0, 200), "{said:?}");
    assert!(progress.iter().all(|p| p.total == 200 && p.cached == 0));
    let longer = format!("{prompt}{}", rng.prompt(60));
    let (again, progress) = start(&e.rt, request(&longer, Some(8))).unwrap().finish_with_progress();
    let cached = again.usage.cached_prompt_tokens;
    assert!(cached >= 192, "{:?}", again.usage);
    let said: Vec<u32> = progress.iter().map(|p| p.processed).collect();
    assert_eq!((said[0], *said.last().unwrap()), (cached, 260), "{said:?}");
    assert!(said.len() >= 3, "the rest was read in steps of a page: {said:?}");
    e.consistent();
}

/// A prompt the model reads in one step that pauses (a pass) hears the share of the
/// pass done at every pause, where a count of rows done would not move until its end.
#[test]
fn a_paused_pass_reports_the_share_it_has_done() {
    let mut rng = Rng(42);
    let prompt = rng.prompt(60);
    let shape = |c: &mut SyntheticConfig| c.own_rows = 16;
    let model = model(Some(CHECKPOINT), shape);
    let config = RuntimeConfig { checkpoint_prompt_interval: 64 * PAGE, ..paused_config() };
    let e = engine_with(&model, config, |host| host.with_pauses(16, 4).with_shares());
    let (run, progress) = start(&e.rt, request(&prompt, Some(8))).unwrap().finish_with_progress();
    assert_eq!(run.text, fresh_run(shape, &prompt, 8).text, "as without pauses");
    let said: Vec<u32> = progress.iter().map(|p| p.processed).collect();
    // The pass is the first 48 rows (the step ends at the last page before the
    // prompt's end, where the cache keeps a checkpoint): the start, a report at
    // each of its three pauses (a quarter of its rows more each), its end, and
    // the end of the step of the last rows.
    assert_eq!(said, [0, 12, 24, 36, 48, 60]);
    e.consistent();
}

/// A step of a prompt that runs long reports how far it has got while it runs, about
/// once a second, from a model that says so ([`ares_runtime::Executor::step_share`]).
#[test]
fn a_long_step_reports_its_share_while_it_runs() {
    let mut rng = Rng(43);
    let prompt = rng.prompt(40);
    let model = model(Some(CHECKPOINT), |_| {});
    let config = RuntimeConfig { checkpoint_prompt_interval: 64 * PAGE, ..config(256, 16) };
    let e = engine_with(&model, config, |host| {
        host.with_step_cost(|rows| Duration::from_millis(if rows > 8 { 2600 } else { 1 }))
            .with_shares()
    });
    let (_, progress) = start(&e.rt, request(&prompt, Some(4))).unwrap().finish_with_progress();
    let said: Vec<u32> = progress.iter().map(|p| p.processed).collect();
    // The long step is the first 32 rows (it ends at the last page before the
    // prompt's end): the start, two reports inside it (after one second and after
    // two, as far as the share of its time), its end, and the end of the step of
    // the last rows.
    assert_eq!(said.len(), 5, "{said:?}");
    assert_eq!((said[0], said[3], said[4]), (0, 32, 40), "{said:?}");
    assert!((10..=15).contains(&said[1]) && (22..=28).contains(&said[2]), "{said:?}");
    let elapsed: Vec<u128> = progress.iter().map(|p| p.elapsed.as_millis()).collect();
    assert!(elapsed[1] >= 1000 && elapsed[2] >= 2000 && elapsed[3] >= 2600, "{elapsed:?}");
    e.consistent();
}
