//! The conductor without a GPU: bookkeeping, planning, speculation, stopping
//! and event delivery, against an executor that computes picks on the host.
//!
//! These run in the default `cargo test`. What they prove is about the
//! conductor's decisions; that the kernels compute what the model should is
//! `tests/gpu.rs`'s business.

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{Engine, EngineError, Event, EventSink, FinishReason, RequestId};
use ares_runtime::testing::{Gate, HostExecutor};
use ares_runtime::{Model, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use common::*;

fn model(configure: impl FnOnce(&mut SyntheticConfig)) -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 512;
    configure(&mut config);
    Arc::new(SyntheticModel::new(config).unwrap())
}

fn config(speculation: bool, pages: u32) -> RuntimeConfig {
    RuntimeConfig {
        pages,
        cells_per_page: 16,
        max_slots: 64,
        // A window of one row is no drafts at all.
        window_rows: if speculation { 8 } else { 1 },
        warm_up: false,
        ..RuntimeConfig::default()
    }
}

fn runtime_with(
    model: &Arc<SyntheticModel>,
    config: RuntimeConfig,
    executor: impl FnOnce(HostExecutor) -> HostExecutor,
) -> Runtime {
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots);
    Runtime::start(model.clone(), Box::new(executor(host)), config).unwrap()
}

fn runtime(model: &Arc<SyntheticModel>, speculation: bool) -> Runtime {
    runtime_with(model, config(speculation, 4096), |e| e)
}

/// A context as large as the pool. A request with no limit of its own then
/// generates for seconds of steps and not for a tenth of one, so a test that
/// cancels it, shuts it down or sends something next to it finds it running
/// however late the test's thread gets to it.
fn roomy(c: &mut SyntheticConfig) {
    c.context_length = 65_536;
}

/// A model whose text loops and changes its loop every 40 positions, and whose
/// draft block gets the first links right more often than the last.
fn repeating(c: &mut SyntheticConfig) {
    c.context.window = 1;
    c.context.epoch = 40;
    c.context.buckets = 512;
    c.own_drafts = 7;
    c.own_accuracy = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
}

/// Gives the model a draft block with these chances for its links.
fn drafter(c: &mut SyntheticConfig, accuracy: &[f64]) {
    c.context.window = 1;
    c.own_drafts = 7;
    c.own_accuracy = accuracy.to_vec();
}

/// A model whose text never repeats inside a request (its next token depends
/// on the position, which changes at every token) and whose draft block is
/// never right: only text another request wrote can be drafted.
fn never_repeating(c: &mut SyntheticConfig) {
    drafter(c, &[0.0]);
    c.context.epoch = 1;
    c.context.buckets = 512;
}

#[test]
fn a_request_says_the_same_alone_and_next_to_others_with_and_without_speculation() {
    for repetitive in [false, true] {
        for speculation in [false, true] {
            let model = model(|c| {
                if repetitive {
                    repeating(c)
                }
            });
            // A little delay per step lets requests arrive between steps.
            let rt = runtime_with(&model, config(speculation, 4096), |e| {
                e.with_delay(Duration::from_micros(300))
            });
            output_is_independent_of_the_batch(&rt, Duration::from_micros(400));
        }
    }
}

#[test]
fn speculation_changes_the_number_of_steps_and_not_the_text() {
    let model = model(repeating);
    // A step costs a fixed time whatever its rows, as reading the weights does,
    // so a verified draft is nearly free and the planner asks for them.
    let slow = |speculation| {
        runtime_with(&model, config(speculation, 4096), |e| {
            e.with_delay(Duration::from_micros(200))
        })
    };
    let (plain, drafted) = (slow(false), slow(true));
    let expected = outputs_of(&plain, 99, 12);
    let zero = snapshot(&slow(false));
    let before = snapshot(&drafted);
    let got = outputs_of(&drafted, 99, 12);
    assert_eq!(got, expected, "speculation must not change what is said");

    let plain = since(&snapshot(&plain), &zero);
    let drafted = since(&snapshot(&drafted), &before);
    assert_eq!(plain["draft_rows"], 0, "no drafts without speculation");
    assert!(drafted["accepted"] > 100, "text that repeats gets drafts accepted: {drafted:?}");
    assert!(
        drafted["rejected"] > 0,
        "the loop changes every 40 positions, so some drafts are wrong: {drafted:?}"
    );
    assert!(
        // Only the stretches of a loop after it has closed can be drafted, so
        // this is far from half.
        drafted["steps"] * 100 < plain["steps"] * 85,
        "accepted drafts save steps: {} steps against {}",
        drafted["steps"],
        plain["steps"]
    );
    assert_eq!(plain["tokens"], drafted["tokens"]);
}

#[test]
fn text_that_does_not_repeat_gets_few_drafts_accepted() {
    let model = model(never_repeating);
    let rt = runtime(&model, true);
    let before = snapshot(&rt);
    outputs_of(&rt, 5, 8);
    let d = since(&snapshot(&rt), &before);
    // A draft is a guess among 256 tokens: nearly all are cut again, and the
    // planner learns to stop asking for them.
    assert!(d["accepted"] * 20 <= d["draft_rows"].max(1), "{d:?}");
}

#[test]
fn a_text_another_request_wrote_is_drafted_from_the_corpus_all_requests_share() {
    // A draft can only come from what another request read and wrote.
    let model = model(never_repeating);
    let rt = runtime_with(&model, config(true, 4096), |e| e.with_delay(Duration::from_micros(200)));
    let first = alone(&rt, "written twice", Some(200));
    let before = snapshot(&rt);
    assert!(before.counters.steps >= 190, "the first request found nothing to copy");
    let second = alone(&rt, "written twice", Some(200));
    assert_eq!(second, first);
    let d = since(&snapshot(&rt), &before);
    assert!(d["accepted"] > 150, "the second request copies the first: {d:?}");
    assert!(d["steps"] < 80, "{d:?}");
    assert_eq!(d["rejected"], 0, "{d:?}");
}

#[test]
fn cancelling_frees_the_pages_at_once() {
    let model = model(roomy);
    let rt = runtime_with(&model, config(true, 4096), |e| e.with_delay(Duration::from_micros(500)));
    cancel_frees_pages_and_spares_the_others(&rt);
}

#[test]
fn more_requests_than_the_pool_holds_wait_in_line_and_say_the_same() {
    // 32 pages of 16 cells. A request of 40 prompt tokens and 100 of output
    // is charged 3 pages and 7: three of them fit next to each other.
    let model = model(|c| c.context_length = 256);
    for speculation in [false, true] {
        let rt = runtime_with(&model, config(speculation, 32), |e| {
            e.with_delay(Duration::from_micros(300))
        });
        more_requests_than_fit_wait_and_say_the_same(&rt, 3);
    }
}

#[test]
fn more_requests_than_the_devices_have_slots_for_wait_in_line() {
    // Pages for everybody, slots for four.
    let model = model(|_| {});
    let config = RuntimeConfig { max_slots: 4, ..config(true, 4096) };
    let rt = runtime_with(&model, config, |e| e.with_delay(Duration::from_micros(300)));
    more_requests_than_fit_wait_and_say_the_same(&rt, 4);
}

#[test]
fn a_request_that_writes_more_than_expected_gives_way_and_comes_back() {
    // 24 pages of 16 cells; a request of 26 prompt tokens and 180 of output
    // ends with 13 pages. The accounting is told to expect a twentieth of
    // the output, so it lets four in where two fit.
    for repetitive in [false, true] {
        let model = model(|c| {
            c.context_length = 256;
            if repetitive {
                repeating(c)
            }
        });
        let config = RuntimeConfig { output_ratio: 0.05, ..config(true, 24) };
        let rt = runtime_with(&model, config, |e| e.with_delay(Duration::from_micros(100)));
        a_request_that_gives_way_comes_back_and_says_the_same(&rt);
    }
}

#[test]
fn a_request_alone_runs_to_the_end_of_the_pool_and_its_drafts_are_cut_there() {
    // A model whose text falls into a loop and stays there, so that drafts
    // are proposed up to the last cell.
    let model = model(|c| {
        drafter(c, &[0.9, 0.8, 0.7]);
        c.context.epoch = 0;
        c.context.buckets = 512;
    });
    // Prompts of several lengths, so that the windows of some of them reach
    // past the pool's last cell.
    let prompts: Vec<String> =
        (0..6).map(|n| format!("alone in a small pool{}", ".".repeat(n))).collect();
    let run = |speculation| {
        // 6 pages of 16 cells in a context of 512.
        let rt = runtime_with(&model, config(speculation, 6), |e| {
            e.with_delay(Duration::from_micros(200))
        });
        let runs: Vec<Run> = prompts
            .iter()
            .map(|prompt| the_last_request_runs_to_the_end_of_the_pool(&rt, prompt, 96))
            .collect();
        (runs, snapshot(&rt).counters)
    };
    let ((plain, plain_counters), (drafted, counters)) = (run(false), run(true));
    assert_eq!(drafted, plain, "the text is the same to the last cell");
    assert_eq!(plain_counters.draft_rows_cut, 0);
    assert!(counters.drafts_accepted > 0, "{counters:?}");
    // Drafts that reach past the pool are dropped, and the request goes on
    // a row at a time to the pool's end.
    assert!(counters.draft_rows_cut > 0, "{counters:?}");
}

#[test]
fn a_request_in_line_can_be_cancelled_and_the_others_go_on() {
    let model = model(|c| c.context_length = 256);
    let gate = Gate::new();
    // Two slots: the third and fourth request wait.
    let config = RuntimeConfig { max_slots: 2, ..config(true, 4096) };
    let rt = runtime_with(&model, config, |e| e.with_gate(gate.clone()));
    let expected = alone(&rt, "in line 3", Some(20));
    gate.close();
    let handles: Vec<Handle> =
        (0..4).map(|i| start(&rt, request(&format!("in line {i}"), Some(20))).unwrap()).collect();
    let mut handles = handles.into_iter();
    let (first, second) = (handles.next().unwrap(), handles.next().unwrap());
    let (third, fourth) = (handles.next().unwrap(), handles.next().unwrap());
    rt.cancel(third.id);
    gate.open();
    let cancelled = third.finish();
    assert_eq!(cancelled.finish, FinishReason::Cancelled);
    assert_eq!(cancelled.usage.completion_tokens, 0, "it never ran");
    assert_eq!(cancelled.text, "");
    for handle in [first, second] {
        assert_eq!(handle.finish().usage.completion_tokens, 20);
    }
    assert_eq!(fourth.finish(), expected);
    let s = snapshot(&rt);
    assert_eq!((s.active, s.waiting), (0, 0));
    assert_eq!(s.pages_free, s.pages_capacity);
}

#[test]
fn a_prompt_the_pool_cannot_hold_is_the_one_refusal() {
    // 4 pages of 16 cells in a context of 512: a prompt of 64 tokens leaves
    // no cell for the first token after it.
    let model = model(|_| {});
    let rt = runtime_with(&model, config(true, 4), |e| e);
    let long = "y".repeat(64);
    assert_eq!(
        start(&rt, request(&long, Some(1))).unwrap_err(),
        EngineError::ContextOverflow { prompt_tokens: 64, limit: 64 }
    );
    let fits = "y".repeat(63);
    let run = run_one(&rt, request(&fits, None));
    assert_eq!((run.finish, run.usage.completion_tokens), (FinishReason::Length, 2));
}

#[test]
fn shutdown_ends_requests_in_flight_and_joins_the_conductor() {
    let model = model(|c| c.context_length = 4096);
    let gate = Gate::new();
    let rt = runtime_with(&model, config(true, 64 * 256), |e| {
        e.with_gate(gate.clone()).with_delay(Duration::from_millis(1))
    });
    shutdown_ends_everything(&rt);
}

#[test]
fn shutting_down_while_a_step_is_held_waits_for_it_and_then_ends_everything() {
    let model = model(roomy);
    let gate = Gate::new();
    // A little delay per step, so that the request is still running when
    // the gate closes.
    let rt = runtime_with(&model, config(true, 4096), |e| {
        e.with_gate(gate.clone()).with_delay(Duration::from_micros(200))
    });
    let handle = start(&rt, request("held", None)).unwrap();
    handle.wait_for_text(1);
    gate.close();
    // The conductor is inside a step; the shutdown queues behind it.
    let opener = {
        let gate = gate.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            gate.open();
        })
    };
    rt.shutdown();
    opener.join().unwrap();
    assert_eq!(handle.finish().finish, FinishReason::Cancelled);
    assert!(!rt.is_running());
}

#[test]
fn dropping_the_runtime_ends_requests_too() {
    let model = model(roomy);
    let rt = runtime_with(&model, config(true, 4096), |e| e.with_delay(Duration::from_millis(1)));
    let handle = start(&rt, request("dropped", None)).unwrap();
    handle.wait_for_text(1);
    drop(rt);
    assert_eq!(handle.finish().finish, FinishReason::Cancelled);
}

#[test]
fn bad_requests_are_refused_before_anything_starts() {
    let model = model(|_| {});
    bad_requests_are_refused_at_once(&runtime(&model, true), 512);
}

#[test]
fn requests_end_with_the_right_reason_and_counts() {
    for (speculation, repetitive) in [(false, false), (true, false), (true, true)] {
        let model = model(|c| {
            if repetitive {
                repeating(c)
            }
        });
        requests_end_for_the_right_reason(&runtime(&model, speculation), 512);
    }
}

#[test]
fn an_end_token_ends_requests_at_its_first_occurrence_with_and_without_speculation() {
    for speculation in [false, true] {
        the_end_token_ends_a_request_where_it_first_comes_up(&|eos| {
            let model = model(|c| {
                repeating(c);
                c.eos = eos;
            });
            runtime(&model, speculation)
        });
    }
}

#[test]
fn a_sink_that_refuses_cancels_the_request_and_frees_its_pages() {
    let model = model(|_| {});
    let rt = runtime_with(&model, config(true, 4096), |e| e.with_delay(Duration::from_micros(200)));
    let total = snapshot(&rt).pages_capacity;
    let delivered = Arc::new(AtomicUsize::new(0));
    let count = delivered.clone();
    let sink = EventSink::new(move |_| count.fetch_add(1, Ordering::SeqCst) < 3);
    let mut r = request("slow client", None);
    r.id = RequestId(900_001);
    rt.submit(r, sink).unwrap();
    let deadline = std::time::Instant::now() + LIMIT;
    while snapshot(&rt).counters.requests_cancelled == 0 {
        assert!(std::time::Instant::now() < deadline, "the request was never cancelled");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(snapshot(&rt).pages_free, total);
    assert_eq!(
        delivered.load(Ordering::SeqCst),
        4,
        "three accepted, then the refusal; nothing after"
    );
    assert_eq!(snapshot(&rt).counters.requests_cancelled, 1);
}

#[test]
fn an_id_in_use_is_refused_by_a_terminal_event() {
    let model = model(roomy);
    let rt = runtime_with(&model, config(true, 4096), |e| e.with_delay(Duration::from_millis(1)));
    let first = request("first", None);
    let id = first.id;
    let first = start(&rt, first).unwrap();
    first.wait_for_text(1);
    let mut second = request("second", Some(3));
    second.id = id;
    let error = start(&rt, second).unwrap().failure();
    assert!(matches!(error, EngineError::InvalidRequest(m) if m.contains("in use")));
    rt.cancel(id);
    assert_eq!(first.finish().finish, FinishReason::Cancelled);
}

#[test]
fn long_prompts_are_read_in_chunks_next_to_generation_and_say_the_same() {
    let mut rng = Rng(42);
    let long = rng.prompt(300);
    // The request that generates has no limit and outlasts the test.
    let model = model(|c| {
        roomy(c);
        c.max_rows = 64;
    });

    // Reference: the long prompt in few big steps, alone.
    let wide = runtime_with(&model, RuntimeConfig { max_rows: 0, ..config(false, 4096) }, |e| e);
    let reference = alone(&wide, &long, Some(30));
    drop(wide);

    // A generating request is running when the long prompt arrives, and the
    // steps are small: the prompt must go through next to generation.
    let rt = runtime_with(&model, RuntimeConfig { max_rows: 16, ..config(false, 4096) }, |e| {
        e.with_delay(Duration::from_micros(300))
    });
    let generating = start(&rt, request("already generating", None)).unwrap();
    generating.wait_for_text(5);
    let before = snapshot(&rt);
    let run = run_one(&rt, request(&long, Some(30)));
    assert_eq!(run, reference, "chunking the prompt must not change the answer");
    let d = since(&snapshot(&rt), &before);
    assert!(d["mixed"] >= 3, "the prompt shared steps with the generating request: {d:?}");
    assert!(d["steps"] >= 300 / 16, "{d:?}");
    rt.cancel(generating.id);
    assert_eq!(generating.finish().finish, FinishReason::Cancelled, "it generated all the while");
}

#[test]
fn the_page_tables_the_devices_hold_match_the_conductors() {
    let model = model(repeating);
    let host = HostExecutor::new(model.clone(), 64, 64).with_delay(Duration::from_micros(300));
    let tables = host.tables();
    let rt = Runtime::start(model.clone(), Box::new(host), config(true, 4096)).unwrap();
    let handles: Vec<Handle> = (0..6)
        .map(|i| {
            start(&rt, request(&format!("table check {i} with a prompt of some length"), Some(200)))
                .unwrap()
        })
        .collect();
    // Look at the tables several times while the sequences grow, are
    // cut by rejected drafts and grow again.
    let mut checked = 0;
    while checked < 20 {
        let sequences = rt.sequences().unwrap();
        // The question is answered between steps, so the devices have been
        // told everything the conductor holds.
        let mirror = tables.lock().unwrap();
        for s in &sequences {
            assert_eq!(&mirror[s.slot as usize][..s.pages.len()], &s.pages[..], "{s:?}");
            assert_eq!(s.pages.len() as u64, s.cells.div_ceil(16));
        }
        checked += 1;
        drop(mirror);
        std::thread::sleep(Duration::from_millis(2));
    }
    for handle in handles {
        handle.finish();
    }
}

#[test]
fn a_failing_device_ends_every_request_with_an_error_and_stops_taking_new_ones() {
    let model = model(|_| {});
    let rt = runtime_with(&model, config(true, 4096), |e| e.failing_at(5));
    assert_eq!(rt.failure(), None, "an engine that works has no failure to tell");
    let handles: Vec<Handle> =
        (0..4).map(|i| start(&rt, request(&format!("doomed {i}"), None)).unwrap()).collect();
    for handle in handles {
        let error = handle.failure();
        assert!(matches!(&error, EngineError::Internal(m) if m.contains("injected")), "{error}");
    }
    let later = start(&rt, request("after", Some(1))).unwrap().failure();
    assert!(matches!(later, EngineError::Internal(_)), "{later}");
    // Whoever holds the engine can ask, without a request, why it serves none.
    let said = rt.failure().expect("the engine says that it failed");
    assert!(said.contains("injected"), "{said}");
}

#[test]
fn a_failing_warm_up_step_reports_the_device_and_the_cause() {
    let model = model(|_| {});
    let host = HostExecutor::new(model.clone(), model.config().max_rows, 4).failing_at(2);
    let config = RuntimeConfig { max_slots: 4, ..RuntimeConfig::default() };
    let error = Runtime::start(model.clone(), Box::new(host), config).unwrap_err();
    assert_eq!(error.to_string(), "device 0: injected failure");
}

#[test]
fn decisions_are_not_implemented_and_say_so() {
    use ares_api::{DecisionRequest, DecisionSink, Prompt};
    let model = model(|_| {});
    let rt = runtime(&model, false);
    let request = DecisionRequest {
        model: "synthetic".into(),
        prompt: Prompt::Text("x".into()),
        questions: Vec::new(),
        id: fresh_id(),
    };
    let error = rt.decide(request, DecisionSink::new(|_| {})).unwrap_err();
    assert!(matches!(error, EngineError::Unsupported(_)));
}

#[test]
fn stats_can_be_taken_and_reset_while_running() {
    let model = model(|_| {});
    let rt = runtime_with(&model, config(true, 4096), |e| e.with_delay(Duration::from_micros(100)));
    let handle = start(&rt, request("measure", Some(100))).unwrap();
    handle.finish();
    let taken = rt.take_snapshot().unwrap();
    assert!(taken.counters.steps >= 50 && taken.counters.tokens == 100, "{taken}");
    let after = rt.snapshot().unwrap();
    assert_eq!(after.counters.steps, 0, "taking the snapshot started the count afresh");
    assert!(taken.to_string().contains("accept/commit"));
}

#[test]
fn many_requests_with_random_sizes_arriving_at_random_times_all_end_correctly() {
    let model = model(repeating);
    let rt = runtime_with(&model, config(true, 4096), |e| e.with_delay(Duration::from_micros(200)));
    let mut rng = Rng(2024);
    let jobs: Vec<(String, u32)> = (0..120)
        .map(|_| {
            let len = 1 + rng.below(120) as usize;
            (rng.prompt(len), 1 + rng.below(80) as u32)
        })
        .collect();
    let expected: Vec<Run> = {
        let reference = runtime(&model, false);
        jobs.iter().map(|(p, m)| alone(&reference, p, Some(*m))).collect()
    };
    let started = Mutex::new(Vec::new());
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut rng = Rng(77);
            for (i, (prompt, max)) in jobs.iter().enumerate() {
                started.lock().unwrap().push((i, start(&rt, request(prompt, Some(*max))).unwrap()));
                std::thread::sleep(Duration::from_micros(rng.below(600)));
            }
            done.store(true, Ordering::SeqCst);
        });
    });
    assert!(done.load(Ordering::SeqCst));
    for (i, handle) in started.into_inner().unwrap() {
        assert_eq!(handle.finish(), expected[i], "request {i}");
    }
    let s = snapshot(&rt);
    assert_eq!(s.pages_free, s.pages_capacity);
    assert!(matches!(
        Event::Done { finish: FinishReason::Length, usage: Default::default() },
        Event::Done { .. }
    ));
}

#[test]
fn a_steady_batch_still_finds_out_that_drafts_are_cheap() {
    // Four requests with prompts of one token, started together: every step
    // has the same four rows, and the step time model cannot tell a cost per
    // step from a cost per row (it has seen one step size only). Drafting must
    // not stay off for ever because of that.
    let model = model(|c| {
        drafter(c, &[0.9, 0.8, 0.7]);
        c.context.epoch = 0;
        c.context.buckets = 512;
    });
    let run = |speculation| {
        let rt = runtime_with(&model, config(speculation, 4096), |e| {
            e.with_delay(Duration::from_micros(200))
        });
        let handles: Vec<Handle> = (0..4)
            .map(|i| start(&rt, request(["a", "b", "c", "d"][i], Some(300))).unwrap())
            .collect();
        let runs: Vec<Run> = handles.into_iter().map(Handle::finish).collect();
        (runs, snapshot(&rt).counters)
    };
    let ((plain_runs, plain), (drafted_runs, drafted)) = (run(false), run(true));
    assert_eq!(drafted_runs, plain_runs, "the text is the same");
    assert!(drafted.drafts_accepted > 300, "{drafted:?}");
    assert!(
        drafted.steps * 100 < plain.steps * 60,
        "{} steps against {} without drafts",
        drafted.steps,
        plain.steps
    );
}

// --- prompts next to generation ----------------------------------------------

/// What a step carried, as the observer sees the record.
#[derive(Clone, Copy, Debug)]
struct Seen {
    rows: u32,
    /// The rows of segments that are a prompt's: certain, and more than one.
    prompt_rows: u32,
    /// Segments that are the newest token of a sequence, without drafts.
    riders: u32,
    /// Windows that verify drafts (their rows are not certain).
    windows: u32,
}

type Steps = Arc<Mutex<Vec<Seen>>>;

fn watch(steps: &Steps) -> impl FnMut(&ares_runtime::RecordView<'_>) + Send + 'static {
    let steps = steps.clone();
    move |record| {
        let mut seen = Seen { rows: record.rows(), prompt_rows: 0, riders: 0, windows: 0 };
        for i in 0..record.segments() {
            let segment = record.segment(i);
            match (segment.direct, segment.n_rows) {
                (false, _) => seen.windows += 1,
                (true, 1) => seen.riders += 1,
                (true, n) => seen.prompt_rows += n,
            }
        }
        steps.lock().unwrap().push(seen);
    }
}

/// A runtime whose steps cost what a device's do, a pass over the weights and
/// a little for every row, with a sequence that is generating (and drafting).
/// The executor says what each step cost, so the turns of prompts and rounds
/// are taken from exactly these prices; it also takes the time, so that the
/// prompts of a test arrive while the sequence generates. The sequence has no
/// limit but the pool, which it would take twenty seconds of steps to fill:
/// it generates for as long as a test goes on, and [`end`] checks that it did.
fn generating(steps: &Steps) -> (Runtime, Handle) {
    let model = model(|c| {
        repeating(c);
        roomy(c);
        c.max_rows = 128;
    });
    let observer = watch(steps);
    let rt = runtime_with(&model, config(true, 4096), |e| {
        e.with_row_cost(Duration::from_micros(3000), Duration::from_micros(30))
            .with_observer(observer)
    });
    let generating = start(&rt, request("a sequence that goes on", None)).unwrap();
    generating.wait_for_text(60);
    (rt, generating)
}

/// Ends the sequence of [`generating`], which was generating to the last.
fn end(rt: Runtime, generating: Handle) {
    rt.cancel(generating.id);
    assert_eq!(generating.finish().finish, FinishReason::Cancelled, "it generated all the while");
    rt.shutdown();
}

#[test]
fn a_prompt_next_to_a_drafting_sequence_is_read_in_pieces_without_draft_rows() {
    let steps: Steps = Arc::default();
    let (rt, generating) = generating(&steps);
    let before = steps.lock().unwrap().len();
    let long = start(&rt, request(&Rng(5).prompt(1000), Some(1))).unwrap();
    long.finish();
    let after = steps.lock().unwrap().len();
    let ours: Vec<Seen> = steps.lock().unwrap()[before..after].to_vec();
    let pieces: Vec<&Seen> = ours.iter().filter(|s| s.prompt_rows > 0).collect();
    assert_eq!(pieces.iter().map(|s| s.prompt_rows).sum::<u32>(), 1000, "the whole prompt");
    // A piece is a step of its own: the prompt's rows and the newest token of
    // the sequence that generates, never a window of drafts.
    assert!(pieces.iter().all(|s| s.windows == 0), "{pieces:?}");
    assert!(pieces.iter().all(|s| s.rows <= 128), "{pieces:?}");
    assert!(
        pieces.iter().all(|s| s.riders == 1),
        "the sequence goes on in every piece: {pieces:?}"
    );
    // The sequence went on in rounds of drafts between the pieces.
    assert!(ours.iter().any(|s| s.windows > 0 && s.rows > 1), "{ours:?}");
    end(rt, generating);
}

#[test]
fn a_long_prompt_takes_half_of_the_time_next_to_a_sequence_that_generates() {
    let steps: Steps = Arc::default();
    let (rt, generating) = generating(&steps);
    let before = steps.lock().unwrap().len();
    start(&rt, request(&Rng(6).prompt(2400), Some(1))).unwrap().finish();
    let after = steps.lock().unwrap().len();
    // The cost the executor charges, which is what the conductor books.
    let ms = |s: &Seen| 3.0 + 0.03 * f64::from(s.rows);
    let ours: Vec<Seen> = steps.lock().unwrap()[before..after].to_vec();
    let prompt: f64 = ours.iter().filter(|s| s.prompt_rows > 0).map(ms).sum();
    let rounds: f64 = ours.iter().filter(|s| s.prompt_rows == 0).map(ms).sum();
    // The default slowdown is 2: the sequence gets as much time as the prompt.
    let owed = rounds / prompt;
    assert!((0.8..1.25).contains(&owed), "the sequence had {rounds:.0} ms, the prompt {prompt:.0}");
    end(rt, generating);
}

#[test]
fn a_short_prompt_next_to_a_drafting_sequence_is_read_whole_in_the_next_step() {
    let steps: Steps = Arc::default();
    let (rt, generating) = generating(&steps);
    for round in 0..4 {
        let before = steps.lock().unwrap().len();
        let short = start(&rt, request(&Rng(7 + round).prompt(60), Some(2))).unwrap();
        short.finish();
        let after = steps.lock().unwrap().len();
        let pieces =
            steps.lock().unwrap()[before..after].iter().filter(|s| s.prompt_rows > 0).count();
        assert_eq!(pieces, 1, "round {round}: a prompt of 60 rows is one piece");
    }
    end(rt, generating);
}

#[test]
fn a_slowdown_that_leaves_generation_no_time_is_refused_at_start() {
    let model = model(|_| {});
    for slowdown in [1.0, 0.5, f64::NAN, f64::INFINITY] {
        let host = HostExecutor::new(model.clone(), model.config().max_rows, 4);
        let config = RuntimeConfig { max_slots: 4, prefill_slowdown: slowdown, ..config(true, 64) };
        let error = Runtime::start(model.clone(), Box::new(host), config).unwrap_err();
        assert!(error.to_string().contains("must be above 1"), "{slowdown}: {error}");
    }
}

// --- the draft policy -------------------------------------------------------

/// A device whose weight kernels read their rows in launches of eight: a step
/// costs 3 ms and 0.03 ms a row, and 3 ms more from the ninth row on and again
/// from the seventeenth. A link of the draft block's chain costs 0.3 ms.
fn launches(rows: u32) -> Duration {
    Duration::from_micros(3000 + 30 * u64::from(rows) + 3000 * u64::from(rows.div_ceil(8) - 1))
}

fn launches_ms(rows: u32) -> f64 {
    launches(rows).as_secs_f64() * 1e3
}

/// A model whose draft block is right this often and whose text gives the
/// lookup nothing, so that every draft is the block's.
fn drafting_model(accuracy: &[f64], max_rows: u32) -> Arc<SyntheticModel> {
    model(|c| {
        never_repeating(c);
        c.own_drafts = 15;
        c.own_accuracy = accuracy.to_vec();
        c.max_rows = max_rows;
        c.context_length = 2048;
    })
}

/// A runtime on the device of [`launches`]. The executor tells the conductor
/// what each step cost and takes no time for it: the prices the policy sees
/// are exactly these, on a machine however busy, so its choices can be
/// checked to the row.
fn drafting_runtime(
    accuracy: &[f64],
    window: u32,
    warm_up: bool,
    steps: Option<&Steps>,
) -> (Runtime, Arc<SyntheticModel>) {
    let model = drafting_model(accuracy, 64);
    let observer = steps.map(watch);
    let config = RuntimeConfig { window_rows: window, warm_up, ..config(true, 4096) };
    let rt = runtime_with(&model, config, |e| {
        let e = e
            .with_step_cost(launches)
            .with_link_cost(Duration::from_micros(300))
            .without_sleeping();
        match observer {
            Some(observer) => e.with_observer(observer),
            None => e,
        }
    });
    (rt, model)
}

/// The windows the policy made, by the number of drafts they verified: `[source][drafts]`.
fn lengths(rt: &Runtime) -> [[u64; 17]; 2] {
    snapshot_of(rt).counters.draft_lengths
}

fn snapshot_of(rt: &Runtime) -> ares_runtime::Snapshot {
    rt.snapshot().expect("the conductor runs")
}

#[test]
fn a_model_with_no_drafter_does_not_speculate_and_the_lookup_does_not_run_alone() {
    // Text that repeats exactly, which the lookup would copy.
    let model = model(|c| {
        c.context.window = 1;
        c.context.epoch = 0;
        c.context.buckets = 512;
    });
    let rt = runtime_with(&model, RuntimeConfig { window_rows: 8, ..config(true, 4096) }, |e| e);
    alone(&rt, "a text that loops", Some(300));
    let c = snapshot_of(&rt).counters;
    assert_eq!((c.draft_rows, c.drafts_accepted), (0, 0), "{c:?}");
    // A token a row: the first came from the prompt's last row.
    assert_eq!(c.tokens, c.window_rows + 1, "{c:?}");
}

#[test]
fn a_window_of_one_row_is_no_speculation_even_for_a_model_with_a_drafter_and_text_that_repeats() {
    // What `--drafters none` makes of a model whose file has a draft block.
    let model = model(repeating);
    let rt = runtime(&model, false);
    alone(&rt, "a text that loops", Some(300));
    let c = snapshot_of(&rt).counters;
    assert_eq!((c.draft_rows, c.drafts_accepted), (0, 0), "{c:?}");
    // The same model and text with room for a window do draft: it is the room
    // that was missing, not the text.
    let rt = runtime_with(&model, config(true, 4096), |e| {
        e.with_delay(Duration::from_micros(200)).without_sleeping()
    });
    alone(&rt, "a text that loops", Some(300));
    assert!(snapshot_of(&rt).counters.drafts_accepted > 100);
}

#[test]
fn the_lookup_joins_the_model_drafter_when_there_is_one() {
    let model = model(never_repeating);
    // A step costs the same whatever it carries, and the executor says so.
    let rt = runtime_with(&model, RuntimeConfig { window_rows: 8, ..config(true, 4096) }, |e| {
        e.with_delay(Duration::from_micros(200)).without_sleeping()
    });
    alone(&rt, "written twice", Some(200));
    let before = snapshot_of(&rt).counters;
    alone(&rt, "written twice", Some(200));
    let c = snapshot_of(&rt).counters;
    // The draft block is never right here: what is drafted and accepted
    // comes from the lookup in the first request's text.
    assert!(c.drafts_accepted_at[0].iter().sum::<u64>() > 150, "{c:?}");
    assert_eq!(c.drafts_accepted_at[1].iter().sum::<u64>(), 0, "{c:?}");
    assert!(c.steps - before.steps < 80, "{c:?}");
}

#[test]
fn drafts_are_as_long_as_they_pay_and_stop_at_the_edge_of_a_launch() {
    // A draft block that is right nearly always: seven links fill the eight
    // rows of a launch, and a second launch costs more than the rows after it
    // are worth.
    let steps: Steps = Arc::default();
    let (rt, _) = drafting_runtime(&[0.97], 16, true, Some(&steps));
    let warm_up = steps.lock().unwrap().len();
    alone(&rt, "a long and easy text", Some(400));
    let windows: Vec<u32> =
        steps.lock().unwrap()[warm_up..].iter().filter(|s| s.windows > 0).map(|s| s.rows).collect();
    // The estimate starts from the belief and grows with every draft that is
    // accepted, and so do the windows, to the eight rows of a launch and
    // never to a ninth. From there every window is exactly the launch, up to
    // the last, which the end of the request may cut short.
    assert!(windows.iter().all(|rows| *rows <= 8), "{windows:?}");
    let full = windows.iter().position(|rows| *rows == 8).expect("a window of a whole launch");
    assert!(full <= 12, "the policy took {full} windows to learn: {windows:?}");
    let settled = &windows[full..windows.len() - 1];
    assert!(settled.len() >= 40 && settled.iter().all(|rows| *rows == 8), "{windows:?}");
    let l = lengths(&rt)[1];
    assert!(l[8..].iter().all(|n| *n == 0), "{l:?}");

    // One that is right a third of the time stops sooner, from the first
    // window on: no window verifies more than three drafts.
    let (rt, _) = drafting_runtime(&[0.33], 16, true, None);
    alone(&rt, "a hard text", Some(400));
    let l = lengths(&rt)[1];
    let windows: u64 = l.iter().sum();
    // (A size that has not run for long is tried again, twice at most: the
    // longer windows are those trials and no more.)
    assert!(windows > 100 && l[4..].iter().all(|n| *n <= 2), "{l:?}");
    assert!(l[4..].iter().sum::<u64>() * 20 < windows, "{l:?}");

    // One that is never right is tried a few times and then hardly at all, as
    // in the scheduler's test of the back-off with the same prices.
    let (rt, _) = drafting_runtime(&[0.0], 16, true, None);
    alone(&rt, "an impossible text", Some(400));
    let c = snapshot_of(&rt).counters;
    let tries: u64 = c.draft_lengths[1].iter().sum();
    assert!((2..=20).contains(&tries), "{tries} windows with drafts: {c:?}");
    assert_eq!(c.drafts_accepted, 0, "{c:?}");
}

/// The rows of the windows the steps after the first `from` made.
fn window_rows(steps: &Steps, from: usize) -> Vec<u32> {
    steps.lock().unwrap()[from..].iter().filter(|s| s.windows > 0).map(|s| s.rows).collect()
}

#[test]
fn a_late_step_does_not_price_a_window_out() {
    // The device of `launches`, on a machine that now and then does something
    // else: the next `late` steps take a hundred times what they cost.
    let late = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let steps: Steps = Arc::default();
    let model = drafting_model(&[0.97], 64);
    let config = RuntimeConfig { window_rows: 16, warm_up: true, ..config(true, 4096) };
    let rt = runtime_with(&model, config, |e| {
        let late = late.clone();
        e.with_step_cost(move |rows| {
            let hit =
                late.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1)).is_ok();
            launches(rows) * if hit { 100 } else { 1 }
        })
        .with_link_cost(Duration::from_micros(300))
        .without_sleeping()
        .with_observer(watch(&steps))
    });
    // The policy settles on windows of the eight rows of a launch.
    alone(&rt, "a long and easy text", Some(200));
    let settled = steps.lock().unwrap().len();
    assert_eq!(window_rows(&steps, settled - 10)[..8], [8; 8]);
    // A step of eight rows takes 300 ms where it costs 3: the price of eight
    // rows is the mean of its last eight steps without the largest, so it
    // has not moved, and every window after it is eight rows as before.
    late.store(1, Ordering::Relaxed);
    alone(&rt, "another long and easy text", Some(200));
    assert_eq!(late.load(Ordering::Relaxed), 0);
    let after = window_rows(&steps, settled);
    assert!(
        after.len() > 20 && after[..after.len() - 1].iter().all(|rows| *rows == 8),
        "{after:?}"
    );
    let now = snapshot_of(&rt).step_ms;
    let eight = now.iter().find(|(rows, _)| *rows == 8).map(|(_, ms)| *ms).expect("measured");
    assert!((eight - launches_ms(8)).abs() < 1e-9, "{now:?}");
}

#[test]
fn windows_priced_out_on_a_cold_start_are_tried_again_and_chosen_when_they_pay() {
    // A device whose rows are dear while it is cold (9 ms a row on top of
    // 3 ms: no draft pays for its row) and cost what `launches` says once it
    // is warm. The warm-up measures it cold.
    let warm = Arc::new(AtomicBool::new(false));
    let steps: Steps = Arc::default();
    let model = drafting_model(&[0.97], 64);
    let config = RuntimeConfig { window_rows: 8, warm_up: true, ..config(true, 4096) };
    let rt = runtime_with(&model, config, |e| {
        let warm = warm.clone();
        e.with_step_cost(move |rows| {
            if warm.load(Ordering::Relaxed) {
                launches(rows)
            } else {
                Duration::from_millis(3 + 9 * u64::from(rows))
            }
        })
        .with_link_cost(Duration::from_micros(300))
        .without_sleeping()
        .with_observer(watch(&steps))
    });
    let cold = snapshot_of(&rt).calibration.expect("a warm-up").step_ms;
    assert_eq!(cold[..3], [(1, 12.0), (2, 21.0), (3, 30.0)]);
    let warm_up = steps.lock().unwrap().len();
    warm.store(true, Ordering::Relaxed);
    alone(&rt, "a long and easy text", Some(700));
    // For as long as the windows' prices are current nothing is drafted: a
    // row that costs 9 ms is not worth a draft. After 256 steps they are old
    // and tried at the price of the one row that runs, and from the first
    // trial on the windows are measured warm and chosen: the whole launch,
    // once the drafts have shown how good they are.
    let seen: Vec<Seen> = steps.lock().unwrap()[warm_up..].to_vec();
    let first = seen.iter().position(|s| s.windows > 0).expect("a window");
    assert_eq!(first, ares_sched::PRICE_OLD as usize + 1, "the first window");
    let rows: Vec<u32> = seen[first..].iter().map(|s| s.rows).collect();
    let full = rows.iter().position(|rows| *rows == 8).expect("a window of a whole launch");
    assert!(full <= 12, "{rows:?}");
    assert!(rows[full..rows.len() - 1].iter().all(|rows| *rows == 8), "{rows:?}");
    // And the table says what the device costs now.
    let now = snapshot_of(&rt).step_ms;
    for rows in [1, 4, 8] {
        let price = now.iter().find(|(n, _)| *n == rows).map(|(_, ms)| *ms).expect("measured");
        assert!((price - launches_ms(rows)).abs() < 1e-9, "{rows} rows: {now:?}");
    }
}

/// A model whose draft block is right six times in ten at every link and
/// says so: 0.95 for a draft that is right and 0.15 for one that is wrong
/// (`sure`), or nothing.
fn unsure_drafts(sure: bool) -> (Run, ares_runtime::Snapshot) {
    let model = model(|c| {
        never_repeating(c);
        c.own_drafts = 7;
        c.own_accuracy = vec![0.6];
        c.own_probability = sure.then_some((0.95, 0.15));
        c.context_length = 2048;
    });
    // A step of 3 ms and 0.03 ms a row, a link of 0.3 ms: the chain is what
    // a window costs.
    let config = RuntimeConfig { window_rows: 8, warm_up: true, ..config(true, 4096) };
    let rt = runtime_with(&model, config, |e| {
        e.with_step_cost(launches).with_link_cost(Duration::from_micros(300)).without_sleeping()
    });
    let run = alone(&rt, "a text that is hard in places", Some(600));
    (run, snapshot_of(&rt))
}

#[test]
fn a_chain_ends_at_the_first_draft_its_drafter_is_unsure_of() {
    let (plain, without) = unsure_drafts(false);
    let (cut, with) = unsure_drafts(true);
    assert_eq!(cut.text, plain.text, "the windows decide speed, not the text");
    let (c, p) = (with.counters, without.counters);
    // The model's own drafts that were made, and those of them that were
    // accepted.
    let made = |c: &ares_runtime::Counters| {
        c.draft_lengths[1].iter().enumerate().map(|(len, n)| len as u64 * n).sum::<u64>()
    };
    let accepted = |c: &ares_runtime::Counters| c.drafts_accepted_at[1].iter().sum::<u64>();
    // A draft given 0.15 starts in the range whose chance is 2.5 in 16, and
    // a link pays when the chain reaches it with more than that (0.55 tokens
    // per millisecond, 0.3 ms, drafts right 0.6 of the time: 0.28): the chain
    // ends behind its first wrong draft, from the first window on. So a
    // window has one draft that is not accepted at most, every draft given
    // 0.95 was accepted and none given 0.15.
    let windows: u64 = c.draft_lengths[1].iter().sum();
    let wasted = made(&c) - accepted(&c);
    assert!(windows > 100 && wasted <= windows, "{c:?}");
    let (sure, unsure) = (15, 2);
    assert_eq!(c.own_tested_by_p[sure], c.own_accepted_by_p[sure]);
    assert_eq!((c.own_tested_by_p[unsure], c.own_accepted_by_p[unsure]), (wasted, 0));
    assert_eq!(accepted(&c), c.own_accepted_by_p[sure]);
    assert!(c.own_links_cut > 0, "{c:?}");
    // The drafter that says nothing makes every link it is asked for, so a
    // link is asked for only where the average draft pays for it; the one
    // that ends its chains itself is asked for more, makes them where its
    // drafts are sure, and gets more tokens out of a step.
    assert_eq!((p.own_links_cut, p.own_tested_by_p), (0, [0; 16]));
    let per_step = |c: &ares_runtime::Counters| c.tokens as f64 / c.steps as f64;
    eprintln!("RES tokens per step {:.3} with the cut, {:.3} without", per_step(&c), per_step(&p));
    assert!(per_step(&c) > 1.1 * per_step(&p), "{} {}", per_step(&c), per_step(&p));
    // What a step costs is known exactly here (3 ms, 0.03 ms a row, and the
    // chain): the same tokens take less time.
    let ms = |c: &ares_runtime::Counters| {
        c.steps as f64 * 3.0 + c.rows as f64 * 0.03 + c.own_draft_us as f64 / 1e3
    };
    assert_eq!(c.tokens, p.tokens);
    eprintln!("RES {:.1} ms with the cut, {:.1} ms without", ms(&c), ms(&p));
    assert!(ms(&c) < 0.95 * ms(&p), "{} ms with the cut, {} without", ms(&c), ms(&p));
}

#[test]
fn the_window_is_the_room_and_not_the_length() {
    let (rt, _) = drafting_runtime(&[0.97], 4, true, None);
    alone(&rt, "a long and easy text", Some(300));
    let l = lengths(&rt)[1];
    // Three drafts fit and three is what nearly every window takes; none has
    // more.
    let windows: u64 = l.iter().sum();
    assert!(l[3] > 60 && l[3] + 3 >= windows, "{l:?}");
    assert!(l[4..].iter().all(|n| *n == 0), "{l:?}");
}

#[test]
fn the_warm_up_measures_every_window_size_and_every_link_of_the_chain() {
    let steps: Steps = Arc::default();
    let (rt, model) = drafting_runtime(&[0.97], 8, true, Some(&steps));
    drop(rt);
    let seen = steps.lock().unwrap().clone();
    // The rows of a window, one to the most a step with drafts has, and a few
    // above up to a piece of prompt.
    let rows: std::collections::BTreeSet<u32> = seen.iter().map(|s| s.rows).collect();
    assert!((1..=model.max_rows().min(24)).all(|n| rows.contains(&n)), "{rows:?}");
    assert!(rows.contains(&32) && rows.contains(&model.prompt_rows()), "{rows:?}");
    // And chains of 1 to 7 links: windows of 2 to 8 rows whose drafts the
    // model fills in.
    assert!(seen.iter().filter(|s| s.windows > 0).count() >= 7, "{seen:?}");
}

#[test]
fn the_calibration_finds_the_edge_of_a_launch_and_the_cost_of_a_link() {
    let (rt, _) = drafting_runtime(&[0.97], 8, true, None);
    let calibration = snapshot_of(&rt).calibration.expect("the warm-up measured");
    // The table holds what the device charges for every size, to the edge:
    // eight rows are one launch and nine are two, 3 ms more.
    assert!(calibration.step_ms.len() >= 24, "{:?}", calibration.step_ms);
    for (rows, ms) in &calibration.step_ms {
        assert!((ms - launches_ms(*rows)).abs() < 1e-9, "{rows} rows: {ms} ms");
    }
    let step = |rows: u32| calibration.step_ms.iter().find(|(n, _)| *n == rows).unwrap().1;
    assert!((step(9) - step(8) - 3.03).abs() < 1e-9, "{:?}", calibration.step_ms);
    // A link of the chain costs 0.3 ms, however long the chain.
    assert_eq!(calibration.chain_ms.len(), 7, "{:?}", calibration.chain_ms);
    assert!(
        calibration.chain_ms.iter().all(|(links, ms)| (ms / f64::from(*links) - 0.3).abs() < 1e-9)
    );
    assert!(calibration.steps > 30 && calibration.seconds > 0.0, "{calibration:?}");
}

#[test]
fn the_conductor_measures_the_steps_of_an_executor_that_does_not_say_what_they_took() {
    // What a device does: the executor takes the time of a step and says
    // nothing, and the conductor measures the step with its own clock, at the
    // warm-up and for every step after it. On a busy machine such a
    // measurement is anything from the time the step took upwards, so this
    // test asserts what holds at any speed and leaves the choices that follow
    // from exact prices to the tests above.
    let model = drafting_model(&[0.97], 16);
    let start = |measured: bool| {
        let config = RuntimeConfig { window_rows: 8, warm_up: true, ..config(true, 4096) };
        runtime_with(&model, config, |e| {
            let e = e.with_step_cost(launches).with_link_cost(Duration::from_micros(300));
            if measured { e.timed_by_the_conductor() } else { e.without_sleeping() }
        })
    };
    let measured = start(true);
    let calibration = snapshot_of(&measured).calibration.expect("the warm-up measured");
    let sizes: Vec<u32> = calibration.step_ms.iter().map(|(rows, _)| *rows).collect();
    assert_eq!(sizes, (1..=16).collect::<Vec<_>>());
    // A step is never measured shorter than the executor took for it.
    for (rows, ms) in &calibration.step_ms {
        assert!(*ms >= launches_ms(*rows) - 1e-6, "{rows} rows in {ms} ms");
    }
    // The chain's time is the executor's word, here as on a device.
    assert!(
        calibration.chain_ms.iter().all(|(links, ms)| (ms / f64::from(*links) - 0.3).abs() < 1e-9),
        "{:?}",
        calibration.chain_ms
    );
    // Whatever the measured prices make of the windows, the text is the one
    // that exact prices give, and every token is counted.
    let run = alone(&measured, "the same words", Some(120));
    assert_eq!(run, alone(&start(false), "the same words", Some(120)));
    assert_eq!(run.usage.completion_tokens, 120);
    let c = snapshot_of(&measured).counters;
    assert_eq!(c.tokens, 120, "{c:?}");
    assert!(c.steps <= 121 && c.drafts_accepted <= c.draft_rows, "{c:?}");
}

#[test]
fn a_request_that_asks_for_no_drafts_gets_none_and_says_the_same() {
    let (rt, _) = drafting_runtime(&[0.97], 8, false, None);
    let with = alone(&rt, "the same words", Some(120));
    let drafted = snapshot_of(&rt).counters.draft_rows;
    assert!(drafted > 0);
    let mut plain = request("the same words", Some(120));
    plain.draft = false;
    let without = run_one(&rt, plain);
    assert_eq!(snapshot_of(&rt).counters.draft_rows, drafted, "no draft rows for the request");
    assert_eq!(without, with, "the text, and the hash of its tokens, are the same");
    assert_ne!(with.usage.output_hash, 0);
}

#[test]
fn a_window_of_no_rows_is_refused_at_start() {
    let model = model(|_| {});
    let host = HostExecutor::new(model.clone(), model.config().max_rows, 4);
    let config = RuntimeConfig { max_slots: 4, window_rows: 0, ..config(true, 64) };
    let error = Runtime::start(model.clone(), Box::new(host), config).unwrap_err();
    assert!(error.to_string().contains("at least one row"), "{error}");
}
