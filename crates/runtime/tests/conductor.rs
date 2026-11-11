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
use ares_runtime::{Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
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
        speculation,
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

/// A model whose text loops and changes its loop every 40 positions.
fn repeating(c: &mut SyntheticConfig) {
    c.context.window = 1;
    c.context.epoch = 40;
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
    let model = model(|_| {});
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
    // A model whose text never repeats inside a request: a draft can only
    // come from what another request read and wrote.
    let model = model(|_| {});
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
    let model = model(|_| {});
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
        c.context.window = 1;
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
    let model = model(|_| {});
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
    let model = model(|_| {});
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
    let model = model(|_| {});
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
    let model = model(|c| c.max_rows = 64);

    // Reference: the long prompt in few big steps, alone.
    let wide = runtime_with(&model, RuntimeConfig { max_rows: 0, ..config(false, 4096) }, |e| e);
    let reference = alone(&wide, &long, Some(30));
    drop(wide);

    // A generating request is running when the long prompt arrives, and the
    // steps are small: the prompt must go through next to generation.
    let rt = runtime_with(&model, RuntimeConfig { max_rows: 16, ..config(false, 4096) }, |e| {
        e.with_delay(Duration::from_micros(300))
    });
    let generating = start(&rt, request("already generating", Some(200))).unwrap();
    generating.wait_for_text(5);
    let before = snapshot(&rt);
    let run = run_one(&rt, request(&long, Some(30)));
    assert_eq!(run, reference, "chunking the prompt must not change the answer");
    let d = since(&snapshot(&rt), &before);
    assert!(d["mixed"] >= 3, "the prompt shared steps with the generating request: {d:?}");
    assert!(d["steps"] >= 300 / 16, "{d:?}");
    generating.finish();
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
    let handles: Vec<Handle> =
        (0..4).map(|i| start(&rt, request(&format!("doomed {i}"), None)).unwrap()).collect();
    for handle in handles {
        let error = handle.failure();
        assert!(matches!(&error, EngineError::Internal(m) if m.contains("injected")), "{error}");
    }
    let later = start(&rt, request("after", Some(1))).unwrap().failure();
    assert!(matches!(later, EngineError::Internal(_)), "{later}");
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
        c.context.window = 1;
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
