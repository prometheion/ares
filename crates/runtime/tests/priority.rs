//! Priorities of requests through the conductor, on the host executor: a
//! request of a better priority takes the place of a running one, a full line
//! takes it in place of a worse one, and a request that gave way says the
//! same as it says alone.

mod common;

use std::sync::Arc;
use std::time::Duration;

use ares_api::{EngineError, FinishReason};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use common::*;

fn model() -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 4096;
    Arc::new(SyntheticModel::new(config).unwrap())
}

fn runtime(model: &Arc<SyntheticModel>, slots: u32, queue: Option<u32>) -> Runtime {
    let config = RuntimeConfig {
        pages: 1024,
        cells_per_page: 16,
        max_slots: slots,
        queue,
        window_rows: 1,
        warm_up: false,
        ..RuntimeConfig::default()
    };
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_delay(Duration::from_micros(300));
    Runtime::start(model.clone(), Box::new(host), config).unwrap()
}

fn ranked(prompt: &str, max_tokens: u32, priority: i32) -> ares_api::GenerateRequest {
    ares_api::GenerateRequest { priority, ..request(prompt, Some(max_tokens)) }
}

#[test]
fn a_better_priority_takes_the_place_of_a_running_request_which_comes_back_the_same() {
    let model = model();
    let rt = runtime(&model, 1, None);
    let background = "a long answer in the background";
    let alone = alone(&rt, background, Some(600));

    // The one place is taken by a request of priority 5 that has much to
    // write. A request of the same priority waits for it to end; one of
    // priority 0 takes its place at once and ends first.
    let long = start(&rt, ranked(background, 600, 5)).unwrap();
    let head = long.wait_for_text(20);
    let before = snapshot(&rt).counters;
    let same = start(&rt, ranked("the same priority", 8, 5)).unwrap();
    let urgent = start(&rt, ranked("an urgent question", 8, 0)).unwrap();
    let urgent = urgent.finish();
    assert_eq!(urgent.finish, FinishReason::Length);
    let after = snapshot(&rt).counters;
    assert_eq!(after.requests_outranked - before.requests_outranked, 1, "{after:?}");
    assert_eq!(rt.sequences().unwrap().len(), 1, "the long one is back, and only it");

    // The long one comes back and says what it says alone, and the request
    // of its own priority starts only after it.
    let long = long.finish();
    assert_eq!(head + &long.text, alone.text);
    assert_eq!(long.usage.output_hash, alone.usage.output_hash);
    let same = same.finish();
    assert!(same.usage.queued.unwrap() > urgent.usage.queued.unwrap());
}

#[test]
fn a_full_line_takes_a_better_priority_in_place_of_a_worse_one() {
    let model = model();
    // One place and one more in line.
    let rt = runtime(&model, 1, Some(1));
    let running = start(&rt, ranked("the one that runs", 400, 0)).unwrap();
    running.wait_for_text(10);
    let waiting = start(&rt, ranked("waits in line", 8, 5)).unwrap();
    // The line is full: the same priority as the one in line is refused at
    // once, a better one takes its place.
    let refused = start(&rt, ranked("no room", 8, 5)).unwrap();
    assert_eq!(refused.failure(), EngineError::Overloaded);
    let better = start(&rt, ranked("a better one", 8, 1)).unwrap();
    assert_eq!(waiting.failure(), EngineError::Overloaded);
    // It does not take the place of the one that runs, whose priority is
    // better still.
    assert_eq!(running.finish().finish, FinishReason::Length);
    assert_eq!(better.finish().finish, FinishReason::Length);
    assert_eq!(snapshot(&rt).counters.requests_displaced, 1);
    // A line without a request it may take the place of refuses.
    let running = start(&rt, ranked("the one that runs", 400, 0)).unwrap();
    running.wait_for_text(10);
    let first = start(&rt, ranked("waits in line", 8, -1)).unwrap();
    let refused = start(&rt, ranked("no room", 8, 0)).unwrap();
    assert_eq!(refused.failure(), EngineError::Overloaded);
    drop(first);
    drop(running);
}
