//! What the conductor tests and the GPU tests share: the ways to drive a
//! runtime, and the scenarios that must hold whatever executes the steps.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use ares_api::{
    Engine, EngineError, Event, EventSink, FinishReason, GenerateRequest, Prompt, RequestId,
    Sampling, TokenProbs, Usage,
};
use ares_runtime::{Runtime, Snapshot, TokenFilter};

/// How long a test waits for the engine before it fails, so that a broken
/// engine ends a test instead of hanging it.
pub const LIMIT: Duration = Duration::from_secs(60);

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// An id no other request in this process has.
pub fn fresh_id() -> RequestId {
    RequestId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

/// A small deterministic generator, so failures reproduce.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Printable ASCII, `len` characters.
    pub fn prompt(&mut self, len: usize) -> String {
        (0..len).map(|_| char::from(b' ' + self.below(95) as u8)).collect()
    }
}

pub fn request(prompt: &str, max_tokens: Option<u32>) -> GenerateRequest {
    GenerateRequest {
        model: "synthetic".to_owned(),
        prompt: Prompt::Text(prompt.to_owned()),
        sampling: Sampling::default(),
        max_tokens,
        stop: Vec::new(),
        draft: true,
        structure: None,
        id: fresh_id(),
    }
}

/// What a request produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub text: String,
    pub finish: FinishReason,
    pub usage: Usage,
    /// The probabilities reported with the text, in order.
    pub probs: Vec<TokenProbs>,
}

/// The receiving end of a request.
#[derive(Debug)]
pub struct Handle {
    pub id: RequestId,
    rx: Receiver<Event>,
}

impl Handle {
    /// Reads events until the terminal one, checking the contract on the
    /// way: text is never empty, exactly one terminal event ends the stream,
    /// and nothing follows it.
    pub fn finish(self) -> Run {
        let deadline = Instant::now() + LIMIT;
        let mut text = String::new();
        let mut probs = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(left) {
                Ok(Event::Text(piece)) => {
                    assert!(!piece.is_empty(), "an empty text event");
                    text.push_str(&piece);
                }
                Ok(Event::Logprobs(more)) => {
                    assert!(!more.is_empty(), "an empty event of probabilities");
                    probs.extend(more);
                }
                Ok(Event::Done { finish, usage }) => {
                    // The sink is closed after the terminal event, so the
                    // channel's sender is dropped with it.
                    assert!(
                        self.rx.recv_timeout(Duration::from_millis(50)).is_err(),
                        "events after the terminal one"
                    );
                    return Run { text, finish, usage, probs };
                }
                Ok(Event::Error(error)) => panic!("request {} failed: {error}", self.id),
                Err(_) => panic!("request {} did not end within {LIMIT:?}", self.id),
            }
        }
    }

    /// Like [`Handle::finish`] for a request that is expected to fail.
    pub fn failure(self) -> EngineError {
        let deadline = Instant::now() + LIMIT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(left) {
                Ok(Event::Text(_) | Event::Logprobs(_)) => {}
                Ok(Event::Done { .. }) => panic!("request {} did not fail", self.id),
                Ok(Event::Error(error)) => return error,
                Err(_) => panic!("request {} did not end within {LIMIT:?}", self.id),
            }
        }
    }

    /// Waits for `n` characters of text without ending the request.
    pub fn wait_for_text(&self, n: usize) -> String {
        let deadline = Instant::now() + LIMIT;
        let mut text = String::new();
        while text.chars().count() < n {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(left) {
                Ok(Event::Text(piece)) => text.push_str(&piece),
                other => panic!("request {} gave {other:?} before {n} characters", self.id),
            }
        }
        text
    }
}

pub fn start(rt: &Runtime, request: GenerateRequest) -> Result<Handle, EngineError> {
    let id = request.id;
    let (tx, rx) = mpsc::channel();
    rt.submit(request, EventSink::new(move |event| tx.send(event).is_ok()))?;
    Ok(Handle { id, rx })
}

/// Starts a request whose events `seen` looks at before the test gets them,
/// on the thread that sends them: the conductor's, after the step whose text
/// they carry and before the next.
pub fn start_seen(
    rt: &Runtime,
    request: GenerateRequest,
    seen: impl Fn(&Event) + Send + Sync + 'static,
) -> Result<Handle, EngineError> {
    let id = request.id;
    let (tx, rx) = mpsc::channel();
    let sink = EventSink::new(move |event| {
        seen(&event);
        tx.send(event).is_ok()
    });
    rt.submit(request, sink)?;
    Ok(Handle { id, rx })
}

/// Starts a request whose picks `filter` restricts.
pub fn start_filtered(
    rt: &Runtime,
    request: GenerateRequest,
    filter: impl TokenFilter + 'static,
) -> Result<Handle, EngineError> {
    let id = request.id;
    let (tx, rx) = mpsc::channel();
    rt.submit_filtered(
        request,
        EventSink::new(move |event| tx.send(event).is_ok()),
        Box::new(filter),
    )?;
    Ok(Handle { id, rx })
}

pub fn run_one(rt: &Runtime, request: GenerateRequest) -> Run {
    start(rt, request).expect("accepted").finish()
}

/// Runs the request alone: the engine has nothing else to do.
pub fn alone(rt: &Runtime, prompt: &str, max_tokens: Option<u32>) -> Run {
    run_one(rt, request(prompt, max_tokens))
}

/// Waits until `check` holds for the running sequences.
pub fn wait_until(what: &str, rt: &Runtime, check: impl Fn(&[ares_runtime::SequenceInfo]) -> bool) {
    let deadline = Instant::now() + LIMIT;
    loop {
        let sequences = rt.sequences().expect("the conductor answers");
        if check(&sequences) {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting until {what}");
        thread::sleep(Duration::from_millis(2));
    }
}

pub fn snapshot(rt: &Runtime) -> Snapshot {
    rt.snapshot().expect("the conductor answers")
}

// --- scenarios -------------------------------------------------------------

/// Prompts of different lengths, and how many tokens each asks for.
fn workload(rng: &mut Rng, n: usize) -> Vec<(String, u32)> {
    (0..n)
        .map(|_| {
            let len = 1 + rng.below(40) as usize;
            (rng.prompt(len), 8 + rng.below(40) as u32)
        })
        .collect()
}

/// A request's output does not depend on what else is in the engine: the same
/// whether it runs alone or next to 1, 7 or 31 others, started at different
/// times, with prompts of different lengths.
pub fn output_is_independent_of_the_batch(rt: &Runtime, stagger: Duration) {
    let mut rng = Rng(0x1234_5678);
    let jobs = workload(&mut rng, 32);
    let expected: Vec<Run> = jobs.iter().map(|(p, m)| alone(rt, p, Some(*m))).collect();
    for (job, run) in jobs.iter().zip(&expected) {
        assert_eq!(run.usage.completion_tokens, job.1, "{run:?}");
        assert_eq!(run.finish, FinishReason::Length);
        assert_eq!(run.text.chars().count() as u32, job.1);
    }
    for neighbours in [1usize, 7, 31] {
        let before = snapshot(rt).counters;
        // Request 0 is the one under test; the others come before and after it.
        let group = neighbours + 1;
        let handles: Vec<(usize, Handle)> = (0..group)
            .map(|i| {
                if !stagger.is_zero() {
                    thread::sleep(stagger * (rng.below(3) as u32));
                }
                let (prompt, max) = &jobs[i];
                (i, start(rt, request(prompt, Some(*max))).expect("accepted"))
            })
            .collect();
        for (i, handle) in handles {
            assert_eq!(handle.finish(), expected[i], "request {i} next to {neighbours} others");
        }
        let after = snapshot(rt).counters;
        let (steps, rows) = (after.steps - before.steps, after.rows - before.rows);
        assert!(
            rows as f64 / steps as f64 > 1.2,
            "the requests did not share steps ({rows} rows in {steps} steps)"
        );
    }
}

/// Speculation changes how many steps a request takes and not what it says.
/// Returns the outputs, to compare runs of two runtimes.
pub fn outputs_of(rt: &Runtime, seed: u64, n: usize) -> Vec<Run> {
    let mut rng = Rng(seed);
    let jobs: Vec<(String, u32)> = (0..n)
        .map(|_| {
            let len = 3 + rng.below(30) as usize;
            (rng.prompt(len), 60 + rng.below(120) as u32)
        })
        .collect();
    let handles: Vec<Handle> =
        jobs.iter().map(|(p, m)| start(rt, request(p, Some(*m))).expect("accepted")).collect();
    handles.into_iter().map(Handle::finish).collect()
}

/// Cancelling frees the pages at once and leaves the others alone.
pub fn cancel_frees_pages_and_spares_the_others(rt: &Runtime) {
    let keep = "stay and finish";
    let expected = alone(rt, keep, Some(150));
    let total = snapshot(rt).pages_capacity;

    let survivor = start(rt, request(keep, Some(150))).expect("accepted");
    let doomed = start(rt, request("this one is cancelled", None)).expect("accepted");
    let other_doomed = start(rt, request("and so is this one", None)).expect("accepted");
    wait_until("three sequences are running", rt, |s| {
        s.len() == 3 && s.iter().all(|q| q.cells > 4)
    });
    let held: u32 = rt.sequences().unwrap().iter().map(|s| s.pages.len() as u32).sum();
    assert!(held >= 3);
    assert!(snapshot(rt).pages_free <= total - held);

    rt.cancel(doomed.id);
    rt.cancel(other_doomed.id);
    // Commands are handled in order, so a question asked after the cancels
    // sees their effect: the pages are back and the sequences are gone.
    let sequences = rt.sequences().unwrap();
    assert!(
        sequences.iter().all(|s| s.id != doomed.id && s.id != other_doomed.id),
        "{sequences:?}"
    );

    let run = doomed.finish();
    assert_eq!(run.finish, FinishReason::Cancelled);
    assert!(run.usage.completion_tokens > 0, "it had produced something: {run:?}");
    assert_eq!(run.usage.prompt_tokens, "this one is cancelled".len() as u32);
    assert_eq!(other_doomed.finish().finish, FinishReason::Cancelled);
    assert_eq!(survivor.finish(), expected, "the other request was not disturbed");
    wait_until("everything is idle", rt, |s| s.is_empty());
    assert_eq!(snapshot(rt).pages_free, total, "every page is back");
    // Cancelling something that is over, or never was, does nothing.
    rt.cancel(doomed_id_again());
}

fn doomed_id_again() -> RequestId {
    RequestId(u64::MAX)
}

/// More requests than the pool holds at once: none is refused, the ones that
/// do not fit wait in line, and every one says what it says alone. Every
/// request has a prompt of 40 tokens and writes 100; `fit` is the most of
/// them the pool can run next to each other.
pub fn more_requests_than_fit_wait_and_say_the_same(rt: &Runtime, fit: usize) {
    let mut rng = Rng(0x0dd_ba11);
    let jobs: Vec<(String, u32)> = (0..3 * fit + 2).map(|_| (rng.prompt(40), 100)).collect();
    let expected: Vec<Run> = jobs.iter().map(|(p, m)| alone(rt, p, Some(*m))).collect();
    let before = snapshot(rt).counters;
    let handles: Vec<Handle> = jobs
        .iter()
        .map(|(prompt, max)| start(rt, request(prompt, Some(*max))).expect("nothing is refused"))
        .collect();
    let mut most = 0;
    let mut in_line = 0;
    let deadline = Instant::now() + LIMIT;
    loop {
        let now = snapshot(rt);
        most = most.max(now.active);
        in_line = in_line.max(now.waiting);
        if now.active == 0 && now.waiting == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "the line never emptied");
        thread::sleep(Duration::from_micros(200));
    }
    for (i, handle) in handles.into_iter().enumerate() {
        assert_eq!(handle.finish(), expected[i], "request {i}");
    }
    let after = snapshot(rt);
    assert!(most as usize <= fit, "{most} requests ran at once in a pool for {fit}");
    assert!(in_line >= 1, "nobody was seen waiting");
    let waited = after.counters.requests_waited - before.requests_waited;
    assert!(waited as usize >= jobs.len() - fit - 1, "{waited} of {} requests waited", jobs.len());
    assert_eq!(after.counters.requests_suspended, before.requests_suspended, "nobody gave way");
    assert_eq!(after.pages_free, after.pages_capacity, "every page is back");
}

/// Requests that write more than the accounting expected of them: when the
/// pool is full the one with the fewest tokens gives way, comes back when
/// there is room and still says what it says alone. Needs a runtime whose
/// `output_ratio` is far below what requests really use, and a pool that
/// holds two of these requests and not three.
pub fn a_request_that_gives_way_comes_back_and_says_the_same(rt: &Runtime) {
    let jobs: Vec<(String, u32)> =
        (0..4).map(|i| (format!("writes more than expected {i}"), 180)).collect();
    let expected: Vec<Run> = {
        // Alone they teach the accounting that requests use what they may;
        // it is told otherwise again below, by requests that end at once.
        jobs.iter().map(|(p, m)| alone(rt, p, Some(*m))).collect()
    };
    for i in 0..60 {
        let mut short = request(&format!("ends early {i}"), Some(200));
        // Every character the model can write is a stop string: the
        // request ends with its first token.
        short.stop = (0..=255u8).map(|b| ares_runtime::printable(b).to_string()).collect();
        run_one(rt, short);
    }
    let before = snapshot(rt).counters;
    let handles: Vec<Handle> = jobs
        .iter()
        .map(|(prompt, max)| start(rt, request(prompt, Some(*max))).expect("nothing is refused"))
        .collect();
    for (i, handle) in handles.into_iter().enumerate() {
        assert_eq!(handle.finish(), expected[i], "request {i}");
    }
    wait_until("everything is idle", rt, |s| s.is_empty());
    let after = snapshot(rt);
    let gave_way = after.counters.requests_suspended - before.requests_suspended;
    assert!(gave_way >= 1, "the pool never ran short: {:?}", after.counters);
    assert_eq!(after.pages_free, after.pages_capacity, "every page is back");
    assert_eq!(after.waiting, 0);
}

/// A request alone in a pool that is smaller than what it may write runs to
/// the last cell of the pool and ends there, as at the end of a context.
/// `pool_cells` is the pool's size. Returns what the request wrote.
pub fn the_last_request_runs_to_the_end_of_the_pool(
    rt: &Runtime,
    prompt: &str,
    pool_cells: u32,
) -> Run {
    let before = snapshot(rt).counters;
    let run = alone(rt, prompt, None);
    assert_eq!(run.finish, FinishReason::Length);
    // Every cell holds a token's row, and the last token picked has none.
    assert_eq!(run.usage.prompt_tokens + run.usage.completion_tokens, pool_cells + 1, "{run:?}");
    assert_eq!(run.text.chars().count() as u32, run.usage.completion_tokens);
    let after = snapshot(rt);
    assert_eq!(after.counters.requests_exhausted - before.requests_exhausted, 1);
    assert_eq!(after.pages_free, after.pages_capacity);
    // The engine goes on.
    assert_eq!(alone(rt, "after it", Some(5)).usage.completion_tokens, 5);
    run
}

/// Shutting down with requests in flight ends each with a terminal event,
/// and the threads are gone when `shutdown` returns.
pub fn shutdown_ends_everything(rt: &Runtime) {
    let handles: Vec<Handle> = (0..12)
        .map(|i| start(rt, request(&format!("in flight {i}"), None)).expect("room"))
        .collect();
    for handle in &handles {
        handle.wait_for_text(3);
    }
    rt.shutdown();
    for handle in handles {
        let run = handle.finish();
        assert_eq!(run.finish, FinishReason::Cancelled);
        assert!(run.usage.completion_tokens >= 3);
    }
    assert!(!rt.is_running(), "the conductor is still alive");
    // Nothing is accepted afterwards, and shutting down again is harmless.
    assert!(matches!(start(rt, request("late", Some(1))), Err(EngineError::Internal(_))));
    rt.cancel(RequestId(1));
    rt.shutdown();
    assert!(rt.snapshot().is_none());
}

/// The names of this process's engine threads that are still alive. Tests run
/// one after the other, so after a shutdown there are none.
pub fn engine_threads() -> Vec<String> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir("/proc/self/task").into_iter().flatten().flatten() {
        if let Ok(name) = std::fs::read_to_string(entry.path().join("comm")) {
            let name = name.trim().to_owned();
            // The kernel keeps the first 15 characters of a thread's name.
            if name.starts_with("ares-conduct") || name.starts_with("ares-device") {
                names.push(name);
            }
        }
    }
    names
}

/// Bad requests are refused at once, and nothing was started.
pub fn bad_requests_are_refused_at_once(rt: &Runtime, context_length: u32) {
    let wrong_model = GenerateRequest { model: "other".into(), ..request("x", Some(1)) };
    assert!(matches!(start(rt, wrong_model), Err(EngineError::ModelNotFound(m)) if m == "other"));

    let sampling = Sampling { temperature: Some(f32::NAN), ..Sampling::default() };
    let bad_sampling = GenerateRequest { sampling, ..request("x", Some(1)) };
    assert!(
        matches!(start(rt, bad_sampling), Err(EngineError::InvalidRequest(m)) if m.contains("temperature"))
    );

    assert!(matches!(start(rt, request("x", Some(0))), Err(EngineError::InvalidRequest(_))));
    assert!(matches!(start(rt, request("", Some(1))), Err(EngineError::InvalidRequest(_))));

    let long = "y".repeat(context_length as usize);
    assert_eq!(
        start(rt, request(&long, Some(1))).unwrap_err(),
        EngineError::ContextOverflow { prompt_tokens: context_length, limit: context_length }
    );
    // One short of the context leaves room for exactly one token.
    let fits = "y".repeat(context_length as usize - 1);
    let run = run_one(rt, request(&fits, None));
    assert_eq!((run.finish, run.usage.completion_tokens), (FinishReason::Length, 1));
    assert_eq!(snapshot(rt).pages_free, snapshot(rt).pages_capacity);
}

/// `max_tokens`, the end of the context and stop strings end a request with
/// the right reason and the right counts.
pub fn requests_end_for_the_right_reason(rt: &Runtime, context_length: u32) {
    let run = alone(rt, "count me", Some(25));
    // The hash of the reply is the hash of its tokens.
    let tokens: Vec<u32> = run
        .text
        .chars()
        .map(|c| u32::from(ares_runtime::unprintable(c).expect("a printable token")))
        .collect();
    assert_eq!(
        (run.finish, run.usage),
        (
            FinishReason::Length,
            Usage {
                prompt_tokens: 8,
                completion_tokens: 25,
                cached_prompt_tokens: 0,
                output_hash: ares_runtime::token_hash(&tokens),
                ..Usage::default()
            }
        )
    );
    assert_eq!(run.text.chars().count(), 25);

    // No limit but the context: prompt and completion together fill it.
    let run = alone(rt, "fill the context", None);
    assert_eq!(run.finish, FinishReason::Length);
    assert_eq!(run.usage.prompt_tokens + run.usage.completion_tokens, context_length);

    // A stop string: pick one that appears in the output, and run again.
    let full = alone(rt, "stop here", Some(60));
    let chars: Vec<char> = full.text.chars().collect();
    let stop: String = chars[20..23].iter().collect();
    let at = full.text.find(&stop).expect("it is in the text");
    let mut stopped = request("stop here", Some(60));
    stopped.stop = vec![stop.clone()];
    let run = run_one(rt, stopped);
    assert_eq!(run.finish, FinishReason::StopString(0), "the first of the request's stop strings");
    assert_eq!(run.text, full.text[..at], "the text ends before the stop string");
    let before = full.text[..at].chars().count() as u32;
    assert_eq!(
        run.usage.completion_tokens,
        before + stop.chars().count() as u32,
        "the tokens that made up the stop string were generated and are counted"
    );
}

/// The counters of two snapshots, as a difference.
pub fn since(now: &Snapshot, before: &Snapshot) -> HashMap<&'static str, u64> {
    let (a, b) = (&now.counters, &before.counters);
    HashMap::from([
        ("steps", a.steps - b.steps),
        ("rows", a.rows - b.rows),
        ("tokens", a.tokens - b.tokens),
        ("draft_rows", a.draft_rows - b.draft_rows),
        ("accepted", a.drafts_accepted - b.drafts_accepted),
        ("rejected", a.drafts_rejected - b.drafts_rejected),
        ("mixed", a.mixed_steps - b.mixed_steps),
    ])
}

/// The end token ends a request at its first occurrence, is counted and not
/// printed, and does the same next to other requests and with speculation.
/// `make(None)` builds a runtime whose model has no end token; `make(Some(b))`
/// one whose end token is the byte `b`.
pub fn the_end_token_ends_a_request_where_it_first_comes_up(make: &dyn Fn(Option<u32>) -> Runtime) {
    let prompt = "where does it end";
    let full = alone(&make(None), prompt, Some(60));
    let chars: Vec<char> = full.text.chars().collect();
    // Make a byte the model says at position 17 the end token. It may have
    // come up before; the request must end at the first time.
    let end = chars[17];
    let first = chars.iter().position(|c| *c == end).unwrap();
    let rt = make(Some(u32::from(ares_runtime::unprintable(end).unwrap())));
    let expected = Run {
        text: chars[..first].iter().collect(),
        finish: FinishReason::Stop,
        usage: Usage {
            prompt_tokens: prompt.len() as u32,
            completion_tokens: first as u32 + 1,
            cached_prompt_tokens: 0,
            // The end token is not part of the reply.
            output_hash: ares_runtime::token_hash(
                &chars[..first]
                    .iter()
                    .map(|c| u32::from(ares_runtime::unprintable(*c).expect("a token")))
                    .collect::<Vec<_>>(),
            ),
            ..Usage::default()
        },
        probs: Vec::new(),
    };
    assert_eq!(alone(&rt, prompt, Some(60)), expected);
    let handles: Vec<Handle> =
        (0..6).map(|_| start(&rt, request(prompt, Some(60))).expect("accepted")).collect();
    for handle in handles {
        assert_eq!(handle.finish(), expected);
    }
}
