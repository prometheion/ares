//! The hand-over between the engine's threads and an async request handler.
//!
//! One [`Run`] per request. The engine pushes events from its step loop into
//! a bounded channel through an [`EventSink`]; the handler pulls them with
//! [`Run::next`].
//!
//! # A full channel
//!
//! The engine must never wait for a client, and text must never be dropped
//! silently. Both hold if the channel is bounded and a full channel is treated
//! as a client that cannot keep up:
//!
//! * [`EventSink`] calls `try_send`. When the channel is full it refuses the
//!   event, which tells the engine (by the sink contract) to cancel the
//!   request, and the sink stays closed.
//! * The handler learns about it from a [`Notify`] and a flag, after it has
//!   delivered every event that was still queued. It then cancels the request
//!   itself as well (cancel is idempotent) and ends the response with a
//!   `slow_client` error. The client sees all text up to the overflow, then an
//!   explicit error. Nothing disappears without a trace.
//!
//! The alternative, an unbounded channel, would keep memory bounded only by
//! `max_tokens` (or by the context length when the client sets none) and, worse,
//! would let the engine spend GPU time on a client that stopped reading. A
//! client that has stopped reading is exactly the case where the request
//! should end early.
//!
//! The buffer is [`Config::event_buffer`](crate::Config::event_buffer) events,
//! which is generous on purpose: before the handler stalls, the kernel's socket
//! buffers on both ends have to fill, which takes thousands of tokens' worth
//! of SSE chunks. The channel therefore overflows only for a client that has
//! really stopped reading, or one so slow that it could never have caught up.
//!
//! One slot is reserved for the terminal event: text may use every slot but the
//! last. A generation that has finished must not be reported as a slow client
//! just because its `Done` found the queue full.
//!
//! # Timings
//!
//! The times of a request are taken where its events arrive, on the engine's
//! thread, so that neither the channel nor a slow client is in them: when the
//! request was handed to the engine, when its first output came (the model's
//! first token ends the prompt), and when it ended. The time of the prompt is
//! the engine's own when it says it ([`Usage::prompt_time`]: from the
//! request's start, after its wait in line, to its first token), and the drafts
//! are the request's own when the engine counts them ([`Usage::drafts`]);
//! from an engine that does not, they are the engine's totals before and after
//! ([`Flight`]), which are this request's own only when no other request ran in
//! between.
//!
//! # The log
//!
//! A request's lines are for a person, laid out as llama.cpp's server lays out
//! its timings: when it starts in the engine (its prompt, what the cache gave),
//! how far a long prompt has got (from 3 s into it, at most once in 3 s, when
//! another tenth is done or 10 s have passed), and at its end the prompt's
//! rate over the rows the engine ran for it, the rate of generation, the total
//! and the drafts. The same in fields, for a program, is the debug line
//! `request finished`.
//!
//! # Decisions
//!
//! A decision has no stream, so [`DecisionRun`] needs none of this: the
//! engine delivers one outcome through a [`DecisionSink`] into a one-shot
//! channel, which cannot overflow and never makes the engine wait. A client
//! that goes away cancels it the same way, by dropping the handler's future,
//! which drops the [`CancelGuard`].

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ares_api::{
    DecisionOutcome, DecisionRequest, DecisionResult, DecisionSink, Engine, EngineStats, Event,
    EventSink, FinishReason, GenerateRequest, Progress, RequestId, TokenProbs, Usage,
};
use tokio::sync::{Notify, mpsc, oneshot, watch};

use crate::error::ApiError;
use crate::metrics::latency::{Latency, Took};

/// What a request produces, minus the failures.
#[derive(Debug)]
pub(crate) enum Item {
    /// How far the prompt has got.
    Progress(Progress),
    Text(String),
    /// The probabilities of the tokens whose text went out.
    Probs(Vec<TokenProbs>),
    Done {
        finish: &'static str,
        /// The place in the request's list of the stop string that ended
        /// the reply.
        stopped_by: Option<usize>,
        usage: Usage,
        timings: Timings,
    },
}

/// Where the time of a finished request went, in a form that can be set next
/// to other engines' numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Timings {
    /// Tokens of the prompt that were computed.
    pub prompt_tokens: u32,
    /// The prompt: from the request's start in the engine to its first token,
    /// as the engine says it ([`Usage::prompt_time`]); from an engine that does
    /// not, `first_ms`.
    pub prompt_ms: f64,
    /// From the hand-over to the engine to the first output: tokenizing, any
    /// wait for room, and the prompt. The first token comes out of the step
    /// that ends the prompt.
    pub first_ms: f64,
    /// All generated tokens, the first one included.
    pub generated_tokens: u32,
    /// From the first output to the end: the time of every generated token
    /// but the first.
    pub generated_ms: f64,
    /// Draft tokens put to the test and accepted for this request, when the
    /// engine's totals can be read as this request's: it ran alone.
    pub drafts: Option<(u64, u64)>,
    /// A hash of the reply's token ids ([`Usage::output_hash`]).
    pub output_hash: u64,
}

impl Timings {
    /// Prompt tokens per second.
    pub fn prompt_per_second(&self) -> Option<f64> {
        (self.prompt_ms > 0.0).then(|| f64::from(self.prompt_tokens) * 1e3 / self.prompt_ms)
    }

    /// Generated tokens per second: the tokens after the first over the time
    /// after the first, because the first is paid for by the prompt.
    pub fn generated_per_second(&self) -> Option<f64> {
        (self.generated_tokens > 1 && self.generated_ms > 0.0)
            .then(|| f64::from(self.generated_tokens - 1) * 1e3 / self.generated_ms)
    }
}

/// When the events of a request arrived, in nanoseconds after its start.
/// Written on the engine's thread, read by the handler after the end.
#[derive(Debug)]
struct Marks {
    started: Instant,
    /// The first output; 0 while there was none.
    first: AtomicU64,
    /// The terminal event.
    end: AtomicU64,
}

impl Marks {
    fn note(&self, event: &Event) {
        let now = (self.started.elapsed().as_nanos() as u64).max(1);
        if !matches!(event, Event::Error(_) | Event::Progress(_)) {
            let _ = self.first.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire);
        }
        if event.is_terminal() {
            self.end.store(now, Ordering::Release);
        }
    }
}

/// The requests in flight, to know which of them ran alone.
///
/// The engine counts drafts for all requests together. A request that ran
/// from start to end with no other one next to it can take the difference of
/// the totals as its own; any other gets no draft counts. The totals after a
/// request that ran alone serve as the totals before the next one, so the
/// engine is asked once per request, after its text has gone out.
#[derive(Debug)]
pub(crate) struct Flight {
    running: AtomicUsize,
    /// How many requests have started so far.
    started: AtomicU64,
    /// The engine's totals after request number `.0` ended, having run alone.
    last: Mutex<Option<(u64, EngineStats)>>,
    /// How long the requests of the engine took.
    pub latency: Latency,
}

impl Flight {
    /// Reads the engine's totals once, as the start of the first request.
    pub fn new(engine: &dyn Engine) -> Self {
        Self {
            running: AtomicUsize::new(0),
            started: AtomicU64::new(0),
            last: Mutex::new(engine.stats().map(|stats| (0, stats))),
            latency: Latency::default(),
        }
    }

    fn last(&self) -> std::sync::MutexGuard<'_, Option<(u64, EngineStats)>> {
        self.last.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// One request's place in a [`Flight`]; gives it up when dropped.
#[derive(Debug)]
struct Seat {
    flight: Arc<Flight>,
    number: u64,
    /// The engine's totals when the request started, if nothing else ran then
    /// and they are known.
    before: Option<EngineStats>,
    released: bool,
}

impl Seat {
    fn take(flight: &Arc<Flight>) -> Self {
        let number = flight.started.fetch_add(1, Ordering::AcqRel) + 1;
        let alone = flight.running.fetch_add(1, Ordering::AcqRel) == 0;
        let before = match *flight.last() {
            Some((previous, stats)) if alone && previous + 1 == number => Some(stats),
            _ => None,
        };
        Self { flight: flight.clone(), number, before, released: false }
    }

    /// Whether no request has started since this one did.
    fn latest(&self) -> bool {
        self.flight.started.load(Ordering::Acquire) == self.number
    }

    /// The drafts of the request that just ended in `engine`, if it ran alone.
    async fn drafts(&self, engine: &Arc<dyn Engine>) -> Option<(u64, u64)> {
        // Others ran next to it when it started, or have started since.
        if self.flight.running.load(Ordering::Acquire) != 1 || !self.latest() {
            return None;
        }
        let engine = engine.clone();
        let after = tokio::task::spawn_blocking(move || engine.stats()).await.ok().flatten()?;
        if !self.latest() {
            return None;
        }
        *self.flight.last() = Some((self.number, after));
        self.before.map(|before| after.drafts_since(&before))
    }

    /// Gives the place up: the request has left the engine. The next request
    /// may start before this one's response is written out, and it then runs
    /// alone all the same.
    fn release(&mut self) {
        if !std::mem::replace(&mut self.released, true) {
            self.flight.running.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        self.release();
    }
}

/// Tells running requests that the server is stopping and will not wait for
/// them any longer. The handlers then cancel them in the engine and end
/// their responses with an error.
#[derive(Debug, Clone)]
pub(crate) struct Stop(watch::Receiver<bool>);

impl Stop {
    /// A signal and the sender that raises it by sending `true`. While the
    /// sender lives and has not sent, and after it is dropped without having
    /// sent, the signal stays down.
    pub fn new() -> (watch::Sender<bool>, Self) {
        let (tx, rx) = watch::channel(false);
        (tx, Self(rx))
    }

    /// Completes when the signal is raised.
    pub async fn raised(&mut self) {
        if self.0.wait_for(|stop| *stop).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

/// Set by the sink (on an engine thread), read by the handler.
#[derive(Debug, Default)]
struct Overflow {
    flagged: AtomicBool,
    wake: Notify,
}

/// Cancels the request in the engine when dropped before the request ended.
///
/// This is how a client disconnect reaches the engine: when the connection
/// closes, hyper drops the handler's future (non-streaming) or the response
/// body (streaming), which drops the [`Run`] and with it this guard. No
/// disconnect detection code is needed, and no path (an early return, an
/// error, a panic that unwinds) can forget to cancel.
struct CancelGuard {
    engine: Arc<dyn Engine>,
    id: RequestId,
    armed: bool,
}

impl CancelGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }

    fn cancel_now(&mut self) {
        if std::mem::take(&mut self.armed) {
            self.engine.cancel(self.id);
        }
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.armed {
            tracing::info!(id = %self.id, "client went away, cancelling the request");
            self.engine.cancel(self.id);
        }
    }
}

/// A request that the engine has accepted.
/// How often a request that waits for its engine looks whether the engine
/// has failed without telling it (a step its watchdog found stuck).
const FAILURE_LOOK: std::time::Duration = std::time::Duration::from_secs(1);

pub(crate) struct Run {
    rx: mpsc::Receiver<Event>,
    overflow: Arc<Overflow>,
    guard: CancelGuard,
    engine: Arc<dyn Engine>,
    marks: Arc<Marks>,
    seat: Seat,
    stop: Stop,
    model: String,
    /// The seed the request asked for, for the log.
    seed: Option<u64>,
    /// The reply's text, kept for the log when it is to have it.
    text: Option<String>,
    /// What the log said of the prompt's progress.
    told: Told,
}

/// What the log said of a prompt's progress: its last line, when and how far.
#[derive(Debug, Default)]
struct Told {
    started: bool,
    last: Option<(Duration, f64)>,
}

impl Told {
    /// Whether a line is due for a prompt that has run `t` and has `share` of
    /// it done (the module documentation says when), and notes it if it is.
    fn due(&mut self, t: Duration, share: f64) -> bool {
        if t < PROGRESS_LINES || share >= 1.0 {
            return false;
        }
        if let Some((at, said)) = self.last
            && (t < at + PROGRESS_LINES || (share < said + 0.1 && t < at + PROGRESS_SILENCE))
        {
            return false;
        }
        self.last = Some((t, share));
        true
    }
}

/// A prompt that is read longer than this has lines of its progress in the log,
/// at most one in this time.
const PROGRESS_LINES: Duration = Duration::from_secs(3);
/// A line of the progress is due when another tenth of the prompt is done or
/// this time has passed since the last.
const PROGRESS_SILENCE: Duration = Duration::from_secs(10);

impl std::fmt::Debug for Run {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Run").field("id", &self.guard.id).finish_non_exhaustive()
    }
}

impl Run {
    /// Hands `request` to the engine. `buffer` is the number of events that
    /// may wait for the handler (at least 1); with `log_text` the reply's text
    /// goes into the log at its end.
    pub fn start(
        engine: &Arc<dyn Engine>,
        request: GenerateRequest,
        buffer: usize,
        flight: &Arc<Flight>,
        stop: Stop,
        log_text: bool,
    ) -> Result<Self, ApiError> {
        let id = request.id;
        let model = request.model.clone();
        let seed = request.sampling.seed;
        // One extra slot for the terminal event, see the module documentation.
        let (tx, rx) = mpsc::channel(buffer.max(1) + 1);
        let overflow = Arc::new(Overflow::default());
        let seat = Seat::take(flight);
        let marks = Arc::new(Marks {
            started: Instant::now(),
            first: AtomicU64::new(0),
            end: AtomicU64::new(0),
        });

        let flag = overflow.clone();
        let clock = marks.clone();
        let room = buffer.max(1) + 1;
        let sink = EventSink::new(move |event| {
            clock.note(&event);
            // A report of progress that finds the queue half full is let go:
            // the next one says more, and it must not end the request.
            if matches!(event, Event::Progress(_)) && tx.capacity() * 2 <= room {
                return true;
            }
            if !event.is_terminal() && tx.capacity() <= 1 {
                flag.flagged.store(true, Ordering::Release);
                flag.wake.notify_one();
                return false;
            }
            // Only this callback fills the channel, and calls do not overlap,
            // so the capacity checked above can only have grown since.
            // `try_send` therefore fails only when the receiver is gone.
            tx.try_send(event).is_ok()
        });

        engine.submit(request, sink).map_err(|e| ApiError::from_engine(&**engine, e))?;
        Ok(Self {
            rx,
            overflow,
            guard: CancelGuard { engine: engine.clone(), id, armed: true },
            engine: engine.clone(),
            marks,
            seat,
            stop,
            model,
            seed,
            text: log_text.then(String::new),
            told: Told::default(),
        })
    }

    /// The next piece of the response, or the reason there will be none.
    ///
    /// After an `Err` (or after `Item::Done`) the request is over and
    /// `next` must not be called again.
    pub async fn next(&mut self) -> Result<Item, ApiError> {
        let id = self.guard.id;
        self.pull().await.map_err(|error| error.of_request(id))
    }

    async fn pull(&mut self) -> Result<Item, ApiError> {
        // `biased`: events that are already queued come before the overflow
        // notice, so the client receives every piece of text the engine
        // managed to deliver.
        let event = loop {
            tokio::select! {
                biased;
                event = self.rx.recv() => break event,
                () = self.overflow.wake.notified() => return Err(self.slow()),
                () = self.stop.raised() => {
                    tracing::info!(id = %self.guard.id, "the server is stopping: ending the request");
                    self.guard.cancel_now();
                    return Err(ApiError::cancelled());
                }
                // An engine whose watchdog found it stuck sends nothing more:
                // the request ends with its reason.
                () = tokio::time::sleep(FAILURE_LOOK) => {
                    if let Some(why) = self.engine.failure() {
                        tracing::error!(id = %self.guard.id, %why, "the engine failed: ending the request");
                        self.guard.cancel_now();
                        return Err(ApiError::internal(why));
                    }
                }
            }
        };
        let Some(event) = event else {
            // The sender is gone and nothing terminal came: either the sink
            // was dropped after an overflow, or the engine lost the request.
            return Err(if self.overflow.flagged.load(Ordering::Acquire) {
                self.slow()
            } else {
                tracing::error!(id = %self.guard.id, "the engine dropped the request without a terminal event");
                self.guard.cancel_now();
                ApiError::engine_vanished()
            });
        };
        match event {
            Event::Progress(progress) => {
                self.tell_progress(&progress);
                Ok(Item::Progress(progress))
            }
            Event::Text(text) => {
                if let Some(kept) = &mut self.text {
                    kept.push_str(&text);
                }
                Ok(Item::Text(text))
            }
            Event::Logprobs(probs) => Ok(Item::Probs(probs)),
            Event::Done { finish, usage } => {
                self.guard.disarm();
                let timings = self.timings(usage).await;
                let rate = |rate: Option<f64>| rate.map_or("-".to_owned(), |r| format!("{r:.1}"));
                let count = |n: Option<u64>| n.map_or("-".to_owned(), |n| n.to_string());
                let queued = usage.queued.map(|queued| queued.as_secs_f64() * 1e3);
                if !matches!(finish, FinishReason::Cancelled) {
                    let ms = |ms: f64| std::time::Duration::from_secs_f64(ms / 1e3);
                    self.seat.flight.latency.observe(Took {
                        queued: usage.queued,
                        first: ms(timings.first_ms),
                        whole: ms(timings.first_ms + timings.generated_ms),
                        tokens: usage.completion_tokens,
                    });
                }
                tell_end(self.guard.id, finish, &usage, &timings);
                tracing::debug!(
                    id = %self.guard.id,
                    model = %self.model,
                    ?finish,
                    prompt_tokens = usage.prompt_tokens,
                    cached_prompt_tokens = usage.cached_prompt_tokens,
                    queued_ms = %queued.map_or("-".to_owned(), |ms| format!("{ms:.1}")),
                    prompt_ms = format_args!("{:.1}", timings.prompt_ms),
                    prompt_tokens_per_s = %rate(timings.prompt_per_second()),
                    completion_tokens = usage.completion_tokens,
                    reasoning_tokens = %count(usage.reasoning_tokens.map(u64::from)),
                    generation_ms = format_args!("{:.1}", timings.generated_ms),
                    tokens_per_s = %rate(timings.generated_per_second()),
                    drafts = %count(timings.drafts.map(|d| d.0)),
                    drafts_accepted = %count(timings.drafts.map(|d| d.1)),
                    seed = %count(self.seed),
                    "request finished"
                );
                if let Some(text) = self.text.take() {
                    tracing::info!(id = %self.guard.id, reply = %text, "the reply");
                }
                let stopped_by = match finish {
                    FinishReason::StopString(index) => Some(index as usize),
                    _ => None,
                };
                match finish {
                    FinishReason::Stop | FinishReason::StopString(_) => {
                        Ok(Item::Done { finish: "stop", stopped_by, usage, timings })
                    }
                    FinishReason::Length => {
                        Ok(Item::Done { finish: "length", stopped_by, usage, timings })
                    }
                    FinishReason::Cancelled => Err(ApiError::cancelled()),
                }
            }
            Event::Error(error) => {
                self.guard.disarm();
                Err(ApiError::from_engine(&*self.engine, error))
            }
        }
    }

    /// The times of the request that just ended with `usage`.
    async fn timings(&mut self, usage: Usage) -> Timings {
        let end = self.marks.end.load(Ordering::Acquire);
        let first = match self.marks.first.load(Ordering::Acquire) {
            0 => end,
            first => first,
        };
        // The engine's own counts of the request's drafts, or the difference of
        // its totals for a request that ran alone.
        let drafts = match usage.drafts {
            Some(drafts) => Some((u64::from(drafts.proposed), u64::from(drafts.accepted))),
            None => self.seat.drafts(&self.engine).await,
        };
        self.seat.release();
        let first_ms = first as f64 / 1e6;
        Timings {
            prompt_tokens: usage.prompt_tokens.saturating_sub(usage.cached_prompt_tokens),
            prompt_ms: usage.prompt_time.map_or(first_ms, |time| time.as_secs_f64() * 1e3),
            first_ms,
            generated_tokens: usage.completion_tokens,
            generated_ms: end.saturating_sub(first) as f64 / 1e6,
            drafts,
            output_hash: usage.output_hash,
        }
    }

    /// The lines of the log of a prompt's progress: where it starts, and how far
    /// it has got while it is read long (see the module documentation).
    fn tell_progress(&mut self, progress: &Progress) {
        let id = self.guard.id;
        if !std::mem::replace(&mut self.told.started, true) {
            let waited = self.marks.started.elapsed().saturating_sub(progress.elapsed);
            tracing::info!(
                "request {id} started after {}: prompt of {} tokens, {} from the cache, {} to compute",
                duration(waited),
                progress.total,
                progress.cached,
                progress.total.saturating_sub(progress.cached),
            );
            return;
        }
        let t = progress.elapsed;
        let share = f64::from(progress.processed) / f64::from(progress.total.max(1));
        if !self.told.due(t, share) {
            return;
        }
        let done = progress.processed.saturating_sub(progress.cached);
        tracing::info!(
            "request {id} prompt processing, n_tokens = {done:6}, progress = {share:.2}, t = {:6.2} s / {:.2} tokens per second",
            t.as_secs_f64(),
            f64::from(done) / t.as_secs_f64(),
        );
    }

    fn slow(&mut self) -> ApiError {
        tracing::warn!(id = %self.guard.id, "event queue overflowed: the client is too slow, cancelling");
        self.guard.cancel_now();
        ApiError::slow_client()
    }
}

/// A short duration for a person: microseconds, milliseconds or seconds.
fn duration(time: Duration) -> String {
    let ms = time.as_secs_f64() * 1e3;
    if ms < 1.0 {
        format!("{:.0} us", ms * 1e3)
    } else if ms < 10_000.0 {
        format!("{ms:.1} ms")
    } else {
        format!("{:.1} s", ms / 1e3)
    }
}

/// The lines of the log of a request that ended: what it was, then its timings
/// as llama.cpp's server prints them (`print_timings`), with the prompt's rate
/// over the rows the engine ran for it (the computed ones and the replayed).
fn tell_end(id: RequestId, finish: FinishReason, usage: &Usage, timings: &Timings) {
    let reason = match finish {
        FinishReason::Stop => "stop".to_owned(),
        FinishReason::StopString(index) => format!("stop string {index}"),
        FinishReason::Length => "length".to_owned(),
        FinishReason::Cancelled => "cancelled".to_owned(),
    };
    let mut cache = format!("{} from the cache", usage.cached_prompt_tokens);
    if usage.replayed_prompt_tokens > 0 {
        cache.push_str(&format!(", {} of them replayed", usage.replayed_prompt_tokens));
    }
    let mut made = format!("{} generated", usage.completion_tokens);
    if let Some(thinking) = usage.reasoning_tokens.filter(|n| *n > 0) {
        made.push_str(&format!(" ({thinking} of them thinking)"));
    }
    if let Some(queued) = usage.queued.filter(|q| *q >= Duration::from_millis(1)) {
        made.push_str(&format!(", {} in line", duration(queued)));
    }
    tracing::info!(
        "request {id} finished ({reason}): prompt of {} tokens ({cache}), {made}",
        usage.prompt_tokens
    );
    let rows = timings.prompt_tokens + usage.replayed_prompt_tokens;
    let per = |ms: f64, n: u32| if n > 0 { ms / f64::from(n) } else { 0.0 };
    let rate = |ms: f64, n: u32| if ms > 0.0 { f64::from(n) * 1e3 / ms } else { 0.0 };
    let (prompt_ms, generated_ms) = (timings.prompt_ms, timings.generated_ms);
    // The first token comes with the prompt: the rate of generation is over
    // the others.
    let after_first = timings.generated_tokens.saturating_sub(1);
    tracing::info!(
        "request {id} prompt eval time = {prompt_ms:10.2} ms / {rows:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
        per(prompt_ms, rows),
        rate(prompt_ms, rows),
    );
    tracing::info!(
        "request {id}        eval time = {generated_ms:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
        timings.generated_tokens,
        per(generated_ms, after_first),
        rate(generated_ms, after_first),
    );
    tracing::info!(
        "request {id}       total time = {:10.2} ms / {:5} tokens",
        prompt_ms + generated_ms,
        rows + timings.generated_tokens,
    );
    if let Some(drafts) = usage.drafts.filter(|d| d.proposed > 0) {
        let windows = f64::from(drafts.windows.max(1));
        tracing::info!(
            "request {id} draft acceptance = {:.5} ({:5} accepted / {:5} proposed), mean len = {:5.2}, mean chain = {:5.2}",
            f64::from(drafts.accepted) / f64::from(drafts.proposed),
            drafts.accepted,
            drafts.proposed,
            1.0 + f64::from(drafts.accepted) / windows,
            f64::from(drafts.proposed) / windows,
        );
    } else if usage.drafts.is_none()
        && let Some((proposed, accepted)) = timings.drafts.filter(|d| d.0 > 0)
    {
        tracing::info!(
            "request {id} draft acceptance = {:.5} ({accepted:5} accepted / {proposed:5} proposed)",
            accepted as f64 / proposed as f64,
        );
    }
}

/// A decision that the engine has accepted.
pub(crate) struct DecisionRun {
    rx: oneshot::Receiver<DecisionOutcome>,
    guard: CancelGuard,
    engine: Arc<dyn Engine>,
    started: Instant,
    stop: Stop,
    model: String,
}

impl std::fmt::Debug for DecisionRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionRun").field("id", &self.guard.id).finish_non_exhaustive()
    }
}

impl DecisionRun {
    /// Hands `request` to the engine.
    pub fn start(
        engine: &Arc<dyn Engine>,
        request: DecisionRequest,
        stop: Stop,
    ) -> Result<Self, ApiError> {
        let id = request.id;
        let model = request.model.clone();
        let (tx, rx) = oneshot::channel();
        // A send fails only when the handler is gone, and then there is
        // nobody to tell.
        let sink = DecisionSink::new(move |outcome| {
            let _ = tx.send(outcome);
        });

        engine.decide(request, sink).map_err(|e| ApiError::from_engine(&**engine, e))?;
        Ok(Self {
            rx,
            guard: CancelGuard { engine: engine.clone(), id, armed: true },
            engine: engine.clone(),
            started: Instant::now(),
            stop,
            model,
        })
    }

    /// Waits for the answers, or the reason there will be none.
    ///
    /// If the future is dropped while waiting (the client went away) the
    /// request is cancelled in the engine.
    pub async fn finish(mut self) -> Result<DecisionResult, ApiError> {
        let outcome = tokio::select! {
            biased;
            outcome = &mut self.rx => outcome,
            () = self.stop.raised() => {
                tracing::info!(id = %self.guard.id, "the server is stopping: ending the decision");
                self.guard.cancel_now();
                return Err(ApiError::cancelled());
            }
        };
        let Ok(outcome) = outcome else {
            // The sink was dropped and nothing was sent: the engine lost the
            // request.
            tracing::error!(id = %self.guard.id, "the engine dropped the decision without an outcome");
            self.guard.cancel_now();
            return Err(ApiError::engine_vanished());
        };
        self.guard.disarm();
        match outcome {
            DecisionOutcome::Decided(result) => {
                tracing::info!(
                    id = %self.guard.id,
                    model = %self.model,
                    questions = result.answers.len(),
                    prompt_tokens = result.usage.prompt_tokens,
                    cached_prompt_tokens = result.usage.cached_prompt_tokens,
                    elapsed_ms = self.started.elapsed().as_millis() as u64,
                    "decision finished"
                );
                Ok(result)
            }
            DecisionOutcome::Cancelled => Err(ApiError::cancelled()),
            DecisionOutcome::Failed(error) => {
                tracing::info!(id = %self.guard.id, %error, "decision failed");
                Err(ApiError::from_engine(&*self.engine, error))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long prompt's progress is in the log from 3 s on, at most once in 3 s,
    /// when another tenth is done or 10 s have passed; never for its end.
    #[test]
    fn the_lines_of_a_prompts_progress_come_every_tenth_or_ten_seconds() {
        let s = Duration::from_secs_f64;
        let mut told = Told::default();
        assert!(!told.due(s(2.9), 0.5), "a prompt that ends within 3 s has none");
        assert!(told.due(s(3.0), 0.5));
        assert!(!told.due(s(4.0), 0.9), "not within 3 s of the last");
        assert!(!told.due(s(6.5), 0.55), "not before another tenth");
        assert!(told.due(s(6.5), 0.6));
        assert!(told.due(s(16.5), 0.61), "after 10 s whatever it did");
        assert!(!told.due(s(30.0), 1.0), "the end is the request's own line");
    }
}
