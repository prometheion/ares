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
//! first token ends the prompt), and when it ended. The counts of draft
//! tokens are the engine's totals before and after ([`Flight`]), which are
//! this request's own only when no other request ran in between; the engine
//! does not report them per request yet.
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
use std::time::Instant;

use ares_api::{
    DecisionOutcome, DecisionRequest, DecisionResult, DecisionSink, Engine, EngineStats, Event,
    EventSink, FinishReason, GenerateRequest, RequestId, TokenProbs, Usage,
};
use tokio::sync::{Notify, mpsc, oneshot, watch};

use crate::error::ApiError;
use crate::latency::{Latency, Took};

/// What a request produces, minus the failures.
#[derive(Debug)]
pub(crate) enum Item {
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
    /// From the hand-over to the engine to the first output: tokenizing, any
    /// wait for room, and the prompt. The first token comes out of the step
    /// that ends the prompt.
    pub prompt_ms: f64,
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
        if !matches!(event, Event::Error(_)) {
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
}

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
        let sink = EventSink::new(move |event| {
            clock.note(&event);
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
        })
    }

    /// The next piece of the response, or the reason there will be none.
    ///
    /// After an `Err` (or after `Item::Done`) the request is over and
    /// `next` must not be called again.
    pub async fn next(&mut self) -> Result<Item, ApiError> {
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
                        first: ms(timings.prompt_ms),
                        whole: ms(timings.prompt_ms + timings.generated_ms),
                        tokens: usage.completion_tokens,
                    });
                }
                tracing::info!(
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
                tracing::info!(id = %self.guard.id, %error, "request failed");
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
        let drafts = self.seat.drafts(&self.engine).await;
        self.seat.release();
        Timings {
            prompt_tokens: usage.prompt_tokens.saturating_sub(usage.cached_prompt_tokens),
            prompt_ms: first as f64 / 1e6,
            generated_tokens: usage.completion_tokens,
            generated_ms: end.saturating_sub(first) as f64 / 1e6,
            drafts,
            output_hash: usage.output_hash,
        }
    }

    fn slow(&mut self) -> ApiError {
        tracing::warn!(id = %self.guard.id, "event queue overflowed: the client is too slow, cancelling");
        self.guard.cancel_now();
        ApiError::slow_client()
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
