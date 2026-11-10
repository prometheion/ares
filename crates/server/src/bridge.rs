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
//! # Decisions
//!
//! A decision has no stream, so [`DecisionRun`] needs none of this: the
//! engine delivers one outcome through a [`DecisionSink`] into a one-shot
//! channel, which cannot overflow and never makes the engine wait. A client
//! that goes away cancels it the same way, by dropping the handler's future,
//! which drops the [`CancelGuard`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ares_api::{
    DecisionOutcome, DecisionRequest, DecisionResult, DecisionSink, Engine, Event, EventSink,
    FinishReason, GenerateRequest, RequestId, Usage,
};
use tokio::sync::{Notify, mpsc, oneshot};

use crate::error::ApiError;

/// What a request produces, minus the failures.
#[derive(Debug)]
pub(crate) enum Item {
    Text(String),
    Done { finish: &'static str, usage: Usage },
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
pub(crate) struct Run {
    rx: mpsc::Receiver<Event>,
    overflow: Arc<Overflow>,
    guard: CancelGuard,
    engine: Arc<dyn Engine>,
    started: Instant,
    model: String,
}

impl std::fmt::Debug for Run {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Run").field("id", &self.guard.id).finish_non_exhaustive()
    }
}

impl Run {
    /// Hands `request` to the engine. `buffer` is the number of events that
    /// may wait for the handler (at least 1).
    pub fn start(
        engine: &Arc<dyn Engine>,
        request: GenerateRequest,
        buffer: usize,
    ) -> Result<Self, ApiError> {
        let id = request.id;
        let model = request.model.clone();
        // One extra slot for the terminal event, see the module documentation.
        let (tx, rx) = mpsc::channel(buffer.max(1) + 1);
        let overflow = Arc::new(Overflow::default());

        let flag = overflow.clone();
        let sink = EventSink::new(move |event| {
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
            started: Instant::now(),
            model,
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
        let event = tokio::select! {
            biased;
            event = self.rx.recv() => event,
            () = self.overflow.wake.notified() => return Err(self.slow()),
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
            Event::Text(text) => Ok(Item::Text(text)),
            Event::Done { finish, usage } => {
                self.guard.disarm();
                tracing::info!(
                    id = %self.guard.id,
                    model = %self.model,
                    ?finish,
                    prompt_tokens = usage.prompt_tokens,
                    cached_prompt_tokens = usage.cached_prompt_tokens,
                    completion_tokens = usage.completion_tokens,
                    elapsed_ms = self.started.elapsed().as_millis() as u64,
                    "request finished"
                );
                match finish {
                    FinishReason::Stop => Ok(Item::Done { finish: "stop", usage }),
                    FinishReason::Length => Ok(Item::Done { finish: "length", usage }),
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
    model: String,
}

impl std::fmt::Debug for DecisionRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionRun").field("id", &self.guard.id).finish_non_exhaustive()
    }
}

impl DecisionRun {
    /// Hands `request` to the engine.
    pub fn start(engine: &Arc<dyn Engine>, request: DecisionRequest) -> Result<Self, ApiError> {
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
            model,
        })
    }

    /// Waits for the answers, or the reason there will be none.
    ///
    /// If the future is dropped while waiting (the client went away) the
    /// request is cancelled in the engine.
    pub async fn finish(mut self) -> Result<DecisionResult, ApiError> {
        let Ok(outcome) = (&mut self.rx).await else {
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
