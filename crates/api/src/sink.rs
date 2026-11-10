//! The channel from the engine's threads to the front end.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{DecisionOutcome, Event};

/// Where the engine delivers the [`Event`]s of one request.
///
/// # Contract
///
/// The sink is called from the engine's step loop, between GPU steps, on a
/// plain OS thread with no async runtime. Therefore:
///
/// * [`send`](EventSink::send) **never blocks**. The callback behind it must
///   not wait for the receiver, for a lock the receiver holds, or for room in
///   a queue. If it cannot take the event now it says so by returning `false`.
///   A step loop that waited on one slow client would stall every other
///   request on the GPU.
/// * `send` returns `false` when the event was not delivered because nobody
///   wants the rest: the client went away, or it reads so slowly that its
///   queue overflowed. The engine must then **cancel the request** and stop
///   sending. The sink enforces the stop: once it has returned `false`, every
///   later `send` returns `false` without calling the callback.
/// * Delivering a terminal event ([`Event::Done`] or [`Event::Error`]) closes
///   the sink in the same way, so a request cannot produce events after its
///   end. The terminal `send` itself returns `true` when it was delivered.
/// * The sink is not tied to one thread: it is `Send`, and the engine may move
///   it between threads. Calls must not overlap in time, which is natural for
///   a request that has one owner.
///
/// Dropping the sink without a terminal event tells the receiver nothing
/// except that no more events will come; an engine that does this is
/// misbehaving, and the front end reports it as an internal error.
pub struct EventSink {
    deliver: Box<dyn Fn(Event) -> bool + Send + Sync>,
    closed: AtomicBool,
}

impl EventSink {
    /// A sink over `deliver`, which must follow the contract above: return
    /// `true` when it took the event, `false` when it did not and the engine
    /// should stop, and never block.
    pub fn new(deliver: impl Fn(Event) -> bool + Send + Sync + 'static) -> Self {
        Self { deliver: Box::new(deliver), closed: AtomicBool::new(false) }
    }

    /// Delivers `event`. Returns `false` when it was not delivered and the
    /// request must be cancelled (see the [type documentation](EventSink)).
    pub fn send(&self, event: Event) -> bool {
        if self.closed.load(Ordering::Acquire) {
            return false;
        }
        let terminal = event.is_terminal();
        let delivered = (self.deliver)(event);
        if terminal || !delivered {
            self.closed.store(true, Ordering::Release);
        }
        delivered
    }

    /// Whether the sink stopped accepting events, for a step loop that wants
    /// to skip detokenizing for a request that nobody reads any more.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

impl fmt::Debug for EventSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventSink").field("closed", &self.is_closed()).finish_non_exhaustive()
    }
}

/// Where the engine delivers the one [`DecisionOutcome`] of a decision.
///
/// A decision is not a stream. It has no text to push while it runs, and it
/// ends exactly once, with its answers or without them. Putting that on
/// [`EventSink`] would need an `Event` that no generation ever produces and
/// that every consumer of events has to know to ignore, and would leave "one
/// terminal event" a promise the engine can break. Here [`send`](Self::send)
/// takes the sink by value, so a second outcome does not compile.
///
/// # Contract
///
/// * `send` **never blocks**. It is called from the engine's step loop, on a
///   plain OS thread with no async runtime, and the callback behind it must
///   not wait for the receiver. A single value cannot overflow anything, so
///   unlike [`EventSink`] there is no refusal to report: if the receiver is
///   gone the outcome is dropped, and the engine learns that the client left
///   from [`Engine::cancel`](crate::Engine::cancel), as for any request.
/// * The sink is `Send`; the engine may move it between threads.
///
/// Dropping the sink without sending tells the receiver only that no outcome
/// will come; an engine that does this is misbehaving, and the front end
/// reports it as an internal error.
pub struct DecisionSink {
    deliver: Box<dyn FnOnce(DecisionOutcome) + Send>,
}

impl DecisionSink {
    /// A sink over `deliver`, which must not block.
    pub fn new(deliver: impl FnOnce(DecisionOutcome) + Send + 'static) -> Self {
        Self { deliver: Box::new(deliver) }
    }

    /// Delivers the outcome and ends the request.
    pub fn send(self, outcome: DecisionOutcome) {
        (self.deliver)(outcome);
    }
}

impl fmt::Debug for DecisionSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecisionSink").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, mpsc};
    use std::thread;

    use super::*;
    use crate::{EngineError, FinishReason, Usage};

    fn done() -> Event {
        Event::Done { finish: FinishReason::Stop, usage: Usage::default() }
    }

    #[test]
    fn works_from_a_plain_thread_without_an_async_runtime() {
        let (tx, rx) = mpsc::channel();
        let sink = EventSink::new(move |event| tx.send(event).is_ok());
        thread::spawn(move || {
            assert!(sink.send(Event::Text("a".into())));
            assert!(sink.send(done()));
        })
        .join()
        .unwrap();
        assert_eq!(rx.iter().collect::<Vec<_>>(), [Event::Text("a".into()), done()]);
    }

    #[test]
    fn after_the_receiver_refuses_once_the_callback_is_never_called_again() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let sink = EventSink::new(move |_| counter.fetch_add(1, Ordering::SeqCst) < 2);
        assert!(sink.send(Event::Text("1".into())));
        assert!(sink.send(Event::Text("2".into())));
        assert!(!sink.is_closed());
        assert!(!sink.send(Event::Text("3".into())), "the callback said no");
        assert!(sink.is_closed());
        assert!(!sink.send(Event::Text("4".into())));
        assert!(!sink.send(done()));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "calls after the refusal must not reach the callback"
        );
    }

    #[test]
    fn nothing_follows_a_terminal_event() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let sink = EventSink::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            true
        });
        assert!(
            sink.send(Event::Error(EngineError::Overloaded)),
            "the terminal event itself is delivered"
        );
        assert!(!sink.send(Event::Text("late".into())));
        assert!(!sink.send(done()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn can_be_moved_to_another_thread_after_creation() {
        fn assert_send<T: Send>() {}
        assert_send::<EventSink>();
        assert_send::<DecisionSink>();
    }

    #[test]
    fn a_decision_outcome_is_delivered_once_from_a_plain_thread() {
        let (tx, rx) = mpsc::channel();
        let sink = DecisionSink::new(move |outcome| tx.send(outcome).unwrap());
        thread::spawn(move || sink.send(DecisionOutcome::Cancelled)).join().unwrap();
        assert_eq!(rx.iter().collect::<Vec<_>>(), [DecisionOutcome::Cancelled]);
    }

    #[test]
    fn a_decision_sink_dropped_unused_never_calls_the_callback() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let (tx, rx) = mpsc::channel::<()>();
        let sink = DecisionSink::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            drop(tx);
        });
        drop(sink);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(rx.recv().is_err(), "the receiver sees the sender go without an outcome");
    }
}
