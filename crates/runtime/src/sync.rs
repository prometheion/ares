//! Waking a thread that waits for work: spin for a bounded time, then park.
//!
//! The conductor and the device threads hand work to each other once or twice
//! per step, and a step is short enough that the cost of that hand-over is
//! part of what the GPUs wait for. A parked thread is woken through the
//! kernel (a futex), which takes tens of microseconds to get the thread back
//! on a core; a thread that spins on an atomic sees a change in well under a
//! microsecond but burns its core while it waits. [`Bell`] does both: it
//! spins for as long as the caller says is worth it and parks after that, so
//! a busy engine pays the spinning price only where it buys latency and an
//! idle engine uses no CPU.

use std::hint::spin_loop;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// A counter that one thread waits on and any thread can ring.
///
/// There is one waiter per bell, bound with [`Bell::bind`]. Ringing is
/// cheap when the waiter is spinning (one atomic add and one load) and costs
/// an `unpark` only when it is parked.
#[derive(Debug, Default)]
pub(crate) struct Bell {
    count: AtomicU64,
    parked: AtomicBool,
    waiter: OnceLock<Thread>,
}

/// How a wait ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Woke {
    /// The bell had rung already or rang while the waiter was spinning.
    Spinning,
    /// The waiter had gone to sleep and was woken by the kernel.
    Parked,
}

impl Bell {
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes the calling thread the one that waits on this bell. Must be
    /// called before the first wait, from the waiting thread.
    pub fn bind(&self) {
        let _ = self.waiter.set(thread::current());
    }

    /// Rings the bell: whatever the ringer wrote before is visible to the
    /// waiter when its wait returns.
    pub fn ring(&self) {
        // SeqCst on both sides pairs this with the waiter's `parked` store:
        // either the waiter sees the new count before it sleeps, or the
        // ringer sees `parked` and unparks it. (A plain release/acquire pair
        // would allow both to miss each other.)
        self.count.fetch_add(1, Ordering::SeqCst);
        if self.parked.load(Ordering::SeqCst)
            && let Some(waiter) = self.waiter.get()
        {
            waiter.unpark();
        }
    }

    /// Waits until the count differs from `seen`, spinning for at most `spin`
    /// and parking after that. Returns the new count.
    pub fn wait_past(&self, seen: u64, spin: Duration) -> (u64, Woke) {
        let now = self.count.load(Ordering::Acquire);
        if now != seen {
            return (now, Woke::Spinning);
        }
        if !spin.is_zero() {
            let start = Instant::now();
            loop {
                // The clock is read once per 64 polls: reading it is as
                // costly as a poll and would double the latency of noticing.
                for _ in 0..64 {
                    spin_loop();
                    let now = self.count.load(Ordering::Acquire);
                    if now != seen {
                        return (now, Woke::Spinning);
                    }
                }
                if start.elapsed() >= spin {
                    break;
                }
            }
        }
        self.park_past(seen)
    }

    /// Parks until the count differs from `seen`.
    fn park_past(&self, seen: u64) -> (u64, Woke) {
        loop {
            self.parked.store(true, Ordering::SeqCst);
            let now = self.count.load(Ordering::SeqCst);
            if now != seen {
                self.parked.store(false, Ordering::SeqCst);
                return (now, Woke::Parked);
            }
            thread::park();
            // A spurious wake-up or a token left over from an earlier ring
            // brings us here with nothing new: look again and sleep again.
        }
    }
}

/// Counts how often waits ended by spinning and how often by being woken.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WakeCounts {
    /// Waits that ended while the waiter was still spinning.
    pub spun: u64,
    /// Waits that had to park.
    pub parked: u64,
}

impl WakeCounts {
    pub(crate) fn record(&mut self, woke: Woke) {
        match woke {
            Woke::Spinning => self.spun += 1,
            Woke::Parked => self.parked += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn a_ring_before_the_wait_is_seen_at_once() {
        let bell = Bell::new();
        bell.bind();
        bell.ring();
        // A wait that does not spin parks unless the ring is there already:
        // that it ended as a spin says that it did not sleep.
        assert_eq!(bell.wait_past(0, Duration::ZERO), (1, Woke::Spinning));
    }

    #[test]
    fn a_spinning_waiter_is_woken_without_parking() {
        let bell = Arc::new(Bell::new());
        let waiter = {
            let bell = bell.clone();
            thread::spawn(move || {
                bell.bind();
                bell.wait_past(0, Duration::from_secs(5))
            })
        };
        thread::sleep(Duration::from_millis(20));
        bell.ring();
        assert_eq!(waiter.join().unwrap(), (1, Woke::Spinning));
    }

    #[test]
    fn a_parked_waiter_is_woken_by_the_ring() {
        let bell = Arc::new(Bell::new());
        let waiter = {
            let bell = bell.clone();
            thread::spawn(move || {
                bell.bind();
                // No spinning: it parks at once, so the ring has to unpark it.
                bell.wait_past(0, Duration::ZERO)
            })
        };
        // The ring comes once the waiter has said that it parks, however long
        // its thread takes to get that far: a ring before that would be seen
        // without parking.
        let deadline = Instant::now() + Duration::from_secs(60);
        while !bell.parked.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the waiter never parked");
            thread::sleep(Duration::from_millis(1));
        }
        bell.ring();
        assert_eq!(waiter.join().unwrap(), (1, Woke::Parked));
    }

    #[test]
    fn many_rounds_of_ping_pong_lose_no_wake_up() {
        // Two bells, two threads, each waits for the other: if a ring could
        // be lost between the check and the park, this would hang.
        let (ping, pong) = (Arc::new(Bell::new()), Arc::new(Bell::new()));
        let worker = {
            let (ping, pong) = (ping.clone(), pong.clone());
            thread::spawn(move || {
                ping.bind();
                let mut seen = 0;
                for _ in 0..2000 {
                    seen = ping.wait_past(seen, Duration::ZERO).0;
                    pong.ring();
                }
            })
        };
        pong.bind();
        let mut seen = 0;
        for _ in 0..2000 {
            ping.ring();
            seen = pong.wait_past(seen, Duration::ZERO).0;
        }
        worker.join().unwrap();
    }
}
