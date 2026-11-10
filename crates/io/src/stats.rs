//! Counters the engine can read at any time.

use std::time::Duration;

/// What one class of reads (demand or prefetch) has done. A read counts under
/// the class it was submitted with, even if it was promoted later.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassStats {
    /// Reads submitted.
    pub requests: u64,
    /// Reads that finished, successfully or not, including cancelled ones.
    pub completed: u64,
    /// Reads that finished with an error (not counting cancelled ones).
    pub failed: u64,
    /// Wanted bytes delivered by successful reads.
    pub bytes: u64,
    /// Bytes the device was actually asked to deliver and delivered: widened
    /// to blocks, and including reads that were cancelled half way.
    pub device_bytes: u64,
}

/// Percentiles of the latencies of the most recent successful reads of one
/// class, from submission to completion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Latency {
    /// Successful reads recorded since the reader was opened.
    pub total: u64,
    /// Reads the percentiles are computed over (the most recent
    /// [`Latency::WINDOW`]).
    pub samples: usize,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

impl Latency {
    /// How many of the latest reads the percentiles cover.
    pub const WINDOW: usize = 4096;
}

/// A snapshot of the reader's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub demand: ClassStats,
    pub prefetch: ClassStats,
    /// Reads cancelled with [`Reader::cancel`](crate::Reader::cancel) in time
    /// to count (their completion carries [`Error::Cancelled`](crate::Error::Cancelled)).
    pub cancelled: u64,
    /// Prefetches turned into demand reads.
    pub promoted: u64,
    /// Reads with chunks still waiting to be started, per class.
    pub queued_demand: usize,
    pub queued_prefetch: usize,
    /// Chunks at the device right now, and the most there have been.
    pub in_flight: usize,
    pub peak_in_flight: usize,
    /// Reads submitted whose completion has not been taken yet.
    pub outstanding: usize,
    /// Latency of demand reads, the number the engine's step time depends on.
    pub demand_latency: Latency,
    pub prefetch_latency: Latency,
}

/// The last [`Latency::WINDOW`] latencies, in nanoseconds.
#[derive(Debug)]
pub(crate) struct LatencyRing {
    samples: Vec<u64>,
    next: usize,
    total: u64,
}

impl LatencyRing {
    pub(crate) fn new() -> Self {
        Self { samples: Vec::with_capacity(Latency::WINDOW), next: 0, total: 0 }
    }

    pub(crate) fn record(&mut self, latency: Duration) {
        let ns = u64::try_from(latency.as_nanos()).unwrap_or(u64::MAX);
        if self.samples.len() < Latency::WINDOW {
            self.samples.push(ns);
        } else {
            self.samples[self.next] = ns;
            self.next = (self.next + 1) % Latency::WINDOW;
        }
        self.total += 1;
    }

    /// A copy of the window to sort outside the lock.
    pub(crate) fn snapshot(&self) -> (u64, Vec<u64>) {
        (self.total, self.samples.clone())
    }
}

/// The percentiles of a window of samples (nearest rank).
pub(crate) fn summarize((total, mut samples): (u64, Vec<u64>)) -> Latency {
    if samples.is_empty() {
        return Latency { total, ..Latency::default() };
    }
    samples.sort_unstable();
    let rank = |p: f64| {
        let i = ((p * samples.len() as f64).ceil() as usize).clamp(1, samples.len()) - 1;
        Duration::from_nanos(samples[i])
    };
    Latency {
        total,
        samples: samples.len(),
        p50: rank(0.50),
        p95: rank(0.95),
        p99: rank(0.99),
        max: Duration::from_nanos(*samples.last().expect("not empty")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_use_the_nearest_rank() {
        let mut ring = LatencyRing::new();
        for ms in 1..=100 {
            ring.record(Duration::from_millis(ms));
        }
        let l = summarize(ring.snapshot());
        assert_eq!(l.total, 100);
        assert_eq!(l.samples, 100);
        assert_eq!(l.p50, Duration::from_millis(50));
        assert_eq!(l.p95, Duration::from_millis(95));
        assert_eq!(l.p99, Duration::from_millis(99));
        assert_eq!(l.max, Duration::from_millis(100));
    }

    #[test]
    fn the_window_forgets_old_samples() {
        let mut ring = LatencyRing::new();
        for _ in 0..Latency::WINDOW {
            ring.record(Duration::from_secs(10));
        }
        for _ in 0..Latency::WINDOW {
            ring.record(Duration::from_millis(1));
        }
        let l = summarize(ring.snapshot());
        assert_eq!(l.total, 2 * Latency::WINDOW as u64);
        assert_eq!(l.samples, Latency::WINDOW);
        assert_eq!(l.max, Duration::from_millis(1), "the slow reads left the window");
    }

    #[test]
    fn no_samples_give_zeroes() {
        assert_eq!(summarize(LatencyRing::new().snapshot()), Latency::default());
    }
}
