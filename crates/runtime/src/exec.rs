//! The interface between the conductor and whatever runs a step.
//!
//! The conductor decides what a step contains and writes it into the step
//! record; an [`Executor`] runs it and brings back the picks. The real
//! executor ([`GpuExecutor`](crate::GpuExecutor)) has a thread per device. The
//! test executor ([`crate::testing`]) computes the picks on the host, so
//! everything the conductor does can be tested without a GPU.

use std::time::{Duration, Instant};

use crate::sync::Woke;
use crate::timing::DeviceTimes;

/// The monotonic clock the threads of the engine stamp events with, so a
/// stamp taken on one thread can be compared with one from another.
#[derive(Clone, Copy, Debug)]
pub struct Clock(Instant);

impl Clock {
    pub fn new() -> Self {
        Self(Instant::now())
    }

    /// Nanoseconds since the clock was made.
    pub fn now_ns(&self) -> u64 {
        self.0.elapsed().as_nanos() as u64
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

/// What the conductor tells the executor about a step whose record it has
/// written.
#[derive(Clone, Copy, Debug)]
pub struct StepMeta {
    pub step: u64,
    /// The bytes of the record in use.
    pub record_bytes: usize,
    /// Whether the engine was idle before this step, so the devices waited
    /// for a request and not for the host.
    pub after_idle: bool,
}

/// The pick of one row: the token, and its logit when it is the most probable
/// one or its score (logit over temperature plus noise) when it was sampled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pick {
    pub token: u32,
    pub logit: f32,
}

/// What the devices brought back from a step.
#[derive(Clone, Debug, Default)]
pub struct StepOutput {
    /// One pick per row whose logits were wanted, in the order of the
    /// record's runs.
    pub picks: Vec<Pick>,
    /// The tokens the model drafted itself for the windows that asked for
    /// them, in the order of the record's segments, window after window.
    pub drafts: Vec<u32>,
    /// Device milliseconds the model's own drafter took.
    pub draft_ms: f64,
    /// What each device measured.
    pub devices: Vec<DeviceTimes>,
}

/// Why a step could not be run.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExecError {
    #[error("device {device}: {message}")]
    Device { device: usize, message: String },
    #[error("the executor has been shut down")]
    Closed,
}

/// Runs steps. Used by one thread, the conductor.
pub trait Executor: Send {
    /// The number of devices a step runs on.
    fn devices(&self) -> usize;

    /// The clock the executor's threads stamp their measurements with.
    fn clock(&self) -> Clock;

    /// The buffer the conductor writes the step record into. Only valid
    /// between [`Executor::collect`] and the next [`Executor::dispatch`].
    fn record(&mut self) -> &mut [u8];

    /// Starts the step whose record has been written. Returns without waiting
    /// for the devices.
    fn dispatch(&mut self, meta: &StepMeta) -> Result<(), ExecError>;

    /// Waits for every device to finish the step started by the last
    /// [`Executor::dispatch`], and fills `out`. The caller spins for `spin`
    /// before it parks, and is told how the wait ended.
    fn collect(&mut self, spin: Duration, out: &mut StepOutput) -> Result<Woke, ExecError>;

    /// How the devices' waits for work have ended so far, summed.
    fn device_wakes(&self) -> crate::sync::WakeCounts;

    /// Bytes of page-locked RAM the devices hold for checkpoints of recurrent
    /// state now. An executor that keeps no checkpoints holds none.
    fn checkpoint_ram(&self) -> u64 {
        0
    }

    /// Stops the devices and joins their threads. Idempotent.
    fn shutdown(&mut self);
}
