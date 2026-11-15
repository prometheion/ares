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
    /// For a mirostat row, the surprise of the token in bits.
    pub aux: f32,
}

/// What the devices brought back from a step.
#[derive(Clone, Debug, Default)]
pub struct StepOutput {
    /// One pick per row whose logits were wanted, in the order of the
    /// record's runs.
    pub picks: Vec<Pick>,
    /// The probabilities those rows report, one record per pick, when any row
    /// asked for them; empty otherwise.
    pub logprobs: Vec<ares_sampler::LogprobOut>,
    /// The tokens the model drafted itself for the windows that asked for
    /// them, in the order of the record's segments, window after window.
    pub drafts: Vec<u32>,
    /// Device milliseconds the model's own drafter took.
    pub draft_ms: f64,
    /// What each device measured.
    pub devices: Vec<DeviceTimes>,
    /// The milliseconds the step took, the drafter's included, from an executor
    /// that knows them without a clock: the conductor takes them for the time
    /// of the step in place of what it measured. Devices leave it `None`; the
    /// test executor says what a test told it a step costs, so that what the
    /// conductor decides from the prices does not depend on how busy the
    /// machine is.
    pub step_ms: Option<f64>,
}

/// What a transfer moves, and which way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferKind {
    /// The state per position of a pool page, from the devices to the host.
    PageOut,
    /// The same, into a pool page that no sequence reads yet.
    PageIn,
    /// A checkpoint of recurrent state, from its place to the host.
    StateOut,
    /// The same, into a place.
    StateIn,
}

/// The share of one device in a transfer: where its bytes are on the host.
#[derive(Debug)]
pub struct TransitPart {
    /// Memory aligned for direct I/O, with room for the device's share in
    /// any form.
    pub mem: ares_store::Aligned,
    /// The bytes of it that are in use, and the tag that says what form they
    /// have, which the device sets on the way out and reads on the way in.
    pub len: usize,
    pub tag: u32,
}

/// One movement of state between the devices and host memory, for the level
/// of the prefix cache that is on SSD ([`Executor::transfer`]).
#[derive(Debug)]
pub struct Transfer {
    pub kind: TransferKind,
    /// The pool page, or the checkpoint place.
    pub object: u32,
    /// For a page on the way in: it keeps the form it was stored in. A device
    /// that stores a page in more than one form (exact while it is new,
    /// converted later) may otherwise bring it in in the later one.
    pub keep_form: bool,
    /// A part per device.
    pub parts: Vec<TransitPart>,
    /// Set by the executor: every device carried it out.
    pub ok: bool,
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

    /// Carries out `batch` on every device, in order, and returns when the
    /// bytes are where they go. Called between steps: after
    /// [`Executor::collect`] and before the next [`Executor::dispatch`]. A
    /// transfer a device could not carry out comes back with `ok` unset, which
    /// costs the cache that state and nothing else. The default is an
    /// executor whose devices move nothing: every transfer fails.
    fn transfer(&mut self, batch: &mut [Transfer]) -> Result<(), ExecError> {
        for transfer in batch {
            transfer.ok = false;
        }
        Ok(())
    }

    /// Stops the devices and joins their threads. Idempotent.
    fn shutdown(&mut self);
}
