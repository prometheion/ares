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
    /// The probability the drafter gave each of `drafts`, in the same order,
    /// from a drafter that reports it; empty otherwise. A link that the
    /// drafter did not make (it ended its chain there, see
    /// [`RecordView::draft_need`](crate::record::RecordView::draft_need)) has
    /// `u32::MAX` for its draft and no probability (NaN).
    pub draft_probs: Vec<f32>,
    /// Device milliseconds the model's own drafter took.
    pub draft_ms: f64,
    /// The rows of the step, in order, whose results are not what the rows
    /// would give alone ([`DeviceModel::rows_not_exact`](crate::DeviceModel::rows_not_exact)):
    /// no draft is accepted from such a row of a window on.
    pub inexact: Vec<u32>,
    /// What each device measured.
    pub devices: Vec<DeviceTimes>,
    /// Whether the model paused the step between two of its pieces
    /// ([`DeviceModel::paused`](crate::DeviceModel::paused)): nothing of it is
    /// done yet, and the same rows run again in a later step, which goes on where
    /// this one stopped. No pick comes back.
    pub paused: bool,
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

/// Host memory that one device's share of a transfer lies in, aligned for
/// direct I/O: a buffer on its way to the drive or back, or a page of the
/// prompt cache in RAM.
#[derive(Debug)]
pub enum HostBytes {
    /// Ordinary memory, outside the engine's books of page-locked RAM.
    Transit(ares_store::Aligned),
    /// A share of a page that the prompt cache keeps in RAM: in the books,
    /// and page-locked on an engine with devices.
    Cache(crate::ram::CacheBuffer),
}

impl std::ops::Deref for HostBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            HostBytes::Transit(buffer) => buffer,
            HostBytes::Cache(buffer) => buffer,
        }
    }
}

impl std::ops::DerefMut for HostBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            HostBytes::Transit(buffer) => buffer,
            HostBytes::Cache(buffer) => buffer,
        }
    }
}

/// The share of one device in a transfer: where its bytes are on the host.
#[derive(Debug)]
pub struct TransitPart {
    /// Memory aligned for direct I/O, with room for the device's share in
    /// any form.
    pub mem: HostBytes,
    /// The bytes of it that are in use, and the tag that says what form they
    /// have, which the device sets on the way out and reads on the way in.
    pub len: usize,
    pub tag: u32,
}

/// One movement of state between the devices and host memory, for the levels
/// of the prefix cache in RAM and on SSD ([`Executor::transfer`]).
#[derive(Debug)]
pub struct Transfer {
    pub kind: TransferKind,
    /// The pool page, or the checkpoint place.
    pub object: u32,
    /// For a page on the way in: it keeps the form it was stored in. A device
    /// that stores a page in more than one form (exact while it is new,
    /// converted later) may otherwise bring it in in the later one.
    pub keep_form: bool,
    /// For a page on the way in: it is a tail of the prefix cache, which holds
    /// fewer cells than a page and keeps its form. A device must not take it
    /// for a full page: a tail stays as it is until it is freed, since a
    /// prompt that continues it copies its leading cells.
    pub tail: bool,
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

    /// The image stage: where the conductor writes the rows of the image cells
    /// of the next step, `f32` rows of the model's width, which the record's
    /// image runs name. Only valid between [`Executor::collect`] and the next
    /// [`Executor::dispatch`]. Empty for a model without images.
    fn image_stage(&mut self) -> &mut [f32] {
        &mut []
    }

    /// Runs a job of the image tower on the device that holds it, between
    /// steps, and returns when it has run (with the image's rows when the job
    /// ends its pass). The default is an executor without a tower.
    fn tower(&mut self, _job: &mut crate::model::TowerJob) -> Result<(), ExecError> {
        Err(ExecError::Device { device: 0, message: "the executor runs no image tower".into() })
    }

    /// The count the engine raises whenever it is asked something (a request,
    /// a cancellation), which a model that runs a step in pieces watches to
    /// pause it for the conductor ([`StepContext::asks`](crate::StepContext));
    /// `None` for an executor whose models never pause.
    fn asks(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicU64>> {
        None
    }

    /// The books of the page-locked RAM the checkpoints lie in, which say
    /// how much of it they may hold. An executor without such memory has
    /// none.
    fn ram(&self) -> Option<crate::ram::PinnedRam> {
        None
    }

    /// Stops the devices and joins their threads. Idempotent.
    fn shutdown(&mut self);
}
