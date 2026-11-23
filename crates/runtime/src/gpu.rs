//! The executor with a thread per device.
//!
//! Each device has a thread that owns its stream and its part of the model.
//! The conductor writes the step record once, in page-locked memory, and hands
//! every device thread a small message; each thread then
//!
//! 1. enqueues the upload of the record (one copy) and the changes to the
//!    sequences' page tables,
//! 2. calls the model, which enqueues all of its kernels for the whole step,
//! 3. enqueues the read-back of the logits that are wanted (one copy: the
//!    model leaves them as consecutive rows),
//! 4. waits once, for the end of the step on its stream, and
//! 5. takes the arg-max of each row it brought back.
//!
//! Between 1 and 4 the thread never waits for the device. Between devices the
//! order is kept on the devices (events, peer kernels); the only host-side
//! meeting is [`Links`], which exists because `ares-cuda`'s sum and
//! hand-over are single calls over several streams.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ares_core::Ledger;
use ares_core::units::MIB;
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, Kernels, PinnedBuffer, SampleBatch, SamplePlan,
    SamplerKernels, Stream,
};
use ares_sampler::{LogprobOut, SamplePick, SampleRow};

use crate::cache::Cache;
use crate::config::RuntimeConfig;
use crate::error::RuntimeError;
use crate::exec::Transfer;
use crate::exec::{Clock, ExecError, Executor, Pick, StepMeta, StepOutput};
use crate::links::Links;
use crate::model::{
    DeviceContext, DeviceModel, DeviceTransfer, Model, ModelError, Split, StepContext, TowerJob,
    TransferContext,
};
use crate::ram::{Machine, PinnedRam};
use crate::record::{self, RecordView};
use crate::sync::{Bell, WakeCounts, Woke};
use crate::timing::DeviceTimes;

/// How the device threads are set up.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// The most sequences at once (page tables are kept per slot).
    pub max_slots: u32,
    /// Page-table entries per slot: the longest sequence in pages.
    pub table_stride: u32,
    /// How long a device thread spins for the next step before it parks.
    pub spin: Duration,
    /// VRAM left unreserved on each device.
    pub vram_headroom: u64,
    /// The engine's RAM limit: the most page-locked RAM the executor and the
    /// models' device parts lock together. What the executor needs to run
    /// ([`required_ram`]) is booked first; the rest is for the tier of
    /// experts of a model that has one and for the checkpoints of recurrent
    /// state, which are locked as they come into use. The buffers of the
    /// prefix cache's level on SSD are ordinary memory and not in this
    /// limit ([`crate::SsdConfig::transit_ram`]): who names a limit for all
    /// the engine's host memory takes them off it first.
    pub ram_limit: u64,
    /// The pool's pages, and the bytes of state they take on each device
    /// (the pages times the model's bytes per page there): reserved in the
    /// device's ledger and allocated before the model is built.
    pub pages: u32,
    pub page_bytes: Vec<u64>,
    /// The cells of a page, which the model lays its pages out for.
    pub cells_per_page: u32,
    /// The places for checkpoints of recurrent state on each device: as many
    /// as the conductor uses, which follow from its checkpoint budget.
    pub checkpoints: u32,
}

/// The page-locked RAM an executor over `model` needs to run at all: the step
/// record, the image stage and the read-back buffers of every device.
pub fn required_ram(model: &dyn Model, max_slots: u32) -> u64 {
    let max_rows = model.max_rows();
    record::max_bytes(max_rows, max_slots, model.logit_rows(), model.vocab()) as u64
        + image_stage_values(model) as u64 * 4
        + model.devices() as u64 * read_back_bytes(model.logit_rows(), max_rows) as u64
        + model.pinned_ram()
}

/// The most image rows one step takes: the stage holds that many. A step
/// with more image rows is cut there by the conductor; the rows a prompt's
/// kernels reach their rate at are far fewer.
pub const IMAGE_STAGE_ROWS: u32 = 1024;

/// The `f32` values of the image stage of an executor over `model`: none for
/// a model without images.
fn image_stage_values(model: &dyn Model) -> usize {
    model
        .vision()
        .map_or(0, |vision| model.max_rows().min(IMAGE_STAGE_ROWS) as usize * vision.width as usize)
}

impl GpuConfig {
    /// The defaults for a runtime of `config` over `model`: page tables for
    /// every slot, long enough for the model's whole context, and a RAM limit
    /// that holds what the executor needs and the prefix cache's RAM of
    /// `config` (for a model with a prefix cache).
    pub fn new(config: &RuntimeConfig, model: &dyn Model) -> Self {
        let cache = model.prefix_cache();
        let checkpoints = cache.as_ref().map_or(0, |spec| Cache::places(spec, config));
        let cache_ram = if cache.is_some() { config.cache_ram } else { 0 };
        Self {
            max_slots: config.max_slots,
            table_stride: model.card().context_length.div_ceil(config.cells_per_page),
            spin: Duration::from_millis(2),
            vram_headroom: 256 * MIB,
            ram_limit: required_ram(model, config.max_slots) + cache_ram,
            checkpoints,
            pages: config.pages,
            page_bytes: {
                let layout = model.page_layout(config.cells_per_page);
                (0..model.devices())
                    .map(|d| layout.bytes_per_page(d) * u64::from(config.pages))
                    .collect()
            },
            cells_per_page: config.cells_per_page,
        }
    }
}

/// The step record in page-locked memory, written by the conductor and read
/// by the device threads.
struct PinnedRecord {
    _owner: PinnedBuffer,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer is owned by this value and lives as long as it. Access
// follows the executor's protocol: the conductor writes only while no step
// is in flight, the device threads read only while one is.
unsafe impl Send for PinnedRecord {}
// SAFETY: as above.
unsafe impl Sync for PinnedRecord {}

impl PinnedRecord {
    fn new(device: &Device, len: usize) -> Result<Self, ares_cuda::Error> {
        let mut owner = PinnedBuffer::new(device, len.max(16))?;
        let ptr = owner.as_mut_ptr();
        Ok(Self { _owner: owner, ptr, len })
    }
}

enum Job {
    Step {
        meta: StepMeta,
        dispatched_ns: u64,
    },
    /// This device's shares of the transfers of a batch
    /// ([`Executor::transfer`]).
    Transfer(Vec<DeviceTransfer>),
    /// A job of the image tower, for the device that holds it.
    Tower(Box<TowerJob>),
    Quit,
}

/// What a device thread leaves for the conductor.
#[derive(Default)]
struct DeviceResult {
    error: Option<String>,
    /// The error is only that another device failed: the cause is theirs.
    secondary: bool,
    times: DeviceTimes,
    /// One pick per wanted row, when this device has an output head.
    picks: Vec<Pick>,
    /// The probabilities of the wanted rows, when any row asked for them.
    logprobs: Vec<LogprobOut>,
    has_picks: bool,
    /// The model paused the step ([`DeviceModel::paused`]).
    paused: bool,
    drafts: Vec<u32>,
    draft_probs: Vec<f32>,
    draft_ms: f64,
    inexact: Vec<u32>,
    /// What this device made of the transfers of a batch, in their order: the
    /// bytes and the tag of its share, or `None` for one it could not carry
    /// out.
    transfers: Vec<Option<(usize, u32)>>,
    /// A job of the image tower that has run, or why it could not.
    tower: Option<Result<Box<TowerJob>, String>>,
}

/// The hand-over between the conductor and one device thread.
struct Mailbox {
    bell: Bell,
    job: Mutex<Option<Job>>,
    result: Mutex<DeviceResult>,
    spun: AtomicU64,
    parked: AtomicU64,
}

struct Shared {
    /// Rung by the last device thread to finish a step.
    done: Bell,
    remaining: AtomicUsize,
    clock: Clock,
    /// What the engine was asked so far ([`StepContext::asks`]).
    asks: Arc<AtomicU64>,
}

struct Handle {
    mailbox: Arc<Mailbox>,
    thread: Option<JoinHandle<()>>,
}

/// An executor with a thread per device.
pub struct GpuExecutor {
    handles: Vec<Handle>,
    shared: Arc<Shared>,
    record: Arc<PinnedRecord>,
    /// The image stage, as many `f32` values as it holds.
    stage: Arc<PinnedRecord>,
    stage_values: usize,
    /// The device part that runs the image tower, for a model with one.
    tower_device: Option<usize>,
    links: Arc<Links>,
    seen: u64,
    in_flight: bool,
    /// The books of the page-locked memory, which the device parts lock their
    /// checkpoint places in.
    ram: PinnedRam,
}

impl std::fmt::Debug for GpuExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuExecutor").field("devices", &self.handles.len()).finish_non_exhaustive()
    }
}

/// What a device thread is built from.
struct Init {
    index: usize,
    model: Arc<dyn Model>,
    device: Device,
    stream: Arc<Stream>,
    kernels: Kernels,
    links: Arc<Links>,
    record: Arc<PinnedRecord>,
    stage: Arc<PinnedRecord>,
    mailbox: Arc<Mailbox>,
    shared: Arc<Shared>,
    ram: PinnedRam,
    config: GpuConfig,
    max_rows: u32,
}

impl GpuExecutor {
    /// Opens the model on `devices` (one part per device) and starts a thread
    /// for each. Returns when every part is built, or with the first error.
    /// The engine holds whatever its RAM limit allows.
    pub fn start(
        model: Arc<dyn Model>,
        devices: Vec<Device>,
        config: GpuConfig,
    ) -> Result<Self, RuntimeError> {
        Self::start_following(model, devices, config, None)
    }

    /// [`GpuExecutor::start`] for an engine that follows the memory of
    /// `machine` below its RAM limit: the tier of experts and the
    /// checkpoints hold what the machine has and give it back when it gets
    /// short.
    pub fn start_following(
        model: Arc<dyn Model>,
        devices: Vec<Device>,
        config: GpuConfig,
        machine: Option<Arc<Machine>>,
    ) -> Result<Self, RuntimeError> {
        if devices.len() != model.devices() {
            return Err(RuntimeError::Config(format!(
                "the model spans {} devices and {} were given",
                model.devices(),
                devices.len()
            )));
        }
        if model.logit_rows() < config.max_slots {
            return Err(RuntimeError::Config(format!(
                "the model gives the logits of {} rows per step, fewer than the {} sequences \
                 that can each want one",
                model.logit_rows(),
                config.max_slots
            )));
        }
        let max_rows = model.max_rows();
        let record_bytes =
            record::max_bytes(max_rows, config.max_slots, model.logit_rows(), model.vocab());

        // Page-locked memory is the host's scarcest kind; it is reserved
        // before it is taken.
        let ram = match &machine {
            Some(machine) => PinnedRam::following(config.ram_limit, machine),
            None => PinnedRam::new(config.ram_limit),
        };
        ram.require(required_ram(&*model, config.max_slots))?;
        // The pages of the prompt cache in RAM are locked for the device that
        // copies each share of them.
        ram.lock_for(devices.clone());

        let streams = devices
            .iter()
            .map(|d| Stream::new(d, 0).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        // The devices of a tensor split also add up the sampler's partial
        // results (each holds the logits of a slice of the vocabulary).
        let mut link_spec = model.links();
        if model.split() == Split::Tensor {
            link_spec.sum_values = link_spec
                .sum_values
                .max(SamplerKernels::exchange_values(model.logit_rows(), devices.len() as u32));
        }
        let links = Arc::new(Links::new(model.split(), &devices, &streams, link_spec)?);
        let record = Arc::new(PinnedRecord::new(&devices[0], record_bytes)?);
        let stage_values = image_stage_values(&*model);
        let stage = Arc::new(PinnedRecord::new(&devices[0], stage_values * 4)?);
        let tower_device = model.vision().map(|vision| vision.device);
        let shared = Arc::new(Shared {
            done: Bell::new(),
            remaining: AtomicUsize::new(0),
            clock: Clock::new(),
            asks: Arc::new(AtomicU64::new(0)),
        });

        let (ready_tx, ready_rx) = mpsc::channel();
        let mut handles = Vec::new();
        for (index, (device, stream)) in devices.into_iter().zip(streams).enumerate() {
            let mailbox = Arc::new(Mailbox {
                bell: Bell::new(),
                job: Mutex::new(None),
                result: Mutex::new(DeviceResult {
                    picks: Vec::with_capacity(max_rows as usize),
                    ..DeviceResult::default()
                }),
                spun: AtomicU64::new(0),
                parked: AtomicU64::new(0),
            });
            let kernels = Kernels::load(&device)?;
            let init = Init {
                index,
                model: model.clone(),
                device,
                stream,
                kernels,
                links: links.clone(),
                record: record.clone(),
                stage: stage.clone(),
                mailbox: mailbox.clone(),
                shared: shared.clone(),
                ram: ram.clone(),
                config: config.clone(),
                max_rows,
            };
            let ready = ready_tx.clone();
            let thread = thread::Builder::new()
                .name(format!("ares-device-{index}"))
                .spawn(move || worker_main(init, ready))
                .map_err(|e| RuntimeError::Thread(format!("cannot start a device thread: {e}")))?;
            handles.push(Handle { mailbox, thread: Some(thread) });
        }
        drop(ready_tx);

        let mut executor = Self {
            handles,
            shared,
            record,
            stage,
            stage_values,
            tower_device,
            links,
            seen: 0,
            in_flight: false,
            ram,
        };
        let mut failure = None;
        for _ in 0..executor.handles.len() {
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => failure = failure.or(Some(error)),
                // Every thread reports before it ends, a panic included
                // (`worker_main`), so a closed channel is not expected.
                Err(_) => {
                    failure = failure.or(Some(RuntimeError::Thread(
                        "a device thread ended without saying why while starting".into(),
                    )));
                }
            }
        }
        if let Some(error) = failure {
            executor.shutdown();
            return Err(error);
        }
        Ok(executor)
    }
}

impl Executor for GpuExecutor {
    fn devices(&self) -> usize {
        self.handles.len()
    }

    fn clock(&self) -> Clock {
        self.shared.clock
    }

    fn record(&mut self) -> &mut [u8] {
        assert!(!self.in_flight, "the record is written between collect and dispatch");
        // SAFETY: no step is in flight, so no device thread reads the record,
        // and `&mut self` makes this the only access from the conductor.
        unsafe { std::slice::from_raw_parts_mut(self.record.ptr, self.record.len) }
    }

    fn image_stage(&mut self) -> &mut [f32] {
        assert!(!self.in_flight, "the image stage is written between collect and dispatch");
        // SAFETY: as for the record: no step is in flight, so no device thread
        // reads the stage, and the buffer holds `stage_values` f32, aligned
        // (page-locked memory starts on a page).
        unsafe { std::slice::from_raw_parts_mut(self.stage.ptr.cast::<f32>(), self.stage_values) }
    }

    fn tower(&mut self, job: &mut TowerJob) -> Result<(), ExecError> {
        assert!(!self.in_flight, "the tower runs between steps");
        let Some(device) = self.tower_device else {
            return Err(ExecError::Device {
                device: 0,
                message: "the model has no image tower".into(),
            });
        };
        let handle = &self.handles[device];
        let taken = TowerJob {
            image: job.image.clone(),
            units: job.units.clone(),
            rows: std::mem::take(&mut job.rows),
            ms: 0.0,
        };
        self.shared.remaining.store(1, Ordering::SeqCst);
        *handle.mailbox.job.lock().expect("job slot") = Some(Job::Tower(Box::new(taken)));
        handle.mailbox.bell.ring();
        self.shared.done.bind();
        let (count, _) = self.shared.done.wait_past(self.seen, Duration::from_millis(1));
        self.seen = count;
        let outcome = handle.mailbox.result.lock().expect("result slot").tower.take();
        match outcome {
            Some(Ok(done)) => {
                job.rows = done.rows;
                job.ms = done.ms;
                Ok(())
            }
            Some(Err(message)) => Err(ExecError::Device { device, message }),
            None => Err(ExecError::Device {
                device,
                message: "the device thread left no outcome of the tower's job".into(),
            }),
        }
    }

    fn dispatch(&mut self, meta: &StepMeta) -> Result<(), ExecError> {
        assert!(!self.in_flight, "a step is already running");
        self.shared.remaining.store(self.handles.len(), Ordering::SeqCst);
        let dispatched_ns = self.shared.clock.now_ns();
        for handle in &self.handles {
            *handle.mailbox.job.lock().expect("job slot") =
                Some(Job::Step { meta: *meta, dispatched_ns });
            handle.mailbox.bell.ring();
        }
        self.in_flight = true;
        Ok(())
    }

    fn collect(&mut self, spin: Duration, out: &mut StepOutput) -> Result<Woke, ExecError> {
        assert!(self.in_flight, "no step is running");
        // The waiter is the thread that collects, not the one that built the
        // executor; the first call from it is what binds the bell, later
        // calls find it bound.
        self.shared.done.bind();
        let (count, woke) = self.shared.done.wait_past(self.seen, spin);
        self.seen = count;
        self.in_flight = false;
        out.picks.clear();
        out.logprobs.clear();
        out.drafts.clear();
        out.draft_probs.clear();
        out.draft_ms = 0.0;
        out.inexact.clear();
        out.devices.clear();
        out.paused = false;
        let mut have_picks = false;
        // The device that failed is the one to name: the others only report
        // that they were left waiting for it.
        let mut failed: Option<(usize, String, bool)> = None;
        for (device, handle) in self.handles.iter().enumerate() {
            let result = handle.mailbox.result.lock().expect("result slot");
            if let Some(message) = &result.error {
                let better = failed.as_ref().is_none_or(|(_, _, secondary)| *secondary);
                if better && (!result.secondary || failed.is_none()) {
                    failed = Some((device, message.clone(), result.secondary));
                }
            }
        }
        if let Some((device, message, _)) = failed {
            return Err(ExecError::Device { device, message });
        }
        for handle in &self.handles {
            let result = handle.mailbox.result.lock().expect("result slot");
            out.devices.push(result.times);
            // The devices of a tensor split all end with the same picks (the
            // sampler joins their slices of the vocabulary on the devices).
            if result.has_picks && !have_picks {
                out.picks.extend_from_slice(&result.picks);
                out.logprobs.extend_from_slice(&result.logprobs);
                have_picks = true;
            }
            if !result.drafts.is_empty() {
                out.draft_probs.extend_from_slice(&result.draft_probs);
                out.drafts.extend_from_slice(&result.drafts);
            }
            out.draft_ms += result.draft_ms;
            out.inexact.extend_from_slice(&result.inexact);
            out.paused |= result.paused;
        }
        out.inexact.sort_unstable();
        out.inexact.dedup();
        Ok(woke)
    }

    fn checkpoint_ram(&self) -> u64 {
        self.ram.checkpoints()
    }

    fn asks(&self) -> Option<Arc<AtomicU64>> {
        Some(self.shared.asks.clone())
    }

    fn transfer(&mut self, batch: &mut [Transfer]) -> Result<(), ExecError> {
        assert!(!self.in_flight, "transfers run between steps");
        if batch.is_empty() {
            return Ok(());
        }
        self.shared.remaining.store(self.handles.len(), Ordering::SeqCst);
        for (device, handle) in self.handles.iter().enumerate() {
            let shares = batch.iter_mut().map(|transfer| {
                let part = &mut transfer.parts[device];
                DeviceTransfer {
                    kind: transfer.kind,
                    object: transfer.object,
                    keep_form: transfer.keep_form,
                    ptr: part.mem.as_mut_ptr(),
                    room: part.mem.len(),
                    len: part.len,
                    tag: part.tag,
                }
            });
            *handle.mailbox.job.lock().expect("job slot") = Some(Job::Transfer(shares.collect()));
            handle.mailbox.bell.ring();
        }
        // The parts are the devices' until every one of them has rung.
        self.shared.done.bind();
        let (count, _) = self.shared.done.wait_past(self.seen, Duration::from_millis(1));
        self.seen = count;
        for transfer in batch.iter_mut() {
            transfer.ok = true;
        }
        for (device, handle) in self.handles.iter().enumerate() {
            let result = handle.mailbox.result.lock().expect("result slot");
            for (i, transfer) in batch.iter_mut().enumerate() {
                match result.transfers.get(i).copied().flatten() {
                    Some((len, tag)) => {
                        (transfer.parts[device].len, transfer.parts[device].tag) = (len, tag);
                    }
                    None => transfer.ok = false,
                }
            }
        }
        Ok(())
    }

    fn ram(&self) -> Option<PinnedRam> {
        Some(self.ram.clone())
    }

    fn device_wakes(&self) -> WakeCounts {
        self.handles.iter().fold(WakeCounts::default(), |sum, h| WakeCounts {
            spun: sum.spun + h.mailbox.spun.load(Ordering::Relaxed),
            parked: sum.parked + h.mailbox.parked.load(Ordering::Relaxed),
        })
    }

    fn shutdown(&mut self) {
        // Let any thread that waits on another give up first, so that a step
        // that failed on one device does not keep the others from quitting.
        // (If the kernels cannot be released, the bounded wait below ends
        // the shutdown all the same.)
        let _ = self.links.abort();
        for handle in &self.handles {
            *handle.mailbox.job.lock().expect("job slot") = Some(Job::Quit);
            handle.mailbox.bell.ring();
        }
        let until = Instant::now() + STOP_PATIENCE;
        for handle in &mut self.handles {
            if let Some(thread) = handle.thread.take() {
                while !thread.is_finished() && Instant::now() < until {
                    thread::sleep(Duration::from_millis(1));
                }
                // A thread that is still stuck in the driver is left behind:
                // waiting for it would hang the caller for good, and the
                // failure that led here has been reported already.
                if thread.is_finished() {
                    let _ = thread.join();
                }
            }
        }
    }
}

impl Drop for GpuExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// How long a device thread may take to stop once it is told to. A thread
/// that has nothing left to run stops at once, one that failed in a step is
/// released from the kernels that waited for a peer ([`Links::abort`]), so
/// the time is only for a device that has stopped answering.
const STOP_PATIENCE: Duration = Duration::from_secs(10);

/// The bytes a device brings back from a step at most: the pick of every
/// wanted row, the tokens the model drafted itself, and the probabilities the
/// wanted rows report.
fn read_back_bytes(logit_rows: u32, max_rows: u32) -> usize {
    logit_rows as usize * SamplePick::BYTES
        + max_rows as usize * 4
        + logit_rows as usize * LogprobOut::BYTES
}

// --- the device thread ---------------------------------------------------

struct Worker {
    index: usize,
    devices: usize,
    shared: Arc<Shared>,
    device: Device,
    stream: Arc<Stream>,
    kernels: Kernels,
    links: Arc<Links>,
    model: Box<dyn DeviceModel>,
    record: Arc<PinnedRecord>,
    stage: Arc<PinnedRecord>,
    record_device: DeviceBuffer,
    tables: DeviceBuffer,
    table_stride: u32,
    max_slots: u32,
    /// The state pages, if the model keeps any.
    pages: Option<DeviceBuffer>,
    bytes_per_page: u64,
    /// The sampler: the picks are made on the device, from the logits the
    /// model's part left there.
    sampler: SamplerKernels,
    /// Its scratch, the partial results the devices of a tensor split add
    /// up, and the picks.
    sample_scratch: DeviceBuffer,
    sample_exchange: DeviceBuffer,
    sample_out: DeviceBuffer,
    /// The probabilities the rows report.
    logprob_out: DeviceBuffer,
    /// For a part whose logit columns are not in the order of the tokens:
    /// the column of every token.
    inverse: Option<DeviceBuffer>,
    vocab: u32,
    /// The devices that each hold a slice of the vocabulary, and which of
    /// them this is.
    sample_devices: u32,
    sample_rank: u32,
    sample_rows: Vec<SampleRow>,
    /// What comes back: the picks, then the model's own drafts.
    back: PinnedBuffer,
    /// The most rows of a step whose logits are wanted.
    logit_rows: u32,
    max_rows: u32,
    start: Event,
    /// In front of the sampler's kernels.
    sample_start: Event,
    ends: [Event; 2],
    /// Which of `ends` the next step records.
    next_end: usize,
    have_previous: bool,
    clock: Clock,
    picks: Vec<Pick>,
    logprobs: Vec<LogprobOut>,
    drafts: Vec<u32>,
    draft_probs: Vec<f32>,
    draft_ms: f64,
    inexact: Vec<u32>,
}

/// A device in an error message: its name and the driver's number for it.
fn device_text(device: &Device) -> String {
    format!("{} (CUDA {})", device.info().name, device.ordinal())
}

/// What a panic carried, as text.
fn panic_text(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "without a message".to_owned())
}

fn worker_main(init: Init, ready: mpsc::Sender<Result<(), RuntimeError>>) {
    init.mailbox.bell.bind();
    let mailbox = init.mailbox.clone();
    let spin = init.config.spin;
    let (index, name, model) = (init.index, device_text(&init.device), init.model.clone());
    // Whatever goes wrong in this thread reaches the caller as an error that
    // names the device: a failure in the usual way, and a panic where panics
    // unwind (the release profile aborts instead, and the process says where).
    let start_failed = |cause: String| RuntimeError::Device {
        device: index,
        message: format!("{name}: could not be set up: {cause}"),
    };
    let mut worker = match panic::catch_unwind(AssertUnwindSafe(|| Worker::new(init))) {
        Ok(Ok(worker)) => {
            let _ = ready.send(Ok(()));
            worker
        }
        Ok(Err(error)) => {
            model.abandon(index);
            let _ = ready.send(Err(start_failed(error.to_string())));
            return;
        }
        Err(payload) => {
            model.abandon(index);
            let _ = ready.send(Err(start_failed(format!("panicked: {}", panic_text(&*payload)))));
            return;
        }
    };
    drop(ready);

    let mut seen = 0;
    loop {
        let (count, woke) = mailbox.bell.wait_past(seen, spin);
        seen = count;
        match woke {
            Woke::Spinning => mailbox.spun.fetch_add(1, Ordering::Relaxed),
            Woke::Parked => mailbox.parked.fetch_add(1, Ordering::Relaxed),
        };
        let job = mailbox.job.lock().expect("job slot").take();
        match job {
            Some(Job::Step { meta, dispatched_ns }) => {
                let outcome =
                    panic::catch_unwind(AssertUnwindSafe(|| worker.run_step(&meta, dispatched_ns)));
                // What a failure is: its cause, and whether it is only that
                // another device failed first.
                let outcome = match outcome {
                    Ok(Ok(done)) => Ok(done),
                    Ok(Err(error)) => {
                        Err((error.to_string(), matches!(error, ModelError::Aborted)))
                    }
                    Err(payload) => {
                        Err((format!("panicked in a step: {}", panic_text(&*payload)), false))
                    }
                };
                // A step that failed leaves the other devices with threads
                // and kernels that wait for this one: they are told to give
                // up, and the error says if that did not work.
                let released = outcome
                    .is_err()
                    .then(|| worker.links.abort_until_idle(&worker.stream, STOP_PATIENCE));
                {
                    let mut result = mailbox.result.lock().expect("result slot");
                    match outcome {
                        Ok((times, produced)) => {
                            result.error = None;
                            result.times = times;
                            result.has_picks = produced;
                            result.paused = worker.model.paused();
                            std::mem::swap(&mut result.picks, &mut worker.picks);
                            std::mem::swap(&mut result.logprobs, &mut worker.logprobs);
                            std::mem::swap(&mut result.drafts, &mut worker.drafts);
                            std::mem::swap(&mut result.draft_probs, &mut worker.draft_probs);
                            std::mem::swap(&mut result.inexact, &mut worker.inexact);
                            result.draft_ms = worker.draft_ms;
                        }
                        Err((cause, secondary)) => {
                            let note = match released {
                                Some(Err(e)) => format!(
                                    " (the kernels that wait for it could not be released: {e})"
                                ),
                                _ => String::new(),
                            };
                            result.error = Some(format!("{name}: {cause}{note}"));
                            result.secondary = secondary;
                        }
                    }
                }
                worker.finish();
            }
            Some(Job::Transfer(mut shares)) => {
                // A share that fails costs the cache that state; the stream
                // has passed every copy before the conductor reads a byte.
                let done = panic::catch_unwind(AssertUnwindSafe(|| {
                    let cx = TransferContext {
                        stream: &worker.stream,
                        pages: worker.pages.as_ref().map_or(DevicePtr::NULL, DeviceBuffer::ptr),
                    };
                    let mut done = Vec::with_capacity(shares.len());
                    for share in &mut shares {
                        let ok = worker.model.transfer(&cx, share).is_ok();
                        done.push(ok.then_some((share.len, share.tag)));
                    }
                    if worker.stream.synchronize().is_err() {
                        done.iter_mut().for_each(|share| *share = None);
                    }
                    done
                }));
                mailbox.result.lock().expect("result slot").transfers =
                    done.unwrap_or_else(|_| vec![None; shares.len()]);
                worker.finish();
            }
            Some(Job::Tower(mut job)) => {
                let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                    let begun = Instant::now();
                    worker.model.tower(&worker.stream, &mut job)?;
                    worker.stream.synchronize()?;
                    job.ms = begun.elapsed().as_secs_f64() * 1e3;
                    Ok::<_, ModelError>(())
                }));
                let outcome = match outcome {
                    Ok(Ok(())) => Ok(job),
                    Ok(Err(error)) => Err(format!("{name}: the image tower: {error}")),
                    Err(payload) => {
                        Err(format!("{name}: the image tower panicked: {}", panic_text(&*payload)))
                    }
                };
                mailbox.result.lock().expect("result slot").tower = Some(outcome);
                worker.finish();
            }
            Some(Job::Quit) => break,
            // A ring whose job was already taken (a quit racing a step).
            None => {}
        }
    }
    // Everything this thread enqueued must have run before its buffers go.
    let _ = worker.stream.synchronize();
}

impl Worker {
    fn new(init: Init) -> Result<Self, RuntimeError> {
        let Init {
            index,
            model,
            device,
            stream,
            kernels,
            links,
            record,
            stage,
            shared,
            ram,
            config,
            max_rows,
            ..
        } = init;
        let devices = model.devices();
        let free = device.mem_info()?.free;
        let mut ledger = Ledger::new(
            format!("CUDA{}", device.ordinal()),
            free.saturating_sub(config.vram_headroom),
        );
        let vocab = model.vocab();
        let record_len =
            record::max_bytes(max_rows, config.max_slots, model.logit_rows(), vocab) as u64;
        let tables_len = u64::from(config.max_slots) * u64::from(config.table_stride) * 4;
        ledger.reserve("step record (device copy)", record_len)?;
        ledger.reserve("page tables", tables_len)?;
        let page_bytes = config.page_bytes.get(index).copied().unwrap_or(0);
        if page_bytes > 0 {
            ledger.reserve("state pages", page_bytes)?;
        }
        let part = model
            .open(DeviceContext {
                index,
                devices,
                device: &device,
                kernels: &kernels,
                stream: &stream,
                ledger: &mut ledger,
                max_rows,
                max_slots: config.max_slots,
                pages: config.pages,
                cells_per_page: config.cells_per_page,
                table_stride: config.table_stride,
                checkpoints: config.checkpoints,
                ram: &ram,
            })
            .map_err(|e| match e {
                ModelError::Memory(e) => RuntimeError::Memory(e),
                other => RuntimeError::Model(other),
            })?;
        let logit_rows = model.logit_rows();
        let tensor = model.split() == Split::Tensor;
        if tensor && part.logits_per_row() == 0 {
            return Err(RuntimeError::Config(format!(
                "device {index} of a tensor split holds no slice of the vocabulary"
            )));
        }
        let sample_devices = if tensor { devices as u32 } else { 1 };
        let scratch_bytes = SamplerKernels::scratch_bytes(logit_rows);
        let exchange_bytes =
            u64::from(SamplerKernels::exchange_values(logit_rows, sample_devices)) * 4;
        let out_bytes = u64::from(logit_rows) * SamplePick::BYTES as u64;
        let logprob_bytes = u64::from(logit_rows) * LogprobOut::BYTES as u64;
        // A part whose columns are a list of tokens needs the column of each.
        let columns = part.logit_tokens();
        let inverse_bytes = if columns == DevicePtr::NULL { 0 } else { u64::from(vocab) * 4 };
        ledger.reserve(
            "sampler",
            scratch_bytes + exchange_bytes + out_bytes + logprob_bytes + inverse_bytes,
        )?;
        let sampler = SamplerKernels::load(&device)?;
        let sample_scratch = DeviceBuffer::new(&device, scratch_bytes)?;
        // SAFETY: the buffer has the size the sampler asks for.
        unsafe { sampler.clear(&stream, sample_scratch.ptr(), logit_rows)? };
        let inverse = if columns == DevicePtr::NULL {
            None
        } else {
            let inverse = DeviceBuffer::new(&device, inverse_bytes)?;
            // SAFETY: the part's list holds a token per column and the table
            // a column per token.
            unsafe {
                sampler.invert(&stream, columns, part.logits_per_row(), inverse.ptr(), vocab)?
            };
            Some(inverse)
        };
        let worker = Self {
            index,
            devices,
            clock: shared.clock,
            shared,
            record_device: DeviceBuffer::new(&device, record_len)?,
            tables: DeviceBuffer::new(&device, tables_len.max(16))?,
            table_stride: config.table_stride,
            max_slots: config.max_slots,
            pages: (page_bytes > 0).then(|| DeviceBuffer::new(&device, page_bytes)).transpose()?,
            bytes_per_page: page_bytes / u64::from(config.pages.max(1)),
            sampler,
            sample_scratch,
            sample_exchange: DeviceBuffer::new(&device, exchange_bytes)?,
            sample_out: DeviceBuffer::new(&device, out_bytes)?,
            logprob_out: DeviceBuffer::new(&device, logprob_bytes)?,
            inverse,
            vocab,
            sample_devices,
            sample_rank: if tensor { index as u32 } else { 0 },
            sample_rows: Vec::with_capacity(logit_rows as usize),
            back: PinnedBuffer::new(&device, read_back_bytes(logit_rows, max_rows))?,
            logit_rows,
            max_rows,
            start: Event::timed(&device)?,
            sample_start: Event::timed(&device)?,
            ends: [Event::timed(&device)?, Event::timed(&device)?],
            next_end: 0,
            have_previous: false,
            picks: Vec::with_capacity(max_rows as usize),
            logprobs: Vec::with_capacity(logit_rows as usize),
            drafts: Vec::with_capacity(max_rows as usize),
            draft_probs: Vec::with_capacity(max_rows as usize),
            draft_ms: 0.0,
            inexact: Vec::with_capacity(max_rows as usize),
            device,
            stream,
            kernels,
            links,
            model: part,
            record,
            stage,
        };
        // Page tables are never cleared in normal operation (a table decides
        // what is valid), but a defined start makes a bug show as a wrong
        // page and not as garbage.
        // SAFETY: the buffer is `tables_len` bytes on the stream's device.
        unsafe { worker.stream.fill(worker.tables.ptr(), 0, tables_len as usize)? };
        worker.stream.synchronize()?;
        Ok(worker)
    }

    /// Where the probabilities start in the read-back buffer: after the picks
    /// and the model's own drafts.
    fn logprobs_at(&self) -> usize {
        self.logit_rows as usize * SamplePick::BYTES + self.max_rows as usize * 4
    }

    /// Tells the conductor this device has finished the step; the last one to
    /// finish wakes it.
    fn finish(&self) {
        if self.shared.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.shared.done.ring();
        }
    }

    /// Runs one step on this device. Returns what was measured and whether
    /// this device brought picks back.
    fn run_step(
        &mut self,
        meta: &StepMeta,
        dispatched_ns: u64,
    ) -> Result<(DeviceTimes, bool), ModelError> {
        let mut times = DeviceTimes::default();
        let begun = Instant::now();
        times.wake_ns = self.clock.now_ns().saturating_sub(dispatched_ns);

        // SAFETY: the conductor wrote `meta.record_bytes` bytes of the record
        // before it handed over this step and does not touch the buffer until
        // every device thread has finished it.
        let bytes = unsafe { std::slice::from_raw_parts(self.record.ptr, meta.record_bytes) };
        let view = RecordView::new(bytes)
            .map_err(|e| ModelError::Invalid(format!("the step record is damaged: {e}")))?;

        // The upload: the record, then the changes to the page tables.
        let stream = &*self.stream;
        self.start.record(stream)?;
        // SAFETY: `bytes` is page-locked memory that stays unchanged until the
        // step has finished; the destination holds `record_bytes` bytes.
        unsafe { stream.copy_from_host(self.record_device.ptr(), bytes.as_ptr(), bytes.len())? };
        let entries_at = view.entries_offset_words() as u64;
        let mut taken = 0u64;
        for i in 0..view.updates() {
            let update = view.update(i);
            if update.slot >= self.max_slots || update.first + update.count > self.table_stride {
                return Err(ModelError::Invalid(format!(
                    "page-table update {update:?} is outside {} slots of {} entries",
                    self.max_slots, self.table_stride
                )));
            }
            let to = self.tables.ptr().offset(
                (u64::from(update.slot) * u64::from(self.table_stride) + u64::from(update.first))
                    * 4,
            );
            let from = self.record_device.ptr().offset((entries_at + taken) * 4);
            // SAFETY: both ranges are inside their buffers (checked above and
            // by the record's layout) and do not overlap.
            unsafe { stream.copy(to, from, update.count as usize * 4)? };
            taken += u64::from(update.count);
        }
        times.upload_ns = begun.elapsed().as_nanos() as u64;

        // The model's kernels for the whole step.
        let enqueue_begun = Instant::now();
        let logits = self.model.enqueue(&StepContext {
            index: self.index,
            devices: self.devices,
            device: &self.device,
            kernels: &self.kernels,
            stream,
            record: view,
            record_on_device: self.record_device.ptr(),
            tables: self.tables.ptr(),
            table_stride: self.table_stride,
            pages: self.pages.as_ref().map_or(ares_cuda::DevicePtr::NULL, DeviceBuffer::ptr),
            bytes_per_page: self.bytes_per_page,
            links: &self.links,
            image_stage: if self.stage.len > 0 {
                self.stage.ptr.cast::<f32>().cast_const()
            } else {
                std::ptr::null()
            },
            asks: &self.shared.asks,
        })?;

        // The picks of the rows whose logits are wanted, made on the device,
        // and their read-back with the tokens the model drafted itself.
        let picked = view.picked_rows();
        if picked > self.logit_rows {
            return Err(ModelError::Invalid(format!(
                "the step wants the logits of {picked} rows; the model gives at most {}",
                self.logit_rows
            )));
        }
        let produced = logits.is_some() && picked > 0;
        let mut wants_logprobs = false;
        if let (Some(logits), true) = (logits, picked > 0) {
            self.sample_rows.clear();
            self.sample_rows.extend((0..picked).map(|i| view.sampler(i)));
            let plan = SamplePlan::of(&self.sample_rows);
            wants_logprobs = plan.logprobs();
            let tokens = self.model.logit_tokens();
            let record = self.record_device.ptr();
            let (masks, mask_words) = view.masks();
            let batch = SampleBatch {
                logits: logits.ptr,
                stride: logits.stride,
                n_rows: picked,
                capacity: self.logit_rows,
                n_local: logits.stride,
                tokens,
                first: if tokens == DevicePtr::NULL { logits.first_token } else { 0 },
                inverse: self.inverse.as_ref().map_or(DevicePtr::NULL, DeviceBuffer::ptr),
                vocab: self.vocab,
                params: record.offset(view.samplers_offset()),
                entries: if view.corrections() > 0 {
                    record.offset(view.corrections_offset())
                } else {
                    DevicePtr::NULL
                },
                masks: if masks > 0 { record.offset(view.masks_offset()) } else { DevicePtr::NULL },
                mask_words,
                out: self.sample_out.ptr(),
                logprobs: self.logprob_out.ptr(),
                scratch: self.sample_scratch.ptr(),
                exchange: self.sample_exchange.ptr(),
                devices: self.sample_devices,
                rank: self.sample_rank,
            };
            let (links, rank) = (&self.links, self.index);
            self.sample_start.record(stream)?;
            // SAFETY: the model left a row of `stride` logits per wanted row
            // (and a token per column when it has a list); the record on the
            // device holds a sampler row per wanted row, and the corrections
            // and masks they name; the sampler's buffers were sized for
            // `logit_rows` rows.
            unsafe {
                self.sampler.sample(stream, &batch, plan, |ptr, n| links.sum(rank, ptr, n))?;
                stream.copy_to_host(
                    self.back.as_mut_ptr(),
                    self.sample_out.ptr(),
                    picked as usize * SamplePick::BYTES,
                )?;
                if wants_logprobs {
                    stream.copy_to_host(
                        self.back.as_mut_ptr().add(self.logprobs_at()),
                        self.logprob_out.ptr(),
                        picked as usize * LogprobOut::BYTES,
                    )?;
                }
            }
        }
        let own = self.model.model_drafts().filter(|own| own.tokens > 0);
        if let Some(own) = own {
            if own.tokens > self.max_rows {
                return Err(ModelError::Invalid(format!(
                    "the model drafted {} tokens in a step of at most {} rows",
                    own.tokens, self.max_rows
                )));
            }
            // SAFETY: the model left `tokens` u32 there; the read-back buffer
            // holds a draft per row of a step after the picks.
            unsafe {
                stream.copy_to_host(
                    self.back.as_mut_ptr().add(self.logit_rows as usize * SamplePick::BYTES),
                    own.ptr,
                    own.tokens as usize * 4,
                )?;
            }
        }
        let end = &self.ends[self.next_end];
        end.record(stream)?;
        times.enqueue_ns = enqueue_begun.elapsed().as_nanos() as u64;

        // The one wait of the step.
        let wait_begun = Instant::now();
        self.model.wait(end)?;
        times.readback_ns = wait_begun.elapsed().as_nanos() as u64;

        times.device_ns = (f64::from(end.elapsed_ms_since(&self.start)?) * 1e6) as u64;
        if self.have_previous && !meta.after_idle {
            let previous = &self.ends[1 - self.next_end];
            times.gap_ns =
                self.start.elapsed_ms_since(previous).ok().map(|ms| (f64::from(ms) * 1e6) as u64);
        }
        self.have_previous = true;
        self.next_end = 1 - self.next_end;

        // What came back. The time of the picks is the sampler's on the
        // device (to the end of the step: its read-back follows it) and the
        // host's to read them.
        let sampler_ns = if produced {
            (f64::from(end.elapsed_ms_since(&self.sample_start)?) * 1e6) as u64
        } else {
            0
        };
        let pick_begun = Instant::now();
        self.picks.clear();
        self.logprobs.clear();
        self.drafts.clear();
        if produced {
            for row in 0..picked as usize {
                let pick = SamplePick::from_bytes(&self.back[row * SamplePick::BYTES..]);
                // A logit that is not finite means a sum or a kernel went
                // wrong, and is an error rather than a pick.
                if pick.status != 0 || pick.token == u32::MAX {
                    return Err(ModelError::Invalid(format!(
                        "row {row}: the sampler reported status {:#x} (1: a logit that is not \
                         finite, 2: no token to pick)",
                        pick.status
                    )));
                }
                self.picks.push(Pick { token: pick.token, logit: pick.value, aux: pick.aux });
                if wants_logprobs {
                    let at = self.logprobs_at() + row * LogprobOut::BYTES;
                    self.logprobs.push(LogprobOut::from_bytes(&self.back[at..]));
                }
            }
        }
        if let Some(own) = own {
            let at = self.logit_rows as usize * SamplePick::BYTES;
            let bytes = &self.back[at..at + own.tokens as usize * 4];
            self.drafts.extend(bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)));
        }
        // What the drafter says of its drafts: a link it did not make is no
        // draft, whatever the device holds in its place.
        self.draft_probs.clear();
        self.model.model_draft_probs(&mut self.draft_probs);
        if self.draft_probs.len() == self.drafts.len() {
            for (draft, p) in self.drafts.iter_mut().zip(&self.draft_probs) {
                if p.is_nan() {
                    *draft = u32::MAX;
                }
            }
        } else {
            self.draft_probs.clear();
        }
        if produced {
            self.model.picked(&self.picks);
        }
        self.draft_ms = self.model.model_draft_ms();
        self.inexact.clear();
        self.model.rows_not_exact(&mut self.inexact);
        times.pick_ns = sampler_ns + pick_begun.elapsed().as_nanos() as u64;
        times.finished_at_ns = self.clock.now_ns();
        Ok((times, produced))
    }
}

#[cfg(test)]
mod tests {
    use crate::{SyntheticConfig, SyntheticModel};

    use super::*;

    fn model(prefix_cache: Option<u64>) -> SyntheticModel {
        let mut config = SyntheticConfig::small(Split::Single, 1);
        config.prefix_cache = prefix_cache;
        SyntheticModel::new(config).unwrap()
    }

    #[test]
    fn the_devices_get_as_many_places_as_the_conductor_counts_and_room_in_the_ram_limit() {
        let config = RuntimeConfig { cache_ram: 10_250, ..RuntimeConfig::default() };
        let cached = model(Some(1000));
        let gpu = GpuConfig::new(&config, &cached);
        // Ten whole checkpoints fit the budget; the conductor counts the same.
        assert_eq!(gpu.checkpoints, 10);
        assert_eq!(gpu.checkpoints, Cache::places(&cached.prefix_cache().unwrap(), &config));
        let needed = required_ram(&cached, config.max_slots);
        assert!(needed > 0);
        assert_eq!(gpu.ram_limit, needed + 10_250, "what the executor needs and the cache's");

        // A model without a cache, or a budget of none, has no places.
        assert_eq!(GpuConfig::new(&config, &model(None)).checkpoints, 0);
        let none = RuntimeConfig { cache_ram: 0, ..config };
        let gpu = GpuConfig::new(&none, &cached);
        assert_eq!((gpu.checkpoints, gpu.ram_limit), (0, needed));
    }
}
