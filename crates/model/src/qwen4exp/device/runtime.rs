//! `qwen4exp` as the runtime drives it: the [`Model`] that tells the runtime
//! what it is and builds a [`Part`] on each device's thread, and the
//! [`DeviceModel`] that turns the runtime's step into the part's.
//!
//! # How a step waits for experts
//!
//! The experts of a layer are known when its router has run, and an expert
//! that is in no slot has to be brought (uploaded, or computed by the CPU)
//! while the device waits for it. So a step is not enqueued blind to its end,
//! and the host has work in it besides enqueuing: following the route records
//! the routers leave and serving the expert store.
//!
//! - A device thread enqueues its part layer by layer and stays at most
//!   [`LEAD_LAYERS`] ahead of the layers whose records the store has handled:
//!   the driver's queue of commands is finite, and a thread that blocks in a
//!   launch can no longer bring the expert its device waits for.
//! - The store has one owner at a time. It sits behind a lock that no thread
//!   waits for in a step: whoever finds it free serves it for all parts
//!   (hands over the records that have arrived, polls reads and copies), and
//!   whoever does not goes on. A thread serves it whenever it would
//!   otherwise wait: for the lead, for the rows of the n-gram table, and for
//!   the end of its step ([`DeviceModel::wait`], which the runtime calls
//!   where it would otherwise block in the driver).
//! - The conductor sees nothing of this. It dispatches a step and collects it
//!   when every device has ended; a step that misses experts is a longer step.
//!   Cancellation and new requests are handled between steps, as for any
//!   model.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, TryLockError};
use std::time::{Duration, Instant};

use ares_api::{ChatFormat, EngineError, ExpertStats, ModelCard, Prompt};
use ares_core::Ledger;
use ares_cuda::{Device, DevicePtr, ENTRY_CELLS, Event, MAX_BATCH, Segment, StepRecord, Stream};
use ares_gguf::Gguf;
use ares_io::Reader;
use ares_runtime::expert_store::{
    CudaSlots, HostRegion, PIECE, Serve, StoreConfig, StorePlan, StoreStats,
};
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, LinkSpec, Logits, Model, ModelError, OwnDrafts, Pick,
    PinnedRam, Split, StepContext,
};
use ares_state::PageLayout;
use ares_tokenizer::{ChatOptions, Tokenizer};

use super::board::Board;
use super::draft::{Chain, DraftBlock, DraftReport, DraftTimes, Drafted, Watch};
use super::feed::{
    DraftLayer, Feed, Layout, build_store, cpu_pool, device_slots, expert_map, places,
    plan_transit, reader,
};
use super::misses::{Misses, Progress, WideStats};
use super::ngram::NgramFeed;
use super::part::{
    ExpertAt, Frame, Options, PAGE_ENTRY, Part, PartSetup, PartShape, Ports, StepEnv,
};
use super::placement::{MARGIN, PartPlan, Placement};
use super::session::{Chosen, Fill, PartReport, SessionOptions, choose};
use super::usage::{Log, ProfileReport, Usage, UsageStats};
use crate::chat;
use crate::qwen4exp::weights::{DraftWeights, Weights};
use crate::qwen4exp::{Config, Error as FileError};

/// How many layers a device thread enqueues ahead of the last layer of its
/// part whose route record the store has handled.
///
/// A step is not enqueued blind to its end: a layer that misses an expert
/// makes the device wait for the host, and the driver's queue of commands for
/// a device is finite. A host that goes on enqueuing the rest of the step (a
/// layer is about 30 launches, a step of 48 layers 1 300) fills that queue,
/// blocks inside a launch, and can then no longer bring the expert the device
/// waits for. Seen as a step that never ends, on one device from the first
/// prompt step on, on two devices with a few hundred timed events more per
/// step. The host enqueues much faster than a device computes, so staying
/// this far ahead never leaves a device without work.
pub(super) const LEAD_LAYERS: usize = 6;

/// How long a part waits for the others to arrive while they open: the
/// longest thing between two meetings is the upload of a device's weights
/// from a cold drive.
const MEET_LIMIT: Duration = Duration::from_secs(1800);

/// How long a failed step serves the store so that the kernels that wait for
/// an expert can end, before the device threads leave it.
const DRAIN: Duration = Duration::from_secs(5);

/// Why a model could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error(transparent)]
    File(#[from] FileError),
    #[error(transparent)]
    Device(#[from] super::Error),
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A choice that a model on the runtime cannot carry out.
    #[error("{0}")]
    Choice(String),
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

fn model_error(error: super::Error) -> ModelError {
    match error {
        super::Error::Model(e) => e,
        other => invalid(other.to_string()),
    }
}

// --- the parts that open together -------------------------------------------

/// Threads that build their parts at the same time and wait for each other at
/// a few points: the plan of the expert store needs every device's room, the
/// store needs every device's slots, and a part needs to know where its peer
/// receives.
struct Meeting<T> {
    state: Mutex<MeetState<T>>,
    changed: Condvar,
}

struct MeetState<T> {
    value: T,
    arrived: usize,
    generation: u64,
    failed: Option<String>,
}

impl<T> Meeting<T> {
    fn new(value: T) -> Self {
        Self {
            state: Mutex::new(MeetState { value, arrived: 0, generation: 0, failed: None }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, MeetState<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Lets every thread that waits give up: one of the parts failed.
    fn fail(&self, message: String) {
        let mut state = self.lock();
        state.failed.get_or_insert(message);
        self.changed.notify_all();
    }

    /// `add` runs on the shared value for each of `n` threads as it arrives;
    /// the last of them also runs `last`. Returns when all have arrived and
    /// `last` has run, or with an error when any of them failed.
    fn meet(
        &self,
        n: usize,
        add: impl FnOnce(&mut T) -> Result<(), ModelError>,
        last: impl FnOnce(&mut T) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let gone = |state: &MeetState<T>| {
            state.failed.as_ref().map(|why| invalid(format!("another part failed: {why}")))
        };
        let mut state = self.lock();
        if let Some(error) = gone(&state) {
            return Err(error);
        }
        let mut outcome = add(&mut state.value);
        if outcome.is_ok() {
            state.arrived += 1;
            if state.arrived == n {
                outcome = last(&mut state.value);
                state.arrived = 0;
                state.generation += 1;
            }
        }
        if let Err(error) = outcome {
            state.failed.get_or_insert(error.to_string());
            self.changed.notify_all();
            return Err(error);
        }
        if state.arrived == 0 {
            self.changed.notify_all();
            return Ok(());
        }
        let generation = state.generation;
        let started = Instant::now();
        while state.generation == generation {
            if let Some(error) = gone(&state) {
                return Err(error);
            }
            let left = MEET_LIMIT.saturating_sub(started.elapsed());
            if left.is_zero() {
                return Err(invalid("the other parts of the model did not arrive"));
            }
            state = self.changed.wait_timeout(state, left).unwrap_or_else(|e| e.into_inner()).0;
        }
        // The last to arrive may have failed in what it made of the others'.
        gone(&state).map_or(Ok(()), Err)
    }
}

/// What the parts hand each other while they open.
struct Build {
    /// Per part: the VRAM it has left for slots after its dense parts and the
    /// store's tables, where its peer writes, which layers it computes the
    /// recurrent or attention half of, and its route records.
    room: Vec<u64>,
    ports: Vec<Option<Ports>>,
    cores: Vec<Vec<bool>>,
    boards: Vec<Option<Arc<Board>>>,
    plan: Option<PlanBox>,
    slots: Vec<Option<CudaSlots>>,
    places: Vec<Vec<ExpertAt>>,
    /// Per part: the expert slots it has in VRAM, and their bytes.
    n_slots: Vec<usize>,
    slot_bytes: Vec<u64>,
    flow: Option<Arc<Flow>>,
    /// The route records of the draft block, and where its experts are.
    draft_board: Option<Arc<Board>>,
    draft_place: Option<ExpertAt>,
}

/// The plan of the expert store and what it was made from.
struct PlanBox {
    plan: StorePlan,
    map: ares_io::ExpertMap,
    reader: Reader,
    /// The bytes of the RAM tier that are locked at the start.
    ram_start: u64,
}

/// How long the tier of experts locks pieces at a time when it may hold
/// more and no step runs: the thread that watches the machine's memory does
/// it, it must look again soon, and a step that begins waits for it.
const GROW_SLICE: Duration = Duration::from_millis(40);

// --- what the device threads share in a step --------------------------------

/// The expert store, the rows of the n-gram table, and the counters that tell
/// the threads how far each part's step has got.
pub(super) struct Flow {
    feed: Mutex<Feed>,
    ngram: Mutex<NgramFeed>,
    /// The tokens the draft block picked for the rows of a step, as (row,
    /// token), and the step they are for: the device thread that holds the
    /// block tells the others, which need them for the embeddings and the
    /// rows of the n-gram table.
    drafts: Mutex<(u64, Vec<(u32, u32)>)>,
    /// Per part: the layers of the running step that it has enqueued, and the
    /// layers whose route records the store has handled.
    enqueued: Vec<AtomicUsize>,
    followed: Vec<AtomicUsize>,
    /// The step the store has begun, and how many parts have ended it.
    begun: AtomicU64,
    finished: AtomicUsize,
    /// Set by a part whose step failed, so that the others give up.
    failed: AtomicBool,
    /// Uploads and reads the store had finished the last time anyone looked.
    uploads_seen: AtomicU64,
    stall: Duration,
    /// What the store counted, summed over the times it was read.
    totals: Mutex<ExpertStats>,
    /// What the wide steps found and waited for, since the engine started.
    wide: Mutex<WideStats>,
}

impl Flow {
    fn parts(&self) -> usize {
        self.enqueued.len()
    }

    /// Adds what the wide steps found since the last call to the totals. The experts of a
    /// layer that a wide step found in a slot are hits and the others are picks the store
    /// counts when it brings them (as misses, never as hits).
    fn fold_wide(&self) {
        let wide = self.lock_feed().take_wide();
        let found = wide.picked - wide.missing;
        let mut totals = self.totals.lock().unwrap_or_else(|e| e.into_inner());
        totals.picks += found;
        totals.hits += found;
        totals.wait_us +=
            (wide.compute_wait + wide.bring_wait + wide.round_wait).as_micros() as u64;
        drop(totals);
        self.wide.lock().unwrap_or_else(|e| e.into_inner()).add(&wide);
    }

    fn fail(&self) {
        self.failed.store(true, Ordering::SeqCst);
    }

    pub(super) fn lock_feed(&self) -> MutexGuard<'_, Feed> {
        self.feed.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The tier of experts in RAM follows `allowance`, the bytes it may hold.
    /// Called by the thread that watches the machine's memory. A running
    /// step gives pieces up, and takes one, when it ends
    /// ([`Flow::finish`]); when no step runs, this gives them up and locks
    /// pieces for a while, and no step begins meanwhile: the driver's unlock
    /// waits for the device, and a step's kernels can wait for the store.
    fn follow_ram(&self, ram: &PinnedRam, allowance: u64) {
        let mut feed = self.lock_feed();
        feed.store.set_ram_target(allowance);
        if feed.store.waiting() > 0 {
            return;
        }
        if feed.store.shrink_ram(Duration::ZERO).is_err() {
            self.fail();
        }
        drop(feed.store.take_released());
        if feed.store.ram_at_floor() {
            ram.experts_floor(feed.store.ram_bytes().0);
        }
        let begun = Instant::now();
        while begun.elapsed() < GROW_SLICE && !feed.store.in_step() {
            if !matches!(feed.store.grow_ram(1), Ok(1)) {
                break;
            }
        }
    }

    fn lock_ngram(&self) -> MutexGuard<'_, NgramFeed> {
        self.ngram.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A step begins on part `index` (device `index`, whose stream is
    /// `stream`): the first part to arrive begins it for the store, every
    /// part makes its stream wait for what the store removed from the tables
    /// since the step before.
    /// `windows` are the step's windows that verify drafts, as (first row,
    /// rows), with what each row of the step is worth (the record's row
    /// worth): their rows past the first wait for the file only while they
    /// are worth it. A step that
    /// only drafts (`windows` is `None`) is no step of the store: its routing
    /// is the draft block's, whose experts are resident, and the work between
    /// two steps of the store (the heat's decay, the slots that move, the RAM
    /// tier that grows) is counted in the model's steps.
    fn begin(
        &self,
        step: u64,
        rows: u32,
        windows: Option<StepWindows<'_>>,
        index: usize,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        let mut feed = self.lock_feed();
        if self.begun.load(Ordering::Acquire) != step {
            for counter in self.enqueued.iter().chain(&self.followed) {
                counter.store(0, Ordering::Release);
            }
            self.finished.store(0, Ordering::Release);
            if let Some((windows, worth)) = windows {
                feed.begin_step(rows).map_err(model_error)?;
                feed.store.set_windows(windows, worth);
            }
            self.begun.store(step, Ordering::Release);
        }
        feed.store.device_mut(index).fence(stream).map_err(|e| invalid(e.to_string()))
    }

    fn enqueued_layers(&self) -> Vec<usize> {
        self.enqueued.iter().map(|n| n.load(Ordering::Acquire)).collect()
    }

    /// Hands the route records that have arrived to the store and lets it and
    /// the reads of table rows work, if no other thread is doing that now.
    /// Never waits.
    pub(super) fn pump(&self) -> Result<(), ModelError> {
        match self.feed.try_lock() {
            Ok(mut feed) => {
                let enqueued = self.enqueued_layers();
                feed.pump(&enqueued).map_err(model_error)?;
                for (counter, followed) in self.followed.iter().zip(feed.followed()) {
                    counter.store(*followed, Ordering::Release);
                }
            }
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Poisoned(_)) => {
                return Err(invalid("the expert store was left broken by a failed step"));
            }
        }
        if let Ok(mut ngram) = self.ngram.try_lock() {
            ngram.poll();
        }
        Ok(())
    }

    /// A number that changes whenever a layer is enqueued or followed or the store finishes
    /// an upload or a read. A wide step that waits for the other part's layers makes no
    /// progress of its own, and has stood still only when this does not change either.
    fn progress(&self) -> u64 {
        if let Ok(feed) = self.feed.try_lock() {
            let stats = feed.store.stats();
            let moved = stats.uploads
                + stats.demand_reads
                + stats.transit_uploads
                + stats.transit_reads
                + stats.transit_ahead_reads;
            self.uploads_seen.store(moved, Ordering::Relaxed);
        }
        let layers: usize =
            self.enqueued.iter().chain(&self.followed).map(|n| n.load(Ordering::Acquire)).sum();
        layers as u64 + self.uploads_seen.load(Ordering::Relaxed)
    }

    /// An error when another part failed, or when the step has stood still
    /// since `started` for longer than a step may.
    pub(super) fn check(&self, started: Instant, what: &str) -> Result<(), ModelError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ModelError::Aborted);
        }
        if started.elapsed() < self.stall {
            return Ok(());
        }
        self.fail();
        let enqueued = self.enqueued_layers();
        let state = match self.feed.try_lock() {
            Ok(feed) => feed.standing_still(&enqueued),
            Err(_) => "the expert store is busy".to_owned(),
        };
        Err(invalid(format!("a step stood still for {:.0?} while {what}: {state}", self.stall)))
    }

    /// Serves the store until `part` has no layer enqueued more than the lead
    /// ahead of the records it has handled.
    fn throttle(&self, part: usize) -> Result<(), ModelError> {
        let started = Instant::now();
        while self.enqueued[part].load(Ordering::Acquire)
            >= self.followed[part].load(Ordering::Acquire) + LEAD_LAYERS
        {
            self.check(started, "the host held back the next layer")?;
            self.pump()?;
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Waits for the rows of the n-gram table of the step, serving the store
    /// meanwhile, and returns them.
    fn await_rows(&self) -> Result<Vec<u8>, ModelError> {
        let started = Instant::now();
        loop {
            {
                let mut ngram = self.lock_ngram();
                ngram.poll();
                if ngram.is_done() {
                    return Ok(ngram.rows().map_err(model_error)?.to_vec());
                }
            }
            self.check(started, "rows of the n-gram table were read")?;
            self.pump()?;
            std::hint::spin_loop();
        }
    }

    /// Serves the store for a while so that kernels that wait for an expert
    /// can end: after a step failed, before its buffers are freed.
    fn drain(&self, idle: impl Fn() -> bool) {
        let started = Instant::now();
        while !idle() && started.elapsed() < DRAIN {
            if self.pump().is_err() {
                break;
            }
            std::hint::spin_loop();
        }
    }

    /// The draft block has made the drafts of step `step`.
    fn publish(&self, step: u64, drafts: Vec<(u32, u32)>) {
        *self.drafts.lock().unwrap_or_else(|e| e.into_inner()) = (step, drafts);
    }

    /// Waits for the drafts of step `step`, serving the store meanwhile.
    fn await_drafts(&self, step: u64) -> Result<Vec<(u32, u32)>, ModelError> {
        let started = Instant::now();
        loop {
            {
                let drafts = self.drafts.lock().unwrap_or_else(|e| e.into_inner());
                if drafts.0 == step {
                    return Ok(drafts.1.clone());
                }
            }
            self.check(started, "the draft block drafted on another device")?;
            self.pump()?;
            std::hint::spin_loop();
        }
    }

    /// A part has ended its step on the device: the last of them ends it for
    /// the store, unless it only drafted (`store` false).
    fn finish(&self, store: bool) -> Result<(), ModelError> {
        if self.finished.fetch_add(1, Ordering::AcqRel) + 1 == self.parts() && store {
            let mut feed = self.lock_feed();
            feed.end_step().map_err(model_error)?;
            // What the tier gave up at the end of this step is unlocked
            // here, where every device has ended the step and none has
            // begun the next.
            drop(feed.store.take_released());
        }
        Ok(())
    }
}

/// The windows of a step that verify drafts, as (first row, rows), and what
/// every row of the step is worth in milliseconds (the record's row worth).
type StepWindows<'a> = (&'a [(u32, u32)], Option<&'a [f32]>);

// --- the embedding table on the host ----------------------------------------

/// The token embeddings, decoded on the host: the first part of the model
/// starts the residual streams from them.
#[derive(Clone)]
pub(super) struct Embedding {
    gguf: Arc<Gguf>,
    tensor: usize,
    ty: ares_core::GgmlType,
    rows: usize,
}

impl Embedding {
    pub(super) fn new(gguf: &Arc<Gguf>, w: &Weights<'_>) -> Result<Self, ModelError> {
        let name = "token_embd.weight";
        let tensor = gguf
            .tensors()
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| invalid(format!("the model has no tensor {name}")))?;
        Ok(Self { gguf: gguf.clone(), tensor, ty: w.token_embd.ty(), rows: w.token_embd.rows() })
    }

    /// Decodes the embedding of `token` into `out`.
    pub(super) fn decode(&self, token: u32, out: &mut [f32]) -> Result<(), ModelError> {
        let data = self.gguf.data(&self.gguf.tensors()[self.tensor]);
        let size = data.len() / self.rows;
        let row = data
            .get(token as usize * size..(token as usize + 1) * size)
            .ok_or_else(|| invalid(format!("token {token} of {} embeddings", self.rows)))?;
        ares_quant::dequantize(self.ty, row, out)
            .map_err(|e| invalid(format!("the embedding of token {token}: {e}")))
    }
}

// --- the model --------------------------------------------------------------

/// The rows of a step to open the model with on `devices`: as many as the scratch of a step
/// may take of the smaller device's memory (a thirty-second; the rest is slots for experts and
/// the pools of K and V), as a power of two between eight (the decode kernels' limit, which
/// is all that a device without the tile kernels runs) and 4096.
///
/// A step of more rows reads the experts of a layer once for more of them, and every byte of
/// the scratch is a byte less for expert slots, which the generating steps miss: measured on
/// Flash-Next, 2048 rows (1.4 GiB of scratch on a 12 GiB card) cost a third to a half of the
/// generation speed through the slots they took. The scratch grows with the rows, about 0.7 MiB
/// a row for Flash-Next; `opts` supplies the other choices of the part, whose `max_rows` is
/// ignored.
pub fn step_rows(gguf: &Gguf, devices: &[Device], opts: &Options) -> Result<u32, OpenError> {
    let config = Config::from_gguf(gguf)?;
    if devices.iter().any(|d| d.info().compute_capability.0 < 8) {
        return Ok(MAX_BATCH);
    }
    let room = devices.iter().map(|d| d.info().total_memory / 32).min().unwrap_or(0);
    let mut rows = WIDE_ROWS;
    while rows > MAX_BATCH {
        let opts = Options { max_rows: rows, ..opts.clone() };
        // The scratch of the part with everything: the widest a part is.
        let shape = PartShape::whole(&config, 0..config.n_layer, true);
        let frame = Frame::new(&config, &opts, &shape, 1)?;
        if frame.step_bytes(&opts) <= room {
            break;
        }
        rows /= 2;
    }
    Ok(rows)
}

/// The most rows of a step the model is opened with.
const WIDE_ROWS: u32 = 4096;

/// Where the layers of the model in `gguf` would go on `devices` with `opts`,
/// and the bytes one page of the pools would take on each device. Nothing is
/// allocated; the placement of two devices that share every layer measures a
/// read of a stream mixer on each.
pub fn place(
    gguf: &Gguf,
    devices: &[Device],
    opts: &SessionOptions,
) -> Result<(Placement, Vec<u64>), OpenError> {
    let config = Config::from_gguf(gguf)?;
    let weights = Weights::load(gguf, &config)?;
    place_in(&config, &weights, devices, opts)
}

fn place_in(
    config: &Config,
    weights: &Weights<'_>,
    devices: &[Device],
    opts: &SessionOptions,
) -> Result<(Placement, Vec<u64>), OpenError> {
    let Chosen { placement, .. } = choose(
        config,
        weights,
        devices,
        &opts.part,
        opts.headroom,
        &opts.spread,
        opts.bounds.as_deref(),
    )?;
    let pages = u64::from(opts.part.pages());
    let mut page_bytes = Vec::with_capacity(devices.len());
    for (index, device) in devices.iter().enumerate() {
        let PartPlan { role, first, shares, head } = placement.part(config, index);
        let shape = PartShape { role, first, shares, head };
        let frame = Frame::new(config, &opts.part, &shape, device.info().multiprocessors)?;
        page_bytes.push(frame.pool_bytes().div_ceil(pages));
    }
    Ok((placement, page_bytes))
}

/// A `qwen4exp` model file, placed.
pub struct Qwen4Exp {
    gguf: Arc<Gguf>,
    config: Config,
    opts: SessionOptions,
    placement: Placement,
    layout: Layout,
    tokenizer: Tokenizer,
    name: String,
    context: u32,
    created: u64,
    chat: ChatOptions,
    /// What the chat template declares, read with `chat`.
    format: ChatFormat,
    meeting: Meeting<Build>,
    flow: OnceLock<Arc<Flow>>,
    reports: Mutex<Vec<PartReport>>,
    /// The page-locked memory of the parts, besides the tier of experts.
    pinned: u64,
    /// The bytes one page of the pools takes on each device.
    page_bytes: Vec<u64>,
    /// The file of the model's draft block, when it drafts.
    draft: Option<Arc<Gguf>>,
    draft_report: Mutex<Option<DraftReport>>,
    draft_watch: Arc<Mutex<Watch>>,
    /// The usage profile of the experts, for a model that keeps one.
    usage: Usage,
}

impl std::fmt::Debug for Qwen4Exp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4Exp")
            .field("name", &self.name)
            .field("devices", &self.placement.n_devices())
            .field("shared", &self.placement.tensor.is_some())
            .finish_non_exhaustive()
    }
}

impl Qwen4Exp {
    /// Reads the configuration of `gguf` and decides the placement on
    /// `devices`. `tokenizer` is the model's (`Tokenizer::from_gguf`);
    /// `context` is the longest sequence the model is served with
    /// (`opts.part.cells`); the pools hold `opts.part.pool_pages` pages in
    /// all. Nothing is uploaded until the runtime opens the parts.
    pub fn new(
        gguf: Gguf,
        tokenizer: Tokenizer,
        name: impl Into<String>,
        devices: &[Device],
        mut opts: SessionOptions,
        context: u32,
    ) -> Result<Self, OpenError> {
        let choice = |why: &str| Err(OpenError::Choice(why.to_owned()));
        if opts.ram_tier.is_some() || opts.routing.is_some() || opts.worker_picks.is_some() {
            return choice(
                "the RAM tier of a model on the runtime comes from the engine's RAM limit, and \
                 its experts are divided by a rule, not by a recorded routing",
            );
        }
        if opts.part.pool_pages == 0 {
            return choice("the pools of K/V and keys need a number of pages");
        }
        let config = Config::from_gguf(&gguf)?;
        let context = context.clamp(1, config.n_ctx as u32);
        if opts.part.cells != context {
            return choice("the cells of a sequence are the context of the model");
        }
        // The sampler reads the logits on the device.
        opts.part.host_logits = false;
        let weights = Weights::load(&gguf, &config)?;
        let (placement, page_bytes) = place_in(&config, &weights, devices, &opts)?;
        let (layout, _map) = Layout::new(&gguf, &config, &weights, &placement, None)?;
        let mut pinned = 0;
        for (index, device) in devices.iter().enumerate() {
            let PartPlan { role, first, shares, head } = placement.part(&config, index);
            let layers = shares.len();
            let shape = PartShape { role, first, shares, head };
            let frame = Frame::new(&config, &opts.part, &shape, device.info().multiprocessors)?;
            let shares = if opts.serve == Serve::Stream {
                0
            } else {
                (layers * layout.per_layer) as u64
                    * layout.shapes[0].host_share_bytes(opts.part.max_rows.min(MAX_BATCH))
            };
            // The entries the store copies through are a ring of 256 KiB.
            pinned += frame.pinned_bytes(&opts.part, layers) + shares + (256 << 10);
        }
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        let n = placement.n_devices();
        let format = chat::chat_format(&tokenizer, &ChatOptions::default());
        Ok(Self {
            gguf: Arc::new(gguf),
            config,
            opts,
            placement,
            layout,
            tokenizer,
            name: name.into(),
            context,
            created,
            format,
            chat: ChatOptions::default(),
            meeting: Meeting::new(Build {
                room: vec![0; n],
                ports: vec![None; n],
                cores: vec![Vec::new(); n],
                boards: vec![None; n],
                plan: None,
                slots: (0..n).map(|_| None).collect(),
                places: vec![Vec::new(); n],
                n_slots: vec![0; n],
                slot_bytes: vec![0; n],
                flow: None,
                draft_board: None,
                draft_place: None,
            }),
            flow: OnceLock::new(),
            reports: Mutex::new(Vec::new()),
            pinned,
            page_bytes,
            draft: None,
            draft_report: Mutex::new(None),
            draft_watch: Arc::default(),
            usage: Usage::default(),
        })
    }

    /// Keeps the usage profile of the model's experts in `dir` (see
    /// `usage.rs`): a profile that is there for this model file fills the
    /// tiers when the engine begins to serve ([`Qwen4Exp::serving`]), and the
    /// profile is written while the engine serves and when the model is
    /// dropped. `log` gets what the owner should hear of (a profile that
    /// cannot be written). Before the runtime opens the parts.
    pub fn with_usage_profile(mut self, dir: &std::path::Path, log: Log) -> Self {
        self.usage = Usage::open(dir, &self.gguf, &self.config, log);
        self
    }

    /// The engine serves from now on: the heat of the experts becomes the
    /// usage profile's (the steps that measured the price of a step are no
    /// text), the tiers are filled from it, and the profile is kept, the
    /// empty slots filled whenever nothing runs. Call it once the runtime
    /// runs and before the first request; it returns when the tiers are
    /// filled, with what was found and filled; `None` without a profile, or
    /// before the runtime has opened the parts.
    pub fn serving(&self) -> Option<ProfileReport> {
        self.usage.serving()
    }

    /// Writes the usage profile now if steps ran since it was last written:
    /// its path, or why it could not be written; `None` when there was
    /// nothing to write.
    pub fn save_profile(&self) -> Option<Result<std::path::PathBuf, String>> {
        self.usage.save()
    }

    /// What the fill of the empty slots did while nothing ran, and what the
    /// tiers of the model's layers hold; `None` without a usage profile.
    pub fn usage_stats(&self) -> Option<UsageStats> {
        self.usage.stats()
    }

    /// Gives the model its draft block from `draft`, a file of the block as
    /// Strata packs it ([`DraftWeights`]): windows then take the block's
    /// drafts, as many rows as `opts.part.window_rows` allows. The block runs
    /// on the device that holds the output head, in every placement, and its
    /// routed experts go through the expert store with the model's. Before the
    /// runtime opens the parts.
    pub fn with_draft(mut self, draft: Gguf) -> Result<Self, OpenError> {
        let choice = |why: &str| Err(OpenError::Choice(why.to_owned()));
        if self.opts.part.window_rows < 2 {
            return choice("a draft block needs windows of at least two rows");
        }
        if self.opts.part.window_rows > MAX_BATCH {
            return choice(
                "a step that verifies drafts runs the decode kernels, which take eight rows",
            );
        }
        let (c, n) = (&self.config, self.placement.n_devices());
        let head = DraftWeights::load(&draft, c)?;
        let weights = Weights::load(&self.gguf, c)?;
        let device = self.head_device();
        let layer = DraftLayer { gguf: &draft, weights: &head, device };
        let (layout, _map) = Layout::new(&self.gguf, c, &weights, &self.placement, Some(&layer))?;
        // The block's host share and its staged rows and picks.
        if self.opts.serve != Serve::Stream {
            self.pinned += layout.shapes[0].host_share_bytes(MAX_BATCH);
        }
        self.pinned += u64::from(MAX_BATCH) * c.wide() as u64 * 4 + (64 << 10);
        self.layout = layout;
        debug_assert!(device < n);
        drop((head, weights));
        self.draft = Some(Arc::new(draft));
        Ok(self)
    }

    /// The device part that holds the final mixer and the output head.
    fn head_device(&self) -> usize {
        (0..self.placement.n_devices())
            .find(|d| self.placement.part(&self.config, *d).head)
            .unwrap_or(0)
    }

    /// What the draft block took on its device, once the runtime has opened
    /// the parts; `None` for a model without one.
    pub fn draft_report(&self) -> Option<DraftReport> {
        *self.draft_report.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the draft block's steps took and how its drafts fared since the
    /// engine started.
    pub fn draft_times(&self) -> DraftTimes {
        self.draft_watch.lock().unwrap_or_else(|e| e.into_inner()).times
    }

    /// The newest chains the draft block made, oldest first (the last 1024),
    /// and forgets them: for tests and measurements that compare its drafts
    /// with the reference's.
    pub fn take_drafted(&self) -> Vec<Drafted> {
        self.draft_watch.lock().unwrap_or_else(|e| e.into_inner()).recent.drain(..).collect()
    }

    /// Renders chat prompts with `options`: whether the model thinks, and the
    /// variables a server passes to the chat template.
    pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
        self.format = chat::chat_format(&self.tokenizer, &options);
        self.chat = options;
        self
    }

    /// The model's configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// What the parts took, by device, once the runtime has opened them.
    pub fn reports(&self) -> Vec<PartReport> {
        let mut reports = self.reports.lock().unwrap_or_else(|e| e.into_inner()).clone();
        reports.sort_by_key(|r| r.index);
        reports
    }

    /// What the wide steps found missing and waited for since the engine started; `None`
    /// before the parts are open.
    pub fn wide_stats(&self) -> Option<WideStats> {
        let flow = self.flow.get()?;
        flow.fold_wide();
        Some(*flow.wide.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// What the expert store counted since the last call; `None` before the
    /// parts are open.
    pub fn take_store_stats(&self) -> Option<StoreStats> {
        let flow = self.flow.get()?;
        flow.fold_wide();
        let (stats, ram_bytes, held) = {
            let mut feed = flow.lock_feed();
            let held = self.layout.model_layers().fold((0, 0), |(vram, ram), at| {
                let (in_vram, in_ram) = feed.store.held(at);
                (vram + in_vram as u64, ram + in_ram as u64)
            });
            (feed.store.take_stats(), feed.store.ram_bytes().0, held)
        };
        let mut totals = flow.totals.lock().unwrap_or_else(|e| e.into_inner());
        totals.ram_tier_bytes = ram_bytes;
        (totals.vram_experts, totals.ram_experts) = held;
        totals.picks += stats.picks;
        totals.hits += stats.hits;
        totals.from_ram += stats.miss_from_ram;
        totals.from_disk += stats.miss_from_disk;
        totals.on_their_way += stats.miss_on_its_way;
        totals.cpu_experts += stats.cpu_experts;
        totals.uploads += stats.uploads + stats.transit_uploads;
        totals.read_bytes +=
            stats.demand_read_bytes + stats.guess_read_bytes + stats.transit_read_bytes;
        totals.wait_us += (stats.wait_ms * 1e3) as u64;
        Some(stats)
    }

    /// The slots of the experts in VRAM and in the RAM tier, over all layers,
    /// and the bytes of the RAM tier, as they are now (the RAM tier follows
    /// the memory the machine has); `None` before the parts are open.
    pub fn expert_slots(&self) -> Option<(usize, usize, u64)> {
        let flow = self.flow.get()?;
        let feed = flow.lock_feed();
        let (vram, ram) = self.layout.model_layers().fold((0, 0), |(v, r), at| {
            let (vram, ram) = feed.store.slots(at);
            (v + vram, r + ram)
        });
        let ram_bytes = feed.store.ram_bytes().0;
        Some((vram, ram, ram_bytes))
    }

    /// The slots of the draft block's experts in VRAM and in the RAM tier;
    /// `None` without a draft block or before the parts are open.
    pub fn draft_slots(&self) -> Option<(usize, usize)> {
        self.draft.as_ref()?;
        Some(self.flow.get()?.lock_feed().store.slots(0))
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers before anything is served, most wanted first, as far as the slots
    /// go (a recorded routing, or every expert for a model that fits); the
    /// draft block is asked for as the layer behind the model's last. Call it
    /// while no request runs; `None` before the parts are open.
    pub fn fill(
        &self,
        experts_of: &mut dyn FnMut(usize) -> Vec<u32>,
    ) -> Option<Result<Fill, ModelError>> {
        let flow = self.flow.get()?;
        let placement = &self.placement;
        let c = &self.config;
        Some(
            flow.lock_feed()
                .fill(
                    c.n_layer,
                    c.experts.count,
                    &|layer, expert| placement.owner(layer, expert),
                    experts_of,
                )
                .map_err(model_error),
        )
    }

    fn open_part(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        let (c, o) = (&self.config, &self.opts.part);
        let (index, n) = (cx.index, self.placement.n_devices());
        if cx.max_slots != o.max_slots
            || cx.pages != o.pages()
            || cx.table_stride != o.table_stride()
            || cx.cells_per_page != ENTRY_CELLS
            || cx.max_rows != o.max_rows
        {
            return Err(invalid(format!(
                "the runtime was set up for {} slots, {} pages of {} cells, tables of {} \
                 entries and steps of {} rows; the model for {}, {}, {ENTRY_CELLS}, {} and {}",
                cx.max_slots,
                cx.pages,
                cx.cells_per_page,
                cx.table_stride,
                cx.max_rows,
                o.max_slots,
                o.pages(),
                o.table_stride(),
                o.max_rows
            )));
        }
        let weights = Weights::load(&self.gguf, c).map_err(|e| invalid(e.to_string()))?;
        let PartPlan { role, first, shares, head } = self.placement.part(c, index);
        let layer_count = shares.len();
        let mut part = Part::open(
            c,
            &weights,
            o,
            PartSetup { index, role, first, shares, head, device: cx.device, ledger: cx.ledger },
        )?;
        let serve = self.opts.serve;
        let layout = &self.layout;
        // The draft block, next to the part that holds the head: before the
        // room for slots is counted.
        let embedding = Embedding::new(&self.gguf, &weights)?;
        let mut block = match (&self.draft, part.output()) {
            (Some(draft), Some(output)) if index == self.head_device() => {
                let head = DraftWeights::load(draft, c).map_err(|e| invalid(e.to_string()))?;
                let block = DraftBlock::open(
                    c,
                    &head,
                    o,
                    index,
                    output,
                    embedding.clone(),
                    self.draft_watch.clone(),
                    cx.device,
                    cx.ledger,
                )?;
                *self.draft_report.lock().unwrap_or_else(|e| e.into_inner()) = Some(block.report());
                Some(block)
            }
            _ => None,
        };
        let tables = 4 * layout.specs.len() as u64 * c.experts.count as u64;
        let host_shares = if serve == Serve::Stream {
            0
        } else {
            layout.specs.len() as u64
                * layout.shapes[0].host_share_bytes(o.max_rows.min(MAX_BATCH)).next_multiple_of(16)
        };
        let room = cx.ledger.free().saturating_sub(MARGIN + tables + host_shares);
        let (ram, device) = (cx.ram, cx.device);

        // The room each device has left decides the slots of the store.
        let (ports, board) = (part.ports(), part.board());
        let cores: Vec<bool> = (0..layer_count).map(|i| part.has_core(i)).collect();
        let draft_board = block.as_ref().map(DraftBlock::board);
        self.meeting.meet(
            n,
            |build| {
                build.room[index] = room;
                build.ports[index] = ports;
                build.cores[index] = cores;
                build.boards[index] = Some(board);
                if draft_board.is_some() {
                    build.draft_board = draft_board;
                }
                Ok(())
            },
            |build| {
                let vram: Vec<u64> = (0..n)
                    .map(|d| {
                        let own = layout.draft_bytes(d)
                            + if self.placement.tensor.is_some() {
                                layout.expert_bytes
                            } else {
                                self.placement.devices[d].experts
                            };
                        build.room[d]
                            .min(own + (own >> 6))
                            .min(self.opts.slot_bytes.unwrap_or(u64::MAX))
                    })
                    .collect();
                let reader = reader(&self.gguf, self.draft.as_deref()).map_err(model_error)?;
                let head = self
                    .draft
                    .as_ref()
                    .map(|draft| DraftWeights::load(draft, c))
                    .transpose()
                    .map_err(|e| invalid(e.to_string()))?;
                let draft_layer = self.draft.as_ref().zip(head.as_ref()).map(|(gguf, weights)| {
                    DraftLayer { gguf, weights, device: self.head_device() }
                });
                let map = expert_map(&self.gguf, c, layout.per_layer, draft_layer.as_ref())
                    .map_err(model_error)?;
                // The tier is planned for all it can ever hold, what the limit
                // leaves and the experts without a slot in VRAM can use, and
                // starts with what it may hold now. Staging is sized for the
                // tier as it starts.
                let reach = ram.obtainable_experts().min(layout.ram_wanted(vram.iter().sum()));
                ram.want_experts(reach, 0);
                let ram_start = ram.experts_allowance().min(reach);
                let mut config = StoreConfig::new(
                    o.max_rows.min(MAX_BATCH),
                    c.experts.used as u32,
                    vram,
                    ram_start,
                );
                if let Some(reserve) = self.opts.reserve {
                    config.vram_reserve = reserve;
                }
                if o.max_rows > MAX_BATCH {
                    plan_transit(&mut config, layout, &reader, self.opts.transit);
                }
                config.ram_bytes = reach;
                config.ram_piece = self.opts.ram_piece.unwrap_or(PIECE) as u64;
                let plan = StorePlan::new(config, &map, &reader, &layout.specs, None)
                    .map_err(|e| invalid(e.to_string()))?;
                build.plan = Some(PlanBox { plan, map, reader, ram_start });
                Ok(())
            },
        )?;

        // The slots of this device; the last to arrive builds the store.
        let pools = {
            let build = self.meeting.lock();
            let planned = build.value.plan.as_ref().ok_or_else(|| invalid("no plan was made"))?;
            planned.plan.device_pools(index)
        };
        let mut pinned = Ledger::new("pinned RAM (booked by the engine)", u64::MAX / 2);
        let slots = device_slots(
            device,
            cx.ledger,
            &mut pinned,
            &pools,
            layout,
            serve,
            o.max_rows.min(MAX_BATCH),
        )
        .map_err(model_error)?;
        self.meeting.meet(
            n,
            |build| {
                build.slots[index] = Some(slots);
                Ok(())
            },
            |build| {
                let PlanBox { plan, map, reader, ram_start } =
                    build.plan.take().ok_or_else(|| invalid("no plan was made"))?;
                let slots: Vec<CudaSlots> =
                    build.slots.iter_mut().filter_map(Option::take).collect();
                for d in 0..n {
                    build.slot_bytes[d] = plan.vram_bytes(d);
                }
                let reach = plan.ram_bytes() as u64;
                let tier = HostRegion::elastic(
                    plan.ram_bytes(),
                    ram_start as usize,
                    ram.expert_pieces(device),
                )
                .map_err(|e| invalid(e.to_string()))?;
                // Cores for the threads that drive the step: the conductor and
                // one for each device.
                let cpu = (serve != Serve::Stream)
                    .then(|| cpu_pool(self.opts.cpu_workers, 1 + n).map(|pool| (pool, serve)))
                    .transpose()
                    .map_err(model_error)?;
                let store = build_store(plan, layout, map, reader, slots, tier, cpu)
                    .map_err(model_error)?;
                let ram_bytes = store.ram_bytes().0;
                let ram_fixed = store.ram_fixed_bytes();
                let mut firsts = Vec::with_capacity(n);
                for d in 0..n {
                    let part = self.placement.part(c, d);
                    firsts.push(part.first);
                    build.places[d] =
                        places(&store, part.first..part.first + part.shares.len(), layout, d);
                    build.n_slots[d] = (part.first..part.first + part.shares.len())
                        .map(|l| store.slots(layout.store_layer(d, l)).0)
                        .sum();
                }
                let boards = build
                    .boards
                    .iter()
                    .map(|b| b.clone().ok_or_else(|| invalid("a part has no route records")))
                    .collect::<Result<Vec<_>, _>>()?;
                if self.draft.is_some() {
                    let (device, pool) = store.place(0);
                    let slots = store.device(device);
                    build.draft_place = Some(ExpertAt {
                        table: slots.table(0),
                        pool: slots.pool(pool).0,
                        host: slots.host_share(0),
                    });
                }
                let mut feed =
                    Feed::new(store, boards, firsts, layout.clone(), c.experts.used as u32);
                feed.set_read_ahead(self.opts.read_ahead);
                if let Some(board) = build.draft_board.clone() {
                    feed.set_draft(board);
                    feed.fill_draft(c.experts.count).map_err(model_error)?;
                }
                let ngram = NgramFeed::open(&self.gguf, c, self.opts.row_cache, o.max_slots)
                    .map_err(model_error)?;
                let flow = Arc::new(Flow {
                    feed: Mutex::new(feed),
                    ngram: Mutex::new(ngram),
                    drafts: Mutex::new((0, Vec::new())),
                    enqueued: (0..n).map(|_| AtomicUsize::new(0)).collect(),
                    followed: (0..n).map(|_| AtomicUsize::new(0)).collect(),
                    begun: AtomicU64::new(0),
                    finished: AtomicUsize::new(0),
                    failed: AtomicBool::new(false),
                    uploads_seen: AtomicU64::new(0),
                    stall: self.opts.stall,
                    wide: Mutex::new(WideStats::default()),
                    totals: Mutex::new(ExpertStats {
                        ram_tier_bytes: ram_bytes,
                        ..Default::default()
                    }),
                });
                let _ = self.flow.set(flow.clone());
                self.usage.attach(&flow, layout, &self.placement);
                // From here on the tier follows what it is allowed: its
                // staging slots are the part it never gives up.
                ram.want_experts(reach, ram_fixed);
                let follower = Arc::downgrade(&flow);
                ram.on_experts(move |ram, allowance| {
                    if let Some(flow) = follower.upgrade() {
                        flow.follow_ram(ram, allowance);
                    }
                });
                build.flow = Some(flow);
                Ok(())
            },
        )?;

        // This part learns where its experts are, and its peer's ports.
        let (places, flow, peer, n_slots, slot_bytes) = {
            let build = self.meeting.lock();
            let build = &build.value;
            if let Some(block) = &mut block {
                let at =
                    build.draft_place.ok_or_else(|| invalid("the draft block has no slots"))?;
                block.set_experts(at)?;
            }
            let peer = self.placement.tensor.as_ref().map(|tensor| {
                let other = 1 - index;
                (build.ports[other], build.cores[other].clone(), tensor.mixer)
            });
            (
                build.places[index].clone(),
                build.flow.clone().ok_or_else(|| invalid("no flow was made"))?,
                peer,
                build.n_slots[index],
                build.slot_bytes[index],
            )
        };
        part.set_experts(places)?;
        if let Some((ports, cores, mixer)) = peer {
            let ports = ports.ok_or_else(|| invalid("a part that shares layers has no ports"))?;
            part.set_peer(ports, cores)?;
            if index == mixer {
                let owner: Vec<u8> = (0..c.n_layer)
                    .flat_map(|layer| {
                        let placement = &self.placement;
                        (0..c.experts.count as u32)
                            .map(move |expert| u8::from(placement.owner(layer, expert) != mixer))
                    })
                    .collect();
                part.set_owner(&owner)?;
            }
        }
        self.reports.lock().unwrap_or_else(|e| e.into_inner()).push(PartReport {
            index,
            device: cx.device.info().name.clone(),
            role: part.role(),
            layers: part.layer_range(),
            weights: part.weight_bytes(),
            slots: slot_bytes,
            n_slots,
            state_per_slot: part.state_bytes_per_slot(),
            kv_per_1000: part.page_bytes() * 1000 / u64::from(ENTRY_CELLS),
            keys_per_1000: part.key_page_bytes() * 1000 / u64::from(ENTRY_CELLS),
            scratch: part.scratch_bytes(),
            reserved: cx.ledger.reserved(),
            upload: part.upload_time(),
        });
        let first = part.layer_range().start == 0 && part.role() != super::Role::Worker;
        Ok(Box::new(Qwen4ExpDevice {
            index,
            devices: n,
            by_layers: self.placement.tensor.is_none(),
            n_vocab: c.n_vocab as u32,
            wide: c.wide() as u32,
            per_step: 8 * (c.n_layer as u32 + 1),
            embedding: first.then_some(embedding),
            draft: block,
            drafts: self.draft.is_some(),
            residual: None,
            chains: Vec::new(),
            only_drafts: false,
            drives_ngram: part.ngram_layer().is_some(),
            routing: Routing {
                mixer: self.placement.tensor.as_ref().map(|t| t.mixer).filter(|m| *m != index),
                owned: self.placement.tensor.as_ref().map(|_| {
                    (0..c.n_layer)
                        .map(|layer| {
                            (0..c.experts.count as u32)
                                .map(|e| self.placement.owner(layer, e) == index)
                                .collect()
                        })
                        .collect()
                }),
            },
            part,
            flow,
            segments: Vec::new(),
            tokens: Vec::new(),
            out_rows: Vec::new(),
            x: Vec::new(),
            accepted: Vec::new(),
            windows: Vec::new(),
            shape: Vec::new(),
        }))
    }
}

impl Model for Qwen4Exp {
    fn card(&self) -> ModelCard {
        ModelCard { id: self.name.clone(), context_length: self.context, created: self.created }
    }

    fn vocab(&self) -> u32 {
        self.config.n_vocab as u32
    }

    fn ends(&self, token: u32) -> bool {
        self.tokenizer.is_end_of_generation(token)
    }

    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        chat::encode(&self.tokenizer, &self.chat, prompt)
    }

    fn chat_format(&self) -> Option<ChatFormat> {
        Some(self.format.clone())
    }

    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        chat::piece(&self.tokenizer, token, text);
    }

    fn context(&self) -> ContextSpec {
        ContextSpec::NONE
    }

    fn split(&self) -> Split {
        match (&self.placement.tensor, self.placement.n_devices()) {
            (Some(_), _) => Split::Shared,
            (None, 1) => Split::Single,
            (None, _) => Split::Layer,
        }
    }

    fn devices(&self) -> usize {
        self.placement.n_devices()
    }

    fn links(&self) -> LinkSpec {
        // A split by layers hands the residual of all rows of a step over at
        // once, in a ring of two parcels; the devices that share every layer
        // write to each other themselves.
        match self.split() {
            Split::Layer => LinkSpec {
                sum_values: 0,
                lane_slots: 2,
                lane_values: self.opts.part.max_rows * self.config.wide() as u32,
            },
            _ => LinkSpec::default(),
        }
    }

    fn steps_wait_for_fetches(&self) -> bool {
        // An expert that no tier in VRAM holds is brought while the step
        // waits for it.
        true
    }

    fn pinned_ram(&self) -> u64 {
        self.pinned
    }

    fn expert_stats(&self) -> Option<ExpertStats> {
        self.take_store_stats()?;
        let totals = self.flow.get()?.totals.lock().unwrap_or_else(|e| e.into_inner());
        Some(*totals)
    }

    fn max_rows(&self) -> u32 {
        self.opts.part.max_rows
    }

    fn logit_rows(&self) -> u32 {
        self.opts.part.logit_rows
    }

    fn draft_rows(&self) -> u32 {
        self.opts.part.window_rows
    }

    fn own_drafts(&self) -> u32 {
        match self.draft {
            Some(_) => self.opts.part.window_rows.saturating_sub(1),
            None => 0,
        }
    }

    fn attention_span(&self) -> u32 {
        self.config.indexer.dense_limit() as u32
    }

    fn page_entry(&self, page: u32) -> u32 {
        page | PAGE_ENTRY
    }

    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        // The attention kernels store 128 cells per entry; `open` refuses any
        // other page size.
        (0..self.placement.n_devices())
            .fold(PageLayout::new(self.placement.n_devices(), cells_per_page), |layout, device| {
                layout.per_page(device, "K, V and keys", self.page_bytes[device])
            })
    }

    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        self.open_part(cx).inspect_err(|error| self.meeting.fail(error.to_string()))
    }
}

// --- one device's part ------------------------------------------------------

/// A part behind the runtime's interface.
struct Qwen4ExpDevice {
    index: usize,
    devices: usize,
    /// Whether the devices hold a range of layers each (and hand the residual
    /// over) rather than share every layer.
    by_layers: bool,
    n_vocab: u32,
    /// Values of one row of the residual streams.
    wide: u32,
    /// Exchanges between a mixer and its worker per step.
    per_step: u32,
    /// The embeddings, for the part that starts the residual streams.
    embedding: Option<Embedding>,
    /// Whether this part keeps the sequences the n-gram table is read for.
    drives_ngram: bool,
    routing: Routing,
    part: Part,
    flow: Arc<Flow>,
    // The step being enqueued, kept to reuse the memory.
    segments: Vec<Segment>,
    tokens: Vec<u32>,
    out_rows: Vec<u32>,
    x: Vec<f32>,
    accepted: Vec<u32>,
    /// The draft windows of the step before, as (slot, first row), which the
    /// record of this step says the outcome of.
    windows: Vec<(u32, u32)>,
    /// The step that ended last: its segments as (slot, direct, rows, first
    /// row), and its wanted rows.
    shape: Vec<(u32, bool, u32, u32)>,
    /// The model's draft block, on the part that holds the output head.
    draft: Option<DraftBlock>,
    /// Whether the model drafts (on this part or another).
    drafts: bool,
    /// Where the part left the residual of the rows of its last step.
    residual: Option<DevicePtr>,
    chains: Vec<Chain>,
    /// The step being run only drafts.
    only_drafts: bool,
}

impl std::fmt::Debug for Qwen4ExpDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4ExpDevice").field("index", &self.index).finish_non_exhaustive()
    }
}

/// The experts of layer `layer` of `part` (the `index`-th) in a step of more than eight
/// rows: the first round is enqueued with the experts that have a slot, then the store is
/// served until the misses are brought and computed (`Misses`).
fn wide_layer(
    part: &mut Part,
    flow: &Flow,
    index: usize,
    env: &StepEnv<'_>,
    layer: usize,
    routing: &Routing,
) -> Result<(), ModelError> {
    let owned = |expert: u32| routing.owned.as_ref().is_none_or(|o| o[layer][expert as usize]);
    let (mut started, mut seen) = (Instant::now(), flow.progress());
    let source = routing.mixer.map(|mixer| flow.lock_feed().board(mixer));
    let mut misses = {
        let feed = flow.lock_feed();
        Misses::begin(part, env, &feed, index, layer, source, &owned).map_err(model_error)?
    };
    loop {
        let routed =
            routing.mixer.is_none_or(|mixer| flow.enqueued[mixer].load(Ordering::Acquire) > layer);
        let progress = {
            let mut feed = flow.lock_feed();
            misses.advance(part, env, &mut feed, routed, &owned).map_err(model_error)?
        };
        if progress == Progress::Done {
            return Ok(());
        }
        let progress = flow.progress();
        if progress != seen {
            (started, seen) = (Instant::now(), progress);
        }
        flow.check(started, "the experts of a wide step were brought")?;
        flow.pump()?;
        std::hint::spin_loop();
    }
}

/// Who routes for a part and which experts it computes, when both devices work on every
/// layer.
struct Routing {
    /// The device that routes, for a part that does not.
    mixer: Option<usize>,
    /// Per layer and expert, whether this part computes the expert; `None` for all.
    owned: Option<Vec<Vec<bool>>>,
}

impl Qwen4ExpDevice {
    /// The stream is idle: nothing of a failed step waits for the host any
    /// more.
    fn settle(&self, stream: &Stream) {
        self.flow.drain(|| stream.is_idle().unwrap_or(true));
    }

    fn enqueue_step(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        let record = &step.record;
        let stream = step.stream;
        let rows = record.rows();
        if rows == 0 {
            return Err(invalid("a step without rows"));
        }
        self.segments.clear();
        self.chains.clear();
        let mut bar = 0;
        for i in 0..record.segments() {
            if record.replay(i) {
                return Err(invalid("the model has no prefix cache to replay from"));
            }
            let segment = record.segment(i);
            if record.own_drafts(i) {
                if !self.drafts {
                    return Err(invalid("the model drafts nothing itself"));
                }
                if segment.n_rows >= 2 {
                    // All rows of a window are wanted, so its sampler rows are
                    // consecutive; `wanted` counts the wanted rows before it.
                    let wanted: u32 = (0..record.runs())
                        .map(|r| record.run(r))
                        .map(|(first, count)| segment.row0.clamp(first, first + count) - first)
                        .sum();
                    // The bars of the window's links, when the record has them.
                    let links = segment.n_rows - 1;
                    let needs = if record.draft_bars() >= bar + links {
                        (bar..bar + links).map(|k| record.draft_need(k)).collect()
                    } else {
                        Vec::new()
                    };
                    bar += links;
                    self.chains.push(Chain {
                        segment: i as usize,
                        rows: (0..links).map(|j| record.sampler(wanted + j).for_draft()).collect(),
                        needs,
                    });
                }
            }
            self.segments.push(segment);
        }
        // The rows that take the block's drafts have no token yet.
        let mut drafted = vec![false; rows as usize];
        for chain in &self.chains {
            let s = &self.segments[chain.segment];
            drafted[(s.row0 + 1) as usize..(s.row0 + s.n_rows) as usize].fill(true);
        }
        self.tokens.clear();
        for row in 0..rows {
            let token = if drafted[row as usize] { 0 } else { record.row(row).token };
            if token >= self.n_vocab {
                return Err(invalid(format!("token {token} of a vocabulary of {}", self.n_vocab)));
            }
            self.tokens.push(token);
        }
        self.out_rows.clear();
        for run in 0..record.runs() {
            let (first, count) = record.run(run);
            self.out_rows.extend(first..first + count);
        }
        // The step begins for the store: the draft block routes in it before
        // any layer of the model. The number of the step marks its route
        // records and its exchanges: a kernel waits for a flag that is behind
        // the number, and the flags start at 0, so no step may be numbered 0
        // (the conductor's first warm-up record is).
        let number = u64::from(record.step()) + 1;
        // The windows whose rows past the first verify drafts; a step that
        // only drafts runs none of the model's rows and is no step of the
        // store.
        let windows: Vec<(u32, u32)> =
            self.segments.iter().filter(|s| !s.direct).map(|s| (s.row0, s.n_rows)).collect();
        let worth: Option<Vec<f32>> = record
            .row_worth(0)
            .map(|_| (0..rows).map(|row| record.row_worth(row).unwrap_or(0.0)).collect());
        self.only_drafts = record.drafts_only();
        let windows = (!self.only_drafts).then_some((windows.as_slice(), worth.as_deref()));
        self.flow.begin(number, rows, windows, self.index, stream)?;

        // The draft windows of the step before: how many rows count.
        if self.part.has_deferred() {
            self.accepted.clear();
            for &(slot, row0) in &self.windows {
                let outcome = (0..record.commits())
                    .map(|i| record.commit(i))
                    .find(|c| c.slot == slot && c.row0 == row0)
                    .ok_or_else(|| invalid(format!("no outcome for the window of slot {slot}")))?;
                self.accepted.push(outcome.accepted);
            }
            self.part.enqueue_commit(stream, &self.accepted)?;
            if self.drives_ngram {
                self.flow.lock_ngram().settle(&self.accepted).map_err(model_error)?;
            }
        }

        // The draft block follows the rows of the last step that count, in
        // the tables of that step.
        if let (Some(block), Some(residual)) = (&mut self.draft, self.residual.take()) {
            let outcome = |slot: u32, row0: u32| {
                (0..record.commits())
                    .map(|i| record.commit(i))
                    .find(|c| c.slot == slot && c.row0 == row0)
                    .map(|c| c.accepted)
            };
            block.follow(stream, &self.flow, &outcome, residual)?;
        }

        // The pages the pool freed give back what the K/V groups keep for them.
        let freed: Vec<u32> = (0..record.freed()).map(|i| record.freed_page(i)).collect();
        self.part.release_pages(freed.iter().copied());
        if let Some(block) = &mut self.draft {
            block.release_pages(&freed);
        }

        // The page tables as the conductor made them.
        let mut taken = 0;
        for i in 0..record.updates() {
            let update = record.update(i);
            if self.part.has_attention() {
                let entries = (taken..taken + update.count).map(|k| record.entry(k));
                self.part.update_table(stream, update.slot, update.first, entries)?;
            }
            if let Some(block) = &mut self.draft {
                let entries = (taken..taken + update.count).map(|k| record.entry(k));
                block.update_table(stream, update.slot, update.first, entries)?;
            }
            taken += update.count;
        }

        // A step that only drafts: the chains, and nothing of the model. It
        // leaves no window for the next record to commit.
        if record.drafts_only() {
            if let Some(block) = &mut self.draft {
                let chances = record.draft_chances();
                block.draft(
                    stream,
                    &self.flow,
                    &self.segments,
                    &mut self.tokens,
                    &self.chains,
                    chances.as_ref(),
                    true,
                )?;
                // The round after this step verifies these drafts, or the
                // first of them: the rows of the n-gram table that its rows
                // need are read now, while the conductor decides.
                let next: Vec<(u32, &[u32])> = self
                    .chains
                    .iter()
                    .map(|chain| {
                        let s = &self.segments[chain.segment];
                        let made = block.made(s.slot) as usize;
                        (s.slot, &self.tokens[s.row0 as usize..s.row0 as usize + 1 + made])
                    })
                    .collect();
                // A failed read ahead costs nothing but the read of the step.
                let _ = self.flow.lock_ngram().expect(&next);
            }
            self.windows.clear();
            return Ok(None);
        }

        // The newest cell of every sequence in the draft block and the chains
        // of the windows that take its drafts; the other device threads wait
        // for the drafts, which are rows of this step.
        if let Some(block) = &mut self.draft {
            let chances = record.draft_chances();
            block.draft(
                stream,
                &self.flow,
                &self.segments,
                &mut self.tokens,
                &self.chains,
                chances.as_ref(),
                false,
            )?;
            let drafts: Vec<(u32, u32)> = drafted
                .iter()
                .enumerate()
                .filter(|(_, drafted)| **drafted)
                .map(|(row, _)| (row as u32, self.tokens[row]))
                .collect();
            self.flow.publish(number, drafts);
            block.keep(&self.segments, &self.tokens);
        } else if !self.chains.is_empty() {
            for (row, token) in self.flow.await_drafts(number)? {
                self.tokens[row as usize] = token;
            }
        }

        // The rows of the n-gram table, and where the sequences are now.
        if self.drives_ngram {
            let mut ngram = self.flow.lock_ngram();
            let mut parts = Vec::with_capacity(self.segments.len());
            for s in &self.segments {
                ngram.begin_sequence(s.slot, s.pos0 as u32).map_err(model_error)?;
                let rows = s.row0 as usize..(s.row0 + s.n_rows) as usize;
                parts.push((s.slot, &self.tokens[rows]));
            }
            ngram.ask(&parts).map_err(model_error)?;
            for (slot, tokens) in parts {
                let direct = self.segments.iter().any(|s| s.slot == slot && s.direct);
                if direct {
                    ngram.advance(slot, tokens);
                } else {
                    ngram.hold(slot, tokens);
                }
            }
        }
        self.windows.clear();
        self.windows.extend(self.segments.iter().filter(|s| !s.direct).map(|s| (s.slot, s.row0)));
        self.shape.clear();
        self.shape.extend(self.segments.iter().map(|s| (s.slot, s.direct, s.n_rows, s.row0)));

        // The residual streams start from the embeddings.
        let n = self.part.n_embd() as usize;
        self.x.clear();
        if let Some(embedding) = &self.embedding {
            self.x.resize(rows as usize * n, 0.0);
            for (token, out) in self.tokens.iter().zip(self.x.chunks_exact_mut(n)) {
                embedding.decode(*token, out)?;
            }
        }
        let host = StepRecord::new(&self.segments, &self.tokens, None, &self.out_rows)?;
        let (segments, x) = (&self.segments, &self.x);

        // The step: every layer, with the host serving the store behind it.
        let (token, xbase) =
            (number as u32, (number as u32).wrapping_mul(self.per_step).wrapping_add(1));
        let inbound = if self.by_layers && self.index > 0 {
            Some(step.links.receive(self.index - 1, stream)?)
        } else {
            None
        };
        macro_rules! env {
            ($fetched:expr, $inbound:expr) => {
                StepEnv {
                    stream,
                    record: &host,
                    segments,
                    token,
                    x,
                    ngram: $fetched,
                    inbound: $inbound,
                    xbase,
                    pools: step.pages,
                }
            };
        }
        self.part.begin(&env!(None, inbound))?;
        if inbound.is_some() {
            step.links.release(self.index - 1, stream)?;
        }
        let mut fetched: Option<Vec<u8>> = None;
        let ngram_at = self.part.ngram_layer();
        for i in 0..self.part.n_layers() {
            if ngram_at == Some(i) && fetched.is_none() {
                fetched = Some(self.flow.await_rows()?);
            }
            self.flow.throttle(self.index)?;
            self.part.enqueue_layer(&env!(fetched.as_deref(), None), i)?;
            // The other part of a shared layer may read this one's record from now on.
            self.flow.enqueued[self.index].store(i + 1, Ordering::Release);
            if rows > MAX_BATCH {
                let env = env!(fetched.as_deref(), None);
                wide_layer(&mut self.part, &self.flow, self.index, &env, i, &self.routing)?;
            }
            self.flow.pump()?;
        }
        let n_out = self.part.end(&env!(fetched.as_deref(), None))?;
        if self.draft.is_some() {
            self.residual = Some(self.part.residual());
        }
        if self.by_layers && self.index + 1 < self.devices {
            step.links.send(self.index, stream, self.part.residual(), rows * self.wide)?;
        }
        Ok((n_out > 0 && self.part.has_head()).then(|| Logits {
            ptr: self.part.head_logits(),
            stride: self.n_vocab,
            first_token: 0,
        }))
    }

    fn wait_step(&mut self, end: &Event) -> Result<(), ModelError> {
        let flow = &self.flow;
        let started = Instant::now();
        while !end.is_done()? {
            flow.check(started, "a device ran the step")?;
            flow.pump()?;
            std::hint::spin_loop();
        }
        // Every route record of this part is with the store.
        let started = Instant::now();
        while flow.followed[self.index].load(Ordering::Acquire)
            < flow.enqueued[self.index].load(Ordering::Acquire)
        {
            flow.check(started, "route records were on their way")?;
            flow.pump()?;
            std::hint::spin_loop();
        }
        flow.finish(!self.only_drafts)
    }
}

impl DeviceModel for Qwen4ExpDevice {
    fn logits_per_row(&self) -> u32 {
        if self.part.has_head() { self.n_vocab } else { 0 }
    }

    fn enqueue(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        let result = self.enqueue_step(step);
        if result.is_err() {
            self.flow.fail();
            self.settle(step.stream);
        }
        result
    }

    fn wait(&mut self, end: &Event) -> Result<(), ModelError> {
        let result = self.wait_step(end);
        if result.is_err() {
            self.flow.fail();
            self.flow.drain(|| end.is_done().unwrap_or(true));
        }
        result
    }

    fn picked(&mut self, picks: &[Pick]) {
        // Every sequence most likely goes on with the token it just picked:
        // the rows of the table for that step are asked for now, before the
        // conductor has decided anything.
        let one_each = picks.len() == self.shape.len()
            && self
                .shape
                .iter()
                .zip(&self.out_rows)
                .all(|(&(_, direct, rows, row0), &out)| direct && out == row0 + rows - 1);
        if !one_each {
            return;
        }
        let next: Vec<(u32, [u32; 1])> =
            self.shape.iter().zip(picks).map(|(&(slot, ..), pick)| (slot, [pick.token])).collect();
        let next: Vec<(u32, &[u32])> =
            next.iter().map(|(slot, token)| (*slot, token.as_slice())).collect();
        // A failed read ahead costs nothing but the read of the step.
        let _ = self.flow.lock_ngram().expect(&next);
    }

    fn own_drafts(&self) -> Option<OwnDrafts> {
        let (ptr, tokens) = self.draft.as_ref()?.drafted()?;
        Some(OwnDrafts { ptr, tokens })
    }

    fn own_draft_ms(&mut self) -> f64 {
        self.draft.as_ref().map_or(0.0, DraftBlock::chain_ms)
    }

    fn own_draft_probs(&mut self, out: &mut Vec<f32>) {
        if let Some(block) = self.draft.as_ref().filter(|block| block.drafted().is_some()) {
            out.extend_from_slice(block.probs());
        }
    }

    fn rows_not_exact(&mut self, out: &mut Vec<u32>) {
        // The store left rows that verify drafts without an expert only the
        // file held; every part reports what the store knows when it has
        // ended, and the last to end knows all of it. A step that only
        // drafted has no rows.
        if self.only_drafts {
            return;
        }
        let left = self.flow.lock_feed().store.rows_left();
        out.extend((0..32).filter(|row| left >> row & 1 != 0));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn parts_that_meet_see_what_the_last_to_arrive_made_of_what_they_added() {
        let meeting = Meeting::new(Vec::<usize>::new());
        let made = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for part in 0..3 {
                let (meeting, made) = (&meeting, &made);
                scope.spawn(move || {
                    meeting
                        .meet(
                            3,
                            |added| {
                                added.push(part);
                                Ok(())
                            },
                            |added| {
                                // Once, and with every part's contribution.
                                made.fetch_add(1, Ordering::SeqCst);
                                added.sort_unstable();
                                assert_eq!(added, &[0, 1, 2]);
                                Ok(())
                            },
                        )
                        .unwrap();
                    // Returned: the last has run.
                    assert_eq!(made.load(Ordering::SeqCst), 1);
                });
            }
        });
        // A second meeting of the same parts works as well.
        std::thread::scope(|scope| {
            for _ in 0..3 {
                scope.spawn(|| meeting.meet(3, |_| Ok(()), |_| Ok(())).unwrap());
            }
        });
    }

    #[test]
    fn a_part_that_fails_lets_the_others_give_up_instead_of_waiting() {
        let meeting = Meeting::new(());
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| meeting.meet(2, |_| Ok(()), |_| Ok(())));
            // The other part fails before it arrives.
            meeting.fail("no memory for the weights".into());
            let error = waiting.join().unwrap().unwrap_err().to_string();
            assert!(error.contains("no memory for the weights"), "{error}");
        });
        // And a part that arrives later does not wait either.
        assert!(meeting.meet(2, |_| Ok(()), |_| Ok(())).is_err());
    }

    #[test]
    fn a_failure_in_what_the_last_makes_reaches_everyone() {
        let meeting = Meeting::new(());
        let errors: Vec<bool> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        meeting
                            .meet(2, |_| Ok(()), |_| Err(invalid("the store does not fit")))
                            .is_err()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(errors, [true, true]);
    }
}
