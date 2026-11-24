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

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, TryLockError};
use std::time::{Duration, Instant};

use ares_api::{ChatFormat, EngineError, ExpertStats, ModelCard, Prompt};
use ares_core::Ledger;
use ares_core::units::Bytes;
use ares_cuda::{Device, DevicePtr, ENTRY_CELLS, Event, MAX_BATCH, Segment, StepRecord, Stream};
use ares_gguf::Gguf;
use ares_io::Reader;
use ares_runtime::expert_store::{
    CudaSlots, ExpertTypes, FillStats, HostRegion, PIECE, Serve, StoreConfig, StoreError,
    StorePlan, StoreStats,
};
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, DeviceTransfer, Drafter, DrafterKind, Durable,
    FarPages, LinkSpec, Logits, Model, ModelDrafts, ModelError, Pick, PinnedRam, PrefixCache,
    Split, StepContext, TowerJob, TransferContext, Vision,
};
use ares_state::PageLayout;
use ares_tokenizer::{ChatOptions, Tokenizer};

use super::board::Board;
use super::feed::{
    Feed, Layout, MtpExperts, build_store, cpu_pool, device_slots, expert_map, owners, places,
    plan_transit, reader,
};
use super::misses::{Misses, Progress, SpanPicks, WideStats};
use super::mtp::{MtpBlock, MtpDrafted, MtpReport, MtpTimes, Watch};
use super::ngram::{self, NgramFeed};
use super::part::{
    ExpertAt, Frame, MTP_KV, Options, PAGE_ENTRY, Part, PartSetup, PartShape, PassSizes, Pieces,
    Ports, StepEnv,
};
use super::placement::{MARGIN, PartPlan, Placement};
use super::prune::{self, PruneMask};
use super::resume::Resume;
use super::session::{Chosen, Fill, PartReport, SessionOptions, choose};
use super::usage::{Log, ProfileReport, Usage, UsageStats};
use crate::blocks::far::FarPool;
use crate::blocks::head::HeadView;
use crate::blocks::mtp::{MtpChain, MtpStep};
use crate::blocks::rows::{self, HostTable, TOKEN_EMBD};
use crate::blocks::slots::{self, Declared, TAILS_PER_SLOT};
use crate::chat;
use crate::qwen3vl::{Projector, TowerAt};
use crate::qwen4exp::reference::mtp_config;
use crate::qwen4exp::weights::{MtpWeights, Weights};
use crate::qwen4exp::{Config, Error as FileError};

mod pass;

use pass::{PassKey, PassLoan, PassRun};

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

/// What the expert store measured of serving a miss on this machine
/// ([`Qwen4Exp::expert_costs`]).
#[derive(Clone, Debug, Default)]
pub struct ExpertCosts {
    /// Per pair of types of the experts (gate and up, down): the seconds a
    /// CPU job takes per (expert, row) pair with one row and with eight.
    pub cpu: Vec<(ExpertTypes, Option<f64>, Option<f64>)>,
    /// Per device: the seconds of one upload, once one was timed.
    pub upload: Vec<Option<f64>>,
}

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
    /// Whether a part holds the MTP block, and where its experts are.
    mtp: bool,
    mtp_place: Option<ExpertAt>,
    /// What the tiers hold, for the routers' pruning.
    prune_mask: Option<Arc<PruneMask>>,
    /// Per part: the bytes of VRAM a step wider than its own scratch runs on,
    /// where the expert slots of its device lend them, and where its input of
    /// a half block lies in them; the multiprocessors of its device; the rows
    /// of the widest step, as far as the slots can lend for it.
    lent: Vec<u64>,
    lent_at: Vec<Option<Pieces>>,
    lent_raw: Vec<u64>,
    /// Per part: the bytes each piece of its loan holds, and what each pool of its device can
    /// lend (what the scratch was laid out in).
    lent_pieces: Vec<Vec<u64>>,
    lendable: Vec<Vec<u64>>,
    multiprocessors: Vec<u32>,
    lone_rows: u32,
    /// The most rows of a prompt read layer by layer (0: none).
    pass_rows: u32,
    /// Per part: the K and V of its attention layers beyond VRAM, if any.
    far: Vec<Option<Arc<Mutex<FarPool>>>>,
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

/// How long the start of a step tries to make room in the RAM limit for the
/// K and V beyond VRAM that it names, and how long each try waits for what
/// is on its way into the tier of experts before the tier gives pieces up.
const FAR_WAIT: Duration = Duration::from_secs(10);
const FAR_SHRINK_WAIT: Duration = Duration::from_millis(200);

// --- what the device threads share in a step --------------------------------

/// The expert store, the rows of the n-gram table, and the counters that tell
/// the threads how far each part's step has got.
pub(super) struct Flow {
    feed: Mutex<Feed>,
    ngram: Mutex<NgramFeed>,
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
    /// The most rows of a step on the parts' own scratch; a wider step runs on
    /// VRAM the expert slots lend: per part its bytes and where they lie.
    own_rows: u32,
    lent: Vec<Vec<u64>>,
    lent_at: Vec<Option<Pieces>>,
    /// Whether the running step runs on the loan.
    on_loan: AtomicBool,
    /// Prompts read layer by layer (`runtime/pass.rs`): the most rows of one (0: none), what
    /// each part keeps of a row, the picks of a row, the loan of the pass that runs or ran
    /// last, and whether the step being run is a pass.
    pass_rows: u32,
    pass_sizes: Vec<PassSizes>,
    top_k: u32,
    pass: Mutex<Option<PassLoan>>,
    pass_completed: AtomicBool,
    /// In a step of a pass: a part paused it; and by layers, the residuals of micro-batches
    /// the device before has handed over and the next has released.
    pass_paused: AtomicBool,
    pass_sent: AtomicUsize,
    pass_released: AtomicUsize,
    /// Both devices on every layer: the pieces of the pass the mixer has let run, and the
    /// picks its router made, which the worker takes.
    pass_go: AtomicU64,
    pass_picks: Mutex<Option<(u64, Arc<SpanPicks>)>>,
    /// The passes that took a loan so far: each is numbered by it.
    pass_count: AtomicU64,
    /// The far pools of the parts (`blocks/far.rs`), the far slots their
    /// pieces cover now, and the engine's RAM, which books them.
    far: Vec<Arc<Mutex<FarPool>>>,
    far_live: Mutex<Vec<bool>>,
    ram: PinnedRam,
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

    pub(super) fn lock_ngram(&self) -> MutexGuard<'_, NgramFeed> {
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
    /// is the MTP block's, whose experts are resident, and the work between
    /// two steps of the store (the heat's decay, the slots that move, the RAM
    /// tier that grows) is counted in the model's steps.
    #[allow(clippy::too_many_arguments)]
    fn begin(
        &self,
        step: u64,
        rows: u32,
        windows: Option<StepWindows<'_>>,
        far_live: &[bool],
        index: usize,
        stream: &Stream,
        pass: Option<&PassKey>,
    ) -> Result<(), ModelError> {
        let mut feed = self.lock_feed();
        if self.begun.load(Ordering::Acquire) != step {
            // The K and V beyond VRAM that the step names are locked before
            // the store begins it: the tier of experts gives up what they take
            // while no step runs.
            self.lock_far(&mut feed, far_live)?;
            for counter in self.enqueued.iter().chain(&self.followed) {
                counter.store(0, Ordering::Release);
            }
            self.finished.store(0, Ordering::Release);
            // A prompt read layer by layer keeps its rows on a loan of its own; the loan of
            // one before goes back first, or serves it.
            if windows.is_some() {
                let far_reach = far_live.iter().rposition(|l| *l).map_or(0, |s| s as u32 + 1);
                self.lend_pass(&mut feed, pass, far_reach)?;
            }
            self.pass_completed.store(false, Ordering::Release);
            self.pass_paused.store(false, Ordering::Release);
            let wide = rows > self.own_rows && windows.is_some() && pass.is_none();
            if wide {
                self.lend(&mut feed)?;
            }
            self.on_loan.store(wide, Ordering::Release);
            if let Some((windows, worth)) = windows {
                feed.begin_step(rows).map_err(model_error)?;
                feed.store.set_windows(windows, worth);
            }
            self.begun.store(step, Ordering::Release);
        }
        feed.store.device_mut(index).fence(stream).map_err(|e| invalid(e.to_string()))
    }

    /// Locks the pieces of every far pool that hold far pages in use (`live`,
    /// a flag per far slot) and gives back the others, between steps, with
    /// the store not in a step. The tier of experts gives up what the RAM
    /// limit has no room for: a running sequence's K and V rank above it
    /// (see "Host memory" in `docs/ARCHITECTURE.md`).
    fn lock_far(&self, feed: &mut Feed, live: &[bool]) -> Result<(), ModelError> {
        if self.far.is_empty() {
            return Ok(());
        }
        let mut last = self.far_live.lock().unwrap_or_else(|e| e.into_inner());
        if last.as_slice() == live {
            return Ok(());
        }
        let mut pools: Vec<_> =
            self.far.iter().map(|p| p.lock().unwrap_or_else(|e| e.into_inner())).collect();
        let wanted: Vec<Vec<bool>> = pools.iter().map(|p| p.wanted(live)).collect();
        // What holds no page in use goes back first; what the pages need is
        // locked in every pool.
        for (pool, wanted) in pools.iter_mut().zip(&wanted) {
            pool.release_unwanted(wanted);
        }
        let bytes: u64 = pools.iter().zip(&wanted).map(|(p, w)| p.bytes_of(w)).sum();
        let locked: u64 = pools.iter().map(|p| p.locked()).sum();
        self.ram.set_context_target(bytes.max(locked));
        let started = Instant::now();
        loop {
            let short = pools
                .iter_mut()
                .zip(&wanted)
                .find_map(|(pool, wanted)| pool.lock_wanted(wanted).err());
            match short {
                None => break,
                Some(Err(error)) => return Err(error),
                Some(Ok(short)) => {
                    if started.elapsed() > FAR_WAIT {
                        self.ram.set_context_target(pools.iter().map(|p| p.locked()).sum());
                        return Err(invalid(format!(
                            "the RAM limit has no room for {} more of K and V beyond VRAM: the \
                             tier of experts is at its floor and the prompt cache holds the rest",
                            Bytes(short.0)
                        )));
                    }
                    // The tier gives up what its allowance no longer has room for.
                    feed.store.set_ram_target(self.ram.experts_allowance());
                    feed.store.shrink_ram(FAR_SHRINK_WAIT).map_err(|e| invalid(e.to_string()))?;
                    drop(feed.store.take_released());
                    if feed.store.ram_at_floor() {
                        self.ram.experts_floor(feed.store.ram_bytes().0);
                    }
                    self.ram.sweep_checkpoints();
                }
            }
        }
        self.ram.set_context_target(pools.iter().map(|p| p.locked()).sum());
        last.clear();
        last.extend_from_slice(live);
        Ok(())
    }

    /// Lends every device the VRAM of a step wider than the parts' own
    /// scratch, if it has not yet, and waits until the memory is free: the
    /// slots' experts moved out and copied down. Between steps.
    fn lend(&self, feed: &mut Feed) -> Result<(), ModelError> {
        let started = Instant::now();
        for (device, pieces) in self.lent.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
            while feed
                .store
                .lend_pieces(device, pieces)
                .map_err(|e| invalid(e.to_string()))?
                .is_none()
            {
                self.check(started, "the expert slots lent VRAM")?;
                feed.store.poll().map_err(|e| invalid(e.to_string()))?;
                std::hint::spin_loop();
            }
            while !feed.store.scratch_ready(device).map_err(|e| invalid(e.to_string()))? {
                self.check(started, "the expert slots lent VRAM")?;
                feed.store.poll().map_err(|e| invalid(e.to_string()))?;
                std::hint::spin_loop();
            }
        }
        Ok(())
    }

    /// Where the VRAM lent to part `index` lies while the expert slots lend it.
    fn lent(&self, index: usize) -> Option<Pieces> {
        let lent = self.lent_at.get(index).cloned().flatten()?;
        self.lock_feed().store.has_scratch(index).then_some(lent)
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

    /// A part has ended its step on the device: the last of them ends it for
    /// the store, unless it only drafted (`store` false).
    fn finish(&self, store: bool) -> Result<(), ModelError> {
        if self.finished.fetch_add(1, Ordering::AcqRel) + 1 == self.parts() {
            let mut feed = self.lock_feed();
            if store {
                feed.end_step().map_err(model_error)?;
                // What the tier gave up at the end of this step is unlocked
                // here, where every device has ended the step and none has
                // begun the next.
                drop(feed.store.take_released());
            }
            // A pass that ended in this step has read its rows; its loan stays until the
            // next step begins.
            if self.pass_completed.load(Ordering::Acquire) {
                self.pass_done();
            }
            // VRAM lent for wide steps goes back after the first step that did
            // not run on it: that step's start still read what the last one
            // left there (the MTP block follows its rows).
            if !self.on_loan.load(Ordering::Acquire) && !self.has_pass_loan() {
                for device in 0..self.lent.len() {
                    feed.store.return_scratch(device).map_err(|e| invalid(e.to_string()))?;
                }
            }
        }
        Ok(())
    }
}

/// The windows of a step that verify drafts, as (first row, rows), and what
/// every row of the step is worth in milliseconds (the record's row worth).
type StepWindows<'a> = (&'a [(u32, u32)], Option<&'a [f32]>);

// --- the model --------------------------------------------------------------

/// The rows of a step to open the model with on `devices`: as many as the scratch of a step
/// may take of the smaller device's memory (a thirty-second; the rest is slots for experts and
/// the pools of K and V), as a power of two between eight (the decode kernels' limit) and
/// 4096.
///
/// A step of more rows reads the experts of a layer once for more of them, and every byte of
/// the scratch is a byte less for expert slots, which the generating steps miss: measured on
/// Flash-Next, 2048 rows (1.4 GiB of scratch on a 12 GiB card) cost a third to a half of the
/// generation speed through the slots they took. The scratch grows with the rows, about 0.7 MiB
/// a row for Flash-Next; `opts` supplies the other choices of the part, whose `max_rows` is
/// ignored.
pub fn step_rows(gguf: &Gguf, devices: &[Device], opts: &Options) -> Result<u32, OpenError> {
    let config = Config::from_gguf(gguf)?;
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

/// Whether a model opened with `opts` on `devices` devices, with a DFlash drafter or
/// without (`dflash`), reads a prompt wider than the parts' own scratch layer by layer
/// ([`SessionOptions::pass_rows`]): its micro-batches take the many-row kernels, K and V
/// have no exact pool (whose tiles convert per step for every layer at once), no DFlash
/// drafter reads the residual of a step's rows at some layers, and the placement is one the
/// pass is built for.
pub fn reads_layer_by_layer(opts: &Options, dflash: bool) -> bool {
    opts.own_rows() >= 2 * MAX_BATCH
        && crate::blocks::kv::Group::window_limit(opts.kv.0, opts.kv.1).is_none()
        && !dflash
}

/// The most rows of a step the model is opened with.
const WIDE_ROWS: u32 = 4096;

/// The rows of a lent step are a multiple of this ([`widest_rows`]).
pub const ROW_GRAIN: u32 = 256;

/// The rows of the widest step from `asked` down to `own` that `fits` holds: `asked`, or the
/// largest multiple of [`ROW_GRAIN`] (of `own` where that is less) below it that fits, or
/// `own` when no more rows do. `fits` must hold for fewer rows wherever it holds for more (the
/// scratch grows with the rows), so the rows are found by halving the range between them.
pub fn widest_rows<E>(
    own: u32,
    asked: u32,
    mut fits: impl FnMut(u32) -> Result<bool, E>,
) -> Result<u32, E> {
    if asked <= own {
        return Ok(own);
    }
    if fits(asked)? {
        return Ok(asked);
    }
    let grain = ROW_GRAIN.min(own).max(1);
    // `lo` grains fit (or are no more than `own`), `hi` grains do not.
    let (mut lo, mut hi) = (own / grain, asked.div_ceil(grain));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if mid * grain <= own || fits(mid * grain)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo * grain).max(own))
}

/// The rows of the widest step a prompt read alone takes on `devices` placed as
/// `placement`: `asked`, or the largest multiple of 256 rows (`ROW_GRAIN`) below it whose
/// scratch the expert slots of every device can lend (`expert_store/scratch.rs`),
/// above `opts.part.own_rows()`; that if none is. `pages` is what the pools of K and
/// V hold, `page_bytes` a page on each device, as [`place`] gave them.
///
/// A device lends what the slots of its pools (each the layers whose experts take
/// slots of one size) hold above a row's picks of each of those layers (the store's
/// floor for a loan), a piece of each pool, in which the scratch is laid out without a
/// buffer cut by the end of a piece; a step of more rows also
/// keeps the rest of what it needs (its inputs, its exchange, its predictions,
/// about 50 KiB a row on Flash-Next against 0.5 to 0.7 MiB of scratch) for
/// good, which the slots lose, as do the parcels a split by layers hands the
/// residual over in (80 KiB a row on the receiving device); what is estimated
/// here leaves a tenth of the lendable room spare for what the plan of the
/// store takes besides (transit). `drafter` is what the drafters take of the bytes for
/// slots on the device with the head (the MTP block's experts stay resident there), which
/// the slots of that device do not have.
#[allow(clippy::too_many_arguments)]
pub fn batch_rows(
    gguf: &Gguf,
    devices: &[Device],
    opts: &SessionOptions,
    placement: &Placement,
    pages: u32,
    page_bytes: &[u64],
    asked: u32,
    drafter: u64,
) -> Result<u32, OpenError> {
    let config = Config::from_gguf(gguf)?;
    let weights = Weights::load(gguf, &config)?;
    let (layout, _) = Layout::new(gguf, &config, &weights, placement, None)?;
    let own = opts.part.own_rows();
    // A layer keeps a row's picks while a loan runs (`ExpertStore::scratch_floor`).
    let floor = config.experts.used as u64;
    // Per device: the bytes of a slot of each of its layers, and of the
    // layers of each of its pools (slots of one size), the largest first: a
    // loan takes a piece of each.
    let rounds: Vec<(u64, Vec<u64>)> = (0..devices.len())
        .map(|d| {
            let mut sizes: Vec<(u64, u64)> = Vec::new();
            for spec in layout.specs.iter().filter(|s| s.device == d && !s.resident) {
                match sizes.iter_mut().find(|(size, _)| *size == spec.slot_bytes) {
                    Some((_, sum)) => *sum += spec.slot_bytes,
                    None => sizes.push((spec.slot_bytes, spec.slot_bytes)),
                }
            }
            let all = sizes.iter().map(|(_, sum)| sum).sum();
            let mut pools: Vec<u64> = sizes.iter().map(|(_, sum)| *sum).collect();
            pools.sort_unstable_by(|a, b| b.cmp(a));
            (all, pools)
        })
        .collect();
    widest_rows(own, asked.max(own), |rows| {
        let wide = Options { max_rows: rows, own_rows: own, ..opts.part.clone() };
        let narrow = Options { max_rows: own, own_rows: 0, ..opts.part.clone() };
        let fits = devices.iter().enumerate().all(|(d, device)| {
            let PartPlan { role, first, shares, head } = placement.part(&config, d);
            let shape = PartShape { role, first, shares, head };
            let mp = device.info().multiprocessors;
            let (Ok(f), Ok(g)) =
                (Frame::new(&config, &wide, &shape, mp), Frame::new(&config, &narrow, &shape, mp))
            else {
                return false;
            };
            let plan = &placement.devices[d];
            // A split by layers hands the residual of a step's rows to the next
            // device in a ring of two parcels, which the receiver holds.
            let lanes = if placement.tensor.is_none() && d > 0 {
                2 * u64::from(rows - own) * config.wide() as u64 * 4
            } else {
                0
            };
            let extra = f.step_bytes(&wide).saturating_sub(g.step_bytes(&narrow)) + lanes;
            let pool = u64::from(pages) * page_bytes.get(d).copied().unwrap_or(0);
            let drafters = if plan.head { drafter } else { 0 };
            let room = plan.slots.saturating_sub(pool + extra + drafters);
            let (all, pools) = &rounds[d];
            let per_layer = (room / (*all).max(1)).saturating_sub(floor);
            let pieces: Vec<u64> = pools.iter().map(|sum| per_layer * sum / 10 * 9).collect();
            Frame::new_in(&config, &wide, &shape, mp, &pieces)
                .is_ok_and(|f| f.lent_pieces().is_some())
        });
        Ok(fits)
    })
}

/// Where the layers of the model in `gguf` would go on `devices` with `opts`,
/// and the bytes one page of the pools would take on each device. Nothing is
/// kept on the devices; the placement of two devices that share every layer
/// measures a read of a stream mixer on each and, unless `opts.spread` gives
/// the worker's role, times both roles with parts opened for the trial and
/// closed again (`roles.rs`): the placement's `TensorPlan::trial` says what
/// that found. A caller that places the model again pins the choice.
pub fn place(
    gguf: &Gguf,
    devices: &[Device],
    opts: &SessionOptions,
) -> Result<(Placement, Vec<u64>), OpenError> {
    let config = Config::from_gguf(gguf)?;
    let weights = Weights::load(gguf, &config)?;
    place_in(gguf, &config, &weights, devices, opts)
}

/// What a page takes on each device of `placement`: the converted entries of
/// K and V of its attention layers, and the indexer's keys. The first lies in
/// host memory for a page beyond VRAM ([`Options::near_pages`]), the second in
/// VRAM for every page.
pub fn page_parts(
    gguf: &Gguf,
    devices: &[Device],
    opts: &SessionOptions,
    placement: &Placement,
) -> Result<Vec<(u64, u64)>, OpenError> {
    let config = Config::from_gguf(gguf)?;
    let mut parts = Vec::with_capacity(devices.len());
    for (index, device) in devices.iter().enumerate() {
        let PartPlan { role, first, shares, head } = placement.part(&config, index);
        let shape = PartShape { role, first, shares, head };
        let frame = Frame::new(&config, &opts.part, &shape, device.info().multiprocessors)?;
        parts.push(frame.page_parts());
    }
    Ok(parts)
}

fn place_in(
    gguf: &Gguf,
    config: &Config,
    weights: &Weights<'_>,
    devices: &[Device],
    opts: &SessionOptions,
) -> Result<(Placement, Vec<u64>), OpenError> {
    let Chosen { placement, .. } = choose(
        gguf,
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
    /// The pages whose K and V lie in host memory, when some do.
    far: Option<FarPages>,
    /// What each device keeps of a sequence and of a page, for the prefix
    /// cache (`resume.rs`).
    declared: Vec<Declared>,
    /// The file of the model's MTP block, when it drafts.
    mtp: Option<Arc<Gguf>>,
    /// A DFlash drafter, when it drafts.
    dflash: Option<Arc<crate::dflash::File>>,
    mtp_report: Mutex<Option<MtpReport>>,
    mtp_watch: Arc<Mutex<Watch>>,
    /// The usage profile of the experts, for a model that keeps one.
    usage: Usage,
    /// The token embedding table where the file is mapped (`blocks/rows.rs`).
    rows: Arc<HostTable>,
    /// The rows of the widest step, once the parts are open: what the expert
    /// slots can lend the scratch for, at most `opts.part.max_rows`.
    lone_rows: OnceLock<u32>,
    /// The most rows of a prompt read layer by layer, once the parts are open: what the
    /// expert slots can lend for its kept rows, at most `opts.pass_rows`; 0 for none.
    pass_span: OnceLock<u32>,
    /// The order of the output head's rows and the draft head's rows, which
    /// every drafter reads (`blocks/head.rs`).
    view: HeadView,
    /// The projector images are read through, and the device part that runs
    /// its tower.
    projector: Option<(Arc<Projector>, usize)>,
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
        // The parts read prompts layer by layer up to this many rows.
        opts.part.pass_rows = opts.pass_rows;
        let weights = Weights::load(&gguf, &config)?;
        let (placement, page_bytes) = place_in(&gguf, &config, &weights, devices, &opts)?;
        let (layout, _map) = Layout::new(&gguf, &config, &weights, &placement, None)?;
        let mut pinned = 0;
        let mut far_bytes = 0;
        let mut declared = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            let PartPlan { role, first, shares, head } = placement.part(&config, index);
            let layers = shares.len();
            // The part that holds the n-gram block drives the feed of its rows,
            // whose checkpoint keeps what the feed knows of a sequence.
            let feeds = role != super::Role::Worker
                && config
                    .ngram
                    .as_ref()
                    .is_some_and(|g| (first..first + layers).contains(&g.layer));
            let shape = PartShape { role, first, shares, head };
            let frame = Frame::new(&config, &opts.part, &shape, device.info().multiprocessors)?;
            far_bytes += frame.page_parts().0;
            let mut part = frame.declared(&opts.part);
            if feeds {
                part.checkpoint += ngram::saved_bytes(config.ngram.as_ref());
            }
            declared.push(part);
            let shares = if opts.serve == Serve::Stream {
                0
            } else {
                (layers * layout.per_layer) as u64
                    * layout.shapes[0].host_share_bytes(opts.part.max_rows.min(MAX_BATCH))
            };
            // The entries the store copies through are a ring of 256 KiB.
            pinned += frame.pinned_bytes(&opts.part, layers) + shares + (256 << 10);
        }
        if opts.part.prune.is_some() {
            pinned += PruneMask::bytes(config.n_layer, config.experts.count);
        }
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        let n = placement.n_devices();
        let format = chat::chat_format(&tokenizer, &ChatOptions::default());
        let gguf = Arc::new(gguf);
        let rows = Arc::new(HostTable::mapped(gguf.clone(), TOKEN_EMBD)?);
        let config_vocab = config.n_vocab as u32;
        let near = opts.part.near_pages();
        let far =
            (near < opts.part.pages()).then_some(FarPages { first: near, page_bytes: far_bytes });
        Ok(Self {
            gguf,
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
                mtp: false,
                mtp_place: None,
                prune_mask: None,
                lent: vec![0; n],
                lent_at: vec![None; n],
                lent_raw: vec![0; n],
                lent_pieces: vec![Vec::new(); n],
                lendable: vec![Vec::new(); n],
                multiprocessors: vec![0; n],
                lone_rows: 0,
                pass_rows: 0,
                far: vec![None; n],
            }),
            flow: OnceLock::new(),
            reports: Mutex::new(Vec::new()),
            pinned,
            page_bytes,
            far,
            declared,
            mtp: None,
            dflash: None,
            mtp_report: Mutex::new(None),
            mtp_watch: Arc::default(),
            usage: Usage::default(),
            rows,
            lone_rows: OnceLock::new(),
            pass_span: OnceLock::new(),
            view: HeadView::new(config_vocab, config_vocab, None).map_err(OpenError::Choice)?,
            projector: None,
        })
    }

    /// Reads images through `projector`, whose tower runs on device part
    /// `device`. Refused for a projector whose rows are not the model's width,
    /// and for a model whose n-gram embedding hashes another token for an image
    /// cell than the pad token the chat's markers write. Before the runtime
    /// opens the parts.
    pub fn with_projector(
        mut self,
        projector: Arc<Projector>,
        device: usize,
    ) -> Result<Self, OpenError> {
        if projector.n_out() as usize != self.config.n_embd {
            return Err(OpenError::Choice(format!(
                "the projector's rows have {} values and the model's have {}",
                projector.n_out(),
                self.config.n_embd
            )));
        }
        if device >= self.placement.n_devices() {
            return Err(OpenError::Choice(format!(
                "the image tower is to run on part {device} of {}",
                self.placement.n_devices()
            )));
        }
        // The n-gram embedding hashes its image token for an image cell; the
        // record carries the pad token the chat's markers write there.
        let pad = self.tokenizer.special_token_id("<|image_pad|>");
        if let Some(ngram) = &self.config.ngram
            && pad != Some(ngram.image_token)
        {
            return Err(OpenError::Choice(format!(
                "the n-gram embedding hashes token {} for an image cell, and the chat writes \
                 {pad:?} there",
                ngram.image_token
            )));
        }
        self.projector = Some((projector, device));
        Ok(self)
    }

    /// The projector images are read through, if there is one.
    pub fn projector(&self) -> Option<&Arc<Projector>> {
        self.projector.as_ref().map(|(p, _)| p)
    }

    /// Gives every drafter of the model a draft vocabulary (`blocks/head.rs`):
    /// the output head's rows of its tokens come first, the drafters read only
    /// them, and the device of the head keeps the rows of the embedding table
    /// of only them. Before the runtime opens the parts.
    pub fn with_draft_vocabulary(mut self, tokens: &[u32]) -> Result<Self, OpenError> {
        let n = self.config.n_vocab as u32;
        self.view = HeadView::new(n, n, Some(tokens)).map_err(OpenError::Choice)?;
        self.opts.part.head_order = self.view.order.clone();
        Ok(self)
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

    /// Fills the tiers from the heat the steps so far left, in one sweep
    /// through the file at the pace of a start, as a start from a profile
    /// does: after a warm-up, so that VRAM holds the hottest experts of every
    /// layer before the first request. Returns what it brought and how long
    /// it took; `None` without a usage profile or before the engine serves.
    /// Call it while no request runs.
    pub fn fill_from_heat(&self) -> Option<(FillStats, Duration)> {
        self.usage.fill_from_heat()
    }

    /// What the fill of the empty slots did while nothing ran, and what the
    /// tiers of the model's layers hold; `None` without a usage profile.
    pub fn usage_stats(&self) -> Option<UsageStats> {
        self.usage.stats()
    }

    /// Gives the model its MTP block from `draft`, a file of the block as
    /// Strata packs it ([`MtpWeights`]): windows then take the block's
    /// drafts, as many rows as `opts.part.window_rows` allows. The block runs
    /// on the device that holds the output head, in every placement, and its
    /// routed experts go through the expert store with the model's. Before the
    /// runtime opens the parts.
    /// Takes a DFlash drafter: windows then take its drafts as well, as many
    /// rows as `opts.part.window_rows` allows. It runs on the device that
    /// holds the streams of every layer it reads and the output head: one
    /// device, or the mixer of two devices on every layer. Before the runtime
    /// opens the parts.
    pub fn with_dflash(mut self, file: Arc<crate::dflash::File>) -> Result<Self, OpenError> {
        let choice = |why: String| Err(OpenError::Choice(why));
        if self.opts.part.window_rows < 2 {
            return choice("the DFlash drafter needs windows of at least two rows".into());
        }
        if self.opts.part.window_rows > MAX_BATCH {
            return choice(
                "a step that verifies drafts runs the decode kernels, which take eight rows".into(),
            );
        }
        let c = &self.config;
        if let Err(e) = file.config.fits(c.n_embd, c.n_layer, c.n_vocab) {
            return choice(e.to_string());
        }
        if self.placement.tensor.is_none() && self.placement.n_devices() > 1 {
            return choice(
                "the DFlash drafter is not built for a split by layers: its taps would have to \
                 cross to the device of the head"
                    .into(),
            );
        }
        self.dflash = Some(file);
        Ok(self)
    }

    pub fn with_mtp(mut self, file: Gguf) -> Result<Self, OpenError> {
        let choice = |why: &str| Err(OpenError::Choice(why.to_owned()));
        if self.opts.part.window_rows < 2 {
            return choice("an MTP block needs windows of at least two rows");
        }
        if self.opts.part.window_rows > MAX_BATCH {
            return choice(
                "a step that verifies drafts runs the decode kernels, which take eight rows",
            );
        }
        let (c, n) = (&self.config, self.placement.n_devices());
        let head = MtpWeights::load(&file, c)?;
        let weights = Weights::load(&self.gguf, c)?;
        let device = self.head_device();
        let layer = MtpExperts { gguf: &file, weights: &head, device };
        let (layout, _map) = Layout::new(&self.gguf, c, &weights, &self.placement, Some(&layer))?;
        // What the block keeps of a sequence (its slot and the carry) and of
        // a page (its K/V group and pooled keys), next to the head's part.
        let block = mtp_config(c);
        let block_opts = Options {
            kv: MTP_KV,
            max_rows: MAX_BATCH,
            logit_rows: MAX_BATCH,
            window_rows: 0,
            host_logits: false,
            profile: None,
            ..self.opts.part.clone()
        };
        let frame = Frame::new(&block, &block_opts, &PartShape::whole(&block, 0..1, true), 1)?;
        let mut declared = frame.declared(&block_opts);
        let part = &mut self.declared[device];
        part.checkpoint += declared.checkpoint + c.wide() as u64 * 4;
        part.groups.append(&mut declared.groups);
        part.arrays += declared.arrays;
        // The block's host share and its staged rows and picks.
        if self.opts.serve != Serve::Stream {
            self.pinned += layout.shapes[0].host_share_bytes(MAX_BATCH);
        }
        self.pinned += u64::from(MAX_BATCH) * c.wide() as u64 * 4 + (64 << 10);
        self.layout = layout;
        debug_assert!(device < n);
        drop((head, weights));
        self.mtp = Some(Arc::new(file));
        Ok(self)
    }

    /// The device part that holds the final mixer and the output head.
    fn head_device(&self) -> usize {
        (0..self.placement.n_devices())
            .find(|d| self.placement.part(&self.config, *d).head)
            .unwrap_or(0)
    }

    /// What the MTP block took on its device, once the runtime has opened
    /// the parts; `None` for a model without one.
    pub fn mtp_report(&self) -> Option<MtpReport> {
        *self.mtp_report.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the MTP block's steps took and how its drafts fared since the
    /// engine started.
    pub fn mtp_times(&self) -> MtpTimes {
        self.mtp_watch.lock().unwrap_or_else(|e| e.into_inner()).times
    }

    /// The newest chains the MTP block made, oldest first (the last 1024),
    /// and forgets them: for tests and measurements that compare its drafts
    /// with the reference's.
    pub fn take_mtp_chains(&self) -> Vec<MtpDrafted> {
        self.mtp_watch.lock().unwrap_or_else(|e| e.into_inner()).recent.drain(..).collect()
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

    /// The counters of the reader of the model file (reads by class, bytes, latencies); `None`
    /// before the parts are open.
    pub fn read_stats(&self) -> Option<ares_io::Stats> {
        Some(self.flow.get()?.lock_feed().store.reader_stats())
    }

    /// How often the predictions of steps of up to eight rows named an expert that no tier
    /// held and that was then picked, by kind of step, distance and place (see
    /// `ExpertStore::ahead_odds`); `None` before the parts are open.
    pub fn ahead_odds(&self) -> Option<[[(f32, f32); 16]; 4]> {
        Some(self.flow.get()?.lock_feed().store.ahead_odds())
    }

    /// How often the guesses of wide steps were then named, by class (see
    /// `ExpertStore::transit_odds`); `None` before the parts are open.
    pub fn guess_odds(&self) -> Option<[(f32, f32); 9]> {
        Some(self.flow.get()?.lock_feed().store.transit_odds())
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
        totals.reads += stats.demand_reads
            + stats.guess_reads
            + stats.transit_reads
            + stats.transit_ahead_reads;
        totals.demand_reads += stats.demand_reads;
        totals.demand_read_us += (stats.demand_read_ms * 1e3) as u64;
        totals.wait_us += (stats.wait_ms * 1e3) as u64;
        totals.predicted += stats.predicted;
        totals.predicted_picked += stats.predicted_picked;
        // Wide steps read their predicted picks ahead through transit's guesses.
        totals.ahead_reads += stats.ahead_reads + stats.transit_ahead_reads;
        totals.ahead_read_bytes += stats.ahead_read_bytes;
        totals.ahead_used += stats.ahead_used + stats.transit_ahead_used;
        totals.ahead_wasted += stats.ahead_wasted + stats.transit_ahead_unused;
        totals.transit_admitted += stats.transit_admitted;
        totals.transit_adopted += stats.transit_adopted;
        totals.transit_guess_uploads += stats.transit_guess_uploads;
        totals.transit_guess_uploads_used += stats.transit_guess_uploads_used;
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

    /// What the expert store measured of serving a miss on this machine: per
    /// pair of types of the experts (gate and up, down), the seconds a CPU job
    /// takes per (expert, row) pair for one row and for eight, and per device
    /// the seconds of one upload (`None` before one was timed). `None` before
    /// the parts are open or without a CPU half.
    pub fn expert_costs(&self) -> Option<ExpertCosts> {
        let flow = self.flow.get()?;
        let feed = flow.lock_feed();
        let model = feed.store.cost_model()?;
        let cpu = model
            .timed_types()
            .into_iter()
            .map(|types| {
                let per = |pairs: usize| model.job_time(types, pairs).map(|t| t / pairs as f64);
                (types, per(1), per(8))
            })
            .collect();
        let devices = self.placement.n_devices();
        let upload = (0..devices).map(|d| model.upload_time(d)).collect();
        Some(ExpertCosts { cpu, upload })
    }

    /// The slots of the MTP block's experts in VRAM and in the RAM tier;
    /// `None` without an MTP block or before the parts are open.
    pub fn mtp_slots(&self) -> Option<(usize, usize)> {
        self.mtp.as_ref()?;
        Some(self.flow.get()?.lock_feed().store.slots(0))
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers before anything is served, most wanted first, as far as the slots
    /// go (a recorded routing, or every expert for a model that fits); the
    /// MTP block is asked for as the layer behind the model's last. Call it
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
            || cx.max_rows != o.max_rows.max(self.opts.pass_rows)
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
                o.max_rows.max(self.opts.pass_rows)
            )));
        }
        // The image tower, on the part that runs it, before the part and its
        // expert slots take what is left of the device.
        let tower = match &self.projector {
            Some((projector, device)) if *device == index => {
                cx.ledger.reserve("image tower", projector.vram_bytes())?;
                let tower = projector
                    .load(cx.device, cx.kernels)
                    .map_err(|e| invalid(format!("the image tower: {e}")))?;
                Some(tower)
            }
            _ => None,
        };
        let weights = Weights::load(&self.gguf, c).map_err(|e| invalid(e.to_string()))?;
        let PartPlan { role, first, shares, head } = self.placement.part(c, index);
        let layer_count = shares.len();
        let mut part = Part::open(
            c,
            &weights,
            o,
            PartSetup { index, role, first, shares, head, device: cx.device, ledger: cx.ledger },
        )?;
        // The K and V of the pages beyond VRAM: host memory the store's flow
        // locks as the pages come into use.
        let far = part.open_far(cx.ram)?.map(|pool| Arc::new(Mutex::new(pool)));
        let near = o.near_pages();
        let serve = self.opts.serve;
        let layout = &self.layout;
        // The MTP block, next to the part that holds the head: before the
        // room for slots is counted.
        part.set_token_rows(self.rows.clone());
        let mut block = match (&self.mtp, part.output()) {
            (Some(draft), Some(output)) if index == self.head_device() => {
                let head = MtpWeights::load(draft, c).map_err(|e| invalid(e.to_string()))?;
                let block = MtpBlock::open(
                    c,
                    &head,
                    o,
                    index,
                    output,
                    part.head_tokens(),
                    &self.view,
                    self.rows.clone(),
                    self.mtp_watch.clone(),
                    cx.device,
                    cx.ledger,
                )?;
                *self.mtp_report.lock().unwrap_or_else(|e| e.into_inner()) = Some(block.report());
                Some(block)
            }
            _ => None,
        };
        // The DFlash drafter on the same part, also before the room for slots
        // is counted: its weights and rings are taken from the experts'.
        let dflash_embedding = match &self.dflash {
            Some(file) if index == self.head_device() => {
                let mut mask = vec![0f32; c.n_embd];
                self.rows.decode(file.config.mask_token, &mut mask)?;
                part.open_dflash(
                    file,
                    cx.device,
                    cx.ledger,
                    cx.max_slots,
                    c.n_layer,
                    c.n_vocab,
                    self.view.rows,
                    mask,
                )?;
                Some(self.rows.clone())
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
        let holds_mtp = block.is_some();
        let lent_bytes = part.lent_bytes();
        self.meeting.meet(
            n,
            |build| {
                build.lent[index] = lent_bytes;
                build.far[index] = far.clone();
                build.multiprocessors[index] = cx.device.info().multiprocessors;
                build.room[index] = room;
                build.ports[index] = ports;
                build.cores[index] = cores;
                build.boards[index] = Some(board);
                build.mtp |= holds_mtp;
                Ok(())
            },
            |build| {
                let vram: Vec<u64> = (0..n)
                    .map(|d| {
                        let own = layout.mtp_bytes(d)
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
                let reader = reader(&self.gguf, self.mtp.as_deref()).map_err(model_error)?;
                let head = self
                    .mtp
                    .as_ref()
                    .map(|draft| MtpWeights::load(draft, c))
                    .transpose()
                    .map_err(|e| invalid(e.to_string()))?;
                let mtp_experts = self.mtp.as_ref().zip(head.as_ref()).map(|(gguf, weights)| {
                    MtpExperts { gguf, weights, device: self.head_device() }
                });
                let map = expert_map(&self.gguf, c, layout.per_layer, mtp_experts.as_ref())
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
                config.cpu_serves = serve != Serve::Stream;
                if o.max_rows > MAX_BATCH {
                    let room: Vec<u64> = build
                        .room
                        .iter()
                        .map(|r| (*r).min(self.opts.slot_bytes.unwrap_or(u64::MAX)))
                        .collect();
                    plan_transit(&mut config, layout, &reader, self.opts.transit, &room);
                }
                config.ram_bytes = reach;
                config.ram_piece = self.opts.ram_piece.unwrap_or(PIECE) as u64;
                let plan =
                    StorePlan::new(config, &map, &reader, &layout.specs, None).map_err(|e| {
                        // With both devices on every layer each holds a share of the
                        // experts of every layer, and every one of those needs its slots.
                        let shared = self.placement.tensor.is_some()
                            && matches!(e, StoreError::Plan(ref text) if text.starts_with("device"));
                        invalid(if shared {
                            format!(
                                "{e}. Both devices hold a share of the experts of each of the {} \
                                 layers; with the layers split between the devices (--split layer) \
                                 each holds the slots of its own layers only",
                                c.n_layer
                            )
                        } else {
                            e.to_string()
                        })
                    })?;
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
                // The widest step a prompt read alone takes: the rows the
                // parts were opened for, or the most below them, in multiples
                // of `ROW_GRAIN`, whose scratch every device's slots can lend,
                // laid out in a piece of each of its pools (the plan's estimate
                // may hold more); and where each device's slots lend it: always
                // the same slots.
                let own = o.own_rows();
                for d in 0..n {
                    build.lendable[d] = store.lendable_pieces(d);
                }
                let lent_of =
                    |d: usize, rows: u32| -> Result<Option<(Vec<u64>, u64)>, ModelError> {
                        let PartPlan { role, first, shares, head } = self.placement.part(c, d);
                        let shape = PartShape { role, first, shares, head };
                        let opts = Options { max_rows: rows, own_rows: own, ..o.clone() };
                        let mp = build.multiprocessors[d];
                        let frame = Frame::new_in(c, &opts, &shape, mp, &build.lendable[d])?;
                        Ok(frame.lent_pieces().map(|pieces| (pieces, frame.lent_raw())))
                    };
                let asked = if build.lent.iter().any(|b| *b > 0) { o.max_rows } else { own };
                let rows = widest_rows(own, asked, |rows| {
                    let mut fits = true;
                    for d in 0..n {
                        fits &= lent_of(d, rows)?.is_some();
                    }
                    Ok::<bool, ModelError>(fits)
                })?;
                build.lone_rows = rows;
                // The widest prompt read layer by layer: its kept rows in what every device's
                // slots can lend.
                // Where K and V lie partly in host memory, beside a copy of every far slot of a
                // layer: what the loan of a pass at the longest context holds of them.
                let far_entry = |d: usize| {
                    build.far[d].as_ref().map_or(0, |far| {
                        far.lock().unwrap_or_else(|e| e.into_inner()).entry_bytes()
                    })
                };
                let far_slots = build
                    .far
                    .iter()
                    .flatten()
                    .map(|far| far.lock().unwrap_or_else(|e| e.into_inner()).slots())
                    .max()
                    .unwrap_or(0);
                let pass_sizes: Vec<PassSizes> = (0..n)
                    .map(|d| {
                        PassSizes::of(c, self.placement.part(c, d).role).with_far(far_entry(d))
                    })
                    .collect();
                build.pass_rows = if self.reads_layer_by_layer() {
                    pass::pass_span(own, self.opts.pass_rows, |rows| {
                        pass::pass_fits(&pass_sizes, &build.lendable, own, rows, far_slots)
                    })
                } else {
                    0
                };
                if build.pass_rows <= own {
                    build.pass_rows = 0;
                }
                for d in 0..n {
                    let (pieces, raw) = match rows > own {
                        true => lent_of(d, rows)?
                            .ok_or_else(|| invalid("the slots lend less than they said"))?,
                        false => (Vec::new(), 0),
                    };
                    build.lent[d] = pieces.iter().sum();
                    build.lent_raw[d] = raw;
                    build.lent_at[d] = None;
                    build.lent_pieces[d] = pieces.clone();
                    if pieces.is_empty() {
                        continue;
                    }
                    let loans = store
                        .scratch_pieces_at(d, &pieces)
                        .ok_or_else(|| invalid("the slots lend less than they said"))?;
                    // Each piece starts in the layout where the pieces before it
                    // end as the scratch was laid out: at what each pool can lend.
                    let mut start = 0;
                    let mut at = Vec::with_capacity(loans.len());
                    let mut loans = loans.iter();
                    for (piece, lendable) in pieces.iter().zip(&build.lendable[d]) {
                        if *piece > 0 {
                            let loan =
                                loans.next().ok_or_else(|| invalid("a piece without its loan"))?;
                            let (base, _) = store.device(d).pool(loan.pool);
                            at.push((start, base.offset(u64::from(loan.first) * loan.stride)));
                        }
                        start += lendable / 256 * 256;
                    }
                    build.lent_at[d] = Some(Pieces(at));
                }
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
                if self.mtp.is_some() {
                    let (device, pool) = store.place(0);
                    let slots = store.device(device);
                    build.mtp_place = Some(ExpertAt {
                        table: slots.table(0),
                        pool: slots.pool(pool).0,
                        host: slots.host_share(0),
                    });
                }
                let mut feed =
                    Feed::new(store, boards, firsts, layout.clone(), c.experts.used as u32);
                feed.set_read_ahead(self.opts.read_ahead);
                if o.prune.is_some() {
                    let mask = PruneMask::new(device, c.n_layer, c.experts.count)?;
                    let mask = Arc::new(mask);
                    let owners = prune::owners(&self.placement, c);
                    feed.set_prune(mask.clone(), owners, c.experts.count);
                    build.prune_mask = Some(mask);
                }
                if self.placement.tensor.is_some() {
                    feed.set_owners(owners(&self.placement, c));
                }
                if build.mtp {
                    feed.set_mtp();
                    feed.fill_mtp(c.experts.count).map_err(model_error)?;
                }
                let ngram = NgramFeed::open(&self.gguf, c, self.opts.row_cache, o.max_slots)
                    .map_err(model_error)?;
                let flow = Arc::new(Flow {
                    feed: Mutex::new(feed),
                    ngram: Mutex::new(ngram),
                    enqueued: (0..n).map(|_| AtomicUsize::new(0)).collect(),
                    followed: (0..n).map(|_| AtomicUsize::new(0)).collect(),
                    begun: AtomicU64::new(0),
                    finished: AtomicUsize::new(0),
                    failed: AtomicBool::new(false),
                    uploads_seen: AtomicU64::new(0),
                    stall: self.opts.stall,
                    wide: Mutex::new(WideStats::default()),
                    own_rows: o.own_rows(),
                    lent: build.lent_pieces.clone(),
                    lent_at: build.lent_at.clone(),
                    on_loan: AtomicBool::new(false),
                    pass_rows: build.pass_rows,
                    pass_sizes,
                    top_k: c.experts.used as u32,
                    pass: Mutex::new(None),
                    pass_completed: AtomicBool::new(false),
                    pass_paused: AtomicBool::new(false),
                    pass_sent: AtomicUsize::new(0),
                    pass_released: AtomicUsize::new(0),
                    pass_go: AtomicU64::new(0),
                    pass_picks: Mutex::new(None),
                    pass_count: AtomicU64::new(0),
                    far: build.far.iter().flatten().cloned().collect(),
                    far_live: Mutex::new(Vec::new()),
                    ram: ram.clone(),
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
        let (places, flow, peer, n_slots, slot_bytes, prune_mask) = {
            let build = self.meeting.lock();
            let build = &build.value;
            if let Some(block) = &mut block {
                let at = build.mtp_place.ok_or_else(|| invalid("the MTP block has no slots"))?;
                block.set_experts(at)?;
            }
            let peer = self.placement.tensor.as_ref().map(|tensor| {
                let other = 1 - index;
                let lent_raw =
                    build.lent_at[other].as_ref().map(|at| at.address(build.lent_raw[other]));
                (build.ports[other], build.cores[other].clone(), tensor.mixer, lent_raw)
            });
            let _ = self.lone_rows.set(build.lone_rows);
            let _ = self.pass_span.set(build.pass_rows);
            (
                build.places[index].clone(),
                build.flow.clone().ok_or_else(|| invalid("no flow was made"))?,
                peer,
                build.n_slots[index],
                build.slot_bytes[index],
                build.prune_mask.clone(),
            )
        };
        part.set_experts(places)?;
        if flow.pass_rows > 0 && !part.takes_passes() {
            return Err(invalid("a part cannot read prompts layer by layer"));
        }
        let PartPlan { role, first, shares, head } = self.placement.part(c, index);
        let lendable = self.meeting.lock().value.lendable[index].clone();
        part.lend_rows(c, &PartShape { role, first, shares, head }, self.max_rows(), &lendable)?;
        if let Some(mask) = prune_mask {
            part.set_prune_mask(mask);
        }
        if let Some((ports, cores, mixer, peer_lent)) = peer {
            let ports = ports.ok_or_else(|| invalid("a part that shares layers has no ports"))?;
            part.set_peer(ports, cores, peer_lent)?;
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
        // The checkpoints of this device: what the model said it keeps of a
        // sequence here is what it copies.
        let feeds = part.ngram_layer().map_or(0, |_| ngram::saved_bytes(c.ngram.as_ref()));
        let resume = Resume::open(device, ram, cx.checkpoints, &part, block.as_ref(), feeds)?;
        if resume.bytes() != self.declared[index].checkpoint {
            return Err(invalid(format!(
                "a checkpoint of device part {index} takes {} bytes; the model declared {}",
                resume.bytes(),
                self.declared[index].checkpoint
            )));
        }
        Ok(Box::new(Qwen4ExpDevice {
            tower,
            rotary: Vec::new(),
            index,
            devices: n,
            by_layers: self.placement.tensor.is_none(),
            n_vocab: c.n_vocab as u32,
            wide: c.wide() as u32,
            // A pass numbers the exchanges of its pieces from the step's base: a block of
            // eight per layer and micro-batch, and one layer more for its end.
            per_step: 8
                * (c.n_layer as u32 + 1)
                * (flow.pass_rows.div_ceil(o.own_rows().max(1)) + 2),
            mtp: block,
            drafts: self.mtp.is_some() || self.dflash.is_some(),
            mtp_id: self.mtp.as_ref().map(|_| 1),
            dflash_id: self.dflash.as_ref().map(|_| 1 + u32::from(self.mtp.is_some())),
            dflash_embedding,
            passes: Vec::new(),
            residual: None,
            picks: None,
            pass_run: None,
            paused: false,
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
            accepted: Vec::new(),
            windows: Vec::new(),
            shape: Vec::new(),
            replay: Vec::new(),
            resume,
            near: (near < o.pages()).then_some(near),
            live: Vec::new(),
        }))
    }
}

impl Qwen4Exp {
    /// Whether the model on `devices` devices reads a prompt wider than the parts' own
    /// scratch layer by layer: asked for, with micro-batches the many-row kernels take, K and
    /// V without an exact pool (whose tiles convert per step for every layer at once), no
    /// DFlash drafter (which reads the residual of a step's rows at some layers), and a
    /// placement the pass is built for.
    fn reads_layer_by_layer(&self) -> bool {
        self.opts.pass_rows > self.opts.part.own_rows()
            && reads_layer_by_layer(&self.opts.part, self.dflash.is_some())
    }

    /// Everything besides the tokens that changes the bytes of a page or of a
    /// checkpoint as the device parts hand them out
    /// ([`ares_runtime::Durable::fingerprint`]):
    ///
    /// - the model's files and the MTP block's (size, header and samples
    ///   of the weights);
    /// - the formats of K and V, of the pooled keys and of the recurrent
    ///   state, and how the products of a prompt round;
    /// - what each device holds of every layer (its role, its heads, the
    ///   shared expert), since the bytes are laid out per device.
    ///
    /// How a missed expert is served is not in it: the CPU's sums of a miss
    /// differ from a device's by rounding, as two runs differ in their misses.
    fn state_fingerprint(&self) -> Vec<u8> {
        let mut out = b"qwen4exp state 1\0".to_vec();
        for part in 0..self.gguf.n_parts() {
            crate::files::identity(self.gguf.path(part as u16), &mut out);
        }
        if let Some(draft) = &self.mtp {
            out.extend_from_slice(b"mtp block\0");
            for part in 0..draft.n_parts() {
                crate::files::identity(draft.path(part as u16), &mut out);
            }
        }
        let o = &self.opts.part;
        let settings = format!(
            "kv {:?} keys {:?} state {:?} precision {:?} mtp {:?}",
            o.kv,
            o.keys,
            o.state,
            o.precision,
            self.mtp.as_ref().map(|_| MTP_KV)
        );
        out.extend_from_slice(settings.as_bytes());
        for device in 0..self.placement.n_devices() {
            let PartPlan { role, first, shares, head } = self.placement.part(&self.config, device);
            out.push(0xff);
            out.extend_from_slice(format!("{role:?} {first} {head}").as_bytes());
            for share in shares {
                out.extend_from_slice(&share.heads.start.to_le_bytes());
                out.extend_from_slice(&share.heads.end.to_le_bytes());
                out.push(u8::from(share.shared));
            }
        }
        out
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
        chat::encode(&self.tokenizer, &self.chat, self.projector(), prompt)
    }

    fn encode_with_boundaries(
        &self,
        prompt: &Prompt,
    ) -> Result<ares_runtime::EncodedPrompt, EngineError> {
        chat::encode_with_boundaries(&self.tokenizer, &self.chat, self.projector(), prompt)
    }

    fn vision(&self) -> Option<Vision> {
        self.projector.as_ref().map(|(projector, device)| Vision {
            width: self.config.n_embd as u32,
            max_rows: projector.limits().max_rows as u32,
            device: *device,
            units: projector.units(),
        })
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
                lane_slots: pass::PARCELS as u32,
                lane_values: self.opts.part.max_rows * self.config.wide() as u32,
            },
            _ => LinkSpec::default(),
        }
    }

    fn saves_inside_steps(&self) -> u32 {
        // A pass saves each layer's state as it goes past a checkpoint, between two
        // micro-batches, each wider than the decode kernels take.
        match self.pass_span.get() {
            Some(&span) if span > 0 => 2 * MAX_BATCH,
            _ => 0,
        }
    }

    fn steps_wait_for_fetches(&self) -> bool {
        // An expert that no tier in VRAM holds is brought while the step
        // waits for it.
        true
    }

    fn pauses_wide_steps(&self) -> bool {
        // A pass pauses between two micro-batches once the record's time has
        // passed.
        self.pass_span.get().is_some_and(|&span| span > 0)
    }

    fn pinned_ram(&self) -> u64 {
        self.pinned + self.projector().map_or(0, |p| p.pinned_bytes())
    }

    fn expert_stats(&self) -> Option<ExpertStats> {
        self.take_store_stats()?;
        let totals = self.flow.get()?.totals.lock().unwrap_or_else(|e| e.into_inner());
        Some(*totals)
    }

    fn max_rows(&self) -> u32 {
        // Once the parts are open: the rows the slots can lend for, a prompt read layer by
        // layer's or a lent step's.
        match (self.pass_span.get(), self.lone_rows.get()) {
            (Some(&pass), _) if pass > 0 => pass,
            (_, Some(&lone)) => lone,
            _ => self.opts.part.max_rows.max(self.opts.pass_rows),
        }
    }

    fn prompt_rows(&self) -> u32 {
        self.opts.part.own_rows()
    }

    fn own_rows(&self) -> u32 {
        self.opts.part.own_rows()
    }

    fn logit_rows(&self) -> u32 {
        self.opts.part.logit_rows
    }

    fn draft_rows(&self) -> u32 {
        self.opts.part.window_rows
    }

    fn drafters(&self) -> Vec<Drafter> {
        let mut drafters = Vec::new();
        if self.mtp.is_some() {
            drafters.push(Drafter {
                kind: DrafterKind::Mtp,
                drafts: self.opts.part.window_rows.saturating_sub(1),
            });
        }
        if let Some(file) = &self.dflash {
            let drafts =
                (file.config.drafts() as u32).min(self.opts.part.window_rows.saturating_sub(1));
            drafters.push(Drafter { kind: DrafterKind::Dflash, drafts });
        }
        drafters
    }

    fn attention_span(&self) -> u32 {
        self.config.indexer.dense_limit() as u32
    }

    fn prefix_cache(&self) -> Option<PrefixCache> {
        Some(slots::prefix_cache(&self.declared, Some(TAILS_PER_SLOT)))
    }

    fn durable(&self, cells_per_page: u32) -> Option<Durable> {
        (cells_per_page == ENTRY_CELLS)
            .then(|| slots::durable(&self.declared, self.state_fingerprint()))
    }

    fn page_entry(&self, page: u32) -> u32 {
        page | PAGE_ENTRY
    }

    fn far_pages(&self) -> Option<FarPages> {
        self.far
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
    /// The image tower, on the part that runs it.
    tower: Option<TowerAt>,
    /// The rotary positions of the step's rows, where they are not their cells.
    rotary: Vec<[i32; 4]>,
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
    /// Whether this part keeps the sequences the n-gram table is read for.
    drives_ngram: bool,
    routing: Routing,
    part: Part,
    flow: Arc<Flow>,
    // The step being enqueued, kept to reuse the memory.
    segments: Vec<Segment>,
    tokens: Vec<u32>,
    out_rows: Vec<u32>,
    accepted: Vec<u32>,
    /// The draft windows of the step before, as (slot, first row), which the
    /// record of this step says the outcome of.
    windows: Vec<(u32, u32)>,
    /// The step that ended last: its segments as (slot, direct, rows, first
    /// row), and its wanted rows.
    shape: Vec<(u32, bool, u32, u32)>,
    /// The model's MTP block, on the part that holds the output head.
    mtp: Option<MtpBlock>,
    /// Whether the model drafts (on this part or another).
    drafts: bool,
    /// The ids the record names the MTP block and the DFlash drafter by.
    mtp_id: Option<u32>,
    dflash_id: Option<u32>,
    /// The embedding rows, for the DFlash drafter's newest tokens, on the part
    /// that holds it.
    dflash_embedding: Option<Arc<HostTable>>,
    /// The windows of the step whose drafts the DFlash drafter makes.
    passes: Vec<crate::dflash::PassSeq>,
    /// Where the part left the residual of the rows of its last step.
    residual: Option<DevicePtr>,
    /// The host's copy of the picks of a pass's layer, kept for the next pass.
    picks: Option<Arc<SpanPicks>>,
    /// The pass this part paused, and whether the step that ended last was paused.
    pass_run: Option<PassRun>,
    paused: bool,
    /// The windows of the step whose drafts the MTP block makes.
    chains: Vec<MtpChain>,
    /// The step being run only drafts.
    only_drafts: bool,
    /// The rows of the step that are a replay.
    replay: Vec<Range<u32>>,
    /// The checkpoints of the prefix cache on this device.
    resume: Resume,
    /// The first page whose K and V are in host memory, when some are, and
    /// which pages the tables name and the pool has not freed: the far pools
    /// keep the pieces of those locked (every device keeps the same books,
    /// any of them may begin a step for the store).
    near: Option<u32>,
    live: Vec<bool>,
}

impl Qwen4ExpDevice {
    /// Page `page` is in use until the pool frees it.
    fn mark_live(&mut self, page: u32) {
        if self.near.is_none() {
            return;
        }
        if self.live.len() <= page as usize {
            self.live.resize(page as usize + 1, false);
        }
        self.live[page as usize] = true;
    }

    /// The pages the tables name after the step's updates, less those the
    /// pool freed: the far pools keep the pieces of the far ones locked.
    fn update_live(&mut self, record: &ares_runtime::RecordView<'_>) {
        if self.near.is_none() {
            return;
        }
        for i in 0..record.freed() {
            if let Some(live) = self.live.get_mut(record.freed_page(i) as usize) {
                *live = false;
            }
        }
        for k in 0..record.entries() {
            self.mark_live(record.entry(k) & !PAGE_ENTRY);
        }
    }
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
        self.passes.clear();
        self.replay.clear();
        // The rows' rotary positions where they are not their cells (a
        // sequence with images): the step's record and the MTP block's take
        // them.
        self.rotary.clear();
        let at_cells = (0..rows).all(|i| {
            let p = record.row(i).position;
            record.rotary(i) == [p, p, p, 0]
        });
        if !at_cells {
            self.rotary.extend((0..rows).map(|i| record.rotary(i).map(|p| p as i32)));
        }
        if let Some(block) = &mut self.mtp {
            block.set_rotary(self.rotary.clone());
        }
        let (mut bar, mut place) = (0, 0);
        for i in 0..record.segments() {
            let segment = record.segment(i);
            if record.replay(i) {
                self.replay.push(segment.row0..segment.row0 + segment.n_rows);
            }
            if let Some(drafter) = record.drafter(i) {
                if !self.drafts {
                    return Err(invalid("the model drafts nothing itself"));
                }
                // The DFlash drafter's windows: one pass each, its drafts at
                // the window's place in the step's list.
                if segment.n_rows >= 2 && Some(drafter) == self.dflash_id {
                    let links = segment.n_rows - 1;
                    bar += links;
                    self.passes.push(crate::dflash::PassSeq {
                        slot: segment.slot,
                        position: segment.pos0 as u32,
                        newest: record.row(segment.row0).token,
                        record_row: segment.row0 + 1,
                        wanted: links,
                        place,
                    });
                    place += links;
                } else if segment.n_rows >= 2 && Some(drafter) != self.mtp_id {
                    return Err(invalid(format!("the model has no drafter {drafter}")));
                } else if segment.n_rows >= 2 {
                    let at = place;
                    place += segment.n_rows - 1;
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
                    self.chains.push(MtpChain {
                        segment: i,
                        token: record.row(segment.row0).token,
                        rows: (0..links).map(|j| record.sampler(wanted + j).for_draft()).collect(),
                        needs,
                        place: at,
                    });
                }
            }
            self.segments.push(segment);
        }
        // The rows that take the block's drafts have no token yet.
        let mut drafted = vec![false; rows as usize];
        for chain in &self.chains {
            let s = &self.segments[chain.segment as usize];
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
        // The step begins for the store: the MTP block routes in it before
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
        // A prompt wider than the parts' own scratch is read layer by layer; a record may
        // give up the pass a step before paused, or go on with it.
        self.paused = false;
        if record.gives_up_paused() {
            self.give_up_pass();
        }
        let pass = self.pass_of(step)?;
        self.update_live(record);
        let far_live =
            self.near.map_or(&[][..], |near| self.live.get(near as usize..).unwrap_or(&[]));
        self.flow.begin(number, rows, windows, far_live, self.index, stream, pass.as_ref())?;
        // What this part keeps of a pass that has ended, or that another pass of the same
        // rows has taken the loan of, goes; a paused pass goes on.
        self.forget_ended_pass();
        let resumes = self.resumes(pass.as_ref());
        self.part.set_lent(self.flow.lent(self.index));

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

        // The MTP block follows the rows of the last step that count, in
        // the tables of that step.
        if let (Some(block), Some(residual)) = (&mut self.mtp, self.residual.take()) {
            let outcome = |slot: u32, row0: u32| {
                (0..record.commits())
                    .map(|i| record.commit(i))
                    .find(|c| c.slot == slot && c.row0 == row0)
                    .map(|c| c.accepted)
            };
            block.follow(stream, &outcome, residual)?;
        }

        // The pages the pool freed give back what the K/V groups keep for them.
        let freed: Vec<u32> = (0..record.freed()).map(|i| record.freed_page(i)).collect();
        self.part.release_pages(freed.iter().copied());
        if let Some(block) = &mut self.mtp {
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
            if let Some(block) = &mut self.mtp {
                let entries = (taken..taken + update.count).map(|k| record.entry(k));
                block.update_table(stream, update.slot, update.first, entries)?;
            }
            taken += update.count;
        }

        // The copies of the prefix cache, behind the commit of the step before
        // and the MTP block's following of it: the saves, the page copies,
        // the restores.
        self.resume.copies(
            stream,
            step.pages,
            record,
            &mut self.part,
            self.mtp.as_mut(),
            &self.flow,
        )?;
        // The places of the saves inside a pass are locked here, where a step's saves lock
        // theirs: not between the pass's micro-batches (`Resume::hold`).
        for i in 0..record.inner_saves() {
            self.resume.hold(record.inner_save(i).place)?;
        }

        // A step that only drafts: the chains, and nothing of the model. It
        // leaves no window for the next record to commit.
        if record.drafts_only() {
            let tokens = step.record_on_device.offset(record.tokens_offset());
            if let Some(block) = &mut self.mtp {
                let chains = std::mem::take(&mut self.chains);
                let step = MtpStep { tokens, chains, chances: record.draft_chances() };
                block.step(stream, &self.segments, &self.tokens, &self.replay, Some(step), true)?;
            }
            // The DFlash drafter's passes, into the step's one list: the MTP
            // block's when it is on this device too.
            if !self.passes.is_empty() && self.part.dflash().is_some() {
                let table = self
                    .dflash_embedding
                    .as_ref()
                    .ok_or_else(|| invalid("the DFlash drafter is on another part"))?;
                let n = self.part.n_embd() as usize;
                let embed = |token: u32| {
                    let mut out = vec![0f32; n];
                    // A token past the table is zeros, as the model's rows are.
                    let _ = table.decode(token, &mut out);
                    out
                };
                let list = self.mtp.as_ref().map(MtpBlock::list);
                self.part.dflash_pass(stream, tokens, &self.passes, &embed, list)?;
            } else {
                self.part.dflash_no_pass();
            }
            self.windows.clear();
            return Ok(None);
        }
        self.part.dflash_no_pass();

        // The newest cell of every sequence in the MTP block. Its chains run
        // in a step that only drafts: the rows that verify them have their
        // tokens, which every device thread needs.
        if !self.chains.is_empty() {
            return Err(invalid("the MTP block's chains run in a step that only drafts"));
        }
        // (A pass's rows are followed by the block after the pass, micro-batch by
        // micro-batch.)
        if let Some(block) = self.mtp.as_mut().filter(|_| pass.is_none()) {
            block.step(stream, &self.segments, &self.tokens, &self.replay, None, false)?;
        }

        // The rows of the n-gram table, and where the sequences are now (a pass that goes on
        // read its rows when it started).
        if self.drives_ngram && !resumes {
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
        if let Some(key) = pass {
            return self.enqueue_pass(step, key, number as u32);
        }

        // The residual streams start from the embeddings, which the part that
        // embeds stages and gathers itself (`blocks/rows.rs`).
        // The rows of the step's image cells, copied over the gathered ones by
        // the part that embeds.
        self.part.splice_rows(rows::image_splices(record, step.image_stage, self.part.n_embd()));
        let rotary = (!self.rotary.is_empty()).then_some(self.rotary.as_slice());
        let mut host = StepRecord::new(&self.segments, &self.tokens, rotary, &self.out_rows)?;
        for (i, s) in self.segments.iter().enumerate() {
            if self.replay.iter().any(|r| r.contains(&s.row0)) {
                host.mark_replay(i)?;
            }
        }
        let (segments, tokens, replay) = (&self.segments, &self.tokens, &self.replay);

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
                    tokens,
                    ngram: $fetched,
                    inbound: $inbound,
                    xbase,
                    pools: step.pages,
                    replay,
                }
            };
        }
        self.part.begin(&env!(None, inbound))?;
        self.part.dflash_begin(&env!(None, None))?;
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
        self.part.dflash_inject(stream)?;
        if self.mtp.is_some() {
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
        // A wide step's last layers may take long: it stands still only when
        // no layer's route record reaches the store.
        let followed = || flow.followed.iter().map(|n| n.load(Ordering::Acquire)).sum::<usize>();
        let (mut started, mut seen) = (Instant::now(), followed());
        while !end.is_done()? {
            let now = followed();
            if now != seen {
                (started, seen) = (Instant::now(), now);
            }
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
        // The MTP block's chains of a step that only drafts made their drafts
        // on the device; the round after it verifies them, or the first of
        // them, and the rows of the n-gram table its rows need are read now,
        // while the conductor decides.
        if let Some(block) = self.mtp.as_mut().filter(|_| self.only_drafts) {
            let chains = block.chains();
            let next: Vec<(u32, &[u32])> =
                chains.iter().map(|(slot, tokens)| (*slot, tokens.as_slice())).collect();
            // A failed read ahead costs nothing but the read of the step.
            let _ = flow.lock_ngram().expect(&next);
        }
        flow.finish(!self.only_drafts)
    }
}

impl DeviceModel for Qwen4ExpDevice {
    fn logit_tokens(&self) -> DevicePtr {
        self.part.head_tokens()
    }

    fn logits_per_row(&self) -> u32 {
        if self.part.has_head() { self.n_vocab } else { 0 }
    }

    fn tower(&mut self, stream: &Stream, job: &mut TowerJob) -> Result<(), ModelError> {
        match &mut self.tower {
            Some(tower) => tower.run(stream, job),
            None => Err(invalid("this part has no image tower")),
        }
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

    fn model_drafts(&self) -> Option<ModelDrafts> {
        // The step's one list: the MTP block's when the device has it (the
        // DFlash drafter writes its drafts into it), else the DFlash
        // drafter's.
        let mtp = self.mtp.as_ref().map_or(0, MtpBlock::drafted);
        let dflash = self.part.dflash().map_or(0, |d| d.drafted());
        let tokens = mtp.max(dflash);
        let ptr = match (&self.mtp, self.part.dflash()) {
            (Some(block), _) => block.list(),
            (None, Some(d)) => d.own_list(),
            (None, None) => return None,
        };
        (tokens > 0).then_some(ModelDrafts { ptr, tokens })
    }

    fn transfer(
        &mut self,
        cx: &TransferContext<'_>,
        item: &mut DeviceTransfer,
    ) -> Result<(), ModelError> {
        // A page that comes in is in use from now on, before a step's tables
        // name it: the piece of the far pool it lies in is not given back.
        if item.kind == ares_runtime::TransferKind::PageIn {
            self.mark_live(item.object);
        }
        self.resume.transfer(cx.stream, cx.pages, item, &mut self.part, self.mtp.as_mut())
    }

    fn model_draft_ms(&mut self) -> f64 {
        let pass = self.part.dflash_pass_ms();
        self.mtp.as_mut().map_or(0.0, MtpBlock::chain_ms) + pass
    }

    fn model_draft_probs(&mut self, out: &mut Vec<f32>) {
        let Some(drafts) = self.model_drafts() else { return };
        let from = out.len();
        out.resize(from + drafts.tokens as usize, f32::NAN);
        if let Some(block) = &self.mtp {
            block.probs(&mut out[from..]);
        }
        if let Some(d) = self.part.dflash().filter(|d| d.drafted() > 0) {
            for s in d.last_pass() {
                for j in s.place..s.place + s.wanted {
                    out[from + j as usize] = d.prob(j);
                }
            }
        }
    }

    fn paused(&self) -> bool {
        self.paused
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
mod widest_tests {
    use super::*;

    #[test]
    fn the_widest_step_is_a_multiple_of_the_grain_that_fits() {
        let fits = |most: u32| move |rows: u32| Ok::<bool, ()>(rows <= most);
        assert_eq!(widest_rows(512, 8192, fits(9000)), Ok(8192));
        assert_eq!(widest_rows(512, 8192, fits(3700)), Ok(3584));
        assert_eq!(widest_rows(512, 8192, fits(767)), Ok(512));
        assert_eq!(widest_rows(512, 8192, fits(768)), Ok(768));
        assert_eq!(widest_rows(512, 512, fits(0)), Ok(512));
        assert_eq!(widest_rows(512, 1000, fits(999)), Ok(768));
        // A part that keeps scratch for fewer rows than the grain takes its own as the grain.
        assert_eq!(widest_rows(8, 32, fits(20)), Ok(16));
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
