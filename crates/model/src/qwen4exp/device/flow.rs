//! What the threads that drive a step share: the expert store behind a lock,
//! the reader of the n-gram table's rows, and counters that say how far each
//! part's step has got. The runtime's device threads (`runtime.rs`) and the
//! hand-driven [`Session`](super::Session) step the parts through it alike.
//!
//! The store has one owner at a time. It sits behind a lock that no thread
//! waits for in a step: whoever finds it free serves it for all parts (hands
//! over the route records that have arrived, polls reads and copies), and
//! whoever does not goes on. A thread serves it whenever it would otherwise
//! wait: for the lead ([`LEAD_LAYERS`]), for the rows of the n-gram table, for
//! the experts a wide step misses, and for the end of its step.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use ares_api::ExpertStats;
use ares_core::units::Bytes;
use ares_cuda::{Event, Stream};
use ares_runtime::{ModelError, PinnedRam, RecordView};

use super::feed::Feed;
use super::invalid;
use super::misses::{Misses, Progress, SpanPicks, WideStats};
use super::ngram::NgramFeed;
use super::part::{Part, PassSizes, Pieces, StepEnv};
use super::placement::Placement;
use super::runtime::{PassKey, PassLoan};
use crate::blocks::PAGE_ENTRY;
use crate::blocks::far::FarPool;
use crate::qwen4exp::Config;

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

/// How long a failed step serves the store so that the kernels that wait for
/// an expert can end, before the device threads leave it.
const DRAIN: Duration = Duration::from_secs(5);

/// How often a wait of a second or more looks at what the steps moved ([`Flow::check`]).
const LOOK_EVERY: Duration = Duration::from_millis(10);

/// How long the tier of experts locks pieces at a time when it may hold
/// more and no step runs: the thread that watches the machine's memory does
/// it, it must look again soon, and a step that begins waits for it.
const GROW_SLICE: Duration = Duration::from_millis(40);

/// How long the conductor tries before a step to make room in the RAM limit
/// for the K and V beyond VRAM that the step names, and how long each try
/// waits for what is on its way into the tier of experts before the tier
/// gives pieces up.
const FAR_WAIT: Duration = Duration::from_secs(10);
const FAR_SHRINK_WAIT: Duration = Duration::from_millis(200);

/// The expert store, the rows of the n-gram table, and the counters that tell
/// the threads how far each part's step has got.
pub(super) struct Flow {
    pub(super) feed: Mutex<Feed>,
    pub(super) ngram: Mutex<NgramFeed>,
    /// Per part: the layers of the running step that it has enqueued, and the
    /// layers whose route records the store has handled.
    pub(super) enqueued: Vec<AtomicUsize>,
    pub(super) followed: Vec<AtomicUsize>,
    /// The step the store has begun, and how many parts have ended it.
    pub(super) begun: AtomicU64,
    pub(super) finished: AtomicUsize,
    /// Set by a part whose step failed, so that the others give up.
    pub(super) failed: AtomicBool,
    /// Uploads and reads the store had finished the last time anyone looked.
    pub(super) uploads_seen: AtomicU64,
    pub(super) stall: Duration,
    /// For a wait that has stood still a while ([`Flow::check`]): what the steps
    /// had moved when it was last seen to change, when, and when it was last looked at.
    pub(super) moved_at: Mutex<(u64, Instant, Instant)>,
    /// Reads of the n-gram table's rows done the last time anyone looked.
    pub(super) ngram_seen: AtomicU64,
    /// What the store counted, summed over the times it was read.
    pub(super) totals: Mutex<ExpertStats>,
    /// What the wide steps found and waited for, since the engine started.
    pub(super) wide: Mutex<WideStats>,
    /// The most rows of a step on the parts' own scratch; a wider step runs on
    /// VRAM the expert slots lend: per part its bytes and where they lie.
    pub(super) own_rows: u32,
    pub(super) lent: Vec<Vec<u64>>,
    pub(super) lent_at: Vec<Option<Pieces>>,
    /// Whether the running step runs on the loan.
    pub(super) on_loan: AtomicBool,
    /// Prompts read layer by layer (`runtime/pass.rs`): the most rows of one (0: none), what
    /// each part keeps of a row, the picks of a row, the loan of the pass that runs or ran
    /// last, and whether the step being run is a pass.
    pub(super) pass_rows: u32,
    pub(super) pass_sizes: Vec<PassSizes>,
    pub(super) top_k: u32,
    pub(super) pass: Mutex<Option<PassLoan>>,
    pub(super) pass_completed: AtomicBool,
    /// In a step of a pass: a part paused it; and by layers, the residuals of micro-batches
    /// the device before has handed over and the next has released.
    pub(super) pass_paused: AtomicBool,
    pub(super) pass_sent: AtomicUsize,
    pub(super) pass_released: AtomicUsize,
    /// Both devices on every layer: the pieces of the pass the mixer has let run, and the
    /// picks its router made, which the worker takes.
    pub(super) pass_go: AtomicU64,
    pub(super) pass_picks: Mutex<Option<(u64, Arc<SpanPicks>)>>,
    /// The passes that took a loan so far: each is numbered by it.
    pub(super) pass_count: AtomicU64,
    /// The far pools of the parts (`blocks/far.rs`), the far slots their
    /// pieces cover now, and the engine's RAM, which books them.
    pub(super) far: Vec<Arc<Mutex<FarPool>>>,
    pub(super) far_live: Mutex<Vec<bool>>,
    /// Per pool page: the tables name it or a transfer brought it in, and the
    /// pool has not freed it since (kept for the far pools only).
    pub(super) pages_live: Mutex<Vec<bool>>,
    /// The number of the last step [`Flow::before_step`] prepared: a step that
    /// begins without it would read far pages whose pieces nobody locked.
    pub(super) prepared: AtomicU64,
    pub(super) ram: PinnedRam,
}

impl Flow {
    /// The flow of `parts` parts over `feed` and `ngram`: no step runs on VRAM the
    /// expert slots lend and none is read layer by layer (the runtime says where its
    /// loans lie), and no K or V lies beyond VRAM. A step that stands still for `stall`
    /// is given up.
    pub(super) fn new(
        feed: Feed,
        ngram: NgramFeed,
        parts: usize,
        stall: Duration,
        own_rows: u32,
        top_k: u32,
        ram: PinnedRam,
    ) -> Self {
        let ram_bytes = feed.store.ram_bytes().0;
        Self {
            feed: Mutex::new(feed),
            ngram: Mutex::new(ngram),
            enqueued: (0..parts).map(|_| AtomicUsize::new(0)).collect(),
            followed: (0..parts).map(|_| AtomicUsize::new(0)).collect(),
            begun: AtomicU64::new(0),
            finished: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
            uploads_seen: AtomicU64::new(0),
            stall,
            moved_at: Mutex::new((0, Instant::now(), Instant::now())),
            ngram_seen: AtomicU64::new(0),
            totals: Mutex::new(ExpertStats { ram_tier_bytes: ram_bytes, ..Default::default() }),
            wide: Mutex::new(WideStats::default()),
            own_rows,
            lent: vec![Vec::new(); parts],
            lent_at: vec![None; parts],
            on_loan: AtomicBool::new(false),
            pass_rows: 0,
            pass_sizes: Vec::new(),
            top_k,
            pass: Mutex::new(None),
            pass_completed: AtomicBool::new(false),
            pass_paused: AtomicBool::new(false),
            pass_sent: AtomicUsize::new(0),
            pass_released: AtomicUsize::new(0),
            pass_go: AtomicU64::new(0),
            pass_picks: Mutex::new(None),
            pass_count: AtomicU64::new(0),
            far: Vec::new(),
            far_live: Mutex::new(Vec::new()),
            pages_live: Mutex::new(Vec::new()),
            prepared: AtomicU64::new(0),
            ram,
        }
    }

    pub(super) fn parts(&self) -> usize {
        self.enqueued.len()
    }

    /// Adds what the wide steps found since the last call to the totals. The experts of a
    /// layer that a wide step found in a slot are hits and the others are picks the store
    /// counts when it brings them (as misses, never as hits).
    pub(super) fn fold_wide(&self) {
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

    pub(super) fn fail(&self) {
        self.failed.store(true, Ordering::SeqCst);
    }

    pub(super) fn lock_feed(&self) -> MutexGuard<'_, Feed> {
        self.feed.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The tier of experts in RAM follows `allowance`, the bytes it may hold.
    /// Called by the thread that watches the machine's memory. A running
    /// step gives pieces up when it ends ([`Flow::finish`]), and the conductor
    /// takes one before each step ([`Flow::before_step`]); when no step runs,
    /// this gives them up and locks pieces for a while, and no step begins
    /// meanwhile: the driver's unlock waits for the device, and a step's
    /// kernels can wait for the store, which this thread holds. What it gives
    /// up while a step runs is unlocked when the step ends
    /// ([`ares_cuda::gate::retire`]).
    pub(super) fn follow_ram(&self, ram: &PinnedRam, allowance: u64) {
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
            // Only while no step runs, without waiting for one: a step that
            // begins would wait for the store, which this thread holds.
            if !matches!(ares_cuda::gate::try_change(|| feed.store.grow_ram(1)), Some(Ok(1))) {
                break;
            }
        }
    }

    /// The step of `record` is about to be handed to the devices (on the
    /// conductor's thread, while no step runs): the far pools lock the
    /// pieces of the far pages the step's tables name and give back the
    /// others, and the tier of experts takes a piece more when it may hold
    /// more (a lock takes tens of milliseconds, so a piece a step, and none
    /// before a step that only drafts). Page-locked memory changes here and
    /// not inside the step (see "Host memory" in `docs/dev/architecture/host-memory.md`).
    pub(super) fn before_step(&self, record: &RecordView<'_>) -> Result<(), ModelError> {
        let mut feed = self.lock_feed();
        if let Some(near) =
            self.far.first().map(|pool| pool.lock().unwrap_or_else(|e| e.into_inner()).near())
        {
            let live = {
                let mut pages = self.pages_live.lock().unwrap_or_else(|e| e.into_inner());
                for i in 0..record.freed() {
                    if let Some(live) = pages.get_mut(record.freed_page(i) as usize) {
                        *live = false;
                    }
                }
                for k in 0..record.entries() {
                    mark_live(&mut pages, record.entry(k) & !PAGE_ENTRY);
                }
                pages.get(near as usize..).unwrap_or(&[]).to_vec()
            };
            self.lock_far(&mut feed, &live)?;
        }
        // The number `begin` knows the step by (see `Qwen4ExpDevice::enqueue_step`).
        self.prepared.store(u64::from(record.step()) + 1, Ordering::Release);
        if !record.drafts_only() {
            feed.store.grow_ram(1).map_err(|e| invalid(e.to_string()))?;
        }
        Ok(())
    }

    /// Pool page `page` came in by a transfer: it is in use from now on,
    /// before a step's tables name it, and the piece of the far pool it lies
    /// in is not given back.
    pub(super) fn page_came_in(&self, page: u32) {
        if !self.far.is_empty() {
            mark_live(&mut self.pages_live.lock().unwrap_or_else(|e| e.into_inner()), page);
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
    pub(super) fn begin(
        &self,
        step: u64,
        rows: u32,
        windows: Option<StepWindows<'_>>,
        index: usize,
        stream: &Stream,
        pass: Option<&PassKey>,
    ) -> Result<(), ModelError> {
        let mut feed = self.lock_feed();
        if self.begun.load(Ordering::Acquire) != step {
            // The K and V beyond VRAM that the step names were locked before
            // the step was handed over ([`Flow::before_step`]); without that
            // the devices would reach pages that nobody locked.
            if !self.far.is_empty() && self.prepared.load(Ordering::Acquire) != step {
                return Err(invalid(format!(
                    "step {step} begins with K and V beyond VRAM that were not locked for it: \
                     `Model::before_step` was not called (a wrapper of the model that does not \
                     pass it on?)"
                )));
            }
            for counter in self.enqueued.iter().chain(&self.followed) {
                counter.store(0, Ordering::Release);
            }
            self.finished.store(0, Ordering::Release);
            // A prompt read layer by layer keeps its rows on a loan of its own; the loan of
            // one before goes back first, or serves it.
            if windows.is_some() {
                let far_live = self.far_live.lock().unwrap_or_else(|e| e.into_inner());
                let far_reach = far_live.iter().rposition(|l| *l).map_or(0, |s| s as u32 + 1);
                drop(far_live);
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
                feed.begin_step(rows)?;
                feed.store.set_windows(windows, worth);
            }
            self.begun.store(step, Ordering::Release);
        }
        feed.store.device_mut(index).fence(stream).map_err(|e| invalid(e.to_string()))
    }

    /// Locks the pieces of every far pool that hold far pages in use (`live`,
    /// a flag per far slot) and gives back the others, between steps (from
    /// [`Flow::before_step`]), with the store not in a step. The tier of
    /// experts gives up what the RAM limit has no room for: a running
    /// sequence's K and V rank above it (see "Host memory" in
    /// `docs/dev/architecture/host-memory.md`).
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
    pub(super) fn lend(&self, feed: &mut Feed) -> Result<(), ModelError> {
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
    pub(super) fn lent(&self, index: usize) -> Option<Pieces> {
        let lent = self.lent_at.get(index).cloned().flatten()?;
        self.lock_feed().store.has_scratch(index).then_some(lent)
    }

    pub(super) fn enqueued_layers(&self) -> Vec<usize> {
        self.enqueued.iter().map(|n| n.load(Ordering::Acquire)).collect()
    }

    /// Hands the route records that have arrived to the store and lets it and
    /// the reads of table rows work, if no other thread is doing that now.
    /// Never waits.
    pub(super) fn pump(&self) -> Result<(), ModelError> {
        match self.feed.try_lock() {
            Ok(mut feed) => {
                let enqueued = self.enqueued_layers();
                feed.pump(&enqueued)?;
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
    pub(super) fn progress(&self) -> u64 {
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

    /// An error when another part failed, or when the step has stood still for
    /// longer than a step may: since the wait began at `started`, and since
    /// anything of the steps last moved ([`Flow::moved`]). What moved is looked at
    /// only in a wait of a second or more; a wait that holds the store (the
    /// lend waits) sees no read or upload end and counts from its start.
    pub(super) fn check(&self, started: Instant, what: &str) -> Result<(), ModelError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(ModelError::Aborted);
        }
        let waited = started.elapsed();
        if waited < self.stall.min(Duration::from_secs(1)) {
            return Ok(());
        }
        // What moved is looked at every few milliseconds at most, by one waiting thread at a
        // time: a look takes the store's lock, which the pumps of the waits need.
        let mut at = match self.moved_at.try_lock() {
            Ok(at) => at,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        let now = Instant::now();
        let still = if now.saturating_duration_since(at.2) >= LOOK_EVERY {
            at.2 = now;
            let mut mark = (at.0, at.1);
            let still = stood_still(waited, self.moved(), &mut mark, now);
            (at.0, at.1) = mark;
            still
        } else {
            waited.min(now.saturating_duration_since(at.1))
        };
        drop(at);
        if still < self.stall {
            return Ok(());
        }
        self.fail();
        let enqueued = self.enqueued_layers();
        let state = match self.feed.try_lock() {
            Ok(feed) => feed.standing_still(&enqueued),
            Err(_) => "the expert store is busy".to_owned(),
        };
        Err(invalid(format!("a step stood still for {still:.0?} while {what}: {state}")))
    }

    /// A number that changes whenever anything of the steps moves: what
    /// [`Flow::progress`] counts, a pass's rows handed on, taken or let run, and
    /// reads of the n-gram table's rows that ended.
    fn moved(&self) -> u64 {
        let ngram = match self.ngram.try_lock() {
            Ok(ngram) => {
                self.ngram_seen.store(ngram.completed(), Ordering::Relaxed);
                ngram.completed()
            }
            Err(_) => self.ngram_seen.load(Ordering::Relaxed),
        };
        let pass =
            self.pass_sent.load(Ordering::Acquire) + self.pass_released.load(Ordering::Acquire);
        self.progress()
            .wrapping_add(pass as u64)
            .wrapping_add(self.pass_go.load(Ordering::Acquire))
            .wrapping_add(ngram)
    }

    /// Serves the store until `part` has no layer enqueued more than the lead
    /// ahead of the records it has handled.
    pub(super) fn throttle(&self, part: usize) -> Result<(), ModelError> {
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
    pub(super) fn await_rows(&self) -> Result<Vec<u8>, ModelError> {
        let started = Instant::now();
        loop {
            {
                let mut ngram = self.lock_ngram();
                ngram.poll();
                if ngram.is_done() {
                    return Ok(ngram.rows()?.to_vec());
                }
            }
            self.check(started, "rows of the n-gram table were read")?;
            self.pump()?;
            std::hint::spin_loop();
        }
    }

    /// Serves the store for a while so that kernels that wait for an expert
    /// can end: after a step failed, before its buffers are freed.
    pub(super) fn drain(&self, idle: impl Fn() -> bool) {
        let started = Instant::now();
        while !idle() && started.elapsed() < DRAIN {
            if self.pump().is_err() {
                break;
            }
            std::hint::spin_loop();
        }
    }

    /// Nothing runs and nobody waits ([`Model::idle`](ares_runtime::Model::idle)):
    /// VRAM lent for wide steps and for a pass that has read its rows goes back
    /// now, so that the tiers have their slots while the engine idles, not at
    /// the next step. No step follows to read what the last one left there.
    /// A pass that has not read all its rows keeps its loan.
    pub(super) fn idle(&self) {
        let mut feed = self.lock_feed();
        if feed.store.in_step() {
            return;
        }
        if !self.forget_done_pass() {
            return;
        }
        self.on_loan.store(false, Ordering::Release);
        for device in 0..self.lent.len() {
            // What fails here (the fill from RAM into the slots given back)
            // fails the next step, which gives back again.
            if feed.store.return_scratch(device).is_err() {
                return;
            }
        }
    }

    /// A part has ended its step on the device: the last of them ends it for
    /// the store, unless it only drafted (`store` false).
    pub(super) fn finish(&self, store: bool) -> Result<(), ModelError> {
        if self.finished.fetch_add(1, Ordering::AcqRel) + 1 == self.parts() {
            let mut feed = self.lock_feed();
            if store {
                feed.end_step()?;
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

    /// Part `index` has enqueued `layers` layers of the running step: the other part of a
    /// shared layer may read its route record from now on.
    pub(super) fn mark_enqueued(&self, index: usize, layers: usize) {
        self.enqueued[index].store(layers, Ordering::Release);
    }

    /// Enqueues layer `i` of `part` (the `index`-th) once the part is no more than
    /// [`LEAD_LAYERS`] ahead of the route records the store has handled.
    pub(super) fn enqueue_layer(
        &self,
        part: &mut Part,
        index: usize,
        env: &StepEnv<'_>,
        i: usize,
    ) -> Result<(), ModelError> {
        self.throttle(index)?;
        part.enqueue_layer(env, i)?;
        self.mark_enqueued(index, i + 1);
        Ok(())
    }

    /// Serves the store until `end` has happened on its device. A wide step's last layers
    /// may take long: it stands still only when no layer's route record reaches the store.
    pub(super) fn wait_for(&self, end: &Event, what: &str) -> Result<(), ModelError> {
        let followed = || self.followed.iter().map(|n| n.load(Ordering::Acquire)).sum::<usize>();
        let (mut started, mut seen) = (Instant::now(), followed());
        while !end.is_done()? {
            let now = followed();
            if now != seen {
                (started, seen) = (Instant::now(), now);
            }
            self.check(started, what)?;
            self.pump()?;
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Serves the store until every route record part `index` has enqueued is with it.
    pub(super) fn follow(&self, index: usize) -> Result<(), ModelError> {
        let started = Instant::now();
        while self.followed[index].load(Ordering::Acquire)
            < self.enqueued[index].load(Ordering::Acquire)
        {
            self.check(started, "route records were on their way")?;
            self.pump()?;
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Serves the store until each of `misses`, one for each of `turns`, has brought and
    /// computed what its part misses of `layer`.
    pub(super) fn complete(
        &self,
        layer: usize,
        turns: &mut [Turn<'_, '_>],
        misses: &mut [Misses],
        what: &str,
    ) -> Result<(), ModelError> {
        let (mut started, mut seen) = (Instant::now(), self.progress());
        loop {
            let mut done = true;
            for (misses, turn) in misses.iter_mut().zip(turns.iter_mut()) {
                // A part that does not route waits for the picks of the part that does,
                // which have reached the host once that part has enqueued the layer; the
                // picks of a pass come with the pass.
                let routed = turn.pass
                    || turn
                        .routing
                        .mixer
                        .is_none_or(|mixer| self.enqueued[mixer].load(Ordering::Acquire) > layer);
                let routing = turn.routing;
                let progress = misses.advance(
                    turn.part,
                    turn.env,
                    &mut self.lock_feed(),
                    routed,
                    &|expert| routing.owns(layer, expert),
                )?;
                done &= progress == Progress::Done;
            }
            if done {
                return Ok(());
            }
            let progress = self.progress();
            if progress != seen {
                (started, seen) = (Instant::now(), progress);
            }
            self.check(started, what)?;
            self.pump()?;
            std::hint::spin_loop();
        }
    }
}

/// How long a wait that began `waited` ago has stood still: since it began or since
/// the steps last moved, whichever is later. `at` is what they had `moved` when that
/// was last seen to change, and when; it takes `moved` at `now` when that differs.
fn stood_still(waited: Duration, moved: u64, at: &mut (u64, Instant), now: Instant) -> Duration {
    if at.0 != moved {
        *at = (moved, now);
    }
    waited.min(now.saturating_duration_since(at.1))
}

/// Marks pool page `page` in use in `pages`, a flag per page.
fn mark_live(pages: &mut Vec<bool>, page: u32) {
    if pages.len() <= page as usize {
        pages.resize(page as usize + 1, false);
    }
    pages[page as usize] = true;
}

/// The windows of a step that verify drafts, as (first row, rows), and what
/// every row of the step is worth in milliseconds (the record's row worth).
type StepWindows<'a> = (&'a [(u32, u32)], Option<&'a [f32]>);

/// One part's share of a layer whose experts it computes in rounds: the part (the
/// `index`-th), the step as it sees it, who routes for it, and whether the layer is one of a
/// pass (whose picks come with the pass).
pub(super) struct Turn<'a, 'e> {
    pub(super) part: &'a mut Part,
    pub(super) index: usize,
    pub(super) env: &'a StepEnv<'e>,
    pub(super) routing: &'a Routing,
    pub(super) pass: bool,
}

/// The experts of layer `layer` of every part of `turns` in a step of more than eight rows:
/// the first round of each is enqueued with the experts that have a slot, then the store is
/// served until the misses are brought and computed (`Misses`). The parts of a layer that
/// two devices share are given together by a thread that drives both.
pub(super) fn wide_layer(
    flow: &Flow,
    layer: usize,
    turns: &mut [Turn<'_, '_>],
) -> Result<(), ModelError> {
    let mut misses = Vec::with_capacity(turns.len());
    for turn in turns.iter_mut() {
        let source = turn.routing.mixer.map(|mixer| flow.lock_feed().board(mixer));
        let routing = turn.routing;
        let feed = flow.lock_feed();
        misses.push(Misses::begin(
            turn.part,
            turn.env,
            &feed,
            turn.index,
            layer,
            source,
            &|expert| routing.owns(layer, expert),
        )?);
    }
    flow.complete(layer, turns, &mut misses, "the experts of a wide step were brought")
}

/// Who routes for a part and which experts it computes, when both devices work on every
/// layer.
#[derive(Debug, Default)]
pub(super) struct Routing {
    /// The device that routes, for a part that does not.
    pub(super) mixer: Option<usize>,
    /// Per layer and expert, whether this part computes the expert; `None` for all.
    owned: Option<Vec<Vec<bool>>>,
}

impl Routing {
    /// What part `index` of `placement` routes and computes.
    pub(super) fn of(placement: &Placement, c: &Config, index: usize) -> Self {
        let Some(tensor) = &placement.tensor else { return Self::default() };
        Self {
            mixer: Some(tensor.mixer).filter(|m| *m != index),
            owned: Some(
                (0..c.n_layer)
                    .map(|layer| {
                        (0..c.experts.count as u32)
                            .map(|expert| placement.owner(layer, expert) == index)
                            .collect()
                    })
                    .collect(),
            ),
        }
    }

    /// Whether the part computes `expert` of `layer`.
    pub(super) fn owns(&self, layer: usize, expert: u32) -> bool {
        self.owned.as_ref().is_none_or(|o| o[layer][expert as usize])
    }

    /// Whether both devices work on every layer.
    pub(super) fn shared(&self) -> bool {
        self.owned.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wait_stands_still_from_its_start_or_the_last_move_whichever_is_later() {
        let s = Duration::from_secs;
        let t0 = Instant::now();
        // Nothing moved since long before the wait: it counts from its start.
        let mut at = (7, t0);
        assert_eq!(stood_still(s(31), 7, &mut at, t0 + s(100)), s(31));
        // Something moved: it counts from when that was seen.
        assert_eq!(stood_still(s(32), 8, &mut at, t0 + s(101)), s(0));
        assert_eq!(at, (8, t0 + s(101)));
        assert_eq!(stood_still(s(50), 8, &mut at, t0 + s(119)), s(18));
        assert_eq!(stood_still(s(70), 8, &mut at, t0 + s(139)), s(38));
    }
}
