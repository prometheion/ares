//! The expert store: the way of an expert of a mixture-of-experts model from
//! the file over RAM into a VRAM slot, and what happens when a step needs an
//! expert that is not in one: it is streamed to the device or computed on
//! the CPU, whichever is sooner.
//!
//! # Tiers
//!
//! - **VRAM.** Each device has pools of slots, one pool per slot size, and
//!   each layer on the device has a number of them ([`ares_weights::SlotSet`],
//!   divided by [`ares_weights::split_slots`]). The device computes an
//!   expert from its slot, which the kernels find through the layer's slot
//!   table on the device (`ares_cuda::ExpertKernels`).
//! - **RAM.** One region of addresses, cut into slots the same way and
//!   page-locked in pieces. Reads from the file land here, in place and in
//!   the file's bytes (a read covers whole blocks, so a slot is the expert
//!   plus at most two blocks per matrix), uploads go from here, and the CPU
//!   computes from here ([`cpu`]). The tier holds what VRAM does not: an
//!   expert that VRAM holds is worth nothing here and gives way first, and an
//!   expert that VRAM evicts comes down into it. The tier follows the memory
//!   the machine has: it gives pieces back to the system, the coldest experts
//!   of every layer with them, and takes them again (`elastic.rs`).
//! - **SSD.** The file itself, read through `ares-io`: demand reads for
//!   experts a running step waits for, prefetches for guesses.
//! - **Transit and staging** (`transit.rs`). A step of many rows uses far
//!   more experts of a layer than the layer has slots, each for all its rows
//!   at once. What such a step misses is uploaded into a small pool of
//!   transit slots per device, from its RAM slot or from the file through
//!   staging slots in host memory, computed from there and released; it takes
//!   a slot of the tiers afterwards only if its heat says so.
//!
//! # A step
//!
//! The step is enqueued whole and never waits for the host. The router of a
//! layer writes a route record that reaches the host on a side stream, with
//! the layer's input rows ([`ExpertStore::routed_rows`]). For every pick
//! that has no slot, the kernels of that expert wait on the device, and the
//! store serves it one of two ways, by what its measured costs say
//! (`hybrid.rs`): it uploads the expert (from RAM, or from the file into RAM
//! first) and writes its table entry, or it claims the expert for the CPU,
//! whose workers compute it from the bytes in RAM, and delivers the sum of
//! the claimed experts' outputs for each row, which the device adds where
//! it would have added theirs. An upload becomes a member of the VRAM tier
//! only when it takes a free slot or beats the coldest member by the rule of
//! a promotion; otherwise it goes up on loan, into a slot that the next such
//! upload takes again ([`ares_weights::SlotSet::lend`]). What VRAM holds is
//! decided apart from that, between steps, from the heat, and never makes a
//! step wait. The routers of later layers applied on the device name the
//! experts of a layer before it runs ([`ExpertStore::ahead`]): only what no
//! tier holds is read, within bounds that keep the drive for the reads a step
//! waits for (`ahead.rs`). The tiers are filled between steps by a
//! [`TierFill`], from the heat ([`ExpertStore::plan_fill`]) or from a list
//! ([`ExpertStore::plan_listed_fill`]: a recorded routing, every expert, the
//! MTP block's).
//!
//! What keeps this safe without a lock or a host wait is the order of the
//! table: an entry appears on the device after the expert's bytes, and an
//! entry is removed, and its slot reused, only when no kernel can be reading
//! it. That is the case between steps, for the layers the running step has
//! passed, and, once a layer's record has arrived, for the experts of that
//! layer the router did not pick. For a layer the step has not reached, the
//! store only fills slots that are free; it keeps a few free per layer
//! ([`StoreConfig::vram_reserve`]) and frees them again between steps.
//!
//! One thread owns the store and calls it; the store never blocks. Reads run
//! on the I/O thread of `ares-io`, copies on the device.
//!
//! # Between steps
//!
//! The heat can outlive the store: [`ExpertProfile`] is a small file of it,
//! made for one model file (`profile.rs`). From the heat, a seeded one or the
//! store's own, [`ExpertStore::plan_fill`] and [`ExpertStore::advance_fill`]
//! fill what is empty in the tiers while no step runs (`fill.rs`): at the
//! start in one sweep through the file, when idle the hottest first and a few
//! at a time, stopped by the next step.

mod ahead;
pub mod cpu;
mod device;
mod elastic;
mod fill;
mod hybrid;
mod near;
mod profile;
mod region;
mod scratch;
mod tier;
mod transit;
mod windows;

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use ares_core::LedgerError;
use ares_io::{Buffer, Class, Completion, ExpertMap, Read, Reader};
use ares_weights::{
    ExpertHeat, GuessQueue, LayerNeed, SlotSet, SplitRule, Start, Swap, Want, split_slots,
};

pub use device::{CudaSlots, SlotDevice};
use elastic::{Layout, Pieces};
pub use fill::{FillOrder, FillPace, FillStats, FillTier, TierFill};
pub use hybrid::{CostModel, ExpertTypes, Miss, Serve, WideCpu, WidePick};
pub use near::{ExpertNeighbours, NEAR_FORMAT, Near};
pub use profile::{ExpertProfile, Ignored, PROFILE_FORMAT, ProfileKey};
pub use region::{HostRegion, Locker, PIECE, Released};
pub use scratch::ScratchLoan;
pub use tier::Pool;
use tier::Tier;
pub use transit::Named;
use transit::{Transit, TransitPlan};

/// Why the store could not be built or could not go on.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Io(#[from] ares_io::Error),
    #[error(transparent)]
    Cuda(#[from] ares_cuda::Error),
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    #[error("expert store: {0}")]
    Plan(String),
}

/// The key at the reader of part `part` (a matrix) of read `read`: an expert
/// is read in three parts.
fn part_key(read: u64, part: usize) -> u64 {
    read * 4 + part as u64
}

/// The read that the reader's key `key` is a part of ([`part_key`]).
fn read_of(key: u64) -> u64 {
    key / 4
}

/// Where a layer's experts are computed and how an expert lies in a slot
/// there (`ares_cuda::ExpertShape` gives the numbers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSpec {
    /// The device that runs the layer.
    pub device: usize,
    /// Where each of the expert's matrices starts in a slot, in the order of
    /// the tensors of the [`ExpertMap`].
    pub offsets: [u64; 3],
    /// Bytes from one slot to the next.
    pub slot_bytes: u64,
    /// Every expert of the layer has a slot in VRAM for as long as the store
    /// lives, outside the tiers: the experts of a drafter, which make every
    /// link of a chain and must never be waited for. Their bytes come out of
    /// the device's budget before the tiers get theirs.
    pub resident: bool,
}

/// The budgets and rules of a store.
#[derive(Clone, Debug)]
pub struct StoreConfig {
    /// The most rows a decode step has, and the experts a row picks: a layer
    /// gets at least the slots one step can need.
    pub max_rows: u32,
    pub top_k: u32,
    /// Bytes of VRAM for expert slots, per device.
    pub vram_bytes: Vec<u64>,
    /// Bytes of page-locked RAM for the tier in host memory.
    pub ram_bytes: u64,
    /// Free slots kept per layer in VRAM for experts that arrive before
    /// their layer runs.
    pub vram_reserve: usize,
    /// The most expert reads in flight for guesses; demand reads always start.
    pub read_lanes: usize,
    /// Heat is multiplied by `heat_decay` every `decay_every` steps.
    pub heat_decay: f32,
    pub decay_every: u64,
    /// Every this many steps, slots pass between the layers of a pool by
    /// need; 0 never moves any.
    pub rebalance_every: u64,
    /// Transit slots per device (`transit.rs`), taken from the device's bytes
    /// for slots; a device that is not named has none.
    pub transit_slots: Vec<u32>,
    /// Staging slots in host memory for experts on their way from the file
    /// into transit, taken from the bytes of the RAM tier.
    pub staging_slots: u32,
    /// The bytes of a piece of the RAM tier, about: what the tier locks or
    /// gives back to the system at a time. A piece is whole slots of one
    /// pool.
    pub ram_piece: u64,
    /// An expert that rows verifying drafts left without a read
    /// ([`ExpertStore::set_windows`]) is read into RAM as a guess, for the
    /// steps after: the token of a left row is most often the next step's,
    /// whose first row needs the same experts. Without the read that row
    /// waits for them in the next step, alone in its layer.
    pub read_left: bool,
    /// Whether the CPU computes the experts a step needs that VRAM cannot take (any
    /// [`Serve`] but [`Serve::Stream`]). Then a layer may have fewer slots in VRAM than a
    /// decode step can pick there, when the device's bytes do not hold that many: the slots
    /// of RAM hold the rest for the CPU, and the store never uploads into a slot whose
    /// expert the running step picked. Without it every layer has a step's picks in VRAM.
    pub cpu_serves: bool,
}

impl StoreConfig {
    /// The rules the engine uses, for steps of up to `max_rows` rows of
    /// `top_k` experts and the given budgets.
    pub fn new(max_rows: u32, top_k: u32, vram_bytes: Vec<u64>, ram_bytes: u64) -> Self {
        Self {
            max_rows,
            top_k,
            vram_bytes,
            ram_bytes,
            vram_reserve: 8,
            read_lanes: 8,
            heat_decay: 0.9,
            decay_every: 32,
            rebalance_every: 256,
            transit_slots: Vec::new(),
            staging_slots: 0,
            ram_piece: PIECE as u64,
            read_left: true,
            cpu_serves: false,
        }
    }

    /// The slots a layer needs in VRAM for steps of `rows` rows: what they
    /// can pick there and the free slots for experts that arrive before the
    /// layer runs.
    pub fn slots_for(&self, rows: u32) -> usize {
        (rows * self.top_k) as usize + self.vram_reserve
    }

    /// The fewest slots a layer keeps in VRAM: a decode step's, or where the
    /// CPU computes what VRAM cannot take ([`StoreConfig::cpu_serves`]) a
    /// row's.
    pub fn vram_floor(&self) -> usize {
        self.slots_for(if self.cpu_serves { 1 } else { self.max_rows })
    }
}

/// Where a layer's matrices are in a slot of each tier.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    device: usize,
    /// The pool of the layer among its device's pools.
    device_pool: usize,
    offsets: [u64; 3],
    lens: [usize; 3],
    /// Where each matrix's read buffer starts in a RAM slot.
    ram_bases: [usize; 3],
    /// Every expert has a VRAM slot for good ([`LayerSpec::resident`]).
    resident: bool,
}

/// The slots a store will have: what [`StorePlan::new`] works out, for
/// the caller to allocate the devices' pools and the RAM region from.
#[derive(Debug)]
pub struct StorePlan {
    config: StoreConfig,
    n_experts: usize,
    block: usize,
    layers: Vec<Geometry>,
    vram: Tier,
    ram: Tier,
    transit: TransitPlan,
    /// Where the RAM tier's slots and staging lie in the region.
    layout: Layout,
}

impl StorePlan {
    /// The pools of `device` as `(slot bytes, slots)`, in the order a
    /// [`SlotDevice`] numbers them: the pools of the tier, then the transit
    /// pool if the device has one.
    pub fn device_pools(&self, device: usize) -> Vec<(u64, u32)> {
        let tier = self.vram.pools.iter().filter(|p| p.place == device);
        let transit = self.transit.pools.get(device).copied().flatten();
        tier.map(|p| (p.stride, p.slots))
            .chain(transit.map(|(_, stride, slots)| (stride, slots)))
            .collect()
    }

    /// Bytes of the expert slots of the tier on `device`.
    pub fn vram_bytes(&self, device: usize) -> u64 {
        self.vram.bytes(device)
    }

    /// Bytes of the transit slots on `device`.
    pub fn transit_bytes(&self, device: usize) -> u64 {
        self.transit.bytes(device)
    }

    /// Transit slots on `device` and staging slots in host memory.
    pub fn transit_slots(&self, device: usize) -> (u32, u32) {
        let slots = self.transit.pools.get(device).copied().flatten().map_or(0, |pool| pool.2);
        (slots, self.transit.stage_slots)
    }

    /// Bytes of the region the RAM tier and staging need, aligned to
    /// [`StorePlan::block`]: their slots, in pieces that start on pages.
    pub fn ram_bytes(&self) -> usize {
        self.layout.len
    }

    /// Bytes from one slot of `layer` to the next in the RAM tier: the expert
    /// and the blocks its reads cover.
    pub fn ram_slot_bytes(&self, layer: usize) -> u64 {
        self.ram.pools[self.ram.pool_of[layer]].stride
    }

    /// The alignment reads need of the region.
    pub fn block(&self) -> usize {
        self.block
    }

    /// The slots `layer` starts with in VRAM and in RAM.
    pub fn slots(&self, layer: usize) -> (usize, usize) {
        (self.vram.slots(layer), self.ram.slots(layer))
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }
}

/// The RAM slots of a resident layer: what its experts pass through from the
/// file into VRAM when the store is filled.
const RESIDENT_PASSAGE: usize = 8;

/// Where a missing expert was when its layer's record arrived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// In RAM: an upload away.
    Ram,
    /// On its way already, as a guess: a read or an upload was running.
    OnItsWay,
    /// Only in the file.
    Disk,
}

#[derive(Clone, Copy, Debug)]
struct Wanted {
    layer: usize,
    expert: u32,
    since: Instant,
    source: Source,
    /// The CPU computes it once its bytes are in RAM; otherwise it is
    /// uploaded.
    cpu: bool,
    /// Uploaded on loan ([`ares_weights::SlotSet::lend`]): for the step that
    /// missed it, into the slot of a member that is worth nothing or the
    /// coldest one, without being worth a member's place itself.
    loan: bool,
}

#[derive(Debug)]
struct ReadOp {
    layer: usize,
    expert: u32,
    swap: Swap,
    parts_left: u8,
    failed: bool,
    demand: bool,
    /// Started ahead of a router for a predicted pick ([`ExpertStore::ahead`]).
    ahead: bool,
    since: Instant,
}

#[derive(Clone, Copy, Debug)]
enum DeviceOp {
    /// RAM to VRAM, the table entry behind the bytes: commit the slot,
    /// release the RAM copy.
    Upload { layer: usize, swap: Swap, since: Instant },
    /// RAM to VRAM, the bytes only: the entry is written when no step is at
    /// the layer (`hybrid.rs`).
    Promotion { layer: usize, swap: Swap, since: Instant },
    /// VRAM to RAM: commit the RAM slot.
    Demote { layer: usize, swap: Swap },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// The running step has not routed the layer yet.
    Ahead,
    /// Its record has arrived; its kernels may be running.
    Open,
    /// The step is past it, or no step runs.
    Passed,
}

/// What the store did, since the last [`ExpertStore::take_stats`].
#[derive(Clone, Debug, Default)]
pub struct StoreStats {
    pub steps: u64,
    /// Experts the routers picked, and how many of them were in a slot when
    /// the record arrived.
    pub picks: u64,
    pub hits: u64,
    /// The misses, by where the expert was.
    pub miss_from_ram: u64,
    pub miss_from_disk: u64,
    pub miss_on_its_way: u64,
    /// Reads of experts from the file, and their bytes (whole blocks).
    pub demand_reads: u64,
    pub guess_reads: u64,
    pub demand_read_bytes: u64,
    pub guess_read_bytes: u64,
    pub failed_reads: u64,
    /// Experts copied into VRAM slots.
    pub uploads: u64,
    pub upload_bytes: u64,
    /// Experts copied from a VRAM slot back to RAM when they were evicted.
    pub demotions: u64,
    pub demote_bytes: u64,
    pub evictions: u64,
    /// Slots that passed from one layer to another.
    pub moved_slots: u64,
    /// For every miss: milliseconds from the arrival of the record to the
    /// expert being usable in its slot.
    pub miss_ms: Vec<f32>,
    /// The same for the misses that had to be read from the file.
    pub disk_miss_ms: Vec<f32>,
    /// And for those that were in RAM: the time of an upload.
    pub ram_miss_ms: Vec<f32>,
    /// Experts the CPU computed, the (expert, row) pairs that was, and the
    /// jobs it was done in.
    pub cpu_experts: u64,
    pub cpu_pairs: u64,
    pub cpu_jobs: u64,
    /// Milliseconds the workers spent in those jobs, summed over the workers.
    pub cpu_busy_ms: f64,
    /// For every layer the CPU had a part in: milliseconds from the arrival
    /// of the record to the delivery of the sums.
    pub cpu_ms: Vec<f32>,
    /// For the same layers: milliseconds from the arrival of the record to
    /// the end of the last task of its jobs (the rest of `cpu_ms` is the
    /// time until the store noticed and delivered the sums).
    pub cpu_done_ms: Vec<f32>,
    /// And from the arrival of the record to the creation of its first job.
    pub cpu_start_ms: Vec<f32>,
    /// Experts that took a VRAM slot between steps, by heat.
    pub promotions: u64,
    /// Experts a step missed that were uploaded on loan.
    pub loans: u64,
    /// Milliseconds the reads of experts a step waited for took, from their
    /// start to the bytes being in RAM, summed over the experts.
    pub demand_read_ms: f64,
    /// Milliseconds layers waited for what they missed, summed over the
    /// layers: from the arrival of a record to the last of its misses being
    /// in a slot or delivered.
    pub wait_ms: f64,
    /// Experts uploaded into transit slots, how many of them from the RAM
    /// tier (the others from the file through staging), and their bytes.
    pub transit_uploads: u64,
    pub transit_from_ram: u64,
    pub transit_upload_bytes: u64,
    /// Reads of experts from the file into staging that a record asked for,
    /// reads that were started ahead of a record on a guess, and the bytes of
    /// both (whole blocks).
    pub transit_reads: u64,
    pub transit_ahead_reads: u64,
    pub transit_read_bytes: u64,
    /// Of the guesses, those a record then named and those that were dropped
    /// after their read had started.
    pub transit_ahead_used: u64,
    pub transit_ahead_unused: u64,
    /// Milliseconds from asking for an expert to its being usable in
    /// transit, summed over the experts.
    pub transit_ms: f64,
    /// Guessed reads that came back before a record named them, and the
    /// milliseconds from the guess to the bytes being in staging, summed.
    pub transit_ahead_done: u64,
    pub transit_ahead_ms: f64,
    /// Guesses of wide steps that were not read: their class of guesses is named less
    /// often than not (`transit.rs`).
    pub transit_ahead_left: u64,
    /// Guesses of wide steps that the RAM tier held and that went up into transit slots
    /// before their layer's record, and of them those it named and those it did not.
    pub transit_guess_uploads: u64,
    pub transit_guess_uploads_used: u64,
    pub transit_guess_uploads_unused: u64,
    /// Experts that came from the file through staging and were kept in the
    /// RAM tier afterwards, by their heat.
    pub transit_admitted: u64,
    /// Experts that took a slot of the VRAM tier from their transit slot, by
    /// their heat.
    pub transit_adopted: u64,
    /// Pieces of the RAM tier given back to the system and taken again, and
    /// the experts that left the tier with the pieces given back.
    pub pieces_released: u64,
    pub pieces_locked: u64,
    pub piece_evictions: u64,
    /// The read-ahead of predicted picks ([`ExpertStore::ahead`]): experts a prediction
    /// named for a layer before its router ran (each once per step), and of them those the
    /// router then picked.
    pub predicted: u64,
    pub predicted_picked: u64,
    /// Reads from the file started ahead of a router for predicted experts that no tier
    /// held, and their bytes (whole blocks); of those, the experts the router then picked
    /// and those it did not (their read cancelled or come back for nothing).
    pub ahead_reads: u64,
    pub ahead_read_bytes: u64,
    pub ahead_used: u64,
    pub ahead_wasted: u64,
    /// Predicted experts in no tier that were not read: their place in the predictions is
    /// picked less often than not, or the bounds of the read-ahead left no room before the
    /// layer's router ran.
    pub ahead_left: u64,
    /// Predicted experts that no tier held when they were named, read ahead or not, and of
    /// them those the router then picked: what a read ahead of every one would have got.
    pub predicted_in_file: u64,
    pub predicted_in_file_picked: u64,
    /// For every pick whose read had started ahead and was still on its way when its
    /// record arrived: milliseconds from the record to its bytes being in RAM.
    pub ahead_miss_ms: Vec<f32>,
    /// VRAM lent by the slots (`scratch.rs`): loans made, and experts moved out of
    /// the lent slots into slots the layers gave.
    pub scratch_loans: u64,
    pub scratch_moves: u64,
}

/// The expert tiers of a model and the work that feeds them.
pub struct ExpertStore<D: SlotDevice> {
    config: StoreConfig,
    /// The block of the files' direct reads: read buffers are whole blocks.
    block: usize,
    map: ExpertMap,
    reader: Reader,
    devices: Vec<D>,
    region: HostRegion,
    layers: Vec<Geometry>,
    vram: Tier,
    ram: Tier,
    heat: ExpertHeat,
    /// The heat as the RAM tier sees it: nothing for experts VRAM holds.
    ram_worth: ExpertHeat,
    worth_stale: Vec<bool>,
    guesses: GuessQueue,
    /// Per RAM pool and slot: where each matrix's bytes start in its buffer.
    ram_skip: Vec<Vec<[u32; 3]>>,
    step: u64,
    in_step: bool,
    /// The tiers are being filled and no step has begun since.
    phase: Vec<Phase>,
    reads: HashMap<u64, ReadOp>,
    next_read: u64,
    completions: Vec<Completion>,
    /// Per device: the uploads and the copies down that are on their way,
    /// each in the order of its stream.
    device_ops: Vec<[VecDeque<(u64, DeviceOp)>; 2]>,
    wants: Vec<Wanted>,
    /// Layers of the running step that miss something, and since when.
    waits: Vec<(usize, Instant)>,
    hybrid: Option<hybrid::Hybrid>,
    transit: Transit,
    pieces: Pieces,
    /// The running step's windows that verify drafts, and the rows left.
    windows: windows::Windows,
    /// What a read from the file that a step waits for takes, in
    /// milliseconds: a mean that follows the reads.
    read_ms: f64,
    /// The read-ahead of predicted picks of the running step and its bounds.
    ahead: ahead::Ahead,
    /// Per device, the VRAM its slots have lent (`scratch.rs`).
    scratch: Vec<Option<scratch::LentScratch>>,
    stats: StoreStats,
}

impl<D: SlotDevice> std::fmt::Debug for ExpertStore<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertStore")
            .field("layers", &self.layers.len())
            .field("step", &self.step)
            .field("wants", &self.wants.len())
            .field("reads", &self.reads.len())
            .finish_non_exhaustive()
    }
}

/// Where a pick that has no slot in VRAM is to come from, counted in `stats`
/// as a miss of that kind: `None` when its upload into VRAM runs already
/// (with its entry behind it; not one a promotion between steps runs,
/// `promoting`), otherwise what [`ram_source`] says.
fn classify_miss(
    vram: &SlotSet,
    ram: &SlotSet,
    stats: &mut StoreStats,
    expert: u32,
    promoting: bool,
) -> Option<Source> {
    if vram.is_loading(expert) && !promoting {
        stats.miss_on_its_way += 1;
        return None;
    }
    Some(ram_source(ram, stats, expert))
}

/// Where a pick that the VRAM tier does not bring is: in RAM, on its way into
/// RAM, or only in the file; counted in `stats` as a miss of that kind.
fn ram_source(ram: &SlotSet, stats: &mut StoreStats, expert: u32) -> Source {
    if ram.contains(expert) {
        stats.miss_from_ram += 1;
        Source::Ram
    } else if ram.is_loading(expert) {
        stats.miss_on_its_way += 1;
        Source::OnItsWay
    } else {
        stats.miss_from_disk += 1;
        Source::Disk
    }
}

/// The bytes of VRAM the experts that stay resident on `device` take: all
/// `n_experts` of each such layer.
fn resident_bytes(layers: &[LayerSpec], device: usize, n_experts: usize) -> u64 {
    layers
        .iter()
        .filter(|s| s.resident && s.device == device)
        .map(|s| s.slot_bytes * n_experts as u64)
        .sum()
}

/// The fewest slots each layer has in each tier.
///
/// A decode step of `max_rows` rows can pick `max_rows * top_k` experts of a layer, and the
/// layer keeps `vram_reserve` slots free for experts that arrive before it runs: that many
/// slots in VRAM let every miss of a step be uploaded. Where the CPU computes what VRAM
/// cannot take ([`StoreConfig::cpu_serves`]) and a device's bytes do not hold that many for
/// each of its layers, its layers have as many as the bytes hold, at least a row's picks and
/// the free ones, and the RAM tier keeps the rest of a step for the CPU: the two floors of a
/// layer together hold a step's picks.
struct Floors {
    /// Per layer.
    vram: Vec<usize>,
    ram: Vec<usize>,
    /// A step's picks and the free slots.
    step: usize,
    /// A row's picks and the free slots.
    least: usize,
}

impl Floors {
    fn new(config: &StoreConfig, layers: &[LayerSpec], budget: &[u64], n_experts: usize) -> Self {
        let step = config.slots_for(config.max_rows);
        let least = config.slots_for(1).min(step);
        let device_floor: Vec<usize> = (0..budget.len())
            .map(|device| {
                let fit = Self::fit(layers, budget, device, n_experts);
                if config.cpu_serves && fit < step { fit.max(least) } else { step }
            })
            .collect();
        let vram: Vec<usize> = layers.iter().map(|s| device_floor[s.device]).collect();
        let ram = vram.iter().map(|v| step.saturating_sub(*v).max(2)).collect();
        Self { vram, ram, step, least }
    }

    /// The slots each layer of `device` can have: the bytes of its budget beside the experts
    /// that stay resident, over the bytes a slot of each of its other layers takes.
    fn fit(layers: &[LayerSpec], budget: &[u64], device: usize, n_experts: usize) -> usize {
        let round: u64 =
            layers.iter().filter(|s| s.device == device && !s.resident).map(|s| s.slot_bytes).sum();
        let resident = resident_bytes(layers, device, n_experts);
        let left = budget.get(device).copied().unwrap_or(0).saturating_sub(resident);
        tier::rounds(left, round, n_experts)
    }

    /// Why the VRAM tier cannot be planned: the first device whose bytes for slots (what is
    /// left beside the experts that stay resident and the transit slots) do not hold the
    /// floor of each of its layers, with what it has and what it would need.
    fn vram_short(
        &self,
        config: &StoreConfig,
        layers: &[LayerSpec],
        budget: &[u64],
        n_experts: usize,
    ) -> String {
        let mib = |bytes: u64| bytes as f64 / f64::from(1u32 << 20);
        let floor_of = |device: usize| {
            layers.iter().zip(&self.vram).find(|(s, _)| s.device == device).map_or(0, |(_, f)| *f)
        };
        let short = (0..budget.len())
            .find(|d| Self::fit(layers, budget, *d, n_experts) < floor_of(*d).min(n_experts));
        let Some(device) = short else {
            return format!(
                "the VRAM budgets do not hold {} slots per layer ({} rows of {} experts and {} \
                 free)",
                self.step, config.max_rows, config.top_k, config.vram_reserve
            );
        };
        let on: Vec<&LayerSpec> =
            layers.iter().filter(|s| s.device == device && !s.resident).collect();
        let round: u64 = on.iter().map(|s| s.slot_bytes).sum();
        let floor = floor_of(device);
        let why = if config.cpu_serves {
            format!(
                "each needs at least {floor} slots there, the {} experts a row picks and {} free \
                 (the CPU computes from RAM what a step of more rows picks beyond them)",
                config.top_k, config.vram_reserve
            )
        } else {
            format!(
                "a step of {} rows picks up to {} experts of a layer, and every miss is uploaded, \
                 so each needs {floor} slots there ({} rows of {} and {} free)",
                config.max_rows,
                config.max_rows * config.top_k,
                config.max_rows,
                config.top_k,
                config.vram_reserve
            )
        };
        format!(
            "device {device} has {:.0} MiB for the slots of experts, room for {} slots in each of \
             the {} layers of experts it holds ({:.1} MiB a slot on average); {why}: {:.0} MiB",
            mib(budget.get(device).copied().unwrap_or(0)),
            Self::fit(layers, budget, device, n_experts),
            on.len(),
            mib(round) / on.len().max(1) as f64,
            mib(round * floor as u64)
        )
    }

    /// Why the RAM tier cannot be planned, with what its floors need.
    fn ram_short(
        &self,
        config: &StoreConfig,
        layers: &[LayerSpec],
        ram_layers: &[(usize, u64)],
        budget: u64,
    ) -> String {
        let mib = |bytes: u64| bytes as f64 / f64::from(1u32 << 20);
        let needed: u64 = ram_layers
            .iter()
            .zip(&self.ram)
            .zip(layers)
            .filter(|(_, spec)| !spec.resident)
            .map(|(((_, stride), floor), _)| stride * *floor as u64)
            .sum();
        let short = layers.iter().zip(&self.vram).find(|(_, v)| **v < self.step);
        match short {
            Some((spec, vram)) => format!(
                "the RAM budget ({:.0} MiB for the tier of experts) does not hold the slots the \
                 CPU needs: device {} holds {vram} slots per layer in VRAM, fewer than the {} a \
                 step of {} rows can pick with the free ones, and the CPU computes the rest from \
                 RAM, so each of those layers needs {} slots in RAM ({:.0} MiB in all; a row \
                 alone needs {} in VRAM)",
                mib(budget),
                spec.device,
                self.step,
                config.max_rows,
                self.step - vram,
                mib(needed),
                self.least
            ),
            None => format!(
                "the RAM budget ({:.0} MiB for the tier of experts) does not hold 2 slots per \
                 layer ({:.0} MiB)",
                mib(budget),
                mib(needed)
            ),
        }
    }
}

impl StorePlan {
    /// Works out the slots of a store: every layer of `map` is described by
    /// `layers`, `reader` reads the files of the model the map was made
    /// from, and `profile` is what is known of the routing (a recorded
    /// profile, or the heat of an earlier run).
    ///
    /// # Errors
    ///
    /// A layer without experts, layers that differ in their number of
    /// experts, and budgets that do not hold the fewest slots a step needs.
    pub fn new(
        config: StoreConfig,
        map: &ExpertMap,
        reader: &Reader,
        layers: &[LayerSpec],
        profile: Option<&ExpertHeat>,
    ) -> Result<StorePlan, StoreError> {
        let plan_error = |text: String| StoreError::Plan(text);
        let n_experts =
            map.n_experts(0).ok_or_else(|| plan_error("layer 0 has no experts".into()))?;
        let block = reader.files().iter().map(|f| f.block as usize).max().unwrap_or(1).max(1);
        let mut geometry = Vec::with_capacity(layers.len());
        let mut ram_layers = Vec::with_capacity(layers.len());
        for (layer, spec) in layers.iter().enumerate() {
            let ranges: Vec<_> = map
                .ranges(layer, 0)
                .ok_or_else(|| plan_error(format!("layer {layer} has no experts")))?
                .collect();
            if ranges.len() != 3 || map.n_experts(layer) != Some(n_experts) {
                return Err(plan_error(format!(
                    "layer {layer}: {} tensors and {:?} experts; a layer has gate, up and down and \
                     {n_experts} experts",
                    ranges.len(),
                    map.n_experts(layer)
                )));
            }
            let mut lens = [0; 3];
            let mut ram_bases = [0; 3];
            let mut at = 0;
            for (part, range) in ranges.iter().enumerate() {
                lens[part] = range.len;
                ram_bases[part] = at;
                // A read covers whole blocks: at most one more than the
                // length rounded up.
                at += range.len.next_multiple_of(block) + block;
            }
            if spec.offsets[2] + lens[2] as u64 > spec.slot_bytes {
                return Err(plan_error(format!("layer {layer}: the matrices do not fit the slot")));
            }
            geometry.push(Geometry {
                device: spec.device,
                device_pool: 0,
                offsets: spec.offsets,
                lens,
                ram_bases,
                resident: spec.resident,
            });
            ram_layers.push((0usize, at as u64));
        }
        let flat = vec![0.0f32; n_experts];
        let heat: Vec<&[f32]> =
            (0..layers.len()).map(|l| profile.map_or(flat.as_slice(), |p| p.layer(l))).collect();
        // A resident layer holds all its experts in VRAM, and in RAM the slots
        // its experts pass through on their way from the file when the store
        // is filled.
        let vram_fixed: Vec<Option<usize>> =
            layers.iter().map(|s| s.resident.then_some(n_experts)).collect();
        let ram_fixed: Vec<Option<usize>> =
            layers.iter().map(|s| s.resident.then_some(RESIDENT_PASSAGE)).collect();
        for (device, budget) in config.vram_bytes.iter().enumerate() {
            let resident = resident_bytes(layers, device, n_experts);
            if resident > *budget {
                return Err(plan_error(format!(
                    "the VRAM budget of device {device} ({} MiB) does not hold the experts that \
                     stay resident there ({} MiB)",
                    budget >> 20,
                    resident >> 20
                )));
            }
        }
        let vram_layers: Vec<(usize, u64)> =
            layers.iter().map(|s| (s.device, s.slot_bytes)).collect();
        // Transit and staging come out of the budgets of the tiers.
        let mut transit =
            TransitPlan { pools: vec![None; config.vram_bytes.len()], ..TransitPlan::default() };
        let mut vram_budget = config.vram_bytes.clone();
        for (device, budget) in vram_budget.iter_mut().enumerate() {
            let slots = config.transit_slots.get(device).copied().unwrap_or(0);
            let stride = layers.iter().filter(|s| s.device == device).map(|s| s.slot_bytes).max();
            if let (true, Some(stride)) = (slots > 0, stride) {
                *budget = budget.checked_sub(stride * u64::from(slots)).ok_or_else(|| {
                    plan_error(format!(
                        "the VRAM budget of device {device} does not hold {slots} transit slots"
                    ))
                })?;
                transit.pools[device] = Some((0, stride, slots));
            }
        }
        transit.stage_slots = config.staging_slots;
        transit.stage_bytes =
            ram_layers.iter().map(|(_, bytes)| *bytes as usize).max().unwrap_or(0);
        let ram_budget =
            config.ram_bytes.checked_sub(transit.stage_len() as u64).ok_or_else(|| {
                plan_error(format!(
                    "the RAM budget does not hold {} staging slots",
                    config.staging_slots
                ))
            })?;
        let floors = Floors::new(&config, layers, &vram_budget, n_experts);
        let vram =
            Tier::plan(n_experts, &vram_layers, &vram_budget, &floors.vram, &heat, &vram_fixed)
                .ok_or_else(|| {
                    plan_error(floors.vram_short(&config, layers, &vram_budget, n_experts))
                })?;
        let ram = Tier::plan(n_experts, &ram_layers, &[ram_budget], &floors.ram, &heat, &ram_fixed)
            .ok_or_else(|| {
                plan_error(floors.ram_short(&config, layers, &ram_layers, ram_budget))
            })?;
        let mut vram = vram;
        for set in &mut vram.sets {
            set.keep_picks();
        }
        let layout = Layout::new(&ram.pools, config.ram_piece, transit.stage_len());
        transit.stage_base = layout.stage_base;
        for (device, pool) in transit.pools.iter_mut().enumerate() {
            if let Some((index, _, _)) = pool {
                *index = vram.pools.iter().filter(|p| p.place == device).count();
            }
        }

        for (layer, g) in geometry.iter_mut().enumerate() {
            let pool = vram.pool_of[layer];
            g.device_pool =
                vram.pools[..pool].iter().filter(|p| p.place == vram.pools[pool].place).count();
        }
        Ok(StorePlan { config, n_experts, block, layers: geometry, vram, ram, transit, layout })
    }
}

impl<D: SlotDevice> ExpertStore<D> {
    /// A store over the slots of `plan`: `devices[d]` has the pools
    /// [`StorePlan::device_pools`] names for device `d` and empty tables,
    /// `region` at least [`StorePlan::ram_bytes`] bytes aligned to
    /// [`StorePlan::block`]. `map` and `reader` are the ones the plan was
    /// made with.
    pub fn new(
        plan: StorePlan,
        map: ExpertMap,
        reader: Reader,
        devices: Vec<D>,
        region: HostRegion,
    ) -> Result<Self, StoreError> {
        let StorePlan { config, n_experts, block, layers, vram, ram, transit, layout } = plan;
        let mut region = region;
        let needed = layout.len;
        if region.len() < needed || !(region.ptr().as_ptr() as usize).is_multiple_of(block) {
            return Err(StoreError::Plan(format!(
                "the RAM region has {} bytes at {:p}; the tier and staging need {needed} aligned \
                 to {block}",
                region.len(),
                region.ptr(),
            )));
        }
        if let Some(missing) =
            (0..layers.len()).map(|l| layers[l].device).find(|d| *d >= devices.len())
        {
            return Err(StoreError::Plan(format!("no device {missing} was given")));
        }
        let heat = ExpertHeat::new(layers.len(), n_experts, config.heat_decay);
        let ram_skip = ram.pools.iter().map(|p| vec![[0; 3]; p.slots as usize]).collect();
        region.cut(&layout.spans);
        let pieces = Pieces::new(layout, &ram);
        let mut store = Self {
            ram_worth: ExpertHeat::new(layers.len(), n_experts, 1.0),
            worth_stale: vec![true; layers.len()],
            guesses: GuessQueue::new(),
            ram_skip,
            step: 0,
            in_step: false,
            phase: vec![Phase::Passed; layers.len()],
            reads: HashMap::new(),
            next_read: 0,
            completions: Vec::new(),
            device_ops: devices.iter().map(|_| [VecDeque::new(), VecDeque::new()]).collect(),
            wants: Vec::new(),
            waits: Vec::new(),
            hybrid: None,
            transit: Transit::new(transit),
            pieces,
            windows: windows::Windows::default(),
            read_ms: 3.0,
            ahead: ahead::Ahead::new(&layers, &reader),
            scratch: devices.iter().map(|_| None).collect(),
            stats: StoreStats::default(),
            block,
            config,
            map,
            reader,
            devices,
            region,
            layers,
            vram,
            ram,
            heat,
        };
        store.settle_pieces()?;
        Ok(store)
    }

    /// The device side of device `device`.
    pub fn device(&self, device: usize) -> &D {
        &self.devices[device]
    }

    pub fn device_mut(&mut self, device: usize) -> &mut D {
        &mut self.devices[device]
    }

    /// The layers of the store.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// The device that runs `layer` and the layer's pool among that device's.
    pub fn place(&self, layer: usize) -> (usize, usize) {
        (self.layers[layer].device, self.layers[layer].device_pool)
    }

    /// The slots `layer` has in VRAM and in RAM.
    pub fn slots(&self, layer: usize) -> (usize, usize) {
        (self.vram.slots(layer), self.ram.slots(layer))
    }

    /// Whether every expert of `layer` has a VRAM slot for good.
    pub fn is_resident(&self, layer: usize) -> bool {
        self.layers[layer].resident
    }

    /// Whether `expert` of `layer` is usable in a VRAM slot, as far as the
    /// store has seen the device confirm.
    pub fn in_vram(&self, layer: usize, expert: u32) -> bool {
        self.vram.sets[layer].contains(expert)
    }

    /// Whether `expert` of `layer` is in a VRAM slot on loan: uploaded for a
    /// step that missed it, its slot the first to be taken again.
    pub fn on_loan(&self, layer: usize, expert: u32) -> bool {
        self.vram.sets[layer].is_loaned(expert)
    }

    /// The experts on loan in VRAM, over all layers.
    pub fn loans(&self) -> usize {
        self.vram.sets.iter().map(|set| set.loans().count()).sum()
    }

    /// The slot of the layer's pool (the number the slot table holds, counted
    /// from the first slot of the pool) that `expert` of `layer` is usable in.
    pub fn vram_slot(&self, layer: usize, expert: u32) -> Option<u32> {
        self.vram.sets[layer].slot(expert).map(|local| self.vram.slot(layer, local))
    }

    /// Whether `expert` of `layer` is in the RAM tier.
    pub fn in_ram(&self, layer: usize, expert: u32) -> bool {
        self.ram.sets[layer].contains(expert)
    }

    /// Sets `out[e]` to 1 for every expert of `layer` that VRAM or the RAM
    /// tier holds or has on its way, and leaves the others as they are: a pick
    /// of such an expert needs no read from the file. The router's pruning
    /// reads this.
    pub fn mark_present(&self, layer: usize, out: &mut [u8]) {
        self.vram.sets[layer].mark_present(out);
        self.ram.sets[layer].mark_present(out);
    }

    pub fn heat(&self) -> &ExpertHeat {
        &self.heat
    }

    pub fn reader(&self) -> &Reader {
        &self.reader
    }

    pub fn stats(&self) -> &StoreStats {
        &self.stats
    }

    /// The counters since the last call, which starts them again.
    pub fn take_stats(&mut self) -> StoreStats {
        std::mem::take(&mut self.stats)
    }

    /// Experts a step waits for that are not in their slot yet.
    pub fn waiting(&self) -> usize {
        self.wants.len() + self.cpu_waiting()
    }

    /// Whether nothing is on its way: no read, no copy, nothing wanted.
    pub fn is_settled(&self) -> bool {
        self.wants.is_empty()
            && self.reads.is_empty()
            && self.guesses.is_empty()
            && self.device_ops.iter().flatten().all(VecDeque::is_empty)
            && self.cpu_is_settled()
            && !self.transit.is_busy()
    }

    /// A step starts: its routers have not run. Call it before the step is
    /// enqueued, and make the step's streams wait for what the store has
    /// enqueued so far (`CudaSlots::fence`).
    pub fn begin_step(&mut self) {
        self.in_step = true;
        self.phase.fill(Phase::Ahead);
        self.windows.set(&[], None);
        self.cpu_begin_step();
        self.transit_begin_step();
        self.ahead_begin_step();
    }

    /// The windows of the step just begun that verify drafts, as (first row,
    /// rows), and what each row of the step is worth in milliseconds
    /// (`worth[row]`, the step record's row worth). A row past the first of a
    /// window waits for the file only while it is worth the wait: an expert
    /// in no tier but the file that only such rows pick is read for them while
    /// the rows it would leave behind are worth more than the read costs the
    /// layer (`windows.rs`); otherwise they and the rows after them in their
    /// window are left without it ([`ExpertStore::rows_left`]). Without
    /// `worth` no row is left. Only a store with a CPU half leaves anything
    /// (the CPU claims the expert and adds nothing).
    pub fn set_windows(&mut self, windows: &[(u32, u32)], worth: Option<&[f32]>) {
        self.windows.set(windows, worth);
    }

    /// The rows of the running or last step, a bit each, whose results are
    /// not what the rows would give alone: the rows of a window from the
    /// first that was left without an expert on.
    pub fn rows_left(&self) -> u32 {
        self.windows.left()
    }

    /// The route record of `layer` of the running step has arrived: `picks`
    /// are the distinct experts its rows use. Returns how many of them are
    /// not in a slot; the store has started to bring those, and the step
    /// waits for them on the device.
    pub fn routed(&mut self, layer: usize, picks: &[u32]) -> Result<usize, StoreError> {
        self.open_layer(layer, picks);
        self.demand(layer, picks)
    }

    /// The route record of `layer` has arrived with `picks`: the layers
    /// before it have passed, the heat learns the picks, the guesses and
    /// what was read ahead for the layer are done with, and the tiers keep
    /// the picks until the step is over.
    fn open_layer(&mut self, layer: usize, picks: &[u32]) {
        for phase in &mut self.phase[..layer] {
            *phase = Phase::Passed;
        }
        self.phase[layer] = Phase::Open;
        self.heat.record(layer, picks);
        self.guesses.forget_layer(layer);
        self.ahead_routed(layer, picks);
        self.worth_stale[layer] = true;
        self.vram.sets[layer].routed(picks, self.step);
        self.ram.sets[layer].routed(picks, self.step);
    }

    /// The kernels that compute `experts` of `layer` in the running step have ended: the
    /// step needs them no more, so they give way before the experts of later requests of
    /// the same layer (a wide step asks for a layer's misses in batches). Without this a
    /// batch could only take the room the step has not claimed yet, and would evict its own
    /// members when the claimed experts make up the rest.
    pub fn finished(&mut self, layer: usize, experts: &[u32]) {
        for expert in experts {
            self.vram.sets[layer].unprotect(*expert);
        }
    }

    /// The experts of `layer` that the kernels enqueued next need, with
    /// everything else of the layer free to give way.
    fn demand(&mut self, layer: usize, picks: &[u32]) -> Result<usize, StoreError> {
        let now = Instant::now();
        let mut missing = 0;
        for expert in picks {
            self.stats.picks += 1;
            if self.vram.sets[layer].contains(*expert) {
                self.stats.hits += 1;
                continue;
            }
            let (vram, ram) = (&self.vram.sets[layer], &self.ram.sets[layer]);
            let source = classify_miss(vram, ram, &mut self.stats, *expert, false)
                .unwrap_or(Source::OnItsWay);
            self.wants.push(Wanted {
                layer,
                expert: *expert,
                since: now,
                source,
                cpu: false,
                loan: false,
            });
            missing += 1;
        }
        if missing > 0 {
            self.waits.push((layer, now));
            self.poll()?;
        }
        Ok(missing)
    }

    /// Adds routing to the heat: `counts[e]` rows of `layer` went to expert
    /// `e`, each weighing `weight`.
    pub fn record_counts(&mut self, layer: usize, counts: &[u32], weight: f32) {
        self.heat.record_counts(layer, counts, weight);
        self.worth_stale[layer] = true;
    }

    /// The running step has ended on the devices. Guesses for it are over,
    /// heat ages, and every layer gets its free slots back.
    pub fn end_step(&mut self) -> Result<(), StoreError> {
        self.cpu_end_step();
        self.in_step = false;
        self.phase.fill(Phase::Passed);
        self.ahead_end_step();
        self.step += 1;
        self.stats.steps += 1;
        // Memory the machine wants back goes before anything new is started,
        // and memory the tier may take again comes a piece a step: a lock
        // takes tens of milliseconds, and the next step waits for it.
        self.shrink_ram(elastic::STEP_WAIT)?;
        self.grow_ram(1)?;
        if self.config.decay_every > 0 && self.step.is_multiple_of(self.config.decay_every) {
            self.heat.decay();
            self.worth_stale.fill(true);
        }
        if self.config.rebalance_every > 0 && self.step.is_multiple_of(self.config.rebalance_every)
        {
            self.rebalance()?;
        }
        if self.serve() == Serve::Stream {
            // All removals first, then the copies down: the next step waits
            // for the removals only.
            let mut evicted = Vec::new();
            for layer in 0..self.layers.len() {
                self.refill_reserve(layer, &mut evicted)?;
            }
            for (layer, expert, slot) in evicted {
                self.demote(layer, expert, slot)?;
            }
        } else {
            // With a CPU half no miss needs a free slot: what moves into
            // VRAM is what the heat says, and it moves now.
            self.promote_hot()?;
        }
        self.poll()
    }

    /// Moves the work on: takes finished reads and copies, starts uploads
    /// and reads for what is wanted. Call it whenever the owner's loop comes
    /// around; it never waits.
    pub fn poll(&mut self) -> Result<(), StoreError> {
        // Reads that finished.
        self.reader.poll(&mut self.completions);
        for done in std::mem::take(&mut self.completions) {
            let Some(op) = self.reads.get_mut(&read_of(done.key)) else {
                self.transit_read_done(done.key, done.result.is_err());
                continue;
            };
            op.parts_left -= 1;
            op.failed |= done.result.is_err();
            if op.parts_left > 0 {
                continue;
            }
            let Some(op) = self.reads.remove(&read_of(done.key)) else { continue };
            if op.ahead {
                self.ahead_read_done(read_of(done.key), op.layer, op.expert, op.failed);
            }
            if op.failed {
                // A read ahead that the layer's record did not need was cancelled. A
                // want of the expert reads it again by itself below.
                self.stats.failed_reads += u64::from(!op.ahead || op.demand);
                self.ram.sets[op.layer].abort(&op.swap);
            } else {
                self.ram.sets[op.layer].commit(&op.swap);
                if op.demand {
                    let ms = op.since.elapsed().as_secs_f64() * 1e3;
                    self.stats.demand_read_ms += ms;
                    self.read_ms += 0.1 * (ms - self.read_ms);
                }
            }
        }

        // Copies the devices have done.
        for (device, queue) in (0..self.devices.len()).flat_map(|d| [(d, 0), (d, 1)]) {
            while let Some((mark, op)) = self.device_ops[device][queue].front().copied() {
                if !self.devices[device].passed(mark)? {
                    break;
                }
                self.device_ops[device][queue].pop_front();
                match op {
                    DeviceOp::Upload { layer, swap, since } => {
                        self.vram.sets[layer].commit(&swap);
                        // The copy in RAM has done its work: it may give
                        // way to the next read.
                        self.ram.sets[layer].release(swap.load);
                        self.ram.sets[layer].unprotect(swap.load);
                        self.worth_stale[layer] = true;
                        self.upload_done(device, since, None);
                    }
                    DeviceOp::Promotion { layer, swap, since } => {
                        self.upload_done(device, since, Some((layer, swap)));
                    }
                    DeviceOp::Demote { layer, swap } => self.ram.sets[layer].commit(&swap),
                }
            }
        }

        // What is wanted: done, or one stage further.
        let mut at = 0;
        while at < self.wants.len() {
            let want = self.wants[at];
            let (layer, expert) = (want.layer, want.expert);
            if want.cpu {
                // The CPU waits for its bytes in RAM, nothing else.
                if self.ram.sets[layer].contains(expert) {
                    self.late_job(layer, expert)?;
                    self.wants.swap_remove(at);
                    continue;
                }
                if self.ram.sets[layer].is_loading(expert) {
                    self.hurry(layer, expert);
                } else {
                    self.read_demanded(layer, expert)?;
                }
                at += 1;
                continue;
            }
            if self.vram.sets[layer].contains(expert) {
                let ms = want.since.elapsed().as_secs_f32() * 1e3;
                self.stats.miss_ms.push(ms);
                match want.source {
                    Source::Disk => self.stats.disk_miss_ms.push(ms),
                    Source::Ram => self.stats.ram_miss_ms.push(ms),
                    Source::OnItsWay => {}
                }
                self.wants.swap_remove(at);
                continue;
            }
            if !self.vram.sets[layer].is_loading(expert) {
                if self.ram.sets[layer].contains(expert) {
                    self.promote(layer, expert, want.loan)?;
                } else if self.ram.sets[layer].is_loading(expert) {
                    self.hurry(layer, expert);
                } else {
                    self.read_demanded(layer, expert)?;
                }
            }
            at += 1;
        }

        // Guesses, while lanes are free. Reads ahead of a router keep within bounds of their
        // own (`ahead.rs`).
        let lanes = self.config.read_lanes;
        while self.reads.len() < lanes && !self.guesses.is_empty() && !self.pieces.shrinking {
            for layer in 0..self.layers.len() {
                self.refresh_worth(layer);
            }
            let ahead = &self.ahead;
            let Some(started) = self.guesses.start_where(
                &mut self.ram.sets,
                &self.ram_worth,
                self.step,
                &mut |layer, expert| ahead.may_start(layer, expert),
            ) else {
                break;
            };
            self.read(started.layer, started.swap, false)?;
        }

        self.poll_cpu()?;
        self.poll_transit()?;

        // Layers that have everything they missed.
        let mut at = 0;
        while at < self.waits.len() {
            let (layer, since) = self.waits[at];
            if self.wants.iter().any(|w| w.layer == layer) || self.cpu_is_open(layer) {
                at += 1;
            } else {
                self.stats.wait_ms += since.elapsed().as_secs_f64() * 1e3;
                self.waits.swap_remove(at);
            }
        }
        Ok(())
    }

    /// The counters of the reader of the model file: reads, bytes and latencies by class.
    pub fn reader_stats(&self) -> ares_io::Stats {
        self.reader.stats()
    }

    /// Whether a slot of `layer` may be emptied now: no kernel can be
    /// reading an expert the step did not pick.
    fn can_evict(&self, layer: usize) -> bool {
        !self.in_step || self.phase[layer] != Phase::Ahead
    }

    fn refresh_worth(&mut self, layer: usize) {
        if !std::mem::take(&mut self.worth_stale[layer]) {
            return;
        }
        let vram = &self.vram.sets[layer];
        let worth = self.ram_worth.layer_mut(layer);
        worth.copy_from_slice(self.heat.layer(layer));
        for (expert, worth) in worth.iter_mut().enumerate() {
            if vram.contains(expert as u32) {
                // Below everything VRAM does not hold, and among the copies
                // the hottest first: VRAM will keep that expert longest, so
                // its copy here is needed last.
                *worth = 1e-6 / (1.0 + *worth);
            }
        }
    }

    /// The address of matrix `part` of the expert in RAM slot `local` of
    /// `layer`: the start of its read buffer.
    fn ram_part(&self, layer: usize, local: u32, part: usize) -> *mut u8 {
        let g = &self.layers[layer];
        let pool = self.ram.pool_of[layer];
        let at = self.pieces.layout.slot_at(pool, self.ram.slot(layer, local)) + g.ram_bases[part];
        debug_assert!(at < self.region.len());
        // SAFETY: the offset is inside the region: the plan sized the pools
        // and `new` checked the region against it.
        unsafe { self.region.ptr().as_ptr().add(at) }
    }

    /// Submits the read of `expert` of `layer` from the file, as `class`: its
    /// three matrices into `parts`. Returns the read's number, the bytes it
    /// reads (whole blocks of the drive) and how far into each buffer its
    /// matrix starts; `None` when the layer has no such expert.
    ///
    /// # Safety
    ///
    /// Each of `parts` holds a part of the store's slot size, and nothing
    /// reads or writes it until the read has come back.
    unsafe fn submit_expert_read(
        &mut self,
        layer: usize,
        expert: u32,
        class: Class,
        parts: &[*mut u8],
    ) -> Result<Option<(u64, u64, [u32; 3])>, StoreError> {
        let Some(ranges) = self.map.ranges(layer, expert as usize) else { return Ok(None) };
        let ranges: Vec<_> = ranges.collect();
        let id = self.next_read;
        self.next_read += 1;
        let (mut bytes, mut skips) = (0, [0u32; 3]);
        for (part, range) in ranges.iter().enumerate() {
            let span = self.reader.span(range.file, range.offset, range.len)?;
            skips[part] = span.skip as u32;
            // SAFETY: the caller's contract.
            let buffer = unsafe { Buffer::from_raw(parts[part], span.len) };
            let read = Read {
                key: part_key(id, part),
                file: range.file,
                offset: range.offset,
                len: range.len,
                class,
                buffer,
            };
            self.reader.submit(read).map_err(|rejected| StoreError::Io(rejected.error))?;
            bytes += span.len as u64;
        }
        Ok(Some((id, bytes, skips)))
    }

    /// Starts the read of an expert from the file into its RAM slot.
    fn read(&mut self, layer: usize, swap: Swap, demand: bool) -> Result<(), StoreError> {
        let class = if demand { Class::Demand } else { Class::Prefetch };
        let parts: Vec<*mut u8> =
            (0..3).map(|part| self.ram_part(layer, swap.slot, part)).collect();
        // SAFETY: the buffers are this slot's parts of the region, which
        // nobody else touches while the slot is loading: no upload reads a
        // slot that is not held, and a held slot is never given to a read.
        let Some((id, bytes, skips)) =
            (unsafe { self.submit_expert_read(layer, swap.load, class, &parts)? })
        else {
            self.ram.sets[layer].abort(&swap);
            return Err(StoreError::Plan(format!("layer {layer} has no expert {}", swap.load)));
        };
        let (pool, slot) = (self.ram.pool_of[layer], self.ram.slot(layer, swap.slot));
        self.ram_skip[pool][slot as usize] = skips;
        if demand {
            self.stats.demand_reads += 1;
            self.stats.demand_read_bytes += bytes;
            self.ahead.demand_bytes += bytes;
        } else {
            self.stats.guess_reads += 1;
            self.stats.guess_read_bytes += bytes;
        }
        let ahead = !demand && self.ahead_started(layer, swap.load, id, bytes);
        self.reads.insert(
            id,
            ReadOp {
                layer,
                expert: swap.load,
                swap,
                parts_left: 3,
                failed: false,
                demand,
                ahead,
                since: Instant::now(),
            },
        );
        Ok(())
    }

    /// Starts a read of `expert` of `layer` into the RAM tier that a step
    /// waits for, if the tier has a slot for it.
    fn read_demanded(&mut self, layer: usize, expert: u32) -> Result<(), StoreError> {
        self.refresh_worth(layer);
        let started = self.ram.sets[layer].start_read(
            expert,
            Want::Demand,
            self.step,
            self.ram_worth.layer(layer),
        );
        if let Start::Slot(swap) = started {
            self.read(layer, swap, true)?;
        }
        Ok(())
    }

    /// A guess that is being read turned out to be needed: its read goes
    /// ahead of the other guesses.
    fn hurry(&mut self, layer: usize, expert: u32) {
        let found = self
            .reads
            .iter_mut()
            .find(|(_, op)| op.layer == layer && op.expert == expert && !op.demand);
        if let Some((id, op)) = found {
            op.demand = true;
            let id = *id;
            self.promote_read(id);
        }
    }

    /// Read `read` goes ahead of the guesses at the reader; a part that has
    /// finished meanwhile is fine as it is.
    fn promote_read(&self, read: u64) {
        for part in 0..3 {
            let _ = self.reader.promote(part_key(read, part));
        }
    }

    /// Read `read` is cancelled: what is at the drive comes back, the rest is
    /// not started.
    fn cancel_read(&self, read: u64) {
        for part in 0..3 {
            let _ = self.reader.cancel(part_key(read, part));
        }
    }

    /// Starts the upload of an expert that a step waits for and that is in
    /// RAM into a VRAM slot, if a slot can be had now, with its table entry
    /// behind it.
    fn promote(&mut self, layer: usize, expert: u32, loan: bool) -> Result<(), StoreError> {
        let started = self.upload(layer, expert, Want::Demand, true)?;
        if started && loan {
            self.vram.sets[layer].lend(expert);
            self.stats.loans += 1;
        }
        Ok(())
    }

    /// `expert` of `layer` has left VRAM slot `slot` (of the layer's pool):
    /// its table entry goes, and it comes down to RAM ([`ExpertStore::demote`]).
    fn evict_vram(&mut self, layer: usize, expert: u32, slot: u32) -> Result<(), StoreError> {
        self.stats.evictions += 1;
        self.devices[self.layers[layer].device].set_entry(layer, expert, -1)?;
        self.demote(layer, expert, slot)
    }

    /// Copies an expert that VRAM evicts from its slot (`slot` of the
    /// layer's pool) down to RAM, unless it is there or was never used.
    fn demote(&mut self, layer: usize, expert: u32, slot: u32) -> Result<(), StoreError> {
        let ram = &self.ram.sets[layer];
        if ram.contains(expert)
            || ram.is_loading(expert)
            || self.heat.layer(layer)[expert as usize] <= 0.0
        {
            return Ok(());
        }
        self.refresh_worth(layer);
        // It comes as an expert that was used, not as a guess: its worth is
        // its heat.
        let Start::Slot(swap) = self.ram.sets[layer].start_read(
            expert,
            Want::Demand,
            self.step,
            self.ram_worth.layer(layer),
        ) else {
            return Ok(());
        };
        let Some(ranges) = self.map.ranges(layer, expert as usize) else { return Ok(()) };
        let ranges: Vec<_> = ranges.collect();
        let g = self.layers[layer];
        let (pool, ram_slot) = (self.ram.pool_of[layer], self.ram.slot(layer, swap.slot));
        for (part, range) in ranges.iter().enumerate() {
            // The bytes go where a read of the expert would put them.
            let span = self.reader.span(range.file, range.offset, range.len)?;
            self.ram_skip[pool][ram_slot as usize][part] = span.skip as u32;
            // SAFETY: the destination is this RAM slot's part, loading until
            // the mark below is passed, so nobody else touches it.
            unsafe {
                let dst = self.ram_part(layer, swap.slot, part).add(span.skip);
                self.devices[g.device].download(
                    g.device_pool,
                    slot,
                    g.offsets[part],
                    dst,
                    g.lens[part],
                )?;
            }
            self.stats.demote_bytes += g.lens[part] as u64;
        }
        let mark = self.devices[g.device].mark(true)?;
        self.device_ops[g.device][1].push_back((mark, DeviceOp::Demote { layer, swap }));
        self.stats.demotions += 1;
        Ok(())
    }

    /// Frees slots of `layer` until [`StoreConfig::vram_reserve`] are free,
    /// the coldest members first, and lists what was evicted (layer, expert,
    /// slot of the pool) for the copies down. Between steps only.
    fn refill_reserve(
        &mut self,
        layer: usize,
        evicted: &mut Vec<(usize, u32, u32)>,
    ) -> Result<(), StoreError> {
        let reserve = self.config.vram_reserve;
        if self.vram.sets[layer].n_free() >= reserve || self.layers[layer].resident {
            return Ok(());
        }
        let given = self.vram.sets[layer].give_back(reserve, self.step, self.heat.layer(layer));
        let device = self.layers[layer].device;
        for slot in &given {
            if let Some(expert) = slot.evicted {
                self.stats.evictions += 1;
                self.devices[device].set_entry(layer, expert, -1)?;
                evicted.push((layer, expert, self.vram.slot(layer, slot.slot)));
            }
        }
        self.vram.sets[layer].take(given.len());
        self.worth_stale[layer] = true;
        Ok(())
    }

    /// Lets slots pass between the layers of each VRAM pool by what the heat
    /// says they need. Between steps only.
    fn rebalance(&mut self) -> Result<(), StoreError> {
        for pool in 0..self.vram.pools.len() {
            let members: Vec<usize> =
                (0..self.layers.len()).filter(|l| self.vram.pool_of[*l] == pool).collect();
            let current: Vec<usize> = members.iter().map(|l| self.vram.slots(*l)).collect();
            let needs: Vec<LayerNeed<'_>> = members
                .iter()
                .map(|l| LayerNeed { heat: self.heat.layer(*l), expert_bytes: 1 })
                .collect();
            let total: usize = current.iter().sum();
            let target = split_slots(&needs, total as u64, Some(&current), &SplitRule::default());
            let mut takers: Vec<usize> = (0..members.len())
                .flat_map(|i| std::iter::repeat_n(i, target[i].saturating_sub(current[i])))
                .collect();
            for (i, layer) in members.iter().enumerate() {
                for _ in target[i]..current[i] {
                    if self.vram.slots(*layer) <= self.vram.floor[*layer] {
                        break;
                    }
                    let Some(to) = takers.pop() else { break };
                    let given =
                        self.vram.sets[*layer].give_back(1, self.step, self.heat.layer(*layer));
                    let Some(slot) = given.first() else { break };
                    if let Some(evicted) = slot.evicted {
                        self.evict_vram(*layer, evicted, self.vram.slot(*layer, slot.slot))?;
                    }
                    self.vram.move_slot(*layer, slot.slot, members[to]);
                    self.worth_stale[*layer] = true;
                    self.stats.moved_slots += 1;
                }
            }
        }
        Ok(())
    }
}
