//! Transit: experts that are computed from VRAM without taking a slot there.
//!
//! A slice of a prompt uses a large part of the experts of every layer, far
//! more than the layer has slots, and uses each of them for all its rows at
//! once and then no more. Giving such an expert a slot means evicting one
//! that generation needs, copying that one down to RAM, and bringing it back
//! afterwards. So the experts a wide step misses pass through memory of their
//! own instead:
//!
//! - **Transit slots**: a small pool on each device, an extra pool of its
//!   [`SlotDevice`]. The kernels of a wide step read an expert from an
//!   address the host gives them, so an expert in a transit slot is computed
//!   like one in a slot of the tier.
//! - **Staging slots**: page-locked host memory behind the RAM tier's, where
//!   an expert that is only in the file is read to (in place, with direct
//!   I/O) and uploaded from.
//!
//! An expert that the RAM tier holds is uploaded from its RAM slot, which is
//! held meanwhile and keeps its expert. One that is in no tier goes from the
//! file through staging. Either way nothing is evicted from VRAM, nothing is
//! copied down, and no table entry is written.
//!
//! What the tiers hold afterwards is decided by the heat, as everywhere: when
//! the kernels have read an expert that came from the file, its bytes are
//! still in staging, and if the heat (which has the counts of the step that
//! used it by then) says it is worth more than what the RAM tier would give
//! up for it, by the margin of a swap, it is copied into the RAM tier there
//! and then ([`ExpertStore::transit_release`]). From the RAM tier the hottest
//! go up into VRAM between steps like any others.
//!
//! # Who asks
//!
//! - [`ExpertStore::transit_want`]: a layer's route record named these experts
//!   and they have no slot. Their reads are demand reads and start at once, as
//!   many as staging takes; their uploads follow each other on the device's
//!   upload stream as transit slots are free.
//! - [`ExpertStore::transit_guess`]: something expects these experts to be
//!   named soon. Those that are in no tier are read ahead into staging as
//!   prefetches, in at most half of it; the want that names one promotes its
//!   read, and [`ExpertStore::transit_forget`] drops the others.
//! - [`ExpertStore::transit_take`] hands over what has arrived, and
//!   [`ExpertStore::transit_release`] gives the slots back once the kernels
//!   that read them have ended.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use ares_io::{Buffer, Class, Read};
use ares_weights::{Start, Swap, SwapRule, Want};

use super::{ExpertStore, Phase, SlotDevice, StoreError};

/// The transit and staging slots of a store: what [`super::StorePlan`] works
/// out.
#[derive(Clone, Debug, Default)]
pub(super) struct TransitPlan {
    /// Per device: the transit pool among the device's pools, the bytes from
    /// one slot to the next, and the slots.
    pub pools: Vec<Option<(usize, u64, u32)>>,
    /// Where staging starts in the host region, the bytes of a staging slot
    /// and how many there are.
    pub stage_base: usize,
    pub stage_bytes: usize,
    pub stage_slots: u32,
}

impl TransitPlan {
    /// Bytes of staging.
    pub fn stage_len(&self) -> usize {
        self.stage_bytes * self.stage_slots as usize
    }

    /// Bytes of the transit pool of `device`.
    pub fn bytes(&self, device: usize) -> u64 {
        match self.pools.get(device) {
            Some(Some((_, stride, slots))) => stride * u64::from(*slots),
            _ => 0,
        }
    }
}

/// What [`ExpertStore::transit_want`] found of the experts that have to be
/// read from the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Named {
    /// Experts a guess had asked for: their read is on its way or done.
    pub guessed: u32,
    /// Experts in no tier that nobody had guessed: their read starts now.
    pub unguessed: u32,
}

/// Where an expert in transit is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Nothing has started.
    Queued,
    /// The file is being read into a staging slot.
    Reading { stage: u32, read: u64, parts_left: u8, failed: bool },
    /// Its bytes are in a staging slot.
    Staged { stage: u32 },
    /// On its way into a transit slot: from a staging slot, or from its RAM
    /// slot, which is held.
    Uploading { slot: u32, stage: Option<u32> },
    /// Usable in its transit slot. What came from the file keeps its staging
    /// slot until it is released.
    Ready { slot: u32, stage: Option<u32> },
    /// Handed to the caller, whose kernels read it.
    Taken { slot: u32, stage: Option<u32> },
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    state: State,
    /// A route record named it. Otherwise it is read ahead on a guess.
    wanted: bool,
    /// Nobody needs it any more: it leaves when nothing of it is on its way.
    dropped: bool,
    since: Instant,
}

/// Transit as the store keeps it.
#[derive(Debug, Default)]
pub(super) struct Transit {
    plan: TransitPlan,
    /// Per device: the transit slots that hold nothing.
    free: Vec<Vec<u32>>,
    stage_free: Vec<u32>,
    /// Per staging slot: where each matrix starts in its read buffer.
    stage_skip: Vec<[u32; 3]>,
    /// Staging slots that guesses hold.
    stage_guessed: usize,
    entries: HashMap<(usize, u32), Entry>,
    /// Named experts that are not on their way into transit yet, in the order
    /// they were asked for; guesses whose read has not started, likewise.
    wanted: VecDeque<(usize, u32)>,
    guesses: VecDeque<(usize, u32)>,
    /// Reads of the file by their number.
    reads: HashMap<u64, (usize, u32)>,
    /// Per device: the uploads on their way, in the order of the stream.
    uploads: Vec<VecDeque<(u64, (usize, u32))>>,
    /// Per device: the copies from a transit slot into a slot of the tier that
    /// are on their way, in the order of the stream.
    adoptions: Vec<VecDeque<(u64, Adoption)>>,
}

/// An expert that leaves transit for a slot of the VRAM tier.
#[derive(Clone, Copy, Debug)]
struct Adoption {
    layer: usize,
    swap: Swap,
    /// The transit slot it is copied from, and its staging slot if it came
    /// from the file.
    slot: u32,
    stage: Option<u32>,
}

impl Transit {
    pub(super) fn new(plan: TransitPlan) -> Self {
        Self {
            free: plan
                .pools
                .iter()
                .map(|pool| pool.map_or_else(Vec::new, |(_, _, slots)| (0..slots).rev().collect()))
                .collect(),
            stage_free: (0..plan.stage_slots).rev().collect(),
            stage_skip: vec![[0; 3]; plan.stage_slots as usize],
            stage_guessed: 0,
            entries: HashMap::new(),
            wanted: VecDeque::new(),
            guesses: VecDeque::new(),
            reads: HashMap::new(),
            uploads: plan.pools.iter().map(|_| VecDeque::new()).collect(),
            adoptions: plan.pools.iter().map(|_| VecDeque::new()).collect(),
            plan,
        }
    }

    /// Whether a read or a copy of transit is on its way.
    pub(super) fn is_busy(&self) -> bool {
        !self.reads.is_empty()
            || self.uploads.iter().any(|queue| !queue.is_empty())
            || self.adoptions.iter().any(|queue| !queue.is_empty())
    }

    fn free_stage(&mut self, stage: u32, guessed: bool) {
        self.stage_free.push(stage);
        if guessed {
            self.stage_guessed = self.stage_guessed.saturating_sub(1);
        }
    }
}

impl<D: SlotDevice> ExpertStore<D> {
    /// The transit pool of the device of `layer`: its number among the pools
    /// of the device (as [`super::StorePlan::device_pools`] lists them), the
    /// bytes from one slot to the next, and the slots. `None` when the store
    /// has no transit there.
    pub fn transit_pool(&self, layer: usize) -> Option<(usize, u64, u32)> {
        self.transit.plan.pools.get(self.layers[layer].device).copied().flatten()
    }

    /// Staging slots of the store.
    pub fn staging_slots(&self) -> usize {
        self.transit.plan.stage_slots as usize
    }

    /// The route record of `layer` named `experts`, which have no slot: they
    /// are brought into transit, in this order. An expert that was asked for
    /// before keeps its place; one that was read ahead on a guess is now
    /// waited for.
    ///
    /// Returns how the experts that are named for the first time and that are
    /// in no tier divide: those a guess had asked for, and those nobody had.
    ///
    /// # Errors
    ///
    /// The device of the layer has no transit slots, or the store has no
    /// staging.
    pub fn transit_want(&mut self, layer: usize, experts: &[u32]) -> Result<Named, StoreError> {
        if self.transit_pool(layer).is_none() || self.transit.plan.stage_slots == 0 {
            return Err(StoreError::Plan(format!("layer {layer} has no transit")));
        }
        let now = Instant::now();
        let mut named = Named::default();
        for expert in experts {
            let key = (layer, *expert);
            match self.transit.entries.get_mut(&key) {
                Some(entry) if entry.wanted => {
                    // One that was let go while its read or its upload ran (the layer was
                    // forgotten) is waited for again, with what is on its way.
                    if std::mem::take(&mut entry.dropped) {
                        entry.since = now;
                        if !self.transit.wanted.contains(&key) {
                            self.transit.wanted.push_back(key);
                        }
                    }
                }
                Some(entry) => {
                    entry.wanted = true;
                    entry.dropped = false;
                    entry.since = now;
                    let state = entry.state;
                    self.stats.picks += 1;
                    if matches!(state, State::Reading { .. } | State::Staged { .. }) {
                        self.stats.transit_ahead_used += 1;
                        self.stats.miss_on_its_way += 1;
                        named.guessed += 1;
                    } else {
                        // The guess had not got as far as a read.
                        self.stats.miss_from_disk += 1;
                        named.unguessed += 1;
                    }
                    match state {
                        State::Reading { read, .. } => {
                            self.transit.stage_guessed =
                                self.transit.stage_guessed.saturating_sub(1);
                            for part in 0..3 {
                                // A part that has finished is fine as it is.
                                let _ = self.reader.promote(read * 4 + part);
                            }
                        }
                        State::Staged { .. } => {
                            self.transit.stage_guessed =
                                self.transit.stage_guessed.saturating_sub(1);
                        }
                        _ => {}
                    }
                    self.transit.wanted.push_back(key);
                }
                None => {
                    // A pick that found no slot, as the store counts the others.
                    let ram = &self.ram.sets[layer];
                    self.stats.picks += 1;
                    if ram.contains(*expert) {
                        self.stats.miss_from_ram += 1;
                    } else if ram.is_loading(*expert) {
                        self.stats.miss_on_its_way += 1;
                    } else {
                        self.stats.miss_from_disk += 1;
                        named.unguessed += 1;
                    }
                    self.transit.entries.insert(
                        key,
                        Entry { state: State::Queued, wanted: true, dropped: false, since: now },
                    );
                    self.transit.wanted.push_back(key);
                }
            }
        }
        self.poll_transit()?;
        Ok(named)
    }

    /// Something expects the record of `layer` to name `experts`, most likely
    /// first. Those that are in no tier are read ahead into staging, in at
    /// most half of it and behind every read a record asked for; the others
    /// are an upload away and are left alone.
    pub fn transit_guess(&mut self, layer: usize, experts: &[u32]) -> Result<(), StoreError> {
        if self.transit_pool(layer).is_none() || self.transit.plan.stage_slots == 0 {
            return Ok(());
        }
        let now = Instant::now();
        for expert in experts {
            let key = (layer, *expert);
            if self.transit.entries.contains_key(&key)
                || self.vram.sets[layer].contains(*expert)
                || self.vram.sets[layer].is_loading(*expert)
                || self.ram.sets[layer].contains(*expert)
                || self.ram.sets[layer].is_loading(*expert)
            {
                continue;
            }
            self.transit.entries.insert(
                key,
                Entry { state: State::Queued, wanted: false, dropped: false, since: now },
            );
            self.transit.guesses.push_back(key);
        }
        self.poll_transit()
    }

    /// Adds to `out` the experts of `layer` that have arrived in transit since
    /// the last call, each with its transit slot. They stay there until
    /// [`ExpertStore::transit_release`].
    pub fn transit_take(&mut self, layer: usize, out: &mut Vec<(u32, u32)>) {
        let first = out.len();
        for (key, entry) in &mut self.transit.entries {
            if let (true, State::Ready { slot, stage }) =
                (key.0 == layer && entry.wanted, entry.state)
            {
                entry.state = State::Taken { slot, stage };
                out.push((key.1, slot));
            }
        }
        out[first..].sort_unstable();
    }

    /// Experts of `layer` a record named that have not arrived in transit yet.
    pub fn transit_pending(&self, layer: usize) -> usize {
        self.transit
            .entries
            .iter()
            .filter(|(key, entry)| {
                key.0 == layer
                    && entry.wanted
                    && !matches!(entry.state, State::Ready { .. } | State::Taken { .. })
            })
            .count()
    }

    /// Experts of `layer` a record named that are on their way now: being read from the
    /// file or uploaded. The others that have not arrived wait for a staging or a transit
    /// slot, which only a release gives them.
    pub fn transit_moving(&self, layer: usize) -> usize {
        self.transit
            .entries
            .iter()
            .filter(|(key, entry)| {
                key.0 == layer
                    && entry.wanted
                    && !entry.dropped
                    && match entry.state {
                        State::Reading { .. } | State::Uploading { .. } => true,
                        State::Queued => self.ram.sets[layer].is_loading(key.1),
                        _ => false,
                    }
            })
            .count()
    }

    /// Every kernel enqueued so far for `layer` in the running step has ended
    /// on the device, and `experts` are those of them it read from transit:
    /// their slots are free for the next ones.
    ///
    /// What becomes of an expert is the heat's to say, which has the counts of
    /// the step by now. One that is hotter than the coldest member of the
    /// layer's VRAM slots, by the margin of a swap, takes that member's slot:
    /// it is copied there from its transit slot, on the device, so the tier
    /// follows a prompt without an upload ([`ExpertStore::transit_adopt`]).
    /// One that came from the file and that the RAM tier would keep is copied
    /// there from staging. The others are forgotten.
    pub fn transit_release(&mut self, layer: usize, experts: &[u32]) -> Result<(), StoreError> {
        // No kernel of the step reads a slot of the layer any more: the
        // members of its tier may give way.
        if self.in_step {
            self.phase[layer] = Phase::Passed;
        }
        let device = self.layers[layer].device;
        for expert in experts {
            if let Some(Entry { state: State::Taken { slot, stage }, .. }) =
                self.transit.entries.get(&(layer, *expert)).copied()
            {
                self.transit.entries.remove(&(layer, *expert));
                if self.transit_adopt(layer, *expert, slot, stage)? {
                    continue;
                }
                self.transit.free[device].push(slot);
                if let Some(stage) = stage {
                    self.transit_admit(layer, *expert, stage);
                    self.transit.free_stage(stage, false);
                }
            }
        }
        self.poll_transit()
    }

    /// Gives `expert` of `layer`, which is in transit slot `slot`, a slot of
    /// the layer's VRAM tier if the heat says so: a free one, or the coldest
    /// member's when the expert is hotter by the margin of a swap
    /// ([`SwapRule`]). Its bytes are copied from the transit slot on the
    /// device, its table entry follows them, and the transit slot (and the
    /// staging slot `stage`) are free when that has run. What it evicts goes
    /// the way of every evicted expert. Returns whether it started.
    fn transit_adopt(
        &mut self,
        layer: usize,
        expert: u32,
        slot: u32,
        stage: Option<u32>,
    ) -> Result<bool, StoreError> {
        let g = self.layers[layer];
        let Some((transit_pool, _, slots)) = self.transit.plan.pools[g.device] else {
            return Ok(false);
        };
        // A transit slot is held until its copy has run, behind the copies
        // down of what the adopted experts evict: half of the slots stay for
        // the experts that are coming in.
        if 2 * (self.transit.adoptions[g.device].len() + 1) > slots as usize {
            return Ok(false);
        }
        let vram = &self.vram.sets[layer];
        let heat = self.heat.layer(layer);
        let own = heat[expert as usize];
        if own <= 0.0 || vram.contains(expert) || vram.is_loading(expert) {
            return Ok(false);
        }
        if vram.n_free() == 0 {
            let rule = SwapRule::default();
            let coldest = (0..heat.len() as u32)
                .filter(|member| vram.contains(*member))
                .map(|member| heat[member as usize])
                .min_by(f32::total_cmp)
                .unwrap_or(0.0);
            if own <= rule.margin * coldest + rule.bonus {
                return Ok(false);
            }
        }
        let started = self.vram.sets[layer].start_read(expert, Want::Demand, self.step, heat);
        let Start::Slot(swap) = started else { return Ok(false) };
        let tier_slot = self.vram.slot(layer, swap.slot);
        if let Some(evicted) = swap.evict {
            self.stats.evictions += 1;
            self.devices[g.device].set_entry(layer, evicted, -1)?;
            self.demote(layer, evicted, tier_slot)?;
        }
        let parts = [0, 1, 2].map(|part| (g.offsets[part], g.lens[part]));
        self.devices[g.device].adopt(
            transit_pool,
            slot,
            g.device_pool,
            tier_slot,
            &parts,
            layer,
            expert,
        )?;
        let mark = self.devices[g.device].mark(true)?;
        self.transit.adoptions[g.device].push_back((mark, Adoption { layer, swap, slot, stage }));
        self.worth_stale[layer] = true;
        self.stats.transit_adopted += 1;
        Ok(true)
    }

    /// Lets `expert` of `layer`, whose bytes are in staging slot `stage`, take
    /// a slot of the RAM tier if the heat knows it and a slot is free or the
    /// swap pays ([`SwapRule`]: the same margin that keeps two experts of
    /// similar heat from trading places in an election).
    fn transit_admit(&mut self, layer: usize, expert: u32, stage: u32) {
        let ram = &self.ram.sets[layer];
        if ram.contains(expert) || ram.is_loading(expert) || self.vram.sets[layer].contains(expert)
        {
            return;
        }
        self.refresh_worth(layer);
        let worth = self.ram_worth.layer(layer);
        let ram = &self.ram.sets[layer];
        let rule = SwapRule::default();
        let coldest = (0..worth.len() as u32)
            .filter(|member| ram.contains(*member))
            .map(|member| worth[member as usize])
            .min_by(f32::total_cmp)
            .unwrap_or(0.0);
        let heat = self.heat.layer(layer)[expert as usize];
        // Nothing the heat has not seen is kept, and a member gives way only to an expert
        // that is hotter by the margin.
        if heat <= 0.0 || (ram.n_free() == 0 && heat <= rule.margin * coldest + rule.bonus) {
            return;
        }
        let started = self.ram.sets[layer].start_read(expert, Want::Demand, self.step, worth);
        let Start::Slot(swap) = started else { return };
        let (pool, slot) = (self.ram.pool_of[layer], self.ram.slot(layer, swap.slot));
        for part in 0..3 {
            let skip = self.transit.stage_skip[stage as usize][part];
            self.ram_skip[pool][slot as usize][part] = skip;
            // The read buffer of a part: the blocks the read covered.
            let len = (skip as usize + self.layers[layer].lens[part]).next_multiple_of(self.block);
            // SAFETY: both are read buffers of this part, of at least `len`
            // bytes (a read covers whole blocks, as the plan sized them), in
            // different slots of the region; the RAM slot is loading, so
            // nothing else reads or writes it, and the staging slot is taken.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.stage_part(layer, stage, part).cast_const(),
                    self.ram_part(layer, swap.slot, part),
                    len,
                );
            }
        }
        self.ram.sets[layer].commit(&swap);
        self.worth_stale[layer] = true;
        self.stats.transit_admitted += 1;
    }

    /// `layer` is done with: whatever was asked for it leaves transit and
    /// staging. A guess that no record named is counted as read for nothing
    /// when its read had started. Nothing of the layer may be read by a
    /// kernel any more.
    pub fn transit_forget(&mut self, layer: usize) -> Result<(), StoreError> {
        let device = self.layers[layer].device;
        let keys: Vec<(usize, u32)> =
            self.transit.entries.keys().filter(|key| key.0 == layer).copied().collect();
        for key in keys {
            let Some(entry) = self.transit.entries.get(&key).copied() else { continue };
            let guess = !entry.wanted;
            match entry.state {
                State::Queued => {
                    self.transit.entries.remove(&key);
                }
                State::Reading { read, .. } => {
                    for part in 0..3 {
                        // What is at the drive comes back; the rest is not started.
                        let _ = self.reader.cancel(read * 4 + part);
                    }
                    if let Some(entry) = self.transit.entries.get_mut(&key) {
                        entry.dropped = true;
                    }
                    self.stats.transit_ahead_unused += u64::from(guess);
                }
                State::Staged { stage } => {
                    self.transit.entries.remove(&key);
                    self.transit.free_stage(stage, guess);
                    self.stats.transit_ahead_unused += u64::from(guess);
                }
                State::Uploading { .. } => {
                    if let Some(entry) = self.transit.entries.get_mut(&key) {
                        entry.dropped = true;
                    }
                }
                State::Ready { slot, stage } | State::Taken { slot, stage } => {
                    self.transit.entries.remove(&key);
                    self.transit.free[device].push(slot);
                    if let Some(stage) = stage {
                        self.transit.free_stage(stage, false);
                    }
                }
            }
        }
        self.transit.wanted.retain(|key| key.0 != layer);
        self.transit.guesses.retain(|key| key.0 != layer);
        self.poll_transit()
    }

    /// A step begins: what the step before left in transit (a step that
    /// failed) is dropped. Guesses stay.
    pub(super) fn transit_begin_step(&mut self) {
        let named: Vec<usize> = {
            let mut layers: Vec<usize> = self
                .transit
                .entries
                .iter()
                .filter(|(_, entry)| entry.wanted)
                .map(|(key, _)| key.0)
                .collect();
            layers.sort_unstable();
            layers.dedup();
            layers
        };
        for layer in named {
            // Nothing is read by a kernel between steps.
            let _ = self.transit_forget(layer);
        }
    }

    /// The address of matrix `part` of staging slot `stage` for an expert of
    /// `layer`: the start of its read buffer.
    fn stage_part(&self, layer: usize, stage: u32, part: usize) -> *mut u8 {
        let plan = &self.transit.plan;
        let at = plan.stage_base
            + stage as usize * plan.stage_bytes
            + self.layers[layer].ram_bases[part];
        debug_assert!(at < self.region.len);
        // SAFETY: the offset is inside the region: the plan put staging behind
        // the RAM tier and `new` checked the region against both.
        unsafe { self.region.ptr.as_ptr().add(at) }
    }

    /// Starts the read of an expert from the file into a staging slot, if one
    /// can be had; returns whether it did.
    fn transit_read(&mut self, key: (usize, u32), demand: bool) -> Result<bool, StoreError> {
        let (layer, expert) = key;
        if !demand && 2 * (self.transit.stage_guessed + 1) > self.transit.plan.stage_slots as usize
        {
            return Ok(false);
        }
        let stage = match self.transit.stage_free.pop() {
            Some(stage) => stage,
            None if demand => {
                // A guess that was read and that no record has named gives way.
                let victim = self.transit.entries.iter().find_map(|(key, entry)| {
                    match (entry.wanted, entry.state) {
                        (false, State::Staged { stage }) => Some((*key, stage)),
                        _ => None,
                    }
                });
                let Some((victim, stage)) = victim else { return Ok(false) };
                self.transit.entries.remove(&victim);
                self.transit.stage_guessed = self.transit.stage_guessed.saturating_sub(1);
                self.stats.transit_ahead_unused += 1;
                stage
            }
            None => return Ok(false),
        };
        let Some(ranges) = self.map.ranges(layer, expert as usize) else {
            self.transit.stage_free.push(stage);
            return Err(StoreError::Plan(format!("layer {layer} has no expert {expert}")));
        };
        let ranges: Vec<_> = ranges.collect();
        let id = self.next_read;
        self.next_read += 1;
        let mut bytes = 0;
        for (part, range) in ranges.iter().enumerate() {
            let span = self.reader.span(range.file, range.offset, range.len)?;
            self.transit.stage_skip[stage as usize][part] = span.skip as u32;
            // SAFETY: the buffer is this staging slot's part of the region,
            // which nobody else touches until the read has come back: a slot
            // is free only when no read and no upload uses it.
            let buffer = unsafe { Buffer::from_raw(self.stage_part(layer, stage, part), span.len) };
            let read = Read {
                key: id * 4 + part as u64,
                file: range.file,
                offset: range.offset,
                len: range.len,
                class: if demand { Class::Demand } else { Class::Prefetch },
                buffer,
            };
            self.reader.submit(read).map_err(|rejected| StoreError::Io(rejected.error))?;
            bytes += span.len as u64;
        }
        self.transit.reads.insert(id, key);
        if let Some(entry) = self.transit.entries.get_mut(&key) {
            entry.state = State::Reading { stage, read: id, parts_left: 3, failed: false };
        }
        if demand {
            self.stats.transit_reads += 1;
        } else {
            self.transit.stage_guessed += 1;
            self.stats.transit_ahead_reads += 1;
        }
        self.stats.transit_read_bytes += bytes;
        Ok(true)
    }

    /// A read of the file has come back. Returns whether it was transit's.
    pub(super) fn transit_read_done(&mut self, read_key: u64, failed: bool) -> bool {
        let id = read_key / 4;
        let Some(key) = self.transit.reads.get(&id).copied() else { return false };
        let Some(entry) = self.transit.entries.get_mut(&key) else {
            self.transit.reads.remove(&id);
            return true;
        };
        let State::Reading { stage, read, parts_left, failed: before } = entry.state else {
            return true;
        };
        let (parts_left, failed) = (parts_left - 1, before || failed);
        if parts_left > 0 {
            entry.state = State::Reading { stage, read, parts_left, failed };
            return true;
        }
        self.transit.reads.remove(&id);
        let guess = !entry.wanted;
        if entry.dropped {
            self.transit.entries.remove(&key);
            self.transit.free_stage(stage, guess);
        } else if failed {
            // It is asked for again from the start.
            self.stats.failed_reads += 1;
            entry.state = State::Queued;
            if guess {
                self.transit.entries.remove(&key);
            }
            self.transit.free_stage(stage, guess);
        } else {
            if guess {
                self.stats.transit_ahead_ms += entry.since.elapsed().as_secs_f64() * 1e3;
                self.stats.transit_ahead_done += 1;
            }
            entry.state = State::Staged { stage };
        }
        true
    }

    /// Starts the upload of an expert into a transit slot of its device, from
    /// its staging slot or (`stage` is `None`) from its RAM slot; returns
    /// whether a slot was free.
    fn transit_upload(
        &mut self,
        key: (usize, u32),
        stage: Option<u32>,
    ) -> Result<bool, StoreError> {
        let (layer, expert) = key;
        let g = self.layers[layer];
        let Some((pool, _, _)) = self.transit.plan.pools[g.device] else { return Ok(false) };
        let Some(slot) = self.transit.free[g.device].pop() else { return Ok(false) };
        let ram_slot = match stage {
            Some(_) => None,
            None => {
                let Some(ram_slot) = self.ram.sets[layer].slot(expert) else {
                    self.transit.free[g.device].push(slot);
                    return Ok(false);
                };
                // The RAM copy stays where it is until the device has read it.
                self.ram.sets[layer].hold(expert);
                Some(ram_slot)
            }
        };
        for part in 0..3 {
            let src = match (stage, ram_slot) {
                (Some(stage), _) => {
                    let skip = self.transit.stage_skip[stage as usize][part] as usize;
                    // SAFETY: inside the staging slot's read buffer of the part.
                    unsafe { self.stage_part(layer, stage, part).add(skip) }
                }
                (None, Some(ram_slot)) => {
                    let pool = self.ram.pool_of[layer];
                    let skip = self.ram_skip[pool][self.ram.slot(layer, ram_slot) as usize][part];
                    // SAFETY: inside the RAM slot's read buffer of the part.
                    unsafe { self.ram_part(layer, ram_slot, part).add(skip as usize) }
                }
                (None, None) => unreachable!("an upload has a source"),
            };
            // SAFETY: the source is the expert's bytes in a staging slot that
            // stays taken, or in its RAM slot that stays held, until the mark
            // below has been passed; the transit slot was free.
            unsafe {
                self.devices[g.device].upload(pool, slot, g.offsets[part], src, g.lens[part])?;
            }
            self.stats.transit_upload_bytes += g.lens[part] as u64;
        }
        let mark = self.devices[g.device].mark(false)?;
        self.transit.uploads[g.device].push_back((mark, key));
        if let Some(entry) = self.transit.entries.get_mut(&key) {
            entry.state = State::Uploading { slot, stage };
        }
        self.stats.transit_uploads += 1;
        self.stats.transit_from_ram += u64::from(stage.is_none());
        Ok(true)
    }

    /// An upload into transit has run on `device`.
    fn transit_uploaded(&mut self, device: usize, key: (usize, u32)) {
        let Some(entry) = self.transit.entries.get(&key).copied() else { return };
        let State::Uploading { slot, stage } = entry.state else { return };
        if stage.is_none() {
            self.ram.sets[key.0].release(key.1);
        }
        if entry.dropped {
            self.transit.entries.remove(&key);
            self.transit.free[device].push(slot);
            if let Some(stage) = stage {
                self.transit.free_stage(stage, false);
            }
            return;
        }
        self.stats.transit_ms += entry.since.elapsed().as_secs_f64() * 1e3;
        if let Some(entry) = self.transit.entries.get_mut(&key) {
            entry.state = State::Ready { slot, stage };
        }
    }

    /// Moves transit on: takes the uploads the devices have done, starts the
    /// next uploads and reads. Never waits.
    pub(super) fn poll_transit(&mut self) -> Result<(), StoreError> {
        if self.transit.entries.is_empty() && !self.transit.is_busy() {
            return Ok(());
        }
        for device in 0..self.devices.len() {
            while let Some((mark, key)) = self.transit.uploads[device].front().copied() {
                if !self.devices[device].passed(mark)? {
                    break;
                }
                self.transit.uploads[device].pop_front();
                self.transit_uploaded(device, key);
            }
            while let Some((mark, adoption)) = self.transit.adoptions[device].front().copied() {
                if !self.devices[device].passed(mark)? {
                    break;
                }
                self.transit.adoptions[device].pop_front();
                // The expert is usable in its slot of the tier, and what it
                // came through is free.
                self.vram.sets[adoption.layer].commit(&adoption.swap);
                self.worth_stale[adoption.layer] = true;
                self.transit.free[device].push(adoption.slot);
                if let Some(stage) = adoption.stage {
                    self.transit.free_stage(stage, false);
                }
            }
        }

        // What records named, in the order asked: an upload when its bytes are
        // in host memory and a transit slot is free, else a read.
        let mut at = 0;
        while at < self.transit.wanted.len() {
            let key = self.transit.wanted[at];
            let Some(entry) = self.transit.entries.get(&key).copied() else {
                self.transit.wanted.remove(at);
                continue;
            };
            let (layer, expert) = key;
            let started = match entry.state {
                State::Staged { stage } => self.transit_upload(key, Some(stage))?,
                State::Queued if self.ram.sets[layer].contains(expert) => {
                    self.transit_upload(key, None)?
                }
                State::Queued if self.ram.sets[layer].is_loading(expert) => {
                    // A read into the tier is on its way; it comes from there.
                    self.hurry(layer, expert);
                    false
                }
                State::Queued => {
                    self.transit_read(key, true)?;
                    false
                }
                State::Reading { .. } => false,
                State::Uploading { .. } | State::Ready { .. } | State::Taken { .. } => true,
            };
            if started {
                self.transit.wanted.remove(at);
            } else {
                at += 1;
            }
        }

        // Guesses, behind them.
        while let Some(key) = self.transit.guesses.front().copied() {
            let waits = self
                .transit
                .entries
                .get(&key)
                .is_some_and(|entry| !entry.wanted && entry.state == State::Queued);
            if waits && !self.transit_read(key, false)? {
                break;
            }
            self.transit.guesses.pop_front();
        }
        Ok(())
    }
}
