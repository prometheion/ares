//! Filling the tiers from the heat while no step runs: at the start, from a
//! usage profile that was saved by an earlier run (`profile.rs`), and
//! whenever the engine is idle, into the slots that are empty.
//!
//! **What is brought.** For every layer, VRAM's slots are made to hold the
//! layer's hottest experts: every free slot is filled, and a member gives
//! way to an expert hotter than itself (the hottest outsider takes the
//! coldest member's slot, and so on while the outsider is the hotter). Behind
//! them the next hottest that RAM does not hold, as many as RAM has room for:
//! a free slot or a slot whose member is worth nothing there (one without
//! heat, or a copy of an expert that VRAM holds); nothing in RAM that is worth
//! something is evicted for a fill. Nothing without heat is brought.
//!
//! **How.** An expert comes from the file into a RAM slot; one meant for VRAM
//! goes on from there into a slot of the layer's VRAM tier (a free one, or
//! the coldest member's, which is copied down to RAM when RAM does not hold
//! it), its table entry behind its bytes, and its RAM copy is then the first
//! to give way. A filled expert is a member like one a step used: it
//! competes with its heat, not as a guess that nobody used. While steps run,
//! VRAM changes only by promotions (`hybrid.rs`), which ask more of an
//! expert than being hotter.
//!
//! **When.** Only between steps: [`ExpertStore::advance_fill`] does nothing
//! while a step runs, so the owner stops a fill by beginning a step. What is
//! on its way then (a few reads, an upload per device at the pace of an idle
//! fill) arrives on its own and takes its slot; nothing else of the fill is
//! started. A fill reads nothing into RAM while the tier gives memory back,
//! and only into slots the tier holds now, so it neither fights a shrink nor
//! fills beyond what the tier may hold.
//!
//! **In what order** ([`FillOrder`]). At the start, before anything is
//! served, in the order of the file: the experts of a layer as they lie in
//! its tensors, one layer after the other. When idle, the hottest first, so
//! that a request that stops the fill finds the experts that matter most.

use std::collections::VecDeque;

use ares_weights::{Start, Want};

use super::{ExpertStore, Serve, SlotDevice, StoreError};

/// What a member of the RAM tier is worth at most and still counts as room:
/// a copy of an expert that VRAM holds is worth less than a millionth there
/// (`refresh_worth`), one that nobody used or that has no heat nothing.
const EMPTY_WORTH: f32 = 1e-6;

/// The order in which a fill brings its experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillOrder {
    /// Layer by layer as the layers lie in the file, and in each layer those
    /// meant for VRAM, then those meant for RAM, each as they lie in the
    /// layer's tensors. For a fill that runs to its end before anything is
    /// served. (The experts meant for VRAM pass through slots of the RAM
    /// tier, so they come first: those meant for RAM stay where they land.)
    File,
    /// The experts meant for VRAM first, then those meant for RAM, each the
    /// hottest first over all layers. For a fill that a step may stop.
    Heat,
}

/// How much of the drive and of the links a fill may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillPace {
    /// Experts that are read from the file at once.
    pub reads: usize,
    /// Uploads into VRAM slots on their way at once, per device (the store's
    /// own included).
    pub uploads: usize,
    /// Whether the reads go as demand reads. Otherwise they are prefetches,
    /// which the reader keeps behind every read a step waits for.
    pub demand: bool,
}

impl FillPace {
    /// Before anything is served: the drive and the links are the fill's.
    pub const START: FillPace = FillPace { reads: 32, uploads: 64, demand: true };
    /// While nothing runs: a step that begins finds two experts on their way
    /// from the file, as prefetches behind its own reads, and an upload per
    /// device.
    pub const IDLE: FillPace = FillPace { reads: 2, uploads: 1, demand: false };
}

/// The tier an expert of a fill is meant for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillTier {
    Vram,
    Ram,
}

#[derive(Clone, Copy, Debug)]
struct Item {
    layer: usize,
    expert: u32,
    tier: FillTier,
}

/// What a fill did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FillStats {
    /// Experts that took a free slot of VRAM.
    pub vram: u64,
    /// Experts that were read into the RAM tier and stay there.
    pub ram: u64,
    /// Experts read from the file, and the bytes of the reads.
    pub reads: u64,
    pub read_bytes: u64,
    /// Experts the fill gave up: a read failed, or the room was taken by
    /// something else meanwhile.
    pub dropped: u64,
}

/// A fill of the tiers from the heat ([`ExpertStore::plan_fill`]): the
/// experts it brings, in their order, and how far it got.
#[derive(Debug, Default)]
pub struct TierFill {
    todo: VecDeque<Item>,
    /// Experts whose read into the RAM tier has started.
    reading: Vec<Item>,
    /// Experts meant for VRAM whose bytes are in RAM, waiting for an upload.
    ready: VecDeque<Item>,
    stats: FillStats,
}

impl TierFill {
    /// Whether nothing is left to do and nothing of the fill is on its way.
    pub fn is_done(&self) -> bool {
        self.todo.is_empty() && self.reading.is_empty() && self.ready.is_empty()
    }

    /// The experts the fill has not finished with.
    pub fn left(&self) -> usize {
        self.todo.len() + self.reading.len() + self.ready.len()
    }

    /// The experts the fill has not started on, in the order it brings them:
    /// (layer, expert, the tier it is meant for).
    pub fn todo(&self) -> Vec<(usize, u32, FillTier)> {
        self.todo.iter().map(|item| (item.layer, item.expert, item.tier)).collect()
    }

    pub fn stats(&self) -> FillStats {
        self.stats
    }
}

impl<D: SlotDevice> ExpertStore<D> {
    /// Sets the heat of `layer`: from a usage profile, before the tiers are
    /// filled from it.
    ///
    /// # Panics
    ///
    /// When `heat` does not have a value for every expert of the layer.
    pub fn set_heat(&mut self, layer: usize, heat: &[f32]) {
        self.heat.layer_mut(layer).copy_from_slice(heat);
        self.worth_stale[layer] = true;
    }

    /// The store's count of steps: it moves at the end of every step (and
    /// every wave of a prompt pass) and at nothing else.
    pub fn clock(&self) -> u64 {
        self.step
    }

    /// The experts `layer` holds in VRAM and in the RAM tier, whose bytes are
    /// in their slots.
    pub fn held(&self, layer: usize) -> (usize, usize) {
        (self.vram.sets[layer].len(), self.ram.sets[layer].len())
    }

    /// Whether no read from the file and no copy to or from a device is on
    /// its way: what a fill started has all arrived.
    pub fn is_quiet(&self) -> bool {
        self.reads.is_empty() && self.device_ops.iter().flatten().all(VecDeque::is_empty)
    }

    /// What a fill of `layers` from the heat would bring now, in `order`: for
    /// every layer the hottest experts that VRAM does not hold, into its free
    /// slots and in place of the members colder than they are, and the next
    /// hottest that RAM does not hold, as many as RAM has room for (a free
    /// slot or one whose member is worth nothing there: one without heat, or
    /// a copy of an expert that VRAM holds); nothing without heat is brought.
    /// Nothing starts before [`ExpertStore::advance_fill`].
    pub fn plan_fill(&mut self, layers: &[usize], order: FillOrder) -> TierFill {
        // Slots that stay free in VRAM for the misses of a step, where the
        // store keeps them free between steps.
        let keep = if self.serve() == Serve::Stream { self.config.vram_reserve } else { 0 };
        let mut items: Vec<(Item, f32)> = Vec::new();
        for &layer in layers {
            self.refresh_worth(layer);
            let heat = self.heat.layer(layer);
            let worth = self.ram_worth.layer(layer);
            let (vram, ram) = (&self.vram.sets[layer], &self.ram.sets[layer]);
            let mut outsiders: Vec<u32> = (0..heat.len() as u32)
                .filter(|e| heat[*e as usize] > 0.0 && !vram.contains(*e) && !vram.is_loading(*e))
                .collect();
            outsiders
                .sort_by(|a, b| heat[*b as usize].total_cmp(&heat[*a as usize]).then(a.cmp(b)));
            // Room in VRAM: free slots (less those kept free), and then the
            // members, the coldest first, each for an outsider hotter than it
            // (the hottest outsider first), such as what the steps that
            // measured the price of a step brought.
            let members = vram.room_worths(self.step, heat);
            let free = vram.n_free().saturating_sub(keep);
            let to_vram = outsiders
                .iter()
                .enumerate()
                .take_while(|(rank, e)| {
                    *rank < free
                        || members.get(rank - free).is_some_and(|m| heat[**e as usize] > *m)
                })
                .count();
            // Room in RAM: free slots, members worth nothing there, and the
            // members that go up to VRAM now (their copies are then worth
            // nothing). Those meant for VRAM that are read pass through a slot
            // of that room and leave it as such a copy.
            let worthless = (0..heat.len() as u32)
                .filter(|e| ram.contains(*e) && worth[*e as usize] <= EMPTY_WORTH)
                .count();
            let going_up = outsiders[..to_vram].iter().filter(|e| ram.contains(**e)).count();
            let mut room = ram.n_free() + worthless + going_up;
            for (rank, expert) in outsiders.iter().enumerate() {
                let tier = if rank < to_vram {
                    FillTier::Vram
                } else if room > 0 && !ram.contains(*expert) && !ram.is_loading(*expert) {
                    room -= 1;
                    FillTier::Ram
                } else {
                    continue;
                };
                items.push((Item { layer, expert: *expert, tier }, heat[*expert as usize]));
            }
        }
        match order {
            FillOrder::File => {
                // Where an expert's first matrix lies in the files; the
                // layer's place is its first expert's, which the layers of
                // the store that share the model's layer share.
                let place = |layer: usize, expert: u32| {
                    self.map
                        .ranges(layer, expert as usize)
                        .and_then(|mut ranges| ranges.next())
                        .map_or((u16::MAX, u64::MAX), |range| (range.file.0, range.offset))
                };
                items.sort_by_key(|(item, _)| {
                    (
                        place(item.layer, 0),
                        item.tier == FillTier::Ram,
                        place(item.layer, item.expert),
                    )
                });
            }
            FillOrder::Heat => items.sort_by(|(a, heat_a), (b, heat_b)| {
                (a.tier == FillTier::Ram)
                    .cmp(&(b.tier == FillTier::Ram))
                    .then(heat_b.total_cmp(heat_a))
                    .then((a.layer, a.expert).cmp(&(b.layer, b.expert)))
            }),
        }
        TierFill { todo: items.into_iter().map(|(item, _)| item).collect(), ..TierFill::default() }
    }

    /// Takes a fill a step further, as far as `pace` allows: hands what has
    /// arrived on, starts uploads and reads. Never waits. Does nothing while
    /// a step runs; call it again between steps, or drop the fill.
    ///
    /// # Errors
    ///
    /// A read that cannot be submitted or a copy that cannot be enqueued.
    pub fn advance_fill(&mut self, fill: &mut TierFill, pace: FillPace) -> Result<(), StoreError> {
        if self.in_step {
            return Ok(());
        }
        self.poll()?;

        // Reads that have landed.
        let mut at = 0;
        while at < fill.reading.len() {
            let item = fill.reading[at];
            let ram = &self.ram.sets[item.layer];
            if ram.is_loading(item.expert) {
                at += 1;
                continue;
            }
            fill.reading.swap_remove(at);
            if !ram.contains(item.expert) {
                // The read failed.
                fill.stats.dropped += 1;
            } else if item.tier == FillTier::Vram {
                fill.ready.push_back(item);
            } else {
                fill.stats.ram += 1;
            }
        }

        // Uploads into VRAM: a free slot, or the slot of a member colder than
        // the expert.
        let keep = if self.serve() == Serve::Stream { self.config.vram_reserve } else { 0 };
        let mut waiting = VecDeque::new();
        while let Some(item) = fill.ready.pop_front() {
            let (layer, expert) = (item.layer, item.expert);
            let device = self.layers[layer].device;
            if self.vram.sets[layer].contains(expert) || self.vram.sets[layer].is_loading(expert) {
                continue;
            }
            if !self.ram.sets[layer].contains(expert) {
                // Gone from RAM meanwhile: a step took its slot.
                fill.stats.dropped += 1;
                continue;
            }
            let (vram, heat) = (&self.vram.sets[layer], self.heat.layer(layer));
            let room = vram.n_free() > keep
                || vram
                    .room_worths(self.step, heat)
                    .first()
                    .is_some_and(|coldest| *coldest < heat[expert as usize]);
            if !room {
                // Something else took the room: it stays in RAM.
                fill.stats.ram += 1;
                continue;
            }
            if self.device_ops[device][0].len() >= pace.uploads {
                waiting.push_back(item);
                continue;
            }
            if self.upload(layer, expert, Want::Demand, true)? {
                // The upload marked it as needed by a step; no step needs it.
                self.vram.sets[layer].unprotect(expert);
                fill.stats.vram += 1;
            } else {
                fill.stats.ram += 1;
            }
        }
        fill.ready = waiting;

        // Reads, as far as the pace and the room allow.
        while fill.reading.len() < pace.reads {
            let Some(item) = fill.todo.front().copied() else { break };
            let (layer, expert) = (item.layer, item.expert);
            let (vram, ram) = (&self.vram.sets[layer], &self.ram.sets[layer]);
            let there = match item.tier {
                FillTier::Vram => vram.contains(expert) || vram.is_loading(expert),
                FillTier::Ram => vram.contains(expert) || ram.contains(expert),
            };
            if there {
                fill.todo.pop_front();
                continue;
            }
            if ram.is_loading(expert) {
                // Someone else's read: the fill takes what it brings.
                fill.todo.pop_front();
                fill.reading.push(item);
                continue;
            }
            if ram.contains(expert) {
                // Meant for VRAM and in RAM already: an upload away.
                fill.todo.pop_front();
                fill.ready.push_back(item);
                continue;
            }
            // Nothing is read into the tier while it gives memory back, or
            // holds more than it may and has something left to give.
            let over = self.region.locked_bytes() as u64 > self.ram_target();
            if self.pieces.shrinking || (over && !self.ram_at_floor()) {
                break;
            }
            self.refresh_worth(layer);
            let worth = self.ram_worth.layer(layer);
            let ram = &self.ram.sets[layer];
            let room = ram.n_free() > 0
                || matches!(ram.victim(self.step, worth), Some((_, w, false)) if w <= EMPTY_WORTH);
            if !room {
                if fill.reading.is_empty() && fill.ready.is_empty() && self.is_quiet() {
                    // Nothing of the fill will make room in this layer.
                    fill.todo.pop_front();
                    fill.stats.dropped += 1;
                    continue;
                }
                break;
            }
            let started = self.ram.sets[layer].start_read(expert, Want::Demand, self.step, worth);
            let Start::Slot(swap) = started else {
                fill.todo.pop_front();
                fill.stats.dropped += 1;
                continue;
            };
            // A member worth its heat, which no step needs.
            self.ram.sets[layer].unprotect(expert);
            let before = self.stats.demand_read_bytes + self.stats.guess_read_bytes;
            self.read(layer, swap, pace.demand)?;
            fill.stats.reads += 1;
            fill.stats.read_bytes +=
                self.stats.demand_read_bytes + self.stats.guess_read_bytes - before;
            fill.todo.pop_front();
            fill.reading.push(item);
        }
        Ok(())
    }
}
