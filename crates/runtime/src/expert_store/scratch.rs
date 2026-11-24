//! VRAM lent by the expert slots: the slots at the end of a device's pool
//! step aside for a while, and their memory holds what a step needs beside
//! them (the scratch of a step wider than the one the model keeps for good).
//!
//! **Which slots.** Always the last slots of the device's pools, the largest
//! first, so a loan of the same size lies at the same addresses every time, and
//! what another device writes into it can be named once. A loan is a piece of
//! each pool it takes (a pool's slots lie in one allocation, the pools apart),
//! so its borrower lays its memory out in pieces ([`ExpertStore::lendable_pieces`]).
//! Those slots belong to
//! several layers and hold whatever was read into them, so they are not taken
//! as they lie: every layer of the pool gives slots in proportion to what it
//! has above the fewest a layer keeps during a loan (a row's picks,
//! [`ExpertStore::scratch_floor`]), each layer gives its coldest,
//! what they held is evicted as any evicted expert is (copied down to RAM when
//! the RAM tier keeps it), and the experts that lie in the lent slots and stay
//! are moved on the device into the slots that were given, whose numbers then
//! change hands (as the RAM tier does when it gives pieces back, `elastic.rs`).
//!
//! **When.** Between steps, once no upload, copy or read touches the device's
//! slots ([`ExpertStore::lend_pieces`] says `None` until then), and the memory
//! is the borrower's when [`ExpertStore::scratch_ready`] says so: the moves and
//! the copies down have run. Given back between steps as well, when nothing
//! the borrower enqueued still uses it ([`ExpertStore::return_scratch`]): the
//! slots go back to the layers that hold their numbers, empty, and fill again
//! the way empty slots do (the promotions of hot experts from RAM at the end
//! of every step, the fill while idle, what a step uploads).

use std::collections::{HashMap, HashSet};

use super::tier::Move;
use super::{ExpertStore, SlotDevice, StoreError};

/// Slots of a device's pool that are lent: `slots` slots from `first` on, the
/// last of the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLoan {
    /// The pool among the device's pools.
    pub pool: usize,
    pub first: u32,
    pub slots: u32,
    /// The bytes from one slot to the next.
    pub stride: u64,
}

impl ScratchLoan {
    /// The bytes the loan holds.
    pub fn bytes(&self) -> u64 {
        u64::from(self.slots) * self.stride
    }
}

/// A loan of a device and the marks of the device's two kinds of work behind
/// which its memory is free.
#[derive(Clone, Debug)]
pub(super) struct LentScratch {
    /// The pieces lent so far, each with the tier's pool of its slots, in the
    /// order of the pools a loan takes; and the bytes asked of each.
    loans: Vec<(ScratchLoan, usize)>,
    asked: Vec<u64>,
    marks: (u64, u64),
    ready: bool,
}

impl LentScratch {
    /// Every piece asked for is lent.
    fn whole(&self) -> bool {
        self.loans.len() == self.asked.iter().filter(|bytes| **bytes > 0).count()
    }
}

impl<D: SlotDevice> ExpertStore<D> {
    /// The fewest slots a layer keeps in VRAM while a loan runs: a row's picks, or the
    /// layer's floor where that is lower. Not the free slots a layer keeps for experts that
    /// arrive before it runs: only a prompt read alone runs on a loan, and the misses of its
    /// wide steps go through transit, not into the tier's slots; the steps that run beside a
    /// loan (a generating sequence's while a prompt read layer by layer is paused, or before
    /// a loan has gone back) upload where a slot their rows did not pick is free and leave the
    /// others to the CPU, as any step on a layer with fewer slots than it picks does. Every
    /// slot above that is VRAM a prompt's step can be wider by, and each step reads again from
    /// the file what the tiers do not hold.
    pub fn scratch_floor(&self, layer: usize) -> usize {
        (self.config.top_k as usize).min(self.vram.floor[layer])
    }

    /// The pools of the tier a loan of `device` takes its slots from, with the
    /// layers each serves: those of the device whose layers can give slots,
    /// the one with the most bytes first.
    fn scratch_pools(&self, device: usize) -> Vec<(usize, Vec<usize>)> {
        let pools = &self.vram.pools;
        let mut found: Vec<(usize, Vec<usize>)> = (0..pools.len())
            .filter(|p| pools[*p].place == device)
            .map(|p| {
                let members: Vec<usize> = (0..self.layers.len())
                    .filter(|l| self.vram.pool_of[*l] == p && !self.layers[*l].resident)
                    .collect();
                (p, members)
            })
            .filter(|(_, members)| !members.is_empty())
            .collect();
        found
            .sort_by_key(|(p, _)| std::cmp::Reverse(pools[*p].stride * u64::from(pools[*p].slots)));
        found
    }

    /// The most bytes each pool of `device` can lend, in the order of the
    /// pieces of a loan: whole slots of the pool above the floor of a loan of
    /// each of its layers.
    pub fn lendable_pieces(&self, device: usize) -> Vec<u64> {
        self.scratch_pools(device)
            .iter()
            .map(|(pool, members)| {
                let spare: usize = members
                    .iter()
                    .map(|l| self.vram.slots(*l).saturating_sub(self.scratch_floor(*l)))
                    .sum();
                spare as u64 * self.vram.pools[*pool].stride
            })
            .collect()
    }

    /// Where a loan of `pieces` of `device` lies (bytes of each pool, in the
    /// order of [`ExpertStore::lendable_pieces`]), made or not: always the
    /// same slots for the same bytes, a loan per piece that is not empty.
    /// `None` when a pool cannot give its piece.
    pub fn scratch_pieces_at(&self, device: usize, pieces: &[u64]) -> Option<Vec<ScratchLoan>> {
        let pools = self.scratch_pools(device);
        let lendable = self.lendable_pieces(device);
        if pieces.len() > pools.len() || pieces.iter().zip(&lendable).any(|(b, l)| b > l) {
            return None;
        }
        let loans: Vec<ScratchLoan> = pieces
            .iter()
            .zip(&pools)
            .filter(|(bytes, _)| **bytes > 0)
            .map(|(bytes, (tier_pool, members))| {
                let pool = self.vram.pools[*tier_pool];
                let n = bytes.div_ceil(pool.stride) as u32;
                ScratchLoan {
                    pool: self.layers[members[0]].device_pool,
                    first: pool.slots - n,
                    slots: n,
                    stride: pool.stride,
                }
            })
            .collect();
        (!loans.is_empty()).then_some(loans)
    }

    /// Whether `device` has lent VRAM, ready or not.
    pub fn has_scratch(&self, device: usize) -> bool {
        self.scratch.get(device).is_some_and(Option::is_some)
    }

    /// The pieces of the loan of `device`, once it is ready.
    pub fn scratch_loans(&self, device: usize) -> Option<Vec<ScratchLoan>> {
        let lent = self.scratch.get(device)?.as_ref().filter(|l| l.ready)?;
        Some(lent.loans.iter().map(|(loan, _)| *loan).collect())
    }

    /// Whether nothing touches the VRAM slots of `device` now: no expert is
    /// wanted or read, nothing is on its way to or from the device.
    fn vram_quiet(&self, device: usize) -> bool {
        self.wants.is_empty()
            && self.reads.is_empty()
            && self.device_ops[device].iter().all(|ops| ops.is_empty())
            && !self.transit.is_busy()
    }

    /// Lends VRAM of `device`, between steps: `pieces` bytes of its pools, in
    /// the order of [`ExpertStore::lendable_pieces`]; the slots at the end of
    /// each pool leave the layers (see "Large prompt steps on lent VRAM" in
    /// `docs/ARCHITECTURE.md`). `None` while something still touches the
    /// device's slots (poll the store and ask again: the pieces lent so far
    /// stay lent) or when a pool cannot give its piece. A device lends once at
    /// a time; a device that has lent at least `pieces` keeps its loan. The
    /// memory is the caller's when [`ExpertStore::scratch_ready`] says so.
    pub fn lend_pieces(
        &mut self,
        device: usize,
        pieces: &[u64],
    ) -> Result<Option<Vec<ScratchLoan>>, StoreError> {
        if self.in_step {
            return Err(StoreError::Plan("VRAM is lent between steps only".into()));
        }
        let pieces: Vec<u64> = {
            let used = pieces.iter().rposition(|b| *b > 0).map_or(0, |at| at + 1);
            pieces[..used].to_vec()
        };
        if let Some(lent) = self.scratch.get(device).and_then(Option::as_ref) {
            let covers = lent.asked.len() >= pieces.len()
                && pieces.iter().zip(&lent.asked).all(|(want, asked)| want <= asked);
            if !covers {
                return Err(StoreError::Plan(format!(
                    "device {device} has lent {:?} bytes and is asked for {pieces:?}",
                    lent.asked
                )));
            }
            if lent.whole() {
                return Ok(Some(lent.loans.iter().map(|(loan, _)| *loan).collect()));
            }
        }
        let Some(loans) = self.scratch_pieces_at(device, &pieces) else { return Ok(None) };
        if !self.vram_quiet(device) {
            return Ok(None);
        }
        let pools = self.scratch_pools(device);
        // The pools of the pieces that are not empty, in the order of the loans.
        let of: Vec<usize> = (0..pieces.len()).filter(|k| pieces[*k] > 0).collect();
        let done = self.scratch[device].as_ref().map_or(0, |lent| lent.loans.len());
        let asked = match &self.scratch[device] {
            Some(lent) => lent.asked.clone(),
            None => pieces.clone(),
        };
        for (loan, k) in loans.iter().zip(&of).skip(done) {
            let (tier_pool, members) = pools[*k].clone();
            if !self.lend_pool(device, tier_pool, &members, loan.first)? {
                // A slot was busy: the pieces lent so far wait for the next try.
                return Ok(None);
            }
            match &mut self.scratch[device] {
                Some(lent) => lent.loans.push((*loan, tier_pool)),
                None => {
                    self.scratch[device] = Some(LentScratch {
                        loans: vec![(*loan, tier_pool)],
                        asked: asked.clone(),
                        marks: (0, 0),
                        ready: false,
                    });
                }
            }
        }
        let marks = (self.devices[device].mark(false)?, self.devices[device].mark(true)?);
        if let Some(lent) = &mut self.scratch[device] {
            lent.marks = marks;
            lent.ready = false;
        }
        self.stats.scratch_loans += 1;
        Ok(Some(loans))
    }

    /// Lends the slots of `tier_pool` (whose layers are `members`) from
    /// `first` to its end ([`Tier::vacate`](super::tier::Tier::vacate)): every
    /// layer gives its coldest in proportion to what it has above the floor of
    /// a loan, what they held is evicted, and what lies in the lent slots and
    /// stays moves into the slots given. `false` when a slot was busy after
    /// all; nothing is lent then.
    fn lend_pool(
        &mut self,
        device: usize,
        tier_pool: usize,
        members: &[usize],
        first: u32,
    ) -> Result<bool, StoreError> {
        let pool = self.vram.pools[tier_pool];
        let lent: Vec<u32> = (first..pool.slots).collect();
        let floor: Vec<usize> = members.iter().map(|l| self.scratch_floor(*l)).collect();
        let vacated = self.vram.vacate(members, &lent, &floor, self.step, |l| self.heat.layer(l));
        // What the slots given held leaves as any evicted expert does.
        for (layer, expert, slot) in vacated.evicted {
            self.stats.evictions += 1;
            self.devices[device].set_entry(layer, expert, -1)?;
            self.demote(layer, expert, slot)?;
        }
        for layer in members {
            self.worth_stale[*layer] = true;
        }
        let Some(moves) = vacated.moves else { return Ok(false) };
        // The experts that lay in the lent slots move on the device.
        let mut held: HashMap<usize, HashMap<u32, u32>> = HashMap::new();
        for Move { layer, number, from, to } in moves {
            let held = held.entry(layer).or_insert_with(|| {
                let set = &self.vram.sets[layer];
                (0..self.heat.layer(layer).len() as u32)
                    .filter_map(|e| set.slot(e).map(|local| (local, e)))
                    .collect()
            });
            if let Some(expert) = held.get(&number).copied() {
                let g = self.layers[layer];
                let parts = [0, 1, 2].map(|part| (g.offsets[part], g.lens[part]));
                self.devices[device].adopt(
                    g.device_pool,
                    from,
                    g.device_pool,
                    to,
                    &parts,
                    layer,
                    expert,
                )?;
                self.stats.scratch_moves += 1;
            }
        }
        // The numbers behind the lent slots stay given back until the loan
        // ends, whatever the layers take meanwhile.
        let in_loan: HashSet<u32> = lent.iter().copied().collect();
        for layer in members {
            let locals: Vec<u32> = (0..self.vram.locals(*layer).len() as u32)
                .filter(|local| in_loan.contains(&self.vram.slot(*layer, *local)))
                .collect();
            for local in locals {
                let lent = self.vram.sets[*layer].set_aside(local);
                debug_assert!(lent, "a lent slot of layer {layer} was not given back");
            }
        }
        Ok(true)
    }

    /// Whether the memory of the loan of `device` is the borrower's: the
    /// moves out of its slots and the copies down have run. Polls the device.
    pub fn scratch_ready(&mut self, device: usize) -> Result<bool, StoreError> {
        let Some(lent) = self.scratch.get(device).and_then(Option::as_ref) else {
            return Ok(false);
        };
        if !lent.whole() {
            return Ok(false);
        }
        if !lent.ready {
            let marks = lent.marks;
            let d = &mut self.devices[device];
            let ready = d.passed(marks.0)? && d.passed(marks.1)?;
            if let Some(lent) = &mut self.scratch[device] {
                lent.ready = ready;
            }
            return Ok(ready);
        }
        Ok(true)
    }

    /// Gives the loan of `device` back: its slots go back to the layers that
    /// hold their numbers, empty. Between steps, once nothing the borrower
    /// enqueued uses the memory any more.
    pub fn return_scratch(&mut self, device: usize) -> Result<(), StoreError> {
        if self.in_step {
            return Err(StoreError::Plan("VRAM is given back between steps only".into()));
        }
        let Some(lent) = self.scratch.get_mut(device).and_then(Option::take) else { return Ok(()) };
        for (loan, tier_pool) in lent.loans {
            let back: HashSet<u32> = (loan.first..loan.first + loan.slots).collect();
            for layer in self.vram.reopen(tier_pool, &back) {
                self.worth_stale[layer] = true;
            }
        }
        // The slots are empty: they fill with the hottest experts of RAM now, not at the end
        // of the next step (every free slot at once, `promote_hot`), so that the tokens after
        // a long prompt find what the prompt's loan took from them as soon as the link has
        // brought it.
        if self.hybrid.is_some() {
            self.promote_hot()?;
        }
        Ok(())
    }
}
