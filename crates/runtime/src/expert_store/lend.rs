//! VRAM lent by the expert slots: the slots at the end of a device's pool
//! step aside for a while, and their memory holds what a step needs beside
//! them (the scratch of a step wider than the one the model keeps for good).
//!
//! **Which slots.** Always the last slots of the device's largest pool, so a
//! loan of the same size lies at the same addresses every time, and what
//! another device writes into it can be named once. Those slots belong to
//! several layers and hold whatever was read into them, so they are not taken
//! as they lie: every layer of the pool gives slots in proportion to what it
//! has above the fewest a layer keeps during a loan (a row's picks and the
//! free slots, [`ExpertStore::loan_floor`]), each layer gives its coldest,
//! what they held is evicted as any evicted expert is (copied down to RAM when
//! the RAM tier keeps it), and the experts that lie in the lent slots and stay
//! are moved on the device into the slots that were given, whose numbers then
//! change hands (as the RAM tier does when it gives pieces back, `elastic.rs`).
//!
//! **When.** Between steps, once no upload, copy or read touches the device's
//! slots ([`ExpertStore::lend_vram`] says `None` until then), and the memory
//! is the borrower's when [`ExpertStore::loan_ready`] says so: the moves and
//! the copies down have run. Given back between steps as well, when nothing
//! the borrower enqueued still uses it ([`ExpertStore::return_vram`]): the
//! slots go back to the layers that hold their numbers, empty, and fill again
//! the way empty slots do (the promotions of hot experts from RAM at the end
//! of every step, the fill while idle, what a step uploads).

use std::collections::{HashMap, HashSet};

use super::{ExpertStore, SlotDevice, StoreError};

/// Slots of a device's pool that are lent: `slots` slots from `first` on, the
/// last of the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loan {
    /// The pool among the device's pools.
    pub pool: usize,
    pub first: u32,
    pub slots: u32,
    /// The bytes from one slot to the next.
    pub stride: u64,
}

impl Loan {
    /// The bytes the loan holds.
    pub fn bytes(&self) -> u64 {
        u64::from(self.slots) * self.stride
    }
}

/// A loan of a device and the marks of the device's two kinds of work behind
/// which its memory is free.
#[derive(Clone, Copy, Debug)]
pub(super) struct Lent {
    loan: Loan,
    /// The tier's pool of the loan's slots.
    tier_pool: usize,
    marks: (u64, u64),
    ready: bool,
}

impl<D: SlotDevice> ExpertStore<D> {
    /// The fewest slots a layer keeps in VRAM while a loan runs: a row's picks
    /// and the free slots, or the layer's floor where that is lower.
    pub fn loan_floor(&self, layer: usize) -> usize {
        (self.config.top_k as usize + self.config.vram_reserve).min(self.vram.floor[layer])
    }

    /// The pool of the tier a loan of `device` takes its slots from: the one
    /// with the most bytes on that device, with the layers it serves; `None`
    /// when the device has no pool whose layers can give slots.
    fn loan_pool(&self, device: usize) -> Option<(usize, Vec<usize>)> {
        let pools = &self.vram.pools;
        (0..pools.len())
            .filter(|p| pools[*p].place == device)
            .map(|p| {
                let members: Vec<usize> = (0..self.layers.len())
                    .filter(|l| self.vram.pool_of[*l] == p && !self.layers[*l].resident)
                    .collect();
                (p, members)
            })
            .filter(|(_, members)| !members.is_empty())
            .max_by_key(|(p, _)| pools[*p].stride * u64::from(pools[*p].slots))
    }

    /// The most bytes `device` can lend: whole slots of its largest pool above
    /// the floor of a loan of each layer of the pool.
    pub fn lendable(&self, device: usize) -> u64 {
        let Some((pool, members)) = self.loan_pool(device) else { return 0 };
        let spare: usize =
            members.iter().map(|l| self.vram.slots(*l).saturating_sub(self.loan_floor(*l))).sum();
        spare as u64 * self.vram.pools[pool].stride
    }

    /// Where a loan of `bytes` of `device` lies, made or not: always the same
    /// slots for the same bytes. `None` when the layers cannot give that much.
    pub fn loan_at(&self, device: usize, bytes: u64) -> Option<Loan> {
        if bytes == 0 || bytes > self.lendable(device) {
            return None;
        }
        let (tier_pool, members) = self.loan_pool(device)?;
        let pool = self.vram.pools[tier_pool];
        let n = bytes.div_ceil(pool.stride) as u32;
        let device_pool = self.layers[members[0]].device_pool;
        Some(Loan { pool: device_pool, first: pool.slots - n, slots: n, stride: pool.stride })
    }

    /// Whether `device` has lent VRAM, ready or not.
    pub fn has_lent(&self, device: usize) -> bool {
        self.loans.get(device).is_some_and(Option::is_some)
    }

    /// The loan of `device`, once it is ready.
    pub fn loan(&self, device: usize) -> Option<Loan> {
        self.loans.get(device).copied().flatten().filter(|l| l.ready).map(|l| l.loan)
    }

    /// Whether nothing touches the VRAM slots of `device` now: no expert is
    /// wanted or read, nothing is on its way to or from the device.
    fn vram_quiet(&self, device: usize) -> bool {
        self.wants.is_empty()
            && self.reads.is_empty()
            && self.device_ops[device].iter().all(|ops| ops.is_empty())
            && !self.transit.is_busy()
    }

    /// Lends `bytes` of VRAM of `device`, between steps: the slots at the end
    /// of its largest pool leave the layers (see the [module
    /// documentation](self)). `None` while something still touches the
    /// device's slots (poll the store and ask again) or when the layers cannot
    /// give that much ([`ExpertStore::lendable`]). A device lends once at a
    /// time; a device that has lent at least `bytes` keeps its loan. The
    /// memory is the caller's when [`ExpertStore::loan_ready`] says so.
    pub fn lend_vram(&mut self, device: usize, bytes: u64) -> Result<Option<Loan>, StoreError> {
        if self.in_step {
            return Err(StoreError::Plan("VRAM is lent between steps only".into()));
        }
        if let Some(lent) = self.loans.get(device).copied().flatten() {
            if lent.loan.bytes() >= bytes {
                return Ok(Some(lent.loan));
            }
            return Err(StoreError::Plan(format!(
                "device {device} has lent {} bytes and is asked for {bytes}",
                lent.loan.bytes()
            )));
        }
        if bytes == 0 || bytes > self.lendable(device) {
            return Ok(None);
        }
        if !self.vram_quiet(device) {
            return Ok(None);
        }
        let Some((tier_pool, members)) = self.loan_pool(device) else { return Ok(None) };
        let pool = self.vram.pools[tier_pool];
        let Loan { first, slots: n, .. } =
            self.loan_at(device, bytes).expect("a loan the layers can give");
        let lent: Vec<u32> = (first..pool.slots).collect();
        let spare: Vec<usize> = members
            .iter()
            .map(|l| self.vram.slots(*l).saturating_sub(self.loan_floor(*l)))
            .collect();
        let total: usize = spare.iter().sum();
        let mut quota: Vec<usize> = spare.iter().map(|s| s * lent.len() / total).collect();
        for _ in quota.iter().sum::<usize>()..lent.len() {
            let most = (0..members.len())
                .filter(|i| spare[*i] > quota[*i])
                .max_by_key(|i| spare[*i] - quota[*i])
                .expect("the layers have the spare slots that were counted");
            quota[most] += 1;
        }
        // Every layer gives its coldest; what they held leaves as any evicted
        // expert does.
        let mut given: Vec<(usize, u32)> = Vec::with_capacity(lent.len());
        for (layer, n) in members.iter().zip(&quota) {
            if *n == 0 {
                continue;
            }
            for slot in self.vram.sets[*layer].give_back(*n, self.step, self.heat.layer(*layer)) {
                if let Some(expert) = slot.evicted {
                    self.stats.evictions += 1;
                    self.devices[device].set_entry(*layer, expert, -1)?;
                    self.demote(*layer, expert, self.vram.slot(*layer, slot.slot))?;
                }
                given.push((*layer, slot.slot));
            }
            self.worth_stale[*layer] = true;
        }
        if given.len() < lent.len() {
            // A slot was busy after all: the layers take theirs back.
            for layer in &members {
                let back: Vec<u32> = given
                    .iter()
                    .filter(|(l, _)| l == layer)
                    .map(|(_, local)| self.vram.slot(*layer, *local))
                    .collect();
                self.vram.retake(*layer, &back);
            }
            return Ok(None);
        }
        // What lies in the lent slots and stays moves into the slots given.
        let owner: HashMap<u32, (usize, u32)> = members
            .iter()
            .flat_map(|layer| {
                let locals = self.vram.locals(*layer).iter().enumerate();
                locals.map(move |(local, slot)| (*slot, (*layer, local as u32)))
            })
            .collect();
        let in_loan: HashSet<u32> = lent.iter().copied().collect();
        let given: Vec<(usize, u32, u32)> = given
            .into_iter()
            .map(|(layer, local)| (layer, local, self.vram.slot(layer, local)))
            .collect();
        let gone: HashSet<u32> = given.iter().map(|(_, _, slot)| *slot).collect();
        let mut homes = given.iter().filter(|(_, _, slot)| !in_loan.contains(slot));
        let mut held: HashMap<usize, HashMap<u32, u32>> = HashMap::new();
        for slot in lent.iter().filter(|slot| !gone.contains(slot)) {
            let (stays, number) = owner[slot];
            let (gives, given_number, home) =
                *homes.next().expect("as many slots were given as are lent");
            let held = held.entry(stays).or_insert_with(|| {
                let set = &self.vram.sets[stays];
                (0..self.heat.layer(stays).len() as u32)
                    .filter_map(|e| set.slot(e).map(|local| (local, e)))
                    .collect()
            });
            if let Some(expert) = held.get(&number).copied() {
                let g = self.layers[stays];
                let parts = [0, 1, 2].map(|part| (g.offsets[part], g.lens[part]));
                self.devices[device].adopt(
                    g.device_pool,
                    *slot,
                    g.device_pool,
                    home,
                    &parts,
                    stays,
                    expert,
                )?;
                self.stats.loan_moves += 1;
            }
            self.vram.remap(stays, number, home);
            self.vram.remap(gives, given_number, *slot);
        }
        // The numbers behind the lent slots stay given back until the loan
        // ends, whatever the layers take meanwhile.
        for layer in &members {
            let locals: Vec<u32> = (0..self.vram.locals(*layer).len() as u32)
                .filter(|local| in_loan.contains(&self.vram.slot(*layer, *local)))
                .collect();
            for local in locals {
                let lent = self.vram.sets[*layer].lend(local);
                debug_assert!(lent, "a lent slot of layer {layer} was not given back");
            }
        }
        let marks = (self.devices[device].mark(false)?, self.devices[device].mark(true)?);
        let device_pool = self.layers[members[0]].device_pool;
        let loan = Loan { pool: device_pool, first, slots: n, stride: pool.stride };
        self.loans[device] = Some(Lent { loan, tier_pool, marks, ready: false });
        self.stats.loans += 1;
        Ok(Some(loan))
    }

    /// Whether the memory of the loan of `device` is the borrower's: the
    /// moves out of its slots and the copies down have run. Polls the device.
    pub fn loan_ready(&mut self, device: usize) -> Result<bool, StoreError> {
        let Some(mut lent) = self.loans.get(device).copied().flatten() else { return Ok(false) };
        if !lent.ready {
            let d = &mut self.devices[device];
            lent.ready = d.passed(lent.marks.0)? && d.passed(lent.marks.1)?;
            self.loans[device] = Some(lent);
        }
        Ok(lent.ready)
    }

    /// Gives the loan of `device` back: its slots go back to the layers that
    /// hold their numbers, empty. Between steps, once nothing the borrower
    /// enqueued uses the memory any more.
    pub fn return_vram(&mut self, device: usize) -> Result<(), StoreError> {
        if self.in_step {
            return Err(StoreError::Plan("VRAM is given back between steps only".into()));
        }
        let Some(lent) = self.loans.get_mut(device).and_then(Option::take) else { return Ok(()) };
        let loan = lent.loan;
        let back: HashSet<u32> = (loan.first..loan.first + loan.slots).collect();
        for layer in 0..self.layers.len() {
            if self.vram.pool_of[layer] != lent.tier_pool {
                continue;
            }
            let own: Vec<(u32, u32)> = (0..self.vram.locals(layer).len() as u32)
                .map(|local| (local, self.vram.slot(layer, local)))
                .filter(|(_, slot)| back.contains(slot))
                .collect();
            if own.is_empty() {
                continue;
            }
            for (local, _) in &own {
                self.vram.sets[layer].end_loan(*local);
            }
            let slots: Vec<u32> = own.iter().map(|(_, slot)| *slot).collect();
            self.vram.retake(layer, &slots);
            self.worth_stale[layer] = true;
        }
        Ok(())
    }
}
