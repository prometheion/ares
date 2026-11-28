//! The VRAM tier gives memory back to the device when other programs need
//! it, and takes it again when they have let it go.
//!
//! **Pieces.** A pool of slots lies in one range of device addresses whose
//! memory is in pieces ([`SlotDevice::pieces`], `ares_cuda::DeviceRegion`):
//! a kernel finds a slot at the pool's address plus its number times its size,
//! so the range stays and only the memory behind a piece goes. A slot that
//! lies in a piece that is given back has no memory; a slot across the edge of
//! two pieces needs both.
//!
//! **Following.** The owner says how many bytes the tier may hold on a device
//! ([`ExpertStore::set_vram_target`]: the watch over the devices' memory,
//! `crate::vram`). When it holds more, pieces go between steps, once nothing
//! touches the device's slots ([`ExpertStore::shrink_vram`]): every layer of
//! the pool gives slots in proportion to what it has above its floor, each its
//! coldest, the experts that lie in the piece and stay move on the device into
//! the slots given, and once the device has done the moves and the copies down
//! of what was evicted, the memory goes back to the driver, where no job runs.
//! This is the bookkeeping of a loan of VRAM to a step's scratch
//! (`scratch.rs`), and the numbers of the slots that have no memory stay set
//! aside until it comes back ([`ExpertStore::grow_vram`]: the slots come back
//! empty).
//!
//! **What never goes.** The pieces at the low end of a pool go first, and the
//! slots at its end that a loan takes ([`ExpertStore::keep_for_loans`]: the
//! steps that run on lent VRAM were sized for them at load) never go, nor
//! does a layer give slots below its floor, nor a pool of a drafter's
//! experts, which every link of a chain needs. When nothing more can go, the
//! tier says so ([`ExpertStore::vram_at_floor`]) and holds what it has.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use super::{ExpertStore, SlotDevice, StoreError};

/// What one shrink of the VRAM tier gave back ([`ExpertStore::on_vram_shrink`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramShrink {
    /// The store's device.
    pub device: usize,
    /// The pieces given back, and their bytes.
    pub pieces: usize,
    pub bytes: u64,
    /// The bytes the tier holds on the device after it, and may hold.
    pub held: u64,
    pub target: u64,
    /// It holds more than it may and has nothing left to give.
    pub at_floor: bool,
}

/// The pieces of one pool of the VRAM tier.
#[derive(Debug, Default)]
struct PoolPieces {
    /// The pool gives memory back: its device has its memory in pieces and
    /// its layers are not a drafter's.
    elastic: bool,
    piece: u64,
    /// Per piece: memory is behind it, as the store counts (a piece given
    /// back counts as gone at once).
    mapped: Vec<bool>,
    /// Pieces whose slots were vacated, and the marks of the device's two
    /// kinds of work behind the moves and copies out of them (`None` until
    /// they are made): the memory goes back to the device when both are
    /// passed.
    leaving: Vec<(usize, Option<(u64, u64)>)>,
    /// The bytes at the end of the pool that loans take: never given back.
    kept: u64,
}

/// The pieces of every pool of the VRAM tier, and what the tier may hold.
#[derive(Debug, Default)]
pub(super) struct VramPieces {
    pools: Vec<PoolPieces>,
    /// Per device: the bytes the tier's elastic pools may hold there.
    target: Vec<u64>,
    /// Per device: it holds more than it may and found nothing to give.
    at_floor: Vec<bool>,
}

impl<D: SlotDevice> ExpertStore<D> {
    /// Learns how the memory of every pool of the VRAM tier is cut: the end of
    /// [`ExpertStore::new`].
    pub(super) fn settle_vram_pieces(&mut self) {
        let mut pools = Vec::with_capacity(self.vram.pools.len());
        for (p, pool) in self.vram.pools.iter().enumerate() {
            let members: Vec<usize> =
                (0..self.layers.len()).filter(|l| self.vram.pool_of[*l] == p).collect();
            let Some(first) = members.first() else {
                pools.push(PoolPieces::default());
                continue;
            };
            let resident = members.iter().any(|l| self.layers[*l].resident);
            let (piece, n) = self.devices[pool.place].pieces(self.layers[*first].device_pool);
            let elastic = !resident && piece > 0 && n > 0 && pool.slots > 0;
            pools.push(PoolPieces {
                elastic,
                piece,
                mapped: vec![true; if elastic { n } else { 0 }],
                leaving: Vec::new(),
                kept: 0,
            });
        }
        self.vram_pieces = VramPieces {
            pools,
            target: vec![u64::MAX; self.devices.len()],
            at_floor: vec![false; self.devices.len()],
        };
    }

    /// The bytes of the elastic pools of `device` that have memory behind
    /// them, and of all of them: what the tier holds there that can go, and
    /// the most it can hold.
    pub fn vram_elastic(&self, device: usize) -> (u64, u64) {
        let mut held = 0;
        let mut most = 0;
        for (p, pieces) in self.vram_pieces.pools.iter().enumerate() {
            if !pieces.elastic || self.vram.pools[p].place != device {
                continue;
            }
            for (k, mapped) in pieces.mapped.iter().enumerate() {
                let bytes = self.piece_span(p, k).1;
                most += bytes;
                held += if *mapped { bytes } else { 0 };
            }
        }
        (held, most)
    }

    /// The bytes the tier may hold on `device` from now on, in its elastic
    /// pools. It follows between steps ([`ExpertStore::shrink_vram`],
    /// [`ExpertStore::grow_vram`]).
    pub fn set_vram_target(&mut self, device: usize, bytes: u64) {
        if self.vram_pieces.target.get(device) != Some(&bytes)
            && let Some(floor) = self.vram_pieces.at_floor.get_mut(device)
        {
            *floor = false;
        }
        if let Some(target) = self.vram_pieces.target.get_mut(device) {
            *target = bytes;
        }
    }

    /// The bytes the tier may hold on `device`.
    pub fn vram_target(&self, device: usize) -> u64 {
        self.vram_pieces.target.get(device).copied().unwrap_or(u64::MAX)
    }

    /// Whether the tier holds more than it may on `device` and found nothing
    /// more to give: what is left are the floors of its layers and the slots
    /// that loans take.
    pub fn vram_at_floor(&self, device: usize) -> bool {
        self.vram_pieces.at_floor.get(device).copied().unwrap_or(false)
    }

    /// The flag that the tier on `device` found nothing more to give: an end
    /// of a step tries again, since what was busy may be free now.
    pub(super) fn vram_pieces_floor_mut(&mut self, device: usize) -> Option<&mut bool> {
        self.vram_pieces.at_floor.get_mut(device)
    }

    /// Whether the tier holds more on some device than it may.
    pub fn vram_over(&self) -> bool {
        (0..self.devices.len())
            .any(|d| !self.vram_at_floor(d) && self.vram_elastic(d).0 > self.vram_target(d))
    }

    /// Tells `f` of every shrink of the VRAM tier that gave up a piece.
    pub fn on_vram_shrink(&mut self, f: Box<dyn FnMut(VramShrink) + Send>) {
        self.on_vram_shrink = Some(f);
    }

    /// The bytes at the end of each pool of `device` that loans take
    /// (`pieces`, in the order of [`ExpertStore::lendable_pieces`]): they
    /// never go back to the device. The owner says so once it has laid out
    /// the steps that run on lent VRAM.
    pub fn keep_for_loans(&mut self, device: usize, pieces: &[u64]) {
        for ((pool, _), bytes) in self.scratch_pools(device).into_iter().zip(pieces) {
            let kept = &mut self.vram_pieces.pools[pool].kept;
            *kept = (*kept).max(*bytes);
        }
    }

    /// Where piece `k` of tier pool `pool` starts in the pool, and its bytes.
    fn piece_span(&self, pool: usize, k: usize) -> (u64, u64) {
        let pieces = &self.vram_pieces.pools[pool];
        let len = self.vram.pools[pool].stride * u64::from(self.vram.pools[pool].slots);
        let start = k as u64 * pieces.piece;
        (start, len.saturating_sub(start).min(pieces.piece))
    }

    /// The slots of tier pool `pool` that lie in piece `k`, wholly or in part.
    fn slots_in_piece(&self, pool: usize, k: usize) -> std::ops::Range<u32> {
        let (start, len) = self.piece_span(pool, k);
        let stride = self.vram.pools[pool].stride.max(1);
        let first = (start / stride) as u32;
        let end = (start + len).div_ceil(stride).min(u64::from(self.vram.pools[pool].slots)) as u32;
        first..end
    }

    /// Whether every piece slot `slot` of tier pool `pool` lies in has memory.
    fn slot_alive(&self, pool: usize, slot: u32) -> bool {
        let pieces = &self.vram_pieces.pools[pool];
        if !pieces.elastic {
            return true;
        }
        let stride = self.vram.pools[pool].stride;
        let start = u64::from(slot) * stride;
        let first = (start / pieces.piece) as usize;
        let last = ((start + stride).saturating_sub(1) / pieces.piece) as usize;
        (first..=last).all(|k| pieces.mapped.get(k).copied().unwrap_or(false))
    }

    /// The slots at the end of tier pool `pool` that have memory behind them,
    /// one after the other: what a loan, which takes the last slots, can take.
    pub(super) fn live_tail(&self, pool: usize) -> u32 {
        let slots = self.vram.pools[pool].slots;
        if !self.vram_pieces.pools.get(pool).is_some_and(|p| p.elastic) {
            return slots;
        }
        (0..slots).rev().take_while(|slot| self.slot_alive(pool, *slot)).count() as u32
    }

    /// Gives the memory of the pieces that have left to the device, once the
    /// device has done what moved out of them. Never waits.
    pub(super) fn settle_vram(&mut self) -> Result<(), StoreError> {
        for p in 0..self.vram_pieces.pools.len() {
            if self.vram_pieces.pools[p].leaving.is_empty() {
                continue;
            }
            let device = self.vram.pools[p].place;
            let device_pool = self.device_pool_of(p);
            let leaving = std::mem::take(&mut self.vram_pieces.pools[p].leaving);
            let mut stay = Vec::new();
            for (k, marks) in leaving {
                let passed = match marks {
                    Some((up, down)) => {
                        self.devices[device].passed(up)? && self.devices[device].passed(down)?
                    }
                    None => false,
                };
                if passed {
                    self.devices[device].give_piece(device_pool, k);
                } else {
                    stay.push((k, marks));
                }
            }
            self.vram_pieces.pools[p].leaving = stay;
        }
        Ok(())
    }

    /// The pool among its device's pools of tier pool `pool`.
    fn device_pool_of(&self, pool: usize) -> usize {
        let place = self.vram.pools[pool].place;
        self.vram.pools[..pool].iter().filter(|p| p.place == place).count()
    }

    /// Gives up pieces of `device` until the tier holds no more there than
    /// its target, if no step runs, no loan is out and nothing touches the
    /// device's slots; waits up to `wait` for what is on its way. Returns the
    /// pieces given up; their memory goes back to the device once the device
    /// has done the moves out of them ([`ExpertStore::poll`]).
    pub fn shrink_vram(&mut self, device: usize, wait: Duration) -> Result<usize, StoreError> {
        self.settle_vram()?;
        let (held, _) = self.vram_elastic(device);
        let target = self.vram_target(device);
        if held <= target || self.vram_at_floor(device) {
            return Ok(0);
        }
        if self.in_step || self.has_scratch(device) {
            return Ok(0);
        }
        let begun = Instant::now();
        loop {
            self.poll()?;
            if self.vram_quiet(device) {
                break;
            }
            if begun.elapsed() >= wait {
                return Ok(0);
            }
            std::thread::yield_now();
        }
        let mut given = 0;
        let mut tried: HashSet<usize> = HashSet::new();
        while self.vram_elastic(device).0 > target {
            let Some((pool, k)) = self.vram_piece_to_give(device, &tried) else {
                self.vram_pieces.at_floor[device] = true;
                break;
            };
            if !self.give_vram_piece(device, pool, k)? {
                tried.insert(pool);
                continue;
            }
            given += 1;
        }
        if given > 0 {
            let marks = (self.devices[device].mark(false)?, self.devices[device].mark(true)?);
            for pieces in &mut self.vram_pieces.pools {
                for leaving in pieces.leaving.iter_mut().filter(|l| l.1.is_none()) {
                    leaving.1 = Some(marks);
                }
            }
            self.stats.vram_pieces_released += given as u64;
            let (now, _) = self.vram_elastic(device);
            let at_floor = self.vram_at_floor(device);
            if let Some(f) = &mut self.on_vram_shrink {
                f(VramShrink {
                    device,
                    pieces: given,
                    bytes: held - now,
                    held: now,
                    target,
                    at_floor,
                });
            }
        }
        Ok(given)
    }

    /// The next piece of `device` to give: of the elastic pool (not in
    /// `tried`) that has the largest share of its pieces, its lowest piece
    /// with memory that lies below the slots loans take.
    fn vram_piece_to_give(&self, device: usize, tried: &HashSet<usize>) -> Option<(usize, usize)> {
        let mut best: Option<(f64, usize, usize)> = None;
        for (p, pieces) in self.vram_pieces.pools.iter().enumerate() {
            if !pieces.elastic || self.vram.pools[p].place != device || tried.contains(&p) {
                continue;
            }
            let pool = self.vram.pools[p];
            let kept_slots = pieces.kept.div_ceil(pool.stride.max(1)).min(u64::from(pool.slots));
            let below = (u64::from(pool.slots) - kept_slots) * pool.stride;
            let Some(k) = (0..pieces.mapped.len()).find(|k| pieces.mapped[*k]) else { continue };
            let (start, len) = self.piece_span(p, k);
            if start + len > below {
                continue;
            }
            let share = pieces.mapped.iter().filter(|m| **m).count() as f64
                / pieces.mapped.len().max(1) as f64;
            if best.is_none_or(|(s, _, _)| share > s) {
                best = Some((share, p, k));
            }
        }
        best.map(|(_, p, k)| (p, k))
    }

    /// Takes the slots of piece `k` of tier pool `pool` out of use, so that
    /// its memory can go: see the [module documentation](self). False, with
    /// nothing changed, when the layers cannot give that many slots and keep
    /// what loans take.
    fn give_vram_piece(
        &mut self,
        device: usize,
        pool: usize,
        k: usize,
    ) -> Result<bool, StoreError> {
        let dying: Vec<u32> =
            self.slots_in_piece(pool, k).filter(|slot| self.slot_alive(pool, *slot)).collect();
        let members: Vec<usize> =
            (0..self.layers.len()).filter(|l| self.vram.pool_of[*l] == pool).collect();
        // What loans take must stay lendable above the floors of a loan.
        let kept_slots =
            self.vram_pieces.pools[pool].kept.div_ceil(self.vram.pools[pool].stride.max(1));
        let spare: u64 = members
            .iter()
            .map(|l| self.vram.slots(*l).saturating_sub(self.scratch_floor(*l)) as u64)
            .sum();
        if spare < kept_slots + dying.len() as u64 {
            return Ok(false);
        }
        let floor: Vec<usize> = members.iter().map(|l| self.vram.floor[*l]).collect();
        if !dying.is_empty() && !self.vacate_on_device(device, &members, &dying, &floor)? {
            return Ok(false);
        }
        self.vram_pieces.pools[pool].mapped[k] = false;
        self.vram_pieces.pools[pool].leaving.push((k, None));
        Ok(true)
    }

    /// Takes up to `most` pieces of `device` again, as far as the tier may
    /// hold more there than it does: their slots go back to the layers that
    /// hold their numbers, empty, and fill the way empty slots do. A piece
    /// the device has no room for now is no error: the tier stays as it is.
    /// Takes device memory: call it between steps.
    pub fn grow_vram(&mut self, device: usize, most: usize) -> Result<usize, StoreError> {
        self.settle_vram()?;
        let mut taken = 0;
        while taken < most {
            let (held, _) = self.vram_elastic(device);
            let Some((pool, k)) = self.vram_piece_to_take(device) else { break };
            if held + self.piece_span(pool, k).1 > self.vram_target(device) {
                break;
            }
            let device_pool = self.device_pool_of(pool);
            if !self.devices[device].take_piece(device_pool, k)? {
                break;
            }
            self.vram_pieces.pools[pool].mapped[k] = true;
            let back: HashSet<u32> =
                self.slots_in_piece(pool, k).filter(|slot| self.slot_alive(pool, *slot)).collect();
            for layer in self.vram.reopen(pool, &back) {
                self.worth_stale[layer] = true;
            }
            self.vram_pieces.at_floor[device] = false;
            self.stats.vram_pieces_taken += 1;
            taken += 1;
        }
        if taken > 0 && self.hybrid.is_some() && !self.in_step {
            self.promote_hot()?;
        }
        Ok(taken)
    }

    /// The next piece of `device` to take again: of the elastic pool with the
    /// smallest share of its pieces, the highest one without memory that is
    /// not still on its way back.
    fn vram_piece_to_take(&self, device: usize) -> Option<(usize, usize)> {
        let mut best: Option<(f64, usize, usize)> = None;
        for (p, pieces) in self.vram_pieces.pools.iter().enumerate() {
            if !pieces.elastic || self.vram.pools[p].place != device {
                continue;
            }
            let leaving: HashSet<usize> = pieces.leaving.iter().map(|l| l.0).collect();
            let Some(k) =
                (0..pieces.mapped.len()).rev().find(|k| !pieces.mapped[*k] && !leaving.contains(k))
            else {
                continue;
            };
            let share = pieces.mapped.iter().filter(|m| **m).count() as f64
                / pieces.mapped.len().max(1) as f64;
            if best.is_none_or(|(s, _, _)| share < s) {
                best = Some((share, p, k));
            }
        }
        best.map(|(_, p, k)| (p, k))
    }
}
