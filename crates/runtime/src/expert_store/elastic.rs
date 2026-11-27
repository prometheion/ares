//! The RAM tier follows the memory it is allowed: pieces of its region are
//! given back to the system and taken again while the store runs.
//!
//! **Pieces.** The slots of every pool are cut into pieces of whole slots
//! ([`Layout`]): a piece starts on a page, is locked for the devices as one
//! range and released as one. No slot lies in two pieces, because a device
//! copies from one locked range and not across two. Staging is a piece of its
//! own that is never given up.
//!
//! **Following.** The owner of the store says how many bytes the tier may
//! hold ([`ExpertStore::set_ram_target`]). When the tier holds more, it gives
//! up pieces between steps, once no read, copy or CPU job touches the tier's
//! memory ([`ExpertStore::shrink_ram`]); when it holds less, pieces are
//! locked again and their slots come back empty
//! ([`ExpertStore::grow_ram`]). Both happen between steps only: the driver's
//! unlock waits for the device, and a device that computes a step may wait
//! for the store (see [`ares_cuda::gate`]). A piece that was given up keeps
//! its lock until the owner drops its release ([`ExpertStore::take_released`])
//! and the unlock is done, and it is not locked again before: the end of a
//! step that gave up pieces of two pools may find room for one of them right
//! away.
//!
//! **Which experts go.** The slots of a piece belong to a few layers and hold
//! whatever was read into them. Giving the piece up as it lies would take
//! everything from those layers and nothing from the others, and it would
//! drop hot experts while cold ones stay. So every layer of the pool gives
//! slots in proportion to what it has above the fewest a layer keeps, each
//! layer gives its coldest ([`ares_weights::SlotSet::give_back`]), and the
//! experts that lie in the piece and stay are copied into the slots that
//! were given, which then change hands. A slot's number in its layer never
//! changes; only the memory behind it does, and only while nothing reads it.

use std::collections::HashSet;
use std::ops::Range;
use std::time::{Duration, Instant};

use super::region::Released;
use super::tier::{Move, Pool, Tier};
use super::{ExpertStore, SlotDevice, StoreError};

/// How long the end of a step waits for the reads and copies that are on
/// their way before it gives pieces up; when they take longer, the next step
/// ends with another try.
pub(super) const STEP_WAIT: Duration = Duration::from_millis(20);

/// How one pool of the RAM tier lies in the region.
#[derive(Clone, Copy, Debug)]
struct PoolLayout {
    /// Where the pool starts, and the bytes from one slot to the next.
    base: usize,
    stride: usize,
    slots: usize,
    /// The slots of a piece, the bytes from one piece to the next, and the
    /// number of the pool's first piece.
    per_piece: usize,
    pitch: usize,
    first_piece: usize,
}

/// Where the slots of the RAM tier and its staging lie in the region, and
/// the pieces the region is locked in.
#[derive(Clone, Debug, Default)]
pub(super) struct Layout {
    pools: Vec<PoolLayout>,
    /// Where every piece starts, and its bytes. The staging piece is last.
    pub spans: Vec<(usize, usize)>,
    /// The piece that holds staging, where it starts, and the bytes of the
    /// whole region.
    pub stage: Option<usize>,
    pub stage_base: usize,
    pub len: usize,
}

impl Layout {
    /// The layout of `pools` in pieces of about `piece` bytes, with
    /// `stage_len` bytes of staging behind them.
    pub fn new(pools: &[Pool], piece: u64, stage_len: usize) -> Self {
        let mut layout = Self::default();
        let mut at = 0;
        for pool in pools {
            let stride = pool.stride as usize;
            let slots = pool.slots as usize;
            let per_piece = (piece as usize / stride.max(1)).max(1);
            let pitch = (per_piece * stride).next_multiple_of(4096);
            layout.pools.push(PoolLayout {
                base: at,
                stride,
                slots,
                per_piece,
                pitch,
                first_piece: layout.spans.len(),
            });
            for first in (0..slots).step_by(per_piece) {
                let len = ((slots - first).min(per_piece) * stride).next_multiple_of(4096);
                layout.spans.push((at, len));
                at += len;
            }
        }
        layout.stage_base = at;
        if stage_len > 0 {
            layout.stage = Some(layout.spans.len());
            let len = stage_len.next_multiple_of(4096);
            layout.spans.push((at, len));
            at += len;
        }
        layout.len = at.max(4096);
        layout
    }

    /// Where slot `slot` of `pool` starts in the region.
    pub fn slot_at(&self, pool: usize, slot: u32) -> usize {
        let p = &self.pools[pool];
        let slot = slot as usize;
        p.base + slot / p.per_piece * p.pitch + slot % p.per_piece * p.stride
    }

    /// The pool whose slots lie in `piece` and which of them do; `None` for
    /// the staging piece.
    fn slots_of(&self, piece: usize) -> Option<(usize, Range<usize>)> {
        let pool = self.pools.iter().rposition(|p| p.first_piece <= piece && p.slots > 0)?;
        let p = &self.pools[pool];
        let first = (piece - p.first_piece) * p.per_piece;
        (first < p.slots).then(|| (pool, first..(first + p.per_piece).min(p.slots)))
    }

    /// The pieces of the pools in the order a tier that starts small locks
    /// them: every pool gets the same share of its pieces.
    fn order(&self) -> Vec<usize> {
        let mut pieces: Vec<(f64, usize)> = Vec::new();
        for p in &self.pools {
            let count = p.slots.div_ceil(p.per_piece);
            pieces.extend((0..count).map(|i| (i as f64 / count as f64, p.first_piece + i)));
        }
        pieces.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        pieces.into_iter().map(|(_, piece)| piece).collect()
    }
}

/// Which pieces of the region the tier holds and what it may hold.
/// What one shrink of the RAM tier gave back ([`ExpertStore::on_shrink`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamShrink {
    /// The pieces given up.
    pub pieces: usize,
    /// Their bytes.
    pub bytes: u64,
    /// The bytes the tier holds after it.
    pub held: u64,
    /// The bytes it may hold.
    pub target: u64,
    /// It holds more than it may and has nothing left to give.
    pub at_floor: bool,
}

#[derive(Debug)]
pub(super) struct Pieces {
    pub layout: Layout,
    /// Per RAM pool and slot: its memory is locked and a layer has it.
    live: Vec<Vec<bool>>,
    /// The bytes the tier may hold.
    target: u64,
    /// Given up and not yet returned to the system.
    released: Vec<Released>,
    /// Bytes wait to be given up: no read is started for a guess.
    pub shrinking: bool,
    /// The tier holds more than it may and has nothing left to give.
    at_floor: bool,
}

impl Pieces {
    pub(super) fn new(layout: Layout, ram: &Tier) -> Self {
        Self {
            layout,
            live: ram.pools.iter().map(|pool| vec![true; pool.slots as usize]).collect(),
            target: u64::MAX,
            released: Vec::new(),
            shrinking: false,
            at_floor: false,
        }
    }
}

impl<D: SlotDevice> ExpertStore<D> {
    /// Locks the pieces the region starts with and takes the slots of the
    /// others from their layers: the end of [`ExpertStore::new`].
    pub(super) fn settle_pieces(&mut self) -> Result<(), StoreError> {
        if let Some(stage) = self.pieces.layout.stage {
            self.region.lock(stage)?;
        }
        for piece in self.pieces.layout.order() {
            let len = self.region.span(piece).1;
            // A piece the books have no room for is left out like one the
            // start does not reach.
            if self.region.locked_bytes() + len > self.region.start()
                || self.region.lock(piece).is_err()
            {
                break;
            }
        }
        // What is not locked has no slots; a piece the layers cannot do
        // without is locked whatever the start says.
        for piece in (0..self.region.n_pieces()).rev() {
            if !self.region.is_locked(piece) && !self.vacate(piece) {
                self.region.lock(piece)?;
            }
        }
        self.pieces.target = self.region.locked_bytes() as u64;
        Ok(())
    }

    /// The bytes of the RAM tier that are locked now, and the bytes of its
    /// whole region.
    pub fn ram_bytes(&self) -> (u64, u64) {
        (self.region.locked_bytes() as u64, self.region.len() as u64)
    }

    /// The slots the RAM tier has now, over all layers.
    pub fn ram_slots(&self) -> usize {
        (0..self.layers.len()).map(|layer| self.ram.slots(layer)).sum()
    }

    /// The bytes of the piece the tier never gives up: its staging slots.
    pub fn ram_fixed_bytes(&self) -> u64 {
        self.pieces.layout.stage.map_or(0, |piece| self.region.span(piece).1 as u64)
    }

    /// Whether a step runs: its kernels may wait on the devices for what
    /// the store brings.
    pub fn in_step(&self) -> bool {
        self.in_step
    }

    /// The bytes the tier may hold.
    pub fn ram_target(&self) -> u64 {
        self.pieces.target
    }

    /// Whether the tier holds more than it may and can give nothing more:
    /// what is left are its staging slots and the fewest slots of its
    /// layers.
    pub fn ram_at_floor(&self) -> bool {
        self.pieces.at_floor
    }

    /// Tells `f` of every shrink of the RAM tier that gave up a piece, where
    /// it gave it up (with the store held: `f` only reports). The runtime
    /// writes no log; the owner of the store does.
    pub fn on_shrink(&mut self, f: Box<dyn FnMut(RamShrink) + Send>) {
        self.on_shrink = Some(f);
    }

    /// Says how many bytes the tier may hold from now on. It follows between
    /// steps ([`ExpertStore::shrink_ram`], [`ExpertStore::grow_ram`]).
    pub fn set_ram_target(&mut self, bytes: u64) {
        if bytes != self.pieces.target {
            self.pieces.at_floor = false;
        }
        self.pieces.target = bytes;
        self.pieces.shrinking = !self.pieces.at_floor && self.region.locked_bytes() as u64 > bytes;
    }

    /// Whether nothing reads or writes the memory of the RAM tier: no read
    /// from the file, no copy to or from a device, no job of the CPU.
    fn ram_quiet(&self) -> bool {
        self.reads.is_empty()
            && self.device_ops.iter().flatten().all(|queue| queue.is_empty())
            && self.cpu_is_settled()
            && !self.transit.is_busy()
    }

    /// Gives up pieces until the tier holds no more than its target, if no
    /// step runs and nothing touches the tier's memory; waits up to `wait`
    /// for what is on its way. Returns the pieces given up; they wait in
    /// [`ExpertStore::take_released`] to be returned to the system.
    pub fn shrink_ram(&mut self, wait: Duration) -> Result<usize, StoreError> {
        let over = (self.region.locked_bytes() as u64).saturating_sub(self.pieces.target);
        if over == 0 || self.pieces.at_floor {
            self.pieces.shrinking = false;
            return Ok(0);
        }
        self.pieces.shrinking = true;
        if self.in_step {
            return Ok(0);
        }
        let begun = Instant::now();
        loop {
            self.poll()?;
            if self.ram_quiet() {
                break;
            }
            if begun.elapsed() >= wait {
                return Ok(0);
            }
            std::thread::yield_now();
        }
        let mut given = 0;
        let mut kept = HashSet::new();
        let before = self.region.locked_bytes() as u64;
        while self.region.locked_bytes() as u64 > self.pieces.target {
            let Some(piece) = self.piece_to_release(&kept) else {
                self.pieces.at_floor = true;
                break;
            };
            if !self.vacate(piece) {
                kept.insert(piece);
                continue;
            }
            if let Some(released) = self.region.release(piece) {
                self.pieces.released.push(released);
                self.stats.pieces_released += 1;
                given += 1;
            }
        }
        self.pieces.shrinking = false;
        if given > 0
            && let Some(f) = &mut self.on_shrink
        {
            let held = self.region.locked_bytes() as u64;
            f(RamShrink {
                pieces: given,
                bytes: before - held,
                held,
                target: self.pieces.target,
                at_floor: self.pieces.at_floor,
            });
        }
        Ok(given)
    }

    /// The pieces the tier has given up. Dropping them unlocks their memory
    /// and returns it to the system. The unlock is the driver's and waits
    /// for the device to finish what it has: drop them where no step runs
    /// on the devices (a kernel that waits for an expert would wait for the
    /// thread that waits for it), which is where
    /// [`ExpertStore::shrink_ram`] gave them up. Until they are dropped,
    /// their pieces are not locked again.
    pub fn take_released(&mut self) -> Vec<Released> {
        std::mem::take(&mut self.pieces.released)
    }

    /// The next piece to lock when the tier may hold more than it does.
    fn ram_piece_to_lock(&self) -> Option<usize> {
        // Asked at every look of the watch: nothing is searched for a tier
        // that holds what it may.
        if self.region.locked_bytes() as u64 >= self.pieces.target {
            return None;
        }
        let piece = self.piece_to_lock()?;
        let len = self.region.span(piece).1;
        (self.region.locked_bytes() as u64 + len as u64 <= self.pieces.target).then_some(piece)
    }

    /// Locks up to `most` pieces, as far as the tier may hold more than it
    /// does: their slots go back to the layers that gave them, empty.
    /// Returns the pieces locked. Locking takes time (tens of milliseconds a
    /// piece) and is done here, by the caller's thread: call it between
    /// steps, where no step waits for the store (a lock waits for no step to
    /// run, and inside one it fails: [`ares_cuda::gate`]).
    pub fn grow_ram(&mut self, most: usize) -> Result<usize, StoreError> {
        let mut locked = 0;
        while locked < most {
            let Some(piece) = self.ram_piece_to_lock() else { break };
            self.region.lock(piece)?;
            self.revive(piece);
            self.stats.pieces_locked += 1;
            self.pieces.at_floor = false;
            locked += 1;
        }
        Ok(locked)
    }

    /// The share of its slots every pool has now.
    fn live_shares(&self) -> Vec<f64> {
        let shares = self.pieces.live.iter();
        shares
            .map(|live| live.iter().filter(|slot| **slot).count() as f64 / live.len().max(1) as f64)
            .collect()
    }

    /// The piece to give up next: of the pool that has the largest share of
    /// its slots, the one at the highest address.
    fn piece_to_release(&self, kept: &HashSet<usize>) -> Option<usize> {
        let shares = self.live_shares();
        (0..self.region.n_pieces())
            .filter(|piece| self.region.is_locked(*piece) && !kept.contains(piece))
            .filter_map(|piece| Some((shares[self.pieces.layout.slots_of(piece)?.0], piece)))
            .max_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .map(|(_, piece)| piece)
    }

    /// The piece to lock next: of the pool that has the smallest share of
    /// its slots, the one at the lowest address. Not a piece whose release
    /// has not been dropped: its lock still covers it.
    fn piece_to_lock(&self) -> Option<usize> {
        let shares = self.live_shares();
        (0..self.region.n_pieces())
            .filter(|piece| !self.region.is_locked(*piece) && !self.region.is_releasing(*piece))
            .filter_map(|piece| Some((shares[self.pieces.layout.slots_of(piece)?.0], piece)))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .map(|(_, piece)| piece)
    }

    /// The layers of a RAM pool.
    fn pool_layers(&self, pool: usize) -> Vec<usize> {
        (0..self.layers.len()).filter(|layer| self.ram.pool_of[*layer] == pool).collect()
    }

    /// Takes every slot of `piece` out of use, so that the piece can be
    /// given up: see the [module documentation](self) ([`Tier::vacate`]).
    /// Returns false, with nothing taken, when the layers cannot give that
    /// many slots.
    fn vacate(&mut self, piece: usize) -> bool {
        let Some((pool, slots)) = self.pieces.layout.slots_of(piece) else { return false };
        let copy = self.region.is_locked(piece);
        let dying: Vec<u32> =
            slots.filter(|slot| self.pieces.live[pool][*slot]).map(|slot| slot as u32).collect();
        if dying.is_empty() {
            return true;
        }
        let members = self.pool_layers(pool);
        let floor: Vec<usize> = members.iter().map(|layer| self.ram.floor[*layer]).collect();
        for layer in &members {
            self.refresh_worth(*layer);
        }
        let vacated =
            self.ram.vacate(&members, &dying, &floor, self.step, |l| self.ram_worth.layer(l));
        self.stats.piece_evictions += vacated.evicted.len() as u64;
        let Some(moves) = vacated.moves else {
            for layer in members {
                self.worth_stale[layer] = true;
            }
            return false;
        };
        // What lies in the piece and stays is copied into the slots given.
        let stride = self.ram.pools[pool].stride as usize;
        for Move { from, to, .. } in moves {
            if copy {
                let (from, to) =
                    (self.pieces.layout.slot_at(pool, from), self.pieces.layout.slot_at(pool, to));
                // SAFETY: both slots are inside the region and locked, they
                // are different slots of one pool, and nothing reads or
                // writes the tier's memory now (`ram_quiet`).
                unsafe {
                    let region = self.region.ptr().as_ptr();
                    std::ptr::copy_nonoverlapping(region.add(from), region.add(to), stride);
                }
            }
            self.ram_skip[pool][to as usize] = self.ram_skip[pool][from as usize];
        }
        for slot in dying {
            self.pieces.live[pool][slot as usize] = false;
        }
        true
    }

    /// The slots of `piece`, which is locked again, go back to the layers
    /// that hold their numbers ([`Tier::reopen`]).
    fn revive(&mut self, piece: usize) {
        let Some((pool, slots)) = self.pieces.layout.slots_of(piece) else { return };
        let reborn: HashSet<u32> =
            slots.filter(|slot| !self.pieces.live[pool][*slot]).map(|slot| slot as u32).collect();
        if reborn.is_empty() {
            return;
        }
        for layer in self.ram.reopen(pool, &reborn) {
            self.worth_stale[layer] = true;
        }
        for slot in reborn {
            self.pieces.live[pool][slot as usize] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_are_whole_slots_on_whole_pages_and_staging_is_one_of_its_own() {
        let pools =
            [Pool { place: 0, stride: 3000, slots: 10 }, Pool { place: 0, stride: 8192, slots: 3 }];
        // Pieces of 10 000 bytes: three slots of the first pool (9000 bytes,
        // three pages), one of the second.
        let layout = Layout::new(&pools, 10_000, 5000);
        assert_eq!(
            layout.spans,
            [
                (0, 12288),
                (12288, 12288),
                (24576, 12288),
                (36864, 4096),
                (40960, 8192),
                (49152, 8192),
                (57344, 8192),
                (65536, 8192)
            ]
        );
        assert_eq!((layout.stage, layout.stage_base, layout.len), (Some(7), 65536, 73728));
        assert_eq!(layout.slot_at(0, 0), 0);
        assert_eq!(layout.slot_at(0, 2), 6000);
        assert_eq!(layout.slot_at(0, 3), 12288, "the next piece starts on a page");
        assert_eq!(layout.slot_at(0, 9), 36864);
        assert_eq!(layout.slot_at(1, 0), 40960);
        assert_eq!(layout.slot_at(1, 2), 57344);
        assert_eq!(layout.slots_of(0), Some((0, 0..3)));
        assert_eq!(layout.slots_of(3), Some((0, 9..10)));
        assert_eq!(layout.slots_of(4), Some((1, 0..1)));
        assert_eq!(layout.slots_of(6), Some((1, 2..3)));
        assert_eq!(layout.slots_of(7), None, "the staging piece has no slots of a pool");
        // A tier that starts small takes pieces of both pools.
        assert_eq!(layout.order(), [0, 4, 1, 5, 2, 6, 3]);
        // No slot lies in two pieces.
        for (pool, p) in pools.iter().enumerate() {
            for slot in 0..p.slots {
                let at = layout.slot_at(pool, slot);
                let inside = layout
                    .spans
                    .iter()
                    .filter(|(start, len)| *start <= at && at + p.stride as usize <= start + len);
                assert_eq!(inside.count(), 1, "pool {pool} slot {slot}");
            }
        }
        // Without staging and without slots.
        let empty = Layout::new(&[], 10_000, 0);
        assert_eq!((empty.spans.len(), empty.stage, empty.len), (0, None, 4096));
    }
}
