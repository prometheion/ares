//! The **far pool** of a K/V group: the converted entries of the pages from
//! `near` on, in page-locked host memory that the device reads and writes over
//! its link (the K/V of a context beyond VRAM).
//!
//! A group whose pages do not all fit in VRAM keeps the converted entries of
//! its first `near` pages in the engine's pool on the device, as every group
//! does, and those of the pages from `near` on here. The device's table names
//! such a page with [`ares_cuda::FAR_ENTRY`] and its slot in this pool, page
//! minus `near`; the attention kernels read the same bytes from either pool.
//! What attention reads past the indexer's dense limit is a list of a few
//! thousand cells, so a page far away costs a step only the cells that are
//! picked from it.
//!
//! The pool is one range of address space per device part, an array of slots
//! per layer one after the other, of which pieces are locked, mapped for the
//! device and booked in the engine's RAM limit as the far pages come into use
//! ([`FarPool::ensure`]), and given back when the pages above them are freed
//! ([`FarPool::shrink`]). Both happen between steps: a lock is the driver's
//! and an unlock waits for the device.

use ares_cuda::{DevicePtr, FAR_ENTRY};
use ares_runtime::{ContextLock, ContextRegion, ContextShort, ModelError, PinnedRam};

use super::{PAGE_ENTRY, invalid};

/// The bytes a piece of the pool takes over all its layers, about: what is
/// locked at a time. A lock costs about half a millisecond per MiB, so a piece
/// is a stall of a few tens of milliseconds once in many thousand cells.
const PIECE: u64 = 64 << 20;

/// The fewest pieces a pool is cut into.
const PIECES_AT_LEAST: u32 = 8;

/// The device's entry of page `page` of a group whose first `near` pages are
/// in VRAM: the page in the pool on the device, or its slot in the far pool.
pub(crate) fn converted_entry(near: Option<u32>, page: u32) -> u32 {
    match near {
        Some(near) if page >= near => PAGE_ENTRY | FAR_ENTRY | (page - near),
        _ => page | PAGE_ENTRY,
    }
}

/// What a group needs of its far pool: where the pages begin to be far, where
/// each of its layers keeps their slots on the device, and the slots of a
/// piece: a copy of the driver's must not cross from one piece to the next
/// (each is a registration of its own), a kernel's read may.
#[derive(Clone, Debug)]
pub(crate) struct FarLayers {
    pub near: u32,
    pub layers: Vec<DevicePtr>,
    pub piece_slots: u32,
}

/// The far pool of one device part.
#[derive(Debug)]
pub struct FarPool {
    region: ContextRegion,
    near: u32,
    /// Slots of each layer: the pages from `near` on.
    capacity: u32,
    /// Per layer: the bytes of a converted entry, and where its array starts
    /// in the region.
    entries: Vec<u64>,
    offsets: Vec<usize>,
    /// Slots of a piece, and the locks of every piece that is locked: one per
    /// layer.
    piece_slots: u32,
    pieces: Vec<Option<Vec<ContextLock>>>,
}

impl FarPool {
    /// The pool of a part whose layers have converted entries of `entries`
    /// bytes each, for the pages `near..pages`, in `ram` for `device`.
    /// Nothing is locked yet.
    pub(crate) fn new(
        ram: &PinnedRam,
        device: &ares_cuda::Device,
        near: u32,
        pages: u32,
        entries: &[u64],
    ) -> Result<Self, ModelError> {
        if near >= pages || entries.is_empty() {
            return Err(invalid(format!(
                "a far pool for pages {near}..{pages} of {} layers",
                entries.len()
            )));
        }
        let capacity = pages - near;
        // A piece of every layer starts on a page of the host: its slots are
        // a multiple of what makes a layer's piece whole pages.
        let unit =
            entries.iter().map(|e| 4096 / gcd(*e, 4096)).fold(1u64, |a, b| a / gcd(a, b) * b);
        let all: u64 = entries.iter().sum();
        // A small pool still grows in steps: at least eight pieces.
        let most = (PIECE / all.max(1)).min(u64::from(capacity.div_ceil(PIECES_AT_LEAST)));
        let piece_slots = most.max(1).next_multiple_of(unit);
        let piece_slots = u32::try_from(piece_slots).unwrap_or(u32::MAX).min(capacity).max(1);
        let mut offsets = Vec::with_capacity(entries.len());
        let mut at = 0usize;
        for entry in entries {
            offsets.push(at);
            at += (u64::from(capacity) * entry).next_multiple_of(4096) as usize;
        }
        let region = ram.context_region(device, at)?;
        let n_pieces = capacity.div_ceil(piece_slots) as usize;
        Ok(Self {
            region,
            near,
            capacity,
            entries: entries.to_vec(),
            offsets,
            piece_slots,
            pieces: (0..n_pieces).map(|_| None).collect(),
        })
    }

    /// The first page that is far.
    pub fn near(&self) -> u32 {
        self.near
    }

    /// The far slots of each layer: the pages from [`FarPool::near`] on.
    pub fn slots(&self) -> u32 {
        self.capacity
    }

    /// The bytes of one layer's converted entry of a page, the most of its
    /// layers: what a copy of one layer's far slots takes per slot.
    pub fn entry_bytes(&self) -> u64 {
        self.entries.iter().copied().max().unwrap_or(0)
    }

    /// What the group needs to address the pool.
    pub(crate) fn layers(&self) -> FarLayers {
        let base = self.region.base();
        FarLayers {
            near: self.near,
            layers: self.offsets.iter().map(|at| base.offset(*at as u64)).collect(),
            piece_slots: self.piece_slots,
        }
    }

    /// The bytes of piece `piece` in layer `layer`, and where they start in
    /// the region.
    fn span(&self, piece: usize, layer: usize) -> (usize, usize) {
        let first = piece as u64 * u64::from(self.piece_slots);
        let end = (first + u64::from(self.piece_slots)).min(u64::from(self.capacity));
        let entry = self.entries[layer];
        let start = (first * entry) as usize;
        let stop = if end == u64::from(self.capacity) {
            (end * entry).next_multiple_of(4096) as usize
        } else {
            (end * entry) as usize
        };
        (self.offsets[layer] + start, stop - start)
    }

    /// The bytes of piece `piece` over all layers.
    fn piece_bytes(&self, piece: usize) -> u64 {
        (0..self.entries.len()).map(|l| self.span(piece, l).1 as u64).sum()
    }

    /// The pieces a pool needs locked when the far slots `live` (one flag per
    /// slot, from slot 0; slots past its end are free) hold pages in use:
    /// every piece with a page in use, and the piece after the last of them,
    /// for the pages the next steps take.
    pub fn wanted(&self, live: &[bool]) -> Vec<bool> {
        let mut wanted = vec![false; self.pieces.len()];
        for (slot, _) in live.iter().enumerate().filter(|(_, l)| **l) {
            if let Some(piece) = wanted.get_mut(slot / self.piece_slots as usize) {
                *piece = true;
            }
        }
        let spare = wanted.iter().rposition(|w| *w).map_or(0, |last| last + 1);
        if let Some(piece) = wanted.get_mut(spare) {
            *piece = true;
        }
        wanted
    }

    /// The bytes the pool holds locked with the pieces `wanted`.
    pub fn bytes_of(&self, wanted: &[bool]) -> u64 {
        (0..self.pieces.len())
            .filter(|p| wanted.get(*p) == Some(&true))
            .map(|p| self.piece_bytes(p))
            .sum()
    }

    /// The bytes locked now.
    pub fn locked(&self) -> u64 {
        (0..self.pieces.len())
            .filter(|p| self.pieces[*p].is_some())
            .map(|p| self.piece_bytes(p))
            .sum()
    }

    /// Gives back the pieces that are not `wanted`, between steps: their
    /// pages are free, and the memory goes back to the system and the books.
    pub fn release_unwanted(&mut self, wanted: &[bool]) {
        for (piece, lock) in self.pieces.iter_mut().enumerate() {
            if wanted.get(piece) != Some(&true) {
                drop(lock.take());
            }
        }
    }

    /// Locks the pieces `wanted`, between steps. `Err(Ok(short))` when the
    /// RAM limit has no room for a piece now: the owner makes room (the tier
    /// of experts gives pieces up) and asks again.
    pub fn lock_wanted(&mut self, wanted: &[bool]) -> Result<(), Result<ContextShort, ModelError>> {
        for piece in 0..self.pieces.len() {
            if wanted.get(piece) != Some(&true) || self.pieces[piece].is_some() {
                continue;
            }
            let mut locks = Vec::with_capacity(self.entries.len());
            for layer in 0..self.entries.len() {
                let (at, len) = self.span(piece, layer);
                // A lock that fails gives back the layers before it as they drop.
                locks.push(self.region.lock(at, len)?);
            }
            self.pieces[piece] = Some(locks);
        }
        Ok(())
    }

    /// Locks every piece: for a driver that keeps no books of its pages.
    pub fn lock_all(&mut self) -> Result<(), Result<ContextShort, ModelError>> {
        self.lock_wanted(&vec![true; self.pieces.len()])
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_far_page_is_named_by_its_slot_in_the_far_pool() {
        assert_eq!(converted_entry(None, 7), PAGE_ENTRY | 7);
        assert_eq!(converted_entry(Some(8), 7), PAGE_ENTRY | 7);
        assert_eq!(converted_entry(Some(8), 8), PAGE_ENTRY | FAR_ENTRY);
        assert_eq!(converted_entry(Some(8), 13), PAGE_ENTRY | FAR_ENTRY | 5);
    }

    #[test]
    fn the_unit_of_a_piece_makes_whole_pages_of_every_layer() {
        for entry in [139_264u64, 75_776, 18_944, 4096, 3000] {
            let unit = 4096 / gcd(entry, 4096);
            assert_eq!((unit * entry) % 4096, 0, "{entry}");
        }
    }
}
