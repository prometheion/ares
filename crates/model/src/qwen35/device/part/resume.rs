//! What a part does for the prefix cache: the places on the host where
//! checkpoints of recurrent state are kept, the copies the step record
//! carries (saves, restores, copies of pages), and the rows of a replay.
//!
//! **A checkpoint** of a part is what a sequence resumes from on this device:
//! its slot of the recurrent state (the convolution windows and delta-rule
//! matrices of its blocks) and, on the device that holds the draft block, the
//! block's carry (the final hidden state of the sequence's last row, which the
//! block's keys and values of the next row are computed from). It lies in
//! page-locked host memory, not in VRAM: a save is a copy from the device and
//! a restore a copy to it, both queued on the step's stream, so the stream
//! orders them against the kernels around them. The places are locked in
//! chunks when they are first used, within the engine's RAM limit
//! ([`PinnedRam`]), and a chunk in which the conductor has no checkpoint is
//! given back when the checkpoints must make room.
//!
//! **A page** is copied whole, for every attention block and for the draft
//! block's own array, because the cells of an entry lie in planes per head;
//! cells past those the sequence has are not read before it writes them.
//!
//! **A replay** appends nothing: its rows read the cells that are cached
//! already and only advance the recurrent state.

use std::sync::Arc;

use ares_cuda::{Device, DevicePtr, Event, StateFormat, Stream};
use ares_runtime::{
    DeviceTransfer, ModelError, PinnedRam, PlaceChunks, RecordView, StateCopyKind, TransferKind,
};

use super::{DRAFT_KV, Options, Part, Profile, geometry, kv};
use crate::qwen35::Config;
use crate::qwen35::device::placement::Placement;

/// The smallest chunk of host memory that is locked at a time: places are
/// locked a chunk of them at once, so that small places do not cost a lock
/// each.
const CHUNK_BYTES: usize = 32 << 20;

/// The bytes of one sequence's recurrent state on device `index` of
/// `placement`.
pub fn state_slot_bytes(
    c: &Config,
    placement: &Placement,
    index: usize,
    state: StateFormat,
) -> u64 {
    let r = &c.recurrent;
    placement.devices[index]
        .blocks
        .iter()
        .enumerate()
        .filter(|(layer, _)| !c.is_attention(*layer))
        .filter_map(|(_, slice)| slice.as_ref().filter(|slice| !slice.heads.is_empty()))
        .fold(0, |at, slice| {
            let n_k = slice.heads.len() as u32;
            let shape = ares_cuda::RecurrentShape {
                n_k_heads: n_k,
                n_v_heads: n_k * (r.n_v_heads / r.n_k_heads) as u32,
                head_dim: r.head_k_dim as u32,
                conv_kernel: r.conv_kernel as u32,
            };
            crate::blocks::recurrent::extent(&shape, state, at).2
        })
}

/// The bytes of one checkpoint on device `index`: its recurrent state, and the
/// draft block's carry row where the block is.
pub fn checkpoint_bytes(
    c: &Config,
    placement: &Placement,
    index: usize,
    state: StateFormat,
) -> u64 {
    let carry = placement
        .draft
        .as_ref()
        .filter(|draft| draft.device == index)
        .map_or(0, |_| c.n_embd as u64 * 4);
    state_slot_bytes(c, placement, index, state) + carry
}

/// The most bytes one page takes on device `index` on its way to the host and
/// back, for the prefix cache's level on SSD: every K/V entry the device holds
/// of the page, each in the larger of its two forms where its formats have two
/// (the converted entry, and the exact one a page has while it is new).
pub fn transit_page_bytes(c: &Config, placement: &Placement, index: usize, opts: &Options) -> u64 {
    let entry = |kv: (ares_cuda::CacheFormat, ares_cuda::CacheFormat), heads: u32| {
        let geo = geometry(c, kv, heads);
        match kv::has_exact(kv.0, kv.1) {
            true => geo.converted_entry_bytes().max(geo.exact_entry_bytes()),
            false => geo.converted_entry_bytes(),
        }
    };
    placement.devices[index]
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(layer, slice)| {
            let slice = slice.as_ref()?;
            (c.is_attention(layer) && !slice.heads.is_empty())
                .then(|| entry(opts.kv, slice.heads.len() as u32))
        })
        .sum::<u64>()
        + placement
            .draft
            .as_ref()
            .filter(|d| d.device == index)
            .map_or(0, |_| entry(DRAFT_KV, c.attention.n_head_kv as u32))
}

/// The cells before the start of a step that may still be exact entries
/// under `placement`: the newest cells that the formats of the attention
/// blocks and of the draft block keep exact. A step converts the tiles that
/// lie before them at its start. 0 when no group has exact entries.
pub fn settle_cells(placement: &Placement, opts: &Options) -> u32 {
    let draft = placement.draft.as_ref().map_or(0, |_| kv::newest_cells(DRAFT_KV.0, DRAFT_KV.1));
    kv::newest_cells(opts.kv.0, opts.kv.1).max(draft)
}

/// The tag of a page in transit: which groups' entries are exact ones.
const EXACT_ATTENTION: u32 = 1;
const EXACT_DRAFT: u32 = 2;

/// How many copies of each kind a step made, and the events around them.
#[derive(Clone, Copy, Debug, Default)]
struct CopyCounts {
    saves: u32,
    restores: u32,
    pages: u32,
    /// Chunks of host memory locked for the places of this step's copies, and
    /// the host's milliseconds that took.
    locks: u32,
    lock_ms: f64,
}

/// The time of the copies of the step that ran last: events in front of the
/// saves, after them, after the page copies and after the restores.
#[derive(Debug)]
struct CopyClock {
    events: [Event; 4],
    pending: Option<CopyCounts>,
}

/// The places for checkpoints on the host, locked as they come into use and
/// given back when the checkpoints must make room ([`PlaceChunks`]).
#[derive(Debug)]
pub(super) struct Places {
    /// The bytes of one place.
    bytes: usize,
    chunks: Arc<PlaceChunks>,
    /// Chunks that had to be locked for a place, since the part was opened.
    fresh: u32,
    clock: Option<CopyClock>,
}

impl Places {
    pub(super) fn new(
        device: &Device,
        ram: &PinnedRam,
        bytes: u64,
        capacity: u32,
        timed: bool,
    ) -> Result<Self, ModelError> {
        let clock = timed
            .then(|| -> Result<CopyClock, ares_cuda::Error> {
                Ok(CopyClock {
                    events: [
                        Event::timed(device)?,
                        Event::timed(device)?,
                        Event::timed(device)?,
                        Event::timed(device)?,
                    ],
                    pending: None,
                })
            })
            .transpose()?;
        Ok(Self {
            bytes: bytes as usize,
            chunks: ram.place_chunks(device, bytes, capacity, CHUNK_BYTES),
            fresh: 0,
            clock,
        })
    }

    /// Where place `place` is, locking the chunk it lies in when it is not
    /// locked: the first time, and after the chunk was given back.
    fn pointer(&mut self, place: u32) -> Result<*mut u8, ModelError> {
        let (pointer, fresh) = self.chunks.pointer(place)?;
        self.fresh += u32::from(fresh);
        Ok(pointer)
    }

    /// Where place `place` is when it holds memory: `None` after its chunk
    /// was given back, and before it was ever locked.
    fn held(&self, place: u32) -> Option<*mut u8> {
        self.chunks.held(place)
    }

    /// Bytes of host memory locked for the places now.
    pub(super) fn locked(&self) -> u64 {
        self.chunks.locked()
    }
}

impl Part {
    /// The bytes of one checkpoint of this part.
    pub fn checkpoint_bytes(&self) -> u64 {
        self.places.bytes as u64
    }

    /// Bytes of page-locked host memory the part's checkpoint places hold now.
    pub fn checkpoint_ram(&self) -> u64 {
        self.places.locked()
    }

    /// Copies the checkpoint state of `slot` into `place` on the host: the
    /// slot's recurrent state and the draft block's carry row of the slot.
    /// Queued on `stream`; the place is not read before the stream has passed.
    pub fn save_state(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError> {
        self.check_slot(slot)?;
        if self.places.bytes == 0 {
            return Ok(());
        }
        let host = self.places.pointer(place)?;
        let (state, carry) = (self.slot_bytes as usize, self.carry_row(slot));
        // SAFETY: the place holds `state` bytes and then the carry row, the
        // slot is inside the arena and the carry (checked above), and nothing
        // reads or writes the place before the stream has passed: restores of
        // it are queued on the same stream, later.
        unsafe {
            if state > 0 {
                stream.copy_to_host(
                    host,
                    self.state.ptr().offset(self.slot_offset(slot)),
                    state,
                )?;
            }
            if let Some((src, bytes)) = carry {
                stream.copy_to_host(host.add(state), src, bytes as usize)?;
            }
        }
        Ok(())
    }

    /// Copies `place` into the slot: the opposite of [`Part::save_state`].
    pub fn restore_state(
        &mut self,
        stream: &Stream,
        slot: u32,
        place: u32,
    ) -> Result<(), ModelError> {
        self.check_slot(slot)?;
        if self.places.bytes == 0 {
            return Ok(());
        }
        let host = self.places.pointer(place)?;
        let (state, carry) = (self.slot_bytes as usize, self.carry_row(slot));
        // SAFETY: as in `save_state`; the place was written by a save that ran
        // before this copy (an earlier record, or earlier in this one).
        unsafe {
            if state > 0 {
                stream.copy_from_host(
                    self.state.ptr().offset(self.slot_offset(slot)),
                    host,
                    state,
                )?;
            }
            if let Some((dst, bytes)) = carry {
                stream.copy_from_host(dst, host.add(state), bytes as usize)?;
            }
        }
        Ok(())
    }

    /// Carries out this part's share of a transfer of the prefix cache's level
    /// on SSD (`ares_runtime::DeviceModel::transfer`): a page's K/V entries of
    /// every attention block and of the draft block, in the form they have,
    /// or a checkpoint place. Runs between steps; the caller waits for the
    /// stream.
    pub fn transfer(
        &mut self,
        stream: &Stream,
        pages: DevicePtr,
        item: &mut DeviceTransfer,
    ) -> Result<(), ModelError> {
        let (kernels, status) = (self.attention.as_ref(), self.at(self.s.status));
        let groups = [
            (Some(&mut self.kv), EXACT_ATTENTION),
            (self.mtp.as_mut().map(|m| &mut m.kv), EXACT_DRAFT),
        ];
        match item.kind {
            TransferKind::PageOut | TransferKind::PageIn if item.object >= self.n_pages => Err(
                super::invalid(format!("page {} of {} in a transfer", item.object, self.n_pages)),
            ),
            TransferKind::PageOut => {
                (item.len, item.tag) = (0, 0);
                for (group, bit) in groups {
                    let Some(group) = group else { continue };
                    // SAFETY: the transfer's memory holds `room` bytes from
                    // `item.ptr`, which the caller reads after it has waited for
                    // the stream.
                    let (bytes, exact) = unsafe {
                        let dst = item.ptr.add(item.len);
                        group.page_out(stream, pages, item.object, dst, item.room - item.len)?
                    };
                    item.len += bytes;
                    item.tag |= if exact { bit } else { 0 };
                }
                Ok(())
            }
            TransferKind::PageIn => {
                if item.len > item.room {
                    return Err(super::invalid("a page in transit is longer than its room"));
                }
                let mut at = 0usize;
                for (group, bit) in groups {
                    let Some(group) = group else { continue };
                    let exact = item.tag & bit != 0;
                    // SAFETY: `at` is within the `len` bytes of the transfer,
                    // which stay as they are until the caller has waited for the
                    // stream.
                    at += unsafe {
                        group.page_in(
                            kernels,
                            stream,
                            pages,
                            status,
                            item.object,
                            item.ptr.add(at),
                            item.len - at,
                            exact,
                            item.keep_form,
                        )?
                    };
                }
                if at != item.len {
                    return Err(super::invalid(format!(
                        "a page in transit holds {} bytes; its entries here take {at}",
                        item.len
                    )));
                }
                Ok(())
            }
            TransferKind::StateOut | TransferKind::StateIn => {
                let bytes = self.places.bytes;
                let out = item.kind == TransferKind::StateOut;
                if bytes > item.room || (!out && item.len != bytes) {
                    return Err(super::invalid(format!(
                        "a checkpoint of {bytes} bytes and one of {} in transit",
                        if out { item.room } else { item.len }
                    )));
                }
                if bytes > 0 {
                    // A checkpoint leaves from memory that holds it. A place
                    // whose chunk was given back holds nothing: the transfer
                    // fails, and no new memory is locked to read from.
                    let place = match out {
                        true => self.places.held(item.object).ok_or_else(|| {
                            super::invalid(format!(
                                "checkpoint place {} holds nothing: its memory was given back",
                                item.object
                            ))
                        })?,
                        false => self.places.pointer(item.object)?,
                    };
                    // SAFETY: the place and the transfer's memory hold `bytes`
                    // bytes each and are different allocations; no copy of a
                    // step reads or writes the place now, between steps.
                    unsafe {
                        match out {
                            true => std::ptr::copy_nonoverlapping(place, item.ptr, bytes),
                            false => std::ptr::copy_nonoverlapping(item.ptr, place, bytes),
                        }
                    }
                }
                (item.len, item.tag) = (bytes, 0);
                Ok(())
            }
        }
    }

    fn check_slot(&self, slot: u32) -> Result<(), ModelError> {
        if slot >= self.max_slots {
            return Err(super::invalid(format!(
                "slot {slot} of {}: a state copy names a slot this part does not have",
                self.max_slots
            )));
        }
        Ok(())
    }

    fn slot_offset(&self, slot: u32) -> u64 {
        u64::from(slot) * self.slot_bytes
    }

    /// Copies page `from` into page `to`, in every attention block and in the
    /// draft block: the converted entry, or, where the formats have a tile
    /// mean, the exact entry (`Group::copy_page` of `blocks/kv.rs`).
    pub fn copy_page(
        &mut self,
        stream: &Stream,
        pages: DevicePtr,
        from: u32,
        to: u32,
    ) -> Result<(), ModelError> {
        if from >= self.n_pages || to >= self.n_pages || from == to {
            return Err(super::invalid(format!(
                "page copy from {from} to {to} among {} pages",
                self.n_pages
            )));
        }
        let kernels = self.attention.clone().ok_or_else(|| super::invalid("attention kernels"))?;
        let status = self.at(self.s.status);
        self.kv.copy_page(&kernels, stream, pages, status, from, to)?;
        if let Some(m) = &mut self.mtp {
            m.kv.copy_page(&kernels, stream, pages, status, from, to)?;
        }
        Ok(())
    }

    /// Enqueues the copies of a step record in the order the record's
    /// documentation gives, after the commit of the step before: the saves,
    /// the page copies, the restores. Also settles the timing of the copies of
    /// the step that ran last.
    pub fn enqueue_copies(
        &mut self,
        stream: &Stream,
        pages: DevicePtr,
        record: &RecordView<'_>,
    ) -> Result<(), ModelError> {
        self.flush_copies()?;
        let (states, pages_n) = (record.state_copies(), record.page_copies());
        if states == 0 && pages_n == 0 {
            return Ok(());
        }
        let mut counts = CopyCounts::default();
        // The chunks of the places the copies name are locked first, so that
        // the host's wait for the lock is not in the time of the copies.
        let locking = std::time::Instant::now();
        let before = self.places.fresh;
        for i in 0..states {
            self.places.pointer(record.state_copy(i).place)?;
        }
        counts.locks = self.places.fresh - before;
        counts.lock_ms = locking.elapsed().as_secs_f64() * 1e3;
        self.copy_mark(stream, 0)?;
        for i in 0..states {
            let copy = record.state_copy(i);
            if copy.kind == StateCopyKind::Save {
                self.save_state(stream, copy.slot, copy.place)?;
                counts.saves += 1;
            }
        }
        self.copy_mark(stream, 1)?;
        for i in 0..pages_n {
            let copy = record.page_copy(i);
            self.copy_page(stream, pages, copy.from, copy.to)?;
            counts.pages += 1;
        }
        self.copy_mark(stream, 2)?;
        for i in 0..states {
            let copy = record.state_copy(i);
            if copy.kind == StateCopyKind::Restore {
                self.restore_state(stream, copy.slot, copy.place)?;
                counts.restores += 1;
            }
        }
        self.copy_mark(stream, 3)?;
        if let Some(clock) = &mut self.places.clock {
            clock.pending = Some(counts);
        }
        Ok(())
    }

    fn copy_mark(&self, stream: &Stream, i: usize) -> Result<(), ModelError> {
        if let Some(clock) = &self.places.clock {
            clock.events[i].record(stream)?;
        }
        Ok(())
    }

    /// Adds the time of the copies of the step that ran last to the profile.
    /// The step must have ended.
    pub(super) fn flush_copies(&mut self) -> Result<(), ModelError> {
        let (Some(clock), Some(profile)) = (&mut self.places.clock, &self.opts.profile) else {
            return Ok(());
        };
        let Some(counts) = clock.pending.take() else { return Ok(()) };
        let ms = |a: usize, b: usize| clock.events[b].elapsed_ms_since(&clock.events[a]);
        profile.add_copies(
            self.index,
            counts,
            [f64::from(ms(0, 1)?), f64::from(ms(1, 2)?), f64::from(ms(2, 3)?)],
        );
        Ok(())
    }
}

/// What the copies of the prefix cache took on one device, added up over the
/// steps of a [`Profile`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CopyTimes {
    pub saves: u64,
    pub restores: u64,
    pub page_copies: u64,
    /// Milliseconds on the device from the first of the saves to their end, of
    /// the page copies, of the restores.
    pub save_ms: f64,
    pub page_ms: f64,
    pub restore_ms: f64,
    /// Chunks of host memory locked for places on first use, and the
    /// milliseconds the host waited for the locks, before the copies were
    /// enqueued (the device idles meanwhile).
    pub locks: u64,
    pub lock_ms: f64,
}

impl Profile {
    fn add_copies(&self, device: usize, counts: CopyCounts, ms: [f64; 3]) {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if data.copies.len() <= device {
            data.copies.resize(device + 1, CopyTimes::default());
        }
        let sum = &mut data.copies[device];
        sum.saves += u64::from(counts.saves);
        sum.restores += u64::from(counts.restores);
        sum.page_copies += u64::from(counts.pages);
        sum.save_ms += ms[0];
        sum.page_ms += ms[1];
        sum.restore_ms += ms[2];
        sum.locks += u64::from(counts.locks);
        sum.lock_ms += counts.lock_ms;
    }

    /// Per device, what the copies of the prefix cache took, over the steps
    /// since the profile was made or reset. A step's copies are counted when
    /// the next step's are enqueued.
    pub fn copies(&self) -> Vec<CopyTimes> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).copies.clone()
    }
}
