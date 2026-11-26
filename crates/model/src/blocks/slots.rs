//! The **slot arena and its checkpoints**: the state a device part keeps per
//! sequence, and what the prefix cache does with it and with the pages.
//!
//! - **The arena** ([`Arena`]) holds a slot for every sequence the engine
//!   runs at once. The blocks that keep state per sequence declare their
//!   extents in it one after the other ([`SlotLayout`]): the recurrent layers
//!   their convolution windows and delta-rule matrices, an indexer its
//!   carried keys, an n-gram block its history. A sequence that starts in a
//!   slot starts from zeros ([`Arena::clear_starting`]).
//! - **A checkpoint** of a device part ([`Checkpoint`]) is what a sequence
//!   resumes from on that device: its slot of the arena and whatever else a
//!   model keeps of a sequence outside it ([`Extent`]: an MTP block's carry,
//!   or the arena of an MTP block that is a part of its own), and bytes the
//!   host keeps of the sequence itself (the tokens an n-gram feed reads). It
//!   lies in page-locked host memory, in one of the places the conductor
//!   names ([`Places`]): a save is a copy from the device and a restore a copy
//!   to it, both queued on the step's stream, so the stream orders them
//!   against the kernels around them. The places are locked in chunks when
//!   they are first used, within the engine's RAM limit ([`PinnedRam`]), and
//!   a chunk in which the conductor has no checkpoint is given back when the
//!   checkpoints must make room.
//! - **A page** ([`Pages`]) is the entries of every K/V group the device holds
//!   and of every array kept per page besides them (an indexer's pooled keys):
//!   a page copy copies all of them, and a page goes to the host and back with
//!   all of them, each group's entries in the form they have.
//! - **What the prefix cache is told** ([`Declared`], [`prefix_cache`],
//!   [`durable`]) follows from what the blocks of each device declare before
//!   the parts open: the bytes of a checkpoint, the K/V groups with their
//!   formats and geometries, the bytes of the arrays.
//!
//! The order of a step record's copies on a device ([`enqueue_copies`]) is the
//! record's: after the commit of the step before, the saves, the page copies,
//! the restores, then the rows.

use std::sync::Arc;

use ares_core::Ledger;
use ares_cuda::{
    AttentionKernels, CacheFormat, CacheGeometry, Device, DeviceBuffer, DevicePtr, Event, Segment,
    Stream,
};
use ares_runtime::{
    DeviceTransfer, Durable, ModelError, PinnedRam, PlaceChunks, PrefixCache, RecordView,
    StateCopyKind, TransferKind,
};

use super::invalid;
use super::kv::{self, Group};

/// The smallest chunk of host memory that is locked at a time: places are
/// locked a chunk of them at once, so that small places do not cost a lock
/// each.
const CHUNK_BYTES: usize = 32 << 20;

/// The most partly used pages (tails) the prefix cache keeps per sequence
/// slot ([`PrefixCache::tails_per_slot`]): a tail keeps its exact K/V entries
/// for as long as it is cached, so the exact pools are sized for them.
pub const TAILS_PER_SLOT: u32 = 1;

/// The extents of a slot, as the blocks declare them one after the other:
/// every extent starts at a multiple of 64 bytes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SlotLayout {
    end: u64,
}

impl SlotLayout {
    /// An empty slot.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Places an extent of `bytes` bytes and returns where it starts.
    pub(crate) fn take(&mut self, bytes: u64) -> u64 {
        let at = self.end;
        self.end = (at + bytes).next_multiple_of(64);
        at
    }

    /// The bytes of the slot so far.
    pub(crate) fn bytes(&self) -> u64 {
        self.end
    }
}

/// The slot arena of a device part: a slot of `slot_bytes` for every
/// sequence slot of the engine.
#[derive(Debug)]
pub(crate) struct Arena {
    buffer: DeviceBuffer,
    slot_bytes: u64,
    slots: u32,
}

impl Arena {
    /// Reserves the arena in `ledger` as `what`, allocates it and clears it:
    /// a slot is cleared again when a sequence starts in it.
    pub(crate) fn new(
        device: &Device,
        ledger: &mut Ledger,
        what: &str,
        slot_bytes: u64,
        slots: u32,
    ) -> Result<Self, ModelError> {
        let bytes = (slot_bytes * u64::from(slots)).max(64);
        ledger.reserve(what, bytes)?;
        let buffer = DeviceBuffer::new(device, bytes)?;
        buffer.write(0, &vec![0u8; bytes as usize])?;
        Ok(Self { buffer, slot_bytes, slots })
    }

    /// The start of the arena on the device.
    pub(crate) fn ptr(&self) -> DevicePtr {
        self.buffer.ptr()
    }

    /// The arena as a buffer, for tests that read it.
    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    /// The bytes of one slot.
    pub(crate) fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// An error unless the arena has slot `slot`.
    pub(crate) fn check(&self, slot: u32) -> Result<(), ModelError> {
        if slot >= self.slots {
            return Err(invalid(format!(
                "slot {slot} of {}: a state copy names a slot this part does not have",
                self.slots
            )));
        }
        Ok(())
    }

    /// The slots as a checkpoint keeps them.
    pub(crate) fn extent(&self) -> Extent {
        Extent { base: self.ptr(), stride: self.slot_bytes, bytes: self.slot_bytes }
    }

    /// Clears the slots in which a sequence of `segments` starts (its first
    /// row is position 0). Queued on `stream`.
    pub(crate) fn clear_starting(
        &self,
        stream: &Stream,
        segments: &[Segment],
    ) -> Result<(), ModelError> {
        if self.slot_bytes == 0 {
            return Ok(());
        }
        for s in segments.iter().filter(|s| s.pos0 == 0) {
            self.check(s.slot)?;
            let at = self.ptr().offset(u64::from(s.slot) * self.slot_bytes);
            // SAFETY: the slot is inside the arena (checked above).
            unsafe { stream.fill(at, 0, self.slot_bytes as usize)? };
        }
        Ok(())
    }
}

/// A part of a sequence's state on the device that a checkpoint keeps: the
/// `bytes` bytes at `base + slot * stride`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extent {
    pub base: DevicePtr,
    pub stride: u64,
    pub bytes: u64,
}

/// What a checkpoint of a device part keeps: its extents on the device in
/// order, then `host` bytes that the host keeps of the sequence itself.
#[derive(Clone, Debug, Default)]
pub(crate) struct Checkpoint {
    pub extents: Vec<Extent>,
    pub host: u64,
}

impl Checkpoint {
    /// The bytes of one checkpoint.
    pub(crate) fn bytes(&self) -> u64 {
        self.extents.iter().map(|e| e.bytes).sum::<u64>() + self.host
    }

    /// Where the host's bytes start in a place.
    fn host_at(&self) -> usize {
        self.extents.iter().map(|e| e.bytes).sum::<u64>() as usize
    }
}

/// How many copies of each kind a step made, and the events around them.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CopyCounts {
    pub saves: u32,
    pub restores: u32,
    pub pages: u32,
    /// Chunks of host memory locked for the places of this step's copies, and
    /// the host's milliseconds that took.
    pub locks: u32,
    pub lock_ms: f64,
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
pub(crate) struct Places {
    /// The bytes of one place.
    bytes: usize,
    chunks: Arc<PlaceChunks>,
    /// Chunks that had to be locked for a place, since the part was opened.
    fresh: u32,
    clock: Option<CopyClock>,
}

impl Places {
    /// `capacity` places of `bytes` each in `ram`, for `device`; with `timed`
    /// the copies of a step are timed ([`Places::take_times`]).
    pub(crate) fn new(
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
            chunks: ram.place_chunks(Some(device), bytes, capacity, CHUNK_BYTES),
            fresh: 0,
            clock,
        })
    }

    /// The bytes of one place.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes as u64
    }

    /// Where place `place` is, locking the chunk it lies in when it is not
    /// locked: the first time, and after the chunk was given back.
    fn pointer(&mut self, place: u32) -> Result<*mut u8, ModelError> {
        let (pointer, fresh) = self.chunks.pointer(place)?;
        self.fresh += u32::from(fresh);
        Ok(pointer)
    }

    /// Locks the chunk place `place` lies in, if it is not locked, so that the
    /// place's later copies (the saves inside a pass) take no new memory.
    pub(crate) fn hold(&mut self, place: u32) -> Result<(), ModelError> {
        if self.bytes == 0 {
            return Ok(());
        }
        self.pointer(place).map(drop)
    }

    /// Where place `place` is when it holds memory: `None` after its chunk
    /// was given back, and before it was ever locked.
    fn held(&self, place: u32) -> Option<*mut u8> {
        self.chunks.held(place)
    }

    /// Bytes of host memory locked for the places now.
    pub(crate) fn locked(&self) -> u64 {
        self.chunks.locked()
    }

    /// Copies the checkpoint of `slot` into `place`: the extents of
    /// `checkpoint`, then `host`, the host's bytes of the sequence (as many as
    /// the checkpoint has). Queued on `stream`; the place is not read before
    /// the stream has passed, except for the host's bytes, which are there at
    /// once.
    pub(crate) fn save(
        &mut self,
        stream: &Stream,
        checkpoint: &Checkpoint,
        slot: u32,
        place: u32,
        host: &[u8],
    ) -> Result<(), ModelError> {
        if self.bytes == 0 {
            return Ok(());
        }
        let at = self.pointer(place)?;
        let mut offset = 0usize;
        for e in &checkpoint.extents {
            let bytes = e.bytes as usize;
            if bytes > 0 {
                // SAFETY: the place holds the checkpoint's bytes, the slot's
                // extent is on the device (the caller checks the slot), and
                // nothing reads or writes the place before the stream has
                // passed: restores of it are queued on the same stream, later.
                unsafe {
                    stream.copy_to_host(
                        at.add(offset),
                        e.base.offset(u64::from(slot) * e.stride),
                        bytes,
                    )?;
                }
            }
            offset += bytes;
        }
        if checkpoint.host > 0 {
            if host.len() as u64 != checkpoint.host {
                return Err(invalid(format!(
                    "{} bytes of a sequence on the host; a checkpoint keeps {}",
                    host.len(),
                    checkpoint.host
                )));
            }
            // SAFETY: the place holds the host's bytes behind the extents;
            // no copy of a step writes them.
            unsafe { std::ptr::copy_nonoverlapping(host.as_ptr(), at.add(offset), host.len()) };
        }
        Ok(())
    }

    /// Copies the bytes `range` of extent `extent` of `slot` into `place` (where the extent's
    /// bytes lie in a place), and with `host` the host's bytes of the sequence: a save made
    /// in parts (a prompt read layer by layer saves each layer's state as it goes past the
    /// position). Queued on `stream`, as [`Places::save`].
    pub(crate) fn save_part(
        &mut self,
        stream: &Stream,
        checkpoint: &Checkpoint,
        slot: u32,
        place: u32,
        extent: usize,
        range: std::ops::Range<u64>,
    ) -> Result<(), ModelError> {
        let e = checkpoint
            .extents
            .get(extent)
            .ok_or_else(|| invalid(format!("a checkpoint has no extent {extent}")))?;
        if self.bytes == 0 || range.is_empty() {
            return Ok(());
        }
        if range.end > e.bytes {
            return Err(invalid(format!("bytes {range:?} of an extent of {}", e.bytes)));
        }
        let offset: u64 = checkpoint.extents[..extent].iter().map(|e| e.bytes).sum();
        let at = self.pointer(place)?;
        // SAFETY: as in `save`; the range lies in the extent and in the place.
        unsafe {
            stream.copy_to_host(
                at.add((offset + range.start) as usize),
                e.base.offset(u64::from(slot) * e.stride + range.start),
                (range.end - range.start) as usize,
            )?;
        }
        Ok(())
    }

    /// Writes the host's bytes of a sequence into `place` (the rest of a save made in parts).
    pub(crate) fn save_host(
        &mut self,
        checkpoint: &Checkpoint,
        place: u32,
        host: &[u8],
    ) -> Result<(), ModelError> {
        if self.bytes == 0 || checkpoint.host == 0 {
            return Ok(());
        }
        if host.len() as u64 != checkpoint.host {
            return Err(invalid(format!(
                "{} bytes of a sequence on the host; a checkpoint keeps {}",
                host.len(),
                checkpoint.host
            )));
        }
        let at = self.pointer(place)?;
        // SAFETY: the place holds the host's bytes behind the extents; no copy writes them.
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), at.add(checkpoint.host_at()), host.len())
        };
        Ok(())
    }

    /// Copies `place` into the slot: the opposite of [`Places::save`]. The
    /// host's bytes of the sequence come back in `host`, at once.
    pub(crate) fn restore(
        &mut self,
        stream: &Stream,
        checkpoint: &Checkpoint,
        slot: u32,
        place: u32,
        host: &mut [u8],
    ) -> Result<(), ModelError> {
        if self.bytes == 0 {
            return Ok(());
        }
        let at = self.pointer(place)?;
        let mut offset = 0usize;
        for e in &checkpoint.extents {
            let bytes = e.bytes as usize;
            if bytes > 0 {
                // SAFETY: as in `save`; the place was written by a save that
                // ran before this copy (an earlier record, or earlier in this
                // one), or by a transfer between steps.
                unsafe {
                    stream.copy_from_host(
                        e.base.offset(u64::from(slot) * e.stride),
                        at.add(offset),
                        bytes,
                    )?;
                }
            }
            offset += bytes;
        }
        if checkpoint.host > 0 {
            if host.len() as u64 != checkpoint.host {
                return Err(invalid("the host's bytes of a checkpoint have another length"));
            }
            debug_assert_eq!(offset, checkpoint.host_at());
            // SAFETY: as in `save`.
            unsafe { std::ptr::copy_nonoverlapping(at.add(offset), host.as_mut_ptr(), host.len()) };
        }
        Ok(())
    }

    /// Carries out a transfer of a checkpoint between its place and the
    /// host's memory of the transfer ([`TransferKind::StateOut`] and
    /// [`TransferKind::StateIn`]), between steps.
    pub(crate) fn transfer(&mut self, item: &mut DeviceTransfer) -> Result<(), ModelError> {
        let bytes = self.bytes;
        let out = item.kind == TransferKind::StateOut;
        if bytes > item.room || (!out && item.len != bytes) {
            return Err(invalid(format!(
                "a checkpoint of {bytes} bytes and one of {} in transit",
                if out { item.room } else { item.len }
            )));
        }
        if bytes > 0 {
            // A checkpoint leaves from memory that holds it. A place whose
            // chunk was given back holds nothing: the transfer fails, and no
            // new memory is locked to read from.
            let place = match out {
                true => self.held(item.object).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint place {} holds nothing: its memory was given back",
                        item.object
                    ))
                })?,
                false => self.pointer(item.object)?,
            };
            // SAFETY: the place and the transfer's memory hold `bytes` bytes
            // each and are different allocations; no copy of a step reads or
            // writes the place now, between steps.
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

    /// Locks the chunks of the places the state copies of `record` name, and
    /// returns how many had to be locked and the host's milliseconds.
    fn lock_for(&mut self, record: &RecordView<'_>) -> Result<(u32, f64), ModelError> {
        let locking = std::time::Instant::now();
        let before = self.fresh;
        for i in 0..record.state_copies() {
            self.pointer(record.state_copy(i).place)?;
        }
        Ok((self.fresh - before, locking.elapsed().as_secs_f64() * 1e3))
    }

    fn mark(&self, stream: &Stream, i: usize) -> Result<(), ModelError> {
        if let Some(clock) = &self.clock {
            clock.events[i].record(stream)?;
        }
        Ok(())
    }

    /// The counts and the device milliseconds (saves, page copies, restores)
    /// of the copies of the step that ran last, once; the step must have
    /// ended. `None` when the copies are not timed or there were none.
    pub(crate) fn take_times(&mut self) -> Result<Option<(CopyCounts, [f64; 3])>, ModelError> {
        let Some(clock) = &mut self.clock else { return Ok(None) };
        let Some(counts) = clock.pending.take() else { return Ok(None) };
        let ms = |a: usize, b: usize| clock.events[b].elapsed_ms_since(&clock.events[a]);
        Ok(Some((counts, [f64::from(ms(0, 1)?), f64::from(ms(1, 2)?), f64::from(ms(2, 3)?)])))
    }
}

/// A device part that carries out the copies of a step record.
pub(crate) trait Copies {
    /// The part's checkpoint places.
    fn places(&mut self) -> &mut Places;

    /// Queues the save of the checkpoint of `slot` into `place`.
    fn save(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError>;

    /// Queues the restore of `place` into `slot`.
    fn restore(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError>;

    /// Queues the copy of page `from` into page `to`; `pages` is the start of
    /// the engine's pool of pages on the device.
    fn copy(
        &mut self,
        stream: &Stream,
        pages: DevicePtr,
        from: u32,
        to: u32,
    ) -> Result<(), ModelError>;

    /// The copies of the step that ran last took this much
    /// ([`Places::take_times`]).
    fn copied(&mut self, counts: CopyCounts, ms: [f64; 3]) {
        let _ = (counts, ms);
    }
}

/// Enqueues the copies of a step record in the order the record's
/// documentation gives, after the commit of the step before: the saves, the
/// page copies, the restores. Also settles the time of the copies of the step
/// that ran last.
pub(crate) fn enqueue_copies<P: Copies + ?Sized>(
    part: &mut P,
    stream: &Stream,
    pages: DevicePtr,
    record: &RecordView<'_>,
) -> Result<(), ModelError> {
    if let Some((counts, ms)) = part.places().take_times()? {
        part.copied(counts, ms);
    }
    let (states, pages_n) = (record.state_copies(), record.page_copies());
    if states == 0 && pages_n == 0 {
        return Ok(());
    }
    let mut counts = CopyCounts::default();
    // The chunks of the places the copies name are locked first, so that the
    // host's wait for the lock is not in the time of the copies.
    (counts.locks, counts.lock_ms) = part.places().lock_for(record)?;
    part.places().mark(stream, 0)?;
    for i in 0..states {
        let copy = record.state_copy(i);
        if copy.kind == StateCopyKind::Save {
            part.save(stream, copy.slot, copy.place)?;
            counts.saves += 1;
        }
    }
    part.places().mark(stream, 1)?;
    for i in 0..pages_n {
        let copy = record.page_copy(i);
        part.copy(stream, pages, copy.from, copy.to)?;
        counts.pages += 1;
    }
    part.places().mark(stream, 2)?;
    for i in 0..states {
        let copy = record.state_copy(i);
        if copy.kind == StateCopyKind::Restore {
            part.restore(stream, copy.slot, copy.place)?;
            counts.restores += 1;
        }
    }
    part.places().mark(stream, 3)?;
    if let Some(clock) = &mut part.places().clock {
        clock.pending = Some(counts);
    }
    Ok(())
}

/// What the copies of the prefix cache took on one device, added up over the
/// steps of a profile.
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

impl CopyTimes {
    /// Adds the copies of one step.
    pub(crate) fn add(&mut self, counts: CopyCounts, ms: [f64; 3]) {
        self.saves += u64::from(counts.saves);
        self.restores += u64::from(counts.restores);
        self.page_copies += u64::from(counts.pages);
        self.save_ms += ms[0];
        self.page_ms += ms[1];
        self.restore_ms += ms[2];
        self.locks += u64::from(counts.locks);
        self.lock_ms += counts.lock_ms;
    }
}

/// A K/V group as the pages of a device part hold it: the group, the start of
/// the memory its pool lies in, and the kernels and the status word its
/// conversions use.
pub(crate) struct PageGroup<'a> {
    pub group: &'a mut Group,
    pub pages: DevicePtr,
    pub kernels: &'a AttentionKernels,
    pub status: DevicePtr,
    /// The group writes a cell after the model's cells of it (an MTP block
    /// one token behind): a partly used page may have none of the group's
    /// cells yet, and then it has no exact entry in a group that keeps them,
    /// and a copy of it copies nothing of the group.
    pub lags: bool,
}

/// Bytes kept per page besides the K/V groups (an indexer's pooled keys of a
/// layer): page `p` has `bytes` bytes at `base + p * bytes`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PageArray {
    pub base: DevicePtr,
    pub bytes: u64,
}

/// Everything a device part keeps per page: its K/V groups, whose tag bits in
/// transit are `1 << i` for the `i`-th, then its arrays.
pub(crate) struct Pages<'a> {
    pub groups: Vec<PageGroup<'a>>,
    pub arrays: Vec<PageArray>,
    /// The pages of the pool.
    pub count: u32,
}

impl Pages<'_> {
    /// Copies page `from` into page `to`: in every K/V group the converted
    /// entry, or, where the formats have a tile mean, the exact entry
    /// (`Group::copy_page`), and the bytes of every array.
    pub(crate) fn copy(&mut self, stream: &Stream, from: u32, to: u32) -> Result<(), ModelError> {
        if from >= self.count || to >= self.count || from == to {
            return Err(invalid(format!(
                "page copy from {from} to {to} among {} pages",
                self.count
            )));
        }
        for g in &mut self.groups {
            if g.lags && g.group.has_exact() && g.group.exact_slot(from).is_none() {
                continue;
            }
            g.group.copy_page(g.kernels, stream, g.pages, g.status, from, to)?;
        }
        for a in &self.arrays {
            let place = |page: u32| a.base.offset(u64::from(page) * a.bytes);
            // SAFETY: both pages are in the array, which has an entry for
            // every page, and they are different pages (checked above).
            unsafe { stream.copy(place(to), place(from), a.bytes as usize)? };
        }
        Ok(())
    }

    /// Carries out a page's share of a transfer of the prefix cache's level
    /// on SSD ([`TransferKind::PageOut`], [`TransferKind::PageIn`]): the
    /// entries of every K/V group in the form they have, then the arrays. Runs
    /// between steps; the caller waits for the stream.
    pub(crate) fn transfer(
        &mut self,
        stream: &Stream,
        item: &mut DeviceTransfer,
    ) -> Result<(), ModelError> {
        if item.object >= self.count {
            return Err(invalid(format!("page {} of {} in a transfer", item.object, self.count)));
        }
        match item.kind {
            TransferKind::PageOut => {
                (item.len, item.tag) = (0, 0);
                for (i, g) in self.groups.iter().enumerate() {
                    // SAFETY: the transfer's memory holds `room` bytes from
                    // `item.ptr`, which the caller reads after it has waited
                    // for the stream.
                    let (bytes, exact) = unsafe {
                        let dst = item.ptr.add(item.len);
                        g.group.page_out(stream, g.pages, item.object, dst, item.room - item.len)?
                    };
                    item.len += bytes;
                    item.tag |= if exact { 1 << i } else { 0 };
                }
                for a in &self.arrays {
                    let bytes = a.bytes as usize;
                    if item.len + bytes > item.room {
                        return Err(invalid("a page in transit has less room than its entries"));
                    }
                    // SAFETY: as above; the page is in the array.
                    unsafe {
                        stream.copy_to_host(
                            item.ptr.add(item.len),
                            a.base.offset(u64::from(item.object) * a.bytes),
                            bytes,
                        )?;
                    }
                    item.len += bytes;
                }
                Ok(())
            }
            TransferKind::PageIn => {
                if item.len > item.room {
                    return Err(invalid("a page in transit is longer than its room"));
                }
                let mut at = 0usize;
                for (i, g) in self.groups.iter_mut().enumerate() {
                    let exact = item.tag & (1 << i) != 0;
                    // SAFETY: `at` is within the `len` bytes of the transfer,
                    // which stay as they are until the caller has waited for
                    // the stream.
                    at += unsafe {
                        g.group.page_in(
                            Some(g.kernels),
                            stream,
                            g.pages,
                            g.status,
                            item.object,
                            item.ptr.add(at),
                            item.len - at,
                            exact,
                            item.keep_form,
                        )?
                    };
                }
                for a in &self.arrays {
                    let bytes = a.bytes as usize;
                    if at + bytes > item.len {
                        return Err(invalid("a page in transit is shorter than its entries"));
                    }
                    // SAFETY: as above.
                    unsafe {
                        stream.copy_from_host(
                            a.base.offset(u64::from(item.object) * a.bytes),
                            item.ptr.add(at),
                            bytes,
                        )?;
                    }
                    at += bytes;
                }
                if at != item.len {
                    return Err(invalid(format!(
                        "a page in transit holds {} bytes; its entries here take {at}",
                        item.len
                    )));
                }
                Ok(())
            }
            TransferKind::StateOut | TransferKind::StateIn => {
                Err(invalid("a checkpoint is no page"))
            }
        }
    }
}

/// A K/V group as a device part declares it before it opens: the formats of
/// K and of V and the geometry of each of its layers on the device.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GroupDecl {
    pub formats: (CacheFormat, CacheFormat),
    pub layers: Vec<CacheGeometry>,
}

impl GroupDecl {
    /// The most bytes the group's entries of one page take in transit: each
    /// layer's in the larger of its two forms where its formats have two (the
    /// converted entry, and the exact one a page has while it is new).
    fn transit_bytes(&self) -> u64 {
        let exact = kv::has_exact(self.formats.0, self.formats.1);
        self.layers
            .iter()
            .map(|geo| match exact {
                true => geo.converted_entry_bytes().max(geo.exact_entry_bytes()),
                false => geo.converted_entry_bytes(),
            })
            .sum()
    }
}

/// What a device part keeps of a sequence and of a page, as its blocks
/// declare it before the part opens.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Declared {
    /// The bytes of one checkpoint ([`Checkpoint::bytes`]).
    pub checkpoint: u64,
    pub groups: Vec<GroupDecl>,
    /// The bytes of the arrays kept per page besides the groups.
    pub arrays: u64,
}

impl Declared {
    /// The most bytes the device's share of a page takes on its way to the
    /// host and back.
    pub(crate) fn transit_page_bytes(&self) -> u64 {
        self.groups.iter().map(GroupDecl::transit_bytes).sum::<u64>() + self.arrays
    }

    /// The cells before the start of a step that may still be exact entries
    /// on the device: the newest cells its groups keep exact. 0 when no group
    /// has exact entries.
    pub(crate) fn settle_cells(&self) -> u32 {
        self.groups.iter().map(|g| kv::newest_cells(g.formats.0, g.formats.1)).max().unwrap_or(0)
    }
}

/// What a model whose device parts declare `parts` offers the prefix cache.
pub(crate) fn prefix_cache(parts: &[Declared], tails_per_slot: Option<u32>) -> PrefixCache {
    PrefixCache { checkpoint_bytes: parts.iter().map(|p| p.checkpoint).collect(), tails_per_slot }
}

/// What such a model offers the prefix cache's level on SSD, with the
/// model's `fingerprint` of everything else that changes the bytes.
pub(crate) fn durable(parts: &[Declared], fingerprint: Vec<u8>) -> Durable {
    Durable {
        fingerprint,
        page_bytes: parts.iter().map(Declared::transit_page_bytes).collect(),
        state_bytes: parts.iter().map(|p| p.checkpoint).collect(),
        settle_cells: parts.iter().map(Declared::settle_cells).max().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo(k: CacheFormat, v: CacheFormat, heads: u32) -> CacheGeometry {
        CacheGeometry {
            k,
            v,
            head_dim: 128,
            kv_heads: heads,
            kv_total: heads,
            first_kv_head: 0,
            q_per_kv: 4,
        }
    }

    #[test]
    fn a_slot_lays_its_extents_out_at_multiples_of_64_bytes() {
        let mut slot = SlotLayout::new();
        assert_eq!(slot.take(100), 0);
        assert_eq!(slot.take(64), 128);
        assert_eq!(slot.take(1), 192);
        assert_eq!(slot.bytes(), 256);
    }

    #[test]
    fn a_checkpoint_counts_its_extents_and_the_hosts_bytes() {
        let at = |bytes| Extent { base: DevicePtr::NULL, stride: bytes, bytes };
        let checkpoint = Checkpoint { extents: vec![at(4096), at(0), at(512)], host: 12 };
        assert_eq!(checkpoint.bytes(), 4096 + 512 + 12);
        assert_eq!(checkpoint.host_at(), 4096 + 512);
    }

    #[test]
    fn what_the_cache_is_told_follows_from_the_declared_groups_and_arrays() {
        let q8 = (CacheFormat::Q8, CacheFormat::Q8);
        let q4 = (CacheFormat::Q4, CacheFormat::Q4);
        let (a, b) = (geo(q8.0, q8.1, 2), geo(q4.0, q4.1, 2));
        let parts = [
            Declared {
                checkpoint: 1000,
                groups: vec![GroupDecl { formats: q8, layers: vec![a, a] }],
                arrays: 77,
            },
            Declared {
                checkpoint: 500,
                groups: vec![GroupDecl { formats: q4, layers: vec![b] }],
                arrays: 0,
            },
        ];
        let cache = prefix_cache(&parts, Some(1));
        assert_eq!(cache.checkpoint_bytes, vec![1000, 500]);
        assert_eq!(cache.bytes(), 1500);
        let durable = durable(&parts, b"x".to_vec());
        assert_eq!(durable.page_bytes[0], 2 * a.converted_entry_bytes() + 77);
        assert_eq!(
            durable.page_bytes[1],
            b.converted_entry_bytes().max(b.exact_entry_bytes()),
            "a group with a tile mean moves a page in the larger of its forms"
        );
        assert_eq!(durable.state_bytes, vec![1000, 500]);
        assert_eq!(durable.settle_cells, kv::newest_cells(q4.0, q4.1));
        assert_eq!(parts[0].settle_cells(), 0, "q8 keeps nothing exact");
    }
}
