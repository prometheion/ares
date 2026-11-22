//! The **K/V group**: attention layers on a device whose K and V share one
//! pair of formats (a model's attention layers, or an MTP block), with their
//! entries in the engine's pages, the pool of exact entries and the group's
//! page table. A group owns the rules of how the rows of a step touch them:
//!
//! - **Append**: the rows of a step write their K and V, except the rows of a
//!   replay, which read the cells their sequence has cached already and write
//!   none ([`Group::append`]).
//! - **Plan**: the rows of a prompt are planned together; a row that is
//!   generated, alone or in a draft window, attends as the only row of a step
//!   would, so that what was drafted changes no bit of it
//!   ([`Group::plan_rows`]).
//! - **Prepare and settle**: at the start of a step the tiles that left their
//!   sequence's newest cells are converted and the pages the rows write get
//!   exact slots ([`Group::prepare`]); the tiles of a draft window settle when
//!   its commit says how many of its rows count ([`Group::commit_windows`]).
//! - **Copy, transfer, release**: a page is copied, and moved to the host and
//!   back, in the form it has; a page the pool freed gives its exact slot back.
//! - **Far pages**: a group whose pages do not all fit in VRAM keeps the
//!   converted entries of the pages from `near` on in host memory
//!   ([`super::far`]); the device's table names them by their slot there.
//!
//! The formats `f16` and `q8` have no exact entries and no pool: the group
//! reads the step's page table, which names the converted entry of every page.
//! With a tile mean (`q7` to `q4`) the group's entries for the exact tiles name
//! slots of the exact pool, so it keeps a table of its own on the device (or,
//! for the model's attention layers, writes into the step's) and brings it up
//! to date at the start of every step, from [`Forms`].
//!
//! The products that feed a group, the norms, the queries and which kernel
//! runs at which row count are the model's.

use std::ops::Range;

use ares_core::Ledger;
use ares_cuda::{
    AttendBuffers, AttentionKernels, AttnRow, CacheFormat, CacheGeometry, Device, DeviceBuffer,
    DevicePtr, ENTRY_CELLS, Event, PinnedBuffer, Plan, PlanBuffers, PlanConfig, Planner, Segment,
    Stream,
};
use ares_runtime::ModelError;

use super::far::{FarLayers, converted_entry};
use super::forms::{Ahead, Convert, Exhausted, Forms, Reach, exact_slots, window_limit};
use super::{PAGE_ENTRY, invalid};

/// The exact cells at the end of a sequence that a policy with these formats
/// keeps (`ares_quant::KvFormat::default_exact_newest_cells`): the larger of
/// the two formats' numbers.
pub(crate) fn newest_cells(k: CacheFormat, v: CacheFormat) -> u32 {
    let of = |format: CacheFormat| match format {
        CacheFormat::F16 | CacheFormat::Q8 => 0,
        CacheFormat::Q7 => 32,
        CacheFormat::Q6 => 64,
        CacheFormat::Q5 => 128,
        CacheFormat::Q4 => 256,
    };
    of(k).max(of(v))
}

/// Whether a group with these formats keeps exact entries.
pub(crate) fn has_exact(k: CacheFormat, v: CacheFormat) -> bool {
    k.has_mean() || v.has_mean()
}

/// The bytes of the exact pool for layers with these geometries: a slot is an
/// entry in every layer.
pub(crate) fn exact_pool_bytes(geos: &[CacheGeometry], slots: u32) -> u64 {
    geos.iter().map(|geo| geo.exact_entry_bytes()).sum::<u64>() * u64::from(slots)
}

/// What a group of layers with these `geos` and `slots` exact slots takes
/// besides the pages when its table is the step's (`entries` entries): bytes
/// on the device (the exact pool, the conversion jobs) and in page-locked host
/// memory (the stage of the table, the jobs). Nothing without exact slots.
pub(crate) fn group_bytes(geos: &[CacheGeometry], slots: u32, entries: u64) -> (u64, u64) {
    if slots == 0 {
        return (0, 0);
    }
    let jobs = 2 * (2 * u64::from(slots) + 8) * 8;
    (exact_pool_bytes(geos, slots) + jobs, (entries * 4).max(16) + jobs)
}

/// What a device needs to size the exact pool of a group.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sizing {
    /// Sequences at once.
    pub slots: u32,
    /// The most rows of a step.
    pub rows: u32,
    /// The partly used pages the prefix cache may hold.
    pub tails: u32,
}

/// The exact slots a group with these formats takes, 0 without exact entries.
pub(crate) fn slots_for(k: CacheFormat, v: CacheFormat, sizing: Sizing) -> u32 {
    if has_exact(k, v) {
        exact_slots(newest_cells(k, v), sizing.slots, sizing.rows, sizing.tails)
    } else {
        0
    }
}

/// A layer of a group: its geometry, and where its entries lie in the group's
/// exact pool and in the engine's pool of pages.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layer {
    pub geo: CacheGeometry,
    pub pool: u64,
    pub exact: u64,
}

/// What the segments of a step do to the sequences of a group: `replay` are
/// the rows of the step that are a replay (rows of the whole step), which read
/// the cells their sequence has cached already and write none.
pub(crate) fn reaches(segments: &[Segment], replay: &[Range<u32>]) -> Vec<Reach> {
    segments
        .iter()
        .map(|s| Reach {
            slot: s.slot,
            pos0: s.pos0 as u32,
            rows: s.n_rows,
            writes: !replay.iter().any(|r| r.start <= s.row0 && s.row0 < r.end),
            settles: s.direct,
        })
        .collect()
}

/// The rows of `first..first + n` of a step that write cells, as ranges of
/// rows counted from `first`: every row but those of the `replay` ranges
/// (rows of the whole step, in order).
pub(crate) fn writing_rows(
    replay: &[Range<u32>],
    first: u32,
    n: u32,
) -> impl Iterator<Item = Range<u32>> + '_ {
    let end = first + n;
    let mut at = first;
    std::iter::from_fn(move || {
        while at < end {
            // Inside a replay: its rows write nothing.
            if let Some(r) = replay.iter().find(|r| r.start <= at && at < r.end) {
                at = r.end;
                continue;
            }
            // Up to the next replay or the end.
            let stop = replay
                .iter()
                .map(|r| r.start)
                .filter(|s| *s > at)
                .min()
                .map_or(end, |s| s.min(end));
            let run = at - first..stop - first;
            at = stop;
            return Some(run);
        }
        None
    })
}

/// The pairs of a conversion job on the device: the exact slot, then the page
/// whose converted entry receives it.
#[derive(Debug)]
struct Jobs {
    host: PinnedBuffer,
    device: DeviceBuffer,
    /// Jobs one region holds; a record has two flushes, each with a region of
    /// its own so that the second does not overwrite what the first copy reads.
    room: usize,
}

/// One K/V group: its layers, their exact pool and the group's table.
#[derive(Debug)]
pub(crate) struct Group {
    layers: Vec<Layer>,
    forms: Option<Forms>,
    exact: Option<DeviceBuffer>,
    /// The table of the group on the device, when it is not the step's.
    own: Option<DeviceBuffer>,
    /// Page-table entries per sequence.
    stride: u32,
    /// What the group's table holds on the device, entry by entry; empty when
    /// the group reads the step's table as the conductor wrote it.
    mirror: Vec<u32>,
    stage: Option<PinnedBuffer>,
    jobs: Option<Jobs>,
    /// Recorded behind the last copy from `stage` and from each region of the
    /// jobs, and whether it is on its way: a part that prepares several steps
    /// of its own within one step of the model (an MTP block's links), or a
    /// record with several page copies, writes a pinned buffer again only when
    /// the device has read it.
    sent: [Event; 3],
    sending: [bool; 3],
    /// The draft windows of the step last prepared: the slot and the position
    /// of the first row.
    windows: Vec<(u32, u32)>,
    /// Where the converted entries of the pages from `near` on are, when the
    /// pages do not all fit in VRAM.
    far: Option<FarLayers>,
}

impl Group {
    /// The group of `layers`, whose K and V are in `formats`, with `slots`
    /// exact slots (none for a group without exact entries). `own_table` gives
    /// the group a table of its own; a group with exact entries has one when
    /// it is not the step's (`step_table`: its entries are written into the
    /// table the runtime keeps).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &Device,
        ledger: &mut Ledger,
        what: &str,
        layers: Vec<Layer>,
        formats: (CacheFormat, CacheFormat),
        slots: u32,
        pages: u32,
        max_slots: u32,
        stride: u32,
        own_table: bool,
    ) -> Result<Self, ModelError> {
        let newest = newest_cells(formats.0, formats.1);
        let geos: Vec<CacheGeometry> = layers.iter().map(|layer| layer.geo).collect();
        let entries = max_slots as usize * stride as usize;
        let exact_bytes = exact_pool_bytes(&geos, slots);
        if slots > 0 {
            ledger.reserve(format!("{what}: exact K/V entries"), exact_bytes)?;
        }
        let own = if own_table {
            let bytes = (entries as u64 * 4).max(16);
            ledger.reserve(format!("{what}: page table"), bytes)?;
            let table = DeviceBuffer::new(device, bytes)?;
            // The mirror of the table starts as zeros, and so must the table: a
            // step writes only the entries that differ from the mirror.
            table.write(0, &vec![0u8; bytes as usize])?;
            Some(table)
        } else {
            None
        };
        let synced = slots > 0 || own_table;
        let stage = synced
            .then(|| PinnedBuffer::new(device, (entries * 4).max(16)))
            .transpose()
            .map_err(ModelError::from)?;
        let jobs = if slots > 0 {
            let room = 2 * slots as usize + 8;
            let bytes = 2 * room * 8;
            ledger.reserve(format!("{what}: conversion jobs"), bytes as u64)?;
            Some(Jobs {
                host: PinnedBuffer::new(device, bytes)?,
                device: DeviceBuffer::new(device, bytes as u64)?,
                room,
            })
        } else {
            None
        };
        Ok(Self {
            layers,
            forms: (slots > 0).then(|| Forms::new(slots, pages, newest)),
            exact: (slots > 0).then(|| DeviceBuffer::new(device, exact_bytes)).transpose()?,
            own,
            stride,
            mirror: if synced { vec![0; entries] } else { Vec::new() },
            stage,
            jobs,
            sent: [Event::new(device)?, Event::new(device)?, Event::new(device)?],
            sending: [false; 3],
            windows: Vec::new(),
            far: None,
        })
    }

    /// The converted entries of the pages from `far.near` on lie in a far pool
    /// ([`super::far::FarPool`]), at `far.layers` for the group's layers in
    /// their order.
    pub(crate) fn set_far(&mut self, far: FarLayers) -> Result<(), ModelError> {
        if far.layers.len() != self.layers.len() {
            return Err(invalid(format!(
                "a far pool of {} layers for a group of {}",
                far.layers.len(),
                self.layers.len()
            )));
        }
        self.far = Some(far);
        Ok(())
    }

    /// The first page whose converted entry is far, when some are.
    pub(crate) fn near(&self) -> Option<u32> {
        self.far.as_ref().map(|far| far.near)
    }

    /// Copies the first `slots` far slots of `layer` to `to` on the device, a
    /// copy per piece of the far pool (a copy of the driver's cannot cross
    /// from one locked range to the next).
    pub(crate) fn copy_far(
        &self,
        stream: &Stream,
        layer: &Layer,
        slots: u32,
        to: DevicePtr,
    ) -> Result<(), ModelError> {
        let Some(far) = &self.far else { return Ok(()) };
        let (from, entry) = (self.far_of(layer), layer.geo.converted_entry_bytes());
        let mut slot = 0;
        while slot < slots {
            let n = far.piece_slots.min(slots - slot);
            let at = u64::from(slot) * entry;
            // SAFETY: the slots below `slots` lie in pieces the step locked,
            // `to` holds them all (the caller's stage), and the copy stays
            // inside one piece.
            unsafe {
                stream.copy_any(to.offset(at), from.offset(at), (u64::from(n) * entry) as usize)?
            };
            slot += n;
        }
        Ok(())
    }

    /// The device's entry of the converted entry of `page`.
    pub(crate) fn converted_entry(&self, page: u32) -> u32 {
        converted_entry(self.near(), page)
    }

    /// The device's entry for an entry the conductor wrote (a page with
    /// [`PAGE_ENTRY`]).
    pub(crate) fn device_entry(&self, entry: u32) -> u32 {
        if entry & PAGE_ENTRY != 0 { self.converted_entry(entry & !PAGE_ENTRY) } else { entry }
    }

    /// The far pool of `layer` (null when no page is far).
    pub(crate) fn far_of(&self, layer: &Layer) -> DevicePtr {
        let Some(far) = &self.far else { return DevicePtr::NULL };
        self.layers
            .iter()
            .position(|l| l.pool == layer.pool && l.exact == layer.exact)
            .map_or(DevicePtr::NULL, |i| far.layers[i])
    }

    /// Where the converted entry of `page` lies in `layer` (`i`-th of the
    /// group), and whether that is in host memory.
    fn converted_place(&self, i: usize, pages: DevicePtr, page: u32) -> (DevicePtr, bool) {
        let layer = &self.layers[i];
        let entry = layer.geo.converted_entry_bytes();
        match &self.far {
            Some(far) if page >= far.near => {
                (far.layers[i].offset(u64::from(page - far.near) * entry), true)
            }
            _ => (pages.offset(layer.pool + u64::from(page) * entry), false),
        }
    }

    /// The group has exact entries.
    pub(crate) fn has_exact(&self) -> bool {
        self.forms.is_some()
    }

    /// Bytes the pool and the table take on the device.
    pub(crate) fn bytes(&self) -> u64 {
        self.exact.as_ref().map_or(0, DeviceBuffer::len)
            + self.own.as_ref().map_or(0, DeviceBuffer::len)
    }

    /// The exact pool, for tests that read it.
    pub(crate) fn exact_buffer(&self) -> Option<&DeviceBuffer> {
        self.exact.as_ref()
    }

    /// The group's own page table, for tests that read it.
    #[cfg(test)]
    pub(crate) fn own_table(&self) -> Option<&DeviceBuffer> {
        self.own.as_ref()
    }

    /// Exact slots in the pool, and in use.
    pub(crate) fn slots(&self) -> (u32, u32) {
        self.forms.as_ref().map_or((0, 0), |f| (f.capacity(), f.in_use()))
    }

    /// The exact pool on the device (null without one).
    pub(crate) fn exact(&self) -> DevicePtr {
        self.exact.as_ref().map_or(DevicePtr::NULL, DeviceBuffer::ptr)
    }

    /// Where the exact entries of `layer` are: null when the formats have no
    /// exact entries.
    pub(crate) fn exact_pool(&self, layer: &Layer) -> DevicePtr {
        if self.has_exact() { self.exact().offset(layer.exact) } else { DevicePtr::NULL }
    }

    /// The table the group's kernels read: its own, or the step's.
    pub(crate) fn table(&self, step: DevicePtr) -> DevicePtr {
        self.own.as_ref().map_or(step, DeviceBuffer::ptr)
    }

    /// The group's page-table entries, which the plans of its attention are
    /// made from: what its table holds, or the conductor's entries.
    pub(crate) fn entries<'a>(&'a self, conductor: &'a [u32]) -> &'a [u32] {
        if self.mirror.is_empty() { conductor } else { &self.mirror }
    }

    /// The runtime wrote `entries` into the step's table at `at`.
    pub(crate) fn written(&mut self, at: usize, entries: &[u32]) {
        if self.own.is_none() && !self.mirror.is_empty() {
            self.mirror[at..at + entries.len()].copy_from_slice(entries);
        }
    }

    /// The exact slot of `page`, when the group holds it as an exact entry.
    pub(crate) fn exact_slot(&self, page: u32) -> Option<u32> {
        self.forms.as_ref()?.slot_of(page)
    }

    /// Makes `page` an exact entry that holds all its cells, whatever the
    /// group kept for it before: a page that comes back from the prefix
    /// cache's level on SSD in that form. Returns its slot and the conversions
    /// that made room, which must run before the slot is written.
    pub(crate) fn adopt_exact(&mut self, page: u32) -> Result<(u32, Vec<Convert>), ModelError> {
        let forms = self.forms.as_mut().ok_or_else(|| invalid("the group has no exact entries"))?;
        forms.release(page);
        let slot = forms.ensure(page, None).map_err(exhausted)?;
        forms.settle_page(page);
        Ok((slot, forms.take_jobs()))
    }

    /// The conversion of `page` into its converted entry, when it is exact.
    pub(crate) fn convert_page(&mut self, page: u32) -> Vec<Convert> {
        match &mut self.forms {
            Some(forms) => {
                forms.convert_page(page);
                forms.take_jobs()
            }
            None => Vec::new(),
        }
    }

    /// The pool freed these pages.
    pub(crate) fn release(&mut self, pages: impl IntoIterator<Item = u32>) {
        if let Some(forms) = &mut self.forms {
            for page in pages {
                forms.release(page);
            }
        }
    }

    /// The first `cells` cells of the sequence in `slot` are settled: the
    /// pages they fill are full.
    pub(crate) fn settled(&mut self, slot: u32, cells: u32, conductor: &[u32]) {
        let base = (slot * self.stride) as usize;
        if let Some(forms) = &mut self.forms {
            forms.settle(|tile| conductor[base + tile as usize] & !PAGE_ENTRY, cells);
        }
    }

    /// Whether a row of the sequence in `slot` that writes cell `cell` would
    /// write into a tile the group holds converted: a tile the sequence filled
    /// before (the cell is not the tile's first) whose page has no exact
    /// entry, because it was cached and another sequence went past it or the
    /// pool needed room. Such a tile is not written again: its other cells
    /// exist only in the converted entry. `conductor` is the conductor's
    /// table, `stride` entries per slot.
    pub(crate) fn writes_converted(&self, slot: u32, cell: u32, conductor: &[u32]) -> bool {
        if !self.has_exact() || cell.is_multiple_of(ENTRY_CELLS) {
            return false;
        }
        let at = (slot * self.stride + cell / ENTRY_CELLS) as usize;
        conductor.get(at).is_some_and(|entry| self.exact_slot(entry & !PAGE_ENTRY).is_none())
    }

    /// The tile that the rows of the draft window of `slot` read converted
    /// from a position inside the window on, in the step last prepared, while
    /// the group's table names its exact entry.
    pub(crate) fn ahead(&self, slot: u32) -> Option<Ahead> {
        self.forms.as_ref().and_then(|forms| forms.ahead(slot))
    }

    /// The most rows of a draft window of the group, when its formats bound
    /// them (`forms::window_limit`).
    pub(crate) fn window_limit(k: CacheFormat, v: CacheFormat) -> Option<u32> {
        has_exact(k, v).then(|| window_limit(newest_cells(k, v)))
    }

    /// Prepares a step: converts what left its window, gives the pages the rows
    /// write exact slots, writes the group's table where it differs from the
    /// entries those imply, and queues the conversions in front of the step's
    /// kernels (`pages` is the start of the engine's pool of pages, `status`
    /// the word the kernels report to). `windows`: the rows of a draft window
    /// attend together and must read every tile as they would alone
    /// ([`Group::ahead`]); a group whose rows attend one at a time has no use
    /// for that.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        kernels: Option<&AttentionKernels>,
        stream: &Stream,
        reaches: &[Reach],
        conductor: &[u32],
        step_table: DevicePtr,
        pages: DevicePtr,
        status: DevicePtr,
        windows: bool,
    ) -> Result<(), ModelError> {
        let jobs = self.prepare_table(stream, reaches, conductor, step_table, windows)?;
        self.windows.clear();
        self.windows.extend(reaches.iter().filter(|r| !r.settles).map(|r| (r.slot, r.pos0)));
        self.convert(kernels, stream, pages, status, &jobs, 1)
    }

    /// The draft windows of the step last prepared ended: for each, its slot
    /// and the rows of it that count. The cells of its sequence up to them are
    /// settled.
    pub(crate) fn commit_windows(&mut self, accepted: &[(u32, u32)], conductor: &[u32]) {
        for &(slot, accepted) in accepted {
            let Some(&(_, pos0)) = self.windows.iter().find(|(s, _)| *s == slot) else { continue };
            self.settled(slot, pos0 + accepted, conductor);
        }
        self.windows.clear();
    }

    /// The first part of [`Group::prepare`]: the forms and the table. Returns
    /// the conversions to run.
    fn prepare_table(
        &mut self,
        stream: &Stream,
        reaches: &[Reach],
        conductor: &[u32],
        step_table: DevicePtr,
        windows: bool,
    ) -> Result<Vec<Convert>, ModelError> {
        let stride = self.stride as usize;
        if let Some(forms) = &mut self.forms {
            forms
                .prepare(
                    reaches,
                    |slot, tile| conductor[slot as usize * stride + tile as usize] & !PAGE_ENTRY,
                    windows,
                )
                .map_err(exhausted)?;
        }
        if self.mirror.is_empty() {
            return Ok(Vec::new());
        }
        self.wait_sent(0)?;
        let Some(stage) = &mut self.stage else { return Ok(Vec::new()) };
        let table = self.own.as_ref().map_or(step_table, DeviceBuffer::ptr);
        let mut used = 0usize;
        for r in reaches {
            let base = r.slot as usize * stride;
            let mut tile = 0u32;
            let end = r.tiles().end;
            while tile < end {
                let near = self.far.as_ref().map(|far| far.near);
                let want = |tile: u32| {
                    let page = conductor[base + tile as usize] & !PAGE_ENTRY;
                    match self.forms.as_ref().map(|forms| forms.entry(page, PAGE_ENTRY)) {
                        Some(exact) if exact & PAGE_ENTRY == 0 => exact,
                        _ => converted_entry(near, page),
                    }
                };
                if self.mirror[base + tile as usize] == want(tile) {
                    tile += 1;
                    continue;
                }
                // A run of entries that differ is written with one copy.
                let first = tile;
                while tile < end && self.mirror[base + tile as usize] != want(tile) {
                    let value = want(tile);
                    self.mirror[base + tile as usize] = value;
                    let at = used * 4;
                    stage[at..at + 4].copy_from_slice(&value.to_le_bytes());
                    used += 1;
                    tile += 1;
                }
                let n = (tile - first) as usize;
                // SAFETY: the run is `n` entries of the sequence's row of the
                // table, which holds `stride` entries per slot, and the staged
                // words are in the pinned buffer, which is not written again
                // before the step has run.
                unsafe {
                    stream.copy_from_host(
                        table.offset((base as u64 + u64::from(first)) * 4),
                        stage.as_ptr().add((used - n) * 4),
                        n * 4,
                    )?;
                }
            }
        }
        if used > 0 {
            self.sent[0].record(stream)?;
            self.sending[0] = true;
        }
        Ok(self.forms.as_mut().map(Forms::take_jobs).unwrap_or_default())
    }

    /// Queues the conversion of `jobs` in every layer of the group (`pages` is
    /// the start of the engine's pool of pages). `region` (0 or 1) is where
    /// the jobs lie on the device: a step's two flushes have one each.
    pub(crate) fn convert(
        &mut self,
        kernels: Option<&AttentionKernels>,
        stream: &Stream,
        pages: DevicePtr,
        status: DevicePtr,
        jobs: &[Convert],
        region: usize,
    ) -> Result<(), ModelError> {
        if jobs.is_empty() {
            return Ok(());
        }
        let kernels = kernels.ok_or_else(|| invalid("attention kernels"))?;
        let device_jobs = self.upload(stream, jobs, region)?;
        for layer in &self.layers {
            // SAFETY: the jobs name exact slots and pages of the pools of this
            // layer, which hold `capacity` and `pages` entries, the far ones in
            // its far pool; the status word is in the part's scratch.
            unsafe {
                kernels.convert(
                    stream,
                    &layer.geo,
                    device_jobs,
                    jobs.len() as u32,
                    self.exact().offset(layer.exact),
                    pages.offset(layer.pool),
                    self.far_of(layer),
                    status,
                )?;
            }
        }
        Ok(())
    }

    /// Copies page `from` into page `to` in every layer of the group: the
    /// converted entry, or, with exact entries, the exact one (a page that is
    /// copied is partly used, so it is exact; the copy gets an exact slot of its
    /// own, and a page that is converted to make room for it is converted first).
    pub(crate) fn copy_page(
        &mut self,
        kernels: &AttentionKernels,
        stream: &Stream,
        pages: DevicePtr,
        status: DevicePtr,
        from: u32,
        to: u32,
    ) -> Result<(), ModelError> {
        let Some(forms) = &mut self.forms else {
            for (i, layer) in self.layers.iter().enumerate() {
                let entry = layer.geo.converted_entry_bytes() as usize;
                let ((dst, far_dst), (src, far_src)) =
                    (self.converted_place(i, pages, to), self.converted_place(i, pages, from));
                // SAFETY: both entries are in the layer's pools (on the device, or
                // locked in the far pool for a page the step's tables name), and
                // they are different entries (the caller checks).
                unsafe {
                    if far_dst || far_src {
                        stream.copy_any(dst, src, entry)?;
                    } else {
                        stream.copy(dst, src, entry)?;
                    }
                }
            }
            return Ok(());
        };
        let source = forms.slot_of(from).ok_or_else(|| {
            invalid(format!(
                "page {from} is copied while it is a converted entry: only a partly used page, \
                 which is exact, is"
            ))
        })?;
        let target = forms.ensure(to, Some(from)).map_err(exhausted)?;
        let jobs = forms.take_jobs();
        self.convert(Some(kernels), stream, pages, status, &jobs, 0)?;
        for layer in &self.layers {
            let entry = layer.geo.exact_entry_bytes();
            let place = |slot: u32| self.exact().offset(layer.exact + u64::from(slot) * entry);
            // SAFETY: both slots are in the exact pool of the layer, and they are different
            // slots (the target was free or given up for the copy).
            unsafe { stream.copy(place(target), place(source), entry as usize)? };
        }
        Ok(())
    }

    /// Adds the rows of `segments` to `planner`, a planner for the geometry of
    /// the group's layers (`stride`: the entries per slot of the tables). The
    /// rows of a prompt (a direct segment of several rows) are planned
    /// together. A row that is generated, alone or in a window with drafts,
    /// attends as the only row of a step would: what was drafted must not
    /// change a bit of it. Where a tile leaves the newest cells inside the
    /// window, the rows from there on read it converted ([`Group::ahead`]).
    pub(crate) fn plan_rows(
        &self,
        planner: &mut Planner,
        segments: &[Segment],
        conductor: &[u32],
        stride: u32,
    ) {
        for s in segments {
            let table = s.slot * stride;
            let need = (s.pos0 as u32 + s.n_rows).div_ceil(ENTRY_CELLS) as usize;
            let entries = &self.entries(conductor)[table as usize..table as usize + need];
            let pos0 = s.pos0 as u32;
            if s.direct && s.n_rows > 1 {
                // The rows of a prompt.
                planner.dense(table, entries, s.row0, s.n_rows, pos0);
                continue;
            }
            let end = pos0 + s.n_rows;
            match self.ahead(s.slot).filter(|a| pos0 < a.from && a.from < end) {
                None => planner.window(table, entries, s.row0, s.n_rows, pos0, None),
                Some(a) => {
                    let before = a.from - pos0;
                    let swap = Some((a.tile, self.converted_entry(a.page)));
                    planner.window(table, entries, s.row0, before, pos0, None);
                    planner.window(
                        table,
                        entries,
                        s.row0 + before,
                        s.n_rows - before,
                        a.from,
                        swap,
                    );
                }
            }
        }
    }

    /// The attention plans of a step of `n_rows` rows with `segments`
    /// ([`Group::plan_rows`]): one for each number of KV heads among the
    /// group's layers, with that number. `state_bytes` are the bytes of
    /// partial states the device reserved for a plan.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plans(
        &self,
        config: PlanConfig,
        n_rows: u32,
        segments: &[Segment],
        conductor: &[u32],
        stride: u32,
        state_bytes: u64,
    ) -> Result<Vec<(u32, Plan)>, ModelError> {
        let mut plans: Vec<(u32, Plan)> = Vec::new();
        for layer in &self.layers {
            if plans.iter().any(|(heads, _)| *heads == layer.geo.kv_heads) {
                continue;
            }
            let mut planner = Planner::new(layer.geo, config, n_rows);
            self.plan_rows(&mut planner, segments, conductor, stride);
            let plan = planner.finish();
            let state = plan.state_bytes(&layer.geo);
            if state > state_bytes {
                return Err(invalid(format!(
                    "the attention plan needs {state} bytes of partial states, {state_bytes} are \
                     reserved (a step of {n_rows} rows, {} KV heads, {} slots; segments (rows at \
                     cell, direct): {:?}; kinds {:?})",
                    layer.geo.kv_heads,
                    plan.state_slots,
                    segments.iter().map(|s| (s.n_rows, s.pos0, s.direct)).collect::<Vec<_>>(),
                    segments
                        .iter()
                        .map(|s| {
                            let table = (s.slot * stride) as usize;
                            let need = (s.pos0 as u32 + s.n_rows).div_ceil(ENTRY_CELLS) as usize;
                            self.entries(conductor)[table..table + need]
                                .iter()
                                .map(|e| if e & PAGE_ENTRY != 0 { 'C' } else { 'E' })
                                .collect::<String>()
                        })
                        .collect::<Vec<_>>(),
                )));
            }
            plans.push((layer.geo.kv_heads, plan));
        }
        Ok(plans)
    }

    /// Appends K and V of the rows `first..first + n` of a step to `layer`,
    /// except those of a `replay` (rows of the whole step, which read the
    /// cells their sequence has cached and write none): one launch per run of
    /// rows that write. `rows` are the plan's rows of the `n` rows, `k` and
    /// `v` their keys and values (`[row][kv head][head_dim]` f32 as the
    /// geometry says), `step_table` the step's page table and `pages` the start
    /// of the engine's pool of pages.
    ///
    /// # Safety
    ///
    /// As for [`AttentionKernels::append`], for the `n` rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn append(
        &self,
        kernels: &AttentionKernels,
        stream: &Stream,
        layer: &Layer,
        rows: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        step_table: DevicePtr,
        pages: DevicePtr,
        status: DevicePtr,
        replay: &[Range<u32>],
        first: u32,
        n: u32,
    ) -> Result<(), ModelError> {
        let geo = &layer.geo;
        let (tables, exact, pool) =
            (self.table(step_table), self.exact_pool(layer), pages.offset(layer.pool));
        let far = self.far_of(layer);
        let kv_row = u64::from(geo.kv_total) * u64::from(geo.head_dim) * 4;
        for run in writing_rows(replay, first, n) {
            let at = u64::from(run.start);
            // SAFETY: the run is inside the `n` rows, which the caller's
            // contract covers.
            unsafe {
                kernels.append(
                    stream,
                    geo,
                    rows.offset(at * std::mem::size_of::<AttnRow>() as u64),
                    run.end - run.start,
                    k.offset(at * kv_row),
                    v.offset(at * kv_row),
                    tables,
                    exact,
                    pool,
                    far,
                    status,
                )?;
            }
        }
        Ok(())
    }

    /// The attention of the rows of `plan` over the cells of `layer`, from the
    /// queries `q` into `out` (gated by `gate` where the model has one), with
    /// `scratch` for the partial states. `far` is where the kernels read the
    /// layer's far entries when not in the far pool: a copy of its first slots
    /// in VRAM.
    ///
    /// # Safety
    ///
    /// As for [`AttentionKernels::attend`]: the tables hold the entries the
    /// plan names and the pools the pages those entries name.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attend(
        &self,
        kernels: &AttentionKernels,
        stream: &Stream,
        layer: &Layer,
        plan: &PlanBuffers,
        step_table: DevicePtr,
        pages: DevicePtr,
        q: DevicePtr,
        gate: Option<DevicePtr>,
        out: DevicePtr,
        scratch: DevicePtr,
        scale: f32,
        far: Option<DevicePtr>,
    ) -> Result<(), ModelError> {
        let buffers = AttendBuffers {
            tables: self.table(step_table),
            exact_pool: self.exact_pool(layer),
            converted_pool: pages.offset(layer.pool),
            far_pool: far.unwrap_or_else(|| self.far_of(layer)),
            q,
            gate,
            out,
            scratch,
        };
        // SAFETY: the caller's contract.
        unsafe { kernels.attend(stream, &layer.geo, plan, &buffers, scale)? };
        Ok(())
    }

    /// Copies the entries of `page` in the group's layers to `dst` on the
    /// host, in the form the group holds them in. Returns the bytes and
    /// whether they are exact entries.
    ///
    /// # Safety
    ///
    /// `dst` must hold `room` bytes, which the caller reads only after it has
    /// waited for the stream.
    pub(crate) unsafe fn page_out(
        &self,
        stream: &Stream,
        pages: DevicePtr,
        page: u32,
        dst: *mut u8,
        room: usize,
    ) -> Result<(usize, bool), ModelError> {
        let (slot, exact_pool) = (self.exact_slot(page), self.exact());
        let mut at = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            let (src, bytes, far) = match slot {
                Some(slot) => {
                    let entry = layer.geo.exact_entry_bytes();
                    (
                        exact_pool.offset(layer.exact + u64::from(slot) * entry),
                        entry as usize,
                        false,
                    )
                }
                None => {
                    let (src, far) = self.converted_place(i, pages, page);
                    (src, layer.geo.converted_entry_bytes() as usize, far)
                }
            };
            if at + bytes > room {
                return Err(invalid("a page in transit has less room than its entries"));
            }
            // SAFETY: the entry is in the layer's pool (the page and the slot
            // are below the pools' sizes; a far one is locked while the page
            // is in use) and `dst` has room for it (checked above); the caller
            // waits for the stream before it reads.
            unsafe {
                if far {
                    let to = DevicePtr::NULL.offset(dst.add(at) as u64);
                    stream.copy_any(to, src, bytes)?;
                } else {
                    stream.copy_to_host(dst.add(at), src, bytes)?;
                }
            }
            at += bytes;
        }
        Ok((at, slot.is_some()))
    }

    /// Copies the entries of `page` in the group's layers from `src` on the
    /// host, which holds them as exact entries (`exact`) or as converted ones.
    /// An exact page that is not to `keep` its form is converted at once.
    /// Returns the bytes read. Runs between steps: conversions that it needs
    /// are waited for.
    ///
    /// # Safety
    ///
    /// `src` must hold `len` bytes, which stay as they are until the caller
    /// has waited for the stream.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn page_in(
        &mut self,
        kernels: Option<&AttentionKernels>,
        stream: &Stream,
        pages: DevicePtr,
        status: DevicePtr,
        page: u32,
        src: *const u8,
        len: usize,
        exact: bool,
        keep: bool,
    ) -> Result<usize, ModelError> {
        let slot = if exact {
            let (slot, jobs) = self.adopt_exact(page)?;
            self.convert_now(kernels, stream, pages, status, &jobs)?;
            Some(slot)
        } else {
            // Whatever the group kept for the page's number belongs to a page
            // that is gone.
            self.release([page]);
            None
        };
        let exact_pool = self.exact();
        let mut at = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            let (dst, bytes, far) = match slot {
                Some(slot) => {
                    let entry = layer.geo.exact_entry_bytes();
                    (
                        exact_pool.offset(layer.exact + u64::from(slot) * entry),
                        entry as usize,
                        false,
                    )
                }
                None => {
                    let (dst, far) = self.converted_place(i, pages, page);
                    (dst, layer.geo.converted_entry_bytes() as usize, far)
                }
            };
            if at + bytes > len {
                return Err(invalid("a page in transit is shorter than its entries"));
            }
            // SAFETY: as in `page_out`, the other way; `src` holds `len` bytes
            // and stays as it is until the caller has waited for the stream.
            unsafe {
                if far {
                    let from = DevicePtr::NULL.offset(src.add(at) as u64);
                    stream.copy_any(dst, from, bytes)?;
                } else {
                    stream.copy_from_host(dst, src.add(at), bytes)?;
                }
            }
            at += bytes;
        }
        if exact && !keep {
            let jobs = self.convert_page(page);
            self.convert_now(kernels, stream, pages, status, &jobs)?;
        }
        Ok(at)
    }

    /// Runs conversions that a page on its way in made necessary, and waits
    /// for them: the group's list of jobs on the device is written once per
    /// step otherwise, and a transfer is no step.
    fn convert_now(
        &mut self,
        kernels: Option<&AttentionKernels>,
        stream: &Stream,
        pages: DevicePtr,
        status: DevicePtr,
        jobs: &[Convert],
    ) -> Result<(), ModelError> {
        if jobs.is_empty() {
            return Ok(());
        }
        self.convert(kernels, stream, pages, status, jobs, 0)?;
        Ok(stream.synchronize()?)
    }

    /// Puts `jobs` on the device, in region `region` (0 or 1), and returns
    /// where. The stream orders the copy in front of its kernels.
    fn upload(
        &mut self,
        stream: &Stream,
        jobs: &[Convert],
        region: usize,
    ) -> Result<DevicePtr, ModelError> {
        self.wait_sent(1 + region)?;
        let slots = self.jobs.as_mut().ok_or_else(|| invalid("no conversion jobs"))?;
        if jobs.len() > slots.room {
            return Err(invalid(format!(
                "{} conversions in one step; the group has room for {}",
                jobs.len(),
                slots.room
            )));
        }
        let at = region * slots.room * 8;
        let near = self.far.as_ref().map(|far| far.near);
        for (i, job) in jobs.iter().enumerate() {
            let word = at + i * 8;
            let to = converted_entry(near, job.page) & !PAGE_ENTRY;
            slots.host[word..word + 4].copy_from_slice(&job.exact.to_le_bytes());
            slots.host[word + 4..word + 8].copy_from_slice(&to.to_le_bytes());
        }
        // SAFETY: both buffers hold `room` jobs per region; the pinned one is
        // not written again before the step has run.
        unsafe {
            stream.copy_from_host(
                slots.device.ptr().offset(at as u64),
                slots.host.as_ptr().add(at),
                jobs.len() * 8,
            )?;
        }
        let at = slots.device.ptr().offset(at as u64);
        self.sent[1 + region].record(stream)?;
        self.sending[1 + region] = true;
        Ok(at)
    }

    /// Waits until the device has read pinned buffer `which` (0: the stage of
    /// the table, 1 and 2: the regions of the jobs) since it was last sent.
    fn wait_sent(&mut self, which: usize) -> Result<(), ModelError> {
        if std::mem::take(&mut self.sending[which]) {
            self.sent[which].synchronize()?;
        }
        Ok(())
    }
}

fn exhausted(e: Exhausted) -> ModelError {
    invalid(format!(
        "{e}: the exact entries of the K/V formats with a tile mean are sized for the sequences \
         the engine runs at once"
    ))
}

#[cfg(test)]
mod tests;
