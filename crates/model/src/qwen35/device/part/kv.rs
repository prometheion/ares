//! The K/V pages of one group of attention layers, as a device keeps them
//! besides the converted pool in the engine's page pool: the pool of exact
//! entries and the page table of the group.
//!
//! A group is the model's attention blocks, or the draft block: layers that
//! share one pair of formats and so one policy of which tiles are exact. The
//! formats `f16` and `q8` have no exact entries and no pool: the group reads
//! the step's page table, which names the converted entry of every page. With a
//! tile mean (`q7` to `q4`) the group's entries for the exact tiles name slots
//! of the exact pool, so it keeps a table of its own on the device (or, for the
//! model's attention blocks, writes into the step's) and brings it up to date
//! at the start of every step, from [`Forms`].

use ares_core::Ledger;
use ares_cuda::{
    AttentionKernels, CacheFormat, CacheGeometry, Device, DeviceBuffer, DevicePtr, PinnedBuffer,
    Stream,
};
use ares_runtime::ModelError;

use super::forms::{Convert, Exhausted, Forms, Reach, exact_slots};
use super::{PAGE_ENTRY, invalid};

/// The exact cells at the end of a sequence that a policy with these formats
/// keeps (`ares_quant::KvFormat::default_exact_newest_cells`): the larger of
/// the two formats' numbers.
pub(super) fn newest_cells(k: CacheFormat, v: CacheFormat) -> u32 {
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
pub(super) fn has_exact(k: CacheFormat, v: CacheFormat) -> bool {
    k.has_mean() || v.has_mean()
}

/// The bytes of the exact pool for layers with these geometries: a slot is an
/// entry in every layer.
pub(super) fn exact_pool_bytes(geos: &[CacheGeometry], slots: u32) -> u64 {
    geos.iter().map(|geo| geo.exact_entry_bytes()).sum::<u64>() * u64::from(slots)
}

/// What a device needs to size the exact pool of a group.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sizing {
    /// Sequences at once.
    pub slots: u32,
    /// The most rows of a step.
    pub rows: u32,
    /// The partly used pages the prefix cache may hold.
    pub tails: u32,
}

/// The exact slots a group with these formats takes, 0 without exact entries.
pub(super) fn slots_for(k: CacheFormat, v: CacheFormat, sizing: Sizing) -> u32 {
    if has_exact(k, v) {
        exact_slots(newest_cells(k, v), sizing.slots, sizing.rows, sizing.tails)
    } else {
        0
    }
}

/// A layer of a group: its geometry, and where its entries lie in the group's
/// exact pool and in the engine's pool of pages.
#[derive(Clone, Copy, Debug)]
pub(super) struct Layer {
    pub geo: CacheGeometry,
    pub pool: u64,
    pub exact: u64,
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

/// The pool and the table of one group.
#[derive(Debug)]
pub(super) struct Pages {
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
}

impl Pages {
    /// The pool and table for layers of these `geos`, with `slots` exact slots
    /// (none for a group without exact entries). `own_table` gives the group a
    /// table of its own; a group with exact entries has one when it is not the
    /// step's (`step_table`: its entries are written into the table the
    /// runtime keeps).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &Device,
        ledger: &mut Ledger,
        what: &str,
        geos: &[CacheGeometry],
        slots: u32,
        newest: u32,
        pages: u32,
        max_slots: u32,
        stride: u32,
        own_table: bool,
    ) -> Result<Self, ModelError> {
        let entries = max_slots as usize * stride as usize;
        let exact_bytes = exact_pool_bytes(geos, slots);
        if slots > 0 {
            ledger.reserve(format!("qwen35 {what}: exact K/V entries"), exact_bytes)?;
        }
        let own = if own_table {
            let bytes = (entries as u64 * 4).max(16);
            ledger.reserve(format!("qwen35 {what}: page table"), bytes)?;
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
            ledger.reserve(format!("qwen35 {what}: conversion jobs"), bytes as u64)?;
            Some(Jobs {
                host: PinnedBuffer::new(device, bytes)?,
                device: DeviceBuffer::new(device, bytes as u64)?,
                room,
            })
        } else {
            None
        };
        Ok(Self {
            forms: (slots > 0).then(|| Forms::new(slots, pages, newest)),
            exact: (slots > 0).then(|| DeviceBuffer::new(device, exact_bytes)).transpose()?,
            own,
            stride,
            mirror: if synced { vec![0; entries] } else { Vec::new() },
            stage,
            jobs,
        })
    }

    /// The group has exact entries.
    pub(super) fn has_exact(&self) -> bool {
        self.forms.is_some()
    }

    /// Bytes the pool and the table take on the device.
    pub(super) fn bytes(&self) -> u64 {
        self.exact.as_ref().map_or(0, DeviceBuffer::len)
            + self.own.as_ref().map_or(0, DeviceBuffer::len)
    }

    /// The exact pool, for tests that read it.
    pub(super) fn exact_buffer(&self) -> Option<&DeviceBuffer> {
        self.exact.as_ref()
    }

    /// The group's own page table, for tests that read it.
    #[cfg(test)]
    pub(super) fn own_table(&self) -> Option<&DeviceBuffer> {
        self.own.as_ref()
    }

    /// Exact slots in the pool, and in use.
    pub(super) fn slots(&self) -> (u32, u32) {
        self.forms.as_ref().map_or((0, 0), |f| (f.capacity(), f.in_use()))
    }

    /// The exact pool on the device (null without one).
    pub(super) fn exact(&self) -> DevicePtr {
        self.exact.as_ref().map_or(DevicePtr::NULL, DeviceBuffer::ptr)
    }

    /// The table the group's kernels read: its own, or the step's.
    pub(super) fn table(&self, step: DevicePtr) -> DevicePtr {
        self.own.as_ref().map_or(step, DeviceBuffer::ptr)
    }

    /// The group's page-table entries, which the plans of its attention are
    /// made from: what its table holds, or the conductor's entries.
    pub(super) fn entries<'a>(&'a self, conductor: &'a [u32]) -> &'a [u32] {
        if self.mirror.is_empty() { conductor } else { &self.mirror }
    }

    /// The runtime wrote `entries` into the step's table at `at`.
    pub(super) fn written(&mut self, at: usize, entries: &[u32]) {
        if self.own.is_none() && !self.mirror.is_empty() {
            self.mirror[at..at + entries.len()].copy_from_slice(entries);
        }
    }

    /// The pool freed these pages.
    pub(super) fn release(&mut self, pages: impl IntoIterator<Item = u32>) {
        if let Some(forms) = &mut self.forms {
            for page in pages {
                forms.release(page);
            }
        }
    }

    /// The first `cells` cells of the sequence in `slot` are settled: the
    /// pages they fill are full.
    pub(super) fn settled(&mut self, slot: u32, cells: u32, conductor: &[u32]) {
        let base = (slot * self.stride) as usize;
        if let Some(forms) = &mut self.forms {
            forms.settle(|tile| conductor[base + tile as usize] & !PAGE_ENTRY, cells);
        }
    }

    /// Prepares a step: converts what left its window, gives the pages the rows
    /// write exact slots, and writes the group's table where it differs from
    /// the entries those imply. Returns the conversions to run.
    pub(super) fn prepare(
        &mut self,
        stream: &Stream,
        reaches: &[Reach],
        conductor: &[u32],
        step_table: DevicePtr,
    ) -> Result<Vec<Convert>, ModelError> {
        let stride = self.stride as usize;
        if let Some(forms) = &mut self.forms {
            forms
                .prepare(reaches, |slot, tile| {
                    conductor[slot as usize * stride + tile as usize] & !PAGE_ENTRY
                })
                .map_err(exhausted)?;
        }
        if self.mirror.is_empty() {
            return Ok(Vec::new());
        }
        let Some(stage) = &mut self.stage else { return Ok(Vec::new()) };
        let table = self.own.as_ref().map_or(step_table, DeviceBuffer::ptr);
        let mut used = 0usize;
        for r in reaches {
            let base = r.slot as usize * stride;
            let mut tile = 0u32;
            let end = r.tiles().end;
            while tile < end {
                let want = |tile: u32| {
                    let page = conductor[base + tile as usize] & !PAGE_ENTRY;
                    match &self.forms {
                        Some(forms) => forms.entry(page, PAGE_ENTRY),
                        None => page | PAGE_ENTRY,
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
        Ok(self.forms.as_mut().map(Forms::take_jobs).unwrap_or_default())
    }

    /// Queues the conversion of `jobs` in every layer of the group: `layers` are
    /// the layers' geometries with their offsets in the exact pool and in the
    /// pages (`pages` is the start of the engine's pool of pages).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn convert(
        &mut self,
        kernels: &AttentionKernels,
        stream: &Stream,
        pages: DevicePtr,
        status: DevicePtr,
        layers: &[Layer],
        jobs: &[Convert],
        region: usize,
    ) -> Result<(), ModelError> {
        if jobs.is_empty() {
            return Ok(());
        }
        let device_jobs = self.upload(stream, jobs, region)?;
        for layer in layers {
            // SAFETY: the jobs name exact slots and pages of the pools of this
            // layer, which hold `capacity` and `pages` entries; the status word
            // is in the part's scratch.
            unsafe {
                kernels.convert(
                    stream,
                    &layer.geo,
                    device_jobs,
                    jobs.len() as u32,
                    self.exact().offset(layer.exact),
                    pages.offset(layer.pool),
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_page(
        &mut self,
        kernels: &AttentionKernels,
        stream: &Stream,
        pages: DevicePtr,
        status: DevicePtr,
        layers: &[Layer],
        from: u32,
        to: u32,
    ) -> Result<(), ModelError> {
        let Some(forms) = &mut self.forms else {
            for layer in layers {
                let entry = layer.geo.converted_entry_bytes();
                let place = |page: u32| pages.offset(layer.pool + u64::from(page) * entry);
                // SAFETY: both entries are in the layer's pool, which has an entry for
                // every page, and they are different entries (the caller checks).
                unsafe { stream.copy(place(to), place(from), entry as usize)? };
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
        self.convert(kernels, stream, pages, status, layers, &jobs, 0)?;
        for layer in layers {
            let entry = layer.geo.exact_entry_bytes();
            let place = |slot: u32| self.exact().offset(layer.exact + u64::from(slot) * entry);
            // SAFETY: both slots are in the exact pool of the layer, and they are different
            // slots (the target was free or given up for the copy).
            unsafe { stream.copy(place(target), place(source), entry as usize)? };
        }
        Ok(())
    }

    /// Puts `jobs` on the device, in region `region` (0 or 1), and returns
    /// where. The stream orders the copy in front of its kernels.
    fn upload(
        &mut self,
        stream: &Stream,
        jobs: &[Convert],
        region: usize,
    ) -> Result<DevicePtr, ModelError> {
        let slots = self.jobs.as_mut().ok_or_else(|| invalid("no conversion jobs"))?;
        if jobs.len() > slots.room {
            return Err(invalid(format!(
                "{} conversions in one step; the group has room for {}",
                jobs.len(),
                slots.room
            )));
        }
        let at = region * slots.room * 8;
        for (i, job) in jobs.iter().enumerate() {
            let word = at + i * 8;
            slots.host[word..word + 4].copy_from_slice(&job.exact.to_le_bytes());
            slots.host[word + 4..word + 8].copy_from_slice(&job.page.to_le_bytes());
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
        Ok(slots.device.ptr().offset(at as u64))
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
