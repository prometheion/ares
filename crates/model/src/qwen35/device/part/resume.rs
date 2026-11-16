//! What a part does for the prefix cache, on the shared slot arena and its
//! checkpoints (`blocks/slots.rs`): what it declares, its checkpoint, the
//! copies the step record carries (saves, restores, copies of pages), the
//! transfers of the level on SSD, and the rows of a replay.
//!
//! **A checkpoint** of a part is what a sequence resumes from on this device:
//! its slot of the arena (the convolution windows and delta-rule matrices of
//! its blocks) and, on the device that holds the draft block, the block's
//! carry (the final hidden state of the sequence's last row, which the
//! block's keys and values of the next row are computed from).
//!
//! **A page** is copied whole, for every attention block and for the draft
//! block's own array, because the cells of an entry lie in planes per head;
//! cells past those the sequence has are not read before it writes them.
//!
//! **A replay** appends nothing: its rows read the cells that are cached
//! already and only advance the recurrent state.

use ares_cuda::{DevicePtr, StateFormat, Stream};
use ares_runtime::{DeviceTransfer, ModelError, RecordView, TransferKind};

use super::{DRAFT_KV, Options, Part, Profile, geometry, invalid};
use crate::blocks::recurrent;
use crate::blocks::slots::{
    self, CopyCounts, CopyTimes, Declared, GroupDecl, PageGroup, Pages, Places, SlotLayout,
};
use crate::qwen35::Config;
use crate::qwen35::device::placement::Placement;

/// The bytes of one sequence's recurrent state on device `index` of
/// `placement`.
pub fn state_slot_bytes(
    c: &Config,
    placement: &Placement,
    index: usize,
    state: StateFormat,
) -> u64 {
    let r = &c.recurrent;
    let mut slot = SlotLayout::new();
    for (layer, slice) in placement.devices[index].blocks.iter().enumerate() {
        let Some(slice) = slice.as_ref().filter(|s| !c.is_attention(layer) && !s.heads.is_empty())
        else {
            continue;
        };
        let n_k = slice.heads.len() as u32;
        let shape = ares_cuda::RecurrentShape {
            n_k_heads: n_k,
            n_v_heads: n_k * (r.n_v_heads / r.n_k_heads) as u32,
            head_dim: r.head_k_dim as u32,
            conv_kernel: r.conv_kernel as u32,
        };
        recurrent::place(&shape, state, &mut slot);
    }
    slot.bytes()
}

/// The bytes of one checkpoint on device `index`: its recurrent state, and the
/// draft block's carry row where the block is.
pub fn checkpoint_bytes(
    c: &Config,
    placement: &Placement,
    index: usize,
    state: StateFormat,
) -> u64 {
    let carry =
        placement.mtp().filter(|draft| draft.device == index).map_or(0, |_| c.n_embd as u64 * 4);
    state_slot_bytes(c, placement, index, state) + carry
}

/// What device `index` of `placement` keeps of a sequence and of a page, for
/// the prefix cache: its checkpoint, the K/V group of its attention blocks
/// (every K/V entry the device holds of a page) and, where it is, the draft
/// block's group.
pub(crate) fn declared(
    c: &Config,
    placement: &Placement,
    index: usize,
    opts: &Options,
) -> Declared {
    let attention = placement.devices[index]
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(layer, slice)| {
            let slice = slice.as_ref()?;
            (c.is_attention(layer) && !slice.heads.is_empty())
                .then(|| geometry(c, opts.kv, slice.heads.len() as u32))
        })
        .collect();
    let mut groups = vec![GroupDecl { formats: opts.kv, layers: attention }];
    // The draft block's formats decide when a page has its final form on
    // every device, wherever the block is.
    if let Some(draft) = placement.mtp() {
        let layers = match draft.device == index {
            true => vec![geometry(c, DRAFT_KV, c.attention.n_head_kv as u32)],
            false => Vec::new(),
        };
        groups.push(GroupDecl { formats: DRAFT_KV, layers });
    }
    Declared { checkpoint: checkpoint_bytes(c, placement, index, opts.state), groups, arrays: 0 }
}

impl Part {
    /// The bytes of one checkpoint of this part.
    pub fn checkpoint_bytes(&self) -> u64 {
        self.places.bytes()
    }

    /// Bytes of page-locked host memory the part's checkpoint places hold now.
    pub fn checkpoint_ram(&self) -> u64 {
        self.places.locked()
    }

    /// Copies the checkpoint state of `slot` into `place` on the host: the
    /// slot's recurrent state and the draft block's carry row of the slot.
    /// Queued on `stream`; the place is not read before the stream has passed.
    pub fn save_state(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError> {
        self.state.check(slot)?;
        self.places.save(stream, &self.checkpoint, slot, place, &[])
    }

    /// Copies `place` into the slot: the opposite of [`Part::save_state`].
    pub fn restore_state(
        &mut self,
        stream: &Stream,
        slot: u32,
        place: u32,
    ) -> Result<(), ModelError> {
        self.state.check(slot)?;
        self.places.restore(stream, &self.checkpoint, slot, place, &mut [])
    }

    /// Everything the part keeps per page: the K/V group of the attention
    /// blocks and the draft block's, both in the engine's pool of pages.
    fn pages(&mut self, pages: DevicePtr) -> Result<Pages<'_>, ModelError> {
        let kernels = self.attention.as_ref().ok_or_else(|| invalid("attention kernels"))?;
        let status = self.scratch.ptr().offset(self.s.status);
        let mut groups = vec![PageGroup { group: &mut self.kv, pages, kernels, status }];
        if let Some(m) = &mut self.mtp {
            groups.push(PageGroup { group: &mut m.kv, pages, kernels, status });
        }
        Ok(Pages { groups, arrays: Vec::new(), count: self.n_pages })
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
        match item.kind {
            TransferKind::PageOut | TransferKind::PageIn => {
                if item.object >= self.n_pages {
                    return Err(invalid(format!(
                        "page {} of {} in a transfer",
                        item.object, self.n_pages
                    )));
                }
                self.pages(pages)?.transfer(stream, item)
            }
            TransferKind::StateOut | TransferKind::StateIn => self.places.transfer(item),
        }
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
            return Err(invalid(format!(
                "page copy from {from} to {to} among {} pages",
                self.n_pages
            )));
        }
        self.pages(pages)?.copy(stream, from, to)
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
        slots::enqueue_copies(self, stream, pages, record)
    }
}

impl slots::Copies for Part {
    fn places(&mut self) -> &mut Places {
        &mut self.places
    }

    fn save(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError> {
        self.save_state(stream, slot, place)
    }

    fn restore(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError> {
        self.restore_state(stream, slot, place)
    }

    fn copy(
        &mut self,
        stream: &Stream,
        pages: DevicePtr,
        from: u32,
        to: u32,
    ) -> Result<(), ModelError> {
        self.copy_page(stream, pages, from, to)
    }

    fn copied(&mut self, counts: CopyCounts, ms: [f64; 3]) {
        if let Some(profile) = &self.opts.profile {
            profile.add_copies(self.index, counts, ms);
        }
    }
}

impl Profile {
    fn add_copies(&self, device: usize, counts: CopyCounts, ms: [f64; 3]) {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if data.copies.len() <= device {
            data.copies.resize(device + 1, CopyTimes::default());
        }
        data.copies[device].add(counts, ms);
    }

    /// Per device, what the copies of the prefix cache took, over the steps
    /// since the profile was made or reset. A step's copies are counted when
    /// the next step's are enqueued.
    pub fn copies(&self) -> Vec<CopyTimes> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).copies.clone()
    }
}
