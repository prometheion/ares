//! The prefix cache of `qwen4exp` on its devices, on the shared slot arena
//! and its checkpoints (`blocks/slots.rs`).
//!
//! **A checkpoint** of a device is what a sequence resumes from there: the
//! slot of its part (the recurrent state, the carried keys of the indexers,
//! the n-gram history); on the device that holds the draft block, the
//! block's slot (the carried keys of its indexer) and its carry (the residual
//! of the sequence's last row that counts, which the block's next cell is
//! paired with); and on the device that drives the n-gram feed, what the feed
//! knows of the sequence (its position and its last tokens, from which the
//! rows of the n-gram table that its next tokens need are found). That last
//! part is the host's: it is written into the place when the save is queued
//! and read from it when the restore is.
//!
//! **A page** is the K/V entries of the part's attention layers and of the
//! draft block, each group's in the form it has, and the pooled keys of the
//! indexers of both: a page copy copies all of it, and a page goes to the
//! host and back with all of it.
//!
//! **A replay** writes neither K and V nor pooled keys: its rows read the
//! cells and keys that are cached already, and they advance the recurrent
//! state, the indexers' carried keys, the n-gram history and the feed. The
//! draft block's cells of the replayed rows are cached too (the sequence that
//! computed them had the same tokens): the block follows a replay only with
//! its carry.

use ares_cuda::{Device, DevicePtr, Stream};
use ares_runtime::{DeviceTransfer, ModelError, PinnedRam, RecordView, TransferKind};

use super::draft::DraftBlock;
use super::part::Part;
use super::runtime::Flow;
use crate::blocks::slots::{self, Checkpoint, Copies, Pages, Places};

fn model_error(error: super::Error) -> ModelError {
    match error {
        super::Error::Model(e) => e,
        other => ModelError::Invalid(other.to_string()),
    }
}

/// A device's checkpoints: what one keeps, and the places for them.
pub(super) struct Resume {
    places: Places,
    checkpoint: Checkpoint,
}

impl Resume {
    /// The checkpoints of the device of `part`: `checkpoints` places in
    /// `ram`. `block` is the draft block where it is on this device;
    /// `ngram` the bytes the n-gram feed keeps of a sequence on the device
    /// that drives it (0 on any other).
    pub(super) fn open(
        device: &Device,
        ram: &PinnedRam,
        checkpoints: u32,
        part: &Part,
        block: Option<&DraftBlock>,
        ngram: u64,
    ) -> Result<Self, ModelError> {
        let mut checkpoint = Checkpoint { extents: vec![part.arena_extent()], host: ngram };
        if let Some(block) = block {
            checkpoint.extents.extend(block.extents());
        }
        let places = Places::new(device, ram, checkpoint.bytes(), checkpoints, false)?;
        Ok(Self { places, checkpoint })
    }

    /// The bytes of one checkpoint of the device.
    pub(super) fn bytes(&self) -> u64 {
        self.checkpoint.bytes()
    }

    /// Enqueues the copies of `record` (after the commit of the step before
    /// and the block's following of it): saves, page copies, restores.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn copies(
        &mut self,
        stream: &Stream,
        pools: DevicePtr,
        record: &RecordView<'_>,
        part: &mut Part,
        block: Option<&mut DraftBlock>,
        flow: &Flow,
    ) -> Result<(), ModelError> {
        let mut holder = Holder { resume: self, part, block, flow };
        slots::enqueue_copies(&mut holder, stream, pools, record)
    }

    /// Carries out the device's share of a transfer of the prefix cache's
    /// level on SSD: a page of the pools at `pools` and of the draft block's,
    /// or a checkpoint place. Between steps; the caller waits for the stream.
    pub(super) fn transfer(
        &mut self,
        stream: &Stream,
        pools: DevicePtr,
        item: &mut DeviceTransfer,
        part: &mut Part,
        block: Option<&mut DraftBlock>,
    ) -> Result<(), ModelError> {
        match item.kind {
            TransferKind::PageOut | TransferKind::PageIn => {
                pages(part, block, pools)?.transfer(stream, item)
            }
            TransferKind::StateOut | TransferKind::StateIn => self.places.transfer(item),
        }
    }
}

/// Everything a page holds on the device: the part's K/V group and pooled
/// keys in the pools at `pools`, then the draft block's in its own.
fn pages<'a>(
    part: &'a mut Part,
    block: Option<&'a mut DraftBlock>,
    pools: DevicePtr,
) -> Result<Pages<'a>, ModelError> {
    let count = part.pages();
    let (group, mut arrays) = part.page_parts(pools)?;
    let mut groups: Vec<_> = group.into_iter().collect();
    if let Some(block) = block {
        let (group, keys) = block.page_parts()?;
        groups.extend(group);
        arrays.extend(keys);
    }
    Ok(Pages { groups, arrays, count })
}

/// What one record's copies borrow of the device.
struct Holder<'a> {
    resume: &'a mut Resume,
    part: &'a mut Part,
    block: Option<&'a mut DraftBlock>,
    flow: &'a Flow,
}

impl Copies for Holder<'_> {
    fn places(&mut self) -> &mut Places {
        &mut self.resume.places
    }

    fn save(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError> {
        self.part.check_slot(slot)?;
        let host = match self.resume.checkpoint.host {
            0 => Vec::new(),
            _ => self.flow.lock_ngram().save(slot).map_err(model_error)?,
        };
        self.resume.places.save(stream, &self.resume.checkpoint, slot, place, &host)
    }

    fn restore(&mut self, stream: &Stream, slot: u32, place: u32) -> Result<(), ModelError> {
        self.part.check_slot(slot)?;
        let mut host = vec![0u8; self.resume.checkpoint.host as usize];
        self.resume.places.restore(stream, &self.resume.checkpoint, slot, place, &mut host)?;
        if !host.is_empty() {
            self.flow.lock_ngram().restore(slot, &host).map_err(model_error)?;
        }
        Ok(())
    }

    fn copy(
        &mut self,
        stream: &Stream,
        pools: DevicePtr,
        from: u32,
        to: u32,
    ) -> Result<(), ModelError> {
        pages(self.part, self.block.as_deref_mut(), pools)?.copy(stream, from, to)
    }
}
