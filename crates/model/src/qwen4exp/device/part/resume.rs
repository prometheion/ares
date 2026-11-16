//! What a part does for the prefix cache (`blocks/slots.rs`): what it
//! declares of a sequence and of a page, the extent of its slot arena in a
//! checkpoint, the K/V group and the pooled keys a page has here, and the
//! settling of a sequence's cells in the group.
//!
//! A part's slot holds the recurrent state of its layers, the carried keys
//! of its indexers (the raw keys of the unfinished block) and the n-gram
//! history: all of it is in a checkpoint. A page holds the K/V entries of
//! its attention layers and their pooled keys: the keys of a table entry lie
//! in the slot of the key pool that bears the page's number, one array per
//! layer.

use ares_cuda::{DevicePtr, ENTRY_CELLS};
use ares_runtime::ModelError;

use super::{Frame, Options, PAGE_ENTRY, Part, invalid};
use crate::blocks::slots::{Declared, Extent, GroupDecl, PageArray, PageGroup};

impl Frame {
    /// What a part of this frame keeps of a sequence (its slot) and of a
    /// page (the K/V group and the pooled keys of its attention layers), with
    /// `opts`, the options it was made with.
    pub(crate) fn declared(&self, opts: &Options) -> Declared {
        let n = self.attention.iter().flatten().count();
        Declared {
            checkpoint: self.slot_bytes,
            groups: match n {
                0 => Vec::new(),
                n => vec![GroupDecl { formats: opts.kv, layers: vec![self.geo; n] }],
            },
            arrays: n as u64 * self.ix.key_slot_bytes(opts.keys),
        }
    }
}

impl Part {
    /// What a checkpoint keeps of the part: its slot of the arena.
    pub(crate) fn arena_extent(&self) -> Extent {
        self.state.extent()
    }

    /// An error unless the part has sequence slot `slot`.
    pub(crate) fn check_slot(&self, slot: u32) -> Result<(), ModelError> {
        self.state.check(slot)
    }

    /// The pages of the pool on this part's device: as many as the tables
    /// name.
    pub(crate) fn pages(&self) -> u32 {
        self.opts.pages()
    }

    /// What a page holds of the part in the pools at `pools`: the K/V group
    /// of its attention layers, and per attention layer the array of its
    /// pooled keys.
    pub(crate) fn page_parts(
        &mut self,
        pools: DevicePtr,
    ) -> Result<(Option<PageGroup<'_>>, Vec<PageArray>), ModelError> {
        let bytes = self.f.ix.key_slot_bytes(self.opts.keys);
        let arrays = self
            .f
            .attention
            .iter()
            .flatten()
            .map(|at| PageArray { base: pools.offset(at.key_pool), bytes })
            .collect();
        let status = self.scratch.ptr().offset(self.f.s.status);
        let group = match (&mut self.kv, &self.attention) {
            (Some(group), Some(kernels)) => {
                Some(PageGroup { group, pages: pools, kernels, status })
            }
            (Some(_), None) => return Err(invalid("a K/V group without attention kernels")),
            (None, _) => None,
        };
        Ok((group, arrays))
    }

    /// The first `cells` cells of the sequence in `slot` are settled in the
    /// K/V group, by the tables of the step enqueued last: the pages they
    /// fill are full.
    pub(crate) fn settle(&mut self, slot: u32, cells: u32) {
        if let Some(kv) = &mut self.kv {
            kv.settled(slot, cells, &self.entries);
        }
    }

    /// Whether a row of the sequence in `slot` that writes cell `cell` would
    /// write into a tile that the K/V group holds converted: a tile the
    /// sequence filled before (the cell is not the tile's first) whose page
    /// has no exact entry, because it was cached and another sequence went
    /// past it or the pool needed room. Such a tile is not written again: its
    /// other cells exist only in the converted entry.
    pub(crate) fn writes_converted(&self, slot: u32, cell: u32) -> bool {
        let Some(kv) = self.kv.as_ref().filter(|kv| kv.has_exact()) else { return false };
        if cell.is_multiple_of(ENTRY_CELLS) || slot >= self.opts.max_slots {
            return false;
        }
        let at = (slot * self.opts.table_stride() + cell / ENTRY_CELLS) as usize;
        self.entries.get(at).is_some_and(|entry| kv.exact_slot(entry & !PAGE_ENTRY).is_none())
    }
}
