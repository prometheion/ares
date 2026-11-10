//! Bookkeeping of the state sequences carry between steps.
//!
//! A sequence's state has two kinds of parts. Some are addressable by
//! position: the K and V rows of attention layers, the pooled keys of a
//! sparse-attention indexer. Others are not: the state of recurrent layers is
//! one blob that every token rewrites. This crate manages the first kind and
//! knows where the second kind can be resumed.
//!
//! Position-addressable state lives in **pages**: a page holds the state of a
//! fixed number of consecutive positions ("cells"), for every such part and on
//! every device. Devices keep one array per part, indexed by page number, so
//! a page number means the same thing everywhere. That gives:
//!
//! - **No memory per slot.** Sequences draw pages from one [`PagePool`] as
//!   they grow. Context that a request does not use costs nothing, and the
//!   number of concurrent sequences is not fixed at start.
//! - **Shared prefixes.** Pages are reference counted. A [`Sequence`] forked
//!   from another, or started from a cached prompt through the
//!   [`PrefixIndex`], reads the same pages and copies only the page it writes
//!   into ([`CopyOnWrite`]).
//! - **Sparse reads and offload.** Attention that selects blocks of cells
//!   (Qwen4-style indexers) reads through the page table, so only the pages a
//!   step selects must be on the GPU. Pages are the unit that moves between
//!   VRAM, pinned RAM and disk.
//!
//! Everything here is plain host-side data owned by the engine's step loop.
//! Nothing locks: one thread decides, the devices execute. The device side
//! (the arrays, the copies a [`CopyOnWrite`] asks for, the page-table upload)
//! belongs to the backends.

mod layout;
mod pool;
mod prefix;
mod sequence;

pub use layout::PageLayout;
pub use pool::{PageId, PagePool};
pub use prefix::{ChainHash, CheckpointId, Evicted, PrefixIndex, PrefixMatch};
pub use sequence::{CopyOnWrite, Grown, OutOfPages, Sequence};
