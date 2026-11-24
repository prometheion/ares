//! The parts of a model on a device that more than one architecture shares.
//!
//! A **block** owns one kind of state and every rule about it: where the state
//! lives, which rows of a step touch it how, and what a draft window, a commit,
//! a replay or a copy does to it. It has no graph and no operator list: it is
//! ordinary code that enqueues kernels, and a model calls its methods in the
//! model's own order, with device pointers its own products wrote.
//!
//! The rule for what becomes a block: a part of a model does when it owns
//! state that the runtime's contracts touch (pages, slot state, checkpoints,
//! windows, commits, replays) or when it drafts. Everything between two pieces
//! of state stays with the model, in its **launch plan**: the products, the
//! norms and gates, the activation forms, which kernel runs at which row count,
//! fused launches and the exchanges between devices.
//!
//! - [`arena`]: a device's arena of weights, laid out and uploaded the same
//!   way for every model.
//! - [`kv`]: the K/V group, attention layers that share a pair of K/V formats,
//!   with their pages and the pool of exact entries of the formats with a tile
//!   mean ([`forms`]), and the pages that lie in host memory when the context
//!   does not fit in VRAM ([`far`]).
//! - [`recurrent`]: the layers of the gated delta rule, their state in a
//!   sequence's slot and the inputs a draft window keeps.
//! - [`slots`]: the slot arena that holds every block's state per sequence,
//!   its checkpoints in host memory, the copies and transfers of the prefix
//!   cache over all K/V groups, and what the prefix cache is told.
//! - [`rows`]: where the rows of a token embedding table come from: the
//!   device, or host memory staged for a step.
//! - [`head`]: the output head's rows ordered by a draft vocabulary, which
//!   every drafter that projects to the vocabulary reads.
//! - [`mtp`]: the driver of an MTP block, for every model that drafts with
//!   one.
//! - [`step`]: what a part reads from the step record (tokens, segments,
//!   replays, rotary positions, the drafters' windows, table updates,
//!   commits) and how the drafters of a part hand out their drafts.

pub(crate) mod arena;
pub mod far;
pub(crate) mod forms;
pub(crate) mod head;
pub(crate) mod kv;
pub(crate) mod mtp;
pub(crate) mod recurrent;
pub(crate) mod rows;
pub(crate) mod slots;
pub(crate) mod step;

use ares_runtime::ModelError;

/// Bit 31 of a table entry: the page is in the pool of stored formats (the
/// only pool the formats without a tile mean use).
pub const PAGE_ENTRY: u32 = 1 << 31;

/// A request the model cannot serve as it is, as the runtime hears of it.
pub(crate) fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}
