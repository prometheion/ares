//! Where weights live.
//!
//! A dense model's weights are placed once. A mixture-of-experts model is
//! different: each token uses a handful of the hundreds of experts per layer,
//! which ones changes slowly with the text, and the whole set is often larger
//! than VRAM, sometimes larger than RAM. So the experts are spread over
//! tiers (VRAM, pinned RAM, SSD), and which expert sits where is decided
//! continuously from what the router picks.
//!
//! This crate holds those decisions as plain bookkeeping:
//!
//! - [`ExpertHeat`] counts routed picks per layer and expert, with decay.
//! - [`SlotSet`] is one tier of one layer: a fixed number of slots, filled
//!   with the hottest experts and re-elected as the heat moves, with swaps
//!   that only take effect once their data has arrived.
//! - [`SlotSet::write_route_bias`] turns a tier into a routing constraint:
//!   the router may be told to prefer or to use only the experts that are
//!   already there. Re-deciding that set every few tokens is dynamic expert
//!   pruning: a run keeps the experts it actually uses and drops the rest,
//!   trading a bounded loss of quality for not having to fetch them.
//!
//! Reading the experts (direct I/O from the GGUF, staging, uploads) and the
//! kernels that compute with slots belong to the layers that own devices and
//! files; they act on the [`Swap`]s decided here.

mod expert;

pub use expert::{ExpertHeat, SlotSet, Swap, SwapRule};
