//! `qwen35` on devices: the weights of a GGUF file placed on one or more
//! GPUs and the step built from the kernels of `ares-cuda`.
//!
//! - [`Placement`] decides which device holds what: everything, a slice of
//!   every block (tensor split) or a range of blocks (layer split).
//! - [`Part`] is one device's share: its weights in one arena, the recurrent
//!   state of every sequence slot, the scratch of a step, and
//!   [`Part::enqueue`], which queues the kernels of a step and does not wait.
//!   Every weight matrix lies there in the layout that timing both on the
//!   device found faster for the rows the model runs ([`Layouts`]).
//! - [`Qwen35`] is the model as the runtime drives it: it opens a part on
//!   every device's thread and hands each step to it.
//! - [`Estimate`] (`plan.rs`) is what the server asks before it opens the
//!   model: the placement of a spread and what each device would take
//!   ([`Footprint`]), without a device.
//! - [`Harness`] drives one part on one device step by step, for tests and
//!   measurements; [`Session`] drives a whole model, on one device or
//!   several, through the runtime's executor.
//!
//! The CPU reference in [`crate::qwen35::reference`] is what every path here
//! is measured against (`tests/qwen35_device.rs`).

mod choose;
mod harness;
mod load;
mod part;
pub mod placement;
mod plan;
mod runtime;
mod session;

pub use crate::blocks::rows::EmbeddingReport;
pub use choose::{Balance, LayoutProbe, LayoutReport, Layouts};
pub use harness::Harness;
pub use part::{
    CopyTimes, Exchange, MTP_KV, MtpBytes, MtpChain, MtpStep, MtpTimes, Options, PAGE_ENTRY,
    PHASES, Part, PartSetup, Phase, Profile, StepEnv, TAILS_PER_SLOT, attention_state_bytes,
    checkpoint_bytes, exact_bytes, page_bytes, partial_state_bytes, state_slot_bytes,
};
pub use placement::{BlockSlice, DevicePlan, DraftPlan, Placement, PlacementError};
pub use plan::{Estimate, Footprint, Whole, attention_by_room, fit_shares, heads_for_context};
pub use runtime::{OpenError, PartReport, Qwen35, Spread};
pub use session::{Rows, Session, StepTime, Wanted};
