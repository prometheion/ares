//! `qwen4exp` on devices: the dense weights of a GGUF file placed on one or
//! more GPUs, the routed experts in the tiers of the expert store, and the
//! step built from the kernels of `ares-cuda`.
//!
//! - [`Placement`] decides which device holds what of which layer and what
//!   each has left for expert slots: a range of layers per device, or both
//!   devices on every layer.
//! - [`Part`] is one device's share: its dense weights in one arena, the
//!   recurrent state of every sequence slot, the K/V and key pools of its
//!   attention layers, the scratch of a step, and [`Part::enqueue`], which
//!   queues the kernels of a step and does not wait.
//! - `draft.rs` runs the model's draft block next to the part that holds the
//!   output head, for a model the runtime drives ([`Qwen4Exp::with_draft`]).
//! - [`Session`] drives a whole model, on one device or several, from one
//!   host thread: it fetches the n-gram rows, hands the residual from device
//!   to device, follows the route records and lets the expert store bring
//!   what a step misses.
//!
//! The CPU reference in [`crate::qwen4exp::reference`] is what every path
//! here is measured against (`tests/qwen4exp_device.rs`).

mod board;
mod draft;
mod feed;
mod load;
mod misses;
mod ngram;
mod part;
pub mod placement;
mod runtime;
mod session;
mod usage;

pub use board::Board;
pub use draft::{DraftReport, DraftTimes, Drafted};
pub use load::{LayerShare, Role};
pub use misses::WideStats;
pub use part::{
    ExpertAt, Options, PAGE_ENTRY, PHASES, Part, PartSetup, Phase, Ports, Profile, StepEnv,
};
pub use placement::{
    DeviceMemory, DevicePlan, PartPlan, Placement, TensorChoice, TensorPlan, ram_tier_bytes,
};
pub use runtime::{OpenError, Qwen4Exp, place, step_rows};
pub use session::{
    Fill, LayerOut, PartReport, Probe, Rows, Session, SessionOptions, Spread, StepTime, Wanted,
    read_time,
};
pub use usage::{IDLE_AFTER, Log, ProfileFound, ProfileReport, UsageStats};

/// Why a model could not be opened on its devices or a step could not run.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A part of the model on a device.
    #[error(transparent)]
    Model(#[from] ares_runtime::ModelError),
    /// The expert store.
    #[error(transparent)]
    Store(#[from] ares_runtime::expert_store::StoreError),
    /// A device call of the session itself.
    #[error(transparent)]
    Cuda(#[from] ares_cuda::Error),
    /// Reading the model file.
    #[error(transparent)]
    Io(#[from] ares_io::Error),
    /// Where the experts or the table rows are in the model file.
    #[error(transparent)]
    Layout(#[from] ares_io::LayoutError),
    /// A budget that does not hold what is asked of it.
    #[error(transparent)]
    Memory(#[from] ares_core::LedgerError),
    /// The request cannot be served as it is.
    #[error("{0}")]
    Invalid(String),
}
