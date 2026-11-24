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
//! - [`Qwen4Exp`] (`runtime.rs`) is the model as the runtime and the server
//!   drive it: a thread per device steps its part, with the expert store's
//!   tiers behind it (`feed.rs`, `misses.rs`, `board.rs`, `usage.rs`), the
//!   n-gram rows (`ngram.rs`), the prefix cache's checkpoints (`resume.rs`)
//!   and a long prompt run layer by layer (`runtime/pass.rs`).
//! - `mtp.rs` runs the model's MTP block next to the part that holds the
//!   output head, for a model the runtime drives ([`Qwen4Exp::with_mtp`]).
//! - `roles.rs` times, when two devices share every layer, the second
//!   computing a share of the dense parts against it holding experts only,
//!   and keeps the faster ([`RoleTrial`]).
//! - `prune.rs` lets the routers leave out, or stand an expert of a tier in
//!   for, a pick that only the file holds ([`PruneSetup`]; off unless asked).
//! - [`Session`] drives a whole model, on one device or several, from one
//!   host thread: it fetches the n-gram rows, hands the residual from device
//!   to device, follows the route records and lets the expert store bring
//!   what a step misses.
//!
//! The CPU reference in [`crate::qwen4exp::reference`] is what every path
//! here is measured against (`tests/qwen4exp_device.rs`).

mod board;
mod feed;
mod flow;
mod load;
mod misses;
mod mtp;
mod ngram;
mod open;
mod options;
mod part;
pub mod placement;
mod prune;
mod resume;
mod roles;
mod runtime;
mod session;
mod usage;

pub use board::Board;
pub use feed::Fill;
pub use load::{LayerShare, Role};
pub use misses::WideStats;
pub use mtp::{MtpDrafted, MtpReport, MtpTimes};
pub use open::PartReport;
pub use options::{ModelOptions, Spread};
pub use part::{
    ExpertAt, Options, PAGE_ENTRY, PHASES, Part, PartSetup, Phase, Ports, Profile, StepEnv,
};
pub use placement::{
    DeviceMemory, DevicePlan, PartPlan, Placement, TensorChoice, TensorPlan, ram_tier_bytes,
    read_time,
};
pub use prune::{PruneMask, PruneSetup, PruneStats};
pub use roles::{RoleTime, RoleTrial};
pub use runtime::{
    ExpertCosts, OpenError, Qwen4Exp, batch_rows, page_parts, place, reads_layer_by_layer,
    step_rows,
};
pub use session::{LayerOut, Probe, Rows, Session, SessionOptions, StepTime, StoreRef, Wanted};
pub use usage::{IDLE_AFTER, Log, ProfileFound, ProfileReport, UsageStats, profile_key};

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

/// A request the model cannot serve as it is, as the runtime hears of it.
fn invalid(message: impl Into<String>) -> ares_runtime::ModelError {
    ares_runtime::ModelError::Invalid(message.into())
}

/// What the runtime hears of an error of the model on its devices: the part's
/// own error as it is, anything else as its message.
impl From<Error> for ares_runtime::ModelError {
    fn from(error: Error) -> Self {
        match error {
            Error::Model(e) => e,
            other => invalid(other.to_string()),
        }
    }
}
