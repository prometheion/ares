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
//!   drive it ([`ModelOptions`], `options.rs`): a thread per device opens its
//!   part (`runtime/parts.rs`) and steps it (`runtime/device.rs`), with the
//!   expert store's tiers behind it (`feed.rs`, `misses.rs`, `board.rs`,
//!   `usage.rs`), the n-gram rows (`ngram.rs`), the prefix cache's checkpoints
//!   (`resume.rs`) and a long prompt run layer by layer (`runtime/pass.rs`).
//! - `open.rs` holds the stages every way of opening the parts goes through,
//!   and `flow.rs` what the threads that drive a step share: the store behind
//!   its lock, the n-gram rows, the lead over the route records, the rounds of
//!   a wide step's missed experts and the waits.
//! - `mtp.rs` runs the model's MTP block next to the part that holds the
//!   output head, for a model the runtime drives ([`Qwen4Exp::with_mtp`]).
//! - `roles.rs` times, when two devices share every layer, the second
//!   computing a share of the dense parts against it holding experts only,
//!   and keeps the faster ([`RoleTrial`]).
//! - `prune.rs` lets the routers leave out, or stand an expert of a tier in
//!   for, a pick that only the file holds ([`PruneSetup`]; off unless asked).
//! - [`Session`] steps a whole model by hand from one host thread, for the
//!   tests and the tools: the same stages and flow, with a hook that sees
//!   every layer ([`Probe`]) and the logits on the host.
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
    ExpertCosts, OpenError, Qwen4Exp, batch_rows, drafter_bytes, page_parts, place,
    reads_layer_by_layer, slot_floors, step_rows,
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

use crate::blocks::invalid;

impl Error {
    /// A request the device part cannot serve as it is.
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    /// Whether the error is a lack of memory: no room in a placement or a
    /// budget, or an allocation the driver refused.
    pub fn is_no_room(&self) -> bool {
        match self {
            Self::Cuda(e) => e.is_out_of_memory(),
            Self::Memory(_) => true,
            Self::Model(e) => e.is_no_room(),
            _ => false,
        }
    }
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
