//! The CUDA device layer of the engine.
//!
//! This crate is the only place that talks to the NVIDIA driver. It binds the
//! driver API at run time (so the toolkit is a build dependency, not a run
//! time one), and wraps the handles the rest of the engine builds on:
//!
//! - [`devices`] and [`peer_access`] describe the hardware without touching
//!   device memory.
//! - [`Device`] opens a device. [`DeviceBuffer`] and [`PinnedBuffer`] own
//!   memory on it and next to it.
//! - [`Stream`] and [`Event`] order work on a device and between devices.
//! - [`Kernels`] holds the engine's compiled kernels on a device.
//! - [`Mesh`] sums partial results across devices (tensor split) and [`Lane`]
//!   hands activations from one device to the next (layer split), over peer
//!   access or the host, without the host waiting.
//!
//! It deliberately has no tensor type and no graph. A model's forward pass is
//! ordinary code that enqueues kernels on streams; shapes are arguments, so a
//! batch of any shape runs without rebuilding anything.

mod attention;
mod device;
mod driver;
mod error;
mod experts;
mod kernels;
mod layers;
mod memory;
mod mesh;
mod module;
mod q4;
mod stream;
mod sys;

pub use attention::{
    ATTN_STATUS_DIRECT_MEAN, ATTN_STATUS_NONFINITE, ATTN_STATUS_OVERFLOW, AttendBuffers,
    AttentionKernels, AttnItem, AttnRow, AttnShape, CacheFormat, CacheGeometry, ENTRY_CELLS, Plan,
    PlanBuffers, PlanConfig, PlanLayout, Planner,
};
pub use device::{
    Device, DeviceInfo, MemInfo, PciAddress, Wait, device_count, devices, peer_access,
};
pub use driver::{LIBRARY_ENV, driver_version};
pub use error::{Error, Result};
pub use kernels::{Kernels, MAX_BATCH, Precision, PrefillPath, PrefillShape};
pub use layers::{
    AttentionShape, CommitSet, LayerKernels, NormKind, OutputGate, RecurrentShape, RopeParams,
    Segment, StateFormat, StepDims, StepRecord, quantized_activation_bytes,
};
pub use memory::{DeviceBuffer, DevicePtr, HostRegistration, PinnedBuffer};
pub use mesh::{Lane, Mesh, PUSH_MAX_VALUES, Parcel, Route};
pub use module::{Args, Function, MAX_ARGS, Module};
pub use stream::{Event, Stream};

pub use experts::{
    EXPERT_MAX_EXPERTS, EXPERT_MAX_K, ExpertKernels, ExpertShape, RouteGroup, RouteRecord,
};

pub use q4::{
    HAND_OVER_MAX_PIECES, IndexerShape, KeyFormat, ListHandOver, NgramShape, Q4Kernels,
    SELECT_MAX_BLOCKS, SelectionRule,
};
