//! The CUDA device layer of the engine.
//!
//! This crate is the only place that talks to the NVIDIA driver. It binds the
//! driver API at run time (so the toolkit is a build dependency, not a run
//! time one), and wraps the handles the rest of the engine builds on:
//!
//! - [`devices`] and [`peer_access`] describe the hardware without touching
//!   device memory.
//! - [`Device`] opens a device of a class this build runs ([`BUILT_CLASSES`]:
//!   the tested ones; other GPUs are for the Vulkan backend). [`DeviceBuffer`]
//!   and [`PinnedBuffer`] own memory on it and next to it.
//! - [`Stream`] and [`Event`] order work on a device and between devices.
//! - [`Kernels`] holds the engine's compiled kernels on a device;
//!   [`TowerKernels`] those of the image tower besides its products.
//! - [`Mesh`] sums partial results across devices (tensor split) and [`Lane`]
//!   hands activations from one device to the next (layer split), over peer
//!   access or the host, without the host waiting.
//!
//! It deliberately has no tensor type and no graph. A model's forward pass is
//! ordinary code that enqueues kernels on streams; shapes are arguments, so a
//! batch of any shape runs without rebuilding anything.

mod attention;
mod device;
mod dflash;
mod driver;
mod error;
mod experts;
mod kernels;
mod layers;
mod memory;
mod mesh;
mod module;
mod q4;
mod sampler;
mod stream;
mod sys;
mod tower;

pub use attention::{
    ATTN_STATUS_DIRECT_MEAN, ATTN_STATUS_NONFINITE, ATTN_STATUS_OVERFLOW, AttendBuffers,
    AttentionKernels, AttnItem, AttnRow, AttnShape, CacheFormat, CacheGeometry, ENTRY_CELLS, Plan,
    PlanBuffers, PlanConfig, PlanLayout, Planner,
};
pub use device::{
    BUILT_CLASSES, Device, DeviceInfo, MemInfo, PciAddress, Wait, device_count, devices,
    peer_access,
};
pub use dflash::{
    DFLASH_HEAD_DIM, DFLASH_MAX_BLOCK, DFLASH_MAX_RANK, DFLASH_MAX_TOP_K, DflashKernels,
    DflashSelect, DflashSeq, Ring, Selector,
};
pub use driver::{LIBRARY_ENV, driver_version};
pub use error::{Error, Result};
pub use experts::{
    Accumulate, EXPERT_MAX_EXPERTS, EXPERT_MAX_K, ExpertKernels, ExpertRowKernels, ExpertShape,
    HostShare, NearEntry, PredictedList, Prune, PruneLayer, PruneRule, PruneWeights, ROWS_TILES,
    RouteGroup, RouteRecord, RowsItem, RowsPlan, RowsPlanLayout,
};
pub use kernels::{
    ActivationForm, FEW_ROWS, FewPlan, Kernels, LayoutKernel, LayoutRows, LayoutTable, MAX_BATCH,
    Precision, PrefillChoice, PrefillPath, PrefillShape, WeightLayout, WeightRange,
};
pub use layers::{
    AttentionShape, CommitSet, EmbedRows, KeptRows, LayerKernels, NormKind, OutputGate,
    RecurrentShape, RopeParams, Segment, StateFormat, StepDims, StepRecord,
    quantized_activation_bytes,
};
pub use layers::{MERGE_MAX_WIDTH, MergeNorm, Wire, partial_wire_bytes};
pub use memory::{DeviceBuffer, DevicePtr, HostRegistration, PinnedBuffer};
pub use mesh::{Lane, Mesh, PUSH_MAX_ROWS, PUSH_MAX_VALUES, Parcel, PushSlot, Route};
pub use module::{Args, Function, MAX_ARGS, Module};
pub use q4::{
    Arrival, Departure, HAND_OVER_MAX_PIECES, IndexerShape, KeyFormat, ListHandOver, MixerMatrix,
    NgramShape, Q4Kernels, ReadWrite, SELECT_MAX_BLOCKS, SelectionRule,
};
pub use sampler::{
    Descent, PassMode, Phase, SAMPLE_MAX_VOCAB, SampleBatch, SamplePlan, SampleStep, SamplerKernels,
};
pub use stream::{Event, Stream};
pub use tower::{HeadShape, PositionGrid, TowerKernels};
