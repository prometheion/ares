//! The Vulkan device layer of the engine.
//!
//! This crate is the only place that talks to Vulkan. It has the role of
//! `ares-cuda` and, where Vulkan has an equivalent, the same surface:
//!
//! - [`devices`] describes the hardware without creating a logical device.
//! - [`Device`] opens one. [`DeviceBuffer`] and [`StagingBuffer`] own memory
//!   on it and next to it.
//! - [`Stream`] and [`Event`] order work on a device, by recording commands
//!   into batches (see the [`Stream`] docs for the design).
//! - [`Kernels`] holds the engine's weight kernels on a device, behind one call
//!   ([`Kernels::product`]) for a matrix in either [`Layout`]. In the file's layout:
//!   the decode kernels for every type a GGUF file can carry (integer dot products
//!   on the quantized data, one pipeline per number of activation rows) and the
//!   tile kernels on the tensor cores (`VK_KHR_cooperative_matrix`). In the device
//!   layout (twelve types, the bytes `ares_quant::layout` specifies and CUDA makes
//!   too): one family on the tensor cores for every number of rows, from one copy
//!   of the matrix, where the device's cooperative matrices allow it
//!   ([`Kernels::layout_for`]). A device without cooperative matrices keeps the
//!   file's layout and runs [`Kernels::matmul_decode`] for many rows.
//! - [`LayerKernels`] holds what a model computes between its products (embedding
//!   gather, norms, convolution window, the gated delta rule, the fused norm and
//!   SwiGLU, the rotary table), driven by the same step record as CUDA's.
//!
//! It deliberately has no tensor type and no graph. A model's forward pass is
//! ordinary code that enqueues kernels on streams; shapes are arguments, so a
//! batch of any shape runs without rebuilding anything. To keep that true on
//! Vulkan, kernels receive buffer addresses (`VK_KHR_buffer_device_address`)
//! and sizes in push constants: no pipeline is made and no descriptor is
//! written by a launch.
//!
//! # Against `ares-cuda`
//!
//! Same: `devices`, `device_count`, `DeviceInfo` (with a Vulkan UUID that
//! equals CUDA's for the same NVIDIA GPU), `Device`, `mem_info`, `MemInfo`,
//! `Wait`, `DeviceBuffer` (`new`, `slice`, bounds-checked synchronous `write`
//! and `read`), `DevicePtr`, `Stream` (`synchronize`, `is_idle`, `wait_event`,
//! `copy`, `fill`), `Event` (`new`, `timed`, `record`, `is_done`,
//! `synchronize`, `elapsed_ms_since`), `Module`, `Function`, `Args`, `Kernels`
//! with the same first four kernels, arguments, limits and results, the weight
//! layouts ([`Layout`], [`ActivationForm`], `layout_for`, `layout_bytes`,
//! `rearrange`, `measure_layout`, `activation_form`, `prepare_activations`,
//! `product`) and [`LayerKernels`] with CUDA's step record.
//!
//! Different, because Vulkan is:
//!
//! - `Device::open(index)` has no `Wait`; `open_with` takes one.
//! - A stream is a recorder: work is recorded and submitted in batches, and
//!   `flush` exists to submit early. There are no stream priorities (the
//!   device has one queue).
//! - `StagingBuffer` instead of `PinnedBuffer`, and
//!   `copy_from_staging` / `copy_to_staging` instead of `copy_from_host` /
//!   `copy_to_host`: a DMA copy names the buffer on the host side, not a raw
//!   pointer. `fill` works on aligned 32-bit words.
//! - `Function::launch` takes no block size or shared memory size (they are
//!   part of the shader).
//! - `PciAddress` has a `function`; `DeviceInfo` has the vendor, the device
//!   type, the Vulkan version, the driver version and the subgroup width.
//! - [`ActivationForm::F32`] exists: the decode kernels of the float types read the
//!   f32 rows as they are (CUDA's int8 form carries that copy inside). One launch of
//!   a decode kernel takes at most [`MAX_BATCH`] rows, so on a device without
//!   cooperative matrices [`Kernels::product`] refuses more rows for a matrix in the
//!   file's layout and [`Kernels::matmul_decode`] runs them in batches.
//! - Pipelines of the weight and layer kernels are made on first use;
//!   [`Kernels::prepare`] and [`LayerKernels::prepare`] make them ahead of a step.
//! - There are no attention, expert, sampler or `qwen4exp` kernels yet.
//! - Dropping a [`DeviceBuffer`], [`StagingBuffer`] or [`Function`] waits for
//!   the whole device (the way `cuMemFree` does), after flushing every stream.
//!
//! Several devices: [`Mesh`] sums partial results across devices and [`Lane`]
//! hands activations from one to the next, in the way of the CUDA layer's
//! `Mesh` and `Lane`, over whichever route the drivers allow (see the `mesh`
//! module's docs for what each route needs and what this machine's driver
//! does). What the layers below provide: [`SharedTimeline`] (a timeline
//! semaphore several devices share, the one cross-device event Vulkan has),
//! [`DeviceBuffer::import_host`] and [`HostPages`] (page-aligned host memory
//! imported by several devices), [`DeviceBuffer::new_exportable`],
//! [`DeviceBuffer::export`] and [`DeviceBuffer::import`] (video memory of one
//! device used by another, where the driver honours it). There is still no
//! `peer_access`, `enable_peer_access` or `Stream::copy_peer`: Vulkan has no
//! peer copy between logical devices (device groups would, see
//! [`device_groups`]; they are not built).

mod attention;
mod device;
mod error;
mod instance;
mod kernels;
mod layers;
mod memory;
mod mesh;
mod module;
mod prefill;
mod stream;
mod timeline;
mod transfer;
mod weights;

pub use attention::{
    ATTN_STATUS_DIRECT_MEAN, ATTN_STATUS_NONFINITE, ATTN_STATUS_OVERFLOW, AttendBuffers,
    AttentionKernels, AttnItem, AttnRow, AttnShape, CacheFormat, CacheGeometry, ENTRY_CELLS, Plan,
    PlanBuffers, PlanConfig, PlanLayout, Planner,
};
pub use device::{
    ComponentType, CooperativeMatrix, Device, DeviceInfo, DeviceType, MemInfo, PciAddress, Wait,
    device_count, devices,
};
pub use error::{Error, Result};
pub use kernels::{
    ActivationForm, Kernels, Layout, LayoutKernel, LayoutRows, LayoutTable, MAX_BATCH,
};
pub use layers::{
    AttentionShape, CommitSet, LayerKernels, MergeNorm, NormKind, OutputGate, RecurrentShape,
    RopeParams, Segment, StateFormat, StepDims, StepRecord, Wire, partial_wire_bytes,
    quantized_activation_bytes, quantized_activation_copy_offset,
};
pub use memory::{DeviceBuffer, DevicePtr, HostPages, ShareKind, SharedMemory, StagingBuffer};
pub use mesh::{Lane, MAX_DEVICES, Mesh, PUSH_MAX_VALUES, Parcel, Route, device_groups};
pub use module::{Args, Function, MAX_ARGS, Module};
pub use prefill::{Precision, PrefillPath};
pub use stream::{DEFAULT_BATCH, Event, MAX_IN_FLIGHT, Stream};
pub use timeline::SharedTimeline;
