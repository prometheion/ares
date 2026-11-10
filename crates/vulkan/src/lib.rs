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
//! - [`Kernels`] holds the engine's compute shaders on a device.
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
//! with the same four kernels, arguments, limits and results.
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
//! - Dropping a [`DeviceBuffer`], [`StagingBuffer`] or [`Function`] waits for
//!   the whole device (the way `cuMemFree` does), after flushing every stream.
//!
//! Left out: `peer_access`, `enable_peer_access`, `Stream::copy_peer` and
//! events between devices (a logical Vulkan device cannot name another's memory
//! or events; exported external memory and semaphores could carry them, which
//! is not built), and `HostRegistration` (`VK_EXT_external_memory_host` could
//! import page-aligned host memory, also not built).

mod device;
mod error;
mod instance;
mod kernels;
mod memory;
mod module;
mod stream;
mod transfer;
mod weights;

pub use device::{
    ComponentType, CooperativeMatrix, Device, DeviceInfo, DeviceType, MemInfo, PciAddress, Wait,
    device_count, devices,
};
pub use error::{Error, Result};
pub use kernels::{Kernels, MAX_BATCH};
pub use memory::{DeviceBuffer, DevicePtr, StagingBuffer};
pub use module::{Args, Function, MAX_ARGS, Module};
pub use stream::{DEFAULT_BATCH, Event, MAX_IN_FLIGHT, Stream};
