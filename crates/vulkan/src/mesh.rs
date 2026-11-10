//! Several devices working on one step: the counterpart of the CUDA layer's
//! `mesh` module.
//!
//! In a tensor split every device holds a slice of each layer and computes a
//! partial result; the layer's output is the sum of the partials, and every
//! device needs that sum to go on. [`Mesh::sum_f32`] is that exchange.
//! [`Lane`] is the other thing a split needs: the hand-over of activations from
//! the device that holds one range of layers to the one that holds the next.
//!
//! Both are built so that the host does not wait inside a step: everything is
//! enqueued on the devices' streams and the order between devices is kept on
//! the devices themselves (the [`Route::Staged`] route is the exception, see
//! below). What differs from CUDA is how little Vulkan lets two logical
//! devices share.
//!
//! # What the routes are, and what they need
//!
//! A mesh picks the best route that **works**, by building each in turn and
//! running a real exchange over it when the mesh is made: that a Vulkan call
//! succeeds does not say that the data arrives (see the findings below), and
//! nothing here depends on a device's name.
//!
//! - [`Route::Peer`]: memory of each device is exported
//!   (`VK_KHR_external_memory_fd`, as an `OPAQUE_FD` or a `DMA_BUF`) and
//!   imported by the others, which then write straight into the other's video
//!   memory with a kernel or the copy engine. Needs drivers that implement the
//!   import across devices (RADV between two AMD cards, for one); it is the
//!   direct route and the fastest one when it exists.
//! - [`Route::Host`]: both devices import the same page-locked host pages
//!   (`VK_EXT_external_memory_host`) and exchange data through them. Needs
//!   nothing but that extension and shared timeline semaphores
//!   (`VK_KHR_external_semaphore_fd`), which every desktop driver of the last
//!   years has.
//! - [`Route::Staged`]: each device has its own host-visible buffer and the
//!   **host** copies between them, waiting for the devices inside every sum.
//!   Works everywhere; it is the safety net, not a route to run a model on.
//!
//! Device groups (`VK_KHR_device_group`) are not a route yet: the loader lists
//! them ([`device_groups`]) and [`Mesh::new`] says when it finds one it does
//! not use, but a logical device spanning a group (peer memory, `deviceMask`
//! copies) is not built, because no machine at hand reports a group.
//!
//! # How the order between devices is kept
//!
//! **Copy path** (any size, any number of devices; the only one on the host
//! route): each device copies its partial into the landing zone of each other
//! device, sets its counter (a [`SharedTimeline`]) to the number of the sum,
//! waits for the others' counters, takes their partials out of its own zone and
//! adds them in device order. Counters are waited for by the devices, in
//! batches the host has submitted, so the host never waits. Each device costs
//! one submission per sum.
//!
//! **Push path** (two devices, up to [`PUSH_MAX_VALUES`] values): one kernel
//! per device and no copy and no semaphore. Each kernel writes its partial
//! into the other's receive words and polls its own. Every word carries the
//! number of its sum next to the f32, so a word is complete when its tag is
//! right: the data is its own flag, and nothing relies on a memory fence
//! reaching across the bus (NCCL's "LL" protocol). The polls are atomics, and
//! a kernel gives up after a bounded number of them (the result is NaN
//! everywhere), so a partner that never arrives cannot hang a device for good.
//! This is the fast path where the receive words are the receiver's own video
//! memory (a peer route). It needs the receiver to see the other device's
//! writes while it polls, which is **not** the case for host memory on this
//! driver (see below); the mesh runs a test exchange over the push kernels when
//! it is made and keeps them only if they work ([`Mesh::push_values`],
//! [`Mesh::notes`]). **They have never run concurrently on this machine**: the
//! GPUs cannot share video memory, and two contexts of one GPU are time-sliced
//! without preempting a spinning kernel. The code is there, built to the same
//! contract as CUDA's push kernel, and untested for lack of hardware.
//!
//! # What is measured on the development machine
//!
//! RTX 4070 (PCIe 4.0 x16) + RTX 3060 (PCIe 3.0 x4), driver 615.78.08, patched
//! for peer-to-peer in CUDA (see `docs/ARCHITECTURE.md`, "Several devices");
//! `examples/vk_mesh_probe.rs` prints all of this:
//!
//! - Memory exported as `OPAQUE_FD` by one card is **imported without error by
//!   the other** (`vkAllocateMemory` returns success in both directions), but
//!   it is not the same memory: writes through the import never reach the
//!   owner, and reads return what the importer wrote earlier. The cards have
//!   different `deviceUUID`s, for which the specification says the import is
//!   invalid; the driver does not refuse it. So there is no direct route on
//!   this driver, and the mesh finds out by writing through the import before
//!   it trusts it. (Two logical devices of one GPU do share memory this way.)
//! - `DMA_BUF`: exporting works, but the importing card reports no memory
//!   type it can place the dma-buf in (`vkGetMemoryFdPropertiesKHR` succeeds
//!   with `memoryTypeBits == 0`).
//! - Importing the other card's video memory mapped through the PCI aperture
//!   as host memory (the trick that would give peer-to-peer through the host
//!   address space) is refused with `ERROR_OUT_OF_DEVICE_MEMORY`.
//! - `VK_KHR_device_group`: the loader reports two groups of one device.
//! - Host pages imported by both cards work in every direction: kernels and
//!   the copy engine of either card write them and read them, and a write by
//!   one card is read by the other.
//! - A kernel that polls host memory while someone else writes it **never sees
//!   the write**: 16 million polls with a `volatile` load, an `atomicAdd(x, 0)`
//!   or an `atomicLoad` on either card, with the CPU writing the word 50 ms
//!   after the kernel started. The GPU keeps the line in its cache; a write
//!   that happens before the kernel starts is seen. So flags and push kernels
//!   cannot be built on host pages here, and the host route's order is kept
//!   by semaphores. (A related trap of GLSL: a `volatile coherent` buffer
//!   reference is only a decoration of the block's member, not of the loads
//!   `glslang` emits, and the driver's compiler hoists such a load out of a
//!   polling loop: poll with atomics.)
//! - A timeline semaphore exported by one card and imported by the other works:
//!   the order is kept on the devices.
//!
//! So on this machine the mesh takes [`Route::Host`] with the copy path
//! only. The numbers are in `examples/vk_link_bench.rs`.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use ash::vk;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::instance::instance;
use crate::memory::{DeviceBuffer, DevicePtr, HostPages, ShareKind, StagingBuffer};
use crate::module::{Args, Function, Module};
use crate::stream::{Event, Stream, StreamInner};
use crate::timeline::SharedTimeline;

static MESH_SUM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh_sum.spv"));
static MESH_PUSH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh_push.spv"));

/// The largest sum the push kernels take, in f32 values (512 KiB). Larger
/// partials are bound by the link's bandwidth, not by its latency, and the
/// copy engine moves them without occupying the GPU's processors.
pub const PUSH_MAX_VALUES: u32 = 131_072;

/// The most devices a mesh has (the sum kernel adds up to eight operands).
pub const MAX_DEVICES: usize = 8;

/// Workgroups of a push kernel. Few, so that all are resident at once; each
/// goes through the values in strides.
const PUSH_BLOCKS: u32 = 16;

/// Polls of a word after which a push kernel gives up: seconds, far beyond any
/// real wait for the other device's layer.
const PUSH_POLLS: u32 = 1 << 24;

/// The same while a route is being tried: tens of milliseconds to half a second.
const PROBE_POLLS: u32 = 1 << 18;

/// How partial results travel between devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Directly into the other device's video memory, imported from it.
    Peer,
    /// Through host pages that all devices import.
    Host,
    /// Through host buffers of the devices that the host copies between,
    /// waiting inside the sum.
    Staged,
}

/// The device groups that the Vulkan loader reports with more than one
/// physical device, as the ordinals [`Device::open`] takes. Empty on a machine
/// whose devices are not linked (SLI or similar); a group would allow peer
/// memory and copies between its devices through a single logical device.
pub fn device_groups() -> Result<Vec<Vec<u32>>> {
    let instance = instance()?;
    // SAFETY: the instance is alive for the process.
    let groups = unsafe { instance.raw.enumerate_physical_device_groups_len() }
        .map_err(|code| Error::vk("vkEnumeratePhysicalDeviceGroups", code))?;
    let mut list = vec![vk::PhysicalDeviceGroupProperties::default(); groups];
    // SAFETY: the list has the length the first call reported.
    unsafe { instance.raw.enumerate_physical_device_groups(&mut list) }
        .map_err(|code| Error::vk("vkEnumeratePhysicalDeviceGroups", code))?;
    Ok(list
        .iter()
        .map(|group| {
            group.physical_devices[..group.physical_device_count as usize]
                .iter()
                .filter_map(|handle| {
                    instance.physical.iter().position(|p| p.raw == *handle).map(|i| i as u32)
                })
                .collect::<Vec<u32>>()
        })
        .filter(|ordinals| ordinals.len() > 1)
        .collect())
}

/// Index of `sender` among the devices other than `owner`, in device order.
fn slot_of(sender: usize, owner: usize) -> usize {
    if sender < owner { sender } else { sender - 1 }
}

fn others(count: usize, device: usize) -> impl Iterator<Item = usize> {
    (0..count).filter(move |p| *p != device)
}

/// Pipelines of the mesh on one device.
struct MeshKernels {
    sum: Function,
    push: Function,
}

impl MeshKernels {
    fn load(device: &Device) -> Result<Self> {
        let main = |image: &[u8]| Module::load(device, image)?.function("main");
        Ok(Self { sum: main(MESH_SUM)?, push: main(MESH_PUSH)? })
    }

    /// `out = operands[0] + operands[1] + ...`, added in this order.
    ///
    /// # Safety
    ///
    /// Every pointer addresses `n` f32 values on the stream's device; `out`
    /// is an operand or does not overlap any.
    unsafe fn sum(
        &self,
        stream: &Stream,
        out: DevicePtr,
        operands: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        debug_assert!(!operands.is_empty() && operands.len() <= MAX_DEVICES);
        let mut args = Args::new().ptr(out);
        for k in 0..MAX_DEVICES {
            args = args.ptr(operands.get(k).copied().unwrap_or(DevicePtr::NULL));
        }
        let args = args.u32(operands.len() as u32).u32(n);
        // SAFETY: the argument list matches mesh_sum.comp; the ranges are the
        // caller's guarantee.
        unsafe { self.sum.launch(stream, [n.div_ceil(256), 1, 1], args) }
    }

    /// One half of a push sum.
    ///
    /// # Safety
    ///
    /// `data` and `out` address `n` f32 values on the stream's device. `peer`
    /// addresses `n` words of 8 bytes that the other device polls, and `own`
    /// `n` words that it writes, both as the stream's device sees them.
    #[allow(clippy::too_many_arguments)]
    unsafe fn push(
        &self,
        stream: &Stream,
        data: DevicePtr,
        out: DevicePtr,
        peer: DevicePtr,
        own: DevicePtr,
        n: u32,
        token: u32,
        polls: u32,
    ) -> Result<()> {
        let args = Args::new().ptr(data).ptr(out).ptr(peer).ptr(own).u32(n).u32(token).u32(polls);
        // SAFETY: the argument list matches mesh_push.comp.
        unsafe { self.push.launch(stream, [PUSH_BLOCKS, 1, 1], args) }
    }
}

/// Memory that every device of a group addresses, each by its own address.
struct Shared {
    /// The start of the memory as device `i` sees it.
    views: Vec<DevicePtr>,
    _buffers: Vec<DeviceBuffer>,
    _pages: Option<Arc<HostPages>>,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared").field("devices", &self.views.len()).finish()
    }
}

impl Shared {
    /// `bytes` of host pages, imported by every device.
    fn host(devices: &[Device], bytes: u64) -> Result<Self> {
        let refs: Vec<&Device> = devices.iter().collect();
        let pages = Arc::new(HostPages::new(&refs, bytes as usize)?);
        let mut views = Vec::new();
        let mut buffers = Vec::new();
        for device in devices {
            let buffer = DeviceBuffer::import_host(device, &pages)?;
            views.push(buffer.ptr());
            buffers.push(buffer);
        }
        Ok(Self { views, _buffers: buffers, _pages: Some(pages) })
    }

    /// `bytes` of `devices[owner]`'s video memory, zeroed, exported and
    /// imported by the other devices. Fails unless a write through every
    /// import reaches the owner's memory: some drivers accept the import and
    /// give the importer memory of its own.
    fn exported(devices: &[Device], owner: usize, bytes: u64, kind: ShareKind) -> Result<Self> {
        let own = DeviceBuffer::new_exportable(&devices[owner], bytes, kind)?;
        zero(&own)?;
        let mut views = vec![DevicePtr::NULL; devices.len()];
        views[owner] = own.ptr();
        let mut buffers = Vec::new();
        for (d, device) in devices.iter().enumerate().filter(|(d, _)| *d != owner) {
            let view = DeviceBuffer::import(device, own.export()?)?;
            // The importer writes; the owner reads what it wrote.
            let probe: Vec<u8> = (0..256u32).map(|i| (i * 7 + 3) as u8).collect();
            view.write(0, &probe)?;
            let mut got = vec![0u8; probe.len()];
            own.read(0, &mut got)?;
            if got != probe {
                return Err(Error::Unsupported {
                    device: device.info().name.clone(),
                    reason: format!(
                        "memory of {} imported as {kind:?} is accepted but is not that memory: \
                         what this device writes through the import never reaches the owner",
                        devices[owner].info().name
                    ),
                });
            }
            view.write(0, &vec![0u8; probe.len()])?;
            views[d] = view.ptr();
            buffers.push(view);
        }
        buffers.push(own);
        Ok(Self { views, _buffers: buffers, _pages: None })
    }
}

/// Fills a buffer with zeros, a megabyte at a time.
fn zero(buffer: &DeviceBuffer) -> Result<()> {
    const CHUNK: usize = 1 << 20;
    let zeros = vec![0u8; CHUNK];
    let mut at = 0u64;
    while at < buffer.len() {
        let len = (buffer.len() - at).min(CHUNK as u64) as usize;
        buffer.write(at, &zeros[..len])?;
        at += len as u64;
    }
    Ok(())
}

/// A host buffer that only the devices' copies and the host's `memcpy` touch.
struct Staging {
    buffer: StagingBuffer,
    ptr: *mut u8,
}

// SAFETY: the pointer addresses memory owned by `buffer`; it is only used for
// `memcpy` while no device copy is in flight on the bytes it touches.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

impl Staging {
    fn new(device: &Device, len: usize) -> Result<Self> {
        let mut buffer = StagingBuffer::new(device, len)?;
        let ptr = buffer.as_mut_ptr();
        Ok(Self { buffer, ptr })
    }
}

/// The landing zones of the shared routes.
///
/// Each device owns a zone: `[push words | two slots of one region per sender]`.
/// The push words are 8 bytes per value, two sums' worth (consecutive sums
/// alternate, so the sender can write the next while the owner still reads the
/// last). The copy regions hold a sender's raw f32 partial; consecutive sums
/// alternate between the two slots too.
struct Fabric {
    /// `Peer`: one entry per owner. `Host`: a single entry holding the zones
    /// of all owners one after the other.
    zones: Vec<Shared>,
    zone_bytes: u64,
    /// Offset of the copy regions in a zone.
    copy_offset: u64,
    /// `progress[d]`: set by device `d` to `k` once its partial of copy sum `k`
    /// is in the zones of the others.
    progress: Vec<SharedTimeline>,
}

impl Fabric {
    /// Where owner's zone is, as `viewer` addresses it.
    fn zone(&self, route: Route, owner: usize, viewer: usize) -> DevicePtr {
        match route {
            Route::Host => self.zones[0].views[viewer].offset(owner as u64 * self.zone_bytes),
            _ => self.zones[owner].views[viewer],
        }
    }
}

struct Member {
    device: Device,
    kernels: MeshKernels,
    /// Where this device collects the others' partials on the routes that
    /// copy them in: one region of `max_values` f32 per other device.
    inbox: Option<DeviceBuffer>,
}

/// The buffers of the staged route.
struct Stage {
    /// Per device: its partial, on the way out.
    out: Vec<Staging>,
    /// Per device: the others' partials, one region each.
    inn: Vec<Staging>,
}

/// A group of devices that exchange partial results.
pub struct Mesh {
    members: Vec<Member>,
    route: Route,
    peer_kind: Option<ShareKind>,
    max_values: u32,
    push_values: u32,
    /// Sums of up to this many values go through the push kernels.
    push_limit: AtomicU32,
    fabric: Option<Fabric>,
    stage: Option<Stage>,
    /// The number of the last push sum.
    token: AtomicU64,
    /// The number of the last sum that went through the copy engine.
    sequence: AtomicU64,
    /// Polls of a word before a push kernel gives up.
    polls: AtomicU32,
    /// Routes that were tried before the one this mesh has, and why they were
    /// not taken.
    rejected: Vec<(Route, String)>,
    /// Features of the route that were tried and left out.
    notes: Vec<String>,
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("devices", &self.members.iter().map(|m| m.device.ordinal()).collect::<Vec<_>>())
            .field("route", &self.route)
            .field("max_values", &self.max_values)
            .field("push_values", &self.push_values)
            .finish()
    }
}

/// How a sum is done.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    /// The push kernels when they apply, else the copy engine.
    Auto,
    /// The copy engine, whatever the size.
    Copy,
}

impl Mesh {
    /// A mesh over `devices` for sums of up to `max_values` f32 values, on the
    /// fastest route that works: every route is tried from the best down, and
    /// a real exchange verifies it (see [`Mesh::rejected`] for what failed).
    pub fn new(devices: &[Device], max_values: u32) -> Result<Self> {
        let mut rejected = Vec::new();
        if let Some(group) = device_groups()?
            .iter()
            .find(|group| devices.iter().all(|d| group.contains(&d.ordinal())))
        {
            rejected.push((
                Route::Peer,
                format!(
                    "the loader reports devices {group:?} as one device group, which would allow \
                     peer memory and copies through a single logical device; that route is not built"
                ),
            ));
        }
        for route in [Route::Peer, Route::Host, Route::Staged] {
            match Self::build(devices, max_values, route) {
                Ok(mut mesh) => {
                    mesh.rejected.splice(0..0, rejected);
                    return Ok(mesh);
                }
                // A request that can never work, whatever the route.
                Err(error @ Error::Invalid { .. }) => return Err(error),
                Err(error) => rejected.push((route, error.to_string())),
            }
        }
        // The staged route needs nothing but host memory; if it failed, so does
        // everything else.
        Err(Error::invalid(
            "mesh",
            format!("no route works: {}", rejected.last().map_or("", |(_, why)| why.as_str())),
        ))
    }

    /// A mesh on a given route. Fails when the route does not work on these
    /// devices.
    pub fn with_route(devices: &[Device], max_values: u32, route: Route) -> Result<Self> {
        Self::build(devices, max_values, route)
    }

    fn build(devices: &[Device], max_values: u32, route: Route) -> Result<Self> {
        if devices.is_empty() || devices.len() > MAX_DEVICES || max_values == 0 {
            return Err(Error::invalid(
                "mesh",
                format!(
                    "a mesh needs between 1 and {MAX_DEVICES} devices and a size ({} devices, \
                     {max_values} values)",
                    devices.len()
                ),
            ));
        }
        for (i, a) in devices.iter().enumerate() {
            if devices[..i].iter().any(|b| a.same_device(b)) {
                return Err(Error::invalid("mesh", "the same device is in the mesh twice"));
            }
        }
        let count = devices.len();
        let max_bytes = u64::from(max_values) * 4;
        let push_values =
            if count == 2 && route != Route::Staged { max_values.min(PUSH_MAX_VALUES) } else { 0 };
        let members = devices
            .iter()
            .map(|device| {
                Ok(Member {
                    device: device.clone(),
                    kernels: MeshKernels::load(device)?,
                    inbox: if count > 1 && route != Route::Peer {
                        Some(DeviceBuffer::new(device, max_bytes * (count as u64 - 1))?)
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut mesh = Self {
            members,
            route: if count == 1 { Route::Staged } else { route },
            peer_kind: None,
            max_values,
            push_values,
            push_limit: AtomicU32::new(push_values),
            fabric: None,
            stage: None,
            token: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            polls: AtomicU32::new(PROBE_POLLS),
            rejected: Vec::new(),
            notes: Vec::new(),
        };
        if count == 1 {
            return Ok(mesh);
        }
        match route {
            Route::Staged => {
                let make = |len: u64| -> Result<Vec<Staging>> {
                    devices.iter().map(|d| Staging::new(d, len as usize)).collect()
                };
                mesh.stage = Some(Stage {
                    out: make(max_bytes)?,
                    inn: make(max_bytes * (count as u64 - 1))?,
                });
            }
            Route::Host | Route::Peer => mesh.build_fabric(devices)?,
        }
        mesh.verify()?;
        mesh.polls.store(PUSH_POLLS, Ordering::Relaxed);
        Ok(mesh)
    }

    /// The landing zones and the counters of the shared routes.
    fn build_fabric(&mut self, devices: &[Device]) -> Result<()> {
        let count = devices.len();
        let max_bytes = u64::from(self.max_values) * 4;
        let push_bytes = 2 * u64::from(self.push_values) * 8;
        let copy_offset = push_bytes.next_multiple_of(256);
        let zone_bytes = (copy_offset + 2 * max_bytes * (count as u64 - 1)).next_multiple_of(4096);
        // Counters first: they are the cheaper test of whether devices can
        // share anything at all, and a value set by one device must be seen by
        // the other before the rest is worth building.
        check_timelines_carry(devices)?;
        let zones = match self.route {
            Route::Host => vec![Shared::host(devices, zone_bytes * count as u64)?],
            Route::Peer => {
                let mut errors = Vec::new();
                let mut found = None;
                for kind in [ShareKind::OpaqueFd, ShareKind::DmaBuf] {
                    let zones: Result<Vec<Shared>> = (0..count)
                        .map(|owner| Shared::exported(devices, owner, zone_bytes, kind))
                        .collect();
                    match zones {
                        Ok(zones) => {
                            found = Some((kind, zones));
                            break;
                        }
                        Err(error) => errors.push(format!("as {kind:?}: {error}")),
                    }
                }
                let Some((kind, zones)) = found else {
                    return Err(Error::Unsupported {
                        device: devices
                            .iter()
                            .map(|d| d.info().name.as_str())
                            .collect::<Vec<_>>()
                            .join(" and "),
                        reason: errors.join("; "),
                    });
                };
                self.peer_kind = Some(kind);
                zones
            }
            Route::Staged => unreachable!("the staged route has no landing zones"),
        };
        let progress = (0..count).map(|_| SharedTimeline::new(devices)).collect::<Result<_>>()?;
        self.fabric = Some(Fabric { zones, zone_bytes, copy_offset, progress });
        Ok(())
    }

    /// Runs real sums over the route and compares them with the exact result:
    /// every device must get the sum, bit for bit, on the push path (when the
    /// route has one) and on the copy path, in a chain of rounds in which each
    /// builds on the last. Nothing here blocks for longer than a few seconds.
    ///
    /// The copy path must work for the route to be taken. The push path is
    /// kept only if it does: it needs the receiving kernel to see the other
    /// device's writes while it polls, which depends on how the driver caches
    /// the memory it polls (see the module docs); a mesh whose push test fails
    /// sends every sum through the copy engine and says so in [`Mesh::notes`].
    fn verify(&mut self) -> Result<()> {
        if self.push_values > 0 {
            let smoke = [(Path::Auto, 64.min(self.push_values), 1)];
            let full =
                [(Path::Auto, 1000.min(self.push_values), 4), (Path::Auto, self.push_values, 4)];
            match self.run_cases(&smoke).and_then(|()| self.run_cases(&full)) {
                Ok(()) => {}
                Err(error) => {
                    self.notes.push(format!("push kernels are not used: {error}"));
                    self.push_values = 0;
                    self.push_limit.store(0, Ordering::Relaxed);
                }
            }
        }
        let copy = [
            (Path::Copy, 1000.min(self.max_values), 4),
            (Path::Copy, self.max_values.min(50_000), 4),
        ];
        self.run_cases(&copy)
    }

    /// Runs the `(path, values, rounds)` cases of [`Mesh::verify`].
    fn run_cases(&self, cases: &[(Path, u32, u32)]) -> Result<()> {
        let count = self.members.len();
        let devices: Vec<Device> = self.members.iter().map(|m| m.device.clone()).collect();
        let streams = devices.iter().map(Stream::new).collect::<Result<Vec<_>>>()?;
        let stream_refs: Vec<&Stream> = streams.iter().collect();
        for &(path, n, rounds) in cases {
            let partials: Vec<DeviceBuffer> = devices
                .iter()
                .enumerate()
                .map(|(d, device)| {
                    let buffer = DeviceBuffer::new(device, u64::from(n) * 4)?;
                    let values: Vec<u8> = (0..n)
                        .flat_map(|v| (((d + 1) as u32 * (v % 13 + 1)) as f32).to_le_bytes())
                        .collect();
                    buffer.write(0, &values)?;
                    Ok(buffer)
                })
                .collect::<Result<_>>()?;
            let pointers: Vec<DevicePtr> = partials.iter().map(DeviceBuffer::ptr).collect();
            for _ in 0..rounds {
                // SAFETY: whole buffers of `n` values on the streams' devices,
                // alive until the streams are idle (below).
                unsafe { self.sum(&stream_refs, &pointers, &pointers, n, path)? };
            }
            for stream in &streams {
                stream.flush()?;
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            for stream in &streams {
                while !stream.is_idle()? {
                    if Instant::now() > deadline {
                        return Err(self.failed(
                            "the devices did not finish a test exchange within ten seconds".into(),
                        ));
                    }
                    std::thread::yield_now();
                }
            }
            let total: u32 = (1..=count as u32).sum();
            for (d, partial) in partials.iter().enumerate() {
                let mut got = vec![0u8; n as usize * 4];
                partial.read(0, &mut got)?;
                for (v, bytes) in got.as_chunks::<4>().0.iter().enumerate() {
                    let got = f32::from_le_bytes(*bytes);
                    let want = (total * (v as u32 % 13 + 1)) as f32
                        * (count as f32).powi(rounds as i32 - 1);
                    if got != want {
                        return Err(self.failed(format!(
                            "a test exchange ({}, {n} values) gave {got} for value {v} on {} \
                             instead of {want}",
                            if path == Path::Copy { "copy engine" } else { "push kernels" },
                            devices[d].info().name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// A route that does not work: not a mistake of the caller.
    fn failed(&self, reason: String) -> Error {
        let names: Vec<&str> = self.members.iter().map(|m| m.device.info().name.as_str()).collect();
        Error::Unsupported { device: names.join(" and "), reason }
    }

    /// The route the mesh takes.
    pub fn route(&self) -> Route {
        self.route
    }

    /// The routes that were tried and did not work, with the reason: what the
    /// driver refused, or what the test exchange got wrong.
    pub fn rejected(&self) -> &[(Route, String)] {
        &self.rejected
    }

    /// What the route could have had and the test exchange did not allow (the
    /// push kernels, for one), with the reason.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// For [`Route::Peer`], how the memory is exported.
    pub fn peer_kind(&self) -> Option<ShareKind> {
        self.peer_kind
    }

    /// The largest sum that goes through the push kernels, in f32 values; 0
    /// when the mesh has none (other than two devices on a shared route).
    pub fn push_values(&self) -> u32 {
        self.push_values
    }

    /// Sets the size up to which sums go through the push kernels (at most
    /// [`Mesh::push_values`]); 0 sends every sum through the copy engine. For
    /// measuring where the two cross.
    pub fn set_push_limit(&self, values: u32) {
        self.push_limit.store(values.min(self.push_values), Ordering::Relaxed);
    }

    /// The devices of the mesh, in the order the mesh was made with.
    pub fn devices(&self) -> impl Iterator<Item = &Device> {
        self.members.iter().map(|m| &m.device)
    }

    /// Enqueues the sum of the devices' partial results: afterwards each
    /// `partials[i]`, `n` f32 values on device `i`, holds the sum of all of
    /// them. `streams[i]` is device `i`'s stream; work enqueued on it later
    /// sees the sum.
    ///
    /// Nothing here waits for a device, except on [`Route::Staged`]. Every
    /// device gets the bitwise same sum on every route, for any number of
    /// devices: the partials are added in device order everywhere.
    ///
    /// The streams are linked (see [`Stream::link`]), and when one of them
    /// had to submit a batch while the sum was enqueued, the others are
    /// flushed too, so that no device waits for work the host has not sent.
    ///
    /// # Safety
    ///
    /// `streams[i]` must be a stream of device `i` of this mesh and
    /// `partials[i]` must address `n` f32 values on that device, alive until
    /// the enqueued work has run. Every sum of this mesh must be enqueued
    /// from one thread at a time and on the same streams, so that consecutive
    /// sums are ordered.
    pub unsafe fn sum_f32(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantees, with each partial as its own output.
        unsafe { self.sum(streams, partials, partials, n, Path::Auto) }
    }

    /// Like [`Mesh::sum_f32`], but the sum is written to `outs[i]` on each
    /// device and the partials are left as they are. Use it where the next
    /// kernel can read its input from another buffer than the partial.
    ///
    /// # Safety
    ///
    /// As for [`Mesh::sum_f32`]; in addition `outs[i]` must address `n` f32
    /// values on device `i` that do not overlap any partial.
    pub unsafe fn sum_f32_to(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        outs: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantees.
        unsafe { self.sum(streams, partials, outs, n, Path::Auto) }
    }

    /// The sum of `partials` into `outs`, where `outs[i]` is `partials[i]` or
    /// does not overlap any partial.
    unsafe fn sum(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        outs: &[DevicePtr],
        n: u32,
        path: Path,
    ) -> Result<()> {
        let count = self.members.len();
        if streams.len() != count || partials.len() != count || outs.len() != count {
            return Err(Error::invalid(
                "mesh sum",
                format!(
                    "{count} devices, {} streams, {} partials, {} outputs",
                    streams.len(),
                    partials.len(),
                    outs.len()
                ),
            ));
        }
        for (member, stream) in self.members.iter().zip(streams) {
            if !member.device.same_device(stream.device()) {
                return Err(Error::invalid(
                    "mesh sum",
                    "a stream is not on the device of the same position in the mesh",
                ));
            }
        }
        if n > self.max_values {
            return Err(Error::invalid(
                "mesh sum",
                format!("{n} values; the mesh was sized for {}", self.max_values),
            ));
        }
        if n == 0 {
            return Ok(());
        }

        if count == 1 {
            if outs[0] != partials[0] {
                // SAFETY: both ranges hold `n` values on the device and do
                // not overlap (the caller's guarantee).
                unsafe { streams[0].copy(outs[0], partials[0], n as usize * 4)? };
            }
            return Ok(());
        }

        // Every address is checked before anything is enqueued: a sum that
        // fails halfway would leave a device waiting for the others for good.
        let bytes = u64::from(n) * 4;
        for (member, (partial, out)) in self.members.iter().zip(partials.iter().zip(outs)) {
            member.device.resolve(*partial, bytes)?;
            member.device.resolve(*out, bytes)?;
        }
        for (i, a) in streams.iter().enumerate() {
            for b in &streams[i + 1..] {
                a.link(b);
            }
        }
        let before: Vec<u64> = streams.iter().map(|s| s.submissions()).collect();
        if self.route == Route::Staged {
            // SAFETY: the caller's guarantees.
            unsafe { self.sum_staged(streams, partials, outs, n)? };
        } else if path == Path::Auto && n <= self.push_limit.load(Ordering::Relaxed) {
            // SAFETY: as above.
            unsafe { self.sum_push(streams, partials, outs, n)? };
        } else {
            // SAFETY: as above.
            unsafe { self.sum_copy(streams, partials, outs, n)? };
        }
        // A device whose batch has gone out waits for the others' kernels:
        // make sure those have gone out too.
        if streams.iter().zip(&before).any(|(s, b)| s.submissions() != *b) {
            for stream in streams {
                stream.flush()?;
            }
        }
        Ok(())
    }

    /// Two devices, one kernel each, data and tags in the receivers' words.
    unsafe fn sum_push(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        outs: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        let fabric = self.fabric.as_ref().expect("a shared route has a fabric");
        // Tokens run from 1 to 0xFFFF_FFFE and then start again, so that the
        // parity of consecutive sums alternates through the wrap as well.
        let counter = self.token.fetch_add(1, Ordering::Relaxed);
        let token = (counter % 0xFFFF_FFFE) as u32 + 1;
        let slot = u64::from(token % 2) * u64::from(self.push_values) * 8;
        let polls = self.polls.load(Ordering::Relaxed);
        for d in 0..2 {
            let peer = fabric.zone(self.route, 1 - d, d).offset(slot);
            let own = fabric.zone(self.route, d, d).offset(slot);
            // SAFETY: each zone holds two slots of `push_values >= n` words;
            // `peer` is the other device's zone as this device addresses it,
            // and the other half of the exchange with the same token is
            // enqueued in this same loop. The data ranges are the caller's
            // guarantee.
            unsafe {
                self.members[d].kernels.push(
                    streams[d],
                    partials[d],
                    outs[d],
                    peer,
                    own,
                    n,
                    token,
                    polls,
                )?;
            }
        }
        Ok(())
    }

    /// Any number of devices, through the copy engine and the counters.
    ///
    /// Each device copies its partial into the zones of the others and sets its
    /// counter to the number of the sum; then it waits for the others' counters,
    /// takes their partials out of its own zone and adds. Consecutive sums use
    /// alternate slots of the zones. A sender may write a slot again without any
    /// further wait: the receiver took the slot's last partial out before it
    /// set its counter for the sum in between, and the sender waited for that
    /// counter.
    unsafe fn sum_copy(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        outs: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        let count = self.members.len();
        let fabric = self.fabric.as_ref().expect("a shared route has a fabric");
        let max_bytes = u64::from(self.max_values) * 4;
        let bytes = n as usize * 4;
        let k = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let slot = (k % 2) * (count as u64 - 1) * max_bytes;
        for d in 0..count {
            let stream = streams[d];
            // This device's partial goes to every other device's zone.
            for p in others(count, d) {
                let region = fabric
                    .zone(self.route, p, d)
                    .offset(fabric.copy_offset + slot + slot_of(d, p) as u64 * max_bytes);
                // SAFETY: the region holds `max_bytes >= bytes` bytes of the
                // zone, which is part of a live buffer of this device; the
                // source is the caller's guarantee.
                unsafe { stream.copy(region, partials[d], bytes)? };
            }
            fabric.progress[d].signal(stream, k)?;
            for p in others(count, d) {
                fabric.progress[p].wait(stream, k)?;
            }
            // The operands in device order; the others' come from the zone
            // directly (video memory) or from a copy of it.
            let own_zone = fabric.zone(self.route, d, d).offset(fabric.copy_offset + slot);
            let mut operands = Vec::with_capacity(count);
            for q in 0..count {
                if q == d {
                    operands.push(partials[d]);
                    continue;
                }
                let j = slot_of(q, d) as u64 * max_bytes;
                if let Some(inbox) = &self.members[d].inbox {
                    // SAFETY: the zone region and the inbox region both hold
                    // `max_bytes >= bytes` bytes; the sender's copy has
                    // finished (the wait above).
                    unsafe { stream.copy(inbox.ptr().offset(j), own_zone.offset(j), bytes)? };
                    operands.push(inbox.ptr().offset(j));
                } else {
                    operands.push(own_zone.offset(j));
                }
            }
            // SAFETY: all ranges hold `n` values on this device; the output is
            // an operand or does not overlap any (the caller's guarantee).
            unsafe { self.members[d].kernels.sum(stream, outs[d], &operands, n)? };
        }
        Ok(())
    }

    /// The host copies between the devices' host buffers and waits for them.
    unsafe fn sum_staged(
        &self,
        streams: &[&Stream],
        partials: &[DevicePtr],
        outs: &[DevicePtr],
        n: u32,
    ) -> Result<()> {
        let count = self.members.len();
        let stage = self.stage.as_ref().expect("the staged route has buffers");
        let max_bytes = u64::from(self.max_values) * 4;
        let bytes = n as usize * 4;
        for d in 0..count {
            // SAFETY: the staging buffer holds `max_bytes >= bytes` bytes and
            // is not touched until the stream is synchronised below; the
            // source is the caller's guarantee.
            unsafe { streams[d].copy_to_staging(&stage.out[d].buffer, 0, partials[d], bytes)? };
            streams[d].flush()?;
        }
        for stream in streams {
            stream.synchronize()?;
        }
        for d in 0..count {
            for q in others(count, d) {
                let j = slot_of(q, d) * max_bytes as usize;
                // SAFETY: both buffers hold the bytes copied; the devices are
                // done with the out buffers (synchronised above) and with the
                // in buffer (its previous contents were copied by work that
                // precedes the copy just waited for).
                unsafe {
                    std::ptr::copy_nonoverlapping(stage.out[q].ptr, stage.inn[d].ptr.add(j), bytes)
                };
            }
        }
        for d in 0..count {
            let mut operands = Vec::with_capacity(count);
            let inbox = self.members[d].inbox.as_ref().expect("the staged route has inboxes");
            for q in 0..count {
                if q == d {
                    operands.push(partials[d]);
                } else {
                    let j = slot_of(q, d) as u64 * max_bytes;
                    // SAFETY: the in buffer holds `max_bytes` per other device and the
                    // inbox the same; the bytes were written by the host above.
                    unsafe {
                        streams[d].copy_from_staging(
                            inbox.ptr().offset(j),
                            &stage.inn[d].buffer,
                            j as usize,
                            bytes,
                        )?;
                    }
                    operands.push(inbox.ptr().offset(j));
                }
            }
            // SAFETY: all ranges hold `n` values on this device.
            unsafe { self.members[d].kernels.sum(streams[d], outs[d], &operands, n)? };
        }
        Ok(())
    }
}

/// Fails unless a value that one device sets on a shared counter is seen by
/// the other devices, which the copy path is built on.
fn check_timelines_carry(devices: &[Device]) -> Result<()> {
    let timeline = SharedTimeline::new(devices)?;
    let stream = Stream::new(&devices[0])?;
    timeline.signal(&stream, 1)?;
    let deadline = Instant::now() + Duration::from_secs(2);
    for device in &devices[1..] {
        while timeline.value(device)? < 1 {
            if Instant::now() > deadline {
                return Err(Error::Unsupported {
                    device: device.info().name.clone(),
                    reason: "a value set on a shared semaphore by another device is not seen"
                        .into(),
                });
            }
            std::thread::yield_now();
        }
    }
    Ok(())
}

/// A one-way path for activations from one device to another: what a layer
/// split needs at each boundary between devices.
///
/// The sender does not wait for the receiver. It writes each parcel into the
/// next slot of a small ring on the receiver's side and goes on with its
/// next step, so both devices stay busy as long as there is independent work
/// (the chunks of a prompt, the rows of several sequences): this is what makes
/// a layer split a pipeline and not a relay. The sender is held back, on the
/// device, only when it is a whole ring ahead of the receiver.
///
/// The ring is the only memory a lane costs: `slots * max_values` f32 on the
/// receiver, plus the same in host memory on the host routes.
///
/// On [`Route::Staged`] the receiving side waits for the sender inside
/// [`Lane::receive`] (the host copies the parcel), so the sender's work up to
/// the parcel must be able to run without the host enqueuing more.
pub struct Lane {
    from: Device,
    to: Device,
    route: Route,
    max_values: u32,
    slots: u32,
    /// `Host` and `Staged`: where parcels land on the receiver.
    inbox: Option<DeviceBuffer>,
    /// `Host` and `Peer`: the zone the sender writes into and the counters.
    link: Option<LaneLink>,
    /// `Staged`: the host buffers and the events.
    stage: Option<LaneStage>,
    sent_count: AtomicU64,
    released_count: AtomicU64,
    /// The stream of the last send, to link with the receiver's.
    sender: Mutex<Option<Weak<StreamInner>>>,
}

struct LaneLink {
    /// One ring of parcels: the sender's address is index 0, the receiver's 1.
    zone: Shared,
    /// Set by the sender to the number of parcels sent.
    sent: SharedTimeline,
    /// Set by the receiver to the number of parcels released.
    consumed: SharedTimeline,
}

struct LaneStage {
    out: Staging,
    inn: Staging,
    /// Per slot, on the sender: the parcel has left.
    sent: Vec<Event>,
    /// Per slot, on the receiver: the parcel has been read.
    consumed: Vec<Event>,
}

impl std::fmt::Debug for Lane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lane")
            .field("from", &self.from.ordinal())
            .field("to", &self.to.ordinal())
            .field("route", &self.route)
            .field("slots", &self.slots)
            .finish()
    }
}

/// A parcel on its way through a [`Lane`]. It must be received and then
/// released, in the order the parcels were sent.
#[derive(Debug)]
#[must_use = "a parcel that is not released blocks its slot of the lane"]
pub struct Parcel {
    sequence: u64,
    slot: u32,
    n: u32,
}

impl Parcel {
    /// The number of f32 values in the parcel.
    pub fn len(&self) -> u32 {
        self.n
    }

    /// Whether the parcel holds no values.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl Lane {
    /// A lane from device `from` to device `to` of `mesh` (indices in the
    /// order the mesh was built with), with a ring of `slots` parcels of up
    /// to `max_values` f32 values. It uses the mesh's route.
    pub fn new(mesh: &Mesh, from: usize, to: usize, slots: u32, max_values: u32) -> Result<Self> {
        let count = mesh.members.len();
        if from >= count || to >= count || from == to || slots == 0 || max_values == 0 {
            return Err(Error::invalid(
                "lane",
                format!(
                    "from {from} to {to} of {count} devices, {slots} slots of {max_values} values"
                ),
            ));
        }
        let (a, b) = (mesh.members[from].device.clone(), mesh.members[to].device.clone());
        let slot_bytes = u64::from(max_values) * 4;
        let ring = slot_bytes * u64::from(slots);
        let pair = [a.clone(), b.clone()];
        let mut lane = Self {
            from: a,
            to: b,
            route: mesh.route,
            max_values,
            slots,
            inbox: None,
            link: None,
            stage: None,
            sent_count: AtomicU64::new(0),
            released_count: AtomicU64::new(0),
            sender: Mutex::new(None),
        };
        match mesh.route {
            Route::Host | Route::Peer => {
                let zone = if mesh.route == Route::Host {
                    Shared::host(&pair, ring)?
                } else {
                    let kind = mesh.peer_kind.expect("a peer mesh knows how it exports");
                    Shared::exported(&pair, 1, ring, kind)?
                };
                if mesh.route == Route::Host {
                    lane.inbox = Some(DeviceBuffer::new(&lane.to, ring)?);
                }
                lane.link = Some(LaneLink {
                    zone,
                    sent: SharedTimeline::new(&pair)?,
                    consumed: SharedTimeline::new(&pair)?,
                });
            }
            Route::Staged => {
                lane.inbox = Some(DeviceBuffer::new(&lane.to, ring)?);
                lane.stage = Some(LaneStage {
                    out: Staging::new(&lane.from, ring as usize)?,
                    inn: Staging::new(&lane.to, ring as usize)?,
                    sent: (0..slots).map(|_| Event::new(&lane.from)).collect::<Result<_>>()?,
                    consumed: (0..slots).map(|_| Event::new(&lane.to)).collect::<Result<_>>()?,
                });
            }
        }
        Ok(lane)
    }

    /// The number of parcels the ring holds.
    pub fn slots(&self) -> u32 {
        self.slots
    }

    fn slot_bytes(&self) -> u64 {
        u64::from(self.max_values) * 4
    }

    /// Where a slot is on the receiver.
    fn received_at(&self, slot: u32) -> DevicePtr {
        let offset = u64::from(slot) * self.slot_bytes();
        match (&self.inbox, &self.link) {
            (Some(inbox), _) => inbox.ptr().offset(offset),
            (None, Some(link)) => link.zone.views[1].offset(offset),
            (None, None) => unreachable!("a lane has an inbox or a zone"),
        }
    }

    /// Enqueues, on the sender's stream, sending `n` f32 values at `src`.
    /// The sender's stream goes on without waiting for the receiver, unless
    /// the slot's previous parcel has not been read yet.
    ///
    /// Fails, with nothing enqueued, when `slots` parcels are sent and not
    /// yet released: the caller enqueues the receiving side of earlier
    /// parcels first.
    ///
    /// # Safety
    ///
    /// `sender` must be a stream of the lane's sending device and `src` must
    /// address `n` f32 values on it, alive until the enqueued work has run.
    /// All calls on one lane must come from one thread at a time.
    pub unsafe fn send_f32(&self, sender: &Stream, src: DevicePtr, n: u32) -> Result<Parcel> {
        if n > self.max_values {
            return Err(Error::invalid(
                "lane",
                format!("{n} values; the lane was sized for {}", self.max_values),
            ));
        }
        if !sender.device().same_device(&self.from) {
            return Err(Error::invalid("lane", "the stream is not on the sending device"));
        }
        let sequence = self.sent_count.load(Ordering::Relaxed);
        let in_flight = sequence - self.released_count.load(Ordering::Relaxed);
        if in_flight >= u64::from(self.slots) {
            return Err(Error::invalid(
                "lane",
                format!("all {} slots hold parcels that are not released yet", self.slots),
            ));
        }
        let slot = (sequence % u64::from(self.slots)) as u32;
        let bytes = n as usize * 4;
        *self.sender.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(sender.weak());
        if let Some(link) = &self.link {
            // The slot's previous parcel must have been released. Its release
            // was enqueued before this call (checked above).
            if sequence >= u64::from(self.slots) {
                link.consumed.wait(sender, sequence - u64::from(self.slots) + 1)?;
            }
            let at = link.zone.views[0].offset(u64::from(slot) * self.slot_bytes());
            // SAFETY: the slot is a region of `max_values >= n` values of the
            // zone, part of a live buffer of the sender; `src` is the caller's
            // guarantee.
            unsafe { sender.copy(at, src, bytes)? };
            link.sent.signal(sender, sequence + 1)?;
        } else {
            let stage = self.stage.as_ref().expect("a staged lane has buffers");
            let at = slot as usize * self.slot_bytes() as usize;
            // SAFETY: the slot of the host buffer is not read by the host
            // until the sent event has happened; `src` is the caller's guarantee.
            unsafe { sender.copy_to_staging(&stage.out.buffer, at, src, bytes)? };
            stage.sent[slot as usize].record(sender)?;
        }
        self.sent_count.store(sequence + 1, Ordering::Relaxed);
        Ok(Parcel { sequence, slot, n })
    }

    /// Enqueues, on the receiver's stream, waiting for `parcel`, and returns
    /// where its values are on the receiving device. Work enqueued on that
    /// stream afterwards can read them there until the parcel is released.
    ///
    /// # Safety
    ///
    /// `receiver` must be a stream of the lane's receiving device, and
    /// `parcel` must come from this lane.
    pub unsafe fn receive(&self, receiver: &Stream, parcel: &Parcel) -> Result<DevicePtr> {
        if !receiver.device().same_device(&self.to) {
            return Err(Error::invalid("lane", "the stream is not on the receiving device"));
        }
        let slot = parcel.slot;
        let bytes = parcel.n as usize * 4;
        if let Some(sender) = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(Weak::upgrade)
        {
            receiver.link_inner(&sender);
        }
        let offset = u64::from(slot) * self.slot_bytes();
        if let Some(link) = &self.link {
            link.sent.wait(receiver, parcel.sequence + 1)?;
            if let Some(inbox) = &self.inbox {
                // SAFETY: the zone slot and the inbox slot both hold
                // `max_values >= parcel.n` values; the sender's copy has
                // finished (the wait above).
                unsafe {
                    receiver.copy(
                        inbox.ptr().offset(offset),
                        link.zone.views[1].offset(offset),
                        bytes,
                    )?
                };
            }
        } else {
            let stage = self.stage.as_ref().expect("a staged lane has buffers");
            let inbox = self.inbox.as_ref().expect("a staged lane has an inbox");
            // The parcel has left the sender, and the receiver has read what
            // this slot held before; then the host moves the bytes.
            stage.sent[slot as usize].synchronize()?;
            stage.consumed[slot as usize].synchronize()?;
            // SAFETY: both buffers hold the slot; the sender's copy has
            // finished and the receiver's copy of the previous parcel has too.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stage.out.ptr.add(offset as usize),
                    stage.inn.ptr.add(offset as usize),
                    bytes,
                );
                receiver.copy_from_staging(
                    inbox.ptr().offset(offset),
                    &stage.inn.buffer,
                    offset as usize,
                    bytes,
                )?;
            }
        }
        Ok(self.received_at(slot))
    }

    /// Enqueues, on the receiver's stream, the end of the receiver's use of
    /// `parcel`: work enqueued before this call has read it, and the slot may
    /// be written again once that work has run.
    ///
    /// Parcels are released in the order they were sent.
    pub fn release(&self, receiver: &Stream, parcel: Parcel) -> Result<()> {
        let expected = self.released_count.load(Ordering::Relaxed);
        if parcel.sequence != expected {
            return Err(Error::invalid(
                "lane",
                format!(
                    "parcel {} released while parcel {expected} is the oldest in flight",
                    parcel.sequence
                ),
            ));
        }
        if let Some(link) = &self.link {
            link.consumed.signal(receiver, expected + 1)?;
        } else {
            let stage = self.stage.as_ref().expect("a staged lane has buffers");
            stage.consumed[parcel.slot as usize].record(receiver)?;
        }
        self.released_count.store(expected + 1, Ordering::Relaxed);
        Ok(())
    }
}
