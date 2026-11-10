//! Devices: describing the hardware, and opening a logical device on it.

use std::collections::BTreeMap;
use std::ffi::c_char;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};

use ash::vk;

use crate::error::{Check, Error, Result};
use crate::instance::{Instance, PUSH_CONSTANT_BYTES, Physical, instance};
use crate::memory::DevicePtr;
use crate::stream::StreamInner;
use crate::transfer::Transfer;

/// Where a device sits on the PCI bus. Two devices behind the same host
/// bridge can exchange data directly when the driver allows peer access.
///
/// Unlike the CUDA layer's, this carries the PCI function too, because Vulkan
/// reports it; the CUDA layer always shows `.0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}:{:02x}:{:02x}.{:x}", self.domain, self.bus, self.device, self.function)
    }
}

/// The kind of a device. Processor-based implementations (llvmpipe, SwiftShader)
/// are never listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    Discrete,
    Integrated,
    Virtual,
    Other,
}

/// What the loader reports about a device without opening it. Reading this
/// creates no logical device and allocates nothing on the GPU.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// Index of the device in [`devices`], and what [`Device::open`] takes.
    /// The loader's order has nothing to do with the CUDA driver's: match
    /// devices of the two layers by `uuid`.
    pub ordinal: u32,
    pub name: String,
    /// PCI vendor id: 0x10de NVIDIA, 0x1002 AMD, 0x8086 Intel.
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: DeviceType,
    /// The highest Vulkan version the device implements, `(major, minor,
    /// patch)`. The engine needs 1.2.
    pub api_version: (u32, u32, u32),
    /// The driver's version number in its vendor's own encoding.
    pub driver_version: u32,
    /// Size in bytes of the device's video memory (the largest device-local
    /// heap). On a GPU that shares system memory it is that memory.
    pub total_memory: u64,
    /// The PCI address, when the device implements `VK_EXT_pci_bus_info`.
    pub pci: Option<PciAddress>,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`. For NVIDIA it is the
    /// UUID that CUDA reports for the same GPU, which is how the engine knows
    /// that a Vulkan device and a CUDA device are one.
    pub uuid: [u8; 16],
    /// Invocations per subgroup (warp). The kernels' reductions use subgroup
    /// operations and need at least 8.
    pub subgroup_size: u32,
}

/// How a thread waits for the device when it synchronises.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Wait {
    /// Poll on a core. Lowest latency; the choice for the threads that drive
    /// a device during inference.
    Spin,
    /// Poll, yielding the core between polls.
    Yield,
    /// Sleep in the driver until the device signals. For threads that wait
    /// rarely.
    #[default]
    Block,
}

/// The number of devices the engine can open: every Vulkan device except
/// processor-based ones.
pub fn device_count() -> Result<u32> {
    Ok(instance()?.physical.len() as u32)
}

/// Describes every device without creating a logical device on any of them.
/// A device the engine cannot use (for lack of a feature) is listed all the
/// same; [`Device::open`] says why it cannot be opened.
pub fn devices() -> Result<Vec<DeviceInfo>> {
    Ok(instance()?.physical.iter().map(|p| p.info.clone()).collect())
}

/// Memory of an opened device right now.
#[derive(Clone, Copy, Debug)]
pub struct MemInfo {
    pub free: u64,
    pub total: u64,
}

/// Locks a mutex whose holder panicked. Everything behind the mutexes of this
/// crate is left consistent at every point a panic could occur.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) struct Inner {
    instance: &'static Instance,
    physical: &'static Physical,
    pub(crate) raw: ash::Device,
    queue: Mutex<vk::Queue>,
    wait: Wait,
    /// The layout of every pipeline: push constants only, no descriptor sets.
    pub(crate) layout: vk::PipelineLayout,
    /// Device buffers by start address, so that a raw address can be turned
    /// back into the buffer handle that Vulkan's copy commands need.
    buffers: RwLock<BTreeMap<u64, (vk::Buffer, u64)>>,
    /// The streams opened on this device, to flush them all when the device
    /// is asked to synchronise.
    streams: Mutex<Vec<Weak<StreamInner>>>,
    /// Staging memory and command buffers for the synchronous
    /// `DeviceBuffer::write` and `read`, made on first use.
    pub(crate) transfer: Mutex<Option<Transfer>>,
    /// Bytes of device memory allocated through this crate, for `mem_info`
    /// on a device without `VK_EXT_memory_budget`.
    allocated: AtomicU64,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: this is the last owner of the device. Every buffer, stream,
        // event and pipeline holds a `Device`, so none is left, and the wait
        // makes sure nothing still runs before the objects below are
        // destroyed.
        unsafe {
            let _ = self.raw.device_wait_idle();
            if let Some(transfer) =
                self.transfer.get_mut().unwrap_or_else(PoisonError::into_inner).take()
            {
                transfer.destroy(&self.raw);
            }
            self.raw.destroy_pipeline_layout(self.layout, None);
            self.raw.destroy_device(None);
        }
    }
}

/// An opened device: a logical device with one compute queue, shared by clones
/// of this handle.
///
/// Opening a device creates no allocation of note (a few MiB of driver state,
/// where CUDA's context takes a few hundred); buffers, streams and pipelines
/// keep it alive until the last of them is dropped.
#[derive(Clone)]
pub struct Device(pub(crate) Arc<Inner>);

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({}: {})", self.0.physical.info.ordinal, self.0.physical.info.name)
    }
}

impl Device {
    /// Opens device `ordinal` (an index into [`devices`]); threads block in
    /// the driver while they wait for it.
    pub fn open(ordinal: u32) -> Result<Self> {
        Self::open_with(ordinal, Wait::default())
    }

    /// Opens device `ordinal`. `wait` chooses how threads wait for it.
    pub fn open_with(ordinal: u32, wait: Wait) -> Result<Self> {
        let instance = instance()?;
        let physical = instance
            .physical
            .get(ordinal as usize)
            .ok_or(Error::NoSuchDevice { ordinal, count: instance.physical.len() as u32 })?;
        if let Some(reason) = &physical.unusable {
            return Err(Error::Unsupported {
                device: physical.info.name.clone(),
                reason: reason.clone(),
            });
        }

        // One queue: the engine's streams are command buffers that are
        // submitted to it in order (see the `stream` module).
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(physical.queue_family)
            .queue_priorities(&priorities)];
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().shader_int64(true))
            .push_next(&mut v12);
        let extensions: Vec<*const c_char> = if physical.memory_budget {
            vec![ash::ext::memory_budget::NAME.as_ptr()]
        } else {
            vec![]
        };
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features);
        // SAFETY: the physical device belongs to the live instance, and every
        // feature and extension requested was checked when it was described.
        let raw = unsafe { instance.raw.create_device(physical.raw, &info, None) }
            .check("vkCreateDevice")?;

        let range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(PUSH_CONSTANT_BYTES)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range);
        // SAFETY: the device was created just above.
        let layout = match unsafe { raw.create_pipeline_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(code) => {
                // SAFETY: nothing was created on the device yet.
                unsafe { raw.destroy_device(None) };
                return Err(Error::vk("vkCreatePipelineLayout", code));
            }
        };
        // SAFETY: the family was requested with one queue just above.
        let queue = unsafe { raw.get_device_queue(physical.queue_family, 0) };

        Ok(Self(Arc::new(Inner {
            instance,
            physical,
            raw,
            queue: Mutex::new(queue),
            wait,
            layout,
            buffers: RwLock::new(BTreeMap::new()),
            streams: Mutex::new(Vec::new()),
            transfer: Mutex::new(None),
            allocated: AtomicU64::new(0),
        })))
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.0.physical.info
    }

    pub fn ordinal(&self) -> u32 {
        self.0.physical.info.ordinal
    }

    pub(crate) fn raw(&self) -> &ash::Device {
        &self.0.raw
    }

    pub(crate) fn physical(&self) -> &'static Physical {
        self.0.physical
    }

    pub(crate) fn same_device(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Free and total video memory as the driver sees it now. With
    /// `VK_EXT_memory_budget` (every current desktop driver) `free` is the
    /// heap's budget minus its usage, and other processes count against it;
    /// the budget is a little below the heap's size, so `free` is what a
    /// process may allocate before the driver starts evicting, which is
    /// slightly less than what `nvidia-smi` shows as free. Without it, `free`
    /// is the total minus what this crate has allocated.
    pub fn mem_info(&self) -> Result<MemInfo> {
        let physical = self.0.physical;
        let total = physical.info.total_memory;
        let free = if physical.memory_budget {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            // SAFETY: the extension is supported (and was enabled when the
            // device was opened); the physical device belongs to the instance.
            unsafe {
                self.0.instance.raw.get_physical_device_memory_properties2(physical.raw, &mut props)
            };
            let heap = physical.device_heap;
            budget.heap_budget[heap].saturating_sub(budget.heap_usage[heap])
        } else {
            total.saturating_sub(self.0.allocated.load(Ordering::Relaxed))
        };
        Ok(MemInfo { free: free.min(total), total })
    }

    pub(crate) fn count_allocation(&self, bytes: u64) {
        self.0.allocated.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn count_release(&self, bytes: u64) {
        self.0.allocated.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Submits one batch to the device's queue. `waits` are timeline
    /// semaphore values the batch waits for before it starts; `signal` is a
    /// timeline value the batch sets when it has finished; `fence` is
    /// signalled too when it is not null.
    pub(crate) fn submit(
        &self,
        command_buffers: &[vk::CommandBuffer],
        waits: &[(vk::Semaphore, u64)],
        signal: Option<(vk::Semaphore, u64)>,
        fence: vk::Fence,
    ) -> Result<()> {
        let wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|w| w.0).collect();
        let wait_values: Vec<u64> = waits.iter().map(|w| w.1).collect();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let (signal_semaphores, signal_values): (&[vk::Semaphore], &[u64]) = match &signal {
            Some((semaphore, value)) => {
                (std::slice::from_ref(semaphore), std::slice::from_ref(value))
            }
            None => (&[], &[]),
        };
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(signal_values);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .push_next(&mut timeline);
        let queue = lock(&self.0.queue);
        // SAFETY: the queue is externally synchronised by the lock; every
        // handle in the batch is alive, and the caller keeps the command
        // buffers and semaphores alive until the batch has finished.
        unsafe { self.0.raw.queue_submit(*queue, &[submit], fence) }.check("vkQueueSubmit")
    }

    /// Waits until the timeline `semaphore` has reached `value`, in the way
    /// the device was opened to wait.
    pub(crate) fn wait_timeline(&self, semaphore: vk::Semaphore, value: u64) -> Result<()> {
        if value == 0 {
            return Ok(());
        }
        match self.0.wait {
            Wait::Block => {
                let semaphores = [semaphore];
                let values = [value];
                let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
                // SAFETY: the semaphore is alive; the caller holds it.
                unsafe { self.0.raw.wait_semaphores(&info, u64::MAX) }.check("vkWaitSemaphores")
            }
            Wait::Spin | Wait::Yield => {
                while self.timeline_value(semaphore)? < value {
                    if self.0.wait == Wait::Yield {
                        std::thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                }
                Ok(())
            }
        }
    }

    /// The current value of a timeline semaphore.
    pub(crate) fn timeline_value(&self, semaphore: vk::Semaphore) -> Result<u64> {
        // SAFETY: the semaphore is alive; the caller holds it.
        unsafe { self.0.raw.get_semaphore_counter_value(semaphore) }
            .check("vkGetSemaphoreCounterValue")
    }

    /// Waits until everything enqueued on the device has finished: every
    /// stream is flushed first, so commands that were recorded and not yet
    /// submitted are included.
    ///
    /// Work another thread records after this call has taken its snapshot of
    /// the streams is not waited for.
    pub fn synchronize(&self) -> Result<()> {
        let streams: Vec<Arc<StreamInner>> = {
            let mut registry = lock(&self.0.streams);
            registry.retain(|stream| stream.strong_count() > 0);
            registry.iter().filter_map(Weak::upgrade).collect()
        };
        let mut first_error = None;
        for stream in &streams {
            if let Err(error) = stream.drain() {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub(crate) fn register_stream(&self, stream: &Arc<StreamInner>) {
        let mut registry = lock(&self.0.streams);
        registry.retain(|stream| stream.strong_count() > 0);
        registry.push(Arc::downgrade(stream));
    }

    pub(crate) fn register_buffer(&self, address: DevicePtr, buffer: vk::Buffer, len: u64) {
        let mut buffers = self.0.buffers.write().unwrap_or_else(PoisonError::into_inner);
        buffers.insert(address.as_raw(), (buffer, len));
    }

    pub(crate) fn unregister_buffer(&self, address: DevicePtr) {
        let mut buffers = self.0.buffers.write().unwrap_or_else(PoisonError::into_inner);
        buffers.remove(&address.as_raw());
    }

    /// The buffer that contains the `len` bytes at `ptr`, and the offset of
    /// `ptr` in it. Fails when the range is not inside one live buffer of this
    /// device.
    pub(crate) fn resolve(&self, ptr: DevicePtr, len: u64) -> Result<(vk::Buffer, u64)> {
        let buffers = self.0.buffers.read().unwrap_or_else(PoisonError::into_inner);
        let found = buffers.range(..=ptr.as_raw()).next_back();
        if let Some((start, (buffer, buffer_len))) = found {
            let offset = ptr.as_raw() - start;
            if offset.checked_add(len).is_some_and(|end| end <= *buffer_len) {
                return Ok((*buffer, offset));
            }
        }
        Err(Error::invalid(
            "device address",
            format!(
                "{len} bytes at {:#x} are not inside one live buffer of {:?}",
                ptr.as_raw(),
                self
            ),
        ))
    }
}
