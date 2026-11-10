//! Device memory and host-visible staging memory.
//!
//! The types here own raw allocations. The engine does not allocate per
//! request: it takes a few large buffers at load time (sized through a
//! `Ledger`) and carves them up itself, so these calls are off the hot path.
//! Each buffer gets its own `VkDeviceMemory`; with the few, large buffers the
//! engine uses that stays far below the driver's allocation count limit, and it
//! keeps every buffer a plain range of one address space.

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use ash::vk;

use crate::device::Device;
use crate::error::{Check, Error, Result};
use crate::instance::Physical;
use crate::transfer;

/// An address in device memory (`VK_KHR_buffer_device_address`).
///
/// This is a plain number: holding one proves nothing about the memory behind
/// it, which is why the calls that take raw addresses are `unsafe` and the
/// bounds-checked calls live on [`DeviceBuffer`]. Kernels receive these
/// addresses in their push constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub(crate) u64);

impl DevicePtr {
    pub const NULL: DevicePtr = DevicePtr(0);

    /// The address `bytes` further on.
    pub fn offset(self, bytes: u64) -> DevicePtr {
        DevicePtr(self.0 + bytes)
    }

    pub fn as_raw(self) -> u64 {
        self.0
    }
}

/// The index of the first memory type allowed by `type_bits` (the bit mask
/// Vulkan gives for a buffer) that satisfies the earliest rule of `rules`.
/// A rule is the flags a type must have and the flags it must not have. Vulkan
/// orders types by what the driver prefers, so among types that satisfy the
/// same rule the first one wins.
pub(crate) fn pick_memory_type(
    memory: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    rules: &[(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags)],
) -> Option<u32> {
    let types = &memory.memory_types[..memory.memory_type_count as usize];
    rules.iter().find_map(|(required, forbidden)| {
        types.iter().enumerate().find_map(|(index, ty)| {
            let allowed = type_bits & (1 << index) != 0;
            let flags = ty.property_flags;
            (allowed && flags.contains(*required) && !flags.intersects(*forbidden))
                .then_some(index as u32)
        })
    })
}

/// Refuses an allocation larger than the heap its memory type lives in, as an
/// out-of-memory error. Vulkan makes asking for one a usage error instead of
/// a failure that can be handled, and a request like that is a plain
/// "does not fit".
fn check_fits_heap(
    memory: &vk::PhysicalDeviceMemoryProperties,
    memory_type: u32,
    size: u64,
    call: &'static str,
) -> Result<()> {
    let heap = memory.memory_types[memory_type as usize].heap_index as usize;
    if size > memory.memory_heaps[heap].size {
        return Err(Error::vk(call, vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
    }
    Ok(())
}

const DEVICE_LOCAL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
const HOST_VISIBLE: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE;
const NONE: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::empty();

/// Video memory the CPU cannot see if there is any; otherwise (a GPU that
/// shares system memory) the device-local type the CPU can see too.
const DEVICE_RULES: &[(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags)] =
    &[(DEVICE_LOCAL, HOST_VISIBLE), (DEVICE_LOCAL, NONE)];

/// System memory the CPU can read and write without cache maintenance, cached
/// when possible (the CPU reads results back from it, and reading
/// write-combined memory is very slow), and never video memory: a window of the
/// card's memory is not what a staging buffer is for.
const HOST_RULES: &[(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags)] = {
    const COHERENT: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    );
    const CACHED: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        COHERENT.as_raw() | vk::MemoryPropertyFlags::HOST_CACHED.as_raw(),
    );
    &[(CACHED, DEVICE_LOCAL), (COHERENT, DEVICE_LOCAL), (CACHED, NONE), (COHERENT, NONE)]
};

/// An allocation of device memory, freed when dropped.
///
/// Dropping a buffer first waits for everything enqueued on its device (the
/// way `cuMemFree` does), so commands that use it cannot outlive it.
#[derive(Debug)]
pub struct DeviceBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: DevicePtr,
    len: u64,
    /// Bytes the driver reserved, for the device's allocation count.
    reserved: u64,
}

impl DeviceBuffer {
    /// Allocates `len` bytes of video memory on `device`. The contents are
    /// undefined.
    pub fn new(device: &Device, len: u64) -> Result<Self> {
        if len == 0 {
            return Err(Error::invalid("device allocation", "a buffer of 0 bytes was requested"));
        }
        let raw = device.raw();
        // Storage and the address are for kernels; both transfer bits for the
        // copy commands of streams and of `write` and `read`.
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let info = vk::BufferCreateInfo::default().size(len).usage(usage);
        // SAFETY: the device is alive and `info` is a valid description.
        let buffer = unsafe { raw.create_buffer(&info, None) }.check("vkCreateBuffer")?;
        // SAFETY: the buffer was created just above.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let destroy_buffer = || {
            // SAFETY: the buffer was created above and is not in use yet.
            unsafe { raw.destroy_buffer(buffer, None) };
        };
        let Some(memory_type) = pick_memory_type(
            &device.physical().memory,
            requirements.memory_type_bits,
            DEVICE_RULES,
        ) else {
            destroy_buffer();
            return Err(Error::invalid(
                "device allocation",
                "the device has no device-local memory type for buffers",
            ));
        };
        if let Err(error) = check_fits_heap(
            &device.physical().memory,
            memory_type,
            requirements.size,
            "vkAllocateMemory",
        ) {
            destroy_buffer();
            return Err(error);
        }
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut flags);
        // SAFETY: valid allocation info; the memory type came from the buffer's
        // own requirements.
        let memory = match unsafe { raw.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(code) => {
                destroy_buffer();
                return Err(Error::vk("vkAllocateMemory", code));
            }
        };
        // SAFETY: the memory is at least as large as the buffer requires and
        // is bound once, at offset 0.
        if let Err(code) = unsafe { raw.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: neither object is in use.
            unsafe { raw.free_memory(memory, None) };
            destroy_buffer();
            return Err(Error::vk("vkBindBufferMemory", code));
        }
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: the buffer was created with SHADER_DEVICE_ADDRESS and the
        // bufferDeviceAddress feature is enabled.
        let address = unsafe { raw.get_buffer_device_address(&address_info) };
        let ptr = DevicePtr(address);
        device.register_buffer(ptr, buffer, len);
        device.count_allocation(requirements.size);
        Ok(Self { device: device.clone(), buffer, memory, ptr, len, reserved: requirements.size })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The address of `len` bytes at `offset`, checked against the buffer.
    pub fn slice(&self, offset: u64, len: u64) -> Result<DevicePtr> {
        match offset.checked_add(len) {
            Some(end) if end <= self.len => Ok(self.ptr.offset(offset)),
            _ => Err(Error::invalid(
                "device buffer range",
                format!("{len} bytes at {offset} do not fit in {} bytes", self.len),
            )),
        }
    }

    /// Copies `data` to `offset` and returns when it is on the device.
    ///
    /// The data goes through a staging buffer of the device and a copy
    /// command that this call submits and waits for, so a kernel launched
    /// after it returns, on any stream, sees all of `data`. (Returning when
    /// the data had only been handed to the driver is the bug this call
    /// exists to avoid.) The copy is ordered after everything already
    /// submitted to the device, but not after commands that are still only
    /// recorded in a stream: flush that stream first.
    ///
    /// This is the convenient path for loading and for tests. The hot path
    /// uses a [`StagingBuffer`] and
    /// [`Stream::copy_from_staging`](crate::Stream::copy_from_staging).
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        self.slice(offset, data.len() as u64)?;
        if data.is_empty() {
            return Ok(());
        }
        transfer::write(&self.device, self.buffer, offset, data)
    }

    /// Copies from `offset` into `out` and waits for the copy. Ordered like
    /// [`write`](Self::write).
    pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        self.slice(offset, out.len() as u64)?;
        if out.is_empty() {
            return Ok(());
        }
        transfer::read(&self.device, self.buffer, offset, out)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // No new command can name the buffer once it is out of the table; the
        // synchronisation then covers the ones that already do.
        self.device.unregister_buffer(self.ptr);
        let _ = self.device.synchronize();
        let raw = self.device.raw();
        // SAFETY: nothing enqueued on the device uses the buffer any more, and
        // both objects are destroyed once.
        unsafe {
            raw.destroy_buffer(self.buffer, None);
            raw.free_memory(self.memory, None);
        }
        self.device.count_release(self.reserved);
    }
}

/// Host memory the device can copy from and to, mapped for the lifetime of the
/// buffer.
pub(crate) struct HostMemory {
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) ptr: NonNull<u8>,
    pub(crate) len: usize,
}

// SAFETY: the mapping is plain memory owned by this value.
unsafe impl Send for HostMemory {}

impl HostMemory {
    /// Allocates `len` zeroed bytes of host-visible, coherent memory and maps
    /// it.
    pub(crate) fn create(raw: &ash::Device, physical: &Physical, len: usize) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(len as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        // SAFETY: the device is alive and `info` is a valid description.
        let buffer = unsafe { raw.create_buffer(&info, None) }.check("vkCreateBuffer")?;
        let fail = |error: Error| {
            // SAFETY: the buffer was created above and is not in use.
            unsafe { raw.destroy_buffer(buffer, None) };
            Err(error)
        };
        // SAFETY: the buffer was created just above.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let Some(memory_type) =
            pick_memory_type(&physical.memory, requirements.memory_type_bits, HOST_RULES)
        else {
            return fail(Error::invalid(
                "host allocation",
                "the device has no host-visible coherent memory type",
            ));
        };
        if let Err(error) =
            check_fits_heap(&physical.memory, memory_type, requirements.size, "vkAllocateMemory")
        {
            return fail(error);
        }
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: valid allocation info; the type came from the requirements.
        let memory = match unsafe { raw.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(code) => return fail(Error::vk("vkAllocateMemory", code)),
        };
        let fail = |error: Error| {
            // SAFETY: neither object is in use.
            unsafe {
                raw.free_memory(memory, None);
                raw.destroy_buffer(buffer, None);
            }
            Err(error)
        };
        // SAFETY: the memory is large enough and bound once.
        if let Err(code) = unsafe { raw.bind_buffer_memory(buffer, memory, 0) } {
            return fail(Error::vk("vkBindBufferMemory", code));
        }
        // SAFETY: the memory is host-visible and not mapped yet.
        let mapped =
            unsafe { raw.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) };
        let mapped = match mapped {
            Ok(mapped) => mapped,
            Err(code) => return fail(Error::vk("vkMapMemory", code)),
        };
        let Some(ptr) = NonNull::new(mapped.cast::<u8>()) else {
            return fail(Error::invalid("host allocation", "the driver mapped memory at null"));
        };
        // SAFETY: the mapping covers at least `len` writable bytes. Zeroing
        // makes the contents initialised so they can be viewed as a slice.
        unsafe { ptr.as_ptr().write_bytes(0, len) };
        Ok(Self { buffer, memory, ptr, len })
    }

    pub(crate) fn destroy(self, raw: &ash::Device) {
        // SAFETY: the caller guarantees nothing uses the buffer any more; the
        // objects were created by `create` and are destroyed once.
        unsafe {
            raw.unmap_memory(self.memory);
            raw.destroy_buffer(self.buffer, None);
            raw.free_memory(self.memory, None);
        }
    }
}

/// Host memory that a device copies from and to, persistently mapped and freed
/// when dropped: the counterpart of the CUDA layer's `PinnedBuffer`.
///
/// Transfers between a staging buffer and a device buffer are DMA copies that
/// run asynchronously on a [`Stream`](crate::Stream). The memory is coherent
/// (no flushing is ever needed) and cached when the platform has such memory,
/// so reading what the device wrote is as fast as reading any other memory.
/// Unlike pinned memory it is tied to the device it was made for.
pub struct StagingBuffer {
    device: Device,
    memory: Option<HostMemory>,
}

// SAFETY: the buffer owns its memory exclusively, like a `Box<[u8]>`.
unsafe impl Send for StagingBuffer {}
// SAFETY: shared access only hands out `&[u8]`.
unsafe impl Sync for StagingBuffer {}

impl std::fmt::Debug for StagingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingBuffer").field("len", &self.len()).finish()
    }
}

impl StagingBuffer {
    /// Allocates `len` zeroed bytes of mapped host memory. It counts against
    /// host RAM, not against video memory.
    pub fn new(device: &Device, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::invalid("staging allocation", "a buffer of 0 bytes was requested"));
        }
        let memory = HostMemory::create(device.raw(), device.physical(), len)?;
        Ok(Self { device: device.clone(), memory: Some(memory) })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn memory(&self) -> &HostMemory {
        self.memory.as_ref().expect("the memory is only taken in drop")
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.memory().buffer
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.memory().ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.memory().ptr.as_ptr()
    }
}

impl Deref for StagingBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        let memory = self.memory();
        // SAFETY: `ptr` points to `len` initialised bytes owned by `self`.
        unsafe { std::slice::from_raw_parts(memory.ptr.as_ptr(), memory.len) }
    }
}

impl DerefMut for StagingBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        let memory = self.memory();
        // SAFETY: as in `deref`, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(memory.ptr.as_ptr(), memory.len) }
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        // Copies that name this buffer may still be recorded or running.
        let _ = self.device.synchronize();
        if let Some(memory) = self.memory.take() {
            memory.destroy(self.device.raw());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn properties(types: &[(u32, vk::MemoryPropertyFlags)]) -> vk::PhysicalDeviceMemoryProperties {
        let mut memory = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: types.len() as u32,
            ..Default::default()
        };
        for (slot, (heap, flags)) in memory.memory_types.iter_mut().zip(types) {
            slot.heap_index = *heap;
            slot.property_flags = *flags;
        }
        memory
    }

    /// The layout this machine's NVIDIA driver reports with resizable BAR.
    fn nvidia() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        let coherent = F::HOST_VISIBLE | F::HOST_COHERENT;
        properties(&[
            (1, F::empty()),
            (0, F::DEVICE_LOCAL),
            (1, coherent),
            (1, coherent | F::HOST_CACHED),
            (0, F::DEVICE_LOCAL | coherent),
        ])
    }

    #[test]
    fn buffers_take_video_memory_the_cpu_cannot_see() {
        assert_eq!(pick_memory_type(&nvidia(), 0b11111, DEVICE_RULES), Some(1));
        // A buffer that may only use the mapped window of video memory.
        assert_eq!(pick_memory_type(&nvidia(), 0b10000, DEVICE_RULES), Some(4));
    }

    #[test]
    fn staging_takes_cached_system_memory_and_never_video_memory() {
        assert_eq!(pick_memory_type(&nvidia(), 0b11111, HOST_RULES), Some(3));
        // Without the cached type: coherent system memory, not the window.
        assert_eq!(pick_memory_type(&nvidia(), 0b10100, HOST_RULES), Some(2));
        // Only the window is allowed: better than nothing.
        assert_eq!(pick_memory_type(&nvidia(), 0b10000, HOST_RULES), Some(4));
        assert_eq!(pick_memory_type(&nvidia(), 0b00011, HOST_RULES), None);
    }

    #[test]
    fn a_gpu_that_shares_system_memory_has_one_type_for_both() {
        use vk::MemoryPropertyFlags as F;
        let all = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED;
        let memory = properties(&[(0, all)]);
        assert_eq!(pick_memory_type(&memory, 1, DEVICE_RULES), Some(0));
        assert_eq!(pick_memory_type(&memory, 1, HOST_RULES), Some(0));
    }

    #[test]
    fn device_pointers_offset_and_compare() {
        let base = DevicePtr(0x1000);
        assert_eq!(base.offset(0x20).as_raw(), 0x1020);
        assert_ne!(base, DevicePtr::NULL);
        assert_eq!(DevicePtr::NULL.as_raw(), 0);
    }
}
