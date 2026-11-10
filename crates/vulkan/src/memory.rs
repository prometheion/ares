//! Device memory and host-visible staging memory.
//!
//! The types here own raw allocations. The engine does not allocate per
//! request: it takes a few large buffers at load time (sized through a
//! `Ledger`) and carves them up itself, so these calls are off the hot path.
//! Each buffer gets its own `VkDeviceMemory`; with the few, large buffers the
//! engine uses that stays far below the driver's allocation count limit, and it
//! keeps every buffer a plain range of one address space.

use std::alloc::Layout;
use std::ops::{Deref, DerefMut};
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Check, Error, Result};
use crate::instance::{BUFFER_USAGE, Physical};
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

/// How memory of one device is handed to another one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShareKind {
    /// A file descriptor in the driver's own format (`VK_KHR_external_memory_fd`,
    /// `OPAQUE_FD`). Only devices of one driver can import it.
    OpaqueFd,
    /// A `dma-buf` (`VK_EXT_external_memory_dma_buf`), which other drivers
    /// can import as well.
    DmaBuf,
}

impl ShareKind {
    fn flag(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            ShareKind::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            ShareKind::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
        }
    }

    fn name(self) -> &'static str {
        match self {
            ShareKind::OpaqueFd => "OPAQUE_FD",
            ShareKind::DmaBuf => "DMA_BUF",
        }
    }

    /// Whether `device` lists this kind as importable and exportable for
    /// buffers.
    pub fn supported_by(self, device: &Device) -> bool {
        let sharing = &device.physical().sharing;
        device.0.memory_fd.is_some()
            && match self {
                ShareKind::OpaqueFd => sharing.opaque_fd,
                ShareKind::DmaBuf => sharing.dma_buf,
            }
    }
}

/// A buffer of one device, exported, on its way to be imported by another.
/// Dropping it before it is imported closes the descriptor.
#[derive(Debug)]
pub struct SharedMemory {
    fd: OwnedFd,
    kind: ShareKind,
    len: u64,
    /// The size of the allocation behind the buffer.
    allocation: u64,
}

/// Page-aligned host memory that several devices can import
/// (`VK_EXT_external_memory_host`) and the host reads and writes through
/// plain pointers.
///
/// The memory is zeroed when made. A device that imports it keeps a
/// [`DeviceBuffer`] over the same physical pages, so a write by one device or
/// by the host is visible to the others once it has left the writer's caches;
/// it is only ever accessed through raw pointers, because devices write it
/// behind the host's back.
pub struct HostPages {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

// SAFETY: the pages are plain memory owned by this value; the type hands out
// only raw pointers, and accesses to them are the callers' responsibility.
unsafe impl Send for HostPages {}
// SAFETY: as above.
unsafe impl Sync for HostPages {}

impl std::fmt::Debug for HostPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostPages").field("len", &self.len).finish()
    }
}

impl HostPages {
    /// The pages of one import: 4 KiB, the alignment every device accepts that
    /// this crate has seen, raised to the largest of `devices`.
    fn alignment(devices: &[&Device]) -> usize {
        devices.iter().map(|d| d.physical().sharing.host_alignment as usize).fold(4096, usize::max)
    }

    /// `len` zeroed bytes (rounded up to the alignment) that all of `devices`
    /// can import.
    pub fn new(devices: &[&Device], len: usize) -> Result<Self> {
        let alignment = Self::alignment(devices);
        let len = len.max(1).next_multiple_of(alignment);
        let layout = Layout::from_size_align(len, alignment)
            .map_err(|e| Error::invalid("host pages", e.to_string()))?;
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::vk("host pages", vk::Result::ERROR_OUT_OF_HOST_MEMORY))?;
        Ok(Self { ptr, len, layout })
    }

    /// The size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the pages hold no bytes (never: they are at least a page).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The start of the pages. Devices write here behind the host's back: read
    /// and write through volatile accesses, never through references.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for HostPages {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout; the devices that
        // imported the pages hold an `Arc` of this value, so none is left.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Where the memory of a [`DeviceBuffer`] comes from.
enum Source {
    /// Fresh video memory.
    Local,
    /// Fresh video memory that can be exported.
    Export(ShareKind),
    /// Video memory of another device, imported.
    ImportFd(SharedMemory),
    /// Host pages, imported.
    ImportHost(Arc<HostPages>),
}

/// An allocation of device memory, freed when dropped.
///
/// Dropping a buffer first waits for everything enqueued on its device (the
/// way `cuMemFree` does), so commands that use it cannot outlive it.
///
/// A buffer can also be a view of memory that lives elsewhere: video memory
/// of another device ([`DeviceBuffer::import`]) or host pages
/// ([`DeviceBuffer::import_host`]). It has an address on this device like any
/// other, kernels and copies use it the same way, and what they read and write
/// is the other memory.
#[derive(Debug)]
pub struct DeviceBuffer {
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: DevicePtr,
    len: u64,
    /// Bytes the driver reserved, for the device's allocation count.
    reserved: u64,
    /// The kind this buffer was made exportable as.
    export: Option<ShareKind>,
    /// The size of the allocation, which an importer needs to know.
    allocation: u64,
    /// The host pages this buffer is a view of.
    _pages: Option<Arc<HostPages>>,
}

impl DeviceBuffer {
    /// Allocates `len` bytes of video memory on `device`. The contents are
    /// undefined.
    pub fn new(device: &Device, len: u64) -> Result<Self> {
        Self::allocate(device, len, Source::Local)
    }

    /// Allocates `len` bytes of video memory that another device can import
    /// (see [`DeviceBuffer::export`]).
    pub fn new_exportable(device: &Device, len: u64, kind: ShareKind) -> Result<Self> {
        if !kind.supported_by(device) {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: format!("it cannot export and import buffers as {}", kind.name()),
            });
        }
        Self::allocate(device, len, Source::Export(kind))
    }

    /// A buffer on `device` over the video memory that `shared` came from,
    /// which belongs to another device. Fails when the driver refuses the
    /// import; that the call succeeds does not say that the other device's
    /// memory can really be read and written from here (the mesh checks that
    /// with data).
    pub fn import(device: &Device, shared: SharedMemory) -> Result<Self> {
        if !shared.kind.supported_by(device) {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: format!("it cannot import buffers as {}", shared.kind.name()),
            });
        }
        Self::allocate(device, shared.len, Source::ImportFd(shared))
    }

    /// A buffer on `device` over the host `pages`: the memory of the buffer is
    /// the pages, also for every other device that imports them. Its address is
    /// valid in kernels of `device` (the device reads and writes host memory
    /// over the bus through it).
    pub fn import_host(device: &Device, pages: &Arc<HostPages>) -> Result<Self> {
        if !device.physical().sharing.host_pointer || device.0.host_memory.is_none() {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: "it cannot import host memory (VK_EXT_external_memory_host)".into(),
            });
        }
        Self::allocate(device, pages.len() as u64, Source::ImportHost(pages.clone()))
    }

    /// Exports the memory of a buffer made with
    /// [`DeviceBuffer::new_exportable`]. Every call gives a new descriptor, to
    /// be imported once by [`DeviceBuffer::import`].
    pub fn export(&self) -> Result<SharedMemory> {
        let Some(kind) = self.export else {
            return Err(Error::invalid("export", "the buffer was not made exportable"));
        };
        let loader = self.device.0.memory_fd.as_ref().ok_or_else(|| Error::Unsupported {
            device: self.device.info().name.clone(),
            reason: "VK_KHR_external_memory_fd is not enabled".into(),
        })?;
        let info = vk::MemoryGetFdInfoKHR::default().memory(self.memory).handle_type(kind.flag());
        // SAFETY: the memory was allocated exportable as this handle type.
        let fd = unsafe { loader.get_memory_fd(&info) }.check("vkGetMemoryFdKHR")?;
        // SAFETY: the driver returned a new descriptor that nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        Ok(SharedMemory { fd, kind, len: self.len, allocation: self.allocation })
    }

    fn allocate(device: &Device, len: u64, source: Source) -> Result<Self> {
        if len == 0 {
            return Err(Error::invalid("device allocation", "a buffer of 0 bytes was requested"));
        }
        let raw = device.raw();
        // Storage and the address are for kernels; both transfer bits for the
        // copy commands of streams and of `write` and `read`.
        let handle_type = match &source {
            Source::Local => vk::ExternalMemoryHandleTypeFlags::empty(),
            Source::Export(kind) => kind.flag(),
            Source::ImportFd(shared) => shared.kind.flag(),
            Source::ImportHost(_) => vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
        };
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
        let mut info = vk::BufferCreateInfo::default().size(len).usage(BUFFER_USAGE);
        if !handle_type.is_empty() {
            info = info.push_next(&mut external);
        }
        // SAFETY: the device is alive and `info` is a valid description.
        let buffer = unsafe { raw.create_buffer(&info, None) }.check("vkCreateBuffer")?;
        // SAFETY: the buffer was created just above.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let destroy_buffer = || {
            // SAFETY: the buffer was created above and is not in use yet.
            unsafe { raw.destroy_buffer(buffer, None) };
        };
        // The memory types the buffer may use, and the rules to choose by.
        let mut type_bits = requirements.memory_type_bits;
        let mut rules = DEVICE_RULES;
        let mut fd_properties = None;
        match &source {
            Source::ImportFd(shared) if shared.kind == ShareKind::DmaBuf => {
                let loader = device.0.memory_fd.as_ref().expect("checked by import");
                let mut props = vk::MemoryFdPropertiesKHR::default();
                // SAFETY: the descriptor is a dma-buf, which this call is for.
                let result = unsafe {
                    (loader.fp().get_memory_fd_properties_khr)(
                        raw.handle(),
                        vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                        std::os::fd::AsRawFd::as_raw_fd(&shared.fd),
                        &mut props,
                    )
                };
                if result != vk::Result::SUCCESS {
                    destroy_buffer();
                    return Err(Error::vk("vkGetMemoryFdPropertiesKHR", result));
                }
                type_bits &= props.memory_type_bits;
                fd_properties = Some(props.memory_type_bits);
            }
            Source::ImportHost(pages) => {
                let loader = device.0.host_memory.as_ref().expect("checked by import_host");
                let mut props = vk::MemoryHostPointerPropertiesEXT::default();
                // SAFETY: the pointer is page aligned and the pages live as long as `pages`.
                let result = unsafe {
                    (loader.fp().get_memory_host_pointer_properties_ext)(
                        raw.handle(),
                        vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                        pages.as_ptr().cast(),
                        &mut props,
                    )
                };
                if result != vk::Result::SUCCESS {
                    destroy_buffer();
                    return Err(Error::vk("vkGetMemoryHostPointerPropertiesEXT", result));
                }
                type_bits &= props.memory_type_bits;
                rules = HOST_RULES;
            }
            _ => {}
        }
        let Some(memory_type) = pick_memory_type(&device.physical().memory, type_bits, rules)
        else {
            destroy_buffer();
            return Err(Error::invalid(
                "device allocation",
                format!(
                    "the device has no suitable memory type for this buffer (types allowed by the \
                     buffer {:#x}, by the import {fd_properties:x?})",
                    requirements.memory_type_bits
                ),
            ));
        };
        let imported = matches!(source, Source::ImportFd(_) | Source::ImportHost(_));
        if !imported
            && let Err(error) = check_fits_heap(
                &device.physical().memory,
                memory_type,
                requirements.size,
                "vkAllocateMemory",
            )
        {
            destroy_buffer();
            return Err(error);
        }
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut allocation_size = requirements.size;
        // The descriptor and the pointer are taken by the call when it succeeds.
        let mut import_fd = None;
        let mut import_host = None;
        match &source {
            Source::ImportFd(shared) => {
                allocation_size = shared.allocation.max(requirements.size);
                import_fd = Some(
                    vk::ImportMemoryFdInfoKHR::default()
                        .handle_type(shared.kind.flag())
                        .fd(std::os::fd::AsRawFd::as_raw_fd(&shared.fd)),
                );
            }
            Source::ImportHost(pages) => {
                allocation_size = (pages.len() as u64).max(requirements.size);
                import_host = Some(
                    vk::ImportMemoryHostPointerInfoEXT::default()
                        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                        .host_pointer(pages.as_ptr().cast()),
                );
            }
            _ => {}
        }
        let mut allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type)
            .push_next(&mut flags);
        if matches!(source, Source::Export(_)) {
            allocate = allocate.push_next(&mut export_info);
        }
        if matches!(source, Source::Export(_) | Source::ImportFd(_)) {
            allocate = allocate.push_next(&mut dedicated);
        }
        if let Some(import) = import_fd.as_mut() {
            allocate = allocate.push_next(import);
        }
        if let Some(import) = import_host.as_mut() {
            allocate = allocate.push_next(import);
        }
        // SAFETY: valid allocation info; the memory type came from the buffer's
        // own requirements, restricted by what the import allows.
        let memory = match unsafe { raw.allocate_memory(&allocate, None) } {
            Ok(memory) => memory,
            Err(code) => {
                destroy_buffer();
                return Err(Error::vk("vkAllocateMemory", code));
            }
        };
        let (export, pages) = match source {
            Source::Export(kind) => (Some(kind), None),
            Source::ImportHost(pages) => (None, Some(pages)),
            Source::ImportFd(shared) => {
                // The driver owns the descriptor now.
                let _ = shared.fd.into_raw_fd();
                (None, None)
            }
            Source::Local => (None, None),
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
        let reserved = if imported { 0 } else { requirements.size };
        device.register_buffer(ptr, buffer, len);
        device.count_allocation(reserved);
        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            ptr,
            len,
            reserved,
            export,
            allocation: allocation_size,
            _pages: pages,
        })
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
