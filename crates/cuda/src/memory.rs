//! Device memory and page-locked host memory.
//!
//! The types here own raw allocations. The engine does not allocate per
//! request: it takes a few large buffers at load time (sized through a
//! `Ledger`) and carves them up itself, so these calls are off the hot path.

use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::sys::*;

/// An address in device memory.
///
/// This is a plain number: holding one proves nothing about the memory behind
/// it, which is why the calls that take raw addresses are `unsafe` and the
/// bounds-checked calls live on [`DeviceBuffer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub(crate) CUdeviceptr);

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

/// An allocation of device memory, freed when dropped.
#[derive(Debug)]
pub struct DeviceBuffer {
    device: Device,
    ptr: DevicePtr,
    len: u64,
}

impl DeviceBuffer {
    /// Allocates `len` bytes on `device`. The contents are undefined.
    pub fn new(device: &Device, len: u64) -> Result<Self> {
        if len == 0 {
            return Err(Error::Invalid {
                what: "device allocation",
                detail: "a buffer of 0 bytes was requested".into(),
            });
        }
        let driver = device.bind()?;
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: the pointer is valid for the write of one address.
        driver.check("cuMemAlloc", unsafe { (driver.api.mem_alloc)(&mut ptr, len as usize) })?;
        Ok(Self { device: device.clone(), ptr: DevicePtr(ptr), len })
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
            _ => Err(Error::Invalid {
                what: "device buffer range",
                detail: format!("{len} bytes at {offset} do not fit in {} bytes", self.len),
            }),
        }
    }

    /// Copies `data` to `offset` and waits until it is on the device.
    ///
    /// This is the convenient path for loading and for tests. It waits for
    /// everything pending on the device, so the hot path uses pinned memory
    /// and [`Stream::copy_from_host`](crate::Stream::copy_from_host) instead.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let dst = self.slice(offset, data.len() as u64)?;
        if data.is_empty() {
            return Ok(());
        }
        let driver = self.device.bind()?;
        // SAFETY: the destination range was checked against this live
        // allocation and the source is a valid slice, which the driver has
        // copied out of when the call returns.
        driver.check("cuMemcpyHtoD", unsafe {
            (driver.api.memcpy_htod)(dst.0, data.as_ptr().cast(), data.len())
        })?;
        // From ordinary (pageable) memory the driver returns once the data is
        // in its staging buffer; the transfer to the device may still be
        // under way. Work on a stream does not wait for it, so a kernel
        // launched right after could read the old contents.
        // SAFETY: a context is current.
        driver.check("cuCtxSynchronize", unsafe { (driver.api.ctx_synchronize)() })
    }

    /// Copies from `offset` into `out` and waits for the copy.
    pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let src = self.slice(offset, out.len() as u64)?;
        if out.is_empty() {
            return Ok(());
        }
        let driver = self.device.bind()?;
        // SAFETY: the source range was checked against this live allocation
        // and the destination is a valid slice; the call is synchronous.
        driver.check("cuMemcpyDtoH", unsafe {
            (driver.api.memcpy_dtoh)(out.as_mut_ptr().cast(), src.0, out.len())
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the address came from cuMemAlloc and is freed once.
            unsafe { (driver.api.mem_free)(self.ptr.0) };
        }
    }
}

/// Page-locked host memory, freed when dropped.
///
/// Transfers between pinned memory and a device go by DMA and can run
/// asynchronously on a stream; transfers from ordinary memory are copied
/// through a driver buffer first and are several times slower. The buffer is
/// usable with every device ("portable").
pub struct PinnedBuffer {
    device: Device,
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer owns its memory exclusively, like a `Box<[u8]>`.
unsafe impl Send for PinnedBuffer {}
// SAFETY: shared access only hands out `&[u8]`.
unsafe impl Sync for PinnedBuffer {}

impl std::fmt::Debug for PinnedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer").field("len", &self.len).finish()
    }
}

impl PinnedBuffer {
    /// Allocates `len` zeroed, page-locked bytes. Pinned memory cannot be
    /// swapped, so it counts against a ledger of host RAM like device memory
    /// counts against VRAM.
    pub fn new(device: &Device, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::Invalid {
                what: "pinned allocation",
                detail: "a buffer of 0 bytes was requested".into(),
            });
        }
        let driver = device.bind()?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: the pointer is valid for the write of one address.
        driver.check("cuMemHostAlloc", unsafe {
            (driver.api.mem_host_alloc)(&mut raw, len, CU_MEMHOSTALLOC_PORTABLE)
        })?;
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or(Error::Invalid {
            what: "pinned allocation",
            detail: "the driver returned a null pointer".into(),
        })?;
        // SAFETY: the driver returned `len` writable bytes at `ptr`. Zeroing
        // makes the contents initialised so they can be viewed as a slice.
        unsafe { ptr.as_ptr().write_bytes(0, len) };
        Ok(Self { device: device.clone(), ptr, len })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Deref for PinnedBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` points to `len` initialised bytes owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for PinnedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `deref`, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the pointer came from cuMemHostAlloc and is freed once.
            unsafe { (driver.api.mem_free_host)(self.ptr.as_ptr().cast()) };
        }
    }
}

/// Existing host memory that was page-locked in place, unlocked when dropped.
///
/// This is how mapped model files become a fast source for device copies
/// without a second copy in RAM: the mapped pages themselves are locked.
#[derive(Debug)]
pub struct HostRegistration {
    device: Device,
    ptr: *mut c_void,
}

// SAFETY: the registration is an opaque token; unregistering may happen on
// any thread.
unsafe impl Send for HostRegistration {}

impl HostRegistration {
    /// Page-locks `len` bytes at `ptr`. `read_only` must be set for memory
    /// that cannot be written, such as a read-only file mapping.
    ///
    /// Locking reads the pages in and keeps them in RAM, so the range counts
    /// against host memory in full.
    ///
    /// # Safety
    ///
    /// The range must stay mapped and unchanged in layout until the returned
    /// value is dropped, and `ptr` must be aligned to the page size with `len`
    /// a multiple of it.
    pub unsafe fn new(
        device: &Device,
        ptr: *const u8,
        len: usize,
        read_only: bool,
    ) -> Result<Self> {
        let driver = device.bind()?;
        let mut flags = CU_MEMHOSTREGISTER_PORTABLE;
        if read_only {
            flags |= CU_MEMHOSTREGISTER_READ_ONLY;
        }
        let ptr = ptr.cast_mut().cast::<c_void>();
        // SAFETY: the caller guarantees the range is mapped and page-aligned.
        driver.check("cuMemHostRegister", unsafe {
            (driver.api.mem_host_register)(ptr, len, flags)
        })?;
        Ok(Self { device: device.clone(), ptr })
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the range was registered by `new` and is unregistered once.
            unsafe { (driver.api.mem_host_unregister)(self.ptr) };
        }
    }
}
