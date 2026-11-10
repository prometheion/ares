//! Devices and their contexts.

use std::cell::Cell;
use std::ffi::{c_char, c_int};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::driver::{Driver, driver};
use crate::error::{Error, Result};
use crate::sys::*;

/// Where a device sits on the PCI bus. Two devices behind the same host
/// bridge can exchange data directly when the driver allows peer access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}:{:02x}:{:02x}.0", self.domain, self.bus, self.device)
    }
}

/// What the driver reports about a device without opening it. Reading this
/// allocates nothing on the device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// The driver's index of the device.
    pub ordinal: u32,
    pub name: String,
    pub uuid: [u8; 16],
    /// Compute capability `(major, minor)`: (8, 9) is an RTX 40 card.
    pub compute_capability: (u32, u32),
    /// Total device memory in bytes.
    pub total_memory: u64,
    pub pci: PciAddress,
    /// Number of streaming multiprocessors.
    pub multiprocessors: u32,
    /// Width of the memory interface in bits and its peak clock in kHz; their
    /// product bounds the memory bandwidth, which bounds decode speed.
    pub memory_bus_width_bits: u32,
    pub memory_clock_khz: u32,
}

/// How a thread waits for the device when it synchronises.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Wait {
    /// Spin on a core. Lowest latency; the choice for the threads that drive
    /// a device during inference.
    #[default]
    Spin,
    /// Yield the core between polls.
    Yield,
    /// Sleep until the device interrupts. For threads that wait rarely.
    Block,
}

/// The number of CUDA devices the driver reports.
pub fn device_count() -> Result<u32> {
    let driver = driver()?;
    let mut count: c_int = 0;
    // SAFETY: the pointer is valid for the write of one int.
    driver.check("cuDeviceGetCount", unsafe { (driver.api.device_get_count)(&mut count) })?;
    Ok(count.max(0) as u32)
}

/// Describes every CUDA device without opening any of them.
pub fn devices() -> Result<Vec<DeviceInfo>> {
    let driver = driver()?;
    (0..device_count()?).map(|ordinal| Ok(describe(driver, ordinal)?.1)).collect()
}

fn describe(driver: &Driver, ordinal: u32) -> Result<(CUdevice, DeviceInfo)> {
    let count = device_count()?;
    if ordinal >= count {
        return Err(Error::NoSuchDevice { ordinal, count });
    }
    let api = &driver.api;
    let mut handle: CUdevice = 0;
    // SAFETY: the pointer is valid for the write of one handle.
    driver.check("cuDeviceGet", unsafe { (api.device_get)(&mut handle, ordinal as c_int) })?;

    let mut name = [0 as c_char; 128];
    // SAFETY: the buffer is valid for `len` bytes; the driver NUL-terminates.
    driver.check("cuDeviceGetName", unsafe {
        (api.device_get_name)(name.as_mut_ptr(), name.len() as c_int, handle)
    })?;
    let name: Vec<u8> = name.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();

    let mut uuid = [0u8; 16];
    // SAFETY: the pointer is valid for the write of 16 bytes.
    driver.check("cuDeviceGetUuid", unsafe { (api.device_get_uuid)(&mut uuid, handle) })?;

    let mut total_memory = 0usize;
    // SAFETY: the pointer is valid for the write of one size.
    let code = unsafe { (api.device_total_mem)(&mut total_memory, handle) };
    driver.check("cuDeviceTotalMem", code)?;

    let attribute = |id: c_int| -> Result<u32> {
        let mut value: c_int = 0;
        // SAFETY: the pointer is valid for the write of one int.
        driver.check("cuDeviceGetAttribute", unsafe {
            (api.device_get_attribute)(&mut value, id, handle)
        })?;
        Ok(value.max(0) as u32)
    };

    let info = DeviceInfo {
        ordinal,
        name: String::from_utf8_lossy(&name).into_owned(),
        uuid,
        compute_capability: (
            attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ),
        total_memory: total_memory as u64,
        pci: PciAddress {
            domain: attribute(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
            bus: attribute(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
            device: attribute(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?,
        },
        multiprocessors: attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        memory_bus_width_bits: attribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
        memory_clock_khz: attribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
    };
    Ok((handle, info))
}

/// Whether the driver lets `from` read and write the memory of `to` directly
/// (peer access over PCIe), and the driver's rank of that link (higher is
/// faster; 0 when unsupported). Opens neither device.
pub fn peer_access(from: u32, to: u32) -> Result<(bool, u32)> {
    let driver = driver()?;
    let (from, _) = describe(driver, from)?;
    let (to, _) = describe(driver, to)?;
    let mut supported: c_int = 0;
    // SAFETY: the pointer is valid for the write of one int.
    driver.check("cuDeviceCanAccessPeer", unsafe {
        (driver.api.device_can_access_peer)(&mut supported, from, to)
    })?;
    let mut rank: c_int = 0;
    // SAFETY: as above. A failure here only means the attribute is unknown.
    let known = unsafe {
        (driver.api.device_get_p2p_attribute)(
            &mut rank,
            CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK,
            from,
            to,
        )
    } == CUDA_SUCCESS;
    let mut access: c_int = supported;
    // SAFETY: as above.
    let _ = unsafe {
        (driver.api.device_get_p2p_attribute)(
            &mut access,
            CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
            from,
            to,
        )
    };
    Ok((supported != 0 && access != 0, if known { rank.max(0) as u32 } else { 0 }))
}

/// Memory of an opened device right now.
#[derive(Clone, Copy, Debug)]
pub struct MemInfo {
    pub free: u64,
    pub total: u64,
}

struct Inner {
    info: DeviceInfo,
    handle: CUdevice,
    context: CUcontext,
    /// Distinguishes this opening of the device from earlier ones, whose
    /// context may have had the same address.
    generation: u64,
}

// SAFETY: a CUDA context may be used from any thread, and the driver API is
// thread-safe; the raw pointer is only an opaque handle.
unsafe impl Send for Inner {}
// SAFETY: as above.
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            // SAFETY: this handle retained the primary context exactly once.
            unsafe { (driver.api.primary_ctx_release)(self.handle) };
        }
    }
}

/// An opened device: its primary context, shared by clones of this handle.
///
/// Opening a device creates its context, which takes device memory (a few
/// hundred MiB) for as long as any clone, buffer, stream or module of the
/// device is alive.
#[derive(Clone)]
pub struct Device(Arc<Inner>);

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({}: {})", self.0.info.ordinal, self.0.info.name)
    }
}

static GENERATION: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// The generation of the device whose context is current on this thread.
    static CURRENT: Cell<u64> = const { Cell::new(0) };
}

impl Device {
    /// Opens device `ordinal`. `wait` chooses how threads wait for it.
    pub fn open(ordinal: u32, wait: Wait) -> Result<Self> {
        let driver = driver()?;
        let (handle, info) = describe(driver, ordinal)?;
        let flags = match wait {
            Wait::Spin => 1,
            Wait::Yield => 2,
            Wait::Block => 4,
        };
        // SAFETY: plain value arguments.
        let code = unsafe { (driver.api.primary_ctx_set_flags)(handle, flags) };
        // An active primary context keeps the flags it was created with.
        if code != CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE {
            driver.check("cuDevicePrimaryCtxSetFlags", code)?;
        }
        let mut context: CUcontext = std::ptr::null_mut();
        // SAFETY: the pointer is valid for the write of one handle.
        driver.check("cuDevicePrimaryCtxRetain", unsafe {
            (driver.api.primary_ctx_retain)(&mut context, handle)
        })?;
        let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
        Ok(Self(Arc::new(Inner { info, handle, context, generation })))
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.0.info
    }

    pub fn ordinal(&self) -> u32 {
        self.0.info.ordinal
    }

    pub(crate) fn context(&self) -> CUcontext {
        self.0.context
    }

    /// Makes this device's context current on the calling thread and returns
    /// the driver. Every call that needs a context goes through here; the
    /// thread-local cache makes the common case (a thread that stays on one
    /// device) a comparison instead of a driver call.
    ///
    /// The cache assumes nothing else in the process changes the thread's
    /// CUDA context, which holds as long as this crate is the only CUDA user.
    pub(crate) fn bind(&self) -> Result<&'static Driver> {
        let driver = driver()?;
        if CURRENT.get() != self.0.generation {
            // SAFETY: the context is alive while `self` is.
            driver.check("cuCtxSetCurrent", unsafe {
                (driver.api.ctx_set_current)(self.0.context)
            })?;
            CURRENT.set(self.0.generation);
        }
        Ok(driver)
    }

    /// Free and total memory of the device as the driver sees it now. Other
    /// processes on the device count against `free`.
    pub fn mem_info(&self) -> Result<MemInfo> {
        let driver = self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both pointers are valid for the write of one size.
        let code = unsafe { (driver.api.mem_get_info)(&mut free, &mut total) };
        driver.check("cuMemGetInfo", code)?;
        Ok(MemInfo { free: free as u64, total: total as u64 })
    }

    /// Waits until everything submitted to the device has finished.
    pub fn synchronize(&self) -> Result<()> {
        let driver = self.bind()?;
        // SAFETY: a context is current.
        driver.check("cuCtxSynchronize", unsafe { (driver.api.ctx_synchronize)() })
    }

    /// Lets this device's context address the memory of `peer` directly.
    /// Needed once per direction before peer copies between the two.
    pub fn enable_peer_access(&self, peer: &Device) -> Result<()> {
        let driver = self.bind()?;
        // SAFETY: both contexts are alive; flags must be 0.
        let code = unsafe { (driver.api.ctx_enable_peer_access)(peer.context(), 0) };
        if code == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            return Ok(());
        }
        driver.check("cuCtxEnablePeerAccess", code)
    }
}
