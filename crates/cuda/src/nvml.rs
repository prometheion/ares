//! What the NVIDIA management library says of a device: how much of its
//! memory this process and every other one holds, and whether it drives a
//! display.
//!
//! NVML (`libnvidia-ml.so.1`) ships with the driver, as `libcuda.so.1` does,
//! and is bound the same way, at run time. It needs no context on the device:
//! a look costs tens of microseconds and allocates nothing there, so the
//! watch over the devices' memory can look often (see "VRAM" in
//! `docs/dev/architecture/devices.md`). The driver's own count of memory per
//! process is what `nvidia-smi` shows: for this process it includes the
//! context and the kernels' code, which no allocation of the engine names.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use libloading::Library;

use crate::device::PciAddress;
use crate::error::{Error, Result};

type NvmlReturn = c_int;
type NvmlDevice = *mut c_void;

const NVML_SUCCESS: NvmlReturn = 0;
const NVML_ERROR_INSUFFICIENT_SIZE: NvmlReturn = 7;
/// What NVML gives for the memory of a process it cannot see (another
/// container, a driver model that does not count per process).
const NVML_VALUE_NOT_AVAILABLE: u64 = u64::MAX;

/// `nvmlMemory_t`.
#[repr(C)]
#[derive(Default)]
struct Memory {
    total: u64,
    free: u64,
    used: u64,
}

/// `nvmlMemory_v2_t`: the driver's reserve apart from what processes use.
#[repr(C)]
#[derive(Default)]
struct MemoryV2 {
    version: c_uint,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}

/// `nvmlProcessInfo_t` (the `_v3` calls).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: u64,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

type Processes = unsafe extern "C" fn(NvmlDevice, *mut c_uint, *mut ProcessInfo) -> NvmlReturn;

/// The bound functions. The library stays loaded for as long as this exists.
struct Api {
    _library: Library,
    error_string: unsafe extern "C" fn(NvmlReturn) -> *const c_char,
    by_pci: unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn,
    memory: unsafe extern "C" fn(NvmlDevice, *mut Memory) -> NvmlReturn,
    memory_v2: Option<unsafe extern "C" fn(NvmlDevice, *mut MemoryV2) -> NvmlReturn>,
    compute: Processes,
    graphics: Processes,
    display_active: unsafe extern "C" fn(NvmlDevice, *mut c_int) -> NvmlReturn,
    display_mode: unsafe extern "C" fn(NvmlDevice, *mut c_int) -> NvmlReturn,
}

// SAFETY: NVML is thread-safe (its documentation: "all NVML functions are
// thread safe"); the table holds function pointers and the library handle.
unsafe impl Send for Api {}
// SAFETY: as above.
unsafe impl Sync for Api {}

static NVML: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

fn api() -> Result<&'static Api> {
    NVML.get_or_init(Api::load).as_ref().map_err(|why| Error::Nvml(why.clone()))
}

impl Api {
    fn load() -> std::result::Result<Self, String> {
        let name = "libnvidia-ml.so.1";
        // SAFETY: loading NVML runs its initialisers, which have no
        // preconditions on the calling process.
        let library = unsafe { Library::new(name) }.map_err(|e| format!("{name}: {e}"))?;
        macro_rules! bind {
            ($symbol:literal) => {
                // SAFETY: the library is NVML, whose symbol of this name has
                // the signature of the field it is stored in (nvml.h).
                unsafe {
                    *library
                        .get(concat!($symbol, "\0").as_bytes())
                        .map_err(|_| format!("{name} lacks {}", $symbol))?
                }
            };
        }
        let init: unsafe extern "C" fn() -> NvmlReturn = bind!("nvmlInit_v2");
        let api = Self {
            error_string: bind!("nvmlErrorString"),
            by_pci: bind!("nvmlDeviceGetHandleByPciBusId_v2"),
            memory: bind!("nvmlDeviceGetMemoryInfo"),
            // SAFETY: as in `bind!`; a driver without it reports no reserve.
            memory_v2: unsafe { library.get(b"nvmlDeviceGetMemoryInfo_v2\0").ok().map(|f| *f) },
            compute: bind!("nvmlDeviceGetComputeRunningProcesses_v3"),
            graphics: bind!("nvmlDeviceGetGraphicsRunningProcesses_v3"),
            display_active: bind!("nvmlDeviceGetDisplayActive"),
            display_mode: bind!("nvmlDeviceGetDisplayMode"),
            _library: library,
        };
        // SAFETY: no arguments; NVML may be initialised more than once.
        let code = unsafe { init() };
        if code != NVML_SUCCESS {
            return Err(format!("nvmlInit failed: {}", api.text(code)));
        }
        Ok(api)
    }

    fn text(&self, code: NvmlReturn) -> String {
        // SAFETY: NVML returns a pointer to a static NUL-terminated string.
        let text = unsafe { (self.error_string)(code) };
        if text.is_null() {
            return format!("error {code}");
        }
        // SAFETY: as above.
        unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
    }

    fn check(&self, call: &str, code: NvmlReturn) -> Result<()> {
        match code {
            NVML_SUCCESS => Ok(()),
            code => Err(Error::Nvml(format!("{call} failed: {}", self.text(code)))),
        }
    }

    fn device(&self, pci: PciAddress) -> Result<NvmlDevice> {
        let bus = CString::new(pci.to_string()).expect("an address has no NUL");
        let mut device: NvmlDevice = std::ptr::null_mut();
        // SAFETY: a NUL-terminated string and a pointer valid for one handle.
        self.check("nvmlDeviceGetHandleByPciBusId", unsafe {
            (self.by_pci)(bus.as_ptr(), &mut device)
        })?;
        Ok(device)
    }

    /// The processes of one of NVML's two lists: what each holds.
    fn processes(&self, call: &str, list: Processes, device: NvmlDevice) -> Result<Vec<Process>> {
        let mut capacity = 32;
        loop {
            let mut entries = vec![ProcessInfo::default(); capacity];
            let mut count = capacity as c_uint;
            // SAFETY: `entries` has room for `count` entries.
            let code = unsafe { list(device, &mut count, entries.as_mut_ptr()) };
            if code == NVML_ERROR_INSUFFICIENT_SIZE && capacity < 4096 {
                capacity = (count as usize).max(capacity * 2);
                continue;
            }
            self.check(call, code)?;
            entries.truncate(count as usize);
            return Ok(entries
                .iter()
                .map(|e| Process {
                    pid: e.pid,
                    bytes: (e.used_gpu_memory != NVML_VALUE_NOT_AVAILABLE)
                        .then_some(e.used_gpu_memory),
                })
                .collect());
        }
    }

    fn flag(
        &self,
        f: unsafe extern "C" fn(NvmlDevice, *mut c_int) -> NvmlReturn,
        d: NvmlDevice,
    ) -> bool {
        let mut value: c_int = 0;
        // SAFETY: the pointer is valid for one int.
        unsafe { f(d, &mut value) == NVML_SUCCESS && value == 1 }
    }
}

/// A process that holds memory on a device, and how much when NVML can say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Process {
    pub pid: u32,
    pub bytes: Option<u64>,
}

/// A device's memory and displays, as NVML reports them now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceUse {
    /// All the memory of the device, the driver's own reserve included.
    pub total: u64,
    /// What no one holds.
    pub free: u64,
    /// What the driver keeps for itself, apart from every process (0 when
    /// the driver does not say).
    pub reserved: u64,
    /// What this process holds there: its context, the kernels' code and its
    /// allocations.
    pub own: u64,
    /// Every other process that holds memory there.
    pub others: Vec<Process>,
    /// The device drives a display: NVML's display active (a display is
    /// initialised on it) or display mode (a display is connected; newer
    /// drivers no longer say).
    pub display: bool,
}

impl DeviceUse {
    /// What the other processes hold together: the sum of what each holds,
    /// or, when NVML cannot say for one of them, what the device holds that
    /// is neither this process's nor the driver's reserve.
    pub fn others_bytes(&self) -> u64 {
        match self.others.iter().map(|p| p.bytes).sum::<Option<u64>>() {
            Some(bytes) => bytes,
            None => self.total.saturating_sub(self.free + self.reserved + self.own),
        }
    }
}

/// What NVML reports of the device at `pci`: an error when NVML cannot be
/// loaded or does not know the device.
pub fn device_use(pci: PciAddress) -> Result<DeviceUse> {
    let api = api()?;
    let device = api.device(pci)?;
    let (total, free, reserved) = match api.memory_v2 {
        Some(memory_v2) => {
            let mut memory = MemoryV2 {
                version: (std::mem::size_of::<MemoryV2>() as c_uint) | (2 << 24),
                ..MemoryV2::default()
            };
            // SAFETY: a versioned struct of the size its version names.
            api.check("nvmlDeviceGetMemoryInfo_v2", unsafe { memory_v2(device, &mut memory) })?;
            (memory.total, memory.free, memory.reserved)
        }
        None => {
            let mut memory = Memory::default();
            // SAFETY: the pointer is valid for the struct.
            api.check("nvmlDeviceGetMemoryInfo", unsafe { (api.memory)(device, &mut memory) })?;
            (memory.total, memory.free, 0)
        }
    };
    let mut processes =
        api.processes("nvmlDeviceGetComputeRunningProcesses", api.compute, device)?;
    // A process that draws and computes is in both lists, with the same
    // memory: it counts once, with the larger number.
    for process in api.processes("nvmlDeviceGetGraphicsRunningProcesses", api.graphics, device)? {
        match processes.iter_mut().find(|p| p.pid == process.pid) {
            Some(known) => known.bytes = known.bytes.max(process.bytes),
            None => processes.push(process),
        }
    }
    let me = std::process::id();
    let own = processes.iter().filter(|p| p.pid == me).filter_map(|p| p.bytes).sum();
    processes.retain(|p| p.pid != me);
    let display = api.flag(api.display_active, device) || api.flag(api.display_mode, device);
    Ok(DeviceUse { total, free, reserved, own, others: processes, display })
}

/// Whether NVML can be used, and why not.
pub fn nvml_available() -> Result<()> {
    api().map(drop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_others_hold_is_their_sum_or_what_is_nobody_elses() {
        let mib = 1u64 << 20;
        let mut seen = DeviceUse {
            total: 12_288 * mib,
            free: 9_000 * mib,
            reserved: 300 * mib,
            own: 2_000 * mib,
            others: vec![
                Process { pid: 10, bytes: Some(600 * mib) },
                Process { pid: 11, bytes: Some(300 * mib) },
            ],
            display: true,
        };
        assert_eq!(seen.others_bytes(), 900 * mib);
        // One NVML cannot see: what the device holds beyond this process and
        // the driver's reserve.
        seen.others[1].bytes = None;
        assert_eq!(seen.others_bytes(), 988 * mib);
        seen.others.clear();
        assert_eq!(seen.others_bytes(), 0);
    }

    #[test]
    fn the_structs_have_the_layout_of_nvml_h() {
        assert_eq!(std::mem::size_of::<ProcessInfo>(), 24);
        assert_eq!(std::mem::size_of::<Memory>(), 24);
        assert_eq!(std::mem::size_of::<MemoryV2>(), 40);
    }
}
