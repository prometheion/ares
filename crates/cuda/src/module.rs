//! Kernel modules and launches.

use std::ffi::{CString, c_void};
use std::sync::Arc;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DevicePtr;
use crate::stream::Stream;
use crate::sys::*;

struct ModuleInner {
    device: Device,
    raw: CUmodule,
}

// SAFETY: a module handle may be used from any thread.
unsafe impl Send for ModuleInner {}
// SAFETY: as above.
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the handle came from cuModuleLoadData and is unloaded once.
            unsafe { (driver.api.module_unload)(self.raw) };
        }
    }
}

/// Compiled kernels loaded on one device.
#[derive(Clone)]
pub struct Module(Arc<ModuleInner>);

impl std::fmt::Debug for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Module(on {:?})", self.0.device)
    }
}

impl Module {
    /// Loads a module image (a fat binary, a cubin or NUL-terminated PTX).
    /// The driver picks the machine code that fits the device, or compiles
    /// the PTX.
    pub fn load(device: &Device, image: &[u8]) -> Result<Self> {
        let driver = device.bind()?;
        let mut raw: CUmodule = std::ptr::null_mut();
        // SAFETY: `image` is a complete module image; the driver copies what
        // it needs before returning.
        driver.check("cuModuleLoadData", unsafe {
            (driver.api.module_load_data)(&mut raw, image.as_ptr().cast())
        })?;
        Ok(Self(Arc::new(ModuleInner { device: device.clone(), raw })))
    }

    /// The kernel named `name`. The returned handle keeps the module loaded.
    pub fn function(&self, name: &str) -> Result<Function> {
        let driver = self.0.device.bind()?;
        let c_name = CString::new(name).map_err(|_| Error::NoSuchKernel(name.to_owned()))?;
        let mut raw: CUfunction = std::ptr::null_mut();
        // SAFETY: the name is NUL-terminated and the module is alive.
        let code =
            unsafe { (driver.api.module_get_function)(&mut raw, self.0.raw, c_name.as_ptr()) };
        if code != CUDA_SUCCESS {
            return Err(Error::NoSuchKernel(name.to_owned()));
        }
        Ok(Function { _module: self.clone(), raw })
    }
}

/// A kernel that can be launched.
#[derive(Clone, Debug)]
pub struct Function {
    _module: Module,
    raw: CUfunction,
}

// SAFETY: a function handle may be used from any thread.
unsafe impl Send for Function {}
// SAFETY: as above.
unsafe impl Sync for Function {}

/// The most arguments a kernel of this engine takes.
pub const MAX_ARGS: usize = 16;

/// The arguments of one launch, in the kernel's parameter order.
///
/// Each value is kept in its own 8-byte slot and the driver reads as many
/// bytes from it as the kernel's parameter has, so the builder methods must be
/// called with the types the kernel declares.
#[derive(Clone, Copy, Debug, Default)]
pub struct Args {
    slots: [u64; MAX_ARGS],
    len: usize,
}

impl Args {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(mut self, bits: u64) -> Self {
        assert!(self.len < MAX_ARGS, "a kernel takes at most {MAX_ARGS} arguments");
        self.slots[self.len] = bits;
        self.len += 1;
        self
    }

    /// A pointer parameter.
    pub fn ptr(self, value: DevicePtr) -> Self {
        self.push(value.0)
    }

    /// A `uint32_t` parameter.
    pub fn u32(self, value: u32) -> Self {
        self.push(value.into())
    }

    /// An `int32_t` parameter.
    pub fn i32(self, value: i32) -> Self {
        self.push(u64::from(value as u32))
    }

    /// A `uint64_t` parameter.
    pub fn u64(self, value: u64) -> Self {
        self.push(value)
    }

    /// A `float` parameter.
    pub fn f32(self, value: f32) -> Self {
        self.push(value.to_bits().into())
    }
}

impl Function {
    /// Enqueues one launch of the kernel on `stream`.
    ///
    /// `grid` and `block` are the grid size in blocks and the block size in
    /// threads; `shared_bytes` is the dynamic shared memory per block.
    ///
    /// # Safety
    ///
    /// `args` must match the kernel's parameter list in number, order and
    /// type. Every pointer argument must address live memory that this
    /// kernel may access as it does, for as long as the launch has not run.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: Args,
    ) -> Result<()> {
        let driver = stream.device().bind()?;
        let mut slots = args.slots;
        let mut pointers = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        for (pointer, slot) in pointers.iter_mut().zip(slots.iter_mut()).take(args.len) {
            *pointer = std::ptr::from_mut(slot).cast();
        }
        // SAFETY: the pointer array has one valid entry per kernel parameter
        // (the caller's guarantee) and the driver copies the values before it
        // returns.
        driver.check("cuLaunchKernel", unsafe {
            (driver.api.launch_kernel)(
                self.raw,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_bytes,
                stream.raw(),
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }
}
