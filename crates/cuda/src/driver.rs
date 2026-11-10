//! Loading and initialising the driver, once per process.

use std::ffi::{CStr, c_char};
use std::sync::OnceLock;

use libloading::Library;

use crate::error::{Error, Result};
use crate::sys::{Api, CUDA_SUCCESS, CUresult};

pub(crate) struct Driver {
    pub(crate) api: Api,
    version: i32,
}

/// The environment variable that names the driver library to load instead of
/// the system's `libcuda.so.1`.
pub const LIBRARY_ENV: &str = "ARES_LIBCUDA";

static DRIVER: OnceLock<std::result::Result<Driver, Error>> = OnceLock::new();

/// The process-wide driver, loaded and initialised on first use. Every later
/// call returns the same outcome, including the same error.
pub(crate) fn driver() -> Result<&'static Driver> {
    DRIVER.get_or_init(Driver::load).as_ref().map_err(Clone::clone)
}

impl Driver {
    fn load() -> Result<Self> {
        // A driver build with extra capabilities (peer-to-peer access between
        // consumer cards needs a patched library next to patched kernel
        // modules) is selected by path; the default is the system's library.
        let path = std::env::var_os(LIBRARY_ENV).unwrap_or_else(|| "libcuda.so.1".into());
        // SAFETY: loading libcuda runs its initialisers, which have no
        // preconditions on the calling process.
        let library = unsafe { Library::new(&path) }
            .map_err(|e| Error::LibraryNotFound(format!("{}: {e}", path.to_string_lossy())))?;
        // SAFETY: the library was found under the driver's soname.
        let api = unsafe { Api::bind(library) }.map_err(Error::MissingSymbol)?;
        let mut driver = Driver { api, version: 0 };
        // SAFETY: cuInit takes flags, which must be 0.
        driver.check("cuInit", unsafe { (driver.api.init)(0) })?;
        let mut version = 0;
        // SAFETY: the pointer is valid for the write of one int.
        driver.check("cuDriverGetVersion", unsafe {
            (driver.api.driver_get_version)(&mut version)
        })?;
        driver.version = version;
        Ok(driver)
    }

    /// The CUDA version the driver implements, as `(major, minor)`.
    pub(crate) fn version(&self) -> (u32, u32) {
        let v = self.version.max(0) as u32;
        (v / 1000, v % 1000 / 10)
    }

    /// Turns a driver return code into a result that names the call.
    pub(crate) fn check(&self, call: &'static str, code: CUresult) -> Result<()> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let text = |f: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult| {
            let mut ptr: *const c_char = std::ptr::null();
            // SAFETY: the driver writes a pointer to a static string, or
            // leaves it null and returns an error for an unknown code.
            let ok = unsafe { f(code, &mut ptr) } == CUDA_SUCCESS && !ptr.is_null();
            if ok {
                // SAFETY: on success the pointer is a NUL-terminated static string.
                unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
            } else {
                "unknown error".to_owned()
            }
        };
        Err(Error::Driver {
            call,
            code,
            name: text(self.api.get_error_name),
            description: text(self.api.get_error_string),
        })
    }
}

/// The CUDA version the installed driver implements, as `(major, minor)`.
pub fn driver_version() -> Result<(u32, u32)> {
    Ok(driver()?.version())
}
