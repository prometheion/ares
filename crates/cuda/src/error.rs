use crate::memory::DevicePtr;

/// Why a CUDA operation failed.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("the CUDA driver library could not be loaded: {0}")]
    LibraryNotFound(String),
    #[error(
        "the CUDA driver library lacks {0}; the driver is older than this engine supports \
         (CUDA 13.0 or newer is needed)"
    )]
    MissingSymbol(&'static str),
    /// A driver call returned an error code.
    #[error("{call} failed: {name} ({code}): {description}")]
    Driver { call: &'static str, code: i32, name: String, description: String },
    #[error("there is no CUDA device {ordinal} (the driver reports {count})")]
    NoSuchDevice { ordinal: u32, count: u32 },
    /// The device is of a class this build has no machine code for
    /// ([`crate::DeviceInfo::check_class`]); `built` names the classes it has.
    #[error(
        "device {ordinal} ({name}) has compute capability {major}.{minor}, a class this build \
         does not run: CUDA runs only on tested GPU classes, and this build has machine code for \
         {built}. Other GPUs are to run through the Vulkan backend, which does not run models yet"
    )]
    UntestedClass { ordinal: u32, name: String, major: u32, minor: u32, built: String },
    #[error("the kernel module has no function '{0}'")]
    NoSuchKernel(String),
    #[error("{what}: {detail}")]
    Invalid { what: &'static str, detail: String },
    /// NVML, the driver's management library, could not be used or refused a
    /// call ([`crate::device_use`]).
    #[error("NVML: {0}")]
    Nvml(String),
}

impl Error {
    /// The driver's error code, when the driver reported one.
    pub fn code(&self) -> Option<i32> {
        match self {
            Error::Driver { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// Whether the driver refused an allocation for lack of memory.
    pub fn is_out_of_memory(&self) -> bool {
        self.code() == Some(2)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// An [`Error::Invalid`] about `what`.
pub(crate) fn invalid(what: &'static str, detail: String) -> Error {
    Error::Invalid { what, detail }
}

/// An [`Error::Invalid`] about `what` unless `ptr` lies on a multiple of `align` bytes.
pub(crate) fn check_align(what: &'static str, ptr: DevicePtr, align: u64) -> Result<()> {
    if !ptr.as_raw().is_multiple_of(align) {
        return Err(invalid(
            what,
            format!("address {:#x} is not aligned to {align} bytes", ptr.as_raw()),
        ));
    }
    Ok(())
}
