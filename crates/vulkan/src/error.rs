use ash::vk;

/// Why a Vulkan operation failed.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("the Vulkan loader (libvulkan.so.1) could not be loaded: {0}")]
    LibraryNotFound(String),
    /// A Vulkan call returned an error code.
    #[error("{call} failed: {name} ({code})")]
    Vulkan { call: &'static str, code: i32, name: String },
    #[error("there is no Vulkan device {ordinal} (the loader reports {count} usable)")]
    NoSuchDevice { ordinal: u32, count: u32 },
    /// The device exists but lacks something the engine needs.
    #[error("{device} cannot be used by the engine: {reason}")]
    Unsupported { device: String, reason: String },
    #[error("the kernel module has no function '{0}'")]
    NoSuchKernel(String),
    #[error("{what}: {detail}")]
    Invalid { what: &'static str, detail: String },
}

impl Error {
    /// The Vulkan result code, when a Vulkan call reported one.
    pub fn code(&self) -> Option<i32> {
        match self {
            Error::Vulkan { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// Whether Vulkan refused an allocation for lack of memory.
    pub fn is_out_of_memory(&self) -> bool {
        matches!(
            self.code(),
            Some(c) if c == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw()
                || c == vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw()
        )
    }

    /// Whether the device was lost (a crash or a reset). Nothing on it can be
    /// used afterwards.
    pub fn is_device_lost(&self) -> bool {
        self.code() == Some(vk::Result::ERROR_DEVICE_LOST.as_raw())
    }

    /// Wraps a result code, naming the call that returned it.
    pub(crate) fn vk(call: &'static str, result: vk::Result) -> Self {
        Error::Vulkan { call, code: result.as_raw(), name: format!("{result:?}") }
    }

    pub(crate) fn invalid(what: &'static str, detail: impl Into<String>) -> Self {
        Error::Invalid { what, detail: detail.into() }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Turns the `Result` of an ash call into one that names the call.
pub(crate) trait Check<T> {
    fn check(self, call: &'static str) -> Result<T>;
}

impl<T> Check<T> for std::result::Result<T, vk::Result> {
    fn check(self, call: &'static str) -> Result<T> {
        self.map_err(|code| Error::vk(call, code))
    }
}
