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
    #[error("the kernel module has no function '{0}'")]
    NoSuchKernel(String),
    #[error("{what}: {detail}")]
    Invalid { what: &'static str, detail: String },
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
