use crate::files::FileError;

/// Why a projector could not be configured, loaded or run.
///
/// Every variant names the key, tensor or size that is wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file is not a projector of the kind implemented here.
    #[error("{key} is {found}, the qwen3vl implementation needs {expected}")]
    Kind {
        /// The key that tells the kinds apart.
        key: &'static str,
        /// What the file says.
        found: String,
        /// What it has to say.
        expected: &'static str,
    },
    /// The file does not hold what the projector reads: a key or a tensor,
    /// named.
    #[error(transparent)]
    File(#[from] FileError),
    /// The keys are individually fine but do not describe a projector.
    #[error("inconsistent configuration: {what}")]
    Inconsistent {
        /// Which relation between the values does not hold.
        what: String,
    },
    /// Something the device implementation cannot lay out or run.
    #[error("{what} cannot run on the device: {problem}")]
    Device {
        /// The tensor or part of the pass.
        what: String,
        /// Why, as a sentence.
        problem: String,
    },
    /// The device failed.
    #[error("device: {0}")]
    Cuda(#[from] ares_cuda::Error),
    /// Pixels that the tower cannot take.
    #[error("an image of {width} x {height} pixels {problem}")]
    Image {
        /// The width in pixels.
        width: usize,
        /// The height in pixels.
        height: usize,
        /// What is wrong with it, as the rest of a sentence.
        problem: String,
    },
}

/// The result of configuring, loading or running a projector.
pub type Result<T> = std::result::Result<T, Error>;
