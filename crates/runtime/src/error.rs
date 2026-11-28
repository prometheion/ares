//! Why the runtime could not start or had to stop.

use ares_core::LedgerError;

use crate::model::ModelError;

/// A failure of the runtime itself, as opposed to a request the runtime
/// refuses (which is an [`EngineError`](ares_api::EngineError)).
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Cuda(#[from] ares_cuda::Error),
    #[error("model: {0}")]
    Model(#[from] ModelError),
    #[error("device memory: {0}")]
    Memory(#[from] LedgerError),
    /// The limit of page-locked memory has no room for what the executor
    /// cannot run without.
    #[error("page-locked memory: {0}")]
    Pinned(String),
    /// The configuration cannot be run.
    #[error("{0}")]
    Config(String),
    /// A thread could not be started or ended in a way it should not.
    #[error("{0}")]
    Thread(String),
    #[error("device {device}: {message}")]
    Device { device: usize, message: String },
    /// A device could not be set up for lack of memory, its own or
    /// page-locked host memory.
    #[error("device {device}: {message}")]
    NoRoom { device: usize, message: String },
}

impl RuntimeError {
    /// Whether the error is a lack of memory on a device or in page-locked
    /// host memory: next to other models, unloading one may make room.
    pub fn is_no_room(&self) -> bool {
        match self {
            Self::NoRoom { .. } | Self::Memory(_) | Self::Pinned(_) => true,
            Self::Cuda(e) => e.is_out_of_memory(),
            Self::Model(e) => e.is_no_room(),
            Self::Config(_) | Self::Thread(_) | Self::Device { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lack_of_memory_is_known_by_its_kind_and_not_by_its_words() {
        let room = |message: &str| ModelError::NoRoom(message.into());
        assert!(RuntimeError::Model(room("the layers do not fit")).is_no_room());
        assert!(RuntimeError::NoRoom { device: 1, message: "set up".into() }.is_no_room());
        assert!(RuntimeError::Pinned("no room in --ram".into()).is_no_room());
        let oom = ares_cuda::Error::Driver {
            call: "cuMemAlloc",
            code: 2,
            name: "CUDA_ERROR_OUT_OF_MEMORY".into(),
            description: "out of memory".into(),
        };
        assert!(RuntimeError::Model(ModelError::Cuda(oom)).is_no_room());
        // Words about memory in another kind of error are no lack of it.
        assert!(!RuntimeError::Device { device: 0, message: "memory".into() }.is_no_room());
        assert!(!RuntimeError::Model(ModelError::Invalid("does not fit".into())).is_no_room());
    }
}
