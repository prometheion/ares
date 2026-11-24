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
}
