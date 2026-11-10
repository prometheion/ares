//! Why the runtime could not start or had to stop.

use ares_core::{BrokerError, LedgerError};

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
    #[error("page-locked memory: {0}")]
    Pinned(#[from] BrokerError),
    /// The configuration cannot be run.
    #[error("{0}")]
    Config(String),
    /// A thread could not be started or ended in a way it should not.
    #[error("{0}")]
    Thread(String),
    #[error("device {device}: {message}")]
    Device { device: usize, message: String },
}
