use std::path::PathBuf;

/// Why a GGUF file could not be read.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: not a GGUF file")]
    NotGguf { path: PathBuf },
    #[error("{path}: GGUF version {version} is not supported (version 3 is)")]
    Version { path: PathBuf, version: u32 },
    #[error("{path}: file ends inside {what} at byte {offset}")]
    Truncated { path: PathBuf, what: &'static str, offset: u64 },
    #[error("{path}: {what}")]
    Invalid { path: PathBuf, what: String },
    #[error(
        "{path}: tensor '{tensor}' has element type id {type_id}, which this engine does not know"
    )]
    UnknownTensorType { path: PathBuf, tensor: String, type_id: u32 },
    #[error("{path}: part {part} of {count} of this split model is missing (expected {expected})")]
    MissingSplit { path: PathBuf, part: u32, count: u32, expected: PathBuf },
}

pub type Result<T> = std::result::Result<T, Error>;
