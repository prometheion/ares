use std::path::PathBuf;

use crate::{FileId, Key};

/// Why a read could not be set up, or why one read failed.
///
/// A failed read is reported on its own [`Completion`](crate::Completion) and
/// never affects the other reads in the queue.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot open {path} for reading: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "cannot set up the io_uring ring: {0} (the kernel must be Linux 6.0 or newer with \
         io_uring enabled; check `sysctl kernel.io_uring_disabled`, seccomp filters of a \
         container, and the limit of locked memory)"
    )]
    Ring(#[source] std::io::Error),

    #[error("cannot start the I/O thread: {0}")]
    Thread(#[source] std::io::Error),

    #[error("invalid configuration: {0}")]
    Config(&'static str),

    #[error("no file with id {0} (this reader was opened with {1} files)")]
    UnknownFile(FileId, usize),

    #[error(
        "the buffer of read {key} holds {have} bytes but the read needs {need}: the range, \
         widened to the {block}-byte blocks direct I/O works in, is that long (size buffers \
         with Reader::span)"
    )]
    BufferTooSmall { key: Key, have: usize, need: usize, block: u64 },

    #[error(
        "the buffer of read {key} starts at address {address:#x}, which is not a multiple of \
         {align}, the alignment direct I/O needs for this file (allocate it with \
         Buffer::aligned using Reader::buffer_alignment)"
    )]
    BufferMisaligned { key: Key, address: usize, align: u64 },

    #[error("the range at offset {offset} of length {len} runs past the end of the address space")]
    RangeOverflow { offset: u64, len: usize },

    #[error("key {0} is already used by a read whose completion has not been taken yet")]
    DuplicateKey(Key),

    #[error(
        "no read with key {0} is outstanding (it was never submitted or its completion was already taken)"
    )]
    UnknownKey(Key),

    #[error("the I/O thread has stopped: {0}")]
    Stopped(String),

    // The errors below are what a single read can end with.
    #[error("{path}: reading {len} bytes at offset {offset} failed: {source}")]
    Io {
        path: PathBuf,
        offset: u64,
        len: usize,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "{path}: asked for {wanted} bytes at offset {offset} but the file delivered only {got} \
         (it is {size} bytes long; the model file may be truncated, or the range is wrong)"
    )]
    ShortRead { path: PathBuf, offset: u64, wanted: usize, got: usize, size: u64 },

    #[error("the read was cancelled")]
    Cancelled,

    #[error(transparent)]
    Layout(#[from] LayoutError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Why a tensor cannot be addressed as stacked experts or as rows.
///
/// These are mistakes in the model description (a name that is not in the
/// file, tensors of a layer that disagree), found when the map is built, not
/// while reading.
#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("tensor '{0}' is not in the model file")]
    UnknownTensor(String),

    #[error(
        "tensor '{name}' has {dims} dimensions; stacked expert tensors have 3 \
         (values per row, rows, experts)"
    )]
    NotStacked { name: String, dims: usize },

    #[error(
        "tensor '{name}' of layer {layer} holds {experts} experts but '{first}' of the same \
         layer holds {first_experts}"
    )]
    ExpertCountMismatch {
        layer: usize,
        name: String,
        experts: u64,
        first: String,
        first_experts: u64,
    },

    #[error("tensor '{name}': its {size} bytes do not divide into {experts} equal experts")]
    Indivisible { name: String, size: u64, experts: u64 },

    #[error("no 3-dimensional tensor has a name containing any of {patterns:?}")]
    NoMatch { patterns: Vec<String> },

    #[error(
        "layer {layer} has tensors for some of the patterns but none whose name contains \
         '{missing}'"
    )]
    IncompleteLayer { layer: usize, missing: String },

    #[error(
        "layer {layer}: both '{first}' and '{second}' contain '{pattern}'; the pattern must \
         match one tensor per layer"
    )]
    AmbiguousMatch { layer: usize, pattern: String, first: String, second: String },

    #[error(
        "tensor '{name}' has {dims} dimensions; a table read by rows has 2 (values per row, rows)"
    )]
    NotATable { name: String, dims: usize },

    #[error("row {row} is outside the table, which has {rows} rows")]
    RowOutOfRange { row: u64, rows: u64 },

    #[error(
        "tensor '{name}' is in part {part} of the model but the reader has {files} files (open it \
         with Reader::open_gguf)"
    )]
    NotInReader { name: String, part: u16, files: usize },

    #[error(
        "the table has {rows} rows of {row_bytes} bytes, which does not fit in 64 bits of offset"
    )]
    TableTooLarge { rows: u64, row_bytes: u64 },
}
