//! The vocabulary of a read: what is asked, and what comes back.

use std::time::Duration;

use crate::buffer::Buffer;
use crate::error::Error;
use crate::file::FileId;

/// The caller's name for a read. Keys of reads whose completions have not
/// been taken must be distinct; cancelling, promoting and waiting address a
/// read by its key.
pub type Key = u64;

/// How urgent a read is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// The model is waiting for these bytes. Demand reads are always started
    /// before any prefetch and are never split into small pieces.
    Demand,
    /// A guess. A prefetch is read in chunks so that a demand read never
    /// waits behind a long one; it can be cancelled when the guess was wrong
    /// and promoted when it was right.
    Prefetch,
}

/// A read to submit: `len` bytes at `offset` of `file`, into `buffer`.
#[derive(Debug)]
pub struct Read {
    pub key: Key,
    pub file: FileId,
    pub offset: u64,
    pub len: usize,
    pub class: Class,
    /// Must be at least [`Span::len`](crate::Span::len) bytes (see
    /// [`Reader::span`](crate::Reader::span)) and aligned to
    /// [`Reader::buffer_alignment`](crate::Reader::buffer_alignment).
    pub buffer: Buffer,
}

/// A read that was refused before it was queued. Everything handed in comes
/// back, so the buffer is not lost. (It is boxed in the `Err` of
/// [`Reader::submit`](crate::Reader::submit) to keep the happy path small.)
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct Rejected {
    pub error: Error,
    pub read: Read,
}

/// A finished read.
#[derive(Debug)]
pub struct Completion {
    pub key: Key,
    /// The class the read was submitted with.
    pub class: Class,
    /// Whether it was promoted from a prefetch to a demand read on the way.
    pub promoted: bool,
    /// `Ok` when the wanted bytes arrived. Otherwise why not: a short read
    /// (the file ends before the range does), an I/O error, or
    /// [`Error::Cancelled`]. Only this read is affected.
    pub result: Result<(), Error>,
    /// The buffer that was submitted, returned for reuse. Its memory is no
    /// longer touched by the kernel.
    pub buffer: Buffer,
    /// Where the first wanted byte is in the buffer.
    pub skip: usize,
    /// How many wanted bytes are there (0 when the read failed).
    pub len: usize,
    /// From the call of `submit` to the moment the last chunk was known to
    /// have finished.
    pub latency: Duration,
    /// Position of this completion in the order the reader finished reads,
    /// starting at 0. It orders completions that are taken in one batch.
    pub order: u64,
}

impl Completion {
    /// The bytes that were asked for; empty when the read failed.
    pub fn data(&self) -> &[u8] {
        match self.result {
            Ok(()) => &self.buffer.as_slice()[self.skip..self.skip + self.len],
            Err(_) => &[],
        }
    }
}

/// What [`Reader::cancel`](crate::Reader::cancel) did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cancel {
    /// The read will complete with [`Error::Cancelled`], as soon as the
    /// chunks already at the device have come back (they cannot be recalled,
    /// and the kernel still writes into the buffer).
    Cancelled,
    /// The read had already finished; its completion is waiting to be taken
    /// and holds the data.
    TooLate,
}

/// What [`Reader::promote`](crate::Reader::promote) did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Promote {
    /// The remaining chunks now go in the demand queue, behind demand reads
    /// that were already waiting and ahead of every prefetch.
    Promoted,
    /// The read was a demand read already.
    AlreadyDemand,
    /// The read had already finished or was cancelled.
    TooLate,
}
