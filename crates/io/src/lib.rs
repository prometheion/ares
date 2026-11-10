//! Reading model weights from the SSD while the model runs.
//!
//! A mixture-of-experts model is larger than VRAM and often larger than RAM,
//! and each token uses a few of the hundreds of experts per layer. The experts
//! that are on neither tier are read from the model file while the model
//! runs; the same is true for rows of the huge hashed-embedding tables some
//! models carry. Deciding *what* to read is the job of `ares-weights`; this
//! crate does the reading, fast, in priority order, cancellably, into memory
//! the caller provides.
//!
//! # Parts
//!
//! - [`Reader`] is the I/O engine. One thread owns an `io_uring` with the
//!   files registered, and reads with `O_DIRECT` where the filesystem allows
//!   it (a file that cannot be read directly is read through the page cache
//!   instead, and [`Reader::files`] says so). The caller queues
//!   [`Read`]s of two [`Class`]es: *demand* reads the model is waiting for,
//!   which always start first, and *prefetches*, guesses that are cut into
//!   small chunks so that a demand read never waits behind a long one, that
//!   can be cancelled when wrong ([`Reader::cancel`]) and promoted when right
//!   ([`Reader::promote`]).
//! - [`ExpertMap`] turns `(layer, expert)` into the byte ranges to read from
//!   the stacked expert tensors of a GGUF model, split files included.
//! - [`RowTable`], [`RowCache`] and [`RowReader`] read rows of a big 2-D
//!   table: row ids become few block-aligned reads, and recently used rows
//!   are kept in a bounded cache.
//!
//! # Why reads are widened
//!
//! GGUF aligns tensor data to 32 bytes, direct I/O works in blocks (the
//! logical block size of the device, 512 or 4096 bytes, queried at open). An
//! expert therefore starts in the middle of a block. The reader reads the
//! surrounding blocks into the caller's buffer and reports where the wanted
//! bytes are in it ([`Completion::skip`]), so the bytes are used where they
//! land: nothing is copied or moved afterwards, and the destination can be a
//! page-locked staging buffer of a GPU backend ([`Buffer::from_raw`]).
//! [`Reader::span`] says how large the buffer must be.
//!
//! # Threads
//!
//! The engine's step loop is the one consumer. It queues reads (a mutex held
//! for a few instructions and, when the I/O thread is asleep, one write to an
//! eventfd), and finds out what has finished with [`Reader::poll`], which is a
//! single atomic load when nothing has: no lock and no system call. It can
//! also block for one key with a timeout ([`Reader::wait`]).
//!
//! ```no_run
//! use std::time::Duration;
//! use ares_gguf::Gguf;
//! use ares_io::{Buffer, Class, Config, ExpertMap, Read, Reader};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gguf::open("model-00001-of-00002.gguf")?;
//! let experts = ExpertMap::discover(&model, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"])?;
//! let reader = Reader::open_gguf(&model, Config::default())?;
//!
//! // Layer 3, expert 17: one demand read per tensor, each into its own buffer.
//! let mut key = 0;
//! for range in experts.ranges(3, 17).expect("a MoE layer") {
//!     let span = reader.span(range.file, range.offset, range.len)?;
//!     let buffer = Buffer::aligned(span.len, reader.buffer_alignment());
//!     reader.submit(Read {
//!         key,
//!         file: range.file,
//!         offset: range.offset,
//!         len: range.len,
//!         class: Class::Demand,
//!         buffer,
//!     })?;
//!     key += 1;
//! }
//! for key in 0..key {
//!     let done = reader.wait(key, Duration::from_secs(1))?.expect("in time");
//!     done.result.as_ref().map_err(|e| e.to_string())?;
//!     let bytes: &[u8] = done.data(); // the expert's matrix, in place
//!     # let _ = bytes;
//! }
//! # Ok(())
//! # }
//! ```

mod buffer;
mod error;
mod expert;
mod file;
mod reader;
mod request;
mod rows;
mod sched;
mod stats;
mod worker;

pub use buffer::Buffer;
pub use error::{Error, LayoutError, Result};
pub use expert::{ExpertMap, ExpertRange};
pub use file::{AlignSource, BufferedReason, DirectAlign, FileId, FileInfo, IoMode, Span};
pub use reader::{Config, FixedRegion, Hold, Reader};
pub use request::{Cancel, Class, Completion, Key, Promote, Read, Rejected};
pub use rows::{CacheStats, PlannedRead, RowCache, RowGather, RowPlan, RowReader, RowTable};
pub use stats::{ClassStats, Latency, Stats};
