//! The engine's page-locked host memory, kept in one broker.
//!
//! The engine has one limit of host RAM (see "Host memory" in
//! `docs/ARCHITECTURE.md`), and the users of that memory make room for each
//! other through an [`ares_core::Broker`]. Page-locked memory is what the
//! devices copy to and from without a bounce buffer, so it is where state that
//! lives in RAM has to be; today there are two users:
//!
//! - the step record and the read-back buffers, which the executor needs to
//!   run at all (required);
//! - the checkpoints of recurrent state that the prefix cache keeps (elastic,
//!   low priority: a cache gives way to whatever needs the memory). A device
//!   part locks them in chunks, when a place is first used, so the memory
//!   the cache holds is the memory it has used and not the whole budget.
//!
//! The books are shared by the executor and the device threads, so they sit
//! behind a lock; a grant is recorded in the same call that makes it, and a
//! chunk gives its bytes back when it is dropped.

use std::sync::{Arc, Mutex, MutexGuard};

use ares_core::{Broker, Class, Consumer, Growth};
use ares_cuda::{Device, PinnedBuffer};

use crate::error::RuntimeError;
use crate::model::ModelError;

/// Checkpoints are a cache of the lowest kind: a prompt that finds none
/// replays more rows, and nothing else suffers.
const CHECKPOINT_PRIORITY: u32 = 1;

struct Books {
    broker: Broker,
    /// The step record and the read-back buffers.
    required: Consumer,
    /// The checkpoints of recurrent state.
    checkpoints: Consumer,
}

/// The page-locked RAM of an engine and what it holds. Clones share the books.
#[derive(Clone)]
pub struct PinnedRam {
    books: Arc<Mutex<Books>>,
}

impl std::fmt::Debug for PinnedRam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let books = self.books();
        f.debug_struct("PinnedRam")
            .field("limit", &books.broker.capacity())
            .field("used", &books.broker.used())
            .finish()
    }
}

impl PinnedRam {
    /// Books for at most `limit` bytes of page-locked RAM.
    pub fn new(limit: u64) -> Self {
        let mut broker = Broker::new("pinned RAM", limit);
        // Two consumers of a fresh broker with distinct names and no floor
        // cannot be refused.
        let required = broker
            .register("step record and read-back", Class::Required)
            .expect("a fresh broker takes a consumer");
        let checkpoints = broker
            .register(
                "prompt cache: checkpoints of recurrent state",
                Class::Elastic { priority: CHECKPOINT_PRIORITY, floor: 0 },
            )
            .expect("a fresh broker takes a consumer");
        Self { books: Arc::new(Mutex::new(Books { broker, required, checkpoints })) }
    }

    fn books(&self) -> MutexGuard<'_, Books> {
        self.books.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The most the engine may lock.
    pub fn limit(&self) -> u64 {
        self.books().broker.capacity()
    }

    /// Bytes locked now, by everyone.
    pub fn used(&self) -> u64 {
        self.books().broker.used()
    }

    /// Bytes the prompt cache's checkpoints hold now.
    pub fn checkpoints(&self) -> u64 {
        let books = self.books();
        books.broker.info(books.checkpoints.id()).map_or(0, |info| info.usage)
    }

    /// Books `bytes` for what the executor cannot run without.
    pub(crate) fn require(&self, bytes: u64) -> Result<(), RuntimeError> {
        let mut books = self.books();
        let Books { broker, required, .. } = &mut *books;
        match broker.grow(required, bytes) {
            Ok(Growth::Granted) => Ok(()),
            Ok(Growth::Reclaim(_)) => unreachable!("nobody else holds page-locked memory yet"),
            Err(error) => Err(RuntimeError::Pinned(error)),
        }
    }

    /// Locks `bytes` of RAM on behalf of the checkpoint cache, or says who
    /// holds the memory that the limit does not leave. The bytes are booked
    /// until the chunk is dropped.
    pub fn lock_checkpoints(
        &self,
        device: &Device,
        bytes: usize,
    ) -> Result<LockedChunk, ModelError> {
        {
            let mut books = self.books();
            let Books { broker, checkpoints, .. } = &mut *books;
            match broker.grow(checkpoints, bytes as u64) {
                Ok(Growth::Granted) => {}
                // Nothing ranks below the checkpoints, so there is nobody to
                // take from; a plan would be a refusal.
                Ok(Growth::Reclaim(plan)) => {
                    return Err(ModelError::Invalid(format!(
                        "the RAM limit of {} bytes has no room for {bytes} more bytes of \
                         checkpoints, and {} would have to come from others",
                        broker.capacity(),
                        plan.total()
                    )));
                }
                Err(error) => return Err(ModelError::Invalid(error.to_string())),
            }
        }
        match PinnedBuffer::new(device, bytes) {
            Ok(buffer) => Ok(LockedChunk { buffer, ram: self.clone() }),
            Err(error) => {
                self.release_checkpoints(bytes as u64);
                Err(error.into())
            }
        }
    }

    fn release_checkpoints(&self, bytes: u64) {
        let mut books = self.books();
        let Books { broker, checkpoints, .. } = &mut *books;
        // The bytes were granted by `lock_checkpoints`, so the books hold them.
        let _ = broker.shrink(checkpoints, bytes);
    }
}

/// Page-locked memory booked for the checkpoint cache; the booking ends with
/// the chunk.
#[derive(Debug)]
pub struct LockedChunk {
    buffer: PinnedBuffer,
    ram: PinnedRam,
}

impl LockedChunk {
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl Drop for LockedChunk {
    fn drop(&mut self) {
        self.ram.release_checkpoints(self.buffer.len() as u64);
    }
}

#[cfg(test)]
mod tests {
    use ares_core::units::MIB;

    use super::*;

    #[test]
    fn the_executors_memory_and_the_checkpoints_share_one_limit() {
        let ram = PinnedRam::new(10 * MIB);
        ram.require(4 * MIB).unwrap();
        assert_eq!((ram.used(), ram.checkpoints(), ram.limit()), (4 * MIB, 0, 10 * MIB));
        // What the required users hold is not left to the checkpoints.
        let error = ram.require(7 * MIB).unwrap_err().to_string();
        assert!(error.contains("step record and read-back"), "{error}");
        assert_eq!(ram.used(), 4 * MIB, "a refusal books nothing");
    }
}
