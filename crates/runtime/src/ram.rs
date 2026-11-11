//! The engine's page-locked host memory, kept in one broker.
//!
//! The engine has one limit of host RAM (see "Host memory" in
//! `docs/ARCHITECTURE.md`), and the users of that memory make room for each
//! other through an [`ares_core::Broker`]. Page-locked memory is what the
//! devices copy to and from without a bounce buffer, so it is where state that
//! lives in RAM has to be; today there are three users:
//!
//! - the step record and the read-back buffers, which the executor needs to
//!   run at all (required), and whatever else a model's parts must have
//!   ([`crate::Model::pinned_ram`]);
//! - the tier of experts in host memory (elastic, high priority: it is what
//!   keeps a mixture-of-experts model from reading its experts from disk, and
//!   it gives way to nothing but what is required). The model takes it at
//!   load, as much as the limit leaves and its experts can use
//!   ([`PinnedRam::obtainable_experts`]);
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

/// The tier of experts ranks above every cache: a step that misses an expert
/// waits for the drive, which is the one thing that makes such a model slow.
const EXPERT_PRIORITY: u32 = 100;

struct Books {
    broker: Broker,
    /// The step record and the read-back buffers.
    required: Consumer,
    /// The checkpoints of recurrent state.
    checkpoints: Consumer,
    /// The tier of experts.
    experts: Consumer,
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
        let experts = broker
            .register("RAM tier of experts", Class::Elastic { priority: EXPERT_PRIORITY, floor: 0 })
            .expect("a fresh broker takes a consumer");
        Self { books: Arc::new(Mutex::new(Books { broker, required, checkpoints, experts })) }
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

    /// Bytes the tier of experts holds now.
    pub fn experts(&self) -> u64 {
        let books = self.books();
        books.broker.info(books.experts.id()).map_or(0, |info| info.usage)
    }

    /// The most the tier of experts could lock now: what the limit leaves
    /// after the required users, counting what the checkpoints could give
    /// back.
    pub fn obtainable_experts(&self) -> u64 {
        let books = self.books();
        books.broker.obtainable(&books.experts).unwrap_or(0)
    }

    /// Locks `bytes` of RAM for the tier of experts, in the form the store
    /// reads the file into, or says who holds the memory the limit does not
    /// leave. The bytes are booked until the region is dropped.
    pub fn lock_experts(
        &self,
        device: &Device,
        bytes: u64,
    ) -> Result<crate::expert_store::HostRegion, ModelError> {
        let len = bytes.max(1).next_multiple_of(4096);
        {
            let mut books = self.books();
            let Books { broker, experts, .. } = &mut *books;
            match broker.grow(experts, len) {
                Ok(Growth::Granted) => {}
                Ok(Growth::Reclaim(plan)) => {
                    return Err(ModelError::Invalid(format!(
                        "the RAM limit of {} bytes has no room for {len} bytes for the tier of \
                         experts: {} would have to come from others",
                        broker.capacity(),
                        plan.total()
                    )));
                }
                Err(error) => return Err(ModelError::Invalid(error.to_string())),
            }
        }
        let booking = ExpertsBooking { ram: self.clone(), bytes: len };
        crate::expert_store::HostRegion::locked(device, len as usize, booking)
            .map_err(|e| ModelError::Invalid(e.to_string()))
    }

    fn release_experts(&self, bytes: u64) {
        let mut books = self.books();
        let Books { broker, experts, .. } = &mut *books;
        let _ = broker.shrink(experts, bytes);
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

/// What a region of the tier of experts gives back to the books when it is
/// dropped.
struct ExpertsBooking {
    ram: PinnedRam,
    bytes: u64,
}

impl Drop for ExpertsBooking {
    fn drop(&mut self) {
        self.ram.release_experts(self.bytes);
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
    fn the_tier_of_experts_may_take_all_the_limit_leaves_and_the_checkpoints_give_way_to_it() {
        let ram = PinnedRam::new(100 * MIB);
        ram.require(10 * MIB).unwrap();
        assert_eq!((ram.obtainable_experts(), ram.experts()), (90 * MIB, 0));
        // The checkpoints hold 40 MiB; the tier of experts, which ranks above
        // them, could still have all 90.
        {
            let mut books = ram.books();
            let Books { broker, checkpoints, .. } = &mut *books;
            assert_eq!(broker.grow(checkpoints, 40 * MIB).unwrap(), Growth::Granted);
        }
        assert_eq!(ram.obtainable_experts(), 90 * MIB);
        {
            let mut books = ram.books();
            let Books { broker, experts, .. } = &mut *books;
            match broker.grow(experts, 90 * MIB).unwrap() {
                Growth::Reclaim(plan) => {
                    assert_eq!(plan.total(), 40 * MIB, "the checkpoints give back what they hold");
                }
                other => panic!("the checkpoints hold memory the tier needs: {other:?}"),
            }
            assert_eq!(broker.grow(experts, 50 * MIB).unwrap(), Growth::Granted);
        }
        // And the checkpoints cannot take memory back from the tier.
        let mut books = ram.books();
        let Books { broker, checkpoints, .. } = &mut *books;
        assert!(broker.grow(checkpoints, 20 * MIB).is_err());
    }

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
