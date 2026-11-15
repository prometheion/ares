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
//! - the tier of experts in host memory (elastic, the highest rank: it is
//!   what keeps a mixture-of-experts model from reading its experts from
//!   disk, and it gives way to nothing but what is required). It is locked in
//!   pieces ([`PinnedRam::expert_pieces`]), as many as its allowance holds;
//! - the checkpoints of recurrent state that the prefix cache keeps (elastic,
//!   the lowest rank: a cache gives way to whatever needs the memory). A
//!   device part locks them in chunks, when a place is first used, so the
//!   memory the cache holds is the memory it has used and not the whole
//!   budget ([`PlaceChunks`]). A cache whose level on SSD moves checkpoints
//!   through a place keeps that one place whatever the machine has: it is
//!   the cache's floor ([`PinnedRam::want_checkpoints`]), and neither the
//!   tier of experts nor the machine is given it.
//!
//! # The limit and the allowances
//!
//! The limit is the most the engine may ever lock: a grant beyond it is
//! refused. Below it the elastic users keep to an **allowance**, which says
//! how much each may hold now. With a fixed limit ([`PinnedRam::new`]) the
//! allowances are what the limit leaves: all of it to the tier of experts as
//! far as the tier can use it, the rest to the checkpoints. An engine that
//! follows the machine ([`PinnedRam::following`]) has its allowances set by
//! the [`Machine`], look by look, from the memory the machine really has; the
//! tier and the checkpoints then give memory back to the system when their
//! allowance falls and take it again when it rises (the tier through the hook
//! of [`PinnedRam::on_experts`], the checkpoints through
//! [`PinnedRam::on_checkpoints`] and [`PinnedRam::sweep_checkpoints`]).
//!
//! The books are shared by the executor and the device threads, so they sit
//! behind a lock; a grant is recorded in the same call that makes it, and a
//! chunk gives its bytes back when it is dropped.

mod machine;

use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ares_core::{Broker, Class, Consumer, Growth};
use ares_cuda::{Device, HostRegistration, PinnedBuffer};

pub use machine::{Follow, Look, Machine, Meminfo, MemorySource, Rule, Verdict, high_watermarks};

use crate::error::RuntimeError;
use crate::expert_store::{Locker, StoreError};
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
    /// The most the tier of experts can use, and the part of it the tier
    /// cannot work without (its staging slots and a few slots per layer).
    experts_want: u64,
    experts_floor: u64,
    /// The most the checkpoints can use: the conductor's budget. And the
    /// part of it the cache cannot work without: one place, when its level
    /// on SSD moves checkpoints through it.
    checkpoints_want: u64,
    checkpoints_floor: u64,
    /// What the machine lets the tier and the checkpoints hold above their
    /// floors; not used with a fixed limit.
    experts_allow: u64,
    checkpoints_allow: u64,
    /// A release that is being timed: when the machine asked for it, and
    /// what the elastic users hold once it is done.
    asked: Option<(std::time::Duration, u64)>,
}

impl Books {
    fn usage(&self, consumer: &Consumer) -> u64 {
        self.broker.info(consumer.id()).map_or(0, |info| info.usage)
    }

    /// What the limit leaves for the elastic users.
    fn elastic_room(&self) -> u64 {
        self.broker.capacity().saturating_sub(self.usage(&self.required))
    }
}

type Hook = Arc<dyn Fn(&PinnedRam, u64) + Send + Sync>;

#[derive(Default)]
struct Hooks {
    experts: Option<Hook>,
    checkpoints: Option<Hook>,
}

/// What an engine's elastic users hold and can use, for the machine.
pub(crate) struct Wants {
    /// Everything the engine has booked.
    pub locked: u64,
    /// The tier of experts, above its floor.
    pub experts_held: u64,
    pub experts_want: u64,
    /// The checkpoints, above their floor.
    pub checkpoints_held: u64,
    pub checkpoints_want: u64,
}

pub(crate) struct Shared {
    books: Mutex<Books>,
    machine: Option<Arc<Machine>>,
    hooks: Mutex<Hooks>,
    /// The checkpoint places the conductor has in use.
    live: Mutex<Vec<bool>>,
    /// The places of every device part.
    chunks: Mutex<Vec<Weak<PlaceChunks>>>,
}

impl Shared {
    fn books(&self) -> MutexGuard<'_, Books> {
        self.books.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the one place of the checkpoints' floor holds locked: the place,
    /// or, where the places are locked in chunks, the chunk it lies in on
    /// every device part (in a small model a chunk holds several places).
    /// Nothing without a floor.
    fn floor_held(&self) -> u64 {
        let floor = self.books().checkpoints_floor;
        if floor == 0 {
            return 0;
        }
        let tables = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        let chunks: u64 = tables
            .iter()
            .filter_map(Weak::upgrade)
            .map(|table| u64::from(table.per_chunk.min(table.capacity)) * table.bytes as u64)
            .sum();
        floor.max(chunks)
    }

    pub(crate) fn wants(&self) -> Wants {
        let floor_held = self.floor_held();
        let books = self.books();
        let floors = books.experts_floor + books.checkpoints_floor;
        let room = books.elastic_room().saturating_sub(floors);
        let experts_want = books.experts_want.saturating_sub(books.experts_floor).min(room);
        let checkpoints_want = books.checkpoints_want.saturating_sub(books.checkpoints_floor);
        Wants {
            locked: books.broker.used(),
            experts_held: books.usage(&books.experts).saturating_sub(books.experts_floor),
            experts_want,
            checkpoints_held: books.usage(&books.checkpoints).saturating_sub(floor_held),
            checkpoints_want: checkpoints_want.min(room - experts_want),
        }
    }

    /// Sets what the tier of experts and the checkpoints may hold above
    /// their floors. `holds` is what the two hold at `now` and
    /// `over` how much of it is more than they may: when a release that was
    /// asked for at an earlier look is done, the time it took comes back.
    pub(crate) fn allow(
        &self,
        experts: u64,
        checkpoints: u64,
        now: std::time::Duration,
        holds: u64,
        over: u64,
    ) -> Option<std::time::Duration> {
        let mut books = self.books();
        books.experts_allow = experts;
        books.checkpoints_allow = checkpoints;
        let mut took = None;
        if let Some((asked, level)) = books.asked
            && holds <= level
        {
            took = Some(now.saturating_sub(asked));
            books.asked = None;
        }
        if books.asked.is_none() && over > 0 {
            books.asked = Some((now, holds - over.min(holds)));
        }
        took
    }

    /// Lets the users act on their allowances: called after every look of
    /// the machine, from its thread, with none of its locks held.
    pub(crate) fn follow(self: &Arc<Self>) {
        let ram = PinnedRam { shared: self.clone() };
        let (experts, checkpoints) = {
            let hooks = self.hooks.lock().unwrap_or_else(|e| e.into_inner());
            (hooks.experts.clone(), hooks.checkpoints.clone())
        };
        if let Some(hook) = experts {
            hook(&ram, ram.experts_allowance());
        }
        // The chunk that the floor's place lies in is not more than allowed.
        let allowance = ram.checkpoints_allowance().max(self.floor_held());
        if ram.checkpoints() > allowance {
            if let Some(hook) = checkpoints {
                hook(&ram, allowance);
            }
            ram.sweep_checkpoints();
        }
    }
}

/// The page-locked RAM of an engine and what it holds. Clones share the books.
#[derive(Clone)]
pub struct PinnedRam {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for PinnedRam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let books = self.books();
        f.debug_struct("PinnedRam")
            .field("limit", &books.broker.capacity())
            .field("used", &books.broker.used())
            .field("follows", &self.shared.machine.is_some())
            .finish()
    }
}

impl PinnedRam {
    /// Books for at most `limit` bytes of page-locked RAM, all of which the
    /// engine may hold whatever the machine has.
    pub fn new(limit: u64) -> Self {
        Self::build(limit, None)
    }

    /// Books for at most `limit` bytes of page-locked RAM, of which the
    /// elastic users hold what `machine` lets them, look by look.
    pub fn following(limit: u64, machine: &Arc<Machine>) -> Self {
        let ram = Self::build(limit, Some(machine.clone()));
        machine.admit(&ram.shared);
        ram
    }

    fn build(limit: u64, machine: Option<Arc<Machine>>) -> Self {
        let mut broker = Broker::new("pinned RAM", limit);
        // Three consumers of a fresh broker with distinct names and no floor
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
        let books = Books {
            broker,
            required,
            checkpoints,
            experts,
            experts_want: 0,
            experts_floor: 0,
            checkpoints_want: 0,
            checkpoints_floor: 0,
            experts_allow: 0,
            checkpoints_allow: 0,
            asked: None,
        };
        Self {
            shared: Arc::new(Shared {
                books: Mutex::new(books),
                machine,
                hooks: Mutex::new(Hooks::default()),
                live: Mutex::new(Vec::new()),
                chunks: Mutex::new(Vec::new()),
            }),
        }
    }

    fn books(&self) -> MutexGuard<'_, Books> {
        self.shared.books()
    }

    /// The watch over the machine this engine follows, if it follows one.
    pub fn machine(&self) -> Option<&Arc<Machine>> {
        self.shared.machine.as_ref()
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

    /// What the tier of experts may hold now: with a fixed limit all that
    /// the limit leaves after the required memory and the floor of the
    /// checkpoints, otherwise what the machine lets it have, and always the
    /// part it cannot work without.
    pub fn experts_allowance(&self) -> u64 {
        let books = self.books();
        let room = books.elastic_room().saturating_sub(books.checkpoints_floor);
        match self.shared.machine {
            None => room,
            Some(_) => (books.experts_floor + books.experts_allow).min(room),
        }
    }

    /// What the checkpoints may hold now: with a fixed limit what the limit
    /// leaves after the required memory and all the tier of experts can use,
    /// otherwise what the machine lets them have, and always their floor
    /// ([`PinnedRam::want_checkpoints`]).
    pub fn checkpoints_allowance(&self) -> u64 {
        let books = self.books();
        match self.shared.machine {
            None => {
                let experts = books.experts_want.max(books.usage(&books.experts));
                books.elastic_room().saturating_sub(experts).max(books.checkpoints_floor)
            }
            Some(_) => books.checkpoints_floor + books.checkpoints_allow,
        }
    }

    /// Says what the tier of experts can use at most and what it cannot
    /// work without, both in bytes. With a machine to follow, what is free
    /// now is the tier's to take at once.
    pub fn want_experts(&self, want: u64, floor: u64) {
        {
            let mut books = self.books();
            books.experts_want = want;
            books.experts_floor = floor.min(want);
        }
        if let Some(machine) = &self.shared.machine {
            machine.claim();
        }
    }

    /// The tier of experts cannot go below `floor` bytes: it found that out
    /// when it gave up all it could.
    pub fn experts_floor(&self, floor: u64) {
        let mut books = self.books();
        books.experts_floor = floor.min(books.experts_want);
    }

    /// Says what the checkpoints can use at most, the conductor's budget,
    /// and what the cache cannot work without, both in bytes. The floor is
    /// one place for a cache whose level on SSD moves checkpoints through it
    /// on their way to the drive and back, and nothing otherwise; it is the
    /// cache's whatever the machine has, as the tier's floor is the tier's.
    pub fn want_checkpoints(&self, want: u64, floor: u64) {
        {
            let mut books = self.books();
            books.checkpoints_want = want;
            books.checkpoints_floor = floor.min(want);
        }
        if let Some(machine) = &self.shared.machine {
            machine.claim();
        }
    }

    /// `hook` is called with these books and the tier's allowance after
    /// every look of the machine, from the machine's thread: the tier
    /// releases pieces or locks more to follow it. The hook is kept by the
    /// books, so it must not keep them: it is handed them.
    pub fn on_experts(&self, hook: impl Fn(&PinnedRam, u64) + Send + Sync + 'static) {
        self.shared.hooks.lock().unwrap_or_else(|e| e.into_inner()).experts = Some(Arc::new(hook));
    }

    /// `hook` is called with these books and the checkpoints' allowance
    /// when they hold more than it, from the machine's thread: the conductor
    /// gives checkpoints up, and their chunks are released after it
    /// ([`PinnedRam::sweep_checkpoints`]).
    pub fn on_checkpoints(&self, hook: impl Fn(&PinnedRam, u64) + Send + Sync + 'static) {
        self.shared.hooks.lock().unwrap_or_else(|e| e.into_inner()).checkpoints =
            Some(Arc::new(hook));
    }

    /// What locks the pieces of the tier of experts: each piece is booked
    /// here, locked for `device`, and gives its booking back when its lock
    /// is dropped.
    pub fn expert_pieces(&self, device: &Device) -> Box<dyn Locker> {
        Box::new(ExpertPieces { ram: self.clone(), device: device.clone() })
    }

    fn book_experts(&self, bytes: u64) -> Result<(), String> {
        let mut books = self.books();
        let Books { broker, experts, .. } = &mut *books;
        match broker.grow(experts, bytes) {
            Ok(Growth::Granted) => Ok(()),
            Ok(Growth::Reclaim(plan)) => Err(format!(
                "the RAM limit of {} bytes has no room for {bytes} bytes for the tier of experts: \
                 {} would have to come from others",
                broker.capacity(),
                plan.total()
            )),
            Err(error) => Err(error.to_string()),
        }
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

    pub(crate) fn book_checkpoints(&self, bytes: u64) -> Result<(), ModelError> {
        let mut books = self.books();
        let Books { broker, checkpoints, .. } = &mut *books;
        match broker.grow(checkpoints, bytes) {
            Ok(Growth::Granted) => Ok(()),
            // Nothing ranks below the checkpoints, so there is nobody to
            // take from; a plan would be a refusal.
            Ok(Growth::Reclaim(plan)) => Err(ModelError::Invalid(format!(
                "the RAM limit of {} bytes has no room for {bytes} more bytes of checkpoints, and \
                 {} would have to come from others",
                broker.capacity(),
                plan.total()
            ))),
            Err(error) => Err(ModelError::Invalid(error.to_string())),
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
        self.book_checkpoints(bytes as u64)?;
        match PinnedBuffer::new(device, bytes) {
            Ok(buffer) => Ok(LockedChunk { buffer, ram: self.clone() }),
            Err(error) => {
                self.release_checkpoints(bytes as u64);
                Err(error.into())
            }
        }
    }

    pub(crate) fn release_checkpoints(&self, bytes: u64) {
        let mut books = self.books();
        let Books { broker, checkpoints, .. } = &mut *books;
        // The bytes were granted by `book_checkpoints`, so the books hold them.
        let _ = broker.shrink(checkpoints, bytes);
    }

    /// The places of a device part's checkpoints: `capacity` places of
    /// `bytes` each, locked on `device` in chunks of at least `chunk` bytes
    /// as they come into use, and released again when the checkpoints must
    /// give memory back and the conductor has none in a chunk.
    pub fn place_chunks(
        &self,
        device: &Device,
        bytes: u64,
        capacity: u32,
        chunk: usize,
    ) -> Arc<PlaceChunks> {
        let bytes = bytes as usize;
        let per_chunk = chunk.checked_div(bytes).unwrap_or(1).max(1) as u32;
        let chunks = Arc::new(PlaceChunks {
            ram: self.clone(),
            device: device.clone(),
            bytes,
            capacity,
            per_chunk,
            chunks: Mutex::new(Vec::new()),
        });
        let mut all = self.shared.chunks.lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|chunks| chunks.strong_count() > 0);
        all.push(Arc::downgrade(&chunks));
        chunks
    }

    /// The conductor says whether checkpoint place `place` is in use. A
    /// place that is not may lose its memory; one that is keeps it. The
    /// conductor calls this between steps only, so that no copy of a step
    /// that is running reads or writes a place it calls unused.
    pub fn place_live(&self, place: u32, live: bool) {
        let mut places = self.shared.live.lock().unwrap_or_else(|e| e.into_inner());
        if places.len() <= place as usize {
            places.resize(place as usize + 1, false);
        }
        places[place as usize] = live;
    }

    /// Whether the conductor has checkpoint place `place` in use.
    pub(crate) fn place_is_live(&self, place: u32) -> bool {
        let places = self.shared.live.lock().unwrap_or_else(|e| e.into_inner());
        places.get(place as usize).copied().unwrap_or(false)
    }

    /// Releases chunks of checkpoint places in which the conductor has no
    /// checkpoint, until the checkpoints hold no more than their allowance.
    /// Returns the bytes released.
    pub fn sweep_checkpoints(&self) -> u64 {
        let allowance = self.checkpoints_allowance().max(self.shared.floor_held());
        let mut held = self.checkpoints();
        let tables: Vec<Arc<PlaceChunks>> = {
            let all = self.shared.chunks.lock().unwrap_or_else(|e| e.into_inner());
            all.iter().filter_map(Weak::upgrade).collect()
        };
        let mut freed = Vec::new();
        for table in &tables {
            let mut chunks = table.chunks.lock().unwrap_or_else(|e| e.into_inner());
            let live = self.shared.live.lock().unwrap_or_else(|e| e.into_inner());
            for index in (0..chunks.len()).rev() {
                if held <= allowance {
                    break;
                }
                let first = index * table.per_chunk as usize;
                let in_use =
                    live.iter().skip(first).take(table.per_chunk as usize).any(|used| *used);
                if in_use {
                    continue;
                }
                if let Some(chunk) = chunks[index].take() {
                    held = held.saturating_sub(chunk.len() as u64);
                    freed.push(chunk);
                }
            }
        }
        // Unlocking takes time and books the bytes back: outside the locks.
        freed.iter().map(|chunk| chunk.len() as u64).sum()
    }
}

/// Locks pieces of the tier of experts for a device and books them.
struct ExpertPieces {
    ram: PinnedRam,
    device: Device,
}

impl Locker for ExpertPieces {
    unsafe fn lock(&mut self, ptr: *mut u8, len: usize) -> Result<Box<dyn Any + Send>, StoreError> {
        self.ram.book_experts(len as u64).map_err(StoreError::Plan)?;
        // Dropped before the registration fails or after it is dropped.
        let booking = ExpertsBooking { ram: self.ram.clone(), bytes: len as u64 };
        // SAFETY: the caller's contract is that of the registration.
        let registration = unsafe { HostRegistration::new(&self.device, ptr, len, false)? };
        Ok(Box::new((registration, booking)))
    }
}

/// What a piece of the tier of experts gives back to the books when it is
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

/// The places of one device part's checkpoints on the host. See
/// [`PinnedRam::place_chunks`].
#[derive(Debug)]
pub struct PlaceChunks {
    ram: PinnedRam,
    device: Device,
    /// The bytes of one place.
    bytes: usize,
    capacity: u32,
    per_chunk: u32,
    chunks: Mutex<Vec<Option<LockedChunk>>>,
}

impl PlaceChunks {
    /// The bytes of one place.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Where place `place` is, and whether its chunk had to be locked for
    /// this call. The memory stays where it is while the conductor has the
    /// place in use ([`PinnedRam::place_live`]).
    pub fn pointer(&self, place: u32) -> Result<(*mut u8, bool), ModelError> {
        if place >= self.capacity {
            return Err(ModelError::Invalid(format!(
                "checkpoint place {place} of {}: the conductor names a place this part does not \
                 have",
                self.capacity
            )));
        }
        let index = (place / self.per_chunk) as usize;
        let mut chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        if chunks.len() <= index {
            chunks.resize_with(index + 1, || None);
        }
        let fresh = chunks[index].is_none();
        if fresh {
            let first = index as u32 * self.per_chunk;
            let places = self.per_chunk.min(self.capacity - first);
            chunks[index] =
                Some(self.ram.lock_checkpoints(&self.device, places as usize * self.bytes)?);
        }
        let chunk = chunks[index].as_mut().expect("locked above");
        let within = (place % self.per_chunk) as usize * self.bytes;
        // SAFETY: the place is inside its chunk, which holds `per_chunk`
        // places (the last one the rest).
        Ok((unsafe { chunk.as_mut_ptr().add(within) }, fresh))
    }

    /// Where place `place` is when its chunk is locked, and `None` when the
    /// chunk was never locked or was given back: the place holds nothing
    /// then. For a reader that must not take new memory for what is gone,
    /// such as the copy of a checkpoint that leaves for the drive.
    pub fn held(&self, place: u32) -> Option<*mut u8> {
        if place >= self.capacity {
            return None;
        }
        let mut chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        let chunk = chunks.get_mut((place / self.per_chunk) as usize)?.as_mut()?;
        let within = (place % self.per_chunk) as usize * self.bytes;
        // SAFETY: as in `pointer`.
        Some(unsafe { chunk.as_mut_ptr().add(within) })
    }

    /// Bytes of host memory locked for the places now.
    pub fn locked(&self) -> u64 {
        let chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        chunks.iter().flatten().map(|chunk| chunk.len() as u64).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use ares_core::units::{GIB, MIB};

    use super::*;

    #[test]
    fn the_tier_of_experts_may_take_all_the_limit_leaves_and_the_checkpoints_give_way_to_it() {
        let ram = PinnedRam::new(100 * MIB);
        ram.require(10 * MIB).unwrap();
        assert_eq!((ram.obtainable_experts(), ram.experts()), (90 * MIB, 0));
        // A model without experts leaves all of it to the checkpoints.
        assert_eq!((ram.experts_allowance(), ram.checkpoints_allowance()), (90 * MIB, 90 * MIB));
        // The checkpoints hold 40 MiB; the tier of experts, which ranks above
        // them, could still have all 90.
        ram.book_checkpoints(40 * MIB).unwrap();
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
        }
        // A tier that can use 70 MiB leaves the checkpoints 20, and one that
        // can use everything leaves them nothing.
        ram.want_experts(70 * MIB, 0);
        assert_eq!((ram.experts_allowance(), ram.checkpoints_allowance()), (90 * MIB, 20 * MIB));
        ram.want_experts(GIB, 0);
        assert_eq!(ram.checkpoints_allowance(), 0);
        // And the checkpoints cannot take memory back from the tier.
        ram.release_checkpoints(40 * MIB);
        ram.book_experts(90 * MIB).unwrap();
        assert!(ram.book_checkpoints(20 * MIB).is_err());
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

    const PIECE: u64 = 64 * MIB;

    /// A machine of `total` bytes on which other programs hold `others`:
    /// what is available is what neither they nor the engines hold.
    struct Sim {
        total: u64,
        others: AtomicU64,
        engines: Mutex<Vec<PinnedRam>>,
        ms: AtomicU64,
    }

    impl Sim {
        fn new(total: u64, others: u64) -> (Arc<Self>, Arc<Machine>) {
            let sim = Arc::new(Self {
                total,
                others: AtomicU64::new(others),
                engines: Mutex::new(Vec::new()),
                ms: AtomicU64::new(0),
            });
            let rule = Rule {
                reserve: 128 * MIB,
                speed: GIB,
                latency: Duration::from_millis(50),
                piece: PIECE,
                ..Rule::default()
            };
            let (memory, clock) = (sim.clone(), sim.clone());
            let machine = Machine::new(
                rule,
                move || Some(memory.available()),
                move || Duration::from_millis(clock.ms.load(Ordering::SeqCst)),
            );
            (sim, machine)
        }

        fn available(&self) -> u64 {
            let engines: u64 = self.engines.lock().unwrap().iter().map(PinnedRam::used).sum();
            self.total.saturating_sub(self.others.load(Ordering::SeqCst) + engines)
        }

        /// An engine that follows `machine` below `limit` and needs
        /// `required` bytes to run.
        fn engine(&self, machine: &Arc<Machine>, limit: u64, required: u64) -> PinnedRam {
            let ram = PinnedRam::following(limit, machine);
            ram.require(required).unwrap();
            self.engines.lock().unwrap().push(ram.clone());
            ram
        }

        /// Looks at the machine every 50 ms for `ms` milliseconds.
        fn run(&self, machine: &Machine, ms: u64) {
            for _ in 0..ms / 50 {
                self.ms.fetch_add(50, Ordering::SeqCst);
                machine.look().expect("the machine says what it has");
            }
        }
    }

    /// A tier of experts that can use `want` bytes and follows its
    /// allowance in pieces, as the store does.
    fn tier(ram: &PinnedRam, want: u64) {
        ram.on_experts(move |ram, allowance| {
            while ram.experts() > allowance {
                ram.release_experts(PIECE.min(ram.experts()));
            }
            while ram.experts() + PIECE <= allowance.min(want) {
                ram.book_experts(PIECE).unwrap();
            }
        });
        ram.want_experts(want, 0);
    }

    /// A prompt cache with a budget of `want` bytes that keeps as many
    /// checkpoints of 64 MiB as it is allowed.
    fn cache(ram: &PinnedRam, want: u64) {
        ram.on_checkpoints(|ram, allowance| {
            while ram.checkpoints() > allowance {
                ram.release_checkpoints(PIECE.min(ram.checkpoints()));
            }
        });
        ram.want_checkpoints(want, 0);
    }

    /// The cache takes checkpoints as far as it may.
    fn fill(ram: &PinnedRam) {
        while ram.checkpoints() + PIECE <= ram.checkpoints_allowance() {
            ram.book_checkpoints(PIECE).unwrap();
        }
    }

    fn required(ram: &PinnedRam) -> u64 {
        ram.used() - ram.experts() - ram.checkpoints()
    }

    #[test]
    fn a_tier_takes_what_the_machine_has_less_the_headroom_and_never_more_than_the_limit() {
        // 32 GiB, 10 of them taken by others; the engine needs 1 GiB to run.
        let (sim, machine) = Sim::new(32 * GIB, 10 * GIB);
        let ram = sim.engine(&machine, 64 * GIB, GIB);
        tier(&ram, 40 * GIB);
        // What is free is taken at once, at the load.
        let headroom = machine.headroom();
        assert!(ram.experts() > 20 * GIB, "{}", ram.experts());
        assert!(sim.available() >= headroom, "{} {headroom}", sim.available());
        assert!(sim.available() < headroom + 4 * PIECE, "{} {headroom}", sim.available());
        // And it stays there while nothing changes.
        let held = ram.experts();
        sim.run(&machine, 10_000);
        assert_eq!(ram.experts(), held);

        // The same machine and a limit of 8 GiB: the limit holds.
        let (sim, machine) = Sim::new(32 * GIB, 10 * GIB);
        let ram = sim.engine(&machine, 8 * GIB, GIB);
        tier(&ram, 40 * GIB);
        sim.run(&machine, 10_000);
        assert_eq!((ram.experts(), ram.used()), (7 * GIB, 8 * GIB));
    }

    #[test]
    fn memory_goes_back_when_another_program_grows_and_is_taken_again_after_a_wait() {
        let (sim, machine) = Sim::new(32 * GIB, 10 * GIB);
        let ram = sim.engine(&machine, 64 * GIB, GIB);
        tier(&ram, 40 * GIB);
        let before = ram.experts();
        // Another program grows by 3 GiB over a second.
        for _ in 0..20 {
            sim.others.fetch_add(3 * GIB / 20, Ordering::SeqCst);
            sim.run(&machine, 50);
        }
        sim.run(&machine, 500);
        let headroom = machine.headroom();
        assert!(sim.available() >= headroom, "{} {headroom}", sim.available());
        let given = before - ram.experts();
        assert!((3 * GIB..3 * GIB + GIB / 2).contains(&given), "{given}");
        assert_eq!(required(&ram), GIB, "what the engine needs to run is never given");

        // The program ends. Nothing is taken for three seconds, then all of it.
        sim.others.fetch_sub(3 * GIB, Ordering::SeqCst);
        let low = ram.experts();
        sim.run(&machine, 2_900);
        assert_eq!(ram.experts(), low, "the memory has not been free for long enough");
        sim.run(&machine, 300);
        assert!(ram.experts() >= low + 3 * GIB - 4 * PIECE, "{} {low}", ram.experts());
        assert!(sim.available() >= machine.headroom());

        // It allocates again ten seconds later: the engine took too early,
        // so the next time it waits twice as long.
        sim.run(&machine, 10_000);
        sim.others.fetch_add(GIB, Ordering::SeqCst);
        sim.run(&machine, 1_000);
        assert_eq!(machine.seen().2, Duration::from_secs(6));
        sim.others.fetch_sub(GIB, Ordering::SeqCst);
        let low = ram.experts();
        sim.run(&machine, 5_900);
        assert_eq!(ram.experts(), low);
        sim.run(&machine, 300);
        assert!(ram.experts() > low);
    }

    #[test]
    fn the_checkpoints_give_way_first_and_the_tier_only_when_they_have_nothing_left() {
        let (sim, machine) = Sim::new(32 * GIB, 16 * GIB);
        let ram = sim.engine(&machine, 64 * GIB, GIB);
        // The tier can use 8 GiB and gets them; the cache has the rest.
        tier(&ram, 8 * GIB);
        cache(&ram, 32 * GIB);
        fill(&ram);
        sim.run(&machine, 5_000);
        fill(&ram);
        let cached = ram.checkpoints();
        assert_eq!(ram.experts(), 8 * GIB);
        assert!(cached > 6 * GIB && sim.available() >= machine.headroom(), "{cached}");

        // 2 GiB go to another program: checkpoints go, no expert does.
        for _ in 0..40 {
            sim.others.fetch_add(2 * GIB / 40, Ordering::SeqCst);
            sim.run(&machine, 50);
        }
        sim.run(&machine, 500);
        assert_eq!(ram.experts(), 8 * GIB);
        let given = cached - ram.checkpoints();
        assert!((2 * GIB - 4 * PIECE..3 * GIB).contains(&given), "{given}");
        assert!(sim.available() >= machine.headroom());

        // 8 GiB more: the cache is empty before the tier gives anything.
        for _ in 0..160 {
            sim.others.fetch_add(8 * GIB / 160, Ordering::SeqCst);
            sim.run(&machine, 50);
        }
        sim.run(&machine, 500);
        assert_eq!(ram.checkpoints(), 0);
        assert_eq!(ram.checkpoints_allowance(), 0, "and it may take no new checkpoint");
        assert!(ram.experts() < 6 * GIB && ram.experts() > 3 * GIB, "{}", ram.experts());
        assert!(sim.available() >= machine.headroom());
        assert_eq!(required(&ram), GIB);

        // When memory is free again the tier is filled first.
        sim.others.fetch_sub(4 * GIB, Ordering::SeqCst);
        sim.run(&machine, 400_000);
        assert_eq!(ram.experts(), 8 * GIB);
        fill(&ram);
        assert!(ram.checkpoints() > 0);
    }

    #[test]
    fn the_floor_of_the_checkpoints_is_theirs_whatever_the_machine_and_the_tier_take() {
        // A fixed limit: the tier of experts may take all the limit leaves
        // but the one place the cache cannot work without.
        let ram = PinnedRam::new(100 * MIB);
        ram.require(10 * MIB).unwrap();
        ram.want_checkpoints(40 * MIB, 8 * MIB);
        ram.want_experts(GIB, 0);
        assert_eq!((ram.experts_allowance(), ram.checkpoints_allowance()), (82 * MIB, 8 * MIB));
        // A floor is never more than what the cache can use at all.
        ram.want_checkpoints(4 * MIB, 8 * MIB);
        assert_eq!(ram.checkpoints_allowance(), 4 * MIB);

        // A machine to follow, on which other programs leave nothing: the
        // checkpoints go down to their floor and no further, and the machine
        // asks for nothing it cannot have: no release is owed.
        let (sim, machine) = Sim::new(32 * GIB, 16 * GIB);
        let ram = sim.engine(&machine, 64 * GIB, GIB);
        tier(&ram, 8 * GIB);
        cache(&ram, 32 * GIB);
        ram.want_checkpoints(32 * GIB, PIECE);
        fill(&ram);
        sim.run(&machine, 5_000);
        fill(&ram);
        assert!(ram.checkpoints() > 6 * GIB);
        for _ in 0..200 {
            sim.others.fetch_add(10 * GIB / 200, Ordering::SeqCst);
            sim.run(&machine, 50);
        }
        sim.run(&machine, 500);
        assert_eq!((ram.checkpoints_allowance(), ram.checkpoints()), (PIECE, PIECE));
        assert!(ram.experts() < 6 * GIB && ram.experts() > 3 * GIB, "{}", ram.experts());
        let look = machine.look().unwrap();
        assert_eq!(look.owed, 0, "the floor is not memory the machine waits for: {look:?}");
        assert_eq!(required(&ram), GIB);
        // When memory is free again the tier is filled first, as without a
        // floor, and the checkpoints take what is left.
        sim.others.fetch_sub(4 * GIB, Ordering::SeqCst);
        sim.run(&machine, 400_000);
        assert_eq!(ram.experts(), 8 * GIB);
        fill(&ram);
        assert!(ram.checkpoints() > PIECE);
    }

    #[test]
    fn two_engines_share_what_the_machine_has() {
        let (sim, machine) = Sim::new(32 * GIB, 10 * GIB);
        let first = sim.engine(&machine, 64 * GIB, GIB);
        tier(&first, 40 * GIB);
        let alone = first.experts();
        assert!(alone > 20 * GIB);
        // A second model is loaded: what it needs to run comes out of the
        // first one's tier, and the tiers end with the same.
        let second = sim.engine(&machine, 64 * GIB, 128 * MIB);
        tier(&second, 40 * GIB);
        sim.run(&machine, 2_000);
        let (a, b) = (first.experts(), second.experts());
        assert!(a.abs_diff(b) <= 2 * PIECE, "{a} {b}");
        assert!(a + b <= alone && a + b + 8 * PIECE >= alone, "{a} {b} {alone}");
        assert!(sim.available() >= machine.headroom());
        // The second is unloaded: its memory is the first one's to take.
        sim.engines.lock().unwrap().pop();
        drop(second);
        sim.run(&machine, 5_000);
        assert!(first.experts() + 4 * PIECE >= alone, "{} {alone}", first.experts());
    }
}
