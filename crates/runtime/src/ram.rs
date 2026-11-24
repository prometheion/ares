//! The engine's page-locked host memory, kept in one set of books.
//!
//! The engine has one limit of host RAM (see "Host memory" in
//! `docs/ARCHITECTURE.md`), and the users of that memory make room for each
//! other by rank. Page-locked memory is what the devices copy to and from
//! without a bounce buffer, so it is where state that lives in RAM has to be;
//! today there are four users:
//!
//! - the step record and the read-back buffers, which the executor needs to
//!   run at all (required), and whatever else a model's parts must have
//!   ([`crate::Model::pinned_ram`]);
//! - the tier of experts in host memory (elastic, the highest rank: it is
//!   what keeps a mixture-of-experts model from reading its experts from
//!   disk, and it gives way to nothing but what is required). It is locked in
//!   pieces ([`PinnedRam::expert_pieces`]), as many as its allowance holds;
//! - the K/V of the context beyond VRAM (required: a running sequence cannot
//!   do without its cells), in a range of address space per device part that
//!   is locked and mapped for the device in pieces as its pages come into use
//!   ([`PinnedRam::context_region`]). It takes its memory from the elastic
//!   users as it grows and gives it back as its pages are freed;
//! - the prompt cache in RAM (elastic, the lowest rank: a cache gives way to
//!   whatever needs the memory): the checkpoints of recurrent state, which a
//!   device part locks in chunks when a place is first used, so the memory
//!   the cache holds is the memory it has used and not the whole budget
//!   ([`PlaceChunks`]), and the pages that left the devices, a buffer per
//!   device share of a page ([`PinnedRam::cache_buffer`]). A cache keeps one
//!   place (for a model with recurrent state) whatever the machine has, and
//!   with a level on SSD a few pages as well: state passes through them on
//!   its way to the drive and back. That is the cache's floor
//!   ([`PinnedRam::want_cache`]), and neither the tier of experts nor the
//!   machine is given it.
//!
//! # The limit and the allowances
//!
//! The limit is the most the engine may ever lock: a booking beyond it is
//! refused, and the refusal says who holds what. Below it the elastic users
//! keep to an **allowance**, which says how much each may hold now. The rank
//! decides it: the required users take what they need, the tier of experts
//! may hold all the limit leaves but the prompt cache's floor, and the prompt
//! cache what the tier does not want, at least its floor. An engine that
//! follows the machine ([`PinnedRam::following`]) is also held to what the
//! [`Machine`] lets each have, look by look, from the memory the machine
//! really has (the machine shares it out by the same rank, the tiers of all
//! engines first); the tier and the prompt cache then give memory back to the
//! system when their allowance falls and take it again when it rises (the
//! tier through the hook of [`PinnedRam::on_experts`], the prompt cache
//! through [`PinnedRam::on_cache`] and [`PinnedRam::sweep_checkpoints`]).
//! With a fixed limit ([`PinnedRam::new`]) nothing else bounds them.
//!
//! The books are shared by the executor and the device threads, so they sit
//! behind a lock; a booking is recorded in the same call that grants it, and
//! a chunk gives its bytes back when it is dropped.

mod machine;

use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ares_core::units::Bytes;
use ares_cuda::{Device, DevicePtr, HostRegistration, PinnedBuffer};
use ares_store::Aligned;

pub use machine::{Look, Machine, Meminfo, MemorySource, Rule, Verdict};

use crate::error::RuntimeError;
use crate::expert_store::{Locker, StoreError};
use crate::model::ModelError;

/// Who holds the engine's page-locked memory, by rank: what the executor
/// cannot run without and the K/V of running sequences beyond VRAM first,
/// then the tier of experts (a step that misses an expert waits for the
/// drive, which is the one thing that makes such a model slow), then the
/// prompt cache (a prompt that finds less of itself there computes more rows,
/// and nothing else suffers).
struct Books {
    /// The most the engine may lock.
    limit: u64,
    /// The step record and the read-back buffers.
    required: u64,
    /// The K/V of the context beyond VRAM, and what it will hold once the
    /// pieces of the step that begins are locked: the elastic users make room
    /// for that much before the pieces are booked. Of it, `context_cached`
    /// are pages that only the prompt cache holds: they rank as prompt cache
    /// (the owner's rule of 2026-11-22), so they count in the prompt cache's
    /// share and not against the tier of experts.
    context: u64,
    context_target: u64,
    context_cached: u64,
    /// The prompt cache in RAM: checkpoints of recurrent state and pages.
    cache: u64,
    /// The bytes of `cache` that are pages; the rest are checkpoints.
    pages: u64,
    /// The tier of experts.
    experts: u64,
    /// The most the tier of experts can use, and the part of it the tier
    /// cannot work without (its staging slots and a few slots per layer).
    experts_want: u64,
    experts_floor: u64,
    /// The most the prompt cache can use: the conductor's budget. And the
    /// part of it the cache keeps whatever the machine has: one checkpoint
    /// place for a model with recurrent state, and a few pages when its level
    /// on SSD moves state through RAM.
    cache_want: u64,
    place_floor: u64,
    pages_floor: u64,
    /// What the machine lets the tier and the prompt cache hold above their
    /// floors; no bound with a fixed limit.
    experts_allow: u64,
    cache_allow: u64,
    /// A release that is being timed: when the machine asked for it, and
    /// what the elastic users hold once it is done.
    asked: Option<(std::time::Duration, u64)>,
}

/// A user of the books.
#[derive(Clone, Copy, Debug)]
enum User {
    Required,
    Context,
    Cache,
    Experts,
}

impl Books {
    fn used(&self) -> u64 {
        self.required + self.context + self.cache + self.experts
    }

    fn held(&mut self, user: User) -> &mut u64 {
        match user {
            User::Required => &mut self.required,
            User::Context => &mut self.context,
            User::Cache => &mut self.cache,
            User::Experts => &mut self.experts,
        }
    }

    /// Books `bytes` for `user` when the limit has room for them, or says
    /// how many bytes it lacks.
    fn grow(&mut self, user: User, bytes: u64) -> Result<(), u64> {
        let missing = self.used().saturating_add(bytes).saturating_sub(self.limit);
        if missing > 0 {
            return Err(missing);
        }
        *self.held(user) += bytes;
        Ok(())
    }

    /// `user` gave `bytes` back.
    fn shrink(&mut self, user: User, bytes: u64) {
        let held = self.held(user);
        *held = held.saturating_sub(bytes);
    }

    /// Why `bytes` more for `what` do not fit: who holds the limit.
    fn refusal(&self, bytes: u64, what: &str) -> String {
        format!(
            "the RAM limit of {} has no room for {} more of {what}: the step record and \
             read-back hold {}, the K/V beyond VRAM {}, the tier of experts {}, the prompt \
             cache {}",
            Bytes(self.limit),
            Bytes(bytes),
            Bytes(self.required),
            Bytes(self.context),
            Bytes(self.experts),
            Bytes(self.cache),
        )
    }

    /// What the limit leaves for the elastic users: the K/V of running
    /// sequences beyond VRAM is not theirs; that of pages only the prompt
    /// cache holds is the prompt cache's.
    fn elastic_room(&self) -> u64 {
        let context = self.context.max(self.context_target);
        let running = context.saturating_sub(self.context_cached);
        self.limit.saturating_sub(self.required + running)
    }

    /// What the prompt cache holds: its own bookings and the K/V beyond
    /// VRAM of the pages only it holds.
    fn cache_usage(&self) -> u64 {
        self.cache + self.context_cached
    }

    /// The prompt cache's floor, never more than it can use.
    fn cache_floor(&self) -> u64 {
        (self.place_floor + self.pages_floor).min(self.cache_want)
    }

    /// What the tier of experts and the prompt cache may hold now, by rank:
    /// the tier all the room the limit leaves the two but the prompt cache's
    /// floor, the prompt cache what the tier does not want (or holds), at
    /// least its floor; each no more than the machine lets it hold above its
    /// floor.
    fn allowances(&self) -> (u64, u64) {
        let room = self.elastic_room();
        let cache_floor = self.cache_floor();
        let experts = self
            .experts_floor
            .saturating_add(self.experts_allow)
            .min(room.saturating_sub(cache_floor));
        let tier = self.experts_want.max(self.experts);
        let left = room.saturating_sub(tier).saturating_sub(cache_floor);
        (experts, cache_floor + self.cache_allow.min(left))
    }

    /// What the tier of experts and the prompt cache can use above their
    /// floors within the room the limit leaves them, by the same rank: what
    /// the machine shares out.
    fn wants(&self) -> (u64, u64) {
        let floors = self.experts_floor + self.cache_floor();
        let room = self.elastic_room().saturating_sub(floors);
        let experts = self.experts_want.saturating_sub(self.experts_floor).min(room);
        (experts, self.cache_want.saturating_sub(self.cache_floor()).min(room - experts))
    }
}

/// What the tier of experts is called with: the books and its allowance.
type ExpertsHook = Arc<dyn Fn(&PinnedRam, u64) + Send + Sync>;
/// What wakes the prompt cache to give memory back.
type CacheHook = Arc<dyn Fn(&PinnedRam) + Send + Sync>;

#[derive(Default)]
struct Hooks {
    experts: Option<ExpertsHook>,
    cache: Option<CacheHook>,
}

/// What an engine's elastic users hold and can use, for the machine.
pub(crate) struct Wants {
    /// Everything the engine has booked.
    pub locked: u64,
    /// The tier of experts, above its floor.
    pub experts_held: u64,
    pub experts_want: u64,
    /// The prompt cache, above its floor.
    pub cache_held: u64,
    pub cache_want: u64,
}

pub(crate) struct Shared {
    books: Mutex<Books>,
    machine: Option<Arc<Machine>>,
    hooks: Mutex<Hooks>,
    /// The checkpoint places the conductor has in use.
    live: Mutex<Vec<bool>>,
    /// The places of every device part.
    chunks: Mutex<Vec<Weak<PlaceChunks>>>,
    /// The devices that the pages of the prompt cache are locked for, one
    /// per device part; none for an executor without devices, whose pages
    /// are ordinary memory, booked all the same.
    devices: Mutex<Vec<Device>>,
    /// For an executor without devices that books its places itself (the
    /// host executor): the bytes of a place, and the places that hold
    /// memory now.
    host_places: Mutex<Option<(u64, Vec<bool>)>>,
}

impl Shared {
    fn books(&self) -> MutexGuard<'_, Books> {
        self.books.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn wants(&self) -> Wants {
        let books = self.books();
        let (experts_want, cache_want) = books.wants();
        Wants {
            locked: books.used(),
            experts_held: books.experts.saturating_sub(books.experts_floor),
            experts_want,
            cache_held: books.cache_usage().saturating_sub(books.cache_floor()),
            cache_want,
        }
    }

    /// Sets what the tier of experts and the prompt cache may hold above
    /// their floors. `holds` is what the two hold at `now` and
    /// `over` how much of it is more than they may: when a release that was
    /// asked for at an earlier look is done, the time it took comes back.
    pub(crate) fn allow(
        &self,
        experts: u64,
        cache: u64,
        now: std::time::Duration,
        holds: u64,
        over: u64,
    ) -> Option<std::time::Duration> {
        let mut books = self.books();
        books.experts_allow = experts;
        books.cache_allow = cache;
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
        let (experts, cache) = {
            let hooks = self.hooks.lock().unwrap_or_else(|e| e.into_inner());
            (hooks.experts.clone(), hooks.cache.clone())
        };
        if let Some(hook) = experts {
            hook(&ram, ram.experts_allowance());
        }
        let (held, allowance) = {
            let books = self.books();
            (books.cache_usage(), books.allowances().1)
        };
        if held > allowance {
            if let Some(hook) = cache {
                hook(&ram);
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
            .field("limit", &books.limit)
            .field("used", &books.used())
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
        // Without a machine nothing but the limit bounds the elastic users;
        // with one they hold nothing until its first look.
        let allow = if machine.is_some() { 0 } else { u64::MAX };
        let books = Books {
            limit,
            required: 0,
            context: 0,
            context_target: 0,
            context_cached: 0,
            cache: 0,
            pages: 0,
            experts: 0,
            experts_want: 0,
            experts_floor: 0,
            cache_want: 0,
            place_floor: 0,
            pages_floor: 0,
            experts_allow: allow,
            cache_allow: allow,
            asked: None,
        };
        Self {
            shared: Arc::new(Shared {
                books: Mutex::new(books),
                machine,
                hooks: Mutex::new(Hooks::default()),
                live: Mutex::new(Vec::new()),
                chunks: Mutex::new(Vec::new()),
                devices: Mutex::new(Vec::new()),
                host_places: Mutex::new(None),
            }),
        }
    }

    /// The pages of the prompt cache are locked for `devices`, one per
    /// device part, from now on.
    pub fn lock_for(&self, devices: Vec<Device>) {
        *self.shared.devices.lock().unwrap_or_else(|e| e.into_inner()) = devices;
    }

    fn books(&self) -> MutexGuard<'_, Books> {
        self.shared.books()
    }

    /// The most the engine may lock.
    pub fn limit(&self) -> u64 {
        self.books().limit
    }

    /// Bytes locked now, by everyone.
    pub fn used(&self) -> u64 {
        self.books().used()
    }

    /// Bytes the prompt cache holds in RAM now: its checkpoints and its
    /// pages.
    pub fn prompt_cache(&self) -> u64 {
        self.books().cache
    }

    /// Bytes of the K/V beyond VRAM that only the prompt cache holds
    /// ([`PinnedRam::set_context_cached`]).
    pub fn context_cached(&self) -> u64 {
        self.books().context_cached
    }

    /// Bytes the prompt cache's checkpoints hold now.
    pub fn checkpoints(&self) -> u64 {
        let books = self.books();
        books.cache.saturating_sub(books.pages)
    }

    /// Bytes the prompt cache's pages hold now.
    pub fn cache_pages(&self) -> u64 {
        self.books().pages
    }

    /// Bytes the tier of experts holds now.
    pub fn experts(&self) -> u64 {
        self.books().experts
    }

    /// Bytes the K/V of the context beyond VRAM holds now.
    pub fn context(&self) -> u64 {
        self.books().context
    }

    /// The K/V of the context beyond VRAM will hold `bytes` once the pieces
    /// that a step needs are locked: the elastic users' allowances leave that
    /// much from now on, so that the tier of experts gives it up and the
    /// bookings of the pieces find it free.
    pub fn set_context_target(&self, bytes: u64) {
        self.books().context_target = bytes;
    }

    /// Of the K/V of the context beyond VRAM, `bytes` hold pages that only
    /// the prompt cache holds: they rank as prompt cache, the lowest user,
    /// and give way first when memory is short (the conductor takes them
    /// off the devices, to the level on SSD or nowhere).
    pub fn set_context_cached(&self, bytes: u64) {
        self.books().context_cached = bytes;
    }

    /// `len` bytes of address space for the K/V of the context beyond VRAM
    /// on `device`, of which nothing is locked yet: [`ContextRegion::lock`]
    /// locks, maps and books pieces of it.
    pub fn context_region(&self, device: &Device, len: usize) -> Result<ContextRegion, ModelError> {
        let map = Arc::new(Mapping::new(len).map_err(ModelError::Invalid)?);
        Ok(ContextRegion { ram: self.clone(), device: device.clone(), map })
    }

    fn book_context(&self, bytes: u64) -> Result<(), u64> {
        self.books().grow(User::Context, bytes)
    }

    fn release_context(&self, bytes: u64) {
        self.books().shrink(User::Context, bytes);
    }

    /// The most the tier of experts could lock now: what the limit leaves
    /// after the required users, counting what the prompt cache, which ranks
    /// below it, could give back.
    pub fn obtainable_experts(&self) -> u64 {
        let books = self.books();
        books.limit.saturating_sub(books.required + books.context + books.experts)
    }

    /// What the tier of experts may hold now: all that the limit leaves
    /// after the required memory and the floor of the prompt cache, and no
    /// more than the machine lets it have above the part it cannot work
    /// without.
    pub fn experts_allowance(&self) -> u64 {
        self.books().allowances().0
    }

    /// What the prompt cache may hold in RAM now, its checkpoints and its
    /// pages together: what the limit leaves after the required memory and
    /// all the tier of experts can use, no more than the machine lets it
    /// have above its floor, and always its floor ([`PinnedRam::want_cache`]).
    pub fn cache_allowance(&self) -> u64 {
        self.books().allowances().1
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

    /// Says what the prompt cache can use at most, the conductor's budget,
    /// and what it keeps whatever the machine has, all in bytes: one
    /// checkpoint `place` (for a model with recurrent state) and some `pages`
    /// for a cache whose level on SSD moves state through them on its way to
    /// the drive and back. The floor is the cache's whatever the machine has,
    /// as the tier's floor is the tier's.
    pub fn want_cache(&self, want: u64, place: u64, pages: u64) {
        {
            let mut books = self.books();
            books.cache_want = want;
            books.place_floor = place;
            books.pages_floor = pages;
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

    /// `hook` is called with these books when the prompt cache holds more
    /// than its allowance ([`PinnedRam::cache_allowance`]), from the
    /// machine's thread: the conductor gives checkpoints and pages up, and
    /// the chunks of the places are released after it
    /// ([`PinnedRam::sweep_checkpoints`]).
    pub fn on_cache(&self, hook: impl Fn(&PinnedRam) + Send + Sync + 'static) {
        self.shared.hooks.lock().unwrap_or_else(|e| e.into_inner()).cache = Some(Arc::new(hook));
    }

    /// What locks the pieces of the tier of experts: each piece is booked
    /// here, locked for `device`, and gives its booking back when its lock
    /// is dropped.
    pub fn expert_pieces(&self, device: &Device) -> Box<dyn Locker> {
        Box::new(ExpertPieces { ram: self.clone(), device: device.clone() })
    }

    /// Books `bytes` for `user`, or says why the limit has no room for them
    /// (`what` names them).
    fn book(&self, user: User, bytes: u64, what: &str) -> Result<(), String> {
        let mut books = self.books();
        books.grow(user, bytes).map_err(|_| books.refusal(bytes, what))
    }

    fn book_experts(&self, bytes: u64) -> Result<(), String> {
        self.book(User::Experts, bytes, "the tier of experts")
    }

    fn release_experts(&self, bytes: u64) {
        self.books().shrink(User::Experts, bytes);
    }

    /// Books `bytes` for what the executor cannot run without.
    pub(crate) fn require(&self, bytes: u64) -> Result<(), RuntimeError> {
        self.book(User::Required, bytes, "the step record and read-back")
            .map_err(RuntimeError::Pinned)
    }

    /// Books `bytes` for the prompt cache, or says why the limit has no
    /// room for them.
    fn book_cache(&self, bytes: u64, what: &str) -> Result<(), String> {
        self.book(User::Cache, bytes, what)
    }

    pub(crate) fn book_checkpoints(&self, bytes: u64) -> Result<(), ModelError> {
        self.book_cache(bytes, "checkpoints").map_err(ModelError::Invalid)
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
        self.books().shrink(User::Cache, bytes);
    }

    /// One device share of a page of the prompt cache in RAM: `bytes` (at
    /// least; padded to a block of direct I/O) for device part `part`,
    /// booked, and page-locked when the books know the devices. `Err` when
    /// the limit has no room or the memory cannot be locked.
    pub fn cache_buffer(&self, part: usize, bytes: usize) -> Result<CacheBuffer, String> {
        let len = ares_store::padded(bytes.max(1));
        self.book_cache(len as u64, "pages")?;
        self.books().pages += len as u64;
        let device = {
            let devices = self.shared.devices.lock().unwrap_or_else(|e| e.into_inner());
            devices.get(part).or(devices.first()).cloned()
        };
        // Booked from here on: a buffer that cannot be locked gives the
        // booking back as it is dropped.
        let mut buffer = CacheBuffer { memory: CacheMemory::Plain(None), ram: self.clone(), len };
        buffer.memory = match device {
            Some(device) => {
                CacheMemory::Locked(PinnedBuffer::new(&device, len).map_err(|e| e.to_string())?)
            }
            None => CacheMemory::Plain(Some(Aligned::new(len))),
        };
        Ok(buffer)
    }

    fn release_cache_pages(&self, bytes: u64) {
        let mut books = self.books();
        books.pages = books.pages.saturating_sub(bytes);
        books.shrink(User::Cache, bytes);
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

    /// An executor without devices books places of `bytes` itself, and says
    /// which hold memory ([`PinnedRam::host_place_held`]).
    pub fn host_places(&self, bytes: u64) {
        *self.shared.host_places.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((bytes, Vec::new()));
    }

    /// Place `place` of an executor without devices holds memory, or not.
    pub fn host_place_held(&self, place: u32, held: bool) {
        let mut host = self.shared.host_places.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, places)) = host.as_mut() {
            if places.len() <= place as usize {
                places.resize(place as usize + 1, false);
            }
            places[place as usize] = held;
        }
    }

    /// The bytes the devices would lock when checkpoint place `place` is
    /// used next: the chunks it lies in that are not locked now, on every
    /// device part. 0 for a place whose memory is there, and for books
    /// whose places nobody books when they are first used.
    pub(crate) fn place_lock(&self, place: u32) -> u64 {
        self.places_lock(&[place])
    }

    /// The bytes the devices would lock when all of `places` are used next:
    /// the chunks they lie in that are not locked now, each once however many
    /// of the places lie in it, on every device part.
    pub(crate) fn places_lock(&self, places: &[u32]) -> u64 {
        if let Some((bytes, held)) =
            &*self.shared.host_places.lock().unwrap_or_else(|e| e.into_inner())
        {
            let mut new: Vec<u32> = places
                .iter()
                .copied()
                .filter(|place| !held.get(*place as usize).copied().unwrap_or(false))
                .collect();
            new.sort_unstable();
            new.dedup();
            return new.len() as u64 * *bytes;
        }
        let tables: Vec<Arc<PlaceChunks>> = {
            let all = self.shared.chunks.lock().unwrap_or_else(|e| e.into_inner());
            all.iter().filter_map(Weak::upgrade).collect()
        };
        tables
            .iter()
            .map(|table| {
                let chunks = table.chunks.lock().unwrap_or_else(|e| e.into_inner());
                let mut new: Vec<usize> = places
                    .iter()
                    .filter(|place| **place < table.capacity)
                    .map(|place| table.chunk_of(*place))
                    .filter(|index| !matches!(chunks.get(*index), Some(Some(_))))
                    .collect();
                new.sort_unstable();
                new.dedup();
                new.iter()
                    .map(|index| u64::from(table.chunk_places(*index).1) * table.bytes as u64)
                    .sum::<u64>()
            })
            .sum()
    }

    /// The bytes the limit leaves unbooked now.
    pub(crate) fn unbooked(&self) -> u64 {
        let books = self.books();
        books.limit.saturating_sub(books.used())
    }

    /// Releases chunks of checkpoint places in which the conductor has no
    /// checkpoint, until the prompt cache holds no more than its allowance.
    /// Returns the bytes released.
    pub fn sweep_checkpoints(&self) -> u64 {
        let allowance = self.cache_allowance();
        let mut held = self.prompt_cache();
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
                let (first, places) = table.chunk_places(index);
                let in_use =
                    live.iter().skip(first as usize).take(places as usize).any(|used| *used);
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

/// Address space of the process, unmapped when the last user of it is gone:
/// the range of a [`ContextRegion`]. It takes memory only where it is locked.
struct Mapping {
    base: *mut libc::c_void,
    len: usize,
}

// SAFETY: the mapping is plain memory; who touches which part of it is the
// business of the region's owner.
unsafe impl Send for Mapping {}
// SAFETY: as above; the mapping itself never changes after it is made.
unsafe impl Sync for Mapping {}

impl Mapping {
    fn new(len: usize) -> Result<Self, String> {
        let len = len.max(1).next_multiple_of(4096);
        // SAFETY: an anonymous mapping of fresh address space.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(format!(
                "no address space for {len} bytes of K/V beyond VRAM: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { base, len })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the range is the one `mmap` returned, and every lock on it
        // is gone: each holds the mapping.
        unsafe { libc::munmap(self.base, self.len) };
    }
}

/// Address space for the K/V of the context beyond VRAM on one device: pages
/// of K and V that the device reads and writes over its link. Pieces of it
/// are locked, mapped for the device and booked as they come into use
/// ([`ContextRegion::lock`]); the rest takes no memory.
pub struct ContextRegion {
    ram: PinnedRam,
    device: Device,
    map: Arc<Mapping>,
}

impl std::fmt::Debug for ContextRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextRegion").field("len", &self.map.len).finish_non_exhaustive()
    }
}

/// What a piece of a [`ContextRegion`] lacks: the bytes the limit would have
/// to take from others before the piece can be booked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextShort(pub u64);

impl ContextRegion {
    /// Where the region starts for the device: the host address, which a
    /// device with unified addressing reads registered memory at.
    pub fn base(&self) -> DevicePtr {
        DevicePtr::NULL.offset(self.map.base as u64)
    }

    /// The bytes of address space.
    pub fn len(&self) -> usize {
        self.map.len
    }

    pub fn is_empty(&self) -> bool {
        self.map.len == 0
    }

    /// Books, locks and maps `len` bytes at `at` (whole pages of the
    /// region). `Err(Ok(short))` when the limit has no room for them now
    /// (the tier of experts may give it up), `Err(Err(_))` when the
    /// device cannot take them or reads them at another address than the
    /// host's.
    pub fn lock(
        &self,
        at: usize,
        len: usize,
    ) -> Result<ContextLock, Result<ContextShort, ModelError>> {
        if !at.is_multiple_of(4096) || !len.is_multiple_of(4096) || at + len > self.map.len {
            return Err(Err(ModelError::Invalid(format!(
                "a piece of {len} bytes at {at} of a region of {} bytes",
                self.map.len
            ))));
        }
        self.ram.book_context(len as u64).map_err(|short| Ok(ContextShort(short)))?;
        // From here on the booking goes back when the lock is dropped, or here.
        let booking = ContextBooking { ram: self.ram.clone(), bytes: len as u64 };
        // SAFETY: the range is whole pages inside the mapping, which the lock
        // keeps alive.
        let ptr = unsafe { self.map.base.cast::<u8>().add(at) };
        // SAFETY: as above.
        let (registration, device) = unsafe { HostRegistration::mapped(&self.device, ptr, len) }
            .map_err(|e| Err(e.into()))?;
        if device.as_raw() != ptr as u64 {
            return Err(Err(ModelError::Invalid(format!(
                "device {} reads page-locked host memory at another address than the host's: \
                 K/V beyond VRAM is not built for it",
                self.device.ordinal()
            ))));
        }
        Ok(ContextLock {
            registration: Some(registration),
            _booking: booking,
            map: self.map.clone(),
            ptr,
            len,
        })
    }
}

/// A locked piece of a [`ContextRegion`]: unlocked, returned to the system and
/// unbooked when dropped. Unlocking waits for the device to finish what it
/// has, so it is dropped between steps.
pub struct ContextLock {
    registration: Option<HostRegistration>,
    _booking: ContextBooking,
    map: Arc<Mapping>,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the lock is a token for memory that only its owner's kernels touch.
unsafe impl Send for ContextLock {}

impl std::fmt::Debug for ContextLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextLock").field("len", &self.len).finish_non_exhaustive()
    }
}

impl ContextLock {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ContextLock {
    fn drop(&mut self) {
        // Unlocked first: locked pages cannot be given back.
        drop(self.registration.take());
        // SAFETY: the range is whole pages inside the mapping, which this
        // holds, and nothing reads the piece any more.
        unsafe { libc::madvise(self.ptr.cast(), self.len, libc::MADV_DONTNEED) };
        let _ = &self.map;
    }
}

/// What a piece of the context's K/V gives back to the books when it is
/// dropped (after its lock: its field comes after the registration's).
struct ContextBooking {
    ram: PinnedRam,
    bytes: u64,
}

impl Drop for ContextBooking {
    fn drop(&mut self) {
        self.ram.release_context(self.bytes);
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

/// One device share of a page of the prompt cache in RAM. See
/// [`PinnedRam::cache_buffer`]. Its bytes go back to the books when it is
/// dropped.
pub struct CacheBuffer {
    memory: CacheMemory,
    ram: PinnedRam,
    len: usize,
}

enum CacheMemory {
    Locked(PinnedBuffer),
    /// Ordinary memory; `None` only while the buffer is being made.
    Plain(Option<Aligned>),
}

impl std::fmt::Debug for CacheBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locked = matches!(self.memory, CacheMemory::Locked(_));
        f.debug_struct("CacheBuffer").field("len", &self.len).field("locked", &locked).finish()
    }
}

impl std::ops::Deref for CacheBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match &self.memory {
            CacheMemory::Locked(buffer) => buffer,
            CacheMemory::Plain(Some(buffer)) => buffer,
            CacheMemory::Plain(None) => &[],
        }
    }
}

impl std::ops::DerefMut for CacheBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        match &mut self.memory {
            CacheMemory::Locked(buffer) => buffer,
            CacheMemory::Plain(Some(buffer)) => buffer,
            CacheMemory::Plain(None) => &mut [],
        }
    }
}

impl Drop for CacheBuffer {
    fn drop(&mut self) {
        self.ram.release_cache_pages(self.len as u64);
    }
}

/// The places of one device part's checkpoints on the host. See
/// [`PinnedRam::place_chunks`]. The first chunk holds place 0 alone: the one
/// place the prompt cache keeps whatever the machine has is then as large as
/// its lock, however small a place is next to a chunk. The other chunks hold
/// `per_chunk` places each.
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

    /// The chunk place `place` lies in.
    fn chunk_of(&self, place: u32) -> usize {
        match (place, self.per_chunk) {
            (_, 1) => place as usize,
            (0, _) => 0,
            (place, per_chunk) => (1 + (place - 1) / per_chunk) as usize,
        }
    }

    /// The first place of chunk `index` and how many places it holds.
    fn chunk_places(&self, index: usize) -> (u32, u32) {
        let first = match (index, self.per_chunk) {
            (_, 1) => index as u32,
            (0, _) => 0,
            (index, per_chunk) => 1 + (index as u32 - 1) * per_chunk,
        };
        let places = if index == 0 { 1 } else { self.per_chunk };
        (first, places.min(self.capacity.saturating_sub(first)))
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
        let index = self.chunk_of(place);
        let (first, places) = self.chunk_places(index);
        let mut chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        if chunks.len() <= index {
            chunks.resize_with(index + 1, || None);
        }
        let fresh = chunks[index].is_none();
        if fresh {
            chunks[index] =
                Some(self.ram.lock_checkpoints(&self.device, places as usize * self.bytes)?);
        }
        let chunk = chunks[index].as_mut().expect("locked above");
        let within = (place - first) as usize * self.bytes;
        // SAFETY: the place is inside its chunk, which holds the places from
        // `first` on (the first chunk one, the others `per_chunk`, the last
        // the rest).
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
        let index = self.chunk_of(place);
        let first = self.chunk_places(index).0;
        let mut chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        let chunk = chunks.get_mut(index)?.as_mut()?;
        let within = (place - first) as usize * self.bytes;
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
        assert_eq!((ram.experts_allowance(), ram.cache_allowance()), (90 * MIB, 90 * MIB));
        // The checkpoints hold 40 MiB; the tier of experts, which ranks above
        // them, could still have all 90, once they give back what they hold.
        ram.book_checkpoints(40 * MIB).unwrap();
        assert_eq!(ram.obtainable_experts(), 90 * MIB);
        assert_eq!(ram.books().grow(User::Experts, 90 * MIB), Err(40 * MIB));
        // A tier that can use 70 MiB leaves the checkpoints 20, and one that
        // can use everything leaves them nothing.
        ram.want_experts(70 * MIB, 0);
        assert_eq!((ram.experts_allowance(), ram.cache_allowance()), (90 * MIB, 20 * MIB));
        ram.want_experts(GIB, 0);
        assert_eq!(ram.cache_allowance(), 0);
        // And the checkpoints cannot take memory back from the tier.
        ram.release_checkpoints(40 * MIB);
        ram.book_experts(90 * MIB).unwrap();
        assert!(ram.book_checkpoints(20 * MIB).is_err());
    }

    #[test]
    fn the_context_beyond_vram_takes_its_room_from_the_tier_of_experts() {
        let ram = PinnedRam::new(100 * MIB);
        ram.require(10 * MIB).unwrap();
        ram.want_experts(GIB, 0);
        ram.book_experts(90 * MIB).unwrap();
        assert_eq!(ram.experts_allowance(), 90 * MIB);
        // A step will hold 30 MiB of K and V beyond VRAM: the tier's allowance
        // leaves them from now on.
        ram.set_context_target(30 * MIB);
        assert_eq!(ram.experts_allowance(), 60 * MIB);
        // Until the tier gives them up the booking finds no room, and says how
        // much others would have to give.
        assert_eq!(ram.book_context(30 * MIB), Err(30 * MIB));
        ram.release_experts(30 * MIB);
        ram.book_context(30 * MIB).unwrap();
        assert_eq!(ram.context(), 30 * MIB);
        // Neither the prompt cache nor the tier takes it back.
        assert!(ram.book_checkpoints(MIB).is_err());
        assert!(ram.book_experts(MIB).is_err());
        // Given back, the room is the tier's again.
        ram.release_context(30 * MIB);
        ram.set_context_target(0);
        assert_eq!((ram.context(), ram.experts_allowance()), (0, 90 * MIB));
    }

    #[test]
    fn far_k_and_v_of_finished_requests_counts_as_prompt_cache_and_not_against_the_tier() {
        let ram = PinnedRam::new(100 * MIB);
        ram.require(10 * MIB).unwrap();
        ram.want_experts(GIB, 0);
        // The far pools hold 30 MiB of K and V: 10 for running sequences, 20
        // for pages only the prompt cache holds.
        ram.book_context(30 * MIB).unwrap();
        ram.set_context_cached(20 * MIB);
        // The tier may have all but the running sequences' 10: the cached 20
        // are the prompt cache's, which ranks below it.
        assert_eq!(ram.experts_allowance(), 80 * MIB);
        assert_eq!(ram.cache_allowance(), 0, "the tier wants everything");
        let wants = ram.shared.wants();
        assert_eq!(wants.cache_held, 20 * MIB, "the machine sees them as the cache's");
        // When the cache gives them up (the conductor takes the pages off
        // the devices) and the pieces are unlocked, the tier gets the memory.
        ram.set_context_cached(0);
        ram.release_context(20 * MIB);
        assert_eq!(ram.experts_allowance(), 80 * MIB);
        ram.book_experts(80 * MIB).unwrap();
        assert_eq!(ram.unbooked(), 0);
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
        ram.on_cache(|ram| {
            let allowance = ram.cache_allowance();
            while ram.checkpoints() > allowance {
                ram.release_checkpoints(PIECE.min(ram.checkpoints()));
            }
        });
        ram.want_cache(want, 0, 0);
    }

    /// The cache takes checkpoints as far as it may.
    fn fill(ram: &PinnedRam) {
        while ram.checkpoints() + PIECE <= ram.cache_allowance() {
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
        assert_eq!(ram.cache_allowance(), 0, "and it may take no new checkpoint");
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
        ram.want_cache(40 * MIB, 8 * MIB, 0);
        ram.want_experts(GIB, 0);
        assert_eq!((ram.experts_allowance(), ram.cache_allowance()), (82 * MIB, 8 * MIB));
        // A floor is never more than what the cache can use at all.
        ram.want_cache(4 * MIB, 8 * MIB, 0);
        assert_eq!(ram.cache_allowance(), 4 * MIB);

        // A machine to follow, on which other programs leave nothing: the
        // checkpoints go down to their floor and no further, and the machine
        // asks for nothing it cannot have: no release is owed.
        let (sim, machine) = Sim::new(32 * GIB, 16 * GIB);
        let ram = sim.engine(&machine, 64 * GIB, GIB);
        tier(&ram, 8 * GIB);
        cache(&ram, 32 * GIB);
        ram.want_cache(32 * GIB, PIECE, 0);
        fill(&ram);
        sim.run(&machine, 5_000);
        fill(&ram);
        assert!(ram.checkpoints() > 6 * GIB);
        for _ in 0..200 {
            sim.others.fetch_add(10 * GIB / 200, Ordering::SeqCst);
            sim.run(&machine, 50);
        }
        sim.run(&machine, 500);
        assert_eq!((ram.cache_allowance(), ram.checkpoints()), (PIECE, PIECE));
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
