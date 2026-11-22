//! The conductor's prefix cache: the index of cached pages, the places of
//! checkpoints of recurrent state, the copies the next step record carries,
//! and the levels below the devices: RAM, then the drive.
//!
//! The conductor decides when to look, publish, capture, restore and evict;
//! this holds what those decisions act on, so that `conductor.rs` keeps to the
//! step loop. A model takes part when [`Model::prefix_cache`] says its device
//! side carries out the record's copies, and keeps pages in RAM and on SSD
//! when it moves them to the host and back ([`Model::durable`]).
//!
//! [`Model::prefix_cache`]: crate::Model::prefix_cache
//! [`Model::durable`]: crate::Model::durable
//!
//! # Checkpoint places
//!
//! A checkpoint is a copy of a slot's recurrent state in one of
//! `capacity` places in RAM; its [`CheckpointId`] is the place. The index
//! decides which of its checkpoints stay ([`PrefixIndex::checkpoint_ranks`]);
//! this frees and reuses the places. A place that a restore of the next record
//! will read is not reused before that record is written, however the index
//! treated its checkpoint meanwhile. A checkpoint hangs on a page of the index
//! or, once that page has left the devices, on the page in RAM: it moves with
//! its page and stays in its place.
//!
//! # The levels
//!
//! A page the index gives up goes into RAM ([`ram`]): it is copied out of the
//! pool before anything writes its page of the pool again
//! ([`Cache::settle_leaving`], which runs before every record and at the start
//! of every transfer). A prompt that finds its start in RAM has it copied back
//! when it arrives, before the step is planned ([`Cache::probe`]); the copy in
//! RAM stays. What leaves RAM goes to the drive when there is a level on SSD
//! and it is worth writing ([`ssd`]), and is dropped otherwise. Nothing is
//! written while RAM holds it, except the coldest of it when RAM is full, so
//! that there is always room for [`RESERVE`] pages; at the end of an engine
//! everything worth writing is written ([`Cache::drain`]).
//!
//! # The allowance
//!
//! The checkpoints and the pages in RAM are the lowest user of the engine's
//! page-locked memory ([`PinnedRam`]): the books say how many bytes they may
//! hold now, and that can be less than the budget the places were counted
//! for, down to nothing. The cache keeps no more than the allowance holds
//! ([`Cache::trim`] gives up what goes for nothing first, then writes what
//! only RAM holds, then the coldest checkpoints), and it tells the books which
//! places are in use, so that the device parts can give the memory of the
//! others back. A place counts as unused only once the record with its last
//! copy has run and its checkpoint has left it for the drive.
//!
//! - **A save that has not run has nothing to write.** A checkpoint that is
//!   given up before the record with its save was written loses that save,
//!   and the level on SSD counts it when it was due there
//!   ([`SsdCounters::states_skipped`]): nothing is read from a place that was
//!   never written.
//! - **A floor.** With the level on SSD in use the cache keeps one place (for
//!   a model with recurrent state) and [`RESERVE`] pages whatever the
//!   allowance says, its floor in the books: state passes through them on the
//!   way to the drive and back, so a prompt resumes from SSD while the
//!   machine has no room for the cache.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use ares_state::{
    Boundary, Chain, ChainHash, CheckpointId, CheckpointPolicy, Evicted, Owner, PageId, PagePool,
    PageState, PrefixIndex, PrefixMatch, Priority, Resume, Sequence,
};
use ares_store::Kind;

use crate::config::RuntimeConfig;
use crate::exec::{Executor, Transfer, TransferKind};
use crate::model::{Durable, PrefixCache};
use crate::ram::PinnedRam;
use crate::record::{PageCopy, RecordFull, RecordWriter, StateCopy, StateCopyKind};

mod ram;
mod ssd;

pub use ram::RamCounters;
use ram::{Class, Entry, Held, Item, RamLevel};
pub use ssd::{SsdConfig, SsdCounters};

/// The pages RAM keeps room for at any time: when it is full, the coldest of
/// what only it holds is written ahead, so that this many pages can come in
/// without waiting for the drive. With a level on SSD it is also the floor of
/// pages: what passes through RAM on its way to the drive and back.
pub(crate) const RESERVE: u64 = 16;

/// What a look into the index found for a text.
pub(crate) struct Probe {
    pub found: PrefixMatch,
    pub resume: Resume,
    /// The matched pages that no sequence holds now: taking them makes them
    /// unavailable for eviction.
    pub idle: u32,
    /// Pages and tails that came back from RAM for this look.
    pub brought: u32,
}

/// What an eviction gave up.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Evictions {
    pub pages: u32,
    /// Checkpoints that went with their pages: none while there is RAM
    /// below the devices, where they go with them.
    pub checkpoints: u32,
}

/// What taking a checkpoint came to.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Saved {
    /// A checkpoint was taken: its copy runs with the next record.
    pub taken: bool,
    /// The checkpoints that went to make room for it.
    pub dropped: u32,
}

/// A page or tail that has left the devices and waits for its copy into RAM,
/// which runs before anything writes its page of the pool again.
struct Leaving {
    page: PageId,
    hash: ChainHash,
    parent: ChainHash,
    depth: u32,
    tail: Option<Vec<u64>>,
    checkpoint: Option<Held>,
    /// The page's text was long enough to be written.
    worth: bool,
}

/// A checkpoint on its way out of its place to the drive.
#[derive(Clone, Copy, Debug)]
struct StateOut {
    place: u32,
    hash: ChainHash,
    parent: ChainHash,
    depth: u32,
    /// It stays in its place and is written ahead; otherwise it has left.
    keep: bool,
}

/// What one transfer of a batch is for.
enum Note {
    /// A page that left the devices, into RAM.
    ToRam(Leaving),
    /// A page or tail from RAM into the transfer's page of the pool, which
    /// enters the index under `owner` with the checkpoint that hangs on it.
    FromRam {
        hash: ChainHash,
        parent: ChainHash,
        owner: Owner,
    },
    /// A page of the devices on its way to the drive, through a buffer of
    /// RAM: when the engine ends.
    CardsOut {
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
    },
    /// A checkpoint out of its place, for the drive.
    StateOut(StateOut),
    /// A page and a checkpoint of a run that comes back from the drive.
    PageIn {
        job: u64,
        index: usize,
    },
    StateIn {
        job: u64,
    },
}

pub(crate) struct Cache {
    pub index: PrefixIndex,
    policy: CheckpointPolicy,
    root: ChainHash,
    cells_per_page: u32,
    /// The model has recurrent state: starts restore it and replay rows.
    recurrent: bool,
    /// The places there are, the next one never used, and the ones given back.
    capacity: u32,
    next_place: u32,
    free_places: Vec<u32>,
    /// Places whose checkpoint was dropped while a restore of them waits for
    /// its record.
    deferred: Vec<u32>,
    /// What hangs in each place in use: the name of the page or tail.
    placed: HashMap<u32, ChainHash>,
    /// The books of the page-locked memory the places and the pages in RAM
    /// lie in, and the bytes of one checkpoint on all devices. Without the
    /// executor's books, the pages in RAM are kept in books of their own.
    books: Option<PinnedRam>,
    bytes: u64,
    /// Places that were freed since the last record was written (a copy of
    /// the next record may still use their memory), and places that were
    /// freed before that record, which may have used them: unused once it
    /// has run.
    settling: Vec<u32>,
    ran: Vec<u32>,
    /// The copies for the next record.
    saves: Vec<StateCopy>,
    restores: Vec<StateCopy>,
    page_copies: Vec<PageCopy>,
    scratch: Vec<StateCopy>,
    /// Counts what was given up, so that waiting requests can tell their
    /// numbers are out of date.
    pub evictions: u64,
    /// What the model offers the levels below the devices.
    durable: Option<Durable>,
    /// The level in RAM, for a model that moves its pages to the host.
    in_ram: Option<RamLevel>,
    /// The level on SSD, when the engine has one and the model can use it,
    /// and why not when it was asked for and is not in use.
    ssd: Option<ssd::Ssd>,
    pub ssd_problem: Option<String>,
    /// Pages that left the devices and wait for their copy into RAM, and
    /// checkpoints that wait for theirs to the drive: before the next record
    /// and before any transfer that writes into the pool or the places.
    leaving: Vec<Leaving>,
    states_out: Vec<StateOut>,
    /// Pages in the index whose text was long enough to be written, and how
    /// far each running sequence has been looked at for that.
    worth: HashSet<ChainHash>,
    cursors: HashMap<Owner, usize>,
    /// The next owner of a run brought back from RAM.
    next_import: u64,
    /// The engine ends: what is due on the drive waits for the writer as
    /// long as it takes.
    draining: bool,
    /// The pages the cache keeps in RAM whatever the allowance says, in
    /// bytes: the floor of pages, which checkpoints do not take.
    page_floor: u64,
}

impl Cache {
    /// The checkpoint places the budget of `config` holds for a model of this
    /// `spec`. The conductor and the device parts both use this number.
    pub(crate) fn places(spec: &PrefixCache, config: &RuntimeConfig) -> u32 {
        let policy = CheckpointPolicy::new(config.cells_per_page, spec.bytes(), config.cache_ram);
        u32::try_from(policy.capacity()).unwrap_or(u32::MAX)
    }

    /// `durable` is what the model offers the levels below the devices: the
    /// level in RAM is in use when the model offers it, the level on SSD when
    /// `config` names a directory too.
    pub(crate) fn new(
        model_name: &str,
        spec: &PrefixCache,
        durable: Option<&Durable>,
        config: &RuntimeConfig,
        cells_per_page: u32,
    ) -> Self {
        let recurrent = spec.bytes() > 0;
        let (ssd, ssd_problem) = match (&config.ssd, durable) {
            (Some(at), Some(durable)) => {
                match ssd::Ssd::open(at, durable, cells_per_page, recurrent) {
                    Ok(ssd) => (Some(ssd), None),
                    Err(why) => (None, Some(why)),
                }
            }
            (Some(_), None) => (None, Some("this model keeps no state on SSD".to_owned())),
            (None, _) => (None, None),
        };
        // What is on SSD is found by the names of its pages, so they start
        // from what the store's fingerprint covers and not from a name that
        // the operator may change.
        let root = match (&ssd, durable) {
            (Some(_), Some(durable)) => {
                ChainHash::root(&ssd::Ssd::fingerprint(durable, cells_per_page).bytes())
            }
            _ => ChainHash::root(model_name.as_bytes()),
        };
        let policy = CheckpointPolicy {
            prompt_interval: config.checkpoint_prompt_interval,
            generation_interval: config.checkpoint_generation_interval,
            boundary_interval: config.checkpoint_boundary_interval,
            ..CheckpointPolicy::new(cells_per_page, spec.bytes(), config.cache_ram)
        };
        let index = match spec.tails_per_slot {
            Some(per_slot) => {
                PrefixIndex::new().with_tail_limit(per_slot as usize * config.max_slots as usize)
            }
            None => PrefixIndex::new(),
        };
        let capacity = Self::places(spec, config);
        Self {
            index,
            root,
            cells_per_page,
            recurrent,
            durable: durable.cloned(),
            in_ram: None,
            ssd,
            ssd_problem,
            leaving: Vec::new(),
            states_out: Vec::new(),
            worth: HashSet::new(),
            cursors: HashMap::new(),
            next_import: 0,
            draining: false,
            page_floor: 0,
            capacity,
            policy,
            next_place: 0,
            free_places: Vec::new(),
            deferred: Vec::new(),
            placed: HashMap::new(),
            books: None,
            bytes: spec.bytes(),
            settling: Vec::new(),
            ran: Vec::new(),
            saves: Vec::new(),
            restores: Vec::new(),
            page_copies: Vec::new(),
            scratch: Vec::new(),
            evictions: 0,
        }
    }

    /// Prompts are read in pieces of `rows` rows: a piece that ends this
    /// close before a message boundary takes the boundary's checkpoint where
    /// it ends, and is not cut ([`CheckpointPolicy::boundary_reach`]).
    pub(crate) fn with_reach(mut self, rows: u32) -> Self {
        self.policy.boundary_reach = rows;
        self
    }

    /// The places and the pages in RAM lie in the page-locked memory of
    /// `books`: the cache keeps to its allowance and tells them which places
    /// are in use. Without books of the executor's, pages in RAM are booked
    /// in books of their own, as large as the budget. One place, for a model
    /// with recurrent state, is what the cache keeps whatever the machine and
    /// the tier of experts leave it: without a checkpoint every hit replays
    /// all the rows it found, so a conversation resumes from its last
    /// checkpoint at least. With the level on SSD in use, [`RESERVE`] pages
    /// are kept as well: state passes through them to the drive and back.
    pub(crate) fn with_ram(mut self, books: Option<PinnedRam>) -> Self {
        let page = self.durable.as_ref().map_or(0, |durable| {
            durable.page_bytes.iter().map(|b| ares_store::padded(*b as usize) as u64).sum()
        });
        let through = self.ssd.is_some();
        let place = if self.recurrent && self.capacity > 0 { self.bytes } else { 0 };
        let pages = if through { RESERVE * page } else { 0 };
        let want = self.policy.budget_bytes.max(place + pages);
        let books = match (books, &self.durable) {
            (Some(books), _) => Some(books),
            // Pages in RAM need books; checkpoints of an executor that has
            // none lie in memory nobody books.
            (None, Some(_)) => Some(PinnedRam::new(want)),
            (None, None) => None,
        };
        if let Some(books) = &books {
            books.want_cache(want, place, pages);
        }
        self.page_floor = pages;
        if let (Some(durable), Some(books)) = (&self.durable, &books) {
            self.in_ram = Some(RamLevel::new(&durable.page_bytes, books.clone(), through));
        }
        self.books = books;
        self
    }

    /// What the books let the prompt cache hold now, places and pages.
    fn allowance(&self) -> u64 {
        self.books.as_ref().map_or(u64::MAX, PinnedRam::cache_allowance)
    }

    /// The bytes the cache holds in RAM now: the places in use and the
    /// buffers of pages, at least their floor, which is the pages' whether
    /// they use it or not.
    fn held(&self) -> u64 {
        let places = u64::from(self.next_place) - self.free_places.len() as u64;
        let pages = self.in_ram.as_ref().map_or(0, RamLevel::bytes);
        places * self.bytes + pages.max(self.page_floor)
    }

    /// The policy with the budget the checkpoints have now: what the books
    /// allow but the floor of pages, and never more than the budget the
    /// places were counted for. Pages in RAM above their floor make room for
    /// checkpoints, not the other way round.
    fn policy_now(&self) -> CheckpointPolicy {
        let allowed = self.allowance().saturating_sub(self.page_floor);
        CheckpointPolicy { budget_bytes: self.policy.budget_bytes.min(allowed), ..self.policy }
    }

    fn mark(&self, place: u32, live: bool) {
        if let Some(books) = &self.books {
            books.place_live(place, live);
        }
    }

    /// The record that was written last has run: the places that were freed
    /// before it are unused. Call it between steps only.
    fn settle(&mut self) {
        for place in std::mem::take(&mut self.ran) {
            if self.free_places.contains(&place) {
                self.mark(place, false);
            }
        }
    }

    /// The names of the full pages of `tokens`.
    pub(crate) fn chain(&self, tokens: &[u64]) -> Chain {
        let mut chain = Chain::new(self.root, self.cells_per_page);
        chain.name(tokens);
        chain
    }

    /// The chain of a sequence that starts from `found`, with the full pages of
    /// `tokens` named.
    pub(crate) fn chain_after(&self, found: &PrefixMatch, tokens: &[u64]) -> Chain {
        let mut chain = Chain::resumed(self.root, self.cells_per_page, found);
        chain.name(tokens);
        chain
    }

    /// What the cache has for a prompt of `tokens` (named in `chain`), whose
    /// last token is always computed. The matched pages count as used.
    ///
    /// With `exec`, what RAM holds of the prompt after the pages the index
    /// has is copied to the devices first and enters the index. Only while
    /// no step is planned: the copies take pages of the pool, which the
    /// accounting of the step has to see.
    pub(crate) fn probe(
        &mut self,
        pool: &mut PagePool,
        exec: Option<&mut dyn Executor>,
        chain: &Chain,
        tokens: &[u64],
    ) -> Probe {
        let looked = &tokens[..tokens.len() - 1];
        let mut found = self.index.lookup_chain(chain, looked);
        let mut brought = 0;
        if let (Some(exec), None) = (exec, found.computing) {
            brought = self.import_from_ram(pool, exec, chain, tokens, &found);
            if brought > 0 {
                found = self.index.lookup_chain(chain, looked);
            }
        }
        // What the index does not have may be on SSD: a run of pages that is
        // brought back holds the prompt as a page that is being computed does.
        if let (Some(ssd), None) = (&mut self.ssd, found.computing) {
            let total = looked.len() / self.cells_per_page as usize;
            found.computing = ssd.want(&mut self.index, chain, found.pages.len(), total);
        }
        let cached = found.cells(self.cells_per_page).min(tokens.len() as u64);
        let resume = if self.recurrent {
            self.policy.resume(&found, tokens.len() as u64)
        } else {
            Resume {
                cached_cells: cached,
                restore: None,
                replay: 0..0,
                compute: cached..tokens.len() as u64,
            }
        };
        let idle = found
            .pages
            .iter()
            .copied()
            .chain(found.tail.map(|tail| tail.page))
            .filter(|page| pool.refcount(*page) == 1)
            .count() as u32;
        Probe { found, resume, idle, brought }
    }

    /// The first position in `(from, to]` at which a checkpoint is due for a
    /// sequence of a prompt of `prompt_len` cells, whose parts end at
    /// `boundaries` and whose last checkpoint is at `last`, or `None`.
    pub(crate) fn capture(
        &self,
        prompt_len: u32,
        boundaries: &[Boundary],
        last: u64,
        from: u64,
        to: u64,
    ) -> Option<u64> {
        self.policy.next_capture(u64::from(prompt_len), boundaries, last, from, to)
    }

    /// Whether a checkpoint at `at` lies at a root of the prompt whose parts
    /// end at `boundaries`.
    pub(crate) fn at_root(&self, boundaries: &[Boundary], at: u64) -> bool {
        self.policy.at_root(boundaries, at)
    }

    /// The place of a new checkpoint, with room made for it, and how many
    /// checkpoints that cost. What goes for nothing makes room first; then,
    /// when the checkpoints are as many as the allowance holds, those of
    /// pages that left the devices go before those of the index.
    fn new_place(&mut self) -> (Option<u32>, u32) {
        let mut dropped = 0;
        while self.held() + self.bytes > self.allowance() && self.give_up_free(&mut dropped) {}
        let capacity = self.policy_now().capacity();
        let in_ram = |cache: &Self| cache.in_ram.as_ref().map_or(0, |ram| ram.checkpoints());
        while self.index.checkpoints() as u64 + in_ram(self) + 1 > capacity as u64 {
            if !self.give_up_ram_checkpoint() && !self.give_up_index_checkpoint(false) {
                break;
            }
            dropped += 1;
        }
        // No room for even one checkpoint now: none is taken.
        if capacity == 0 {
            return (None, dropped);
        }
        let place = self.free_place(&mut dropped);
        if let Some(place) = place {
            self.mark(place, true);
        }
        (place, dropped)
    }

    /// A place for a checkpoint whose memory the limit of the books holds: a
    /// free place whose memory is locked already if there is one, else one
    /// whose first use the limit has room for, made by giving up what goes
    /// for nothing. The devices lock a place when a copy of a step first uses
    /// it, and a lock beyond the limit would fail that step: pages in RAM
    /// share the limit with the places, so this is not given by the number
    /// of places alone. `None` when there is no such place. `dropped` counts
    /// the checkpoints that went to make room.
    fn free_place(&mut self, dropped: &mut u32) -> Option<u32> {
        let Some(books) = self.books.clone() else {
            return self.free_places.pop().or_else(|| self.unused_place());
        };
        loop {
            let locked = self.free_places.iter().rposition(|place| books.place_lock(*place) == 0);
            if let Some(at) = locked {
                return Some(self.free_places.remove(at));
            }
            let place = self
                .free_places
                .last()
                .copied()
                .or_else(|| (self.next_place < self.capacity).then_some(self.next_place))?;
            if books.unbooked() >= books.place_lock(place) + self.pending_locks() {
                return match self.free_places.last() == Some(&place) {
                    true => self.free_places.pop(),
                    false => self.unused_place(),
                };
            }
            if !self.give_up_free(dropped) {
                return None;
            }
        }
    }

    /// The bytes the devices will lock for the saves of the next record: the
    /// places they use whose memory is not locked yet. Nothing else may book
    /// that room of the limit before the record has run.
    fn pending_locks(&self) -> u64 {
        let Some(books) = &self.books else { return 0 };
        self.saves.iter().map(|copy| books.place_lock(copy.place)).sum()
    }

    /// A place never used so far, if there is one.
    fn unused_place(&mut self) -> Option<u32> {
        (self.next_place < self.capacity).then(|| {
            self.next_place += 1;
            self.next_place - 1
        })
    }

    /// Gives one thing up that is worth little: a buffer that holds nothing,
    /// a page or checkpoint in RAM that the devices or the drive hold too or
    /// that is not worth writing, or a checkpoint of the index that later
    /// requests have left behind (written all the same when the drive keeps
    /// nothing deeper of its text). `false` when there is nothing of the
    /// kind. `dropped` counts the checkpoints.
    fn give_up_free(&mut self, dropped: &mut u32) -> bool {
        if let Some(ram) = &mut self.in_ram {
            if ram.free_bytes() > 0 {
                ram.shed(0);
                return true;
            }
            if let Some((Class::Copy | Class::Free, item, hash)) = ram.first() {
                *dropped += self.ram_give_up(item, hash);
                return true;
            }
        }
        if self.give_up_index_checkpoint(true) {
            *dropped += 1;
            return true;
        }
        false
    }

    /// Gives up the page or the checkpoint `hash` in RAM, the checkpoint
    /// with its page; what is due on the drive goes there. Returns the
    /// checkpoints that went.
    fn ram_give_up(&mut self, item: Item, hash: ChainHash) -> u32 {
        let writes = self.ssd.is_some();
        let Some(ram) = &mut self.in_ram else { return 0 };
        let (held, parent, depth) = match item {
            Item::Page => {
                let Some(entry) = ram.remove(hash) else { return 0 };
                let unwritten = !entry.on_cards && entry.worth && !entry.stored;
                if unwritten && entry.tail.is_none() && writes {
                    ram.counters.pages_dropped += 1;
                }
                if let Some(parts) = entry.parts {
                    ram.recycle(parts);
                }
                (entry.checkpoint, entry.parent, entry.depth)
            }
            Item::Checkpoint => {
                let mut held = None;
                ram.update(hash, |entry| held = entry.checkpoint.take());
                let entry = ram.get(hash).expect("the entry is there");
                (held, entry.parent, entry.depth)
            }
        };
        let Some(held) = held else { return 0 };
        let out = self.state_out_of(&held, hash, parent, depth);
        self.give_up(held.place, out);
        1
    }

    /// Gives up the checkpoint that goes first of those that hang on pages
    /// in RAM. `false` when there is none.
    fn give_up_ram_checkpoint(&mut self) -> bool {
        let Some(hash) = self.in_ram.as_ref().and_then(RamLevel::first_checkpoint) else {
            return false;
        };
        self.ram_give_up(Item::Checkpoint, hash) > 0
    }

    /// Gives up the checkpoint of the index that its ranking drops first, or,
    /// with `left_behind`, only one that later requests have left behind. It
    /// goes to the drive unless later requests left it behind at deeper page
    /// boundaries ([`Cache::left_behind`]). `false` when there is none.
    fn give_up_index_checkpoint(&mut self, left_behind: bool) -> bool {
        if self.index.checkpoints() == 0 {
            return false;
        }
        let Some(first) = self.index.checkpoint_ranks(&self.policy).first().copied() else {
            return false;
        };
        if left_behind && !first.aged {
            return false;
        }
        let behind = self.left_behind().contains(&first.hash);
        let Ok(Some(id)) = self.index.set_checkpoint(first.hash, None) else { return false };
        let out = match self.index.page(first.hash) {
            Some((_, parent, depth)) if !behind => {
                self.state_out(id.0 as u32, first.hash, parent, depth)
            }
            _ => None,
        };
        self.give_up(id.0 as u32, out);
        true
    }

    /// The checkpoints of the index at page boundaries that later requests
    /// have left behind at deeper page boundaries. Only a level on SSD asks:
    /// such a checkpoint is not written, since the drive keeps no tails and a
    /// prompt resumes there from the deepest checkpoint at a page boundary.
    fn left_behind(&self) -> HashSet<ChainHash> {
        if self.ssd.is_none() || self.index.checkpoints() == 0 {
            return HashSet::new();
        }
        self.index.left_behind_on_pages(&self.policy).into_iter().collect()
    }

    /// A page at `depth` lies deep enough for its text to be written.
    fn deep(&self, depth: u32) -> bool {
        self.ssd
            .as_ref()
            .is_some_and(|ssd| u64::from(depth) * u64::from(self.cells_per_page) >= ssd.min_cells())
    }

    /// The checkpoint in `place`, at the end of page `hash`, goes to the
    /// drive when it leaves its place: there is a level on SSD, it lies deep
    /// enough, and the drive does not have it.
    fn state_out(
        &self,
        place: u32,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
    ) -> Option<StateOut> {
        let ssd = self.ssd.as_ref()?;
        let due = self.deep(depth) && !ssd.has_or_queued(Kind::State, hash);
        due.then_some(StateOut { place, hash, parent, depth, keep: false })
    }

    fn state_out_of(
        &self,
        held: &Held,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
    ) -> Option<StateOut> {
        (held.worth && !held.stored).then(|| self.state_out(held.place, hash, parent, depth))?
    }

    /// The checkpoint in `place` is given up: `out` says it goes to the
    /// drive first. The place is free once nothing reads it.
    fn give_up(&mut self, place: u32, out: Option<StateOut>) {
        self.placed.remove(&place);
        // A save of it that still waits for its record has nothing to save,
        // and the level on SSD nothing to write from the place.
        let unsaved = self.saves.iter().any(|copy| copy.place == place);
        self.saves.retain(|copy| copy.place != place);
        if let (Some(ssd), Some(out)) = (&mut self.ssd, out) {
            if unsaved {
                ssd.counters.states_skipped += 1;
            } else {
                ssd.queue(Kind::State, out.hash);
                self.states_out.push(out);
            }
        }
        if self.restores.iter().any(|copy| copy.place == place) {
            self.deferred.push(place);
        } else {
            self.free(place);
        }
    }

    /// Gives a checkpoint's place back once nothing reads it, writing the
    /// checkpoint to the drive first when it is due there.
    pub(crate) fn release(&mut self, id: CheckpointId) {
        let place = id.0 as u32;
        let out = self
            .placed
            .get(&place)
            .and_then(|hash| Some((*hash, self.index.page(*hash)?)))
            .and_then(|(hash, (_, parent, depth))| self.state_out(place, hash, parent, depth));
        self.give_up(place, out);
    }

    /// Place `place` holds no checkpoint any more and no restore waits for
    /// it: it can be taken again, and it is unused once the record that was
    /// written last has run (a copy of it may still use the memory).
    fn free(&mut self, place: u32) {
        self.free_places.push(place);
        self.settling.push(place);
    }

    /// Takes a checkpoint of the state in `slot`, which is at the end of the
    /// page named `hash` or of the tail so named, unless the index has one
    /// there already. With `root`, it is at the end of a system prompt or a
    /// tool list, and kept longest. The copy runs with the next record.
    pub(crate) fn save(&mut self, slot: u32, hash: ChainHash, owner: Owner, root: bool) -> Saved {
        if !self.recurrent || self.capacity == 0 {
            return Saved::default();
        }
        if self.index.checkpoint(hash).is_some() {
            // A root of this prompt that another text keeps as a plain one.
            if root {
                self.index.mark_root(hash);
            }
            return Saved::default();
        }
        let (place, dropped) = self.new_place();
        let Some(place) = place else { return Saved { taken: false, dropped } };
        match self.index.attach_checkpoint(hash, CheckpointId(u64::from(place)), owner) {
            Ok(before) => {
                debug_assert!(before.is_none(), "the entry had no checkpoint");
                if root {
                    self.index.mark_root(hash);
                }
                self.saves.push(StateCopy { kind: StateCopyKind::Save, slot, place });
                self.placed.insert(place, hash);
                Saved { taken: true, dropped }
            }
            Err(id) => {
                self.free(id.0 as u32);
                Saved { taken: false, dropped }
            }
        }
    }

    /// Takes the checkpoint of a request's live state at its end, as
    /// [`Cache::save`], unless the budget holds one checkpoint besides the
    /// roots and that is a fresh one of the same request (its anchor). The
    /// end state lies less than a page and the reply's rows after the anchor,
    /// and the next turn of a conversation often parts from it inside the
    /// reply: Claude Code sends back a call of its Edit tool with an argument
    /// the model never wrote, Qwen's templates leave out the thinking of an
    /// answer once a new question follows, and a client asks the same prompt
    /// again. With the end state in the one place such a turn replays every
    /// row it shares after the roots; with the anchor, the reply's rows.
    ///
    /// The roots count: they go after every other checkpoint, so once they
    /// hold all places but one, the conversations have that one place, as
    /// with a budget of one.
    pub(crate) fn save_end(&mut self, slot: u32, hash: ChainHash, owner: Owner) -> Saved {
        let in_ram = self.in_ram.as_ref().map_or(0, RamLevel::checkpoints);
        if in_ram == 0 && self.index.checkpoints() > 0 {
            let ranks = self.index.checkpoint_ranks(&self.policy);
            let roots = ranks.iter().filter(|rank| rank.root).count();
            let mut others = ranks.iter().filter(|rank| !rank.root);
            let one = self.policy_now().capacity() <= roots + 1;
            if let (true, Some(only), None) = (one, others.next(), others.next())
                && only.by == Some(owner)
                && !only.aged
            {
                return Saved::default();
            }
        }
        self.save(slot, hash, owner, false)
    }

    /// Queues the restore of `id` into `slot` for the next record.
    pub(crate) fn restore(&mut self, slot: u32, id: CheckpointId) {
        self.restores.push(StateCopy { kind: StateCopyKind::Restore, slot, place: id.0 as u32 });
    }

    /// Drops the restore queued for `slot`: the sequence it was for left the
    /// devices before the record that carries it was written, and the slot
    /// may be taken by another. Returns whether there was one.
    pub(crate) fn cancel_restore(&mut self, slot: u32) -> bool {
        let before = self.restores.len();
        self.restores.retain(|copy| copy.slot != slot);
        if self.restores.len() == before {
            return false;
        }
        // Places that were waiting for it are free if nothing else reads them.
        let restores = &self.restores;
        let (free, waiting): (Vec<u32>, Vec<u32>) = std::mem::take(&mut self.deferred)
            .into_iter()
            .partition(|place| !restores.iter().any(|copy| copy.place == *place));
        self.settling.extend(&free);
        self.free_places.extend(free);
        self.deferred = waiting;
        true
    }

    /// Queues the copy of a shared page for the next record.
    pub(crate) fn copy_page(&mut self, from: PageId, to: PageId, cells: u32) {
        self.page_copies.push(PageCopy { from: from.0, to: to.0, cells });
    }

    /// A page or tail that the index gave up outside of [`Cache::evict`]
    /// (one tail too many) leaves the devices. Returns whether a checkpoint
    /// went with it, which happens only without RAM below the devices.
    pub(crate) fn forget(&mut self, evicted: Evicted) -> bool {
        self.leave_cards(evicted, &HashSet::new())
    }

    /// Gives up `pages` cached pages from the ends of chains. They go to
    /// RAM, with their checkpoints, before the next record.
    pub(crate) fn evict(&mut self, pool: &mut PagePool, pages: u32) -> Evictions {
        let left_behind = self.left_behind();
        let mut given = Evictions::default();
        for _ in 0..pages {
            let Some(evicted) = self.index.evict(pool) else { break };
            given.pages += 1;
            given.checkpoints += u32::from(self.leave_cards(evicted, &left_behind));
        }
        if given.pages > 0 {
            self.evictions += 1;
        }
        given
    }

    /// What the index gave up leaves the devices: into RAM, where a copy of
    /// it may lie already. Returns whether a checkpoint went with it, which
    /// happens only without RAM below the devices.
    fn leave_cards(&mut self, evicted: Evicted, left_behind: &HashSet<ChainHash>) -> bool {
        let Evicted { page, checkpoint, root, hash, parent, depth, tail, .. } = evicted;
        let full = tail.is_none();
        let deep = full && self.deep(depth);
        let worth = full && (self.worth.remove(&hash) || deep);
        let stored = self.ssd.as_ref().is_some_and(|ssd| ssd.has_or_queued(Kind::State, hash));
        let held = checkpoint.map(|id| Held {
            place: id.0 as u32,
            worth: deep && !left_behind.contains(&hash),
            stored,
            root,
        });
        let Some(ram) = &mut self.in_ram else {
            // Nothing below the devices: the checkpoint goes with its page.
            if let Some(held) = held {
                self.give_up(held.place, None);
                return true;
            }
            return false;
        };
        match ram.get(hash) {
            // RAM has the page already: nothing to copy.
            Some(entry) => {
                let spare = held.filter(|_| entry.checkpoint.is_some());
                ram.left_the_cards(hash, |entry| {
                    entry.worth |= worth;
                    if entry.checkpoint.is_none() {
                        entry.checkpoint = held;
                    }
                });
                if let Some(held) = spare {
                    self.give_up(held.place, None);
                }
            }
            None => {
                let leaving = Leaving { page, hash, parent, depth, tail, checkpoint: held, worth };
                self.leaving.push(leaving);
            }
        }
        false
    }

    /// Tells the cache that the sequence of `owner` holds `settled` cells
    /// whose full pages are published under the names of `chain`, and with
    /// `last` that it ends here. Once it is long enough for the drive, all
    /// its pages are worth writing when they leave RAM.
    pub(crate) fn note(&mut self, owner: Owner, chain: &Chain, settled: u64, last: bool) {
        let Some(ssd) = &self.ssd else { return };
        if settled < ssd.min_cells() {
            if last {
                self.cursors.remove(&owner);
            }
            return;
        }
        let full = ((settled / u64::from(self.cells_per_page)) as usize).min(chain.hashes().len());
        let from = self.cursors.get(&owner).copied().unwrap_or(0).min(full);
        for hash in &chain.hashes()[from..full] {
            match &mut self.in_ram {
                Some(ram) if ram.get(*hash).is_some() => {
                    ram.update(*hash, |entry| entry.worth = true);
                }
                _ => {
                    self.worth.insert(*hash);
                }
            }
        }
        match last {
            true => self.cursors.remove(&owner),
            false => self.cursors.insert(owner, full),
        };
    }

    /// Moves what has left the devices into RAM, and checkpoints that leave
    /// RAM out of their places for the drive: before anything writes the
    /// pool or the places again. Call it before every record.
    pub(crate) fn settle_leaving(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        if self.leaving.is_empty() && self.states_out.is_empty() {
            return;
        }
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        self.push_leaving(pool, exec, &mut batch, &mut notes);
        self.carry_out(pool, exec, batch, notes);
    }

    /// Adds the copies of what is leaving to `batch`, in front of anything
    /// that is added after them. When what it needs is held by transfers of
    /// the batch, the batch so far is carried out first: the order of the
    /// transfers stays what it was.
    fn push_leaving(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        batch: &mut Vec<Transfer>,
        notes: &mut Vec<Note>,
    ) {
        for leaving in std::mem::take(&mut self.leaving) {
            let precious = leaving.worth
                && self
                    .ssd
                    .as_ref()
                    .is_some_and(|ssd| !ssd.has_or_queued(Kind::Page, leaving.hash));
            match self.page_room(pool, exec, batch, notes, precious) {
                Some(parts) => {
                    batch.push(Transfer {
                        kind: TransferKind::PageOut,
                        object: leaving.page.0,
                        keep_form: false,
                        parts,
                        ok: false,
                    });
                    notes.push(Note::ToRam(leaving));
                }
                None => self.lost(leaving),
            }
        }
        // Checkpoints out of their places, for the drive. Making room for
        // the pages above may have added some.
        for out in std::mem::take(&mut self.states_out) {
            match self.state_room(pool, exec, batch, notes) {
                Some(parts) => {
                    batch.push(Transfer {
                        kind: TransferKind::StateOut,
                        object: out.place,
                        keep_form: false,
                        parts,
                        ok: false,
                    });
                    notes.push(Note::StateOut(out));
                }
                None => self.state_not_written(out),
            }
        }
    }

    /// A page that left the devices found no room in RAM: it is gone, and so
    /// is its checkpoint, which goes to the drive when it is due there.
    fn lost(&mut self, leaving: Leaving) {
        if let Some(ram) = &mut self.in_ram
            && leaving.tail.is_none()
        {
            ram.counters.pages_dropped += 1;
        }
        if let Some(held) = leaving.checkpoint {
            let out = self.state_out_of(&held, leaving.hash, leaving.parent, leaving.depth);
            self.give_up(held.place, out);
        }
    }

    /// A buffer for a page in RAM: one that holds nothing, a new one when the
    /// allowance has room, or one that something which goes for nothing gave
    /// up. Never waits.
    fn ram_buffer(&mut self) -> Option<Vec<crate::exec::TransitPart>> {
        let mut dropped = 0;
        loop {
            let ram = self.in_ram.as_mut()?;
            if let Some(parts) = ram.spare() {
                return Some(parts);
            }
            // The floor of pages is the level's whatever else the allowance
            // holds; and the room in the limit that saves of the next record
            // will lock is theirs.
            let (bytes, page) = (ram.bytes(), ram.page_bytes());
            let room = self.held() + page <= self.allowance() || bytes + page <= self.page_floor;
            let unbooked = self.books.as_ref().map_or(u64::MAX, PinnedRam::unbooked);
            if room
                && unbooked >= page + self.pending_locks()
                && let Ok(parts) = self.in_ram.as_mut()?.make()
            {
                return Some(parts);
            }
            if !self.give_up_free(&mut dropped) {
                return None;
            }
        }
    }

    /// [`Cache::ram_buffer`] for a page that is added to `batch`. A
    /// `precious` page (one that would be lost otherwise) waits for room when
    /// there is no other way: the buffers in the batch so far go into RAM
    /// first, and the writer writes what only RAM holds, briefly.
    fn page_room(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        batch: &mut Vec<Transfer>,
        notes: &mut Vec<Note>,
        precious: bool,
    ) -> Option<Vec<crate::exec::TransitPart>> {
        loop {
            if let Some(parts) = self.ram_buffer() {
                return Some(parts);
            }
            if !precious || self.ssd.is_none() {
                return None;
            }
            if !batch.is_empty() {
                self.carry_out(pool, exec, std::mem::take(batch), std::mem::take(notes));
                continue;
            }
            self.clean_ahead();
            if !self.wait_for_writer() {
                return None;
            }
        }
    }

    /// Gives what only RAM holds to the writer, the coldest first, until it
    /// is no more than the allowance less room for [`RESERVE`] pages and one
    /// checkpoint: what comes next then finds room without waiting for the
    /// drive.
    fn clean_ahead(&mut self) {
        if self.ssd.is_none() {
            return;
        }
        let Some(ram) = &self.in_ram else { return };
        let reserve = RESERVE * ram.page_bytes() + self.bytes;
        let room = self.allowance().saturating_sub(reserve);
        // Every turn takes one thing out of the dirty part of the order.
        for _ in 0..=ram.len() * 2 {
            let Some(ram) = &self.in_ram else { return };
            let index = self.index.checkpoints() as u64 * self.bytes;
            if ram.dirty_bytes(self.bytes) + index <= room {
                return;
            }
            let Some((_, item, hash)) = ram.first_from(Class::Dirty) else { return };
            self.clean(item, hash);
        }
    }

    /// Writes the page or the checkpoint `hash` that only RAM holds, which
    /// stays in RAM: what the writer gives back is on the drive too.
    fn clean(&mut self, item: Item, hash: ChainHash) {
        let (Some(ram), Some(ssd)) = (&mut self.in_ram, &mut self.ssd) else { return };
        let Some(entry) = ram.get(hash) else { return };
        let (parent, depth) = (entry.parent, entry.depth);
        match item {
            Item::Page => {
                if ssd.has_or_queued(Kind::Page, hash) {
                    ram.update(hash, |entry| entry.stored = true);
                    return;
                }
                let Some(parts) = ram.take(hash) else { return };
                // A text on the drive is found from its first page on: what
                // comes before it is worth writing too.
                match ram.get(parent).map(|entry| entry.worth) {
                    Some(false) => ram.update(parent, |entry| entry.worth = true),
                    Some(true) => {}
                    None => {
                        if self.index.page(parent).is_some() {
                            self.worth.insert(parent);
                        }
                    }
                }
                ssd.write(Kind::Page, hash, parent, depth, parts);
            }
            Item::Checkpoint => {
                let Some(held) = entry.checkpoint else { return };
                if ssd.has_or_queued(Kind::State, hash) {
                    ram.update(hash, |entry| {
                        if let Some(held) = &mut entry.checkpoint {
                            held.stored = true;
                        }
                    });
                    return;
                }
                ssd.queue(Kind::State, hash);
                // It counts as on the drive from now on: it is copied out
                // before its place can be written again, and a write that
                // fails costs the drive this checkpoint and nothing else.
                ram.update(hash, |entry| {
                    if let Some(held) = &mut entry.checkpoint {
                        held.stored = true;
                    }
                });
                let out = StateOut { place: held.place, hash, parent, depth, keep: true };
                self.states_out.push(out);
            }
        }
    }

    /// Gives up what the allowance of the books does not hold any more: what
    /// goes for nothing first, then what only RAM holds, which is written
    /// first (its memory goes when the writer is done), then the coldest
    /// checkpoints of the index. Returns the checkpoints that went. Call it
    /// between steps only, after [`Cache::pump`]: when the allowance fell,
    /// also while nothing runs.
    pub(crate) fn trim(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) -> u32 {
        if self.books.is_none() {
            return 0;
        }
        self.settle_leaving(pool, exec);
        self.settle();
        let target = self.allowance();
        let mut dropped = 0;
        while self.held() > target && self.give_up_free(&mut dropped) {}
        // What is being written goes once it is: it counts as gone.
        let mut going = self.in_ram.as_ref().map_or(0, RamLevel::away_bytes);
        while self.held().saturating_sub(going) > target {
            let next = self.in_ram.as_ref().and_then(|ram| ram.first_from(Class::Dirty));
            match next {
                Some((_, Item::Page, hash)) => {
                    going += self.in_ram.as_ref().map_or(0, RamLevel::page_bytes);
                    self.clean(Item::Page, hash);
                }
                Some((_, Item::Checkpoint, hash)) => {
                    dropped += self.ram_give_up(Item::Checkpoint, hash);
                }
                None if self.give_up_index_checkpoint(false) => dropped += 1,
                None => break,
            }
        }
        // The checkpoints that went are copied out of their places for the
        // drive before their places can be called unused.
        self.settle_leaving(pool, exec);
        // Nothing runs: a freed place that no copy of the next record uses
        // is unused already, and the others wait for that record.
        let (saves, restores) = (&self.saves, &self.restores);
        let (waiting, unused): (Vec<u32>, Vec<u32>) = std::mem::take(&mut self.settling)
            .into_iter()
            .partition(|place| saves.iter().chain(restores).any(|copy| copy.place == *place));
        for place in unused {
            if self.free_places.contains(&place) {
                self.mark(place, false);
            }
        }
        self.settling = waiting;
        dropped
    }

    /// Copies what RAM holds after the `found` pages of the index for a
    /// prompt of `tokens` (named in `chain`) back to the devices, where it
    /// enters the index. For a model with recurrent state the run ends at
    /// the deepest checkpoint it has, unless a tail with the state at its
    /// end follows or the drive goes on to a deeper one. Returns the pages
    /// and tails that came.
    fn import_from_ram(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        chain: &Chain,
        tokens: &[u64],
        found: &PrefixMatch,
    ) -> u32 {
        let Some(ram) = &self.in_ram else { return 0 };
        let per_page = self.cells_per_page as usize;
        let looked = &tokens[..tokens.len() - 1];
        let have = found.pages.len();
        let total = (looked.len() / per_page).min(chain.hashes().len());
        let names = &chain.hashes()[have.min(total)..total];
        let run = names
            .iter()
            .take_while(|hash| ram.ready(**hash) && self.index.state(**hash) == PageState::Unknown)
            .count();
        // A tail where the run ends, unless the index has a better one.
        let rest = &looked[(have + run) * per_page..];
        let tail = (found.tail.is_none() || run > 0)
            .then(|| {
                ram.tails_under(chain.before(have + run))
                    .map(|(hash, entry)| {
                        let tokens = entry.tail.as_deref().unwrap_or(&[]);
                        let shared = tokens.iter().zip(rest).take_while(|(a, b)| a == b).count();
                        (shared, shared == tokens.len(), entry.checkpoint.is_some(), hash)
                    })
                    .filter(|(shared, ..)| *shared > 0)
                    .max_by_key(|(shared, whole, ..)| (*shared, *whole))
            })
            .flatten();
        let (run, tail) = if self.recurrent {
            let deepest = (0..run)
                .rev()
                .find(|i| ram.get(names[*i]).is_some_and(|entry| entry.checkpoint.is_some()));
            let drive = self.ssd.as_ref().is_some_and(|ssd| ssd.reaches_state(&names[run..]));
            match tail {
                Some((_, true, true, hash)) => (run, Some(hash)),
                _ if drive => (run, None),
                _ => (deepest.map_or(0, |at| at + 1), None),
            }
        } else {
            (run, tail.map(|(.., hash)| hash))
        };
        if run == 0 && tail.is_none() {
            return 0;
        }
        // What the index matched is held, so that room made for the run does
        // not take it; the run is taken out of RAM's order first, so that
        // room made for pages that leave the devices does not take it either.
        let kept = Sequence::from_shared(pool, &found.table(), found.cells(per_page as u32));
        let run: Vec<(ChainHash, ChainHash, u32)> =
            (0..run).map(|i| (names[i], chain.before(have + i), (have + i) as u32 + 1)).collect();
        let tail = tail.map(|hash| {
            let entry = self.in_ram.as_ref().and_then(|ram| ram.get(hash)).expect("listed above");
            (hash, entry.parent, entry.tail.as_ref().map_or(0, Vec::len))
        });
        let mut taken: Vec<(ChainHash, ChainHash, u32, Vec<crate::exec::TransitPart>)> = Vec::new();
        let ram = self.in_ram.as_mut().expect("checked above");
        for (hash, parent, cells) in run
            .iter()
            .map(|(hash, parent, _)| (*hash, *parent, per_page as u32))
            .chain(tail.map(|(hash, parent, cells)| (hash, parent, cells as u32)))
        {
            match ram.take(hash) {
                Some(parts) => taken.push((hash, parent, cells, parts)),
                None => break,
            }
        }
        let owner = Owner(u64::MAX / 2 - self.next_import);
        self.next_import += 1;
        let settle = u64::from(self.durable.as_ref().map_or(0, |durable| durable.settle_cells));
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        let mut held = Sequence::new();
        let mut left = Vec::new();
        let count = taken.len();
        for (i, (hash, parent, cells, parts)) in taken.into_iter().enumerate() {
            let mut grown = held.append(pool, u64::from(cells));
            if grown.is_err() {
                // Room in the pool: a cached page nobody holds leaves the
                // devices, and its copy into RAM runs before this one.
                self.evict(pool, 1);
                self.push_leaving(pool, exec, &mut batch, &mut notes);
                grown = held.append(pool, u64::from(cells));
            }
            if grown.is_err() {
                left.push((hash, parts));
                continue;
            }
            let page = *held.pages().last().expect("a page was appended");
            // The first page and the pages among the newest cells of the run
            // keep the form they have; the others take the one a sequence
            // that went on would have given them.
            let tail_page = tail.is_some() && i + 1 == count;
            let behind = (count - 1 - i) as u64 * per_page as u64;
            let depth = run.get(i).map_or(0, |(_, _, depth)| *depth);
            let keep_form = tail_page || depth == 1 || behind <= settle + per_page as u64;
            batch.push(Transfer {
                kind: TransferKind::PageIn,
                object: page.0,
                keep_form,
                parts,
                ok: false,
            });
            notes.push(Note::FromRam { hash, parent, owner });
        }
        if let Some(ram) = &mut self.in_ram {
            for (hash, parts) in left {
                ram.give(hash, parts);
            }
        }
        let read = self.in_ram.as_ref().map_or(0, |ram| ram.counters.pages_read);
        self.carry_out(pool, exec, batch, notes);
        held.release(pool);
        kept.release(pool);
        (self.in_ram.as_ref().map_or(0, |ram| ram.counters.pages_read) - read) as u32
    }

    /// Carries out `batch` and what it came to.
    fn carry_out(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        mut batch: Vec<Transfer>,
        notes: Vec<Note>,
    ) {
        if batch.is_empty() {
            return;
        }
        let begun = Instant::now();
        if exec.transfer(&mut batch).is_err() {
            for transfer in &mut batch {
                transfer.ok = false;
            }
        }
        let ns = begun.elapsed().as_nanos() as u64;
        if let Some(ram) = &mut self.in_ram {
            ram.counters.transfers += 1;
            ram.counters.transfer_ns += ns;
        }
        if let Some(ssd) = &mut self.ssd {
            ssd.counters.transfers += 1;
            ssd.counters.transfer_ns += ns;
        }
        self.finish(pool, batch, notes);
    }

    /// What the transfers of a batch came to, in their order.
    fn finish(&mut self, pool: &mut PagePool, batch: Vec<Transfer>, notes: Vec<Note>) {
        let mut state_in: Option<(u64, u32, bool)> = None;
        for (transfer, note) in batch.into_iter().zip(notes) {
            let Transfer { object, parts, ok, .. } = transfer;
            match note {
                Note::ToRam(leaving) => {
                    let Some(ram) = &mut self.in_ram else { continue };
                    if !ok {
                        ram.recycle(parts);
                        self.lost(leaving);
                        continue;
                    }
                    let Leaving { hash, parent, depth, tail, checkpoint, worth, .. } = leaving;
                    let stored =
                        self.ssd.as_ref().is_some_and(|ssd| ssd.has_or_queued(Kind::Page, hash));
                    let mut entry = Entry::new(parent, depth, tail);
                    entry.checkpoint = checkpoint;
                    entry.worth = worth;
                    entry.stored = stored;
                    ram.counters.pages_in += 1;
                    if ram.get(hash).is_some() {
                        // A copy came in meanwhile: this one is not needed,
                        // and the checkpoint goes on the one that is there.
                        ram.recycle(parts);
                        let spare = checkpoint.filter(|_| {
                            ram.get(hash).is_some_and(|entry| entry.checkpoint.is_some())
                        });
                        ram.update(hash, |entry| {
                            if entry.checkpoint.is_none() {
                                entry.checkpoint = checkpoint;
                            }
                        });
                        if let Some(held) = spare {
                            self.give_up(held.place, None);
                        }
                        continue;
                    }
                    ram.insert(hash, entry, parts);
                }
                Note::FromRam { hash, parent, owner } => {
                    let Some(ram) = &mut self.in_ram else { continue };
                    ram.give(hash, parts);
                    // A run from RAM enters the index in order, and ends at
                    // a page that did not come: what follows it would hang
                    // on nothing.
                    let follows = parent == self.root || self.index.page(parent).is_some();
                    if !ok || !follows {
                        continue;
                    }
                    let tail = ram.get(hash).and_then(|entry| entry.tail.clone());
                    let mut held = None;
                    ram.update(hash, |entry| {
                        entry.on_cards = true;
                        held = entry.checkpoint.take();
                    });
                    ram.counters.pages_read += 1;
                    match tail {
                        None => {
                            self.index.insert(pool, parent, hash, PageId(object));
                        }
                        Some(tokens) => {
                            let page = PageId(object);
                            let dropped = self.index.insert_tail(
                                pool,
                                parent,
                                &tokens,
                                page,
                                Priority::NORMAL,
                            );
                            if let Some(evicted) = dropped {
                                self.leave_cards(evicted, &HashSet::new());
                            }
                        }
                    }
                    if let Some(held) = held {
                        let id = CheckpointId(u64::from(held.place));
                        match self.index.attach_checkpoint(hash, id, owner) {
                            Ok(before) => {
                                if held.root {
                                    self.index.mark_root(hash);
                                }
                                if let Some(before) = before {
                                    self.release(before);
                                }
                            }
                            Err(_) => self.give_up(held.place, None),
                        }
                    }
                }
                Note::CardsOut { hash, parent, depth } => {
                    let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                    match ok {
                        true => ssd.write(Kind::Page, hash, parent, depth, parts),
                        false => {
                            ssd.unqueue(Kind::Page, hash);
                            if let Some(ram) = &mut self.in_ram {
                                ram.recycle(parts);
                            }
                        }
                    }
                }
                Note::StateOut(out) => {
                    let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                    match ok {
                        true => ssd.write(Kind::State, out.hash, out.parent, out.depth, parts),
                        false => {
                            ssd.unqueue(Kind::State, out.hash);
                            ssd.free_states.push(parts);
                            ssd.counters.states_skipped += 1;
                        }
                    }
                }
                Note::StateIn { job } => {
                    state_in = Some((job, object, ok));
                    if let Some(ssd) = &mut self.ssd {
                        ssd.free_states.push(parts);
                    }
                }
                Note::PageIn { job, index } => {
                    let state = state_in.take_if(|(of, ..)| *of == job);
                    self.page_in(pool, job, index, object, parts, ok, state);
                }
            }
        }
    }

    /// The pages and checkpoints in RAM that hang on the index's pages and
    /// tails: for the counters.
    pub(crate) fn ram_counters(&self) -> RamCounters {
        self.in_ram.as_ref().map_or_else(RamCounters::default, RamLevel::snapshot)
    }

    /// Writes the copies for this record and starts the next ones afresh.
    /// What is leaving must have left before ([`Cache::settle_leaving`]).
    pub(crate) fn write_copies(&mut self, writer: &mut RecordWriter<'_>) -> Result<(), RecordFull> {
        debug_assert!(self.leaving.is_empty() && self.states_out.is_empty());
        // The record before this one has run.
        self.settle();
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.saves);
        self.scratch.extend_from_slice(&self.restores);
        writer.copies(&self.scratch, &self.page_copies)?;
        self.saves.clear();
        self.restores.clear();
        self.page_copies.clear();
        // What was freed while this record was planned, and the places whose
        // restores run with it, are unused once it has run.
        self.ran = std::mem::take(&mut self.settling);
        self.ran.extend(&self.deferred);
        self.free_places.append(&mut self.deferred);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ares_state::Priority;

    use super::*;

    const PER_PAGE: u32 = 4;

    /// A cache for a model whose checkpoints take 100 bytes, with room for
    /// `places` of them.
    fn cache(places: u64) -> Cache {
        let config = RuntimeConfig { cache_ram: places * 100, ..RuntimeConfig::default() };
        Cache::new(
            "m",
            &PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: None },
            None,
            &config,
            PER_PAGE,
        )
    }

    /// Three pages of one sequence in the index, which has let go of them.
    fn three_pages(cache: &mut Cache, pool: &mut PagePool) -> Vec<ChainHash> {
        let tokens: Vec<u64> = (0..12).collect();
        let mut chain = cache.chain(&tokens);
        let mut seq = Sequence::new();
        seq.append(pool, 12).unwrap();
        cache.index.publish(pool, &mut seq, &mut chain, &tokens, 12, Priority::NORMAL);
        seq.release(pool);
        chain.hashes().to_vec()
    }

    fn written(cache: &mut Cache) -> (Vec<StateCopy>, Vec<PageCopy>) {
        let mut buffer = vec![0u8; 1024];
        let mut writer = RecordWriter::new(&mut buffer, 1, 0, 0).unwrap();
        cache.write_copies(&mut writer).unwrap();
        let used = writer.finish().unwrap();
        let record = crate::RecordView::new(&buffer[..used]).unwrap();
        (
            (0..record.state_copies()).map(|i| record.state_copy(i)).collect(),
            (0..record.page_copies()).map(|i| record.page_copy(i)).collect(),
        )
    }

    #[test]
    fn a_root_keeps_its_place_while_the_other_checkpoints_take_turns() {
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(2);
        let hashes = three_pages(&mut cache, &mut pool);
        // The end of a system prompt, then two more checkpoints of the same
        // request: the budget holds two, and the plain one goes.
        assert!(cache.save(0, hashes[0], Owner(1), true).taken);
        assert!(cache.save(0, hashes[1], Owner(1), false).taken);
        let third = cache.save(0, hashes[2], Owner(1), false);
        assert_eq!((third.taken, third.dropped), (true, 1));
        assert!(cache.index.is_root(hashes[0]));
        assert_eq!(cache.index.checkpoint(hashes[1]), None);
        // A prompt that finds the root as another text's plain checkpoint
        // marks it.
        assert!(!cache.save(1, hashes[2], Owner(2), true).taken);
        assert!(cache.index.is_root(hashes[2]));
    }

    #[test]
    fn with_one_place_an_end_state_does_not_push_out_its_own_anchor() {
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(1);
        let hashes = three_pages(&mut cache, &mut pool);
        assert!(cache.save(0, hashes[1], Owner(1), false).taken);
        assert!(!cache.save_end(0, hashes[2], Owner(1)).taken);
        assert!(cache.index.checkpoint(hashes[1]).is_some());
        // Another request's end state takes the place as before.
        let other = cache.save_end(1, hashes[2], Owner(2));
        assert_eq!((other.taken, other.dropped), (true, 1));
        // With room for two the ranking decides as for any checkpoint.
        let mut roomy = super::tests::cache(2);
        let hashes = three_pages(&mut roomy, &mut pool);
        assert!(roomy.save(0, hashes[0], Owner(1), false).taken);
        assert!(roomy.save(0, hashes[1], Owner(1), false).taken);
        let end = roomy.save_end(0, hashes[2], Owner(1));
        assert_eq!((end.taken, end.dropped), (true, 1));
    }

    #[test]
    fn with_one_place_besides_the_roots_an_end_state_does_not_push_out_its_own_anchor() {
        // The agent suite: the roots of the system prompt keep their places,
        // and the one left over held the end state of a prompt instead of its
        // anchor, so the same prompt asked again replayed from the roots.
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(2);
        let hashes = three_pages(&mut cache, &mut pool);
        assert!(cache.save(0, hashes[0], Owner(1), true).taken);
        assert!(cache.save(0, hashes[1], Owner(1), false).taken);
        assert!(!cache.save_end(0, hashes[2], Owner(1)).taken);
        assert!(cache.index.is_root(hashes[0]));
        assert!(cache.index.checkpoint(hashes[1]).is_some());
        // Another request's end state takes the place as before; the root stays.
        let other = cache.save_end(1, hashes[2], Owner(2));
        assert_eq!((other.taken, other.dropped), (true, 1));
        assert!(cache.index.checkpoint(hashes[0]).is_some());
        assert_eq!(cache.index.checkpoint(hashes[1]), None);
    }

    #[test]
    fn a_place_whose_checkpoint_is_dropped_waits_for_the_restore_that_reads_it() {
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(1);
        let hashes = three_pages(&mut cache, &mut pool);
        assert!(cache.save(0, hashes[0], Owner(1), false).taken);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Save, slot: 0, place: 0 }]);

        // A start restores the place with the next record. Before that record
        // another checkpoint is taken: the budget holds one, so the first is
        // dropped from the index, and its place is not reused yet.
        cache.restore(3, CheckpointId(0));
        let second = cache.save(1, hashes[1], Owner(2), false);
        assert_eq!((second.taken, second.dropped), (false, 1), "the only place is still read");
        assert_eq!(cache.index.checkpoint(hashes[0]), None);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Restore, slot: 3, place: 0 }]);

        // With the record written the place is free for the next checkpoint.
        assert!(cache.save(1, hashes[1], Owner(2), false).taken);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Save, slot: 1, place: 0 }]);
    }

    #[test]
    fn a_restore_that_is_cancelled_frees_the_place_that_waited_for_it() {
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(1);
        let hashes = three_pages(&mut cache, &mut pool);
        assert!(cache.save(0, hashes[0], Owner(1), false).taken);
        written(&mut cache);
        cache.restore(3, CheckpointId(0));
        assert!(!cache.save(1, hashes[1], Owner(2), false).taken);
        // The sequence that was to restore it ended before the record: the
        // restore is not carried out, and the place is free.
        assert!(cache.cancel_restore(3));
        assert!(!cache.cancel_restore(3));
        assert!(cache.save(1, hashes[1], Owner(2), false).taken);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Save, slot: 1, place: 0 }]);
    }

    #[test]
    fn a_model_that_keeps_state_per_tail_limits_the_tails_per_slot() {
        let config = RuntimeConfig { max_slots: 2, ..RuntimeConfig::default() };
        let spec = PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: Some(1) };
        let mut cache = Cache::new("m", &spec, None, &config, PER_PAGE);
        let mut pool = PagePool::new(16, PER_PAGE);
        let root = ChainHash::root(b"m");
        let mut dropped = Vec::new();
        for i in 0..3u32 {
            let mut seq = Sequence::new();
            seq.append(&mut pool, 2).unwrap();
            let prior = root.extend(&[u64::from(i)]);
            dropped.push(cache.index.insert_tail(
                &mut pool,
                prior,
                &[7, 8],
                seq.pages()[0],
                Priority::NORMAL,
            ));
            seq.release(&mut pool);
        }
        assert_eq!(cache.index.tails(), 2, "one tail for each of the two slots");
        assert_eq!(dropped.iter().filter(|d| d.is_some()).count(), 1);

        // No limit without the model asking for one.
        let spec = PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: None };
        let mut cache = Cache::new("m", &spec, None, &config, PER_PAGE);
        for i in 0..3u32 {
            let mut seq = Sequence::new();
            seq.append(&mut pool, 2).unwrap();
            cache.index.insert_tail(
                &mut pool,
                root.extend(&[u64::from(i)]),
                &[7, 8],
                seq.pages()[0],
                Priority::NORMAL,
            );
            seq.release(&mut pool);
        }
        assert_eq!(cache.index.tails(), 3);
    }

    #[test]
    fn eviction_gives_up_the_checkpoints_of_the_pages_it_takes_and_counts_the_rounds() {
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(1);
        let hashes = three_pages(&mut cache, &mut pool);
        assert!(cache.save(0, hashes[2], Owner(1), false).taken);
        written(&mut cache);
        // The chain goes from its end: the last page takes its checkpoint, and
        // its place is free for the next one.
        let given = cache.evict(&mut pool, 1);
        assert_eq!((given.pages, given.checkpoints), (1, 1));
        assert_eq!(cache.evictions, 1);
        assert!(cache.save(5, hashes[1], Owner(1), false).taken);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Save, slot: 5, place: 0 }]);
        assert_eq!(cache.evict(&mut pool, 5).pages, 2, "two pages are left");
        assert_eq!(cache.evict(&mut pool, 5).pages, 0);
        assert_eq!(cache.evictions, 2, "an eviction that found nothing is not one");
        assert_eq!(pool.free_pages(), 8);
    }

    // --- the levels in RAM and on SSD, and the allowance of the books ------

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use crate::exec::{Clock, ExecError, StepMeta, StepOutput};
    use crate::ram::{Machine, Rule};

    const PLENTY: u64 = 64 << 30;
    /// A checkpoint, and the share of a page on the one device, which takes
    /// a block in RAM.
    const STATE: u64 = 16 << 10;
    const PAGE: u64 = ares_store::BLOCK as u64;

    /// A cache with a level on SSD in a directory of its own, for a model
    /// whose checkpoints take [`STATE`] bytes, with a budget of four places
    /// and the floor of pages, in books that follow a machine whose
    /// available memory the test sets.
    struct Following {
        cache: Cache,
        pool: PagePool,
        hashes: Vec<ChainHash>,
        ram: PinnedRam,
        machine: Arc<Machine>,
        available: Arc<AtomicU64>,
        dir: std::path::PathBuf,
    }

    impl Following {
        fn new(tag: &str) -> Self {
            Self::with(tag, true, 4 * STATE + RESERVE * PAGE)
        }

        fn with(tag: &str, ssd: bool, budget: u64) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("ares-cache-following-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let available = Arc::new(AtomicU64::new(PLENTY));
            let source = available.clone();
            let machine = Machine::new(
                Rule::default(),
                move || Some(source.load(Ordering::SeqCst)),
                || Duration::ZERO,
            );
            let ram = PinnedRam::following(1 << 30, &machine);
            let durable = Durable {
                fingerprint: b"cache tests".to_vec(),
                page_bytes: vec![64],
                state_bytes: vec![STATE],
                settle_cells: 0,
            };
            let config = RuntimeConfig {
                cache_ram: budget,
                // A text of one page is deep enough to be written.
                ssd: ssd
                    .then(|| SsdConfig { min_cells: PER_PAGE, ..SsdConfig::new(&dir, 1 << 30) }),
                ..RuntimeConfig::default()
            };
            let spec = PrefixCache { checkpoint_bytes: vec![STATE], tails_per_slot: None };
            let mut cache = Cache::new("m", &spec, Some(&durable), &config, PER_PAGE)
                .with_ram(Some(ram.clone()));
            assert!(cache.ssd_problem.is_none(), "{:?}", cache.ssd_problem);
            let mut pool = PagePool::new(64, PER_PAGE);
            let hashes = three_pages(&mut cache, &mut pool);
            Self { cache, pool, hashes, ram, machine, available, dir }
        }

        /// Text `i` of three pages, which the index holds and nobody else.
        fn text(&mut self, i: u32) -> Vec<ChainHash> {
            let tokens: Vec<u64> = (0..12).map(|t| 1000 * (u64::from(i) + 1) + t).collect();
            let mut chain = self.cache.chain(&tokens);
            let mut seq = Sequence::new();
            seq.append(&mut self.pool, 12).unwrap();
            let pool = &mut self.pool;
            self.cache.index.publish(pool, &mut seq, &mut chain, &tokens, 12, Priority::NORMAL);
            seq.release(&mut self.pool);
            chain.hashes().to_vec()
        }

        /// The devices give up every page nobody holds: into RAM.
        fn all_into_ram(&mut self, devices: &mut Devices) {
            self.cache.evict(&mut self.pool, u32::MAX);
            self.cache.settle_leaving(&mut self.pool, devices);
        }

        /// The machine has `available` bytes from now on, and has been
        /// looked at.
        fn machine_has(&self, available: u64) {
            self.available.store(available, Ordering::SeqCst);
            self.machine.look().expect("the machine says what it has");
        }

        /// Pumps the levels against `devices`, and returns what they were
        /// asked to carry out.
        fn pumped(&mut self, devices: &mut Devices) -> Vec<(TransferKind, u32)> {
            devices.asked.clear();
            self.cache.pump(&mut self.pool, devices);
            std::mem::take(&mut devices.asked)
        }

        fn trimmed(&mut self, devices: &mut Devices) -> u32 {
            self.cache.trim(&mut self.pool, devices)
        }

        /// Waits until the writer has written what it was given.
        fn written(&mut self) -> SsdCounters {
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            loop {
                let counters = self.cache.ssd_counters();
                if counters.pending == 0 {
                    return counters;
                }
                assert!(std::time::Instant::now() < deadline, "the writer is stuck");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    impl Drop for Following {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Devices that write down the transfers they are asked for. A page that
    /// leaves holds its number; a page that comes in must hold the number of
    /// the page it left from; a checkpoint is whatever its place holds.
    #[derive(Default)]
    struct Devices {
        asked: Vec<(TransferKind, u32)>,
        /// Which page of the pool each page that came in had left from.
        came: Vec<(u32, u32)>,
    }

    impl Executor for Devices {
        fn devices(&self) -> usize {
            1
        }

        fn clock(&self) -> Clock {
            Clock::new()
        }

        fn record(&mut self) -> &mut [u8] {
            &mut []
        }

        fn dispatch(&mut self, _: &StepMeta) -> Result<(), ExecError> {
            Ok(())
        }

        fn collect(&mut self, _: Duration, _: &mut StepOutput) -> Result<crate::Woke, ExecError> {
            Ok(crate::Woke::Spinning)
        }

        fn device_wakes(&self) -> crate::WakeCounts {
            crate::WakeCounts::default()
        }

        fn transfer(&mut self, batch: &mut [Transfer]) -> Result<(), ExecError> {
            for transfer in batch {
                self.asked.push((transfer.kind, transfer.object));
                let part = &mut transfer.parts[0];
                match transfer.kind {
                    TransferKind::PageOut => {
                        part.mem[..4].copy_from_slice(&transfer.object.to_le_bytes());
                        (part.len, part.tag) = (64, 1);
                    }
                    TransferKind::PageIn => {
                        let from = u32::from_le_bytes(part.mem[..4].try_into().unwrap());
                        self.came.push((transfer.object, from));
                    }
                    TransferKind::StateOut => (part.len, part.tag) = (STATE as usize, 2),
                    TransferKind::StateIn => {}
                }
                transfer.ok = true;
            }
            Ok(())
        }

        fn shutdown(&mut self) {}
    }

    fn of(asked: &[(TransferKind, u32)], kind: TransferKind) -> Vec<u32> {
        asked.iter().filter(|(k, _)| *k == kind).map(|(_, object)| *object).collect()
    }

    #[test]
    fn the_cache_keeps_its_floor_when_the_machine_has_no_room() {
        let f = Following::new("floor");
        assert_eq!(f.ram.cache_allowance(), 4 * STATE + RESERVE * PAGE, "all of the budget");
        f.machine_has(0);
        assert_eq!(f.ram.cache_allowance(), STATE + RESERVE * PAGE, "a place and the pages");
        let mut f = f;
        // A checkpoint is still taken, in the one place, and the next one
        // takes the place over.
        assert!(f.cache.save(0, f.hashes[0], Owner(1), false).taken);
        let second = f.cache.save(1, f.hashes[1], Owner(1), false);
        assert_eq!((second.taken, second.dropped), (true, 1));
        assert_eq!(f.cache.index.checkpoint(f.hashes[0]), None);
        assert!(f.cache.index.checkpoint(f.hashes[1]).is_some());
        // The first never ran its save: it had nothing to write.
        assert_eq!(f.cache.ssd_counters().states_skipped, 1);

        // Without a level on SSD the floor is the one place, without pages:
        // the same books still let a checkpoint be taken, and the next one
        // takes the place over.
        let config = RuntimeConfig { cache_ram: 400, ..RuntimeConfig::default() };
        let spec = PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: None };
        let mut plain =
            Cache::new("m", &spec, None, &config, PER_PAGE).with_ram(Some(f.ram.clone()));
        assert_eq!(f.ram.cache_allowance(), 100, "one place");
        let hashes = three_pages(&mut plain, &mut f.pool);
        assert!(plain.save(0, hashes[0], Owner(1), false).taken);
        let second = plain.save(1, hashes[1], Owner(1), false);
        assert_eq!((second.taken, second.dropped), (true, 1));
    }

    #[test]
    fn a_checkpoint_goes_to_the_drive_when_it_leaves_ram_and_not_while_ram_holds_it() {
        let mut f = Following::new("leave");
        let mut devices = Devices::default();
        // One request's checkpoints: neither is left behind by the other.
        assert!(f.cache.save(0, f.hashes[1], Owner(1), false).taken);
        assert!(f.cache.save(1, f.hashes[2], Owner(1), false).taken);
        assert_eq!(written(&mut f.cache).0.len(), 2);
        // RAM holds both: nothing goes to the drive.
        let asked = f.pumped(&mut devices);
        assert!(of(&asked, TransferKind::StateOut).is_empty(), "{asked:?}");
        assert_eq!(f.written().states_written, 0);

        // The machine runs short: the floor is one place. One checkpoint goes,
        // and leaves its place for the drive before the place is called unused.
        f.machine_has(0);
        assert_eq!(f.trimmed(&mut devices), 1);
        let out = of(&devices.asked, TransferKind::StateOut);
        assert_eq!(out.len(), 1, "{:?}", devices.asked);
        assert_eq!(f.written().states_written, 1);
        assert!(!f.ram.place_is_live(out[0]), "unused once it has left");
        let kept = 1 - out[0];
        assert!(f.ram.place_is_live(kept));

        // A checkpoint that a later request has left behind is not written
        // when it goes: a deeper one on its branch is where a prompt resumes.
        let mut f = Following::new("behind");
        let mut devices = Devices::default();
        assert!(f.cache.save(0, f.hashes[1], Owner(1), false).taken);
        assert!(f.cache.save(1, f.hashes[2], Owner(2), false).taken);
        written(&mut f.cache);
        f.machine_has(0);
        assert_eq!(f.trimmed(&mut devices), 1);
        assert!(of(&devices.asked, TransferKind::StateOut).is_empty(), "{:?}", devices.asked);
        assert_eq!(f.cache.index.checkpoint(f.hashes[1]), None, "the one left behind went");
        assert_eq!(f.written().states_written, 0);
    }

    #[test]
    fn a_checkpoint_given_up_before_its_save_ran_is_neither_saved_nor_written() {
        let mut f = Following::new("unsaved");
        let mut devices = Devices::default();
        // A sequence reaches two page boundaries in the same step: two saves
        // wait for the next record.
        assert!(f.cache.save(0, f.hashes[1], Owner(1), false).taken);
        assert!(f.cache.save(1, f.hashes[2], Owner(1), false).taken);
        // The machine runs short before that record is written: the cache
        // keeps the one place of its floor and gives the other checkpoint up.
        f.machine_has(0);
        assert_eq!(f.trimmed(&mut devices), 1);
        assert!(of(&devices.asked, TransferKind::StateOut).is_empty(), "nothing to read");
        assert_eq!(f.cache.ssd_counters().states_skipped, 1);
        let (copies, _) = written(&mut f.cache);
        assert_eq!(copies.len(), 1, "the record carries the one save that is left");
        let gone = 1 - copies[0].place;
        assert!(!f.ram.place_is_live(gone) && f.ram.place_is_live(copies[0].place));
    }

    #[test]
    fn a_page_that_leaves_the_devices_goes_to_ram_and_comes_back_from_there() {
        let mut f = Following::new("pages");
        let mut devices = Devices::default();
        assert!(f.cache.save(0, f.hashes[2], Owner(1), false).taken);
        written(&mut f.cache);
        let tokens: Vec<u64> = (0..13).collect();
        let chain = f.cache.chain(&tokens);
        // The pool gives all three pages up: they are copied out before the
        // next record, the last one with its checkpoint, which stays put.
        let place = f.cache.index.checkpoint(f.hashes[2]).unwrap();
        let given = f.cache.evict(&mut f.pool, 3);
        assert_eq!((given.pages, given.checkpoints), (3, 0));
        f.cache.settle_leaving(&mut f.pool, &mut devices);
        let out = of(&devices.asked, TransferKind::PageOut);
        assert_eq!(out.len(), 3);
        let ram = f.cache.ram_counters();
        assert_eq!((ram.pages, ram.pages_in, ram.checkpoints), (3, 3, 1));
        assert_eq!(f.cache.index.len(), 0);
        // Nothing is written while RAM holds them.
        f.pumped(&mut devices);
        assert_eq!(f.written().pages_written, 0);

        // The text again: its pages come back from RAM when it arrives, into
        // pages of the pool, with the checkpoint in the place it had.
        devices.asked.clear();
        let mut pool = std::mem::replace(&mut f.pool, PagePool::new(1, PER_PAGE));
        let probe = f.cache.probe(&mut pool, Some(&mut devices), &chain, &tokens);
        f.pool = pool;
        assert_eq!(probe.found.pages.len(), 3);
        assert_eq!(probe.resume.restore, Some(place));
        assert_eq!(probe.resume.replay, 12..12);
        assert_eq!(of(&devices.asked, TransferKind::PageIn).len(), 3);
        let copies: Vec<u32> = devices.came.iter().map(|(_, from)| *from).collect();
        let mut left = out.clone();
        left.sort_unstable();
        let mut came = copies.clone();
        came.sort_unstable();
        assert_eq!(came, left, "each page came from the copy of one that left");
        assert_eq!(f.cache.ram_counters().pages_read, 3);
        // RAM keeps its copies: when the pages leave again, nothing is copied.
        devices.asked.clear();
        f.cache.evict(&mut f.pool, 3);
        f.cache.settle_leaving(&mut f.pool, &mut devices);
        assert!(devices.asked.is_empty(), "{:?}", devices.asked);
        assert_eq!(f.cache.ram_counters().checkpoints, 1);
    }

    #[test]
    fn what_leaves_ram_goes_to_the_drive_and_without_one_it_is_dropped() {
        for ssd in [true, false] {
            let mut f = Following::with(&format!("shrink-{ssd}"), ssd, 4 * STATE + 64 * PAGE);
            let mut devices = Devices::default();
            for i in 0..10 {
                f.text(i);
            }
            f.all_into_ram(&mut devices);
            assert_eq!(f.cache.ram_counters().pages, 33);
            assert_eq!(f.ram.cache_pages(), 33 * PAGE);
            assert_eq!(f.written().pages_written, 0, "nothing while RAM holds them");
            // The machine has no room: RAM keeps its floor of pages with a
            // level on SSD, and what goes is written before its memory goes;
            // without one it is dropped at once, down to what the one
            // checkpoint place holds while no checkpoint uses it.
            f.machine_has(0);
            f.trimmed(&mut devices);
            match ssd {
                true => {
                    let floor = STATE + RESERVE * PAGE;
                    let written = f.written();
                    assert!(written.pages_written >= 33 - floor / PAGE, "{written:?}");
                    assert!(written.bytes_written >= written.pages_written * PAGE);
                    f.pumped(&mut devices);
                    f.trimmed(&mut devices);
                    assert!(f.ram.cache_pages() <= floor, "{}", f.ram.cache_pages());
                    assert_eq!(f.cache.ram_counters().pages_dropped, 0);
                }
                false => {
                    assert!(f.ram.cache_pages() <= STATE, "{}", f.ram.cache_pages());
                    assert_eq!(f.cache.ram_counters().pages, f.ram.cache_pages() / PAGE);
                }
            }
        }
    }

    #[test]
    fn the_coldest_of_a_full_ram_is_written_ahead_and_the_rest_waits() {
        // Room for the floor and four pages more: RAM writes what is beyond
        // room for its reserve of sixteen pages.
        let mut f = Following::with("ahead", true, STATE + RESERVE * PAGE + 4 * PAGE);
        let mut devices = Devices::default();
        f.all_into_ram(&mut devices);
        f.pumped(&mut devices);
        let written = f.written();
        assert_eq!(written.pages_written, 0, "three pages and the reserve fit: {written:?}");
        // More texts: RAM fills, and writes the coldest, the first text first.
        for i in 0..8 {
            f.text(i);
            f.all_into_ram(&mut devices);
            f.pumped(&mut devices);
        }
        let written = f.written();
        assert!(written.pages_written > 0, "{written:?}");
        assert!(written.pages_written < 27, "the newest stay unwritten: {written:?}");
        // RAM holds what its budget does; what went had been written.
        let ram = f.cache.ram_counters();
        assert_eq!((ram.pages, ram.pages_dropped), (24, 0), "{ram:?}");
        let ssd = f.cache.ssd.as_ref().unwrap();
        assert!(f.hashes.iter().all(|hash| ssd.has_or_queued(Kind::Page, *hash)), "coldest first");
    }
}
