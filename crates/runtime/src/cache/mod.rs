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
//! - **A floor.** The cache keeps one place (for a model with recurrent
//!   state) whatever the allowance says, so that a conversation resumes from
//!   its last checkpoint at least, and with the level on SSD in use
//!   [`RESERVE`] pages as well: state passes through them on the way to the
//!   drive and back, so a prompt resumes from SSD while the machine has no
//!   room for the cache. That is its floor in the books.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use ares_state::{
    Boundary, Chain, ChainHash, CheckpointId, CheckpointPolicy, Evicted, Owner, PageId, PagePool,
    PageState, PrefixIndex, PrefixMatch, Priority, Resume, Sequence, best_tail,
};
use ares_store::Kind;

use crate::config::RuntimeConfig;
use crate::exec::{Executor, Transfer, TransferKind};
use crate::model::{Durable, FarPages, PrefixCache};
use crate::ram::PinnedRam;
use crate::record::{PageCopy, RecordFull, RecordWriter, StateCopy, StateCopyKind};

mod moves;
mod places;
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

/// How many of `names`, the pages of a prompt after those the index has, a
/// level below the devices can bring back: the leading ones it has (`has`)
/// that the index does not know.
fn run_of(index: &PrefixIndex, names: &[ChainHash], has: impl Fn(ChainHash) -> bool) -> usize {
    names.iter().take_while(|hash| has(**hash) && index.state(**hash) == PageState::Unknown).count()
}

/// The deepest of the first `run` of `names` whose checkpoint the level has
/// too (`has_state`): for a model with recurrent state a run that comes back
/// ends there, since the pages after it would be replayed row by row, which
/// takes as long as computing them.
fn deepest(
    names: &[ChainHash],
    run: usize,
    has_state: impl Fn(ChainHash) -> bool,
) -> Option<usize> {
    (0..run).rev().find(|i| has_state(names[*i]))
}

/// The pages of a run of `run` pages that comes back after the `have` of
/// `chain` the index has: each page's name, the name of the one before it
/// and its depth.
fn run_pages(chain: &Chain, have: usize, run: usize) -> Vec<(ChainHash, ChainHash, u32)> {
    (0..run)
        .map(|i| (chain.hashes()[have + i], chain.before(have + i), (have + i) as u32 + 1))
        .collect()
}

/// Whether a page of a run that comes back keeps the form it has, at `depth`
/// and `behind` cells before the run's end: the first page and those among
/// the newest cells (the device's settled cells and a page more) do; the
/// others take the one a sequence that went on would have given them.
fn keeps_form(depth: u32, behind: u64, settle: u64, per_page: u64) -> bool {
    depth == 1 || behind <= settle + per_page
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
    /// The places taken for saves inside a step ([`Cache::reserve`]) until the commit names
    /// or frees them: their memory is locked when the step begins, as a save's.
    reserved: Vec<u32>,
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
    /// The pool pages the model keeps in host memory ([`Model::far_pages`]),
    /// and the bytes of host memory that those only the index holds take:
    /// they rank as prompt cache (the owner's rule of 2026-11-22).
    ///
    /// [`Model::far_pages`]: crate::Model::far_pages
    far: Option<FarPages>,
    far_held: u64,
    /// The bytes of those far pages that are part of the cache's floor
    /// ([`PinnedRam::keep_context_cached`]).
    far_keep: u64,
    /// The cache gives way to running sequences ([`Cache::give_way`]): what
    /// leaves the devices finds no buffer in RAM.
    giving_way: bool,
}

impl Cache {
    /// The checkpoint places the budget of `config` holds for a model of this
    /// `spec`. The conductor and the device parts both use this number.
    pub(crate) fn places(spec: &PrefixCache, config: &RuntimeConfig) -> u32 {
        u32::try_from(Self::policy(spec, config).capacity()).unwrap_or(u32::MAX)
    }

    /// The checkpoint policy of `config` for a model of this `spec`.
    fn policy(spec: &PrefixCache, config: &RuntimeConfig) -> CheckpointPolicy {
        CheckpointPolicy {
            prompt_interval: config.checkpoint_prompt_interval,
            generation_interval: config.checkpoint_generation_interval,
            boundary_interval: config.checkpoint_boundary_interval,
            ..CheckpointPolicy::new(config.cells_per_page, spec.bytes(), config.cache_ram)
        }
    }

    /// `durable` is what the model offers the levels below the devices: the
    /// level in RAM is in use when the model offers it, the level on SSD when
    /// `config` names a directory too.
    pub(crate) fn new(
        model_name: &str,
        spec: &PrefixCache,
        durable: Option<&Durable>,
        config: &RuntimeConfig,
    ) -> Self {
        let cells_per_page = config.cells_per_page;
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
        let policy = Self::policy(spec, config);
        let index = match spec.tails_per_slot {
            Some(per_slot) => {
                PrefixIndex::new().with_tail_limit(per_slot as usize * config.max_slots as usize)
            }
            None => PrefixIndex::new(),
        };
        let capacity = u32::try_from(policy.capacity()).unwrap_or(u32::MAX);
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
            reserved: Vec::new(),
            restores: Vec::new(),
            page_copies: Vec::new(),
            scratch: Vec::new(),
            evictions: 0,
            far: None,
            far_held: 0,
            far_keep: 0,
            giving_way: false,
        }
    }

    /// The model keeps the pool pages from `far.first` on in host memory:
    /// those only the index holds count as the prompt cache's and leave the
    /// devices first when it must give memory back ([`Cache::trim`]). As many
    /// of them as one request of `cells` cells holds beyond VRAM are part of
    /// the cache's floor, which the tier of experts leaves it: the next turn of
    /// a conversation finds them, and its last checkpoint is of use.
    pub(crate) fn with_far(mut self, far: Option<FarPages>, cells: u32) -> Self {
        self.far = far;
        if let Some(far) = far {
            let pages = cells.div_ceil(self.cells_per_page.max(1));
            self.far_keep = u64::from(pages.saturating_sub(far.first)) * far.page_bytes;
        }
        if let Some(books) = &self.books {
            books.keep_context_cached(self.far_keep);
        }
        self
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
        let through = self.ssd.is_some();
        let place = if self.recurrent && self.capacity > 0 { self.bytes } else { 0 };
        let pages = match (through, &self.durable) {
            (true, Some(durable)) => SsdConfig::floor_pages(durable),
            _ => 0,
        };
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
        places * self.bytes + pages.max(self.page_floor) + self.far_held
    }

    /// Counts the far pages only the index holds and tells the books.
    fn count_far(&mut self, pool: &PagePool) {
        let Some(far) = self.far else { return };
        let cached = (far.first..pool.capacity())
            .map(PageId)
            .filter(|page| pool.refcount(*page) == 1 && pool.is_cached(*page))
            .count() as u64;
        self.far_held = cached * far.page_bytes;
        if let Some(books) = &self.books {
            books.set_context_cached(self.far_held);
        }
    }

    /// Whether the model keeps pool page `page` in host memory.
    fn is_far(&self, page: PageId) -> bool {
        self.far.is_some_and(|far| page.0 >= far.first)
    }

    /// Gives up a far page that only the index holds, from the end of a
    /// chain: it leaves the devices as any evicted page does, into RAM when
    /// the allowance has room, and otherwise nowhere: it never waits for the
    /// drive ([`Cache::push_leaving`]). Whether one went.
    fn evict_far(&mut self, pool: &mut PagePool) -> bool {
        let Some(far) = self.far else { return false };
        let left_behind = self.left_behind();
        let Some(evicted) = self.index.evict_where(pool, |page| page.0 >= far.first) else {
            return false;
        };
        self.leave_cards(evicted, &left_behind);
        self.evictions += 1;
        self.far_held = self.far_held.saturating_sub(far.page_bytes);
        if let Some(books) = &self.books {
            books.set_context_cached(self.far_held);
        }
        true
    }

    /// The policy with the budget the checkpoints have now: what the books
    /// allow but the floor of pages and the far pages the cache holds on the
    /// devices (the floor keeps them: they are no room for a place), and never
    /// more than the budget the places were counted for. Pages in RAM above
    /// their floor make room for checkpoints, not the other way round.
    fn policy_now(&self) -> CheckpointPolicy {
        let allowed =
            self.allowance().saturating_sub(self.page_floor).saturating_sub(self.far_held);
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
        if self.ssd.is_some() && found.computing.is_none() {
            let total = looked.len() / self.cells_per_page as usize;
            let owner = self.import_owner();
            let ssd = self.ssd.as_mut().expect("checked above");
            found.computing = ssd.want(&mut self.index, chain, found.pages.len(), total, owner);
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
mod tests;
