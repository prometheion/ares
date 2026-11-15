//! The conductor's prefix cache: the index of cached pages, the places of
//! checkpoints of recurrent state, and the copies the next step record carries.
//!
//! The conductor decides when to look, publish, capture, restore and evict;
//! this holds what those decisions act on, so that `conductor.rs` keeps to the
//! step loop. A model takes part when [`Model::prefix_cache`] says its device
//! side carries out the record's copies.
//!
//! # Checkpoint places
//!
//! A checkpoint is a copy of a slot's recurrent state in one of
//! `capacity` places on the devices; its [`CheckpointId`] is the place. The
//! index decides which checkpoints stay ([`PrefixIndex::make_room_for_checkpoints`]);
//! this frees and reuses the places. A place that a restore of the next record
//! will read is not reused before that record is written, however the index
//! treated its checkpoint meanwhile.
//!
//! # The allowance
//!
//! The checkpoints lie in page-locked host memory, of which they are the
//! lowest user ([`PinnedRam`]): the books say how many bytes they may hold
//! now, and that can be less than the budget the places were counted for,
//! down to nothing. The cache keeps no more checkpoints than the allowance
//! holds ([`Cache::trim`] gives the coldest up when it falls), and it tells
//! the books which places are in use, so that the device parts can give the
//! memory of the others back. A place counts as unused only once the record
//! with its last copy has run.
//!
//! # The allowance and the level on SSD
//!
//! A checkpoint at a page boundary also goes to the drive, from its place,
//! after the record with its save has run ([`Cache::pump`]). Where the two
//! rules meet, this holds:
//!
//! - **One order between steps.** The conductor pumps first and trims after:
//!   every checkpoint whose save has run has left its place for a buffer of
//!   the level on SSD (or was given up there, when the writer holds every
//!   buffer) before a place is called unused. What the writer has is in its
//!   own memory and is written whatever happens to the place.
//! - **A save that has not run has nothing to write.** A checkpoint that is
//!   given up before the record with its save was written loses that save,
//!   and the level on SSD forgets it with it and counts it
//!   ([`SsdCounters::states_skipped`]): nothing is read from a place that was
//!   never written or that went back to the system.
//! - **One place stays.** With the level on SSD in use for a model with
//!   recurrent state the cache keeps one place whatever the allowance says,
//!   its floor in the books: a checkpoint passes through it on the way to
//!   the drive, and one that comes back from the drive lands in it, so a
//!   prompt resumes from SSD while the machine has no room for checkpoints.

use ares_state::{
    Chain, ChainHash, CheckpointId, CheckpointPolicy, Evicted, Owner, PageId, PagePool,
    PrefixIndex, PrefixMatch, Resume,
};

use crate::config::RuntimeConfig;
use crate::model::{Durable, PrefixCache};
use crate::ram::PinnedRam;
use crate::record::{PageCopy, RecordFull, RecordWriter, StateCopy, StateCopyKind};

mod ssd;

pub use ssd::{SsdConfig, SsdCounters};

/// What a look into the index found for a text.
pub(crate) struct Probe {
    pub found: PrefixMatch,
    pub resume: Resume,
    /// The matched pages that no sequence holds now: taking them makes them
    /// unavailable for eviction.
    pub idle: u32,
}

/// What an eviction gave up.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Evictions {
    pub pages: u32,
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
    /// The books of the page-locked memory the places lie in, when the
    /// executor has any, and the bytes of one checkpoint on all devices.
    ram: Option<PinnedRam>,
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
    /// The level on SSD, when the engine has one and the model can use it,
    /// and why not when it was asked for and is not in use.
    ssd: Option<ssd::Ssd>,
    pub ssd_problem: Option<String>,
    /// The saves of checkpoints, each with its place and what it hangs on,
    /// in their order: queued for the next record, and carried by the record
    /// that was written last.
    saved: Vec<(u32, ChainHash)>,
    flying: Vec<(u32, ChainHash)>,
}

impl Cache {
    /// The checkpoint places the budget of `config` holds for a model of this
    /// `spec`. The conductor and the device parts both use this number.
    pub(crate) fn places(spec: &PrefixCache, config: &RuntimeConfig) -> u32 {
        let policy =
            CheckpointPolicy::new(config.cells_per_page, spec.bytes(), config.checkpoint_budget);
        u32::try_from(policy.capacity()).unwrap_or(u32::MAX)
    }

    /// `durable` is what the model offers the level on SSD; the level is in
    /// use when the model offers it and `config` names a directory.
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
            ..CheckpointPolicy::new(cells_per_page, spec.bytes(), config.checkpoint_budget)
        };
        let index = match spec.tails_per_slot {
            Some(per_slot) => {
                PrefixIndex::new().with_tail_limit(per_slot as usize * config.max_slots as usize)
            }
            None => PrefixIndex::new(),
        };
        Self {
            index,
            root,
            cells_per_page,
            recurrent,
            ssd,
            ssd_problem,
            saved: Vec::new(),
            flying: Vec::new(),
            capacity: Self::places(spec, config),
            policy,
            next_place: 0,
            free_places: Vec::new(),
            deferred: Vec::new(),
            ram: None,
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

    /// The places lie in the page-locked memory of `ram`: the cache keeps to
    /// the allowance of its books and tells them which places are in use.
    /// With the level on SSD in use, one place is what the cache cannot work
    /// without: checkpoints pass through it to the drive and back.
    pub(crate) fn with_ram(mut self, ram: Option<PinnedRam>) -> Self {
        if let Some(ram) = &ram {
            let through = self.ssd.is_some() && self.recurrent && self.capacity > 0;
            let floor = if through { self.bytes } else { 0 };
            ram.want_checkpoints(self.policy.budget_bytes, floor);
        }
        self.ram = ram;
        self
    }

    /// The policy with the budget the checkpoints have now: what the books
    /// allow, and never more than the budget the places were counted for.
    fn policy_now(&self) -> CheckpointPolicy {
        let allowed = self.ram.as_ref().map_or(u64::MAX, PinnedRam::checkpoints_allowance);
        CheckpointPolicy { budget_bytes: self.policy.budget_bytes.min(allowed), ..self.policy }
    }

    fn mark(&self, place: u32, live: bool) {
        if let Some(ram) = &self.ram {
            ram.place_live(place, live);
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

    /// Gives up the coldest checkpoints until those that are left fit the
    /// allowance of the books, and says how many went. Call it between
    /// steps only (every record that was written has run), and after
    /// [`Cache::pump`], which takes the checkpoints that are due on SSD out
    /// of their places first: when the allowance fell, also while nothing
    /// runs.
    pub(crate) fn trim(&mut self) -> u32 {
        if self.ram.is_none() || self.bytes == 0 {
            return 0;
        }
        self.settle();
        let policy = self.policy_now();
        let dropped = self.index.make_room_for_checkpoints(&policy, 0);
        let count = dropped.len() as u32;
        for id in dropped {
            self.release(id);
        }
        // Nothing runs: a freed place that no copy of the next record uses
        // is unused already, and the others wait for that record. So does a
        // place whose checkpoint the level on SSD has yet to take out.
        let (saves, restores, flying) = (&self.saves, &self.restores, &self.flying);
        let (waiting, unused): (Vec<u32>, Vec<u32>) =
            std::mem::take(&mut self.settling).into_iter().partition(|place| {
                saves.iter().chain(restores).any(|copy| copy.place == *place)
                    || flying.iter().any(|(of, _)| of == place)
            });
        for place in unused {
            if self.free_places.contains(&place) {
                self.mark(place, false);
            }
        }
        self.settling = waiting;
        count
    }

    /// The names of the full pages of `tokens`.
    pub(crate) fn chain(&self, tokens: &[u32]) -> Chain {
        let mut chain = Chain::new(self.root, self.cells_per_page);
        chain.name(tokens);
        chain
    }

    /// The chain of a sequence that starts from `found`, with the full pages of
    /// `tokens` named.
    pub(crate) fn chain_after(&self, found: &PrefixMatch, tokens: &[u32]) -> Chain {
        let mut chain = Chain::resumed(self.root, self.cells_per_page, found);
        chain.name(tokens);
        chain
    }

    /// What the cache has for a prompt of `tokens` (named in `chain`), whose
    /// last token is always computed. The matched pages count as used.
    pub(crate) fn probe(&mut self, pool: &PagePool, chain: &Chain, tokens: &[u32]) -> Probe {
        let mut found = self.index.lookup_chain(chain, &tokens[..tokens.len() - 1]);
        // What the index does not have may be on SSD: a run of pages that is
        // brought back holds the prompt as a page that is being computed does.
        if let (Some(ssd), None) = (&mut self.ssd, found.computing) {
            let total = (tokens.len() - 1) / self.cells_per_page as usize;
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
        Probe { found, resume, idle }
    }

    /// The first position in `(from, to]` at which a checkpoint is due for a
    /// sequence of a prompt of `prompt_len` cells whose last checkpoint is at
    /// `last`, or `None`.
    pub(crate) fn capture(&self, prompt_len: u64, last: u64, from: u64, to: u64) -> Option<u64> {
        self.policy.next_capture(prompt_len, last, from, to)
    }

    /// The place of a new checkpoint, with room made for it, and how many
    /// checkpoints that cost.
    fn new_place(&mut self) -> (Option<u32>, u32) {
        let policy = self.policy_now();
        let dropped = self.index.make_room_for_checkpoints(&policy, 1);
        let count = dropped.len() as u32;
        for id in dropped {
            self.release(id);
        }
        // No room for even one checkpoint now: none is taken.
        if policy.capacity() == 0 {
            return (None, count);
        }
        let place = self.free_places.pop().or_else(|| {
            (self.next_place < self.capacity).then(|| {
                self.next_place += 1;
                self.next_place - 1
            })
        });
        if let Some(place) = place {
            self.mark(place, true);
        }
        (place, count)
    }

    /// Gives a checkpoint's place back, once nothing reads it.
    pub(crate) fn release(&mut self, id: CheckpointId) {
        let place = id.0 as u32;
        // A save of it that still waits for its record has nothing to save,
        // and the level on SSD nothing to write from the place.
        self.saves.retain(|copy| copy.place != place);
        self.unsave(place);
        if self.restores.iter().any(|copy| copy.place == place) {
            self.deferred.push(place);
        } else {
            self.free(place);
        }
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
    /// there already. The copy runs with the next record.
    pub(crate) fn save(&mut self, slot: u32, hash: ChainHash, owner: Owner) -> Saved {
        if !self.recurrent || self.capacity == 0 || self.index.checkpoint(hash).is_some() {
            return Saved::default();
        }
        let (place, dropped) = self.new_place();
        let Some(place) = place else { return Saved { taken: false, dropped } };
        match self.index.attach_checkpoint(hash, CheckpointId(u64::from(place)), owner) {
            Ok(before) => {
                debug_assert!(before.is_none(), "the entry had no checkpoint");
                self.saves.push(StateCopy { kind: StateCopyKind::Save, slot, place });
                if self.ssd.is_some() {
                    self.saved.push((place, hash));
                }
                Saved { taken: true, dropped }
            }
            Err(id) => {
                self.free(id.0 as u32);
                Saved { taken: false, dropped }
            }
        }
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

    /// Gives up the checkpoint of an evicted page or tail, when it had one.
    pub(crate) fn forget(&mut self, evicted: Evicted) -> bool {
        match evicted.checkpoint {
            Some(id) => {
                self.release(id);
                true
            }
            None => false,
        }
    }

    /// Gives up `pages` cached pages from the ends of chains.
    pub(crate) fn evict(&mut self, pool: &mut PagePool, pages: u32) -> Evictions {
        let mut given = Evictions::default();
        for _ in 0..pages {
            let Some(evicted) = self.index.evict(pool) else { break };
            given.pages += 1;
            given.checkpoints += u32::from(self.forget(evicted));
        }
        if given.pages > 0 {
            self.evictions += 1;
        }
        given
    }

    /// Writes the copies for this record and starts the next ones afresh.
    pub(crate) fn write_copies(&mut self, writer: &mut RecordWriter<'_>) -> Result<(), RecordFull> {
        // The record before this one has run.
        self.settle();
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.saves);
        self.scratch.extend_from_slice(&self.restores);
        writer.copies(&self.scratch, &self.page_copies)?;
        self.saves.clear();
        self.flying.append(&mut self.saved);
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
    use ares_state::{Priority, Sequence};

    use super::*;

    const PER_PAGE: u32 = 4;

    /// A cache for a model whose checkpoints take 100 bytes, with room for
    /// `places` of them.
    fn cache(places: u64) -> Cache {
        let config = RuntimeConfig { checkpoint_budget: places * 100, ..RuntimeConfig::default() };
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
        let tokens: Vec<u32> = (0..12).collect();
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
    fn a_place_whose_checkpoint_is_dropped_waits_for_the_restore_that_reads_it() {
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(1);
        let hashes = three_pages(&mut cache, &mut pool);
        assert!(cache.save(0, hashes[0], Owner(1)).taken);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Save, slot: 0, place: 0 }]);

        // A start restores the place with the next record. Before that record
        // another checkpoint is taken: the budget holds one, so the first is
        // dropped from the index, and its place is not reused yet.
        cache.restore(3, CheckpointId(0));
        let second = cache.save(1, hashes[1], Owner(2));
        assert_eq!((second.taken, second.dropped), (false, 1), "the only place is still read");
        assert_eq!(cache.index.checkpoint(hashes[0]), None);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Restore, slot: 3, place: 0 }]);

        // With the record written the place is free for the next checkpoint.
        assert!(cache.save(1, hashes[1], Owner(2)).taken);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Save, slot: 1, place: 0 }]);
    }

    #[test]
    fn a_restore_that_is_cancelled_frees_the_place_that_waited_for_it() {
        let mut pool = PagePool::new(8, PER_PAGE);
        let mut cache = cache(1);
        let hashes = three_pages(&mut cache, &mut pool);
        assert!(cache.save(0, hashes[0], Owner(1)).taken);
        written(&mut cache);
        cache.restore(3, CheckpointId(0));
        assert!(!cache.save(1, hashes[1], Owner(2)).taken);
        // The sequence that was to restore it ended before the record: the
        // restore is not carried out, and the place is free.
        assert!(cache.cancel_restore(3));
        assert!(!cache.cancel_restore(3));
        assert!(cache.save(1, hashes[1], Owner(2)).taken);
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
            let prior = root.extend(&[i]);
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
                root.extend(&[i]),
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
        assert!(cache.save(0, hashes[2], Owner(1)).taken);
        written(&mut cache);
        // The chain goes from its end: the last page takes its checkpoint, and
        // its place is free for the next one.
        let given = cache.evict(&mut pool, 1);
        assert_eq!((given.pages, given.checkpoints), (1, 1));
        assert_eq!(cache.evictions, 1);
        assert!(cache.save(5, hashes[1], Owner(1)).taken);
        let (copies, _) = written(&mut cache);
        assert_eq!(copies, [StateCopy { kind: StateCopyKind::Save, slot: 5, place: 0 }]);
        assert_eq!(cache.evict(&mut pool, 5).pages, 2, "two pages are left");
        assert_eq!(cache.evict(&mut pool, 5).pages, 0);
        assert_eq!(cache.evictions, 2, "an eviction that found nothing is not one");
        assert_eq!(pool.free_pages(), 8);
    }

    // --- the allowance of the books and the level on SSD ----------------------

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use crate::exec::{Clock, ExecError, Executor, StepMeta, StepOutput, Transfer, TransferKind};
    use crate::ram::{Machine, Rule};

    const PLENTY: u64 = 64 << 30;

    /// A cache with a level on SSD in a directory of its own, for a model
    /// whose checkpoints take 100 bytes, with places for four of them in
    /// books that follow a machine whose available memory the test sets.
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
            let ram = PinnedRam::following(1 << 20, &machine);
            let durable = Durable {
                fingerprint: b"cache tests".to_vec(),
                page_bytes: vec![64],
                state_bytes: vec![100],
                settle_cells: 0,
            };
            let config = RuntimeConfig {
                checkpoint_budget: 400,
                // A text of one page is deep enough to be written.
                ssd: Some(SsdConfig { min_cells: PER_PAGE, ..SsdConfig::new(&dir, 1 << 30) }),
                ..RuntimeConfig::default()
            };
            let spec = PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: None };
            let mut cache = Cache::new("m", &spec, Some(&durable), &config, PER_PAGE)
                .with_ram(Some(ram.clone()));
            assert!(cache.ssd_problem.is_none(), "{:?}", cache.ssd_problem);
            let mut pool = PagePool::new(8, PER_PAGE);
            let hashes = three_pages(&mut cache, &mut pool);
            Self { cache, pool, hashes, ram, machine, available, dir }
        }

        /// The machine has `available` bytes from now on, and has been
        /// looked at.
        fn machine_has(&self, available: u64) {
            self.available.store(available, Ordering::SeqCst);
            self.machine.look().expect("the machine says what it has");
        }

        /// Pumps the level on SSD against devices that carry out nothing, and
        /// returns the transfers it asked for.
        fn pumped(&mut self) -> Vec<(TransferKind, u32)> {
            let mut devices = Devices::default();
            self.cache.pump(&mut self.pool, &mut devices);
            devices.asked
        }
    }

    impl Drop for Following {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Devices that write down the transfers they are asked for and carry
    /// none of them out.
    #[derive(Default)]
    struct Devices {
        asked: Vec<(TransferKind, u32)>,
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
                transfer.ok = false;
            }
            Ok(())
        }

        fn shutdown(&mut self) {}
    }

    #[test]
    fn the_level_on_ssd_keeps_one_place_when_the_machine_has_no_room_for_checkpoints() {
        let mut f = Following::new("floor");
        assert_eq!(f.ram.checkpoints_allowance(), 400, "all of the budget while memory is free");
        f.machine_has(0);
        assert_eq!(f.ram.checkpoints_allowance(), 100, "one place whatever the machine has");
        // A checkpoint is still taken, in the one place, and the next one
        // takes the place over.
        assert!(f.cache.save(0, f.hashes[0], Owner(1)).taken);
        let second = f.cache.save(1, f.hashes[1], Owner(1));
        assert_eq!((second.taken, second.dropped), (true, 1));
        assert_eq!(f.cache.index.checkpoint(f.hashes[0]), None);
        assert!(f.cache.index.checkpoint(f.hashes[1]).is_some());

        // Without a level on SSD nothing is kept: the same books allow no
        // checkpoint at all.
        let config = RuntimeConfig { checkpoint_budget: 400, ..RuntimeConfig::default() };
        let spec = PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: None };
        let mut plain =
            Cache::new("m", &spec, None, &config, PER_PAGE).with_ram(Some(f.ram.clone()));
        assert_eq!(f.ram.checkpoints_allowance(), 0);
        let hashes = three_pages(&mut plain, &mut f.pool);
        assert!(!plain.save(0, hashes[0], Owner(1)).taken);
    }

    #[test]
    fn a_checkpoint_given_up_before_its_save_ran_is_neither_saved_nor_written_to_ssd() {
        let mut f = Following::new("unsaved");
        // Two sequences reach a page boundary in the same step: two saves
        // wait for the next record, and the level on SSD waits for both.
        assert!(f.cache.save(0, f.hashes[1], Owner(1)).taken);
        assert!(f.cache.save(1, f.hashes[2], Owner(2)).taken);
        assert_eq!((f.cache.saves.len(), f.cache.saved.len()), (2, 2));

        // The machine runs short before that record is written: the cache
        // keeps the one place of its floor and gives the other checkpoint up.
        f.machine_has(0);
        assert_eq!(f.cache.trim(), 1);
        let kept: Vec<u32> = [f.hashes[1], f.hashes[2]]
            .iter()
            .filter_map(|hash| f.cache.index.checkpoint(*hash))
            .map(|id| id.0 as u32)
            .collect();
        assert_eq!(kept.len(), 1, "one checkpoint is left");
        let (kept, gone) = (kept[0], 1 - kept[0]);
        // Its save went with it, and so did what the level on SSD knew of
        // it: that is counted, since the checkpoint was due on the drive.
        assert_eq!(f.cache.saved.iter().map(|(place, _)| *place).collect::<Vec<_>>(), [kept]);
        assert_eq!(f.cache.ssd_counters().states_skipped, 1);
        // The place that never held it is unused at once; the other is not.
        assert!(!f.ram.place_is_live(gone) && f.ram.place_is_live(kept));

        // The record carries the one save, and after it has run the level
        // on SSD reads that place and no other.
        let (copies, _) = written(&mut f.cache);
        assert_eq!(copies.len(), 1);
        assert_eq!((copies[0].kind, copies[0].place), (StateCopyKind::Save, kept));
        let asked = f.pumped();
        let states: Vec<_> =
            asked.iter().filter(|(kind, _)| *kind == TransferKind::StateOut).collect();
        assert_eq!(states, [&(TransferKind::StateOut, kept)]);
        assert_eq!(f.cache.ssd_counters().states_skipped, 1, "nothing else was given up");
    }

    #[test]
    fn a_place_stays_in_use_until_the_level_on_ssd_has_read_its_checkpoint() {
        let mut f = Following::new("order");
        assert!(f.cache.save(0, f.hashes[1], Owner(1)).taken);
        assert!(f.cache.save(1, f.hashes[2], Owner(2)).taken);
        // The record with both saves is written and has run: both places
        // hold their checkpoints, and the level on SSD has yet to read them.
        assert_eq!(written(&mut f.cache).0.len(), 2);
        assert_eq!(f.cache.flying.len(), 2);

        // The machine runs short now. Whichever comes first, the trim or the
        // pump: a place is not called unused, and so cannot lose its memory,
        // before its checkpoint was read out of it.
        f.machine_has(0);
        assert_eq!(f.cache.trim(), 1, "one checkpoint goes: the floor is one place");
        assert!(f.ram.place_is_live(0) && f.ram.place_is_live(1));
        let asked = f.pumped();
        let mut states: Vec<u32> = asked
            .iter()
            .filter(|(kind, _)| *kind == TransferKind::StateOut)
            .map(|(_, place)| *place)
            .collect();
        states.sort_unstable();
        assert_eq!(states, [0, 1], "both left for the drive from places that held them");
        assert_eq!(f.cache.ssd_counters().states_skipped, 0);
        // With the next trim the place of the checkpoint that went is unused.
        assert_eq!(f.cache.trim(), 0);
        let live = [f.ram.place_is_live(0), f.ram.place_is_live(1)];
        assert_eq!(live.iter().filter(|live| **live).count(), 1, "{live:?}");
    }

    #[test]
    fn a_place_that_is_given_up_and_taken_again_before_the_next_record_stays_in_use() {
        let mut f = Following::new("retaken");
        f.machine_has(0);
        // The one place holds a checkpoint whose save waits for its record.
        assert!(f.cache.save(0, f.hashes[1], Owner(1)).taken);
        let place = f.cache.saves[0].place;
        // The checkpoint is given up before that record. The rule of the
        // allowance drops its save, and what the level on SSD knew of it.
        let id = f.cache.index.checkpoint(f.hashes[1]).unwrap();
        f.cache.index.set_checkpoint(f.hashes[1], None).unwrap();
        f.cache.release(id);
        assert!(f.cache.saves.is_empty() && f.cache.saved.is_empty());
        // The place is taken again in the same turn, as a checkpoint that
        // comes back from the drive takes it: the level on SSD finds no stale
        // save to remove there, and neither the next record nor the next trim
        // touches what the place holds now.
        assert_eq!(f.cache.new_place().0, Some(place));
        assert!(written(&mut f.cache).0.is_empty());
        assert_eq!(f.cache.trim(), 0);
        assert!(f.ram.place_is_live(place), "a place that was taken again is not called unused");
        assert!(f.pumped().is_empty());
    }
}
