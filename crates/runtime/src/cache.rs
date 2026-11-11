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

use ares_state::{
    Chain, ChainHash, CheckpointId, CheckpointPolicy, Evicted, Owner, PageId, PagePool,
    PrefixIndex, PrefixMatch, Resume,
};

use crate::config::RuntimeConfig;
use crate::model::PrefixCache;
use crate::record::{PageCopy, RecordFull, RecordWriter, StateCopy, StateCopyKind};

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
    /// The copies for the next record.
    saves: Vec<StateCopy>,
    restores: Vec<StateCopy>,
    page_copies: Vec<PageCopy>,
    scratch: Vec<StateCopy>,
    /// Counts what was given up, so that waiting requests can tell their
    /// numbers are out of date.
    pub evictions: u64,
}

impl Cache {
    /// The checkpoint places the budget of `config` holds for a model of this
    /// `spec`. The conductor and the device parts both use this number.
    pub(crate) fn places(spec: &PrefixCache, config: &RuntimeConfig) -> u32 {
        let policy =
            CheckpointPolicy::new(config.cells_per_page, spec.bytes(), config.checkpoint_budget);
        u32::try_from(policy.capacity()).unwrap_or(u32::MAX)
    }

    pub(crate) fn new(
        model_name: &str,
        spec: &PrefixCache,
        config: &RuntimeConfig,
        cells_per_page: u32,
    ) -> Self {
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
            root: ChainHash::root(model_name.as_bytes()),
            cells_per_page,
            recurrent: spec.bytes() > 0,
            capacity: Self::places(spec, config),
            policy,
            next_place: 0,
            free_places: Vec::new(),
            deferred: Vec::new(),
            saves: Vec::new(),
            restores: Vec::new(),
            page_copies: Vec::new(),
            scratch: Vec::new(),
            evictions: 0,
        }
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
        let found = self.index.lookup_chain(chain, &tokens[..tokens.len() - 1]);
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
        let dropped = self.index.make_room_for_checkpoints(&self.policy, 1);
        let count = dropped.len() as u32;
        for id in dropped {
            self.release(id);
        }
        if let Some(place) = self.free_places.pop() {
            return (Some(place), count);
        }
        let fresh = (self.next_place < self.capacity).then(|| {
            self.next_place += 1;
            self.next_place - 1
        });
        (fresh, count)
    }

    /// Gives a checkpoint's place back, once nothing reads it.
    pub(crate) fn release(&mut self, id: CheckpointId) {
        let place = id.0 as u32;
        if self.restores.iter().any(|copy| copy.place == place) {
            self.deferred.push(place);
        } else {
            self.free_places.push(place);
        }
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
                Saved { taken: true, dropped }
            }
            Err(id) => {
                self.free_places.push(id.0 as u32);
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
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.saves);
        self.scratch.extend_from_slice(&self.restores);
        writer.copies(&self.scratch, &self.page_copies)?;
        self.saves.clear();
        self.restores.clear();
        self.page_copies.clear();
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
        let mut cache = Cache::new("m", &spec, &config, PER_PAGE);
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
        let mut cache = Cache::new("m", &spec, &config, PER_PAGE);
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
}
