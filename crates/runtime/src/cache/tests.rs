//! The cache's own tests: places, checkpoints, the levels and the allowance.

use ares_state::Priority;

use super::*;

const PER_PAGE: u32 = 4;

/// A cache for a model whose checkpoints take 100 bytes, with room for
/// `places` of them.
fn cache(places: u64) -> Cache {
    let config = RuntimeConfig {
        cells_per_page: PER_PAGE,
        cache_ram: places * 100,
        ..RuntimeConfig::default()
    };
    Cache::new(
        "m",
        &PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: None },
        None,
        &config,
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
fn a_checkpoint_dropped_before_its_save_ran_keeps_the_save_a_restore_of_its_record_reads() {
    let mut pool = PagePool::new(8, PER_PAGE);
    let mut cache = cache(1);
    let hashes = three_pages(&mut cache, &mut pool);
    // A request's end state is saved with the next record, and the next turn restores it
    // in that record; a place for a save inside that turn's step then drops it. The save
    // stays, before the restore.
    assert!(cache.save(0, hashes[0], Owner(1), false).taken);
    cache.restore(0, CheckpointId(0));
    assert_eq!(cache.reserve(), (None, 1), "the only place is still read");
    assert_eq!(cache.index.checkpoint(hashes[0]), None);
    let (copies, _) = written(&mut cache);
    assert_eq!(
        copies,
        [
            StateCopy { kind: StateCopyKind::Save, slot: 0, place: 0 },
            StateCopy { kind: StateCopyKind::Restore, slot: 0, place: 0 },
        ]
    );
    // When the restore is cancelled instead, its save has nothing to save.
    assert!(cache.save(1, hashes[1], Owner(2), false).taken);
    cache.restore(3, CheckpointId(0));
    assert_eq!(cache.reserve(), (None, 1));
    assert!(cache.cancel_restore(3));
    let (copies, _) = written(&mut cache);
    assert!(copies.is_empty(), "{copies:?}");
}

#[test]
fn a_model_that_keeps_state_per_tail_limits_the_tails_per_slot() {
    let config =
        RuntimeConfig { cells_per_page: PER_PAGE, max_slots: 2, ..RuntimeConfig::default() };
    let spec = PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: Some(1) };
    let mut cache = Cache::new("m", &spec, None, &config);
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
    let mut cache = Cache::new("m", &spec, None, &config);
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

use std::time::Duration;

use crate::exec::{Clock, ExecError, StepMeta, StepOutput};
use crate::testing::TestMachine;

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
    machine: TestMachine,
    dir: std::path::PathBuf,
}

impl Following {
    fn new(tag: &str) -> Self {
        Self::with(tag, true, 4 * STATE + RESERVE * PAGE)
    }

    fn with(tag: &str, ssd: bool, budget: u64) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ares-cache-following-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let machine = TestMachine::new(PLENTY);
        let ram = PinnedRam::following(1 << 30, machine.machine());
        let durable = Durable {
            fingerprint: b"cache tests".to_vec(),
            page_bytes: vec![64],
            state_bytes: vec![STATE],
            settle_cells: 0,
        };
        let config = RuntimeConfig {
            cells_per_page: PER_PAGE,
            cache_ram: budget,
            // A text of one page is deep enough to be written.
            ssd: ssd.then(|| SsdConfig { min_cells: PER_PAGE, ..SsdConfig::new(&dir, 1 << 30) }),
            ..RuntimeConfig::default()
        };
        let spec = PrefixCache { checkpoint_bytes: vec![STATE], tails_per_slot: None };
        let mut cache = Cache::new("m", &spec, Some(&durable), &config).with_ram(Some(ram.clone()));
        assert!(cache.ssd_problem.is_none(), "{:?}", cache.ssd_problem);
        let mut pool = PagePool::new(64, PER_PAGE);
        let hashes = three_pages(&mut cache, &mut pool);
        Self { cache, pool, hashes, ram, machine, dir }
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
    f.machine.has(0);
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
    let config =
        RuntimeConfig { cells_per_page: PER_PAGE, cache_ram: 400, ..RuntimeConfig::default() };
    let spec = PrefixCache { checkpoint_bytes: vec![100], tails_per_slot: None };
    let mut plain = Cache::new("m", &spec, None, &config).with_ram(Some(f.ram.clone()));
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
    f.machine.has(0);
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
    f.machine.has(0);
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
    f.machine.has(0);
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
        f.machine.has(0);
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
fn far_pages_that_find_no_room_in_ram_are_dropped_without_waiting_for_the_drive() {
    // RAM holds the one place and the reserve of pages that pass through it; the
    // model keeps the pages from page 3 on in host memory, so every text's pages
    // are far: once only the index holds them they rank as prompt cache, the
    // lowest user of the RAM, and leave the devices at the next trim.
    let mut f = Following::with("far", true, STATE + RESERVE * PAGE);
    f.cache.far = Some(FarPages { first: 3, page_bytes: PAGE });
    let mut devices = Devices::default();
    let texts = 12;
    for i in 0..texts {
        f.text(i);
    }
    let far = 3 * u64::from(texts);
    assert!(far > RESERVE, "more far pages than RAM has room for");
    f.trimmed(&mut devices);
    // As many far pages left as the allowance needs: what RAM had room for
    // went there, the rest was dropped at once. None of them waited for the
    // writer to make room in RAM (a wait each, while the next request waits
    // for the conductor), which would have taken more of them into RAM.
    let ram = f.cache.ram_counters();
    let still = f.ram.context_cached() / PAGE;
    assert!(f.cache.held() <= f.cache.allowance());
    assert_eq!(ram.pages_in + ram.pages_dropped + still, far, "{ram:?}");
    assert!(ram.pages_in <= RESERVE, "{ram:?}");
    assert!(ram.pages_dropped > 0, "{ram:?}");
    assert_eq!(of(&devices.asked, TransferKind::PageOut).len() as u64, ram.pages_in);
    assert_eq!(f.written().pages_written, 0, "nothing of them went to the drive");
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
