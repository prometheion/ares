//! The store against real files: restarts, the size limit, damage, sharing a
//! root between fingerprints, and the lock.

use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use ares_state::ChainHash;
use ares_store::{Aligned, BLOCK, Config, Error, Fingerprint, Kind, PageStore, Part, padded};

const PER_PAGE: u32 = 4;
/// Bytes of a test page and of its record on disk (a header block and the
/// page padded to a block).
const PAGE: usize = 1000;
const RECORD: u64 = (BLOCK + padded(PAGE)) as u64;

/// A fresh directory under the system temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("ares-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fingerprint(name: &str) -> Fingerprint {
    Fingerprint::new(&[name.as_bytes(), b"q5_1 kv", &PER_PAGE.to_le_bytes()])
}

fn root_hash() -> ChainHash {
    ChainHash::root(b"model")
}

/// The contents of page `i` of the conversation `tokens`.
fn page_bytes(tokens: &[u64], i: usize) -> Vec<u8> {
    (0..PAGE).map(|b| (tokens[i * PER_PAGE as usize] as usize * 31 + i * 7 + b) as u8).collect()
}

fn put_page(store: &mut PageStore, hash: ChainHash, parent: ChainHash, depth: u32, bytes: &[u8]) {
    assert!(store.put(Kind::Page, hash, parent, depth, &[Part::whole(bytes)]).unwrap());
}

/// Stores every full page of `tokens`, a checkpoint on every second page, and
/// returns the page hashes.
fn put_conversation(store: &mut PageStore, tokens: &[u64]) -> Vec<ChainHash> {
    let mut hash = root_hash();
    let mut hashes = Vec::new();
    let per_page = std::hint::black_box(PER_PAGE as usize);
    for (i, page_tokens) in tokens.chunks_exact(per_page).enumerate() {
        let parent = hash;
        hash = hash.extend(page_tokens);
        put_page(store, hash, parent, i as u32 + 1, &page_bytes(tokens, i));
        if i % 2 == 1 {
            // Two parts, as a model on two devices has them.
            let (a, b) = (vec![i as u8; 300], vec![!(i as u8); 70]);
            let parts = [Part { buf: &a, len: 300, tag: 7 }, Part { buf: &b, len: 70, tag: 9 }];
            assert!(store.put(Kind::State, hash, parent, i as u32 + 1, &parts).unwrap());
        }
        hashes.push(hash);
    }
    hashes
}

fn segments(dir: &Path, extension: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == extension))
        .collect();
    files.sort();
    files
}

#[test]
fn records_survive_a_restart_and_are_found_by_walking_the_prompt() {
    let dir = TempDir::new("restart");
    let tokens: Vec<u64> = (100..122).collect();
    let fp = fingerprint("a");

    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert!(store.is_empty());
    let hashes = put_conversation(&mut store, &tokens);
    assert_eq!(hashes.len(), 5, "22 tokens are 5 full pages of 4");
    assert_eq!((store.len(), store.states()), (5, 2));
    // Storing a record again writes nothing.
    let before = store.bytes();
    assert!(!store.put(Kind::Page, hashes[0], root_hash(), 1, &[Part::whole(&[1, 2, 3])]).unwrap());
    assert!(!store.put(Kind::State, hashes[1], hashes[0], 2, &[Part::whole(&[1])]).unwrap());
    assert_eq!(store.bytes(), before);
    store.flush().sync().unwrap();
    drop(store);

    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert_eq!((store.len(), store.states()), (5, 2));
    assert_eq!(store.bytes(), before);
    assert_eq!(store.lookup(root_hash(), &tokens, PER_PAGE), hashes);

    let mut out = Vec::new();
    for (i, hash) in hashes.iter().enumerate() {
        assert_eq!(store.read(Kind::Page, *hash, &mut out).unwrap(), Some(vec![0]));
        assert_eq!(out, [page_bytes(&tokens, i)]);
        let stored = store.stat(Kind::Page, *hash).unwrap();
        assert_eq!(stored.parent, if i == 0 { root_hash() } else { hashes[i - 1] });
        assert_eq!(stored.depth, i as u32 + 1);
        if i % 2 == 1 {
            assert_eq!(store.read(Kind::State, *hash, &mut out).unwrap(), Some(vec![7, 9]));
            assert_eq!(out, [vec![i as u8; 300], vec![!(i as u8); 70]]);
            assert_eq!(store.stat(Kind::State, *hash).unwrap().parts, [(300, 7), (70, 9)]);
        } else {
            assert!(!store.contains(Kind::State, *hash));
            assert_eq!(store.read(Kind::State, *hash, &mut out).unwrap(), None);
        }
    }

    // A prompt that shares the first 9 tokens resumes after 2 pages, and a
    // different context finds nothing.
    let mut other = tokens[..9].to_vec();
    other.extend([7; 12]);
    assert_eq!(store.lookup(root_hash(), &other, PER_PAGE), hashes[..2]);
    assert!(store.lookup(ChainHash::root(b"other model"), &tokens, PER_PAGE).is_empty());
    assert_eq!(store.read(Kind::Page, root_hash().extend(&[9; 4]), &mut out).unwrap(), None);
}

#[test]
fn parts_are_read_into_aligned_buffers_outside_of_the_store() {
    let dir = TempDir::new("aligned");
    let mut store = PageStore::open(&dir.0, fingerprint("a"), Config::default()).unwrap();
    let hash = root_hash().extend(&[1; 4]);
    let mut source = Aligned::new(3 * BLOCK);
    for (i, byte) in source.iter_mut().enumerate() {
        *byte = (i * 13) as u8;
    }
    let len = 2 * BLOCK + 77;
    let mut pending =
        store.begin(Kind::Page, hash, root_hash(), 1, &[len as u32, 5]).unwrap().unwrap();
    // Nothing is known before the record is written and committed.
    assert!(store.locate(Kind::Page, hash).is_none());
    let small = [1u8, 2, 3, 4, 5];
    pending
        .write(&[Part { buf: &source, len, tag: 3 }, Part { buf: &small, len: 5, tag: 4 }])
        .unwrap();
    assert!(store.commit(pending));
    assert_eq!(store.bytes(), (BLOCK + 3 * BLOCK + BLOCK) as u64);

    let located = store.locate(Kind::Page, hash).unwrap();
    assert_eq!(located.parts(), [(len as u32, 3), (5, 4)]);
    let reader = std::thread::spawn(move || {
        let (mut a, mut b) = (Aligned::new(3 * BLOCK), vec![0u8; 5]);
        located.read_part(0, &mut a).unwrap();
        located.read_part(1, &mut b).unwrap();
        assert!(located.verify(&[&a[..len], &b]));
        assert!(!located.verify(&[&a[..len - 1], &b]));
        (a, b)
    });
    let (a, b) = reader.join().unwrap();
    assert_eq!(a[..len], source[..len]);
    assert_eq!(b, small);

    // A record that was begun and not written is not there after a restart,
    // and the next record takes its place.
    let lost = root_hash().extend(&[2; 4]);
    let pending = store.begin(Kind::Page, lost, root_hash(), 1, &[100]).unwrap().unwrap();
    assert!(!store.commit(pending), "nothing was written");
    put_page(&mut store, root_hash().extend(&[3; 4]), root_hash(), 1, &[9; 100]);
    drop(store);
    let store = PageStore::open(&dir.0, fingerprint("a"), Config::default()).unwrap();
    assert_eq!(store.len(), 2);
    assert!(!store.contains(Kind::Page, lost));
}

#[test]
fn the_least_recently_used_segments_go_when_the_limit_is_reached() {
    let dir = TempDir::new("limit");
    // Three records per segment, room for four segments.
    let config = Config { max_bytes: 12 * RECORD, segment_bytes: 3 * RECORD };
    let mut store = PageStore::open(&dir.0, fingerprint("a"), config).unwrap();
    let hash = |i: u32| root_hash().extend(&[u64::from(i); 4]);
    let page = vec![0xabu8; PAGE];
    for i in 0..9 {
        put_page(&mut store, hash(i), root_hash(), 1, &page);
    }
    assert_eq!(store.len(), 9);

    // Page 0 is used again, which makes the second segment the oldest.
    let mut out = Vec::new();
    assert!(store.read(Kind::Page, hash(0), &mut out).unwrap().is_some());
    for i in 9..15 {
        put_page(&mut store, hash(i), root_hash(), 1, &page);
    }
    assert!(store.bytes() <= config.max_bytes);
    assert!(store.contains(Kind::Page, hash(0)), "the segment that was read stays");
    assert!(!store.contains(Kind::Page, hash(3)), "the segment not used the longest went");
    assert!(store.contains(Kind::Page, hash(14)));
    assert_eq!(store.len(), 12);
    assert_eq!(store.evicted_segments(), 1);
    assert_eq!(segments(&dir.0.join(fingerprint("a").hex()), "pages").len(), 4);

    // A drive that is full gets room whatever the limit says.
    assert_eq!(store.make_room(1), 3 * RECORD);
    assert_eq!(store.len(), 9);

    // The limit also holds after a restart with a smaller budget.
    drop(store);
    let small = Config { max_bytes: 6 * RECORD, segment_bytes: 3 * RECORD };
    let store = PageStore::open(&dir.0, fingerprint("a"), small).unwrap();
    assert!(store.bytes() <= small.max_bytes);
    assert_eq!(store.len(), 6);

    // A record that cannot fit in a segment is refused.
    drop(store);
    let mut store = PageStore::open(&dir.0, fingerprint("a"), small).unwrap();
    let huge = vec![0u8; 8 * BLOCK];
    assert!(matches!(
        store.put(Kind::Page, hash(99), root_hash(), 1, &[Part::whole(&huge)]),
        Err(Error::TooLarge { .. })
    ));
}

#[test]
fn what_nothing_can_reach_goes_before_what_was_used_least_recently() {
    let dir = TempDir::new("unreachable");
    // Two records per segment.
    let config = Config { max_bytes: 9 * RECORD, segment_bytes: 2 * RECORD };
    let mut store = PageStore::open(&dir.0, fingerprint("a"), config).unwrap();
    let page = vec![1u8; PAGE];
    // Conversation A: pages 1 and 2 in the first segment, 3 and 4 in the
    // second; conversation B: two pages in the third segment.
    let a: Vec<ChainHash> = (1..=4u64).map(|i| root_hash().extend(&[i; 8])).collect();
    let b: Vec<ChainHash> = (1..=2u64).map(|i| root_hash().extend(&[50 + i; 8])).collect();
    for (i, hash) in a.iter().enumerate() {
        let parent = if i == 0 { root_hash() } else { a[i - 1] };
        put_page(&mut store, *hash, parent, i as u32 + 1, &page);
    }
    put_page(&mut store, b[0], root_hash(), 1, &page);
    put_page(&mut store, b[1], b[0], 2, &page);
    // The checkpoint at the end of A's third page, in a segment of states.
    assert!(store.put(Kind::State, a[2], a[1], 3, &[Part::whole(&[5; 100])]).unwrap());
    // Everything of A is used after B.
    assert_eq!(store.lookup(root_hash(), &[], PER_PAGE), []);
    for hash in &a {
        assert!(store.locate(Kind::Page, *hash).is_some());
    }
    assert!(store.locate(Kind::State, a[2]).is_some());
    // A's first segment is lost (as when the limit took it earlier): its later
    // pages and its checkpoint cannot be reached by any prompt.
    let first = segments(&dir.0.join(fingerprint("a").hex()), "pages").remove(0);
    drop(store);
    std::fs::remove_file(first).unwrap();
    let mut store = PageStore::open(&dir.0, fingerprint("a"), config).unwrap();
    assert_eq!((store.len(), store.states()), (4, 1));

    // New pages pass the limit: A's unreachable segment goes although B's was
    // used longer ago.
    let c: Vec<ChainHash> = (1..=5u64).map(|i| root_hash().extend(&[90 + i; 8])).collect();
    for (i, hash) in c.iter().enumerate() {
        let parent = if i == 0 { root_hash() } else { c[i - 1] };
        put_page(&mut store, *hash, parent, i as u32 + 1, &page);
    }
    assert!(store.bytes() <= config.max_bytes);
    assert!(!store.contains(Kind::Page, a[2]) && !store.contains(Kind::Page, a[3]));
    assert!(store.contains(Kind::Page, b[0]) && store.contains(Kind::Page, b[1]));
    assert!(c.iter().all(|hash| store.contains(Kind::Page, *hash)));
}

#[test]
fn the_order_of_use_carries_over_a_restart() {
    let dir = TempDir::new("order");
    let config = Config { max_bytes: 6 * RECORD, segment_bytes: 2 * RECORD };
    // A clock the test moves.
    let seconds = Arc::new(AtomicU64::new(1_000_000));
    let clock = {
        let seconds = seconds.clone();
        Arc::new(move || {
            SystemTime::UNIX_EPOCH + Duration::from_secs(seconds.load(Ordering::SeqCst))
        })
    };
    let hash = |i: u32| root_hash().extend(&[u64::from(i); 4]);
    let page = vec![3u8; PAGE];
    let mut store = PageStore::open_with(&dir.0, fingerprint("a"), config, clock.clone()).unwrap();
    for i in 0..6 {
        seconds.fetch_add(1000, Ordering::SeqCst);
        put_page(&mut store, hash(i), root_hash(), 1, &page);
    }
    // The first segment is used an hour later than the others.
    seconds.fetch_add(3600, Ordering::SeqCst);
    assert!(store.locate(Kind::Page, hash(0)).is_some());
    drop(store);

    let mut store = PageStore::open_with(&dir.0, fingerprint("a"), config, clock).unwrap();
    assert_eq!(store.len(), 6);
    seconds.fetch_add(10, Ordering::SeqCst);
    put_page(&mut store, hash(6), root_hash(), 1, &page);
    assert!(store.contains(Kind::Page, hash(0)), "used last before the restart");
    assert!(!store.contains(Kind::Page, hash(2)), "the second segment was the oldest");
    assert!(store.contains(Kind::Page, hash(4)));
}

#[test]
fn damage_costs_only_the_damaged_records() {
    let dir = TempDir::new("damage");
    let fp = fingerprint("a");
    let tokens: Vec<u64> = (0..12).collect();
    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    let hashes = put_conversation(&mut store, &tokens);
    drop(store);
    let segment = segments(&dir.0.join(fp.hex()), "pages").remove(0);
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&segment).unwrap();
    let full = file.metadata().unwrap().len();

    // A crash while the last record was written: the file ends early.
    file.set_len(full - 10).unwrap();
    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert_eq!(store.len(), 2, "the torn record is gone, the two before it are kept");
    assert_eq!(store.states(), 1, "the checkpoints are in a segment of their own");
    assert_eq!(store.lookup(root_hash(), &tokens, PER_PAGE), hashes[..2]);
    let good_len = std::fs::metadata(&segment).unwrap().len();
    assert_eq!(good_len, 2 * RECORD);
    drop(store);

    // Garbage after the last record is cut off the same way.
    file.write_all_at(&[0x5a; 5000], good_len).unwrap();
    let store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert_eq!(store.len(), 2);
    assert_eq!(std::fs::metadata(&segment).unwrap().len(), good_len);
    drop(store);

    // A flipped bit inside a page: reading it reports the page as absent and
    // forgets it; the other page is unaffected.
    let at = BLOCK as u64 + 17;
    let mut byte = [0u8; 1];
    file.read_exact_at(&mut byte, at).unwrap();
    file.write_all_at(&[byte[0] ^ 0x04], at).unwrap();
    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    let mut out = Vec::new();
    assert_eq!(store.read(Kind::Page, hashes[0], &mut out).unwrap(), None);
    assert!(!store.contains(Kind::Page, hashes[0]));
    assert!(store.read(Kind::Page, hashes[1], &mut out).unwrap().is_some());
    assert_eq!(out, [page_bytes(&tokens, 1)]);
    // The page can be stored again.
    put_page(&mut store, hashes[0], root_hash(), 1, &page_bytes(&tokens, 0));
    assert!(store.read(Kind::Page, hashes[0], &mut out).unwrap().is_some());
    assert_eq!(out, [page_bytes(&tokens, 0)]);

    // A directory that vanishes: what is open can still be read, a new segment
    // cannot be made, and the caller is told.
    std::fs::remove_dir_all(dir.0.join(fp.hex())).unwrap();
    assert!(store.read(Kind::Page, hashes[1], &mut out).unwrap().is_some());
    let small = Config { max_bytes: 1 << 30, segment_bytes: RECORD };
    drop(store);
    let mut store = PageStore::open(&dir.0, fp, small).unwrap();
    put_page(&mut store, hashes[0], root_hash(), 1, &page_bytes(&tokens, 0));
    std::fs::remove_dir_all(dir.0.join(fp.hex())).unwrap();
    let next = store.put(Kind::Page, hashes[1], hashes[0], 2, &[Part::whole(&[1; 10])]);
    assert!(matches!(next, Err(Error::Io { .. })), "{next:?}");
}

#[test]
fn fingerprints_do_not_see_each_other_and_old_ones_make_room_first() {
    let dir = TempDir::new("fingerprints");
    let config = Config { max_bytes: 12 * RECORD, segment_bytes: 3 * RECORD };
    let tokens: Vec<u64> = (0..24).collect();

    let mut old = PageStore::open(&dir.0, fingerprint("old quantization"), config).unwrap();
    let hashes = put_conversation(&mut old, &tokens);
    assert_eq!(old.len(), 6);
    let old_dir = dir.0.join(fingerprint("old quantization").hex());

    // While the old store is open elsewhere it is not touched, even when the
    // new one cannot stay within the limit because of it.
    let mut new = PageStore::open(&dir.0, fingerprint("new quantization"), config).unwrap();
    assert!(new.is_empty(), "pages of another fingerprint are never offered");
    assert!(new.lookup(root_hash(), &tokens, PER_PAGE).is_empty());
    put_conversation(&mut new, &tokens);
    assert!(old_dir.exists());
    assert!(old.contains(Kind::Page, hashes[0]));
    drop(new);
    drop(old);

    // With the old store closed, opening the new one removes it to get
    // within the limit, and keeps its own pages.
    let new = PageStore::open(&dir.0, fingerprint("new quantization"), config).unwrap();
    assert!(!old_dir.exists());
    assert_eq!(new.len(), 6);
    assert_ne!(fingerprint("a").hex(), fingerprint("b").hex());
    assert_eq!(fingerprint("a").hex().len(), 32);
}

#[test]
fn one_process_uses_a_store_at_a_time() {
    let dir = TempDir::new("lock");
    let first = PageStore::open(&dir.0, fingerprint("a"), Config::default()).unwrap();
    match PageStore::open(&dir.0, fingerprint("a"), Config::default()) {
        Err(Error::Locked { .. }) => {}
        other => panic!("expected the store to be locked, got {other:?}"),
    }
    drop(first);
    PageStore::open(&dir.0, fingerprint("a"), Config::default()).unwrap();
}
