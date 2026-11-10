//! The store against real files: restarts, the size limit, damage, sharing a
//! root between fingerprints, and the lock.

use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use ares_state::ChainHash;
use ares_store::{Config, Error, Fingerprint, PageStore};

const PER_PAGE: u32 = 4;
/// Bytes of a test page and of its record on disk (64-byte header).
const PAGE: usize = 1000;
const RECORD: u64 = 64 + PAGE as u64;

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
fn page_bytes(tokens: &[u32], i: usize) -> Vec<u8> {
    (0..PAGE).map(|b| (tokens[i * PER_PAGE as usize] as usize * 31 + i * 7 + b) as u8).collect()
}

/// Stores every full page of `tokens`, a checkpoint on every second page, and
/// returns the page hashes.
fn put_conversation(store: &mut PageStore, tokens: &[u32]) -> Vec<ChainHash> {
    let mut hash = root_hash();
    let mut hashes = Vec::new();
    let per_page = std::hint::black_box(PER_PAGE as usize);
    for (i, page_tokens) in tokens.chunks_exact(per_page).enumerate() {
        let parent = hash;
        hash = hash.extend(page_tokens);
        let checkpoint = (i % 2 == 1).then(|| vec![i as u8; 300]);
        store.put(hash, parent, &page_bytes(tokens, i), checkpoint.as_deref()).unwrap();
        hashes.push(hash);
    }
    hashes
}

fn segments(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "pages"))
        .collect();
    files.sort();
    files
}

#[test]
fn pages_survive_a_restart_and_are_found_by_walking_the_prompt() {
    let dir = TempDir::new("restart");
    let tokens: Vec<u32> = (100..122).collect();
    let fp = fingerprint("a");

    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert!(store.is_empty());
    let hashes = put_conversation(&mut store, &tokens);
    assert_eq!(hashes.len(), 5, "22 tokens are 5 full pages of 4");
    assert_eq!(store.len(), 5);
    // Storing a page again writes nothing.
    let before = store.bytes();
    assert!(!store.put(hashes[0], root_hash(), &[1, 2, 3], None).unwrap());
    assert_eq!(store.bytes(), before);
    drop(store);

    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert_eq!(store.len(), 5);
    assert_eq!(store.bytes(), before);
    assert_eq!(store.lookup(root_hash(), &tokens, PER_PAGE), hashes);

    let (mut page, mut checkpoint) = (Vec::new(), Vec::new());
    for (i, hash) in hashes.iter().enumerate() {
        assert!(store.read(*hash, &mut page, &mut checkpoint).unwrap());
        assert_eq!(page, page_bytes(&tokens, i));
        let stored = store.stat(*hash).unwrap();
        assert_eq!(stored.parent, if i == 0 { root_hash() } else { hashes[i - 1] });
        if i % 2 == 1 {
            assert_eq!(checkpoint, vec![i as u8; 300]);
            assert_eq!(stored.checkpoint_bytes, 300);
        } else {
            assert!(checkpoint.is_empty());
        }
    }

    // A prompt that shares the first 9 tokens resumes after 2 pages, and a
    // different context finds nothing.
    let mut other = tokens[..9].to_vec();
    other.extend([7; 12]);
    assert_eq!(store.lookup(root_hash(), &other, PER_PAGE), hashes[..2]);
    assert!(store.lookup(ChainHash::root(b"other model"), &tokens, PER_PAGE).is_empty());
    assert!(!store.read(root_hash().extend(&[9, 9, 9, 9]), &mut page, &mut checkpoint).unwrap());
}

#[test]
fn the_least_recently_used_segments_go_when_the_limit_is_reached() {
    let dir = TempDir::new("limit");
    // Three records per segment, room for four segments.
    let config = Config { max_bytes: 12 * RECORD, segment_bytes: 3 * RECORD };
    let mut store = PageStore::open(&dir.0, fingerprint("a"), config).unwrap();
    let hash = |i: u32| root_hash().extend(&[i; 4]);
    let page = vec![0xabu8; PAGE];
    for i in 0..9 {
        store.put(hash(i), root_hash(), &page, None).unwrap();
    }
    assert_eq!(store.len(), 9);

    // Page 0 is used again, which makes the second segment the oldest.
    let (mut out, mut none) = (Vec::new(), Vec::new());
    assert!(store.read(hash(0), &mut out, &mut none).unwrap());
    for i in 9..15 {
        store.put(hash(i), root_hash(), &page, None).unwrap();
    }
    assert!(store.bytes() <= config.max_bytes);
    assert!(store.stat(hash(0)).is_some(), "the segment that was read stays");
    assert!(store.stat(hash(3)).is_none(), "the segment not used the longest went");
    assert!(store.stat(hash(14)).is_some());
    assert_eq!(store.len(), 12);
    assert_eq!(segments(&dir.0.join(fingerprint("a").hex())).len(), 4);

    // The limit also holds after a restart with a smaller budget.
    drop(store);
    let small = Config { max_bytes: 6 * RECORD, segment_bytes: 3 * RECORD };
    let store = PageStore::open(&dir.0, fingerprint("a"), small).unwrap();
    assert!(store.bytes() <= small.max_bytes);
    assert_eq!(store.len(), 6);

    // A record that cannot fit in a segment is refused.
    drop(store);
    let mut store = PageStore::open(&dir.0, fingerprint("a"), small).unwrap();
    let huge = vec![0u8; 4 * PAGE];
    assert!(matches!(store.put(hash(99), root_hash(), &huge, None), Err(Error::TooLarge { .. })));
}

#[test]
fn damage_costs_only_the_damaged_pages() {
    let dir = TempDir::new("damage");
    let fp = fingerprint("a");
    let tokens: Vec<u32> = (0..12).collect();
    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    let hashes = put_conversation(&mut store, &tokens);
    drop(store);
    let segment = segments(&dir.0.join(fp.hex())).remove(0);
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&segment).unwrap();
    let full = file.metadata().unwrap().len();

    // A crash while the last record was written: the file ends early.
    file.set_len(full - 10).unwrap();
    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert_eq!(store.len(), 2, "the torn record is gone, the two before it are kept");
    assert_eq!(store.lookup(root_hash(), &tokens, PER_PAGE), hashes[..2]);
    assert_eq!(std::fs::metadata(&segment).unwrap().len(), store.bytes());
    let good_len = store.bytes();
    drop(store);

    // Garbage after the last record is cut off the same way.
    file.write_all_at(&[0x5a; 500], good_len).unwrap();
    let store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    assert_eq!((store.len(), store.bytes()), (2, good_len));
    drop(store);

    // A flipped bit inside a page: reading it reports the page as absent and
    // forgets it; the other page is unaffected.
    let mut byte = [0u8; 1];
    file.read_exact_at(&mut byte, 64 + 17).unwrap();
    file.write_all_at(&[byte[0] ^ 0x04], 64 + 17).unwrap();
    let mut store = PageStore::open(&dir.0, fp, Config::default()).unwrap();
    let (mut page, mut checkpoint) = (Vec::new(), Vec::new());
    assert!(!store.read(hashes[0], &mut page, &mut checkpoint).unwrap());
    assert!(store.stat(hashes[0]).is_none());
    assert!(store.read(hashes[1], &mut page, &mut checkpoint).unwrap());
    assert_eq!(page, page_bytes(&tokens, 1));
    // The page can be stored again.
    assert!(store.put(hashes[0], root_hash(), &page_bytes(&tokens, 0), None).unwrap());
    assert!(store.read(hashes[0], &mut page, &mut checkpoint).unwrap());
    assert_eq!(page, page_bytes(&tokens, 0));
}

#[test]
fn fingerprints_do_not_see_each_other_and_old_ones_make_room_first() {
    let dir = TempDir::new("fingerprints");
    let config = Config { max_bytes: 8 * RECORD, segment_bytes: 3 * RECORD };
    let tokens: Vec<u32> = (0..24).collect();

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
    assert!(old.stat(hashes[0]).is_some());
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
