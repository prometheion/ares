//! Sequence state on disk.
//!
//! The prompt cache that survives a restart. Every full page of a sequence
//! has a name, its [`ChainHash`], that depends on the context and on all
//! tokens up to the end of the page. The store keeps pages under that name,
//! with the checkpoint of recurrent state that belongs to the page's end when
//! there is one. A new prompt then finds its cached start on disk exactly as
//! it finds it in memory: by walking its page hashes from the beginning.
//!
//! What follows from naming pages by content:
//!
//! - **Writes are incremental.** A page is written once, when it is full, and
//!   never again. Nothing is written when a model is unloaded.
//! - **Shared prefixes are stored once.** Two conversations with the same
//!   system prompt have the same first pages.
//! - **A restart loses nothing that was written.** Opening the store reads
//!   only record headers.
//!
//! The store is a cache: it never has the only copy of anything, so it trades
//! durability for speed. It does not sync, a crash can cost the last pages
//! written, and anything damaged is detected by checksums and dropped.
//!
//! # Layout
//!
//! ```text
//! <root>/<fingerprint>/LOCK
//! <root>/<fingerprint>/seg-00000000.pages
//! <root>/<fingerprint>/seg-00000001.pages
//! ```
//!
//! The [`Fingerprint`] covers everything besides tokens that changes the bytes
//! of a page (model, quantization, state formats, page size). Stores with
//! another fingerprint in the same root are never read; they count against
//! the size limit and are the first to go.
//!
//! Pages are appended to segment files. A conversation's pages are written in
//! order, so they sit together in a segment, and space is reclaimed by
//! deleting the segment that was used least recently, as a whole. That can
//! take the middle out of a chain, which makes its later pages unreachable;
//! they go with their own segments in time.
//!
//! The store holds opaque bytes. Which parts of the state a page's bytes are
//! is the business of whoever owns the devices.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use ares_state::ChainHash;

/// `EPG1`, at the start of every record.
const MAGIC: u32 = u32::from_le_bytes(*b"EPG1");
/// magic, page length, checkpoint length, reserved (4 bytes each), the page's
/// hash and its parent's (16 each), the checksum of the payload and of this
/// header (8 each).
const HEADER: usize = 64;

/// Why the store could not be used.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path}: this state store is in use by another process")]
    Locked { path: PathBuf },
    #[error(
        "a record of {record} bytes does not fit in a segment of {segment} bytes; \
         raise the segment size"
    )]
    TooLarge { record: u64, segment: u64 },
}

pub type Result<T> = std::result::Result<T, Error>;

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> Error + '_ {
    move |source| Error::Io { path: path.to_owned(), source }
}

/// The identity of what a store's pages were computed with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint([u8; 16]);

impl Fingerprint {
    /// A fingerprint of `parts`: for example the model's identity, the
    /// quantization of its weights, the formats of K and V, the page size.
    /// Settings that do not change what a page holds (drafters, sampling,
    /// how the model is split over devices) do not belong here; including
    /// them would throw the cache away for nothing.
    pub fn new(parts: &[&[u8]]) -> Self {
        let mut bytes = Vec::new();
        for part in parts {
            bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
            bytes.extend_from_slice(part);
        }
        Self(ChainHash::root(&bytes).to_bytes())
    }

    /// The name of this fingerprint's directory.
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Limits of a store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The most bytes all stores under the root may take together.
    pub max_bytes: u64,
    /// The size at which a segment is closed and a new one started. Space is
    /// reclaimed a segment at a time, so this is also the unit of eviction.
    pub segment_bytes: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self { max_bytes: 32 << 30, segment_bytes: 256 << 20 }
    }
}

/// What the store knows about a page without reading it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stored {
    pub parent: ChainHash,
    pub page_bytes: u32,
    /// 0 when the page has no checkpoint.
    pub checkpoint_bytes: u32,
}

struct Location {
    segment: u32,
    /// Of the record's header within the segment.
    offset: u64,
    page_len: u32,
    checkpoint_len: u32,
    parent: [u8; 16],
    payload_sum: u64,
}

struct Segment {
    file: File,
    path: PathBuf,
    len: u64,
    last_used: u64,
    /// When the file's modification time was last set, to do it rarely.
    stamped: Option<Instant>,
    hashes: Vec<[u8; 16]>,
}

/// A directory of another fingerprint under the same root.
struct Foreign {
    path: PathBuf,
    bytes: u64,
    modified: SystemTime,
}

/// The pages of one fingerprint on disk.
pub struct PageStore {
    dir: PathBuf,
    _lock: File,
    config: Config,
    segments: BTreeMap<u32, Segment>,
    index: HashMap<[u8; 16], Location>,
    /// The segment new pages are appended to.
    active: Option<u32>,
    foreign: Vec<Foreign>,
    clock: u64,
    bytes: u64,
}

impl std::fmt::Debug for PageStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageStore")
            .field("dir", &self.dir)
            .field("pages", &self.index.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// A 64-bit checksum over `parts`, eight bytes at a time.
fn checksum(parts: &[&[u8]]) -> u64 {
    fn fold(a: u64, b: u64) -> u64 {
        let product = u128::from(a) * u128::from(b);
        (product as u64) ^ ((product >> 64) as u64)
    }
    let mut sum = 0x9e37_79b9_7f4a_7c15u64;
    for part in parts {
        let (words, tail) = part.as_chunks::<8>();
        for word in words {
            sum = fold(sum ^ u64::from_le_bytes(*word), 0xbf58_476d_1ce4_e5b9);
        }
        let mut last = [0u8; 8];
        last[..tail.len()].copy_from_slice(tail);
        sum =
            fold(sum ^ u64::from_le_bytes(last) ^ (part.len() as u64) << 56, 0x94d0_49bb_1331_11eb);
    }
    sum
}

fn encode_header(
    hash: [u8; 16],
    parent: [u8; 16],
    page: u32,
    checkpoint: u32,
    sum: u64,
) -> [u8; HEADER] {
    let mut header = [0u8; HEADER];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&page.to_le_bytes());
    header[8..12].copy_from_slice(&checkpoint.to_le_bytes());
    header[16..32].copy_from_slice(&hash);
    header[32..48].copy_from_slice(&parent);
    header[48..56].copy_from_slice(&sum.to_le_bytes());
    let header_sum = checksum(&[&header[..56]]);
    header[56..64].copy_from_slice(&header_sum.to_le_bytes());
    header
}

/// The fields of a header that is intact: `(hash, location without segment)`.
fn decode_header(header: &[u8; HEADER], offset: u64) -> Option<([u8; 16], Location)> {
    let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("4 bytes"));
    let long = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("8 bytes"));
    if word(0) != MAGIC || long(56) != checksum(&[&header[..56]]) {
        return None;
    }
    let hash = header[16..32].try_into().expect("16 bytes");
    Some((
        hash,
        Location {
            segment: 0,
            offset,
            page_len: word(4),
            checkpoint_len: word(8),
            parent: header[32..48].try_into().expect("16 bytes"),
            payload_sum: long(48),
        },
    ))
}

fn segment_path(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("seg-{id:08}.pages"))
}

fn segment_id(path: &Path) -> Option<u32> {
    path.file_name()?.to_str()?.strip_prefix("seg-")?.strip_suffix(".pages")?.parse().ok()
}

impl PageStore {
    /// Opens the store of `fingerprint` under `root`, creating it if needed.
    ///
    /// Existing segments are read header by header; a record that is cut off
    /// or damaged ends its segment, which is shortened to the last good
    /// record. One process uses a store at a time.
    pub fn open(root: impl AsRef<Path>, fingerprint: Fingerprint, config: Config) -> Result<Self> {
        let root = root.as_ref();
        let dir = root.join(fingerprint.hex());
        fs::create_dir_all(&dir).map_err(io_error(&dir))?;

        let lock_path = dir.join("LOCK");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(io_error(&lock_path))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::Locked { path: dir }),
            Err(TryLockError::Error(source)) => return Err(Error::Io { path: lock_path, source }),
        }

        let mut store = Self {
            dir: dir.clone(),
            _lock: lock,
            config,
            segments: BTreeMap::new(),
            index: HashMap::new(),
            active: None,
            foreign: Vec::new(),
            clock: 0,
            bytes: 0,
        };

        // Own segments, least recently modified first, so the order of use
        // carries over a restart.
        let mut found = Vec::new();
        for entry in fs::read_dir(&dir).map_err(io_error(&dir))? {
            let path = entry.map_err(io_error(&dir))?.path();
            if let Some(id) = segment_id(&path) {
                let modified = fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                found.push((modified, id, path));
            }
        }
        found.sort();
        for (_, id, path) in found {
            store.load_segment(id, path)?;
        }

        // Stores of other fingerprints under the same root.
        for entry in fs::read_dir(root).map_err(io_error(root))? {
            let path = entry.map_err(io_error(root))?.path();
            if path == dir || !path.is_dir() {
                continue;
            }
            let (mut bytes, mut modified) = (0, SystemTime::UNIX_EPOCH);
            if let Ok(entries) = fs::read_dir(&path) {
                for file in entries.flatten() {
                    if let Ok(meta) = file.metadata() {
                        bytes += meta.len();
                        modified = modified.max(meta.modified().unwrap_or(modified));
                    }
                }
            }
            store.foreign.push(Foreign { path, bytes, modified });
        }
        store.foreign.sort_by_key(|f| std::cmp::Reverse(f.modified));

        store.evict_to_fit();
        Ok(store)
    }

    fn load_segment(&mut self, id: u32, path: PathBuf) -> Result<()> {
        let file =
            OpenOptions::new().read(true).write(true).open(&path).map_err(io_error(&path))?;
        let file_len = file.metadata().map_err(io_error(&path))?.len();
        let mut hashes = Vec::new();
        let mut offset = 0u64;
        let mut header = [0u8; HEADER];
        while offset + HEADER as u64 <= file_len {
            file.read_exact_at(&mut header, offset).map_err(io_error(&path))?;
            let Some((hash, mut location)) = decode_header(&header, offset) else { break };
            let end = offset
                + HEADER as u64
                + u64::from(location.page_len)
                + u64::from(location.checkpoint_len);
            if end > file_len {
                break;
            }
            location.segment = id;
            self.index.insert(hash, location);
            hashes.push(hash);
            offset = end;
        }
        if offset < file_len {
            // A torn or damaged tail: everything before it is kept.
            file.set_len(offset).map_err(io_error(&path))?;
        }
        if offset == 0 {
            drop(file);
            let _ = fs::remove_file(&path);
            return Ok(());
        }
        self.clock += 1;
        self.bytes += offset;
        self.segments.insert(
            id,
            Segment { file, path, len: offset, last_used: self.clock, stamped: None, hashes },
        );
        Ok(())
    }

    /// The number of pages in the store.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The bytes this fingerprint's segments take.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// What is stored under `hash`, without reading it or counting as a use.
    pub fn stat(&self, hash: ChainHash) -> Option<Stored> {
        let location = self.index.get(&hash.to_bytes())?;
        Some(Stored {
            parent: ChainHash::from_bytes(location.parent),
            page_bytes: location.page_len,
            checkpoint_bytes: location.checkpoint_len,
        })
    }

    fn touch(&mut self, segment: u32) {
        self.clock += 1;
        if let Some(segment) = self.segments.get_mut(&segment) {
            segment.last_used = self.clock;
            // The modification time carries the order of use over a restart.
            // Once a minute per segment is exact enough.
            if segment.stamped.is_none_or(|at| at.elapsed() > Duration::from_secs(60)) {
                let _ = segment.file.set_modified(SystemTime::now());
                segment.stamped = Some(Instant::now());
            }
        }
    }

    /// The longest prefix of `tokens` whose pages are all in the store, as
    /// their hashes in order, starting from `root`. The pages count as used.
    pub fn lookup(
        &mut self,
        root: ChainHash,
        tokens: &[u32],
        cells_per_page: u32,
    ) -> Vec<ChainHash> {
        let mut found = Vec::new();
        let mut hash = root;
        for page_tokens in tokens.chunks_exact(cells_per_page as usize) {
            hash = hash.extend(page_tokens);
            let Some(segment) = self.index.get(&hash.to_bytes()).map(|l| l.segment) else { break };
            self.touch(segment);
            found.push(hash);
        }
        found
    }

    /// Stores a page under `hash`, with the checkpoint of the state at its
    /// end if there is one. `parent` is the hash of the page before it (the
    /// root for a first page).
    ///
    /// Returns `false` and writes nothing when the page is stored already.
    pub fn put(
        &mut self,
        hash: ChainHash,
        parent: ChainHash,
        page: &[u8],
        checkpoint: Option<&[u8]>,
    ) -> Result<bool> {
        let key = hash.to_bytes();
        if let Some(segment) = self.index.get(&key).map(|l| l.segment) {
            self.touch(segment);
            return Ok(false);
        }
        let checkpoint = checkpoint.unwrap_or(&[]);
        let record = HEADER as u64 + page.len() as u64 + checkpoint.len() as u64;
        let fits_fields =
            u32::try_from(page.len()).is_ok() && u32::try_from(checkpoint.len()).is_ok();
        if record > self.config.segment_bytes || !fits_fields {
            return Err(Error::TooLarge { record, segment: self.config.segment_bytes });
        }

        let has_room = |segment: &Segment| segment.len + record <= self.config.segment_bytes;
        let id = match self.active.filter(|id| self.segments.get(id).is_some_and(has_room)) {
            Some(id) => id,
            None => {
                let id = self.segments.keys().next_back().map_or(0, |last| last + 1);
                let path = segment_path(&self.dir, id);
                let file = OpenOptions::new()
                    .create_new(true)
                    .read(true)
                    .write(true)
                    .open(&path)
                    .map_err(io_error(&path))?;
                self.segments.insert(
                    id,
                    Segment { file, path, len: 0, last_used: 0, stamped: None, hashes: Vec::new() },
                );
                self.active = Some(id);
                id
            }
        };

        let payload_sum = checksum(&[page, checkpoint]);
        let parent = parent.to_bytes();
        let header =
            encode_header(key, parent, page.len() as u32, checkpoint.len() as u32, payload_sum);
        let segment = self.segments.get_mut(&id).expect("the active segment");
        let offset = segment.len;
        // The header goes last: a record without a valid header does not
        // exist, so a crash in the middle leaves nothing to misread.
        let write = |bytes: &[u8], at: u64| segment.file.write_all_at(bytes, at);
        write(page, offset + HEADER as u64)
            .and_then(|()| write(checkpoint, offset + HEADER as u64 + page.len() as u64))
            .and_then(|()| write(&header, offset))
            .map_err(io_error(&segment.path))?;
        segment.len += record;
        segment.hashes.push(key);
        self.bytes += record;
        self.index.insert(
            key,
            Location {
                segment: id,
                offset,
                page_len: page.len() as u32,
                checkpoint_len: checkpoint.len() as u32,
                parent,
                payload_sum,
            },
        );
        self.touch(id);
        self.evict_to_fit();
        Ok(true)
    }

    /// Reads the page stored under `hash` into `page` and its checkpoint into
    /// `checkpoint` (left empty when the page has none).
    ///
    /// Returns `false` when the page is not in the store, or when its bytes
    /// no longer match their checksum; a damaged page is forgotten.
    pub fn read(
        &mut self,
        hash: ChainHash,
        page: &mut Vec<u8>,
        checkpoint: &mut Vec<u8>,
    ) -> Result<bool> {
        let key = hash.to_bytes();
        let Some(location) = self.index.get(&key) else { return Ok(false) };
        let (id, offset, sum) = (location.segment, location.offset, location.payload_sum);
        page.resize(location.page_len as usize, 0);
        checkpoint.resize(location.checkpoint_len as usize, 0);
        let segment = self.segments.get(&id).expect("indexed pages are in a segment");
        let start = offset + HEADER as u64;
        segment
            .file
            .read_exact_at(page, start)
            .and_then(|()| segment.file.read_exact_at(checkpoint, start + page.len() as u64))
            .map_err(io_error(&segment.path))?;
        if checksum(&[page, checkpoint]) != sum {
            self.index.remove(&key);
            page.clear();
            checkpoint.clear();
            return Ok(false);
        }
        self.touch(id);
        Ok(true)
    }

    /// Frees space until the limit holds: stores of other fingerprints first
    /// (the most recently used last), then this store's least recently used
    /// segments. The segment being written is never removed.
    fn evict_to_fit(&mut self) {
        let foreign_bytes = |foreign: &[Foreign]| foreign.iter().map(|f| f.bytes).sum::<u64>();
        while self.bytes + foreign_bytes(&self.foreign) > self.config.max_bytes {
            if let Some(foreign) = self.foreign.pop() {
                // Leave a store alone that another process is using.
                let in_use = File::open(foreign.path.join("LOCK"))
                    .is_ok_and(|lock| matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
                if !in_use {
                    let _ = fs::remove_dir_all(&foreign.path);
                }
                continue;
            }
            let victim = self
                .segments
                .iter()
                .filter(|(id, _)| Some(**id) != self.active)
                .min_by_key(|(_, segment)| segment.last_used)
                .map(|(id, _)| *id);
            let Some(id) = victim else { break };
            let segment = self.segments.remove(&id).expect("the victim exists");
            for hash in &segment.hashes {
                if self.index.get(hash).is_some_and(|l| l.segment == id) {
                    self.index.remove(hash);
                }
            }
            self.bytes -= segment.len;
            let _ = fs::remove_file(&segment.path);
        }
    }
}
