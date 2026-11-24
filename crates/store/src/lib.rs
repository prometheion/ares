//! Sequence state on disk.
//!
//! The prompt cache that survives a restart. Every full page of a sequence
//! has a name, its [`ChainHash`], that depends on the context and on all
//! tokens up to the end of the page. The store keeps two kinds of records
//! under that name: the page itself ([`Kind::Page`]) and the checkpoint of the
//! state that is not addressable by position at the page's end
//! ([`Kind::State`]). A new prompt then finds its cached start on disk exactly
//! as it finds it in memory: by walking its page hashes from the beginning.
//!
//! What follows from naming records by content:
//!
//! - **Writes are incremental.** A record is written once and never again.
//!   Nothing is written when a model is unloaded.
//! - **Shared prefixes are stored once.** Two conversations with the same
//!   system prompt have the same first pages.
//! - **A restart loses nothing that was written.** Opening the store reads
//!   only record headers.
//!
//! The store is a cache: it never has the only copy of anything a request
//! cannot compute again, and anything damaged is detected by checksums and
//! dropped. Whoever drops the copy in memory after a write calls
//! [`Flush::sync`] first; a crash before that can cost the records written
//! since the last one.
//!
//! # Layout
//!
//! ```text
//! <root>/<fingerprint>/LOCK
//! <root>/<fingerprint>/seg-00000000.pages
//! <root>/<fingerprint>/seg-00000001.states
//! ```
//!
//! The [`Fingerprint`] covers everything besides tokens that changes the bytes
//! of a record (model, quantization, state formats, page size, how the bytes
//! are laid out). Stores with another fingerprint in the same root are never
//! read; they count against the size limit and are the first to go.
//!
//! Records are appended to segment files, pages and states to segments of
//! their own: a conversation's pages sit together in order, and its
//! checkpoints, which later turns leave behind, age and go without taking
//! pages along. Space is reclaimed a segment at a time: first the segments
//! whose pages nothing can reach any more (a page is reachable while every
//! page before it in its chain is stored; a state while its page is), then the
//! segment that was used least recently.
//!
//! A record is a header block of [`BLOCK`] bytes followed by up to
//! [`MAX_PARTS`] parts, each padded to a multiple of [`BLOCK`], so that parts
//! are read and written with direct I/O, straight from and into buffers that
//! are aligned to [`BLOCK`] ([`Aligned`]). On a filesystem without direct I/O,
//! or with a buffer that is not aligned, the same calls go through the page
//! cache.
//!
//! # Threads
//!
//! The index is one object behind `&mut self`; the bytes are moved outside of
//! it. [`PageStore::begin`] reserves the place of a record and
//! [`PageStore::locate`] finds one; the [`Pending`] and [`Located`] they
//! return do the I/O on any thread, with no reference to the store, and
//! [`PageStore::commit`] makes a written record known.
//!
//! The store holds opaque bytes. Which parts of the state a record's parts
//! are is the business of whoever owns the devices; each part carries a tag of
//! the caller's.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use ares_state::ChainHash;

/// The unit of direct I/O: records start at multiples of it, parts are padded
/// to it, and buffers that are read into or written from directly are aligned
/// to it.
pub const BLOCK: usize = 4096;
/// The most parts a record has.
pub const MAX_PARTS: usize = 8;

/// `EPG2`, at the start of every record.
const MAGIC: u32 = u32::from_le_bytes(*b"EPG2");
/// The bytes of the header block that carry the header; the rest is zero.
const HEADER: usize = 128;

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
    #[error("a record has {parts} parts; the store takes at most {MAX_PARTS}")]
    TooManyParts { parts: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> Error + '_ {
    move |source| Error::Io { path: path.to_owned(), source }
}

/// The identity of what a store's records were computed with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint([u8; 16]);

impl Fingerprint {
    /// A fingerprint of `parts`: for example the model's identity, the
    /// quantization of its weights, the formats of K and V, the page size.
    /// Settings that do not change what a record holds (sampling, drafters
    /// that keep no state) do not belong here; including them would throw the
    /// cache away for nothing.
    pub fn new(parts: &[&[u8]]) -> Self {
        let mut bytes = Vec::new();
        for part in parts {
            bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
            bytes.extend_from_slice(part);
        }
        Self(ChainHash::root(&bytes).to_bytes())
    }

    /// The fingerprint as bytes, for example as the context of a chain's root.
    pub fn bytes(&self) -> [u8; 16] {
        self.0
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

/// What a record is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// The state per position of one page.
    Page,
    /// The checkpoint of the state at the end of a page.
    State,
}

impl Kind {
    fn extension(self) -> &'static str {
        match self {
            Kind::Page => "pages",
            Kind::State => "states",
        }
    }
}

/// `len` rounded up to a multiple of [`BLOCK`].
pub const fn padded(len: usize) -> usize {
    len.div_ceil(BLOCK) * BLOCK
}

/// Zeroed memory aligned to [`BLOCK`], whose length is a multiple of it: what
/// direct I/O reads into and writes from.
pub struct Aligned {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer owns its memory exclusively, like a `Box<[u8]>`.
unsafe impl Send for Aligned {}
// SAFETY: shared access only hands out `&[u8]`.
unsafe impl Sync for Aligned {}

impl Aligned {
    /// At least `len` zeroed bytes (and at least one block).
    pub fn new(len: usize) -> Self {
        let len = padded(len.max(1));
        let layout = std::alloc::Layout::from_size_align(len, BLOCK).expect("a block layout");
        // SAFETY: the layout has a size above zero.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else { std::alloc::handle_alloc_error(layout) };
        Self { ptr, len }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, BLOCK).expect("a block layout");
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

impl std::ops::Deref for Aligned {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: `len` bytes are allocated and initialised, owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `deref`, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl std::fmt::Debug for Aligned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Aligned").field("len", &self.len).finish()
    }
}

/// One part of a record to write: the first `len` bytes of `buf`, with a tag
/// of the caller's. The part is written with direct I/O when `buf` is aligned
/// to [`BLOCK`] and holds the padded length.
#[derive(Clone, Copy, Debug)]
pub struct Part<'a> {
    pub buf: &'a [u8],
    pub len: usize,
    pub tag: u32,
}

impl<'a> Part<'a> {
    /// All of `bytes`, with tag 0.
    #[cfg(test)]
    pub(crate) fn whole(bytes: &'a [u8]) -> Self {
        Self { buf: bytes, len: bytes.len(), tag: 0 }
    }
}

/// The length and the tag of each part of a stored record.
pub type Parts = Vec<(u32, u32)>;

/// What the store knows about a record without reading it.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stored {
    pub parent: ChainHash,
    /// The number of the page in its chain, 1 for a first page.
    pub depth: u32,
    pub parts: Parts,
}

#[derive(Clone, Copy, Debug)]
struct Location {
    segment: u32,
    /// Of the record's header block within the segment.
    offset: u64,
    parent: [u8; 16],
    depth: u32,
    payload_sum: u64,
    n_parts: u8,
    parts: [(u32, u32); MAX_PARTS],
}

impl Location {
    fn parts(&self) -> &[(u32, u32)] {
        &self.parts[..self.n_parts as usize]
    }

    fn record_bytes(&self) -> u64 {
        record_bytes(self.parts().iter().map(|(len, _)| *len))
    }
}

fn record_bytes(lens: impl Iterator<Item = u32>) -> u64 {
    BLOCK as u64 + lens.map(|len| padded(len as usize) as u64).sum::<u64>()
}

/// A segment file, opened for the page cache and, where the filesystem has
/// it, for direct I/O.
#[derive(Debug)]
struct SegFile {
    buffered: File,
    direct: Option<File>,
    path: PathBuf,
}

impl SegFile {
    fn open(path: &Path, create: bool) -> io::Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(create);
        let buffered = options.open(path)?;
        let direct =
            OpenOptions::new().read(true).write(true).custom_flags(libc::O_DIRECT).open(path).ok();
        Ok(Self { buffered, direct, path: path.to_owned() })
    }

    fn direct_for(&self, ptr: *const u8, room: usize, len: usize) -> Option<&File> {
        let aligned = (ptr as usize).is_multiple_of(BLOCK) && room >= padded(len);
        self.direct.as_ref().filter(|_| aligned)
    }

    /// Writes `len` bytes of `buf` at `offset` (a multiple of [`BLOCK`]) and
    /// zeros up to the next block.
    fn write(&self, buf: &[u8], len: usize, offset: u64) -> io::Result<()> {
        if let Some(direct) = self.direct_for(buf.as_ptr(), buf.len(), len)
            && direct.write_all_at(&buf[..padded(len)], offset).is_ok()
        {
            return Ok(());
        }
        self.buffered.write_all_at(&buf[..len], offset)?;
        let pad = padded(len) - len;
        self.buffered.write_all_at(&[0u8; BLOCK][..pad], offset + len as u64)
    }

    /// Reads `len` bytes at `offset` into the start of `buf`.
    fn read(&self, buf: &mut [u8], len: usize, offset: u64) -> io::Result<()> {
        if let Some(direct) = self.direct_for(buf.as_ptr(), buf.len(), len)
            && direct.read_exact_at(&mut buf[..padded(len)], offset).is_ok()
        {
            return Ok(());
        }
        self.buffered.read_exact_at(&mut buf[..len], offset)
    }
}

struct Segment {
    file: Arc<SegFile>,
    kind: Kind,
    len: u64,
    last_used: u64,
    /// When the file's modification time was last set, to do it rarely.
    stamped: Option<SystemTime>,
    hashes: Vec<[u8; 16]>,
    /// Records were written since the last [`PageStore::flush`].
    dirty: bool,
}

/// A directory of another fingerprint under the same root.
struct Foreign {
    path: PathBuf,
    bytes: u64,
    modified: SystemTime,
}

/// The time the store stamps its segments with.
pub type Now = Arc<dyn Fn() -> SystemTime + Send + Sync>;

/// The records of one fingerprint on disk.
pub struct PageStore {
    dir: PathBuf,
    _lock: File,
    config: Config,
    segments: BTreeMap<u32, Segment>,
    pages: HashMap<[u8; 16], Location>,
    states: HashMap<[u8; 16], Location>,
    /// The segments new records are appended to: pages, states.
    active: [Option<u32>; 2],
    foreign: Vec<Foreign>,
    clock: u64,
    bytes: u64,
    now: Now,
    evicted: u64,
}

impl std::fmt::Debug for PageStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageStore")
            .field("dir", &self.dir)
            .field("pages", &self.pages.len())
            .field("states", &self.states.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// A 64-bit checksum over `parts`, eight bytes at a time.
fn checksum<'a>(parts: impl Iterator<Item = &'a [u8]>) -> u64 {
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

fn encode_header(kind: Kind, hash: [u8; 16], location: &Location) -> [u8; HEADER] {
    let mut header = [0u8; HEADER];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&(kind as u32).to_le_bytes());
    header[8..12].copy_from_slice(&location.depth.to_le_bytes());
    header[12..16].copy_from_slice(&u32::from(location.n_parts).to_le_bytes());
    header[16..32].copy_from_slice(&hash);
    header[32..48].copy_from_slice(&location.parent);
    header[48..56].copy_from_slice(&location.payload_sum.to_le_bytes());
    for (i, (len, tag)) in location.parts.iter().enumerate() {
        header[56 + i * 8..60 + i * 8].copy_from_slice(&len.to_le_bytes());
        header[60 + i * 8..64 + i * 8].copy_from_slice(&tag.to_le_bytes());
    }
    let sum = checksum([&header[..120]].into_iter());
    header[120..128].copy_from_slice(&sum.to_le_bytes());
    header
}

/// The fields of a header that is intact: its kind, its hash and its location
/// (without the segment).
fn decode_header(header: &[u8; HEADER], offset: u64) -> Option<(Kind, [u8; 16], Location)> {
    let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("4 bytes"));
    let long = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().expect("8 bytes"));
    if word(0) != MAGIC || long(120) != checksum([&header[..120]].into_iter()) {
        return None;
    }
    let kind = match word(4) {
        0 => Kind::Page,
        1 => Kind::State,
        _ => return None,
    };
    let n_parts = word(12);
    if n_parts as usize > MAX_PARTS {
        return None;
    }
    let mut parts = [(0, 0); MAX_PARTS];
    for (i, part) in parts.iter_mut().enumerate() {
        *part = (word(56 + i * 8), word(60 + i * 8));
    }
    let location = Location {
        segment: 0,
        offset,
        parent: header[32..48].try_into().expect("16 bytes"),
        depth: word(8),
        payload_sum: long(48),
        n_parts: n_parts as u8,
        parts,
    };
    Some((kind, header[16..32].try_into().expect("16 bytes"), location))
}

fn segment_path(dir: &Path, id: u32, kind: Kind) -> PathBuf {
    dir.join(format!("seg-{id:08}.{}", kind.extension()))
}

fn segment_id(path: &Path) -> Option<(u32, Kind)> {
    let name = path.file_name()?.to_str()?.strip_prefix("seg-")?;
    let (id, extension) = name.split_once('.')?;
    let kind = [Kind::Page, Kind::State].into_iter().find(|k| k.extension() == extension)?;
    Some((id.parse().ok()?, kind))
}

/// The place of a record that is being written. Dropping it without
/// [`PageStore::commit`] leaves nothing behind that a reader could find.
#[derive(Debug)]
pub struct Pending {
    file: Arc<SegFile>,
    kind: Kind,
    hash: [u8; 16],
    location: Location,
    written: bool,
}

impl Pending {
    /// Writes the record: its parts, then its header, so that a crash in the
    /// middle leaves nothing to misread. `parts` are the parts the record was
    /// begun with, in order, with the same lengths.
    pub fn write(&mut self, parts: &[Part<'_>]) -> Result<()> {
        assert_eq!(parts.len(), self.location.n_parts as usize, "the parts the record began with");
        let mut at = self.location.offset + BLOCK as u64;
        self.location.payload_sum = checksum(parts.iter().map(|part| &part.buf[..part.len]));
        let path = self.file.path.clone();
        for (i, part) in parts.iter().enumerate() {
            assert_eq!(part.len as u32, self.location.parts[i].0, "the length it began with");
            self.location.parts[i].1 = part.tag;
            self.file.write(part.buf, part.len, at).map_err(io_error(&path))?;
            at += padded(part.len) as u64;
        }
        let mut block = Aligned::new(BLOCK);
        block[..HEADER].copy_from_slice(&encode_header(self.kind, self.hash, &self.location));
        self.file.write(&block, BLOCK, self.location.offset).map_err(io_error(&path))?;
        self.written = true;
        Ok(())
    }
}

/// A record that was found: reads its parts on any thread.
#[derive(Clone, Debug)]
pub struct Located {
    file: Arc<SegFile>,
    offset: u64,
    payload_sum: u64,
    parts: Parts,
}

impl Located {
    /// The length and the tag of each part.
    pub fn parts(&self) -> &[(u32, u32)] {
        &self.parts
    }

    /// Reads part `i` into the start of `buf`, with direct I/O when `buf` is
    /// aligned to [`BLOCK`] and holds the part's padded length.
    pub fn read_part(&self, i: usize, buf: &mut [u8]) -> io::Result<()> {
        let before: u64 = self.parts[..i].iter().map(|(len, _)| padded(*len as usize) as u64).sum();
        let len = self.parts[i].0 as usize;
        if buf.len() < len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "the buffer is too small"));
        }
        self.file.read(buf, len, self.offset + BLOCK as u64 + before)
    }

    /// Whether `parts`, the bytes that were read for each part in order, are
    /// what was written.
    pub fn verify(&self, parts: &[&[u8]]) -> bool {
        parts.len() == self.parts.len()
            && parts.iter().zip(&self.parts).all(|(part, (len, _))| part.len() == *len as usize)
            && checksum(parts.iter().copied()) == self.payload_sum
    }
}

/// The segments that were written since the last flush; syncing them makes
/// their records durable.
#[derive(Debug)]
pub struct Flush(Vec<Arc<SegFile>>);

impl Flush {
    /// Waits until the drive has the data.
    pub fn sync(&self) -> io::Result<()> {
        self.0.iter().try_for_each(|file| file.buffered.sync_data())
    }
}

impl PageStore {
    /// Opens the store of `fingerprint` under `root`, creating it if needed.
    ///
    /// Existing segments are read header by header; a record that is cut off
    /// or damaged ends its segment, which is shortened to the last good
    /// record. One process uses a store at a time.
    pub fn open(root: impl AsRef<Path>, fingerprint: Fingerprint, config: Config) -> Result<Self> {
        Self::open_with(root, fingerprint, config, Arc::new(SystemTime::now))
    }

    /// [`PageStore::open`] with the clock that stamps segments with the time
    /// of their last use, which carries the order of use over a restart.
    pub fn open_with(
        root: impl AsRef<Path>,
        fingerprint: Fingerprint,
        config: Config,
        now: Now,
    ) -> Result<Self> {
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
            pages: HashMap::new(),
            states: HashMap::new(),
            active: [None; 2],
            foreign: Vec::new(),
            clock: 0,
            bytes: 0,
            now,
            evicted: 0,
        };

        // Own segments, least recently modified first, so the order of use
        // carries over a restart.
        let mut found = Vec::new();
        for entry in fs::read_dir(&dir).map_err(io_error(&dir))? {
            let path = entry.map_err(io_error(&dir))?.path();
            if let Some((id, kind)) = segment_id(&path) {
                let modified = fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                found.push((modified, id, kind, path));
            }
        }
        found.sort();
        for (_, id, kind, path) in found {
            store.load_segment(id, kind, path)?;
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

    fn index(&mut self, kind: Kind) -> &mut HashMap<[u8; 16], Location> {
        match kind {
            Kind::Page => &mut self.pages,
            Kind::State => &mut self.states,
        }
    }

    fn index_of(&self, kind: Kind) -> &HashMap<[u8; 16], Location> {
        match kind {
            Kind::Page => &self.pages,
            Kind::State => &self.states,
        }
    }

    fn load_segment(&mut self, id: u32, kind: Kind, path: PathBuf) -> Result<()> {
        let file = SegFile::open(&path, false).map_err(io_error(&path))?;
        let file_len = file.buffered.metadata().map_err(io_error(&path))?.len();
        let mut hashes = Vec::new();
        let mut offset = 0u64;
        let mut header = [0u8; HEADER];
        while offset + BLOCK as u64 <= file_len {
            file.buffered.read_exact_at(&mut header, offset).map_err(io_error(&path))?;
            let Some((of, hash, mut location)) = decode_header(&header, offset) else { break };
            let end = offset + location.record_bytes();
            if end > file_len || of != kind {
                break;
            }
            location.segment = id;
            self.index(kind).insert(hash, location);
            hashes.push(hash);
            offset = end;
        }
        if offset < file_len {
            // A torn or damaged tail: everything before it is kept.
            file.buffered.set_len(offset).map_err(io_error(&path))?;
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
            Segment {
                file: Arc::new(file),
                kind,
                len: offset,
                last_used: self.clock,
                stamped: None,
                hashes,
                dirty: false,
            },
        );
        Ok(())
    }

    /// The number of pages in the store.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// The number of checkpoints in the store.
    pub fn states(&self) -> usize {
        self.states.len()
    }

    /// The bytes this fingerprint's segments take.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The segments the size limit has removed since the store was opened.
    #[cfg(test)]
    pub(crate) fn evicted_segments(&self) -> u64 {
        self.evicted
    }

    /// Whether a record of `kind` is stored under `hash`.
    pub fn contains(&self, kind: Kind, hash: ChainHash) -> bool {
        self.index_of(kind).contains_key(&hash.to_bytes())
    }

    /// What is stored under `hash`, without reading it or counting as a use.
    #[cfg(test)]
    pub(crate) fn stat(&self, kind: Kind, hash: ChainHash) -> Option<Stored> {
        let location = self.index_of(kind).get(&hash.to_bytes())?;
        Some(Stored {
            parent: ChainHash::from_bytes(location.parent),
            depth: location.depth,
            parts: location.parts().to_vec(),
        })
    }

    fn touch(&mut self, segment: u32) {
        self.clock += 1;
        if let Some(segment) = self.segments.get_mut(&segment) {
            segment.last_used = self.clock;
            // The modification time carries the order of use over a restart.
            // Once a minute per segment is exact enough.
            let now = (self.now)();
            let due = segment.stamped.is_none_or(|at| {
                now.duration_since(at).map_or(true, |since| since > Duration::from_secs(60))
            });
            if due {
                let _ = segment.file.buffered.set_modified(now);
                segment.stamped = Some(now);
            }
        }
    }

    /// The longest prefix of `tokens` (the names of cells, `ares_state`) whose pages are all in the store, as
    /// their hashes in order, starting from `root`. The pages count as used.
    #[cfg(test)]
    pub(crate) fn lookup(
        &mut self,
        root: ChainHash,
        tokens: &[u64],
        cells_per_page: u32,
    ) -> Vec<ChainHash> {
        let mut found = Vec::new();
        let mut hash = root;
        for page_tokens in tokens.chunks_exact(cells_per_page as usize) {
            hash = hash.extend(page_tokens);
            let Some(segment) = self.pages.get(&hash.to_bytes()).map(|l| l.segment) else { break };
            self.touch(segment);
            found.push(hash);
        }
        found
    }

    /// Reserves the place of a record of `kind` under `hash`, whose parts have
    /// the lengths `lens`. `parent` is the hash of the page before it (the root
    /// for a first page) and `depth` the number of the page in its chain.
    ///
    /// Returns `None` and reserves nothing when such a record is stored
    /// already. The record exists once [`Pending::write`] has run and
    /// [`PageStore::commit`] was told; a reservation that is dropped is a gap
    /// that the next record of the segment overwrites or the next open cuts.
    pub fn begin(
        &mut self,
        kind: Kind,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
        lens: &[u32],
    ) -> Result<Option<Pending>> {
        let key = hash.to_bytes();
        if let Some(segment) = self.index_of(kind).get(&key).map(|l| l.segment) {
            self.touch(segment);
            return Ok(None);
        }
        if lens.len() > MAX_PARTS {
            return Err(Error::TooManyParts { parts: lens.len() });
        }
        let record = record_bytes(lens.iter().copied());
        if record > self.config.segment_bytes {
            return Err(Error::TooLarge { record, segment: self.config.segment_bytes });
        }
        let which = kind as usize;
        let has_room = |segment: &Segment| segment.len + record <= self.config.segment_bytes;
        let id = match self.active[which].filter(|id| self.segments.get(id).is_some_and(has_room)) {
            Some(id) => id,
            None => {
                let id = self.segments.keys().next_back().map_or(0, |last| last + 1);
                let path = segment_path(&self.dir, id, kind);
                let file = SegFile::open(&path, true).map_err(io_error(&path))?;
                self.segments.insert(
                    id,
                    Segment {
                        file: Arc::new(file),
                        kind,
                        len: 0,
                        last_used: 0,
                        stamped: None,
                        hashes: Vec::new(),
                        dirty: false,
                    },
                );
                self.active[which] = Some(id);
                id
            }
        };
        let segment = self.segments.get(&id).expect("the active segment");
        let mut parts = [(0, 0); MAX_PARTS];
        for (part, len) in parts.iter_mut().zip(lens) {
            part.0 = *len;
        }
        Ok(Some(Pending {
            file: segment.file.clone(),
            kind,
            hash: key,
            location: Location {
                segment: id,
                offset: segment.len,
                parent: parent.to_bytes(),
                depth,
                payload_sum: 0,
                n_parts: lens.len() as u8,
                parts,
            },
            written: false,
        }))
    }

    /// Makes the record `pending` wrote known. A record that was not written,
    /// or whose segment the store has given up or moved past meanwhile, is
    /// dropped. Frees space when the limit is passed.
    pub fn commit(&mut self, pending: Pending) -> bool {
        let Pending { kind, hash, location, written, .. } = pending;
        let record = location.record_bytes();
        let fits = self
            .segments
            .get(&location.segment)
            .is_some_and(|segment| segment.len == location.offset);
        if !written || !fits || self.index_of(kind).contains_key(&hash) {
            return false;
        }
        let segment = self.segments.get_mut(&location.segment).expect("checked above");
        segment.len += record;
        segment.hashes.push(hash);
        segment.dirty = true;
        self.bytes += record;
        self.index(kind).insert(hash, location);
        self.touch(location.segment);
        self.evict_to_fit();
        true
    }

    /// The segments written since the last call, to sync outside of the store.
    pub fn flush(&mut self) -> Flush {
        let dirty = self.segments.values_mut().filter(|segment| segment.dirty);
        Flush(
            dirty
                .map(|segment| {
                    segment.dirty = false;
                    segment.file.clone()
                })
                .collect(),
        )
    }

    /// Stores a record in one call: [`PageStore::begin`], [`Pending::write`]
    /// and [`PageStore::commit`]. Returns `false` and writes nothing when the
    /// record is stored already.
    #[cfg(test)]
    pub(crate) fn put(
        &mut self,
        kind: Kind,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
        parts: &[Part<'_>],
    ) -> Result<bool> {
        let lens: Vec<u32> = parts.iter().map(|part| part.len as u32).collect();
        if parts.iter().any(|part| u32::try_from(part.len).is_err()) {
            let record = parts.iter().map(|part| part.len as u64).sum();
            return Err(Error::TooLarge { record, segment: self.config.segment_bytes });
        }
        let Some(mut pending) = self.begin(kind, hash, parent, depth, &lens)? else {
            return Ok(false);
        };
        pending.write(parts)?;
        Ok(self.commit(pending))
    }

    /// Finds the record of `kind` under `hash`, which counts as a use.
    pub fn locate(&mut self, kind: Kind, hash: ChainHash) -> Option<Located> {
        let location = *self.index_of(kind).get(&hash.to_bytes())?;
        let file = self.segments.get(&location.segment)?.file.clone();
        self.touch(location.segment);
        Some(Located {
            file,
            offset: location.offset,
            payload_sum: location.payload_sum,
            parts: location.parts().to_vec(),
        })
    }

    /// Forgets the record of `kind` under `hash`: its bytes were read and are
    /// not what was written. Its space goes with its segment.
    pub fn forget(&mut self, kind: Kind, hash: ChainHash) {
        self.index(kind).remove(&hash.to_bytes());
    }

    /// Reads the record of `kind` under `hash` in one call, a vector per
    /// part, and returns the parts' tags.
    ///
    /// Returns `None` when there is no such record, or when its bytes no
    /// longer match their checksum; a damaged record is forgotten.
    #[cfg(test)]
    pub(crate) fn read(
        &mut self,
        kind: Kind,
        hash: ChainHash,
        out: &mut Vec<Vec<u8>>,
    ) -> Result<Option<Vec<u32>>> {
        out.clear();
        let Some(located) = self.locate(kind, hash) else { return Ok(None) };
        for (i, (len, _)) in located.parts().iter().enumerate() {
            let mut part = vec![0u8; *len as usize];
            located.read_part(i, &mut part).map_err(io_error(&located.file.path))?;
            out.push(part);
        }
        let slices: Vec<&[u8]> = out.iter().map(Vec::as_slice).collect();
        if !located.verify(&slices) {
            self.forget(kind, hash);
            out.clear();
            return Ok(None);
        }
        Ok(Some(located.parts().iter().map(|(_, tag)| *tag).collect()))
    }

    /// Per segment that is not being written, whether a request can still
    /// reach one of its records: a page while every page before it in its
    /// chain is stored, a state while its page is.
    fn reachable_segments(&self) -> HashMap<u32, bool> {
        let mut by_depth: Vec<(&[u8; 16], &Location)> = self.pages.iter().collect();
        by_depth.sort_by_key(|(_, location)| location.depth);
        let mut reachable: HashMap<&[u8; 16], bool> = HashMap::with_capacity(by_depth.len());
        let mut segments: HashMap<u32, bool> = HashMap::new();
        for (hash, location) in by_depth {
            let is = location.depth <= 1 || reachable.get(&location.parent) == Some(&true);
            reachable.insert(hash, is);
            *segments.entry(location.segment).or_default() |= is;
        }
        for (hash, location) in &self.states {
            let is = reachable.get(hash) == Some(&true);
            *segments.entry(location.segment).or_default() |= is;
        }
        segments
    }

    fn remove_segment(&mut self, id: u32) {
        let Some(segment) = self.segments.remove(&id) else { return };
        let index = self.index(segment.kind);
        for hash in &segment.hashes {
            if index.get(hash).is_some_and(|l| l.segment == id) {
                index.remove(hash);
            }
        }
        self.bytes -= segment.len;
        self.evicted += 1;
        let _ = fs::remove_file(&segment.file.path);
    }

    /// The segment to give up next: one that nothing reaches before one that
    /// something does, and among equals the least recently used. The segments
    /// being written stay.
    fn victim(&self) -> Option<u32> {
        let reachable = self.reachable_segments();
        self.segments
            .iter()
            .filter(|(id, _)| !self.active.contains(&Some(**id)))
            .min_by_key(|(id, segment)| {
                (reachable.get(id).copied().unwrap_or(false), segment.last_used)
            })
            .map(|(id, _)| *id)
    }

    /// Frees space until the limit holds: stores of other fingerprints first
    /// (the most recently used last), then this store's segments in the order
    /// of [`PageStore::victim`].
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
            let Some(id) = self.victim() else { break };
            self.remove_segment(id);
        }
    }

    /// Gives up segments until at least `bytes` are free again, whatever the
    /// limit says: for a drive that is full. Returns the bytes freed.
    pub fn make_room(&mut self, bytes: u64) -> u64 {
        let before = self.bytes;
        while before - self.bytes < bytes {
            let Some(id) = self.victim() else { break };
            self.remove_segment(id);
        }
        before - self.bytes
    }
}

#[cfg(test)]
mod tests;
