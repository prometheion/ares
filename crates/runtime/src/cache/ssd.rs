//! The level of the prefix cache that is on SSD.
//!
//! Pages and checkpoints of recurrent state are written to an
//! [`ares_store::PageStore`] under the names the index gives them, and a
//! prompt whose start is in none of the levels above but in the store has it
//! brought back before it starts. The cache decides what moves and when; the
//! bytes move in three places, none of which is a step:
//!
//! - between the devices and host memory in [`Executor::transfer`], which the
//!   conductor calls between steps ([`Cache::pump`]);
//! - from host memory to the drive on one writer thread;
//! - from the drive to host memory on a few reader threads.
//!
//! # What is written, and when
//!
//! Only what leaves RAM (`cache/ram.rs`), and everything worth writing when
//! the engine ends ([`Cache::drain`]). Nothing is written as it is computed:
//!
//! - **A page** when it leaves RAM, straight from its buffer there, if its
//!   text was long enough ([`SsdConfig::min_cells`]); RAM writes the coldest
//!   of what only it holds a little ahead when it is full, so that what comes
//!   from the devices finds room. A page that the store has is not written
//!   again: its name says what it holds.
//! - **A checkpoint** at a page boundary, at least that deep, when it leaves
//!   RAM: copied out of its place into a buffer of its own first. One that
//!   later requests have left behind at deeper page boundaries is not
//!   written: a deeper one on its branch is where a prompt resumes.
//! - **Never** a tail (the partly used last page of a sequence) and the state
//!   at its end: a prompt that resumes from the drive restores the checkpoint
//!   before them and replays the rows between.
//!
//! The writer syncs the drive a second after it has nothing left to write,
//! and every ten seconds while records keep coming, so what was written is
//! durable shortly after.
//!
//! # A hit
//!
//! [`Cache::probe`] asks the store for the pages after the ones the index
//! (and RAM) have. A run of them is announced in the index under an owner of
//! its own, which holds the prompt back as a page that another sequence
//! computes would. The readers bring the pages into buffers of RAM and the
//! deepest checkpoint of the run into a buffer of its own; the cache gives
//! each page a page of the pool, the devices take the bytes, and the page
//! enters the index (and stays in RAM as a copy). The checkpoint goes into a
//! checkpoint place and is attached to its page before that page is
//! published. The prompt then starts as from any cached prefix: it restores
//! the checkpoint, replays the rows after it and computes the rest.
//!
//! # Failure
//!
//! The store is a cache. A record that cannot be written is dropped, a
//! record that cannot be read or fails its checksum ends its run (the prompt
//! computes from there), a full drive gives up old segments, and a directory
//! that is gone is made again. None of it reaches a request.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ares_state::{Chain, ChainHash, CheckpointId, Owner, PageId, PagePool, PageState, Sequence};
use ares_store::{Aligned, Fingerprint, Kind, Located, PageStore, Part};

use super::ram::{Entry, Item};
use super::{Cache, Note, RESERVE, StateOut};
use crate::exec::{Executor, HostBytes, Transfer, TransferKind, TransitPart};
use crate::model::Durable;

/// Checkpoints in transit at once.
const STATE_BUFFERS: usize = 3;
/// How long the step loop waits for the writer to give a buffer back, for a
/// checkpoint that must leave its place now or a page that would be lost
/// otherwise, before it gives that up.
const PATIENCE: Duration = Duration::from_millis(250);
/// Threads that read from the drive.
const READERS: usize = 4;
/// Pages read from the drive that wait for their way to the devices, at
/// most: half the floor of pages in RAM, so that the other half is there for
/// pages that leave the devices to make room for them.
const READ_AHEAD: usize = RESERVE as usize / 2;
/// How long an engine that ends waits for the writer at most.
const DRAIN_LIMIT: Duration = Duration::from_secs(120);

/// Where the level on SSD lies and how large it may get.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsdConfig {
    /// The directory. Stores of several models and settings share it; each
    /// has a directory of its own in it, named by its fingerprint.
    pub dir: PathBuf,
    /// The most bytes everything in the directory may take.
    pub max_bytes: u64,
    /// The size of a segment file, the unit in which space is reclaimed.
    pub segment_bytes: u64,
    /// The settled cells a sequence must hold before anything of it is
    /// written.
    pub min_cells: u32,
}

impl SsdConfig {
    /// The most host memory the level on SSD takes for checkpoints in
    /// transit, for a model that is `durable`: ordinary memory, not
    /// page-locked, allocated as the buffers first come into use and kept.
    /// It is not in the books of the page-locked RAM
    /// ([`crate::PinnedRam`]); who sets the engine's RAM limit counts it
    /// next to them. Pages pass through buffers of the prompt cache in RAM,
    /// which are in those books ([`SsdConfig::floor_ram`]).
    pub fn transit_ram(durable: &Durable) -> u64 {
        STATE_BUFFERS as u64 * Self::room(&durable.state_bytes)
    }

    /// The page-locked RAM the prompt cache keeps whatever the machine has
    /// when it has a level on SSD, for a model that is `durable` and whose
    /// checkpoints take `checkpoint` bytes: one checkpoint place, and the
    /// pages that pass through RAM on their way to the drive and back.
    pub fn floor_ram(durable: &Durable, checkpoint: u64) -> u64 {
        checkpoint + RESERVE * Self::room(&durable.page_bytes)
    }

    fn room(bytes: &[u64]) -> u64 {
        bytes.iter().map(|b| ares_store::padded(*b as usize) as u64).sum()
    }

    /// The engine's own numbers for a store in `dir` of at most `max_bytes`.
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> Self {
        Self {
            dir: dir.into(),
            max_bytes,
            segment_bytes: ares_store::Config::default().segment_bytes,
            min_cells: 2048,
        }
    }
}

/// What the level on SSD has done since the engine started, and what it
/// holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsdCounters {
    /// The level is in use.
    pub enabled: bool,
    pub pages_written: u64,
    pub states_written: u64,
    /// Bytes of records written, with their headers and padding.
    pub bytes_written: u64,
    /// Nanoseconds the writer thread spent writing and syncing.
    pub write_ns: u64,
    /// Records that could not be written.
    pub write_errors: u64,
    pub pages_read: u64,
    pub states_read: u64,
    pub bytes_read: u64,
    /// Nanoseconds the reader threads spent reading and checking, summed.
    pub read_ns: u64,
    /// Runs of pages brought back from the drive, and the ones that ended
    /// early.
    pub imports: u64,
    pub imports_cut: u64,
    /// Transfers between the devices and the host: calls, and the
    /// nanoseconds the conductor waited for them.
    pub transfers: u64,
    pub transfer_ns: u64,
    /// Checkpoints that were due and were not written: none could be in
    /// transit (the writer held every buffer), or the checkpoint was given
    /// up before the record with its save had run, so that its place never
    /// held it.
    pub states_skipped: u64,
    /// What the store holds now.
    pub store_pages: u64,
    pub store_states: u64,
    pub store_bytes: u64,
    /// Host memory the buffers of checkpoints in transit take.
    pub transit_bytes: u64,
    /// Pages and checkpoints that are being written.
    pub pending: u64,
}

struct Shared {
    store: Mutex<PageStore>,
    root: PathBuf,
    fingerprint: Fingerprint,
    config: ares_store::Config,
    /// Nanoseconds the writer spent in syncs.
    sync_ns: AtomicU64,
}

impl Shared {
    fn store(&self) -> MutexGuard<'_, PageStore> {
        self.store.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes one record. Returns the bytes it takes on the drive, 0 when the
    /// store had it already.
    fn write(
        &self,
        kind: Kind,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
        parts: &[TransitPart],
    ) -> ares_store::Result<u64> {
        let lens: Vec<u32> = parts.iter().map(|part| part.len as u32).collect();
        let io: Vec<Part<'_>> =
            parts.iter().map(|p| Part { buf: &p.mem, len: p.len, tag: p.tag }).collect();
        let mut attempt = 0;
        loop {
            attempt += 1;
            // The store is locked to reserve the record's place and to make
            // it known, not while its bytes are written.
            let mut store = self.store();
            let begun = store.begin(kind, hash, parent, depth, &lens);
            drop(store);
            let written = begun.and_then(|begun| {
                let Some(mut pending) = begun else { return Ok(0) };
                pending.write(&io)?;
                let bytes = ares_store::BLOCK as u64
                    + io.iter().map(|part| ares_store::padded(part.len) as u64).sum::<u64>();
                Ok(if self.store().commit(pending) { bytes } else { 0 })
            });
            match written {
                Err(error) if attempt == 1 && self.recover(&error) => {}
                other => return other,
            }
        }
    }

    /// Whether something was done about `error` that makes another try worth
    /// it: room on a full drive, a new directory for one that is gone.
    fn recover(&self, error: &ares_store::Error) -> bool {
        let ares_store::Error::Io { source, .. } = error else { return false };
        match source.raw_os_error() {
            Some(libc::ENOSPC) | Some(libc::EDQUOT) => {
                self.store().make_room(self.config.segment_bytes) > 0
            }
            Some(libc::ENOENT) => self.reopen(),
            _ => false,
        }
    }

    /// Opens the store anew, in a new directory when the old one is gone.
    fn reopen(&self) -> bool {
        match PageStore::open(&self.root, self.fingerprint, self.config) {
            Ok(fresh) => {
                *self.store() = fresh;
                true
            }
            Err(_) => false,
        }
    }

    /// Lets what was written go through to the drive. A store whose
    /// directory has vanished since is opened anew: what it wrote since then
    /// is lost, and what it writes from now on is not.
    fn sync(&self) {
        let mut store = self.store();
        let flush = store.flush();
        drop(store);
        let _ = flush.sync();
        if !self.root.join(self.fingerprint.hex()).exists() {
            self.reopen();
        }
    }
}

struct WriteJob {
    kind: Kind,
    hash: ChainHash,
    parent: ChainHash,
    depth: u32,
    parts: Vec<TransitPart>,
}

struct WriteDone {
    kind: Kind,
    hash: ChainHash,
    parts: Vec<TransitPart>,
    /// The bytes written, or `None` when the record could not be.
    bytes: Option<u64>,
    ns: u64,
}

/// How long the writer waits for the next record before it syncs what it
/// wrote, and the longest that a written record waits for a sync while more
/// keep coming. A sync of a segment took 0.1 to 1 s on the drive this was
/// measured on (btrfs, busy), in which the writer writes nothing; a sync
/// after every record or every half second kept it from the drive's rate.
const SYNC_IDLE: Duration = Duration::from_secs(1);
const SYNC_EVERY: Duration = Duration::from_secs(10);

fn writer(shared: Arc<Shared>, jobs: Receiver<WriteJob>, done: Sender<WriteDone>) {
    // When the oldest record that no sync has covered was written.
    let mut unsynced: Option<Instant> = None;
    loop {
        // What was written goes through to the drive when nothing else comes
        // for a moment, and every few seconds while records keep coming.
        let job = match unsynced {
            None => jobs.recv().map_err(|_| RecvTimeoutError::Disconnected),
            Some(since) if since.elapsed() >= SYNC_EVERY => Err(RecvTimeoutError::Timeout),
            Some(_) => jobs.recv_timeout(SYNC_IDLE),
        };
        let job = match job {
            Ok(job) => job,
            Err(RecvTimeoutError::Timeout) => {
                let begun = Instant::now();
                shared.sync();
                unsynced = None;
                shared.sync_ns.fetch_add(begun.elapsed().as_nanos() as u64, Ordering::Relaxed);
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let WriteJob { kind, hash, parent, depth, parts } = job;
        let begun = Instant::now();
        let bytes = shared.write(kind, hash, parent, depth, &parts).ok();
        unsynced = unsynced.or(Some(begun));
        let ns = begun.elapsed().as_nanos() as u64;
        if done.send(WriteDone { kind, hash, parts, bytes, ns }).is_err() {
            break;
        }
    }
    shared.sync();
}

/// The index of a read that is a run's checkpoint and not one of its pages.
const STATE: usize = usize::MAX;

struct ReadJob {
    job: u64,
    index: usize,
    located: Located,
    parts: Vec<TransitPart>,
}

struct ReadDone {
    job: u64,
    index: usize,
    parts: Vec<TransitPart>,
    ok: bool,
    bytes: u64,
    ns: u64,
}

fn reader(jobs: Arc<Mutex<Receiver<ReadJob>>>, done: Sender<ReadDone>) {
    loop {
        let job = jobs.lock().unwrap_or_else(|e| e.into_inner()).recv();
        let Ok(ReadJob { job, index, located, mut parts }) = job else { break };
        let begun = Instant::now();
        let stored = located.parts();
        let mut ok = stored.len() == parts.len();
        let mut bytes = 0u64;
        for (i, (len, tag)) in stored.iter().enumerate().take(parts.len()) {
            let part = &mut parts[i];
            ok = ok
                && *len as usize <= part.mem.len()
                && located.read_part(i, &mut part.mem).is_ok();
            if !ok {
                break;
            }
            (part.len, part.tag) = (*len as usize, *tag);
            bytes += u64::from(*len);
        }
        if ok {
            let read: Vec<&[u8]> = parts.iter().map(|part| &part.mem[..part.len]).collect();
            ok = located.verify(&read);
        }
        let ns = begun.elapsed().as_nanos() as u64;
        if done.send(ReadDone { job, index, parts, ok, bytes, ns }).is_err() {
            break;
        }
    }
}

/// A run of pages on its way back from the drive.
struct Import {
    id: u64,
    /// Who the index says computes the pages.
    owner: Owner,
    /// Each page's name, the name it follows and its number in the chain.
    pages: Vec<(ChainHash, ChainHash, u32)>,
    /// The page at whose end the checkpoint to bring back lies.
    state: Option<usize>,
    state_parts: Option<Vec<TransitPart>>,
    state_asked: bool,
    /// Pages the readers were asked for, reads that have not come back, and
    /// pages that are in the index.
    asked: usize,
    outstanding: usize,
    done: usize,
    ready: BTreeMap<usize, Vec<TransitPart>>,
    /// Holds the run's pages until it is whole, so that making room for its
    /// later pages does not take its earlier ones.
    held: Sequence,
    failed: bool,
}

impl Import {
    fn over(&self) -> bool {
        self.failed || self.done == self.pages.len()
    }
}

pub(crate) struct Ssd {
    shared: Arc<Shared>,
    durable: Durable,
    min_cells: u64,
    /// The model resumes only from a checkpoint.
    recurrent: bool,
    pub(super) free_states: Vec<Vec<TransitPart>>,
    made_states: usize,
    /// What is with the writer or on its way there.
    queued: HashSet<(Kind, ChainHash)>,
    to_writer: Option<Sender<WriteJob>>,
    from_writer: Receiver<WriteDone>,
    writer: Option<JoinHandle<()>>,
    writing: usize,
    to_readers: Option<Sender<ReadJob>>,
    from_readers: Receiver<ReadDone>,
    readers: Vec<JoinHandle<()>>,
    jobs: VecDeque<Import>,
    next_job: u64,
    pub(super) counters: SsdCounters,
}

impl std::fmt::Debug for Ssd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ssd").field("root", &self.shared.root).finish_non_exhaustive()
    }
}

fn spawn<T: Send + 'static>(
    name: &str,
    body: impl FnOnce() -> T + Send + 'static,
) -> Result<JoinHandle<T>, String> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(body)
        .map_err(|e| format!("cannot start the thread {name}: {e}"))
}

impl Ssd {
    /// The fingerprint of the store of a model that is `durable`, with pages
    /// of `cells_per_page` cells.
    pub(super) fn fingerprint(durable: &Durable, cells_per_page: u32) -> Fingerprint {
        let sizes: Vec<u8> = durable
            .page_bytes
            .iter()
            .chain(&durable.state_bytes)
            .flat_map(|bytes| bytes.to_le_bytes())
            .collect();
        Fingerprint::new(&[
            b"ares state 1",
            &durable.fingerprint,
            &cells_per_page.to_le_bytes(),
            &(durable.page_bytes.len() as u32).to_le_bytes(),
            &sizes,
        ])
    }

    /// Opens the store and starts the threads. `Err` says why the level is
    /// not in use; the cache works without it.
    pub(crate) fn open(
        config: &SsdConfig,
        durable: &Durable,
        cells_per_page: u32,
        recurrent: bool,
    ) -> Result<Self, String> {
        let fingerprint = Self::fingerprint(durable, cells_per_page);
        let limits =
            ares_store::Config { max_bytes: config.max_bytes, segment_bytes: config.segment_bytes };
        std::fs::create_dir_all(&config.dir)
            .map_err(|e| format!("{}: {e}", config.dir.display()))?;
        let store = PageStore::open(&config.dir, fingerprint, limits).map_err(|e| e.to_string())?;
        let shared = Arc::new(Shared {
            store: Mutex::new(store),
            root: config.dir.clone(),
            fingerprint,
            config: limits,
            sync_ns: AtomicU64::new(0),
        });
        let (to_writer, writer_jobs) = mpsc::channel();
        let (writer_done, from_writer) = mpsc::channel();
        let writer = {
            let shared = shared.clone();
            spawn("ares-cache-write", move || writer(shared, writer_jobs, writer_done))?
        };
        let (to_readers, reader_jobs) = mpsc::channel();
        let (reader_done, from_readers) = mpsc::channel();
        let reader_jobs = Arc::new(Mutex::new(reader_jobs));
        let mut readers = Vec::new();
        for i in 0..READERS {
            let (jobs, done) = (reader_jobs.clone(), reader_done.clone());
            readers.push(spawn(&format!("ares-cache-read-{i}"), move || reader(jobs, done))?);
        }
        Ok(Self {
            shared,
            durable: durable.clone(),
            min_cells: u64::from(config.min_cells),
            recurrent,
            free_states: Vec::new(),
            made_states: 0,
            queued: HashSet::new(),
            to_writer: Some(to_writer),
            from_writer,
            writer: Some(writer),
            writing: 0,
            to_readers: Some(to_readers),
            from_readers,
            readers,
            jobs: VecDeque::new(),
            next_job: 0,
            counters: SsdCounters { enabled: true, ..SsdCounters::default() },
        })
    }

    /// The settled cells a text must hold before anything of it is written.
    pub(super) fn min_cells(&self) -> u64 {
        self.min_cells
    }

    /// What the level has done and holds.
    pub(crate) fn counters(&self) -> SsdCounters {
        let store = self.shared.store();
        SsdCounters {
            store_pages: store.len() as u64,
            store_states: store.states() as u64,
            store_bytes: store.bytes(),
            write_ns: self.counters.write_ns + self.shared.sync_ns.load(Ordering::Relaxed),
            transit_bytes: self.made_states as u64 * SsdConfig::room(&self.durable.state_bytes),
            pending: self.writing as u64,
            ..self.counters
        }
    }

    /// Whether anything is on its way: the conductor does not sleep for long
    /// while it is.
    fn busy(&self) -> bool {
        self.writing > 0 || !self.jobs.is_empty()
    }

    /// The store has the record, or it is on its way there.
    pub(super) fn has_or_queued(&self, kind: Kind, hash: ChainHash) -> bool {
        self.queued.contains(&(kind, hash)) || self.shared.store().contains(kind, hash)
    }

    /// A record is on its way to the writer.
    pub(super) fn queue(&mut self, kind: Kind, hash: ChainHash) {
        self.queued.insert((kind, hash));
    }

    /// It is not, after all.
    pub(super) fn unqueue(&mut self, kind: Kind, hash: ChainHash) {
        self.queued.remove(&(kind, hash));
    }

    /// The store has the pages `names` from the first on, and a checkpoint
    /// at the end of one of them: a run of them would be brought back.
    pub(super) fn reaches_state(&self, names: &[ChainHash]) -> bool {
        let store = self.shared.store();
        names
            .iter()
            .take_while(|hash| store.contains(Kind::Page, **hash))
            .any(|hash| store.contains(Kind::State, *hash))
    }

    /// Hands a record whose bytes are in `parts` to the writer. Its parts
    /// come back when it is written ([`Cache::collect`]).
    pub(super) fn write(
        &mut self,
        kind: Kind,
        hash: ChainHash,
        parent: ChainHash,
        depth: u32,
        parts: Vec<TransitPart>,
    ) {
        self.queued.insert((kind, hash));
        let job = WriteJob { kind, hash, parent, depth, parts };
        match &self.to_writer {
            Some(writer) => match writer.send(job) {
                Ok(()) => self.writing += 1,
                Err(back) => self.refused(back.0),
            },
            None => self.refused(job),
        }
    }

    /// The writer is gone: a record that was for it is not written.
    fn refused(&mut self, job: WriteJob) {
        self.queued.remove(&(job.kind, job.hash));
        self.counters.write_errors += 1;
        if job.kind == Kind::State {
            self.free_states.push(job.parts);
        }
    }

    /// A buffer for a checkpoint in transit, without waiting.
    fn state_buffer(&mut self) -> Option<Vec<TransitPart>> {
        self.free_states.pop().or_else(|| {
            (self.made_states < STATE_BUFFERS).then(|| {
                self.made_states += 1;
                self.durable
                    .state_bytes
                    .iter()
                    .map(|bytes| TransitPart {
                        mem: HostBytes::Transit(Aligned::new(*bytes as usize)),
                        len: 0,
                        tag: 0,
                    })
                    .collect()
            })
        })
    }

    /// Looks for the pages of a prompt after the `have` the index has, among
    /// the `total` full pages the prompt may take from a cache. Starts
    /// bringing a run of them back and returns its first page, which the
    /// index then says is being computed.
    ///
    /// For a model with recurrent state the run ends at its deepest
    /// checkpoint in the store: the pages after it would be replayed row by
    /// row, which takes as long as computing them, and without a checkpoint
    /// nothing is brought back.
    pub(super) fn want(
        &mut self,
        index: &mut ares_state::PrefixIndex,
        chain: &Chain,
        have: usize,
        total: usize,
    ) -> Option<(ChainHash, Owner)> {
        let names = chain.hashes().get(have..total)?;
        let (run, state) = {
            let store = self.shared.store();
            let run = names
                .iter()
                .take_while(|hash| {
                    store.contains(Kind::Page, **hash) && index.state(**hash) == PageState::Unknown
                })
                .count();
            let state = (0..run).rev().find(|i| store.contains(Kind::State, names[*i]));
            (run, state)
        };
        let run = match (self.recurrent, state) {
            (true, Some(state)) => state + 1,
            (true, None) => 0,
            (false, _) => run,
        };
        if run == 0 {
            return None;
        }
        let owner = Owner(u64::MAX - self.next_job);
        let pages = (0..run).map(|i| (names[i], chain.before(have + i), (have + i) as u32 + 1));
        self.jobs.push_back(Import {
            id: self.next_job,
            owner,
            pages: pages.collect(),
            state: state.filter(|_| self.recurrent),
            state_parts: None,
            state_asked: false,
            asked: 0,
            outstanding: 0,
            done: 0,
            ready: BTreeMap::new(),
            held: Sequence::new(),
            failed: false,
        });
        self.next_job += 1;
        self.counters.imports += 1;
        index.announce(owner, &names[..run]);
        Some((names[0], owner))
    }

    /// Stops the threads, after the writer has written what it was given.
    pub(crate) fn close(&mut self) {
        self.to_writer = None;
        self.to_readers = None;
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}

impl Drop for Ssd {
    fn drop(&mut self) {
        self.close();
    }
}

impl Cache {
    /// Whether anything waits to move or is moving: the conductor comes back
    /// to [`Cache::pump`] soon while it does.
    pub(crate) fn busy(&self) -> bool {
        !self.leaving.is_empty()
            || !self.states_out.is_empty()
            || self.ssd.as_ref().is_some_and(Ssd::busy)
    }

    /// What the level on SSD has done, when there is one.
    pub(crate) fn ssd_counters(&mut self) -> SsdCounters {
        self.collect();
        self.ssd.as_ref().map_or_else(SsdCounters::default, Ssd::counters)
    }

    /// A buffer for a checkpoint on its way out of its place, which is added
    /// to `batch`: when the batch so far holds every one, it is carried out
    /// first, and when the writer holds every one, the loop waits for it
    /// briefly. This is the one place where the step loop waits for the
    /// drive, besides a page that would be lost otherwise.
    pub(super) fn state_room(
        &mut self,
        pool: &mut PagePool,
        exec: &mut dyn Executor,
        batch: &mut Vec<Transfer>,
        notes: &mut Vec<Note>,
    ) -> Option<Vec<TransitPart>> {
        let begun = Instant::now();
        let patience = if self.draining { DRAIN_LIMIT } else { PATIENCE };
        loop {
            if let Some(parts) = self.ssd.as_mut()?.state_buffer() {
                return Some(parts);
            }
            if notes.iter().any(|note| matches!(note, Note::StateOut(_) | Note::StateIn { .. })) {
                self.carry_out(pool, exec, std::mem::take(batch), std::mem::take(notes));
                continue;
            }
            if begun.elapsed() >= patience || !self.wait_for_writer() {
                return None;
            }
        }
    }

    /// A checkpoint that was due on the drive found no buffer: it is not
    /// written.
    pub(super) fn state_not_written(&mut self, out: StateOut) {
        if let Some(ssd) = &mut self.ssd {
            ssd.unqueue(Kind::State, out.hash);
            ssd.counters.states_skipped += 1;
        }
    }

    /// Waits a moment for the writer to finish a record, which gives its
    /// buffer back. `false` when it holds none or does not finish one in time.
    pub(super) fn wait_for_writer(&mut self) -> bool {
        let Some(ssd) = &mut self.ssd else { return false };
        if ssd.writing == 0 {
            return false;
        }
        match ssd.from_writer.recv_timeout(PATIENCE) {
            Ok(done) => {
                self.written(done);
                true
            }
            Err(_) => false,
        }
    }

    /// The writer has finished a record: a page's buffers go back to its
    /// entry in RAM, which the drive has now too, and a checkpoint's buffer
    /// to the others.
    fn written(&mut self, done: WriteDone) {
        let Some(ssd) = &mut self.ssd else { return };
        ssd.writing -= 1;
        ssd.queued.remove(&(done.kind, done.hash));
        ssd.counters.write_ns += done.ns;
        match (done.bytes, done.kind) {
            (None, _) => ssd.counters.write_errors += 1,
            (Some(0), _) => {}
            (Some(bytes), Kind::Page) => {
                ssd.counters.pages_written += 1;
                ssd.counters.bytes_written += bytes;
            }
            (Some(bytes), Kind::State) => {
                ssd.counters.states_written += 1;
                ssd.counters.bytes_written += bytes;
            }
        }
        match done.kind {
            Kind::State => ssd.free_states.push(done.parts),
            Kind::Page => {
                let Some(ram) = &mut self.in_ram else { return };
                let ok = done.bytes.is_some();
                let away = ram.get(done.hash).is_some_and(|entry| entry.parts.is_none());
                if !away {
                    ram.recycle(done.parts);
                    return;
                }
                ram.give(done.hash, done.parts);
                // A page that cannot be written is not tried again.
                ram.update(done.hash, |entry| match ok {
                    true => entry.stored = true,
                    false => entry.worth = false,
                });
            }
        }
    }

    /// Takes what the writer and the readers have finished.
    pub(super) fn collect(&mut self) {
        loop {
            let Some(ssd) = &mut self.ssd else { return };
            let Ok(done) = ssd.from_writer.try_recv() else { break };
            self.written(done);
        }
        loop {
            let Some(ssd) = &mut self.ssd else { return };
            let Ok(done) = ssd.from_readers.try_recv() else { break };
            ssd.counters.read_ns += done.ns;
            let Some(job) = ssd.jobs.iter_mut().find(|job| job.id == done.job) else {
                self.recycle(done.parts, done.index == STATE);
                continue;
            };
            job.outstanding -= 1;
            let state = done.index == STATE;
            if done.ok && !job.failed {
                ssd.counters.bytes_read += done.bytes;
                if state {
                    ssd.counters.states_read += 1;
                    job.state_parts = Some(done.parts);
                } else {
                    ssd.counters.pages_read += 1;
                    job.ready.insert(done.index, done.parts);
                }
                continue;
            }
            // A record that is damaged or gone: the store forgets it, and the
            // run ends before it.
            if !done.ok {
                let (kind, at) = if state {
                    (Kind::State, job.state.unwrap_or(0))
                } else {
                    (Kind::Page, done.index)
                };
                ssd.shared.store().forget(kind, job.pages[at].0);
            }
            job.failed = true;
            self.recycle(done.parts, state);
        }
    }

    /// A buffer that came back from a read nobody needs.
    fn recycle(&mut self, parts: Vec<TransitPart>, state: bool) {
        match (state, &mut self.ssd, &mut self.in_ram) {
            (true, Some(ssd), _) => ssd.free_states.push(parts),
            (false, _, Some(ram)) => ram.recycle(parts),
            _ => {}
        }
    }

    /// Hands the readers what the runs need next, as far as buffers go and
    /// up to [`READ_AHEAD`] pages in all.
    fn ask(&mut self) {
        let Some(readers) = self.ssd.as_ref().and_then(|ssd| ssd.to_readers.clone()) else {
            return;
        };
        let reading = |ssd: &Ssd| -> usize {
            ssd.jobs.iter().map(|job| job.outstanding + job.ready.len()).sum()
        };
        let jobs = self.ssd.as_ref().map_or(0, |ssd| ssd.jobs.len());
        for at in 0..jobs {
            let ssd = self.ssd.as_mut().expect("checked above");
            let job = &ssd.jobs[at];
            if job.failed {
                continue;
            }
            if let (Some(state), false) = (job.state, job.state_asked) {
                let hash = job.pages[state].0;
                let located = ssd.shared.store().locate(Kind::State, hash);
                match (located, ssd.state_buffer()) {
                    (Some(located), Some(parts)) => {
                        let job = &mut ssd.jobs[at];
                        job.state_asked = true;
                        job.outstanding += 1;
                        let _ = readers.send(ReadJob { job: job.id, index: STATE, located, parts });
                    }
                    // The limit took the checkpoint since the run was planned.
                    (None, parts) => {
                        ssd.free_states.extend(parts);
                        ssd.jobs[at].failed = true;
                        continue;
                    }
                    // No buffer now: the pages wait for the checkpoint.
                    (Some(_), None) => {}
                }
            }
            loop {
                let ssd = self.ssd.as_mut().expect("checked above");
                let job = &ssd.jobs[at];
                if job.asked >= job.pages.len() || job.failed {
                    break;
                }
                if reading(ssd) >= READ_AHEAD {
                    return;
                }
                let (index, hash) = (job.asked, job.pages[job.asked].0);
                let Some(parts) = self.ram_buffer() else { return };
                let ssd = self.ssd.as_mut().expect("checked above");
                let located = ssd.shared.store().locate(Kind::Page, hash);
                let Some(located) = located else {
                    // The limit took the page since the run was planned.
                    ssd.jobs[at].failed = true;
                    if let Some(ram) = &mut self.in_ram {
                        ram.recycle(parts);
                    }
                    break;
                };
                let job = &mut ssd.jobs[at];
                job.asked += 1;
                job.outstanding += 1;
                let _ = readers.send(ReadJob { job: job.id, index, located, parts });
            }
        }
    }

    /// Moves what is due between the devices, the host and the drive:
    /// what left the devices into RAM, checkpoints that leave RAM to the
    /// drive, and the pages and the checkpoint of the runs that come back
    /// from the drive, which enter the index here. Then RAM writes what only
    /// it holds when it is too full. Called between steps.
    pub(crate) fn pump(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        self.collect();
        match self.ssd.is_some() {
            true => self.round(pool, exec),
            false => self.settle_leaving(pool, exec),
        }
        self.clean_ahead();
        self.settle_leaving(pool, exec);
    }

    /// One round of [`Cache::pump`] with a level on SSD: what the readers
    /// need next, one batch of transfers, and what it came to.
    fn round(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        self.ask();
        let per_page = u64::from(self.cells_per_page);
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        // What left the devices and what leaves RAM, first: the pages of the
        // pool and the places they leave may be taken by what comes back.
        self.push_leaving(pool, exec, &mut batch, &mut notes);
        let settle = u64::from(self.durable.as_ref().map_or(0, |durable| durable.settle_cells));
        let jobs = self.ssd.as_ref().map_or(0, |ssd| ssd.jobs.len());
        for at in 0..jobs {
            loop {
                let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                let job = &mut ssd.jobs[at];
                let in_batch =
                    |note: &&Note| matches!(note, Note::PageIn { job: id, .. } if *id == job.id);
                let next = job.done + notes.iter().filter(in_batch).count();
                if job.failed || !job.ready.contains_key(&next) {
                    break;
                }
                let with_state = job.state == Some(next);
                if with_state && job.state_parts.is_none() {
                    break;
                }
                // A page of the pool for it; cached pages nobody holds make
                // room, and leave for RAM before it comes in.
                let mut grown = job.held.append(pool, per_page);
                if grown.is_err() {
                    self.evict(pool, 1);
                    self.push_leaving(pool, exec, &mut batch, &mut notes);
                    let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                    grown = ssd.jobs[at].held.append(pool, per_page);
                }
                let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                let job = &mut ssd.jobs[at];
                if grown.is_err() {
                    job.failed = true;
                    break;
                }
                let page = *job.held.pages().last().expect("a page was appended");
                let id = job.id;
                if with_state {
                    let parts = job.state_parts.take().expect("checked above");
                    match self.new_place().0 {
                        Some(place) => {
                            // The checkpoint that had the place leaves it
                            // for the drive before this one comes in.
                            self.push_leaving(pool, exec, &mut batch, &mut notes);
                            batch.push(Transfer {
                                kind: TransferKind::StateIn,
                                object: place,
                                keep_form: false,
                                parts,
                                ok: false,
                            });
                            notes.push(Note::StateIn { job: id });
                        }
                        // No place for it: the run ends before this page.
                        None => {
                            let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                            ssd.free_states.push(parts);
                            ssd.jobs[at].failed = true;
                            break;
                        }
                    }
                }
                let ssd = self.ssd.as_mut().expect("only with a level on SSD");
                let job = &mut ssd.jobs[at];
                let parts = job.ready.remove(&next).expect("checked above");
                let depth = job.pages[next].2;
                let behind = (job.pages.len() - 1 - next) as u64 * per_page;
                let keep_form = depth == 1 || behind <= settle + per_page;
                batch.push(Transfer {
                    kind: TransferKind::PageIn,
                    object: page.0,
                    keep_form,
                    parts,
                    ok: false,
                });
                notes.push(Note::PageIn { job: id, index: next });
            }
        }
        self.carry_out(pool, exec, batch, notes);

        // Runs that are whole, or cut: the index learns that nobody computes
        // the rest, and the pages are the index's alone.
        let mut at = 0;
        loop {
            let Some(ssd) = &mut self.ssd else { return };
            let Some(job) = ssd.jobs.get(at) else { break };
            if !job.over() || job.outstanding > 0 {
                at += 1;
                continue;
            }
            let mut job = ssd.jobs.remove(at).expect("it is there");
            ssd.counters.imports_cut += u64::from(job.done < job.pages.len());
            self.index.withdraw(job.owner);
            std::mem::take(&mut job.held).release(pool);
            for parts in std::mem::take(&mut job.ready).into_values() {
                self.recycle(parts, false);
            }
            if let Some(parts) = job.state_parts.take() {
                self.recycle(parts, true);
            }
        }
    }

    /// A page of a run from the drive came into pool page `object` (`ok`),
    /// with the checkpoint of the run when `state` says one came with it: it
    /// enters the index, and its buffer stays in RAM as a copy.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn page_in(
        &mut self,
        pool: &mut PagePool,
        id: u64,
        index: usize,
        object: u32,
        parts: Vec<TransitPart>,
        ok: bool,
        state: Option<(u64, u32, bool)>,
    ) {
        let Some(ssd) = &mut self.ssd else { return };
        let Some(job) = ssd.jobs.iter_mut().find(|job| job.id == id) else {
            self.recycle(parts, false);
            return;
        };
        let (hash, parent, depth) = job.pages[index];
        let with_state = job.state == Some(index);
        let came = ok && !job.failed;
        if came {
            self.index.insert(pool, parent, hash, PageId(object));
            job.done += 1;
        } else {
            job.failed = true;
        }
        let owner = job.owner;
        if let (true, Some((_, place, state_ok))) = (with_state, state) {
            let id = CheckpointId(u64::from(place));
            let attached =
                (came && state_ok).then(|| self.index.attach_checkpoint(hash, id, owner));
            match attached {
                Some(Ok(before)) => {
                    self.placed.insert(place, hash);
                    if let Some(before) = before {
                        self.release(before);
                    }
                }
                Some(Err(_)) | None => self.free(place),
            }
        }
        match (came, &mut self.in_ram) {
            (true, Some(ram)) if ram.get(hash).is_none() => {
                let mut entry = Entry::new(parent, depth, None);
                entry.on_cards = true;
                entry.stored = true;
                entry.worth = true;
                ram.insert(hash, entry, parts);
            }
            _ => self.recycle(parts, false),
        }
    }

    /// Writes everything worth writing that the drive does not have, when
    /// the engine ends: the checkpoints first (a few large records, which
    /// pass through a few buffers), then the pages, what only RAM holds the
    /// coldest first, then the pages of the devices. Waits for the writer,
    /// at most for [`DRAIN_LIMIT`], and stops the threads.
    pub(crate) fn drain(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        self.settle_leaving(pool, exec);
        let Some(ssd) = &mut self.ssd else { return };
        // Runs that come back are for requests that are gone.
        for job in &mut ssd.jobs {
            job.failed = true;
        }
        self.draining = true;
        let begun = Instant::now();
        let order = self.in_ram.as_ref().map_or_else(Vec::new, |ram| ram.in_order());
        let cards = self.index.cached_pages();
        let left_behind = self.left_behind();
        // Checkpoints: of RAM, and of the devices' pages.
        for hash in &order {
            let due = self.in_ram.as_ref().and_then(|ram| ram.get(*hash)).is_some_and(|entry| {
                entry.parts.is_some() && entry.checkpoint.is_some_and(|h| h.worth && !h.stored)
            });
            if due {
                self.clean(Item::Checkpoint, *hash);
                self.settle_leaving(pool, exec);
            }
        }
        for page in &cards {
            if let Some(id) = page.checkpoint
                && !left_behind.contains(&page.hash)
                && let Some(mut out) =
                    self.state_out(id.0 as u32, page.hash, page.parent, page.depth)
            {
                out.keep = true;
                self.ssd.as_mut().expect("checked above").queue(Kind::State, page.hash);
                self.states_out.push(out);
                self.settle_leaving(pool, exec);
            }
        }
        // Pages: what only RAM holds, from its buffers.
        for hash in &order {
            let due = self.in_ram.as_ref().and_then(|ram| ram.get(*hash)).is_some_and(|entry| {
                entry.worth && !entry.stored && entry.tail.is_none() && entry.parts.is_some()
            });
            if due {
                self.clean(Item::Page, *hash);
            }
        }
        // And the devices' pages, through buffers of RAM, a batch at a time.
        let (mut batch, mut notes) = (Vec::new(), Vec::new());
        for page in &cards {
            let worth = self.worth.contains(&page.hash) || self.deep(page.depth);
            let ssd = self.ssd.as_ref().expect("checked above");
            let in_ram = self.in_ram.as_ref().is_some_and(|ram| ram.get(page.hash).is_some());
            if !worth || in_ram || ssd.has_or_queued(Kind::Page, page.hash) {
                continue;
            }
            let room = self.page_room(pool, exec, &mut batch, &mut notes, true);
            let Some(parts) = room else { break };
            self.ssd.as_mut().expect("checked above").queue(Kind::Page, page.hash);
            batch.push(Transfer {
                kind: TransferKind::PageOut,
                object: page.page.0,
                keep_form: false,
                parts,
                ok: false,
            });
            notes.push(Note::CardsOut { hash: page.hash, parent: page.parent, depth: page.depth });
            if batch.len() as u64 >= RESERVE {
                self.carry_out(pool, exec, std::mem::take(&mut batch), std::mem::take(&mut notes));
            }
            if begun.elapsed() > DRAIN_LIMIT {
                break;
            }
        }
        self.carry_out(pool, exec, batch, notes);
        // The writer finishes what it has.
        while self.ssd.as_ref().is_some_and(|ssd| ssd.writing > 0) && begun.elapsed() < DRAIN_LIMIT
        {
            self.wait_for_writer();
        }
        let Some(ssd) = &mut self.ssd else { return };
        for job in &mut ssd.jobs {
            std::mem::take(&mut job.held).release(pool);
        }
        ssd.jobs.clear();
        ssd.close();
    }
}
