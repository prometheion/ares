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
//! - **A tail** (a partly used page) with the state at its end, a prompt's
//!   anchor at the end of its last message or a request's end, as a page and
//!   its checkpoint are: a page record under the tail's name, one deeper than
//!   the page it hangs under, and the state beside it. A tail without its
//!   state is not written, and neither is a state whose save still waits for
//!   a record when the engine ends.
//!
//! The writer syncs the drive a second after it has nothing left to write,
//! and every ten seconds while records keep coming, so what was written is
//! durable shortly after.
//!
//! # A hit
//!
//! [`Cache::probe`] asks the store for the pages after the ones the index
//! (and RAM) have, and for a tail with its state they go on into when that
//! state lies deepest. A run of them is announced in the index under an owner of
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

use ares_state::{Chain, ChainHash, Owner, PageId, PagePool, Priority, Sequence};
use ares_store::{Aligned, Fingerprint, Kind, Located, PageStore, Part};

use super::ram::{Entry, Item};
use super::{Cache, Note, RESERVE, StateOut, deepest, keeps_form, run_of, run_pages};
use crate::exec::{Executor, HostBytes, Transfer, TransferKind, TransitPart};
use crate::model::Durable;

mod pump;

/// The longest tail under `prior` with the state at its end that `rest`
/// starts with, of `store`, of those `wanted` takes: its name and its cells.
/// `next`, the full page that goes on from `prior` in the text, is no tail.
fn stored_tail(
    store: &PageStore,
    prior: ChainHash,
    next: Option<ChainHash>,
    rest: &[u64],
    wanted: impl Fn(ChainHash) -> bool,
) -> Option<(ChainHash, usize)> {
    let tails: Vec<ChainHash> = store
        .children(prior)
        .filter(|hash| Some(*hash) != next && store.contains(Kind::State, *hash) && wanted(*hash))
        .collect();
    if tails.is_empty() {
        return None;
    }
    prior
        .extensions(rest)
        .enumerate()
        .filter(|(_, hash)| tails.contains(hash))
        .last()
        .map(|(cells, hash)| (hash, cells + 1))
}

/// Checkpoints in transit at once.
const STATE_BUFFERS: usize = 3;

/// The most runs whose import was cut that the level remembers ([`Ssd::cut`]).
const CUT_RUNS: usize = 64;

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
/// How long the level waits for a store that is locked before it serves
/// without it: the engine before this one may hold it a moment longer, and a
/// process forked from this one holds its lock until it runs its program.
const LOCKED_PATIENCE: Duration = Duration::from_secs(1);
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
    /// which are in those books ([`SsdConfig::floor_pages`]).
    pub fn transit_ram(durable: &Durable) -> u64 {
        STATE_BUFFERS as u64 * durable.state_room()
    }

    /// The pages of page-locked RAM the prompt cache keeps whatever the
    /// machine has when it has a level on SSD, for a model that is
    /// `durable`: what state passes through on its way to the drive and
    /// back. Beside them the cache keeps one checkpoint place.
    pub fn floor_pages(durable: &Durable) -> u64 {
        RESERVE * durable.page_room()
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
                Ok(self.store().commit(pending).unwrap_or(0))
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
    /// The tokens of the run's last entry when it is a tail (a partly used
    /// page with the state at its end), which enters the index as one.
    tail: Option<Vec<u64>>,
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
    /// The last pages of runs whose import was cut (no place for the run's
    /// checkpoint, no page of the pool, a read that failed): the next look
    /// for such a run does not ask the drive for it again, and the prompt
    /// goes on with what the cache has ([`Ssd::want`]).
    cut: HashSet<ChainHash>,
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
        let begun = Instant::now();
        let store = loop {
            match PageStore::open(&config.dir, fingerprint, limits) {
                Err(ares_store::Error::Locked { .. }) if begun.elapsed() < LOCKED_PATIENCE => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                other => break other.map_err(|e| e.to_string())?,
            }
        };
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
            cut: HashSet::new(),
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
            transit_bytes: self.made_states as u64 * self.durable.state_room(),
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

    /// The store has a tail with the state at its end under `prior` that
    /// `rest` goes on with (`next`, the full page of the text after `prior`,
    /// is none): a run that ends at `prior` goes on from the drive.
    pub(super) fn has_tail(&self, prior: ChainHash, next: Option<ChainHash>, rest: &[u64]) -> bool {
        stored_tail(&self.shared.store(), prior, next, rest, |_| true).is_some()
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

    /// Looks for the pages of a prompt of `looked` tokens (its last one is
    /// computed) after the `have` the index has, among the `total` full pages
    /// the prompt may take from a cache, with `cells` cells a page. Starts
    /// bringing a run of them back for `owner` and returns its first page,
    /// which the index then says is being computed.
    ///
    /// For a model with recurrent state the run ends at its deepest
    /// checkpoint in the store: the pages after it would be replayed row by
    /// row, which takes as long as computing them, and without a checkpoint
    /// nothing is brought back. The deepest may be the state at the end of a
    /// tail the prompt goes on from (an anchor at the end of a message, a
    /// request's end), deeper than `beyond`, where the cache can resume
    /// already: the run is then the pages before it and the tail.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn want(
        &mut self,
        index: &mut ares_state::PrefixIndex,
        chain: &Chain,
        looked: &[u64],
        cells: u64,
        have: usize,
        total: usize,
        beyond: u64,
        owner: Owner,
    ) -> Option<(ChainHash, Owner)> {
        let names = chain.hashes().get(have..total)?;
        let (run, state, tail) = {
            let store = self.shared.store();
            let run = run_of(index, names, |hash| store.contains(Kind::Page, hash));
            let state = deepest(names, run, |hash| store.contains(Kind::State, hash));
            let floor = state.map_or(0, |state| (have + state + 1) as u64 * cells).max(beyond);
            let tail = (0..=run).rev().filter(|_| self.recurrent).find_map(|j| {
                let at = (have + j) as u64 * cells;
                if at + cells <= floor {
                    return None;
                }
                let rest = looked.get(at as usize..)?;
                let rest = &rest[..rest.len().min(cells as usize - 1)];
                let next = chain.hashes().get(have + j).copied();
                // One the index holds is the cache's already (`beyond`).
                let new = |hash: ChainHash| !index.holds(hash);
                let (hash, n) = stored_tail(&store, chain.before(have + j), next, rest, new)?;
                (at + n as u64 > floor).then_some((j, hash, n))
            });
            (run, state, tail)
        };
        // A tail a run brings back already: the prompt waits for it.
        if let Some((_, hash, _)) = tail
            && let ares_state::PageState::Computing(by) = index.state(hash)
        {
            return Some((hash, by));
        }
        let (run, state) = match (self.recurrent, tail, state) {
            (true, Some((j, ..)), _) => (j, Some(j)),
            (true, None, Some(state)) => (state + 1, Some(state)),
            (true, None, None) => return None,
            (false, ..) => (run, None),
        };
        let mut pages = run_pages(chain, have, run);
        let mut announced = names[..run].to_vec();
        let tail = tail.map(|(_, hash, n)| {
            let at = (have + run) * cells as usize;
            pages.push((hash, chain.before(have + run), (have + run) as u32));
            announced.push(hash);
            looked[at..at + n].to_vec()
        });
        let first = *announced.first()?;
        // A run whose import was cut is not asked for again by the next look
        // for it: a prompt that waited for it starts with what the cache has,
        // and does not wait for the same import again and again (#98).
        if announced.last().is_some_and(|last| self.cut.remove(last)) {
            return None;
        }
        self.jobs.push_back(Import {
            id: self.next_job,
            owner,
            pages,
            state,
            tail,
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
        index.announce(owner, &announced);
        Some((first, owner))
    }

    /// The import of the run that ends at `last` was cut ([`Ssd::want`]).
    /// Runs that nobody looks for again are forgotten as more are cut.
    pub(super) fn cut(&mut self, last: Option<ChainHash>) {
        let Some(last) = last else { return };
        if self.cut.len() >= CUT_RUNS {
            self.cut.clear();
        }
        self.cut.insert(last);
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
