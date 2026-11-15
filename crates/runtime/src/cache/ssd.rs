//! The level of the prefix cache that is on SSD.
//!
//! Pages and checkpoints of recurrent state are written to an
//! [`ares_store::PageStore`] under the names the index gives them, and a
//! prompt whose start is not in the index but in the store has it brought
//! back before it starts. The conductor decides what moves and when; the
//! bytes move in three places, none of which is a step:
//!
//! - between the devices and host memory in [`Executor::transfer`], which the
//!   conductor calls between steps ([`Cache::pump`]);
//! - from host memory to the drive on one writer thread;
//! - from the drive to host memory on a few reader threads.
//!
//! # What is written, and when
//!
//! - **A page** when it is full and its sequence's last step began
//!   [`Durable::settle_cells`] past it, or when the sequence ends (ends,
//!   gives way, is cancelled), in the form the device holds it in then. A
//!   page that the store has is not written again: its name says what it
//!   holds.
//! - **A checkpoint** at a page boundary, after the step whose record carried
//!   its save. The partly used last page of a sequence and the state at its
//!   end are not written: a prompt that resumes from the drive restores the
//!   checkpoint before them and replays the rows between.
//! - **Nothing of a short text.** A sequence writes its pages once it holds
//!   [`SsdConfig::min_cells`] settled cells, from its first page on, and a
//!   checkpoint is written when it lies at least that deep. A text below
//!   that is computed again faster than its bytes are worth: what would be
//!   written is mostly the checkpoint, which is as large as the pages of
//!   thousands of cells.
//!
//! The writer syncs the drive a second after it has nothing left to write,
//! and every ten seconds while records keep coming, so what was written is
//! durable shortly after, whatever happens to the copies on the devices.
//!
//! # A hit
//!
//! [`Cache::probe`] asks the store for the pages after the ones the index
//! has. A run of them is announced in the index under an owner of its own,
//! which holds the prompt back as a page that another sequence computes
//! would. The readers bring the pages and the deepest checkpoint of the run
//! into host memory; the conductor gives each page a page of the pool, the
//! devices take the bytes, and the page enters the index. The checkpoint goes
//! into a checkpoint place and is attached to its page before that page is
//! published. The prompt then starts as from any cached prefix: it restores
//! the checkpoint, replays the rows after it and computes the rest.
//!
//! # Failure
//!
//! The store is a cache. A record that cannot be written is dropped, a
//! record that cannot be read or fails its checksum ends its run (the prompt
//! computes from there), a full drive gives up old segments, and a directory
//! that is gone is made again. None of it reaches a request.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ares_state::{Chain, ChainHash, CheckpointId, Owner, PageId, PagePool, PageState, Sequence};
use ares_store::{Aligned, Fingerprint, Kind, Located, PageStore, Part};

use super::Cache;
use crate::exec::{Executor, Transfer, TransferKind, TransitPart};
use crate::model::Durable;

/// Pages in transit at once, to the drive and from it together.
const PAGE_BUFFERS: usize = 16;
/// Checkpoints in transit at once.
const STATE_BUFFERS: usize = 3;
/// How long the step loop waits for a buffer for a checkpoint that must
/// leave its place now, before it gives that checkpoint up.
const STATE_PATIENCE: Duration = Duration::from_millis(250);
/// Threads that read from the drive.
const READERS: usize = 4;

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
    /// Checkpoints that were not written because none could be in transit.
    pub states_skipped: u64,
    /// What the store holds now.
    pub store_pages: u64,
    pub store_states: u64,
    pub store_bytes: u64,
    /// Host memory the buffers in transit take.
    pub transit_bytes: u64,
    /// Pages and checkpoints that wait to be written or are being written.
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

/// A full page on its way to the drive.
struct PageOut {
    page: PageId,
    hash: ChainHash,
    parent: ChainHash,
    depth: u32,
}

/// A checkpoint on its way to the drive.
struct StateOut {
    place: u32,
    hash: ChainHash,
    parent: ChainHash,
    depth: u32,
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

/// What a transfer of a batch was for.
enum Note {
    PageIn { job: u64, index: usize },
    StateIn { job: u64 },
    PageOut(PageOut),
    StateOut(StateOut),
}

pub(crate) struct Ssd {
    shared: Arc<Shared>,
    durable: Durable,
    cells_per_page: u32,
    min_cells: u64,
    /// The model resumes only from a checkpoint.
    recurrent: bool,
    free_pages: Vec<Vec<TransitPart>>,
    made_pages: usize,
    free_states: Vec<Vec<TransitPart>>,
    made_states: usize,
    /// Per running sequence, the pages of it that were looked at, and the
    /// cells it held when it was last noted: where its last step began.
    cursors: HashMap<Owner, (usize, u64)>,
    page_out: VecDeque<PageOut>,
    state_out: VecDeque<StateOut>,
    /// What is queued here or with the writer.
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
    counters: SsdCounters,
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
            cells_per_page,
            min_cells: u64::from(config.min_cells),
            recurrent,
            free_pages: Vec::new(),
            made_pages: 0,
            free_states: Vec::new(),
            made_states: 0,
            cursors: HashMap::new(),
            page_out: VecDeque::new(),
            state_out: VecDeque::new(),
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

    /// What the level has done and holds.
    pub(crate) fn counters(&self) -> SsdCounters {
        let store = self.shared.store();
        let room = |bytes: &[u64]| bytes.iter().map(|b| ares_store::padded(*b as usize)).sum();
        let (page, state): (usize, usize) =
            (room(&self.durable.page_bytes), room(&self.durable.state_bytes));
        SsdCounters {
            store_pages: store.len() as u64,
            store_states: store.states() as u64,
            store_bytes: store.bytes(),
            write_ns: self.counters.write_ns + self.shared.sync_ns.load(Ordering::Relaxed),
            transit_bytes: (self.made_pages * page + self.made_states * state) as u64,
            pending: (self.page_out.len() + self.state_out.len() + self.writing) as u64,
            ..self.counters
        }
    }

    /// Whether anything waits to move or is moving: the conductor does not
    /// sleep for long while it does.
    pub(crate) fn busy(&self) -> bool {
        !self.page_out.is_empty() || !self.state_out.is_empty() || !self.jobs.is_empty()
    }

    /// Whether the writer still holds records.
    fn flushing(&self) -> bool {
        self.writing > 0
    }

    fn buffer(bytes: &[u64]) -> Vec<TransitPart> {
        bytes
            .iter()
            .map(|bytes| TransitPart { mem: Aligned::new(*bytes as usize), len: 0, tag: 0 })
            .collect()
    }

    fn page_buffer(&mut self) -> Option<Vec<TransitPart>> {
        self.free_pages.pop().or_else(|| {
            (self.made_pages < PAGE_BUFFERS).then(|| {
                self.made_pages += 1;
                Self::buffer(&self.durable.page_bytes)
            })
        })
    }

    fn state_buffer(&mut self) -> Option<Vec<TransitPart>> {
        self.free_states.pop().or_else(|| {
            (self.made_states < STATE_BUFFERS).then(|| {
                self.made_states += 1;
                Self::buffer(&self.durable.state_bytes)
            })
        })
    }

    /// Queues the pages of the sequence of `owner` that are due: it holds
    /// `settled` cells, whose full pages are in `index` under the names of
    /// `chain`, and `last` says that it ends here.
    fn note(
        &mut self,
        index: &ares_state::PrefixIndex,
        owner: Owner,
        chain: &Chain,
        settled: u64,
        last: bool,
    ) {
        let per_page = u64::from(self.cells_per_page);
        if settled < self.min_cells {
            if last {
                self.cursors.remove(&owner);
            }
            return;
        }
        let full = (settled / per_page) as usize;
        let (from, began) = self.cursors.get(&owner).copied().unwrap_or((0, 0));
        // A device brings a page into the form it keeps at the start of a
        // step, by where the sequence stands then: what the step that has
        // just run left behind its start is final now.
        let due = match last {
            true => full,
            false => {
                (began.saturating_sub(u64::from(self.durable.settle_cells)) / per_page) as usize
            }
        };
        let due = due.min(full).min(chain.hashes().len());
        if from < due {
            let store = self.shared.store();
            for i in from..due {
                let hash = chain.hashes()[i];
                if self.queued.contains(&(Kind::Page, hash)) || store.contains(Kind::Page, hash) {
                    continue;
                }
                let Some((page, ..)) = index.page(hash) else { continue };
                self.queued.insert((Kind::Page, hash));
                self.page_out.push_back(PageOut {
                    page,
                    hash,
                    parent: chain.before(i),
                    depth: i as u32 + 1,
                });
            }
        }
        if last {
            self.cursors.remove(&owner);
        } else {
            self.cursors.insert(owner, (from.max(due), settled));
        }
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

    /// Waits a moment for the writer to finish a record, which gives its
    /// buffer back. `false` when it holds none or does not finish one in time.
    fn wait_for_writer(&mut self) -> bool {
        if self.writing == 0 {
            return false;
        }
        match self.from_writer.recv_timeout(STATE_PATIENCE) {
            Ok(done) => {
                self.written(done);
                true
            }
            Err(_) => false,
        }
    }

    fn written(&mut self, done: WriteDone) {
        {
            self.writing -= 1;
            self.queued.remove(&(done.kind, done.hash));
            self.counters.write_ns += done.ns;
            match (done.bytes, done.kind) {
                (None, _) => self.counters.write_errors += 1,
                (Some(0), _) => {}
                (Some(bytes), Kind::Page) => {
                    self.counters.pages_written += 1;
                    self.counters.bytes_written += bytes;
                }
                (Some(bytes), Kind::State) => {
                    self.counters.states_written += 1;
                    self.counters.bytes_written += bytes;
                }
            }
            match done.kind {
                Kind::Page => self.free_pages.push(done.parts),
                Kind::State => self.free_states.push(done.parts),
            }
        }
    }

    /// Takes what the writer and the readers have finished.
    fn collect(&mut self) {
        while let Ok(done) = self.from_writer.try_recv() {
            self.written(done);
        }
        while let Ok(done) = self.from_readers.try_recv() {
            self.counters.read_ns += done.ns;
            let Some(job) = self.jobs.iter_mut().find(|job| job.id == done.job) else {
                continue;
            };
            job.outstanding -= 1;
            let state = done.index == STATE;
            if done.ok && !job.failed {
                self.counters.bytes_read += done.bytes;
                if state {
                    self.counters.states_read += 1;
                    job.state_parts = Some(done.parts);
                } else {
                    self.counters.pages_read += 1;
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
                self.shared.store().forget(kind, job.pages[at].0);
            }
            job.failed = true;
            match state {
                true => self.free_states.push(done.parts),
                false => self.free_pages.push(done.parts),
            }
        }
    }

    /// Hands the readers what the runs need next, as far as buffers go.
    fn ask(&mut self) {
        let Some(readers) = self.to_readers.clone() else { return };
        for at in 0..self.jobs.len() {
            if self.jobs[at].failed {
                continue;
            }
            if let (Some(state), false) = (self.jobs[at].state, self.jobs[at].state_asked) {
                let hash = self.jobs[at].pages[state].0;
                let located = self.shared.store().locate(Kind::State, hash);
                match (located, self.state_buffer()) {
                    (Some(located), Some(parts)) => {
                        let job = &mut self.jobs[at];
                        job.state_asked = true;
                        job.outstanding += 1;
                        let _ = readers.send(ReadJob { job: job.id, index: STATE, located, parts });
                    }
                    // The limit took the checkpoint since the run was planned.
                    (None, parts) => {
                        self.free_states.extend(parts);
                        self.jobs[at].failed = true;
                        continue;
                    }
                    // No buffer now: the pages wait for the checkpoint.
                    (Some(_), None) => {}
                }
            }
            while self.jobs[at].asked < self.jobs[at].pages.len() {
                let index = self.jobs[at].asked;
                let Some(parts) = self.page_buffer() else { return };
                let located = self.shared.store().locate(Kind::Page, self.jobs[at].pages[index].0);
                let Some(located) = located else {
                    // The limit took the page since the run was planned.
                    self.free_pages.push(parts);
                    self.jobs[at].failed = true;
                    break;
                };
                let job = &mut self.jobs[at];
                job.asked += 1;
                job.outstanding += 1;
                let _ = readers.send(ReadJob { job: job.id, index, located, parts });
            }
        }
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
    /// Tells the level on SSD that the sequence of `owner` holds `settled`
    /// cells whose full pages are published under the names of `chain`, and
    /// with `last` that it ends here: the pages that are due are queued.
    pub(crate) fn note(&mut self, owner: Owner, chain: &Chain, settled: u64, last: bool) {
        if let Some(ssd) = &mut self.ssd {
            ssd.note(&self.index, owner, chain, settled, last);
        }
    }

    /// Whether the level on SSD has something to move, so that the conductor
    /// comes back to [`Cache::pump`] soon.
    pub(crate) fn ssd_busy(&self) -> bool {
        self.ssd.as_ref().is_some_and(Ssd::busy)
    }

    /// What the level on SSD has done, when there is one.
    pub(crate) fn ssd_counters(&mut self) -> SsdCounters {
        match &mut self.ssd {
            Some(ssd) => {
                ssd.collect();
                ssd.counters()
            }
            None => SsdCounters::default(),
        }
    }

    /// Moves what is due between the devices and the host: pages and
    /// checkpoints that go to the drive, and the pages and the checkpoint of
    /// the runs that come back from it, which enter the index here. Called
    /// between steps.
    pub(crate) fn pump(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        let Some(mut ssd) = self.ssd.take() else { return };
        self.pump_with(&mut ssd, pool, exec);
        self.ssd = Some(ssd);
    }

    /// Pumps until everything queued has left for the drive and the writer
    /// has written it, when the engine stops: nothing new is queued then.
    pub(crate) fn drain(&mut self, pool: &mut PagePool, exec: &mut dyn Executor) {
        let Some(mut ssd) = self.ssd.take() else { return };
        // Runs that come back are for requests that are gone.
        for job in &mut ssd.jobs {
            job.failed = true;
        }
        let until = Instant::now() + std::time::Duration::from_secs(30);
        loop {
            self.pump_with(&mut ssd, pool, exec);
            let moving =
                ssd.busy() || ssd.flushing() || ssd.jobs.iter().any(|job| job.outstanding > 0);
            if !moving || Instant::now() > until {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        for job in &mut ssd.jobs {
            std::mem::take(&mut job.held).release(pool);
        }
        ssd.jobs.clear();
        ssd.close();
        self.ssd = Some(ssd);
    }

    fn pump_with(&mut self, ssd: &mut Ssd, pool: &mut PagePool, exec: &mut dyn Executor) {
        ssd.collect();
        let per_page = u64::from(ssd.cells_per_page);

        // The saves the last record carried have run: the checkpoints at
        // page boundaries that lie deep enough go to the drive now, while
        // their places hold them. A place holds what the last save of the
        // record put there (a checkpoint that was dropped before its save ran
        // gave its place to a later one).
        let flying = std::mem::take(&mut self.flying);
        for (i, (place, hash)) in flying.iter().copied().enumerate() {
            if flying[i + 1..].iter().any(|(later, _)| *later == place) {
                continue;
            }
            let Some((_, parent, depth)) = self.index.page(hash) else { continue };
            let key = (Kind::State, hash);
            if u64::from(depth) * per_page < ssd.min_cells
                || ssd.queued.contains(&key)
                || ssd.shared.store().contains(Kind::State, hash)
            {
                continue;
            }
            ssd.queued.insert(key);
            ssd.state_out.push_back(StateOut { place, hash, parent, depth });
        }

        // The step loop does not wait for the drive: pages that it takes
        // slower than they are computed stay queued, in the pool, and a page
        // the pool gives up before its turn is not written.
        self.round(ssd, pool, exec);
    }

    /// One round of [`Cache::pump`]: what the readers need next, one batch of
    /// transfers, and what it came to.
    fn round(&mut self, ssd: &mut Ssd, pool: &mut PagePool, exec: &mut dyn Executor) {
        let per_page = u64::from(ssd.cells_per_page);
        ssd.ask();

        let mut batch: Vec<Transfer> = Vec::new();
        let mut notes: Vec<Note> = Vec::new();
        let transfer =
            |kind, object, keep_form, parts| Transfer { kind, object, keep_form, parts, ok: false };

        // Checkpoints that go: all that landed, since their places may be
        // taken again by the next record, and first in the batch, since a
        // checkpoint that comes back in the same batch may take one of their
        // places.
        while let Some(out) = ssd.state_out.pop_front() {
            // A checkpoint cannot wait for a later pump: its place may be
            // taken by then. With all buffers at the writer the loop waits
            // for one, briefly: this is the one place where the step loop
            // waits for the drive, and it takes three checkpoints that the
            // writer has not written yet.
            let begun = Instant::now();
            let mut buffer = ssd.state_buffer();
            while buffer.is_none() && begun.elapsed() < STATE_PATIENCE && ssd.wait_for_writer() {
                buffer = ssd.state_buffer();
            }
            match buffer {
                Some(parts) => {
                    batch.push(transfer(TransferKind::StateOut, out.place, false, parts));
                    notes.push(Note::StateOut(out));
                }
                None => {
                    ssd.queued.remove(&(Kind::State, out.hash));
                    ssd.counters.states_skipped += 1;
                }
            }
        }
        // Runs that come back: their pages in order, and a run's checkpoint
        // with the page it belongs to.
        let settle = u64::from(ssd.durable.settle_cells);
        for at in 0..ssd.jobs.len() {
            loop {
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
                // room.
                let mut grown = job.held.append(pool, per_page);
                if grown.is_err() {
                    self.evict(pool, 1);
                    grown = ssd.jobs[at].held.append(pool, per_page);
                }
                let job = &mut ssd.jobs[at];
                if grown.is_err() {
                    job.failed = true;
                    break;
                }
                let page = *job.held.pages().last().expect("a page was appended");
                if with_state {
                    let parts = job.state_parts.take().expect("checked above");
                    match self.new_place().0 {
                        Some(place) => {
                            // A save that waits for the next record and names
                            // this place is of a checkpoint that was dropped
                            // before it ran: it must not run over what the
                            // place holds from now on.
                            self.saves.retain(|copy| copy.place != place);
                            self.saved.retain(|(of, _)| *of != place);
                            batch.push(transfer(TransferKind::StateIn, place, false, parts));
                            notes.push(Note::StateIn { job: job.id });
                        }
                        // No place for it: the run ends before this page.
                        None => {
                            job.failed = true;
                            ssd.free_states.push(parts);
                            break;
                        }
                    }
                }
                let job = &mut ssd.jobs[at];
                let parts = job.ready.remove(&next).expect("checked above");
                let depth = job.pages[next].2;
                let behind = (job.pages.len() - 1 - next) as u64 * per_page;
                let keep_form = depth == 1 || behind <= settle + per_page;
                batch.push(transfer(TransferKind::PageIn, page.0, keep_form, parts));
                notes.push(Note::PageIn { job: job.id, index: next });
            }
        }

        // Pages that go, as far as buffers go. A page the index has given up
        // since is not written: its page of the pool may hold something else.
        while let Some(out) = ssd.page_out.front() {
            if self.index.page(out.hash).map(|(page, ..)| page) != Some(out.page) {
                ssd.queued.remove(&(Kind::Page, out.hash));
                ssd.page_out.pop_front();
                continue;
            }
            let Some(parts) = ssd.page_buffer() else { break };
            let out = ssd.page_out.pop_front().expect("checked above");
            batch.push(transfer(TransferKind::PageOut, out.page.0, false, parts));
            notes.push(Note::PageOut(out));
        }

        if !batch.is_empty() {
            let begun = Instant::now();
            if exec.transfer(&mut batch).is_err() {
                for transfer in &mut batch {
                    transfer.ok = false;
                }
            }
            ssd.counters.transfers += 1;
            ssd.counters.transfer_ns += begun.elapsed().as_nanos() as u64;
        }

        // What the transfers came to, in their order.
        let mut state_in: Option<(u64, u32, bool)> = None;
        for (transfer, note) in batch.into_iter().zip(notes) {
            let Transfer { object, parts, ok, .. } = transfer;
            match note {
                Note::StateIn { job } => {
                    state_in = Some((job, object, ok));
                    ssd.free_states.push(parts);
                }
                Note::PageIn { job: id, index } => {
                    ssd.free_pages.push(parts);
                    let state = state_in.take_if(|(job, ..)| *job == id);
                    let Some(job) = ssd.jobs.iter_mut().find(|job| job.id == id) else { continue };
                    let (hash, parent, _) = job.pages[index];
                    let with_state = job.state == Some(index);
                    if ok && !job.failed {
                        self.index.insert(pool, parent, hash, PageId(object));
                        job.done += 1;
                    } else {
                        job.failed = true;
                    }
                    if let (true, Some((_, place, state_ok))) = (with_state, state) {
                        let id = CheckpointId(u64::from(place));
                        let attached = (ok && state_ok && !job.failed)
                            .then(|| self.index.attach_checkpoint(hash, id, job.owner));
                        match attached {
                            Some(Ok(before)) => {
                                if let Some(before) = before {
                                    self.release(before);
                                }
                            }
                            Some(Err(_)) | None => self.free_places.push(place),
                        }
                    }
                }
                Note::PageOut(out) => self.hand_over(ssd, Kind::Page, &out, parts, ok),
                Note::StateOut(out) => {
                    let page = PageOut {
                        page: PageId(0),
                        hash: out.hash,
                        parent: out.parent,
                        depth: out.depth,
                    };
                    self.hand_over(ssd, Kind::State, &page, parts, ok);
                }
            }
        }

        // Runs that are whole, or cut: the index learns that nobody computes
        // the rest, and the pages are the index's alone.
        let mut at = 0;
        while at < ssd.jobs.len() {
            let job = &mut ssd.jobs[at];
            if !job.over() || job.outstanding > 0 {
                at += 1;
                continue;
            }
            let mut job = ssd.jobs.remove(at).expect("it is there");
            ssd.counters.imports_cut += u64::from(job.done < job.pages.len());
            self.index.withdraw(job.owner);
            std::mem::take(&mut job.held).release(pool);
            ssd.free_pages.extend(std::mem::take(&mut job.ready).into_values());
            ssd.free_states.extend(job.state_parts.take());
        }
    }

    fn hand_over(
        &mut self,
        ssd: &mut Ssd,
        kind: Kind,
        out: &PageOut,
        parts: Vec<TransitPart>,
        ok: bool,
    ) {
        let job = WriteJob { kind, hash: out.hash, parent: out.parent, depth: out.depth, parts };
        let sent = match (&ssd.to_writer, ok) {
            (Some(writer), true) => writer.send(job).map_err(|e| e.0),
            _ => Err(job),
        };
        match sent {
            Ok(()) => ssd.writing += 1,
            Err(job) => {
                ssd.queued.remove(&(kind, out.hash));
                match kind {
                    Kind::Page => ssd.free_pages.push(job.parts),
                    Kind::State => ssd.free_states.push(job.parts),
                }
            }
        }
    }
}
