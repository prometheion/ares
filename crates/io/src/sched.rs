//! The queues: which chunk of which read goes to the device next.
//!
//! Everything here is plain data behind the reader's mutex; the I/O thread
//! and the callers meet at this one structure and nowhere else. It does no
//! I/O, so the order rules (demand before prefetch, FIFO within a class,
//! promotion, cancellation) are tested without a ring.
//!
//! A read is a *span* (the wanted range widened to blocks) cut into chunks as
//! it is started: `next` is how much of the span has been handed to the ring.
//! A read sits in its class's queue exactly while `next < total`, so a queue
//! entry costs one index and cutting a chunk costs no allocation.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::buffer::Buffer;
use crate::error::Error;
use crate::file::FileInfo;
use crate::request::{Cancel, Class, Completion, Key, Promote, Read, Rejected};
use crate::stats::{ClassStats, DriveStats, LatencyRing, Stats, summarize};

/// A hasher for integer keys: the keys are small counters or packed ids, for
/// which SipHash is pure overhead and a multiplicative mix is plenty.
#[derive(Default)]
pub(crate) struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.write_u64(u64::from(*byte) ^ self.0);
        }
    }

    fn write_u64(&mut self, n: u64) {
        let x = (n ^ self.0).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = x ^ (x >> 32);
    }
}

pub(crate) type IdMap<V> = HashMap<u64, V, BuildHasherDefault<IdHasher>>;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    /// Chunks at one drive at once, and how many of them may be prefetches.
    pub depth: usize,
    pub prefetch_depth: usize,
    pub prefetch_chunk: usize,
    pub demand_chunk: usize,
}

/// One copy of a file as the queues see it: the descriptor to read (its index
/// among all copies of all files, which is how the ring knows it) and the
/// drive it is on.
#[derive(Clone, Copy, Debug)]
struct CopyRef {
    index: u16,
    fd: u16,
    drive: u16,
}

/// How long a measurement of a drive counts. A drive that has not been
/// measured for longer is tried again as if it never had been, so one slow
/// moment does not keep it out for good and a change of the drive is noticed.
const MEASUREMENT_LIFE: Duration = Duration::from_secs(1);

/// A time per MiB, in nanoseconds, averaged over the chunks that were
/// measured (each new one takes an eighth of the weight), and when the last
/// one was.
#[derive(Clone, Copy, Default)]
struct Pace {
    ns_per_mib: u64,
    at: Option<Instant>,
}

impl Pace {
    /// The pace, if it was measured and the measurement still counts.
    fn get(&self, now: Instant) -> Option<u64> {
        let at = self.at?;
        (now.saturating_duration_since(at) <= MEASUREMENT_LIFE).then_some(self.ns_per_mib)
    }

    fn record(&mut self, ns_per_mib: u64, now: Instant) {
        self.ns_per_mib = match self.get(now) {
            Some(old) => (old / 8 * 7).saturating_add(ns_per_mib / 8),
            None => ns_per_mib,
        };
        self.at = Some(now);
    }
}

/// A drive: what is at it, and how it has answered lately.
///
/// A drive is described by two numbers, which is what a flash drive's reads
/// look like: a chunk that is alone at the drive takes `alone` per MiB, and
/// when several are there each takes their number times `shared` per MiB (the
/// drive is then at its throughput). The time a chunk takes at a load of `n`
/// is the larger of the two.
#[derive(Default)]
struct Drive {
    in_flight: usize,
    in_flight_prefetch: usize,
    peak_in_flight: usize,
    chunks: u64,
    bytes: u64,
    /// Measured by chunks that had the drive to themselves.
    alone: Pace,
    /// Measured by chunks that shared the drive: their time per MiB over the
    /// number of chunks that were there.
    shared: Pace,
}

impl Drive {
    fn has_room(&self, class: Class, limits: &Limits) -> bool {
        self.in_flight < limits.depth
            && (class == Class::Demand || self.in_flight_prefetch < limits.prefetch_depth)
    }

    /// How long a chunk started now is expected to take per MiB. A number that
    /// has not been measured (or not lately) counts as zero, so the drive is
    /// tried under that load and measured by it.
    fn expected(&self, now: Instant) -> u128 {
        let load = self.in_flight as u128 + 1;
        let alone = u128::from(self.alone.get(now).unwrap_or(0));
        let shared = u128::from(self.shared.get(now).unwrap_or(0));
        alone.max(load * shared)
    }

    /// A chunk goes to the drive; returns the load it meets, itself included.
    fn started(&mut self, class: Class) -> u32 {
        self.in_flight += 1;
        self.peak_in_flight = self.peak_in_flight.max(self.in_flight);
        if class == Class::Prefetch {
            self.in_flight_prefetch += 1;
        }
        self.in_flight as u32
    }

    fn came_back(&mut self, chunk: &Picked, delivered: usize, now: Instant) {
        // The load the chunk saw: the mean of what was at the drive when it
        // started and now, itself included, counted in halves.
        let load_x2 = u128::from(chunk.load) + self.in_flight as u128;
        self.in_flight -= 1;
        if chunk.class == Class::Prefetch {
            self.in_flight_prefetch -= 1;
        }
        self.chunks += 1;
        self.bytes += delivered as u64;
        if delivered == 0 {
            return;
        }
        let took = now.saturating_duration_since(chunk.started).as_nanos();
        let ns_per_mib = took * (1 << 20) / delivered as u128;
        if load_x2 <= 2 {
            self.alone.record(u64::try_from(ns_per_mib).unwrap_or(u64::MAX), now);
        } else {
            self.shared.record(u64::try_from(ns_per_mib * 2 / load_x2).unwrap_or(u64::MAX), now);
        }
    }
}

#[derive(Clone, Copy)]
enum KeyState {
    /// In the slab at this index.
    Active(u32),
    /// Finished; the completion waits in `done`.
    Done,
}

struct Req {
    key: Key,
    file: u16,
    /// The class the read is in now.
    class: Class,
    submitted_as: Class,
    promoted: bool,
    cancelled: bool,
    /// File offset of the first byte of the buffer.
    start: u64,
    skip: usize,
    wanted: usize,
    /// Bytes of the span to read, and how many of them were handed to the ring.
    total: usize,
    next: usize,
    chunk: usize,
    /// Chunks at the device.
    in_flight: u32,
    /// Where the file ended inside the span, if a chunk came back short.
    eof_at: usize,
    error: Option<Error>,
    device_bytes: u64,
    submitted_at: Instant,
    buffer: Option<Buffer>,
}

/// A chunk to hand to the ring.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Picked {
    pub req: u32,
    /// Offset of the chunk within the span.
    pub rel: usize,
    pub len: usize,
    /// The queue it came from, which decides which depth limit it counts
    /// against until it comes back.
    pub class: Class,
    /// Which copy of the file is read: its index in the file, the descriptor
    /// (index among all copies of all files) and the drive.
    pub copy: u16,
    pub fd: u16,
    pub drive: u16,
    /// When it went to the drive, and how many chunks were there then, itself
    /// included: what the drive is measured by when it comes back.
    pub started: Instant,
    pub load: u32,
    pub offset: u64,
    pub ptr: *mut u8,
}

/// How a chunk ended.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ChunkResult {
    Bytes(usize),
    Errno(i32),
}

pub(crate) struct Sched {
    limits: Limits,
    files: Arc<[FileInfo]>,
    /// The copies of each file.
    copies: Vec<Vec<CopyRef>>,
    drives: Vec<Drive>,
    slab: Vec<Option<Req>>,
    free: Vec<u32>,
    keys: IdMap<KeyState>,
    demand: VecDeque<u32>,
    prefetch: VecDeque<u32>,
    pub(crate) done: VecDeque<Completion>,
    in_flight: usize,
    peak_in_flight: usize,
    pub(crate) held: u32,
    pub(crate) shutdown: bool,
    failed: Option<String>,
    /// Threads blocked in `wait`, so that completions only wake when needed.
    pub(crate) waiters: usize,
    /// Set when a completion was added; the lock guard turns it into a wake.
    pub(crate) news: bool,
    order: u64,
    demand_stats: ClassStats,
    prefetch_stats: ClassStats,
    cancelled: u64,
    promoted: u64,
    demand_latency: LatencyRing,
    prefetch_latency: LatencyRing,
    trace: Option<Vec<(Key, usize)>>,
}

impl Sched {
    /// `files` name the drive of each copy; the descriptors of the copies are
    /// numbered in the order of the files and, within a file, of its copies.
    pub(crate) fn new(limits: Limits, files: Arc<[FileInfo]>, trace: bool) -> Self {
        let mut fd = 0u16;
        let mut n_drives = 0;
        let copies = files
            .iter()
            .map(|file| {
                let refs = file.copies.iter().enumerate().map(|(index, copy)| {
                    let copy = CopyRef { index: index as u16, fd, drive: copy.drive.0 };
                    fd += 1;
                    n_drives = n_drives.max(usize::from(copy.drive) + 1);
                    copy
                });
                refs.collect()
            })
            .collect();
        Self {
            limits,
            files,
            copies,
            drives: (0..n_drives).map(|_| Drive::default()).collect(),
            slab: Vec::new(),
            free: Vec::new(),
            keys: IdMap::default(),
            demand: VecDeque::new(),
            prefetch: VecDeque::new(),
            done: VecDeque::new(),
            in_flight: 0,
            peak_in_flight: 0,
            held: 0,
            shutdown: false,
            failed: None,
            waiters: 0,
            news: false,
            order: 0,
            demand_stats: ClassStats::default(),
            prefetch_stats: ClassStats::default(),
            cancelled: 0,
            promoted: 0,
            demand_latency: LatencyRing::new(),
            prefetch_latency: LatencyRing::new(),
            trace: trace.then(Vec::new),
        }
    }

    fn stats_mut(&mut self, class: Class) -> &mut ClassStats {
        match class {
            Class::Demand => &mut self.demand_stats,
            Class::Prefetch => &mut self.prefetch_stats,
        }
    }

    fn queue_mut(&mut self, class: Class) -> &mut VecDeque<u32> {
        match class {
            Class::Demand => &mut self.demand,
            Class::Prefetch => &mut self.prefetch,
        }
    }

    pub(crate) fn submit(&mut self, read: Read, now: Instant) -> Result<(), Box<Rejected>> {
        let reject = |error, read| Err(Box::new(Rejected { error, read }));
        if let Some(why) = &self.failed {
            return reject(Error::Stopped(why.clone()), read);
        }
        let span = match crate::file::span_in(&self.files, read.file, read.offset, read.len) {
            Ok(span) => span,
            Err(error) => return reject(error, read),
        };
        let block = self.files[usize::from(read.file.0)].block;
        if read.buffer.len() < span.len {
            let error = Error::BufferTooSmall {
                key: read.key,
                have: read.buffer.len(),
                need: span.len,
                block,
            };
            return reject(error, read);
        }
        if block > 1 && !read.buffer.is_aligned_to(block) {
            let error = Error::BufferMisaligned {
                key: read.key,
                address: read.buffer.as_ptr() as usize,
                align: block,
            };
            return reject(error, read);
        }
        if self.keys.contains_key(&read.key) {
            return reject(Error::DuplicateKey(read.key), read);
        }

        let base = match read.class {
            Class::Demand => self.limits.demand_chunk,
            Class::Prefetch => self.limits.prefetch_chunk,
        };
        let block = block as usize;
        let req = Req {
            key: read.key,
            file: read.file.0,
            class: read.class,
            submitted_as: read.class,
            promoted: false,
            cancelled: false,
            start: span.start,
            skip: span.skip,
            wanted: span.wanted,
            total: span.len,
            next: 0,
            chunk: base.div_ceil(block) * block,
            in_flight: 0,
            eof_at: usize::MAX,
            error: None,
            device_bytes: 0,
            submitted_at: now,
            buffer: Some(read.buffer),
        };
        self.stats_mut(read.class).requests += 1;
        let idx = match self.free.pop() {
            Some(idx) => {
                self.slab[idx as usize] = Some(req);
                idx
            }
            None => {
                self.slab.push(Some(req));
                (self.slab.len() - 1) as u32
            }
        };
        self.keys.insert(read.key, KeyState::Active(idx));
        if span.len == 0 {
            self.finish(idx, now);
        } else {
            self.queue_mut(read.class).push_back(idx);
        }
        Ok(())
    }

    fn active(&self, key: Key) -> Result<Option<u32>, Error> {
        match self.keys.get(&key) {
            None => Err(Error::UnknownKey(key)),
            Some(KeyState::Done) => Ok(None),
            Some(KeyState::Active(idx)) => Ok(Some(*idx)),
        }
    }

    fn req(&mut self, idx: u32) -> &mut Req {
        self.slab[idx as usize].as_mut().expect("an indexed request exists")
    }

    /// Takes the request out of its queue if it is in one.
    fn unqueue(&mut self, idx: u32) {
        let class = self.req(idx).class;
        let queue = self.queue_mut(class);
        if let Some(pos) = queue.iter().position(|i| *i == idx) {
            queue.remove(pos);
        }
    }

    /// No further chunk of this request will be started.
    fn stop_issuing(&mut self, idx: u32) {
        if self.req(idx).next < self.req(idx).total {
            self.unqueue(idx);
            let req = self.req(idx);
            req.next = req.total;
        }
    }

    pub(crate) fn cancel(&mut self, key: Key, now: Instant) -> Result<Cancel, Error> {
        let Some(idx) = self.active(key)? else { return Ok(Cancel::TooLate) };
        if self.req(idx).cancelled {
            return Ok(Cancel::Cancelled);
        }
        self.req(idx).cancelled = true;
        self.cancelled += 1;
        self.stop_issuing(idx);
        if self.req(idx).in_flight == 0 {
            self.finish(idx, now);
        }
        Ok(Cancel::Cancelled)
    }

    pub(crate) fn promote(&mut self, key: Key) -> Result<Promote, Error> {
        let Some(idx) = self.active(key)? else { return Ok(Promote::TooLate) };
        let req = self.req(idx);
        if req.cancelled {
            return Ok(Promote::TooLate);
        }
        if req.class == Class::Demand {
            return Ok(Promote::AlreadyDemand);
        }
        let queued = req.next < req.total;
        if queued {
            self.unqueue(idx);
        }
        let req = self.req(idx);
        req.class = Class::Demand;
        req.promoted = true;
        if queued {
            self.demand.push_back(idx);
        }
        self.promoted += 1;
        Ok(Promote::Promoted)
    }

    /// Whether `take_chunk` would return something.
    #[cfg(test)]
    pub(crate) fn startable(&self) -> bool {
        self.next_startable(Instant::now()).is_some()
    }

    /// The copy of `file` to read the next chunk from: among the drives that
    /// have room for the class, the one expected to deliver it soonest; of
    /// equals the one with fewer chunks, then the first. `None` when every
    /// drive with a copy is at its limit.
    fn pick_copy(&self, file: u16, class: Class, now: Instant) -> Option<CopyRef> {
        let copies = &self.copies[usize::from(file)];
        if let [only] = copies.as_slice() {
            let room = self.drives[usize::from(only.drive)].has_room(class, &self.limits);
            return room.then_some(*only);
        }
        let mut best: Option<((u128, usize), CopyRef)> = None;
        for copy in copies {
            let drive = &self.drives[usize::from(copy.drive)];
            if !drive.has_room(class, &self.limits) {
                continue;
            }
            let rank = (drive.expected(now), drive.in_flight);
            if best.is_none_or(|(least, _)| rank < least) {
                best = Some((rank, *copy));
            }
        }
        best.map(|(_, copy)| copy)
    }

    /// What starts next: the oldest demand read that a drive with room can
    /// serve, else the oldest such prefetch. Returns its place in its queue,
    /// the request, the class and the copy to read.
    ///
    /// With one drive this is the head of a queue or nothing. With several, a
    /// read whose drives are all busy does not hold up a later one that an
    /// idle drive can serve.
    fn next_startable(&self, now: Instant) -> Option<(usize, u32, Class, CopyRef)> {
        if self.held > 0 {
            return None;
        }
        for class in [Class::Demand, Class::Prefetch] {
            let queue = match class {
                Class::Demand => &self.demand,
                Class::Prefetch => &self.prefetch,
            };
            if queue.is_empty() || !self.drives.iter().any(|d| d.has_room(class, &self.limits)) {
                continue;
            }
            for (pos, idx) in queue.iter().enumerate() {
                let req = self.slab[*idx as usize].as_ref().expect("a queued request exists");
                if let Some(copy) = self.pick_copy(req.file, class, now) {
                    return Some((pos, *idx, class, copy));
                }
            }
        }
        None
    }

    /// The next chunk to start, within the depth limits of the drives; see
    /// `next_startable` for which one that is.
    pub(crate) fn take_chunk(&mut self, now: Instant) -> Option<Picked> {
        let (pos, idx, class, copy) = self.next_startable(now)?;
        let req = self.req(idx);
        let rel = req.next;
        let len = req.chunk.min(req.total - rel);
        req.next += len;
        req.in_flight += 1;
        let finished_queueing = req.next == req.total;
        let key = req.key;
        let chunk_index = rel / req.chunk;
        let offset = req.start + rel as u64;
        let buffer = req.buffer.as_mut().expect("a queued request holds its buffer");
        // SAFETY: `rel + len <= total <= buffer.len()` (checked at submit),
        // so the pointer stays inside the buffer.
        let ptr = unsafe { buffer.as_mut_ptr().add(rel) };
        if finished_queueing {
            self.queue_mut(class).remove(pos);
        }
        self.in_flight += 1;
        self.peak_in_flight = self.peak_in_flight.max(self.in_flight);
        let load = self.drives[usize::from(copy.drive)].started(class);
        if let Some(trace) = &mut self.trace {
            trace.push((key, chunk_index));
        }
        Some(Picked {
            req: idx,
            rel,
            len,
            class,
            copy: copy.index,
            fd: copy.fd,
            drive: copy.drive,
            started: now,
            load,
            offset,
            ptr,
        })
    }

    /// A chunk came back.
    pub(crate) fn chunk_done(&mut self, chunk: &Picked, result: ChunkResult, now: Instant) {
        self.in_flight -= 1;
        let delivered = match result {
            ChunkResult::Bytes(n) => n,
            ChunkResult::Errno(_) => 0,
        };
        self.drives[usize::from(chunk.drive)].came_back(chunk, delivered, now);
        let idx = chunk.req;
        // Field-wise borrows: the path is only cloned when an error needs it.
        let req = self.slab[idx as usize].as_mut().expect("an indexed request exists");
        req.in_flight -= 1;
        let stop = match result {
            ChunkResult::Errno(errno) => {
                if req.error.is_none() {
                    req.error = Some(Error::Io {
                        path: self.files[usize::from(req.file)].copies[usize::from(chunk.copy)]
                            .path
                            .clone(),
                        offset: chunk.offset,
                        len: chunk.len,
                        source: std::io::Error::from_raw_os_error(errno),
                    });
                }
                true
            }
            ChunkResult::Bytes(n) => {
                req.device_bytes += n as u64;
                if n < chunk.len {
                    // The file ends here; chunks after this one read nothing.
                    req.eof_at = req.eof_at.min(chunk.rel + n);
                    true
                } else {
                    false
                }
            }
        };
        if stop {
            self.stop_issuing(idx);
        }
        let req = self.req(idx);
        if req.in_flight == 0 && req.next == req.total {
            self.finish(idx, now);
        }
    }

    fn finish(&mut self, idx: u32, now: Instant) {
        let mut req = self.slab[idx as usize].take().expect("finishing a live request");
        self.free.push(idx);
        let file = &self.files[usize::from(req.file)];
        let wanted_end = req.skip + req.wanted;
        let result = if req.cancelled {
            Err(Error::Cancelled)
        } else if let Some(error) = req.error.take() {
            Err(error)
        } else if req.eof_at < wanted_end {
            Err(Error::ShortRead {
                path: file.path.clone(),
                offset: req.start + req.skip as u64,
                wanted: req.wanted,
                got: req.eof_at.saturating_sub(req.skip),
                size: file.size,
            })
        } else {
            Ok(())
        };
        let latency = now.saturating_duration_since(req.submitted_at);
        let ok = result.is_ok();
        let class = req.submitted_as;
        {
            let stats = self.stats_mut(class);
            stats.completed += 1;
            stats.device_bytes += req.device_bytes;
            if ok {
                stats.bytes += req.wanted as u64;
            } else if !req.cancelled {
                stats.failed += 1;
            }
        }
        if ok {
            match class {
                Class::Demand => self.demand_latency.record(latency),
                Class::Prefetch => self.prefetch_latency.record(latency),
            }
        }
        let order = self.order;
        self.order += 1;
        self.keys.insert(req.key, KeyState::Done);
        self.done.push_back(Completion {
            key: req.key,
            class,
            promoted: req.promoted,
            result,
            buffer: req.buffer.take().expect("a request holds its buffer until it finishes"),
            skip: req.skip,
            len: if ok { req.wanted } else { 0 },
            latency,
            order,
        });
        self.news = true;
    }

    /// Cancels everything that has not started, at shutdown. Reads with
    /// chunks at the device finish when those come back.
    pub(crate) fn cancel_queued(&mut self, now: Instant) {
        let queued: Vec<u32> = self.demand.iter().chain(&self.prefetch).copied().collect();
        for idx in queued {
            let key = self.req(idx).key;
            let _ = self.cancel(key, now);
        }
    }

    /// The ring failed for good. Reads that never started complete with the
    /// reason, and from now on nothing is accepted. Reads with chunks at the
    /// device cannot be finished (the kernel may still write into their
    /// buffers), so their buffers are leaked rather than freed.
    pub(crate) fn fail(&mut self, why: String, now: Instant) {
        for idx in 0..self.slab.len() as u32 {
            if self.slab[idx as usize].is_none() {
                continue;
            }
            self.stop_issuing(idx);
            let req = self.req(idx);
            if req.in_flight == 0 {
                req.error = Some(Error::Stopped(why.clone()));
                self.finish(idx, now);
            } else {
                std::mem::forget(req.buffer.take());
            }
        }
        self.in_flight = 0;
        for drive in &mut self.drives {
            drive.in_flight = 0;
            drive.in_flight_prefetch = 0;
        }
        self.failed = Some(why);
    }

    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight
    }

    pub(crate) fn is_known(&self, key: Key) -> bool {
        self.keys.contains_key(&key)
    }

    pub(crate) fn is_done(&self, key: Key) -> bool {
        matches!(self.keys.get(&key), Some(KeyState::Done))
    }

    /// Takes the completion of `key` out of the queue of finished reads.
    pub(crate) fn take_done(&mut self, key: Key) -> Option<Completion> {
        let pos = self.done.iter().position(|c| c.key == key)?;
        self.keys.remove(&key);
        self.done.remove(pos)
    }

    pub(crate) fn take_next_done(&mut self) -> Option<Completion> {
        let completion = self.done.pop_front()?;
        self.keys.remove(&completion.key);
        Some(completion)
    }

    pub(crate) fn take_all_done(&mut self, out: &mut Vec<Completion>) -> usize {
        let n = self.done.len();
        for completion in self.done.drain(..) {
            self.keys.remove(&completion.key);
            out.push(completion);
        }
        n
    }

    pub(crate) fn trace(&self) -> Vec<(Key, usize)> {
        self.trace.clone().unwrap_or_default()
    }

    /// What each drive has done, in the order of the drives.
    pub(crate) fn drive_stats(&self) -> Vec<DriveStats> {
        let stats = |d: &Drive| DriveStats {
            chunks: d.chunks,
            bytes: d.bytes,
            in_flight: d.in_flight,
            peak_in_flight: d.peak_in_flight,
            alone_ns_per_mib: d.alone.ns_per_mib,
            shared_ns_per_mib: d.shared.ns_per_mib,
        };
        self.drives.iter().map(stats).collect()
    }

    /// Counters, and the latency windows to sort outside the lock.
    pub(crate) fn stats(&self) -> (Stats, [(u64, Vec<u64>); 2]) {
        let stats = Stats {
            demand: self.demand_stats,
            prefetch: self.prefetch_stats,
            cancelled: self.cancelled,
            promoted: self.promoted,
            queued_demand: self.demand.len(),
            queued_prefetch: self.prefetch.len(),
            in_flight: self.in_flight,
            peak_in_flight: self.peak_in_flight,
            outstanding: self.keys.len(),
            ..Stats::default()
        };
        (stats, [self.demand_latency.snapshot(), self.prefetch_latency.snapshot()])
    }
}

/// Fills in the latency percentiles of a snapshot.
pub(crate) fn finish_stats((mut stats, windows): (Stats, [(u64, Vec<u64>); 2])) -> Stats {
    let [demand, prefetch] = windows;
    stats.demand_latency = summarize(demand);
    stats.prefetch_latency = summarize(prefetch);
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{BufferedReason, CopyInfo, DriveId, FileId, IoMode};
    use std::path::PathBuf;

    const BLOCK: u64 = 512;

    /// A file of 1 MiB with one copy on each of `drives`.
    fn file_on(name: &str, drives: &[u16]) -> FileInfo {
        let mode = IoMode::Buffered(BufferedReason::Disabled);
        let mut file =
            FileInfo::single(PathBuf::from(name), 1 << 20, mode, None, BLOCK, DriveId(drives[0]));
        for drive in &drives[1..] {
            file.copies.push(CopyInfo {
                path: PathBuf::from(format!("{name}.copy-on-{drive}")),
                mode,
                direct_align: None,
                drive: DriveId(*drive),
            });
        }
        file
    }

    fn sched_of(files: Vec<FileInfo>, depth: usize, prefetch_depth: usize) -> Sched {
        let limits = Limits { depth, prefetch_depth, prefetch_chunk: 1024, demand_chunk: 1 << 20 };
        Sched::new(limits, files.into(), true)
    }

    /// One file on one drive.
    fn sched(depth: usize, prefetch_depth: usize) -> Sched {
        sched_of(vec![file_on("test.bin", &[0])], depth, prefetch_depth)
    }

    fn read(key: Key, class: Class, offset: u64, len: usize) -> Read {
        Read {
            key,
            file: FileId(0),
            offset,
            len,
            class,
            // Tests of the queues never start real reads, so the memory is
            // never written; it only has to exist and be aligned.
            buffer: Buffer::aligned(len + 2 * BLOCK as usize, BLOCK as usize),
        }
    }

    /// Starts every chunk that may start and reports them as (key, chunk).
    fn drain_starts(s: &mut Sched) -> Vec<(Key, usize)> {
        let mut started = Vec::new();
        while let Some(p) = s.take_chunk(Instant::now()) {
            let key = s.req(p.req).key;
            started.push((key, p.rel / s.req(p.req).chunk));
            s.chunk_done(&p, ChunkResult::Bytes(p.len), Instant::now());
        }
        started
    }

    #[test]
    fn demand_reads_start_before_every_prefetch_chunk() {
        let mut s = sched(4, 4);
        s.submit(read(1, Class::Prefetch, 0, 4096), Instant::now()).unwrap();
        s.submit(read(2, Class::Demand, 8192, 512), Instant::now()).unwrap();
        s.submit(read(3, Class::Demand, 9000, 512), Instant::now()).unwrap();
        let started = drain_starts(&mut s);
        let mut want = vec![(2, 0), (3, 0)];
        want.extend((0..4).map(|c| (1, c)));
        assert_eq!(started, want, "demand first (FIFO), then the prefetch chunk by chunk");
    }

    #[test]
    fn a_prefetch_is_cut_into_chunks_and_a_demand_read_is_not() {
        let mut s = sched(4, 4);
        s.submit(read(1, Class::Prefetch, 0, 4096), Instant::now()).unwrap();
        s.submit(read(2, Class::Demand, 0, 4096), Instant::now()).unwrap();
        assert_eq!(drain_starts(&mut s), [(2, 0), (1, 0), (1, 1), (1, 2), (1, 3)]);
    }

    #[test]
    fn the_prefetch_limit_keeps_room_for_demand() {
        let mut s = sched(8, 2);
        s.submit(read(1, Class::Prefetch, 0, 8192), Instant::now()).unwrap();
        let a = s.take_chunk(Instant::now()).unwrap();
        let b = s.take_chunk(Instant::now()).unwrap();
        assert!(
            s.take_chunk(Instant::now()).is_none(),
            "only two prefetch chunks may be at the device"
        );
        // A demand read still starts at once, though prefetch is at its limit.
        s.submit(read(2, Class::Demand, 0, 512), Instant::now()).unwrap();
        assert!(s.startable());
        let d = s.take_chunk(Instant::now()).unwrap();
        assert_eq!(s.req(d.req).key, 2);
        s.chunk_done(&a, ChunkResult::Bytes(a.len), Instant::now());
        assert!(s.startable(), "a finished prefetch chunk frees a prefetch slot");
        s.chunk_done(&b, ChunkResult::Bytes(b.len), Instant::now());
        s.chunk_done(&d, ChunkResult::Bytes(d.len), Instant::now());
    }

    #[test]
    fn the_total_depth_is_a_hard_limit() {
        let mut s = sched(2, 2);
        for key in 0..5 {
            s.submit(read(key, Class::Demand, 0, 512), Instant::now()).unwrap();
        }
        assert!(s.take_chunk(Instant::now()).is_some());
        assert!(s.take_chunk(Instant::now()).is_some());
        assert!(s.take_chunk(Instant::now()).is_none());
        assert_eq!(s.in_flight(), 2);
    }

    #[test]
    fn promotion_moves_the_remaining_chunks_ahead_of_other_prefetches() {
        let mut s = sched(1, 1);
        s.submit(read(1, Class::Prefetch, 0, 2048), Instant::now()).unwrap();
        s.submit(read(2, Class::Prefetch, 0, 2048), Instant::now()).unwrap();
        s.submit(read(3, Class::Demand, 0, 512), Instant::now()).unwrap();
        // One chunk of the first prefetch goes out before anything is decided.
        let first = s.take_chunk(Instant::now()).unwrap();
        assert_eq!(s.req(first.req).key, 3, "the demand read is first");
        s.chunk_done(&first, ChunkResult::Bytes(first.len), Instant::now());
        let second = s.take_chunk(Instant::now()).unwrap();
        assert_eq!(s.req(second.req).key, 1);
        s.chunk_done(&second, ChunkResult::Bytes(second.len), Instant::now());

        assert_eq!(s.promote(2).unwrap(), Promote::Promoted);
        assert_eq!(s.promote(2).unwrap(), Promote::AlreadyDemand);
        let rest = drain_starts(&mut s);
        assert_eq!(rest, [(2, 0), (2, 1), (1, 1)], "promoted chunks first");
        assert_eq!(s.stats().0.promoted, 1);
    }

    #[test]
    fn promoted_reads_queue_behind_waiting_demand_reads() {
        let mut s = sched(1, 1);
        s.submit(read(1, Class::Prefetch, 0, 1024), Instant::now()).unwrap();
        s.submit(read(2, Class::Demand, 0, 512), Instant::now()).unwrap();
        s.promote(1).unwrap();
        let started = drain_starts(&mut s);
        assert_eq!(started, [(2, 0), (1, 0)]);
    }

    #[test]
    fn cancelling_a_queued_read_completes_it_at_once_and_returns_the_buffer() {
        let mut s = sched(1, 1);
        let r = read(7, Class::Prefetch, 0, 4096);
        let ptr = r.buffer.as_ptr();
        s.submit(r, Instant::now()).unwrap();
        assert_eq!(s.cancel(7, Instant::now()).unwrap(), Cancel::Cancelled);
        let c = s.take_done(7).expect("completed without any chunk running");
        assert!(matches!(c.result, Err(Error::Cancelled)));
        assert_eq!(c.buffer.as_ptr(), ptr);
        assert!(c.data().is_empty());
        assert!(drain_starts(&mut s).is_empty());
        assert_eq!(s.stats().0.cancelled, 1);
    }

    #[test]
    fn cancelling_a_running_read_waits_for_its_chunks_and_stops_the_rest() {
        let mut s = sched(2, 2);
        s.submit(read(7, Class::Prefetch, 0, 4096), Instant::now()).unwrap();
        let a = s.take_chunk(Instant::now()).unwrap();
        let b = s.take_chunk(Instant::now()).unwrap();
        assert_eq!(s.cancel(7, Instant::now()).unwrap(), Cancel::Cancelled);
        assert!(s.take_done(7).is_none(), "two chunks are still at the device");
        assert!(s.take_chunk(Instant::now()).is_none(), "the rest of the read is not started");
        s.chunk_done(&a, ChunkResult::Bytes(a.len), Instant::now());
        assert!(s.take_done(7).is_none());
        s.chunk_done(&b, ChunkResult::Bytes(b.len), Instant::now());
        let c = s.take_done(7).unwrap();
        assert!(matches!(c.result, Err(Error::Cancelled)));
        assert_eq!(s.trace(), [(7, 0), (7, 1)]);
    }

    #[test]
    fn a_finished_read_cannot_be_cancelled_or_promoted() {
        let mut s = sched(1, 1);
        s.submit(read(1, Class::Prefetch, 0, 512), Instant::now()).unwrap();
        drain_starts(&mut s);
        assert_eq!(s.cancel(1, Instant::now()).unwrap(), Cancel::TooLate);
        assert_eq!(s.promote(1).unwrap(), Promote::TooLate);
        assert!(matches!(s.cancel(99, Instant::now()), Err(Error::UnknownKey(99))));
        assert!(matches!(s.promote(99), Err(Error::UnknownKey(99))));
        assert!(s.take_done(1).unwrap().result.is_ok());
        assert!(matches!(s.cancel(1, Instant::now()), Err(Error::UnknownKey(1))));
    }

    #[test]
    fn a_short_chunk_ends_the_read_and_fails_it_when_the_data_is_missing() {
        let mut s = sched(4, 4);
        // 3072 bytes wanted in three chunks of 1024; the file ends after
        // 1500 bytes, inside the second chunk.
        s.submit(read(1, Class::Prefetch, 0, 3072), Instant::now()).unwrap();
        let a = s.take_chunk(Instant::now()).unwrap();
        let b = s.take_chunk(Instant::now()).unwrap();
        s.chunk_done(&a, ChunkResult::Bytes(a.len), Instant::now());
        s.chunk_done(&b, ChunkResult::Bytes(1500 - a.len), Instant::now());
        assert!(
            s.take_chunk(Instant::now()).is_none(),
            "nothing after the end of the file is read"
        );
        let c = s.take_done(1).unwrap();
        match c.result {
            Err(Error::ShortRead { wanted: 3072, got: 1500, size: 1048576, .. }) => {}
            other => panic!("unexpected {other:?}"),
        }
        assert!(c.data().is_empty());
    }

    #[test]
    fn a_short_chunk_past_the_wanted_bytes_is_fine() {
        let mut s = sched(4, 4);
        // 600 bytes wanted at offset 100: the span is [0, 1024); the file
        // ends at 800, inside the last block, after the wanted bytes.
        s.submit(read(1, Class::Demand, 100, 600), Instant::now()).unwrap();
        let p = s.take_chunk(Instant::now()).unwrap();
        assert_eq!(p.len, 1024);
        s.chunk_done(&p, ChunkResult::Bytes(800), Instant::now());
        let c = s.take_done(1).unwrap();
        assert!(c.result.is_ok());
        assert_eq!((c.skip, c.len), (100, 600));
    }

    #[test]
    fn an_io_error_fails_only_its_read() {
        let mut s = sched(4, 4);
        s.submit(read(1, Class::Demand, 0, 512), Instant::now()).unwrap();
        s.submit(read(2, Class::Demand, 512, 512), Instant::now()).unwrap();
        let a = s.take_chunk(Instant::now()).unwrap();
        let b = s.take_chunk(Instant::now()).unwrap();
        s.chunk_done(&a, ChunkResult::Errno(libc::EIO), Instant::now());
        s.chunk_done(&b, ChunkResult::Bytes(b.len), Instant::now());
        let (c1, c2) = (s.take_done(1).unwrap(), s.take_done(2).unwrap());
        assert!(matches!(c1.result, Err(Error::Io { .. })));
        assert!(c2.result.is_ok());
        let stats = s.stats().0;
        assert_eq!((stats.demand.completed, stats.demand.failed), (2, 1));
    }

    #[test]
    fn submit_checks_buffer_key_and_file() {
        let mut s = sched(1, 1);
        let mut small = read(1, Class::Demand, 100, 600);
        small.buffer = Buffer::aligned(512, 512);
        let rejected = s.submit(small, Instant::now()).unwrap_err();
        assert!(matches!(rejected.error, Error::BufferTooSmall { need: 1024, have: 512, .. }));
        assert_eq!(rejected.read.buffer.len(), 512, "the buffer comes back");

        let mut off = read(1, Class::Demand, 0, 100);
        let mut backing = vec![0u8; 2048];
        // SAFETY: one byte into `backing`, which outlives the buffer; the
        // scheduler only looks at the address in this test.
        off.buffer = unsafe { Buffer::from_raw(backing.as_mut_ptr().add(1), 1024) };
        assert!(matches!(
            s.submit(off, Instant::now()).unwrap_err().error,
            Error::BufferMisaligned { .. }
        ));

        s.submit(read(1, Class::Demand, 0, 100), Instant::now()).unwrap();
        assert!(matches!(
            s.submit(read(1, Class::Demand, 0, 100), Instant::now()).unwrap_err().error,
            Error::DuplicateKey(1)
        ));
        let mut other = read(2, Class::Demand, 0, 100);
        other.file = FileId(3);
        assert!(matches!(
            s.submit(other, Instant::now()).unwrap_err().error,
            Error::UnknownFile(FileId(3), 1)
        ));
    }

    #[test]
    fn a_key_is_free_again_once_its_completion_is_taken() {
        let mut s = sched(1, 1);
        s.submit(read(1, Class::Demand, 0, 100), Instant::now()).unwrap();
        drain_starts(&mut s);
        assert!(s.submit(read(1, Class::Demand, 0, 100), Instant::now()).is_err());
        s.take_done(1).unwrap();
        s.submit(read(1, Class::Demand, 0, 100), Instant::now()).unwrap();
    }

    #[test]
    fn an_empty_read_completes_without_touching_the_device() {
        let mut s = sched(1, 1);
        s.submit(read(1, Class::Demand, 4000, 0), Instant::now()).unwrap();
        let c = s.take_done(1).unwrap();
        assert!(c.result.is_ok());
        assert!(c.data().is_empty());
        assert!(s.trace().is_empty());
    }

    #[test]
    fn the_failure_of_the_ring_completes_queued_reads_and_refuses_new_ones() {
        let mut s = sched(1, 1);
        s.submit(read(1, Class::Demand, 0, 100), Instant::now()).unwrap();
        s.submit(read(2, Class::Demand, 0, 100), Instant::now()).unwrap();
        let running = s.take_chunk(Instant::now()).unwrap();
        s.fail("test".into(), Instant::now());
        assert!(matches!(s.take_done(2).unwrap().result, Err(Error::Stopped(_))));
        assert!(s.take_done(1).is_none(), "its buffer may still be written by the kernel");
        assert!(matches!(
            s.submit(read(3, Class::Demand, 0, 100), Instant::now()).unwrap_err().error,
            Error::Stopped(_)
        ));
        let _ = running;
    }

    #[test]
    fn completions_are_numbered_in_the_order_reads_finish() {
        let mut s = sched(2, 2);
        s.submit(read(1, Class::Demand, 0, 512), Instant::now()).unwrap();
        s.submit(read(2, Class::Demand, 0, 512), Instant::now()).unwrap();
        let (a, b) = (s.take_chunk(Instant::now()).unwrap(), s.take_chunk(Instant::now()).unwrap());
        // The second read comes back first.
        s.chunk_done(&b, ChunkResult::Bytes(b.len), Instant::now());
        s.chunk_done(&a, ChunkResult::Bytes(a.len), Instant::now());
        assert_eq!(s.take_done(2).unwrap().order, 0);
        assert_eq!(s.take_done(1).unwrap().order, 1);
    }

    #[test]
    fn statistics_count_per_class() {
        let mut s = sched(4, 4);
        s.submit(read(1, Class::Demand, 100, 600), Instant::now()).unwrap();
        s.submit(read(2, Class::Prefetch, 0, 2048), Instant::now()).unwrap();
        s.submit(read(3, Class::Prefetch, 0, 2048), Instant::now()).unwrap();
        s.cancel(3, Instant::now()).unwrap();
        s.promote(2).unwrap();
        drain_starts(&mut s);
        let st = s.stats().0;
        assert_eq!((st.demand.requests, st.demand.completed, st.demand.bytes), (1, 1, 600));
        assert_eq!(st.demand.device_bytes, 1024);
        assert_eq!((st.prefetch.requests, st.prefetch.completed, st.prefetch.bytes), (2, 2, 2048));
        assert_eq!((st.cancelled, st.promoted), (1, 1));
        assert_eq!((st.queued_demand, st.queued_prefetch, st.in_flight), (0, 0, 0));
        assert_eq!(st.outstanding, 3, "completions not yet taken still hold their keys");
        let latency = finish_stats(s.stats()).demand_latency;
        assert_eq!(latency.total, 1);
    }

    // ---- several drives ----

    use std::time::Duration;

    /// Starts every chunk that may start, without finishing any, and returns
    /// them.
    fn start_all(s: &mut Sched, now: Instant) -> Vec<Picked> {
        std::iter::from_fn(|| s.take_chunk(now)).collect()
    }

    fn demand(key: Key, file: u16, len: usize) -> Read {
        let mut r = read(key, Class::Demand, 0, len);
        r.file = FileId(file);
        r
    }

    #[test]
    fn every_drive_has_a_depth_of_its_own() {
        let mut s = sched_of(vec![file_on("model.bin", &[0, 1])], 2, 2);
        for key in 0..6 {
            s.submit(demand(key, 0, 512), Instant::now()).unwrap();
        }
        let started = start_all(&mut s, Instant::now());
        assert_eq!(started.len(), 4, "two chunks at each of the two drives");
        let at = |drive| started.iter().filter(|p| p.drive == drive).count();
        assert_eq!((at(0), at(1)), (2, 2));
        assert_eq!(s.in_flight(), 4);
        assert!(!s.startable());
        // A chunk that comes back frees a place at its own drive only.
        s.chunk_done(&started[0], ChunkResult::Bytes(512), Instant::now());
        let next = s.take_chunk(Instant::now()).unwrap();
        assert_eq!(next.drive, started[0].drive);
        assert!(s.take_chunk(Instant::now()).is_none());
    }

    #[test]
    fn chunks_of_a_file_with_copies_spread_over_idle_drives_from_the_first() {
        // Nothing has been measured yet: the bytes at each drive decide, so
        // the chunks alternate, starting with the first copy.
        let mut s = sched_of(vec![file_on("model.bin", &[0, 1])], 8, 8);
        for key in 0..6 {
            s.submit(demand(key, 0, 512), Instant::now()).unwrap();
        }
        let drives: Vec<u16> = start_all(&mut s, Instant::now()).iter().map(|p| p.drive).collect();
        assert_eq!(drives, [0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn each_copy_is_read_through_its_own_descriptor() {
        // Two files of two copies: descriptors 0 and 1 are the copies of the
        // first file, 2 and 3 those of the second, on drives 0 and 1 each.
        let files = vec![file_on("a.bin", &[0, 1]), file_on("b.bin", &[0, 1])];
        let mut s = sched_of(files, 8, 8);
        for (key, file) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
            s.submit(demand(key, file, 512), Instant::now()).unwrap();
        }
        let started = start_all(&mut s, Instant::now());
        let seen: Vec<(u16, u16, u16)> = started.iter().map(|p| (p.fd, p.copy, p.drive)).collect();
        assert_eq!(seen, [(0, 0, 0), (1, 1, 1), (2, 0, 0), (3, 1, 1)]);
    }

    /// A file with a copy on each of two drives, and a file on each drive
    /// alone, with which a test puts chunks on the drive it wants: files 0
    /// (both), 1 (drive 0 only) and 2 (drive 1 only).
    fn mirrored_and_one_on_each() -> Sched {
        let files =
            vec![file_on("model.bin", &[0, 1]), file_on("on0.bin", &[0]), file_on("on1.bin", &[1])];
        sched_of(files, 64, 64)
    }

    /// Runs `n` chunks of 512 bytes at once on the drive of `file`, all of
    /// which come back `took` after `at`.
    fn measure(s: &mut Sched, file: u16, n: u64, at: Instant, took: Duration) {
        for key in 0..n {
            s.submit(demand(10_000 + key, file, 512), at).unwrap();
        }
        let started = start_all(s, at);
        assert_eq!(started.len() as u64, n);
        for p in &started {
            s.chunk_done(p, ChunkResult::Bytes(512), at + took);
        }
        for key in 0..n {
            s.take_done(10_000 + key).unwrap();
        }
    }

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn a_drive_is_measured_alone_and_under_load() {
        let mut s = mirrored_and_one_on_each();
        let t0 = Instant::now();
        // One chunk alone at drive 0 takes 1 ms: 2048 ms per MiB for 512 bytes.
        measure(&mut s, 1, 1, t0, MS);
        let stats = s.drive_stats();
        assert_eq!((stats[0].alone_ns_per_mib, stats[0].shared_ns_per_mib), (2_048_000_000, 0));
        assert_eq!((stats[1].alone_ns_per_mib, stats[1].shared_ns_per_mib), (0, 0));
        // Three at once take 6 ms each: each saw two chunks on average (three
        // when it started or when it ended, one at the other end), so the
        // drive's time per chunk and MiB is half of what one took.
        measure(&mut s, 1, 3, t0 + 10 * MS, 6 * MS);
        let stats = s.drive_stats();
        assert_eq!(stats[0].alone_ns_per_mib, 2_048_000_000, "unchanged");
        assert_eq!(stats[0].shared_ns_per_mib, 6 * 2_048_000_000 / 2);
        assert_eq!((stats[0].chunks, stats[0].bytes), (4, 4 * 512));
    }

    #[test]
    fn a_single_read_goes_to_the_drive_that_is_faster_alone() {
        let mut s = mirrored_and_one_on_each();
        let t0 = Instant::now();
        measure(&mut s, 1, 1, t0, 3 * MS);
        measure(&mut s, 2, 1, t0, MS);
        for key in 0..3 {
            let at = t0 + (10 + 2 * key as u32) * MS;
            s.submit(demand(key, 0, 512), at).unwrap();
            let p = s.take_chunk(at).unwrap();
            assert_eq!(p.drive, 1, "the second drive answers a lone read in a third of the time");
            s.chunk_done(&p, ChunkResult::Bytes(512), at + MS);
        }
    }

    #[test]
    fn under_load_the_drives_share_the_chunks_by_their_throughput() {
        let mut s = mirrored_and_one_on_each();
        let t0 = Instant::now();
        // Three chunks at once: 2 ms each at drive 0, 7 ms each at drive 1.
        measure(&mut s, 1, 3, t0, 2 * MS);
        measure(&mut s, 2, 3, t0, 7 * MS);
        let stats = s.drive_stats();
        assert_eq!(2 * stats[1].shared_ns_per_mib, 7 * stats[0].shared_ns_per_mib);
        // Eight chunks of the mirrored file at once: drive 0 takes chunks
        // until a fourth there would take longer than a first at drive 1.
        let t1 = t0 + 10 * MS;
        for key in 0..8 {
            s.submit(demand(key, 0, 512), t1).unwrap();
        }
        let drives: Vec<u16> = start_all(&mut s, t1).iter().map(|p| p.drive).collect();
        assert_eq!(drives, [0, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn a_drive_that_has_not_been_measured_is_tried() {
        let mut s = mirrored_and_one_on_each();
        let t0 = Instant::now();
        // Drive 0 is known alone; drive 1 not at all, so a lone read tries it.
        measure(&mut s, 1, 1, t0, MS);
        let t1 = t0 + 5 * MS;
        s.submit(demand(0, 0, 512), t1).unwrap();
        let p = s.take_chunk(t1).unwrap();
        assert_eq!(p.drive, 1);
        // It turns out three times as slow, and the next lone read knows.
        s.chunk_done(&p, ChunkResult::Bytes(512), t1 + 3 * MS);
        let t2 = t1 + 5 * MS;
        s.submit(demand(1, 0, 512), t2).unwrap();
        let p = s.take_chunk(t2).unwrap();
        assert_eq!(p.drive, 0);
        s.chunk_done(&p, ChunkResult::Bytes(512), t2 + MS);

        // Neither drive has been measured under load.
        let t3 = t2 + 5 * MS;
        for key in 10..16 {
            s.submit(demand(key, 0, 512), t3).unwrap();
        }
        let started = start_all(&mut s, t3);
        // A lone chunk at drive 0 is expected in 1 ms and at drive 1 in 3 ms,
        // and more chunks are not yet known to slow either: all go to drive 0.
        assert!(started.iter().all(|p| p.drive == 0));
        // They come back after 7 ms, which measures drive 0 under load (six
        // chunks, 3.5 on average while each was there: 2 ms per chunk). Now a
        // second chunk at drive 0 is expected in 4 ms and a burst goes on to
        // drive 1, which is not yet known to slow down under load.
        for p in &started {
            s.chunk_done(p, ChunkResult::Bytes(512), t3 + 7 * MS);
        }
        assert_eq!(s.drive_stats()[0].shared_ns_per_mib, 2 * 2_048_000_000);
        let t4 = t3 + 10 * MS;
        for key in 20..24 {
            s.submit(demand(key, 0, 512), t4).unwrap();
        }
        let drives: Vec<u16> = start_all(&mut s, t4).iter().map(|p| p.drive).collect();
        assert_eq!(drives, [0, 1, 1, 1]);
    }

    #[test]
    fn a_measurement_stops_counting_after_a_second() {
        let mut s = mirrored_and_one_on_each();
        let t0 = Instant::now();
        measure(&mut s, 1, 1, t0, MS);
        measure(&mut s, 2, 1, t0, 3 * MS);
        // The slower drive is left alone while its measurement counts.
        let lone = |s: &mut Sched, key: Key, at: Instant| {
            s.submit(demand(key, 0, 512), at).unwrap();
            let p = s.take_chunk(at).unwrap();
            s.chunk_done(&p, ChunkResult::Bytes(512), at + MS);
            p.drive
        };
        assert_eq!(lone(&mut s, 0, t0 + 100 * MS), 0);
        assert_eq!(lone(&mut s, 1, t0 + 900 * MS), 0);
        // After a second without a chunk it is tried once more, measured
        // again (1 ms this time: it is as fast as the other now), and used.
        let later = t0 + 3 * MS + MEASUREMENT_LIFE + MS;
        assert_eq!(lone(&mut s, 2, later), 1);
        let stats = s.drive_stats();
        assert_eq!(stats[1].alone_ns_per_mib, 2_048_000_000, "replaced, not averaged in");
    }

    #[test]
    fn a_read_whose_drive_is_busy_does_not_hold_up_one_for_an_idle_drive() {
        // Two files without copies, each on its own drive, as the parts of a
        // split model can be.
        let files = vec![file_on("part1.bin", &[0]), file_on("part2.bin", &[1])];
        let mut s = sched_of(files, 1, 1);
        s.submit(demand(1, 0, 512), Instant::now()).unwrap();
        s.submit(demand(2, 0, 512), Instant::now()).unwrap();
        s.submit(demand(3, 1, 512), Instant::now()).unwrap();
        let a = s.take_chunk(Instant::now()).unwrap();
        assert_eq!((s.req(a.req).key, a.drive), (1, 0));
        // Read 2 is older than read 3, but its drive is busy and the other is idle.
        let b = s.take_chunk(Instant::now()).unwrap();
        assert_eq!((s.req(b.req).key, b.drive), (3, 1));
        assert!(s.take_chunk(Instant::now()).is_none());
        s.chunk_done(&a, ChunkResult::Bytes(512), Instant::now());
        let c = s.take_chunk(Instant::now()).unwrap();
        assert_eq!((s.req(c.req).key, c.drive), (2, 0));
        assert_eq!(s.trace(), [(1, 0), (3, 0), (2, 0)]);
    }

    #[test]
    fn the_prefetch_limit_holds_at_every_drive_and_demand_still_goes_first() {
        let mut s = sched_of(vec![file_on("model.bin", &[0, 1])], 4, 1);
        s.submit(read(1, Class::Prefetch, 0, 8192), Instant::now()).unwrap();
        let prefetches = start_all(&mut s, Instant::now());
        let drives: Vec<u16> = prefetches.iter().map(|p| p.drive).collect();
        assert_eq!(drives, [0, 1], "one prefetch chunk at each drive, no more");
        // Demand reads use the rest of both depths, ahead of the prefetch.
        for key in 10..20 {
            s.submit(demand(key, 0, 512), Instant::now()).unwrap();
        }
        let demands = start_all(&mut s, Instant::now());
        assert_eq!(demands.len(), 6, "three more at each drive");
        assert!(demands.iter().all(|p| p.class == Class::Demand));
        // A prefetch chunk that returns is replaced by a demand read while
        // there are any, not by the next prefetch chunk.
        s.chunk_done(&prefetches[0], ChunkResult::Bytes(1024), Instant::now());
        let next = s.take_chunk(Instant::now()).unwrap();
        assert_eq!((next.class, next.drive), (Class::Demand, 0));
    }

    #[test]
    fn an_error_names_the_copy_it_came_from() {
        let mut s = sched_of(vec![file_on("model.bin", &[0, 1])], 4, 4);
        s.submit(demand(1, 0, 512), Instant::now()).unwrap();
        s.submit(demand(2, 0, 512), Instant::now()).unwrap();
        let started = start_all(&mut s, Instant::now());
        assert_eq!(started[1].copy, 1);
        s.chunk_done(&started[0], ChunkResult::Bytes(512), Instant::now());
        s.chunk_done(&started[1], ChunkResult::Errno(libc::EIO), Instant::now());
        assert!(s.take_done(1).unwrap().result.is_ok());
        match s.take_done(2).unwrap().result {
            Err(Error::Io { path, .. }) => assert_eq!(path, PathBuf::from("model.bin.copy-on-1")),
            other => panic!("unexpected {other:?}"),
        }
        // The failed chunk counts for its drive, with no bytes.
        let stats = s.drive_stats();
        assert_eq!((stats[1].chunks, stats[1].bytes, stats[1].in_flight), (1, 0, 0));
    }

    #[test]
    fn drive_statistics_follow_the_chunks() {
        let mut s = sched_of(vec![file_on("model.bin", &[0, 1])], 4, 4);
        assert_eq!(s.drive_stats(), [DriveStats::default(); 2]);
        for key in 0..3 {
            s.submit(demand(key, 0, 2048), Instant::now()).unwrap();
        }
        let started = start_all(&mut s, Instant::now());
        let stats = s.drive_stats();
        assert_eq!((stats[0].in_flight, stats[1].in_flight), (2, 1));
        for p in &started {
            s.chunk_done(p, ChunkResult::Bytes(p.len), Instant::now());
        }
        let stats = s.drive_stats();
        assert_eq!((stats[0].chunks, stats[0].bytes, stats[0].peak_in_flight), (2, 4096, 2));
        assert_eq!((stats[1].chunks, stats[1].bytes, stats[1].peak_in_flight), (1, 2048, 1));
        assert_eq!((stats[0].in_flight, stats[1].in_flight), (0, 0));
    }
}
