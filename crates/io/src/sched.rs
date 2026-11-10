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
use std::time::Instant;

use crate::buffer::Buffer;
use crate::error::Error;
use crate::file::FileInfo;
use crate::request::{Cancel, Class, Completion, Key, Promote, Read, Rejected};
use crate::stats::{ClassStats, LatencyRing, Stats, summarize};

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
    pub depth: usize,
    pub prefetch_depth: usize,
    pub prefetch_chunk: usize,
    pub demand_chunk: usize,
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
    pub file: u16,
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
    slab: Vec<Option<Req>>,
    free: Vec<u32>,
    keys: IdMap<KeyState>,
    demand: VecDeque<u32>,
    prefetch: VecDeque<u32>,
    pub(crate) done: VecDeque<Completion>,
    in_flight: usize,
    peak_in_flight: usize,
    in_flight_prefetch: usize,
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
    pub(crate) fn new(limits: Limits, files: Arc<[FileInfo]>, trace: bool) -> Self {
        Self {
            limits,
            files,
            slab: Vec::new(),
            free: Vec::new(),
            keys: IdMap::default(),
            demand: VecDeque::new(),
            prefetch: VecDeque::new(),
            done: VecDeque::new(),
            in_flight: 0,
            peak_in_flight: 0,
            in_flight_prefetch: 0,
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
    pub(crate) fn startable(&self) -> bool {
        self.held == 0
            && self.in_flight < self.limits.depth
            && (!self.demand.is_empty()
                || (self.in_flight_prefetch < self.limits.prefetch_depth
                    && !self.prefetch.is_empty()))
    }

    /// The next chunk to start: the oldest demand read's next piece, else
    /// the oldest prefetch's, within the depth limits.
    pub(crate) fn take_chunk(&mut self) -> Option<Picked> {
        if !self.startable() {
            return None;
        }
        let (idx, class) = match self.demand.front() {
            Some(idx) => (*idx, Class::Demand),
            None => (*self.prefetch.front().expect("startable"), Class::Prefetch),
        };
        let req = self.req(idx);
        let rel = req.next;
        let len = req.chunk.min(req.total - rel);
        req.next += len;
        req.in_flight += 1;
        let finished_queueing = req.next == req.total;
        let key = req.key;
        let chunk_index = rel / req.chunk;
        let offset = req.start + rel as u64;
        let file = req.file;
        let buffer = req.buffer.as_mut().expect("a queued request holds its buffer");
        // SAFETY: `rel + len <= total <= buffer.len()` (checked at submit),
        // so the pointer stays inside the buffer.
        let ptr = unsafe { buffer.as_mut_ptr().add(rel) };
        if finished_queueing {
            self.queue_mut(class).pop_front();
        }
        self.in_flight += 1;
        self.peak_in_flight = self.peak_in_flight.max(self.in_flight);
        if class == Class::Prefetch {
            self.in_flight_prefetch += 1;
        }
        if let Some(trace) = &mut self.trace {
            trace.push((key, chunk_index));
        }
        Some(Picked { req: idx, rel, len, class, file, offset, ptr })
    }

    /// A chunk came back.
    pub(crate) fn chunk_done(&mut self, chunk: &Picked, result: ChunkResult, now: Instant) {
        self.in_flight -= 1;
        if chunk.class == Class::Prefetch {
            self.in_flight_prefetch -= 1;
        }
        let idx = chunk.req;
        // Field-wise borrows: the path is only cloned when an error needs it.
        let req = self.slab[idx as usize].as_mut().expect("an indexed request exists");
        req.in_flight -= 1;
        let stop = match result {
            ChunkResult::Errno(errno) => {
                if req.error.is_none() {
                    req.error = Some(Error::Io {
                        path: self.files[usize::from(req.file)].path.clone(),
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
        self.in_flight_prefetch = 0;
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
    use crate::file::{BufferedReason, FileId, IoMode};
    use std::path::PathBuf;

    const BLOCK: u64 = 512;

    fn sched(depth: usize, prefetch_depth: usize) -> Sched {
        let files: Arc<[FileInfo]> = Arc::new([FileInfo {
            path: PathBuf::from("test.bin"),
            size: 1 << 20,
            mode: IoMode::Buffered(BufferedReason::Disabled),
            direct_align: None,
            block: BLOCK,
        }]);
        let limits = Limits { depth, prefetch_depth, prefetch_chunk: 1024, demand_chunk: 1 << 20 };
        Sched::new(limits, files, true)
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
        while let Some(p) = s.take_chunk() {
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
        let a = s.take_chunk().unwrap();
        let b = s.take_chunk().unwrap();
        assert!(s.take_chunk().is_none(), "only two prefetch chunks may be at the device");
        // A demand read still starts at once, though prefetch is at its limit.
        s.submit(read(2, Class::Demand, 0, 512), Instant::now()).unwrap();
        assert!(s.startable());
        let d = s.take_chunk().unwrap();
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
        assert!(s.take_chunk().is_some());
        assert!(s.take_chunk().is_some());
        assert!(s.take_chunk().is_none());
        assert_eq!(s.in_flight(), 2);
    }

    #[test]
    fn promotion_moves_the_remaining_chunks_ahead_of_other_prefetches() {
        let mut s = sched(1, 1);
        s.submit(read(1, Class::Prefetch, 0, 2048), Instant::now()).unwrap();
        s.submit(read(2, Class::Prefetch, 0, 2048), Instant::now()).unwrap();
        s.submit(read(3, Class::Demand, 0, 512), Instant::now()).unwrap();
        // One chunk of the first prefetch goes out before anything is decided.
        let first = s.take_chunk().unwrap();
        assert_eq!(s.req(first.req).key, 3, "the demand read is first");
        s.chunk_done(&first, ChunkResult::Bytes(first.len), Instant::now());
        let second = s.take_chunk().unwrap();
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
        let a = s.take_chunk().unwrap();
        let b = s.take_chunk().unwrap();
        assert_eq!(s.cancel(7, Instant::now()).unwrap(), Cancel::Cancelled);
        assert!(s.take_done(7).is_none(), "two chunks are still at the device");
        assert!(s.take_chunk().is_none(), "the rest of the read is not started");
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
        let a = s.take_chunk().unwrap();
        let b = s.take_chunk().unwrap();
        s.chunk_done(&a, ChunkResult::Bytes(a.len), Instant::now());
        s.chunk_done(&b, ChunkResult::Bytes(1500 - a.len), Instant::now());
        assert!(s.take_chunk().is_none(), "nothing after the end of the file is read");
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
        let p = s.take_chunk().unwrap();
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
        let a = s.take_chunk().unwrap();
        let b = s.take_chunk().unwrap();
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
        let running = s.take_chunk().unwrap();
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
        let (a, b) = (s.take_chunk().unwrap(), s.take_chunk().unwrap());
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
}
