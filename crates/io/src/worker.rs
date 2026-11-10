//! The I/O thread: the only owner of the ring.
//!
//! Its loop is: take every chunk the queues allow (under the lock, no system
//! call), push them to the submission queue, then enter the kernel once to
//! submit and sleep until at least one completion is there. New work from
//! the engine thread wakes it through an eventfd that is read by the ring
//! itself, so one `io_uring_enter` serves submission, completion and wake-up.

use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};
use std::sync::mpsc::Sender;
use std::time::Instant;

use io_uring::{IoUring, opcode, types};

use crate::error::Error;
use crate::reader::{FixedRegion, Shared};
use crate::sched::{ChunkResult, Picked};

/// `user_data` of the eventfd read; chunk slots are small indexes.
const WAKE: u64 = u64::MAX;
/// `user_data` of the cancel of the eventfd read at shutdown.
const CANCEL: u64 = u64::MAX - 1;
/// How often a chunk is resubmitted after `EINTR` or `EAGAIN` before the
/// error is reported.
const MAX_RETRIES: u8 = 16;

struct Worker {
    shared: Arc<Shared>,
    ring: IoUring,
    /// The chunks at the device, indexed by their `user_data`.
    chunks: Vec<Option<Picked>>,
    retries: Vec<u8>,
    free_slots: Vec<u32>,
    picked: Vec<Picked>,
    cqes: Vec<(u64, i32)>,
    finished: Vec<(Picked, ChunkResult)>,
    /// Where the eventfd read lands. Owned here and freed only after the
    /// read is known to be finished.
    wake_buf: *mut u64,
    wake_armed: bool,
    regions: Vec<FixedRegion>,
}

/// Marks the reader failed if the thread dies by panic, so that nobody waits
/// for completions that cannot come.
struct PanicGuard(Arc<Shared>);

impl Drop for PanicGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.lock().fail("the I/O thread panicked".into(), Instant::now());
        }
    }
}

pub(crate) fn run(
    shared: Arc<Shared>,
    depth: usize,
    regions: Vec<FixedRegion>,
    ready: Sender<Result<(), Error>>,
) {
    let _guard = PanicGuard(Arc::clone(&shared));
    let mut worker = match Worker::new(shared, depth, regions) {
        Ok(worker) => {
            let _ = ready.send(Ok(()));
            worker
        }
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    worker.run();
}

impl Worker {
    fn new(shared: Arc<Shared>, depth: usize, regions: Vec<FixedRegion>) -> Result<Self, Error> {
        // Room for every chunk, the eventfd read and its cancel.
        let entries = (depth + 2).next_power_of_two() as u32;
        // One thread submits and reaps (SINGLE_ISSUER), and completions are
        // posted when that thread next enters the kernel instead of
        // interrupting it (COOP_TASKRUN): it enters after every batch anyway.
        let ring = IoUring::builder()
            .setup_single_issuer()
            .setup_coop_taskrun()
            .build(entries)
            .map_err(Error::Ring)?;
        let fds: Vec<_> = shared.fds.iter().map(|f| f.as_raw_fd()).collect();
        if !fds.is_empty() {
            ring.submitter().register_files(&fds).map_err(Error::Ring)?;
        }
        if !regions.is_empty() {
            let iovecs: Vec<libc::iovec> = regions
                .iter()
                .map(|r| libc::iovec { iov_base: r.ptr.cast(), iov_len: r.len })
                .collect();
            // SAFETY: `FixedRegion::new` obliges its caller to keep the memory
            // valid for the life of the reader; the worker (and with it the
            // ring) ends before the reader's drop returns.
            unsafe { ring.submitter().register_buffers(&iovecs) }.map_err(Error::Ring)?;
        }
        Ok(Self {
            shared,
            ring,
            chunks: vec![None; depth],
            retries: vec![0; depth],
            free_slots: (0..depth as u32).rev().collect(),
            picked: Vec::with_capacity(depth),
            cqes: Vec::with_capacity(entries as usize * 2),
            finished: Vec::with_capacity(depth),
            wake_buf: Box::into_raw(Box::new(0u64)),
            wake_armed: false,
            regions,
        })
    }

    fn run(&mut self) {
        loop {
            // Announce the sleep before looking at the queues for the last
            // time (see `Shared::wake_if_sleeping`).
            self.shared.sleeping.store(true, Ordering::SeqCst);
            fence(Ordering::SeqCst);
            if self.start_chunks() {
                break;
            }
            let entered = self.ring.submit_and_wait(1);
            self.shared.sleeping.store(false, Ordering::SeqCst);
            if let Err(error) = entered {
                match error.raw_os_error() {
                    Some(libc::EINTR | libc::EAGAIN | libc::EBUSY) => {}
                    _ => return self.fatal(format!("io_uring_enter failed: {error}")),
                }
            }
            self.reap();
        }
        self.disarm_wake();
    }

    /// Moves what the queues allow into the submission queue. Returns true
    /// when the reader is shutting down and nothing is left at the device.
    fn start_chunks(&mut self) -> bool {
        self.picked.clear();
        {
            let mut sched = self.shared.lock();
            if sched.shutdown {
                sched.cancel_queued(Instant::now());
                if sched.in_flight() == 0 {
                    return true;
                }
            }
            let now = Instant::now();
            while self.picked.len() < self.free_slots.len() {
                let Some(chunk) = sched.take_chunk(now) else { break };
                self.picked.push(chunk);
            }
        }
        let mut full = false;
        {
            let mut sq = self.ring.submission();
            if !self.wake_armed {
                let read = opcode::Read::new(
                    types::Fd(self.shared.wake.as_raw_fd()),
                    self.wake_buf.cast(),
                    8,
                )
                .build()
                .user_data(WAKE);
                // SAFETY: `wake_buf` is 8 writable bytes owned by this worker
                // and stays allocated until the read is known to be finished
                // (`disarm_wake`); the eventfd is open as long as `shared` is.
                full |= unsafe { sq.push(&read) }.is_err();
                self.wake_armed = true;
            }
            for chunk in self.picked.drain(..) {
                let slot = self.free_slots.pop().expect("a free slot per chunk below the depth");
                self.chunks[slot as usize] = Some(chunk);
                self.retries[slot as usize] = 0;
                // SAFETY: the buffer behind `chunk.ptr` is held by the
                // scheduler until every chunk of its read has come back (a
                // read finishes only with `in_flight == 0`, and cancelling
                // only stops new chunks), and `chunk.len` bytes from `ptr`
                // lie inside it.
                full |= unsafe { sq.push(&read_entry(&self.regions, &chunk, slot)) }.is_err();
            }
        }
        if full {
            self.fatal("the submission queue is full".into());
            return true;
        }
        false
    }

    /// Takes the finished chunks from the completion queue and reports them.
    fn reap(&mut self) {
        let mut cqes = std::mem::take(&mut self.cqes);
        cqes.clear();
        {
            let mut cq = self.ring.completion();
            cq.sync();
            cqes.extend(cq.map(|cqe| (cqe.user_data(), cqe.result())));
        }
        let now = Instant::now();
        let mut finished = std::mem::take(&mut self.finished);
        for &(user_data, res) in &cqes {
            match user_data {
                WAKE => self.wake_armed = false,
                CANCEL => {}
                slot => {
                    let chunk = self.chunks[slot as usize].expect("a completion of a live chunk");
                    if res < 0 && self.should_retry(-res, slot) {
                        self.retries[slot as usize] += 1;
                        let mut sq = self.ring.submission();
                        // SAFETY: as in `start_chunks`: the same chunk, still
                        // held by the scheduler.
                        if unsafe { sq.push(&read_entry(&self.regions, &chunk, slot as u32)) }
                            .is_err()
                        {
                            drop(sq);
                            return self.fatal("the submission queue is full".into());
                        }
                        continue;
                    }
                    self.chunks[slot as usize] = None;
                    self.free_slots.push(slot as u32);
                    let result = if res < 0 {
                        ChunkResult::Errno(-res)
                    } else {
                        ChunkResult::Bytes(res as usize)
                    };
                    finished.push((chunk, result));
                }
            }
        }
        self.cqes = cqes;
        if !finished.is_empty() {
            let mut sched = self.shared.lock();
            for (chunk, result) in finished.drain(..) {
                sched.chunk_done(&chunk, result, now);
            }
        }
        self.finished = finished;
    }

    fn should_retry(&self, errno: i32, slot: u64) -> bool {
        matches!(errno, libc::EINTR | libc::EAGAIN) && self.retries[slot as usize] < MAX_RETRIES
    }

    /// At shutdown: cancels the eventfd read and waits for it to end, so that
    /// nothing writes into `wake_buf` after it is freed.
    fn disarm_wake(&mut self) {
        if self.wake_armed {
            let cancel = opcode::AsyncCancel::new(WAKE).build().user_data(CANCEL);
            // SAFETY: a cancel entry holds no pointers.
            if unsafe { self.ring.submission().push(&cancel) }.is_err() {
                return; // leak the 8 bytes rather than risk a write into freed memory
            }
            while self.wake_armed {
                if self.ring.submit_and_wait(1).is_err() {
                    return;
                }
                self.reap();
            }
        }
        // SAFETY: allocated by `Box::into_raw` in `new` and freed only here,
        // after the read that wrote into it has ended.
        drop(unsafe { Box::from_raw(self.wake_buf) });
        self.wake_buf = std::ptr::null_mut();
    }

    /// The ring cannot be used any more. Queued reads fail with the reason;
    /// chunks at the device are abandoned with their buffers leaked.
    fn fatal(&mut self, why: String) {
        self.shared.lock().fail(why, Instant::now());
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Free the eventfd buffer unless a read into it may still be pending
        // (after a fatal error); then 8 bytes are leaked instead.
        if !self.wake_buf.is_null() && !self.wake_armed {
            // SAFETY: allocated by `Box::into_raw` in `new`; no read targets
            // it, and `disarm_wake` nulls the pointer when it frees it.
            drop(unsafe { Box::from_raw(self.wake_buf) });
        }
    }
}

fn read_entry(regions: &[FixedRegion], chunk: &Picked, slot: u32) -> io_uring::squeue::Entry {
    let file = types::Fixed(u32::from(chunk.fd));
    let addr = chunk.ptr as usize;
    let fixed = regions
        .iter()
        .position(|r| addr >= r.ptr as usize && addr + chunk.len <= r.ptr as usize + r.len);
    let entry = match fixed {
        Some(index) => opcode::ReadFixed::new(file, chunk.ptr, chunk.len as u32, index as u16)
            .offset(chunk.offset)
            .build(),
        None => opcode::Read::new(file, chunk.ptr, chunk.len as u32).offset(chunk.offset).build(),
    };
    entry.user_data(u64::from(slot))
}
