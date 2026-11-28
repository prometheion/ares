//! The reader: the handle the engine's step loop holds.

use std::ops::{Deref, DerefMut};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ares_core::units::{KIB, MIB};
use ares_gguf::Gguf;

use crate::error::{Error, Result};
use crate::file::{self, CopyInfo, DiskKey, DriveId, DriveInfo, FileId, FileInfo, IoMode, Span};
use crate::request::{Cancel, Completion, Key, Promote, Read, Rejected};
use crate::sched::{Limits, Sched, finish_stats};
use crate::stats::{DriveStats, Stats};
use crate::worker;

/// How the reader is set up. The defaults are what the engine uses.
#[derive(Clone, Debug)]
pub struct Config {
    /// The most chunks at one drive at once, both classes together. On the
    /// development machine's NVMe drive, random 512 KiB reads reach their
    /// throughput at 32, 64 gains nothing and 16 gives about 60 % (the block
    /// layer splits each read into two requests); more only lengthens the
    /// wait of a demand read that arrives behind them. Every drive the files
    /// are on has this depth of its own.
    pub queue_depth: usize,
    /// Of those, the most that may come from prefetches, again per drive,
    /// and half of it (rounded up) while a demand chunk is at the drive.
    /// This is what bounds the wait of a demand read behind guesses to
    /// `prefetch_depth * prefetch_chunk` bytes of device time, and what a
    /// demand read that is at the drive shares it with to half of that
    /// (Run My AI's rule: four chunks of 512 KiB, two beside a demand read).
    pub prefetch_depth: usize,
    /// Size of the pieces a prefetch is cut into (rounded up to the file's
    /// block size).
    pub prefetch_chunk: usize,
    /// Demand reads are cut only when they are longer than this.
    pub demand_chunk: usize,
    /// Read with `O_DIRECT` where the filesystem allows it. `false` reads
    /// every file through the page cache; it is for measuring the difference
    /// and for tests.
    pub direct: bool,
    /// Keep a list of the order in which chunks were handed to the ring,
    /// readable with [`Reader::started`]. It grows without bound, so it is
    /// for tests and diagnosis, not for a running engine.
    pub record_starts: bool,
    /// Memory registered with the ring at start. Reads into a [`Buffer`](crate::Buffer)
    /// that lies inside one of the regions skip the kernel's per-read pinning
    /// of the pages. The pinned pages count against the limit of locked
    /// memory (`ulimit -l`; 8 MiB by default), so this suits a few staging
    /// buffers, not a large pool, unless the limit is raised.
    pub fixed_regions: Vec<FixedRegion>,
}

/// A range of memory to register with the ring, see [`Config::fixed_regions`].
#[derive(Clone, Copy, Debug)]
pub struct FixedRegion {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

// SAFETY: a region is an address range; `FixedRegion::new` makes its owner
// responsible for the memory behind it, wherever that is used.
unsafe impl Send for FixedRegion {}
// SAFETY: as above; the type has no interior mutability.
unsafe impl Sync for FixedRegion {}

impl FixedRegion {
    /// # Safety
    ///
    /// The memory must stay allocated, and mapped at the same address, until
    /// the [`Reader`] the region is given to has been dropped.
    pub unsafe fn new(ptr: *mut u8, len: usize) -> Self {
        Self { ptr, len }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            queue_depth: 32,
            prefetch_depth: 4,
            prefetch_chunk: (512 * KIB) as usize,
            demand_chunk: (4 * MIB) as usize,
            direct: true,
            record_starts: false,
            fixed_regions: Vec::new(),
        }
    }
}

impl Config {
    fn validate(&self) -> Result<()> {
        const MAX_CHUNK: usize = 1 << 30;
        if self.queue_depth == 0 || self.queue_depth > 4096 {
            return Err(Error::Config("queue_depth must be between 1 and 4096"));
        }
        if self.prefetch_depth == 0 || self.prefetch_depth > self.queue_depth {
            return Err(Error::Config("prefetch_depth must be between 1 and queue_depth"));
        }
        if !(1..=MAX_CHUNK).contains(&self.prefetch_chunk)
            || !(1..=MAX_CHUNK).contains(&self.demand_chunk)
        {
            return Err(Error::Config("chunk sizes must be between 1 byte and 1 GiB"));
        }
        Ok(())
    }
}

pub(crate) struct Shared {
    sched: Mutex<Sched>,
    done: Condvar,
    /// Completions waiting to be taken. `poll` reads this and returns without
    /// a lock or a system call when it is zero.
    done_count: AtomicUsize,
    /// Set while the I/O thread is (about to be) blocked in the kernel; a
    /// caller that queued work then writes the eventfd to wake it.
    pub(crate) sleeping: AtomicBool,
    pub(crate) wake: OwnedFd,
    /// Every copy of every file, open: the copies of file 0 in their order,
    /// then those of file 1, and so on, which is how the queues number them.
    pub(crate) fds: Vec<std::fs::File>,
    pub(crate) infos: Arc<[FileInfo]>,
    pub(crate) drives: Vec<DriveInfo>,
}

/// The scheduler locked. When it is released the number of finished reads is
/// published for `poll`, and waiters are woken if there is news.
pub(crate) struct Locked<'a> {
    guard: MutexGuard<'a, Sched>,
    shared: &'a Shared,
}

impl Deref for Locked<'_> {
    type Target = Sched;
    fn deref(&self) -> &Sched {
        &self.guard
    }
}

impl DerefMut for Locked<'_> {
    fn deref_mut(&mut self) -> &mut Sched {
        &mut self.guard
    }
}

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        self.shared.publish(&mut self.guard);
    }
}

impl Shared {
    pub(crate) fn lock(&self) -> Locked<'_> {
        Locked { guard: self.sched.lock().unwrap_or_else(PoisonError::into_inner), shared: self }
    }

    fn publish(&self, sched: &mut Sched) {
        self.done_count.store(sched.done.len(), Ordering::Release);
        if std::mem::take(&mut sched.news) && sched.waiters > 0 {
            self.done.notify_all();
        }
    }

    /// Wakes the I/O thread if it is asleep. Called after queueing work.
    ///
    /// The thread announces that it is going to sleep (`sleeping`), then looks
    /// at the queues one last time; the caller queues, then looks at
    /// `sleeping`. With a full fence on both sides one of the two must see the
    /// other, so work is never left waiting for a thread that sleeps.
    pub(crate) fn wake_if_sleeping(&self) {
        fence(Ordering::SeqCst);
        if self.sleeping.load(Ordering::SeqCst) {
            self.wake();
        }
    }

    pub(crate) fn wake(&self) {
        let one = 1u64.to_ne_bytes();
        // SAFETY: `wake` is a valid eventfd and `one` is 8 readable bytes. A
        // failed write (the counter is full) means a wake-up is pending
        // already, so the result is ignored.
        let _ = unsafe { libc::write(self.wake.as_raw_fd(), one.as_ptr().cast(), one.len()) };
    }
}

/// The SSD reader: queues reads, runs them on one I/O thread through
/// `io_uring` in priority order, and hands the finished ones back.
///
/// It is meant to be used from the engine's step thread, but every method
/// takes `&self` and may be called from any thread.
///
/// # Ordering
///
/// Demand reads start before prefetches, always, and each class is served in
/// submission order. A prefetch is cut into chunks (see
/// [`Config::prefetch_chunk`]); a demand read that arrives while a long
/// prefetch runs goes next, whatever is left of the prefetch. At most
/// [`Config::queue_depth`] chunks are at a drive, at most
/// [`Config::prefetch_depth`] of them from prefetches.
///
/// # Several drives
///
/// The files may be on different drives, and a file may have copies with the
/// same bytes on several ([`Reader::open_copies`],
/// [`Reader::open_gguf_mirrored`]). Every drive has its own depth, so two
/// drives carry twice the chunks of one. A chunk of a file with copies goes
/// to the drive that is expected to deliver it soonest: what is at the drive
/// already, at the pace the drive has shown lately, so a faster drive gets
/// more and an idle one is used first. A read whose drives are all busy does
/// not hold up a later one that an idle drive can serve; within what a drive
/// can serve, the order above holds.
///
/// # Completion
///
/// Finished reads are taken with [`poll`](Reader::poll),
/// [`wait`](Reader::wait) or [`wait_any`](Reader::wait_any). A key stays
/// reserved until its completion is taken.
///
/// Dropping the reader cancels what has not started, waits for the chunks at
/// the device (the kernel is writing into the buffers until they return),
/// and joins the I/O thread.
#[derive(Debug)]
pub struct Reader {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    /// Bytes of prefetches one drive takes at once.
    prefetch_bytes: usize,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared").field("files", &self.infos.len()).finish_non_exhaustive()
    }
}

impl Reader {
    /// Opens `paths` and starts the I/O thread. The files are numbered in
    /// order ([`FileId`]).
    pub fn open<P: AsRef<Path>>(paths: &[P], config: Config) -> Result<Self> {
        let files: Vec<[&Path; 1]> = paths.iter().map(|path| [path.as_ref()]).collect();
        Self::open_copies(&files, config)
    }

    /// Opens files of which each may have several copies, and starts the I/O
    /// thread. `files[n]` lists the paths of file `n` ([`FileId`]): one or more
    /// files with the same bytes, usually on different drives. Reads of a file
    /// are spread over the drives of its copies (see "Several drives" above),
    /// so a model that is mirrored on a second SSD is read at the speed of
    /// both.
    ///
    /// The copies of a file must be identical. Their sizes are compared and
    /// their contents sampled at open ([`Error::CopySize`],
    /// [`Error::CopyDiffers`]); that catches a wrong or damaged copy, it is not
    /// a proof.
    pub fn open_copies<F, P>(files: &[F], config: Config) -> Result<Self>
    where
        F: AsRef<[P]>,
        P: AsRef<Path>,
    {
        config.validate()?;
        let mut disks: Vec<DiskKey> = Vec::new();
        let mut fds = Vec::new();
        let mut infos = Vec::with_capacity(files.len());
        for (index, paths) in files.iter().enumerate() {
            let paths = paths.as_ref();
            if paths.is_empty() {
                return Err(Error::NoCopies(index));
            }
            let opened = paths
                .iter()
                .map(|path| file::open(path.as_ref(), config.direct))
                .collect::<Result<Vec<_>>>()?;
            file::check_copies(&opened)?;
            let copies = opened
                .iter()
                .map(|copy| {
                    let drive = disks.iter().position(|d| *d == copy.disk).unwrap_or_else(|| {
                        disks.push(copy.disk);
                        disks.len() - 1
                    });
                    CopyInfo {
                        path: copy.path.clone(),
                        mode: copy.mode,
                        direct_align: copy.direct_align,
                        drive: DriveId(drive as u16),
                    }
                })
                .collect();
            let first = &opened[0];
            infos.push(FileInfo {
                path: first.path.clone(),
                size: first.size,
                mode: opened
                    .iter()
                    .map(|copy| copy.mode)
                    .find(|mode| *mode != IoMode::Direct)
                    .unwrap_or(IoMode::Direct),
                direct_align: first.direct_align,
                block: opened.iter().map(|copy| copy.block).max().unwrap_or(1),
                copies,
            });
            fds.extend(opened.into_iter().map(|copy| copy.file));
        }
        if infos.len() > usize::from(u16::MAX) || fds.len() > usize::from(u16::MAX) {
            return Err(Error::Config("at most 65535 files, copies included"));
        }
        let infos: Arc<[FileInfo]> = infos.into();
        let drives: Vec<DriveInfo> = disks.iter().map(|disk| disk.info()).collect();

        // SAFETY: eventfd takes plain integers and returns a new descriptor
        // or -1.
        let raw = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
        if raw < 0 {
            return Err(Error::Ring(std::io::Error::last_os_error()));
        }
        // SAFETY: `raw` is a fresh descriptor that nothing else owns.
        let wake = unsafe { OwnedFd::from_raw_fd(raw) };

        let limits = Limits {
            depth: config.queue_depth,
            prefetch_depth: config.prefetch_depth,
            prefetch_chunk: config.prefetch_chunk,
            demand_chunk: config.demand_chunk,
        };
        // Every drive has the depth of its own, and the ring room for all.
        let depth = config.queue_depth * drives.len().max(1);
        let shared = Arc::new(Shared {
            sched: Mutex::new(Sched::new(limits, Arc::clone(&infos), config.record_starts)),
            done: Condvar::new(),
            done_count: AtomicUsize::new(0),
            sleeping: AtomicBool::new(false),
            wake,
            fds,
            infos,
            drives,
        });

        let (ready_tx, ready_rx) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let regions = config.fixed_regions.clone();
        let thread = std::thread::Builder::new()
            .name("ares-io".into())
            .spawn(move || worker::run(worker_shared, depth, regions, ready_tx))
            .map_err(Error::Thread)?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                shared,
                thread: Some(thread),
                prefetch_bytes: config.prefetch_depth * config.prefetch_chunk,
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => {
                let _ = thread.join();
                Err(Error::Stopped("the I/O thread ended before it was ready".into()))
            }
        }
    }

    /// Opens every part of a GGUF model. [`FileId`] `n` is part `n`, which is
    /// what `TensorInfo::part` holds, so tensors can be read without a lookup.
    pub fn open_gguf(gguf: &Gguf, config: Config) -> Result<Self> {
        Self::open_gguf_mirrored(gguf, &[] as &[&Path], config)
    }

    /// Opens every part of a GGUF model together with its mirrors: copies of
    /// the model on other drives, which the reads are spread over.
    ///
    /// A mirror is a directory that holds a copy of every part under the same
    /// file name. For a model of one file it may also be the path of the copy
    /// itself. [`FileId`] `n` is part `n`, as for [`Reader::open_gguf`].
    pub fn open_gguf_mirrored<P: AsRef<Path>>(
        gguf: &Gguf,
        mirrors: &[P],
        config: Config,
    ) -> Result<Self> {
        let n_parts = gguf.n_parts();
        let mut files = Vec::with_capacity(n_parts);
        for part in 0..n_parts {
            let path = gguf.path(part as u16);
            let mut copies = vec![path.to_path_buf()];
            for mirror in mirrors {
                let mirror = mirror.as_ref();
                let refuse = |reason| Error::Mirror { mirror: mirror.to_owned(), reason };
                if mirror.is_dir() {
                    let name =
                        path.file_name().ok_or_else(|| refuse("the model has no file name"))?;
                    copies.push(mirror.join(name));
                } else if n_parts == 1 {
                    copies.push(mirror.to_owned());
                } else {
                    return Err(refuse(
                        "a model of several files needs a directory that holds a copy of every \
                         part under the same file name",
                    ));
                }
            }
            files.push(copies);
        }
        Self::open_copies(&files, config)
    }

    /// The drives the files are on, in the order they were met at open;
    /// [`CopyInfo::drive`] indexes this list. Two partitions or filesystems of
    /// one disk are one drive.
    pub fn drives(&self) -> &[DriveInfo] {
        &self.shared.drives
    }

    /// Bytes of prefetches the drives take at once, over all drives: what a reader of
    /// guesses can have at the drives without holding up demand reads more than
    /// [`Config::prefetch_depth`] says.
    pub fn prefetch_room(&self) -> usize {
        self.prefetch_bytes * self.shared.drives.len().max(1)
    }

    /// What each drive has delivered and is doing, in the order of
    /// [`Reader::drives`].
    pub fn drive_stats(&self) -> Vec<DriveStats> {
        self.shared.lock().drive_stats()
    }

    /// The files and how each is read. A file that could not be read with
    /// direct I/O says so in [`FileInfo::mode`]; this is the only place the
    /// fallback is reported.
    pub fn files(&self) -> &[FileInfo] {
        &self.shared.infos
    }

    /// Whether every file is read with direct I/O.
    pub fn all_direct(&self) -> bool {
        self.shared.infos.iter().all(|f| f.mode == crate::IoMode::Direct)
    }

    /// The alignment a buffer needs to serve a read of any file of this
    /// reader (the largest block size; 1 if everything is read buffered).
    pub fn buffer_alignment(&self) -> usize {
        self.shared.infos.iter().map(|f| f.block).max().unwrap_or(1) as usize
    }

    /// How a read of `len` bytes at `offset` of `file` is widened: the buffer
    /// must hold [`Span::len`] bytes, and the wanted data will be at
    /// [`Span::skip`] in it.
    pub fn span(&self, file: FileId, offset: u64, len: usize) -> Result<Span> {
        file::span_in(&self.shared.infos, file, offset, len)
    }

    /// Queues a read. It returns at once; the finished read is taken with
    /// [`poll`](Reader::poll) or [`wait`](Reader::wait). A read that is
    /// refused (wrong buffer, key in use, reader stopped) comes back whole in
    /// the error.
    ///
    /// A range that runs past the end of the file is not refused here: the
    /// read fails alone, on its own completion, like any other I/O error.
    pub fn submit(&self, read: Read) -> std::result::Result<(), Box<Rejected>> {
        let now = Instant::now();
        self.shared.lock().submit(read, now)?;
        self.shared.wake_if_sleeping();
        Ok(())
    }

    /// Stops a read that has not finished. Chunks not yet started are never
    /// started; the completion comes when the chunks already at the device
    /// are back, with [`Error::Cancelled`] and the buffer.
    pub fn cancel(&self, key: Key) -> Result<Cancel> {
        self.shared.lock().cancel(key, Instant::now())
    }

    /// Turns a prefetch into a demand read: its remaining chunks leave the
    /// prefetch queue and go in the demand queue, behind demand reads that
    /// were already waiting and ahead of every prefetch.
    pub fn promote(&self, key: Key) -> Result<Promote> {
        let outcome = self.shared.lock().promote(key)?;
        self.shared.wake_if_sleeping();
        Ok(outcome)
    }

    /// Appends all finished reads to `out` and returns how many there were.
    ///
    /// When nothing has finished this is one atomic load: no lock and no
    /// system call, so the step loop can call it between kernels.
    pub fn poll(&self, out: &mut Vec<Completion>) -> usize {
        if self.shared.done_count.load(Ordering::Acquire) == 0 {
            return 0;
        }
        self.shared.lock().take_all_done(out)
    }

    /// Waits up to `timeout` for the read `key` and takes its completion.
    /// `Ok(None)` means it did not finish in time; it is still queued.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownKey`] when no read with this key is outstanding.
    pub fn wait(&self, key: Key, timeout: Duration) -> Result<Option<Completion>> {
        let deadline = Instant::now().checked_add(timeout);
        let mut guard = self.shared.sched.lock().unwrap_or_else(PoisonError::into_inner);
        let result = loop {
            if !guard.is_known(key) {
                break Err(Error::UnknownKey(key));
            }
            if guard.is_done(key) {
                break Ok(guard.take_done(key));
            }
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => Duration::from_secs(3600),
            };
            if remaining.is_zero() {
                break Ok(None);
            }
            guard.waiters += 1;
            guard = self
                .shared
                .done
                .wait_timeout(guard, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
            guard.waiters -= 1;
        };
        self.shared.publish(&mut guard);
        result
    }

    /// Waits up to `timeout` for any read to finish and takes its completion.
    pub fn wait_any(&self, timeout: Duration) -> Option<Completion> {
        let deadline = Instant::now().checked_add(timeout);
        let mut guard = self.shared.sched.lock().unwrap_or_else(PoisonError::into_inner);
        let result = loop {
            if let Some(completion) = guard.take_next_done() {
                break Some(completion);
            }
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => Duration::from_secs(3600),
            };
            if remaining.is_zero() {
                break None;
            }
            guard.waiters += 1;
            guard = self
                .shared
                .done
                .wait_timeout(guard, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
            guard.waiters -= 1;
        };
        self.shared.publish(&mut guard);
        result
    }

    /// The counters, at this moment.
    pub fn stats(&self) -> Stats {
        let snapshot = self.shared.lock().stats();
        finish_stats(snapshot)
    }

    /// Stops the I/O thread from starting new chunks while the returned guard
    /// lives. Chunks at the device finish and everything else waits, so the
    /// queues can be filled to a known state and then released. It exists for
    /// tests of the order rules and for measurements.
    pub fn hold(&self) -> Hold<'_> {
        self.shared.lock().held += 1;
        Hold { reader: self }
    }

    /// The chunks handed to the ring so far, in order, as (key, index of the
    /// chunk within its read). Empty unless [`Config::record_starts`] is on.
    pub fn started(&self) -> Vec<(Key, usize)> {
        self.shared.lock().trace()
    }

    /// Binds the I/O thread to the logical CPUs `cpus`; none leaves it where
    /// the scheduler puts it. The I/O thread wakes for every completion and
    /// works for microseconds: the engine keeps it on the cores its CPU
    /// workers leave free, so that it does not wait for a core behind a
    /// worker that spins.
    pub fn run_on(&self, cpus: &[usize]) -> Result<()> {
        use std::os::unix::thread::JoinHandleExt;
        let Some(thread) = self.thread.as_ref().filter(|_| !cpus.is_empty()) else {
            return Ok(());
        };
        // SAFETY: an all-zero set is a valid `cpu_set_t`; only CPUs below its
        // size are set, and the call reads the set only. The thread is alive:
        // it is joined only when the reader is dropped.
        let result = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &cpu in cpus.iter().filter(|cpu| **cpu < libc::CPU_SETSIZE as usize) {
                libc::CPU_SET(cpu, &mut set);
            }
            libc::pthread_setaffinity_np(thread.as_pthread_t(), size_of::<libc::cpu_set_t>(), &set)
        };
        match result {
            0 => Ok(()),
            error => Err(Error::Affinity(std::io::Error::from_raw_os_error(error))),
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// While it lives, no new chunk is started. See [`Reader::hold`].
#[derive(Debug)]
pub struct Hold<'a> {
    reader: &'a Reader,
}

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        self.reader.shared.lock().held -= 1;
        self.reader.shared.wake_if_sleeping();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Buffer, Class};

    /// A scratch directory that goes away even when the test fails.
    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn polling_with_nothing_finished_takes_no_lock_and_makes_no_call() {
        let dir =
            Scratch(std::env::temp_dir().join(format!("ares-io-poll-{}", std::process::id())));
        std::fs::create_dir_all(&dir.0).unwrap();
        let path = dir.0.join("data.bin");
        std::fs::write(&path, [7u8; 4096]).unwrap();
        let reader = Reader::open(&[&path], Config::default()).unwrap();

        // With the scheduler locked by this thread, a poll that tried to take
        // the lock would never return. One atomic load does. (The lock is
        // released after the wait either way, so a poll that does lock makes
        // the test fail instead of hanging it.)
        let guard = reader.shared.sched.lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let polled = std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut out = Vec::new();
                tx.send(reader.poll(&mut out)).unwrap();
            });
            let polled = rx.recv_timeout(Duration::from_secs(3));
            drop(guard);
            polled
        });
        assert_eq!(polled, Ok(0), "poll must not wait for the scheduler lock");

        // Once something has finished, the same call delivers it.
        let read = Read {
            key: 1,
            file: FileId(0),
            offset: 0,
            len: 100,
            class: Class::Demand,
            buffer: Buffer::aligned(
                reader.span(FileId(0), 0, 100).unwrap().len,
                reader.buffer_alignment(),
            ),
        };
        reader.submit(read).unwrap();
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while out.is_empty() && Instant::now() < deadline {
            reader.poll(&mut out);
            std::thread::yield_now();
        }
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data(), &[7u8; 100][..]);
    }

    #[test]
    fn the_io_thread_runs_on_the_cpus_it_is_given() {
        use std::os::unix::thread::JoinHandleExt;
        let dir =
            Scratch(std::env::temp_dir().join(format!("ares-io-cpus-{}", std::process::id())));
        std::fs::create_dir_all(&dir.0).unwrap();
        let path = dir.0.join("data.bin");
        std::fs::write(&path, [7u8; 4096]).unwrap();
        let reader = Reader::open(&[&path], Config::default()).unwrap();
        let allowed = |reader: &Reader| -> Vec<usize> {
            let thread = reader.thread.as_ref().unwrap().as_pthread_t();
            // SAFETY: an all-zero set is valid and the call writes at most its size.
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                assert_eq!(
                    libc::pthread_getaffinity_np(thread, size_of::<libc::cpu_set_t>(), &mut set),
                    0
                );
                (0..libc::CPU_SETSIZE as usize).filter(|cpu| libc::CPU_ISSET(*cpu, &set)).collect()
            }
        };
        let before = allowed(&reader);
        // None leaves it as it is.
        reader.run_on(&[]).unwrap();
        assert_eq!(allowed(&reader), before);
        let last = *before.last().unwrap();
        reader.run_on(&[last]).unwrap();
        assert_eq!(allowed(&reader), vec![last]);
    }
}
