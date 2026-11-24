//! The memory of the RAM tier: one range of addresses, page-locked in pieces.
//!
//! The tier's slots lie at fixed offsets in one range, so that a slot's
//! address never changes. The range is address space only: a piece of it
//! takes memory when it is locked and gives it back to the system when it is
//! released (unlocked, then `MADV_DONTNEED`), which is how the tier follows
//! the memory the machine has. Page-locked memory is not reclaimed by the
//! kernel, so nothing but this gives it back.
//!
//! A piece is locked once: a released piece keeps its lock until its
//! [`Released`] is dropped, and it is not locked again before. The driver
//! refuses to lock a range that a lock still covers
//! (`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`), and a release dropped after
//! a second lock would give back memory the tier uses again.

use std::any::Any;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ares_core::Ledger;

use super::StoreError;

/// The bytes of a piece, about: what is locked or released at a time (a
/// piece is whole slots). See "Host memory" in `docs/dev/architecture/host-memory.md` for the
/// times that chose it.
pub const PIECE: usize = 64 << 20;

/// The region starts on this boundary, which is that of a huge page.
const ALIGN: usize = 2 << 20;

/// Page-locks pieces of a region.
pub trait Locker: Send {
    /// Page-locks `len` bytes at `ptr`. What comes back unlocks them when it
    /// is dropped. A range that a lock still covers is not locked again (a
    /// device's lock fails then).
    ///
    /// # Safety
    ///
    /// The range must be mapped, start on a page and be whole pages, and
    /// stay mapped until the returned value has been dropped.
    unsafe fn lock(&mut self, ptr: *mut u8, len: usize) -> Result<Box<dyn Any + Send>, StoreError>;
}

/// Locks nothing: for tests and for a store whose device is not a GPU.
struct NoLock;

impl Locker for NoLock {
    unsafe fn lock(&mut self, _: *mut u8, _: usize) -> Result<Box<dyn Any + Send>, StoreError> {
        Ok(Box::new(()))
    }
}

/// Locks for a device, with the bytes reserved elsewhere.
struct DeviceLock(ares_cuda::Device);

impl Locker for DeviceLock {
    unsafe fn lock(&mut self, ptr: *mut u8, len: usize) -> Result<Box<dyn Any + Send>, StoreError> {
        // SAFETY: the caller's contract is that of the registration.
        Ok(Box::new(unsafe { ares_cuda::HostRegistration::new(&self.0, ptr, len, false)? }))
    }
}

/// Address space of the process, unmapped when the last user of it is gone.
struct Mapping {
    base: *mut libc::c_void,
    mapped: usize,
    ptr: NonNull<u8>,
}

// SAFETY: the mapping is plain memory; who may touch which part of it is the
// business of the region and the store.
unsafe impl Send for Mapping {}
// SAFETY: as above; the mapping itself is never changed after it is made.
unsafe impl Sync for Mapping {}

impl Mapping {
    fn new(len: usize) -> Result<Self, StoreError> {
        let mapped = len.next_multiple_of(ALIGN) + ALIGN;
        // SAFETY: an anonymous mapping of fresh address space.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(StoreError::Plan(format!(
                "no address space for a RAM tier of {len} bytes: {}",
                std::io::Error::last_os_error()
            )));
        }
        let start = (base as usize).next_multiple_of(ALIGN);
        let ptr = NonNull::new(start as *mut u8)
            .ok_or_else(|| StoreError::Plan("a mapping without an address".into()))?;
        Ok(Self { base, mapped, ptr })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the range is the one `mmap` returned, and every lock on it
        // and every user of it is gone: they hold the mapping.
        unsafe { libc::munmap(self.base, self.mapped) };
    }
}

/// A piece that the tier has given up: dropping it unlocks the memory and
/// returns it to the system. That takes time, so it is dropped where nothing
/// waits for it. Until then the piece is not locked again.
pub struct Released {
    lock: Option<Box<dyn Any + Send>>,
    map: Arc<Mapping>,
    /// The region's mark that the piece is on its way back: cleared once it
    /// is unlocked and returned.
    releasing: Arc<AtomicBool>,
    at: usize,
    len: usize,
}

impl Released {
    /// The bytes of the piece.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::fmt::Debug for Released {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Released({} bytes at {})", self.len, self.at)
    }
}

impl Drop for Released {
    fn drop(&mut self) {
        // Unlocked first: locked pages cannot be given back.
        drop(self.lock.take());
        // SAFETY: the range is whole pages inside the mapping, which this
        // holds, and no slot of the tier lies in the piece any more.
        unsafe {
            libc::madvise(self.map.ptr.as_ptr().add(self.at).cast(), self.len, libc::MADV_DONTNEED);
        }
        // The piece may be locked again; whoever sees this sees it unlocked.
        self.releasing.store(false, Ordering::Release);
    }
}

/// The memory of the RAM tier.
pub struct HostRegion {
    /// Per piece: what holds it locked. Dropped before the mapping.
    locks: Vec<Option<Box<dyn Any + Send>>>,
    /// Per piece: it was released and its [`Released`] is not dropped yet.
    releasing: Vec<Arc<AtomicBool>>,
    locker: Box<dyn Locker>,
    map: Arc<Mapping>,
    len: usize,
    /// Where every piece starts, and its bytes: one piece, the whole region,
    /// until the store says how its slots lie ([`HostRegion::cut`]).
    spans: Vec<(usize, usize)>,
    /// The bytes the store locks when it is built.
    start: usize,
}

// SAFETY: the region is memory that only the store uses, and the store is used
// by one thread at a time.
unsafe impl Send for HostRegion {}

impl std::fmt::Debug for HostRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostRegion({} bytes, {} locked)", self.len, self.locked_bytes())
    }
}

impl HostRegion {
    /// `len` bytes of address space, of which the store locks `start` bytes
    /// with `locker` when it is built, in the pieces its slots lie in, and
    /// more or fewer as its owner says later.
    pub fn elastic(len: usize, start: usize, locker: Box<dyn Locker>) -> Result<Self, StoreError> {
        let len = len.max(1).next_multiple_of(4096);
        let map = Arc::new(Mapping::new(len)?);
        Ok(Self {
            locks: vec![None],
            releasing: vec![Arc::default()],
            locker,
            map,
            len,
            spans: vec![(0, len)],
            start: start.min(len),
        })
    }

    /// `len` page-locked bytes, reserved in the ledger of pinned RAM.
    ///
    /// The memory is the process's own, locked in place for the device
    /// (`cuMemHostRegister`), not memory the driver allocates
    /// (`cuMemHostAlloc`): direct I/O cannot read into the driver's mapping,
    /// and the reads of the tier would fall back to the page cache.
    pub fn pinned(
        device: &ares_cuda::Device,
        ledger: &mut Ledger,
        len: usize,
    ) -> Result<Self, StoreError> {
        let len = len.max(1).next_multiple_of(4096);
        ledger.reserve("expert RAM tier", len as u64)?;
        Self::elastic(len, len, Box::new(DeviceLock(device.clone())))
    }

    /// `len` bytes of ordinary memory aligned to `align` (at most 2 MiB):
    /// for tests and for a store whose device is not a GPU.
    pub fn unpinned(len: usize, align: usize) -> Self {
        assert!(align <= ALIGN, "a region is aligned to 2 MiB");
        Self::elastic(len, len, Box::new(NoLock)).expect("address space for a region")
    }

    /// The same region, of which only `start` bytes are locked when the
    /// store is built: for tests of a tier that grows and shrinks.
    pub fn starting_with(mut self, start: usize) -> Self {
        self.start = start.min(self.len);
        self
    }

    /// Cuts the region into the pieces `spans` (where each starts and its
    /// bytes: whole pages that do not overlap), before anything is locked.
    pub(super) fn cut(&mut self, spans: &[(usize, usize)]) {
        debug_assert!(self.locks.iter().all(Option::is_none), "before anything is locked");
        debug_assert!(spans.iter().all(|(at, len)| at + len <= self.len));
        self.spans = spans.to_vec();
        self.locks = spans.iter().map(|_| None).collect();
        self.releasing = spans.iter().map(|_| Arc::default()).collect();
    }

    pub(super) fn ptr(&self) -> NonNull<u8> {
        self.map.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn n_pieces(&self) -> usize {
        self.spans.len()
    }

    pub(super) fn start(&self) -> usize {
        self.start
    }

    /// Where piece `index` starts in the region, and its bytes.
    pub fn span(&self, index: usize) -> (usize, usize) {
        self.spans[index]
    }

    pub fn is_locked(&self, index: usize) -> bool {
        self.locks[index].is_some()
    }

    /// Whether piece `index` was released and its [`Released`] is not
    /// dropped yet: its lock still covers it, so it cannot be locked again.
    pub fn is_releasing(&self, index: usize) -> bool {
        self.releasing[index].load(Ordering::Acquire)
    }

    /// Bytes that are locked now.
    pub fn locked_bytes(&self) -> usize {
        (0..self.locks.len()).filter(|i| self.is_locked(*i)).map(|i| self.span(i).1).sum()
    }

    /// The address of piece `index` and its bytes.
    pub fn piece_ptr(&self, index: usize) -> (*mut u8, usize) {
        let (at, len) = self.span(index);
        // SAFETY: the span is inside the mapping.
        (unsafe { self.map.ptr.as_ptr().add(at) }, len)
    }

    /// Locks piece `index` with the region's own locker. A piece whose
    /// release is not dropped yet is refused.
    pub(super) fn lock(&mut self, index: usize) -> Result<(), StoreError> {
        if self.is_locked(index) {
            return Ok(());
        }
        if self.is_releasing(index) {
            return Err(StoreError::Plan(format!(
                "piece {index} of the RAM tier would be locked again while its release still \
                 holds its lock"
            )));
        }
        let (ptr, len) = self.piece_ptr(index);
        // SAFETY: the piece is whole pages of the mapping, which outlives
        // the lock: the locks are dropped before the mapping, and a released
        // piece holds the mapping.
        self.locks[index] = Some(unsafe { self.locker.lock(ptr, len)? });
        Ok(())
    }

    /// Gives piece `index` up. Nothing of the tier may lie in it.
    pub(super) fn release(&mut self, index: usize) -> Option<Released> {
        let lock = self.locks[index].take()?;
        let (at, len) = self.span(index);
        let releasing = self.releasing[index].clone();
        releasing.store(true, Ordering::Release);
        Some(Released { lock: Some(lock), map: self.map.clone(), releasing, at, len })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes of `len` bytes at `ptr` that are in memory.
    fn resident(ptr: *mut u8, len: usize) -> usize {
        let mut pages = vec![0u8; len.div_ceil(4096)];
        // SAFETY: the range is mapped and `pages` has a byte per page.
        let failed = unsafe { libc::mincore(ptr.cast(), len, pages.as_mut_ptr()) } != 0;
        assert!(!failed, "mincore: {}", std::io::Error::last_os_error());
        pages.iter().filter(|page| **page & 1 == 1).count() * 4096
    }

    #[test]
    fn a_region_takes_memory_only_for_what_is_written_and_a_released_piece_gives_it_back() {
        const MIB: usize = 1 << 20;
        let mut region = HostRegion::unpinned(64 * MIB, 4096).starting_with(16 * MIB);
        let spans: Vec<(usize, usize)> = (0..8).map(|i| (i * 8 * MIB, 8 * MIB)).collect();
        region.cut(&spans);
        assert_eq!((region.n_pieces(), region.span(1), region.start()), (8, spans[1], 16 * MIB));
        assert_eq!(region.locked_bytes(), 0);
        let whole = (region.ptr().as_ptr(), region.len());
        assert_eq!(resident(whole.0, whole.1), 0, "address space only");
        for piece in 0..2 {
            region.lock(piece).unwrap();
            let (ptr, len) = region.piece_ptr(piece);
            // SAFETY: the piece is the region's own memory.
            unsafe { std::ptr::write_bytes(ptr, 0xA5, len) };
        }
        assert_eq!(region.locked_bytes(), 16 * MIB);
        assert_eq!(resident(whole.0, whole.1), 16 * MIB);
        let released = region.release(1).expect("locked");
        assert_eq!((released.len(), region.locked_bytes()), (8 * MIB, 8 * MIB));
        assert_eq!(resident(whole.0, whole.1), 16 * MIB, "not before it is dropped");
        // Its lock is in the release until that is dropped: no second lock.
        assert!(region.is_releasing(1) && !region.is_locked(1));
        assert!(region.lock(1).is_err(), "a piece on its way back is not locked again");
        drop(released);
        assert!(!region.is_releasing(1));
        assert_eq!(resident(whole.0, whole.1), 8 * MIB, "the piece went back to the system");
        assert!(region.release(1).is_none(), "not locked any more");
        // The piece reads as new memory when it is taken again.
        region.lock(1).unwrap();
        let (ptr, _) = region.piece_ptr(1);
        // SAFETY: as above.
        assert_eq!(unsafe { *ptr }, 0);
    }

    /// How long a device's lock on a GiB of the region takes in pieces of
    /// several sizes, and its release, with what the kernel counts as this
    /// process's resident memory before and after. Opens a GPU: skipped
    /// unless `ARES_TEST_CUDA=1`.
    #[test]
    fn a_gib_is_locked_and_released_in_pieces_and_the_kernel_counts_it() {
        if std::env::var_os("ARES_TEST_CUDA").is_none() {
            eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
            return;
        }
        const MIB: usize = 1 << 20;
        let rss = || {
            let text = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
            text.split_whitespace().nth(1).and_then(|p| p.parse::<usize>().ok()).unwrap_or(0) * 4096
        };
        let device = ares_cuda::Device::open(0, ares_cuda::Wait::Block).expect("device 0");
        for piece in [16 * MIB, 64 * MIB, 256 * MIB] {
            let locker = Box::new(DeviceLock(device.clone()));
            let mut region = HostRegion::elastic(1024 * MIB, 0, locker).unwrap();
            let pieces = 1024 * MIB / piece;
            region.cut(&(0..pieces).map(|i| (i * piece, piece)).collect::<Vec<_>>());
            for round in 0..2 {
                let before = rss();
                let mut slowest = std::time::Duration::ZERO;
                let begun = std::time::Instant::now();
                for index in 0..pieces {
                    let one = std::time::Instant::now();
                    region.lock(index).unwrap();
                    slowest = slowest.max(one.elapsed());
                }
                let locking = begun.elapsed();
                let locked = rss();
                assert!(
                    locked >= before + 1000 * MIB,
                    "a locked GiB is resident: {before} {locked}"
                );
                let begun = std::time::Instant::now();
                let mut slowest_release = std::time::Duration::ZERO;
                for index in 0..pieces {
                    let one = std::time::Instant::now();
                    drop(region.release(index).expect("locked"));
                    slowest_release = slowest_release.max(one.elapsed());
                }
                let releasing = begun.elapsed();
                let after = rss();
                assert!(after + 1000 * MIB <= locked, "a released GiB is gone: {locked} {after}");
                eprintln!(
                    "RES pieces of {} MiB, round {round}: 1 GiB locked in {:.1} ms (slowest piece \
                     {:.2} ms), released in {:.1} ms (slowest piece {:.2} ms); resident {} -> {} \
                     -> {} MiB",
                    piece / MIB,
                    locking.as_secs_f64() * 1e3,
                    slowest.as_secs_f64() * 1e3,
                    releasing.as_secs_f64() * 1e3,
                    slowest_release.as_secs_f64() * 1e3,
                    before / MIB,
                    locked / MIB,
                    after / MIB,
                );
            }
        }
    }
}
