//! Device memory that gives back pieces: one range of addresses whose memory
//! is taken and returned to the driver a piece at a time, while the addresses
//! of the other pieces stay as they are.
//!
//! This is what the pools of expert slots lie in: a kernel finds a slot at
//! the pool's address plus the slot's number times its size, so the pool must
//! be one range, and when another program needs the device's memory the pool
//! gives some of it back without moving what stays (see "VRAM" in
//! `docs/dev/architecture/devices.md`). It is built on the driver's virtual
//! memory calls: the range is reserved once (`cuMemAddressReserve`), and every
//! piece is memory of its own (`cuMemCreate`) mapped into it (`cuMemMap`,
//! `cuMemSetAccess`). Measured on an RTX 3060 (2026-11-28): giving back 64 MiB
//! (`cuMemUnmap`, `cuMemRelease`) took 0.17 ms, taking it again 1 ms; a kernel
//! of the other card wrote into such memory over the patched driver's peer
//! access once the access was set for both cards.
//!
//! A piece is given back where no job runs ([`crate::gate::retire`]), as a
//! free of device memory is: the caller makes sure nothing reads or writes it
//! any more.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::device::Device;
use crate::error::{Error, Result, invalid};
use crate::gate;
use crate::memory::DevicePtr;
use crate::sys::*;

/// A piece of a [`DeviceRegion`].
#[derive(Debug)]
enum Piece {
    /// Memory is behind it.
    Mapped(CUmemGenericAllocationHandle),
    /// Given back; true once the driver has its memory again.
    Releasing(Arc<AtomicBool>),
}

/// A range of device memory in pieces that can be given back and taken again.
#[derive(Debug)]
pub struct DeviceRegion {
    device: Device,
    base: CUdeviceptr,
    /// The bytes of the range: whole pieces, the last one maybe shorter.
    len: usize,
    piece: usize,
    pieces: Vec<Piece>,
    /// The devices that read and write the memory: its own and its peers.
    access: Vec<CUmemAccessDesc>,
}

// SAFETY: the handles and addresses are plain numbers of the driver, whose
// calls are thread-safe.
unsafe impl Send for DeviceRegion {}
// SAFETY: as above; `&self` only reads.
unsafe impl Sync for DeviceRegion {}

/// The properties of memory of the device `ordinal`.
fn properties(ordinal: u32) -> CUmemAllocationProp {
    CUmemAllocationProp {
        kind: CU_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_types: 0,
        location: CUmemLocation { kind: CU_MEM_LOCATION_TYPE_DEVICE, id: ordinal as i32 },
        win32_handle_meta_data: std::ptr::null_mut(),
        alloc_flags: [0; 8],
    }
}

impl DeviceRegion {
    /// The size memory of `device` is taken in: every piece is a multiple of it.
    pub fn granularity(device: &Device) -> Result<u64> {
        let driver = device.bind()?;
        let mut granularity = 0usize;
        let prop = properties(device.ordinal());
        // SAFETY: valid pointers for the write and the properties.
        driver.check("cuMemGetAllocationGranularity", unsafe {
            (driver.api.mem_get_allocation_granularity)(
                &mut granularity,
                &prop,
                CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        })?;
        Ok(granularity.max(1) as u64)
    }

    /// A range of `len` bytes on `device` in pieces of about `piece` bytes
    /// (rounded up to the device's granularity), all of them with memory
    /// behind them, which `device` and `peers` may read and write. The
    /// contents are undefined.
    pub fn new(device: &Device, len: u64, piece: u64, peers: &[Device]) -> Result<Self> {
        if len == 0 {
            return Err(invalid("device region", "a region of 0 bytes was requested".into()));
        }
        let granularity = Self::granularity(device)?;
        let piece = piece.max(1).next_multiple_of(granularity) as usize;
        let len = len.next_multiple_of(granularity) as usize;
        let driver = device.bind()?;
        let mut base: CUdeviceptr = 0;
        // SAFETY: a valid pointer for the write; no fixed address is asked.
        driver.check("cuMemAddressReserve", unsafe {
            (driver.api.mem_address_reserve)(&mut base, len, 0, 0, 0)
        })?;
        let mut access = vec![CUmemAccessDesc {
            location: CUmemLocation {
                kind: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device.ordinal() as i32,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        }];
        for peer in peers.iter().filter(|p| p.ordinal() != device.ordinal()) {
            access.push(CUmemAccessDesc {
                location: CUmemLocation {
                    kind: CU_MEM_LOCATION_TYPE_DEVICE,
                    id: peer.ordinal() as i32,
                },
                flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            });
        }
        let mut region = Self {
            device: device.clone(),
            base,
            len,
            piece,
            pieces: Vec::with_capacity(len.div_ceil(piece)),
            access,
        };
        for k in 0..len.div_ceil(piece) {
            let done = Arc::new(AtomicBool::new(true));
            region.pieces.push(Piece::Releasing(done));
            if !region.map(k)? {
                return Err(Error::Driver {
                    call: "cuMemCreate",
                    code: 2,
                    name: "CUDA_ERROR_OUT_OF_MEMORY".into(),
                    description: format!(
                        "no room for piece {k} of {} bytes of a region of {len} bytes",
                        region.span(k).1
                    ),
                });
            }
        }
        Ok(region)
    }

    /// Where the range starts.
    pub fn ptr(&self) -> DevicePtr {
        DevicePtr(self.base)
    }

    /// The bytes of the range.
    pub fn len(&self) -> u64 {
        self.len as u64
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes of a piece (the last one may be shorter).
    pub fn piece_bytes(&self) -> u64 {
        self.piece as u64
    }

    pub fn n_pieces(&self) -> usize {
        self.pieces.len()
    }

    /// Where piece `k` starts in the range, and its bytes.
    pub fn span(&self, k: usize) -> (u64, u64) {
        let start = k * self.piece;
        (start as u64, (self.len - start).min(self.piece) as u64)
    }

    /// Whether memory is behind piece `k`.
    pub fn is_mapped(&self, k: usize) -> bool {
        matches!(self.pieces[k], Piece::Mapped(_))
    }

    /// The bytes that have memory behind them.
    pub fn mapped_bytes(&self) -> u64 {
        (0..self.pieces.len()).filter(|k| self.is_mapped(*k)).map(|k| self.span(k).1).sum()
    }

    /// Gives the memory of piece `k` back to the driver, where no job runs
    /// ([`gate::retire`]). Its addresses must not be read or written from now
    /// on.
    pub fn unmap(&mut self, k: usize) {
        let done = Arc::new(AtomicBool::new(false));
        let Piece::Mapped(handle) =
            std::mem::replace(&mut self.pieces[k], Piece::Releasing(done.clone()))
        else {
            return;
        };
        let (start, len) = self.span(k);
        gate::retire(Unmap {
            device: self.device.clone(),
            ptr: self.base + start,
            len: len as usize,
            handle,
            done,
        });
    }

    /// Takes memory for piece `k` again. False, and nothing changed, when the
    /// device has no room for it now or its memory is not yet given back.
    pub fn map(&mut self, k: usize) -> Result<bool> {
        match &self.pieces[k] {
            Piece::Mapped(_) => return Ok(true),
            Piece::Releasing(done) if !done.load(Ordering::Acquire) => return Ok(false),
            Piece::Releasing(_) => {}
        }
        let (start, len) = self.span(k);
        let (ptr, len) = (self.base + start, len as usize);
        let driver = self.device.bind()?;
        let prop = properties(self.device.ordinal());
        let mut handle: CUmemGenericAllocationHandle = 0;
        // SAFETY: valid pointers for the write and the properties.
        let code = unsafe { (driver.api.mem_create)(&mut handle, len, &prop, 0) };
        if let Err(error) = driver.check("cuMemCreate", code) {
            return if error.is_out_of_memory() { Ok(false) } else { Err(error) };
        }
        // SAFETY: the range is reserved by this region and has no memory
        // behind it; the handle is of exactly its size.
        let mapped =
            driver.check("cuMemMap", unsafe { (driver.api.mem_map)(ptr, len, 0, handle, 0) });
        let accessible = mapped.and_then(|()| {
            // SAFETY: the range is mapped; the descriptions are valid.
            driver.check("cuMemSetAccess", unsafe {
                (driver.api.mem_set_access)(ptr, len, self.access.as_ptr(), self.access.len())
            })
        });
        if let Err(error) = accessible {
            // SAFETY: what was mapped and created here is undone once.
            unsafe {
                (driver.api.mem_unmap)(ptr, len);
                (driver.api.mem_release)(handle);
            }
            return if error.is_out_of_memory() { Ok(false) } else { Err(error) };
        }
        self.pieces[k] = Piece::Mapped(handle);
        Ok(true)
    }
}

impl Drop for DeviceRegion {
    fn drop(&mut self) {
        for k in 0..self.pieces.len() {
            self.unmap(k);
        }
        gate::retire(AddressFree { device: self.device.clone(), ptr: self.base, len: self.len });
    }
}

/// The return of a piece's memory to the driver.
struct Unmap {
    device: Device,
    ptr: CUdeviceptr,
    len: usize,
    handle: CUmemGenericAllocationHandle,
    done: Arc<AtomicBool>,
}

impl Drop for Unmap {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the range was mapped with this handle and is unmapped
            // and released once.
            unsafe {
                (driver.api.mem_unmap)(self.ptr, self.len);
                (driver.api.mem_release)(self.handle);
            }
        }
        self.done.store(true, Ordering::Release);
    }
}

/// The return of a region's range of addresses, after its pieces.
struct AddressFree {
    device: Device,
    ptr: CUdeviceptr,
    len: usize,
}

impl Drop for AddressFree {
    fn drop(&mut self) {
        if let Ok(driver) = self.device.bind() {
            // SAFETY: the range was reserved by the region and has no memory
            // behind it any more: its pieces were retired before this.
            unsafe { (driver.api.mem_address_free)(self.ptr, self.len) };
        }
    }
}
