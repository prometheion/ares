//! The device side of the expert store: slot memory and slot tables.
//!
//! The store decides; a [`SlotDevice`] moves bytes. Everything is enqueued
//! and nothing waits: copies into and out of slots and writes of table
//! entries run in the order they were enqueued, on a stream of their own
//! next to the model's, and a [`SlotDevice::mark`] tells the store later
//! that the device has got that far.

use std::collections::{HashMap, VecDeque};

use ares_core::Ledger;
use ares_cuda::{Device, DeviceBuffer, DevicePtr, Event, HostShare, PinnedBuffer, Stream};

use super::StoreError;

/// Slot memory of one device and the slot tables its kernels read.
pub trait SlotDevice {
    /// Enqueues a copy of `len` bytes from `src` into slot `slot` of pool
    /// `pool`, `offset` bytes into the slot.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of `len` bytes and stay valid and
    /// unchanged until a mark made after this call has been passed.
    unsafe fn upload(
        &mut self,
        pool: usize,
        slot: u32,
        offset: u64,
        src: *const u8,
        len: usize,
    ) -> Result<(), StoreError>;

    /// Enqueues a copy of `len` bytes out of a slot into `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes, stay valid, and not be
    /// read or written by anyone else, until a mark made after this call has
    /// been passed.
    unsafe fn download(
        &mut self,
        pool: usize,
        slot: u32,
        offset: u64,
        dst: *mut u8,
        len: usize,
    ) -> Result<(), StoreError>;

    /// Enqueues the write of a table entry: `expert` of `layer` is in slot
    /// `slot` of the layer's pool, or (-1) in none.
    fn set_entry(&mut self, layer: usize, expert: u32, slot: i32) -> Result<(), StoreError>;

    /// A mark behind the uploads and table entries enqueued so far, or
    /// (`down`) behind the copies out of slots: the two kinds run side by
    /// side, each in its own order.
    fn mark(&mut self, down: bool) -> Result<u64, StoreError>;

    /// Whether the device has done everything enqueued before `mark`.
    fn passed(&mut self, mark: u64) -> Result<bool, StoreError>;

    /// Enqueues the host's claim of `groups` of the record `token` of
    /// `layer`: the CPU computes those experts, and the kernels that wait
    /// for them stop waiting (`ares_cuda::HostShare`). The store keeps the
    /// rule that goes with it: no table entry for a claimed expert while the
    /// step is at the layer.
    fn claim(&mut self, layer: usize, token: u32, groups: &[u32]) -> Result<(), StoreError>;

    /// Enqueues the host's sums for the record `token` of `layer` (rows of
    /// `n_embd` values), and behind them the mark that lets the layer go on.
    fn deliver(&mut self, layer: usize, token: u32, sums: &[f32]) -> Result<(), StoreError>;
}

/// Table entries on their way to the device: a ring in pinned memory. An
/// entry's copy reads its cell when the stream gets to it, so a cell is
/// reused only after the stream has been waited for.
const ENTRY_RING: usize = 1 << 16;

/// The expert slots of one CUDA device.
///
/// It owns the pools (one buffer per slot size), the slot tables of all
/// layers (`n_experts` i32 per layer, -1 where an expert has no slot) and
/// two streams of its own: one for uploads and table entries, so that
/// bringing an expert in never waits behind the model's kernels and never
/// holds them up, and one for the copies down to RAM of what was evicted, so
/// that an upload a step waits for never queues behind them. An upload into
/// a slot whose old content is still on its way down waits for that copy,
/// on the device.
#[derive(Debug)]
pub struct CudaSlots {
    stream: Stream,
    down: Stream,
    /// Slots (pool, slot) with a copy down enqueued and no upload since, and
    /// the event behind that copy.
    going_down: HashMap<(usize, u32), Event>,

    pools: Vec<(DeviceBuffer, u64)>,
    tables: DeviceBuffer,
    n_experts: u32,
    entries: PinnedBuffer,
    next_entry: usize,
    /// The marks of the upload stream and of the stream of copies down. A
    /// mark's number says which it is on (its lowest bit).
    marks: [VecDeque<(u64, Event)>; 2],
    spare: Vec<Event>,
    next_mark: u64,
    passed: [u64; 2],
    fence: Event,
    /// Whether `fence` has been recorded at all.
    removed: bool,
    shares: Option<Shares>,
}

/// The host shares of the layers on a device, and what feeds them.
#[derive(Debug)]
struct Shares {
    /// Claims and sums go on a stream of their own: they are a few
    /// kilobytes that a step waits for, and on the upload stream they
    /// would queue behind megabytes of experts (the promotions that are on
    /// their way while a step runs): measured once that way, a layer's sums
    /// took 1.8 ms from its record at the median instead of 0.2 to 0.3 ms.
    reply: Stream,
    /// One share per layer, `bytes` apart.
    device: DeviceBuffer,
    /// The same layout in pinned memory: what the copies read. A layer's
    /// part is written once per step, when the step is at the layer, so a
    /// copy of the step before has long run.
    staging: PinnedBuffer,
    bytes: u64,
}

impl CudaSlots {
    /// Allocates the pools (`(slot bytes, slots)` each, in the order of the
    /// store's plan for this device) and the tables of `n_layers` layers of
    /// `n_experts` experts, after reserving them in the ledgers: `vram` for
    /// this device, `pinned` for page-locked host memory.
    pub fn new(
        device: &Device,
        vram: &mut Ledger,
        pinned: &mut Ledger,
        pools: &[(u64, u32)],
        n_layers: usize,
        n_experts: u32,
    ) -> Result<Self, StoreError> {
        let stream = Stream::new(device, 0)?;
        let table_bytes = 4 * n_layers as u64 * u64::from(n_experts);
        vram.reserve("expert slot tables", table_bytes)?;
        pinned.reserve(
            format!("expert table entries CUDA{}", device.ordinal()),
            4 * ENTRY_RING as u64,
        )?;
        let tables = DeviceBuffer::new(device, table_bytes)?;
        // SAFETY: the whole buffer, on this stream's device.
        unsafe { stream.fill(tables.ptr(), 0xFF, table_bytes as usize)? };
        let mut buffers = Vec::with_capacity(pools.len());
        for (index, (stride, slots)) in pools.iter().enumerate() {
            let bytes = stride * u64::from(*slots);
            vram.reserve(format!("expert slots {index} ({slots} of {stride} bytes)"), bytes)?;
            buffers.push((DeviceBuffer::new(device, bytes.max(16))?, *stride));
        }
        stream.synchronize()?;
        Ok(Self {
            down: Stream::new(device, 0)?,
            going_down: HashMap::new(),
            stream,
            pools: buffers,
            tables,
            n_experts,
            entries: PinnedBuffer::new(device, 4 * ENTRY_RING)?,
            next_entry: 0,
            marks: [VecDeque::new(), VecDeque::new()],
            spare: Vec::new(),
            next_mark: 0,
            passed: [0, 0],
            fence: Event::new(device)?,
            removed: false,
            shares: None,
        })
    }

    /// Gives every layer a host share of `share_bytes` bytes
    /// (`ExpertShape::host_share_bytes` for the most rows of a step), after
    /// reserving them in the ledgers: the device's memory and the pinned
    /// copy the host writes. Without shares the device can only be streamed
    /// to.
    pub fn add_host_shares(
        &mut self,
        vram: &mut Ledger,
        pinned: &mut Ledger,
        share_bytes: u64,
    ) -> Result<(), StoreError> {
        let device = self.stream.device();
        let n_layers = self.tables.len() / (4 * u64::from(self.n_experts));
        let bytes = share_bytes.next_multiple_of(16);
        let total = n_layers * bytes;
        vram.reserve("expert host shares", total)?;
        pinned.reserve(format!("expert host shares CUDA{}", device.ordinal()), total)?;
        let buffer = DeviceBuffer::new(device, total.max(16))?;
        // SAFETY: the whole buffer, on this stream's device.
        unsafe { self.stream.fill(buffer.ptr(), 0, total as usize)? };
        self.stream.synchronize()?;
        self.shares = Some(Shares {
            reply: Stream::new(device, 0)?,
            device: buffer,
            staging: PinnedBuffer::new(device, total.max(16) as usize)?,
            bytes,
        });
        Ok(())
    }

    /// The host share of `layer`, for the expert kernels; `None` when the
    /// device has none.
    pub fn host_share(&self, layer: usize) -> Option<DevicePtr> {
        self.shares.as_ref().map(|s| s.device.ptr().offset(layer as u64 * s.bytes))
    }

    /// Where pool `pool` starts, and the bytes from one slot to the next.
    pub fn pool(&self, pool: usize) -> (DevicePtr, u64) {
        (self.pools[pool].0.ptr(), self.pools[pool].1)
    }

    /// The slot table of `layer`: what the expert kernels take.
    pub fn table(&self, layer: usize) -> DevicePtr {
        self.tables.ptr().offset(4 * layer as u64 * u64::from(self.n_experts))
    }

    /// Makes work enqueued on `stream` after this call wait until the device
    /// has removed every table entry whose removal was enqueued so far. A
    /// step's stream calls it before its first expert kernel: what the store
    /// evicted between the steps is then gone from the tables before a
    /// router can find it. Copies that were enqueued after the last removal
    /// are not waited for.
    pub fn fence(&mut self, stream: &Stream) -> Result<(), StoreError> {
        if self.removed {
            stream.wait_event(&self.fence)?;
        }
        Ok(())
    }

    fn slot_ptr(&self, pool: usize, slot: u32, offset: u64) -> DevicePtr {
        let (base, stride) = self.pool(pool);
        base.offset(u64::from(slot) * stride + offset)
    }

    fn event(&mut self) -> Result<Event, StoreError> {
        match self.spare.pop() {
            Some(event) => Ok(event),
            None => Ok(Event::new(self.stream.device())?),
        }
    }
}

impl SlotDevice for CudaSlots {
    unsafe fn upload(
        &mut self,
        pool: usize,
        slot: u32,
        offset: u64,
        src: *const u8,
        len: usize,
    ) -> Result<(), StoreError> {
        if let Some(event) = self.going_down.remove(&(pool, slot)) {
            // The slot's old content first.
            self.stream.wait_event(&event)?;
            self.spare.push(event);
        }
        // SAFETY: the slot is inside the pool (the store's plan sized it) and
        // the source is the caller's guarantee.
        unsafe { self.stream.copy_from_host(self.slot_ptr(pool, slot, offset), src, len)? };
        Ok(())
    }

    unsafe fn download(
        &mut self,
        pool: usize,
        slot: u32,
        offset: u64,
        dst: *mut u8,
        len: usize,
    ) -> Result<(), StoreError> {
        // SAFETY: as in `upload`.
        unsafe { self.down.copy_to_host(dst, self.slot_ptr(pool, slot, offset), len)? };
        let event = match self.going_down.remove(&(pool, slot)) {
            Some(event) => event,
            None => self.event()?,
        };
        event.record(&self.down)?;
        self.going_down.insert((pool, slot), event);
        Ok(())
    }

    fn set_entry(&mut self, layer: usize, expert: u32, slot: i32) -> Result<(), StoreError> {
        if self.next_entry == ENTRY_RING {
            // Every cell has a copy that may not have run yet.
            self.stream.synchronize()?;
            self.next_entry = 0;
        }
        // SAFETY: the cell is inside the pinned ring, and no enqueued copy
        // reads it any more (the ring was waited for when it wrapped).
        let cell = unsafe { self.entries.as_mut_ptr().add(4 * self.next_entry) };
        self.next_entry += 1;
        // SAFETY: as above; a pinned allocation is aligned for i32.
        unsafe { cell.cast::<i32>().write(slot) };
        let entry = self.table(layer).offset(4 * u64::from(expert));
        // SAFETY: the entry is inside the tables, the cell stays unchanged
        // until the ring wraps.
        unsafe { self.stream.copy_from_host(entry, cell, 4)? };
        if slot < 0 {
            // What `fence` waits for: the last removal.
            self.fence.record(&self.stream)?;
            self.removed = true;
        }
        Ok(())
    }

    fn mark(&mut self, down: bool) -> Result<u64, StoreError> {
        let event = self.event()?;
        event.record(if down { &self.down } else { &self.stream })?;
        self.next_mark += 1;
        let id = self.next_mark * 2 + u64::from(down);
        self.marks[usize::from(down)].push_back((id, event));
        Ok(id)
    }

    fn passed(&mut self, mark: u64) -> Result<bool, StoreError> {
        let queue = (mark & 1) as usize;
        while let Some((id, event)) = self.marks[queue].front() {
            if !event.is_done()? {
                break;
            }
            self.passed[queue] = *id;
            if let Some((_, event)) = self.marks[queue].pop_front() {
                self.spare.push(event);
            }
        }
        Ok(mark <= self.passed[queue])
    }

    fn claim(&mut self, layer: usize, token: u32, groups: &[u32]) -> Result<(), StoreError> {
        let Some(shares) = self.shares.as_mut() else {
            return Err(StoreError::Plan("a claim on a device without host shares".into()));
        };
        let Some(words) = groups.iter().max().map(|g| *g as usize + 1) else { return Ok(()) };
        if words > HostShare::MAX_GROUPS as usize {
            return Err(StoreError::Plan(format!("a claim of group {} of a record", words - 1)));
        }
        let at = layer * shares.bytes as usize;
        // SAFETY: the claim words of this layer are inside the pinned copy
        // and inside the share, and no enqueued copy reads them any more
        // (they were last written a step ago).
        unsafe {
            let cells = shares.staging.as_mut_ptr().add(at).cast::<u32>();
            for word in 0..words {
                cells.add(word).write(0);
            }
            for group in groups {
                cells.add(*group as usize).write(HostShare::mark(token));
            }
            shares.reply.copy_from_host(
                shares.device.ptr().offset(at as u64),
                cells.cast::<u8>().cast_const(),
                4 * words,
            )?;
        }
        Ok(())
    }

    fn deliver(&mut self, layer: usize, token: u32, sums: &[f32]) -> Result<(), StoreError> {
        let Some(shares) = self.shares.as_mut() else {
            return Err(StoreError::Plan("sums for a device without host shares".into()));
        };
        let len = size_of_val(sums);
        if HostShare::SUM_OFFSET + len as u64 > shares.bytes {
            return Err(StoreError::Plan(format!("{} sums do not fit a host share", sums.len())));
        }
        let at = layer * shares.bytes as usize;
        // SAFETY: the sums and the mark of this layer are inside the pinned
        // copy and inside the share; as in `claim`, nothing reads them now.
        unsafe {
            let base = shares.staging.as_mut_ptr().add(at);
            let staged = base.add(HostShare::SUM_OFFSET as usize);
            std::ptr::copy_nonoverlapping(sums.as_ptr().cast::<u8>(), staged, len);
            let done = base.add(HostShare::DONE_OFFSET as usize);
            done.cast::<u32>().write(HostShare::mark(token));
            let share = shares.device.ptr().offset(at as u64);
            // The sums first, then their mark: the kernels read the sums
            // once they see the mark.
            shares.reply.copy_from_host(share.offset(HostShare::SUM_OFFSET), staged, len)?;
            shares.reply.copy_from_host(share.offset(HostShare::DONE_OFFSET), done, 4)?;
        }
        Ok(())
    }
}

enum HostOp {
    Upload { pool: usize, at: usize, src: *const u8, len: usize },
    Download { pool: usize, at: usize, dst: *mut u8, len: usize },
    Entry { layer: usize, expert: u32, slot: i32 },
    Claim { layer: usize, token: u32, groups: Vec<u32> },
    Deliver { layer: usize, token: u32, sums: Vec<f32> },
}

/// What the mock device has of a layer's host share.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HostShareState {
    /// The claims of the latest record: its token and the claimed groups.
    pub claimed: Option<(u32, Vec<u32>)>,
    /// The sums that were delivered last, and for which token.
    pub delivered: Option<(u32, Vec<f32>)>,
}

/// Expert slots in ordinary memory: a [`SlotDevice`] without a device, for
/// tests of the store. What is enqueued runs only when [`HostSlots::run`]
/// says so, as a device runs it some time later.
pub struct HostSlots {
    pools: Vec<(Vec<u8>, u64)>,
    tables: Vec<Vec<i32>>,
    shares: Vec<HostShareState>,
    queue: VecDeque<(u64, HostOp)>,
    enqueued: u64,
    done: u64,
}

impl std::fmt::Debug for HostSlots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HostSlots({} pools, {} waiting)", self.pools.len(), self.queue.len())
    }
}

impl HostSlots {
    /// Pools of `(slot bytes, slots)` and empty tables for `n_layers` layers
    /// of `n_experts` experts.
    pub fn new(pools: &[(u64, u32)], n_layers: usize, n_experts: u32) -> Self {
        Self {
            pools: pools
                .iter()
                .map(|(stride, slots)| (vec![0; (*stride * u64::from(*slots)) as usize], *stride))
                .collect(),
            tables: vec![vec![-1; n_experts as usize]; n_layers],
            shares: vec![HostShareState::default(); n_layers],
            queue: VecDeque::new(),
            enqueued: 0,
            done: 0,
        }
    }

    /// Runs up to `n` of the enqueued operations, in order; returns how many ran.
    pub fn run(&mut self, n: usize) -> usize {
        let mut ran = 0;
        while ran < n {
            let Some((id, op)) = self.queue.pop_front() else { break };
            match op {
                HostOp::Upload { pool, at, src, len } => {
                    // SAFETY: `upload`'s caller keeps the source valid until
                    // a later mark is passed, which needs this to have run.
                    let src = unsafe { std::slice::from_raw_parts(src, len) };
                    self.pools[pool].0[at..at + len].copy_from_slice(src);
                }
                HostOp::Download { pool, at, dst, len } => {
                    // SAFETY: as above, for `download`'s destination.
                    let dst = unsafe { std::slice::from_raw_parts_mut(dst, len) };
                    dst.copy_from_slice(&self.pools[pool].0[at..at + len]);
                }
                HostOp::Entry { layer, expert, slot } => self.tables[layer][expert as usize] = slot,
                HostOp::Claim { layer, token, groups } => {
                    self.shares[layer].claimed = Some((token, groups));
                }
                HostOp::Deliver { layer, token, sums } => {
                    self.shares[layer].delivered = Some((token, sums));
                }
            }
            self.done = id;
            ran += 1;
        }
        ran
    }

    /// The table entry of an expert as the device has it now.
    pub fn entry(&self, layer: usize, expert: u32) -> i32 {
        self.tables[layer][expert as usize]
    }

    /// The host share of a layer as the device has it now.
    pub fn share(&self, layer: usize) -> &HostShareState {
        &self.shares[layer]
    }

    /// The bytes of a slot as the device has them now.
    pub fn slot(&self, pool: usize, slot: u32) -> &[u8] {
        let (bytes, stride) = &self.pools[pool];
        &bytes[(u64::from(slot) * stride) as usize..][..*stride as usize]
    }

    /// How many operations wait.
    pub fn waiting(&self) -> usize {
        self.queue.len()
    }

    fn push(&mut self, op: HostOp) {
        self.enqueued += 1;
        self.queue.push_back((self.enqueued, op));
    }
}

impl SlotDevice for HostSlots {
    unsafe fn upload(
        &mut self,
        pool: usize,
        slot: u32,
        offset: u64,
        src: *const u8,
        len: usize,
    ) -> Result<(), StoreError> {
        let at = (u64::from(slot) * self.pools[pool].1 + offset) as usize;
        self.push(HostOp::Upload { pool, at, src, len });
        Ok(())
    }

    unsafe fn download(
        &mut self,
        pool: usize,
        slot: u32,
        offset: u64,
        dst: *mut u8,
        len: usize,
    ) -> Result<(), StoreError> {
        let at = (u64::from(slot) * self.pools[pool].1 + offset) as usize;
        self.push(HostOp::Download { pool, at, dst, len });
        Ok(())
    }

    fn set_entry(&mut self, layer: usize, expert: u32, slot: i32) -> Result<(), StoreError> {
        self.push(HostOp::Entry { layer, expert, slot });
        Ok(())
    }

    fn mark(&mut self, _down: bool) -> Result<u64, StoreError> {
        Ok(self.enqueued)
    }

    fn passed(&mut self, mark: u64) -> Result<bool, StoreError> {
        Ok(mark <= self.done)
    }

    fn claim(&mut self, layer: usize, token: u32, groups: &[u32]) -> Result<(), StoreError> {
        self.push(HostOp::Claim { layer, token, groups: groups.to_vec() });
        Ok(())
    }

    fn deliver(&mut self, layer: usize, token: u32, sums: &[f32]) -> Result<(), StoreError> {
        self.push(HostOp::Deliver { layer, token, sums: sums.to_vec() });
        Ok(())
    }
}
