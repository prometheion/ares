//! The device side of the expert store: slot memory and slot tables.
//!
//! The store decides; a [`SlotDevice`] moves bytes. Everything is enqueued
//! and nothing waits: copies into and out of slots and writes of table
//! entries run in the order they were enqueued, on a stream of their own
//! next to the model's, and a [`SlotDevice::mark`] tells the store later
//! that the device has got that far.

use std::collections::{HashMap, VecDeque};

use ares_core::Ledger;
use ares_cuda::{Device, DeviceBuffer, DevicePtr, Event, PinnedBuffer, Stream};

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
        })
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
}

enum HostOp {
    Upload { pool: usize, at: usize, src: *const u8, len: usize },
    Download { pool: usize, at: usize, dst: *mut u8, len: usize },
    Entry { layer: usize, expert: u32, slot: i32 },
}

/// Expert slots in ordinary memory: a [`SlotDevice`] without a device, for
/// tests of the store. What is enqueued runs only when [`HostSlots::run`]
/// says so, as a device runs it some time later.
pub struct HostSlots {
    pools: Vec<(Vec<u8>, u64)>,
    tables: Vec<Vec<i32>>,
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
}
