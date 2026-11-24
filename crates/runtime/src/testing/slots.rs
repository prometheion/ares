//! Expert slots in ordinary memory, for tests of the expert store without a
//! device.

use std::collections::VecDeque;

use crate::expert_store::{SlotDevice, StoreError};

enum HostOp {
    Upload { pool: usize, at: usize, src: *const u8, len: usize },
    Download { pool: usize, at: usize, dst: *mut u8, len: usize },
    Copy { from_pool: usize, from: usize, pool: usize, at: usize, len: usize },
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
                HostOp::Copy { from_pool, from, pool, at, len } => {
                    let bytes = self.pools[from_pool].0[from..from + len].to_vec();
                    self.pools[pool].0[at..at + len].copy_from_slice(&bytes);
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

    fn adopt(
        &mut self,
        from_pool: usize,
        from: u32,
        pool: usize,
        slot: u32,
        parts: &[(u64, usize)],
        layer: usize,
        expert: u32,
    ) -> Result<(), StoreError> {
        for (offset, len) in parts {
            let from = (u64::from(from) * self.pools[from_pool].1 + offset) as usize;
            let at = (u64::from(slot) * self.pools[pool].1 + offset) as usize;
            self.push(HostOp::Copy { from_pool, from, pool, at, len: *len });
        }
        self.push(HostOp::Entry { layer, expert, slot: slot as i32 });
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
