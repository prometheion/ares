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
    /// The bytes of a piece of a pool's memory (0: the memory stays), per
    /// pool and piece whether memory is behind it, and how many pieces more
    /// the device has room to take (`None`: any).
    piece: u64,
    mapped: Vec<Vec<bool>>,
    room: Option<usize>,
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
            piece: 0,
            mapped: vec![Vec::new(); pools.len()],
            room: None,
            tables: vec![vec![-1; n_experts as usize]; n_layers],
            shares: vec![HostShareState::default(); n_layers],
            queue: VecDeque::new(),
            enqueued: 0,
            done: 0,
        }
    }

    /// The same, with the memory of every pool in pieces of `piece` bytes
    /// that the store gives back and takes again.
    pub fn with_pieces(mut self, piece: u64) -> Self {
        self.piece = piece;
        self.mapped = self
            .pools
            .iter()
            .map(|(bytes, _)| vec![true; (bytes.len() as u64).div_ceil(piece) as usize])
            .collect();
        self
    }

    /// Whether piece `piece` of pool `pool` has memory behind it.
    pub fn is_mapped(&self, pool: usize, piece: usize) -> bool {
        self.mapped[pool].get(piece).copied().unwrap_or(self.piece == 0)
    }

    /// The device has room to take `pieces` pieces more (`None`: any).
    pub fn set_room(&mut self, pieces: Option<usize>) {
        self.room = pieces;
    }

    /// Panics when `len` bytes at `at` of pool `pool` lie in a piece without
    /// memory: what a device would fault on.
    fn check_mapped(&self, pool: usize, at: usize, len: usize) {
        if self.piece == 0 || len == 0 {
            return;
        }
        let (first, last) = (at as u64 / self.piece, (at + len - 1) as u64 / self.piece);
        for k in first..=last {
            assert!(
                self.mapped[pool][k as usize],
                "pool {pool}: bytes {at}+{len} in piece {k}, which has no memory"
            );
        }
    }

    /// Runs up to `n` of the enqueued operations, in order; returns how many ran.
    pub fn run(&mut self, n: usize) -> usize {
        let mut ran = 0;
        while ran < n {
            let Some((id, op)) = self.queue.pop_front() else { break };
            match &op {
                HostOp::Upload { pool, at, len, .. } | HostOp::Download { pool, at, len, .. } => {
                    self.check_mapped(*pool, *at, *len);
                }
                HostOp::Copy { from_pool, from, pool, at, len } => {
                    self.check_mapped(*from_pool, *from, *len);
                    self.check_mapped(*pool, *at, *len);
                }
                _ => {}
            }
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

    fn pieces(&self, pool: usize) -> (u64, usize) {
        if self.piece == 0 { (0, 0) } else { (self.piece, self.mapped[pool].len()) }
    }

    fn give_piece(&mut self, pool: usize, piece: usize) {
        // Nothing that was enqueued may still touch it.
        let (start, end) = (piece as u64 * self.piece, (piece as u64 + 1) * self.piece);
        for (_, op) in &self.queue {
            let touches = |p: usize, at: usize, len: usize| {
                p == pool && (at as u64) < end && (at + len) as u64 > start
            };
            let busy = match op {
                HostOp::Upload { pool, at, len, .. } | HostOp::Download { pool, at, len, .. } => {
                    touches(*pool, *at, *len)
                }
                HostOp::Copy { from_pool, from, pool, at, len } => {
                    touches(*from_pool, *from, *len) || touches(*pool, *at, *len)
                }
                _ => false,
            };
            assert!(!busy, "piece {piece} of pool {pool} is given back while a copy touches it");
        }
        self.mapped[pool][piece] = false;
    }

    fn take_piece(&mut self, pool: usize, piece: usize) -> Result<bool, StoreError> {
        match &mut self.room {
            Some(0) => return Ok(false),
            Some(room) => *room -= 1,
            None => {}
        }
        self.mapped[pool][piece] = true;
        Ok(true)
    }
}
