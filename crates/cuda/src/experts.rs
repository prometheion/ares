//! The mixture of experts on a device: the router of a layer and its routed
//! experts for the rows of a decode step (`kernels/experts.cu`).
//!
//! A layer's experts are not all on the device. Those that are sit in
//! **slots** of a pool, each expert's three matrices one behind the other in
//! the file's quantized bytes ([`ExpertShape`] says where), and a **slot
//! table** per layer maps an expert to its slot or to -1. The kernels find
//! every expert through that table:
//!
//! - [`ExpertKernels::route`] computes the router's logits for the rows,
//!   picks each row's experts and writes the layer's **route record**: the
//!   picks and weights per row and the list of distinct experts with the rows
//!   that picked each ([`RouteRecord`] reads a copy of it on the host).
//! - [`ExpertKernels::experts`] computes the picked experts, each one's
//!   weights read once for all its rows, and adds the weighted sum to the
//!   rows' outputs.
//!
//! An expert that was picked and has no slot when its kernels run is waited
//! for **on the device**: the thread blocks of that expert poll its table
//! entry until the host has uploaded the expert and written the entry, while
//! the blocks of the other experts compute. A step can therefore be enqueued
//! whole, across all layers, without the host knowing which experts it will
//! need; the host follows the copies of the route records and repairs what is
//! missing. The rules for changing the table under running kernels are in
//! `kernels/expert_layout.cuh`.
//!
//! The host may repair a miss in a second way: it computes the expert on the
//! CPU. A layer then has a **host share** on the device ([`HostShare`] says
//! how it is laid out): the host marks the groups of the record it computes,
//! the kernels leave them out, and the down kernel adds the host's sums to
//! the rows' outputs, waiting for them on the device. A layer whose experts
//! are all in slots never reads its share.

use ares_core::GgmlType;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::MAX_BATCH;
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

/// The kernels of the mixture of experts as a fat binary (`build.rs`).
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/experts.fatbin"));

/// The most experts a layer may have (`ARES_EXPERT_MAX_EXPERTS`).
pub const EXPERT_MAX_EXPERTS: u32 = 1024;

/// The most experts the router picks per row (`ARES_EXPERT_MAX_K`).
pub const EXPERT_MAX_K: u32 = 16;

/// The types gate and up may have, with their kernels' names and the
/// alignment the decoders need of a matrix.
const GATE_UP_TYPES: [(GgmlType, &str, u64); 3] = [
    (GgmlType::IQ2_S, "ares_expert_gate_up_iq2_s", 2),
    (GgmlType::IQ2_XXS, "ares_expert_gate_up_iq2_xxs", 2),
    (GgmlType::IQ1_M, "ares_expert_gate_up_iq1_m", 8),
];

/// The types down may have.
const DOWN_TYPES: [(GgmlType, &str, u64); 1] = [(GgmlType::Q2_0, "ares_expert_down_q2_0", 2)];

/// The host share of a layer (`AresHostShare` in
/// `kernels/expert_layout.cuh`): where the host writes which groups of a
/// record it computes on the CPU, and their sums.
///
/// The share is `claim[HostShare::MAX_GROUPS]` (`u32`), a `done` word, and
/// from [`HostShare::SUM_OFFSET`] on the sums: `n_rows` rows of `n_embd`
/// `f32`, row `b`'s the sum over the claimed experts it picked of weight
/// times expert output. For the record with token `t`:
///
/// 1. the host writes [`HostShare::mark`]`(t)` into `claim[g]` of every
///    group `g` it computes: the kernels that waited for those experts stop
///    waiting and leave them out;
/// 2. it writes the sums, and after them the same mark into `done`: the
///    down kernel, which waits for that, adds the sums.
///
/// The rules the kernels rely on: a missing expert of the running step gets
/// a table entry or a claim, never both; the sums are on the device before
/// `done` is marked; and two records that use a share one after the other
/// have different tokens.
#[derive(Clone, Copy, Debug)]
pub struct HostShare;

impl HostShare {
    /// The most groups a record has (`ARES_HOST_MAX_GROUPS`).
    pub const MAX_GROUPS: u32 = 128;
    /// Byte offset of the `done` word.
    pub const DONE_OFFSET: u64 = 4 * Self::MAX_GROUPS as u64;
    /// Byte offset of the sums (`ARES_HOST_SUM_OFFSET`).
    pub const SUM_OFFSET: u64 = Self::DONE_OFFSET + 16;

    /// What marks a claim or the sums as those of the record with `token`.
    /// It is never 0 for a token below `u32::MAX`, so a zeroed share marks
    /// nothing.
    pub fn mark(token: u32) -> u32 {
        token.wrapping_add(1)
    }

    /// Byte offset of the claim word of group `group` of a record.
    pub fn claim_offset(group: u32) -> u64 {
        4 * u64::from(group)
    }
}

/// Bytes of the record's header and of one group (`AresRouteHeader`,
/// `AresExpertGroup`).
const HEADER_BYTES: usize = 16;
const GROUP_BYTES: usize = 48;

/// Threads of a block of the gate and up kernel, and the hidden values it
/// computes (`ARES_EXPERT_GU_WARPS`).
const GATE_UP_THREADS: u32 = 256;
const GATE_UP_VALUES: u32 = 32;

/// The launch shape of the down kernel and of the logits: that of the decode
/// kernels (`ARES_MMVQ_WARPS`, `ARES_MMVQ_ROWS`).
const MMVQ_THREADS: u32 = 128;
const MMVQ_ROWS_PER_BLOCK: u32 = 8;

fn invalid(what: &'static str, detail: String) -> Error {
    Error::Invalid { what, detail }
}

fn check_align(what: &'static str, ptr: DevicePtr, align: u64) -> Result<()> {
    if !ptr.as_raw().is_multiple_of(align) {
        return Err(invalid(
            what,
            format!("address {:#x} is not aligned to {align} bytes", ptr.as_raw()),
        ));
    }
    Ok(())
}

/// The routed experts of one layer: sizes, types, and how an expert lies in
/// its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertShape {
    /// Width of the model: the columns of gate and up, the rows of down.
    pub n_embd: u32,
    /// Inner width of an expert.
    pub n_ff: u32,
    /// Experts of the layer.
    pub n_experts: u32,
    /// Experts the router picks per row.
    pub top_k: u32,
    /// The type of gate and up (a layer's two always agree).
    pub gate_ty: GgmlType,
    /// The type of down.
    pub down_ty: GgmlType,
}

impl ExpertShape {
    /// Bytes of one expert's gate matrix, and of its up matrix: `n_ff` rows of
    /// `n_embd` values. `None` when `n_embd` is not whole blocks of the type.
    pub fn gate_bytes(&self) -> Option<u64> {
        Some(self.gate_ty.bytes_for(u64::from(self.n_embd))? * u64::from(self.n_ff))
    }

    /// Bytes of one expert's down matrix: `n_embd` rows of `n_ff` values.
    pub fn down_bytes(&self) -> Option<u64> {
        Some(self.down_ty.bytes_for(u64::from(self.n_ff))? * u64::from(self.n_embd))
    }

    /// Bytes of one expert in the file: gate, up and down.
    pub fn expert_bytes(&self) -> Option<u64> {
        Some(2 * self.gate_bytes()? + self.down_bytes()?)
    }

    /// Where the up matrix starts in a slot. The gate matrix starts at 0;
    /// each matrix starts on a multiple of 16 bytes.
    pub fn up_offset(&self) -> Option<u64> {
        Some(self.gate_bytes()?.next_multiple_of(16))
    }

    /// Where the down matrix starts in a slot.
    pub fn down_offset(&self) -> Option<u64> {
        Some((self.up_offset()? + self.gate_bytes()?).next_multiple_of(16))
    }

    /// Bytes of a slot: the distance from one slot of a pool to the next.
    pub fn slot_bytes(&self) -> Option<u64> {
        Some((self.down_offset()? + self.down_bytes()?).next_multiple_of(16))
    }

    /// Bytes of the layer's router matrix (F32, `n_experts` rows of `n_embd`).
    pub fn router_bytes(&self) -> u64 {
        4 * u64::from(self.n_experts) * u64::from(self.n_embd)
    }

    /// Bytes of a route record for a step of `n_rows` rows.
    pub fn record_bytes(&self, n_rows: u32) -> u64 {
        let cap = u64::from(self.top_k) * u64::from(n_rows);
        HEADER_BYTES as u64 + cap * (8 + GROUP_BYTES as u64)
    }

    /// Bytes of a layer's [`HostShare`] for steps of up to `n_rows` rows.
    pub fn host_share_bytes(&self, n_rows: u32) -> u64 {
        HostShare::SUM_OFFSET + 4 * u64::from(n_rows) * u64::from(self.n_embd)
    }

    /// Bytes of the logits scratch for `n_rows` rows.
    pub fn logits_bytes(&self, n_rows: u32) -> u64 {
        4 * u64::from(self.n_experts) * u64::from(n_rows)
    }

    /// Bytes of the scratch that holds the quantized hidden values of every
    /// (row, expert) pair of a step of `n_rows` rows.
    pub fn hidden_bytes(&self, n_rows: u32) -> u64 {
        self.hidden_scales_offset(n_rows) + self.pairs(n_rows) * u64::from(self.n_ff / 32) * 8
    }

    fn pairs(&self, n_rows: u32) -> u64 {
        u64::from(self.top_k) * u64::from(n_rows)
    }

    fn hidden_scales_offset(&self, n_rows: u32) -> u64 {
        (self.pairs(n_rows) * u64::from(self.n_ff)).next_multiple_of(16)
    }

    /// Whether the kernels can compute a layer of this shape, and if not, why.
    pub fn check(&self) -> Result<()> {
        let what = "expert shape";
        if !GATE_UP_TYPES.iter().any(|(ty, _, _)| *ty == self.gate_ty) {
            return Err(invalid(
                what,
                format!(
                    "gate and up of type {} have no expert kernel (there is one for {})",
                    self.gate_ty,
                    GATE_UP_TYPES.map(|(t, _, _)| t.name()).join(", ")
                ),
            ));
        }
        if !DOWN_TYPES.iter().any(|(ty, _, _)| *ty == self.down_ty) {
            return Err(invalid(
                what,
                format!(
                    "down of type {} has no expert kernel (there is one for {})",
                    self.down_ty,
                    DOWN_TYPES.map(|(t, _, _)| t.name()).join(", ")
                ),
            ));
        }
        if self.n_embd == 0 || !self.n_embd.is_multiple_of(self.gate_ty.block_values()) {
            return Err(invalid(
                what,
                format!("width {} is not whole blocks of {}", self.n_embd, self.gate_ty),
            ));
        }
        if self.n_ff == 0
            || !self.n_ff.is_multiple_of(self.down_ty.block_values())
            || !self.n_ff.is_multiple_of(GATE_UP_VALUES)
        {
            return Err(invalid(
                what,
                format!(
                    "inner width {} is not whole blocks of {} and groups of {GATE_UP_VALUES}",
                    self.n_ff, self.down_ty
                ),
            ));
        }
        if self.n_experts == 0 || self.n_experts > EXPERT_MAX_EXPERTS {
            return Err(invalid(
                what,
                format!("{} experts; a layer has 1 to {EXPERT_MAX_EXPERTS}", self.n_experts),
            ));
        }
        if self.top_k == 0 || self.top_k > EXPERT_MAX_K || self.top_k > self.n_experts {
            return Err(invalid(
                what,
                format!(
                    "{} experts per row; the router picks 1 to {EXPERT_MAX_K}, at most all {}",
                    self.top_k, self.n_experts
                ),
            ));
        }
        Ok(())
    }
}

/// One picked expert of a route record and the rows that picked it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteGroup {
    pub expert: u32,
    /// The expert's slot when the router ran; `None` when it had none, which
    /// is what the host repairs.
    pub slot: Option<u32>,
    /// Bit `b` is set when row `b` of the step picked the expert.
    pub rows: u32,
    /// Row `b`'s renormalised weight for the expert (0 for rows not in `rows`).
    pub weight: [f32; MAX_BATCH as usize],
}

/// A route record as the host reads it: a copy of what
/// [`ExpertKernels::route`] wrote on the device.
#[derive(Clone, Copy, Debug)]
pub struct RouteRecord<'a> {
    bytes: &'a [u8],
    top_k: usize,
    n_rows: usize,
}

impl<'a> RouteRecord<'a> {
    /// Reads `bytes` as the record of a step of `n_rows` rows with `top_k`
    /// picks each. `None` when the bytes are too few or the header does not
    /// describe such a step.
    pub fn parse(bytes: &'a [u8], top_k: u32, n_rows: u32) -> Option<Self> {
        let (top_k, n_rows) = (top_k as usize, n_rows as usize);
        let cap = top_k * n_rows;
        if bytes.len() < HEADER_BYTES + cap * (8 + GROUP_BYTES) {
            return None;
        }
        let record = Self { bytes, top_k, n_rows };
        (record.word(12) as usize == cap && record.n_groups() <= cap).then_some(record)
    }

    fn word(&self, at: usize) -> u32 {
        let mut word = [0u8; 4];
        word.copy_from_slice(&self.bytes[at..at + 4]);
        u32::from_le_bytes(word)
    }

    fn cap(&self) -> usize {
        self.top_k * self.n_rows
    }

    /// The number the record was written with.
    pub fn token(&self) -> u32 {
        self.word(0)
    }

    /// Distinct experts the rows picked.
    pub fn n_groups(&self) -> usize {
        self.word(4) as usize
    }

    /// Of those, how many had no slot when the router ran.
    pub fn n_missing(&self) -> usize {
        self.word(8) as usize
    }

    /// The experts row `row` picked, the largest probability first.
    pub fn ids(&self, row: usize) -> impl Iterator<Item = u32> + '_ {
        let base = HEADER_BYTES + 4 * row * self.top_k;
        (0..self.top_k).map(move |j| self.word(base + 4 * j))
    }

    /// The weights of row `row`'s picks, in the order of [`RouteRecord::ids`].
    pub fn weights(&self, row: usize) -> impl Iterator<Item = f32> + '_ {
        let base = HEADER_BYTES + 4 * self.cap() + 4 * row * self.top_k;
        (0..self.top_k).map(move |j| f32::from_bits(self.word(base + 4 * j)))
    }

    /// The picked experts, each once, in ascending order.
    pub fn groups(&self) -> impl Iterator<Item = RouteGroup> + '_ {
        let base = HEADER_BYTES + 8 * self.cap();
        (0..self.n_groups()).map(move |g| {
            let at = base + g * GROUP_BYTES;
            let slot = self.word(at + 4) as i32;
            let mut weight = [0.0; MAX_BATCH as usize];
            for (b, w) in weight.iter_mut().enumerate() {
                *w = f32::from_bits(self.word(at + 16 + 4 * b));
            }
            RouteGroup {
                expert: self.word(at),
                slot: u32::try_from(slot).ok(),
                rows: self.word(at + 8),
                weight,
            }
        })
    }
}

/// The kernels of the mixture of experts on one device.
#[derive(Debug, Clone)]
pub struct ExpertKernels {
    device: Device,
    logits: Function,
    select: Function,
    pick: Function,
    group: Function,
    gate_up: Vec<(GgmlType, u64, Function)>,
    down: Vec<(GgmlType, u64, Function)>,
}

impl ExpertKernels {
    /// Loads the kernels on `device`.
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        let typed = |types: &[(GgmlType, &str, u64)]| {
            types
                .iter()
                .map(|(ty, name, align)| Ok((*ty, *align, module.function(name)?)))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            device: device.clone(),
            logits: module.function("ares_expert_logits")?,
            select: module.function("ares_expert_select")?,
            pick: module.function("ares_expert_pick")?,
            group: module.function("ares_expert_group")?,
            gate_up: typed(&GATE_UP_TYPES)?,
            down: typed(&DOWN_TYPES)?,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Routes `n_rows` rows (1 to [`MAX_BATCH`]) through a router: the logits
    /// of every expert ([`ExpertKernels::logits`]), then the picks
    /// ([`ExpertKernels::select`]). Everything goes into the route record at
    /// `record`, with `token` in its header; copy the record to the host to
    /// read it with [`RouteRecord`].
    ///
    /// The router need not be the one of the layer whose table is given: the
    /// router of a later layer on this layer's rows is the prediction of that
    /// layer's picks, and its record says which of them have no slot.
    ///
    /// # Safety
    ///
    /// Those of [`ExpertKernels::logits`] and [`ExpertKernels::select`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn route(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        router: DevicePtr,
        act: DevicePtr,
        logits: DevicePtr,
        slot_table: DevicePtr,
        record: DevicePtr,
        n_rows: u32,
        token: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantees are those of the two functions.
        unsafe {
            self.logits(stream, shape, router, act, logits, n_rows)?;
            self.select(stream, shape, logits, slot_table, record, n_rows, token)
        }
    }

    /// The router's logits: `logits[b][e]` is row `e` of the router matrix
    /// times row `b` of the step, exactly (the matrix is F32 and multiplies
    /// the unquantized activations).
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `router`
    /// must address the router matrix (F32, `n_experts` rows of `n_embd`,
    /// 4-byte aligned); `act` the rows' activations as
    /// `Kernels::quantize_activations` writes them for `n_embd` and `n_rows`
    /// (16-byte aligned); `logits` [`ExpertShape::logits_bytes`] bytes that
    /// overlap neither.
    pub unsafe fn logits(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        router: DevicePtr,
        act: DevicePtr,
        logits: DevicePtr,
        n_rows: u32,
    ) -> Result<()> {
        shape.check()?;
        check_rows(n_rows)?;
        check_align("router matrix", router, 4)?;
        check_align("activations", act, 16)?;
        check_align("router logits", logits, 4)?;
        let args = Args::new()
            .ptr(router)
            .ptr(act)
            .ptr(logits)
            .u32(shape.n_embd)
            .u32(shape.n_experts)
            .u32(n_rows);
        let blocks = shape.n_experts.div_ceil(MMVQ_ROWS_PER_BLOCK);
        // SAFETY: the argument list matches ares_expert_logits (the float
        // weight kernel); the ranges are the caller's guarantee.
        unsafe { self.logits.launch(stream, [blocks, 1, 1], [MMVQ_THREADS, 1, 1], 0, args) }
    }

    /// Picks the experts of `n_rows` rows from their logits: each row's
    /// `top_k` experts with the largest probability (equal values: the lower
    /// index first), their renormalised weights, and the distinct experts
    /// with the rows that picked each and the slot the table has for them.
    /// A step of several rows overwrites `logits`.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launches have run:
    /// `logits` must address `n_rows` rows of `n_experts` f32; `slot_table`
    /// `n_experts` i32; `record` [`ExpertShape::record_bytes`] bytes, 4-byte
    /// aligned, overlapping neither.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn select(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        logits: DevicePtr,
        slot_table: DevicePtr,
        record: DevicePtr,
        n_rows: u32,
        token: u32,
    ) -> Result<()> {
        shape.check()?;
        check_rows(n_rows)?;
        check_align("router logits", logits, 4)?;
        check_align("slot table", slot_table, 4)?;
        check_align("route record", record, 4)?;
        let threads = shape.n_experts.next_multiple_of(32);
        if n_rows == 1 {
            let args = Args::new()
                .ptr(logits)
                .ptr(slot_table)
                .ptr(record)
                .u32(shape.n_experts)
                .u32(shape.top_k)
                .u32(token);
            // SAFETY: the argument list matches ares_expert_select.
            return unsafe { self.select.launch(stream, [1, 1, 1], [threads, 1, 1], 0, args) };
        }
        // Several rows: a block per row picks, then one block makes the groups.
        let args =
            Args::new().ptr(logits).ptr(record).u32(shape.n_experts).u32(shape.top_k).u32(n_rows);
        // SAFETY: the argument list matches ares_expert_pick.
        unsafe { self.pick.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args)? };
        let args = Args::new()
            .ptr(logits)
            .ptr(slot_table)
            .ptr(record)
            .u32(shape.n_experts)
            .u32(shape.top_k)
            .u32(n_rows)
            .u32(token);
        // SAFETY: the argument list matches ares_expert_group.
        unsafe { self.group.launch(stream, [1, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// Computes the experts a route record names for the rows it was routed
    /// from and adds, for every row, the sum of its experts' outputs times
    /// their weights to the row's `n_embd` values at `out`.
    ///
    /// `groups` is the number of distinct experts in the record when the host
    /// knows it; `None` launches for the most a step of `n_rows` rows can
    /// have, and the surplus blocks return at once.
    ///
    /// An expert of the record whose table entry is -1 is waited for on the
    /// device until the entry is written, or, with a `host` share, until the
    /// host has claimed its group there; only its own blocks wait. If any
    /// group was claimed, the launch ends with a wait for the host's sums,
    /// which it adds to `out` ([`HostShare`]).
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launches have run: `pool`
    /// must address the layer's slots, [`ExpertShape::slot_bytes`] apart and
    /// 16-byte aligned, each expert's matrices at the offsets the shape
    /// gives; `slot_table` `n_experts` i32 whose entries name slots of the
    /// pool that hold the expert (or are negative), and that change only by
    /// the rules of `kernels/expert_layout.cuh`; `record` the record that
    /// [`ExpertKernels::route`] wrote for these rows and this table; `act` the
    /// same activations; `hidden` [`ExpertShape::hidden_bytes`] bytes of
    /// scratch, 16-byte aligned; `out` `n_rows * n_embd` f32; `host`, if
    /// given, [`ExpertShape::host_share_bytes`] bytes, 16-byte aligned, that
    /// only change by the rules of [`HostShare`]. `hidden` and `out` must not
    /// overlap anything else named here.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        pool: DevicePtr,
        slot_table: DevicePtr,
        record: DevicePtr,
        act: DevicePtr,
        hidden: DevicePtr,
        out: DevicePtr,
        host: Option<DevicePtr>,
        n_rows: u32,
        groups: Option<u32>,
    ) -> Result<()> {
        shape.check()?;
        check_rows(n_rows)?;
        let host = host.unwrap_or(DevicePtr::NULL);
        check_align("host share", host, 16)?;
        let geometry = || invalid("expert shape", "sizes overflow".into());
        let stride = shape.slot_bytes().ok_or_else(geometry)?;
        let up_offset =
            u32::try_from(shape.up_offset().ok_or_else(geometry)?).map_err(|_| geometry())?;
        let down_offset =
            u32::try_from(shape.down_offset().ok_or_else(geometry)?).map_err(|_| geometry())?;
        let find = |kernels: &[(GgmlType, u64, Function)], ty: GgmlType| {
            kernels.iter().find(|(t, _, _)| *t == ty).map(|(_, align, f)| (*align, f.clone()))
        };
        let (Some((gate_align, gate_up)), Some((down_align, down))) =
            (find(&self.gate_up, shape.gate_ty), find(&self.down, shape.down_ty))
        else {
            return Err(invalid("expert shape", "no kernel for the types".into()));
        };
        check_align("expert pool", pool, 16.max(gate_align).max(down_align))?;
        check_align("slot table", slot_table, 4)?;
        check_align("route record", record, 4)?;
        check_align("activations", act, 16)?;
        check_align("hidden scratch", hidden, 16)?;
        check_align("expert output", out, 4)?;
        let cap = shape.top_k * n_rows;
        let groups = groups.unwrap_or(cap).min(cap);
        if groups == 0 {
            return Ok(());
        }
        let hidden_scales = hidden.offset(shape.hidden_scales_offset(n_rows));

        let args = Args::new()
            .ptr(pool)
            .u64(stride)
            .u32(up_offset)
            .ptr(slot_table)
            .ptr(record)
            .ptr(act)
            .ptr(hidden)
            .ptr(hidden_scales)
            .ptr(host)
            .u32(shape.n_embd)
            .u32(shape.n_ff)
            .u32(n_rows)
            .u32(shape.top_k);
        let tiles = shape.n_ff / GATE_UP_VALUES;
        // SAFETY: the argument list matches the ares_expert_gate_up_* kernels;
        // the ranges are the caller's guarantee.
        unsafe { gate_up.launch(stream, [tiles, groups, 1], [GATE_UP_THREADS, 1, 1], 0, args)? };

        let args = Args::new()
            .ptr(pool)
            .u64(stride)
            .u32(down_offset)
            .ptr(slot_table)
            .ptr(record)
            .ptr(hidden)
            .ptr(hidden_scales)
            .ptr(out)
            .ptr(host)
            .u32(shape.n_ff)
            .u32(shape.n_embd)
            .u32(n_rows)
            .u32(shape.top_k);
        let blocks = shape.n_embd.div_ceil(MMVQ_ROWS_PER_BLOCK);
        // SAFETY: the argument list matches the ares_expert_down_* kernels.
        unsafe { down.launch(stream, [blocks, 1, 1], [MMVQ_THREADS, 1, 1], 0, args) }
    }
}

fn check_rows(n_rows: u32) -> Result<()> {
    if n_rows == 0 || n_rows > MAX_BATCH {
        return Err(invalid(
            "expert kernels",
            format!("{n_rows} rows; one launch takes 1 to {MAX_BATCH}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flash_next(gate_ty: GgmlType) -> ExpertShape {
        ExpertShape {
            n_embd: 2560,
            n_ff: 640,
            n_experts: 512,
            top_k: 10,
            gate_ty,
            down_ty: GgmlType::Q2_0,
        }
    }

    #[test]
    fn slot_layout_matches_the_file_sizes() {
        let shape = flash_next(GgmlType::IQ2_S);
        assert_eq!(shape.gate_bytes(), Some(524_800));
        assert_eq!(shape.down_bytes(), Some(460_800));
        assert_eq!(shape.expert_bytes(), Some(1_510_400));
        assert_eq!(shape.up_offset(), Some(524_800));
        assert_eq!(shape.down_offset(), Some(1_049_600));
        assert_eq!(shape.slot_bytes(), Some(1_510_400));
        assert_eq!(flash_next(GgmlType::IQ2_XXS).slot_bytes(), Some(1_305_600));
        assert_eq!(flash_next(GgmlType::IQ1_M).slot_bytes(), Some(1_177_600));
        assert!(shape.check().is_ok());
        assert_eq!(HostShare::SUM_OFFSET, 528);
        assert_eq!(shape.host_share_bytes(2), 528 + 2 * 2560 * 4);
        assert_eq!(HostShare::mark(0), 1);
    }

    #[test]
    fn refuses_shapes_without_a_kernel() {
        assert!(flash_next(GgmlType::Q4_K).check().is_err());
        let mut shape = flash_next(GgmlType::IQ2_S);
        shape.top_k = 17;
        assert!(shape.check().is_err());
        shape.top_k = 10;
        shape.n_ff = 600;
        assert!(shape.check().is_err());
    }

    #[test]
    fn reads_a_record() {
        let shape = flash_next(GgmlType::IQ2_S);
        let mut bytes = vec![0u8; shape.record_bytes(1) as usize];
        let mut put = |at: usize, v: u32| bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(0, 7);
        put(4, 1);
        put(8, 1);
        put(12, 10);
        put(16, 33);
        put(16 + 40, 0.5f32.to_bits());
        let group = 16 + 80;
        put(group, 33);
        put(group + 4, (-1i32) as u32);
        put(group + 8, 1);
        put(group + 16, 0.5f32.to_bits());
        let record = RouteRecord::parse(&bytes, 10, 1).expect("a record");
        assert_eq!((record.token(), record.n_groups(), record.n_missing()), (7, 1, 1));
        assert_eq!(record.ids(0).next(), Some(33));
        assert_eq!(record.weights(0).next(), Some(0.5));
        let group = record.groups().next().expect("one group");
        assert_eq!((group.expert, group.slot, group.rows, group.weight[0]), (33, None, 1, 0.5));
        assert!(RouteRecord::parse(&bytes, 10, 2).is_none());
    }
}
