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
//! An expert's matrices are stored in any weight type a model file has
//! ([`ExpertShape::types`]): gate and up of a layer share one type, down has
//! its own, and both families of kernels have one per type and role. What a
//! shape must have is rows of whole blocks of its types
//! ([`ExpertShape::check`], and [`ExpertShape::check_rows`] for the many-row
//! kernels).
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
//!
//! Those kernels take at most [`MAX_BATCH`] rows. [`ExpertRowKernels`], further down,
//! takes as many as there is memory for, for prompts and windows of drafts: a plan sorts
//! the rows' picks by expert on the device and tensor-core kernels read each expert once
//! for all the rows that picked it, from an address the caller gives for the expert (a
//! slot, transit memory).

use ares_core::GgmlType;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::MAX_BATCH;
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::stream::{Event, Stream};

/// The kernels of the mixture of experts as a fat binary (`build.rs`).
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/experts.fatbin"));

/// The most experts a layer may have (`ARES_EXPERT_MAX_EXPERTS`).
pub const EXPERT_MAX_EXPERTS: u32 = 1024;

/// The most experts the router picks per row (`ARES_EXPERT_MAX_K`).
pub const EXPERT_MAX_K: u32 = 16;

/// The weight types an expert's matrices may be stored in: every one a model file can hold
/// a weight in, with the suffix of its kernels' names (`ares_expert_gate_up_<suffix>`,
/// `ares_expert_down_<suffix>`, `ares_er_gate_up_<suffix>_n1` ...; `kernels/experts.cu` and
/// `kernels/expertrows*.cu` list the same). Gate and up of a layer share one type, down has
/// its own. The decoders load nothing wider than 16 bytes and every matrix of a slot starts
/// on a multiple of 16, so no type asks more of a slot's address than that.
const EXPERT_TYPES: [(GgmlType, &str); 28] = [
    (GgmlType::Q4_K, "q4_K"),
    (GgmlType::Q5_K, "q5_K"),
    (GgmlType::Q6_K, "q6_K"),
    (GgmlType::Q3_K, "q3_K"),
    (GgmlType::Q2_K, "q2_K"),
    (GgmlType::IQ4_XS, "iq4_xs"),
    (GgmlType::IQ4_NL, "iq4_nl"),
    (GgmlType::IQ3_S, "iq3_s"),
    (GgmlType::IQ3_XXS, "iq3_xxs"),
    (GgmlType::IQ2_S, "iq2_s"),
    (GgmlType::IQ2_XS, "iq2_xs"),
    (GgmlType::IQ2_XXS, "iq2_xxs"),
    (GgmlType::IQ1_S, "iq1_s"),
    (GgmlType::IQ1_M, "iq1_m"),
    (GgmlType::Q8_0, "q8_0"),
    (GgmlType::Q4_0, "q4_0"),
    (GgmlType::Q4_1, "q4_1"),
    (GgmlType::Q5_0, "q5_0"),
    (GgmlType::Q5_1, "q5_1"),
    (GgmlType::TQ1_0, "tq1_0"),
    (GgmlType::TQ2_0, "tq2_0"),
    (GgmlType::MXFP4, "mxfp4"),
    (GgmlType::NVFP4, "nvfp4"),
    (GgmlType::Q1_0, "q1_0"),
    (GgmlType::Q2_0, "q2_0"),
    (GgmlType::F32, "f32"),
    (GgmlType::F16, "f16"),
    (GgmlType::BF16, "bf16"),
];

/// The index of `ty` in [`EXPERT_TYPES`]: `None` for a type no weight is stored in (the
/// integer types, F64 and the two that ggml keeps for itself).
fn expert_type(ty: GgmlType) -> Option<usize> {
    EXPERT_TYPES.iter().position(|(t, _)| *t == ty)
}

/// The values of a row that an activation scale covers: a row of an expert's matrix is
/// whole groups of them, whatever the type's own block.
const ACT_GROUP: u32 = 32;

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

    /// Bytes of one list of predicted experts ([`ExpertKernels::predict`]) for a step of
    /// `n_rows` rows: the step's token, the number of experts and at most `top_k` per row,
    /// in words, rounded up to 64 bytes.
    pub fn predict_list_bytes(&self, n_rows: u32) -> u64 {
        (4 * (2 + u64::from(self.top_k) * u64::from(n_rows))).next_multiple_of(64)
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

    /// The weight types the kernels compute an expert in: every type a model file stores
    /// weights in.
    pub fn types() -> impl Iterator<Item = GgmlType> {
        EXPERT_TYPES.iter().map(|(ty, _)| *ty)
    }

    /// Whether the kernels can compute a layer of this shape, and if not, why. Every
    /// weight type is taken, for gate and up and for down; what is refused is a width
    /// that is not whole blocks of the type or whole groups of 32 values (the activations
    /// and the hidden values are scaled per 32).
    pub fn check(&self) -> Result<()> {
        let what = "expert shape";
        for (matrix, ty) in [("gate and up", self.gate_ty), ("down", self.down_ty)] {
            if expert_type(ty).is_none() {
                return Err(invalid(
                    what,
                    format!("{matrix} of type {ty}, which holds no weights"),
                ));
            }
        }
        if self.n_embd == 0
            || !self.n_embd.is_multiple_of(self.gate_ty.block_values())
            || !self.n_embd.is_multiple_of(ACT_GROUP)
        {
            return Err(invalid(
                what,
                format!(
                    "width {} is not whole blocks of {} and groups of {ACT_GROUP}",
                    self.n_embd, self.gate_ty
                ),
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

    /// The rows of the step the record was written for.
    pub fn rows(&self) -> usize {
        self.n_rows
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

/// What the router does with a pick whose expert only the file holds
/// ([`ExpertKernels::route_pruned`], `kernels/expert_router.cuh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PruneRule {
    /// The pick is left out and contributes nothing.
    Skip,
    /// The router's best expert that is in a tier and that the row did not
    /// pick stands in for it.
    Router,
    /// The first of the pick's most similar experts ([`NearEntry`]) that is
    /// in a tier and that the row did not pick stands in for it; a pick
    /// without one stays.
    Similar,
}

/// The weights of what a pruned row computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PruneWeights {
    /// Every untouched pick keeps the router's weight; a stand-in takes the
    /// weight of the pick it stands in for.
    Keep,
    /// Every computed expert gets its probability over the sum of the
    /// probabilities of the computed experts.
    Renormalise,
    /// As [`PruneWeights::Keep`], a stand-in's weight times the scale of its
    /// entry in the table of similar experts.
    Scaled,
}

/// The dial of the pruning: which picks a rule may touch. A pick is touched
/// when its expert is in no tier, its rank (0 the strongest) is `from_rank`
/// or more, its weight is below `below_weight`, and, for
/// [`PruneRule::Similar`], a stand-in has a closeness of `min_closeness` or
/// more.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prune {
    pub rule: PruneRule,
    pub weights: PruneWeights,
    pub from_rank: u32,
    pub below_weight: f32,
    pub min_closeness: f32,
}

impl Prune {
    /// `rule` for the picks from rank `from_rank` on (0 the strongest), with
    /// the weights the rule is measured best with: the router's kept for
    /// skip and similar, renormalised for the router's own stand-ins.
    pub fn new(rule: PruneRule, from_rank: u32) -> Self {
        let weights = match rule {
            PruneRule::Router => PruneWeights::Renormalise,
            PruneRule::Skip | PruneRule::Similar => PruneWeights::Keep,
        };
        Self { rule, weights, from_rank, below_weight: f32::INFINITY, min_closeness: 0.0 }
    }
}

/// One entry of an expert's list of most similar experts, as the router's
/// pruning reads it (`AresNear`): 8 bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NearEntry {
    pub expert: u16,
    /// How alike the two experts' outputs are, 0 to 1.
    pub closeness: f32,
    /// The factor that brings the stand-in's output nearest to the pick's.
    pub scale: f32,
}

impl NearEntry {
    /// Bytes of an entry.
    pub const BYTES: usize = 8;

    /// The entry as the kernels read it.
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[0..2].copy_from_slice(&self.expert.to_le_bytes());
        out[2..4].copy_from_slice(&closeness_code(self.closeness).to_le_bytes());
        out[4..8].copy_from_slice(&self.scale.to_le_bytes());
        out
    }
}

/// A closeness of 0 to 1 as the kernels compare it: 0 to 65535.
fn closeness_code(closeness: f32) -> u16 {
    (closeness.clamp(0.0, 1.0) * 65535.0).round() as u16
}

/// Where the pruning of one layer reads on the device.
#[derive(Clone, Copy, Debug)]
pub struct PruneLayer {
    /// `n_experts` bytes: 0 for an expert only the file holds.
    pub cached: DevicePtr,
    /// `n_experts * near_count` [`NearEntry`]s, each expert's most similar
    /// first; null when there is no table.
    pub near: DevicePtr,
    pub near_count: u32,
}

/// A list of predicted experts as the host reads it: a copy of what
/// [`ExpertKernels::predict`] wrote for one router.
#[derive(Clone, Copy, Debug)]
pub struct PredictedList<'a> {
    bytes: &'a [u8],
    n: usize,
}

impl<'a> PredictedList<'a> {
    /// Reads `bytes` as the list of the step with `token`. `None` when the list is of
    /// another step (no prediction was made in this one) or does not hold what it says.
    pub fn parse(bytes: &'a [u8], token: u32) -> Option<Self> {
        let word = |at: usize| -> Option<u32> {
            Some(u32::from_le_bytes(bytes.get(4 * at..4 * at + 4)?.try_into().ok()?))
        };
        if word(0)? != token {
            return None;
        }
        let n = word(1)? as usize;
        (bytes.len() >= 4 * (2 + n)).then_some(Self { bytes, n })
    }

    /// The predicted experts, by their best rank among the rows' picks, then by number.
    pub fn experts(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.n).map(move |i| {
            let at = 4 * (2 + i);
            u32::from_le_bytes(self.bytes[at..at + 4].try_into().unwrap_or([0; 4]))
        })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
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
    select_pruned: Function,
    pick_pruned: Function,
    predict_logits: Function,
    predict_pick: Function,
    /// Per type of [`EXPERT_TYPES`]: the gate and up kernel and the down kernel.
    typed: Vec<(Function, Function)>,
}

impl ExpertKernels {
    /// Loads the kernels on `device`.
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        let typed = EXPERT_TYPES
            .iter()
            .map(|(_, suffix)| {
                Ok((
                    module.function(&format!("ares_expert_gate_up_{suffix}"))?,
                    module.function(&format!("ares_expert_down_{suffix}"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device: device.clone(),
            logits: module.function("ares_expert_logits")?,
            select: module.function("ares_expert_select")?,
            pick: module.function("ares_expert_pick")?,
            group: module.function("ares_expert_group")?,
            select_pruned: module.function("ares_expert_select_pruned")?,
            pick_pruned: module.function("ares_expert_pick_pruned")?,
            predict_logits: module.function("ares_expert_predict_logits")?,
            predict_pick: module.function("ares_expert_predict_pick")?,
            typed,
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

    /// [`ExpertKernels::route`] with pruning: the same picks and weights go
    /// into the record (the router's), and the groups, which are what the
    /// expert kernels compute, follow `prune` for the picks whose expert
    /// `layer.cached` says only the file holds. A row with no such pick
    /// computes what [`ExpertKernels::route`] gives it.
    ///
    /// # Safety
    ///
    /// Those of [`ExpertKernels::route`], and `layer.cached` must address
    /// `n_experts` bytes and `layer.near`, for [`PruneRule::Similar`],
    /// `n_experts * near_count` entries, on the stream's device and alive
    /// until the launches have run.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn route_pruned(
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
        prune: &Prune,
        layer: &PruneLayer,
    ) -> Result<()> {
        shape.check()?;
        check_rows(n_rows)?;
        check_align("router logits", logits, 4)?;
        check_align("slot table", slot_table, 4)?;
        check_align("route record", record, 4)?;
        check_align("pruning table", layer.near, 4)?;
        if prune.rule == PruneRule::Similar
            && (layer.near == DevicePtr::NULL || layer.near_count == 0)
        {
            return Err(invalid(
                "expert pruning",
                "the rule of similar experts without a table".into(),
            ));
        }
        let rule = match prune.rule {
            PruneRule::Skip => 1,
            PruneRule::Router => 2,
            PruneRule::Similar => 3,
        };
        let mode = match prune.weights {
            PruneWeights::Keep => 0,
            PruneWeights::Renormalise => 1,
            PruneWeights::Scaled => 2,
        };
        let below = if prune.below_weight.is_finite() { prune.below_weight } else { f32::MAX };
        let tail = |args: Args| {
            args.ptr(layer.cached)
                .ptr(layer.near)
                .u32(rule)
                .u32(mode)
                .u32(prune.from_rank)
                .f32(below)
                .u32(u32::from(closeness_code(prune.min_closeness)))
                .u32(layer.near_count)
        };
        // SAFETY: the caller's guarantees.
        unsafe { self.logits(stream, shape, router, act, logits, n_rows)? };
        let threads = shape.n_experts.next_multiple_of(32);
        if n_rows == 1 {
            let args = tail(
                Args::new()
                    .ptr(logits)
                    .ptr(slot_table)
                    .ptr(record)
                    .u32(shape.n_experts)
                    .u32(shape.top_k)
                    .u32(token),
            );
            // SAFETY: the argument list matches ares_expert_select_pruned.
            return unsafe {
                self.select_pruned.launch(stream, [1, 1, 1], [threads, 1, 1], 0, args)
            };
        }
        let args = tail(
            Args::new().ptr(logits).ptr(record).u32(shape.n_experts).u32(shape.top_k).u32(n_rows),
        );
        // SAFETY: the argument list matches ares_expert_pick_pruned.
        unsafe { self.pick_pruned.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args)? };
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
        let args = Args::new()
            .ptr(logits)
            .ptr(record)
            .u32(shape.n_experts)
            .u32(shape.top_k)
            .u32(n_rows)
            .u32(0)
            .u32(n_rows);
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

    /// Predicts what the routers `routers` of later layers (one or two, F32 like a layer's
    /// own, of the layers of `shape`) will pick for the `n_rows` rows of `act`, the rows a
    /// router of this step read (Pre-gated MoE, FATE): per router the experts that some row
    /// would pick among its `top_k` (ranked by logit, which is the router's own order),
    /// by their best rank over the rows, then by number. A list per router is written at
    /// `lists`, `list_bytes` apart ([`ExpertShape::predict_list_bytes`], read with
    /// [`PredictedList`]), headed by `token`. Two launches whatever the number of routers.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launches have run: `routers` as the router
    /// of [`ExpertKernels::logits`]; `act` the activations of `n_rows` rows as
    /// `Kernels::quantize_activations` writes them; `logits` `routers.len()` times
    /// [`ExpertShape::logits_bytes`] bytes; `lists` `routers.len() * list_bytes` bytes, 4-byte
    /// aligned. The outputs overlap neither each other nor the inputs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn predict(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        routers: &[DevicePtr],
        act: DevicePtr,
        logits: DevicePtr,
        lists: DevicePtr,
        list_bytes: u64,
        n_rows: u32,
        token: u32,
    ) -> Result<()> {
        shape.check()?;
        check_rows(n_rows)?;
        let (first, second) = match routers {
            [one] => (*one, *one),
            [one, two] => (*one, *two),
            _ => {
                return Err(invalid(
                    "expert prediction",
                    format!("{} routers; a prediction takes one or two", routers.len()),
                ));
            }
        };
        if list_bytes < shape.predict_list_bytes(n_rows) || !list_bytes.is_multiple_of(4) {
            return Err(invalid(
                "expert prediction",
                format!("lists of {list_bytes} bytes for {n_rows} rows"),
            ));
        }
        for router in routers {
            check_align("router matrix", *router, 4)?;
        }
        check_align("activations", act, 16)?;
        check_align("router logits", logits, 4)?;
        check_align("predicted experts", lists, 4)?;
        let n = routers.len() as u32;
        let args = Args::new()
            .ptr(first)
            .ptr(second)
            .ptr(act)
            .ptr(logits)
            .u32(shape.n_embd)
            .u32(shape.n_experts)
            .u32(n_rows);
        let blocks = shape.n_experts.div_ceil(MMVQ_ROWS_PER_BLOCK);
        // SAFETY: the argument list matches ares_expert_predict_logits; the ranges are the
        // caller's guarantee.
        unsafe {
            self.predict_logits.launch(stream, [blocks, n, 1], [MMVQ_THREADS, 1, 1], 0, args)?;
        }
        let threads = shape.n_experts.next_multiple_of(32);
        let args = Args::new()
            .ptr(logits)
            .ptr(lists)
            .u32((list_bytes / 4) as u32)
            .u32(shape.n_experts)
            .u32(shape.top_k)
            .u32(n_rows)
            .u32(token);
        // SAFETY: the argument list matches ares_expert_predict_pick.
        unsafe { self.predict_pick.launch(stream, [n, 1, 1], [threads, 1, 1], 0, args) }
    }

    /// The router's logits for any number of rows, from activations in the layout of
    /// [`Kernels::quantize_activations`](crate::Kernels::quantize_activations) for `n_rows`
    /// rows (what the fused kernels that quantize write). The float router multiplies the f32
    /// section of the buffer, in launches of [`MAX_BATCH`] rows that each read the router
    /// matrix once. `logits[b][e]` is as [`ExpertKernels::logits`] gives it.
    ///
    /// # Safety
    ///
    /// Those of [`ExpertKernels::logits`], with `act` the layout of `n_rows` rows and `logits`
    /// [`ExpertShape::logits_bytes`] of `n_rows`.
    pub unsafe fn logits_rows(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        router: DevicePtr,
        act: DevicePtr,
        logits: DevicePtr,
        n_rows: u32,
    ) -> Result<()> {
        shape.check()?;
        let row_bytes = 4 * u64::from(shape.n_embd);
        if n_rows == 0 || u64::from(n_rows) * u64::from(shape.n_embd) >= 1 << 32 {
            return Err(invalid(
                "expert kernels",
                format!("{n_rows} rows of {} values", shape.n_embd),
            ));
        }
        let f32_at = |rows: u32| {
            crate::Kernels::quantized_activation_bytes(shape.n_embd, rows)
                - row_bytes * u64::from(rows)
        };
        let mut first = 0;
        while first < n_rows {
            let rows = (n_rows - first).min(MAX_BATCH);
            // A launch of `rows` rows finds its f32 values `f32_at(rows)` bytes behind its
            // pointer. Those are the chunk's own rows, which lie in the caller's buffer, so
            // the pointer is inside it too.
            let chunk = act.offset(f32_at(n_rows) + row_bytes * u64::from(first) - f32_at(rows));
            let out = logits.offset(4 * u64::from(first) * u64::from(shape.n_experts));
            // SAFETY: the chunk's rows are inside the caller's buffers.
            unsafe { self.logits(stream, shape, router, chunk, out, rows)? };
            first += rows;
        }
        Ok(())
    }

    /// Picks the experts of any number of rows from their logits: each row's `top_k` experts
    /// with the largest probability (equal values: the lower index first) and their
    /// renormalised weights go into `record`, in the form the many-row kernels read
    /// ([`ExpertRowKernels::plan_rows`]): a header of 16 bytes, `n_rows * top_k` ids
    /// (`u32`), then as many weights (`f32`). The header and the groups of the decode kernels
    /// are not written. `logits` is overwritten.
    ///
    /// # Safety
    ///
    /// `logits` must address `n_rows` rows of `n_experts` f32 and `record`
    /// `16 + 8 * n_rows * top_k` bytes, 4-byte aligned, on the stream's device, overlapping
    /// neither, alive until the launch has run.
    pub unsafe fn pick_rows(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        logits: DevicePtr,
        record: DevicePtr,
        n_rows: u32,
    ) -> Result<()> {
        // SAFETY: the caller's guarantee, for a record of the step's rows.
        unsafe { self.pick_rows_into(stream, shape, logits, record, n_rows, 0, n_rows) }
    }

    /// [`ExpertKernels::pick_rows`] into a record of `record_rows` rows, of which the
    /// `n_rows` rows of `logits` are the rows from `row0` on: their ids and weights go where
    /// those rows' go in the larger record. A prompt read layer by layer routes its
    /// micro-batches one after the other into the record of their chunk this way.
    ///
    /// # Safety
    ///
    /// `logits` must address `n_rows` rows of `n_experts` f32 and `record`
    /// `16 + 8 * record_rows * top_k` bytes, 4-byte aligned, on the stream's device,
    /// overlapping neither, alive until the launch has run.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn pick_rows_into(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        logits: DevicePtr,
        record: DevicePtr,
        n_rows: u32,
        row0: u32,
        record_rows: u32,
    ) -> Result<()> {
        shape.check()?;
        if n_rows == 0 {
            return Err(invalid("expert kernels", "a pick of no rows".into()));
        }
        if row0.checked_add(n_rows).is_none_or(|end| end > record_rows) {
            return Err(invalid(
                "expert kernels",
                format!("rows {row0}..+{n_rows} of a record of {record_rows} rows"),
            ));
        }
        check_align("router logits", logits, 4)?;
        check_align("route record", record, 4)?;
        let args = Args::new()
            .ptr(logits)
            .ptr(record)
            .u32(shape.n_experts)
            .u32(shape.top_k)
            .u32(n_rows)
            .u32(row0)
            .u32(record_rows);
        let threads = shape.n_experts.next_multiple_of(32);
        // SAFETY: the argument list matches ares_expert_pick.
        unsafe { self.pick.launch(stream, [n_rows, 1, 1], [threads, 1, 1], 0, args) }
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
        let kernel = |ty: GgmlType| {
            expert_type(ty)
                .map(|i| &self.typed[i])
                .ok_or_else(|| invalid("expert shape", "no kernel for the types".into()))
        };
        let (gate_up, down) = (&kernel(shape.gate_ty)?.0, &kernel(shape.down_ty)?.1);
        check_align("expert pool", pool, 16)?;
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

// ---------------------------------------------------------------------------
// Many rows: a plan on the device and tile kernels on the tensor cores.

/// The tensor-core kernels of the experts for any number of rows as fat binaries
/// (`build.rs`), each with the suffixes of the types whose gate and up kernels and whose
/// down kernels it holds. `kernels/expertrows.cu` has the plan, the ordered sum and the
/// types with a decoder written for these kernels; the other four have every other type,
/// through the decoders of the dense tile kernels (`kernels/expertrows_mm.cuh`).
type RowsModule = (&'static [u8], &'static [&'static str], &'static [&'static str]);
static ROWS_MODULES: [RowsModule; 5] = [
    (
        include_bytes!(concat!(env!("OUT_DIR"), "/expertrows.fatbin")),
        &["iq2_s", "iq2_xxs", "iq1_m", "iq1_s"],
        &["q2_0"],
    ),
    (
        include_bytes!(concat!(env!("OUT_DIR"), "/expertrows_k.fatbin")),
        &["q4_K", "q5_K", "q6_K", "q3_K", "q2_K", "q8_0", "iq4_xs", "iq4_nl"],
        &["q4_K", "q5_K", "q6_K", "q3_K", "q2_K", "q8_0", "iq4_xs", "iq4_nl"],
    ),
    (
        include_bytes!(concat!(env!("OUT_DIR"), "/expertrows_i.fatbin")),
        &["iq3_s", "iq3_xxs", "iq2_xs", "q2_0"],
        &["iq3_s", "iq3_xxs", "iq2_xs", "iq2_s", "iq2_xxs", "iq1_s", "iq1_m"],
    ),
    (
        include_bytes!(concat!(env!("OUT_DIR"), "/expertrows_l.fatbin")),
        &["q4_0", "q4_1", "q5_0", "q5_1", "tq1_0", "tq2_0"],
        &["q4_0", "q4_1", "q5_0", "q5_1", "tq1_0", "tq2_0"],
    ),
    (
        include_bytes!(concat!(env!("OUT_DIR"), "/expertrows_f.fatbin")),
        &["mxfp4", "nvfp4", "q1_0", "f32", "f16", "bf16"],
        &["mxfp4", "nvfp4", "q1_0", "f32", "f16", "bf16"],
    ),
];

/// The sizes of an item that the many-row kernels exist for: the plan cuts an
/// expert's run of pairs into items of at most this many, and a kernel built for a
/// tile holds that many rows of activations in shared memory. Smaller tiles waste
/// fewer tensor-core columns on an expert with few pairs; larger ones read the
/// expert's weights once for more of them.
pub const ROWS_TILES: [u32; 4] = [8, 16, 32, 64];

/// Pairs a block of the plan sorts (`ARES_ER_PLAN_THREADS`).
const ROWS_PLAN_BLOCK: u32 = 1024;

/// Threads of a block of the tile kernels (`ARES_ER_THREADS`): four warps with 16
/// rows of the weight matrices each, 64 rows to a block.
const ROWS_THREADS: u32 = 128;
const ROWS_WEIGHT_ROWS: u32 = 64;

/// The stages of activations in shared memory of the gate and up kernels by tile (the
/// `STAGES` of the instances in `kernels/expertrows.cu`): three, and two for the largest
/// tile, which would otherwise leave room for one block on a multiprocessor. The down
/// kernel has 4 for every tile.
const ROWS_GATE_UP_STAGES: [u64; 4] = [3, 3, 3, 2];
const ROWS_DOWN_STAGES: u64 = 4;

/// The values of a row the many-row gate and up kernel takes per step, and the values of
/// a row of down per stage (`kernels/expertrows_gate_up.cuh`, `expertrows_down.cuh`).
const ROWS_GATE_CHUNK: u32 = 128;
const ROWS_DOWN_BLOCK: u32 = 64;

/// The gate and up decoders written for the many-row kernels (`kernels/expertrows_types.cuh`,
/// and IQ1_S's in `mm_types_more.cuh`) and the bytes of their tables in shared memory. Every
/// other decoder is the item decoder of the dense tile kernels, whose table is
/// [`Kernels::tile_table_bytes`](crate::Kernels::tile_table_bytes).
const ROWS_OWN_GATE_UP: [(GgmlType, u64); 4] = [
    (GgmlType::IQ2_S, 8192 + 4096),
    (GgmlType::IQ2_XXS, 2048 + 4096),
    (GgmlType::IQ1_M, 16384),
    (GgmlType::IQ1_S, 16384),
];

/// The bytes of shared memory the decoder of `ty` keeps its tables in: in the gate and up
/// kernel, and in the down kernel (Q2_0, whose down decoder is written for it, has none
/// either way).
fn rows_table_bytes(ty: GgmlType) -> (u64, u64) {
    let tile = u64::from(crate::Kernels::tile_table_bytes(ty).unwrap_or(0));
    let gate_up = ROWS_OWN_GATE_UP.iter().find(|(t, _)| *t == ty).map_or(tile, |(_, bytes)| *bytes);
    (gate_up, tile)
}

/// How the many-row down kernel adds an expert's output to a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accumulate {
    /// `red.add` of the weighted output into the row of `out`. Needs no scratch. The
    /// order in which a row's picks are added depends on the schedule, so the sum can
    /// differ in its last bit from run to run.
    Atomic,
    /// The weighted outputs go to a scratch of [`ExpertShape::rows_scratch_bytes`] and
    /// [`ExpertRowKernels::rows_combine`] adds them to the row in the order of the
    /// router. The row's bits do not depend on the other rows of the call.
    Ordered,
}

/// Where the parts of a plan lie in its buffer (`AresErLayout` in
/// `kernels/expertrows_layout.cuh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowsPlanLayout {
    n_experts: u32,
    n_pairs: u32,
    max_items: u32,
    /// Bytes of the whole plan.
    pub bytes: u64,
}

/// Bytes of the header of a plan: `n_items`, `n_valid`, `missing_pairs`, `token`, and
/// room for more.
const PLAN_HEADER_BYTES: u64 = 64;
/// Bytes of an item of a plan: the expert, its first position, the count and a pad.
const PLAN_ITEM_BYTES: u64 = 16;

impl RowsPlanLayout {
    /// The layout of a plan of `n_pairs` pairs (rows times picks) over `n_experts`
    /// experts with items of at most `tile` pairs.
    pub fn new(n_experts: u32, n_pairs: u32, tile: u32) -> Self {
        let max_items = n_pairs.min(n_experts) + n_pairs / tile.max(1);
        let mut layout = Self { n_experts, n_pairs, max_items, bytes: 0 };
        // The plan of more than one block of pairs keeps the block counts behind it.
        let blocks = if n_pairs > ROWS_PLAN_BLOCK {
            4 * u64::from(n_pairs.div_ceil(ROWS_PLAN_BLOCK)) * u64::from(n_experts)
        } else {
            0
        };
        layout.bytes = layout.block_counts_at() + blocks;
        layout
    }

    /// The most items the plan can hold. The tile kernels are launched for this many
    /// blocks in x; those past the plan's own count return at once.
    pub fn max_items(&self) -> u32 {
        self.max_items
    }

    fn expert_base_at(&self) -> u64 {
        PLAN_HEADER_BYTES
    }

    fn items_at(&self) -> u64 {
        (self.expert_base_at() + 4 * (u64::from(self.n_experts) + 1)).next_multiple_of(16)
    }

    fn pair_row_at(&self) -> u64 {
        self.items_at() + PLAN_ITEM_BYTES * u64::from(self.max_items)
    }

    fn pair_weight_at(&self) -> u64 {
        self.pair_row_at() + 4 * u64::from(self.n_pairs)
    }

    fn pair_pos_at(&self) -> u64 {
        self.pair_weight_at() + 4 * u64::from(self.n_pairs)
    }

    fn missing_at(&self) -> u64 {
        self.pair_pos_at() + 4 * u64::from(self.n_pairs)
    }

    fn block_counts_at(&self) -> u64 {
        (self.missing_at() + 4 * u64::from(self.n_experts)).next_multiple_of(16)
    }
}

/// One item of a plan: up to a tile of the pairs of one expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowsItem {
    pub expert: u32,
    /// The sorted position of the item's first pair.
    pub first: u32,
    pub count: u32,
}

/// A plan as the host reads it: a copy of the buffer that
/// [`ExpertRowKernels::plan_rows`] wrote.
///
/// The pairs are numbered `row * top_k + j` for the `j`-th pick of a row, as the route
/// record has them. The plan lists them sorted by expert, the pairs of an expert in the
/// order of their numbers; a *sorted position* is an index into that list.
#[derive(Clone, Copy, Debug)]
pub struct RowsPlan<'a> {
    bytes: &'a [u8],
    layout: RowsPlanLayout,
}

impl<'a> RowsPlan<'a> {
    /// Reads `bytes` as the plan of `n_rows` rows of `shape` with items of `tile`
    /// pairs. `None` when there are too few bytes.
    pub fn parse(bytes: &'a [u8], shape: &ExpertShape, n_rows: u32, tile: u32) -> Option<Self> {
        let n_pairs = n_rows.checked_mul(shape.top_k)?;
        let layout = RowsPlanLayout::new(shape.n_experts, n_pairs, tile);
        (bytes.len() as u64 >= layout.bytes).then_some(Self { bytes, layout })
    }

    fn word(&self, at: u64) -> u32 {
        let at = at as usize;
        let mut word = [0u8; 4];
        word.copy_from_slice(&self.bytes[at..at + 4]);
        u32::from_le_bytes(word)
    }

    /// Items in the plan.
    pub fn n_items(&self) -> u32 {
        self.word(0)
    }

    /// Pairs that have a position: those whose expert id was in range.
    pub fn n_valid(&self) -> u32 {
        self.word(4)
    }

    /// Pairs whose expert had no address when a tile kernel looked: the sum of the
    /// counts of their items. Zero when the plan is built; the gate and up kernel adds
    /// to it.
    pub fn missing_pairs(&self) -> u32 {
        self.word(8)
    }

    /// The number the plan was built with.
    pub fn token(&self) -> u32 {
        self.word(12)
    }

    /// The first sorted position of `expert`; for `n_experts` it is `n_valid`.
    pub fn first(&self, expert: u32) -> u32 {
        self.word(self.layout.expert_base_at() + 4 * u64::from(expert))
    }

    /// How many pairs picked `expert`.
    pub fn count(&self, expert: u32) -> u32 {
        self.first(expert + 1) - self.first(expert)
    }

    /// The items, in the order of their experts.
    pub fn items(&self) -> impl Iterator<Item = RowsItem> + '_ {
        (0..self.n_items()).map(move |i| {
            let at = self.layout.items_at() + PLAN_ITEM_BYTES * u64::from(i);
            RowsItem { expert: self.word(at), first: self.word(at + 4), count: self.word(at + 8) }
        })
    }

    /// The row of the pair at sorted position `pos`.
    pub fn pair_row(&self, pos: u32) -> u32 {
        self.word(self.layout.pair_row_at() + 4 * u64::from(pos))
    }

    /// Its router weight.
    pub fn pair_weight(&self, pos: u32) -> f32 {
        f32::from_bits(self.word(self.layout.pair_weight_at() + 4 * u64::from(pos)))
    }

    /// The sorted position of pair `pair` (`row * top_k + j`); `None` when its expert
    /// id was out of range.
    pub fn position(&self, pair: u32) -> Option<u32> {
        let pos = self.word(self.layout.pair_pos_at() + 4 * u64::from(pair));
        (pos != u32::MAX).then_some(pos)
    }

    /// Whether a tile kernel found `expert` without an address.
    pub fn was_missing(&self, expert: u32) -> bool {
        self.word(self.layout.missing_at() + 4 * u64::from(expert)) != 0
    }
}

impl ExpertShape {
    /// Whether the many-row kernels ([`ExpertRowKernels`]) can compute a layer of this shape:
    /// what [`ExpertShape::check`] asks, and widths in whole steps of those kernels (128
    /// values of the model's width, 64 of the inner one).
    pub fn check_rows(&self) -> Result<()> {
        self.check()?;
        if !self.n_embd.is_multiple_of(ROWS_GATE_CHUNK)
            || !self.n_ff.is_multiple_of(ROWS_DOWN_BLOCK)
        {
            return Err(invalid(
                "expert shape",
                format!(
                    "the many-row kernels take widths in steps of {ROWS_GATE_CHUNK} and inner \
                     widths in steps of {ROWS_DOWN_BLOCK}, not {} and {}",
                    self.n_embd, self.n_ff
                ),
            ));
        }
        Ok(())
    }

    /// Bytes of the plan of `n_rows` rows with items of `tile` pairs.
    pub fn rows_plan_bytes(&self, n_rows: u32, tile: u32) -> u64 {
        RowsPlanLayout::new(self.n_experts, n_rows.saturating_mul(self.top_k), tile).bytes
    }

    /// Bytes of the hidden values of `n_rows` rows: `n_ff` f16 values for each pair,
    /// then (at the next multiple of 16 bytes) the f32 power of two that each block of
    /// 64 of them was divided by, `n_ff / 64` rows of one per pair.
    pub fn rows_hidden_bytes(&self, n_rows: u32) -> u64 {
        self.rows_hidden_scales_offset(n_rows) + 4 * u64::from(self.n_ff / 64) * self.pairs(n_rows)
    }

    fn rows_hidden_scales_offset(&self, n_rows: u32) -> u64 {
        (2 * u64::from(self.n_ff) * self.pairs(n_rows)).next_multiple_of(16)
    }

    /// Bytes of the scratch of [`Accumulate::Ordered`]: a row of `n_embd` f32 for each
    /// pair.
    pub fn rows_scratch_bytes(&self, n_rows: u32) -> u64 {
        4 * u64::from(self.n_embd) * self.pairs(n_rows)
    }

    /// Bytes of the activations of `n_rows` rows in the form the many-row kernels read:
    /// that of [`PrefillPath::F16`](crate::PrefillPath::F16), the same number as
    /// [`Kernels::prefill_activation_bytes`](crate::Kernels::prefill_activation_bytes)
    /// gives for it.
    pub fn rows_activation_bytes(&self, n_rows: u32) -> u64 {
        self.rows_activation_scales_offset(n_rows) + 4 * u64::from(n_rows)
    }

    fn rows_activation_scales_offset(&self, n_rows: u32) -> u64 {
        let lines = (u64::from(self.n_embd) * 2).div_ceil(128) + 1;
        ((lines | 1) * 128 * u64::from(n_rows)).next_multiple_of(16)
    }
}

/// Dynamic shared memory of the gate and up kernel for items of `tile` pairs: the
/// type's tables, the stages of activations (256 bytes a row each), the rows of the
/// item and the largest hidden value of each of the four warps for each column.
fn rows_gate_up_shared(table: u64, tile: u32) -> u64 {
    let stages = ROWS_GATE_UP_STAGES[ROWS_TILES.iter().position(|t| *t == tile).unwrap_or(0)];
    table + u64::from(tile) * (stages * 256 + 4 + 16)
}

/// Dynamic shared memory of the down kernel for items of `tile` pairs: the type's tables,
/// the stages of hidden values (128 bytes a row each), and the rows and weights of the item.
fn rows_down_shared(table: u64, tile: u32) -> u64 {
    table + u64::from(tile) * (ROWS_DOWN_STAGES * 128 + 8)
}

/// The index of `tile` in [`ROWS_TILES`].
fn tile_index(tile: u32) -> Result<usize> {
    ROWS_TILES.iter().position(|t| *t == tile).ok_or_else(|| {
        invalid(
            "many-row experts",
            format!("items of {tile} pairs; the kernels exist for {ROWS_TILES:?}"),
        )
    })
}

/// The tensor-core kernels of the routed experts for any number of rows on one
/// device: a plan built on the device from the rows' picks, and kernels for gate and
/// up and for down that read each expert's weights once for all the rows that picked
/// it. Where the decode kernels ([`ExpertKernels`]) take up to eight rows, these take
/// as many as there is memory for, and are meant for prompts and windows of drafts.
///
/// A call goes through stages that are all on the device, none needs the host:
///
/// 1. [`ExpertRowKernels::plan_rows`] sorts the pairs of the rows' picks by expert and
///    cuts each expert's run into items of at most `tile` pairs.
/// 2. [`ExpertRowKernels::rows_gate_up`] computes `silu(gate x) * (up x)` of every
///    pair and leaves it as f16 in the hidden buffer.
/// 3. [`ExpertRowKernels::rows_down`] multiplies the hidden values by down, scales by
///    the router weights and adds to the rows ([`Accumulate`]).
/// 4. With [`Accumulate::Ordered`], [`ExpertRowKernels::rows_combine`].
///
/// [`ExpertRowKernels::rows_experts`] runs stages 2 to 4. The plan does not depend on
/// where the experts are, so one plan serves several calls of stages 2 to 4 with
/// different address tables (the waves of a layer).
///
/// **The address table.** The kernels find an expert through a table of one device
/// address per expert (`u64`, 0 for none): the expert's *slot*, in which the gate
/// matrix starts at the slot's first byte, up at [`ExpertShape::up_offset`] and down
/// at [`ExpertShape::down_offset`], each in the file's quantized bytes. A slot of the
/// expert store, a place in transit memory or any other 16-byte aligned memory with
/// that layout will do, and the table can name different ones in each call: the
/// experts that are resident are read in place and the others from where they were
/// streamed to. An expert of the plan without an address is **not read**: its pairs add
/// nothing to the output, are counted in [`RowsPlan::missing_pairs`] and the expert is
/// flagged in [`RowsPlan::was_missing`].
///
/// **Arithmetic.** Weights are decoded to f16 and multiplied with f16 activations on
/// the tensor cores with f32 accumulation, so each product is the exact one up to the
/// f16 rounding of the weights and the activations (the activations are divided per
/// row by a power of two, see
/// [`Kernels::prepare_prefill_activations`](crate::Kernels::prepare_prefill_activations),
/// so that nothing overflows). The hidden values are f16 as well, with a power of two
/// scale per pair and 64 values. The value of a pair is the same bits whatever other
/// rows the call has.
///
/// **Types.** Every weight type has its kernels for gate and up and for down. IQ2_S,
/// IQ2_XXS, IQ1_M and IQ1_S as gate and up and Q2_0 as down have a decoder written for
/// these kernels (`kernels/expertrows_types.cuh`); every other type and role decodes with
/// the item decoder of the dense tile kernels (`kernels/expertrows_mm.cuh`), which holds
/// more in registers per step. `docs/ARCHITECTURE.md` has the measured times of each.
///
/// Needs compute capability 8.0 or newer (the `mma` shape and `cp.async` of Ampere).
#[derive(Debug, Clone)]
pub struct ExpertRowKernels {
    device: Device,
    plan_small: Function,
    plan_count: Function,
    plan_offsets: Function,
    plan_scatter: Function,
    combine: Function,
    /// Per type of [`EXPERT_TYPES`]: the bytes of shared memory of its tables and the
    /// kernels by tile, for gate and up and for down.
    gate_up: Vec<(u64, [Function; 4])>,
    down: Vec<(u64, [Function; 4])>,
}

impl ExpertRowKernels {
    /// Loads the kernels on `device`. Fails on a device below compute capability 8.0.
    pub fn load(device: &Device) -> Result<Self> {
        let (major, minor) = device.info().compute_capability;
        if major < 8 {
            return Err(invalid(
                "many-row experts",
                format!(
                    "{} has compute capability {major}.{minor}; the tensor-core kernels need 8.0",
                    device.info().name
                ),
            ));
        }
        let limit = u64::from(device.info().max_shared_per_block_optin);
        // The kernels of one type by tile, each allowed the shared memory `shared` gives.
        let tiled = |module: &Module, name: &str, shared: &dyn Fn(u32) -> u64| {
            let mut functions = Vec::with_capacity(ROWS_TILES.len());
            for (tile, suffix) in ROWS_TILES.into_iter().zip(["n1", "n2", "n4", "n8"]) {
                let function = module.function(&format!("{name}_{suffix}"))?;
                let bytes = shared(tile);
                if bytes > limit {
                    return Err(invalid(
                        "many-row experts",
                        format!(
                            "{name} for {tile} pairs needs {bytes} bytes of shared memory; the \
                             device allows {limit}"
                        ),
                    ));
                }
                function.allow_dynamic_shared(bytes as u32)?;
                functions.push(function);
            }
            <[Function; 4]>::try_from(functions)
                .map_err(|_| invalid("many-row experts", "a kernel per tile".into()))
        };
        let mut gate_up: Vec<Option<(u64, [Function; 4])>> = vec![None; EXPERT_TYPES.len()];
        let mut down = gate_up.clone();
        let mut base = None;
        for (image, gate_up_types, down_types) in &ROWS_MODULES {
            let module = Module::load(device, image)?;
            for (types, kernels, is_down) in
                [(gate_up_types, &mut gate_up, false), (down_types, &mut down, true)]
            {
                for suffix in types.iter() {
                    let index = EXPERT_TYPES
                        .iter()
                        .position(|(_, s)| s == suffix)
                        .ok_or_else(|| invalid("many-row experts", format!("no type {suffix}")))?;
                    let tables = rows_table_bytes(EXPERT_TYPES[index].0);
                    kernels[index] = Some(if is_down {
                        let name = format!("ares_er_down_{suffix}");
                        (tables.1, tiled(&module, &name, &|tile| rows_down_shared(tables.1, tile))?)
                    } else {
                        let name = format!("ares_er_gate_up_{suffix}");
                        let shared = |tile| rows_gate_up_shared(tables.0, tile);
                        (tables.0, tiled(&module, &name, &shared)?)
                    });
                }
            }
            base.get_or_insert(module);
        }
        let module = base.ok_or_else(|| invalid("many-row experts", "no module".into()))?;
        let all = |kernels: Vec<Option<(u64, [Function; 4])>>| {
            kernels.into_iter().collect::<Option<Vec<_>>>().ok_or_else(|| {
                invalid("many-row experts", "a weight type without its kernels".into())
            })
        };
        Ok(Self {
            device: device.clone(),
            plan_small: module.function("ares_er_plan_small")?,
            plan_count: module.function("ares_er_plan_count")?,
            plan_offsets: module.function("ares_er_plan_offsets")?,
            plan_scatter: module.function("ares_er_plan_scatter")?,
            combine: module.function("ares_er_combine")?,
            gate_up: all(gate_up)?,
            down: all(down)?,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Checks what every call has to: the shape, the rows and the tile. Returns the
    /// number of pairs and the index of the tile.
    fn check_call(shape: &ExpertShape, n_rows: u32, tile: u32) -> Result<(u32, usize)> {
        shape.check_rows()?;
        let index = tile_index(tile)?;
        let pairs = n_rows.checked_mul(shape.top_k).filter(|p| n_rows > 0 && *p < 1 << 28);
        let Some(pairs) = pairs else {
            return Err(invalid(
                "many-row experts",
                format!("{n_rows} rows of {} picks; a call takes 1 to 2^28 pairs", shape.top_k),
            ));
        };
        Ok((pairs, index))
    }

    /// Builds the plan of `n_rows` rows from their route record.
    ///
    /// The record is the one the router writes
    /// ([`ExpertShape::record_bytes`] for 8 rows or less, in general): a header of 16
    /// bytes, then `n_rows * top_k` expert ids (`u32`, row `b`'s picks at
    /// `[b * top_k, (b + 1) * top_k)`) and `n_rows * top_k` weights (`f32`). Only
    /// those two arrays are read, so a record written by the picking kernel for a
    /// larger step has the same form. An id not below `n_experts` is dropped: the
    /// pair has no position and no kernel computes it.
    ///
    /// `token` goes into the plan's header ([`RowsPlan::token`]). The plan is
    /// `ExpertShape::rows_plan_bytes(n_rows, tile)` bytes.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launches have run: `record` must
    /// address the arrays above, 16-byte aligned; `plan` the plan's bytes, 16-byte
    /// aligned, overlapping neither.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn plan_rows(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        n_rows: u32,
        tile: u32,
        record: DevicePtr,
        plan: DevicePtr,
        token: u32,
    ) -> Result<()> {
        let (pairs, _) = Self::check_call(shape, n_rows, tile)?;
        check_align("route record", record, 16)?;
        check_align("plan", plan, 16)?;
        let block = [ROWS_PLAN_BLOCK, 1, 1];
        if pairs <= ROWS_PLAN_BLOCK {
            let args = Args::new()
                .ptr(record)
                .ptr(plan)
                .u32(shape.n_experts)
                .u32(pairs)
                .u32(shape.top_k)
                .u32(tile)
                .u32(token);
            // SAFETY: the argument list matches ares_er_plan_small.
            return unsafe { self.plan_small.launch(stream, [1, 1, 1], block, 0, args) };
        }
        let blocks = pairs.div_ceil(ROWS_PLAN_BLOCK);
        let args = Args::new()
            .ptr(record)
            .ptr(plan)
            .u32(shape.n_experts)
            .u32(pairs)
            .u32(shape.top_k)
            .u32(tile);
        // SAFETY: the argument list matches ares_er_plan_count.
        unsafe { self.plan_count.launch(stream, [blocks, 1, 1], block, 0, args)? };
        let args = Args::new().ptr(plan).u32(shape.n_experts).u32(pairs).u32(tile).u32(token);
        // SAFETY: the argument list matches ares_er_plan_offsets.
        unsafe { self.plan_offsets.launch(stream, [1, 1, 1], block, 0, args)? };
        let args = Args::new()
            .ptr(record)
            .ptr(plan)
            .u32(shape.n_experts)
            .u32(pairs)
            .u32(shape.top_k)
            .u32(tile);
        // SAFETY: the argument list matches ares_er_plan_scatter.
        unsafe { self.plan_scatter.launch(stream, [blocks, 1, 1], block, 0, args) }
    }

    /// Computes the hidden values `silu(gate x) * (up x)` of every pair of the plan
    /// whose expert has an address, for the activations `act` of `n_rows` rows.
    ///
    /// `act` is the f16 form of [`Kernels::prepare_prefill_activations`](
    /// crate::Kernels::prepare_prefill_activations) for `n_embd` values per row
    /// ([`ExpertShape::rows_activation_bytes`]). `hidden` receives
    /// [`ExpertShape::rows_hidden_bytes`] bytes: the pair at sorted position `p` has its
    /// `n_ff` f16 values at `p * n_ff` (in the order [`ExpertRowKernels::rows_down`]
    /// reads, not the order of the matrix rows) and the power of two they were divided
    /// by at the scale rows behind them. The plan's `missing_pairs` is added to for
    /// every pair of an expert without an address.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launches have run: `addresses` must
    /// address `n_experts` `u64`, 8-byte aligned, each 0 or the address of a slot of the
    /// shape's layout (16-byte aligned, all three matrices readable) that nothing writes
    /// while the launch runs; `plan` the plan of these rows and this `tile` that
    /// [`ExpertRowKernels::plan_rows`] wrote; `act` and `hidden` the bytes above,
    /// 16-byte aligned, overlapping nothing else named here.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows_gate_up(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        n_rows: u32,
        tile: u32,
        addresses: DevicePtr,
        plan: DevicePtr,
        act: DevicePtr,
        hidden: DevicePtr,
    ) -> Result<()> {
        let (pairs, index) = Self::check_call(shape, n_rows, tile)?;
        check_align("expert addresses", addresses, 8)?;
        check_align("plan", plan, 16)?;
        check_align("activations", act, 16)?;
        check_align("hidden values", hidden, 16)?;
        let Some((table, kernels)) = expert_type(shape.gate_ty).map(|i| &self.gate_up[i]) else {
            return Err(invalid("expert shape", "no many-row kernel for the types".into()));
        };
        let geometry = || invalid("expert shape", "sizes overflow".into());
        let up_offset =
            u32::try_from(shape.up_offset().ok_or_else(geometry)?).map_err(|_| geometry())?;
        let layout = RowsPlanLayout::new(shape.n_experts, pairs, tile);
        let args = Args::new()
            .ptr(addresses)
            .u32(up_offset)
            .ptr(plan)
            .ptr(act)
            .ptr(act.offset(shape.rows_activation_scales_offset(n_rows)))
            .ptr(hidden)
            .ptr(hidden.offset(shape.rows_hidden_scales_offset(n_rows)))
            .u32(shape.n_embd)
            .u32(shape.n_ff)
            .u32(shape.n_experts)
            .u32(pairs);
        let grid = [layout.max_items(), shape.n_ff / ROWS_WEIGHT_ROWS, 1];
        let shared = rows_gate_up_shared(*table, tile) as u32;
        // SAFETY: the argument list matches the ares_er_gate_up_* kernels; the ranges
        // are the caller's guarantee.
        unsafe { kernels[index].launch(stream, grid, [ROWS_THREADS, 1, 1], shared, args) }
    }

    /// Multiplies the hidden values of every pair of the plan whose expert has an
    /// address by the expert's down matrix and the pair's router weight, and adds the
    /// product to the pair's row: into `out` with [`Accumulate::Atomic`], into `scratch`
    /// with [`Accumulate::Ordered`] (then [`ExpertRowKernels::rows_combine`] adds it to
    /// `out`; the scratch of a missing expert's pairs is zeroed).
    ///
    /// # Safety
    ///
    /// Those of [`ExpertRowKernels::rows_gate_up`], and: `hidden` must hold what that
    /// wrote for this plan and these `addresses` (or for an address table with
    /// the same experts present); `out` `n_rows * n_embd` f32, 4-byte aligned; `scratch`
    /// (for `Ordered` only, else it may be [`DevicePtr::NULL`])
    /// [`ExpertShape::rows_scratch_bytes`] bytes, 4-byte aligned. `out` and `scratch`
    /// must not overlap anything else named.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows_down(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        n_rows: u32,
        tile: u32,
        way: Accumulate,
        addresses: DevicePtr,
        plan: DevicePtr,
        hidden: DevicePtr,
        out: DevicePtr,
        scratch: DevicePtr,
    ) -> Result<()> {
        let (pairs, index) = Self::check_call(shape, n_rows, tile)?;
        check_align("expert addresses", addresses, 8)?;
        check_align("plan", plan, 16)?;
        check_align("hidden values", hidden, 16)?;
        check_align("expert output", out, 4)?;
        check_align("scratch", scratch, 4)?;
        if way == Accumulate::Ordered && scratch == DevicePtr::NULL {
            return Err(invalid("many-row experts", "the ordered sum needs a scratch".into()));
        }
        let geometry = || invalid("expert shape", "sizes overflow".into());
        let down_offset =
            u32::try_from(shape.down_offset().ok_or_else(geometry)?).map_err(|_| geometry())?;
        let layout = RowsPlanLayout::new(shape.n_experts, pairs, tile);
        let args = Args::new()
            .ptr(addresses)
            .u32(down_offset)
            .ptr(plan)
            .ptr(hidden)
            .ptr(hidden.offset(shape.rows_hidden_scales_offset(n_rows)))
            .ptr(out)
            .ptr(scratch)
            .u32(shape.n_ff)
            .u32(shape.n_embd)
            .u32(shape.n_experts)
            .u32(pairs)
            .u32(u32::from(way == Accumulate::Atomic));
        let grid = [layout.max_items(), shape.n_embd / ROWS_WEIGHT_ROWS, 1];
        let Some((table, kernels)) = expert_type(shape.down_ty).map(|i| &self.down[i]) else {
            return Err(invalid("expert shape", "no many-row kernel for the types".into()));
        };
        let shared = rows_down_shared(*table, tile) as u32;
        // SAFETY: the argument list matches the ares_er_down_* kernels.
        unsafe { kernels[index].launch(stream, grid, [ROWS_THREADS, 1, 1], shared, args) }
    }

    /// Adds the experts' outputs that [`Accumulate::Ordered`] left in `scratch` to the
    /// rows of `out`, each row's picks in the order of the router.
    ///
    /// # Safety
    ///
    /// On the stream's device and alive until the launch has run: `plan` the plan of
    /// these rows, `scratch` what [`ExpertRowKernels::rows_down`] wrote, `out`
    /// `n_rows * n_embd` f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows_combine(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        n_rows: u32,
        tile: u32,
        plan: DevicePtr,
        scratch: DevicePtr,
        out: DevicePtr,
    ) -> Result<()> {
        let (pairs, _) = Self::check_call(shape, n_rows, tile)?;
        check_align("plan", plan, 16)?;
        check_align("scratch", scratch, 4)?;
        check_align("expert output", out, 4)?;
        let args = Args::new()
            .ptr(plan)
            .ptr(scratch)
            .ptr(out)
            .u32(shape.n_experts)
            .u32(pairs)
            .u32(tile)
            .u32(shape.top_k)
            .u32(shape.n_embd);
        // SAFETY: the argument list matches ares_er_combine.
        unsafe { self.combine.launch(stream, [n_rows, 1, 1], [256, 1, 1], 0, args) }
    }

    /// Stages 2 to 4: gate and up, down and, for [`Accumulate::Ordered`], the
    /// combine. `scratch` is used by `Ordered` only.
    ///
    /// # Safety
    ///
    /// Those of [`ExpertRowKernels::rows_gate_up`] and [`ExpertRowKernels::rows_down`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows_experts(
        &self,
        stream: &Stream,
        shape: &ExpertShape,
        n_rows: u32,
        tile: u32,
        way: Accumulate,
        addresses: DevicePtr,
        plan: DevicePtr,
        act: DevicePtr,
        hidden: DevicePtr,
        scratch: DevicePtr,
        out: DevicePtr,
    ) -> Result<()> {
        // SAFETY: the caller's guarantees are those of the functions called.
        unsafe {
            self.rows_gate_up(stream, shape, n_rows, tile, addresses, plan, act, hidden)?;
            self.rows_down(
                stream, shape, n_rows, tile, way, addresses, plan, hidden, out, scratch,
            )?;
            if way == Accumulate::Ordered {
                self.rows_combine(stream, shape, n_rows, tile, plan, scratch, out)?;
            }
        }
        Ok(())
    }
}

impl ExpertShape {
    /// The route record of the decode kernels for up to [`MAX_BATCH`] rows, written on the
    /// host instead of by the router: the header, the ids and weights (every pick of a row
    /// with the same weight), and the groups (each picked expert once, ascending, with the
    /// rows that picked it and the slot `slot_of` gives it). `picks` has `top_k` experts for
    /// each row. For measurements, which need picks and no router.
    pub fn host_record(&self, picks: &[Vec<u32>], slot_of: &[i32]) -> Vec<u8> {
        decode_record(self, picks, slot_of)
    }
}

fn decode_record(shape: &ExpertShape, picks: &[Vec<u32>], slot_of: &[i32]) -> Vec<u8> {
    let k = shape.top_k as usize;
    let cap = picks.len() * k;
    let mut bytes = vec![0u8; shape.record_bytes(picks.len() as u32) as usize];
    let mut put = |at: usize, v: u32| bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let mut groups: std::collections::BTreeMap<u32, (u32, [f32; MAX_BATCH as usize])> =
        Default::default();
    for (b, ids) in picks.iter().enumerate() {
        for (j, id) in ids.iter().enumerate() {
            let weight = 1.0 / k as f32;
            put(HEADER_BYTES + 4 * (b * k + j), *id);
            put(HEADER_BYTES + 4 * cap + 4 * (b * k + j), weight.to_bits());
            let entry = groups.entry(*id).or_insert((0, [0.0; MAX_BATCH as usize]));
            entry.0 |= 1 << b;
            entry.1[b] = weight;
        }
    }
    put(4, groups.len() as u32);
    put(12, cap as u32);
    let mut first_pair = 0;
    for (g, (expert, (mask, weight))) in groups.iter().enumerate() {
        let at = HEADER_BYTES + 8 * cap + GROUP_BYTES * g;
        put(at, *expert);
        put(at + 4, slot_of[*expert as usize] as u32);
        put(at + 8, *mask);
        put(at + 12, first_pair);
        for (b, w) in weight.iter().enumerate() {
            put(at + 16 + 4 * b, w.to_bits());
        }
        first_pair += mask.count_ones();
    }
    bytes
}

impl ExpertRowKernels {
    /// The tile for a call of `n_rows` rows: the size of the items the plan cuts, which
    /// decides how many rows of activations a block of the tile kernels takes.
    ///
    /// The choice follows the number of pairs an expert gets on average, `n_rows * top_k /
    /// n_experts`: below 2.5 the smallest tile (an expert is picked by a row or two and a
    /// wider tile only adds columns of zeros to the tensor cores' work), below 10 the second
    /// (16), below 40 the third (32), else 64. Those limits are where the tiles crossed
    /// when timed on the recorded routes of Qwen3.8-Flash-Next on an RTX 4070 and an RTX
    /// 3060 (`examples/expert_rows_bench.rs`); they depend on the density of the routes
    /// and not on the card, which only shifts the times of all tiles together.
    pub fn tile_for(shape: &ExpertShape, n_rows: u32) -> u32 {
        let per_expert =
            f64::from(n_rows) * f64::from(shape.top_k) / f64::from(shape.n_experts.max(1));
        let index = [2.5, 10.0, 40.0].iter().filter(|limit| per_expert >= **limit).count();
        ROWS_TILES[index]
    }

    /// Measures, on this device and for this shape, up to how many rows the decode
    /// kernels ([`ExpertKernels::experts`]) are faster than these: the largest `n` of
    /// 1 to [`MAX_BATCH`] such that for every row count up to `n` the decode kernels take
    /// no longer for the experts of the rows. 0 means the many-row kernels are faster
    /// from one row on. A caller that wants the best of both families routes the
    /// windows of up to that many rows through the decode kernels.
    ///
    /// The measurement is a few milliseconds of kernels on a temporary layer of up to
    /// `MAX_BATCH * top_k` experts (random bytes, activations of zeros: the time of the
    /// kernels does not depend on the values) that picks different experts for every
    /// row, which is as much weight traffic as the rows can cause. It needs about 12
    /// MB of device memory per expert of the shape's size and 100 ms; do it once, when
    /// the model is loaded.
    pub fn few_rows_break_even(&self, decode: &ExpertKernels, shape: &ExpertShape) -> Result<u32> {
        shape.check()?;
        let device = &self.device;
        let stream = Stream::new(device, 0)?;
        let max_experts = (MAX_BATCH * shape.top_k).min(shape.n_experts);
        let stride =
            shape.slot_bytes().ok_or_else(|| invalid("expert shape", "sizes overflow".into()))?;
        let pool = crate::memory::DeviceBuffer::new(device, u64::from(max_experts) * stride)?;
        let buffer = |bytes: u64| crate::memory::DeviceBuffer::new(device, bytes.max(16));
        let table = buffer(8 * u64::from(shape.n_experts))?;
        let slot_table = buffer(4 * u64::from(shape.n_experts))?;
        let mut addresses = vec![0u64; shape.n_experts as usize];
        let mut slots = vec![-1i32; shape.n_experts as usize];
        for e in 0..max_experts {
            addresses[e as usize] = pool.ptr().as_raw() + u64::from(e) * stride;
            slots[e as usize] = e as i32;
        }
        table.write(0, &addresses.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>())?;
        slot_table.write(0, &slots.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())?;
        let n = u64::from(shape.n_embd);
        let decode_act =
            buffer(crate::Kernels::quantized_activation_bytes(shape.n_embd, MAX_BATCH))?;
        let decode_hidden = buffer(shape.hidden_bytes(MAX_BATCH))?;
        let decode_out = buffer(4 * n * u64::from(MAX_BATCH))?;
        let act = buffer(shape.rows_activation_bytes(MAX_BATCH))?;
        let hidden = buffer(shape.rows_hidden_bytes(MAX_BATCH))?;
        let scratch = buffer(shape.rows_scratch_bytes(MAX_BATCH))?;
        let out = buffer(4 * n * u64::from(MAX_BATCH))?;
        let tile = ROWS_TILES[0];
        let plan = buffer(shape.rows_plan_bytes(MAX_BATCH, tile))?;
        // SAFETY: every buffer is live, and `len` is its size.
        unsafe {
            for (buf, len) in [
                (&pool, u64::from(max_experts) * stride),
                (&decode_act, decode_act.len()),
                (&act, act.len()),
                (&decode_out, decode_out.len()),
                (&out, out.len()),
                (&hidden, hidden.len()),
                (&decode_hidden, decode_hidden.len()),
                (&scratch, scratch.len()),
            ] {
                stream.fill(buf.ptr(), 0, len as usize)?;
            }
        }
        // Scales of the pool's blocks must be finite: zeros are.
        let (start, end) = (Event::timed(device)?, Event::timed(device)?);
        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            let reps = 20u32;
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                start.record(&stream)?;
                for _ in 0..reps {
                    f()?;
                }
                end.record(&stream)?;
                end.synchronize()?;
                best = best.min(f64::from(end.elapsed_ms_since(&start)?) / f64::from(reps));
            }
            Ok(best)
        };
        let mut break_even = 0;
        for n_rows in 1..=MAX_BATCH {
            // Row r picks the experts r * top_k ...: no expert twice.
            let picks: Vec<Vec<u32>> = (0..n_rows)
                .map(|r| (0..shape.top_k).map(|j| (r * shape.top_k + j) % max_experts).collect())
                .collect();
            let old_record = decode_record(shape, &picks, &slots);
            let old_record_buf = buffer(old_record.len() as u64)?;
            old_record_buf.write(0, &old_record)?;
            let mut record = vec![0u8; HEADER_BYTES];
            record.extend(picks.iter().flatten().flat_map(|id| id.to_le_bytes()));
            record.extend(
                picks.iter().flatten().flat_map(|_| (1.0 / shape.top_k as f32).to_le_bytes()),
            );
            let record_buf = buffer(record.len() as u64)?;
            record_buf.write(0, &record)?;
            // SAFETY: the buffers are sized for this shape and number of rows, and every
            // picked expert has a slot and an address.
            let old = time(&mut || unsafe {
                decode.experts(
                    &stream,
                    shape,
                    pool.ptr(),
                    slot_table.ptr(),
                    old_record_buf.ptr(),
                    decode_act.ptr(),
                    decode_hidden.ptr(),
                    decode_out.ptr(),
                    None,
                    n_rows,
                    None,
                )
            })?;
            // SAFETY: as above.
            let new = time(&mut || unsafe {
                self.plan_rows(&stream, shape, n_rows, tile, record_buf.ptr(), plan.ptr(), 0)?;
                self.rows_experts(
                    &stream,
                    shape,
                    n_rows,
                    tile,
                    Accumulate::Ordered,
                    table.ptr(),
                    plan.ptr(),
                    act.ptr(),
                    hidden.ptr(),
                    scratch.ptr(),
                    out.ptr(),
                )
            })?;
            if old > new {
                break;
            }
            break_even = n_rows;
        }
        Ok(break_even)
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
    fn every_weight_type_is_taken_and_a_row_must_be_whole_blocks() {
        assert_eq!(ExpertShape::types().count(), 28);
        for ty in ExpertShape::types() {
            // Gate and up: rows of 2560 values are whole blocks of every type.
            assert!(flash_next(ty).check().is_ok(), "gate and up of {ty}");
            assert!(flash_next(ty).check_rows().is_ok(), "gate and up of {ty}, many rows");
            // Down: rows of 640 values are whole blocks of the types of up to 128 values.
            let mut shape = flash_next(GgmlType::IQ2_S);
            shape.down_ty = ty;
            assert_eq!(shape.check().is_ok(), 640 % ty.block_values() == 0, "down of {ty}");
            shape.n_ff = 768;
            assert!(shape.check_rows().is_ok(), "down of {ty} with rows of 768 values");
            let (gate_up, down) = rows_table_bytes(ty);
            assert!(gate_up.is_multiple_of(16) && down.is_multiple_of(16), "{ty}");
        }
        // Mixed: any pair of types whose blocks divide the widths.
        let mut shape = flash_next(GgmlType::Q4_K);
        shape.down_ty = GgmlType::BF16;
        assert!(shape.check().is_ok());
        // What holds no weights.
        for ty in [GgmlType::I8, GgmlType::I32, GgmlType::F64, GgmlType::Q8_K, GgmlType::Q8_1] {
            assert!(flash_next(ty).check().is_err(), "{ty}");
        }
        let mut shape = flash_next(GgmlType::IQ2_S);
        shape.top_k = 17;
        assert!(shape.check().is_err());
        shape.top_k = 10;
        shape.n_ff = 600;
        assert!(shape.check().is_err());
        // Widths the decode kernels take and the many-row kernels do not.
        let narrow = ExpertShape { n_embd: 96, n_ff: 32, ..flash_next(GgmlType::Q4_0) };
        let narrow = ExpertShape { down_ty: GgmlType::Q4_0, ..narrow };
        assert!(narrow.check().is_ok() && narrow.check_rows().is_err());
    }

    #[test]
    fn every_type_has_its_many_row_kernels_in_one_module() {
        for (ty, suffix) in EXPERT_TYPES {
            for down in [false, true] {
                let holders = ROWS_MODULES
                    .iter()
                    .map(|(_, gate_up, downs)| if down { downs } else { gate_up })
                    .filter(|types| types.contains(&suffix))
                    .count();
                assert_eq!(holders, 1, "{ty} ({})", if down { "down" } else { "gate and up" });
            }
        }
    }

    #[test]
    fn the_plan_layout_is_what_the_kernels_compute() {
        // 1024 rows of ten picks over 512 experts, items of 32: the numbers are those
        // of `ares_er_layout` in kernels/expertrows_layout.cuh worked out by hand (the
        // device tests then check that the kernels write where the host reads).
        let layout = RowsPlanLayout::new(512, 10240, 32);
        assert_eq!(layout.max_items(), 512 + 320);
        assert_eq!(layout.items_at(), 2128);
        assert_eq!(layout.pair_row_at(), 2128 + 16 * 832);
        assert_eq!(layout.missing_at(), 138_320);
        assert_eq!(layout.bytes, 140_368 + 4 * 10 * 512);
        // Up to 1024 pairs there is one block and no block counts behind the plan.
        let small = RowsPlanLayout::new(512, 80, 8);
        assert_eq!(small.max_items(), 80 + 10);
        assert_eq!(small.bytes, small.block_counts_at());
    }

    #[test]
    fn the_sizes_of_the_buffers_of_many_rows_follow_the_shape() {
        assert_eq!(flash_next(GgmlType::IQ2_S).rows_hidden_bytes(2), 2 * 640 * 20 + 10 * 4 * 20);
        assert_eq!(flash_next(GgmlType::IQ2_S).rows_scratch_bytes(2), 4 * 2560 * 20);
    }

    #[test]
    fn the_tile_follows_the_pairs_per_expert() {
        let shape = flash_next(GgmlType::IQ2_S);
        let tiles: Vec<u32> = [1, 8, 64, 256, 1024, 4096, 16384]
            .iter()
            .map(|rows| ExpertRowKernels::tile_for(&shape, *rows))
            .collect();
        assert_eq!(tiles, [8, 8, 8, 16, 32, 64, 64]);
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
