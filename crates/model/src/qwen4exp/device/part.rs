//! One device's part of the model: a range of layers with their dense
//! weights, the recurrent state of every sequence slot, the K/V and key pools
//! of its attention layers, the scratch of a step, and the kernels of a step.
//!
//! A step is enqueued layer by layer on the device's stream and nothing here
//! waits for the device. A layer is, in the order of section 3 of the
//! specification:
//!
//! ```text
//! (layer with the n-gram block)  decode the fetched rows, key and value, gate, window, commit
//! read                           stream norm, down, squash, up, inject, mix
//! recurrent or attention         window, and the commit of the rows that are certain
//! write + read                   one pass over the residual
//! router                         logits, picks, the route record (copied to the host on a side stream)
//! shared expert                  gate, up, down, the scalar gate
//! routed experts                 by id through the slot table; a missing one is waited for on the device
//! ```
//!
//! The write of a layer's mixture of experts is fused with the norm of the
//! next layer's read ([`Part::flush`] does it alone where a caller needs the
//! residual between layers).
//!
//! A read of a stream mixer is two launches (`Q4Kernels::read_down`,
//! `read_up`) when the mixer's matrices are float types, which they are in
//! every file seen so far; the first of them also writes the output of the
//! half block before it, so the residual alternates between two buffers.
//!
//! **Two devices on every layer.** A part is then a *mixer* or a *worker*
//! ([`Role`]). The mixer holds the streams and runs every read; its read
//! writes the mixed input of the half block into the worker's memory. The
//! worker computes its share of the half (key heads of a recurrent layer, the
//! shared expert, the routed experts it owns) and writes its partial output
//! into the mixer's memory, where the next read waits for it and adds it to
//! the mixer's own partial. The router runs on the mixer, whose split kernel
//! gives each device the record of the experts it owns. Nothing of this
//! passes through the host.
//!
//! Windows and commits: rows of a segment that is *direct* (a slice of a
//! prompt, a generated token without drafts) advance the recurrent state in
//! their own launches. Rows of any other segment are a window that leaves
//! the state untouched; the window keeps what a commit needs, and
//! [`Part::enqueue_commit`] replays the accepted prefix before the next
//! step.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{
    Arrival, AttendBuffers, AttentionKernels, AttentionShape, AttnItem, AttnRow, AttnShape,
    CacheFormat, CacheGeometry, CommitSet, Departure, Device, DeviceBuffer, DevicePtr, ENTRY_CELLS,
    Event, ExpertKernels, IndexerShape, Kernels, KeyFormat, LayerKernels, ListHandOver, MAX_BATCH,
    MixerMatrix, NgramShape, OutputGate, PinnedBuffer, Plan, PlanConfig, Planner, Q4Kernels,
    ReadWrite, RopeParams, Segment, SelectionRule, StateFormat, StepRecord, Stream,
};
use ares_runtime::ModelError;
use ares_runtime::expert_store::cpu::act_bytes as cpu_act_bytes;

use super::load::{
    AttentionW, CoreW, DeviceWeights, LayerShare, LayerW, Mat, MixerW, MoeW, NgramW, RecurrentW,
    Role, SharedW,
};
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

/// Bit 31 of a table entry: the page is in the pool of stored formats (the
/// only pool the formats without a tile mean use).
pub const PAGE_ENTRY: u32 = 1 << 31;

/// The most pieces a row's device-made list is cut into.
const MAX_PIECES: u32 = 16;

/// The choices of a device implementation.
#[derive(Clone, Debug)]
pub struct Options {
    /// The formats of K and of V. Only the formats without a tile mean (`f16`,
    /// `q8`) are supported: the others need the exact pool and its conversion.
    pub kv: (CacheFormat, CacheFormat),
    /// How the pooled keys of the indexer are stored.
    pub keys: KeyFormat,
    /// How the delta-rule matrices are stored.
    pub state: StateFormat,
    /// How a row past the dense limit picks its cells.
    pub rule: SelectionRule,
    /// The most rows of a step, at most [`MAX_BATCH`]: the expert kernels
    /// take no more.
    pub max_rows: u32,
    /// The most rows of a step whose logits are wanted.
    pub logit_rows: u32,
    /// The most rows of a step that verifies drafts (a window whose rows may
    /// be rejected keeps its inputs per layer); 0 for none.
    pub window_rows: u32,
    /// Sequences at once: the slots of recurrent state.
    pub max_slots: u32,
    /// The most cells of a sequence: decides the K/V and key pools.
    pub cells: u32,
    /// Where the parts add up the device time of their kernels, by kind.
    pub profile: Option<Arc<Profile>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            kv: (CacheFormat::Q8, CacheFormat::Q8),
            keys: KeyFormat::F32,
            state: StateFormat::F32,
            rule: SelectionRule::CellBudget,
            max_rows: MAX_BATCH,
            logit_rows: 1,
            window_rows: 0,
            max_slots: 1,
            cells: 4096,
            profile: None,
        }
    }
}

impl Options {
    /// Table entries per sequence.
    pub fn table_stride(&self) -> u32 {
        self.cells.div_ceil(ENTRY_CELLS).max(1)
    }

    /// Pages of the pools: every sequence slot owns its pages from the start.
    pub fn pages(&self) -> u32 {
        self.max_slots * self.table_stride()
    }
}

/// What a stretch of a step's kernels is, for [`Profile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Products with the dense weights of the recurrent and attention
    /// layers, the shared experts and the output head, and the preparation
    /// of their activations.
    Dense = 0,
    /// The stream mixers: norm, write, squash, mix and their three products
    /// (down, up, inject) per read.
    Mixers = 1,
    /// Convolution, delta rule and gated norm of the recurrent layers.
    Recurrent = 2,
    /// The attention path: preparing q and k, writing K and V, the indexer,
    /// block scores, selection, hand-over and attention.
    Attention = 3,
    /// Router and routed experts, with the wait for experts that were not in
    /// a slot.
    Experts = 4,
    /// Hand-overs between devices and the copies of a step's inputs.
    Exchange = 5,
    /// Everything else: the n-gram block's kernels, the gather for the head.
    Other = 6,
    /// The stream idle in front of the n-gram block: the host had not yet
    /// received the rows of the table.
    Rows = 7,
}

/// The phases, in the order of their numbers.
pub const PHASES: [Phase; 8] = [
    Phase::Dense,
    Phase::Mixers,
    Phase::Recurrent,
    Phase::Attention,
    Phase::Experts,
    Phase::Exchange,
    Phase::Other,
    Phase::Rows,
];

/// Device time per phase and device, added up over steps.
#[derive(Debug, Default)]
pub struct Profile {
    inner: Mutex<Vec<(u64, [f64; 8])>>,
    /// Steps are not timed while this is set: the timed events cost a part
    /// of a step themselves.
    paused: AtomicBool,
}

impl Profile {
    /// An empty profile.
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&self, device: usize, ms: &[f64; 8]) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.len() <= device {
            inner.resize(device + 1, (0, [0.0; 8]));
        }
        inner[device].0 += 1;
        for (total, ms) in inner[device].1.iter_mut().zip(ms) {
            *total += ms;
        }
    }

    /// Per device: the steps counted and the milliseconds per phase, in the
    /// order of [`PHASES`].
    pub fn totals(&self) -> Vec<(u64, [f64; 8])> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Forgets everything counted so far.
    pub fn reset(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Stops (`true`) or resumes the timing of steps, from the next step on.
    pub fn pause(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
    }
}

struct Marks {
    profile: Arc<Profile>,
    /// Whether the step being enqueued is timed.
    on: bool,
    events: Vec<Event>,
    /// The phase that ends at each event recorded in this step (the first
    /// event starts the step).
    phases: Vec<Phase>,
}

/// One step, as a part needs it.
#[derive(Clone, Copy)]
pub struct StepEnv<'a> {
    /// The stream the step is enqueued on.
    pub stream: &'a Stream,
    /// The step record: segments, tokens, positions, the rows whose logits
    /// are wanted.
    pub record: &'a StepRecord,
    /// The segments of the record.
    pub segments: &'a [Segment],
    /// A number that differs from step to step: it marks the route records.
    pub token: u32,
    /// The token embeddings of the rows, `rows * n_embd` values: for the
    /// part that holds the first layer.
    pub x: &'a [f32],
    /// The stored bytes of the n-gram table rows of every row, in head
    /// order: for the layer with the n-gram block, when it is enqueued.
    pub ngram: Option<&'a [u8]>,
    /// Where the residual of the previous device arrives on this device (the
    /// stream waits for it already), for a part that does not hold the first
    /// layer.
    pub inbound: Option<DevicePtr>,
    /// The number of the step's first exchange between a mixer and its
    /// worker; the exchanges of a step are numbered from it, alike on both.
    pub xbase: u32,
}

impl std::fmt::Debug for StepEnv<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepEnv").field("dims", &self.record.dims()).finish_non_exhaustive()
    }
}

/// Where a layer's experts are on the device: its slot table, its pool and
/// its host share, all the expert store's.
#[derive(Clone, Copy, Debug)]
pub struct ExpertAt {
    /// The layer's slot table (`n_experts` i32).
    pub table: DevicePtr,
    /// The pool the layer's slots are in.
    pub pool: DevicePtr,
    /// Where the host claims experts it computes itself and delivers their
    /// sums (`ares_cuda::HostShare`); `None` when every missing expert is
    /// uploaded.
    pub host: Option<DevicePtr>,
}

/// Where a recurrent layer keeps its state and what a draft window keeps.
#[derive(Clone, Copy, Debug)]
struct RecurrentAt {
    /// Offsets in a slot of the state arena.
    conv: u64,
    delta: u64,
    /// Offsets in the arena of kept inputs: qkv, kv, alpha, beta.
    kept: [u64; 4],
}

/// Where an attention layer has its pools and state.
#[derive(Clone, Copy, Debug)]
struct AttentionAt {
    /// Offsets in the pool arena.
    kv_pool: u64,
    key_pool: u64,
    /// Offset of the carried indexer state in a slot of the state arena.
    carry: u64,
    /// Offset in the arena of kept inputs: the raw indexer keys of a window.
    kept_raw: u64,
}

/// What the peer of a part needs to know of it: where the kernels of the peer
/// write what they send, in this part's memory.
#[derive(Clone, Copy, Debug)]
pub struct Ports {
    /// The flags that announce arrivals (`u32`): one per row for a partial
    /// output, one per block of the read for a mixed input.
    pub flags: DevicePtr,
    /// The flag of a route record.
    pub record_flag: DevicePtr,
    /// Where a partial output arrives: rows of `n_embd` f32.
    pub partial: DevicePtr,
    /// Where the mixed input of a recurrent half arrives, as activations.
    pub m: DevicePtr,
    /// Where the mixed input of each layer's mixture of experts arrives, and
    /// the bytes from one layer to the next.
    pub acts: DevicePtr,
    pub act_stride: u64,
    /// Where each layer's route record arrives, and the bytes between them.
    pub records: DevicePtr,
    pub record_stride: u64,
}

/// The exchanges of a layer between a mixer and its worker, in the order
/// they happen: the number of an exchange is the step's base plus eight
/// times the layer plus one of these.
const X_CORE_INPUT: u32 = 0;
const X_CORE_PARTIAL: u32 = 1;
const X_MOE_INPUT: u32 = 2;
const X_RECORD: u32 = 3;
const X_MOE_PARTIAL: u32 = 4;

/// The output of a half block that is not yet in the residual (mixer, whole
/// part) or not yet sent (worker).
#[derive(Clone, Copy, Debug)]
enum Pending {
    /// The recurrent or attention half: whether this part computed a share
    /// of it, and the exchange in which the peer's share crosses.
    Core { own: bool, exchange: Option<u32> },
    /// The mixture of experts: the exchange in which the peer's share
    /// crosses, and whether this part computed the shared expert.
    Moe { exchange: Option<u32>, shared: bool },
}

/// What a part is and holds: its role and, per layer from `first` on, its
/// share.
#[derive(Clone, Debug)]
pub(crate) struct PartShape {
    pub role: Role,
    pub first: usize,
    pub shares: Vec<LayerShare>,
    pub head: bool,
}

impl PartShape {
    /// Whole layers `layers`, as a split by layers gives them to a device.
    pub(crate) fn whole(c: &Config, layers: Range<usize>, head: bool) -> Self {
        Self {
            role: Role::Whole,
            first: layers.start,
            shares: layers.map(|l| LayerShare::whole(c, l)).collect(),
            head,
        }
    }
}

/// Offsets of the step scratch.
#[derive(Clone, Copy, Debug, Default)]
struct Scratch {
    r: u64,
    /// The second buffer of the residual: a read that writes goes from one
    /// to the other.
    r2: u64,
    /// The partial sums between the two launches of a read.
    partials: u64,
    /// The router's record before it is cut between two devices.
    rec_full: u64,
    zeros: u64,
    xn: u64,
    lo: u64,
    down: u64,
    gate: u64,
    inj_a: u64,
    inj_f: u64,
    m: u64,
    p: u64,
    out: u64,
    // Recurrent and attention: two wide buffers and the smaller ones.
    w1: u64,
    w2: u64,
    alpha: u64,
    beta: u64,
    q: u64,
    kv: u64,
    o: u64,
    k: u64,
    v: u64,
    agate: u64,
    iq: u64,
    raw: u64,
    rope: u64,
    scores: u64,
    counts: u64,
    act: u64,
    attn_state: u64,
    status: u64,
    // Mixture of experts.
    logits: u64,
    hidden: u64,
    sg: u64,
    su: u64,
    ysh: u64,
    s: u64,
    // n-gram block.
    emb: u64,
    key: u64,
    value: u64,
    gated: u64,
    nrm: u64,
    // Head.
    picked: u64,
    act_p: u64,
    head: u64,
    len: u64,
}

/// Offsets of what the host sends with a step.
#[derive(Clone, Copy, Debug, Default)]
struct Inbox {
    record: u64,
    record_cap: u64,
    x: u64,
    plan: u64,
    plan_cap: u64,
    hands: u64,
    ngram: u64,
    /// The record of the step before, with its accepted counts.
    commit: u64,
    len: u64,
}

/// Every size and offset of a part: what [`Part::open`] allocates and what a
/// placement counts.
#[derive(Clone, Debug)]
pub(crate) struct Frame {
    slot_bytes: u64,
    recurrent: Vec<Option<RecurrentAt>>,
    attention: Vec<Option<AttentionAt>>,
    ngram_history: Option<u64>,
    pool_bytes: u64,
    kept_bytes: u64,
    s: Scratch,
    inbox: Inbox,
    record_stride: u64,
    /// Bytes from one layer's input of the mixture of experts to the next,
    /// on the device (the whole activations) and on the host (what the CPU
    /// reads of them).
    act_stride: u64,
    act_host_stride: u64,
    geo: CacheGeometry,
    ix: IndexerShape,
    ng: Option<NgramShape>,
    plan_config: PlanConfig,
    /// The flags of the exchange with a peer and the bytes of the exchange
    /// buffer (flags, then a partial output); 0 for a whole part.
    x_flags: u32,
    x_bytes: u64,
    /// Bytes of the table that says which experts the peer owns.
    owner_bytes: u64,
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

fn act_bytes(n_in: u64, rows: u64) -> u64 {
    Kernels::quantized_activation_bytes(n_in as u32, rows as u32)
}

/// The K/V geometry of an attention layer: every KV head on the layer's
/// device.
pub(crate) fn geometry(c: &Config, kv: (CacheFormat, CacheFormat)) -> CacheGeometry {
    let a = &c.attention;
    CacheGeometry {
        k: kv.0,
        v: kv.1,
        head_dim: a.head_dim as u32,
        kv_heads: a.n_head_kv as u32,
        kv_total: a.n_head_kv as u32,
        first_kv_head: 0,
        q_per_kv: (a.n_head / a.n_head_kv) as u32,
    }
}

fn indexer_shape(c: &Config) -> IndexerShape {
    IndexerShape {
        n_head: c.indexer.n_head as u32,
        key_dim: c.indexer.key_dim as u32,
        block: c.indexer.block as u32,
        top_k: c.indexer.top_k as u32,
        rope_dim: c.attention.rope.dim as u32,
    }
}

fn ngram_shape(c: &Config) -> Option<NgramShape> {
    c.ngram.as_ref().map(|g| NgramShape {
        n_stream: c.n_stream as u32,
        n_embd: c.n_embd as u32,
        heads: g.n_heads() as u32,
        row_dim: g.row_dim as u32,
        conv_kernel: g.conv_kernel as u32,
        dilation: g.ngram_size as u32,
    })
}

impl Frame {
    /// The sizes of a part of `shape` on a device with `multiprocessors`
    /// multiprocessors (the attention plans are cut by it).
    pub(crate) fn new(
        c: &Config,
        opts: &Options,
        shape: &PartShape,
        multiprocessors: u32,
    ) -> Result<Self, ModelError> {
        let head = shape.head;
        let streams = shape.role != Role::Worker;
        if opts.kv.0.has_mean() || opts.kv.1.has_mean() {
            return Err(invalid(
                "K/V formats with a tile mean (q7 to q4) need the exact pool and its conversion, \
                 which this implementation does not manage yet: use f16 or q8",
            ));
        }
        if opts.max_rows == 0 || opts.max_rows > MAX_BATCH {
            return Err(invalid(format!(
                "steps of up to {} rows: the expert kernels take 1 to {MAX_BATCH}",
                opts.max_rows
            )));
        }
        if opts.logit_rows == 0 || opts.logit_rows > opts.max_rows || opts.max_slots == 0 {
            return Err(invalid(
                "a part needs a logit row (at most the rows of a step) and a slot",
            ));
        }
        if opts.window_rows > opts.max_rows {
            return Err(invalid("a draft window has at most the rows of a step"));
        }
        let (rows, window) = (u64::from(opts.max_rows), u64::from(opts.window_rows));
        let (n, hc) = (c.n_embd as u64, c.n_stream as u64);
        let (wide, rank) = (c.wide() as u64, c.mixer_rank as u64);
        let geo = geometry(c, opts.kv);
        let ix = indexer_shape(c);
        let ng = ngram_shape(c);
        let plan_config = PlanConfig::for_multiprocessors(multiprocessors);
        let pages = u64::from(opts.pages());

        let r = &c.recurrent;
        let (cd, kd, vd, nv) =
            (r.conv_dim() as u64, r.key_dim() as u64, r.value_dim() as u64, r.n_v_heads as u64);
        let (hq, hkv) = c.attention_dims();
        let (hq, hkv) = (hq as u64, hkv as u64);
        let iq = (c.indexer.n_head * c.indexer.key_dim) as u64;
        let id = c.indexer.key_dim as u64;

        let (mut slot_bytes, mut pool_bytes, mut kept_bytes) = (0u64, 0u64, 0u64);
        let mut recurrent = Vec::new();
        let mut attention = Vec::new();
        let mut ngram_history = None;
        let (mut any_recurrent, mut any_attention, mut any_ngram) = (false, false, false);
        let mut keep = |bytes: u64| {
            let at = kept_bytes;
            kept_bytes = (kept_bytes + window * bytes).next_multiple_of(64);
            at
        };
        for (i, share) in shape.shares.iter().enumerate() {
            let layer = shape.first + i;
            let heads = u64::from(share.heads.end.saturating_sub(share.heads.start));
            let (mut r_at, mut a_at) = (None, None);
            if heads == 0 {
                // The layer's heads are all on the peer.
            } else if c.is_attention(layer) {
                any_attention = true;
                let kv_pool = pool_bytes;
                let key_pool = (kv_pool + pages * geo.converted_entry_bytes()).next_multiple_of(64);
                pool_bytes = (key_pool + pages * ix.key_slot_bytes(opts.keys)).next_multiple_of(64);
                let carry = slot_bytes;
                slot_bytes = (carry + ix.carry_len() * 4).next_multiple_of(64);
                a_at = Some(AttentionAt { kv_pool, key_pool, carry, kept_raw: keep(id * 4) });
            } else {
                any_recurrent = true;
                // The part's key heads with their value heads: a narrower
                // layer.
                let s = r.head_dim as u64;
                let values = heads * (r.n_v_heads / r.n_k_heads) as u64;
                let channels = (2 * heads + values) * s;
                let conv = slot_bytes;
                let delta = (conv + channels * (r.conv_kernel as u64 - 1) * 4).next_multiple_of(64);
                slot_bytes = (delta + values * s.pow(2) * opts.state.bytes()).next_multiple_of(64);
                let kept = [
                    keep(channels * 4),
                    keep((heads + values) * s * 4),
                    keep(values * 4),
                    keep(values * 4),
                ];
                r_at = Some(RecurrentAt { conv, delta, kept });
            }
            if streams && c.ngram.as_ref().is_some_and(|g| g.layer == layer) {
                any_ngram = true;
                let at = slot_bytes;
                let history = ng.as_ref().map_or(0, NgramShape::history_len);
                slot_bytes = (at + history * 4).next_multiple_of(64);
                ngram_history = Some(at);
            }
            recurrent.push(r_at);
            attention.push(a_at);
        }

        // The scratch of a step.
        let mut s = Scratch::default();
        let mut len = 0u64;
        let mut take = |bytes: u64| {
            let at = len;
            len = (len + bytes.max(16)).next_multiple_of(64);
            at
        };
        s.r = take(rows * wide * 4);
        s.r2 = take(u64::from(streams) * rows * wide * 4);
        s.partials = take(
            Q4Kernels::read_partials(opts.max_rows, c.n_stream as u32, c.mixer_rank as u32) * 4,
        );
        s.zeros = take(rows * hc * 4);
        s.xn = take(act_bytes(wide, rows));
        s.lo = take(act_bytes(rank, rows));
        s.down = take(rows * rank * 4);
        s.gate = take(rows * wide * 4);
        s.inj_a = take(rows * hc * 4);
        s.inj_f = take(rows * hc * 4);
        s.m = take(act_bytes(n, rows));
        s.p = take(rows * n * 4);
        s.out = take(rows * n * 4);
        let (rec, att) = (u64::from(any_recurrent), u64::from(any_attention));
        s.w1 = take(rows * (rec * cd).max(att * 2 * hq) * 4);
        s.w2 = take(rows * (rec * vd).max(att * hkv) * 4);
        s.alpha = take(rows * rec * nv * 4);
        s.beta = take(rows * rec * nv * 4);
        s.q = take(rows * (rec * kd).max(att * hq) * 4);
        s.kv = take(rows * rec * (kd + vd) * 4);
        s.o = take(rows * (rec * vd).max(att * hq) * 4);
        s.k = take(rows * att * hkv * 4);
        s.v = take(rows * att * hkv * 4);
        s.agate = take(rows * att * hq * 4);
        s.iq = take(rows * att * iq * 4);
        s.raw = take(rows * att * id * 4);
        s.rope = take(rows * att * (c.attention.rope.dim as u64 / 2) * 8);
        let max_blocks = u64::from(opts.cells.div_ceil(ix.block)).max(1);
        s.scores = take(rows * att * max_blocks * 4);
        s.counts = take(rows * 16);
        let shared_ff = c.experts.shared_ff as u64;
        s.act = take(act_bytes((rec * vd).max(att * hq).max(shared_ff), rows));
        // The most partial states a plan makes: the planner's own bound for
        // dense rows (`Planner::finish`) and the pieces of device-made lists.
        let state_slots = (20 * u64::from(plan_config.target_blocks) / u64::from(geo.kv_heads)
            + 64)
            .max(rows * u64::from(MAX_PIECES));
        // A state is, per query head, the weighted V and the log of the sum.
        let state = c.attention.n_head as u64 * (u64::from(geo.head_dim) + 1) * 4;
        s.attn_state = take(att * state_slots * state);
        s.status = take(16);
        let expert_shape = ares_cuda::ExpertShape {
            n_embd: n as u32,
            n_ff: c.experts.ff as u32,
            n_experts: c.experts.count as u32,
            top_k: c.experts.used as u32,
            gate_ty: GgmlType::IQ2_S,
            down_ty: GgmlType::Q2_0,
        };
        s.rec_full = take(expert_shape.record_bytes(opts.max_rows));
        s.logits = take(expert_shape.logits_bytes(opts.max_rows));
        s.hidden = take(expert_shape.hidden_bytes(opts.max_rows));
        s.sg = take(rows * shared_ff * 4);
        s.su = take(rows * shared_ff * 4);
        s.ysh = take(rows * n * 4);
        s.s = take(rows * 4);
        let gram = u64::from(any_ngram);
        s.emb = take(gram * act_bytes(n, rows));
        s.key = take(gram * rows * wide * 4);
        s.value = take(gram * rows * n * 4);
        s.gated = take(gram * rows * wide * 4);
        s.nrm = take(gram * rows * wide * 4);
        let logit_rows = u64::from(opts.logit_rows);
        let heads = u64::from(head);
        s.picked = take(heads * logit_rows * n * 4);
        s.act_p = take(heads * act_bytes(n, logit_rows));
        s.head = take(heads * logit_rows * c.n_vocab as u64 * 4);
        s.len = len;

        // A flag per block of a read, and one for a route record.
        let x_flags = Q4Kernels::read_flags(opts.max_rows, c.n_embd as u32) + 16;

        // What the host sends with a step.
        let mut inbox = Inbox::default();
        let mut len = 0u64;
        let mut take = |bytes: u64| {
            let at = len;
            len = (len + bytes.max(16)).next_multiple_of(64);
            at
        };
        // A record: header, a segment per slot, a token and four positions
        // per row, the output rows.
        inbox.record_cap = (64 + 32 * u64::from(opts.max_slots) + 28 * rows).next_multiple_of(64);
        inbox.record = take(inbox.record_cap);
        inbox.x = take(rows * n * 4);
        // A plan: the planner's dense plans are a few dozen bytes per row and
        // item; a plan of device-made lists has `MAX_PIECES` items per row
        // and KV head and three words per range.
        let listed = rows
            * (u64::from(MAX_PIECES) * u64::from(geo.kv_heads) * 48
                + u64::from(ix.max_ranges(0)) * 12
                + 32);
        inbox.plan_cap = att * (listed.max(64 << 10) + 96 * rows + 256).next_multiple_of(64);
        inbox.plan = take(inbox.plan_cap);
        inbox.hands = take(att * rows * ListHandOver::BYTES as u64);
        inbox.ngram = take(gram * rows * ng.as_ref().map_or(0, NgramShape::fetched_row_bytes));
        inbox.commit = take(inbox.record_cap);
        inbox.len = len;

        Ok(Self {
            slot_bytes: slot_bytes.max(64),
            recurrent,
            attention,
            ngram_history,
            pool_bytes: pool_bytes.max(64),
            kept_bytes,
            s,
            inbox,
            record_stride: expert_shape.record_bytes(opts.max_rows).next_multiple_of(64),
            act_stride: act_bytes(n, rows).next_multiple_of(64),
            act_host_stride: (cpu_act_bytes(n as usize, rows as usize) as u64).next_multiple_of(64),
            geo,
            ix,
            ng,
            plan_config,
            x_flags,
            x_bytes: if shape.role == Role::Whole {
                0
            } else {
                (u64::from(x_flags) * 4).next_multiple_of(64) + rows * n * 4
            },
            owner_bytes: if shape.role == Role::Mixer {
                (shape.shares.len() as u64 * c.experts.count as u64).next_multiple_of(64)
            } else {
                0
            },
        })
    }

    /// Bytes of recurrent state per sequence slot.
    pub(crate) fn state_bytes_per_slot(&self) -> u64 {
        self.slot_bytes
    }

    /// Bytes of the K/V pools and key pools.
    pub(crate) fn pool_bytes(&self) -> u64 {
        self.pool_bytes
    }

    /// Bytes of everything a step needs besides weights, state and pools:
    /// scratch, kept window inputs, route records, the step's inputs, tables.
    pub(crate) fn step_bytes(&self, opts: &Options) -> u64 {
        self.s.len
            + self.kept_bytes
            + (self.record_stride + self.act_stride) * self.recurrent.len() as u64
            + self.inbox.len
            + self.tables_bytes(opts)
            + self.x_bytes
            + self.owner_bytes
    }

    fn tables_bytes(&self, opts: &Options) -> u64 {
        (2 * 4 * u64::from(opts.pages())).next_multiple_of(64)
    }
}

/// The attention plan of a step on the device.
struct StepPlan {
    plan: Plan,
    /// The row capacity of a list in ranges, when the lists are made on the
    /// device; `None` for a plan the planner made for dense rows.
    listed: Option<u32>,
    /// The blocks the longest row sees.
    max_blocks: u32,
}

/// One device's part of the model.
pub struct Part {
    index: usize,
    role: Role,
    /// Whether the part starts the residual from the embeddings.
    first: bool,
    /// Whether the reads run as two fused launches.
    fused: bool,
    /// Which of the two residual buffers holds the residual.
    flipped: bool,
    opts: Options,
    device: Device,
    kernels: Kernels,
    layers: LayerKernels,
    q4: Q4Kernels,
    attention: Option<AttentionKernels>,
    experts: ExpertKernels,
    side: Stream,
    w: DeviceWeights,
    f: Frame,
    // Model sizes.
    n: u32,
    hc: u32,
    wide: u32,
    rank: u32,
    eps: f32,
    n_vocab: u32,
    rope: RopeParams,
    attn_shape: AttentionShape,
    scratch: DeviceBuffer,
    state: DeviceBuffer,
    pools: DeviceBuffer,
    tables: DeviceBuffer,
    /// The host copy of the attention tables.
    entries: Vec<u32>,
    kept: Option<DeviceBuffer>,
    records: DeviceBuffer,
    records_host: PinnedBuffer,
    /// Per layer the input rows of its mixture of experts as activations,
    /// and the host's copy of what the CPU reads of them.
    acts: DeviceBuffer,
    acts_host: PinnedBuffer,
    routed: Vec<Event>,
    copied: Vec<Event>,
    inbox_host: PinnedBuffer,
    inbox: DeviceBuffer,
    logits_host: PinnedBuffer,
    expert_at: Vec<ExpertAt>,
    /// The plan of the step being enqueued.
    plan: Option<StepPlan>,
    /// The output of the last half block, not yet written to the residual
    /// (or, on a worker, not yet sent).
    pending: Option<Pending>,
    /// The flags and the partial output a peer writes here.
    exchange: Option<DeviceBuffer>,
    /// Per layer and expert, whether the peer owns the expert (a mixer's).
    owner: Option<DeviceBuffer>,
    /// The peer's ports, and per layer whether the peer computes a share of
    /// the recurrent or attention half.
    peer: Option<(Ports, Vec<bool>)>,
    /// The record of a step whose deferred segments wait for their commit,
    /// and which of its segments were direct.
    deferred: Option<(StepRecord, Vec<bool>)>,
    status_host: PinnedBuffer,
    status_pending: bool,
    marks: Option<Marks>,
}

impl std::fmt::Debug for Part {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Part")
            .field("index", &self.index)
            .field("layers", &self.w.layers.len())
            .field("weights", &self.w.bytes())
            .finish_non_exhaustive()
    }
}

/// What a part is built from.
#[derive(Debug)]
pub struct PartSetup<'a> {
    /// Which part of the placement this is.
    pub index: usize,
    /// What the part is to the residual streams.
    pub role: Role,
    /// The first layer the part holds or shares.
    pub first: usize,
    /// What it holds of each layer from `first` on.
    pub shares: Vec<LayerShare>,
    /// Whether the part holds the final mixer and the output head.
    pub head: bool,
    pub device: &'a Device,
    /// The device's memory accounts.
    pub ledger: &'a mut Ledger,
}

impl Part {
    /// Builds a part: uploads its dense weights and reserves and allocates
    /// its state, pools and scratch.
    pub fn open(
        c: &Config,
        weights: &Weights<'_>,
        opts: &Options,
        setup: PartSetup<'_>,
    ) -> Result<Self, ModelError> {
        let PartSetup { index, role, first, shares, head, device, ledger } = setup;
        if shares.is_empty() || first + shares.len() > c.n_layer {
            return Err(invalid(format!(
                "part {index} holds {} layers from layer {first} on",
                shares.len()
            )));
        }
        let shape = PartShape { role, first, shares, head };
        let f = Frame::new(c, opts, &shape, device.info().multiprocessors)?;
        let w = DeviceWeights::load(device, ledger, c, weights, role, first, &shape.shares, head)?;
        let mut mixer_types = Vec::new();
        let mixers = w.layers.iter().filter_map(|l| l.mixers).flat_map(|(a, b)| [a, b]);
        for mixer in mixers.chain(w.head.map(|h| h.mixer)) {
            mixer_types.extend([mixer.down.ty, mixer.up.ty]);
            mixer_types.extend(mixer.inject.map(|m| m.ty));
        }
        let fused = Q4Kernels::reads_fused(
            c.n_stream as u32,
            c.n_embd as u32,
            c.mixer_rank as u32,
            &mixer_types,
        );
        if role != Role::Whole && !fused {
            return Err(invalid(
                "two devices share layers only when the mixers run as fused reads (float \
                 matrices, a rank that is a multiple of 4 and at most 1024)",
            ));
        }
        let state_bytes = f.slot_bytes * u64::from(opts.max_slots);
        ledger.reserve("qwen4exp recurrent state", state_bytes)?;
        ledger.reserve("qwen4exp K/V and key pools", f.pool_bytes)?;
        ledger.reserve("qwen4exp step scratch", f.s.len)?;
        let records_bytes = (f.record_stride * w.layers.len() as u64).max(64);
        ledger.reserve("qwen4exp route records", records_bytes)?;
        let acts_bytes = (f.act_stride * w.layers.len() as u64).max(64);
        ledger.reserve("qwen4exp expert inputs", acts_bytes)?;
        ledger.reserve("qwen4exp step inputs", f.inbox.len)?;
        let tables_bytes = f.tables_bytes(opts);
        ledger.reserve("qwen4exp page tables", tables_bytes)?;
        let exchange = if f.x_bytes > 0 {
            ledger.reserve("qwen4exp exchange", f.x_bytes)?;
            let buffer = DeviceBuffer::new(device, f.x_bytes)?;
            buffer.write(0, &vec![0u8; f.x_bytes as usize])?;
            Some(buffer)
        } else {
            None
        };
        let owner = if f.owner_bytes > 0 {
            ledger.reserve("qwen4exp expert owners", f.owner_bytes)?;
            let buffer = DeviceBuffer::new(device, f.owner_bytes)?;
            buffer.write(0, &vec![0u8; f.owner_bytes as usize])?;
            Some(buffer)
        } else {
            None
        };
        if f.kept_bytes > 0 {
            ledger.reserve("qwen4exp draft window inputs", f.kept_bytes)?;
        }
        let scratch = DeviceBuffer::new(device, f.s.len)?;
        let state = DeviceBuffer::new(device, state_bytes)?;
        let pools = DeviceBuffer::new(device, f.pool_bytes)?;
        let tables = DeviceBuffer::new(device, tables_bytes)?;
        let kept =
            (f.kept_bytes > 0).then(|| DeviceBuffer::new(device, f.kept_bytes)).transpose()?;
        // A defined start: the state of a slot is cleared again when a
        // sequence starts in it, the zeros stay zeros.
        state.write(0, &vec![0u8; state_bytes as usize])?;
        scratch.write(f.s.zeros, &vec![0u8; (f.s.xn - f.s.zeros) as usize])?;
        scratch.write(f.s.status, &[0u8; 16])?;
        // Every sequence slot owns its pages: entry `i` of slot `s` is page
        // `s * stride + i`, in the pool of stored formats; the key pool has
        // the same slots.
        let pages = opts.pages();
        let entries: Vec<u32> = (0..pages).map(|page| PAGE_ENTRY | page).collect();
        let mut bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
        bytes.extend((0..pages).flat_map(u32::to_le_bytes));
        tables.write(0, &bytes)?;

        let stream_events = |n: usize| -> Result<Vec<Event>, ModelError> {
            (0..n).map(|_| Event::new(device).map_err(ModelError::from)).collect()
        };
        let has_attention = f.attention.iter().any(Option::is_some);
        let attention = has_attention.then(|| AttentionKernels::load(device)).transpose()?;
        let marks = opts.profile.as_ref().map(|profile| Marks {
            profile: profile.clone(),
            on: false,
            events: Vec::new(),
            phases: Vec::new(),
        });
        let a = &c.attention;
        Ok(Self {
            index,
            role,
            first: first == 0 && role != Role::Worker,
            fused,
            flipped: false,
            opts: opts.clone(),
            device: device.clone(),
            kernels: Kernels::load(device)?,
            layers: LayerKernels::load(device)?,
            q4: Q4Kernels::load(device)?,
            attention,
            experts: ExpertKernels::load(device)?,
            side: Stream::new(device, 0)?,
            n: c.n_embd as u32,
            hc: c.n_stream as u32,
            wide: c.wide() as u32,
            rank: c.mixer_rank as u32,
            eps: c.rms_eps,
            n_vocab: c.n_vocab as u32,
            rope: RopeParams {
                dim: a.rope.dim as u32,
                freq_base: a.rope.freq_base,
                sections: a.rope.sections.map(|s| s as u32),
            },
            attn_shape: AttentionShape {
                n_head: a.n_head as u32,
                n_kv_head: a.n_head_kv as u32,
                head_dim: a.head_dim as u32,
            },
            scratch,
            state,
            pools,
            tables,
            entries,
            kept,
            records: DeviceBuffer::new(device, records_bytes)?,
            records_host: PinnedBuffer::new(device, records_bytes as usize)?,
            acts: DeviceBuffer::new(device, acts_bytes)?,
            acts_host: PinnedBuffer::new(
                device,
                (f.act_host_stride * w.layers.len() as u64).max(64) as usize,
            )?,
            routed: stream_events(w.layers.len())?,
            copied: stream_events(w.layers.len())?,
            inbox_host: PinnedBuffer::new(device, f.inbox.len as usize)?,
            inbox: DeviceBuffer::new(device, f.inbox.len)?,
            logits_host: PinnedBuffer::new(
                device,
                (u64::from(opts.logit_rows) * c.n_vocab as u64 * 4 * u64::from(head)).max(16)
                    as usize,
            )?,
            expert_at: Vec::new(),
            plan: None,
            pending: None,
            exchange,
            owner,
            peer: None,
            deferred: None,
            status_host: PinnedBuffer::new(device, 16)?,
            status_pending: false,
            marks,
            w,
            f,
        })
    }

    // --- facts -------------------------------------------------------------

    /// The device the part is on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The layers of the model the part holds.
    pub fn layer_range(&self) -> Range<usize> {
        let first = self.w.layers.first().map_or(0, |l| l.layer);
        first..first + self.w.layers.len()
    }

    /// Bytes of dense weights on the device.
    pub fn weight_bytes(&self) -> u64 {
        self.w.bytes()
    }

    /// How long the upload of the dense weights took.
    pub fn upload_time(&self) -> std::time::Duration {
        self.w.upload_time()
    }

    /// Bytes of recurrent state per sequence slot.
    pub fn state_bytes_per_slot(&self) -> u64 {
        self.f.slot_bytes
    }

    /// Bytes of K, V and pooled keys per page of [`ENTRY_CELLS`] cells.
    pub fn page_bytes(&self) -> u64 {
        self.f.pool_bytes / u64::from(self.opts.pages().max(1))
    }

    /// Of those, the bytes of pooled keys.
    pub fn key_page_bytes(&self) -> u64 {
        self.f.attention.iter().flatten().count() as u64 * self.f.ix.key_slot_bytes(self.opts.keys)
    }

    /// Bytes of the step scratch, the kept window inputs, the route records
    /// and the step's inputs.
    pub fn scratch_bytes(&self) -> u64 {
        self.f.step_bytes(&self.opts)
    }

    /// Whether the part holds the output head.
    pub fn has_head(&self) -> bool {
        self.w.head.is_some()
    }

    /// Whether the part holds the layer with the n-gram block, and which of
    /// its layers that is.
    pub fn ngram_layer(&self) -> Option<usize> {
        self.w.layers.iter().position(|l| l.ngram.is_some())
    }

    /// The number of layers of the part.
    pub fn n_layers(&self) -> usize {
        self.w.layers.len()
    }

    /// Tells the part where the expert store keeps the experts of its
    /// layers, in the order of the part's layers.
    pub fn set_experts(&mut self, at: Vec<ExpertAt>) -> Result<(), ModelError> {
        if at.len() != self.w.layers.len() {
            return Err(invalid(format!(
                "{} expert places for {} layers",
                at.len(),
                self.w.layers.len()
            )));
        }
        self.expert_at = at;
        Ok(())
    }

    /// What the part is to the residual streams.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Where a peer's kernels write what they send to this part; `None` for
    /// a part that holds whole layers.
    pub fn ports(&self) -> Option<Ports> {
        let exchange = self.exchange.as_ref()?;
        let flags = u64::from(self.f.x_flags) * 4;
        Some(Ports {
            flags: exchange.ptr(),
            record_flag: exchange.ptr().offset(flags - 4),
            partial: exchange.ptr().offset(flags.next_multiple_of(64)),
            m: self.at(self.f.s.m),
            acts: self.acts.ptr(),
            act_stride: self.f.act_stride,
            records: self.records.ptr(),
            record_stride: self.f.record_stride,
        })
    }

    /// Tells a mixer or a worker about its peer: the peer's ports (memory of
    /// a device this one has peer access to), and per layer whether the peer
    /// computes a share of the recurrent or attention half.
    pub fn set_peer(&mut self, ports: Ports, peer_core: Vec<bool>) -> Result<(), ModelError> {
        if self.role == Role::Whole || peer_core.len() != self.w.layers.len() {
            return Err(invalid("a peer for a part that shares no layers, or of other layers"));
        }
        self.peer = Some((ports, peer_core));
        Ok(())
    }

    /// Tells a mixer which experts its worker owns: a byte per layer and
    /// expert, not 0 for the worker's. Waits for the device.
    pub fn set_owner(&mut self, owner: &[u8]) -> Result<(), ModelError> {
        match &self.owner {
            Some(buffer) if owner.len() as u64 <= buffer.len() => Ok(buffer.write(0, owner)?),
            _ => Err(invalid("an owner table for a part that is no mixer, or of another size")),
        }
    }

    /// Whether the part computes a share of the recurrent or attention half
    /// of its layer `i`.
    pub fn has_core(&self, i: usize) -> bool {
        self.w.layers.get(i).is_some_and(|l| !matches!(l.core, CoreW::None))
    }

    fn at(&self, offset: u64) -> DevicePtr {
        self.scratch.ptr().offset(offset)
    }

    /// The residual buffer that holds the residual, and the other one.
    fn r(&self) -> DevicePtr {
        self.at(if self.flipped { self.f.s.r2 } else { self.f.s.r })
    }

    fn r_other(&self) -> DevicePtr {
        self.at(if self.flipped { self.f.s.r } else { self.f.s.r2 })
    }

    /// The number of exchange `kind` of the part's layer `i` in the step.
    fn exchange_no(&self, env: &StepEnv<'_>, i: usize, kind: u32) -> u32 {
        let layer = self.w.layers.get(i).map_or(i, |l| l.layer) as u32;
        env.xbase.wrapping_add(layer * 8 + kind)
    }

    /// Where the peer's partial output of exchange `no` arrives here.
    fn arrival(&self, no: u32) -> Option<Arrival> {
        let ports = self.ports()?;
        Some(Arrival { data: ports.partial, flags: ports.flags, token: no })
    }

    /// The f32 values of the activations of `rows` rows of `n_in` values at
    /// `offset` of the scratch.
    fn act_values(&self, offset: u64, n_in: u32, rows: u32) -> DevicePtr {
        let values = u64::from(n_in) * u64::from(rows);
        self.at(offset + act_bytes(u64::from(n_in), u64::from(rows)) - values * 4)
    }

    fn inbox_at(&self, offset: u64) -> DevicePtr {
        self.inbox.ptr().offset(offset)
    }

    /// The residual of the step's rows on the device, `rows * D` values:
    /// where it is once everything enqueued so far has run.
    pub fn residual(&self) -> DevicePtr {
        self.r()
    }

    /// Reads the residual of `rows` rows back. Waits for the device: for
    /// tests and measurements, after [`Part::flush`].
    pub fn read_residual(&self, rows: usize) -> Result<Vec<f32>, ModelError> {
        let at = if self.flipped { self.f.s.r2 } else { self.f.s.r };
        self.read_scratch(at, rows * self.wide as usize)
    }

    /// Overwrites the residual of the step's rows. Waits for the device.
    pub fn write_residual(&self, r: &[f32]) -> Result<(), ModelError> {
        let bytes: Vec<u8> = r.iter().flat_map(|v| v.to_le_bytes()).collect();
        let at = if self.flipped { self.f.s.r2 } else { self.f.s.r };
        Ok(self.scratch.write(at, &bytes)?)
    }

    /// The output of the recurrent or attention part of the layer enqueued
    /// last (`rows * N` values). Waits for the device.
    pub fn read_core_out(&self, rows: usize) -> Result<Vec<f32>, ModelError> {
        self.read_scratch(self.f.s.p, rows * self.n as usize)
    }

    /// The output of the mixture of experts of the layer enqueued last
    /// (`rows * N` values), as far as this part computed it. Waits for the
    /// device.
    pub fn read_ffn_out(&self, rows: usize) -> Result<Vec<f32>, ModelError> {
        self.read_scratch(self.f.s.out, rows * self.n as usize)
    }

    /// The partial output the peer sent last (`rows * N` values): after a
    /// layer and its flush, the peer's share of the layer's mixture of
    /// experts. Waits for the device.
    pub fn read_peer_partial(&self, rows: usize) -> Result<Vec<f32>, ModelError> {
        let Some(exchange) = &self.exchange else { return Ok(vec![0.0; rows * self.n as usize]) };
        let mut bytes = vec![0u8; rows * self.n as usize * 4];
        exchange.read((u64::from(self.f.x_flags) * 4).next_multiple_of(64), &mut bytes)?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    fn read_scratch(&self, offset: u64, n: usize) -> Result<Vec<f32>, ModelError> {
        let mut bytes = vec![0u8; n * 4];
        self.scratch.read(offset, &mut bytes)?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    /// The route record of the part's layer `i` as the host has it: valid
    /// once [`Part::record_arrived`] is true for the step.
    pub fn route_record(&self, i: usize) -> &[u8] {
        let stride = self.f.record_stride as usize;
        // SAFETY: the buffer holds a record per layer, `stride` apart, and
        // the caller reads it after the copy has finished.
        unsafe { std::slice::from_raw_parts(self.records_host.as_ptr().add(i * stride), stride) }
    }

    /// The input rows of the mixture of experts of the part's layer `i`, as
    /// the CPU reads them for `rows` rows (the codes and scales of the
    /// quantized activations): valid with the layer's route record.
    pub fn route_rows(&self, i: usize, rows: usize) -> &[u8] {
        let stride = self.f.act_host_stride as usize;
        let len = cpu_act_bytes(self.n as usize, rows).min(stride);
        // SAFETY: the buffer holds `stride` bytes per layer, and the caller
        // reads them after the copy has finished.
        unsafe { std::slice::from_raw_parts(self.acts_host.as_ptr().add(i * stride), len) }
    }

    /// Whether the route record of the part's layer `i` of the step being
    /// run has reached the host, with the layer's input rows. Never waits.
    pub fn record_arrived(&self, i: usize) -> Result<bool, ModelError> {
        Ok(self.copied[i].is_done()?)
    }

    /// The logits of the wanted rows of the step that ran last, `n_vocab`
    /// values per row: valid when the step's stream has run empty.
    pub fn logits(&self, rows: usize) -> &[f32] {
        let n = (rows * self.n_vocab as usize).min(self.logits_host_len() / 4);
        // SAFETY: the pinned buffer holds `logit_rows * n_vocab` f32 values
        // and is aligned to pages.
        unsafe { std::slice::from_raw_parts(self.logits_host.as_ptr().cast::<f32>(), n) }
    }

    fn logits_host_len(&self) -> usize {
        if self.w.head.is_some() {
            self.opts.logit_rows as usize * self.n_vocab as usize * 4
        } else {
            0
        }
    }

    // --- profile -------------------------------------------------------------

    fn mark(&mut self, stream: &Stream, phase: Phase) -> Result<(), ModelError> {
        let Some(marks) = self.marks.as_mut().filter(|m| m.on) else { return Ok(()) };
        let used = marks.phases.len();
        if marks.events.len() <= used {
            marks.events.push(Event::timed(&self.device)?);
        }
        marks.events[used].record(stream)?;
        marks.phases.push(phase);
        Ok(())
    }

    /// Adds the device time of the step that ran last to the profile. Waits
    /// for the device; call it when the step has ended.
    pub fn flush_profile(&mut self) -> Result<(), ModelError> {
        let Some(marks) = &mut self.marks else { return Ok(()) };
        let mut ms = [0.0f64; 8];
        for i in 1..marks.phases.len() {
            let elapsed = marks.events[i].elapsed_ms_since(&marks.events[i - 1])?;
            ms[marks.phases[i] as usize] += f64::from(elapsed);
        }
        if marks.phases.len() > 1 {
            marks.profile.add(self.index, &ms);
        }
        marks.phases.clear();
        Ok(())
    }

    // --- the step ------------------------------------------------------------

    fn check(&self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let dims = env.record.dims();
        if dims.n_rows == 0 || dims.n_rows > self.opts.max_rows {
            return Err(invalid(format!(
                "a step of {} rows; the part takes 1 to {}",
                dims.n_rows, self.opts.max_rows
            )));
        }
        if dims.n_out > self.opts.logit_rows {
            return Err(invalid(format!(
                "logits of {} rows; the part holds {}",
                dims.n_out, self.opts.logit_rows
            )));
        }
        let stride = self.opts.table_stride();
        for s in env.segments {
            if s.slot >= self.opts.max_slots {
                return Err(invalid(format!("slot {} of {}", s.slot, self.opts.max_slots)));
            }
            let end = s.pos0 as i64 + i64::from(s.n_rows);
            if s.pos0 < 0 || end > i64::from(stride) * i64::from(ENTRY_CELLS) {
                return Err(invalid(format!(
                    "cells {}..{end} of a sequence of {} cells",
                    s.pos0,
                    stride * ENTRY_CELLS
                )));
            }
            if !s.direct && (self.kept.is_none() || s.n_rows > self.opts.window_rows) {
                return Err(invalid(format!(
                    "a draft window of {} rows; {} were reserved",
                    s.n_rows, self.opts.window_rows
                )));
            }
        }
        if self.expert_at.len() != self.w.layers.len() {
            return Err(invalid("the part was not told where its experts are"));
        }
        if self.deferred.is_some() {
            return Err(invalid("the draft windows of the last step were not committed"));
        }
        Ok(())
    }

    /// The attention plan of the step. While every row is within the dense
    /// limit the planner cuts dense causal attention; a step with a row past
    /// it gets a plan whose lists the selection kernel fills on the device
    /// (rows within the limit get all their cells from it).
    fn build_plan(&self, env: &StepEnv<'_>) -> StepPlan {
        let (geo, ix) = (self.f.geo, self.f.ix);
        let n_rows = env.record.dims().n_rows;
        let stride = self.opts.table_stride();
        let last = env.segments.iter().map(|s| s.pos0 as u32 + s.n_rows).max().unwrap_or(0);
        let max_blocks = (last / ix.block).max(1);
        if last <= ix.dense_limit() {
            let mut planner = Planner::new(geo, self.f.plan_config, n_rows);
            for s in env.segments {
                let table = s.slot * stride;
                let entries = &self.entries[table as usize..(table + stride) as usize];
                planner.dense(table, entries, s.row0, s.n_rows, s.pos0 as u32);
            }
            return StepPlan { plan: planner.finish(), listed: None, max_blocks };
        }
        let per_row = n_rows * geo.kv_heads;
        let pieces = (self.f.plan_config.target_blocks / per_row).clamp(1, MAX_PIECES);
        let cap = ix.max_ranges(0);
        let mut plan = Plan {
            rows: vec![AttnRow::default(); n_rows as usize],
            row_ids: (0..n_rows).collect(),
            ranges: vec![0; (n_rows * cap * 3) as usize],
            state_slots: n_rows * pieces,
            needs_merge: true,
            ..Plan::default()
        };
        for s in env.segments {
            for t in 0..s.n_rows {
                let row = s.row0 + t;
                plan.rows[row as usize] = AttnRow {
                    table: s.slot * stride,
                    pos: s.pos0 as u32 + t,
                    slot_base: row * pieces,
                    n_slots: pieces,
                };
            }
        }
        for row in 0..n_rows {
            for piece in 0..pieces {
                for kv_head in 0..geo.kv_heads {
                    // Flag 1: the item reads converted entries.
                    plan.items.push(AttnItem {
                        table: plan.rows[row as usize].table,
                        row0: row,
                        n_rows: 1,
                        kv_head,
                        flags: 1,
                        slot: piece,
                        ..AttnItem::default()
                    });
                }
            }
        }
        plan.groups.push((AttnShape::for_vecs(geo.q_per_kv), 0..plan.items.len()));
        StepPlan { plan, listed: Some(pieces), max_blocks }
    }

    /// Starts a step: sends the record, the embeddings and the attention
    /// plan to the device, starts the residual (from the embeddings, or
    /// from what the previous device handed over) and builds the rotary
    /// table.
    pub fn begin(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        self.check(env)?;
        let stream = env.stream;
        let dims = env.record.dims();
        let n = dims.n_rows;
        if let Some(marks) = &mut self.marks {
            marks.phases.clear();
            marks.on = !marks.profile.paused.load(Ordering::Relaxed);
        }
        self.mark(stream, Phase::Other)?;
        if std::mem::take(&mut self.status_pending) {
            // SAFETY: the status of the step before, copied when it ended.
            let status =
                unsafe { u32::from_le_bytes(*self.status_host.as_ptr().cast::<[u8; 4]>()) };
            if status != 0 {
                return Err(invalid(format!(
                    "writing K and V reported status {status:#x} in the step before"
                )));
            }
        }
        let record = env.record.as_bytes();
        if record.len() as u64 > self.f.inbox.record_cap {
            return Err(invalid(format!("a step record of {} bytes", record.len())));
        }
        let has_attention = self.attention.is_some();
        let plan = has_attention.then(|| self.build_plan(env));
        let ib = self.f.inbox;
        let host = self.inbox_host.as_mut_ptr();
        // SAFETY: the pinned buffer has the sections of `Inbox`; the copies
        // of the step before have run (its stream ran empty).
        unsafe {
            std::ptr::copy_nonoverlapping(
                record.as_ptr(),
                host.add(ib.record as usize),
                record.len(),
            );
            if self.first {
                if env.x.len() != (n * self.n) as usize {
                    return Err(invalid(format!("{} embedding values for {n} rows", env.x.len())));
                }
                std::ptr::copy_nonoverlapping(
                    env.x.as_ptr().cast::<u8>(),
                    host.add(ib.x as usize),
                    env.x.len() * 4,
                );
            }
        }
        let mut sent = if self.first { ib.x + u64::from(n * self.n) * 4 } else { ib.record_cap };
        if let Some(step_plan) = &plan {
            let layout = step_plan.plan.layout();
            if layout.bytes > ib.plan_cap {
                return Err(invalid(format!("an attention plan of {} bytes", layout.bytes)));
            }
            // SAFETY: the plan section holds `plan_cap` bytes, the hands a
            // record per row.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(host.add(ib.plan as usize), layout.bytes as usize)
            };
            step_plan.plan.write(bytes);
            sent = ib.plan + layout.bytes;
            if let Some(pieces) = step_plan.listed {
                let cap = self.f.ix.max_ranges(0);
                let hands: Vec<ListHandOver> = (0..n)
                    .map(|row| ListHandOver {
                        ranges_off: row * cap * 3,
                        item_conv: row * pieces * self.f.geo.kv_heads,
                        pieces_conv: pieces,
                        kv_heads: self.f.geo.kv_heads,
                        ..ListHandOver::default()
                    })
                    .collect();
                let bytes = ListHandOver::to_bytes(&hands);
                // SAFETY: as above.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        host.add(ib.hands as usize),
                        bytes.len(),
                    );
                }
                sent = ib.hands + bytes.len() as u64;
            }
        }
        self.plan = plan;
        self.pending = None;
        self.flipped = false;
        let r = self.r();
        // SAFETY: the inbox and the scratch are this part's, sized for the
        // most rows; the inbound residual is the caller's guarantee.
        unsafe {
            stream.copy_from_host(self.inbox.ptr(), self.inbox_host.as_ptr(), sent as usize)?;
            if self.first {
                // 3.1: the embedding in every stream. The write kernel adds
                // `x * 2 sigmoid(0)`, which is `x`, to the zeroed streams.
                stream.fill(r, 0, (n * self.wide * 4) as usize)?;
                self.q4.stream_write(
                    stream,
                    r,
                    self.inbox_at(ib.x),
                    self.at(self.f.s.zeros),
                    n,
                    self.hc,
                    self.n,
                )?;
            } else if self.role == Role::Whole {
                let from =
                    env.inbound.ok_or_else(|| invalid("no residual from the device before"))?;
                stream.copy(r, from, (n * self.wide * 4) as usize)?;
            }
            self.mark(stream, Phase::Exchange)?;
            if has_attention {
                self.layers.rope_table(
                    stream,
                    self.inbox_at(ib.record),
                    dims,
                    self.at(self.f.s.rope),
                    self.rope,
                )?;
            }
        }
        // A sequence that starts in a slot starts from zero state.
        for s in env.segments.iter().filter(|s| s.pos0 == 0) {
            let at = self.state.ptr().offset(u64::from(s.slot) * self.f.slot_bytes);
            // SAFETY: the slot is inside the state arena (`check`).
            unsafe { stream.fill(at, 0, self.f.slot_bytes as usize)? };
        }
        self.mark(stream, Phase::Other)
    }

    /// `y = m x` for the `rows` rows of activations at `act`, with the decode
    /// kernels.
    ///
    /// # Safety
    ///
    /// `act` must hold the activations of `rows` rows of `m.n_in` values in
    /// the form the matrix's type reads, `y` `rows * m.n_out` f32 values.
    unsafe fn product(
        &self,
        stream: &Stream,
        m: &Mat,
        act: DevicePtr,
        y: DevicePtr,
        rows: u32,
    ) -> Result<(), ModelError> {
        // SAFETY: the matrix is in the arena as the file stores it; the rest
        // is the caller's guarantee.
        unsafe {
            self.kernels.matvec(stream, m.ty, self.w.mat(m), act, y, m.n_in, m.n_out, rows)?;
        }
        Ok(())
    }

    /// The read of a stream mixer, after the write `write` of the half block
    /// before it: leaves the mixed input as activations at `m` (with `push`
    /// in the peer's memory as well) and the inject values in `inj`.
    #[allow(clippy::too_many_arguments)]
    fn read(
        &mut self,
        stream: &Stream,
        mixer: &MixerW,
        write: Option<ReadWrite>,
        inj: Option<DevicePtr>,
        m: DevicePtr,
        push: Option<Departure>,
        rows: u32,
    ) -> Result<(), ModelError> {
        let s = self.f.s;
        let (hc, n, wide, rank) = (self.hc, self.n, self.wide, self.rank);
        let xn = self.act_values(s.xn, wide, rows);
        let norm = self.w.at(mixer.norm);
        if self.fused {
            let matrix = |m: &Mat| MixerMatrix { ptr: self.w.mat(m), ty: m.ty };
            let inject = mixer.inject.as_ref().filter(|_| inj.is_some()).map(matrix);
            // A read that writes goes from one residual buffer to the other:
            // every block of the launch reads the streams.
            let writes = write.is_some_and(|w| w.own.is_some() || w.peer.is_some());
            let (src, dst) = (self.r(), if writes { self.r_other() } else { self.r() });
            // SAFETY: scratch of this part sized for the most rows; weights
            // in the arena; a peer's memory as `set_peer` was told.
            unsafe {
                self.q4.read_down(
                    stream,
                    src,
                    dst,
                    write,
                    Some((norm, xn, matrix(&mixer.down), inject)),
                    self.at(s.partials),
                    rows,
                    hc,
                    n,
                    rank,
                    self.eps,
                )?;
                self.q4.read_up(
                    stream,
                    xn,
                    self.at(s.partials),
                    matrix(&mixer.up),
                    m,
                    inj.filter(|_| inject.is_some()),
                    push,
                    rows,
                    hc,
                    n,
                    rank,
                )?;
            }
            self.flipped ^= writes;
            return self.mark(stream, Phase::Mixers);
        }
        if push.is_some() || write.is_some_and(|w| w.peer.is_some() || w.own.is_none()) {
            return Err(invalid("an exchange with a peer needs the fused read"));
        }
        let r = self.r();
        let lo = self.act_values(s.lo, rank, rows);
        // SAFETY: scratch of this part sized for the most rows; weights in
        // the arena.
        unsafe {
            match write.and_then(|w| w.own.map(|out| (out, w.inj))) {
                Some((out, inj)) => {
                    self.q4.stream_write_norm(stream, r, out, inj, norm, xn, rows, hc, n, self.eps)
                }
                None => self.q4.stream_norm(stream, r, norm, xn, rows, hc, n, self.eps),
            }?;
            self.product(stream, &mixer.down, self.at(s.xn), self.at(s.down), rows)?;
            self.q4.squash(stream, self.at(s.down), lo, rows * rank, hc)?;
            self.product(stream, &mixer.up, self.at(s.lo), self.at(s.gate), rows)?;
            if let (Some(inject), Some(inj)) = (&mixer.inject, inj) {
                self.product(stream, inject, self.at(s.xn), inj, rows)?;
            }
            self.q4.mix_quant(stream, xn, self.at(s.gate), m, rows, hc, n)?;
        }
        self.mark(stream, Phase::Mixers)
    }

    /// The pending output as the write in front of a read takes it.
    fn take_write(&mut self) -> Option<ReadWrite> {
        let s = self.f.s;
        Some(match self.pending.take()? {
            Pending::Core { own, exchange } => ReadWrite {
                own: own.then(|| self.at(s.p)),
                inj: self.at(s.inj_a),
                peer: exchange.and_then(|no| self.arrival(no)),
            },
            Pending::Moe { exchange, .. } => ReadWrite {
                own: Some(self.at(s.out)),
                inj: self.at(s.inj_f),
                peer: exchange.and_then(|no| self.arrival(no)),
            },
        })
    }

    /// A worker's exchange: sends the pending partial output to the mixer,
    /// then waits for the mixed input of exchange `wait`. After the wait it
    /// computes the scalar gate of the shared expert (`gate`: its weights,
    /// the rows that arrived, where the gate goes) and clears `zero`.
    fn swap(
        &mut self,
        env: &StepEnv<'_>,
        wait: Option<u32>,
        gate: Option<(DevicePtr, DevicePtr, DevicePtr)>,
        zero: Option<DevicePtr>,
    ) -> Result<(), ModelError> {
        let s = self.f.s;
        let rows = env.record.dims().n_rows;
        let (Some((to, _)), Some(own)) = (self.peer.as_ref(), self.ports()) else {
            return Err(invalid("a worker without its mixer"));
        };
        let depart = |no: u32| Departure { data: to.partial, flags: to.flags, token: no };
        let push = match self.pending.take() {
            Some(Pending::Core { exchange: Some(no), .. }) => {
                Some((Some(self.at(s.p)), None, None, depart(no)))
            }
            Some(Pending::Moe { exchange: Some(no), shared }) => Some((
                Some(self.at(s.out)),
                shared.then(|| self.at(s.ysh)),
                shared.then(|| self.at(s.s)),
                depart(no),
            )),
            _ => None,
        };
        let wait = wait.map(|no| (own.flags, Q4Kernels::read_flags(rows, self.n), no));
        // SAFETY: scratch of this part; the peer's ports are memory of a
        // device this one has peer access to (`set_peer`).
        unsafe { self.q4.swap(env.stream, push, wait, gate, zero, rows, self.n)? };
        self.mark(env.stream, Phase::Exchange)
    }

    /// Writes the pending output of the last half block to the residual, or
    /// on a worker sends it to the mixer: for a caller that needs the
    /// residual between layers, and at the end of a step.
    pub fn flush(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.role == Role::Worker {
            return if self.pending.is_some() { self.swap(env, None, None, None) } else { Ok(()) };
        }
        let Some(write) = self.take_write() else { return Ok(()) };
        let s = self.f.s;
        let rows = env.record.dims().n_rows;
        let r = self.r();
        // SAFETY: scratch of this part; the write names the last half
        // block's results, a peer's share arrives in the exchange buffer.
        unsafe {
            if self.fused {
                self.q4.read_down(
                    env.stream,
                    r,
                    r,
                    Some(write),
                    None,
                    self.at(s.partials),
                    rows,
                    self.hc,
                    self.n,
                    self.rank,
                    self.eps,
                )?;
            } else {
                let out = write.own.ok_or_else(|| invalid("nothing of this part to write"))?;
                self.q4.stream_write(env.stream, r, out, write.inj, rows, self.hc, self.n)?;
            }
        }
        self.mark(env.stream, Phase::Mixers)
    }

    /// Enqueues layer `i` of the part for the rows of the step.
    pub fn enqueue_layer(&mut self, env: &StepEnv<'_>, i: usize) -> Result<(), ModelError> {
        let layer: LayerW =
            *self.w.layers.get(i).ok_or_else(|| {
                invalid(format!("layer {i} of a part of {}", self.w.layers.len()))
            })?;
        if self.role == Role::Worker {
            return self.worker_layer(env, i, &layer);
        }
        let stream = env.stream;
        let rows = env.record.dims().n_rows;
        let s = self.f.s;
        let (attn_mixer, ffn_mixer) =
            layer.mixers.ok_or_else(|| invalid("a layer without its mixers"))?;
        // The peer's ports, and whether it computes a share of this layer's
        // first half.
        let peer = self.peer.as_ref().map(|(ports, core)| (*ports, core[i]));
        if let Some(g) = &layer.ngram {
            // The host enqueues this layer when the rows of the table have
            // arrived; the stream may have run empty before that.
            self.mark(stream, Phase::Rows)?;
            // The n-gram block changes the residual in front of the read.
            self.flush(env)?;
            self.ngram(env, g)?;
        }
        // The recurrent or attention half.
        let write = self.take_write();
        let push = peer.filter(|(_, core)| *core).map(|(ports, _)| Departure {
            data: ports.m,
            flags: ports.flags,
            token: self.exchange_no(env, i, X_CORE_INPUT),
        });
        self.read(stream, &attn_mixer, write, Some(self.at(s.inj_a)), self.at(s.m), push, rows)?;
        let own = match &layer.core {
            CoreW::Recurrent(r) => {
                self.recurrent(env, i, r)?;
                true
            }
            CoreW::Attention(a) => {
                self.attend(env, i, a)?;
                true
            }
            CoreW::None => false,
        };
        self.pending = Some(Pending::Core {
            own,
            exchange: push.map(|_| self.exchange_no(env, i, X_CORE_PARTIAL)),
        });
        // The mixture of experts. Its input stays where the host can copy it
        // from until the next step: a miss is computed from it.
        let m = self.acts.ptr().offset(i as u64 * self.f.act_stride);
        let write = self.take_write();
        let push = peer.map(|(ports, _)| Departure {
            data: ports.acts.offset(i as u64 * ports.act_stride),
            flags: ports.flags,
            token: self.exchange_no(env, i, X_MOE_INPUT),
        });
        self.read(stream, &ffn_mixer, write, Some(self.at(s.inj_f)), m, push, rows)?;
        self.moe(env, i, &layer.moe)?;
        self.pending = Some(Pending::Moe {
            exchange: peer.map(|_| self.exchange_no(env, i, X_MOE_PARTIAL)),
            shared: layer.moe.shared.is_some(),
        });
        Ok(())
    }

    /// A layer on a worker: its share of the two halves, each between two
    /// exchanges with the mixer.
    fn worker_layer(
        &mut self,
        env: &StepEnv<'_>,
        i: usize,
        layer: &LayerW,
    ) -> Result<(), ModelError> {
        let rows = env.record.dims().n_rows;
        let s = self.f.s;
        match &layer.core {
            CoreW::Recurrent(r) => {
                self.swap(env, Some(self.exchange_no(env, i, X_CORE_INPUT)), None, None)?;
                self.recurrent(env, i, r)?;
                self.pending = Some(Pending::Core {
                    own: true,
                    exchange: Some(self.exchange_no(env, i, X_CORE_PARTIAL)),
                });
            }
            CoreW::Attention(_) => return Err(invalid("an attention layer on a worker")),
            CoreW::None => {}
        }
        // The mixer writes the input of the mixture of experts straight into
        // this layer's buffer. The shared expert's scalar gate is computed
        // from it while the kernel that waited is still there, and the
        // experts' output starts from zero.
        let m = self.acts.ptr().offset(i as u64 * self.f.act_stride);
        let values = u64::from(rows) * u64::from(self.n) * 4;
        let m_values = m.offset(act_bytes(u64::from(self.n), u64::from(rows)) - values);
        let gate = layer.moe.shared.map(|sh| (self.w.mat(&sh.inp), m_values, self.at(s.s)));
        let wait = self.exchange_no(env, i, X_MOE_INPUT);
        self.swap(env, Some(wait), gate, Some(self.at(s.out)))?;
        self.moe(env, i, &layer.moe)?;
        self.pending = Some(Pending::Moe {
            exchange: Some(self.exchange_no(env, i, X_MOE_PARTIAL)),
            shared: layer.moe.shared.is_some(),
        });
        Ok(())
    }

    fn ngram(&mut self, env: &StepEnv<'_>, g: &NgramW) -> Result<(), ModelError> {
        let (stream, dims) = (env.stream, env.record.dims());
        let rows = dims.n_rows;
        let s = self.f.s;
        let ib = self.f.inbox;
        let shape = self.f.ng.ok_or_else(|| invalid("the model has no n-gram block"))?;
        let history =
            self.f.ngram_history.ok_or_else(|| invalid("the part has no n-gram history"))?;
        let fetched = env.ngram.ok_or_else(|| invalid("no n-gram table rows for the step"))?;
        let bytes = (u64::from(rows) * shape.fetched_row_bytes()) as usize;
        if fetched.len() != bytes {
            return Err(invalid(format!(
                "{} bytes of n-gram table rows, {bytes} are needed",
                fetched.len()
            )));
        }
        let emb = self.act_values(s.emb, self.n, rows);
        let step = self.inbox_at(ib.record);
        // SAFETY: the inbox section holds the rows of the most rows; the
        // scratch and the state arena are this part's.
        unsafe {
            std::ptr::copy_nonoverlapping(
                fetched.as_ptr(),
                self.inbox_host.as_mut_ptr().add(ib.ngram as usize),
                bytes,
            );
            stream.copy_from_host(
                self.inbox_at(ib.ngram),
                self.inbox_host.as_ptr().add(ib.ngram as usize),
                bytes,
            )?;
            self.mark(stream, Phase::Exchange)?;
            self.q4.ngram_decode(stream, shape, self.inbox_at(ib.ngram), emb, rows)?;
            self.mark(stream, Phase::Other)?;
            self.product(stream, &g.key, self.at(s.emb), self.at(s.key), rows)?;
            self.product(stream, &g.value, self.at(s.emb), self.at(s.value), rows)?;
            self.mark(stream, Phase::Dense)?;
            self.q4.ngram_gate(
                stream,
                shape,
                self.eps,
                self.at(s.key),
                self.at(s.value),
                self.r(),
                self.w.at(g.norm_key),
                self.w.at(g.norm_query),
                self.w.at(g.norm_conv),
                self.at(s.gated),
                self.at(s.nrm),
                None,
                rows,
            )?;
            let history = self.state.ptr().offset(history);
            let stride = self.f.slot_bytes / 4;
            self.q4.ngram_window(
                stream,
                step,
                dims,
                shape,
                self.r(),
                self.at(s.gated),
                self.at(s.nrm),
                self.w.at(g.taps),
                GgmlType::F32,
                history,
                stride,
            )?;
            self.q4.ngram_commit(
                stream,
                step,
                dims,
                shape,
                CommitSet::Direct,
                self.at(s.nrm),
                history,
                stride,
            )?;
        }
        // The `nrm` of a draft window stays in its own buffer, which no other
        // layer writes, until the commit.
        self.mark(stream, Phase::Other)
    }

    fn recurrent(&mut self, env: &StepEnv<'_>, i: usize, r: &RecurrentW) -> Result<(), ModelError> {
        let (stream, dims) = (env.stream, env.record.dims());
        let rows = dims.n_rows;
        let s = self.f.s;
        let shape = r.shape;
        let at = self.f.recurrent[i].ok_or_else(|| invalid("not a recurrent layer"))?;
        let deferred = env.segments.iter().any(|s| !s.direct);
        // A step that verifies drafts keeps the inputs of this layer until
        // its commit; any other step uses the scratch every layer shares.
        let (qkv, kv, alpha, beta) = match (&self.kept, deferred) {
            (Some(kept), true) => {
                let k = |j: usize| kept.ptr().offset(at.kept[j]);
                (k(0), k(1), k(2), k(3))
            }
            (None, true) => return Err(invalid("no draft windows were reserved")),
            _ => (self.at(s.w1), self.at(s.kv), self.at(s.alpha), self.at(s.beta)),
        };
        let (z, o, q) = (self.at(s.w2), self.at(s.o), self.at(s.q));
        let m = self.at(s.m);
        let step = self.inbox_at(self.f.inbox.record);
        let conv_state = self.state.ptr().offset(at.conv);
        let delta_state = self.state.ptr().offset(at.delta);
        let (conv_stride, delta_stride) =
            (self.f.slot_bytes / 4, self.f.slot_bytes / self.opts.state.bytes());
        // SAFETY: every buffer is scratch of this part sized for the most
        // rows, weights in the arena, or the state arena with a slot per
        // sequence (segments checked in `check`); the kept buffers hold
        // `window_rows` rows, which bounds a deferred step.
        unsafe {
            self.product(stream, &r.qkv, m, qkv, rows)?;
            self.product(stream, &r.z, m, z, rows)?;
            self.product(stream, &r.beta, m, beta, rows)?;
            self.product(stream, &r.alpha, m, alpha, rows)?;
            self.mark(stream, Phase::Dense)?;
            self.layers.conv_window(
                stream,
                step,
                dims,
                shape,
                Some(self.eps),
                qkv,
                self.w.at(r.taps),
                conv_state,
                conv_stride,
                q,
                kv,
            )?;
            self.layers.delta_window(
                stream,
                step,
                dims,
                shape,
                self.opts.state,
                q,
                kv,
                alpha,
                beta,
                self.w.at(r.dt_bias),
                self.w.at(r.ssm_a),
                delta_state,
                delta_stride,
                o,
            )?;
            self.layers.conv_commit(
                stream,
                step,
                dims,
                shape,
                CommitSet::Direct,
                qkv,
                conv_state,
                conv_stride,
            )?;
            self.layers.gated_norm_quant(
                stream,
                OutputGate::Sigmoid,
                o,
                z,
                self.w.at(r.ssm_norm),
                self.at(s.act),
                rows * shape.n_v_heads,
                shape.head_dim,
                self.eps,
            )?;
            self.mark(stream, Phase::Recurrent)?;
            self.product(stream, &r.out, self.at(s.act), self.at(s.p), rows)?;
        }
        self.mark(stream, Phase::Dense)
    }

    fn attend(&mut self, env: &StepEnv<'_>, i: usize, a: &AttentionW) -> Result<(), ModelError> {
        let (stream, dims) = (env.stream, env.record.dims());
        let rows = dims.n_rows;
        let s = self.f.s;
        let (geo, ix) = (self.f.geo, self.f.ix);
        let at = self.f.attention[i].ok_or_else(|| invalid("not an attention layer"))?;
        let kernels = self.attention.clone().ok_or_else(|| invalid("no attention kernels"))?;
        let hq = self.attn_shape.n_head * self.attn_shape.head_dim;
        let deferred = env.segments.iter().any(|s| !s.direct);
        let raw = match (&self.kept, deferred) {
            (Some(kept), true) => kept.ptr().offset(at.kept_raw),
            (None, true) => return Err(invalid("no draft windows were reserved")),
            _ => self.at(s.raw),
        };
        let (qg, k_raw, v) = (self.at(s.w1), self.at(s.w2), self.at(s.v));
        let (q, gate, k, o) = (self.at(s.q), self.at(s.agate), self.at(s.k), self.at(s.o));
        let (iq, rope, m) = (self.at(s.iq), self.at(s.rope), self.at(s.m));
        let ib = self.f.inbox;
        let step = self.inbox_at(ib.record);
        let tables = self.tables.ptr();
        let key_table = self.tables.ptr().offset(4 * u64::from(self.opts.pages()));
        let kv_pool = self.pools.ptr().offset(at.kv_pool);
        let key_pool = self.pools.ptr().offset(at.key_pool);
        let carry = self.state.ptr().offset(at.carry);
        let carry_stride = self.f.slot_bytes / 4;
        // Taken out for the launches and put back: the marks need the part.
        let step_plan = self.plan.take().ok_or_else(|| invalid("the step has no plan"))?;
        let layout = step_plan.plan.layout();
        let base = self.inbox_at(ib.plan);
        let plan_rows = base.offset(layout.rows);
        // SAFETY: scratch and weights as in `recurrent`; the tables hold
        // the entries the plan names (the plan was built from their host
        // copy), the pools the pages those entries name, and the plan's
        // bytes are in the inbox for the whole step.
        unsafe {
            let buffers = step_plan.plan.on_device(base);
            self.product(stream, &a.q, m, qg, rows)?;
            self.product(stream, &a.k, m, k_raw, rows)?;
            self.product(stream, &a.v, m, v, rows)?;
            self.product(stream, &a.indexer_q, m, iq, rows)?;
            self.product(stream, &a.indexer_k, m, raw, rows)?;
            self.mark(stream, Phase::Dense)?;
            self.layers.attn_prep(
                stream,
                step,
                dims,
                self.attn_shape,
                self.rope.dim,
                self.eps,
                qg,
                k_raw,
                self.w.at(a.q_norm),
                self.w.at(a.k_norm),
                rope,
                q,
                Some(gate),
                k,
            )?;
            kernels.append(
                stream,
                &geo,
                plan_rows,
                rows,
                k,
                v,
                tables,
                DevicePtr::NULL,
                kv_pool,
                self.at(s.status),
            )?;
            self.q4.indexer_window(
                stream,
                step,
                dims,
                ix,
                self.opts.keys,
                self.eps,
                plan_rows,
                iq,
                raw,
                self.w.at(a.indexer_q_norm),
                self.w.at(a.indexer_k_norm),
                rope,
                iq,
                carry,
                carry_stride,
                key_table,
                key_pool,
            )?;
            self.q4.indexer_commit(
                stream,
                step,
                dims,
                ix,
                CommitSet::Direct,
                plan_rows,
                raw,
                rope,
                carry,
                carry_stride,
            )?;
            if step_plan.listed.is_some() {
                let (ranges, items) = (base.offset(layout.ranges), base.offset(layout.items));
                let (scores, counts) = (self.at(s.scores), self.at(s.counts));
                self.q4.block_scores(
                    stream,
                    ix,
                    self.opts.keys,
                    plan_rows,
                    iq,
                    key_table,
                    key_pool,
                    scores,
                    0,
                    rows,
                    step_plan.max_blocks,
                )?;
                // Every entry is in the pool of stored formats: one list.
                self.q4.select(
                    stream,
                    ix,
                    self.opts.rule,
                    plan_rows,
                    scores,
                    None,
                    ranges,
                    counts,
                    0,
                    rows,
                    step_plan.max_blocks,
                    ix.max_ranges(0),
                )?;
                self.q4.hand_over(stream, self.inbox_at(ib.hands), counts, ranges, items, rows)?;
            }
            kernels.attend(
                stream,
                &geo,
                &buffers,
                &AttendBuffers {
                    tables,
                    exact_pool: DevicePtr::NULL,
                    converted_pool: kv_pool,
                    q,
                    gate: Some(gate),
                    out: o,
                    scratch: self.at(s.attn_state),
                },
                1.0 / (geo.head_dim as f32).sqrt(),
            )?;
            self.mark(stream, Phase::Attention)?;
            self.kernels.quantize_activations(stream, o, self.at(s.act), hq, rows)?;
            self.product(stream, &a.out, self.at(s.act), self.at(s.p), rows)?;
        }
        self.plan = Some(step_plan);
        self.mark(stream, Phase::Dense)
    }

    fn moe(&mut self, env: &StepEnv<'_>, i: usize, w: &MoeW) -> Result<(), ModelError> {
        let stream = env.stream;
        let rows = env.record.dims().n_rows;
        let s = self.f.s;
        let at = self.expert_at[i];
        let m = self.acts.ptr().offset(i as u64 * self.f.act_stride);
        let record = self.records.ptr().offset(i as u64 * self.f.record_stride);
        let cap = w.shape.top_k * rows;
        let peer = self.peer.as_ref().map(|(ports, _)| *ports);
        // SAFETY: scratch and weights as in `recurrent`; the slot table and
        // the pool are the expert store's, which keeps the rules of the
        // table (`kernels/expert_layout.cuh`); the record of this layer is
        // not written again before the next step; a peer's ports are memory
        // of a device this one has peer access to.
        unsafe {
            if self.role == Role::Worker {
                // The shared expert while the mixer routes: its output is
                // scaled by the scalar gate when it is sent.
                if let Some(shared) = &w.shared {
                    self.shared(stream, shared, m, rows)?;
                    self.mark(stream, Phase::Dense)?;
                }
                let flag = self.ports().ok_or_else(|| invalid("a worker without ports"))?;
                let arrival = Arrival {
                    data: record,
                    flags: flag.record_flag,
                    token: self.exchange_no(env, i, X_RECORD),
                };
                self.q4.route_recv(stream, arrival, at.table, cap)?;
            } else {
                // The router first: the host learns of a missing expert
                // while the shared expert is computed.
                let router = w.router.ok_or_else(|| invalid("a layer without its router"))?;
                let full = if peer.is_some() { self.at(s.rec_full) } else { record };
                self.experts.route(
                    stream,
                    &w.shape,
                    self.w.at(router),
                    m,
                    self.at(s.logits),
                    at.table,
                    full,
                    rows,
                    env.token,
                )?;
                if let (Some(ports), Some(owner)) = (peer, &self.owner) {
                    // Each device gets the record of the experts it owns.
                    // Without the shared expert this part's experts add to
                    // zero.
                    let to = Departure {
                        data: ports.records.offset(i as u64 * ports.record_stride),
                        flags: ports.record_flag,
                        token: self.exchange_no(env, i, X_RECORD),
                    };
                    let zero = w.shared.is_none().then(|| (self.at(s.out), rows * self.n));
                    let owner = owner.ptr().offset(i as u64 * u64::from(w.shape.n_experts));
                    self.q4.route_split(stream, full, owner, record, to, zero, cap)?;
                }
            }
            self.routed[i].record(stream)?;
            self.side.wait_event(&self.routed[i])?;
            self.side.copy_to_host(
                self.records_host.as_mut_ptr().add(i * self.f.record_stride as usize),
                record,
                w.shape.record_bytes(rows) as usize,
            )?;
            // The rows as the kernels read them, behind the record: what the
            // CPU needs if it computes an expert the layer misses.
            self.side.copy_to_host(
                self.acts_host.as_mut_ptr().add(i * self.f.act_host_stride as usize),
                m,
                cpu_act_bytes(self.n as usize, rows as usize),
            )?;
            self.copied[i].record(&self.side)?;
            self.mark(stream, Phase::Experts)?;
            if self.role != Role::Worker
                && let Some(shared) = &w.shared
            {
                // The shared expert, scaled by its scalar gate: what the
                // routed experts add their weighted outputs to.
                self.product(stream, &shared.inp, m, self.at(s.s), rows)?;
                self.shared(stream, shared, m, rows)?;
                self.q4.gate_scale(
                    stream,
                    self.at(s.ysh),
                    self.at(s.s),
                    self.at(s.out),
                    rows,
                    self.n,
                )?;
                self.mark(stream, Phase::Dense)?;
            }
            self.experts.experts(
                stream,
                &w.shape,
                at.pool,
                at.table,
                record,
                m,
                self.at(s.hidden),
                self.at(s.out),
                at.host,
                rows,
                None,
            )?;
        }
        self.mark(stream, Phase::Experts)
    }

    /// The shared expert of a layer for the rows at `m`, without its scalar
    /// gate, into the scratch `ysh`.
    ///
    /// # Safety
    ///
    /// `m` must hold the activations of `rows` rows.
    unsafe fn shared(
        &self,
        stream: &Stream,
        shared: &SharedW,
        m: DevicePtr,
        rows: u32,
    ) -> Result<(), ModelError> {
        let s = self.f.s;
        // SAFETY: scratch and weights of this part; `m` is the caller's.
        unsafe {
            self.product(stream, &shared.gate, m, self.at(s.sg), rows)?;
            self.product(stream, &shared.up, m, self.at(s.su), rows)?;
            self.layers.swiglu_quant(
                stream,
                self.at(s.sg),
                self.at(s.su),
                self.at(s.act),
                rows * shared.gate.n_out,
            )?;
            self.product(stream, &shared.down, self.at(s.act), self.at(s.ysh), rows)?;
        }
        Ok(())
    }

    /// Ends the part's share of a step. A part without the output head
    /// writes the last layer's output to the residual, which the caller
    /// hands to the next device ([`Part::residual`]). The part with the head
    /// computes the logits of the wanted rows and copies them to the host
    /// ([`Part::logits`]); it returns how many rows that is.
    pub fn end(&mut self, env: &StepEnv<'_>) -> Result<u32, ModelError> {
        let stream = env.stream;
        let dims = env.record.dims();
        let s = self.f.s;
        if self.attention.is_some() {
            // SAFETY: the status word of the scratch and its pinned copy.
            unsafe {
                stream.copy_to_host(self.status_host.as_mut_ptr(), self.at(s.status), 4)?;
            }
            self.status_pending = true;
        }
        if env.segments.iter().any(|s| !s.direct) {
            self.deferred =
                Some((env.record.clone(), env.segments.iter().map(|s| s.direct).collect()));
        }
        let Some(head) = self.w.head.filter(|_| self.role != Role::Worker) else {
            self.flush(env)?;
            return Ok(0);
        };
        if dims.n_out == 0 {
            self.flush(env)?;
            return Ok(0);
        }
        let rows = dims.n_rows;
        let write = self.take_write();
        self.read(stream, &head.mixer, write, None, self.at(s.m), None, rows)?;
        let bytes = dims.n_out as usize * self.n_vocab as usize * 4;
        // SAFETY: `picked`, `act_p` and `head` hold `logit_rows` rows
        // (`check` bounds `n_out`), the pinned buffer the same.
        unsafe {
            self.layers.gather_rows(
                stream,
                self.inbox_at(self.f.inbox.record),
                dims,
                self.act_values(s.m, self.n, rows),
                None,
                self.at(s.picked),
                self.n,
                self.eps,
            )?;
            self.kernels.quantize_activations(
                stream,
                self.at(s.picked),
                self.at(s.act_p),
                self.n,
                dims.n_out,
            )?;
            self.mark(stream, Phase::Other)?;
            self.product(stream, &head.output, self.at(s.act_p), self.at(s.head), dims.n_out)?;
            self.mark(stream, Phase::Dense)?;
            stream.copy_to_host(self.logits_host.as_mut_ptr(), self.at(s.head), bytes)?;
        }
        self.mark(stream, Phase::Exchange)?;
        Ok(dims.n_out)
    }

    /// Enqueues the whole share of the part in a step: [`Part::begin`],
    /// every layer, [`Part::end`]. The n-gram rows must be in `env` when the
    /// part holds that layer.
    pub fn enqueue(&mut self, env: &StepEnv<'_>) -> Result<u32, ModelError> {
        self.begin(env)?;
        for i in 0..self.w.layers.len() {
            self.enqueue_layer(env, i)?;
        }
        self.end(env)
    }

    /// Whether the last step left draft windows that wait for their commit.
    pub fn has_deferred(&self) -> bool {
        self.deferred.is_some()
    }

    /// Commits the draft windows of the step enqueued last: `accepted[j]`
    /// rows of its `j`-th segment that was not direct count. The recurrent
    /// state, the n-gram history and the carried indexer state advance over
    /// the accepted rows; K and V of rejected rows are overwritten by the
    /// rows that take their cells. Enqueue it before the next step.
    pub fn enqueue_commit(&mut self, stream: &Stream, accepted: &[u32]) -> Result<(), ModelError> {
        let Some((mut record, segments)) = self.deferred.take() else {
            return Err(invalid("no draft windows wait for a commit"));
        };
        let dims = record.dims();
        let mut given = accepted.iter();
        for (j, direct) in segments.iter().enumerate() {
            if !direct {
                let count =
                    given.next().ok_or_else(|| invalid("an outcome per draft window is needed"))?;
                record.set_accepted(j, *count)?;
            }
        }
        if given.next().is_some() {
            return Err(invalid("more outcomes than draft windows"));
        }
        let ib = self.f.inbox;
        let bytes = record.as_bytes();
        let kept = self.kept.as_ref().ok_or_else(|| invalid("no draft windows were reserved"))?;
        let step = self.inbox_at(ib.commit);
        let s = self.f.s;
        // SAFETY: the commit section of the inbox holds a record; the kept
        // buffers, the plan rows and the rotary table still hold what the
        // window of the step left (nothing was enqueued since).
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.inbox_host.as_mut_ptr().add(ib.commit as usize),
                bytes.len(),
            );
            stream.copy_from_host(
                step,
                self.inbox_host.as_ptr().add(ib.commit as usize),
                bytes.len(),
            )?;
            for (i, layer) in self.w.layers.iter().enumerate() {
                if let (Some(_), Some(shape), Some(history)) =
                    (&layer.ngram, self.f.ng, self.f.ngram_history)
                {
                    self.q4.ngram_commit(
                        stream,
                        step,
                        dims,
                        shape,
                        CommitSet::Deferred,
                        self.at(s.nrm),
                        self.state.ptr().offset(history),
                        self.f.slot_bytes / 4,
                    )?;
                }
                match (&layer.core, self.f.recurrent[i], self.f.attention[i]) {
                    (CoreW::Recurrent(r), Some(at), _) => {
                        let k = |j: usize| kept.ptr().offset(at.kept[j]);
                        self.layers.commit(
                            stream,
                            step,
                            dims,
                            r.shape,
                            self.opts.state,
                            k(0),
                            k(1),
                            k(2),
                            k(3),
                            self.w.at(r.dt_bias),
                            self.w.at(r.ssm_a),
                            self.state.ptr().offset(at.conv),
                            self.f.slot_bytes / 4,
                            self.state.ptr().offset(at.delta),
                            self.f.slot_bytes / self.opts.state.bytes(),
                        )?;
                    }
                    (CoreW::Attention(_), _, Some(at)) => {
                        let plan =
                            self.plan.as_ref().ok_or_else(|| invalid("the step had no plan"))?;
                        let plan_rows = self.inbox_at(ib.plan).offset(plan.plan.layout().rows);
                        self.q4.indexer_commit(
                            stream,
                            step,
                            dims,
                            self.f.ix,
                            CommitSet::Deferred,
                            plan_rows,
                            kept.ptr().offset(at.kept_raw),
                            self.at(s.rope),
                            self.state.ptr().offset(at.carry),
                            self.f.slot_bytes / 4,
                        )?;
                    }
                    (CoreW::None, _, _) => {}
                    _ => return Err(invalid("a layer without its state")),
                }
            }
        }
        Ok(())
    }
}
