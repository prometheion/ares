//! One device's part of the model: its weights, its recurrent state, its
//! scratch, and the kernels of a step.
//!
//! A step is enqueued whole on the device's stream and nothing here waits for
//! the device, with one exception that is not the part's own: the attention
//! plan of a step is uploaded with a blocking copy (`Plan::upload` of
//! `ares-cuda`), which the part does before it enqueues anything, when the
//! stream is idle.
//!
//! Under a tensor split a part is a narrower model (fewer heads, fewer inner
//! rows) whose two outputs per block are partial sums; under a layer split it
//! is a range of whole blocks.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{
    ActivationForm, AttendBuffers, AttentionKernels, AttentionShape, CacheFormat, CacheGeometry,
    CommitSet, Device, DeviceBuffer, DevicePtr, ENTRY_CELLS, Event, Kernels, LayerKernels,
    MAX_BATCH, OutputGate, PUSH_MAX_VALUES, PinnedBuffer, PlanBuffers, PlanConfig, Planner,
    Precision, PrefillPath, RopeParams, Segment, StateFormat, StepDims, Stream, WeightLayout,
};
use ares_cuda::{PUSH_MAX_ROWS, PushSlot};
use ares_runtime::{Logits, ModelError, PinnedRam, Split};

use super::choose::{Balance, LayoutReport, Layouts, decode_in_batches, file_path};
use super::load::{AttentionW, BlockW, DeviceWeights, FfnW, LoadOptions, Mat, MixerW, RecurrentW};
use super::placement::Placement;
use crate::qwen35::Config;
use crate::qwen35::weights::Weights;

#[path = "mtp.rs"]
mod mtp;
mod resume;
mod tensor;

pub use mtp::{DraftBytes, DraftChain, DraftStep, DraftTimes, Drafted};
pub use resume::{CopyTimes, checkpoint_bytes, state_slot_bytes};

/// Bit 31 of a table entry: the page is in the pool of stored formats (the
/// only pool the formats without a tile mean use).
pub const PAGE_ENTRY: u32 = 1 << 31;

/// The choices of a device implementation.
#[derive(Clone, Debug)]
pub struct Options {
    /// How the matrix products round. A matrix in its device layout is
    /// multiplied with f16 operands at any number of rows, and the two
    /// differ only in how the tensor cores add up: [`Precision::Exact`]
    /// always in f32. For a matrix in the file's layout
    /// [`Precision::Fast`] takes the decode kernels (8-bit activations) for
    /// steps of up to eight rows and each type's fastest tensor-core path
    /// above; [`Precision::Exact`] takes the f16 tensor-core path for every
    /// product.
    pub precision: Precision,
    /// How each matrix gets its layout on the device: by default measured
    /// per matrix when the model is loaded, decode steps first.
    pub layouts: Layouts,
    /// How the delta-rule matrices are stored.
    pub state: StateFormat,
    /// The formats of K and of V. Only the formats without a tile mean (`f16`,
    /// `q8`) are supported: the others need the two pools of the state manager.
    pub kv: (CacheFormat, CacheFormat),
    /// The most rows of a step: decides the scratch.
    pub max_rows: u32,
    /// The most rows of a step whose logits are wanted.
    pub logit_rows: u32,
    /// The most rows of a step that verifies drafts (a window whose rows may
    /// be rejected keeps its inputs per recurrent block); 0 for none.
    pub window_rows: u32,
    /// The tokens the model's own draft block may propose (a model with a
    /// next-token-prediction block drafts with it whenever `window_rows`
    /// allows drafts): the rows of these tokens come first in the output
    /// head, and the draft head reads only them. `None`: the tokens whose
    /// rows are on the block's device, which are all of them unless a tensor
    /// split cuts the head.
    pub draft_vocabulary: Option<Arc<[u32]>>,
    /// Under a layer split, the rows a device hands to the next at a time: a
    /// step of at least twice as many certain rows is cut into groups of this
    /// size, and the devices work on different groups of the step at once.
    /// Under a tensor split, a step of more certain rows than this is cut
    /// into groups of this size, and a group's partial results travel while
    /// the device computes the next group. 0 keeps every step whole.
    ///
    /// The default of 256 is a measurement on compute capabilities 8.6 and
    /// 8.9 (RTX 3060 and RTX 4070, Qwen3.8-27B: 128 rows read a prompt 25 %
    /// slower, 512 rows without a second group 34 % slower). No other class
    /// was measured; the right size is where the tensor-core kernels of a
    /// class reach their rate and a group's parcel still crosses the link
    /// in less time than the next group computes.
    pub pipeline_rows: u32,
    /// Where the parts add up the device time of their kernels, by kind.
    pub profile: Option<Arc<Profile>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            precision: Precision::Fast,
            layouts: Layouts::default(),
            state: StateFormat::F32,
            kv: (CacheFormat::Q8, CacheFormat::Q8),
            max_rows: 512,
            logit_rows: 8,
            window_rows: 0,
            draft_vocabulary: None,
            pipeline_rows: 256,
            profile: None,
        }
    }
}

/// What a stretch of a step's kernels is, for [`Profile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Matrix products and the preparation of their activations.
    Weights = 0,
    /// Norms, convolution, delta rule, fused activations, gathers, adds.
    Layers = 1,
    /// Attention: preparing q and k, writing K and V, the attention itself.
    Attention = 2,
    /// Sums across devices and hand-overs. For the sums and hand-overs that
    /// make the stream wait inside one call (the push of a few rows, a
    /// hand-over), including the wait for the other device.
    Exchange = 3,
    /// Everything else between the start of a step and its end.
    Other = 4,
    /// The stream idle in front of a merge of a tensor split: the other
    /// devices' partial results are not there yet.
    Wait = 5,
}

/// The phases, in the order of their numbers.
pub const PHASES: [Phase; 6] =
    [Phase::Weights, Phase::Layers, Phase::Attention, Phase::Exchange, Phase::Other, Phase::Wait];

/// Device time per phase and device, added up over steps.
#[derive(Debug, Default)]
pub struct Profile {
    inner: Mutex<ProfileData>,
}

#[derive(Clone, Debug, Default)]
struct ProfileData {
    /// `[device][phase]` milliseconds.
    ms: Vec<[f64; 6]>,
    steps: Vec<u64>,
    /// `[device][merge of the step]`: the milliseconds of the device's own
    /// work in front of the merge, and of the merge with its waiting.
    merges: Vec<Vec<[f64; 2]>>,
    /// `[device]` what the copies of the prefix cache took.
    copies: Vec<CopyTimes>,
}

impl Profile {
    /// An empty profile.
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&self, device: usize, ms: &[f64; 6], merges: &[[f64; 2]]) {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if data.ms.len() <= device {
            data.ms.resize(device + 1, [0.0; 6]);
            data.steps.resize(device + 1, 0);
            data.merges.resize(device + 1, Vec::new());
        }
        for (sum, v) in data.ms[device].iter_mut().zip(ms) {
            *sum += v;
        }
        data.steps[device] += 1;
        let sums = &mut data.merges[device];
        if sums.len() < merges.len() {
            sums.resize(merges.len(), [0.0; 2]);
        }
        for (sum, v) in sums.iter_mut().zip(merges) {
            sum[0] += v[0];
            sum[1] += v[1];
        }
    }

    /// Per device and merge of a step under a tensor split, summed over the
    /// steps counted: the milliseconds of the device's own work between the
    /// merge before and this one, and of this merge with its waiting for the
    /// other devices. Merge `j` of a step that is not cut into groups follows
    /// half block `j - 1` (the mixer of block 0 is half 0); merge 0 follows
    /// the embedding.
    pub fn merges(&self) -> Vec<Vec<[f64; 2]>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).merges.clone()
    }

    /// Per device: the steps counted and the milliseconds per phase (indexed
    /// by the phase's number), summed over those steps.
    pub fn totals(&self) -> Vec<(u64, [f64; 6])> {
        let data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        data.steps.iter().copied().zip(data.ms.iter().copied()).collect()
    }

    /// Forgets everything counted so far.
    pub fn reset(&self) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = ProfileData::default();
    }
}

/// The marks of one step on a part's stream.
#[derive(Debug)]
struct Marks {
    profile: Arc<Profile>,
    events: Vec<Event>,
    /// The marks that end a merge of a tensor split.
    merges: Vec<usize>,
    /// The phase that ends at each event (the first event starts the step).
    phases: Vec<Phase>,
}

/// The paths to the other devices, as a part uses them.
pub trait Exchange {
    /// The bytes one pushed sum may write, or 0 when the devices have no
    /// push path.
    fn push_bytes(&self) -> u64;
    /// The next pushed sum of this device, for its merge kernel.
    fn push(&self) -> Result<PushSlot, ModelError>;
    /// The bytes between the regions [`Exchange::arrive`] returns.
    fn slot_bytes(&self) -> u64;
    /// Where this device writes its next parcel of `slot`.
    fn outbox(&self, slot: u32) -> Result<DevicePtr, ModelError>;
    /// Sends `bytes` bytes of this device's parcel of `slot` to the other
    /// devices; the stream does not wait for it.
    fn post(&self, slot: u32, bytes: u64) -> Result<(), ModelError>;
    /// Makes the stream wait for the other devices' parcels of `slot` and
    /// returns where they are, in device order, until [`Exchange::consumed`].
    fn arrive(&self, slot: u32) -> Result<DevicePtr, ModelError>;
    /// Ends the use of the parcels of `slot` that arrived.
    fn consumed(&self, slot: u32) -> Result<(), ModelError>;
    /// Hands `n` f32 values at `src` to the next device.
    fn send(&self, src: DevicePtr, n: u32) -> Result<(), ModelError>;
    /// Where the values the previous device handed over are, until
    /// [`Exchange::release`].
    fn receive(&self) -> Result<DevicePtr, ModelError>;
    /// Ends the use of the received values.
    fn release(&self) -> Result<(), ModelError>;
}

/// What a part is built from.
#[derive(Debug)]
pub struct PartSetup<'a> {
    /// Which part of the placement this is.
    pub index: usize,
    pub device: &'a Device,
    pub kernels: &'a Kernels,
    /// The device's memory accounts.
    pub ledger: &'a mut Ledger,
    /// Sequences at once: the slots of recurrent state.
    pub max_slots: u32,
    /// Pages of the pool: the K/V entries of every attention block.
    pub pages: u32,
    /// Entries per sequence in the page tables.
    pub table_stride: u32,
    /// The places for checkpoints of recurrent state, locked in `ram` when
    /// they are first used.
    pub checkpoints: u32,
    pub ram: &'a PinnedRam,
}

/// One step, as a part needs it.
pub struct StepEnv<'a> {
    pub stream: &'a Stream,
    /// The step record on the device, in the layout of the layer kernels.
    pub step: DevicePtr,
    pub dims: StepDims,
    /// The segments of the record.
    pub segments: &'a [Segment],
    /// The page tables on the device (`table_stride` entries per slot) and the
    /// memory of the pages.
    pub tables: DevicePtr,
    pub pages: DevicePtr,
    pub exchange: Option<&'a dyn Exchange>,
    /// The rows of the step that are a replay (rows of the whole step): they
    /// read the cells their sequence has cached already and append none.
    pub replay: &'a [Range<u32>],
}

impl std::fmt::Debug for StepEnv<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepEnv").field("dims", &self.dims).finish_non_exhaustive()
    }
}

/// Offsets of the step scratch.
#[derive(Clone, Copy, Debug, Default)]
struct Scratch {
    x: u64,
    xn: u64,
    p: u64,
    w1: u64,
    w2: u64,
    alpha: u64,
    beta: u64,
    q: u64,
    kv: u64,
    o: u64,
    k: u64,
    v: u64,
    rope: u64,
    act_q: u64,
    act_h: u64,
    picked: u64,
    logits: u64,
    logit_parts: u64,
    attn_state: u64,
    attn_state_bytes: u64,
    status: u64,
    len: u64,
}

/// Where a recurrent block keeps its state and what a draft window keeps.
#[derive(Clone, Copy, Debug)]
struct RecurrentAt {
    /// Offsets in a slot of the state arena.
    conv: u64,
    delta: u64,
    /// Offsets in the arena of kept inputs: qkv, kv, alpha, beta.
    kept: [u64; 4],
}

/// Where an attention block has its pool, and its geometry.
#[derive(Clone, Copy, Debug)]
struct AttentionAt {
    geo: CacheGeometry,
    /// Bytes from the start of the pages.
    pool: u64,
}

/// The prepared forms of an input that the products of a group of matrices
/// read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Forms {
    /// Quantized, all rows in one layout at `act_q`.
    int8: bool,
    /// Quantized launch by launch, for the decode kernels in several
    /// launches ([`Part::in_batches`]).
    batched: bool,
    /// f16, at `act_h`.
    f16: bool,
}

/// What a merge has prepared of an input already.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Prepared {
    /// All rows quantized in one layout at `act_q`, with their f32 copy.
    int8: bool,
    /// All rows as f16 at `act_h`.
    f16: bool,
}

/// What the merge of a tensor split left of the norm of the rows being
/// enqueued, for the half block that reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Normed {
    /// Nothing: the half block normalises the residual stream itself.
    #[default]
    No,
    /// The normalised rows.
    Rows,
    /// The normalised rows as the activations the half's products read.
    Prepared(Prepared),
}

/// A group of consecutive rows of a step that a layer split hands over on
/// its own, or whose partial results a tensor split sums on their own: a
/// smaller step, whose record is at `record` of the group records.
#[derive(Clone, Debug)]
struct Group {
    first_row: u32,
    dims: StepDims,
    segments: Vec<Segment>,
    record: u64,
}

/// One device's part of the model.
pub struct Part {
    index: usize,
    n_devices: usize,
    split: Split,
    opts: Options,
    device: Device,
    kernels: Kernels,
    layers: LayerKernels,
    attention: Option<AttentionKernels>,
    w: DeviceWeights,
    // Model sizes.
    d: u32,
    eps: f32,
    rope: RopeParams,
    attn_head_dim: u32,
    scratch: DeviceBuffer,
    s: Scratch,
    /// Per block of `w.blocks`.
    recurrent_at: Vec<Option<RecurrentAt>>,
    attention_at: Vec<Option<AttentionAt>>,
    state: DeviceBuffer,
    slot_bytes: u64,
    /// The places for checkpoints on the host (`resume.rs`).
    places: resume::Places,
    /// Pages of the pool.
    n_pages: u32,
    kept: Option<DeviceBuffer>,
    /// The step whose deferred segments wait for their commit.
    has_deferred: bool,
    table_stride: u32,
    max_slots: u32,
    /// The host copy of the page tables.
    tables: Vec<u32>,
    plan_config: PlanConfig,
    /// The attention plans of the step, by group and number of KV heads.
    plans: Vec<(usize, u32, PlanBuffers)>,
    /// Where the plans of a step are written on the host and where one copy
    /// on the step's stream puts them on the device.
    plan_host: PinnedBuffer,
    plan_device: DeviceBuffer,
    /// The groups of the step, when it is cut into groups; their records on
    /// the host and on the device.
    groups: Vec<Group>,
    group_host: PinnedBuffer,
    group_records: DeviceBuffer,
    /// The group being enqueued and its first row: buffers of the step's
    /// rows are addressed from there.
    group: usize,
    row_base: u32,
    status_host: PinnedBuffer,
    status_pending: bool,
    marks: Option<Marks>,
    /// The most rows of a sum that is pushed (tensor split).
    push_rows: u32,
    /// Per slot of the exchange: where this device's parcel is.
    parcels: Vec<DevicePtr>,
    /// What the last merge left for the half block that follows it.
    normed: Normed,
    /// Whether the last [`Part::prepare`] or fused kernel prepared for the
    /// decode kernels in several launches ([`Part::in_batches`]).
    batched: std::cell::Cell<bool>,
    page_bytes: u64,
    /// The model's own draft block, on the device that holds it.
    mtp: Option<mtp::Mtp>,
}

impl std::fmt::Debug for Part {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Part")
            .field("index", &self.index)
            .field("blocks", &self.w.blocks.len())
            .field("weights", &self.w.bytes())
            .finish_non_exhaustive()
    }
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

/// The bytes one page (128 cells) takes on a device under `placement`, for
/// K and V in `kv`: the entries of the attention blocks the device holds.
pub fn page_bytes(
    c: &Config,
    placement: &Placement,
    index: usize,
    kv: (CacheFormat, CacheFormat),
) -> u64 {
    placement.devices[index]
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(layer, slice)| {
            let slice = slice.as_ref()?;
            (c.is_attention(layer) && !slice.heads.is_empty())
                .then(|| geometry(c, kv, slice.heads.len() as u32).converted_entry_bytes())
        })
        .sum::<u64>()
        // The draft block's keys and values, on the device that holds it.
        + placement.draft.as_ref().filter(|d| d.device == index).map_or(0, |_| {
            geometry(c, kv, c.attention.n_head_kv as u32).converted_entry_bytes()
        })
}

fn geometry(c: &Config, kv: (CacheFormat, CacheFormat), kv_heads: u32) -> CacheGeometry {
    let a = &c.attention;
    CacheGeometry {
        k: kv.0,
        v: kv.1,
        head_dim: a.head_dim as u32,
        kv_heads,
        kv_total: kv_heads,
        first_kv_head: 0,
        q_per_kv: (a.n_head / a.n_head_kv) as u32,
    }
}

impl Part {
    /// Builds part `setup.index` of `placement`: uploads its weights and
    /// reserves and allocates its state and scratch.
    pub fn open(
        c: &Config,
        weights: &Weights<'_>,
        placement: &Placement,
        opts: &Options,
        setup: PartSetup<'_>,
    ) -> Result<Self, ModelError> {
        Self::open_balanced(c, weights, placement, opts, setup, None)
    }

    /// [`Part::open`] for the parts of a tensor split that are opened at the
    /// same time, one per device: through `balance` they tell each other
    /// what the weight products cost them, and each chooses the layouts of
    /// its matrices knowing where a step waits for it ([`Layouts`]). Every
    /// part of the placement must be opened with the same `balance`, or the
    /// ones that are wait for the others in vain before they decide alone.
    pub fn open_balanced(
        c: &Config,
        weights: &Weights<'_>,
        placement: &Placement,
        opts: &Options,
        setup: PartSetup<'_>,
        balance: Option<&Balance>,
    ) -> Result<Self, ModelError> {
        let PartSetup {
            index,
            device,
            kernels,
            ledger,
            max_slots,
            pages,
            table_stride,
            checkpoints,
            ram,
        } = setup;
        let plan = placement
            .devices
            .get(index)
            .ok_or_else(|| invalid(format!("the placement has no device {index}")))?;
        if opts.kv.0.has_mean() || opts.kv.1.has_mean() {
            return Err(invalid(
                "K/V formats with a tile mean (q7 to q4) need the exact pool and its conversion, \
                 which the runtime does not manage yet: use f16 or q8",
            ));
        }
        if opts.max_rows == 0 || opts.logit_rows == 0 || max_slots == 0 {
            return Err(invalid("a part needs at least one row, one logit row and one slot"));
        }
        // The rows the matrices will be multiplied with, for the choice of
        // their layouts: a decode step is one row, and so is the probe of a
        // model that verifies windows of drafts. The layout decides the
        // arithmetic of a product (the device layout's is the same for a row
        // in any window, the file's integer kernels' is not that of its tile
        // kernels), so a choice that depended on the window would let drafts
        // change the text; prompts are read in groups under a split
        // (`Part::plan_groups`) and in whole steps otherwise.
        let decode_rows = 1;
        let prompt_rows = match (placement.split, opts.pipeline_rows) {
            (Split::Single, _) | (_, 0) => opts.max_rows,
            (Split::Layer, per) if opts.max_rows < 2 * per => opts.max_rows,
            (_, per) => per.min(opts.max_rows),
        };
        let layers = LayerKernels::load(device)?;
        let w = DeviceWeights::load(
            device,
            kernels,
            &layers,
            ledger,
            c,
            weights,
            plan,
            placement.draft.as_ref(),
            index,
            LoadOptions {
                layouts: opts.layouts,
                precision: opts.precision,
                decode_rows,
                prompt_rows,
                merged: placement.split == Split::Tensor && placement.n_devices() > 1,
            },
            balance,
        )?;

        let (d, rows) = (c.n_embd as u64, u64::from(opts.max_rows));
        let (mut cd, mut vd, mut kd, mut nv, mut hq, mut hkv, mut f) = (0u64, 0, 0, 0, 0, 0, 0);
        let mut recurrent_at = Vec::new();
        let mut attention_at = Vec::new();
        let (mut slot_bytes, mut kept_bytes, mut pool) = (0u64, 0u64, 0u64);
        let window = u64::from(opts.window_rows);
        let mut state_slots = 0u64;
        let plan_config = PlanConfig::for_multiprocessors(device.info().multiprocessors);
        for block in &w.blocks {
            let (mut r_at, mut a_at) = (None, None);
            match &block.mixer {
                MixerW::Recurrent(r) => {
                    let s = r.shape;
                    cd = cd.max(u64::from(s.conv_dim()));
                    vd = vd.max(u64::from(s.value_dim()));
                    kd = kd.max(u64::from(s.key_dim()));
                    nv = nv.max(u64::from(s.n_v_heads));
                    let (conv, delta, end) = resume::state_extent(&s, opts.state, slot_bytes);
                    slot_bytes = end;
                    let mut kept = [0u64; 4];
                    let sizes = [
                        u64::from(s.conv_dim()),
                        u64::from(s.key_dim() + s.value_dim()),
                        u64::from(s.n_v_heads),
                        u64::from(s.n_v_heads),
                    ];
                    for (at, size) in kept.iter_mut().zip(sizes) {
                        *at = kept_bytes;
                        kept_bytes = (kept_bytes + window * size * 4).next_multiple_of(64);
                    }
                    r_at = Some(RecurrentAt { conv, delta, kept });
                }
                MixerW::Attention(a) => {
                    let geo = geometry(c, opts.kv, a.n_kv);
                    hq = hq.max(u64::from(a.n_head) * u64::from(geo.head_dim));
                    hkv = hkv.max(u64::from(a.n_kv) * u64::from(geo.head_dim));
                    a_at = Some(AttentionAt { geo, pool });
                    pool += u64::from(pages) * geo.converted_entry_bytes();
                    // The most partial states a plan makes: see `Planner::finish`.
                    let slots = 20 * u64::from(plan_config.target_blocks) / u64::from(a.n_kv) + 64;
                    state_slots = state_slots
                        .max(slots * u64::from(a.n_head) * (u64::from(geo.head_dim) + 1) * 4);
                }
                MixerW::Empty => {}
            }
            if let Some(ffn) = &block.ffn {
                f = f.max(u64::from(ffn.gate.n_out));
            }
            recurrent_at.push(r_at);
            attention_at.push(a_at);
        }
        // The draft block: its pool after the blocks', and room in the step's
        // buffers for the rows it computes there (`mtp.rs`).
        let draft_sizes =
            mtp::sizes(c, &w, opts, max_slots, pages, plan_config.target_blocks, pool);
        let without_draft = (hq, hkv, state_slots);
        if let Some(sizes) = &draft_sizes {
            pool += sizes.pool;
            hq = hq.max(sizes.kv_values);
            hkv = hkv.max(sizes.kv_values);
            state_slots = state_slots.max(sizes.attn_state);
        }
        let page_bytes = pool / u64::from(pages.max(1));

        // The scratch of a step.
        let mut s = Scratch::default();
        let mut len = 0u64;
        let mut take = |bytes: u64| {
            let at = len;
            len = (len + bytes.max(16)).next_multiple_of(64);
            at
        };
        let rope_pairs = (c.attention.rope.dim / 2) as u64;
        s.x = take(rows * d * 4);
        s.xn = take(rows * d * 4);
        s.p = take(rows * d * 4);
        s.w1 = take(rows * cd.max(2 * hq).max(f) * 4);
        s.w2 = take(rows * vd.max(hkv).max(f) * 4);
        s.alpha = take(rows * nv * 4);
        s.beta = take(rows * nv * 4);
        s.q = take(rows * kd.max(hq) * 4);
        s.kv = take(rows * (kd + vd).max(hq) * 4);
        s.o = take(rows * vd.max(hq) * 4);
        s.k = take(rows * hkv * 4);
        s.v = take(rows * hkv * 4);
        s.rope = take(rows * rope_pairs * 8);
        let widest = d.max(vd).max(hq).max(f);
        let logit_rows = u64::from(opts.logit_rows);
        // The decode kernels in several launches (`Part::in_batches`) read the rows in batches
        // of `MAX_BATCH`, each batch with its scales behind it: room for the last, partial one
        // as for a whole batch, and for the padding between the batches.
        s.act_q = take(
            Kernels::quantized_activation_bytes(
                widest as u32,
                opts.max_rows.next_multiple_of(MAX_BATCH),
            )
            .max(Kernels::quantized_activation_bytes(
                d as u32,
                opts.logit_rows.next_multiple_of(MAX_BATCH),
            ))
            .max(draft_sizes.as_ref().map_or(0, |sizes| sizes.act_q))
                + 256,
        );
        s.act_h = take(
            Kernels::prefill_activation_bytes(
                PrefillPath::F16,
                widest as u32,
                opts.max_rows.max(opts.logit_rows),
            )
            .max(draft_sizes.as_ref().map_or(0, |sizes| sizes.act_h)),
        );
        s.picked = take(logit_rows * d * 4);
        let head_rows = w.head.as_ref().map_or(0, |h| u64::from(h.n_out));
        s.logits = take(logit_rows * head_rows * 4);
        // A head in parts computes several rows part by part before they are
        // put in their place.
        let in_parts = w.head.as_ref().is_some_and(|h| h.parts.len() > 1);
        s.logit_parts = take(if in_parts { logit_rows * head_rows * 4 } else { 0 });
        s.attn_state = take(state_slots);
        s.attn_state_bytes = state_slots;
        s.status = take(16);
        s.len = len;

        ledger.reserve("qwen35 step scratch", s.len)?;
        let state_bytes = (slot_bytes * u64::from(max_slots)).max(64);
        ledger.reserve("qwen35 recurrent state", state_bytes)?;
        if kept_bytes > 0 {
            ledger.reserve("qwen35 draft window inputs", kept_bytes)?;
        }
        let scratch = DeviceBuffer::new(device, s.len)?;
        let state = DeviceBuffer::new(device, state_bytes)?;
        let kept = (kept_bytes > 0).then(|| DeviceBuffer::new(device, kept_bytes)).transpose()?;
        // A defined start: a slot is cleared again when a sequence starts in it.
        state.write(0, &vec![0u8; state_bytes as usize])?;
        scratch.write(s.status, &[0u8; 16])?;

        // The records of the groups of a step: per group a header, its
        // segments, and a token and four positions per row.
        let group_bytes = if placement.split != Split::Single && opts.pipeline_rows > 0 {
            let groups = u64::from(opts.max_rows.div_ceil(opts.pipeline_rows));
            (groups * (96 + 32 * u64::from(max_slots)) + 24 * rows).next_multiple_of(64)
        } else {
            64
        };
        ledger.reserve("qwen35 group records", group_bytes)?;
        // Room for the attention plans of a step: one per group of rows and
        // number of KV heads a block of this part has. A plan is a few dozen
        // bytes per row and per item; one that does not fit is uploaded on
        // its own (`Part::begin`), which makes the host wait for the device.
        let plan_bytes = {
            let mut heads: Vec<u32> =
                attention_at.iter().flatten().map(|a| a.geo.kv_heads).collect();
            heads.sort_unstable();
            heads.dedup();
            let groups = match (placement.split, opts.pipeline_rows) {
                (Split::Single, _) | (_, 0) => 1,
                (_, per) => u64::from(opts.max_rows.div_ceil(per)),
            };
            (heads.len() as u64 * (groups * (64 << 10) + 96 * rows)).next_multiple_of(64).max(64)
        };
        ledger.reserve("qwen35 attention plans", plan_bytes)?;
        let has_attention = attention_at.iter().any(Option::is_some) || draft_sizes.is_some();
        let attention = has_attention.then(|| AttentionKernels::load(device)).transpose()?;
        let marks = opts.profile.as_ref().map(|profile| Marks {
            profile: profile.clone(),
            events: Vec::new(),
            merges: Vec::new(),
            phases: Vec::new(),
        });
        // What the step's buffers grew by for the draft block, at most: K and
        // V of all its heads, queries as wide, partial states.
        let draft_growth = rows * 4 * (2 * (hkv - without_draft.1) + 5 * (hq - without_draft.0))
            + (state_slots - without_draft.2);
        let mtp =
            mtp::Mtp::open(c, device, ledger, &w, opts, max_slots, draft_sizes, draft_growth)?;
        // A checkpoint is the slot's recurrent state and, where the draft
        // block is, one row of its carry.
        let checkpoint = slot_bytes + if mtp.is_some() { d * 4 } else { 0 };
        let places =
            resume::Places::new(device, ram, checkpoint, checkpoints, opts.profile.is_some())?;
        let rope = &c.attention.rope;
        Ok(Self {
            index,
            n_devices: placement.n_devices(),
            split: placement.split,
            opts: opts.clone(),
            device: device.clone(),
            kernels: kernels.clone(),
            layers,
            attention,
            w,
            d: d as u32,
            eps: c.rms_eps,
            rope: RopeParams {
                dim: rope.dim as u32,
                freq_base: rope.freq_base,
                sections: rope.sections.map(|s| s as u32),
            },
            attn_head_dim: c.attention.head_dim as u32,
            scratch,
            s,
            recurrent_at,
            attention_at,
            state,
            slot_bytes,
            places,
            n_pages: pages,
            kept,
            has_deferred: false,
            table_stride,
            max_slots,
            tables: vec![0; max_slots as usize * table_stride as usize],
            plan_config,
            plans: Vec::new(),
            plan_host: PinnedBuffer::new(device, plan_bytes as usize)?,
            plan_device: DeviceBuffer::new(device, plan_bytes)?,
            groups: Vec::new(),
            group_host: PinnedBuffer::new(device, group_bytes as usize)?,
            group_records: DeviceBuffer::new(device, group_bytes)?,
            group: 0,
            row_base: 0,
            status_host: PinnedBuffer::new(device, 16)?,
            status_pending: false,
            marks,
            push_rows: (PUSH_MAX_VALUES / d as u32).clamp(1, opts.max_rows).min(PUSH_MAX_ROWS),
            parcels: Vec::new(),
            normed: Normed::No,
            batched: std::cell::Cell::new(false),
            page_bytes,
            mtp,
        })
    }

    // --- facts -------------------------------------------------------------

    /// Bytes of weights on the device.
    pub fn weight_bytes(&self) -> u64 {
        self.w.bytes()
    }

    /// How long the upload of the weights took.
    pub fn upload_time(&self) -> std::time::Duration {
        self.w.upload
    }

    /// Which matrices got which layout on this device, and what the choice
    /// took.
    pub fn layouts(&self) -> &LayoutReport {
        &self.w.layouts
    }

    /// Bytes of recurrent state per sequence on this device.
    pub fn state_bytes_per_slot(&self) -> u64 {
        self.slot_bytes
    }

    /// Bytes of scratch for a step of the most rows.
    pub fn scratch_bytes(&self) -> u64 {
        self.s.len
    }

    /// Bytes a draft window's kept inputs take.
    pub fn kept_bytes(&self) -> u64 {
        self.kept.as_ref().map_or(0, DeviceBuffer::len)
    }

    /// Bytes of K and V per page (128 cells) on this device.
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    /// Logits per row this part produces: its slice of the vocabulary.
    pub fn logits_per_row(&self) -> u32 {
        self.w.head.as_ref().map_or(0, |h| h.n_out)
    }

    /// The number of blocks (slices of blocks) on this device.
    pub fn n_blocks(&self) -> usize {
        self.w.blocks.len()
    }

    /// The model's block index of block `i` of this part.
    pub fn layer_of(&self, i: usize) -> usize {
        self.w.blocks[i].layer
    }

    fn at(&self, offset: u64) -> DevicePtr {
        self.scratch.ptr().offset(offset)
    }

    /// Where the rows of the group being enqueued start in a buffer of the
    /// step's rows with `width` f32 values per row.
    fn rows(&self, offset: u64, width: u32) -> DevicePtr {
        self.at(offset + u64::from(self.row_base) * u64::from(width) * 4)
    }

    /// The scratch buffer, for reading results in tests.
    pub fn scratch(&self) -> &DeviceBuffer {
        &self.scratch
    }

    /// Offset of the residual stream in [`Part::scratch`].
    pub fn x_offset(&self) -> u64 {
        self.s.x
    }

    /// Offset of the logits in [`Part::scratch`].
    pub fn logits_offset(&self) -> u64 {
        self.s.logits
    }

    // --- page tables ---------------------------------------------------------

    /// Tells the part that entries `first..` of the table of `slot` are now
    /// `entries` (as the device copy will be when the step runs).
    pub fn update_table(
        &mut self,
        slot: u32,
        first: u32,
        entries: impl ExactSizeIterator<Item = u32>,
    ) -> Result<(), ModelError> {
        if slot >= self.max_slots || first as usize + entries.len() > self.table_stride as usize {
            return Err(invalid(format!(
                "table update of slot {slot} at {first} is outside {} slots of {} entries",
                self.max_slots, self.table_stride
            )));
        }
        let at = slot as usize * self.table_stride as usize + first as usize;
        for (dst, entry) in self.tables[at..].iter_mut().zip(entries) {
            *dst = entry;
        }
        Ok(())
    }

    // --- profile ---------------------------------------------------------------

    fn mark(&mut self, stream: &Stream, phase: Phase) -> Result<(), ModelError> {
        let Some(marks) = &mut self.marks else { return Ok(()) };
        let i = marks.phases.len();
        if marks.events.len() <= i {
            marks.events.push(Event::timed(&self.device)?);
        }
        marks.events[i].record(stream)?;
        marks.phases.push(phase);
        Ok(())
    }

    /// Notes that the last mark ended a merge of a tensor split.
    fn mark_merge(&mut self) {
        if let Some(marks) = &mut self.marks {
            marks.merges.push(marks.phases.len().saturating_sub(1));
        }
    }

    /// Adds the marks of the last step to the profile. The step must have
    /// run: the next step's enqueue calls this, and a caller that reads the
    /// profile after the last step does.
    pub fn flush_profile(&mut self) -> Result<(), ModelError> {
        self.draft_flush()?;
        let Some(marks) = &mut self.marks else { return Ok(()) };
        if marks.phases.len() > 1 {
            let mut ms = [0.0f64; 6];
            let mut merges = Vec::with_capacity(marks.merges.len());
            let (mut own, mut exchange, mut next) = (0.0f64, 0.0f64, 0);
            for i in 1..marks.phases.len() {
                let elapsed = f64::from(marks.events[i].elapsed_ms_since(&marks.events[i - 1])?);
                ms[marks.phases[i] as usize] += elapsed;
                match marks.phases[i] {
                    Phase::Exchange | Phase::Wait => exchange += elapsed,
                    _ => own += elapsed,
                }
                if marks.merges.get(next) == Some(&i) {
                    merges.push([own, exchange]);
                    (own, exchange, next) = (0.0, 0.0, next + 1);
                }
            }
            marks.profile.add(self.index, &ms, &merges);
        }
        marks.phases.clear();
        marks.merges.clear();
        Ok(())
    }

    // --- products ----------------------------------------------------------------

    /// Which kernel family multiplies a matrix of `ty` in the file's layout
    /// with `n` rows: `None` for the decode kernels.
    fn path(&self, ty: GgmlType, n: u32) -> Result<Option<PrefillPath>, ModelError> {
        file_path(&self.kernels, self.opts.precision, ty, n)
    }

    /// Whether the products of `mats` with `n` rows are taken, for those in
    /// the file's layout, by the decode kernels in several launches of up to
    /// [`MAX_BATCH`] rows each: a window of a few more rows than one launch
    /// takes, where every such matrix's decode kernel still beats its
    /// tensor-core kernel, and any number of rows when one of the matrices
    /// has no tensor-core kernel. Where that is, is not decided here:
    /// `Kernels::decode_kernel_rows` and `Kernels::has_prefill_matmul` say
    /// it per weight type. (The tile kernels of the file's layout decode a
    /// whole matrix whatever the rows. On Qwen3.8-27B over an RTX 4070 and
    /// an RTX 3060 that is a step of about 80 ms from 9 to 24 rows, while the
    /// decode kernels take 36 ms for 8. A matrix in its device layout has no
    /// such gap.) The decision is the group's: the matrices that read the
    /// same input are all multiplied in launches or none.
    fn in_batches(&self, mats: &[&Mat], n: u32) -> bool {
        let mut file = mats.iter().filter(|m| m.layout == WeightLayout::File).peekable();
        file.peek().is_some() && decode_in_batches(self.opts.precision, file.map(|m| m.ty), n)
    }

    /// The forms of an input of `n` rows that the products with `mats` read.
    fn forms(&self, mats: &[&Mat], n: u32) -> Result<Forms, ModelError> {
        let batched = self.in_batches(mats, n);
        let mut forms = Forms { batched, ..Forms::default() };
        for m in mats {
            match m.layout {
                WeightLayout::Device => forms.f16 = true,
                WeightLayout::File if batched => {}
                WeightLayout::File => match self.path(m.ty, n)? {
                    None | Some(PrefillPath::Int8) => forms.int8 = true,
                    Some(PrefillPath::F16 | PrefillPath::F16Flush) => forms.f16 = true,
                },
            }
        }
        Ok(forms)
    }

    /// The rows of each launch that writes or reads the quantized activations
    /// of `n` rows of `n_in` values, and where that launch's activations are:
    /// all rows at `act_q`, or, for a group of matrices that [`Part::in_batches`]
    /// decided on (`batched`, the group's decision, not any one matrix's), the
    /// first row, rows and place of every launch.
    fn act_ranges(
        &self,
        batched: bool,
        n_in: u32,
        n: u32,
    ) -> impl Iterator<Item = (u32, u32, DevicePtr)> + use<> {
        let act_q = self.at(self.s.act_q);
        let per = if batched { MAX_BATCH } else { n.max(1) };
        let stride = Kernels::quantized_activation_bytes(n_in, per).next_multiple_of(64);
        (0..n.div_ceil(per))
            .map(move |b| (b * per, per.min(n - b * per), act_q.offset(u64::from(b) * stride)))
    }

    /// Prepares `n` rows of `n_in` values at `x` for the products with
    /// `mats`: quantized into `act_q`, as f16 into `act_h`, or both, as their
    /// layouts and kernels ask. What `have` names is there already (a merge
    /// wrote it) and only the rest is made; with `have.int8` the rows are
    /// read from the copy in the quantized layout and `x` is not used.
    ///
    /// # Safety
    ///
    /// `x` must address `n * n_in` f32 values on the stream's device, unless
    /// `have.int8`; the scratch must hold what `have` names.
    unsafe fn prepare(
        &self,
        stream: &Stream,
        x: DevicePtr,
        n_in: u32,
        n: u32,
        mats: &[&Mat],
        have: Prepared,
    ) -> Result<(), ModelError> {
        let forms = self.forms(mats, n)?;
        self.batched.set(forms.batched);
        let (act_q, act_h) = (self.at(self.s.act_q), self.at(self.s.act_h));
        let rows = if have.int8 {
            // The f32 copy at the end of the quantized layout.
            let values = u64::from(n_in) * u64::from(n);
            act_q.offset(ares_cuda::quantized_activation_bytes(values) - values * 4)
        } else {
            x
        };
        if (forms.batched && (have.int8 || have.f16)) || (forms.int8 && have.f16 && !have.int8) {
            return Err(invalid(
                "a merge prepared activations in another form than the half block reads",
            ));
        }
        // SAFETY: the scratch buffers hold the prepared forms of the widest
        // input for the most rows; the rows are the caller's guarantee.
        unsafe {
            if forms.batched {
                for (first, count, act) in self.act_ranges(forms.batched, n_in, n) {
                    let at = rows.offset(u64::from(first) * u64::from(n_in) * 4);
                    self.kernels.quantize_activations(stream, at, act, n_in, count)?;
                }
            }
            if forms.int8 && !have.int8 {
                if n <= MAX_BATCH {
                    self.kernels.quantize_activations(stream, rows, act_q, n_in, n)?;
                } else {
                    self.kernels.prepare_prefill_activations(
                        stream,
                        PrefillPath::Int8,
                        rows,
                        act_q,
                        n_in,
                        n,
                    )?;
                }
            }
            if forms.f16 && !have.f16 {
                self.kernels.prepare_activations(
                    stream,
                    ActivationForm::F16,
                    rows,
                    act_h,
                    n_in,
                    n,
                )?;
            }
        }
        Ok(())
    }

    /// The gated norm of the output `o` of a recurrent block with the gate
    /// `z`, `n` rows, prepared for the output projection in the one form it
    /// reads, by one fused kernel.
    ///
    /// # Safety
    ///
    /// `o` and `z` must address `n` rows of the block's value heads on the
    /// stream's device.
    unsafe fn gated_norm(
        &self,
        stream: &Stream,
        r: &RecurrentW,
        o: DevicePtr,
        z: DevicePtr,
        n: u32,
    ) -> Result<(), ModelError> {
        let shape = r.shape;
        let forms = self.forms(&[&r.out], n)?;
        self.batched.set(forms.batched);
        let weight = self.w.at(r.ssm_norm);
        // SAFETY: the scratch holds either form of the widest input for the
        // most rows; the weight is in the arena.
        unsafe {
            if forms.f16 && shape.n_v_heads <= LayerKernels::GATED_NORM_F16_MAX_HEADS {
                self.layers.gated_norm_f16(
                    stream,
                    OutputGate::Silu,
                    o,
                    z,
                    weight,
                    self.at(self.s.act_h),
                    n,
                    shape.n_v_heads,
                    shape.head_dim,
                    self.eps,
                )?;
                return Ok(());
            }
            for (first, rows, act) in self.act_ranges(forms.batched, shape.value_dim(), n) {
                let at = u64::from(first) * u64::from(shape.value_dim()) * 4;
                self.layers.gated_norm_quant(
                    stream,
                    OutputGate::Silu,
                    o.offset(at),
                    z.offset(at),
                    weight,
                    act,
                    rows * shape.n_v_heads,
                    shape.head_dim,
                    self.eps,
                )?;
            }
            // More heads than the fused kernel for f16 keeps: converted from
            // the copy the quantizing one left.
            if forms.f16 {
                let have = Prepared { int8: true, f16: false };
                self.prepare(stream, o, shape.value_dim(), n, &[&r.out], have)?;
            }
        }
        Ok(())
    }

    /// `silu(gate) * up` of `n` rows of `inner` values, prepared for the
    /// product with `down` in the one form it reads, by one fused kernel.
    ///
    /// # Safety
    ///
    /// `gate` and `up` must address `n * inner` f32 values on the stream's
    /// device.
    unsafe fn swiglu(
        &self,
        stream: &Stream,
        gate: DevicePtr,
        up: DevicePtr,
        inner: u32,
        n: u32,
        down: &Mat,
    ) -> Result<(), ModelError> {
        let forms = self.forms(&[down], n)?;
        self.batched.set(forms.batched);
        // SAFETY: the scratch holds either form of the widest input for the
        // most rows.
        unsafe {
            if forms.f16 && inner <= LayerKernels::SWIGLU_F16_MAX_WIDTH {
                self.layers.swiglu_f16(stream, gate, up, self.at(self.s.act_h), n, inner)?;
                return Ok(());
            }
            for (first, rows, act) in self.act_ranges(forms.batched, inner, n) {
                let at = u64::from(first) * u64::from(inner) * 4;
                self.layers.swiglu_quant(
                    stream,
                    gate.offset(at),
                    up.offset(at),
                    act,
                    rows * inner,
                )?;
            }
            // A wider row than the fused kernel for f16 keeps: converted from
            // the copy the quantizing one left.
            if forms.f16 {
                let have = Prepared { int8: true, f16: false };
                self.prepare(stream, gate, inner, n, &[down], have)?;
            }
        }
        Ok(())
    }

    /// The norm of the residual stream with `weight`, prepared for the
    /// products with `mats`. Under a tensor split the merge in front of the
    /// half block has normalised the rows already.
    ///
    /// # Safety
    ///
    /// `weight` must address `d` f32 values on the stream's device.
    unsafe fn normed_input(
        &mut self,
        env: &StepEnv<'_>,
        weight: DevicePtr,
        mats: &[&Mat],
    ) -> Result<(), ModelError> {
        let (stream, n) = (env.stream, env.dims.n_rows);
        let (x, xn) = (self.rows(self.s.x, self.d), self.rows(self.s.xn, self.d));
        // SAFETY: `x` and `xn` hold the step's rows, the scratch the prepared
        // forms of the widest input; the weight is the caller's guarantee.
        unsafe {
            match std::mem::take(&mut self.normed) {
                Normed::No => {
                    self.kernels.rms_norm_f32(stream, x, weight, xn, self.d, n, self.eps)?;
                    self.mark(stream, Phase::Layers)?;
                    self.prepare(stream, xn, self.d, n, mats, Prepared::default())
                }
                Normed::Rows => self.prepare(stream, xn, self.d, n, mats, Prepared::default()),
                Normed::Prepared(have) => self.prepare(stream, xn, self.d, n, mats, have),
            }
        }
    }

    /// `y = m x` for the `n` rows that [`Part::prepare`] or a fused kernel
    /// prepared.
    ///
    /// # Safety
    ///
    /// `y` must address `n * m.n_out` f32 values on the stream's device.
    unsafe fn product(
        &self,
        stream: &Stream,
        m: &Mat,
        n: u32,
        y: DevicePtr,
    ) -> Result<(), ModelError> {
        let w = self.w.mat(m);
        if m.layout == WeightLayout::Device {
            // SAFETY: the matrix is in the arena in its device layout and the
            // f16 activations were prepared for it.
            unsafe {
                self.kernels.product(
                    stream,
                    m.ty,
                    WeightLayout::Device,
                    self.opts.precision,
                    w,
                    self.at(self.s.act_h),
                    y,
                    m.n_in,
                    m.n_out,
                    n,
                )?;
            }
            return Ok(());
        }
        if self.batched.get() {
            for (first, rows, act) in self.act_ranges(true, m.n_in, n) {
                let at = y.offset(u64::from(first) * u64::from(m.n_out) * 4);
                // SAFETY: as below; the range is inside the caller's rows.
                unsafe {
                    self.kernels.matvec(stream, m.ty, w, act, at, m.n_in, m.n_out, rows)?;
                }
            }
            return Ok(());
        }
        // SAFETY: the matrix is in the arena as the file stores it and the
        // activations were prepared for this path by `prepare`.
        unsafe {
            match self.path(m.ty, n)? {
                None => self.kernels.matvec(
                    stream,
                    m.ty,
                    w,
                    self.at(self.s.act_q),
                    y,
                    m.n_in,
                    m.n_out,
                    n,
                )?,
                Some(path) => {
                    let act = match path {
                        PrefillPath::Int8 => self.s.act_q,
                        PrefillPath::F16 | PrefillPath::F16Flush => self.s.act_h,
                    };
                    self.kernels.matmul(
                        stream,
                        m.ty,
                        path,
                        w,
                        self.at(act),
                        y,
                        m.n_in,
                        m.n_out,
                        n,
                    )?
                }
            }
        }
        Ok(())
    }

    // --- exchange ----------------------------------------------------------------

    /// Adds the result `p` of a half block to the residual stream. (Under a
    /// tensor split `p` is a partial result and the merges of
    /// `Part::enqueue_tensor` do this.)
    fn merge(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let n = env.dims.n_rows;
        let (x, p) = (self.rows(self.s.x, self.d), self.rows(self.s.p, self.d));
        // SAFETY: `x` and `p` hold the step's rows of `d` values.
        unsafe { self.kernels.add_f32(env.stream, x, p, n * self.d)? };
        self.mark(env.stream, Phase::Layers)
    }

    /// Zeroes the partial result: this device holds nothing of the part of the
    /// block that is being summed.
    fn zero_partial(&self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let bytes = env.dims.n_rows as usize * self.d as usize * 4;
        // SAFETY: `p` holds the step's rows.
        unsafe { env.stream.fill(self.rows(self.s.p, self.d), 0, bytes)? };
        Ok(())
    }

    // --- the step ------------------------------------------------------------------

    fn check(&self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let dims = env.dims;
        if dims.n_rows > self.opts.max_rows || dims.n_out > self.opts.logit_rows {
            return Err(invalid(format!(
                "a step of {} rows with {} logit rows; the model was opened for {} and {}",
                dims.n_rows, dims.n_out, self.opts.max_rows, self.opts.logit_rows
            )));
        }
        for s in env.segments {
            let end = u64::from(s.pos0.max(0) as u32) + u64::from(s.n_rows);
            if s.slot >= self.max_slots
                || s.pos0 < 0
                || end > u64::from(self.table_stride) * u64::from(ENTRY_CELLS)
            {
                return Err(invalid(format!(
                    "segment {s:?} is outside {} slots of {} cells",
                    self.max_slots,
                    u64::from(self.table_stride) * u64::from(ENTRY_CELLS)
                )));
            }
        }
        let deferred = env.segments.iter().any(|s| !s.direct);
        if deferred && dims.n_rows > self.opts.window_rows {
            return Err(invalid(format!(
                "a step of {} rows verifies drafts; the model keeps the inputs of {} rows",
                dims.n_rows, self.opts.window_rows
            )));
        }
        Ok(())
    }

    /// What must happen before the kernels of a step are enqueued: the result
    /// of the last step's checks, its profile, and the attention plans.
    fn begin(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        self.check(env)?;
        if self.has_deferred {
            return Err(invalid(
                "the draft windows of the last step were not committed before this step",
            ));
        }
        if std::mem::take(&mut self.status_pending) {
            let status = u32::from_le_bytes([
                self.status_host[0],
                self.status_host[1],
                self.status_host[2],
                self.status_host[3],
            ]);
            if status != 0 {
                return Err(invalid(format!(
                    "the K/V writer reported status {status:#x} in the last step \
                     (1: a value that is not finite, 2: overflow)"
                )));
            }
        }
        self.flush_profile()?;
        self.plan_groups(env)?;
        self.plans.clear();
        let mut planned = 0usize;
        let whole =
            [Group { first_row: 0, dims: env.dims, segments: env.segments.to_vec(), record: 0 }];
        let groups = if self.groups.is_empty() { &whole[..] } else { &self.groups[..] };
        for (g, group) in groups.iter().enumerate() {
            for at in self.attention_at.iter().flatten() {
                if self.plans.iter().any(|(pg, heads, _)| *pg == g && *heads == at.geo.kv_heads) {
                    continue;
                }
                let mut planner = Planner::new(at.geo, self.plan_config, group.dims.n_rows);
                for s in &group.segments {
                    let table = s.slot * self.table_stride;
                    let need = (s.pos0 as u32 + s.n_rows).div_ceil(ENTRY_CELLS) as usize;
                    let entries = &self.tables[table as usize..table as usize + need];
                    planner.dense(table, entries, s.row0, s.n_rows, s.pos0 as u32);
                }
                let plan = planner.finish();
                let state = plan.state_bytes(&at.geo);
                if state > self.s.attn_state_bytes {
                    return Err(invalid(format!(
                        "the attention plan needs {state} bytes of partial states, {} are reserved",
                        self.s.attn_state_bytes
                    )));
                }
                // The plan goes to the device with the one copy below, which
                // the stream orders in front of the step's kernels; nothing
                // is allocated and the host does not wait.
                let bytes = plan.layout().bytes as usize;
                let buffers = match self.plan_host.get_mut(planned..planned + bytes) {
                    Some(host) => {
                        plan.write(host);
                        let base = self.plan_device.ptr().offset(planned as u64);
                        planned = (planned + bytes).next_multiple_of(64);
                        // SAFETY: the copy below puts the plan's bytes at
                        // `base` before any launch of the step reads them,
                        // and the buffer lives as long as the part.
                        unsafe { plan.on_device(base) }
                    }
                    None => plan.upload(&self.device)?,
                };
                self.plans.push((g, at.geo.kv_heads, buffers));
            }
        }
        if planned > 0 {
            // SAFETY: both buffers hold `planned` bytes (the plans were
            // written inside the pinned one), and the pinned one is not
            // written again before the next step, when this step has run.
            unsafe {
                env.stream.copy_from_host(
                    self.plan_device.ptr(),
                    self.plan_host.as_ptr(),
                    planned,
                )?;
            }
        }
        self.mark(env.stream, Phase::Other)
    }

    /// Cuts a step into groups of rows, which a layer split hands over one by
    /// one and a tensor split sums one by one, and enqueues the upload of
    /// their records. A step that verifies drafts, or a short one, stays
    /// whole.
    fn plan_groups(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        self.groups.clear();
        let n = env.dims.n_rows;
        // A layer split hands over groups of `pipeline_rows` rows once there
        // are two full ones; a tensor split cuts any step of more rows. The
        // groups are full ones and a rest, not equal parts, so that all rows
        // but the rest run at the size the parameter names: the tensor-core
        // kernels are sensitive to it (Qwen3.8-27B on an RTX 4070 with an
        // RTX 3060, the only cards measured: a prompt in groups of 171 rows
        // is read at 1007 tokens per second and in groups of 256 at 1464).
        let per = match (self.split, self.opts.pipeline_rows) {
            (_, 0) | (Split::Single, _) => return Ok(()),
            (Split::Layer, per) if n >= 2 * per => per,
            (Split::Tensor, per) if n > per => per,
            _ => return Ok(()),
        };
        if env.segments.iter().any(|s| !s.direct) {
            return Ok(());
        }
        let mut at = 0u64;
        let mut first = 0;
        while first < env.dims.n_rows {
            let end = (first + per).min(env.dims.n_rows);
            // The parts of the step's segments inside the group, with rows
            // counted from the group's first.
            let segments: Vec<Segment> = env
                .segments
                .iter()
                .filter_map(|s| {
                    let (a, b) = (s.row0.max(first), (s.row0 + s.n_rows).min(end));
                    (a < b).then(|| Segment {
                        row0: a - first,
                        n_rows: b - a,
                        slot: s.slot,
                        pos0: s.pos0 + (a - s.row0) as i32,
                        direct: true,
                    })
                })
                .collect();
            let record =
                ares_cuda::StepRecord::new(&segments, &vec![0; (end - first) as usize], None, &[])?;
            let bytes = record.as_bytes();
            if at as usize + bytes.len() > self.group_host.len() {
                return Err(invalid("the records of the step's groups do not fit their buffer"));
            }
            // SAFETY: the range is inside the pinned buffer (checked above),
            // which no copy reads now: the last step has run.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.group_host.as_mut_ptr().add(at as usize),
                    bytes.len(),
                );
            }
            self.groups.push(Group { first_row: first, dims: record.dims(), segments, record: at });
            at = (at + bytes.len() as u64).next_multiple_of(64);
            first = end;
        }
        // SAFETY: both buffers hold `at` bytes; the pinned one stays unchanged
        // until the step has run.
        unsafe {
            env.stream.copy_from_host(
                self.group_records.ptr(),
                self.group_host.as_ptr(),
                at as usize,
            )?;
        }
        Ok(())
    }

    /// Clears the recurrent state of the slots in which a sequence starts.
    fn start_sequences(&self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.slot_bytes == 0 {
            return Ok(());
        }
        for s in env.segments.iter().filter(|s| s.pos0 == 0) {
            let at = self.state.ptr().offset(u64::from(s.slot) * self.slot_bytes);
            // SAFETY: the slot is inside the state arena (checked in `check`).
            unsafe { env.stream.fill(at, 0, self.slot_bytes as usize)? };
        }
        Ok(())
    }

    /// Enqueues one step: the embedding (or the hand-over from the device
    /// before), this device's blocks, and the head or the hand-over to the
    /// next device. Returns where the logits of the wanted rows will be.
    pub fn enqueue(&mut self, env: &StepEnv<'_>) -> Result<Option<Logits>, ModelError> {
        self.begin(env)?;
        self.start_sequences(env)?;
        // The draft block's chains come first: the rows that verify their
        // drafts read them from the step record.
        self.draft_before(env)?;
        if self.split == Split::Tensor {
            self.enqueue_tensor(env)?;
        } else if self.groups.is_empty() {
            self.enqueue_input(env)?;
            for i in 0..self.w.blocks.len() {
                self.enqueue_block(env, i)?;
            }
            self.enqueue_handover(env)?;
        } else {
            // The first device embeds the whole step; every device then takes
            // the step group by group, so that the next device has a group to
            // work on while this one computes the following.
            if self.index == 0 {
                self.enqueue_input(env)?;
            } else {
                self.enqueue_rope(env)?;
            }
            let groups = std::mem::take(&mut self.groups);
            let mut result = Ok(());
            for (g, group) in groups.iter().enumerate() {
                let group_env = StepEnv {
                    stream: env.stream,
                    step: self.group_records.ptr().offset(group.record),
                    dims: group.dims,
                    segments: &group.segments,
                    tables: env.tables,
                    pages: env.pages,
                    exchange: env.exchange,
                    replay: env.replay,
                };
                (self.group, self.row_base) = (g, group.first_row);
                result = self.enqueue_group(&group_env);
                if result.is_err() {
                    break;
                }
            }
            (self.group, self.row_base) = (0, 0);
            self.groups = groups;
            result?;
        }
        let logits = self.enqueue_head(env)?;
        self.draft_after(env)?;
        self.finish(env)?;
        Ok(logits)
    }

    /// One group of a step under a layer split: its rows from the device
    /// before, this device's blocks, its rows to the device after.
    fn enqueue_group(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.index > 0 {
            self.enqueue_receive(env)?;
        }
        for i in 0..self.w.blocks.len() {
            self.enqueue_block(env, i)?;
        }
        self.enqueue_handover(env)
    }

    /// Takes the rows of the step (or group) from the device before.
    fn enqueue_receive(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let exchange =
            env.exchange.ok_or_else(|| invalid("a layer split needs the paths between devices"))?;
        let bytes = env.dims.n_rows as usize * self.d as usize * 4;
        let received = exchange.receive()?;
        // SAFETY: the parcel holds the rows and `x` has room for them; the
        // copy is ordered after the parcel's arrival by the stream.
        unsafe { env.stream.copy(self.rows(self.s.x, self.d), received, bytes)? };
        exchange.release()?;
        self.mark(env.stream, Phase::Exchange)
    }

    /// Hands the rows of the step (or group) to the device after, under a
    /// layer split.
    fn enqueue_handover(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.split == Split::Layer && self.index + 1 < self.n_devices {
            let exchange = env
                .exchange
                .ok_or_else(|| invalid("a layer split needs the paths between devices"))?;
            exchange.send(self.rows(self.s.x, self.d), env.dims.n_rows * self.d)?;
            self.mark(env.stream, Phase::Exchange)?;
        }
        Ok(())
    }

    /// Enqueues the commit of the deferred segments of the last step:
    /// `step` is a record with those segments and their accepted rows.
    pub fn enqueue_commit(
        &mut self,
        stream: &Stream,
        step: DevicePtr,
        dims: StepDims,
    ) -> Result<(), ModelError> {
        if !std::mem::take(&mut self.has_deferred) {
            return Err(invalid("there is no draft window to commit"));
        }
        self.draft_commit(stream, step, dims)?;
        let kept = self.kept.as_ref().ok_or_else(|| invalid("no draft windows were reserved"))?;
        let (conv_stride, delta_stride) =
            (self.slot_bytes / 4, self.slot_bytes / self.opts.state.bytes());
        for (block, at) in self.w.blocks.iter().zip(&self.recurrent_at) {
            let (MixerW::Recurrent(r), Some(at)) = (&block.mixer, at) else { continue };
            let k = |i: usize| kept.ptr().offset(at.kept[i]);
            // SAFETY: the kept buffers hold the inputs of the last step's
            // rows (it had at most `window_rows` rows), the state arena every
            // slot the record names; all on this device.
            unsafe {
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
                    conv_stride,
                    self.state.ptr().offset(at.delta),
                    delta_stride,
                )?;
            }
        }
        Ok(())
    }

    /// The first stage of [`Part::enqueue`], for tests that drive a block
    /// alone: checks and plans.
    pub fn enqueue_begin(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        self.begin(env)
    }

    /// The last stage of [`Part::enqueue`].
    pub fn enqueue_finish(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        self.finish(env)
    }

    fn finish(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.attention.is_some() {
            // SAFETY: the pinned word lives as long as the part and is read
            // only after the step has run.
            unsafe {
                env.stream.copy_to_host(
                    self.status_host.as_mut_ptr(),
                    self.at(self.s.status),
                    4,
                )?;
            }
            self.status_pending = true;
        }
        self.has_deferred = env.segments.iter().any(|s| !s.direct);
        self.mark(env.stream, Phase::Other)
    }

    fn enqueue_input(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let (stream, dims) = (env.stream, env.dims);
        let x = self.at(self.s.x);
        if let Some(table) = &self.w.embedding {
            // SAFETY: the table is in the arena, `x` holds the step's rows.
            unsafe {
                self.layers.embed_gather(
                    stream,
                    env.step,
                    dims,
                    self.w.mat(table),
                    table.ty,
                    table.n_out,
                    table.n_in,
                    x,
                )?;
            }
            self.mark(stream, Phase::Layers)?;
        }
        if self.split == Split::Layer && self.index > 0 {
            self.enqueue_receive(env)?;
        }
        self.enqueue_rope(env)
    }

    /// Enqueues the rotary table of the step's rows, which every attention
    /// block of the step reads. [`Part::enqueue`] does this with the input.
    pub fn enqueue_rope(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        if self.attention.is_some() {
            // SAFETY: the table buffer holds a row of pairs per step row.
            unsafe {
                self.layers.rope_table(
                    env.stream,
                    env.step,
                    env.dims,
                    self.at(self.s.rope),
                    self.rope,
                )?;
            }
            self.mark(env.stream, Phase::Attention)?;
        }
        Ok(())
    }

    fn enqueue_head(&mut self, env: &StepEnv<'_>) -> Result<Option<Logits>, ModelError> {
        let (stream, dims) = (env.stream, env.dims);
        let Some(head) = self.w.head.clone() else { return Ok(None) };
        if dims.n_out == 0 {
            return Ok(None);
        }
        let (picked, logits) = (self.at(self.s.picked), self.at(self.s.logits));
        let parts: Vec<&Mat> = head.parts.iter().collect();
        // SAFETY: `picked` holds `logit_rows` rows of `d` values, `logits`
        // as many rows of this device's slice of the vocabulary and, for a
        // head in parts, `logit_parts` the same; `n_out` is at most
        // `logit_rows` (checked in `check`).
        unsafe {
            self.layers.gather_rows(
                stream,
                env.step,
                dims,
                self.at(self.s.x),
                Some(self.w.at(self.w.output_norm)),
                picked,
                self.d,
                self.eps,
            )?;
            self.mark(stream, Phase::Layers)?;
            self.prepare(stream, picked, self.d, dims.n_out, &parts, Prepared::default())?;
            match (parts.as_slice(), dims.n_out) {
                ([whole], n) => self.product(stream, whole, n, logits)?,
                // One row: every part writes its columns where they belong.
                (parts, 1) => {
                    let mut at = 0u64;
                    for part in parts {
                        self.product(stream, part, 1, logits.offset(at * 4))?;
                        at += u64::from(part.n_out);
                    }
                }
                // Several rows: a part's rows lie as far apart as it is wide,
                // so they are computed aside and put in their place.
                (parts, n) => {
                    let (mut aside, mut at) = (self.at(self.s.logit_parts), 0u32);
                    for part in parts {
                        self.product(stream, part, n, aside)?;
                        self.layers
                            .place_rows(stream, aside, logits, n, part.n_out, head.n_out, at)?;
                        aside = aside.offset(u64::from(n) * u64::from(part.n_out) * 4);
                        at += part.n_out;
                    }
                }
            }
        }
        self.mark(stream, Phase::Weights)?;
        Ok(Some(Logits { ptr: logits, stride: head.n_out, first_token: head.first_token }))
    }

    /// Enqueues block `i` of this part for the rows of the step.
    pub fn enqueue_block(&mut self, env: &StepEnv<'_>, i: usize) -> Result<(), ModelError> {
        if self.split == Split::Tensor {
            return Err(invalid("a block of a tensor split cannot be enqueued without its merges"));
        }
        let block: BlockW = self.w.blocks[i].clone();
        match &block.mixer {
            MixerW::Recurrent(r) => {
                let at = self.recurrent_at[i].ok_or_else(|| invalid("recurrent state"))?;
                self.recurrent(env, &block, r, at)?;
            }
            MixerW::Attention(a) => {
                let at = self.attention_at[i].ok_or_else(|| invalid("attention pool"))?;
                self.attend(env, &block, a, at)?;
            }
            MixerW::Empty => self.zero_partial(env)?,
        }
        self.merge(env)?;
        match &block.ffn {
            Some(ffn) => self.ffn(env, &block, ffn)?,
            None => self.zero_partial(env)?,
        }
        self.merge(env)
    }

    fn recurrent(
        &mut self,
        env: &StepEnv<'_>,
        block: &BlockW,
        r: &RecurrentW,
        at: RecurrentAt,
    ) -> Result<(), ModelError> {
        let (stream, dims, n) = (env.stream, env.dims, env.dims.n_rows);
        let s = self.s;
        let shape = r.shape;
        let deferred = env.segments.iter().any(|s| !s.direct);
        // A step that verifies drafts keeps the inputs of this block until its
        // commit; any other step uses the scratch every block shares.
        let (qkv, kv, alpha, beta) = match (&self.kept, deferred) {
            (Some(kept), true) => {
                let k = |i: usize| kept.ptr().offset(at.kept[i]);
                (k(0), k(1), k(2), k(3))
            }
            (None, true) => return Err(invalid("no draft windows were reserved")),
            _ => (
                self.rows(s.w1, shape.conv_dim()),
                self.rows(s.kv, shape.key_dim() + shape.value_dim()),
                self.rows(s.alpha, shape.n_v_heads),
                self.rows(s.beta, shape.n_v_heads),
            ),
        };
        let (z, o) = (self.rows(s.w2, shape.value_dim()), self.rows(s.o, shape.value_dim()));
        let q = self.rows(s.q, shape.key_dim());
        let conv_state = self.state.ptr().offset(at.conv);
        let delta_state = self.state.ptr().offset(at.delta);
        let (conv_stride, delta_stride) =
            (self.slot_bytes / 4, self.slot_bytes / self.opts.state.bytes());
        // SAFETY: every buffer is scratch of this part sized for the most
        // rows and the widest block, weights in the arena, or the state
        // arena with a slot per sequence (segments checked in `check`); the
        // kept buffers hold `window_rows` rows, which bounds a deferred step.
        unsafe {
            self.normed_input(env, self.w.at(block.attn_norm), &[&r.qkv, &r.z, &r.beta, &r.alpha])?;
            self.product(stream, &r.qkv, n, qkv)?;
            self.product(stream, &r.z, n, z)?;
            self.product(stream, &r.beta, n, beta)?;
            self.product(stream, &r.alpha, n, alpha)?;
            self.mark(stream, Phase::Weights)?;
            self.layers.conv_window(
                stream,
                env.step,
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
                env.step,
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
                env.step,
                dims,
                shape,
                CommitSet::Direct,
                qkv,
                conv_state,
                conv_stride,
            )?;
            self.gated_norm(stream, r, o, z, n)?;
            self.mark(stream, Phase::Layers)?;
            self.product(stream, &r.out, n, self.rows(s.p, self.d))?;
        }
        self.mark(stream, Phase::Weights)
    }

    fn attend(
        &mut self,
        env: &StepEnv<'_>,
        block: &BlockW,
        a: &AttentionW,
        at: AttentionAt,
    ) -> Result<(), ModelError> {
        let (stream, dims, n) = (env.stream, env.dims, env.dims.n_rows);
        let s = self.s;
        let kernels = self.attention.clone().ok_or_else(|| invalid("attention kernels"))?;
        let hd = at.geo.head_dim;
        let (hq, hkv) = (a.n_head * hd, a.n_kv * hd);
        let (qg, k_raw, v) = (self.rows(s.w1, 2 * hq), self.rows(s.w2, hkv), self.rows(s.v, hkv));
        let (q, gate) = (self.rows(s.q, hq), self.rows(s.kv, hq));
        let (k, o) = (self.rows(s.k, hkv), self.rows(s.o, hq));
        let rope = self.rows(s.rope, self.rope.dim);
        let pool = env.pages.offset(at.pool);
        let shape = AttentionShape { n_head: a.n_head, n_kv_head: a.n_kv, head_dim: hd };
        // SAFETY: scratch and weights as in `recurrent`; the tables hold the
        // entries the plan names (the plan was built from their host copy)
        // and the pool the pages those entries name.
        unsafe {
            self.normed_input(env, self.w.at(block.attn_norm), &[&a.q, &a.k, &a.v])?;
            self.product(stream, &a.q, n, qg)?;
            self.product(stream, &a.k, n, k_raw)?;
            self.product(stream, &a.v, n, v)?;
            self.mark(stream, Phase::Weights)?;
            self.layers.attn_prep(
                stream,
                env.step,
                dims,
                shape,
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
            let plan = self
                .plans
                .iter()
                .find(|(group, heads, _)| *group == self.group && *heads == a.n_kv)
                .map(|(_, _, plan)| plan)
                .ok_or_else(|| invalid("the step has no attention plan"))?;
            // The rows of a replay read cells that are cached already and
            // append none.
            resume::append_writing(
                &kernels,
                stream,
                &at.geo,
                plan.rows(),
                k,
                v,
                env.tables,
                pool,
                self.at(s.status),
                env.replay,
                self.row_base,
                n,
            )?;
            kernels.attend(
                stream,
                &at.geo,
                plan,
                &AttendBuffers {
                    tables: env.tables,
                    exact_pool: DevicePtr::NULL,
                    converted_pool: pool,
                    q,
                    gate: Some(gate),
                    out: o,
                    scratch: self.at(s.attn_state),
                },
                1.0 / (self.attn_head_dim as f32).sqrt(),
            )?;
            self.mark(stream, Phase::Attention)?;
            self.prepare(stream, o, a.out.n_in, n, &[&a.out], Prepared::default())?;
            self.product(stream, &a.out, n, self.rows(s.p, self.d))?;
        }
        self.mark(stream, Phase::Weights)
    }

    fn ffn(&mut self, env: &StepEnv<'_>, block: &BlockW, f: &FfnW) -> Result<(), ModelError> {
        let (stream, n) = (env.stream, env.dims.n_rows);
        let s = self.s;
        let inner = f.gate.n_out;
        let (gate, up) = (self.rows(s.w1, inner), self.rows(s.w2, inner));
        // SAFETY: scratch and weights as in `recurrent`.
        unsafe {
            self.normed_input(env, self.w.at(block.post_norm), &[&f.gate, &f.up])?;
            self.product(stream, &f.gate, n, gate)?;
            self.product(stream, &f.up, n, up)?;
            self.mark(stream, Phase::Weights)?;
            self.swiglu(stream, gate, up, inner, n, &f.down)?;
            self.mark(stream, Phase::Layers)?;
            self.product(stream, &f.down, n, self.rows(s.p, self.d))?;
        }
        self.mark(stream, Phase::Weights)
    }

    // --- access for tests ------------------------------------------------------------

    /// Where block `i` of this part keeps the convolution windows and the
    /// delta-rule matrices of `slot`, as offsets and lengths in bytes of
    /// [`Part::state`]; `None` for a block that is not recurrent here.
    pub fn recurrent_state_at(&self, i: usize, slot: u32) -> Option<[(u64, u64); 2]> {
        let (MixerW::Recurrent(r), Some(at)) = (&self.w.blocks[i].mixer, self.recurrent_at[i])
        else {
            return None;
        };
        let base = u64::from(slot) * self.slot_bytes;
        Some([
            (base + at.conv, r.shape.conv_state_len() * 4),
            (base + at.delta, r.shape.delta_state_len() * self.opts.state.bytes()),
        ])
    }

    /// The arena of recurrent state.
    pub fn state(&self) -> &DeviceBuffer {
        &self.state
    }

    /// Writes K and V rows of `n` consecutive cells from cell `pos0` of the
    /// sequence in `slot` into the pool of block `i`, as a step does:
    /// `k` and `v` are `[cell][kv head][head_dim]` f32 on the device. Waits
    /// for the device.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv(
        &mut self,
        stream: &Stream,
        tables: DevicePtr,
        pages: DevicePtr,
        i: usize,
        slot: u32,
        pos0: u32,
        n: u32,
        k: DevicePtr,
        v: DevicePtr,
    ) -> Result<(), ModelError> {
        let at = self.attention_at[i].ok_or_else(|| invalid("not an attention block"))?;
        let kernels = self.attention.clone().ok_or_else(|| invalid("attention kernels"))?;
        let table = slot * self.table_stride;
        let need = (pos0 + n).div_ceil(ENTRY_CELLS) as usize;
        let mut planner = Planner::new(at.geo, self.plan_config, n);
        planner.dense(table, &self.tables[table as usize..table as usize + need], 0, n, pos0);
        let plan = planner.finish().upload(&self.device)?;
        // SAFETY: the caller's buffers hold `n` rows of the block's KV heads;
        // tables and pool as in `attend`.
        unsafe {
            kernels.append(
                stream,
                &at.geo,
                plan.rows(),
                n,
                k,
                v,
                tables,
                DevicePtr::NULL,
                pages.offset(at.pool),
                self.at(self.s.status),
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }
}
