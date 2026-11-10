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

use std::sync::{Arc, Mutex};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{
    AttendBuffers, AttentionKernels, AttentionShape, CacheFormat, CacheGeometry, CommitSet, Device,
    DeviceBuffer, DevicePtr, ENTRY_CELLS, Event, Kernels, LayerKernels, MAX_BATCH, OutputGate,
    PUSH_MAX_VALUES, PinnedBuffer, PlanBuffers, PlanConfig, Planner, Precision, PrefillPath,
    RopeParams, Segment, StateFormat, StepDims, Stream,
};
use ares_runtime::{Logits, ModelError, Split};

use super::load::{AttentionW, BlockW, DeviceWeights, FfnW, Mat, MixerW, RecurrentW};
use super::placement::Placement;
use crate::qwen35::Config;
use crate::qwen35::weights::Weights;

/// Bit 31 of a table entry: the page is in the pool of stored formats (the
/// only pool the formats without a tile mean use).
pub const PAGE_ENTRY: u32 = 1 << 31;

/// The choices of a device implementation.
#[derive(Clone, Debug)]
pub struct Options {
    /// How the matrix products round: [`Precision::Fast`] takes the decode
    /// kernels (8-bit activations) for steps of up to eight rows and each
    /// type's fastest tensor-core path above; [`Precision::Exact`] takes the
    /// f16 tensor-core path for every product.
    pub precision: Precision,
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
    /// Under a layer split, the rows a device hands to the next at a time: a
    /// step of at least twice as many certain rows is cut into groups of this
    /// size, and the devices work on different groups of the step at once.
    /// 0 hands every step over whole.
    pub pipeline_rows: u32,
    /// Where the parts add up the device time of their kernels, by kind.
    pub profile: Option<Arc<Profile>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            precision: Precision::Fast,
            state: StateFormat::F32,
            kv: (CacheFormat::Q8, CacheFormat::Q8),
            max_rows: 512,
            logit_rows: 8,
            window_rows: 0,
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
    /// Sums across devices and hand-overs, including the wait for the other
    /// device.
    Exchange = 3,
    /// Everything else between the start of a step and its end.
    Other = 4,
}

/// The phases, in the order of their numbers.
pub const PHASES: [Phase; 5] =
    [Phase::Weights, Phase::Layers, Phase::Attention, Phase::Exchange, Phase::Other];

/// Device time per phase and device, added up over steps.
#[derive(Debug, Default)]
pub struct Profile {
    inner: Mutex<ProfileData>,
}

#[derive(Clone, Debug, Default)]
struct ProfileData {
    /// `[device][phase]` milliseconds.
    ms: Vec<[f64; 5]>,
    steps: Vec<u64>,
}

impl Profile {
    /// An empty profile.
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&self, device: usize, ms: &[f64; 5]) {
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if data.ms.len() <= device {
            data.ms.resize(device + 1, [0.0; 5]);
            data.steps.resize(device + 1, 0);
        }
        for (sum, v) in data.ms[device].iter_mut().zip(ms) {
            *sum += v;
        }
        data.steps[device] += 1;
    }

    /// Per device: the steps counted and the milliseconds per phase (indexed
    /// by the phase's number), summed over those steps.
    pub fn totals(&self) -> Vec<(u64, [f64; 5])> {
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
    /// The phase that ends at each event (the first event starts the step).
    phases: Vec<Phase>,
}

/// The paths to the other devices, as a part uses them.
pub trait Exchange {
    /// Sums `n` f32 values at `partial` across all devices, in place on each.
    fn sum(&self, partial: DevicePtr, n: u32) -> Result<(), ModelError>;
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

/// A group of consecutive rows of a step that a layer split hands over on
/// its own: a smaller step, whose record is at `record` of the group records.
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
    rows_per_sum: u32,
    page_bytes: u64,
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
        .sum()
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
        let PartSetup { index, device, kernels, ledger, max_slots, pages, table_stride } = setup;
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
        let w = DeviceWeights::load(device, ledger, c, weights, plan)?;

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
                    let conv = slot_bytes;
                    let delta = (conv + s.conv_state_len() * 4).next_multiple_of(64);
                    slot_bytes =
                        (delta + s.delta_state_len() * opts.state.bytes()).next_multiple_of(64);
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
        s.act_q = take(
            Kernels::quantized_activation_bytes(widest as u32, opts.max_rows)
                .max(Kernels::quantized_activation_bytes(d as u32, opts.logit_rows)),
        );
        s.act_h = take(Kernels::prefill_activation_bytes(
            PrefillPath::F16,
            widest as u32,
            opts.max_rows.max(opts.logit_rows),
        ));
        s.picked = take(logit_rows * d * 4);
        let head_rows = w.head.as_ref().map_or(0, |(m, _)| u64::from(m.n_out));
        s.logits = take(logit_rows * head_rows * 4);
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
        let group_bytes = if placement.split == Split::Layer && opts.pipeline_rows > 0 {
            let groups = u64::from(opts.max_rows.div_ceil(opts.pipeline_rows));
            (groups * (96 + 32 * u64::from(max_slots)) + 24 * rows).next_multiple_of(64)
        } else {
            64
        };
        ledger.reserve("qwen35 group records", group_bytes)?;
        let has_attention = attention_at.iter().any(Option::is_some);
        let attention = has_attention.then(|| AttentionKernels::load(device)).transpose()?;
        let marks = opts.profile.as_ref().map(|profile| Marks {
            profile: profile.clone(),
            events: Vec::new(),
            phases: Vec::new(),
        });
        let rope = &c.attention.rope;
        Ok(Self {
            index,
            n_devices: placement.n_devices(),
            split: placement.split,
            opts: opts.clone(),
            device: device.clone(),
            kernels: kernels.clone(),
            layers: LayerKernels::load(device)?,
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
            kept,
            has_deferred: false,
            table_stride,
            max_slots,
            tables: vec![0; max_slots as usize * table_stride as usize],
            plan_config,
            plans: Vec::new(),
            groups: Vec::new(),
            group_host: PinnedBuffer::new(device, group_bytes as usize)?,
            group_records: DeviceBuffer::new(device, group_bytes)?,
            group: 0,
            row_base: 0,
            status_host: PinnedBuffer::new(device, 16)?,
            status_pending: false,
            marks,
            rows_per_sum: (PUSH_MAX_VALUES / d as u32).clamp(1, opts.max_rows),
            page_bytes,
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
        self.w.head.as_ref().map_or(0, |(m, _)| m.n_out)
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

    /// Adds the marks of the last step to the profile. The step must have
    /// run: the next step's enqueue calls this, and a caller that reads the
    /// profile after the last step does.
    pub fn flush_profile(&mut self) -> Result<(), ModelError> {
        let Some(marks) = &mut self.marks else { return Ok(()) };
        if marks.phases.len() > 1 {
            let mut ms = [0.0f64; 5];
            for i in 1..marks.phases.len() {
                let elapsed = marks.events[i].elapsed_ms_since(&marks.events[i - 1])?;
                ms[marks.phases[i] as usize] += f64::from(elapsed);
            }
            marks.profile.add(self.index, &ms);
        }
        marks.phases.clear();
        Ok(())
    }

    // --- products ----------------------------------------------------------------

    /// Which kernel family multiplies a matrix of `ty` with `n` rows: `None`
    /// for the decode kernels.
    fn path(&self, ty: GgmlType, n: u32) -> Result<Option<PrefillPath>, ModelError> {
        match self.opts.precision {
            Precision::Exact => Ok(Some(self.kernels.prefill_path(Precision::Exact)?)),
            Precision::Fast => {
                let decode = Kernels::decode_kernel_rows(ty)
                    .or_else(|| Kernels::has_fast_matvec(ty).then_some(MAX_BATCH));
                match decode {
                    Some(limit) if n <= limit.min(MAX_BATCH) => Ok(None),
                    _ => Ok(Some(self.kernels.prefill_path_for(ty, Precision::Fast)?)),
                }
            }
        }
    }

    /// Prepares `n` rows of `n_in` values at `x` for the products with
    /// `mats`: quantized into `act_q`, as f16 into `act_h`, or both. With
    /// `fused`, `act_q` already holds the rows in the layout of the fused
    /// kernels and only the f16 form may be missing.
    ///
    /// # Safety
    ///
    /// `x` must address `n * n_in` f32 values on the stream's device.
    unsafe fn prepare(
        &self,
        stream: &Stream,
        x: DevicePtr,
        n_in: u32,
        n: u32,
        mats: &[&Mat],
        fused: bool,
    ) -> Result<(), ModelError> {
        let (mut int8, mut f16) = (false, false);
        for m in mats {
            match self.path(m.ty, n)? {
                None | Some(PrefillPath::Int8) => int8 = true,
                Some(PrefillPath::F16 | PrefillPath::F16Flush) => f16 = true,
            }
        }
        let (act_q, act_h) = (self.at(self.s.act_q), self.at(self.s.act_h));
        // SAFETY: the scratch buffers hold the prepared form of the widest
        // input for the most rows; `x` is the caller's guarantee.
        unsafe {
            if int8 && !fused {
                if n <= MAX_BATCH {
                    self.kernels.quantize_activations(stream, x, act_q, n_in, n)?;
                } else {
                    self.kernels.prepare_prefill_activations(
                        stream,
                        PrefillPath::Int8,
                        x,
                        act_q,
                        n_in,
                        n,
                    )?;
                }
            }
            if f16 {
                let source = if fused {
                    // The f32 copy at the end of the fused layout.
                    let values = u64::from(n_in) * u64::from(n);
                    act_q.offset(ares_cuda::quantized_activation_bytes(values) - values * 4)
                } else {
                    x
                };
                self.kernels.prepare_prefill_activations(
                    stream,
                    PrefillPath::F16,
                    source,
                    act_h,
                    n_in,
                    n,
                )?;
            }
        }
        Ok(())
    }

    /// `y = m x` for the `n` rows that [`Part::prepare`] prepared.
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

    /// Sums the partial result `p` of the step's rows across the devices
    /// (tensor split) and adds it to the residual stream.
    fn merge(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        let n = env.dims.n_rows;
        let (x, p) = (self.rows(self.s.x, self.d), self.rows(self.s.p, self.d));
        if self.split == Split::Tensor {
            self.sum(env, self.s.p)?;
            self.mark(env.stream, Phase::Exchange)?;
        }
        // SAFETY: `x` and `p` hold the step's rows of `d` values.
        unsafe { self.kernels.add_f32(env.stream, x, p, n * self.d)? };
        self.mark(env.stream, Phase::Layers)
    }

    fn sum(&self, env: &StepEnv<'_>, offset: u64) -> Result<(), ModelError> {
        let exchange = env
            .exchange
            .ok_or_else(|| invalid("a tensor split needs the paths between devices"))?;
        let mut row = 0;
        while row < env.dims.n_rows {
            let rows = self.rows_per_sum.min(env.dims.n_rows - row);
            let at = self.at(offset).offset(u64::from(row) * u64::from(self.d) * 4);
            exchange.sum(at, rows * self.d)?;
            row += rows;
        }
        Ok(())
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
                self.plans.push((g, at.geo.kv_heads, plan.upload(&self.device)?));
            }
        }
        self.mark(env.stream, Phase::Other)
    }

    /// Cuts a step of a layer split into groups of rows that are handed over
    /// one by one, and enqueues the upload of their records. A step that
    /// verifies drafts, or a short one, stays whole.
    fn plan_groups(&mut self, env: &StepEnv<'_>) -> Result<(), ModelError> {
        self.groups.clear();
        let per = self.opts.pipeline_rows;
        if self.split != Split::Layer
            || per == 0
            || env.dims.n_rows < 2 * per
            || env.segments.iter().any(|s| !s.direct)
        {
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
        if self.groups.is_empty() {
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
        let bytes = dims.n_rows as usize * self.d as usize * 4;
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
        match self.split {
            Split::Single => {}
            Split::Tensor => {
                if self.w.embedding.is_none() {
                    // SAFETY: `x` holds the step's rows.
                    unsafe { stream.fill(x, 0, bytes)? };
                }
                self.sum(env, self.s.x)?;
                self.mark(stream, Phase::Exchange)?;
            }
            Split::Layer => {
                if self.index > 0 {
                    self.enqueue_receive(env)?;
                }
            }
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
        let Some((head, first_token)) = self.w.head else { return Ok(None) };
        if dims.n_out == 0 {
            return Ok(None);
        }
        let (picked, logits) = (self.at(self.s.picked), self.at(self.s.logits));
        // SAFETY: `picked` holds `logit_rows` rows of `d` values and `logits`
        // as many rows of this device's slice of the vocabulary; `n_out` is
        // at most `logit_rows` (checked in `check`).
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
            self.prepare(stream, picked, self.d, dims.n_out, &[&head], false)?;
            self.product(stream, &head, dims.n_out, logits)?;
        }
        self.mark(stream, Phase::Weights)?;
        Ok(Some(Logits { ptr: logits, stride: head.n_out, first_token }))
    }

    /// Enqueues block `i` of this part for the rows of the step.
    pub fn enqueue_block(&mut self, env: &StepEnv<'_>, i: usize) -> Result<(), ModelError> {
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
        let (x, xn) = (self.rows(s.x, self.d), self.rows(s.xn, self.d));
        let (z, o) = (self.rows(s.w2, shape.value_dim()), self.rows(s.o, shape.value_dim()));
        let (q, act_q) = (self.rows(s.q, shape.key_dim()), self.at(s.act_q));
        let conv_state = self.state.ptr().offset(at.conv);
        let delta_state = self.state.ptr().offset(at.delta);
        let (conv_stride, delta_stride) =
            (self.slot_bytes / 4, self.slot_bytes / self.opts.state.bytes());
        // SAFETY: every buffer is scratch of this part sized for the most
        // rows and the widest block, weights in the arena, or the state
        // arena with a slot per sequence (segments checked in `check`); the
        // kept buffers hold `window_rows` rows, which bounds a deferred step.
        unsafe {
            self.kernels.rms_norm_f32(
                stream,
                x,
                self.w.at(block.attn_norm),
                xn,
                self.d,
                n,
                self.eps,
            )?;
            self.mark(stream, Phase::Layers)?;
            self.prepare(stream, xn, self.d, n, &[&r.qkv, &r.z, &r.beta, &r.alpha], false)?;
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
            self.layers.gated_norm_quant(
                stream,
                OutputGate::Silu,
                o,
                z,
                self.w.at(r.ssm_norm),
                act_q,
                n * shape.n_v_heads,
                shape.head_dim,
                self.eps,
            )?;
            self.mark(stream, Phase::Layers)?;
            self.prepare(stream, act_q, shape.value_dim(), n, &[&r.out], true)?;
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
        let (x, xn) = (self.rows(s.x, self.d), self.rows(s.xn, self.d));
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
            self.kernels.rms_norm_f32(
                stream,
                x,
                self.w.at(block.attn_norm),
                xn,
                self.d,
                n,
                self.eps,
            )?;
            self.mark(stream, Phase::Layers)?;
            self.prepare(stream, xn, self.d, n, &[&a.q, &a.k, &a.v], false)?;
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
            kernels.append(
                stream,
                &at.geo,
                plan.rows(),
                n,
                k,
                v,
                env.tables,
                DevicePtr::NULL,
                pool,
                self.at(s.status),
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
            self.prepare(stream, o, a.out.n_in, n, &[&a.out], false)?;
            self.product(stream, &a.out, n, self.rows(s.p, self.d))?;
        }
        self.mark(stream, Phase::Weights)
    }

    fn ffn(&mut self, env: &StepEnv<'_>, block: &BlockW, f: &FfnW) -> Result<(), ModelError> {
        let (stream, n) = (env.stream, env.dims.n_rows);
        let s = self.s;
        let inner = f.gate.n_out;
        let (x, xn) = (self.rows(s.x, self.d), self.rows(s.xn, self.d));
        let (gate, up, act_q) = (self.rows(s.w1, inner), self.rows(s.w2, inner), self.at(s.act_q));
        // SAFETY: scratch and weights as in `recurrent`.
        unsafe {
            self.kernels.rms_norm_f32(
                stream,
                x,
                self.w.at(block.post_norm),
                xn,
                self.d,
                n,
                self.eps,
            )?;
            self.mark(stream, Phase::Layers)?;
            self.prepare(stream, xn, self.d, n, &[&f.gate, &f.up], false)?;
            self.product(stream, &f.gate, n, gate)?;
            self.product(stream, &f.up, n, up)?;
            self.mark(stream, Phase::Weights)?;
            self.layers.swiglu_quant(stream, gate, up, act_q, n * inner)?;
            self.mark(stream, Phase::Layers)?;
            self.prepare(stream, act_q, inner, n, &[&f.down], true)?;
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
