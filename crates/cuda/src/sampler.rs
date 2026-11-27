//! The sampler's kernels on a device: the pick of each row from its logits.
//!
//! Logits stay on the device. Per row, the kernels of `kernels/sampler.cu`
//! correct the logits (the entries and the mask of the row), find the best few
//! tokens (the greedy pick and the candidates a grammar may fall back to), run
//! the stages of the row's program that cut the set of tokens or scale the
//! temperature, draw the pick, and report the probabilities the row asks for.
//! Only an 80-byte record per row ([`SamplePick`]) and, when asked, a record of
//! probabilities ([`LogprobOut`]) reach the host.
//!
//! What a pick is, the arithmetic and the reference that the kernels are
//! compared with bit for bit are in the `ares-sampler` crate; its
//! documentation defines every stage.
//!
//! # Several devices
//!
//! In a tensor split each device holds the logits of a slice of the
//! vocabulary. Every reduction of the sampler is a maximum or an integer sum,
//! so the devices run the same kernels on their slices and exchange small
//! partial results between the steps of a pick ([`SampleStep`]); the exchange
//! is the sum across devices the model has anyway (f32 values that add
//! exactly: numbers are split into limbs of 16 or 20 bits). All devices end
//! with the same record.

use ares_sampler::{
    ENTRIES_PER_ROW, LOGPROBS_MAX, MAX_STAGES, SAMPLE_CANDIDATES, SampleRow, Stage,
};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sampler.fatbin"));

/// The most logits one device holds per row: the histograms' limbs are sized
/// for it.
pub const SAMPLE_MAX_VOCAB: u32 = 1 << 18;

const THREADS: u32 = 256;
const BINS: u32 = 2048;
const BIN_WORDS: u32 = 4;
const CHUNK: u32 = 4096;
/// A thread of a kernel that passes over a row reads at most this many logits
/// of its chunk (`ARES_SAMPLE_PER` in the kernels).
const PER_THREAD: u32 = 16;
const _: () = assert!(CHUNK <= PER_THREAD * THREADS);
const MAX_CHUNKS: u32 = SAMPLE_MAX_VOCAB / CHUNK;
/// Entries per chunk in the list of a top kernel with the most candidates.
const LIST: u32 = LOGPROBS_MAX as u32 + 1;
/// f32 values per device and row in the exchange of the candidates: a logit
/// and a token each, and two words of status.
const fn top_values(candidates: usize) -> u32 {
    2 * candidates as u32 + 2
}
const PICK_VALUES: u32 = 2;
const PASS_VALUES: u32 = 24;
/// The tokens mirostat 1 estimates its exponent from.
const GATHER: u32 = ares_sampler::MIROSTAT_M as u32;
const GATHER_VALUES: u32 = 1 + 2 * GATHER;
const STATE_BYTES: u64 = 256;
/// The blocks that cover the entries of a row.
const ENTRY_BLOCKS: u32 = (ENTRIES_PER_ROW as u32).div_ceil(THREADS);

/// Which cut a level of the radix descent works on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Descent {
    /// The `k` most probable tokens, by count.
    TopK = 0,
    /// The most probable tokens that hold a share, by weight.
    TopP = 1,
    /// The radius of locally typical sampling, by weight.
    Typical = 2,
    /// The hundredth most probable token (mirostat 1), by count.
    K100 = 3,
}

/// What a pass over the row's tokens adds up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassMode {
    /// The sums of the set: count, weight, entropy term, deviation terms.
    Stats = 0,
    /// The tokens above XTC's threshold.
    Xtc = 1,
    /// The keys within the radius of typical sampling.
    Typical = 2,
    /// The keys below the hundredth (mirostat 1).
    Gather = 3,
    /// The sums of the set that is left, after the draw.
    Tail = 4,
    /// The sums of all tokens as the model gave their logits.
    Raw = 5,
}

/// What the kernel after a pass does with its totals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    NSigma = 0,
    TypicalMean = 1,
    TypicalSet = 2,
    XtcTarget = 3,
    XtcCut = 4,
    Temperature = 5,
    Mirostat2 = 6,
    Mirostat1 = 7,
    Tail = 8,
    Raw = 9,
}

/// One part of the picks of a batch. The parts run in the order of
/// [`SamplePlan::steps`]; where a part ends with partial results that the
/// devices must add up, [`SamplerKernels::enqueue`] says how many values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleStep {
    /// The best tokens of this device's slice of the model's own logits (for
    /// the rows that report the model's probabilities).
    RawTop,
    /// The best tokens of all devices, the same (several devices only).
    RawJoin,
    /// A pass at `position`: this device's partial result.
    Pass(PassMode, u32),
    /// What a stage at `position` does with the totals of its pass.
    Stage(Phase, u32),
    /// The corrections of the rows: entries, then masks.
    Correct,
    /// The best tokens of this device's slice.
    Top,
    /// The best tokens of all devices (several devices only).
    Join,
    /// A level of a descent at `position`: this device's histogram.
    Level(Descent, u32, u32),
    /// The same level: one level down from the histogram of all devices.
    Decide(Descent, u32, u32),
    /// Mirostat 1 at `position`: the first hundred keys of this device.
    Gather(u32),
    /// Mirostat 1 at `position`: the `k` of the keys of all devices.
    Mirostat1(u32),
    /// The best score of this device's slice.
    Pick,
    /// The pick.
    Finish,
    /// The best tokens of the set the chain left (after the chain).
    PostTop,
    /// The same for all devices (several devices only).
    PostJoin,
    /// The logit of the pick, from the device that holds it.
    PickLogit,
    /// The probabilities and the surprise of the pick.
    Report,
}

/// What the rows of a batch need, which decides the kernels that run: a batch
/// of greedy rows takes two launches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplePlan {
    /// Some row is drawn and not the most probable token.
    sampled: bool,
    /// Some row has entries or a mask.
    entries: bool,
    mask: bool,
    /// Some row reports the model's own probabilities / those after the chain.
    raw: bool,
    post: bool,
    /// Some row reports a surprise.
    surprise: bool,
    /// For each position of the programs, a bit for every stage some row has
    /// there (bit `n` for the stage with code `n`, bit 10 for a dynamic
    /// temperature).
    stages: [u16; MAX_STAGES],
}

const DYNAMIC: u16 = 1 << 10;

fn bit(stage: Stage) -> u16 {
    1 << (stage as u16)
}

impl SamplePlan {
    /// The plan of a batch with these rows.
    pub fn of<'a>(rows: impl IntoIterator<Item = &'a SampleRow>) -> Self {
        let mut plan = Self::default();
        for row in rows {
            plan.sampled |= row.sampled();
            plan.entries |= row.entries().1 > 0;
            plan.mask |= row.mask().is_some();
            if let Some((after_chain, _)) = row.logprobs() {
                plan.raw |= !after_chain;
                plan.post |= after_chain;
            }
            plan.surprise |= row.mirostat();
            if row.sampled() {
                for (i, stage) in row.program().into_iter().enumerate() {
                    plan.stages[i] |= bit(stage);
                    if stage == Stage::Temperature && row.dynamic_temperature() {
                        plan.stages[i] |= DYNAMIC;
                    }
                }
            }
        }
        plan
    }

    /// Whether any row reports probabilities, which come back in a record of
    /// their own.
    pub fn logprobs(&self) -> bool {
        self.raw || self.post
    }

    /// Whether any row reports a probability or a surprise after its pick.
    fn reports(&self) -> bool {
        self.raw || self.post || self.surprise
    }

    /// The parts of the picks, in order, for `devices` devices.
    pub fn steps(&self, devices: u32) -> Vec<SampleStep> {
        let several = devices > 1;
        let mut steps = Vec::new();
        if self.raw {
            steps.push(SampleStep::RawTop);
            if several {
                steps.push(SampleStep::RawJoin);
            }
            steps.push(SampleStep::Pass(PassMode::Raw, 0));
            steps.push(SampleStep::Stage(Phase::Raw, 0));
        }
        if self.entries || self.mask {
            steps.push(SampleStep::Correct);
        }
        steps.push(SampleStep::Top);
        if several {
            steps.push(SampleStep::Join);
        }
        if self.sampled {
            for (position, kinds) in self.stages.iter().enumerate() {
                let position = position as u32;
                let has = |stage: Stage| kinds & bit(stage) != 0;
                if has(Stage::Mirostat1) {
                    self.levels(&mut steps, Descent::K100, position);
                    steps.push(SampleStep::Gather(position));
                    steps.push(SampleStep::Mirostat1(position));
                }
                if has(Stage::TopK) || has(Stage::Mirostat1) {
                    self.levels(&mut steps, Descent::TopK, position);
                }
                if has(Stage::TopNSigma) {
                    steps.push(SampleStep::Pass(PassMode::Stats, position));
                    steps.push(SampleStep::Stage(Phase::NSigma, position));
                }
                if has(Stage::Typical) {
                    steps.push(SampleStep::Pass(PassMode::Stats, position));
                    steps.push(SampleStep::Stage(Phase::TypicalMean, position));
                    self.levels(&mut steps, Descent::Typical, position);
                    steps.push(SampleStep::Pass(PassMode::Typical, position));
                    steps.push(SampleStep::Stage(Phase::TypicalSet, position));
                }
                if has(Stage::TopP) {
                    self.levels(&mut steps, Descent::TopP, position);
                }
                if has(Stage::Xtc) {
                    steps.push(SampleStep::Pass(PassMode::Stats, position));
                    steps.push(SampleStep::Stage(Phase::XtcTarget, position));
                    steps.push(SampleStep::Pass(PassMode::Xtc, position));
                    steps.push(SampleStep::Stage(Phase::XtcCut, position));
                }
                if has(Stage::Temperature) && kinds & DYNAMIC != 0 {
                    steps.push(SampleStep::Pass(PassMode::Stats, position));
                    steps.push(SampleStep::Stage(Phase::Temperature, position));
                }
                if has(Stage::Mirostat2) {
                    steps.push(SampleStep::Pass(PassMode::Stats, position));
                    steps.push(SampleStep::Stage(Phase::Mirostat2, position));
                }
            }
            steps.push(SampleStep::Pick);
            steps.push(SampleStep::Finish);
        }
        if self.post || self.surprise {
            steps.push(SampleStep::Pass(PassMode::Tail, 0));
            steps.push(SampleStep::Stage(Phase::Tail, 0));
        }
        if self.post {
            steps.push(SampleStep::PostTop);
            if several {
                steps.push(SampleStep::PostJoin);
            }
        }
        if self.reports() {
            steps.push(SampleStep::PickLogit);
            steps.push(SampleStep::Report);
        }
        steps
    }

    fn levels(&self, steps: &mut Vec<SampleStep>, descent: Descent, position: u32) {
        for level in 0..3 {
            steps.push(SampleStep::Level(descent, position, level));
            steps.push(SampleStep::Decide(descent, position, level));
        }
    }
}

/// One device's view of the rows to pick.
#[derive(Clone, Copy, Debug)]
pub struct SampleBatch {
    /// Row `r` has its logits at `logits + r * stride * 4`, `n_local` f32.
    /// Rows that have corrections are changed in place.
    pub logits: DevicePtr,
    pub stride: u32,
    pub n_rows: u32,
    /// The rows the scratch area and the exchange buffer were sized for. The
    /// place of a row's scratch follows from this and not from `n_rows`, so
    /// that batches of different sizes use the same memory the same way.
    pub capacity: u32,
    /// The logits this device holds per row, at most [`SAMPLE_MAX_VOCAB`].
    pub n_local: u32,
    /// Column `i` is token `tokens[first + i]` (u32), or `first + i` when
    /// this is null.
    pub tokens: DevicePtr,
    pub first: u32,
    /// For the columns of a token list: column of every token of the
    /// vocabulary (u32, `u32::MAX` where this device holds none), see
    /// [`SamplerKernels::invert`]. Null without a token list.
    pub inverse: DevicePtr,
    /// The number of tokens of the vocabulary.
    pub vocab: u32,
    /// [`SampleRow::BYTES`] bytes per row.
    pub params: DevicePtr,
    /// The corrections of the rows (16 bytes each, `Entry`); the rows name
    /// theirs. Null when no row has any.
    pub entries: DevicePtr,
    /// The masks of the rows: [`SampleBatch::mask_words`] u32 each, a bit
    /// per token. Null when no row has one.
    pub masks: DevicePtr,
    pub mask_words: u32,
    /// [`SamplePick::BYTES`](ares_sampler::SamplePick::BYTES) bytes per row,
    /// written by the kernels.
    pub out: DevicePtr,
    /// [`LogprobOut::BYTES`](ares_sampler::LogprobOut::BYTES) bytes per row,
    /// written for the rows that report probabilities. Null when none does.
    pub logprobs: DevicePtr,
    /// [`SamplerKernels::scratch_bytes`]`(capacity)` bytes that the caller
    /// cleared once ([`SamplerKernels::clear`]) and that nothing else writes.
    pub scratch: DevicePtr,
    /// [`SamplerKernels::exchange_values`]`(capacity, devices)` f32 values:
    /// what the devices add up between the parts.
    pub exchange: DevicePtr,
    /// The devices that share the vocabulary, and which of them this is.
    pub devices: u32,
    pub rank: u32,
}

/// The sampler's kernels on one device.
#[derive(Debug, Clone)]
pub struct SamplerKernels {
    device: Device,
    top: Function,
    top_merge: Function,
    top_join: Function,
    raw_top: Function,
    post_top: Function,
    lp_merge: Function,
    lp_join: Function,
    scatter: Function,
    mask: Function,
    hist: Function,
    export: Function,
    decide: Function,
    pass: Function,
    pass_export: Function,
    stage_decide: Function,
    gather_export: Function,
    miro1_decide: Function,
    pick: Function,
    pick_export: Function,
    finish: Function,
    pick_logit: Function,
    report: Function,
    invert: Function,
    scores: Function,
    draft_put: Function,
    draft_rows: Function,
}

impl SamplerKernels {
    /// Loads the kernels on `device`.
    pub fn load(device: &Device) -> Result<Self> {
        let module = Module::load(device, IMAGE)?;
        Ok(Self {
            device: device.clone(),
            top: module.function("ares_sample_top")?,
            top_merge: module.function("ares_sample_top_merge")?,
            top_join: module.function("ares_sample_top_join")?,
            raw_top: module.function("ares_lp_top_raw")?,
            post_top: module.function("ares_lp_top_post")?,
            lp_merge: module.function("ares_lp_top_merge")?,
            lp_join: module.function("ares_lp_top_join")?,
            scatter: module.function("ares_sample_scatter")?,
            mask: module.function("ares_sample_mask")?,
            hist: module.function("ares_sample_hist")?,
            export: module.function("ares_sample_export")?,
            decide: module.function("ares_sample_decide")?,
            pass: module.function("ares_sample_pass")?,
            pass_export: module.function("ares_sample_pass_export")?,
            stage_decide: module.function("ares_sample_stage_decide")?,
            gather_export: module.function("ares_sample_gather_export")?,
            miro1_decide: module.function("ares_sample_miro1_decide")?,
            pick: module.function("ares_sample_pick")?,
            pick_export: module.function("ares_sample_pick_export")?,
            finish: module.function("ares_sample_finish")?,
            pick_logit: module.function("ares_lp_pick_logit")?,
            report: module.function("ares_lp_finish")?,
            invert: module.function("ares_sample_invert")?,
            scores: module.function("ares_sample_scores")?,
            draft_put: module.function("ares_draft_put")?,
            draft_rows: module.function("ares_draft_rows")?,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The scratch memory of a batch of up to `rows` rows: the rows' state,
    /// the lists of the chunks and the histograms.
    pub fn scratch_bytes(rows: u32) -> u64 {
        u64::from(rows) * (STATE_BYTES + Self::lists_bytes() + Self::hist_bytes())
    }

    fn lists_bytes() -> u64 {
        u64::from(MAX_CHUNKS * LIST) * 8
    }

    fn hist_bytes() -> u64 {
        u64::from(BINS * BIN_WORDS) * 4
    }

    /// The f32 values of the exchange buffer for up to `rows` rows on
    /// `devices` devices.
    pub fn exchange_values(rows: u32, devices: u32) -> u32 {
        let per_device = top_values(LOGPROBS_MAX).max(PASS_VALUES).max(GATHER_VALUES);
        rows * (BINS * BIN_WORDS).max(devices * per_device)
    }

    /// Enqueues clearing a scratch area for `rows` rows. Once, before its
    /// first use: the kernels leave it as they need it.
    ///
    /// # Safety
    ///
    /// `scratch` must address [`SamplerKernels::scratch_bytes`]`(rows)` bytes
    /// of live memory on the stream's device.
    pub unsafe fn clear(&self, stream: &Stream, scratch: DevicePtr, rows: u32) -> Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe { stream.fill(scratch, 0, Self::scratch_bytes(rows) as usize) }
    }

    /// Enqueues the table of the columns of a device that holds a token list:
    /// `inverse[token]` is the column of the token, or `u32::MAX`. Once, when
    /// the device part is built.
    ///
    /// # Safety
    ///
    /// `tokens` must address `n_local` u32 and `inverse` `vocab` u32 of live
    /// memory on the stream's device, every token below `vocab`.
    pub unsafe fn invert(
        &self,
        stream: &Stream,
        tokens: DevicePtr,
        n_local: u32,
        inverse: DevicePtr,
        vocab: u32,
    ) -> Result<()> {
        let blocks = n_local.div_ceil(THREADS);
        // SAFETY: the arguments match ares_sample_invert; the memory is the caller's guarantee.
        unsafe {
            stream.fill(inverse, 0xff, vocab as usize * 4)?;
            self.invert.launch(
                stream,
                [blocks, 1, 1],
                [THREADS, 1, 1],
                0,
                Args::new().ptr(tokens).u32(n_local).ptr(inverse).u32(vocab),
            )
        }
    }

    /// Enqueues reading the logits of chosen tokens at the wanted rows of
    /// `batch`, before the picks change them: `pairs` holds `n` pairs of u32
    /// (the wanted row, the token), and `out[k]` becomes the logit of pair `k`,
    /// or NaN when the token is not among this device's columns.
    ///
    /// # Safety
    ///
    /// `batch.logits` must hold `batch.n_rows` rows of `batch.stride` values
    /// (and `batch.inverse`, when it is not null, `batch.vocab` u32), `pairs`
    /// `2 * n` u32 with every row below `batch.n_rows`, and `out` `n` f32, all
    /// live memory of the stream's device.
    pub unsafe fn scores(
        &self,
        stream: &Stream,
        batch: &SampleBatch,
        pairs: DevicePtr,
        n: u32,
        out: DevicePtr,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let first = if batch.inverse == DevicePtr::NULL { batch.first } else { 0 };
        // SAFETY: the arguments match ares_sample_scores; the memory is the caller's guarantee.
        unsafe {
            self.scores.launch(
                stream,
                [n.div_ceil(THREADS), 1, 1],
                [THREADS, 1, 1],
                0,
                Args::new()
                    .ptr(batch.logits)
                    .u32(batch.stride)
                    .u32(first)
                    .u32(batch.n_local)
                    .ptr(batch.inverse)
                    .u32(batch.vocab)
                    .ptr(pairs)
                    .u32(n)
                    .ptr(out),
            )
        }
    }

    /// Enqueues one part of the picks of `batch` and returns the number of
    /// f32 values at `batch.exchange` that all devices must add up before the
    /// next part (0: nothing to exchange). Every device runs the same parts
    /// in the same order.
    ///
    /// # Safety
    ///
    /// The pointers of `batch` must address live memory of the stream's
    /// device with the sizes its fields document, and the scratch area must
    /// hold `batch.n_rows` rows.
    pub unsafe fn enqueue(
        &self,
        stream: &Stream,
        batch: &SampleBatch,
        step: SampleStep,
    ) -> Result<u32> {
        if batch.n_rows == 0 {
            return Ok(0);
        }
        if batch.n_local == 0 || batch.n_local > SAMPLE_MAX_VOCAB {
            return Err(Error::Invalid {
                what: "sampler",
                detail: format!(
                    "{} logits per row; a device holds 1 to {SAMPLE_MAX_VOCAB}",
                    batch.n_local
                ),
            });
        }
        if batch.rank >= batch.devices || batch.devices > 16 {
            return Err(Error::Invalid {
                what: "sampler",
                detail: format!("device {} of {}", batch.rank, batch.devices),
            });
        }
        if batch.n_rows > batch.capacity {
            return Err(Error::Invalid {
                what: "sampler",
                detail: format!("{} rows in a scratch area for {}", batch.n_rows, batch.capacity),
            });
        }
        let rows = batch.n_rows;
        let devices = batch.devices;
        let chunks = batch.n_local.div_ceil(CHUNK);
        let state = batch.scratch;
        // The histograms rely on being left clear by the batch before, so
        // the arrays lie where the capacity puts them, whatever this batch's
        // size.
        let capacity = u64::from(batch.capacity);
        let lists = state.offset(capacity * STATE_BYTES);
        let hist = lists.offset(capacity * Self::lists_bytes());
        let lists_words = (Self::lists_bytes() / 4) as u32;
        let several = devices > 1;
        let block = [THREADS, 1, 1];
        let small = [32, 1, 1];
        let chunk_grid = [chunks, rows, 1];
        // The sum across devices has to cover this many values.
        let sum = |per_row: u32| if several { rows * per_row } else { 0 };
        // SAFETY: the arguments of every launch below match its kernel's
        // parameter list in kernels/sampler_*.cuh, and the caller guarantees
        // the memory behind the pointers.
        unsafe {
            match step {
                SampleStep::RawTop => {
                    self.raw_top.launch(
                        stream,
                        chunk_grid,
                        block,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .ptr(batch.tokens)
                            .u32(batch.first)
                            .u32(CHUNK)
                            .ptr(lists)
                            .ptr(batch.params),
                    )?;
                    self.top_list(stream, batch, lists, chunks, true)?;
                    Ok(sum(devices * top_values(LOGPROBS_MAX)))
                }
                SampleStep::RawJoin => {
                    self.join_list(stream, batch, true)?;
                    Ok(0)
                }
                SampleStep::PostTop => {
                    self.post_top.launch(
                        stream,
                        chunk_grid,
                        block,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .ptr(batch.tokens)
                            .u32(batch.first)
                            .u32(CHUNK)
                            .ptr(lists)
                            .ptr(batch.params)
                            .ptr(state),
                    )?;
                    self.top_list(stream, batch, lists, chunks, false)?;
                    Ok(sum(devices * top_values(LOGPROBS_MAX)))
                }
                SampleStep::PostJoin => {
                    self.join_list(stream, batch, false)?;
                    Ok(0)
                }
                SampleStep::Correct => {
                    if batch.entries != DevicePtr::NULL {
                        self.scatter.launch(
                            stream,
                            [ENTRY_BLOCKS, rows, 1],
                            block,
                            0,
                            Args::new()
                                .ptr(batch.logits)
                                .u32(batch.stride)
                                .u32(batch.n_local)
                                .u32(batch.first)
                                .ptr(batch.inverse)
                                .u32(batch.vocab)
                                .ptr(batch.params)
                                .ptr(batch.entries),
                        )?;
                    }
                    if batch.masks != DevicePtr::NULL {
                        self.mask.launch(
                            stream,
                            chunk_grid,
                            block,
                            0,
                            Args::new()
                                .ptr(batch.logits)
                                .u32(batch.stride)
                                .u32(batch.n_local)
                                .ptr(batch.tokens)
                                .u32(batch.first)
                                .u32(CHUNK)
                                .ptr(batch.params)
                                .ptr(batch.masks)
                                .u32(batch.mask_words),
                        )?;
                    }
                    Ok(0)
                }
                SampleStep::Top => {
                    self.top.launch(
                        stream,
                        chunk_grid,
                        block,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .ptr(batch.tokens)
                            .u32(batch.first)
                            .u32(CHUNK)
                            .ptr(lists),
                    )?;
                    self.top_merge.launch(
                        stream,
                        [rows, 1, 1],
                        block,
                        0,
                        Args::new()
                            .ptr(lists)
                            .u32(chunks)
                            .u32(devices)
                            .u32(batch.rank)
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(batch.out),
                    )?;
                    Ok(sum(devices * top_values(SAMPLE_CANDIDATES)))
                }
                SampleStep::Join => {
                    self.top_join.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(batch.exchange)
                            .u32(devices)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(batch.out),
                    )?;
                    Ok(0)
                }
                SampleStep::Level(descent, position, level) => {
                    self.hist.launch(
                        stream,
                        chunk_grid,
                        block,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .u32(CHUNK)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(hist)
                            .u32(descent as u32)
                            .u32(level)
                            .u32(position),
                    )?;
                    let bins = if level == 1 { BINS } else { BINS / 2 };
                    self.export.launch(
                        stream,
                        [bins / THREADS, rows, 1],
                        block,
                        0,
                        Args::new()
                            .ptr(hist)
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .u32(descent as u32)
                            .u32(position),
                    )?;
                    Ok(sum(BINS * BIN_WORDS))
                }
                SampleStep::Decide(descent, position, level) => {
                    self.decide.launch(
                        stream,
                        [rows, 1, 1],
                        block,
                        0,
                        Args::new()
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .u32(descent as u32)
                            .u32(level)
                            .u32(position),
                    )?;
                    Ok(0)
                }
                SampleStep::Pass(mode, position) => {
                    self.pass.launch(
                        stream,
                        chunk_grid,
                        block,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .u32(CHUNK)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(lists)
                            .u32(lists_words)
                            .u32(mode as u32)
                            .u32(position),
                    )?;
                    self.pass_export.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(state)
                            .ptr(batch.params)
                            .ptr(batch.exchange)
                            .u32(devices)
                            .u32(batch.rank)
                            .u32(mode as u32)
                            .u32(position),
                    )?;
                    Ok(sum(devices * PASS_VALUES))
                }
                SampleStep::Stage(phase, position) => {
                    self.stage_decide.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .u32(devices)
                            .u32(phase as u32)
                            .u32(position),
                    )?;
                    Ok(0)
                }
                SampleStep::Gather(position) => {
                    self.pass.launch(
                        stream,
                        chunk_grid,
                        block,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .u32(CHUNK)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(lists)
                            .u32(lists_words)
                            .u32(PassMode::Gather as u32)
                            .u32(position),
                    )?;
                    self.gather_export.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(state)
                            .ptr(batch.params)
                            .ptr(lists)
                            .u32(lists_words)
                            .ptr(batch.exchange)
                            .u32(devices)
                            .u32(batch.rank)
                            .u32(position),
                    )?;
                    Ok(sum(devices * GATHER_VALUES))
                }
                SampleStep::Mirostat1(position) => {
                    self.miro1_decide.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .u32(devices)
                            .u32(position),
                    )?;
                    Ok(0)
                }
                SampleStep::Pick => {
                    self.pick.launch(
                        stream,
                        chunk_grid,
                        block,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .ptr(batch.tokens)
                            .u32(batch.first)
                            .u32(CHUNK)
                            .ptr(batch.params)
                            .ptr(state),
                    )?;
                    if !several {
                        return Ok(0);
                    }
                    self.pick_export.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(state)
                            .ptr(batch.params)
                            .ptr(batch.exchange)
                            .u32(devices)
                            .u32(batch.rank),
                    )?;
                    Ok(rows * devices * PICK_VALUES)
                }
                SampleStep::Finish => {
                    self.finish.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(state)
                            .ptr(batch.exchange)
                            .u32(devices)
                            .ptr(batch.params)
                            .ptr(batch.out),
                    )?;
                    Ok(0)
                }
                SampleStep::PickLogit => {
                    self.pick_logit.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .u32(batch.first)
                            .ptr(batch.inverse)
                            .u32(batch.vocab)
                            .ptr(batch.params)
                            .ptr(batch.entries)
                            .ptr(batch.out)
                            .ptr(batch.exchange)
                            .u32(devices)
                            .u32(batch.rank),
                    )?;
                    Ok(sum(devices * PASS_VALUES))
                }
                SampleStep::Report => {
                    self.report.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(batch.exchange)
                            .u32(devices)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(batch.out)
                            .ptr(batch.logprobs),
                    )?;
                    Ok(0)
                }
            }
        }
    }

    /// The merge of the chunks' lists of the rows that report probabilities
    /// (`raw`: the model's own, else after the chain).
    ///
    /// # Safety
    ///
    /// As [`SamplerKernels::enqueue`].
    unsafe fn top_list(
        &self,
        stream: &Stream,
        batch: &SampleBatch,
        lists: DevicePtr,
        chunks: u32,
        raw: bool,
    ) -> Result<()> {
        // SAFETY: the arguments match ares_lp_top_merge; the memory is the caller's guarantee.
        unsafe {
            self.lp_merge.launch(
                stream,
                [batch.n_rows, 1, 1],
                [THREADS, 1, 1],
                0,
                Args::new()
                    .ptr(lists)
                    .u32(chunks)
                    .u32(batch.devices)
                    .u32(batch.rank)
                    .ptr(batch.exchange)
                    .ptr(batch.params)
                    .ptr(batch.scratch)
                    .ptr(batch.logprobs)
                    .u32(u32::from(raw)),
            )
        }
    }

    /// The join of the devices' lists of the rows that report probabilities.
    ///
    /// # Safety
    ///
    /// As [`SamplerKernels::enqueue`].
    unsafe fn join_list(&self, stream: &Stream, batch: &SampleBatch, raw: bool) -> Result<()> {
        // SAFETY: the arguments match ares_lp_top_join; the memory is the caller's guarantee.
        unsafe {
            self.lp_join.launch(
                stream,
                [batch.n_rows, 1, 1],
                [32, 1, 1],
                0,
                Args::new()
                    .ptr(batch.exchange)
                    .u32(batch.devices)
                    .ptr(batch.params)
                    .ptr(batch.scratch)
                    .ptr(batch.logprobs)
                    .u32(u32::from(raw)),
            )
        }
    }

    /// Enqueues all parts of the picks of `batch` for rows that need `plan`.
    /// `exchange(ptr, n)` is called between parts when there are several
    /// devices, to enqueue the sum of `n` f32 values at `ptr` over all of
    /// them.
    ///
    /// # Safety
    ///
    /// As [`SamplerKernels::enqueue`].
    pub unsafe fn sample<E: From<Error>>(
        &self,
        stream: &Stream,
        batch: &SampleBatch,
        plan: SamplePlan,
        mut exchange: impl FnMut(DevicePtr, u32) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        for step in plan.steps(batch.devices) {
            // SAFETY: the caller's guarantee.
            let values = unsafe { self.enqueue(stream, batch, step)? };
            if values > 0 {
                exchange(batch.exchange, values)?;
            }
        }
        Ok(())
    }
}

/// What drafts that stay on the device need besides the block that computes
/// them (`kernels/sampler_draft.cuh`).
///
/// The *carry* of a sequence is `carry_rows` rows of `d` f32 values per
/// sequence slot. After a step it holds the final hidden state of the
/// sequence's rows of that step: of the last row in row 0 when all rows were
/// certain, of every row of a window that verified drafts otherwise; the
/// commit of the window then moves the row of the last accepted token to row
/// 0. So before a step's kernels, row 0 is the hidden state the sequence's
/// newest token was picked from, which is what a draft chain starts from.
impl SamplerKernels {
    /// The tokens of `n` picks at `picks` (`ares_sampler::SamplePick` records) go where
    /// the next kernels read them: pick `c` to `next_tokens[c]` for `c <
    /// n_next` (the token array of the next link's step record), to
    /// `record_tokens[record_rows[c]]` (the token of the row that verifies
    /// it) and to `drafts[draft_at[c]]` (what the host reads).
    ///
    /// # Safety
    ///
    /// The pointers must address live memory of the stream's device: `n`
    /// picks, `n` indices in `record_rows` and `draft_at`, and u32 arrays that
    /// hold every index named.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn draft_put(
        &self,
        stream: &Stream,
        picks: DevicePtr,
        n: u32,
        next_tokens: DevicePtr,
        n_next: u32,
        record_tokens: DevicePtr,
        record_rows: DevicePtr,
        drafts: DevicePtr,
        draft_at: DevicePtr,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let args = Args::new()
            .ptr(picks)
            .u32(n)
            .ptr(next_tokens)
            .u32(n_next)
            .ptr(record_tokens)
            .ptr(record_rows)
            .ptr(drafts)
            .ptr(draft_at);
        // SAFETY: the arguments match ares_draft_put; the memory is the caller's guarantee.
        unsafe { self.draft_put.launch(stream, [1, 1, 1], [32, 1, 1], 0, args) }
    }

    /// `dst[r] = src[index[r]]` for `n` rows of `d` f32 values; `index` has
    /// `n` u32.
    ///
    /// # Safety
    ///
    /// The pointers must address live memory of the stream's device of those
    /// sizes, and `src` every row `index` names.
    pub unsafe fn gather_hidden(
        &self,
        stream: &Stream,
        dst: DevicePtr,
        src: DevicePtr,
        index: DevicePtr,
        n: u32,
        d: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(dst).ptr(src).ptr(index).u32(d);
        // SAFETY: the arguments match ares_draft_rows; the memory is the caller's guarantee.
        unsafe { self.draft_rows.launch(stream, [n, 1, 1], [THREADS, 1, 1], 0, args) }
    }
}
