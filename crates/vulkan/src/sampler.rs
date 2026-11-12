//! The sampler: the pick of each row from its logits, on the device.
//!
//! This is `ares_cuda`'s sampler for Vulkan: the same definitions of a pick, the same arithmetic
//! operation for operation, the same records and the same steps, so that a request gets the same
//! tokens on either backend. Logits stay on the device. Per row, the kernels of `shaders/sample_*.comp`
//! find the best few tokens (the greedy pick and the candidates a grammar may fall back to), the
//! exact cut-offs of top-k, top-p and min-p, and the pick, and only an 80-byte record per row
//! reaches the host ([`SamplePick`]).
//!
//! # What a pick is
//!
//! With `p` the softmax of a row's logits and `p_max` its largest value, every cut is a
//! threshold on `p / p_max`, which is `exp(-d)` for the distance `d` of a logit below the best
//! one:
//!
//! - **top-k** keeps the tokens at least as probable as the `k`-th;
//! - **top-p** keeps the most probable tokens that together hold the share `top_p` of what top-k
//!   left;
//! - **min-p** keeps the tokens with `p / p_max >= min_p`.
//!
//! The cuts apply in that order at temperature 1, as in llama.cpp's default chain, and tokens
//! tied at a cut are all kept. Among the tokens left, the pick is the one with the largest
//! `logit / temperature + g`, where `g` is the Gumbel noise of (seed, position, token) from a
//! counter-based generator (Philox 4x32-10). That is a draw from the distribution of the kept
//! tokens at that temperature, and because a token's noise depends on nothing but those three
//! numbers, a row's pick is the same alone and in any batch, on one device and on several, and
//! verifying a draft is comparing it with the pick for its position.
//!
//! # Exactness
//!
//! The cut-offs come from a radix descent over the bit patterns of `d` with integer counts and
//! fixed-point weights, the exponential and the logarithm are polynomials of IEEE operations (no
//! fused multiply-add: `precise` in the shaders), so the order in which invocations, workgroups
//! and devices add does not matter and the host can repeat a pick bit for bit.
//! `ares_cuda::sample_reference` is the same arithmetic written from the definitions above with a
//! sort instead of the descent; the tests of this crate hold the kernels to it.
//!
//! # Against `ares_cuda`
//!
//! - The host types and constants below are copies of `ares_cuda`'s (`SampleParams`,
//!   `SampleRow`, `SamplePick`, `SamplePlan`, `SampleStep`, `SampleBatch`); the host arithmetic
//!   (`sample_reference` and the functions under it) is not copied: it lives in `ares_cuda` and
//!   the tests of this crate take it from there, so that the two backends are held to one
//!   definition. A backend-free crate for both is the place for them
//!   (`docs/ARCHITECTURE.md`).
//! - The best score of a row is not raised by a 64-bit atomic (an optional feature on Vulkan):
//!   every chunk writes its best, and `sample_pick_merge` takes the maximum. The row's state is
//!   64 bytes here as [`SamplerKernels::scratch_bytes`] counts it (CUDA's struct is 72 bytes
//!   against 64 reserved per row).
//! - The histograms of top-k and top-p are built in 32 KiB of shared memory: a device with less
//!   is refused at [`SamplerKernels::load`].
//! - Pipelines are made on first use or by [`SamplerKernels::prepare`].
//!
//! # Several devices
//!
//! In a tensor split each device holds the logits of a slice of the vocabulary. Every reduction
//! of the sampler is a maximum or an integer sum, so the devices run the same kernels on their
//! slices and exchange small partial results between the steps of a pick ([`SampleStep`]); the
//! exchange is the sum across devices the model has anyway (f32 values that add exactly). All
//! devices end with the same record.

use std::sync::Arc;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::kernels::parallel_map;
use crate::memory::DevicePtr;
use crate::module::{Args, Lazy};
use crate::stream::Stream;
use crate::weights::image;

/// Bytes of shared memory the histograms of top-k and top-p need in a workgroup.
const HISTOGRAM_SHARED_BYTES: u32 = BINS * BIN_WORDS * 4;

/// The candidates recorded per row: its best tokens by logit.
pub const SAMPLE_CANDIDATES: usize = 8;
/// Status bit of a [`SamplePick`]: a logit of the row was NaN or plus
/// infinity; the pick is not valid.
pub const SAMPLE_STATUS_NONFINITE: u32 = 1;
/// Status bit of a [`SamplePick`]: no token was left to pick.
pub const SAMPLE_STATUS_EMPTY: u32 = 2;

/// The most logits one device holds per row: the histograms' limbs are sized
/// for it.
pub const SAMPLE_MAX_VOCAB: u32 = 1 << 18;

const THREADS: u32 = 256;
const BINS: u32 = 2048;
const BIN_WORDS: u32 = 4;
const CHUNK: u32 = 4096;
const MAX_CHUNKS: u32 = SAMPLE_MAX_VOCAB / CHUNK;
const LIST: u32 = SAMPLE_CANDIDATES as u32 + 1;
const TOP_VALUES: u32 = 2 * SAMPLE_CANDIDATES as u32 + 2;
const PICK_VALUES: u32 = 2;
const STATE_BYTES: u64 = 64;

const F_SAMPLE: u32 = 1;
const F_TOP_K: u32 = 2;
const F_TOP_P: u32 = 4;
const KEY_MAX: u32 = 0x7f7f_ffff;

/// How a sequence's tokens are picked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
    /// 0 picks the most probable token.
    pub temperature: f32,
    /// 0 is off.
    pub top_k: u32,
    /// 1 (or more) is off.
    pub top_p: f32,
    /// 0 is off.
    pub min_p: f32,
    pub seed: u64,
}

impl SampleParams {
    /// The most probable token.
    pub const GREEDY: SampleParams =
        SampleParams { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0 };

    /// The device's form of these settings for a row at `position`.
    pub fn row(&self, position: u32) -> SampleRow {
        let sampled = self.temperature > 0.0;
        let mut flags = 0;
        let mut row = SampleRow {
            flags: 0,
            inv_temperature: 1.0,
            top_k: 0,
            top_p_q32: 0,
            min_p_key: KEY_MAX,
            seed_lo: self.seed as u32,
            seed_hi: (self.seed >> 32) as u32,
            position,
        };
        if sampled {
            flags |= F_SAMPLE;
            row.inv_temperature = 1.0 / self.temperature;
            if self.top_k > 0 {
                flags |= F_TOP_K;
                row.top_k = self.top_k;
            }
            if self.top_p > 0.0 && self.top_p < 1.0 {
                flags |= F_TOP_P;
                let q = (f64::from(self.top_p) * 4_294_967_296.0).round();
                row.top_p_q32 = q.clamp(1.0, 4_294_967_295.0) as u32;
            }
            if self.min_p > 0.0 {
                let d = (-self.min_p.ln()).max(0.0);
                row.min_p_key = d.to_bits().min(KEY_MAX);
            }
        }
        row.flags = flags;
        row
    }
}

/// One row's settings as the kernels read them (`AresSampleParams`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRow {
    flags: u32,
    inv_temperature: f32,
    top_k: u32,
    top_p_q32: u32,
    min_p_key: u32,
    seed_lo: u32,
    seed_hi: u32,
    position: u32,
}

impl SampleRow {
    /// The bytes of a row on the device.
    pub const BYTES: usize = 32;

    /// Whether the row is sampled (and not the most probable token).
    pub fn sampled(&self) -> bool {
        self.flags & F_SAMPLE != 0
    }

    /// Reads a row as [`SampleRow::to_bytes`] wrote it.
    pub fn from_bytes(bytes: &[u8; Self::BYTES]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        Self {
            flags: word(0),
            inv_temperature: f32::from_bits(word(1)),
            top_k: word(2),
            top_p_q32: word(3),
            min_p_key: word(4),
            seed_lo: word(5),
            seed_hi: word(6),
            position: word(7),
        }
    }

    /// The row as the device reads it.
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let words = [
            self.flags,
            self.inv_temperature.to_bits(),
            self.top_k,
            self.top_p_q32,
            self.min_p_key,
            self.seed_lo,
            self.seed_hi,
            self.position,
        ];
        let mut bytes = [0u8; Self::BYTES];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *chunk = word.to_le_bytes();
        }
        bytes
    }
}

/// What the device reports per row (`AresSampleOut`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplePick {
    /// The picked token, or `u32::MAX` when there is none (see `status`).
    pub token: u32,
    /// 0, or `SAMPLE_STATUS_*` bits.
    pub status: u32,
    /// The logit of a greedy pick; the score (`logit / temperature + noise`)
    /// of a sampled one.
    pub value: f32,
    /// The row's best tokens by logit, best first, the lower token first
    /// among equal logits. `u32::MAX` where the row has fewer tokens.
    pub cand_token: [u32; SAMPLE_CANDIDATES],
    pub cand_logit: [f32; SAMPLE_CANDIDATES],
}

impl SamplePick {
    /// The bytes of a record on the device.
    pub const BYTES: usize = 16 + 8 * SAMPLE_CANDIDATES;

    /// Reads a record as the device wrote it.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        let mut pick = SamplePick {
            token: word(0),
            status: word(1),
            value: f32::from_bits(word(2)),
            cand_token: [0; SAMPLE_CANDIDATES],
            cand_logit: [0.0; SAMPLE_CANDIDATES],
        };
        for c in 0..SAMPLE_CANDIDATES {
            pick.cand_token[c] = word(4 + c);
            pick.cand_logit[c] = f32::from_bits(word(4 + SAMPLE_CANDIDATES + c));
        }
        pick
    }
}

/// Which cut a level of the descent works on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleCut {
    TopK = 0,
    TopP = 1,
}

/// One part of the picks of a batch. The parts run in the order of
/// [`SamplePlan::steps`]; where a part ends with partial results that the
/// devices must add up, [`SamplerKernels::enqueue`] says how many values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleStep {
    /// The best tokens of this device's slice.
    Top,
    /// The best tokens of all devices (several devices only).
    Join,
    /// A level of a descent: this device's histogram.
    Level(SampleCut, u32),
    /// The same level: one level down from the histogram of all devices.
    Decide(SampleCut, u32),
    /// The best score of this device's slice.
    Pick,
    /// The pick.
    Finish,
}

/// What the rows of a batch need, which decides the kernels that run: a batch
/// of greedy rows takes two launches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplePlan {
    pub sampled: bool,
    pub top_k: bool,
    pub top_p: bool,
}

impl SamplePlan {
    /// The plan of a batch with these rows.
    pub fn of<'a>(rows: impl IntoIterator<Item = &'a SampleRow>) -> Self {
        rows.into_iter().fold(Self::default(), |plan, row| Self {
            sampled: plan.sampled || row.flags & F_SAMPLE != 0,
            top_k: plan.top_k || row.flags & F_TOP_K != 0,
            top_p: plan.top_p || row.flags & F_TOP_P != 0,
        })
    }

    /// The parts of the picks, in order, for `devices` devices.
    pub fn steps(&self, devices: u32) -> Vec<SampleStep> {
        let mut steps = vec![SampleStep::Top];
        if devices > 1 {
            steps.push(SampleStep::Join);
        }
        for (wanted, cut) in [(self.top_k, SampleCut::TopK), (self.top_p, SampleCut::TopP)] {
            if self.sampled && wanted {
                for level in 0..3 {
                    steps.push(SampleStep::Level(cut, level));
                    steps.push(SampleStep::Decide(cut, level));
                }
            }
        }
        if self.sampled {
            steps.push(SampleStep::Pick);
            steps.push(SampleStep::Finish);
        }
        steps
    }
}

/// One device's view of the rows to pick.
#[derive(Clone, Copy, Debug)]
pub struct SampleBatch {
    /// Row `r` has its logits at `logits + r * stride * 4`, `n_local` f32.
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
    /// [`SampleRow::BYTES`] bytes per row.
    pub params: DevicePtr,
    /// [`SamplePick::BYTES`] bytes per row, written by the kernels.
    pub out: DevicePtr,
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
pub struct SamplerKernels(Arc<Set>);

#[derive(Debug)]
struct Set {
    device: Device,
    top: Arc<Lazy>,
    top_merge: Arc<Lazy>,
    top_join: Arc<Lazy>,
    hist: Arc<Lazy>,
    export: Arc<Lazy>,
    decide: Arc<Lazy>,
    pick: Arc<Lazy>,
    pick_merge: Arc<Lazy>,
    pick_export: Arc<Lazy>,
    finish: Arc<Lazy>,
    draft_put: Arc<Lazy>,
    draft_rows: Arc<Lazy>,
    draft_shift: Arc<Lazy>,
    draft_keep: Arc<Lazy>,
    draft_commit: Arc<Lazy>,
}

impl SamplerKernels {
    /// Prepares the kernels on `device`; a pipeline is made when its kernel is first launched, or
    /// by [`prepare`](Self::prepare).
    ///
    /// Refused on a device that cannot run them: one without what the quantized weight kernels
    /// need (clustered subgroup operations among them), or with less than 32 KiB of shared memory
    /// for the histograms of top-k and top-p.
    pub fn load(device: &Device) -> Result<Self> {
        if let Some(reason) = &device.physical().quantized_unsupported {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: format!("the sampler needs what the weight kernels need: {reason}"),
            });
        }
        if device.physical().max_shared_memory < HISTOGRAM_SHARED_BYTES {
            return Err(Error::Unsupported {
                device: device.info().name.clone(),
                reason: format!(
                    "the sampler's histograms need {HISTOGRAM_SHARED_BYTES} bytes of shared memory, the device has {}",
                    device.physical().max_shared_memory
                ),
            });
        }
        let lazy = |name: &str| -> Result<Arc<Lazy>> { Ok(Lazy::new(device, image(name)?)) };
        Ok(Self(Arc::new(Set {
            device: device.clone(),
            top: lazy("sample_top")?,
            top_merge: lazy("sample_top_merge")?,
            top_join: lazy("sample_top_join")?,
            hist: lazy("sample_hist")?,
            export: lazy("sample_export")?,
            decide: lazy("sample_decide")?,
            pick: lazy("sample_pick")?,
            pick_merge: lazy("sample_pick_merge")?,
            pick_export: lazy("sample_pick_export")?,
            finish: lazy("sample_finish")?,
            draft_put: lazy("draft_put")?,
            draft_rows: lazy("draft_rows")?,
            draft_shift: lazy("draft_shift")?,
            draft_keep: lazy("draft_keep")?,
            draft_commit: lazy("draft_commit")?,
        })))
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// Makes the pipelines of every kernel now, on all cores: the runtime calls this when it loads
    /// a model, so that no step pays for a compilation.
    pub fn prepare(&self) -> Result<()> {
        let s = &*self.0;
        let all = [
            &s.top,
            &s.top_merge,
            &s.top_join,
            &s.hist,
            &s.export,
            &s.decide,
            &s.pick,
            &s.pick_merge,
            &s.pick_export,
            &s.finish,
            &s.draft_put,
            &s.draft_rows,
            &s.draft_shift,
            &s.draft_keep,
            &s.draft_commit,
        ];
        parallel_map(&all, |lazy| lazy.get().map(|_| ())).map(|_| ())
    }

    /// The scratch memory of a batch of up to `rows` rows: the rows' state, the lists of the
    /// chunks and the histograms.
    pub fn scratch_bytes(rows: u32) -> u64 {
        u64::from(rows) * (STATE_BYTES + Self::lists_bytes() + Self::hist_bytes())
    }

    fn lists_bytes() -> u64 {
        u64::from(MAX_CHUNKS * LIST) * 8
    }

    fn hist_bytes() -> u64 {
        u64::from(BINS * BIN_WORDS) * 4
    }

    /// The f32 values of the exchange buffer for up to `rows` rows on `devices` devices.
    pub fn exchange_values(rows: u32, devices: u32) -> u32 {
        rows * (BINS * BIN_WORDS).max(devices * TOP_VALUES)
    }

    /// Enqueues clearing a scratch area for `rows` rows. Once, before its first use: the kernels
    /// leave it as they need it.
    ///
    /// # Safety
    ///
    /// `scratch` must address [`SamplerKernels::scratch_bytes`]`(rows)` bytes of live memory on
    /// the stream's device.
    pub unsafe fn clear(&self, stream: &Stream, scratch: DevicePtr, rows: u32) -> Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe { stream.fill(scratch, 0, Self::scratch_bytes(rows) as usize) }
    }

    /// Enqueues one part of the picks of `batch` and returns the number of f32 values at
    /// `batch.exchange` that all devices must add up before the next part (0: nothing to
    /// exchange). Every device runs the same parts in the same order.
    ///
    /// # Safety
    ///
    /// The pointers of `batch` must address live memory of the stream's device with the sizes its
    /// fields document, and the scratch area must hold `batch.n_rows` rows.
    pub unsafe fn enqueue(
        &self,
        stream: &Stream,
        batch: &SampleBatch,
        step: SampleStep,
    ) -> Result<u32> {
        if batch.n_rows == 0 {
            return Ok(0);
        }
        let invalid = |detail: String| Err(Error::invalid("sampler", detail));
        if batch.n_local == 0 || batch.n_local > SAMPLE_MAX_VOCAB {
            return invalid(format!(
                "{} logits per row; a device holds 1 to {SAMPLE_MAX_VOCAB}",
                batch.n_local
            ));
        }
        if batch.rank >= batch.devices || batch.devices > 16 {
            return invalid(format!("device {} of {}", batch.rank, batch.devices));
        }
        if batch.n_rows > batch.capacity {
            return invalid(format!(
                "{} rows in a scratch area for {}",
                batch.n_rows, batch.capacity
            ));
        }
        let rows = batch.n_rows;
        let chunks = batch.n_local.div_ceil(CHUNK);
        let state = batch.scratch;
        // The histograms rely on being left clear by the batch before, so the arrays lie where
        // the capacity puts them, whatever this batch's size.
        let capacity = u64::from(batch.capacity);
        let lists = state.offset(capacity * STATE_BYTES);
        let hist = lists.offset(capacity * Self::lists_bytes());
        let several = batch.devices > 1;
        let s = &*self.0;
        // SAFETY: the arguments of every launch below match its shader's push constants in
        // shaders/sample_*.comp, and the caller guarantees the memory behind the pointers.
        unsafe {
            match step {
                SampleStep::Top => {
                    s.top.get()?.launch(
                        stream,
                        [chunks, rows, 1],
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .ptr(batch.tokens)
                            .u32(batch.first)
                            .u32(CHUNK)
                            .ptr(lists),
                    )?;
                    s.top_merge.get()?.launch(
                        stream,
                        [rows, 1, 1],
                        Args::new()
                            .ptr(lists)
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(batch.out)
                            .u32(chunks)
                            .u32(batch.devices)
                            .u32(batch.rank),
                    )?;
                    Ok(if several { rows * batch.devices * TOP_VALUES } else { 0 })
                }
                SampleStep::Join => {
                    s.top_join.get()?.launch(
                        stream,
                        [rows, 1, 1],
                        Args::new()
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(batch.out)
                            .u32(batch.devices),
                    )?;
                    Ok(0)
                }
                SampleStep::Level(cut, level) => {
                    s.hist.get()?.launch(
                        stream,
                        [chunks, rows, 1],
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .u32(CHUNK)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(hist)
                            .u32(cut as u32)
                            .u32(level),
                    )?;
                    let bins = if level == 1 { BINS } else { BINS / 2 };
                    s.export.get()?.launch(
                        stream,
                        [bins / THREADS, rows, 1],
                        Args::new().ptr(hist).ptr(batch.exchange).ptr(batch.params).u32(cut as u32),
                    )?;
                    Ok(if several { rows * BINS * BIN_WORDS } else { 0 })
                }
                SampleStep::Decide(cut, level) => {
                    s.decide.get()?.launch(
                        stream,
                        [rows, 1, 1],
                        Args::new()
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .u32(cut as u32)
                            .u32(level),
                    )?;
                    Ok(0)
                }
                SampleStep::Pick => {
                    s.pick.get()?.launch(
                        stream,
                        [chunks, rows, 1],
                        Args::new()
                            .ptr(batch.logits)
                            .u32(batch.stride)
                            .u32(batch.n_local)
                            .ptr(batch.tokens)
                            .u32(batch.first)
                            .u32(CHUNK)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(lists),
                    )?;
                    s.pick_merge.get()?.launch(
                        stream,
                        [rows, 1, 1],
                        Args::new().ptr(lists).ptr(batch.params).ptr(state).u32(chunks),
                    )?;
                    if !several {
                        return Ok(0);
                    }
                    s.pick_export.get()?.launch(
                        stream,
                        [rows, 1, 1],
                        Args::new()
                            .ptr(state)
                            .ptr(batch.params)
                            .ptr(batch.exchange)
                            .u32(batch.devices)
                            .u32(batch.rank),
                    )?;
                    Ok(rows * batch.devices * PICK_VALUES)
                }
                SampleStep::Finish => {
                    s.finish.get()?.launch(
                        stream,
                        [rows, 1, 1],
                        Args::new()
                            .ptr(state)
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(batch.out)
                            .u32(batch.devices),
                    )?;
                    Ok(0)
                }
            }
        }
    }

    /// Enqueues all parts of the picks of `batch` for rows that need `plan`. `exchange(ptr, n)`
    /// is called between parts when there are several devices, to enqueue the sum of `n` f32
    /// values at `ptr` over all of them.
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

/// What drafts that stay on the device need besides the block that computes them
/// (`shaders/draft_*.comp`).
///
/// The *carry* of a sequence is `carry_rows` rows of `d` f32 values per sequence slot. After a
/// step it holds the final hidden state of the sequence's rows of that step: of the last row in
/// row 0 when all rows were certain, of every row of a window that verified drafts otherwise;
/// the commit of the window then moves the row of the last accepted token to row 0. So before a
/// step's kernels, row 0 is the hidden state the sequence's newest token was picked from, which
/// is what a draft chain starts from.
impl SamplerKernels {
    /// The tokens of `n` picks at `picks` ([`SamplePick`] records) go where the next kernels read
    /// them: pick `c` to `next_tokens[c]` for `c < n_next` (the token array of the next link's
    /// step record), to `record_tokens[record_rows[c]]` (the token of the row that verifies it)
    /// and to `drafts[draft_at[c]]` (what the host reads).
    ///
    /// # Safety
    ///
    /// The pointers must address live memory of the stream's device: `n` picks, `n` indices in
    /// `record_rows` and `draft_at`, and u32 arrays that hold every index named.
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
            .ptr(next_tokens)
            .ptr(record_tokens)
            .ptr(record_rows)
            .ptr(drafts)
            .ptr(draft_at)
            .u32(n)
            .u32(n_next);
        // SAFETY: the arguments match draft_put.comp; the memory is the caller's guarantee.
        unsafe { self.0.draft_put.get()?.launch(stream, [1, 1, 1], args) }
    }

    /// `dst[r] = src[index[r]]` for `n` rows of `d` f32 values; `index` has `n` u32.
    ///
    /// # Safety
    ///
    /// The pointers must address live memory of the stream's device of those sizes, and `src`
    /// every row `index` names.
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
        // SAFETY: the arguments match draft_rows.comp; the memory is the caller's guarantee.
        unsafe { self.0.draft_rows.get()?.launch(stream, [n, 1, 1], args) }
    }

    /// For every row of the step at `step` (a record in the layout of the layer kernels, with
    /// `segments` segments of at most `max_seg_rows` rows), the final hidden state of the
    /// position before it: `out[r]` is `hidden[r - 1]` inside a segment and the sequence's carry
    /// (zeros at position 0) for a segment's first row.
    ///
    /// # Safety
    ///
    /// `step` must address the uploaded record, `hidden` and `out` a row of `d` f32 per row of
    /// the step, `carry` `carry_rows * d` f32 per slot the record names, on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn shift_hidden(
        &self,
        stream: &Stream,
        step: DevicePtr,
        segments: u32,
        max_seg_rows: u32,
        hidden: DevicePtr,
        carry: DevicePtr,
        carry_rows: u32,
        out: DevicePtr,
        d: u32,
    ) -> Result<()> {
        if segments == 0 || max_seg_rows == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(step).ptr(hidden).ptr(carry).ptr(out).u32(carry_rows).u32(d);
        // SAFETY: the arguments match draft_shift.comp; the memory is the caller's guarantee.
        unsafe { self.0.draft_shift.get()?.launch(stream, [max_seg_rows, segments, 1], args) }
    }

    /// Keeps the final hidden state of the step's rows in the carries of their sequences.
    ///
    /// # Safety
    ///
    /// As [`SamplerKernels::shift_hidden`]; a segment that is not direct has at most
    /// `carry_rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn keep_hidden(
        &self,
        stream: &Stream,
        step: DevicePtr,
        segments: u32,
        max_seg_rows: u32,
        hidden: DevicePtr,
        carry: DevicePtr,
        carry_rows: u32,
        d: u32,
    ) -> Result<()> {
        if segments == 0 || max_seg_rows == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(step).ptr(hidden).ptr(carry).u32(carry_rows).u32(d);
        // SAFETY: the arguments match draft_keep.comp; the memory is the caller's guarantee.
        unsafe { self.0.draft_keep.get()?.launch(stream, [max_seg_rows, segments, 1], args) }
    }

    /// The commit of the windows of the last step, whose record with the accepted rows is at
    /// `commit` (`segments` segments): row 0 of each sequence's carry becomes the hidden state of
    /// its last accepted row.
    ///
    /// # Safety
    ///
    /// `commit` must address the uploaded commit record and `carry` the carries of every slot it
    /// names, on the stream's device.
    pub unsafe fn commit_hidden(
        &self,
        stream: &Stream,
        commit: DevicePtr,
        segments: u32,
        carry: DevicePtr,
        carry_rows: u32,
        d: u32,
    ) -> Result<()> {
        if segments == 0 {
            return Ok(());
        }
        let args = Args::new().ptr(commit).ptr(carry).u32(carry_rows).u32(d);
        // SAFETY: the arguments match draft_commit.comp; the memory is the caller's guarantee.
        unsafe { self.0.draft_commit.get()?.launch(stream, [segments, 1, 1], args) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_of_settings_round_trips_through_its_bytes() {
        let p =
            SampleParams { temperature: 0.7, top_k: 40, top_p: 0.9, min_p: 0.05, seed: 1 << 40 };
        let row = p.row(123);
        assert_eq!(SampleRow::from_bytes(&row.to_bytes()), row);
        assert!(row.sampled());
        assert!(!SampleParams::GREEDY.row(0).sampled());
    }

    #[test]
    fn the_plan_of_a_batch_names_the_steps_in_order() {
        let steps = SamplePlan { sampled: true, top_k: true, top_p: false }.steps(2);
        assert_eq!(steps[0], SampleStep::Top);
        assert_eq!(steps[1], SampleStep::Join);
        assert_eq!(steps[2], SampleStep::Level(SampleCut::TopK, 0));
        assert_eq!(*steps.last().unwrap(), SampleStep::Finish);
        assert_eq!(SamplePlan::default().steps(1), [SampleStep::Top]);
    }

    #[test]
    fn the_scratch_holds_a_state_a_list_and_a_histogram_per_row() {
        assert_eq!(SamplerKernels::scratch_bytes(2), 2 * (64 + 64 * 9 * 8 + 2048 * 4 * 4));
        assert_eq!(SamplerKernels::exchange_values(3, 2), 3 * 8192);
    }
}
