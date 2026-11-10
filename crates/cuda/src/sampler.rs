//! The sampler: the pick of each row from its logits, on the device.
//!
//! Logits stay on the device. Per row, the kernels of `kernels/sampler.cu`
//! find the best few tokens (the greedy pick and the candidates a grammar may
//! fall back to), the exact cut-offs of top-k, top-p and min-p, and the pick,
//! and only an 80-byte record per row reaches the host ([`SamplePick`]).
//!
//! # What a pick is
//!
//! With `p` the softmax of a row's logits and `p_max` its largest value, every
//! cut is a threshold on `p / p_max`, which is `exp(-d)` for the distance `d`
//! of a logit below the best one:
//!
//! - **top-k** keeps the tokens at least as probable as the `k`-th;
//! - **top-p** keeps the most probable tokens that together hold the share
//!   `top_p` of what top-k left;
//! - **min-p** keeps the tokens with `p / p_max >= min_p`.
//!
//! The cuts apply in that order at temperature 1, as in llama.cpp's default
//! chain, and tokens tied at a cut are all kept. Among the tokens left, the
//! pick is the one with the largest `logit / temperature + g`, where `g` is
//! the Gumbel noise of (seed, position, token) from a counter-based generator
//! (Philox 4x32-10). That is a draw from the distribution of the kept tokens
//! at that temperature, and because a token's noise depends on nothing but
//! those three numbers, a row's pick is the same alone and in any batch, on
//! one device and on several, and verifying a draft is comparing it with the
//! pick for its position.
//!
//! # Exactness
//!
//! The cut-offs come from a radix descent over the bit patterns of `d` with
//! integer counts and fixed-point weights, the exponential and the logarithm
//! are polynomials of IEEE operations, so the order in which threads, blocks
//! and devices add does not matter and the host can repeat a pick bit for bit:
//! [`sample_reference`] is the same arithmetic, written from the definitions
//! above with a sort instead of the descent.
//!
//! # Several devices
//!
//! In a tensor split each device holds the logits of a slice of the
//! vocabulary. Every reduction of the sampler is a maximum or an integer sum,
//! so the devices run the same kernels on their slices and exchange small
//! partial results between the steps of a pick ([`SampleStep`]); the exchange
//! is the sum across devices the model has anyway (f32 values that add
//! exactly). All devices end with the same record.

use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DevicePtr;
use crate::module::{Args, Function, Module};
use crate::stream::Stream;

static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sampler.fatbin"));

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

// --- the arithmetic, on the host --------------------------------------------

fn sortable(v: f32) -> u32 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

fn unsortable(s: u32) -> f32 {
    f32::from_bits(if s & 0x8000_0000 != 0 { s & 0x7fff_ffff } else { !s })
}

fn pack(value: f32, token: u32) -> u64 {
    u64::from(sortable(value)) << 32 | u64::from(u32::MAX - token)
}

/// The key of a logit: the bits of its distance below the best logit.
fn key_of(l_max: f32, logit: f32) -> u32 {
    let d = l_max - logit;
    if d > 0.0 { d.to_bits() } else { 0 }
}

/// `exp(-d)` for the distance with bit pattern `key`, as a fixed-point number
/// with 40 fractional bits. The polynomial of `ares_sample_weight`.
pub fn weight_fixed(key: u32) -> u64 {
    let d = f32::from_bits(key);
    let t = d * std::f32::consts::LOG2_E;
    // Written this way so that a NaN gives 0 too.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(t < 41.0) {
        return 0;
    }
    let n = t as u32;
    let f = t - n as f32;
    let r = f * std::f32::consts::LN_2;
    let mut p = -1.0f32 / 362_880.0;
    for c in [
        1.0f32 / 40_320.0,
        -1.0 / 5_040.0,
        1.0 / 720.0,
        -1.0 / 120.0,
        1.0 / 24.0,
        -1.0 / 6.0,
        0.5,
        -1.0,
        1.0,
    ] {
        p = p * r + c;
    }
    ((p * 1_099_511_627_776.0) as u64) >> n
}

/// `ln(x)` for a positive normal `x`: the polynomial of `ares_sample_ln`.
fn ln_exact(x: f32) -> f32 {
    let b = x.to_bits();
    let mut e = (b >> 23) as i32 - 127;
    let mant = b & 0x007f_ffff;
    let m = if mant > 0x0035_04f3 {
        e += 1;
        f32::from_bits(mant | 0x3f00_0000)
    } else {
        f32::from_bits(mant | 0x3f80_0000)
    };
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let mut p = 1.0f32 / 9.0;
    for c in [1.0f32 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0] {
        p = p * z + c;
    }
    let lm = (2.0 * s) * p;
    e as f32 * std::f32::consts::LN_2 + lm
}

/// Philox 4x32 with 10 rounds: the output words for a counter and a key.
pub fn philox4x32(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = counter;
    let [mut k0, mut k1] = key;
    for _ in 0..10 {
        let p0 = 0xD251_1F53u64 * u64::from(c0);
        let p1 = 0xCD9E_8D57u64 * u64::from(c2);
        let next = [(p1 >> 32) as u32 ^ c1 ^ k0, p1 as u32, (p0 >> 32) as u32 ^ c3 ^ k1, p0 as u32];
        [c0, c1, c2, c3] = next;
        k0 = k0.wrapping_add(0x9E37_79B9);
        k1 = k1.wrapping_add(0xBB67_AE85);
    }
    [c0, c1, c2, c3]
}

/// The Gumbel noise of `token` at `position` for `seed`.
pub fn gumbel(seed: u64, position: u32, token: u32) -> f32 {
    let x = philox4x32([token, position, 0, 0], [seed as u32, (seed >> 32) as u32])[0];
    let u = ((x >> 9) as f32 + 0.5) * (1.0 / 8_388_608.0);
    let e = -ln_exact(u);
    -ln_exact(e)
}

/// The pick of one row on the host: the arithmetic of the kernels, written
/// from the definitions (a sort where the kernels descend by radix).
///
/// `logits[i]` is the logit of token `token_of(i)`.
pub fn sample_reference(
    logits: &[f32],
    token_of: impl Fn(usize) -> u32,
    row: &SampleRow,
) -> SamplePick {
    let mut status = 0;
    let canon: Vec<f32> = logits
        .iter()
        .map(|&l| {
            let l = l + 0.0;
            if l <= f32::MAX {
                l
            } else {
                status |= SAMPLE_STATUS_NONFINITE;
                f32::NEG_INFINITY
            }
        })
        .collect();
    let mut packed: Vec<u64> =
        canon.iter().enumerate().map(|(i, &l)| pack(l, token_of(i))).collect();
    packed.sort_unstable_by(|a, b| b.cmp(a));
    let mut pick = SamplePick {
        token: u32::MAX,
        status: 0,
        value: f32::NEG_INFINITY,
        cand_token: [u32::MAX; SAMPLE_CANDIDATES],
        cand_logit: [f32::NEG_INFINITY; SAMPLE_CANDIDATES],
    };
    for (c, &p) in packed.iter().take(SAMPLE_CANDIDATES).enumerate() {
        pick.cand_token[c] = u32::MAX - p as u32;
        pick.cand_logit[c] = unsortable((p >> 32) as u32);
    }
    let l_max = pick.cand_logit[0];
    if l_max == f32::NEG_INFINITY {
        status |= SAMPLE_STATUS_EMPTY;
    }
    pick.status = status;
    pick.value = l_max;
    if row.flags & F_SAMPLE == 0 {
        pick.token = pick.cand_token[0];
        return pick;
    }

    // Keys ascending: most probable first.
    let mut keyed: Vec<(u32, u64)> = canon
        .iter()
        .map(|&l| {
            let key = key_of(l_max, l);
            (key, weight_fixed(key))
        })
        .collect();
    keyed.sort_unstable_by_key(|(key, _)| *key);
    let mut cut = row.min_p_key.min(KEY_MAX);
    let total: u64 = keyed.iter().map(|(_, w)| w).sum();
    let mut z = total;
    if row.flags & F_TOP_K != 0 && (row.top_k as usize) <= keyed.len() {
        let k_key = keyed[row.top_k as usize - 1].0;
        cut = cut.min(k_key);
        z = keyed.iter().take_while(|(key, _)| *key <= k_key).map(|(_, w)| w).sum();
    }
    if row.flags & F_TOP_P != 0 {
        let target = ((u128::from(z) * u128::from(row.top_p_q32)) >> 32) as u64;
        let target = target.max(1);
        let mut run = 0u64;
        let mut i = 0;
        while i < keyed.len() {
            let key = keyed[i].0;
            while i < keyed.len() && keyed[i].0 == key {
                run += keyed[i].1;
                i += 1;
            }
            if run >= target {
                cut = cut.min(key);
                break;
            }
        }
    }
    let seed = u64::from(row.seed_hi) << 32 | u64::from(row.seed_lo);
    let mut best = 0u64;
    for (i, &l) in canon.iter().enumerate() {
        if key_of(l_max, l) > cut {
            continue;
        }
        let token = token_of(i);
        let score = l * row.inv_temperature + gumbel(seed, row.position, token);
        best = best.max(pack(score, token));
    }
    if best == 0 {
        pick.status |= SAMPLE_STATUS_EMPTY;
        return pick;
    }
    pick.token = u32::MAX - best as u32;
    pick.value = unsortable((best >> 32) as u32);
    pick
}

// --- the kernels ----------------------------------------------------------------

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
pub struct SamplerKernels {
    device: Device,
    top: Function,
    top_merge: Function,
    top_join: Function,
    hist: Function,
    export: Function,
    decide: Function,
    pick: Function,
    pick_export: Function,
    finish: Function,
    draft_put: Function,
    draft_rows: Function,
    draft_shift: Function,
    draft_keep: Function,
    draft_commit: Function,
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
            hist: module.function("ares_sample_hist")?,
            export: module.function("ares_sample_export")?,
            decide: module.function("ares_sample_decide")?,
            pick: module.function("ares_sample_pick")?,
            pick_export: module.function("ares_sample_pick_export")?,
            finish: module.function("ares_sample_finish")?,
            draft_put: module.function("ares_draft_put")?,
            draft_rows: module.function("ares_draft_rows")?,
            draft_shift: module.function("ares_draft_shift")?,
            draft_keep: module.function("ares_draft_keep")?,
            draft_commit: module.function("ares_draft_commit")?,
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
        rows * (BINS * BIN_WORDS).max(devices * TOP_VALUES)
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
        let chunks = batch.n_local.div_ceil(CHUNK);
        let state = batch.scratch;
        // The histograms rely on being left clear by the batch before, so
        // the arrays lie where the capacity puts them, whatever this batch's
        // size.
        let capacity = u64::from(batch.capacity);
        let lists = state.offset(capacity * STATE_BYTES);
        let hist = lists.offset(capacity * Self::lists_bytes());
        let several = batch.devices > 1;
        let block = [THREADS, 1, 1];
        let small = [32, 1, 1];
        // SAFETY: the arguments of every launch below match its kernel's
        // parameter list in kernels/sampler_*.cuh, and the caller guarantees
        // the memory behind the pointers.
        unsafe {
            match step {
                SampleStep::Top => {
                    self.top.launch(
                        stream,
                        [chunks, rows, 1],
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
                            .u32(batch.devices)
                            .u32(batch.rank)
                            .ptr(batch.exchange)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(batch.out),
                    )?;
                    Ok(if several { rows * batch.devices * TOP_VALUES } else { 0 })
                }
                SampleStep::Join => {
                    self.top_join.launch(
                        stream,
                        [rows, 1, 1],
                        small,
                        0,
                        Args::new()
                            .ptr(batch.exchange)
                            .u32(batch.devices)
                            .ptr(batch.params)
                            .ptr(state)
                            .ptr(batch.out),
                    )?;
                    Ok(0)
                }
                SampleStep::Level(cut, level) => {
                    self.hist.launch(
                        stream,
                        [chunks, rows, 1],
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
                            .u32(cut as u32)
                            .u32(level),
                    )?;
                    let bins = if level == 1 { BINS } else { BINS / 2 };
                    self.export.launch(
                        stream,
                        [bins / THREADS, rows, 1],
                        block,
                        0,
                        Args::new().ptr(hist).ptr(batch.exchange).ptr(batch.params).u32(cut as u32),
                    )?;
                    Ok(if several { rows * BINS * BIN_WORDS } else { 0 })
                }
                SampleStep::Decide(cut, level) => {
                    self.decide.launch(
                        stream,
                        [rows, 1, 1],
                        block,
                        0,
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
                    self.pick.launch(
                        stream,
                        [chunks, rows, 1],
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
                            .u32(batch.devices)
                            .u32(batch.rank),
                    )?;
                    Ok(rows * batch.devices * PICK_VALUES)
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
                            .u32(batch.devices)
                            .ptr(batch.params)
                            .ptr(batch.out),
                    )?;
                    Ok(0)
                }
            }
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
    /// The tokens of `n` picks at `picks` ([`SamplePick`] records) go where
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

    /// For every row of the step at `step` (a record in the layout of the
    /// layer kernels, with `segments` segments of at most `max_seg_rows`
    /// rows), the final hidden state of the position before it: `out[r]` is
    /// `hidden[r - 1]` inside a segment and the sequence's carry (zeros at
    /// position 0) for a segment's first row.
    ///
    /// # Safety
    ///
    /// `step` must address the uploaded record, `hidden` and `out` a row of
    /// `d` f32 per row of the step, `carry` `carry_rows * d` f32 per slot the
    /// record names, on the stream's device.
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
        let args = Args::new().ptr(step).ptr(hidden).ptr(carry).u32(carry_rows).ptr(out).u32(d);
        let grid = [max_seg_rows, segments, 1];
        // SAFETY: the arguments match ares_draft_shift; the memory is the caller's guarantee.
        unsafe { self.draft_shift.launch(stream, grid, [THREADS, 1, 1], 0, args) }
    }

    /// Keeps the final hidden state of the step's rows in the carries of
    /// their sequences.
    ///
    /// # Safety
    ///
    /// As [`SamplerKernels::shift_hidden`]; a segment that is not direct has
    /// at most `carry_rows` rows.
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
        let grid = [max_seg_rows, segments, 1];
        // SAFETY: the arguments match ares_draft_keep; the memory is the caller's guarantee.
        unsafe { self.draft_keep.launch(stream, grid, [THREADS, 1, 1], 0, args) }
    }

    /// The commit of the windows of the last step, whose record with the
    /// accepted rows is at `commit` (`segments` segments): row 0 of each
    /// sequence's carry becomes the hidden state of its last accepted row.
    ///
    /// # Safety
    ///
    /// `commit` must address the uploaded commit record and `carry` the
    /// carries of every slot it names, on the stream's device.
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
        // SAFETY: the arguments match ares_draft_commit; the memory is the caller's guarantee.
        unsafe { self.draft_commit.launch(stream, [segments, 1, 1], [THREADS, 1, 1], 0, args) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_gives_the_known_answers() {
        // The known-answer tests of Random123 (kat_vectors, philox4x32 10).
        assert_eq!(
            philox4x32([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
    }

    #[test]
    fn the_fixed_point_weight_is_the_exponential() {
        assert_eq!(weight_fixed(0), 1 << 40);
        let mut worst = 0f64;
        let mut d = 1e-6f32;
        while d < 27.0 {
            let exact = (-f64::from(d)).exp() * (1u64 << 40) as f64;
            let got = weight_fixed(d.to_bits()) as f64;
            // The polynomial's error, beyond the unit the fixed point cuts.
            let error = ((got - exact).abs() - 1.0).max(0.0) / exact;
            worst = worst.max(error);
            d *= 1.01;
        }
        assert!(worst < 1e-6, "relative error {worst}");
        assert_eq!(weight_fixed(f32::INFINITY.to_bits()), 0);
        assert_eq!(weight_fixed(40.0f32.to_bits()), 0);
        // Never above the weight of the best token.
        assert!((0..2000).all(|i| weight_fixed((i as f32 * 1e-9).to_bits()) <= 1 << 40));
    }

    #[test]
    fn the_logarithm_is_the_logarithm() {
        let mut x = 5e-8f32;
        while x < 20.0 {
            let exact = f64::from(x).ln();
            let got = f64::from(ln_exact(x));
            assert!((got - exact).abs() < 2e-6 * exact.abs().max(1.0), "ln({x}) = {got}, {exact}");
            x *= 1.003;
        }
    }

    fn params(temperature: f32, top_k: u32, top_p: f32, min_p: f32) -> SampleParams {
        SampleParams { temperature, top_k, top_p, min_p, seed: 42 }
    }

    /// How often each token of `logits` is picked over `n` positions.
    fn frequencies(logits: &[f32], params: &SampleParams, n: u32) -> Vec<f64> {
        let mut counts = vec![0u32; logits.len()];
        for position in 0..n {
            let pick = sample_reference(logits, |i| i as u32, &params.row(position));
            assert_eq!(pick.status, 0);
            counts[pick.token as usize] += 1;
        }
        counts.iter().map(|c| f64::from(*c) / f64::from(n)).collect()
    }

    #[test]
    fn greedy_takes_the_best_logit_and_the_lower_token_on_a_tie() {
        let logits = [0.5, 3.0, -1.0, 3.0, 2.0, f32::NEG_INFINITY];
        let pick = sample_reference(&logits, |i| 10 + i as u32, &SampleParams::GREEDY.row(7));
        assert_eq!((pick.token, pick.value, pick.status), (11, 3.0, 0));
        assert_eq!(pick.cand_token[..6], [11, 13, 14, 10, 12, 15]);
        assert_eq!(pick.cand_token[6], u32::MAX);
        // A token map decides the tie, not the column.
        let pick = sample_reference(&logits, |i| 20 - i as u32, &SampleParams::GREEDY.row(7));
        assert_eq!(pick.token, 17);
    }

    #[test]
    fn a_logit_that_is_not_a_number_is_reported() {
        let pick = sample_reference(&[1.0, f32::NAN], |i| i as u32, &SampleParams::GREEDY.row(0));
        assert_eq!(pick.status, SAMPLE_STATUS_NONFINITE);
        let none = [f32::NEG_INFINITY; 3];
        let pick = sample_reference(&none, |i| i as u32, &params(1.0, 0, 1.0, 0.0).row(0));
        assert_eq!(pick.status, SAMPLE_STATUS_EMPTY);
    }

    #[test]
    fn sampling_draws_from_the_softmax_at_the_temperature() {
        let logits = [1.0f32, 0.0, -1.0, 2.0, -3.0];
        for temperature in [1.0f32, 0.5, 2.0] {
            let z: f64 =
                logits.iter().map(|l| (f64::from(*l) / f64::from(temperature)).exp()).sum();
            let got = frequencies(&logits, &params(temperature, 0, 1.0, 0.0), 40_000);
            for (l, f) in logits.iter().zip(&got) {
                let p = (f64::from(*l) / f64::from(temperature)).exp() / z;
                assert!((f - p).abs() < 0.012, "T {temperature}: p {p:.4}, drawn {f:.4}");
            }
        }
    }

    #[test]
    fn the_cuts_keep_what_they_say() {
        // Probabilities 0.4, 0.3, 0.2, 0.1 (logits are their logarithms).
        let logits: Vec<f32> = [0.4f32, 0.3, 0.2, 0.1].iter().map(|p| p.ln()).collect();
        let kept = |p: &SampleParams| -> Vec<bool> {
            frequencies(&logits, p, 4000).iter().map(|f| *f > 0.0).collect()
        };
        assert_eq!(kept(&params(1.0, 2, 1.0, 0.0)), [true, true, false, false]);
        // The smallest set that holds 0.75 is the first three.
        assert_eq!(kept(&params(1.0, 0, 0.75, 0.0)), [true, true, true, false]);
        assert_eq!(kept(&params(1.0, 0, 0.65, 0.0)), [true, true, false, false]);
        // top-p counts within what top-k left: 0.75 of (0.4 + 0.3 + 0.2).
        assert_eq!(kept(&params(1.0, 3, 0.75, 0.0)), [true, true, false, false]);
        // min-p 0.6 of the best (0.4) is 0.24.
        assert_eq!(kept(&params(1.0, 0, 1.0, 0.6)), [true, true, false, false]);
        // The cuts are made at temperature 1, whatever the temperature.
        assert_eq!(kept(&params(5.0, 2, 1.0, 0.0)), [true, true, false, false]);
        // Ties at the cut are all kept.
        let tied = [0.0f32, 0.0, 0.0, -5.0];
        let f = frequencies(&tied, &params(1.0, 1, 1.0, 0.0), 4000);
        assert!(f[..3].iter().all(|f| *f > 0.25) && f[3] == 0.0, "{f:?}");
        // After the cut the draw is from the kept tokens' own distribution.
        let f = frequencies(&logits, &params(1.0, 2, 1.0, 0.0), 40_000);
        assert!((f[0] - 4.0 / 7.0).abs() < 0.012, "{f:?}");
    }

    #[test]
    fn a_pick_depends_on_seed_position_and_token_only() {
        let logits: Vec<f32> = (0..500).map(|i| ((i * 37 % 101) as f32) * 0.05).collect();
        let p = params(1.0, 40, 0.95, 0.02);
        let a = sample_reference(&logits, |i| i as u32, &p.row(11));
        // The same row with its columns in another order.
        let reversed: Vec<f32> = logits.iter().rev().copied().collect();
        let b = sample_reference(&reversed, |i| (499 - i) as u32, &p.row(11));
        assert_eq!(a, b);
        let other = sample_reference(&logits, |i| i as u32, &p.row(12));
        let seeded =
            sample_reference(&logits, |i| i as u32, &SampleParams { seed: 43, ..p }.row(11));
        assert!(a.token != other.token || a.token != seeded.token, "the noise does not vary");
    }
}
