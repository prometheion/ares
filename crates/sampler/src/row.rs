//! What a row of the sampler is told, and what it reports.
//!
//! [`SampleParams`] are the settings of a sequence. For each row whose logits
//! are wanted, [`SampleParams::row`] makes a [`SampleRow`]: the 128 bytes the
//! kernels read, with the position of the row, the program of stages that
//! applies, the place of the row's corrections and mask in the step record,
//! and the state a stateful sampler (mirostat) is in.

use crate::arith::fraction_q32;

/// The candidates recorded per row: its best tokens by logit.
pub const SAMPLE_CANDIDATES: usize = 8;
/// The most alternatives a row reports next to the token it picked.
pub const LOGPROBS_MAX: usize = 20;
/// The most corrections ([`Entry`]) one row takes.
pub const ENTRIES_PER_ROW: usize = 4096;
/// The most stages of a row's program.
pub const MAX_STAGES: usize = 16;
/// Status bit of a [`SamplePick`]: a logit of the row was NaN or plus
/// infinity; the pick is not valid.
pub const SAMPLE_STATUS_NONFINITE: u32 = 1;
/// Status bit of a [`SamplePick`]: no token was left to pick.
pub const SAMPLE_STATUS_EMPTY: u32 = 2;
/// The smallest temperature that scales the logits. A smaller one picks the
/// most probable token, because its reciprocal would not be a number.
pub const TEMPERATURE_MIN: f32 = 1e-6;
/// The temperature at which mirostat 1 estimates the zipf exponent from this
/// many of the most probable tokens (llama.cpp's `m`).
pub const MIROSTAT_M: usize = 100;

/// Row flags (word 0 of a [`SampleRow`]).
pub(crate) const F_SAMPLE: u32 = 1;
pub(crate) const F_MASK: u32 = 2;
pub(crate) const F_LOGPROBS_RAW: u32 = 4;
pub(crate) const F_LOGPROBS_POST: u32 = 8;
pub(crate) const F_MIROSTAT: u32 = 16;

/// One stage of the chain of samplers. The numbers are what a program holds
/// in four bits each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Stage {
    /// Keeps the tokens within `n` standard deviations of the logits below
    /// the best.
    TopNSigma = 1,
    /// Keeps the `k` most probable tokens.
    TopK = 2,
    /// Locally typical sampling.
    Typical = 3,
    /// Keeps the most probable tokens that hold the share `p`.
    TopP = 4,
    /// Keeps the tokens at least `p` times as probable as the best.
    MinP = 5,
    /// Removes the most probable tokens, but the least probable of them, with
    /// a probability.
    Xtc = 6,
    /// Scales the logits; a dynamic temperature follows the entropy.
    Temperature = 7,
    /// Mirostat 1: the top-k of a `k` that follows the target surprise.
    Mirostat1 = 8,
    /// Mirostat 2: keeps the tokens whose surprise is at most `mu`.
    Mirostat2 = 9,
}

impl Stage {
    /// The stage with this code, or `None`.
    pub fn from_code(code: u32) -> Option<Stage> {
        Some(match code {
            1 => Stage::TopNSigma,
            2 => Stage::TopK,
            3 => Stage::Typical,
            4 => Stage::TopP,
            5 => Stage::MinP,
            6 => Stage::Xtc,
            7 => Stage::Temperature,
            8 => Stage::Mirostat1,
            9 => Stage::Mirostat2,
            _ => return None,
        })
    }
}

/// The order in which llama.cpp applies the samplers that are not penalties
/// (which come first in its chain and are worked out by the host).
pub const DEFAULT_ORDER: [Stage; 7] = [
    Stage::TopNSigma,
    Stage::TopK,
    Stage::Typical,
    Stage::TopP,
    Stage::MinP,
    Stage::Xtc,
    Stage::Temperature,
];

/// The order of the stages a request asks for: at most [`MAX_STAGES`] entries
/// of the stages that are not mirostat. A stage that is not listed is not
/// applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Order {
    stages: [Stage; MAX_STAGES],
    len: u8,
}

impl Order {
    /// llama.cpp's order, as a constant.
    const STANDARD: Order = {
        let mut stages = [Stage::Temperature; MAX_STAGES];
        let mut i = 0;
        while i < DEFAULT_ORDER.len() {
            stages[i] = DEFAULT_ORDER[i];
            i += 1;
        }
        Order { stages, len: DEFAULT_ORDER.len() as u8 }
    };

    /// `stages` in order, or `None` for more than [`MAX_STAGES`] or a
    /// mirostat stage (which is not part of an order).
    pub fn new(stages: &[Stage]) -> Option<Order> {
        if stages.len() > MAX_STAGES
            || stages.iter().any(|s| matches!(s, Stage::Mirostat1 | Stage::Mirostat2))
        {
            return None;
        }
        let mut order = Order { stages: [Stage::Temperature; MAX_STAGES], len: stages.len() as u8 };
        order.stages[..stages.len()].copy_from_slice(stages);
        Some(order)
    }

    /// llama.cpp's order.
    pub fn standard() -> Order {
        Order::STANDARD
    }

    pub fn stages(&self) -> &[Stage] {
        &self.stages[..self.len as usize]
    }
}

/// Which mirostat a sequence runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mirostat {
    V1,
    V2,
}

/// What a row reports about the probabilities of its tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Logprobs {
    /// The alternatives reported next to the picked token (at most
    /// [`LOGPROBS_MAX`]); 0 for none.
    pub top: u8,
    /// Whether the chosen token's probability is asked for although `top` is
    /// 0, as OpenAI's `logprobs: true` with `top_logprobs: 0` does.
    pub chosen: bool,
    /// The distribution the probabilities belong to: the model's own
    /// (`false`: the softmax of the logits before any processing) or the one
    /// the pick is drawn from (`true`: after the chain, over the tokens it
    /// left, at its temperature).
    pub after_chain: bool,
}

impl Logprobs {
    pub fn is_asked(&self) -> bool {
        self.top > 0 || self.chosen
    }
}

/// How a sequence's tokens are picked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
    /// 0 picks the most probable token (unless a dynamic temperature or a
    /// mirostat is on).
    pub temperature: f32,
    /// 0 is off.
    pub top_k: u32,
    /// 1 (or more) is off.
    pub top_p: f32,
    /// 0 is off.
    pub min_p: f32,
    /// Locally typical sampling; 1 (or more) is off.
    pub typical_p: f32,
    /// 0 (or less) is off.
    pub top_n_sigma: f32,
    /// The chance that XTC removes tokens; 0 is off.
    pub xtc_probability: f32,
    /// The probability a token needs to be removed by XTC; above 0.5 is off.
    pub xtc_threshold: f32,
    /// The dynamic temperature's range around `temperature`; 0 is off.
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub mirostat: Option<Mirostat>,
    /// Mirostat's `mu`, which the host updates from every pick.
    pub mirostat_mu: f32,
    /// The number of tokens of the model, which mirostat 1 uses.
    pub vocab: u32,
    pub order: Order,
    pub seed: u64,
    pub logprobs: Logprobs,
}

impl SampleParams {
    /// The most probable token.
    pub const GREEDY: SampleParams = SampleParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        typical_p: 1.0,
        top_n_sigma: 0.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        dynatemp_range: 0.0,
        dynatemp_exponent: 1.0,
        mirostat: None,
        mirostat_mu: 0.0,
        vocab: 0,
        order: Order::STANDARD,
        seed: 0,
        logprobs: Logprobs { top: 0, chosen: false, after_chain: false },
    };

    /// Whether the pick is a draw and not the most probable token.
    pub fn sampled(&self) -> bool {
        self.temperature > 0.0 || self.dynatemp_range > 0.0 || self.mirostat.is_some()
    }

    /// Whether the picks of this sequence depend on every token before them
    /// by a state that only a pick can advance (mirostat's `mu`): a window
    /// that verifies drafts cannot know it for the rows after the first.
    pub fn stateful(&self) -> bool {
        self.mirostat.is_some()
    }

    /// The stages that apply, in order.
    pub fn program(&self) -> Vec<Stage> {
        let temperature = self.temperature != 1.0 || self.dynatemp_range > 0.0;
        if let Some(mirostat) = self.mirostat {
            let mut stages = Vec::with_capacity(2);
            if temperature {
                stages.push(Stage::Temperature);
            }
            stages.push(match mirostat {
                Mirostat::V1 => Stage::Mirostat1,
                Mirostat::V2 => Stage::Mirostat2,
            });
            return stages;
        }
        self.order
            .stages()
            .iter()
            .copied()
            .filter(|stage| match stage {
                Stage::TopNSigma => self.top_n_sigma > 0.0,
                Stage::TopK => self.top_k > 0,
                Stage::Typical => self.typical_p > 0.0 && self.typical_p < 1.0,
                Stage::TopP => self.top_p > 0.0 && self.top_p < 1.0,
                Stage::MinP => self.min_p > 0.0,
                Stage::Xtc => self.xtc_probability > 0.0 && self.xtc_threshold <= 0.5,
                Stage::Temperature => temperature,
                Stage::Mirostat1 | Stage::Mirostat2 => false,
            })
            .collect()
    }

    /// The device's form of these settings for a row at `position`, with no
    /// corrections and no mask.
    pub fn row(&self, position: u32) -> SampleRow {
        let mut row = SampleRow::NONE;
        row.words[W_POSITION] = position;
        row.words[W_SEED_LO] = self.seed as u32;
        row.words[W_SEED_HI] = (self.seed >> 32) as u32;
        row.words[W_MASK] = u32::MAX;
        let mut flags = 0;
        if self.logprobs.is_asked() {
            flags |= if self.logprobs.after_chain { F_LOGPROBS_POST } else { F_LOGPROBS_RAW };
            row.words[W_LOGPROBS] = u32::from(self.logprobs.top.min(LOGPROBS_MAX as u8));
        }
        if self.sampled() {
            flags |= F_SAMPLE;
            let (mut lo, mut hi) = (0u32, 0u32);
            for (i, stage) in self.program().iter().take(MAX_STAGES).enumerate() {
                let word = if i < 8 { &mut lo } else { &mut hi };
                *word |= (*stage as u32) << (4 * (i % 8));
            }
            row.words[W_PROGRAM_LO] = lo;
            row.words[W_PROGRAM_HI] = hi;
            row.words[W_TOP_K] = self.top_k;
            row.words[W_TOP_P] = fraction_q32(self.top_p);
            row.words[W_MIN_P] = (-self.min_p.ln()).max(0.0).to_bits();
            row.words[W_TYPICAL] = fraction_q32(self.typical_p);
            row.words[W_N_SIGMA] = self.top_n_sigma.to_bits();
            row.words[W_XTC_PROB] = self.xtc_probability.to_bits();
            row.words[W_XTC_THR] = fraction_q32(self.xtc_threshold);
            row.words[W_TEMP] = self.temperature.to_bits();
            row.words[W_INV_TEMP] = if self.temperature >= TEMPERATURE_MIN {
                (1.0 / self.temperature).to_bits()
            } else {
                0
            };
            row.words[W_DYN_DELTA] = self.dynatemp_range.to_bits();
            row.words[W_DYN_EXP] = self.dynatemp_exponent.to_bits();
            row.words[W_MU] = self.mirostat_mu.to_bits();
            row.words[W_VOCAB] = self.vocab;
            if self.mirostat.is_some() {
                flags |= F_MIROSTAT;
            }
        }
        row.words[W_FLAGS] = flags;
        row
    }
}

// Word indices of a row.
pub(crate) const W_FLAGS: usize = 0;
pub(crate) const W_SEED_LO: usize = 1;
pub(crate) const W_SEED_HI: usize = 2;
pub(crate) const W_POSITION: usize = 3;
pub(crate) const W_PROGRAM_LO: usize = 4;
pub(crate) const W_PROGRAM_HI: usize = 5;
pub(crate) const W_TOP_K: usize = 6;
pub(crate) const W_TOP_P: usize = 7;
pub(crate) const W_MIN_P: usize = 8;
pub(crate) const W_TYPICAL: usize = 9;
pub(crate) const W_N_SIGMA: usize = 10;
pub(crate) const W_XTC_PROB: usize = 11;
pub(crate) const W_XTC_THR: usize = 12;
pub(crate) const W_TEMP: usize = 13;
pub(crate) const W_INV_TEMP: usize = 14;
pub(crate) const W_DYN_DELTA: usize = 15;
pub(crate) const W_DYN_EXP: usize = 16;
pub(crate) const W_MU: usize = 17;
pub(crate) const W_VOCAB: usize = 18;
pub(crate) const W_ENTRIES_FIRST: usize = 19;
pub(crate) const W_ENTRIES_COUNT: usize = 20;
pub(crate) const W_MASK: usize = 21;
pub(crate) const W_LOGPROBS: usize = 22;
const ROW_WORDS: usize = 32;

/// One row's settings as the kernels read them (`AresSampleParams`): 32
/// little-endian words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRow {
    words: [u32; ROW_WORDS],
}

impl SampleRow {
    /// The bytes of a row on the device.
    pub const BYTES: usize = ROW_WORDS * 4;

    /// A row that takes the most probable token and nothing else.
    pub const NONE: SampleRow = {
        let mut words = [0u32; ROW_WORDS];
        words[W_MASK] = u32::MAX;
        SampleRow { words }
    };

    /// Whether the row is sampled (and not the most probable token).
    pub fn sampled(&self) -> bool {
        self.words[W_FLAGS] & F_SAMPLE != 0
    }

    pub fn position(&self) -> u32 {
        self.words[W_POSITION]
    }

    /// The stages that apply to the row, in order.
    pub fn program(&self) -> Vec<Stage> {
        (0..MAX_STAGES)
            .map_while(|i| {
                let word = self.words[if i < 8 { W_PROGRAM_LO } else { W_PROGRAM_HI }];
                Stage::from_code((word >> (4 * (i % 8))) & 0xf)
            })
            .collect()
    }

    /// Whether the row's temperature follows the entropy of the tokens left.
    pub fn dynamic_temperature(&self) -> bool {
        self.sampled() && self.float(W_DYN_DELTA) > 0.0
    }

    /// Whether the row reports probabilities, and which kind.
    pub fn logprobs(&self) -> Option<(bool, u32)> {
        let flags = self.words[W_FLAGS];
        (flags & (F_LOGPROBS_RAW | F_LOGPROBS_POST) != 0)
            .then(|| (flags & F_LOGPROBS_POST != 0, self.words[W_LOGPROBS]))
    }

    /// Whether the row's pick feeds a state the host keeps (mirostat's `mu`):
    /// the device reports the observed surprise in [`SamplePick::aux`].
    pub fn mirostat(&self) -> bool {
        self.words[W_FLAGS] & F_MIROSTAT != 0
    }

    /// Whether the kernels need the second half of the pick for this row
    /// (a surprise or probabilities after the chain).
    pub fn tail(&self) -> bool {
        self.words[W_FLAGS] & (F_LOGPROBS_POST | F_MIROSTAT) != 0
    }

    /// The row with its corrections at `first..first + count` of the step's
    /// entries (see [`Entry`]).
    pub fn with_entries(mut self, first: u32, count: u32) -> SampleRow {
        self.words[W_ENTRIES_FIRST] = first;
        self.words[W_ENTRIES_COUNT] = count;
        self
    }

    pub fn entries(&self) -> (u32, u32) {
        (self.words[W_ENTRIES_FIRST], self.words[W_ENTRIES_COUNT])
    }

    /// The row with the bit mask number `slot` of the step: only the tokens
    /// whose bit is set can be picked.
    pub fn with_mask(mut self, slot: u32) -> SampleRow {
        self.words[W_MASK] = slot;
        self.words[W_FLAGS] |= F_MASK;
        self
    }

    pub fn mask(&self) -> Option<u32> {
        (self.words[W_FLAGS] & F_MASK != 0).then_some(self.words[W_MASK])
    }

    /// The row as a model's MTP block picks with it: the same settings,
    /// noise and position, with what only the step's record can supply (the
    /// corrections and the mask) and what only the verifying row reports
    /// (probabilities, the surprise) taken away. A draft is a guess that the
    /// verifying row confirms, so it may be made from the logits as they are.
    pub fn for_draft(&self) -> SampleRow {
        let mut row = *self;
        row.words[W_ENTRIES_FIRST] = 0;
        row.words[W_ENTRIES_COUNT] = 0;
        row.words[W_MASK] = u32::MAX;
        row.words[W_LOGPROBS] = 0;
        row.words[W_FLAGS] &= F_SAMPLE;
        row
    }

    /// Reads a row as [`SampleRow::to_bytes`] wrote it.
    pub fn from_bytes(bytes: &[u8; Self::BYTES]) -> Self {
        let mut words = [0u32; ROW_WORDS];
        for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*chunk);
        }
        SampleRow { words }
    }

    /// The row as the device reads it.
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut bytes = [0u8; Self::BYTES];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(self.words) {
            *chunk = word.to_le_bytes();
        }
        bytes
    }

    // The fields the reference and the tests read.
    pub(crate) fn word(&self, at: usize) -> u32 {
        self.words[at]
    }

    pub(crate) fn float(&self, at: usize) -> f32 {
        f32::from_bits(self.words[at])
    }

    pub(crate) fn seed(&self) -> u64 {
        u64::from(self.words[W_SEED_HI]) << 32 | u64::from(self.words[W_SEED_LO])
    }
}

/// A correction of one token's logit, made before anything else looks at the
/// row: `l = l + bias`, then `l = l <= 0 ? l * rep : l / rep` (llama.cpp's
/// repetition penalty, which pushes a logit toward less probable on either
/// side of zero), then `l = l - sub`. One entry per token and row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub token: u32,
    pub bias: f32,
    pub rep: f32,
    pub sub: f32,
}

impl Entry {
    /// The bytes of an entry on the device.
    pub const BYTES: usize = 16;

    /// An entry that changes nothing.
    pub const fn new(token: u32) -> Entry {
        Entry { token, bias: 0.0, rep: 1.0, sub: 0.0 }
    }

    /// The corrected logit.
    pub fn apply(&self, logit: f32) -> f32 {
        let l = logit + self.bias;
        let l = if l <= 0.0 { l * self.rep } else { l / self.rep };
        l - self.sub
    }

    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut bytes = [0u8; Self::BYTES];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip([
            self.token,
            self.bias.to_bits(),
            self.rep.to_bits(),
            self.sub.to_bits(),
        ]) {
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
    /// The logit of a greedy pick; the score (`logit * inv_t + noise`) of a
    /// sampled one.
    pub value: f32,
    /// For a mirostat row, the surprise of the picked token in bits: minus the
    /// binary logarithm of its probability among the tokens that were kept.
    pub aux: f32,
    /// The row's best tokens by logit (after the corrections), best first, the
    /// lower token first among equal logits. `u32::MAX` where the row has
    /// fewer tokens.
    pub cand_token: [u32; SAMPLE_CANDIDATES],
    pub cand_logit: [f32; SAMPLE_CANDIDATES],
}

impl SamplePick {
    /// The bytes of a record on the device.
    pub const BYTES: usize = 16 + 8 * SAMPLE_CANDIDATES;

    /// The probability of the picked token among the row's candidates, by
    /// their logits as they are: how sure the row was of its pick. The tokens
    /// past the candidates are left out, so this is at least the token's
    /// probability over the whole row; a pick that is not among the candidates
    /// has 0.
    pub fn confidence(&self) -> f32 {
        let best = self.cand_logit[0];
        let (mut all, mut own) = (0.0f32, 0.0f32);
        for (token, logit) in self.cand_token.iter().zip(&self.cand_logit) {
            if *token == u32::MAX {
                break;
            }
            let weight = (logit - best).exp();
            all += weight;
            if *token == self.token {
                own = weight;
            }
        }
        let share = own / all;
        if share.is_finite() { share } else { 0.0 }
    }

    /// Reads a record as the device wrote it.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        let mut pick = SamplePick {
            token: word(0),
            status: word(1),
            value: f32::from_bits(word(2)),
            aux: f32::from_bits(word(3)),
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

/// The probabilities a row reports (`AresLogprobOut`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogprobOut {
    /// How many alternatives are valid.
    pub n: u32,
    /// The natural logarithm of the probability of the picked token.
    pub chosen: f32,
    /// The alternatives, most probable first, the lower token first among
    /// equal ones, with their log-probabilities.
    pub token: [u32; LOGPROBS_MAX],
    pub logprob: [f32; LOGPROBS_MAX],
}

impl LogprobOut {
    /// The bytes of a record on the device.
    pub const BYTES: usize = 8 + 8 * LOGPROBS_MAX;

    pub const NONE: LogprobOut = LogprobOut {
        n: 0,
        chosen: f32::NEG_INFINITY,
        token: [u32::MAX; LOGPROBS_MAX],
        logprob: [f32::NEG_INFINITY; LOGPROBS_MAX],
    };

    /// Reads a record as the device wrote it.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        let mut out = LogprobOut { n: word(0), chosen: f32::from_bits(word(1)), ..Self::NONE };
        for c in 0..LOGPROBS_MAX {
            out.token[c] = word(2 + c);
            out.logprob[c] = f32::from_bits(word(2 + LOGPROBS_MAX + c));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_survives_its_bytes() {
        let params = SampleParams {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.05,
            seed: 0xdead_beef_0123_4567,
            ..SampleParams::GREEDY
        };
        let row = params.row(77).with_entries(5, 9).with_mask(3);
        assert_eq!(SampleRow::from_bytes(&row.to_bytes()), row);
        assert_eq!((row.position(), row.entries(), row.mask()), (77, (5, 9), Some(3)));
        assert_eq!(row.seed(), 0xdead_beef_0123_4567);
        assert!(row.sampled() && !SampleParams::GREEDY.row(0).sampled());
        assert_eq!(SampleRow::NONE, SampleParams::GREEDY.row(0));
    }

    #[test]
    fn the_program_follows_the_order_and_leaves_out_what_is_off() {
        let params = SampleParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            ..SampleParams::GREEDY
        };
        assert_eq!(
            params.program(),
            [Stage::TopK, Stage::TopP, Stage::MinP, Stage::Temperature],
            "llama.cpp's order"
        );
        assert_eq!(params.row(0).program(), params.program());
        let order = Order::new(&[Stage::Temperature, Stage::MinP, Stage::TopK]).unwrap();
        let params = SampleParams { order, ..params };
        assert_eq!(params.program(), [Stage::Temperature, Stage::MinP, Stage::TopK]);
        // A temperature of 1 changes nothing.
        let one = SampleParams { temperature: 1.0, top_k: 5, ..SampleParams::GREEDY };
        assert_eq!(one.program(), [Stage::TopK]);
        let mirostat = SampleParams {
            temperature: 0.5,
            mirostat: Some(Mirostat::V2),
            top_k: 10,
            ..SampleParams::GREEDY
        };
        assert_eq!(mirostat.program(), [Stage::Temperature, Stage::Mirostat2]);
        assert!(mirostat.row(0).mirostat() && mirostat.stateful());
    }

    #[test]
    fn an_order_has_limits() {
        assert!(Order::new(&[Stage::TopK; MAX_STAGES]).is_some());
        assert!(Order::new(&[Stage::TopK; MAX_STAGES + 1]).is_none());
        assert!(Order::new(&[Stage::Mirostat2]).is_none());
        assert_eq!(Order::standard().stages(), DEFAULT_ORDER);
    }

    #[test]
    fn an_entry_corrects_in_the_order_of_llama_cpp() {
        let e = Entry { token: 1, bias: 1.0, rep: 2.0, sub: 0.5 };
        // (3 + 1) / 2 - 0.5
        assert_eq!(e.apply(3.0), 1.5);
        // (-3 + 1) * 2 - 0.5: a negative logit is multiplied, so it gets lower.
        assert_eq!(e.apply(-3.0), -4.5);
        assert_eq!(Entry::new(7).apply(-1.5), -1.5);
        assert_eq!(
            Entry { bias: f32::NEG_INFINITY, ..Entry::new(7) }.apply(2.0),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn a_draft_row_keeps_the_noise_and_loses_the_rest() {
        let params = SampleParams {
            temperature: 0.8,
            top_k: 20,
            seed: 9,
            logprobs: Logprobs { top: 5, chosen: true, after_chain: true },
            ..SampleParams::GREEDY
        };
        let row = params.row(12).with_entries(0, 4).with_mask(1);
        assert!(row.tail() && row.mask().is_some());
        let draft = row.for_draft();
        assert_eq!(draft, params.row(12).for_draft());
        assert!(draft.mask().is_none() && !draft.tail() && draft.logprobs().is_none());
        assert_eq!((draft.entries(), draft.program()), ((0, 0), row.program()));
    }
}
