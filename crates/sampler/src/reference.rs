//! The pick of one row on the host: the definition every backend is held to.
//!
//! The kernels find their cut-offs by radix descents over integer histograms
//! and reduce with atomics; this module gets the same numbers from sorts and
//! loops, written from the definitions in the documentation of the crate. The
//! arithmetic that touches a floating-point number is the same function in
//! both, so the two agree on every bit of every record.
//!
//! # The chain
//!
//! A row's logits are first *corrected*: each [`Entry`] changes the logit of
//! its token, and a token the mask does not allow becomes minus infinity.
//! A logit that is NaN or plus infinity is reported (`SAMPLE_STATUS_NONFINITE`)
//! and counts as minus infinity, which no stage keeps.
//!
//! A row that is not sampled picks the most probable token, the lower token on
//! a tie. A sampled row starts with the set `S` of every finite token, ordered
//! by their *key*, the bits of the distance `d` of the logit below the best
//! logit. `S` always is the tokens with a key from `lo` to `hi`, the temperature
//! `T` the logits are read at is `1 / inv_t`, and the stages of the row's
//! program change them in order. A token's distance below the best token of
//! `S` is `d' = d - d(lo)`, and at the temperature `x = d' * inv_t`: its
//! probability within `S` is `exp(-x)` over the sum of that over `S`.
//!
//! - **top-n-sigma** keeps `d' <= n * sigma`, with sigma the standard
//!   deviation of the `d'` of `S`.
//! - **top-k** keeps the `k` tokens with the smallest keys of `S` and every
//!   token tied with the last of them.
//! - **typical** keeps the tokens of `S` whose `|x - mean(x)|`, `mean(x)` being
//!   the average of `x` weighted by the probability, is at most the smallest
//!   radius that holds more than the share `p` of the probability; `S`
//!   becomes the keys from the smallest to the largest of those tokens.
//! - **top-p** keeps the tokens with the smallest keys of `S`, up to the
//!   first key at which they hold at least the share `p` of the probability of
//!   `S` (at least one token), and every token tied with it.
//! - **min-p** keeps the tokens at least `p` times as probable as the best of
//!   `S`: `d' <= -ln(p) / inv_t`.
//! - **xtc** does nothing unless the row's uniform draw is at most the
//!   probability; then, if at least two tokens have a probability of `thr` or
//!   more, `S` loses every token more probable than the least probable of them.
//! - **temperature** `t` scales `inv_t` by `1 / t`; a `t` below
//!   [`TEMPERATURE_MIN`](crate::TEMPERATURE_MIN) keeps only the tokens tied for
//!   the best. A dynamic temperature first sets `t` from the entropy `H` of
//!   `S` (nothing happens for a single token): between `max(0, t - r)` and
//!   `t + r`, by the power `e` of `H / ln(n)`.
//! - **mirostat 2** keeps the tokens of `S` whose surprise `-log2 p` is at
//!   most `mu`, or the best one if there is none; **mirostat 1** estimates the
//!   exponent of a zipf law from the 100 most probable tokens, takes the `k`
//!   it implies with `mu`, and keeps that top-k.
//!
//! The pick is the token of `S` with the largest `logit * inv_t + noise`, the
//! noise being Gumbel of the row's seed, position and the token.
//!
//! After the pick, a row may report the probabilities of its tokens
//! ([`LogprobOut`]), and a mirostat row the surprise of its pick.

use std::collections::HashMap;

use crate::arith::{
    KEY_MAX, Sums, exp_d, ln_d, ln_fixed, pack, pow_d, share, unsortable, weight_fixed, xtc_draw,
};
use crate::mask::allows;
use crate::row::*;

/// The tokens a draw was made from and the temperature it was made at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kept {
    /// The tokens whose key (the bits of their distance below the best logit)
    /// is from `lo` to `hi`.
    pub lo: u32,
    pub hi: u32,
    /// One over the temperature the logits were read at.
    pub inv_t: f32,
}

impl Kept {
    /// Whether the token with this `logit` was kept, given the best logit.
    pub fn keeps(&self, l_max: f32, logit: f32) -> bool {
        let key = crate::arith::key_of(l_max, logit);
        logit > f32::NEG_INFINITY && key >= self.lo && key <= self.hi
    }
}

/// What the host computes for a row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reference {
    pub pick: SamplePick,
    pub logprobs: LogprobOut,
    pub kept: Kept,
}

/// A finite token of the row.
#[derive(Clone, Copy)]
struct Tok {
    token: u32,
    logit: f32,
    key: u32,
}

/// The state of the chain: `S` is the tokens with a key from `lo` to `hi`.
struct Chain {
    toks: Vec<Tok>,
    lo: u32,
    hi: u32,
    inv_t: f32,
    /// The `k` mirostat 1 chose.
    k: u32,
}

impl Chain {
    fn in_set(&self) -> impl Iterator<Item = &Tok> {
        self.toks.iter().filter(|t| t.key >= self.lo && t.key <= self.hi)
    }

    /// The distance below the best token of the set and the scaled distance.
    fn x_of(&self, key: u32) -> (f32, f32) {
        let d = f32::from_bits(key) - f32::from_bits(self.lo);
        (d, d * self.inv_t)
    }

    fn weight(&self, key: u32) -> u64 {
        weight_fixed(self.x_of(key).1.to_bits())
    }

    fn sums(&self) -> Sums {
        let mut sums = Sums::default();
        for t in self.in_set() {
            let (d, x) = self.x_of(t.key);
            sums.add(&Sums::of(d, x));
        }
        sums
    }

    /// The keys of the set, ascending.
    fn keys(&self) -> Vec<u32> {
        let mut keys: Vec<u32> = self.in_set().map(|t| t.key).collect();
        keys.sort_unstable();
        keys
    }

    fn cap(&mut self, hi: u32) {
        self.hi = self.hi.min(hi.min(KEY_MAX));
    }

    /// Keeps the `k` smallest keys and their ties.
    fn top_k(&mut self, k: u32) {
        let keys = self.keys();
        if k > 0 && (k as usize) <= keys.len() {
            self.cap(keys[k as usize - 1]);
        }
    }

    /// Keeps the smallest keys whose weights (by `weight_of(token)`, grouped
    /// by `key_of(token)`) add up to `target`.
    fn cut_by_weight(&self, target: u64, key_of: impl Fn(&Tok) -> u32) -> Option<u32> {
        let mut by_key: Vec<(u32, u64)> =
            self.in_set().map(|t| (key_of(t), self.weight(t.key))).collect();
        by_key.sort_unstable_by_key(|(key, _)| *key);
        let mut run = 0u64;
        let mut i = 0;
        while i < by_key.len() {
            let key = by_key[i].0;
            while i < by_key.len() && by_key[i].0 == key {
                run += by_key[i].1;
                i += 1;
            }
            if run >= target {
                return Some(key);
            }
        }
        None
    }

    fn temperature(&mut self, row: &SampleRow) {
        let mut t = row.float(W_TEMP);
        let mut inv = row.float(W_INV_TEMP);
        let delta = row.float(W_DYN_DELTA);
        if delta > 0.0 {
            let sums = self.sums();
            if sums.count <= 1 {
                return;
            }
            let (t0, delta) = (f64::from(t), f64::from(delta));
            let (min_t, max_t) = ((t0 - delta).max(0.0), t0 + delta);
            let normalized = (sums.entropy() / ln_d(sums.count as f64)).clamp(0.0, 1.0);
            t = (min_t + (max_t - min_t) * pow_d(normalized, f64::from(row.float(W_DYN_EXP))))
                as f32;
            inv = 1.0 / t;
        }
        if t < TEMPERATURE_MIN {
            self.hi = self.lo;
        } else {
            self.inv_t *= inv;
        }
    }

    fn mirostat1(&mut self, row: &SampleRow) {
        let keys = self.keys();
        let mut k = 1u32;
        if keys.len() >= 2 {
            let m = MIROSTAT_M.min(keys.len());
            let x: Vec<f64> = keys[..m].iter().map(|&key| f64::from(self.x_of(key).1)).collect();
            let (mut sum_tb, mut sum_tt) = (0.0f64, 0.0f64);
            for i in 0..m - 1 {
                let t = ln_d((i + 2) as f64 / (i + 1) as f64);
                sum_tb += t * (x[i + 1] - x[i]);
                sum_tt += t * t;
            }
            let s_hat = sum_tb / sum_tt;
            // A flat top (no exponent to estimate) leaves the best token.
            if s_hat.is_finite() && s_hat > 0.0 {
                let eps = s_hat - 1.0;
                let n_vocab = f64::from(row.word(W_VOCAB));
                let two_mu = exp_d(f64::from(row.float(W_MU)) * std::f64::consts::LN_2);
                let base = (eps * two_mu) / (1.0 - exp_d(-eps * ln_d(n_vocab)));
                if base.is_finite() && base > 0.0 {
                    let value = pow_d(base, 1.0 / s_hat);
                    if value.is_finite() && value >= 1.0 {
                        k = value.min(f64::from(u32::MAX)) as u32;
                    }
                }
            }
        }
        self.k = k;
        self.top_k(k);
    }

    fn mirostat2(&mut self, row: &SampleRow) {
        let z = self.sums().z;
        let limit = f64::from(row.float(W_MU)) * std::f64::consts::LN_2 - ln_fixed(z);
        let kept =
            self.in_set().filter(|t| f64::from(self.x_of(t.key).1) <= limit).map(|t| t.key).max();
        self.hi = kept.unwrap_or(self.lo);
    }

    fn apply(&mut self, stage: Stage, row: &SampleRow) {
        match stage {
            Stage::TopNSigma => {
                let sums = self.sums();
                if sums.count > 1 {
                    let lim = f32::from_bits(self.lo) + row.float(W_N_SIGMA) * sums.sigma();
                    self.cap(lim.to_bits());
                }
            }
            Stage::TopK => self.top_k(row.word(W_TOP_K)),
            Stage::Typical => {
                let sums = self.sums();
                let mean = sums.mean_x();
                let radius = |t: &Tok| (self.x_of(t.key).1 - mean).abs().to_bits();
                let target = share(sums.z, row.word(W_TYPICAL)) + 1;
                if let Some(cut) = self.cut_by_weight(target, radius) {
                    let kept: Vec<u32> =
                        self.in_set().filter(|t| radius(t) <= cut).map(|t| t.key).collect();
                    let lo = *kept.iter().min().expect("the cut keeps a token");
                    let hi = *kept.iter().max().expect("the cut keeps a token");
                    self.lo = lo;
                    self.hi = hi;
                }
            }
            Stage::TopP => {
                let z = self.sums().z;
                let target = share(z, row.word(W_TOP_P)).max(1);
                if let Some(cut) = self.cut_by_weight(target, |t| t.key) {
                    self.cap(cut);
                }
            }
            Stage::MinP => {
                let lim = f32::from_bits(self.lo) + row.float(W_MIN_P) / self.inv_t;
                self.cap(lim.to_bits());
            }
            Stage::Xtc => {
                if xtc_draw(row.seed(), row.word(W_POSITION)) > row.float(W_XTC_PROB) {
                    return;
                }
                let z = self.sums().z;
                let q = u128::from(row.word(W_XTC_THR));
                let target = (((u128::from(z) * q) + 0xffff_ffff) >> 32).max(1) as u64;
                let qualifying: Vec<u32> =
                    self.in_set().filter(|t| self.weight(t.key) >= target).map(|t| t.key).collect();
                if qualifying.len() >= 2 {
                    self.lo = *qualifying.iter().max().expect("two tokens qualify");
                }
            }
            Stage::Temperature => self.temperature(row),
            Stage::Mirostat1 => self.mirostat1(row),
            Stage::Mirostat2 => self.mirostat2(row),
        }
    }
}

/// The probability of `x` within a set with the sum `z`, as a logarithm.
fn log_probability(x: f32, z: u64) -> f32 {
    (-f64::from(x) - ln_fixed(z)) as f32
}

/// The pick of one row on the host: the arithmetic of the kernels, written
/// from the definitions (a sort where the kernels descend by radix).
///
/// `logits[i]` is the logit of token `token_of(i)` as the model computed it.
/// `entries` correct it, and `mask` (words, a bit per token) restricts the
/// tokens, when the row says so.
pub fn sample_reference(
    logits: &[f32],
    token_of: impl Fn(usize) -> u32,
    row: &SampleRow,
    entries: &[Entry],
    mask: Option<&[u32]>,
) -> Reference {
    let by_token: HashMap<u32, &Entry> = entries.iter().map(|e| (e.token, e)).collect();
    let masked = row.mask().is_some();
    assert!(!masked || mask.is_some(), "the row names a mask and none was given");
    let mut status = 0;
    let canon: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            let token = token_of(i);
            let mut l = by_token.get(&token).map_or(raw, |e| e.apply(raw));
            if masked && !allows(mask.expect("checked"), token) {
                l = f32::NEG_INFINITY;
            }
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
        aux: 0.0,
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

    let mut chain = Chain {
        toks: canon
            .iter()
            .enumerate()
            .filter(|(_, l)| **l > f32::NEG_INFINITY)
            .map(|(i, &logit)| Tok { token: token_of(i), logit, key: key_of_logit(l_max, logit) })
            .collect(),
        lo: 0,
        hi: 0,
        inv_t: 1.0,
        k: 0,
    };
    if !row.sampled() {
        pick.token = pick.cand_token[0];
    } else {
        chain.hi = KEY_MAX;
        for stage in row.program() {
            chain.apply(stage, row);
        }
        let mut best = 0u64;
        for t in chain.in_set() {
            let score =
                t.logit * chain.inv_t + crate::arith::gumbel(row.seed(), row.position(), t.token);
            best = best.max(pack(score, t.token));
        }
        if best == 0 {
            pick.status |= SAMPLE_STATUS_EMPTY;
        } else {
            pick.token = u32::MAX - best as u32;
            pick.value = unsortable((best >> 32) as u32);
        }
    }

    let mut logprobs = LogprobOut::NONE;
    if let Some((after_chain, top)) = row.logprobs() {
        logprobs = if after_chain {
            post_logprobs(&chain, &pick, top as usize)
        } else {
            let raw: Vec<(u32, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &l)| {
                    let l = l + 0.0;
                    (token_of(i), if l <= f32::MAX { l } else { f32::NEG_INFINITY })
                })
                .collect();
            raw_logprobs(&raw, &pick, top as usize)
        };
    }
    if row.mirostat() && pick.token != u32::MAX {
        let sums = chain.sums();
        let t = chain.in_set().find(|t| t.token == pick.token).expect("the pick is in the set");
        let x = f64::from(chain.x_of(t.key).1);
        pick.aux = ((ln_fixed(sums.z) + x) / std::f64::consts::LN_2) as f32;
    }
    Reference { pick, logprobs, kept: Kept { lo: chain.lo, hi: chain.hi, inv_t: chain.inv_t } }
}

fn key_of_logit(l_max: f32, logit: f32) -> u32 {
    crate::arith::key_of(l_max, logit)
}

/// The probabilities of the model's own softmax: over every token, before
/// any processing.
fn raw_logprobs(raw: &[(u32, f32)], pick: &SamplePick, top: usize) -> LogprobOut {
    let mut out = LogprobOut::NONE;
    let l_max = raw.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    if l_max == f32::NEG_INFINITY {
        return out;
    }
    let mut z = 0u64;
    for &(_, l) in raw {
        if l > f32::NEG_INFINITY {
            z += weight_fixed(key_of_logit(l_max, l));
        }
    }
    let logprob = |l: f32| log_probability(f32::from_bits(key_of_logit(l_max, l)), z);
    let mut packed: Vec<u64> = raw
        .iter()
        .filter(|(_, l)| *l > f32::NEG_INFINITY)
        .map(|&(token, l)| pack(l, token))
        .collect();
    packed.sort_unstable_by(|a, b| b.cmp(a));
    for (c, &p) in packed.iter().take(top.min(LOGPROBS_MAX)).enumerate() {
        out.token[c] = u32::MAX - p as u32;
        out.logprob[c] = logprob(unsortable((p >> 32) as u32));
        out.n += 1;
    }
    if let Some(&(_, l)) = raw.iter().find(|(token, _)| *token == pick.token) {
        out.chosen = logprob(l);
    }
    out
}

/// The probabilities the pick is drawn from: over the tokens the chain left.
/// A row that takes the most probable token has the tokens tied for it.
fn post_logprobs(chain: &Chain, pick: &SamplePick, top: usize) -> LogprobOut {
    let mut out = LogprobOut::NONE;
    if chain.in_set().next().is_none() {
        return out;
    }
    let z = chain.sums().z;
    let logprob = |key: u32| log_probability(chain.x_of(key).1, z);
    let mut set: Vec<(u32, u32)> = chain.in_set().map(|t| (t.key, t.token)).collect();
    set.sort_unstable();
    for (c, &(key, token)) in set.iter().take(top.min(LOGPROBS_MAX)).enumerate() {
        out.token[c] = token;
        out.logprob[c] = logprob(key);
        out.n += 1;
    }
    if let Some(t) = chain.in_set().find(|t| t.token == pick.token) {
        out.chosen = logprob(t.key);
    }
    out
}
