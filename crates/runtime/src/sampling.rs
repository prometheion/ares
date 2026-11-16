//! What a request asks of the sampler, and the part of it the host works out.
//!
//! [`RequestSampler`] holds one request's settings for its whole life. Before
//! a step it makes the [`SampleRow`] of every wanted row of the request: the
//! settings, the position, and the corrections and the mask of that row. After
//! the step it takes what the rows report: the surprise that moves mirostat's
//! state, and the probabilities of the tokens the request asked about.
//!
//! The corrections (penalties, DRY, logit bias) depend on the tokens before a
//! row, and a row of a window that verifies drafts comes after the drafts
//! before it: each row is corrected for its own path, so that the window picks
//! what the sequence would pick one token at a time.

use std::collections::HashMap;
use std::sync::Arc;

use ares_api::{
    Alternative, BiasKey, EngineError, PENALTY_WINDOW_MAX, Prompt, SamplerKind, Sampling,
    TokenProbs,
};
use ares_sampler::penalties::{
    BIAS_MAX, Breakers, Corrections, DryParams, PenaltyParams, WINDOW_MAX, entries_for,
};
use ares_sampler::{
    Entry, LogprobOut, Logprobs, Mirostat, Order, SampleParams, SampleRow, Stage, TokenMask,
};

use crate::exec::Pick;
use crate::model::Model;

const _: () =
    assert!(PENALTY_WINDOW_MAX == WINDOW_MAX, "the API and the sampler agree on the window");

/// A rule about which tokens the next token of a request may be, for what
/// decides it from the text so far (a grammar).
///
/// The sampler is asked for a mask before each step, for every row of the
/// request that wants its logits: for the row of a window that verifies drafts,
/// `drafts` are the drafts before the row, which the row's token follows.
pub trait TokenFilter: Send {
    /// Sets in `mask` (which allows every token when it is handed over) the
    /// tokens that may follow `reply`, the tokens the request has generated,
    /// and `drafts`. Returns whether the row is restricted at all: a row that
    /// is not takes no mask.
    fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool;

    /// How many of the leading `drafts` the rule allows after `reply`: the
    /// drafts before the first token it forbids. A draft it forbids cannot be
    /// picked, so the rows from there on would be wasted; the conductor cuts
    /// them from the window. A rule that cannot tell allows all of them.
    fn admitted(&mut self, _reply: &[u32], drafts: &[u32]) -> usize {
        drafts.len()
    }

    /// How many rows of a window the rule leaves free, counted from the first
    /// row: row `k` follows `reply` and the first `k` of `tokens`, and it is
    /// free when the rule restricts nothing there and [`mask`] would say so.
    /// The count stops at the first row the rule binds, so it is 0 when the
    /// rule binds the token after `reply` and at most one more than `tokens`
    /// has.
    ///
    /// This is what lets a request with a rule take the drafts the model makes
    /// on the device, which the host does not know when it writes the masks:
    /// such a window is verified only while its first row is free, none of
    /// its rows carries a mask, and of what it picked the request keeps the
    /// tokens of its free rows: the conductor asks again when the picks are
    /// back, with the drafts they confirmed as `tokens`. A rule that cannot
    /// tell binds every row.
    ///
    /// [`mask`]: TokenFilter::mask
    fn free_rows(&mut self, _reply: &[u32], _tokens: &[u32]) -> usize {
        0
    }

    /// Why the rule can no longer be followed, if it cannot (the engine behind
    /// it failed in the middle of the reply). The request ends with an error:
    /// text that goes on without the rule is not what was asked for.
    fn failure(&mut self) -> Option<String> {
        None
    }
}

/// What the conductor keeps for all requests: the tokens that end a
/// generation and the DRY breakers made of a list of strings, both of which
/// take a pass over the vocabulary to find.
#[derive(Default)]
pub(crate) struct Context {
    ends: Option<Arc<Vec<u32>>>,
    breakers: HashMap<Vec<String>, Arc<Breakers>>,
}

impl Context {
    fn ends(&mut self, model: &dyn Model) -> Arc<Vec<u32>> {
        self.ends
            .get_or_insert_with(|| {
                Arc::new((0..model.vocab()).filter(|t| model.ends(*t)).collect())
            })
            .clone()
    }

    fn breakers(&mut self, model: &dyn Model, strings: &[String]) -> Arc<Breakers> {
        self.breakers
            .entry(strings.to_vec())
            .or_insert_with(|| {
                Arc::new(Breakers::build(
                    strings,
                    model.vocab(),
                    |token| {
                        let mut bytes = Vec::new();
                        model.piece(token, &mut bytes);
                        bytes
                    },
                    |text| {
                        let text = String::from_utf8_lossy(text).into_owned();
                        model.encode(&Prompt::Text(text)).unwrap_or_default()
                    },
                ))
            })
            .clone()
    }
}

/// The corrections and masks of the rows of one step, as the record holds them.
#[derive(Default)]
pub(crate) struct StepRows {
    pub entries: Vec<Entry>,
    /// The masks, `words` words each.
    pub masks: Vec<u32>,
}

impl StepRows {
    pub fn clear(&mut self) {
        self.entries.clear();
        self.masks.clear();
    }
}

/// A probability that waits for the text of its token to go out.
struct Queued {
    probs: TokenProbs,
    /// Bytes of generated text up to and including this token's.
    end: usize,
}

/// One request's settings and the state they keep.
pub(crate) struct RequestSampler {
    params: SampleParams,
    corrections: Corrections,
    /// How many tokens of the path the corrections read.
    window: usize,
    filter: Option<Box<dyn TokenFilter>>,
    mask: TokenMask,
    mirostat_tau: f32,
    mirostat_eta: f32,
    /// The tokens of a row's path the corrections read, made fresh for every row.
    path: Vec<u32>,
    /// The probabilities of the picks of the step, and then of the tokens
    /// whose text is not out yet.
    stash: Vec<Option<LogprobOut>>,
    queue: std::collections::VecDeque<Queued>,
    generated_bytes: usize,
    text_bytes: usize,
}

fn invalid(message: String) -> EngineError {
    EngineError::InvalidRequest(message)
}

impl RequestSampler {
    /// Builds the sampler of a request from what it asked for and what the
    /// model leaves open. `seed` is the seed of a request that names none.
    pub fn new(
        sampling: &Sampling,
        model: &dyn Model,
        context: &mut Context,
        seed: u64,
        filter: Option<Box<dyn TokenFilter>>,
    ) -> Result<Self, EngineError> {
        let vocab = model.vocab();
        let mut p = model.sampling();
        p.vocab = vocab;
        p.seed = sampling.seed.unwrap_or(seed);
        p.temperature = sampling.temperature.unwrap_or(p.temperature);
        p.top_k = sampling.top_k.unwrap_or(p.top_k);
        p.top_p = sampling.top_p.unwrap_or(p.top_p);
        p.min_p = sampling.min_p.unwrap_or(p.min_p);
        p.typical_p = sampling.typical_p.unwrap_or(p.typical_p);
        p.top_n_sigma = sampling.top_n_sigma.unwrap_or(p.top_n_sigma);
        p.xtc_probability = sampling.xtc_probability.unwrap_or(p.xtc_probability);
        p.xtc_threshold = sampling.xtc_threshold.unwrap_or(p.xtc_threshold);
        p.dynatemp_range = sampling.dynatemp_range.unwrap_or(p.dynatemp_range);
        p.dynatemp_exponent = sampling.dynatemp_exponent.unwrap_or(p.dynatemp_exponent);
        let tau = sampling.mirostat_tau.unwrap_or(5.0);
        let eta = sampling.mirostat_eta.unwrap_or(0.1);
        p.mirostat = match sampling.mirostat {
            None | Some(0) => None,
            Some(1) => Some(Mirostat::V1),
            Some(2) => Some(Mirostat::V2),
            Some(other) => {
                return Err(invalid(format!("mirostat must be 0, 1 or 2, got {other}")));
            }
        };
        p.mirostat_mu = 2.0 * tau;
        p.logprobs = match sampling.logprobs {
            Some(l) => Logprobs { top: l.top as u8, chosen: true, after_chain: l.after_chain },
            None => Logprobs::default(),
        };

        // The order of the stages, and whether the penalties and DRY are
        // among them: they come first or not at all.
        let (mut penalties_on, mut dry_on) = (true, true);
        if let Some(kinds) = &sampling.samplers {
            let mut stages = Vec::new();
            let (mut penalties_seen, mut dry_seen) = (false, false);
            for kind in kinds {
                let stage = match kind {
                    SamplerKind::Penalties | SamplerKind::Dry => {
                        if !stages.is_empty() {
                            return Err(invalid(format!(
                                "samplers: {} comes after a sampler that cuts or scales; it \
                                 corrects the logits and has to come before them",
                                if *kind == SamplerKind::Dry { "dry" } else { "penalties" }
                            )));
                        }
                        let seen = if *kind == SamplerKind::Dry {
                            &mut dry_seen
                        } else {
                            &mut penalties_seen
                        };
                        if std::mem::replace(seen, true) {
                            return Err(invalid(
                                "samplers: penalties and dry can be listed once each".into(),
                            ));
                        }
                        continue;
                    }
                    SamplerKind::TopNSigma => Stage::TopNSigma,
                    SamplerKind::TopK => Stage::TopK,
                    SamplerKind::TypicalP => Stage::Typical,
                    SamplerKind::TopP => Stage::TopP,
                    SamplerKind::MinP => Stage::MinP,
                    SamplerKind::Xtc => Stage::Xtc,
                    SamplerKind::Temperature => Stage::Temperature,
                };
                stages.push(stage);
            }
            p.order = Order::new(&stages).ok_or_else(|| {
                invalid(format!("samplers: at most {} entries", ares_sampler::MAX_STAGES))
            })?;
            (penalties_on, dry_on) = (penalties_seen, dry_seen);
        }

        let mut corrections = Corrections::none();
        // Mirostat replaces the chain: llama.cpp applies neither penalties nor
        // DRY with it, and neither does this.
        if p.mirostat.is_none() {
            let last_n = sampling.repeat_last_n.unwrap_or(64);
            corrections.penalties = PenaltyParams {
                last_n,
                repeat: sampling.repeat_penalty.unwrap_or(1.0),
                frequency: sampling.frequency_penalty.unwrap_or(0.0),
                presence: sampling.presence_penalty.unwrap_or(0.0),
            };
            if !penalties_on {
                corrections.penalties = PenaltyParams::OFF;
            }
            corrections.dry = DryParams {
                multiplier: sampling.dry_multiplier.unwrap_or(0.0),
                base: sampling.dry_base.unwrap_or(1.75),
                allowed_length: sampling.dry_allowed_length.unwrap_or(2),
                last_n: sampling.dry_penalty_last_n.unwrap_or(64),
            };
            if !dry_on {
                corrections.dry = DryParams::OFF;
            }
            if corrections.dry.is_on() {
                let default = ["\n", ":", "\"", "*"].map(String::from);
                let strings = sampling.dry_sequence_breakers.as_deref().unwrap_or(&default);
                corrections.breakers = context.breakers(model, strings);
            }
        }
        // The bias applies to every chain, mirostat's included.
        for entry in &sampling.logit_bias {
            let bias = entry.bias.unwrap_or(f32::NEG_INFINITY);
            match &entry.key {
                BiasKey::Token(token) => {
                    if *token >= vocab {
                        return Err(invalid(format!(
                            "logit_bias: token {token} is not one of the {vocab} tokens of the model"
                        )));
                    }
                    corrections.bias.push((*token, bias));
                }
                BiasKey::Text(text) => {
                    let tokens = model.encode(&Prompt::Text(text.clone()))?;
                    corrections.bias.extend(tokens.into_iter().map(|t| (t, bias)));
                }
            }
        }
        if sampling.ignore_eos {
            corrections.bias.extend(context.ends(model).iter().map(|t| (*t, f32::NEG_INFINITY)));
        }
        if corrections.bias.len() > BIAS_MAX {
            return Err(invalid(format!(
                "logit_bias names {} tokens; at most {BIAS_MAX} are served",
                corrections.bias.len()
            )));
        }
        let window = if corrections.penalties.is_on() { corrections.penalties.last_n } else { 0 }
            .max(if corrections.dry.is_on() { corrections.dry.last_n } else { 0 });
        Ok(Self {
            params: p,
            corrections,
            window: window as usize,
            filter,
            mask: TokenMask::all(vocab),
            mirostat_tau: tau,
            mirostat_eta: eta,
            path: Vec::new(),
            stash: Vec::new(),
            queue: std::collections::VecDeque::new(),
            generated_bytes: 0,
            text_bytes: 0,
        })
    }

    /// Whether a window of this request may verify drafts. Mirostat's state
    /// moves with every pick, by the probability of the token picked, which a
    /// row after the first of a window cannot know before the row before it
    /// is picked.
    pub fn allows_drafts(&self) -> bool {
        !self.params.stateful()
    }

    /// Whether the model's own drafter is asked for a chain for this request.
    /// Its chain runs in a step of its own, before the round, and knows no
    /// rule and no correction; the round verifies its drafts as it verifies
    /// any, with their masks and corrections. A request with a rule asks for
    /// a chain while the rule is not bound (after `tokens`, the tokens of the
    /// request so far, the prompt first, the next token is free): a chain
    /// made where the rule binds would mostly be cut.
    pub fn allows_model_drafts(&mut self, tokens: &[u32], prompt_len: usize) -> bool {
        self.allows_drafts() && self.free_rows(tokens, prompt_len, &[]) > 0
    }

    /// How many rows after `tokens` (the request so far, the prompt first) and
    /// the first `k` of `accepted` are rows the request's rule leaves free.
    /// Without a rule every row is free.
    pub fn free_rows(&mut self, tokens: &[u32], prompt_len: usize, accepted: &[u32]) -> usize {
        match &mut self.filter {
            Some(filter) => filter.free_rows(&tokens[prompt_len.min(tokens.len())..], accepted),
            None => accepted.len() + 1,
        }
    }

    /// Cuts from `drafts` (and from `chances`, the chance of each) the first
    /// draft the request's rule forbids and the ones after it. `tokens` are the
    /// tokens of the request so far, the prompt first.
    pub fn cut_drafts(
        &mut self,
        tokens: &[u32],
        prompt_len: usize,
        drafts: &mut Vec<u32>,
        chances: &mut Vec<f64>,
    ) {
        let Some(filter) = &mut self.filter else { return };
        let allowed = filter.admitted(&tokens[prompt_len.min(tokens.len())..], drafts);
        drafts.truncate(allowed);
        chances.truncate(allowed);
    }

    /// Why the request's rule failed, if it did.
    pub fn failure(&mut self) -> Option<String> {
        self.filter.as_mut().and_then(|filter| filter.failure())
    }

    /// The sampler row of the wanted row at `position` of a request whose
    /// tokens so far are `tokens` (the prompt first, `prompt_len` of them) and
    /// whose window has the drafts `drafts`: the row follows the first `k` of
    /// them. Its corrections and mask go to `rows`.
    pub fn row(
        &mut self,
        position: u32,
        tokens: &[u32],
        prompt_len: usize,
        drafts: &[u32],
        k: usize,
        rows: &mut StepRows,
    ) -> SampleRow {
        let mut row = self.params.row(position);
        if self.corrections.is_on() {
            self.path.clear();
            self.path.extend_from_slice(&tokens[tokens.len().saturating_sub(self.window)..]);
            self.path.extend_from_slice(&drafts[..k]);
            let entries = entries_for(&self.path, &self.corrections);
            if !entries.is_empty() {
                row = row.with_entries(rows.entries.len() as u32, entries.len() as u32);
                rows.entries.extend_from_slice(&entries);
            }
        }
        if let Some(filter) = &mut self.filter {
            self.mask.fill(true);
            if filter.mask(&tokens[prompt_len.min(tokens.len())..], &drafts[..k], &mut self.mask) {
                row = row.with_mask((rows.masks.len() / self.mask.words().len()) as u32);
                rows.masks.extend_from_slice(self.mask.words());
            }
        }
        row
    }

    /// Takes what the rows of a step reported. `picks` are the picks of this
    /// request's rows that become tokens, in order, and `probs` the
    /// probabilities of the rows (empty when no row reported any).
    pub fn picked(&mut self, picks: &[Pick], probs: &[LogprobOut]) {
        for pick in picks {
            if self.params.mirostat.is_some() {
                self.params.mirostat_mu -= self.mirostat_eta * (pick.aux - self.mirostat_tau);
            }
        }
        self.stash.clear();
        let asked = self.params.logprobs.is_asked();
        self.stash.extend((0..picks.len()).map(|i| probs.get(i).filter(|_| asked).copied()));
    }

    /// Whether the request asked for probabilities.
    pub fn wants_probs(&self) -> bool {
        self.params.logprobs.is_asked()
    }

    /// The token number `i` of the step's picks became a token of the reply:
    /// `bytes` is its text.
    pub fn token_taken(&mut self, model: &dyn Model, i: usize, token: u32, bytes: &[u8]) {
        self.generated_bytes += bytes.len();
        let Some(Some(lp)) = self.stash.get(i) else { return };
        let piece = |token: u32| {
            let mut bytes = Vec::new();
            model.piece(token, &mut bytes);
            bytes
        };
        let alternatives = (0..lp.n as usize)
            // A probability that no float holds is not reported.
            .filter(|&c| lp.logprob[c] > LOGPROB_FLOOR)
            .map(|c| Alternative {
                token: lp.token[c],
                bytes: piece(lp.token[c]),
                logprob: lp.logprob[c],
            })
            .collect();
        self.queue.push_back(Queued {
            probs: TokenProbs { token, bytes: bytes.to_vec(), logprob: lp.chosen, alternatives },
            end: self.generated_bytes,
        });
    }

    /// The reply ended at its token limit or at a token that ends it, not at
    /// a stop string. The tokens whose text has not gone out whole belong to
    /// the reply all the same (the limit can cut a character in two, and the
    /// bytes of half a character never print): the next [`text_sent`]
    /// returns their probabilities too.
    ///
    /// [`text_sent`]: RequestSampler::text_sent
    pub fn reply_ended(&mut self) {
        self.text_bytes = usize::MAX;
    }

    /// `n` bytes of text went out. Returns the probabilities of the tokens
    /// whose text has gone out whole. The probabilities of tokens whose text
    /// never goes out because a stop string held it back are never reported.
    pub fn text_sent(&mut self, n: usize) -> Vec<TokenProbs> {
        self.text_bytes = self.text_bytes.saturating_add(n);
        let mut out = Vec::new();
        while self.queue.front().is_some_and(|q| q.end <= self.text_bytes) {
            out.push(self.queue.pop_front().expect("checked").probs);
        }
        out
    }
}

/// The least log-probability a token is reported with: below it the
/// probability is not a number of a float.
const LOGPROB_FLOOR: f32 = -100.0;

#[cfg(test)]
mod tests {
    use super::*;
    use ares_api::{LogitBias, LogprobRequest};

    use crate::{Split, SyntheticConfig, SyntheticModel};

    fn model() -> SyntheticModel {
        SyntheticModel::new(SyntheticConfig::small(Split::Single, 1)).unwrap()
    }

    fn sampler(sampling: &Sampling) -> Result<RequestSampler, EngineError> {
        RequestSampler::new(sampling, &model(), &mut Context::default(), 7, None)
    }

    #[test]
    fn a_plain_request_has_nothing_to_correct() {
        let mut s =
            sampler(&Sampling { temperature: Some(0.8), top_k: Some(40), ..Sampling::default() })
                .unwrap();
        assert!(s.allows_drafts() && !s.wants_probs());
        let mut rows = StepRows::default();
        let row = s.row(5, &[1, 2, 3], 3, &[], 0, &mut rows);
        assert!(rows.entries.is_empty() && rows.masks.is_empty());
        assert_eq!((row.entries().1, row.mask()), (0, None));
        assert!(row.sampled());
    }

    #[test]
    fn each_row_of_a_window_is_corrected_for_its_own_path() {
        let sampling = Sampling {
            frequency_penalty: Some(1.0),
            repeat_last_n: Some(3),
            ..Sampling::default()
        };
        let mut s = sampler(&sampling).unwrap();
        let mut rows = StepRows::default();
        let tokens = [9, 1, 2, 3];
        let drafts = [3, 4];
        let r0 = s.row(10, &tokens, 1, &drafts, 0, &mut rows);
        let r1 = s.row(11, &tokens, 1, &drafts, 1, &mut rows);
        let r2 = s.row(12, &tokens, 1, &drafts, 2, &mut rows);
        assert_eq!((r0.entries(), r1.entries(), r2.entries()), ((0, 3), (3, 2), (5, 2)));
        // Row 1 continues 1 2 3 3: the last three are 2 3 3.
        let of = |at: u32, n: u32| &rows.entries[at as usize..(at + n) as usize];
        let sub = |entries: &[Entry], t| entries.iter().find(|e| e.token == t).map(|e| e.sub);
        assert_eq!((sub(of(3, 2), 3), sub(of(3, 2), 2)), (Some(2.0), Some(1.0)));
        // Row 2 continues 3 3 4: the token 3 twice, the token 4 once.
        assert_eq!((sub(of(5, 2), 3), sub(of(5, 2), 4)), (Some(2.0), Some(1.0)));
    }

    #[test]
    fn a_window_turned_off_by_the_list_of_samplers_is_off() {
        let sampling = Sampling {
            presence_penalty: Some(1.0),
            samplers: Some(vec![SamplerKind::TopK, SamplerKind::Temperature]),
            ..Sampling::default()
        };
        let mut s = sampler(&sampling).unwrap();
        let mut rows = StepRows::default();
        s.row(0, &[1, 2], 2, &[], 0, &mut rows);
        assert!(rows.entries.is_empty(), "the penalties are not in the list");
        // Penalties after a cut would need the logits as they were.
        let late = Sampling {
            samplers: Some(vec![SamplerKind::TopK, SamplerKind::Penalties]),
            ..Sampling::default()
        };
        let error = sampler(&late).err().unwrap();
        assert!(error.to_string().contains("penalties"), "{error}");
    }

    #[test]
    fn a_bias_a_ban_and_ignore_eos_become_corrections() {
        let mut config = SyntheticConfig::small(Split::Single, 1);
        config.eos = Some(256);
        let model = SyntheticModel::new(config).unwrap();
        let sampling = Sampling {
            logit_bias: vec![
                LogitBias { key: BiasKey::Token(65), bias: Some(2.5) },
                LogitBias { key: BiasKey::Token(66), bias: None },
                LogitBias { key: BiasKey::Text("hi".into()), bias: Some(-1.0) },
            ],
            ignore_eos: true,
            ..Sampling::default()
        };
        let mut s =
            RequestSampler::new(&sampling, &model, &mut Context::default(), 1, None).unwrap();
        let mut rows = StepRows::default();
        s.row(0, &[1], 1, &[], 0, &mut rows);
        let bias = |t| rows.entries.iter().find(|e| e.token == t).map(|e| e.bias);
        assert_eq!(bias(65), Some(2.5));
        assert_eq!(bias(66), Some(f32::NEG_INFINITY));
        assert_eq!(bias(u32::from(b'h')), Some(-1.0));
        assert_eq!(bias(256), Some(f32::NEG_INFINITY), "the token that ends the text");
        let bad = Sampling {
            logit_bias: vec![LogitBias { key: BiasKey::Token(1 << 30), bias: Some(1.0) }],
            ..Sampling::default()
        };
        assert!(RequestSampler::new(&bad, &model, &mut Context::default(), 1, None).is_err());
    }

    struct Only(Vec<u32>);

    impl TokenFilter for Only {
        fn mask(&mut self, reply: &[u32], drafts: &[u32], mask: &mut TokenMask) -> bool {
            // Only the tokens after the reply and the drafts so far, counted.
            let n = reply.len() + drafts.len();
            mask.fill(false);
            mask.set(self.0[n % self.0.len()], true);
            true
        }
    }

    #[test]
    fn a_filter_makes_a_mask_for_every_row() {
        let mut s = RequestSampler::new(
            &Sampling::default(),
            &model(),
            &mut Context::default(),
            1,
            Some(Box::new(Only(vec![10, 11, 12]))),
        )
        .unwrap();
        let mut rows = StepRows::default();
        let words = TokenMask::words_for(model().vocab());
        let r0 = s.row(5, &[1, 2, 3], 2, &[], 0, &mut rows);
        let r1 = s.row(6, &[1, 2, 3], 2, &[7, 8], 2, &mut rows);
        assert_eq!((r0.mask(), r1.mask()), (Some(0), Some(1)));
        assert_eq!(rows.masks.len(), 2 * words);
        // After one reply token: token 11; after the reply and two drafts: 10.
        assert_eq!(rows.masks[0], 1 << 11);
        assert_eq!(rows.masks[words], 1 << 10);
    }

    #[test]
    fn mirostat_moves_its_state_and_takes_no_drafts() {
        let sampling = Sampling {
            mirostat: Some(2),
            mirostat_tau: Some(5.0),
            mirostat_eta: Some(0.5),
            ..Sampling::default()
        };
        let mut s = sampler(&sampling).unwrap();
        assert!(!s.allows_drafts());
        let mut rows = StepRows::default();
        assert!(s.row(0, &[1], 1, &[], 0, &mut rows).mirostat());
        // mu starts at 2 tau, and moves against the error of the surprise.
        s.picked(&[Pick { token: 3, logit: 0.0, aux: 7.0 }], &[]);
        assert_eq!(s.row(1, &[1], 1, &[], 0, &mut rows), {
            let mut p = s.params;
            p.mirostat_mu = 10.0 - 0.5 * (7.0 - 5.0);
            p.row(1)
        });
        let wrong = Sampling { mirostat: Some(3), ..Sampling::default() };
        assert!(sampler(&wrong).is_err());
    }

    #[test]
    fn probabilities_wait_for_their_text() {
        let sampling = Sampling {
            logprobs: Some(LogprobRequest { top: 2, after_chain: false }),
            ..Sampling::default()
        };
        let m = model();
        let mut s = RequestSampler::new(&sampling, &m, &mut Context::default(), 1, None).unwrap();
        let mut lp = LogprobOut::NONE;
        (lp.n, lp.chosen) = (2, -0.5);
        lp.token[..2].copy_from_slice(&[65, 66]);
        lp.logprob[..2].copy_from_slice(&[-0.5, -100.5]);
        s.picked(&[Pick::default(), Pick::default()], &[lp, lp]);
        s.token_taken(&m, 0, 65, b"A");
        s.token_taken(&m, 1, 66, b"B");
        // Nothing has gone out; then the first byte; then both.
        assert!(s.text_sent(0).is_empty());
        let first = s.text_sent(1);
        assert_eq!((first.len(), first[0].token, first[0].logprob), (1, 65, -0.5));
        assert_eq!(first[0].alternatives.len(), 1, "a probability below the floor is not reported");
        assert_eq!(s.text_sent(1).len(), 1);
        // A token whose text never goes out reports nothing.
        s.picked(&[Pick::default()], &[lp]);
        s.token_taken(&m, 0, 67, b"C");
        assert!(s.text_sent(0).is_empty());
    }

    #[test]
    fn a_reply_that_ends_reports_the_tokens_whose_text_never_went_out() {
        let sampling = Sampling {
            logprobs: Some(LogprobRequest { top: 1, after_chain: false }),
            ..Sampling::default()
        };
        let m = model();
        let mut s = RequestSampler::new(&sampling, &m, &mut Context::default(), 1, None).unwrap();
        let mut lp = LogprobOut::NONE;
        (lp.n, lp.chosen) = (1, -0.5);
        lp.token[0] = 65;
        lp.logprob[0] = -0.5;
        s.picked(&[Pick::default(); 3], &[lp, lp, lp]);
        // "A", then the first two bytes of a character of three that the
        // token limit cuts off.
        s.token_taken(&m, 0, 65, b"A");
        s.token_taken(&m, 1, 66, &[0xe2, 0x82]);
        s.token_taken(&m, 2, 67, &[]);
        assert_eq!(s.text_sent(1).len(), 1);
        assert!(s.text_sent(0).is_empty());
        s.reply_ended();
        let rest = s.text_sent(0);
        assert_eq!(rest.iter().map(|p| p.token).collect::<Vec<_>>(), [66, 67]);
        assert!(s.text_sent(5).is_empty(), "each probability goes out once");
    }
}
