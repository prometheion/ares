//! Sampling parameters.

use serde::{Deserialize, Serialize};

use crate::EngineError;

/// The most tokens of history the repetition, presence and frequency
/// penalties and DRY look at.
pub const PENALTY_WINDOW_MAX: u32 = 1024;
/// The most tokens a logit bias may name.
pub const LOGIT_BIAS_MAX: usize = 1024;
/// The most alternatives next to a token whose probability is reported.
pub const TOP_LOGPROBS_MAX: u32 = 20;

/// The most samplers of an order that cut or scale: the engine's limit
/// ([`SamplerKind::order_problem`]).
pub const SAMPLER_STAGES_MAX: usize = 16;

/// A token, or a text whose tokens, that a logit bias names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BiasKey {
    Token(u32),
    /// Every token the text is made of (as llama.cpp reads a string key).
    Text(String),
}

/// One entry of a logit bias.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitBias {
    pub key: BiasKey,
    /// Added to the logit of the token, or `None` for a token that must never
    /// be picked.
    pub bias: Option<f32>,
}

/// The samplers of llama.cpp's chain that a request can put in an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SamplerKind {
    Penalties,
    Dry,
    TopNSigma,
    TopK,
    TypicalP,
    TopP,
    MinP,
    Xtc,
    Temperature,
}

impl SamplerKind {
    /// The name llama.cpp's `samplers` field gives the sampler.
    pub fn name(self) -> &'static str {
        match self {
            SamplerKind::Penalties => "penalties",
            SamplerKind::Dry => "dry",
            SamplerKind::TopNSigma => "top_n_sigma",
            SamplerKind::TopK => "top_k",
            SamplerKind::TypicalP => "typ_p",
            SamplerKind::TopP => "top_p",
            SamplerKind::MinP => "min_p",
            SamplerKind::Xtc => "xtc",
            SamplerKind::Temperature => "temperature",
        }
    }

    /// The sampler with this name in llama.cpp's `samplers` field, which
    /// reads a name in any case, with dashes or underscores or none, and a
    /// few other spellings.
    pub fn from_name(name: &str) -> Option<SamplerKind> {
        let plain: String = name
            .chars()
            .filter(|c| *c != '-' && *c != '_')
            .map(|c| c.to_ascii_lowercase())
            .collect();
        Some(match plain.as_str() {
            "penalties" => SamplerKind::Penalties,
            "dry" => SamplerKind::Dry,
            "topnsigma" => SamplerKind::TopNSigma,
            "topk" => SamplerKind::TopK,
            "typp" | "typ" | "typicalp" | "typical" => SamplerKind::TypicalP,
            "topp" | "nucleus" => SamplerKind::TopP,
            "minp" => SamplerKind::MinP,
            "xtc" => SamplerKind::Xtc,
            "temperature" | "temp" => SamplerKind::Temperature,
            _ => return None,
        })
    }

    /// Whether the sampler corrects the logits (the penalties, DRY) rather
    /// than cutting or scaling them.
    pub fn corrects(self) -> bool {
        matches!(self, SamplerKind::Penalties | SamplerKind::Dry)
    }

    /// Why `order` is no order the engine runs, or `None` when it is one:
    /// the penalties and DRY correct the logits, so they stand before the
    /// samplers that cut or scale, once each, and an order holds at most
    /// [`SAMPLER_STAGES_MAX`] of those. The requests and the server's
    /// defaults are held to it alike.
    pub fn order_problem(order: &[SamplerKind]) -> Option<String> {
        let (mut corrections, mut stages) = (Vec::new(), 0);
        for kind in order {
            if !kind.corrects() {
                stages += 1;
            } else if stages > 0 {
                return Some(format!(
                    "{} comes after a sampler that cuts or scales; it corrects the logits and \
                     has to come before them",
                    kind.name()
                ));
            } else if corrections.contains(kind) {
                return Some("penalties and dry can be listed once each".to_owned());
            } else {
                corrections.push(*kind);
            }
        }
        (stages > SAMPLER_STAGES_MAX).then(|| format!("at most {SAMPLER_STAGES_MAX} entries"))
    }

    /// The sampler a letter of llama.cpp's `samplers` string stands for.
    pub fn from_letter(letter: char) -> Option<SamplerKind> {
        Some(match letter {
            'e' => SamplerKind::Penalties,
            'd' => SamplerKind::Dry,
            's' => SamplerKind::TopNSigma,
            'k' => SamplerKind::TopK,
            'y' => SamplerKind::TypicalP,
            'p' => SamplerKind::TopP,
            'm' => SamplerKind::MinP,
            'x' => SamplerKind::Xtc,
            't' => SamplerKind::Temperature,
            _ => return None,
        })
    }
}

/// The probabilities a request wants back with its tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogprobRequest {
    /// How many of the most probable alternatives come with each token (at
    /// most [`TOP_LOGPROBS_MAX`]); 0 for the token's own probability only.
    pub top: u32,
    /// Which distribution the probabilities belong to. `false`: the model's
    /// own, the softmax of its logits before any sampler (OpenAI's
    /// `logprobs`, llama.cpp's `n_probs`). `true`: the one the token is drawn
    /// from, over the tokens that the samplers left, at their temperature
    /// (llama.cpp's `post_sampling_probs`).
    pub after_chain: bool,
}

/// How the next token is picked.
///
/// Every field is optional and `None` means "the model's own default": the
/// sampling settings that suit a model are part of the model, not of the
/// server, so the server must not invent values for clients that say nothing.
/// The fields that are not optional are lists and flags that are empty and off
/// by default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampling {
    /// 0 picks the most likely token every time.
    pub temperature: Option<f32>,
    /// Keep only the `top_k` most likely tokens. `Some(0)` switches the filter
    /// off, which is different from `None` (the model's default may be on).
    pub top_k: Option<u32>,
    /// Nucleus sampling: keep the smallest set of tokens whose probabilities
    /// add up to `top_p`. 1 keeps everything.
    pub top_p: Option<f32>,
    /// Drop tokens less likely than `min_p` times the most likely one.
    pub min_p: Option<f32>,
    /// Flat penalty on every token that appeared in the output (OpenAI).
    pub presence_penalty: Option<f32>,
    /// Penalty that grows with how often a token appeared in the output (OpenAI).
    pub frequency_penalty: Option<f32>,
    /// Multiplicative penalty on tokens that appeared so far (llama.cpp).
    /// 1 changes nothing.
    pub repeat_penalty: Option<f32>,
    /// How many of the last tokens (prompt and output) the three penalties
    /// above look at: at most [`PENALTY_WINDOW_MAX`]; 0 switches them off.
    pub repeat_last_n: Option<u32>,
    /// Locally typical sampling; 1 keeps everything.
    pub typical_p: Option<f32>,
    /// Keep the tokens within this many standard deviations of the logits
    /// below the best; 0 switches it off.
    pub top_n_sigma: Option<f32>,
    /// The chance that XTC removes the most probable tokens at a step.
    pub xtc_probability: Option<f32>,
    /// The probability a token needs to be a candidate for XTC's removal.
    pub xtc_threshold: Option<f32>,
    /// The range around `temperature` a dynamic temperature moves in.
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    /// 0 (off), 1 or 2: the mirostat that replaces the chain of samplers.
    pub mirostat: Option<u8>,
    /// Mirostat's target surprise and learning rate.
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    /// DRY: 0 switches it off.
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<u32>,
    /// How many of the last tokens DRY looks at: at most [`PENALTY_WINDOW_MAX`].
    pub dry_penalty_last_n: Option<u32>,
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// Bias added to the logits of tokens before anything else is done.
    pub logit_bias: Vec<LogitBias>,
    /// Never pick a token that ends the text.
    pub ignore_eos: bool,
    /// The samplers to apply and their order; `None` is llama.cpp's order.
    pub samplers: Option<Vec<SamplerKind>>,
    /// The probabilities to report with the tokens.
    pub logprobs: Option<LogprobRequest>,
    /// Makes sampling reproducible. `None` draws a fresh seed.
    pub seed: Option<u64>,
    /// The most tokens a reply of a model that thinks may think (llama.cpp's
    /// `reasoning_budget_tokens`, vLLM's `thinking_token_budget`): once its
    /// thinking has taken this many, the engine has the next rows pick
    /// [`reasoning_budget_message`](Sampling::reasoning_budget_message) and
    /// the end of the thinking, and the answer follows. `None`: no budget.
    pub reasoning_budget_tokens: Option<u32>,
    /// What is written before the end of the thinking when the budget is
    /// spent (llama.cpp's `reasoning_budget_message`); `None` or empty:
    /// nothing.
    pub reasoning_budget_message: Option<String>,
}

/// A sampling parameter outside its valid range.
///
/// It names the field and what was wrong with it so that a client can fix the
/// request without reading the engine's source.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{field} must be {requirement}, got {value}")]
pub struct SamplingError {
    pub field: &'static str,
    pub requirement: &'static str,
    pub value: f64,
}

impl Sampling {
    /// Checks every set field. NaN and infinities are refused everywhere: they
    /// do not come out of a JSON parser, but a front end that builds values
    /// from other sources must not be able to push one into the sampler,
    /// where it would silently poison the logits of a whole batch.
    pub fn validate(&self) -> Result<(), SamplingError> {
        check("temperature", "a finite number >= 0", self.temperature, |v| {
            v.is_finite() && v >= 0.0
        })?;
        check("top_p", "greater than 0 and at most 1", self.top_p, |v| v > 0.0 && v <= 1.0)?;
        check("min_p", "between 0 and 1", self.min_p, |v| (0.0..=1.0).contains(&v))?;
        check("presence_penalty", "between -2 and 2", self.presence_penalty, |v| {
            (-2.0..=2.0).contains(&v)
        })?;
        check("frequency_penalty", "between -2 and 2", self.frequency_penalty, |v| {
            (-2.0..=2.0).contains(&v)
        })?;
        check("repeat_penalty", "a finite number > 0", self.repeat_penalty, |v| {
            v.is_finite() && v > 0.0
        })?;
        check("typical_p", "greater than 0 and at most 1", self.typical_p, |v| {
            v > 0.0 && v <= 1.0
        })?;
        check("top_n_sigma", "a finite number", self.top_n_sigma, f32::is_finite)?;
        check("xtc_probability", "between 0 and 1", self.xtc_probability, |v| {
            (0.0..=1.0).contains(&v)
        })?;
        check("xtc_threshold", "between 0 and 1", self.xtc_threshold, |v| {
            (0.0..=1.0).contains(&v)
        })?;
        check("dynatemp_range", "a finite number >= 0", self.dynatemp_range, |v| {
            v.is_finite() && v >= 0.0
        })?;
        check("dynatemp_exponent", "a finite number >= 0", self.dynatemp_exponent, |v| {
            v.is_finite() && v >= 0.0
        })?;
        check("mirostat_tau", "a finite number > 0", self.mirostat_tau, |v| {
            v.is_finite() && v > 0.0
        })?;
        check("mirostat_eta", "a finite number >= 0", self.mirostat_eta, |v| {
            v.is_finite() && v >= 0.0
        })?;
        check("dry_multiplier", "a finite number >= 0", self.dry_multiplier, |v| {
            v.is_finite() && v >= 0.0
        })?;
        check("dry_base", "a finite number >= 0", self.dry_base, |v| v.is_finite() && v >= 0.0)?;
        if let Some(m) = self.mirostat
            && m > 2
        {
            return Err(SamplingError {
                field: "mirostat",
                requirement: "0, 1 or 2",
                value: f64::from(m),
            });
        }
        for (field, value) in [
            ("repeat_last_n", self.repeat_last_n),
            ("dry_penalty_last_n", self.dry_penalty_last_n),
            ("dry_allowed_length", self.dry_allowed_length),
        ] {
            if let Some(n) = value
                && n > PENALTY_WINDOW_MAX
            {
                return Err(SamplingError {
                    field,
                    requirement: "at most 1024 (the history the penalties look at is bounded)",
                    value: f64::from(n),
                });
            }
        }
        if self.logit_bias.len() > LOGIT_BIAS_MAX {
            return Err(SamplingError {
                field: "logit_bias",
                requirement: "at most 1024 entries",
                value: self.logit_bias.len() as f64,
            });
        }
        for entry in &self.logit_bias {
            if let Some(bias) = entry.bias
                && !bias.is_finite()
            {
                return Err(SamplingError {
                    field: "logit_bias",
                    requirement: "a finite number, or false to ban the token",
                    value: f64::from(bias),
                });
            }
        }
        if let Some(l) = self.logprobs
            && l.top > TOP_LOGPROBS_MAX
        {
            return Err(SamplingError {
                field: "top_logprobs",
                requirement: "at most 20",
                value: f64::from(l.top),
            });
        }
        Ok(())
    }
}

impl From<SamplingError> for EngineError {
    fn from(error: SamplingError) -> Self {
        EngineError::InvalidRequest(error.to_string())
    }
}

fn check(
    field: &'static str,
    requirement: &'static str,
    value: Option<f32>,
    valid: impl Fn(f32) -> bool,
) -> Result<(), SamplingError> {
    match value {
        Some(v) if !valid(v) => Err(SamplingError { field, requirement, value: f64::from(v) }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(f: impl FnOnce(&mut Sampling)) -> Sampling {
        let mut sampling = Sampling::default();
        f(&mut sampling);
        sampling
    }

    #[test]
    fn the_corrections_come_first_once_each_and_the_stages_are_bounded() {
        use SamplerKind::*;
        assert_eq!(SamplerKind::order_problem(&[Penalties, Dry, TopK, Temperature]), None);
        assert_eq!(SamplerKind::order_problem(&[TopK, MinP]), None);
        let after = SamplerKind::order_problem(&[TopK, Penalties]).unwrap();
        assert!(after.starts_with("penalties comes after a sampler that cuts"), "{after}");
        let twice = SamplerKind::order_problem(&[Dry, Dry, TopK]).unwrap();
        assert_eq!(twice, "penalties and dry can be listed once each");
        let many = vec![TopK; SAMPLER_STAGES_MAX + 1];
        assert_eq!(SamplerKind::order_problem(&many[..SAMPLER_STAGES_MAX]), None);
        assert_eq!(SamplerKind::order_problem(&many).unwrap(), "at most 16 entries");
    }

    #[test]
    fn nothing_set_is_valid_and_so_are_the_boundaries() {
        assert_eq!(Sampling::default().validate(), Ok(()));
        let edges = Sampling {
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
            presence_penalty: Some(-2.0),
            frequency_penalty: Some(2.0),
            repeat_penalty: Some(f32::MIN_POSITIVE),
            seed: Some(u64::MAX),
            ..Sampling::default()
        };
        assert_eq!(edges.validate(), Ok(()));
        assert_eq!(with(|s| s.min_p = Some(1.0)).validate(), Ok(()));
    }

    #[test]
    fn every_bad_value_is_refused_by_naming_its_field() {
        let cases = [
            ("temperature", with(|s| s.temperature = Some(f32::NAN))),
            ("temperature", with(|s| s.temperature = Some(-0.1))),
            ("temperature", with(|s| s.temperature = Some(f32::INFINITY))),
            ("top_p", with(|s| s.top_p = Some(0.0))),
            ("top_p", with(|s| s.top_p = Some(1.01))),
            ("top_p", with(|s| s.top_p = Some(f32::NAN))),
            ("min_p", with(|s| s.min_p = Some(-0.5))),
            ("min_p", with(|s| s.min_p = Some(1.5))),
            ("min_p", with(|s| s.min_p = Some(f32::NAN))),
            ("presence_penalty", with(|s| s.presence_penalty = Some(2.5))),
            ("presence_penalty", with(|s| s.presence_penalty = Some(f32::NAN))),
            ("frequency_penalty", with(|s| s.frequency_penalty = Some(-2.5))),
            ("frequency_penalty", with(|s| s.frequency_penalty = Some(f32::NEG_INFINITY))),
            ("repeat_penalty", with(|s| s.repeat_penalty = Some(0.0))),
            ("repeat_penalty", with(|s| s.repeat_penalty = Some(-1.0))),
            ("repeat_penalty", with(|s| s.repeat_penalty = Some(f32::INFINITY))),
        ];
        for (field, sampling) in cases {
            let error = sampling.validate().expect_err(field);
            assert_eq!(error.field, field, "{sampling:?}");
            assert!(error.to_string().starts_with(field), "{error}");
        }
    }

    #[test]
    fn the_message_says_what_was_wrong_in_plain_words() {
        let error = with(|s| s.temperature = Some(f32::NAN)).validate().unwrap_err();
        assert_eq!(error.to_string(), "temperature must be a finite number >= 0, got NaN");
        let error = with(|s| s.top_p = Some(1.5)).validate().unwrap_err();
        assert_eq!(error.to_string(), "top_p must be greater than 0 and at most 1, got 1.5");
    }

    #[test]
    fn the_first_bad_field_is_reported() {
        let sampling =
            Sampling { temperature: Some(-1.0), top_p: Some(9.0), ..Sampling::default() };
        assert_eq!(sampling.validate().unwrap_err().field, "temperature");
    }

    #[test]
    fn missing_fields_deserialize_to_none() {
        let sampling: Sampling = serde_json::from_str(r#"{"top_k":40}"#).unwrap();
        assert_eq!(sampling, Sampling { top_k: Some(40), ..Sampling::default() });
    }

    #[test]
    fn the_fields_of_the_other_samplers_are_checked_too() {
        let bad = [
            ("typical_p", with(|s| s.typical_p = Some(0.0))),
            ("typical_p", with(|s| s.typical_p = Some(1.5))),
            ("top_n_sigma", with(|s| s.top_n_sigma = Some(f32::NAN))),
            ("xtc_probability", with(|s| s.xtc_probability = Some(1.5))),
            ("xtc_threshold", with(|s| s.xtc_threshold = Some(-0.1))),
            ("dynatemp_range", with(|s| s.dynatemp_range = Some(-1.0))),
            ("mirostat", with(|s| s.mirostat = Some(3))),
            ("mirostat_tau", with(|s| s.mirostat_tau = Some(0.0))),
            ("mirostat_eta", with(|s| s.mirostat_eta = Some(-0.1))),
            ("dry_multiplier", with(|s| s.dry_multiplier = Some(f32::INFINITY))),
            ("dry_base", with(|s| s.dry_base = Some(-1.0))),
            ("repeat_last_n", with(|s| s.repeat_last_n = Some(PENALTY_WINDOW_MAX + 1))),
            ("dry_penalty_last_n", with(|s| s.dry_penalty_last_n = Some(100_000))),
            (
                "logit_bias",
                with(|s| {
                    s.logit_bias = vec![LogitBias { key: BiasKey::Token(1), bias: Some(f32::NAN) }];
                }),
            ),
            (
                "top_logprobs",
                with(|s| {
                    s.logprobs =
                        Some(LogprobRequest { top: TOP_LOGPROBS_MAX + 1, after_chain: false });
                }),
            ),
        ];
        for (field, sampling) in bad {
            assert_eq!(sampling.validate().expect_err(field).field, field);
        }
        let ok = with(|s| {
            s.repeat_last_n = Some(PENALTY_WINDOW_MAX);
            s.mirostat = Some(2);
            s.logit_bias = vec![LogitBias { key: BiasKey::Text("a".into()), bias: None }];
        });
        assert_eq!(ok.validate(), Ok(()));
    }

    #[test]
    fn the_samplers_have_the_names_llama_cpp_gives_them() {
        for kind in [
            SamplerKind::Penalties,
            SamplerKind::Dry,
            SamplerKind::TopNSigma,
            SamplerKind::TopK,
            SamplerKind::TypicalP,
            SamplerKind::TopP,
            SamplerKind::MinP,
            SamplerKind::Xtc,
            SamplerKind::Temperature,
        ] {
            assert_eq!(SamplerKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(SamplerKind::from_name("typical_p"), Some(SamplerKind::TypicalP));
        assert_eq!(SamplerKind::from_name("Top-K"), Some(SamplerKind::TopK));
        assert_eq!(SamplerKind::from_name("topnsigma"), Some(SamplerKind::TopNSigma));
        assert_eq!(SamplerKind::from_name("nucleus"), Some(SamplerKind::TopP));
        assert_eq!(SamplerKind::from_name("nonsense"), None);
        let letters: Vec<_> = "dkypsmtxe".chars().map(SamplerKind::from_letter).collect();
        assert!(letters.iter().all(Option::is_some));
        assert_eq!(SamplerKind::from_letter('i'), None);
    }
}
