//! Sampling parameters.

use serde::{Deserialize, Serialize};

use crate::EngineError;

/// How the next token is picked.
///
/// Every field is optional and `None` means "the model's own default": the
/// sampling settings that suit a model are part of the model, not of the
/// server, so the server must not invent values for clients that say nothing.
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
    /// Makes sampling reproducible. `None` draws a fresh seed.
    pub seed: Option<u64>,
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
}
