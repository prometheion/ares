//! What the engine reports while a request runs.

use serde::{Deserialize, Serialize};

use crate::EngineError;

/// One step of a request's life, as seen by the front end.
///
/// A request produces any number of [`Event::Text`] and [`Event::Logprobs`] events followed by exactly
/// one terminal event, [`Event::Done`] or [`Event::Error`]. Nothing follows the
/// terminal event.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A piece of output text.
    ///
    /// It is valid UTF-8 by construction (a `String`) and must never split a
    /// character: the engine holds back the bytes of a character that a token
    /// boundary cut in half until the rest arrives. The front end forwards
    /// pieces as they are, so a split character would reach the client as
    /// two broken halves.
    Text(String),
    /// The probabilities of the tokens whose text has been sent with the
    /// [`Event::Text`] events so far (and not reported before), for a request
    /// that asked for them. A token's probability is reported once the text
    /// of the token has gone out whole.
    Logprobs(Vec<TokenProbs>),
    /// Generation ended.
    Done { finish: FinishReason, usage: Usage },
    /// The request failed. Text sent before this stays valid but incomplete.
    Error(EngineError),
}

impl Event {
    /// Whether this event ends the request.
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Event::Text(_) | Event::Logprobs(_))
    }
}

/// The probability of a generated token and of the alternatives it had.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenProbs {
    pub token: u32,
    /// The bytes the token prints: part of a character where a character
    /// spans tokens.
    pub bytes: Vec<u8>,
    /// The natural logarithm of its probability.
    pub logprob: f32,
    /// The most probable tokens, most probable first, with the same measure.
    pub alternatives: Vec<Alternative>,
}

/// A token that could have been generated instead.
#[derive(Debug, Clone, PartialEq)]
pub struct Alternative {
    pub token: u32,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FinishReason {
    /// The model emitted an end-of-generation token or a stop string matched.
    Stop,
    /// `max_tokens` or the end of the context was reached.
    Length,
    /// [`Engine::cancel`](crate::Engine::cancel) took effect, or the engine
    /// is shutting down. A front end whose client is still connected treats
    /// this as a failure: the text is incomplete for a reason the client did
    /// not choose.
    Cancelled,
}

/// Token counts of a finished request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// All tokens of the prompt, including the cached ones.
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// The part of the prompt that was served from the engine's cache instead
    /// of being computed.
    pub cached_prompt_tokens: u32,
    /// A hash of the token ids of the reply (FNV-1a over their four bytes
    /// each, little end first), so that two replies can be compared without
    /// keeping their tokens: the text with drafts is the text without, or it
    /// is not.
    pub output_hash: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_text_and_probabilities_are_non_terminal() {
        assert!(!Event::Text("a".into()).is_terminal());
        assert!(!Event::Logprobs(Vec::new()).is_terminal());
        assert!(Event::Done { finish: FinishReason::Stop, usage: Usage::default() }.is_terminal());
        assert!(Event::Error(EngineError::Overloaded).is_terminal());
    }
}
