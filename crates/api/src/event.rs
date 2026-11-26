//! What the engine reports while a request runs.

use serde::{Deserialize, Serialize};

use crate::EngineError;

/// One step of a request's life, as seen by the front end.
///
/// A request produces any number of [`Event::Progress`], [`Event::Text`] and
/// [`Event::Logprobs`] events followed by exactly one terminal event,
/// [`Event::Done`] or [`Event::Error`]. Nothing follows the terminal event.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// How far the prompt has got: when the request starts, and after every
    /// step that ran rows of its prompt, up to the step that ends it, whose
    /// progress comes before its text. Never after the first text.
    Progress(Progress),
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
        !matches!(self, Event::Text(_) | Event::Logprobs(_) | Event::Progress(_))
    }
}

/// How far the prompt of a request has got ([`Event::Progress`]), in the
/// terms of llama.cpp's `prompt_progress`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// All tokens of the prompt.
    pub total: u32,
    /// The tokens the prompt cache gave when the request started
    /// ([`Usage::cached_prompt_tokens`]).
    pub cached: u32,
    /// The tokens done, the cached ones included: from `cached` when the
    /// request starts to `total` when its prompt is read, rising with every
    /// report. What is left is counted in the rows the engine still runs for
    /// the prompt, the rows that rebuild the state from a checkpoint
    /// included, spread over the tokens the cache did not give; a step that
    /// runs its rows layer by layer (a pass) counts them by the share of its
    /// work done.
    pub processed: u32,
    /// Since the request started in the engine: its wait in line is not in it.
    pub elapsed: std::time::Duration,
}

/// The drafts of one request ([`Usage::drafts`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Drafts {
    /// Draft tokens the request's windows carried to be verified.
    pub proposed: u32,
    /// Those the model's picks confirmed.
    pub accepted: u32,
    /// The windows that carried drafts: the steps that verified some.
    pub windows: u32,
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
    /// The model emitted an end-of-generation token.
    Stop,
    /// A stop string of the request matched: its place in
    /// [`GenerateRequest::stop`](crate::GenerateRequest::stop).
    StopString(u32),
    /// `max_tokens` or the end of the context was reached.
    Length,
    /// [`Engine::cancel`](crate::Engine::cancel) took effect, or the engine
    /// is shutting down. A front end whose client is still connected treats
    /// this as a failure: the text is incomplete for a reason the client did
    /// not choose.
    Cancelled,
}

/// What a finished request took: its tokens, and how long it waited.
///
/// Two usages are equal when their counts of tokens are: the wait in line and
/// the time of the prompt are measurements of the moment, which two runs of
/// the same request never share, and the replayed rows and the drafts depend
/// on what else ran and on the drafts that were allowed; they are left out of
/// the comparison.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
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
    /// The tokens of the reply the model generated while it thought: from
    /// the start of its thinking to the tag that closes it, the tag included
    /// (a call of a tool that ends the thinking is not part of it). `None`
    /// from an engine that does not count them, or for a model that does not
    /// think.
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
    /// How long the request waited in the engine's line before it first
    /// started; `None` from an engine that does not say, or for a request
    /// that ended before it started.
    #[serde(default)]
    pub queued: Option<std::time::Duration>,
    /// The rows of the cached part of the prompt that were run again to
    /// rebuild the recurrent state from a checkpoint before it (part of
    /// `cached_prompt_tokens`: their cells came from the cache).
    #[serde(default)]
    pub replayed_prompt_tokens: u32,
    /// From the request's start in the engine (after its wait in line) to
    /// the step that ended its prompt and gave its first token; `None` from
    /// an engine that does not say, or for a request that ended before.
    #[serde(default)]
    pub prompt_time: Option<std::time::Duration>,
    /// The request's own drafts; `None` from an engine that does not count
    /// them per request.
    #[serde(default)]
    pub drafts: Option<Drafts>,
}

impl PartialEq for Usage {
    fn eq(&self, other: &Self) -> bool {
        let counts = |u: &Usage| {
            (u.prompt_tokens, u.completion_tokens, u.cached_prompt_tokens, u.output_hash)
        };
        counts(self) == counts(other) && self.reasoning_tokens == other.reasoning_tokens
    }
}

impl Eq for Usage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_text_and_probabilities_are_non_terminal() {
        assert!(!Event::Text("a".into()).is_terminal());
        assert!(!Event::Logprobs(Vec::new()).is_terminal());
        let progress =
            Progress { total: 8, cached: 2, processed: 2, elapsed: std::time::Duration::ZERO };
        assert!(!Event::Progress(progress).is_terminal());
        assert!(Event::Done { finish: FinishReason::Stop, usage: Usage::default() }.is_terminal());
        assert!(Event::Error(EngineError::Overloaded).is_terminal());
    }
}
