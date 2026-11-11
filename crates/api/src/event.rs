//! What the engine reports while a request runs.

use serde::{Deserialize, Serialize};

use crate::EngineError;

/// One step of a request's life, as seen by the front end.
///
/// A request produces any number of [`Event::Text`] events followed by exactly
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
    /// Generation ended.
    Done { finish: FinishReason, usage: Usage },
    /// The request failed. Text sent before this stays valid but incomplete.
    Error(EngineError),
}

impl Event {
    /// Whether this event ends the request.
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Event::Text(_))
    }
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
    fn only_text_is_non_terminal() {
        assert!(!Event::Text("a".into()).is_terminal());
        assert!(Event::Done { finish: FinishReason::Stop, usage: Usage::default() }.is_terminal());
        assert!(Event::Error(EngineError::Overloaded).is_terminal());
    }
}
