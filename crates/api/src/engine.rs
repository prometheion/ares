//! The trait the runtime implements.

use serde::{Deserialize, Serialize};

use crate::{
    ChatFormat, DecisionRequest, DecisionSink, EngineError, EngineStats, EventSink,
    GenerateRequest, Prompt, RequestId,
};

/// A model the engine serves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCard {
    /// The name clients put in `GenerateRequest::model`.
    pub id: String,
    /// The most tokens (prompt plus completion) one request can hold.
    pub context_length: u32,
    /// When the model became available, in seconds since the Unix epoch.
    pub created: u64,
}

/// The inference runtime as the server sees it.
///
/// `models`, `chat_format`, `submit`, `decide` and `cancel` are called from async code, so
/// none of them may block: the engine does its work on its own threads and
/// reports back through the [`EventSink`] or the [`DecisionSink`].
///
/// # Generation and decisions share steps
///
/// There are two kinds of request, generation ([`submit`](Engine::submit))
/// and decisions ([`decide`](Engine::decide)), and they run at the same
/// time, in the same engine steps. The rows of one step can be the next
/// tokens of generating sequences, their drafts, the option rows of
/// decisions and a slice of a prompt; the planner (`ares_sched::plan`)
/// puts all of them in one batch. Neither kind therefore waits for the other
/// to finish: a decision submitted while a long answer is streaming is
/// answered within a few steps of it, and the stream does not stall. An
/// implementation that served decisions from a queue of their own, or only
/// between generations, would break this contract.
pub trait Engine: Send + Sync + 'static {
    /// The models that can be requested right now.
    fn models(&self) -> Vec<ModelCard>;

    /// What the chat template of `model` declares, or `None` for a model the
    /// engine does not serve or one that has no chat template. It is asked
    /// for every chat request, so it only reads what the engine already has.
    fn chat_format(&self, _model: &str) -> Option<ChatFormat> {
        None
    }

    /// How many tokens `prompt` comes to for `model`: the prompt as
    /// [`submit`](Engine::submit) reads it (a conversation goes through the
    /// chat template with its tools and its variables), counted and not
    /// computed. Nothing is generated and no device is asked.
    ///
    /// Like [`stats`](Engine::stats) this one may take a while, since a
    /// prompt of any length is rendered and tokenized: async code calls it
    /// from a thread that may block. The default is an engine that cannot
    /// count.
    fn count_tokens(&self, _model: &str, _prompt: &Prompt) -> Result<u32, EngineError> {
        Err(EngineError::Unsupported("this engine does not count the tokens of a prompt".into()))
    }

    /// The tokens `prompt` comes to for `model`: what
    /// [`count_tokens`](Engine::count_tokens) counts, the ids themselves. A
    /// text is read as a prompt to continue is read, a conversation goes
    /// through the chat template. Like `count_tokens` it may take a while and
    /// is called from a thread that may block. The default is an engine that
    /// cannot tokenize.
    fn tokenize(&self, _model: &str, _prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        Err(EngineError::Unsupported("this engine does not tokenize".into()))
    }

    /// The bytes each of `tokens` stands for in a reply of `model`, in order:
    /// part of a character where a character spans tokens, and nothing for a
    /// token the engine does not write as text (a control token). An id that
    /// is not one of the model's is an [`EngineError::InvalidRequest`] that
    /// names it. The default is an engine that cannot say.
    fn pieces(&self, _model: &str, _tokens: &[u32]) -> Result<Vec<Vec<u8>>, EngineError> {
        Err(EngineError::Unsupported("this engine does not detokenize".into()))
    }

    /// The text of the chat template `model` renders conversations with: the
    /// model file's, or the one it was started with in its place. `None` for
    /// a model without one, or an engine that does not say. Must not block.
    fn chat_template(&self, _model: &str) -> Option<String> {
        None
    }

    /// Whether `model` reads images: it was loaded with an image tower (a
    /// projector). `false` for a model the engine does not serve, or an
    /// engine that does not say. Must not block.
    fn reads_images(&self, _model: &str) -> bool {
        false
    }

    /// Starts a request.
    ///
    /// Must return without waiting for the work. On `Ok` the engine owns the
    /// request and promises to end it with exactly one terminal event on
    /// `events`, unless the sink refuses an event (see [`EventSink`]), in
    /// which case the request is cancelled. On `Err` nothing has been
    /// started and no event will be sent.
    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError>;

    /// Starts a decision: answers the questions of `request` about its prompt
    /// by choosing among their options, and generates no text.
    ///
    /// Any model can decide ("Jev mode"), not only a native decision model.
    /// The prompt is computed (or found in the prefix cache), each option of
    /// each question is scored from the logits after it, and an option longer
    /// than one token is a fork of the prompt's sequence whose rows run in
    /// one step with the others. [`Question::answer`](crate::Question::answer)
    /// turns the scores into the numbers of the answer.
    ///
    /// Must return without waiting for the work. On `Ok` the engine owns the
    /// request and promises to deliver exactly one
    /// [`DecisionOutcome`](crate::DecisionOutcome) through `outcome` (the sink
    /// makes a second one impossible). On `Err`
    /// nothing has been started and nothing will be delivered. The request is
    /// validated ([`DecisionRequest::validate`]) by the front end; the engine
    /// refuses with [`EngineError::InvalidRequest`] what only it can know,
    /// such as more options than it can score or an option text that
    /// tokenizes to nothing, naming the question.
    ///
    /// Decisions and generation share steps, see the [trait
    /// documentation](Engine). Cancel with [`cancel`](Engine::cancel).
    fn decide(&self, request: DecisionRequest, outcome: DecisionSink) -> Result<(), EngineError>;

    /// Stops the request `id` as soon as the engine can and frees what it
    /// holds. The request still ends with a terminal event
    /// ([`FinishReason::Cancelled`](crate::FinishReason::Cancelled)), or for
    /// a decision with
    /// [`DecisionOutcome::Cancelled`](crate::DecisionOutcome::Cancelled), but
    /// whoever cancelled has usually stopped listening.
    ///
    /// Works on both kinds of request: ids are unique among all the requests
    /// the engine holds, generations and decisions together. Idempotent:
    /// cancelling a finished, cancelled or unknown request does nothing. Must
    /// not block.
    fn cancel(&self, id: RequestId);

    /// Why the engine can serve no request any more, once that is so: a device
    /// it runs on has failed, and every request from then on is refused with
    /// this reason. `None` while it works. Whoever holds several engines asks
    /// this to take a broken one down and leave the others serving. Must not
    /// block.
    fn failure(&self) -> Option<String> {
        None
    }

    /// Whether the failure is a step that never came back ([`Engine::failure`]
    /// says so): the engine's threads wait in it for good and hold the memory
    /// of the devices, which no unload gives back and the driver may not
    /// answer for any more; only a new image of the process frees it. Must
    /// not block.
    fn stuck(&self) -> bool {
        false
    }

    /// What the engine has done since it started and what it holds now, or
    /// `None` for an engine that keeps no such books (or has shut down).
    ///
    /// Unlike the other methods this one may wait, for the end of the step
    /// the engine is in: async code calls it from a thread that may block.
    fn stats(&self) -> Option<EngineStats> {
        None
    }
}
