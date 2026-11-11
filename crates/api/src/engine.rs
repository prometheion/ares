//! The trait the runtime implements.

use serde::{Deserialize, Serialize};

use crate::{
    ChatFormat, DecisionRequest, DecisionSink, EngineError, EngineStats, EventSink,
    GenerateRequest, RequestId,
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

    /// What the engine has done since it started and what it holds now, or
    /// `None` for an engine that keeps no such books (or has shut down).
    ///
    /// Unlike the other methods this one may wait, for the end of the step
    /// the engine is in: async code calls it from a thread that may block.
    fn stats(&self) -> Option<EngineStats> {
        None
    }
}
