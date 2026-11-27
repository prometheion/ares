//! The contract between the HTTP server and the inference runtime.
//!
//! The server knows nothing about models and the runtime knows nothing about
//! HTTP. They meet here: the server turns a wire request into a
//! [`GenerateRequest`], hands it to an [`Engine`] together with an
//! [`EventSink`], and the engine's own threads push [`Event`]s into the sink
//! until the request ends.
//!
//! A second kind of request is the decision: typed questions answered by
//! choosing among fixed options, with no generation. The server turns a wire
//! request into a [`DecisionRequest`] and hands it to [`Engine::decide`]
//! with a [`DecisionSink`], and the engine delivers one [`DecisionOutcome`].
//! Any model can decide, and decisions run in the same engine steps as
//! generation. How a chat model is asked a question ([`Question::render`])
//! and how its logits become an answer ([`Question::answer`]) are here, so
//! that every engine asks and answers alike.
//!
//! The crate has no async-runtime and no HTTP dependency on purpose. The
//! runtime side is a set of OS threads that step a GPU, and nothing in this
//! API makes them wait for, or even know about, an executor.
//!
//! Chat templates are the runtime's business, so everything the template
//! might read (tool definitions, tool calls, message names) crosses this
//! boundary untouched as [`Json`].

mod answer;
mod chat;
mod decision;
mod engine;
mod error;
mod event;
mod json;
mod message;
mod python;
mod request;
mod sampling;
mod sink;
mod stats;
mod structure;

pub use answer::{Answer, DecisionOutcome, DecisionResult, Verdict};
pub use chat::{ChatFormat, ThinkingFormat, ToolFormat};
pub use decision::{
    ChoiceOption, DecisionError, DecisionRequest, Labels, MAX_BRANCH_TOKENS, MAX_OPTIONS,
    MAX_REQUEST_TOKENS, Question, QuestionKind, Rendered, SINGLE_LABELS, question_prompt,
    state_text, user_turn,
};
pub use engine::{Engine, ModelCard};
pub use error::EngineError;
pub use event::{Alternative, Drafts, Event, FinishReason, Progress, TokenProbs, Usage};
pub use json::Json;
pub use message::{Image, Message, Role};
pub use request::{GenerateRequest, Prompt, RequestId};
pub use sampling::{
    BiasKey, LOGIT_BIAS_MAX, LogitBias, LogprobRequest, PENALTY_WINDOW_MAX, SAMPLER_STAGES_MAX,
    SamplerKind, Sampling, SamplingError, TOP_LOGPROBS_MAX,
};
pub use sink::{DecisionSink, EventSink};
pub use stats::{EngineStats, ExpertStats, RamCacheStats, SsdStats};
pub use structure::{Interrupt, Shape, Start, Structure};
