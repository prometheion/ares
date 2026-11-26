//! The core of every route that generates. A route of any API runs a
//! conversation in three steps, and none of them knows a wire format:
//!
//! 1. It reads its body into a [`Parsed`] chat request
//!    (`openai::request::from_json` for a body in the chat format; a route of
//!    another API writes its body over into that format first, as the
//!    `anthropic` module does).
//! 2. [`prepare`] places its system messages as the model's template takes
//!    them, applies what the request says about thinking and tools to it, and
//!    returns the [`Sorter`] of its reply.
//! 3. [`Generation::start`] hands the request to the engine, and
//!    [`Generation::next`] gives the reply as it comes: [`Piece`]s
//!    (reasoning, content, the start of a call, fragments of its arguments)
//!    and at the end the reason, the usage and the timings. The route writes
//!    them in its own format.
//!
//! A streaming route awaits the first step before it starts its response, so
//! that a request the engine refuses gets an HTTP status; dropping the
//! [`Generation`] (the client went away) cancels the request in the engine.
//! A route that counts a prompt prepares it the same way and asks the engine
//! for its count ([`count_prompt`]).

use std::time::Duration;

use ares_api::{ChatFormat, GenerateRequest, Message, Progress, Prompt, Role, TokenProbs, Usage};

use crate::ReasoningFormat;
use crate::bridge::{Item, Run, Timings};
use crate::error::ApiError;
use crate::models::Lease;
use crate::openai::request::Parsed;
use crate::reasoning;
use crate::routes::{App, model_of};
use crate::sorter::{Piece, Sorter};
use crate::tools;

/// Takes the model a chat request names ([`model_of`]: the request waits
/// for its load), places its system messages as the model's chat template
/// takes them ([`place_system_messages`] where the template declares
/// [`ChatFormat::system_first`]), gives the request the model's defaults for what it leaves
/// out and holds the length of its reply to the model's limit (the
/// `defaults` module), applies what the request says about thinking and
/// about tools to the request for the engine (the variables of the chat
/// template, the rule of the reply), and returns what sorts the text of its
/// reply with the hold on the model. `format` is how the reasoning is
/// returned when the request does not say; `None`: as the model's own
/// options say. `api` names the API the body was written for, in the log.
pub(crate) async fn prepare(
    app: &App,
    parsed: &mut Parsed,
    format: Option<ReasoningFormat>,
    api: &'static str,
) -> Result<(Sorter, Lease), ApiError> {
    let lease = model_of(app, &mut parsed.request.model).await?;
    let chat = lease.engine().chat_format(&parsed.request.model);
    if let (Some(ChatFormat { system_first: true, .. }), Prompt::Chat { messages, .. }) =
        (chat, &mut parsed.request.prompt)
    {
        place_system_messages(messages);
    }
    lease.defaults().apply(parsed);
    let format = format.unwrap_or(lease.reasoning_format());
    let thinking = reasoning::apply(parsed, &**lease.engine(), format)?;
    let calls = tools::apply(parsed, &**lease.engine(), &thinking)?;
    log_request(app, parsed, api);
    Ok((Sorter::new(thinking.splitter, calls), lease))
}

/// What the system messages of a conversation join: an empty line, as the
/// Responses route joins `instructions` with them.
const BETWEEN_SYSTEM_MESSAGES: &str = "\n\n";

/// The messages of the system (and of the developer, which is the system
/// here) as a chat template takes them that refuses a system message anywhere
/// but first ([`ChatFormat::system_first`], as Qwen's): the ones that stand
/// before every other message are one system message, their texts joined by
/// an empty line; one after another message becomes a message of the user
/// with its text, which is where a model reads what came in between (Claude
/// Code sends its environment and the tokens left that way, Codex its
/// instructions of a turn).
///
/// Every route that generates comes through [`prepare`]: the Messages and the
/// Responses routes write their bodies over into a chat body, with a system
/// turn where it stood.
pub(crate) fn place_system_messages(messages: &mut Vec<Message>) {
    let leading = messages.iter().take_while(|m| m.role == Role::System).count();
    if leading > 1 {
        let texts: Vec<String> =
            messages.drain(1..leading).map(|m| m.content).filter(|text| !text.is_empty()).collect();
        let first = &mut messages[0];
        for text in texts {
            if !first.content.is_empty() {
                first.content.push_str(BETWEEN_SYSTEM_MESSAGES);
            }
            first.content.push_str(&text);
        }
    }
    let from = usize::from(leading > 0);
    for message in messages.iter_mut().skip(from).filter(|m| m.role == Role::System) {
        message.role = Role::User;
    }
}

/// The line of the log for a request that is taken: its API, its parameters
/// as the engine gets them, and its prompt with `--log-text`.
fn log_request(app: &App, parsed: &Parsed, api: &'static str) {
    let request = &parsed.request;
    let sampling = &request.sampling;
    let (tools, continues) = match &request.prompt {
        ares_api::Prompt::Chat { tools, continue_final, .. } => {
            (tools.as_ref().map_or(0, Vec::len), *continue_final)
        }
        ares_api::Prompt::Text(_) => (0, false),
    };
    tracing::info!("request {} taken: {}", request.id, said(parsed, api, tools, continues));
    tracing::debug!(
        id = %request.id,
        model = %request.model,
        api,
        stream = parsed.stream,
        n = parsed.n,
        max_tokens = ?request.max_tokens,
        temperature = ?sampling.temperature,
        top_k = ?sampling.top_k,
        top_p = ?sampling.top_p,
        min_p = ?sampling.min_p,
        seed = ?sampling.seed,
        stops = request.stop.len(),
        tools,
        structure = request.structure.is_some(),
        draft = request.draft,
        continues,
        priority = request.priority,
        "request"
    );
    if app.config.log_text {
        let prompt = serde_json::to_string(&request.prompt).unwrap_or_default();
        tracing::info!(id = %request.id, %prompt, "the prompt");
    }
}

/// What a request asks, for a person: its API and model, and of its
/// parameters those it gives.
fn said(parsed: &Parsed, api: &str, tools: usize, continues: bool) -> String {
    let request = &parsed.request;
    let sampling = &request.sampling;
    let mut said = format!("{api}, model {}", request.model);
    if parsed.stream {
        said.push_str(", streamed");
    }
    if parsed.n > 1 {
        said.push_str(&format!(", {} completions", parsed.n));
    }
    let mut add = |name: &str, value: Option<String>| {
        if let Some(value) = value {
            said.push_str(&format!(", {name} {value}"));
        }
    };
    add("max_tokens", request.max_tokens.map(|n| n.to_string()));
    add("temperature", sampling.temperature.map(|t| t.to_string()));
    add("top_k", sampling.top_k.map(|k| k.to_string()));
    add("top_p", sampling.top_p.map(|p| p.to_string()));
    add("min_p", sampling.min_p.map(|p| p.to_string()));
    add("seed", sampling.seed.map(|seed| seed.to_string()));
    add("stop strings", (!request.stop.is_empty()).then(|| request.stop.len().to_string()));
    add("tools", (tools > 0).then(|| tools.to_string()));
    add("priority", (request.priority != 0).then(|| request.priority.to_string()));
    if request.structure.is_some() {
        said.push_str(", a grammar or schema");
    }
    if !request.draft {
        said.push_str(", no drafts");
    }
    if continues {
        said.push_str(", continues the last message");
    }
    said
}

/// A reply the engine is generating: the request in the engine and the
/// sorter of its text. Dropping it before its end cancels the request.
#[derive(Debug)]
pub(crate) struct Generation {
    run: Run,
    sorter: Sorter,
    /// The request's stop strings, to name the one that ended the reply.
    stops: Vec<String>,
    /// Whether the client hears how far the prompt has got
    /// ([`Step::Progress`]).
    progress: bool,
    /// Dropped after the run: the model may be unloaded when its requests
    /// have let go of the engine.
    _lease: Lease,
}

/// The next thing of a reply.
#[derive(Debug)]
pub(crate) enum Step {
    /// How far the prompt has got, for a client that asked
    /// ([`Generation::reporting`]).
    Progress(Progress),
    /// What the sorter found in the text that came; nothing when all of it
    /// is held back.
    Pieces(Vec<Piece>),
    /// The probabilities of the tokens whose text went out.
    Probs(Vec<TokenProbs>),
    End(End),
}

/// How a reply ended.
#[derive(Debug)]
pub(crate) struct End {
    /// What the sorter still held.
    pub rest: Vec<Piece>,
    /// `stop` (the model ended the reply, or a stop string did) or `length`.
    pub finish: &'static str,
    /// The stop string that ended the reply, if one did.
    pub stop: Option<String>,
    pub usage: Usage,
    pub timings: Timings,
}

impl Generation {
    /// Hands `request` to the engine of `lease`; `sorter` sorts its reply.
    pub fn start(
        app: &App,
        lease: &Lease,
        request: GenerateRequest,
        sorter: Sorter,
    ) -> Result<Self, ApiError> {
        let stops = request.stop.clone();
        let run = Run::start(
            lease.engine(),
            request,
            app.config.event_buffer,
            lease.flight(),
            app.stop.clone(),
            app.config.log_text,
        )?;
        Ok(Self { run, sorter, stops, progress: false, _lease: lease.clone() })
    }

    /// The reply, whose client hears how far the prompt has got when
    /// `progress` (llama.cpp's `return_progress` on a stream).
    pub fn reporting(self, progress: bool) -> Self {
        Self { progress, ..self }
    }

    /// The first step of a reply that is streamed, if the engine produces it
    /// within `wait`; `None` if it has not by then.
    ///
    /// A stream waits for its first step before its response starts, so that
    /// a request the engine refuses gets an HTTP status: the engine refuses
    /// at once, when it has read and counted the prompt. A long prompt is
    /// computed for minutes before the first token, though (16 thousand
    /// tokens of Claude Code took Flash-Next three and a half), and a client
    /// that sees no response for that long gives up: Claude Code went away
    /// after six minutes and sent the request again. So the response starts
    /// after `wait` without the first step; the keep-alive of the stream
    /// holds the connection while the prompt is computed, and a failure
    /// after that is an error in the stream.
    pub async fn first(&mut self, wait: Duration) -> Result<Option<Step>, ApiError> {
        // `next` can be dropped while it waits: the events stay queued.
        match tokio::time::timeout(wait, self.next()).await {
            Ok(step) => step.map(Some),
            Err(_) => Ok(None),
        }
    }

    /// The next step of the reply, or the reason there will be none. After
    /// an `Err` or a [`Step::End`] the reply is over.
    pub async fn next(&mut self) -> Result<Step, ApiError> {
        let item = loop {
            match self.run.next().await? {
                Item::Progress(progress) if self.progress => break Item::Progress(progress),
                Item::Progress(_) => {}
                item => break item,
            }
        };
        Ok(match item {
            Item::Progress(progress) => Step::Progress(progress),
            Item::Text(text) => Step::Pieces(self.sorter.push(&text)),
            Item::Probs(probs) => Step::Probs(probs),
            Item::Done { finish, stopped_by, usage, timings } => {
                let stop = stopped_by.and_then(|index| self.stops.get(index).cloned());
                Step::End(End { rest: self.sorter.finish(), finish, stop, usage, timings })
            }
        })
    }
}

/// How many tokens the prompt of `request` comes to: the request as a route
/// read and prepared it, counted by the engine and not run. An engine that
/// cannot count answers 501.
pub(crate) async fn count_prompt(lease: &Lease, request: GenerateRequest) -> Result<u32, ApiError> {
    let engine = lease.engine().clone();
    // Rendering and tokenizing a long prompt takes its time.
    let counted =
        tokio::task::spawn_blocking(move || engine.count_tokens(&request.model, &request.prompt))
            .await
            .map_err(|e| ApiError::internal(format!("the count of the prompt was lost: {e}")))?;
    counted.map_err(|e| ApiError::from_engine(&**lease.engine(), e))
}
