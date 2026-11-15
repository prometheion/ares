//! The OpenAI Responses API, without state: `POST /v1/responses` and
//! `POST /v1/responses/input_tokens`.
//!
//! A Responses request is served as the chat request that asks the same, on
//! the same path through the engine: [`request`] writes its body over into a
//! chat body, the reader of chat bodies (`wire::from_json`) and the core of
//! the routes (`routes::prepare`, `routes::Reply`) do the rest, and [`reply`]
//! writes what comes back as output items and as the typed events of a
//! stream. A reply is therefore the reply `/v1/chat/completions` gives, with
//! the same prefix cache, the same rule for the calls of tools and the same
//! end when the client goes away.
//!
//! # The request
//!
//! | field | here |
//! |---|---|
//! | `model`, `temperature`, `top_p`, `stream` | as on the chat route |
//! | `instructions` | a system message |
//! | `input` | a string (one message of the user), or items; see below |
//! | `tools` | functions: `type: "function"`, `name`, `description`, `parameters`, `strict` |
//! | `tool_choice` | `auto`, `none`, `required`, `{"type": "function", "name"}` |
//! | `parallel_tool_calls` | as on the chat route |
//! | `max_output_tokens` | the limit of tokens; a reply it cuts is `incomplete` |
//! | `reasoning` | `effort` is the `reasoning_effort` of the chat route (`none` switches thinking off); `summary: "auto"` is accepted and no summary is written |
//! | `text.format` | `text`, `json_object`, `json_schema`: the `response_format` of the chat route |
//! | `include` | `reasoning.encrypted_content` is accepted and adds nothing; see below |
//! | `truncation` | `disabled`, which is what the server does |
//! | `store`, `background` | `false` |
//! | `metadata`, `user`, `safety_identifier`, `service_tier`, `client_metadata` | bookkeeping: accepted, without effect; `metadata` and `user` are said back |
//! | `prompt_cache_key`, `prompt_cache_retention`, `prompt_cache_options` | accepted, without effect: the server caches every prefix by itself and promises no retention |
//!
//! Items of `input`: messages (`role` `user`, `assistant`, `system` or
//! `developer`, with a string or parts of `input_text` and `output_text`),
//! `function_call`, `function_call_output` (a string or `input_text` parts)
//! and `reasoning`. The parts of one message are concatenated, as the chat
//! route does it. `instructions` and the messages of the roles `system` and
//! `developer` that stand before the conversation are one system message,
//! joined by an empty line: Codex puts its instructions at the head of
//! `input`, and the chat template of Qwen3.5-0.8B refuses a system message
//! that is not the first one. Such a message inside a conversation stays
//! where it stands. The items one reply returned for a turn (reasoning,
//! message, calls) are one message of the assistant again.
//!
//! Anything else is refused with a 400 that names it: `input_image`,
//! `input_file` and `input_audio` parts (input other than text is not
//! built), tools that are not functions (`web_search`, `file_search`,
//! `code_interpreter`, `computer_use`, `mcp`, `local_shell`, `custom` and
//! the like) and the items they leave in a history, `prompt` templates,
//! `top_logprobs`, `text.verbosity`, `reasoning.summary` of a given kind,
//! `truncation: "auto"`, the sampler's fields of the chat route (only
//! members of this API are read here), and every member this module does not
//! know.
//!
//! # Decisions
//!
//! **No state.** The route keeps no responses. A response says `"store":
//! false`, also when the request left `store` out (OpenAI's default is
//! `true`). A request that sets `store: true` is refused: it asks for
//! something the server does not do, and the clients that work without
//! stored responses say so themselves (Codex sends `store: false` with every
//! request; the OpenAI SDKs send the member only when their caller sets it).
//! `previous_response_id`, `conversation`, `background: true`, an
//! `item_reference` and a `prompt` template are refused by name, and the
//! routes of a stored response (`GET` and `DELETE /v1/responses/{id}`,
//! `cancel`, `input_items`) answer 404 and say why.
//!
//! **Reasoning items.** The reasoning of a reply is one item of the type
//! `reasoning`, before the message: its `content` is the reasoning as the
//! model wrote it (a part of the type `reasoning_text`, what the API has for
//! models whose reasoning is not hidden), its `summary` is empty, and it has
//! no `encrypted_content`: the server has nothing to hide and no secret to
//! encrypt with. A reasoning item that comes back is the reasoning of its
//! turn: the text of its `content`, whatever its id says. The item of a reply
//! sent back unchanged is therefore written into the prompt as the model
//! generated it, on every turn the same, which is what lets the prefix cache
//! continue the conversation (see "A conversation that continues from the
//! cache" in `docs/ARCHITECTURE.md`). A `summary` and an `encrypted_content`
//! of an item are accepted and nothing of them enters the prompt: a summary
//! is not the reasoning, and encrypted content is text only the service that
//! wrote it can read.
//!
//! **`include`.** `reasoning.encrypted_content` asks for reasoning items a
//! client without stored responses can send back, and Codex asks for it with
//! every request. Every reasoning item of this server is such an item, so
//! the value is accepted and adds nothing. The other values ask for data the
//! server does not have (probabilities of tokens, results of tools it does
//! not run, images) and are refused by name.
//!
//! **`truncation`.** `disabled` is what the server does: an input that does
//! not fit the context is an error (`context_length_exceeded`). `auto` would
//! drop items from a conversation, which the server does not do.
//!
//! **The usage.** `input_tokens` is the whole prompt and
//! `input_tokens_details.cached_tokens` the part of it the prefix cache gave.
//! There are no `output_tokens_details`: the engine does not count the
//! tokens of the reasoning apart.
//!
//! **`output_text`.** The response has no member of that name: the official
//! SDKs put it together from the items themselves.
//!
//! **Where this is not llama-server.** llama-server drops tools that are not
//! functions, `store`, `background`, `truncation`, `include`, `text` and
//! every member it does not know without a word, and reads a `tool_choice`
//! that names a function as `auto`; here each is served or refused. It
//! requires a reasoning item to have a `summary` and a `content` that is not
//! empty, and returns `encrypted_content: ""`. Its stream has no
//! `sequence_number`, no `response.reasoning_text.done`, no
//! `response.function_call_arguments.done` and no events of the parts of a
//! reasoning item, and it calls every response `completed`, also one the
//! limit of tokens cut.

mod reply;
mod request;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::convert::Infallible;

use ares_api::RequestId;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive};
use axum::response::{IntoResponse, Response};

use self::reply::{Ending, Head, Output, Written, change_events, error_event};
use self::request::{Echo, Purpose};
use crate::ReasoningFormat;
use crate::error::ApiError;
use crate::models::Lease;
use crate::reply::Sorter;
use crate::routes::{
    self, App, End, Reply, Step, json_response, next_request_id, read_body, sse_response,
};
use crate::wire::{self, Endpoint, Parsed};

/// `POST /v1/responses`.
pub(crate) async fn create(State(app): State<App>, request: Request) -> Response {
    match try_create(app, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// `POST /v1/responses/input_tokens`.
pub(crate) async fn input_tokens(State(app): State<App>, request: Request) -> Response {
    match try_count(app, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// The routes of a stored response: `/v1/responses/{id}` and what hangs
/// below it. The server keeps none, so there is none to find.
pub(crate) async fn stored(request: Request) -> ApiError {
    let path = request.uri().path();
    let message = if path.ends_with("/compact") {
        format!(
            "there is no endpoint '{path}': the server does not compact conversations; send the \
             whole conversation to /v1/responses"
        )
    } else {
        format!(
            "there is no stored response at '{path}': this server keeps no responses (every \
             response says \"store\": false), so none can be fetched, deleted, cancelled or \
             listed"
        )
    };
    ApiError { status: StatusCode::NOT_FOUND, code: "not_found", ..ApiError::invalid(message) }
}

/// Reads the body of a request into the chat request it asks for, what
/// sorts its reply, and what the response says back.
///
/// The reasoning is always taken out of the text: a reasoning item is where
/// this API has it, whatever `--reasoning-format` says about
/// `reasoning_content`.
async fn read(
    app: &App,
    body: &[u8],
    id: RequestId,
    purpose: Purpose,
) -> Result<(Parsed, Sorter, Echo, Lease), ApiError> {
    let (chat, echo) = request::to_chat(&wire::read_json(body)?, purpose)?;
    let mut parsed = wire::from_json(Endpoint::Chat, &chat, id).map_err(request::renamed)?;
    let (sorter, lease) = routes::prepare(app, &mut parsed, Some(ReasoningFormat::Auto))
        .await
        .map_err(request::renamed)?;
    Ok((parsed, sorter, echo, lease))
}

async fn try_create(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, app.config.max_body_bytes).await?;
    let id = next_request_id();
    let (parsed, sorter, echo, lease) = read(&app, &body, id, Purpose::Reply).await?;
    let head = Head {
        id: format!("resp_{id}"),
        model: parsed.request.model.clone(),
        created_at: wire::unix_time(),
        echo,
    };
    tracing::info!(
        %id,
        model = %parsed.request.model,
        endpoint = "Responses",
        stream = parsed.stream,
        max_tokens = ?parsed.request.max_tokens,
        "request"
    );
    let reply = Reply::start(&app, &lease, parsed.request, sorter)?;
    let output = Output::new(id.to_string());
    if parsed.stream { stream(reply, head, output).await } else { whole(reply, head, output).await }
}

/// Counts the prompt of a request and generates nothing: the body is read
/// as `/v1/responses` reads it.
async fn try_count(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, app.config.max_body_bytes).await?;
    let (parsed, _, _, lease) = read(&app, &body, next_request_id(), Purpose::Count).await?;
    let tokens = routes::count_prompt(&lease, parsed.request).await?;
    Ok(json_response(wire::input_tokens(tokens)))
}

/// The answer that is not streamed: the response object.
async fn whole(mut reply: Reply, head: Head, mut output: Output) -> Result<Response, ApiError> {
    loop {
        match reply.next().await? {
            Step::Pieces(pieces) => {
                output.push(pieces);
            }
            // The probabilities of tokens are not served on this route.
            Step::Probs(_) => {}
            Step::End(End { rest, finish, usage, timings }) => {
                let ending = Ending { finish, usage, timings };
                output.push(rest);
                output.finish(ending.cut());
                return Ok(json_response(head.response(output.items(), &ending)));
            }
        }
    }
}

/// The streamed answer: `response.created`, `response.in_progress`, the
/// events of the items, and `response.completed` or `response.incomplete`
/// with the response as it is returned whole.
///
/// As on the chat route the response starts when the engine has produced its
/// first step, so that a request the engine refuses gets an HTTP status. A
/// failure after that is an `error` event, and the stream ends with it.
async fn stream(mut reply: Reply, head: Head, output: Output) -> Result<Response, ApiError> {
    let first = reply.next().await?;
    let mut sequence = 0;
    let queue = VecDeque::from(head.begin(&mut sequence));
    let state = Events { reply, head, output, queue, next: Some(first), sequence, over: false };
    let events = futures_util::stream::unfold(state, |mut state| async move {
        let (name, data) = state.step().await?;
        Some((Ok::<_, Infallible>(SseEvent::default().event(name).data(data)), state))
    });
    Ok(sse_response(events, KeepAlive::default()))
}

/// The state of a stream.
#[derive(Debug)]
struct Events {
    reply: Reply,
    head: Head,
    output: Output,
    /// Events that are ready and have not gone out.
    queue: VecDeque<Written>,
    /// A step of the reply that has come and is not written yet.
    next: Option<Step>,
    /// The number of the next event.
    sequence: u64,
    over: bool,
}

impl Events {
    /// The next event, or `None` when the stream is over. Dropping the
    /// stream half way drops the reply, which cancels the request in the
    /// engine.
    async fn step(&mut self) -> Option<Written> {
        loop {
            if let Some(event) = self.queue.pop_front() {
                return Some(event);
            }
            if self.over {
                return None;
            }
            let step = match self.next.take() {
                Some(step) => Ok(step),
                None => self.reply.next().await,
            };
            match step {
                Ok(Step::Pieces(pieces)) => {
                    let changes = self.output.push(pieces);
                    self.write(&changes);
                }
                Ok(Step::Probs(_)) => {}
                Ok(Step::End(End { rest, finish, usage, timings })) => {
                    let ending = Ending { finish, usage, timings };
                    let mut changes = self.output.push(rest);
                    changes.extend(self.output.finish(ending.cut()));
                    self.write(&changes);
                    let end = self.head.end(self.output.items(), &ending, &mut self.sequence);
                    self.queue.push_back(end);
                    self.over = true;
                }
                Err(error) => {
                    self.over = true;
                    return Some(error_event(&error, &mut self.sequence));
                }
            }
        }
    }

    fn write(&mut self, changes: &[reply::Change]) {
        for change in changes {
            change_events(&self.output, change, &mut self.sequence, &mut self.queue);
        }
    }
}
