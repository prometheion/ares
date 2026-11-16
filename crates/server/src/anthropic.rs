//! The Anthropic Messages API: `POST /v1/messages` and
//! `POST /v1/messages/count_tokens`.
//!
//! A Messages request is served as the chat request that asks the same, on the
//! same path through the engine: [`request`] writes its body over into a chat
//! body, the reader of chat bodies (`wire::from_json`) and the core of the
//! routes (`routes::prepare`, `routes::Reply`) do the rest, and [`reply`]
//! writes what comes back as content blocks and as Anthropic's events. A
//! reply is therefore the reply `/v1/chat/completions` gives, with the same
//! prefix cache, the same rule for the calls of tools and the same end when
//! the client goes away.
//!
//! # The request
//!
//! | field | here |
//! |---|---|
//! | `model`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream` | as on the chat route; `max_tokens` is required |
//! | `system` | a string or text blocks: one system message |
//! | `messages` | `user` and `assistant` turns, a string or blocks; see below |
//! | `tools` | `name`, `description`, `input_schema`: the tools of the chat request |
//! | `tool_choice` | `auto`, `any` (a call is required), `tool` (that tool), `none`; `disable_parallel_tool_use` allows one call |
//! | `thinking` | `enabled` and `adaptive` switch thinking on, `disabled` off; without it the server's setting holds |
//! | `stop_sequences` | the stop strings |
//! | `output_config` | `format` with a JSON schema is the `response_format` of the chat route, `effort` its `reasoning_effort` |
//! | `metadata`, `service_tier` | bookkeeping: accepted, without effect |
//! | `cache_control`, on the body and on blocks and tools | accepted, without effect: see below |
//!
//! A sampling field the request leaves out takes the default of its model,
//! also the fields this API has no member for, and `max_tokens` is held to
//! the most the model allows, as on the chat route (the `defaults` module).
//!
//! Blocks of a user turn: `text`, and `tool_result` (with a string or text
//! blocks), which becomes a message of the role `tool` where it stands. Blocks
//! of an assistant turn: `text`, `thinking`, `redacted_thinking` and
//! `tool_use`. The text blocks of one message are joined with a line break.
//!
//! Anything else is refused with a 400 that names it: `image` and `document`
//! blocks (input other than text is not built), tools of a `type` that
//! Anthropic's service defines or runs (`web_search_...`, `bash_...` and the
//! like) and the blocks they leave in a history, `defer_loading`,
//! `mcp_servers`, `container`, `context_management`, citations, a last
//! message of the assistant (a prefill, which asks to continue a message),
//! and every member this module does not know.
//!
//! # Decisions
//!
//! **A budget for thinking.** `thinking.budget_tokens` is checked and not
//! enforced. The engine has no budget for thinking (see the `reasoning`
//! module), and a client such as Claude Code sends one with every request that
//! thinks, so refusing it would refuse the client. What the request gets is
//! thinking that is on and as long as the model makes it; `max_tokens` bounds
//! the thinking and the answer together, as it does at Anthropic, where the
//! budget is a target and no hard limit either.
//!
//! **Signatures.** A thinking block of a reply carries the signature
//! [`reply::SIGNATURE`], a fixed text: the server has no secret to sign with.
//! A thinking block that comes back in a history is the reasoning of its
//! message whatever its signature says, also without one, and several blocks
//! of one message are joined with a line break. A reply's thinking that comes
//! back unchanged is written into the prompt as the model generated it, on
//! every turn the same, which is what lets the prefix cache continue the
//! conversation (see "A conversation that continues from the cache" in
//! `docs/ARCHITECTURE.md`). A `redacted_thinking` block holds text only
//! Anthropic's service can read: it is accepted and nothing of it enters the
//! prompt.
//!
//! **Cache markers.** The server keeps what it computed of every prompt and
//! starts the next one from the longest prefix it has. A `cache_control`
//! marker asks for nothing more than that, so it is accepted (its shape is
//! checked) and changes nothing; its `ttl` is not a promise the server makes,
//! since it drops what it kept by its own budget. The usage says how much of
//! the prompt came from the cache in `cache_read_input_tokens`, and
//! `input_tokens` is the rest, so the two add up to the prompt;
//! `cache_creation_input_tokens` is 0.
//!
//! **Why a reply ended.** `end_turn`, `max_tokens` or `tool_use`. The engine
//! has one word for a reply the model ended and one a stop string ended, so a
//! reply that a stop sequence cut says `end_turn` and its `stop_sequence` is
//! `null`.
//!
//! **The billing line of Claude Code.** Claude Code begins its system prompt
//! with a line of its own bookkeeping whose stamp (`cch=...`) is another one
//! in every request, which would part every prompt from the cached one at its
//! first line. The five characters of the stamp are overwritten, as
//! llama-server does; nothing else of a prompt is changed.
//!
//! **Where this is not llama-server.** llama-server concatenates system
//! blocks without anything between them, moves the text of a user turn before
//! its tool results, reads `tool_choice` `tool` as `any` and `none` not at
//! all, drops `thinking: disabled`, `is_error`, `disable_parallel_tool_use`
//! and every member it does not know, and passes images on. Here the blocks
//! keep their order, the choices are enforced, `is_error` puts `Error: `
//! before the text of the result, and what is not served is refused.
//!
//! # Headers
//!
//! `x-api-key`, `anthropic-version` and `anthropic-beta` are not looked at:
//! the server has no keys, one version of the API, and serves a field of a
//! beta or refuses it by its name in the body.

mod reply;
mod request;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;
use std::convert::Infallible;

use ares_api::RequestId;
use axum::extract::{Request, State};
use axum::response::sse::{Event as SseEvent, KeepAlive};
use axum::response::{IntoResponse, Response};

use self::reply::{Blocks, Content, Head, Written, block_event, message_end, stop_reason};
use self::request::Purpose;
use crate::ReasoningFormat;
use crate::error::{Anthropic, ApiError};
use crate::models::Lease;
use crate::reply::Sorter;
use crate::routes::{
    self, App, End, Reply, Step, json_response, next_request_id, read_body, sse_response,
    with_limit,
};
use crate::wire::{self, Endpoint, Parsed};

/// `POST /v1/messages`.
pub(crate) async fn messages(State(app): State<App>, request: Request) -> Response {
    match try_messages(app, request).await {
        Ok(response) => response,
        Err(error) => Anthropic(error).into_response(),
    }
}

/// `POST /v1/messages/count_tokens`.
pub(crate) async fn count_tokens(State(app): State<App>, request: Request) -> Response {
    match try_count(app, request).await {
        Ok(response) => response,
        Err(error) => Anthropic(error).into_response(),
    }
}

/// Another method on one of the two paths.
pub(crate) async fn method_not_allowed(request: Request) -> Response {
    let error = ApiError::method_not_allowed(request.method().as_str(), request.uri().path());
    Anthropic(error).into_response()
}

/// Reads the body of a request into the chat request it asks for and what
/// sorts its reply.
///
/// The reasoning is always taken out of the text: a thinking block is where
/// this API has it, whatever `--reasoning-format` says about
/// `reasoning_content`.
async fn read(
    app: &App,
    body: &[u8],
    id: RequestId,
    purpose: Purpose,
) -> Result<(Parsed, Sorter, Lease), ApiError> {
    let chat = request::to_chat(&wire::read_json(body)?, purpose)?;
    let mut parsed = wire::from_json(Endpoint::Chat, &chat, id).map_err(request::renamed)?;
    let (sorter, lease) = routes::prepare(app, &mut parsed, Some(ReasoningFormat::Auto))
        .await
        .map_err(request::renamed)?;
    Ok((parsed, sorter.with_call_ids(format!("toolu_{id}_")), lease))
}

async fn try_messages(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let id = next_request_id();
    let (parsed, sorter, lease) = read(&app, &body, id, Purpose::Reply).await?;
    let head = Head { id: format!("msg_{id}"), model: parsed.request.model.clone() };
    tracing::info!(
        %id,
        model = %parsed.request.model,
        endpoint = "Messages",
        stream = parsed.stream,
        max_tokens = ?parsed.request.max_tokens,
        "request"
    );
    let reply = Reply::start(&app, &lease, parsed.request, sorter)?;
    let response = if parsed.stream { stream(reply, head).await } else { whole(reply, head).await };
    Ok(with_limit(response?, parsed.lowered))
}

/// Counts the prompt of a request and generates nothing: the body is read
/// as `/v1/messages` reads it, and the engine counts the tokens of the prompt
/// it would have been given.
async fn try_count(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let (parsed, _, lease) = read(&app, &body, next_request_id(), Purpose::Count).await?;
    let tokens = routes::count_prompt(&lease, parsed.request).await?;
    Ok(json_response(format!(r#"{{"input_tokens":{tokens}}}"#)))
}

/// The answer that is not streamed: the events of the blocks, put together.
async fn whole(mut reply: Reply, head: Head) -> Result<Response, ApiError> {
    let mut blocks = Blocks::default();
    let mut content = Content::default();
    loop {
        match reply.next().await? {
            Step::Pieces(pieces) => content.take(blocks.push(pieces)),
            // This API has no field that asks for them.
            Step::Probs(_) => {}
            Step::End(End { rest, finish, stop, usage, .. }) => {
                content.take(blocks.push(rest));
                content.take(blocks.finish());
                let reason = stop_reason(finish, blocks.calls(), stop.as_deref());
                return Ok(json_response(head.message(&content, reason, stop.as_deref(), usage)));
            }
        }
    }
}

/// The streamed answer: `message_start`, the events of the blocks,
/// `message_delta` and `message_stop`.
///
/// As on the chat route the response starts when the engine has produced its
/// first step, so that a request the engine refuses gets an HTTP status. A
/// failure after that is an `error` event, and the stream ends without
/// `message_stop`.
async fn stream(mut reply: Reply, head: Head) -> Result<Response, ApiError> {
    let first = reply.next().await?;
    let state = Events {
        reply,
        blocks: Blocks::default(),
        queue: VecDeque::from([head.message_start()]),
        next: Some(first),
        over: false,
    };
    let events = futures_util::stream::unfold(state, |mut state| async move {
        let (name, data) = state.step().await?;
        Some((Ok::<_, Infallible>(SseEvent::default().event(name).data(data)), state))
    });
    let ping = SseEvent::default().event("ping").data(reply::PING);
    Ok(sse_response(events, KeepAlive::new().event(ping)))
}

/// The state of a stream.
#[derive(Debug)]
struct Events {
    reply: Reply,
    blocks: Blocks,
    /// Events that are ready and have not gone out.
    queue: VecDeque<Written>,
    /// A step of the reply that has come and is not written yet.
    next: Option<Step>,
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
                    let events = self.blocks.push(pieces);
                    self.queue.extend(events.iter().map(block_event));
                }
                Ok(Step::Probs(_)) => {}
                Ok(Step::End(End { rest, finish, stop, usage, .. })) => {
                    let mut events = self.blocks.push(rest);
                    events.extend(self.blocks.finish());
                    self.queue.extend(events.iter().map(block_event));
                    let reason = stop_reason(finish, self.blocks.calls(), stop.as_deref());
                    self.queue.extend(message_end(reason, stop.as_deref(), usage));
                    self.over = true;
                }
                Err(error) => {
                    self.over = true;
                    let data = serde_json::to_string(&error.anthropic()).expect("errors serialize");
                    return Some(("error", data));
                }
            }
        }
    }
}
