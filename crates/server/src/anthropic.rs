//! The Anthropic Messages API: `POST /v1/messages` and
//! `POST /v1/messages/count_tokens`.
//!
//! A Messages request is served as the chat request that asks the same, on the
//! same path through the engine: [`request`] writes its body over into a chat
//! body, the reader of chat bodies (`openai::request::from_json`) and the core of the
//! routes (`generate::prepare`, `generate::Generation`) do the rest, and [`reply`]
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
//! | `messages` | `user`, `assistant` and `system` turns, a string or blocks; see below |
//! | `tools` | `name`, `description`, `input_schema`: the tools of the chat request |
//! | `tool_choice` | `auto`, `any` (a call is required), `tool` (that tool), `none`; `disable_parallel_tool_use` allows one call |
//! | `thinking` | `enabled` and `adaptive` switch thinking on, `disabled` off; without it the server's setting holds; `display: "omitted"` gives the thinking blocks an empty text and a signature that carries it, `summarized` and `updates` the thinking whole |
//! | `stop_sequences` | the stop strings |
//! | `output_config` | `format` with a JSON schema is the `response_format` of the chat route, `effort` its `reasoning_effort` (`high`, Anthropic's default, is the template's default) |
//! | `context_management` | the edits `clear_thinking_20251015` and `clear_tool_uses_20250919`, carried out as Anthropic's service does (the [`edits`] module); the response lists what they cleared in `context_management.applied_edits` |
//! | `metadata`, `service_tier` | bookkeeping: accepted, without effect |
//! | `cache_control`, on the body and on blocks and tools | accepted, without effect: see below |
//!
//! A sampling field the request leaves out takes the default of its model,
//! also the fields this API has no member for, and `max_tokens` is held to
//! the most the model allows, as on the chat route (the `defaults` module).
//!
//! Blocks of a user turn: `text`, `image` (a `base64` source, which becomes an
//! `image_url` part of the chat body), and `tool_result` (with a string, or
//! text and image blocks), which becomes a message of the role `tool` where it
//! stands. Blocks
//! of an assistant turn: `text`, `thinking`, `redacted_thinking` and
//! `tool_use`. The text blocks of one message are joined with a line break.
//! A system turn (a string or text blocks) before every other turn joins the
//! system prompt; one between other turns is given to the chat template as a
//! user turn with its text (the rule of every route), since the templates of Qwen refuse a system
//! message anywhere but first (Claude Code sends its environment after the
//! first question that way, and the tokens it has left as the last turn).
//!
//! Anything else is refused with a 400 that names it: `document` blocks
//! (input other than text and images is not built), an image elsewhere or at
//! a URL (images are not fetched), tools of a `type` that
//! Anthropic's service defines or runs (`web_search_...`, `bash_...` and the
//! like) and the blocks they leave in a history, `defer_loading`,
//! `mcp_servers`, `container`, other edits of `context_management`
//! (compaction), citations, and every member this module does not know. A last message of the assistant is a
//! prefill, and the reply continues it (`continue_final_message` of the chat
//! body).
//!
//! # Decisions
//!
//! **A budget for thinking.** `thinking.budget_tokens` is the chat route's
//! `reasoning_budget_tokens` (vLLM maps it the same way): once the thinking
//! has taken that many tokens the engine ends it, as llama-server's sampler
//! does (see the `reasoning` module), and the answer follows. At Anthropic the
//! budget is a target the model may overshoot; here it is a bound, which is
//! what a client that sets one counts on. `max_tokens` bounds the thinking and
//! the answer together, as it does at Anthropic.
//!
//! **Signatures.** A thinking block of a reply carries the signature
//! [`reply::SIGNATURE`], a fixed text: the server has no secret to sign with.
//! With `thinking.display: "omitted"` the block's text is empty and its
//! signature is [`reply::CARRIED`] and the thinking in base64, which is what
//! Anthropic's API does (an empty `thinking` and a signature that carries
//! the thinking it does not show, so that the block sent back continues the
//! conversation): a block that comes back with an empty text and such a
//! signature is the reasoning of its turn, so the next prompt is the one a
//! client that shows the thinking sends, and the prefix cache continues.
//! A thinking block that comes back in a history is the reasoning of its
//! message whatever its signature says, also without one, and several blocks
//! of one message are joined with a line break. A reply's thinking that comes
//! back unchanged is written into the prompt as the model generated it, on
//! every turn the same, which is what lets the prefix cache continue the
//! conversation (see "A conversation that continues from the cache" in
//! `docs/dev/architecture/api.md`). A `redacted_thinking` block holds text only
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
//! **Why a reply ended.** `end_turn`, `max_tokens`, `tool_use`, or
//! `stop_sequence` with the string in `stop_sequence`: the engine names the
//! stop string that matched. A reply with calls ended to call them, also when
//! a stop string cut it.
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
//! `x-api-key` carries a key of the server (the `access` module looks at it,
//! as at `Authorization: Bearer`). `anthropic-version` and `anthropic-beta`
//! are not looked at: the server has one version of the API, and serves a
//! field of a beta or refuses it by its name in the body.

mod edits;
mod reply;
mod request;
#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use ares_api::RequestId;
use axum::extract::{Request, State};
use axum::http::HeaderValue;
use axum::response::sse::{Event as SseEvent, KeepAlive};
use axum::response::{IntoResponse, Response};

use self::reply::{BlockEvent, Blocks, Content, Head, block_event, stop_reason};
use crate::ReasoningFormat;
use crate::answer::{Writer, Written, answer};
use crate::chat_body::Purpose;
use crate::error::{Anthropic, ApiError};
use crate::generate::{self, End};
use crate::models::Lease;
use crate::openai::request::{Endpoint, PRIORITY_HEADER, Parsed, from_json, read_json};
use crate::routes::{App, json_response, next_request_id, read_body};
use crate::sorter::{Piece, Sorter};

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
    priority: Option<&HeaderValue>,
    id: RequestId,
    purpose: Purpose,
) -> Result<Read, ApiError> {
    let (chat, asked) = request::to_chat(&read_json(body)?, purpose)?;
    let mut parsed = from_json(Endpoint::Chat, &chat, id).map_err(request::renamed)?;
    parsed.take_priority(priority)?;
    let (sorter, lease) =
        generate::prepare(app, &mut parsed, Some(ReasoningFormat::Auto), "messages")
            .await
            .map_err(request::renamed)?;
    let edits = match &asked.edits {
        Some(edits) => Some(edits::apply(edits, &asked.turns, &mut parsed, &lease).await?),
        None => None,
    };
    let sorter = sorter.with_call_ids(format!("toolu_{id}_"));
    Ok(Read { parsed, sorter, lease, omit_thinking: asked.omit_thinking, edits })
}

/// A request as [`read`] reads it.
struct Read {
    parsed: Parsed,
    sorter: Sorter,
    lease: Lease,
    omit_thinking: bool,
    /// What the edits of `context_management` did, for a request that has
    /// the member.
    edits: Option<edits::Outcome>,
}

async fn try_messages(app: App, request: Request) -> Result<Response, ApiError> {
    let priority = request.headers().get(PRIORITY_HEADER).cloned();
    let body = read_body(request, &app.config).await?;
    let id = next_request_id();
    let Read { parsed, sorter, lease, omit_thinking, edits } =
        read(&app, &body, priority.as_ref(), id, Purpose::Reply).await?;
    let head = Head {
        id: format!("msg_{id}"),
        model: parsed.request.model.clone(),
        edits: edits.map(|outcome| outcome.applied),
    };
    let writer = Message { head, blocks: Blocks::new(omit_thinking), content: Content::default() };
    let ping = SseEvent::default().event("ping").data(reply::PING);
    answer(&app, &lease, parsed, sorter, writer, KeepAlive::new().event(ping)).await
}

/// Counts the prompt of a request and generates nothing: the body is read
/// as `/v1/messages` reads it, and the engine counts the tokens of the prompt
/// it would have been given.
async fn try_count(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let Read { parsed, lease, edits, .. } =
        read(&app, &body, None, next_request_id(), Purpose::Count).await?;
    let tokens = generate::count_prompt(&lease, parsed.request).await?;
    // With edits, the tokens before them too, as Anthropic counts them.
    Ok(json_response(match edits {
        Some(outcome) => format!(
            r#"{{"input_tokens":{tokens},"context_management":{{"original_input_tokens":{}}}}}"#,
            outcome.original_tokens.unwrap_or(tokens)
        ),
        None => format!(r#"{{"input_tokens":{tokens}}}"#),
    }))
}

/// A Messages reply as it is written: the events of its blocks, or the
/// message whole.
#[derive(Debug)]
struct Message {
    head: Head,
    blocks: Blocks,
    /// The blocks of a message that is answered whole.
    content: Content,
}

impl Message {
    /// The blocks the end of the reply closes, and its stop reason.
    fn close(
        &mut self,
        rest: Vec<Piece>,
        finish: &str,
        stop: Option<&str>,
    ) -> (Vec<BlockEvent>, &'static str) {
        let mut events = self.blocks.push(rest);
        events.extend(self.blocks.finish());
        (events, stop_reason(finish, self.blocks.calls(), stop))
    }
}

impl Writer for Message {
    fn begin(&mut self, out: &mut VecDeque<Written>) {
        out.push_back(self.head.message_start());
    }

    fn pieces(&mut self, pieces: Vec<Piece>, out: &mut VecDeque<Written>) {
        out.extend(self.blocks.push(pieces).iter().map(block_event));
    }

    fn end(&mut self, end: End, out: &mut VecDeque<Written>) {
        let End { rest, finish, stop, usage, .. } = end;
        let (events, reason) = self.close(rest, finish, stop.as_deref());
        out.extend(events.iter().map(block_event));
        out.extend(self.head.message_end(reason, stop.as_deref(), usage));
    }

    fn error(&mut self, error: &ApiError) -> Written {
        ("error", serde_json::to_string(&error.anthropic()).expect("errors serialize"))
    }

    fn take(&mut self, pieces: Vec<Piece>) {
        self.content.take(self.blocks.push(pieces));
    }

    fn whole(mut self, end: End) -> String {
        let End { rest, finish, stop, usage, .. } = end;
        let (events, reason) = self.close(rest, finish, stop.as_deref());
        self.content.take(events);
        self.head.message(&self.content, reason, stop.as_deref(), usage)
    }
}
