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
//! | `truncation` | `disabled` (the default), or `auto`: the oldest turns of a conversation that does not fit leave it |
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
//! that is not the first one. Such a message inside a conversation is a
//! message of the user with its text, as on every route
//! (`wire::place_system_messages`). The items one reply returned for a turn (reasoning,
//! message, calls) are one message of the assistant again.
//!
//! Anything else is refused with a 400 that names it: `input_image`,
//! `input_file` and `input_audio` parts (input other than text is not
//! built), tools that are not functions (`web_search`, `file_search`,
//! `code_interpreter`, `computer_use`, `mcp`, `local_shell`, `custom` and
//! the like) and the items they leave in a history, `prompt` templates,
//! `top_logprobs`, `text.verbosity`, `reasoning.summary` of a given kind,
//! the sampler's fields of the chat route (only
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
//! **What was applied.** A response says back `temperature`, `top_p` and
//! `max_output_tokens`. Here they are what the request ran with: its own
//! value, or the model's default where it left the member out (`null` where
//! the model has none and the engine's own value held), and for
//! `max_output_tokens` the limit the reply has, which is the model's when
//! the request named none or more than the model allows (the `defaults`
//! module).
//!
//! **`truncation`.** With `disabled` an input that does not fit the context
//! is an error (`context_length_exceeded`). With `auto` the oldest turns of
//! the conversation leave it, whole, until it fits ([`truncate`]): what
//! OpenAI describes as dropping items of the conversation to fit.
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

use ares_api::{Prompt, RequestId};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::KeepAlive;
use axum::response::{IntoResponse, Response};

use self::reply::{Ending, Head, Output, change_events, error_event};
use self::request::Echo;
use crate::ReasoningFormat;
use crate::answer::{Writer, Written, answer};
use crate::chat_body::Purpose;
use crate::error::ApiError;
use crate::models::Lease;
use crate::reply::{Piece, Sorter};
use crate::routes::{self, App, End, json_response, next_request_id, read_body};
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
    priority: Option<&HeaderValue>,
    id: RequestId,
    purpose: Purpose,
) -> Result<(Parsed, Sorter, Echo, Lease), ApiError> {
    let (chat, mut echo) = request::to_chat(&wire::read_json(body)?, purpose)?;
    let mut parsed = wire::from_json(Endpoint::Chat, &chat, id).map_err(request::renamed)?;
    parsed.take_priority(priority)?;
    let (sorter, lease) =
        routes::prepare(app, &mut parsed, Some(ReasoningFormat::Auto), "responses")
            .await
            .map_err(request::renamed)?;
    echo.applied(&parsed.request);
    if echo.truncation == "auto" {
        truncate(&mut parsed, &lease).await?;
    }
    Ok((parsed, sorter, echo, lease))
}

/// `truncation: "auto"`: a conversation whose prompt does not fit the context
/// of the model (the prompt and a token of the reply) loses its oldest turns,
/// whole, until it does: a turn is a message of the user and everything after
/// it up to the next one (the assistant's answer, its calls and their
/// results), so that no result is left without its call. The system messages
/// before the conversation and the last turn stay. A conversation that does
/// not fit with its last turn alone is not cut and gets the error it gets
/// without truncation. The number of turns to drop is found by halving: the
/// prompt is counted a few times (a rendering and a tokenization each).
async fn truncate(parsed: &mut Parsed, lease: &Lease) -> Result<(), ApiError> {
    let id = parsed.request.model.clone();
    let Some(card) = lease.engine().models().into_iter().find(|card| card.id == id) else {
        return Ok(());
    };
    let fits = |tokens: u32| tokens < card.context_length;
    let count = |dropped: usize, parsed: &Parsed| {
        let mut request = parsed.request.clone();
        if let Prompt::Chat { messages, .. } = &mut request.prompt {
            drop_turns(messages, dropped);
        }
        routes::count_prompt(lease, request)
    };
    let turns = match &parsed.request.prompt {
        Prompt::Chat { messages, .. } => starts(messages).len(),
        Prompt::Text(_) => return Ok(()),
    };
    if turns < 2 || fits(count(0, parsed).await?) {
        return Ok(());
    }
    // The fewest turns to drop that fit, between 1 and all but the last.
    let (mut low, mut high) = (1, turns - 1);
    if !fits(count(high, parsed).await?) {
        return Ok(());
    }
    while low < high {
        let middle = (low + high) / 2;
        if fits(count(middle, parsed).await?) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    tracing::info!(id = %parsed.request.id, dropped = low, of = turns, "truncation dropped turns");
    if let Prompt::Chat { messages, .. } = &mut parsed.request.prompt {
        drop_turns(messages, low);
    }
    Ok(())
}

/// Where the turns of a conversation begin: at every message of the user.
fn starts(messages: &[ares_api::Message]) -> Vec<usize> {
    let users = messages.iter().enumerate().filter(|(_, m)| m.role == ares_api::Role::User);
    users.map(|(i, _)| i).collect()
}

/// Takes the first `n` turns out of `messages`, and with them what stands
/// between the system messages and the first turn.
fn drop_turns(messages: &mut Vec<ares_api::Message>, n: usize) {
    let starts = starts(messages);
    if n == 0 || n >= starts.len() {
        return;
    }
    let first = messages.iter().take_while(|m| m.role == ares_api::Role::System).count();
    messages.drain(first..starts[n]);
}

async fn try_create(app: App, request: Request) -> Result<Response, ApiError> {
    let priority = request.headers().get(wire::PRIORITY_HEADER).cloned();
    let body = read_body(request, &app.config).await?;
    let id = next_request_id();
    let (parsed, sorter, echo, lease) =
        read(&app, &body, priority.as_ref(), id, Purpose::Reply).await?;
    let head = Head {
        id: format!("resp_{id}"),
        model: parsed.request.model.clone(),
        created_at: wire::unix_time(),
        echo,
    };
    let writer = Created { head, output: Output::new(id.to_string()), sequence: 0 };
    answer(&app, &lease, parsed, sorter, writer, KeepAlive::default()).await
}

/// Counts the prompt of a request and generates nothing: the body is read
/// as `/v1/responses` reads it.
async fn try_count(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let (parsed, _, _, lease) = read(&app, &body, None, next_request_id(), Purpose::Count).await?;
    let tokens = routes::count_prompt(&lease, parsed.request).await?;
    Ok(json_response(wire::input_tokens(tokens)))
}

/// A Responses reply as it is written: the events of its items, or the
/// response whole.
#[derive(Debug)]
struct Created {
    head: Head,
    output: Output,
    /// The number of the next event.
    sequence: u64,
}

impl Created {
    /// Takes the end of the reply into the output: what the sorter still
    /// held, and what the end closes.
    fn close(&mut self, end: End) -> (Ending, Vec<reply::Change>) {
        let End { rest, finish, usage, timings, .. } = end;
        let ending = Ending { finish, usage, timings };
        let mut changes = self.output.push(rest);
        changes.extend(self.output.finish(ending.cut()));
        (ending, changes)
    }

    fn write(&mut self, changes: &[reply::Change], out: &mut VecDeque<Written>) {
        for change in changes {
            change_events(&self.output, change, &mut self.sequence, out);
        }
    }
}

impl Writer for Created {
    fn begin(&mut self, out: &mut VecDeque<Written>) {
        out.extend(self.head.begin(&mut self.sequence));
    }

    fn pieces(&mut self, pieces: Vec<Piece>, out: &mut VecDeque<Written>) {
        let changes = self.output.push(pieces);
        self.write(&changes, out);
    }

    fn end(&mut self, end: End, out: &mut VecDeque<Written>) {
        let (ending, changes) = self.close(end);
        self.write(&changes, out);
        out.push_back(self.head.end(self.output.items(), &ending, &mut self.sequence));
    }

    fn error(&mut self, error: &ApiError) -> Written {
        error_event(error, &mut self.sequence)
    }

    fn take(&mut self, pieces: Vec<Piece>) {
        self.output.push(pieces);
    }

    fn whole(mut self, end: End) -> String {
        let (ending, _) = self.close(end);
        self.head.response(self.output.items(), &ending)
    }
}
