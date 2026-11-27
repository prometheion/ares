//! The body of a Messages request, written over into the body of the chat
//! request that asks the same.
//!
//! Everything that has Anthropic's shape is checked here, under Anthropic's
//! names, so that the reader of the chat body (`openai::request::from_json`) finds
//! nothing to refuse in what this module wrote; the fields that go by the same
//! name in both APIs (`model`, `max_tokens`, `temperature`, `top_p`, `top_k`,
//! `stream`) are handed over as they are and checked there.

use ares_api::Json;

use super::edits::{Edit, Keep, ToolUses, Trigger, Turns};
use crate::chat_body::{
    self, Chat, Purpose, cache_control, known, object, refuse, string, text_member,
};
use crate::error::ApiError;
use crate::openai::request::{Fields, members, wrong_type};
use crate::reasoning::DEFAULT_EFFORT;

/// The members of a body that are served.
const SERVED: &[&str] = &[
    "model",
    "max_tokens",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "stream",
    "output_config",
    "cache_control",
    "context_management",
    "sse_ping_interval",
];

/// Bookkeeping of the client or of Anthropic's service: accepted, and by
/// their own definition without effect on the reply.
const BOOKKEEPING: &[&str] = &["metadata", "service_tier"];

/// What the text blocks of one message are joined with.
const BETWEEN_BLOCKS: &str = "\n";

/// What comes before the text of a tool's result that says `is_error`.
const ERROR_PREFIX: &str = "Error";

/// What a Messages request asks for beyond its chat body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Asked {
    /// `thinking.display: "omitted"`: the thinking blocks of the reply have no
    /// text, and their signature carries it.
    pub omit_thinking: bool,
    /// The edits of `context_management`, in their order; `None` when the
    /// request has none.
    pub edits: Option<Vec<Edit>>,
    /// The turns of the conversation, for the edits.
    pub turns: Turns,
}

/// Writes the body of a Messages request over into the body of a chat
/// request.
pub(crate) fn to_chat(body: &Json, purpose: Purpose) -> Result<(Json, Asked), ApiError> {
    let fields = Fields(members(body)?);
    for (name, value) in fields.0 {
        let name = name.as_str();
        if value.is_null() || SERVED.contains(&name) {
            continue;
        }
        if BOOKKEEPING.contains(&name) {
            let expected = if name == "metadata" { "an object" } else { "a string" };
            match (name, value) {
                ("metadata", Json::Object(_)) | ("service_tier", Json::String(_)) => continue,
                _ => return Err(wrong_type(name, expected, value)),
            }
        }
        return Err(refuse(name, format!("{name} is not supported: {}", unserved(name))));
    }

    let mut chat = Chat::new();
    // llama.cpp's `sse_ping_interval` sets the interval of the stream's pings.
    for name in ["model", "temperature", "top_p", "top_k", "sse_ping_interval"] {
        if let Some(value) = fields.get(name) {
            chat.push((name, value.clone()));
        }
    }
    match fields.int("max_tokens")? {
        Some(limit) if limit >= 1 => chat.push(("max_tokens", Json::Int(limit))),
        Some(limit) => {
            return Err(refuse(
                "max_tokens",
                format!("max_tokens must be at least 1, got {limit}"),
            ));
        }
        None if purpose == Purpose::Reply => {
            return Err(refuse("max_tokens", "max_tokens is required"));
        }
        None => {}
    }
    // A count has no stream; the field is checked all the same.
    if let Some(stream) = fields.bool("stream")?
        && purpose == Purpose::Reply
    {
        chat.push(("stream", Json::Bool(stream)));
    }

    let mut system = match fields.get("system") {
        Some(system) => system_text(system)?,
        None => String::new(),
    };
    let edits = context_management(&fields)?;
    // A last message of the assistant is a prefill: the reply continues it.
    let (turns, prefill, counted) = conversation(&fields, &mut system)?;
    let mut messages = Vec::new();
    if !system.is_empty() {
        messages.push(object(vec![("role", string("system")), ("content", string(system))]));
    }
    messages.extend(turns);
    chat.push(("messages", Json::Array(messages)));
    if prefill {
        chat.push(("continue_final_message", Json::Bool(true)));
    }

    let names = tools(&fields, &mut chat)?;
    let callable = tool_choice(&fields, &names, &mut chat)?;
    let omit_thinking = thinking(&fields, &mut chat)?;
    if let Some(stop) = fields.get("stop_sequences") {
        chat.push(("stop", stop_sequences(stop)?));
    }
    output_config(&fields, callable, &mut chat)?;
    if let Some(marker) = fields.get("cache_control") {
        cache_control("cache_control", marker)?;
    }
    Ok((object(chat), Asked { omit_thinking, edits, turns: counted }))
}

/// Why a member of the body that is not served is not.
fn unserved(name: &str) -> &'static str {
    match name {
        "mcp_servers" | "container" => {
            "the server runs no tools and connects to no MCP server: the client runs the tools \
             and sends their results"
        }
        "output_format" => "structured output is asked for in output_config.format",
        "stop" => "the stop strings of this API are stop_sequences",
        _ => "it is not a parameter of this route",
    }
}

/// The members of a block (or of another object of the body), which may
/// have the members `known` and no others. `cache_control` is checked here
/// wherever it is known.
fn block<'a>(at: &str, value: &'a Json, names: &[&str]) -> Result<Fields<'a>, ApiError> {
    let fields = known(at, value, names)?;
    if let Some(marker) = fields.get("cache_control") {
        cache_control(&format!("{at}.cache_control"), marker)?;
    }
    Ok(fields)
}

/// The kind of a block, which every block names.
fn kind<'a>(at: &str, value: &'a Json) -> Result<&'a str, ApiError> {
    chat_body::kind(at, value)?.ok_or_else(|| refuse(at, format!("{at}.type must be a string")))
}

/// The text of a text block. Citations are not served: a block may say that
/// it has none.
fn text_block<'a>(at: &str, value: &'a Json) -> Result<&'a str, ApiError> {
    let fields = block(at, value, &["type", "text", "cache_control", "citations"])?;
    match fields.get("citations") {
        None => {}
        Some(Json::Array(none)) if none.is_empty() => {}
        Some(_) => return Err(refuse(at, format!("{at}.citations is not supported"))),
    }
    text_member(at, &fields, "text")
}

/// Why a block of a kind that is not served is refused.
fn refuse_block(at: &str, kind: &str) -> ApiError {
    let why = match kind {
        "document" => "documents cannot be read: input other than text and images is not built",
        "image" => "an image stands in a turn of the user or in the result of a tool",
        "server_tool_use"
        | "web_search_tool_result"
        | "web_fetch_tool_result"
        | "code_execution_tool_result"
        | "mcp_tool_use"
        | "mcp_tool_result"
        | "container_upload" => "the server runs no tools of its own",
        _ => "it is not a kind of block this server reads here",
    };
    refuse(at, format!("{at}: a block of the type '{kind}' is not supported: {why}"))
}

/// An image block as the part of a chat message the chat route reads
/// (`image_url` with a data URL): base64 data with its media type. An image
/// at a URL is not fetched.
fn image_part(at: &str, value: &Json) -> Result<Json, ApiError> {
    let fields = block(at, value, &["type", "source", "cache_control"])?;
    let at_source = format!("{at}.source");
    let source = fields
        .get("source")
        .ok_or_else(|| refuse(at_source.clone(), format!("{at_source} is required")))?;
    let url = match kind(&at_source, source)? {
        "base64" => {
            let source = block(&at_source, source, &["type", "media_type", "data"])?;
            let media = text_member(&at_source, &source, "media_type")?;
            if !media.starts_with("image/") {
                return Err(refuse(
                    at_source.clone(),
                    format!("{at_source}.media_type '{media}' is not an image"),
                ));
            }
            format!("data:{media};base64,{}", text_member(&at_source, &source, "data")?)
        }
        "url" => {
            return Err(refuse(
                at_source.clone(),
                format!(
                    "{at_source}: images are not fetched from URLs; send the image itself (a \
                     source of the type 'base64')"
                ),
            ));
        }
        other => {
            return Err(refuse(
                at_source.clone(),
                format!("{at_source}: a source of the type '{other}' is not read"),
            ));
        }
    };
    Ok(object(vec![
        ("type", string("image_url")),
        ("image_url", object(vec![("url", string(url))])),
    ]))
}

/// The parts of a message with images, as the chat route reads them: every
/// text block a text part, every image an `image_url` part, in their order.
/// The chat route joins them as llama-server does (a line break between two
/// texts, none next to an image).
fn text_part(text: &str) -> Json {
    object(vec![("type", string("text")), ("text", string(text))])
}

/// Appends `text` to the text of a message, after a line break if there is
/// text already.
fn join(whole: &mut Option<String>, text: &str) {
    chat_body::join(whole, BETWEEN_BLOCKS, text);
}

/// `system`: a string, or text blocks, which are joined with a line break.
fn system_text(value: &Json) -> Result<String, ApiError> {
    match value {
        Json::String(text) => Ok(steady_billing_header(text)),
        Json::Array(blocks) => {
            let mut whole = None;
            for (i, value) in blocks.iter().enumerate() {
                let at = format!("system[{i}]");
                match kind(&at, value)? {
                    "text" => join(&mut whole, &steady_billing_header(text_block(&at, value)?)),
                    other => return Err(refuse_block(&at, other)),
                }
            }
            Ok(whole.unwrap_or_default())
        }
        other => Err(wrong_type("system", "a string or an array of text blocks", other)),
    }
}

/// Claude Code begins its system prompt with a line of its own bookkeeping,
/// `x-anthropic-billing-header: cc_version=..; cc_entrypoint=..; cch=a5145;`,
/// whose `cch` is another one in every request. It says nothing to a model,
/// and left as it is no request of such a client would find a prefix of its
/// prompt in the cache: the prompts part at their first line. The five
/// bytes after `cch=` become `fffff` when a `;` follows them, exactly as
/// upstream llama-server does it (`normalize_anthropic_billing_header`,
/// llama.cpp PR 21793); a stamp of another form is left as it is.
fn steady_billing_header(text: &str) -> String {
    const HEADER: &str = "x-anthropic-billing-header:";
    const STAMP: usize = 5;
    let Some(at) = text
        .starts_with(HEADER)
        .then(|| text[HEADER.len()..].find("cch=").map(|at| HEADER.len() + at + 4))
        .flatten()
    else {
        return text.to_owned();
    };
    // The byte after the stamp is the ASCII `;`, so both ends of the stamp
    // lie between characters.
    if text.as_bytes().get(at + STAMP) != Some(&b';') {
        tracing::warn!("a billing header of Claude Code whose cch stamp is not of the known form");
        return text.to_owned();
    }
    format!("{}{}{}", &text[..at], "f".repeat(STAMP), &text[at + STAMP..])
}

/// `messages`: the turns of the user, of the assistant and of the system, as
/// chat messages, whether the last turn is the assistant's, which the reply
/// then continues, and the turns as the edits of `context_management` count
/// them (a turn of the user that is not only results of tools begins one).
///
/// A turn of the system before any other turn joins `system`, the system
/// prompt, as its blocks are joined. One between other turns (Claude Code
/// sends its environment and the tokens left that way) stays a system message
/// where it stands, which every route gives to a template that takes a
/// system message only first as a turn of the user
/// (`generate::place_system_messages`).
///
fn conversation(
    fields: &Fields<'_>,
    system: &mut String,
) -> Result<(Vec<Json>, bool, Turns), ApiError> {
    let mut chat = Vec::new();
    let turns = match fields.get("messages") {
        Some(Json::Array(turns)) if !turns.is_empty() => turns,
        Some(Json::Array(_)) => return Err(refuse("messages", "messages must not be empty")),
        Some(other) => return Err(wrong_type("messages", "an array", other)),
        None => return Err(refuse("messages", "messages is required")),
    };
    let mut counted = Turns::default();
    let mut begun = false;
    for (i, turn) in turns.iter().enumerate() {
        let at = format!("messages[{i}]");
        let turn = block(&at, turn, &["role", "content"])?;
        let content = turn
            .get("content")
            .ok_or_else(|| refuse(format!("{at}.content"), format!("{at}.content is required")))?;
        match turn.get("role") {
            Some(Json::String(role)) if role == "user" => {
                // Results of tools alone continue the turn in progress.
                let results_only = matches!(content, Json::Array(blocks)
                    if blocks.iter().all(|b| matches!(b.get("type"),
                        Some(Json::String(kind)) if kind == "tool_result")));
                if !results_only {
                    counted.current += usize::from(begun);
                    begun = true;
                }
                user(&at, content, &mut chat)?;
            }
            Some(Json::String(role)) if role == "assistant" => {
                counted.of_assistant.push(counted.current);
                chat.push(assistant(&at, content)?);
            }
            Some(Json::String(role)) if role == "system" => {
                let text = system_turn(&at, content)?;
                if chat.is_empty() {
                    if !system.is_empty() && !text.is_empty() {
                        system.push_str(BETWEEN_BLOCKS);
                    }
                    system.push_str(&text);
                } else {
                    chat.push(object(vec![("role", string("system")), ("content", string(text))]));
                }
            }
            _ => {
                return Err(refuse(
                    format!("{at}.role"),
                    format!("{at}.role must be \"user\", \"assistant\" or \"system\""),
                ));
            }
        }
    }
    let last = turns.last().and_then(|turn| turn.get("role"));
    let prefill = matches!(last, Some(Json::String(role)) if role == "assistant");
    Ok((chat, prefill, counted))
}

/// The text of a turn of the system: a string or text blocks, joined as the
/// blocks of `system` are.
fn system_turn(at: &str, content: &Json) -> Result<String, ApiError> {
    match content {
        Json::String(text) => Ok(text.clone()),
        Json::Array(blocks) => {
            let mut whole = None;
            for (j, value) in blocks.iter().enumerate() {
                let at = format!("{at}.content[{j}]");
                match kind(&at, value)? {
                    "text" => join(&mut whole, text_block(&at, value)?),
                    other => return Err(refuse_block(&at, other)),
                }
            }
            Ok(whole.unwrap_or_default())
        }
        other => Err(content_type(at, other)),
    }
}

/// `context_management`: the edits a client asks the server to make to the
/// conversation before the model reads it, which are carried out once the
/// model is known (the `edits` module), as Anthropic's service makes them.
///
/// `clear_thinking_20251015` takes the thinking out of the turns of the
/// assistant before the last `keep.value` turns, the one in progress counted
/// (`keep: "all"` keeps every one; without `keep`, one). Claude Code sends it
/// with `keep: "all"`, which changes nothing. It stands first when there are
/// several edits, as Anthropic asks.
///
/// `clear_tool_uses_20250919` replaces the results of the oldest calls of
/// tools with a note that they were cleared, once the prompt is past its
/// `trigger`: `{"type": "input_tokens" | "tool_uses", "value": n}`, 100 000
/// tokens without one; all but the last `keep.value` calls (`{"type":
/// "tool_uses", "value": n}`, 3 without one) and the calls of the tools in
/// `exclude_tools`; `clear_tool_inputs: true` clears their arguments too;
/// `clear_at_least` (`{"type": "input_tokens", "value": n}`) leaves the
/// conversation alone when clearing would take fewer tokens away.
fn context_management(fields: &Fields<'_>) -> Result<Option<Vec<Edit>>, ApiError> {
    let at = "context_management";
    let Some(value) = fields.get(at) else {
        return Ok(None);
    };
    let management = block(at, value, &["edits"])?;
    let edits = match management.get("edits") {
        None => return Ok(Some(Vec::new())),
        Some(Json::Array(edits)) => edits,
        Some(other) => return Err(wrong_type("context_management.edits", "an array", other)),
    };
    let mut read = Vec::new();
    for (i, edit) in edits.iter().enumerate() {
        let at = format!("context_management.edits[{i}]");
        match kind(&at, edit)? {
            "clear_thinking_20251015" => {
                if i > 0 {
                    return Err(refuse(
                        at.clone(),
                        format!(
                            "{at}: clear_thinking_20251015 must be the first of the edits, as at \
                             Anthropic"
                        ),
                    ));
                }
                let edit = block(&at, edit, &["type", "keep"])?;
                let keep = match edit.get("keep") {
                    None => Keep::Turns(1),
                    Some(Json::String(all)) if all == "all" => Keep::All,
                    Some(keep @ Json::Object(_)) => {
                        let n = amount(&format!("{at}.keep"), keep, "thinking_turns")?;
                        Keep::Turns(usize::try_from(n).unwrap_or(usize::MAX))
                    }
                    Some(_) => return Err(refuse_keep(&at)),
                };
                read.push(Edit::ClearThinking { keep });
            }
            "clear_tool_uses_20250919" => {
                let names = [
                    "type",
                    "trigger",
                    "keep",
                    "clear_at_least",
                    "exclude_tools",
                    "clear_tool_inputs",
                ];
                let edit = block(&at, edit, &names)?;
                let mut rule = ToolUses::default();
                if let Some(trigger) = edit.get("trigger") {
                    let at = format!("{at}.trigger");
                    rule.trigger = match kind(&at, trigger)? {
                        "input_tokens" => {
                            Trigger::InputTokens(amount(&at, trigger, "input_tokens")?)
                        }
                        "tool_uses" => Trigger::ToolUses(amount(&at, trigger, "tool_uses")?),
                        other => {
                            return Err(refuse(
                                at.clone(),
                                format!(
                                    "{at}.type must be \"input_tokens\" or \"tool_uses\", got \
                                     \"{other}\""
                                ),
                            ));
                        }
                    };
                }
                if let Some(keep) = edit.get("keep") {
                    let n = amount(&format!("{at}.keep"), keep, "tool_uses")?;
                    rule.keep = usize::try_from(n).unwrap_or(usize::MAX);
                }
                if let Some(least) = edit.get("clear_at_least") {
                    rule.clear_at_least =
                        Some(amount(&format!("{at}.clear_at_least"), least, "input_tokens")?);
                }
                match edit.get("exclude_tools") {
                    None => {}
                    Some(Json::Array(names)) => {
                        for (j, name) in names.iter().enumerate() {
                            match name {
                                Json::String(name) => rule.exclude.push(name.clone()),
                                other => {
                                    let at = format!("{at}.exclude_tools[{j}]");
                                    return Err(wrong_type(&at, "a string", other));
                                }
                            }
                        }
                    }
                    Some(other) => {
                        return Err(wrong_type(&format!("{at}.exclude_tools"), "an array", other));
                    }
                }
                if let Some(inputs) = edit.bool("clear_tool_inputs")? {
                    rule.clear_inputs = inputs;
                }
                read.push(Edit::ClearToolUses(rule));
            }
            other => {
                return Err(refuse(
                    at.clone(),
                    format!(
                        "{at}: an edit of the type '{other}' is not supported: the server clears \
                         thinking (clear_thinking_20251015) and the results of tools \
                         (clear_tool_uses_20250919)"
                    ),
                ));
            }
        }
    }
    Ok(Some(read))
}

/// `{"type": kind, "value": n}` with `n` a whole number, at least 1 for the
/// turns that keep their thinking.
fn amount(at: &str, value: &Json, kind: &str) -> Result<u32, ApiError> {
    let fields = block(at, value, &["type", "value"])?;
    let named = matches!(fields.get("type"), Some(Json::String(k)) if k == kind);
    let least = i64::from(kind == "thinking_turns");
    match fields.int("value") {
        Ok(Some(n)) if named && n >= least => Ok(u32::try_from(n).unwrap_or(u32::MAX)),
        _ if kind == "thinking_turns" => Err(refuse_keep(at.trim_end_matches(".keep"))),
        _ => Err(refuse(
            at,
            format!("{at} must be {{\"type\": \"{kind}\", \"value\": n}} with n a whole number"),
        )),
    }
}

fn refuse_keep(at: &str) -> ApiError {
    refuse(
        at,
        format!(
            "{at}.keep must be \"all\" or {{\"type\": \"thinking_turns\", \"value\": n}} with \
             n at least 1"
        ),
    )
}

fn content_type(at: &str, other: &Json) -> ApiError {
    refuse(
        format!("{at}.content"),
        format!("{at}.content must be a string or an array of blocks, got {}", other.kind()),
    )
}

/// A turn of the user: its text blocks are one user message, and every
/// result of a tool is a message of the role `tool`, in the order the blocks
/// stand in.
fn user(at: &str, content: &Json, chat: &mut Vec<Json>) -> Result<(), ApiError> {
    let message = |text: String| object(vec![("role", string("user")), ("content", string(text))]);
    let blocks = match content {
        Json::String(text) => {
            chat.push(message(text.clone()));
            return Ok(());
        }
        Json::Array(blocks) if !blocks.is_empty() => blocks,
        Json::Array(_) => {
            return Err(refuse(format!("{at}.content"), format!("{at}.content must not be empty")));
        }
        other => return Err(content_type(at, other)),
    };
    // The user's blocks up to a result of a tool: their text, and their parts
    // once an image is among them.
    let mut text: Option<String> = None;
    let mut parts: Vec<Json> = Vec::new();
    let flush = |text: &mut Option<String>, parts: &mut Vec<Json>, chat: &mut Vec<Json>| {
        if parts.is_empty() {
            chat.extend(text.take().map(message));
        } else {
            chat.push(object(vec![
                ("role", string("user")),
                ("content", Json::Array(std::mem::take(parts))),
            ]));
            *text = None;
        }
    };
    for (j, value) in blocks.iter().enumerate() {
        let at = format!("{at}.content[{j}]");
        match kind(&at, value)? {
            "text" => {
                let piece = text_block(&at, value)?;
                if parts.is_empty() {
                    join(&mut text, piece);
                } else {
                    parts.push(text_part(piece));
                }
            }
            "image" => {
                if parts.is_empty()
                    && let Some(before) = text.take()
                {
                    parts.push(text_part(&before));
                }
                parts.push(image_part(&at, value)?);
            }
            "tool_result" => {
                flush(&mut text, &mut parts, chat);
                chat.push(tool_result(&at, value)?);
            }
            other => return Err(refuse_block(&at, other)),
        }
    }
    flush(&mut text, &mut parts, chat);
    Ok(())
}

/// The result of a tool as a message of the role `tool`. A result that says
/// `is_error` has `Error: ` before its text: the chat template has no other
/// place for it.
fn tool_result(at: &str, value: &Json) -> Result<Json, ApiError> {
    let known = ["type", "tool_use_id", "content", "is_error", "cache_control"];
    let fields = block(at, value, &known)?;
    let id = text_member(at, &fields, "tool_use_id")?;
    // A result with images is parts, as a user's turn with images.
    let mut parts: Vec<Json> = Vec::new();
    let mut text = match fields.get("content") {
        None => String::new(),
        Some(Json::String(text)) => text.clone(),
        Some(Json::Array(blocks)) => {
            let mut whole = None;
            for (k, value) in blocks.iter().enumerate() {
                let at = format!("{at}.content[{k}]");
                match kind(&at, value)? {
                    "text" if parts.is_empty() => join(&mut whole, text_block(&at, value)?),
                    "text" => parts.push(text_part(text_block(&at, value)?)),
                    "image" => {
                        if parts.is_empty()
                            && let Some(before) = whole.take()
                        {
                            parts.push(text_part(&before));
                        }
                        parts.push(image_part(&at, value)?);
                    }
                    other => return Err(refuse_block(&at, other)),
                }
            }
            whole.unwrap_or_default()
        }
        Some(other) => return Err(content_type(at, other)),
    };
    match fields.get("is_error") {
        None | Some(Json::Bool(false)) => {}
        Some(Json::Bool(true)) if text.is_empty() => text = ERROR_PREFIX.to_owned(),
        Some(Json::Bool(true)) => text = format!("{ERROR_PREFIX}: {text}"),
        Some(other) => {
            return Err(refuse(
                at,
                format!("{at}.is_error must be a boolean, got {}", other.kind()),
            ));
        }
    }
    let content = if parts.is_empty() {
        string(text)
    } else {
        // The mark of an error goes before the parts.
        if !text.is_empty() {
            parts.insert(0, text_part(&text));
        }
        Json::Array(parts)
    };
    Ok(object(vec![("role", string("tool")), ("tool_call_id", string(id)), ("content", content)]))
}

/// A turn of the assistant: text, thinking and the calls of tools.
///
/// The thinking goes back into the prompt as the reasoning of the message,
/// whatever its signature says: there is nothing to check it against. A
/// block whose thinking was omitted has its reasoning in the signature
/// (`thinking.display`), which is read instead of the empty text. A
/// `redacted_thinking` block holds what only Anthropic's service can read, so
/// nothing of it enters the prompt.
fn assistant(at: &str, content: &Json) -> Result<Json, ApiError> {
    let mut text: Option<String> = None;
    let mut reasoning: Option<String> = None;
    let mut calls = Vec::new();
    match content {
        Json::String(whole) => text = Some(whole.clone()),
        Json::Array(blocks) => {
            for (j, value) in blocks.iter().enumerate() {
                let at = format!("{at}.content[{j}]");
                match kind(&at, value)? {
                    "text" => join(&mut text, text_block(&at, value)?),
                    "thinking" => {
                        let fields = block(&at, value, &["type", "thinking", "signature"])?;
                        let signature = match fields.get("signature") {
                            None => "",
                            Some(Json::String(signature)) => signature.as_str(),
                            Some(other) => {
                                return Err(refuse(
                                    at.clone(),
                                    format!(
                                        "{at}.signature must be a string, got {}",
                                        other.kind()
                                    ),
                                ));
                            }
                        };
                        let text = text_member(&at, &fields, "thinking")?;
                        match signature.strip_prefix(super::reply::CARRIED) {
                            // A block whose thinking was omitted: the signature
                            // carries it.
                            Some(carried) if text.is_empty() => {
                                let reasoning_of = crate::base64::decode(carried)
                                    .and_then(|bytes| String::from_utf8(bytes).ok());
                                let Some(carried) = reasoning_of else {
                                    return Err(refuse(
                                        format!("{at}.signature"),
                                        format!(
                                            "{at}.signature begins as the signature of an \
                                             omitted thinking of this server, and the rest is \
                                             not base64 of UTF-8: send the block as it came"
                                        ),
                                    ));
                                };
                                join(&mut reasoning, &carried);
                            }
                            _ => join(&mut reasoning, text),
                        }
                    }
                    "redacted_thinking" => {
                        let fields = block(&at, value, &["type", "data"])?;
                        text_member(&at, &fields, "data")?;
                    }
                    "tool_use" => calls.push(tool_use(&at, value)?),
                    other => return Err(refuse_block(&at, other)),
                }
            }
        }
        other => return Err(content_type(at, other)),
    }
    let mut message =
        vec![("role", string("assistant")), ("content", string(text.unwrap_or_default()))];
    if let Some(reasoning) = reasoning {
        message.push(("reasoning_content", string(reasoning)));
    }
    if !calls.is_empty() {
        message.push(("tool_calls", Json::Array(calls)));
    }
    Ok(object(message))
}

/// A call of a tool in a turn of the assistant, as a call of a chat message.
fn tool_use(at: &str, value: &Json) -> Result<Json, ApiError> {
    let fields = block(at, value, &["type", "id", "name", "input", "cache_control"])?;
    let id = text_member(at, &fields, "id")?;
    let name = text_member(at, &fields, "name")?;
    if name.is_empty() {
        return Err(refuse(at, format!("{at}.name must not be empty")));
    }
    let input = match fields.get("input") {
        Some(input @ Json::Object(_)) => input.clone(),
        _ => return Err(refuse(at, format!("{at}.input must be an object"))),
    };
    let function = object(vec![("name", string(name)), ("arguments", input)]);
    Ok(object(vec![("id", string(id)), ("type", string("function")), ("function", function)]))
}

/// `tools`, as the tools of a chat request. Returns their names.
fn tools<'a>(fields: &Fields<'a>, chat: &mut Chat) -> Result<Vec<&'a str>, ApiError> {
    let tools = match fields.get("tools") {
        None => return Ok(Vec::new()),
        Some(Json::Array(tools)) => tools,
        Some(other) => return Err(wrong_type("tools", "an array", other)),
    };
    let mut names = Vec::new();
    let mut functions = Vec::new();
    for (i, value) in tools.iter().enumerate() {
        let at = format!("tools[{i}]");
        let refuse = |message: String| refuse("tools", message);
        match value.get("type") {
            None | Some(Json::Null) => {}
            Some(Json::String(kind)) if kind == "custom" => {}
            Some(Json::String(kind)) => {
                return Err(refuse(format!(
                    "{at}: a tool of the type '{kind}' is not supported: it is run or defined \
                     by Anthropic's service, and this server takes tools with a name and an \
                     input_schema that the client runs"
                )));
            }
            Some(other) => {
                return Err(refuse(format!("{at}.type must be a string, got {}", other.kind())));
            }
        }
        let known = [
            "type",
            "name",
            "description",
            "input_schema",
            "cache_control",
            "strict",
            "eager_input_streaming",
            "defer_loading",
        ];
        let tool = block(&at, value, &known).map_err(|e| refuse(e.message))?;
        // The arguments are always held to the schema, and their text is
        // streamed as the model writes it.
        for name in ["strict", "eager_input_streaming", "defer_loading"] {
            tool.bool(name).map_err(|_| refuse(format!("{at}.{name} must be a boolean")))?;
        }
        if tool.bool("defer_loading").ok().flatten() == Some(true) {
            return Err(refuse(format!(
                "{at}.defer_loading is not supported: every tool of a request is written into \
                 its prompt"
            )));
        }
        let name = match tool.get("name") {
            Some(Json::String(name)) if !name.is_empty() => name.as_str(),
            _ => return Err(refuse(format!("{at}.name must be a string that is not empty"))),
        };
        let mut function = vec![("name", string(name))];
        match tool.get("description") {
            None => {}
            Some(description @ Json::String(_)) => {
                function.push(("description", description.clone()))
            }
            Some(_) => return Err(refuse(format!("{at}.description must be a string"))),
        }
        match tool.get("input_schema") {
            Some(schema @ Json::Object(_)) => function.push(("parameters", schema.clone())),
            _ => return Err(refuse(format!("{at}.input_schema must be an object"))),
        }
        names.push(name);
        functions.push(object(vec![("type", string("function")), ("function", object(function))]));
    }
    // An empty list is no tools: the chat template is not to open a section
    // for them.
    if !functions.is_empty() {
        chat.push(("tools", Json::Array(functions)));
    }
    Ok(names)
}

/// `tool_choice`, as the one of a chat request. Returns whether the reply
/// may call a tool.
fn tool_choice(fields: &Fields<'_>, tools: &[&str], chat: &mut Chat) -> Result<bool, ApiError> {
    let at = "tool_choice";
    let Some(value) = fields.get(at) else {
        return Ok(!tools.is_empty());
    };
    let kind = kind(at, value)?;
    let known: &[&str] = match kind {
        "auto" | "any" => &["type", "disable_parallel_tool_use"],
        "tool" => &["type", "name", "disable_parallel_tool_use"],
        "none" => &["type"],
        other => {
            return Err(refuse(
                at,
                format!(
                    "tool_choice.type must be \"auto\", \"any\", \"tool\" or \"none\", got \
                     \"{other}\""
                ),
            ));
        }
    };
    let choice = block(at, value, known)?;
    if matches!(kind, "any" | "tool") && tools.is_empty() {
        return Err(refuse(at, format!("tool_choice is \"{kind}\" and the request has no tools")));
    }
    let chosen = match kind {
        "any" => string("required"),
        "tool" => {
            let name = text_member(at, &choice, "name")?;
            if !tools.contains(&name) {
                return Err(refuse(
                    at,
                    format!("tool_choice names the tool '{name}', which is not among the tools"),
                ));
            }
            let function = object(vec![("name", string(name))]);
            object(vec![("type", string("function")), ("function", function)])
        }
        other => string(other),
    };
    chat.push(("tool_choice", chosen));
    match choice.get("disable_parallel_tool_use") {
        None => {}
        Some(Json::Bool(one)) => chat.push(("parallel_tool_calls", Json::Bool(!one))),
        Some(other) => {
            return Err(refuse(
                at,
                format!(
                    "tool_choice.disable_parallel_tool_use must be a boolean, got {}",
                    other.kind()
                ),
            ));
        }
    }
    Ok(kind != "none" && !tools.is_empty())
}

/// `thinking`, as the variable of the chat template that switches it.
///
/// `enabled` and `adaptive` switch thinking on and `disabled` off; a request
/// that does not say leaves it to the server's setting. `budget_tokens` is
/// the budget of thinking tokens of the chat route (`reasoning_budget_tokens`,
/// as vLLM maps it): the thinking ends there and the answer follows;
/// `max_tokens` bounds the thinking and the answer together, as it does at
/// Anthropic.
fn thinking(fields: &Fields<'_>, chat: &mut Chat) -> Result<bool, ApiError> {
    let at = "thinking";
    let Some(value) = fields.get(at) else {
        return Ok(false);
    };
    let kind = kind(at, value)?;
    let (on, known): (bool, &[&str]) = match kind {
        "enabled" => (true, &["type", "budget_tokens", "display"]),
        "adaptive" => (true, &["type", "display"]),
        "disabled" => (false, &["type"]),
        other => {
            return Err(refuse(
                at,
                format!(
                    "thinking.type must be \"enabled\", \"adaptive\" or \"disabled\", got \
                     \"{other}\""
                ),
            ));
        }
    };
    let asked = block(at, value, known)?;
    let omit = match asked.get("display") {
        None => false,
        Some(Json::String(display)) if DISPLAYS.contains(&display.as_str()) => display == "omitted",
        Some(_) => {
            return Err(refuse(
                at,
                "thinking.display must be \"summarized\", \"omitted\" or \"updates\"",
            ));
        }
    };
    match asked.int("budget_tokens") {
        Ok(None) => {}
        Ok(Some(budget)) if budget >= 1 => {
            chat.push(("reasoning_budget_tokens", Json::Int(budget)))
        }
        _ => {
            return Err(refuse(at, "thinking.budget_tokens must be a whole number of at least 1"));
        }
    }
    chat.push(("chat_template_kwargs", object(vec![("enable_thinking", Json::Bool(on))])));
    Ok(omit)
}

/// The values of `thinking.display`.
///
/// `omitted` is what Anthropic's API does: a thinking block whose `thinking`
/// is the empty text, and a signature that carries the thinking, so that the
/// block sent back continues the conversation as the model generated it. Here
/// the signature carries the reasoning in base64 after
/// [`super::reply::CARRIED`], and a block that comes back with an empty text
/// and such a signature is the reasoning of its message (see `assistant`):
/// the next prompt is the one a client that shows the thinking sends, and the
/// prefix cache continues the same way.
///
/// `summarized` and `updates` get the thinking whole: the server has no
/// summary of it and no notes of progress to send, and the thinking itself
/// shows more than a summary, never less.
const DISPLAYS: &[&str] = &["summarized", "omitted", "updates"];

/// `stop_sequences`, as the stop strings of a chat request.
fn stop_sequences(value: &Json) -> Result<Json, ApiError> {
    let at = "stop_sequences";
    let Json::Array(sequences) = value else {
        return Err(wrong_type(at, "an array of strings", value));
    };
    for sequence in sequences {
        match sequence {
            Json::String(text) if !text.is_empty() => {}
            Json::String(_) => {
                return Err(refuse(at, "stop_sequences must not hold an empty string"));
            }
            other => {
                return Err(refuse(
                    at,
                    format!("stop_sequences must hold strings, found {}", other.kind()),
                ));
            }
        }
    }
    Ok(value.clone())
}

/// `output_config`: `format` is a JSON schema the reply follows (the
/// `response_format` of a chat request), `effort` is the `reasoning_effort`
/// of one. `high` is Anthropic's default level, so a request that names it
/// gets the default of the chat template (the most for the templates of
/// Qwen3.8, one of which does not know the name `high`).
fn output_config(fields: &Fields<'_>, callable: bool, chat: &mut Chat) -> Result<(), ApiError> {
    let at = "output_config";
    let Some(value) = fields.get(at) else {
        return Ok(());
    };
    let config = block(at, value, &["format", "effort"])?;
    if let Some(format) = config.get("format") {
        let at = "output_config.format";
        let format = block(at, format, &["type", "schema"])?;
        let schema = match (format.get("type"), format.get("schema")) {
            (Some(Json::String(kind)), Some(schema @ Json::Object(_))) if kind == "json_schema" => {
                schema.clone()
            }
            _ => {
                return Err(refuse(
                    at,
                    "output_config.format must be {\"type\": \"json_schema\", \"schema\": {...}}",
                ));
            }
        };
        if callable {
            return Err(refuse(
                at,
                "output_config.format cannot be combined with tools that may be called: the \
                 reply follows one rule. Send tool_choice {\"type\": \"none\"} to have the tools \
                 in the prompt and the reply follow the schema",
            ));
        }
        let wrapper = object(vec![("schema", schema)]);
        chat.push((
            "response_format",
            object(vec![("type", string("json_schema")), ("json_schema", wrapper)]),
        ));
    }
    match config.get("effort") {
        None => {}
        // Anthropic's default level, which a request at Anthropic gets when
        // it names none: the chat template's default, whatever the template
        // knows (the chat route takes it so only where the template refuses
        // it).
        Some(Json::String(effort)) if effort == DEFAULT_EFFORT => {}
        Some(Json::String(effort)) if !effort.is_empty() => {
            chat.push(("reasoning_effort", string(effort.as_str())));
        }
        Some(_) => {
            return Err(refuse(at, "output_config.effort must be a string that is not empty"));
        }
    }
    Ok(())
}

/// An error of the reader of the chat body, under the name the field has in
/// this API.
pub(crate) fn renamed(error: ApiError) -> ApiError {
    chat_body::renamed(
        error,
        &[
            ("reasoning_effort", "output_config.effort"),
            ("response_format", "output_config.format"),
        ],
    )
}
