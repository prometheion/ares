//! The body of a Messages request, written over into the body of the chat
//! request that asks the same.
//!
//! Everything that has Anthropic's shape is checked here, under Anthropic's
//! names, so that the reader of the chat body (`wire::from_json`) finds
//! nothing to refuse in what this module wrote; the fields that go by the same
//! name in both APIs (`model`, `max_tokens`, `temperature`, `top_p`, `top_k`,
//! `stream`) are handed over as they are and checked there.

use ares_api::Json;

use crate::error::ApiError;
use crate::wire::{Fields, members, wrong_type};

/// What a body is read for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Purpose {
    /// `/v1/messages`: a reply is generated.
    Reply,
    /// `/v1/messages/count_tokens`: the prompt is counted, so `max_tokens`
    /// may be missing and `stream` says nothing.
    Count,
}

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
];

/// Bookkeeping of the client or of Anthropic's service: accepted, and by
/// their own definition without effect on the reply.
const BOOKKEEPING: &[&str] = &["metadata", "service_tier"];

/// What the text blocks of one message are joined with.
const BETWEEN_BLOCKS: &str = "\n";

/// What comes before the text of a tool's result that says `is_error`.
const ERROR_PREFIX: &str = "Error";

fn refuse(param: impl Into<String>, message: impl Into<String>) -> ApiError {
    ApiError::invalid_param(param, message)
}

fn object(members: Vec<(&str, Json)>) -> Json {
    Json::Object(members.into_iter().map(|(name, value)| (name.to_owned(), value)).collect())
}

/// The members of the chat body as they are written.
type Chat = Vec<(&'static str, Json)>;

fn string(text: impl Into<String>) -> Json {
    Json::String(text.into())
}

/// Writes the body of a Messages request over into the body of a chat
/// request.
pub(crate) fn to_chat(body: &Json, purpose: Purpose) -> Result<Json, ApiError> {
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
    for name in ["model", "temperature", "top_p", "top_k"] {
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

    let mut messages = Vec::new();
    if let Some(system) = fields.get("system") {
        let text = system_text(system)?;
        if !text.is_empty() {
            messages.push(object(vec![("role", string("system")), ("content", string(text))]));
        }
    }
    // A last message of the assistant is a prefill: the reply continues it.
    let prefill = conversation(&fields, &mut messages)?;
    chat.push(("messages", Json::Array(messages)));
    if prefill {
        chat.push(("continue_final_message", Json::Bool(true)));
    }

    let names = tools(&fields, &mut chat)?;
    let callable = tool_choice(&fields, &names, &mut chat)?;
    thinking(&fields, &mut chat)?;
    if let Some(stop) = fields.get("stop_sequences") {
        chat.push(("stop", stop_sequences(stop)?));
    }
    output_config(&fields, callable, &mut chat)?;
    if let Some(marker) = fields.get("cache_control") {
        cache_control("cache_control", marker)?;
    }
    Ok(object(chat))
}

/// Why a member of the body that is not served is not.
fn unserved(name: &str) -> &'static str {
    match name {
        "mcp_servers" | "container" => {
            "the server runs no tools and connects to no MCP server: the client runs the tools \
             and sends their results"
        }
        "context_management" => "the server does not edit the history of a conversation",
        "output_format" => "structured output is asked for in output_config.format",
        "stop" => "the stop strings of this API are stop_sequences",
        _ => "it is not a parameter of this route",
    }
}

/// The members of a block (or of another object of the body), which may
/// have the members `known` and no others. `cache_control` is checked here
/// wherever it is known.
fn block<'a>(at: &str, value: &'a Json, known: &[&str]) -> Result<Fields<'a>, ApiError> {
    let Json::Object(members) = value else {
        return Err(refuse(at, format!("{at} must be an object, got {}", value.kind())));
    };
    for (name, member) in members {
        if member.is_null() {
            continue;
        }
        if !known.contains(&name.as_str()) {
            return Err(refuse(at, format!("{at}.{name} is not supported")));
        }
        if name == "cache_control" {
            cache_control(&format!("{at}.cache_control"), member)?;
        }
    }
    Ok(Fields(members))
}

/// A cache marker: `{"type": "ephemeral"}`, with a `ttl` of `5m` or `1h`.
///
/// It is accepted and changes nothing: the server keeps what it computed of
/// every prompt and starts the next one from the longest prefix it has, with
/// or without a marker, and it drops what it kept by its own budget and not
/// after a time.
fn cache_control(at: &str, value: &Json) -> Result<(), ApiError> {
    let fields = block(at, value, &["type", "ttl"])?;
    match fields.get("type") {
        Some(Json::String(kind)) if kind == "ephemeral" => {}
        _ => return Err(refuse(at, format!("{at}.type must be \"ephemeral\""))),
    }
    match fields.get("ttl") {
        None => Ok(()),
        Some(Json::String(ttl)) if ttl == "5m" || ttl == "1h" => Ok(()),
        Some(_) => Err(refuse(at, format!("{at}.ttl must be \"5m\" or \"1h\""))),
    }
}

/// A member of a block that must be a string.
fn text_member<'a>(at: &str, fields: &Fields<'a>, name: &str) -> Result<&'a str, ApiError> {
    match fields.get(name) {
        Some(Json::String(text)) => Ok(text),
        Some(other) => {
            Err(refuse(at, format!("{at}.{name} must be a string, got {}", other.kind())))
        }
        None => Err(refuse(at, format!("{at}.{name} is required"))),
    }
}

/// The kind of a block.
fn kind<'a>(at: &str, value: &'a Json) -> Result<&'a str, ApiError> {
    match value.get("type") {
        Some(Json::String(kind)) => Ok(kind),
        _ if matches!(value, Json::Object(_)) => {
            Err(refuse(at, format!("{at}.type must be a string")))
        }
        _ => Err(refuse(at, format!("{at} must be an object, got {}", value.kind()))),
    }
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
        "image" | "document" => {
            "images and documents cannot be read: input other than text is not built"
        }
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

/// Appends `text` to the text of a message, after a line break if there is
/// text already.
fn join(whole: &mut Option<String>, text: &str) {
    match whole {
        Some(whole) => {
            whole.push_str(BETWEEN_BLOCKS);
            whole.push_str(text);
        }
        None => *whole = Some(text.to_owned()),
    }
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
/// characters are overwritten, as llama-server does it.
fn steady_billing_header(text: &str) -> String {
    const HEADER: &str = "x-anthropic-billing-header:";
    const STAMP: usize = 5;
    let stamp = text
        .starts_with(HEADER)
        .then(|| text[HEADER.len()..].find("cch=").map(|at| HEADER.len() + at + 4))
        .flatten()
        .filter(|&at| {
            text.as_bytes().get(at + STAMP) == Some(&b';')
                && text.as_bytes()[at..at + STAMP].iter().all(u8::is_ascii_alphanumeric)
        });
    match stamp {
        Some(at) => format!("{}{}{}", &text[..at], "f".repeat(STAMP), &text[at + STAMP..]),
        None => text.to_owned(),
    }
}

/// `messages`: the turns of the user and of the assistant, appended to
/// `chat` as chat messages. Whether the last turn is the assistant's, which
/// the reply then continues.
fn conversation(fields: &Fields<'_>, chat: &mut Vec<Json>) -> Result<bool, ApiError> {
    let turns = match fields.get("messages") {
        Some(Json::Array(turns)) if !turns.is_empty() => turns,
        Some(Json::Array(_)) => return Err(refuse("messages", "messages must not be empty")),
        Some(other) => return Err(wrong_type("messages", "an array", other)),
        None => return Err(refuse("messages", "messages is required")),
    };
    for (i, turn) in turns.iter().enumerate() {
        let at = format!("messages[{i}]");
        let turn = block(&at, turn, &["role", "content"])?;
        let content = turn
            .get("content")
            .ok_or_else(|| refuse(format!("{at}.content"), format!("{at}.content is required")))?;
        match turn.get("role") {
            Some(Json::String(role)) if role == "user" => user(&at, content, chat)?,
            Some(Json::String(role)) if role == "assistant" => {
                chat.push(assistant(&at, content)?);
            }
            _ => {
                return Err(refuse(
                    format!("{at}.role"),
                    format!("{at}.role must be \"user\" or \"assistant\""),
                ));
            }
        }
    }
    let last = turns.last().and_then(|turn| turn.get("role"));
    Ok(matches!(last, Some(Json::String(role)) if role == "assistant"))
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
    let mut text: Option<String> = None;
    for (j, value) in blocks.iter().enumerate() {
        let at = format!("{at}.content[{j}]");
        match kind(&at, value)? {
            "text" => join(&mut text, text_block(&at, value)?),
            "tool_result" => {
                chat.extend(text.take().map(message));
                chat.push(tool_result(&at, value)?);
            }
            other => return Err(refuse_block(&at, other)),
        }
    }
    chat.extend(text.map(message));
    Ok(())
}

/// The result of a tool as a message of the role `tool`. A result that says
/// `is_error` has `Error: ` before its text: the chat template has no other
/// place for it.
fn tool_result(at: &str, value: &Json) -> Result<Json, ApiError> {
    let known = ["type", "tool_use_id", "content", "is_error", "cache_control"];
    let fields = block(at, value, &known)?;
    let id = text_member(at, &fields, "tool_use_id")?;
    let mut text = match fields.get("content") {
        None => String::new(),
        Some(Json::String(text)) => text.clone(),
        Some(Json::Array(blocks)) => {
            let mut whole = None;
            for (k, value) in blocks.iter().enumerate() {
                let at = format!("{at}.content[{k}]");
                match kind(&at, value)? {
                    "text" => join(&mut whole, text_block(&at, value)?),
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
    Ok(object(vec![
        ("role", string("tool")),
        ("tool_call_id", string(id)),
        ("content", string(text)),
    ]))
}

/// A turn of the assistant: text, thinking and the calls of tools.
///
/// The thinking goes back into the prompt as the reasoning of the message,
/// whatever its signature says: there is nothing to check it against. A
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
                        if let Some(other) = fields.get("signature")
                            && !matches!(other, Json::String(_))
                        {
                            return Err(refuse(
                                at.clone(),
                                format!("{at}.signature must be a string, got {}", other.kind()),
                            ));
                        }
                        join(&mut reasoning, text_member(&at, &fields, "thinking")?);
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
/// checked and not enforced: the engine has no budget for thinking, the model
/// thinks for as long as it does, and `max_tokens` bounds the thinking and
/// the answer together, as it does at Anthropic.
fn thinking(fields: &Fields<'_>, chat: &mut Chat) -> Result<(), ApiError> {
    let at = "thinking";
    let Some(value) = fields.get(at) else {
        return Ok(());
    };
    let kind = kind(at, value)?;
    let (on, known): (bool, &[&str]) = match kind {
        "enabled" => (true, &["type", "budget_tokens"]),
        "adaptive" => (true, &["type"]),
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
    match asked.int("budget_tokens") {
        Ok(None) => {}
        Ok(Some(budget)) if budget >= 1 => {}
        _ => {
            return Err(refuse(at, "thinking.budget_tokens must be a whole number of at least 1"));
        }
    }
    chat.push(("chat_template_kwargs", object(vec![("enable_thinking", Json::Bool(on))])));
    Ok(())
}

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
/// of one.
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
pub(crate) fn renamed(mut error: ApiError) -> ApiError {
    const NAMES: [(&str, &str); 2] =
        [("reasoning_effort", "output_config.effort"), ("response_format", "output_config.format")];
    if let Some((chat, here)) = NAMES.iter().find(|(chat, _)| error.param.as_deref() == Some(*chat))
    {
        error.message = error.message.replace(chat, here);
        error.param = Some((*here).to_owned());
    }
    error
}
