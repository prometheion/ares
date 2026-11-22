//! The body of a Responses request, written over into the body of the chat
//! request that asks the same, and what the response says back about it.
//!
//! Everything that has the shape of this API is checked here, under its
//! names, so that the reader of the chat body (`wire::from_json`) finds
//! nothing to refuse in what this module wrote; `model`, `temperature` and
//! `top_p` go by the same name in both APIs, are handed over as they are and
//! checked there.

use ares_api::{GenerateRequest, Json};

use crate::defaults::decimal;
use crate::error::ApiError;
use crate::wire::{Fields, members, wrong_type};

/// What a body is read for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Purpose {
    /// `/v1/responses`: a reply is generated.
    Reply,
    /// `/v1/responses/input_tokens`: the prompt is counted, so `stream` says
    /// nothing.
    Count,
}

/// The members of a body that are served.
const SERVED: &[&str] = &[
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_output_tokens",
    "temperature",
    "top_p",
    "reasoning",
    "text",
    "stream",
    "stream_options",
    "include",
    "truncation",
    "store",
    "background",
];

/// Bookkeeping of the client or of OpenAI's service that is a string:
/// accepted, and by its own definition without effect on the reply.
///
/// The members about the cache of prompts are among them: the server keeps
/// what it computed of every prompt and starts the next one from the longest
/// prefix it has, with or without a key, and it drops what it kept by its
/// own budget, so a retention is no promise it makes.
const NOTES: &[&str] =
    &["user", "safety_identifier", "service_tier", "prompt_cache_key", "prompt_cache_retention"];

/// The same, as an object. `client_metadata` is what Codex sends with every
/// request.
const RECORDS: &[&str] = &["metadata", "client_metadata", "prompt_cache_options"];

/// What the instructions before a conversation are joined with: an empty
/// line.
const BETWEEN_INSTRUCTIONS: &str = "\n\n";

/// What the parts of the reasoning of one turn are joined with.
const BETWEEN_REASONING: &str = "\n";

/// What a response says back about its request: the members clients read to
/// see what was applied.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Echo {
    /// A string, or `null`.
    pub instructions: Json,
    pub max_output_tokens: Json,
    /// An object; empty when the request has none.
    pub metadata: Json,
    pub parallel_tool_calls: bool,
    /// `reasoning.effort`, or `null`.
    pub effort: Json,
    pub temperature: Json,
    /// `text.format`; `{"type": "text"}` when the request has none.
    pub format: Json,
    pub tool_choice: Json,
    /// The tools as the request wrote them.
    pub tools: Json,
    pub top_p: Json,
    pub user: Json,
}

impl Echo {
    /// Makes the response say what was applied to `request`, the request as
    /// the engine gets it: the model's default where the body left
    /// `temperature` or `top_p` out (`null` where the model has none and the
    /// engine's own value holds), and the limit of tokens the reply has, which
    /// is the model's when the body named none or more than the model allows.
    pub fn applied(&mut self, request: &GenerateRequest) {
        let number = |value: Option<f32>| value.map_or(Json::Null, |v| Json::Float(decimal(v)));
        if self.temperature.is_null() {
            self.temperature = number(request.sampling.temperature);
        }
        if self.top_p.is_null() {
            self.top_p = number(request.sampling.top_p);
        }
        self.max_output_tokens =
            request.max_tokens.map_or(Json::Null, |limit| Json::Int(i64::from(limit)));
    }
}

fn refuse(param: impl Into<String>, message: impl Into<String>) -> ApiError {
    ApiError::invalid_param(param, message)
}

fn object(members: Vec<(&str, Json)>) -> Json {
    Json::Object(members.into_iter().map(|(name, value)| (name.to_owned(), value)).collect())
}

fn string(text: impl Into<String>) -> Json {
    Json::String(text.into())
}

/// The members of the chat body as they are written.
type Chat = Vec<(&'static str, Json)>;

/// Writes the body of a Responses request over into the body of a chat
/// request, and keeps what the response says back.
pub(crate) fn to_chat(body: &Json, purpose: Purpose) -> Result<(Json, Echo), ApiError> {
    let fields = Fields(members(body)?);
    for (name, value) in fields.0 {
        let name = name.as_str();
        if value.is_null() || SERVED.contains(&name) {
            continue;
        }
        match value {
            Json::String(_) if NOTES.contains(&name) => continue,
            Json::Object(_) if RECORDS.contains(&name) => continue,
            other if NOTES.contains(&name) => return Err(wrong_type(name, "a string", other)),
            other if RECORDS.contains(&name) => return Err(wrong_type(name, "an object", other)),
            _ => return Err(refuse(name, format!("{name} is not supported: {}", unserved(name)))),
        }
    }
    stateless(&fields)?;

    let mut chat = Chat::new();
    for name in ["model", "temperature", "top_p"] {
        if let Some(value) = fields.get(name) {
            chat.push((name, value.clone()));
        }
    }
    match fields.int("max_output_tokens")? {
        Some(limit) if limit >= 1 => chat.push(("max_tokens", Json::Int(limit))),
        Some(limit) => {
            return Err(refuse(
                "max_output_tokens",
                format!("max_output_tokens must be at least 1, got {limit}"),
            ));
        }
        None => {}
    }
    // A count has no stream; the field is checked all the same.
    if let Some(stream) = fields.bool("stream")?
        && purpose == Purpose::Reply
    {
        chat.push(("stream", Json::Bool(stream)));
    }

    chat.push(("messages", Json::Array(conversation(&fields)?)));
    let has_tools = tools(&fields, &mut chat)?;
    let callable = tool_choice(&fields, has_tools, &mut chat)?;
    if let Some(parallel) = fields.bool("parallel_tool_calls")? {
        chat.push(("parallel_tool_calls", Json::Bool(parallel)));
    }
    let effort = reasoning(&fields, &mut chat)?;
    let format = text(&fields, callable, &mut chat)?;

    let given = |name: &str| fields.get(name).cloned().unwrap_or(Json::Null);
    let echo = Echo {
        instructions: given("instructions"),
        max_output_tokens: given("max_output_tokens"),
        metadata: fields.get("metadata").cloned().unwrap_or_else(|| object(Vec::new())),
        parallel_tool_calls: fields.bool("parallel_tool_calls")?.unwrap_or(true),
        effort,
        temperature: given("temperature"),
        format,
        tool_choice: fields.get("tool_choice").cloned().unwrap_or_else(|| string("auto")),
        tools: fields.get("tools").cloned().unwrap_or_else(|| Json::Array(Vec::new())),
        top_p: given("top_p"),
        user: given("user"),
    };
    Ok((object(chat), echo))
}

/// Why a member of the body that is not served is not.
fn unserved(name: &str) -> &'static str {
    match name {
        "previous_response_id" | "conversation" => {
            "this route keeps no responses and no conversations (every response says \"store\": \
             false), so there is nothing to continue: send the whole conversation in input"
        }
        "prompt" => "prompt templates are kept by OpenAI's service; send instructions and input",
        "top_logprobs" => {
            "the probabilities of tokens are not served on this route: the reply's tokens are not \
             told apart into those of the answer, of the reasoning and of calls"
        }
        "max_tool_calls" => "the server runs no tools of its own, so it has no calls to count",
        "context_management" => "the server does not edit or compact the history of a conversation",
        "messages" => "the conversation of this API is input",
        "max_tokens" | "max_completion_tokens" => "the limit of this API is max_output_tokens",
        "response_format" | "json_schema" => "structured output is asked for in text.format",
        "reasoning_effort" => "the effort of this API is reasoning.effort",
        _ => "it is not a parameter of this route",
    }
}

/// The members that ask the server to keep something, or to change what it
/// was given: this route keeps nothing and answers the request it got.
fn stateless(fields: &Fields<'_>) -> Result<(), ApiError> {
    if fields.bool("store")? == Some(true) {
        return Err(refuse(
            "store",
            "store: true is not supported: this route keeps no responses. Leave store out or \
             send false; the response says \"store\": false",
        ));
    }
    if fields.bool("background")? == Some(true) {
        return Err(refuse(
            "background",
            "background: true is not supported: a response is generated while its request waits, \
             and none is kept to be fetched later",
        ));
    }
    match fields.string("truncation")? {
        None | Some("disabled") => {}
        Some("auto") => {
            return Err(refuse(
                "truncation",
                "truncation \"auto\" is not supported: the server drops nothing from a \
                 conversation, and an input that does not fit the context is an error \
                 (\"disabled\")",
            ));
        }
        Some(other) => {
            return Err(refuse(
                "truncation",
                format!("truncation must be \"disabled\" or \"auto\", got \"{other}\""),
            ));
        }
    }
    match fields.get("include") {
        None => {}
        Some(Json::Array(values)) => {
            for value in values {
                let Json::String(value) = value else {
                    return Err(refuse(
                        "include",
                        format!("include must hold strings, found {}", value.kind()),
                    ));
                };
                included(value)?;
            }
        }
        Some(other) => return Err(wrong_type("include", "an array of strings", other)),
    }
    if let Some(options) = fields.get("stream_options") {
        let options = known("stream_options", options, &["include_obfuscation"])?;
        if options.bool("include_obfuscation").map_err(|e| refuse("stream_options", e.message))?
            == Some(true)
        {
            return Err(refuse(
                "stream_options",
                "stream_options.include_obfuscation: true is not supported: the events of a \
                 stream carry no padding",
            ));
        }
    }
    Ok(())
}

/// One value of `include`.
///
/// `reasoning.encrypted_content` asks for a reasoning item that a client
/// without stored responses can send back. That is what every reasoning
/// item of a reply is: it carries its reasoning in the clear, in `content`,
/// and the server has nothing to encrypt it with or to hide. So the value is
/// accepted and adds nothing. Every other value asks for data the server
/// does not have.
fn included(value: &str) -> Result<(), ApiError> {
    let why = match value {
        "reasoning.encrypted_content" => return Ok(()),
        "message.output_text.logprobs" => {
            "the probabilities of tokens are not served on this route: the reply's tokens are not \
             told apart into those of the answer, of the reasoning and of calls"
        }
        "message.input_image.image_url" | "computer_call_output.output.image_url" => {
            "images cannot be read: input other than text is not built"
        }
        "file_search_call.results"
        | "web_search_call.results"
        | "web_search_call.action.sources"
        | "code_interpreter_call.outputs" => "the server runs no tools of its own",
        _ => "it is not a value this server knows",
    };
    Err(refuse("include", format!("include \"{value}\" is not supported: {why}")))
}

/// The members of an object of the body, which may have the members `names`
/// and no others. A `null` is a member that is not set.
fn known<'a>(at: &str, value: &'a Json, names: &[&str]) -> Result<Fields<'a>, ApiError> {
    let Json::Object(members) = value else {
        return Err(refuse(at, format!("{at} must be an object, got {}", value.kind())));
    };
    for (name, member) in members {
        if !member.is_null() && !names.contains(&name.as_str()) {
            return Err(refuse(at, format!("{at}.{name} is not supported")));
        }
    }
    Ok(Fields(members))
}

/// A member of an object that must be a string, if it is there.
fn optional_text<'a>(
    at: &str,
    fields: &Fields<'a>,
    name: &str,
) -> Result<Option<&'a str>, ApiError> {
    match fields.get(name) {
        None => Ok(None),
        Some(Json::String(text)) => Ok(Some(text)),
        Some(other) => {
            Err(refuse(at, format!("{at}.{name} must be a string, got {}", other.kind())))
        }
    }
}

/// A member of an object that must be a string.
fn text_member<'a>(at: &str, fields: &Fields<'a>, name: &str) -> Result<&'a str, ApiError> {
    optional_text(at, fields, name)?.ok_or_else(|| refuse(at, format!("{at}.{name} is required")))
}

/// The same, and not empty.
fn name_member<'a>(at: &str, fields: &Fields<'a>, name: &str) -> Result<&'a str, ApiError> {
    match text_member(at, fields, name)? {
        "" => Err(refuse(at, format!("{at}.{name} must not be empty"))),
        text => Ok(text),
    }
}

/// The `type` of an item, of a part or of a tool, if it says one.
fn kind<'a>(at: &str, value: &'a Json) -> Result<Option<&'a str>, ApiError> {
    match value {
        Json::Object(_) => match value.get("type") {
            None | Some(Json::Null) => Ok(None),
            Some(Json::String(kind)) => Ok(Some(kind)),
            Some(other) => {
                Err(refuse(at, format!("{at}.type must be a string, got {}", other.kind())))
            }
        },
        other => Err(refuse(at, format!("{at} must be an object, got {}", other.kind()))),
    }
}

/// Appends `text` to `whole`, after `between` if there is text already.
fn join(whole: &mut Option<String>, between: &str, text: &str) {
    match whole {
        Some(whole) => {
            whole.push_str(between);
            whole.push_str(text);
        }
        None => *whole = Some(text.to_owned()),
    }
}

/// A turn of the assistant as the items of a conversation build it: its
/// reasoning, its text and its calls, which a reply returns as items of
/// their own, in that order.
#[derive(Default)]
struct Turn {
    reasoning: Option<String>,
    text: Option<String>,
    calls: Vec<Json>,
}

impl Turn {
    /// Whether the turn has its answer: what is said after the reasoning.
    fn answered(&self) -> bool {
        self.text.is_some() || !self.calls.is_empty()
    }

    /// The turn as a chat message, if it has anything.
    fn message(self) -> Option<Json> {
        if self.reasoning.is_none() && !self.answered() {
            return None;
        }
        let mut message =
            vec![("role", string("assistant")), ("content", string(self.text.unwrap_or_default()))];
        if let Some(reasoning) = self.reasoning {
            message.push(("reasoning_content", string(reasoning)));
        }
        if !self.calls.is_empty() {
            message.push(("tool_calls", Json::Array(self.calls)));
        }
        Some(object(message))
    }
}

/// The messages of a chat request as `instructions` and `input` give them.
struct Conversation {
    chat: Vec<Json>,
    /// The instructions before the conversation, while it has not begun:
    /// `instructions` and the messages of the roles `system` and `developer`
    /// that stand first. They are one system message.
    instructions: Option<Vec<String>>,
    /// The turn of the assistant that is being put together.
    turn: Turn,
}

impl Conversation {
    /// Ends what stands before the next message: the instructions, and the
    /// turn of the assistant.
    fn close(&mut self) {
        if let Some(mut parts) = self.instructions.take() {
            parts.retain(|part| !part.is_empty());
            if !parts.is_empty() {
                let text = parts.join(BETWEEN_INSTRUCTIONS);
                self.chat.push(object(vec![("role", string("system")), ("content", string(text))]));
            }
        }
        self.chat.extend(std::mem::take(&mut self.turn).message());
    }

    /// Ends the instructions, and the turn of the assistant if it has its
    /// answer: what comes next begins another turn.
    fn next_turn(&mut self) {
        if self.instructions.is_some() || self.turn.answered() {
            self.close();
        }
    }

    fn push(&mut self, message: Vec<(&str, Json)>) {
        self.close();
        self.chat.push(object(message));
    }
}

/// `instructions` and `input`, as the messages of a chat request.
///
/// The items a reply returned for one turn (a reasoning item, a message,
/// calls) are one message of the assistant again, so that the chat template
/// writes the turn as the model generated it. An item that does not continue
/// a turn in that order begins another message.
fn conversation(fields: &Fields<'_>) -> Result<Vec<Json>, ApiError> {
    let mut instructions = Vec::new();
    match fields.get("instructions") {
        None => {}
        Some(Json::String(text)) if text.is_empty() => {}
        Some(Json::String(text)) => instructions.push(text.clone()),
        Some(other) => return Err(wrong_type("instructions", "a string", other)),
    }
    let mut all =
        Conversation { chat: Vec::new(), instructions: Some(instructions), turn: Turn::default() };
    match fields.get("input") {
        Some(Json::String(text)) => {
            all.push(vec![("role", string("user")), ("content", string(text.as_str()))]);
        }
        Some(Json::Array(items)) => {
            for (i, value) in items.iter().enumerate() {
                item(&format!("input[{i}]"), value, &mut all)?;
            }
        }
        Some(other) => return Err(wrong_type("input", "a string or an array of items", other)),
        None => return Err(refuse("input", "input is required")),
    }
    all.close();
    if all.chat.is_empty() {
        return Err(refuse("input", "input must not be empty"));
    }
    Ok(all.chat)
}

/// One item of `input`.
fn item(at: &str, value: &Json, all: &mut Conversation) -> Result<(), ApiError> {
    let kind = match kind(at, value)? {
        Some(kind) => kind,
        // A message may leave its type out.
        None if value.get("role").is_some_and(|role| !role.is_null()) => "message",
        None => return Err(refuse(at, format!("{at}.type is required"))),
    };
    match kind {
        "message" => message(at, value, all),
        "reasoning" => {
            let text = reasoning_item(at, value)?;
            all.next_turn();
            if let Some(text) = text {
                join(&mut all.turn.reasoning, BETWEEN_REASONING, &text);
            }
            Ok(())
        }
        "function_call" => {
            let call = function_call(at, value)?;
            if all.instructions.is_some() {
                all.close();
            }
            all.turn.calls.push(call);
            Ok(())
        }
        "function_call_output" => {
            let known = ["type", "id", "status", "call_id", "output", "name"];
            let fields = known_item(at, value, &known)?;
            let mut message = vec![
                ("role", string("tool")),
                ("tool_call_id", string(name_member(at, &fields, "call_id")?)),
                ("content", string(call_output(at, &fields)?)),
            ];
            if let Some(name) = optional_text(at, &fields, "name")? {
                message.push(("name", string(name)));
            }
            all.push(message);
            Ok(())
        }
        other => Err(refuse_item(at, other)),
    }
}

/// Why an item of a type that is not served is refused.
fn refuse_item(at: &str, kind: &str) -> ApiError {
    let why = match kind {
        "item_reference" => {
            "this route keeps no responses, so an item cannot be named by its id: send the item \
             itself"
        }
        "custom_tool_call" | "custom_tool_call_output" => {
            "custom tools are not built: a tool of this server is a function with arguments"
        }
        "compaction" => "it holds a conversation only OpenAI's service can read",
        "file_search_call"
        | "web_search_call"
        | "computer_call"
        | "computer_call_output"
        | "image_generation_call"
        | "code_interpreter_call"
        | "local_shell_call"
        | "local_shell_call_output"
        | "shell_call"
        | "shell_call_output"
        | "apply_patch_call"
        | "apply_patch_call_output"
        | "tool_search_call"
        | "tool_search_output"
        | "mcp_list_tools"
        | "mcp_approval_request"
        | "mcp_approval_response"
        | "mcp_call" => "the server runs no tools of its own",
        _ => "it is not a kind of item this server reads",
    };
    refuse(at, format!("{at}: an item of the type '{kind}' is not supported: {why}"))
}

/// The members of an item, which may have the members `names` and no others.
/// `id` and `status` are the bookkeeping of the response an item came from.
fn known_item<'a>(at: &str, value: &'a Json, names: &[&str]) -> Result<Fields<'a>, ApiError> {
    let fields = known(at, value, names)?;
    for name in ["id", "status"] {
        optional_text(at, &fields, name)?;
    }
    Ok(fields)
}

/// A message: of the user, of the assistant, or instructions (`system`,
/// `developer`).
fn message(at: &str, value: &Json, all: &mut Conversation) -> Result<(), ApiError> {
    let fields = known_item(at, value, &["type", "role", "content", "id", "status"])?;
    let role = text_member(at, &fields, "role")?;
    if !["user", "assistant", "system", "developer"].contains(&role) {
        return Err(refuse(
            format!("{at}.role"),
            format!(
                "{at}.role must be \"user\", \"assistant\", \"system\" or \"developer\", got \
                 \"{role}\""
            ),
        ));
    }
    // A message of the user with images: its parts as the chat route reads
    // them (`text` and `image_url`), joined there as llama-server joins them.
    if role == "user"
        && let Some(Json::Array(parts)) = fields.get("content")
        && parts.iter().any(|p| p.get("type").and_then(Json::as_str) == Some("input_image"))
    {
        let mut chat_parts = Vec::with_capacity(parts.len());
        for (j, part) in parts.iter().enumerate() {
            let at = format!("{at}.content[{j}]");
            if part.get("type").and_then(Json::as_str) == Some("input_image") {
                chat_parts.push(image_part(&at, part)?);
            } else {
                let text = text_part(&at, part, role)?;
                chat_parts.push(object(vec![("type", string("text")), ("text", string(text))]));
            }
        }
        all.push(vec![("role", string("user")), ("content", Json::Array(chat_parts))]);
        return Ok(());
    }
    let text = match fields.get("content") {
        Some(Json::String(text)) => text.clone(),
        Some(Json::Array(parts)) => {
            let mut whole = String::new();
            for (j, part) in parts.iter().enumerate() {
                whole.push_str(text_part(&format!("{at}.content[{j}]"), part, role)?);
            }
            whole
        }
        Some(other) => {
            return Err(refuse(
                format!("{at}.content"),
                format!("{at}.content must be a string or an array of parts, got {}", other.kind()),
            ));
        }
        None => return Err(refuse(format!("{at}.content"), format!("{at}.content is required"))),
    };
    match role {
        "system" | "developer" => match &mut all.instructions {
            Some(instructions) => instructions.push(text),
            None => all.push(vec![("role", string(role)), ("content", string(text))]),
        },
        "assistant" => {
            all.next_turn();
            all.turn.text = Some(text);
        }
        _ => all.push(vec![("role", string(role)), ("content", string(text))]),
    }
    Ok(())
}

/// The text of a part of a message. `input_text` and `output_text` are text
/// whoever wrote it; a `refusal` is what the assistant said.
fn text_part<'a>(at: &str, part: &'a Json, role: &str) -> Result<&'a str, ApiError> {
    let kind = kind(at, part)?.ok_or_else(|| refuse(at, format!("{at}.type is required")))?;
    match kind {
        "input_text" => text_member(at, &known(at, part, &["type", "text"])?, "text"),
        "output_text" => {
            let fields = known(at, part, &["type", "text", "annotations", "logprobs"])?;
            for name in ["annotations", "logprobs"] {
                if let Some(other) = fields.get(name)
                    && !matches!(other, Json::Array(_))
                {
                    return Err(refuse(at, format!("{at}.{name} must be an array")));
                }
            }
            text_member(at, &fields, "text")
        }
        "refusal" if role == "assistant" => {
            text_member(at, &known(at, part, &["type", "refusal"])?, "refusal")
        }
        other => Err(refuse_part(at, other)),
    }
}

/// An `input_image` part as the part of a chat message the chat route reads
/// (`image_url`): an image given by a data URL. One given by `file_id` or at a
/// remote URL is not read.
fn image_part(at: &str, part: &Json) -> Result<Json, ApiError> {
    let fields = known(at, part, &["type", "image_url", "file_id", "detail"])?;
    if fields.get("file_id").is_some_and(|v| !v.is_null()) {
        return Err(refuse(
            at,
            format!("{at}.file_id: files are not kept by this server; send the image as image_url"),
        ));
    }
    match fields.get("detail") {
        None | Some(Json::Null) => {}
        Some(Json::String(detail)) if ["auto", "low", "high"].contains(&detail.as_str()) => {}
        Some(_) => {
            return Err(refuse(at, format!("{at}.detail must be \"auto\", \"low\" or \"high\"")));
        }
    }
    let url = text_member(at, &fields, "image_url")?;
    Ok(object(vec![
        ("type", string("image_url")),
        ("image_url", object(vec![("url", string(url))])),
    ]))
}

fn refuse_part(at: &str, kind: &str) -> ApiError {
    let why = match kind {
        "input_image" => "an image stands in a message of the user",
        "input_file" => "files cannot be read: input other than text and images is not built",
        "input_audio" => "audio cannot be read: input other than text is not built",
        _ => "it is not a kind of part this server reads here",
    };
    refuse(at, format!("{at}: a part of the type '{kind}' is not supported: {why}"))
}

/// The reasoning a reasoning item carries, if the server can read it.
///
/// A reasoning item of this server has its reasoning in `content`, in the
/// clear. That is what goes back into the prompt, so that the chat template
/// writes the turn as the model generated it. A `summary` is not the
/// reasoning, and an `encrypted_content` is text only the service that wrote
/// it can read: both are accepted, and nothing of them enters the prompt.
fn reasoning_item(at: &str, value: &Json) -> Result<Option<String>, ApiError> {
    let known = ["type", "id", "status", "summary", "content", "encrypted_content"];
    let fields = known_item(at, value, &known)?;
    optional_text(at, &fields, "encrypted_content")?;
    match fields.get("summary") {
        None | Some(Json::Array(_)) => {}
        Some(other) => {
            return Err(refuse(at, format!("{at}.summary must be an array, got {}", other.kind())));
        }
    }
    let parts = match fields.get("content") {
        None => return Ok(None),
        Some(Json::Array(parts)) => parts,
        Some(other) => {
            return Err(refuse(at, format!("{at}.content must be an array, got {}", other.kind())));
        }
    };
    let mut whole = None;
    for (j, part) in parts.iter().enumerate() {
        let at = format!("{at}.content[{j}]");
        match kind(&at, part)? {
            Some("reasoning_text" | "text") => {
                let text = text_member(&at, &known_part(&at, part)?, "text")?;
                join(&mut whole, BETWEEN_REASONING, text);
            }
            other => return Err(refuse_part(&at, other.unwrap_or("(none)"))),
        }
    }
    Ok(whole)
}

fn known_part<'a>(at: &str, part: &'a Json) -> Result<Fields<'a>, ApiError> {
    known(at, part, &["type", "text"])
}

/// A call of a function in a turn of the assistant, as a call of a chat
/// message. The arguments are the text of a JSON object, as a reply returns
/// them.
fn function_call(at: &str, value: &Json) -> Result<Json, ApiError> {
    let fields = known_item(at, value, &["type", "id", "status", "call_id", "name", "arguments"])?;
    let id = name_member(at, &fields, "call_id")?;
    let name = name_member(at, &fields, "name")?;
    let arguments = text_member(at, &fields, "arguments")?;
    let is_object = arguments.trim().is_empty()
        || matches!(serde_json::from_str::<Json>(arguments), Ok(Json::Object(_)));
    if !is_object {
        return Err(refuse(at, format!("{at}.arguments must be the text of a JSON object")));
    }
    let function = object(vec![("name", string(name)), ("arguments", string(arguments))]);
    Ok(object(vec![("id", string(id)), ("type", string("function")), ("function", function)]))
}

/// The output of a call: a string, or parts of text.
fn call_output(at: &str, fields: &Fields<'_>) -> Result<String, ApiError> {
    match fields.get("output") {
        Some(Json::String(text)) => Ok(text.clone()),
        Some(Json::Array(parts)) => {
            let mut whole = String::new();
            for (j, part) in parts.iter().enumerate() {
                let at = format!("{at}.output[{j}]");
                match kind(&at, part)? {
                    Some("input_text") => {
                        whole.push_str(text_member(&at, &known_part(&at, part)?, "text")?);
                    }
                    other => return Err(refuse_part(&at, other.unwrap_or("(none)"))),
                }
            }
            Ok(whole)
        }
        Some(other) => Err(refuse(
            at,
            format!("{at}.output must be a string or an array of parts, got {}", other.kind()),
        )),
        None => Err(refuse(at, format!("{at}.output is required"))),
    }
}

/// `tools`, as the tools of a chat request: the functions, whose members
/// stand in the tool itself here and under `function` there. Returns whether
/// there are any.
fn tools(fields: &Fields<'_>, chat: &mut Chat) -> Result<bool, ApiError> {
    let tools = match fields.get("tools") {
        None => return Ok(false),
        Some(Json::Array(tools)) => tools,
        Some(other) => return Err(wrong_type("tools", "an array", other)),
    };
    let mut functions = Vec::new();
    for (i, value) in tools.iter().enumerate() {
        let at = format!("tools[{i}]");
        let refuse = |message: String| refuse("tools", message);
        match kind(&at, value).map_err(|e| refuse(e.message))? {
            Some("function") => {}
            Some(other) => return Err(refuse(refused_tool(&at, other))),
            None => return Err(refuse(format!("{at}.type is required"))),
        }
        let names = ["type", "name", "description", "parameters", "strict"];
        let tool = known(&at, value, &names).map_err(|e| refuse(e.message))?;
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
        match tool.get("parameters") {
            None => {}
            Some(schema @ Json::Object(_)) => function.push(("parameters", schema.clone())),
            Some(_) => return Err(refuse(format!("{at}.parameters must be an object"))),
        }
        // The arguments are held to the schema whether a tool says so or not.
        tool.bool("strict").map_err(|_| refuse(format!("{at}.strict must be a boolean")))?;
        functions.push(object(vec![("type", string("function")), ("function", object(function))]));
    }
    // An empty list is no tools: the chat template is not to open a section
    // for them.
    let any = !functions.is_empty();
    if any {
        chat.push(("tools", Json::Array(functions)));
    }
    Ok(any)
}

/// Why a tool of a type that is not a function is refused.
fn refused_tool(at: &str, kind: &str) -> String {
    let why = match kind {
        "custom" => {
            "a custom tool takes free text and not arguments of a schema, and the calls this \
             server reads and enforces are calls of functions"
        }
        _ => {
            "it is run or defined by OpenAI's service; this server takes tools of the type \
             \"function\", which the client runs"
        }
    };
    format!("{at}: a tool of the type '{kind}' is not supported: {why}")
}

/// `tool_choice`, as the one of a chat request. Returns whether the reply
/// may call a tool.
fn tool_choice(fields: &Fields<'_>, has_tools: bool, chat: &mut Chat) -> Result<bool, ApiError> {
    let at = "tool_choice";
    let chosen = match fields.get(at) {
        None => return Ok(has_tools),
        // The words are the chat route's, which checks them.
        Some(word @ Json::String(_)) => word.clone(),
        Some(value @ Json::Object(_)) => match kind(at, value)? {
            Some("function") => {
                let name = name_member(at, &known(at, value, &["type", "name"])?, "name")?;
                let function = object(vec![("name", string(name))]);
                object(vec![("type", string("function")), ("function", function)])
            }
            Some("allowed_tools") => {
                return Err(refuse(
                    at,
                    "tool_choice of the type 'allowed_tools' is not supported: the reply may call \
                     every tool of the request or one named function, not a subset",
                ));
            }
            Some(other) => {
                return Err(refuse(
                    at,
                    format!(
                        "tool_choice of the type '{other}' is not supported: the server runs no \
                         tools of its own, and a tool to call is named as {{\"type\": \
                         \"function\", \"name\": ...}}"
                    ),
                ));
            }
            None => return Err(refuse(at, "tool_choice.type is required")),
        },
        Some(other) => return Err(wrong_type(at, "a string or an object", other)),
    };
    let none = chosen.as_str() == Some("none");
    chat.push((at, chosen));
    Ok(has_tools && !none)
}

/// `reasoning`: `effort` is the `reasoning_effort` of a chat request.
/// Returns the effort for the response to say back.
///
/// The server writes no summary of the reasoning: a reasoning item has the
/// reasoning itself. `summary: "auto"` leaves the choice to the server and is
/// served with that; a request for a summary of a given kind is refused.
fn reasoning(fields: &Fields<'_>, chat: &mut Chat) -> Result<Json, ApiError> {
    let at = "reasoning";
    let Some(value) = fields.get(at) else {
        return Ok(Json::Null);
    };
    let asked = known(at, value, &["effort", "summary"])?;
    match optional_text(at, &asked, "summary")? {
        None | Some("auto") => {}
        Some(kind @ ("concise" | "detailed")) => {
            return Err(refuse(
                at,
                format!(
                    "reasoning.summary \"{kind}\" is not supported: the server writes no \
                     summaries; the reasoning itself is in the content of the reasoning item"
                ),
            ));
        }
        Some(other) => {
            return Err(refuse(
                at,
                format!(
                    "reasoning.summary must be \"auto\", \"concise\" or \"detailed\", got \
                     \"{other}\""
                ),
            ));
        }
    }
    match optional_text(at, &asked, "effort")? {
        None => Ok(Json::Null),
        Some("") => Err(refuse(at, "reasoning.effort must not be empty")),
        Some(effort) => {
            chat.push(("reasoning_effort", string(effort)));
            Ok(string(effort))
        }
    }
}

/// `text`: `format` is the `response_format` of a chat request. Returns the
/// format for the response to say back.
fn text(fields: &Fields<'_>, callable: bool, chat: &mut Chat) -> Result<Json, ApiError> {
    let plain = || object(vec![("type", string("text"))]);
    let Some(value) = fields.get("text") else {
        return Ok(plain());
    };
    let asked = known("text", value, &["format", "verbosity"])?;
    if asked.get("verbosity").is_some() {
        return Err(refuse(
            "text",
            "text.verbosity is not supported: how much a model writes is not a setting of the \
             engine or of the chat template",
        ));
    }
    let Some(format) = asked.get("format") else {
        return Ok(plain());
    };
    let at = "text.format";
    let names = ["type", "name", "schema", "strict", "description"];
    let given = known(at, format, &names)?;
    for name in ["name", "description"] {
        optional_text(at, &given, name)?;
    }
    given.bool("strict").map_err(|_| refuse(at, "text.format.strict must be a boolean"))?;
    let rule = match (kind(at, format)?, given.get("schema")) {
        (Some("text"), None) => return Ok(format.clone()),
        (Some("json_object"), None) => object(vec![("type", string("json_object"))]),
        (Some("json_schema"), Some(schema @ Json::Object(_))) => {
            let wrapper = object(vec![("schema", schema.clone())]);
            object(vec![("type", string("json_schema")), ("json_schema", wrapper)])
        }
        (Some("json_schema"), _) => {
            return Err(refuse(
                at,
                "text.format of the type \"json_schema\" needs its schema: {\"type\": \
                 \"json_schema\", \"name\": ..., \"schema\": {...}}",
            ));
        }
        (Some("text" | "json_object"), Some(_)) => {
            return Err(refuse(at, "text.format.schema belongs to the type \"json_schema\""));
        }
        (other, _) => {
            return Err(refuse(
                at,
                format!(
                    "text.format.type must be \"text\", \"json_object\" or \"json_schema\", got \
                     {}",
                    other.map_or("nothing".to_owned(), |kind| format!("\"{kind}\""))
                ),
            ));
        }
    };
    if callable {
        return Err(refuse(
            at,
            "text.format cannot be combined with tools that may be called: the reply follows one \
             rule. Send tool_choice \"none\" to have the tools in the prompt and the reply \
             follow the format",
        ));
    }
    chat.push(("response_format", rule));
    Ok(format.clone())
}

/// An error of the reader of the chat body, under the name the field has in
/// this API.
pub(crate) fn renamed(mut error: ApiError) -> ApiError {
    const NAMES: [(&str, &str); 3] = [
        ("reasoning_effort", "reasoning.effort"),
        ("response_format", "text.format"),
        ("max_tokens", "max_output_tokens"),
    ];
    for (chat, here) in NAMES {
        if error.param.as_deref().is_some_and(|param| param.starts_with(chat)) {
            error.message = error.message.replace(chat, here);
            error.param = error.param.take().map(|param| param.replacen(chat, here, 1));
        }
    }
    error
}
