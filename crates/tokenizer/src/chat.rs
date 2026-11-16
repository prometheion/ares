//! Chat templates: a conversation into the text a model was trained on.
//!
//! The text is made by the model's own Jinja template, `tokenizer.chat_template`
//! in the model file, the way llama.cpp does it with `--jinja`: the messages are
//! brought into the shape of the OpenAI API, a few conventions of llama.cpp are
//! applied (the role `developer` is `system`, the arguments of a tool call are
//! parsed into an object), and the template runs with `messages`, `tools`,
//! `add_generation_prompt`, `enable_thinking`, `bos_token` and `eos_token`.
//!
//! The dialect of Jinja that is supported is described at [`ChatTemplate`]. A
//! template that needs more is refused by [`ChatTemplate::new`] with an error
//! that says what and where.

use minijinja::{Environment, Value};

use crate::jinja;
use crate::json::Json;

/// Why a conversation could not be turned into a prompt.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ChatError {
    /// The model file has no chat template.
    #[error("the model has no chat template ('tokenizer.chat_template')")]
    NoTemplate,
    /// The template uses something outside the supported dialect of Jinja.
    #[error("the chat template uses {what} (line {line}), which is not supported")]
    Unsupported {
        /// What it uses.
        what: String,
        /// The line in the template.
        line: u32,
    },
    /// The template is not valid Jinja.
    #[error("the chat template is not valid: {0}")]
    Syntax(String),
    /// The template failed on this conversation: it raised an exception
    /// (`raise_exception`, the way templates refuse input they cannot show), or
    /// used a value in a way that is not supported.
    #[error("the chat template failed: {0}")]
    Render(String),
    /// A message or tool is not what the template can be given.
    #[error("invalid conversation: {0}")]
    Invalid(String),
}

/// The text of a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Content {
    /// A string, which is what almost all messages are.
    Text(String),
    /// A list of text parts, each of which the template sees as
    /// `{"type": "text", "text": ...}`.
    Parts(Vec<String>),
}

impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}

/// A call of a tool that the assistant made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// The id the call has in the conversation, which the answer refers back to.
    pub id: Option<String>,
    /// The name of the tool.
    pub name: String,
    /// The arguments as JSON text, normally an object: `{"city": "Berlin"}`.
    pub arguments: String,
}

/// One message of a conversation.
///
/// As in llama.cpp, an empty string in an optional field is the same as the
/// field being absent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Message {
    /// `system`, `user`, `assistant` or `tool` (`developer` counts as `system`).
    pub role: String,
    /// What the message says. A message without content has empty content.
    pub content: Content,
    /// What the assistant thought before it answered.
    pub reasoning_content: String,
    /// For a message with the role `tool`: the name of the tool.
    pub name: String,
    /// For a message with the role `tool`: the id of the call it answers.
    pub tool_call_id: String,
    /// The tools the assistant called in this message.
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    /// A message with a role and a text.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: Content::Text(content.into()), ..Self::default() }
    }
}

/// A tool the assistant may call.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    /// The name the assistant calls it by.
    pub name: String,
    /// What it does; the template shows it to the model.
    pub description: String,
    /// The JSON schema of its arguments, normally an object.
    pub parameters: Json,
}

/// Settings of one rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatOptions {
    /// End the prompt with the start of the assistant's turn, so that the model
    /// continues with its answer. Leave it off for a prompt that is only a
    /// conversation to be scored.
    pub add_generation_prompt: bool,
    /// Whether the model should think before it answers (`enable_thinking`).
    /// Templates of models that cannot think ignore it.
    pub enable_thinking: bool,
    /// Further variables for the template, like `chat_template_kwargs` of the
    /// API (`preserve_thinking`, `reasoning_effort`, ...). They override a variable
    /// of the same name that this module sets, except `bos_token` and
    /// `eos_token`.
    pub variables: Vec<(String, Json)>,
    /// A template that takes the place of the model file's, for every
    /// conversation, with tools or without ([`Tokenizer::compile_template`]
    /// makes one). `None`: the file's.
    ///
    /// [`Tokenizer::compile_template`]: crate::Tokenizer::compile_template
    pub template: Option<std::sync::Arc<ChatTemplate>>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            variables: Vec::new(),
            template: None,
        }
    }
}

/// A chat template that can be rendered.
///
/// # The dialect
///
/// Templates are written for Python's Jinja and run by llama.cpp's own
/// interpreter, which differs from Python's in small ways that change the
/// prompt: `{{ true }}` writes `True`, `{{ none }}` writes nothing, `|trim` strips
/// ASCII white space only, `|length` of a string counts bytes, `|tojson` writes
/// `{"a": 1, "b": [1, 2]}` like Python's `json.dumps`. This crate runs templates
/// the same way, and only those that stay inside what it has verified:
///
/// - statements: `if`, `elif`, `else`, `for` (with `loop`, `else`, an `if` filter and
///   unpacking), `set` (also of an attribute of a `namespace`, and unpacking), `macro`;
/// - expressions: literals, lists, indexing, `list[::-1]`, attributes, `+ - *`, `~`,
///   comparisons, `in`, `and`, `or`, `not`, `is`, `x if c else y`;
/// - filters `default`, `items`, `length`, `safe`, `string`, `tojson` (without
///   arguments) and `trim`; tests `defined`, `false`, `iterable`, `mapping`, `none`,
///   `sequence`, `string`, `true`, `undefined`;
/// - functions `namespace` and `raise_exception`, and the macros the template defines
///   (called with positional arguments);
/// - methods of strings (`startswith`, `endswith`, `split`, `rsplit`, `strip`, `lstrip`,
///   `rstrip`) and of mappings (`items`, `keys`, `values`, `get`).
///
/// Anything else makes [`ChatTemplate::new`] fail with [`ChatError::Unsupported`], naming
/// it and its line, if it is in the template where it can run; the other filters and
/// tests, other functions and methods, `//`, `/`, `%`, `**`, slices other than `[::-1]`,
/// dictionary literals, `with`, `call` and `filter` blocks, `{% set x %}` blocks and the
/// clock (`strftime_now`, `date_string`). It is a short list on purpose: each entry was
/// compared with llama.cpp on values of every type before it was allowed.
pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
    bos_token: String,
    eos_token: String,
}

impl PartialEq for ChatTemplate {
    /// Two templates are the same when they are the same text with the same
    /// special tokens.
    fn eq(&self, other: &Self) -> bool {
        (&self.source, &self.bos_token, &self.eos_token)
            == (&other.source, &other.bos_token, &other.eos_token)
    }
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate").field("source_bytes", &self.source.len()).finish()
    }
}

const NAME: &str = "chat";

impl ChatTemplate {
    /// Compiles a template. `bos_token` and `eos_token` are the texts the
    /// template gets as those variables; use the empty string for a model that
    /// has none.
    ///
    /// # Errors
    ///
    /// [`ChatError::Syntax`] if the template is not valid Jinja,
    /// [`ChatError::Unsupported`] if it uses something outside the supported
    /// dialect (the first thing it finds is named).
    pub fn new(source: &str, bos_token: &str, eos_token: &str) -> Result<Self, ChatError> {
        let mut env = jinja::environment();
        // A template that is not valid Jinja is reported by the parser below, with
        // its message; `prepare` leaves such a template alone.
        let compiled = jinja::prepare(source).map_err(|problems| {
            let first = problems.into_iter().next().expect("at least one problem");
            ChatError::Unsupported { what: first.what, line: first.line }
        })?;
        env.add_template_owned(NAME, compiled).map_err(|e| ChatError::Syntax(describe(&e)))?;
        Ok(Self {
            env,
            source: source.to_owned(),
            bos_token: bos_token.to_owned(),
            eos_token: eos_token.to_owned(),
        })
    }

    /// The text of the template.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Renders the conversation.
    ///
    /// # Errors
    ///
    /// [`ChatError::Invalid`] for a message the template cannot be given (a
    /// message with both text and parts, tool arguments that are not JSON), and
    /// [`ChatError::Render`] when the template raises an exception or fails.
    pub fn render(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &ChatOptions,
    ) -> Result<String, ChatError> {
        let mut variables: Vec<(String, Value)> = Vec::new();
        let mut set = |name: &str, value: Value| match variables.iter_mut().find(|(n, _)| n == name)
        {
            Some(slot) => slot.1 = value,
            None => variables.push((name.to_owned(), value)),
        };

        // llama.cpp maps `developer` to `system` unless the template deals with
        // the channels of gpt-oss, which this dialect does not.
        let developer_is_system = !self.source.contains("<|channel|>");
        let messages = messages
            .iter()
            .map(|m| message_value(m, developer_is_system))
            .collect::<Result<Vec<_>, _>>()?;
        set("messages", Value::from(messages));
        set("enable_thinking", Value::from(options.enable_thinking));
        if !tools.is_empty() {
            set("tools", Value::from(tools.iter().map(tool_value).collect::<Vec<_>>()));
        }

        // Settings that templates read under several names. llama.cpp sets the
        // aliases first, so a variable of the caller wins over them.
        for (name, value) in &options.variables {
            match (name.as_str(), value) {
                ("preserve_reasoning", Json::Bool(on)) => {
                    set("preserve_thinking", Value::from(*on));
                    set("clear_thinking", Value::from(!*on));
                    set("truncate_history_thinking", Value::from(!*on));
                    set("drop_thinking", Value::from(!*on));
                }
                ("reasoning_effort", Json::String(effort)) if !effort.is_empty() => {
                    set("reasoning_strength", Value::from(effort.as_str()));
                }
                _ => {}
            }
        }
        for (name, value) in &options.variables {
            set(name, json_value(value));
        }
        if options.add_generation_prompt {
            set("add_generation_prompt", Value::from(true));
        }
        // These come from the model, not from the caller.
        set("bos_token", Value::from(self.bos_token.as_str()));
        set("eos_token", Value::from(self.eos_token.as_str()));

        let template = self.env.get_template(NAME).map_err(|e| ChatError::Render(describe(&e)))?;
        template.render(Value::from_iter(variables)).map_err(|e| ChatError::Render(describe(&e)))
    }
}

/// The error with its causes, on one line each.
fn describe(error: &minijinja::Error) -> String {
    use std::error::Error as _;
    let mut text = match error.detail() {
        Some(detail) => format!("{}: {detail}", error.kind()),
        None => error.kind().to_string(),
    };
    if let Some(line) = error.line() {
        text.push_str(&format!(" (line {line})"));
    }
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(&format!("; {cause}"));
        source = cause.source();
    }
    text
}

/// A JSON value as a template value, keeping the order of keys.
pub(crate) fn json_value(json: &Json) -> Value {
    match json {
        Json::Null => Value::from(()),
        Json::Bool(b) => Value::from(*b),
        Json::Int(i) => Value::from(*i),
        Json::Float(f) => Value::from(*f),
        Json::String(s) => Value::from(s.as_str()),
        Json::Array(items) => Value::from(items.iter().map(json_value).collect::<Vec<_>>()),
        Json::Object(pairs) => {
            Value::from_iter(pairs.iter().map(|(k, v)| (k.as_str(), json_value(v))))
        }
    }
}

fn object<'a>(pairs: impl IntoIterator<Item = (&'a str, Value)>) -> Value {
    Value::from_iter(pairs)
}

/// The shape of a tool in the OpenAI API, which is the shape llama.cpp gives
/// templates: only the name, the description (empty if there is none) and the
/// parameters are kept.
fn tool_value(tool: &Tool) -> Value {
    object([
        ("type", Value::from("function")),
        (
            "function",
            object([
                ("name", Value::from(tool.name.as_str())),
                ("description", Value::from(tool.description.as_str())),
                ("parameters", json_value(&tool.parameters)),
            ]),
        ),
    ])
}

fn message_value(message: &Message, developer_is_system: bool) -> Result<Value, ChatError> {
    let role =
        if developer_is_system && message.role == "developer" { "system" } else { &message.role };
    let mut fields: Vec<(&str, Value)> = vec![("role", Value::from(role))];
    fields.push((
        "content",
        match &message.content {
            Content::Text(text) => Value::from(text.as_str()),
            Content::Parts(parts) if parts.is_empty() => Value::from(""),
            Content::Parts(parts) => Value::from(
                parts
                    .iter()
                    .map(|text| {
                        object([
                            ("type", Value::from("text")),
                            ("text", Value::from(text.as_str())),
                        ])
                    })
                    .collect::<Vec<_>>(),
            ),
        },
    ));
    if !message.reasoning_content.is_empty() {
        fields.push(("reasoning_content", Value::from(message.reasoning_content.as_str())));
    }
    if !message.name.is_empty() {
        fields.push(("name", Value::from(message.name.as_str())));
    }
    if !message.tool_call_id.is_empty() {
        fields.push(("tool_call_id", Value::from(message.tool_call_id.as_str())));
    }
    if !message.tool_calls.is_empty() {
        let calls = message
            .tool_calls
            .iter()
            .map(|call| {
                let arguments = Json::parse(&call.arguments).map_err(|e| {
                    ChatError::Invalid(format!(
                        "the arguments of the call of '{}' are not JSON: {e}",
                        call.name
                    ))
                })?;
                let mut fields = vec![
                    ("type", Value::from("function")),
                    (
                        "function",
                        object([
                            ("name", Value::from(call.name.as_str())),
                            ("arguments", json_value(&arguments)),
                        ]),
                    ),
                ];
                if let Some(id) = call.id.as_ref().filter(|id| !id.is_empty()) {
                    fields.push(("id", Value::from(id.as_str())));
                }
                Ok(object(fields))
            })
            .collect::<Result<Vec<_>, ChatError>>()?;
        fields.push(("tool_calls", Value::from(calls)));
    }
    Ok(object(fields))
}
