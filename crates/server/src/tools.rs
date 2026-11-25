//! Tools in chat requests: `tools`, `tool_choice` and `parallel_tool_calls`.
//!
//! The model's chat template writes the tools into the prompt, and the model
//! calls one by writing the call into its reply the way the template writes
//! the calls of earlier turns. The engine reads that form off the template
//! ([`ToolFormat`]); a model whose template writes calls in a form the engine
//! does not know gets no tools, and a request that brings some is refused.
//!
//! The reply is held to the calls by a grammar, not trusted to be one
//! (`ares_grammar::calls`, after llama-server's grammar for the format): the
//! name is one of the tools of the request and the arguments follow the tool's
//! JSON schema, whether or not the tool says `strict`.
//!
//! | `tool_choice` | the reply |
//! |---|---|
//! | `auto` (the default) | free until the model opens a call; from there calls, and nothing after them |
//! | `required` | calls from the first token of the answer: at least one |
//! | a named function | the same, and every call is of that function |
//! | `none` | text; the tools are in the prompt, as with llama-server, and a call the model writes stays text |
//!
//! `parallel_tool_calls: false` allows one call. The reasoning of a model
//! that thinks is free in every case and the rule binds the answer after it.
//!
//! # A call inside the thinking
//!
//! A model that thinks sometimes opens a call without closing its reasoning
//! first. When calls are allowed, the marker that opens a call
//! ([`ToolFormat::trigger`]) ends the reasoning where it stands, as
//! llama-server has it for these templates: the text before it is the
//! reasoning, and what follows is a call under the same rule as a call after
//! the closing tag ([`Interrupt`] in the engine, the reasoning's splitter and
//! [`CallSplitter`] in the reply). The white space between the reasoning and
//! the marker is layout and belongs to neither. After such a call the rule
//! allows what it allows after any call (white space, another call, the end),
//! so the reply cannot write the closing tag any more, and nothing is missing
//! without it: the template writes the tag itself when the turn comes back.
//! With `required` or a named function the reply cannot end while it thinks
//! (its end tokens are not allowed there), so it ends with a call or at the
//! limit of tokens. With `none`, or without tools, the marker inside the
//! thinking is reasoning like any other text.
//!
//! Where this is not llama-server: `required` there lets text come before the
//! first call; a named function is read as `auto` there (its `tool_choice` is
//! a string); a `response_format` or `json_schema` next to tools wins over
//! them silently there and is refused here, as `grammar` is by both; the
//! marker ends the reasoning there without tools and with `none` too (the
//! rest of the thinking is then content), and the white space before it
//! stays in `reasoning_content`.

mod split;

use std::collections::HashMap;

use ares_api::{Engine, Interrupt, Json, Shape, Start, Structure, ToolFormat};
use ares_grammar::calls::{CallRule, Callable};
use serde_json::Value;

use crate::chat_body::cache_control;
use crate::error::ApiError;
use crate::openai::request::{Fields, Parsed, wrong_type};
use crate::reasoning::Applied;

pub(crate) use split::{CallSplitter, Part};

/// The members of a request body that are about tools.
pub(crate) const FIELDS: &[&str] = &["tools", "tool_choice", "parallel_tool_calls"];

/// A tool of a request.
#[derive(Debug, Clone, PartialEq)]
struct Tool {
    name: String,
    /// The JSON schema of its arguments.
    parameters: Option<Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum Choice {
    #[default]
    Auto,
    None,
    Required,
    Named(String),
}

/// What a request says about tools.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Asked {
    tools: Vec<Tool>,
    choice: Choice,
    /// Whether a reply may hold several calls.
    parallel: bool,
}

/// Reads the fields of a chat request that are about tools.
pub(crate) fn asked(fields: &Fields<'_>) -> Result<Asked, ApiError> {
    let tools = match fields.get("tools") {
        None => Vec::new(),
        Some(Json::Array(tools)) => {
            tools.iter().enumerate().map(|(i, t)| tool(i, t)).collect::<Result<Vec<_>, _>>()?
        }
        Some(other) => return Err(wrong_type("tools", "an array", other)),
    };
    for (i, tool) in tools.iter().enumerate() {
        if tools[..i].iter().any(|earlier| earlier.name == tool.name) {
            return Err(ApiError::invalid_param(
                "tools",
                format!("tools[{i}]: there are two tools named '{}'", tool.name),
            ));
        }
    }
    let choice = match fields.get("tool_choice") {
        None => Choice::Auto,
        Some(Json::String(choice)) => match choice.as_str() {
            "auto" => Choice::Auto,
            "none" => Choice::None,
            "required" => Choice::Required,
            other => {
                return Err(ApiError::invalid_param(
                    "tool_choice",
                    format!(
                        "tool_choice must be \"auto\", \"none\", \"required\" or a function, \
                         got \"{other}\""
                    ),
                ));
            }
        },
        Some(named @ Json::Object(_)) => {
            let name = named.get("function").and_then(|f| f.get("name")).and_then(Json::as_str);
            match (named.get("type").and_then(Json::as_str), name) {
                (Some("function"), Some(name)) => Choice::Named(name.to_owned()),
                _ => {
                    return Err(ApiError::invalid_param(
                        "tool_choice",
                        "tool_choice as an object must be {\"type\": \"function\", \"function\": \
                         {\"name\": ...}}",
                    ));
                }
            }
        }
        Some(other) => return Err(wrong_type("tool_choice", "a string or an object", other)),
    };
    match &choice {
        Choice::Required if tools.is_empty() => {
            return Err(ApiError::invalid_param(
                "tool_choice",
                "tool_choice is \"required\" and the request has no tools",
            ));
        }
        Choice::Named(name) if !tools.iter().any(|tool| tool.name == *name) => {
            return Err(ApiError::invalid_param(
                "tool_choice",
                format!("tool_choice names the function '{name}', which is not among the tools"),
            ));
        }
        _ => {}
    }
    let parallel = match fields.get("parallel_tool_calls") {
        None => true,
        Some(Json::Bool(parallel)) => *parallel,
        Some(other) => return Err(wrong_type("parallel_tool_calls", "a boolean", other)),
    };
    Ok(Asked { tools, choice, parallel })
}

/// One entry of `tools`: `{"type": "function", "function": {"name",
/// "description", "parameters", "strict"}}`, and a cache marker
/// (`cache_control`), which changes nothing. Other members are refused.
fn tool(index: usize, value: &Json) -> Result<Tool, ApiError> {
    let at = format!("tools[{index}]");
    let refuse = |message: String| ApiError::invalid_param("tools", message);
    let Json::Object(members) = value else {
        return Err(refuse(format!("{at} must be an object, got {}", value.kind())));
    };
    match value.get("type").and_then(Json::as_str) {
        Some("function") => {}
        other => {
            return Err(refuse(format!(
                "{at}.type must be \"function\", got {}: other kinds of tools are not served",
                other.map_or("nothing".to_owned(), |kind| format!("\"{kind}\""))
            )));
        }
    }
    unknown(&at, members, &["type", "function", "cache_control"])?;
    if let Some(marker) = Fields(members).get("cache_control") {
        cache_control(&format!("{at}.cache_control"), marker).map_err(|e| refuse(e.message))?;
    }
    let Some(function @ Json::Object(inner)) = value.get("function") else {
        return Err(refuse(format!("{at}.function must be an object")));
    };
    unknown(&format!("{at}.function"), inner, &["name", "description", "parameters", "strict"])?;
    let name = match function.get("name") {
        Some(Json::String(name)) if !name.is_empty() => name.clone(),
        _ => return Err(refuse(format!("{at}.function.name must be a string that is not empty"))),
    };
    if !matches!(function.get("description"), None | Some(Json::Null | Json::String(_))) {
        return Err(refuse(format!("{at}.function.description must be a string")));
    }
    // The arguments are held to the schema either way.
    if !matches!(function.get("strict"), None | Some(Json::Null | Json::Bool(_))) {
        return Err(refuse(format!("{at}.function.strict must be a boolean")));
    }
    let parameters = match function.get("parameters") {
        None | Some(Json::Null) => None,
        Some(schema @ Json::Object(_)) => Some(
            serde_json::to_value(schema)
                .map_err(|e| refuse(format!("{at}.function.parameters cannot be read: {e}")))?,
        ),
        Some(other) => {
            return Err(refuse(format!(
                "{at}.function.parameters must be an object, got {}",
                other.kind()
            )));
        }
    };
    Ok(Tool { name, parameters })
}

/// Refuses a member of the object at `at` of a tool that is not one of
/// `names`. A member that is `null` is not set.
fn unknown(at: &str, members: &[(String, Json)], names: &[&str]) -> Result<(), ApiError> {
    let Some((name, _)) =
        members.iter().find(|(name, value)| !value.is_null() && !names.contains(&name.as_str()))
    else {
        return Ok(());
    };
    let why = match name.as_str() {
        "defer_loading" => "every tool of a request is written into the prompt",
        _ => "it is not a member of a tool this server reads",
    };
    Err(ApiError::invalid_param("tools", format!("{at}.{name} is not supported: {why}")))
}

/// Applies what `parsed` asks about tools to its request: the rule its reply
/// has to follow. Returns the splitter that finds the calls in the reply, or
/// `None` for a reply that is text. `thinking` is what the request came to
/// for its reasoning.
pub(crate) fn apply(
    parsed: &mut Parsed,
    engine: &dyn Engine,
    thinking: &Applied,
) -> Result<Option<CallSplitter>, ApiError> {
    let asked = std::mem::take(&mut parsed.tools);
    // llama-server: with `none` the tools stay in the prompt and the reply
    // is text.
    if asked.tools.is_empty() || asked.choice == Choice::None {
        return Ok(None);
    }
    let request = &mut parsed.request;
    let Some(format) = engine.chat_format(&request.model).and_then(|format| format.tools) else {
        return Err(ApiError::invalid_param(
            "tools",
            "tools are not supported with this model: its chat template writes the calls of \
             tools in no form this server reads (it reads calls written as \
             <function=NAME><parameter=KEY>VALUE</parameter></function> between tags)",
        ));
    };
    if request.structure.is_some() {
        return Err(ApiError::invalid_param(
            "tools",
            "tools cannot be combined with grammar, json_schema or response_format: the reply \
             follows one rule. Send tool_choice \"none\" to have the tools in the prompt and \
             the reply follow the other rule",
        ));
    }

    let offered: Vec<&Tool> = match &asked.choice {
        Choice::Named(name) => asked.tools.iter().filter(|tool| tool.name == *name).collect(),
        _ => asked.tools.iter().collect(),
    };
    let callable: Vec<Callable<'_>> = offered
        .iter()
        .map(|tool| Callable { name: &tool.name, parameters: tool.parameters.as_ref() })
        .collect();
    let triggered = asked.choice == Choice::Auto;
    let rule = |triggered: bool| {
        CallRule { format: &format, tools: &callable, parallel: asked.parallel, triggered }
            .lark()
            .map(Shape::Lark)
            .map_err(|e| ApiError::invalid_param("tools", format!("tools: {e}")))
    };
    let shape = rule(triggered)?;
    // A reply that thinks may open a call before it closes its reasoning:
    // the marker ends the reasoning, and the rule of a call that has been
    // opened binds what follows.
    let interrupt = match &thinking.start {
        Start::Reply => None,
        Start::AfterReasoning { .. } => Some(Interrupt {
            text: format.trigger().to_owned(),
            shape: if triggered { shape.clone() } else { rule(true)? },
        }),
    };
    request.structure = Some(Structure {
        shape,
        start: thinking.start.clone(),
        trigger: triggered.then(|| format.trigger().to_owned()),
        interrupt,
    });
    Ok(Some(splitter(format, &asked.tools, request.id)))
}

fn splitter(format: ToolFormat, tools: &[Tool], id: ares_api::RequestId) -> CallSplitter {
    let schemas: HashMap<String, Option<Value>> =
        tools.iter().map(|tool| (tool.name.clone(), tool.parameters.clone())).collect();
    CallSplitter::new(format, schemas, call_ids(id))
}

/// What the ids of the calls of the request `id` start with.
pub(crate) fn call_ids(id: ares_api::RequestId) -> String {
    format!("call_{id}_")
}

#[cfg(test)]
mod templates;
#[cfg(test)]
mod tests;
