//! The chat template of a model as the front end needs to know it, the
//! variables a request adds to it, and the way from a request's prompt to the
//! model's tokens, for the architectures that share the Qwen chat template and
//! tokenizer.

use ares_api::{ChatFormat, EngineError, Prompt, Role, ThinkingFormat, ToolFormat};
use ares_tokenizer::{
    ChatOptions, EncodeOptions, Json, Message, ReasoningFormat, Tokenizer, Tool, ToolCall,
};

/// What the chat template of `tokenizer` declares when it is rendered with
/// `chat`, the options the server started the model with.
///
/// The template is read, not the name of the model: the tags that mark the
/// thinking, how the end of the prompt opens or closes the block, whether
/// the template reads `reasoning_effort`, and the tags it writes the calls of
/// tools with. Whether the model thinks when a request does not say is the
/// template's default with `chat` applied.
pub fn chat_format(tokenizer: &Tokenizer, chat: &ChatOptions) -> ChatFormat {
    ChatFormat {
        thinking: tokenizer.reasoning_format().map(|format| thinking_format(format, chat)),
        tools: tokenizer.tool_format().map(|format| ToolFormat {
            call_open: format.call_open,
            name_close: format.name_close,
            param_open: format.param_open,
            key_close: format.key_close,
            param_close: format.param_close,
            call_close: format.call_close,
            before_calls: format.before_calls,
            between_calls: format.between_calls,
        }),
    }
}

/// The thinking a template declares, for a model rendered with `chat`.
pub fn thinking_format(format: ReasoningFormat, chat: &ChatOptions) -> ThinkingFormat {
    let on_by_default =
        match chat.variables.iter().rev().find(|(name, _)| name == "enable_thinking") {
            Some((_, Json::Bool(on))) => *on,
            _ => chat.enable_thinking,
        };
    ThinkingFormat {
        on_by_default,
        effort: format.effort,
        start: format.start,
        end: format.end,
        prompt_on: format.prompt_on,
        prompt_off: format.prompt_off,
    }
}

/// `chat` with the variables of a request on top: a request's variable wins
/// over one of the same name that the server set.
pub fn with_variables(chat: &ChatOptions, variables: &[(String, ares_api::Json)]) -> ChatOptions {
    let mut options = chat.clone();
    for (name, value) in variables {
        options.variables.retain(|(known, _)| known != name);
        options.variables.push((name.clone(), template_value(value)));
    }
    options
}

/// A JSON value of the API as a value for the template.
fn template_value(value: &ares_api::Json) -> Json {
    use ares_api::Json as Api;
    match value {
        Api::Null => Json::Null,
        Api::Bool(b) => Json::Bool(*b),
        Api::Int(i) => Json::Int(*i),
        // Above `i64::MAX`, where the template's numbers become floats too.
        Api::Uint(u) => Json::Float(*u as f64),
        Api::Float(f) => Json::Float(*f),
        Api::String(s) => Json::String(s.clone()),
        Api::Array(items) => Json::Array(items.iter().map(template_value).collect()),
        Api::Object(members) => {
            Json::Object(members.iter().map(|(k, v)| (k.clone(), template_value(v))).collect())
        }
    }
}

/// The tokens of `prompt`: plain text as it is, a conversation through the
/// chat template of the model file with the variables of the request on top of
/// `chat`.
pub(crate) fn encode(
    tokenizer: &Tokenizer,
    chat: &ChatOptions,
    prompt: &Prompt,
) -> Result<Vec<u32>, EngineError> {
    match prompt {
        Prompt::Text(text) => {
            Ok(tokenizer.encode(text, EncodeOptions { add_special: true, parse_special: false }))
        }
        Prompt::Chat { messages, tools, variables } => {
            let invalid = |what: String| EngineError::InvalidRequest(what);
            let tools: Vec<Tool> =
                tools.iter().flatten().map(tool).collect::<Result<_, _>>().map_err(invalid)?;
            let messages: Vec<Message> = messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    };
                    let mut message = Message::new(role, m.content.clone());
                    message.name = m.name.clone().unwrap_or_default();
                    message.tool_call_id = m.tool_call_id.clone().unwrap_or_default();
                    message.reasoning_content = m.reasoning_content.clone().unwrap_or_default();
                    message.tool_calls =
                        m.tool_calls.iter().flatten().map(tool_call).collect::<Result<_, _>>()?;
                    Ok(message)
                })
                .collect::<Result<_, String>>()
                .map_err(invalid)?;
            let text = tokenizer
                .render_chat(&messages, &tools, &with_variables(chat, variables))
                .map_err(|e| EngineError::InvalidRequest(format!("chat template: {e}")))?;
            Ok(tokenizer.encode(&text, EncodeOptions { add_special: true, parse_special: true }))
        }
    }
}

/// A tool of a request, in the OpenAI form (`{"type": "function", "function":
/// {"name", "description", "parameters"}}`), as the template is given it.
fn tool(value: &ares_api::Json) -> Result<Tool, String> {
    let function = value.get("function").unwrap_or(value);
    let name = function.get("name").and_then(ares_api::Json::as_str);
    Ok(Tool {
        name: name.ok_or("a tool has no name")?.to_owned(),
        description: function
            .get("description")
            .and_then(ares_api::Json::as_str)
            .unwrap_or_default()
            .to_owned(),
        parameters: function.get("parameters").map_or(Json::Object(Vec::new()), template_value),
    })
}

/// A call of an assistant message, in the OpenAI form (`{"id", "type":
/// "function", "function": {"name", "arguments"}}`). The arguments are JSON
/// text there; an object is taken as well.
fn tool_call(value: &ares_api::Json) -> Result<ToolCall, String> {
    let function = value.get("function").unwrap_or(value);
    let name = function.get("name").and_then(ares_api::Json::as_str);
    let arguments = match function.get("arguments") {
        None | Some(ares_api::Json::Null) => "{}".to_owned(),
        Some(ares_api::Json::String(text)) if text.trim().is_empty() => "{}".to_owned(),
        Some(ares_api::Json::String(text)) => text.clone(),
        Some(other) => template_value(other).dump(),
    };
    Ok(ToolCall {
        id: value.get("id").and_then(ares_api::Json::as_str).map(str::to_owned),
        name: name.ok_or("a tool call has no name")?.to_owned(),
        arguments,
    })
}

/// Appends the bytes of the text of `token` to `text`.
pub(crate) fn piece(tokenizer: &Tokenizer, token: u32, text: &mut Vec<u8>) {
    if let Ok(bytes) = tokenizer.piece(token, false) {
        text.extend_from_slice(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_variable_wins_over_the_one_the_server_set() {
        let chat = ChatOptions {
            variables: vec![
                ("enable_thinking".into(), Json::Bool(false)),
                ("reasoning_effort".into(), Json::String("low".into())),
            ],
            ..ChatOptions::default()
        };
        let request = [
            ("enable_thinking".to_owned(), ares_api::Json::Bool(true)),
            ("extra".to_owned(), ares_api::Json::Uint(u64::MAX)),
        ];
        let merged = with_variables(&chat, &request);
        assert_eq!(
            merged.variables,
            vec![
                ("reasoning_effort".into(), Json::String("low".into())),
                ("enable_thinking".into(), Json::Bool(true)),
                ("extra".into(), Json::Float(u64::MAX as f64)),
            ]
        );
        // The server's options are not touched.
        assert_eq!(chat.variables.len(), 2);
    }
}
