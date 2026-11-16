//! The chat template of a model as the front end needs to know it, the
//! variables a request adds to it, and the way from a request's prompt to the
//! model's tokens, for the architectures that share the Qwen chat template and
//! tokenizer.

use ares_api::{ChatFormat, EngineError, Prompt, Role, ThinkingFormat, ToolFormat};
use ares_tokenizer::{
    ChatOptions, Content, EncodeOptions, Json, Message, ReasoningFormat, Tokenizer, Tool, ToolCall,
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
    // A template the server was started with takes the place of the file's.
    let (reasoning, tools) = match &chat.template {
        Some(template) => {
            (ReasoningFormat::detect(template), ares_tokenizer::ToolFormat::detect(template))
        }
        None => (tokenizer.reasoning_format(), tokenizer.tool_format()),
    };
    ChatFormat {
        thinking: reasoning.map(|format| thinking_format(format, chat)),
        tools: tools.map(|format| ToolFormat {
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
        Prompt::Chat { messages, tools, variables, continue_final } => {
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
            let options = with_variables(chat, variables);
            let text = if *continue_final {
                continued(tokenizer, messages, &tools, options)?
            } else {
                tokenizer
                    .render_chat(&messages, &tools, &options)
                    .map_err(|e| EngineError::InvalidRequest(format!("chat template: {e}")))?
            };
            Ok(tokenizer.encode(&text, EncodeOptions { add_special: true, parse_special: true }))
        }
    }
}

/// What marks the place where the reply goes on, in a conversation that is
/// rendered to be continued: characters of the private use area, which no
/// text of a request has (one that does is refused).
const MARK: &str = "\u{f8ff}ares-continues-here\u{f8ff}";

/// The prompt of a conversation whose last message, the assistant's, the
/// reply continues: the conversation as the template writes it, up to the
/// end of that message's text. That is its content, or its reasoning when
/// the content is empty and the reasoning is not; the reply then goes on
/// thinking. The template writes the turn as it writes every turn (an empty
/// block of thinking before a content without reasoning, in Qwen3.8's
/// template), and the prompt ends where the message's text does: a mark is
/// put at its end and the rendering cut there, so that what the template
/// writes after the text (the end of the turn) is left out.
fn continued(
    tokenizer: &Tokenizer,
    mut messages: Vec<Message>,
    tools: &[Tool],
    mut options: ChatOptions,
) -> Result<String, EngineError> {
    let invalid = |what: &str| EngineError::InvalidRequest(what.to_owned());
    if !messages.last().is_some_and(|m| m.role == "assistant") {
        return Err(invalid(
            "the last message is not the assistant's: there is no message to continue",
        ));
    }
    let text = |content: &Content| match content {
        Content::Text(text) => text.clone(),
        Content::Parts(parts) => parts.concat(),
    };
    let mentioned =
        |m: &Message| text(&m.content).contains(MARK) || m.reasoning_content.contains(MARK);
    if messages.iter().any(mentioned) {
        return Err(invalid("a message holds characters this server marks a prompt with"));
    }
    let last = messages.last_mut().expect("there is a last message");
    let thinking = text(&last.content).is_empty() && !last.reasoning_content.is_empty();
    if thinking {
        last.reasoning_content.push_str(MARK);
    } else {
        match &mut last.content {
            Content::Text(text) => text.push_str(MARK),
            Content::Parts(parts) => match parts.last_mut() {
                Some(part) => part.push_str(MARK),
                None => parts.push(MARK.to_owned()),
            },
        }
    }
    options.add_generation_prompt = false;
    let text = tokenizer
        .render_chat(&messages, tools, &options)
        .map_err(|e| EngineError::InvalidRequest(format!("chat template: {e}")))?;
    let mut marks = text.match_indices(MARK);
    match (marks.next(), marks.next()) {
        (Some((at, _)), None) => Ok(text[..at].to_owned()),
        _ => Err(invalid(if thinking {
            "the chat template does not write the reasoning of the last message as it is given, \
             so the reply cannot continue it"
        } else {
            "the chat template does not write the last message as it is given, so the reply \
             cannot continue it"
        })),
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

    /// The tools of a request, the calls of an assistant message and their
    /// results come out of the chat template of a model file, when one is
    /// named: `ARES_TEST_GGUF` (Qwen3.5-0.8B), `ARES_TEST_QWEN38_27B` or
    /// `ARES_TEST_FLASH_NEXT`. Reading the metadata of a file needs no GPU.
    #[test]
    fn tools_calls_and_results_reach_the_template_of_a_model_file() {
        use ares_api::Json as Api;
        let object = |members: &[(&str, Api)]| {
            Api::Object(members.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect())
        };
        let text = |value: &str| Api::String(value.to_owned());
        let schema = object(&[
            ("type", text("object")),
            (
                "properties",
                object(&[
                    ("city", object(&[("type", text("string"))])),
                    ("days", object(&[("type", text("integer"))])),
                ]),
            ),
        ]);
        let function = object(&[("name", text("get_weather")), ("parameters", schema)]);
        let tools = vec![object(&[("type", text("function")), ("function", function)])];
        // The arguments as JSON text, which is OpenAI's form, and as an object.
        let call = |arguments: Api| {
            let function = object(&[("name", text("get_weather")), ("arguments", arguments)]);
            object(&[("id", text("call_1")), ("type", text("function")), ("function", function)])
        };
        let calls = vec![
            call(text(r#"{"city": "Köln", "days": 3}"#)),
            call(object(&[("city", text("Bonn"))])),
        ];
        let mut assistant = ares_api::Message::new(Role::Assistant, "");
        assistant.tool_calls = Some(calls);
        let mut result = ares_api::Message::new(Role::Tool, "12 degrees");
        result.tool_call_id = Some("call_1".to_owned());
        let prompt = Prompt::Chat {
            messages: vec![ares_api::Message::new(Role::User, "The weather?"), assistant, result],
            tools: Some(tools),
            variables: Vec::new(),
            continue_final: false,
        };
        for variable in ["ARES_TEST_GGUF", "ARES_TEST_QWEN38_27B", "ARES_TEST_FLASH_NEXT"] {
            let Some(path) = std::env::var_os(variable) else {
                eprintln!("skipped: set {variable} to a model file");
                continue;
            };
            let tokenizer = Tokenizer::from_gguf(&ares_gguf::Gguf::open(path).unwrap()).unwrap();
            let chat = ChatOptions::default();
            let format = chat_format(&tokenizer, &chat).tools.expect("a format for calls");
            assert_eq!(format.trigger(), "<tool_call>", "{variable}");
            let tokens = encode(&tokenizer, &chat, &prompt).unwrap();
            let text = String::from_utf8(tokenizer.decode(&tokens, true).unwrap()).unwrap();
            assert!(text.contains("<tools>\n{\"type\": \"function\""), "{variable}: {text}");
            let calls = "<tool_call>\n<function=get_weather>\n<parameter=city>\nKöln\n\
                         </parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n\
                         </tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\n\
                         Bonn\n</parameter>\n</function>\n</tool_call><|im_end|>";
            assert!(text.contains(calls), "{variable}: {text}");
            assert!(text.contains("<tool_response>\n12 degrees\n</tool_response>"), "{variable}");
        }
    }

    /// The tokenizers of the model files the tests name, with the name of
    /// the variable: `ARES_TEST_GGUF` (Qwen3.5-0.8B), `ARES_TEST_QWEN38_27B`,
    /// `ARES_TEST_FLASH_NEXT`. Reading the metadata of a file needs no GPU.
    fn model_tokenizers() -> Vec<(&'static str, Tokenizer)> {
        let mut found = Vec::new();
        for variable in ["ARES_TEST_GGUF", "ARES_TEST_QWEN38_27B", "ARES_TEST_FLASH_NEXT"] {
            match std::env::var_os(variable) {
                Some(path) => found.push((
                    variable,
                    Tokenizer::from_gguf(&ares_gguf::Gguf::open(path).unwrap()).unwrap(),
                )),
                None => eprintln!("skipped: set {variable} to a model file"),
            }
        }
        found
    }

    fn rendered(tokenizer: &Tokenizer, chat: &ChatOptions, prompt: &Prompt) -> String {
        let tokens = encode(tokenizer, chat, prompt).unwrap();
        String::from_utf8(tokenizer.decode(&tokens, true).unwrap()).unwrap()
    }

    #[test]
    fn a_continued_message_ends_the_prompt_where_its_text_does() {
        let user = ares_api::Message::new(Role::User, "Write a haiku.");
        let conversation = |content: &str, reasoning: Option<&str>, continue_final: bool| {
            let mut assistant = ares_api::Message::new(Role::Assistant, content);
            assistant.reasoning_content = reasoning.map(str::to_owned);
            Prompt::Chat {
                messages: vec![user.clone(), assistant],
                tools: None,
                variables: Vec::new(),
                continue_final,
            }
        };
        for (variable, tokenizer) in model_tokenizers() {
            let chat = ChatOptions::default();
            // The answer goes on after its text, as a whole turn has it.
            let whole = rendered(&tokenizer, &chat, &conversation("Autumn ", None, false));
            let continued = rendered(&tokenizer, &chat, &conversation("Autumn ", None, true));
            let turn =
                |text: &str| text[text.rfind("<|im_start|>assistant\n").unwrap()..].to_owned();
            assert!(continued.ends_with("Autumn "), "{variable}: {continued:?}");
            assert!(!turn(&continued).contains("<|im_end|>"), "{variable}: {continued:?}");
            assert!(whole.starts_with(continued.trim_end()), "{variable}: {whole:?}");
            // Reasoning without an answer goes on thinking.
            let thinking = conversation("", Some("The user wants"), true);
            let continued = rendered(&tokenizer, &chat, &thinking);
            assert!(continued.ends_with("The user wants"), "{variable}: {continued:?}");
            assert!(turn(&continued).contains("<think>"), "{variable}: {continued:?}");
            assert!(!turn(&continued).contains("</think>"), "{variable}: {continued:?}");
            // The tags of the thinking print, so the engine finds them in a prompt.
            for tag in ["<think>", "</think>"] {
                let id = tokenizer
                    .encode(tag, EncodeOptions { add_special: false, parse_special: true });
                let mut text = Vec::new();
                piece(&tokenizer, id[0], &mut text);
                assert_eq!((id.len(), text.as_slice()), (1, tag.as_bytes()), "{variable}");
            }
            // A last message that is not the assistant's has nothing to continue.
            let prompt = Prompt::Chat {
                messages: vec![user.clone()],
                tools: None,
                variables: Vec::new(),
                continue_final: true,
            };
            assert!(matches!(
                encode(&tokenizer, &chat, &prompt),
                Err(EngineError::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn a_template_of_a_file_takes_the_place_of_the_models() {
        let source = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}\
                      {% if add_generation_prompt %}[assistant] {% endif %}";
        for (variable, tokenizer) in model_tokenizers() {
            let template = tokenizer.compile_template(source).unwrap();
            let chat = ChatOptions {
                template: Some(std::sync::Arc::new(template)),
                ..ChatOptions::default()
            };
            let prompt = Prompt::Chat {
                messages: vec![ares_api::Message::new(Role::User, "hi")],
                tools: None,
                variables: Vec::new(),
                continue_final: false,
            };
            assert_eq!(
                rendered(&tokenizer, &chat, &prompt),
                "[user] hi\n[assistant] ",
                "{variable}"
            );
            // What is read off a template is read off this one: it has no thinking.
            let format = chat_format(&tokenizer, &chat);
            assert_eq!(format.thinking, None, "{variable}");
            assert!(chat_format(&tokenizer, &ChatOptions::default()).thinking.is_some());
            assert!(tokenizer.compile_template("{% for %}").is_err());
        }
    }

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
