//! The chat template of a model as the front end needs to know it, the
//! variables a request adds to it, and the way from a request's prompt to the
//! model's tokens, for the architectures that share the Qwen chat template and
//! tokenizer.

use std::sync::Arc;

use ares_api::{ChatFormat, EngineError, Prompt, Role, ThinkingFormat, ToolFormat};
use ares_runtime::{Boundary, EncodedPrompt, ImageInput, ImageSpan};
use ares_tokenizer::{
    ChatOptions, Content, EncodeOptions, Json, Message, ReasoningFormat, Tokenizer, Tool, ToolCall,
};

use crate::qwen3vl::Projector;

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
/// `chat`. An image takes as many cells as it has rows, by the size rule of
/// `projector`, without being resized.
pub(crate) fn encode(
    tokenizer: &Tokenizer,
    chat: &ChatOptions,
    projector: Option<&Arc<Projector>>,
    prompt: &Prompt,
) -> Result<Vec<u32>, EngineError> {
    encode_prompt(tokenizer, chat, projector, prompt, false).map(|encoded| encoded.tokens)
}

/// The tokens of `prompt` as [`encode`] gives them, for a conversation the
/// places where its parts end ([`Boundary`]), read off the chat template
/// itself (see [`boundaries`]), and its images as the tower takes them.
pub(crate) fn encode_with_boundaries(
    tokenizer: &Tokenizer,
    chat: &ChatOptions,
    projector: Option<&Arc<Projector>>,
    prompt: &Prompt,
) -> Result<EncodedPrompt, EngineError> {
    encode_prompt(tokenizer, chat, projector, prompt, true)
}

/// What stands for an image in a message's text while the template renders
/// it: a character of the private use area (another one than [`MARK`]'s and
/// [`BOUNDARY`]'s), which no text of a request has (one that does is refused).
const IMAGE: char = '\u{f8fd}';

/// The text of the markers around an image's rows, as llama.cpp's `mtmd`
/// writes them for Qwen3-VL's projector, with one pad token where the rows go.
const IMAGE_START: &str = "<|vision_start|>";
const IMAGE_PAD: &str = "<|image_pad|>";
const IMAGE_END: &str = "<|vision_end|>";

/// A message's text with [`IMAGE`] at the place of each of its images.
fn with_image_marks(message: &ares_api::Message) -> Result<String, EngineError> {
    if message.content.contains(IMAGE) {
        return Err(EngineError::InvalidRequest(
            "a message holds a character this server marks the places of images with".into(),
        ));
    }
    let mut text = String::with_capacity(message.content.len() + message.images.len() * 3);
    let mut from = 0;
    for image in &message.images {
        let at = image.at.min(message.content.len());
        if at < from || !message.content.is_char_boundary(at) {
            return Err(EngineError::InvalidRequest(
                "the images of a message are not in the order of its text".into(),
            ));
        }
        text.push_str(&message.content[from..at]);
        text.push(IMAGE);
        from = at;
    }
    text.push_str(&message.content[from..]);
    Ok(text)
}

/// The tokens of a rendering with [`IMAGE`] at each image: the text between
/// the images tokenized piece by piece, as llama.cpp's `mtmd` does, and for
/// each image its start marker, a pad token per row and its end marker. Also
/// the tokens with one pad per image (whose text is the rendering with the
/// markers in place of [`IMAGE`]) and where each such pad is, for the
/// boundaries.
struct WithImages {
    tokens: Vec<u32>,
    spans: Vec<ImageSpan>,
    single: Vec<u32>,
    pads: Vec<(usize, u32)>,
}

fn tokens_with_images(
    tokenizer: &Tokenizer,
    text: &str,
    images: &[(u32, Option<Arc<ImageInput>>)],
) -> Result<WithImages, EngineError> {
    let id = |text: &str| {
        tokenizer.special_token_id(text).ok_or_else(|| {
            EngineError::InvalidRequest(format!(
                "the model's vocabulary has no token {text}, which marks an image"
            ))
        })
    };
    let (start, pad, end) = (id(IMAGE_START)?, id(IMAGE_PAD)?, id(IMAGE_END)?);
    let pieces: Vec<&str> = text.split(IMAGE).collect();
    if pieces.len() != images.len() + 1 {
        return Err(EngineError::InvalidRequest(
            "the chat template does not write every image of the conversation once".into(),
        ));
    }
    let mut out =
        WithImages { tokens: Vec::new(), spans: Vec::new(), single: Vec::new(), pads: Vec::new() };
    for (i, piece) in pieces.iter().enumerate() {
        let options = EncodeOptions { add_special: i == 0, parse_special: true };
        let piece = tokenizer.encode(piece, options);
        out.tokens.extend_from_slice(&piece);
        out.single.extend_from_slice(&piece);
        let Some((rows, image)) = images.get(i) else { continue };
        out.tokens.push(start);
        if let Some(image) = image {
            out.spans.push(ImageSpan { first: out.tokens.len() as u32, image: image.clone() });
        }
        out.tokens.extend(std::iter::repeat_n(pad, *rows as usize));
        out.tokens.push(end);
        out.single.push(start);
        out.pads.push((out.single.len(), *rows));
        out.single.extend([pad, end]);
    }
    Ok(out)
}

fn encode_prompt(
    tokenizer: &Tokenizer,
    chat: &ChatOptions,
    projector: Option<&Arc<Projector>>,
    prompt: &Prompt,
    with_boundaries: bool,
) -> Result<EncodedPrompt, EngineError> {
    match prompt {
        Prompt::Text(text) => Ok(EncodedPrompt {
            tokens: tokenizer
                .encode(text, EncodeOptions { add_special: true, parse_special: false }),
            ..EncodedPrompt::default()
        }),
        Prompt::Chat { messages, tools, variables, continue_final } => {
            let invalid = |what: String| EngineError::InvalidRequest(what);
            let tools: Vec<Tool> =
                tools.iter().flatten().map(tool).collect::<Result<_, _>>().map_err(invalid)?;
            // The images, in the order of the conversation, with their rows: by
            // the size rule only when they are counted, resized and named when
            // they are to run.
            let images: Vec<(u32, Option<Arc<ImageInput>>)> = {
                let all: Vec<&ares_api::Image> = messages.iter().flat_map(|m| &m.images).collect();
                if !all.is_empty() && projector.is_none() {
                    return Err(EngineError::InvalidRequest(
                        "image input needs a projector, and this model was loaded without one                          (--mmproj)"
                            .into(),
                    ));
                }
                if messages.iter().any(|m| m.role == Role::System && !m.images.is_empty()) {
                    return Err(EngineError::InvalidRequest(
                        "a system message cannot hold images".into(),
                    ));
                }
                let image_error = |e: crate::qwen3vl::Error| {
                    EngineError::InvalidRequest(format!("an image: {e}"))
                };
                all.iter()
                    .map(|image| {
                        let projector = projector.expect("checked above");
                        if with_boundaries {
                            let input = projector
                                .prepare(&image.pixels, image.width, image.height)
                                .map_err(image_error)?;
                            Ok((input.rows(), Some(Arc::new(input))))
                        } else {
                            let (nx, ny) =
                                projector.grid(image.width, image.height).map_err(image_error)?;
                            Ok((nx * ny, None))
                        }
                    })
                    .collect::<Result<_, EngineError>>()?
            };
            let messages: Vec<Message> = messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    };
                    let content = if m.images.is_empty() {
                        m.content.clone()
                    } else {
                        with_image_marks(m).map_err(|e| e.to_string())?
                    };
                    let mut message = Message::new(role, content);
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
            let render = |messages: Vec<Message>| {
                if *continue_final {
                    continued(tokenizer, messages, &tools, options.clone())
                } else {
                    tokenizer
                        .render_chat(&messages, &tools, &options)
                        .map_err(|e| EngineError::InvalidRequest(format!("chat template: {e}")))
                }
            };
            let special = EncodeOptions { add_special: true, parse_special: true };
            if !images.is_empty() {
                let text = render(messages.clone())?;
                let encoded = tokens_with_images(tokenizer, &text, &images)?;
                if !with_boundaries {
                    return Ok(EncodedPrompt {
                        tokens: encoded.tokens,
                        ..EncodedPrompt::default()
                    });
                }
                // The boundaries of the rendering with one pad token per image,
                // moved behind the rows of the images before them.
                let markers = format!("{IMAGE_START}{IMAGE_PAD}{IMAGE_END}");
                let with_markers = |messages: Vec<Message>| {
                    render(messages).map(|text| text.replace(IMAGE, &markers))
                };
                let single_text = text.replace(IMAGE, &markers);
                let mut places = boundaries(
                    tokenizer,
                    &messages,
                    *continue_final,
                    &single_text,
                    &encoded.single,
                    with_markers,
                );
                for place in &mut places {
                    let before: u32 = encoded
                        .pads
                        .iter()
                        .filter(|(at, _)| (*at as u64) < place.at)
                        .map(|(_, rows)| rows - 1)
                        .sum();
                    place.at += u64::from(before);
                }
                return Ok(EncodedPrompt {
                    tokens: encoded.tokens,
                    boundaries: places,
                    images: encoded.spans,
                });
            }
            if !with_boundaries {
                let tokens = tokenizer.encode(render(messages)?, special);
                return Ok(EncodedPrompt { tokens, ..EncodedPrompt::default() });
            }
            let text = render(messages.clone())?;
            let tokens = tokenizer.encode(&text, special);
            let boundaries =
                boundaries(tokenizer, &messages, *continue_final, &text, &tokens, render);
            Ok(EncodedPrompt { tokens, boundaries, images: Vec::new() })
        }
    }
}

/// What marks where the text of a message starts and ends in a rendering of
/// a conversation: a character of the private use area (another one than
/// [`MARK`]'s), the message's number, the character again.
const BOUNDARY: char = '\u{f8fe}';

/// Where the parts of a conversation end in its prompt `tokens`, rendered as
/// `text`: the start and the end of the text of every message, and among
/// them the roots, the start and the end of a system prompt and the start of
/// the first message after it. A template that writes the tools before the
/// system prompt (Qwen3.5's and Qwen3.8's) ends the tool list where the
/// system prompt starts; one that writes them after it, where the first
/// message starts.
///
/// The template itself says where: every message whose text is not empty is
/// rendered again with a mark before and after its text (inside its white
/// space at either end, which templates trim), and the marks are found in
/// that text. Their places count only when the text without them is the
/// prompt's text to the byte: a template that reads a message's text
/// otherwise than writing it (a test whether it is empty, a split of it)
/// renders something else with the marks, and the prompt then has no
/// boundaries. A mark the template does not write once is left out. A place
/// inside a token counts as the end of the token before it.
fn boundaries(
    tokenizer: &Tokenizer,
    messages: &[Message],
    continue_final: bool,
    text: &str,
    tokens: &[u32],
    render: impl Fn(Vec<Message>) -> Result<String, EngineError>,
) -> Vec<Boundary> {
    let content = |m: &Message| match &m.content {
        Content::Text(text) => Some(text.clone()),
        Content::Parts(_) => None,
    };
    if messages.iter().any(|m| content(m).is_none_or(|text| text.contains(BOUNDARY))) {
        return Vec::new();
    }
    // The message a reply continues ends the prompt inside its text.
    let marked_count = messages.len().saturating_sub(usize::from(continue_final));
    let mut marked = messages.to_vec();
    for (i, message) in marked.iter_mut().enumerate().take(marked_count) {
        let Content::Text(text) = &message.content else { continue };
        let core = text.trim();
        if core.is_empty() {
            continue;
        }
        let lead = &text[..text.len() - text.trim_start().len()];
        let trail = &text[text.trim_end().len()..];
        message.content = Content::Text(format!(
            "{lead}{BOUNDARY}{}{BOUNDARY}{core}{BOUNDARY}{}{BOUNDARY}{trail}",
            2 * i,
            2 * i + 1
        ));
    }
    let Ok(rendered) = render(marked) else { return Vec::new() };
    // The text without the marks, and the byte in it where each mark was.
    let mut plain = String::with_capacity(rendered.len());
    let mut found: Vec<Option<usize>> = vec![None; 2 * marked_count];
    let mut twice = vec![false; 2 * marked_count];
    let mut rest = rendered.as_str();
    while let Some(at) = rest.find(BOUNDARY) {
        plain.push_str(&rest[..at]);
        let after = &rest[at + BOUNDARY.len_utf8()..];
        let Some(end) = after.find(BOUNDARY) else { return Vec::new() };
        let Some(number) = after[..end].parse::<usize>().ok().filter(|n| *n < found.len()) else {
            return Vec::new();
        };
        twice[number] |= found[number].is_some();
        found[number] = Some(plain.len());
        rest = &after[end + BOUNDARY.len_utf8()..];
    }
    plain.push_str(rest);
    if plain != text {
        return Vec::new();
    }
    // The byte where each token ends.
    let skip = usize::from(tokenizer.adds_bos() && !tokens.is_empty());
    let mut ends = Vec::with_capacity(tokens.len() + 1);
    ends.push(0usize);
    for &id in &tokens[skip..] {
        let Ok(piece) = tokenizer.piece(id, true) else { return Vec::new() };
        ends.push(ends.last().copied().unwrap_or(0) + piece.len());
    }
    if *ends.last().unwrap_or(&0) != text.len() {
        return Vec::new();
    }
    let position = |byte: usize| (skip + ends.partition_point(|end| *end <= byte) - 1) as u64;
    let system = messages.iter().take_while(|m| m.role == "system").count();
    let mut places: Vec<Boundary> = found
        .iter()
        .enumerate()
        .filter(|(number, at)| at.is_some() && !twice[*number])
        .map(|(number, at)| {
            let message = number / 2;
            let start = number % 2 == 0;
            Boundary {
                at: position(at.expect("found")),
                root: message < system || (message == system && start),
            }
        })
        .collect();
    places.sort_by_key(|b| (b.at, !b.root));
    places.dedup_by_key(|b| b.at);
    places
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

    /// A byte-level tokenizer of its own, written as a model file: every byte
    /// a token, `he` and `llo` merged (so that a token can span a place the
    /// tests look for), and the two tags of a turn as control tokens.
    fn tiny_tokenizer() -> Tokenizer {
        // GPT-2's byte-level alphabet.
        let char_of = |byte: u8| {
            if matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF) {
                return char::from(byte);
            }
            let moved =
                (0..byte).filter(|b| !matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF)).count();
            char::from_u32(0x100 + moved as u32).expect("a character")
        };
        let text = |s: &str| s.bytes().map(char_of).collect::<String>();
        let mut tokens: Vec<String> = (0..=255u8).map(|b| char_of(b).to_string()).collect();
        tokens.extend(["he", "ll", "llo"].map(text));
        tokens.extend(["<|im_start|>".to_owned(), "<|im_end|>".to_owned()]);
        let mut types = vec![1i32; tokens.len() - 2];
        types.extend([3, 3]);
        let merges =
            [("h", "e"), ("l", "l"), ("ll", "o")].map(|(a, b)| format!("{} {}", text(a), text(b)));
        let file = ares_gguf::GgufBuilder::new()
            .str("general.architecture", "tiny")
            .str("tokenizer.ggml.model", "gpt2")
            .str("tokenizer.ggml.pre", "qwen35")
            .str_array("tokenizer.ggml.tokens", &tokens)
            .i32_array("tokenizer.ggml.token_type", &types)
            .str_array("tokenizer.ggml.merges", &merges)
            .u32("tokenizer.ggml.eos_token_id", 260)
            .build();
        let dir = std::env::temp_dir().join(format!("ares-model-chat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("tiny-{:?}.gguf", std::thread::current().id()));
        std::fs::write(&path, file).unwrap();
        let tokenizer = Tokenizer::from_gguf(&ares_gguf::Gguf::open(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        tokenizer
    }

    /// What Qwen3.5's and Qwen3.8's templates do with the text of a message,
    /// in short: the tools first in the system turn, then the system prompt;
    /// every text trimmed; an empty system prompt left out.
    const QWEN_LIKE: &str = concat!(
        r"{%- if tools %}{{ '<|im_start|>system\n# Tools' }}",
        r"{% for t in tools %}{{ '\n' + t.function.name }}{% endfor %}",
        r"{% if messages[0].role == 'system' %}{% set c = messages[0].content|trim %}",
        r"{% if c %}{{ '\n\n' + c }}{% endif %}{% endif %}{{ '<|im_end|>\n' }}",
        r"{%- elif messages[0].role == 'system' %}",
        r"{{ '<|im_start|>system\n' + messages[0].content|trim + '<|im_end|>\n' }}{%- endif %}",
        r"{% for m in messages %}{% if m.role != 'system' %}",
        r"{{ '<|im_start|>' + m.role + '\n' + m.content|trim + '<|im_end|>\n' }}",
        r"{% endif %}{% endfor %}",
        r"{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    );

    fn conversation(messages: &[(Role, &str)], tools: bool) -> Prompt {
        let tool = ares_api::Json::Object(vec![
            ("type".into(), ares_api::Json::String("function".into())),
            (
                "function".into(),
                ares_api::Json::Object(vec![(
                    "name".into(),
                    ares_api::Json::String("hello_tool".into()),
                )]),
            ),
        ]);
        Prompt::Chat {
            messages: messages.iter().map(|(r, t)| ares_api::Message::new(*r, *t)).collect(),
            tools: tools.then(|| vec![tool]),
            variables: Vec::new(),
            continue_final: false,
        }
    }

    /// The places the boundaries name, as the text before each of them.
    fn before(tokenizer: &Tokenizer, encoded: &EncodedPrompt) -> Vec<(String, bool)> {
        encoded
            .boundaries
            .iter()
            .map(|b| {
                let head = tokenizer.decode(&encoded.tokens[..b.at as usize], true).unwrap();
                let head = String::from_utf8(head).unwrap();
                let tail = head.rfind(['\n', '>']).map_or(head.as_str(), |at| &head[at + 1..]);
                (tail.to_owned(), b.root)
            })
            .collect()
    }

    #[test]
    fn the_parts_of_a_conversation_end_where_the_template_writes_them() {
        use Role::{Assistant, System, User};
        let tokenizer = tiny_tokenizer();
        let template = tokenizer.compile_template(QWEN_LIKE).unwrap();
        let chat =
            ChatOptions { template: Some(std::sync::Arc::new(template)), ..Default::default() };
        let messages = [
            (System, "  You help.\n"),
            (User, "hello"),
            (Assistant, "hello there"),
            (User, " shell\n"),
        ];
        let prompt = conversation(&messages, true);
        let encoded = encode_with_boundaries(&tokenizer, &chat, None, &prompt).unwrap();
        // The tokens are those of `encode`, and the text is the template's.
        assert_eq!(encoded.tokens, encode(&tokenizer, &chat, None, &prompt).unwrap());
        let text = String::from_utf8(tokenizer.decode(&encoded.tokens, true).unwrap()).unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\n# Tools\nhello_tool\n\nYou help.<|im_end|>\n<|im_start|>user\n\
             hello<|im_end|>\n<|im_start|>assistant\nhello there<|im_end|>\n<|im_start|>user\n\
             shell<|im_end|>\n<|im_start|>assistant\n"
        );
        // The start of the system prompt is the end of the tool list here;
        // it, the end of the system prompt and the start of the first message
        // are the roots.
        assert_eq!(
            before(&tokenizer, &encoded),
            [
                ("".to_owned(), true),             // after "\n\n", before "You help."
                ("You help.".to_owned(), true),    // the end of the system prompt
                ("".to_owned(), true),             // the start of the first message
                ("hello".to_owned(), false),       // its end
                ("".to_owned(), false),            // the answer's start
                ("hello there".to_owned(), false), // its end
                ("".to_owned(), false),
                ("shell".to_owned(), false),
            ]
        );
        let starts: Vec<u64> = encoded.boundaries.iter().map(|b| b.at).collect();
        assert!(starts.windows(2).all(|w| w[0] < w[1]), "{starts:?}");

        // Without tools and without a system prompt only the first message
        // is a root.
        let plain = conversation(&messages[1..], false);
        let encoded = encode_with_boundaries(&tokenizer, &chat, None, &plain).unwrap();
        let roots: Vec<bool> = encoded.boundaries.iter().map(|b| b.root).collect();
        assert_eq!(roots, [true, false, false, false, false, false]);
        // Plain text has none.
        let text = Prompt::Text("hello".into());
        assert!(
            encode_with_boundaries(&tokenizer, &chat, None, &text).unwrap().boundaries.is_empty()
        );

        // A place inside a token counts where the token starts: `hel` and
        // `lo` are the tokens `he` and `llo`.
        let glued =
            tokenizer.compile_template("{% for m in messages %}hel{{ m.content }}{% endfor %}");
        let chat = ChatOptions { template: Some(std::sync::Arc::new(glued.unwrap())), ..chat };
        let encoded =
            encode_with_boundaries(&tokenizer, &chat, None, &conversation(&[(User, "lo")], false));
        let encoded = encoded.unwrap();
        assert_eq!(encoded.tokens.len(), 2);
        assert_eq!(
            encoded.boundaries,
            [Boundary { at: 1, root: true }, Boundary { at: 2, root: false }]
        );
    }

    #[test]
    fn a_template_that_reads_a_text_otherwise_than_writing_it_gives_no_boundaries() {
        use Role::{Assistant, User};
        let tokenizer = tiny_tokenizer();
        let with = |source: &str| {
            let template = tokenizer.compile_template(source).unwrap();
            ChatOptions { template: Some(std::sync::Arc::new(template)), ..Default::default() }
        };
        let messages = [(User, "hello"), (Assistant, "hello")];
        // A template that tests the first characters of a text renders
        // something else with the marks: the prompt then has no boundaries,
        // and its tokens are the same.
        let peeks = with(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content.startswith('he') %}\
             !{% endif %}{{ m.content }}<|im_end|>\n{% endfor %}",
        );
        let prompt = conversation(&messages, false);
        let encoded = encode_with_boundaries(&tokenizer, &peeks, None, &prompt).unwrap();
        assert!(encoded.boundaries.is_empty());
        assert_eq!(encoded.tokens, encode(&tokenizer, &peeks, None, &prompt).unwrap());
        // One that writes a text twice: that message has no boundary, the
        // others do.
        let twice = with(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}{% if loop.last %}\
             {{ m.content }}{% endif %}<|im_end|>\n{% endfor %}",
        );
        let encoded = encode_with_boundaries(&tokenizer, &twice, None, &prompt).unwrap();
        assert_eq!(encoded.boundaries.len(), 2);
        // A message that holds the mark's character has none either.
        let prompt = conversation(&[(User, "he\u{f8fe}llo")], false);
        let encoded = encode_with_boundaries(&tokenizer, &twice, None, &prompt).unwrap();
        assert!(encoded.boundaries.is_empty());
    }

    /// The templates of the model files: every message of a conversation with
    /// tools, tool calls, results and reasoning has its boundaries, and the
    /// text before each is a prefix of the prompt that ends where a message's
    /// text starts or ends.
    #[test]
    fn the_templates_of_the_model_files_give_the_boundaries_of_every_message() {
        use Role::{Assistant, System, Tool, User};
        for (variable, tokenizer) in model_tokenizers() {
            let chat = ChatOptions::default();
            let mut call = ares_api::Message::new(Assistant, "Let me look.");
            call.tool_calls = Some(vec![ares_api::Json::Object(vec![
                ("id".into(), ares_api::Json::String("call_1".into())),
                ("type".into(), ares_api::Json::String("function".into())),
                (
                    "function".into(),
                    ares_api::Json::Object(vec![
                        ("name".into(), ares_api::Json::String("get_weather".into())),
                        ("arguments".into(), ares_api::Json::String("{\"city\": \"Bonn\"}".into())),
                    ]),
                ),
            ])]);
            call.reasoning_content = Some("The user wants the weather.".into());
            let mut result = ares_api::Message::new(Tool, "12 degrees");
            result.tool_call_id = Some("call_1".into());
            let mut answer = ares_api::Message::new(Assistant, "It is 12 degrees in Bonn.");
            answer.reasoning_content = Some("Easy.".into());
            let messages = vec![
                ares_api::Message::new(System, "You are a helpful assistant."),
                ares_api::Message::new(User, "Wie ist das Wetter in Bonn?"),
                call,
                result,
                answer,
                ares_api::Message::new(User, "Und morgen?"),
            ];
            let tool = ares_api::Json::Object(vec![
                ("type".into(), ares_api::Json::String("function".into())),
                (
                    "function".into(),
                    ares_api::Json::Object(vec![
                        ("name".into(), ares_api::Json::String("get_weather".into())),
                        ("description".into(), ares_api::Json::String("The weather.".into())),
                    ]),
                ),
            ]);
            let prompt = Prompt::Chat {
                messages: messages.clone(),
                tools: Some(vec![tool]),
                variables: Vec::new(),
                continue_final: false,
            };
            let encoded = encode_with_boundaries(&tokenizer, &chat, None, &prompt).unwrap();
            assert_eq!(
                encoded.tokens,
                encode(&tokenizer, &chat, None, &prompt).unwrap(),
                "{variable}"
            );
            let text = String::from_utf8(tokenizer.decode(&encoded.tokens, true).unwrap()).unwrap();
            // Two places per message, the roots first.
            assert_eq!(encoded.boundaries.len(), 2 * messages.len(), "{variable}: {encoded:?}");
            let roots = encoded.boundaries.iter().filter(|b| b.root).count();
            assert_eq!(roots, 3, "{variable}");
            for (boundary, message) in encoded.boundaries.chunks(2).zip(&messages) {
                let head = |b: &Boundary| {
                    String::from_utf8(
                        tokenizer.decode(&encoded.tokens[..b.at as usize], true).unwrap(),
                    )
                    .unwrap()
                };
                assert!(head(&boundary[1]).ends_with(&message.content), "{variable}: {message:?}");
                let after = &text[head(&boundary[0]).len()..];
                assert!(after.starts_with(&message.content), "{variable}: {message:?}");
            }
            // The system prompt's end is the first place after the tools.
            let system_end = encoded.boundaries[1].at as usize;
            let head = tokenizer.decode(&encoded.tokens[..system_end], true).unwrap();
            assert!(String::from_utf8(head).unwrap().contains("</tools>"), "{variable}");
        }
    }

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
            let tokens = encode(&tokenizer, &chat, None, &prompt).unwrap();
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
        let tokens = encode(tokenizer, chat, None, prompt).unwrap();
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
                encode(&tokenizer, &chat, None, &prompt),
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

    /// A conversation with an image, on the 27B's tokenizer and projector
    /// (`ARES_TEST_QWEN35_27B`, `ARES_TEST_QWEN3VL_27B`): the image is its start
    /// marker, a pad token for each of its rows and its end marker, where the
    /// template put the message's text; the text after it is tokenized on its
    /// own; the boundaries lie behind the rows; counting gives as many tokens.
    #[test]
    fn an_image_becomes_its_markers_and_a_pad_token_per_row() {
        let (Some(model), Some(mmproj)) = (
            std::env::var_os("ARES_TEST_QWEN35_27B"),
            std::env::var_os("ARES_TEST_QWEN3VL_27B"),
        ) else {
            eprintln!("skipped: set ARES_TEST_QWEN35_27B and ARES_TEST_QWEN3VL_27B");
            return;
        };
        let gguf = ares_gguf::Gguf::open(std::path::Path::new(&model)).unwrap();
        let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
        let limits = crate::qwen3vl::Limits::default();
        let projector = Arc::new(
            Projector::open(
                ares_gguf::Gguf::open(std::path::Path::new(&mmproj)).unwrap(),
                limits,
                crate::qwen3vl::Home::Ram,
            )
            .unwrap(),
        );
        let image = ares_api::Image {
            at: 0,
            width: 640,
            height: 480,
            pixels: Arc::new(vec![200; 640 * 480 * 3]),
        };
        let mut first = ares_api::Message::new(Role::User, "What is this?");
        first.images.push(image.clone());
        let mut second = ares_api::Message::new(Role::User, "And this?\nIt is here.");
        second.images.push(ares_api::Image { at: 10, ..image });
        let prompt = Prompt::Chat {
            messages: vec![
                ares_api::Message::new(Role::System, "Be brief."),
                first,
                ares_api::Message::new(Role::Assistant, "A gray square."),
                second,
            ],
            tools: None,
            variables: Vec::new(),
            continue_final: false,
        };
        let chat = ChatOptions::default();
        let encoded = encode_with_boundaries(&tokenizer, &chat, Some(&projector), &prompt).unwrap();
        let counted = encode(&tokenizer, &chat, Some(&projector), &prompt).unwrap();
        assert_eq!(counted, encoded.tokens);
        let id = |text: &str| tokenizer.special_token_id(text).unwrap();
        let (start, pad, end) = (id(IMAGE_START), id(IMAGE_PAD), id(IMAGE_END));
        assert_eq!(encoded.images.len(), 2);
        for span in &encoded.images {
            assert_eq!((span.image.nx, span.image.ny), (20, 15));
            let first = span.first as usize;
            assert_eq!(encoded.tokens[first - 1], start);
            assert!(encoded.tokens[first..first + 300].iter().all(|t| *t == pad));
            assert_eq!(encoded.tokens[first + 300], end);
        }
        assert_eq!(encoded.tokens.iter().filter(|t| **t == pad).count(), 600);
        // The first image opens its message, the second stands after
        // "And this?\n" and before "It is here."
        let text = |tokens: &[u32]| {
            let mut out = Vec::new();
            for t in tokens {
                out.extend_from_slice(tokenizer.piece(*t, true).unwrap());
            }
            String::from_utf8(out).unwrap()
        };
        let one = encoded.images[0].first as usize;
        assert!(text(&encoded.tokens[..one - 1]).ends_with("<|im_start|>user\n"));
        assert!(text(&encoded.tokens[one + 301..]).starts_with("What is this?<|im_end|>"));
        let two = encoded.images[1].first as usize;
        assert!(text(&encoded.tokens[..two - 1]).ends_with("And this?\n"));
        assert!(text(&encoded.tokens[two + 301..]).starts_with("It is here.<|im_end|>"));
        // The boundaries: as many as without the images, each one behind the
        // rows of the images before it.
        let plain = Prompt::Chat {
            messages: match &prompt {
                Prompt::Chat { messages, .. } => messages
                    .iter()
                    .map(|m| ares_api::Message { images: Vec::new(), ..m.clone() })
                    .collect(),
                Prompt::Text(_) => unreachable!(),
            },
            tools: None,
            variables: Vec::new(),
            continue_final: false,
        };
        let without = encode_with_boundaries(&tokenizer, &chat, None, &plain).unwrap();
        assert_eq!(encoded.boundaries.len(), without.boundaries.len());
        assert!(!encoded.boundaries.is_empty());
        for (with, without) in encoded.boundaries.iter().zip(&without.boundaries) {
            let images_before =
                encoded.images.iter().filter(|s| u64::from(s.first) < with.at).count() as u64;
            assert_eq!(with.at, without.at + images_before * 302, "{with:?} {without:?}");
        }
        // A model without a projector refuses the image by name.
        let refused = encode(&tokenizer, &chat, None, &prompt).unwrap_err();
        assert!(refused.to_string().contains("--mmproj"), "{refused}");
    }
}
