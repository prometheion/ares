//! The OpenAI wire format: request bodies in, response bodies out.
//!
//! Requests are read from a generic [`Json`] value instead of being derived
//! into a struct. That costs some code, and buys what a client needs most
//! when it sends something wrong: an error that names the field
//! (`temperature must be a number, got a string`) instead of serde's
//! `invalid type: string "hot", expected f64 at line 1 column 29`, which
//! does not say which field it is about. It also gives one place to be
//! lenient where real clients are sloppy (`null` for "not set", `-1` for
//! "no limit", whole numbers written as `100.0`).

use std::time::{SystemTime, UNIX_EPOCH};

use ares_api::{GenerateRequest, Json, Message, Prompt, RequestId, Role, Sampling, Usage};
use serde::Serialize;

use crate::error::ApiError;

/// Which of the two endpoints a request came in on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Endpoint {
    /// `/v1/chat/completions`
    Chat,
    /// `/v1/completions`
    Completions,
}

impl Endpoint {
    fn id_prefix(self) -> &'static str {
        match self {
            Endpoint::Chat => "chatcmpl",
            Endpoint::Completions => "cmpl",
        }
    }
}

/// A request body, understood.
#[derive(Debug)]
pub(crate) struct Parsed {
    pub request: GenerateRequest,
    pub stream: bool,
    pub include_usage: bool,
}

/// Parses and checks the body of a completion request.
pub(crate) fn parse(endpoint: Endpoint, body: &[u8], id: RequestId) -> Result<Parsed, ApiError> {
    from_json(endpoint, &read_json(body)?, id)
}

/// The JSON syntax of a request body.
pub(crate) fn read_json(body: &[u8]) -> Result<Json, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid(format!("the request body is not valid JSON: {e}")))
}

/// The members of a request body, which must be an object.
pub(crate) fn members(json: &Json) -> Result<&[(String, Json)], ApiError> {
    match json {
        Json::Object(members) => Ok(members),
        _ => Err(ApiError::invalid(format!(
            "the request body must be a JSON object, got {}",
            json.kind()
        ))),
    }
}

/// [`parse`] after the JSON syntax is dealt with.
fn from_json(endpoint: Endpoint, json: &Json, id: RequestId) -> Result<Parsed, ApiError> {
    let fields = Fields(members(json)?);

    let model = fields
        .string("model")?
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::invalid_param("model", "model is required"))?
        .to_owned();

    let prompt = match endpoint {
        Endpoint::Chat => chat_prompt(&fields)?,
        Endpoint::Completions => text_prompt(&fields)?,
    };

    if let Some(n) = fields.int("n")?
        && n != 1
    {
        return Err(ApiError::invalid_param(
            "n",
            format!("n must be 1, got {n}: this server returns one completion per request"),
        ));
    }

    let sampling = sampling(&fields)?;
    sampling.validate().map_err(|e| ApiError::invalid_param(e.field, e.to_string()))?;

    let request = GenerateRequest {
        model,
        prompt,
        sampling,
        max_tokens: max_tokens(&fields, endpoint)?,
        stop: stop(&fields)?,
        id,
    };
    let include_usage = match fields.get("stream_options") {
        None => false,
        Some(Json::Object(members)) => Fields(members).bool("include_usage")?.unwrap_or(false),
        Some(other) => return Err(wrong_type("stream_options", "an object", other)),
    };
    Ok(Parsed { request, stream: fields.bool("stream")?.unwrap_or(false), include_usage })
}

/// The members of a JSON object, with typed access that names the field in
/// every error. `null` counts as "not set" everywhere, as it does for OpenAI.
pub(crate) struct Fields<'a>(pub(crate) &'a [(String, Json)]);

impl<'a> Fields<'a> {
    pub(crate) fn get(&self, name: &str) -> Option<&'a Json> {
        self.0.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v).filter(|v| !v.is_null())
    }

    pub(crate) fn string(&self, name: &str) -> Result<Option<&'a str>, ApiError> {
        self.get(name).map(|v| expect_str(name, v)).transpose()
    }

    fn bool(&self, name: &str) -> Result<Option<bool>, ApiError> {
        match self.get(name) {
            None => Ok(None),
            Some(Json::Bool(b)) => Ok(Some(*b)),
            Some(other) => Err(wrong_type(name, "a boolean", other)),
        }
    }

    fn float(&self, name: &str) -> Result<Option<f64>, ApiError> {
        match self.get(name) {
            None => Ok(None),
            Some(Json::Int(i)) => Ok(Some(*i as f64)),
            Some(Json::Uint(u)) => Ok(Some(*u as f64)),
            Some(Json::Float(f)) => Ok(Some(*f)),
            Some(other) => Err(wrong_type(name, "a number", other)),
        }
    }

    /// A whole number. `100.0` is accepted, `100.5` is not.
    fn int(&self, name: &str) -> Result<Option<i64>, ApiError> {
        match self.get(name) {
            None => Ok(None),
            Some(Json::Int(i)) => Ok(Some(*i)),
            Some(Json::Uint(_)) => {
                Err(ApiError::invalid_param(name, format!("{name} is too large")))
            }
            Some(Json::Float(f)) if f.fract() == 0.0 && f.abs() < 9.0e18 => Ok(Some(*f as i64)),
            Some(Json::Float(f)) => Err(ApiError::invalid_param(
                name,
                format!("{name} must be a whole number, got {f}"),
            )),
            Some(other) => Err(wrong_type(name, "a whole number", other)),
        }
    }

    /// An `f32` parameter. Values that do not fit become infinite and are then
    /// refused by `Sampling::validate`, which names the field.
    fn f32(&self, name: &str) -> Result<Option<f32>, ApiError> {
        Ok(self.float(name)?.map(|v| v as f32))
    }
}

pub(crate) fn wrong_type(name: &str, expected: &str, got: &Json) -> ApiError {
    ApiError::invalid_param(name, format!("{name} must be {expected}, got {}", got.kind()))
}

pub(crate) fn expect_str<'a>(name: &str, value: &'a Json) -> Result<&'a str, ApiError> {
    value.as_str().ok_or_else(|| wrong_type(name, "a string", value))
}

fn sampling(fields: &Fields<'_>) -> Result<Sampling, ApiError> {
    Ok(Sampling {
        temperature: fields.f32("temperature")?,
        // llama.cpp and vLLM write "no top-k" as 0 or -1.
        top_k: fields.int("top_k")?.map(|k| u32::try_from(k.max(0)).unwrap_or(u32::MAX)),
        top_p: fields.f32("top_p")?,
        min_p: fields.f32("min_p")?,
        presence_penalty: fields.f32("presence_penalty")?,
        frequency_penalty: fields.f32("frequency_penalty")?,
        repeat_penalty: fields.f32("repeat_penalty")?,
        seed: seed(fields)?,
    })
}

fn seed(fields: &Fields<'_>) -> Result<Option<u64>, ApiError> {
    match fields.get("seed") {
        Some(Json::Uint(seed)) => Ok(Some(*seed)),
        // llama.cpp writes "pick a random seed" as -1.
        _ => Ok(fields.int("seed")?.and_then(|seed| u64::try_from(seed).ok())),
    }
}

fn max_tokens(fields: &Fields<'_>, endpoint: Endpoint) -> Result<Option<u32>, ApiError> {
    // `max_completion_tokens` replaces `max_tokens` on the chat endpoint.
    let (name, value) = match endpoint {
        Endpoint::Chat if fields.get("max_completion_tokens").is_some() => {
            ("max_completion_tokens", fields.int("max_completion_tokens")?)
        }
        _ => ("max_tokens", fields.int("max_tokens")?),
    };
    match value {
        // llama.cpp clients write "no limit" as -1.
        Some(n) if n < 0 => Ok(None),
        Some(0) => Err(ApiError::invalid_param(name, format!("{name} must be at least 1"))),
        Some(n) => Ok(Some(u32::try_from(n).unwrap_or(u32::MAX))),
        None => Ok(None),
    }
}

fn stop(fields: &Fields<'_>) -> Result<Vec<String>, ApiError> {
    let sequences = match fields.get("stop") {
        None => return Ok(Vec::new()),
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(items)) => items
            .iter()
            .map(|item| match item {
                Json::String(s) => Ok(s.clone()),
                other => Err(ApiError::invalid_param(
                    "stop",
                    format!("stop must hold strings, found {}", other.kind()),
                )),
            })
            .collect::<Result<_, _>>()?,
        Some(other) => {
            return Err(ApiError::invalid_param(
                "stop",
                format!("stop must be a string or an array of strings, got {}", other.kind()),
            ));
        }
    };
    // An empty stop string matches before the first token and would end every
    // generation at once.
    if sequences.iter().any(String::is_empty) {
        return Err(ApiError::invalid_param("stop", "stop strings must not be empty"));
    }
    Ok(sequences)
}

fn chat_prompt(fields: &Fields<'_>) -> Result<Prompt, ApiError> {
    let Some(Json::Array(raw)) = fields.get("messages") else {
        return Err(match fields.get("messages") {
            None => ApiError::invalid_param("messages", "messages is required"),
            Some(other) => wrong_type("messages", "an array", other),
        });
    };
    if raw.is_empty() {
        return Err(ApiError::invalid_param("messages", "messages must not be empty"));
    }
    let messages =
        raw.iter().enumerate().map(|(i, m)| message(i, m)).collect::<Result<Vec<_>, _>>()?;

    let tools = match fields.get("tools") {
        None => None,
        Some(Json::Array(tools)) => Some(tools.clone()),
        Some(other) => return Err(wrong_type("tools", "an array", other)),
    };
    Ok(Prompt::Chat { messages, tools })
}

fn message(index: usize, value: &Json) -> Result<Message, ApiError> {
    let at = |field: &str| format!("messages[{index}].{field}");
    let Json::Object(members) = value else {
        return Err(ApiError::invalid_param(
            format!("messages[{index}]"),
            format!("messages[{index}] must be an object, got {}", value.kind()),
        ));
    };
    let fields = Fields(members);

    let role = match fields.string("role")? {
        Some("system" | "developer") => Role::System,
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("tool") => Role::Tool,
        Some(other) => {
            return Err(ApiError::invalid_param(
                at("role"),
                format!(
                    "{}: unknown role '{other}', expected system, user, assistant or tool",
                    at("role")
                ),
            ));
        }
        None => {
            return Err(ApiError::invalid_param(at("role"), format!("{} is required", at("role"))));
        }
    };

    let string_field = |name: &str| -> Result<Option<String>, ApiError> {
        match fields.get(name) {
            None => Ok(None),
            Some(Json::String(s)) => Ok(Some(s.clone())),
            Some(other) => Err(ApiError::invalid_param(
                at(name),
                format!("{} must be a string, got {}", at(name), other.kind()),
            )),
        }
    };
    let tool_calls = match fields.get("tool_calls") {
        None => None,
        Some(Json::Array(calls)) => Some(calls.clone()),
        Some(other) => {
            return Err(ApiError::invalid_param(
                at("tool_calls"),
                format!("{} must be an array, got {}", at("tool_calls"), other.kind()),
            ));
        }
    };

    Ok(Message {
        role,
        content: content(index, fields.0)?,
        name: string_field("name")?,
        tool_call_id: string_field("tool_call_id")?,
        tool_calls,
    })
}

/// Flattens `content` to text. It is a string, a list of typed parts, or
/// `null` (an assistant message that only carries tool calls).
fn content(index: usize, members: &[(String, Json)]) -> Result<String, ApiError> {
    let at = format!("messages[{index}].content");
    match Fields(members).get("content") {
        None => Ok(String::new()),
        Some(Json::String(text)) => Ok(text.clone()),
        Some(Json::Array(parts)) => {
            let mut text = String::new();
            for (j, part) in parts.iter().enumerate() {
                let kind = part.get("type").and_then(Json::as_str);
                match (kind, part.get("text").and_then(Json::as_str)) {
                    (Some("text"), Some(piece)) => text.push_str(piece),
                    (Some("text"), None) => {
                        return Err(ApiError::invalid_param(
                            format!("{at}[{j}]"),
                            format!("{at}[{j}] is a text part without a 'text' string"),
                        ));
                    }
                    (other, _) => {
                        return Err(ApiError::invalid_param(
                            format!("{at}[{j}]"),
                            format!(
                                "{at}[{j}]: content parts of type '{}' are not supported, \
                                 only 'text' parts are",
                                other.unwrap_or("(none)")
                            ),
                        ));
                    }
                }
            }
            Ok(text)
        }
        Some(other) => Err(ApiError::invalid_param(
            at.clone(),
            format!("{at} must be a string or an array of parts, got {}", other.kind()),
        )),
    }
}

fn text_prompt(fields: &Fields<'_>) -> Result<Prompt, ApiError> {
    match fields.get("prompt") {
        None => Err(ApiError::invalid_param("prompt", "prompt is required")),
        Some(Json::String(text)) => Ok(Prompt::Text(text.clone())),
        Some(Json::Array(items)) => match items.as_slice() {
            [Json::String(text)] => Ok(Prompt::Text(text.clone())),
            _ => Err(ApiError::invalid_param(
                "prompt",
                "prompt must be one string: arrays of several prompts and token ids are not \
                 supported, send one request per prompt",
            )),
        },
        Some(other) => Err(wrong_type("prompt", "a string", other)),
    }
}

/// Seconds since the Unix epoch, for `created`.
pub(crate) fn unix_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The fields every response object of one request repeats.
#[derive(Debug, Clone)]
pub(crate) struct Head {
    pub endpoint: Endpoint,
    pub id: String,
    pub created: u64,
    pub model: String,
}

impl Head {
    pub fn new(endpoint: Endpoint, id: RequestId, model: &str) -> Self {
        Self {
            endpoint,
            id: format!("{}-{id}", endpoint.id_prefix()),
            created: unix_time(),
            model: model.to_owned(),
        }
    }

    fn object(&self, streaming: bool) -> &'static str {
        match (self.endpoint, streaming) {
            (Endpoint::Chat, false) => "chat.completion",
            (Endpoint::Chat, true) => "chat.completion.chunk",
            (Endpoint::Completions, _) => "text_completion",
        }
    }

    /// The whole answer of a non-streaming request.
    pub fn complete(&self, text: &str, finish: &'static str, usage: Usage) -> String {
        let choice = match self.endpoint {
            Endpoint::Chat => Choice {
                index: 0,
                message: Some(FullMessage { role: "assistant", content: text }),
                text: None,
                delta: None,
                finish_reason: Some(finish),
            },
            Endpoint::Completions => Choice {
                index: 0,
                message: None,
                text: Some(text),
                delta: None,
                finish_reason: Some(finish),
            },
        };
        self.render(false, vec![choice], Some(usage))
    }

    /// The first chunk of a chat stream: it announces the role.
    pub fn role_chunk(&self) -> String {
        let delta = Delta { role: Some("assistant"), content: Some("") };
        self.render(true, vec![chunk_choice(Some(delta), None, None)], None)
    }

    /// A piece of output text.
    pub fn text_chunk(&self, text: &str) -> String {
        let choice = match self.endpoint {
            Endpoint::Chat => {
                chunk_choice(Some(Delta { role: None, content: Some(text) }), None, None)
            }
            Endpoint::Completions => chunk_choice(None, Some(text), None),
        };
        self.render(true, vec![choice], None)
    }

    /// The last chunk of the text: carries `finish_reason` and no content.
    pub fn finish_chunk(&self, finish: &'static str) -> String {
        let choice = match self.endpoint {
            Endpoint::Chat => chunk_choice(Some(Delta::default()), None, Some(finish)),
            Endpoint::Completions => chunk_choice(None, Some(""), Some(finish)),
        };
        self.render(true, vec![choice], None)
    }

    /// The chunk after the last one when the client asked for usage: no
    /// choices, only the totals.
    pub fn usage_chunk(&self, usage: Usage) -> String {
        self.render(true, Vec::new(), Some(usage))
    }

    fn render(&self, streaming: bool, choices: Vec<Choice<'_>>, usage: Option<Usage>) -> String {
        let body = Completion {
            id: &self.id,
            object: self.object(streaming),
            created: self.created,
            model: &self.model,
            choices,
            usage: usage.map(UsageBody::from),
        };
        serde_json::to_string(&body).expect("response objects serialize")
    }
}

fn chunk_choice<'a>(
    delta: Option<Delta<'a>>,
    text: Option<&'a str>,
    finish_reason: Option<&'static str>,
) -> Choice<'a> {
    Choice { index: 0, message: None, text, delta, finish_reason }
}

#[derive(Serialize)]
struct Completion<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<Choice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageBody>,
}

#[derive(Serialize)]
struct Choice<'a> {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<FullMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<Delta<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct FullMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize, Default)]
struct Delta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

#[derive(Serialize)]
struct UsageBody {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u64,
    prompt_tokens_details: PromptDetails,
}

#[derive(Serialize)]
struct PromptDetails {
    cached_tokens: u32,
}

impl From<Usage> for UsageBody {
    fn from(usage: Usage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: u64::from(usage.prompt_tokens) + u64::from(usage.completion_tokens),
            prompt_tokens_details: PromptDetails { cached_tokens: usage.cached_prompt_tokens },
        }
    }
}

/// The model list for `GET /v1/models`.
pub(crate) fn model_list(models: &[ares_api::ModelCard]) -> serde_json::Value {
    let data: Vec<_> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": m.created,
                "owned_by": "ares",
                "context_length": m.context_length,
            })
        })
        .collect();
    serde_json::json!({ "object": "list", "data": data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(members: &[(&str, Json)]) -> Json {
        Json::Object(members.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect())
    }

    fn chat_with(extra: (&str, Json)) -> Json {
        let message = object(&[
            ("role", Json::String("user".into())),
            ("content", Json::String("hi".into())),
        ]);
        object(&[
            ("model", Json::String("m".into())),
            ("messages", Json::Array(vec![message])),
            extra,
        ])
    }

    #[test]
    fn values_a_json_parser_cannot_produce_are_still_refused() {
        // JSON text cannot spell NaN or an infinity, but the request is
        // checked on the value, so no way of building one gets through.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300] {
            let json = chat_with(("temperature", Json::Float(bad)));
            let error = from_json(Endpoint::Chat, &json, RequestId(1)).unwrap_err();
            assert_eq!(error.param.as_deref(), Some("temperature"), "{bad}: {error:?}");
            assert!(error.message.starts_with("temperature must be"), "{}", error.message);
        }
        let json = chat_with(("top_p", Json::Float(f64::NAN)));
        let error = from_json(Endpoint::Chat, &json, RequestId(1)).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("top_p"));
    }

    #[test]
    fn whole_numbers_may_be_written_as_floats() {
        let json = chat_with(("max_tokens", Json::Float(100.0)));
        let parsed = from_json(Endpoint::Chat, &json, RequestId(1)).unwrap();
        assert_eq!(parsed.request.max_tokens, Some(100));
        let json = chat_with(("max_tokens", Json::Float(100.5)));
        assert_eq!(
            from_json(Endpoint::Chat, &json, RequestId(1)).unwrap_err().param.as_deref(),
            Some("max_tokens")
        );
    }
}
