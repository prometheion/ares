//! The answer of the OpenAI chat and completion API: the response objects of
//! a whole reply and the chunks of a streamed one, the count of a prompt, and
//! the model list.

use std::time::{SystemTime, UNIX_EPOCH};

use ares_api::{RequestId, TokenProbs, Usage};
use serde::Serialize;

use super::request::{Endpoint, LogprobFormat};
use crate::bridge::Timings;
use crate::reasoning::Split;

/// The answer of a route that counts the tokens of a prompt, in the shape
/// OpenAI's `/v1/responses/input_tokens` has.
pub(crate) fn input_tokens(tokens: u32) -> String {
    format!(r#"{{"object":"response.input_tokens","input_tokens":{tokens}}}"#)
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
    /// How the probabilities are written, when the request asked for them.
    pub logprobs: Option<LogprobFormat>,
    /// The completion this head belongs to, of the `n` of the request.
    pub index: u32,
}

impl Head {
    pub fn new(endpoint: Endpoint, id: RequestId, model: &str) -> Self {
        Self {
            endpoint,
            id: format!("{}-{id}", endpoint.id_prefix()),
            created: unix_time(),
            model: model.to_owned(),
            logprobs: None,
            index: 0,
        }
    }

    /// The same head for the completion `index`.
    pub fn choice(&self, index: u32) -> Self {
        Self { index, ..self.clone() }
    }

    fn object(&self, streaming: bool) -> &'static str {
        match (self.endpoint, streaming) {
            (Endpoint::Chat, false) => "chat.completion",
            (Endpoint::Chat, true) => "chat.completion.chunk",
            (Endpoint::Completions, _) => "text_completion",
        }
    }

    /// One completion of a non-streaming request.
    pub fn complete_choice<'a>(
        &self,
        text: &'a Split,
        calls: &'a [Call],
        finish: &'static str,
        stop: Option<&'a str>,
        probs: &[TokenProbs],
    ) -> Choice<'a> {
        let logprobs = self.logprobs_value(probs, 0);
        let tool_calls: Vec<CallBody<'a>> = calls
            .iter()
            .map(|call| CallBody {
                index: None,
                id: Some(&call.id),
                kind: Some("function"),
                function: FunctionBody { name: Some(&call.name), arguments: &call.arguments },
            })
            .collect();
        match self.endpoint {
            Endpoint::Chat => Choice {
                index: self.index,
                message: Some(FullMessage {
                    role: "assistant",
                    content: &text.content,
                    reasoning_content: non_empty(&text.reasoning),
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                }),
                text: None,
                delta: None,
                logprobs,
                finish_reason: Some(finish),
                stop_reason: stop,
            },
            Endpoint::Completions => Choice {
                index: self.index,
                message: None,
                text: Some(&text.content),
                delta: None,
                logprobs,
                finish_reason: Some(finish),
                stop_reason: stop,
            },
        }
    }

    /// The whole answer of a non-streaming request.
    pub fn complete(&self, choices: Vec<Choice<'_>>, usage: Usage, timings: Timings) -> String {
        self.render(false, choices, Some(usage), Some(timings))
    }

    /// The probabilities of tokens, as a chunk of a stream. `offset` is how
    /// many characters of the text have gone out before them.
    pub fn probs_chunk(&self, probs: &[TokenProbs], offset: usize) -> String {
        let mut choice = match self.endpoint {
            Endpoint::Chat => self.chunk_choice(Some(Delta::default()), None, None),
            Endpoint::Completions => self.chunk_choice(None, Some(""), None),
        };
        choice.logprobs = self.logprobs_value(probs, offset);
        self.render(true, vec![choice], None, None)
    }

    fn logprobs_value(&self, probs: &[TokenProbs], offset: usize) -> Option<serde_json::Value> {
        self.logprobs.map(|format| render_logprobs(format, probs, offset))
    }

    fn chunk_choice<'a>(
        &self,
        delta: Option<Delta<'a>>,
        text: Option<&'a str>,
        finish_reason: Option<&'static str>,
    ) -> Choice<'a> {
        Choice {
            index: self.index,
            message: None,
            text,
            delta,
            logprobs: None,
            finish_reason,
            stop_reason: None,
        }
    }

    /// The first chunk of a chat stream: it announces the role.
    pub fn role_chunk(&self) -> String {
        let delta = Delta { role: Some("assistant"), content: Some(""), ..Delta::default() };
        self.render(true, vec![self.chunk_choice(Some(delta), None, None)], None, None)
    }

    /// A piece of output text.
    pub fn text_chunk(&self, text: &Split) -> String {
        let choice = match self.endpoint {
            Endpoint::Chat => self.chunk_choice(
                Some(Delta {
                    content: non_empty(&text.content),
                    reasoning_content: non_empty(&text.reasoning),
                    ..Delta::default()
                }),
                None,
                None,
            ),
            Endpoint::Completions => self.chunk_choice(None, Some(&text.content), None),
        };
        self.render(true, vec![choice], None, None)
    }

    /// The chunk that begins the call `index` of a chat stream: its id and
    /// the name of the tool, as OpenAI streams them.
    pub fn call_chunk(&self, index: usize, id: &str, name: &str) -> String {
        self.call_delta(CallBody {
            index: Some(index),
            id: Some(id),
            kind: Some("function"),
            function: FunctionBody { name: Some(name), arguments: "" },
        })
    }

    /// The next fragment of the arguments of the call `index`.
    pub fn arguments_chunk(&self, index: usize, json: &str) -> String {
        self.call_delta(CallBody {
            index: Some(index),
            id: None,
            kind: None,
            function: FunctionBody { name: None, arguments: json },
        })
    }

    fn call_delta(&self, call: CallBody<'_>) -> String {
        let delta = Delta { tool_calls: Some(vec![call]), ..Delta::default() };
        self.render(true, vec![self.chunk_choice(Some(delta), None, None)], None, None)
    }

    /// The last chunk of the text: carries `finish_reason` (and the stop
    /// string that ended the reply, if one did), the timings and no content.
    pub fn finish_chunk(
        &self,
        finish: &'static str,
        stop: Option<&str>,
        timings: Timings,
    ) -> String {
        let mut choice = match self.endpoint {
            Endpoint::Chat => self.chunk_choice(Some(Delta::default()), None, Some(finish)),
            Endpoint::Completions => self.chunk_choice(None, Some(""), Some(finish)),
        };
        choice.stop_reason = stop;
        self.render(true, vec![choice], None, Some(timings))
    }

    /// The chunk after the last one when the client asked for usage: no
    /// choices, only the totals.
    pub fn usage_chunk(&self, usage: Usage) -> String {
        self.render(true, Vec::new(), Some(usage), None)
    }

    fn render(
        &self,
        streaming: bool,
        choices: Vec<Choice<'_>>,
        usage: Option<Usage>,
        timings: Option<Timings>,
    ) -> String {
        let body = Completion {
            id: &self.id,
            object: self.object(streaming),
            created: self.created,
            model: &self.model,
            choices,
            usage: usage.map(UsageBody::from),
            timings: timings.map(TimingsBody::from),
        };
        serde_json::to_string(&body).expect("response objects serialize")
    }
}

/// The text of a token's bytes. A token that is part of a character prints as
/// replacement characters, and its `bytes` say what it is.
fn token_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The probabilities of `probs` as the format says. `offset` is the number of
/// characters of the completion before them (for the completions format).
fn render_logprobs(
    format: LogprobFormat,
    probs: &[TokenProbs],
    offset: usize,
) -> serde_json::Value {
    use serde_json::{Value, json};
    let post = format == LogprobFormat::ContentProbs;
    // llama.cpp writes a probability under `prob` after the chain and a
    // logarithm under `logprob` before it.
    let measure = |logprob: f32| -> (&'static str, f64) {
        if post { ("prob", f64::from(logprob).exp()) } else { ("logprob", f64::from(logprob)) }
    };
    match format {
        LogprobFormat::Content | LogprobFormat::ContentProbs => {
            let content: Vec<Value> = probs
                .iter()
                .map(|p| {
                    let alternatives: Vec<Value> = p
                        .alternatives
                        .iter()
                        .map(|a| {
                            let (key, value) = measure(a.logprob);
                            json!({ "id": a.token, "token": token_text(&a.bytes), "bytes": a.bytes, key: value })
                        })
                        .collect();
                    let (key, value) = measure(p.logprob);
                    let list = if post { "top_probs" } else { "top_logprobs" };
                    json!({
                        "id": p.token,
                        "token": token_text(&p.bytes),
                        "bytes": p.bytes,
                        key: value,
                        list: alternatives,
                    })
                })
                .collect();
            json!({ "content": content })
        }
        LogprobFormat::Legacy => {
            let mut at = offset;
            let mut offsets = Vec::with_capacity(probs.len());
            for p in probs {
                offsets.push(at);
                at += token_text(&p.bytes).chars().count();
            }
            let top: Vec<Value> = probs
                .iter()
                .map(|p| {
                    let mut map = serde_json::Map::new();
                    for a in &p.alternatives {
                        map.insert(token_text(&a.bytes), json!(a.logprob));
                    }
                    Value::Object(map)
                })
                .collect();
            json!({
                "tokens": probs.iter().map(|p| token_text(&p.bytes)).collect::<Vec<_>>(),
                "token_logprobs": probs.iter().map(|p| p.logprob).collect::<Vec<_>>(),
                "top_logprobs": top,
                "text_offset": offsets,
            })
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<TimingsBody>,
}

/// The timings of a request under the names llama.cpp's server gives them,
/// so that tools written for one read the other.
///
/// `prompt_ms` runs from the arrival of the request to the first token and
/// `predicted_ms` from the first token to the end. `predicted_per_second` is
/// therefore the tokens after the first over `predicted_ms`: the first token
/// comes out of the step that ends the prompt and is paid for there. The
/// draft counts are left out for a request that did not run alone, because
/// the engine counts drafts for all requests together.
#[derive(Serialize)]
pub(crate) struct TimingsBody {
    prompt_n: u32,
    prompt_ms: f64,
    prompt_per_second: Option<f64>,
    predicted_n: u32,
    predicted_ms: f64,
    predicted_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    draft_n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    draft_n_accepted: Option<u64>,
    /// A hash of the reply's token ids, 16 hex digits: two replies are the
    /// same tokens when it is the same.
    token_hash: String,
}

impl From<Timings> for TimingsBody {
    fn from(timings: Timings) -> Self {
        // Microseconds are as fine as the times are meaningful.
        let round = |value: f64| (value * 1e3).round() / 1e3;
        Self {
            prompt_n: timings.prompt_tokens,
            prompt_ms: round(timings.prompt_ms),
            prompt_per_second: timings.prompt_per_second().map(round),
            predicted_n: timings.generated_tokens,
            predicted_ms: round(timings.generated_ms),
            predicted_per_second: timings.generated_per_second().map(round),
            draft_n: timings.drafts.map(|drafts| drafts.0),
            draft_n_accepted: timings.drafts.map(|drafts| drafts.1),
            token_hash: format!("{:016x}", timings.output_hash),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct Choice<'a> {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<FullMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<Delta<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<serde_json::Value>,
    finish_reason: Option<&'static str>,
    /// The stop string that ended the reply, under vLLM's name; only when
    /// one did.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<&'a str>,
}

#[derive(Serialize)]
struct FullMessage<'a> {
    role: &'static str,
    /// The empty text for a reply that is only calls, as llama-server writes
    /// it (OpenAI writes `null`).
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CallBody<'a>>>,
}

/// A call of a tool that a reply made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Call {
    pub id: String,
    pub name: String,
    /// The arguments, a JSON object as text.
    pub arguments: String,
}

/// A call in a message, or a part of one in a delta of a stream: the first
/// delta of a call has its place, its id, its type and the name, the ones
/// after it the place and a fragment of the arguments.
#[derive(Serialize)]
struct CallBody<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a str>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    function: FunctionBody<'a>,
}

#[derive(Serialize)]
struct FunctionBody<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    arguments: &'a str,
}

#[derive(Serialize, Default)]
struct Delta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CallBody<'a>>>,
}

/// `text`, or nothing for the empty text: a field that has no text is left out.
fn non_empty(text: &str) -> Option<&str> {
    (!text.is_empty()).then_some(text)
}

#[derive(Serialize)]
struct UsageBody {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u64,
    prompt_tokens_details: PromptDetails,
    /// The tokens of the thinking, for a model that thinks.
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens_details: Option<CompletionDetails>,
}

#[derive(Serialize)]
struct PromptDetails {
    cached_tokens: u32,
}

#[derive(Serialize)]
struct CompletionDetails {
    reasoning_tokens: u32,
}

impl From<Usage> for UsageBody {
    fn from(usage: Usage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: u64::from(usage.prompt_tokens) + u64::from(usage.completion_tokens),
            prompt_tokens_details: PromptDetails { cached_tokens: usage.cached_prompt_tokens },
            completion_tokens_details: usage
                .reasoning_tokens
                .map(|reasoning_tokens| CompletionDetails { reasoning_tokens }),
        }
    }
}

/// The model list for `GET /v1/models`.
pub(crate) fn model_list(models: &[crate::models::Listed]) -> serde_json::Value {
    let data: Vec<_> = models
        .iter()
        .map(|m| {
            let mut model = serde_json::json!({
                "id": m.id,
                "object": "model",
                "owned_by": "ares",
            });
            // What only a loaded model knows; kept from its last load.
            if let Some(card) = &m.card {
                model["created"] = card.created.into();
                model["context_length"] = card.context_length.into();
            }
            // A server with presets says where each model stands, in the
            // shape llama-server's router has; a failure is a state here,
            // with its reason.
            if let Some(status) = &m.status {
                model["aliases"] = m.aliases.clone().into();
                model["status"] = serde_json::json!({ "value": status.status });
                if let Some(reason) = &status.reason {
                    model["status"]["reason"] = reason.as_str().into();
                }
            }
            model
        })
        .collect();
    serde_json::json!({ "object": "list", "data": data })
}
