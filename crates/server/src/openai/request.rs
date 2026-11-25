//! The request of the OpenAI chat and completion API, read and checked: the
//! request every route comes to, since a route of another API writes its
//! body over into a chat body first (`chat_body`).
//!
//! Requests are read from a generic [`Json`] value instead of being derived
//! into a struct. That costs some code, and buys what a client needs most
//! when it sends something wrong: an error that names the field
//! (`temperature must be a number, got a string`) instead of serde's
//! `invalid type: string "hot", expected f64 at line 1 column 29`, which
//! does not say which field it is about. It also gives one place to be
//! lenient where real clients are sloppy (`null` for "not set", `-1` for
//! "no limit", whole numbers written as `100.0`).

use ares_api::{
    BiasKey, GenerateRequest, Image, Json, LogitBias, LogprobRequest, Message, Prompt, RequestId,
    Role, SamplerKind, Sampling, TOP_LOGPROBS_MAX,
};

use crate::error::ApiError;
use crate::reasoning::{self, Asked};
use crate::structured;
use crate::tools;

/// Which of the two endpoints a request came in on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Endpoint {
    /// `/v1/chat/completions`
    Chat,
    /// `/v1/completions`
    Completions,
}

impl Endpoint {
    pub(crate) fn id_prefix(self) -> &'static str {
        match self {
            Endpoint::Chat => "chatcmpl",
            Endpoint::Completions => "cmpl",
        }
    }

    /// The API, as the log names it.
    pub(crate) fn api(self) -> &'static str {
        match self {
            Endpoint::Chat => "chat",
            Endpoint::Completions => "completions",
        }
    }
}

/// How the probabilities a request asked for are written in the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogprobFormat {
    /// `choices[].logprobs.content[]` with `logprob` and `top_logprobs`
    /// (OpenAI's chat format, which llama.cpp's server writes with a token
    /// `id` added).
    Content,
    /// The same with `prob` and `top_probs`: llama.cpp's
    /// `post_sampling_probs`.
    ContentProbs,
    /// OpenAI's completions format: `tokens`, `token_logprobs`,
    /// `top_logprobs` and `text_offset`.
    Legacy,
}

/// A request body, understood.
#[derive(Debug)]
pub(crate) struct Parsed {
    pub request: GenerateRequest,
    pub stream: bool,
    pub include_usage: bool,
    /// What a chat request says about thinking: the variables of the chat
    /// template are not in `request` yet.
    pub reasoning: Asked,
    /// What a chat request says about tools: the rule for the reply is not
    /// in `request` yet.
    pub tools: tools::Asked,
    /// How many completions to make of the prompt.
    pub n: u32,
    pub logprobs: Option<LogprobFormat>,
    /// The request asks for a seed of its own draw (llama.cpp's `seed: -1`),
    /// which is not the same as saying nothing where the model has a seed.
    pub fresh_seed: bool,
    /// The limit of tokens the reply got in place of the larger one the
    /// request asked for (the model's `--max-tokens-limit`); set where the
    /// model's defaults are applied.
    pub lowered: Option<u32>,
    /// The priority the body names (`priority`), which the request has
    /// unless a header names another ([`Parsed::take_priority`]).
    pub priority: Option<i32>,
}

/// The header that gives a request its priority, for a client that cannot
/// add a member to a body (Claude Code's `ANTHROPIC_CUSTOM_HEADERS`, the
/// `http_headers` of a provider of Codex).
pub(crate) const PRIORITY_HEADER: &str = "x-ares-priority";

impl Parsed {
    /// Gives the request its priority: the body's, or the header's
    /// ([`PRIORITY_HEADER`]) when the body names none. A request that names
    /// two different ones is refused.
    pub(crate) fn take_priority(
        &mut self,
        header: Option<&axum::http::HeaderValue>,
    ) -> Result<(), ApiError> {
        let header = match header {
            None => None,
            Some(value) => {
                Some(value.to_str().ok().and_then(|v| v.trim().parse::<i32>().ok()).ok_or_else(
                    || {
                        ApiError::invalid_param(
                            PRIORITY_HEADER,
                            format!(
                                "the header {PRIORITY_HEADER} is a whole number from {} to {}",
                                i32::MIN,
                                i32::MAX
                            ),
                        )
                    },
                )?)
            }
        };
        self.request.priority = match (self.priority, header) {
            (Some(body), Some(header)) if body != header => {
                return Err(ApiError::invalid_param(
                    "priority",
                    format!(
                        "the body asks for priority {body} and the header {PRIORITY_HEADER} for \
                         {header}: give one"
                    ),
                ));
            }
            (Some(priority), _) | (None, Some(priority)) => priority,
            (None, None) => 0,
        };
        Ok(())
    }
}

/// The most completions one request may ask for.
pub(crate) const N_MAX: u32 = 16;

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

/// The fields of a request that are read and handed to the engine.
const SERVED: &[&str] = &[
    "model",
    "stream",
    "stream_options",
    "max_tokens",
    "n_predict",
    "stop",
    "n",
    "n_cmpl",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "seed",
    "draft",
    "presence_penalty",
    "frequency_penalty",
    "repeat_penalty",
    "repetition_penalty",
    "repeat_last_n",
    "typical_p",
    "typical",
    "top_n_sigma",
    "xtc_probability",
    "xtc_threshold",
    "dynatemp_range",
    "dynatemp_exponent",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
    "dry_sequence_breakers",
    "logit_bias",
    "ignore_eos",
    "samplers",
    "logprobs",
    "top_logprobs",
    "n_probs",
    "post_sampling_probs",
    "min_keep",
    "priority",
];

/// Fields that carry no instruction for the generation: bookkeeping of the
/// client or of the service it was written for. They are accepted and have no
/// effect by their own definition.
const BOOKKEEPING: &[&str] =
    &["user", "metadata", "store", "service_tier", "prompt_cache_key", "safety_identifier"];

/// Refuses every field that asks for something the server does not do.
///
/// A request must not get an answer that silently ignores part of it, so a
/// field is served, or says nothing by its value (`null`, or the value that
/// switches the feature off), or the request is refused with the field's
/// name. That holds for fields nobody has heard of as well: an option of
/// another server would otherwise be dropped without a word.
fn refuse_unserved(endpoint: Endpoint, fields: &Fields<'_>) -> Result<(), ApiError> {
    for (name, value) in fields.0 {
        let name = name.as_str();
        let own = match endpoint {
            Endpoint::Chat => {
                [
                    "messages",
                    "max_completion_tokens",
                    "continue_final_message",
                    "add_generation_prompt",
                ]
                .contains(&name)
                    || reasoning::FIELDS.contains(&name)
                    || tools::FIELDS.contains(&name)
            }
            Endpoint::Completions => name == "prompt",
        } || structured::FIELDS.contains(&name);
        if value.is_null() || own || SERVED.contains(&name) || BOOKKEEPING.contains(&name) {
            continue;
        }
        if let Some(why) = unserved(name, value) {
            return Err(ApiError::invalid_param(name, format!("{name} is not supported: {why}")));
        }
    }
    Ok(())
}

/// Why the field `name` with `value` cannot be served, or `None` when the
/// value asks for nothing.
fn unserved(name: &str, value: &Json) -> Option<&'static str> {
    let number = |off: f64| match value {
        Json::Int(i) => *i as f64 == off,
        Json::Uint(u) => *u as f64 == off,
        Json::Float(f) => *f == off,
        _ => false,
    };
    let off = match name {
        "best_of" => number(1.0),
        "echo" => matches!(value, Json::Bool(false)),
        _ => false,
    };
    if off {
        return None;
    }
    Some(match name {
        "best_of" => {
            "this server returns the completions of a prompt as n, not the best of several"
        }
        "adaptive_target" | "adaptive_decay" => {
            "adaptive-p is not served: like mirostat it follows the probability of each token it \
             picks, and unlike mirostat it has no draw that the sampler can do without drafts"
        }
        "echo" | "suffix" => "the completion is returned on its own",
        "functions" | "function_call" => {
            "the functions of the first chat API are not served: send tools and tool_choice"
        }
        name if tools::FIELDS.contains(&name) => {
            "tools belong to chat requests: /v1/completions has no chat template"
        }
        name if reasoning::FIELDS.contains(&name) => {
            "thinking belongs to chat requests: /v1/completions has no chat template"
        }
        _ => "it is not a parameter of this server",
    })
}

/// [`parse`] after the JSON syntax is dealt with.
pub(crate) fn from_json(
    endpoint: Endpoint,
    json: &Json,
    id: RequestId,
) -> Result<Parsed, ApiError> {
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

    let n = completions(&fields)?;
    let (logprobs, format) = logprobs(endpoint, &fields)?;
    let sampling = Sampling { logprobs, ..sampling(&fields)? };
    sampling.validate().map_err(|e| ApiError::invalid_param(e.field, e.to_string()))?;
    refuse_unserved(endpoint, &fields)?;
    // A seed that is there and is none is llama.cpp's -1.
    let fresh_seed = sampling.seed.is_none() && fields.get("seed").is_some();

    let (reasoning, tools) = match endpoint {
        Endpoint::Chat => (reasoning::asked(&fields)?, tools::asked(&fields)?),
        Endpoint::Completions => (Asked::default(), tools::Asked::default()),
    };
    let sampling = Sampling {
        reasoning_budget_tokens: reasoning.budget,
        reasoning_budget_message: reasoning.budget_message.clone(),
        ..sampling
    };
    let request = GenerateRequest {
        model,
        prompt,
        sampling,
        max_tokens: max_tokens(&fields, endpoint)?,
        stop: stop(&fields)?,
        draft: fields.bool("draft")?.unwrap_or(true),
        structure: structured::asked(&fields)?,
        priority: 0,
        id,
    };
    let priority = match fields.int("priority")? {
        None => None,
        Some(priority) => Some(i32::try_from(priority).map_err(|_| {
            ApiError::invalid_param(
                "priority",
                format!("priority must be from {} to {}, got {priority}", i32::MIN, i32::MAX),
            )
        })?),
    };
    let include_usage = match fields.get("stream_options") {
        None => false,
        Some(Json::Object(members)) => Fields(members).bool("include_usage")?.unwrap_or(false),
        Some(other) => return Err(wrong_type("stream_options", "an object", other)),
    };
    Ok(Parsed {
        request,
        stream: fields.bool("stream")?.unwrap_or(false),
        include_usage,
        reasoning,
        tools,
        n,
        logprobs: format,
        fresh_seed,
        lowered: None,
        priority,
    })
}

/// How many completions the request asks for: `n` of OpenAI or llama.cpp's
/// `n_cmpl`.
fn completions(fields: &Fields<'_>) -> Result<u32, ApiError> {
    let Some((name, n)) = fields.alias(&["n", "n_cmpl"], Fields::int)? else {
        return Ok(1);
    };
    u32::try_from(n).ok().filter(|n| (1..=N_MAX).contains(n)).ok_or_else(|| {
        ApiError::invalid_param(name, format!("{name} must be between 1 and {N_MAX}, got {n}"))
    })
}

/// The probabilities a request asks for, and the format of the answer.
///
/// OpenAI's chat `logprobs: true` with `top_logprobs`, its completions
/// `logprobs: n`, and llama.cpp's `n_probs` with `post_sampling_probs`. A
/// request that asks in both forms must ask for the same (`n_probs: 0` is
/// nothing, as `logprobs: false` is), and then gets llama.cpp's format, as
/// llama-server answers it. The probabilities are the model's own unless
/// `post_sampling_probs` says they belong to the distribution the token is
/// drawn from.
fn logprobs(
    endpoint: Endpoint,
    fields: &Fields<'_>,
) -> Result<(Option<LogprobRequest>, Option<LogprobFormat>), ApiError> {
    let post = fields.bool("post_sampling_probs")?.unwrap_or(false);
    let top = |name: &str, value: i64| {
        u32::try_from(value).ok().filter(|top| *top <= TOP_LOGPROBS_MAX).ok_or_else(|| {
            ApiError::invalid_param(
                name,
                format!("{name} must be between 0 and {TOP_LOGPROBS_MAX}, got {value}"),
            )
        })
    };
    // llama.cpp's n_probs: 0 asks for nothing.
    let llama = match fields.int("n_probs")? {
        None => None,
        Some(0) => Some(None),
        Some(n) => Some(Some(top("n_probs", n)?)),
    };
    if post && !matches!(llama, Some(Some(_))) {
        return Err(ApiError::invalid_param(
            "post_sampling_probs",
            "post_sampling_probs needs n_probs to be greater than 0",
        ));
    }
    let openai = openai_logprobs(endpoint, fields, top)?;
    let nothing = |asked: Option<u32>| asked.map_or("nothing".to_owned(), |n| n.to_string());
    match (llama, openai) {
        (Some(asked), Some(other)) if asked != other => Err(ApiError::invalid_param(
            "n_probs",
            format!(
                "n_probs asks for {} alternatives of each token and logprobs for {}: they are \
                 the same parameter and differ, give one",
                nothing(asked),
                nothing(other)
            ),
        )),
        (Some(Some(top)), _) => {
            let format = if post { LogprobFormat::ContentProbs } else { LogprobFormat::Content };
            Ok((Some(LogprobRequest { top, after_chain: post }), Some(format)))
        }
        (None, Some(Some(top))) => {
            let format = match endpoint {
                Endpoint::Chat => LogprobFormat::Content,
                Endpoint::Completions => LogprobFormat::Legacy,
            };
            Ok((Some(LogprobRequest { top, after_chain: false }), Some(format)))
        }
        _ => Ok((None, None)),
    }
}

/// The alternatives OpenAI's fields ask for: `None` when the request names
/// none of them, `Some(None)` for `logprobs: false`.
fn openai_logprobs(
    endpoint: Endpoint,
    fields: &Fields<'_>,
    top: impl Fn(&str, i64) -> Result<u32, ApiError>,
) -> Result<Option<Option<u32>>, ApiError> {
    let without = || {
        if fields.get("top_logprobs").is_some() {
            return Err(ApiError::invalid_param(
                "top_logprobs",
                "top_logprobs requires logprobs to be set to true",
            ));
        }
        Ok(())
    };
    match (endpoint, fields.get("logprobs")) {
        (_, None) => without().map(|()| None),
        (Endpoint::Chat, Some(Json::Bool(false))) => without().map(|()| Some(None)),
        (Endpoint::Chat, Some(Json::Bool(true))) => {
            let n = fields.int("top_logprobs")?.unwrap_or(0);
            Ok(Some(Some(top("top_logprobs", n)?)))
        }
        (Endpoint::Chat, Some(other)) => Err(wrong_type("logprobs", "a boolean", other)),
        (Endpoint::Completions, Some(_)) => {
            if fields.get("top_logprobs").is_some() {
                return Err(ApiError::invalid_param(
                    "top_logprobs",
                    "top_logprobs belongs to the chat endpoint; /v1/completions takes the \
                     number in logprobs",
                ));
            }
            let n = fields.int("logprobs")?.expect("present");
            Ok(Some(Some(top("logprobs", n)?)))
        }
    }
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

    pub(crate) fn bool(&self, name: &str) -> Result<Option<bool>, ApiError> {
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
    pub(crate) fn int(&self, name: &str) -> Result<Option<i64>, ApiError> {
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

    /// A parameter that goes by several names (OpenAI's, llama.cpp's and
    /// another engine's), each read with `read`: the value and the first
    /// name it is given under. No name wins over another: a request that
    /// gives two of them different values is refused.
    fn alias<T: PartialEq>(
        &self,
        names: &[&'static str],
        read: impl Fn(&Self, &str) -> Result<Option<T>, ApiError>,
    ) -> Result<Option<(&'static str, T)>, ApiError> {
        let mut found: Option<(&'static str, T)> = None;
        for &name in names {
            let Some(value) = read(self, name)? else { continue };
            match &found {
                Some((first, other)) if *other != value => {
                    return Err(ApiError::invalid_param(
                        name,
                        format!("{first} and {name} are the same parameter and differ: give one"),
                    ));
                }
                Some(_) => {}
                None => found = Some((name, value)),
            }
        }
        Ok(found)
    }

    /// An `f32` parameter that goes by several names ([`Fields::alias`]).
    fn f32_alias(&self, names: &[&'static str]) -> Result<Option<f32>, ApiError> {
        Ok(self.alias(names, Self::f32)?.map(|(_, value)| value))
    }

    /// A whole number of at least 0.
    fn uint(&self, name: &str) -> Result<Option<u32>, ApiError> {
        match self.int(name)? {
            None => Ok(None),
            Some(n) => u32::try_from(n).map(Some).map_err(|_| {
                ApiError::invalid_param(name, format!("{name} must be at least 0, got {n}"))
            }),
        }
    }

    /// The number of tokens of history a penalty looks at. llama.cpp writes
    /// "the whole context" as -1, which this server does not serve.
    fn window(&self, name: &str) -> Result<Option<u32>, ApiError> {
        match self.int(name)? {
            None => Ok(None),
            Some(n) if n < 0 => Err(ApiError::invalid_param(
                name,
                format!(
                    "{name} must be between 0 and {}, got {n}: the history the penalties look \
                     at is bounded, and the whole context is not served",
                    ares_api::PENALTY_WINDOW_MAX
                ),
            )),
            Some(n) => Ok(Some(u32::try_from(n).unwrap_or(u32::MAX))),
        }
    }
}

pub(crate) fn wrong_type(name: &str, expected: &str, got: &Json) -> ApiError {
    ApiError::invalid_param(name, format!("{name} must be {expected}, got {}", got.kind()))
}

pub(crate) fn expect_str<'a>(name: &str, value: &'a Json) -> Result<&'a str, ApiError> {
    value.as_str().ok_or_else(|| wrong_type(name, "a string", value))
}

fn sampling(fields: &Fields<'_>) -> Result<Sampling, ApiError> {
    let min_keep = fields.int("min_keep")?;
    if let Some(n) = min_keep
        && !(0..=1).contains(&n)
    {
        return Err(ApiError::invalid_param(
            "min_keep",
            format!(
                "min_keep must be 0 or 1, got {n}: every cut keeps the most probable token, and \
                 keeping more would take a count of the tokens a cut left, which the cuts do not \
                 make"
            ),
        ));
    }
    Ok(Sampling {
        temperature: fields.f32("temperature")?,
        // llama.cpp and vLLM write "no top-k" as 0 or -1.
        top_k: fields.int("top_k")?.map(|k| u32::try_from(k.max(0)).unwrap_or(u32::MAX)),
        top_p: fields.f32("top_p")?,
        min_p: fields.f32("min_p")?,
        presence_penalty: fields.f32("presence_penalty")?,
        frequency_penalty: fields.f32("frequency_penalty")?,
        repeat_penalty: fields.f32_alias(&["repeat_penalty", "repetition_penalty"])?,
        repeat_last_n: fields.window("repeat_last_n")?,
        typical_p: fields.f32_alias(&["typical_p", "typical"])?,
        top_n_sigma: fields.f32("top_n_sigma")?,
        xtc_probability: fields.f32("xtc_probability")?,
        xtc_threshold: fields.f32("xtc_threshold")?,
        dynatemp_range: fields.f32("dynatemp_range")?,
        dynatemp_exponent: fields.f32("dynatemp_exponent")?,
        mirostat: fields
            .int("mirostat")?
            .map(|m| {
                u8::try_from(m).ok().filter(|m| *m <= 2).ok_or_else(|| {
                    ApiError::invalid_param(
                        "mirostat",
                        format!("mirostat must be 0, 1 or 2, got {m}"),
                    )
                })
            })
            .transpose()?,
        mirostat_tau: fields.f32("mirostat_tau")?,
        mirostat_eta: fields.f32("mirostat_eta")?,
        dry_multiplier: fields.f32("dry_multiplier")?,
        dry_base: fields.f32("dry_base")?,
        dry_allowed_length: fields.uint("dry_allowed_length")?,
        dry_penalty_last_n: fields.window("dry_penalty_last_n")?,
        dry_sequence_breakers: breakers(fields)?,
        logit_bias: logit_bias(fields)?,
        ignore_eos: fields.bool("ignore_eos")?.unwrap_or(false),
        // Read with the rest of thinking (`reasoning::asked`).
        reasoning_budget_tokens: None,
        reasoning_budget_message: None,
        samplers: samplers(fields)?,
        logprobs: None,
        seed: seed(fields)?,
    })
}

/// DRY's sequence breakers: an array of strings.
fn breakers(fields: &Fields<'_>) -> Result<Option<Vec<String>>, ApiError> {
    match fields.get("dry_sequence_breakers") {
        None => Ok(None),
        Some(Json::Array(items)) => items
            .iter()
            .map(|item| match item {
                Json::String(s) => Ok(s.clone()),
                other => Err(ApiError::invalid_param(
                    "dry_sequence_breakers",
                    format!("dry_sequence_breakers must hold strings, found {}", other.kind()),
                )),
            })
            .collect::<Result<_, _>>()
            .map(Some),
        Some(other) => Err(wrong_type("dry_sequence_breakers", "an array of strings", other)),
    }
}

/// The logit bias, in OpenAI's form (an object from a token to a bias) or in
/// llama.cpp's (a list of `[token, bias]`), where a token is a number, or a
/// string of which every token gets the bias, and a bias is a number, or
/// `false` for a token that must never be picked.
fn logit_bias(fields: &Fields<'_>) -> Result<Vec<LogitBias>, ApiError> {
    let bias = |at: &str, value: &Json| -> Result<Option<f32>, ApiError> {
        match value {
            Json::Bool(false) => Ok(None),
            Json::Int(i) => Ok(Some(*i as f32)),
            Json::Uint(u) => Ok(Some(*u as f32)),
            Json::Float(f) => Ok(Some(*f as f32)),
            other => Err(ApiError::invalid_param(
                "logit_bias",
                format!("{at}: a bias is a number or false, got {}", other.kind()),
            )),
        }
    };
    match fields.get("logit_bias") {
        None => Ok(Vec::new()),
        Some(Json::Array(items)) => items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let at = format!("logit_bias[{i}]");
                let Json::Array(pair) = item else {
                    return Err(ApiError::invalid_param(
                        "logit_bias",
                        format!("{at} must be [token, bias], got {}", item.kind()),
                    ));
                };
                let [key, value] = pair.as_slice() else {
                    return Err(ApiError::invalid_param(
                        "logit_bias",
                        format!("{at} must be [token, bias]"),
                    ));
                };
                let key = match key {
                    Json::Int(t) if *t >= 0 => {
                        BiasKey::Token(u32::try_from(*t).unwrap_or(u32::MAX))
                    }
                    Json::Uint(t) => BiasKey::Token(u32::try_from(*t).unwrap_or(u32::MAX)),
                    Json::String(text) => BiasKey::Text(text.clone()),
                    other => {
                        return Err(ApiError::invalid_param(
                            "logit_bias",
                            format!(
                                "{at}: the token is a non-negative number or a string, got {}",
                                other.kind()
                            ),
                        ));
                    }
                };
                Ok(LogitBias { key, bias: bias(&at, value)? })
            })
            .collect(),
        Some(Json::Object(members)) => members
            .iter()
            .map(|(name, value)| {
                // A key that is a number is a token; any other is a text.
                let key = match name.parse::<u32>() {
                    Ok(token) => BiasKey::Token(token),
                    Err(_) => BiasKey::Text(name.clone()),
                };
                Ok(LogitBias { key, bias: bias(&format!("logit_bias[\"{name}\"]"), value)? })
            })
            .collect(),
        Some(other) => Err(wrong_type("logit_bias", "an object or a list of pairs", other)),
    }
}

/// llama.cpp's `samplers`: the names in the order they apply in, or a string
/// of their letters.
fn samplers(fields: &Fields<'_>) -> Result<Option<Vec<SamplerKind>>, ApiError> {
    let refuse = |what: &str| {
        ApiError::invalid_param(
            "samplers",
            format!(
                "samplers: {what} is not served (infill completes code around a gap; \
                 adaptive_p follows the probability of each token it picks)"
            ),
        )
    };
    match fields.get("samplers") {
        None => Ok(None),
        Some(Json::Array(items)) => items
            .iter()
            .map(|item| {
                let Json::String(name) = item else {
                    return Err(ApiError::invalid_param(
                        "samplers",
                        format!("samplers must hold strings, found {}", item.kind()),
                    ));
                };
                SamplerKind::from_name(name).ok_or_else(|| match name.as_str() {
                    "infill" | "adaptive_p" => refuse(name),
                    _ => ApiError::invalid_param(
                        "samplers",
                        format!("samplers: '{name}' is not the name of a sampler"),
                    ),
                })
            })
            .collect::<Result<_, _>>()
            .map(Some),
        Some(Json::String(letters)) => letters
            .chars()
            .map(|letter| {
                SamplerKind::from_letter(letter).ok_or_else(|| match letter {
                    'i' | 'a' => refuse(&letter.to_string()),
                    _ => ApiError::invalid_param(
                        "samplers",
                        format!("samplers: '{letter}' is not the letter of a sampler"),
                    ),
                })
            })
            .collect::<Result<_, _>>()
            .map(Some),
        Some(other) => {
            Err(wrong_type("samplers", "an array of names or a string of letters", other))
        }
    }
}

fn seed(fields: &Fields<'_>) -> Result<Option<u64>, ApiError> {
    match fields.get("seed") {
        Some(Json::Uint(seed)) => Ok(Some(*seed)),
        // llama.cpp writes "pick a random seed" as -1.
        _ => Ok(fields.int("seed")?.and_then(|seed| u64::try_from(seed).ok())),
    }
}

fn max_tokens(fields: &Fields<'_>, endpoint: Endpoint) -> Result<Option<u32>, ApiError> {
    // llama.cpp's `n_predict` is the same limit; `max_completion_tokens`
    // replaces `max_tokens` on the chat endpoint.
    let names: &[&'static str] = match endpoint {
        Endpoint::Chat => &["max_tokens", "max_completion_tokens", "n_predict"],
        Endpoint::Completions => &["max_tokens", "n_predict"],
    };
    match fields.alias(names, Fields::int)? {
        // llama.cpp clients write "no limit" as -1.
        Some((_, n)) if n < 0 => Ok(None),
        Some((name, 0)) => Err(ApiError::invalid_param(name, format!("{name} must be at least 1"))),
        Some((_, n)) => Ok(Some(u32::try_from(n).unwrap_or(u32::MAX))),
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
    let continue_final = continues(fields, &messages)?;
    Ok(Prompt::Chat { messages, tools, variables: Vec::new(), continue_final })
}

/// Whether the reply continues the last message of the conversation: when it
/// is the assistant's, as llama-server prefills it, and as a request says
/// with vLLM's `continue_final_message` (or `add_generation_prompt: false`,
/// which vLLM asks to go with it). A last message that calls tools begins a
/// turn of its own unless the request asks to continue it, which is refused:
/// what follows calls are their results.
fn continues(fields: &Fields<'_>, messages: &[Message]) -> Result<bool, ApiError> {
    let last = messages.last().filter(|message| message.role == Role::Assistant);
    let calls = last.is_some_and(|message| message.tool_calls.is_some());
    let continues = match (
        fields.bool("continue_final_message")?,
        fields.bool("add_generation_prompt")?,
    ) {
        (Some(true), Some(true)) => {
            return Err(ApiError::invalid_param(
                "continue_final_message",
                "continue_final_message and add_generation_prompt cannot both be true: a reply \
                 continues the last message or begins a turn of its own",
            ));
        }
        (Some(false), Some(false)) => {
            return Err(ApiError::invalid_param(
                "add_generation_prompt",
                "add_generation_prompt false asks for a prompt that ends with a closed turn and \
                 no start of the reply, which is not served: a reply continues the last message \
                 (continue_final_message) or begins a turn of its own",
            ));
        }
        (Some(asked), _) => asked,
        (None, Some(generation)) => !generation,
        (None, None) => last.is_some() && !calls,
    };
    let name = if fields.get("continue_final_message").is_some() {
        "continue_final_message"
    } else {
        "add_generation_prompt"
    };
    if continues && last.is_none() {
        return Err(ApiError::invalid_param(
            name,
            "the last message is not the assistant's: there is no message to continue",
        ));
    }
    if continues && calls {
        return Err(ApiError::invalid_param(
            name,
            "the last message calls tools, which a reply cannot continue: send the results of \
             the calls as messages of the role tool",
        ));
    }
    Ok(continues)
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
        Some(Json::Array(calls)) => {
            for (j, call) in calls.iter().enumerate() {
                tool_call(&format!("{}[{j}]", at("tool_calls")), call)?;
            }
            Some(calls.clone())
        }
        Some(other) => {
            return Err(ApiError::invalid_param(
                at("tool_calls"),
                format!("{} must be an array, got {}", at("tool_calls"), other.kind()),
            ));
        }
    };

    let (content, images) = content(index, fields.0)?;
    Ok(Message {
        role,
        content,
        images,
        name: string_field("name")?,
        tool_call_id: string_field("tool_call_id")?,
        tool_calls,
        reasoning_content: string_field("reasoning_content")?,
    })
}

/// Checks a call of an assistant message: `{"id", "type": "function",
/// "function": {"name", "arguments"}}`, the arguments an object as JSON text
/// (OpenAI's form) or as it is.
fn tool_call(at: &str, call: &Json) -> Result<(), ApiError> {
    let refuse = |message: String| Err(ApiError::invalid_param(at, message));
    let Some(function @ Json::Object(_)) = call.get("function") else {
        return refuse(format!("{at}.function must be an object"));
    };
    if !matches!(function.get("name"), Some(Json::String(name)) if !name.is_empty()) {
        return refuse(format!("{at}.function.name must be a string that is not empty"));
    }
    let object = match function.get("arguments") {
        None | Some(Json::Null | Json::Object(_)) => true,
        Some(Json::String(text)) if text.trim().is_empty() => true,
        Some(Json::String(text)) => {
            matches!(serde_json::from_str::<Json>(text), Ok(Json::Object(_)))
        }
        Some(_) => false,
    };
    if !object {
        return refuse(format!(
            "{at}.function.arguments must be a JSON object, as text or as it is"
        ));
    }
    Ok(())
}

/// Flattens `content` to text and its images. It is a string, a list of
/// typed parts (`text`, `image_url`), or `null` (an assistant message that
/// only carries tool calls).
///
/// The text parts of a message without images are joined as they are. A
/// message with images is joined as llama-server joins it: a line break between
/// two text parts, none before or after an image, which stands at its place
/// in the text.
fn content(index: usize, members: &[(String, Json)]) -> Result<(String, Vec<Image>), ApiError> {
    let at = format!("messages[{index}].content");
    match Fields(members).get("content") {
        None => Ok((String::new(), Vec::new())),
        Some(Json::String(text)) => Ok((text.clone(), Vec::new())),
        Some(Json::Array(parts)) => {
            let has_images = parts
                .iter()
                .any(|part| part.get("type").and_then(Json::as_str) == Some("image_url"));
            let mut text = String::new();
            let mut images: Vec<Image> = Vec::new();
            let mut after_image = false;
            for (j, part) in parts.iter().enumerate() {
                let kind = part.get("type").and_then(Json::as_str);
                match (kind, part.get("text").and_then(Json::as_str)) {
                    (Some("text"), Some(piece)) => {
                        if has_images && !after_image && (!text.is_empty() || !images.is_empty()) {
                            text.push('\n');
                        }
                        text.push_str(piece);
                        after_image = false;
                    }
                    (Some("text"), None) => {
                        return Err(ApiError::invalid_param(
                            format!("{at}[{j}]"),
                            format!("{at}[{j}] is a text part without a 'text' string"),
                        ));
                    }
                    (Some("image_url"), _) => {
                        let place = format!("{at}[{j}].image_url");
                        let url = match part.get("image_url") {
                            Some(Json::String(url)) => url.as_str(),
                            Some(object @ Json::Object(_)) => {
                                object.get("url").and_then(Json::as_str).ok_or_else(|| {
                                    ApiError::invalid_param(
                                        place.clone(),
                                        format!("{place}.url must be a string"),
                                    )
                                })?
                            }
                            _ => {
                                return Err(ApiError::invalid_param(
                                    place.clone(),
                                    format!("{place} must be an object with a 'url'"),
                                ));
                            }
                        };
                        images.push(crate::images::from_url(&place, url)?.at(text.len()));
                        after_image = true;
                    }
                    (other, _) => {
                        return Err(ApiError::invalid_param(
                            format!("{at}[{j}]"),
                            format!(
                                "{at}[{j}]: content parts of type '{}' are not supported, \
                                 only 'text' and 'image_url' parts are",
                                other.unwrap_or("(none)")
                            ),
                        ));
                    }
                }
            }
            Ok((text, images))
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

    /// A PNG of one pixel, as a data URL.
    const PIXEL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

    fn parts(parts: Vec<Json>) -> Json {
        let kind = |kind: &str| ("type", Json::String(kind.into()));
        let parts = parts
            .into_iter()
            .map(|p| match p {
                Json::String(t) if t == "IMAGE" => object(&[
                    kind("image_url"),
                    ("image_url", object(&[("url", Json::String(PIXEL.into()))])),
                ]),
                Json::String(t) => object(&[kind("text"), ("text", Json::String(t))]),
                other => other,
            })
            .collect();
        let message =
            object(&[("role", Json::String("user".into())), ("content", Json::Array(parts))]);
        object(&[("model", Json::String("m".into())), ("messages", Json::Array(vec![message]))])
    }

    fn first_message(json: &Json) -> Message {
        let parsed = from_json(Endpoint::Chat, json, RequestId(1)).ok().unwrap();
        let Prompt::Chat { messages, .. } = parsed.request.prompt else { panic!("a chat") };
        messages.into_iter().next().unwrap()
    }

    #[test]
    fn a_message_with_images_is_joined_as_llama_server_joins_it() {
        let s = |t: &str| Json::String(t.into());
        // Text parts with a line break between them, none next to an image.
        let m = first_message(&parts(vec![s("A"), s("IMAGE"), s("B"), s("C"), s("IMAGE")]));
        assert_eq!(m.content, "AB\nC");
        assert_eq!(m.images.iter().map(|i| i.at).collect::<Vec<_>>(), [1, 4]);
        assert_eq!((m.images[0].width, m.images[0].height, m.images[0].pixels.len()), (1, 1, 3));
        let m = first_message(&parts(vec![s("IMAGE"), s("A"), s("B")]));
        assert_eq!((m.content.as_str(), m.images[0].at), ("A\nB", 0));
        // Without images the parts are joined as they are.
        let m = first_message(&parts(vec![s("A"), s("B")]));
        assert_eq!((m.content.as_str(), m.images.len()), ("AB", 0));
        // A part that names an image another way is refused by name.
        let remote = object(&[
            ("type", s("image_url")),
            ("image_url", object(&[("url", s("https://example.com/a.png"))])),
        ]);
        let error = from_json(Endpoint::Chat, &parts(vec![remote]), RequestId(1)).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("messages[0].content[0].image_url"));
        assert!(error.message.contains("data URL"), "{}", error.message);
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
