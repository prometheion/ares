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

use ares_api::{
    BiasKey, GenerateRequest, Image, Json, LogitBias, LogprobRequest, Message, Prompt, RequestId,
    Role, SamplerKind, Sampling, TOP_LOGPROBS_MAX, TokenProbs, Usage,
};
use serde::Serialize;

use crate::bridge::Timings;
use crate::error::ApiError;
use crate::reasoning::{self, Asked, Split};
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
    fn id_prefix(self) -> &'static str {
        match self {
            Endpoint::Chat => "chatcmpl",
            Endpoint::Completions => "cmpl",
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
    let request = GenerateRequest {
        model,
        prompt,
        sampling,
        max_tokens: max_tokens(&fields, endpoint)?,
        stop: stop(&fields)?,
        draft: fields.bool("draft")?.unwrap_or(true),
        structure: structured::asked(&fields)?,
        id,
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
    })
}

/// How many completions the request asks for: `n` of OpenAI or llama.cpp's
/// `n_cmpl`.
fn completions(fields: &Fields<'_>) -> Result<u32, ApiError> {
    let (name, n) = match (fields.int("n_cmpl")?, fields.int("n")?) {
        (Some(n), _) => ("n_cmpl", n),
        (None, Some(n)) => ("n", n),
        (None, None) => return Ok(1),
    };
    u32::try_from(n).ok().filter(|n| (1..=N_MAX).contains(n)).ok_or_else(|| {
        ApiError::invalid_param(name, format!("{name} must be between 1 and {N_MAX}, got {n}"))
    })
}

/// The probabilities a request asks for, and the format of the answer.
///
/// OpenAI's chat `logprobs: true` with `top_logprobs`, its completions
/// `logprobs: n`, and llama.cpp's `n_probs` (which wins when both are there,
/// as it does in llama-server) with `post_sampling_probs`. The probabilities
/// are the model's own unless `post_sampling_probs` says they belong to the
/// distribution the token is drawn from.
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
    if let Some(n) = fields.int("n_probs")? {
        if n == 0 {
            if post {
                return Err(ApiError::invalid_param(
                    "post_sampling_probs",
                    "post_sampling_probs needs n_probs to be greater than 0",
                ));
            }
            return Ok((None, None));
        }
        let format = if post { LogprobFormat::ContentProbs } else { LogprobFormat::Content };
        return Ok((
            Some(LogprobRequest { top: top("n_probs", n)?, after_chain: post }),
            Some(format),
        ));
    }
    if post {
        return Err(ApiError::invalid_param(
            "post_sampling_probs",
            "post_sampling_probs needs n_probs to be greater than 0",
        ));
    }
    match (endpoint, fields.get("logprobs")) {
        (_, None) => {
            if fields.get("top_logprobs").is_some() {
                return Err(ApiError::invalid_param(
                    "top_logprobs",
                    "top_logprobs requires logprobs to be set to true",
                ));
            }
            Ok((None, None))
        }
        (Endpoint::Chat, Some(Json::Bool(false))) => {
            if fields.get("top_logprobs").is_some() {
                return Err(ApiError::invalid_param(
                    "top_logprobs",
                    "top_logprobs requires logprobs to be set to true",
                ));
            }
            Ok((None, None))
        }
        (Endpoint::Chat, Some(Json::Bool(true))) => {
            let n = fields.int("top_logprobs")?.unwrap_or(0);
            let request = LogprobRequest { top: top("top_logprobs", n)?, after_chain: false };
            Ok((Some(request), Some(LogprobFormat::Content)))
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
            let request = LogprobRequest { top: top("logprobs", n)?, after_chain: false };
            Ok((Some(request), Some(LogprobFormat::Legacy)))
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

    /// An `f32` parameter that goes by several names (llama.cpp's and
    /// another engine's): the first one given, and a request that gives two
    /// with different values is refused.
    fn f32_alias(&self, names: &[&str]) -> Result<Option<f32>, ApiError> {
        let mut found: Option<(&str, f32)> = None;
        for name in names {
            if let Some(value) = self.f32(name)? {
                match found {
                    Some((first, other)) if other != value => {
                        return Err(ApiError::invalid_param(
                            *name,
                            format!("{first} and {name} are the same parameter and differ"),
                        ));
                    }
                    Some(_) => {}
                    None => found = Some((name, value)),
                }
            }
        }
        Ok(found.map(|(_, value)| value))
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
        && n > 1
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
    let (name, value) = match endpoint {
        _ if fields.get("n_predict").is_some() => ("n_predict", fields.int("n_predict")?),
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

/// The body of `GET /metrics`.
pub(crate) fn metrics(stats: &ares_api::EngineStats) -> String {
    let (ram, ssd) = (&stats.ram, &stats.ssd);
    let rows: [(&str, &str, &str, u64); 56] = [
        (
            "cache_ram_pages",
            "gauge",
            "Pages of sequence state the prompt cache holds in RAM",
            ram.pages,
        ),
        ("cache_ram_bytes", "gauge", "Page-locked RAM the prompt cache's pages hold", ram.bytes),
        (
            "cache_ram_pages_in_total",
            "counter",
            "Pages that left the devices and were copied into the prompt cache in RAM",
            ram.pages_in,
        ),
        (
            "cache_ram_pages_read_total",
            "counter",
            "Pages a prompt took back to the devices from the prompt cache in RAM",
            ram.pages_read,
        ),
        (
            "cache_ram_pages_dropped_total",
            "counter",
            "Pages worth keeping that the prompt cache in RAM gave up unwritten",
            ram.pages_dropped,
        ),
        (
            "cache_ram_transfer_nanoseconds_total",
            "counter",
            "Time the step loop waited while pages moved between the devices and RAM",
            ram.transfer_ns,
        ),
        (
            "cache_ssd_pages_written_total",
            "counter",
            "Pages of sequence state the prompt cache wrote to SSD",
            ssd.pages_written,
        ),
        (
            "cache_ssd_checkpoints_written_total",
            "counter",
            "Checkpoints of recurrent state the prompt cache wrote to SSD",
            ssd.checkpoints_written,
        ),
        (
            "cache_ssd_written_bytes_total",
            "counter",
            "Bytes the prompt cache wrote to SSD",
            ssd.bytes_written,
        ),
        (
            "cache_ssd_write_nanoseconds_total",
            "counter",
            "Time the prompt cache's writer spent writing and syncing",
            ssd.write_ns,
        ),
        (
            "cache_ssd_pages_read_total",
            "counter",
            "Pages of sequence state the prompt cache read back from SSD",
            ssd.pages_read,
        ),
        (
            "cache_ssd_checkpoints_read_total",
            "counter",
            "Checkpoints of recurrent state the prompt cache read back from SSD",
            ssd.checkpoints_read,
        ),
        (
            "cache_ssd_read_bytes_total",
            "counter",
            "Bytes the prompt cache read back from SSD",
            ssd.bytes_read,
        ),
        (
            "cache_ssd_read_nanoseconds_total",
            "counter",
            "Time the prompt cache's readers spent reading and checking, summed",
            ssd.read_ns,
        ),
        (
            "cache_ssd_transfer_nanoseconds_total",
            "counter",
            "Time the step loop waited while cached state moved between devices and host",
            ssd.transfer_ns,
        ),
        ("cache_ssd_bytes", "gauge", "Bytes the model's store takes on SSD", ssd.store_bytes),
        (
            "cache_ssd_pending",
            "gauge",
            "Pages and checkpoints that wait to be written to SSD",
            ssd.pending,
        ),
        (
            "cache_ssd_checkpoints_skipped_total",
            "counter",
            "Checkpoints of recurrent state that were due on SSD and were given up unwritten",
            ssd.checkpoints_skipped,
        ),
        (
            "requests_finished_total",
            "counter",
            "Requests that ran to their end",
            stats.requests_finished,
        ),
        (
            "requests_cancelled_total",
            "counter",
            "Requests that were cancelled",
            stats.requests_cancelled,
        ),
        (
            "requests_refused_total",
            "counter",
            "Requests refused for lack of room",
            stats.requests_refused,
        ),
        ("steps_total", "counter", "Steps of the engine", stats.steps),
        ("prompt_tokens_total", "counter", "Prompt tokens computed", stats.prompt_tokens),
        ("generated_tokens_total", "counter", "Tokens generated", stats.generated_tokens),
        ("drafts_proposed_total", "counter", "Draft tokens put to the test", stats.drafts_proposed),
        ("drafts_accepted_total", "counter", "Draft tokens accepted", stats.drafts_accepted),
        (
            "prefix_hits_total",
            "counter",
            "Starts of a request from cells the prefix cache held",
            stats.prefix_hits,
        ),
        (
            "cached_cells_total",
            "counter",
            "Cells taken from the prefix cache instead of computed",
            stats.cached_cells,
        ),
        (
            "replayed_rows_total",
            "counter",
            "Rows run again to rebuild recurrent state from cached cells",
            stats.replayed_rows,
        ),
        (
            "checkpoints_saved_total",
            "counter",
            "Checkpoints of recurrent state taken",
            stats.checkpoints_saved,
        ),
        (
            "checkpoints_restored_total",
            "counter",
            "Checkpoints restored by the start of a request",
            stats.checkpoints_restored,
        ),
        (
            "checkpoints_dropped_total",
            "counter",
            "Checkpoints given up for the budget or with their page",
            stats.checkpoints_dropped,
        ),
        (
            "page_copies_total",
            "counter",
            "Pages copied before a sequence wrote into a page that others read",
            stats.page_copies,
        ),
        ("sequences", "gauge", "Sequences running", u64::from(stats.sequences)),
        ("pages_free", "gauge", "Free pages of sequence state", u64::from(stats.pages_free)),
        ("pages", "gauge", "Pages of sequence state", u64::from(stats.pages)),
        (
            "cached_pages",
            "gauge",
            "Pages and tails the prefix cache holds",
            u64::from(stats.cached_pages),
        ),
        (
            "checkpoint_ram_bytes",
            "gauge",
            "Page-locked RAM held for checkpoints of recurrent state",
            stats.checkpoint_ram,
        ),
        ("expert_picks_total", "counter", "Experts the routers picked", stats.experts.picks),
        (
            "expert_hits_total",
            "counter",
            "Picked experts that were in a VRAM slot",
            stats.experts.hits,
        ),
        (
            "expert_misses_ram_total",
            "counter",
            "Picked experts that were not in a slot and were in the RAM tier",
            stats.experts.from_ram,
        ),
        (
            "expert_misses_disk_total",
            "counter",
            "Picked experts that were not in a slot and only in the model file",
            stats.experts.from_disk,
        ),
        (
            "expert_misses_on_their_way_total",
            "counter",
            "Picked experts that were not in a slot and already on their way",
            stats.experts.on_their_way,
        ),
        (
            "expert_cpu_total",
            "counter",
            "Experts the CPU computed instead of the device",
            stats.experts.cpu_experts,
        ),
        (
            "expert_uploads_total",
            "counter",
            "Experts uploaded to a VRAM slot",
            stats.experts.uploads,
        ),
        (
            "expert_read_bytes_total",
            "counter",
            "Bytes of experts read from the model file",
            stats.experts.read_bytes,
        ),
        (
            "expert_wait_microseconds_total",
            "counter",
            "Time layers waited for the experts they missed, summed over layers",
            stats.experts.wait_us,
        ),
        (
            "expert_predicted_total",
            "counter",
            "Experts the routers of later layers, applied early, named for a layer",
            stats.experts.predicted,
        ),
        (
            "expert_predicted_picked_total",
            "counter",
            "Predicted experts the layer's own router then picked",
            stats.experts.predicted_picked,
        ),
        (
            "expert_ahead_reads_total",
            "counter",
            "Predicted experts in no tier read from the model file ahead of their router",
            stats.experts.ahead_reads,
        ),
        (
            "expert_ahead_read_bytes_total",
            "counter",
            "Bytes of experts read ahead of their router",
            stats.experts.ahead_read_bytes,
        ),
        (
            "expert_ahead_used_total",
            "counter",
            "Experts read ahead that their router then picked",
            stats.experts.ahead_used,
        ),
        (
            "expert_ahead_wasted_total",
            "counter",
            "Experts read ahead that their router did not pick",
            stats.experts.ahead_wasted,
        ),
        (
            "expert_ram_tier_bytes",
            "gauge",
            "Page-locked RAM the tier of experts holds now",
            stats.experts.ram_tier_bytes,
        ),
        ("expert_vram_experts", "gauge", "Experts in VRAM slots now", stats.experts.vram_experts),
        (
            "expert_ram_experts",
            "gauge",
            "Experts in the RAM tier now, copies of those in VRAM among them",
            stats.experts.ram_experts,
        ),
    ];
    let mut text = String::new();
    for (name, kind, help, value) in rows {
        text.push_str(&format!(
            "# HELP ares_{name} {help}\n# TYPE ares_{name} {kind}\nares_{name} {value}\n"
        ));
    }
    text
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
