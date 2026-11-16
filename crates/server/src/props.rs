//! What a model is and how it reads text: `GET /props`, `POST /tokenize` and
//! `POST /detokenize`, in the shapes of llama.cpp's server.
//!
//! `/tokenize` takes a text (`content`) and gives the tokens the engine reads
//! it as when it is the prompt of `/v1/completions`; or it takes a chat
//! request (`messages`, with `tools`, `chat_template_kwargs` and the other
//! fields of `/v1/chat/completions`) and gives the tokens of the prompt that
//! request comes to, through the chat template with the model's settings, as
//! `/v1/chat/completions/input_tokens` counts them. `with_pieces` adds the
//! text of each token: a string where the bytes are whole characters, else
//! the bytes. `/detokenize` takes `tokens` and gives the text they write in a
//! reply (`content`). A token the engine does not write as text (a control
//! token such as `<|im_end|>`) is the empty text in both.
//!
//! llama.cpp reads a text with its own switches (`add_special`,
//! `parse_special`); here a text is read the one way a prompt is, so that the
//! tokens are those a request gets, and the switches are refused by name.
//!
//! `/props` (`?model=<id>` when the server has several) says what a loaded
//! model is: its context, its chat template, what a request gets for the
//! fields it leaves out, what its template can do (tools, thinking and its
//! switches), and what input it takes (text alone: images are not built).

use ares_api::{Json, Prompt};
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};

use crate::error::ApiError;
use crate::models::Lease;
use crate::routes::{App, json_response, next_request_id, prepare, read_body};
use crate::wire::{self, Endpoint, Fields};

/// `GET /props`.
pub(crate) async fn props(State(app): State<App>, request: Request) -> Response {
    let asked = crate::routes::model_in_query(request.uri().query());
    match describe(&app, asked).await {
        Ok(body) => json_response(body.to_string()),
        Err(error) => error.into_response(),
    }
}

/// The loaded model a request names, or the only one of the server, and its
/// id. A model that is not loaded is not loaded for this.
async fn loaded(app: &App, asked: Option<String>) -> Result<(Lease, String), ApiError> {
    let name = match (asked, app.models.offered().as_deref()) {
        (Some(name), _) => name,
        (None, Some([only])) => only.clone(),
        (None, None) => String::new(),
        (None, Some(_)) => {
            return Err(ApiError::invalid_param(
                "model",
                "several models are served: name one (model, or ?model=<id>)",
            ));
        }
    };
    let lease = app.models.loaded(&name).await?;
    let id = match lease.id() {
        Some(id) => id.to_owned(),
        None if !name.is_empty() => name,
        // One engine: its first model.
        None => lease
            .engine()
            .models()
            .into_iter()
            .next()
            .map(|card| card.id)
            .ok_or_else(|| ApiError::internal("the engine serves no model"))?,
    };
    Ok((lease, id))
}

async fn describe(app: &App, asked: Option<String>) -> Result<serde_json::Value, ApiError> {
    let (lease, id) = loaded(app, asked).await?;
    let engine = lease.engine();
    let card = engine.models().into_iter().find(|card| card.id == id);
    let format = engine.chat_format(&id);
    let thinking = format.as_ref().and_then(|format| format.thinking.as_ref());
    let tools = format.as_ref().is_some_and(|format| format.tools.is_some());
    let defaults = lease.defaults();
    let mut params = defaults.fields();
    if let Some(limit) = defaults.max_tokens_limit {
        params["max_tokens_limit"] = limit.into();
    }
    let mut body = serde_json::json!({
        "model": id,
        "modalities": {"vision": false, "audio": false},
        "default_generation_settings": {"params": params},
        "chat_template_caps": {
            "supports_tools": tools,
            "supports_tool_calls": tools,
            "supports_parallel_tool_calls": tools,
            "supports_reasoning": thinking.is_some(),
            "supports_reasoning_effort": thinking.is_some_and(|t| t.effort),
            "supports_enable_thinking": thinking.is_some_and(|t| t.switchable()),
            "thinks_by_default": thinking.is_some_and(|t| t.on_by_default),
        },
    });
    if let Some(card) = &card {
        body["context_length"] = card.context_length.into();
        body["default_generation_settings"]["n_ctx"] = card.context_length.into();
    }
    if let Some(template) = engine.chat_template(&id) {
        body["chat_template"] = template.into();
    }
    if let Some(thinking) = thinking {
        body["reasoning"] = serde_json::json!({
            "start": thinking.start_tag(),
            "end": thinking.end_tag(),
        });
    }
    Ok(body)
}

/// `POST /tokenize`.
pub(crate) async fn tokenize(State(app): State<App>, request: Request) -> Response {
    match try_tokenize(app, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn try_tokenize(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let json = wire::read_json(&body)?;
    let mut members = wire::members(&json)?.to_vec();
    let with_pieces = take_bool(&mut members, "with_pieces")?;
    for name in ["add_special", "parse_special"] {
        if members.iter().any(|(key, value)| key == name && !value.is_null()) {
            return Err(ApiError::invalid_param(
                name,
                format!(
                    "{name} is not supported: a text is read as /v1/completions reads a prompt, \
                     so that the tokens are those a request gets"
                ),
            ));
        }
    }
    let (lease, prompt, model) = if members.iter().any(|(key, _)| key == "messages") {
        chat_prompt(&app, members).await?
    } else {
        text_prompt(&app, members).await?
    };
    let engine = lease.engine().clone();
    let tokens = blocking(move || {
        let tokens = engine.tokenize(&model, &prompt)?;
        let pieces = if with_pieces { Some(engine.pieces(&model, &tokens)?) } else { None };
        Ok((tokens, pieces))
    });
    let (tokens, pieces) = tokens.await.map_err(|e| ApiError::from_engine(&**lease.engine(), e))?;
    let tokens: Vec<serde_json::Value> = match pieces {
        None => tokens.into_iter().map(serde_json::Value::from).collect(),
        Some(pieces) => tokens
            .into_iter()
            .zip(pieces)
            .map(|(id, bytes)| {
                let piece = match String::from_utf8(bytes) {
                    Ok(text) => serde_json::Value::from(text),
                    Err(bytes) => serde_json::Value::from(bytes.into_bytes()),
                };
                serde_json::json!({"id": id, "piece": piece})
            })
            .collect(),
    };
    Ok(json_response(serde_json::json!({ "tokens": tokens }).to_string()))
}

/// A chat request's prompt, as the chat route prepares it.
async fn chat_prompt(
    app: &App,
    mut members: Vec<(String, Json)>,
) -> Result<(Lease, Prompt, String), ApiError> {
    if !members.iter().any(|(key, _)| key == "model") {
        let (_, id) = loaded(app, None).await?;
        members.push(("model".to_owned(), Json::String(id)));
    }
    let mut parsed = wire::from_json(Endpoint::Chat, &Json::Object(members), next_request_id())?;
    let (_, lease) = prepare(app, &mut parsed, None).await?;
    Ok((lease, parsed.request.prompt, parsed.request.model))
}

/// A text, as `/v1/completions` reads its prompt.
async fn text_prompt(
    app: &App,
    members: Vec<(String, Json)>,
) -> Result<(Lease, Prompt, String), ApiError> {
    let fields = Fields(&members);
    let content = match fields.get("content") {
        Some(Json::String(text)) => text.clone(),
        Some(other) => {
            return Err(ApiError::invalid_param(
                "content",
                format!("content must be a string, got {}", other.kind()),
            ));
        }
        None => {
            return Err(ApiError::invalid_param(
                "content",
                "content (a text) or messages (a conversation) is required",
            ));
        }
    };
    refuse_others(&members, &["content", "model"])?;
    let (lease, model) = named(app, &fields).await?;
    Ok((lease, Prompt::Text(content), model))
}

/// `POST /detokenize`.
pub(crate) async fn detokenize(State(app): State<App>, request: Request) -> Response {
    match try_detokenize(app, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn try_detokenize(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let json = wire::read_json(&body)?;
    let members = wire::members(&json)?;
    refuse_others(members, &["tokens", "model"])?;
    let fields = Fields(members);
    let wrong = || ApiError::invalid_param("tokens", "tokens must be an array of token ids");
    let tokens: Vec<u32> = match fields.get("tokens") {
        Some(Json::Array(items)) => items
            .iter()
            .map(|item| match item {
                Json::Int(id) => u32::try_from(*id).map_err(|_| wrong()),
                Json::Uint(id) => u32::try_from(*id).map_err(|_| wrong()),
                _ => Err(wrong()),
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(wrong()),
        None => return Err(ApiError::invalid_param("tokens", "tokens is required")),
    };
    let (lease, model) = named(&app, &fields).await?;
    let engine = lease.engine().clone();
    let pieces = blocking(move || engine.pieces(&model, &tokens))
        .await
        .map_err(|e| ApiError::from_engine(&**lease.engine(), e))?;
    let content = String::from_utf8_lossy(&pieces.concat()).into_owned();
    Ok(json_response(serde_json::json!({ "content": content }).to_string()))
}

/// The model of a body's `model`, or the only one of the server: loaded if it
/// is not, as for a request that generates.
async fn named(app: &App, fields: &Fields<'_>) -> Result<(Lease, String), ApiError> {
    match fields.string("model")? {
        Some(name) => {
            let mut model = name.to_owned();
            let lease = crate::routes::model_of(app, &mut model).await?;
            Ok((lease, model))
        }
        None => loaded(app, None).await,
    }
}

/// Runs work that may take a while (tokenizing a long text) where waiting is
/// allowed.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, ares_api::EngineError> + Send + 'static,
) -> Result<T, ares_api::EngineError> {
    tokio::task::spawn_blocking(work).await.unwrap_or_else(|e| {
        Err(ares_api::EngineError::Internal(format!("the work on the text was lost: {e}")))
    })
}

/// Takes a switch out of the members; it is not the chat request's.
fn take_bool(members: &mut Vec<(String, Json)>, name: &str) -> Result<bool, ApiError> {
    let Some(at) = members.iter().position(|(key, _)| key == name) else { return Ok(false) };
    match members.remove(at).1 {
        Json::Bool(value) => Ok(value),
        Json::Null => Ok(false),
        other => Err(ApiError::invalid_param(
            name,
            format!("{name} must be a boolean, got {}", other.kind()),
        )),
    }
}

/// Refuses a member that is not one of `known`.
fn refuse_others(members: &[(String, Json)], known: &[&str]) -> Result<(), ApiError> {
    match members.iter().find(|(key, value)| !known.contains(&key.as_str()) && !value.is_null()) {
        Some((key, _)) => Err(ApiError::invalid_param(
            key.as_str(),
            format!("'{key}' is not a member of this request: it has {}", known.join(", ")),
        )),
        None => Ok(()),
    }
}
