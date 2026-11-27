//! The routes of the server that are no API's own: the health check, the
//! model list and, in router mode, the loads, unloads and events of models,
//! and what is not found; and what every handler uses: [`App`], the model a
//! request names ([`model_of`]), the body ([`read_body`]), the ids of
//! requests, and the JSON and SSE responses.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ares_api::RequestId;
use axum::Json;
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

use crate::Config;
use crate::bridge::Stop;
use crate::defaults::LIMIT_HEADER;
use crate::error::ApiError;
use crate::models::{Lease, Listed, ModelEvent, Models};
use crate::openai::reply::model_list;
use crate::openai::request::{members, read_json};

/// What every handler needs.
#[derive(Clone)]
pub(crate) struct App {
    pub models: Arc<Models>,
    pub config: Config,
    pub stop: Stop,
}

pub(crate) async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// `GET /v1/models`: every model, and with each what a field means that a
/// request to it leaves out (`defaults`, under the names of the fields) and
/// the most tokens of a reply, if the model has such a limit.
pub(crate) async fn models(State(app): State<App>) -> Json<serde_json::Value> {
    let listed = app.models.list();
    let mut list = model_list(&listed);
    let models = list["data"].as_array_mut().into_iter().flatten();
    for (model, listed) in models.zip(&listed) {
        model["defaults"] = listed.defaults.fields();
        if let Some(limit) = listed.defaults.max_tokens_limit {
            model["max_tokens_limit"] = limit.into();
        }
    }
    Json(list)
}

/// The model a request names, loaded if it was not, which the request waits
/// for: behind a load that runs, and behind the requests of a model that has
/// to make room. `model` becomes the id the engine knows the model by, which
/// is what the answer says, whatever alias the request used. A client that
/// goes away gives the wait up, and so does the end of the server.
pub(crate) async fn model_of(app: &App, model: &mut String) -> Result<Lease, ApiError> {
    let mut stop = app.stop.clone();
    let lease = tokio::select! {
        lease = app.models.acquire(model) => lease?,
        error = stop.raised() => return Err(error),
    };
    lease.id().clone_into(model);
    Ok(lease)
}

/// The name of the model a request names (`asked`), or of the only one of the
/// server; with several, the request must name one.
pub(crate) fn wanted(app: &App, asked: Option<String>) -> Result<String, ApiError> {
    if let Some(name) = asked {
        return Ok(name);
    }
    match app.models.offered().as_slice() {
        [only] => Ok(only.clone()),
        _ => Err(ApiError::invalid_param(
            "model",
            "several models are served: name one (model, or ?model=<id>)",
        )),
    }
}

/// The body of `POST /models/load` and `/models/unload`.
fn named_model(body: &[u8]) -> Result<String, ApiError> {
    let json = read_json(body)?;
    let mut name = None;
    for (key, value) in members(&json)? {
        match (key.as_str(), value) {
            ("model", ares_api::Json::String(model)) => name = Some(model.clone()),
            ("model", _) => return Err(ApiError::invalid_param("model", "model is a string")),
            (other, _) => {
                return Err(ApiError::invalid_param(
                    other,
                    format!("'{other}' is not a member of this request: it has 'model'"),
                ));
            }
        }
    }
    name.ok_or_else(|| ApiError::invalid_param("model", "model is required"))
}

/// `POST /models/load`: puts a model in the line of loads and answers at
/// once; the load shows in `GET /v1/models` and in `GET /models/sse`.
pub(crate) async fn load_model(State(app): State<App>, request: Request) -> Response {
    let asked = async {
        let name = named_model(&read_body(request, &app.config).await?)?;
        app.models.ask_load(&name)?;
        Ok::<_, ApiError>(json_response(r#"{"success":true}"#.to_owned()))
    };
    asked.await.unwrap_or_else(IntoResponse::into_response)
}

/// `POST /models/unload`: the model gets no new requests and is unloaded
/// when those that run have ended. Answers at once.
pub(crate) async fn unload_model(State(app): State<App>, request: Request) -> Response {
    let asked = async {
        let name = named_model(&read_body(request, &app.config).await?)?;
        app.models.ask_unload(&name)?;
        Ok::<_, ApiError>(json_response(r#"{"success":true}"#.to_owned()))
    };
    asked.await.unwrap_or_else(IntoResponse::into_response)
}

fn status_event(model: &str, status: &str, reason: Option<&str>) -> Result<SseEvent, Infallible> {
    let mut data = serde_json::json!({ "status": status });
    if let Some(reason) = reason {
        data["reason"] = reason.into();
    }
    let event = serde_json::json!({ "model": model, "event": "model_status", "data": data });
    Ok(SseEvent::default().data(event.to_string()))
}

/// `GET /models/sse`: the state of every model as it is, then every change.
pub(crate) async fn model_events(State(app): State<App>) -> Response {
    let (now, changes) = app.models.watch();
    let first = now.into_iter().map(|Listed { status, .. }| {
        let ModelEvent { model, status, reason } = status;
        status_event(&model, status, reason.as_deref())
    });
    let later = futures_util::stream::unfold(changes, |mut changes| async move {
        loop {
            match changes.recv().await {
                Ok(ModelEvent { model, status, reason }) => {
                    return Some((status_event(&model, status, reason.as_deref()), changes));
                }
                // A reader too slow for the changes misses some; the next
                // ones are still true.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    sse_response(futures_util::stream::iter(first).chain(later), Some(KeepAlive::default()))
}

/// The value of `model` in the query of a URL (`?model=<id>`), with its
/// percent escapes read.
pub(crate) fn model_in_query(query: Option<&str>) -> Option<String> {
    let value = query?.split('&').find_map(|pair| pair.strip_prefix("model="))?;
    let mut bytes = Vec::with_capacity(value.len());
    let mut rest = value.as_bytes();
    while let [first, tail @ ..] = rest {
        let escaped = match (first, tail) {
            (b'%', [high, low, ..]) => std::str::from_utf8(&[*high, *low])
                .ok()
                .and_then(|hex| u8::from_str_radix(hex, 16).ok()),
            _ => None,
        };
        match escaped {
            Some(byte) => {
                bytes.push(byte);
                rest = &tail[2..];
            }
            None => {
                bytes.push(if *first == b'+' { b' ' } else { *first });
                rest = tail;
            }
        }
    }
    String::from_utf8(bytes).ok()
}

/// Writes every refusal and failure that a client gets as its response into
/// the log, with its route: the error the response carries
/// ([`Refused`](crate::error::Refused): its status, code and message), or its
/// status alone. A failure in the middle of a stream is told where the stream
/// writes it.
pub(crate) async fn tell_refusals(request: Request, next: axum::middleware::Next) -> Response {
    let (method, uri) = (request.method().clone(), request.uri().clone());
    let response = next.run(request).await;
    let status = response.status();
    if status.is_client_error() || status.is_server_error() {
        let at = format!("{method} {}", uri.path());
        match response.extensions().get::<crate::error::Refused>() {
            Some(refused) => refused.0.tell(&at),
            None => tracing::warn!("{at}: {status}"),
        }
    }
    response
}

pub(crate) async fn not_found(request: Request) -> ApiError {
    ApiError::not_found(request.uri().path())
}

pub(crate) async fn method_not_allowed(request: Request) -> ApiError {
    ApiError::method_not_allowed(request.method().as_str(), request.uri().path())
}

/// Says in a header of `response` that the reply got `lowered` tokens at
/// most in place of the larger limit its request asked for
/// ([`LIMIT_HEADER`]); nothing for a request the model's limit left alone.
pub(crate) fn with_limit(mut response: Response, lowered: Option<u32>) -> Response {
    if let Some(limit) = lowered {
        response.headers_mut().insert(HeaderName::from_static(LIMIT_HEADER), limit.into());
    }
    response
}

/// The response of a stream of events. `keep_alive` is what goes out while
/// the engine has nothing to say.
pub(crate) fn sse_response<S>(events: S, keep_alive: Option<KeepAlive>) -> Response
where
    S: futures_util::Stream<Item = Result<SseEvent, Infallible>> + Send + 'static,
{
    let mut response = match keep_alive {
        Some(keep_alive) => Sse::new(events).keep_alive(keep_alive).into_response(),
        None => Sse::new(events).into_response(),
    };
    // Reverse proxies that buffer would hold the tokens back.
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    response
}

/// The pings of a stream: the route's own (`route`, every 15 s of silence),
/// at the interval a request asks for with llama.cpp's `sse_ping_interval`
/// (seconds of silence before a ping; 0 or -1: none), as llama.cpp's server
/// sends them while a stream stays silent (`tools/server/server-context.cpp` of
/// <https://github.com/ggml-org/llama.cpp>, commit d92ea9b, MIT).
pub(crate) fn pings(route: KeepAlive, asked: Option<i32>) -> Option<KeepAlive> {
    match asked {
        None => Some(route),
        Some(seconds) if seconds > 0 => {
            Some(route.interval(std::time::Duration::from_secs(seconds.unsigned_abs().into())))
        }
        Some(_) => None,
    }
}

/// Reads the whole body, refusing more than the configuration's limit of
/// bytes, and a body that does not arrive within its timeout.
///
/// The limit is enforced here and not by the framework's extractors so that
/// the refusal comes out in the same error format as every other one, and so
/// that a prompt with a long context (megabytes of JSON) is allowed on
/// purpose instead of hitting the framework's small default.
pub(crate) async fn read_body(request: Request, config: &Config) -> Result<Vec<u8>, ApiError> {
    let timeout = config.timeout;
    tokio::time::timeout(timeout, read_whole(request, config.max_body_bytes))
        .await
        .unwrap_or_else(|_| Err(ApiError::too_slow(timeout)))
}

async fn read_whole(request: Request, limit: usize) -> Result<Vec<u8>, ApiError> {
    let declared = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|len| len > limit) {
        return Err(ApiError::too_large(limit));
    }
    let mut chunks = request.into_body().into_data_stream();
    let mut body = Vec::with_capacity(declared.unwrap_or(0).min(1 << 20));
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk
            .map_err(|e| ApiError::invalid(format!("could not read the request body: {e}")))?;
        if body.len() + chunk.len() > limit {
            return Err(ApiError::too_large(limit));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Ids are unique within the process and, because the counter starts at the
/// time of the first request, almost surely across restarts, so that a
/// `chatcmpl-...` id found in a client's log points at one request.
pub(crate) fn next_request_id() -> RequestId {
    static NEXT: LazyLock<AtomicU64> = LazyLock::new(|| {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        AtomicU64::new(nanos as u64)
    });
    RequestId(NEXT.fetch_add(1, Ordering::Relaxed))
}

pub(crate) fn json_response(body: String) -> Response {
    ([(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))], body).into_response()
}
