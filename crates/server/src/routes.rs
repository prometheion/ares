//! The HTTP handlers.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ares_api::{Engine, RequestId, Usage};
use axum::Json;
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

use crate::Config;
use crate::bridge::{DecisionRun, Flight, Item, Run, Stop};
use crate::error::ApiError;
use crate::systemone;
use crate::wire::{self, Endpoint, Head};

/// What every handler needs.
#[derive(Clone)]
pub(crate) struct App {
    pub engine: Arc<dyn Engine>,
    pub config: Config,
    pub flight: Arc<Flight>,
    pub stop: Stop,
}

pub(crate) async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub(crate) async fn models(State(app): State<App>) -> Json<serde_json::Value> {
    Json(wire::model_list(&app.engine.models()))
}

/// `GET /metrics`: what the engine has done and holds, in the text format
/// monitoring systems scrape.
pub(crate) async fn metrics(State(app): State<App>) -> Response {
    // The engine may wait for the end of its step before it answers.
    let engine = app.engine.clone();
    let stats = tokio::task::spawn_blocking(move || engine.stats()).await.ok().flatten();
    match stats {
        Some(stats) => (
            [(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"))],
            wire::metrics(&stats),
        )
            .into_response(),
        None => ApiError::unsupported("the engine keeps no statistics").into_response(),
    }
}

pub(crate) async fn chat_completions(State(app): State<App>, request: Request) -> Response {
    completion(app, request, Endpoint::Chat).await
}

pub(crate) async fn completions(State(app): State<App>, request: Request) -> Response {
    completion(app, request, Endpoint::Completions).await
}

pub(crate) async fn systemone(State(app): State<App>, request: Request) -> Response {
    match try_systemone(app, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

pub(crate) async fn not_found(request: Request) -> ApiError {
    ApiError::not_found(request.uri().path())
}

pub(crate) async fn method_not_allowed(request: Request) -> ApiError {
    ApiError::method_not_allowed(request.method().as_str(), request.uri().path())
}

async fn completion(app: App, request: Request, endpoint: Endpoint) -> Response {
    match try_completion(app, request, endpoint).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn try_completion(
    app: App,
    request: Request,
    endpoint: Endpoint,
) -> Result<Response, ApiError> {
    let body = read_body(request, app.config.max_body_bytes).await?;
    let id = next_request_id();
    let parsed = wire::parse(endpoint, &body, id)?;
    let head = Head::new(endpoint, id, &parsed.request.model);
    tracing::info!(
        %id,
        model = %parsed.request.model,
        ?endpoint,
        stream = parsed.stream,
        max_tokens = ?parsed.request.max_tokens,
        "request"
    );

    let run = Run::start(
        &app.engine,
        parsed.request,
        app.config.event_buffer,
        &app.flight,
        app.stop.clone(),
    )?;
    if parsed.stream {
        stream(run, head, parsed.include_usage).await
    } else {
        collect(run, head).await
    }
}

/// `POST /v1/systemone`: one JSON answer, no streaming (the API has none).
///
/// Nothing here waits on the generation endpoints or on other decisions: the
/// handler hands the request to the engine and awaits its own outcome, and
/// the engine runs decision rows in the same steps as everything else.
async fn try_systemone(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, app.config.max_body_bytes).await?;
    let id = next_request_id();
    let parsed = systemone::parse(&body, id, &app.engine.models())?;
    tracing::info!(
        %id,
        model = %parsed.request.model,
        questions = parsed.request.questions.len(),
        "decision request"
    );

    let result =
        DecisionRun::start(&app.engine, parsed.request, app.stop.clone())?.finish().await?;
    Ok(json_response(parsed.reply.render(&result)?))
}

/// Reads the whole body, refusing more than `limit` bytes.
///
/// The limit is enforced here and not by the framework's extractors so that
/// the refusal comes out in the same error format as every other one, and so
/// that a prompt with a long context (megabytes of JSON) is allowed on
/// purpose instead of hitting the framework's small default.
async fn read_body(request: Request, limit: usize) -> Result<Vec<u8>, ApiError> {
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
fn next_request_id() -> RequestId {
    static NEXT: LazyLock<AtomicU64> = LazyLock::new(|| {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        AtomicU64::new(nanos as u64)
    });
    RequestId(NEXT.fetch_add(1, Ordering::Relaxed))
}

/// The non-streaming answer: collect everything, then reply once.
///
/// The text is held in memory until the end. Its size is bounded by the
/// request's token limit (or the context length), which is what the engine
/// would have spent on it anyway.
async fn collect(mut run: Run, head: Head) -> Result<Response, ApiError> {
    let mut text = String::new();
    loop {
        match run.next().await? {
            Item::Text(piece) => text.push_str(&piece),
            Item::Done { finish, usage, timings } => {
                return Ok(json_response(head.complete(&text, finish, usage, timings)));
            }
        }
    }
}

fn json_response(body: String) -> Response {
    ([(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))], body).into_response()
}

/// The streaming answer.
///
/// The response is not started before the engine has produced its first event.
/// A request that is going to fail (the prompt overflows the context, the
/// model does not exist) fails while the engine is still looking at it, and
/// that failure then becomes a proper HTTP status instead of an error message
/// inside a `200`. Clients and proxies handle the two very differently.
async fn stream(mut run: Run, head: Head, include_usage: bool) -> Result<Response, ApiError> {
    let first = run.next().await?;
    let state = Chunks { run, head, include_usage, phase: Phase::Start(first) };
    let events = futures_util::stream::unfold(state, |mut state| async move {
        let data = state.step().await?;
        Some((Ok::<_, Infallible>(SseEvent::default().data(data)), state))
    });
    let mut response = Sse::new(events).keep_alive(KeepAlive::default()).into_response();
    // Reverse proxies that buffer would hold the tokens back.
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    Ok(response)
}

/// Where the SSE state machine is. Every `step` produces one `data:` line.
enum Phase {
    /// The first item has arrived but nothing was sent yet.
    Start(Item),
    /// An item that is ready to be sent.
    Ready(Item),
    /// Waiting for the engine.
    Pull,
    /// The finish chunk went out; the usage chunk is next.
    Usage(Usage),
    /// Everything went out; `[DONE]` is next.
    Marker,
    End,
}

struct Chunks {
    run: Run,
    head: Head,
    include_usage: bool,
    phase: Phase,
}

impl Chunks {
    /// The payload of the next SSE event, or `None` when the stream is over.
    ///
    /// When a stream is dropped half way (the client disconnected), `run` is
    /// dropped with it and cancels the request in the engine.
    ///
    /// A failure is sent as an error object and ends the stream without
    /// `[DONE]`: a stream that ends with `[DONE]` reads as complete, and this
    /// one is not.
    async fn step(&mut self) -> Option<String> {
        loop {
            match std::mem::replace(&mut self.phase, Phase::End) {
                Phase::End => return None,
                Phase::Start(first) => {
                    self.phase = Phase::Ready(first);
                    if self.head.endpoint == Endpoint::Chat {
                        return Some(self.head.role_chunk());
                    }
                }
                Phase::Ready(Item::Text(text)) => {
                    self.phase = Phase::Pull;
                    // An empty piece would only add an empty chunk.
                    if !text.is_empty() {
                        return Some(self.head.text_chunk(&text));
                    }
                }
                Phase::Ready(Item::Done { finish, usage, timings }) => {
                    self.phase =
                        if self.include_usage { Phase::Usage(usage) } else { Phase::Marker };
                    return Some(self.head.finish_chunk(finish, timings));
                }
                Phase::Pull => match self.run.next().await {
                    Ok(item) => self.phase = Phase::Ready(item),
                    Err(error) => {
                        return Some(
                            serde_json::to_string(&error.body()).expect("errors serialize"),
                        );
                    }
                },
                Phase::Usage(usage) => {
                    self.phase = Phase::Marker;
                    return Some(self.head.usage_chunk(usage));
                }
                Phase::Marker => return Some("[DONE]".to_owned()),
            }
        }
    }
}

impl std::fmt::Debug for Chunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunks").field("run", &self.run).finish_non_exhaustive()
    }
}
