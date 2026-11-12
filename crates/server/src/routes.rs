//! The HTTP handlers.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ares_api::{Engine, RequestId, TokenProbs, Usage};
use axum::Json;
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

use crate::Config;
use crate::bridge::{DecisionRun, Flight, Item, Run, Stop, Timings};
use crate::error::ApiError;
use crate::reasoning::{self, Split};
use crate::reply::{Piece, Sorter};
use crate::systemone;
use crate::tools;
use crate::wire::{self, Call, Choice, Endpoint, Head};

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
    let mut parsed = wire::parse(endpoint, &body, id)?;
    let thinking = reasoning::apply(&mut parsed, &*app.engine, app.config.reasoning_format)?;
    let calls = tools::apply(&mut parsed, &*app.engine, &thinking)?;
    let sorter = Sorter::new(thinking.splitter, calls);
    let mut head = Head::new(endpoint, id, &parsed.request.model);
    head.logprobs = parsed.logprobs;
    tracing::info!(
        %id,
        model = %parsed.request.model,
        ?endpoint,
        stream = parsed.stream,
        n = parsed.n,
        max_tokens = ?parsed.request.max_tokens,
        "request"
    );

    // The completions of a prompt are requests of their own, with their own
    // seeds: the engine computes the prompt once and the other requests start
    // from what it cached.
    let mut runs = Vec::with_capacity(parsed.n as usize);
    for i in 0..parsed.n {
        let mut request = parsed.request.clone();
        if i > 0 {
            request.id = next_request_id();
            request.sampling.seed =
                request.sampling.seed.map(|seed| seed.wrapping_add(u64::from(i)));
        }
        let sorter = sorter.with_call_ids(tools::call_ids(request.id));
        runs.push((
            sorter,
            Run::start(
                &app.engine,
                request,
                app.config.event_buffer,
                &app.flight,
                app.stop.clone(),
            )?,
        ));
    }
    if parsed.stream {
        stream(runs, head, parsed.include_usage).await
    } else {
        collect(runs, head).await
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

/// What one completion of a non-streaming request came to.
struct Collected {
    text: Split,
    calls: Vec<Call>,
    probs: Vec<TokenProbs>,
    finish: &'static str,
    usage: Usage,
    timings: Timings,
}

/// The reason a reply ended, as the answer says it: a reply that ended by
/// itself (or at a stop string) with calls in it ended to call them. A reply
/// the limit of tokens cut says so, whatever it holds.
fn finish_reason(finish: &'static str, calls: usize) -> &'static str {
    if finish == "stop" && calls > 0 { "tool_calls" } else { finish }
}

async fn collect_one(mut run: Run, mut sorter: Sorter) -> Result<Collected, ApiError> {
    let mut text = Split::default();
    let mut calls: Vec<Call> = Vec::new();
    let mut probs = Vec::new();
    let mut take = |pieces: Vec<Piece>| {
        for piece in pieces {
            match piece {
                Piece::Text(sorted) => {
                    text.reasoning.push_str(&sorted.reasoning);
                    text.content.push_str(&sorted.content);
                }
                Piece::Call { id, name, .. } => {
                    calls.push(Call { id, name, arguments: String::new() });
                }
                Piece::Arguments { index, json } => calls[index].arguments.push_str(&json),
            }
        }
    };
    loop {
        match run.next().await? {
            Item::Text(piece) => take(sorter.push(&piece)),
            Item::Probs(more) => probs.extend(more),
            Item::Done { finish, usage, timings } => {
                take(sorter.finish());
                let finish = finish_reason(finish, calls.len());
                return Ok(Collected { text, calls, probs, finish, usage, timings });
            }
        }
    }
}

/// The non-streaming answer: collect everything, then reply once.
///
/// The text is held in memory until the end. Its size is bounded by the
/// request's token limit (or the context length), which is what the engine
/// would have spent on it anyway. The completions of an `n` above 1 are
/// collected side by side, so that none of them waits in its queue while
/// another is being read.
async fn collect(runs: Vec<(Sorter, Run)>, head: Head) -> Result<Response, ApiError> {
    let done = futures_util::future::try_join_all(
        runs.into_iter().map(|(sorter, run)| collect_one(run, sorter)),
    )
    .await?;
    let choices: Vec<Choice<'_>> = done
        .iter()
        .enumerate()
        .map(|(i, c)| head.choice(i as u32).complete_choice(&c.text, &c.calls, c.finish, &c.probs))
        .collect();
    // One prompt, however many completions of it.
    let usage = Usage {
        completion_tokens: done.iter().map(|c| c.usage.completion_tokens).sum(),
        ..done[0].usage
    };
    Ok(json_response(head.complete(choices, usage, done[0].timings)))
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
///
/// The completions of an `n` above 1 are streamed side by side, their chunks
/// told apart by `index`; the usage and the end marker come once, after all.
async fn stream(
    runs: Vec<(Sorter, Run)>,
    head: Head,
    include_usage: bool,
) -> Result<Response, ApiError> {
    let many = runs.len() > 1;
    let firsts =
        futures_util::future::try_join_all(runs.into_iter().map(|(sorter, mut run)| async move {
            let first = run.next().await?;
            Ok::<_, ApiError>((sorter, run, first))
        }))
        .await?;
    let shared = many.then(|| Arc::new(Shared::default()));
    let parts: Vec<_> = firsts
        .into_iter()
        .enumerate()
        .map(|(i, (sorter, run, first))| {
            let state = Chunks {
                run,
                head: head.choice(i as u32),
                include_usage,
                sorter,
                phase: Phase::Start(first),
                queue: VecDeque::new(),
                calls: 0,
                sent: 0,
                shared: shared.clone(),
            };
            futures_util::stream::unfold(state, |mut state| async move {
                let data = state.step().await?;
                Some((Ok::<_, Infallible>(SseEvent::default().data(data)), state))
            })
            .boxed()
        })
        .collect();
    let merged = futures_util::stream::select_all(parts);
    let events = match shared {
        // The end of several streams is one: the usage, then the marker.
        Some(shared) => merged
            .chain(
                futures_util::stream::once(async move { shared.end(&head, include_usage) })
                    .flat_map(futures_util::stream::iter),
            )
            .boxed(),
        None => merged.boxed(),
    };
    let mut response = Sse::new(events).keep_alive(KeepAlive::default()).into_response();
    // Reverse proxies that buffer would hold the tokens back.
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    Ok(response)
}

/// What the completions of one request share while they stream.
#[derive(Default)]
struct Shared {
    usage: std::sync::Mutex<Option<Usage>>,
    failed: std::sync::atomic::AtomicBool,
}

impl Shared {
    fn add(&self, usage: Usage) {
        let mut total = self.usage.lock().unwrap_or_else(|p| p.into_inner());
        *total = Some(match *total {
            None => usage,
            Some(sum) => {
                Usage { completion_tokens: sum.completion_tokens + usage.completion_tokens, ..sum }
            }
        });
    }

    /// The chunks after the last completion: the usage if it was asked for,
    /// and the marker; none if a completion failed.
    fn end(&self, head: &Head, include_usage: bool) -> Vec<Result<SseEvent, Infallible>> {
        if self.failed.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut events = Vec::new();
        if include_usage && let Some(usage) = *self.usage.lock().unwrap_or_else(|p| p.into_inner())
        {
            events.push(Ok(SseEvent::default().data(head.usage_chunk(usage))));
        }
        events.push(Ok(SseEvent::default().data("[DONE]")));
        events
    }
}

/// Where the SSE state machine is. Every `step` produces one `data:` line.
enum Phase {
    /// The first item has arrived but nothing was sent yet.
    Start(Item),
    /// An item that is ready to be sent.
    Ready(Item),
    /// Waiting for the engine.
    Pull,
    /// The text is out; the finish chunk is next.
    Finish {
        finish: &'static str,
        usage: Usage,
        timings: Timings,
    },
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
    sorter: Sorter,
    phase: Phase,
    /// Chunks that are ready and have not gone out: a piece of the reply can
    /// be several (content, the start of a call, its arguments).
    queue: VecDeque<String>,
    /// How many calls the reply has begun.
    calls: usize,
    /// Characters of text sent so far, for the offsets of probabilities.
    sent: usize,
    /// Set when this is one of several completions.
    shared: Option<Arc<Shared>>,
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
            if let Some(chunk) = self.queue.pop_front() {
                return Some(chunk);
            }
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
                    // A piece that is held back or empty adds no chunk.
                    let pieces = self.sorter.push(&text);
                    self.enqueue(pieces);
                }
                Phase::Ready(Item::Probs(probs)) => {
                    self.phase = Phase::Pull;
                    return Some(self.head.probs_chunk(&probs, self.sent));
                }
                Phase::Ready(Item::Done { finish, usage, timings }) => {
                    let rest = self.sorter.finish();
                    self.enqueue(rest);
                    let finish = finish_reason(finish, self.calls);
                    self.phase = Phase::Finish { finish, usage, timings };
                }
                Phase::Finish { finish, usage, timings } => {
                    self.phase = match &self.shared {
                        Some(shared) => {
                            shared.add(usage);
                            Phase::End
                        }
                        None if self.include_usage => Phase::Usage(usage),
                        None => Phase::Marker,
                    };
                    return Some(self.head.finish_chunk(finish, timings));
                }
                Phase::Pull => match self.run.next().await {
                    Ok(item) => self.phase = Phase::Ready(item),
                    Err(error) => {
                        if let Some(shared) = &self.shared {
                            shared.failed.store(true, Ordering::Release);
                        }
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

impl Chunks {
    /// Queues the chunks of what the sorter found in a piece of the reply.
    fn enqueue(&mut self, pieces: Vec<Piece>) {
        for piece in pieces {
            let chunk = match piece {
                Piece::Text(sorted) => {
                    self.sent += sorted.content.chars().count();
                    self.head.text_chunk(&sorted)
                }
                Piece::Call { index, id, name } => {
                    self.calls += 1;
                    self.head.call_chunk(index, &id, &name)
                }
                Piece::Arguments { index, json } => self.head.arguments_chunk(index, &json),
            };
            self.queue.push_back(chunk);
        }
    }
}

impl std::fmt::Debug for Chunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunks").field("run", &self.run).finish_non_exhaustive()
    }
}
