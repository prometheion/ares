//! The HTTP handlers.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ares_api::{GenerateRequest, RequestId, TokenProbs, Usage};
use axum::Json;
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

use crate::bridge::{DecisionRun, Item, Run, Stop, Timings};
use crate::defaults::LIMIT_HEADER;
use crate::error::ApiError;
use crate::models::{Lease, Listed, ModelEvent, Models};
use crate::reasoning::{self, Split};
use crate::reply::{Piece, Sorter};
use crate::systemone;
use crate::tools;
use crate::wire::{self, Call, Choice, Endpoint, Head, Parsed};
use crate::{Config, ReasoningFormat};

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
    let mut list = wire::model_list(&listed);
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
        () = stop.raised() => return Err(ApiError::cancelled()),
    };
    if let Some(id) = lease.id() {
        id.clone_into(model);
    }
    Ok(lease)
}

/// The body of `POST /models/load` and `/models/unload`.
fn named_model(body: &[u8]) -> Result<String, ApiError> {
    let json = wire::read_json(body)?;
    let mut name = None;
    for (key, value) in wire::members(&json)? {
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
    let Some((now, changes)) = app.models.watch() else {
        return ApiError::unsupported(
            "this server was given one engine and no presets: its model does not change state",
        )
        .into_response();
    };
    let first = now.into_iter().filter_map(|Listed { status, .. }| status).map(|s| {
        let ModelEvent { model, status, reason } = s;
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
    sse_response(futures_util::stream::iter(first).chain(later), KeepAlive::default())
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

/// `GET /metrics`: what an engine has done and holds, in the text format
/// monitoring systems scrape. The engine is the one of `?model=<id>`, or of
/// the only model there is; a model that is not loaded has none.
pub(crate) async fn metrics(State(app): State<App>, request: Request) -> Response {
    let name = match (model_in_query(request.uri().query()), app.models.offered().as_deref()) {
        (Some(name), _) => name,
        (None, Some([only])) => only.clone(),
        // One engine and no presets: there is nothing to choose.
        (None, None) => String::new(),
        (None, Some(_)) => {
            return ApiError::invalid_param(
                "model",
                "several models are served: name one, /metrics?model=<id>",
            )
            .into_response();
        }
    };
    let lease = match app.models.loaded(&name).await {
        Ok(lease) => lease,
        Err(error) => return ApiError::from(error).into_response(),
    };
    // The engine may wait for the end of its step before it answers.
    let latency = lease.flight().clone();
    let stats = tokio::task::spawn_blocking(move || lease.engine().stats()).await.ok().flatten();
    let latency = &latency.latency;
    match stats {
        Some(stats) => (
            [(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"))],
            wire::metrics(&stats) + &latency.render(),
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

/// `POST /v1/chat/completions/input_tokens`: the tokens of the prompt a chat
/// request comes to. The body is read as `/v1/chat/completions` reads it,
/// and nothing is generated.
pub(crate) async fn chat_input_tokens(State(app): State<App>, request: Request) -> Response {
    match try_chat_input_tokens(app, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn try_chat_input_tokens(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let mut parsed = wire::parse(Endpoint::Chat, &body, next_request_id())?;
    let (_, lease) = prepare(&app, &mut parsed, None).await?;
    let tokens = count_prompt(&lease, parsed.request).await?;
    Ok(json_response(wire::input_tokens(tokens)))
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

// --- the core of every route that generates ----------------------------------
//
// A route of any API runs a conversation in three steps, and none of them
// knows a wire format:
//
// 1. It reads its body into a [`Parsed`] chat request (`wire::from_json` for
//    a body in the chat format; a route of another API writes its body over
//    into that format first, as the `anthropic` module does).
// 2. [`prepare`] applies what the request says about thinking and tools to
//    it, and returns the [`Sorter`] of its reply.
// 3. [`Reply::start`] hands the request to the engine, and [`Reply::next`]
//    gives the reply as it comes: [`Piece`]s (reasoning, content, the start
//    of a call, fragments of its arguments) and at the end the reason, the
//    usage and the timings. The route writes them in its own format.
//
// A streaming route awaits the first step before it starts its response, so
// that a request the engine refuses gets an HTTP status; dropping the
// [`Reply`] (the client went away) cancels the request in the engine.

/// Takes the model a chat request names ([`model_of`]: the request waits
/// for its load), gives the request the model's defaults for what it leaves
/// out and holds the length of its reply to the model's limit (the
/// `defaults` module), applies what the request says about thinking and
/// about tools to the request for the engine (the variables of the chat
/// template, the rule of the reply), and returns what sorts the text of its
/// reply with the hold on the model. `format` is how the reasoning is
/// returned when the request does not say; `None`: as the model's own
/// options say.
pub(crate) async fn prepare(
    app: &App,
    parsed: &mut Parsed,
    format: Option<ReasoningFormat>,
) -> Result<(Sorter, Lease), ApiError> {
    let lease = model_of(app, &mut parsed.request.model).await?;
    lease.defaults().apply(parsed);
    let format = format.unwrap_or(lease.reasoning_format());
    let thinking = reasoning::apply(parsed, &**lease.engine(), format)?;
    let calls = tools::apply(parsed, &**lease.engine(), &thinking)?;
    log_request(app, parsed);
    Ok((Sorter::new(thinking.splitter, calls), lease))
}

/// The line of the log for a request that is taken: its parameters as the
/// engine gets them, and its prompt with `--log-text`.
fn log_request(app: &App, parsed: &Parsed) {
    let request = &parsed.request;
    let sampling = &request.sampling;
    let (tools, continues) = match &request.prompt {
        ares_api::Prompt::Chat { tools, continue_final, .. } => {
            (tools.as_ref().map_or(0, Vec::len), *continue_final)
        }
        ares_api::Prompt::Text(_) => (0, false),
    };
    tracing::info!(
        id = %request.id,
        model = %request.model,
        stream = parsed.stream,
        n = parsed.n,
        max_tokens = ?request.max_tokens,
        temperature = ?sampling.temperature,
        top_k = ?sampling.top_k,
        top_p = ?sampling.top_p,
        min_p = ?sampling.min_p,
        seed = ?sampling.seed,
        stops = request.stop.len(),
        tools,
        structure = request.structure.is_some(),
        draft = request.draft,
        continues,
        "request"
    );
    if app.config.log_text {
        let prompt = serde_json::to_string(&request.prompt).unwrap_or_default();
        tracing::info!(id = %request.id, %prompt, "the prompt");
    }
}

/// A reply the engine is generating: the request in the engine and the
/// sorter of its text. Dropping it before its end cancels the request.
#[derive(Debug)]
pub(crate) struct Reply {
    run: Run,
    sorter: Sorter,
    /// The request's stop strings, to name the one that ended the reply.
    stops: Vec<String>,
    /// Dropped after the run: the model may be unloaded when its requests
    /// have let go of the engine.
    _lease: Lease,
}

/// The next thing of a reply.
#[derive(Debug)]
pub(crate) enum Step {
    /// What the sorter found in the text that came; nothing when all of it
    /// is held back.
    Pieces(Vec<Piece>),
    /// The probabilities of the tokens whose text went out.
    Probs(Vec<TokenProbs>),
    End(End),
}

/// How a reply ended.
#[derive(Debug)]
pub(crate) struct End {
    /// What the sorter still held.
    pub rest: Vec<Piece>,
    /// `stop` (the model ended the reply, or a stop string did) or `length`.
    pub finish: &'static str,
    /// The stop string that ended the reply, if one did.
    pub stop: Option<String>,
    pub usage: Usage,
    pub timings: Timings,
}

impl Reply {
    /// Hands `request` to the engine of `lease`; `sorter` sorts its reply.
    pub fn start(
        app: &App,
        lease: &Lease,
        request: GenerateRequest,
        sorter: Sorter,
    ) -> Result<Self, ApiError> {
        let stops = request.stop.clone();
        let run = Run::start(
            lease.engine(),
            request,
            app.config.event_buffer,
            lease.flight(),
            app.stop.clone(),
            app.config.log_text,
        )?;
        Ok(Self { run, sorter, stops, _lease: lease.clone() })
    }

    /// The next step of the reply, or the reason there will be none. After
    /// an `Err` or a [`Step::End`] the reply is over.
    pub async fn next(&mut self) -> Result<Step, ApiError> {
        Ok(match self.run.next().await? {
            Item::Text(text) => Step::Pieces(self.sorter.push(&text)),
            Item::Probs(probs) => Step::Probs(probs),
            Item::Done { finish, stopped_by, usage, timings } => {
                let stop = stopped_by.and_then(|index| self.stops.get(index).cloned());
                Step::End(End { rest: self.sorter.finish(), finish, stop, usage, timings })
            }
        })
    }
}

/// How many tokens the prompt of `request` comes to: the request as a route
/// read and prepared it, counted by the engine and not run. An engine that
/// cannot count answers 501.
pub(crate) async fn count_prompt(lease: &Lease, request: GenerateRequest) -> Result<u32, ApiError> {
    let engine = lease.engine().clone();
    // Rendering and tokenizing a long prompt takes its time.
    let counted =
        tokio::task::spawn_blocking(move || engine.count_tokens(&request.model, &request.prompt))
            .await
            .map_err(|e| ApiError::internal(format!("the count of the prompt was lost: {e}")))?;
    counted.map_err(|e| ApiError::from_engine(&**lease.engine(), e))
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
pub(crate) fn sse_response<S>(events: S, keep_alive: KeepAlive) -> Response
where
    S: futures_util::Stream<Item = Result<SseEvent, Infallible>> + Send + 'static,
{
    let mut response = Sse::new(events).keep_alive(keep_alive).into_response();
    // Reverse proxies that buffer would hold the tokens back.
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-accel-buffering"), HeaderValue::from_static("no"));
    response
}

// --- the completion routes of the OpenAI API ---------------------------------

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
    let body = read_body(request, &app.config).await?;
    let id = next_request_id();
    let mut parsed = wire::parse(endpoint, &body, id)?;
    let (sorter, lease) = prepare(&app, &mut parsed, None).await?;
    let mut head = Head::new(endpoint, id, &parsed.request.model);
    head.logprobs = parsed.logprobs;

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
        runs.push(Reply::start(&app, &lease, request, sorter)?);
    }
    let response = if parsed.stream {
        stream(runs, head, parsed.include_usage).await
    } else {
        collect(runs, head).await
    };
    Ok(with_limit(response?, parsed.lowered))
}

/// `POST /v1/systemone`: one JSON answer, no streaming (the API has none).
///
/// Nothing here waits on the generation endpoints or on other decisions: the
/// handler hands the request to the engine and awaits its own outcome, and
/// the engine runs decision rows in the same steps as everything else.
async fn try_systemone(app: App, request: Request) -> Result<Response, ApiError> {
    let body = read_body(request, &app.config).await?;
    let id = next_request_id();
    let served: Vec<_> = app.models.list().into_iter().map(|listed| listed.id).collect();
    let mut parsed = systemone::parse(&body, id, &served)?;
    let lease = model_of(&app, &mut parsed.request.model).await?;
    tracing::info!(
        %id,
        model = %parsed.request.model,
        questions = parsed.request.questions.len(),
        "decision request"
    );

    let result =
        DecisionRun::start(lease.engine(), parsed.request, app.stop.clone())?.finish().await?;
    Ok(json_response(parsed.reply.render(&result)?))
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

/// What one completion of a non-streaming request came to.
struct Collected {
    text: Split,
    calls: Vec<Call>,
    probs: Vec<TokenProbs>,
    finish: &'static str,
    stop: Option<String>,
    usage: Usage,
    timings: Timings,
}

/// The reason a reply ended, as the answer says it: a reply that ended by
/// itself (or at a stop string) with calls in it ended to call them. A reply
/// the limit of tokens cut says so, whatever it holds.
fn finish_reason(finish: &'static str, calls: usize) -> &'static str {
    if finish == "stop" && calls > 0 { "tool_calls" } else { finish }
}

async fn collect_one(mut reply: Reply) -> Result<Collected, ApiError> {
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
        match reply.next().await? {
            Step::Pieces(pieces) => take(pieces),
            Step::Probs(more) => probs.extend(more),
            Step::End(End { rest, finish, stop, usage, timings }) => {
                take(rest);
                let finish = finish_reason(finish, calls.len());
                return Ok(Collected { text, calls, probs, finish, stop, usage, timings });
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
async fn collect(runs: Vec<Reply>, head: Head) -> Result<Response, ApiError> {
    let done = futures_util::future::try_join_all(runs.into_iter().map(collect_one)).await?;
    let choices: Vec<Choice<'_>> = done
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let choice = head.choice(i as u32);
            choice.complete_choice(&c.text, &c.calls, c.finish, c.stop.as_deref(), &c.probs)
        })
        .collect();
    // One prompt, however many completions of it.
    let usage = done.iter().skip(1).fold(done[0].usage, |sum, c| add_completion(sum, c.usage));
    Ok(json_response(head.complete(choices, usage, done[0].timings)))
}

pub(crate) fn json_response(body: String) -> Response {
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
async fn stream(runs: Vec<Reply>, head: Head, include_usage: bool) -> Result<Response, ApiError> {
    let many = runs.len() > 1;
    let firsts = futures_util::future::try_join_all(runs.into_iter().map(|mut reply| async move {
        let first = reply.next().await?;
        Ok::<_, ApiError>((reply, first))
    }))
    .await?;
    let shared = many.then(|| Arc::new(Shared::default()));
    let parts: Vec<_> = firsts
        .into_iter()
        .enumerate()
        .map(|(i, (reply, first))| {
            let state = Chunks {
                reply,
                head: head.choice(i as u32),
                include_usage,
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
    Ok(sse_response(events, KeepAlive::default()))
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
            Some(sum) => add_completion(sum, usage),
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

/// The usage of the completions of one prompt: the prompt once, the tokens
/// of every completion.
fn add_completion(sum: Usage, more: Usage) -> Usage {
    let reasoning = match (sum.reasoning_tokens, more.reasoning_tokens) {
        (Some(a), Some(b)) => Some(a + b),
        (a, b) => a.or(b),
    };
    Usage {
        completion_tokens: sum.completion_tokens + more.completion_tokens,
        reasoning_tokens: reasoning,
        ..sum
    }
}

/// Where the SSE state machine is. Every `step` produces one `data:` line.
enum Phase {
    /// The first step has arrived but nothing was sent yet.
    Start(Step),
    /// A step that is ready to be sent.
    Ready(Step),
    /// Waiting for the engine.
    Pull,
    /// The text is out; the finish chunk is next.
    Finish {
        finish: &'static str,
        stop: Option<String>,
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
    reply: Reply,
    head: Head,
    include_usage: bool,
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
    /// When a stream is dropped half way (the client disconnected), `reply`
    /// is dropped with it and cancels the request in the engine.
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
                Phase::Ready(Step::Pieces(pieces)) => {
                    self.phase = Phase::Pull;
                    // A piece that is held back or empty adds no chunk.
                    self.enqueue(pieces);
                }
                Phase::Ready(Step::Probs(probs)) => {
                    self.phase = Phase::Pull;
                    return Some(self.head.probs_chunk(&probs, self.sent));
                }
                Phase::Ready(Step::End(End { rest, finish, stop, usage, timings })) => {
                    self.enqueue(rest);
                    let finish = finish_reason(finish, self.calls);
                    self.phase = Phase::Finish { finish, stop, usage, timings };
                }
                Phase::Finish { finish, stop, usage, timings } => {
                    self.phase = match &self.shared {
                        Some(shared) => {
                            shared.add(usage);
                            Phase::End
                        }
                        None if self.include_usage => Phase::Usage(usage),
                        None => Phase::Marker,
                    };
                    return Some(self.head.finish_chunk(finish, stop.as_deref(), timings));
                }
                Phase::Pull => match self.reply.next().await {
                    Ok(step) => self.phase = Phase::Ready(step),
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
        f.debug_struct("Chunks").field("reply", &self.reply).finish_non_exhaustive()
    }
}
