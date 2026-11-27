//! The OpenAI chat and completion API: `POST /v1/chat/completions` and
//! `/v1/completions`, plain and streamed, with `n` completions of one
//! prompt side by side, and `POST /v1/chat/completions/input_tokens`, which
//! counts the prompt of a chat request. [`request`] reads a body, which is
//! the request every other API is written over into; [`reply`] writes the
//! answers, and the model list of `GET /v1/models`.

pub(crate) mod reply;
pub(crate) mod request;

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use ares_api::{TokenProbs, Usage};
use axum::extract::{Request, State};
use axum::response::sse::{Event as SseEvent, KeepAlive};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

use self::reply::{Call, Choice, Head};
use self::request::Endpoint;
use crate::bridge::Timings;
use crate::error::ApiError;
use crate::generate::{End, Generation, Step, count_prompt, prepare};
use crate::reasoning::Split;
use crate::routes::{App, json_response, next_request_id, read_body, sse_response, with_limit};
use crate::sorter::Piece;
use crate::tools;

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
    let mut parsed = request::parse(Endpoint::Chat, &body, next_request_id())?;
    let (_, lease) = prepare(&app, &mut parsed, None, "chat").await?;
    let tokens = count_prompt(&lease, parsed.request).await?;
    Ok(json_response(reply::input_tokens(tokens)))
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
    let priority = request.headers().get(request::PRIORITY_HEADER).cloned();
    let body = read_body(request, &app.config).await?;
    let id = next_request_id();
    let mut parsed = request::parse(endpoint, &body, id)?;
    parsed.take_priority(priority.as_ref())?;
    let (sorter, lease) = prepare(&app, &mut parsed, None, endpoint.api()).await?;
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
        let progress = parsed.stream && parsed.progress;
        runs.push(Generation::start(&app, &lease, request, sorter)?.reporting(progress));
    }
    let response = if parsed.stream {
        let pings = crate::routes::pings(KeepAlive::default(), parsed.ping);
        stream(runs, head, parsed.include_usage, app.config.stream_wait, pings).await
    } else {
        collect(runs, head).await
    };
    Ok(with_limit(response?, parsed.lowered))
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

async fn collect_one(mut reply: Generation) -> Result<Collected, ApiError> {
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
            Step::Progress(_) => {}
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
async fn collect(runs: Vec<Generation>, head: Head) -> Result<Response, ApiError> {
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

/// The streaming answer.
///
/// The response is not started before the engine has produced its first event
/// or `wait` has passed ([`Generation::first`]). A request that is going to fail
/// (the prompt overflows the context, the model does not exist) fails while
/// the engine is still looking at it, and that failure then becomes a proper
/// HTTP status instead of an error message inside a `200`. Clients and
/// proxies handle the two very differently.
///
/// The completions of an `n` above 1 are streamed side by side, their chunks
/// told apart by `index`; the usage and the end marker come once, after all.
async fn stream(
    runs: Vec<Generation>,
    head: Head,
    include_usage: bool,
    wait: Duration,
    pings: Option<KeepAlive>,
) -> Result<Response, ApiError> {
    let many = runs.len() > 1;
    let firsts = futures_util::future::try_join_all(runs.into_iter().map(|mut reply| async move {
        let first = reply.first(wait).await?;
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
    Ok(sse_response(events, pings))
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
    /// Nothing was sent yet; the first step, if it has arrived.
    Start(Option<Step>),
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
    reply: Generation,
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
                    self.phase = first.map_or(Phase::Pull, Phase::Ready);
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
                Phase::Ready(Step::Progress(progress)) => {
                    self.phase = Phase::Pull;
                    return Some(self.head.progress_chunk(&progress));
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
                        error.tell("the stream failed");
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
