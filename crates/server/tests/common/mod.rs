//! Test support: a scripted engine and helpers to drive the router.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use ares_api::{
    ChatFormat, DecisionOutcome, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats,
    Event, EventSink, FinishReason, GenerateRequest, ModelCard, Prompt, RequestId, ThinkingFormat,
    ToolFormat, Usage,
};
use ares_server::Config;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, Response, StatusCode, header};
use serde_json::json;
use tower::ServiceExt;

pub mod process;
pub mod socket;

/// How long an engine thread or a test waits for the other side before it
/// fails, so that a broken test ends instead of hanging.
const WAIT_LIMIT: Duration = Duration::from_secs(10);

/// A latch a test opens to let the engine's thread continue.
#[derive(Clone, Default)]
pub struct Gate(Arc<(Mutex<bool>, Condvar)>);

impl Gate {
    pub fn open(&self) {
        *self.0.0.lock().unwrap() = true;
        self.0.1.notify_all();
    }

    pub fn wait(&self) {
        let (lock, condvar) = &*self.0;
        let (_open, timeout) =
            condvar.wait_timeout_while(lock.lock().unwrap(), WAIT_LIMIT, |open| !*open).unwrap();
        assert!(!timeout.timed_out(), "the test never opened the gate");
    }
}

/// One thing the engine thread does.
#[derive(Clone)]
pub enum Step {
    /// Push an event into the sink. The thread stops when the sink refuses.
    Send(Event),
    /// Block until a test opens the gate.
    Wait(Gate),
    /// Block until the front end calls `cancel` for this request, then end it
    /// the way a real engine would.
    UntilCancelled,
    /// Give up the sink without a terminal event.
    Abandon,
    /// End the request with `finish` and the usage of its prompt as
    /// [`tokens`] counts it.
    DoneCounted { finish: FinishReason, completion: u32 },
}

pub fn text(piece: &str) -> Step {
    Step::Send(Event::Text(piece.to_owned()))
}

pub fn done(finish: FinishReason, prompt: u32, completion: u32) -> Step {
    Step::Send(Event::Done {
        finish,
        usage: Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            cached_prompt_tokens: 0,
            ..Usage::default()
        },
    })
}

pub fn fail(error: EngineError) -> Step {
    Step::Send(Event::Error(error))
}

/// The reply of a model in pieces of `size` characters, ended by `finish`.
pub fn says(reply: &str, size: usize, finish: FinishReason) -> Vec<Step> {
    let chars: Vec<char> = reply.chars().collect();
    let mut steps: Vec<Step> =
        chars.chunks(size).map(|piece| text(&piece.iter().collect::<String>())).collect();
    steps.push(done(finish, 20, chars.len() as u32));
    steps
}

/// What stands for the tokens of a prompt in these tests: the bytes of the
/// prompt as the engine was given it.
pub fn tokens(prompt: &Prompt) -> u32 {
    serde_json::to_string(prompt).unwrap().len() as u32
}

/// What `submit` does.
#[derive(Clone)]
pub enum Plan {
    /// Accept, then run the steps on a separate thread.
    Script(Vec<Step>),
    /// Refuse at once.
    Reject(EngineError),
}

/// One thing the engine thread of a decision does.
#[derive(Clone)]
pub enum DecisionStep {
    /// Block until a test opens the gate.
    Wait(Gate),
    /// Deliver the outcome made from the request the engine received. The
    /// decision ends here.
    Deliver(Arc<dyn Fn(&DecisionRequest) -> DecisionOutcome + Send + Sync>),
    /// Block until the front end calls `cancel` for this request, then end it
    /// the way a real engine would.
    UntilCancelled,
    /// Give up the sink without an outcome.
    Abandon,
}

/// Deliver `outcome` whatever the request was.
pub fn deliver(outcome: DecisionOutcome) -> DecisionStep {
    DecisionStep::Deliver(Arc::new(move |_| outcome.clone()))
}

/// Deliver the outcome `make` builds from the request.
pub fn deliver_with(
    make: impl Fn(&DecisionRequest) -> DecisionOutcome + Send + Sync + 'static,
) -> DecisionStep {
    DecisionStep::Deliver(Arc::new(make))
}

/// What `decide` does.
#[derive(Clone)]
pub enum DecisionPlan {
    /// Accept, then run the steps on a separate thread.
    Script(Vec<DecisionStep>),
    /// Refuse at once.
    Reject(EngineError),
}

#[derive(Default)]
struct Book {
    submitted: Vec<GenerateRequest>,
    decided: Vec<DecisionRequest>,
    cancelled: Vec<RequestId>,
    /// What every `sink.send` returned, in order.
    sends: Vec<bool>,
    /// What `stats` answers; `None` for an engine that keeps none.
    stats: Option<EngineStats>,
    /// What the chat template of `mock` declares.
    format: Option<ChatFormat>,
    /// Whether the engine counts the tokens of a prompt.
    counting: bool,
    /// Why the engine serves no more requests, once a test broke it.
    failure: Option<String>,
}

/// The part of the engine its threads share with the test.
#[derive(Default)]
struct Shared {
    book: Mutex<Book>,
    changed: Condvar,
}

impl Shared {
    fn record(&self, change: impl FnOnce(&mut Book)) {
        change(&mut self.book.lock().unwrap());
        self.changed.notify_all();
    }

    fn wait_for_cancel(&self, id: RequestId) {
        let (_book, timeout) = self
            .changed
            .wait_timeout_while(self.book.lock().unwrap(), WAIT_LIMIT, |b| {
                !b.cancelled.contains(&id)
            })
            .unwrap();
        assert!(!timeout.timed_out(), "the front end never cancelled the request");
    }

    fn run(&self, id: RequestId, prompt_tokens: u32, steps: Vec<Step>, sink: EventSink) {
        for step in steps {
            match step {
                Step::DoneCounted { finish, completion } => {
                    let usage =
                        Usage { prompt_tokens, completion_tokens: completion, ..Usage::default() };
                    sink.send(Event::Done { finish, usage });
                }
                Step::Send(event) => {
                    let delivered = sink.send(event);
                    self.record(|book| book.sends.push(delivered));
                    if !delivered {
                        return;
                    }
                }
                Step::Wait(gate) => gate.wait(),
                Step::UntilCancelled => {
                    self.wait_for_cancel(id);
                    sink.send(Event::Done {
                        finish: FinishReason::Cancelled,
                        usage: Usage::default(),
                    });
                }
                Step::Abandon => return,
            }
        }
    }

    fn run_decision(
        &self,
        id: RequestId,
        request: &DecisionRequest,
        steps: Vec<DecisionStep>,
        sink: DecisionSink,
    ) {
        for step in steps {
            match step {
                DecisionStep::Wait(gate) => gate.wait(),
                DecisionStep::Deliver(make) => {
                    sink.send(make(request));
                    return;
                }
                DecisionStep::UntilCancelled => {
                    self.wait_for_cancel(id);
                    sink.send(DecisionOutcome::Cancelled);
                    return;
                }
                DecisionStep::Abandon => return,
            }
        }
    }
}

/// An engine that does what its plans say, each request on its own thread, so
/// that a slow generation and a decision run at the same time as they do in
/// the real engine.
pub struct MockEngine {
    models: Vec<ModelCard>,
    plan: Plan,
    decisions: DecisionPlan,
    shared: Arc<Shared>,
}

impl MockEngine {
    /// Serves generation by `plan`. Decisions are refused.
    pub fn new(plan: Plan) -> Arc<Self> {
        let refusal = EngineError::Unsupported("the mock engine was given no decisions".into());
        Self::both(plan, DecisionPlan::Reject(refusal))
    }

    /// Serves decisions by `decisions`. Generation is refused.
    pub fn deciding(decisions: DecisionPlan) -> Arc<Self> {
        let refusal = EngineError::Unsupported("the mock engine was given no generation".into());
        Self::both(Plan::Reject(refusal), decisions)
    }

    /// Serves both kinds of request.
    pub fn both(plan: Plan, decisions: DecisionPlan) -> Arc<Self> {
        let models = vec![
            ModelCard { id: "mock".into(), context_length: 4096, created: 1_700_000_000 },
            ModelCard { id: "mock-2".into(), context_length: 8192, created: 1_700_000_001 },
        ];
        Self::with_models(models, plan, decisions)
    }

    pub fn with_models(models: Vec<ModelCard>, plan: Plan, decisions: DecisionPlan) -> Arc<Self> {
        Arc::new(Self { models, plan, decisions, shared: Arc::default() })
    }

    pub fn script(steps: Vec<Step>) -> Arc<Self> {
        Self::new(Plan::Script(steps))
    }

    pub fn reject(error: EngineError) -> Arc<Self> {
        Self::new(Plan::Reject(error))
    }

    pub fn submitted(&self) -> Vec<GenerateRequest> {
        self.shared.book.lock().unwrap().submitted.clone()
    }

    /// The decision requests the engine accepted or refused, in order.
    pub fn decided(&self) -> Vec<DecisionRequest> {
        self.shared.book.lock().unwrap().decided.clone()
    }

    pub fn cancelled(&self) -> Vec<RequestId> {
        self.shared.book.lock().unwrap().cancelled.clone()
    }

    pub fn sends(&self) -> Vec<bool> {
        self.shared.book.lock().unwrap().sends.clone()
    }

    /// Makes the engine one that keeps statistics, and sets what they say.
    pub fn set_stats(&self, stats: EngineStats) {
        self.shared.book.lock().unwrap().stats = Some(stats);
    }

    /// Makes `mock` a model whose chat template declares `format`.
    pub fn set_chat_format(&self, format: ChatFormat) {
        self.shared.book.lock().unwrap().format = Some(format);
    }

    /// Makes the engine count the tokens of a prompt of `mock` as [`tokens`]
    /// does.
    pub fn set_counting(&self) {
        self.shared.book.lock().unwrap().counting = true;
    }

    /// The engine as the Messages and Responses tests serve it: `mock`
    /// declares Qwen3.8's format ([`qwen`]) and the tokens of a prompt are
    /// counted.
    pub fn qwen(self: Arc<Self>) -> Arc<Self> {
        self.set_chat_format(qwen());
        self.set_counting();
        self
    }

    /// Breaks the engine: it reports `why` as its failure.
    pub fn set_failure(&self, why: &str) {
        self.shared.book.lock().unwrap().failure = Some(why.to_owned());
    }
}

impl Engine for MockEngine {
    fn models(&self) -> Vec<ModelCard> {
        self.models.clone()
    }

    fn chat_format(&self, model: &str) -> Option<ChatFormat> {
        (model == "mock").then(|| self.shared.book.lock().unwrap().format.clone()).flatten()
    }

    fn count_tokens(&self, model: &str, prompt: &Prompt) -> Result<u32, EngineError> {
        if !self.shared.book.lock().unwrap().counting {
            let what = "the scripted engine does not count the tokens of a prompt";
            return Err(EngineError::Unsupported(what.into()));
        }
        if model != "mock" {
            return Err(EngineError::ModelNotFound(model.to_owned()));
        }
        Ok(tokens(prompt))
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        let (id, prompt_tokens) = (request.id, tokens(&request.prompt));
        self.shared.record(|book| book.submitted.push(request));
        match &self.plan {
            Plan::Reject(error) => Err(error.clone()),
            Plan::Script(steps) => {
                // Like a real engine, `submit` returns at once and a thread of
                // the engine's own pushes the events.
                let (shared, steps) = (self.shared.clone(), steps.clone());
                thread::Builder::new()
                    .name("mock-engine".into())
                    .spawn(move || shared.run(id, prompt_tokens, steps, events))
                    .unwrap();
                Ok(())
            }
        }
    }

    fn decide(&self, request: DecisionRequest, outcome: DecisionSink) -> Result<(), EngineError> {
        let id = request.id;
        self.shared.record(|book| book.decided.push(request.clone()));
        match &self.decisions {
            DecisionPlan::Reject(error) => Err(error.clone()),
            DecisionPlan::Script(steps) => {
                let (shared, steps) = (self.shared.clone(), steps.clone());
                thread::Builder::new()
                    .name("mock-engine-decision".into())
                    .spawn(move || shared.run_decision(id, &request, steps, outcome))
                    .unwrap();
                Ok(())
            }
        }
    }

    fn cancel(&self, id: RequestId) {
        self.shared.record(|book| book.cancelled.push(id));
    }

    fn stats(&self) -> Option<EngineStats> {
        self.shared.book.lock().unwrap().stats
    }

    fn failure(&self) -> Option<String> {
        self.shared.book.lock().unwrap().failure.clone()
    }
}

// --- what the models declare ------------------------------------------------

/// What the template of Qwen3.8 declares.
pub fn qwen() -> ChatFormat {
    ChatFormat { thinking: Some(qwen_thinking()), tools: Some(qwen_tools()) }
}

/// How the template of Qwen3.8 thinks.
pub fn qwen_thinking() -> ThinkingFormat {
    ThinkingFormat {
        start: "<think>\n".into(),
        end: "\n</think>\n\n".into(),
        prompt_on: "<think>\n".into(),
        prompt_off: "<think>\n\n</think>\n\n".into(),
        on_by_default: true,
        effort: true,
        refused_efforts: Vec::new(),
    }
}

/// How the template of Qwen3.8 writes calls of tools.
pub fn qwen_tools() -> ToolFormat {
    ToolFormat {
        call_open: "<tool_call>\n<function=".into(),
        name_close: ">\n".into(),
        param_open: "<parameter=".into(),
        key_close: ">\n".into(),
        param_close: "\n</parameter>\n".into(),
        call_close: "</function>\n</tool_call>".into(),
        before_calls: "\n\n".into(),
        between_calls: "\n".into(),
    }
}

/// What a model that thinks about a call says before it.
pub const THINKS: &str = "The user asks for the weather.\n</think>\n\n";
/// A call of `get_weather` as Qwen3.8 writes it.
pub const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nKöln\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
/// A call of `now`, which takes no arguments.
pub const NOW: &str = "<tool_call>\n<function=now>\n</function>\n</tool_call>";

// --- requests ---------------------------------------------------------------

/// `base` with the members of `extra` set over it, as a body.
pub fn merged(mut base: serde_json::Value, extra: serde_json::Value) -> String {
    base.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    base.to_string()
}

/// A chat request to `mock` that says hi, with `extra` set over it.
pub fn chat_body(extra: serde_json::Value) -> String {
    merged(json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]}), extra)
}

/// A completion request to `mock`, with `extra` set over it.
pub fn completion_body(extra: serde_json::Value) -> String {
    merged(json!({"model": "mock", "prompt": "Once upon"}), extra)
}

/// An engine that answers `ok` to every request and reports the count of
/// its prompt as [`tokens`] counts it, as [`MockEngine::qwen`] counts it
/// too.
pub fn counting() -> Arc<MockEngine> {
    let ok = vec![text("ok"), Step::DoneCounted { finish: FinishReason::Stop, completion: 1 }];
    MockEngine::script(ok).qwen()
}

/// An engine that ends every request at once, for tests that look at what
/// it received.
pub fn instant() -> Arc<MockEngine> {
    MockEngine::script(vec![done(FinishReason::Stop, 1, 0)])
}

/// The request the engine got for `body` on `path`, which is answered with 200.
pub async fn submitted(
    engine: &MockEngine,
    app: &Router,
    path: &str,
    body: &str,
) -> GenerateRequest {
    let (status, value) = json(post(app, path, body).await).await;
    assert_eq!(status, StatusCode::OK, "{path}: {value}");
    engine.submitted().pop().expect("the request reached the engine")
}

/// The request a fresh [`instant`] engine got for `body` on `path`.
pub async fn received(path: &str, body: &str) -> GenerateRequest {
    let engine = instant();
    submitted(&engine, &app(&engine), path, body).await
}

// --- errors -----------------------------------------------------------------

/// An error of the engine and what every route of the OpenAI format makes of it.
pub struct ErrorCase {
    pub error: EngineError,
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: &'static str,
    /// Words the message must contain.
    pub mentions: Vec<&'static str>,
}

/// Every kind of error the engine can return; `invalid` is the text of its
/// invalid request.
pub fn error_cases(invalid: &'static str) -> Vec<ErrorCase> {
    vec![
        ErrorCase {
            error: EngineError::ModelNotFound("nope".into()),
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            code: "model_not_found",
            mentions: vec!["'nope'", "mock, mock-2"],
        },
        ErrorCase {
            error: EngineError::InvalidRequest(invalid.into()),
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: "invalid_request",
            mentions: vec![invalid],
        },
        ErrorCase {
            error: EngineError::ContextOverflow { prompt_tokens: 9000, limit: 4096 },
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: "context_length_exceeded",
            mentions: vec!["9000", "4096"],
        },
        ErrorCase {
            error: EngineError::Overloaded,
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            code: "overloaded",
            mentions: vec!["capacity"],
        },
        ErrorCase {
            error: EngineError::Unsupported("no such architecture".into()),
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "server_error",
            code: "not_implemented",
            mentions: vec!["no such architecture"],
        },
        ErrorCase {
            error: EngineError::Internal("kernel exploded".into()),
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            code: "internal_error",
            mentions: vec!["kernel exploded"],
        },
    ]
}

/// Checks that `response` is the error `case` in the OpenAI format: its
/// status, its type and code, its words, and Retry-After on 503 only.
pub async fn assert_error(case: &ErrorCase, response: Response<Body>, how: &str) {
    let retry_after = response.headers().get(header::RETRY_AFTER).cloned();
    let (status, body) = json(response).await;
    assert_eq!(status, case.status, "{how}: {body}");
    let error = &body["error"];
    assert_eq!(error["type"], case.kind, "{how}");
    assert_eq!(error["code"], case.code, "{how}");
    let message = error["message"].as_str().unwrap();
    for word in &case.mentions {
        assert!(message.contains(word), "{how}: '{message}' should mention {word}");
    }
    assert!(error.get("param").is_some(), "{how}: the error format always has a param");
    assert_eq!(body.as_object().unwrap().len(), 1, "{how}: nothing but the error object");
    let expected_retry = (case.status == StatusCode::SERVICE_UNAVAILABLE).then_some("1");
    assert_eq!(retry_after.as_ref().and_then(|v| v.to_str().ok()), expected_retry, "{how}");
}

// --- driving the router -----------------------------------------------------

pub fn app(engine: &Arc<MockEngine>) -> Router {
    ares_server::router(engine.clone())
}

pub fn app_with(engine: &Arc<MockEngine>, config: Config) -> Router {
    ares_server::router_with(engine.clone(), config)
}

pub fn post_request(path: &str, body: &str) -> Request<Body> {
    Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

pub async fn post(app: &Router, path: &str, body: &str) -> Response<Body> {
    app.clone().oneshot(post_request(path, body)).await.unwrap()
}

pub async fn get(app: &Router, path: &str) -> Response<Body> {
    app.clone().oneshot(Request::get(path).body(Body::empty()).unwrap()).await.unwrap()
}

pub async fn body_text(response: Response<Body>) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).expect("the body is valid UTF-8")
}

/// Status and parsed body of a JSON response.
pub async fn json(response: Response<Body>) -> (StatusCode, serde_json::Value) {
    let status = response.status();
    let text = body_text(response).await;
    (status, serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON ({e}): {text}")))
}

/// The `data:` payloads of an SSE body, in order. Comment lines (keep-alives)
/// are skipped.
pub fn sse_payloads(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|event| !event.is_empty())
        .filter_map(|event| {
            let data: Vec<_> = event.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
            (!data.is_empty()).then(|| data.join("\n"))
        })
        .collect()
}

/// [`sse_payloads`] with the JSON chunks parsed; the final `[DONE]` is
/// returned separately as `done`.
pub struct Stream {
    pub chunks: Vec<serde_json::Value>,
    pub done: bool,
}

pub fn parse_sse(body: &str) -> Stream {
    let mut payloads = sse_payloads(body);
    let done = payloads.last().is_some_and(|p| p == "[DONE]");
    if done {
        payloads.pop();
    }
    let chunks = payloads
        .iter()
        .map(|p| serde_json::from_str(p).unwrap_or_else(|e| panic!("chunk is not JSON ({e}): {p}")))
        .collect();
    Stream { chunks, done }
}

/// A fresh directory of the build's for the files of a test: `name` must be
/// unique among the tests of every suite.
pub fn scratch(name: &str) -> PathBuf {
    fresh(PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name))
}

/// `dir`, emptied of what an earlier run left.
pub fn fresh(dir: PathBuf) -> PathBuf {
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Polls until `condition` holds; fails the test after a few seconds.
pub async fn eventually(what: &str, condition: impl Fn() -> bool) {
    let deadline = tokio::time::Instant::now() + WAIT_LIMIT;
    while !condition() {
        assert!(tokio::time::Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
