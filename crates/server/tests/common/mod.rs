//! Test support: a scripted engine and helpers to drive the router.
#![allow(dead_code)]

use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use ares_api::{
    DecisionOutcome, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, Event,
    EventSink, FinishReason, GenerateRequest, ModelCard, RequestId, Usage,
};
use ares_server::Config;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, Response, StatusCode, header};
use tower::ServiceExt;

pub mod process;

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

    fn wait(&self) {
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

    fn run(&self, id: RequestId, steps: Vec<Step>, sink: EventSink) {
        for step in steps {
            match step {
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
}

impl Engine for MockEngine {
    fn models(&self) -> Vec<ModelCard> {
        self.models.clone()
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        let id = request.id;
        self.shared.record(|book| book.submitted.push(request));
        match &self.plan {
            Plan::Reject(error) => Err(error.clone()),
            Plan::Script(steps) => {
                // Like a real engine, `submit` returns at once and a thread of
                // the engine's own pushes the events.
                let (shared, steps) = (self.shared.clone(), steps.clone());
                thread::Builder::new()
                    .name("mock-engine".into())
                    .spawn(move || shared.run(id, steps, events))
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

/// Polls until `condition` holds; fails the test after a few seconds.
pub async fn eventually(what: &str, condition: impl Fn() -> bool) {
    let deadline = tokio::time::Instant::now() + WAIT_LIMIT;
    while !condition() {
        assert!(tokio::time::Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
