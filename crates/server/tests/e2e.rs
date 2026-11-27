//! The server on a real socket: framing, disconnects and shutdown as hyper
//! and the kernel deliver them, which the in-process tests cannot show.
//! Everything binds 127.0.0.1:0.

mod common;

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_api::{
    DecisionOutcome, DecisionRequest, DecisionResult, EngineError, FinishReason, Usage,
};
use ares_server::options::ModelOptions;
use ares_server::{Config, serve_models};
use common::socket::{Reply, find, request_bytes};
use common::*;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

struct Server {
    addr: SocketAddr,
    shutdown: oneshot::Sender<()>,
    task: JoinHandle<io::Result<()>>,
}

async fn start(engine: &Arc<MockEngine>, config: Config) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (shutdown, signal) = oneshot::channel();
    let models = models_of(engine.clone(), ModelOptions::default()).await;
    let stop = async {
        let _ = signal.await;
    };
    let task = tokio::spawn(serve_models(listener, models, config, stop, std::future::pending()));
    Server { addr, shutdown, task }
}

impl Server {
    /// Signals shutdown and waits for `serve_models` to return.
    async fn stop(self) -> io::Result<()> {
        self.shutdown.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), self.task)
            .await
            .expect("serve returned in time")
            .unwrap()
    }
}

/// One request on a fresh connection, read until the server closes it.
async fn call(addr: SocketAddr, method: &str, path: &str, body: &str) -> Reply {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(&request_bytes(method, path, body)).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut raw))
        .await
        .expect("the server closed the connection")
        .unwrap();
    Reply::parse(&raw)
}

/// Reads from `stream` until `needle` has arrived. Returns everything read.
async fn read_until(stream: &mut TcpStream, needle: &str) -> Vec<u8> {
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    let deadline = Instant::now() + Duration::from_secs(10);
    while find(&seen, needle.as_bytes()).is_none() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {needle:?}, got {:?}",
            String::from_utf8_lossy(&seen)
        );
        let n = tokio::time::timeout(Duration::from_secs(10), stream.read(&mut buf))
            .await
            .expect("the server sent more")
            .unwrap();
        assert!(n > 0, "the server closed the connection before sending {needle:?}");
        seen.extend_from_slice(&buf[..n]);
    }
    seen
}

#[tokio::test]
async fn serves_plain_and_streamed_requests_over_a_socket() {
    let pieces = ["Grüße ", "🦀 ", "日本語"];
    let mut steps: Vec<_> = pieces.iter().map(|p| text(p)).collect();
    steps.push(done(FinishReason::Stop, 4, 3));
    let engine = MockEngine::script(steps);
    let server = start(&engine, Config::default()).await;

    let health = call(server.addr, "GET", "/health", "").await;
    assert_eq!(health.status, 200);
    assert_eq!(health.json(), json!({"status": "ok"}));
    assert_eq!(call(server.addr, "GET", "/v1/models", "").await.json()["data"][0]["id"], "mock");

    let plain = call(server.addr, "POST", "/v1/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(plain.status, 200);
    assert_eq!(plain.json()["choices"][0]["message"]["content"], pieces.concat());

    let streamed =
        call(server.addr, "POST", "/v1/chat/completions", &chat_body(json!({"stream": true})))
            .await;
    assert_eq!(streamed.status, 200);
    assert!(streamed.is("text/event-stream"));
    let stream = parse_sse(&streamed.body);
    assert!(stream.done);
    let joined: String =
        stream.chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(joined, pieces.concat());

    let bad = call(server.addr, "POST", "/v1/chat/completions", "{").await;
    assert_eq!(bad.status, 400);
    assert_eq!(bad.json()["error"]["type"], "invalid_request_error");

    server.stop().await.unwrap();
}

#[tokio::test]
async fn a_connection_that_closes_during_a_plain_request_cancels_it() {
    let engine = MockEngine::script(vec![Step::UntilCancelled]);
    let server = start(&engine, Config::default()).await;

    let mut client = TcpStream::connect(server.addr).await.unwrap();
    client
        .write_all(&request_bytes("POST", "/v1/chat/completions", &chat_body(json!({}))))
        .await
        .unwrap();
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;
    assert!(engine.cancelled().is_empty());

    drop(client);

    eventually("hyper to notice the closed connection and the cancel to arrive", || {
        !engine.cancelled().is_empty()
    })
    .await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);
    server.stop().await.unwrap();
}

#[tokio::test]
async fn a_connection_that_closes_during_a_stream_cancels_it() {
    let engine = MockEngine::script(vec![text("first"), Step::UntilCancelled]);
    let server = start(&engine, Config::default()).await;

    let mut client = TcpStream::connect(server.addr).await.unwrap();
    client
        .write_all(&request_bytes(
            "POST",
            "/v1/chat/completions",
            &chat_body(json!({"stream": true})),
        ))
        .await
        .unwrap();
    read_until(&mut client, "\"first\"").await;
    assert!(engine.cancelled().is_empty());

    drop(client);

    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);
    server.stop().await.unwrap();
}

#[tokio::test]
async fn an_idle_server_stops_at_once_and_refuses_new_connections() {
    let engine = MockEngine::script(vec![]);
    let server = start(&engine, Config::default()).await;
    let addr = server.addr;
    assert_eq!(call(addr, "GET", "/health", "").await.status, 200);

    let started = Instant::now();
    server.stop().await.unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "nothing was running, so nothing to wait for"
    );
    assert!(TcpStream::connect(addr).await.is_err(), "the listener is closed");
}

#[tokio::test]
async fn shutdown_lets_a_running_stream_finish() {
    let gate = Gate::default();
    let engine = MockEngine::script(vec![
        text("one"),
        Step::Wait(gate.clone()),
        text("two"),
        done(FinishReason::Stop, 1, 2),
    ]);
    let server = start(&engine, Config::default()).await;

    let mut client = TcpStream::connect(server.addr).await.unwrap();
    client
        .write_all(&request_bytes(
            "POST",
            "/v1/chat/completions",
            &chat_body(json!({"stream": true})),
        ))
        .await
        .unwrap();
    let mut seen = read_until(&mut client, "\"one\"").await;

    server.shutdown.send(()).unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!server.task.is_finished(), "a running request keeps the server up");

    gate.open();
    let mut rest = Vec::new();
    client.read_to_end(&mut rest).await.unwrap();
    seen.extend(rest);
    let reply = Reply::parse(&seen);
    let stream = parse_sse(&reply.body);
    assert!(stream.done, "the request ran to its end");
    assert!(reply.body.contains("two"));

    tokio::time::timeout(Duration::from_secs(10), server.task).await.unwrap().unwrap().unwrap();
    assert!(engine.cancelled().is_empty());
}

#[tokio::test]
async fn shutdown_gives_up_on_a_stuck_request_after_the_grace_period() {
    let engine = MockEngine::script(vec![Step::UntilCancelled]);
    let config = Config { shutdown_grace: Duration::from_millis(300), ..Config::default() };
    let server = start(&engine, config).await;

    let mut client = TcpStream::connect(server.addr).await.unwrap();
    client
        .write_all(&request_bytes("POST", "/v1/chat/completions", &chat_body(json!({}))))
        .await
        .unwrap();
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;

    let started = Instant::now();
    server.stop().await.unwrap();
    let waited = started.elapsed();
    assert!(waited >= Duration::from_millis(300), "the grace period was respected: {waited:?}");
    assert!(waited < Duration::from_secs(5), "and then the server stopped: {waited:?}");

    // Let the engine thread finish instead of waiting out its own timeout.
    drop(client);
    eventually("the cancel", || !engine.cancelled().is_empty()).await;
}

#[tokio::test]
async fn a_second_signal_ends_running_requests_and_tells_their_clients() {
    let engine = MockEngine::script(vec![text("one"), Step::UntilCancelled]);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (drain, drained) = oneshot::channel::<()>();
    let (hurry, hurried) = oneshot::channel::<()>();
    // The grace period alone would keep the server up for half a minute.
    let task = tokio::spawn(serve_models(
        listener,
        models_of(engine.clone(), ModelOptions::default()).await,
        Config::default(),
        async {
            let _ = drained.await;
        },
        async {
            let _ = hurried.await;
        },
    ));

    let mut client = TcpStream::connect(addr).await.unwrap();
    client
        .write_all(&request_bytes(
            "POST",
            "/v1/chat/completions",
            &chat_body(json!({"stream": true})),
        ))
        .await
        .unwrap();
    let mut seen = read_until(&mut client, "\"one\"").await;

    drain.send(()).unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!task.is_finished(), "the first signal waits for the running request");
    assert!(engine.cancelled().is_empty());

    let started = Instant::now();
    hurry.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(10), task).await.unwrap().unwrap().unwrap();
    assert!(started.elapsed() < Duration::from_secs(3), "the second one does not");
    assert_eq!(engine.cancelled().len(), 1, "the request left the engine");

    // The client is told: an error object ends its stream, and no [DONE].
    let mut rest = Vec::new();
    client.read_to_end(&mut rest).await.unwrap();
    seen.extend(rest);
    let stream = parse_sse(&Reply::parse(&seen).body);
    assert!(!stream.done);
    assert_eq!(stream.chunks.last().unwrap()["error"]["code"], "cancelled");
}

#[tokio::test]
async fn an_error_status_line_reaches_a_real_client() {
    let engine =
        MockEngine::reject(EngineError::ContextOverflow { prompt_tokens: 9000, limit: 4096 });
    let server = start(&engine, Config::default()).await;
    let reply =
        call(server.addr, "POST", "/v1/chat/completions", &chat_body(json!({"stream": true})))
            .await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["code"], "context_length_exceeded");
    server.stop().await.unwrap();
}

fn decision(extra: Value) -> String {
    merged(
        json!({
            "model": "mock",
            "state": "Customer message: I was charged twice.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"billing": null, "shipping": null}
                },
                "angry": {"type": "noul", "instructions": "Is the customer angry?"}
            }
        }),
        extra,
    )
}

/// Every option of every question equally likely.
fn ties(request: &DecisionRequest) -> DecisionOutcome {
    let answers =
        request.questions.iter().map(|q| q.answer(&vec![0.0; q.n_options()]).unwrap()).collect();
    let usage = Usage {
        prompt_tokens: 12,
        completion_tokens: request.questions.len() as u32,
        cached_prompt_tokens: 0,
        ..Usage::default()
    };
    DecisionOutcome::Decided(DecisionResult { answers, usage, prefix_tokens: 4 })
}

#[tokio::test]
async fn a_decision_is_answered_on_its_own_connection_while_a_stream_is_open() {
    let gate = Gate::default();
    let engine = MockEngine::both(
        Plan::Script(vec![
            text("first"),
            Step::Wait(gate.clone()),
            text("second"),
            done(FinishReason::Stop, 1, 2),
        ]),
        DecisionPlan::Script(vec![deliver_with(ties)]),
    );
    let server = start(&engine, Config::default()).await;

    // The stream is open on one connection and stuck behind the gate.
    let mut client = TcpStream::connect(server.addr).await.unwrap();
    client
        .write_all(&request_bytes(
            "POST",
            "/v1/chat/completions",
            &chat_body(json!({"stream": true})),
        ))
        .await
        .unwrap();
    let mut seen = read_until(&mut client, "\"first\"").await;
    eventually("the engine to have sent the first piece", || engine.sends() == [true]).await;

    // Another connection asks a question and gets its answer meanwhile.
    let reply = call(server.addr, "POST", "/v1/systemone", &decision(json!({}))).await;
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert!(reply.is("application/json"));
    let answer = reply.json();
    assert_eq!(answer["model"], "mock");
    assert_eq!(answer["answers"]["route"]["choice"], "billing", "a tie goes to the first option");
    assert_eq!(answer["answers"]["angry"]["noul"], 0.5);
    assert_eq!(answer["usage"], json!({"input_tokens": 12, "output_tokens": 2}));
    assert_eq!(engine.sends(), [true], "the stream was still stuck when the decision was answered");

    // The stream then finishes with everything in it.
    gate.open();
    let mut rest = Vec::new();
    client.read_to_end(&mut rest).await.unwrap();
    seen.extend(rest);
    let reply = Reply::parse(&seen);
    let stream = parse_sse(&reply.body);
    assert!(stream.done, "the stream ran to its end");
    let joined: String =
        stream.chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(joined, "firstsecond");
    assert!(engine.cancelled().is_empty());
    server.stop().await.unwrap();
}

#[tokio::test]
async fn a_connection_that_closes_during_a_decision_cancels_it() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![DecisionStep::UntilCancelled]));
    let server = start(&engine, Config::default()).await;

    let mut client = TcpStream::connect(server.addr).await.unwrap();
    client.write_all(&request_bytes("POST", "/v1/systemone", &decision(json!({})))).await.unwrap();
    eventually("the decision to reach the engine", || engine.decided().len() == 1).await;
    assert!(engine.cancelled().is_empty());

    drop(client);

    eventually("hyper to notice the closed connection and the cancel to arrive", || {
        !engine.cancelled().is_empty()
    })
    .await;
    assert_eq!(engine.cancelled(), [engine.decided()[0].id]);
    server.stop().await.unwrap();
}

#[tokio::test]
async fn shutdown_lets_a_running_decision_finish() {
    let gate = Gate::default();
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![
        DecisionStep::Wait(gate.clone()),
        deliver_with(ties),
    ]));
    let server = start(&engine, Config::default()).await;
    let mut client = TcpStream::connect(server.addr).await.unwrap();
    client.write_all(&request_bytes("POST", "/v1/systemone", &decision(json!({})))).await.unwrap();
    eventually("the decision to reach the engine", || engine.decided().len() == 1).await;

    server.shutdown.send(()).unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!server.task.is_finished(), "a running decision keeps the server up");

    gate.open();
    let mut raw = Vec::new();
    client.read_to_end(&mut raw).await.unwrap();
    let reply = Reply::parse(&raw);
    assert_eq!(reply.status, 200, "the decision ran to its end: {}", reply.body);
    assert_eq!(reply.json()["answers"]["angry"]["noul"], 0.5);
    tokio::time::timeout(Duration::from_secs(10), server.task).await.unwrap().unwrap().unwrap();
    assert!(engine.cancelled().is_empty());
}
