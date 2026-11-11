//! The `ares` binary serving a real model from a GPU, over real sockets.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL` (a small `qwen35`
//! file, Qwen3.5-0.8B); `ARES_TEST_DEVICE` names the device (default 0) and
//! `ARES_TEST_PORT` the first of the nine ports on 127.0.0.1 the tests
//! listen on (default 18951). Every test starts its own server, so they can
//! run one after the other or together:
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test -p ares-server --test real -- --test-threads=1
//! ```
//!
//! The servers run with exact products, so that a text does not depend on
//! what else is in a step and texts can be compared.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_ares");

/// How long anything may take before a test fails instead of hanging.
const LIMIT: Duration = Duration::from_secs(60);

/// The model to serve, or `None` (and a note) when the tests are not enabled.
fn model() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    match std::env::var_os("ARES_TEST_QWEN35_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to a small qwen35 model file");
            None
        }
    }
}

fn port(offset: u16) -> u16 {
    let base = std::env::var("ARES_TEST_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(18951);
    base + offset
}

/// A running server process. Killed when dropped, so that a failing test
/// leaves nothing on the device.
struct Server {
    child: Child,
    lines: mpsc::Receiver<String>,
    log: Vec<String>,
    client: Client,
}

/// Where a server listens and what it serves: all a request needs, and
/// something threads can share.
struct Client {
    address: String,
    /// The name the model is served under.
    model: String,
}

impl std::ops::Deref for Server {
    type Target = Client;

    fn deref(&self) -> &Client {
        &self.client
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One answer of the server.
struct Reply {
    status: u16,
    body: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }

    /// The `data:` payloads of a streamed answer and whether `[DONE]` ended it.
    fn events(&self) -> (Vec<Value>, bool) {
        events(&self.body)
    }
}

fn events(body: &str) -> (Vec<Value>, bool) {
    let mut payloads: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    let done = payloads.last() == Some(&"[DONE]");
    if done {
        payloads.pop();
    }
    let chunks = payloads
        .iter()
        .map(|p| serde_json::from_str(p).unwrap_or_else(|e| panic!("not JSON ({e}): {p}")))
        .collect();
    (chunks, done)
}

/// The text of a streamed answer: its deltas put together, the reasoning of a
/// model that thinks (its own field) before the content.
fn streamed_text(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|chunk| {
            let choice = chunk["choices"].get(0)?;
            let delta = &choice["delta"];
            Some(
                [
                    delta["reasoning_content"].as_str(),
                    delta["content"].as_str(),
                    choice["text"].as_str(),
                ]
                .into_iter()
                .flatten()
                .collect::<String>(),
            )
        })
        .collect()
}

/// Removes the chunked transfer coding.
fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(line_end) = raw.windows(2).position(|w| w == b"\r\n") {
        let size = std::str::from_utf8(&raw[..line_end])
            .ok()
            .and_then(|s| usize::from_str_radix(s.trim(), 16).ok());
        let Some(size) = size else { break };
        let start = line_end + 2;
        if size == 0 || raw.len() < start + size {
            break;
        }
        out.extend_from_slice(&raw[start..start + size]);
        raw = &raw[(start + size + 2).min(raw.len())..];
    }
    out
}

fn parse(raw: &[u8]) -> Reply {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("a complete header");
    let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
    let status = head.split(' ').nth(1).and_then(|s| s.parse().ok()).expect("a status line");
    let body = &raw[split + 4..];
    let body =
        if head.contains("transfer-encoding: chunked") { dechunk(body) } else { body.to_vec() };
    Reply { status, body: String::from_utf8(body).expect("the body is valid UTF-8") }
}

impl Server {
    /// Starts `ares` with the model on one device and `flags`, and waits
    /// until it listens.
    fn start(model: &PathBuf, port: u16, flags: &[&str]) -> Self {
        let device = std::env::var("ARES_TEST_DEVICE").unwrap_or_else(|_| "0".to_owned());
        let mut child = Command::new(BIN)
            .arg("--model")
            .arg(model)
            .args(["--port", &port.to_string(), "--devices", &device])
            .args(["--precision", "exact", "--context", "8192"])
            .args(flags)
            .env_remove("RUST_LOG")
            .stdin(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = mpsc::channel();
        // The reader keeps draining after the start, so that the child can
        // never block on a full pipe.
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let client = Client { address: format!("127.0.0.1:{port}"), model: String::new() };
        let mut server = Self { child, lines, log: Vec::new(), client };
        server.wait_for_log("listening");
        let models = server.get("/v1/models").json();
        server.client.model =
            models["data"][0]["id"].as_str().expect("a model is served").to_owned();
        server
    }

    /// Reads log lines until one contains `needle`.
    fn wait_for_log(&mut self, needle: &str) -> String {
        loop {
            let line = self
                .lines
                .recv_timeout(LIMIT)
                .unwrap_or_else(|e| panic!("no log line with {needle:?} ({e}): {:#?}", self.log));
            self.log.push(line.clone());
            if line.contains(needle) {
                return line;
            }
        }
    }

    /// Everything the server has logged so far.
    fn logged(&mut self) -> &[String] {
        while let Ok(line) = self.lines.try_recv() {
            self.log.push(line);
        }
        &self.log
    }

    fn signal(&self, name: &str) {
        let status = Command::new("kill")
            .args([&format!("-{name}"), &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
    }

    fn wait_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + LIMIT;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "the process did not exit: {:#?}", self.logged());
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Stops the server with one signal and checks that it left in order.
    fn stop(mut self) {
        self.signal("TERM");
        let status = self.wait_exit();
        let log = self.logged().join("\n");
        assert!(status.success(), "a clean stop: {status:?}\n{log}");
        assert!(log.contains("threads joined, devices released"), "{log}");
    }
}

impl Client {
    fn connect(&self) -> std::io::Result<TcpStream> {
        let stream = TcpStream::connect(&self.address)?;
        stream.set_read_timeout(Some(LIMIT))?;
        Ok(stream)
    }

    /// Sends a request and leaves the answer unread.
    fn send(&self, method: &str, path: &str, body: &str) -> TcpStream {
        let mut stream = self.connect().expect("the server accepts connections");
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        stream
    }

    fn request(&self, method: &str, path: &str, body: &str) -> Reply {
        let mut raw = Vec::new();
        self.send(method, path, body).read_to_end(&mut raw).unwrap();
        parse(&raw)
    }

    fn get(&self, path: &str) -> Reply {
        self.request("GET", path, "")
    }

    fn post(&self, path: &str, body: &Value) -> Reply {
        self.request("POST", path, &body.to_string())
    }

    /// A chat request body for this server's model: `extra` on top of one
    /// user message.
    fn chat(&self, question: &str, extra: Value) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
        });
        for (key, value) in extra.as_object().expect("an object") {
            body[key] = value.clone();
        }
        body
    }

    /// One value of `GET /metrics`.
    fn metric(&self, name: &str) -> u64 {
        let reply = self.get("/metrics");
        assert_eq!(reply.status, 200, "{}", reply.body);
        reply
            .body
            .lines()
            .find_map(|line| line.strip_prefix(&format!("ares_{name} ")))
            .unwrap_or_else(|| panic!("no metric {name}: {}", reply.body))
            .parse()
            .unwrap()
    }
}

/// Polls until `condition` holds.
fn eventually(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + LIMIT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(20));
    }
}

/// Reads from `stream` until `count` events of a streamed answer have come.
fn read_events(stream: &mut TcpStream, count: usize) -> Vec<u8> {
    let mut seen = Vec::new();
    let mut buffer = [0u8; 4096];
    while String::from_utf8_lossy(&seen).matches("data: ").count() < count {
        let n = stream.read(&mut buffer).unwrap();
        assert!(n > 0, "the stream ended early: {}", String::from_utf8_lossy(&seen));
        seen.extend_from_slice(&buffer[..n]);
    }
    seen
}

const GERMAN: &str = "Erkl\u{e4}re in drei S\u{e4}tzen, warum der Himmel blau ist.";
const ENGLISH: &str = "Explain in three sentences why the sky is blue.";

#[test]
fn a_chat_gives_the_same_text_streamed_and_plain_for_the_same_seed() {
    let Some(model) = model() else { return };
    let server = Server::start(&model, port(0), &["--chat-var", "enable_thinking=false"]);

    assert_eq!(server.get("/health").body, r#"{"status":"ok"}"#);
    let models = server.get("/v1/models").json();
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(models["data"][0]["context_length"], 8192);
    assert!(models["data"][0]["created"].as_u64().unwrap() > 1_600_000_000, "{models}");

    for question in [GERMAN, ENGLISH] {
        let sampled = json!({
            "max_tokens": 48, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.02,
            "seed": 20_261_110,
        });
        let plain = server.post("/v1/chat/completions", &server.chat(question, sampled.clone()));
        assert_eq!(plain.status, 200, "{}", plain.body);
        let plain = plain.json();
        let text = plain["choices"][0]["message"]["content"].as_str().unwrap().to_owned();
        assert!(text.chars().filter(|c| c.is_alphabetic()).count() > 20, "an answer: {text:?}");
        assert_eq!(plain["choices"][0]["finish_reason"], "length");
        let usage = &plain["usage"];
        assert_eq!(usage["completion_tokens"], 48);
        let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap();
        assert!((10..80).contains(&prompt_tokens), "the question through the template: {usage}");
        assert_eq!(usage["total_tokens"], prompt_tokens + 48);
        // What the engine computed before may start the prompt: the start of
        // the template, at least, is the same for every question.
        let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
        assert!(cached < prompt_tokens, "the last token is always computed: {usage}");
        let timings = &plain["timings"];
        assert_eq!(timings["prompt_n"], prompt_tokens - cached, "the cached part is not computed");
        assert_eq!(timings["predicted_n"], 48);
        assert!(timings["prompt_ms"].as_f64().unwrap() > 0.0, "{timings}");
        assert!(timings["predicted_per_second"].as_f64().unwrap() > 1.0, "{timings}");
        assert!(timings["draft_n"].as_u64().is_some(), "it ran alone: {timings}");
        assert!(timings["draft_n_accepted"].as_u64() <= timings["draft_n"].as_u64());

        let mut streaming = sampled.clone();
        streaming["stream"] = json!(true);
        streaming["stream_options"] = json!({"include_usage": true});
        let reply = server.post("/v1/chat/completions", &server.chat(question, streaming));
        assert_eq!(reply.status, 200, "{}", reply.body);
        let (chunks, done) = reply.events();
        assert!(done, "the stream ends with [DONE]");
        assert_eq!(streamed_text(&chunks), text, "streamed and plain, same seed");
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let finish = &chunks[chunks.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        assert_eq!(finish["timings"]["predicted_n"], 48);
        // The same prompt a second time comes from the cache, all but its
        // last token; the rest of the usage is the same.
        let mut streamed_usage = chunks[chunks.len() - 1]["usage"].clone();
        let streamed_cached = streamed_usage["prompt_tokens_details"]["cached_tokens"].as_u64();
        assert_eq!(streamed_cached, Some(prompt_tokens - 1), "{streamed_usage}");
        streamed_usage["prompt_tokens_details"] = usage["prompt_tokens_details"].clone();
        assert_eq!(streamed_usage, *usage, "the usage chunk");

        // Another seed is another text; the most probable token needs none.
        let mut other = sampled.clone();
        other["seed"] = json!(4711);
        let other = server.post("/v1/chat/completions", &server.chat(question, other)).json();
        assert_ne!(other["choices"][0]["message"]["content"], text, "the seed is used");
        let greedy = json!({"max_tokens": 24, "temperature": 0});
        let ask = || {
            let reply = server.post("/v1/chat/completions", &server.chat(question, greedy.clone()));
            reply.json()["choices"][0]["message"]["content"].as_str().unwrap().to_owned()
        };
        assert_eq!(ask(), ask(), "the most probable token, twice");
    }
    server.stop();
}

#[test]
fn completions_stop_strings_and_token_limits_end_a_text_where_they_should() {
    let Some(model) = model() else { return };
    let server = Server::start(&model, port(1), &[]);
    let complete = |extra: Value| {
        let mut body = json!({
            "model": server.model,
            "prompt": "The first ten prime numbers are 2, 3, 5,",
            "temperature": 0,
        });
        for (key, value) in extra.as_object().unwrap() {
            body[key] = value.clone();
        }
        let reply = server.post("/v1/completions", &body);
        assert_eq!(reply.status, 200, "{}", reply.body);
        reply.json()
    };

    let whole = complete(json!({"max_tokens": 32}));
    assert_eq!(whole["object"], "text_completion");
    let text = whole["choices"][0]["text"].as_str().unwrap();
    assert_eq!(whole["usage"]["completion_tokens"], 32);
    assert_eq!(whole["choices"][0]["finish_reason"], "length");
    assert!(text.contains('7'), "the model goes on counting: {text:?}");

    // Fewer tokens are the start of the same text.
    let short = complete(json!({"max_tokens": 5}));
    let start = short["choices"][0]["text"].as_str().unwrap();
    assert_eq!(short["usage"]["completion_tokens"], 5);
    assert!(text.starts_with(start) && start.len() < text.len(), "{start:?} of {text:?}");

    // A stop string from the middle of the text ends it right before.
    let middle = text.len() / 2;
    let from = (middle..text.len()).find(|i| text.is_char_boundary(*i)).unwrap();
    let stop: String = text[from..].chars().take(3).collect();
    let cut = text.find(&stop).unwrap();
    let stopped = complete(json!({"max_tokens": 32, "stop": [stop.as_str(), "\u{1}never"]}));
    assert_eq!(stopped["choices"][0]["text"], text[..cut], "stopped before {stop:?}");
    assert_eq!(stopped["choices"][0]["finish_reason"], "stop");
    assert!(stopped["usage"]["completion_tokens"].as_u64().unwrap() <= 32);

    // The same through a stream: the held-back text never shows.
    let reply = server.post(
        "/v1/completions",
        &json!({
            "model": server.model, "prompt": "The first ten prime numbers are 2, 3, 5,",
            "temperature": 0, "max_tokens": 32, "stop": stop, "stream": true,
        }),
    );
    let (chunks, done) = reply.events();
    assert!(done);
    assert_eq!(streamed_text(&chunks), text[..cut]);
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "stop");
    server.stop();
}

#[test]
fn a_client_that_leaves_frees_its_sequence() {
    let Some(model) = model() else { return };
    let server = Server::start(&model, port(2), &[]);
    let pages = server.metric("pages");
    // A page is free, or the prefix cache holds what was computed on it.
    let unused = || server.metric("pages_free") + server.metric("cached_pages");
    assert_eq!(unused(), pages);
    assert_eq!(server.metric("sequences"), 0);

    // No token limit: the request would run until the context is full.
    let body = server.chat("Write a very long story about a lighthouse.", json!({"stream": true}));
    let mut stream = server.send("POST", "/v1/chat/completions", &body.to_string());
    read_events(&mut stream, 20);
    assert_eq!(server.metric("sequences"), 1);
    assert!(server.metric("pages_free") < pages, "the sequence holds pages");
    drop(stream);

    eventually("the sequence to leave the engine", || server.metric("sequences") == 0);
    assert_eq!(unused(), pages, "all pages are back, free or cached");
    assert!(server.metric("cached_pages") >= 1, "what it computed stays in the cache");
    assert_eq!(server.metric("requests_cancelled_total"), 1);
    assert_eq!(server.metric("requests_finished_total"), 0);

    // The same for a client that leaves a request that is not streamed.
    let plain = server.chat("Write a very long story about a harbour.", json!({}));
    let waiting = server.send("POST", "/v1/chat/completions", &plain.to_string());
    eventually("the request to run", || server.metric("sequences") == 1);
    drop(waiting);
    eventually("the sequence to leave the engine", || server.metric("sequences") == 0);
    assert_eq!(unused(), pages);
    assert_eq!(server.metric("requests_cancelled_total"), 2);

    // The engine is as good as new for the next request.
    let reply = server.post(
        "/v1/chat/completions",
        &server.chat(ENGLISH, json!({"max_tokens": 16, "temperature": 0})),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(reply.json()["usage"]["completion_tokens"], 16);
    server.stop();
}

#[test]
fn two_requests_run_at_once_and_each_gets_its_own_text() {
    let Some(model) = model() else { return };
    let process = Server::start(&model, port(3), &["--parallel", "2"]);
    let server: &Client = &process;
    let ask = |question: &str, seed: u64| {
        let extra = json!({"max_tokens": 160, "temperature": 0.7, "seed": seed, "stream": true});
        let reply = server.post("/v1/chat/completions", &server.chat(question, extra));
        assert_eq!(reply.status, 200, "{}", reply.body);
        let (chunks, done) = reply.events();
        assert!(done);
        let finish = chunks.last().unwrap()["choices"][0]["finish_reason"].clone();
        (streamed_text(&chunks), finish, chunks.last().unwrap()["timings"].clone())
    };

    // Alone, one after the other.
    let (german, _, alone) = ask(GERMAN, 1);
    let (english, _, _) = ask(ENGLISH, 2);
    assert!(alone["draft_n"].as_u64().is_some(), "{alone}");

    // Together.
    let most = std::sync::atomic::AtomicU64::new(0);
    let (german_together, english_together) = thread::scope(|scope| {
        let first = scope.spawn(|| ask(GERMAN, 1));
        let second = scope.spawn(|| ask(ENGLISH, 2));
        while !(first.is_finished() && second.is_finished()) {
            most.fetch_max(server.metric("sequences"), std::sync::atomic::Ordering::Relaxed);
        }
        (first.join().unwrap(), second.join().unwrap())
    });
    assert_eq!(most.into_inner(), 2, "both sequences were in the engine at the same time");
    assert_eq!(german_together.0, german, "the same text next to another request");
    assert_eq!(english_together.0, english);
    assert_eq!(german_together.1, "length");
    for timings in [&german_together.2, &english_together.2] {
        assert_eq!(timings["predicted_n"], 160);
        assert!(timings.get("draft_n").is_none(), "drafts are counted for both: {timings}");
    }
    assert_eq!(server.metric("requests_finished_total"), 4);
    assert_eq!(server.metric("sequences"), 0);
    assert_eq!(server.metric("pages_free") + server.metric("cached_pages"), server.metric("pages"));
    process.stop();
}

#[test]
fn what_the_engine_cannot_do_is_refused_by_name() {
    let Some(model) = model() else { return };
    let server = Server::start(&model, port(4), &[]);
    let refused = |path: &str, body: Value, status: u16, words: &[&str]| {
        let reply = server.post(path, &body);
        let error = reply.json()["error"].clone();
        assert_eq!(reply.status, status, "{body}: {}", reply.body);
        let message = error["message"].as_str().unwrap().to_owned();
        for word in words {
            assert!(message.contains(word), "{body}: '{message}' should mention '{word}'");
        }
        error
    };
    let chat = |extra: Value| server.chat("Hi", extra);
    let chat_path = "/v1/chat/completions";

    // The sampler on the device has no penalties.
    for (name, value) in
        [("presence_penalty", 0.5), ("frequency_penalty", -0.5), ("repeat_penalty", 1.1)]
    {
        refused(chat_path, chat(json!({name: value})), 400, &[name, "not implemented"]);
    }
    // Nor has the server what these ask for.
    let error = refused(chat_path, chat(json!({"logit_bias": {"5": -100}})), 400, &["logit_bias"]);
    assert_eq!(error["param"], "logit_bias");
    refused(chat_path, chat(json!({"grammar": "root ::= \"yes\""})), 400, &["grammar"]);
    refused(chat_path, chat(json!({"n": 2})), 400, &["n must be 1"]);
    refused(
        chat_path,
        chat(json!({"response_format": {"type": "json_object"}})),
        400,
        &["response_format", "structured output"],
    );
    let tool = json!({"type": "function", "function": {"name": "f", "parameters": {}}});
    refused(chat_path, chat(json!({"tools": [tool]})), 501, &["tool"]);
    let image = json!({
        "model": server.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
    });
    refused(chat_path, image, 400, &["image_url", "only 'text' parts"]);
    let error = refused(chat_path, json!({"model": "gpt-4", "messages": []}), 400, &["messages"]);
    assert_eq!(error["param"], "messages");
    let mut other = chat(json!({}));
    other["model"] = json!("gpt-4");
    let error = refused(chat_path, other, 404, &["'gpt-4'", &server.model]);
    assert_eq!(error["code"], "model_not_found");
    // A prompt that does not fit is told so with both numbers.
    let long = "word ".repeat(9000);
    let error =
        refused("/v1/completions", json!({"model": server.model, "prompt": long}), 400, &["8192"]);
    assert_eq!(error["code"], "context_length_exceeded");
    // Decisions need more of the runtime than it has.
    let decision = json!({
        "model": server.model, "state": "s",
        "questions": {"q": {"type": "noul", "instructions": "x"}},
    });
    refused("/v1/systemone", decision, 501, &["decisions are not implemented"]);
    refused("/v1/embeddings", json!({"model": server.model, "input": "x"}), 404, &["no endpoint"]);

    // None of that reached the devices, and a neutral value is no refusal.
    assert_eq!(server.metric("steps_total"), 0);
    let quiet = chat(json!({
        "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1.0, "logit_bias": {},
        "n": 1, "max_tokens": 4, "user": "someone",
    }));
    let reply = server.post(chat_path, &quiet);
    assert_eq!(reply.status, 200, "{}", reply.body);
    server.stop();
}

#[test]
fn a_signal_lets_a_running_request_finish_and_then_stops_in_order() {
    let Some(model) = model() else { return };
    let mut server = Server::start(&model, port(5), &[]);
    let body = server.chat(ENGLISH, json!({"stream": true, "max_tokens": 400, "temperature": 0}));
    let mut stream = server.send("POST", "/v1/chat/completions", &body.to_string());
    let mut raw = read_events(&mut stream, 5);

    server.signal("TERM");
    server.wait_for_log("no new connections");
    // The listener is closed; the request that runs is not disturbed.
    eventually("the listener to close", || server.connect().is_err());
    stream.read_to_end(&mut raw).unwrap();
    let (chunks, done) = parse(&raw).events();
    assert!(done, "the request ran to its end");
    let finish = chunks.last().unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(finish["timings"]["predicted_n"], 400);

    let status = server.wait_exit();
    let log = server.logged().join("\n");
    assert!(status.success(), "{status:?}\n{log}");
    assert!(log.contains("released"), "the device is given back: {log}");
    assert!(log.contains("threads joined, devices released"), "{log}");
    assert!(log.contains("running=false"), "the engine's threads have ended: {log}");
    assert!(log.trim_end().ends_with("stopped"), "{log}");
}

#[test]
fn a_second_signal_ends_a_running_request_and_still_stops_in_order() {
    let Some(model) = model() else { return };
    let mut server = Server::start(&model, port(6), &[]);
    // No token limit: this request does not end by itself in time.
    let body = server.chat("Write a very long story about a river.", json!({"stream": true}));
    let mut stream = server.send("POST", "/v1/chat/completions", &body.to_string());
    let mut raw = read_events(&mut stream, 5);

    server.signal("INT");
    server.wait_for_log("no new connections");
    thread::sleep(Duration::from_millis(300));
    assert!(server.child.try_wait().unwrap().is_none(), "the first signal waits for the request");
    server.signal("INT");

    // The client is told, and the stream does not look complete.
    stream.read_to_end(&mut raw).unwrap();
    let (chunks, done) = parse(&raw).events();
    assert!(!done, "no [DONE] after a cut");
    assert_eq!(chunks.last().unwrap()["error"]["code"], "cancelled");
    assert!(streamed_text(&chunks).len() > 10, "the text up to the cut arrived");

    let status = server.wait_exit();
    let log = server.logged().join("\n");
    assert!(status.success(), "a stop that was asked for twice is still a clean one: {log}");
    assert!(log.contains("the server is stopping: ending the request"), "{log}");
    assert!(log.contains("threads joined, devices released"), "{log}");
    assert!(log.contains("running=false"), "{log}");
}

/// The sentences of a text that a model copies without a doubt: no two are
/// alike.
fn passage(sentences: usize) -> String {
    const THINGS: [&str; 8] = [
        "lanterns",
        "marshes",
        "apples",
        "weather vanes",
        "letters",
        "paper boats",
        "keys",
        "timetables",
    ];
    const PEOPLE: [&str; 8] = [
        "The harbour master",
        "A courier",
        "Our engineer",
        "The baker",
        "Her grandmother",
        "The clockmaker",
        "A botanist",
        "The watchman",
    ];
    (0..sentences)
        .map(|i| {
            format!(
                "{} counted {} {} on day {}.",
                PEOPLE[i * 5 % 8],
                11 + i * 7 % 90,
                THINGS[i * 3 % 8],
                i + 1
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Two turns of a conversation on `server`: the first message to copy a text,
/// then its answer and a second message. Returns the usage, the timings and
/// the answer of each turn.
fn two_turns(server: &Client) -> [(Value, Value, String); 2] {
    let first = format!("{}\n\nCopy the text above word for word.", passage(60));
    let ask = |messages: Value| {
        let body = json!({
            "model": server.model, "messages": messages, "max_tokens": 48, "temperature": 0,
        });
        let reply = server.post("/v1/chat/completions", &body);
        assert_eq!(reply.status, 200, "{}", reply.body);
        let reply = reply.json();
        let text = reply["choices"][0]["message"]["content"].as_str().unwrap().to_owned();
        (reply["usage"].clone(), reply["timings"].clone(), text)
    };
    let one = ask(json!([{"role": "user", "content": first}]));
    let two = ask(json!([
        {"role": "user", "content": first},
        {"role": "assistant", "content": one.2},
        {"role": "user", "content": "Now copy the first sentence again."},
    ]));
    [one, two]
}

#[test]
fn a_second_turn_starts_from_what_the_first_turn_computed_and_says_so() {
    let Some(model) = model() else { return };
    let mut server = Server::start(&model, port(7), &["--chat-var", "enable_thinking=false"]);
    let [(usage, timings, answer), (usage2, timings2, answer2)] = two_turns(&server);

    // The first turn computes its whole prompt.
    let prompt = usage["prompt_tokens"].as_u64().unwrap();
    assert!(prompt > 5 * 128, "five pages and a bit: {usage}");
    let first_cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(first_cached < 16, "nothing but the start of the template: {usage}");
    assert_eq!(timings["prompt_n"], prompt - first_cached);

    // The second turn reads the first one's pages and goes on from its
    // checkpoint: the whole pages of the first prompt at least.
    let prompt2 = usage2["prompt_tokens"].as_u64().unwrap();
    let cached = usage2["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(prompt2 > prompt, "the second prompt holds the first, the answer and a question");
    assert!(cached >= prompt / 128 * 128, "{usage2} after a prompt of {prompt} tokens");
    assert_eq!(timings2["prompt_n"], prompt2 - cached, "the cached part is not computed");
    eprintln!(
        "RES second turn: {prompt2} prompt tokens, {cached} from the cache, {} computed",
        prompt2 - cached
    );
    assert!(answer.contains("harbour master"), "a copy of the text: {answer:?}");
    assert!(answer2.chars().count() > 10, "an answer: {answer2:?}");

    // The counters say the same.
    assert_eq!(server.metric("prefix_hits_total"), 1 + u64::from(first_cached > 0));
    assert!(server.metric("cached_cells_total") >= cached);
    assert!(server.metric("checkpoints_saved_total") >= 1);
    assert_eq!(server.metric("checkpoints_restored_total"), 1);
    assert!(server.metric("replayed_rows_total") < prompt2, "not everything is read again");
    let held = server.metric("checkpoint_ram_bytes");
    assert!(held > 0 && held < 1 << 30, "page-locked memory for the checkpoints: {held} bytes");
    assert!(server.logged().iter().any(|l| l.contains("plan: RAM")), "{:#?}", server.logged());
    server.stop();
}

#[test]
fn a_server_with_no_ram_to_spare_keeps_no_checkpoints_and_answers_as_one_with() {
    let Some(model) = model() else { return };
    let flags = ["--chat-var", "enable_thinking=false"];
    let with = Server::start(&model, port(7), &flags);
    let [_, with_second] = two_turns(&with);
    with.stop();

    // The 0.8B's checkpoint is 20 MB: 4 MB of RAM holds none, so a start from
    // cached pages replays every cached row.
    let mut few = Vec::from(flags);
    few.extend(["--ram", "4M"]);
    let mut without = Server::start(&model, port(8), &few);
    assert!(
        without.logged().iter().any(|l| l.contains("plan: RAM") && l.contains("as asked")),
        "{:#?}",
        without.logged()
    );
    let [_, without_second] = two_turns(&without);
    let cached = without_second.0["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert!(cached >= 5 * 128, "the pages are cached all the same: {}", without_second.0);
    assert_eq!(without.metric("checkpoints_saved_total"), 0);
    assert_eq!(without.metric("checkpoints_restored_total"), 0);
    assert_eq!(without.metric("checkpoint_ram_bytes"), 0);
    assert!(
        without.metric("replayed_rows_total") >= cached - 1,
        "all {cached} cached rows were replayed"
    );
    assert_eq!(without_second.2, with_second.2, "the second answer, with and without checkpoints");
    without.stop();
}
