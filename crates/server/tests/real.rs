//! The `ares` binary serving a real model from a GPU, over real sockets.
//!
//! Needs `ARES_TEST_CUDA=1` and `ARES_TEST_QWEN35_MODEL` (a small `qwen35`
//! file, Qwen3.5-0.8B); `ARES_TEST_DEVICE` names the device (default 0) and
//! `ARES_TEST_PORT` the first of the ten ports on 127.0.0.1 the tests
//! listen on (default 18951). Every test starts its own server, so they can
//! run one after the other or together. The tests of Qwen3.8-Flash-Next at the
//! end need `ARES_TEST_QWEN4EXP_MODEL` (the first part of its split file) and
//! two devices, and are slow: a server loads in about 15 seconds, and the
//! first answer comes from experts that are still being fetched.
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
    /// How long the server may take for anything before a test fails.
    limit: Duration,
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
    /// until it listens. The test device (`ARES_TEST_DEVICE`) serves it unless
    /// `flags` name `--devices`.
    fn start(model: &PathBuf, port: u16, flags: &[&str]) -> Self {
        let device = std::env::var("ARES_TEST_DEVICE").unwrap_or_else(|_| "0".to_owned());
        // The test device, exact products and a context of 8192 unless the test
        // says otherwise.
        let mut base: Vec<&str> = Vec::new();
        if !flags.contains(&"--devices") {
            base.extend(["--devices", device.as_str()]);
        }
        if !flags.contains(&"--precision") {
            base.extend(["--precision", "exact"]);
        }
        if !flags.contains(&"--context") {
            base.extend(["--context", "8192"]);
        }
        Self::start_with(model, port, &base, flags, LIMIT)
    }

    /// Starts `ares` with the model and exactly the flags given, and waits
    /// until it listens; nothing it does takes longer than `limit`.
    fn start_with(
        model: &PathBuf,
        port: u16,
        base: &[&str],
        flags: &[&str],
        limit: Duration,
    ) -> Self {
        let mut child = Command::new(BIN)
            .arg("--model")
            .arg(model)
            .args(["--port", &port.to_string()])
            .args(base)
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
        let client = Client { address: format!("127.0.0.1:{port}"), model: String::new(), limit };
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
                .recv_timeout(self.limit)
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
        let deadline = Instant::now() + self.limit;
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
        stream.set_read_timeout(Some(self.limit))?;
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

    // The sampler has its limits, and a request past them is told which.
    refused(chat_path, chat(json!({"repeat_last_n": 5000})), 400, &["repeat_last_n", "1024"]);
    refused(chat_path, chat(json!({"adaptive_target": 0.5})), 400, &["adaptive_target"]);
    refused(chat_path, chat(json!({"logit_bias": {"99999999": -100}})), 400, &["logit_bias"]);
    refused(chat_path, chat(json!({"samplers": ["top_k", "penalties"]})), 400, &["penalties"]);
    refused(chat_path, chat(json!({"top_logprobs": 2})), 400, &["top_logprobs"]);
    // Nor has the server what this asks for.
    refused(chat_path, chat(json!({"n": 17})), 400, &["n must be between 1 and 16"]);
    // A grammar is served, and one that is wrong is told by name.
    refused(
        chat_path,
        chat(json!({"grammar": "root ::= nothing"})),
        400,
        &["grammar", "'nothing'"],
    );
    refused(
        chat_path,
        chat(json!({"response_format": {"type": "xml"}})),
        400,
        &["response_format", "xml"],
    );
    // Tools are served, as functions the client runs. What a request asks
    // about them beyond that is told by name: a tool of another kind, two
    // tools of one name, a choice the tools do not cover, a second rule for
    // the reply next to them, the functions of the first chat API, and tools
    // where there is no chat template.
    let tool = json!({"type": "function", "function": {"name": "f", "parameters": {}}});
    let tools = |extra: Value| {
        let mut body = chat(extra);
        body["tools"] = json!([tool]);
        body
    };
    refused(
        chat_path,
        chat(json!({"tools": [{"type": "web_search"}]})),
        400,
        &["tools[0].type", "\"web_search\"", "other kinds of tools are not served"],
    );
    refused(chat_path, chat(json!({"tools": [tool, tool]})), 400, &["two tools named 'f'"]);
    refused(
        chat_path,
        chat(json!({"tool_choice": "required"})),
        400,
        &["tool_choice", "the request has no tools"],
    );
    let named = json!({"type": "function", "function": {"name": "g"}});
    refused(
        chat_path,
        tools(json!({"tool_choice": named})),
        400,
        &["tool_choice", "'g'", "not among the tools"],
    );
    for (rule, value) in [
        ("grammar", json!("root ::= \"a\"")),
        ("json_schema", json!({"type": "object"})),
        ("response_format", json!({"type": "json_object"})),
    ] {
        let mut body = tools(json!({}));
        body[rule] = value;
        let error = refused(chat_path, body, 400, &["tools cannot be combined with", rule]);
        assert_eq!(error["param"], "tools");
    }
    refused(
        chat_path,
        chat(json!({"functions": [tool["function"]]})),
        400,
        &["functions", "tools"],
    );
    refused(
        "/v1/completions",
        json!({"model": server.model, "prompt": "Hi", "tools": [tool]}),
        400,
        &["tools is not supported", "/v1/completions has no chat template"],
    );
    // A budget for the thinking is not built.
    refused(
        chat_path,
        chat(json!({"thinking_budget_tokens": 100})),
        400,
        &["thinking_budget_tokens", "a budget for thinking is not built"],
    );
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
        "n": 1, "max_tokens": 4, "user": "someone", "logprobs": false,
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

/// The server run with `flags` on the small model, which is refused before
/// any device is touched: its exit status and what it said.
fn refused(model: &PathBuf, flags: &[&str]) -> (ExitStatus, String) {
    let output = Command::new(BIN)
        .arg("--model")
        .arg(model)
        .args(flags)
        .env_remove("RUST_LOG")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    (output.status, String::from_utf8_lossy(&output.stderr).into_owned())
}

#[test]
fn the_choice_of_drafters_is_refused_by_name_before_a_device_is_touched() {
    let Some(model) = std::env::var_os("ARES_TEST_QWEN35_MODEL").map(PathBuf::from) else {
        eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to a small qwen35 model file");
        return;
    };
    // A block drafter from a file is accepted by the option and not built.
    let (status, said) = refused(&model, &["--drafters", "mtp,/models/dflash2.gguf"]);
    assert!(!status.success() && said.contains("not built yet"), "{said}");
    // The small model has no draft block of its own.
    let (status, said) = refused(&model, &["--drafters", "mtp"]);
    assert!(!status.success() && said.contains("no draft block"), "{said}");
    // A window, or a vocabulary, for no drafter does nothing.
    let (status, said) = refused(&model, &["--drafters", "none", "--window", "8"]);
    assert!(!status.success() && said.contains("without a model drafter"), "{said}");
    let (status, said) = refused(&model, &["--drafters", "none", "--draft-vocab", "v.txt"]);
    assert!(!status.success() && said.contains("no drafter of that kind"), "{said}");
}

// --- drafts -------------------------------------------------------------------
//
// These need a model with a draft block: `ARES_TEST_DRAFT_MODEL`, a
// quantization of the 27B that fits one card, such as
// `Qwen3.8-27B-UD-IQ2_XXS-with-draft-block.gguf` (served with the fast
// kernels: it has no exact prefill kernels), and for the test of exact
// products `ARES_TEST_QWEN35_27B`, the Q4_K_S file, on two cards.

fn draft_model() -> Option<PathBuf> {
    std::env::var_os("ARES_TEST_CUDA")?;
    match std::env::var_os("ARES_TEST_DRAFT_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_DRAFT_MODEL to a model file with a draft block");
            None
        }
    }
}

/// The 27B with its draft block for the tests that need exact products and so
/// a type of weights with a prefill kernel for them: `ARES_TEST_QWEN35_27B`
/// (Qwen3.8-27B-UD-Q4_K_S.gguf), served on both cards.
fn two_card_model() -> Option<PathBuf> {
    std::env::var_os("ARES_TEST_CUDA")?;
    match std::env::var_os("ARES_TEST_QWEN35_27B") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_27B to the 27B Q4_K_S file (two GPUs)");
            None
        }
    }
}

/// The reply to `question`, greedy, with drafts or without (`"draft": false`):
/// the hash of its token ids as the last chunk of the stream and the final
/// message of a plain request give it, and the drafts verified for it.
fn reply_tokens(server: &Client, question: &str, drafts: bool) -> (String, String, Option<u64>) {
    let mut extra = json!({"max_tokens": 160, "temperature": 0});
    if !drafts {
        extra["draft"] = json!(false);
    }
    let plain = server.post("/v1/chat/completions", &server.chat(question, extra.clone()));
    assert_eq!(plain.status, 200, "{}", plain.body);
    let body = plain.json();
    let hash = body["timings"]["token_hash"].as_str().expect("a token hash").to_owned();
    let message = &body["choices"][0]["message"];
    let text = [message["reasoning_content"].as_str(), message["content"].as_str()]
        .into_iter()
        .flatten()
        .collect::<String>();
    // The stream carries the same hash in its last event with timings.
    extra["stream"] = json!(true);
    let streamed = server.post("/v1/chat/completions", &server.chat(question, extra));
    let (chunks, done) = streamed.events();
    assert!(done);
    let last = chunks.iter().rev().find(|c| c["timings"].is_object()).expect("timings at the end");
    assert_eq!(last["timings"]["token_hash"].as_str(), Some(hash.as_str()), "plain and stream");
    assert_eq!(streamed_text(&chunks), text);
    (hash, text, body["timings"]["draft_n"].as_u64())
}

#[test]
fn a_request_can_switch_drafts_off_and_the_reply_has_a_hash_of_its_tokens() {
    let Some(model) = draft_model() else { return };
    let server = Server::start(&model, port(9), &["--parallel", "1", "--precision", "fast"]);
    let (_, _, drafted) = reply_tokens(&server, ENGLISH, true);
    assert!(drafted.unwrap_or(0) > 0, "the draft block and the lookup draft: {drafted:?}");
    let (_, _, none) = reply_tokens(&server, ENGLISH, false);
    assert_eq!(none, Some(0), "no draft row for a request that asks for none");
    // The hash is a function of the tokens.
    let (a, ..) = reply_tokens(&server, GERMAN, false);
    let (b, ..) = reply_tokens(&server, GERMAN, false);
    assert_eq!(a, b);
    let (c, ..) = reply_tokens(&server, ENGLISH, false);
    assert_ne!(a, c);
    server.stop();
}

/// What the tests of drafts ask.
const QUESTIONS: [&str; 3] =
    [GERMAN, ENGLISH, "Write a Rust function that reverses a string, with a test."];

/// Asks every question with drafts and without, and says for each whether the
/// replies are the same tokens: `None`, or where they part (the characters
/// they share).
fn replies_with_and_without_drafts(server: &Client) -> Vec<(&'static str, Option<usize>)> {
    QUESTIONS
        .into_iter()
        .map(|question| {
            let (with, with_text, drafted) = reply_tokens(server, question, true);
            let (without, without_text, _) = reply_tokens(server, question, false);
            assert!(drafted.unwrap_or(0) > 0, "{question}: drafts were verified");
            let shared = with_text.chars().zip(without_text.chars()).take_while(|(a, b)| a == b);
            let parted = (with != without).then(|| shared.count());
            assert_eq!(parted.is_none(), with_text == without_text, "{question}: hash and text");
            (question, parted)
        })
        .collect()
}

/// With exact products a window of several rows and single rows round alike,
/// so the tokens with drafts are the tokens without them, and the hash of the
/// reply says so over HTTP.
#[test]
fn with_exact_products_the_reply_with_drafts_is_the_reply_without() {
    // Exact products need a prefill kernel for every type of the weights, which
    // the one-card file (IQ2_XXS) has none for: the 27B in Q4_K_S, on both cards.
    let Some(model) = two_card_model() else { return };
    let flags =
        ["--devices", "0,1", "--split", "tensor", "--parallel", "1", "--precision", "exact"];
    let server = Server::start(&model, port(9), &flags);
    for (question, parted) in replies_with_and_without_drafts(&server) {
        assert_eq!(parted, None, "{question}: the replies part after this many characters");
    }
    server.stop();
}

/// With the fast kernels (8-bit activations for a few rows) a window of
/// several rows rounds differently from single rows, and a reply with drafts
/// can part from the reply without them at a token whose two best candidates
/// are nearly equal. That is a defect of the kernels, which another piece of
/// work is to remove. Until then this says how often it happens (the test's
/// output has it) and holds on to what is true: the replies share their start,
/// and the drafts that were verified were verified. When the kernels agree the
/// replies are equal, exactly as in the test above: `differing` is then
/// always empty, the assertion on it becomes `assert!(differing.is_empty())`
/// and the two tests become one over both precisions.
#[test]
fn with_the_fast_kernels_a_reply_with_drafts_can_part_from_the_reply_without() {
    let Some(model) = draft_model() else { return };
    let server = Server::start(&model, port(9), &["--parallel", "1", "--precision", "fast"]);
    let differing: Vec<_> = replies_with_and_without_drafts(&server)
        .into_iter()
        .filter_map(|(question, parted)| Some((question, parted?)))
        .collect();
    eprintln!(
        "fast kernels: {} of {} replies with drafts part from the reply without: {differing:?}",
        differing.len(),
        QUESTIONS.len()
    );
    for (question, shared) in &differing {
        assert!(*shared >= SHARED_START, "{question}: the replies part after {shared} characters");
    }
    server.stop();
}

/// How much of its start (in characters) a reply with drafts shares with the
/// reply without them at the least, with the fast kernels. The replies part
/// at a token where two candidates are nearly equal, which was after 300 to
/// 430 characters where this was measured (the notes in ARCHITECTURE.md have
/// 150 to 650); a reply that parts within its first words is more likely a
/// window that was read wrong than rounding.
const SHARED_START: usize = 16;

/// A server started with no drafters does not speculate, whatever the text
/// repeats: the lookup in earlier text is no drafter on its own.
#[test]
fn a_server_with_no_drafters_never_drafts_and_one_with_the_draft_block_does() {
    let Some(model) = draft_model() else { return };
    let repeat = "Write the sentence 'The quick brown fox jumps over the lazy dog.' ten times.";
    let flags = ["--parallel", "1", "--precision", "fast"];
    let mut none = Vec::from(flags);
    none.extend(["--drafters", "none"]);
    let mut server = Server::start(&model, port(9), &none);
    assert!(
        server.logged().iter().any(|l| l.contains("no drafters") && l.contains("no speculation")),
        "{:#?}",
        server.logged()
    );
    for drafts in [true, false] {
        let (_, text, drafted) = reply_tokens(&server, repeat, drafts);
        assert!(text.contains("quick brown fox"), "{text}");
        assert_eq!(drafted, Some(0), "no draft rows with --drafters none (draft: {drafts})");
    }
    assert_eq!(server.metric("drafts_proposed_total"), 0, "the counters say the same");
    server.stop();
    let server = Server::start(&model, port(9), &flags);
    let (_, _, drafted) = reply_tokens(&server, repeat, true);
    assert!(drafted.unwrap_or(0) > 0, "the same request drafts with the model's draft block");
    server.stop();
}

/// A model with no draft block has no drafter, and the lookup does not run
/// without one.
#[test]
fn a_model_with_no_draft_block_is_served_with_no_speculation_at_all() {
    let Some(model) = model() else { return };
    let repeat = "Write the sentence 'The quick brown fox jumps over the lazy dog.' ten times.";
    let mut server = Server::start(&model, port(9), &["--parallel", "1"]);
    assert!(
        server
            .logged()
            .iter()
            .any(|l| l.contains("no model drafter") && l.contains("no speculation")),
        "{:#?}",
        server.logged()
    );
    let (_, text, drafted) = reply_tokens(&server, repeat, true);
    assert!(text.contains("quick brown fox"), "{text}");
    assert_eq!(drafted, Some(0));
    assert_eq!(server.metric("drafts_proposed_total"), 0);
    server.stop();
}

// --- Qwen3.8-Flash-Next ----------------------------------------------------

/// How long anything may take with the real Flash-Next: a server loads in
/// about 15 seconds, and an answer from experts that are still being fetched
/// can take a minute.
const SLOW: Duration = Duration::from_secs(900);

/// The first part of the split file of Qwen3.8-Flash-Next, or `None` (and a
/// note) when the tests of it are not enabled.
fn flash_model() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    match std::env::var_os("ARES_TEST_QWEN4EXP_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN4EXP_MODEL to the first part of Flash-Next");
            None
        }
    }
}

/// A server on two devices (`ARES_TEST_DEVICES`, default `0,1`) that spread
/// the model as `split` says.
fn flash_server(model: &PathBuf, port: u16, split: &str) -> Server {
    let devices = std::env::var("ARES_TEST_DEVICES").unwrap_or_else(|_| "0,1".to_owned());
    Server::start_with(
        model,
        port,
        &["--devices", &devices, "--split", split, "--chat-var", "enable_thinking=false"],
        &[],
        SLOW,
    )
}

/// The text of a chat answer, checked to be one.
fn answer(reply: &Reply) -> String {
    assert_eq!(reply.status, 200, "{}", reply.body);
    reply.json()["choices"][0]["message"]["content"].as_str().expect("a text").to_owned()
}

/// Whether `text` says one of `words`, in any case.
fn mentions(text: &str, words: &[&str]) -> bool {
    let text = text.to_lowercase();
    words.iter().any(|w| text.contains(w))
}

const HASH_TABLE: &str = "Explain in three sentences how a hash table works.";
const GERMAN_WORDS: &[&str] = &["licht", "streu", "rayleigh", "wellenl", "atmosph"];
const ENGLISH_WORDS: &[&str] = &["light", "scatter", "rayleigh", "wavelength", "atmosphere"];

/// Flash-Next through the binary on both devices, in both placements: it
/// answers in German and in English, two requests at once and a request that
/// is cancelled while it streams leave the server able to answer, and the
/// counters say how the experts were found.
#[test]
fn flash_next_answers_in_both_placements() {
    let Some(model) = flash_model() else { return };
    for (offset, split, placement) in
        [(0, "tensor", "both devices on every layer"), (1, "layer", "layers 0..")]
    {
        let mut server = flash_server(&model, port(offset), split);
        let log = server.logged().join("\n");
        assert!(log.contains(placement), "{split}: the plan says where the layers are: {log}");
        assert!(log.contains("plan: RAM"), "{split}: the plan says where the RAM goes: {log}");
        let client: &Client = &server;
        assert_eq!(client.get("/health").body, r#"{"status":"ok"}"#);

        let ask = |question: &str| {
            let body = client.chat(question, json!({"max_tokens": 80, "temperature": 0}));
            client.post("/v1/chat/completions", &body)
        };
        let german = ask(GERMAN);
        let text = answer(&german);
        eprintln!("RES {split}: German answer {text:?}");
        assert!(mentions(&text, GERMAN_WORDS), "{split}: a German answer about light: {text:?}");
        let timings = &german.json()["timings"];
        assert!(timings["predicted_n"].as_u64().unwrap_or(0) >= 10, "{split}: {timings}");
        let text = answer(&ask(ENGLISH));
        eprintln!("RES {split}: English answer {text:?}");
        assert!(mentions(&text, ENGLISH_WORDS), "{split}: an English answer about light: {text:?}");
        // And the answer to another question is the CPU reference's, as far
        // as its first sentence (the reference's greedy tokens are in
        // `/var/tmp/ares-qwen4exp-trace/english.txt`).
        let text = answer(&ask(HASH_TABLE));
        assert!(
            text.starts_with(
                "A hash table stores data by mapping keys to specific indices in an array using a \
                 hash function."
            ),
            "{split}: the reference's first sentence: {text:?}"
        );
        // A request that turns thinking on gets the reasoning apart from the
        // content, and neither has the tags that divide them.
        let body = client.chat(
            ENGLISH,
            json!({
                "max_tokens": 160,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": true},
            }),
        );
        let reply = client.post("/v1/chat/completions", &body);
        assert_eq!(reply.status, 200, "{}", reply.body);
        let message = &reply.json()["choices"][0]["message"];
        let reasoning = message["reasoning_content"].as_str().unwrap_or("");
        let content = message["content"].as_str().unwrap_or("");
        eprintln!("RES {split}: reasoning {reasoning:?}, content {content:?}");
        assert!(reasoning.len() > 40, "{split}: the thinking is the reasoning: {message}");
        assert!(content.len() > 40, "{split}: and the answer is the content: {message}");
        assert!(!reasoning.contains("think>") && !content.contains("think>"), "{split}: {message}");

        // Two requests at once share the steps.
        let together: Vec<String> = thread::scope(|scope| {
            let handles: Vec<_> = [GERMAN, ENGLISH]
                .into_iter()
                .map(|question| scope.spawn(move || answer(&ask(question))))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(mentions(&together[0], GERMAN_WORDS), "{split}: {:?}", together[0]);
        assert!(mentions(&together[1], ENGLISH_WORDS), "{split}: {:?}", together[1]);

        // A client that goes away in the middle of an answer frees its place.
        let body = client.chat(ENGLISH, json!({"max_tokens": 400, "stream": true}));
        let mut stream = client.send("POST", "/v1/chat/completions", &body.to_string());
        read_events(&mut stream, 3);
        drop(stream);
        eventually("the cancelled request to be counted", || {
            client.metric("requests_cancelled_total") == 1
        });
        let text = answer(&ask(GERMAN));
        assert!(mentions(&text, GERMAN_WORDS), "{split}: after a cancelled request: {text:?}");

        // How the experts were found.
        let (picks, hits) =
            (client.metric("expert_picks_total"), client.metric("expert_hits_total"));
        let misses = client.metric("expert_misses_ram_total")
            + client.metric("expert_misses_disk_total")
            + client.metric("expert_misses_on_their_way_total");
        eprintln!(
            "RES {split}: {picks} picks, {hits} in a slot, {misses} not; {} computed on the CPU, {} uploaded",
            client.metric("expert_cpu_total"),
            client.metric("expert_uploads_total"),
        );
        assert!(picks > 1000 && hits > 0 && hits <= picks, "{split}: {picks} picks, {hits} hits");
        assert_eq!(picks, hits + misses, "{split}: every pick was a hit or a miss");
        assert!(client.metric("steps_total") > 100);
        server.stop();
    }
}

/// The text of the first choice of an answer, and the ids of its tokens with
/// the probabilities (`logprob`) the answer reports.
fn first_choice(reply: &Reply) -> (String, Vec<Value>) {
    assert_eq!(reply.status, 200, "{}", reply.body);
    let value = reply.json();
    let choice = &value["choices"][0];
    let tokens = choice["logprobs"]["content"].as_array().cloned().unwrap_or_default();
    (choice["message"]["content"].as_str().unwrap_or_default().to_owned(), tokens)
}

#[test]
fn the_sampler_serves_what_a_request_sets_and_reports_the_probabilities() {
    let Some(model) = model() else { return };
    // The 0.8B thinks first unless told not to: the answers are the reply.
    let server = Server::start(&model, port(9), &["--chat-var", "enable_thinking=false"]);
    let chat_path = "/v1/chat/completions";
    let question = "Name three colours.";
    let greedy = |extra: Value| {
        let mut body = server.chat(question, json!({"temperature": 0, "max_tokens": 16}));
        for (key, value) in extra.as_object().unwrap() {
            body[key] = value.clone();
        }
        body
    };

    // The probabilities of a greedy answer: the token with the best
    // probability is the one that came out, and the alternatives are in order.
    let (plain, tokens) = first_choice(
        &server.post(chat_path, &greedy(json!({"logprobs": true, "top_logprobs": 5}))),
    );
    assert!(tokens.len() >= 8, "{tokens:?}");
    for token in &tokens {
        let top = token["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 5);
        assert_eq!(top[0]["id"], token["id"], "the greedy token is the most probable");
        assert_eq!(top[0]["logprob"], token["logprob"]);
        let lp: Vec<f64> = top.iter().map(|t| t["logprob"].as_f64().unwrap()).collect();
        assert!(lp.windows(2).all(|w| w[0] >= w[1]) && lp[0] <= 0.0);
        assert!(lp.iter().map(|l| l.exp()).sum::<f64>() <= 1.0 + 1e-4);
    }
    // After the chain a greedy row has one token left, with probability 1.
    let (_, after) = first_choice(
        &server.post(chat_path, &greedy(json!({"n_probs": 3, "post_sampling_probs": true}))),
    );
    assert!(
        after.iter().all(|t| t["prob"] == 1.0 && t["top_probs"].as_array().unwrap().len() == 1)
    );

    // A banned token is not picked, and the best of the rest is: the first
    // token of the answer becomes the second best of the first answer.
    let first = &tokens[0];
    let second = first["top_logprobs"][1]["id"].clone();
    let (banned, banned_tokens) =
        first_choice(&server.post(
            chat_path,
            &greedy(json!({"logit_bias": [[first["id"], false]], "logprobs": true})),
        ));
    assert_eq!(banned_tokens[0]["id"], second, "the answer {plain:?} then {banned:?}");
    // The probabilities are the model's own, so they do not move with the ban
    // (the prompt came from the cache the second time, which rounds a little
    // differently).
    let (a, b) = (
        banned_tokens[0]["logprob"].as_f64().unwrap(),
        first["top_logprobs"][1]["logprob"].as_f64().unwrap(),
    );
    assert!((a - b).abs() < 0.05, "{a} against {b}");

    // A penalty and a seed: the same request says the same twice, and
    // a strong penalty says something else than none.
    let penalised = |seed: u64| {
        let body = greedy(json!({
            "temperature": 0.7, "seed": seed, "repeat_penalty": 1.5, "frequency_penalty": 0.8,
            "max_tokens": 40,
        }));
        first_choice(&server.post(chat_path, &body)).0
    };
    assert_eq!(penalised(5), penalised(5));
    let free = first_choice(
        &server.post(chat_path, &greedy(json!({"temperature": 0.7, "seed": 5, "max_tokens": 40}))),
    )
    .0;
    assert_ne!(penalised(5), free);

    // Every other sampler answers.
    for extra in [
        json!({"mirostat": 2, "mirostat_tau": 3.0}),
        json!({"mirostat": 1}),
        json!({"typical_p": 0.9, "top_n_sigma": 3.0}),
        json!({"xtc_probability": 0.5, "xtc_threshold": 0.2}),
        json!({"dynatemp_range": 0.5, "dynatemp_exponent": 1.5}),
        json!({"dry_multiplier": 0.8, "dry_penalty_last_n": 64}),
        json!({"samplers": ["temperature", "min_p", "top_p"], "min_p": 0.05, "top_p": 0.9}),
        json!({"samplers": "ktpmt", "top_k": 40}),
        json!({"ignore_eos": true, "logit_bias": {"Hello": 1.5}}),
    ] {
        let mut body = greedy(extra.clone());
        body["temperature"] = json!(0.8);
        body["seed"] = json!(3);
        let (text, _) = first_choice(&server.post(chat_path, &body));
        assert!(!text.is_empty(), "{extra}");
    }

    // The completions of one prompt are requests of their own with their own
    // seeds, and the engine reads the prompt once.
    let steps = server.metric("prefix_hits_total");
    let mut body = greedy(json!({"temperature": 0.9, "seed": 40, "n": 3, "max_tokens": 24}));
    body["messages"][0]["content"] = json!("Tell me a very short story about a lighthouse keeper.");
    let reply = server.post(chat_path, &body);
    assert_eq!(reply.status, 200, "{}", reply.body);
    let choices = reply.json()["choices"].as_array().unwrap().clone();
    assert_eq!(choices.len(), 3);
    let texts: Vec<&str> =
        choices.iter().map(|c| c["message"]["content"].as_str().unwrap()).collect();
    assert!(texts[0] != texts[1] || texts[1] != texts[2], "{texts:?}");
    for (i, text) in texts.iter().enumerate() {
        let mut one = body.clone();
        one["n"] = json!(1);
        one["seed"] = json!(40 + i as u64);
        // The text the seed gives, up to the model's rounding: the choices ran
        // side by side, so a sampled text can part from the one it has alone
        // where two tokens are that close.
        let alone = first_choice(&server.post(chat_path, &one)).0;
        let same = text.chars().zip(alone.chars()).take_while(|(a, b)| a == b).count();
        eprintln!("RES n choice {i}: {same} of {} characters equal the request alone", text.len());
        assert!(same >= 10, "choice {i}: {text:?} against {alone:?}");
    }
    assert!(
        server.metric("prefix_hits_total") >= steps + 2,
        "the other completions read the cache"
    );
    server.stop();
}
