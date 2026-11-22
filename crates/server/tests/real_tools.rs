//! Tool calls through the server, with the real Qwen3.5-0.8B on a GPU and
//! with the 27B on two: the template's tools reach the model, a required call
//! and a named function come back as valid calls, `auto` calls when the
//! question needs the tool and the reply is the same streamed, a history with
//! calls and results is answered, a call the model opens inside its thinking
//! is a call, and the reply with drafts is the reply without, also where the
//! 27B thinks before it calls and takes the drafts of its MTP block. Lines
//! that start with `RES` say what the models did.
//!
//! The server runs in this process over the real engine (requests go through
//! the router without a socket). Needs `ARES_TEST_CUDA=1` and
//! `ARES_TEST_QWEN35_MODEL` (the 0.8B); `ARES_TEST_DEVICE` names the device
//! (default 0). With `ARES_TEST_API_MODEL` the tests that are not about one
//! model run on that file instead, loaded as `ares --model` loads it
//! (Flash-Next, the 27B: see `api_model/mod.rs`). The tests on the 27B with
//! its drafts need `ARES_TEST_QWEN35_27B` (Qwen3.8-27B-UD-Q4_K_S.gguf) and
//! two GPUs.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     ARES_TEST_QWEN35_27B=Qwen3.8-27B-UD-Q4_K_S.gguf \
//!     cargo test --release -p ares-server --test real_tools -- --test-threads=1 --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use ares_api::Engine;
use ares_cuda::{Device, Precision, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, header};
use serde_json::{Value, json};
use tower::ServiceExt;

mod api_model;

const CHAT: &str = "/v1/chat/completions";

fn file(variable: &str, what: &str) -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    let path = std::env::var_os(variable).map(PathBuf::from);
    if path.is_none() {
        eprintln!("skipped: set {variable} to {what}");
    }
    path
}

fn small() -> Option<PathBuf> {
    file("ARES_TEST_QWEN35_MODEL", "the Qwen3.5-0.8B model file")
}

fn large() -> Option<PathBuf> {
    file("ARES_TEST_QWEN35_27B", "the 27B Q4_K_S file (two GPUs)")
}

/// The server over the engine.
struct Served {
    app: Router,
    engine: Arc<dyn Engine>,
    /// The runtime, when it was built here (not for a model loaded as the
    /// server loads it).
    runtime: Option<Arc<Runtime>>,
    path: PathBuf,
    /// Whether a request gives the same tokens every time.
    repeatable: bool,
    /// The model loaded as the server loads it, which this keeps loaded.
    _loaded: Option<api_model::ApiModel>,
}

impl Served {
    /// The model of `ARES_TEST_API_MODEL` if one is named, else the 0.8B;
    /// `None` (and a note) when neither is.
    fn any() -> Option<Self> {
        if let Some(path) = api_model::path() {
            let model = api_model::ApiModel::load(&path, None);
            let engine = model.engine();
            return Some(Self {
                app: ares_server::router(engine.clone()),
                engine,
                runtime: None,
                path,
                repeatable: model.repeatable,
                _loaded: Some(model),
            });
        }
        small().map(|path| Self::start(&path))
    }

    /// The 0.8B on one device.
    fn start(path: &PathBuf) -> Self {
        let options = Options {
            precision: Precision::Exact,
            max_rows: 128,
            logit_rows: 16,
            window_rows: 8,
            pipeline_rows: 32,
            ..Options::default()
        };
        let model =
            Qwen35::new(Gguf::open(path).unwrap(), "q", &Spread::Single, options, 4096).unwrap();
        let config = RuntimeConfig {
            pages: 96,
            cells_per_page: 128,
            max_slots: 4,
            ..RuntimeConfig::default()
        };
        let ordinal =
            std::env::var("ARES_TEST_DEVICE").ok().and_then(|d| d.parse().ok()).unwrap_or(0);
        Self::serve(model, config, vec![Device::open(ordinal, Wait::Block).unwrap()], path)
    }

    /// The 27B over both cards with its MTP block and the n-gram drafter
    ///, with exact products so that a row computes the same in a window
    /// as alone. The context is 1024 tokens, as in `real_sampling.rs`: the
    /// prompt with the tools and the replies of the test fit.
    fn start_drafting(path: &PathBuf) -> Self {
        let options = Options {
            precision: Precision::Exact,
            max_rows: 128,
            logit_rows: 16,
            window_rows: 4,
            ..Options::default()
        };
        let spread = Spread::Tensor(vec![0.625, 0.375]);
        let model = Qwen35::new(Gguf::open(path).unwrap(), "q", &spread, options, 1024).unwrap();
        let config = RuntimeConfig {
            pages: 24,
            cells_per_page: 128,
            max_slots: 4,
            window_rows: 4,
            ..RuntimeConfig::default()
        };
        let devices =
            vec![Device::open(0, Wait::Block).unwrap(), Device::open(1, Wait::Block).unwrap()];
        Self::serve(model, config, devices, path)
    }

    fn serve(model: Qwen35, config: RuntimeConfig, devices: Vec<Device>, path: &PathBuf) -> Self {
        let as_model: Arc<dyn Model> = Arc::new(model);
        let gpu = GpuConfig::new(&config, &*as_model);
        let runtime = Arc::new(Runtime::start_gpu(as_model, devices, config, gpu).unwrap());
        let engine: Arc<dyn Engine> = runtime.clone();
        Self {
            app: ares_server::router(engine.clone()),
            engine,
            runtime: Some(runtime),
            path: path.clone(),
            repeatable: true,
            _loaded: None,
        }
    }

    /// Checks that two replies to the same request are the same reply, where
    /// they must be (`api_model::must_agree`), and each on its own.
    fn agree(&self, reply: &Reply, other: &Reply, what: &str) {
        check(reply);
        check(other);
        if api_model::must_agree(self.repeatable, &reply.hash, &other.hash, what) {
            assert_eq!(reply, other, "{what}");
        }
    }

    async fn post(&self, body: &Value) -> (u16, String) {
        let request = Request::post(CHAT)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = self.app.clone().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    /// The reply to a request that is to succeed.
    async fn ask(&self, body: &Value) -> Reply {
        self.ask_timed(body).await.0
    }

    /// The drafts of the model's MTP block that the engine has verified
    /// and accepted so far, and those of the n-gram drafter.
    fn drafts(&self) -> [(u64, u64); 2] {
        let runtime = self.runtime.as_ref().expect("a runtime built here");
        let counters = runtime.snapshot().expect("the conductor answers").counters;
        [1, 0].map(|source| {
            let verified: u64 = counters.draft_lengths[source]
                .iter()
                .enumerate()
                .map(|(drafts, windows)| drafts as u64 * windows)
                .sum();
            (verified, counters.drafts_accepted_at[source].iter().sum())
        })
    }

    /// The reply to a request that is to succeed, and the timings the server
    /// reports with it.
    async fn ask_timed(&self, body: &Value) -> (Reply, Value) {
        let (status, text) = self.post(body).await;
        assert_eq!(status, 200, "{text}");
        let answer: Value = serde_json::from_str(&text).unwrap();
        let choice = &answer["choices"][0];
        let message = &choice["message"];
        let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
        let reply = Reply {
            content: message["content"].as_str().unwrap().to_owned(),
            reasoning: message["reasoning_content"].as_str().unwrap_or_default().to_owned(),
            calls: calls
                .iter()
                .map(|call| {
                    let function = &call["function"];
                    (
                        function["name"].as_str().unwrap().to_owned(),
                        function["arguments"].as_str().unwrap().to_owned(),
                    )
                })
                .collect(),
            finish: choice["finish_reason"].as_str().unwrap().to_owned(),
            hash: answer["timings"]["token_hash"].as_str().unwrap_or_default().to_owned(),
        };
        (reply, answer["timings"].clone())
    }

    /// The same request streamed, with the deltas put together.
    async fn stream(&self, body: &Value) -> Reply {
        let mut body = body.clone();
        body["stream"] = json!(true);
        let (status, text) = self.post(&body).await;
        assert_eq!(status, 200, "{text}");
        let mut reply = Reply::default();
        for event in text.split("\n\n") {
            let Some(data) = event.lines().find_map(|line| line.strip_prefix("data: ")) else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            let chunk: Value = serde_json::from_str(data).unwrap();
            let choice = &chunk["choices"][0];
            let delta = &choice["delta"];
            let content = delta["content"].as_str().unwrap_or_default();
            assert!(!content.contains("tool_call>"), "markup in a content delta: {content:?}");
            reply.content.push_str(content);
            reply.reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let index = call["index"].as_u64().unwrap() as usize;
                if let Some(name) = call["function"]["name"].as_str() {
                    assert_eq!(index, reply.calls.len(), "the name comes first: {call}");
                    assert!(call["id"].is_string() && call["type"] == "function", "{call}");
                    reply.calls.push((name.to_owned(), String::new()));
                }
                reply.calls[index].1.push_str(call["function"]["arguments"].as_str().unwrap());
            }
            if let Some(finish) = choice["finish_reason"].as_str() {
                reply.finish = finish.to_owned();
                reply.hash = chunk["timings"]["token_hash"].as_str().unwrap_or_default().to_owned();
            }
        }
        reply
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Reply {
    content: String,
    reasoning: String,
    /// Each call: its name and its arguments as the text that came.
    calls: Vec<(String, String)>,
    finish: String,
    /// The hash of the reply's tokens.
    hash: String,
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Get the weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "days"],
            },
        }},
        {"type": "function", "function": {
            "name": "get_time",
            "description": "Get the current time.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ])
}

fn request(question: &str, extra: Value) -> Value {
    let mut body = json!({
        "model": "q",
        "messages": [{"role": "user", "content": question}],
        "tools": tools(),
        "max_tokens": 400,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": false},
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

/// The calls are of the tools of the request, with whole JSON arguments of
/// their schema.
fn check(reply: &Reply) {
    assert!(!reply.content.contains("tool_call>"), "markup in the content: {reply:?}");
    for (name, arguments) in &reply.calls {
        let arguments: Value = serde_json::from_str(arguments)
            .unwrap_or_else(|e| panic!("the arguments are not JSON ({e}): {arguments}"));
        let arguments = arguments.as_object().expect("an object");
        match name.as_str() {
            "get_time" => assert!(arguments.is_empty(), "{arguments:?}"),
            "get_weather" => {
                assert!(arguments["city"].is_string(), "{arguments:?}");
                let days = arguments["days"].as_i64().expect("an integer");
                assert!((1..=7).contains(&days), "{days}");
                if let Some(unit) = arguments.get("unit") {
                    assert!(unit == "celsius" || unit == "fahrenheit", "{unit}");
                }
                assert!(arguments.keys().all(|k| ["city", "days", "unit"].contains(&k.as_str())));
            }
            other => panic!("a tool nobody offered: {other}"),
        }
    }
}

const WEATHER: &str = "What will the weather be in Paris over the next 3 days?";

#[tokio::test(flavor = "multi_thread")]
async fn the_template_of_the_model_declares_the_tagged_calls() {
    let Some(served) = Served::any() else { return };
    let format = served.engine.chat_format("q").expect("a chat format");
    let tools = format.tools.expect("a format for calls");
    assert_eq!(tools.trigger(), "<tool_call>");
    assert_eq!(tools.param_close, "\n</parameter>\n");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_that_is_required_or_named_is_a_valid_call() {
    let Some(served) = Served::any() else { return };
    for (what, extra) in [
        ("required", json!({"tool_choice": "required"})),
        ("required, one call", json!({"tool_choice": "required", "parallel_tool_calls": false})),
        (
            "named",
            json!({"tool_choice": {"type": "function", "function": {"name": "get_time"}},
                   "parallel_tool_calls": false}),
        ),
        (
            "required, thinking",
            json!({"tool_choice": "required", "chat_template_kwargs": {"enable_thinking": true},
                   "max_tokens": 1500}),
        ),
        ("required, sampled", json!({"tool_choice": "required", "temperature": 0.8, "seed": 3})),
    ] {
        let reply = served.ask(&request(WEATHER, extra)).await;
        println!("RES {what}: {reply:?}");
        assert_eq!(reply.finish, "tool_calls", "{what}: {reply:?}");
        assert!(!reply.calls.is_empty(), "{what}");
        check(&reply);
        assert_eq!(reply.content, "", "{what}: a required call has no text before it");
        if what.contains("one call") || what == "named" {
            assert_eq!(reply.calls.len(), 1, "{what}");
        }
        if what == "named" {
            assert_eq!(reply.calls[0].0, "get_time");
        }
        if what.contains("thinking") {
            assert!(!reply.reasoning.is_empty(), "the reasoning is free and comes back");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn with_auto_the_model_calls_or_answers_and_a_stream_is_the_same_reply() {
    let Some(served) = Served::any() else { return };
    let mut called = 0;
    for question in [
        WEATHER,
        "Say hello in German.",
        "What time is it, and how is the weather in Rome tomorrow?",
    ] {
        let body = request(question, json!({}));
        let reply = served.ask(&body).await;
        println!("RES auto {question:?}: {reply:?}");
        check(&reply);
        let expected = if reply.calls.is_empty() { "stop" } else { "tool_calls" };
        assert_eq!(reply.finish, expected, "{reply:?}");
        called += usize::from(!reply.calls.is_empty());
        let streamed = served.stream(&body).await;
        served.agree(&streamed, &reply, "the stream is the whole reply");
    }
    println!("RES auto: {called} of 3 questions were answered with calls");
    assert!(called >= 1, "the model never called a tool it was offered for the weather");
}

#[tokio::test(flavor = "multi_thread")]
async fn with_none_the_reply_is_text() {
    let Some(served) = Served::any() else { return };
    let reply = served.ask(&request(WEATHER, json!({"tool_choice": "none"}))).await;
    println!("RES none: {reply:?}");
    assert!(reply.calls.is_empty() && reply.finish != "tool_calls", "{reply:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_history_with_calls_and_results_is_answered() {
    let Some(served) = Served::any() else { return };
    let first = served.ask(&request(WEATHER, json!({"tool_choice": "required"}))).await;
    check(&first);
    let calls: Vec<Value> = first
        .calls
        .iter()
        .enumerate()
        .map(|(i, (name, arguments))| {
            json!({"id": format!("call_{i}"), "type": "function",
                   "function": {"name": name, "arguments": arguments}})
        })
        .collect();
    let mut messages = vec![
        json!({"role": "user", "content": WEATHER}),
        json!({"role": "assistant", "content": first.content, "tool_calls": calls}),
    ];
    for i in 0..first.calls.len() {
        messages.push(json!({"role": "tool", "tool_call_id": format!("call_{i}"),
                             "content": "Sunny, 21 degrees on all three days."}));
    }
    let reply = served.ask(&request(WEATHER, json!({"messages": messages}))).await;
    println!("RES after the results: {reply:?}");
    check(&reply);
    assert!(!reply.content.is_empty() || !reply.calls.is_empty(), "{reply:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_the_limit_cuts_ends_with_length_and_no_markup() {
    let Some(served) = Served::any() else { return };
    for limit in [3, 8, 14, 20] {
        let body = request(WEATHER, json!({"tool_choice": "required", "max_tokens": limit}));
        let reply = served.ask(&body).await;
        println!("RES cut at {limit}: {reply:?}");
        assert!(!reply.content.contains('<'), "{reply:?}");
        if reply.finish == "length" {
            assert!(reply.calls.len() <= 1, "{reply:?}");
        }
        let streamed = served.stream(&body).await;
        assert!(!streamed.content.contains('<'), "{streamed:?}");
        served.agree(&streamed, &reply, &format!("cut at {limit}"));
    }
}

/// The tools of the test of a call inside the thinking: no argument of them
/// is free text. The text of a string may be anything up to its closing tag,
/// the marker of a call included, and the bias that test puts on the marker
/// would write the marker there.
fn tools_without_text() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Get the weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "enum": ["Paris", "Rome", "Berlin"]},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                },
                "required": ["city", "days"],
            },
        }},
        {"type": "function", "function": {
            "name": "get_time",
            "description": "Get the current time.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ])
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_the_model_opens_inside_its_thinking_is_a_call() {
    let Some(served) = Served::any() else { return };
    // The model thinks, and the tag that closes its thinking is banned: a
    // call it makes it has to open inside the thinking. The tag and the
    // marker are the template's.
    //
    // Whether a model does that is the model's business: the 0.8B under the
    // ban alone thinks on to any limit and never writes the marker. So the
    // marker's token gets a bias, the least of a ladder with which the reply
    // has a call. With the first steps the model thinks before it calls; the
    // last is more than any logit is apart from another, and the marker is
    // the first token of such a reply whatever the model would have written.
    // One call is allowed, because the bias would open another after every
    // call.
    let format = served.engine.chat_format("q").expect("a chat format");
    let close = format.thinking.expect("the model thinks").end_tag().to_owned();
    let marker = format.tools.expect("a format for calls").trigger().to_owned();
    let tokenizer =
        ares_tokenizer::Tokenizer::from_gguf(&Gguf::open(&served.path).unwrap()).unwrap();
    let options = ares_tokenizer::EncodeOptions { add_special: false, parse_special: true };
    let closing = tokenizer.encode(&close, options);
    assert_eq!(closing.len(), 1, "{close} is {closing:?}");
    let opening = tokenizer.encode(&marker, options);
    assert_eq!(opening.len(), 1, "{marker} is {opening:?}");
    let pinned = |extra: Value, bias: Option<f64>| {
        let mut body = request(WEATHER, extra);
        body["tools"] = tools_without_text();
        body["parallel_tool_calls"] = json!(false);
        body["chat_template_kwargs"] = json!({"enable_thinking": true});
        body["logit_bias"] = match bias {
            Some(bias) => json!([[closing[0], false], [opening[0], bias]]),
            None => json!([[closing[0], false]]),
        };
        body["max_tokens"] = json!(300);
        body
    };
    const LADDER: [f64; 10] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 28.0, 1000.0];

    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    let mut thought = 0;
    for (what, extra) in [
        ("required", json!({"tool_choice": "required"})),
        ("named", json!({"tool_choice": named})),
        ("required, sampled", json!({"tool_choice": "required", "temperature": 0.8, "seed": 3})),
        ("auto", json!({})),
    ] {
        let mut found = None;
        for bias in LADDER {
            let body = pinned(extra.clone(), Some(bias));
            let reply = served.ask(&body).await;
            println!("RES in the thinking, {what}, a bias of {bias} on the marker: {reply:?}");
            assert_eq!(reply.content, "", "{what}: {reply:?}");
            assert!(!reply.reasoning.contains(&marker), "{what}: the marker in the reasoning");
            if reply.finish != "tool_calls" {
                // The model thought to the limit (or into a call that the
                // limit cut), or, with auto, which may, ended its reply
                // thinking.
                let ended = what == "auto" && reply.finish == "stop" && reply.calls.is_empty();
                assert!(reply.finish == "length" || ended, "{what}: {reply:?}");
                continue;
            }
            check(&reply);
            found = Some((body, reply));
            break;
        }
        let (body, reply) = found.expect("the marker comes with the last bias of the ladder");
        // The same reply raw shows that the model never left its thinking:
        // the content is the thinking from the tag the prompt ended with on,
        // and has no closing tag (which the model could still spell out
        // with other tokens; then nothing was tested here).
        let mut raw = body.clone();
        raw["reasoning_format"] = json!("none");
        let raw = served.ask(&raw).await;
        assert!(raw.content.starts_with("<think>"), "{what}: {raw:?}");
        if api_model::must_agree(served.repeatable, &raw.hash, &reply.hash, what) {
            assert!(
                !raw.content.contains(&close),
                "{what}: the model spelled the banned tag out, so no call was inside the thinking"
            );
            assert_eq!(raw.calls, reply.calls, "{what}: the raw reply has the same calls");
            // Nothing of the marker in the text, and no answer: the reasoning
            // up to the marker and the call after it are the whole reply.
            assert!(!raw.content.contains(&marker), "{what}: the marker in the content");
            let thought = format!("<think>\n{}", reply.reasoning);
            assert_eq!(raw.content.trim_end(), thought.trim_end());
        }
        assert_eq!(reply.finish, "tool_calls", "{what}: {reply:?}");
        assert_eq!(reply.calls, [reply.calls[0].clone()], "{what}: one call");
        if what == "named" {
            assert_eq!(reply.calls[0].0, "get_weather");
        }
        thought += usize::from(!reply.reasoning.is_empty());
        let streamed = served.stream(&body).await;
        served.agree(&streamed, &reply, &format!("{what}: the stream is the whole reply"));
        let mut plain = body.clone();
        plain["draft"] = json!(false);
        let without = served.ask(&plain).await;
        let same = format!("{what}: the tokens with drafts are those without");
        served.agree(&without, &reply, &same);
    }
    println!("RES in the thinking: {thought} of 4 replies thought before the call");
    assert!(thought > 0, "no reply thought before its call: the reasoning before one is not shown");

    // A reply that must call cannot end while it thinks, and one that never
    // writes the marker thinks to the limit: under the ban alone the reply is
    // reasoning, cut, with no call and no answer.
    let reply = served.ask(&pinned(json!({"tool_choice": "required"}), None)).await;
    println!("RES in the thinking, required, the ban alone: {reply:?}");
    if reply.calls.is_empty() {
        assert_eq!(reply.finish, "length", "{reply:?}");
        assert!(reply.content.is_empty() && !reply.reasoning.is_empty(), "{reply:?}");
    }

    // With none the same model under the same ban only thinks: whatever it
    // writes is reasoning, and there is no call.
    let body = pinned(json!({"tool_choice": "none"}), None);
    let reply = served.ask(&body).await;
    println!("RES in the thinking, none: {reply:?}");
    assert!(reply.calls.is_empty() && reply.finish != "tool_calls", "{reply:?}");
    assert_eq!(reply.content, "", "the thinking was never closed");
    let streamed = served.stream(&body).await;
    served.agree(&streamed, &reply, "none: the stream is the whole reply");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_27b_calls_the_same_with_drafts_and_without() {
    let Some(path) = large() else { return };
    let served = Served::start_drafting(&path);
    // The replies of the first two are a call from their first token: the
    // rule binds all of them, and the drafts are the n-gram drafter's. The third
    // thinks first, which the rule leaves free: there the request takes the
    // drafts of the model's MTP block, up to the token that makes the rule
    // bind.
    let thinking = json!({"chat_template_kwargs": {"enable_thinking": true}, "max_tokens": 600});
    for (what, extra) in [
        ("auto", json!({})),
        ("required", json!({"tool_choice": "required"})),
        ("auto, thinking", thinking),
    ] {
        let body = request(WEATHER, extra);
        let before = served.drafts();
        let (reply, timings) = served.ask_timed(&body).await;
        let after = served.drafts();
        println!("RES 27B {what}: {reply:?}");
        check(&reply);
        assert!(!reply.calls.is_empty(), "{what}: the 27B does not call the tool: {reply:?}");
        assert_eq!(reply.finish, "tool_calls", "{what}");
        let mut plain = body.clone();
        plain["draft"] = json!(false);
        let (without, plain_timings) = served.ask_timed(&plain).await;
        // What the engine counted for the request with drafts, and the speed
        // the server reports for the two: one run each, and the n-gram drafter has
        // the text of the replies before this one, so these are no benchmark.
        let [own, ngram] = [0, 1].map(|i| (after[i].0 - before[i].0, after[i].1 - before[i].1));
        println!(
            "RES 27B {what}: {} tokens; of the MTP block's drafts {} verified and {} accepted, \
             of the n-gram drafter's {} and {}; {} tokens/s with drafts, {} without",
            timings["predicted_n"],
            own.0,
            own.1,
            ngram.0,
            ngram.1,
            timings["predicted_per_second"],
            plain_timings["predicted_per_second"]
        );
        if what.contains("thinking") {
            assert!(!reply.reasoning.is_empty(), "{what}: the 27B did not think: {reply:?}");
            assert!(own.1 > 0, "{what}: no draft of the MTP block was accepted in the thinking");
        }
        assert_eq!(without.hash, reply.hash, "{what}: the tokens with drafts are those without");
        assert_eq!(without, reply, "{what}");
        assert_eq!(served.stream(&body).await, reply, "{what}: the stream is the whole reply");
    }
}
