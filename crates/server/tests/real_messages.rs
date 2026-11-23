//! The Anthropic Messages API through the server, with the real Qwen3.5-0.8B
//! on a GPU: a message comes back as blocks, a stream put together is the
//! message that is returned whole, a required call of a tool has an input of
//! its schema and its result is answered, a conversation that is sent back
//! (thinking blocks included) is counted and run, and the count of a request
//! is the input it reports when it runs. Lines that start with `RES` say what
//! the model did.
//!
//! The server runs in this process over the real engine (requests go through
//! the router without a socket). Needs `ARES_TEST_CUDA=1` and
//! `ARES_TEST_QWEN35_MODEL` (the 0.8B); `ARES_TEST_DEVICE` names the device
//! (default 0). With `ARES_TEST_API_MODEL` the tests run on that file instead,
//! loaded as `ares --model` loads it (Flash-Next, the 27B: see
//! `api_model/mod.rs`); a message of a model whose replies are not repeatable
//! is compared with its stream only when the two are the same.
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test --release -p ares-server --test real_messages -- --test-threads=1 --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

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

const MESSAGES: &str = "/v1/messages";
const COUNT: &str = "/v1/messages/count_tokens";

fn small() -> Option<PathBuf> {
    if std::env::var_os("ARES_TEST_CUDA").is_none() {
        eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
        return None;
    }
    let path = std::env::var_os("ARES_TEST_QWEN35_MODEL").map(PathBuf::from);
    if path.is_none() {
        eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to the Qwen3.5-0.8B model file");
    }
    path
}

/// The server, and what it serves.
struct Served {
    app: Router,
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
            let app = ares_server::router(model.engine());
            return Some(Self { app, repeatable: model.repeatable, _loaded: Some(model) });
        }
        let app = served(&small()?);
        Some(Self { app, repeatable: true, _loaded: None })
    }

    /// Checks that a stream put together is the message the same request
    /// gave whole: always for a repeatable model, and for one that is not
    /// when they are the same reply (the API has no hash of the tokens).
    fn agree(&self, streamed: Value, whole: Value, what: &str) {
        let (streamed, whole) = (anonymous(streamed), anonymous(whole));
        if self.repeatable {
            assert_eq!(streamed, whole, "{what}");
        } else if streamed != whole {
            println!("RES {what}: the stream is another reply (not repeatable): {streamed}");
        }
    }
}

/// The server over the 0.8B on one device.
fn served(path: &PathBuf) -> Router {
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
    let config =
        RuntimeConfig { pages: 96, cells_per_page: 128, max_slots: 4, ..RuntimeConfig::default() };
    let ordinal = std::env::var("ARES_TEST_DEVICE").ok().and_then(|d| d.parse().ok()).unwrap_or(0);
    let devices = vec![Device::open(ordinal, Wait::Block).unwrap()];
    let model: Arc<dyn Model> = Arc::new(model);
    let gpu = GpuConfig::new(&config, &*model);
    ares_server::router(Arc::new(Runtime::start_gpu(model, devices, config, gpu).unwrap()))
}

async fn post(app: &Router, path: &str, body: &Value) -> (u16, String) {
    let request = Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-api-key", "none")
        .header("anthropic-version", "2023-06-01")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn request(extra: Value) -> Value {
    let mut body = json!({
        "model": "q",
        "max_tokens": 200,
        "temperature": 0,
        "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

/// The message a request is answered with.
async fn message(app: &Router, body: &Value) -> Value {
    let (status, text) = post(app, MESSAGES, body).await;
    assert_eq!(status, 200, "{text}");
    let message: Value = serde_json::from_str(&text).unwrap();
    assert_eq!((&message["type"], &message["role"]), (&json!("message"), &json!("assistant")));
    message
}

/// The same request streamed, with its events put together into a message.
async fn streamed(app: &Router, body: &Value) -> Value {
    let mut body = body.clone();
    body["stream"] = json!(true);
    let (status, text) = post(app, MESSAGES, &body).await;
    assert_eq!(status, 200, "{text}");
    let mut message = Value::Null;
    let mut content: Vec<Value> = Vec::new();
    let mut inputs: Vec<String> = Vec::new();
    for event in text.split("\n\n").filter(|event| !event.trim().is_empty()) {
        let name = event.lines().find_map(|l| l.strip_prefix("event: ")).expect("a name");
        let data = event.lines().find_map(|l| l.strip_prefix("data: ")).expect("data");
        let data: Value = serde_json::from_str(data).unwrap();
        assert_eq!(data["type"], name);
        let index = data["index"].as_u64().unwrap_or(0) as usize;
        match name {
            "message_start" => message = data["message"].clone(),
            "content_block_start" => {
                assert_eq!(index, content.len(), "blocks in order");
                content.push(data["content_block"].clone());
                inputs.push(String::new());
            }
            "content_block_delta" => {
                let delta = &data["delta"];
                let member = match delta["type"].as_str().unwrap() {
                    "thinking_delta" => "thinking",
                    "signature_delta" => "signature",
                    "text_delta" => "text",
                    "input_json_delta" => {
                        inputs[index].push_str(delta["partial_json"].as_str().unwrap());
                        continue;
                    }
                    other => panic!("a delta of the type {other}"),
                };
                let so_far = content[index][member].as_str().unwrap().to_owned();
                content[index][member] = json!(so_far + delta[member].as_str().unwrap());
            }
            "content_block_stop" => {
                if content[index]["type"] == "tool_use" {
                    content[index]["input"] =
                        serde_json::from_str(&inputs[index]).unwrap_or_else(|_| json!({}));
                }
            }
            "message_delta" => {
                message["stop_reason"] = data["delta"]["stop_reason"].clone();
                message["stop_sequence"] = data["delta"]["stop_sequence"].clone();
                message["usage"] = data["usage"].clone();
            }
            "message_stop" | "ping" => {}
            other => panic!("an event named {other}: {data}"),
        }
    }
    message["content"] = Value::Array(content);
    message
}

/// A message without what differs between two requests.
fn anonymous(mut message: Value) -> Value {
    message["id"] = Value::Null;
    // The second of two equal requests reads its prompt from the cache.
    message["usage"]["cache_read_input_tokens"] = Value::Null;
    message["usage"]["input_tokens"] = Value::Null;
    for (place, block) in message["content"].as_array_mut().unwrap().iter_mut().enumerate() {
        if block["type"] == "tool_use" {
            block["id"] = json!(place);
        }
    }
    message
}

fn prompt_tokens(message: &Value) -> u64 {
    let usage = &message["usage"];
    usage["input_tokens"].as_u64().unwrap() + usage["cache_read_input_tokens"].as_u64().unwrap()
}

fn weather() -> Value {
    json!([{
        "name": "get_weather",
        "description": "Get the current weather of a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    }])
}

#[tokio::test(flavor = "multi_thread")]
async fn a_message_is_blocks_and_its_stream_is_the_same_message() {
    let Some(served) = Served::any() else { return };
    let app = &served.app;

    // Thinking off: one text block.
    let body = request(json!({"thinking": {"type": "disabled"}}));
    let plain = message(app, &body).await;
    println!("RES plain: {plain}");
    let content = plain["content"].as_array().unwrap();
    assert_eq!(content.len(), 1, "{plain}");
    assert_eq!(content[0]["type"], "text");
    assert!(content[0]["text"].as_str().unwrap().contains("Paris"), "{plain}");
    assert_eq!(plain["stop_reason"], "end_turn");
    assert!(plain["usage"]["output_tokens"].as_u64().unwrap() > 0);
    served.agree(streamed(app, &body).await, plain, "plain");

    // Thinking on: a thinking block first, signed with the fixed text.
    let body =
        request(json!({"thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 600}));
    let thought = message(app, &body).await;
    println!("RES thinking: {thought}");
    let content = thought["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "thinking", "{thought}");
    assert!(!content[0]["thinking"].as_str().unwrap().is_empty());
    assert_eq!(content[0]["signature"], "unsigned");
    served.agree(streamed(app, &body).await, thought.clone(), "thinking");

    // The reply sent back with the next question: counted, and run with the
    // count as its prompt. How much of it the cache gave is printed.
    let mut next = body.clone();
    next["messages"] = json!([
        body["messages"][0],
        {"role": "assistant", "content": thought["content"]},
        {"role": "user", "content": "And of Italy?"},
    ]);
    let (status, counted) = post(app, COUNT, &next).await;
    assert_eq!(status, 200, "{counted}");
    let counted: Value = serde_json::from_str(&counted).unwrap();
    let second = message(app, &next).await;
    println!("RES second turn: {second}");
    assert_eq!(counted["input_tokens"], prompt_tokens(&second), "{counted} {second}");
    println!(
        "RES second turn: {} of {} prompt tokens from the cache (first turn: {} prompt, {} generated)",
        second["usage"]["cache_read_input_tokens"],
        prompt_tokens(&second),
        prompt_tokens(&thought),
        thought["usage"]["output_tokens"]
    );

    // The limit: the reason says so.
    let cut = message(app, &request(json!({"max_tokens": 3}))).await;
    assert_eq!(cut["stop_reason"], "max_tokens", "{cut}");
    assert_eq!(cut["usage"]["output_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_required_call_is_a_tool_use_block_and_its_result_is_answered() {
    let Some(served) = Served::any() else { return };
    let app = &served.app;
    let question = json!({"role": "user", "content": "How is the weather in Paris, in celsius?"});
    for choice in [json!({"type": "any"}), json!({"type": "tool", "name": "get_weather"})] {
        let body = request(json!({
            "messages": [question],
            "tools": weather(),
            "tool_choice": choice,
            "thinking": {"type": "disabled"},
        }));
        let called = message(app, &body).await;
        println!("RES call ({}): {called}", choice["type"]);
        assert_eq!(called["stop_reason"], "tool_use", "{called}");
        let content = called["content"].as_array().unwrap();
        let call = content.last().unwrap();
        assert_eq!((&call["type"], &call["name"]), (&json!("tool_use"), &json!("get_weather")));
        assert!(call["id"].as_str().unwrap().starts_with("toolu_"));
        assert!(call["input"]["city"].is_string(), "{call}");
        assert!(["celsius", "fahrenheit"].contains(&call["input"]["unit"].as_str().unwrap()));
        served.agree(streamed(app, &body).await, called.clone(), "a call");

        // The result goes back, and the model answers in text.
        let mut next = body.clone();
        next["tool_choice"] = json!({"type": "auto"});
        next["messages"] = json!([
            question,
            {"role": "assistant", "content": called["content"]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call["id"], "content": "Sunny, 21 degrees."},
            ]},
        ]);
        let (status, counted) = post(app, COUNT, &next).await;
        assert_eq!(status, 200, "{counted}");
        let counted: Value = serde_json::from_str(&counted).unwrap();
        let answer = message(app, &next).await;
        println!("RES answer: {answer}");
        assert_eq!(counted["input_tokens"], prompt_tokens(&answer));
        assert!(answer["content"].as_array().unwrap().iter().any(|block| block["type"] == "text"));
    }

    // auto with a question that needs no tool, and none: text either way.
    for choice in ["auto", "none"] {
        let body = request(json!({
            "tools": weather(),
            "tool_choice": {"type": choice},
            "thinking": {"type": "disabled"},
        }));
        let reply = message(app, &body).await;
        println!("RES {choice}: {reply}");
        if choice == "none" {
            assert_ne!(reply["stop_reason"], "tool_use", "{reply}");
        }
    }

    // What is not served is refused in Anthropic's shape.
    let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}});
    let (status, text) =
        post(app, MESSAGES, &request(json!({"messages": [{"role": "user", "content": [image]}]})))
            .await;
    assert_eq!(status, 400, "{text}");
    let error: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        (&error["type"], &error["error"]["type"]),
        (&json!("error"), &json!("invalid_request_error"))
    );
}

/// A budget of thinking ends the thinking at its number of tokens and the
/// answer follows: `thinking.budget_tokens` here, `reasoning_budget_tokens`
/// on the chat route (whose usage counts the tokens of the thinking).
/// `thinking.display: "omitted"` gives the block an empty text and a
/// signature the next turn reads it from.
#[tokio::test(flavor = "multi_thread")]
async fn a_budget_ends_the_thinking_and_the_answer_follows() {
    let Some(served) = Served::any() else { return };
    let app = &served.app;
    let question = "Think step by step: how many prime numbers are there below 60? \
                    Then answer with the number.";
    let budget = 24;
    let body = json!({
        "model": "q",
        "max_tokens": 400,
        "temperature": 0,
        "messages": [{"role": "user", "content": question}],
        "chat_template_kwargs": {"enable_thinking": true},
        "reasoning_budget_tokens": budget,
    });
    let (status, text) = post(app, "/v1/chat/completions", &body).await;
    assert_eq!(status, 200, "{text}");
    let reply: Value = serde_json::from_str(&text).unwrap();
    let thought = reply["usage"]["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap();
    let answer = &reply["choices"][0]["message"];
    println!(
        "RES a budget of {budget}: {thought} tokens of thinking (the closing tag's among them), \
         answer {:?}",
        answer["content"].as_str().unwrap_or_default()
    );
    // The budget, then the forced end: a newline and the closing tag.
    assert!(thought <= budget + 4, "{thought} tokens of thinking: {reply}");
    assert!(!answer["reasoning_content"].as_str().unwrap_or_default().is_empty(), "{reply}");
    assert!(!answer["content"].as_str().unwrap_or_default().trim().is_empty(), "{reply}");

    let body = json!({
        "model": "q",
        "max_tokens": 400,
        "temperature": 0,
        "messages": [{"role": "user", "content": question}],
        "thinking": {"type": "enabled", "budget_tokens": budget, "display": "omitted"},
    });
    let whole = message(app, &body).await;
    let blocks = whole["content"].as_array().unwrap();
    assert_eq!(blocks[0]["type"], "thinking", "{whole}");
    assert_eq!(blocks[0]["thinking"], "");
    assert!(blocks[0]["signature"].as_str().unwrap().starts_with("ares-reasoning:"));
    assert!(blocks.iter().any(|b| b["type"] == "text"), "an answer after the thinking: {whole}");
    // Sent back, the omitted thinking is in the prompt: the count of the
    // conversation is what it is with the thinking shown.
    let next = |content: Value| {
        json!({"model": "q", "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": content},
            {"role": "user", "content": "And below 100?"},
        ]})
    };
    let (_, omitted) = post(app, COUNT, &next(json!(blocks))).await;
    let mut shown = blocks.clone();
    let carried = blocks[0]["signature"].as_str().unwrap().trim_start_matches("ares-reasoning:");
    shown[0] = json!({"type": "thinking", "thinking": decode(carried), "signature": "unsigned"});
    let (_, counted) = post(app, COUNT, &next(json!(shown))).await;
    assert_eq!(omitted, counted);
}

/// Base64 back to text, for the test's own check.
fn decode(text: &str) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut bits = 0u32;
    let mut count = 0;
    let mut out = Vec::new();
    for c in text.bytes().filter(|c| *c != b'=') {
        bits = bits << 6 | ALPHABET.iter().position(|a| *a == c).unwrap() as u32;
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push((bits >> count) as u8);
        }
    }
    String::from_utf8(out).unwrap()
}
