//! The OpenAI Responses API through the server, with the real Qwen3.5-0.8B on
//! a GPU: a response comes back as items, a stream put together is the
//! response that is returned whole, a required call of a function has
//! arguments of its schema and its output is answered, a conversation whose
//! items are sent back (the reasoning item included) is counted and run, and
//! the counts of the two OpenAI routes are the prompts their requests report
//! when they run. Lines that start with `RES` say what the model did.
//!
//! The server runs in this process over the real engine (requests go through
//! the router without a socket). Needs `ARES_TEST_CUDA=1` and
//! `ARES_TEST_QWEN35_MODEL` (the 0.8B); `ARES_TEST_DEVICE` names the device
//! (default 0). With `ARES_TEST_API_MODEL` the tests run on that file instead,
//! loaded as `ares --model` loads it (Flash-Next, the 27B: see
//! `api_model/mod.rs`).
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test --release -p ares-server --test real_responses -- --test-threads=1 --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use ares_cuda::{Device, Precision, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Options, Qwen35, Spread};
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig};
use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, header};
use serde_json::{Value, json};
use tower::ServiceExt;

mod api_model;

const RESPONSES: &str = "/v1/responses";
const COUNT: &str = "/v1/responses/input_tokens";

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

    /// Checks that a stream put together is the response the same request
    /// gave whole, where they must be the same (`api_model::must_agree`, by
    /// the hash of the tokens in the timings).
    fn agree(&self, streamed: Value, whole: Value, what: &str) {
        let hash = |r: &Value| r["timings"]["token_hash"].as_str().unwrap_or_default().to_owned();
        if api_model::must_agree(self.repeatable, &hash(&streamed), &hash(&whole), what) {
            assert_eq!(anonymous(streamed), anonymous(whole), "{what}");
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
    let runtime = Runtime::start_gpu(model, devices, config, gpu).unwrap();
    ares_server::router(Arc::new(runtime))
}

async fn send(app: &Router, method: Method, path: &str, body: &Value) -> (u16, String) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer none")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn post(app: &Router, path: &str, body: &Value) -> (u16, String) {
    send(app, Method::POST, path, body).await
}

const QUESTION: &str = "What is the capital of France? One word.";

fn request(extra: Value) -> Value {
    let mut body = json!({
        "model": "q",
        "max_output_tokens": 200,
        "temperature": 0,
        "input": QUESTION,
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

/// What switches the thinking of a request off.
fn no_thinking() -> Value {
    json!({"effort": "none"})
}

/// The response a request is answered with.
async fn response(app: &Router, body: &Value) -> Value {
    let (status, text) = post(app, RESPONSES, body).await;
    assert_eq!(status, 200, "{text}");
    let response: Value = serde_json::from_str(&text).unwrap();
    assert_eq!((&response["object"], &response["store"]), (&json!("response"), &json!(false)));
    response
}

/// The same request streamed, with its events put together into a response.
async fn streamed(app: &Router, body: &Value) -> Value {
    let mut body = body.clone();
    body["stream"] = json!(true);
    let (status, text) = post(app, RESPONSES, &body).await;
    assert_eq!(status, 200, "{text}");
    let mut output: Vec<Value> = Vec::new();
    let mut last = Value::Null;
    let mut number = 0;
    for event in text.split("\n\n").filter(|event| !event.trim().is_empty()) {
        // Lines without a name are the comments that keep a stream alive.
        let Some(name) = event.lines().find_map(|l| l.strip_prefix("event: ")) else { continue };
        let data = event.lines().find_map(|l| l.strip_prefix("data: ")).expect("data");
        let data: Value = serde_json::from_str(data).unwrap();
        assert_eq!(data["type"], name);
        assert_eq!(data["sequence_number"], number, "{name}");
        number += 1;
        let index = data["output_index"].as_u64().unwrap_or(0) as usize;
        let append = |item: &mut Value, member: &str| {
            let so_far = item[member].as_str().unwrap().to_owned();
            item[member] = json!(so_far + data["delta"].as_str().unwrap());
        };
        match name {
            "response.created" | "response.in_progress" => {
                assert_eq!(data["response"]["status"], "in_progress");
            }
            "response.output_item.added" => {
                assert_eq!(index, output.len(), "items in order");
                output.push(data["item"].clone());
            }
            "response.content_part.added" => output[index]["content"] = json!([data["part"]]),
            "response.reasoning_text.delta" | "response.output_text.delta" => {
                append(&mut output[index]["content"][0], "text");
            }
            "response.function_call_arguments.delta" => append(&mut output[index], "arguments"),
            "response.reasoning_text.done" | "response.output_text.done" => {
                assert_eq!(data["text"], output[index]["content"][0]["text"]);
            }
            "response.function_call_arguments.done" => {
                assert_eq!(data["arguments"], output[index]["arguments"]);
            }
            "response.content_part.done" => assert_eq!(data["part"], output[index]["content"][0]),
            "response.output_item.done" => {
                output[index]["status"] = data["item"]["status"].clone();
                assert_eq!(data["item"], output[index], "the item is what its events said");
            }
            "response.completed" | "response.incomplete" => last = data["response"].clone(),
            other => panic!("an event named {other}: {data}"),
        }
    }
    assert_eq!(last["output"], Value::Array(output), "the output is what the events said");
    last
}

/// A response without what differs between two requests.
fn anonymous(response: Value) -> Value {
    let id = response["id"].as_str().expect("an id").to_owned();
    let request = id.strip_prefix("resp_").expect("the id of a response");
    let mut response: Value =
        serde_json::from_str(&response.to_string().replace(request, "ID")).unwrap();
    response["created_at"] = Value::Null;
    response.as_object_mut().unwrap().remove("timings");
    // The second of two equal requests reads its prompt from the cache.
    response["usage"]["input_tokens_details"] = Value::Null;
    response
}

fn kinds(response: &Value) -> Vec<&str> {
    response["output"].as_array().unwrap().iter().map(|i| i["type"].as_str().unwrap()).collect()
}

fn weather() -> Value {
    json!([{
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather of a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
        "strict": true,
    }])
}

/// The count of a request, which is the input the same request reports.
async fn counted(app: &Router, body: &Value) -> u64 {
    let (status, text) = post(app, COUNT, body).await;
    assert_eq!(status, 200, "{text}");
    let counted: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(counted["object"], "response.input_tokens");
    counted["input_tokens"].as_u64().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_response_is_items_and_its_stream_is_the_same_response() {
    let Some(served) = Served::any() else { return };
    let app = &served.app;

    // Thinking off: one message.
    let body = request(json!({"reasoning": no_thinking(), "instructions": "Answer briefly."}));
    let plain = response(app, &body).await;
    println!("RES plain: {plain}");
    assert_eq!(kinds(&plain), ["message"], "{plain}");
    let text = plain["output"][0]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Paris"), "{plain}");
    assert_eq!(plain["status"], "completed");
    assert!(plain["usage"]["output_tokens"].as_u64().unwrap() > 0);
    assert_eq!(counted(app, &body).await, plain["usage"]["input_tokens"].as_u64().unwrap());
    served.agree(streamed(app, &body).await, plain, "plain");

    // Thinking as the template has it: if the model thinks, its reasoning is
    // the first item, with the reasoning itself as its content.
    let body =
        request(json!({"max_output_tokens": 600, "include": ["reasoning.encrypted_content"]}));
    let thought = response(app, &body).await;
    println!("RES thinking: {thought}");
    if kinds(&thought).contains(&"reasoning") {
        assert_eq!(kinds(&thought)[0], "reasoning", "{thought}");
        let reasoning = &thought["output"][0];
        assert_eq!(reasoning["content"][0]["type"], "reasoning_text");
        assert!(!reasoning["content"][0]["text"].as_str().unwrap().is_empty());
        assert_eq!(reasoning["summary"], json!([]));
    } else {
        println!("RES thinking: the template leaves thinking off without a request for it");
    }
    served.agree(streamed(app, &body).await, thought.clone(), "thinking");

    // The output sent back with the next question: counted, and run with
    // the count as its input. How much of it the cache gave is printed.
    let mut input = vec![json!({"role": "user", "content": QUESTION})];
    input.extend(thought["output"].as_array().unwrap().clone());
    input.push(json!({"role": "user", "content": "And of Italy?"}));
    let mut next = body.clone();
    next["input"] = Value::Array(input);
    let count = counted(app, &next).await;
    let second = response(app, &next).await;
    println!("RES second turn: {second}");
    assert_eq!(second["usage"]["input_tokens"], count, "{second}");
    println!(
        "RES second turn: {} of {} input tokens from the cache (first turn: {} input, {} generated)",
        second["usage"]["input_tokens_details"]["cached_tokens"],
        second["usage"]["input_tokens"],
        thought["usage"]["input_tokens"],
        thought["usage"]["output_tokens"]
    );

    // The limit: the response is incomplete and says why.
    let cut = response(app, &request(json!({"max_output_tokens": 3}))).await;
    assert_eq!(cut["status"], "incomplete", "{cut}");
    assert_eq!(cut["incomplete_details"], json!({"reason": "max_output_tokens"}));
    assert_eq!(cut["usage"]["output_tokens"], 3);
    assert_eq!(cut["output"].as_array().unwrap().last().unwrap()["status"], "incomplete");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_required_call_is_a_function_call_item_and_its_output_is_answered() {
    let Some(served) = Served::any() else { return };
    let app = &served.app;
    let question = json!({"role": "user", "content": "How is the weather in Paris, in celsius?"});
    for choice in [json!("required"), json!({"type": "function", "name": "get_weather"})] {
        let body = request(json!({
            "input": [question],
            "tools": weather(),
            "tool_choice": choice,
            "reasoning": no_thinking(),
        }));
        let called = response(app, &body).await;
        println!("RES call ({choice}): {called}");
        assert_eq!(called["status"], "completed", "{called}");
        let call = called["output"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(
            (&call["type"], &call["name"]),
            (&json!("function_call"), &json!("get_weather"))
        );
        assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
        let arguments: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
        assert!(arguments["city"].is_string(), "{call}");
        assert!(["celsius", "fahrenheit"].contains(&arguments["unit"].as_str().unwrap()));
        served.agree(streamed(app, &body).await, called.clone(), "a call");

        // The output goes back, and the model answers in text.
        let mut input = vec![question.clone()];
        input.extend(called["output"].as_array().unwrap().clone());
        input.push(json!({
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": "Sunny, 21 degrees.",
        }));
        let mut next = body.clone();
        next["tool_choice"] = json!("auto");
        next["input"] = Value::Array(input);
        let count = counted(app, &next).await;
        let answer = response(app, &next).await;
        println!("RES answer: {answer}");
        assert_eq!(answer["usage"]["input_tokens"], count);
        assert!(kinds(&answer).contains(&"message"), "{answer}");
    }

    // auto with a question that needs no tool, and none: text either way.
    for choice in ["auto", "none"] {
        let body = request(json!({
            "tools": weather(),
            "tool_choice": choice,
            "reasoning": no_thinking(),
        }));
        let reply = response(app, &body).await;
        println!("RES {choice}: {reply}");
        if choice == "none" {
            assert!(!kinds(&reply).contains(&"function_call"), "{reply}");
        }
    }

    // A schema for the text of the answer.
    let schema = json!({
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
        "additionalProperties": false,
    });
    let format =
        json!({"type": "json_schema", "name": "capital", "strict": true, "schema": schema});
    let body = request(json!({"text": {"format": format}, "reasoning": no_thinking()}));
    let reply = response(app, &body).await;
    println!("RES schema: {reply}");
    let text = reply["output"].as_array().unwrap().last().unwrap()["content"][0]["text"].clone();
    let answer: Value = serde_json::from_str(text.as_str().unwrap()).unwrap();
    assert!(answer["capital"].is_string(), "{reply}");

    // What is not served is refused in OpenAI's shape, and there are no
    // stored responses.
    let image = json!({"type": "input_image", "image_url": "data:image/png;base64,AA=="});
    let refusals = [
        request(json!({"input": [{"role": "user", "content": [image]}]})),
        request(json!({"store": true})),
        request(json!({"previous_response_id": "resp_0000000000000000"})),
        request(json!({"tools": [{"type": "web_search"}]})),
    ];
    for body in refusals {
        let (status, text) = post(app, RESPONSES, &body).await;
        assert_eq!(status, 400, "{text}");
        let error: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error", "{text}");
    }
    let (status, text) =
        send(app, Method::GET, "/v1/responses/resp_0000000000000000", &json!({})).await;
    assert_eq!(status, 404, "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_count_of_a_chat_request_is_the_prompt_it_reports() {
    let Some(served) = Served::any() else { return };
    let app = &served.app;
    let chat_tools = json!([{"type": "function", "function": weather()[0]}]);
    let bodies = [
        json!({"messages": [{"role": "user", "content": QUESTION}]}),
        json!({
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": QUESTION},
            ],
            "tools": chat_tools,
            "tool_choice": "none",
            "chat_template_kwargs": {"enable_thinking": false},
        }),
    ];
    for extra in bodies {
        let mut body = json!({"model": "q", "max_tokens": 20, "temperature": 0});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let (status, text) = post(app, "/v1/chat/completions/input_tokens", &body).await;
        assert_eq!(status, 200, "{text}");
        let counted: Value = serde_json::from_str(&text).unwrap();
        let (status, text) = post(app, "/v1/chat/completions", &body).await;
        assert_eq!(status, 200, "{text}");
        let answer: Value = serde_json::from_str(&text).unwrap();
        println!("RES chat count: {counted} for {}", answer["usage"]);
        assert_eq!(counted["input_tokens"], answer["usage"]["prompt_tokens"], "{answer}");
    }
}
