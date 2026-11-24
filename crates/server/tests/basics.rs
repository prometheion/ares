//! The basics of the server over HTTP: tokens and properties of a model, the
//! stop string that ended a reply, the tokens of the thinking, continuing the
//! last message, and the line of requests. Through the real runtime with the
//! synthetic model on a host executor (one token for each byte, which prints
//! as one character), and through a scripted engine for the shapes of the
//! other APIs.

mod common;

use std::sync::Arc;
use std::time::Duration;

use ares_api::{ChatFormat, Engine, Event, FinishReason, ThinkingFormat, Usage};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use ares_server::options::ModelOptions;
use axum::Router;
use axum::http::{StatusCode, header};
use common::*;
use serde_json::{Value, json};

/// The synthetic model behind the server; `close` makes it a model that
/// thinks until it writes `close` (the prompt opens the block), `queue` gives
/// the engine one sequence and a line of that many.
fn synthetic(close: Option<&str>, queue: Option<u32>) -> Arc<Runtime> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.eos = Some(u32::from(b'~'));
    config.chat = close.map(|close| ChatFormat {
        thinking: Some(ThinkingFormat {
            start: "<think>".into(),
            end: close.into(),
            prompt_on: "<think>".into(),
            prompt_off: format!("<think>{close}"),
            on_by_default: true,
            effort: false,
            refused_efforts: Vec::new(),
        }),
        tools: None,
        system_first: true,
    });
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let config = RuntimeConfig {
        pages: 4096,
        cells_per_page: 16,
        max_slots: if queue.is_some() { 1 } else { 8 },
        queue,
        window_rows: 1,
        warm_up: false,
        ..RuntimeConfig::default()
    };
    let cost = if queue.is_some() { 2000 } else { 30 };
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_row_cost(Duration::from_micros(cost), Duration::from_micros(5));
    let runtime = Runtime::start(model, Box::new(host), config).unwrap();
    Arc::new(runtime.with_chat_template(Some("{{ messages }}".into())))
}

async fn served(runtime: &Arc<Runtime>) -> Router {
    let engine: Arc<dyn Engine> = runtime.clone();
    router_of(engine, ModelOptions::default(), ares_server::Config::default()).await
}

async fn call(app: &Router, path: &str, body: Value) -> (StatusCode, Value) {
    json(post(app, path, &body.to_string()).await).await
}

/// The bytes of a text as the synthetic model's tokens.
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

#[tokio::test]
async fn text_and_conversations_become_the_tokens_a_request_gets() {
    let app = served(&synthetic(None, None)).await;
    let (status, body) = call(&app, "/tokenize", json!({"content": "Hi!"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"tokens": bytes("Hi!")}));
    let (_, body) = call(
        &app,
        "/tokenize",
        json!({"content": "Hi", "with_pieces": true, "model": "synthetic"}),
    )
    .await;
    assert_eq!(body, json!({"tokens": [{"id": 72, "piece": "H"}, {"id": 105, "piece": "i"}]}));

    // A conversation is the prompt the chat route makes of it, and its count
    // is the one of input_tokens.
    let chat = json!({"messages": [{"role": "user", "content": "hi"}]});
    let (_, body) = call(&app, "/tokenize", chat.clone()).await;
    assert_eq!(body["tokens"], json!(bytes("user: hi\nassistant: ")));
    let mut counted = chat.clone();
    counted["model"] = json!("synthetic");
    let (_, count) = call(&app, "/v1/chat/completions/input_tokens", counted).await;
    assert_eq!(count["input_tokens"], 20);

    // A last message of the assistant is continued, unless the request says no.
    let prefill =
        json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hel"}]);
    let (_, body) = call(&app, "/tokenize", json!({"messages": prefill})).await;
    assert_eq!(body["tokens"], json!(bytes("user: hi\nassistant: Hel")));
    let asked = json!({"messages": prefill, "continue_final_message": false});
    let (_, body) = call(&app, "/tokenize", asked).await;
    assert_eq!(body["tokens"], json!(bytes("user: hi\nassistant: Hel\nassistant: ")));

    // Ids back to text.
    let (status, body) = call(&app, "/detokenize", json!({"tokens": bytes("Hi!")})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"content": "Hi!"}));

    // llama-server's default for reading control tokens is what is done.
    let (status, body) =
        call(&app, "/tokenize", json!({"content": "Hi", "parse_special": true})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["tokens"], json!(bytes("Hi")));

    // What is not served is refused by name.
    for (request, param) in [
        (json!({"content": "x", "add_special": false}), "add_special"),
        (json!({"content": "x", "parse_special": false}), "parse_special"),
        (json!({"content": 3}), "content"),
        (json!({}), "content"),
        (json!({"content": "x", "echo": true}), "echo"),
    ] {
        let (status, body) = call(&app, "/tokenize", request.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{request}");
        assert_eq!(body["error"]["param"], param, "{request}: {body}");
    }
    for (request, param) in [
        (json!({"tokens": [1, -2]}), "tokens"),
        (json!({"tokens": "1 2"}), "tokens"),
        (json!({"tokens": [1], "text": "x"}), "text"),
    ] {
        let (status, body) = call(&app, "/detokenize", request.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{request}");
        assert_eq!(body["error"]["param"], param, "{request}: {body}");
    }
    let (status, body) = call(&app, "/detokenize", json!({"tokens": [100000]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("100000"));
}

#[tokio::test]
async fn props_say_what_the_model_is() {
    let app = served(&synthetic(Some("</think>"), None)).await;
    let (status, body) = json(get(&app, "/props").await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "synthetic");
    assert_eq!(body["context_length"], 2048);
    assert_eq!(body["default_generation_settings"]["n_ctx"], 2048);
    assert_eq!(body["default_generation_settings"]["params"], json!({}));
    assert_eq!(body["chat_template"], "{{ messages }}");
    assert_eq!(body["modalities"], json!({"vision": false, "audio": false}));
    let caps = &body["chat_template_caps"];
    assert_eq!(caps["supports_reasoning"], true);
    assert_eq!(caps["supports_tools"], false);
    assert_eq!(caps["supports_enable_thinking"], true);
    assert_eq!(caps["thinks_by_default"], true);
    assert_eq!(body["reasoning"], json!({"start": "<think>", "end": "</think>"}));
    let (_, same) = json(get(&app, "/props?model=synthetic").await).await;
    assert_eq!(same, body);
}

#[tokio::test]
async fn the_stop_string_that_ended_a_reply_is_named() {
    let runtime = synthetic(None, None);
    let app = served(&runtime).await;
    let ask = |extra: Value| {
        let mut body = json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "say something"}],
            "max_tokens": 60,
        });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        body
    };
    let (_, free) = call(&app, "/v1/chat/completions", ask(json!({}))).await;
    let text = free["choices"][0]["message"]["content"].as_str().unwrap().to_owned();
    assert!(text.chars().count() > 20, "{text:?}");
    assert!(free["choices"][0].get("stop_reason").is_none(), "{free}");
    // A stop string from the middle of the greedy reply, behind one that never comes.
    let stop: String = text.chars().skip(10).take(3).collect();
    let before = &text[..text.find(&stop).unwrap()];
    let (_, stopped) =
        call(&app, "/v1/chat/completions", ask(json!({"stop": ["\u{2603}never", stop]}))).await;
    let choice = &stopped["choices"][0];
    assert_eq!(choice["message"]["content"], before);
    assert_eq!(choice["finish_reason"], "stop");
    assert_eq!(choice["stop_reason"], stop, "{stopped}");

    // Streamed, on the finish chunk.
    let response = post(
        &app,
        "/v1/chat/completions",
        &ask(json!({"stop": [stop], "stream": true})).to_string(),
    )
    .await;
    let stream = parse_sse(&body_text(response).await);
    let last = stream.chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["choices"][0]["stop_reason"], stop);

    // Anthropic's stop_sequence, whole and streamed.
    let messages = json!({
        "model": "synthetic",
        "max_tokens": 60,
        "messages": [{"role": "user", "content": "say something"}],
        "stop_sequences": [stop],
    });
    let (_, message) = call(&app, "/v1/messages", messages.clone()).await;
    assert_eq!(message["stop_reason"], "stop_sequence", "{message}");
    assert_eq!(message["stop_sequence"], stop);
    let mut streamed = messages.clone();
    streamed["stream"] = json!(true);
    let text = body_text(post(&app, "/v1/messages", &streamed.to_string()).await).await;
    let delta = sse_payloads(&text)
        .into_iter()
        .map(|p| serde_json::from_str::<Value>(&p).unwrap())
        .find(|event| event["type"] == "message_delta")
        .unwrap();
    assert_eq!(delta["delta"], json!({"stop_reason": "stop_sequence", "stop_sequence": stop}));
    let (_, ended) = call(
        &app,
        "/v1/messages",
        json!({
            "model": "synthetic", "max_tokens": 60,
            "messages": [{"role": "user", "content": "say something"}],
        }),
    )
    .await;
    assert_eq!(
        (ended["stop_sequence"].clone(), ended["stop_reason"].clone()),
        (Value::Null, json!("max_tokens"))
    );
}

#[tokio::test]
async fn the_tokens_of_the_thinking_are_counted() {
    // A prompt that opens the thinking: the reply begins inside it.
    let prompt = "think about it <think>";
    let ask = json!({"model": "synthetic", "prompt": prompt, "max_tokens": 200});
    let (_, free) =
        call(&served(&synthetic(None, None)).await, "/v1/completions", ask.clone()).await;
    let text = free["choices"][0]["text"].as_str().unwrap().to_owned();
    assert!(free["usage"].get("completion_tokens_details").is_none(), "{free}");
    // The same model thinks until a text from the middle of that reply.
    let close: String = text.chars().skip(12).take(2).collect();
    let (_, thought) =
        call(&served(&synthetic(Some(&close), None)).await, "/v1/completions", ask).await;
    assert_eq!(thought["choices"][0]["text"], text, "the same greedy reply");
    let at = text.find(&close).unwrap();
    let expected = text[..at].chars().count() + close.chars().count();
    assert_eq!(
        thought["usage"]["completion_tokens_details"]["reasoning_tokens"], expected,
        "{thought}"
    );
    // A prompt that does not open it: the reply never does either.
    let other = json!({"model": "synthetic", "prompt": "no tags", "max_tokens": 20});
    let (_, body) =
        call(&served(&synthetic(Some(&close), None)).await, "/v1/completions", other).await;
    assert_eq!(body["usage"]["completion_tokens_details"]["reasoning_tokens"], 0, "{body}");
}

#[tokio::test]
async fn a_continued_message_goes_on_where_it_stands() {
    let close = "</think>";
    let app = served(&synthetic(Some(close), None)).await;
    // A reply that continues reasoning is reasoning until the closing tag.
    let request = json!({
        "model": "synthetic",
        "messages": [
            {"role": "user", "content": "think"},
            {"role": "assistant", "content": "", "reasoning_content": "Let me see"},
        ],
        "max_tokens": 5,
    });
    let (status, body) = call(&app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], "", "{body}");
    assert!(!message["reasoning_content"].as_str().unwrap_or_default().is_empty(), "{body}");

    // A reply that continues the answer is answer.
    let request = json!({
        "model": "synthetic",
        "messages": [
            {"role": "user", "content": "say"},
            {"role": "assistant", "content": "Well,"},
        ],
        "max_tokens": 5,
    });
    let (_, body) = call(&app, "/v1/chat/completions", request).await;
    let message = &body["choices"][0]["message"];
    assert!(message.get("reasoning_content").is_none(), "{body}");

    // What cannot be continued is refused by name.
    let calls =
        json!([{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]);
    for (extra, messages, param) in [
        (
            json!({"continue_final_message": true}),
            json!([{"role": "user", "content": "a"}]),
            "continue_final_message",
        ),
        (
            json!({"add_generation_prompt": false}),
            json!([{"role": "user", "content": "a"}]),
            "add_generation_prompt",
        ),
        (
            json!({"continue_final_message": true, "add_generation_prompt": true}),
            json!([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]),
            "continue_final_message",
        ),
        (
            json!({"continue_final_message": true}),
            json!([{"role": "user", "content": "a"}, {"role": "assistant", "content": "", "tool_calls": calls}]),
            "continue_final_message",
        ),
    ] {
        let mut request = json!({"model": "synthetic", "messages": messages, "max_tokens": 4});
        request.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let (status, body) = call(&app, "/v1/chat/completions", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["param"], param, "{body}");
    }
    // A last message that calls tools begins a turn of its own unless asked.
    let request = json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "", "tool_calls": calls}],
        "max_tokens": 4,
    });
    let (status, body) = call(&app, "/v1/chat/completions", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn anthropics_prefill_is_continued() {
    let engine = MockEngine::script(vec![text("ok"), done(FinishReason::Stop, 3, 1)]);
    let body = json!({
        "model": "mock",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "{\"x\":"}],
    });
    let (status, reply) =
        json(post(&app(&engine).await, "/v1/messages", &body.to_string()).await).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    let ares_api::Prompt::Chat { messages, continue_final, .. } = &engine.submitted()[0].prompt
    else {
        panic!()
    };
    assert!(*continue_final);
    assert_eq!(messages.last().unwrap().content, "{\"x\":");
}

#[tokio::test]
async fn the_reports_of_a_scripted_engine_reach_every_api() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 8,
        reasoning_tokens: Some(5),
        ..Usage::default()
    };
    let script = || {
        MockEngine::script(vec![
            text("ok"),
            Step::Send(Event::Done { finish: FinishReason::StopString(1), usage }),
        ])
    };
    let chat = json!({"model": "mock", "messages": [{"role": "user", "content": "a"}], "stop": ["x", "END"]});
    let (_, body) = call(&app(&script()).await, "/v1/chat/completions", chat).await;
    assert_eq!(body["choices"][0]["stop_reason"], "END");
    assert_eq!(body["usage"]["completion_tokens_details"], json!({"reasoning_tokens": 5}));
    let messages = json!({
        "model": "mock", "max_tokens": 8,
        "messages": [{"role": "user", "content": "a"}],
        "stop_sequences": ["x", "END"],
    });
    let (_, body) = call(&app(&script()).await, "/v1/messages", messages).await;
    assert_eq!(
        (body["stop_reason"].clone(), body["stop_sequence"].clone()),
        (json!("stop_sequence"), json!("END"))
    );
    let responses = json!({"model": "mock", "input": "a"});
    let (_, body) = call(&app(&script()).await, "/v1/responses", responses).await;
    assert_eq!(body["usage"]["output_tokens_details"], json!({"reasoning_tokens": 5}), "{body}");
}

#[tokio::test]
async fn a_full_line_refuses_and_says_when_to_retry() {
    let app = served(&synthetic(None, Some(0))).await;
    let long = json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "go on"}],
        "max_tokens": 1500,
        "stream": true,
    });
    // The one sequence runs a long reply; the line holds nobody.
    let running = post(&app, "/v1/chat/completions", &long.to_string()).await;
    assert_eq!(running.status(), StatusCode::OK);
    let short = json!({"model": "synthetic", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2});
    let refused = post(&app, "/v1/chat/completions", &short.to_string()).await;
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused.headers()[header::RETRY_AFTER], "1");
    let (_, body) = json(refused).await;
    assert_eq!(body["error"]["code"], "overloaded");
    // When the running reply goes away, the line is free again.
    drop(running);
    let mut status = StatusCode::SERVICE_UNAVAILABLE;
    for _ in 0..200 {
        status = post(&app, "/v1/chat/completions", &short.to_string()).await.status();
        if status == StatusCode::OK {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn metrics_have_the_times_of_the_requests() {
    let usage = Usage {
        prompt_tokens: 4,
        completion_tokens: 3,
        queued: Some(Duration::from_millis(5)),
        ..Usage::default()
    };
    let engine = MockEngine::script(vec![
        text("a"),
        text("b"),
        Step::Send(Event::Done { finish: FinishReason::Stop, usage }),
    ]);
    engine.set_stats(ares_api::EngineStats::default());
    let app = app(&engine).await;
    let body = json!({"model": "mock", "messages": [{"role": "user", "content": "a"}]});
    for _ in 0..2 {
        let (status, _) = call(&app, "/v1/chat/completions", body.clone()).await;
        assert_eq!(status, StatusCode::OK);
    }
    let text = body_text(get(&app, "/metrics").await).await;
    for name in [
        "ares_request_queue_seconds",
        "ares_request_prompt_seconds",
        "ares_time_to_first_token_seconds",
        "ares_time_per_output_token_seconds",
        "ares_request_seconds",
    ] {
        assert!(text.contains(&format!("# TYPE {name} histogram\n")), "{name}:\n{text}");
        assert!(text.contains(&format!("{name}_count 2\n")), "{name}:\n{text}");
    }
    assert!(text.contains("ares_request_queue_seconds_bucket{le=\"0.005\"} 2\n"), "{text}");
}

#[tokio::test]
async fn a_body_that_does_not_come_is_given_up() {
    let engine = instant();
    let config = ares_server::Config { timeout: Duration::from_millis(100), ..Default::default() };
    let app = app_with(&engine, config).await;
    let never = futures_util::stream::pending::<Result<axum::body::Bytes, std::io::Error>>();
    let request = axum::http::Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from_stream(never))
        .unwrap();
    let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
    let (status, body) = json(response).await;
    assert_eq!(status, StatusCode::REQUEST_TIMEOUT);
    assert_eq!(body["error"]["code"], "request_timeout");
    assert!(engine.submitted().is_empty());
}
