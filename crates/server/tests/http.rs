//! The HTTP behaviour of the router, driven in-process against a scripted
//! engine. No sockets are involved; `tests/e2e.rs` covers those.

mod common;

use ares_api::{EngineError, Event, FinishReason, GenerateRequest, Prompt, Role, Usage};
use ares_server::{Config, NoModelEngine};
use axum::http::{StatusCode, header};
use common::*;
use futures_util::StreamExt;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";

fn chat_body(extra: Value) -> String {
    let mut body = json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body.to_string()
}

fn completion_body(extra: Value) -> String {
    let mut body = json!({"model": "mock", "prompt": "Once upon"});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body.to_string()
}

/// A request that finishes at once, for tests that look at what the engine received.
fn instant() -> std::sync::Arc<MockEngine> {
    MockEngine::script(vec![done(FinishReason::Stop, 1, 0)])
}

async fn submitted_from(path: &str, body: &str) -> GenerateRequest {
    let engine = instant();
    let response = post(&app(&engine), path, body).await;
    let (status, value) = json(response).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    engine.submitted().remove(0)
}

// --- the small endpoints -----------------------------------------------------

#[tokio::test]
async fn health_answers_ok() {
    let engine = instant();
    let (status, body) = json(get(&app(&engine), "/health").await).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"status": "ok"}));
}

#[tokio::test]
async fn models_come_from_the_engine_in_the_openai_list_format() {
    let engine = instant();
    let (status, body) = json(get(&app(&engine), "/v1/models").await).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(
        body["data"][0],
        json!({"id": "mock", "object": "model", "created": 1_700_000_000u64, "owned_by": "ares", "context_length": 4096})
    );
    assert_eq!(body["data"][1]["id"], "mock-2");
}

#[tokio::test]
async fn unknown_paths_and_wrong_methods_answer_in_the_error_format() {
    let engine = instant();
    let app = app(&engine);
    let (status, body) = json(get(&app, "/v2/nothing").await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
    assert!(body["error"]["message"].as_str().unwrap().contains("/v2/nothing"));

    let (status, body) = json(get(&app, CHAT).await).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(body["error"]["code"], "method_not_allowed");
    let (status, _) = json(post(&app, "/health", "{}").await).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

// --- non-streaming -----------------------------------------------------------

#[tokio::test]
async fn chat_completion_returns_one_json_object() {
    let engine = MockEngine::script(vec![
        text("Hel"),
        text("lo"),
        Step::Send(Event::Done {
            finish: FinishReason::Stop,
            usage: Usage { prompt_tokens: 12, completion_tokens: 2, cached_prompt_tokens: 8 },
        }),
    ]);
    let (status, body) = json(post(&app(&engine), CHAT, &chat_body(json!({}))).await).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "mock");
    assert!(body["created"].as_u64().unwrap() > 1_600_000_000);
    assert_eq!(body["choices"].as_array().unwrap().len(), 1);
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["message"], json!({"role": "assistant", "content": "Hello"}));
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        body["usage"],
        json!({
            "prompt_tokens": 12,
            "completion_tokens": 2,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 8},
        })
    );
    // The response id carries the id the engine saw, to find a request in the logs.
    let request = engine.submitted().remove(0);
    assert_eq!(body["id"], format!("chatcmpl-{}", request.id));
}

#[tokio::test]
async fn completion_returns_text_and_the_length_finish_reason() {
    let engine = MockEngine::script(vec![text(" a time"), done(FinishReason::Length, 3, 5)]);
    let (status, body) =
        json(post(&app(&engine), COMPLETIONS, &completion_body(json!({"max_tokens": 5}))).await)
            .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], " a time");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["total_tokens"], 8);
    assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(engine.submitted()[0].prompt, Prompt::Text("Once upon".into()));
    assert_eq!(engine.submitted()[0].max_tokens, Some(5));
}

#[tokio::test]
async fn every_request_gets_its_own_id_even_when_they_run_together() {
    let engine = instant();
    let app = app(&engine);
    let calls: Vec<_> = (0..16)
        .map(|_| {
            let app = app.clone();
            tokio::spawn(async move { json(post(&app, CHAT, &chat_body(json!({}))).await).await })
        })
        .collect();
    let mut ids = std::collections::HashSet::new();
    for call in calls {
        let (status, body) = call.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert!(ids.insert(body["id"].as_str().unwrap().to_owned()));
    }
    let seen: std::collections::HashSet<_> = engine.submitted().iter().map(|r| r.id).collect();
    assert_eq!(seen.len(), 16);
}

// --- what the engine receives ------------------------------------------------

#[tokio::test]
async fn the_engine_receives_everything_the_client_sent_and_nothing_else() {
    let body = r#"{
        "model": "mock",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "name": "anna", "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"z\":1,\"a\":2}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            {"role": "developer", "content": "newer name for system"}
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"z": {"type": "integer"}, "a": {"type": "string"}}}}}],
        "temperature": 0.5, "top_p": 0.75, "top_k": 40, "min_p": 0.0625,
        "presence_penalty": 0.5, "frequency_penalty": -0.25, "repeat_penalty": 1.5, "seed": 42,
        "max_tokens": 100, "max_completion_tokens": 50,
        "stop": ["\n\n", "END"],
        "logit_bias": {"1": 5}, "response_format": {"type": "text"}, "user": "u", "something_new": [1, 2]
    }"#;
    let request = submitted_from(CHAT, body).await;

    assert_eq!(request.model, "mock");
    assert_eq!(request.max_tokens, Some(50), "max_completion_tokens replaces max_tokens");
    assert_eq!(request.stop, ["\n\n", "END"]);
    let sampling = &request.sampling;
    assert_eq!(sampling.temperature, Some(0.5));
    assert_eq!(sampling.top_p, Some(0.75));
    assert_eq!(sampling.top_k, Some(40));
    assert_eq!(sampling.min_p, Some(0.0625));
    assert_eq!(sampling.presence_penalty, Some(0.5));
    assert_eq!(sampling.frequency_penalty, Some(-0.25));
    assert_eq!(sampling.repeat_penalty, Some(1.5));
    assert_eq!(sampling.seed, Some(42));

    let Prompt::Chat { messages, tools } = request.prompt else { panic!("chat prompt expected") };
    let roles: Vec<_> = messages.iter().map(|m| m.role).collect();
    assert_eq!(roles, [Role::System, Role::User, Role::Assistant, Role::Tool, Role::System]);
    assert_eq!(messages[1].content, "Hello", "text parts are joined");
    assert_eq!(messages[1].name.as_deref(), Some("anna"));
    assert_eq!(messages[2].content, "", "null content is empty text");
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(messages[0].tool_calls, None);
    // Tool calls and tool definitions are not interpreted: same members, same order.
    assert_eq!(
        serde_json::to_string(&messages[2].tool_calls).unwrap(),
        r#"[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"z\":1,\"a\":2}"}}]"#
    );
    assert_eq!(
        serde_json::to_string(&tools).unwrap(),
        r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"z":{"type":"integer"},"a":{"type":"string"}}}}}]"#
    );
}

#[tokio::test]
async fn unset_means_unset_so_the_model_default_applies() {
    let request = submitted_from(CHAT, &chat_body(json!({}))).await;
    assert_eq!(request.sampling, Default::default());
    assert_eq!(request.max_tokens, None);
    assert!(request.stop.is_empty());
    let Prompt::Chat { tools, .. } = request.prompt else { panic!() };
    assert_eq!(tools, None);

    // `null` is how many clients write "not set".
    let nulls = json!({"temperature": null, "top_p": null, "max_tokens": null, "stop": null,
                       "seed": null, "tools": null, "stream": null, "n": null});
    let request = submitted_from(CHAT, &chat_body(nulls)).await;
    assert_eq!(request.sampling, Default::default());
    assert_eq!(request.max_tokens, None);
}

#[tokio::test]
async fn conventions_of_other_servers_are_understood() {
    let body = chat_body(json!({"stop": "###", "max_tokens": -1, "top_k": -1, "seed": -1, "n": 1}));
    let request = submitted_from(CHAT, &body).await;
    assert_eq!(request.stop, ["###"], "stop may be one string");
    assert_eq!(request.max_tokens, None, "-1 means no limit");
    assert_eq!(request.sampling.top_k, Some(0), "-1 switches top-k off");
    assert_eq!(request.sampling.seed, None, "-1 means a random seed");

    let request = submitted_from(CHAT, &chat_body(json!({"seed": 18446744073709551615u64}))).await;
    assert_eq!(request.sampling.seed, Some(u64::MAX));
    let request = submitted_from(CHAT, &chat_body(json!({"max_tokens": 100.0}))).await;
    assert_eq!(request.max_tokens, Some(100));

    // The plain completions endpoint takes `max_tokens`, and a one-element prompt list.
    let request = submitted_from(
        COMPLETIONS,
        &completion_body(json!({"prompt": ["only one"], "stop": ["x"]})),
    )
    .await;
    assert_eq!(request.prompt, Prompt::Text("only one".into()));
    assert_eq!(request.stop, ["x"]);
}

// --- streaming ---------------------------------------------------------------

#[tokio::test]
async fn chat_stream_starts_with_the_role_and_ends_with_the_finish_reason() {
    let engine =
        MockEngine::script(vec![text("Hel"), text(""), text("lo"), done(FinishReason::Stop, 3, 2)]);
    let response = post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await;
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers()[header::CONTENT_TYPE].to_str().unwrap().to_owned();
    assert!(content_type.starts_with("text/event-stream"), "{content_type}");
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");

    let stream = parse_sse(&body_text(response).await);
    assert!(stream.done, "the stream ends with [DONE]");
    let chunks = &stream.chunks;
    assert_eq!(chunks.len(), 4, "role, two pieces (the empty one is dropped), finish: {chunks:?}");

    assert_eq!(chunks[0]["choices"][0]["delta"], json!({"role": "assistant", "content": ""}));
    assert_eq!(chunks[1]["choices"][0]["delta"], json!({"content": "Hel"}));
    assert_eq!(chunks[2]["choices"][0]["delta"], json!({"content": "lo"}));
    assert_eq!(chunks[3]["choices"][0]["delta"], json!({}));
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
    for chunk in &chunks[..3] {
        assert_eq!(chunk["choices"][0]["finish_reason"], Value::Null);
    }
    for chunk in chunks {
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["id"], chunks[0]["id"], "one id for the whole stream");
        assert_eq!(chunk["model"], "mock");
        assert!(chunk.get("usage").is_none(), "usage was not asked for: {chunk}");
    }
}

#[tokio::test]
async fn usage_chunk_follows_the_last_chunk_when_asked_for() {
    let engine = MockEngine::script(vec![
        text("Hi"),
        Step::Send(Event::Done {
            finish: FinishReason::Length,
            usage: Usage { prompt_tokens: 7, completion_tokens: 1, cached_prompt_tokens: 4 },
        }),
    ]);
    let body = chat_body(json!({"stream": true, "stream_options": {"include_usage": true}}));
    let body = body_text(post(&app(&engine), CHAT, &body).await).await;

    let payloads = sse_payloads(&body);
    assert_eq!(payloads.last().unwrap(), "[DONE]");
    let stream = parse_sse(&body);
    let chunks = &stream.chunks;
    assert_eq!(chunks.len(), 4, "role, text, finish, usage");
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "length");
    assert!(chunks[2].get("usage").is_none());
    assert_eq!(chunks[3]["choices"], json!([]), "the usage chunk has no choices");
    assert_eq!(
        chunks[3]["usage"],
        json!({
            "prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 4},
        })
    );
}

#[tokio::test]
async fn include_usage_false_adds_nothing() {
    let engine = MockEngine::script(vec![text("x"), done(FinishReason::Stop, 1, 1)]);
    let body = chat_body(json!({"stream": true, "stream_options": {"include_usage": false}}));
    let stream = parse_sse(&body_text(post(&app(&engine), CHAT, &body).await).await);
    assert_eq!(stream.chunks.len(), 3);
    assert!(stream.chunks.iter().all(|c| c.get("usage").is_none()));
}

#[tokio::test]
async fn completion_stream_carries_text_without_a_role_chunk() {
    let engine =
        MockEngine::script(vec![text(" a"), text(" time"), done(FinishReason::Stop, 2, 2)]);
    let body = completion_body(json!({"stream": true, "stream_options": {"include_usage": true}}));
    let stream = parse_sse(&body_text(post(&app(&engine), COMPLETIONS, &body).await).await);

    assert!(stream.done);
    let chunks = &stream.chunks;
    assert_eq!(chunks.len(), 4, "two pieces, finish, usage: {chunks:?}");
    assert_eq!(chunks[0]["choices"][0]["text"], " a");
    assert_eq!(chunks[1]["choices"][0]["text"], " time");
    assert_eq!(chunks[2]["choices"][0]["text"], "");
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks[3]["usage"]["total_tokens"], 4);
    assert!(chunks.iter().all(|c| c["object"] == "text_completion"));
}

#[tokio::test]
async fn a_stream_without_text_still_has_role_finish_and_marker() {
    let engine = MockEngine::script(vec![done(FinishReason::Stop, 1, 0)]);
    let stream = parse_sse(
        &body_text(post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await).await,
    );
    assert!(stream.done);
    assert_eq!(stream.chunks.len(), 2);
    assert_eq!(stream.chunks[1]["choices"][0]["finish_reason"], "stop");
}

// --- UTF-8 -------------------------------------------------------------------

/// Pieces the way a detokenizer emits them: whole characters, of every width.
const PIECES: [&str; 6] = ["Grüße", " ", "🦀", " 日本語", " é", "\u{10FFFF}\n\"q\"\\"];

#[tokio::test]
async fn multibyte_text_passes_through_a_stream_intact() {
    let mut steps: Vec<_> = PIECES.iter().map(|p| text(p)).collect();
    steps.push(done(FinishReason::Stop, 1, 6));
    let engine = MockEngine::script(steps);
    let body =
        body_text(post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await).await;

    assert!(body.contains("🦀") && body.contains("日本語"), "sent as UTF-8, not escaped: {body}");
    let stream = parse_sse(&body);
    let joined: String =
        stream.chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(joined, PIECES.concat());
    // Every chunk carries a whole piece: no character was cut between events.
    let pieces: Vec<_> = stream.chunks[1..stream.chunks.len() - 1]
        .iter()
        .map(|c| c["choices"][0]["delta"]["content"].as_str().unwrap())
        .collect();
    assert_eq!(pieces, PIECES);
}

#[tokio::test]
async fn multibyte_text_passes_through_a_plain_response_intact() {
    let mut steps: Vec<_> = PIECES.iter().map(|p| text(p)).collect();
    steps.push(done(FinishReason::Stop, 1, 6));
    let engine = MockEngine::script(steps);
    let (_, body) = json(post(&app(&engine), CHAT, &chat_body(json!({}))).await).await;
    assert_eq!(body["choices"][0]["message"]["content"], PIECES.concat());
}

// --- errors ------------------------------------------------------------------

struct Case {
    error: EngineError,
    status: StatusCode,
    kind: &'static str,
    code: &'static str,
    /// Words the message must contain.
    mentions: &'static [&'static str],
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            error: EngineError::ModelNotFound("nope".into()),
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            code: "model_not_found",
            mentions: &["'nope'", "mock, mock-2"],
        },
        Case {
            error: EngineError::InvalidRequest("max_tokens is silly".into()),
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: "invalid_request",
            mentions: &["max_tokens is silly"],
        },
        Case {
            error: EngineError::ContextOverflow { prompt_tokens: 9000, limit: 4096 },
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: "context_length_exceeded",
            mentions: &["9000", "4096"],
        },
        Case {
            error: EngineError::Overloaded,
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "server_error",
            code: "overloaded",
            mentions: &["capacity"],
        },
        Case {
            error: EngineError::Unsupported("no such architecture".into()),
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "server_error",
            code: "not_implemented",
            mentions: &["no such architecture"],
        },
        Case {
            error: EngineError::Internal("kernel exploded".into()),
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            code: "internal_error",
            mentions: &["kernel exploded"],
        },
    ]
}

async fn assert_error(case: &Case, response: axum::http::Response<axum::body::Body>, how: &str) {
    let retry_after = response.headers().get(header::RETRY_AFTER).cloned();
    let (status, body) = json(response).await;
    assert_eq!(status, case.status, "{how}: {body}");
    let error = &body["error"];
    assert_eq!(error["type"], case.kind, "{how}");
    assert_eq!(error["code"], case.code, "{how}");
    let message = error["message"].as_str().unwrap();
    for word in case.mentions {
        assert!(message.contains(word), "{how}: '{message}' should mention {word}");
    }
    assert!(error.get("param").is_some(), "{how}: the OpenAI format always has a param");
    assert_eq!(body.as_object().unwrap().len(), 1, "{how}: nothing but the error object");
    let expected_retry = (case.status == StatusCode::SERVICE_UNAVAILABLE).then_some("1");
    assert_eq!(retry_after.as_ref().and_then(|v| v.to_str().ok()), expected_retry, "{how}");
}

#[tokio::test]
async fn an_error_returned_by_submit_maps_to_its_status() {
    for case in cases() {
        for (path, body) in
            [(CHAT, chat_body(json!({}))), (COMPLETIONS, completion_body(json!({})))]
        {
            let engine = MockEngine::reject(case.error.clone());
            assert_error(&case, post(&app(&engine), path, &body).await, "rejected at submit").await;
        }
    }
}

#[tokio::test]
async fn an_error_event_before_any_text_maps_to_its_status() {
    for case in cases() {
        let engine = MockEngine::script(vec![fail(case.error.clone())]);
        let response = post(&app(&engine), CHAT, &chat_body(json!({}))).await;
        assert_error(&case, response, "error event").await;
    }
}

#[tokio::test]
async fn a_streaming_request_that_fails_first_gets_the_status_and_not_a_200() {
    // The prompt overflows the context: the engine finds out only after
    // tokenizing, which is after the request was accepted.
    for case in cases() {
        let engine = MockEngine::script(vec![fail(case.error.clone())]);
        let body = chat_body(json!({"stream": true}));
        assert_error(&case, post(&app(&engine), CHAT, &body).await, "streaming, error first").await;
    }
}

#[tokio::test]
async fn an_error_after_text_ends_the_plain_response_with_the_error() {
    let engine =
        MockEngine::script(vec![text("partial"), fail(EngineError::Internal("boom".into()))]);
    let (status, body) = json(post(&app(&engine), CHAT, &chat_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["error"]["code"], "internal_error");
    assert!(!body.to_string().contains("partial"), "half an answer must not look like an answer");
}

#[tokio::test]
async fn an_error_after_text_ends_the_stream_with_an_error_object_and_no_done_marker() {
    let engine =
        MockEngine::script(vec![text("partial"), fail(EngineError::Internal("boom".into()))]);
    let response = post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await;
    assert_eq!(response.status(), StatusCode::OK, "the status was sent before the failure");
    let body = body_text(response).await;

    let stream = parse_sse(&body);
    assert!(!stream.done, "[DONE] would make a cut-off answer look complete: {body}");
    assert_eq!(stream.chunks.len(), 3, "role, text, error");
    assert_eq!(stream.chunks[1]["choices"][0]["delta"]["content"], "partial");
    assert_eq!(stream.chunks[2]["error"]["code"], "internal_error");
    assert!(stream.chunks[2]["error"]["message"].as_str().unwrap().contains("boom"));
}

#[tokio::test]
async fn a_request_the_engine_cancelled_under_a_waiting_client_is_an_error() {
    let engine = MockEngine::script(vec![text("cut"), done(FinishReason::Cancelled, 1, 1)]);
    let (status, body) = json(post(&app(&engine), CHAT, &chat_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "cancelled");

    let engine = MockEngine::script(vec![text("cut"), done(FinishReason::Cancelled, 1, 1)]);
    let body =
        body_text(post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await).await;
    let stream = parse_sse(&body);
    assert!(!stream.done);
    assert_eq!(stream.chunks.last().unwrap()["error"]["code"], "cancelled");
    assert!(engine.cancelled().is_empty(), "the engine ended it; there is nothing left to cancel");
}

#[tokio::test]
async fn an_engine_that_drops_the_sink_without_finishing_is_reported_not_waited_for() {
    // Nothing at all.
    let engine = MockEngine::script(vec![Step::Abandon]);
    let (status, body) = json(post(&app(&engine), CHAT, &chat_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body["error"]["message"].as_str().unwrap().contains("without finishing"));
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id], "the request is released");

    // After some text, while streaming.
    let engine = MockEngine::script(vec![text("a"), Step::Abandon]);
    let body =
        body_text(post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await).await;
    let stream = parse_sse(&body);
    assert!(!stream.done);
    assert_eq!(stream.chunks.last().unwrap()["error"]["code"], "internal_error");
}

// --- request validation ------------------------------------------------------

#[tokio::test]
async fn bad_requests_are_refused_before_the_engine_sees_them() {
    let chat = |extra: Value| (CHAT, chat_body(extra));
    let cases: Vec<(&str, String, Option<&str>, &str)> = vec![
        // (path, body, param, words the message must contain)
        (CHAT, "{not json".into(), None, "not valid JSON"),
        (CHAT, String::new(), None, "not valid JSON"),
        (CHAT, "[1, 2]".into(), None, "must be a JSON object"),
        (CHAT, r#"{"messages": [{"role": "user", "content": "x"}]}"#.into(), Some("model"), "model is required"),
        (CHAT, r#"{"model": "", "messages": [{"role": "user", "content": "x"}]}"#.into(), Some("model"), "model is required"),
        (CHAT, r#"{"model": 5, "messages": []}"#.into(), Some("model"), "model must be a string"),
        (CHAT, r#"{"model": "mock"}"#.into(), Some("messages"), "messages is required"),
        (CHAT, r#"{"model": "mock", "messages": "hi"}"#.into(), Some("messages"), "must be an array"),
        (CHAT, r#"{"model": "mock", "messages": []}"#.into(), Some("messages"), "must not be empty"),
        (CHAT, r#"{"model": "mock", "messages": ["hi"]}"#.into(), Some("messages[0]"), "must be an object"),
        (CHAT, r#"{"model": "mock", "messages": [{"content": "hi"}]}"#.into(), Some("messages[0].role"), "role"),
        (CHAT, r#"{"model": "mock", "messages": [{"role": "robot", "content": "hi"}]}"#.into(), Some("messages[0].role"), "unknown role 'robot'"),
        (CHAT, r#"{"model": "mock", "messages": [{"role": "user", "content": 5}]}"#.into(), Some("messages[0].content"), "string or an array"),
        (CHAT, r#"{"model": "mock", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}"#.into(), Some("messages[0].content[0]"), "image_url"),
        (CHAT, r#"{"model": "mock", "messages": [{"role": "user", "content": "x", "name": 5}]}"#.into(), Some("messages[0].name"), "must be a string"),
        (CHAT, r#"{"model": "mock", "messages": [{"role": "assistant", "content": "x", "tool_calls": {}}]}"#.into(), Some("messages[0].tool_calls"), "must be an array"),
        chat(json!({"tools": "none"})).into_case(Some("tools"), "must be an array"),
        chat(json!({"temperature": -1})).into_case(Some("temperature"), "temperature must be a finite number >= 0, got -1"),
        chat(json!({"temperature": "hot"})).into_case(Some("temperature"), "temperature must be a number, got a string"),
        chat(json!({"temperature": "NaN"})).into_case(Some("temperature"), "temperature must be a number"),
        (CHAT, chat_body(json!({})).replace("\"mock\"", "\"mock\", \"temperature\": 1e999"), None, "number out of range"),
        chat(json!({"top_p": 0})).into_case(Some("top_p"), "top_p must be greater than 0"),
        chat(json!({"top_p": 1.5})).into_case(Some("top_p"), "top_p"),
        chat(json!({"min_p": 2})).into_case(Some("min_p"), "min_p"),
        chat(json!({"presence_penalty": 3})).into_case(Some("presence_penalty"), "presence_penalty"),
        chat(json!({"frequency_penalty": -3})).into_case(Some("frequency_penalty"), "frequency_penalty"),
        chat(json!({"repeat_penalty": 0})).into_case(Some("repeat_penalty"), "repeat_penalty"),
        chat(json!({"top_k": "all"})).into_case(Some("top_k"), "top_k must be a whole number"),
        chat(json!({"seed": 1.5})).into_case(Some("seed"), "seed must be a whole number"),
        chat(json!({"max_tokens": 0})).into_case(Some("max_tokens"), "max_tokens must be at least 1"),
        chat(json!({"max_completion_tokens": 0})).into_case(Some("max_completion_tokens"), "at least 1"),
        chat(json!({"max_tokens": "ten"})).into_case(Some("max_tokens"), "whole number"),
        chat(json!({"max_tokens": 1.5})).into_case(Some("max_tokens"), "whole number"),
        chat(json!({"stop": 5})).into_case(Some("stop"), "string or an array"),
        chat(json!({"stop": ["ok", 5]})).into_case(Some("stop"), "must hold strings"),
        chat(json!({"stop": ""})).into_case(Some("stop"), "must not be empty"),
        chat(json!({"n": 2})).into_case(Some("n"), "n must be 1"),
        chat(json!({"stream": "yes"})).into_case(Some("stream"), "stream must be a boolean"),
        chat(json!({"stream_options": 5})).into_case(Some("stream_options"), "must be an object"),
        (COMPLETIONS, r#"{"model": "mock"}"#.into(), Some("prompt"), "prompt is required"),
        (COMPLETIONS, completion_body(json!({"prompt": 5})), Some("prompt"), "prompt must be a string"),
        (COMPLETIONS, completion_body(json!({"prompt": []})), Some("prompt"), "one string"),
        (COMPLETIONS, completion_body(json!({"prompt": ["a", "b"]})), Some("prompt"), "one string"),
        (COMPLETIONS, completion_body(json!({"prompt": [1, 2, 3]})), Some("prompt"), "token ids"),
        (COMPLETIONS, completion_body(json!({"temperature": -0.5})), Some("temperature"), "temperature"),
    ];

    let engine = instant();
    let app = app(&engine);
    for (path, body, param, words) in cases {
        let (status, value) = json(post(&app, path, &body).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {value}");
        let error = &value["error"];
        assert_eq!(error["type"], "invalid_request_error", "{body}");
        assert_eq!(error["code"], "invalid_request", "{body}");
        assert_eq!(error["param"].as_str(), param, "{body}: {value}");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains(words), "{body}: '{message}' should contain '{words}'");
    }
    assert!(engine.submitted().is_empty(), "no invalid request may reach the engine");
}

/// Lets the table above read `chat(..).into_case(..)` instead of repeating the path.
trait IntoCase {
    fn into_case(
        self,
        param: Option<&'static str>,
        words: &'static str,
    ) -> (&'static str, String, Option<&'static str>, &'static str);
}

impl IntoCase for (&'static str, String) {
    fn into_case(
        self,
        param: Option<&'static str>,
        words: &'static str,
    ) -> (&'static str, String, Option<&'static str>, &'static str) {
        (self.0, self.1, param, words)
    }
}

#[tokio::test]
async fn an_oversized_body_is_refused_with_413() {
    let engine = instant();
    let config = Config { max_body_bytes: 64, ..Config::default() };
    let app = app_with(&engine, config);
    let big = chat_body(json!({"padding": "x".repeat(200)}));

    // Without a declared length the body is counted as it arrives.
    let (status, body) = json(post(&app, CHAT, &big).await).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["error"]["code"], "request_too_large");
    assert!(body["error"]["message"].as_str().unwrap().contains("64"));

    // With one, it is refused before it is read.
    let mut request = post_request(CHAT, &big);
    request.headers_mut().insert(header::CONTENT_LENGTH, big.len().into());
    let response = tower::ServiceExt::oneshot(app.clone(), request).await.unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

    assert!(engine.submitted().is_empty());
    // A body that fits is fine.
    let small = json!({"model": "m", "messages": [{"role": "user", "content": "x"}]}).to_string();
    assert!(small.len() <= 64);
    assert_eq!(post(&app, CHAT, &small).await.status(), StatusCode::OK);
}

// --- no model ----------------------------------------------------------------

#[tokio::test]
async fn the_engine_without_models_lists_none_and_refuses_with_501() {
    let app = ares_server::router(std::sync::Arc::new(NoModelEngine::new()));
    let (status, body) = json(get(&app, "/v1/models").await).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"object": "list", "data": []}));
    assert_eq!(json(get(&app, "/health").await).await.0, StatusCode::OK);

    for (path, body) in
        [(CHAT, chat_body(json!({}))), (COMPLETIONS, completion_body(json!({"stream": true})))]
    {
        let (status, value) = json(post(&app, path, &body).await).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(value["error"]["code"], "not_implemented");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("no model architecture is implemented yet"), "{message}");
    }

    let named = NoModelEngine::new().with_architecture(Some("qwen3".into()));
    let app = ares_server::router(std::sync::Arc::new(named));
    let (_, value) = json(post(&app, CHAT, &chat_body(json!({}))).await).await;
    assert!(value["error"]["message"].as_str().unwrap().contains("'qwen3'"));
}

// --- cancellation ------------------------------------------------------------

#[tokio::test]
async fn a_client_that_leaves_a_plain_request_cancels_it_in_the_engine() {
    let engine = MockEngine::script(vec![Step::UntilCancelled]);
    let app = app(&engine);
    let call = tokio::spawn({
        let app = app.clone();
        async move { tower::ServiceExt::oneshot(app, post_request(CHAT, &chat_body(json!({})))).await }
    });
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;
    assert!(engine.cancelled().is_empty(), "nothing to cancel while the client waits");

    // What hyper does when the connection closes: it drops the handler's future.
    call.abort();

    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);
}

#[tokio::test]
async fn a_client_that_leaves_a_stream_cancels_it_in_the_engine() {
    let engine = MockEngine::script(vec![text("a"), Step::UntilCancelled]);
    let response = post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await;
    assert_eq!(response.status(), StatusCode::OK);

    let mut body = response.into_body().into_data_stream();
    let role = body.next().await.unwrap().unwrap();
    let piece = body.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&role).contains("assistant"));
    assert!(String::from_utf8_lossy(&piece).contains("\"a\""));
    assert!(engine.cancelled().is_empty(), "the client is still reading");

    drop(body);

    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);
}

#[tokio::test]
async fn a_client_that_leaves_before_the_first_event_of_a_stream_cancels_too() {
    let engine = MockEngine::script(vec![Step::UntilCancelled]);
    let app = app(&engine);
    let call = tokio::spawn({
        let app = app.clone();
        async move {
            tower::ServiceExt::oneshot(app, post_request(CHAT, &chat_body(json!({"stream": true}))))
                .await
        }
    });
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;
    call.abort();
    eventually("the cancel", || !engine.cancelled().is_empty()).await;
}

#[tokio::test]
async fn finished_requests_are_never_cancelled() {
    let script = vec![text("done"), done(FinishReason::Stop, 1, 1)];
    let engine = MockEngine::script(script.clone());
    let app = app(&engine);
    assert_eq!(post(&app, CHAT, &chat_body(json!({}))).await.status(), StatusCode::OK);
    let _ = body_text(post(&app, CHAT, &chat_body(json!({"stream": true}))).await).await;
    let _ =
        body_text(post(&app, COMPLETIONS, &completion_body(json!({"stream": true}))).await).await;
    // A failed request is over as well.
    let failing = MockEngine::script(vec![fail(EngineError::Overloaded)]);
    let _ = post(&common::app(&failing), CHAT, &chat_body(json!({}))).await;

    assert_eq!(engine.submitted().len(), 3);
    assert!(engine.cancelled().is_empty(), "{:?}", engine.cancelled());
    assert!(failing.cancelled().is_empty());
}

#[tokio::test]
async fn an_invalid_request_is_not_cancelled_because_it_was_never_started() {
    let engine = instant();
    let _ = post(&app(&engine), CHAT, "{").await;
    assert!(engine.submitted().is_empty() && engine.cancelled().is_empty());
}

// --- a full channel ----------------------------------------------------------

fn tiny_buffer() -> Config {
    Config { event_buffer: 4, ..Config::default() }
}

/// `t0`, then (after the gate) `t1` ... `t{n}`.
fn numbered(gate: &Gate, n: usize) -> Vec<Step> {
    let mut steps = vec![text("t0"), Step::Wait(gate.clone())];
    steps.extend((1..=n).map(|i| text(&format!("t{i}"))));
    steps
}

fn delta_texts(stream: &Stream) -> Vec<String> {
    stream
        .chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn a_client_that_does_not_read_is_cut_off_with_an_error_and_loses_no_text_before_it() {
    let gate = Gate::default();
    let mut steps = numbered(&gate, 40);
    steps.push(done(FinishReason::Stop, 1, 41));
    let engine = MockEngine::script(steps);
    let response =
        post(&app_with(&engine, tiny_buffer()), CHAT, &chat_body(json!({"stream": true}))).await;
    assert_eq!(response.status(), StatusCode::OK);

    // The client holds the response and reads nothing while the engine writes.
    gate.open();
    eventually("the engine to be refused", || engine.sends().contains(&false)).await;

    // The engine never waited: it was told `false` at the first event that did
    // not fit (t0 was taken by the handler, t1..t4 fill the 4-event buffer,
    // t5 does not fit) and stopped there.
    assert_eq!(engine.sends(), [true, true, true, true, true, false]);

    let body = body_text(response).await;
    let stream = parse_sse(&body);
    assert_eq!(
        delta_texts(&stream),
        ["t0", "t1", "t2", "t3", "t4"],
        "everything that was accepted arrives, in order"
    );
    assert!(!stream.done, "a cut-off stream does not end with [DONE]");
    let last = stream.chunks.last().unwrap();
    assert_eq!(last["error"]["code"], "slow_client");
    assert!(last["error"]["message"].as_str().unwrap().contains("not read fast enough"));
    assert_eq!(
        engine.cancelled(),
        [engine.submitted()[0].id],
        "the request is cancelled in the engine"
    );
}

#[tokio::test]
async fn the_terminal_event_always_finds_room_so_a_finished_answer_is_never_a_slow_client() {
    let gate = Gate::default();
    // t0 goes to the handler; t1..t4 fill the buffer; Done takes the reserved slot.
    let mut steps = vec![text("t0"), Step::Wait(gate.clone())];
    steps.extend((1..=4).map(|i| text(&format!("t{i}"))));
    steps.push(done(FinishReason::Stop, 1, 5));
    let engine = MockEngine::script(steps);
    let response =
        post(&app_with(&engine, tiny_buffer()), CHAT, &chat_body(json!({"stream": true}))).await;

    gate.open();
    eventually("all events to be sent", || engine.sends().len() == 6).await;
    assert!(engine.sends().iter().all(|sent| *sent), "{:?}", engine.sends());

    let stream = parse_sse(&body_text(response).await);
    assert!(stream.done);
    assert_eq!(delta_texts(&stream), ["t0", "t1", "t2", "t3", "t4"]);
    assert_eq!(stream.chunks.last().unwrap()["choices"][0]["finish_reason"], "stop");
    assert!(engine.cancelled().is_empty());
}

#[tokio::test]
async fn the_default_buffer_takes_a_long_burst_without_cutting_anyone_off() {
    // 1000 events pushed in a tight loop, faster than any engine steps: the
    // default buffer (1024) is sized so that such a burst is not a slow client.
    let mut steps: Vec<_> = (0..1000).map(|i| text(&format!("{i},"))).collect();
    steps.push(done(FinishReason::Stop, 1, 1000));
    let engine = MockEngine::script(steps);
    let (status, body) = json(post(&app(&engine), CHAT, &chat_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::OK);
    let expected: String = (0..1000).map(|i| format!("{i},")).collect();
    assert_eq!(body["choices"][0]["message"]["content"], expected);
}
