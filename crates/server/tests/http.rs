//! The HTTP behaviour of the router, driven in-process against a scripted
//! engine. No sockets are involved; `tests/e2e.rs` covers those.

mod common;

use ares_api::{
    EngineError, EngineStats, Event, ExpertStats, FinishReason, GenerateRequest, Prompt, Role,
    Usage,
};
use ares_server::Config;
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
            usage: Usage {
                prompt_tokens: 12,
                completion_tokens: 2,
                cached_prompt_tokens: 8,
                ..Usage::default()
            },
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
        "response_format": {"type": "text"}, "user": "u"
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

    let Prompt::Chat { messages, tools, .. } = request.prompt else {
        panic!("chat prompt expected")
    };
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
            usage: Usage {
                prompt_tokens: 7,
                completion_tokens: 1,
                cached_prompt_tokens: 4,
                ..Usage::default()
            },
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
        // What the server does not do is refused by name, never dropped.
        chat(json!({"logit_bias": {"1": 5}})).into_case(Some("logit_bias"), "logit_bias is not supported: the sampler has no logit bias"),
        chat(json!({"grammar": "root ::= \"a\""})).into_case(Some("grammar"), "structured output"),
        chat(json!({"json_schema": {"type": "object"}})).into_case(Some("json_schema"), "structured output"),
        chat(json!({"response_format": {"type": "json_object"}})).into_case(Some("response_format"), "structured output"),
        chat(json!({"logprobs": true})).into_case(Some("logprobs"), "token probabilities"),
        chat(json!({"top_logprobs": 5})).into_case(Some("top_logprobs"), "token probabilities"),
        chat(json!({"tool_choice": "required"})).into_case(Some("tool_choice"), "tool calls are not built"),
        chat(json!({"functions": []})).into_case(Some("functions"), "tool calls are not built"),
        // The engine of these tests declares no template, so it has no effort to set.
        chat(json!({"reasoning_effort": "high"})).into_case(Some("reasoning_effort"), "does not use it"),
        (COMPLETIONS, completion_body(json!({"chat_template_kwargs": {"enable_thinking": false}})), Some("chat_template_kwargs"), "no chat template"),
        chat(json!({"typical_p": 0.9})).into_case(Some("typical_p"), "typical_p is not supported: it is not a parameter of this server"),
        chat(json!({"mirostat": 2})).into_case(Some("mirostat"), "not a parameter of this server"),
        chat(json!({"something_new": [1, 2]})).into_case(Some("something_new"), "something_new is not supported"),
        (COMPLETIONS, completion_body(json!({"best_of": 3})), Some("best_of"), "one completion per request"),
        (COMPLETIONS, completion_body(json!({"echo": true})), Some("echo"), "returned on its own"),
        (COMPLETIONS, completion_body(json!({"suffix": "end"})), Some("suffix"), "returned on its own"),
        (COMPLETIONS, completion_body(json!({"messages": []})), Some("messages"), "messages is not supported"),
        (COMPLETIONS, completion_body(json!({"max_completion_tokens": 5})), Some("max_completion_tokens"), "not supported"),
        chat(json!({"prompt": "x"})).into_case(Some("prompt"), "prompt is not supported"),
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

#[tokio::test]
async fn a_field_that_asks_for_nothing_is_not_a_reason_to_refuse() {
    // `null`, the value that switches a feature off, and the bookkeeping
    // fields of the API: none of them changes what is generated.
    let quiet = json!({
        "logit_bias": {}, "logprobs": false, "top_logprobs": 0, "n": 1,
        "response_format": {"type": "text"}, "tool_choice": "auto", "parallel_tool_calls": true,
        "grammar": null, "json_schema": null, "typical_p": null, "something_new": null,
        "user": "u", "metadata": {"k": "v"}, "store": false, "service_tier": "auto",
    });
    let request = submitted_from(CHAT, &chat_body(quiet)).await;
    assert_eq!(request.sampling, Default::default());

    let quiet = json!({"best_of": 1, "echo": false, "suffix": null, "logprobs": null, "user": "u"});
    let request = submitted_from(COMPLETIONS, &completion_body(quiet)).await;
    assert_eq!(request.sampling, Default::default());

    // Penalties cross to the engine, which knows whether its sampler has them.
    let request =
        submitted_from(CHAT, &chat_body(json!({"presence_penalty": 0, "repeat_penalty": 1}))).await;
    assert_eq!(request.sampling.presence_penalty, Some(0.0));
    assert_eq!(request.sampling.repeat_penalty, Some(1.0));
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

// --- timings and statistics ----------------------------------------------------

#[tokio::test]
async fn a_finished_request_reports_where_its_time_went() {
    let (first, second) = (Gate::default(), Gate::default());
    let engine = MockEngine::script(vec![
        Step::Wait(first.clone()),
        text("a"),
        Step::Wait(second.clone()),
        text("b"),
        Step::Send(Event::Done {
            finish: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 12,
                completion_tokens: 5,
                cached_prompt_tokens: 4,
                ..Usage::default()
            },
        }),
    ]);
    let app = app(&engine);
    let request = tokio::spawn({
        let app = app.clone();
        async move { json(post(&app, CHAT, &chat_body(json!({}))).await).await }
    });
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;
    let pause = std::time::Duration::from_millis(30);
    tokio::time::sleep(pause).await;
    first.open();
    tokio::time::sleep(pause).await;
    second.open();
    let (status, body) = request.await.unwrap();
    assert_eq!(status, StatusCode::OK);

    // The prompt's time ends with the first text, the generation's starts there.
    let timings = &body["timings"];
    assert_eq!(timings["prompt_n"], 8, "the cached part of the prompt was not computed");
    assert_eq!(timings["predicted_n"], 5);
    let prompt_ms = timings["prompt_ms"].as_f64().unwrap();
    assert!((25.0..5000.0).contains(&prompt_ms), "the wait before the first text: {timings}");
    let predicted_ms = timings["predicted_ms"].as_f64().unwrap();
    assert!((25.0..5000.0).contains(&predicted_ms), "the wait after it: {timings}");
    let per_second = timings["prompt_per_second"].as_f64().unwrap();
    assert!((per_second - 8.0 * 1e3 / prompt_ms).abs() < 0.01 * per_second, "{timings}");
    // Four tokens came after the first, in `predicted_ms`.
    let per_second = timings["predicted_per_second"].as_f64().unwrap();
    assert!((per_second - 4.0 * 1e3 / predicted_ms).abs() < 0.01 * per_second, "{timings}");
    // This engine keeps no statistics, so nothing is said about drafts.
    assert!(timings.get("draft_n").is_none() && timings.get("draft_n_accepted").is_none());
}

#[tokio::test]
async fn a_stream_carries_the_timings_in_its_finish_chunk() {
    let engine = MockEngine::script(vec![text("Hi"), done(FinishReason::Length, 7, 1)]);
    let body = body_text(post(&app(&engine), CHAT, &chat_body(json!({"stream": true}))).await);
    let stream = parse_sse(&body.await);
    let finish = stream.chunks.last().unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(finish["timings"]["prompt_n"], 7);
    assert_eq!(finish["timings"]["predicted_n"], 1);
    assert_eq!(
        finish["timings"]["predicted_per_second"],
        Value::Null,
        "one token: none came after the first"
    );
    for chunk in &stream.chunks[..stream.chunks.len() - 1] {
        assert!(chunk.get("timings").is_none(), "{chunk}");
    }
}

fn drafted(proposed: u64, accepted: u64) -> EngineStats {
    EngineStats { drafts_proposed: proposed, drafts_accepted: accepted, ..EngineStats::default() }
}

#[tokio::test]
async fn a_request_that_ran_alone_gets_the_drafts_the_engine_counted_meanwhile() {
    let gate = Gate::default();
    let engine = MockEngine::script(vec![
        text("a"),
        Step::Wait(gate.clone()),
        done(FinishReason::Stop, 3, 9),
    ]);
    engine.set_stats(drafted(100, 60));
    let app = app(&engine);

    let ask = |app: axum::Router| {
        tokio::spawn(async move { json(post(&app, CHAT, &chat_body(json!({}))).await).await.1 })
    };
    let first = ask(app.clone());
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;
    engine.set_stats(drafted(112, 67));
    gate.open();
    let body = first.await.unwrap();
    assert_eq!(body["timings"]["draft_n"], 12, "{body}");
    assert_eq!(body["timings"]["draft_n_accepted"], 7);

    // The next one starts from the totals the first one ended with. The gate
    // is open, so it runs through at once.
    engine.set_stats(drafted(120, 70));
    let body = ask(app.clone()).await.unwrap();
    assert_eq!(body["timings"]["draft_n"], 8, "{body}");
    assert_eq!(body["timings"]["draft_n_accepted"], 3);
}

#[tokio::test]
async fn requests_that_ran_together_get_no_draft_counts() {
    let gate = Gate::default();
    let engine = MockEngine::script(vec![
        text("a"),
        Step::Wait(gate.clone()),
        done(FinishReason::Stop, 3, 9),
    ]);
    engine.set_stats(drafted(0, 0));
    let app = app(&engine);
    let ask = |app: axum::Router| {
        tokio::spawn(async move { json(post(&app, CHAT, &chat_body(json!({}))).await).await.1 })
    };
    let (first, second) = (ask(app.clone()), ask(app.clone()));
    eventually("both requests to reach the engine", || engine.submitted().len() == 2).await;
    engine.set_stats(drafted(40, 30));
    gate.open();
    for body in [first.await.unwrap(), second.await.unwrap()] {
        let timings = &body["timings"];
        assert_eq!(timings["predicted_n"], 9);
        assert!(
            timings.get("draft_n").is_none(),
            "the engine's count is not one request's: {body}"
        );
    }

    // Counting resumes: the first request alone after the pair leaves totals
    // behind (whether it has some to start from depends on how the pair
    // ended), and the one after it is counted.
    ask(app.clone()).await.unwrap();
    engine.set_stats(drafted(50, 31));
    let counted = ask(app.clone()).await.unwrap();
    assert_eq!(counted["timings"]["draft_n"], 10, "{counted}");
    assert_eq!(counted["timings"]["draft_n_accepted"], 1);
}

#[tokio::test]
async fn metrics_are_the_engines_statistics_as_text() {
    let engine = instant();
    let (status, body) = json(get(&app(&engine), "/metrics").await).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "an engine without statistics: {body}");

    engine.set_stats(EngineStats {
        steps: 5,
        prompt_tokens: 40,
        generated_tokens: 9,
        drafts_proposed: 6,
        drafts_accepted: 4,
        requests_finished: 2,
        requests_cancelled: 1,
        requests_refused: 0,
        prefix_hits: 3,
        cached_cells: 640,
        replayed_rows: 12,
        checkpoints_saved: 5,
        checkpoints_restored: 2,
        checkpoints_dropped: 1,
        page_copies: 4,
        sequences: 1,
        pages_free: 30,
        pages: 34,
        cached_pages: 7,
        checkpoint_ram: 157_286_400,
        experts: ExpertStats {
            picks: 480,
            hits: 470,
            from_ram: 6,
            from_disk: 4,
            cpu_experts: 2,
            uploads: 8,
            read_bytes: 5_000_000,
            wait_us: 12_000,
            ..ExpertStats::default()
        },
    });
    let response = get(&app(&engine), "/metrics").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers()[header::CONTENT_TYPE].to_str().unwrap().starts_with("text/plain"));
    let text = body_text(response).await;
    for line in [
        "ares_requests_finished_total 2",
        "ares_requests_cancelled_total 1",
        "ares_requests_refused_total 0",
        "ares_steps_total 5",
        "ares_prompt_tokens_total 40",
        "ares_generated_tokens_total 9",
        "ares_drafts_proposed_total 6",
        "ares_drafts_accepted_total 4",
        "ares_sequences 1",
        "ares_pages_free 30",
        "ares_pages 34",
        "ares_prefix_hits_total 3",
        "ares_cached_cells_total 640",
        "ares_replayed_rows_total 12",
        "ares_checkpoints_saved_total 5",
        "ares_checkpoints_restored_total 2",
        "ares_checkpoints_dropped_total 1",
        "ares_page_copies_total 4",
        "ares_cached_pages 7",
        "ares_checkpoint_ram_bytes 157286400",
        "ares_expert_picks_total 480",
        "ares_expert_hits_total 470",
        "ares_expert_misses_ram_total 6",
        "ares_expert_misses_disk_total 4",
        "ares_expert_cpu_total 2",
        "ares_expert_uploads_total 8",
        "ares_expert_read_bytes_total 5000000",
        "ares_expert_wait_microseconds_total 12000",
        "# TYPE ares_pages gauge",
        "# TYPE ares_steps_total counter",
    ] {
        assert!(text.lines().any(|l| l == line), "'{line}' is missing:\n{text}");
    }
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
