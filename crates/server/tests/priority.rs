//! The priority of a request on every route that generates: the member
//! `priority` of the OpenAI routes, and the header `x-ares-priority`, which a
//! client that cannot add a member to a body sends (and the only way on
//! `/v1/messages`, whose API has no such member).

mod common;

use ares_api::FinishReason;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use common::*;
use serde_json::{Value, json};
use tower::ServiceExt;

const HEADER: &str = "x-ares-priority";

/// Posts `body` to `path` with the header of the priority if one is given,
/// and returns the status, the body and the priority the engine got.
async fn priority_of(path: &str, body: Value, header: Option<&str>) -> (StatusCode, Value, i32) {
    let engine = MockEngine::script(vec![done(FinishReason::Stop, 1, 0)]);
    let mut request = Request::post(path).header(header::CONTENT_TYPE, "application/json");
    if let Some(value) = header {
        request = request.header(HEADER, value);
    }
    let request = request.body(Body::from(body.to_string())).unwrap();
    let response = app(&engine).oneshot(request).await.unwrap();
    let (status, value) = json(response).await;
    let priority = engine.submitted().first().map_or(i32::MAX, |r| r.priority);
    (status, value, priority)
}

fn bodies() -> [(&'static str, Value); 4] {
    [
        (
            "/v1/chat/completions",
            json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]}),
        ),
        ("/v1/completions", json!({"model": "mock", "prompt": "Once upon"})),
        ("/v1/responses", json!({"model": "mock", "input": "hi"})),
        (
            "/v1/messages",
            json!({"model": "mock", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
        ),
    ]
}

#[tokio::test]
async fn a_request_without_a_priority_has_priority_0_and_the_header_gives_one() {
    for (path, body) in bodies() {
        let (status, value, priority) = priority_of(path, body.clone(), None).await;
        assert_eq!((status, priority), (StatusCode::OK, 0), "{path}: {value}");
        let (status, value, priority) = priority_of(path, body.clone(), Some(" -7 ")).await;
        assert_eq!((status, priority), (StatusCode::OK, -7), "{path}: {value}");
        let (status, value, _) = priority_of(path, body, Some("high")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {value}");
        assert!(value.to_string().contains(HEADER), "{path}: {value}");
    }
}

#[tokio::test]
async fn the_member_priority_is_read_on_the_openai_routes_and_refused_on_messages() {
    for (path, mut body) in bodies() {
        body["priority"] = json!(3);
        let (status, value, priority) = priority_of(path, body.clone(), None).await;
        if path == "/v1/messages" {
            assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
            assert!(value.to_string().contains("priority"), "{value}");
            continue;
        }
        assert_eq!((status, priority), (StatusCode::OK, 3), "{path}: {value}");
        // The same in the header is no conflict; another one is.
        let (status, _, priority) = priority_of(path, body.clone(), Some("3")).await;
        assert_eq!((status, priority), (StatusCode::OK, 3), "{path}");
        let (status, value, _) = priority_of(path, body.clone(), Some("1")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {value}");
        // A number outside the 32 bits of a priority, and a string, are refused.
        body["priority"] = json!(1u64 << 40);
        let (status, value, _) = priority_of(path, body.clone(), None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {value}");
        body["priority"] = json!("high");
        let (status, value, _) = priority_of(path, body, None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {value}");
    }
}
