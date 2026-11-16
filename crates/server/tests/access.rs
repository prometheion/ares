//! Who may use the server: API keys, the names and the web pages a server on
//! a local address answers, and CORS. Driven in-process over HTTP against a
//! scripted engine.

mod common;

use ares_api::FinishReason;
use ares_server::{Access, Config};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use common::*;
use serde_json::Value;
use tower::ServiceExt;

const CHAT_BODY: &str = r#"{"model": "mock", "messages": [{"role": "user", "content": "hi"}]}"#;
const MESSAGES_BODY: &str =
    r#"{"model": "mock", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}"#;

fn served(access: Access) -> Router {
    let engine = MockEngine::script(vec![text("ok"), done(FinishReason::Stop, 1, 1)]);
    app_with(&engine, Config { access, ..Config::default() })
}

fn locked() -> Access {
    Access { keys: vec!["first-key".into(), "second-key".into()], ..Access::default() }
}

/// A request with `headers`; a body makes it a `POST`.
async fn send(
    app: &Router,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> (StatusCode, axum::http::HeaderMap, String) {
    let mut request = Request::builder().method(method).uri(path);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let body = match body {
        Some(text) => {
            request = request.header(header::CONTENT_TYPE, "application/json");
            Body::from(text.to_owned())
        }
        None => Body::empty(),
    };
    let response = app.clone().oneshot(request.body(body).unwrap()).await.unwrap();
    let (status, headers) = (response.status(), response.headers().clone());
    (status, headers, body_text(response).await)
}

fn parsed(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON ({e}): {body}"))
}

#[tokio::test]
async fn with_keys_every_route_but_health_wants_one() {
    let app = served(locked());
    let (status, _, _) = send(&app, "GET", "/health", &[], None).await;
    assert_eq!(status, StatusCode::OK, "health stays open");

    for (path, body) in
        [("/v1/models", None), ("/metrics", None), ("/v1/chat/completions", Some(CHAT_BODY))]
    {
        let method = if body.is_some() { "POST" } else { "GET" };
        let (status, headers, text) = send(&app, method, path, &[], body).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(headers[header::WWW_AUTHENTICATE], "Bearer");
        let error = parsed(&text);
        assert_eq!(error["error"]["code"], "invalid_api_key", "{path}");
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }

    // Either key, in either header; the scheme's case does not matter.
    for headers in [
        [("authorization", "Bearer first-key")],
        [("authorization", "bearer second-key")],
        [("x-api-key", "second-key")],
    ] {
        let (status, _, text) =
            send(&app, "POST", "/v1/chat/completions", &headers, Some(CHAT_BODY)).await;
        assert_eq!(status, StatusCode::OK, "{headers:?}: {text}");
        assert_eq!(parsed(&text)["choices"][0]["message"]["content"], "ok");
    }
    // A key that is almost one, a key in the wrong place, another scheme.
    for headers in [
        [("authorization", "Bearer first-ke")],
        [("authorization", "Bearer first-key2")],
        [("authorization", "first-key")],
        [("authorization", "Basic first-key")],
        [("x-api-key", "")],
    ] {
        let (status, _, _) = send(&app, "GET", "/v1/models", &headers, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{headers:?}");
    }
}

#[tokio::test]
async fn the_messages_routes_refuse_in_anthropics_shape_and_take_its_header() {
    let app = served(locked());
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let (status, _, text) = send(&app, "POST", path, &[], Some(MESSAGES_BODY)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let error = parsed(&text);
        assert_eq!(error["type"], "error", "{text}");
        assert_eq!(error["error"]["type"], "authentication_error");
    }
    let headers = [("x-api-key", "first-key"), ("anthropic-version", "2023-06-01")];
    let (status, _, text) = send(&app, "POST", "/v1/messages", &headers, Some(MESSAGES_BODY)).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    assert_eq!(parsed(&text)["content"][0]["text"], "ok");
}

#[tokio::test]
async fn a_local_server_answers_only_to_its_local_names() {
    let app = served(Access { hosts: Some(Vec::new()), ..Access::default() });
    for host in ["localhost:8000", "LocalHost", "127.0.0.1:8000", "[::1]:8000", "app.localhost"] {
        let (status, _, _) = send(&app, "GET", "/v1/models", &[("host", host)], None).await;
        assert_eq!(status, StatusCode::OK, "{host}");
    }
    // A name a web page rebound to the server's address.
    for host in ["attacker.example:8000", "localhost.attacker.example"] {
        let (status, _, text) = send(&app, "GET", "/v1/models", &[("host", host)], None).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{host}");
        let error = parsed(&text);
        assert_eq!(error["error"]["code"], "forbidden_host");
        assert!(error["error"]["message"].as_str().unwrap().contains("--allowed-hosts"));
    }
    // A request without a name does not come from a browser.
    let (status, _, _) = send(&app, "GET", "/v1/models", &[], None).await;
    assert_eq!(status, StatusCode::OK);

    // Names the operator adds, in any case; health is checked too.
    let app = served(Access { hosts: Some(vec!["gpu.lan".into()]), ..Access::default() });
    let (status, _, _) = send(&app, "GET", "/v1/models", &[("host", "GPU.lan:8000")], None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = send(&app, "GET", "/health", &[("host", "other.lan")], None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    // `None` is every name.
    let (status, _, _) =
        send(&served(Access::default()), "GET", "/health", &[("host", "other.lan")], None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn without_keys_a_page_of_another_site_is_refused() {
    let app = served(Access { hosts: Some(Vec::new()), ..Access::default() });
    let foreign = [("host", "127.0.0.1:8000"), ("origin", "https://attacker.example")];
    let (status, _, text) =
        send(&app, "POST", "/v1/chat/completions", &foreign, Some(CHAT_BODY)).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(parsed(&text)["error"]["code"], "forbidden_origin");
    let (status, _, _) = send(&app, "POST", "/v1/messages", &foreign, Some(MESSAGES_BODY)).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let null = [("host", "127.0.0.1:8000"), ("origin", "null")];
    let (status, _, _) = send(&app, "GET", "/v1/models", &null, None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    // Local pages and pages of the server's own name pass.
    for origin in ["http://localhost:3000", "http://127.0.0.1:5173", "http://[::1]:8080"] {
        let headers = [("host", "127.0.0.1:8000"), ("origin", origin)];
        let (status, headers, _) = send(&app, "GET", "/v1/models", &headers, None).await;
        assert_eq!(status, StatusCode::OK, "{origin}");
        assert!(!headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN), "no CORS asked");
    }
    let open = served(Access::default());
    let own = [("host", "gpu.lan:8000"), ("origin", "http://gpu.lan:8000")];
    let (status, _, _) = send(&open, "GET", "/v1/models", &own, None).await;
    assert_eq!(status, StatusCode::OK);

    // With keys the key decides.
    let app = served(locked());
    let keyed = [("origin", "https://attacker.example"), ("authorization", "Bearer first-key")];
    let (status, _, _) = send(&app, "GET", "/v1/models", &keyed, None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn cors_lets_the_listed_origins_read_answers_and_answers_their_preflight() {
    let access = Access { origins: vec!["https://chat.example.org".into()], ..locked() };
    let app = served(access);
    let preflight = [
        ("origin", "https://chat.example.org"),
        ("access-control-request-method", "POST"),
        ("access-control-request-headers", "authorization, content-type"),
    ];
    // The preflight carries no key, and is answered.
    let (status, headers, _) =
        send(&app, "OPTIONS", "/v1/chat/completions", &preflight, None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], "https://chat.example.org");
    assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_METHODS], "GET, POST");
    assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_HEADERS], "authorization, content-type");
    assert_eq!(headers[header::VARY], "Origin");

    // An answer and an error both carry the header, so the page can read either.
    let keyed = [("origin", "https://chat.example.org"), ("authorization", "Bearer first-key")];
    let (status, headers, _) =
        send(&app, "POST", "/v1/chat/completions", &keyed, Some(CHAT_BODY)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], "https://chat.example.org");
    assert!(
        headers[header::ACCESS_CONTROL_EXPOSE_HEADERS].to_str().unwrap().contains("retry-after")
    );
    let (status, headers, _) =
        send(&app, "GET", "/v1/models", &[("origin", "https://chat.example.org")], None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], "https://chat.example.org");

    // Another origin gets no preflight and no header.
    let other = [("origin", "http://other:3000"), ("access-control-request-method", "POST")];
    let (status, headers, _) = send(&app, "OPTIONS", "/v1/chat/completions", &other, None).await;
    assert_ne!(status, StatusCode::NO_CONTENT);
    assert!(!headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));

    // `*` is every origin, and lifts the check of pages for a server without keys.
    let app =
        served(Access { origins: vec!["*".into()], hosts: Some(Vec::new()), ..Access::default() });
    let headers = [("host", "localhost"), ("origin", "https://anywhere.example")];
    let (status, headers, _) = send(&app, "GET", "/v1/models", &headers, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");
}
