//! How far a prompt has got, as llama.cpp's server streams it (`return_progress`
//! and `prompt_progress`), and the log of a request: through the real runtime with
//! the synthetic model on a host executor, whose prompts are read in steps of 64
//! rows.

mod common;

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::Engine;
use ares_runtime::testing::HostExecutor;
use ares_runtime::{Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use ares_server::options::ModelOptions;
use axum::Router;
use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";

async fn served() -> Router {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.eos = Some(u32::from(b'~'));
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let config = RuntimeConfig {
        pages: 4096,
        cells_per_page: 16,
        max_slots: 4,
        window_rows: 1,
        warm_up: false,
        ..RuntimeConfig::default()
    };
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_row_cost(Duration::from_micros(200), Duration::from_micros(5));
    let runtime = Runtime::start(model, Box::new(host), config).unwrap();
    let engine: Arc<dyn Engine> =
        Arc::new(runtime.with_chat_template(Some("{{ messages }}".into())));
    router_of(engine, ModelOptions::default(), ares_server::Config::default()).await
}

/// A prompt long enough to take several steps.
fn long_text() -> String {
    "the quick brown fox jumps over the lazy dog ".repeat(7)
}

async fn streamed(app: &Router, path: &str, body: Value) -> Vec<Value> {
    let response = post(app, path, &body.to_string()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let stream = parse_sse(&body_text(response).await);
    assert!(stream.done, "the stream ends with [DONE]");
    stream.chunks
}

/// The `prompt_progress` of the chunks that have one, and the place of the first
/// chunk with text.
fn progress_of(
    chunks: &[Value],
    text_of: impl Fn(&Value) -> Option<String>,
) -> (Vec<Value>, usize) {
    let progress: Vec<Value> =
        chunks.iter().filter_map(|chunk| chunk.get("prompt_progress").cloned()).collect();
    let first_text = chunks
        .iter()
        .position(|chunk| text_of(chunk).is_some_and(|text| !text.is_empty()))
        .expect("the reply has text");
    let last_progress = chunks.iter().rposition(|chunk| chunk.get("prompt_progress").is_some());
    assert!(last_progress.is_none_or(|at| at < first_text), "progress after the text: {chunks:?}");
    (progress, first_text)
}

/// What llama.cpp's server promises of its progress (its test
/// `test_return_progress`): the first says the cached tokens are done, `total` and
/// `cache` do not change, `processed` rises with every chunk, and the last says
/// all of them are done.
fn check_progress(progress: &[Value], cached: u64) {
    assert!(progress.len() >= 3, "a report per step: {progress:?}");
    let total = progress[0]["total"].as_u64().unwrap();
    assert!(total > 128, "{progress:?}");
    assert_eq!(progress[0]["cache"], cached, "{progress:?}");
    assert_eq!(progress[0]["processed"], cached, "{progress:?}");
    for pair in progress.windows(2) {
        assert_eq!(pair[1]["total"], total);
        assert_eq!(pair[1]["cache"], pair[0]["cache"]);
        assert!(pair[1]["processed"].as_u64() > pair[0]["processed"].as_u64(), "{progress:?}");
        assert!(pair[1]["time_ms"].as_u64() >= pair[0]["time_ms"].as_u64(), "{progress:?}");
    }
    assert_eq!(progress.last().unwrap()["processed"], total, "{progress:?}");
    for p in progress {
        let keys: Vec<&String> = p.as_object().unwrap().keys().collect();
        assert_eq!(keys.len(), 4, "llama.cpp's four fields: {p}");
    }
}

#[tokio::test]
async fn a_streamed_chat_says_how_far_its_prompt_has_got_before_its_text() {
    let app = served().await;
    let body = |progress: bool| {
        json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": long_text()}],
            "max_tokens": 6,
            "stream": true,
            "return_progress": progress,
        })
    };
    let content =
        |chunk: &Value| chunk["choices"][0]["delta"]["content"].as_str().map(str::to_owned);
    let chunks = streamed(&app, CHAT, body(true)).await;
    let (progress, _) = progress_of(&chunks, content);
    check_progress(&progress, 0);
    // A progress chunk is a chunk of the stream: the role, no text, and the
    // fields every chunk has.
    let chunk = chunks.iter().find(|c| c.get("prompt_progress").is_some()).unwrap();
    assert_eq!(chunk["object"], "chat.completion.chunk", "{chunk}");
    assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant", "{chunk}");
    assert_eq!(chunk["choices"][0]["finish_reason"], Value::Null, "{chunk}");
    // The same prompt again starts from the cache... of a model that has none
    // here, so it is computed again; without the member a stream has no progress.
    let chunks = streamed(&app, CHAT, body(false)).await;
    assert!(chunks.iter().all(|c| c.get("prompt_progress").is_none()), "{chunks:?}");
}

#[tokio::test]
async fn a_streamed_completion_says_how_far_its_prompt_has_got() {
    let app = served().await;
    let body = json!({
        "model": "synthetic",
        "prompt": long_text(),
        "max_tokens": 6,
        "stream": true,
        "return_progress": true,
    });
    let text = |chunk: &Value| chunk["choices"][0]["text"].as_str().map(str::to_owned);
    let chunks = streamed(&app, COMPLETIONS, body).await;
    let (progress, _) = progress_of(&chunks, text);
    check_progress(&progress, 0);
    let chunk = chunks.iter().find(|c| c.get("prompt_progress").is_some()).unwrap();
    assert_eq!(chunk["object"], "text_completion", "{chunk}");
    assert_eq!(chunk["choices"][0]["text"], "", "{chunk}");
}

#[tokio::test]
async fn a_streamed_response_says_how_far_its_prompt_has_got_in_its_in_progress_events() {
    let app = served().await;
    let body = json!({
        "model": "synthetic",
        "input": long_text(),
        "max_output_tokens": 6,
        "stream": true,
        "return_progress": true,
    });
    let response = post(&app, "/v1/responses", &body.to_string()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let events: Vec<Value> = sse_payloads(&body_text(response).await)
        .iter()
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let progress: Vec<Value> = events
        .iter()
        .filter_map(|event| {
            let progress = event.get("prompt_progress")?;
            assert_eq!(event["type"], "response.in_progress", "{event}");
            assert_eq!(event["response"]["status"], "in_progress", "{event}");
            Some(progress.clone())
        })
        .collect();
    check_progress(&progress, 0);
    let first_delta = events.iter().position(|e| e["type"] == "response.output_text.delta");
    let last_progress = events.iter().rposition(|e| e.get("prompt_progress").is_some());
    assert!(last_progress < first_delta, "{events:?}");
}

#[tokio::test]
async fn return_progress_without_a_stream_changes_nothing_and_must_be_a_boolean() {
    let app = served().await;
    let body = |progress: Value| {
        json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "return_progress": progress,
        })
    };
    let (status, answer) = json(post(&app, CHAT, &body(json!(true)).to_string()).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert!(answer.get("prompt_progress").is_none(), "{answer}");
    let (status, answer) = json(post(&app, CHAT, &body(json!("yes")).to_string()).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
    assert_eq!(answer["error"]["param"], "return_progress", "{answer}");
}

/// Where the lines of the log go in a test: a buffer.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn lines(&self) -> Vec<String> {
        String::from_utf8_lossy(&self.0.lock().unwrap()).lines().map(str::to_owned).collect()
    }
}

/// The log of this process at its default level, as the binary writes it. One
/// for all: a log of one thread misses lines while other tests run.
fn captured() -> &'static Captured {
    static LOG: std::sync::OnceLock<Captured> = std::sync::OnceLock::new();
    LOG.get_or_init(|| {
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("the one log of the tests");
        captured
    })
}

/// The default log of a request is for a person, as llama.cpp's server writes
/// its timings: the request as it is taken, its start in the engine with what the
/// cache gave, and at its end the prompt's rate over the rows the engine ran for
/// it, the rate of generation and the total; a refused request is a warning with
/// its route, status, code and the message the client got.
#[tokio::test]
async fn the_log_of_a_request_says_what_it_took_as_llama_cpp_does() {
    let captured = captured();
    let app = served().await;
    let body = json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": long_text()}],
        "max_tokens": 6,
        "temperature": 0,
    });
    let (status, answer) = json(post(&app, CHAT, &body.to_string()).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let prompt = answer["usage"]["prompt_tokens"].as_u64().unwrap();
    let id = answer["id"].as_str().unwrap().trim_start_matches("chatcmpl-").to_owned();
    let refused = json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "hello"}],
        "return_progress": "yes",
    });
    let (status, _) = json(post(&app, CHAT, &refused.to_string()).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let lines = captured.lines();
    let find = |what: &str| {
        lines
            .iter()
            .find(|line| line.contains(what))
            .unwrap_or_else(|| panic!("no line with '{what}' in {lines:#?}"))
            .clone()
    };
    let taken = find(&format!("request {id} taken: "));
    assert!(taken.contains(" INFO ") && taken.contains("chat, model synthetic"), "{taken}");
    assert!(taken.contains("max_tokens 6") && taken.contains("temperature 0"), "{taken}");
    let started = find(&format!("request {id} started after "));
    assert!(started.contains(&format!("prompt of {prompt} tokens, 0 from the cache")), "{started}");
    let finished = find(&format!("request {id} finished (length): prompt of {prompt} tokens"));
    assert!(finished.contains("6 generated"), "{finished}");
    let eval = find(&format!("request {id} prompt eval time = "));
    assert!(eval.contains(&format!("/ {prompt:5} tokens (")), "{eval}");
    assert!(eval.ends_with("tokens per second)"), "{eval}");
    let generated = find(&format!("request {id}        eval time = "));
    assert!(generated.contains("/     6 tokens ("), "{generated}");
    let total = find(&format!("request {id}       total time = "));
    assert!(total.ends_with(&format!("/ {:5} tokens", prompt + 6)), "{total}");
    // No key=value line of the request at the default level.
    let ours: Vec<&String> = lines.iter().filter(|line| line.contains(&id)).collect();
    assert!(ours.iter().all(|line| !line.contains("prompt_tokens_per_s")), "{ours:#?}");
    let refusal = find("POST /v1/chat/completions: 400 invalid_request");
    assert!(refusal.contains(" WARN ") && refusal.contains("return_progress"), "{refusal}");
}
