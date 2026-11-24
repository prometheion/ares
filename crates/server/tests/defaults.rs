//! The defaults of a model's requests, over HTTP on a scripted engine: what
//! the engine is handed for a request that leaves fields out, on each of the
//! four routes that generate, the limit of a reply's length, and what the
//! server says about both.

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{Engine, FinishReason, GenerateRequest, ModelCard, SamplerKind, Sampling};
use ares_server::defaults::{Defaults, LIMIT_HEADER};
use ares_server::models::{LoadFailure, Loader, Models, Site, SystemClock, Unit};
use ares_server::options::{self, ModelOptions};
use ares_server::preset::{Preset, Presets};
use ares_server::{Config, router_over};
use axum::Router;
use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";
const MESSAGES: &str = "/v1/messages";
const RESPONSES: &str = "/v1/responses";

/// The least body of a request on each route that generates.
fn routes() -> [(&'static str, Value); 4] {
    [
        (CHAT, json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]})),
        (COMPLETIONS, json!({"model": "mock", "prompt": "Once upon"})),
        (
            MESSAGES,
            json!({"model": "mock", "max_tokens": 64,
                   "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (RESPONSES, json!({"model": "mock", "input": "hi"})),
    ]
}

fn body(base: &Value, extra: Value) -> String {
    merged(base.clone(), extra)
}

/// The defaults that the options `given` set, as the command line reads them.
fn defaults(given: &[(&str, &str)]) -> Defaults {
    let mut model = ModelOptions::default();
    for (name, value) in given {
        let (flag, _) = options::lookup(name).unwrap_or_else(|| panic!("{name} is an option"));
        options::set(&mut model, flag, value).unwrap_or_else(|e| panic!("{e}"));
    }
    model.check().expect("the options hold together");
    model.defaults
}

/// A server over one engine whose model has the defaults of `given`.
fn served(given: &[(&str, &str)]) -> (Arc<MockEngine>, Router) {
    let engine = MockEngine::script(vec![text("ok"), done(FinishReason::Stop, 3, 1)]);
    let app = app_with(&engine, Config { defaults: defaults(given), ..Config::default() });
    (engine, app)
}

/// What the engine was handed for `body` on `path`.
async fn submitted(given: &[(&str, &str)], path: &str, body: &str) -> GenerateRequest {
    let (engine, app) = served(given);
    let response = post(&app, path, body).await;
    let (status, value) = json(response).await;
    assert_eq!(status, StatusCode::OK, "{path}: {value}");
    engine.submitted().remove(0)
}

/// A default for every field of the sampler that can have one.
const EVERY: &[(&str, &str)] = &[
    ("--temperature", "0.7"),
    ("--top-k", "20"),
    ("--top-p", "0.8"),
    ("--min-p", "0.05"),
    ("--typical-p", "0.9"),
    ("--top-n-sigma", "1.5"),
    ("--xtc-probability", "0.5"),
    ("--xtc-threshold", "0.2"),
    ("--dynatemp-range", "0.25"),
    ("--dynatemp-exponent", "1.5"),
    ("--repeat-penalty", "1.1"),
    ("--presence-penalty", "1.5"),
    ("--frequency-penalty", "0.5"),
    ("--repeat-last-n", "256"),
    ("--dry-multiplier", "0.8"),
    ("--dry-base", "2"),
    ("--dry-allowed-length", "3"),
    ("--dry-penalty-last-n", "512"),
    ("--dry-sequence-breakers", r#"["\n", "--"]"#),
    ("--mirostat", "2"),
    ("--mirostat-tau", "4.5"),
    ("--mirostat-eta", "0.2"),
    ("--samplers", "penalties,top_k,temperature"),
    ("--seed", "42"),
];

/// What [`EVERY`] sets.
fn every() -> Sampling {
    Sampling {
        temperature: Some(0.7),
        top_k: Some(20),
        top_p: Some(0.8),
        min_p: Some(0.05),
        typical_p: Some(0.9),
        top_n_sigma: Some(1.5),
        xtc_probability: Some(0.5),
        xtc_threshold: Some(0.2),
        dynatemp_range: Some(0.25),
        dynatemp_exponent: Some(1.5),
        repeat_penalty: Some(1.1),
        presence_penalty: Some(1.5),
        frequency_penalty: Some(0.5),
        repeat_last_n: Some(256),
        dry_multiplier: Some(0.8),
        dry_base: Some(2.0),
        dry_allowed_length: Some(3),
        dry_penalty_last_n: Some(512),
        dry_sequence_breakers: Some(vec!["\n".into(), "--".into()]),
        mirostat: Some(2),
        mirostat_tau: Some(4.5),
        mirostat_eta: Some(0.2),
        samplers: Some(vec![SamplerKind::Penalties, SamplerKind::TopK, SamplerKind::Temperature]),
        seed: Some(42),
        ..Sampling::default()
    }
}

#[tokio::test]
async fn a_request_that_leaves_the_fields_out_gets_the_model_s_on_every_route() {
    for (path, base) in routes() {
        let request = submitted(EVERY, path, &base.to_string()).await;
        assert_eq!(request.sampling, every(), "{path}");
        // Without defaults the fields stay open, and the engine's own hold.
        let request = submitted(&[], path, &base.to_string()).await;
        assert_eq!(request.sampling, Sampling::default(), "{path}");
    }
}

#[tokio::test]
async fn the_values_of_a_request_hold_over_the_model_s_on_every_route() {
    // The two routes of the chat API have a field for every setting.
    let own = json!({
        "temperature": 0.2, "top_k": 5, "top_p": 0.5, "min_p": 0.2, "typical_p": 0.6,
        "top_n_sigma": 3.0, "xtc_probability": 0.1, "xtc_threshold": 0.3,
        "dynatemp_range": 0.75, "dynatemp_exponent": 2.5, "repeat_penalty": 1.3,
        "presence_penalty": -1.0, "frequency_penalty": -0.5, "repeat_last_n": 32,
        "dry_multiplier": 0.4, "dry_base": 1.5, "dry_allowed_length": 5,
        "dry_penalty_last_n": 128, "dry_sequence_breakers": ["."], "mirostat": 1,
        "mirostat_tau": 6.0, "mirostat_eta": 0.3, "samplers": ["min_p", "temperature"],
        "seed": 7
    });
    let expected = Sampling {
        temperature: Some(0.2),
        top_k: Some(5),
        top_p: Some(0.5),
        min_p: Some(0.2),
        typical_p: Some(0.6),
        top_n_sigma: Some(3.0),
        xtc_probability: Some(0.1),
        xtc_threshold: Some(0.3),
        dynatemp_range: Some(0.75),
        dynatemp_exponent: Some(2.5),
        repeat_penalty: Some(1.3),
        presence_penalty: Some(-1.0),
        frequency_penalty: Some(-0.5),
        repeat_last_n: Some(32),
        dry_multiplier: Some(0.4),
        dry_base: Some(1.5),
        dry_allowed_length: Some(5),
        dry_penalty_last_n: Some(128),
        dry_sequence_breakers: Some(vec![".".into()]),
        mirostat: Some(1),
        mirostat_tau: Some(6.0),
        mirostat_eta: Some(0.3),
        samplers: Some(vec![SamplerKind::MinP, SamplerKind::Temperature]),
        seed: Some(7),
        ..Sampling::default()
    };
    let [chat, completions, messages, responses] = routes();
    for (path, base) in [chat, completions] {
        let request = submitted(EVERY, path, &body(&base, own.clone())).await;
        assert_eq!(request.sampling, expected, "{path}");
        // A value that switches a sampler off is a value, and holds too.
        let off = json!({"temperature": 0, "top_k": 0, "mirostat": 0, "repeat_last_n": 0});
        let request = submitted(EVERY, path, &body(&base, off)).await;
        let expected = Sampling {
            temperature: Some(0.0),
            top_k: Some(0),
            mirostat: Some(0),
            repeat_last_n: Some(0),
            ..every()
        };
        assert_eq!(request.sampling, expected, "{path}");
    }
    // The Messages API has three of the fields, the Responses API two; the
    // model's defaults hold for all the others.
    let own = json!({"temperature": 0.2, "top_p": 0.5, "top_k": 5});
    let request = submitted(EVERY, messages.0, &body(&messages.1, own)).await;
    let expected = Sampling { temperature: Some(0.2), top_p: Some(0.5), top_k: Some(5), ..every() };
    assert_eq!(request.sampling, expected);
    let own = json!({"temperature": 0.2, "top_p": 0.5});
    let request = submitted(EVERY, responses.0, &body(&responses.1, own)).await;
    assert_eq!(request.sampling, Sampling { temperature: Some(0.2), top_p: Some(0.5), ..every() });
    // One field of a request leaves the model's other defaults in place.
    let request = submitted(EVERY, CHAT, &body(&routes()[0].1, json!({"top_p": 1}))).await;
    assert_eq!(request.sampling, Sampling { top_p: Some(1.0), ..every() });
}

#[tokio::test]
async fn the_completions_of_one_prompt_draw_from_the_model_s_seed_and_the_ones_after_it() {
    let (engine, app) = served(&[("--seed", "42")]);
    let (status, _) = json(post(&app, CHAT, &body(&routes()[0].1, json!({"n": 3}))).await).await;
    assert_eq!(status, StatusCode::OK);
    let seeds: Vec<_> = engine.submitted().iter().map(|r| r.sampling.seed).collect();
    assert_eq!(seeds, [Some(42), Some(43), Some(44)]);
    // A request that asks for a draw of its own gets one, in llama.cpp's words.
    let sent = body(&routes()[0].1, json!({"seed": -1}));
    assert_eq!(submitted(&[("--seed", "42")], CHAT, &sent).await.sampling.seed, None);
}

const LENGTHS: &[(&str, &str)] = &[("--max-tokens", "100"), ("--max-tokens-limit", "200")];

/// The names a route has for the limit of tokens.
fn limit_names(path: &str) -> &'static [&'static str] {
    match path {
        CHAT => &["max_tokens", "max_completion_tokens", "n_predict"],
        COMPLETIONS => &["max_tokens", "n_predict"],
        MESSAGES => &["max_tokens"],
        _ => &["max_output_tokens"],
    }
}

/// Sends `body` and returns the limit the engine got and the header about a
/// lowered limit.
async fn limited(given: &[(&str, &str)], path: &str, body: &str) -> (Option<u32>, Option<String>) {
    let (engine, app) = served(given);
    let response = post(&app, path, body).await;
    let header = response.headers().get(LIMIT_HEADER).map(|v| v.to_str().unwrap().to_owned());
    assert_eq!(response.status(), StatusCode::OK, "{path} {body}: {}", body_text(response).await);
    (engine.submitted().remove(0).max_tokens, header)
}

#[tokio::test]
async fn a_request_that_names_no_limit_gets_the_model_s_on_every_route() {
    for (path, base) in routes() {
        // The Messages API requires the member, so it never names none.
        if path != MESSAGES {
            assert_eq!(limited(LENGTHS, path, &base.to_string()).await, (Some(100), None));
            assert_eq!(limited(&[], path, &base.to_string()).await, (None, None), "{path}");
            // The limit alone is the default too.
            let only_limit = &LENGTHS[1..];
            assert_eq!(limited(only_limit, path, &base.to_string()).await, (Some(200), None));
        }
        for &name in limit_names(path) {
            // A limit the request names holds, below the default or above it.
            for own in [7, 150, 200] {
                let sent = body(&base, json!({ name: own }));
                assert_eq!(limited(LENGTHS, path, &sent).await, (Some(own), None), "{path} {name}");
            }
            // The default alone bounds nothing.
            let sent = body(&base, json!({ name: 5000 }));
            let got = limited(&LENGTHS[..1], path, &sent).await;
            assert_eq!(got, (Some(5000), None), "{path} {name}");
        }
    }
    // llama.cpp's -1 names no limit, as it does for llama-server.
    for (path, base) in &routes()[..2] {
        for name in ["max_tokens", "n_predict"] {
            let sent = body(base, json!({ name: -1 }));
            assert_eq!(limited(LENGTHS, path, &sent).await, (Some(100), None), "{path} {name}");
        }
    }
}

#[tokio::test]
async fn a_request_above_the_model_s_limit_gets_the_limit_and_is_told_so_on_every_route() {
    for (path, base) in routes() {
        for &name in limit_names(path) {
            for stream in [false, true] {
                let sent = body(&base, json!({ name: 201, "stream": stream }));
                let got = limited(LENGTHS, path, &sent).await;
                assert_eq!(got, (Some(200), Some("200".into())), "{path} {name} {stream}");
                let sent = body(&base, json!({ name: 4_000_000_000u32, "stream": stream }));
                let got = limited(&LENGTHS[1..], path, &sent).await;
                assert_eq!(got, (Some(200), Some("200".into())), "{path} {name} {stream}");
            }
        }
    }
}

/// What a reply that the limit cut says, on the routes' own terms.
#[tokio::test]
async fn a_reply_the_limit_cut_says_that_it_was_cut_and_how_long_it_is() {
    let cut = || MockEngine::script(vec![text("so far"), done(FinishReason::Length, 9, 200)]);
    let config = || Config { defaults: defaults(LENGTHS), ..Config::default() };
    let [chat, completions, messages, responses] = routes();

    for (path, base) in [chat, completions] {
        let (_, answer) = json(
            post(&app_with(&cut(), config()), path, &body(&base, json!({"max_tokens": 900}))).await,
        )
        .await;
        assert_eq!(answer["choices"][0]["finish_reason"], "length", "{path}");
        assert_eq!(answer["usage"]["completion_tokens"], 200, "{path}");
    }
    let sent = body(&messages.1, json!({"max_tokens": 900}));
    let (_, answer) = json(post(&app_with(&cut(), config()), MESSAGES, &sent).await).await;
    assert_eq!(answer["stop_reason"], "max_tokens");
    assert_eq!(answer["usage"]["output_tokens"], 200);

    // The response object of the Responses API says the limit in effect.
    let sent = body(&responses.1, json!({"max_output_tokens": 900}));
    let (_, answer) = json(post(&app_with(&cut(), config()), RESPONSES, &sent).await).await;
    assert_eq!(answer["status"], "incomplete");
    assert_eq!(answer["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(answer["max_output_tokens"], 200);
    assert_eq!(answer["usage"]["output_tokens"], 200);
}

#[tokio::test]
async fn a_response_of_the_responses_api_says_what_was_applied() {
    let given = [("--temperature", "0.7"), ("--top-p", "0.95"), ("--max-tokens", "100")];
    let base = &routes()[3].1;
    let (_, app) = served(&given);
    // The model's defaults where the request says nothing.
    let (_, answer) = json(post(&app, RESPONSES, &base.to_string()).await).await;
    assert_eq!(
        (&answer["temperature"], &answer["top_p"], &answer["max_output_tokens"]),
        (&json!(0.7), &json!(0.95), &json!(100))
    );
    // The request's own where it says something.
    let own = json!({"temperature": 0.25, "top_p": 0.5, "max_output_tokens": 30});
    let (_, answer) = json(post(&app, RESPONSES, &body(base, own)).await).await;
    assert_eq!(
        (&answer["temperature"], &answer["top_p"], &answer["max_output_tokens"]),
        (&json!(0.25), &json!(0.5), &json!(30))
    );
    // Null where neither says anything: the engine's own value.
    let (_, app) = served(&[]);
    let (_, answer) = json(post(&app, RESPONSES, &base.to_string()).await).await;
    assert!(answer["temperature"].is_null() && answer["top_p"].is_null(), "{answer}");
    assert!(answer["max_output_tokens"].is_null(), "{answer}");
}

#[tokio::test]
async fn a_count_of_tokens_takes_no_harm_from_the_defaults() {
    // The routes that count read a request as the routes that generate do.
    let (_, app) = served(EVERY);
    let sent = routes()[0].1.to_string();
    let (status, answer) = json(post(&app, "/v1/chat/completions/input_tokens", &sent).await).await;
    // The scripted engine cannot count; the request got as far as the engine.
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{answer}");
}

#[tokio::test]
async fn the_model_list_says_what_a_field_means_that_a_request_leaves_out() {
    let (_, app) = served(&[
        ("--temperature", "1.0"),
        ("--top-p", "0.95"),
        ("--top-k", "20"),
        ("--max-tokens", "100"),
        ("--max-tokens-limit", "200"),
    ]);
    let (status, list) = json(get(&app, "/v1/models").await).await;
    assert_eq!(status, StatusCode::OK);
    for model in list["data"].as_array().unwrap() {
        assert_eq!(
            model["defaults"],
            json!({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 100})
        );
        assert_eq!(model["max_tokens_limit"], 200);
    }
    // A server without defaults says that too: the engine's own hold.
    let (_, app) = served(&[]);
    let (_, list) = json(get(&app, "/v1/models").await).await;
    assert_eq!(list["data"][0]["defaults"], json!({}));
    assert!(list["data"][0].get("max_tokens_limit").is_none());
}

// --- several models, each with its own ---------------------------------------

/// The engines a test's loader made, by the ids of their models.
type Engines = Vec<(String, Arc<MockEngine>)>;

/// Loads a scripted engine for every preset, and keeps them for the test.
#[derive(Default, Clone)]
struct Scripted(Arc<Mutex<Engines>>);

struct Loaded(Arc<MockEngine>);

impl Unit for Loaded {
    fn engine(&self) -> Arc<dyn Engine> {
        self.0.clone()
    }

    fn ram(&self) -> u64 {
        0
    }

    fn unload(self: Box<Self>) {}
}

impl Loader for Scripted {
    fn load(&self, preset: &Preset, _: &Site) -> Result<Box<dyn Unit>, LoadFailure> {
        let card = ModelCard { id: preset.id.clone(), context_length: 4096, created: 1 };
        let refusal = ares_api::EngineError::Unsupported("no decisions here".into());
        let engine = MockEngine::with_models(
            vec![card],
            Plan::Script(vec![text("ok"), done(FinishReason::Stop, 3, 1)]),
            DecisionPlan::Reject(refusal),
        );
        self.0.lock().unwrap().push((preset.id.clone(), engine.clone()));
        Ok(Box::new(Loaded(engine)))
    }
}

impl Scripted {
    /// The last request the engine of `model` was handed.
    fn last(&self, model: &str) -> GenerateRequest {
        let engines = self.0.lock().unwrap();
        let (_, engine) = engines.iter().find(|(id, _)| id == model).expect("the model is loaded");
        engine.submitted().pop().expect("the engine got a request")
    }
}

const TWO: &str = "\
[*]
temperature = 1.0
top-p = 0.95
top-k = 20
max-tokens = 4096

[thinker]
model = /a.gguf
aliases = default
presence-penalty = 0.0
max-tokens-limit = 8192

[writer]
model = /b.gguf
temperature = 0.7
top-p = 0.8
presence-penalty = 1.5
max-tokens = 512
";

#[tokio::test]
async fn every_model_of_a_preset_file_has_its_own_defaults() {
    let presets = Presets::parse(TWO, Path::new("models.ini"), &[], &|_| true).unwrap();
    let loader = Scripted::default();
    let clock = Arc::new(SystemClock::default());
    let models = Arc::new(Models::start(presets, Arc::new(loader.clone()), clock));
    let app = router_over(models.clone(), Config::default());

    // The list has them before a model is loaded.
    let (_, list) = json(get(&app, "/v1/models").await).await;
    assert_eq!(list["data"][0]["status"]["value"], "unloaded");
    assert_eq!(
        list["data"][0]["defaults"],
        json!({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0,
               "max_tokens": 4096})
    );
    assert_eq!(list["data"][0]["max_tokens_limit"], 8192);
    assert_eq!(
        list["data"][1]["defaults"],
        json!({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
               "max_tokens": 512})
    );
    assert!(list["data"][1].get("max_tokens_limit").is_none());

    // A request gets the defaults of the model it names, by id or alias.
    let chat = |model: &str, extra: Value| {
        body(&json!({"model": model, "messages": [{"role": "user", "content": "hi"}]}), extra)
    };
    let (status, answer) = json(post(&app, CHAT, &chat("default", json!({}))).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let request = loader.last("thinker");
    let thinker = Sampling {
        temperature: Some(1.0),
        top_p: Some(0.95),
        top_k: Some(20),
        presence_penalty: Some(0.0),
        ..Sampling::default()
    };
    assert_eq!((request.sampling, request.max_tokens), (thinker, Some(4096)));

    let sent = chat("writer", json!({"temperature": 0.1, "max_tokens": 100_000}));
    let response = post(&app, CHAT, &sent).await;
    assert!(response.headers().get(LIMIT_HEADER).is_none(), "the writer has no limit");
    assert_eq!(response.status(), StatusCode::OK);
    let request = loader.last("writer");
    let writer = Sampling {
        temperature: Some(0.1),
        top_p: Some(0.8),
        top_k: Some(20),
        presence_penalty: Some(1.5),
        ..Sampling::default()
    };
    assert_eq!((request.sampling, request.max_tokens), (writer, Some(100_000)));

    // The thinker's limit holds on the other routes as on this one.
    let sent = json!({"model": "thinker", "input": "hi", "max_output_tokens": 100_000});
    let response = post(&app, RESPONSES, &sent.to_string()).await;
    assert_eq!(response.headers().get(LIMIT_HEADER).unwrap(), "8192");
    let (_, answer) = json(response).await;
    assert_eq!(answer["max_output_tokens"], 8192);
    assert_eq!(loader.last("thinker").max_tokens, Some(8192));

    tokio::time::timeout(Duration::from_secs(10), models.shutdown()).await.unwrap();
}
