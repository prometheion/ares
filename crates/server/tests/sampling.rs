//! What a request can say about sampling: the fields of the OpenAI API and of
//! llama.cpp's server, under their names, what reaches the engine, what is
//! refused and how the probabilities are written in the answer.

mod common;

use ares_api::{
    Alternative, BiasKey, Event, FinishReason, LogprobRequest, SamplerKind, TokenProbs,
};
use axum::http::StatusCode;
use common::*;
use futures_util::StreamExt;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";

/// The refusal of a request: its status and the field it names.
async fn refused(path: &str, body: &str) -> (StatusCode, Option<String>, String) {
    let engine = instant();
    let (status, value) = json(post(&app(&engine).await, path, body).await).await;
    assert!(engine.submitted().is_empty(), "a refused request reaches the engine: {value}");
    let error = &value["error"];
    (
        status,
        error["param"].as_str().map(str::to_owned),
        error["message"].as_str().unwrap_or_default().to_owned(),
    )
}

#[tokio::test]
async fn the_sampler_fields_of_llama_cpp_reach_the_engine() {
    let r = received(
        CHAT,
        &chat_body(json!({
            "temperature": 0.7, "top_k": 30, "top_p": 0.9, "min_p": 0.05,
            "typical_p": 0.95, "top_n_sigma": 2.0,
            "xtc_probability": 0.4, "xtc_threshold": 0.15,
            "dynatemp_range": 0.5, "dynatemp_exponent": 1.5,
            "mirostat": 2, "mirostat_tau": 4.0, "mirostat_eta": 0.2,
            "repeat_penalty": 1.1, "repeat_last_n": 128,
            "presence_penalty": 0.5, "frequency_penalty": 0.25,
            "dry_multiplier": 0.8, "dry_base": 2.0, "dry_allowed_length": 3,
            "dry_penalty_last_n": 256, "dry_sequence_breakers": ["\n", "###"],
            "ignore_eos": true, "min_keep": 1, "seed": 42, "n_predict": 33
        })),
    )
    .await;
    let s = &r.sampling;
    assert_eq!(
        (s.temperature, s.top_k, s.top_p, s.min_p),
        (Some(0.7), Some(30), Some(0.9), Some(0.05))
    );
    assert_eq!((s.typical_p, s.top_n_sigma), (Some(0.95), Some(2.0)));
    assert_eq!((s.xtc_probability, s.xtc_threshold), (Some(0.4), Some(0.15)));
    assert_eq!((s.dynatemp_range, s.dynatemp_exponent), (Some(0.5), Some(1.5)));
    assert_eq!((s.mirostat, s.mirostat_tau, s.mirostat_eta), (Some(2), Some(4.0), Some(0.2)));
    assert_eq!((s.repeat_penalty, s.repeat_last_n), (Some(1.1), Some(128)));
    assert_eq!((s.presence_penalty, s.frequency_penalty), (Some(0.5), Some(0.25)));
    assert_eq!(
        (s.dry_multiplier, s.dry_base, s.dry_allowed_length, s.dry_penalty_last_n),
        (Some(0.8), Some(2.0), Some(3), Some(256))
    );
    assert_eq!(s.dry_sequence_breakers, Some(vec!["\n".to_owned(), "###".to_owned()]));
    assert!(s.ignore_eos && s.seed == Some(42));
    assert_eq!(r.max_tokens, Some(33), "n_predict is the limit of the tokens");
}

#[tokio::test]
async fn nothing_set_is_nothing_sent() {
    let r = received(CHAT, &chat_body(json!({}))).await;
    assert_eq!(r.sampling, ares_api::Sampling::default());
}

#[tokio::test]
async fn the_names_other_servers_use_are_understood() {
    // vLLM, TGI and TabbyAPI write the repetition penalty differently.
    let r =
        received(COMPLETIONS, &completion_body(json!({"repetition_penalty": 1.2, "typical": 0.9})))
            .await;
    assert_eq!((r.sampling.repeat_penalty, r.sampling.typical_p), (Some(1.2), Some(0.9)));
    // Two names for one value must agree.
    let same = received(
        COMPLETIONS,
        &completion_body(json!({"repeat_penalty": 1.2, "repetition_penalty": 1.2})),
    )
    .await;
    assert_eq!(same.sampling.repeat_penalty, Some(1.2));
    let (status, param, message) = refused(
        COMPLETIONS,
        &completion_body(json!({"repeat_penalty": 1.2, "repetition_penalty": 1.3})),
    )
    .await;
    assert_eq!((status, param.as_deref()), (StatusCode::BAD_REQUEST, Some("repetition_penalty")));
    assert!(message.contains("repeat_penalty"), "{message}");
    // The limit of tokens by any of its names.
    for field in ["max_tokens", "max_completion_tokens", "n_predict"] {
        let r = received(CHAT, &chat_body(json!({ field: 12 }))).await;
        assert_eq!(r.max_tokens, Some(12), "{field}");
    }
}

#[tokio::test]
async fn a_logit_bias_comes_in_the_form_of_openai_or_of_llama_cpp() {
    // OpenAI: an object from a token (a string of digits) to a number.
    let r = received(CHAT, &chat_body(json!({"logit_bias": {"15043": 1.5, "7": -100}}))).await;
    let keys: Vec<_> = r.sampling.logit_bias.iter().map(|b| (b.key.clone(), b.bias)).collect();
    assert_eq!(keys, [(BiasKey::Token(15043), Some(1.5)), (BiasKey::Token(7), Some(-100.0))]);
    // llama.cpp: pairs, false for a ban, a string for the tokens of a text.
    let r = received(
        CHAT,
        &chat_body(json!({"logit_bias": [[15043, -1.0], [99, false], ["Hello, World!", -0.5]]})),
    )
    .await;
    let keys: Vec<_> = r.sampling.logit_bias.iter().map(|b| (b.key.clone(), b.bias)).collect();
    assert_eq!(
        keys,
        [
            (BiasKey::Token(15043), Some(-1.0)),
            (BiasKey::Token(99), None),
            (BiasKey::Text("Hello, World!".into()), Some(-0.5)),
        ]
    );
    // A string key of the object form is a text too.
    let r = received(CHAT, &chat_body(json!({"logit_bias": {"Hello": false}}))).await;
    assert_eq!(r.sampling.logit_bias[0].key, BiasKey::Text("Hello".into()));
    // Empty is nothing.
    let r = received(CHAT, &chat_body(json!({"logit_bias": {}}))).await;
    assert!(r.sampling.logit_bias.is_empty());
    for bad in [
        json!({"logit_bias": [[1]]}),
        json!({"logit_bias": [[1, "x"]]}),
        json!({"logit_bias": [[1.5, 1.0]]}),
        json!({"logit_bias": {"1": true}}),
        json!({"logit_bias": 3}),
    ] {
        let (status, param, _) = refused(CHAT, &chat_body(bad.clone())).await;
        assert_eq!(
            (status, param.as_deref()),
            (StatusCode::BAD_REQUEST, Some("logit_bias")),
            "{bad}"
        );
    }
}

#[tokio::test]
async fn the_order_of_the_samplers_is_names_or_letters() {
    let r = received(
        CHAT,
        &chat_body(json!({"samplers": ["penalties", "Top-K", "typ_p", "top_p", "min_p", "temp"]})),
    )
    .await;
    assert_eq!(
        r.sampling.samplers,
        Some(vec![
            SamplerKind::Penalties,
            SamplerKind::TopK,
            SamplerKind::TypicalP,
            SamplerKind::TopP,
            SamplerKind::MinP,
            SamplerKind::Temperature,
        ])
    );
    let r = received(CHAT, &chat_body(json!({"samplers": "edkypmxt"}))).await;
    assert_eq!(r.sampling.samplers.as_ref().map(Vec::len), Some(8));
    let r = received(CHAT, &chat_body(json!({"samplers": []}))).await;
    assert_eq!(r.sampling.samplers, Some(Vec::new()));
    for bad in [
        json!({"samplers": ["top_k", "nonsense"]}),
        json!({"samplers": ["infill"]}),
        json!({"samplers": "adaptive_p"}),
        json!({"samplers": "z"}),
        json!({"samplers": [1]}),
        json!({"samplers": 3}),
    ] {
        let (status, param, _) = refused(CHAT, &chat_body(bad.clone())).await;
        assert_eq!(
            (status, param.as_deref()),
            (StatusCode::BAD_REQUEST, Some("samplers")),
            "{bad}"
        );
    }
}

#[tokio::test]
async fn a_value_out_of_its_range_is_refused_by_the_name_of_its_field() {
    for (field, value) in [
        ("typical_p", json!(0.0)),
        ("typical_p", json!(1.5)),
        ("xtc_probability", json!(2.0)),
        ("xtc_threshold", json!(-0.1)),
        ("dynatemp_range", json!(-1.0)),
        ("mirostat", json!(3)),
        ("mirostat_tau", json!(0.0)),
        ("mirostat_eta", json!(-1.0)),
        ("dry_multiplier", json!(-0.5)),
        ("dry_base", json!(-2.0)),
        ("dry_allowed_length", json!(-1)),
        ("repeat_last_n", json!(100_000)),
        ("dry_penalty_last_n", json!(5000)),
        ("min_keep", json!(5)),
        ("min_keep", json!(-1)),
        ("dry_sequence_breakers", json!("\n")),
        ("ignore_eos", json!("yes")),
        ("top_n_sigma", json!("a lot")),
    ] {
        let (status, param, _) = refused(CHAT, &chat_body(json!({ field: value.clone() }))).await;
        assert_eq!(
            (status, param.as_deref()),
            (StatusCode::BAD_REQUEST, Some(field)),
            "{field} {value}"
        );
    }
    // The whole context is not a window this server has.
    let (_, param, message) = refused(CHAT, &chat_body(json!({"repeat_last_n": -1}))).await;
    assert_eq!(param.as_deref(), Some("repeat_last_n"));
    assert!(message.contains("whole context"), "{message}");
    // What llama.cpp has and this server does not stays an error that says so.
    let (_, param, message) = refused(CHAT, &chat_body(json!({"adaptive_target": 0.5}))).await;
    assert_eq!(param.as_deref(), Some("adaptive_target"));
    assert!(message.contains("adaptive-p"), "{message}");
    // vLLM's and TGI's cut of a prompt to its last tokens, with the way that is served.
    let (_, param, message) = refused(CHAT, &chat_body(json!({"truncate_prompt_tokens": 9}))).await;
    assert_eq!(param.as_deref(), Some("truncate_prompt_tokens"));
    assert!(message.contains("truncation: auto"), "{message}");
    let (_, param, _) = refused(CHAT, &chat_body(json!({"made_up_option": 1}))).await;
    assert_eq!(param.as_deref(), Some("made_up_option"));
}

fn probs() -> Vec<TokenProbs> {
    vec![
        TokenProbs {
            token: 1000,
            bytes: b"Hel".to_vec(),
            logprob: -0.25,
            alternatives: vec![
                Alternative { token: 1000, bytes: b"Hel".to_vec(), logprob: -0.25 },
                Alternative { token: 1001, bytes: b"He".to_vec(), logprob: -2.0 },
            ],
        },
        TokenProbs {
            token: 1002,
            bytes: vec![0xc3],
            logprob: -1.5,
            alternatives: vec![Alternative { token: 1002, bytes: vec![0xc3], logprob: -1.5 }],
        },
    ]
}

fn with_probs() -> std::sync::Arc<MockEngine> {
    MockEngine::script(vec![
        Step::Send(Event::Logprobs(probs())),
        text("Hel"),
        done(FinishReason::Stop, 3, 2),
    ])
}

#[tokio::test]
async fn chat_probabilities_are_asked_for_with_logprobs_and_come_back_per_token() {
    let engine = with_probs();
    let body = chat_body(json!({"logprobs": true, "top_logprobs": 2}));
    let (status, value) = json(post(&app(&engine).await, CHAT, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    assert_eq!(
        engine.submitted()[0].sampling.logprobs,
        Some(LogprobRequest { top: 2, after_chain: false })
    );
    let content = &value["choices"][0]["logprobs"]["content"];
    assert_eq!(content[0]["token"], "Hel");
    assert_eq!(content[0]["id"], 1000);
    assert_eq!(content[0]["bytes"], json!([72, 101, 108]));
    assert_eq!(content[0]["logprob"], -0.25);
    assert_eq!(content[0]["top_logprobs"][1]["token"], "He");
    assert_eq!(content[0]["top_logprobs"][1]["logprob"], -2.0);
    // A byte that is part of a character prints as the replacement character
    // and `bytes` says what it is.
    assert_eq!(content[1]["bytes"], json!([195]));
    assert_eq!(content[1]["token"], "\u{fffd}");
}

#[tokio::test]
async fn top_logprobs_needs_logprobs_and_has_a_limit() {
    for bad in [
        json!({"top_logprobs": 2}),
        json!({"logprobs": false, "top_logprobs": 2}),
        json!({"logprobs": true, "top_logprobs": 21}),
        json!({"logprobs": true, "top_logprobs": -1}),
        json!({"logprobs": "yes"}),
    ] {
        let (status, param, _) = refused(CHAT, &chat_body(bad.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
        assert!(matches!(param.as_deref(), Some("top_logprobs" | "logprobs")), "{bad}: {param:?}");
    }
    // `logprobs: false` asks for nothing.
    let r = received(CHAT, &chat_body(json!({"logprobs": false}))).await;
    assert_eq!(r.sampling.logprobs, None);
    // `logprobs: true` alone is the probability of each token, no alternatives.
    let r = received(CHAT, &chat_body(json!({"logprobs": true}))).await;
    assert_eq!(r.sampling.logprobs, Some(LogprobRequest { top: 0, after_chain: false }));
}

#[tokio::test]
async fn completions_probabilities_come_in_the_legacy_form() {
    let engine = with_probs();
    let body = completion_body(json!({"logprobs": 2}));
    let (status, value) = json(post(&app(&engine).await, COMPLETIONS, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    assert_eq!(
        engine.submitted()[0].sampling.logprobs,
        Some(LogprobRequest { top: 2, after_chain: false })
    );
    let lp = &value["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"], json!(["Hel", "\u{fffd}"]));
    assert_eq!(lp["token_logprobs"], json!([-0.25, -1.5]));
    assert_eq!(lp["top_logprobs"][0]["He"], -2.0);
    assert_eq!(lp["text_offset"], json!([0, 3]));
}

#[tokio::test]
async fn llama_cpp_asks_for_probabilities_with_n_probs() {
    let engine = with_probs();
    let body = completion_body(json!({"n_probs": 2}));
    let (_, value) = json(post(&app(&engine).await, COMPLETIONS, &body).await).await;
    assert_eq!(
        engine.submitted()[0].sampling.logprobs,
        Some(LogprobRequest { top: 2, after_chain: false })
    );
    let first = &value["choices"][0]["logprobs"]["content"][0];
    assert_eq!(first["logprob"], -0.25);
    assert_eq!(first["top_logprobs"][0]["token"], "Hel");
    // After the chain: probabilities, under the names llama.cpp gives them.
    let engine = with_probs();
    let body = completion_body(json!({"n_probs": 2, "post_sampling_probs": true}));
    let (_, value) = json(post(&app(&engine).await, COMPLETIONS, &body).await).await;
    assert_eq!(
        engine.submitted()[0].sampling.logprobs,
        Some(LogprobRequest { top: 2, after_chain: true })
    );
    let first = &value["choices"][0]["logprobs"]["content"][0];
    assert!((first["prob"].as_f64().unwrap() - (-0.25f64).exp()).abs() < 1e-6, "{first}");
    assert!(first.get("logprob").is_none());
    assert!(first["top_probs"][1]["prob"].as_f64().unwrap() < 0.2);
    // post_sampling_probs without n_probs asks for nothing it can give.
    let (_, param, _) =
        refused(COMPLETIONS, &completion_body(json!({"post_sampling_probs": true}))).await;
    assert_eq!(param.as_deref(), Some("post_sampling_probs"));
    // Both forms that ask for the same are llama.cpp's.
    let r = received(COMPLETIONS, &completion_body(json!({"n_probs": 3, "logprobs": 3}))).await;
    assert_eq!(r.sampling.logprobs, Some(LogprobRequest { top: 3, after_chain: false }));
}

#[tokio::test]
async fn two_names_of_one_parameter_must_agree() {
    // Each pair: the request is served when both names give the same value
    // and refused, by the second name, when they differ. No name wins.
    let pairs = [
        (CHAT, json!({"max_tokens": 9, "n_predict": 9}), json!({"n_predict": 10}), "n_predict"),
        (
            CHAT,
            json!({"max_tokens": 9, "max_completion_tokens": 9}),
            json!({"max_completion_tokens": 10}),
            "max_completion_tokens",
        ),
        (
            CHAT,
            json!({"max_completion_tokens": -1, "n_predict": -1}),
            json!({"n_predict": 10}),
            "n_predict",
        ),
        (
            COMPLETIONS,
            json!({"max_tokens": 9, "n_predict": 9}),
            json!({"n_predict": 10}),
            "n_predict",
        ),
        (CHAT, json!({"n": 2, "n_cmpl": 2}), json!({"n_cmpl": 3}), "n_cmpl"),
        (CHAT, json!({"typical_p": 0.5, "typical": 0.5}), json!({"typical": 0.6}), "typical"),
        (
            CHAT,
            json!({"repeat_penalty": 1.1, "repetition_penalty": 1.1}),
            json!({"repetition_penalty": 1.2}),
            "repetition_penalty",
        ),
        (
            CHAT,
            json!({"logprobs": true, "top_logprobs": 2, "n_probs": 2}),
            json!({"n_probs": 3}),
            "n_probs",
        ),
        (CHAT, json!({"logprobs": false, "n_probs": 0}), json!({"n_probs": 1}), "n_probs"),
        (COMPLETIONS, json!({"logprobs": 2, "n_probs": 2}), json!({"logprobs": 0}), "n_probs"),
    ];
    for (path, agree, change, param) in pairs {
        let body = |fields: &Value| match path {
            CHAT => chat_body(fields.clone()),
            _ => completion_body(fields.clone()),
        };
        received(path, &body(&agree)).await;
        let mut differ = agree.clone();
        for (name, value) in change.as_object().unwrap() {
            differ[name] = value.clone();
        }
        let (status, refused_param, message) = refused(path, &body(&differ)).await;
        assert_eq!(
            (status, refused_param.as_deref()),
            (StatusCode::BAD_REQUEST, Some(param)),
            "{path} {differ}: {message}"
        );
        assert!(message.contains("differ"), "{message}");
    }
}

#[tokio::test]
async fn a_stream_carries_the_probabilities_in_chunks_of_their_own() {
    let engine = with_probs();
    let body = chat_body(json!({"logprobs": true, "top_logprobs": 1, "stream": true}));
    let response = post(&app(&engine).await, CHAT, &body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    let mut raw = String::new();
    while let Some(chunk) = stream.next().await {
        raw.push_str(&String::from_utf8_lossy(&chunk.unwrap()));
    }
    let chunks: Vec<Value> = raw
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let with: Vec<&Value> =
        chunks.iter().filter(|c| c["choices"][0].get("logprobs").is_some()).collect();
    assert_eq!(with.len(), 1, "{raw}");
    assert_eq!(with[0]["choices"][0]["logprobs"]["content"].as_array().unwrap().len(), 2);
    assert!(raw.contains("[DONE]"));
}

#[tokio::test]
async fn n_completions_are_n_requests_with_their_own_seeds() {
    let engine = MockEngine::script(vec![text("ab"), done(FinishReason::Stop, 5, 2)]);
    let body = chat_body(json!({"n": 3, "seed": 100}));
    let (status, value) = json(post(&app(&engine).await, CHAT, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    let choices = value["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3);
    for (i, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], i);
        assert_eq!(choice["message"]["content"], "ab");
    }
    // One prompt, the completions of all.
    assert_eq!(value["usage"]["prompt_tokens"], 5);
    assert_eq!(value["usage"]["completion_tokens"], 6);
    let sent = engine.submitted();
    let mut seeds: Vec<_> = sent.iter().map(|r| r.sampling.seed).collect();
    seeds.sort();
    assert_eq!(seeds, [Some(100), Some(101), Some(102)]);
    let mut ids: Vec<_> = sent.iter().map(|r| r.id).collect();
    ids.dedup();
    assert_eq!(ids.len(), 3);
    // llama.cpp's name for the same.
    let engine = MockEngine::script(vec![text("x"), done(FinishReason::Stop, 1, 1)]);
    let (_, value) =
        json(post(&app(&engine).await, COMPLETIONS, &completion_body(json!({"n_cmpl": 2}))).await)
            .await;
    assert_eq!(value["choices"].as_array().unwrap().len(), 2);
    for bad in [json!({"n": 0}), json!({"n": 17}), json!({"n": -1})] {
        let (status, param, _) = refused(CHAT, &chat_body(bad.clone())).await;
        assert_eq!((status, param.as_deref()), (StatusCode::BAD_REQUEST, Some("n")), "{bad}");
    }
}

#[tokio::test]
async fn a_request_that_allows_no_drafts_says_so_with_every_sampler_field_and_for_every_completion()
{
    let engine = MockEngine::script(vec![text("ab"), done(FinishReason::Stop, 5, 2)]);
    let body = chat_body(json!({
        "draft": false, "n": 2, "seed": 7, "repeat_penalty": 1.2, "mirostat": 2,
        "logit_bias": [[" the", false]], "logprobs": true, "top_logprobs": 2,
    }));
    let (status, value) = json(post(&app(&engine).await, CHAT, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    let sent = engine.submitted();
    assert_eq!(sent.len(), 2);
    for request in &sent {
        assert!(!request.draft, "a completion of a request that allows no drafts");
        assert_eq!(request.sampling.repeat_penalty, Some(1.2));
        assert_eq!(request.sampling.mirostat, Some(2));
        assert_eq!(request.sampling.logit_bias.len(), 1);
        assert_eq!(request.sampling.logprobs, Some(LogprobRequest { top: 2, after_chain: false }));
    }
    // Without the field a request allows drafts.
    let request = received(CHAT, &chat_body(json!({"repeat_penalty": 1.2}))).await;
    assert!(request.draft);
}

#[tokio::test]
async fn n_streams_are_told_apart_by_index_and_end_once() {
    let engine = MockEngine::script(vec![text("hi"), done(FinishReason::Stop, 4, 1)]);
    let body =
        chat_body(json!({"n": 2, "stream": true, "stream_options": {"include_usage": true}}));
    let response = post(&app(&engine).await, CHAT, &body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    let mut raw = String::new();
    while let Some(chunk) = stream.next().await {
        raw.push_str(&String::from_utf8_lossy(&chunk.unwrap()));
    }
    let data: Vec<&str> = raw.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(data.iter().filter(|d| **d == "[DONE]").count(), 1, "{raw}");
    assert_eq!(*data.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> =
        data.iter().filter(|d| **d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
    for index in [0, 1] {
        let finishes = chunks
            .iter()
            .filter(|c| {
                c["choices"][0]["index"] == index && c["choices"][0]["finish_reason"] == "stop"
            })
            .count();
        assert_eq!(finishes, 1, "choice {index} finishes once: {raw}");
    }
    let usage: Vec<&Value> = chunks.iter().filter(|c| c.get("usage").is_some()).collect();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["usage"]["completion_tokens"], 2);
}
