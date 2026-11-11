//! `POST /v1/systemone`, driven in-process against a scripted engine: the wire
//! format of llama.cpp's endpoint, what the engine receives, the errors, and
//! the point of the endpoint, decisions that run next to generation. No
//! sockets are involved; `tests/e2e.rs` covers those.

mod common;

use std::sync::Arc;

use ares_api::{
    Answer, ChoiceOption, DecisionOutcome, DecisionRequest, DecisionResult, EngineError,
    FinishReason, ModelCard, Prompt, QuestionKind, Usage, Verdict,
};
use axum::http::{StatusCode, header};
use common::*;
use futures_util::StreamExt;
use serde_json::{Value, json};

const SYSTEMONE: &str = "/v1/systemone";
const CHAT: &str = "/v1/chat/completions";

/// A valid request with one choice question.
fn decision_body(extra: Value) -> String {
    let mut body = json!({
        "model": "mock",
        "state": "Customer message: I was charged twice.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": null, "shipping": null, "technical": null}
            }
        }
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body.to_string()
}

fn chat_body(extra: Value) -> String {
    let mut body = json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body.to_string()
}

fn usage(prompt_tokens: u32) -> Usage {
    Usage { prompt_tokens, completion_tokens: 0, cached_prompt_tokens: 0, ..Usage::default() }
}

/// Scores that favour the first option of every question: option `i` gets
/// `-i`, the log-likelihoods of a model that prefers earlier options.
fn favour_the_first(request: &DecisionRequest) -> DecisionOutcome {
    let answers = request
        .questions
        .iter()
        .map(|question| {
            let scores: Vec<f64> = (0..question.n_options()).map(|i| -(i as f64)).collect();
            question.answer(&scores).unwrap()
        })
        .collect();
    DecisionOutcome::Decided(DecisionResult { answers, usage: usage(7) })
}

/// A decision engine that answers at once.
fn deciding() -> Arc<MockEngine> {
    MockEngine::deciding(DecisionPlan::Script(vec![deliver_with(favour_the_first)]))
}

fn failing(error: EngineError) -> DecisionStep {
    deliver(DecisionOutcome::Failed(error))
}

/// The request the engine received for `body`.
async fn received(body: &str) -> DecisionRequest {
    let engine = deciding();
    let (status, value) = json(post(&app(&engine), SYSTEMONE, body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    engine.decided().remove(0)
}

fn close(value: &Value, expected: f64) {
    let value = value.as_f64().unwrap_or_else(|| panic!("{value} is not a number"));
    assert!((value - expected).abs() < 1e-4, "{value} should be about {expected}");
}

// --- the wire format -----------------------------------------------------------

/// The request of llama.cpp's README, as written there.
const README_REQUEST: &str = r#"{
    "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": null, "shipping": null, "technical": null}
        },
        "angry": {
            "type": "noul",
            "instructions": "Is the customer angry?"
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "this week", "today", "right now"]
        }
    }
}"#;

/// Answers with numbers that are exact in binary, so the body can be compared
/// as text.
fn readme_answers(_: &DecisionRequest) -> DecisionOutcome {
    let answer = |question: &str, verdict| Answer { question: question.into(), verdict };
    DecisionOutcome::Decided(DecisionResult {
        answers: vec![
            answer(
                "route",
                Verdict::Choice {
                    choice: "billing".into(),
                    probabilities: vec![
                        ("billing".into(), 0.75),
                        ("shipping".into(), 0.125),
                        ("technical".into(), 0.125),
                    ],
                    confidence: 0.625,
                },
            ),
            answer("angry", Verdict::Noul { probability: 0.5 }),
            answer(
                "urgency",
                Verdict::Score {
                    score: 2.1875,
                    probabilities: vec![0.0625, 0.125, 0.375, 0.4375],
                    confidence: 0.5,
                },
            ),
        ],
        usage: usage(239),
    })
}

#[tokio::test]
async fn the_readme_example_gets_the_documented_response_byte_for_byte() {
    // llama.cpp serves one model and its clients send no `model`.
    let models = vec![ModelCard { id: "openjev".into(), context_length: 8192, created: 1 }];
    let engine = MockEngine::with_models(
        models,
        Plan::Reject(EngineError::Unsupported("no generation".into())),
        DecisionPlan::Script(vec![deliver_with(readme_answers)]),
    );
    let response = post(&app(&engine), SYSTEMONE, README_REQUEST).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");

    // Members in the order llama.cpp writes them: the questions in the order
    // of the request, `type` first, `legend` between `score` and
    // `probabilities`, and nothing else (no id, no `object`, no `created`).
    assert_eq!(
        body_text(response).await,
        concat!(
            r#"{"model":"openjev","answers":{"#,
            r#""route":{"type":"choice","choice":"billing","#,
            r#""probabilities":{"billing":0.75,"shipping":0.125,"technical":0.125},"#,
            r#""confidence":0.625},"#,
            r#""angry":{"type":"noul","noul":0.5},"#,
            r#""urgency":{"type":"score","score":2.1875,"#,
            r#""legend":{"0":"can wait","1":"this week","2":"today","3":"right now"},"#,
            r#""probabilities":{"0":0.0625,"1":0.125,"2":0.375,"3":0.4375},"#,
            r#""confidence":0.5}},"#,
            r#""usage":{"input_tokens":239,"output_tokens":0}}"#
        )
    );

    // And the engine got the questions as the client wrote them.
    let request = engine.decided().remove(0);
    assert_eq!(request.model, "openjev", "the only served model is the default");
    assert_eq!(
        request.prompt,
        Prompt::Text(
            "Customer message: I was charged twice for my order last week and nobody has replied."
                .into()
        )
    );
    let ids: Vec<_> = request.questions.iter().map(|q| q.id.as_str()).collect();
    assert_eq!(ids, ["route", "angry", "urgency"], "in the order of the request");
    assert_eq!(request.questions[0].instructions, "Which team should handle this?");
    let QuestionKind::Choice { options } = &request.questions[0].kind else { panic!() };
    let keys: Vec<_> = options.iter().map(|o| (o.key.as_str(), o.description.as_deref())).collect();
    assert_eq!(keys, [("billing", None), ("shipping", None), ("technical", None)]);
    assert_eq!(request.questions[1].kind, QuestionKind::Noul { when_false: None, when_true: None });
    assert_eq!(
        request.questions[2].kind,
        QuestionKind::Score {
            levels: ["can wait", "this week", "today", "right now"]
                .map(|l| Some(l.to_owned()))
                .to_vec()
        }
    );
}

#[tokio::test]
async fn a_choice_is_answered_with_the_option_its_probabilities_and_its_confidence() {
    let body = decision_body(json!({"questions": {"route": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "shipping": null, "technical": {"area": "bugs"}}
    }}}));
    let engine = deciding();
    let response = post(&app(&engine), SYSTEMONE, &body).await;
    let text = body_text(response).await;
    let answer: Value = serde_json::from_str(&text).unwrap();

    assert_eq!(answer["model"], "mock");
    let route = &answer["answers"]["route"];
    assert_eq!(route["type"], "choice");
    assert_eq!(route["choice"], "billing");
    // softmax(0, -1, -2): the model prefers the first option.
    close(&route["probabilities"]["billing"], 0.66524);
    close(&route["probabilities"]["shipping"], 0.24473);
    close(&route["probabilities"]["technical"], 0.09003);
    let sum: f64 =
        route["probabilities"].as_object().unwrap().values().map(|p| p.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-9, "the probabilities sum to 1");
    close(&route["confidence"], 0.49786);
    assert_eq!(answer["usage"], json!({"input_tokens": 7, "output_tokens": 0}));
    assert_eq!(answer.as_object().unwrap().len(), 3, "model, answers, usage");

    // The probabilities come in the order of the options of the request.
    let at = |key: &str| text.find(&format!("\"{key}\":0.")).unwrap();
    assert!(at("billing") < at("shipping") && at("shipping") < at("technical"), "{text}");

    // Descriptions reach the engine as text: strings as they are, objects as
    // compact JSON, null as nothing.
    let QuestionKind::Choice { options } = &engine.decided()[0].questions[0].kind else { panic!() };
    assert_eq!(
        options.to_vec(),
        [
            ChoiceOption { key: "billing".into(), description: Some("money".into()) },
            ChoiceOption { key: "shipping".into(), description: None },
            ChoiceOption {
                key: "technical".into(),
                description: Some(r#"{"area":"bugs"}"#.into())
            },
        ]
    );
}

#[tokio::test]
async fn a_score_is_answered_with_the_expected_level_and_a_legend_as_the_client_wrote_it() {
    let body = decision_body(json!({"questions": {"urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["can wait", {"en": "this week"}, null, ["a", "b"]]
    }}}));
    let engine = deciding();
    let (status, answer) = json(post(&app(&engine), SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");

    let urgency = &answer["answers"]["urgency"];
    assert_eq!(urgency["type"], "score");
    // softmax(0, -1, -2, -3), expected level = sum of i * p_i.
    close(&urgency["score"], 0.50740);
    close(&urgency["probabilities"]["0"], 0.64391);
    close(&urgency["probabilities"]["3"], 0.03206);
    close(&urgency["confidence"], 0.49260);
    // The legend repeats the descriptions as they came in, whatever their type.
    assert_eq!(
        urgency["legend"],
        json!({"0": "can wait", "1": {"en": "this week"}, "2": null, "3": ["a", "b"]})
    );

    // The engine reads them as text.
    let QuestionKind::Score { levels } = &engine.decided()[0].questions[0].kind else { panic!() };
    assert_eq!(
        levels.to_vec(),
        [
            Some("can wait".to_owned()),
            Some(r#"{"en":"this week"}"#.into()),
            None,
            Some(r#"["a","b"]"#.into())
        ]
    );
}

#[tokio::test]
async fn a_noul_question_is_answered_with_the_probability_of_true() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![deliver_with(|request| {
        // [false, true]: the model finds true more likely.
        let answer = request.questions[0].answer(&[-2.0, 0.0]).unwrap();
        DecisionOutcome::Decided(DecisionResult { answers: vec![answer], usage: usage(3) })
    })]));
    let body = decision_body(json!({"questions": {"angry": {
        "type": "noul",
        "instructions": "Is the customer angry?",
        "criteria": {"true": "shouts", "false": "calm"}
    }}}));
    let (status, answer) = json(post(&app(&engine), SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["answers"]["angry"]["type"], "noul");
    close(&answer["answers"]["angry"]["noul"], 0.88080);
    assert_eq!(answer["answers"]["angry"].as_object().unwrap().len(), 2, "type and noul only");
    assert_eq!(
        engine.decided()[0].questions[0].kind,
        QuestionKind::Noul { when_false: Some("calm".into()), when_true: Some("shouts".into()) }
    );
}

// --- what the engine receives -------------------------------------------------

/// A request with this `state`, spelled out: `json!` would sort the members.
fn with_state(state: &str) -> String {
    format!(
        r#"{{"model":"mock","state":{state},"questions":{{"q":{{"type":"noul","instructions":"x"}}}}}}"#
    )
}

#[tokio::test]
async fn a_state_that_is_not_text_is_given_to_the_model_as_compact_json() {
    // A string is the text itself.
    let request = received(&with_state(r#""plain \"text\"""#)).await;
    assert_eq!(request.prompt, Prompt::Text("plain \"text\"".into()));

    // An object is its JSON with the members in the order written.
    let request = received(&with_state(r#"{ "z": 1, "a": [true, null], "m": "\u00e4" }"#)).await;
    assert_eq!(request.prompt, Prompt::Text(r#"{"z":1,"a":[true,null],"m":"ä"}"#.into()));

    // So is an array, and a state made of chat messages: the model reads them
    // as JSON text, as the API says.
    let request = received(&with_state(r#"[{"role": "user", "content": "hi"}]"#)).await;
    assert_eq!(request.prompt, Prompt::Text(r#"[{"role":"user","content":"hi"}]"#.into()));
    let wrapped =
        r#"{"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}"#;
    let request = received(&with_state(wrapped)).await;
    assert_eq!(
        request.prompt,
        Prompt::Text(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#.into()
        ),
        "text parts stay where they are"
    );
}

#[tokio::test]
async fn instructions_and_descriptions_are_text_for_the_model_whatever_their_type() {
    let body = decision_body(json!({"questions": {
        "a": {"type": "noul", "instructions": ["Is it?", {"hint": "look"}],
              "criteria": {"yes": "it is", "no": "it is not"}},
        "b": {"type": "noul", "instructions": {"q": "Is it?"},
              "criteria": {"true": "preferred", "yes": "ignored", "unknown": 5}},
        "c": {"type": "noul", "instructions": "", "criteria": null},
    }}));
    let request = received(&body).await;
    let by_id = |id: &str| request.questions.iter().find(|q| q.id == id).unwrap();
    assert_eq!(by_id("a").instructions, r#"["Is it?",{"hint":"look"}]"#);
    assert_eq!(
        by_id("a").kind,
        QuestionKind::Noul {
            when_false: Some("it is not".into()),
            when_true: Some("it is".into())
        },
        "'yes' and 'no' are read as llama.cpp reads them for h2o-lightning and lfm2-d1-omni"
    );
    assert_eq!(by_id("b").instructions, r#"{"q":"Is it?"}"#);
    assert_eq!(
        by_id("b").kind,
        QuestionKind::Noul { when_false: None, when_true: Some("preferred".into()) },
        "'true' wins over 'yes', and other members are ignored as llama.cpp's default does"
    );
    assert_eq!(by_id("c").instructions, "", "empty instructions are accepted as they are there");
    assert_eq!(by_id("c").kind, QuestionKind::Noul { when_false: None, when_true: None });
}

#[tokio::test]
async fn the_model_is_named_or_is_the_only_one_served() {
    // Named: it goes to the engine as it is.
    let request = received(&decision_body(json!({"model": "mock-2"}))).await;
    assert_eq!(request.model, "mock-2");

    // Two models and none named: the client must say which, and is told what exists.
    let engine = deciding();
    let body = decision_body(json!({})).replace(r#""model":"mock","#, "");
    for body in [body.clone(), body.replace(r#""state""#, r#""model":null,"state""#)] {
        let (status, value) = json(post(&app(&engine), SYSTEMONE, &body).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
        assert_eq!(value["error"]["param"], "model");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("mock, mock-2") && message.contains("name one"), "{message}");
    }
    assert!(engine.decided().is_empty());

    // One model and none named: that one. An empty name counts as none.
    let one = vec![ModelCard { id: "only".into(), context_length: 4096, created: 1 }];
    for body in [body.clone(), decision_body(json!({"model": ""}))] {
        let engine = MockEngine::with_models(
            one.clone(),
            Plan::Reject(EngineError::Unsupported("no generation".into())),
            DecisionPlan::Script(vec![deliver_with(favour_the_first)]),
        );
        let (status, value) = json(post(&app(&engine), SYSTEMONE, &body).await).await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert_eq!(value["model"], "only");
        assert_eq!(engine.decided()[0].model, "only");
    }

    // A model the engine does not know is its answer to give.
    let engine =
        MockEngine::deciding(DecisionPlan::Reject(EngineError::ModelNotFound("nope".into())));
    let (status, value) =
        json(post(&app(&engine), SYSTEMONE, &decision_body(json!({"model": "nope"}))).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(value["error"]["code"], "model_not_found");
    assert_eq!(value["error"]["param"], "model");
    assert!(value["error"]["message"].as_str().unwrap().contains("mock, mock-2"));
}

// --- options that only a native decision model has -----------------------------

#[tokio::test]
async fn images_and_audio_are_refused_with_501_after_the_rest_of_the_request_is_checked() {
    let data = "data:image/png;base64,iVBORw0KGgo=";
    let media = [
        json!({"files": [data]}),
        json!({"images": [data]}),
        json!({"state": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": data}}]}]}),
        json!({"state": {"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}]}}),
    ];
    let engine = deciding();
    let app = app(&engine);
    for extra in media {
        let (status, value) =
            json(post(&app, SYSTEMONE, &decision_body(extra.clone())).await).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{extra}: {value}");
        assert_eq!(value["error"]["code"], "not_implemented");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains("text only") && message.contains("'files'"), "{message}");
    }

    // A request that is wrong in another way is told that first.
    let body = decision_body(json!({"files": [data], "questions": {}}));
    let (status, value) = json(post(&app, SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
    assert_eq!(value["error"]["param"], "questions");
    assert!(engine.decided().is_empty());

    // No media is not media: empty lists, null, and parts that are not images.
    let text_only = json!({"files": [], "images": null, "videos": [], "state": {"messages": [
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [{"type": "text", "text": "parts"}, {"type": "image_url"}]}]}});
    assert_eq!(received(&decision_body(text_only)).await.questions.len(), 1);
}

#[tokio::test]
async fn videos_are_refused_as_llama_cpp_refuses_them() {
    let engine = deciding();
    let body = decision_body(json!({"videos": ["data:video/mp4;base64,AAAA"]}));
    let (status, value) = json(post(&app(&engine), SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(value["error"]["param"], "videos");
    assert!(value["error"]["message"].as_str().unwrap().contains("not supported"));
    assert!(engine.decided().is_empty());
}

// --- request validation ---------------------------------------------------------

/// A question body with one thing changed.
fn with_question(question: Value) -> String {
    decision_body(json!({"questions": {"q": question}}))
}

#[tokio::test]
async fn bad_requests_are_refused_before_the_engine_sees_them_and_name_the_field() {
    let choice = |criteria: Value| {
        with_question(json!({"type": "choice", "instructions": "x", "criteria": criteria}))
    };
    let score = |criteria: Value| {
        with_question(json!({"type": "score", "instructions": "x", "criteria": criteria}))
    };
    let noul = |criteria: Value| {
        with_question(json!({"type": "noul", "instructions": "x", "criteria": criteria}))
    };
    let state = |state: Value| decision_body(json!({ "state": state }));
    let ok_question = r#"{"type":"noul","instructions":"x"}"#;
    let cases: Vec<(String, Option<&str>, &str)> = vec![
        // (body, param, words the message must contain)
        ("{not json".into(), None, "not valid JSON"),
        (String::new(), None, "not valid JSON"),
        ("[1, 2]".into(), None, "must be a JSON object"),
        (decision_body(json!({"model": 5})), Some("model"), "model must be a string"),
        (decision_body(json!({"state": null})), Some("state"), "state is required"),
        (r#"{"model":"mock","questions":{}}"#.into(), Some("state"), "state is required"),
        (state(json!(5)), Some("state"), "state must be text, an object or an array, got a number"),
        (state(json!(true)), Some("state"), "state must be text"),
        (r#"{"model":"mock","state":"s"}"#.into(), Some("questions"), "questions is required"),
        (decision_body(json!({"questions": "x"})), Some("questions"), "questions must be an object"),
        (decision_body(json!({"questions": [1]})), Some("questions"), "got an array"),
        (decision_body(json!({"questions": {}})), Some("questions"), "questions must not be empty"),
        (decision_body(json!({"questions": {"q": 5}})), Some("questions.q"), "questions.q must be an object"),
        (with_question(json!({"type": "noul"})), Some("questions.q.instructions"), "questions.q.instructions is required"),
        (with_question(json!({"type": "noul", "instructions": null})), Some("questions.q.instructions"), "is required"),
        (with_question(json!({"type": "noul", "instructions": 5})), Some("questions.q.instructions"), "must be text, an object or an array, got a number"),
        (with_question(json!({"instructions": "x"})), Some("questions.q.type"), "questions.q.type is required"),
        (with_question(json!({"type": 5, "instructions": "x"})), Some("questions.q.type"), "questions.q.type must be a string"),
        (with_question(json!({"type": "pick", "instructions": "x"})), Some("questions.q.type"), "must be one of: choice, score, noul, got 'pick'"),
        (with_question(json!({"type": "choice", "instructions": "x"})), Some("questions.q.criteria"), "questions.q.criteria is required"),
        (choice(json!(["a", "b"])), Some("questions.q.criteria"), "must be an object that maps each option"),
        (choice(json!({})), Some("questions.q.criteria"), "at least one option"),
        (choice(json!({"a": 5})), Some("questions.q.criteria.a"), "questions.q.criteria.a must be text, an object or an array, got a number"),
        (choice(json!({"a": null, "": null})), Some("questions.q.criteria"), "has an empty option key"),
        (score(json!({"a": 1})), Some("questions.q.criteria"), "must be an array of 2 to 10 level descriptions"),
        (with_question(json!({"type": "score", "instructions": "x"})), Some("questions.q.criteria"), "is required"),
        (score(json!(["only"])), Some("questions.q.criteria"), "must have 2 to 10 levels, got 1"),
        (score(json!(vec!["l"; 11])), Some("questions.q.criteria"), "got 11"),
        (score(json!(["a", 5])), Some("questions.q.criteria[1]"), "questions.q.criteria[1] must be text"),
        (noul(json!(["true"])), Some("questions.q.criteria"), "must be an object that describes"),
        (noul(json!({"true": 5})), Some("questions.q.criteria.true"), "must be text, an object or an array"),
        (noul(json!({"no": [1], "false": 5})), Some("questions.q.criteria.false"), "must be text"),
        (decision_body(json!({"files": "data:x"})), Some("files"), "files must be an array"),
        (decision_body(json!({"files": ["https://x/y.png"]})), Some("files[0]"), "files[0] must be a data URL"),
        (decision_body(json!({"images": ["data:ok", 5]})), Some("images[1]"), "images[1] must be a data URL"),
        // The same member twice. Most parsers keep one of them silently; here
        // the client is asked what it meant.
        (
            r#"{"model":"mock","state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":null,"a":"again"}}}}"#.into(),
            Some("questions.q.criteria"),
            "lists the option 'a' twice",
        ),
        (
            format!(r#"{{"model":"mock","state":"s","questions":{{"q":{ok_question},"q":{ok_question}}}}}"#),
            Some("questions.q"),
            "questions.q is listed twice",
        ),
    ];

    let engine = deciding();
    let app = app(&engine);
    for (body, param, words) in cases {
        let (status, value) = json(post(&app, SYSTEMONE, &body).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {value}");
        let error = &value["error"];
        assert_eq!(error["type"], "invalid_request_error", "{body}");
        assert_eq!(error["code"], "invalid_request", "{body}");
        assert_eq!(error["param"].as_str(), param, "{body}: {value}");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains(words), "{body}: '{message}' should contain '{words}'");
    }
    assert!(engine.decided().is_empty(), "no invalid request may reach the engine");
}

#[tokio::test]
async fn an_oversized_body_is_refused_with_413() {
    let engine = deciding();
    let app = app_with(&engine, ares_server::Config { max_body_bytes: 64, ..Default::default() });
    let (status, body) = json(post(&app, SYSTEMONE, &decision_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["error"]["code"], "request_too_large");
    assert!(engine.decided().is_empty());
}

// --- routes ------------------------------------------------------------------

#[tokio::test]
async fn the_endpoint_is_on_v1_systemone_and_nowhere_else() {
    let engine = deciding();
    let app = app(&engine);
    for path in ["/v1/decision", "/systemone", "/v1/systemone/", "/v1/systemone/x", "/v2/systemone"]
    {
        let (status, body) = json(post(&app, path, &decision_body(json!({}))).await).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
        assert_eq!(body["error"]["code"], "not_found");
    }
    let (status, body) = json(get(&app, SYSTEMONE).await).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(body["error"]["code"], "method_not_allowed");
    assert!(engine.decided().is_empty());
}

// --- errors from the engine -----------------------------------------------------

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
            error: EngineError::InvalidRequest("questions.q has too many options".into()),
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: "invalid_request",
            mentions: &["questions.q has too many options"],
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
    assert!(error.get("param").is_some(), "{how}: the error format always has a param");
    assert_eq!(body.as_object().unwrap().len(), 1, "{how}: nothing but the error object");
    let expected_retry = (case.status == StatusCode::SERVICE_UNAVAILABLE).then_some("1");
    assert_eq!(retry_after.as_ref().and_then(|v| v.to_str().ok()), expected_retry, "{how}");
}

#[tokio::test]
async fn an_error_returned_by_decide_maps_to_the_status_of_every_other_endpoint() {
    for case in cases() {
        let engine = MockEngine::deciding(DecisionPlan::Reject(case.error.clone()));
        let response = post(&app(&engine), SYSTEMONE, &decision_body(json!({}))).await;
        assert_error(&case, response, "rejected at decide").await;
    }
}

#[tokio::test]
async fn a_failure_delivered_as_the_outcome_maps_the_same_way() {
    for case in cases() {
        let engine = MockEngine::deciding(DecisionPlan::Script(vec![failing(case.error.clone())]));
        let response = post(&app(&engine), SYSTEMONE, &decision_body(json!({}))).await;
        assert_error(&case, response, "failed outcome").await;
        assert!(engine.cancelled().is_empty(), "the engine ended it; there is nothing to cancel");
    }
}

#[tokio::test]
async fn a_decision_the_engine_cancelled_under_a_waiting_client_is_an_error() {
    let engine =
        MockEngine::deciding(DecisionPlan::Script(vec![deliver(DecisionOutcome::Cancelled)]));
    let response = post(&app(&engine), SYSTEMONE, &decision_body(json!({}))).await;
    assert_eq!(response.headers()[header::RETRY_AFTER], "1");
    let (status, body) = json(response).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "cancelled");
    assert!(engine.cancelled().is_empty(), "the engine ended it; there is nothing left to cancel");
}

#[tokio::test]
async fn an_engine_that_drops_the_sink_without_an_outcome_is_reported_not_waited_for() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![DecisionStep::Abandon]));
    let (status, body) =
        json(post(&app(&engine), SYSTEMONE, &decision_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body["error"]["message"].as_str().unwrap().contains("without finishing"));
    assert_eq!(engine.cancelled(), [engine.decided()[0].id], "the request is released");
}

#[tokio::test]
async fn an_answer_that_does_not_match_the_questions_is_an_error_and_not_a_response() {
    let answer = |question: &str, verdict| Answer { question: question.into(), verdict };
    let choice = |question: &str| {
        answer(
            question,
            Verdict::Choice {
                choice: "billing".into(),
                probabilities: vec![
                    ("billing".into(), 0.5),
                    ("shipping".into(), 0.25),
                    ("technical".into(), 0.25),
                ],
                confidence: 0.25,
            },
        )
    };
    let wrong: Vec<(Vec<Answer>, &str)> = vec![
        (vec![], "0 answers for 1 questions"),
        (vec![choice("route"), choice("route")], "2 answers for 1 questions"),
        (vec![choice("other")], "the answer to 'other' where 'route' was asked"),
        (vec![answer("route", Verdict::Noul { probability: 0.5 })], "wrong kind for 'route'"),
        (
            vec![answer(
                "route",
                Verdict::Choice {
                    choice: "billing".into(),
                    probabilities: vec![("billing".into(), 1.0)],
                    confidence: 1.0,
                },
            )],
            "1 probabilities for the 3 options of 'route'",
        ),
    ];
    for (answers, words) in wrong {
        let outcome = DecisionOutcome::Decided(DecisionResult { answers, usage: usage(1) });
        let engine = MockEngine::deciding(DecisionPlan::Script(vec![deliver(outcome)]));
        let (status, body) =
            json(post(&app(&engine), SYSTEMONE, &decision_body(json!({}))).await).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{words}: {body}");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains(words), "'{message}' should contain '{words}'");
    }
}

// --- cancellation ----------------------------------------------------------------

#[tokio::test]
async fn a_client_that_leaves_a_decision_cancels_it_in_the_engine() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![DecisionStep::UntilCancelled]));
    let app = app(&engine);
    let call = tokio::spawn({
        let app = app.clone();
        async move {
            tower::ServiceExt::oneshot(app, post_request(SYSTEMONE, &decision_body(json!({}))))
                .await
        }
    });
    eventually("the decision to reach the engine", || engine.decided().len() == 1).await;
    assert!(engine.cancelled().is_empty(), "nothing to cancel while the client waits");

    // What hyper does when the connection closes: it drops the handler's future.
    call.abort();

    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.decided()[0].id]);
}

#[tokio::test]
async fn finished_and_invalid_decisions_are_never_cancelled() {
    let engine = deciding();
    let app = app(&engine);
    for _ in 0..3 {
        assert_eq!(post(&app, SYSTEMONE, &decision_body(json!({}))).await.status(), StatusCode::OK);
    }
    let broken = MockEngine::deciding(DecisionPlan::Script(vec![failing(EngineError::Overloaded)]));
    let _ = post(&common::app(&broken), SYSTEMONE, &decision_body(json!({}))).await;
    // Invalid: never started.
    let _ = post(&app, SYSTEMONE, "{").await;

    assert_eq!(engine.decided().len(), 3);
    assert!(engine.cancelled().is_empty(), "{:?}", engine.cancelled());
    assert!(broken.cancelled().is_empty());
}

// --- decisions next to generation -------------------------------------------------

/// Reads `body` until `needle` has arrived. Returns everything read so far.
async fn read_stream_until(
    body: &mut (impl futures_util::Stream<Item = Result<axum::body::Bytes, axum::Error>> + Unpin),
    seen: &mut String,
    needle: &str,
) {
    while !seen.contains(needle) {
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(10), body.next())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {needle:?} in {seen:?}"))
            .unwrap_or_else(|| panic!("the stream ended before {needle:?}: {seen:?}"))
            .unwrap();
        seen.push_str(std::str::from_utf8(&chunk).unwrap());
    }
}

fn streamed_text(body: &str) -> String {
    parse_sse(body)
        .chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

#[tokio::test]
async fn a_decision_completes_while_a_chat_stream_is_open_and_the_stream_then_finishes_intact() {
    let gate = Gate::default();
    let engine = MockEngine::both(
        Plan::Script(vec![
            text("Hel"),
            Step::Wait(gate.clone()),
            text("lo "),
            text("world"),
            done(FinishReason::Stop, 3, 3),
        ]),
        DecisionPlan::Script(vec![deliver_with(favour_the_first)]),
    );
    let app = app(&engine);

    // The stream is open: its first piece arrived, and the engine thread of
    // the stream now waits for the gate, which the test holds shut.
    let response = post(&app, CHAT, &chat_body(json!({"stream": true}))).await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    let mut seen = String::new();
    read_stream_until(&mut stream, &mut seen, "Hel").await;
    eventually("the engine to have sent the first piece", || engine.sends() == [true]).await;

    // A decision submitted now is answered, with the stream still waiting.
    let (status, answer) = json(post(&app, SYSTEMONE, &decision_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["answers"]["route"]["choice"], "billing");
    assert_eq!(engine.sends(), [true], "the stream did not move: it was still running");
    assert_eq!(engine.submitted().len(), 1);
    assert!(engine.cancelled().is_empty());

    // Now the stream goes on, and nothing the decision did is in it.
    gate.open();
    while let Some(chunk) = stream.next().await {
        seen.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
    }
    let parsed = parse_sse(&seen);
    assert!(parsed.done, "the stream ended with [DONE]: {seen}");
    assert_eq!(streamed_text(&seen), "Hello world");
    assert_eq!(parsed.chunks.last().unwrap()["choices"][0]["finish_reason"], "stop");
    assert!(!seen.contains("route") && !seen.contains("billing"), "{seen}");
    assert!(engine.cancelled().is_empty());
}

#[tokio::test]
async fn decisions_between_every_piece_of_a_stream_are_all_answered_and_the_stream_is_undisturbed()
{
    const PIECES: [&str; 4] = ["one ", "two ", "three ", "four"];
    let gates: Vec<Gate> = PIECES.iter().map(|_| Gate::default()).collect();
    // A streamed response starts with the engine's first event, so the first
    // piece is not behind a gate.
    let mut steps = vec![text("zero ")];
    for (gate, piece) in gates.iter().zip(PIECES) {
        steps.push(Step::Wait(gate.clone()));
        steps.push(text(piece));
    }
    steps.push(done(FinishReason::Stop, 1, 4));
    let engine = MockEngine::both(
        Plan::Script(steps),
        DecisionPlan::Script(vec![deliver_with(favour_the_first)]),
    );
    let app = app(&engine);

    let response = post(&app, CHAT, &chat_body(json!({"stream": true}))).await;
    let mut stream = response.into_body().into_data_stream();
    let mut seen = String::new();
    let mut decisions = 0;
    for (gate, piece) in gates.iter().zip(PIECES) {
        gate.open();
        read_stream_until(&mut stream, &mut seen, piece).await;
        // The stream is open and waits for the next gate; decide meanwhile,
        // three at once.
        let calls: Vec<_> = (0..3)
            .map(|_| {
                let app = app.clone();
                tokio::spawn(async move {
                    json(post(&app, SYSTEMONE, &decision_body(json!({}))).await).await
                })
            })
            .collect();
        for call in calls {
            let (status, answer) = call.await.unwrap();
            assert_eq!(status, StatusCode::OK, "{answer}");
            decisions += 1;
        }
    }
    while let Some(chunk) = stream.next().await {
        seen.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
    }

    assert_eq!(decisions, 12);
    assert_eq!(engine.decided().len(), 12);
    assert_eq!(streamed_text(&seen), format!("zero {}", PIECES.concat()));
    assert!(parse_sse(&seen).done);
    assert!(engine.sends().iter().all(|sent| *sent), "no event of the stream was refused");
    assert!(engine.cancelled().is_empty());

    // One id space for both kinds of request.
    let mut ids: Vec<_> = engine.decided().iter().map(|r| r.id).collect();
    ids.extend(engine.submitted().iter().map(|r| r.id));
    let distinct: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(distinct.len(), 13);
}

#[tokio::test]
async fn a_slow_decision_does_not_hold_up_a_chat_completion() {
    let gate = Gate::default();
    let engine = MockEngine::both(
        Plan::Script(vec![text("fast"), done(FinishReason::Stop, 1, 1)]),
        DecisionPlan::Script(vec![
            DecisionStep::Wait(gate.clone()),
            deliver_with(favour_the_first),
        ]),
    );
    let app = app(&engine);
    let slow = tokio::spawn({
        let app = app.clone();
        async move { json(post(&app, SYSTEMONE, &decision_body(json!({}))).await).await }
    });
    eventually("the decision to reach the engine", || engine.decided().len() == 1).await;

    // The decision is stuck in the engine; a chat completion still runs through.
    let (status, chat) = json(post(&app, CHAT, &chat_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(chat["choices"][0]["message"]["content"], "fast");
    assert!(!slow.is_finished(), "the decision is still waiting");

    gate.open();
    let (status, answer) = slow.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["answers"]["route"]["choice"], "billing");
    assert!(engine.cancelled().is_empty());
}
