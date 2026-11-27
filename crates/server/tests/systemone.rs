//! `POST /v1/systemone`, driven in-process against a scripted engine: the wire
//! format of Flide's endpoint, what the engine receives, the errors, several
//! requests in one body, and the point of the endpoint, decisions that run
//! next to generation. No sockets are involved; `tests/e2e.rs` covers those.

mod common;

use std::sync::Arc;

use ares_api::{
    Answer, ChoiceOption, DecisionOutcome, DecisionRequest, DecisionResult, EngineError,
    FinishReason, Json, ModelCard, QuestionKind, Usage, Verdict,
};
use axum::http::{StatusCode, header};
use common::*;
use futures_util::StreamExt;
use serde_json::{Value, json};

const SYSTEMONE: &str = "/v1/systemone";
const CHAT: &str = "/v1/chat/completions";

/// A valid request with one choice question.
fn decision_body(extra: Value) -> String {
    merged(
        json!({
            "model": "mock",
            "state": "Customer message: I was charged twice.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"billing": null, "shipping": null, "technical": null}
                }
            }
        }),
        extra,
    )
}

fn usage(prompt_tokens: u32, questions: u32) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens: questions,
        cached_prompt_tokens: 0,
        ..Usage::default()
    }
}

fn result(answers: Vec<Answer>, prompt_tokens: u32) -> DecisionResult {
    let questions = answers.len() as u32;
    DecisionResult { answers, usage: usage(prompt_tokens, questions), prefix_tokens: 5 }
}

/// Logits that favour the first option of every question: option `i` gets
/// `-i`.
fn favour_the_first(request: &DecisionRequest) -> DecisionOutcome {
    let answers = request
        .questions
        .iter()
        .map(|question| {
            let logits: Vec<f32> = (0..question.n_options()).map(|i| -(i as f32)).collect();
            question.answer(&logits).unwrap()
        })
        .collect();
    DecisionOutcome::Decided(result(answers, 7))
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
    let (status, value) = json(post(&app(&engine).await, SYSTEMONE, body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    engine.decided().remove(0)
}

fn close(value: &Value, expected: f64) {
    let value = value.as_f64().unwrap_or_else(|| panic!("{value} is not a number"));
    assert!((value - expected).abs() < 2e-6, "{value} should be about {expected}");
}

fn string(s: &str) -> Json {
    Json::String(s.into())
}

/// The one problem of a 422, as `(loc, msg, type)`.
fn problem(value: &Value) -> (Value, String, String) {
    let detail =
        value["detail"].as_array().unwrap_or_else(|| panic!("a list of problems: {value}"));
    assert_eq!(detail.len(), 1, "{value}");
    let p = &detail[0];
    (
        p["loc"].clone(),
        p["msg"].as_str().unwrap().to_owned(),
        p["type"].as_str().unwrap().to_owned(),
    )
}

// --- the wire format -----------------------------------------------------------

/// The request of the TypeSafe API's example, as Flide serves it.
const EXAMPLE_REQUEST: &str = r#"{
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
fn example_answers(_: &DecisionRequest) -> DecisionOutcome {
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
        usage: Usage { cached_prompt_tokens: 64, ..usage(239, 3) },
        prefix_tokens: 80,
    })
}

#[tokio::test]
async fn the_example_gets_flides_response_byte_for_byte() {
    // Flide serves one model and its clients may send no `model`.
    let models = vec![ModelCard { id: "openjev".into(), context_length: 8192, created: 1 }];
    let engine = MockEngine::with_models(
        models,
        Plan::Reject(EngineError::Unsupported("no generation".into())),
        DecisionPlan::Script(vec![deliver_with(example_answers)]),
    );
    let response = post(&app(&engine).await, SYSTEMONE, EXAMPLE_REQUEST).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
    let id = response.headers()["x-typesafe-request-id"].to_str().unwrap().to_owned();
    assert!(id.starts_with("ares-"), "{id}");

    // Members in the order Flide writes them: the questions in the order of
    // the request, `type` first, `legend` between `score` and
    // `probabilities`, then `usage` and `meta`.
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
            r#""usage":{"input_tokens":239,"output_tokens":3},"#,
            r#""meta":{"format":"next_token","prefix_tokens":80,"branches":3,"cached_tokens":64}}"#
        )
    );

    // And the engine got the questions as the client wrote them.
    let request = engine.decided().remove(0);
    assert_eq!(request.model, "openjev", "the only served model is the default");
    assert_eq!(
        request.state,
        string(
            "Customer message: I was charged twice for my order last week and nobody has replied."
        )
    );
    let ids: Vec<_> = request.questions.iter().map(|q| q.id.as_str()).collect();
    assert_eq!(ids, ["route", "angry", "urgency"], "in the order of the request");
    assert_eq!(request.questions[0].instructions, string("Which team should handle this?"));
    let QuestionKind::Choice { options } = &request.questions[0].kind else { panic!() };
    let keys: Vec<_> = options.iter().map(|o| (o.key.as_str(), o.description.clone())).collect();
    assert_eq!(
        keys,
        [("billing", Json::Null), ("shipping", Json::Null), ("technical", Json::Null)]
    );
    assert_eq!(
        request.questions[1].kind,
        QuestionKind::Noul { when_true: Json::Null, when_false: Json::Null }
    );
    assert_eq!(
        request.questions[2].kind,
        QuestionKind::Score {
            levels: ["can wait", "this week", "today", "right now"].map(string).to_vec()
        }
    );
}

#[tokio::test]
async fn the_request_id_header_comes_back_as_the_client_sent_it() {
    let engine = deciding();
    let request = axum::http::Request::builder()
        .method("POST")
        .uri(SYSTEMONE)
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-typesafe-request-id", "client-7")
        .body(axum::body::Body::from(decision_body(json!({}))))
        .unwrap();
    let response = tower::ServiceExt::oneshot(app(&engine).await, request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-typesafe-request-id"], "client-7");
    // An error carries it too.
    let (status, _) = json(post(&app(&engine).await, SYSTEMONE, "{").await).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn a_choice_is_answered_with_the_option_its_probabilities_and_its_confidence() {
    let body = decision_body(json!({"questions": {"route": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "shipping": null, "technical": {"area": "bugs"}}
    }}}));
    let engine = deciding();
    let response = post(&app(&engine).await, SYSTEMONE, &body).await;
    let text_body = body_text(response).await;
    let answer: Value = serde_json::from_str(&text_body).unwrap();

    assert_eq!(answer["model"], "mock");
    let route = &answer["answers"]["route"];
    assert_eq!(route["type"], "choice");
    assert_eq!(route["choice"], "billing");
    // softmax(0, -1, -2), rounded to six places as Flide rounds.
    assert_eq!(route["probabilities"]["billing"], 0.665241);
    assert_eq!(route["probabilities"]["shipping"], 0.244728);
    assert_eq!(route["probabilities"]["technical"], 0.090031);
    close(&route["confidence"], 0.497862);
    assert_eq!(answer["usage"], json!({"input_tokens": 7, "output_tokens": 1}));
    assert_eq!(answer.as_object().unwrap().len(), 4, "model, answers, usage, meta");

    // The probabilities come in the order of the options of the request.
    let at = |key: &str| text_body.find(&format!("\"{key}\":0.")).unwrap();
    assert!(at("billing") < at("shipping") && at("shipping") < at("technical"), "{text_body}");

    // Descriptions reach the engine as the client wrote them.
    let QuestionKind::Choice { options } = &engine.decided()[0].questions[0].kind else { panic!() };
    assert_eq!(
        options.to_vec(),
        [
            ChoiceOption { key: "billing".into(), description: string("money") },
            ChoiceOption { key: "shipping".into(), description: Json::Null },
            ChoiceOption {
                key: "technical".into(),
                description: Json::Object(vec![("area".into(), string("bugs"))])
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
    let (status, answer) = json(post(&app(&engine).await, SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");

    let urgency = &answer["answers"]["urgency"];
    assert_eq!(urgency["type"], "score");
    // softmax(0, -1, -2, -3), expected level = sum of i * p_i.
    close(&urgency["score"], 0.507347);
    close(&urgency["probabilities"]["0"], 0.643914);
    close(&urgency["probabilities"]["3"], 0.032059);
    close(&urgency["confidence"], 0.492653);
    // The legend repeats the levels as they came in, whatever their type.
    assert_eq!(
        urgency["legend"],
        json!({"0": "can wait", "1": {"en": "this week"}, "2": null, "3": ["a", "b"]})
    );
    let QuestionKind::Score { levels } = &engine.decided()[0].questions[0].kind else { panic!() };
    assert_eq!(levels[2], Json::Null);
}

#[tokio::test]
async fn a_noul_question_is_answered_with_the_probability_of_yes() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![deliver_with(|request| {
        // [yes, no]: the model finds yes more likely.
        let answer = request.questions[0].answer(&[0.0, -2.0]).unwrap();
        DecisionOutcome::Decided(result(vec![answer], 3))
    })]));
    let body = decision_body(json!({"questions": {"angry": {
        "type": "noul",
        "instructions": "Is the customer angry?",
        "criteria": {"true": "shouts", "false": "calm", "yes": "not read"}
    }}}));
    let (status, answer) = json(post(&app(&engine).await, SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["answers"]["angry"]["type"], "noul");
    assert_eq!(answer["answers"]["angry"]["noul"], 0.880797);
    assert_eq!(answer["answers"]["angry"].as_object().unwrap().len(), 2, "type and noul only");
    assert_eq!(
        engine.decided()[0].questions[0].kind,
        QuestionKind::Noul { when_true: string("shouts"), when_false: string("calm") }
    );
}

// --- what the engine receives -------------------------------------------------

/// A request with this `state`, spelled out: `json!` would sort the members.
fn with_state(state: &str) -> String {
    format!(r#"{{"model":"mock","state":{state},"questions":{{"q":{{"type":"noul"}}}}}}"#)
}

#[tokio::test]
async fn the_state_and_the_instructions_reach_the_engine_as_the_client_wrote_them() {
    let request = received(&with_state(r#""plain \"text\"""#)).await;
    assert_eq!(request.state, string("plain \"text\""));
    let request = received(&with_state(r#"{ "z": 1, "a": [true, null], "z": "ä" }"#)).await;
    assert_eq!(
        request.state,
        Json::Object(vec![
            ("z".into(), Json::Int(1)),
            ("a".into(), Json::Array(vec![Json::Bool(true), Json::Null])),
            ("z".into(), string("ä")),
        ]),
        "the engine writes it as Python does, a repeated key with its last value"
    );
    // No state is null, which the model reads as the text `null`, as in Flide.
    let request = received(r#"{"questions":{"q":{"type":"noul"}}}"#).await;
    assert_eq!(request.state, Json::Null);
    assert_eq!(request.questions[0].instructions, Json::Null, "instructions are optional");
    let request = received(&decision_body(json!({"questions": {"q": {
        "type": "noul", "instructions": ["Is it?", {"hint": "look"}], "criteria": null}}})))
    .await;
    assert_eq!(
        request.questions[0].instructions,
        Json::Array(vec![string("Is it?"), Json::Object(vec![("hint".into(), string("look"))])])
    );
}

#[tokio::test]
async fn a_repeated_question_or_option_keeps_its_last_value_as_python_reads_it() {
    let body = r#"{"state":"s","questions":{
        "q":{"type":"noul"},
        "r":{"type":"choice","criteria":{"a":null,"b":null,"a":"again"}},
        "q":{"type":"score","criteria":["x","y"]}}}"#;
    let request = received(body).await;
    let ids: Vec<_> = request.questions.iter().map(|q| (q.id.as_str(), q.type_name())).collect();
    assert_eq!(ids, [("q", "score"), ("r", "choice")]);
    let QuestionKind::Choice { options } = &request.questions[1].kind else { panic!() };
    assert_eq!(options.len(), 2);
    assert_eq!(options[0].description, string("again"));
}

#[tokio::test]
async fn the_model_is_named_or_is_the_one_served() {
    // Named: it goes to the engine as it is.
    let two = vec![
        ModelCard { id: "mock".into(), context_length: 4096, created: 1 },
        ModelCard { id: "mock-2".into(), context_length: 8192, created: 2 },
    ];
    let decides = || DecisionPlan::Script(vec![deliver_with(favour_the_first)]);
    let refuses = || Plan::Reject(EngineError::Unsupported("no generation".into()));
    let engine = MockEngine::with_models(two.clone(), refuses(), decides());
    let named = decision_body(json!({"model": "mock-2"}));
    let (status, value) = json(post(&app(&engine).await, SYSTEMONE, &named).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    assert_eq!(engine.decided()[0].model, "mock-2");
    assert_eq!(value["model"], "mock-2");

    // One model, and none named, `jev-latest`, an empty name or null: that one.
    let one = vec![ModelCard { id: "only".into(), context_length: 4096, created: 1 }];
    let unnamed = decision_body(json!({})).replace(r#""model":"mock","#, "");
    for body in [
        unnamed.clone(),
        decision_body(json!({"model": "jev-latest"})),
        decision_body(json!({"model": ""})),
        decision_body(json!({"model": null})),
    ] {
        let engine = MockEngine::with_models(one.clone(), refuses(), decides());
        let (status, value) = json(post(&app(&engine).await, SYSTEMONE, &body).await).await;
        assert_eq!(status, StatusCode::OK, "{body}: {value}");
        assert_eq!(value["model"], "only");
        assert_eq!(engine.decided()[0].model, "only");
    }

    // Two models and none named: Flide's 404, which lists what is served.
    let engine = MockEngine::with_models(two, refuses(), decides());
    let (status, value) = json(post(&app(&engine).await, SYSTEMONE, &unnamed).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{value}");
    let detail = value["detail"].as_str().unwrap();
    assert!(
        detail.contains("'jev-latest' is not served") && detail.contains("mock, mock-2"),
        "{detail}"
    );
    assert!(engine.decided().is_empty());

    // A model that is not served.
    let engine = deciding();
    let (status, value) =
        json(post(&app(&engine).await, SYSTEMONE, &decision_body(json!({"model": "nope"}))).await)
            .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        value["detail"],
        "model 'nope' is not served; available: mock plus aliases ['jev-latest']"
    );
}

// --- several requests in one body -------------------------------------------------

#[tokio::test]
async fn several_requests_in_one_body_are_answered_together_in_their_order() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![deliver_with(|request| {
        let first = match &request.state {
            Json::String(s) if s == "second" => 1,
            _ => 0,
        };
        let answers = request
            .questions
            .iter()
            .map(|q| {
                let logits: Vec<f32> =
                    (0..q.n_options()).map(|i| if i == first { 0.0 } else { -5.0 }).collect();
                q.answer(&logits).unwrap()
            })
            .collect();
        DecisionOutcome::Decided(result(answers, 11))
    })]));
    let item = |state: &str| {
        json!({"state": state, "questions": {"route": {"type": "choice",
            "criteria": {"billing": null, "shipping": null}}}})
    };
    let body = json!({"model": "mock", "requests": [item("first"), item("second")]}).to_string();
    let (status, value) = json(post(&app(&engine).await, SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    assert_eq!(value["model"], "mock");
    let results = value["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["answers"]["route"]["choice"], "billing");
    assert_eq!(results[1]["answers"]["route"]["choice"], "shipping");
    assert!(results[0].get("model").is_none(), "a result has no model of its own");
    assert_eq!(results[1]["usage"], json!({"input_tokens": 11, "output_tokens": 1}));
    assert_eq!(engine.decided().len(), 2);
}

#[tokio::test]
async fn a_batch_is_checked_whole_before_any_of_it_runs() {
    let engine = deciding();
    let app = app(&engine).await;
    let good = json!({"questions": {"q": {"type": "noul"}}});
    let cases: Vec<(Value, StatusCode, Value, &str)> = vec![
        (
            json!({"requests": []}),
            StatusCode::UNPROCESSABLE_ENTITY,
            json!(["body", "requests"]),
            "requests must be a non-empty list",
        ),
        (
            json!({"requests": "x"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            json!(["body", "requests"]),
            "requests must be a non-empty list",
        ),
        (
            json!({"requests": [good], "state": "s"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            json!(["body"]),
            "a batch cannot also contain state or questions",
        ),
        (
            json!({"requests": vec![good.clone(); 1025]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            json!(["body", "requests"]),
            "requests exceeds 1024 items",
        ),
        (
            json!({"requests": [good, {"questions": {}}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            json!(["body", "requests", 1]),
            "Value error, at least one question is required",
        ),
    ];
    for (body, status, loc, msg) in cases {
        let (got, value) = json(post(&app, SYSTEMONE, &body.to_string()).await).await;
        assert_eq!(got, status, "{value}");
        let (got_loc, got_msg, _) = problem(&value);
        assert_eq!((got_loc, got_msg.as_str()), (loc, msg));
    }
    let body = json!({"requests": [good, {"model": "nope", "questions": {"q": {"type": "noul"}}}]});
    let (status, value) = json(post(&app, SYSTEMONE, &body.to_string()).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{value}");
    assert!(engine.decided().is_empty(), "nothing of a refused batch runs");
}

#[tokio::test]
async fn one_failing_request_of_a_batch_fails_the_batch() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![deliver_with(|request| {
        if request.state == Json::String("bad".into()) {
            return DecisionOutcome::Failed(EngineError::InvalidRequest(
                "questions.q.criteria: 300 options; a choice reads at most 256".into(),
            ));
        }
        favour_the_first(request)
    })]));
    let item = |state: &str| json!({"state": state, "questions": {"q": {"type": "noul"}}});
    let body = json!({"requests": [item("good"), item("bad")]}).to_string();
    let (status, value) = json(post(&app(&engine).await, SYSTEMONE, &body).await).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{value}");
    let (loc, msg, kind) = problem(&value);
    assert_eq!(loc, json!(["body", "requests", 1]));
    assert!(msg.contains("reads at most 256") && kind == "value_error", "{msg}");
}

// --- request validation ---------------------------------------------------------

/// A question body with one thing changed.
fn with_question(question: Value) -> String {
    decision_body(json!({"questions": {"q": question}}))
}

#[tokio::test]
async fn bad_requests_are_refused_with_flides_422_before_the_engine_sees_them() {
    let choice = |criteria: Value| with_question(json!({"type": "choice", "criteria": criteria}));
    let score = |criteria: Value| with_question(json!({"type": "score", "criteria": criteria}));
    let noul = |criteria: Value| with_question(json!({"type": "noul", "criteria": criteria}));
    let q = |more: &[&str]| {
        let mut loc = vec![json!("body"), json!("questions"), json!("q")];
        loc.extend(more.iter().map(|s| json!(s)));
        Value::Array(loc)
    };
    let many: serde_json::Map<String, Value> =
        (0..257).map(|i| (format!("k{i}"), Value::Null)).collect();
    let cases: Vec<(String, Value, &str, &str)> = vec![
        // (body, loc, msg, type)
        ("{not json".into(), json!(["body"]), "invalid JSON", "json_invalid"),
        (String::new(), json!(["body"]), "invalid JSON", "json_invalid"),
        (
            "[1, 2]".into(),
            json!(["body"]),
            "Input should be a valid dictionary or object to extract fields from",
            "model_type",
        ),
        (
            decision_body(json!({"model": 5})),
            json!(["body", "model"]),
            "Input should be a valid string",
            "string_type",
        ),
        (r#"{"state":"s"}"#.into(), json!(["body", "questions"]), "Field required", "missing"),
        (
            decision_body(json!({"questions": "x"})),
            json!(["body", "questions"]),
            "Input should be a valid dictionary",
            "dict_type",
        ),
        (
            decision_body(json!({"questions": {}})),
            json!(["body"]),
            "Value error, at least one question is required",
            "value_error",
        ),
        (
            decision_body(json!({"questions": {"q": 5}})),
            q(&[]),
            "Input should be a valid dictionary or object to extract fields from",
            "model_type",
        ),
        (with_question(json!({"instructions": "x"})), q(&["type"]), "Field required", "missing"),
        (
            with_question(json!({"type": "pick"})),
            q(&["type"]),
            "Input should be 'noul', 'choice' or 'score'",
            "literal_error",
        ),
        (with_question(json!({"type": "choice"})), q(&["criteria"]), "Field required", "missing"),
        (
            choice(json!(["a", "b"])),
            q(&["criteria"]),
            "Input should be a valid dictionary",
            "dict_type",
        ),
        (
            choice(json!({})),
            q(&[]),
            "Value error, choice question needs at least one option in criteria",
            "value_error",
        ),
        (
            choice(Value::Object(many)),
            q(&[]),
            "Value error, choice question has 257 options, maximum is 256",
            "value_error",
        ),
        (with_question(json!({"type": "score"})), q(&["criteria"]), "Field required", "missing"),
        (score(json!({"a": 1})), q(&["criteria"]), "Input should be a valid list", "list_type"),
        (
            score(json!([])),
            q(&[]),
            "Value error, score question needs at least one level in criteria",
            "value_error",
        ),
        (
            score(json!(vec!["l"; 257])),
            q(&[]),
            "Value error, score question has 257 levels, maximum is 256",
            "value_error",
        ),
        (
            noul(json!(["true"])),
            q(&["criteria"]),
            "Input should be a valid dictionary or object to extract fields from",
            "model_type",
        ),
        (
            decision_body(json!({"images": ["data:image/png;base64,AAAA"]})),
            json!(["body"]),
            "images: this model reads text only",
            "value_error",
        ),
    ];
    let engine = deciding();
    let app = app(&engine).await;
    for (body, loc, msg, kind) in cases {
        let (status, value) = json(post(&app, SYSTEMONE, &body).await).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}: {value}");
        assert_eq!(problem(&value), (loc, msg.to_owned(), kind.to_owned()), "{body}");
    }
    assert!(engine.decided().is_empty(), "no invalid request may reach the engine");

    // What a request may leave out or add: no instructions, no criteria for a
    // noul, an empty key, one option, one level, keys this route does not read.
    let fine = decision_body(json!({"debug": true, "questions": {
        "a": {"type": "noul"},
        "b": {"type": "choice", "criteria": {"": null}},
        "c": {"type": "score", "criteria": ["only"], "weight": 3},
    }}));
    assert_eq!(received(&fine).await.questions.len(), 3);
}

#[tokio::test]
async fn the_query_takes_flides_debias_and_debug() {
    let engine = deciding();
    let app = app(&engine).await;
    let body = decision_body(json!({}));
    for query in ["?debias=0", "?debug=true", "?debias=3&debug=off", "?other=x"] {
        let path = format!("{SYSTEMONE}{query}");
        assert_eq!(post(&app, &path, &body).await.status(), StatusCode::OK, "{query}");
    }
    for (query, msg) in [
        ("?debias=x", "debias is a whole number, not 'x'"),
        ("?debug=maybe", "debug is true or false, not 'maybe'"),
    ] {
        let (status, value) = json(post(&app, &format!("{SYSTEMONE}{query}"), &body).await).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(problem(&value), (json!(["query"]), msg.to_owned(), "value_error".to_owned()));
    }
}

#[tokio::test]
async fn an_oversized_body_is_refused_with_413() {
    let engine = deciding();
    let app =
        app_with(&engine, ares_server::Config { max_body_bytes: 64, ..Default::default() }).await;
    let (status, body) = json(post(&app, SYSTEMONE, &decision_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(body["detail"].as_str().unwrap().contains("64"), "{body}");
    assert!(engine.decided().is_empty());
}

// --- routes ------------------------------------------------------------------

#[tokio::test]
async fn the_endpoint_is_on_v1_systemone_and_nowhere_else() {
    let engine = deciding();
    let app = app(&engine).await;
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

/// Every error the engine can give, with the status and the detail this route
/// answers it with.
fn engine_errors() -> Vec<(EngineError, StatusCode, Option<&'static str>, &'static str)> {
    vec![
        (
            EngineError::InvalidRequest("questions.q.criteria: too many".into()),
            StatusCode::UNPROCESSABLE_ENTITY,
            Some("value_error"),
            "too many",
        ),
        (
            EngineError::ContextOverflow { prompt_tokens: 9000, limit: 4096 },
            StatusCode::UNPROCESSABLE_ENTITY,
            Some("value_error"),
            "9000",
        ),
        (EngineError::Overloaded, StatusCode::TOO_MANY_REQUESTS, None, "request queue is full"),
        (
            EngineError::Unsupported("no such architecture".into()),
            StatusCode::NOT_IMPLEMENTED,
            None,
            "no such architecture",
        ),
        (
            EngineError::Internal("kernel exploded".into()),
            StatusCode::INTERNAL_SERVER_ERROR,
            None,
            "kernel exploded",
        ),
    ]
}

async fn assert_engine_error(
    response: axum::http::Response<axum::body::Body>,
    status: StatusCode,
    kind: Option<&str>,
    words: &str,
) {
    let retry = response.headers().get(header::RETRY_AFTER).is_some();
    let (got, value) = json(response).await;
    assert_eq!(got, status, "{value}");
    match kind {
        Some(kind) => {
            let (loc, msg, got_kind) = problem(&value);
            assert_eq!((loc, got_kind.as_str()), (json!(["body"]), kind));
            assert!(msg.contains(words), "{msg}");
        }
        None => assert!(value["detail"].as_str().unwrap().contains(words), "{value}"),
    }
    assert_eq!(retry, status == StatusCode::TOO_MANY_REQUESTS, "Retry-After with a full line");
}

#[tokio::test]
async fn an_error_returned_by_decide_is_answered_as_flide_answers_it() {
    for (error, status, kind, words) in engine_errors() {
        let engine = MockEngine::deciding(DecisionPlan::Reject(error));
        let response = post(&app(&engine).await, SYSTEMONE, &decision_body(json!({}))).await;
        assert_engine_error(response, status, kind, words).await;
    }
}

#[tokio::test]
async fn a_failure_delivered_as_the_outcome_is_answered_the_same_way() {
    for (error, status, kind, words) in engine_errors() {
        let engine = MockEngine::deciding(DecisionPlan::Script(vec![failing(error)]));
        let response = post(&app(&engine).await, SYSTEMONE, &decision_body(json!({}))).await;
        assert_engine_error(response, status, kind, words).await;
        assert!(engine.cancelled().is_empty(), "the engine ended it; there is nothing to cancel");
    }
}

#[tokio::test]
async fn a_decision_the_engine_cancelled_under_a_waiting_client_is_an_error() {
    let engine =
        MockEngine::deciding(DecisionPlan::Script(vec![deliver(DecisionOutcome::Cancelled)]));
    let response = post(&app(&engine).await, SYSTEMONE, &decision_body(json!({}))).await;
    assert_eq!(response.headers()[header::RETRY_AFTER], "1");
    let (status, body) = json(response).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["detail"].as_str().unwrap().contains("cancelled"), "{body}");
    assert!(engine.cancelled().is_empty(), "the engine ended it; there is nothing left to cancel");
}

#[tokio::test]
async fn an_engine_that_drops_the_sink_without_an_outcome_is_reported_not_waited_for() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![DecisionStep::Abandon]));
    let (status, body) =
        json(post(&app(&engine).await, SYSTEMONE, &decision_body(json!({}))).await).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body["detail"].as_str().unwrap().contains("without finishing"));
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
        let outcome = DecisionOutcome::Decided(result(answers, 1));
        let engine = MockEngine::deciding(DecisionPlan::Script(vec![deliver(outcome)]));
        let (status, body) =
            json(post(&app(&engine).await, SYSTEMONE, &decision_body(json!({}))).await).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{words}: {body}");
        let message = body["detail"].as_str().unwrap();
        assert!(message.contains(words), "'{message}' should contain '{words}'");
    }
}

// --- cancellation ----------------------------------------------------------------

#[tokio::test]
async fn a_client_that_leaves_a_decision_cancels_it_in_the_engine() {
    let engine = MockEngine::deciding(DecisionPlan::Script(vec![DecisionStep::UntilCancelled]));
    let app = app(&engine).await;
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
    let app = app(&engine).await;
    for _ in 0..3 {
        assert_eq!(post(&app, SYSTEMONE, &decision_body(json!({}))).await.status(), StatusCode::OK);
    }
    let broken = MockEngine::deciding(DecisionPlan::Script(vec![failing(EngineError::Overloaded)]));
    let _ = post(&common::app(&broken).await, SYSTEMONE, &decision_body(json!({}))).await;
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
    let app = app(&engine).await;

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
    let app = app(&engine).await;

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
    let app = app(&engine).await;
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
