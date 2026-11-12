//! Structured output in requests, against a scripted engine: what `grammar`,
//! `json_schema` and `response_format` come to when they reach the engine, which
//! combinations are refused, and where the grammar starts for a model that
//! thinks.

mod common;

use std::sync::Arc;

use ares_api::{
    ChatFormat, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, EventSink,
    FinishReason, GenerateRequest, ModelCard, RequestId, Shape, Start, Structure, ThinkingFormat,
};
use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";
const COMPLETIONS: &str = "/v1/completions";

/// The scripted engine, with a model that declares `format`.
struct Declaring {
    inner: Arc<MockEngine>,
    format: ChatFormat,
}

impl Engine for Declaring {
    fn models(&self) -> Vec<ModelCard> {
        self.inner.models()
    }

    fn chat_format(&self, model: &str) -> Option<ChatFormat> {
        (model == "mock").then(|| self.format.clone())
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        self.inner.submit(request, events)
    }

    fn decide(&self, request: DecisionRequest, outcome: DecisionSink) -> Result<(), EngineError> {
        self.inner.decide(request, outcome)
    }

    fn cancel(&self, id: RequestId) {
        self.inner.cancel(id);
    }

    fn stats(&self) -> Option<EngineStats> {
        self.inner.stats()
    }
}

fn qwen() -> ThinkingFormat {
    ThinkingFormat {
        start: "<think>\n".into(),
        end: "\n</think>\n\n".into(),
        prompt_on: "<think>\n".into(),
        prompt_off: "<think>\n\n</think>\n\n".into(),
        on_by_default: true,
        effort: true,
    }
}

/// A template that writes nothing into the prompt when the model may think
/// (Qwen3's): the model opens the block itself.
fn qwen3() -> ThinkingFormat {
    ThinkingFormat { prompt_on: String::new(), ..qwen() }
}

fn instant() -> Arc<MockEngine> {
    MockEngine::script(vec![done(FinishReason::Stop, 1, 0)])
}

fn chat(extra: Value) -> String {
    let mut body = json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body.to_string()
}

fn completion(extra: Value) -> String {
    let mut body = json!({"model": "mock", "prompt": "Once upon"});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body.to_string()
}

async fn structure_of(
    path: &str,
    body: &str,
    thinking: Option<ThinkingFormat>,
) -> Option<Structure> {
    let inner = instant();
    let engine = Declaring { inner: inner.clone(), format: ChatFormat { thinking, tools: None } };
    let app = ares_server::router_with(Arc::new(engine), ares_server::Config::default());
    let (status, value) = json(post(&app, path, body).await).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    inner.submitted().remove(0).structure
}

async fn refused(path: &str, body: &str) -> Value {
    let engine = instant();
    let (status, value) = json(post(&app(&engine), path, body).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
    assert!(engine.submitted().is_empty(), "the engine was not to see the request");
    value["error"].clone()
}

fn gbnf(text: &str) -> Option<Structure> {
    Some(Structure::new(Shape::Gbnf(text.to_owned())))
}

fn schema(value: Value) -> Option<Structure> {
    Some(Structure::new(Shape::JsonSchema(value.to_string())))
}

#[tokio::test]
async fn a_grammar_reaches_the_engine_on_chat_and_on_completions() {
    let rule = r#"root ::= "yes" | "no""#;
    let got = structure_of(CHAT, &chat(json!({"grammar": rule})), None).await;
    assert_eq!(got, gbnf(rule));
    let got = structure_of(COMPLETIONS, &completion(json!({"grammar": rule})), None).await;
    assert_eq!(got, gbnf(rule));
    // An empty grammar asks for nothing, as it does for llama-server.
    let got = structure_of(CHAT, &chat(json!({"grammar": ""})), None).await;
    assert_eq!(got, None);
}

#[tokio::test]
async fn a_json_schema_reaches_the_engine_as_text_and_an_empty_one_is_any_object() {
    let s = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]});
    for (path, body) in [
        (CHAT, chat(json!({"json_schema": s}))),
        (COMPLETIONS, completion(json!({"json_schema": s}))),
    ] {
        assert_eq!(structure_of(path, &body, None).await, schema(s.clone()), "{path}");
    }
    let got = structure_of(CHAT, &chat(json!({"json_schema": {}})), None).await;
    assert_eq!(got, schema(json!({"type": "object"})));
}

#[tokio::test]
async fn the_response_format_of_openai_and_llama_server_is_read_as_llama_server_reads_it() {
    let s = json!({"type": "object", "properties": {"n": {"type": "number"}}});
    // OpenAI's form, with the name and strict that say nothing more.
    let format =
        json!({"type": "json_schema", "json_schema": {"name": "n", "strict": true, "schema": s}});
    assert_eq!(
        structure_of(CHAT, &chat(json!({"response_format": format})), None).await,
        schema(s.clone())
    );
    let format =
        json!({"type": "json_schema", "json_schema": {"name": "n", "strict": false, "schema": s}});
    assert_eq!(
        structure_of(CHAT, &chat(json!({"response_format": format})), None).await,
        schema(s.clone())
    );
    // Without a schema member it is any object, which llama-server does too.
    let format = json!({"type": "json_schema"});
    assert_eq!(
        structure_of(CHAT, &chat(json!({"response_format": format})), None).await,
        schema(json!({"type": "object"}))
    );
    // json_object: any object, or the schema llama-server's extension carries.
    let format = json!({"type": "json_object"});
    assert_eq!(
        structure_of(CHAT, &chat(json!({"response_format": format})), None).await,
        schema(json!({"type": "object"}))
    );
    let format = json!({"type": "json_object", "schema": s});
    assert_eq!(
        structure_of(CHAT, &chat(json!({"response_format": format})), None).await,
        schema(s.clone())
    );
    // Text asks for nothing.
    let format = json!({"type": "text"});
    assert_eq!(structure_of(CHAT, &chat(json!({"response_format": format})), None).await, None);
    // The same on completions.
    let format = json!({"type": "json_object"});
    assert_eq!(
        structure_of(COMPLETIONS, &completion(json!({"response_format": format})), None).await,
        schema(json!({"type": "object"}))
    );
}

#[tokio::test]
async fn requests_that_ask_for_two_things_or_a_thing_that_is_wrong_are_refused_by_name() {
    let both = refused(CHAT, &chat(json!({"grammar": "root ::= \"a\"", "json_schema": {}}))).await;
    assert_eq!(both["param"], "json_schema");
    assert!(both["message"].as_str().unwrap().contains("grammar"), "{both}");
    let both = refused(
        COMPLETIONS,
        &completion(
            json!({"grammar": "root ::= \"a\"", "response_format": {"type": "json_object"}}),
        ),
    )
    .await;
    assert_eq!(both["param"], "response_format");
    let kind = refused(CHAT, &chat(json!({"response_format": {"type": "xml"}}))).await;
    assert_eq!(kind["param"], "response_format");
    assert_eq!(
        kind["message"],
        "response_format type must be one of \"text\", \"json_object\" or \"json_schema\", but got: xml"
    );
    for (extra, param) in [
        (json!({"grammar": 5}), "grammar"),
        (json!({"json_schema": "{}"}), "json_schema"),
        (json!({"response_format": "json"}), "response_format"),
        (json!({"response_format": {"type": 3}}), "response_format.type"),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": []}}),
            "response_format.json_schema",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema": 3}}}),
            "response_format.json_schema.schema",
        ),
        (
            json!({"response_format": {"type": "json_object", "schema": "x"}}),
            "response_format.schema",
        ),
    ] {
        let error = refused(CHAT, &chat(extra.clone())).await;
        assert_eq!(error["param"], param, "{extra}: {error}");
    }
}

#[tokio::test]
async fn what_the_engine_says_about_a_grammar_is_a_400_that_names_it() {
    let engine = MockEngine::reject(EngineError::InvalidRequest(
        "grammar: grammar, line 1, column 10: the rule 'a' is used and never defined".into(),
    ));
    let (status, body) =
        json(post(&app(&engine), CHAT, &chat(json!({"grammar": "root ::= a"}))).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("never defined"), "{body}");
}

#[tokio::test]
async fn a_model_that_thinks_gets_its_grammar_after_the_reasoning() {
    let rule = r#"root ::= "yes""#;
    let body = chat(json!({"grammar": rule}));
    let after = |open: Option<&str>| {
        Some(Structure {
            shape: Shape::Gbnf(rule.into()),
            start: Start::AfterReasoning {
                open: open.map(str::to_owned),
                close: "</think>".into(),
            },
            trigger: None,
        })
    };
    // The prompt opened the block: the reply is reasoning from its first token.
    assert_eq!(structure_of(CHAT, &body, Some(qwen())).await, after(None));
    // The prompt wrote no tag: the reply may open the block itself.
    assert_eq!(structure_of(CHAT, &body, Some(qwen3())).await, after(Some("<think>")));
    // The reasoning format does not change where the answer starts.
    let raw = chat(json!({"grammar": rule, "reasoning_format": "none"}));
    assert_eq!(structure_of(CHAT, &raw, Some(qwen())).await, after(None));
    // Thinking switched off: the prompt closed the block, the answer is the whole reply.
    let off = chat(json!({"grammar": rule, "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(structure_of(CHAT, &off, Some(qwen())).await, gbnf(rule));
    let off = chat(json!({"grammar": rule, "reasoning_effort": "none"}));
    assert_eq!(structure_of(CHAT, &off, Some(qwen())).await, gbnf(rule));
    // A template without thinking, and a completion: no reasoning to wait for.
    assert_eq!(structure_of(CHAT, &body, None).await, gbnf(rule));
    let plain = completion(json!({"grammar": rule}));
    assert_eq!(structure_of(COMPLETIONS, &plain, Some(qwen())).await, gbnf(rule));
}
