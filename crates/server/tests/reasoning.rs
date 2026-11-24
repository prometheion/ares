//! Thinking over HTTP, against a scripted engine whose model declares what
//! the Qwen3.5 family's templates declare.

mod common;

use std::sync::Arc;

use ares_api::{ChatFormat, FinishReason, Json, Prompt, ThinkingFormat};
use ares_server::options::ModelOptions;
use ares_server::{Config, ReasoningFormat};
use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";

async fn serve(
    steps: Vec<Step>,
    thinking: Option<ThinkingFormat>,
    options: ModelOptions,
) -> (Arc<MockEngine>, axum::Router) {
    let engine = MockEngine::script(steps);
    engine.set_chat_format(ChatFormat { thinking, tools: None });
    let app = router_of(engine.clone(), options, Config::default()).await;
    (engine, app)
}

/// What a model that thinks says: the reasoning, the end of it, the answer.
fn reply() -> Vec<Step> {
    vec![
        text("Two and"),
        text(" two."),
        text("\n</think>\n\n"),
        text("The answer"),
        text(" is 4."),
        done(FinishReason::Stop, 12, 11),
    ]
}

fn chat(extra: Value) -> String {
    merged(json!({"model": "mock", "messages": [{"role": "user", "content": "2 + 2?"}]}), extra)
}

async fn refused(app: &axum::Router, extra: Value) -> Value {
    let (status, body) = json(post(app, CHAT, &chat(extra)).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    body["error"].clone()
}

#[tokio::test]
async fn the_reasoning_of_an_answer_is_returned_apart_from_the_content() {
    let (_, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let (status, body) = json(post(&app, CHAT, &chat(json!({}))).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], "The answer is 4.");
    assert_eq!(message["reasoning_content"], "Two and two.");
    assert_eq!(body["usage"]["completion_tokens"], 11, "the thinking is generated too");
}

#[tokio::test]
async fn the_reasoning_of_a_stream_comes_in_deltas_of_its_own() {
    let (_, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let body = chat(json!({"stream": true, "stream_options": {"include_usage": true}}));
    let stream = parse_sse(&body_text(post(&app, CHAT, &body).await).await);
    assert!(stream.done);
    let deltas: Vec<&Value> =
        stream.chunks.iter().map(|chunk| &chunk["choices"][0]["delta"]).collect();
    assert_eq!(deltas[0], &json!({"role": "assistant", "content": ""}));
    // The reasoning first, a field of its own, then the content.
    assert_eq!(deltas[1], &json!({"reasoning_content": "Two and"}));
    assert_eq!(deltas[2], &json!({"reasoning_content": " two."}));
    assert_eq!(deltas[3], &json!({"content": "The answer"}));
    assert_eq!(deltas[4], &json!({"content": " is 4."}));
    let reasoning: String = deltas.iter().filter_map(|d| d["reasoning_content"].as_str()).collect();
    let content: String = deltas.iter().filter_map(|d| d["content"].as_str()).collect();
    assert_eq!((reasoning.as_str(), content.as_str()), ("Two and two.", "The answer is 4."));
    let finish = &stream.chunks[stream.chunks.len() - 2];
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(finish["choices"][0]["delta"], json!({}));
    assert_eq!(stream.chunks.last().unwrap()["usage"]["completion_tokens"], 11);
}

#[tokio::test]
async fn text_that_may_be_a_tag_waits_for_the_next_piece() {
    let steps =
        vec![text("Fine\n</th"), text("ink>\n"), text("\nSo."), done(FinishReason::Stop, 1, 5)];
    let (_, app) = serve(steps, Some(qwen_thinking()), ModelOptions::default()).await;
    let stream =
        parse_sse(&body_text(post(&app, CHAT, &chat(json!({"stream": true}))).await).await);
    let deltas: Vec<&Value> =
        stream.chunks.iter().map(|chunk| &chunk["choices"][0]["delta"]).skip(1).collect();
    assert_eq!(deltas[0], &json!({"reasoning_content": "Fine"}));
    assert_eq!(deltas[1], &json!({"content": "So."}));
}

#[tokio::test]
async fn a_reply_cut_inside_the_thinking_has_reasoning_and_an_empty_content() {
    let steps = vec![text("Still "), text("thinking\n"), done(FinishReason::Length, 5, 3)];
    let (_, app) = serve(steps, Some(qwen_thinking()), ModelOptions::default()).await;
    let (_, body) = json(post(&app, CHAT, &chat(json!({}))).await).await;
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["choices"][0]["message"]["content"], "");
    assert_eq!(body["choices"][0]["message"]["reasoning_content"], "Still thinking\n");

    // The same streamed: what was held back comes before the finish chunk.
    let steps = vec![text("Still "), text("thinking\n"), done(FinishReason::Length, 5, 3)];
    let (_, app) = serve(steps, Some(qwen_thinking()), ModelOptions::default()).await;
    let stream =
        parse_sse(&body_text(post(&app, CHAT, &chat(json!({"stream": true}))).await).await);
    let deltas: Vec<&Value> =
        stream.chunks.iter().map(|chunk| &chunk["choices"][0]["delta"]).collect();
    assert_eq!(deltas[1], &json!({"reasoning_content": "Still "}));
    assert_eq!(deltas[2], &json!({"reasoning_content": "thinking"}));
    assert_eq!(deltas[3], &json!({"reasoning_content": "\n"}));
    assert_eq!(stream.chunks.last().unwrap()["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn an_answer_without_thinking_has_no_reasoning_field() {
    let steps = vec![text("Four."), done(FinishReason::Stop, 5, 2)];
    let (_, app) = serve(steps.clone(), Some(qwen_thinking()), ModelOptions::default()).await;
    let kwargs = json!({"chat_template_kwargs": {"enable_thinking": false}});
    let (_, body) = json(post(&app, CHAT, &chat(kwargs)).await).await;
    assert_eq!(body["choices"][0]["message"], json!({"role": "assistant", "content": "Four."}));

    // A model whose template has no thinking at all.
    let (_, app) = serve(steps, None, ModelOptions::default()).await;
    let (_, body) = json(post(&app, CHAT, &chat(json!({}))).await).await;
    assert_eq!(body["choices"][0]["message"], json!({"role": "assistant", "content": "Four."}));
}

#[tokio::test]
async fn reasoning_format_none_leaves_the_thinking_in_the_content_tags_and_all() {
    let raw = "<think>\nTwo and two.\n</think>\n\nThe answer is 4.";
    // Per request.
    let (_, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let (_, body) = json(post(&app, CHAT, &chat(json!({"reasoning_format": "none"}))).await).await;
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], raw);
    assert!(message.get("reasoning_content").is_none(), "{message}");

    // As the default of the server, and the request wins over it.
    let none = ModelOptions { reasoning_format: ReasoningFormat::None, ..ModelOptions::default() };
    let (_, app) = serve(reply(), Some(qwen_thinking()), none.clone()).await;
    let (_, body) = json(post(&app, CHAT, &chat(json!({}))).await).await;
    assert_eq!(body["choices"][0]["message"]["content"], raw);
    let (_, body) = json(post(&app, CHAT, &chat(json!({"reasoning_format": "auto"}))).await).await;
    assert_eq!(body["choices"][0]["message"]["reasoning_content"], "Two and two.");

    // Streamed, the raw text is the content deltas.
    let (_, app) = serve(reply(), Some(qwen_thinking()), none).await;
    let stream =
        parse_sse(&body_text(post(&app, CHAT, &chat(json!({"stream": true}))).await).await);
    let content: String = stream
        .chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, raw);
}

#[tokio::test]
async fn the_variables_of_a_request_reach_the_chat_template() {
    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let body = chat(json!({
        "chat_template_kwargs": {"enable_thinking": true, "depth": 3},
        "reasoning_effort": "low",
    }));
    assert_eq!(post(&app, CHAT, &body).await.status(), StatusCode::OK);
    let Prompt::Chat { variables, .. } = engine.submitted().remove(0).prompt else { panic!() };
    let get =
        |name: &str| variables.iter().find(|(known, _)| known == name).map(|(_, v)| v.clone());
    assert_eq!(get("enable_thinking"), Some(Json::Bool(true)));
    assert_eq!(get("depth"), Some(Json::Int(3)));
    assert_eq!(get("reasoning_effort"), Some(Json::String("low".into())));

    // `reasoning_effort: none` switches thinking off and is not an effort.
    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    assert_eq!(
        post(&app, CHAT, &chat(json!({"reasoning_effort": "none"}))).await.status(),
        StatusCode::OK
    );
    let Prompt::Chat { variables, .. } = engine.submitted().remove(0).prompt else { panic!() };
    assert_eq!(variables, vec![("enable_thinking".to_owned(), Json::Bool(false))]);

    // A request that says nothing adds nothing.
    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    post(&app, CHAT, &chat(json!({}))).await;
    let Prompt::Chat { variables, .. } = engine.submitted().remove(0).prompt else { panic!() };
    assert!(variables.is_empty());
}

#[tokio::test]
async fn the_reasoning_of_an_earlier_answer_goes_back_into_the_prompt() {
    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let body = json!({
        "model": "mock",
        "messages": [
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "Two and two."},
            {"role": "assistant", "content": null, "reasoning_content": null},
            {"role": "user", "content": "And 3 + 3?"},
        ],
    });
    assert_eq!(post(&app, CHAT, &body.to_string()).await.status(), StatusCode::OK);
    let Prompt::Chat { messages, .. } = engine.submitted().remove(0).prompt else { panic!() };
    assert_eq!(messages[1].reasoning_content.as_deref(), Some("Two and two."));
    assert_eq!(messages[0].reasoning_content, None);
    assert_eq!(messages[2].reasoning_content, None, "null is not set");

    let wrong = json!({
        "model": "mock",
        "messages": [{"role": "assistant", "content": "4", "reasoning_content": 4}],
    });
    let (status, body) = json(post(&app, CHAT, &wrong.to_string()).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "messages[0].reasoning_content");
}

#[tokio::test]
async fn what_the_template_cannot_do_is_refused_by_name() {
    let (engine, app) = serve(
        reply(),
        Some(ThinkingFormat { effort: false, ..qwen_thinking() }),
        ModelOptions::default(),
    )
    .await;
    let error = refused(&app, json!({"reasoning_effort": "low"})).await;
    assert_eq!(error["param"], "reasoning_effort");
    assert!(error["message"].as_str().unwrap().contains("does not use it"), "{error}");

    let stuck = ThinkingFormat { prompt_off: "<think>\n".into(), ..qwen_thinking() };
    let (_, app) = serve(reply(), Some(stuck), ModelOptions::default()).await;
    let error = refused(&app, json!({"reasoning_effort": "none"})).await;
    assert!(error["message"].as_str().unwrap().contains("cannot switch"), "{error}");

    let (_, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let error = refused(&app, json!({"chat_template_kwargs": {"enable_thinking": "no"}})).await;
    assert!(error["message"].as_str().unwrap().contains("must be a boolean"), "{error}");
    let error = refused(&app, json!({"chat_template_kwargs": {"messages": []}})).await;
    assert!(error["message"].as_str().unwrap().contains("set by the server"), "{error}");
    let error = refused(&app, json!({"reasoning_format": "think"})).await;
    assert_eq!(error["param"], "reasoning_format");

    // A budget for the thinking reaches the engine under every name it has;
    // the realtime control is not built.
    for field in ["reasoning_budget_tokens", "thinking_budget_tokens", "thinking_token_budget"] {
        let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
        let body = chat(json!({ field: 256, "reasoning_budget_message": "Time is up." }));
        assert_eq!(post(&app, CHAT, &body).await.status(), StatusCode::OK);
        let sampling = engine.submitted().remove(0).sampling;
        assert_eq!(sampling.reasoning_budget_tokens, Some(256), "{field}");
        assert_eq!(sampling.reasoning_budget_message.as_deref(), Some("Time is up."));
    }
    let (_, plain) = serve(reply(), None, ModelOptions::default()).await;
    let error = refused(&plain, json!({"reasoning_budget_tokens": 256})).await;
    assert!(error["message"].as_str().unwrap().contains("model that thinks"), "{error}");
    let error = refused(&app, json!({"reasoning_control": true})).await;
    assert_eq!(error["param"], "reasoning_control");
    // No budget is not a refusal.
    let quiet = chat(json!({
        "reasoning_budget_tokens": -1, "thinking_budget_tokens": -1,
        "reasoning_budget_message": "", "reasoning_control": false,
    }));
    assert_eq!(post(&app, CHAT, &quiet).await.status(), StatusCode::OK);
    // None of the refusals reached the engine.
    assert!(engine.submitted().is_empty());
}

#[tokio::test]
async fn the_completions_endpoint_has_no_thinking_to_ask_for() {
    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    for field in
        ["reasoning_format", "reasoning_effort", "chat_template_kwargs", "reasoning_budget_tokens"]
    {
        let body = json!({"model": "mock", "prompt": "Once", field: "x"}).to_string();
        let (status, body) = json(post(&app, "/v1/completions", &body).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field}: {body}");
        assert_eq!(body["error"]["param"], field);
        assert!(body["error"]["message"].as_str().unwrap().contains("chat template"), "{body}");
    }
    assert!(engine.submitted().is_empty());

    // A text is a text: the tags in it are not looked at.
    let (_, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let body = json!({"model": "mock", "prompt": "Once"}).to_string();
    let (_, body) = json(post(&app, "/v1/completions", &body).await).await;
    assert_eq!(body["choices"][0]["text"], "Two and two.\n</think>\n\nThe answer is 4.");
}

/// An effort the template refuses: `high` is the template's default, as the
/// Messages route takes Anthropic's `high` (Flash-Next's template knows
/// `xhigh`, `medium` and `low`); another one is refused by name.
#[tokio::test]
async fn an_effort_the_template_refuses_is_its_default_for_high_and_refused_else() {
    let flash = ThinkingFormat {
        refused_efforts: vec!["minimal".into(), "high".into()],
        ..qwen_thinking()
    };
    let (engine, app) = serve(reply(), Some(flash.clone()), ModelOptions::default()).await;
    let (status, body) =
        json(post(&app, CHAT, &chat(json!({"reasoning_effort": "high"}))).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let Prompt::Chat { variables, .. } = engine.submitted().remove(0).prompt else { panic!() };
    assert!(variables.iter().all(|(name, _)| name != "reasoning_effort"), "{variables:?}");

    let (engine, app) = serve(reply(), Some(flash), ModelOptions::default()).await;
    let (status, _) = json(post(&app, CHAT, &chat(json!({"reasoning_effort": "low"}))).await).await;
    assert_eq!(status, StatusCode::OK);
    let Prompt::Chat { variables, .. } = engine.submitted().remove(0).prompt else { panic!() };
    assert!(variables.contains(&("reasoning_effort".to_owned(), Json::String("low".into()))));
    let error = refused(&app, json!({"reasoning_effort": "minimal"})).await;
    assert_eq!(error["param"], "reasoning_effort");
    assert!(error["message"].as_str().unwrap().contains("refuses minimal, high"), "{error}");

    // A template that takes `high` gets it.
    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let (status, _) =
        json(post(&app, CHAT, &chat(json!({"reasoning_effort": "high"}))).await).await;
    assert_eq!(status, StatusCode::OK);
    let Prompt::Chat { variables, .. } = engine.submitted().remove(0).prompt else { panic!() };
    assert!(variables.contains(&("reasoning_effort".to_owned(), Json::String("high".into()))));
}

/// A system or developer message inside a conversation is a message of the
/// user on the chat and Responses routes, as on the Messages route; the ones
/// before every other message are one system message.
#[tokio::test]
async fn system_and_developer_messages_are_placed_as_on_the_messages_route() {
    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let body = json!({"model": "mock", "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "developer", "content": "Answer in German."},
        {"role": "user", "content": "2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "developer", "content": "Now in English."},
        {"role": "user", "content": "3 + 3?"},
    ]});
    let (status, body) = json(post(&app, CHAT, &body.to_string()).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let Prompt::Chat { messages, .. } = engine.submitted().remove(0).prompt else { panic!() };
    let turns: Vec<(&str, &str)> =
        messages.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect();
    assert_eq!(
        turns,
        [
            ("system", "Be brief.\n\nAnswer in German."),
            ("user", "2 + 2?"),
            ("assistant", "4"),
            ("user", "Now in English."),
            ("user", "3 + 3?"),
        ]
    );

    let (engine, app) = serve(reply(), Some(qwen_thinking()), ModelOptions::default()).await;
    let body = json!({"model": "mock", "instructions": "Be brief.", "input": [
        {"role": "developer", "content": "Answer in German."},
        {"role": "user", "content": "2 + 2?"},
        {"role": "system", "content": "Now in English."},
    ]});
    let (status, body) = json(post(&app, "/v1/responses", &body.to_string()).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let Prompt::Chat { messages, .. } = engine.submitted().remove(0).prompt else { panic!() };
    let turns: Vec<(&str, &str)> =
        messages.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect();
    assert_eq!(
        turns,
        [
            ("system", "Be brief.\n\nAnswer in German."),
            ("user", "2 + 2?"),
            ("user", "Now in English.")
        ]
    );
}
