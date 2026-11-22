//! The Anthropic Messages API against a scripted engine: what a request comes
//! to when it reaches the engine, how a reply comes back, whole and as
//! Anthropic's events, what is refused, and the count of a prompt.

mod common;

use std::sync::Arc;

use ares_api::{
    ChatFormat, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, Event, EventSink,
    FinishReason, GenerateRequest, Json, ModelCard, Prompt, RequestId, Role, Shape, Start,
    ThinkingFormat, ToolFormat, Usage,
};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use common::*;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tower::ServiceExt;

const MESSAGES: &str = "/v1/messages";
const COUNT: &str = "/v1/messages/count_tokens";

/// What stands for the tokens of a prompt in these tests: the bytes of the
/// prompt as the engine was given it.
fn tokens(prompt: &Prompt) -> u32 {
    serde_json::to_string(prompt).unwrap().len() as u32
}

/// The scripted engine, with a model that declares `format` and counts.
struct Declaring {
    inner: Arc<MockEngine>,
    format: Option<ChatFormat>,
}

impl Engine for Declaring {
    fn models(&self) -> Vec<ModelCard> {
        self.inner.models()
    }

    fn chat_format(&self, model: &str) -> Option<ChatFormat> {
        (model == "mock").then(|| self.format.clone()).flatten()
    }

    fn count_tokens(&self, model: &str, prompt: &Prompt) -> Result<u32, EngineError> {
        if model != "mock" {
            return Err(EngineError::ModelNotFound(model.to_owned()));
        }
        Ok(tokens(prompt))
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

/// What the template of Qwen3.8 declares.
fn qwen() -> ChatFormat {
    ChatFormat {
        thinking: Some(ThinkingFormat {
            start: "<think>\n".into(),
            end: "\n</think>\n\n".into(),
            prompt_on: "<think>\n".into(),
            prompt_off: "<think>\n\n</think>\n\n".into(),
            on_by_default: true,
            effort: true,
        }),
        tools: Some(ToolFormat {
            call_open: "<tool_call>\n<function=".into(),
            name_close: ">\n".into(),
            param_open: "<parameter=".into(),
            key_close: ">\n".into(),
            param_close: "\n</parameter>\n".into(),
            call_close: "</function>\n</tool_call>".into(),
            before_calls: "\n\n".into(),
            between_calls: "\n".into(),
        }),
    }
}

fn serve_with(steps: Vec<Step>, format: Option<ChatFormat>) -> (Arc<MockEngine>, Router) {
    let inner = MockEngine::script(steps);
    let engine = Declaring { inner: inner.clone(), format };
    (inner, ares_server::router_with(Arc::new(engine), ares_server::Config::default()))
}

fn serve(steps: Vec<Step>) -> (Arc<MockEngine>, Router) {
    serve_with(steps, Some(qwen()))
}

fn tools() -> Value {
    json!([
        {
            "name": "get_weather",
            "description": "The weather of a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
        {"name": "now", "input_schema": {"type": "object"}},
    ])
}

fn body(extra: Value) -> String {
    let mut body = json!({
        "model": "mock",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "How is the weather in Köln?"}],
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body.to_string()
}

/// The reply of a model in pieces of `size` characters, ended by `finish`.
fn says(reply: &str, size: usize, finish: FinishReason) -> Vec<Step> {
    let chars: Vec<char> = reply.chars().collect();
    let mut steps: Vec<Step> =
        chars.chunks(size).map(|piece| text(&piece.iter().collect::<String>())).collect();
    steps.push(done(finish, 20, chars.len() as u32));
    steps
}

const THINKS: &str = "The user asks for the weather.\n</think>\n\n";
const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nKöln\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
const NOW: &str = "<tool_call>\n<function=now>\n</function>\n</tool_call>";

/// The request the engine gets for a body.
async fn submitted(extra: Value) -> GenerateRequest {
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
    let (status, answer) = json(post(&app, MESSAGES, &body(extra)).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    engine.submitted().remove(0)
}

/// The message a body is answered with.
async fn whole(steps: Vec<Step>, extra: Value) -> Value {
    let (_, app) = serve(steps);
    let (status, message) = json(post(&app, MESSAGES, &body(extra)).await).await;
    assert_eq!(status, StatusCode::OK, "{message}");
    message
}

/// The events of an SSE body: the name of each and its data.
fn events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter(|event| !event.trim().is_empty())
        .map(|event| {
            let line = |prefix: &str| {
                let mut lines = event.lines().filter_map(|l| l.strip_prefix(prefix));
                let line = lines.next().unwrap_or_else(|| panic!("no {prefix:?} in {event:?}"));
                assert!(lines.next().is_none(), "two {prefix:?} lines in {event:?}");
                line.to_owned()
            };
            let data: Value = serde_json::from_str(&line("data: ")).unwrap();
            (line("event: "), data)
        })
        .collect()
}

/// The events a body is answered with when it is streamed.
async fn streamed(steps: Vec<Step>, extra: Value) -> Vec<(String, Value)> {
    let (_, app) = serve(steps);
    let mut extra = extra;
    extra["stream"] = json!(true);
    let response = post(&app, MESSAGES, &body(extra)).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "text/event-stream");
    let text = body_text(response).await;
    assert!(!text.contains("[DONE]"), "this API has no end marker: {text}");
    events(&text)
}

/// Puts the events of a stream together into the message they describe, the
/// way a client does, and checks their order on the way.
fn assemble(events: &[(String, Value)]) -> Value {
    for (name, data) in events {
        assert_eq!(data["type"], name.as_str(), "the event is named as its data says");
    }
    let (first, rest) = events.split_first().expect("a first event");
    assert_eq!(first.0, "message_start");
    let mut message = first.1["message"].clone();
    assert_eq!(message["content"], json!([]));
    assert_eq!((&message["stop_reason"], &message["stop_sequence"]), (&Value::Null, &Value::Null));
    assert_eq!(rest.last().expect("a last event").0, "message_stop");

    let mut content: Vec<Value> = Vec::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut open: Option<usize> = None;
    let mut ended = false;
    for (name, data) in &rest[..rest.len() - 1] {
        assert!(!ended, "{name} after message_delta");
        let index = data["index"].as_u64().map(|index| index as usize);
        match name.as_str() {
            "content_block_start" => {
                assert_eq!(open, None, "a block opens while another is open");
                assert_eq!(index, Some(content.len()), "the blocks are numbered in order");
                open = index;
                content.push(data["content_block"].clone());
                inputs.push(String::new());
            }
            "content_block_delta" => {
                assert_eq!(index, open, "a delta of the block that is open");
                let block = &mut content[index.unwrap()];
                let delta = &data["delta"];
                let (kind, member, value) = match delta["type"].as_str().unwrap() {
                    "thinking_delta" => ("thinking", "thinking", &delta["thinking"]),
                    "signature_delta" => ("thinking", "signature", &delta["signature"]),
                    "text_delta" => ("text", "text", &delta["text"]),
                    "input_json_delta" => {
                        assert_eq!(block["type"], "tool_use");
                        inputs[index.unwrap()].push_str(delta["partial_json"].as_str().unwrap());
                        continue;
                    }
                    other => panic!("a delta of the type {other}"),
                };
                assert_eq!(block["type"], kind, "{delta}");
                let so_far = block[member].as_str().unwrap().to_owned();
                block[member] = json!(so_far + value.as_str().unwrap());
            }
            "content_block_stop" => {
                assert_eq!(index, open, "the block that is open closes");
                let index = open.take().unwrap();
                if content[index]["type"] == "tool_use" {
                    assert_eq!(content[index]["input"], json!({}), "the input starts empty");
                    // What the SDKs do with a fragment that is no whole JSON.
                    content[index]["input"] =
                        serde_json::from_str(&inputs[index]).unwrap_or_else(|_| json!({}));
                }
            }
            "message_delta" => {
                assert_eq!(open, None, "the message ends while a block is open");
                message["stop_reason"] = data["delta"]["stop_reason"].clone();
                message["stop_sequence"] = data["delta"]["stop_sequence"].clone();
                message["usage"] = data["usage"].clone();
                ended = true;
            }
            other => panic!("an event named {other}"),
        }
    }
    assert!(ended, "no message_delta");
    message["content"] = Value::Array(content);
    message
}

/// A message without what differs between two requests: its id, and the id
/// of the request in the ids of its calls.
fn anonymous(mut message: Value) -> Value {
    let id = message["id"].as_str().expect("an id").to_owned();
    let request = id.strip_prefix("msg_").expect("the id of a message");
    for block in message["content"].as_array_mut().unwrap() {
        if block["type"] == "tool_use" {
            let call = block["id"].as_str().unwrap();
            let place = call
                .strip_prefix(&format!("toolu_{request}_"))
                .unwrap_or_else(|| panic!("the id {call} is not of the request {request}"));
            block["id"] = json!(place);
        }
    }
    message["id"] = Value::Null;
    message
}

// --- the request ---------------------------------------------------------------

#[tokio::test]
async fn a_plain_request_is_the_chat_request_and_its_reply_one_text_block() {
    // A model whose template has no thinking: the reply is its text.
    let steps = vec![text("Mild, "), text("12 °C."), done(FinishReason::Stop, 9, 5)];
    let (engine, app) = serve_with(steps, None);
    // The headers of Anthropic's clients are accepted, and so is the query
    // Claude Code adds.
    let request = Request::post(format!("{MESSAGES}?beta=true"))
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-api-key", "sk-ant-none")
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "interleaved-thinking-2025-05-14")
        .body(Body::from(body(json!({"temperature": 0.2, "top_p": 0.9, "top_k": 40}))))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let (status, message) = json(response).await;
    assert_eq!(status, StatusCode::OK, "{message}");
    let id = message["id"].as_str().unwrap();
    assert!(id.starts_with("msg_") && id.len() == 20, "{id}");
    assert_eq!(
        anonymous(message),
        json!({
            "id": null,
            "type": "message",
            "role": "assistant",
            "model": "mock",
            "content": [{"type": "text", "text": "Mild, 12 °C."}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 9,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 5,
            },
        })
    );

    let request = engine.submitted().remove(0);
    assert_eq!((request.model.as_str(), request.max_tokens), ("mock", Some(256)));
    assert_eq!(request.sampling.temperature, Some(0.2));
    assert_eq!((request.sampling.top_p, request.sampling.top_k), (Some(0.9), Some(40)));
    assert!(request.draft && request.structure.is_none() && request.stop.is_empty());
    let Prompt::Chat { messages, tools, variables, .. } = &request.prompt else { panic!() };
    assert_eq!(messages.len(), 1);
    assert_eq!(
        (messages[0].role, messages[0].content.as_str()),
        (Role::User, "How is the weather in Köln?")
    );
    // Thinking is left to the server when the request does not say.
    assert!(tools.is_none() && variables.is_empty(), "{variables:?}");
}

#[tokio::test]
async fn system_blocks_and_the_turns_of_a_conversation_reach_the_template() {
    let request = submitted(json!({
        "system": [
            {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Answer in German."},
        ],
        "tools": tools(),
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Weather in Köln?"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I look it up.", "signature": "unsigned"},
                {"type": "text", "text": "One moment."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Köln", "days": 3}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                    {"type": "text", "text": "12 °C"},
                ]},
                {"type": "text", "text": "And tomorrow?"},
            ]},
        ],
    }))
    .await;
    let Prompt::Chat { messages, .. } = &request.prompt else { panic!() };
    let roles: Vec<Role> = messages.iter().map(|m| m.role).collect();
    assert_eq!(roles, [Role::System, Role::User, Role::Assistant, Role::Tool, Role::User]);
    assert_eq!(messages[0].content, "You are terse.\nAnswer in German.");
    assert_eq!(messages[1].content, "Weather in Köln?");
    let turn = &messages[2];
    assert_eq!(turn.content, "One moment.");
    assert_eq!(turn.reasoning_content.as_deref(), Some("I look it up."));
    assert_eq!(
        serde_json::to_value(turn.tool_calls.as_ref().unwrap()).unwrap(),
        json!([{"id": "toolu_1", "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Köln", "days": 3}}}])
    );
    assert_eq!(
        (messages[3].content.as_str(), messages[3].tool_call_id.as_deref()),
        ("12 °C", Some("toolu_1"))
    );
    assert_eq!(messages[4].content, "And tomorrow?");
}

#[tokio::test]
async fn tools_reach_the_template_and_the_reply_gets_the_rule_of_the_tool_choice() {
    let after_reasoning = Start::AfterReasoning { open: None, close: "</think>".into() };
    let lark = |request: &GenerateRequest| match &request.structure.as_ref().unwrap().shape {
        Shape::Lark(text) => text.clone(),
        other => panic!("not a grammar of calls: {other:?}"),
    };

    // auto, which is the default: free until the trigger, then calls.
    for extra in [json!({}), json!({"tool_choice": {"type": "auto"}})] {
        let mut extra = extra;
        extra["tools"] = tools();
        let request = submitted(extra).await;
        let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
        assert_eq!(
            serde_json::to_value(sent).unwrap()[0],
            json!({"type": "function", "function": {
                "name": "get_weather",
                "description": "The weather of a city.",
                "parameters": tools()[0]["input_schema"],
            }})
        );
        let structure = request.structure.as_ref().unwrap();
        assert_eq!(structure.trigger.as_deref(), Some("<tool_call>"));
        assert_eq!(structure.start, after_reasoning, "the reasoning is free");
        assert!(lark(&request).contains("more: (WS? next)* WS?\n"), "several calls");
    }

    // any: calls from the first token of the answer.
    let request = submitted(json!({"tools": tools(), "tool_choice": {"type": "any"}})).await;
    let structure = request.structure.as_ref().unwrap();
    assert_eq!((structure.trigger.as_deref(), &structure.start), (None, &after_reasoning));
    assert!(lark(&request).contains("<function=now>"));

    // tool: calls of that tool, and one of them when the request says so.
    let named = json!({"type": "tool", "name": "get_weather", "disable_parallel_tool_use": true});
    let request = submitted(json!({"tools": tools(), "tool_choice": named})).await;
    let grammar = lark(&request);
    assert!(grammar.contains("<function=get_weather>") && !grammar.contains("<function=now>"));
    assert!(!grammar.contains("(WS? next)*"), "one call: {grammar}");
    let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
    assert_eq!(sent.as_ref().unwrap().len(), 2, "the prompt lists every tool");

    // none: the tools are in the prompt and the reply is text.
    let request = submitted(json!({"tools": tools(), "tool_choice": {"type": "none"}})).await;
    let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
    assert!(sent.is_some() && request.structure.is_none());
}

#[tokio::test]
async fn thinking_is_switched_by_the_request_and_comes_back_as_a_block() {
    let variables = |request: &GenerateRequest| match &request.prompt {
        Prompt::Chat { variables, .. } => variables.clone(),
        Prompt::Text(_) => panic!(),
    };
    let on = vec![("enable_thinking".to_owned(), Json::Bool(true))];
    let enabled = json!({"thinking": {"type": "enabled", "budget_tokens": 4096}});
    assert_eq!(variables(&submitted(enabled.clone()).await), on);
    assert_eq!(variables(&submitted(json!({"thinking": {"type": "adaptive"}})).await), on);
    let off = submitted(json!({"thinking": {"type": "disabled"}})).await;
    assert_eq!(variables(&off), [("enable_thinking".to_owned(), Json::Bool(false))]);

    // A model that thinks: the reasoning is a block of its own, signed with
    // the text that says there is no signature.
    let reply = format!("{THINKS}Mild.");
    let message = whole(says(&reply, 5, FinishReason::Stop), enabled).await;
    assert_eq!(
        message["content"],
        json!([
            {"type": "thinking", "thinking": "The user asks for the weather.", "signature": "unsigned"},
            {"type": "text", "text": "Mild."},
        ])
    );
    // Thinking off: the template has closed the block, the reply is the answer.
    let message =
        whole(says("Mild.", 2, FinishReason::Stop), json!({"thinking": {"type": "disabled"}}))
            .await;
    assert_eq!(message["content"], json!([{"type": "text", "text": "Mild."}]));
    // The limit inside the thinking: a thinking block and nothing else.
    let message = whole(says("The user asks", 4, FinishReason::Length), json!({})).await;
    assert_eq!(message["stop_reason"], "max_tokens");
    assert_eq!(message["content"].as_array().unwrap().len(), 1);
    assert_eq!(message["content"][0]["thinking"], "The user asks");
}

#[tokio::test]
async fn the_reasoning_is_a_block_whatever_the_format_of_the_chat_route() {
    // `--reasoning-format none` is about reasoning_content.
    let inner = MockEngine::script(says(&format!("{THINKS}Mild."), 7, FinishReason::Stop));
    let engine = Declaring { inner, format: Some(qwen()) };
    let config = ares_server::Config {
        reasoning_format: ares_server::ReasoningFormat::None,
        ..ares_server::Config::default()
    };
    let app = ares_server::router_with(Arc::new(engine), config);
    let (_, message) = json(post(&app, MESSAGES, &body(json!({}))).await).await;
    assert_eq!(message["content"][0]["type"], "thinking", "{message}");
    assert_eq!(message["content"][1], json!({"type": "text", "text": "Mild."}));
}

#[tokio::test]
async fn stop_sequences_the_limit_and_the_rest_of_the_body_reach_the_engine() {
    let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}});
    let request = submitted(json!({
        "stop_sequences": ["END", "\n\n"],
        "max_tokens": 7,
        "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": "low"},
        "metadata": {"user_id": "u"},
    }))
    .await;
    assert_eq!(request.stop, ["END", "\n\n"]);
    assert_eq!(request.max_tokens, Some(7));
    let structure = request.structure.as_ref().unwrap();
    assert_eq!(structure.shape, Shape::JsonSchema(schema.to_string()));
    assert!(matches!(structure.start, Start::AfterReasoning { .. }));
    let Prompt::Chat { variables, .. } = &request.prompt else { panic!() };
    assert_eq!(variables, &[("reasoning_effort".to_owned(), Json::String("low".into()))]);

    // The engine has one word for the end token and for a stop string.
    let message = whole(
        says("Mild.", 2, FinishReason::Stop),
        json!({"stop_sequences": ["END"],
        "thinking": {"type": "disabled"}}),
    )
    .await;
    assert_eq!(
        (&message["stop_reason"], &message["stop_sequence"]),
        (&json!("end_turn"), &Value::Null)
    );
    let message =
        whole(says("Mild", 2, FinishReason::Length), json!({"thinking": {"type": "disabled"}}))
            .await;
    assert_eq!(message["stop_reason"], "max_tokens");
}

// --- the reply -------------------------------------------------------------------

#[tokio::test]
async fn the_calls_of_a_reply_are_tool_use_blocks() {
    let reply = format!("{THINKS}I will look.\n\n{CALL}\n{NOW}");
    let message = whole(says(&reply, 3, FinishReason::Stop), json!({"tools": tools()})).await;
    assert_eq!(message["stop_reason"], "tool_use");
    assert_eq!(
        anonymous(message)["content"],
        json!([
            {"type": "thinking", "thinking": "The user asks for the weather.", "signature": "unsigned"},
            {"type": "text", "text": "I will look."},
            {"type": "tool_use", "id": "0", "name": "get_weather", "input": {"city": "Köln", "days": 3}},
            {"type": "tool_use", "id": "1", "name": "now", "input": {}},
        ])
    );
    // A reply that is only a call has no text block.
    let extra = json!({"tools": tools(), "thinking": {"type": "disabled"}});
    let message = whole(says(CALL, 4, FinishReason::Stop), extra.clone()).await;
    let content = anonymous(message)["content"].clone();
    assert_eq!(content.as_array().unwrap().len(), 1, "{content}");
    assert_eq!(content[0]["type"], "tool_use");
    // The limit inside a call: the call as far as it came, with no input.
    let cut = &CALL[..CALL.find("</parameter>").unwrap() - 3];
    let message = whole(says(cut, 4, FinishReason::Length), extra).await;
    assert_eq!(message["stop_reason"], "max_tokens");
    assert_eq!(message["content"][0]["name"], "get_weather");
    assert_eq!(message["content"][0]["input"], json!({}));
}

#[tokio::test]
async fn a_stream_is_the_events_of_the_api_in_their_order() {
    let reply = format!("{THINKS}I look.\n\n{CALL}");
    let mut steps = says(&reply, 400, FinishReason::Stop);
    *steps.last_mut().unwrap() = Step::Send(Event::Done {
        finish: FinishReason::Stop,
        usage: Usage {
            prompt_tokens: 50,
            completion_tokens: 30,
            cached_prompt_tokens: 32,
            ..Usage::default()
        },
    });
    let events = streamed(steps, json!({"tools": tools()})).await;
    let outline: Vec<String> = events
        .iter()
        .map(|(name, data)| {
            let detail = match name.as_str() {
                "content_block_start" => {
                    format!(" {} {}", data["index"], data["content_block"]["type"])
                }
                "content_block_delta" => format!(" {} {}", data["index"], data["delta"]["type"]),
                "content_block_stop" => format!(" {}", data["index"]),
                _ => String::new(),
            };
            format!("{name}{detail}").replace('"', "")
        })
        .collect();
    assert_eq!(
        outline,
        [
            "message_start",
            "content_block_start 0 thinking",
            "content_block_delta 0 thinking_delta",
            "content_block_delta 0 signature_delta",
            "content_block_stop 0",
            "content_block_start 1 text",
            "content_block_delta 1 text_delta",
            "content_block_stop 1",
            "content_block_start 2 tool_use",
            "content_block_delta 2 input_json_delta",
            "content_block_stop 2",
            "message_delta",
            "message_stop",
        ]
    );
    let data = |at: usize| &events[at].1;
    let start = &data(0)["message"];
    assert!(start["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(
        (&start["type"], &start["role"], &start["model"]),
        (&json!("message"), &json!("assistant"), &json!("mock"))
    );
    // The prompt is counted when the reply ends.
    assert_eq!(start["usage"]["input_tokens"], 0);
    assert_eq!(
        data(1)["content_block"],
        json!({"type": "thinking", "thinking": "", "signature": ""})
    );
    assert_eq!(
        data(2)["delta"],
        json!({"type": "thinking_delta", "thinking": "The user asks for the weather."})
    );
    assert_eq!(data(3)["delta"], json!({"type": "signature_delta", "signature": "unsigned"}));
    assert_eq!(data(5)["content_block"], json!({"type": "text", "text": ""}));
    assert_eq!(data(6)["delta"], json!({"type": "text_delta", "text": "I look."}));
    let call = &data(8)["content_block"];
    assert_eq!(
        (&call["type"], &call["name"], &call["input"]),
        (&json!("tool_use"), &json!("get_weather"), &json!({}))
    );
    assert!(call["id"].as_str().unwrap().starts_with("toolu_"));
    let input = json!({"type": "input_json_delta", "partial_json": r#"{"city":"Köln","days":3}"#});
    assert_eq!(data(9)["delta"], input);
    assert_eq!(
        data(11),
        &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": null},
            "usage": {
                "input_tokens": 18,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 32,
                "output_tokens": 30,
            },
        })
    );
    assert_eq!(data(12), &json!({"type": "message_stop"}));
}

#[tokio::test]
async fn a_stream_put_together_is_the_message_that_is_returned_whole() {
    let cut_call = &CALL[..CALL.find("</parameter>").unwrap() - 3];
    let replies: Vec<(String, FinishReason, Value)> = vec![
        ("Mild, 12 °C.".into(), FinishReason::Stop, json!({"thinking": {"type": "disabled"}})),
        (format!("{THINKS}Mild, 12 °C."), FinishReason::Stop, json!({})),
        (
            format!("{THINKS}I look.\n\n{CALL}\n{NOW}"),
            FinishReason::Stop,
            json!({"tools": tools()}),
        ),
        (
            format!("{THINKS}{CALL}"),
            FinishReason::Stop,
            json!({"tools": tools(), "tool_choice": {"type": "any"}}),
        ),
        ("The user asks for".into(), FinishReason::Length, json!({"tools": tools()})),
        (format!("{THINKS}Mild, 12"), FinishReason::Length, json!({})),
        (format!("{THINKS}{cut_call}"), FinishReason::Length, json!({"tools": tools()})),
        (String::new(), FinishReason::Stop, json!({})),
        (
            format!("{THINKS}{CALL}"),
            FinishReason::Stop,
            json!({"tools": tools(), "tool_choice": {"type": "none"}}),
        ),
    ];
    for (reply, finish, extra) in replies {
        let message = anonymous(whole(says(&reply, 1000, finish), extra.clone()).await);
        assert!(message["stop_reason"].is_string(), "{message}");
        // However the tokens cut the text.
        for size in [1, 2, 3, 7, 1000] {
            let events = streamed(says(&reply, size, finish), extra.clone()).await;
            assert_eq!(anonymous(assemble(&events)), message, "pieces of {size}: {reply:?}");
        }
    }
}

// --- failures --------------------------------------------------------------------

async fn refused(app: &Router, path: &str, body: &str, status: StatusCode, kind: &str) -> String {
    let (got, value) = json(post(app, path, body).await).await;
    assert_eq!(got, status, "{value}");
    assert_eq!(value["type"], "error", "{value}");
    assert_eq!(value["error"]["type"], kind, "{value}");
    assert_eq!(value.as_object().unwrap().len(), 2, "{value}");
    assert_eq!(value["error"].as_object().unwrap().len(), 2, "{value}");
    value["error"]["message"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn what_the_engine_cannot_do_is_a_400_in_the_shape_of_the_api() {
    let image =
        json!({"type": "image", "source": {"type": "url", "url": "https://example.org/a.png"}});
    let bodies = [
        (
            body(json!({"messages": [{"role": "user", "content": [image]}]})),
            "'image' is not supported",
        ),
        (
            body(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})),
            "'web_search_20250305'",
        ),
        (body(json!({"mcp_servers": []})), "mcp_servers is not supported"),
        (body(json!({"max_tokens": null})), "max_tokens is required"),
        (body(json!({"temperature": "hot"})), "temperature must be a number"),
        (body(json!({"model": null})), "model is required"),
        ("{".to_owned(), "not valid JSON"),
        ("[1]".to_owned(), "must be a JSON object"),
    ];
    for path in [MESSAGES, COUNT] {
        let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
        for (body, fragment) in &bodies {
            if path == COUNT && fragment.starts_with("max_tokens") {
                continue;
            }
            let message =
                refused(&app, path, body, StatusCode::BAD_REQUEST, "invalid_request_error").await;
            assert!(message.contains(fragment), "{path}: {message:?} lacks {fragment:?}");
        }
        assert!(engine.submitted().is_empty(), "the engine was not to see any of them");
    }

    // What only the model's template decides.
    let (engine, app) = serve_with(vec![done(FinishReason::Stop, 1, 0)], None);
    let message = refused(
        &app,
        MESSAGES,
        &body(json!({"tools": tools()})),
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
    )
    .await;
    assert!(message.starts_with("tools are not supported with this model"), "{message}");
    let effort = body(json!({"output_config": {"effort": "low"}}));
    let message =
        refused(&app, MESSAGES, &effort, StatusCode::BAD_REQUEST, "invalid_request_error").await;
    assert!(message.starts_with("output_config.effort is not supported"), "{message}");
    assert!(engine.submitted().is_empty());
}

#[tokio::test]
async fn failures_have_the_status_of_the_chat_route_and_the_shape_of_this_api() {
    let (_, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
    let response =
        app.clone().oneshot(Request::get(MESSAGES).body(Body::empty()).unwrap()).await.unwrap();
    let (status, value) = json(response).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(value["error"]["type"], "invalid_request_error", "{value}");

    let cases = [
        (EngineError::ModelNotFound("nope".into()), StatusCode::NOT_FOUND, "not_found_error"),
        (EngineError::Overloaded, StatusCode::SERVICE_UNAVAILABLE, "overloaded_error"),
        (
            EngineError::ContextOverflow { prompt_tokens: 9000, limit: 8192 },
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
        ),
        (
            EngineError::Internal("a device failed".into()),
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
        ),
    ];
    for (error, status, kind) in cases {
        // Refused at once, and found out while the prompt is read: a stream
        // has not started then, and gets the status too.
        for plan in [Plan::Reject(error.clone()), Plan::Script(vec![fail(error.clone())])] {
            for stream in [false, true] {
                let engine =
                    Declaring { inner: MockEngine::new(plan.clone()), format: Some(qwen()) };
                let app = ares_server::router(Arc::new(engine));
                refused(&app, MESSAGES, &body(json!({"stream": stream})), status, kind).await;
            }
        }
    }
}

#[tokio::test]
async fn a_failure_inside_a_stream_is_an_error_event_and_the_stream_ends() {
    let steps = vec![text("Mild"), fail(EngineError::Internal("a device failed".into()))];
    let (_, app) = serve(steps);
    let extra = json!({"stream": true, "thinking": {"type": "disabled"}});
    let response = post(&app, MESSAGES, &body(extra)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let events = events(&body_text(response).await);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["message_start", "content_block_start", "content_block_delta", "error"]);
    assert_eq!(
        events[3].1,
        json!({"type": "error", "error": {
            "type": "api_error",
            "message": "internal engine error: a device failed",
        }})
    );
}

#[tokio::test]
async fn a_client_that_leaves_cancels_its_request_in_the_engine() {
    // Waiting for a whole message.
    let (engine, app) = serve(vec![Step::UntilCancelled]);
    let call = tokio::spawn({
        let app = app.clone();
        async move { app.oneshot(post_request(MESSAGES, &body(json!({})))).await }
    });
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;
    assert!(engine.cancelled().is_empty(), "nothing to cancel while the client waits");
    call.abort();
    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);

    // Reading a stream.
    let (engine, app) = serve(vec![text("a"), Step::UntilCancelled]);
    let extra = json!({"stream": true, "thinking": {"type": "disabled"}});
    let response = post(&app, MESSAGES, &body(extra)).await;
    let mut stream = response.into_body().into_data_stream();
    let first = stream.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&first).contains("message_start"));
    assert!(engine.cancelled().is_empty(), "the client is still reading");
    drop(stream);
    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);
}

// --- the count ---------------------------------------------------------------------

/// An engine that reports for a request the count it gives for its prompt.
struct Counting;

impl Engine for Counting {
    fn models(&self) -> Vec<ModelCard> {
        vec![ModelCard { id: "mock".into(), context_length: 4096, created: 0 }]
    }

    fn chat_format(&self, _: &str) -> Option<ChatFormat> {
        Some(qwen())
    }

    fn count_tokens(&self, _: &str, prompt: &Prompt) -> Result<u32, EngineError> {
        Ok(tokens(prompt))
    }

    fn submit(&self, request: GenerateRequest, events: EventSink) -> Result<(), EngineError> {
        let usage = Usage {
            prompt_tokens: tokens(&request.prompt),
            completion_tokens: 1,
            ..Usage::default()
        };
        std::thread::spawn(move || {
            events.send(Event::Text("ok".into()));
            events.send(Event::Done { finish: FinishReason::Stop, usage });
        });
        Ok(())
    }

    fn decide(&self, _: DecisionRequest, _: DecisionSink) -> Result<(), EngineError> {
        Err(EngineError::Unsupported("no decisions".into()))
    }

    fn cancel(&self, _: RequestId) {}
}

#[tokio::test]
async fn the_count_of_a_request_is_the_input_the_same_request_reports_when_run() {
    let app = ares_server::router(Arc::new(Counting));
    let conversation = json!([
        {"role": "user", "content": [{"type": "text", "text": "Weather in Köln?"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "I look it up.", "signature": "unsigned"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Köln"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "12 °C", "is_error": true},
        ]},
    ]);
    let bodies = [
        json!({}),
        json!({"system": "Be brief."}),
        json!({"system": [{"type": "text", "text": "Be brief."}], "tools": tools(),
            "tool_choice": {"type": "any"}, "messages": conversation}),
        json!({"thinking": {"type": "enabled", "budget_tokens": 2048}}),
        json!({"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}),
    ];
    let mut counts = Vec::new();
    for extra in bodies {
        let (status, counted) = json(post(&app, COUNT, &body(extra.clone())).await).await;
        assert_eq!(status, StatusCode::OK, "{counted}");
        assert_eq!(counted.as_object().unwrap().len(), 1, "{counted}");
        let count = counted["input_tokens"].as_u64().expect("a count");

        let (_, message) = json(post(&app, MESSAGES, &body(extra.clone())).await).await;
        let usage = &message["usage"];
        assert_eq!(usage["input_tokens"], count, "{extra}");
        assert_eq!(usage["cache_read_input_tokens"], 0);

        // Without the limit and the fields of the generation it is the same count.
        let mut bare: Value = serde_json::from_str(&body(extra)).unwrap();
        bare.as_object_mut().unwrap().remove("max_tokens");
        let (_, again) = json(post(&app, COUNT, &bare.to_string()).await).await;
        assert_eq!(again["input_tokens"], count);
        counts.push(count);
    }
    counts.sort_unstable();
    counts.dedup();
    assert_eq!(counts.len(), 5, "every one of these prompts is another one");
}

#[tokio::test]
async fn a_count_generates_nothing_and_needs_an_engine_that_counts() {
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
    let (status, counted) = json(post(&app, COUNT, &body(json!({"tools": tools()}))).await).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    assert!(counted["input_tokens"].as_u64().unwrap() > 0);
    assert!(engine.submitted().is_empty(), "nothing is generated for a count");

    let unknown = body(json!({"model": "nope"}));
    refused(&app, COUNT, &unknown, StatusCode::NOT_FOUND, "not_found_error").await;

    // The scripted engine alone does not count.
    let plain = app_of(MockEngine::script(vec![done(FinishReason::Stop, 1, 0)]));
    let message =
        refused(&plain, COUNT, &body(json!({})), StatusCode::NOT_IMPLEMENTED, "api_error").await;
    assert!(message.contains("does not count the tokens of a prompt"), "{message}");
}

fn app_of(engine: Arc<MockEngine>) -> Router {
    ares_server::router(engine)
}
