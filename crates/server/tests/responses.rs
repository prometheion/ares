//! The OpenAI Responses API against a scripted engine: what a request comes
//! to when it reaches the engine, how a reply comes back, whole and as the
//! typed events of a stream, what is refused, and the counts of a prompt.

mod common;

use std::sync::Arc;

use ares_api::{
    ChatFormat, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, Event, EventSink,
    FinishReason, GenerateRequest, Json, ModelCard, Prompt, RequestId, Role, Shape, Start,
    ThinkingFormat, ToolFormat, Usage,
};
use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use common::*;
use futures_util::StreamExt;
use serde_json::{Value, json};
use tower::ServiceExt;

const RESPONSES: &str = "/v1/responses";
const COUNT: &str = "/v1/responses/input_tokens";
const CHAT: &str = "/v1/chat/completions";
const CHAT_COUNT: &str = "/v1/chat/completions/input_tokens";

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
            "type": "function",
            "name": "get_weather",
            "description": "The weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
            "strict": true,
        },
        {"type": "function", "name": "now", "parameters": {"type": "object"}},
    ])
}

fn body(extra: Value) -> String {
    let mut body = json!({"model": "mock", "input": "How is the weather in Köln?"});
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

/// What switches the thinking of a request off.
fn no_thinking() -> Value {
    json!({"reasoning": {"effort": "none"}})
}

/// The request the engine gets for a body.
async fn submitted(extra: Value) -> GenerateRequest {
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
    let (status, answer) = json(post(&app, RESPONSES, &body(extra)).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    engine.submitted().remove(0)
}

/// The response a body is answered with.
async fn whole(steps: Vec<Step>, extra: Value) -> Value {
    let (_, app) = serve(steps);
    let (status, response) = json(post(&app, RESPONSES, &body(extra)).await).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    response
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
    let response = post(&app, RESPONSES, &body(extra)).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "text/event-stream");
    let text = body_text(response).await;
    assert!(!text.contains("[DONE]"), "this API has no end marker: {text}");
    events(&text)
}

fn names(events: &[(String, Value)]) -> Vec<&str> {
    events.iter().map(|(name, _)| name.as_str()).collect()
}

/// Puts the events of a stream together into the response they describe,
/// the way a client does, and checks their order and their numbers on the
/// way.
fn assemble(events: &[(String, Value)]) -> Value {
    for (number, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], name.as_str(), "the event is named as its data says");
        assert_eq!(data["sequence_number"], number, "{name}: the events are numbered in order");
    }
    assert_eq!(events[0].0, "response.created");
    assert_eq!(events[1].0, "response.in_progress");
    let begun = &events[0].1["response"];
    assert_eq!(begun, &events[1].1["response"]);
    assert_eq!((&begun["status"], &begun["output"]), (&json!("in_progress"), &json!([])));
    assert_eq!(begun["usage"], Value::Null);
    let (last, between) = events[2..].split_last().expect("a last event");

    let mut output: Vec<Value> = Vec::new();
    // The item that is open, and whether its part is.
    let mut open: Option<usize> = None;
    let mut part = false;
    for (name, data) in between {
        let index = data["output_index"].as_u64().expect("an output_index") as usize;
        if name == "response.output_item.added" {
            assert_eq!(open, None, "an item opens while another is open");
            assert_eq!(index, output.len(), "the items are numbered in order");
            let item = &data["item"];
            assert_eq!(item["status"], "in_progress");
            match item["type"].as_str().unwrap() {
                "function_call" => assert_eq!(item["arguments"], ""),
                _ => assert_eq!(item["content"], json!([])),
            }
            output.push(item.clone());
            open = Some(index);
            continue;
        }
        assert_eq!(Some(index), open, "{name} of the item that is open");
        let item = &mut output[index];
        if name != "response.output_item.done" {
            assert_eq!(data["item_id"], item["id"], "{name}");
        }
        let text_of = |item: &Value| item["content"][0]["text"].as_str().unwrap().to_owned();
        let kind = item["type"].as_str().unwrap().to_owned();
        match (name.as_str(), kind.as_str()) {
            ("response.content_part.added", "reasoning" | "message") => {
                assert!(!part && item["content"] == json!([]), "one part, once");
                assert_eq!(data["content_index"], 0);
                assert_eq!(data["part"]["text"], "");
                item["content"] = json!([data["part"]]);
                part = true;
            }
            ("response.reasoning_text.delta", "reasoning")
            | ("response.output_text.delta", "message") => {
                assert!(part, "a delta of a part that is open");
                assert_eq!(data["content_index"], 0);
                let more = data["delta"].as_str().unwrap();
                assert!(!more.is_empty(), "no empty deltas");
                item["content"][0]["text"] = json!(text_of(item) + more);
            }
            ("response.reasoning_text.done", "reasoning")
            | ("response.output_text.done", "message") => {
                assert!(part);
                assert_eq!(data["text"], text_of(item), "the text is the deltas");
            }
            ("response.content_part.done", "reasoning" | "message") => {
                assert!(part);
                assert_eq!(data["part"], item["content"][0]);
                part = false;
            }
            ("response.function_call_arguments.delta", "function_call") => {
                let so_far = item["arguments"].as_str().unwrap().to_owned();
                item["arguments"] = json!(so_far + data["delta"].as_str().unwrap());
            }
            ("response.function_call_arguments.done", "function_call") => {
                assert_eq!(data["arguments"], item["arguments"], "the arguments are the deltas");
                assert_eq!(data["name"], item["name"]);
            }
            ("response.output_item.done", _) => {
                assert!(!part, "an item closes while its part is open");
                let done = &data["item"];
                assert!(done["status"] == "completed" || done["status"] == "incomplete");
                item["status"] = done["status"].clone();
                assert_eq!(done, item, "the item is what its events said");
                open = None;
            }
            (name, kind) => panic!("{name} for an item of the type {kind}"),
        }
    }
    assert_eq!(open, None, "the response ends while an item is open");
    let response = last.1["response"].clone();
    let name = match response["status"].as_str().unwrap() {
        "completed" => "response.completed",
        "incomplete" => "response.incomplete",
        other => panic!("a response that ends as {other}"),
    };
    assert_eq!(last.0, name);
    assert_eq!(response["output"], Value::Array(output), "the output is what the events said");
    response
}

/// A response without what differs between two requests: the id of the
/// request in its ids, when it was made, and how long it took.
fn anonymous(response: Value) -> Value {
    let id = response["id"].as_str().expect("an id").to_owned();
    let request = id.strip_prefix("resp_").expect("the id of a response");
    assert_eq!(request.len(), 16, "{id}");
    let mut response: Value =
        serde_json::from_str(&response.to_string().replace(request, "ID")).unwrap();
    assert!(response["created_at"].as_u64().unwrap() > 1_700_000_000);
    response["created_at"] = Value::Null;
    response.as_object_mut().unwrap().remove("timings");
    response
}

// --- the request ---------------------------------------------------------------

#[tokio::test]
async fn a_plain_request_is_the_chat_request_and_its_reply_one_message() {
    // A model whose template has no thinking: the reply is its text.
    let steps = vec![text("Mild, "), text("12 °C."), done(FinishReason::Stop, 9, 5)];
    let (engine, app) = serve_with(steps, None);
    // The header of OpenAI's clients is accepted.
    let extra = json!({
        "instructions": "Be brief.",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 256,
        "store": false,
        "metadata": {"run": "7"},
        "user": "u1",
    });
    let request = Request::post(RESPONSES)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer sk-none")
        .body(Body::from(body(extra)))
        .unwrap();
    let (status, response) = json(app.clone().oneshot(request).await.unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["timings"]["predicted_n"], 5, "{response}");
    assert_eq!(
        anonymous(response),
        json!({
            "id": "resp_ID",
            "object": "response",
            "created_at": null,
            "status": "completed",
            "background": false,
            "error": null,
            "incomplete_details": null,
            "instructions": "Be brief.",
            "max_output_tokens": 256,
            "model": "mock",
            "output": [{
                "type": "message",
                "id": "msg_ID_0",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text", "text": "Mild, 12 °C.", "annotations": [],
                    "logprobs": [],
                }],
            }],
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "reasoning": {"effort": null, "summary": null},
            "store": false,
            "temperature": 0.2,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 0.9,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 9,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "total_tokens": 14,
            },
            "user": "u1",
            "metadata": {"run": "7"},
        })
    );

    let request = engine.submitted().remove(0);
    assert_eq!((request.model.as_str(), request.max_tokens), ("mock", Some(256)));
    assert_eq!((request.sampling.temperature, request.sampling.top_p), (Some(0.2), Some(0.9)));
    assert!(request.draft && request.structure.is_none() && request.stop.is_empty());
    let Prompt::Chat { messages, tools, variables, .. } = &request.prompt else { panic!() };
    let said: Vec<_> = messages.iter().map(|m| (m.role, m.content.as_str())).collect();
    assert_eq!(said, [(Role::System, "Be brief."), (Role::User, "How is the weather in Köln?")]);
    // Thinking is left to the server when the request does not say.
    assert!(tools.is_none() && variables.is_empty(), "{variables:?}");
    // Without a limit the request has none.
    assert_eq!(submitted(json!({})).await.max_tokens, None);
}

#[tokio::test]
async fn the_items_of_a_conversation_reach_the_template_as_its_turns() {
    let input = json!([
        {"type": "message", "role": "developer", "content": [
            {"type": "input_text", "text": "You are a coding agent."},
        ]},
        {"role": "user", "content": [{"type": "input_text", "text": "Weather in Köln?"}]},
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": null, "content": [
            {"type": "reasoning_text", "text": "I look it up."},
        ]},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": "One moment.", "annotations": [], "logprobs": []},
        ]},
        {"type": "function_call", "id": "fc_1", "status": "completed", "call_id": "call_1",
         "name": "get_weather", "arguments": "{\"city\":\"Köln\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "12 °C"},
    ]);
    let extra = json!({"instructions": "Be brief.", "input": input, "tools": tools()});
    let request = submitted(extra).await;
    let Prompt::Chat { messages, .. } = &request.prompt else { panic!() };
    let turns: Vec<_> = messages.iter().map(|m| (m.role, m.content.as_str())).collect();
    assert_eq!(
        turns,
        [
            (Role::System, "Be brief.\n\nYou are a coding agent."),
            (Role::User, "Weather in Köln?"),
            (Role::Assistant, "One moment."),
            (Role::Tool, "12 °C"),
        ]
    );
    assert_eq!(messages[2].reasoning_content.as_deref(), Some("I look it up."));
    assert_eq!(
        serde_json::to_value(&messages[2].tool_calls).unwrap(),
        json!([{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": "{\"city\":\"Köln\"}",
        }}])
    );
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_1"));
}

#[tokio::test]
async fn a_request_as_codex_writes_it_is_served() {
    // The members Codex sends with every request, as its source has them:
    // no instructions (they stand first in the input), a stream, the
    // include, its own bookkeeping, and the items of a turn sent back
    // without their ids.
    let text = |text: &str| json!([{"type": "input_text", "text": text}]);
    let body = json!({
        "model": "mock",
        "stream": true,
        "input": [
            {"type": "message", "role": "developer", "content": text("You are a coding agent.")},
            {"type": "message", "role": "developer", "content": text("Sandbox: read-only.")},
            {"type": "message", "role": "user", "content": text("<cwd>/srv</cwd>")},
            {"type": "message", "role": "user", "content": text("How is the weather in Köln?")},
            {"type": "reasoning", "summary": [], "encrypted_content": null, "content": [
                {"type": "reasoning_text", "text": "I look it up."},
            ]},
            {"type": "function_call", "name": "get_weather", "call_id": "call_1",
             "arguments": "{\"city\":\"Köln\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "12 °C"},
        ],
        "tools": tools(),
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "reasoning": {"effort": "medium"},
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "0199a213-81c0-7800-8aa1-bbab2a035a53",
        "client_metadata": {"x-codex-installation-id": "7"},
    });
    let (engine, app) = serve(says(&format!("{THINKS}Mild."), 4, FinishReason::Stop));
    let response = post(&app, RESPONSES, &body.to_string()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let response = assemble(&events(&body_text(response).await));
    assert_eq!((&response["status"], &response["store"]), (&json!("completed"), &json!(false)));
    assert_eq!(response["output"][1]["content"][0]["text"], "Mild.");

    let request = engine.submitted().remove(0);
    let Prompt::Chat { messages, variables, .. } = &request.prompt else { panic!() };
    let turns: Vec<_> = messages.iter().map(|m| (m.role, m.content.as_str())).collect();
    assert_eq!(
        turns,
        [
            (Role::System, "You are a coding agent.\n\nSandbox: read-only."),
            (Role::User, "<cwd>/srv</cwd>"),
            (Role::User, "How is the weather in Köln?"),
            (Role::Assistant, ""),
            (Role::Tool, "12 °C"),
        ]
    );
    assert_eq!(messages[3].reasoning_content.as_deref(), Some("I look it up."));
    assert_eq!(variables, &[("reasoning_effort".to_owned(), Json::String("medium".into()))]);
}

#[tokio::test]
async fn tools_reach_the_template_and_the_reply_gets_the_rule_of_the_tool_choice() {
    let after_reasoning = Start::AfterReasoning { open: None, close: "</think>".into() };
    let lark = |request: &GenerateRequest| match &request.structure.as_ref().unwrap().shape {
        Shape::Lark(text) => text.clone(),
        other => panic!("not a grammar of calls: {other:?}"),
    };

    // auto, which is the default: free until the trigger, then calls.
    for extra in [json!({}), json!({"tool_choice": "auto"})] {
        let mut extra = extra;
        extra["tools"] = tools();
        let request = submitted(extra).await;
        let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
        assert_eq!(
            serde_json::to_value(sent).unwrap()[0],
            json!({"type": "function", "function": {
                "name": "get_weather",
                "description": "The weather of a city.",
                "parameters": tools()[0]["parameters"],
            }})
        );
        let structure = request.structure.as_ref().unwrap();
        assert_eq!(structure.trigger.as_deref(), Some("<tool_call>"));
        assert_eq!(structure.start, after_reasoning, "the reasoning is free");
        assert!(lark(&request).contains("more: (WS? next)* WS?\n"), "several calls");
    }

    // required: calls from the first token of the answer.
    let request = submitted(json!({"tools": tools(), "tool_choice": "required"})).await;
    let structure = request.structure.as_ref().unwrap();
    assert_eq!((structure.trigger.as_deref(), &structure.start), (None, &after_reasoning));
    assert!(lark(&request).contains("<function=now>"));

    // A named function: calls of it, and one of them when the request says so.
    let named = json!({"type": "function", "name": "get_weather"});
    let extra = json!({"tools": tools(), "tool_choice": named, "parallel_tool_calls": false});
    let request = submitted(extra).await;
    let grammar = lark(&request);
    assert!(grammar.contains("<function=get_weather>") && !grammar.contains("<function=now>"));
    assert!(!grammar.contains("(WS? next)*"), "one call: {grammar}");
    let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
    assert_eq!(sent.as_ref().unwrap().len(), 2, "the prompt lists every tool");

    // none: the tools are in the prompt and the reply is text.
    let request = submitted(json!({"tools": tools(), "tool_choice": "none"})).await;
    let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
    assert!(sent.is_some() && request.structure.is_none());
}

#[tokio::test]
async fn reasoning_is_switched_by_the_request_and_comes_back_as_an_item() {
    let variables = |request: &GenerateRequest| match &request.prompt {
        Prompt::Chat { variables, .. } => variables.clone(),
        Prompt::Text(_) => panic!(),
    };
    let low = submitted(json!({"reasoning": {"effort": "low", "summary": "auto"}})).await;
    assert_eq!(variables(&low), [("reasoning_effort".to_owned(), Json::String("low".into()))]);
    let off = submitted(no_thinking()).await;
    assert_eq!(variables(&off), [("enable_thinking".to_owned(), Json::Bool(false))]);
    // What Codex sends: the include, and a reasoning that may say nothing.
    let codex = json!({"reasoning": null, "include": ["reasoning.encrypted_content"]});
    assert!(variables(&submitted(codex).await).is_empty());

    // A model that thinks: the reasoning is an item of its own, before the
    // message, with the reasoning itself as its content.
    let reply = format!("{THINKS}Mild.");
    let extra =
        json!({"reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]});
    let response = anonymous(whole(says(&reply, 5, FinishReason::Stop), extra).await);
    assert_eq!(response["reasoning"], json!({"effort": "high", "summary": null}));
    assert_eq!(
        response["output"],
        json!([
            {
                "type": "reasoning",
                "id": "rs_ID_0",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "The user asks for the weather."}],
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_ID_1",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text", "text": "Mild.", "annotations": [], "logprobs": [],
                }],
            },
        ])
    );
    // Thinking off: the template has closed the block, the reply is the answer.
    let response = whole(says("Mild.", 2, FinishReason::Stop), no_thinking()).await;
    assert_eq!(response["output"].as_array().unwrap().len(), 1, "{response}");
    assert_eq!(response["output"][0]["type"], "message");
}

#[tokio::test]
async fn the_reasoning_is_an_item_whatever_the_format_of_the_chat_route() {
    // `--reasoning-format none` is about reasoning_content.
    let inner = MockEngine::script(says(&format!("{THINKS}Mild."), 7, FinishReason::Stop));
    let engine = Declaring { inner, format: Some(qwen()) };
    let config = ares_server::Config {
        reasoning_format: ares_server::ReasoningFormat::None,
        ..ares_server::Config::default()
    };
    let app = ares_server::router_with(Arc::new(engine), config);
    let (status, response) = json(post(&app, RESPONSES, &body(json!({}))).await).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let kinds: Vec<&str> = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["reasoning", "message"]);
    assert_eq!(response["output"][1]["content"][0]["text"], "Mild.");
}

#[tokio::test]
async fn text_format_is_the_rule_of_the_reply() {
    let schema = json!({"type": "object", "properties": {"degrees": {"type": "integer"}}});
    let format =
        json!({"type": "json_schema", "name": "weather", "strict": true, "schema": schema});
    let request = submitted(json!({"text": {"format": format}})).await;
    let structure = request.structure.as_ref().expect("a rule");
    assert_eq!(structure.shape, Shape::JsonSchema(schema.to_string()));
    // The reasoning stays free, as on the chat route.
    assert!(matches!(structure.start, Start::AfterReasoning { .. }));

    let request = submitted(json!({"text": {"format": {"type": "json_object"}}})).await;
    let structure = request.structure.as_ref().expect("a rule");
    assert_eq!(structure.shape, Shape::JsonSchema(r#"{"type":"object"}"#.to_owned()));
    let request = submitted(json!({"text": {"format": {"type": "text"}}})).await;
    assert!(request.structure.is_none());

    // The response says the format back.
    let extra = json!({"text": {"format": format}, "reasoning": {"effort": "none"}});
    let response = whole(says("{\"degrees\":12}", 4, FinishReason::Stop), extra).await;
    assert_eq!(response["text"], json!({"format": format}));
    assert_eq!(response["output"][0]["content"][0]["text"], "{\"degrees\":12}");
}

#[tokio::test]
async fn the_limit_of_tokens_makes_a_response_incomplete() {
    let request = submitted(json!({"max_output_tokens": 7})).await;
    assert_eq!(request.max_tokens, Some(7));

    let extra = json!({"max_output_tokens": 7});
    let reply = format!("{THINKS}Mild, 12");
    let response = whole(says(&reply, 3, FinishReason::Length), extra.clone()).await;
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["incomplete_details"], json!({"reason": "max_output_tokens"}));
    assert_eq!(response["max_output_tokens"], 7);
    let statuses: Vec<&Value> =
        response["output"].as_array().unwrap().iter().map(|item| &item["status"]).collect();
    assert_eq!(statuses, ["completed", "incomplete"]);
    assert_eq!(response["output"][1]["content"][0]["text"], "Mild, 12");

    // The limit inside the thinking: a reasoning item and nothing else.
    let response = whole(says("The user asks", 4, FinishReason::Length), extra).await;
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["output"].as_array().unwrap().len(), 1);
    assert_eq!(response["output"][0]["content"][0]["text"], "The user asks");
    assert_eq!(response["output"][0]["status"], "incomplete");
}

// --- the reply -------------------------------------------------------------------

#[tokio::test]
async fn the_calls_of_a_reply_are_function_call_items() {
    let reply = format!("{THINKS}I will look.\n\n{CALL}\n{NOW}");
    let response =
        anonymous(whole(says(&reply, 3, FinishReason::Stop), json!({"tools": tools()})).await);
    assert_eq!(response["status"], "completed");
    assert_eq!(response["tools"], tools());
    let output = response["output"].as_array().unwrap();
    assert_eq!(output.len(), 4, "{response}");
    assert_eq!((&output[0]["type"], &output[1]["type"]), (&json!("reasoning"), &json!("message")));
    assert_eq!(output[1]["content"][0]["text"], "I will look.");
    assert_eq!(
        output[2..],
        [
            json!({
                "type": "function_call",
                "id": "fc_ID_2",
                "status": "completed",
                "call_id": "call_ID_0",
                "name": "get_weather",
                "arguments": "{\"city\":\"Köln\",\"days\":3}",
            }),
            json!({
                "type": "function_call",
                "id": "fc_ID_3",
                "status": "completed",
                "call_id": "call_ID_1",
                "name": "now",
                "arguments": "{}",
            }),
        ]
    );

    // A reply that is only a call has no message, and a call inside the
    // thinking is a call.
    let mut extra = no_thinking();
    extra["tools"] = tools();
    extra["tool_choice"] = json!("required");
    let response = whole(says(CALL, 4, FinishReason::Stop), extra.clone()).await;
    assert_eq!(response["output"].as_array().unwrap().len(), 1, "{response}");
    assert_eq!(response["output"][0]["type"], "function_call");
    let reply = format!("I should call.\n{CALL}");
    let response = whole(says(&reply, 4, FinishReason::Stop), json!({"tools": tools()})).await;
    let kinds: Vec<&str> = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["reasoning", "function_call"], "{response}");
    assert_eq!(response["output"][0]["content"][0]["text"], "I should call.");

    // The limit inside a call: the call as far as it came.
    let cut = &CALL[..CALL.find("</parameter>").unwrap() - 3];
    let response = whole(says(cut, 4, FinishReason::Length), extra).await;
    assert_eq!(response["status"], "incomplete");
    let call = &response["output"][0];
    assert_eq!((&call["name"], &call["status"]), (&json!("get_weather"), &json!("incomplete")));
    assert_eq!(call["arguments"], "{\"city\":\"Kö");
}

#[tokio::test]
async fn the_output_of_a_response_sent_back_is_the_turn_the_model_wrote() {
    // The reply of a first request, and the request a client makes of it:
    // the items as they came, and the outputs of the calls.
    let reply = format!("{THINKS}I will look.\n\n{CALL}");
    let first = whole(says(&reply, 6, FinishReason::Stop), json!({"tools": tools()})).await;
    let mut input = vec![json!({"role": "user", "content": "How is the weather in Köln?"})];
    input.extend(first["output"].as_array().unwrap().clone());
    let call_id = first["output"][2]["call_id"].clone();
    input.push(json!({"type": "function_call_output", "call_id": call_id, "output": "12 °C"}));

    let request = submitted(json!({"input": input, "tools": tools()})).await;
    let Prompt::Chat { messages, .. } = &request.prompt else { panic!() };
    assert_eq!(messages.len(), 3, "{messages:?}");
    let turn = &messages[1];
    assert_eq!((turn.role, turn.content.as_str()), (Role::Assistant, "I will look."));
    assert_eq!(turn.reasoning_content.as_deref(), Some("The user asks for the weather."));
    assert_eq!(
        serde_json::to_value(&turn.tool_calls).unwrap(),
        json!([{"id": call_id, "type": "function", "function": {
            "name": "get_weather", "arguments": "{\"city\":\"Köln\",\"days\":3}",
        }}])
    );
    assert_eq!((messages[2].role, messages[2].content.as_str()), (Role::Tool, "12 °C"));
    assert_eq!(messages[2].tool_call_id.as_deref(), call_id.as_str());
}

#[tokio::test]
async fn a_stream_is_the_events_of_the_api_in_their_order() {
    let reply = format!("{THINKS}I look.\n\n{CALL}");
    let events = streamed(says(&reply, 1000, FinishReason::Stop), json!({"tools": tools()})).await;
    assert_eq!(
        names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let numbers: Vec<u64> =
        events.iter().map(|(_, data)| data["sequence_number"].as_u64().unwrap()).collect();
    assert_eq!(numbers, (0..19).collect::<Vec<u64>>());
    let response = anonymous(assemble(&events));
    assert_eq!(response["status"], "completed");
    assert_eq!(response["usage"]["input_tokens"], 20);
    assert_eq!(events[4].1["delta"], "The user asks for the weather.");
    assert_eq!(events[10].1["delta"], "I look.");
    assert_eq!(events[15].1["delta"], "{\"city\":\"Köln\",\"days\":3}");
    assert_eq!(events[14].1["item"]["name"], "get_weather");

    // A reply the limit cut ends as incomplete.
    let events = streamed(says("The user", 3, FinishReason::Length), json!({})).await;
    assert_eq!(events.last().unwrap().0, "response.incomplete");
    assert_eq!(assemble(&events)["incomplete_details"], json!({"reason": "max_output_tokens"}));
}

#[tokio::test]
async fn a_stream_put_together_is_the_response_that_is_returned_whole() {
    let cut_call = &CALL[..CALL.find("</parameter>").unwrap() - 3];
    let with_tools = |extra: Value| {
        let mut extra = extra;
        extra["tools"] = tools();
        extra
    };
    let replies: Vec<(String, FinishReason, Value)> = vec![
        ("Mild, 12 °C.".into(), FinishReason::Stop, no_thinking()),
        (format!("{THINKS}Mild, 12 °C."), FinishReason::Stop, json!({"instructions": "Brief."})),
        (format!("{THINKS}I look.\n\n{CALL}\n{NOW}"), FinishReason::Stop, with_tools(json!({}))),
        (
            format!("{THINKS}{CALL}"),
            FinishReason::Stop,
            with_tools(json!({"tool_choice": "required"})),
        ),
        (format!("I should call.\n\n{CALL}\n{NOW}"), FinishReason::Stop, with_tools(json!({}))),
        ("The user asks for".into(), FinishReason::Length, with_tools(json!({}))),
        (format!("{THINKS}Mild, 12"), FinishReason::Length, json!({"max_output_tokens": 9})),
        (format!("{THINKS}{cut_call}"), FinishReason::Length, with_tools(json!({}))),
        (String::new(), FinishReason::Stop, json!({})),
        (format!("{THINKS}{CALL}"), FinishReason::Stop, with_tools(json!({"tool_choice": "none"}))),
    ];
    for (reply, finish, extra) in replies {
        let response = anonymous(whole(says(&reply, 1000, finish), extra.clone()).await);
        assert!(response["status"].is_string(), "{response}");
        // However the tokens cut the text.
        for size in [1, 2, 3, 7, 1000] {
            let events = streamed(says(&reply, size, finish), extra.clone()).await;
            assert_eq!(anonymous(assemble(&events)), response, "pieces of {size}: {reply:?}");
        }
    }
}

// --- failures --------------------------------------------------------------------

/// A refusal in the shape of OpenAI's errors; returns its message.
async fn refused(app: &Router, path: &str, body: &str, status: StatusCode, code: &str) -> String {
    let (got, value) = json(post(app, path, body).await).await;
    assert_eq!(got, status, "{value}");
    assert_eq!(value.as_object().unwrap().len(), 1, "{value}");
    let error = value["error"].as_object().expect("an error");
    assert_eq!(error.len(), 4, "{value}");
    assert_eq!(error["code"], code, "{value}");
    assert!(error.contains_key("param") && error["type"].is_string(), "{value}");
    error["message"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn what_the_server_does_not_do_is_a_400_that_names_it() {
    let bodies = [
        (json!({"store": true}), "store: true is not supported"),
        (json!({"previous_response_id": "resp_1"}), "previous_response_id is not supported"),
        (json!({"conversation": "conv_1"}), "conversation is not supported"),
        (json!({"background": true}), "background: true is not supported"),
        (json!({"prompt": {"id": "pmpt_1"}}), "prompt is not supported"),
        (json!({"truncation": "auto"}), "truncation \"auto\" is not supported"),
        (json!({"include": ["message.output_text.logprobs"]}), "is not supported"),
        (json!({"tools": [{"type": "web_search"}]}), "a tool of the type 'web_search'"),
        (json!({"tools": [{"type": "custom", "name": "p"}]}), "a tool of the type 'custom'"),
        (
            json!({"input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "https://example.com/a.png"},
            ]}]}),
            "a part of the type 'input_image'",
        ),
        (json!({"input": [{"type": "item_reference", "id": "msg_1"}]}), "item_reference"),
        (json!({"reasoning": {"summary": "detailed"}}), "reasoning.summary \"detailed\""),
        (json!({"top_k": 20}), "top_k is not supported"),
        (json!({"model": null}), "model is required"),
        (json!({"temperature": "hot"}), "temperature must be a number"),
        (json!({"tool_choice": "required"}), "the request has no tools"),
        (
            json!({"tools": tools(), "tool_choice": {"type": "function", "name": "then"}}),
            "which is not among the tools",
        ),
        (json!({"tools": [tools()[0], tools()[0]]}), "two tools named"),
    ];
    for path in [RESPONSES, COUNT] {
        let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
        for (extra, fragment) in &bodies {
            let body = body(extra.clone());
            let message =
                refused(&app, path, &body, StatusCode::BAD_REQUEST, "invalid_request").await;
            assert!(message.contains(fragment), "{path}: {message:?} lacks {fragment:?}");
        }
        let message =
            refused(&app, path, "{\"model\":", StatusCode::BAD_REQUEST, "invalid_request").await;
        assert!(message.contains("not valid JSON"), "{message}");
        assert!(engine.submitted().is_empty(), "the engine was not to see any of them");
    }

    // What only the model's template decides.
    let (engine, app) = serve_with(vec![done(FinishReason::Stop, 1, 0)], None);
    let with_tools = body(json!({"tools": tools()}));
    let message =
        refused(&app, RESPONSES, &with_tools, StatusCode::BAD_REQUEST, "invalid_request").await;
    assert!(message.starts_with("tools are not supported with this model"), "{message}");
    let effort = body(json!({"reasoning": {"effort": "high"}}));
    let (status, value) = json(post(&app, RESPONSES, &effort).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(value["error"]["param"], "reasoning.effort", "{value}");
    let message = value["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("reasoning.effort is not supported"), "{message}");
    assert!(engine.submitted().is_empty());
}

#[tokio::test]
async fn the_routes_of_a_stored_response_say_that_none_is_kept() {
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
    let cases = [
        (Method::GET, "/v1/responses/resp_0123456789abcdef"),
        (Method::DELETE, "/v1/responses/resp_0123456789abcdef"),
        (Method::POST, "/v1/responses/resp_0123456789abcdef/cancel"),
        (Method::GET, "/v1/responses/resp_0123456789abcdef/input_items"),
    ];
    for (method, path) in cases {
        let request = Request::builder().method(method).uri(path).body(Body::empty()).unwrap();
        let (status, value) = json(app.clone().oneshot(request).await.unwrap()).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path}: {value}");
        assert_eq!(value["error"]["code"], "not_found");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains(path) && message.contains("keeps no responses"), "{message}");
    }
    let (status, value) = json(post(&app, "/v1/responses/compact", &body(json!({}))).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(value["error"]["message"].as_str().unwrap().contains("does not compact"), "{value}");
    // The upgrade to a socket, and other methods on the routes that exist.
    for path in [RESPONSES, COUNT, CHAT_COUNT] {
        let (status, value) = json(get(&app, path).await).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED, "{path}: {value}");
        assert_eq!(value["error"]["code"], "method_not_allowed");
    }
    assert!(engine.submitted().is_empty());
}

#[tokio::test]
async fn failures_have_the_status_and_the_shape_of_the_chat_route() {
    let cases = [
        (EngineError::ModelNotFound("nope".into()), StatusCode::NOT_FOUND, "model_not_found"),
        (EngineError::Overloaded, StatusCode::SERVICE_UNAVAILABLE, "overloaded"),
        (
            EngineError::ContextOverflow { prompt_tokens: 9000, limit: 8192 },
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
        ),
        (
            EngineError::Internal("a device failed".into()),
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
        ),
    ];
    for (error, status, code) in cases {
        // Refused at once, and found out while the prompt is read: a stream
        // has not started then, and gets the status too.
        for plan in [Plan::Reject(error.clone()), Plan::Script(vec![fail(error.clone())])] {
            for stream in [false, true] {
                let engine =
                    Declaring { inner: MockEngine::new(plan.clone()), format: Some(qwen()) };
                let app = ares_server::router(Arc::new(engine));
                refused(&app, RESPONSES, &body(json!({"stream": stream})), status, code).await;
            }
        }
    }
}

#[tokio::test]
async fn a_failure_inside_a_stream_is_an_error_event_and_the_stream_ends() {
    let steps = vec![text("Mild"), fail(EngineError::Internal("a device failed".into()))];
    let (_, app) = serve(steps);
    let mut extra = no_thinking();
    extra["stream"] = json!(true);
    let response = post(&app, RESPONSES, &body(extra)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let events = events(&body_text(response).await);
    assert_eq!(
        names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "error",
        ]
    );
    assert_eq!(
        events[5].1,
        json!({
            "type": "error",
            "sequence_number": 5,
            "code": "internal_error",
            "message": "internal engine error: a device failed",
            "param": null,
        })
    );
}

#[tokio::test]
async fn a_client_that_leaves_cancels_its_request_in_the_engine() {
    // Waiting for a whole response.
    let (engine, app) = serve(vec![Step::UntilCancelled]);
    let call = tokio::spawn({
        let app = app.clone();
        async move { app.oneshot(post_request(RESPONSES, &body(json!({})))).await }
    });
    eventually("the request to reach the engine", || engine.submitted().len() == 1).await;
    assert!(engine.cancelled().is_empty(), "nothing to cancel while the client waits");
    call.abort();
    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);

    // Reading a stream.
    let (engine, app) = serve(vec![text("a"), Step::UntilCancelled]);
    let mut extra = no_thinking();
    extra["stream"] = json!(true);
    let response = post(&app, RESPONSES, &body(extra)).await;
    let mut stream = response.into_body().into_data_stream();
    let first = stream.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&first).contains("response.created"));
    assert!(engine.cancelled().is_empty(), "the client is still reading");
    drop(stream);
    eventually("the cancel", || !engine.cancelled().is_empty()).await;
    assert_eq!(engine.cancelled(), [engine.submitted()[0].id]);
}

// --- the counts ------------------------------------------------------------------

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

/// The answer of a count: its one number.
async fn counted(app: &Router, path: &str, body: &str) -> u64 {
    let (status, counted) = json(post(app, path, body).await).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    assert_eq!(counted.as_object().unwrap().len(), 2, "{counted}");
    assert_eq!(counted["object"], "response.input_tokens");
    counted["input_tokens"].as_u64().expect("a count")
}

#[tokio::test]
async fn the_count_of_a_request_is_the_input_the_same_request_reports_when_run() {
    let app = ares_server::router(Arc::new(Counting));
    let conversation = json!([
        {"role": "developer", "content": "You are a coding agent."},
        {"role": "user", "content": [{"type": "input_text", "text": "Weather in Köln?"}]},
        {"type": "reasoning", "summary": [], "content": [
            {"type": "reasoning_text", "text": "I look it up."},
        ]},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather",
         "arguments": "{\"city\":\"Köln\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "12 °C"},
    ]);
    let bodies = [
        json!({}),
        json!({"instructions": "Be brief."}),
        json!({"instructions": "Be brief.", "tools": tools(), "tool_choice": "required",
            "input": conversation}),
        json!({"reasoning": {"effort": "low"}}),
        json!({"reasoning": {"effort": "none"}, "tools": tools(), "tool_choice": "none",
            "text": {"format": {"type": "json_object"}}}),
    ];
    let mut counts = Vec::new();
    for extra in bodies {
        let count = counted(&app, COUNT, &body(extra.clone())).await;
        let (_, response) = json(post(&app, RESPONSES, &body(extra.clone())).await).await;
        assert_eq!(response["usage"]["input_tokens"], count, "{extra}");

        // The fields of the generation do not change the count.
        let mut more = extra;
        more["stream"] = json!(true);
        more["max_output_tokens"] = json!(5);
        more["temperature"] = json!(0.3);
        assert_eq!(counted(&app, COUNT, &body(more)).await, count);
        counts.push(count);
    }
    counts.sort_unstable();
    counts.dedup();
    assert_eq!(counts.len(), 5, "every one of these prompts is another one");
}

#[tokio::test]
async fn the_count_of_a_chat_request_is_the_prompt_the_same_request_reports_when_run() {
    let app = ares_server::router(Arc::new(Counting));
    let chat_tools = json!([{"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]);
    let bodies = [
        json!({}),
        json!({"messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather in Köln?"},
            {"role": "assistant", "content": "", "reasoning_content": "I look it up.",
             "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\":\"Köln\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "12 °C"},
        ], "tools": chat_tools, "tool_choice": "required"}),
        json!({"reasoning_effort": "low"}),
        json!({"chat_template_kwargs": {"enable_thinking": false},
            "response_format": {"type": "json_object"}}),
    ];
    let mut counts = Vec::new();
    for extra in bodies {
        let mut body = json!({
            "model": "mock",
            "messages": [{"role": "user", "content": "How is the weather in Köln?"}],
        });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let count = counted(&app, CHAT_COUNT, &body.to_string()).await;
        let (status, answer) = json(post(&app, CHAT, &body.to_string()).await).await;
        assert_eq!(status, StatusCode::OK, "{answer}");
        assert_eq!(answer["usage"]["prompt_tokens"], count, "{body}");

        // The fields of the generation do not change the count.
        body["stream"] = json!(true);
        body["max_tokens"] = json!(5);
        body["n"] = json!(3);
        assert_eq!(counted(&app, CHAT_COUNT, &body.to_string()).await, count);
        counts.push(count);
    }
    counts.sort_unstable();
    counts.dedup();
    assert_eq!(counts.len(), 4, "every one of these prompts is another one");

    // What the chat route refuses, its count refuses.
    let wrong = json!({"model": "mock", "messages": []}).to_string();
    let message =
        refused(&app, CHAT_COUNT, &wrong, StatusCode::BAD_REQUEST, "invalid_request").await;
    assert!(message.contains("messages must not be empty"), "{message}");
}

#[tokio::test]
async fn a_count_generates_nothing_and_needs_an_engine_that_counts() {
    let chat = json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]});
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)]);
    assert!(counted(&app, COUNT, &body(json!({"tools": tools()}))).await > 0);
    assert!(counted(&app, CHAT_COUNT, &chat.to_string()).await > 0);
    assert!(engine.submitted().is_empty(), "nothing is generated for a count");

    let unknown = body(json!({"model": "nope"}));
    refused(&app, COUNT, &unknown, StatusCode::NOT_FOUND, "model_not_found").await;

    // The scripted engine alone does not count.
    let plain = ares_server::router(MockEngine::script(vec![done(FinishReason::Stop, 1, 0)]));
    for (path, body) in [(COUNT, body(json!({}))), (CHAT_COUNT, chat.to_string())] {
        let message =
            refused(&plain, path, &body, StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
        assert!(message.contains("does not count the tokens of a prompt"), "{message}");
    }
}
