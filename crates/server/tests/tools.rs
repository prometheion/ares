//! Tools in chat requests, against a scripted engine: what `tools`,
//! `tool_choice` and `parallel_tool_calls` come to when they reach the engine,
//! how the calls in a reply come back, whole and streamed, and what is
//! refused.

mod common;

use std::sync::Arc;

use ares_api::{
    ChatFormat, DecisionRequest, DecisionSink, Engine, EngineError, EngineStats, EventSink,
    FinishReason, GenerateRequest, ModelCard, Prompt, RequestId, Role, Shape, Start,
    ThinkingFormat, ToolFormat,
};
use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";

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

fn serve(steps: Vec<Step>, format: ChatFormat) -> (Arc<MockEngine>, axum::Router) {
    let inner = MockEngine::script(steps);
    let engine = Declaring { inner: inner.clone(), format };
    (inner, ares_server::router_with(Arc::new(engine), ares_server::Config::default()))
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "The weather of a city.",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        }},
        {"type": "function", "function": {"name": "now"}},
    ])
}

fn chat(extra: Value) -> String {
    let mut body = json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "How is the weather in Köln?"}],
        "tools": tools(),
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

/// A reply as a client puts it together.
#[derive(Debug, Default, PartialEq)]
struct Reply {
    content: String,
    reasoning: String,
    /// Each call: its name and its arguments.
    calls: Vec<(String, String)>,
    finish: String,
}

async fn whole(steps: Vec<Step>, extra: Value) -> Reply {
    let (_, app) = serve(steps, qwen());
    let (status, body) = json(post(&app, CHAT, &chat(extra)).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let choice = &body["choices"][0];
    let message = &choice["message"];
    let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
    let calls = message["tool_calls"].as_array().cloned().unwrap_or_default();
    for call in &calls {
        assert_eq!(call["type"], "function", "{call}");
        assert!(call["id"].as_str().is_some_and(|id| id.starts_with("call_")), "{call}");
        assert!(call.get("index").is_none(), "a message has no index: {call}");
    }
    Reply {
        content: text(&message["content"]),
        reasoning: text(&message["reasoning_content"]),
        calls: calls
            .iter()
            .map(|call| (text(&call["function"]["name"]), text(&call["function"]["arguments"])))
            .collect(),
        finish: text(&choice["finish_reason"]),
    }
}

/// The same request streamed, with the deltas put together and checked for
/// the shape OpenAI gives them.
async fn streamed(steps: Vec<Step>, extra: Value) -> Reply {
    let (_, app) = serve(steps, qwen());
    let mut extra = extra;
    extra["stream"] = json!(true);
    let response = post(&app, CHAT, &chat(extra)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let stream = parse_sse(&body_text(response).await);
    assert!(stream.done, "the stream ends with [DONE]");
    let mut reply = Reply::default();
    let mut ids: Vec<String> = Vec::new();
    for chunk in &stream.chunks {
        let choice = &chunk["choices"][0];
        let delta = &choice["delta"];
        let answering = !reply.content.is_empty() || !reply.calls.is_empty();
        if let Some(content) = delta["content"].as_str() {
            assert!(
                !content.contains("tool_call") && !content.contains("function"),
                "markup in a content delta: {content:?}"
            );
            reply.content.push_str(content);
        }
        if let Some(reasoning) = delta["reasoning_content"].as_str() {
            assert!(
                !reasoning.contains("tool_call") && !reasoning.contains("function"),
                "markup in a reasoning delta: {reasoning:?}"
            );
            assert!(!answering, "reasoning after the answer began: {chunk}");
            reply.reasoning.push_str(reasoning);
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            let index = call["index"].as_u64().expect("an index") as usize;
            let function = &call["function"];
            if index == reply.calls.len() {
                // The first delta of a call: its id, its type and the name.
                assert_eq!(call["type"], "function", "{call}");
                ids.push(call["id"].as_str().expect("an id").to_owned());
                let name = function["name"].as_str().expect("the name first");
                assert_eq!(function["arguments"], "", "{call}");
                reply.calls.push((name.to_owned(), String::new()));
            } else {
                assert_eq!(index + 1, reply.calls.len(), "a fragment of the call that is open");
                assert!(call.get("id").is_none() && call.get("type").is_none(), "{call}");
                assert!(function.get("name").is_none(), "{call}");
                reply.calls[index].1.push_str(function["arguments"].as_str().unwrap());
            }
        }
        if let Some(finish) = choice["finish_reason"].as_str() {
            reply.finish = finish.to_owned();
        }
    }
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), reply.calls.len(), "every call has an id of its own");
    reply
}

async fn submitted(extra: Value, format: ChatFormat) -> GenerateRequest {
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)], format);
    let (status, body) = json(post(&app, CHAT, &chat(extra)).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    engine.submitted().remove(0)
}

async fn refused(body: &str, format: ChatFormat) -> Value {
    refused_on(CHAT, body, format).await
}

async fn refused_on(path: &str, body: &str, format: ChatFormat) -> Value {
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)], format);
    let (status, value) = json(post(&app, path, body).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{value}");
    assert!(engine.submitted().is_empty(), "the engine was not to see the request");
    value["error"].clone()
}

fn lark(request: &GenerateRequest) -> &str {
    match &request.structure.as_ref().expect("a rule for the reply").shape {
        Shape::Lark(text) => text,
        other => panic!("not a grammar of calls: {other:?}"),
    }
}

#[tokio::test]
async fn tools_reach_the_template_and_the_reply_gets_the_rule_of_the_tool_choice() {
    let after_reasoning = Start::AfterReasoning { open: None, close: "</think>".into() };

    // auto, which is the default: free until the trigger, then calls.
    for extra in [json!({}), json!({"tool_choice": "auto"})] {
        let request = submitted(extra, qwen()).await;
        let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
        assert_eq!(serde_json::to_value(sent).unwrap(), tools(), "the tools as the client sent");
        let structure = request.structure.as_ref().unwrap();
        assert_eq!(structure.trigger.as_deref(), Some("<tool_call>"));
        assert_eq!(structure.start, after_reasoning, "the reasoning is free");
        // The marker inside the reasoning ends it, and the same rule binds.
        let interrupt = structure.interrupt.as_ref().expect("an interrupt");
        assert_eq!((interrupt.text.as_str(), &interrupt.shape), ("<tool_call>", &structure.shape));
        let grammar = lark(&request);
        assert!(grammar.starts_with("start: first more\n"), "{grammar}");
        assert!(grammar.contains(r#""\n<function=get_weather>\n""#), "{grammar}");
        assert!(grammar.contains(r#""<tool_call>\n<function=now>\n""#), "{grammar}");
        assert!(grammar.contains("more: (WS? next)* WS?\n"), "several calls: {grammar}");
        assert!(grammar.contains(r#"%json {"type":"integer""#), "{grammar}");
    }

    // required: calls from the first token of the answer.
    let request = submitted(json!({"tool_choice": "required"}), qwen()).await;
    let structure = request.structure.as_ref().unwrap();
    assert_eq!((structure.trigger.as_deref(), &structure.start), (None, &after_reasoning));
    assert!(lark(&request).starts_with("start: WS? next more\n"));
    assert!(lark(&request).contains("<function=now>"));
    // A call opened inside the reasoning goes on by the rule of a call
    // whose marker is written.
    let opened = |request: &GenerateRequest| {
        let interrupt =
            request.structure.as_ref().unwrap().interrupt.clone().expect("an interrupt");
        assert_eq!(interrupt.text, "<tool_call>");
        let Shape::Lark(grammar) = interrupt.shape else { panic!("not a grammar of calls") };
        assert!(grammar.starts_with("start: first more\n"), "{grammar}");
        grammar
    };
    let grammar = opened(&request);
    assert!(grammar.contains(r#""\n<function=get_weather>\n""#) && grammar.contains("=now>"));

    // A named function: the same, with that function alone.
    let choice = json!({"type": "function", "function": {"name": "now"}});
    let request = submitted(json!({"tool_choice": choice}), qwen()).await;
    assert_eq!(request.structure.as_ref().unwrap().trigger, None);
    let grammar = lark(&request);
    assert!(grammar.starts_with("start: WS? next more\n") && grammar.contains("<function=now>"));
    assert!(!grammar.contains("get_weather"), "{grammar}");
    let inside = opened(&request);
    assert!(inside.contains("<function=now>") && !inside.contains("get_weather"), "{inside}");
    let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
    assert_eq!(sent.as_ref().map(Vec::len), Some(2), "the prompt has all the tools");

    // One call at most, inside the reasoning too.
    let request = submitted(json!({"parallel_tool_calls": false}), qwen()).await;
    assert!(lark(&request).contains("more: WS?\n"), "{}", lark(&request));
    let one = json!({"parallel_tool_calls": false, "tool_choice": "required"});
    assert!(opened(&submitted(one, qwen()).await).contains("more: WS?\n"));

    // none: the tools are in the prompt and the reply is free, as with
    // llama-server.
    let request = submitted(json!({"tool_choice": "none"}), qwen()).await;
    assert_eq!(request.structure, None);
    let Prompt::Chat { tools: sent, .. } = &request.prompt else { panic!() };
    assert_eq!(sent.as_ref().map(Vec::len), Some(2));

    // Thinking off: the rule binds from the first token of the reply.
    let off = json!({"chat_template_kwargs": {"enable_thinking": false}});
    let request = submitted(off, qwen()).await;
    let structure = request.structure.as_ref().unwrap();
    assert_eq!((&structure.start, &structure.interrupt), (&Start::Reply, &None));
}

#[tokio::test]
async fn a_reply_that_calls_a_tool_comes_back_as_tool_calls() {
    let said = format!("{THINKS}I will look it up.\n\n{CALL}");
    let reply = whole(says(&said, 7, FinishReason::Stop), json!({})).await;
    assert_eq!(reply.reasoning, "The user asks for the weather.");
    assert_eq!(reply.content, "I will look it up.");
    assert_eq!(reply.finish, "tool_calls");
    assert_eq!(reply.calls, [("get_weather".to_owned(), r#"{"city":"Köln","days":3}"#.to_owned())]);

    // A reply that is only a call has the empty content.
    let (_, app) = serve(says(&format!("{THINKS}{NOW}"), 5, FinishReason::Stop), qwen());
    let (_, body) = json(post(&app, CHAT, &chat(json!({}))).await).await;
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], "");
    assert_eq!(message["tool_calls"][0]["function"], json!({"name": "now", "arguments": "{}"}));
    assert_eq!(body["usage"]["completion_tokens"], format!("{THINKS}{NOW}").chars().count());
}

#[tokio::test]
async fn a_reply_without_a_call_is_text_as_before() {
    let said = format!("{THINKS}It is sunny, 12 < 14.\n");
    let reply = whole(says(&said, 4, FinishReason::Stop), json!({})).await;
    assert_eq!(reply.content, "It is sunny, 12 < 14.\n");
    assert_eq!((reply.calls.len(), reply.finish.as_str()), (0, "stop"));
    let (_, app) = serve(says(&said, 4, FinishReason::Stop), qwen());
    let (_, body) = json(post(&app, CHAT, &chat(json!({}))).await).await;
    assert!(body["choices"][0]["message"].get("tool_calls").is_none(), "{body}");
}

#[tokio::test]
async fn several_calls_come_back_in_order_and_a_stream_equals_the_whole_reply() {
    let said = format!("{THINKS}Two things.\n\n{CALL}\n{NOW}\n");
    let expected = Reply {
        content: "Two things.".into(),
        reasoning: "The user asks for the weather.".into(),
        calls: vec![
            ("get_weather".into(), r#"{"city":"Köln","days":3}"#.into()),
            ("now".into(), "{}".into()),
        ],
        finish: "tool_calls".into(),
    };
    // However the engine cuts the text: every marker is split somewhere.
    for size in [1, 2, 3, 5, 11, 1000] {
        let reply = whole(says(&said, size, FinishReason::Stop), json!({})).await;
        assert_eq!(reply, expected, "pieces of {size}");
        let reply = streamed(says(&said, size, FinishReason::Stop), json!({})).await;
        assert_eq!(reply, expected, "streamed in pieces of {size}");
    }
}

#[tokio::test]
async fn a_forced_call_comes_back_the_same_way() {
    // The answer is a call from its first token, after the template's blank
    // line; thinking off, the reply starts with it.
    let said = format!("{THINKS}{CALL}");
    for extra in [
        json!({"tool_choice": "required"}),
        json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
    ] {
        let reply = streamed(says(&said, 3, FinishReason::Stop), extra.clone()).await;
        assert_eq!(reply.calls.len(), 1, "{reply:?}");
        assert_eq!((reply.content.as_str(), reply.finish.as_str()), ("", "tool_calls"));
        assert_eq!(reply, whole(says(&said, 3, FinishReason::Stop), extra).await);
    }
    let off =
        json!({"tool_choice": "required", "chat_template_kwargs": {"enable_thinking": false}});
    let reply = whole(says(CALL, 3, FinishReason::Stop), off).await;
    assert_eq!((reply.calls.len(), reply.reasoning.as_str()), (1, ""));
}

#[tokio::test]
async fn with_tool_choice_none_a_call_the_model_writes_stays_text() {
    let said = format!("{THINKS}{CALL}");
    for stream in [false, true] {
        let steps = says(&said, 6, FinishReason::Stop);
        let extra = json!({"tool_choice": "none"});
        let reply =
            if stream { streamed_raw(steps, extra).await } else { whole(steps, extra).await };
        assert_eq!(reply.content, CALL);
        assert_eq!((reply.calls.len(), reply.finish.as_str()), (0, "stop"));
    }
}

/// The model opens the call while it still thinks: no closing tag.
const THINKS_ON: &str = "The user asks for the weather.";

#[tokio::test]
async fn a_call_inside_the_thinking_comes_back_as_a_call_and_ends_the_reasoning() {
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    let choices = [
        json!({}),
        json!({"tool_choice": "auto"}),
        json!({"tool_choice": "required"}),
        json!({"tool_choice": named}),
    ];
    let expected = Reply {
        content: String::new(),
        reasoning: THINKS_ON.into(),
        calls: vec![("get_weather".into(), r#"{"city":"Köln","days":3}"#.into())],
        finish: "tool_calls".into(),
    };
    for extra in choices {
        // The white space before the marker is neither the reasoning's nor
        // the call's, whatever the model writes there.
        for space in ["\n\n", "\n", "", " "] {
            let said = format!("{THINKS_ON}{space}{CALL}");
            // However the engine cuts the text: the marker is split somewhere.
            for size in [1, 2, 3, 5, 11, 1000] {
                let reply = whole(says(&said, size, FinishReason::Stop), extra.clone()).await;
                assert_eq!(reply, expected, "{extra} {space:?} in pieces of {size}");
                let reply = streamed(says(&said, size, FinishReason::Stop), extra.clone()).await;
                assert_eq!(reply, expected, "{extra} {space:?} streamed in pieces of {size}");
            }
        }
    }

    // Several calls, the first of them inside the thinking; and a reply
    // that opens a call at once has no reasoning.
    let said = format!("{THINKS_ON}\n\n{CALL}\n{NOW}\n");
    let reply = streamed(says(&said, 4, FinishReason::Stop), json!({})).await;
    assert_eq!(reply.reasoning, THINKS_ON);
    assert_eq!(reply.calls.len(), 2, "{reply:?}");
    assert_eq!((reply.content.as_str(), reply.finish.as_str()), ("", "tool_calls"));
    assert_eq!(reply, whole(says(&said, 4, FinishReason::Stop), json!({})).await);
    let (_, app) = serve(says(NOW, 3, FinishReason::Stop), qwen());
    let (_, body) = json(post(&app, CHAT, &chat(json!({}))).await).await;
    let message = &body["choices"][0]["message"];
    assert!(message.get("reasoning_content").is_none_or(|r| r == ""), "{body}");
    assert_eq!(message["tool_calls"][0]["function"]["name"], "now");
    assert_eq!(message["content"], "");
}

#[tokio::test]
async fn a_call_inside_the_thinking_that_the_limit_cuts_is_returned_as_far_as_it_came() {
    let cut = CALL.find("</parameter>").unwrap() - 2;
    let said = format!("{THINKS_ON}\n\n{}", &CALL[..cut]);
    let expected = Reply {
        content: String::new(),
        reasoning: THINKS_ON.into(),
        calls: vec![("get_weather".into(), r#"{"city":"Köl"#.into())],
        finish: "length".into(),
    };
    for size in [1, 4, 1000] {
        assert_eq!(whole(says(&said, size, FinishReason::Length), json!({})).await, expected);
        assert_eq!(streamed(says(&said, size, FinishReason::Length), json!({})).await, expected);
    }
    // Cut right after the marker, or inside it: no call. The whole marker
    // has ended the reasoning; a part of one is text of it.
    for (end, reasoning) in
        [("\n<tool_call>", THINKS_ON.to_owned()), ("\n<tool", format!("{THINKS_ON}\n<tool"))]
    {
        let said = format!("{THINKS_ON}{end}");
        let reply = whole(says(&said, 2, FinishReason::Length), json!({})).await;
        assert_eq!(reply, Reply { reasoning, finish: "length".into(), ..Reply::default() });
    }
}

#[tokio::test]
async fn with_tool_choice_none_or_without_tools_a_call_inside_the_thinking_is_reasoning() {
    let said = format!("I could call.\n\n{CALL}\nBut I answer.\n</think>\n\nIt is sunny.");
    let expected = Reply {
        content: "It is sunny.".into(),
        reasoning: format!("I could call.\n\n{CALL}\nBut I answer."),
        calls: Vec::new(),
        finish: "stop".into(),
    };
    for extra in [json!({"tool_choice": "none"}), json!({"tools": null}), json!({"tools": []})] {
        for size in [1, 6, 1000] {
            let steps = says(&said, size, FinishReason::Stop);
            assert_eq!(whole(steps, extra.clone()).await, expected, "{extra}");
            // The stream, with its reasoning as it is.
            let (_, app) = serve(says(&said, size, FinishReason::Stop), qwen());
            let mut body = extra.clone();
            body["stream"] = json!(true);
            let stream = parse_sse(&body_text(post(&app, CHAT, &chat(body)).await).await);
            let mut reply = Reply { finish: "stop".into(), ..Reply::default() };
            for chunk in &stream.chunks {
                let delta = &chunk["choices"][0]["delta"];
                assert!(delta.get("tool_calls").is_none(), "{chunk}");
                reply.content.push_str(delta["content"].as_str().unwrap_or_default());
                reply.reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
            }
            assert_eq!(reply, expected, "{extra} streamed in pieces of {size}");
        }
    }
    // A reply that never closes its thinking is reasoning to its end.
    let said = format!("I could call.\n\n{CALL}");
    let reply = whole(says(&said, 5, FinishReason::Stop), json!({"tool_choice": "none"})).await;
    assert_eq!(reply, Reply { reasoning: said, finish: "stop".into(), ..Reply::default() });
}

#[tokio::test]
async fn a_raw_reply_has_the_call_it_opened_inside_the_thinking() {
    // `reasoning_format: none`: the thinking stays in the content, from the
    // tag the prompt ended with on, and the call is a call.
    let said = format!("{THINKS_ON}\n\n{CALL}");
    let extra = json!({"reasoning_format": "none"});
    let expected = Reply {
        content: format!("<think>\n{THINKS_ON}"),
        reasoning: String::new(),
        calls: vec![("get_weather".into(), r#"{"city":"Köln","days":3}"#.into())],
        finish: "tool_calls".into(),
    };
    for size in [1, 7, 1000] {
        let reply = whole(says(&said, size, FinishReason::Stop), extra.clone()).await;
        assert_eq!(reply, expected, "pieces of {size}");
        let reply = streamed(says(&said, size, FinishReason::Stop), extra.clone()).await;
        assert_eq!(reply, expected, "streamed in pieces of {size}");
    }
}

/// A stream whose content may hold markup.
async fn streamed_raw(steps: Vec<Step>, extra: Value) -> Reply {
    let (_, app) = serve(steps, qwen());
    let mut extra = extra;
    extra["stream"] = json!(true);
    let stream = parse_sse(&body_text(post(&app, CHAT, &chat(extra)).await).await);
    let mut reply = Reply::default();
    for chunk in &stream.chunks {
        let choice = &chunk["choices"][0];
        assert!(choice["delta"].get("tool_calls").is_none(), "{chunk}");
        reply.content.push_str(choice["delta"]["content"].as_str().unwrap_or_default());
        reply.reasoning.push_str(choice["delta"]["reasoning_content"].as_str().unwrap_or_default());
        if let Some(finish) = choice["finish_reason"].as_str() {
            reply.finish = finish.to_owned();
        }
    }
    reply.reasoning.clear();
    reply
}

#[tokio::test]
async fn a_call_the_limit_of_tokens_cuts_is_returned_as_far_as_it_came() {
    // llama-server and OpenAI do the same: `length`, and arguments that are
    // not whole JSON.
    let cut = CALL.find("</parameter>").unwrap() - 2;
    let said = format!("{THINKS}Looking.\n\n{}", &CALL[..cut]);
    let expected = Reply {
        content: "Looking.".into(),
        reasoning: "The user asks for the weather.".into(),
        calls: vec![("get_weather".into(), r#"{"city":"Köl"#.into())],
        finish: "length".into(),
    };
    for size in [1, 4, 1000] {
        assert_eq!(whole(says(&said, size, FinishReason::Length), json!({})).await, expected);
        assert_eq!(streamed(says(&said, size, FinishReason::Length), json!({})).await, expected);
    }
    // Cut before the name is whole: there is no call, and no markup either.
    let said = format!("{THINKS}Looking.\n\n<tool_call>\n<function=get_wea");
    let reply = streamed(says(&said, 2, FinishReason::Length), json!({})).await;
    assert_eq!((reply.content.as_str(), reply.calls.len()), ("Looking.", 0));
    assert_eq!(reply.finish, "length");
    // A whole call and then the limit: the call is there, the reason is the
    // limit.
    let said = format!("{THINKS}{CALL}\n<tool_call>\n<function=now>\n");
    let reply = whole(says(&said, 9, FinishReason::Length), json!({})).await;
    assert_eq!(reply.finish, "length");
    assert_eq!(reply.calls[0].1, r#"{"city":"Köln","days":3}"#);
    assert_eq!(reply.calls[1], ("now".to_owned(), "{".to_owned()));
}

#[tokio::test]
async fn a_call_a_stop_string_cuts_is_returned_as_llama_server_returns_it() {
    // The engine ends the reply before the stop string; the reason is the
    // one of a reply that ended with calls.
    let cut = CALL.find("\n</parameter>").unwrap();
    let said = format!("{THINKS}{}", &CALL[..cut]);
    let extra = json!({"stop": ["\n</parameter>"]});
    let reply = whole(says(&said, 5, FinishReason::Stop), extra.clone()).await;
    assert_eq!(reply.calls, [("get_weather".to_owned(), r#"{"city":"Köln"#.to_owned())]);
    assert_eq!(reply.finish, "tool_calls");
    assert_eq!(streamed(says(&said, 5, FinishReason::Stop), extra).await, reply);
}

#[tokio::test]
async fn the_calls_and_results_of_earlier_turns_reach_the_template() {
    let (engine, app) = serve(vec![done(FinishReason::Stop, 1, 0)], qwen());
    let calls = json!([
        {"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"Köln\"}"}},
        {"id": "call_2", "type": "function", "function": {"name": "now", "arguments": {}}},
    ]);
    let body = json!({
        "model": "mock",
        "tools": tools(),
        "messages": [
            {"role": "user", "content": "How is the weather?"},
            {"role": "assistant", "content": null, "reasoning_content": "I look.",
             "tool_calls": calls},
            {"role": "tool", "tool_call_id": "call_1", "content": "12 degrees"},
            {"role": "tool", "tool_call_id": "call_2", "content": "noon"},
        ],
    });
    let (status, answer) = json(post(&app, CHAT, &body.to_string()).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let request = engine.submitted().remove(0);
    let Prompt::Chat { messages, .. } = &request.prompt else { panic!() };
    assert_eq!(messages.len(), 4);
    assert_eq!(serde_json::to_value(&messages[1].tool_calls).unwrap(), calls);
    assert_eq!(messages[1].content, "");
    assert_eq!(messages[1].reasoning_content.as_deref(), Some("I look."));
    assert_eq!(
        (messages[2].role, messages[2].tool_call_id.as_deref()),
        (Role::Tool, Some("call_1"))
    );
    assert_eq!(messages[3].content, "noon");

    // A call the template could not write is refused by its place.
    for (bad, what) in [
        (json!([{"id": "c", "type": "function"}]), "function must be an object"),
        (json!([{"function": {"arguments": "{}"}}]), "function.name"),
        (json!([{"function": {"name": "now", "arguments": "not json"}}]), "function.arguments"),
        (json!([{"function": {"name": "now", "arguments": "[1]"}}]), "function.arguments"),
    ] {
        let body = json!({"model": "mock", "messages": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": bad}]});
        let error = refused(&body.to_string(), qwen()).await;
        assert_eq!(error["param"], "messages[1].tool_calls[0]", "{error}");
        assert!(error["message"].as_str().unwrap().contains(what), "{error}");
    }
}

#[tokio::test]
async fn what_cannot_be_served_is_refused_with_its_reason() {
    let cases = [
        // A rule of its own next to the tools: llama-server refuses the
        // grammar and lets a schema win over the tools without a word.
        (json!({"grammar": "root ::= \"a\""}), "tools", "cannot be combined"),
        (json!({"json_schema": {"type": "object"}}), "tools", "cannot be combined"),
        (json!({"response_format": {"type": "json_object"}}), "tools", "cannot be combined"),
        (json!({"tool_choice": "any"}), "tool_choice", "must be \"auto\""),
        (json!({"tool_choice": 1}), "tool_choice", "a string or an object"),
        (json!({"tool_choice": {"type": "function"}}), "tool_choice", "as an object"),
        (
            json!({"tool_choice": {"type": "function", "function": {"name": "get_time"}}}),
            "tool_choice",
            "not among the tools",
        ),
        (json!({"tool_choice": "required", "tools": []}), "tool_choice", "has no tools"),
        (json!({"parallel_tool_calls": "no"}), "parallel_tool_calls", "a boolean"),
        (json!({"tools": "none"}), "tools", "must be an array"),
        (json!({"tools": [{"type": "web_search"}]}), "tools", "other kinds of tools"),
        (json!({"tools": [{"type": "function"}]}), "tools", "function must be an object"),
        (json!({"tools": [{"type": "function", "function": {"name": ""}}]}), "tools", "name"),
        (
            json!({"tools": [{"type": "function", "function": {"name": "f", "parameters": 3}}]}),
            "tools",
            "parameters must be an object",
        ),
        (
            json!({"tools": [{"type": "function", "function": {"name": "f"}},
                             {"type": "function", "function": {"name": "f"}}]}),
            "tools",
            "two tools named 'f'",
        ),
        (
            json!({"tools": [{"type": "function", "function": {"name": "f>\ng"}}]}),
            "tools",
            "cannot be written in the model's format",
        ),
        (json!({"functions": [{"name": "f"}]}), "functions", "send tools and tool_choice"),
        (json!({"function_call": "auto"}), "function_call", "send tools and tool_choice"),
    ];
    for (extra, param, what) in cases {
        let error = refused(&chat(extra.clone()), qwen()).await;
        assert_eq!(error["param"], param, "{extra}: {error}");
        assert!(error["message"].as_str().unwrap().contains(what), "{extra}: {error}");
    }

    // A model whose template writes calls in no form the server reads.
    let plain = ChatFormat { tools: None, ..qwen() };
    let error = refused(&chat(json!({})), plain.clone()).await;
    assert_eq!(error["param"], "tools");
    assert!(error["message"].as_str().unwrap().contains("chat template"), "{error}");
    // Without tools, or with none of them to be called, it is served.
    for extra in [json!({"tools": []}), json!({"tools": null}), json!({"tool_choice": "none"})] {
        assert_eq!(submitted(extra, plain.clone()).await.structure, None);
    }
    // A rule of its own with tools that are not to be called is that rule.
    let extra = json!({"tool_choice": "none", "grammar": "root ::= \"a\""});
    let structure = submitted(extra, qwen()).await.structure.expect("the grammar");
    assert!(matches!(structure.shape, Shape::Gbnf(_)) && structure.trigger.is_none());

    // Completions have no chat template and no tools.
    for field in ["tools", "tool_choice", "parallel_tool_calls"] {
        let mut body = json!({"model": "mock", "prompt": "Once"});
        body[field] = match field {
            "tools" => tools(),
            "tool_choice" => json!("auto"),
            _ => json!(true),
        };
        let error = refused_on("/v1/completions", &body.to_string(), qwen()).await;
        assert_eq!(error["param"], field, "{error}");
        assert!(error["message"].as_str().unwrap().contains("chat requests"), "{error}");
    }
}

#[tokio::test]
async fn each_completion_of_n_has_calls_with_ids_of_its_own() {
    let said = format!("{THINKS}{NOW}");
    let (_, app) = serve(says(&said, 8, FinishReason::Stop), qwen());
    let (status, body) = json(post(&app, CHAT, &chat(json!({"n": 2}))).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = |choice: usize| body["choices"][choice]["message"]["tool_calls"][0]["id"].clone();
    assert!(id(0).is_string() && id(1).is_string(), "{body}");
    assert_ne!(id(0), id(1));
}
