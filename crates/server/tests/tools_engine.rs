//! Tool calls from the request to the answer on the host: the server over the
//! real runtime, with the synthetic model on a host executor. The rule the
//! server writes for the tools is followed by the grammar engine behind the
//! sampler's mask, and the server reads the calls out of what the model
//! generated under it.
//!
//! The synthetic model has one token for each byte, no line break among its
//! printable bytes and no chat template, so it declares a format for calls
//! with blanks where the format of Qwen3.8's template has line breaks, and
//! with a trigger a test picks from the text the model writes anyway. For the
//! tests of a call inside the thinking it declares thinking too, with a
//! closing tag that it writes late or never.

mod common;

use std::sync::Arc;
use std::time::Duration;

use ares_api::{ChatFormat, Engine, ThinkingFormat, ToolFormat};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use axum::Router;
use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";

/// The server over a runtime whose model writes calls after `trigger`.
fn served(trigger: Option<&str>) -> Router {
    served_thinking(trigger, None)
}

/// The same, with a model that thinks until it writes `close`: the prompt
/// has opened the block.
fn served_thinking(trigger: Option<&str>, close: Option<&str>) -> Router {
    served_ending(trigger, close, b'~')
}

/// The same, with a model whose replies end at the character `end`.
fn served_ending(trigger: Option<&str>, close: Option<&str>, end: u8) -> Router {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.eos = Some(u32::from(end));
    // Text that loops, so that the n-gram drafter has drafts to offer,
    // next to an MTP block of the model's.
    config.context.window = 1;
    config.context.epoch = 40;
    config.context.buckets = 512;
    config.mtp_links = 7;
    config.mtp_accuracy = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
    config.chat = trigger.map(|trigger| ChatFormat {
        thinking: close.map(|close| ThinkingFormat {
            start: "<think>".into(),
            end: close.into(),
            prompt_on: "<think>".into(),
            prompt_off: format!("<think>{close}"),
            on_by_default: true,
            effort: false,
            refused_efforts: Vec::new(),
        }),
        tools: Some(ToolFormat {
            call_open: format!("{trigger} <function="),
            name_close: "> ".into(),
            param_open: "<parameter=".into(),
            key_close: "> ".into(),
            param_close: " </parameter> ".into(),
            call_close: "</function> </tool_call>".into(),
            before_calls: "  ".into(),
            between_calls: " ".into(),
        }),
    });
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let config = RuntimeConfig {
        pages: 4096,
        cells_per_page: 16,
        max_slots: 64,
        window_rows: 8,
        warm_up: false,
        ..RuntimeConfig::default()
    };
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_row_cost(Duration::from_micros(3000), Duration::from_micros(30));
    let engine: Arc<dyn Engine> = Arc::new(Runtime::start(model, Box::new(host), config).unwrap());
    ares_server::router_with(engine, ares_server::Config::default())
}

fn tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "set_volume",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "minimum": 0, "maximum": 99},
                    "mode": {"enum": ["quiet", "loud", "off"]},
                    "on": {"type": "boolean"},
                },
                "required": ["level", "mode"],
            },
        }},
        {"type": "function", "function": {"name": "mute"}},
    ])
}

fn body(extra: Value) -> Value {
    let mut body = json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "turn it down"}],
        "max_tokens": 900,
        "temperature": 0.8,
        "seed": 5,
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

/// A reply as a client puts it together: content, each call as its name and
/// its arguments, and the reason it ended.
#[derive(Debug, PartialEq)]
struct Reply {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
    finish: String,
}

async fn whole(app: &Router, extra: Value) -> Reply {
    let (status, answer) = json(post(app, CHAT, &body(extra).to_string()).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let choice = &answer["choices"][0];
    let calls = choice["message"]["tool_calls"].as_array().cloned().unwrap_or_default();
    Reply {
        reasoning: choice["message"]["reasoning_content"].as_str().unwrap_or_default().to_owned(),
        content: choice["message"]["content"].as_str().unwrap().to_owned(),
        calls: calls
            .iter()
            .map(|call| {
                let function = &call["function"];
                let arguments = function["arguments"].as_str().unwrap();
                let arguments = serde_json::from_str(arguments)
                    .unwrap_or_else(|e| panic!("the arguments are not JSON ({e}): {arguments}"));
                (function["name"].as_str().unwrap().to_owned(), arguments)
            })
            .collect(),
        finish: choice["finish_reason"].as_str().unwrap().to_owned(),
    }
}

async fn streamed(app: &Router, extra: Value) -> Reply {
    let mut extra = extra;
    extra["stream"] = json!(true);
    let response = post(app, CHAT, &body(extra).to_string()).await;
    assert_eq!(response.status(), StatusCode::OK);
    let stream = parse_sse(&body_text(response).await);
    assert!(stream.done);
    let mut reply = Reply {
        reasoning: String::new(),
        content: String::new(),
        calls: Vec::new(),
        finish: String::new(),
    };
    let mut arguments: Vec<String> = Vec::new();
    for chunk in &stream.chunks {
        let choice = &chunk["choices"][0];
        let reasoning = choice["delta"]["reasoning_content"].as_str().unwrap_or_default();
        reply.reasoning.push_str(reasoning);
        reply.content.push_str(choice["delta"]["content"].as_str().unwrap_or_default());
        for call in choice["delta"]["tool_calls"].as_array().into_iter().flatten() {
            let index = call["index"].as_u64().unwrap() as usize;
            if let Some(name) = call["function"]["name"].as_str() {
                assert_eq!(index, reply.calls.len());
                reply.calls.push((name.to_owned(), Value::Null));
                arguments.push(String::new());
            }
            arguments[index].push_str(call["function"]["arguments"].as_str().unwrap());
        }
        if let Some(finish) = choice["finish_reason"].as_str() {
            reply.finish = finish.to_owned();
        }
    }
    for (call, arguments) in reply.calls.iter_mut().zip(&arguments) {
        call.1 = serde_json::from_str(arguments)
            .unwrap_or_else(|e| panic!("the arguments are not JSON ({e}): {arguments}"));
    }
    reply
}

/// The calls are of the tools of the request, with arguments of their schema.
fn check(calls: &[(String, Value)]) {
    for (name, arguments) in calls {
        let arguments = arguments.as_object().expect("an object");
        match name.as_str() {
            "mute" => assert!(arguments.is_empty(), "{arguments:?}"),
            "set_volume" => {
                let level = arguments["level"].as_i64().expect("an integer");
                assert!((0..=99).contains(&level), "{level}");
                let mode = arguments["mode"].as_str().expect("a string");
                assert!(["quiet", "loud", "off"].contains(&mode), "{mode}");
                assert!(arguments.get("on").is_none_or(Value::is_boolean), "{arguments:?}");
                assert!(
                    arguments.keys().all(|key| ["level", "mode", "on"].contains(&key.as_str()))
                );
            }
            other => panic!("a tool nobody offered: {other}"),
        }
    }
}

/// The reply to `extra` three ways: whole, streamed, and without drafts. They
/// are one reply.
async fn reply(app: &Router, extra: Value) -> Reply {
    let reply = whole(app, extra.clone()).await;
    assert_eq!(streamed(app, extra.clone()).await, reply, "the stream is the whole reply");
    let mut plain = extra;
    plain["draft"] = json!(false);
    assert_eq!(whole(app, plain).await, reply, "the reply with drafts is the reply without");
    reply
}

#[tokio::test(flavor = "multi_thread")]
async fn a_required_call_is_a_call_of_the_tools_with_arguments_of_their_schema() {
    let app = served(Some("<tool_call>"));
    let one = json!({"tools": tools(), "tool_choice": "required", "parallel_tool_calls": false});
    let got = reply(&app, one).await;
    assert_eq!((got.content.as_str(), got.finish.as_str()), ("", "tool_calls"), "{got:?}");
    assert_eq!(got.calls.len(), 1, "{got:?}");
    check(&got.calls);

    // A named function is that function.
    for name in ["set_volume", "mute"] {
        let choice = json!({"type": "function", "function": {"name": name}});
        let named = json!({"tools": tools(), "tool_choice": choice, "parallel_tool_calls": false});
        let got = reply(&app, named).await;
        assert_eq!(got.calls.len(), 1, "{got:?}");
        assert_eq!(got.calls[0].0, name);
        check(&got.calls);
    }

    // Several calls when they are allowed: at least one, all of them valid.
    let many = json!({"tools": tools(), "tool_choice": "required", "max_tokens": 300});
    let got = whole(&app, many).await;
    assert!(!got.calls.is_empty(), "{got:?}");
    if got.finish == "tool_calls" {
        check(&got.calls);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn with_auto_the_text_is_free_and_a_call_follows_the_trigger() {
    // What the model writes when nothing binds it, and a character of it to
    // be the trigger: one that the text has not had before.
    let free = whole(&served(None), json!({"max_tokens": 60})).await;
    assert!(free.calls.is_empty());
    let text = &free.content;
    let (at, trigger) = text
        .char_indices()
        .skip(3)
        .find(|&(at, c)| c.is_ascii_graphic() && c != '<' && !text[..at].contains(c))
        .unwrap_or_else(|| panic!("no character to be the trigger in {text:?}"));

    let app = served(Some(&trigger.to_string()));
    let auto = json!({"tools": tools(), "parallel_tool_calls": false});
    let got = reply(&app, auto).await;
    // The same text up to the trigger, then one call and the end.
    assert_eq!(got.content, text[..at].trim_end(), "{got:?}");
    assert_eq!((got.calls.len(), got.finish.as_str()), (1, "tool_calls"), "{got:?}");
    check(&got.calls);

    // With tool_choice none the same model writes what it writes anyway.
    let none = json!({"tools": tools(), "tool_choice": "none", "max_tokens": 60});
    assert_eq!(whole(&app, none).await, free);
}

/// What the model writes when nothing binds it, and two characters of it:
/// each is new to the text where it first stands, the second after the first.
async fn free_text() -> (String, (usize, char), (usize, char)) {
    let free = whole(&served(None), json!({"max_tokens": 60})).await;
    let text = free.content;
    let new = |from: usize| {
        text.char_indices()
            .skip(from)
            .find(|&(at, c)| c.is_ascii_graphic() && c != '<' && !text[..at].contains(c))
            .unwrap_or_else(|| panic!("no new character after {from} in {text:?}"))
    };
    let first = new(3);
    let second = new(first.0 + 3);
    (text, first, second)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_the_model_opens_while_it_thinks_is_a_call_under_the_rule() {
    let (text, _, (at, marker)) = free_text().await;
    // The model thinks and never writes the closing tag: it writes the
    // marker of a call in the middle of its thinking.
    let app = served_thinking(Some(&marker.to_string()), Some("</think>"));

    // auto: the text up to the marker is the reasoning, then one call under
    // the rule and the end; whole, streamed and without drafts.
    let auto = json!({"tools": tools(), "parallel_tool_calls": false});
    let got = reply(&app, auto).await;
    assert_eq!(got.reasoning, text[..at].trim_end(), "{got:?}");
    assert_eq!((got.content.as_str(), got.finish.as_str()), ("", "tool_calls"), "{got:?}");
    assert_eq!(got.calls.len(), 1, "{got:?}");
    check(&got.calls);

    // required and a named function: the rows of the thinking carry a mask
    // (every token but the end), which takes nothing from this text; the
    // reply ends its reasoning at the marker and is a call.
    let named = json!({"type": "function", "function": {"name": "mute"}});
    for choice in [json!("required"), named] {
        let must = json!({"tools": tools(), "tool_choice": choice, "parallel_tool_calls": false});
        let got = reply(&app, must).await;
        assert_eq!(got.reasoning, text[..at].trim_end(), "{got:?}");
        assert_eq!((got.content.as_str(), got.finish.as_str()), ("", "tool_calls"), "{got:?}");
        assert_eq!(got.calls.len(), 1, "{got:?}");
        check(&got.calls);
        if choice.is_object() {
            assert_eq!(got.calls[0].0, "mute");
        }
    }

    // Several calls when they are allowed, the first of them in the thinking.
    let many = json!({"tools": tools(), "max_tokens": 300});
    let got = reply(&app, many).await;
    assert_eq!(got.reasoning, text[..at].trim_end(), "{got:?}");
    assert!(!got.calls.is_empty() && got.content.is_empty(), "{got:?}");
    if got.finish == "tool_calls" {
        check(&got.calls);
    }

    // none, and no tools: the marker is a character of the reasoning, and
    // the reply is the text the model writes anyway.
    for extra in [json!({"tools": tools(), "tool_choice": "none"}), json!({})] {
        let mut extra = extra;
        extra["max_tokens"] = json!(60);
        let got = reply(&app, extra).await;
        assert_eq!(got.reasoning, text, "{got:?}");
        assert!(got.reasoning.contains(marker) && got.calls.is_empty(), "{got:?}");
        assert_eq!(got.content, "");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_after_the_closing_tag_is_a_call_as_before() {
    let (text, (closed, close), (at, marker)) = free_text().await;
    // The model closes its thinking, writes on and then opens a call.
    let app = served_thinking(Some(&marker.to_string()), Some(&close.to_string()));
    let auto = json!({"tools": tools(), "parallel_tool_calls": false});
    let got = reply(&app, auto).await;
    assert_eq!(got.reasoning, text[..closed], "{got:?}");
    assert_eq!(got.content, text[closed + 1..at].trim_end(), "{got:?}");
    assert_eq!((got.calls.len(), got.finish.as_str()), (1, "tool_calls"), "{got:?}");
    check(&got.calls);

    // A reply that must call: its answer is a call from the first token
    // after the closing tag, whatever came of the thinking under its mask.
    let must = json!({"tools": tools(), "tool_choice": "required", "parallel_tool_calls": false});
    let got = reply(&app, must).await;
    assert_eq!((got.content.as_str(), got.finish.as_str()), ("", "tool_calls"), "{got:?}");
    assert_eq!(got.calls.len(), 1, "{got:?}");
    check(&got.calls);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_that_must_call_does_not_end_in_its_thinking() {
    // A character of the model's text to be the end of its replies, with
    // the character after it in the alphabet for the marker of a call: the
    // host's sampler writes that one where the end is not allowed.
    let (text, ..) = free_text().await;
    let graphic = |c: char| c.is_ascii_graphic() && c != '<' && c != '~';
    let (at, end) = text
        .char_indices()
        .skip(3)
        .find(|&(at, c)| {
            let next = char::from(c as u8 + 1);
            let new = !text[..at].contains(c) && !text[..at].contains(next);
            c.is_ascii() && graphic(c) && graphic(next) && new
        })
        .unwrap_or_else(|| panic!("no character to be the end in {text:?}"));
    let marker = char::from(end as u8 + 1).to_string();
    let app = served_ending(Some(&marker), Some("</think>"), end as u8);

    // Left alone the model ends its reply there, in its thinking.
    let auto = json!({"tools": tools(), "parallel_tool_calls": false});
    let got = reply(&app, auto).await;
    assert_eq!((got.finish.as_str(), got.calls.len()), ("stop", 0), "{got:?}");
    assert_eq!((got.reasoning.as_str(), got.content.as_str()), (&text[..at], ""), "{got:?}");

    // A reply that must call cannot end there: the end is not allowed while
    // it thinks, so it goes on, opens a call and ends when the call is whole.
    let named = json!({"type": "function", "function": {"name": "set_volume"}});
    for choice in [json!("required"), named] {
        let must = json!({"tools": tools(), "tool_choice": choice, "parallel_tool_calls": false});
        let got = reply(&app, must).await;
        assert_eq!((got.finish.as_str(), got.calls.len()), ("tool_calls", 1), "{got:?}");
        assert_eq!((got.reasoning.as_str(), got.content.as_str()), (text[..at].trim_end(), ""));
        check(&got.calls);
    }

    // With a marker the model never writes, it thinks on to the limit of
    // tokens: the reply has no call, and has not ended by itself.
    let app = served_ending(Some("<tool_call>"), Some("</think>"), end as u8);
    let must = json!({"tools": tools(), "tool_choice": "required", "max_tokens": at + 20});
    let got = reply(&app, must).await;
    assert_eq!((got.finish.as_str(), got.calls.len()), ("length", 0), "{got:?}");
    assert!(got.reasoning.starts_with(&text[..at]) && got.content.is_empty(), "{got:?}");
    assert!(got.reasoning.chars().count() > text[..at].chars().count(), "{got:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_runtime_counts_the_prompt_a_run_of_the_same_request_reports() {
    let app = served(Some("@"));
    let body = json!({
        "model": "synthetic",
        "system": "be brief",
        "messages": [{"role": "user", "content": "turn it down"}],
        "max_tokens": 20,
    })
    .to_string();
    let (status, counted) = json(post(&app, "/v1/messages/count_tokens", &body).await).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    let counted = counted["input_tokens"].as_u64().unwrap();
    assert!(counted > 0);

    // The prompt of a run is what was computed plus what the prefix cache gave.
    let (status, answer) = json(post(&app, "/v1/messages", &body).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    let usage = &answer["usage"];
    let ran = usage["input_tokens"].as_u64().unwrap()
        + usage["cache_read_input_tokens"].as_u64().unwrap();
    assert_eq!(counted, ran, "{answer}");

    // A model the runtime does not hold is not counted for.
    let other = body.replace("synthetic", "another");
    let (status, _) = json(post(&app, "/v1/messages/count_tokens", &other).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_counts_of_the_openai_routes_are_the_prompts_their_requests_report() {
    let app = served(Some("@"));

    // A Responses request: the count is the input of its run, what the
    // prefix cache gave included.
    let request = json!({
        "model": "synthetic",
        "instructions": "be brief",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "turn it down"}]},
            {"role": "assistant", "content": "to what"},
            {"role": "user", "content": "to nine"},
        ],
        "tools": [{
            "type": "function",
            "name": "set_volume",
            "parameters": {"type": "object", "properties": {"level": {"type": "integer"}}},
        }],
        "tool_choice": "none",
        "max_output_tokens": 20,
    })
    .to_string();
    let (status, counted) = json(post(&app, "/v1/responses/input_tokens", &request).await).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    assert_eq!(counted["object"], "response.input_tokens");
    let counted = counted["input_tokens"].as_u64().unwrap();
    assert!(counted > 0);
    for _ in 0..2 {
        let (status, response) = json(post(&app, "/v1/responses", &request).await).await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(response["usage"]["input_tokens"], counted, "{response}");
    }

    // A chat request.
    let body = body(json!({"tools": tools(), "max_tokens": 20})).to_string();
    let path = "/v1/chat/completions/input_tokens";
    let (status, counted) = json(post(&app, path, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{counted}");
    assert_eq!(counted["object"], "response.input_tokens");
    let (status, answer) = json(post(&app, CHAT, &body).await).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(counted["input_tokens"], answer["usage"]["prompt_tokens"], "{answer}");

    // A model the runtime does not hold is not counted for.
    let other = json!({"model": "another", "messages": [{"role": "user", "content": "hi"}]});
    let (status, _) = json(post(&app, path, &other.to_string()).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let other = json!({"model": "another", "input": "hi"}).to_string();
    let (status, _) = json(post(&app, "/v1/responses/input_tokens", &other).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
