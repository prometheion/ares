//! Tool calls from the request to the answer on the host: the server over the
//! real runtime, with the synthetic model on a host executor. The rule the
//! server writes for the tools is followed by the grammar engine behind the
//! sampler's mask, and the server reads the calls out of what the model
//! generated under it.
//!
//! The synthetic model has one token for each byte, no line break among its
//! printable bytes and no chat template, so it declares a format for calls
//! with blanks where the format of Qwen3.8's template has line breaks, and
//! with a trigger a test picks from the text the model writes anyway.

mod common;

use std::sync::Arc;
use std::time::Duration;

use ares_api::{ChatFormat, Engine, ToolFormat};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use axum::Router;
use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const CHAT: &str = "/v1/chat/completions";

/// The server over a runtime whose model writes calls after `trigger`.
fn served(trigger: Option<&str>) -> Router {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 2048;
    config.eos = Some(u32::from(b'~'));
    // Text that loops, so that the lookup in earlier text has drafts to offer,
    // next to a draft block of the model's own.
    config.context.window = 1;
    config.context.epoch = 40;
    config.context.buckets = 512;
    config.own_drafts = 7;
    config.own_accuracy = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
    config.chat = trigger.map(|trigger| ChatFormat {
        thinking: None,
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
    let mut reply = Reply { content: String::new(), calls: Vec::new(), finish: String::new() };
    let mut arguments: Vec<String> = Vec::new();
    for chunk in &stream.chunks {
        let choice = &chunk["choices"][0];
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
