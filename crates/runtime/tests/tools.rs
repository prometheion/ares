//! Replies that call tools, through the step loop on the synthetic model with
//! a host executor: the grammar of the calls behind the sampler's mask, from
//! the first token or from a trigger on, and from a marker inside the
//! reasoning on, with drafts and without.
//!
//! The synthetic model has one token for each byte, and `~` ends a reply. It
//! has no line break among its printable bytes, so the format of the calls
//! here has blanks where the format of Qwen3.8's template has line breaks.

mod common;

use std::sync::{Arc, Mutex};

use ares_api::{Engine, FinishReason, Interrupt, Shape, Start, Structure, ToolFormat};
use ares_grammar::calls::{CallRule, Callable};
use ares_runtime::RecordView;
use common::*;
use serde_json::json;

const END: u8 = b'~';

/// The grammar of calls of two tools whose arguments the model cannot write
/// without end: numbers and one string of a few.
fn calls(format: &ToolFormat, parallel: bool, triggered: bool) -> Shape {
    let set = json!({
        "type": "object",
        "properties": {
            "level": {"type": "integer", "minimum": 0, "maximum": 99},
            "mode": {"enum": ["quiet", "loud", "off"]},
            "on": {"type": "boolean"},
        },
        "required": ["level", "mode"],
    });
    let tools = [
        Callable { name: "set_volume", parameters: Some(&set) },
        Callable { name: "mute", parameters: None },
    ];
    Shape::Lark(CallRule { format, tools: &tools, parallel, triggered }.lark().unwrap())
}

/// The calls in `text`, each as its name and its arguments; panics when the
/// text is not calls in the format and white space.
fn parse(text: &str, trigger: &str) -> Vec<(String, Vec<(String, String)>)> {
    let mut calls = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        let body = rest.strip_prefix(&format!("{trigger} <function=")).expect(text);
        let (name, mut body) = body.split_once("> ").expect(text);
        let mut arguments = Vec::new();
        while let Some(argument) = body.strip_prefix("<parameter=") {
            let (key, value) = argument.split_once("> ").expect(text);
            let (value, after) = value.split_once(" </parameter> ").expect(text);
            arguments.push((key.to_owned(), value.to_owned()));
            body = after;
        }
        rest = body.strip_prefix("</function> </tool_call>").expect(text).trim_start();
        calls.push((name.to_owned(), arguments));
    }
    calls
}

fn check(calls: &[(String, Vec<(String, String)>)]) {
    for (name, arguments) in calls {
        let keys: Vec<&str> = arguments.iter().map(|(key, _)| key.as_str()).collect();
        match name.as_str() {
            "mute" => assert!(arguments.is_empty(), "{arguments:?}"),
            "set_volume" => {
                let mut required: Vec<&str> = keys.iter().take(2).copied().collect();
                required.sort_unstable();
                assert_eq!(required, ["level", "mode"], "{keys:?}");
                assert!(keys[2..].iter().all(|key| *key == "on"), "{keys:?}");
            }
            other => panic!("a tool nobody offered: {other}"),
        }
        for (key, value) in arguments {
            match key.as_str() {
                "level" => assert!((0..=99).contains(&value.parse::<i64>().unwrap()), "{value}"),
                "mode" => assert!(["quiet", "loud", "off"].contains(&value.as_str()), "{value}"),
                _ => assert!(["true", "false"].contains(&value.as_str()), "{value}"),
            }
        }
    }
}

/// How many rows of a run wanted a pick, and how many of them had no mask.
#[derive(Default)]
struct Rows {
    wanted: usize,
    unmasked: usize,
}

fn watch(rows: &Arc<Mutex<Rows>>) -> impl FnMut(&RecordView<'_>) + Send + 'static {
    let rows = rows.clone();
    move |record| {
        let mut rows = rows.lock().unwrap();
        for index in 0..record.picked_rows() {
            rows.wanted += 1;
            rows.unmasked += usize::from(record.sampler(index).mask().is_none());
        }
    }
}

#[test]
fn a_reply_that_must_call_is_calls_and_drafts_change_nothing() {
    let model = looping_model(1024, Some(u32::from(END)));
    let format = tool_format("<tool_call>");
    for parallel in [false, true] {
        let mut texts = Vec::new();
        for speculation in [false, true] {
            let rows = Arc::new(Mutex::new(Rows::default()));
            let rt = observed_runtime(&model, speculation, watch(&rows));
            let mut request = request("a prompt", Some(900));
            request.structure = Some(Structure::new(calls(&format, parallel, false)));
            let run = run_one(&rt, request);
            assert_eq!(run.finish, FinishReason::Stop, "{run:?}");
            let calls = parse(&run.text, "<tool_call>");
            check(&calls);
            assert!(!calls.is_empty(), "{}", run.text);
            assert!(parallel || calls.len() == 1, "{}", run.text);
            assert_eq!(rows.lock().unwrap().unmasked, 0, "every wanted row carries a mask");
            if speculation {
                let stats = rt.stats().expect("stats");
                assert!(stats.drafts_proposed > 0, "no drafts were verified: nothing was tested");
            }
            texts.push(run.text);
        }
        assert_eq!(texts[0], texts[1], "the text with drafts is the text without");
    }
}

#[test]
fn a_reply_is_free_until_the_trigger_and_calls_after_it_with_drafts_and_without() {
    let model = looping_model(1024, Some(u32::from(END)));
    // The model writes an '8' at its third token: that is the trigger here.
    let format = tool_format("8");
    let structure = Structure {
        shape: calls(&format, true, true),
        start: Start::Reply,
        trigger: Some(format.trigger().to_owned()),
        interrupt: None,
    };
    let mut texts = Vec::new();
    for speculation in [false, true] {
        let rows = Arc::new(Mutex::new(Rows::default()));
        let rt = observed_runtime(&model, speculation, watch(&rows));
        let mut request = request("a prompt", Some(900));
        request.structure = Some(structure.clone());
        let run = run_one(&rt, request);
        assert_eq!(run.finish, FinishReason::Stop, "{run:?}");
        let at = run.text.find('8').expect("the model wrote the trigger");
        assert!(at > 0, "free text comes first: {}", run.text);
        let calls = parse(&run.text[at..], "8");
        check(&calls);
        assert!(!calls.is_empty(), "{}", run.text);
        let rows = rows.lock().unwrap();
        assert!(
            rows.unmasked >= at && rows.unmasked < rows.wanted,
            "the trigger is at {at}: {} rows, {} without a mask",
            rows.wanted,
            rows.unmasked
        );
        texts.push(run.text);
    }
    assert_eq!(texts[0], texts[1], "the text with drafts is the text without");
}

#[test]
fn a_call_opened_inside_the_reasoning_is_bound_with_drafts_and_without() {
    let model = looping_model(1024, Some(u32::from(END)));
    // The model writes an '8' at its third token: that is the marker of a
    // call here. It thinks, and never writes the tag that closes its thinking.
    let format = tool_format("8");
    for must_call in [false, true] {
        let structure = Structure {
            shape: calls(&format, true, !must_call),
            start: Start::AfterReasoning { open: None, close: "</think>".into() },
            trigger: (!must_call).then(|| format.trigger().to_owned()),
            interrupt: Some(Interrupt {
                text: format.trigger().to_owned(),
                shape: calls(&format, true, true),
            }),
        };
        let mut texts = Vec::new();
        for speculation in [false, true] {
            let rows = Arc::new(Mutex::new(Rows::default()));
            let rt = observed_runtime(&model, speculation, watch(&rows));
            let mut request = request("a prompt", Some(900));
            request.structure = Some(structure.clone());
            let run = run_one(&rt, request);
            assert_eq!(run.finish, FinishReason::Stop, "{run:?}");
            let at = run.text.find('8').expect("the model wrote the marker");
            assert!(at > 0, "reasoning comes first: {}", run.text);
            let calls = parse(&run.text[at..], "8");
            check(&calls);
            assert!(!calls.is_empty(), "{}", run.text);
            let rows = rows.lock().unwrap();
            if must_call {
                // The reasoning may not end the reply: its rows carry a mask.
                assert_eq!(rows.unmasked, 0, "every wanted row carries a mask");
            } else {
                assert!(
                    rows.unmasked >= at && rows.unmasked < rows.wanted,
                    "the marker is at {at}: {} rows, {} without a mask",
                    rows.wanted,
                    rows.unmasked
                );
            }
            if speculation {
                let stats = rt.stats().expect("stats");
                assert!(stats.drafts_proposed > 0, "no drafts were verified: nothing was tested");
            }
            texts.push(run.text);
        }
        assert_eq!(texts[0], texts[1], "the text with drafts is the text without");
    }
}
