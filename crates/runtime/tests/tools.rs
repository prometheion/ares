//! Replies that call tools, through the step loop on the synthetic model with
//! a host executor: the grammar of the calls behind the sampler's mask, from
//! the first token or from a trigger on, with drafts and without.
//!
//! The synthetic model has one token for each byte, and `~` ends a reply. It
//! has no line break among its printable bytes, so the format of the calls
//! here has blanks where the format of Qwen3.8's template has line breaks.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{Engine, FinishReason, Shape, Start, Structure, ToolFormat};
use ares_grammar::calls::{CallRule, Callable};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{RecordView, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use common::*;
use serde_json::json;

const END: u8 = b'~';

fn model() -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 1024;
    config.eos = Some(u32::from(END));
    // Text that loops, so that the lookup in earlier text has drafts to offer,
    // next to a draft block of the model's own.
    config.context.window = 1;
    config.context.epoch = 40;
    config.context.buckets = 512;
    config.own_drafts = 7;
    config.own_accuracy = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
    Arc::new(SyntheticModel::new(config).unwrap())
}

fn runtime(
    model: &Arc<SyntheticModel>,
    speculation: bool,
    observer: impl FnMut(&RecordView<'_>) + Send + 'static,
) -> Runtime {
    let config = RuntimeConfig {
        pages: 4096,
        cells_per_page: 16,
        max_slots: 64,
        window_rows: if speculation { 8 } else { 1 },
        warm_up: false,
        ..RuntimeConfig::default()
    };
    let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
        .with_row_cost(Duration::from_micros(3000), Duration::from_micros(30))
        .with_observer(observer);
    Runtime::start(model.clone(), Box::new(host), config).unwrap()
}

/// The format of Qwen3.8's template with blanks for its line breaks, and
/// with `trigger` where it has `<tool_call>`.
fn format(trigger: &str) -> ToolFormat {
    ToolFormat {
        call_open: format!("{trigger} <function="),
        name_close: "> ".into(),
        param_open: "<parameter=".into(),
        key_close: "> ".into(),
        param_close: " </parameter> ".into(),
        call_close: "</function> </tool_call>".into(),
        before_calls: "  ".into(),
        between_calls: " ".into(),
    }
}

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
    let model = model();
    let format = format("<tool_call>");
    for parallel in [false, true] {
        let mut texts = Vec::new();
        for speculation in [false, true] {
            let rows = Arc::new(Mutex::new(Rows::default()));
            let rt = runtime(&model, speculation, watch(&rows));
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
    let model = model();
    // The model writes an '8' at its third token: that is the trigger here.
    let format = format("8");
    let structure = Structure {
        shape: calls(&format, true, true),
        start: Start::Reply,
        trigger: Some(format.trigger().to_owned()),
    };
    let mut texts = Vec::new();
    for speculation in [false, true] {
        let rows = Arc::new(Mutex::new(Rows::default()));
        let rt = runtime(&model, speculation, watch(&rows));
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
