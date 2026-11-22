//! The two splitters together on written-out replies: a call the model opens
//! inside its thinking, at every place relative to the end of the reasoning
//! and however the pieces of a stream cut the text.

use std::collections::HashMap;

use ares_api::{ThinkingFormat, ToolFormat};
use serde_json::{Value, json};

use super::*;

fn thinking() -> ThinkingFormat {
    ThinkingFormat {
        start: "<think>\n".into(),
        end: "\n</think>\n\n".into(),
        prompt_on: "<think>\n".into(),
        prompt_off: "<think>\n\n</think>\n\n".into(),
        on_by_default: true,
        effort: true,
        refused_efforts: Vec::new(),
    }
}

fn calls() -> CallSplitter {
    let format = ToolFormat {
        call_open: "<tool_call>\n<function=".into(),
        name_close: ">\n".into(),
        param_open: "<parameter=".into(),
        key_close: ">\n".into(),
        param_close: "\n</parameter>\n".into(),
        call_close: "</function>\n</tool_call>".into(),
        before_calls: "\n\n".into(),
        between_calls: "\n".into(),
    };
    let weather = json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    });
    let tools: HashMap<String, Option<Value>> =
        [("get_weather".to_owned(), Some(weather)), ("now".to_owned(), None)].into();
    CallSplitter::new(format, tools, "call_".into())
}

/// The sorter of a reply to a prompt that opened the block, for a request
/// whose tools may be called.
fn calling() -> Sorter {
    let format = thinking();
    Sorter::new(Splitter::extracting(&format, &format.prompt_on), Some(calls()))
}

/// A reply as a client puts it together.
#[derive(Debug, Default, PartialEq, Eq)]
struct Reply {
    reasoning: String,
    content: String,
    /// Each call: its name and its arguments.
    calls: Vec<(String, String)>,
}

impl Reply {
    fn take(&mut self, pieces: Vec<Piece>) {
        for piece in pieces {
            match piece {
                Piece::Text(split) => {
                    assert!(!split.is_empty(), "an empty piece");
                    // The order of a reply: reasoning, content, calls.
                    assert!(
                        split.reasoning.is_empty()
                            || (self.content.is_empty() && self.calls.is_empty()),
                        "reasoning after the answer began: {split:?} after {self:?}"
                    );
                    self.reasoning.push_str(&split.reasoning);
                    self.content.push_str(&split.content);
                }
                Piece::Call { index, name, .. } => {
                    assert_eq!(index, self.calls.len(), "calls begin in order");
                    self.calls.push((name, String::new()));
                }
                Piece::Arguments { index, json } => self.calls[index].1.push_str(&json),
            }
        }
    }
}

/// The reply `sorter` makes of `text` when it arrives in the pieces `cuts`
/// make of it.
fn sorted_in(mut sorter: Sorter, text: &str, cuts: &[usize]) -> Reply {
    let mut reply = Reply::default();
    let mut from = 0;
    for &cut in cuts.iter().chain([&text.len()]) {
        reply.take(sorter.push(&text[from..cut]));
        from = cut;
    }
    reply.take(sorter.finish());
    reply
}

fn sorted(text: &str) -> Reply {
    sorted_in(calling(), text, &[])
}

fn reply(reasoning: &str, content: &str, calls: &[(&str, &str)]) -> Reply {
    Reply {
        reasoning: reasoning.to_owned(),
        content: content.to_owned(),
        calls: calls.iter().map(|(name, json)| ((*name).to_owned(), (*json).to_owned())).collect(),
    }
}

const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nKöln\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
const NOW: &str = "<tool_call>\n<function=now>\n</function>\n</tool_call>";
const WEATHER: (&str, &str) = ("get_weather", r#"{"city":"Köln","days":3}"#);

/// Replies with a call at every place relative to the end of the reasoning,
/// and what each comes to.
fn replies() -> Vec<(String, Reply)> {
    vec![
        // Inside the thinking: after white space, right after the text, as
        // the first thing of the reply, and with a second call after it.
        (format!("I look it up.\n\n{CALL}"), reply("I look it up.", "", &[WEATHER])),
        (format!("I look it up.\n{CALL}\n"), reply("I look it up.", "", &[WEATHER])),
        (format!("I look it up.{CALL}"), reply("I look it up.", "", &[WEATHER])),
        (CALL.to_owned(), reply("", "", &[WEATHER])),
        (format!("\n\n{CALL}"), reply("", "", &[WEATHER])),
        (
            format!("Two things. {CALL}\n{NOW}\n"),
            reply("Two things.", "", &[WEATHER, ("now", "{}")]),
        ),
        // After the closing tag: at once, after the template's blank line,
        // after text of the answer.
        (format!("I look.\n</think>{CALL}"), reply("I look.", "", &[WEATHER])),
        (format!("I look.\n</think>\n\n{CALL}"), reply("I look.", "", &[WEATHER])),
        (format!("I look.\n</think>\n\nSure.\n\n{CALL}"), reply("I look.", "Sure.", &[WEATHER])),
        // What only begins like the marker is reasoning.
        (
            format!("I look. <tool_cal\n</think>\n\n{NOW}"),
            reply("I look. <tool_cal", "", &[("now", "{}")]),
        ),
        (
            format!("a <b <tool <tool_call {NOW}"),
            reply("a <b <tool <tool_call", "", &[("now", "{}")]),
        ),
        // The limit of tokens cuts the call, the marker, or the reasoning.
        (
            "I look.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nKö".to_owned(),
            reply("I look.", "", &[("get_weather", r#"{"city":"Kö"#)]),
        ),
        ("I look.\n<tool_call>\n<function=get_w".to_owned(), reply("I look.", "", &[])),
        ("I look.\n<tool_call>".to_owned(), reply("I look.", "", &[])),
        ("I look.\n<tool_ca".to_owned(), reply("I look.\n<tool_ca", "", &[])),
        ("I look.\n\n".to_owned(), reply("I look.\n\n", "", &[])),
        // No call at all.
        ("I look.\n</think>\n\nIt is sunny.\n".to_owned(), reply("I look.", "It is sunny.\n", &[])),
    ]
}

#[test]
fn a_call_inside_the_thinking_is_a_call_and_the_text_before_it_the_reasoning() {
    for (text, expected) in replies() {
        assert_eq!(sorted(&text), expected, "{text:?}");
    }
}

#[test]
fn the_pieces_of_a_stream_give_the_same_however_they_cut_the_text() {
    for (text, expected) in replies() {
        let boundaries: Vec<usize> =
            (1..text.len()).filter(|at| text.is_char_boundary(*at)).collect();
        // Three pieces, cut at every pair of positions: the marker, the
        // white space before it and the closing tag are split everywhere.
        for (i, &first) in boundaries.iter().enumerate() {
            for &second in &boundaries[i..] {
                let got = sorted_in(calling(), &text, &[first, second]);
                assert_eq!(got, expected, "{text:?} cut at {first} and {second}");
            }
        }
        // One character at a time, and in pieces of three and of seven.
        for step in [1, 3, 7] {
            let cuts: Vec<usize> = boundaries.iter().copied().step_by(step).collect();
            assert_eq!(sorted_in(calling(), &text, &cuts), expected, "{text:?} in {step}s");
        }
    }
}

#[test]
fn no_markup_of_a_call_reaches_the_reasoning_or_the_content() {
    // But for the replies whose reasoning holds a part of the marker as text.
    let whole = |(text, reply): &(String, Reply)| {
        text.contains("</tool_call>") && !reply.reasoning.contains("<tool")
    };
    for (text, _) in replies().into_iter().filter(whole) {
        let boundaries: Vec<usize> =
            (1..text.len()).filter(|at| text.is_char_boundary(*at)).collect();
        let mut sorter = calling();
        let mut from = 0;
        let mut pieces = Vec::new();
        for &cut in boundaries.iter().chain([&text.len()]) {
            pieces.extend(sorter.push(&text[from..cut]));
            from = cut;
        }
        pieces.extend(sorter.finish());
        for piece in pieces {
            if let Piece::Text(split) = piece {
                for part in [&split.reasoning, &split.content] {
                    assert!(
                        !part.contains("tool_call") && !part.contains("function"),
                        "{part:?} of {text:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn text_after_a_call_inside_the_thinking_is_content_the_closing_tag_too() {
    // The rule of the request allows nothing but white space and calls after
    // a call, so an engine that holds the reply to it cannot write this. From
    // one that does not, the text is content, as after any call.
    let got = sorted(&format!("I look.\n{NOW}\n</think>\n\nDone."));
    assert_eq!(got, reply("I look.", "\n</think>\n\nDone.", &[("now", "{}")]));
    // The marker and then no call: the reasoning has ended all the same.
    let got = sorted("I could write <tool_call> here.\n</think>\n\nSure.");
    assert_eq!(got, reply("I could write", "<tool_call> here.\n</think>\n\nSure.", &[]));
}

#[test]
fn without_tools_to_call_the_marker_inside_the_thinking_is_reasoning() {
    // No tools, or `tool_choice: none`: there is no splitter of calls.
    let format = thinking();
    let text = format!("I could call.\n\n{CALL}\nBut I may not.\n</think>\n\nIt is sunny.");
    let plain = || Sorter::new(Splitter::extracting(&format, &format.prompt_on), None);
    let expected = Reply {
        reasoning: format!("I could call.\n\n{CALL}\nBut I may not."),
        content: "It is sunny.".into(),
        calls: Vec::new(),
    };
    assert_eq!(sorted_in(plain(), &text, &[]), expected);
    let cuts: Vec<usize> = (1..text.len()).filter(|at| text.is_char_boundary(*at)).collect();
    assert_eq!(sorted_in(plain(), &text, &cuts), expected);
}

#[test]
fn a_raw_reply_keeps_its_reasoning_in_the_content_and_has_the_call() {
    // `reasoning_format: none`: the end of the prompt and the thinking are
    // content, and the call inside the thinking is a call.
    let raw = || Sorter::new(Splitter::raw("<think>\n"), Some(calls()));
    let text = format!("I look it up.\n\n{CALL}");
    let expected = reply("", "<think>\nI look it up.", &[WEATHER]);
    assert_eq!(sorted_in(raw(), &text, &[]), expected);
    let cuts: Vec<usize> = (1..text.len()).filter(|at| text.is_char_boundary(*at)).collect();
    assert_eq!(sorted_in(raw(), &text, &cuts), expected);
}

#[test]
fn each_reply_of_a_request_ends_its_reasoning_at_the_marker() {
    // The sorter of another completion of the same request.
    let other = calling().with_call_ids("call_x_".into());
    let got = sorted_in(other, &format!("I look.\n{NOW}"), &[]);
    assert_eq!(got, reply("I look.", "", &[("now", "{}")]));
}
