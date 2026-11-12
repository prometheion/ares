//! The splitter of calls on written-out replies.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::*;

fn format() -> ToolFormat {
    ToolFormat {
        call_open: "<tool_call>\n<function=".into(),
        name_close: ">\n".into(),
        param_open: "<parameter=".into(),
        key_close: ">\n".into(),
        param_close: "\n</parameter>\n".into(),
        call_close: "</function>\n</tool_call>".into(),
        before_calls: "\n\n".into(),
        between_calls: "\n".into(),
    }
}

fn splitter() -> CallSplitter {
    let weather = json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "where": {"type": "object"},
            "note": {"type": ["string", "null", "number"]},
        },
    });
    let tools: HashMap<String, Option<Value>> =
        [("get_weather".to_owned(), Some(weather)), ("now".to_owned(), None)].into();
    CallSplitter::new(format(), tools, "call_".into())
}

/// What a reply comes to: its content, and each call as its id, its name and
/// its arguments.
#[derive(Debug, Default, PartialEq, Eq)]
struct Sorted {
    content: String,
    calls: Vec<(String, String, String)>,
}

impl Sorted {
    fn take(&mut self, parts: Vec<Part>) {
        for part in parts {
            match part {
                Part::Content(text) => {
                    assert!(!text.is_empty(), "an empty part");
                    self.content.push_str(&text);
                }
                Part::Call { index, id, name } => {
                    assert_eq!(index, self.calls.len(), "calls begin in order");
                    self.calls.push((id, name, String::new()));
                }
                Part::Arguments { index, json } => {
                    assert_eq!(index + 1, self.calls.len(), "arguments of the call that is open");
                    assert!(!json.is_empty(), "an empty fragment");
                    self.calls[index].2.push_str(&json);
                }
            }
        }
    }
}

/// The reply sorted when it arrives in the pieces `cuts` make of it.
fn sorted_in(text: &str, cuts: &[usize]) -> Sorted {
    let mut splitter = splitter();
    let mut sorted = Sorted::default();
    let mut from = 0;
    for &cut in cuts.iter().chain([&text.len()]) {
        sorted.take(splitter.push(&text[from..cut]));
        from = cut;
    }
    sorted.take(splitter.finish());
    sorted
}

fn sorted(text: &str) -> Sorted {
    sorted_in(text, &[])
}

fn call(name: &str, arguments: &str) -> (String, String, String) {
    (String::new(), name.to_owned(), arguments.to_owned())
}

fn calls(sorted: &Sorted) -> Vec<(String, String, String)> {
    sorted.calls.iter().map(|(_, name, arguments)| call(name, arguments)).collect()
}

const TWO: &str = "I will look.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nKöln \"am\" Rhein\n\\o/\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=where>\n{\"lat\": 50.9,\n \"lon\": 6.9}\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n</tool_call>";

#[test]
fn a_reply_with_calls_is_content_and_calls_with_json_arguments() {
    let got = sorted(TWO);
    assert_eq!(got.content, "I will look.");
    assert_eq!(
        calls(&got),
        [
            call(
                "get_weather",
                "{\"city\":\"Köln \\\"am\\\" Rhein\\n\\\\o/\",\"days\":3,\"where\":{\"lat\": 50.9,\n \"lon\": 6.9}}"
            ),
            call("now", "{}"),
        ]
    );
    // The arguments are JSON objects with the values the model wrote.
    let first: Value = serde_json::from_str(&got.calls[0].2).unwrap();
    assert_eq!(
        first,
        json!({"city": "Köln \"am\" Rhein\n\\o/", "days": 3, "where": {"lat": 50.9, "lon": 6.9}})
    );
    assert_eq!(got.calls[0].0, "call_0");
    assert_eq!(got.calls[1].0, "call_1");
}

#[test]
fn the_pieces_of_a_stream_give_the_same_however_the_text_is_cut() {
    let whole = sorted(TWO);
    let boundaries: Vec<usize> = (1..TWO.len()).filter(|at| TWO.is_char_boundary(*at)).collect();
    // Two pieces, cut at every position: every marker is split everywhere.
    for &cut in &boundaries {
        assert_eq!(sorted_in(TWO, &[cut]), whole, "cut at {cut}");
    }
    // One character at a time, and in pieces of three and of seven.
    assert_eq!(sorted_in(TWO, &boundaries), whole);
    for step in [3, 7] {
        let cuts: Vec<usize> = boundaries.iter().copied().step_by(step).collect();
        assert_eq!(sorted_in(TWO, &cuts), whole, "pieces of {step}");
    }
}

#[test]
fn no_markup_reaches_the_content_of_a_stream() {
    let boundaries: Vec<usize> = (1..TWO.len()).filter(|at| TWO.is_char_boundary(*at)).collect();
    let mut splitter = splitter();
    let mut from = 0;
    for &cut in boundaries.iter().chain([&TWO.len()]) {
        for part in splitter.push(&TWO[from..cut]) {
            if let Part::Content(text) = part {
                assert!(!text.contains('<') && !text.contains('\n'), "{text:?}");
            }
        }
        from = cut;
    }
}

#[test]
fn text_without_a_call_is_content_as_it_is() {
    for text in ["Sunny.\n", "a < b and <tool> is no tag\n\n", "", "  \n", "ends in <tool_cal"] {
        let got = sorted(text);
        assert_eq!((got.content.as_str(), got.calls.len()), (text, 0));
        let cuts: Vec<usize> = (1..text.len()).collect();
        assert_eq!(sorted_in(text, &cuts), got, "{text:?}");
    }
}

#[test]
fn a_value_is_a_string_or_json_as_the_schema_of_its_argument_says() {
    let one = |key: &str, value: &str| {
        let text = format!(
            "<tool_call>\n<function=get_weather>\n<parameter={key}>\n{value}\n</parameter>\n</function>\n</tool_call>"
        );
        let got = sorted(&text);
        assert_eq!(got.content, "");
        serde_json::from_str::<Value>(&got.calls[0].2).unwrap()[key].clone()
    };
    // A string argument is a string whatever it looks like.
    assert_eq!(one("city", "12"), json!("12"));
    assert_eq!(one("city", ""), json!(""));
    assert_eq!(one("city", "{\"a\": 1}"), json!("{\"a\": 1}"));
    // An argument that is no string is the JSON the model wrote.
    assert_eq!(one("days", "12"), json!(12));
    // One that may be either: JSON of a kind the schema allows, else a string.
    assert_eq!(one("note", "12"), json!(12));
    assert_eq!(one("note", "null"), json!(null));
    assert_eq!(one("note", "true"), json!("true"));
    assert_eq!(one("note", "twelve"), json!("twelve"));
    assert_eq!(one("note", "\"quoted\""), json!("\"quoted\""));
    // An argument the schema does not describe is told the same way.
    assert_eq!(one("extra", "[1, 2]"), json!([1, 2]));
    assert_eq!(one("extra", "some text"), json!("some text"));
}

#[test]
fn a_call_the_end_cuts_stays_as_far_as_it_came() {
    let cut_at = |marker: &str| {
        let at = TWO.find(marker).unwrap() + marker.len();
        sorted(&TWO[..at])
    };
    // Inside the opening or the name: no call.
    for marker in ["<tool_call>", "<tool_call>\n<func", "<function=get_wea"] {
        let got = cut_at(marker);
        assert_eq!((got.content.as_str(), got.calls.len()), ("I will look.", 0), "{marker}");
    }
    // After the name: the call, with arguments that do not close.
    assert_eq!(calls(&cut_at("<function=get_weather>\n")), [call("get_weather", "{")]);
    assert_eq!(calls(&cut_at("<parameter=ci")), [call("get_weather", "{")]);
    assert_eq!(calls(&cut_at("<parameter=city>\n")), [call("get_weather", "{\"city\":")]);
    assert_eq!(calls(&cut_at("Köln")), [call("get_weather", "{\"city\":\"Köln")]);
    assert_eq!(
        calls(&cut_at("<parameter=days>\n3\n</para")),
        [call("get_weather", "{\"city\":\"Köln \\\"am\\\" Rhein\\n\\\\o/\",\"days\":3")]
    );
    // Inside the closing tag of the call: the arguments are all there, and
    // the object is not closed.
    let got = cut_at("</function>\n</tool_c");
    assert!(got.calls[0].2.ends_with("\"lon\": 6.9}"), "{:?}", got.calls);
    assert_eq!(got.content, "I will look.");
    // A cut reply gives the same in a stream.
    let text = &TWO[..TWO.find("Rhein").unwrap()];
    let cuts: Vec<usize> = (1..text.len()).filter(|at| text.is_char_boundary(*at)).collect();
    assert_eq!(sorted_in(text, &cuts), sorted(text));
}

#[test]
fn what_is_not_a_call_in_the_format_is_content() {
    // The trigger and then something else.
    let got = sorted("See <tool_call> for the tag.");
    assert_eq!((got.content.as_str(), got.calls.len()), ("See <tool_call> for the tag.", 0));
    // A call that goes on with what is neither an argument nor its end.
    let got = sorted("<tool_call>\n<function=now>\nand then text");
    assert_eq!(calls(&got), [call("now", "{}")]);
    assert_eq!(got.content, "and then text");
    // Text after a call.
    let got = sorted("<tool_call>\n<function=now>\n</function>\n</tool_call>\nDone.\n");
    assert_eq!(calls(&got), [call("now", "{}")]);
    assert_eq!(got.content, "\nDone.");
}

#[test]
fn a_raw_reply_has_its_calls_at_the_first_marker_after_the_reasoning_or_inside_it() {
    // The reasoning of a raw reply is content, tags and all. The calls
    // start where the rule binds: at the first marker.
    let now = "<tool_call>\n<function=now>\n</function>\n</tool_call>";
    for (text, content) in [
        (
            format!("<think>\nI look.\n</think>\n\nSure.\n\n{now}"),
            "<think>\nI look.\n</think>\n\nSure.",
        ),
        (format!("<think>\nI look.\n</think>\n\n{now}"), "<think>\nI look.\n</think>"),
        // The model did not close its thinking: nothing closes it here.
        (format!("<think>\nI look.\n\n{now}\n"), "<think>\nI look."),
        (format!("<think>\n{now}"), "<think>"),
    ] {
        let got = sorted(&text);
        assert_eq!(got.content, content);
        assert_eq!(calls(&got), [call("now", "{}")]);
        let cuts: Vec<usize> = (1..text.len()).collect();
        assert_eq!(sorted_in(&text, &cuts), got);
    }
}
