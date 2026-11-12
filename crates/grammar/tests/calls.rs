//! The grammar of a reply that calls tools, on the toy vocabulary: what is a
//! sentence of it and what is not, and the guide that waits for the trigger.

mod common;

use ares_api::{Shape, Start, Structure, ToolFormat};
use ares_grammar::Source;
use ares_grammar::calls::{CallRule, Callable};
use common::*;
use serde_json::{Value, json};

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

fn weather() -> Value {
    json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "unit": {"enum": ["celsius", "fahrenheit"]},
            "where": {"$ref": "#/$defs/place"},
            "note": {"type": ["string", "null"]},
        },
        "required": ["city", "days"],
        "$defs": {"place": {"type": "object", "properties": {"lat": {"type": "number"}}, "required": ["lat"], "additionalProperties": false}},
    })
}

struct Rule {
    parallel: bool,
    triggered: bool,
}

fn lark(rule: &Rule) -> String {
    let (weather, format) = (weather(), format());
    let tools = [
        Callable { name: "get_weather", parameters: Some(&weather) },
        Callable { name: "get", parameters: None },
    ];
    CallRule { format: &format, tools: &tools, parallel: rule.parallel, triggered: rule.triggered }
        .lark()
        .expect("a grammar")
}

/// How much of `text` the grammar takes, in bytes, and whether a reply may end
/// after it.
fn taken(toy: &Toy, lark: &str, text: &str) -> (usize, bool) {
    let mut grammar = toy.grammar(Source::Lark(lark));
    let mut cursor = grammar.start();
    let tokens = toy.tokens(text);
    let n = cursor.goto(&tokens);
    (toy.text(&tokens[..n]).len(), n == tokens.len() && cursor.accepting())
}

fn is_sentence(toy: &Toy, lark: &str, text: &str) -> bool {
    taken(toy, lark, text) == (text.len(), true)
}

fn call(name: &str, arguments: &[(&str, &str)]) -> String {
    let mut text = format!("<tool_call>\n<function={name}>\n");
    for (key, value) in arguments {
        text.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
    }
    text + "</function>\n</tool_call>"
}

const WHOLE: Rule = Rule { parallel: true, triggered: false };

#[test]
fn a_call_in_the_format_is_a_sentence() {
    let toy = toy();
    let lark = lark(&WHOLE);
    let plain = call("get_weather", &[("city", "Köln"), ("days", "3")]);
    assert!(is_sentence(&toy, &lark, &plain), "{lark}");
    // The white space the template writes before the first call.
    assert!(is_sentence(&toy, &lark, &format!("\n\n{plain}")));
    // A tool without arguments, and one whose name starts another's.
    assert!(is_sentence(&toy, &lark, &call("get", &[])));
    // A reply does not end inside a call.
    for cut in [10, 30, plain.len() - 1] {
        let (n, ends) = taken(&toy, &lark, &plain[..cut]);
        assert_eq!((n, ends), (cut, false), "cut at {cut}");
    }
}

#[test]
fn the_name_and_the_arguments_are_the_tools() {
    let toy = toy();
    let lark = lark(&WHOLE);
    let sentence =
        |name: &str, arguments: &[(&str, &str)]| is_sentence(&toy, &lark, &call(name, arguments));
    // The required arguments in either order, then the others in any order.
    assert!(sentence("get_weather", &[("days", "3"), ("city", "Köln")]));
    assert!(sentence(
        "get_weather",
        &[("city", "Köln"), ("days", "3"), ("note", "hi"), ("unit", "celsius")]
    ));
    // A required argument is missing, comes twice, or follows an optional one.
    assert!(!sentence("get_weather", &[("city", "Köln")]));
    assert!(!sentence("get_weather", &[("city", "a"), ("city", "b"), ("days", "3")]));
    assert!(!sentence("get_weather", &[("unit", "celsius"), ("city", "a"), ("days", "3")]));
    // A tool or an argument nobody offered.
    assert!(!sentence("get_time", &[]));
    assert!(!sentence("get_weather", &[("city", "a"), ("days", "3"), ("zip", "1")]));
    assert!(!sentence("get", &[("city", "a")]));
}

#[test]
fn a_value_follows_the_schema_of_its_argument() {
    let toy = toy();
    let lark = lark(&WHOLE);
    let with = |key: &str, value: &str| {
        is_sentence(
            &toy,
            &lark,
            &call("get_weather", &[("city", "Köln"), ("days", "3"), (key, value)]),
        )
    };
    // A string stands as it is: any text, tags and line breaks included.
    assert!(with("note", "two\nlines, a <b>tag</b> and \"quotes\""));
    assert!(with("note", ""));
    // A number is a JSON number of the schema's kind.
    let days = |value: &str| {
        is_sentence(&toy, &lark, &call("get_weather", &[("city", "Köln"), ("days", value)]))
    };
    assert!(days("14") && days("-2"));
    assert!(!days("three") && !days("1.5") && !days("\"3\"") && !days(""));
    // One of the strings of an enum, without quotes.
    assert!(with("unit", "fahrenheit"));
    assert!(!with("unit", "kelvin") && !with("unit", "\"celsius\""));
    // An object, through the reference into the tool's definitions.
    assert!(with("where", r#"{"lat": 50.9}"#));
    assert!(!with("where", r#"{"lat": "north"}"#) && !with("where", r#"{"lon": 1}"#));
}

#[test]
fn a_string_ends_at_the_first_closing_tag() {
    let toy = toy();
    let lark = lark(&WHOLE);
    // What follows the first closing tag is the next argument or the end of
    // the call, never more of the string.
    let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\na\n</parameter>\nb";
    let (n, _) = taken(&toy, &lark, text);
    assert_eq!(n, text.len() - 1);
}

#[test]
fn calls_follow_each_other_only_when_several_are_allowed() {
    let toy = toy();
    let one = call("get", &[]);
    let two = format!("{one}\n{one}");
    let three = format!("{two}\n{one}\n");
    let many = lark(&WHOLE);
    assert!(is_sentence(&toy, &many, &two) && is_sentence(&toy, &many, &three));
    let single = lark(&Rule { parallel: false, triggered: false });
    assert!(is_sentence(&toy, &single, &one) && is_sentence(&toy, &single, &format!("{one}\n")));
    assert_eq!(taken(&toy, &single, &two), (one.len() + 1, false));
    // Nothing but calls: text after a call is refused.
    assert_eq!(taken(&toy, &many, &format!("{one}\nDone.")), (one.len() + 1, false));
    assert_eq!(taken(&toy, &many, &format!("Sure. {one}")).0, 0);
}

#[test]
fn a_triggered_grammar_starts_after_the_trigger() {
    let toy = toy();
    let lark = lark(&Rule { parallel: true, triggered: true });
    let one = call("get", &[]);
    let rest = one.strip_prefix("<tool_call>").unwrap();
    assert!(is_sentence(&toy, &lark, rest));
    assert!(is_sentence(&toy, &lark, &format!("{rest}\n{one}")));
    assert_eq!(taken(&toy, &lark, &one).0, 0);
}

#[test]
fn many_required_arguments_come_in_any_order_up_to_the_limit() {
    let toy = toy();
    let format = format();
    let tool = |count: usize| {
        let keys: Vec<String> = (0..count).map(|i| format!("k{i}")).collect();
        let properties: serde_json::Map<String, Value> =
            keys.iter().map(|k| (k.clone(), json!({"type": "integer"}))).collect();
        json!({"type": "object", "properties": properties, "required": keys})
    };
    let sentence = |count: usize, order: &[usize]| {
        let schema = tool(count);
        let tools = [Callable { name: "f", parameters: Some(&schema) }];
        let lark = CallRule { format: &format, tools: &tools, parallel: false, triggered: false }
            .lark()
            .unwrap();
        let keys: Vec<String> = order.iter().map(|i| format!("k{i}")).collect();
        let arguments: Vec<(&str, &str)> = keys.iter().map(|k| (k.as_str(), "1")).collect();
        is_sentence(&toy, &lark, &call("f", &arguments))
    };
    assert!(sentence(6, &[5, 3, 0, 1, 4, 2]) && sentence(6, &[0, 1, 2, 3, 4, 5]));
    assert!(!sentence(6, &[5, 3, 0, 1, 4]));
    // More than the limit: the order of the schema.
    assert!(sentence(7, &[0, 1, 2, 3, 4, 5, 6]));
    assert!(!sentence(7, &[1, 0, 2, 3, 4, 5, 6]));
}

#[test]
fn a_tool_that_the_format_cannot_write_is_refused() {
    let format = format();
    let rule = |tools: &[Callable<'_>]| {
        CallRule { format: &format, tools, parallel: true, triggered: true }.lark()
    };
    assert!(rule(&[]).is_err());
    assert!(rule(&[Callable { name: "a>\nb", parameters: None }]).is_err());
    let key = json!({"type": "object", "properties": {"k>\n": {}}});
    assert!(rule(&[Callable { name: "f", parameters: Some(&key) }]).is_err());
    let value = json!({"type": "object", "properties": {"k": {"enum": ["a\n</parameter>\nb"]}}});
    assert!(rule(&[Callable { name: "f", parameters: Some(&value) }]).is_err());
}

fn guide(toy: &Toy, triggered: bool, start: Start) -> ares_grammar::Guide {
    let structure = Structure {
        shape: Shape::Lark(lark(&Rule { parallel: true, triggered })),
        start,
        trigger: triggered.then(|| "<tool_call>".to_owned()),
    };
    toy.compiler.guide(&structure).expect("a guide")
}

#[test]
fn a_reply_is_free_until_the_trigger_and_bound_after_it() {
    let toy = toy();
    let mut g = guide(&toy, true, Start::Reply);
    let text = toy.tokens("Let me look. <tool_cal");
    assert!(g.mask(&[], &[]).unwrap().is_none(), "free from the first token");
    assert!(g.mask(&text, &[]).unwrap().is_none(), "free inside what may be the trigger");
    // Drafts that complete the trigger: the row after them is bound.
    let rest = toy.tokens("l>");
    let mask = g.mask(&text, &rest).unwrap().expect("bound after the trigger");
    assert_eq!(allowed_tokens(&mask), vec![u32::from(b'\n')]);
    // Every draft before the grammar is admitted, and the grammar cuts the
    // first one it refuses.
    let drafts = toy.tokens("l>\n<function=put");
    assert_eq!(g.admitted(&text, &drafts).unwrap(), drafts.len() - 3);
    let mut reply = text.clone();
    reply.extend(toy.tokens("l>\n<function=get>\n</function>\n</tool_call>"));
    let mask = g.mask(&reply, &[]).unwrap().expect("bound to the end");
    let allowed = allowed_tokens(&mask);
    assert!(allowed.contains(&toy.eos) && allowed.contains(&u32::from(b'\n')), "{allowed:?}");
    assert!(!allowed.contains(&u32::from(b'D')), "no text after a call");
    // A reply that never writes the trigger stays free.
    let plain = toy.tokens("It is sunny. <tool> is not the tag.");
    assert!(g.mask(&plain, &[]).unwrap().is_none());
}

#[test]
fn the_trigger_counts_only_after_the_reasoning() {
    let toy = toy();
    let after = Start::AfterReasoning { open: None, close: "</think>".into() };
    let mut g = guide(&toy, true, after);
    let thinking = toy.tokens("I could write <tool_call> now.");
    assert!(g.mask(&thinking, &[]).unwrap().is_none());
    let mut reply = thinking.clone();
    reply.extend(toy.tokens("</think>\n\nSure."));
    assert!(g.mask(&reply, &[]).unwrap().is_none(), "the answer is free too");
    reply.extend(toy.tokens("\n\n<tool_call>"));
    let mask = g.mask(&reply, &[]).unwrap().expect("bound after the trigger");
    assert_eq!(allowed_tokens(&mask), vec![u32::from(b'\n')]);

    // A reply that may open the block itself is free from its first token, and
    // what looks like the opening tag may be the answer.
    let lead = Start::AfterReasoning { open: Some("<think>".into()), close: "</think>".into() };
    let mut g = guide(&toy, true, lead);
    assert!(g.mask(&[], &[]).unwrap().is_none());
    let answer = toy.tokens("<thi is not a tag. <tool_call>");
    assert!(g.mask(&answer, &[]).unwrap().is_some(), "the trigger in the answer binds");
    let thought = toy.tokens("<think><tool_call>");
    assert!(g.mask(&thought, &[]).unwrap().is_none(), "the trigger in the reasoning does not");
}

#[test]
fn a_reply_that_must_call_is_bound_from_the_answer_on() {
    let toy = toy();
    let after = Start::AfterReasoning { open: None, close: "</think>".into() };
    let mut g = guide(&toy, false, after);
    let reply = toy.tokens("thinking</think>");
    let mask = g.mask(&reply, &[]).unwrap().expect("bound after the reasoning");
    // The template's blank line, or the call at once; no text and no end.
    assert_eq!(
        allowed_tokens(&mask),
        vec![u32::from(b'\n'), u32::from(b' '), u32::from(b'<')]
            .into_iter()
            .chain(toy.vocab.find(b"\n\n"))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    );
}
