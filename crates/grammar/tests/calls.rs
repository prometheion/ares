//! The grammar of a reply that calls tools, on the toy vocabulary: what is a
//! sentence of it and what is not, and the guide that waits for the trigger.

mod common;

use ares_api::{Interrupt, Shape, Start, Structure, ToolFormat};
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
        interrupt: None,
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

// A call that the model opens while it still thinks.

const MARKER: &str = "<tool_call>";

fn thinking() -> Start {
    Start::AfterReasoning { open: None, close: "</think>".into() }
}

/// The structure the server writes for a request with tools: with `auto` the
/// reply is free until the marker, with `required` it is calls from its
/// answer on, and either way the marker inside the reasoning ends it.
fn calling(toy: &Toy, required: bool, start: Start) -> ares_grammar::Guide {
    let after_marker = Shape::Lark(lark(&Rule { parallel: true, triggered: true }));
    let structure = Structure {
        shape: if required {
            Shape::Lark(lark(&Rule { parallel: true, triggered: false }))
        } else {
            after_marker.clone()
        },
        start,
        trigger: (!required).then(|| MARKER.to_owned()),
        interrupt: Some(Interrupt { text: MARKER.into(), shape: after_marker }),
    };
    toy.compiler.guide(&structure).expect("a guide")
}

/// What a reply may be bound to after `tokens`.
#[derive(Debug, PartialEq, Eq)]
enum Next {
    /// Nothing binds it.
    Free,
    /// Every token but the end.
    Unending,
    /// The rest of a call whose marker is written: a line break.
    InCall,
    /// A call from its marker on, after white space if any.
    CallOrSpace,
}

fn next(toy: &Toy, g: &mut ares_grammar::Guide, tokens: &[u32]) -> Next {
    let Some(mask) = g.mask(tokens, &[]).unwrap() else { return Next::Free };
    let allowed = allowed_tokens(&mask);
    if allowed == [u32::from(b'\n')] {
        Next::InCall
    } else if allowed.len() as u32 == toy.vocab.len() - 1 && !allowed.contains(&toy.eos) {
        Next::Unending
    } else {
        // White space, or what the marker starts with.
        assert!(allowed.contains(&u32::from(b'<')), "{allowed:?}");
        let starts_a_call = |token: &u32| {
            let piece = toy.vocab.piece(*token);
            MARKER.as_bytes().starts_with(piece) || piece.iter().all(|b| b" \n\t".contains(b))
        };
        assert!(allowed.iter().all(starts_a_call), "{allowed:?}");
        Next::CallOrSpace
    }
}

#[test]
fn the_marker_inside_the_reasoning_ends_it_and_the_call_is_bound() {
    let toy = toy_with(&[MARKER]);
    let one = call("get", &[]);
    for required in [false, true] {
        let free = if required { Next::Unending } else { Next::Free };
        let mut g = calling(&toy, required, thinking());
        let mut reply = toy.tokens("I look it up. <tool_cal is not it.\n\n");
        assert_eq!(next(&toy, &mut g, &[]), free);
        assert_eq!(next(&toy, &mut g, &reply), free, "thinking");
        reply.push(toy.token(MARKER));
        assert_eq!(next(&toy, &mut g, &reply), Next::InCall, "required: {required}");
        // The name is a tool's, as after the closing tag.
        let mut wrong = reply.clone();
        wrong.extend(toy.tokens("\n<function=get_t"));
        assert!(g.mask(&wrong, &[]).unwrap().is_none(), "no tool of that name");
        assert_eq!(g.admitted(&reply, &toy.tokens("\n<function=put")).unwrap(), 11);
        // After the call: white space, another call, the end. Not the tag
        // that closes the reasoning, and no text.
        reply.extend(toy.tokens(one.strip_prefix(MARKER).unwrap()));
        let mask = g.mask(&reply, &[]).unwrap().expect("bound to the end");
        let after = allowed_tokens(&mask);
        assert!(after.contains(&toy.eos) && after.contains(&u32::from(b'\n')), "{after:?}");
        assert!(after.contains(&u32::from(b'<')) && after.contains(&toy.token(MARKER)));
        assert!(!after.contains(&toy.token("</think>")), "the reasoning is over");
        assert!(!after.contains(&u32::from(b'D')), "no text after a call");
        let mut closing = reply.clone();
        closing.extend(toy.tokens("\n<"));
        let mask = g.mask(&closing, &[]).unwrap().expect("bound");
        assert_eq!(allowed_tokens(&mask), vec![u32::from(b't')], "only another call");
        // A second call follows by the same rule.
        reply.extend(toy.tokens(&format!("\n{one}")));
        assert!(mask_allows(&g.mask(&reply, &[]).unwrap().unwrap(), toy.eos));
    }
}

#[test]
fn the_marker_binds_however_the_tokens_cut_it() {
    // A token for every start and every end of the marker, alone and with
    // text around it.
    let mut pieces = vec![MARKER.to_owned()];
    for cut in 1..MARKER.len() {
        let (head, tail) = MARKER.split_at(cut);
        pieces.extend([
            head.to_owned(),
            tail.to_owned(),
            format!("so {head}"),
            format!("{tail}\n"),
        ]);
    }
    let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let toy = toy_with(&pieces);
    for required in [false, true] {
        let free = if required { Next::Unending } else { Next::Free };
        for cut in 1..MARKER.len() {
            let (head, tail) = MARKER.split_at(cut);
            // Two tokens.
            let mut g = calling(&toy, required, thinking());
            let mut reply = toy.tokens("Now ");
            reply.push(toy.token(head));
            assert_eq!(next(&toy, &mut g, &reply), free, "cut at {cut}");
            reply.push(toy.token(tail));
            assert_eq!(next(&toy, &mut g, &reply), Next::InCall, "cut at {cut}");
            // The first with text before the marker.
            let mut g = calling(&toy, required, thinking());
            let reply = [toy.token(&format!("so {head}")), toy.token(tail)];
            assert_eq!(next(&toy, &mut g, &reply[..1]), free, "cut at {cut}");
            assert_eq!(next(&toy, &mut g, &reply), Next::InCall, "cut at {cut}");
            // The second arrives as a draft.
            let mut g = calling(&toy, required, thinking());
            let mask = g.mask(&reply[..1], &reply[1..]).unwrap().expect("bound after the draft");
            assert_eq!(allowed_tokens(&mask), vec![u32::from(b'\n')], "cut at {cut}");
            // One byte a token.
            let mut g = calling(&toy, required, thinking());
            let bytes: Vec<u32> = MARKER.bytes().map(u32::from).collect();
            assert_eq!(next(&toy, &mut g, &bytes[..bytes.len() - 1]), free);
            assert_eq!(next(&toy, &mut g, &bytes), Next::InCall);
            // A token that goes on after the marker: the rule is checked
            // from the next token on, and that token's rest is not.
            let mut g = calling(&toy, required, thinking());
            let reply = [toy.token(head), toy.token(&format!("{tail}\n"))];
            let mask = g.mask(&reply, &[]).unwrap().expect("bound");
            assert_eq!(allowed_tokens(&mask), vec![u32::from(b'\n')]);
        }
    }
}

#[test]
fn the_marker_binds_wherever_it_stands_to_the_end_of_the_reasoning() {
    // Tokens that hold the closing tag and the start of the marker.
    let mut pieces = vec![MARKER.to_owned(), format!("</think>{MARKER}")];
    for cut in 1..MARKER.len() {
        pieces.extend([format!("</think>{}", &MARKER[..cut]), MARKER[cut..].to_owned()]);
    }
    let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let toy = toy_with(&pieces);
    let marker = toy.token(MARKER);
    let close = toy.token("</think>");
    let with = |before: &str, tokens: &[u32]| {
        let mut reply = toy.tokens(before);
        reply.extend(tokens);
        reply
    };
    for required in [false, true] {
        let free = if required { Next::Unending } else { Next::Free };
        let after_close = if required { Next::CallOrSpace } else { Next::Free };
        // A guide follows one reply: each of these gets its own.
        let at = |reply: &[u32]| next(&toy, &mut calling(&toy, required, thinking()), reply);
        // At the very start of the reasoning, in the middle, at its end.
        assert_eq!(at(&[marker]), Next::InCall);
        assert_eq!(at(&with("One. ", &[marker])), Next::InCall);
        assert_eq!(at(&with("One.\n", &[marker])), Next::InCall);
        // The closing tag after the marker is not the end of any reasoning:
        // the rule of the call holds, and refuses it.
        let mut g = calling(&toy, required, thinking());
        assert_eq!(g.admitted(&with("One. ", &[marker]), &[close]).unwrap(), 0);
        // What only begins like the marker is reasoning, and the closing
        // tag after it ends the reasoning as ever.
        assert_eq!(at(&with("One. <tool_call", &[])), free);
        assert_eq!(at(&with("One. <tool_call", &[close])), after_close);
        // Right after the closing tag, after the template's blank line, and
        // after text of the answer (which a reply that must call cannot write).
        assert_eq!(at(&with("One.\n", &[close])), after_close);
        assert_eq!(at(&with("One.\n", &[close, marker])), Next::InCall);
        let mut blank = with("One.\n", &[close]);
        blank.extend(toy.tokens("\n\n"));
        assert_eq!(at(&blank), after_close);
        blank.push(marker);
        assert_eq!(at(&blank), Next::InCall);
        if !required {
            let mut answer = with("One.\n", &[close]);
            answer.extend(toy.tokens("\n\nSure. <tool_cal"));
            assert_eq!(at(&answer), Next::Free);
            answer.extend(toy.tokens("l>"));
            assert_eq!(at(&answer), Next::InCall);
            // One token for the closing tag and all or a part of the marker:
            // the marker is the answer's, and it binds.
            assert_eq!(at(&with("One.", &[toy.token(&format!("</think>{MARKER}"))])), Next::InCall);
            for cut in 1..MARKER.len() {
                let head = toy.token(&format!("</think>{}", &MARKER[..cut]));
                assert_eq!(at(&with("One.", &[head])), Next::Free, "cut at {cut}");
                let tail = toy.token(&MARKER[cut..]);
                assert_eq!(at(&with("One.", &[head, tail])), Next::InCall, "cut at {cut}");
            }
        }
    }
}

#[test]
fn a_reply_that_must_call_cannot_end_while_it_thinks() {
    let toy = toy_with(&[MARKER]);
    let mut g = calling(&toy, true, thinking());
    let mut reply = Vec::new();
    for piece in ["I ", "think ", "and ", "never ", "close."] {
        reply.extend(toy.tokens(piece));
        let mask = g.mask(&reply, &[]).unwrap().expect("a mask while it thinks");
        assert!(!mask_allows(&mask, toy.eos), "no end after {piece:?}");
        assert!(mask_allows(&mask, u32::from(b'x')) && mask_allows(&mask, toy.control));
        assert!(mask_allows(&mask, toy.token(MARKER)) && mask_allows(&mask, toy.token("</think>")));
    }
    // A draft that ends the reply there is cut, with what follows it.
    assert_eq!(g.admitted(&reply, &[u32::from(b'a'), toy.eos, u32::from(b'b')]).unwrap(), 1);
    assert!(g.mask(&reply, &[u32::from(b'a'), toy.eos]).unwrap().is_none());
    // Drafts through the marker are admitted into the call, up to the first
    // one the rule refuses.
    let mut drafts = vec![u32::from(b' '), toy.token(MARKER)];
    drafts.extend(toy.tokens("\n<function=get>\n</function>\nX"));
    assert_eq!(g.admitted(&reply, &drafts).unwrap(), drafts.len() - 1);
    // With `auto` the reply may end in its reasoning, and no row of it
    // carries a mask.
    let mut g = calling(&toy, false, thinking());
    assert!(g.mask(&reply, &[]).unwrap().is_none());
    assert_eq!(g.admitted(&reply, &[toy.eos]).unwrap(), 1);
}

#[test]
fn without_an_interrupt_or_without_reasoning_nothing_changes() {
    let toy = toy_with(&[MARKER]);
    // A reply that does not think: the interrupt says nothing about it.
    let mut g = calling(&toy, true, Start::Reply);
    assert_eq!(next(&toy, &mut g, &[]), Next::CallOrSpace);
    let mut g = calling(&toy, false, Start::Reply);
    assert_eq!(next(&toy, &mut g, &toy.tokens("Sure.")), Next::Free);
    assert_eq!(next(&toy, &mut g, &toy.tokens("Sure. <tool_call>")), Next::InCall);
    // A reply that may open its reasoning itself: the marker ends it too.
    let lead = Start::AfterReasoning { open: Some("<think>".into()), close: "</think>".into() };
    let at = |required: bool, text: &str| {
        next(&toy, &mut calling(&toy, required, lead.clone()), &toy.tokens(text))
    };
    assert_eq!(at(false, "<think>so <tool_call>"), Next::InCall);
    assert_eq!(at(false, "<tool_call>"), Next::InCall, "no reasoning");
    assert_eq!(at(true, "<think>so "), Next::Unending);
    assert_eq!(at(true, "<think>so <tool_call>"), Next::InCall);
    // An interrupt whose text is empty is refused.
    let shape = Shape::Lark(lark(&WHOLE));
    let empty = Structure {
        shape: shape.clone(),
        start: thinking(),
        trigger: None,
        interrupt: Some(Interrupt { text: String::new(), shape }),
    };
    assert!(toy.compiler.guide(&empty).is_err());
}
