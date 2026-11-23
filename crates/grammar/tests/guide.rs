mod common;

use ares_api::{Shape, Start, Structure};
use ares_grammar::Guide;
use common::*;

fn guide(toy: &Toy, shape: Shape, start: Start) -> Guide {
    toy.compiler
        .guide(&Structure { shape, start, trigger: None, interrupt: None })
        .expect("a guide")
}

fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

const YES_NO: &str = r#"root ::= "yes" | "no""#;

fn after_reasoning(open: Option<&str>) -> Start {
    Start::AfterReasoning { open: open.map(str::to_owned), close: "</think>".into() }
}

#[test]
fn a_structure_from_the_reply_start_binds_the_first_token() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(YES_NO.into()), Start::Reply);
    let mask = g.mask(&[], &[]).unwrap().expect("a mask");
    assert_eq!(allowed_tokens(&mask), vec![b'n' as u32, b'y' as u32]);
    let y = toy.tokens("y");
    let mask = g.mask(&y, &[]).unwrap().expect("a mask");
    assert_eq!(allowed_tokens(&mask), vec![b'e' as u32]);
}

#[test]
fn the_reasoning_is_free_and_the_answer_after_its_closing_tag_is_bound() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(YES_NO.into()), after_reasoning(None));
    let think = toy.tokens("hmm, maybe 100? ");
    assert!(g.mask(&[], &[]).unwrap().is_none(), "free from the first token");
    assert!(g.mask(&think, &[]).unwrap().is_none(), "free while it thinks");
    let mut reply = think.clone();
    reply.extend(toy.tokens("</think>"));
    let mask = g.mask(&reply, &[]).unwrap().expect("bound after the tag");
    assert_eq!(allowed_tokens(&mask), vec![b'n' as u32, b'y' as u32]);
    reply.extend(toy.tokens("no"));
    let mask = g.mask(&reply, &[]).unwrap().expect("a mask");
    assert_eq!(
        allowed_tokens(&mask),
        vec![toy.eos],
        "the answer is complete: only the end is left"
    );
}

#[test]
fn a_closing_tag_spelled_with_several_tokens_closes_the_reasoning_too() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(YES_NO.into()), after_reasoning(None));
    // "</" "think" ">" as separate byte tokens.
    let reply = bytes("</think>");
    assert!(g.mask(&reply[..7], &[]).unwrap().is_none());
    let mask = g.mask(&reply, &[]).unwrap().expect("bound");
    assert_eq!(allowed_tokens(&mask), vec![b'n' as u32, b'y' as u32]);
}

#[test]
fn drafts_that_close_the_reasoning_are_followed_into_the_grammar() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(YES_NO.into()), after_reasoning(None));
    let reply = toy.tokens("so ");
    let mut drafts = toy.tokens("</think>");
    drafts.extend(toy.tokens("y"));
    // After the tag and a 'y' only 'e' may come.
    let mask = g.mask(&reply, &drafts).unwrap().expect("a mask");
    assert_eq!(allowed_tokens(&mask), vec![b'e' as u32]);
    // A draft the grammar refuses cuts the rest, and the mask after it is not used.
    let mut bad = toy.tokens("</think>");
    bad.extend(toy.tokens("z"));
    assert_eq!(g.admitted(&reply, &bad).unwrap(), bad.len() - 1);
    assert!(g.mask(&reply, &bad).unwrap().is_none());
}

#[test]
fn a_reply_that_may_begin_with_the_opening_tag_may_think_or_answer() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(YES_NO.into()), after_reasoning(Some("<think>")));
    let think = toy.tokens("<think>")[0];
    let mask = g.mask(&[], &[]).unwrap().expect("a mask");
    let first = allowed_tokens(&mask);
    assert!(first.contains(&think), "the tag may open");
    assert!(first.contains(&(b'y' as u32)) && first.contains(&(b'n' as u32)), "or the answer");
    assert!(!first.contains(&(b'z' as u32)));
    // The tag was taken: the reasoning is free until it is closed.
    assert!(g.mask(&[think], &[]).unwrap().is_none());
    let mut reply = vec![think];
    reply.extend(toy.tokens("ok</think>"));
    let mask = g.mask(&reply, &[]).unwrap().expect("bound");
    assert_eq!(allowed_tokens(&mask), vec![b'n' as u32, b'y' as u32]);
    // The answer taken first leaves the reasoning out.
    let y = toy.tokens("y");
    let mask = g.mask(&y, &[]).unwrap().expect("bound from the start");
    assert_eq!(allowed_tokens(&mask), vec![b'e' as u32]);
}

#[test]
fn an_opening_tag_in_several_tokens_is_followed_token_by_token() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(YES_NO.into()), after_reasoning(Some("<think>")));
    let reply = bytes("<thi");
    let mask = g.mask(&reply, &[]).unwrap().expect("still in the tag");
    assert!(mask_allows(&mask, b'n' as u32) && !mask_allows(&mask, b'y' as u32));
    let mut reply = bytes("<think>");
    assert!(g.mask(&reply, &[]).unwrap().is_none(), "thinking");
    reply.extend(toy.tokens("</think>"));
    assert!(g.mask(&reply, &[]).unwrap().is_some(), "answering");
}

#[test]
fn drafts_are_admitted_up_to_the_first_the_grammar_refuses() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(r#"root ::= "abc" "def""#.into()), Start::Reply);
    let reply = toy.tokens("a");
    assert_eq!(g.admitted(&reply, &bytes("bcdef")).unwrap(), 5);
    assert_eq!(g.admitted(&reply, &bytes("bcxef")).unwrap(), 2);
    assert_eq!(g.admitted(&reply, &bytes("zbc")).unwrap(), 0);
    assert_eq!(g.admitted(&[], &bytes("abcdefg")).unwrap(), 6);
    assert_eq!(g.admitted(&reply, &[]).unwrap(), 0);
}

#[test]
fn asking_again_for_other_drafts_after_the_same_reply_gives_the_right_masks() {
    let toy = toy();
    let mut g = guide(&toy, Shape::Gbnf(r#"root ::= "ab" ("c" | "d") "e""#.into()), Start::Reply);
    let a = toy.tokens("a");
    let first = g.mask(&a, &toy.tokens("b")).unwrap().unwrap();
    assert_eq!(allowed_tokens(&first), vec![b'c' as u32, b'd' as u32]);
    let second = g.mask(&a, &[]).unwrap().unwrap();
    assert_eq!(allowed_tokens(&second), vec![b'b' as u32]);
    let third = g.mask(&toy.tokens("ab"), &toy.tokens("d")).unwrap().unwrap();
    assert_eq!(allowed_tokens(&third), vec![b'e' as u32]);
    let again = g.mask(&a, &toy.tokens("b")).unwrap().unwrap();
    assert_eq!(again, first);
}

#[test]
fn a_walk_through_thinking_and_a_schema_ends_in_valid_json() {
    let toy = toy();
    let schema = r#"{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}"#;
    let mut g = guide(&toy, Shape::JsonSchema(schema.into()), after_reasoning(None));
    let mut reply = toy.tokens("it is { not json } at all ");
    reply.extend(toy.tokens("</think>"));
    let reasoning = reply.len();
    let mut seed = 99u64;
    for _ in 0..200 {
        let mask = g.mask(&reply, &[]).unwrap().expect("bound");
        let options: Vec<u32> =
            allowed_tokens(&mask).into_iter().filter(|&t| t != toy.eos).collect();
        if mask_allows(&mask, toy.eos) && (options.is_empty() || seed.is_multiple_of(3)) {
            break;
        }
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        reply.push(options[(seed >> 33) as usize % options.len()]);
    }
    let answer = toy.text(&reply[reasoning..]);
    let value: serde_json::Value =
        serde_json::from_str(&answer).unwrap_or_else(|e| panic!("{answer:?}: {e}"));
    assert!(value["id"].is_i64() || value["id"].is_u64(), "{answer}");
}
