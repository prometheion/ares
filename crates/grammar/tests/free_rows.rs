//! The rows of a window a structure leaves free ([`ares_grammar::Guide::free_rows`]):
//! what lets a reply take drafts whose tokens nobody knows when the masks of
//! the window are written. The count must be what the masks say, row for row.

mod common;

use ares_api::{Shape, Start, Structure};
use common::*;

const YES_NO: &str = r#"root ::= "yes" | "no""#;

/// One token for each byte.
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn structure(open: Option<&str>, close: Option<&str>, trigger: Option<&str>) -> Structure {
    Structure {
        shape: Shape::Gbnf(YES_NO.into()),
        start: match close {
            Some(close) => {
                Start::AfterReasoning { open: open.map(str::to_owned), close: close.to_owned() }
            }
            None => Start::Reply,
        },
        trigger: trigger.map(str::to_owned),
    }
}

/// Checks `free_rows` against the masks for every way to cut `text` into a
/// reply and the tokens of a window after it: the free rows are the rows from
/// the first on for which there is no mask. `text` is text the structure
/// allows, so a row without a mask is a row the structure does not bind.
/// Returns the count for the cut after `reply` tokens.
fn check(toy: &Toy, structure: &Structure, text: &str, reply: usize) -> usize {
    let path = bytes(text);
    let mut counted = toy.compiler.guide(structure).expect("a guide");
    let mut at_reply = None;
    for cut in 0..=path.len() {
        let (before, tokens) = path.split_at(cut);
        // A guide of its own for the masks: the one that counts is asked as
        // the conductor asks it, for a reply that grows.
        let mut masks = toy.compiler.guide(structure).expect("a guide");
        let want = (0..=tokens.len())
            .take_while(|&k| masks.mask(before, &tokens[..k]).unwrap().is_none())
            .count();
        let got = counted.free_rows(before, tokens).unwrap();
        assert_eq!(got, want, "{text:?} cut after {cut} tokens");
        // Fewer tokens of the same window: the same rows, as far as they go.
        for shorter in 0..tokens.len() {
            let got = counted.free_rows(before, &tokens[..shorter]).unwrap();
            assert_eq!(got, want.min(shorter + 1), "{text:?} cut after {cut}, {shorter} tokens");
        }
        if cut == reply {
            at_reply = Some(got);
        }
    }
    at_reply.expect("the cut is inside the text")
}

#[test]
fn the_rows_of_the_reasoning_are_free_up_to_the_one_after_its_closing_tag() {
    let toy = toy();
    let text = "hm, 1? </think>yes";
    let s = structure(None, Some("</think>"), None);
    // From the first token: every row up to the one that follows the tag's
    // last token, which is the first the grammar binds. The token that ends
    // the tag is itself picked by a free row.
    let tag_ends = text.find("</think>").unwrap() + "</think>".len();
    assert_eq!(check(&toy, &s, text, 0), tag_ends);
    // In the middle of the tag.
    assert_eq!(check(&toy, &s, text, tag_ends - 3), 3);
    // After it nothing is free, whatever follows.
    assert_eq!(check(&toy, &s, text, tag_ends), 0);
    assert_eq!(check(&toy, &s, text, tag_ends + 1), 0);
}

#[test]
fn the_rows_before_a_trigger_are_free_and_the_ones_after_it_are_not() {
    let toy = toy();
    let s = structure(None, None, Some("<call>"));
    let text = "let me <c see <call>no";
    let trigger_ends = text.find("<call>").unwrap() + "<call>".len();
    assert_eq!(check(&toy, &s, text, 0), trigger_ends);
    assert_eq!(check(&toy, &s, text, trigger_ends - 1), 1);
    assert_eq!(check(&toy, &s, text, trigger_ends), 0);
    // After the reasoning the answer waits for the trigger: both are free.
    let s = structure(None, Some("</think>"), Some("<call>"));
    let text = "think <call> </think>then <call>yes";
    assert_eq!(check(&toy, &s, text, 0), text.len() - "yes".len());
}

#[test]
fn a_reply_that_may_open_its_reasoning_is_free_only_if_it_waits_for_a_trigger() {
    let toy = toy();
    // Without a trigger the first tokens are held to the tag or the grammar.
    let s = structure(Some("<think>"), Some("</think>"), None);
    let text = "<think>so </think>no";
    assert_eq!(check(&toy, &s, text, 0), 0);
    assert_eq!(check(&toy, &s, text, "<thi".len()), 0);
    // Once the tag is whole the reasoning is free.
    let tag_ends = text.find("</think>").unwrap() + "</think>".len();
    assert_eq!(check(&toy, &s, text, "<think>".len()), tag_ends - "<think>".len());
    assert_eq!(check(&toy, &s, "yes", 0), 0);
    // With a trigger the reply is free from its first token, as reasoning or
    // as an answer.
    let s = structure(Some("<think>"), Some("</think>"), Some("<call>"));
    let text = "<think>so </think>well <call>no";
    assert_eq!(check(&toy, &s, text, 0), text.len() - "no".len());
    let text = "<thin air <call>yes";
    assert_eq!(check(&toy, &s, text, 0), text.len() - "yes".len());
}

#[test]
fn a_structure_that_binds_from_the_first_token_has_no_free_row() {
    let toy = toy();
    let s = structure(None, None, None);
    assert_eq!(check(&toy, &s, "yes", 0), 0);
    assert_eq!(check(&toy, &s, "no", 1), 0);
}
