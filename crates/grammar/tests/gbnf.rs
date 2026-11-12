mod common;

use ares_grammar::{Error, Source};
use common::*;

fn compile_error(toy: &Toy, text: &str) -> String {
    match toy.compiler.compile(Source::Gbnf(text)) {
        Ok(_) => panic!("{text:?} should not compile"),
        Err(Error::Gbnf(message)) => message,
        Err(other) => panic!("{text:?}: the wrong kind of error: {other:?}"),
    }
}

#[test]
fn a_literal_and_a_choice_accept_their_sentences_and_nothing_else() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf(r#"root ::= "yes" | "no" | "maybe""#));
    for ok in ["yes", "no", "maybe"] {
        assert!(accepts(&toy, &mut g, ok), "{ok}");
    }
    for bad in ["", "ye", "yess", "nope", "Yes"] {
        assert!(!accepts(&toy, &mut g, bad), "{bad}");
    }
    assert!(viable(&toy, &mut g, "ye") && !viable(&toy, &mut g, "yx"));
}

#[test]
fn llama_cpps_json_grammar_makes_json() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf(include_str!("grammars/json.gbnf")));
    let mut ended = 0;
    for seed in 1..40 {
        let (tokens, done) = walk(&toy, &mut g, seed, 300);
        if done {
            ended += 1;
            let text = toy.text(&tokens);
            // A number with an exponent past what a double holds is still JSON.
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(value) => assert!(value.is_object(), "{text:?}"),
                Err(e) => assert!(e.to_string().contains("out of range"), "{text:?}: {e}"),
            }
        }
    }
    assert!(ended > 5, "only {ended} of 39 walks ended");
}

#[test]
fn llama_cpps_arithmetic_grammar_follows_its_rules() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf(include_str!("grammars/arithmetic.gbnf")));
    assert!(accepts(&toy, &mut g, "x=3\n"));
    assert!(accepts(&toy, &mut g, "a1+2*(b-3)=c\n"));
    assert!(accepts(&toy, &mut g, "1=1\n2 =  y_9\n"));
    assert!(!accepts(&toy, &mut g, "x=3"), "the line is not ended");
    assert!(!viable(&toy, &mut g, "x==3"));
    assert!(!viable(&toy, &mut g, "A=1"), "identifiers are lower case");
}

#[test]
fn llama_cpps_list_grammar_excludes_line_breaks_in_items() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf(include_str!("grammars/list.gbnf")));
    assert!(accepts(&toy, &mut g, "- one\n- two words\n"));
    assert!(accepts(&toy, &mut g, "- \u{e9}\u{4e2d}\n"));
    assert!(!viable(&toy, &mut g, "- a\u{2028}"));
    assert!(!accepts(&toy, &mut g, "- \n"), "an item has at least one character");
}

#[test]
fn counted_repetitions_take_the_counts_they_name() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf(
        "root ::= \"[\" d{3} \"|\" d{1,2} \"|\" d{2,} \"|\" d? \"]\"\nd ::= [0-9]",
    ));
    assert!(accepts(&toy, &mut g, "[123|4|56|]"));
    assert!(accepts(&toy, &mut g, "[123|45|5678901|7]"));
    assert!(!viable(&toy, &mut g, "[12|"), "three digits first");
    assert!(!viable(&toy, &mut g, "[123|456"), "at most two");
    assert!(!accepts(&toy, &mut g, "[123|4|5|]"), "at least two");
}

#[test]
fn a_dot_takes_any_character_and_a_negated_class_all_others() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf("root ::= . [^a-c\\n] .\n"));
    assert!(accepts(&toy, &mut g, "axb"));
    assert!(accepts(&toy, &mut g, "\n\u{e9}\u{1f600}"));
    assert!(!viable(&toy, &mut g, "xa"));
    assert!(!viable(&toy, &mut g, "x\n"));
}

#[test]
fn escapes_in_literals_and_classes_are_read_as_llama_cpp_reads_them() {
    let toy = toy();
    let mut g =
        toy.grammar(Source::Gbnf(r#"root ::= "\x41\u00e9\U0001F600\t\"\\" [\]\[\-\x30-\x32]+"#));
    assert!(accepts(&toy, &mut g, "A\u{e9}\u{1f600}\t\"\\]-[012"));
}

#[test]
fn comments_and_line_ends_split_rules_the_way_llama_cpp_splits_them() {
    let toy = toy();
    let text = "# a comment\n\
                root ::= a |  # the rest\n\
                  b\n\
                \n\
                a ::=\n  \"x\"\n\
                b ::= (\"y\"\n   \"z\"\n )\n";
    let mut g = toy.grammar(Source::Gbnf(text));
    assert!(accepts(&toy, &mut g, "x") && accepts(&toy, &mut g, "yz"));
    assert!(!accepts(&toy, &mut g, "y"));
}

#[test]
fn rules_may_be_left_recursive_and_empty() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf("root ::= e\ne ::= e \"+\" \"1\" | \"1\"\nunused ::=\n"));
    assert!(accepts(&toy, &mut g, "1+1+1"));
    assert!(!accepts(&toy, &mut g, "1+"));
    let mut g = toy.grammar(Source::Gbnf("root ::= \"a\" ws \"b\"\nws ::=\n"));
    assert!(accepts(&toy, &mut g, "ab"));
}

#[test]
fn tokens_are_named_by_number_or_by_text() {
    let toy = toy();
    // Token 97 is the byte 'a'; "<think>" is a token of the toy vocabulary.
    let mut g = toy.grammar(Source::Gbnf("root ::= <[97]> <think>"));
    let mut cursor = g.start();
    let mask = cursor.mask().unwrap();
    assert_eq!(allowed_tokens(&mask), vec![97]);
    assert_eq!(cursor.goto(&[97]), 1);
    let mask = cursor.mask().unwrap();
    let think = toy.tokens("<think>")[0];
    assert_eq!(allowed_tokens(&mask), vec![think]);
    // Every token but one.
    let mut g = toy.grammar(Source::Gbnf("root ::= !<[97]>"));
    let mask = g.start().mask().unwrap();
    assert!(!allowed(&mask, 97) && allowed(&mask, 98) && allowed(&mask, 0));
}

#[test]
fn a_grammar_that_is_wrong_is_refused_with_the_place() {
    let toy = toy();
    let cases: &[(&str, &str)] = &[
        ("root ::= a", "line 1, column 10: the rule 'a' is used and never defined"),
        ("a ::= \"x\"", "there is no rule named 'root'"),
        ("root ::= \"x", "line 1, column 10: a string is not closed"),
        ("root ::= [a-", "a character class is not closed"),
        ("root ::= \"\\q\"", "unknown escape '\\q'"),
        ("root ::= \"\\u12\"", "expecting 4 hex digits"),
        ("root ::= \"\\ud800\"", "is not a character"),
        ("root \"x\"", "line 1, column 6: expecting '::='"),
        ("root ::= (\"x\"", "expecting ')'"),
        ("root ::= \"x\" )", "expecting a line end"),
        ("root ::= *", "expecting an item before"),
        ("root ::= \"x\" {", "expecting a number after '{'"),
        ("root ::= \"x\"{3000}", "more than 2000 times"),
        ("root ::= \"x\"{3,2}", "the most is less than the least"),
        ("root ::= \"x\"\nroot ::= \"y\"", "line 2, column 1: the rule 'root' is defined twice"),
        ("root ::= []", "no characters"),
        ("root ::= <[999999]>", "is not in a vocabulary"),
        ("root ::= <nothing>", "is not a token with that text"),
        ("", "there is no rule named 'root'"),
    ];
    for (text, wants) in cases {
        let message = compile_error(&toy, text);
        assert!(message.contains(wants), "{text:?}: {message:?} lacks {wants:?}");
    }
}

#[test]
fn a_string_rule_written_character_by_character_reads_quotes_and_escapes() {
    let toy = toy();
    let mut g = toy.grammar(Source::Gbnf(
        "root ::= string (\",\" string)*\nstring ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" .)* \"\\\"\"",
    ));
    assert!(accepts(&toy, &mut g, r#""abc""#));
    assert!(accepts(&toy, &mut g, r#""a\"b","\\","""#));
    assert!(!accepts(&toy, &mut g, r#""abc"#), "not closed");
    assert!(!viable(&toy, &mut g, r#""a"b"#), "a quote ends the string");
    assert!(viable(&toy, &mut g, r#""a\"#), "an escape takes the next character");
}

#[test]
fn repetitions_leave_what_the_rest_of_the_rule_needs() {
    let toy = toy();
    // `d+` must give back the last digit: the grammar reads one character at a time.
    let mut g = toy.grammar(Source::Gbnf("root ::= d+ [0-9]\nd ::= [0-9]"));
    assert!(accepts(&toy, &mut g, "12"));
    assert!(accepts(&toy, &mut g, "9876"));
    assert!(!accepts(&toy, &mut g, "1"));
    let mut g = toy.grammar(Source::Gbnf("root ::= num \"x\"\nnum ::= [0-9]+"));
    assert!(accepts(&toy, &mut g, "123x") && !accepts(&toy, &mut g, "x"));
}
