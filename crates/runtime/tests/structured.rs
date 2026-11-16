//! Structured output through the step loop, on the synthetic model with a
//! host executor: the grammar engine behind the sampler's mask, with drafts and
//! without, and with several requests in one step.
//!
//! The synthetic model has one token for each byte, and `~` ends a reply.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ares_api::{Engine, EngineError, Shape, Start, Structure};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{RecordView, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use common::*;

const END: u8 = b'~';

fn model() -> Arc<SyntheticModel> {
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.context_length = 1024;
    config.eos = Some(u32::from(END));
    // Text that loops, so that the n-gram drafter has drafts to offer,
    // next to an MTP block of the model's (the n-gram drafter takes part only
    // next to one).
    config.context.window = 1;
    config.context.epoch = 40;
    config.context.buckets = 512;
    config.mtp_links = 7;
    config.mtp_accuracy = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
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

fn gbnf(grammar: &str) -> Structure {
    Structure::new(Shape::Gbnf(grammar.to_owned()))
}

fn structured(structure: Structure, max_tokens: u32) -> ares_api::GenerateRequest {
    let mut r = request("a prompt", Some(max_tokens));
    r.structure = Some(structure);
    r
}

/// Counts what the records of a run say: the rows that want a pick, and how
/// many of those carry no mask.
#[derive(Default)]
struct Rows {
    wanted: usize,
    unmasked: usize,
    /// Steps that carry the picks of two or more requests.
    shared: usize,
}

fn watch(rows: &Arc<Mutex<Rows>>) -> impl FnMut(&RecordView<'_>) + Send + 'static {
    let rows = rows.clone();
    move |record| {
        let mut rows = rows.lock().unwrap();
        let mut owners = std::collections::HashSet::new();
        for index in 0..record.picked_rows() {
            rows.wanted += 1;
            rows.unmasked += usize::from(record.sampler(index).mask().is_none());
        }
        for segment in 0..record.segments() {
            owners.insert(record.segment(segment).slot);
        }
        rows.shared += usize::from(owners.len() > 1);
    }
}

#[test]
fn a_grammar_is_obeyed_and_drafts_change_nothing() {
    let model = model();
    let grammar = "root ::= [a-c]{60} \"!\"";
    let mut runs = Vec::new();
    for speculation in [false, true] {
        let rows = Arc::new(Mutex::new(Rows::default()));
        let rt = runtime(&model, speculation, watch(&rows));
        let run = run_one(&rt, structured(gbnf(grammar), 400));
        assert_eq!(run.finish, ares_api::FinishReason::Stop, "{run:?}");
        assert_eq!(run.text.chars().count(), 61, "{}", run.text);
        assert!(run.text.chars().take(60).all(|c| ('a'..='c').contains(&c)), "{}", run.text);
        assert!(run.text.ends_with('!'));
        let rows = rows.lock().unwrap();
        assert_eq!(rows.unmasked, 0, "every wanted row carries a mask");
        if speculation {
            let stats = rt.stats().expect("stats");
            assert!(stats.drafts_proposed > 0, "no drafts were verified: nothing was tested");
        }
        runs.push(run.text);
    }
    assert_eq!(runs[0], runs[1], "the text with drafts is the text without");
}

#[test]
fn a_draft_the_grammar_forbids_is_cut_before_it_is_verified() {
    // Runs of one letter: the n-gram drafter goes on with the letter
    // the grammar is about to leave, and the grammar leaves one token to pick.
    let model = model();
    let grammar = "root ::= \"a\"{30} \"b\"{30} \"a\"{30} \"b\"{30}";
    let rows = Arc::new(Mutex::new(Rows::default()));
    let rt = runtime(&model, true, watch(&rows));
    let run = run_one(&rt, structured(gbnf(grammar), 400));
    assert_eq!(run.text, format!("{0}{1}{0}{1}", "a".repeat(30), "b".repeat(30)));
    let stats = rt.stats().expect("stats");
    assert!(stats.drafts_proposed > 10, "{stats:?}");
    // Nothing forced is ever refused: every draft verified was right, and
    // the rows after a forbidden draft were never in a window.
    assert_eq!(stats.drafts_accepted, stats.drafts_proposed, "{stats:?}");
    assert_eq!(rows.lock().unwrap().unmasked, 0);
}

#[test]
fn requests_with_different_grammars_in_one_step_get_their_own() {
    let model = model();
    let first = "root ::= [a-c]{50}";
    let second = "root ::= [x-z]{50}";
    let alone = |grammar: &str| {
        let rt = runtime(&model, true, |_| {});
        run_one(&rt, structured(gbnf(grammar), 100)).text
    };
    let (a, b) = (alone(first), alone(second));
    assert!(a.chars().all(|c| ('a'..='c').contains(&c)) && a.len() == 50, "{a}");
    assert!(b.chars().all(|c| ('x'..='z').contains(&c)) && b.len() == 50, "{b}");

    let rows = Arc::new(Mutex::new(Rows::default()));
    let rt = runtime(&model, true, watch(&rows));
    let start = |grammar: &str| start(&rt, structured(gbnf(grammar), 100)).unwrap();
    let (ha, hb, hc) = (start(first), start(second), start(first));
    let (together_a, together_b, together_c) =
        (ha.finish().text, hb.finish().text, hc.finish().text);
    assert_eq!(together_a, a, "the first, with another grammar next to it");
    assert_eq!(together_b, b, "the second, with another grammar next to it");
    assert_eq!(together_c, a, "the same grammar twice");
    let rows = rows.lock().unwrap();
    assert!(rows.shared > 5, "the requests did not share steps: {}", rows.shared);
    assert_eq!(rows.unmasked, 0);
}

#[test]
fn the_reasoning_is_free_and_the_answer_after_its_closing_tag_is_bound() {
    let model = model();
    // The reasoning ends where the model writes an '8', which it does at its
    // third token; what follows is five letters from x to z, and then the end.
    let structure = Structure {
        shape: Shape::Gbnf("root ::= [x-z]{5}".into()),
        start: Start::AfterReasoning { open: None, close: "8".into() },
        trigger: None,
        interrupt: None,
    };
    let mut texts = Vec::new();
    for speculation in [false, true] {
        let rt = runtime(&model, speculation, |_| {});
        let run = run_one(&rt, structured(structure.clone(), 1000));
        let (reasoning, answer) = run.text.split_once('8').expect("the model wrote an 8");
        assert!(reasoning.chars().any(|c| !('x'..='z').contains(&c)), "free text: {reasoning}");
        assert_eq!(answer.len(), 5, "{}", run.text);
        assert!(answer.chars().all(|c| ('x'..='z').contains(&c)), "{answer}");
        assert_eq!(run.finish, ares_api::FinishReason::Stop);
        texts.push(run.text);
    }
    assert_eq!(texts[0], texts[1]);
}

#[test]
fn a_grammar_or_a_schema_that_is_wrong_is_refused_when_the_request_comes() {
    let rt = runtime(&model(), true, |_| {});
    let refused = |structure: Structure| match rt
        .submit(structured(structure, 10), ares_api::EventSink::new(|_| true))
    {
        Err(EngineError::InvalidRequest(message)) => message,
        other => panic!("{other:?}"),
    };
    let message = refused(gbnf("root ::= missing"));
    assert!(message.starts_with("grammar: "), "{message}");
    assert!(message.contains("line 1") && message.contains("'missing'"), "{message}");
    let message = refused(Structure::new(Shape::JsonSchema("{\"type\": 5".into())));
    assert!(
        message.starts_with("json_schema: ") && message.contains("not valid JSON"),
        "{message}"
    );
    let message = refused(Structure::new(Shape::JsonSchema(r#"{"type":"nonsense"}"#.into())));
    assert!(message.starts_with("json_schema: "), "{message}");
}

#[test]
fn a_reply_the_grammar_cannot_end_ends_in_an_error_and_not_in_free_text() {
    // A grammar that is whole and allows nothing more ends with the end token.
    // A model without one has no way to end, which the request cannot be
    // served with: it ends in an error, and its text is not sent unbound.
    let mut config = SyntheticConfig::small(Split::Single, 1);
    config.eos = None;
    let model = Arc::new(SyntheticModel::new(config).unwrap());
    let rt = runtime(&model, false, |_| {});
    let error = start(&rt, structured(gbnf("root ::= \"ab\""), 50)).unwrap().failure();
    assert!(error.to_string().contains("no token"), "{error}");
}
