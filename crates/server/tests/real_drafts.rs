//! The `ares` binary with drafts, over real sockets: a request can switch
//! them off, the reply with drafts is the reply without them (exact products,
//! the fast kernels, the DFlash drafter), and a server without drafters, or
//! with a model without an MTP block, does not speculate.
//!
//! These need a model with an MTP block: `ARES_TEST_DRAFT_MODEL`, a
//! quantization of the 27B that fits one card, such as
//! `Qwen3.8-27B-UD-IQ2_XXS-with-draft-block.gguf` (served with the fast
//! kernels: it has no exact prefill kernels), and for the tests on both cards
//! `ARES_TEST_QWEN38_27B`, the Q4_K_S file; `ARES_TEST_DFLASH_27B` is the
//! 27B's DFlash2 file. The test of a model without an MTP block needs
//! `ARES_TEST_QWEN35_0_8B` (Qwen3.5-0.8B). `ARES_TEST_PORT` is the port the
//! servers listen on, one after the other (default 18960).
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_DRAFT_MODEL=Qwen3.8-27B-UD-IQ2_XXS-with-draft-block.gguf \
//!     ARES_TEST_QWEN38_27B=Qwen3.8-27B-UD-Q4_K_S.gguf \
//!     cargo test -p ares-server --test real_drafts -- --test-threads=1
//! ```

use std::path::PathBuf;

use common::model_file;
use common::process::{Client, ENGLISH, GERMAN, serve};
use common::socket::streamed_text;
use serde_json::{Value, json};

mod common;

/// The 0.8B, which has no MTP block.
fn model() -> Option<PathBuf> {
    model_file("ARES_TEST_QWEN35_0_8B", "a small qwen35 model file")
}

/// The port of every server: one runs at a time.
fn port() -> u16 {
    common::process::port(18951, 9)
}

fn draft_model() -> Option<PathBuf> {
    model_file("ARES_TEST_DRAFT_MODEL", "a model file with an MTP block")
}

/// The 27B with its MTP block for the tests that need exact products and so
/// a type of weights with a prefill kernel for them: `ARES_TEST_QWEN38_27B`
/// (Qwen3.8-27B-UD-Q4_K_S.gguf), served on both cards.
fn two_card_model() -> Option<PathBuf> {
    model_file("ARES_TEST_QWEN38_27B", "the 27B Q4_K_S file (two GPUs)")
}

/// The reply to `question`, greedy, with drafts or without (`"draft": false`):
/// the hash of its token ids as the last chunk of the stream and the final
/// message of a plain request give it, and the drafts verified for it.
fn reply_tokens(server: &Client, question: &str, drafts: bool) -> (String, String, Option<u64>) {
    reply_tokens_as(server, question, drafts, json!({"max_tokens": 160, "temperature": 0}))
}

/// [`reply_tokens`] with the settings of `extra` (the tokens, the sampler).
fn reply_tokens_as(
    server: &Client,
    question: &str,
    drafts: bool,
    mut extra: Value,
) -> (String, String, Option<u64>) {
    if !drafts {
        extra["draft"] = json!(false);
    }
    let plain = server.post("/v1/chat/completions", &server.chat(question, extra.clone()));
    assert_eq!(plain.status, 200, "{}", plain.body);
    let body = plain.json();
    let hash = body["timings"]["token_hash"].as_str().expect("a token hash").to_owned();
    let message = &body["choices"][0]["message"];
    let text = [message["reasoning_content"].as_str(), message["content"].as_str()]
        .into_iter()
        .flatten()
        .collect::<String>();
    // The stream carries the same hash in its last event with timings.
    extra["stream"] = json!(true);
    let streamed = server.post("/v1/chat/completions", &server.chat(question, extra));
    let (chunks, done) = streamed.events();
    assert!(done);
    let last = chunks.iter().rev().find(|c| c["timings"].is_object()).expect("timings at the end");
    assert_eq!(last["timings"]["token_hash"].as_str(), Some(hash.as_str()), "plain and stream");
    assert_eq!(streamed_text(&chunks), text);
    (hash, text, body["timings"]["draft_n"].as_u64())
}

#[test]
fn a_request_can_switch_drafts_off_and_the_reply_has_a_hash_of_its_tokens() {
    let Some(model) = draft_model() else { return };
    let server = serve(&model, port(), &["--parallel", "1", "--precision", "fast"]);
    let (_, _, drafted) = reply_tokens(&server, ENGLISH, true);
    assert!(drafted.unwrap_or(0) > 0, "the MTP block and the n-gram drafter: {drafted:?}");
    let (_, _, none) = reply_tokens(&server, ENGLISH, false);
    assert_eq!(none, Some(0), "no draft row for a request that asks for none");
    // The hash is a function of the tokens.
    let (a, ..) = reply_tokens(&server, GERMAN, false);
    let (b, ..) = reply_tokens(&server, GERMAN, false);
    assert_eq!(a, b);
    let (c, ..) = reply_tokens(&server, ENGLISH, false);
    assert_ne!(a, c);
    server.stop();
}

/// What the tests of drafts ask.
const QUESTIONS: [&str; 3] =
    [GERMAN, ENGLISH, "Write a Rust function that reverses a string, with a test."];

/// Asks every question with drafts and without, and says for each whether the
/// replies are the same tokens: `None`, or where they part (the characters
/// they share).
fn replies_with_and_without_drafts(server: &Client) -> Vec<(&'static str, Option<usize>)> {
    replies_with_and_without_drafts_as(server, &json!({"max_tokens": 160, "temperature": 0}))
}

/// [`replies_with_and_without_drafts`] with the settings of `extra`.
fn replies_with_and_without_drafts_as(
    server: &Client,
    extra: &Value,
) -> Vec<(&'static str, Option<usize>)> {
    QUESTIONS
        .into_iter()
        .map(|question| {
            let (with, with_text, drafted) = reply_tokens_as(server, question, true, extra.clone());
            let (without, without_text, _) =
                reply_tokens_as(server, question, false, extra.clone());
            assert!(drafted.unwrap_or(0) > 0, "{question}: drafts were verified");
            let shared = with_text.chars().zip(without_text.chars()).take_while(|(a, b)| a == b);
            let parted = (with != without).then(|| shared.count());
            assert_eq!(parted.is_none(), with_text == without_text, "{question}: hash and text");
            (question, parted)
        })
        .collect()
}

/// With exact products, too, the tokens with drafts are the tokens without
/// them, and the hash of the reply says so over HTTP.
#[test]
fn with_exact_products_the_reply_with_drafts_is_the_reply_without() {
    // Exact products need a prefill kernel for every type of the weights, which
    // the one-card file (IQ2_XXS) has none for: the 27B in Q4_K_S, on both cards.
    let Some(model) = two_card_model() else { return };
    let flags =
        ["--devices", "0,1", "--split", "tensor", "--parallel", "1", "--precision", "exact"];
    let server = serve(&model, port(), &flags);
    for (question, parted) in replies_with_and_without_drafts(&server) {
        assert_eq!(parted, None, "{question}: the replies part after this many characters");
    }
    server.stop();
}

/// With the fast kernels, which is how a model is served, the reply with
/// drafts is the reply without them, token for token: a row of a draft window
/// computes with the bits it has alone (the weight kernels, the recurrent
/// blocks and attention all do; `crates/model/tests/qwen35_windows.rs` holds
/// every block to it). Over replies of 300 tokens, so that a rare difference
/// would show, with the most probable token and sampled with a seed, on the
/// one-card file; and on both cards in a tensor split when the larger file is
/// there.
#[test]
fn with_the_fast_kernels_the_reply_with_drafts_is_the_reply_without() {
    let Some(model) = draft_model() else { return };
    let greedy = json!({"max_tokens": 300, "temperature": 0});
    let sampled =
        json!({"max_tokens": 300, "temperature": 0.8, "top_k": 20, "top_p": 0.95, "seed": 4711});
    let server = serve(&model, port(), &["--parallel", "1", "--precision", "fast"]);
    for (how, extra) in [("the most probable token", &greedy), ("sampled", &sampled)] {
        for (question, parted) in replies_with_and_without_drafts_as(&server, extra) {
            assert_eq!(
                parted, None,
                "{how}, {question}: the replies part after this many characters"
            );
        }
    }
    server.stop();
    let Some(model) = two_card_model() else { return };
    let flags = ["--devices", "0,1", "--split", "tensor", "--parallel", "1", "--precision", "fast"];
    let server = serve(&model, port(), &flags);
    for (how, extra) in [("the most probable token", &greedy), ("sampled", &sampled)] {
        for (question, parted) in replies_with_and_without_drafts_as(&server, extra) {
            assert_eq!(parted, None, "both cards, {how}, {question}: the replies part");
        }
    }
    server.stop();
}

/// The DFlash drafter (`ARES_TEST_DFLASH_27B`, the 27B's DFlash2 file) next
/// to the MTP block and alone, on both cards with the fast kernels: the
/// reply with drafts is the reply without them, token for token, greedy and
/// sampled, and the drafter's drafts are verified.
#[test]
fn with_the_dflash_drafter_the_reply_with_drafts_is_the_reply_without() {
    let Some(model) = two_card_model() else { return };
    let Some(dflash) = std::env::var_os("ARES_TEST_DFLASH_27B") else {
        eprintln!("skipped: set ARES_TEST_DFLASH_27B to the 27B's DFlash2 file");
        return;
    };
    let dflash = dflash.to_string_lossy().into_owned();
    let greedy = json!({"max_tokens": 300, "temperature": 0});
    let sampled =
        json!({"max_tokens": 300, "temperature": 0.8, "top_k": 20, "top_p": 0.95, "seed": 4711});
    for drafters in [format!("mtp,{dflash}"), dflash.clone()] {
        let flags = [
            "--devices",
            "0,1",
            "--split",
            "tensor",
            "--parallel",
            "1",
            "--precision",
            "fast",
            "--drafters",
            drafters.as_str(),
        ];
        let server = serve(&model, port(), &flags);
        for (how, extra) in [("the most probable token", &greedy), ("sampled", &sampled)] {
            for (question, parted) in replies_with_and_without_drafts_as(&server, extra) {
                assert_eq!(parted, None, "--drafters {drafters}, {how}, {question}: they part");
            }
        }
        let (_, _, drafted) = reply_tokens(&server, ENGLISH, true);
        assert!(drafted.unwrap_or(0) > 0, "--drafters {drafters}: drafts were verified");
        server.stop();
    }
}

/// A server started with no drafters does not speculate, whatever the text
/// repeats: the n-gram drafter is no drafter on its own.
#[test]
fn a_server_with_no_drafters_never_drafts_and_one_with_the_mtp_block_does() {
    let Some(model) = draft_model() else { return };
    let repeat = "Write the sentence 'The quick brown fox jumps over the lazy dog.' ten times.";
    let flags = ["--parallel", "1", "--precision", "fast"];
    let mut none = Vec::from(flags);
    none.extend(["--drafters", "none"]);
    let mut server = serve(&model, port(), &none);
    assert!(
        server.logged().iter().any(|l| l.contains("no drafters") && l.contains("no speculation")),
        "{:#?}",
        server.logged()
    );
    for drafts in [true, false] {
        let (_, text, drafted) = reply_tokens(&server, repeat, drafts);
        assert!(text.contains("quick brown fox"), "{text}");
        assert_eq!(drafted, Some(0), "no draft rows with --drafters none (draft: {drafts})");
    }
    assert_eq!(server.metric("drafts_proposed_total"), 0, "the counters say the same");
    server.stop();
    let server = serve(&model, port(), &flags);
    let (_, _, drafted) = reply_tokens(&server, repeat, true);
    assert!(drafted.unwrap_or(0) > 0, "the same request drafts with the model's MTP block");
    server.stop();
}

/// A model with no MTP block has no drafter, and the n-gram drafter does not run
/// without one.
#[test]
fn a_model_with_no_mtp_block_is_served_with_no_speculation_at_all() {
    let Some(model) = model() else { return };
    let repeat = "Write the sentence 'The quick brown fox jumps over the lazy dog.' ten times.";
    let mut server = serve(&model, port(), &["--parallel", "1"]);
    assert!(
        server
            .logged()
            .iter()
            .any(|l| l.contains("no model drafter") && l.contains("no speculation")),
        "{:#?}",
        server.logged()
    );
    let (_, text, drafted) = reply_tokens(&server, repeat, true);
    assert!(text.contains("quick brown fox"), "{text}");
    assert_eq!(drafted, Some(0));
    assert_eq!(server.metric("drafts_proposed_total"), 0);
    server.stop();
}
