//! The 27B on two cards in this process, its attention heads placed three
//! ways: with the weights (the shares of the split), by shares given, and
//! where the devices have the room for the most context. The plan's context
//! grows with what the second card holds of K and V, and the text stays the
//! model's.
//!
//! Skipped unless `ARES_TEST_CUDA=1` and the 27B file are set, and two GPUs
//! are there:
//!
//! ```text
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_27B=<Q4_K_S file> \
//!   cargo test --release -p ares-server --test real_devices -- --nocapture
//! ```

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Sampling};
use ares_server::load;
use ares_server::plan::{AttentionShares, LoadOptions, SplitKind};

/// The text of a greedy answer of `max_tokens` tokens to `text`.
fn answer(engine: &Arc<dyn Engine>, model: &str, text: &str, max_tokens: u32) -> String {
    let (done, wait) = mpsc::channel();
    let pieces = Arc::new(std::sync::Mutex::new(String::new()));
    let sink = {
        let pieces = pieces.clone();
        EventSink::new(move |event| {
            match event {
                Event::Text(piece) => pieces.lock().unwrap().push_str(&piece),
                Event::Done { .. } => {
                    let _ = done.send(Ok(()));
                }
                Event::Error(error) => {
                    let _ = done.send(Err(error.to_string()));
                }
                Event::Logprobs(_) => {}
            }
            true
        })
    };
    let request = GenerateRequest {
        model: model.to_owned(),
        prompt: Prompt::Text(text.to_owned()),
        sampling: Sampling { temperature: Some(0.0), ..Sampling::default() },
        max_tokens: Some(max_tokens),
        stop: Vec::new(),
        draft: false,
        structure: None,
        id: RequestId(1),
    };
    engine.submit(request, sink).expect("the engine takes the request");
    wait.recv_timeout(Duration::from_secs(300)).expect("an answer in time").expect("no error");
    pieces.lock().unwrap().clone()
}

#[test]
fn the_attention_heads_go_where_the_room_is_and_the_context_grows() {
    let (Some(_), Some(model)) =
        (std::env::var_os("ARES_TEST_CUDA"), std::env::var_os("ARES_TEST_QWEN35_27B"))
    else {
        eprintln!("skipped: set ARES_TEST_CUDA=1 and ARES_TEST_QWEN35_27B");
        return;
    };
    if ares_cuda::devices().map_or(0, |d| d.len()) < 2 {
        eprintln!("skipped: one GPU");
        return;
    }
    let question = "Erkläre in drei Sätzen, warum der Himmel blau ist.\n\n";
    let mut results = Vec::new();
    for (name, attention) in [
        ("with the weights", None),
        ("a quarter on the first card", Some(AttentionShares::Given(vec![1.0, 3.0]))),
        ("for the most context", Some(AttentionShares::ByContext)),
    ] {
        let mut options = LoadOptions::new(PathBuf::from(&model));
        options.devices = Some(vec![0, 1]);
        options.split = Some(SplitKind::Tensor);
        options.shares = Some(vec![0.625, 0.375]);
        options.drafters = Some(Vec::new());
        options.attention = attention;
        let loaded = load::load(&options).expect("the model loads");
        let engine = loaded.engine();
        let pages = engine.stats().expect("stats").pages;
        let text = answer(&engine, loaded.name(), question, 40);
        eprintln!("RES heads {name}: {pages} pages of K and V; {text:?}");
        drop(engine);
        loaded.unload();
        results.push((pages, text));
    }
    let (plain, given, most) = (&results[0], &results[1], &results[2]);
    // Three quarters of K and V on the card that holds three eighths of the
    // weights: the plan holds more than half as much again.
    assert!(given.0 * 2 > plain.0 * 3, "{} pages against {}", given.0, plain.0);
    assert!(most.0 >= given.0, "{} pages against {}", most.0, given.0);
    // The heads compute what they computed; the sums of the attention
    // blocks add their partials in another grouping, so a near tie may part
    // the texts late, never at their start.
    for (pages, text) in [given, most] {
        let same = text.chars().zip(plain.1.chars()).take_while(|(a, b)| a == b).count();
        assert!(same >= 40, "{pages} pages: {text:?} against {:?}", plain.1);
    }
}
