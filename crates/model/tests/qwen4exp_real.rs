//! The `qwen4exp` reference on Qwen3.8-Flash-Next. The test needs the model
//! file (63.5 GiB in two parts, never in the repository) and skips with a
//! message unless it is told where it is:
//!
//! ```sh
//! ARES_TEST_QWEN4EXP_MODEL=<...>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
//!   cargo test --release -p ares-model --test qwen4exp_real -- --nocapture
//! ```
//!
//! It runs 26 tokens and took 224 seconds on a machine that was busy with
//! other work (opening the file and building the weight table: 0.1 seconds).

use ares_gguf::Gguf;
use ares_model::qwen4exp::{Config, LogitRows, Reference, SequenceState, Weights};

/// `<|im_start|>user\nErkläre in drei Sätzen, warum der Himmel blau
/// ist.<|im_end|>\n<|im_start|>assistant\n`
const PROMPT: [u32; 22] = [
    248045, 846, 198, 18871, 10248, 45125, 303, 55593, 326, 179212, 11, 183857, 2607, 199908,
    242880, 5810, 13, 248046, 198, 248045, 74455, 198,
];

/// What the model continues with under greedy picks: `<think>\nThe user ...`.
/// Strata 0.1.48 gives the same tokens for this prompt (and the same 64
/// tokens after it; `docs/reference/qwen4exp-architecture.md`, section 10).
const CONTINUATION: [u32; 4] = [248068, 198, 760, 1156];

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
fn flash_next_continues_a_german_prompt_as_strata_does() {
    let Some(path) = std::env::var_os("ARES_TEST_QWEN4EXP_MODEL") else {
        eprintln!("skipped: set ARES_TEST_QWEN4EXP_MODEL to the first part of the model file");
        return;
    };
    let started = std::time::Instant::now();
    let gguf = Gguf::open(&path).expect("the model file opens");
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    // The table is built without reading the experts or the n-gram table.
    eprintln!("configuration and weight table in {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!((config.n_layer, config.n_embd, config.n_stream), (48, 2560, 4));
    assert_eq!((config.experts.count, config.experts.used, config.experts.ff), (512, 10, 640));
    assert_eq!(config.n_attention_layers(), 12);
    assert_eq!(config.indexer.dense_limit(), 2051);
    let ngram = config.ngram.as_ref().expect("the model has the n-gram layer");
    assert_eq!((ngram.layer, ngram.n_heads(), ngram.row_dim), (1, 16, 160));
    assert_eq!(ngram.table_rows, 320_001_536);
    assert_eq!(weights.layers[0].moe.down.expert_bytes(), 460_800);

    let reference = Reference::new(&config, &weights).keep_residual(false);
    let mut state = SequenceState::new(&config);
    let window = reference.forward(&mut state, &PROMPT, LogitRows::Last).expect("prompt");
    let logits = window.logits_of(PROMPT.len() - 1).expect("logits of the last row");
    let mut next = argmax(logits);
    // 28.833 when this was written; Strata's value for the row is 29.597.
    let top = logits[next as usize];
    assert!((top - 28.833).abs() < 0.02, "the largest logit is {top}");
    for (i, &expected) in CONTINUATION.iter().enumerate() {
        assert_eq!(next, expected, "generated token {i}");
        let window = reference.forward(&mut state, &[next], LogitRows::Last).expect("token");
        next = argmax(window.logits_of(0).expect("logits"));
    }
    eprintln!("{} tokens in {:.0} s", state.position(), started.elapsed().as_secs_f64());
}
