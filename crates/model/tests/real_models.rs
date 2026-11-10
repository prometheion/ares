//! The two real models and what llama.cpp computed for them. These tests need
//! files that are not in the repository and say so when they skip:
//!
//! - `ARES_TEST_QWEN35_MODEL`: Qwen3.5-0.8B-Q4_K_M.gguf (configuration, table, a run);
//! - `ARES_TEST_QWEN35_CAPTURE`: a directory made by `tools/capture` for that model
//!   (`english/base`, `german/base`, `code/base`), see its README;
//! - `ARES_TEST_QWEN35_27B`: Qwen3.8-27B-UD-Q4_K_S.gguf (configuration and
//!   table only; the reference is not run on it).

use std::path::{Path, PathBuf};

use ares_gguf::Gguf;
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("skipped: set {name} to run this test");
            None
        }
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (f64::from(*x) - f64::from(*y)).powi(2);
        n += f64::from(*y).powi(2);
    }
    (d / n).sqrt()
}

fn argmax(x: &[f32]) -> (usize, f32) {
    let mut best = 0;
    let mut second = f32::NEG_INFINITY;
    for i in 1..x.len() {
        if x[i] > x[best] {
            second = x[best];
            best = i;
        } else if x[i] > second {
            second = x[i];
        }
    }
    (best, x[best] - second)
}

#[test]
fn the_small_model_is_configured_as_its_file_says() {
    let Some(path) = env_path("ARES_TEST_QWEN35_MODEL") else { return };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    assert_eq!(
        (c.n_layer, c.n_embd, c.n_ff, c.n_vocab, c.n_ctx),
        (24, 1024, 3584, 248_320, 262_144)
    );
    assert_eq!((c.attention.n_head, c.attention.n_head_kv, c.attention.head_dim), (8, 2, 256));
    assert_eq!((c.attention.rope.dim, c.attention.rope.sections), (64, [11, 11, 10, 0]));
    assert_eq!(
        (c.recurrent.conv_kernel, c.recurrent.n_k_heads, c.recurrent.n_v_heads),
        (4, 16, 16)
    );
    assert_eq!(
        (c.recurrent.head_k_dim, c.recurrent.head_v_dim, c.recurrent.conv_dim()),
        (128, 128, 6144)
    );
    assert_eq!(c.n_nextn, 0);
    assert_eq!(c.n_attention_layers(), 6);
    let w = Weights::load(&gguf, &c).unwrap();
    assert!(w.output.is_none(), "the small model ties its head to the embeddings");
    assert!(w.nextn.is_none());
}

#[test]
fn the_large_model_is_configured_as_its_file_says() {
    let Some(path) = env_path("ARES_TEST_QWEN35_27B") else { return };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    assert_eq!((c.n_layer, c.n_embd, c.n_ff, c.n_vocab), (64, 5120, 17408, 248_320));
    assert_eq!((c.attention.n_head, c.attention.n_head_kv, c.attention.head_dim), (24, 4, 256));
    assert_eq!(
        (c.recurrent.n_k_heads, c.recurrent.n_v_heads, c.recurrent.conv_dim()),
        (16, 48, 10240)
    );
    assert_eq!(c.n_nextn, 1);
    assert_eq!(c.n_attention_layers(), 16);
    let w = Weights::load(&gguf, &c).unwrap();
    assert!(w.output.is_some());
    let nextn = w.nextn.expect("the file carries the next-token block");
    assert_eq!(nextn.eh_proj.cols(), 2 * 5120);
    assert!(nextn.shared_head_norm.is_some());
    assert!(nextn.embed_tokens.is_none() && nextn.shared_head_head.is_none());
}

/// llama.cpp quantizes the activations that meet quantized weights and keeps
/// K and V in f16, so the two do not agree to rounding: the bounds here are
/// the measured differences (`docs/reference/qwen35-architecture.md`) with a
/// margin, which still catch any difference in the computation itself.
#[test]
fn the_small_model_stays_within_the_measured_distance_of_llama_cpp() {
    let (Some(model), Some(captures)) =
        (env_path("ARES_TEST_QWEN35_MODEL"), env_path("ARES_TEST_QWEN35_CAPTURE"))
    else {
        return;
    };
    let gguf = Gguf::open(model).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);

    for prompt_name in ["english", "german", "code"] {
        let dir = captures.join(prompt_name).join("base");
        let prompt = read_ids(&dir.join("prompt.tokens"));
        assert!((100..=300).contains(&prompt.len()), "{prompt_name}: {} tokens", prompt.len());
        let llama_tokens = read_ids(&dir.join("generated.tokens"));
        let llama_logits = read_f32(&dir.join("steps.logits.f32"));

        let mut state = SequenceState::new(&c);
        let window = reference.forward(&mut state, &prompt, LogitRows::Last).unwrap();
        let last = prompt.len() - 1;

        // The chain: three blocks, all rows, and the logits of the last row.
        for block in [0, 11, 23] {
            let theirs = read_f32(&dir.join(format!("prefill.l_out-{block}.f32")));
            let err = rel(&window.hidden[block], &theirs);
            assert!(err < 8e-2, "{prompt_name}: block {block} differs by {err:e}");
        }
        let logits = window.logits_of(last).unwrap();
        let err = rel(logits, &read_f32(&dir.join("prefill.logits_last.f32")));
        assert!(err < 8e-2, "{prompt_name}: logits of the last prompt token differ by {err:e}");

        // One block alone, fed what llama.cpp fed it.
        let empty = SequenceState::new(&c);
        for block in [3, 12] {
            let input = read_f32(&dir.join(format!("prefill.l_out-{}.f32", block - 1)));
            let out = reference.block(block, &empty, &input);
            let err = rel(&out, &read_f32(&dir.join(format!("prefill.l_out-{block}.f32"))));
            assert!(err < 4e-2, "{prompt_name}: block {block} alone differs by {err:e}");
        }

        // Greedy generation of the reference itself. A token may differ from
        // llama.cpp's only where llama.cpp's own two best logits are closer than
        // the noise measured between two runs of llama.cpp (0.1).
        let n_vocab = c.n_vocab;
        let mut next = logits.to_vec();
        for (step, &theirs) in llama_tokens.iter().enumerate() {
            let (pick, _) = argmax(&next);
            if pick as u32 != theirs {
                let (_, margin) = argmax(&llama_logits[step * n_vocab..(step + 1) * n_vocab]);
                assert!(
                    margin < 0.1,
                    "{prompt_name}: step {step}: token {pick} against {theirs}, llama.cpp's margin is {margin}"
                );
                break;
            }
            next = reference.forward(&mut state, &[pick as u32], LogitRows::Last).unwrap().logits;
        }
    }
}
