//! The device implementation of `qwen35` on one GPU against the CPU reference
//! and against what llama.cpp computed, on the real Qwen3.5-0.8B.
//!
//! Needs `ARES_TEST_CUDA=1`, `ARES_TEST_QWEN35_MODEL` (the 0.8B file) and
//! `ARES_TEST_QWEN35_CAPTURE` (the capture directory of `tools/capture`), and
//! says so when it skips. `ARES_TEST_DEVICE` picks the GPU (default 0). Lines
//! that start with `ERR` or `RES` are the measurements.

use std::path::Path;

use ares_cuda::{CacheFormat, Precision, Segment, StateFormat};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{Harness, Options};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
use common::gpu::gpu;
use common::helpers::{argmax, argmax_lead, env_path, read_f32, rel_l2, rel_max};

mod common;

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

fn read_i32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect()
}

fn options(precision: Precision) -> Options {
    Options { precision, max_rows: 256, logit_rows: 16, window_rows: 16, ..Options::default() }
}

fn direct(n: usize, pos0: usize) -> [Segment; 1] {
    [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: pos0 as i32, direct: true }]
}

/// Every block of the model alone, fed with the hidden state and the state
/// the reference has there: the error of one block's kernels.
#[test]
fn every_block_of_the_0_8b_matches_the_reference() {
    let (Some(device), Some(path), Some(capture)) =
        (gpu(), env_path("ARES_TEST_QWEN35_MODEL"), env_path("ARES_TEST_QWEN35_CAPTURE"))
    else {
        return;
    };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    for language in ["english", "german"] {
        let ids = read_ids(&capture.join(language).join("base/prompt.tokens"));
        let history = 40usize;
        let mut state = SequenceState::new(&c);
        reference.forward(&mut state, &ids[..history], LogitRows::None).unwrap();
        // Exact products with K and V as f16 show the attention kernel alone;
        // with q8, the default, the format's rounding is in the number too.
        for (precision, kv) in [
            (Precision::Exact, CacheFormat::F16),
            (Precision::Exact, CacheFormat::Q8),
            (Precision::Fast, CacheFormat::Q8),
        ] {
            let opts = Options { kv: (kv, kv), ..options(precision) };
            let mut harness = Harness::open(device, &c, &w, &opts, 1, 512).unwrap();
            // One row (decode kernels), eight (their limit), 21 (odd, tensor cores).
            for n in [1usize, 8, 21] {
                let tokens = &ids[history..history + n];
                let window = reference.window(&state, tokens, LogitRows::None).unwrap();
                let mut committed = state.clone();
                reference.commit(&mut committed, &window, n).unwrap();
                let (mut worst_r, mut worst_a, mut worst_state) = (0.0f64, 0.0f64, 0.0f64);
                for layer in 0..c.n_layer {
                    let x_in: Vec<f32> = if layer == 0 {
                        tokens
                            .iter()
                            .flat_map(|t| {
                                let mut row = vec![0.0f32; c.n_embd];
                                w.token_embd.decode_row(*t as usize, &mut row);
                                row
                            })
                            .collect()
                    } else {
                        window.hidden[layer - 1].clone()
                    };
                    let want = reference.block(layer, &state, &x_in);
                    if c.is_attention(layer) {
                        let (k, v) = state.kv(layer).unwrap();
                        harness.write_kv(layer, 0, history as u32, k, v).unwrap();
                    } else {
                        harness
                            .write_recurrent_state(
                                layer,
                                0,
                                state.conv(layer).unwrap(),
                                state.delta(layer).unwrap(),
                            )
                            .unwrap();
                    }
                    let got = harness.block(layer, &direct(n, history), &x_in).unwrap();
                    let e = rel_l2(&got, &want);
                    if c.is_attention(layer) {
                        worst_a = worst_a.max(e);
                    } else {
                        worst_r = worst_r.max(e);
                        let (conv, delta) = harness.read_recurrent_state(layer, 0).unwrap();
                        worst_state = worst_state
                            .max(rel_max(&conv, committed.conv(layer).unwrap()))
                            .max(rel_max(&delta, committed.delta(layer).unwrap()));
                    }
                    eprintln!(
                        "{language} block {layer:2} n={n:2} {precision:?} kv {kv:?}: output {e:.2e}"
                    );
                }
                eprintln!(
                    "ERR block {language} n={n} {precision:?} kv {kv:?} recurrent {worst_r:.2e} attention {worst_a:.2e} state {worst_state:.2e}"
                );
                // An attention block also carries the f16 operands of its
                // tensor-core kernel and, with q8, the format of K and V.
                let (limit, attention_limit, state_limit) = match precision {
                    Precision::Fast => (5e-2, 5e-2, 2e-2),
                    Precision::Exact => (5e-3, 2e-2, 2e-3),
                };
                assert!(worst_r < limit, "{precision:?} n={n}: recurrent block {worst_r}");
                assert!(
                    worst_a < attention_limit,
                    "{precision:?} n={n}: attention block {worst_a}"
                );
                assert!(worst_state < state_limit, "{precision:?} n={n}: state {worst_state}");
            }
        }
    }
}

/// Feeds `tokens` from position `pos0` in steps of `chunk` rows and returns
/// the arg-max of every row's logits and the logits of the last row.
fn feed(
    harness: &mut Harness,
    tokens: &[u32],
    pos0: usize,
    chunk: usize,
    all_rows: bool,
) -> (Vec<u32>, Vec<f32>) {
    let mut picks = Vec::new();
    let mut last = Vec::new();
    let mut at = 0;
    while at < tokens.len() {
        let n = chunk.min(tokens.len() - at);
        let out: Vec<u32> = if all_rows {
            (0..n as u32).collect()
        } else if at + n == tokens.len() {
            vec![n as u32 - 1]
        } else {
            Vec::new()
        };
        let logits = harness.step(&direct(n, pos0 + at), &tokens[at..at + n], &out).unwrap();
        if !out.is_empty() {
            let vocab = logits.len() / out.len();
            picks.extend(logits.chunks_exact(vocab).map(argmax));
            last = logits[logits.len() - vocab..].to_vec();
        }
        at += n;
    }
    (picks, last)
}

/// The whole forward pass over the captured prompts: top-1 agreement of every
/// row with the reference and with llama.cpp, the logits of the last row, and
/// 40 greedy tokens.
#[test]
fn the_forward_pass_agrees_with_the_reference_and_llama_cpp() {
    let (Some(device), Some(path), Some(capture)) =
        (gpu(), env_path("ARES_TEST_QWEN35_MODEL"), env_path("ARES_TEST_QWEN35_CAPTURE"))
    else {
        return;
    };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    for language in ["english", "german", "code"] {
        let dir = capture.join(language).join("base");
        let ids = read_ids(&dir.join("prompt.tokens"));
        let llama_picks = read_i32(&dir.join("prefill.argmax.i32"));
        let llama_last = read_f32(&dir.join("prefill.logits_last.f32"));
        let llama_generated = read_ids(&dir.join("generated.tokens"));
        let n = ids.len();

        // The reference: every row's pick, then 40 greedy tokens.
        let mut state = SequenceState::new(&c);
        let window = reference.forward(&mut state, &ids, LogitRows::All).unwrap();
        let ref_rows: Vec<(u32, f32)> =
            (0..n).map(|r| argmax_lead(window.logits_of(r).unwrap())).collect();
        let ref_last = window.logits_of(n - 1).unwrap().to_vec();
        let mut ref_generated = vec![ref_rows[n - 1].0];
        let mut ref_margins = vec![ref_rows[n - 1].1];
        while ref_generated.len() < 40 {
            let token = *ref_generated.last().unwrap();
            let window = reference.forward(&mut state, &[token], LogitRows::Last).unwrap();
            let (pick, margin) = argmax_lead(window.logits_of(0).unwrap());
            ref_generated.push(pick);
            ref_margins.push(margin);
        }

        for precision in [Precision::Exact, Precision::Fast] {
            for state_format in [StateFormat::F32, StateFormat::F16] {
                if state_format == StateFormat::F16 && precision == Precision::Exact {
                    continue;
                }
                let opts = Options { state: state_format, ..options(precision) };
                let mut harness = Harness::open(device, &c, &w, &opts, 1, 512).unwrap();
                // Rows in steps of 16, every row picked.
                let (picks, last) = feed(&mut harness, &ids, 0, 16, true);
                let agree_ref = picks.iter().zip(&ref_rows).filter(|(p, r)| **p == r.0).count();
                let agree_llama = picks.iter().zip(&llama_picks).filter(|(p, l)| p == l).count();
                // Where the device disagrees with the reference, how close the call was.
                let widest_miss = picks
                    .iter()
                    .zip(&ref_rows)
                    .filter(|(p, r)| **p != r.0)
                    .map(|(_, r)| r.1)
                    .fold(0.0f32, f32::max);
                let tag = format!("{language} {precision:?} state {state_format:?}");
                eprintln!(
                    "RES top1 {tag}: {agree_ref}/{n} with the reference (widest missed margin {widest_miss:.3}), {agree_llama}/{n} with llama.cpp; reference with llama.cpp {}/{n}",
                    ref_rows.iter().zip(&llama_picks).filter(|(r, l)| r.0 == **l).count()
                );
                eprintln!(
                    "ERR logits_last {tag}: {:.2e} from the reference, {:.2e} from llama.cpp (reference from llama.cpp {:.2e})",
                    rel_l2(&last, &ref_last),
                    rel_l2(&last, &llama_last),
                    rel_l2(&ref_last, &llama_last)
                );
                assert!(agree_ref * 100 >= n * 90, "{tag}: top-1 {agree_ref}/{n}");

                // 40 greedy tokens, one row per step.
                let mut generated = vec![argmax(&last)];
                while generated.len() < 40 {
                    let token = *generated.last().unwrap();
                    let position = n + generated.len() - 1;
                    let logits = harness.step(&direct(1, position), &[token], &[0]).unwrap();
                    generated.push(argmax(&logits));
                }
                let same_ref =
                    generated.iter().zip(&ref_generated).take_while(|(a, b)| a == b).count();
                let same_llama =
                    generated.iter().zip(&llama_generated).take_while(|(a, b)| a == b).count();
                let margin = ref_margins.get(same_ref).copied().unwrap_or(f32::NAN);
                eprintln!(
                    "RES greedy {tag}: {same_ref}/40 tokens equal the reference's (its margin at the first difference: {margin:.3}), {same_llama}/{} llama.cpp's",
                    llama_generated.len().min(40)
                );

                // The same prompt as one wide step: only the last row's logits.
                let mut wide = Harness::open(device, &c, &w, &opts, 1, 512).unwrap();
                let (wide_pick, wide_last) = feed(&mut wide, &ids, 0, 256, false);
                eprintln!(
                    "ERR logits_last one step of {n} rows {tag}: {:.2e} from the reference, pick {} (reference {})",
                    rel_l2(&wide_last, &ref_last),
                    wide_pick[0],
                    ref_rows[n - 1].0
                );
            }
        }
    }
}

/// A draft window whose rows are partly rejected, then its commit: the
/// sequence goes on as if only the accepted rows had ever run.
#[test]
fn a_draft_window_and_its_commit_equal_the_direct_steps() {
    let (Some(device), Some(path), Some(capture)) =
        (gpu(), env_path("ARES_TEST_QWEN35_MODEL"), env_path("ARES_TEST_QWEN35_CAPTURE"))
    else {
        return;
    };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let ids = read_ids(&capture.join("german/base/prompt.tokens"));
    let prompt = 60usize;
    for precision in [Precision::Exact, Precision::Fast] {
        let opts = options(precision);
        // The direct run: the prompt, then eight more tokens one at a time.
        let mut plain = Harness::open(device, &c, &w, &opts, 1, 512).unwrap();
        feed(&mut plain, &ids[..prompt], 0, 64, false);
        let mut want = Vec::new();
        for i in 0..8 {
            want.push(plain.step(&direct(1, prompt + i), &[ids[prompt + i]], &[0]).unwrap());
        }
        // With drafts: a window of six rows of which three are accepted (the
        // rejected rows carry wrong tokens), then a window of five, all accepted.
        let mut drafted = Harness::open(device, &c, &w, &opts, 1, 512).unwrap();
        feed(&mut drafted, &ids[..prompt], 0, 64, false);
        let mut first: Vec<u32> = ids[prompt..prompt + 3].to_vec();
        first.extend([11u32, 12, 13]);
        let window = [Segment { row0: 0, n_rows: 6, slot: 0, pos0: prompt as i32, direct: false }];
        let rows: Vec<u32> = (0..6).collect();
        let got = drafted.step(&window, &first, &rows).unwrap();
        drafted.commit(&[3]).unwrap();
        let vocab = got.len() / 6;
        let mut worst = 0.0f64;
        for i in 0..3 {
            worst = worst.max(rel_l2(&got[i * vocab..(i + 1) * vocab], &want[i]));
        }
        let second =
            [Segment { row0: 0, n_rows: 5, slot: 0, pos0: prompt as i32 + 3, direct: false }];
        let rows: Vec<u32> = (0..5).collect();
        let got = drafted.step(&second, &ids[prompt + 3..prompt + 8], &rows).unwrap();
        drafted.commit(&[5]).unwrap();
        for i in 0..5 {
            worst = worst.max(rel_l2(&got[i * vocab..(i + 1) * vocab], &want[3 + i]));
        }
        // And the state after both commits is the state of the direct run.
        let mut worst_state = 0.0f64;
        for layer in (0..c.n_layer).filter(|l| !c.is_attention(*l)) {
            let (conv_a, delta_a) = plain.read_recurrent_state(layer, 0).unwrap();
            let (conv_b, delta_b) = drafted.read_recurrent_state(layer, 0).unwrap();
            worst_state =
                worst_state.max(rel_max(&conv_b, &conv_a)).max(rel_max(&delta_b, &delta_a));
        }
        eprintln!(
            "ERR drafts {precision:?}: logits {worst:.2e}, state {worst_state:.2e} from the direct run"
        );
        let limit = match precision {
            Precision::Fast => 5e-2,
            Precision::Exact => 5e-3,
        };
        assert!(worst < limit && worst_state < limit, "{precision:?}: {worst} {worst_state}");
    }
}
