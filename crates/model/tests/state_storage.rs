//! What it costs in error to hold the delta-rule matrices of the recurrent
//! blocks in a smaller type between steps, over a long sequence on the real
//! 0.8B. The arithmetic of a step is the reference's (f32 storage of results,
//! f64 sums) in every case; after each commit the matrices are rounded to the
//! format (`Reference::state_storage`), which is exactly what a state held in
//! that format does. The exact run keeps them in f32.
//!
//! Needs `ARES_TEST_QWEN35_MODEL` (Qwen3.5-0.8B-Q4_K_M.gguf) and
//! `ARES_TEST_QWEN35_CAPTURE` (the directory of the model crate's capture tool,
//! for the token ids of three real prompts) and reports itself skipped without
//! them. It is slow (the CPU reference decodes every weight at every call), so it
//! runs only with `ARES_TEST_STATE_STORAGE=1`; `ARES_STATE_TOKENS` sets the length
//! of the sequence (default 1000). Results are printed as `ERR` lines.
//!
//! ```sh
//! ARES_TEST_STATE_STORAGE=1 ARES_TEST_QWEN35_MODEL=... ARES_TEST_QWEN35_CAPTURE=/var/tmp/ares-model-capture \
//!     cargo test --release -p ares-model --test state_storage -- --nocapture
//! ```

use std::path::PathBuf;

use ares_gguf::Gguf;
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, StateStorage, Weights};

fn ids(path: PathBuf) -> Vec<u32> {
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (f64::from(*x) - f64::from(*y)).powi(2);
        n += f64::from(*y).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
    let sum: f64 = x.iter().map(|v| f64::from(v - max).exp()).sum();
    let log_sum = sum.ln() + f64::from(max);
    x.iter().map(|v| f64::from(*v) - log_sum).collect()
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).fold(0, |b, i| if x[i] > x[b] { i } else { b })
}

#[derive(Default, Clone)]
struct Bucket {
    steps: usize,
    hidden: f64,
    hidden_max: f64,
    logit_samples: usize,
    logits: f64,
    kl: f64,
    kl_max: f64,
    agree: usize,
}

#[test]
fn state_storage_error_over_a_long_sequence() {
    if std::env::var_os("ARES_TEST_STATE_STORAGE").is_none() {
        eprintln!("skipped: set ARES_TEST_STATE_STORAGE=1 (slow)");
        return;
    }
    let (Some(model), Some(capture)) =
        (std::env::var_os("ARES_TEST_QWEN35_MODEL"), std::env::var_os("ARES_TEST_QWEN35_CAPTURE"))
    else {
        eprintln!("skipped: set ARES_TEST_QWEN35_MODEL and ARES_TEST_QWEN35_CAPTURE");
        return;
    };
    let capture = PathBuf::from(capture);
    let total: usize =
        std::env::var("ARES_STATE_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let rows_per_step: usize =
        std::env::var("ARES_STATE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut text = Vec::new();
    for name in ["english", "german", "code"] {
        let dir = capture.join(name).join("base");
        text.extend(ids(dir.join("prompt.tokens")));
        text.extend(ids(dir.join("generated.tokens")));
    }
    let tokens: Vec<u32> = text.iter().copied().cycle().take(total).collect();

    let gguf = Gguf::open(model).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let formats = [
        ("f16", StateStorage::F16),
        ("bf16", StateStorage::Bf16),
        ("int8_per_head", StateStorage::Int8PerHead),
        ("int8_per_row", StateStorage::Int8PerRow),
    ];
    // "step": the state is rounded to the format after every commit, as a live state held in
    // that format is. "checkpoint": the state is exact and rounded once every
    // `ARES_STATE_CKPT` tokens, as a state that is saved and resumed from a checkpoint is.
    let checkpoint_mode = std::env::var("ARES_STATE_MODE").is_ok_and(|m| m == "checkpoint");
    let ckpt: usize =
        std::env::var("ARES_STATE_CKPT").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    let exact = Reference::new(&c, &w);
    let refs: Vec<Reference<'_, '_>> = formats
        .iter()
        .map(|(_, f)| {
            if checkpoint_mode {
                Reference::new(&c, &w)
            } else {
                Reference::new(&c, &w).state_storage(*f)
            }
        })
        .collect();
    let mut exact_state = SequenceState::new(&c);
    let mut states: Vec<SequenceState> = formats.iter().map(|_| SequenceState::new(&c)).collect();
    let bucket_len = 250usize;
    let mut buckets: Vec<Vec<Bucket>> =
        formats.iter().map(|_| vec![Bucket::default(); total.div_ceil(bucket_len)]).collect();

    let mut pos = 0usize;
    let mut step = 0usize;
    while pos < total {
        let end = (pos + rows_per_step).min(total);
        let chunk = &tokens[pos..end];
        // Logits are the expensive part (the head decodes 254 M values): every 20th step.
        let want_logits = step % 20 == 19;
        let kind = if want_logits { LogitRows::Last } else { LogitRows::None };
        let a = exact.forward(&mut exact_state, chunk, kind).unwrap();
        let last = chunk.len() - 1;
        let a_hidden = &a.final_norm[last * c.n_embd..(last + 1) * c.n_embd];
        for (i, reference) in refs.iter().enumerate() {
            let b = reference.forward(&mut states[i], chunk, kind).unwrap();
            let b_hidden = &b.final_norm[last * c.n_embd..(last + 1) * c.n_embd];
            let bucket = &mut buckets[i][pos / bucket_len];
            let e = rel_l2(b_hidden, a_hidden);
            bucket.steps += 1;
            bucket.hidden += e;
            bucket.hidden_max = bucket.hidden_max.max(e);
            if want_logits {
                let (la, lb) = (a.logits_of(last).unwrap(), b.logits_of(last).unwrap());
                let (pa, pb) = (log_softmax(la), log_softmax(lb));
                let kl: f64 = pa.iter().zip(&pb).map(|(x, y)| x.exp() * (x - y)).sum();
                bucket.logit_samples += 1;
                bucket.logits += rel_l2(lb, la);
                bucket.kl += kl;
                bucket.kl_max = bucket.kl_max.max(kl);
                bucket.agree += usize::from(argmax(la) == argmax(lb));
            }
        }
        pos = end;
        step += 1;
        if checkpoint_mode && pos % ckpt < rows_per_step {
            for (i, (_, f)) in formats.iter().enumerate() {
                states[i].round_delta(&c, *f);
            }
        }
    }

    eprintln!(
        "ERR state-storage tokens={total} rows_per_step={rows_per_step} mode={}",
        if checkpoint_mode { format!("checkpoint every {ckpt}") } else { "every step".into() }
    );
    for (i, (name, _)) in formats.iter().enumerate() {
        for (b, bucket) in buckets[i].iter().enumerate() {
            if bucket.steps == 0 {
                continue;
            }
            let n = bucket.logit_samples.max(1) as f64;
            eprintln!(
                "ERR {name} tokens {:4}..{:4}: hidden rel_l2 mean {:.2e} max {:.2e}; logits rel_l2 {:.2e}, KL mean {:.2e} max {:.2e}, top-1 agree {}/{}",
                b * bucket_len,
                ((b + 1) * bucket_len).min(total),
                bucket.hidden / bucket.steps as f64,
                bucket.hidden_max,
                bucket.logits / n,
                bucket.kl / n,
                bucket.kl_max,
                bucket.agree,
                bucket.logit_samples,
            );
        }
        // How far the stored matrices ended apart, per recurrent block.
        let mut worst = 0.0f64;
        let mut mean = 0.0f64;
        let mut count = 0;
        for layer in 0..c.n_layer {
            if let (Some(a), Some(b)) = (exact_state.delta(layer), states[i].delta(layer)) {
                let e = rel_l2(b, a);
                worst = worst.max(e);
                mean += e;
                count += 1;
            }
        }
        eprintln!(
            "ERR {name} final matrices rel_l2 vs exact: mean {:.2e} max over blocks {:.2e}",
            mean / count as f64,
            worst
        );
    }
}
