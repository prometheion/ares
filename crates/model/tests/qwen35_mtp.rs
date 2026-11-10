//! The CPU reference of the next-token-prediction block (`reference/mtp.rs`).
//!
//! On a made-up model: the block's rows do not depend on how they are
//! batched, and a draft chain is the block run on its own output.
//!
//! On the real 27B against the fork (llama.cpp-native), on the same hidden
//! states: `ARES_TEST_QWEN35_27B` is the model file,
//! `ARES_TEST_QWEN35_CAPTURE27` a directory with `german/base` and
//! `english/base` made by `tools/capture` for it (tokens and the output of the
//! final norm of every position), and `ARES_TEST_QWEN35_MTP_CAPTURE` a
//! directory with `german` and `english` holding what the fork's MTP context
//! computed from those hidden states (`mtp.*` files; the program that writes
//! them is named in the test's skip message).

mod common;

use std::path::{Path, PathBuf};

use ares_gguf::Gguf;
use ares_model::qwen35::{Config, LogitRows, MtpState, Reference, SequenceState, Weights};
use common::{Spec, Synth, TempDir};

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (f64::from(*x) - f64::from(*y)).powi(2);
        n += f64::from(*y).powi(2);
    }
    (d / n).sqrt()
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
fn the_block_gives_the_same_rows_however_they_are_batched() {
    let synth = Synth::new(Spec { nextn: true, ..Spec::default() });
    let dir = TempDir::new("mtp");
    let gguf = Gguf::open(dir.file("model.gguf", &synth.builder().build())).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    let reference = Reference::new(&config, &weights).threads(2);
    let d = config.n_embd;

    // The model's hidden states for a prompt.
    let tokens: Vec<u32> = (0..12).map(|i| ((i * 37 + 11) % config.n_vocab) as u32).collect();
    let mut state = SequenceState::new(&config);
    let window = reference.forward(&mut state, &tokens, LogitRows::All).unwrap();
    // Row i of the block reads the hidden state of position i - 1.
    let mut hidden = vec![0.0f32; d];
    hidden.extend_from_slice(&window.final_norm[..(tokens.len() - 1) * d]);

    // All at once.
    let all = reference.mtp_rows(&MtpState::new(), &tokens, &hidden, true).unwrap();
    assert_eq!(all.rows(), tokens.len());
    // One at a time, in two parts, and in one part: the same bits.
    for parts in [vec![1; 12], vec![5, 7], vec![12]] {
        let mut mtp = MtpState::new();
        let mut at = 0;
        for n in parts {
            let rows = reference
                .mtp_rows(&mtp, &tokens[at..at + n], &hidden[at * d..(at + n) * d], true)
                .unwrap();
            assert_eq!(rows.hidden, all.hidden[at * d..(at + n) * d]);
            assert_eq!(
                rows.logits,
                all.logits[at * config.n_vocab..(at + n) * config.n_vocab],
                "rows {at}..{}",
                at + n
            );
            reference.mtp_commit(&mut mtp, &rows, n).unwrap();
            at += n;
        }
        assert_eq!(mtp.position(), tokens.len());
        assert_eq!(mtp.kv(), all.kv());
    }

    // Keys and values of a position depend on its own input only (its
    // token, its hidden state and the position), not on what came before.
    let other_tokens: Vec<u32> =
        tokens[..7].iter().map(|t| (t + 5) % config.n_vocab as u32).collect();
    let mut other = MtpState::new();
    let rows = reference.mtp_rows(&other, &other_tokens, &vec![0.25; 7 * d], false).unwrap();
    reference.mtp_commit(&mut other, &rows, 7).unwrap();
    let alone = reference.mtp_rows(&other, &tokens[7..8], &hidden[7 * d..8 * d], false).unwrap();
    let kv_dim = config.attention_dims().1;
    assert_eq!(alone.kv().0, &all.kv().0[7 * kv_dim..8 * kv_dim]);
    assert_eq!(alone.kv().1, &all.kv().1[7 * kv_dim..8 * kv_dim]);
    assert_ne!(alone.hidden, all.hidden[7 * d..8 * d], "the output does depend on the cache");

    // A chain: link 0 is the block on the newest token and the model's hidden
    // state, link j on the pick and the output of link j - 1.
    let mut mtp = MtpState::new();
    reference.mtp_commit(&mut mtp, &all, tokens.len()).unwrap();
    let newest = window.logits_of(tokens.len() - 1).map(argmax).unwrap();
    let h_last = &window.final_norm[(tokens.len() - 1) * d..];
    let chain = reference.mtp_draft(&mtp, newest, h_last, 3, None).unwrap();
    assert_eq!(chain.len(), 3);
    let mut by_hand = mtp.clone();
    let (mut token, mut h) = (newest, h_last.to_vec());
    for link in &chain {
        let rows = reference.mtp_rows(&by_hand, &[token], &h, true).unwrap();
        assert_eq!(rows.hidden, link.rows.hidden);
        assert_eq!(argmax(&rows.logits), link.token);
        assert_eq!(rows.logits[link.token as usize], link.logit);
        reference.mtp_commit(&mut by_hand, &rows, 1).unwrap();
        token = link.token;
        h = rows.hidden;
    }
    // The chain left the state alone.
    assert_eq!(mtp.position(), tokens.len());

    // A draft vocabulary restricts the picks.
    let even: Vec<u32> = (0..config.n_vocab as u32).filter(|t| t % 2 == 0).collect();
    let restricted = reference.mtp_draft(&mtp, newest, h_last, 3, Some(&even)).unwrap();
    assert!(restricted.iter().all(|link| link.token % 2 == 0));
    let first = &chain[0].rows.logits;
    let best_even = even
        .iter()
        .copied()
        .max_by(|a, b| first[*a as usize].total_cmp(&first[*b as usize]).then(b.cmp(a)));
    assert_eq!(Some(restricted[0].token), best_even);

    // A model without the block says so.
    let plain = Synth::new(Spec::default());
    let gguf = Gguf::open(dir.file("plain.gguf", &plain.builder().build())).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    let reference = Reference::new(&config, &weights);
    let zeros = vec![0.0; config.n_embd];
    assert!(reference.mtp_rows(&MtpState::new(), &[1], &zeros, false).is_err());
}

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(path) => Some(path.into()),
        None => {
            eprintln!(
                "skipped: set {name} to run this test (the fork's side is written by \
                 mtp_capture.cpp, see the notes of the MTP stage)"
            );
            None
        }
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn read_i32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect()
}

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

/// The largest and the mean relative distance of the rows of `a` from `b`.
fn rows_rel(a: &[f32], b: &[f32], width: usize) -> (f64, f64) {
    let per_row: Vec<f64> =
        a.chunks_exact(width).zip(b.chunks_exact(width)).map(|(x, y)| rel(x, y)).collect();
    let worst = per_row.iter().copied().fold(0.0, f64::max);
    (worst, per_row.iter().sum::<f64>() / per_row.len() as f64)
}

#[test]
fn the_block_of_the_27b_computes_what_the_fork_computes_on_the_same_hidden_states() {
    let Some(model) = env_path("ARES_TEST_QWEN35_27B") else { return };
    let Some(capture) = env_path("ARES_TEST_QWEN35_CAPTURE27") else { return };
    let Some(fork) = env_path("ARES_TEST_QWEN35_MTP_CAPTURE") else { return };
    let gguf = Gguf::open(&model).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    let reference = Reference::new(&config, &weights);
    let (d, n_vocab) = (config.n_embd, config.n_vocab);

    for prompt_name in ["german", "english"] {
        let base = capture.join(prompt_name).join("base");
        let theirs = fork.join(prompt_name);
        let prompt = read_ids(&base.join("prompt.tokens"));
        let generated = read_ids(&base.join("generated.tokens"));
        let h_prompt = read_f32(&base.join("prefill.result_norm.f32"));
        let h_steps = read_f32(&base.join("steps.result_norm.f32"));
        let n = prompt.len();
        assert_eq!(h_prompt.len(), n * d);

        // Every prompt position with the hidden state of the one before.
        let mut hidden = vec![0.0f32; d];
        hidden.extend_from_slice(&h_prompt[..(n - 1) * d]);
        let rows = reference.mtp_rows(&MtpState::new(), &prompt, &hidden, true).unwrap();
        let (worst, mean) =
            rows_rel(&rows.hidden, &read_f32(&theirs.join("mtp.prefill.h_nextn.f32")), d);
        let their_picks = read_i32(&theirs.join("mtp.prefill.argmax.i32"));
        let picks: Vec<u32> = rows.logits.chunks_exact(n_vocab).map(argmax).collect();
        let same = picks.iter().zip(&their_picks).filter(|(a, b)| a == b).count();
        let right = (0..n - 1).filter(|&i| picks[i] == prompt[i + 1]).count();
        let their_right = (0..n - 1).filter(|&i| their_picks[i] == prompt[i + 1]).count();
        let last = rel(
            &rows.logits[(n - 1) * n_vocab..],
            &read_f32(&theirs.join("mtp.prefill.logits_last.f32")),
        );
        println!(
            "{prompt_name}: prompt of {n}: block output rel L2 worst {worst:.2e}, mean {mean:.2e}; \
             top-1 equal to the fork's for {same} of {n}; equal to the token after next for \
             {right} (fork {their_right}) of {}; logits of the last row rel L2 {last:.2e}",
            n - 1
        );
        assert!(mean < 5e-2 && worst < 2e-1, "the block's output is not the fork's");
        assert!(same * 100 >= n * 90, "the picks are not the fork's");

        // A chain from the first generated token.
        let mut mtp = MtpState::new();
        reference.mtp_commit(&mut mtp, &rows, n).unwrap();
        let their_chain = read_i32(&theirs.join("mtp.chain.tokens.i32"));
        let their_logits = read_f32(&theirs.join("mtp.chain.logits.f32"));
        let their_hidden = read_f32(&theirs.join("mtp.chain.h_nextn.f32"));
        let chain = reference
            .mtp_draft(&mtp, generated[0], &h_prompt[(n - 1) * d..], their_chain.len(), None)
            .unwrap();
        let mut agree = true;
        for (j, link) in chain.iter().enumerate() {
            // Once a link picked another token the chains are different texts.
            let hidden = rel(&link.rows.hidden, &their_hidden[j * d..(j + 1) * d]);
            let logits = rel(&link.rows.logits, &their_logits[j * n_vocab..(j + 1) * n_vocab]);
            println!(
                "{prompt_name}: chain link {j}: token {} (fork {}), output rel L2 {hidden:.2e}, \
                 logits rel L2 {logits:.2e}{}",
                link.token,
                their_chain[j],
                if agree { "" } else { " (after a different pick)" }
            );
            if agree {
                assert!(hidden < 1e-1, "link {j} is not the fork's");
            }
            agree &= link.token == their_chain[j];
        }
        assert_eq!(chain[0].token, their_chain[0], "the first link's pick");

        // Every generated position with the model's hidden state.
        let their_picks = read_i32(&theirs.join("mtp.steps.argmax.i32"));
        let m = their_picks.len();
        let mut hidden = h_prompt[(n - 1) * d..].to_vec();
        hidden.extend_from_slice(&h_steps[..(m - 1) * d]);
        let rows = reference.mtp_rows(&mtp, &generated[..m], &hidden, true).unwrap();
        let (worst, mean) =
            rows_rel(&rows.hidden, &read_f32(&theirs.join("mtp.steps.h_nextn.f32")), d);
        let picks: Vec<u32> = rows.logits.chunks_exact(n_vocab).map(argmax).collect();
        let same = picks.iter().zip(&their_picks).filter(|(a, b)| a == b).count();
        let right = (0..m - 1).filter(|&i| picks[i] == generated[i + 1]).count();
        let their_right = (0..m - 1).filter(|&i| their_picks[i] == generated[i + 1]).count();
        println!(
            "{prompt_name}: {m} generated positions: block output rel L2 worst {worst:.2e}, mean \
             {mean:.2e}; top-1 equal to the fork's for {same}; equal to the model's next token for \
             {right} (fork {their_right}) of {}",
            m - 1
        );
        assert!(mean < 5e-2 && worst < 2e-1, "the block's output is not the fork's");
        assert!(same * 100 >= m * 85, "the picks are not the fork's");
    }
}
