//! The `ares` binary and a `qwen4exp` model file, as far as that needs no
//! GPU: the options that mean nothing for a mixture of experts, and the
//! numbers that cannot work, are refused by name before a device is touched.
//! `tests/real.rs` serves the real model.

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::LazyLock;

use ares_core::GgmlType;
use ares_model::qwen4exp::synthetic::{Shape, Synthetic, Types};

const BIN: &str = env!("CARGO_BIN_EXE_ares");

/// A made-up model of the architecture at sizes the kernels take.
fn made_up() -> Synthetic {
    let shape = Shape {
        n_embd: 256,
        mixer_rank: 32,
        n_head: 8,
        n_head_kv: 2,
        head_dim: 128,
        rope_dim: 64,
        rope_sections: [11, 11, 10, 0],
        indexer_heads: 3,
        indexer_dim: 128,
        n_k_heads: 2,
        n_v_heads: 4,
        state_size: 128,
        n_expert: 48,
        n_expert_used: 3,
        expert_ff: 64,
        shared_ff: 64,
        ngram: None,
        ..Shape::small()
    };
    Synthetic { shape, seed: 1, types: Types::Shipped, lazy_type: None }
}

/// A small model of the architecture, in a file of its own. The tests run at
/// once and share it, so it is written once.
fn model() -> &'static PathBuf {
    static PATH: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qwen4exp-binary");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("made.gguf");
        std::fs::write(&path, made_up().bytes_with_experts(GgmlType::IQ2_S, GgmlType::Q2_0))
            .unwrap();
        path
    });
    &PATH
}

fn ares(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(["--port", "0", "--model", model().to_str().unwrap()])
        .args(args)
        .env_remove("RUST_LOG")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

/// Runs `ares` with `args` and checks that it stopped with an error that
/// names `option`.
fn refuses(args: &[&str], option: &str) -> String {
    let output = ares(args);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(output.status.code(), Some(1), "{args:?}: {stderr}");
    assert!(stderr.contains(option), "{args:?} should name {option}: {stderr}");
    assert!(!stderr.contains("listening"), "{args:?}: nothing is served: {stderr}");
    stderr
}

#[test]
fn the_file_is_recognised_as_a_model_the_build_runs() {
    // The architecture is known: whatever stops the start, it is not that.
    let stderr = refuses(&["--shares", "0.5,0.5"], "--shares");
    assert!(stderr.contains(r#"architecture="qwen4exp""#), "{stderr}");
    assert!(!stderr.contains("has no implementation"), "{stderr}");
}

#[test]
fn options_that_mean_nothing_for_a_mixture_of_experts_are_refused_by_name() {
    let stderr = refuses(&["--shares", "0.5,0.5"], "--shares");
    assert!(stderr.contains("--split"), "it says what to use instead: {stderr}");
    refuses(&["--precision", "exact"], "--precision");
    refuses(&["--draft-vocab", "tokens.txt"], "--draft-vocab");
    let stderr = refuses(&["--kv", "q4"], "q4");
    assert!(stderr.contains("f16 or q8"), "{stderr}");
}

#[test]
fn rows_that_do_not_fit_a_step_are_refused_with_the_options_that_make_them() {
    let stderr = refuses(&["--parallel", "9", "--drafts", "0"], "--parallel");
    assert!(stderr.contains("8 rows"), "{stderr}");
    let stderr = refuses(&["--parallel", "4", "--drafts", "3"], "--parallel");
    assert!(stderr.contains("--drafts"), "{stderr}");
    assert!(stderr.contains("16 rows"), "{stderr}");
}
