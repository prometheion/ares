//! The configuration, the weight table and the reference forward pass of
//! `qwen35`, on small made-up models. Nothing here needs a model file.

mod common;

use ares_core::GgmlType;
use ares_gguf::Gguf;
use ares_model::qwen35::weights::Mixer;
use ares_model::qwen35::{Config, Error, LogitRows, Reference, SequenceState, Weights};
use common::{Spec, Synth, TempDir, naive};

/// Writes the model to a file, opens it and hands the pieces to `run`.
fn with_model<R>(
    builder: &ares_gguf::GgufBuilder,
    run: impl FnOnce(&Config, &Weights<'_>) -> R,
) -> R {
    let dir = TempDir::new("model");
    let gguf = Gguf::open(dir.file("model.gguf", &builder.build())).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    run(&config, &weights)
}

fn load_error(builder: &ares_gguf::GgufBuilder) -> Error {
    let dir = TempDir::new("error");
    let gguf = Gguf::open(dir.file("model.gguf", &builder.build())).unwrap();
    match Config::from_gguf(&gguf).and_then(|config| Weights::load(&gguf, &config).map(|_| ())) {
        Ok(()) => panic!("the model should have been refused"),
        Err(e) => e,
    }
}

fn tokens(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + 11) % vocab) as u32).collect()
}

fn max_relative(a: &[f32], b: &[f64]) -> f64 {
    let norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let diff = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - y).powi(2)).sum::<f64>().sqrt();
    diff / norm
}

// --- configuration ---------------------------------------------------------

#[test]
fn the_configuration_is_read_from_the_metadata() {
    let synth = Synth::new(Spec::default());
    with_model(&synth.builder(), |c, w| {
        assert_eq!((c.n_layer, c.n_embd, c.n_ff, c.n_vocab, c.n_ctx), (8, 64, 96, 128, 512));
        assert_eq!((c.attention.n_head, c.attention.n_head_kv, c.attention.head_dim), (4, 2, 32));
        assert_eq!(c.attention.rope.dim, 16);
        assert_eq!(c.attention.rope.sections, [3, 3, 2, 0]);
        assert_eq!(c.attention.rope.freq_base, 10_000_000.0);
        assert_eq!(c.recurrent.conv_kernel, 4);
        assert_eq!((c.recurrent.n_k_heads, c.recurrent.n_v_heads), (2, 6));
        assert_eq!((c.recurrent.head_k_dim, c.recurrent.head_v_dim), (16, 16));
        assert_eq!(
            (c.recurrent.key_dim(), c.recurrent.value_dim(), c.recurrent.conv_dim()),
            (32, 96, 160)
        );
        assert_eq!(c.rms_eps, 1e-6);
        assert_eq!(c.n_nextn, 0);
        // Blocks 3 and 7 are attention blocks.
        let kinds: Vec<bool> = (0..8).map(|l| c.is_attention(l)).collect();
        assert_eq!(kinds, [false, false, false, true, false, false, false, true]);
        assert_eq!(c.n_attention_layers(), 2);
        for (l, layer) in w.layers.iter().enumerate() {
            assert_eq!(matches!(layer.mixer, Mixer::Attention(_)), c.is_attention(l));
        }
        assert!(w.output.is_some());
    });
}

#[test]
fn a_model_without_an_output_head_uses_the_embeddings() {
    let synth = Synth::new(Spec { tied: true, ..Spec::default() });
    with_model(&synth.builder(), |_, w| {
        assert!(w.output.is_none());
        assert_eq!(w.head().rows(), 128);
    });
}

#[test]
fn the_next_token_block_is_part_of_the_table_but_not_of_the_model() {
    let synth = Synth::new(Spec { nextn: true, ..Spec::default() });
    with_model(&synth.builder(), |c, w| {
        assert_eq!((c.n_layer, c.n_nextn), (8, 1));
        assert_eq!(w.layers.len(), 8);
        let nextn = w.nextn.as_ref().expect("the block is in the file");
        assert_eq!(nextn.eh_proj.cols(), 128);
        assert!(nextn.shared_head_norm.is_some());
        assert!(nextn.embed_tokens.is_none() && nextn.shared_head_head.is_none());
    });
}

#[test]
fn the_next_token_block_is_required_when_the_metadata_announces_it() {
    let mut synth = Synth::new(Spec { nextn: true, ..Spec::default() });
    synth.tensors.retain(|t| t.name != "blk.8.nextn.eh_proj.weight");
    let e = load_error(&synth.builder());
    assert!(
        matches!(&e, Error::MissingTensor { name } if name == "blk.8.nextn.eh_proj.weight"),
        "{e}"
    );
}

#[test]
fn another_architecture_is_refused() {
    let dir = TempDir::new("arch");
    let bytes = ares_gguf::GgufBuilder::new().str("general.architecture", "llama").build();
    let gguf = Gguf::open(dir.file("m.gguf", &bytes)).unwrap();
    let e = Config::from_gguf(&gguf).unwrap_err();
    assert!(matches!(&e, Error::Architecture { found } if found == "llama"), "{e}");
}

#[test]
fn a_missing_key_is_named() {
    let synth = Synth::new(Spec::default());
    for key in [
        "block_count",
        "ssm.conv_kernel",
        "rope.dimension_sections",
        "attention.layer_norm_rms_epsilon",
    ] {
        let mut b = synth.metadata(Some(key));
        for t in &synth.tensors {
            b = b.tensor_f32(&t.name, &t.dims, &t.values);
        }
        let e = load_error(&b);
        assert!(
            matches!(&e, Error::MissingKey { key: k } if *k == format!("qwen35.{key}")),
            "{key}: {e}"
        );
    }
}

#[test]
fn a_key_of_the_architecture_that_is_not_known_is_refused() {
    let synth = Synth::new(Spec::default());
    let mut b = synth.metadata(None).f32("qwen35.rope.scaling.factor", 4.0);
    for t in &synth.tensors {
        b = b.tensor_f32(&t.name, &t.dims, &t.values);
    }
    let e = load_error(&b);
    assert!(matches!(&e, Error::UnknownKey { key } if key == "qwen35.rope.scaling.factor"), "{e}");
}

#[test]
fn values_that_do_not_describe_a_model_are_refused() {
    // Each case changes the spec so that one relation of the architecture breaks.
    let cases: Vec<(Spec, &str)> = vec![
        (Spec { n_head: 5, ..Spec::default() }, "cannot be grouped"),
        (Spec { rope_dim: 15, ..Spec::default() }, "must be even"),
        (Spec { sections: [3, 3, 3, 0], ..Spec::default() }, "add up to half"),
        (Spec { v_heads: 5, ..Spec::default() }, "multiple of ssm.group_count"),
    ];
    for (spec, what) in cases {
        let mut synth = Synth::new(Spec::default());
        synth.spec = spec;
        let mut b = synth.metadata(None);
        for t in &synth.tensors {
            b = b.tensor_f32(&t.name, &t.dims, &t.values);
        }
        let dir = TempDir::new("values");
        let gguf = Gguf::open(dir.file("m.gguf", &b.build())).unwrap();
        let e = Config::from_gguf(&gguf).unwrap_err();
        assert!(
            matches!(e, Error::Inconsistent { .. }) && e.to_string().contains(what),
            "{what}: {e}"
        );
    }
}

#[test]
fn a_tensor_with_other_dimensions_is_named() {
    let mut synth = Synth::new(Spec::default());
    let t = synth.tensors.iter_mut().find(|t| t.name == "blk.2.ffn_up.weight").unwrap();
    t.dims = vec![64, 95];
    t.values.truncate(64 * 95);
    let e = load_error(&synth.builder());
    match &e {
        Error::TensorShape { name, found, expected } => {
            assert_eq!(name, "blk.2.ffn_up.weight");
            assert_eq!(
                (found.as_slice(), expected.as_slice()),
                (&[64u64, 95][..], &[64u64, 96][..])
            );
        }
        other => panic!("{other}"),
    }
}

#[test]
fn a_missing_tensor_is_named() {
    let mut synth = Synth::new(Spec::default());
    synth.tensors.retain(|t| t.name != "blk.5.ssm_a");
    let e = load_error(&synth.builder());
    assert!(matches!(&e, Error::MissingTensor { name } if name == "blk.5.ssm_a"), "{e}");
}

#[test]
fn a_tensor_the_architecture_has_no_use_for_is_named() {
    let mut synth = Synth::new(Spec::default());
    synth.tensors.push(common::Tensor {
        name: "blk.0.attn_q.weight".into(),
        dims: vec![64, 64],
        values: vec![0.0; 64 * 64],
        ty: GgmlType::F32,
    });
    let e = load_error(&synth.builder());
    assert!(matches!(&e, Error::UnexpectedTensor { name } if name == "blk.0.attn_q.weight"), "{e}");
}

#[test]
fn a_vocabulary_that_disagrees_with_the_embedding_table_is_refused() {
    let mut synth = Synth::new(Spec::default());
    synth.listed_tokens = Some(100);
    let dir = TempDir::new("vocab");
    let gguf = Gguf::open(dir.file("m.gguf", &synth.builder().build())).unwrap();
    let e = Config::from_gguf(&gguf).unwrap_err();
    assert!(
        matches!(e, Error::Inconsistent { .. }) && e.to_string().contains("100 entries"),
        "{e}"
    );
}

// --- the forward pass ------------------------------------------------------

/// The reference against the independent implementation, for each way of
/// storing the matrices.
#[test]
fn every_block_matches_an_independent_implementation() {
    for matrix_type in [GgmlType::F32, GgmlType::Q8_0] {
        let synth = Synth::new(Spec { matrix_type, ..Spec::default() });
        let prompt = tokens(13, 128);
        let expected = naive(&synth, &prompt);
        with_model(&synth.builder(), |c, w| {
            let reference = Reference::new(c, w);
            let mut state = SequenceState::new(c);
            let window = reference.forward(&mut state, &prompt, LogitRows::All).unwrap();
            // The reference stores every intermediate result as f32, the
            // independent implementation computes in f64 from the same weights.
            let mut worst = 0.0f64;
            for block in 0..c.n_layer {
                for row in 0..prompt.len() {
                    let err =
                        max_relative(window.hidden_row(block, row), &expected.hidden[block][row]);
                    worst = worst.max(err);
                    assert!(
                        err < 2e-5,
                        "{matrix_type}: block {block} row {row}: relative error {err:e}"
                    );
                }
            }
            for row in 0..prompt.len() {
                let err = max_relative(window.logits_of(row).unwrap(), &expected.logits[row]);
                worst = worst.max(err);
                assert!(err < 2e-5, "{matrix_type}: logits of row {row}: relative error {err:e}");
            }
            eprintln!(
                "{matrix_type}: worst relative difference to the independent implementation {worst:e}"
            );
        });
    }
}

#[test]
fn a_model_with_the_other_head_layout_matches_too() {
    // One value head per key head (the 0.8B), one query head per key/value head, tied embeddings.
    let spec = Spec {
        v_heads: 2,
        n_head: 2,
        n_head_kv: 2,
        tied: true,
        n_layer: 4,
        conv_kernel: 3,
        ..Spec::default()
    };
    let synth = Synth::new(spec);
    let prompt = tokens(9, 128);
    let expected = naive(&synth, &prompt);
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let mut state = SequenceState::new(c);
        let window = reference.forward(&mut state, &prompt, LogitRows::Last).unwrap();
        let last = prompt.len() - 1;
        assert!(max_relative(window.logits_of(last).unwrap(), &expected.logits[last]) < 2e-5);
        for block in 0..c.n_layer {
            assert!(
                max_relative(window.hidden_row(block, last), &expected.hidden[block][last]) < 2e-5
            );
        }
    });
}

/// The same weights and prompt were run by llama.cpp-native (f8f346786) on the
/// CPU with `-fa off -ctk f32 -ctv f32`, which keeps every activation in f32
/// (`tools/capture/README.md`, "The made-up model"). The values below are what
/// it computed, so this test holds the reference to llama.cpp without needing
/// llama.cpp, for a model with three value heads per key head and grouped
/// query heads.
#[test]
fn the_made_up_model_gives_what_llama_cpp_computed_for_it() {
    let synth = Synth::new(Spec::default());
    let prompt = tokens(48, 128);
    let llama_first_logits = [
        4.256732e-01f32,
        1.250511e+00,
        -9.106012e-02,
        -3.592424e-02,
        5.6892e-01,
        1.14566e+00,
        1.099439e+00,
        -4.469521e-01,
    ];
    let llama_hidden_last_row =
        [-4.441704e-01f32, -1.930968e+00, -1.899117e+00, 2.654997e+00, -1.015235e+00, 3.867682e+00];
    let llama_tokens = [
        85u32, 2, 52, 69, 33, 34, 24, 1, 28, 70, 34, 78, 7, 98, 30, 27, 123, 6, 8, 52, 118, 13, 70,
        28, 99, 78, 95, 31, 58, 53, 54, 49, 56, 11, 18, 11, 28, 28, 88, 11,
    ];
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let mut state = SequenceState::new(c);
        let window = reference.forward(&mut state, &prompt, LogitRows::Last).unwrap();
        let logits = window.logits_of(47).unwrap();
        for (i, (&mine, &theirs)) in logits.iter().zip(&llama_first_logits).enumerate() {
            assert!((mine - theirs).abs() < 1e-4, "logit {i}: {mine} against {theirs}");
        }
        let hidden = window.hidden_row(7, 47);
        for (i, (&mine, &theirs)) in hidden.iter().zip(&llama_hidden_last_row).enumerate() {
            assert!((mine - theirs).abs() < 1e-4, "hidden value {i}: {mine} against {theirs}");
        }
        // 40 greedy tokens, each chosen from the reference's own logits.
        let mut next = logits.to_vec();
        for (step, &expected) in llama_tokens.iter().enumerate() {
            let pick = next.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            assert_eq!(pick, expected, "greedy token {step}");
            next = reference.forward(&mut state, &[pick], LogitRows::Last).unwrap().logits;
        }
    });
}

#[test]
fn a_batch_of_rows_gives_the_bits_of_the_rows_one_at_a_time() {
    let synth = Synth::new(Spec::default());
    let prompt = tokens(17, 128);
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let mut batch_state = SequenceState::new(c);
        let batch = reference.forward(&mut batch_state, &prompt, LogitRows::All).unwrap();

        let mut state = SequenceState::new(c);
        for (row, &token) in prompt.iter().enumerate() {
            let one = reference.forward(&mut state, &[token], LogitRows::Last).unwrap();
            for block in 0..c.n_layer {
                assert_eq!(
                    one.hidden_row(block, 0),
                    batch.hidden_row(block, row),
                    "block {block} row {row}"
                );
            }
            assert_eq!(
                one.logits_of(0).unwrap(),
                batch.logits_of(row).unwrap(),
                "logits of row {row}"
            );
        }
        assert_eq!(
            state, batch_state,
            "the state after the rows must not depend on how they were batched"
        );
    });
}

#[test]
fn splitting_a_prompt_anywhere_gives_the_same_bits() {
    let synth = Synth::new(Spec::default());
    let prompt = tokens(14, 128);
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let mut whole_state = SequenceState::new(c);
        let whole = reference.forward(&mut whole_state, &prompt, LogitRows::Last).unwrap();
        for first in 1..prompt.len() {
            let mut state = SequenceState::new(c);
            reference.forward(&mut state, &prompt[..first], LogitRows::None).unwrap();
            let rest = reference.forward(&mut state, &prompt[first..], LogitRows::Last).unwrap();
            assert_eq!(
                rest.logits_of(rest.rows() - 1),
                whole.logits_of(whole.rows() - 1),
                "split at {first}"
            );
            assert_eq!(state, whole_state, "split at {first}");
        }
    });
}

#[test]
fn rejected_rows_leave_no_trace() {
    let synth = Synth::new(Spec::default());
    let prompt = tokens(6, 128);
    let draft = [5u32, 17, 99, 3, 42];
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let mut committed = SequenceState::new(c);
        reference.forward(&mut committed, &prompt, LogitRows::None).unwrap();
        let before = committed.clone();

        // Verify five draft rows from the committed state; the state stays as it was.
        let window = reference.window(&committed, &draft, LogitRows::All).unwrap();
        assert_eq!(committed, before);

        // Accept two. The state must be the one of a sequence that only ever saw those two.
        reference.commit(&mut committed, &window, 2).unwrap();
        let mut only = before.clone();
        reference.forward(&mut only, &draft[..2], LogitRows::None).unwrap();
        assert_eq!(committed, only);
        assert_eq!(committed.position(), prompt.len() + 2);

        // And the next token gives the same bits as in a sequence without the rejected rows.
        let next = reference.forward(&mut committed, &[draft[2]], LogitRows::Last).unwrap();
        let reject_free = reference.forward(&mut only, &[draft[2]], LogitRows::Last).unwrap();
        assert_eq!(next.logits, reject_free.logits);
        // The accepted rows of the window computed what they would have alone.
        assert_eq!(window.hidden_row(7, 1), {
            let mut s = before.clone();
            reference
                .forward(&mut s, &draft[..2], LogitRows::None)
                .unwrap()
                .hidden_row(7, 1)
                .to_vec()
        });
    });
}

#[test]
fn a_window_that_is_committed_whole_equals_forward() {
    let synth = Synth::new(Spec::default());
    let prompt = tokens(10, 128);
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let mut a = SequenceState::new(c);
        let window = reference.window(&a, &prompt, LogitRows::None).unwrap();
        reference.commit(&mut a, &window, prompt.len()).unwrap();
        let mut b = SequenceState::new(c);
        reference.forward(&mut b, &prompt, LogitRows::None).unwrap();
        assert_eq!(a, b);
    });
}

#[test]
fn a_hidden_state_does_not_depend_on_later_tokens() {
    let synth = Synth::new(Spec::default());
    let mut prompt = tokens(12, 128);
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let first =
            reference.forward(&mut SequenceState::new(c), &prompt, LogitRows::None).unwrap();
        prompt[9] = (prompt[9] + 1) % 128;
        let second =
            reference.forward(&mut SequenceState::new(c), &prompt, LogitRows::None).unwrap();
        for block in 0..c.n_layer {
            for row in 0..9 {
                assert_eq!(first.hidden_row(block, row), second.hidden_row(block, row));
            }
            assert_ne!(first.hidden_row(block, 9), second.hidden_row(block, 9));
        }
    });
}

#[test]
fn the_number_of_threads_does_not_change_a_bit() {
    let synth = Synth::new(Spec::default());
    let prompt = tokens(20, 128);
    with_model(&synth.builder(), |c, w| {
        let one = Reference::new(c, w).threads(1);
        let many = Reference::new(c, w).threads(7);
        let a = one.forward(&mut SequenceState::new(c), &prompt, LogitRows::All).unwrap();
        let b = many.forward(&mut SequenceState::new(c), &prompt, LogitRows::All).unwrap();
        assert_eq!(a.hidden, b.hidden);
        assert_eq!(a.logits, b.logits);
    });
}

#[test]
fn one_block_alone_gives_what_it_gives_inside_the_model() {
    let synth = Synth::new(Spec::default());
    let prompt = tokens(7, 128);
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let window = reference.window(&SequenceState::new(c), &prompt, LogitRows::None).unwrap();
        let empty = SequenceState::new(c);
        for block in 1..c.n_layer {
            let out = reference.block(block, &empty, &window.hidden[block - 1]);
            assert_eq!(out, window.hidden[block], "block {block}");
        }
    });
}

#[test]
fn the_state_holds_what_the_document_says() {
    let synth = Synth::new(Spec::default());
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let mut state = SequenceState::new(c);
        assert_eq!(state.position(), 0);
        reference.forward(&mut state, &tokens(5, 128), LogitRows::None).unwrap();
        assert_eq!(state.position(), 5);
        let r = &c.recurrent;
        for layer in 0..c.n_layer {
            if c.is_attention(layer) {
                let (k, v) = state.kv(layer).unwrap();
                let row = c.attention.n_head_kv * c.attention.head_dim;
                assert_eq!((k.len(), v.len()), (5 * row, 5 * row));
                assert!(state.conv(layer).is_none() && state.delta(layer).is_none());
            } else {
                assert_eq!(state.conv(layer).unwrap().len(), r.conv_dim() * (r.conv_kernel - 1));
                assert_eq!(
                    state.delta(layer).unwrap().len(),
                    r.n_v_heads * r.head_v_dim * r.head_k_dim
                );
                assert!(state.kv(layer).is_none());
                assert!(state.delta(layer).unwrap().iter().any(|&x| x != 0.0));
            }
        }
    });
}

#[test]
fn bad_requests_are_refused_with_what_is_wrong() {
    let synth = Synth::new(Spec::default());
    with_model(&synth.builder(), |c, w| {
        let reference = Reference::new(c, w);
        let state = SequenceState::new(c);
        assert!(matches!(reference.window(&state, &[], LogitRows::None), Err(Error::NoTokens)));
        assert!(matches!(
            reference.window(&state, &[3, 128], LogitRows::None),
            Err(Error::Token { token: 128, n_vocab: 128 })
        ));
        let long = vec![1u32; 513];
        assert!(matches!(
            reference.window(&state, &long, LogitRows::None),
            Err(Error::Context { end: 513, limit: 512 })
        ));

        let window = reference.window(&state, &[1, 2, 3], LogitRows::None).unwrap();
        let mut other = SequenceState::new(c);
        assert!(matches!(
            reference.commit(&mut other, &window, 4),
            Err(Error::Accepted { accepted: 4, rows: 3 })
        ));
        reference.forward(&mut other, &[9], LogitRows::None).unwrap();
        assert!(matches!(
            reference.commit(&mut other, &window, 1),
            Err(Error::WindowPosition { window: 0, state: 1 })
        ));
    });
}
