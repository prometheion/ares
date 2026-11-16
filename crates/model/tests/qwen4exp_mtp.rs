//! The MTP block of `qwen4exp` in the CPU reference, on a small made-up
//! model and a made-up block for it: no model file is needed.
//!
//! What is tested is the block against itself (a row has the same bits alone,
//! in a window and after a commit; a chain is its links one by one and leaves
//! no trace) and its input against a naive transcription of section 4 of
//! `docs/reference/qwen4exp-architecture.md`. What the block of the real model
//! guesses is measured in `qwen4exp_real.rs`.

use std::path::PathBuf;

use ares_core::GgmlType;
use ares_gguf::Gguf;
use ares_model::qwen4exp::synthetic::{Synthetic, Types};
use ares_model::qwen4exp::{
    Config, Error, LogitRows, MtpState, MtpWeights, Reference, SequenceState, Weights,
};

fn open(name: &str, bytes: &[u8]) -> Gguf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4-draft-{name}.gguf"));
    std::fs::write(&path, bytes).expect("the test directory is writable");
    Gguf::open(&path).expect("the writer makes a valid file")
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The trunk's residual after its last layer for `tokens`, `D` values a row.
fn trunk_out(reference: &Reference<'_, '_>, config: &Config, tokens: &[u32]) -> Vec<f32> {
    let state = SequenceState::new(config);
    reference.window(&state, tokens, LogitRows::None).expect("the trunk").trunk_out
}

#[test]
fn the_made_up_block_loads_as_described() {
    let model = Synthetic::small(11).types(Types::Shipped);
    let gguf = open("loads-model", &model.bytes());
    let file = open("loads", &model.draft_bytes(Some((GgmlType::Q8_0, GgmlType::F16))));
    let config = Config::from_gguf(&gguf).expect("config");
    let head = MtpWeights::load(&file, &config).expect("the block");
    let s = &model.shape;

    // Gate and up are the two halves of an expert's slab of the fused tensor.
    let fused = file.data(file.tensor("mtp.layers.0.mlp.experts.gate_up_proj").expect("tensor"));
    let per = fused.len() / s.n_expert;
    let moe = &head.layer.moe;
    assert_eq!((moe.gate.ty(), moe.down.ty()), (GgmlType::Q8_0, GgmlType::F16));
    assert_eq!(moe.gate.expert_bytes(), per / 2);
    assert_eq!(moe.gate.expert(5).bytes(), &fused[5 * per..5 * per + per / 2]);
    assert_eq!(moe.up.expert(5).bytes(), &fused[5 * per + per / 2..6 * per]);
    assert_eq!((moe.up.span(), moe.down.span().0), ((per / 2, per), 0));
    assert_eq!((moe.up.expert(0).rows(), moe.up.expert(0).cols()), (s.expert_ff, s.n_embd));

    // The indexer's projections are the rows of one tensor: queries, then the key.
    let ares_model::qwen4exp::weights::Mixer::Attention(a) = &head.layer.mixer else {
        panic!("the block is an attention layer");
    };
    assert_eq!(a.indexer_q.rows(), s.indexer_heads * s.indexer_dim);
    assert_eq!(a.indexer_k.rows(), s.indexer_dim);
    let both =
        file.data(file.tensor("mtp.layers.0.self_attn.indexer.index_qk_proj.weight").expect("t"));
    let row = both.len() / ((s.indexer_heads + 1) * s.indexer_dim);
    assert_eq!(a.indexer_k.bytes(), &both[s.indexer_heads * s.indexer_dim * row..]);

    // The file's norm weights are `w`; the table holds `1 + w`.
    let raw = file.data(file.tensor("mtp.pre_fc_norm_embedding.weight").expect("tensor"));
    let w0 = f32::from_le_bytes(raw[..4].try_into().expect("four bytes"));
    assert_eq!(head.norm_embedding[0], 1.0 + w0);
    assert!(head.output_mixer.inject.is_none() && head.layer.attn_mixer.inject.is_some());
    assert_eq!(head.norm_hidden.len(), config.wide());
}

#[test]
fn a_model_file_is_not_an_mtp_block_and_a_block_of_another_shape_is_refused() {
    let model = Synthetic::small(12);
    let gguf = open("refused-model", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let error = MtpWeights::load(&gguf, &config).expect_err("a model is no block");
    assert!(matches!(error, Error::NotAnMtpBlock { .. }), "{error}");

    let mut other = model.clone();
    other.shape.expert_ff *= 2;
    let file = open("refused-shape", &other.draft_bytes(None));
    let error = MtpWeights::load(&file, &config).expect_err("another shape");
    assert!(matches!(error, Error::TensorShape { .. }), "{error}");
}

#[test]
fn the_input_is_the_projected_streams_plus_the_projected_embedding() {
    let model = Synthetic::small(13);
    let gguf = open("input-model", &model.bytes());
    let file = open("input", &model.draft_bytes(None));
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let head = MtpWeights::load(&file, &config).expect("the block");
    let reference = Reference::new(&config, &weights).threads(2);
    let tokens = model.shape.tokens(4, 3);
    let rt = trunk_out(&reference, &config, &tokens[..3]);
    let input = reference.mtp_input(&head, &rt, &tokens[1..]);

    // The same in plain f64 sums, one value at a time.
    let (n, wide) = (config.n_embd, config.wide());
    let rms = |x: &[f32]| {
        (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64
            + f64::from(config.rms_eps))
        .sqrt()
    };
    let product = |w: &ares_model::qwen4exp::weights::Matrix<'_>, x: &[f64]| -> Vec<f64> {
        let mut row = vec![0.0f32; w.cols()];
        (0..w.rows())
            .map(|r| {
                w.decode_row(r, &mut row);
                row.iter().zip(x).map(|(a, b)| f64::from(*a) * b).sum()
            })
            .collect()
    };
    let mut worst = 0.0f64;
    for t in 0..3 {
        let mut emb = vec![0.0f32; n];
        weights.token_embd.decode_row(tokens[t + 1] as usize, &mut emb);
        let scale = rms(&emb);
        let en: Vec<f64> = emb
            .iter()
            .zip(&head.norm_embedding)
            .map(|(x, w)| f64::from(*x) / scale * f64::from(*w))
            .collect();
        let e = product(&head.fc_embedding, &en);
        let row = &rt[t * wide..(t + 1) * wide];
        for c in 0..config.n_stream {
            // Every stream is normed by itself.
            let scale = rms(&row[c * n..(c + 1) * n]);
            let hn: Vec<f64> = (0..n)
                .map(|i| f64::from(row[c * n + i]) / scale * f64::from(head.norm_hidden[c * n + i]))
                .collect();
            let h = product(&head.fc_hidden, &hn);
            for i in 0..n {
                let got = f64::from(input[t * wide + c * n + i]);
                worst = worst.max((got - (h[i] + e[i])).abs());
            }
        }
    }
    assert!(worst < 2e-5, "the input is {worst} from the transcription");
}

#[test]
fn a_row_has_the_same_bits_alone_in_a_window_and_after_a_commit() {
    let model = Synthetic::small(14).types(Types::Shipped);
    let gguf = open("rows-model", &model.bytes());
    let file = open("rows", &model.draft_bytes(None));
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let head = MtpWeights::load(&file, &config).expect("the block");
    let reference = Reference::new(&config, &weights).threads(2);
    // Past the dense limit of the made-up model (11 cells), so the block's
    // own indexer selects.
    let n = 19;
    let tokens = model.shape.tokens(n + 1, 5);
    let rt = trunk_out(&reference, &config, &tokens[..n]);
    let wide = config.wide();

    let whole = reference
        .draft_window(&head, &MtpState::new(), &rt, &tokens[1..], LogitRows::All)
        .expect("one window");
    assert_eq!(whole.rows(), n);
    assert!(whole.logits.iter().all(|v| v.is_finite()));

    // Row by row, each committed; and in two windows with a rejected tail.
    let mut state = MtpState::new();
    for t in 0..n {
        let row = reference
            .draft_window(
                &head,
                &state,
                &rt[t * wide..(t + 1) * wide],
                &tokens[t + 1..t + 2],
                LogitRows::All,
            )
            .expect("a row");
        assert_eq!(bits(&row.out), bits(&whole.out[t * wide..(t + 1) * wide]), "row {t}");
        assert_eq!(bits(row.logits_of(0).unwrap()), bits(whole.logits_of(t).unwrap()));
        assert_eq!(row.draft(0), whole.draft(t));
        reference.mtp_commit(&mut state, &row, 1).expect("commit");
    }
    let mut split = MtpState::new();
    let first = reference
        .draft_window(&head, &split, &rt[..9 * wide], &tokens[1..10], LogitRows::None)
        .expect("nine rows");
    // Six of the nine count; the other three leave no trace.
    reference.mtp_commit(&mut split, &first, 6).expect("commit");
    assert_eq!(split.position(), 6);
    let rest = reference
        .draft_window(&head, &split, &rt[6 * wide..], &tokens[7..], LogitRows::All)
        .expect("the rest");
    assert_eq!(bits(&rest.out), bits(&whole.out[6 * wide..]));
    reference.mtp_commit(&mut split, &rest, n - 6).expect("commit");
    assert_eq!(split, state);
    assert_eq!(state.kv().0.len(), n * config.attention_dims().1);

    assert!(matches!(
        reference.mtp_commit(&mut split, &rest, 1),
        Err(Error::WindowPosition { .. })
    ));
}

#[test]
fn a_chain_is_its_links_one_by_one_and_leaves_the_state() {
    let model = Synthetic::small(15);
    let gguf = open("chain-model", &model.bytes());
    let file = open("chain", &model.draft_bytes(None));
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let head = MtpWeights::load(&file, &config).expect("the block");
    let reference = Reference::new(&config, &weights).threads(2);
    let n = 6;
    let tokens = model.shape.tokens(n + 1, 8);
    let rt = trunk_out(&reference, &config, &tokens[..n]);
    let wide = config.wide();

    // The block has followed the first five cells; the chain starts at the sixth.
    let mut state = MtpState::new();
    let followed = reference
        .draft_window(&head, &state, &rt[..5 * wide], &tokens[1..6], LogitRows::None)
        .expect("follow");
    reference.mtp_commit(&mut state, &followed, 5).expect("commit");
    let before = state.clone();
    let drafts =
        reference.draft_chain(&head, &state, &rt[5 * wide..], tokens[6], 3).expect("a chain");
    assert_eq!(state, before, "a chain is provisional");
    assert_eq!(drafts.len(), 3);

    // By hand: each link starts from the residual of the one before and is
    // paired with its draft.
    let (mut r, mut token) = (rt[5 * wide..].to_vec(), tokens[6]);
    for (j, draft) in drafts.iter().enumerate() {
        let link =
            reference.draft_window(&head, &state, &r, &[token], LogitRows::All).expect("link");
        assert_eq!(link.start, 5 + j);
        assert_eq!(link.draft(0), Some(*draft), "link {j}");
        reference.mtp_commit(&mut state, &link, 1).expect("commit");
        (r, token) = (link.out, *draft);
    }
}
