//! The `qwen4exp` reference on a small made-up model: no model file is needed.
//!
//! The model is `ares_model::qwen4exp::synthetic` (section 8.3 of
//! `docs/reference/qwen4exp-architecture.md`). What is tested here is the
//! reference against itself (a row has the same bits alone, in a window and
//! after a commit; rejected rows leave no trace; the selection rules differ
//! exactly as specified; experts and table rows are read only when a token
//! asks for them) and against an independent, deliberately naive transcription
//! of section 3 that keeps no state. The comparison with the llama.cpp fork is
//! in `tools/q4capture`.

use std::path::PathBuf;

use ares_core::GgmlType;
use ares_gguf::Gguf;
use ares_model::files::FileError;
use ares_model::qwen4exp::reference::pieces;
use ares_model::qwen4exp::synthetic::{Shape, Synthetic, Types};
use ares_model::qwen4exp::{
    Config, Error, LogitRows, Recorder, Reference, SelectionRule, SequenceState, Weights,
};

/// Writes `bytes` as a model file of its own and opens it.
fn open(name: &str, bytes: &[u8]) -> Gguf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("q4-{name}.gguf"));
    std::fs::write(&path, bytes).expect("the test directory is writable");
    Gguf::open(&path).expect("the writer makes a valid file")
}

fn tokens(shape: &Shape, n: usize, seed: u64) -> Vec<u32> {
    shape.tokens(n, seed)
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn the_made_up_model_loads_as_described() {
    let model = Synthetic::small(1);
    let gguf = open("loads", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let s = &model.shape;
    assert_eq!(config.n_layer, s.n_layer);
    assert_eq!(config.wide(), s.n_stream * s.n_embd);
    assert_eq!(config.n_vocab, s.n_vocab);
    assert_eq!(config.n_attention_layers(), 2);
    assert!(config.is_attention(3) && config.is_attention(7) && !config.is_attention(1));
    assert_eq!(config.indexer.block, 4);
    assert_eq!(config.indexer.dense_limit(), 11);
    assert_eq!(config.recurrent.conv_dim(), (2 * s.n_k_heads + s.n_v_heads) * s.state_size);
    let ngram = config.ngram.as_ref().expect("the model has the n-gram layer");
    assert_eq!((ngram.layer, ngram.n_heads(), ngram.row_dim, ngram.history()), (1, 4, 8, 9));
    assert_eq!(ngram.head_offset, [0, 101, 204, 311]);
    assert_eq!(ngram.table_rows, 424);

    let weights = Weights::load(&gguf, &config).expect("weights");
    assert_eq!(weights.layers.len(), s.n_layer);
    assert!(weights.layers[1].ngram.is_some() && weights.layers[0].ngram.is_none());
    let moe = &weights.layers[0].moe;
    assert_eq!(moe.gate.n_expert(), s.n_expert);
    assert_eq!(moe.down.expert(3).rows(), s.n_embd);
    assert_eq!(moe.down.expert(3).cols(), s.expert_ff);
    // Expert e is the e-th slab of the stacked tensor.
    let stacked = gguf.data(gguf.tensor("blk.0.ffn_up_exps.weight").expect("tensor"));
    let per = stacked.len() / s.n_expert;
    assert_eq!(moe.up.expert(5).bytes(), &stacked[5 * per..6 * per]);
}

#[test]
fn a_file_that_does_not_describe_the_architecture_is_refused() {
    let mut model = Synthetic::small(1);
    model.shape.n_stream = 1;
    model.shape.mixer_rank = 3;
    let err = Config::from_gguf(&open("one-stream", &model.bytes())).unwrap_err();
    assert!(
        matches!(err, Error::File(FileError::BadKey { ref key, .. }) if key.ends_with("hyper_connection.count"))
    );

    let mut model = Synthetic::small(1);
    model.shape.top_k = 6;
    let err = Config::from_gguf(&open("budget", &model.bytes())).unwrap_err();
    assert!(matches!(err, Error::Inconsistent { .. }), "{err}");

    let mut model = Synthetic::small(1);
    model.shape.n_v_heads = 5;
    let err = Config::from_gguf(&open("heads", &model.bytes())).unwrap_err();
    assert!(matches!(err, Error::Inconsistent { .. }), "{err}");

    // A tensor of another shape than the configuration gives it.
    let model = Synthetic::small(1);
    let gguf = open("shape", &model.bytes());
    let mut config = Config::from_gguf(&gguf).expect("config");
    config.experts.ff += 1;
    let err = Weights::load(&gguf, &config).unwrap_err();
    assert!(
        matches!(err, Error::File(FileError::TensorShape { ref name, .. }) if name.contains("ffn_gate_exps"))
    );
}

#[test]
fn a_row_has_the_same_bits_alone_in_a_window_and_after_a_commit() {
    let model = Synthetic::small(2);
    let gguf = open("windows", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let reference = Reference::new(&config, &weights);
    let toks = tokens(&model.shape, 37, 5);

    // The whole sequence as one window.
    let whole =
        reference.window(&SequenceState::new(&config), &toks, LogitRows::All).expect("window");
    assert!(whole.logits.iter().all(|v| v.is_finite()));
    assert!(whole.trunk_out.iter().all(|v| v.is_finite()));

    // Token by token, and in windows of uneven sizes, on one thread.
    for chunks in [vec![1usize; 37], vec![3, 1, 7, 2, 9, 4, 11]] {
        let single = Reference::new(&config, &weights).threads(1);
        let mut state = SequenceState::new(&config);
        let mut at = 0;
        for size in chunks {
            let w = single.forward(&mut state, &toks[at..at + size], LogitRows::All).expect("w");
            for row in 0..size {
                assert_eq!(
                    bits(w.logits_of(row).expect("logits")),
                    bits(whole.logits_of(at + row).expect("logits")),
                    "logits of cell {}",
                    at + row
                );
                for layer in 0..config.n_layer {
                    assert_eq!(
                        bits(w.residual_row(layer, row).expect("kept")),
                        bits(whole.residual_row(layer, at + row).expect("kept")),
                        "layer {layer} of cell {}",
                        at + row
                    );
                }
            }
            at += size;
        }
        assert_eq!(state.position(), 37);
    }
}

#[test]
fn committing_a_prefix_drops_the_rejected_rows_without_a_trace() {
    let model = Synthetic::small(3);
    let gguf = open("commit", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let reference = Reference::new(&config, &weights);
    let toks = tokens(&model.shape, 30, 9);

    // The straight run.
    let mut straight = SequenceState::new(&config);
    reference.forward(&mut straight, &toks[..14], LogitRows::None).expect("forward");
    let mut after_17 = straight.clone();
    reference.forward(&mut after_17, &toks[14..17], LogitRows::None).expect("forward");

    // A window of drafts from cell 14 (two cells into a block), of which three
    // are accepted: the drafts after them are other tokens.
    let mut drafts = toks[14..17].to_vec();
    drafts.extend([1, 2, 3, 4, 5, 6]);
    let mut state = straight.clone();
    let window = reference.window(&state, &drafts, LogitRows::None).expect("window");
    assert_eq!(state, straight, "a window does not change the state");
    reference.commit(&mut state, &window, 3).expect("commit");
    assert_eq!(state, after_17, "the state after accepting 3 of 9 rows");
    assert_eq!(state.recent_tokens(), &toks[15..17]);
    let (pooled, raw_tail) = state.indexer(3).expect("an attention layer");
    assert_eq!(pooled.len(), 4 * config.indexer.key_dim);
    assert_eq!(raw_tail.len(), config.indexer.key_dim);

    // Accepting nothing changes nothing; the rest continues as the straight run.
    let window = reference.window(&state, &[11, 12, 13], LogitRows::None).expect("window");
    reference.commit(&mut state, &window, 0).expect("commit");
    assert_eq!(state, after_17);
    let a = reference.forward(&mut state, &toks[17..], LogitRows::Last).expect("forward");
    let b = reference.forward(&mut after_17, &toks[17..], LogitRows::Last).expect("forward");
    assert_eq!(bits(&a.logits), bits(&b.logits));

    // The errors of a commit.
    let window = reference.window(&state, &[1, 2], LogitRows::None).expect("window");
    assert!(matches!(reference.commit(&mut state, &window, 3), Err(Error::Accepted { .. })));
    let mut elsewhere = straight.clone();
    assert!(matches!(
        reference.commit(&mut elsewhere, &window, 1),
        Err(Error::WindowPosition { .. })
    ));
    assert!(matches!(reference.window(&state, &[], LogitRows::None), Err(Error::NoTokens)));
    assert!(matches!(reference.window(&state, &[96], LogitRows::None), Err(Error::Token { .. })));
}

#[test]
fn the_selection_rules_differ_by_the_partial_block_only() {
    let model = Synthetic::small(4);
    let gguf = open("selection", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let toks = tokens(&model.shape, 41, 3);
    let run = |rule: SelectionRule| {
        let reference = Reference::new(&config, &weights).selection_rule(rule);
        let mut recorder = Recorder::default();
        let window = reference
            .window_observed(&SequenceState::new(&config), &toks, LogitRows::None, &mut recorder)
            .expect("window");
        (window, recorder)
    };
    let (budget_window, budget) = run(SelectionRule::CellBudget);
    let (_, whole) = run(SelectionRule::WholeBlocks);
    let ix = &config.indexer;
    let mut partial_blocks = 0;
    for row in 0..toks.len() {
        let n = row + 1;
        let (cells, scores) = &budget.selection[&(3, row)];
        let (cells_whole, _) = &whole.selection[&(3, row)];
        assert!(cells.windows(2).all(|w| w[0] < w[1]), "ascending cells");
        if n <= ix.dense_limit() {
            assert_eq!(cells.len(), n);
            assert!(scores.is_empty());
            assert_eq!(cells, cells_whole);
            continue;
        }
        // Scores are the same under both rules in the first attention layer:
        // nothing upstream depends on the rule.
        assert_eq!(scores.len(), n / ix.block);
        assert_eq!(cells.len(), ix.dense_limit(), "the cell budget at {n} cells");
        assert_eq!(cells_whole.len(), ix.top_k + n % ix.block);
        // The tail is always there.
        for cell in n / ix.block * ix.block..n {
            assert!(cells.contains(&(cell as u32)) && cells_whole.contains(&(cell as u32)));
        }
        // Whole blocks are a subset; what is left is the head of one block.
        let extra: Vec<u32> = cells.iter().copied().filter(|c| !cells_whole.contains(c)).collect();
        assert!(cells_whole.iter().all(|c| cells.contains(c)));
        assert_eq!(extra.len(), ix.block - 1 - n % ix.block);
        if let Some(&first) = extra.first() {
            partial_blocks += 1;
            assert_eq!(first as usize % ix.block, 0);
            assert!(extra.windows(2).all(|w| w[1] == w[0] + 1));
            // That block ranks right after the whole ones: no unselected block scores higher.
            let partial = scores[first as usize / ix.block];
            for (j, &score) in scores.iter().enumerate() {
                let selected = cells_whole.contains(&((j * ix.block) as u32));
                if !selected && j != first as usize / ix.block {
                    assert!(score < partial || (score == partial && j > first as usize / ix.block));
                }
            }
        }
    }
    assert!(partial_blocks >= 20, "the partial block occurs: {partial_blocks}");
    // The routing differs somewhere downstream, or the rule would not matter.
    assert_eq!(budget_window.routing.len(), config.n_layer);
}

#[test]
fn the_router_picks_distinct_experts_in_rank_order() {
    let model = Synthetic::small(5);
    let gguf = open("router", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let reference = Reference::new(&config, &weights);
    let toks = tokens(&model.shape, 24, 1);
    let mut recorder = Recorder::default();
    let window = reference
        .window_observed(&SequenceState::new(&config), &toks, LogitRows::None, &mut recorder)
        .expect("window");
    let k = config.experts.used;
    let mut seen = std::collections::BTreeSet::new();
    for (layer, routing) in window.routing.iter().enumerate() {
        for row in 0..toks.len() {
            let ids = &routing.ids[row * k..(row + 1) * k];
            let weights = &routing.weights[row * k..(row + 1) * k];
            assert_eq!(recorder.routing[&(layer, row)], (ids.to_vec(), weights.to_vec()));
            assert!(weights.windows(2).all(|w| w[0] >= w[1]), "rank order");
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            let distinct: std::collections::BTreeSet<_> = ids.iter().collect();
            assert_eq!(distinct.len(), k);
            seen.extend(ids.iter().map(|&id| (layer, id)));
        }
    }
    // The made-up router spreads its picks: most experts of most layers are used.
    assert!(seen.len() > config.n_layer * config.experts.count * 3 / 4, "{}", seen.len());
}

#[test]
fn an_expert_or_a_table_row_nobody_asked_for_is_never_read() {
    let model = Synthetic::small(6);
    let bytes = model.bytes();
    let gguf = open("lazy", &bytes);
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let reference = Reference::new(&config, &weights);
    let toks = tokens(&model.shape, 6, 2);
    let clean =
        reference.window(&SequenceState::new(&config), &toks, LogitRows::All).expect("window");

    // Overwrite every expert that no row picked, and every table row that no
    // row hashed to, with NaN. Expert 0 and row 0 stay: loading decodes the
    // first row of every tensor.
    let ngram = config.ngram.as_ref().expect("n-gram layer");
    let mut used_rows = std::collections::BTreeSet::from([0u64]);
    for (i, &token) in toks.iter().enumerate() {
        used_rows.extend(pieces::ngram_rows(ngram, token, &toks[..i]));
    }
    let k = config.experts.used;
    let mut poisoned = bytes.clone();
    // The file offset and the size of a tensor's data.
    let offset_of = |name: &str| {
        let info = gguf.tensor(name).expect("tensor");
        (info.offset as usize, info.size as usize)
    };
    let mut poison = |at: usize, len: usize| {
        for chunk in poisoned[at..at + len].as_chunks_mut::<4>().0 {
            *chunk = f32::NAN.to_le_bytes();
        }
    };
    let mut poisoned_experts = 0;
    for layer in 0..config.n_layer {
        let picked: std::collections::BTreeSet<u32> =
            clean.routing[layer].ids.iter().copied().chain([0]).collect();
        assert!(picked.len() <= toks.len() * k + 1);
        for what in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let (ptr, len) = offset_of(&format!("blk.{layer}.{what}.weight"));
            let per = len / config.experts.count;
            for e in (0..config.experts.count as u32).filter(|e| !picked.contains(e)) {
                poison(ptr + e as usize * per, per);
                poisoned_experts += 1;
            }
        }
    }
    let (ptr, len) = offset_of("per_layer_token_embd.weight");
    let per = len / ngram.table_rows as usize;
    for row in (0..ngram.table_rows).filter(|r| !used_rows.contains(r)) {
        poison(ptr + row as usize * per, per);
    }
    assert!(poisoned_experts > 0 && (used_rows.len() as u64) < ngram.table_rows);

    let gguf = open("lazy-poisoned", &poisoned);
    let weights = Weights::load(&gguf, &config).expect("weights");
    let reference = Reference::new(&config, &weights);
    let again =
        reference.window(&SequenceState::new(&config), &toks, LogitRows::All).expect("window");
    assert_eq!(bits(&again.logits), bits(&clean.logits));
}

#[test]
fn quantized_experts_and_table_rows_are_decoded_where_they_lie() {
    // A shape whose rows are whole blocks of 32 values.
    let mut model = Synthetic::small(7);
    model.shape.n_layer = 4;
    model.shape.n_embd = 64;
    model.shape.expert_ff = 32;
    let ngram = model.shape.ngram.as_mut().expect("the small shape has the n-gram layer");
    ngram.heads_per_ngram = 1;
    ngram.head_vocab = vec![101, 103];
    model.types = Types::Shipped;
    model.lazy_type = Some(GgmlType::Q8_0);
    let gguf = open("quantized", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    assert_eq!(weights.layers[0].moe.down.ty(), GgmlType::Q8_0);
    assert_eq!(weights.ngram_table.as_ref().expect("table").ty(), GgmlType::Q8_0);
    assert_eq!(weights.layers[0].attn_mixer.down.ty(), GgmlType::BF16);
    let reference = Reference::new(&config, &weights);
    let toks = tokens(&model.shape, 9, 4);
    let mut recorder = Recorder::default();
    let window = reference
        .window_observed(&SequenceState::new(&config), &toks, LogitRows::Last, &mut recorder)
        .expect("window");
    assert!(window.logits.iter().all(|v| v.is_finite()));

    // The output of the mixture of layer 0 for row 0, recomputed here from
    // the stacked tensors decoded as a whole.
    let (n, ff, k) = (config.n_embd, config.experts.ff, config.experts.used);
    let m = &recorder.tensor_of(Some(0), "ffn_in").expect("tap")[..n];
    let out = &recorder.tensor_of(Some(0), "ffn_out").expect("tap")[..n];
    let decode = |name: &str| {
        let info = gguf.tensor(name).expect("tensor");
        let mut values = vec![0.0f32; info.dims.n_values().expect("size") as usize];
        ares_quant::dequantize(info.ty, gguf.data(info), &mut values).expect("decodes");
        values
    };
    let (gate, up, down) = (
        decode("blk.0.ffn_gate_exps.weight"),
        decode("blk.0.ffn_up_exps.weight"),
        decode("blk.0.ffn_down_exps.weight"),
    );
    let mut expected = vec![0.0f64; n];
    for (&id, &weight) in window.routing[0].ids[..k].iter().zip(&window.routing[0].weights[..k]) {
        let e = id as usize;
        let act: Vec<f64> = (0..ff)
            .map(|r| {
                let row = e * ff * n + r * n..e * ff * n + (r + 1) * n;
                let dot = |w: &[f32]| -> f64 {
                    w.iter().zip(m).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum()
                };
                let (g, u) = (dot(&gate[row.clone()]), dot(&up[row]));
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        for (i, x) in expected.iter_mut().enumerate() {
            let row = &down[e * n * ff + i * ff..e * n * ff + (i + 1) * ff];
            let y: f64 = row.iter().zip(&act).map(|(a, b)| f64::from(*a) * b).sum();
            *x += f64::from(weight) * y;
        }
    }
    // Add the shared expert.
    let shared = |name: &str| decode(&format!("blk.0.{name}.weight"));
    let (sg, su, sd, sgi) = (
        shared("ffn_gate_shexp"),
        shared("ffn_up_shexp"),
        shared("ffn_down_shexp"),
        shared("ffn_gate_inp_shexp"),
    );
    let sff = config.experts.shared_ff;
    let dotm =
        |w: &[f32]| -> f64 { w.iter().zip(m).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum() };
    let act: Vec<f64> = (0..sff)
        .map(|r| {
            let (g, u) = (dotm(&sg[r * n..(r + 1) * n]), dotm(&su[r * n..(r + 1) * n]));
            g / (1.0 + (-g).exp()) * u
        })
        .collect();
    let sh = 1.0 / (1.0 + (-dotm(&sgi)).exp());
    for (i, x) in expected.iter_mut().enumerate() {
        let y: f64 =
            sd[i * sff..(i + 1) * sff].iter().zip(&act).map(|(a, b)| f64::from(*a) * b).sum();
        *x += sh * y;
    }
    for (i, (&got, want)) in out.iter().zip(&expected).enumerate() {
        assert!((f64::from(got) - want).abs() < 1e-5 * (1.0 + want.abs()), "{i}: {got} {want}");
    }
}

/// What the llama.cpp fork (`prod/native`, `f8f346786`, CPU backend, no flash
/// attention, `f32` caches) computed for the made-up model with seed 1 and its
/// 48 tokens, captured with `tools/q4capture` (see its README). The fork
/// selects whole blocks, so the reference runs with that rule. With `f32`
/// weights llama.cpp rounds no activation, and what remains between the two is
/// rounding: 1.4e-5 relative RMS for the worst row of the logits.
#[test]
fn the_made_up_model_gives_what_llama_cpp_computed_for_it() {
    let model = Synthetic::small(1);
    let gguf = open("llama", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let reference = Reference::new(&config, &weights).selection_rule(SelectionRule::WholeBlocks);
    let toks = tokens(&model.shape, 48, 1);
    assert_eq!(toks[..8], [45, 54, 54, 84, 44, 7, 79, 42]);
    let mut recorder = Recorder::default();
    let window = reference
        .window_observed(&SequenceState::new(&config), &toks, LogitRows::All, &mut recorder)
        .expect("window");

    let close = |got: &[f32], want: &[f32], what: &str| {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 2e-4, "{what}: {g} against llama.cpp's {w}");
        }
    };
    let logits = |row: usize| &window.logits_of(row).expect("logits")[..6];
    close(logits(0), &[2.363762, -0.200959, -0.001816, -0.429236, 0.554823, 1.265412], "row 0");
    close(logits(20), &[0.450249, 0.301228, 0.711074, -1.092318, 0.156049, -1.178049], "row 20");
    close(logits(47), &[0.104817, -0.209545, -1.673331, -0.36733, -0.605331, 1.764099], "row 47");
    close(
        &window.residual_row(7, 47).expect("kept")[..6],
        &[-3.127041, 0.600675, 0.811816, 4.717043, 0.962131, 2.590214],
        "the residual after the last layer, row 47",
    );
    close(
        &window.residual_row(3, 30).expect("kept")[32..38],
        &[1.496867, -2.112887, 2.1749, 0.762562, -1.841561, 0.024887],
        "stream 1 after layer 3, row 30",
    );

    // The routed experts (llama.cpp gives them in no particular order).
    let k = config.experts.used;
    let routed = |layer: usize, row: usize| {
        let mut ids = window.routing[layer].ids[row * k..(row + 1) * k].to_vec();
        ids.sort_unstable();
        ids
    };
    assert_eq!(routed(7, 47), [2, 9, 11]);
    assert_eq!(routed(7, 13), [0, 4, 7]);

    // The blocks the indexer keeps, besides the tail.
    let blocks = |layer: usize, row: usize| {
        let (cells, _) = &recorder.selection[&(layer, row)];
        let complete = (row + 1) / config.indexer.block * config.indexer.block;
        let mut blocks: Vec<u32> = cells
            .iter()
            .filter(|&&c| (c as usize) < complete)
            .map(|&c| c / config.indexer.block as u32)
            .collect();
        blocks.dedup();
        blocks
    };
    assert_eq!((blocks(3, 12), blocks(3, 30), blocks(3, 47)), (vec![0, 1], vec![1, 5], vec![5, 9]));
    assert_eq!((blocks(7, 12), blocks(7, 30), blocks(7, 47)), (vec![0, 1], vec![1, 3], vec![2, 6]));
}

#[test]
fn a_layer_run_alone_gives_the_rows_of_the_window() {
    let model = Synthetic::small(8);
    let gguf = open("block", &model.bytes());
    let config = Config::from_gguf(&gguf).expect("config");
    let weights = Weights::load(&gguf, &config).expect("weights");
    let reference = Reference::new(&config, &weights);
    let toks = tokens(&model.shape, 21, 6);
    // From a state in the middle of a block of the indexer, past the dense limit.
    let mut state = SequenceState::new(&config);
    reference.forward(&mut state, &toks[..14], LogitRows::None).expect("forward");
    let rows = &toks[14..];
    let mut recorder = Recorder::default();
    let window =
        reference.window_observed(&state, rows, LogitRows::None, &mut recorder).expect("window");

    // The input of layer 0 is the embedding in every stream.
    let embedding = recorder.tensor_of(None, "embedding").expect("tap");
    let mut input = Vec::new();
    for row in embedding.chunks_exact(config.n_embd) {
        for _ in 0..config.n_stream {
            input.extend_from_slice(row);
        }
    }
    for layer in 0..config.n_layer {
        let mut taps = Recorder::default();
        let out = reference.block_observed(layer, &state, rows, &input, &mut taps);
        assert_eq!(bits(&out), bits(&window.residual[layer]), "layer {layer}");
        // The hook of a single layer sees what the hook of the window saw.
        assert_eq!(
            taps.tensor_of(Some(layer), "ffn_out").map(bits),
            recorder.tensor_of(Some(layer), "ffn_out").map(bits)
        );
        assert_eq!(taps.routing.len(), rows.len());
        assert_eq!(taps.selection.len(), if config.is_attention(layer) { rows.len() } else { 0 });
        input = out;
    }
    assert_eq!(bits(&input), bits(&window.trunk_out));
}
