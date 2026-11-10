//! Compares the `qwen4exp` reference with a dump of the llama.cpp fork made by
//! `tools/q4capture` on the same model file and tokens.
//!
//! ```sh
//! cargo run --release -p ares-model --example q4compare -- MODEL.gguf CAPTURE_DIR [budget]
//! ```
//!
//! The reference runs the tokens of `CAPTURE_DIR/tokens.txt` with the fork's
//! selection rule (whole blocks; `budget` switches to Strata's cell budget, to
//! see what difference 1 does) and prints, per layer, how far each
//! intermediate result is from the fork's: the largest absolute difference
//! and the largest relative RMS difference of a row, the rows whose routed
//! experts differ, and the rows whose selected blocks differ.

use std::collections::BTreeSet;
use std::path::Path;

use ares_gguf::Gguf;
use ares_model::qwen4exp::{
    Config, LogitRows, Recorder, Reference, SelectionRule, SequenceState, Weights,
};

type AnyError = Box<dyn std::error::Error>;

fn f32s(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

fn i32s(path: &Path) -> Option<Vec<i32>> {
    let bytes = std::fs::read(path).ok()?;
    Some(bytes.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect())
}

/// The largest absolute difference, and the largest and the median relative
/// RMS difference of a row.
fn distance(ours: &[f32], theirs: &[f32], rows: usize) -> (f64, f64, f64) {
    assert_eq!(ours.len(), theirs.len(), "the two results have different sizes");
    let width = ours.len() / rows;
    let mut max_abs = 0.0f64;
    let mut rel: Vec<f64> = Vec::with_capacity(rows);
    for (a, b) in ours.chunks_exact(width).zip(theirs.chunks_exact(width)) {
        let (mut diff, mut norm) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let d = f64::from(x) - f64::from(y);
            max_abs = max_abs.max(d.abs());
            diff += d * d;
            norm += f64::from(x) * f64::from(x);
        }
        rel.push((diff / norm.max(1e-300)).sqrt());
    }
    rel.sort_by(f64::total_cmp);
    (max_abs, rel[rel.len() - 1], rel[rel.len() / 2])
}

/// The first two dimensions of every call of a dumped node, from `index.txt`:
/// the width of a row and the rows of the call.
fn call_dims(index: &str, file: &str) -> Vec<(usize, usize)> {
    index
        .lines()
        .filter_map(|line| {
            let (name, rest) = line.split_once(" call ")?;
            let mut words = rest.split(' ').skip(2);
            let dims = (words.next()?.parse().ok()?, words.next()?.parse().ok()?);
            (name == file).then_some(dims)
        })
        .collect()
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(model), Some(dir)) = (args.first(), args.get(1)) else {
        return Err("usage: q4compare MODEL.gguf CAPTURE_DIR [budget]".into());
    };
    let rule = match args.get(2).map(String::as_str) {
        Some("budget") => SelectionRule::CellBudget,
        _ => SelectionRule::WholeBlocks,
    };
    let dir = Path::new(dir);
    let tokens: Vec<u32> = std::fs::read_to_string(dir.join("tokens.txt"))?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let index = std::fs::read_to_string(dir.join("index.txt"))?;
    let rows = tokens.len();

    let gguf = Gguf::open(model)?;
    let config = Config::from_gguf(&gguf)?;
    let weights = Weights::load(&gguf, &config)?;
    let reference = Reference::new(&config, &weights).selection_rule(rule);
    let mut rec = Recorder::default();
    let window = reference.window_observed(
        &SequenceState::new(&config),
        &tokens,
        LogitRows::All,
        &mut rec,
    )?;
    println!(
        "{rows} rows, rule {rule:?}; columns: max |diff|, worst row rel rms, median row rel rms"
    );

    let (k, ix) = (config.experts.used, &config.indexer);
    let mut worst_layer_out = 0.0f64;
    for layer in 0..config.n_layer {
        let attention = config.is_attention(layer);
        let out_node = if attention { "attn_output" } else { "linear_attn_out" };
        let pairs = [
            ("attn_in", format!("hc_mixed-{layer}.0.f32")),
            ("attn_out", format!("{out_node}-{layer}.0.f32")),
            ("ffn_in", format!("hc_mixed-{layer}.1.f32")),
            ("ffn_out", format!("ffn_out-{layer}.0.f32")),
            ("layer_out", format!("l_last-{layer}.0.f32")),
        ];
        let mut line = format!("layer {layer:2} {}", if attention { "att" } else { "rec" });
        for (ours, file) in pairs {
            let Some(theirs) = f32s(&dir.join(&file)) else {
                return Err(format!("{file} is missing from the capture").into());
            };
            let ours_data = rec.tensor_of(Some(layer), ours).ok_or("the reference records it")?;
            let (abs, rel, median) = distance(ours_data, &theirs, rows);
            line += &format!(" | {ours} {abs:.1e} {rel:.1e} {median:.1e}");
            if ours == "layer_out" {
                worst_layer_out = worst_layer_out.max(rel);
            }
        }
        println!("{line}");

        // Routing: the fork's ids are in no particular order.
        let theirs = i32s(&dir.join(format!("ffn_moe_topk-{layer}.0.i32"))).ok_or("topk")?;
        let their_weights =
            f32s(&dir.join(format!("ffn_moe_weights_norm-{layer}.0.f32"))).ok_or("weights")?;
        let (mut differ, mut weight_diff) = (Vec::new(), 0.0f32);
        for row in 0..rows {
            let ours = &window.routing[layer].ids[row * k..(row + 1) * k];
            let our_weights = &window.routing[layer].weights[row * k..(row + 1) * k];
            let them = &theirs[row * k..(row + 1) * k];
            let same = ours.iter().map(|&i| i as i32).collect::<BTreeSet<_>>()
                == them.iter().copied().collect::<BTreeSet<_>>();
            if !same {
                differ.push(row);
                continue;
            }
            for (j, id) in them.iter().enumerate() {
                let at = ours.iter().position(|&o| o as i32 == *id).ok_or("same sets")?;
                weight_diff = weight_diff.max((our_weights[at] - their_weights[row * k + j]).abs());
            }
        }
        if !differ.is_empty() || weight_diff > 1e-4 {
            println!(
                "         routing: rows with other experts {differ:?}, max weight diff {weight_diff:.1e}"
            );
        }

        // Selection: scores (the fork scales by 1/sqrt(ID)) and the blocks kept.
        if attention {
            let score_file = format!("indexer_score-{layer}.0.f32");
            let top_file = format!("indexer_top_k-{layer}.0.i32");
            let scores = f32s(&dir.join(&score_file)).ok_or("scores")?;
            let top = i32s(&dir.join(&top_file)).ok_or("top_k")?;
            let (pools, kept) = (call_dims(&index, &score_file), call_dims(&index, &top_file));
            // Walk the calls: each holds [pools, rows of the call] and [kept, rows of the call].
            let (mut s_at, mut t_at, mut row) = (0usize, 0usize, 0usize);
            let (mut score_rel, mut selecting, mut other_blocks) = (0.0f64, 0usize, Vec::new());
            let scale = (ix.key_dim as f64).sqrt();
            for (&(n_pool, call_rows), &(n_kept, _)) in pools.iter().zip(&kept) {
                for _ in 0..call_rows {
                    let (our_cells, our_scores) = &rec.selection[&(layer, row)];
                    let their_scores = &scores[s_at..s_at + n_pool];
                    let their_top = &top[t_at..t_at + n_kept];
                    s_at += n_pool;
                    t_at += n_kept;
                    let visible = (row + 1) / ix.block;
                    if !our_scores.is_empty() {
                        selecting += 1;
                        for (j, &ours) in our_scores.iter().enumerate() {
                            let theirs = f64::from(their_scores[j]) * scale;
                            let d =
                                (f64::from(ours) - theirs).abs() / f64::from(ours).abs().max(1e-3);
                            score_rel = score_rel.max(d);
                        }
                        let our_blocks: BTreeSet<i32> = our_cells
                            .iter()
                            .filter(|&&c| (c as usize) < visible * ix.block)
                            .map(|&c| c as i32 / ix.block as i32)
                            .collect();
                        let their_blocks: BTreeSet<i32> =
                            their_top.iter().copied().filter(|&j| (j as usize) < visible).collect();
                        if our_blocks != their_blocks {
                            other_blocks.push(row);
                        }
                    }
                    row += 1;
                }
            }
            println!(
                "         selection: {selecting} rows select, max rel score diff {score_rel:.1e}, rows with other blocks {other_blocks:?}"
            );
        }
    }

    let theirs = f32s(&dir.join("result_norm.0.f32")).ok_or("result_norm")?;
    let (abs, rel, median) = distance(&window.final_mix, &theirs, rows);
    println!("final_mix {abs:.1e} {rel:.1e} {median:.1e}");
    let theirs = f32s(&dir.join("logits.f32")).ok_or("logits")?;
    let (abs, rel, median) = distance(&window.logits, &theirs, rows);
    let v = config.n_vocab;
    let argmax = |x: &[f32]| x.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    let same = (0..rows)
        .filter(|&r| {
            argmax(&window.logits[r * v..(r + 1) * v]) == argmax(&theirs[r * v..(r + 1) * v])
        })
        .count();
    println!(
        "logits    {abs:.1e} {rel:.1e} {median:.1e}; same greedy pick in {same} of {rows} rows"
    );
    println!("worst layer_out row: {worst_layer_out:.1e}");
    Ok(())
}
