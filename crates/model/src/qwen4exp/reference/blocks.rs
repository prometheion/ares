//! The operations of the forward pass that are particular to `qwen4exp`, one
//! function per formula of `docs/reference/qwen4exp-architecture.md`, section
//! 3. Each takes plain slices, so a GPU kernel can be given the same input and
//! compared with the output. Sums are in `f64` in the order the comments name;
//! every result is rounded to `f32` once.

use crate::qwen4exp::config::{Indexer, Ngram, Rope, SelectionRule};
use crate::qwen4exp::reference::ops::{dot, rms_norm, rms_scale, sigmoid, silu, softplus, widen};

// ---------------------------------------------------------------------------
// 3.3 The stream mixer
// ---------------------------------------------------------------------------

/// The norm of a read: `out[c] = rms(r[c]) * norm[c]` for every stream `c` of
/// `n_embd` values, the RMS over the stream's own values.
pub fn stream_norm(r: &[f32], norm: &[f32], n_embd: usize, eps: f32, out: &mut [f32]) {
    debug_assert!(r.len() == norm.len() && r.len() == out.len());
    for ((src, w), dst) in
        r.chunks_exact(n_embd).zip(norm.chunks_exact(n_embd)).zip(out.chunks_exact_mut(n_embd))
    {
        rms_norm(src, w, eps, dst);
    }
}

/// The low-rank activation of a read: `out = silu(down / n_stream)`, where
/// `down` is the down projection of the normed streams. The division is
/// inside the `silu`.
pub fn mixer_squash(down: &[f32], n_stream: usize, out: &mut [f32]) {
    for (o, &x) in out.iter_mut().zip(down) {
        *o = silu(f64::from(x) / n_stream as f64) as f32;
    }
}

/// The gated mean of a read: `out[i] = (1 / hc) * sum_c xn[c][i] *
/// sigmoid(gate[c][i])`, the sum over the streams in order. `xn` and `gate`
/// hold `hc` streams of `out.len()` values.
pub fn mixer_mix(xn: &[f32], gate: &[f32], out: &mut [f32]) {
    let n = out.len();
    let hc = xn.len() / n;
    debug_assert!(xn.len() == hc * n && gate.len() == xn.len());
    for (i, o) in out.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for c in 0..hc {
            sum += f64::from(xn[c * n + i]) * sigmoid(f64::from(gate[c * n + i]));
        }
        *o = (sum / hc as f64) as f32;
    }
}

/// The write of a block's output back to the streams: `r[c] += block_out * 2 *
/// sigmoid(inj[c] / hc)` with `hc = inj.len()`.
pub fn mixer_write(r: &mut [f32], block_out: &[f32], inj: &[f32]) {
    let (n, hc) = (block_out.len(), inj.len());
    debug_assert_eq!(r.len(), n * hc);
    for (stream, &j) in r.chunks_exact_mut(n).zip(inj) {
        let factor = 2.0 * sigmoid(f64::from(j) / hc as f64);
        for (x, &o) in stream.iter_mut().zip(block_out) {
            *x = (f64::from(*x) + f64::from(o) * factor) as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// 3.4 The hashed n-gram embedding
// ---------------------------------------------------------------------------

/// The table rows of the token at a cell: `token` is the cell's own token and
/// `before` the tokens of the cells before it in the sequence, oldest first
/// (only the last `ngram_size - 1` are read; fewer means the sequence starts
/// there). Head `h` of n-gram order `o` (`o = 2 ..= ngram_size`) is at index
/// `(o - 2) * heads_per_ngram + h`.
///
/// The window slot of the token itself never cuts; a predecessor that is
/// missing or is the cut token replaces itself and everything older by the
/// cut token. Products and the exclusive or are unsigned 64-bit, wrapping.
pub fn ngram_rows(ngram: &Ngram, token: u32, before: &[u32]) -> Vec<u64> {
    let mut rows = Vec::with_capacity(ngram.n_heads());
    let mut mixed = u64::from(token).wrapping_mul(ngram.multipliers[0]);
    let mut cut = false;
    for slot in 1..ngram.ngram_size {
        let found = before.len().checked_sub(slot).map(|i| before[i]);
        cut = cut || found.is_none_or(|t| t == ngram.eos_token);
        let ctx = if cut { ngram.eos_token } else { found.unwrap_or(ngram.eos_token) };
        mixed ^= u64::from(ctx).wrapping_mul(ngram.multipliers[slot]);
        for h in (slot - 1) * ngram.heads_per_ngram..slot * ngram.heads_per_ngram {
            rows.push(mixed % ngram.head_vocab[h] + ngram.head_offset[h]);
        }
    }
    rows
}

/// The gate of the n-gram block for one row: for every stream `c`
///
/// ```text
/// kn = rms(key[c]) * norm_key[c]        qn = rms(r[c]) * norm_query[c]
/// s  = (kn . qn) / sqrt(n_embd)
/// gate[c] = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
/// ```
///
/// `kn` and `qn` are rounded to `f32` before the dot product, which is
/// [`dot`]. `sign(0)` is 0.
pub fn ngram_gate(
    key: &[f32],
    r: &[f32],
    norm_key: &[f32],
    norm_query: &[f32],
    eps: f32,
    gate: &mut [f32],
) {
    let n = key.len() / gate.len();
    let (mut kn, mut qn) = (vec![0.0f32; n], vec![0.0f32; n]);
    for (c, g) in gate.iter_mut().enumerate() {
        let at = c * n..(c + 1) * n;
        rms_norm(&key[at.clone()], &norm_key[at.clone()], eps, &mut kn);
        rms_norm(&r[at.clone()], &norm_query[at], eps, &mut qn);
        let s = dot(&widen(&kn), &qn) / (n as f64).sqrt();
        let sign = if s > 0.0 {
            1.0
        } else if s < 0.0 {
            -1.0
        } else {
            0.0
        };
        *g = sigmoid(sign * s.abs().max(1e-6).sqrt()) as f32;
    }
}

/// The gated value of the n-gram block: `out[c] = value * gate[c]`.
pub fn ngram_gated(value: &[f32], gate: &[f32], out: &mut [f32]) {
    for (stream, &g) in out.chunks_exact_mut(value.len()).zip(gate) {
        for (o, &v) in stream.iter_mut().zip(value) {
            *o = (f64::from(v) * f64::from(g)) as f32;
        }
    }
}

/// The residual update of the n-gram block: `r += gated + silu(conv)`.
pub fn ngram_update(r: &mut [f32], gated: &[f32], conv: &[f32]) {
    for ((x, &g), &c) in r.iter_mut().zip(gated).zip(conv) {
        *x = (f64::from(*x) + f64::from(g) + silu(f64::from(c))) as f32;
    }
}

// ---------------------------------------------------------------------------
// 3.5 The recurrent layer
// ---------------------------------------------------------------------------

/// The exponent of the decay of one value head: `softplus(alpha + dt_bias) *
/// a`, with `a < 0`.
pub fn decay(alpha: f32, dt_bias: f32, a: f32) -> f32 {
    (softplus(f64::from(alpha) + f64::from(dt_bias)) * f64::from(a)) as f32
}

/// The output norm of one value head: `out = rms(y0) * norm * sigmoid(z)`,
/// the RMS over the head's values. (Qwen3.5 has `silu(z)` here.)
pub fn gated_head_norm(y0: &[f32], norm: &[f32], z: &[f32], eps: f32, out: &mut [f32]) {
    let scale = rms_scale(y0, eps);
    for (((o, &y), &w), &z) in out.iter_mut().zip(y0).zip(norm).zip(z) {
        *o = (f64::from(y) * scale * f64::from(w) * sigmoid(f64::from(z))) as f32;
    }
}

// ---------------------------------------------------------------------------
// 3.6 The attention layer
// ---------------------------------------------------------------------------

/// `(cos, sin)` of every rotated pair at the position triple `pos = (t, h,
/// w)`. Pair `i` rotates by `P(i) * freq_base^(-2 i / dim)`, where `P(i)` is
/// `t`, `h` or `w` for `i % 3 == 0, 1, 2` while the pair lies inside that
/// kind's section, and zero for a pair of the extra section. Text has `t = h =
/// w =` the cell index. Angles are computed in `f64`.
pub fn rope_table(rope: &Rope, pos: [f64; 3]) -> Vec<(f64, f64)> {
    let total: usize = rope.sections.iter().sum();
    (0..rope.dim / 2)
        .map(|i| {
            let sector = i % total;
            let kind = sector % 3;
            let position = if sector < 3 * rope.sections[kind] { pos[kind] } else { 0.0 };
            let freq = f64::from(rope.freq_base).powf(-2.0 * i as f64 / rope.dim as f64);
            let angle = position * freq;
            (angle.cos(), angle.sin())
        })
        .collect()
}

/// Rotates the first `2 * table.len()` values of a head and leaves the rest:
/// pair `i` is `(x[i], x[i + table.len()])` and becomes `(x0 cos - x1 sin, x0
/// sin + x1 cos)`.
pub fn rope(table: &[(f64, f64)], head: &mut [f32]) {
    let half = table.len();
    for (i, &(cos, sin)) in table.iter().enumerate() {
        let (x0, x1) = (f64::from(head[i]), f64::from(head[i + half]));
        head[i] = (x0 * cos - x1 * sin) as f32;
        head[i + half] = (x0 * sin + x1 * cos) as f32;
    }
}

/// A head through its norm and the rotation: `out = rope(rms(x) * norm)`.
/// This is what queries, keys and indexer queries go through.
pub fn norm_rope(x: &[f32], norm: &[f32], eps: f32, table: &[(f64, f64)], out: &mut [f32]) {
    rms_norm(x, norm, eps, out);
    rope(table, out);
}

/// The pooled key of a complete block: the mean of the raw keys of its cells
/// (`raw` holds them one after the other, `out.len()` values each, summed in
/// cell order and divided by their number), then `rms * norm`, then the
/// rotation, whose `table` must be the one of the block's first cell.
pub fn pool_key(raw: &[f32], norm: &[f32], eps: f32, table: &[(f64, f64)], out: &mut [f32]) {
    let dim = out.len();
    let cells = raw.len() / dim;
    let mut mean = vec![0.0f32; dim];
    for (i, m) in mean.iter_mut().enumerate() {
        let sum: f64 = (0..cells).map(|c| f64::from(raw[c * dim + i])).sum();
        *m = (sum / cells as f64) as f32;
    }
    norm_rope(&mean, norm, eps, table, out);
}

/// The score of one block for one row: `sum_e max(0, u[e] . pooled)` over the
/// indexer query heads `e` (`u` holds them one after the other), each dot
/// product a [`dot`], the rectifier per head before the sum over heads in
/// order. No scale and no softmax.
pub fn block_score(u: &[f32], pooled: &[f32]) -> f32 {
    let p = widen(pooled);
    let sum: f64 = u.chunks_exact(pooled.len()).map(|head| dot(&p, head).max(0.0)).sum();
    sum as f32
}

/// The cells a row attends to, ascending. `n` is the number of cells the row
/// sees, its own included (`n = cell + 1`), and `scores` the scores of its
/// `n / block` complete blocks; it is not read while `n` is within
/// [`Indexer::dense_limit`], where the row attends to all `n` cells.
///
/// Past that, the tail (the `n % block` newest cells) is always selected and
/// the blocks are taken in the order of their score, highest first, equal
/// scores by lower block index first: under [`SelectionRule::CellBudget`]
/// until `top_k + block - 1` cells are selected in all, the last block giving
/// only its first cells when the budget ends inside it; under
/// [`SelectionRule::WholeBlocks`] the best `top_k / block` blocks whole.
///
/// # Panics
///
/// Panics when selection is needed and `scores` is not `n / block` long.
pub fn select_cells(rule: SelectionRule, indexer: &Indexer, n: usize, scores: &[f32]) -> Vec<u32> {
    let block = indexer.block;
    if n <= indexer.dense_limit() {
        return (0..n as u32).collect();
    }
    let (blocks, tail) = (n / block, n % block);
    assert_eq!(scores.len(), blocks, "a row that sees {n} cells scores {blocks} blocks");
    let mut order: Vec<u32> = (0..blocks as u32).collect();
    // A stable sort keeps the lower index first among equal scores.
    order.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
    let budget = match rule {
        SelectionRule::CellBudget => indexer.dense_limit() - tail,
        SelectionRule::WholeBlocks => indexer.top_k,
    };
    let mut cells = Vec::with_capacity(budget + tail);
    for &j in &order {
        let take = block.min(budget - cells.len());
        if take == 0 {
            break;
        }
        cells.extend((0..take as u32).map(|i| j * block as u32 + i));
    }
    cells.extend((blocks * block) as u32..n as u32);
    cells.sort_unstable();
    cells
}

/// One query head over a list of cells:
///
/// ```text
/// w[c] = softmax over the cells of (q . key(c)) / sqrt(q.len())
/// out  = (sum_c w[c] * value(c)) * sigmoid(gate)
/// ```
///
/// Scores are [`dot`]s; the softmax subtracts the largest score, sums the
/// exponentials in list order and divides once at the end; the weighted sum
/// of the values runs in list order.
pub fn attend<'a>(
    q: &[f32],
    gate: &[f32],
    cells: &[u32],
    key: impl Fn(u32) -> &'a [f32],
    value: impl Fn(u32) -> &'a [f32],
    out: &mut [f32],
) {
    let qh = widen(q);
    let scale = 1.0 / (q.len() as f64).sqrt();
    let scores: Vec<f64> = cells.iter().map(|&c| dot(&qh, key(c)) * scale).collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut acc = vec![0.0f64; out.len()];
    for (&c, &w) in cells.iter().zip(&weights) {
        for (a, &v) in acc.iter_mut().zip(value(c)) {
            *a += w * f64::from(v);
        }
    }
    for ((o, a), &g) in out.iter_mut().zip(&acc).zip(gate) {
        *o = (a / total * sigmoid(f64::from(g))) as f32;
    }
}

// ---------------------------------------------------------------------------
// 3.7 The mixture of experts
// ---------------------------------------------------------------------------

/// The router's choice for one row: the `ids.len()` experts with the largest
/// logit, largest first and equal logits by lower index first, and their
/// weights `prob[id] / max(sum of the chosen prob, 2^-14)`, where `prob` is
/// the softmax over all experts (largest logit subtracted, exponentials summed
/// in index order, in `f64`).
pub fn route(logits: &[f32], ids: &mut [u32], weights: &mut [f32]) {
    debug_assert_eq!(ids.len(), weights.len());
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    ids.copy_from_slice(&order[..ids.len()]);

    let max = f64::from(logits[ids[0] as usize]);
    let total: f64 = logits.iter().map(|&l| (f64::from(l) - max).exp()).sum();
    let prob = |id: u32| (f64::from(logits[id as usize]) - max).exp() / total;
    let chosen: f64 = ids.iter().map(|&id| prob(id)).sum();
    let norm = chosen.max(2.0f64.powi(-14));
    for (w, &id) in weights.iter_mut().zip(ids.iter()) {
        *w = (prob(id) / norm) as f32;
    }
}

/// The inner activation of an expert: `out = silu(gate) * up`.
pub fn expert_act(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate).zip(up) {
        *o = (silu(f64::from(g)) * f64::from(u)) as f32;
    }
}

/// The scalar gate of the shared expert: `sigmoid(w . m)`, a [`dot`].
pub fn shared_gate(w: &[f32], m: &[f32]) -> f32 {
    sigmoid(dot(&widen(m), w)) as f32
}

/// The output of the mixture: the weighted sum of the routed experts plus the
/// gated shared expert,
///
/// ```text
/// out = sum_j weights[j] * routed[j]  +  shared_gate * shared
/// ```
///
/// with the routed experts in rank order (`routed` holds their outputs one
/// after the other) and the shared expert last.
pub fn moe_combine(
    routed: &[f32],
    weights: &[f32],
    shared: &[f32],
    shared_gate: f32,
    out: &mut [f32],
) {
    let n = out.len();
    for (i, o) in out.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (j, &w) in weights.iter().enumerate() {
            sum += f64::from(w) * f64::from(routed[j * n + i]);
        }
        *o = (sum + f64::from(shared_gate) * f64::from(shared[i])) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_ngram() -> Ngram {
        let head_vocab: Vec<u64> = vec![
            20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
            20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
        ];
        let mut head_offset = Vec::new();
        let mut at = 0u64;
        for size in &head_vocab {
            head_offset.push(at);
            at += size;
        }
        Ngram {
            layer: 1,
            ngram_size: 3,
            heads_per_ngram: 8,
            conv_kernel: 4,
            eos_token: 248044,
            image_token: 248056,
            row_dim: 160,
            multipliers: vec![23703573157769, 20109073645365, 8052911324071],
            head_vocab,
            head_offset,
            table_rows: 320001536,
        }
    }

    /// Strata's oracle rows (`src/kernels/ple_oracle_vectors.inc`), with the
    /// constants of Qwen3.8-Flash-Next.
    #[test]
    fn the_hash_gives_stratas_oracle_rows() {
        let ngram = real_ngram();
        assert_eq!(
            ngram_rows(&ngram, 11, &[]),
            [
                13555483, 24877159, 54278900, 63631058, 98394898, 117447587, 136774765, 157274238,
                173487882, 192109366, 213364329, 236577563, 258758149, 262164274, 283572184,
                312653920
            ]
        );
        assert_eq!(
            ngram_rows(&ngram, 44, &[33, 22]),
            [
                12721672, 27113445, 44309803, 60385209, 97021817, 115900810, 134219364, 151977589,
                161183764, 181257609, 201344488, 221596875, 241635416, 261674107, 281687144,
                301751935
            ]
        );
    }

    #[test]
    fn the_cut_token_replaces_itself_and_everything_older() {
        let ngram = real_ngram();
        let eos = ngram.eos_token;
        // No predecessors hash as two cut tokens.
        assert_eq!(ngram_rows(&ngram, 11, &[]), ngram_rows(&ngram, 11, &[eos, eos]));
        // A cut token one back also hides the token two back.
        assert_eq!(ngram_rows(&ngram, 11, &[77, eos]), ngram_rows(&ngram, 11, &[eos, eos]));
        // A cut token two back leaves the token one back alone.
        let two_back = ngram_rows(&ngram, 11, &[eos, 5]);
        assert_eq!(two_back, ngram_rows(&ngram, 11, &[5]));
        assert_ne!(two_back[..8], ngram_rows(&ngram, 11, &[])[..8]);
        // The token itself never cuts.
        assert_ne!(ngram_rows(&ngram, eos, &[3, 4]), ngram_rows(&ngram, eos, &[eos, eos]));
        // Only the newest two predecessors are read.
        assert_eq!(ngram_rows(&ngram, 44, &[9, 9, 33, 22]), ngram_rows(&ngram, 44, &[33, 22]));
    }

    fn indexer() -> Indexer {
        Indexer { n_head: 2, key_dim: 4, block: 4, top_k: 8 }
    }

    #[test]
    fn selection_takes_everything_up_to_the_dense_limit() {
        let ix = indexer();
        for rule in [SelectionRule::CellBudget, SelectionRule::WholeBlocks] {
            assert_eq!(select_cells(rule, &ix, 11, &[]), (0..11).collect::<Vec<u32>>());
        }
    }

    #[test]
    fn the_cell_budget_takes_part_of_the_last_block() {
        let ix = indexer();
        // 14 cells: blocks 0..3 complete, tail of 2 (cells 12, 13). Budget 11 - 2 = 9 cells:
        // blocks 2 and 0 whole, and the first cell of block 1.
        let scores = [5.0, 3.0, 9.0];
        let cells = select_cells(SelectionRule::CellBudget, &ix, 14, &scores);
        assert_eq!(cells, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]);
        let cells = select_cells(SelectionRule::WholeBlocks, &ix, 14, &scores);
        assert_eq!(cells, [0, 1, 2, 3, 8, 9, 10, 11, 12, 13]);
        // A tail of 3 leaves no partial block: both rules agree.
        let scores = [5.0, 3.0, 9.0];
        assert_eq!(
            select_cells(SelectionRule::CellBudget, &ix, 15, &scores),
            select_cells(SelectionRule::WholeBlocks, &ix, 15, &scores)
        );
        // No tail: the row's own block is an ordinary block.
        let scores = [5.0, 3.0, 9.0, 0.0];
        let cells = select_cells(SelectionRule::CellBudget, &ix, 16, &scores);
        assert_eq!(cells, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]);
    }

    #[test]
    fn equal_scores_go_to_the_lower_block() {
        let ix = indexer();
        let scores = [0.0, 7.0, 0.0, 0.0];
        let cells = select_cells(SelectionRule::WholeBlocks, &ix, 16, &scores);
        assert_eq!(cells, [0, 1, 2, 3, 4, 5, 6, 7]);
        let cells = select_cells(SelectionRule::CellBudget, &ix, 17, &scores);
        assert_eq!(cells, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16]);
    }

    #[test]
    fn the_block_score_rectifies_each_head_before_the_sum() {
        // Two heads of two values: +3 and -5 against the key (1, 1).
        let u = [1.0f32, 2.0, -2.0, -3.0];
        assert_eq!(block_score(&u, &[1.0, 1.0]), 3.0);
        assert_eq!(block_score(&u, &[-1.0, -1.0]), 5.0);
    }

    #[test]
    fn the_router_orders_by_weight_and_breaks_ties_by_index() {
        let logits = [0.0f32, 2.0, 2.0, -1.0, 1.0];
        let (mut ids, mut weights) = ([0u32; 3], [0.0f32; 3]);
        route(&logits, &mut ids, &mut weights);
        assert_eq!(ids, [1, 2, 4]);
        let e = std::f64::consts::E;
        let expected = [e / (2.0 * e + 1.0), e / (2.0 * e + 1.0), 1.0 / (2.0 * e + 1.0)];
        for (w, x) in weights.iter().zip(expected) {
            assert!((f64::from(*w) - x).abs() < 1e-7, "{w} against {x}");
        }
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_leaves_the_values_past_the_rotated_width() {
        let rope_cfg = Rope { dim: 4, sections: [1, 1, 0, 0], freq_base: 100.0 };
        let table = rope_table(&rope_cfg, [3.0, 5.0, 7.0]);
        // Pair 0 reads t with frequency 1, pair 1 reads h with frequency 100^(-1/2).
        assert!((table[0].0 - 3.0f64.cos()).abs() < 1e-15);
        assert!((table[1].1 - 0.5f64.sin()).abs() < 1e-15);
        let mut head = [1.0f32, 0.0, 0.0, 1.0, 9.0, 8.0];
        rope(&table, &mut head);
        assert!((f64::from(head[0]) - 3.0f64.cos()).abs() < 1e-7);
        assert!((f64::from(head[2]) - 3.0f64.sin()).abs() < 1e-7);
        assert!((f64::from(head[1]) + 0.5f64.sin()).abs() < 1e-7);
        assert_eq!(head[4..], [9.0, 8.0]);
    }

    #[test]
    fn the_write_scales_each_stream_by_its_own_factor() {
        let mut r = [1.0f32, 1.0, 1.0, 1.0];
        mixer_write(&mut r, &[1.0, 2.0], &[0.0, 200.0]);
        // sigmoid(0) = 0.5 gives a factor of 1; a large inject gives 2.
        assert_eq!(r, [2.0, 3.0, 3.0, 5.0]);
    }
}
