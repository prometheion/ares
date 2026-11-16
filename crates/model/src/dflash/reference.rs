//! The DFlash drafter on the CPU, in f32: what the device part is held to.
//!
//! Two operations, as the model's steps use them:
//!
//! - **inject**: for each computed row of the model, its taps (the residual
//!   stream before the target layers, concatenated) are fused,
//!   `g = norm(fc . taps)`, and every layer's key and value of `g` (keys with
//!   the per-head norm and the rotary turn of the row's position) are stored
//!   at that position;
//! - **pass**: the block `[newest token, mask x (block - 1)]` at the next
//!   positions runs through the layers, each row attending to the stored
//!   cells of the last `window` positions before it and to every row of the
//!   block (not causal); the final norm, the model's head, the best `top_k`
//!   tokens of rows 1 to `block - 1` and the selector's scores between the
//!   candidates of neighbouring rows make the *lattice*, and the greedy walk
//!   through it is the chain of drafts.

use std::collections::HashMap;

use super::config::{Config, LayerWeights, Matrix, Weights};

/// `x` (rows of `m.n_in` values) times the matrix: rows of `m.n_out` values.
/// The weights are decoded a row at a time, on all cores.
pub fn product(m: &Matrix<'_>, x: &[f32]) -> Vec<f32> {
    let rows = x.len() / m.n_in;
    assert_eq!(rows * m.n_in, x.len(), "{}: rows of {} values", m.name, m.n_in);
    let row_bytes = m.data.len() / m.n_out;
    let mut y = vec![0f32; rows * m.n_out];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let per = m.n_out.div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let range = (t * per).min(m.n_out)..((t + 1) * per).min(m.n_out);
                scope.spawn(move || {
                    let mut w = vec![0f32; m.n_in];
                    let mut out = vec![0f32; rows * range.len()];
                    for (k, o) in range.clone().enumerate() {
                        let bytes = &m.data[o * row_bytes..(o + 1) * row_bytes];
                        ares_quant::dequantize(m.ty, bytes, &mut w).expect("a whole row of blocks");
                        for r in 0..rows {
                            let xr = &x[r * m.n_in..(r + 1) * m.n_in];
                            out[r * range.len() + k] = dot(&w, xr);
                        }
                    }
                    (range, out)
                })
            })
            .collect();
        for handle in handles {
            let (range, out) = handle.join().expect("a product thread");
            for r in 0..rows {
                y[r * m.n_out + range.start..r * m.n_out + range.end]
                    .copy_from_slice(&out[r * range.len()..(r + 1) * range.len()]);
            }
        }
    });
    y
}

/// Row `token` of a table of rows (the selector's), decoded.
pub fn table_row(m: &Matrix<'_>, token: u32) -> Vec<f32> {
    let row_bytes = m.data.len() / m.n_out;
    let mut out = vec![0f32; m.n_in];
    let at = token as usize * row_bytes;
    ares_quant::dequantize(m.ty, &m.data[at..at + row_bytes], &mut out).expect("a whole row");
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// RMS norm of each row of `x` (rows of `w.len()` values), times `w`.
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(w.len()) {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for (v, w) in row.iter_mut().zip(w) {
            *v *= scale * w;
        }
    }
}

/// Turns each head of `x` (heads of `dim` values) by `position`, in the pairs
/// `(i, i + dim / 2)`.
pub fn rope(x: &mut [f32], dim: usize, position: u32, base: f32) {
    let half = dim / 2;
    for head in x.chunks_exact_mut(dim) {
        for i in 0..half {
            let theta = position as f32 * base.powf(-2.0 * i as f32 / dim as f32);
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

/// The dynamic convolution over the rows of one block: row `t`, channel `c`
/// is `sum over taps j of (dynamic[t][g(c), j, side] + base[c, j, side]) *
/// x[t - j][c]`, where a row before the block is zero and `g(c)` is the
/// channel's group.
pub fn conv(c: &Config, x: &[f32], dynamic: &[f32], base: &[f32], side: usize) -> Vec<f32> {
    let (d, k, groups) = (c.n_embd, c.conv_kernel, c.conv_groups());
    let rows = x.len() / d;
    let mut y = vec![0f32; x.len()];
    for t in 0..rows {
        for ch in 0..d {
            let g = ch / c.conv_group;
            let mut sum = 0f32;
            for j in 0..k.min(rows) {
                if t < j {
                    continue;
                }
                let at = j + k * side;
                let w = dynamic[t * c.conv_values() + g + groups * at] + base[ch + d * at];
                sum += w * x[(t - j) * d + ch];
            }
            y[t * d + ch] = sum;
        }
    }
    y
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The key and the value of one cell, each `n_head_kv * head_dim` values.
pub type Cell = (Vec<f32>, Vec<f32>);

/// The keys and values the drafter stored for a sequence, per layer and
/// position.
#[derive(Clone, Debug, Default)]
pub struct Cells {
    /// `[layer][position] -> (k, v)`.
    pub layers: Vec<HashMap<u32, Cell>>,
}

/// One pass's candidates and scores: for each draft position `j` (rows 1 to
/// `block - 1` of the block), the `top_k` best tokens of the head (by their
/// logits, the best first) and their logits, and the scores of each
/// candidate after each candidate of the position before (`scores[j][p][c]`;
/// position 1 follows the newest token, so its rows are all one row).
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    pub tokens: Vec<Vec<u32>>,
    pub unary: Vec<Vec<f32>>,
    pub scores: Vec<Vec<Vec<f32>>>,
    /// The final hidden state of every row of the block, after the norm.
    pub hidden: Vec<f32>,
}

impl Lattice {
    /// The greedy walk: at each position the candidate with the highest score
    /// after the pick before, and how sure the scores are of it (its share of
    /// their softmax).
    pub fn chain(&self) -> Vec<(u32, f32)> {
        let mut out = Vec::with_capacity(self.tokens.len());
        let mut before = 0usize;
        for (j, scores) in self.scores.iter().enumerate() {
            let row = &scores[before];
            let best = argmax(row);
            let sum: f32 = row.iter().map(|s| (s - row[best]).exp()).sum();
            out.push((self.tokens[j][best], 1.0 / sum));
            before = best;
        }
        out
    }
}

/// The first index of the largest value.
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

/// The drafter on the CPU.
#[derive(Debug)]
pub struct Reference<'g> {
    pub c: Config,
    pub w: Weights<'g>,
}

impl<'g> Reference<'g> {
    pub fn new(c: Config, w: Weights<'g>) -> Self {
        Self { c, w }
    }

    /// Stores the cells of rows whose taps are `taps` (rows of
    /// [`Config::tap_values`]) at `positions`.
    pub fn inject(&self, cells: &mut Cells, taps: &[f32], positions: &[u32]) {
        let c = &self.c;
        cells.layers.resize_with(c.n_layer, HashMap::new);
        let mut g = product(&self.w.fc, taps);
        rms_norm(&mut g, &self.w.fc_out_norm.values(), c.eps);
        for (l, lw) in self.w.layers.iter().enumerate() {
            let mut k = product(&lw.k, &g);
            let v = product(&lw.v, &g);
            let kv = c.n_head_kv * c.head_dim;
            rms_norm(&mut k, &lw.k_norm.values(), c.eps);
            for (r, &p) in positions.iter().enumerate() {
                let mut kr = k[r * kv..(r + 1) * kv].to_vec();
                rope(&mut kr, c.head_dim, p, c.rope_base);
                cells.layers[l].insert(p, (kr, v[r * kv..(r + 1) * kv].to_vec()));
            }
        }
    }

    /// The pass of a block at `position` (the newest token's) for a sequence
    /// whose stored cells are `cells`, of which positions below `from` are not
    /// valid. `rows` are the embeddings of the newest token and of the mask
    /// token; `head` gives the logits of the model's head for rows of hidden
    /// states, `head_tokens` the token of each of its columns (none: column
    /// `i` is token `i`).
    #[allow(clippy::too_many_arguments)]
    pub fn pass(
        &self,
        cells: &Cells,
        from: u32,
        position: u32,
        newest: u32,
        rows: (&[f32], &[f32]),
        head: &dyn Fn(&[f32]) -> Vec<f32>,
        head_tokens: Option<&[u32]>,
    ) -> Lattice {
        let c = &self.c;
        let (d, b) = (c.n_embd, c.block);
        let mut x = Vec::with_capacity(b * d);
        x.extend_from_slice(rows.0);
        for _ in 1..b {
            x.extend_from_slice(rows.1);
        }
        for (l, lw) in self.w.layers.iter().enumerate() {
            x = self.layer(l, lw, cells, from, position, &x);
        }
        let mut h = x;
        rms_norm(&mut h, &self.w.output_norm.values(), c.eps);
        // Rows 1 to block - 1 have candidates.
        let drafted = &h[d..];
        let logits = head(drafted);
        let n_head = logits.len() / (b - 1);
        let gate = product(&self.w.selector_hidden, drafted);
        let rank = c.selector_rank;
        let mut lattice = Lattice {
            tokens: Vec::new(),
            unary: Vec::new(),
            scores: Vec::new(),
            hidden: h.clone(),
        };
        let mut before: Vec<u32> = vec![newest];
        for j in 0..b - 1 {
            let row = &logits[j * n_head..(j + 1) * n_head];
            let mut order: Vec<usize> = (0..n_head).collect();
            // The largest first; equal logits by column.
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            order.truncate(c.top_k);
            let tokens: Vec<u32> =
                order.iter().map(|&col| head_tokens.map_or(col as u32, |t| t[col])).collect();
            let unary: Vec<f32> = order.iter().map(|&col| row[col]).collect();
            let g = &gate[j * rank..(j + 1) * rank];
            let successors: Vec<Vec<f32>> =
                tokens.iter().map(|t| table_row(&self.w.selector_successor, *t)).collect();
            let scores: Vec<Vec<f32>> = before
                .iter()
                .map(|p| {
                    let pred = table_row(&self.w.selector_predecessor, *p);
                    let cond: Vec<f32> = pred.iter().zip(g).map(|(a, b)| a * b).collect();
                    successors.iter().zip(&unary).map(|(s, u)| u + dot(s, &cond)).collect()
                })
                .collect();
            lattice.tokens.push(tokens.clone());
            lattice.unary.push(unary);
            lattice.scores.push(scores);
            before = tokens;
        }
        lattice
    }

    fn layer(
        &self,
        l: usize,
        lw: &LayerWeights<'_>,
        cells: &Cells,
        from: u32,
        position: u32,
        x: &[f32],
    ) -> Vec<f32> {
        let c = &self.c;
        let (b, hd) = (c.block, c.head_dim);
        let mut xn = x.to_vec();
        rms_norm(&mut xn, &lw.attn_norm.values(), c.eps);
        let dynamic = product(&lw.attn_conv_proj, &xn);
        let base = lw.attn_conv_base.values();
        let xc = conv(c, &xn, &dynamic, &base, 0);
        let mut q = product(&lw.q, &xc);
        let mut k = product(&lw.k, &xc);
        let v = product(&lw.v, &xc);
        rms_norm(&mut q, &lw.q_norm.values(), c.eps);
        rms_norm(&mut k, &lw.k_norm.values(), c.eps);
        let (qd, kv) = (c.n_head * hd, c.n_head_kv * hd);
        for r in 0..b {
            let p = position + r as u32;
            rope(&mut q[r * qd..(r + 1) * qd], hd, p, c.rope_base);
            rope(&mut k[r * kv..(r + 1) * kv], hd, p, c.rope_base);
        }
        // Attention: row r sees the stored cells of positions above
        // `position + r - window` (and from `from` on) and every row of the
        // block.
        let group = c.n_head / c.n_head_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let empty = HashMap::new();
        let stored = cells.layers.get(l).unwrap_or(&empty);
        let mut o = vec![0f32; b * qd];
        for r in 0..b {
            let p = position as i64 + r as i64;
            let lo = (p - c.window as i64 + 1).max(i64::from(from)).max(0) as u32;
            for h in 0..c.n_head {
                let kh = h / group;
                let qh = &q[r * qd + h * hd..r * qd + (h + 1) * hd];
                let mut keys: Vec<(&[f32], &[f32])> = Vec::new();
                for at in lo..position {
                    if let Some((kc, vc)) = stored.get(&at) {
                        keys.push((&kc[kh * hd..(kh + 1) * hd], &vc[kh * hd..(kh + 1) * hd]));
                    }
                }
                for j in 0..b {
                    keys.push((
                        &k[j * kv + kh * hd..j * kv + (kh + 1) * hd],
                        &v[j * kv + kh * hd..j * kv + (kh + 1) * hd],
                    ));
                }
                let logits: Vec<f32> = keys.iter().map(|(kc, _)| dot(qh, kc) * scale).collect();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = logits.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = weights.iter().sum();
                let out = &mut o[r * qd + h * hd..r * qd + (h + 1) * hd];
                for ((_, vc), w) in keys.iter().zip(&weights) {
                    for (o, v) in out.iter_mut().zip(*vc) {
                        *o += w / sum * v;
                    }
                }
            }
        }
        let attn = product(&lw.out, &o);
        let attn = conv(c, &attn, &dynamic, &base, 1);
        let ffn_in: Vec<f32> = attn.iter().zip(x).map(|(a, x)| a + x).collect();
        let mut fx = ffn_in.clone();
        rms_norm(&mut fx, &lw.ffn_norm.values(), c.eps);
        let fdynamic = product(&lw.ffn_conv_proj, &fx);
        let fbase = lw.ffn_conv_base.values();
        let fx = conv(c, &fx, &fdynamic, &fbase, 0);
        let gate = product(&lw.gate, &fx);
        let up = product(&lw.up, &fx);
        let inner: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        let down = product(&lw.down, &inner);
        let down = conv(c, &down, &fdynamic, &fbase, 1);
        down.iter().zip(&ffn_in).map(|(a, b)| a + b).collect()
    }
}
