//! The recurrent and the attention block, as formulas of the architecture
//! document. Shapes in comments are `[rows, ...]` with the window's rows first.

use crate::qwen35::config::Recurrent;
use crate::qwen35::reference::ops::{
    dot_f32, l2_norm, par_ranges, rms_norm, rms_scale, sigmoid, silu, softplus, widen,
};
use crate::qwen35::reference::{LayerRows, Reference};
use crate::qwen35::weights::{AttentionWeights, RecurrentWeights};

/// One token for one value head of the delta rule, changing the matrix `s`
/// (`head_v_dim` rows of `head_k_dim` values, stored as `f32`):
///
/// ```text
/// S      = exp(g) * S
/// delta  = beta * (v - S k)            one value per row of S
/// S      = S + delta k^T
/// out    = S q / sqrt(head_k_dim)
/// ```
///
/// Sums run over `head_k_dim` in index order, in `f64`. The matrix is rounded
/// to `f32` after the update and the output is computed from the rounded
/// matrix, so a state that was stored and loaded again gives the same output.
pub fn delta_step(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    mut out: Option<&mut [f32]>,
) {
    let (dk, dv) = (k.len(), v.len());
    let decay = f64::from(g).exp();
    let scale = 1.0 / (dk as f64).sqrt();
    for j in 0..dv {
        let row = &mut s[j * dk..(j + 1) * dk];
        let mut sk = 0.0f64;
        for i in 0..dk {
            sk += f64::from(row[i]) * decay * f64::from(k[i]);
        }
        let delta = (f64::from(v[j]) - sk) * f64::from(beta);
        let mut o = 0.0f64;
        for i in 0..dk {
            row[i] = (f64::from(row[i]) * decay + f64::from(k[i]) * delta) as f32;
            o += f64::from(row[i]) * f64::from(q[i]);
        }
        if let Some(out) = out.as_deref_mut() {
            out[j] = (o * scale) as f32;
        }
    }
}

/// The causal convolution of a recurrent block with `silu`: `n` rows of the
/// projection `qkv` (`[n, cd]`), preceded by the committed window `window`
/// (`[cd][kk]`, the last `kk` inputs of every channel, oldest first), with the
/// taps `taps` (`[cd][kk + 1]`, tap `kk` on the newest input). Returns `[n, cd]`.
/// The sums are in `f64`.
pub fn conv_silu(
    taps: &[f32],
    kk: usize,
    window: &[f32],
    qkv: &[f32],
    cd: usize,
    n: usize,
) -> Vec<f32> {
    let mut mixed = vec![0.0f32; n * cd];
    for ch in 0..cd {
        let tap = &taps[ch * (kk + 1)..(ch + 1) * (kk + 1)];
        let input = |i: usize| -> f64 {
            f64::from(if i < kk { window[ch * kk + i] } else { qkv[(i - kk) * cd + ch] })
        };
        for t in 0..n {
            let sum: f64 = (0..=kk).map(|i| f64::from(tap[i]) * input(t + i)).sum();
            mixed[t * cd + ch] = silu(sum) as f32;
        }
    }
    mixed
}

/// Advances all value heads of one recurrent block over row `t` of the kept
/// inputs, without computing outputs. This is what a commit does.
#[allow(clippy::too_many_arguments)]
pub(super) fn advance_delta(
    r: &Recurrent,
    delta: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    t: usize,
) {
    let (dk, dv, nv, kd, vd) =
        (r.head_k_dim, r.head_v_dim, r.n_v_heads, r.key_dim(), r.value_dim());
    for h in 0..nv {
        // Value head h pairs with key head h mod n_k_heads (the heads are tiled).
        let kh = h % r.n_k_heads;
        delta_step(
            &mut delta[h * dv * dk..(h + 1) * dv * dk],
            &q[t * kd + kh * dk..][..dk],
            &k[t * kd + kh * dk..][..dk],
            &v[t * vd + h * dv..][..dv],
            g[t * nv + h],
            beta[t * nv + h],
            None,
        );
    }
}

impl Reference<'_, '_> {
    /// A recurrent block over `n` rows `x` (`[n, n_embd]`, already normed),
    /// from the committed convolution window `conv` and matrices `delta`.
    /// Returns the block's output (`[n, n_embd]`) and what a commit needs.
    pub(super) fn recurrent(
        &self,
        w: &RecurrentWeights<'_>,
        conv: &[f32],
        delta: &[f32],
        x: &[f32],
        n: usize,
    ) -> (Vec<f32>, LayerRows) {
        let c = self.config;
        let r = &c.recurrent;
        let (dk, dv, nv) = (r.head_k_dim, r.head_v_dim, r.n_v_heads);
        let (kd, vd, cd, kk) = (r.key_dim(), r.value_dim(), r.conv_dim(), r.conv_kernel - 1);

        let qkv = self.matmul(&w.qkv, x, n); // [n, conv_dim]
        let z = self.matmul(&w.gate, x, n); // [n, value_dim]
        let beta_in = self.matmul(&w.beta, x, n); // [n, n_v_heads]
        let alpha = self.matmul(&w.alpha, x, n); // [n, n_v_heads]

        // beta = sigmoid(beta_in); g = softplus(alpha + dt_bias) * a, with a < 0.
        let beta: Vec<f32> = beta_in.iter().map(|&b| sigmoid(f64::from(b)) as f32).collect();
        let g: Vec<f32> = alpha
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let h = i % nv;
                (softplus(f64::from(a) + f64::from(w.dt_bias[h])) * f64::from(w.a[h])) as f32
            })
            .collect();

        // Causal convolution over the inputs of the window, preceded by the
        // committed inputs; tap kk multiplies the newest input. Then silu.
        let mixed = conv_silu(&w.conv, kk, conv, &qkv, cd, n);

        // Split into query, key, value; normalise each query and key head.
        let mut q = vec![0.0f32; n * kd];
        let mut k = vec![0.0f32; n * kd];
        let mut v = vec![0.0f32; n * vd];
        for t in 0..n {
            let row = &mixed[t * cd..(t + 1) * cd];
            for h in 0..r.n_k_heads {
                let at = t * kd + h * dk;
                l2_norm(&row[h * dk..(h + 1) * dk], c.rms_eps, &mut q[at..at + dk]);
                l2_norm(&row[kd + h * dk..kd + (h + 1) * dk], c.rms_eps, &mut k[at..at + dk]);
            }
            v[t * vd..(t + 1) * vd].copy_from_slice(&row[2 * kd..]);
        }

        // The delta rule, one value head per work item, from the committed matrices.
        let per_head = par_ranges(self.threads, nv, n * dv * dk * 3, |heads| {
            let mut out = Vec::with_capacity(heads.len() * n * dv);
            for h in heads {
                let mut s = delta[h * dv * dk..(h + 1) * dv * dk].to_vec();
                let kh = h % r.n_k_heads;
                for t in 0..n {
                    let mut o = vec![0.0f32; dv];
                    delta_step(
                        &mut s,
                        &q[t * kd + kh * dk..][..dk],
                        &k[t * kd + kh * dk..][..dk],
                        &v[t * vd + h * dv..][..dv],
                        g[t * nv + h],
                        beta[t * nv + h],
                        Some(&mut o),
                    );
                    // The next row goes on from the matrix as the state holds it.
                    self.storage.round(&mut s, dv, dk);
                    out.extend_from_slice(&o);
                }
            }
            out
        });

        // Gated RMS norm per value head: rms_norm(o) * w * silu(z).
        let mut gated = vec![0.0f32; n * vd];
        for h in 0..nv {
            for t in 0..n {
                let o = &per_head[(h * n + t) * dv..(h * n + t + 1) * dv];
                let scale = rms_scale(o, c.rms_eps);
                for j in 0..dv {
                    let zj = f64::from(z[t * vd + h * dv + j]);
                    gated[t * vd + h * dv + j] =
                        (f64::from(o[j]) * scale * f64::from(w.norm[j]) * silu(zj)) as f32;
                }
            }
        }
        let out = self.matmul(&w.out, &gated, n);
        (out, LayerRows::Recurrent { qkv, q, k, v, g, beta })
    }

    /// An attention block over `n` rows `x` (`[n, n_embd]`, already normed) at
    /// positions `start..start + n`, after `start` committed positions whose
    /// keys and values are `k_cache` and `v_cache` (`[start, n_head_kv * head_dim]`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &self,
        w: &AttentionWeights<'_>,
        k_cache: &[f32],
        v_cache: &[f32],
        start: usize,
        x: &[f32],
        n: usize,
        rotary: Option<&[[usize; 3]]>,
    ) -> (Vec<f32>, LayerRows) {
        let c = self.config;
        let a = &c.attention;
        let hd = a.head_dim;
        let (q_dim, kv_dim) = c.attention_dims();

        let qg = self.matmul(&w.q, x, n); // [n, n_head * 2 * head_dim]
        let k_in = self.matmul(&w.k, x, n); // [n, n_head_kv * head_dim]
        let v = self.matmul(&w.v, x, n);

        // Per head the projection holds the query, then the gate.
        let mut q = vec![0.0f32; n * q_dim];
        let mut gate = vec![0.0f32; n * q_dim];
        let mut k = vec![0.0f32; n * kv_dim];
        for t in 0..n {
            for h in 0..a.n_head {
                let src = &qg[t * 2 * q_dim + h * 2 * hd..][..2 * hd];
                rms_norm(&src[..hd], &w.q_norm, c.rms_eps, &mut q[t * q_dim + h * hd..][..hd]);
                gate[t * q_dim + h * hd..][..hd].copy_from_slice(&src[hd..]);
            }
            for h in 0..a.n_head_kv {
                let src = &k_in[t * kv_dim + h * hd..][..hd];
                rms_norm(src, &w.k_norm, c.rms_eps, &mut k[t * kv_dim + h * hd..][..hd]);
            }
        }
        for t in 0..n {
            let table = match rotary {
                Some(rotary) => self.rope_table_at(rotary[t]),
                None => self.rope_table(start + t),
            };
            for h in 0..a.n_head {
                rope(&table, &mut q[t * q_dim + h * hd..][..hd]);
            }
            for h in 0..a.n_head_kv {
                rope(&table, &mut k[t * kv_dim + h * hd..][..hd]);
            }
        }

        // Causal attention: row t sees the committed positions and rows 0..=t.
        let group = a.n_head / a.n_head_kv;
        let scale = 1.0 / (hd as f64).sqrt();
        let (k_all, v_all) = (&k, &v);
        let heads = par_ranges(self.threads, n * a.n_head, (start + n) * hd * 2, |items| {
            let mut out = Vec::with_capacity(items.len() * hd);
            for item in items {
                let (t, h) = (item / a.n_head, item % a.n_head);
                let kv_head = h / group;
                let visible = start + t + 1;
                let qh = widen(&q[t * q_dim + h * hd..][..hd]);
                let key = |j: usize| -> &[f32] {
                    if j < start {
                        &k_cache[j * kv_dim + kv_head * hd..][..hd]
                    } else {
                        &k_all[(j - start) * kv_dim + kv_head * hd..][..hd]
                    }
                };
                let value = |j: usize| -> &[f32] {
                    if j < start {
                        &v_cache[j * kv_dim + kv_head * hd..][..hd]
                    } else {
                        &v_all[(j - start) * kv_dim + kv_head * hd..][..hd]
                    }
                };
                let scores: Vec<f64> = (0..visible).map(|j| dot_f32(&qh, key(j)) * scale).collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                let mut acc = vec![0.0f64; hd];
                for (j, &wj) in weights.iter().enumerate() {
                    for (a, &vv) in acc.iter_mut().zip(value(j)) {
                        *a += wj * f64::from(vv);
                    }
                }
                // Output gate: sigmoid of the second half of the projection.
                for (d, a) in acc.iter().enumerate() {
                    let gate_value = f64::from(gate[t * q_dim + h * hd + d]);
                    out.push((a / total * sigmoid(gate_value)) as f32);
                }
            }
            out
        });
        let out = self.matmul(&w.out, &heads, n);
        (out, LayerRows::Attention { k, v })
    }

    /// `(cos, sin)` of every rotated pair at `pos`.
    ///
    /// Pair `p` rotates by `pos * freq_base^(-2p / dim)` when its section is one
    /// of the three position kinds a text token has (time, height and width all
    /// equal the token position, in the interleaved layout the sections of
    /// `qwen35` use), and by zero when it belongs to the extra kind.
    fn rope_table(&self, pos: usize) -> Vec<(f64, f64)> {
        self.rope_table_at([pos; 3])
    }

    /// `(cos, sin)` of every rotated pair at the rotary position
    /// `(time, height, width)`: pair `p` takes the kind of position its
    /// section names, as `rope_table` does for a text token, and zero for the
    /// extra kind.
    fn rope_table_at(&self, at: [usize; 3]) -> Vec<(f64, f64)> {
        let rope = &self.config.attention.rope;
        let total: usize = rope.sections.iter().sum();
        (0..rope.dim / 2)
            .map(|p| {
                let sector = p % total;
                let s = &rope.sections;
                let kind = sector % 3;
                let in_section = (kind == 1 && sector < 3 * s[1])
                    || (kind == 2 && sector < 3 * s[2])
                    || (kind == 0 && sector < 3 * s[0]);
                let position = if in_section { at[kind] as f64 } else { 0.0 };
                let freq = f64::from(rope.freq_base).powf(-2.0 * p as f64 / rope.dim as f64);
                let angle = position * freq;
                (angle.cos(), angle.sin())
            })
            .collect()
    }
}

/// Rotates the first `2 * table.len()` values of a head: pair `p` is
/// `(x[p], x[p + table.len()])`.
fn rope(table: &[(f64, f64)], head: &mut [f32]) {
    let half = table.len();
    for (p, &(cos, sin)) in table.iter().enumerate() {
        let (x0, x1) = (f64::from(head[p]), f64::from(head[p + half]));
        head[p] = (x0 * cos - x1 * sin) as f32;
        head[p + half] = (x0 * sin + x1 * cos) as f32;
    }
}
