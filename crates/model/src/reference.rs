//! What the CPU references of every model are built from: the elementary
//! operations of a forward pass, in `f64` where a sum or a transcendental
//! function is involved and rounded to `f32` once per result, the delta rule
//! and the convolution of a recurrent layer, and what a window takes besides
//! its tokens.
//!
//! Every function does its sums in one fixed order that depends only on the
//! length of the data, never on how many rows are processed together or on
//! the number of threads, so a row has the same bits alone and inside a batch.
//! (The image tower's and the DFlash drafter's references keep their own sums,
//! in other orders: they are held to other implementations.)

use std::ops::Range;

use crate::files::Matrix;

/// The dot product of an `f64` vector with a stored `f32` vector, summed in
/// four interleaved `f64` accumulators: element `i` goes to accumulator
/// `i % 4` over the whole multiples of four, the accumulators are added as
/// `(0 + 1) + (2 + 3)`, and the remaining elements follow in order. A product
/// of two `f32` is exact in `f64`.
pub fn dot(a: &[f64], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f64; 4];
    let ((a4, a_tail), (b4, b_tail)) = (a.as_chunks::<4>(), b.as_chunks::<4>());
    for (x, y) in a4.iter().zip(b4) {
        for lane in 0..4 {
            acc[lane] += x[lane] * f64::from(y[lane]);
        }
    }
    let tail: f64 = a_tail.iter().zip(b_tail).map(|(x, y)| x * f64::from(*y)).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

/// `f32` values widened to `f64`.
pub fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// Runs `work` on contiguous ranges of `0..n` on up to `threads` threads and
/// concatenates what the ranges return, in order. With little work, or one
/// thread, nothing is spawned.
pub fn par_ranges<T: Send>(
    threads: usize,
    n: usize,
    cost: usize,
    work: impl Fn(Range<usize>) -> Vec<T> + Sync,
) -> Vec<T> {
    // Spawning a thread costs about as much as 20 000 multiply-adds.
    let threads = threads.min(n).min(n.saturating_mul(cost) / 20_000).max(1);
    if threads == 1 {
        return work(0..n);
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let work = &work;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (start, end) = ((t * per).min(n), ((t + 1) * per).min(n));
                scope.spawn(move || work(start..end))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| match h.join() {
                Ok(part) => part,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

/// The outputs `range` of `W x[t]` for `n` rows, as `[output][row]`: every
/// stored row is decoded once ([`ares_quant::dequantize`]) and multiplied
/// with each input row by [`dot`].
fn matmul_range(w: &Matrix<'_>, xs: &[f64], n: usize, range: Range<usize>) -> Vec<f32> {
    let cols = w.cols();
    let mut stored = vec![0.0f32; cols];
    let mut out = Vec::with_capacity(range.len() * n);
    for r in range {
        w.decode_row(r, &mut stored);
        for t in 0..n {
            out.push(dot(&xs[t * cols..(t + 1) * cols], &stored) as f32);
        }
    }
    out
}

/// A matrix times rows: `y[t] = W x[t]` for `n` rows. Row `t` of `x` has
/// `w.cols()` values and row `t` of the result `w.rows()`. Every output is one
/// [`dot`] of the input row with the dequantized weight row, rounded to `f32`;
/// the result does not depend on `threads`.
///
/// # Panics
///
/// Panics when `x` is not `n` rows of `w.cols()` values.
pub fn matmul(threads: usize, w: &Matrix<'_>, x: &[f32], n: usize) -> Vec<f32> {
    let (rows, cols) = (w.rows(), w.cols());
    assert_eq!(x.len(), n * cols, "x must be {n} rows of {cols} values");
    let xs = widen(x);
    let by_row = par_ranges(threads, rows, cols * n.max(1), |range| matmul_range(w, &xs, n, range));
    if n == 1 {
        return by_row;
    }
    let mut y = vec![0.0f32; n * rows];
    for (r, per_token) in by_row.chunks_exact(n).enumerate() {
        for (t, &value) in per_token.iter().enumerate() {
            y[t * rows + r] = value;
        }
    }
    y
}

/// `1 / (1 + e^-x)`.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `x * sigmoid(x)`.
pub fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// `ln(1 + e^x)`, which is `x` to within an `f32` for large `x`.
pub fn softplus(x: f64) -> f64 {
    if x > 34.0 { x } else { x.exp().ln_1p() }
}

/// The factor of an RMS norm: `1 / sqrt(mean(x^2) + eps)`.
pub fn rms_scale(x: &[f32], eps: f32) -> f64 {
    let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    1.0 / (sum / x.len() as f64 + f64::from(eps)).sqrt()
}

/// `x * rms_scale(x) * weight`, one head or one row.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert!(x.len() == weight.len() && x.len() == out.len());
    let scale = rms_scale(x, eps);
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = (f64::from(v) * scale * f64::from(w)) as f32;
    }
}

/// `x / sqrt(sum(x^2) + eps)`: the normalisation of queries and keys in the
/// recurrent blocks. The epsilon is inside the root, not a floor on the norm.
pub fn l2_norm(x: &[f32], eps: f32, out: &mut [f32]) {
    let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let scale = 1.0 / (sum + f64::from(eps)).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (f64::from(v) * scale) as f32;
    }
}

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

/// What a window of a model's reference takes besides its tokens
/// (`Reference::window_with` of each model).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inputs {
    /// Rows of the window whose input is these `n_embd` values in place of
    /// their token's embedding (an image's rows).
    pub rows: Vec<(usize, Vec<f32>)>,
    /// Per row of the window its rotary position `(time, height, width)`;
    /// `None`: the row's cell for all three.
    pub rotary: Option<Vec<[usize; 3]>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_sums_every_element_including_the_tail() {
        let a: Vec<f64> = (1..=11).map(f64::from).collect();
        let b32 = vec![2.0f32; 11];
        assert_eq!(dot(&a, &b32), 132.0);
    }

    #[test]
    fn par_ranges_keeps_the_order_for_any_thread_count() {
        let one = par_ranges(1, 1000, 1_000_000, |r| r.map(|i| i * i).collect());
        let many = par_ranges(7, 1000, 1_000_000, |r| r.map(|i| i * i).collect());
        assert_eq!(one, many);
        assert_eq!(many.len(), 1000);
    }

    #[test]
    fn rms_norm_of_a_constant_row_is_the_weight() {
        let x = [3.0f32; 8];
        let w: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 8];
        rms_norm(&x, &w, 0.0, &mut out);
        for (o, w) in out.iter().zip(&w) {
            assert!((o - w).abs() < 1e-6);
        }
    }

    #[test]
    fn l2_norm_gives_a_unit_vector_up_to_epsilon() {
        let x = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        l2_norm(&x, 0.0, &mut out);
        assert!((out[0] - 0.6).abs() < 1e-7 && (out[1] - 0.8).abs() < 1e-7);
        // The epsilon sits inside the root, it is not a floor on the norm: a
        // vector whose squares sum to eps is scaled by 1 / sqrt(2 eps).
        l2_norm(&[1e-3, 0.0], 1e-6, &mut out);
        assert!((f64::from(out[0]) - 0.5f64.sqrt()).abs() < 1e-7, "{}", out[0]);
        l2_norm(&[0.0, 0.0], 1e-6, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn softplus_and_silu_match_their_definitions() {
        assert!((softplus(0.0) - 2.0f64.ln()).abs() < 1e-15);
        assert_eq!(softplus(100.0), 100.0);
        assert!((softplus(-40.0) - (-40.0f64).exp()).abs() < 1e-30);
        assert!((silu(1.0) - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-15);
        assert_eq!(silu(0.0), 0.0);
    }
}
