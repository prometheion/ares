//! The elementary operations of the forward pass, in `f64` where a sum or a
//! transcendental function is involved and rounded to `f32` once per result.
//!
//! Every function does its sums in one fixed order that depends only on the
//! length of the data, never on how many rows are processed together or on the
//! number of threads, so a row has the same bits alone and inside a batch.
//! The scalar functions and the norms are those of the `qwen35` reference.

use std::ops::Range;

use crate::qwen4exp::weights::Matrix;
pub use crate::qwen35::reference::pieces::{l2_norm, rms_norm, rms_scale, sigmoid, silu, softplus};

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
pub(crate) fn par_ranges<T: Send>(
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

/// The causal convolution over a history: for each of `n` rows and each of
/// `d` channels
///
/// ```text
/// out[t][ch] = sum_{k = 0 .. kernel-1} taps[ch][k] * in_at(t - (kernel-1-k) * dilation)[ch]
/// ```
///
/// where `in_at(q)` is row `q` of `rows` for `q >= 0`, and below that row
/// `back + q` of `history`: `[back][d]`, oldest first, with `back` the
/// product of `kernel - 1` and `dilation`. The taps are `[d][kernel]`, the
/// last one on the row itself. The sum runs over `k` in order, in `f64`. No
/// activation is applied.
pub fn conv_dilated(
    taps: &[f32],
    kernel: usize,
    dilation: usize,
    history: &[f32],
    rows: &[f32],
    d: usize,
    n: usize,
) -> Vec<f32> {
    let back = (kernel - 1) * dilation;
    debug_assert_eq!(history.len(), back * d);
    let mut out = vec![0.0f32; n * d];
    for t in 0..n {
        for ch in 0..d {
            let mut sum = 0.0f64;
            for k in 0..kernel {
                // Index in "history followed by rows".
                let at = t + back - (kernel - 1 - k) * dilation;
                let x = if at < back { history[at * d + ch] } else { rows[(at - back) * d + ch] };
                sum += f64::from(taps[ch * kernel + k]) * f64::from(x);
            }
            out[t * d + ch] = sum as f32;
        }
    }
    out
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
    fn the_dilated_convolution_reads_the_cells_its_taps_name() {
        // One channel, kernel 3, dilation 2: out[t] = 1*x[t-4] + 10*x[t-2] + 100*x[t].
        let taps = [1.0f32, 10.0, 100.0];
        let history = [1.0f32, 2.0, 3.0, 4.0];
        let rows = [5.0f32, 6.0, 7.0];
        let out = conv_dilated(&taps, 3, 2, &history, &rows, 1, 3);
        assert_eq!(out, [1.0 + 30.0 + 500.0, 2.0 + 40.0 + 600.0, 3.0 + 50.0 + 700.0]);
    }
}
