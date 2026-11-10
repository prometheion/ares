//! The elementary operations of the forward pass, in `f64` where a sum or a
//! transcendental function is involved and rounded to `f32` once per result.
//!
//! Every function does its sums in one fixed order that depends only on the
//! length of the data, never on how many rows are processed together or on the
//! number of threads, so a row has the same bits alone and inside a batch.

use std::ops::Range;

/// The dot product of an `f64` vector with a stored `f32` vector, summed in four
/// interleaved `f64` accumulators. A product of two `f32` is exact in `f64`.
pub(crate) fn dot_f32(a: &[f64], b: &[f32]) -> f64 {
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
pub(crate) fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// `1 / (1 + e^-x)`.
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `x * sigmoid(x)`.
pub(crate) fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// `ln(1 + e^x)`, which is `x` to within an `f32` for large `x`.
pub(crate) fn softplus(x: f64) -> f64 {
    if x > 34.0 { x } else { x.exp().ln_1p() }
}

/// The factor of an RMS norm: `1 / sqrt(mean(x^2) + eps)`.
pub(crate) fn rms_scale(x: &[f32], eps: f32) -> f64 {
    let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    1.0 / (sum / x.len() as f64 + f64::from(eps)).sqrt()
}

/// `x * rms_scale(x) * weight`, one head or one row.
pub(crate) fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert!(x.len() == weight.len() && x.len() == out.len());
    let scale = rms_scale(x, eps);
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = (f64::from(v) * scale * f64::from(w)) as f32;
    }
}

/// `x / sqrt(sum(x^2) + eps)`: the normalisation of queries and keys in the
/// recurrent blocks. The epsilon is inside the root, not a floor on the norm.
pub(crate) fn l2_norm(x: &[f32], eps: f32, out: &mut [f32]) {
    let sum: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let scale = 1.0 / (sum + f64::from(eps)).sqrt();
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (f64::from(v) * scale) as f32;
    }
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
        handles.into_iter().flat_map(|h| h.join().expect("a worker thread panicked")).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_sums_every_element_including_the_tail() {
        let a: Vec<f64> = (1..=11).map(f64::from).collect();
        let b32 = vec![2.0f32; 11];
        assert_eq!(dot_f32(&a, &b32), 132.0);
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

    #[test]
    fn par_ranges_keeps_the_order_for_any_thread_count() {
        let one = par_ranges(1, 1000, 1_000_000, |r| r.map(|i| i * i).collect());
        let many = par_ranges(7, 1000, 1_000_000, |r| r.map(|i| i * i).collect());
        assert_eq!(one, many);
        assert_eq!(many.len(), 1000);
    }
}
