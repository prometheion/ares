//! The operations of the forward pass that are Flash-Next's own; the others
//! are every model's (`crate::reference`), with the same fixed orders of
//! summation.

pub use crate::reference::{
    dot, l2_norm, matmul, par_ranges, rms_norm, rms_scale, sigmoid, silu, softplus, widen,
};

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
    fn the_dilated_convolution_reads_the_cells_its_taps_name() {
        // One channel, kernel 3, dilation 2: out[t] = 1*x[t-4] + 10*x[t-2] + 100*x[t].
        let taps = [1.0f32, 10.0, 100.0];
        let history = [1.0f32, 2.0, 3.0, 4.0];
        let rows = [5.0f32, 6.0, 7.0];
        let out = conv_dilated(&taps, 3, 2, &history, &rows, 1, 3);
        assert_eq!(out, [1.0 + 30.0 + 500.0, 2.0 + 40.0 + 600.0, 3.0 + 50.0 + 700.0]);
    }
}
