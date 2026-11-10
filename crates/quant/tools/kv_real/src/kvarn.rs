//! A re-implementation of the quantizer of kvarn (beellama.cpp,
//! `src/llama-kvarn.cpp` at 1cef3bf), following the CPU reference functions
//! `llama_kvarn_variance_normalize`, `llama_kvarn_quantize_tile` and
//! `llama_kvarn_dequantize_*_tile` operation by operation (same float and
//! double types, same order), so that `validate` can compare it with the
//! original compiled from the source.
//!
//! A tile is 128 x 128 values. For K it is `[channel][token]`, for V
//! `[token][channel]`; the quantizer does not care: it normalises both axes,
//! then gives every row its own min/max grid of `2^bits` levels.

use ares_quant::{f16_to_f32, f32_to_f16};

pub const T: usize = 128;
const N: usize = T * T;

/// The 128-point transform of `llama_kvarn_hadamard_128`.
pub fn hadamard_128(v: &mut [f32]) {
    debug_assert_eq!(v.len(), 128);
    let mut stride = 1;
    while stride < 128 {
        let mut base = 0;
        while base < 128 {
            for i in 0..stride {
                let a = v[base + i];
                let b = v[base + stride + i];
                v[base + i] = a + b;
                v[base + stride + i] = a - b;
            }
            base += 2 * stride;
        }
        stride *= 2;
    }
    for x in v.iter_mut() {
        *x *= 0.08838834764831845_f32;
    }
}

/// The head-wide transform of `apply_reference_kvarn_wht_head` for 128, 256
/// and 512 values (kvarn: one normalised Walsh-Hadamard transform over the
/// whole head, no sign flips, built from 128-point transforms and a butterfly
/// across the slices).
pub fn rotate_head(x: &mut [f32]) {
    let slices = x.len() / 128;
    for s in 0..slices {
        hadamard_128(&mut x[s * 128..(s + 1) * 128]);
    }
    if slices == 1 {
        return;
    }
    let scale = if slices == 2 { 0.7071067811865475_f32 } else { 0.5_f32 };
    for d in 0..128 {
        let mut v = [0.0f32; 4];
        for s in 0..slices {
            v[s] = x[s * 128 + d];
        }
        let mut stride = 1;
        while stride < slices {
            let mut base = 0;
            while base < slices {
                for i in 0..stride {
                    let a = v[base + i];
                    let b = v[base + stride + i];
                    v[base + i] = a + b;
                    v[base + stride + i] = a - b;
                }
                base += 2 * stride;
            }
            stride <<= 1;
        }
        for s in 0..slices {
            x[s * 128 + d] = v[s] * scale;
        }
    }
}

/// `llama_kvarn_sample_std`: sums in double, result as float.
fn sample_std(v: &[f32], offset: usize, stride: usize) -> f32 {
    let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
    for i in 0..T {
        let x = f64::from(v[offset + i * stride]);
        sum += x;
        sum_sq += x * x;
    }
    let n = T as f64;
    let mean = sum / n;
    let variance = (0.0f64).max((sum_sq - n * mean * mean) / (n - 1.0));
    variance.sqrt() as f32
}

fn imbalance(tile: &[f32]) -> f32 {
    let (mut col_min, mut col_max, mut row_min, mut row_max) = (f32::INFINITY, 0.0f32, f32::INFINITY, 0.0f32);
    for c in 0..T {
        let v = sample_std(tile, c, T);
        col_min = col_min.min(v);
        col_max = col_max.max(v);
    }
    for r in 0..T {
        let v = sample_std(tile, r * T, 1);
        row_min = row_min.min(v);
        row_max = row_max.max(v);
    }
    col_max / col_min.max(1e-8) + row_max / row_min.max(1e-8)
}

/// The result of the two-axis normalisation: `tile[r][c] / (s_col[c] * s_row[r])`.
pub struct Normalised {
    pub s_col: [f32; T],
    pub s_row: [f32; T],
    pub balanced: Vec<f32>,
}

/// `llama_kvarn_variance_normalize` with `iters` Sinkhorn iterations (16 in
/// kvarn): divide the columns, then the rows, by their sample standard
/// deviations (accumulated in the log domain, clamped to [-0.3, 10]); keep the
/// iterate (the identity included) with the least sum of the max/min spreads.
pub fn normalise(tile: &[f32], iters: usize) -> Normalised {
    assert_eq!(tile.len(), N);
    let mut log_s_col = [0.0f32; T];
    let mut log_s_row = [0.0f32; T];
    let mut cur = tile.to_vec();
    let mut s_col_best = [1.0f32; T];
    let mut s_row_best = [1.0f32; T];
    let mut best = imbalance(&cur);

    let rebuild = |cur: &mut [f32], log_s_col: &[f32; T], log_s_row: &[f32; T]| {
        let ec: Vec<f32> = log_s_col.iter().map(|l| l.exp()).collect();
        for r in 0..T {
            let s_row = log_s_row[r].exp();
            for c in 0..T {
                cur[r * T + c] = tile[r * T + c] / (ec[c] * s_row);
            }
        }
    };

    for _ in 0..iters {
        for c in 0..T {
            let std = sample_std(&cur, c, T).clamp(1e-3, 1e3);
            log_s_col[c] = (log_s_col[c] + std.ln()).clamp(-0.3, 10.0);
        }
        rebuild(&mut cur, &log_s_col, &log_s_row);
        for r in 0..T {
            let std = sample_std(&cur, r * T, 1).clamp(1e-3, 1e3);
            log_s_row[r] = (log_s_row[r] + std.ln()).clamp(-0.3, 10.0);
        }
        rebuild(&mut cur, &log_s_col, &log_s_row);
        let now = imbalance(&cur);
        if now <= best {
            best = now;
            for i in 0..T {
                s_col_best[i] = log_s_col[i].exp();
                s_row_best[i] = log_s_row[i].exp();
            }
        }
    }

    let mut balanced = vec![0.0f32; N];
    for r in 0..T {
        for c in 0..T {
            balanced[r * T + c] = tile[r * T + c] / (s_col_best[c] * s_row_best[r]);
        }
    }
    Normalised { s_col: s_col_best, s_row: s_row_best, balanced }
}

/// What kvarn stores for a tile: the codes and three f16 vectors.
pub struct Stored {
    pub codes: Vec<u8>,
    /// Per row: `s_row * grid step`, f16 bits.
    pub scale: [u16; T],
    /// Per row: `s_row * grid minimum`, f16 bits.
    pub zero: [u16; T],
    /// Per column: `s_col`, f16 bits.
    pub other: [u16; T],
}

/// `llama_kvarn_quantize_tile` after the normalisation: every row gets a grid
/// from its minimum to its maximum with `2^bits - 1` steps.
pub fn quantise(norm: &Normalised, bits: u32) -> Stored {
    let qmax = ((1u32 << bits) - 1) as i32;
    let mut codes = vec![0u8; N];
    let mut scale = [0u16; T];
    let mut zero = [0u16; T];
    let mut other = [0u16; T];
    for r in 0..T {
        let row = &norm.balanced[r * T..(r + 1) * T];
        let lo = row.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let step = ((hi - lo) / qmax as f32).max(1e-10);
        for c in 0..T {
            let v = ((row[c] - lo) / step).round();
            codes[r * T + c] = v.clamp(0.0, qmax as f32) as u8;
        }
        scale[r] = f32_to_f16(norm.s_row[r] * step);
        zero[r] = f32_to_f16(norm.s_row[r] * lo);
    }
    for i in 0..T {
        other[i] = f32_to_f16(norm.s_col[i]);
    }
    Stored { codes, scale, zero, other }
}

/// `llama_kvarn_dequantize_*_tile`: `(code * scale[row] + zero[row]) * other[col]`.
pub fn dequantise(s: &Stored) -> Vec<f32> {
    let mut out = vec![0.0f32; N];
    for r in 0..T {
        let scale = f16_to_f32(s.scale[r]);
        let zero = f16_to_f32(s.zero[r]);
        for c in 0..T {
            let other = f16_to_f32(s.other[c]);
            out[r * T + c] = (f32::from(s.codes[r * T + c]) * scale + zero) * other;
        }
    }
    out
}

/// Quantise and dequantise a tile.
pub fn roundtrip(tile: &[f32], bits: u32, iters: usize) -> Vec<f32> {
    dequantise(&quantise(&normalise(tile, iters), bits))
}
