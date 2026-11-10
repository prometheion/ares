//! The encoders for the types the KV cache uses.
//!
//! Each function mirrors the `quantize_row_*_ref` of the same name in ggml's
//! `ggml-quants.c` operation by operation, because the goal is the same bytes,
//! not merely a good approximation: the comparison and the float-to-integer
//! conversions below look odd on purpose.
//!
//! Inputs must be finite. ggml converts floats to small integers with C casts,
//! which are undefined for NaN and out-of-range values, so a NaN or infinite
//! input has no reference result to match (a block's scale then becomes NaN or
//! infinity and every value derived from it is meaningless anyway).

use ares_core::GgmlType::{Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};

use crate::blocks::{block_bytes, block_values, encode_blocks};
use crate::half::f32_to_f16;

/// The value of `x` with the largest magnitude, sign included. The first of
/// equal magnitudes wins, which decides the sign of the scale.
fn signed_abs_max(x: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    max
}

/// The smallest and largest value of `x`.
fn min_max(x: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = -f32::MAX;
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

/// `1/d`, or 0 for a zero scale (so an all-zero block quantizes to the zero
/// code instead of NaN).
fn reciprocal(d: f32) -> f32 {
    if d != 0.0 { 1.0 / d } else { 0.0 }
}

/// ggml's `MIN(limit, (int8_t)(v))` stored in a `uint8_t`: truncates towards
/// zero, caps at `limit` and keeps the low 8 bits.
fn capped(v: f32, limit: i8) -> u8 {
    (v as i8).min(limit) as u8
}

/// Q4_0: scale `d = max / -8` where `max` is the signed value of largest
/// magnitude, so that value maps exactly to code 0 (the lowest level, -8).
pub(crate) fn q4_0(src: &[f32], dst: &mut [u8]) {
    encode_blocks::<{ block_bytes(Q4_0) }, { block_values(Q4_0) }>(src, dst, |x, y| {
        let max = signed_abs_max(x);
        let d = max / -8.0;
        let id = reciprocal(d);
        y[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            let xi0 = capped(x[j] * id + 8.5, 15);
            let xi1 = capped(x[16 + j] * id + 8.5, 15);
            y[2 + j] = xi0 | (xi1 << 4);
        }
    });
}

/// Q4_1: scale `d = (max - min) / 15`, offset `min`; codes are `(x - min) / d`
/// rounded to nearest. The offset is stored as f16 but the *unrounded* `min` is
/// what the codes are computed against.
pub(crate) fn q4_1(src: &[f32], dst: &mut [u8]) {
    encode_blocks::<{ block_bytes(Q4_1) }, { block_values(Q4_1) }>(src, dst, |x, y| {
        let (min, max) = min_max(x);
        let d = (max - min) / 15.0;
        let id = reciprocal(d);
        y[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        y[2..4].copy_from_slice(&f32_to_f16(min).to_le_bytes());
        for j in 0..16 {
            let xi0 = capped((x[j] - min) * id + 0.5, 15);
            let xi1 = capped((x[16 + j] - min) * id + 0.5, 15);
            y[4 + j] = xi0 | (xi1 << 4);
        }
    });
}

/// Q5_0: like Q4_0 with `d = max / -16` and a fifth code bit per value
/// collected into `qh`.
pub(crate) fn q5_0(src: &[f32], dst: &mut [u8]) {
    encode_blocks::<{ block_bytes(Q5_0) }, { block_values(Q5_0) }>(src, dst, |x, y| {
        let max = signed_abs_max(x);
        let d = max / -16.0;
        let id = reciprocal(d);
        y[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        let mut qh = 0u32;
        for j in 0..16 {
            let xi0 = capped(x[j] * id + 16.5, 31);
            let xi1 = capped(x[16 + j] * id + 16.5, 31);
            y[6 + j] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            qh |= u32::from((xi0 & 0x10) >> 4) << j;
            qh |= u32::from((xi1 & 0x10) >> 4) << (j + 16);
        }
        y[2..6].copy_from_slice(&qh.to_le_bytes());
    });
}

/// Q5_1: like Q4_1 with `d = (max - min) / 31` and a fifth code bit per value.
/// Unlike Q4_0, Q4_1 and Q5_0, ggml converts straight to `u8` without capping
/// the code (it cannot reach 32 for finite input anyway).
pub(crate) fn q5_1(src: &[f32], dst: &mut [u8]) {
    encode_blocks::<{ block_bytes(Q5_1) }, { block_values(Q5_1) }>(src, dst, |x, y| {
        let (min, max) = min_max(x);
        let d = (max - min) / 31.0;
        let id = reciprocal(d);
        y[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        y[2..4].copy_from_slice(&f32_to_f16(min).to_le_bytes());
        let mut qh = 0u32;
        for j in 0..16 {
            let xi0 = ((x[j] - min) * id + 0.5) as u8;
            let xi1 = ((x[16 + j] - min) * id + 0.5) as u8;
            y[8 + j] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
            qh |= u32::from((xi0 & 0x10) >> 4) << j;
            qh |= u32::from((xi1 & 0x10) >> 4) << (j + 16);
        }
        y[4..8].copy_from_slice(&qh.to_le_bytes());
    });
}

/// Q8_0: scale `d = amax / 127`; codes are `x / d` rounded half away from zero.
pub(crate) fn q8_0(src: &[f32], dst: &mut [u8]) {
    encode_blocks::<{ block_bytes(Q8_0) }, { block_values(Q8_0) }>(src, dst, |x, y| {
        let mut amax = 0.0f32;
        for &v in x.iter() {
            // ggml's MAX(a, b), which differs from `f32::max` only for NaN.
            let a = v.abs();
            amax = if amax > a { amax } else { a };
        }
        let d = amax / 127.0;
        let id = reciprocal(d);
        y[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (code, &v) in y[2..].iter_mut().zip(x.iter()) {
            *code = (v * id).round() as i8 as u8;
        }
    });
}
