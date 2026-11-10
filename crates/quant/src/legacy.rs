//! The plain block formats: one f16 scale (and for the `_1` types one offset)
//! per block of 32 values, or 64 and 128 for the 2-bit and 1-bit types.
//!
//! Each function mirrors the `dequantize_row_*` of the same name in ggml's
//! `ggml-quants.c`, including the order of the float operations (`x0*d + m`,
//! never a fused multiply-add), so the results are bit-identical.

use ares_core::GgmlType::{Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};

use crate::blocks::{block_bytes, block_values, decode_blocks, f16_at, u32_at};

/// Q1_0: `{ f16 d; u8 qs[16] }`. Bit `j` (least significant first) of `qs` is
/// the sign of value `j`: set is `+d`, clear is `-d`.
pub(crate) fn q1_0(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q1_0) }, { block_values(Q1_0) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let neg_d = -d;
        for (j, v) in y.iter_mut().enumerate() {
            let bit = (b[2 + j / 8] >> (j % 8)) & 1;
            *v = if bit != 0 { d } else { neg_d };
        }
    });
}

/// Q2_0: `{ f16 d; u8 qs[16] }`, four 2-bit codes per byte, the first value in
/// the low bits. Codes 0..=3 mean -1, 0, +1, +2 times `d`.
pub(crate) fn q2_0(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q2_0) }, { block_values(Q2_0) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        for (j, v) in y.iter_mut().enumerate() {
            let q = (b[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
            *v = (i32::from(q) - 1) as f32 * d;
        }
    });
}

/// Q4_0: `{ f16 d; u8 qs[16] }`. Low nibbles are values 0..16, high nibbles
/// values 16..32, each minus 8, times `d`.
pub(crate) fn q4_0(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q4_0) }, { block_values(Q4_0) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        for j in 0..16 {
            let x0 = i32::from(b[2 + j] & 0x0f) - 8;
            let x1 = i32::from(b[2 + j] >> 4) - 8;
            y[j] = x0 as f32 * d;
            y[j + 16] = x1 as f32 * d;
        }
    });
}

/// Q4_1: `{ f16 d; f16 m; u8 qs[16] }`; value = `q*d + m`, nibbles laid out
/// as in Q4_0.
pub(crate) fn q4_1(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q4_1) }, { block_values(Q4_1) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let m = f16_at(b, 2);
        for j in 0..16 {
            let x0 = i32::from(b[4 + j] & 0x0f);
            let x1 = i32::from(b[4 + j] >> 4);
            y[j] = x0 as f32 * d + m;
            y[j + 16] = x1 as f32 * d + m;
        }
    });
}

/// Q5_0: `{ f16 d; u32 qh; u8 qs[16] }`. Like Q4_0, with a fifth bit per value
/// in `qh`: bit `j` for value `j`, bit `j + 16` for value `j + 16`.
pub(crate) fn q5_0(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q5_0) }, { block_values(Q5_0) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let qh = u32_at(b, 2);
        for j in 0..16 {
            // The fifth bit lands on bit 4 (0x10) of the 5-bit code.
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = i32::from((b[6 + j] & 0x0f) | xh_0) - 16;
            let x1 = i32::from((b[6 + j] >> 4) | xh_1) - 16;
            y[j] = x0 as f32 * d;
            y[j + 16] = x1 as f32 * d;
        }
    });
}

/// Q5_1: `{ f16 d; f16 m; u32 qh; u8 qs[16] }`; Q5_0's layout with Q4_1's
/// `q*d + m`.
pub(crate) fn q5_1(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q5_1) }, { block_values(Q5_1) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let m = f16_at(b, 2);
        let qh = u32_at(b, 4);
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = i32::from((b[8 + j] & 0x0f) | xh_0);
            let x1 = i32::from((b[8 + j] >> 4) | xh_1);
            y[j] = x0 as f32 * d + m;
            y[j + 16] = x1 as f32 * d + m;
        }
    });
}

/// Q8_0: `{ f16 d; i8 qs[32] }`; value = `q*d`.
pub(crate) fn q8_0(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q8_0) }, { block_values(Q8_0) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        for (v, &q) in y.iter_mut().zip(&b[2..]) {
            *v = f32::from(q as i8) * d;
        }
    });
}
