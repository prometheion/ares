//! The ternary types: every value is -1, 0 or +1 times one f16 scale per block
//! of 256.
//!
//! Each function mirrors the `dequantize_row_*` of the same name in ggml's
//! `ggml-quants.c`.

use ares_core::GgmlType::{TQ1_0, TQ2_0};

use crate::blocks::{block_bytes, block_values, decode_blocks, f16_at};

/// The powers of three TQ1_0 multiplies a byte by to expose one base-3 digit
/// (ggml's table also lists 3^5, which nothing reads: a byte holds five digits).
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// One base-3 digit of a TQ1_0 byte, as -1, 0 or 1.
///
/// Five digits are packed in a byte as a number below 243, most significant
/// digit first. Multiplying by `3^n` (modulo 256) moves digit `n` to the top
/// of the byte's fixed-point fraction, and `* 3 >> 8` reads that digit out
/// as 0, 1 or 2.
#[inline]
fn tq1_digit(byte: u8, n: usize) -> i32 {
    let q = byte.wrapping_mul(POW3[n]);
    ((u16::from(q) * 3) >> 8) as i32 - 1
}

/// TQ1_0: `{ u8 qs[48]; u8 qh[4]; f16 d }`. 240 values in the 48 bytes of `qs`
/// (five per byte) and the last 16 in `qh` (four per byte); `qs` is read as one
/// run of 32 bytes (160 values) and one of 16 bytes (80 values), digit-plane by
/// digit-plane.
pub(crate) fn tq1_0(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(TQ1_0) }, { block_values(TQ1_0) }>(src, dst, |b, y| {
        let d = f16_at(b, 52);
        let mut out = 0;
        // (first byte, bytes in the run) of the two runs of `qs`.
        for (start, len) in [(0, 32), (32, 16)] {
            for n in 0..5 {
                for m in 0..len {
                    y[out] = tq1_digit(b[start + m], n) as f32 * d;
                    out += 1;
                }
            }
        }
        for n in 0..4 {
            for j in 0..4 {
                y[out] = tq1_digit(b[48 + j], n) as f32 * d;
                out += 1;
            }
        }
    });
}

/// TQ2_0: `{ u8 qs[64]; f16 d }`. Four 2-bit codes per byte (0, 1, 2 for -1, 0,
/// +1; 3 is not produced by ggml's quantizer and decodes to +2), taken in
/// planes: for each run of 32 bytes, all the low codes, then the next ones.
pub(crate) fn tq2_0(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(TQ2_0) }, { block_values(TQ2_0) }>(src, dst, |b, y| {
        let d = f16_at(b, 64);
        let mut out = 0;
        for start in [0, 32] {
            for l in 0..4 {
                for m in 0..32 {
                    let q = i32::from((b[start + m] >> (2 * l)) & 3);
                    y[out] = (q - 1) as f32 * d;
                    out += 1;
                }
            }
        }
    });
}
