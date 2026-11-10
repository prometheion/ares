//! The 4-bit float types: nibbles index the 16 E2M1 levels of
//! [`KVALUES_MXFP4`], scaled by a power of two (MXFP4) or by a small float
//! (NVFP4).
//!
//! Each function mirrors the `dequantize_row_*` of the same name in ggml's
//! `ggml-quants.c`.

use ares_core::GgmlType::{MXFP4, NVFP4};

use crate::blocks::{block_bytes, block_values, decode_blocks};
use crate::tables::KVALUES_MXFP4;

/// The scale of an MXFP4 block: `2^(e - 127) / 2` for the E8M0 byte `e` (ggml's
/// `ggml_e8m0_to_fp32_half`). The `/ 2` is there because the table holds the
/// E2M1 levels doubled. `e = 0` is the subnormal `2^-128`; `e = 255`, which the
/// OCP spec reserves for NaN, is treated as the large finite `2^127`.
fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits = if e < 2 { 0x0020_0000u32 << e } else { u32::from(e - 1) << 23 };
    f32::from_bits(bits)
}

/// The scale of an NVFP4 sub-block: the UE4M3 byte `x` (4 exponent bits with
/// bias 7, 3 mantissa bits) as a float, halved for the doubled table (ggml's
/// `ggml_ue4m3_to_fp32`). Zero and the NaN code `0x7f` give 0.
fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = u32::from((x >> 3) & 0xf);
    let man = f32::from(x & 0x7);
    let raw = if exp == 0 {
        // Subnormal: man * 2^-9.
        man * f32::from_bits(0x3b00_0000)
    } else {
        // (1 + man/8) * 2^(exp - 7); both factors are exact in f32.
        (1.0 + man / 8.0) * f32::from_bits((exp + 120) << 23)
    };
    raw * 0.5
}

/// MXFP4: `{ u8 e; u8 qs[16] }`; E8M0 scale, then nibbles laid out as in Q4_0.
pub(crate) fn mxfp4(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(MXFP4) }, { block_values(MXFP4) }>(src, dst, |b, y| {
        let d = e8m0_to_f32_half(b[0]);
        for j in 0..16 {
            let x0 = KVALUES_MXFP4[usize::from(b[1 + j] & 0x0f)];
            let x1 = KVALUES_MXFP4[usize::from(b[1 + j] >> 4)];
            y[j] = f32::from(x0) * d;
            y[j + 16] = f32::from(x1) * d;
        }
    });
}

/// NVFP4: `{ u8 d[4]; u8 qs[32] }`; four sub-blocks of 16 values, each with its
/// own UE4M3 scale and 8 bytes of nibbles (low nibbles the first 8 values,
/// high nibbles the next 8).
pub(crate) fn nvfp4(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(NVFP4) }, { block_values(NVFP4) }>(src, dst, |b, y| {
        for s in 0..4 {
            let d = ue4m3_to_f32(b[s]);
            for j in 0..8 {
                let byte = b[4 + 8 * s + j];
                y[16 * s + j] = f32::from(KVALUES_MXFP4[usize::from(byte & 0x0f)]) * d;
                y[16 * s + j + 8] = f32::from(KVALUES_MXFP4[usize::from(byte >> 4)]) * d;
            }
        }
    });
}
