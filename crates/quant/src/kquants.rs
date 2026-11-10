//! The K-quants: super-blocks of 256 values, split into 16 sub-blocks of 16
//! (Q2_K, Q3_K, Q6_K) or 8 sub-blocks of 32 (Q4_K, Q5_K), each sub-block with
//! its own small integer scale. Q8_K is the 8-bit sibling ggml uses for
//! activations.
//!
//! Each function mirrors the `dequantize_row_*` of the same name in ggml's
//! `ggml-quants.c`, including the order of the float operations.

use ares_core::GgmlType::{Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K};

use crate::blocks::{block_bytes, block_values, decode_blocks, f16_at, u32_at};

/// The 6-bit scale and 6-bit min of sub-block `j` (0..8) from the 12 packed
/// bytes of Q4_K and Q5_K (ggml's `get_scale_min_k4`). Sub-blocks 0..4 keep
/// their 6 bits in bytes 0..4 (scales) and 4..8 (mins); sub-blocks 4..8 split
/// theirs between a nibble of bytes 8..12 and the top two bits of bytes 0..8.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Q2_K: `{ u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }`. Sub-block scale in
/// the low nibble of `scales[i]`, sub-block min in the high one; 2-bit codes,
/// value = `d*scale*q - dmin*min`.
pub(crate) fn q2_k(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q2_K) }, { block_values(Q2_K) }>(src, dst, |b, y| {
        let d = f16_at(b, 80);
        let min = f16_at(b, 82);
        let mut is = 0;
        let mut out = 0;
        for half in 0..2 {
            let q = &b[16 + 32 * half..16 + 32 * half + 32];
            for step in 0..4 {
                let shift = 2 * step;
                // Each 2-bit plane holds two sub-blocks of 16, one per half of `q`.
                for part in 0..2 {
                    let sc = b[is];
                    is += 1;
                    let dl = d * f32::from(sc & 0xf);
                    let ml = min * f32::from(sc >> 4);
                    for l in 0..16 {
                        let code = (q[16 * part + l] >> shift) & 3;
                        y[out] = dl * f32::from(code) - ml;
                        out += 1;
                    }
                }
            }
        }
    });
}

/// Q3_K: `{ u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }`. 3-bit codes: two
/// low bits in `qs`, the high bit in `hmask` (clear means subtract 4). The 16
/// signed 6-bit scales (stored with an offset of 32) are packed into 12 bytes.
pub(crate) fn q3_k(src: &[u8], dst: &mut [f32]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    decode_blocks::<{ block_bytes(Q3_K) }, { block_values(Q3_K) }>(src, dst, |b, y| {
        let d_all = f16_at(b, 108);
        let hm = &b[..32];

        // Unpack the scales exactly like ggml: the low 4 bits of each come from
        // the first 8 bytes, the high 2 bits from the last 4.
        let mut aux = [u32_at(b, 96), u32_at(b, 100), u32_at(b, 104), 0];
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scales = [0u8; 16];
        for (word, bytes) in aux.iter().zip(scales.as_chunks_mut::<4>().0) {
            *bytes = word.to_le_bytes();
        }

        let mut is = 0;
        let mut out = 0;
        let mut m = 1u8;
        for half in 0..2 {
            let q = &b[32 + 32 * half..32 + 32 * half + 32];
            for step in 0..4 {
                let shift = 2 * step;
                for part in 0..2 {
                    let dl = d_all * (i32::from(scales[is]) - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let low = i32::from((q[16 * part + l] >> shift) & 3);
                        let high_clear = if hm[16 * part + l] & m != 0 { 0 } else { 4 };
                        y[out] = dl * (low - high_clear) as f32;
                        out += 1;
                    }
                }
                m <<= 1;
            }
        }
    });
}

/// Q4_K: `{ f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }`. Eight sub-blocks of
/// 32 with a 6-bit scale and min each; sub-block `2k` is the low nibbles of 32
/// bytes and `2k+1` the high nibbles of the same bytes. Value = `d*sc*q - dmin*m`.
pub(crate) fn q4_k(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q4_K) }, { block_values(Q4_K) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let min = f16_at(b, 2);
        let scales = &b[4..16];
        for pair in 0..4 {
            let q = &b[16 + 32 * pair..16 + 32 * pair + 32];
            let (sc, m) = scale_min_k4(2 * pair, scales);
            let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
            let (sc, m) = scale_min_k4(2 * pair + 1, scales);
            let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
            for l in 0..32 {
                y[64 * pair + l] = d1 * f32::from(q[l] & 0x0f) - m1;
                y[64 * pair + 32 + l] = d2 * f32::from(q[l] >> 4) - m2;
            }
        }
    });
}

/// Q5_K: `{ f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }`. Q4_K plus
/// a fifth bit per value in `qh`: byte `l` holds bit `l` of every 32-value
/// sub-block, sub-block `i` at bit `i`.
pub(crate) fn q5_k(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q5_K) }, { block_values(Q5_K) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let min = f16_at(b, 2);
        let scales = &b[4..16];
        let qh = &b[16..48];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for pair in 0..4 {
            let ql = &b[48 + 32 * pair..48 + 32 * pair + 32];
            let (sc, m) = scale_min_k4(2 * pair, scales);
            let (d1, m1) = (d * f32::from(sc), min * f32::from(m));
            let (sc, m) = scale_min_k4(2 * pair + 1, scales);
            let (d2, m2) = (d * f32::from(sc), min * f32::from(m));
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                y[64 * pair + l] = d1 * f32::from((ql[l] & 0x0f) + hi1) - m1;
                y[64 * pair + 32 + l] = d2 * f32::from((ql[l] >> 4) + hi2) - m2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    });
}

/// Q6_K: `{ u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }`. 6-bit codes (4 low
/// bits in `ql`, 2 high bits in `qh`) minus 32, times `d` times the signed
/// 8-bit scale of the 16-value sub-block.
pub(crate) fn q6_k(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q6_K) }, { block_values(Q6_K) }>(src, dst, |b, y| {
        let d = f16_at(b, 208);
        for half in 0..2 {
            let ql = &b[64 * half..64 * half + 64];
            let qh = &b[128 + 32 * half..128 + 32 * half + 32];
            let sc = &b[192 + 8 * half..192 + 8 * half + 8];
            let y = &mut y[128 * half..128 * half + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) - 32;
                let q2 = i32::from((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * f32::from(sc[is] as i8) * q1 as f32;
                y[l + 32] = d * f32::from(sc[is + 2] as i8) * q2 as f32;
                y[l + 64] = d * f32::from(sc[is + 4] as i8) * q3 as f32;
                y[l + 96] = d * f32::from(sc[is + 6] as i8) * q4 as f32;
            }
        }
    });
}

/// Q8_K: `{ f32 d; i8 qs[256]; i16 bsums[16] }`; value = `d*q`. The block sums
/// only serve ggml's dot products and are ignored. ggml uses this type for
/// activations, never in model files, and does not register a `to_float` for
/// it in its type table, but `dequantize_row_q8_K` exists and is what this
/// mirrors.
pub(crate) fn q8_k(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(Q8_K) }, { block_values(Q8_K) }>(src, dst, |b, y| {
        let d = f32::from_bits(u32_at(b, 0));
        for (v, &q) in y.iter_mut().zip(&b[4..260]) {
            *v = d * f32::from(q as i8);
        }
    });
}
