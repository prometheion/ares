//! The i-quants: values come from codebooks (a table of small vectors indexed by
//! bits of the block) instead of from a linear scale, so the tables in
//! [`crate::tables`] are part of the format.
//!
//! Each function mirrors the `dequantize_row_*` of the same name in ggml's
//! `ggml-quants.c`, including the order of the float operations.

use ares_core::GgmlType::{IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS};

use crate::blocks::{block_bytes, block_values, decode_blocks, f16_at, u16_at, u32_at};
use crate::half::f16_to_f32;
use crate::tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

/// `-1.0` if bit `j` of the sign byte is set, else `1.0`.
#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 }
}

/// The 8 magnitudes of a group of the 64-bit grids (IQ2_XXS, IQ2_XS, IQ2_S).
#[inline]
fn grid8(entry: u64) -> [u8; 8] {
    entry.to_le_bytes()
}

/// The 4 magnitudes of a group of the 32-bit grids (IQ3_XXS, IQ3_S).
#[inline]
fn grid4(entry: u32) -> [u8; 4] {
    entry.to_le_bytes()
}

/// IQ2_XXS: `{ f16 d; u16 qs[32] }`. Eight groups of 32 values; each takes two
/// `u32`s: four codebook indices (one byte each), then four 7-bit sign indices
/// and a 4-bit scale in the top bits.
pub(crate) fn iq2_xxs(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ2_XXS) }, { block_values(IQ2_XXS) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        for ib32 in 0..8 {
            let idx = &b[2 + 8 * ib32..2 + 8 * ib32 + 4];
            let aux1 = u32_at(b, 2 + 8 * ib32 + 4);
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = grid8(IQ2XXS_GRID[usize::from(idx[l])]);
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    y[32 * ib32 + 8 * l + j] = db * f32::from(grid[j]) * sign(signs, j);
                }
            }
        }
    });
}

/// IQ2_XS: `{ f16 d; u16 qs[32]; u8 scales[8] }`. Each `u16` of `qs` is a 9-bit
/// codebook index and a 7-bit sign index; each scale byte holds two 4-bit
/// scales, one per 16 values.
pub(crate) fn iq2_xs(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ2_XS) }, { block_values(IQ2_XS) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        for ib32 in 0..8 {
            let scale = b[66 + ib32];
            let db = [
                d * (0.5 + f32::from(scale & 0xf)) * 0.25,
                d * (0.5 + f32::from(scale >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let q = u16_at(b, 2 + 2 * (4 * ib32 + l));
                let grid = grid8(IQ2XS_GRID[usize::from(q & 511)]);
                let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
                for j in 0..8 {
                    y[32 * ib32 + 8 * l + j] = db[l / 2] * f32::from(grid[j]) * sign(signs, j);
                }
            }
        }
    });
}

/// IQ2_S: `{ f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] }`. The first 32 bytes of
/// `qs` are the low 8 bits of the 10-bit codebook indices (the top 2 bits are in
/// `qh`), the last 32 are plain sign bytes; scales as in IQ2_XS.
pub(crate) fn iq2_s(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ2_S) }, { block_values(IQ2_S) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let qs = &b[2..66];
        let qh = &b[66..74];
        let scales = &b[74..82];
        for ib32 in 0..8 {
            let db = [
                d * (0.5 + f32::from(scales[ib32] & 0xf)) * 0.25,
                d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let dl = db[l / 2];
                let high = (usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300;
                let grid = grid8(IQ2S_GRID[usize::from(qs[4 * ib32 + l]) | high]);
                let signs = qs[32 + 4 * ib32 + l];
                for j in 0..8 {
                    y[32 * ib32 + 8 * l + j] = dl * f32::from(grid[j]) * sign(signs, j);
                }
            }
        }
    });
}

/// IQ3_XXS: `{ f16 d; u8 qs[96] }`. The first 64 bytes are codebook indices, two
/// per group of 8 values; the last 32 are one `u32` per 32 values: four 7-bit
/// sign indices and a 4-bit scale in the top bits.
pub(crate) fn iq3_xxs(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ3_XXS) }, { block_values(IQ3_XXS) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let qs = &b[2..66];
        for ib32 in 0..8 {
            let aux = u32_at(b, 66 + 4 * ib32);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let grid1 = grid4(IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])]);
                let grid2 = grid4(IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])]);
                for j in 0..4 {
                    let at = 32 * ib32 + 8 * l;
                    y[at + j] = db * f32::from(grid1[j]) * sign(signs, j);
                    y[at + j + 4] = db * f32::from(grid2[j]) * sign(signs, j + 4);
                }
            }
        }
    });
}

/// IQ3_S: `{ f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] }`. A 9-bit
/// codebook index per 4 values (8 bits in `qs`, the ninth in `qh`), plain sign
/// bytes, and one 4-bit odd scale `1 + 2*s` per 32 values.
pub(crate) fn iq3_s(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ3_S) }, { block_values(IQ3_S) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let qs = &b[2..66];
        let qh = &b[66..74];
        let signs = &b[74..106];
        let scales = &b[106..110];
        for ib32 in 0..8 {
            let scale = (scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf;
            let db = d * (1 + 2 * i32::from(scale)) as f32;
            for l in 0..4 {
                let hi1 = (usize::from(qh[ib32]) << (8 - 2 * l)) & 256;
                let hi2 = (usize::from(qh[ib32]) << (7 - 2 * l)) & 256;
                let grid1 = grid4(IQ3S_GRID[usize::from(qs[8 * ib32 + 2 * l]) | hi1]);
                let grid2 = grid4(IQ3S_GRID[usize::from(qs[8 * ib32 + 2 * l + 1]) | hi2]);
                let sg = signs[4 * ib32 + l];
                for j in 0..4 {
                    let at = 32 * ib32 + 8 * l;
                    y[at + j] = db * f32::from(grid1[j]) * sign(sg, j);
                    y[at + j + 4] = db * f32::from(grid2[j]) * sign(sg, j + 4);
                }
            }
        }
    });
}

/// How far IQ1_S and IQ1_M shift every value of a group towards the sign of its
/// delta bit.
const IQ1_DELTA: f32 = 0.125;

/// IQ1_S: `{ f16 d; u8 qs[32]; u16 qh[8] }`. Per 32 values one `u16` of `qh`:
/// four 3-bit high parts of 11-bit codebook indices (low 8 bits in `qs`), a
/// 3-bit scale at bits 12..15 (odd scale `2*s + 1`) and the sign of the delta at
/// bit 15.
pub(crate) fn iq1_s(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ1_S) }, { block_values(IQ1_S) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let qs = &b[2..34];
        for ib in 0..8 {
            let qh = u16_at(b, 34 + 2 * ib);
            let dl = d * (2 * i32::from((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 { -IQ1_DELTA } else { IQ1_DELTA };
            for l in 0..4 {
                let index = usize::from(qs[4 * ib + l]) | (usize::from((qh >> (3 * l)) & 7) << 8);
                let grid = IQ1S_GRID[index].to_le_bytes();
                for j in 0..8 {
                    y[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta);
                }
            }
        }
    });
}

/// IQ1_M: `{ u8 qs[32]; u8 qh[16]; u8 scales[8] }`. Like IQ1_S, but the block has
/// no scale of its own: the f16 `d` is spread over the top nibbles of the four
/// `u16`s of `scales`, whose other 12 bits hold sixteen 3-bit scales (one per
/// 16 values). `qh` carries a 3-bit index part and a delta sign per 8 values.
pub(crate) fn iq1_m(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ1_M) }, { block_values(IQ1_M) }>(src, dst, |b, y| {
        let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
        let scale =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16_to_f32(scale);
        let qs = &b[..32];
        let qh = &b[32..48];
        for ib in 0..8 {
            let shift = 6 * (ib % 2);
            let dl1 = d * (2 * i32::from((sc[ib / 2] >> shift) & 7) + 1) as f32;
            let dl2 = d * (2 * i32::from((sc[ib / 2] >> (shift + 3)) & 7) + 1) as f32;
            let (h0, h1) = (usize::from(qh[2 * ib]), usize::from(qh[2 * ib + 1]));
            let idx = [
                usize::from(qs[4 * ib]) | ((h0 << 8) & 0x700),
                usize::from(qs[4 * ib + 1]) | ((h0 << 4) & 0x700),
                usize::from(qs[4 * ib + 2]) | ((h1 << 8) & 0x700),
                usize::from(qs[4 * ib + 3]) | ((h1 << 4) & 0x700),
            ];
            let delta = [
                if h0 & 0x08 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
                if h0 & 0x80 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
                if h1 & 0x08 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
                if h1 & 0x80 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
            ];
            for l in 0..4 {
                let dl = if l < 2 { dl1 } else { dl2 };
                let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                for j in 0..8 {
                    y[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta[l]);
                }
            }
        }
    });
}

/// IQ4_NL: `{ f16 d; u8 qs[16] }`; nibbles index the 16 non-linear levels,
/// layout as Q4_0.
pub(crate) fn iq4_nl(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ4_NL) }, { block_values(IQ4_NL) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        for j in 0..16 {
            y[j] = d * f32::from(KVALUES_IQ4NL[usize::from(b[2 + j] & 0xf)]);
            y[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(b[2 + j] >> 4)]);
        }
    });
}

/// IQ4_XS: `{ f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }`. IQ4_NL's levels
/// with one 6-bit scale (minus 32) per 32 values: the low 4 bits in
/// `scales_l`, the high 2 in `scales_h`.
pub(crate) fn iq4_xs(src: &[u8], dst: &mut [f32]) {
    decode_blocks::<{ block_bytes(IQ4_XS) }, { block_values(IQ4_XS) }>(src, dst, |b, y| {
        let d = f16_at(b, 0);
        let scales_h = u16_at(b, 2);
        for ib in 0..8 {
            let low = i32::from((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf);
            let high = i32::from((scales_h >> (2 * ib)) & 3);
            let dl = d * ((low | (high << 4)) - 32) as f32;
            let qs = &b[8 + 16 * ib..8 + 16 * ib + 16];
            for j in 0..16 {
                y[32 * ib + j] = dl * f32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0xf)]);
                y[32 * ib + j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)]);
            }
        }
    });
}
