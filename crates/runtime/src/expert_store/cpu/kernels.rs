//! The products of an expert on the CPU, on the file's quantized bytes.
//!
//! The arithmetic is that of the device kernels (`kernels/expert_ffn.cuh`):
//! the activations are the int8 codes with a scale per 32 values that
//! `ares_quantize_act` made, a product is an exact integer sum per 32 values
//! times the two scales, and the hidden values `silu(gate) * up` are
//! quantized the same way before the down product. What differs is the order
//! of the float additions, so the results agree to rounding, not to the bit.
//!
//! Everything here needs AVX2, FMA and F16C (x86-64-v3, checked by
//! [`super::Features::detect`] before a kernel is ever called); the down
//! product has a second form for AVX-VNNI.
//!
//! - **Gate and up** (IQ2_S, IQ2_XXS, IQ1_M): row `j` of both matrices is
//!   multiplied with the same activations in one pass. For every 32 values
//!   the weights are decoded once (codebook magnitudes by a gather, signs,
//!   the scales of the two halves), and each row of the step that uses the
//!   expert then costs five instructions per matrix: the signs go onto the
//!   activation codes, `maddubs` with the magnitudes, `madd` with the scales,
//!   and the sum joins a float accumulator times the activation's scale.
//! - **Down** (Q2_0): a block is 64 two-bit codes in 16 bytes, value `i` in
//!   bits `2 (i % 4)` of byte `i / 4`. Plane `k` (the bits `2 k` of every
//!   byte) holds the values `4 i + k`, so the hidden codes are stored in that
//!   order once ([`store_hidden`]) and a plane multiplies them without the
//!   codes being unpacked. The code means `c - 1`; the `-1` is the sum of the
//!   hidden codes, taken off in integers.
//! - **Every other weight type**, for gate and up and for down
//!   ([`GateKind::Other`], [`DownKind::Other`]): a row of the matrix is decoded
//!   to `f32` by the engine's reference decoder (`ares_quant`, which is ggml's
//!   bit for bit) and multiplied with the same rounded activations (code times
//!   scale) in `f32`. That is the product the device kernels of the type
//!   compute, `W x^`, summed in another order and without their exact integer
//!   sums per 32 values, so the two agree to `f32` rounding as the written
//!   kernels do. A row is decoded once for all the rows of the step that use
//!   the expert; the decoding is what such an expert costs (the types of the
//!   model the engine was built on have the kernels above because they are the
//!   ones a miss was measured on).

use std::arch::x86_64::*;

use ares_core::GgmlType;

use super::tables::{Lanes16, Tables};

/// One row of a step as the gate and up kernels read it.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowIn {
    /// The row's `n_in` activation codes.
    pub q: *const i8,
    /// Its `(d, s)` pairs, two `f32` per 32 values, at any alignment.
    pub ds: *const u8,
    /// For IQ1_M: per 32 values, the sums of the four groups of 8 codes in
    /// lanes 0, 4, 8 and 12 ([`sum_groups`]).
    pub sum8: *const Lanes16,
}

/// The quantized hidden values of one (expert, row) pair as the down kernel
/// reads them and [`store_hidden`] writes them.
#[derive(Clone, Copy, Debug)]
pub(super) struct Hidden {
    /// The codes in plane order: value `128 p + 64 h + 4 i + k` is at
    /// `128 p + 32 k + 16 h + i`.
    pub q: *mut i8,
    /// The sum of the codes of every 16 values.
    pub sum: *mut i32,
    /// The scale of the codes of every 16 values (that of their 32).
    pub scale: *mut f32,
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
}

/// What 32 weights decode to: magnitudes as unsigned bytes, a byte per value
/// that is negative where the weight is (and never zero), and the integer
/// scale of every pair of values as `i16`.
type Decoded = (__m256i, __m256i, __m256i);

/// One row's part of 32 values: `acc + da * sum(scale * magnitude * sign * q)`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn dot32(w: Decoded, q: __m256i, da: __m256, acc: __m256) -> __m256 {
    let pairs = _mm256_maddubs_epi16(w.0, _mm256_sign_epi8(q, w.1));
    let sums = _mm256_madd_epi16(pairs, w.2);
    _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), da, acc)
}

/// IQ2_S, values `32 ib ..` of the block at `blk`:
/// `{ f16 d; u8 index[32]; u8 signs[32]; u8 high[8]; u8 scales[8] }`.
///
/// # Safety
///
/// `blk` must address a whole block (82 bytes) and `ib` be below 8.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_iq2_s(t: &Tables, blk: *const u8, ib: usize) -> Decoded {
    // SAFETY: every read is inside the block the caller vouches for, and a
    // 10-bit index is inside the table of 1024 entries.
    unsafe {
        let low = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
            blk.add(2 + 4 * ib).cast::<i32>().read_unaligned(),
        ));
        let high = _mm_sllv_epi32(
            _mm_set1_epi32(i32::from(*blk.add(66 + ib))),
            _mm_setr_epi32(8, 6, 4, 2),
        );
        let index = _mm_or_si128(low, _mm_and_si128(high, _mm_set1_epi32(0x300)));
        let grid = _mm256_i32gather_epi64::<8>(t.iq2s_grid.as_ptr().cast(), index);
        // Sign byte l covers values 8 l ..: spread each to 8 bytes and test
        // one bit in each.
        let signs = _mm256_set1_epi32(blk.add(34 + 4 * ib).cast::<i32>().read_unaligned());
        let spread = _mm256_shuffle_epi8(
            signs,
            _mm256_setr_epi64x(
                0,
                0x0101_0101_0101_0101,
                0x0202_0202_0202_0202,
                0x0303_0303_0303_0303,
            ),
        );
        let bit = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
        let negative = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit), bit);
        let scales =
            _mm256_load_si256(t.iq2s_scales.as_ptr().add(usize::from(*blk.add(74 + ib))).cast());
        (grid, _mm256_or_si256(negative, _mm256_set1_epi8(1)), scales)
    }
}

/// IQ2_XXS, values `32 ib ..` of the block at `blk`:
/// `{ f16 d; 8 x { u8 index[4]; u32: four 7-bit sign indices, 4-bit scale } }`.
///
/// # Safety
///
/// `blk` must address a whole block (66 bytes) and `ib` be below 8.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_iq2_xxs(t: &Tables, blk: *const u8, ib: usize) -> Decoded {
    // SAFETY: the reads are inside the block; an 8-bit index is inside the
    // grid of 256 entries and a 7-bit one inside the 128 sign entries.
    unsafe {
        let index = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
            blk.add(2 + 8 * ib).cast::<i32>().read_unaligned(),
        ));
        let grid = _mm256_i32gather_epi64::<8>(t.iq2xxs_grid.as_ptr().cast(), index);
        let aux = blk.add(6 + 8 * ib).cast::<u32>().read_unaligned();
        let sign_index = _mm_and_si128(
            _mm_srlv_epi32(_mm_set1_epi32(aux as i32), _mm_setr_epi32(0, 7, 14, 21)),
            _mm_set1_epi32(127),
        );
        let negative = _mm256_i32gather_epi64::<8>(t.iq2xxs_signs.as_ptr().cast(), sign_index);
        let scales = _mm256_set1_epi16((2 * (aux >> 28) + 1) as i16);
        (grid, _mm256_or_si256(negative, _mm256_set1_epi8(1)), scales)
    }
}

/// Four codebook entries by four loads, where a gather is slow (the
/// efficiency cores): entry `l` goes to values `8 l ..`.
///
/// # Safety
///
/// Every 16-bit field of `index` must be inside the table at `table`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load4(table: *const u64, index: u64) -> __m256i {
    // SAFETY: the caller's guarantee for the four entries.
    unsafe {
        let low = _mm_unpacklo_epi64(
            _mm_loadl_epi64(table.add((index & 0xffff) as usize).cast()),
            _mm_loadl_epi64(table.add((index >> 16 & 0xffff) as usize).cast()),
        );
        let high = _mm_unpacklo_epi64(
            _mm_loadl_epi64(table.add((index >> 32 & 0xffff) as usize).cast()),
            _mm_loadl_epi64(table.add((index >> 48) as usize).cast()),
        );
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high)
    }
}

/// [`decode_iq2_s`] with loads instead of a gather.
///
/// # Safety
///
/// As [`decode_iq2_s`].
#[inline]
#[target_feature(enable = "avx2,fma,bmi2")]
unsafe fn decode_iq2_s_loads(t: &Tables, blk: *const u8, ib: usize) -> Decoded {
    // SAFETY: as in the gather form; the four indices have 10 bits each.
    unsafe {
        let low = u64::from(blk.add(2 + 4 * ib).cast::<u32>().read_unaligned());
        let high = u64::from(*blk.add(66 + ib));
        let index = _pdep_u64(low, 0x00ff_00ff_00ff_00ff) | _pdep_u64(high, 0x0300_0300_0300_0300);
        let grid = load4(t.iq2s_grid.as_ptr(), index);
        let signs = _mm256_set1_epi32(blk.add(34 + 4 * ib).cast::<i32>().read_unaligned());
        let spread = _mm256_shuffle_epi8(
            signs,
            _mm256_setr_epi64x(
                0,
                0x0101_0101_0101_0101,
                0x0202_0202_0202_0202,
                0x0303_0303_0303_0303,
            ),
        );
        let bit = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
        let negative = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit), bit);
        let scales =
            _mm256_load_si256(t.iq2s_scales.as_ptr().add(usize::from(*blk.add(74 + ib))).cast());
        (grid, _mm256_or_si256(negative, _mm256_set1_epi8(1)), scales)
    }
}

/// [`decode_iq2_xxs`] with loads instead of gathers.
///
/// # Safety
///
/// As [`decode_iq2_xxs`].
#[inline]
#[target_feature(enable = "avx2,fma,bmi2")]
unsafe fn decode_iq2_xxs_loads(t: &Tables, blk: *const u8, ib: usize) -> Decoded {
    // SAFETY: as in the gather form; indices of 8 and of 7 bits.
    unsafe {
        let index = u64::from(blk.add(2 + 8 * ib).cast::<u32>().read_unaligned());
        let grid = load4(t.iq2xxs_grid.as_ptr(), _pdep_u64(index, 0x00ff_00ff_00ff_00ff));
        let aux = blk.add(6 + 8 * ib).cast::<u32>().read_unaligned();
        let negative =
            load4(t.iq2xxs_signs.as_ptr(), _pdep_u64(u64::from(aux), 0x007f_007f_007f_007f));
        let scales = _mm256_set1_epi16((2 * (aux >> 28) + 1) as i16);
        (grid, _mm256_or_si256(negative, _mm256_set1_epi8(1)), scales)
    }
}

/// [`decode_iq1_m`] with loads instead of a gather.
///
/// # Safety
///
/// As [`decode_iq1_m`].
#[inline]
#[target_feature(enable = "avx2,fma,bmi2")]
unsafe fn decode_iq1_m_loads(t: &Tables, blk: *const u8, ib: usize) -> Decoded {
    // SAFETY: as in the gather form; the four indices have 11 bits each.
    unsafe {
        let low = u64::from(blk.add(4 * ib).cast::<u32>().read_unaligned());
        let qh = u32::from(blk.add(32 + 2 * ib).cast::<u16>().read_unaligned());
        let high = _pext_u64(u64::from(qh), 0x7777);
        let index = _pdep_u64(low, 0x00ff_00ff_00ff_00ff) | _pdep_u64(high, 0x0700_0700_0700_0700);
        let grid = load4(t.iq1_grid.as_ptr(), index);
        let word = u32::from(blk.add(48 + 2 * (ib / 2)).cast::<u16>().read_unaligned());
        let scale_bits = (word >> (6 * (ib % 2))) & 63;
        let scales = _mm256_load_si256(t.iq1_scales.as_ptr().add(scale_bits as usize).cast());
        let delta_bits = (qh >> 3 & 1) | (qh >> 6 & 2) | (qh >> 9 & 4) | (qh >> 12 & 8);
        let deltas = _mm256_load_si256(t.iq1_deltas.as_ptr().add(delta_bits as usize).cast());
        let delta_scales = _mm256_sign_epi16(_mm256_srai_epi16::<3>(scales), deltas);
        (grid, scales, delta_scales)
    }
}

/// The f16 at `at` as `f32`.
///
/// # Safety
///
/// `at` must address two readable bytes.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f16_at(at: *const u8) -> f32 {
    // SAFETY: the caller's guarantee.
    let bits = unsafe { at.cast::<u16>().read_unaligned() };
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits))))
}

/// The scale of an IQ2 block times the eighth that the integer scales
/// `2 s + 1` leave out (`d * (0.5 + s) * 0.25`).
///
/// # Safety
///
/// `blk` must address a block.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq2_block_scale(blk: *const u8) -> f32 {
    // SAFETY: the block starts with its f16 scale.
    unsafe { f16_at(blk) * 0.125 }
}

/// The scale of an IQ1_M block, an f16 spread over the top nibbles of its
/// four scale words, times the 1/8 by which the kernel's integers are
/// larger (it computes `8 g + delta sign` for `g + delta / 8`).
///
/// # Safety
///
/// `blk` must address a whole block (56 bytes).
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn iq1_block_scale(blk: *const u8) -> f32 {
    // SAFETY: the scale words are bytes 48..56 of the block.
    let sc: [u16; 4] = unsafe { blk.add(48).cast::<[u16; 4]>().read_unaligned() };
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits)))) * 0.125
}

/// Defines the gate and up product of one hidden row for an IQ2 type:
/// `out[r] = [gate_row . x_r, up_row . x_r]` for `R` rows of the step.
macro_rules! iq2_gate_up_row {
    ($name:ident, $block_bytes:expr, $decode:ident) => {
        /// # Safety
        ///
        /// `gate` and `up` must address `blocks` whole blocks of the type,
        /// and every row `256 * blocks` codes and `8 * blocks` scale pairs.
        #[target_feature(enable = "avx2,fma,f16c,bmi2")]
        unsafe fn $name<const R: usize>(
            t: &Tables,
            gate: *const u8,
            up: *const u8,
            blocks: usize,
            rows: &[RowIn; R],
            out: &mut [[f32; 2]; R],
        ) {
            let mut acc_g = [_mm256_setzero_ps(); R];
            let mut acc_u = [_mm256_setzero_ps(); R];
            for b in 0..blocks {
                // SAFETY: block b of both rows, and the 256 codes and 8
                // scales of every activation row that belong to it, are
                // inside what the caller vouches for.
                unsafe {
                    let (gb, ub) = (gate.add($block_bytes * b), up.add($block_bytes * b));
                    let mut blk_g = [_mm256_setzero_ps(); R];
                    let mut blk_u = [_mm256_setzero_ps(); R];
                    for ib in 0..8 {
                        let wg = $decode(t, gb, ib);
                        let wu = $decode(t, ub, ib);
                        let chunk = 8 * b + ib;
                        for r in 0..R {
                            let q = _mm256_loadu_si256(rows[r].q.add(32 * chunk).cast());
                            let da = _mm256_set1_ps(
                                rows[r].ds.add(8 * chunk).cast::<f32>().read_unaligned(),
                            );
                            blk_g[r] = dot32(wg, q, da, blk_g[r]);
                            blk_u[r] = dot32(wu, q, da, blk_u[r]);
                        }
                    }
                    let dg = _mm256_set1_ps(iq2_block_scale(gb));
                    let du = _mm256_set1_ps(iq2_block_scale(ub));
                    for r in 0..R {
                        acc_g[r] = _mm256_fmadd_ps(blk_g[r], dg, acc_g[r]);
                        acc_u[r] = _mm256_fmadd_ps(blk_u[r], du, acc_u[r]);
                    }
                }
            }
            for r in 0..R {
                out[r] = [hsum(acc_g[r]), hsum(acc_u[r])];
            }
        }
    };
}

iq2_gate_up_row!(gate_up_row_iq2_s, 82, decode_iq2_s);
iq2_gate_up_row!(gate_up_row_iq2_s_loads, 82, decode_iq2_s_loads);
iq2_gate_up_row!(gate_up_row_iq2_xxs, 66, decode_iq2_xxs);
iq2_gate_up_row!(gate_up_row_iq2_xxs_loads, 66, decode_iq2_xxs_loads);

/// IQ1_M, values `32 ib ..` of the block at `blk`
/// (`{ u8 index[32]; u8 high[16]; u16 scales[4] }`): the grid values (-1, 0,
/// 1 as bytes), the scales `8 (2 s + 1)` of the two halves, and the same
/// scales with the delta's sign in lanes 0, 4, 8 and 12, zero elsewhere.
///
/// # Safety
///
/// `blk` must address a whole block (56 bytes) and `ib` be below 8.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_iq1_m(t: &Tables, blk: *const u8, ib: usize) -> Decoded {
    // SAFETY: the reads are inside the block; an 11-bit index is inside the
    // grid of 2048 entries, 6 scale bits and 4 sign bits inside their tables.
    unsafe {
        let low =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(blk.add(4 * ib).cast::<i32>().read_unaligned()));
        let qh = u32::from(blk.add(32 + 2 * ib).cast::<u16>().read_unaligned());
        // Three high bits per group, at bits 0, 4, 8 and 12.
        let high = _mm_and_si128(
            _mm_srlv_epi32(_mm_set1_epi32(qh as i32), _mm_setr_epi32(0, 4, 8, 12)),
            _mm_set1_epi32(7),
        );
        let index = _mm_or_si128(low, _mm_slli_epi32::<8>(high));
        let grid = _mm256_i32gather_epi64::<8>(t.iq1_grid.as_ptr().cast(), index);
        let word = u32::from(blk.add(48 + 2 * (ib / 2)).cast::<u16>().read_unaligned());
        let scale_bits = (word >> (6 * (ib % 2))) & 63;
        let scales = _mm256_load_si256(t.iq1_scales.as_ptr().add(scale_bits as usize).cast());
        // The delta's sign of each group: bits 3, 7, 11 and 15.
        let delta_bits = (qh >> 3 & 1) | (qh >> 6 & 2) | (qh >> 9 & 4) | (qh >> 12 & 8);
        let deltas = _mm256_load_si256(t.iq1_deltas.as_ptr().add(delta_bits as usize).cast());
        // 8 (2 s + 1) / 8 with the sign: the delta is an eighth.
        let delta_scales = _mm256_sign_epi16(_mm256_srai_epi16::<3>(scales), deltas);
        (grid, scales, delta_scales)
    }
}

/// Defines the gate and up product of one hidden row for IQ1_M. A weight is
/// `scale * (g + delta / 8)` with `g` in -1, 0, 1, so a group of 8 gives
/// `scale * (8 sum(g q) + delta sign * sum(q))` eighths: the second term
/// comes from the sums of the codes that the caller prepared.
macro_rules! iq1_gate_up_row {
    ($name:ident, $decode:ident) => {
        /// # Safety
        ///
        /// `gate` and `up` must address `blocks` whole blocks, and every row
        /// `256 * blocks` codes, `8 * blocks` scale pairs and `8 * blocks` group sums.
        #[target_feature(enable = "avx2,fma,f16c,bmi2")]
        unsafe fn $name<const R: usize>(
            t: &Tables,
            gate: *const u8,
            up: *const u8,
            blocks: usize,
            rows: &[RowIn; R],
            out: &mut [[f32; 2]; R],
        ) {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            fn dot(w: Decoded, q: __m256i, sum8: __m256i, da: __m256, acc: __m256) -> __m256 {
                let pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(w.0), _mm256_sign_epi8(q, w.0));
                let sums =
                    _mm256_add_epi32(_mm256_madd_epi16(pairs, w.1), _mm256_madd_epi16(sum8, w.2));
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums), da, acc)
            }
            let mut acc_g = [_mm256_setzero_ps(); R];
            let mut acc_u = [_mm256_setzero_ps(); R];
            for b in 0..blocks {
                // SAFETY: as in the IQ2 kernel, with the group sums of the chunk.
                unsafe {
                    let (gb, ub) = (gate.add(56 * b), up.add(56 * b));
                    let mut blk_g = [_mm256_setzero_ps(); R];
                    let mut blk_u = [_mm256_setzero_ps(); R];
                    for ib in 0..8 {
                        let wg = $decode(t, gb, ib);
                        let wu = $decode(t, ub, ib);
                        let chunk = 8 * b + ib;
                        for r in 0..R {
                            let q = _mm256_loadu_si256(rows[r].q.add(32 * chunk).cast());
                            let sum8 = _mm256_load_si256(rows[r].sum8.add(chunk).cast());
                            let da = _mm256_set1_ps(
                                rows[r].ds.add(8 * chunk).cast::<f32>().read_unaligned(),
                            );
                            blk_g[r] = dot(wg, q, sum8, da, blk_g[r]);
                            blk_u[r] = dot(wu, q, sum8, da, blk_u[r]);
                        }
                    }
                    let dg = _mm256_set1_ps(iq1_block_scale(gb));
                    let du = _mm256_set1_ps(iq1_block_scale(ub));
                    for r in 0..R {
                        acc_g[r] = _mm256_fmadd_ps(blk_g[r], dg, acc_g[r]);
                        acc_u[r] = _mm256_fmadd_ps(blk_u[r], du, acc_u[r]);
                    }
                }
            }
            for r in 0..R {
                out[r] = [hsum(acc_g[r]), hsum(acc_u[r])];
            }
        }
    };
}

iq1_gate_up_row!(gate_up_row_iq1_m, decode_iq1_m);
iq1_gate_up_row!(gate_up_row_iq1_m_loads, decode_iq1_m_loads);

/// The sums of every 8 activation codes of a row in the layout the IQ1_M
/// kernel multiplies: per 32 codes one vector with the four sums in lanes 0,
/// 4, 8 and 12.
pub(super) fn sum_groups(q: &[i8], out: &mut [Lanes16]) {
    for (chunk, lanes) in q.as_chunks::<32>().0.iter().zip(out) {
        lanes.0 = [0; 16];
        for (group, codes) in chunk.as_chunks::<8>().0.iter().enumerate() {
            lanes.0[4 * group] = codes.iter().map(|c| i16::from(*c)).sum();
        }
    }
}

/// Whether an expert's matrix can be stored in `ty`: the 28 types a model file
/// holds weights in, which are the ones the device kernels compute.
fn holds_weights(ty: GgmlType) -> bool {
    ares_cuda::ExpertShape::types().any(|t| t == ty)
}

/// The type of gate and up: one of the three with a kernel written for it, or
/// any other weight type, decoded row by row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GateKind {
    Iq2S,
    Iq2Xxs,
    Iq1M,
    Other(GgmlType),
}

impl GateKind {
    /// The kind of gate and up matrices of `ty`; `None` for a type that holds
    /// no weights.
    pub(super) fn of(ty: GgmlType) -> Option<Self> {
        match ty {
            GgmlType::IQ2_S => Some(Self::Iq2S),
            GgmlType::IQ2_XXS => Some(Self::Iq2Xxs),
            GgmlType::IQ1_M => Some(Self::Iq1M),
            ty => holds_weights(ty).then_some(Self::Other(ty)),
        }
    }

    /// The class of the kind among the three a thread measures itself in
    /// (`ThreadSpeed`). The decoded path has none of its own and takes the
    /// first: what a team compares is how its threads differ, which is their
    /// cores and not the type.
    pub(super) fn index(self) -> usize {
        match self {
            Self::Iq2S | Self::Other(_) => 0,
            Self::Iq2Xxs => 1,
            Self::Iq1M => 2,
        }
    }

    fn ty(self) -> GgmlType {
        match self {
            Self::Iq2S => GgmlType::IQ2_S,
            Self::Iq2Xxs => GgmlType::IQ2_XXS,
            Self::Iq1M => GgmlType::IQ1_M,
            Self::Other(ty) => ty,
        }
    }

    /// Bytes of a block of the type.
    pub(super) fn block_bytes(self) -> usize {
        self.ty().block_bytes() as usize
    }
}

/// The type of down: Q2_0, which has a kernel written for it, or any other
/// weight type, decoded row by row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DownKind {
    Q2_0,
    Other(GgmlType),
}

impl DownKind {
    /// The kind of a down matrix of `ty`; `None` for a type that holds no
    /// weights.
    pub(super) fn of(ty: GgmlType) -> Option<Self> {
        match ty {
            GgmlType::Q2_0 => Some(Self::Q2_0),
            ty => holds_weights(ty).then_some(Self::Other(ty)),
        }
    }
}

/// `sum(a[i] * b[i])` over `n` values in `f32`, eight lanes of four sums.
///
/// # Safety
///
/// `a` and `b` must address `n` values each.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (k, acc) in acc.iter_mut().enumerate() {
            // SAFETY: values `i + 8 k ..` of both are below `n`.
            *acc = unsafe {
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.add(i + 8 * k)),
                    _mm256_loadu_ps(b.add(i + 8 * k)),
                    *acc,
                )
            };
        }
        i += 32;
    }
    let mut sum = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
    while i < n {
        // SAFETY: `i` is below `n`.
        sum = unsafe { (*a.add(i)).mul_add(*b.add(i), sum) };
        i += 1;
    }
    sum
}

/// Row `row` of a matrix of `ty` with rows of `out.len()` values, decoded.
///
/// # Safety
///
/// `matrix` must address at least `row + 1` rows of whole blocks of `ty`.
unsafe fn decode_row(ty: GgmlType, matrix: *const u8, row: usize, out: &mut [f32]) {
    let bytes = out.len() / ty.block_values() as usize * ty.block_bytes() as usize;
    // SAFETY: the row's bytes are inside the matrix the caller vouches for.
    let stored = unsafe { std::slice::from_raw_parts(matrix.add(row * bytes), bytes) };
    // The job was made for a weight type and rows of whole blocks, which is
    // all the decoder can refuse.
    ares_quant::dequantize(ty, stored, out).expect("a row of whole blocks of a weight type");
}

/// [`gate_up_tile`] for a type without a kernel of its own: the rows of gate
/// and up decoded, times the rounded activations.
///
/// # Safety
///
/// As [`gate_up_tile`].
#[target_feature(enable = "avx2,fma")]
unsafe fn gate_up_tile_decoded(
    ty: GgmlType,
    gate: *const u8,
    up: *const u8,
    n_in: usize,
    tile: usize,
    rows: &[RowIn],
    hidden: &[Hidden],
) {
    // The activations the device kernels multiply: code times the scale of its 32.
    let mut x = vec![0.0f32; rows.len() * n_in];
    for (row, x) in rows.iter().zip(x.chunks_exact_mut(n_in)) {
        for (i, x) in x.iter_mut().enumerate() {
            // SAFETY: code `i` and the scale pair of its group are the row's
            // (the caller's guarantee).
            *x = unsafe {
                f32::from(*row.q.add(i)) * row.ds.add(8 * (i / 32)).cast::<f32>().read_unaligned()
            };
        }
    }
    let mut weights = vec![0.0f32; 2 * n_in];
    let (wg, wu) = weights.split_at_mut(n_in);
    let mut h = [[0.0f32; 32]; 8];
    for (v, j) in (32 * tile..32 * tile + 32).enumerate() {
        // SAFETY: row j of both matrices is inside them; the vectors are
        // `n_in` values long.
        unsafe {
            decode_row(ty, gate, j, wg);
            decode_row(ty, up, j, wu);
            for (r, x) in x.chunks_exact(n_in).enumerate() {
                let g = dot_f32(wg.as_ptr(), x.as_ptr(), n_in);
                let u = dot_f32(wu.as_ptr(), x.as_ptr(), n_in);
                h[r][v] = silu(g) * u;
            }
        }
    }
    for (values, hidden) in h.iter().zip(hidden) {
        // SAFETY: the caller's guarantee for `hidden`.
        unsafe { store_hidden(values, tile, hidden) };
    }
}

/// The hidden values of a pair in the order of the matrix, as the down
/// kernels of the device multiply them: code times the scale of its 32.
///
/// # Safety
///
/// `hidden` must be complete for `out.len()` values ([`store_hidden`]).
unsafe fn hidden_values(hidden: &Hidden, out: &mut [f32]) {
    for (n, out) in out.iter_mut().enumerate() {
        let (pair, rest) = (n / 128, n % 128);
        let (half, i, k) = (rest / 64, rest % 64 / 4, rest % 4);
        // SAFETY: the place `store_hidden` put value `n` and its scale.
        *out = unsafe {
            f32::from(*hidden.q.add(128 * pair + 32 * k + 16 * half + i))
                * *hidden.scale.add(n / 16)
        };
    }
}

/// [`down_rows`] for a type without a kernel of its own: each row of down
/// decoded once for the pairs of its expert.
///
/// # Safety
///
/// As [`down_rows`], with matrices of `ty`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn down_rows_decoded(
    ty: GgmlType,
    experts: &[DownExpert<'_>],
    n_ff: usize,
    n_out: usize,
    n_rows: usize,
    first: usize,
    count: usize,
    out: *mut f32,
) {
    let pairs: usize = experts.iter().map(|e| e.hidden.len()).sum();
    let mut h = vec![0.0f32; pairs * n_ff];
    for (hidden, h) in experts.iter().flat_map(|e| e.hidden).zip(h.chunks_exact_mut(n_ff)) {
        // SAFETY: the hiddens are complete for `n_ff` values.
        unsafe { hidden_values(hidden, h) };
    }
    let mut w = vec![0.0f32; n_ff];
    for j in first..first + count {
        let mut total = [0.0f32; 8];
        debug_assert!(n_rows <= total.len());
        let mut pair = 0;
        for expert in experts {
            // SAFETY: row j of the expert's matrix is inside it.
            unsafe { decode_row(ty, expert.down, j, &mut w) };
            for (b, weight) in expert.row.iter().zip(expert.weight) {
                // SAFETY: both vectors are `n_ff` values long.
                let dot = unsafe { dot_f32(w.as_ptr(), h[pair * n_ff..].as_ptr(), n_ff) };
                total[usize::from(*b)] = weight.mul_add(dot, total[usize::from(*b)]);
                pair += 1;
            }
        }
        for (b, total) in total.iter().enumerate().take(n_rows) {
            // SAFETY: value j of row b is inside `out`, and this range of j
            // is this call's alone.
            unsafe { *out.add(b * n_out + j) = *total };
        }
    }
}

/// `x / (1 + e^-x)`, as the device computes it.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Hidden values `32 tile ..` of one expert for the rows that use it:
/// `silu(gate x) * (up x)`, quantized per row to int8 with one scale and
/// stored for the down product ([`store_hidden`]).
///
/// # Safety
///
/// The CPU must have AVX2, FMA, F16C and BMI2. `gate` and `up` must address
/// matrices of `n_ff` rows of `n_in` values of `kind` with `32 tile + 32 <=
/// n_ff`; `rows` (1 to 8) and `hidden` are as long as each other, each row as
/// [`RowIn`] says for `n_in` values, each hidden as [`Hidden`] says for
/// `n_ff` values and written by nobody else in this tile's range.
// `v` is a column of every row's buffer, not an index into one slice.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma,f16c,bmi2")]
pub(super) unsafe fn gate_up_tile(
    t: &Tables,
    kind: GateKind,
    gather: bool,
    gate: *const u8,
    up: *const u8,
    n_in: usize,
    tile: usize,
    rows: &[RowIn],
    hidden: &[Hidden],
) {
    if let GateKind::Other(ty) = kind {
        // SAFETY: the caller's guarantees are the function's.
        return unsafe { gate_up_tile_decoded(ty, gate, up, n_in, tile, rows, hidden) };
    }
    let blocks = n_in / 256;
    let row_bytes = blocks * kind.block_bytes();
    let mut h = [[0.0f32; 32]; 8];
    for (v, j) in (32 * tile..32 * tile + 32).enumerate() {
        // SAFETY: row j of both matrices is inside them (the caller's
        // guarantee), as are the rows' activations.
        let (g, u) = unsafe { (gate.add(j * row_bytes), up.add(j * row_bytes)) };
        let mut at = 0;
        while at < rows.len() {
            let n = (rows.len() - at).min(4);
            // Runs the kernel of `kind` for `N` rows starting at `at`.
            macro_rules! batch {
                ($n:literal) => {{
                    let mut batch = [rows[at]; $n];
                    batch.copy_from_slice(&rows[at..at + $n]);
                    let mut out = [[0.0f32; 2]; $n];
                    // SAFETY: the caller's guarantees are the kernels'.
                    unsafe {
                        match (kind, gather) {
                            (GateKind::Iq2S, true) => {
                                gate_up_row_iq2_s::<$n>(t, g, u, blocks, &batch, &mut out)
                            }
                            (GateKind::Iq2S, false) => {
                                gate_up_row_iq2_s_loads::<$n>(t, g, u, blocks, &batch, &mut out)
                            }
                            (GateKind::Iq2Xxs, true) => {
                                gate_up_row_iq2_xxs::<$n>(t, g, u, blocks, &batch, &mut out)
                            }
                            (GateKind::Iq2Xxs, false) => {
                                gate_up_row_iq2_xxs_loads::<$n>(t, g, u, blocks, &batch, &mut out)
                            }
                            (GateKind::Iq1M, true) => {
                                gate_up_row_iq1_m::<$n>(t, g, u, blocks, &batch, &mut out)
                            }
                            (GateKind::Iq1M, false) => {
                                gate_up_row_iq1_m_loads::<$n>(t, g, u, blocks, &batch, &mut out)
                            }
                            // Taken above.
                            (GateKind::Other(_), _) => {}
                        }
                    }
                    for (r, [gate, up]) in out.into_iter().enumerate() {
                        h[at + r][v] = silu(gate) * up;
                    }
                }};
            }
            match n {
                1 => batch!(1),
                2 => batch!(2),
                3 => batch!(3),
                _ => batch!(4),
            }
            at += n;
        }
    }
    for (values, hidden) in h.iter().zip(hidden) {
        // SAFETY: the caller's guarantee for `hidden`.
        unsafe { store_hidden(values, tile, hidden) };
    }
}

/// Quantizes 32 hidden values as `ares_expert_gate_up` does (int8 codes
/// `round(v / d)` with `d = max |v| / 127`) and stores them where the down
/// kernel reads them.
///
/// # Safety
///
/// `hidden` must address the arrays of a pair with at least `32 tile + 32`
/// values, and nobody else may write this tile's part of them.
pub(super) unsafe fn store_hidden(values: &[f32; 32], tile: usize, hidden: &Hidden) {
    let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let mut codes = [0i8; 32];
    if amax > 0.0 {
        for (code, v) in codes.iter_mut().zip(values) {
            *code = (v / d).round() as i8;
        }
    }
    let first = 32 * tile;
    let (pair, half) = (first / 128, first % 128 / 64);
    // The tile is values `first % 64 ..` of its block of 64: `i` runs over
    // 8 of the block's 16 bytes.
    let i0 = first % 64 / 4;
    // SAFETY: the plane-order positions of this tile's 32 values, and its
    // two entries of the sums and scales, are inside the pair's arrays.
    unsafe {
        for (n, code) in codes.iter().enumerate() {
            let (i, k) = (i0 + n / 4, n % 4);
            *hidden.q.add(128 * pair + 32 * k + 16 * half + i) = *code;
        }
        for piece in 0..2 {
            let sum: i32 = codes[16 * piece..16 * piece + 16].iter().map(|c| i32::from(*c)).sum();
            *hidden.sum.add(first / 16 + piece) = sum;
            *hidden.scale.add(first / 16 + piece) = d;
        }
    }
}

/// Defines the down product of one output row for `R` (expert, row) pairs
/// that share the expert: `out[r] = down_row . hidden_r`. `$dot` multiplies
/// the four planes of two blocks with a pair's codes and returns the eight
/// sums of 16 values each as `i32`.
macro_rules! q2_down_row {
    ($name:ident, $features:literal, $dot:ident) => {
        /// # Safety
        ///
        /// `row` must address `blocks` whole Q2_0 blocks (18 bytes each),
        /// and every hidden the arrays of `64 * blocks` values rounded up to
        /// 128, as [`store_hidden`] filled them.
        #[target_feature(enable = $features)]
        unsafe fn $name<const R: usize>(
            row: *const u8,
            blocks: usize,
            hidden: &[Hidden; R],
            out: &mut [f32; R],
        ) {
            let three = _mm256_set1_epi8(3);
            let mut acc = [_mm256_setzero_ps(); R];
            for p in 0..blocks.div_ceil(2) {
                // SAFETY: blocks 2 p and, if the row has it, 2 p + 1, and
                // pair p of the hidden arrays, are inside what the caller
                // vouches for.
                unsafe {
                    let b0 = row.add(36 * p);
                    let low = _mm_loadu_si128(b0.add(2).cast());
                    let d0 = u32::from(b0.cast::<u16>().read_unaligned());
                    // An odd count of blocks: the last pair's second half
                    // is nothing times nothing.
                    let (high, d1) = if 2 * p + 1 < blocks {
                        (
                            _mm_loadu_si128(b0.add(20).cast()),
                            u32::from(b0.add(18).cast::<u16>().read_unaligned()),
                        )
                    } else {
                        (_mm_setzero_si128(), 0)
                    };
                    let codes = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high);
                    let planes = [
                        _mm256_and_si256(codes, three),
                        _mm256_and_si256(_mm256_srli_epi16::<2>(codes), three),
                        _mm256_and_si256(_mm256_srli_epi16::<4>(codes), three),
                        _mm256_and_si256(_mm256_srli_epi16::<6>(codes), three),
                    ];
                    // The scales of the two blocks, each over its four sums.
                    let d = _mm_cvtph_ps(_mm_cvtsi32_si128((d0 | d1 << 16) as i32));
                    let d = _mm256_permutevar8x32_ps(
                        _mm256_castps128_ps256(d),
                        _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
                    );
                    for r in 0..R {
                        let h = &hidden[r];
                        let sums = $dot(&planes, h.q.add(128 * p).cast_const());
                        let sums =
                            _mm256_sub_epi32(sums, _mm256_loadu_si256(h.sum.add(8 * p).cast()));
                        let scaled = _mm256_mul_ps(
                            _mm256_cvtepi32_ps(sums),
                            _mm256_loadu_ps(h.scale.add(8 * p)),
                        );
                        acc[r] = _mm256_fmadd_ps(scaled, d, acc[r]);
                    }
                }
            }
            for r in 0..R {
                out[r] = hsum(acc[r]);
            }
        }
    };
}

/// The planes times the codes with AVX2: sums of pairs as `i16` (at most
/// `2 * 3 * 127` each, four of them added), then of four values.
///
/// # Safety
///
/// `q` must address 128 codes.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn planes_dot_avx2(planes: &[__m256i; 4], q: *const i8) -> __m256i {
    // SAFETY: four loads of 32 codes inside the 128.
    unsafe {
        let s01 = _mm256_add_epi16(
            _mm256_maddubs_epi16(planes[0], _mm256_loadu_si256(q.cast())),
            _mm256_maddubs_epi16(planes[1], _mm256_loadu_si256(q.add(32).cast())),
        );
        let s23 = _mm256_add_epi16(
            _mm256_maddubs_epi16(planes[2], _mm256_loadu_si256(q.add(64).cast())),
            _mm256_maddubs_epi16(planes[3], _mm256_loadu_si256(q.add(96).cast())),
        );
        _mm256_madd_epi16(_mm256_add_epi16(s01, s23), _mm256_set1_epi16(1))
    }
}

/// The same with AVX-VNNI: each plane's products are summed in fours
/// straight into `i32`.
///
/// # Safety
///
/// `q` must address 128 codes.
#[inline]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn planes_dot_vnni(planes: &[__m256i; 4], q: *const i8) -> __m256i {
    // SAFETY: four loads of 32 codes inside the 128.
    unsafe {
        let s = _mm256_dpbusd_avx_epi32(
            _mm256_setzero_si256(),
            planes[0],
            _mm256_loadu_si256(q.cast()),
        );
        let s = _mm256_dpbusd_avx_epi32(s, planes[1], _mm256_loadu_si256(q.add(32).cast()));
        let s = _mm256_dpbusd_avx_epi32(s, planes[2], _mm256_loadu_si256(q.add(64).cast()));
        _mm256_dpbusd_avx_epi32(s, planes[3], _mm256_loadu_si256(q.add(96).cast()))
    }
}

q2_down_row!(down_row_q2_0_avx2, "avx2,fma,f16c", planes_dot_avx2);
q2_down_row!(down_row_q2_0_vnni, "avx2,fma,f16c,avxvnni", planes_dot_vnni);

/// One expert of a down task: its matrix and the pairs of the rows that use
/// it, with each row's index in the step and its weight for the expert.
#[derive(Clone, Copy, Debug)]
pub(super) struct DownExpert<'a> {
    pub down: *const u8,
    pub hidden: &'a [Hidden],
    pub row: &'a [u8],
    pub weight: &'a [f32],
}

/// Output values `first .. first + count` of every row of the step: the sum
/// over `experts`, in their order, of `weight * (down_row . hidden)`, written
/// to `out[row * n_out + j]`.
///
/// # Safety
///
/// The CPU must have AVX2, FMA and F16C, and AVX-VNNI if `vnni`. Every
/// `down` must address `n_out` rows of `n_ff` values of `kind`'s type with
/// `first + count <= n_out`; an expert's `hidden`, `row` and `weight` are as long as
/// each other (1 to 8), the hiddens complete for `n_ff` values; `out` must
/// address `n_rows * n_out` values of which nobody else writes this range,
/// and every `row` be below `n_rows`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn down_rows(
    vnni: bool,
    kind: DownKind,
    experts: &[DownExpert<'_>],
    n_ff: usize,
    n_out: usize,
    n_rows: usize,
    first: usize,
    count: usize,
    out: *mut f32,
) {
    if let DownKind::Other(ty) = kind {
        // SAFETY: the caller's guarantees are the function's.
        return unsafe { down_rows_decoded(ty, experts, n_ff, n_out, n_rows, first, count, out) };
    }
    let blocks = n_ff / 64;
    let row_bytes = blocks * 18;
    for j in first..first + count {
        let mut total = [0.0f32; 8];
        debug_assert!(n_rows <= total.len());
        for expert in experts {
            // SAFETY: row j of the expert's matrix is inside it.
            let row = unsafe { expert.down.add(j * row_bytes) };
            let mut at = 0;
            while at < expert.hidden.len() {
                let n = (expert.hidden.len() - at).min(4);
                // Runs the kernel for `N` pairs starting at `at`.
                macro_rules! batch {
                    ($n:literal) => {{
                        let mut batch = [expert.hidden[at]; $n];
                        batch.copy_from_slice(&expert.hidden[at..at + $n]);
                        let mut dots = [0.0f32; $n];
                        // SAFETY: the caller's guarantees are the kernels';
                        // the VNNI form runs only where the caller says the
                        // CPU has it.
                        unsafe {
                            if vnni {
                                down_row_q2_0_vnni::<$n>(row, blocks, &batch, &mut dots);
                            } else {
                                down_row_q2_0_avx2::<$n>(row, blocks, &batch, &mut dots);
                            }
                        }
                        for (r, dot) in dots.into_iter().enumerate() {
                            let b = usize::from(expert.row[at + r]);
                            total[b] = expert.weight[at + r].mul_add(dot, total[b]);
                        }
                    }};
                }
                match n {
                    1 => batch!(1),
                    2 => batch!(2),
                    3 => batch!(3),
                    _ => batch!(4),
                }
                at += n;
            }
        }
        for (b, total) in total.iter().enumerate().take(n_rows) {
            // SAFETY: value j of row b is inside `out`, and this range of j
            // is this call's alone.
            unsafe { *out.add(b * n_out + j) = *total };
        }
    }
}

#[cfg(test)]
mod tests {
    //! Every weight type through a job on the CPU, against exact arithmetic on the
    //! decoded weights: the decoded path for gate and up and for down, next to the
    //! written kernels (whose own tests are in the parent module).

    use ares_core::GgmlType;
    use ares_cuda::ExpertShape;

    use super::super::{CpuExpert, Features, LayerJob, MAX_ROWS, quantize_activations};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// The byte offsets of the f16 scales of a block of `ty`.
    fn scale_fields(ty: GgmlType) -> &'static [usize] {
        use GgmlType::*;
        match ty {
            Q4_K | Q5_K | Q4_1 | Q5_1 => &[0, 2],
            Q2_K => &[80, 82],
            Q6_K => &[208],
            Q3_K => &[108],
            TQ1_0 => &[52],
            TQ2_0 => &[64],
            IQ1_M | MXFP4 | NVFP4 | F32 | F16 | BF16 => &[],
            _ => &[0],
        }
    }

    /// Random blocks of `ty` for `values` values, with finite scales of moderate size (any
    /// other bytes are a block in every weight type); random values for a float type.
    fn random_matrix(rng: &mut Rng, ty: GgmlType, values: usize) -> Vec<u8> {
        let block = ty.block_bytes() as usize;
        let mut bytes = vec![0u8; values / ty.block_values() as usize * block];
        for b in &mut bytes {
            *b = rng.next() as u8;
        }
        for blk in bytes.chunks_exact_mut(block) {
            // An f16 between 1/1024 and 1/128.
            let d = (0x1400u16 + (rng.next() % 0xC00) as u16).to_le_bytes();
            for &at in scale_fields(ty) {
                blk[at..at + 2].copy_from_slice(&d);
            }
            let small = (rng.unit() - 0.5) * 0.25;
            match ty {
                GgmlType::IQ1_M => {
                    blk[49] = blk[49] & 0x0f | (d[0] & 0x0f) << 4;
                    blk[51] = blk[51] & 0x0f | d[0] & 0xf0;
                    blk[53] = blk[53] & 0x0f | (d[1] & 0x0f) << 4;
                    blk[55] = blk[55] & 0x0f | d[1] & 0xf0;
                }
                // 2^-9 to 2^-6, and the same range in four exponent bits.
                GgmlType::MXFP4 => blk[0] = 119 + (rng.next() % 4) as u8,
                GgmlType::NVFP4 => {
                    for byte in blk.iter_mut().take(4) {
                        *byte = (((1 + rng.next() % 3) << 3) | (rng.next() % 8)) as u8;
                    }
                }
                GgmlType::F32 => blk.copy_from_slice(&small.to_le_bytes()),
                GgmlType::F16 => blk.copy_from_slice(&ares_quant::f32_to_f16(small).to_le_bytes()),
                GgmlType::BF16 => {
                    blk.copy_from_slice(&ares_quant::f32_to_bf16(small).to_le_bytes());
                }
                _ => {}
            }
        }
        bytes
    }

    fn decoded(ty: GgmlType, bytes: &[u8]) -> Vec<f64> {
        let mut out =
            vec![0.0f32; bytes.len() / ty.block_bytes() as usize * ty.block_values() as usize];
        ares_quant::dequantize(ty, bytes, &mut out).expect("whole blocks");
        out.iter().map(|v| f64::from(*v)).collect()
    }

    /// The expert on the rounded activations `x`, with the hidden values rounded as the
    /// kernels round them.
    fn reference(gate: &[f64], up: &[f64], down: &[f64], x: &[f64], n_ff: usize) -> Vec<f64> {
        let n_in = x.len();
        let dot = |w: &[f64], x: &[f64]| w.iter().zip(x).map(|(w, x)| w * x).sum::<f64>();
        let mut h: Vec<f64> = (0..n_ff)
            .map(|j| {
                let g = dot(&gate[j * n_in..(j + 1) * n_in], x) as f32;
                let u = dot(&up[j * n_in..(j + 1) * n_in], x) as f32;
                f64::from(g / (1.0 + (-g).exp()) * u)
            })
            .collect();
        for chunk in h.as_chunks_mut::<32>().0.iter_mut() {
            let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs() as f32));
            let d = amax / 127.0;
            for v in chunk {
                *v = if amax > 0.0 { f64::from((*v as f32 / d).round() * d) } else { 0.0 };
            }
        }
        (0..n_in).map(|i| dot(&down[i * n_ff..(i + 1) * n_ff], &h)).collect()
    }

    /// A layer of three experts of the two types and `n_rows` rows: the job's sums against
    /// the reference, as a relative rms.
    fn check(gate_ty: GgmlType, down_ty: GgmlType, n_rows: usize) -> f64 {
        let (n_in, n_ff) = (256, down_ty.block_values().max(64) as usize);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (gate_ty as u64) << 8 ^ down_ty as u64);
        let shape = ExpertShape {
            n_embd: n_in as u32,
            n_ff: n_ff as u32,
            n_experts: 16,
            top_k: 2,
            gate_ty,
            down_ty,
        };
        let matrices: Vec<[Vec<u8>; 3]> = (0..3)
            .map(|_| {
                [
                    random_matrix(&mut rng, gate_ty, n_ff * n_in),
                    random_matrix(&mut rng, gate_ty, n_ff * n_in),
                    random_matrix(&mut rng, down_ty, n_in * n_ff),
                ]
            })
            .collect();
        let x: Vec<f32> = (0..n_rows * n_in).map(|_| 4.0 * (rng.unit() - 0.5)).collect();
        let act = quantize_activations(&x, n_in);
        let ds_at = (n_rows * n_in).next_multiple_of(16);
        let rounded: Vec<f64> = (0..n_rows * n_in)
            .map(|i| {
                let at = ds_at + 8 * (i / 32);
                let d = f32::from_le_bytes([act[at], act[at + 1], act[at + 2], act[at + 3]]);
                f64::from(act[i] as i8) * f64::from(d)
            })
            .collect();
        let mut experts = Vec::new();
        for (e, m) in matrices.iter().enumerate() {
            let mut rows = 0u32;
            let mut weight = [0.0f32; MAX_ROWS];
            for (b, weight) in weight.iter_mut().enumerate().take(n_rows) {
                if !(b + e).is_multiple_of(3) || n_rows == 1 {
                    rows |= 1 << b;
                    *weight = 0.2 + rng.unit();
                }
            }
            experts.push(CpuExpert {
                gate: m[0].as_ptr(),
                up: m[1].as_ptr(),
                down: m[2].as_ptr(),
                rows,
                weight,
            });
        }
        let features = Features::detect().expect("x86-64-v3");
        // SAFETY: the matrices are whole and outlive the job.
        let job =
            unsafe { LayerJob::new(features, &shape, n_rows, &act, &experts) }.expect("a job");
        job.work();
        let sums = job.sums().expect("the job has run");
        assert!(sums.iter().all(|v| v.is_finite()), "{gate_ty} and {down_ty}: not a number");

        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for b in 0..n_rows {
            let mut want = vec![0.0f64; n_in];
            for (expert, m) in experts.iter().zip(&matrices) {
                if expert.rows >> b & 1 == 0 {
                    continue;
                }
                let part = reference(
                    &decoded(gate_ty, &m[0]),
                    &decoded(gate_ty, &m[1]),
                    &decoded(down_ty, &m[2]),
                    &rounded[b * n_in..(b + 1) * n_in],
                    n_ff,
                );
                let w = f64::from(expert.weight[b]);
                want.iter_mut().zip(&part).for_each(|(s, p)| *s += w * p);
            }
            for (got, want) in sums[b * n_in..(b + 1) * n_in].iter().zip(&want) {
                err += (f64::from(*got) - want).powi(2);
                norm += want * want;
            }
        }
        (err / norm).sqrt()
    }

    #[test]
    fn experts_of_every_weight_type_agree_with_the_reference() {
        let types: Vec<GgmlType> = ExpertShape::types().collect();
        assert_eq!(types.len(), 28);
        let mut pairs: Vec<(GgmlType, GgmlType)> = Vec::new();
        for ty in &types {
            pairs.push((*ty, GgmlType::Q2_0));
            pairs.push((GgmlType::IQ2_S, *ty));
        }
        pairs.extend([(GgmlType::Q4_K, GgmlType::Q4_K), (GgmlType::BF16, GgmlType::F16)]);
        for (gate_ty, down_ty) in pairs {
            for n_rows in [1, 5, 8] {
                let error = check(gate_ty, down_ty, n_rows);
                // Against the same rounding of activations and hidden values only the
                // order of the float additions differs (and, rarely, a hidden code that sat
                // on a rounding boundary): the bound of the written kernels.
                assert!(
                    error.is_finite() && error < 2e-3,
                    "{gate_ty} and {down_ty}, {n_rows} rows: {error:e} against the same rounding"
                );
            }
        }
    }

    #[test]
    fn types_that_hold_no_weights_are_refused() {
        for ty in [GgmlType::I8, GgmlType::I32, GgmlType::F64, GgmlType::Q8_K] {
            assert!(super::GateKind::of(ty).is_none() && super::DownKind::of(ty).is_none());
        }
    }
}
