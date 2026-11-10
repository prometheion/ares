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

use std::arch::x86_64::*;

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

/// The type of gate and up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GateKind {
    Iq2S,
    Iq2Xxs,
    Iq1M,
}

impl GateKind {
    /// Bytes of a block of 256 values.
    pub(super) fn block_bytes(self) -> usize {
        match self {
            Self::Iq2S => 82,
            Self::Iq2Xxs => 66,
            Self::Iq1M => 56,
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
    let blocks = n_in / 256;
    let row_bytes = blocks * kind.block_bytes();
    let mut h = [[0.0f32; 32]; 8];
    for v in 0..32 {
        let j = 32 * tile + v;
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
/// `down` must address `n_out` rows of `n_ff` Q2_0 values with `first +
/// count <= n_out`; an expert's `hidden`, `row` and `weight` are as long as
/// each other (1 to 8), the hiddens complete for `n_ff` values; `out` must
/// address `n_rows * n_out` values of which nobody else writes this range,
/// and every `row` be below `n_rows`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn down_rows(
    vnni: bool,
    experts: &[DownExpert<'_>],
    n_ff: usize,
    n_out: usize,
    n_rows: usize,
    first: usize,
    count: usize,
    out: *mut f32,
) {
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
