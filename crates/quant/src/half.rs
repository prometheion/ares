//! Conversions between `f32` and the two 16-bit float formats GGUF stores.
//!
//! Both formats travel as raw `u16` bit patterns, because that is what a tensor
//! file holds and because Rust's own `f16` type is not stable. The functions
//! return exactly what ggml's `ggml_compute_fp16_to_fp32`,
//! `ggml_compute_fp32_to_fp16` and the bf16 pair return for every one of the
//! 2^32 inputs (the test suite checks this against fixtures made by ggml, and a
//! one-off exhaustive run was compared with ggml's C code). Where IEEE leaves a
//! choice, ggml's choice is the one made here: NaN payloads of an `f32` are
//! dropped when converting to f16, and a NaN of either format becomes a quiet
//! NaN when widened.

/// 2^-24, the value of the smallest f16 subnormal (bit pattern `0x0001`).
const F16_SUBNORMAL_UNIT: f32 = f32::from_bits(0x3380_0000);

/// Widens an IEEE binary16 bit pattern to `f32`. Exact for every finite input.
///
/// A signalling NaN comes out quiet (the quiet bit is set), keeping the rest
/// of its payload, like the hardware conversion does and like ggml's does.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x03ff);
    let out = match exp {
        // Zero and subnormals are `man * 2^-24`; both factors are exact in f32.
        0 => (man as f32 * F16_SUBNORMAL_UNIT).to_bits(),
        0x1f if man == 0 => 0x7f80_0000,
        0x1f => 0x7fc0_0000 | (man << 13),
        // Rebias the exponent from 15 to 127.
        _ => ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(sign | out)
}

/// Narrows an `f32` to the nearest IEEE binary16, ties to even.
///
/// Values past 65520 (the midpoint between the largest finite f16, 65504, and
/// the next step) become infinity, values below half the smallest subnormal
/// become zero, and every NaN becomes `0x7e00` with the sign kept.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;

    if abs > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    // 65520.0: the first value that rounds to infinity.
    if abs >= 0x477f_f000 {
        return sign | 0x7c00;
    }
    if abs >= 0x3880_0000 {
        // Normal f16 (>= 2^-14). Rebias the exponent from 127 to 15, drop 13
        // mantissa bits and round to nearest even by adding half an ulp minus
        // one plus the lowest kept bit. A carry out of the mantissa bumps the
        // exponent, which is the correct result.
        let rebased = abs - 0x3800_0000;
        let rounded = rebased + 0x0fff + ((rebased >> 13) & 1);
        return sign | (rounded >> 13) as u16;
    }
    // Subnormal f16 (or zero): the result is the number of 2^-24 units, rounded
    // to nearest even. The f32 value is `m * 2^(exp - 150)` with the implicit
    // bit made explicit in `m`.
    let exp = abs >> 23;
    if exp < 102 {
        // Below 2^-25, which is less than half of the smallest subnormal.
        return sign;
    }
    let m = (abs & 0x007f_ffff) | 0x0080_0000;
    let shift = 126 - exp; // 14..=24
    let kept = m >> shift;
    let rest = m & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let up = rest > half || (rest == half && kept & 1 == 1);
    sign | (kept + u32::from(up)) as u16
}

/// Widens a bfloat16 bit pattern to `f32`. Exact: bf16 is the top half of an
/// `f32`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Narrows an `f32` to bfloat16, ties to even.
///
/// A NaN stays a NaN (the quiet bit is forced, because dropping the low payload
/// bits could otherwise produce an infinity). Infinity and values that round
/// past the largest finite bf16 become infinity.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) | 0x40) as u16;
    }
    // Add half an ulp minus one plus the lowest kept bit, then cut. `bits` is
    // at most 0xff80_0000 here, so the sum cannot wrap.
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact value of a finite f16, computed in f64 from its fields alone.
    fn f16_value(bits: u16) -> f64 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = i32::from((bits >> 10) & 0x1f);
        let man = f64::from(bits & 0x3ff);
        sign * if exp == 0 { man * 2f64.powi(-24) } else { (1024.0 + man) * 2f64.powi(exp - 25) }
    }

    #[test]
    fn f16_widening_is_exact_for_every_finite_pattern() {
        // Proves f16_to_f32 against an independent computation in f64, which
        // can represent every f16 and every f32 exactly.
        for bits in 0..=u16::MAX {
            if (bits >> 10) & 0x1f == 0x1f {
                continue;
            }
            let wide = f64::from(f16_to_f32(bits));
            assert_eq!(wide, f16_value(bits), "f16 {bits:#06x}");
            // The sign of zero must survive too.
            assert_eq!(wide.is_sign_negative(), bits & 0x8000 != 0, "f16 {bits:#06x}");
        }
    }

    #[test]
    fn f16_infinity_and_nan_widen_like_ggml() {
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        // Quiet bit forced, payload kept, sign kept.
        assert_eq!(f16_to_f32(0x7e00).to_bits(), 0x7fc0_0000);
        assert_eq!(f16_to_f32(0x7c01).to_bits(), 0x7fc0_2000);
        assert_eq!(f16_to_f32(0xfdff).to_bits(), 0xffc0_0000 | (0x1ff << 13));
    }

    #[test]
    fn f16_narrowing_inverts_widening_for_every_pattern() {
        // Every f16 is an f32, so narrowing must give the same pattern back.
        for bits in 0..=u16::MAX {
            let exp_all_ones = (bits >> 10) & 0x1f == 0x1f;
            let is_nan = exp_all_ones && bits & 0x3ff != 0;
            let back = f32_to_f16(f16_to_f32(bits));
            if is_nan {
                assert_eq!(back, (bits & 0x8000) | 0x7e00, "f16 NaN {bits:#06x}");
            } else {
                assert_eq!(back, bits, "f16 {bits:#06x}");
            }
        }
    }

    #[test]
    fn f16_narrowing_rounds_to_nearest_even_between_neighbours() {
        // For each pair of adjacent non-negative f16 values, the exact midpoint
        // is an f32 (it needs one more mantissa bit). It must round to the
        // neighbour with an even pattern, a hair below it to the lower one and
        // a hair above it to the upper one.
        for lo in 0..0x7bffu16 {
            let hi = lo + 1;
            let a = f16_value(lo);
            let b = f16_value(hi);
            let mid = ((a + b) / 2.0) as f32;
            assert_eq!(f64::from(mid), (a + b) / 2.0, "midpoint of {lo:#06x} is not an f32");
            let even = if lo % 2 == 0 { lo } else { hi };
            assert_eq!(f32_to_f16(mid), even, "midpoint above {lo:#06x}");
            assert_eq!(
                f32_to_f16(f32::from_bits(mid.to_bits() - 1)),
                lo,
                "below midpoint {lo:#06x}"
            );
            assert_eq!(
                f32_to_f16(f32::from_bits(mid.to_bits() + 1)),
                hi,
                "above midpoint {lo:#06x}"
            );
            // And the sign is carried through symmetrically.
            assert_eq!(f32_to_f16(-mid), even | 0x8000);
        }
    }

    #[test]
    fn f16_narrowing_handles_the_range_ends() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(f32::from_bits(0x477f_efff)), 0x7bff, "just below the tie");
        assert_eq!(f32_to_f16(65520.0), 0x7c00, "the tie goes to the even neighbour, infinity");
        assert_eq!(f32_to_f16(1e30), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        // 2^-24 is the smallest subnormal; half of it is a tie that goes to zero.
        assert_eq!(f32_to_f16(f32::from_bits(0x3380_0000)), 0x0001);
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0000)), 0x0000);
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0001)), 0x0001);
        assert_eq!(f32_to_f16(1e-30), 0x0000);
        assert_eq!(f32_to_f16(f32::MIN_POSITIVE), 0x0000);
        // Any NaN becomes the canonical quiet NaN with its sign.
        assert_eq!(f32_to_f16(f32::NAN), 0x7e00);
        assert_eq!(f32_to_f16(f32::from_bits(0xffff_ffff)), 0xfe00);
        assert_eq!(f32_to_f16(f32::from_bits(0x7f80_0001)), 0x7e00);
    }

    #[test]
    fn bf16_widening_is_the_top_half_of_an_f32() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
        assert_eq!(bf16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(bf16_to_f32(0x7f80), f32::INFINITY);
    }

    #[test]
    fn bf16_narrowing_inverts_widening_for_every_pattern() {
        for bits in 0..=u16::MAX {
            let is_nan = bits & 0x7fff > 0x7f80;
            let back = f32_to_bf16(bf16_to_f32(bits));
            // A NaN comes back with its quiet bit forced; everything else is identical.
            let expected = if is_nan { bits | 0x40 } else { bits };
            assert_eq!(back, expected, "bf16 {bits:#06x}");
        }
    }

    #[test]
    fn bf16_narrowing_rounds_to_nearest_even() {
        // 1 + 2^-8 is the midpoint between bf16 1.0 (0x3f80) and the next value
        // 0x3f81; it goes to the even neighbour 0x3f80. 1 + 3 * 2^-8 is the
        // midpoint between 0x3f81 and 0x3f82 and goes to the even 0x3f82.
        assert_eq!(f32_to_bf16(1.0 + f32::from_bits(0x3b80_0000)), 0x3f80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 * f32::from_bits(0x3b80_0000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        // The largest finite f32 rounds up to infinity.
        assert_eq!(f32_to_bf16(f32::MAX), 0x7f80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
        // A NaN whose payload sits in the low bits must not turn into infinity.
        assert_eq!(f32_to_bf16(f32::from_bits(0x7f80_0001)), 0x7fc0);
        assert_eq!(f32_to_bf16(f32::from_bits(0xff80_ffff)), 0xffc0);
    }
}
