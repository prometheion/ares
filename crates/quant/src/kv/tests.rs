//! Tests of the parts of the KV formats that are not reachable from outside:
//! the bit packing, the scale rule and the rotation constant. The behaviour of
//! the public functions is tested in `tests/kv.rs`.

use super::*;

/// The code range of a width, and one beyond it: `-2^(B-1)` is a valid B-bit
/// number that no encoder writes but a packer must still carry intact.
fn all_b_bit_values(bits: u32) -> std::ops::RangeInclusive<i32> {
    -(1 << (bits - 1))..=code_max(bits)
}

#[test]
fn the_rotation_constant_is_the_f32_nearest_to_one_over_root_32() {
    let exact = 1.0 / 32f64.sqrt();
    let error = |v: f32| (f64::from(v) - exact).abs();
    let bits = INV_SQRT_32.to_bits();
    assert_eq!(bits, 0x3e35_04f3);
    assert!(error(INV_SQRT_32) < error(f32::from_bits(bits + 1)));
    assert!(error(INV_SQRT_32) < error(f32::from_bits(bits - 1)));
}

#[test]
fn code_ranges_are_symmetric_and_use_the_whole_width_but_one_value() {
    let expected = [(8, 127), (7, 63), (6, 31), (5, 15), (4, 7)];
    for (bits, max) in expected {
        assert_eq!(code_max(bits), max, "q{bits}");
    }
}

#[test]
fn every_code_survives_packing_at_every_position() {
    // Proves pack and unpack are inverse for each width, value and position,
    // across the planes a code is split into and including the unused
    // -2^(B-1). The other 31 codes are a background that would show a bleed
    // into a neighbour: zeros, all ones, and alternating extremes.
    for bits in 4..=8u32 {
        let max = code_max(bits);
        let backgrounds: [Box<dyn Fn(usize) -> i32>; 3] = [
            Box::new(|_| 0),
            Box::new(|_| -1),
            Box::new(move |i| if i % 2 == 0 { max } else { -max }),
        ];
        for background in &backgrounds {
            for position in 0..GROUP_VALUES {
                for value in all_b_bit_values(bits) {
                    let mut codes: [i32; GROUP_VALUES] = std::array::from_fn(background);
                    codes[position] = value;
                    let mut bytes = vec![0xa5u8; 4 * bits as usize];
                    pack_codes(&codes, bits, &mut bytes);
                    let back = unpack_codes(&bytes, bits);
                    assert_eq!(back, codes, "q{bits}: code {value} at position {position}");
                }
            }
        }
    }
}

#[test]
fn packing_writes_exactly_the_bytes_of_the_planes() {
    // Expected bytes worked out by hand (and cross-checked with a bit-by-bit
    // script): the low four bits of the codes two per byte, the low nibble
    // first, then the planes of the high bits, each code in two's complement.
    //
    // q5, codes 1, -1, 15, -15, 7, -7, 0, 3 (00001, 11111, 01111, 10001, 00111,
    // 11001, 00000, 00011): the nibbles are 1 f f 1 7 9 0 3, and bit 4 is set
    // for the codes 1, 3 and 5, which is 0x2a in the plane of single bits.
    /// Bytes at an offset among the code bytes of a group.
    type Piece = (usize, &'static [u8]);
    let cases: [(u32, &[i32], &[Piece]); 5] = [
        (5, &[1, -1, 15, -15, 7, -7, 0, 3], &[(0, &[0xf1, 0x1f, 0x97, 0x30]), (16, &[0x2a])]),
        // q7: nibbles 1 f f 1 5 b 0 0; bits 4 and 5 are 0 3 3 0 and 0 3 0 2; bit
        // 6 is set for the codes 1, 3 and 5.
        (
            7,
            &[1, -1, 63, -63, 5, -5, 0, 32],
            &[(0, &[0xf1, 0x1f, 0xb5]), (16, &[0x3c, 0x8c]), (24, &[0x2a])],
        ),
        // q6: nibbles 1 f f 1; bits 4 and 5 are 0 3 1 2.
        (6, &[1, -1, 31, -31], &[(0, &[0xf1, 0x1f]), (16, &[0x9c])]),
        // q4: the low nibble of a byte is the earlier code.
        (4, &[1, -1, 7, -7], &[(0, &[0xf1, 0x97])]),
        // q8: one byte per code.
        (8, &[1, -1, 127, -127, -128], &[(0, &[0x01, 0xff, 0x7f, 0x81, 0x80])]),
    ];
    for (bits, head, pieces) in cases {
        let mut codes = [0i32; GROUP_VALUES];
        codes[..head.len()].copy_from_slice(head);
        let mut bytes = vec![0xa5u8; 4 * bits as usize];
        pack_codes(&codes, bits, &mut bytes);
        let mut expected = vec![0u8; 4 * bits as usize];
        for (at, piece) in pieces {
            expected[*at..*at + piece.len()].copy_from_slice(piece);
        }
        assert_eq!(bytes, expected, "q{bits}");
    }
}

#[test]
fn the_planes_of_every_width_fill_its_bytes() {
    for bits in 4..=8u32 {
        let widths = plane_widths(bits);
        assert_eq!(widths.iter().sum::<u32>(), bits, "q{bits}: the planes hold every bit");
        assert!(widths.iter().all(|w| [0, 1, 2, 4, 8].contains(w)), "q{bits}: {widths:?}");
        // The low four bits come first for every width below 8, so that plane
        // is read the same way whatever the format.
        assert_eq!(widths[0], if bits == 8 { 8 } else { 4 }, "q{bits}");
    }
}

#[test]
fn all_minus_one_codes_fill_every_bit_of_the_planes() {
    for bits in 4..=8u32 {
        let mut bytes = vec![0u8; 4 * bits as usize];
        pack_codes(&[-1; GROUP_VALUES], bits, &mut bytes);
        assert!(bytes.iter().all(|&b| b == 0xff), "q{bits}: {bytes:02x?}");
    }
}

/// The f16 pattern `bits` as f32.
fn f(bits: u16) -> f32 {
    f16_to_f32(bits)
}

#[test]
fn the_scale_is_the_smallest_f16_not_below_the_quotient() {
    // For amax across the whole f32 range (every exponent, several mantissas,
    // subnormals included) and every width: the scale is never below
    // amax/qmax, the next smaller f16 is below it (so it is the smallest), and
    // no code reaches past the range. Above the f16 range it saturates.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut mantissas = vec![0u32, 1, 0x40_0000, 0x7f_ffff, 0x12_3456];
    for _ in 0..8 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        mantissas.push((state >> 41) as u32);
    }
    let (mut checked, mut saturated) = (0, 0);
    for bits in 4..=8u32 {
        let max = code_max(bits);
        for exponent in 0..=254u32 {
            for &mantissa in &mantissas {
                let amax = f32::from_bits(exponent << 23 | mantissa);
                if amax == 0.0 {
                    continue;
                }
                let scale = scale_bits(amax, bits);
                let q = amax / max as f32;
                let what = format!("q{bits}, amax {amax:e}, scale {scale:#06x}");
                assert!((1..=F16_MAX_BITS).contains(&scale), "{what}");
                if q > F16_MAX {
                    assert_eq!(scale, F16_MAX_BITS, "saturated: {what}");
                    saturated += 1;
                    continue;
                }
                assert!(f(scale) >= q, "not below the quotient: {what}");
                assert!(scale == 1 || f(scale - 1) < q, "not the smallest: {what}");
                assert!(
                    (amax / f(scale)).round_ties_even() <= max as f32,
                    "code overflows: {what}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 9_000 && saturated > 2_000, "{checked} checked, {saturated} saturated");
}

#[test]
fn the_scale_boundary_is_exact_at_representable_quotients() {
    // When amax is exactly qmax * s0 for an f16 s0, the scale is s0 and the
    // largest code is exactly qmax. One f32 step above, the quotient is above
    // s0 and the scale is the next f16.
    let scales = [1u16, 2, 3, 0x03ff, 0x0400, 0x0401, 0x1234, 0x3c00, 0x3c01, 0x5555, 0x7bfe];
    for bits in 4..=8u32 {
        let max = code_max(bits);
        for s0 in scales {
            let amax = f(s0) * max as f32;
            assert_eq!(f(s0) * max as f32 / max as f32, f(s0), "the product is exact");
            assert_eq!(scale_bits(amax, bits), s0, "q{bits}, scale {s0:#06x}, on the boundary");
            assert_eq!((amax / f(s0)).round_ties_even(), max as f32);

            let above = f32::from_bits(amax.to_bits() + 1);
            assert_eq!(scale_bits(above, bits), s0 + 1, "q{bits}, scale {s0:#06x}, above");

            let below = f32::from_bits(amax.to_bits() - 1);
            assert_eq!(scale_bits(below, bits), s0, "q{bits}, scale {s0:#06x}, below");
        }
    }
}

#[test]
fn a_scale_rounded_to_nearest_would_overflow_the_codes_where_the_f16_is_coarse() {
    // Why the scale rounds up. Near the smallest subnormals an f16 has one or
    // two significant bits: q = 1.4 * 2^-24 is nearest to 1 * 2^-24, and dividing
    // amax by that gives 1.4 * qmax, a code of 177 in 8 bits.
    let max = code_max(8) as f32;
    let amax = 1.4 * f(1) * max;
    let nearest = f32_to_f16(amax / max);
    assert_eq!(nearest, 1);
    assert!((amax / f(nearest)).round_ties_even() > max, "nearest rounding overflows");
    // The rule's scale is the next f16, and the largest code is back in range.
    let scale = scale_bits(amax, 8);
    assert_eq!(scale, 2);
    assert!((amax / f(scale)).round_ties_even() <= max);
}

#[test]
fn the_scale_saturates_at_the_largest_finite_f16_and_not_before() {
    for bits in 4..=8u32 {
        let max = code_max(bits) as f32;
        // q = 65504 exactly is the largest finite scale, reached without saturation.
        assert_eq!(scale_bits(F16_MAX * max, bits), F16_MAX_BITS);
        // Between 65504 and 65520 rounding up would reach infinity, from 65520
        // the nearest f16 already is infinity; both saturate.
        assert_eq!(scale_bits(65_510.0 * max, bits), F16_MAX_BITS);
        assert_eq!(scale_bits(65_520.0 * max, bits), F16_MAX_BITS);
        assert_eq!(scale_bits(f32::MAX, bits), F16_MAX_BITS);
    }
}

#[test]
fn only_a_group_of_zeros_has_scale_zero() {
    for bits in 4..=8u32 {
        assert_eq!(scale_bits(0.0, bits), 0);
        // Even the smallest f32 (a subnormal whose quotient by qmax underflows to
        // zero) gets the smallest f16 subnormal, so the codes can be computed.
        assert_eq!(scale_bits(f32::from_bits(1), bits), 1, "q{bits}");
        assert_eq!(scale_bits(f32::MIN_POSITIVE, bits), 1, "q{bits}");
    }
}

#[test]
fn unpacking_sign_extends_each_width() {
    // The lowest pattern, 10..0, is -2^(B-1): not a code the encoder writes, but
    // what a decoder reads it as.
    for bits in 4..=8u32 {
        let mut codes = [0i32; GROUP_VALUES];
        codes[3] = -(1 << (bits - 1));
        let mut bytes = vec![0u8; 4 * bits as usize];
        pack_codes(&codes, bits, &mut bytes);
        assert_eq!(unpack_codes(&bytes, bits)[3], -(1 << (bits - 1)));
    }
}

#[test]
fn the_sum_of_32_pairs_the_elements_16_apart_first() {
    // 1e8 and -1e8 are 16 apart, so they meet in the first addition and cancel;
    // the thirty ones then add up exactly, to 30. A sequential sum, or one that
    // adds neighbours first, loses ones against 1e8 (the spacing of f32 there is
    // 8) and gets something else. This pins the order a warp of lanes produces
    // with five exchanges, which kernels must reproduce.
    let mut v = [1.0f32; GROUP_VALUES];
    v[0] = 1.0e8;
    v[16] = -1.0e8;
    assert_eq!(sum32(v), 30.0);
    let sequential = v.iter().fold(0.0f32, |s, x| s + x);
    assert_ne!(sequential, 30.0, "the case must tell the orders apart");
    let neighbours = {
        let mut level = v.to_vec();
        while level.len() > 1 {
            level = level.chunks(2).map(|p| p[0] + p[1]).collect();
        }
        level[0]
    };
    assert_ne!(neighbours, 30.0, "the case must tell the orders apart");

    // Spelled out for four distinct powers of two spread over the classes: the
    // pairs at distance 16, 8, 4, 2, 1 are added in this order.
    let mut v = [0.0f32; GROUP_VALUES];
    v[0] = 1.0;
    v[16] = 2.0;
    v[8] = 4.0;
    v[24] = 8.0;
    v[1] = 16.0;
    assert_eq!(sum32(v), ((1.0 + 2.0) + (4.0 + 8.0)) + 16.0);
}

#[test]
fn the_search_tries_two_four_eight_and_sixteen_reduced_scales() {
    let steps: Vec<u32> = (4..=7).rev().map(search_steps).collect();
    assert_eq!(steps, [2, 4, 8, 16], "q7, q6, q5, q4");
    // The full-range ratio of each width, qmax / (qmax + 1) = 1 - 1/2^(B-1), is
    // at 64 / 2^(B-1) steps of 1/64, half of what the search covers: the search
    // goes to twice the ratio at which the extra code pays.
    for bits in 4..=7u32 {
        assert_eq!(search_steps(bits), 2 * (SEARCH_GRID >> (bits - 1)), "q{bits}");
    }
}

/// A rotated group with the given extremes at positions 3 and 17 and a small
/// background elsewhere.
fn group_with_extremes(positive: f32, negative: f32) -> [f32; GROUP_VALUES] {
    let mut r: [f32; GROUP_VALUES] = std::array::from_fn(|i| 0.01 * ((i as f32 * 0.7).sin() + 0.1));
    r[3] = positive;
    r[17] = negative;
    r
}

#[test]
fn the_sign_of_the_scale_follows_the_larger_extreme_and_a_tie_is_positive() {
    for bits in 4..=7u32 {
        let negative_scale = |r: &[f32; GROUP_VALUES]| {
            let amax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            searched_codes(r, amax, bits).0 & 0x8000 != 0
        };
        // Equal magnitudes: the negative scale is only for a larger positive one.
        assert!(!negative_scale(&group_with_extremes(2.0, -2.0)), "q{bits}: tie");
        let above = f32::from_bits(2.0f32.to_bits() + 1);
        assert!(negative_scale(&group_with_extremes(above, -2.0)), "q{bits}: one ulp more");
        assert!(!negative_scale(&group_with_extremes(2.0, -above)), "q{bits}: negative larger");
        assert!(negative_scale(&group_with_extremes(5.0, -0.5)), "q{bits}");
        assert!(!negative_scale(&group_with_extremes(0.5, -5.0)), "q{bits}");
    }
}

#[test]
fn the_longer_end_of_the_codes_is_used_on_the_side_of_the_larger_extreme() {
    // A group that is exactly the codes -qmax-1 .. qmax times a step, with the
    // largest magnitude on the negative side (-(qmax+1) steps): a positive scale
    // of one step stores it exactly, using the code -(qmax+1) that q8's rule
    // never writes. Mirrored, the same group is stored with the negative scale.
    for bits in 4..=7u32 {
        let max = code_max(bits);
        let step = 0.5f32;
        let mut r = [0.0f32; GROUP_VALUES];
        for (i, v) in r.iter_mut().enumerate() {
            let code = -(max + 1) + (i as i32 * (2 * max + 2)) / GROUP_VALUES as i32;
            *v = code as f32 * step;
        }
        let amax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (scale, codes) = searched_codes(&r, amax, bits);
        assert_eq!(scale, 0x3800, "q{bits}: the step 0.5 as a positive scale");
        assert_eq!(codes[0], -(max + 1), "q{bits}: the longer end is used");
        for (code, v) in codes.iter().zip(&r) {
            assert_eq!(*code as f32 * 0.5, *v, "q{bits}: stored exactly");
        }

        let mirrored = r.map(|v| -v);
        let amax = mirrored.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (scale, codes) = searched_codes(&mirrored, amax, bits);
        assert_eq!(scale, 0xb800, "q{bits}: the same step as a negative scale");
        assert_eq!(codes[0], -(max + 1), "q{bits}: the longer end is used");
        for (code, v) in codes.iter().zip(&mirrored) {
            assert_eq!(*code as f32 * -0.5, *v, "q{bits}: stored exactly");
        }
    }
}

#[test]
fn a_group_whose_codes_are_all_zero_keeps_a_positive_scale_whatever_its_extreme() {
    // 3e-9 is far below half of the smallest scale, 2^-24: the scale is the
    // smallest f16 and every code is zero, so the sign carries nothing. It must
    // not depend on which extreme is larger.
    for bits in 4..=7u32 {
        for sign in [1.0f32, -1.0] {
            let mut r = [0.0f32; GROUP_VALUES];
            r[5] = sign * 3.0e-9;
            let (scale, codes) = searched_codes(&r, 3.0e-9, bits);
            assert_eq!(scale, 1, "q{bits}, extreme {sign}");
            assert_eq!(codes, [0; GROUP_VALUES]);
        }
    }
}

#[test]
fn q8_keeps_the_rule_it_always_had() {
    // The scale is the smallest f16 not below amax / qmax, always positive, and
    // the codes are clamped to -qmax ..= qmax: a group of -128 steps is stored with
    // 127 steps' worth of scale, never with the code -128.
    let mut r = [0.0f32; GROUP_VALUES];
    r[0] = -127.5;
    r[1] = 3.0;
    let (scale, codes) = absmax_codes(&r, 127.5, 8);
    assert_eq!(scale, scale_bits(127.5, 8));
    assert_eq!(scale & 0x8000, 0);
    assert!(codes.iter().all(|&c| (-127..=127).contains(&c)));
    assert_eq!(codes[0], -127);
}

#[test]
fn the_mean_of_a_tile_sums_the_rows_in_their_order_from_plus_zero_and_saturates() {
    let group = |v: f32| [v; GROUP_VALUES];
    let mean = |rows: &[f32]| mean_bits(rows.iter().map(|&v| group(v)), rows.len() as f32);

    // The plain case, with a division that rounds.
    assert_eq!(mean(&[3.0, 4.0]), [f32_to_f16(3.5); GROUP_VALUES]);
    assert_eq!(mean(&[1.0, 1.0, 2.0]), [f32_to_f16(4.0 / 3.0); GROUP_VALUES]);

    // The order of the rows is the order of the additions: 2^30 + 1 is 2^30 in
    // f32, so the 1 is lost when it is added second and kept when it is last.
    let a = 2.0f32.powi(30);
    assert_eq!(mean(&[a, 1.0, -a]), [0; GROUP_VALUES]);
    assert_eq!(mean(&[a, -a, 1.0]), [f32_to_f16(1.0 / 3.0); GROUP_VALUES]);

    // The sum starts at +0, so a single -0 has the mean +0, not -0.
    assert_eq!(mean(&[-0.0]), [0; GROUP_VALUES]);
    // A quotient that underflows keeps its sign, as the f16 conversion gives it.
    assert_eq!(mean(&[-1.0e-45, 0.0, 0.0, 0.0]), [0x8000; GROUP_VALUES]);

    // Beyond the range of f16, and when the sum itself overflows f32, the mean
    // is the largest finite f16 of its sign.
    assert_eq!(mean(&[1.0e6]), [F16_MAX_BITS; GROUP_VALUES]);
    assert_eq!(mean(&[-1.0e6, -3.0e6]), [F16_MAX_BITS | 0x8000; GROUP_VALUES]);
    assert_eq!(mean(&[f32::MAX, f32::MAX]), [F16_MAX_BITS; GROUP_VALUES]);
    assert_eq!(mean(&[f32::MIN, f32::MIN, 1.0]), [F16_MAX_BITS | 0x8000; GROUP_VALUES]);
    // The largest value that still rounds to 65504 is not infinity either.
    assert_eq!(mean(&[65519.0]), [F16_MAX_BITS; GROUP_VALUES]);
}
