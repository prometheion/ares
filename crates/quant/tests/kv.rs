//! Tests of the engine's KV storage formats (`ares_quant::kv`) through the
//! public API.
//!
//! The format is specified in the module documentation of `kv`. Besides the
//! checks of each property, this file holds an independent model of the
//! specification (`model_*`): written separately from the implementation, bit
//! by bit, with a search over the f16 patterns where the implementation rounds
//! and the sums written as a recursion, so that the two agreeing means the
//! specification and the code say the same thing. The model also has the rule
//! of `q8` (`model_absmax`), which is the rule all the formats had before `q7`
//! to `q4` got their search of the scale.
//!
//! The section "Golden fixture" is `tests/golden/kv/`: the bytes and floats the
//! GPU kernels of both backends are compared with, for two inputs (the first is
//! the one of the first version of the formats; its files for `f16` and `q8` are
//! frozen by checksum and never change). After a change to the format that is
//! meant to change those files, rewrite them with
//!
//! ```sh
//! ARES_BLESS_KV_GOLDEN=1 cargo test -p ares-quant --test kv golden
//! ```
//!
//! and review the diff of the files. A run of the whole test suite shows the
//! measured signal-to-noise ratios with `-- --nocapture`; the techniques that
//! were measured for the formats, and the policy of exact cells, are in
//! `kv_techniques.rs` and at the end of this file.

mod common;

use std::fs;
use std::path::PathBuf;

use ares_quant::{
    Error, KV_GROUP_VALUES, KV_TILE_CELLS, KvFormat, KvPolicy, decode_row, decode_row_rotated,
    decode_tile, decode_tile_rotated, encode_row, encode_tile, f16_to_f32, f32_to_f16,
    kv_bytes_per_cell, kv_mean_bytes_per_tile, rotate_groups,
};
use common::{
    Rng, channel_rows, heavy_tailed_row, mean_shifted_row, offset_rows, realistic_row, rows,
    spike_row, wide_range_rows,
};

const GROUP: usize = KV_GROUP_VALUES;
const TILE: usize = KV_TILE_CELLS;

/// The `f32` nearest to 1/sqrt(32), as the specification gives it.
const C: f32 = f32::from_bits(0x3e35_04f3);

// ---------------------------------------------------------------------------
// An independent model of the specification
// ---------------------------------------------------------------------------

fn qmax(bits: u32) -> f32 {
    ((1 << (bits - 1)) - 1) as f32
}

/// Round half to even, written out instead of calling `round_ties_even`.
fn model_rint(v: f32) -> f32 {
    let floor = v.floor();
    let fraction = v - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 || floor % 2.0 != 0.0 {
        floor + 1.0
    } else {
        floor
    }
}

/// The smallest positive f16 pattern whose value is not below `amax / qmax`,
/// found by bisection over the increasing patterns 1 ..= 0x7bff; the largest
/// one if none is.
fn model_scale(amax: f32, bits: u32) -> u16 {
    if amax == 0.0 {
        return 0;
    }
    let q = amax / qmax(bits);
    let (mut low, mut high) = (1u16, 0x7bffu16);
    if f16_to_f32(high) < q {
        return high;
    }
    while low < high {
        let mid = low + (high - low) / 2;
        if f16_to_f32(mid) >= q {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

/// The planes a code of `bits` bits is stored in, as (width, lowest bit).
fn model_planes(bits: u32) -> Vec<(u32, u32)> {
    match bits {
        8 => vec![(8, 0)],
        7 => vec![(4, 0), (2, 4), (1, 6)],
        6 => vec![(4, 0), (2, 4)],
        5 => vec![(4, 0), (1, 4)],
        4 => vec![(4, 0)],
        _ => panic!("no format has {bits} bits"),
    }
}

/// The 32 codes in their planes, one bit at a time.
fn model_pack(codes: &[i32], bits: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (width, lowest) in model_planes(bits) {
        let mut plane = Vec::new();
        for &code in codes {
            for bit in lowest..lowest + width {
                plane.push((code >> bit) & 1 == 1);
            }
        }
        bytes.extend(
            plane
                .chunks(8)
                .map(|byte| byte.iter().enumerate().map(|(i, &b)| u8::from(b) << i).sum::<u8>()),
        );
    }
    bytes
}

fn model_unpack(bytes: &[u8], bits: u32) -> Vec<i32> {
    let mut codes = vec![0i32; 32];
    let mut at = 0;
    for (width, lowest) in model_planes(bits) {
        let plane = &bytes[at..at + 4 * width as usize];
        for (i, code) in codes.iter_mut().enumerate() {
            for bit in 0..width {
                let n = i * width as usize + bit as usize;
                *code |= i32::from((plane[n / 8] >> (n % 8)) & 1) << (lowest + bit);
            }
        }
        at += plane.len();
    }
    codes
        .into_iter()
        .map(|raw| if raw >= 1 << (bits - 1) { raw - (1 << bits) } else { raw })
        .collect()
}

/// The sum of 32 values in the order of the specification: the additions of
/// elements 16 apart first, then 8, 4, 2 and 1. Written as a recursion on the
/// low bits of the indices: the sum of the elements whose index ends in a
/// given pattern of `m` bits is the sum of the two classes that extend it by one
/// more bit, so the last addition joins even indices to odd ones and the first
/// joins `i` to `i + 16`.
fn model_sum32(v: &[f32]) -> f32 {
    fn class(v: &[f32], pattern: usize, bits: u32) -> f32 {
        if bits == 5 {
            v[pattern]
        } else {
            class(v, pattern, bits + 1) + class(v, pattern + (1 << bits), bits + 1)
        }
    }
    assert_eq!(v.len(), 32);
    class(v, 0, 0)
}

/// The f16 pattern nearest to `x` (positive, below 65520), ties to the even
/// pattern, found by bisection on the increasing patterns instead of by
/// converting; 0x7bff for anything beyond the largest finite f16.
fn model_nearest_f16(x: f32) -> u16 {
    if f16_to_f32(0x7bff) <= x {
        return 0x7bff;
    }
    // The largest pattern whose value is at most x.
    let (mut low, mut high) = (0u16, 0x7bffu16);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if f16_to_f32(mid) <= x {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let (below, above) = (f64::from(x) - f64::from(f16_to_f32(low)), {
        f64::from(f16_to_f32(low + 1)) - f64::from(x)
    });
    if below < above || (below == above && low % 2 == 0) { low } else { low + 1 }
}

/// How many scales the search tries after the first: `2^(8-B)`.
fn model_steps(bits: u32) -> u32 {
    2u32.pow(8 - bits)
}

/// The scale (stored f16 pattern, sign included) and codes of one rotated group
/// of a `q7` to `q4` format, by step 3 of the specification.
fn model_search(r: &[f32], bits: u32) -> (u16, Vec<i32>) {
    let q = qmax(bits);
    let amax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return (0, vec![0; r.len()]);
    }
    let positive_max = r.iter().fold(0.0f32, |m, v| m.max(*v));
    let negative_max = r.iter().fold(0.0f32, |m, v| m.max(-*v));
    // The longer side of the codes, -q - 1, goes to the larger extreme.
    let negative_scale = positive_max > negative_max;

    let mut best: Option<(f32, u16, Vec<i32>)> = None;
    for j in 0..=model_steps(bits) {
        let base = model_scale(amax * ((64 - j) as f32 / 64.0), bits);
        let grid = f16_to_f32(base) * if negative_scale { -1.0 } else { 1.0 };
        let codes: Vec<i32> =
            r.iter().map(|&v| model_rint(v / grid).clamp(-q - 1.0, q) as i32).collect();
        let n: i32 = codes.iter().map(|c| c * c).sum();
        let products: Vec<f32> = r.iter().zip(&codes).map(|(&v, &c)| v * c as f32).collect();
        let magnitude = if n == 0 {
            base
        } else {
            let fitted = (model_sum32(&products).abs() / n as f32).min(65504.0);
            model_nearest_f16(fitted).max(1)
        };
        let scale = if n > 0 && negative_scale { magnitude | 0x8000 } else { magnitude };
        let s = f16_to_f32(scale);
        let squares: Vec<f32> = r
            .iter()
            .zip(&codes)
            .map(|(&v, &c)| {
                let d = v - c as f32 * s;
                d * d
            })
            .collect();
        let error = model_sum32(&squares);
        if best.as_ref().is_none_or(|(least, ..)| error < *least) {
            best = Some((error, scale, codes));
        }
    }
    let (_, scale, codes) = best.unwrap();
    (scale, codes)
}

/// The scale and codes of one rotated group by the rule of `q8`, which is the
/// rule the formats had before the techniques: the scale from the largest
/// magnitude, rounded up, and the symmetric range.
fn model_absmax(r: &[f32], bits: u32) -> (u16, Vec<i32>) {
    let amax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = model_scale(amax, bits);
    let s = f16_to_f32(scale);
    let codes =
        r.iter()
            .map(|&v| {
                if scale == 0 { 0 } else { model_rint(v / s).clamp(-qmax(bits), qmax(bits)) as i32 }
            })
            .collect();
    (scale, codes)
}

/// Encodes a row by the specification of the format with `bits` bits: the rule of
/// `q8` for 8 bits, the search for the others.
fn model_encode(bits: u32, x: &[f32]) -> Vec<u8> {
    model_encode_with(bits, x, true, bits < 8)
}

/// Encodes a row with the rotation or without it, by the search or by the rule
/// of `q8`. Without the rotation and without the search this is the plain "scale
/// by the largest magnitude of the group" quantizer the rotation is compared
/// with.
fn model_encode_with(bits: u32, x: &[f32], rotate: bool, search: bool) -> Vec<u8> {
    let mut out = Vec::new();
    for group in x.chunks(GROUP) {
        let mut r = group.to_vec();
        if rotate {
            rotate_groups(&mut r).unwrap();
        }
        let (scale, codes) = if search { model_search(&r, bits) } else { model_absmax(&r, bits) };
        out.extend(scale.to_le_bytes());
        out.extend(model_pack(&codes, bits));
    }
    out
}

/// The rotated values `code * scale` of a stored row, by the specification.
fn model_decode_rotated(bits: u32, bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    for group in bytes.chunks(2 + 4 * bits as usize) {
        let s = f16_to_f32(u16::from_le_bytes([group[0], group[1]]));
        out.extend(model_unpack(&group[2..], bits).into_iter().map(|c| c as f32 * s));
    }
    out
}

/// What a quantizer returns for `x`: encoded and decoded, in the original domain.
fn model_round_trip(bits: u32, x: &[f32], rotate: bool, search: bool) -> Vec<f32> {
    let mut y = model_decode_rotated(bits, &model_encode_with(bits, x, rotate, search));
    if rotate {
        rotate_groups(&mut y).unwrap();
    }
    y
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bits_of(format: KvFormat) -> u32 {
    format.quant_bits().expect("a quantized format")
}

fn quantized_formats() -> impl Iterator<Item = KvFormat> {
    KvFormat::ALL.into_iter().filter(|f| f.is_rotated())
}

fn encode(format: KvFormat, x: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0; format.row_bytes(x.len()).unwrap()];
    encode_row(format, x, &mut bytes).unwrap();
    bytes
}

fn decode_rotated(format: KvFormat, bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / format.group_bytes() * GROUP;
    let mut out = vec![0.0; n];
    decode_row_rotated(format, bytes, &mut out).unwrap();
    out
}

fn decode(format: KvFormat, bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / format.group_bytes() * GROUP;
    let mut out = vec![0.0; n];
    decode_row(format, bytes, &mut out).unwrap();
    out
}

fn rotated(x: &[f32]) -> Vec<f32> {
    let mut r = x.to_vec();
    rotate_groups(&mut r).unwrap();
    r
}

/// The stored scale of group `g` of a `qB` row, widened.
fn stored_scale(format: KvFormat, bytes: &[u8], g: usize) -> f32 {
    let at = g * format.group_bytes();
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

fn bits32(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

// ---------------------------------------------------------------------------
// Sizes and names
// ---------------------------------------------------------------------------

#[test]
fn every_format_has_the_group_and_row_sizes_of_the_specification() {
    // name, bytes per group, bits per value, bytes of a 128-value and of a
    // 256-value row.
    let table: [(KvFormat, &str, usize, f32, usize, usize); 6] = [
        (KvFormat::F16, "f16", 64, 16.0, 256, 512),
        (KvFormat::Q8, "q8", 34, 8.5, 136, 272),
        (KvFormat::Q7, "q7", 30, 7.5, 120, 240),
        (KvFormat::Q6, "q6", 26, 6.5, 104, 208),
        (KvFormat::Q5, "q5", 22, 5.5, 88, 176),
        (KvFormat::Q4, "q4", 18, 4.5, 72, 144),
    ];
    assert_eq!(table.map(|row| row.0), KvFormat::ALL, "the list is in order of the ladder");
    for (format, name, group, bits_per_value, row128, row256) in table {
        assert_eq!(format.name(), name);
        assert_eq!(format.group_bytes(), group, "{name}");
        assert_eq!(format.bits_per_value(), bits_per_value, "{name}");
        assert_eq!(format.row_bytes(32), Ok(group), "{name}");
        assert_eq!(format.row_bytes(128), Ok(row128), "{name}");
        assert_eq!(format.row_bytes(256), Ok(row256), "{name}");
        assert_eq!(format.row_bytes(0), Ok(0), "{name}");
        // B + 0.5 bits per value, and the layout: a 2-byte scale and 4 B bytes of codes.
        if let Some(bits) = format.quant_bits() {
            assert_eq!(group, 2 + 4 * bits as usize, "{name}");
            assert_eq!(bits_per_value, bits as f32 + 0.5, "{name}");
            assert_eq!(KvFormat::quantized(bits), Some(format), "{name}");
            assert!(format.is_rotated(), "{name}");
        } else {
            assert!(!format.is_rotated(), "{name}");
        }
    }
    for bits in [0, 1, 2, 3, 9, 16, 32] {
        assert_eq!(KvFormat::quantized(bits), None, "{bits} bits is not on the ladder");
    }
}

#[test]
fn a_row_must_be_a_whole_number_of_groups() {
    for format in KvFormat::ALL {
        for n in [1, 31, 33, 100, 255] {
            assert_eq!(format.row_bytes(n), Err(Error::KvPartialGroup { values: n }), "{format}");
        }
    }
    let message = KvFormat::Q4.row_bytes(33).unwrap_err().to_string();
    assert!(message.contains("33") && message.contains("32"), "{message}");
    // A size past usize does not wrap into a plausible one.
    assert_eq!(KvFormat::F16.row_bytes(usize::MAX - usize::MAX % 32), Ok(usize::MAX));
}

#[test]
fn names_round_trip_through_from_str_and_display() {
    let names: Vec<&str> = KvFormat::ALL.iter().map(|f| f.name()).collect();
    assert_eq!(names, ["f16", "q8", "q7", "q6", "q5", "q4"]);
    for format in KvFormat::ALL {
        assert_eq!(format.to_string().parse::<KvFormat>(), Ok(format));
        assert_eq!(format.name().parse::<KvFormat>(), Ok(format));
        assert_eq!(format!("{format}"), format.name());
    }
}

#[test]
fn unknown_names_are_rejected_with_the_list_of_valid_ones() {
    for name in
        ["q3", "q9", "q4_0", "Q8", "F16", "q8_0", "q5_1", "bf16", "f32", "int8", "", " q8", "q8 "]
    {
        let err = name.parse::<KvFormat>().unwrap_err();
        assert_eq!(err, Error::KvFormatName { name: name.to_owned() });
        let message = err.to_string();
        assert!(message.contains(&format!("{name:?}")), "names the input: {message}");
        assert!(message.contains("f16, q8, q7, q6, q5, q4"), "lists the valid names: {message}");
    }
}

#[test]
fn bytes_per_cell_adds_k_and_v_of_every_kv_head() {
    // 2 KV heads of 256 values: the row of one head in each format.
    let row = |f: KvFormat| {
        [512, 272, 240, 208, 176, 144][KvFormat::ALL.iter().position(|&x| x == f).unwrap()]
    };
    for k in KvFormat::ALL {
        for v in KvFormat::ALL {
            let cell = kv_bytes_per_cell(k, v, 2, 256).unwrap();
            assert_eq!(cell, 2 * (row(k) + row(v)), "K {k}, V {v}");
        }
    }
    assert_eq!(kv_bytes_per_cell(KvFormat::F16, KvFormat::F16, 2, 256), Ok(2048));
    assert_eq!(kv_bytes_per_cell(KvFormat::Q8, KvFormat::Q8, 2, 256), Ok(1088));
    assert_eq!(kv_bytes_per_cell(KvFormat::Q8, KvFormat::Q4, 2, 256), Ok(832));
    assert_eq!(kv_bytes_per_cell(KvFormat::Q4, KvFormat::Q4, 2, 256), Ok(576));
    assert_eq!(kv_bytes_per_cell(KvFormat::Q5, KvFormat::Q4, 0, 256), Ok(0));
    assert_eq!(
        kv_bytes_per_cell(KvFormat::Q5, KvFormat::Q4, 1, 100),
        Err(Error::KvPartialGroup { values: 100 })
    );
    assert_eq!(kv_bytes_per_cell(KvFormat::F16, KvFormat::F16, usize::MAX, 256), Ok(usize::MAX));
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// Builds an input whose rotation is known: `known` rotated values at chosen
/// positions, zero elsewhere. The rotation is its own inverse, so rotating the
/// wanted values gives the input, up to float rounding of a few ulps. That noise
/// is far below the half step that the codes are rounded with, so the codes the
/// encoder finds are the ones asked for.
fn input_for_rotated(known: &[(usize, f32)]) -> Vec<f32> {
    let mut r = vec![0.0f32; GROUP];
    for &(at, value) in known {
        r[at] = value;
    }
    rotated(&r)
}

#[test]
fn every_code_at_every_position_survives_encode_and_decode() {
    // For each width, every code value (the extremes -qmax and +qmax included) at
    // each of the 32 positions, whose bit-stream offsets i*B cross byte
    // boundaries at different bits. A second value, +-qmax at another position,
    // fixes the scale, so the wanted code k is a multiple of it: the input is
    // built in rotated space (see `input_for_rotated`) and lies exactly on the
    // grid, where the first candidate of the search has no error and the others
    // have some. Checked three ways: the codes inside the stored bytes (against a
    // bit-by-bit packer), the decoded rotated value, and that nothing leaks into
    // the other 30 values. The encoder may store the group with a negative scale
    // and the negated codes (it does when the larger extreme is positive), which
    // is the same group.
    for format in quantized_formats() {
        let bits = bits_of(format);
        let max = qmax(bits) as i32;
        for step in [0.0625f32, 1.0, 20.0] {
            for position in 0..GROUP {
                for code in -max..=max {
                    let anchor = (position + 13) % GROUP;
                    let anchor_code = if position % 2 == 0 { max } else { -max };
                    let x = input_for_rotated(&[
                        (anchor, anchor_code as f32 * step),
                        (position, code as f32 * step),
                    ]);
                    let bytes = encode(format, &x);

                    let mut expected_codes = vec![0i32; GROUP];
                    expected_codes[anchor] = anchor_code;
                    expected_codes[position] = code;
                    let what = format!("{format}, step {step}, code {code} at {position}");

                    let s = stored_scale(format, &bytes, 0);
                    assert!(
                        s.abs() >= step * 0.999_999 && s.abs() <= step * 1.002,
                        "scale {s}: {what}"
                    );
                    let stored_codes: Vec<i32> = if s < 0.0 {
                        expected_codes.iter().map(|c| -c).collect()
                    } else {
                        expected_codes.clone()
                    };
                    assert_eq!(&bytes[2..], model_pack(&stored_codes, bits), "{what}");

                    let decoded = decode_rotated(format, &bytes);
                    for (i, &v) in decoded.iter().enumerate() {
                        assert_eq!(v, expected_codes[i] as f32 * s.abs(), "value {i}: {what}");
                    }
                }
            }
        }
    }
}

#[test]
fn stored_bytes_follow_the_layout_of_the_specification() {
    // Hand-made groups decode as the layout says. The scale is the first two
    // bytes little endian; the codes follow in planes (the low four bits two
    // per byte, then the planes of the high bits), in two's complement. The
    // pieces are (offset among the code bytes, bytes); the rest is zero.
    let group = |format: KvFormat, scale: u16, pieces: &[(usize, &[u8])]| {
        let mut bytes = vec![0u8; format.group_bytes()];
        bytes[..2].copy_from_slice(&scale.to_le_bytes());
        for (at, piece) in pieces {
            bytes[2 + at..2 + at + piece.len()].copy_from_slice(piece);
        }
        decode_rotated(format, &bytes)
    };
    // q4, scale 1.0: the low nibble of a byte is the earlier code.
    let q4 = group(KvFormat::Q4, 0x3c00, &[(0, &[0xf1, 0x97, 0x08])]);
    assert_eq!(&q4[..6], &[1.0, -1.0, 7.0, -7.0, -8.0, 0.0], "-8 is the unused pattern 1000");
    // q8, scale 0.5: one signed byte per code.
    let q8 = group(KvFormat::Q8, 0x3800, &[(0, &[0x01, 0xff, 0x7f, 0x81, 0x80])]);
    assert_eq!(&q8[..6], &[0.5, -0.5, 63.5, -63.5, -64.0, 0.0]);
    // q5, scale 2.0: codes 1, -1, 15, -15, 7, -7, 0, 3. Their low nibbles, and
    // in the plane of single bits at byte 16 the top bits of codes 1, 3 and 5.
    let q5 = group(KvFormat::Q5, 0x4000, &[(0, &[0xf1, 0x1f, 0x97, 0x30]), (16, &[0x2a])]);
    assert_eq!(&q5[..9], &[2.0, -2.0, 30.0, -30.0, 14.0, -14.0, 0.0, 6.0, 0.0]);
    // q7, scale 1.0: nibbles, then bits 4 and 5 four codes per byte at byte 16,
    // then bit 6 at byte 24.
    let q7 = group(
        KvFormat::Q7,
        0x3c00,
        &[(0, &[0xf1, 0x1f, 0xb5]), (16, &[0x3c, 0x8c]), (24, &[0x2a])],
    );
    assert_eq!(&q7[..9], &[1.0, -1.0, 63.0, -63.0, 5.0, -5.0, 0.0, 32.0, 0.0]);
    // q6, scale 1.0: nibbles, then bits 4 and 5 at byte 16.
    let q6 = group(KvFormat::Q6, 0x3c00, &[(0, &[0xf1, 0x1f]), (16, &[0x9c])]);
    assert_eq!(&q6[..5], &[1.0, -1.0, 31.0, -31.0, 0.0]);
    // f16: little-endian halves; 0x3c00 is 1.0, 0xc000 is -2.0.
    let mut half = vec![0x00, 0x3c, 0x00, 0xc0];
    half.resize(64, 0);
    assert_eq!(&decode_rotated(KvFormat::F16, &half)[..3], &[1.0, -2.0, 0.0]);
}

#[test]
fn the_decoder_uses_the_scale_as_stored() {
    // The scale is read as it stands: a NaN one gives NaN values, a negative one
    // negative values, and the decoder does not special-case anything.
    let mut bytes = 0x7e00u16.to_le_bytes().to_vec();
    bytes.resize(KvFormat::Q4.group_bytes(), 0x11);
    let out = decode_rotated(KvFormat::Q4, &bytes);
    assert!(out.iter().all(|v| v.is_nan()));

    let mut bytes = 0xbc00u16.to_le_bytes().to_vec(); // -1.0
    bytes.resize(KvFormat::Q4.group_bytes(), 0x11);
    assert!(decode_rotated(KvFormat::Q4, &bytes).iter().all(|&v| v == -1.0));
}

#[test]
fn the_encoder_agrees_with_the_independent_model_on_many_rows() {
    // The model rounds with its own code, searches for the scale of the f16
    // pattern instead of rounding up, sums in the order of the specification
    // written as a recursion, and packs bit by bit. Rows of all the shapes used
    // below, plus a hostile one: magnitudes from 1e-9 to 1e4 within a row, ties
    // between codes (small integers), and a row at the edge of the f16 scale
    // range.
    let mut rng = Rng(7);
    let mut data = rows(&mut rng, 24, 256, realistic_row);
    data.extend(rows(&mut rng, 24, 128, spike_row));
    data.extend(channel_rows(&mut rng, 24, 256));
    data.extend(wide_range_rows(&mut rng, 24, 256));
    data.extend(rows(&mut rng, 24, 256, heavy_tailed_row));
    data.extend(rows(&mut rng, 24, 256, mean_shifted_row));
    data.extend(offset_rows(&mut rng, 24, 256));
    data.push((0..128).map(|i| ((i * 7) % 13) as f32 - 6.0).collect());
    data.push((0..64).map(|i| (i % 5) as f32 * 1.0e6).collect());
    data.push((0..64).map(|i| 3.0e7 * (i as f32 - 20.0)).collect());

    for format in quantized_formats() {
        let bits = bits_of(format);
        for (n, x) in data.iter().enumerate() {
            let bytes = encode(format, x);
            assert_eq!(bytes, model_encode(bits, x), "{format}, row {n}");
            let rotated_back = decode_rotated(format, &bytes);
            assert_eq!(bits32(&rotated_back), bits32(&model_decode_rotated(bits, &bytes)));
        }
    }
}

#[test]
fn q8_is_still_the_codec_it_was_before_the_techniques() {
    // q8 and f16 are the default and stay exactly as they were: the scale from
    // the largest magnitude rounded up, the symmetric codes, always a positive
    // scale. The old rule is the model's `model_absmax`; the golden fixtures of
    // f16 and q8 are frozen below by checksum.
    let mut rng = Rng(8);
    let mut data = rows(&mut rng, 40, 256, realistic_row);
    data.extend(rows(&mut rng, 40, 128, spike_row));
    data.extend(channel_rows(&mut rng, 40, 256));
    data.extend(wide_range_rows(&mut rng, 40, 256));
    data.extend(rows(&mut rng, 40, 256, mean_shifted_row));
    for x in &data {
        let bytes = encode(KvFormat::Q8, x);
        assert_eq!(bytes, model_encode_with(8, x, true, false));
        for group in bytes.chunks(KvFormat::Q8.group_bytes()) {
            let scale = u16::from_le_bytes([group[0], group[1]]);
            assert_eq!(scale & 0x8000, 0, "a q8 scale is never negative");
            assert!(
                group[2..].iter().all(|&b| b != 0x80),
                "the code -128 is never written by q8: {group:02x?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

#[test]
fn rotation_is_the_orthonormal_hadamard_transform_in_natural_order() {
    // Against the definition y_k = C * sum_j (-1)^popcount(j & k) x_j, summed in
    // f64: the butterfly network matches it up to f32 rounding, which fixes
    // the ordering of the outputs (natural, not sequency) and the signs.
    let mut rng = Rng(11);
    for _ in 0..50 {
        let x: Vec<f32> = (0..GROUP).map(|_| 3.0 * rng.gaussian()).collect();
        let norm = dot(&x, &x).sqrt();
        let y = rotated(&x);
        for (k, &yk) in y.iter().enumerate() {
            let exact: f64 =
                (0..GROUP)
                    .map(|j| {
                        if (j & k).count_ones() % 2 == 0 {
                            f64::from(x[j])
                        } else {
                            -f64::from(x[j])
                        }
                    })
                    .sum::<f64>()
                    * f64::from(C);
            assert!((f64::from(yk) - exact).abs() <= 1e-6 * norm, "output {k}: {} vs {exact}", yk);
        }
    }
}

#[test]
fn rotation_gives_the_exact_bits_of_the_butterfly_on_simple_inputs() {
    // The butterflies add and subtract small integers exactly, so the only
    // rounding is the final multiply by C. These are the bit patterns a kernel
    // must reproduce.
    let mut unit = [0.0f32; GROUP];
    unit[0] = 1.0;
    let r = rotated(&unit);
    assert!(r.iter().all(|v| v.to_bits() == C.to_bits()), "e_0 rotates to C everywhere");

    let r = rotated(&[1.0; GROUP]);
    assert_eq!(r[0].to_bits(), (32.0 * C).to_bits(), "a constant row is a single peak");
    assert!(r[1..].iter().all(|v| v.to_bits() == 0), "and +0.0 elsewhere");

    // e_5 rotates to C * (-1)^popcount(5 & k).
    let mut unit = [0.0f32; GROUP];
    unit[5] = 1.0;
    for (k, v) in rotated(&unit).iter().enumerate() {
        let sign = if (5 & k).count_ones() % 2 == 0 { C } else { -C };
        assert_eq!(v.to_bits(), sign.to_bits(), "output {k}");
    }

    // The order of the stages matters: with these values, summing in the
    // specified order (width 1 first) gives exactly 0 in output 0, where
    // width 16 first would give 2 * C. The expected values are the same
    // operations written out by hand.
    let mut x = [0.0f32; GROUP];
    x[..4].copy_from_slice(&[1.0e8, 1.0, -1.0e8, 1.0]);
    let r = rotated(&x);
    let width1 = [x[0] + x[1], x[0] - x[1], x[2] + x[3], x[2] - x[3]];
    let width2 = [
        width1[0] + width1[2],
        width1[1] + width1[3],
        width1[0] - width1[2],
        width1[1] - width1[3],
    ];
    for k in 0..4 {
        assert_eq!(r[k].to_bits(), (width2[k] * C).to_bits(), "output {k}");
    }
    assert_eq!(r[0], 0.0);
}

#[test]
fn rotating_twice_returns_the_input_within_float_rounding() {
    let mut rng = Rng(21);
    let mut worst = 0.0f64;
    for x in
        rows(&mut rng, 40, 256, realistic_row).iter().chain(&rows(&mut rng, 40, 256, spike_row))
    {
        let mut y = x.clone();
        rotate_groups(&mut y).unwrap();
        rotate_groups(&mut y).unwrap();
        for (group_x, group_y) in x.chunks(GROUP).zip(y.chunks(GROUP)) {
            let norm = dot(group_x, group_x).sqrt();
            for (a, b) in group_x.iter().zip(group_y) {
                worst = worst.max((f64::from(*a) - f64::from(*b)).abs() / norm);
            }
        }
    }
    println!("rotating twice: worst error {worst:.2e} of the norm of the group");
    assert!(worst < 1e-6, "error {worst:e}");
}

#[test]
fn rotation_preserves_the_dot_product_of_two_rows() {
    // The property attention relies on: q.k equals (Rq).(Rk). Float rounding
    // moves the two sides apart by about 1e-7 of |q||k|; the tolerance is 1e-6.
    let mut rng = Rng(31);
    let mut worst = 0.0f64;
    for _ in 0..200 {
        let q = realistic_row(&mut rng, 256);
        let k = realistic_row(&mut rng, 256);
        let before = dot(&q, &k);
        let after = dot(&rotated(&q), &rotated(&k));
        worst = worst.max((before - after).abs() / (dot(&q, &q).sqrt() * dot(&k, &k).sqrt()));
    }
    println!("dot product under rotation: worst error {worst:.2e} of |q||k|");
    assert!(worst < 1e-6, "error {worst:e}");
}

#[test]
fn decode_row_is_the_rotation_of_decode_row_rotated_exactly() {
    let mut rng = Rng(41);
    for x in
        rows(&mut rng, 20, 256, realistic_row).iter().chain(&rows(&mut rng, 20, 128, spike_row))
    {
        for format in quantized_formats() {
            let bytes = encode(format, x);
            let mut expected = decode_rotated(format, &bytes);
            rotate_groups(&mut expected).unwrap();
            assert_eq!(bits32(&decode(format, &bytes)), bits32(&expected), "{format}");
        }
        // f16 is stored as it is, so both decoders return the same values.
        let bytes = encode(KvFormat::F16, x);
        assert_eq!(
            bits32(&decode(KvFormat::F16, &bytes)),
            bits32(&decode_rotated(KvFormat::F16, &bytes))
        );
    }
}

#[test]
fn rotation_refuses_a_partial_group_and_changes_nothing() {
    for n in [1, 31, 33, 255] {
        let mut x = vec![1.5f32; n];
        assert_eq!(rotate_groups(&mut x), Err(Error::KvPartialGroup { values: n }));
        assert!(x.iter().all(|&v| v == 1.5), "{n} values were touched");
    }
    assert_eq!(rotate_groups(&mut []), Ok(()));
}

// ---------------------------------------------------------------------------
// Accuracy
// ---------------------------------------------------------------------------

/// 10 log10 of signal power over error power of `y` as an approximation of `x`.
fn snr_db(x: &[Vec<f32>], y: &[Vec<f32>]) -> f64 {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (row, approx) in x.iter().zip(y) {
        for (a, b) in row.iter().zip(approx) {
            signal += f64::from(*a) * f64::from(*a);
            noise += (f64::from(*a) - f64::from(*b)).powi(2);
        }
    }
    10.0 * (signal / noise).log10()
}

fn mse(x: &[Vec<f32>], y: &[Vec<f32>]) -> f64 {
    let n: usize = x.iter().map(Vec::len).sum();
    x.iter()
        .zip(y)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        / n as f64
}

/// The stored rows round-tripped through `format`, in the original domain.
fn round_trip(format: KvFormat, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    data.iter().map(|x| decode(format, &encode(format, x))).collect()
}

#[test]
fn every_q8_value_is_within_half_a_step_in_rotated_space() {
    // The bound that makes the error of q8 predictable: after the rotation, each
    // decoded value is within half the stored scale of the rotated input. The
    // tolerance of 2^-17 steps more is the rounding of the one f32 division (the
    // quotient is below 128, where half an ulp is 2^-18). Rows with one huge
    // value, rows whose groups span 14 orders of magnitude, and groups that need
    // the rounded-up scale are included. The searched widths clip on purpose and
    // do not have this bound; what they have is the next test.
    let mut rng = Rng(51);
    let mut data = rows(&mut rng, 40, 256, realistic_row);
    data.extend(rows(&mut rng, 40, 256, spike_row));
    data.extend(channel_rows(&mut rng, 20, 256));
    data.extend(wide_range_rows(&mut rng, 40, 256));
    let format = KvFormat::Q8;
    let mut worst = 0.0f64;
    for x in &data {
        let bytes = encode(format, x);
        let r = rotated(x);
        let d = decode_rotated(format, &bytes);
        for g in 0..x.len() / GROUP {
            let s = f64::from(stored_scale(format, &bytes, g));
            for i in g * GROUP..(g + 1) * GROUP {
                let error = (f64::from(r[i]) - f64::from(d[i])).abs();
                assert!(
                    error <= s * (0.5 + 2f64.powi(-17)),
                    "{format}: group {g} value {i}: error {error:e} of step {s:e}"
                );
                if s > 0.0 {
                    worst = worst.max(error / s);
                }
            }
        }
    }
    println!("{format}: worst error {worst:.4} steps (bound 0.5)");
}

#[test]
fn no_group_of_q7_to_q4_is_stored_worse_than_with_the_largest_magnitude_scale() {
    // The first candidate of the search is the scale from the largest magnitude
    // that the formats had before, with the same codes, and the fitting of the
    // scale and the choice of the best candidate can only lower the squared error
    // of a group. So in every group, in the rotated domain where the error is
    // measured, the stored group is at least as good as the old one (to the
    // rounding of the f32 sums that choose between candidates), and on real-like
    // data it is better in most.
    let mut rng = Rng(53);
    let mut data = rows(&mut rng, 30, 256, realistic_row);
    data.extend(rows(&mut rng, 30, 256, spike_row));
    data.extend(channel_rows(&mut rng, 30, 256));
    data.extend(wide_range_rows(&mut rng, 30, 256));
    data.extend(rows(&mut rng, 30, 256, heavy_tailed_row));
    data.extend(rows(&mut rng, 30, 256, mean_shifted_row));
    data.extend(offset_rows(&mut rng, 30, 256));
    for format in quantized_formats().filter(|f| bits_of(*f) < 8) {
        let bits = bits_of(format);
        let (mut better, mut equal, mut groups) = (0, 0, 0);
        for x in &data {
            let r = rotated(x);
            let new = decode_rotated(format, &encode(format, x));
            let old = model_decode_rotated(bits, &model_encode_with(bits, x, true, false));
            for g in 0..x.len() / GROUP {
                let range = g * GROUP..(g + 1) * GROUP;
                let error = |d: &[f32]| -> f64 {
                    range.clone().map(|i| (f64::from(r[i]) - f64::from(d[i])).powi(2)).sum::<f64>()
                };
                let energy: f64 = range.clone().map(|i| f64::from(r[i]).powi(2)).sum();
                let (e_new, e_old) = (error(&new), error(&old));
                assert!(
                    e_new <= e_old * (1.0 + 1e-5) + energy * 1e-12,
                    "{format}: group {g}: {e_new:e} against {e_old:e}"
                );
                groups += 1;
                if e_new < e_old * (1.0 - 1e-5) {
                    better += 1;
                } else {
                    equal += 1;
                }
            }
        }
        println!("{format}: {better} of {groups} groups better, {equal} the same");
        assert!(better > groups / 2, "{format}: {better} of {groups} groups are better");
    }
}

#[test]
fn signal_to_noise_per_width_with_and_without_the_techniques_and_the_rotation() {
    let mut rng = Rng(61);
    let sets: [(&str, Vec<Vec<f32>>); 3] = [
        ("Gaussian with outliers", rows(&mut rng, 64, 256, realistic_row)),
        ("fixed outlier channels", channel_rows(&mut rng, 64, 256)),
        ("one large value, rest small", rows(&mut rng, 64, 256, spike_row)),
    ];

    println!();
    println!("signal-to-noise ratio of encode + decode, dB (64 rows of 256 values)");
    println!(
        "{:<30} {:>6}  {}",
        "data",
        "",
        KvFormat::ALL.map(|f| format!("{:>8}", f.name())).join(" ")
    );
    // The formats; the formats with the rule of q8 (the rotated codec before
    // the techniques); the same quantizer without the rotation.
    type Snrs<'a> = (&'a str, Vec<f64>, Vec<f64>, Vec<f64>);
    let mut table: Vec<Snrs> = Vec::new();
    for (name, data) in &sets {
        let built: Vec<f64> =
            KvFormat::ALL.iter().map(|&f| snr_db(data, &round_trip(f, data))).collect();
        let model = |rotate: bool| -> Vec<f64> {
            KvFormat::ALL
                .iter()
                .map(|f| match f.quant_bits() {
                    None => snr_db(data, &round_trip(*f, data)),
                    Some(bits) => {
                        let y: Vec<Vec<f32>> =
                            data.iter().map(|x| model_round_trip(bits, x, rotate, false)).collect();
                        snr_db(data, &y)
                    }
                })
                .collect()
        };
        let (before, without) = (model(true), model(false));
        let line = |label: &str, values: &[f64]| {
            println!(
                "{name:<30} {label:>6}  {}",
                values.iter().map(|v| format!("{v:>8.1}")).collect::<Vec<_>>().join(" ")
            );
        };
        line("built", &built);
        line("before", &before);
        line("no rot", &without);
        table.push((name, built, before, without));
    }

    // The mean squared error shrinks from q4 to q8 and f16 is better than all
    // of them, for the formats. That holds for data with the statistics of real
    // rows. A single huge value is the exception: the small values of its group
    // are lost to a degree that depends on where the grid falls against them, so
    // q6, q5 and q4 all end at about the same error, and q7, whose finer grid
    // splits the rotated group of 32 nearly equal magnitudes over two codes, is
    // worse than they are (the third data set; only the ends and the plateau
    // are asserted for it).
    for (name, data) in &sets {
        let errors: Vec<f64> =
            KvFormat::ALL.iter().map(|&f| mse(data, &round_trip(f, data))).collect();
        // ALL is f16, q8, q7, q6, q5, q4: the error grows along the list.
        if name.starts_with("one large value") {
            assert!(errors[0] < errors[1] && errors[1] < errors[2], "{name}: {errors:?}");
            let plateau = &errors[3..];
            let (low, high) = (
                plateau.iter().fold(f64::MAX, |m, e| m.min(*e)),
                plateau.iter().fold(0.0f64, |m, e| m.max(*e)),
            );
            assert!(high < 1.3 * low, "{name}: q6 to q4 should be on a plateau: {errors:?}");
        } else {
            for pair in errors.windows(2) {
                assert!(
                    pair[0] < pair[1],
                    "{name}: the error must grow from f16 to q4: {errors:?}"
                );
            }
        }
    }

    // Each step of one bit adds close to 6 dB (the step halves); with the
    // rotation the groups are close to Gaussian and it does so very evenly.
    let (_, built, ..) = &table[0];
    for pair in built[1..].windows(2) {
        let gain = pair[0] - pair[1];
        assert!((4.5..7.5).contains(&gain), "one bit should give about 6 dB, got {gain:.2}");
    }

    // The point of the rotation: on data with outliers, the rotated codec has the
    // better ratio at every width. (A single huge value among small ones is the
    // opposite case, which the third data set shows and does not assert.)
    for (name, built, _, without) in table.iter().take(2) {
        for (i, format) in KvFormat::ALL.iter().enumerate().skip(1) {
            assert!(
                built[i] > without[i] + 1.0,
                "{name}, {format}: {:.1} dB with the rotation, {:.1} dB without",
                built[i],
                without[i]
            );
        }
    }

    // The point of the techniques of q7 to q4, whose widths are indices 2 to 5:
    // on rows with outliers the formats gain over the rule q8 has, by about
    // 0.6, 0.8, 1.1 and 1.6 dB, here asserted with a margin; q8 and f16 gain
    // nothing.
    let needed = [0.4, 0.6, 0.9, 1.3];
    for (name, built, before, _) in table.iter().take(2) {
        for (i, format) in KvFormat::ALL.iter().enumerate().skip(2) {
            let gain = built[i] - before[i];
            assert!(
                gain > needed[i - 2],
                "{name}, {format}: gains {gain:.2} dB, needs {}",
                needed[i - 2]
            );
        }
        assert!((built[1] - before[1]).abs() < 1e-9, "{name}: q8 must not change");
        assert!((built[0] - before[0]).abs() < 1e-9, "{name}: f16 must not change");
    }

    // And the weakness of the rotation for one large value among small ones,
    // which the search of the scale removes at q6 and below: the error is no
    // longer the spread of the small values around the common magnitude but
    // their own power, as without the rotation.
    let (_, built, before, without) = &table[2];
    for i in 3..=5 {
        assert!(
            built[i] > before[i] + 6.0,
            "{}: {:.1} against {:.1}",
            KvFormat::ALL[i],
            built[i],
            before[i]
        );
        assert!(
            built[i] > without[i] - 1.0,
            "{}: {:.1} against {:.1}",
            KvFormat::ALL[i],
            built[i],
            without[i]
        );
    }
    assert!(built[2] > before[2], "q7 gains too: {:.1} against {:.1}", built[2], before[2]);
}

// ---------------------------------------------------------------------------
// The search of the scale
// ---------------------------------------------------------------------------

/// The codes of every group of a stored row, sign-extended, and its scales.
fn stored_groups(format: KvFormat, bytes: &[u8]) -> Vec<(f32, Vec<i32>)> {
    bytes
        .chunks(format.group_bytes())
        .map(|g| {
            let scale = f16_to_f32(u16::from_le_bytes([g[0], g[1]]));
            (scale, model_unpack(&g[2..], bits_of(format)))
        })
        .collect()
}

#[test]
fn the_stored_scale_fits_the_stored_codes_in_the_least_squares_sense() {
    // The encoder replaces the scale of the winning candidate by the one that
    // fits its codes best: |sum(r c)| / sum(c^2), rounded to the nearest f16.
    // Checked from the outside, with the sums in f64 from the rotated input and
    // the codes in the stored bytes: the stored scale is the nearest f16 to the
    // optimum within one step of f16 (the encoder's own sums are f32). This is
    // what makes the error of a group orthogonal to its decoded values, and what
    // keeps a dot product of a row with itself from being biased low or high.
    let mut rng = Rng(57);
    let mut data = rows(&mut rng, 30, 256, realistic_row);
    data.extend(rows(&mut rng, 30, 256, spike_row));
    data.extend(channel_rows(&mut rng, 30, 256));
    data.extend(wide_range_rows(&mut rng, 30, 256));
    data.extend(rows(&mut rng, 30, 256, heavy_tailed_row));
    for format in quantized_formats().filter(|f| bits_of(*f) < 8) {
        let mut checked = 0;
        for x in &data {
            let r = rotated(x);
            for (g, (scale, codes)) in stored_groups(format, &encode(format, x)).iter().enumerate()
            {
                let r = &r[g * GROUP..(g + 1) * GROUP];
                let n: f64 = codes.iter().map(|&c| f64::from(c * c)).sum();
                if n == 0.0 {
                    continue;
                }
                let p: f64 = r.iter().zip(codes).map(|(&v, &c)| f64::from(v) * f64::from(c)).sum();
                let optimum = p.abs() / n;
                if optimum >= 65_000.0 {
                    continue; // saturated scales are not fitted
                }
                let allowed = (optimum * 2f64.powi(-10)).max(2f64.powi(-24));
                assert!(
                    (f64::from(scale.abs()) - optimum).abs() <= allowed,
                    "{format}: group {g}: stored {scale}, optimum {optimum}"
                );
                checked += 1;
            }
        }
        assert!(checked > 1000, "{format}: only {checked} groups checked");
    }
}

#[test]
fn q7_to_q4_write_the_longer_end_of_the_codes_and_q8_does_not() {
    // The extra code -2^(B-1) is a plain B-bit two's complement number that every
    // decoder reads (a test above has it decoded), and the encoders of q7 to q4
    // write it, on the side of the larger extreme of a group. It is rare in
    // absolute terms but there in every kind of row; q8 never writes it.
    let mut rng = Rng(59);
    let sets = [
        rows(&mut rng, 40, 256, realistic_row),
        channel_rows(&mut rng, 40, 256),
        rows(&mut rng, 40, 256, heavy_tailed_row),
    ];
    for format in quantized_formats() {
        let lowest = -(1 << (bits_of(format) - 1));
        for data in &sets {
            let (mut longer, mut groups) = (0, 0);
            for x in data {
                for (_, codes) in stored_groups(format, &encode(format, x)) {
                    groups += 1;
                    longer += usize::from(codes.contains(&lowest));
                }
            }
            if bits_of(format) == 8 {
                assert_eq!(longer, 0, "{format}");
            } else {
                // A search at q7 tries only the first and two reduced scales and
                // takes the extra code in one group of ten; at q4 in nearly every
                // other one.
                assert!(longer * 20 > groups, "{format}: {longer} of {groups} groups");
            }
        }
    }
}

#[test]
fn the_scale_is_negative_when_the_larger_extreme_of_the_group_is_positive() {
    // Rotated groups with a +10 and a -4, and the other way round (built in
    // rotated space, see `input_for_rotated`). The sign of the stored scale puts
    // the longer end of the codes on the side of the larger extreme; the decoded
    // extreme is close to the input in both cases.
    for format in quantized_formats() {
        let searched = bits_of(format) < 8;
        for (positive, negative) in [(10.0f32, -4.0f32), (4.0, -10.0)] {
            let x = input_for_rotated(&[(5, positive), (9, negative)]);
            let bytes = encode(format, &x);
            let s = stored_scale(format, &bytes, 0);
            let expect_negative = searched && positive > -negative;
            assert_eq!(s < 0.0, expect_negative, "{format}: +{positive} -{negative}: scale {s}");
            let decoded = decode_rotated(format, &bytes);
            assert!((decoded[5] - positive).abs() < 0.05 * 10.0, "{format}: {}", decoded[5]);
            assert!((decoded[9] - negative).abs() < 0.05 * 10.0, "{format}: {}", decoded[9]);
        }
    }
}

#[test]
fn a_dot_product_with_the_rotated_query_ignores_the_sign_of_the_scale() {
    // The property attention relies on, with groups stored under either sign: the
    // dot product of the rotated query with the decoded rotated row equals that
    // of the query with the decoded row, and the stored scale's sign never
    // appears in the formula (scale times code, summed against the query).
    let mut rng = Rng(63);
    for format in quantized_formats() {
        let (mut negative, mut positive) = (0, 0);
        for _ in 0..40 {
            let key = realistic_row(&mut rng, 256);
            let query = realistic_row(&mut rng, 256);
            let bytes = encode(format, &key);
            let scales: Vec<f32> = stored_groups(format, &bytes).iter().map(|g| g.0).collect();
            negative += scales.iter().filter(|s| **s < 0.0).count();
            positive += scales.iter().filter(|s| **s > 0.0).count();
            let direct = dot(&query, &decode(format, &bytes));
            let in_rotated_space = dot(&rotated(&query), &decode_rotated(format, &bytes));
            let size = dot(&query, &query).sqrt() * dot(&key, &key).sqrt();
            assert!((direct - in_rotated_space).abs() < 1e-6 * size, "{format}");
        }
        if bits_of(format) < 8 {
            assert!(negative > 50 && positive > 50, "{format}: {negative} negative, {positive}");
        } else {
            assert_eq!(negative, 0);
        }
    }
}

#[test]
fn the_number_of_scales_tried_per_group_grows_as_the_width_shrinks() {
    let table = [
        (KvFormat::F16, 0),
        (KvFormat::Q8, 1),
        (KvFormat::Q7, 3),
        (KvFormat::Q6, 5),
        (KvFormat::Q5, 9),
        (KvFormat::Q4, 17),
    ];
    for (format, candidates) in table {
        assert_eq!(format.scale_candidates(), candidates, "{format}");
    }
}

/// One head of attention over `n` cached positions, the way a kernel does it:
/// K and V stored in their formats, the query rotated once when K is rotated,
/// `softmax * V` accumulated in the stored domain and un-rotated once when V is
/// rotated. Returns the output.
fn attention(
    k_format: KvFormat,
    v_format: KvFormat,
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    query: &[f32],
) -> Vec<f64> {
    let d = query.len();
    let mut q = query.to_vec();
    if k_format.is_rotated() {
        rotate_groups(&mut q).unwrap();
    }
    let logit_scale = 3.0 / (dot(query, query).sqrt() * dot(&keys[0], &keys[0]).sqrt());
    let logits: Vec<f64> = keys
        .iter()
        .map(|k| {
            let stored = decode_rotated(k_format, &encode(k_format, k));
            dot(&q, &stored) * logit_scale
        })
        .collect();
    let top = logits.iter().cloned().fold(f64::MIN, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut out = vec![0.0f64; d];
    for (w, v) in weights.iter().zip(values) {
        let stored = decode_rotated(v_format, &encode(v_format, v));
        for (o, x) in out.iter_mut().zip(&stored) {
            *o += w / total * f64::from(*x);
        }
    }
    if v_format.is_rotated() {
        let mut rotated_out: Vec<f32> = out.iter().map(|x| *x as f32).collect();
        rotate_groups(&mut rotated_out).unwrap();
        out = rotated_out.iter().map(|x| f64::from(*x)).collect();
    }
    out
}

#[test]
fn every_k_format_combines_with_every_v_format_in_attention() {
    // K and V are chosen separately, and the kernels do the same for any pair:
    // rotate the query when K is rotated, accumulate in the stored domain,
    // un-rotate the output when V is rotated. The output of one head over 40
    // positions, 36 pairs, against the same attention on the unquantized rows in
    // f64; the groups of q7 to q4 are stored with positive and negative scales,
    // and the formula never looks at the sign. The tolerance doubles per bit of
    // the narrower format; the measured errors are a fifth of it or less.
    let tolerance = [2e-3, 0.02, 0.04, 0.08, 0.16, 0.32];
    let mut rng = Rng(77);
    let keys = rows(&mut rng, 40, 128, realistic_row);
    let values = rows(&mut rng, 40, 128, realistic_row);
    let query = realistic_row(&mut rng, 128);

    let exact = {
        let logit_scale = 3.0 / (dot(&query, &query).sqrt() * dot(&keys[0], &keys[0]).sqrt());
        let logits: Vec<f64> = keys.iter().map(|k| dot(&query, k) * logit_scale).collect();
        let top = logits.iter().cloned().fold(f64::MIN, f64::max);
        let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut out = vec![0.0f64; 128];
        for (w, v) in weights.iter().zip(&values) {
            for (o, x) in out.iter_mut().zip(v) {
                *o += w / total * f64::from(*x);
            }
        }
        out
    };
    let size = exact.iter().map(|x| x * x).sum::<f64>().sqrt();

    let mut worst = [0.0f64; 6];
    for (i, k) in KvFormat::ALL.into_iter().enumerate() {
        for (j, v) in KvFormat::ALL.into_iter().enumerate() {
            let out = attention(k, v, &keys, &values, &query);
            let error = out.iter().zip(&exact).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            let relative = error / size;
            let narrower = i.max(j);
            worst[narrower] = worst[narrower].max(relative);
            assert!(
                relative < tolerance[narrower],
                "K {k}, V {v}: relative error {relative:.4}, tolerance {}",
                tolerance[narrower]
            );
        }
    }
    println!("worst relative error of attention by the narrower format: {worst:.4?}");
}

// ---------------------------------------------------------------------------
// The policy of first and newest cells
// ---------------------------------------------------------------------------

#[test]
fn exact_cells_default_to_none_for_f16_and_q8_and_to_the_first_tile_and_a_window_below() {
    let table = [
        (KvFormat::F16, 0, 0),
        (KvFormat::Q8, 0, 0),
        (KvFormat::Q7, 128, 32),
        (KvFormat::Q6, 128, 64),
        (KvFormat::Q5, 128, 128),
        (KvFormat::Q4, 128, 256),
    ];
    assert_eq!(table.map(|row| row.0), KvFormat::ALL, "in the order of the ladder");
    for (format, first, newest) in table {
        assert_eq!(format.default_exact_first_cells(), first, "{format}");
        assert_eq!(format.default_exact_newest_cells(), newest, "{format}");
        // The formats with a mean keep the first tile and a window of the newest
        // cells exact; the plain ones nothing.
        assert_eq!(first, if format.has_mean() { TILE } else { 0 }, "{format}");
        assert_eq!(newest > 0, format.has_mean(), "{format}");
    }
    // The least width of the newest window grows along the ladder, doubling per
    // bit from q7 to q4.
    for pair in table.windows(2) {
        assert!(pair[0].1 <= pair[1].1 && pair[0].2 <= pair[1].2);
    }
    for pair in table[2..].windows(2) {
        assert_eq!(pair[1].2, 2 * pair[0].2);
    }
}

#[test]
fn the_policy_of_a_pair_takes_the_exact_cells_of_the_narrower_format() {
    for (i, k) in KvFormat::ALL.into_iter().enumerate() {
        for (j, v) in KvFormat::ALL.into_iter().enumerate() {
            let narrower = KvFormat::ALL[i.max(j)];
            let policy = KvPolicy::new(k, v);
            assert_eq!((policy.k, policy.v), (k, v));
            assert_eq!(policy.exact_first_cells, narrower.default_exact_first_cells(), "{k}/{v}");
            assert_eq!(policy.exact_newest_cells, narrower.default_exact_newest_cells(), "{k}/{v}");
        }
    }
    // The default pair of the engine, q8 for both, has none at all.
    let q8 = KvPolicy::new(KvFormat::Q8, KvFormat::Q8);
    assert_eq!((q8.exact_first_cells, q8.exact_newest_cells), (0, 0));
    assert_eq!(q8.exact_cells(1_000_000), 0);
    assert_eq!(q8.quantized_cells(1_000_000), 0..1_000_000);
}

#[test]
fn without_a_mean_exact_cells_are_the_two_windows_up_to_the_length_of_the_sequence() {
    // q8 and f16 have no mean, so cells are stored in the formats one by one and
    // the windows are exactly as wide as the policy says.
    let policy = KvPolicy {
        exact_first_cells: 128,
        exact_newest_cells: 256,
        ..KvPolicy::new(KvFormat::Q8, KvFormat::Q8)
    };
    assert!(!policy.has_mean());
    assert_eq!(policy.quantized_cells(385), 128..129);
    assert_eq!(policy.quantized_cells(100_000), 128..99_744);
    assert!(policy.quantized_cells(384).is_empty());
    for (n, exact) in
        [(0, 0), (1, 1), (127, 127), (128, 128), (300, 300), (384, 384), (385, 384), (100_000, 384)]
    {
        assert_eq!(policy.exact_cells(n), exact, "{n} cells");
    }
    // Sums that overflow stop at the sequence.
    let huge = KvPolicy { exact_first_cells: usize::MAX, exact_newest_cells: 5, ..policy };
    assert_eq!(huge.exact_cells(7), 7);
    assert_eq!(huge.exact_cells(usize::MAX), usize::MAX);
}

#[test]
fn without_a_mean_the_exact_cells_are_the_first_and_the_newest_and_the_count_agrees() {
    let policy = KvPolicy {
        exact_first_cells: 128,
        exact_newest_cells: 256,
        ..KvPolicy::new(KvFormat::F16, KvFormat::Q8)
    };
    let n = 1000;
    for (position, exact) in [
        (0, true),
        (127, true),
        (128, false),
        (500, false),
        (743, false),
        (744, true),
        (999, true),
        (1000, false),
        (5000, false),
    ] {
        assert_eq!(policy.is_exact(position, n), exact, "position {position} of {n}");
    }
    // Counting the exact positions gives `exact_cells`, for sequences shorter
    // than a window, between the two, overlapping and long.
    for n in [0, 1, 5, 127, 128, 129, 255, 300, 383, 384, 385, 1000] {
        let counted = (0..n + 3).filter(|&p| policy.is_exact(p, n)).count();
        assert_eq!(counted, policy.exact_cells(n), "{n} cells");
    }
    // No newest window: only the first cells.
    let only_first = KvPolicy { exact_newest_cells: 0, ..policy };
    assert!(only_first.is_exact(127, 1000) && !only_first.is_exact(999, 1000));
}

#[test]
fn with_a_mean_cells_are_stored_in_the_formats_a_whole_tile_at_a_time() {
    // The default of q4: the first 128 cells and at least the newest 256 exact.
    let policy = KvPolicy::new(KvFormat::Q4, KvFormat::Q4);
    assert!(policy.has_mean());
    assert_eq!((policy.exact_first_cells, policy.exact_newest_cells), (128, 256));
    for (n, quantized) in [
        (0, 0..0),
        (1, 0..0),
        (384, 0..0),
        (385, 0..0),
        (511, 0..0),
        (512, 128..256),
        (513, 128..256),
        (639, 128..256),
        (640, 128..384),
        (1000, 128..640),
        (65_536, 128..65_280),
    ] {
        assert_eq!(policy.quantized_cells(n), quantized, "{n} cells");
        assert_eq!(policy.exact_cells(n), n - quantized.len(), "{n} cells");
    }
    // The second tile is stored in the formats when the sequence reaches 512
    // cells; from then on the newest window is 256 to 383 cells wide, and a tile
    // leaves it whole.
    for n in 512..3000 {
        let quantized = policy.quantized_cells(n);
        assert_eq!(quantized.start, TILE);
        assert_eq!(quantized.end % TILE, 0, "{n} cells");
        assert!((256..256 + TILE).contains(&(n - quantized.end)), "{n} cells");
        let counted = (0..n + 3).filter(|&p| policy.is_exact(p, n)).count();
        assert_eq!(counted, policy.exact_cells(n), "{n} cells");
        assert!(policy.is_exact(0, n) && policy.is_exact(127, n) && !policy.is_exact(128, n));
        assert!(!policy.is_exact(quantized.end - 1, n) && policy.is_exact(quantized.end, n));
        assert!(policy.is_exact(n - 1, n) && !policy.is_exact(n, n));
    }

    // Windows that are not whole tiles: a tile with an exact cell stays exact as
    // a whole, and so does the tile that is still filling.
    let one_first = KvPolicy { exact_first_cells: 1, exact_newest_cells: 0, ..policy };
    assert_eq!(one_first.quantized_cells(1000), 128..896);
    let none = KvPolicy { exact_first_cells: 0, exact_newest_cells: 0, ..policy };
    assert_eq!(none.quantized_cells(1000), 0..896);
    assert_eq!(none.quantized_cells(128), 0..128);
    assert!(none.quantized_cells(127).is_empty());
    assert_eq!(none.exact_cells(1000), 104, "the open tile");
    let one_newest = KvPolicy { exact_first_cells: 0, exact_newest_cells: 1, ..policy };
    assert_eq!(one_newest.quantized_cells(128), 0..0);
    assert_eq!(one_newest.quantized_cells(129), 0..128);

    // One side with a mean is enough: the cells are shared by K and V.
    for (k, v) in [(KvFormat::Q8, KvFormat::Q5), (KvFormat::Q7, KvFormat::F16)] {
        let mixed = KvPolicy { exact_first_cells: 0, exact_newest_cells: 0, ..KvPolicy::new(k, v) };
        assert!(mixed.has_mean());
        assert_eq!(mixed.quantized_cells(1000), 0..896, "{k}/{v}");
    }
    for (k, v) in [(KvFormat::Q8, KvFormat::Q8), (KvFormat::F16, KvFormat::Q8)] {
        let plain = KvPolicy { exact_first_cells: 0, exact_newest_cells: 0, ..KvPolicy::new(k, v) };
        assert!(!plain.has_mean());
        assert_eq!(plain.quantized_cells(1000), 0..1000, "{k}/{v}");
    }

    // Windows beyond the sequence or beyond usize leave everything exact.
    let huge = KvPolicy { exact_first_cells: usize::MAX, exact_newest_cells: 5, ..policy };
    assert_eq!(huge.exact_cells(7), 7);
    assert_eq!(huge.exact_cells(usize::MAX), usize::MAX);
    let all_newest = KvPolicy { exact_newest_cells: usize::MAX, ..policy };
    assert_eq!(all_newest.exact_cells(usize::MAX), usize::MAX);
    assert_eq!(none.quantized_cells(usize::MAX), 0..usize::MAX - usize::MAX % TILE);
}

#[test]
fn the_bytes_of_a_sequence_are_exact_cells_at_f16_the_rest_in_the_formats_and_a_mean_per_tile() {
    // 2 KV heads of 256 values: an f16 cell is 2048 bytes, a q4/q4 cell 576, a
    // q8/q4 cell 832; the means of a q4/q4 tile are 2 heads times 2 rows of 256
    // f16, 2048 bytes.
    let per_cell = |k, v| kv_bytes_per_cell(k, v, 2, 256).unwrap();
    let per_tile = |k, v| kv_mean_bytes_per_tile(k, v, 2, 256).unwrap();
    assert_eq!(per_cell(KvFormat::F16, KvFormat::F16), 2048);
    assert_eq!(per_cell(KvFormat::Q4, KvFormat::Q4), 576);
    assert_eq!(per_tile(KvFormat::Q4, KvFormat::Q4), 2048);
    assert_eq!(per_tile(KvFormat::Q7, KvFormat::Q5), 2048, "the same for every width");
    assert_eq!(per_tile(KvFormat::Q8, KvFormat::Q4), 1024, "V only");
    assert_eq!(per_tile(KvFormat::Q6, KvFormat::F16), 1024, "K only");
    assert_eq!(per_tile(KvFormat::Q8, KvFormat::Q8), 0);
    assert_eq!(per_tile(KvFormat::F16, KvFormat::F16), 0);

    let policy = KvPolicy::new(KvFormat::Q4, KvFormat::Q4);
    let bytes = |p: &KvPolicy, n| p.bytes_for_cells(n, 2, 256).unwrap();
    assert_eq!(bytes(&policy, 0), 0);
    assert_eq!(bytes(&policy, 100), 100 * 2048, "all cells exact");
    assert_eq!(bytes(&policy, 511), 511 * 2048);
    assert_eq!(bytes(&policy, 512), 384 * 2048 + 128 * 576 + 2048, "one tile and its means");
    assert_eq!(bytes(&policy, 1000), 488 * 2048 + 512 * 576 + 4 * 2048);
    assert_eq!(bytes(&policy, 65_536), 384 * 2048 + 65_152 * 576 + 509 * 2048);

    // K q5 and V q4 follow q4's windows, at their own cell size.
    let mixed = KvPolicy::new(KvFormat::Q5, KvFormat::Q4);
    assert_eq!(
        bytes(&mixed, 1000),
        488 * 2048 + 512 * per_cell(KvFormat::Q5, KvFormat::Q4) + 4 * 2048
    );
    // K q8 and V q4: tiles, with a mean for V alone.
    let half = KvPolicy::new(KvFormat::Q8, KvFormat::Q4);
    assert_eq!(bytes(&half, 1000), 488 * 2048 + 512 * 832 + 4 * 1024);

    // Formats without exact cells and without a mean cost n cells of their own
    // size.
    let default = KvPolicy::new(KvFormat::Q8, KvFormat::Q8);
    assert_eq!(bytes(&default, 5000), 5000 * 1088);
    // f16/f16 costs the same whatever the windows are.
    let f16 = KvPolicy { exact_first_cells: 999, ..KvPolicy::new(KvFormat::F16, KvFormat::F16) };
    assert_eq!(bytes(&f16, 5000), 5000 * 2048);

    // What a mean costs: 16 bits per value of a row, once per 128 cells, which is
    // 0.125 bits per value, 2.8% of a q4 tile.
    let tile = TILE * 576;
    assert_eq!(per_tile(KvFormat::Q4, KvFormat::Q4) * 8, 2 * 2 * 256 * 16);
    assert_eq!(per_tile(KvFormat::Q4, KvFormat::Q4) * 36, tile);
}

#[test]
fn the_bytes_of_a_sequence_report_a_bad_head_dimension_and_saturate() {
    let policy = KvPolicy::new(KvFormat::Q4, KvFormat::Q4);
    assert_eq!(policy.bytes_for_cells(10, 2, 100), Err(Error::KvPartialGroup { values: 100 }));
    assert_eq!(policy.bytes_for_cells(0, 2, 100), Err(Error::KvPartialGroup { values: 100 }));
    assert_eq!(policy.bytes_for_cells(10, 0, 256), Ok(0));
    assert_eq!(policy.bytes_for_cells(usize::MAX, 2, 256), Ok(usize::MAX));
    assert_eq!(policy.bytes_for_cells(10, usize::MAX, 256), Ok(usize::MAX));
}

// ---------------------------------------------------------------------------
// Tiles and their mean
// ---------------------------------------------------------------------------

fn mean_formats() -> impl Iterator<Item = KvFormat> {
    KvFormat::ALL.into_iter().filter(|f| f.has_mean())
}

/// The stored mean and the stored rows of the tile `x`, rows of `n` values.
fn encode_tile_of(format: KvFormat, n: usize, x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let mut mean = vec![0; format.mean_bytes(n).unwrap()];
    let mut bytes = vec![0; x.len() / n * format.row_bytes(n).unwrap()];
    encode_tile(format, n, x, &mut mean, &mut bytes).unwrap();
    (mean, bytes)
}

fn decode_tile_rotated_of(format: KvFormat, n: usize, mean: &[u8], bytes: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0; bytes.len() / format.row_bytes(n).unwrap() * n];
    decode_tile_rotated(format, n, mean, bytes, &mut out).unwrap();
    out
}

fn decode_tile_of(format: KvFormat, n: usize, mean: &[u8], bytes: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0; bytes.len() / format.row_bytes(n).unwrap() * n];
    decode_tile(format, n, mean, bytes, &mut out).unwrap();
    out
}

/// The rows of a tile through `format` and back, in the original domain.
fn tile_round_trip(format: KvFormat, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = data[0].len();
    let (mean, bytes) = encode_tile_of(format, n, &data.concat());
    decode_tile_of(format, n, &mean, &bytes).chunks(n).map(<[f32]>::to_vec).collect()
}

/// The f16 patterns of a stored mean.
fn mean_patterns(mean: &[u8]) -> Vec<u16> {
    mean.as_chunks::<2>().0.iter().map(|half| u16::from_le_bytes(*half)).collect()
}

/// A stored mean as the rotated f32 values it stands for.
fn mean_values(mean: &[u8]) -> Vec<f32> {
    mean_patterns(mean).into_iter().map(f16_to_f32).collect()
}

/// The f16 nearest to a signed value by the model: the magnitude through
/// `model_nearest_f16`, which saturates, and the sign bit of the value.
fn model_nearest_f16_signed(x: f32) -> u16 {
    let magnitude = model_nearest_f16(x.abs());
    if x.is_sign_negative() { magnitude | 0x8000 } else { magnitude }
}

/// The stored mean and the stored rows of a tile of a `q7` to `q4` format, by
/// the specification and apart from the implementation: all rows rotated up
/// front, the mean of each rotated position summed over the rows in their
/// order, and each row minus the stored mean through the model of the search.
fn model_encode_tile(bits: u32, n: usize, x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let rows: Vec<Vec<f32>> = x.chunks(n).map(rotated).collect();
    let mean: Vec<u16> = (0..n)
        .map(|j| {
            let mut sum = 0.0f32;
            for row in &rows {
                sum += row[j];
            }
            model_nearest_f16_signed(sum / rows.len() as f32)
        })
        .collect();
    let mut bytes = Vec::new();
    for row in &rows {
        let centred: Vec<f32> = row.iter().zip(&mean).map(|(r, m)| r - f16_to_f32(*m)).collect();
        for group in centred.chunks(GROUP) {
            let (scale, codes) = model_search(group, bits);
            bytes.extend(scale.to_le_bytes());
            bytes.extend(model_pack(&codes, bits));
        }
    }
    (mean.iter().flat_map(|m| m.to_le_bytes()).collect(), bytes)
}

#[test]
fn q7_to_q4_have_a_mean_of_one_f16_per_value_and_f16_and_q8_have_none() {
    assert_eq!(TILE, 128);
    for format in KvFormat::ALL {
        let has_mean = matches!(format.quant_bits(), Some(4..=7));
        assert_eq!(format.has_mean(), has_mean, "{format}");
        assert_eq!(format.mean_bytes(256), Ok(if has_mean { 512 } else { 0 }), "{format}");
        assert_eq!(format.mean_bytes(32), Ok(if has_mean { 64 } else { 0 }), "{format}");
        assert_eq!(format.mean_bytes(0), Ok(0), "{format}");
        assert_eq!(format.mean_bytes(100), Err(Error::KvPartialGroup { values: 100 }));
        // A size beyond usize saturates instead of wrapping.
        let huge = usize::MAX - 31;
        assert_eq!(format.mean_bytes(huge), Ok(if has_mean { usize::MAX } else { 0 }));
    }
}

#[test]
fn the_tile_encoder_agrees_with_the_independent_model_on_many_tiles() {
    // Tiles of 1 to 130 rows (the division by a number of rows that is not a
    // power of two included), rows of 1 to 8 groups, rows that share offsets,
    // rows that do not, and rows whose groups differ by orders of magnitude.
    let mut rng = Rng(0x7117);
    let mut tiles = 0;
    for (n_rows, n) in [(1, 32), (2, 64), (3, 96), (5, 256), (127, 64), (128, 128), (130, 32)] {
        let kinds = [
            offset_rows(&mut rng, n_rows, n),
            rows(&mut rng, n_rows, n, realistic_row),
            wide_range_rows(&mut rng, n_rows, n),
        ];
        for data in kinds {
            let x = data.concat();
            for format in mean_formats() {
                let (mean, bytes) = encode_tile_of(format, n, &x);
                let (model_mean, model_bytes) = model_encode_tile(bits_of(format), n, &x);
                assert_eq!(mean, model_mean, "{format}: mean of {n_rows} rows of {n}");
                assert_eq!(bytes, model_bytes, "{format}: {n_rows} rows of {n}");
                tiles += 1;
            }
        }
    }
    assert_eq!(tiles, 7 * 3 * 4);
}

#[test]
fn a_tile_of_a_format_without_a_mean_is_its_rows_one_by_one() {
    let mut rng = Rng(0x7118);
    let data = offset_rows(&mut rng, 9, 96);
    let x = data.concat();
    for format in [KvFormat::F16, KvFormat::Q8] {
        let (mean, bytes) = encode_tile_of(format, 96, &x);
        assert!(mean.is_empty(), "{format}");
        let by_row: Vec<u8> = data.iter().flat_map(|row| encode(format, row)).collect();
        assert_eq!(bytes, by_row, "{format}");
        let stored: Vec<f32> = bytes
            .chunks(format.row_bytes(96).unwrap())
            .flat_map(|b| decode_rotated(format, b))
            .collect();
        assert_eq!(bits32(&decode_tile_rotated_of(format, 96, &mean, &bytes)), bits32(&stored));
        let original: Vec<f32> =
            bytes.chunks(format.row_bytes(96).unwrap()).flat_map(|b| decode(format, b)).collect();
        assert_eq!(bits32(&decode_tile_of(format, 96, &mean, &bytes)), bits32(&original));
    }
}

#[test]
fn rows_that_cancel_have_a_mean_of_zero_and_are_stored_as_rows_on_their_own() {
    // The rotation of -x is the negated rotation of x, so the two rows of this
    // tile sum to +0 at every position, and a mean of +0 changes no value: the
    // stored rows are exactly what `encode_row` writes.
    let mut rng = Rng(0x7119);
    let x = realistic_row(&mut rng, 128);
    let negated: Vec<f32> = x.iter().map(|v| -v).collect();
    for format in mean_formats() {
        let (mean, bytes) = encode_tile_of(format, 128, &[x.clone(), negated.clone()].concat());
        assert!(mean.iter().all(|&b| b == 0), "{format}: the mean is +0");
        assert_eq!(bytes, [encode(format, &x), encode(format, &negated)].concat(), "{format}");
    }
}

#[test]
fn decoding_a_tile_adds_the_stored_mean_to_every_row_and_then_un_rotates() {
    let mut rng = Rng(0x711a);
    let n = 96;
    let x = offset_rows(&mut rng, 7, n).concat();
    for format in mean_formats() {
        let (mean, bytes) = encode_tile_of(format, n, &x);
        let mean_f32 = mean_values(&mean);
        assert!(mean_f32.iter().any(|m| *m != 0.0));

        // code * scale, which is what `decode_row_rotated` gives for a row of a
        // tile, plus the mean: one f32 addition per value.
        let mut expected = Vec::new();
        for row in bytes.chunks(format.row_bytes(n).unwrap()) {
            expected.extend(decode_rotated(format, row).iter().zip(&mean_f32).map(|(v, m)| v + m));
        }
        let stored = decode_tile_rotated_of(format, n, &mean, &bytes);
        assert_eq!(bits32(&stored), bits32(&expected), "{format}");
        assert_eq!(
            bits32(&decode_tile_of(format, n, &mean, &bytes)),
            bits32(&rotated(&stored)),
            "{format}"
        );
    }
}

#[test]
fn the_mean_is_summed_in_the_order_of_the_rows_starting_from_plus_zero() {
    // A row with one value at index 0 rotates to that value times C at every
    // position, so the rotated rows are under control here: a, 1 and -a with
    // a = 2^30, where a + 1 is a in f32.
    let row = |v: f32| {
        let mut x = vec![0.0f32; GROUP];
        x[0] = v;
        x
    };
    let a = 2.0f32.powi(30);
    for format in mean_formats() {
        let (lost, _) = encode_tile_of(format, GROUP, &[row(a), row(1.0), row(-a)].concat());
        assert_eq!(mean_patterns(&lost), [0u16; GROUP], "{format}: (a + 1) - a is 0");
        let (kept, _) = encode_tile_of(format, GROUP, &[row(a), row(-a), row(1.0)].concat());
        assert_eq!(mean_patterns(&kept), [f32_to_f16(C / 3.0); GROUP], "{format}: (a - a) + 1");

        // The sum starts at +0: one row of -0 has the mean +0 at position 0,
        // where the rotated value is -0.
        let zeros = vec![-0.0f32; GROUP];
        assert_eq!(rotated(&zeros)[0].to_bits(), (-0.0f32).to_bits());
        let (mean, bytes) = encode_tile_of(format, GROUP, &zeros);
        assert_eq!(mean_patterns(&mean), [0u16; GROUP], "{format}");
        assert!(bytes.iter().all(|&b| b == 0), "{format}: scale zero, codes zero");
    }
}

#[test]
fn a_mean_shared_by_the_rows_of_a_tile_costs_no_codes() {
    // Rows that share an offset per value (three quarters of their energy here)
    // are stored much better a tile at a time, because the codes then carry the
    // part that differs from row to row only; rows that share nothing lose
    // nothing. Signal-to-noise ratio of 128 rows of 256 values, in dB.
    let mut rng = Rng(0x5eed_711e);
    let shared = offset_rows(&mut rng, TILE, 256);
    let unrelated = rows(&mut rng, TILE, 256, realistic_row);
    println!("SNR in dB, rows on their own / as a tile:");
    for format in mean_formats() {
        let alone = snr_db(&shared, &round_trip(format, &shared));
        let tile = snr_db(&shared, &tile_round_trip(format, &shared));
        let plain_alone = snr_db(&unrelated, &round_trip(format, &unrelated));
        let plain_tile = snr_db(&unrelated, &tile_round_trip(format, &unrelated));
        println!(
            "  {format}: shared offsets {alone:.1} / {tile:.1}, unrelated rows {plain_alone:.1} / {plain_tile:.1}"
        );
        assert!(tile > alone + 5.0, "{format}: {alone:.2} dB alone, {tile:.2} dB as a tile");
        assert!(plain_tile > plain_alone - 0.1, "{format}: {plain_alone:.2} / {plain_tile:.2}");
    }
}

#[test]
fn a_tile_of_one_row_stores_the_row_as_its_mean() {
    let mut rng = Rng(0x711b);
    let x = realistic_row(&mut rng, 64);
    let r = rotated(&x);
    for format in mean_formats() {
        let (mean, bytes) = encode_tile_of(format, 64, &x);
        let expected: Vec<u16> = r.iter().map(|v| f32_to_f16(*v)).collect();
        assert_eq!(mean_patterns(&mean), expected, "{format}");
        // The codes hold what the rounding to f16 left, so the row comes back
        // closer than f16 alone would bring it.
        let residual =
            r.iter().zip(mean_values(&mean)).map(|(v, m)| (v - m).abs()).fold(0.0, f32::max);
        let stored = decode_tile_rotated_of(format, 64, &mean, &bytes);
        let error = r.iter().zip(&stored).map(|(v, s)| (v - s).abs()).fold(0.0, f32::max);
        assert!(residual > 0.0 && error < residual, "{format}: {error} against {residual}");
    }
}

#[test]
fn the_mean_of_a_tile_enters_attention_once_per_tile() {
    // What a kernel does with tiles, against the reference decoder. K: the dot
    // product of the rotated query with the stored codes of each position, plus
    // one dot product with the mean per tile. V: the weighted sum of the stored
    // codes, plus the mean of each tile times the sum of the weights of its
    // positions, un-rotated at the end. Three tiles of 16 cells, every pair of
    // formats with a mean on at least one side.
    let mut rng = Rng(0x711c);
    let (n, tile_cells, tiles) = (128, 16, 3);
    let keys = offset_rows(&mut rng, tile_cells * tiles, n);
    let values = offset_rows(&mut rng, tile_cells * tiles, n);
    let query = realistic_row(&mut rng, n);
    let logit_scale = 3.0 / (dot(&query, &query).sqrt() * dot(&keys[0], &keys[0]).sqrt());
    let softmax = |logits: &[f64]| -> Vec<f64> {
        let top = logits.iter().cloned().fold(f64::MIN, f64::max);
        let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    };

    let mut checked = 0;
    for k_format in KvFormat::ALL {
        for v_format in KvFormat::ALL {
            if !k_format.has_mean() && !v_format.has_mean() {
                continue;
            }
            let k_tiles: Vec<(Vec<u8>, Vec<u8>)> = keys
                .chunks(tile_cells)
                .map(|tile| encode_tile_of(k_format, n, &tile.concat()))
                .collect();
            let v_tiles: Vec<(Vec<u8>, Vec<u8>)> = values
                .chunks(tile_cells)
                .map(|tile| encode_tile_of(v_format, n, &tile.concat()))
                .collect();

            // The reference: every row decoded to the original domain.
            let decoded = |format: KvFormat, tiles: &[(Vec<u8>, Vec<u8>)]| -> Vec<Vec<f32>> {
                tiles
                    .iter()
                    .flat_map(|(mean, bytes)| {
                        let rows = decode_tile_of(format, n, mean, bytes);
                        rows.chunks(n).map(<[f32]>::to_vec).collect::<Vec<_>>()
                    })
                    .collect()
            };
            let (k_rows, v_rows) = (decoded(k_format, &k_tiles), decoded(v_format, &v_tiles));
            let logits: Vec<f64> = k_rows.iter().map(|k| dot(&query, k) * logit_scale).collect();
            let weights = softmax(&logits);
            let mut reference = vec![0.0f64; n];
            for (w, v) in weights.iter().zip(&v_rows) {
                for (o, x) in reference.iter_mut().zip(v) {
                    *o += w * f64::from(*x);
                }
            }

            // The kernel's way.
            let q = if k_format.is_rotated() { rotated(&query) } else { query.clone() };
            let mut kernel_logits = Vec::new();
            for (mean, bytes) in &k_tiles {
                // Empty for a format without a mean, and then the term is 0.
                let mean_dot = dot(&q, &mean_values(mean));
                for row in bytes.chunks(k_format.row_bytes(n).unwrap()) {
                    let stored = decode_rotated(k_format, row);
                    kernel_logits.push((dot(&q, &stored) + mean_dot) * logit_scale);
                }
            }
            let kernel_weights = softmax(&kernel_logits);
            let mut out = vec![0.0f64; n];
            for ((mean, bytes), weights) in v_tiles.iter().zip(kernel_weights.chunks(tile_cells)) {
                for (w, row) in weights.iter().zip(bytes.chunks(v_format.row_bytes(n).unwrap())) {
                    for (o, x) in out.iter_mut().zip(decode_rotated(v_format, row)) {
                        *o += w * f64::from(x);
                    }
                }
                let weight_sum: f64 = weights.iter().sum();
                for (o, m) in out.iter_mut().zip(mean_values(mean)) {
                    *o += weight_sum * f64::from(m);
                }
            }
            if v_format.is_rotated() {
                let narrow: Vec<f32> = out.iter().map(|x| *x as f32).collect();
                out = rotated(&narrow).iter().map(|x| f64::from(*x)).collect();
            }

            let size = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
            let difference =
                out.iter().zip(&reference).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            assert!(
                difference < 1e-5 * size,
                "K {k_format}, V {v_format}: the two ways differ by {:.2e} of the output",
                difference / size
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 36 - 4);
}

#[test]
fn a_mean_beyond_the_range_of_f16_saturates_and_the_tile_still_encodes() {
    // Three equal rows of 4e5: the rotation puts 32 * 4e5 * C, about 2.26e6, at
    // position 0 and exactly 0 everywhere else. The mean saturates at 65504 and
    // the rows are stored relative to that.
    let x = vec![4.0e5f32; 3 * GROUP];
    let r0 = rotated(&x[..GROUP])[0];
    assert!(r0 > 2.2e6 && rotated(&x[..GROUP])[1..].iter().all(|v| *v == 0.0));
    for format in mean_formats() {
        let (mean, bytes) = encode_tile_of(format, GROUP, &x);
        let patterns = mean_patterns(&mean);
        assert_eq!(patterns[0], 0x7bff, "{format}");
        assert!(patterns[1..].iter().all(|&m| m == 0), "{format}");
        // What is left, 2.2e6, over qmax is within f16 for q7 (34 877), which
        // brings the value back within a step; q6 (70 879) to q4 saturate their
        // scale as a row on its own would.
        let stored = decode_tile_rotated_of(format, GROUP, &mean, &bytes);
        let scale = stored_scale(format, &bytes, 0).abs();
        if bits_of(format) == 7 {
            assert!((stored[0] - r0).abs() <= scale, "{format}: {} for {r0}", stored[0]);
        } else {
            assert_eq!(scale, 65504.0, "{format}");
        }
    }

    // Seven rows whose rotated position 0 is 5.66e37 each: their sum overflows
    // f32, which the mean takes as the largest f16 too.
    let x = vec![1.0e37f32; 7 * GROUP];
    assert!(rotated(&x[..GROUP])[0] * 7.0 == f32::INFINITY);
    for format in mean_formats() {
        let (mean, bytes) = encode_tile_of(format, GROUP, &x);
        assert_eq!(mean_patterns(&mean)[0], 0x7bff, "{format}");
        assert_eq!(stored_scale(format, &bytes, 0).abs(), 65504.0, "{format}");
    }
}

#[test]
fn tile_errors_name_the_problem_and_nothing_is_written() {
    let untouched = |mean: &[u8], out: &[u8]| mean.iter().chain(out).all(|&b| b == 0xa5);
    for format in KvFormat::ALL {
        let row_bytes = format.row_bytes(64).unwrap();
        let mean_bytes = format.mean_bytes(64).unwrap();
        let x = vec![1.0f32; 3 * 64];
        let attempt = |n: usize, values: &[f32], mean_len: usize, out_len: usize| {
            let (mut mean, mut out) = (vec![0xa5; mean_len], vec![0xa5; out_len]);
            let result = encode_tile(format, n, values, &mut mean, &mut out);
            assert!(untouched(&mean, &out), "{format}: wrote on {result:?}");
            result
        };

        // Rows: none, a part of one, rows of no values, rows of partial groups.
        assert_eq!(attempt(64, &[], mean_bytes, 0), Err(Error::KvTileRows { values: 0, row: 64 }));
        assert_eq!(
            attempt(64, &x[..96], mean_bytes, row_bytes),
            Err(Error::KvTileRows { values: 96, row: 64 })
        );
        assert_eq!(attempt(0, &x, 0, 0), Err(Error::KvTileRows { values: 192, row: 0 }));
        assert_eq!(
            attempt(48, &x, mean_bytes, 3 * row_bytes),
            Err(Error::KvPartialGroup { values: 48 })
        );

        // The mean: exactly as long as the format has it.
        for len in [mean_bytes + 1, mean_bytes + 128, mean_bytes.saturating_sub(1)] {
            if len == mean_bytes {
                continue;
            }
            assert_eq!(
                attempt(64, &x, len, 3 * row_bytes),
                Err(Error::KvMeanLength { format, values: 64, expected: mean_bytes, actual: len })
            );
        }

        // The rows: exactly the bytes of all of them.
        for len in [0, row_bytes, 3 * row_bytes - 1, 3 * row_bytes + 1] {
            assert_eq!(
                attempt(64, &x, mean_bytes, len),
                Err(Error::KvByteLength {
                    format,
                    values: 192,
                    expected: 3 * row_bytes,
                    actual: len
                })
            );
        }

        // A value that is not finite is named by its position in the tile, a
        // rotation that overflows by its group in the tile; an error in the last
        // row leaves the first rows unwritten too.
        let mut bad = x.clone();
        bad[2 * 64 + 5] = f32::NAN;
        assert_eq!(
            attempt(64, &bad, mean_bytes, 3 * row_bytes),
            Err(Error::KvNonFinite { position: 133, kind: "NaN" })
        );
        bad[64 + 7] = f32::NEG_INFINITY;
        assert_eq!(
            attempt(64, &bad, mean_bytes, 3 * row_bytes),
            Err(Error::KvNonFinite { position: 71, kind: "-infinity" })
        );
        if format.is_rotated() {
            let mut overflow = x.clone();
            overflow[2 * 64 + 32..].fill(f32::MAX);
            assert_eq!(
                attempt(64, &overflow, mean_bytes, 3 * row_bytes),
                Err(Error::KvRotationOverflow { group: 5 })
            );
        }

        // Decoding checks the same lengths and writes nothing either.
        let (mean, bytes) = encode_tile_of(format, 64, &x);
        for decoder in [decode_tile_rotated, decode_tile] {
            let mut y = vec![7.0f32; 3 * 64];
            let wrong_mean = vec![0u8; mean_bytes + 2];
            assert_eq!(
                decoder(format, 64, &wrong_mean, &bytes, &mut y),
                Err(Error::KvMeanLength {
                    format,
                    values: 64,
                    expected: mean_bytes,
                    actual: mean_bytes + 2
                })
            );
            assert_eq!(
                decoder(format, 64, &mean, &bytes[1..], &mut y),
                Err(Error::KvByteLength {
                    format,
                    values: 192,
                    expected: 3 * row_bytes,
                    actual: 3 * row_bytes - 1
                })
            );
            assert_eq!(
                decoder(format, 64, &mean, &bytes, &mut y[..100]),
                Err(Error::KvTileRows { values: 100, row: 64 })
            );
            assert_eq!(
                decoder(format, 0, &mean, &bytes, &mut y),
                Err(Error::KvTileRows { values: 192, row: 0 })
            );
            assert!(y.iter().all(|&v| v == 7.0), "{format}: decode wrote");
            decoder(format, 64, &mean, &bytes, &mut y).unwrap();
            assert!(y.iter().all(|&v| v != 7.0), "{format}");
        }
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn a_group_of_zeros_has_scale_zero_and_zero_codes() {
    for format in quantized_formats() {
        let bytes = encode(format, &[0.0; 64]);
        assert!(bytes.iter().all(|&b| b == 0), "{format}: {bytes:02x?}");
        assert!(decode(format, &bytes).iter().all(|v| v.to_bits() == 0));
        // The sign of a zero is not stored.
        assert_eq!(encode(format, &[-0.0; 32]), vec![0; format.group_bytes()], "{format}");
    }
    // f16 keeps it.
    let bytes = encode(KvFormat::F16, &[-0.0; 32]);
    assert!(bytes.chunks(2).all(|h| h == [0x00, 0x80]));
    assert!(decode(KvFormat::F16, &bytes).iter().all(|v| v.to_bits() == 0x8000_0000));
}

#[test]
fn a_group_far_below_the_smallest_scale_is_stored_as_zeros_with_the_smallest_scale() {
    // 1e-9 is below half of 2^-24 * qmax for every width, so the scale is the
    // smallest f16 subnormal and every code is 0.
    let x = [1.0e-9f32; 32];
    for format in quantized_formats() {
        let bytes = encode(format, &x);
        assert_eq!(&bytes[..2], &[0x01, 0x00], "{format}: the smallest positive f16");
        assert_eq!(decode_rotated(format, &bytes), vec![0.0; 32], "{format}");
    }
    // The smallest f32 subnormal still gets a usable scale.
    let tiny = [f32::from_bits(1); 32];
    for format in quantized_formats() {
        assert_eq!(&encode(format, &tiny)[..2], &[0x01, 0x00], "{format}");
    }
}

#[test]
fn a_group_with_the_largest_f32_saturates_the_scale_instead_of_failing() {
    // One f32::MAX: its rotation is +-MAX*C in all 32 places (6.0e37), so the
    // scale would be 6.0e37/qmax, far above the largest finite f16. It saturates
    // at 65504 (0x7bff), every code clamps to its end, and the decoded values are
    // finite but nowhere near the input: the format cannot hold such a value.
    // The end of the codes is +-qmax for q8. For q7 to q4 it is the longer end,
    // -qmax-1, on the side of the extreme: a positive extreme is stored with the
    // negative scale -65504 (0xfbff) and decodes to (qmax+1) * 65504, a negative
    // one with +65504 to -(qmax+1) * 65504.
    let mut x = [0.0f32; 32];
    x[0] = f32::MAX;
    for format in quantized_formats() {
        let max = qmax(bits_of(format));
        let searched = bits_of(format) < 8;
        let (positive_scale, peak) = if searched {
            ([0xff, 0xfb], (max + 1.0) * 65504.0)
        } else {
            ([0xff, 0x7b], max * 65504.0)
        };
        let bytes = encode(format, &x);
        assert_eq!(&bytes[..2], &positive_scale, "{format}");
        let r = decode_rotated(format, &bytes);
        assert!(r.iter().all(|&v| v == peak), "{format}: {r:?}");
        let original = decode(format, &bytes);
        assert!(original.iter().all(|v| v.is_finite()), "{format}");
        assert!(original[0] > 0.0 && original[0] < f32::MAX);
        // And the opposite sign: the scale is positive in both kinds of format.
        x[0] = -f32::MAX;
        let bytes = encode(format, &x);
        assert_eq!(&bytes[..2], &[0xff, 0x7b], "{format}");
        let r = decode_rotated(format, &bytes);
        assert!(r.iter().all(|&v| v == -peak), "{format}");
        x[0] = f32::MAX;
    }
    // Just below saturation nothing is clamped: the scale is a finite f16 that
    // the 32 equal rotated values divide into qmax (or, with the longer end of
    // the codes, qmax + 1) steps, and the decoded peak is within the rounding of
    // that scale of the input.
    for format in quantized_formats() {
        let max = qmax(bits_of(format));
        let mut x = [0.0f32; 32];
        x[0] = 65_000.0 * max / C;
        let peak = 65_000.0 * max;
        let bytes = encode(format, &x);
        let s = stored_scale(format, &bytes, 0);
        assert!(s.abs() <= 65_504.0 && s.abs() >= peak / (max + 1.0) * 0.999, "{format}: {s}");
        let decoded = decode_rotated(format, &bytes)[0];
        assert!((decoded - peak).abs() <= 0.002 * peak, "{format}: peak {decoded} for {peak}");
    }
}

#[test]
fn f16_saturates_values_beyond_its_range_to_the_largest_finite_half() {
    let mut x = [1.0f32; 32];
    x[..8].copy_from_slice(&[
        65_504.0,
        65_519.0,
        65_520.0,
        1.0e9,
        f32::MAX,
        -65_520.0,
        -f32::MAX,
        -65_504.0,
    ]);
    let bytes = encode(KvFormat::F16, &x);
    let halves: Vec<u16> = bytes.chunks(2).map(|h| u16::from_le_bytes([h[0], h[1]])).collect();
    assert_eq!(&halves[..8], &[0x7bff, 0x7bff, 0x7bff, 0x7bff, 0x7bff, 0xfbff, 0xfbff, 0xfbff]);
    assert!(decode(KvFormat::F16, &bytes).iter().all(|v| v.is_finite()));
}

#[test]
fn f16_rows_are_the_halves_of_the_values_rounded_to_nearest_even() {
    let mut rng = Rng(71);
    let x = realistic_row(&mut rng, 256);
    let bytes = encode(KvFormat::F16, &x);
    assert_eq!(bytes.len(), 512);
    for (i, v) in x.iter().enumerate() {
        assert_eq!(&bytes[2 * i..2 * i + 2], &f32_to_f16(*v).to_le_bytes(), "value {i}");
    }
    let y = decode(KvFormat::F16, &bytes);
    for (a, b) in x.iter().zip(&y) {
        assert!((a - b).abs() <= a.abs() * 2f32.powi(-11), "{a} vs {b}");
    }
}

#[test]
fn a_rotation_that_overflows_f32_is_an_error_naming_the_group() {
    // Two f32::MAX in one group: the first stage already sums to infinity.
    let mut x = vec![1.0f32; 96];
    x[64] = f32::MAX;
    x[65] = f32::MAX;
    for format in quantized_formats() {
        let mut out = vec![0xa5; format.row_bytes(96).unwrap()];
        assert_eq!(encode_row(format, &x, &mut out), Err(Error::KvRotationOverflow { group: 2 }));
        assert!(out.iter().all(|&b| b == 0xa5), "{format}: a failed call must not write");
    }
    // f16 has no rotation: the same values saturate there.
    let mut out = vec![0; 192];
    assert_eq!(encode_row(KvFormat::F16, &x, &mut out), Ok(()));
}

#[test]
fn nan_and_infinity_are_errors_naming_the_first_position() {
    for format in KvFormat::ALL {
        for (value, kind) in
            [(f32::NAN, "NaN"), (f32::INFINITY, "+infinity"), (f32::NEG_INFINITY, "-infinity")]
        {
            for position in [0, 31, 32, 70, 95] {
                let mut x = vec![0.5f32; 96];
                x[position] = value;
                let mut out = vec![0xa5; format.row_bytes(96).unwrap()];
                let err = encode_row(format, &x, &mut out).unwrap_err();
                assert_eq!(err, Error::KvNonFinite { position, kind }, "{format}");
                let message = err.to_string();
                assert!(
                    message.contains(&position.to_string()) && message.contains(kind),
                    "{message}"
                );
                assert!(out.iter().all(|&b| b == 0xa5), "{format}: a failed call must not write");
            }
        }
        // Several: the lowest position is the one reported.
        let mut x = vec![0.5f32; 96];
        x[70] = f32::NAN;
        x[40] = f32::NEG_INFINITY;
        x[90] = f32::INFINITY;
        let mut out = vec![0; format.row_bytes(96).unwrap()];
        assert_eq!(
            encode_row(format, &x, &mut out),
            Err(Error::KvNonFinite { position: 40, kind: "-infinity" })
        );
    }
}

#[test]
fn wrong_lengths_are_errors_and_nothing_is_written() {
    for format in KvFormat::ALL {
        let group = format.group_bytes();
        let x = vec![1.0f32; 64];

        // Encode: a byte slice that is too short, too long, or empty.
        for len in [0, 2 * group - 1, 2 * group + 1, 3 * group] {
            let mut out = vec![0xa5; len];
            assert_eq!(
                encode_row(format, &x, &mut out),
                Err(Error::KvByteLength { format, values: 64, expected: 2 * group, actual: len })
            );
            assert!(out.iter().all(|&b| b == 0xa5), "{format}: encode wrote");
        }
        // Encode: values that are not whole groups.
        let mut out = vec![0xa5; 2 * group];
        assert_eq!(
            encode_row(format, &x[..63], &mut out),
            Err(Error::KvPartialGroup { values: 63 })
        );
        assert!(out.iter().all(|&b| b == 0xa5), "{format}: encode wrote");

        // Decode, both kinds.
        let bytes = encode(format, &x);
        for decoder in [decode_row_rotated, decode_row] {
            for len in [0, bytes.len() - 1, bytes.len() + 1] {
                let wrong: Vec<u8> = (0..len).map(|i| bytes.get(i).copied().unwrap_or(0)).collect();
                let mut y = vec![7.0f32; 64];
                assert_eq!(
                    decoder(format, &wrong, &mut y),
                    Err(Error::KvByteLength {
                        format,
                        values: 64,
                        expected: 2 * group,
                        actual: len
                    })
                );
                assert!(y.iter().all(|&v| v == 7.0), "{format}: decode wrote");
            }
            let mut y = vec![7.0f32; 63];
            assert_eq!(decoder(format, &bytes, &mut y), Err(Error::KvPartialGroup { values: 63 }));
            assert!(y.iter().all(|&v| v == 7.0), "{format}: decode wrote");
        }
        let message = encode_row(format, &x, &mut [0; 1]).unwrap_err().to_string();
        assert!(
            message.contains(format.name()) && message.contains(&(2 * group).to_string()),
            "{message}"
        );
    }
    // An empty row is a whole number of groups.
    for format in KvFormat::ALL {
        assert_eq!(encode_row(format, &[], &mut []), Ok(()));
        assert_eq!(decode_row(format, &[], &mut []), Ok(()));
    }
}

// ---------------------------------------------------------------------------
// Golden fixture
// ---------------------------------------------------------------------------

/// The five groups of the first fixture input, chosen to hold the cases in which
/// two implementations are most likely to differ:
///
/// 0. a row like a key, with one outlier;
/// 1. one value of 100 among small ones;
/// 2. zeros, some of them `-0.0`;
/// 3. values near 1e-9, which need the smallest scale;
/// 4. values near 1000 with a 1e9 among them, whose scale saturates for every
///    width (and which saturates in `f16`).
///
/// This input and its files for `f16` and `q8` are the ones of the first version
/// of the formats and never change (see
/// `the_fixtures_of_f16_and_q8_are_the_bytes_of_the_first_version`).
fn golden_input() -> Vec<f32> {
    let mut rng = Rng(0x2026_1109);
    let mut x = Vec::with_capacity(5 * GROUP);

    let mut group: Vec<f32> = (0..GROUP).map(|_| rng.gaussian()).collect();
    group[9] = 12.5;
    x.extend(group);

    let mut group: Vec<f32> = (0..GROUP).map(|_| 0.1 * rng.gaussian()).collect();
    group[17] = -100.0;
    x.extend(group);

    let mut group = vec![0.0f32; GROUP];
    group[3] = -0.0;
    group[20] = -0.0;
    x.extend(group);

    x.extend((0..GROUP).map(|_| 1.0e-9 * rng.gaussian()));

    let mut group: Vec<f32> = (0..GROUP).map(|_| 1000.0 * rng.gaussian()).collect();
    group[0] = 1.0e9;
    x.extend(group);
    x
}

/// The eleven groups of the second fixture input, which holds the cases of the
/// search of the scale in `q7` to `q4`:
///
/// 0. a rotated group whose largest value is positive (+4), which gets a
///    negative scale;
/// 1. one whose largest value is negative (-4), which gets a positive scale;
/// 2. one large value among small ones (100 among 0.05), whose rotation is a
///    cluster of equal magnitudes that only the fitted scale resolves;
/// 3. a group with a mean of +2 and a spread of 0.8, whose largest rotated
///    value is the mean and the search clips it;
/// 4. a group on the grid of every width (rotated values that are multiples of
///    0.25 in -7 ..= 7), where several candidates have no error;
/// 5. a heavy-tailed group;
/// 6. a single value of 1e-6 among zeros, whose scale is the smallest f16;
/// 7. a single -1e9 among values near 1000, which saturates the scale with the
///    extreme on the negative side;
/// 8. values near 6e5, whose scale saturates at q5 and q4 but not at q7 and q6;
/// 9. values near 1e-7, around the smallest f16 scale;
/// 10. a group with two rotated extremes of 3 and -3, which tie up to rounding.
///
/// Groups built from rotated values are the rotation of them, which is its own
/// inverse up to rounding.
fn golden_search_input() -> Vec<f32> {
    let mut rng = Rng(0x2026_1110);
    let mut x = Vec::with_capacity(11 * GROUP);

    let from_rotated = |r: &[f32]| rotated(r);

    let mut r: Vec<f32> = (0..GROUP).map(|_| rng.gaussian()).collect();
    r[5] = 4.0;
    x.extend(from_rotated(&r));

    let mut r: Vec<f32> = (0..GROUP).map(|_| rng.gaussian()).collect();
    r[11] = -4.0;
    x.extend(from_rotated(&r));

    let mut group: Vec<f32> = (0..GROUP).map(|_| 0.05 * rng.gaussian()).collect();
    group[8] = 100.0;
    x.extend(group);

    x.extend((0..GROUP).map(|_| 2.0 + 0.8 * rng.gaussian()));

    let r: Vec<f32> = (0..GROUP).map(|i| ((i as i32 * 5) % 15 - 7) as f32 * 0.25).collect();
    x.extend(from_rotated(&r));

    x.extend(heavy_tailed_row(&mut rng, GROUP));

    let mut group = vec![0.0f32; GROUP];
    group[3] = 1.0e-6;
    x.extend(group);

    let mut group: Vec<f32> = (0..GROUP).map(|_| 1000.0 * rng.gaussian()).collect();
    group[0] = -1.0e9;
    x.extend(group);

    x.extend((0..GROUP).map(|_| 6.0e5 * rng.gaussian()));

    x.extend((0..GROUP).map(|_| 1.0e-7 * rng.gaussian()));

    let mut r: Vec<f32> = (0..GROUP).map(|_| 0.3 * rng.gaussian()).collect();
    r[2] = 3.0;
    r[7] = -3.0;
    x.extend(from_rotated(&r));
    x
}

/// Rows and values per row of the tile of the fixture.
const GOLDEN_TILE_ROWS: usize = 5;
const GOLDEN_TILE_VALUES: usize = 3 * GROUP;

/// The tile of the fixture: five rows (a number that is not a power of two, so
/// that the division of the mean rounds) of three groups, the same kind of
/// group at the same place in every row:
///
/// 0. values that share an offset per position and differ by a third of it
///    from row to row, which is what the mean of a tile is for;
/// 1. zeros, two of them `-0.0` in the third row: the mean is `+0`, the scale
///    zero;
/// 2. values near 4e5, whose mean at the first rotated position (2.26e6) is
///    beyond f16 and saturates, and whose scale then saturates for `q6` to
///    `q4` but not for `q7`.
fn golden_tile_input() -> Vec<f32> {
    let mut rng = Rng(0x2026_1111);
    let offsets: Vec<f32> = (0..GROUP).map(|_| 1.5 * rng.gaussian()).collect();
    let mut x = Vec::with_capacity(GOLDEN_TILE_ROWS * GOLDEN_TILE_VALUES);
    for row in 0..GOLDEN_TILE_ROWS {
        x.extend(offsets.iter().map(|offset| offset + 0.5 * rng.gaussian()));

        let mut group = vec![0.0f32; GROUP];
        if row == 2 {
            group[3] = -0.0;
            group[20] = -0.0;
        }
        x.extend(group);

        x.extend((0..GROUP).map(|_| 4.0e5 + 1000.0 * rng.gaussian()));
    }
    x
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/kv")
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn le_floats(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// FNV-1a, 64 bits: the checksum with which the frozen files are pinned.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
}

/// Version of the container of the `.bin` files, in their header.
const GOLDEN_VERSION: u32 = 1;
const GOLDEN_HEADER: usize = 32;

/// One input of the fixture and the part of the names of its files after the
/// name of the format.
struct GoldenSet {
    suffix: &'static str,
    input: Vec<f32>,
}

fn golden_sets() -> [GoldenSet; 2] {
    [
        GoldenSet { suffix: "", input: golden_input() },
        GoldenSet { suffix: "_search", input: golden_search_input() },
    ]
}

/// All files of the fixture: name and contents. For each of the two inputs,
/// with `S` the suffix of the set (empty for the first, `_search` for the
/// second):
///
/// - `input<S>.f32` is the input, little-endian `f32`: 160 values for the
///   first set, 352 for the second.
/// - `rotated<S>.f32` is `rotate_groups(input)`, in the same layout.
/// - `<format><S>.bin` has a 32-byte header of little-endian `u32` words (the
///   magic `EQKV`, the version, the width of the codes or 16 for `f16`, the
///   number of values, the bytes per group, the length of the encoded part and
///   two zero words), then the encoded row, then `decode_row_rotated` of it as
///   little-endian `f32`, then `decode_row` of it in the same layout.
///
/// And for the tile of `golden_tile_input` (5 rows of 96 values):
///
/// - `input_tile.f32` and `rotated_tile.f32`, as above, row after row.
/// - `<format>_tile.bin`, for `q7` to `q4`, has a 32-byte header of little-endian
///   `u32` words (the magic `EQKT`, the version, the width of the codes, the
///   values of a row, the number of rows, the bytes per group, the length of the
///   mean and the length of the encoded rows), then the mean, then the encoded
///   rows, then `decode_tile_rotated` of them as little-endian `f32`, then
///   `decode_tile` in the same layout.
fn golden_files() -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    for GoldenSet { suffix, input } in golden_sets() {
        files.push((format!("input{suffix}.f32"), le_bytes(&input)));
        files.push((format!("rotated{suffix}.f32"), le_bytes(&rotated(&input))));
        for format in KvFormat::ALL {
            let encoded = encode(format, &input);
            let mut file = b"EQKV".to_vec();
            for word in [
                GOLDEN_VERSION,
                format.quant_bits().unwrap_or(16),
                input.len() as u32,
                format.group_bytes() as u32,
                encoded.len() as u32,
                0,
                0,
            ] {
                file.extend(word.to_le_bytes());
            }
            file.extend(&encoded);
            file.extend(le_bytes(&decode_rotated(format, &encoded)));
            file.extend(le_bytes(&decode(format, &encoded)));
            files.push((format!("{}{suffix}.bin", format.name()), file));
        }
    }

    let input = golden_tile_input();
    files.push(("input_tile.f32".to_owned(), le_bytes(&input)));
    files.push(("rotated_tile.f32".to_owned(), le_bytes(&rotated(&input))));
    for format in mean_formats() {
        let (mean, encoded) = encode_tile_of(format, GOLDEN_TILE_VALUES, &input);
        let mut file = b"EQKT".to_vec();
        for word in [
            GOLDEN_VERSION,
            bits_of(format),
            GOLDEN_TILE_VALUES as u32,
            GOLDEN_TILE_ROWS as u32,
            format.group_bytes() as u32,
            mean.len() as u32,
            encoded.len() as u32,
        ] {
            file.extend(word.to_le_bytes());
        }
        file.extend(&mean);
        file.extend(&encoded);
        file.extend(le_bytes(&decode_tile_rotated_of(format, GOLDEN_TILE_VALUES, &mean, &encoded)));
        file.extend(le_bytes(&decode_tile_of(format, GOLDEN_TILE_VALUES, &mean, &encoded)));
        files.push((format!("{}_tile.bin", format.name()), file));
    }
    files
}

fn blessing() -> bool {
    std::env::var_os("ARES_BLESS_KV_GOLDEN").is_some()
}

#[test]
fn the_golden_fixture_is_what_the_code_regenerates() {
    let files = golden_files();
    if blessing() {
        fs::create_dir_all(golden_dir()).unwrap();
        for (name, bytes) in &files {
            fs::write(golden_dir().join(name), bytes).unwrap();
        }
        return;
    }
    for (name, expected) in &files {
        let path = golden_dir().join(name);
        let on_disk = fs::read(&path).unwrap_or_else(|e| {
            panic!("{}: {e}; rewrite the fixture with ARES_BLESS_KV_GOLDEN=1", path.display())
        });
        let first_difference = on_disk.iter().zip(expected).position(|(a, b)| a != b);
        assert!(
            on_disk == *expected,
            "{name} differs from what the code produces (lengths {} and {}, first different byte {first_difference:?}); \
             if the format changed on purpose, rewrite it with ARES_BLESS_KV_GOLDEN=1",
            on_disk.len(),
            expected.len(),
        );
    }
    let mut names: Vec<String> = fs::read_dir(golden_dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    let mut expected: Vec<String> = files.into_iter().map(|(name, _)| name).collect();
    expected.sort();
    assert_eq!(names, expected, "the fixture directory holds exactly the generated files");
}

#[test]
fn the_fixtures_of_f16_and_q8_are_the_bytes_of_the_first_version() {
    // f16 and q8 were not touched by the techniques of q7 to q4, so the files of
    // the first input for them are the ones that were written when the formats
    // were first specified, and so is that input. They are pinned by checksum
    // (FNV-1a) of the files as they were then, and compared with what the code
    // makes of the input too, so that neither a regenerated file nor a changed
    // encoder can move them without this test failing. (A deliberate change of
    // q8 or f16 would change these constants; that is the point of them.)
    let frozen: [(&str, u64); 4] = [
        ("f16.bin", 0xa82d_0a8d_1a72_94f8),
        ("q8.bin", 0x5a28_b9dc_31a7_1ace),
        ("input.f32", 0xda29_b586_a96f_64c2),
        ("rotated.f32", 0x13d1_3d1b_f541_02d8),
    ];
    let files = golden_files();
    for (name, checksum) in frozen {
        let generated = &files.iter().find(|(n, _)| n == name).expect(name).1;
        assert_eq!(fnv1a(generated), checksum, "{name}: what the code generates");
        if !blessing() {
            let on_disk = fs::read(golden_dir().join(name)).unwrap();
            assert_eq!(fnv1a(&on_disk), checksum, "{name}: the file on disk");
        }
    }
}

#[test]
fn the_golden_fixture_holds_through_the_public_api_alone() {
    // What a kernel test does with the files, using only what they contain:
    // the encoder reproduces the stored bytes from the input, both decoders the
    // stored floats from the bytes, bit for bit, and the rotation the stored
    // rotated input.
    if blessing() {
        return;
    }
    let read = |name: &str| fs::read(golden_dir().join(name)).unwrap();
    for GoldenSet { suffix, input: expected_input } in golden_sets() {
        let input = le_floats(&read(&format!("input{suffix}.f32")));
        assert_eq!(input.len(), expected_input.len());
        assert_eq!(
            bits32(&rotated(&input)),
            bits32(&le_floats(&read(&format!("rotated{suffix}.f32"))))
        );

        for format in KvFormat::ALL {
            let file = read(&format!("{}{suffix}.bin", format.name()));
            let word =
                |i: usize| u32::from_le_bytes(file[4 + 4 * i..8 + 4 * i].try_into().unwrap());
            assert_eq!(&file[..4], b"EQKV");
            assert_eq!(word(0), GOLDEN_VERSION);
            assert_eq!(word(1), format.quant_bits().unwrap_or(16));
            assert_eq!(word(2) as usize, input.len());
            assert_eq!(word(3) as usize, format.group_bytes());
            assert_eq!(word(4) as usize, format.row_bytes(input.len()).unwrap());
            assert_eq!((word(5), word(6)), (0, 0));

            let encoded_len = word(4) as usize;
            let n = input.len() * 4;
            assert_eq!(file.len(), GOLDEN_HEADER + encoded_len + 2 * n, "{format}: size");
            let encoded = &file[GOLDEN_HEADER..GOLDEN_HEADER + encoded_len];
            let stored_rotated = le_floats(&file[GOLDEN_HEADER + encoded_len..][..n]);
            let stored_decoded = le_floats(&file[GOLDEN_HEADER + encoded_len + n..]);

            assert_eq!(encode(format, &input), encoded, "{format}{suffix}: encoding");
            assert_eq!(
                bits32(&decode_rotated(format, encoded)),
                bits32(&stored_rotated),
                "{format}{suffix}"
            );
            assert_eq!(
                bits32(&decode(format, encoded)),
                bits32(&stored_decoded),
                "{format}{suffix}"
            );
        }
    }
}

#[test]
fn the_golden_inputs_exercise_the_cases_they_were_chosen_for() {
    // If the generators drifted, the fixture would still match itself and quietly
    // stop covering the edge cases; this pins what each group does.
    let scale = |format: KvFormat, bytes: &[u8], g: usize| {
        u16::from_le_bytes([bytes[g * format.group_bytes()], bytes[g * format.group_bytes() + 1]])
    };

    let input = golden_input();
    for format in quantized_formats() {
        let bytes = encode(format, &input);
        let scale = |g| scale(format, &bytes, g);
        assert_eq!(scale(2), 0, "{format}: the zero group");
        assert_eq!(scale(3), 1, "{format}: the tiny group has the smallest scale, positive");
        assert_eq!(scale(4) & 0x7fff, 0x7bff, "{format}: the large group saturates");
        let magnitude = |g| scale(g) & 0x7fff;
        assert!(magnitude(0) > 1 && magnitude(0) < 0x7bff);
        assert!(magnitude(1) > 1 && magnitude(1) < 0x7bff);
    }
    let halves = encode(KvFormat::F16, &input);
    let at = |i: usize| u16::from_le_bytes([halves[2 * i], halves[2 * i + 1]]);
    assert_eq!(at(2 * GROUP + 3), 0x8000, "-0.0 is kept by f16");
    assert_eq!(at(4 * GROUP), 0x7bff, "1e9 saturates in f16");

    let input = golden_search_input();
    assert_eq!(input.len(), 11 * GROUP);
    for format in quantized_formats() {
        let bytes = encode(format, &input);
        let scale = |g| scale(format, &bytes, g);
        let negative = |g| scale(g) & 0x8000 != 0;
        let magnitude = |g| scale(g) & 0x7fff;
        let searched = bits_of(format) < 8;
        if searched {
            assert!(negative(0), "{format}: a positive extreme gives a negative scale");
            assert!(!negative(1), "{format}: a negative extreme gives a positive scale");
            assert!(negative(3), "{format}: the positive mean is the largest value");
            assert!(!negative(7), "{format}: the -1e9 is the extreme");
            assert_eq!(magnitude(7), 0x7bff, "{format}: the large group saturates");
            assert_eq!(magnitude(6), 1, "{format}: the single 1e-6 has the smallest scale");
            assert!(magnitude(9) < 0x10, "{format}: values near 1e-7 are in the f16 floor");
        } else {
            assert!((0..11).all(|g| !negative(g)), "q8 never stores a negative scale");
        }
        // Whatever the width, every group here has a scale.
        assert!((0..11).all(|g| scale(g) != 0), "{format}");
        // The saturation of 6e5 depends on the width: its rotated peak over qmax.
        let saturated = magnitude(8) == 0x7bff;
        assert_eq!(saturated, bits_of(format) <= 5, "{format}: values near 6e5");
    }
}

#[test]
fn the_tile_fixture_holds_through_the_public_api_alone() {
    // The same as for the rows: from the files alone, the encoder reproduces the
    // stored mean and rows from the input, and both decoders the stored floats.
    if blessing() {
        return;
    }
    let read = |name: &str| fs::read(golden_dir().join(name)).unwrap();
    let input = le_floats(&read("input_tile.f32"));
    assert_eq!(input.len(), GOLDEN_TILE_ROWS * GOLDEN_TILE_VALUES);
    assert_eq!(bits32(&rotated(&input)), bits32(&le_floats(&read("rotated_tile.f32"))));

    for format in mean_formats() {
        let file = read(&format!("{}_tile.bin", format.name()));
        let word = |i: usize| u32::from_le_bytes(file[4 + 4 * i..8 + 4 * i].try_into().unwrap());
        assert_eq!(&file[..4], b"EQKT");
        assert_eq!(word(0), GOLDEN_VERSION);
        assert_eq!(word(1), bits_of(format));
        let (n, rows) = (word(2) as usize, word(3) as usize);
        assert_eq!((n, rows), (GOLDEN_TILE_VALUES, GOLDEN_TILE_ROWS));
        assert_eq!(word(4) as usize, format.group_bytes());
        let (mean_len, encoded_len) = (word(5) as usize, word(6) as usize);
        assert_eq!(mean_len, format.mean_bytes(n).unwrap());
        assert_eq!(encoded_len, rows * format.row_bytes(n).unwrap());

        let floats = 4 * rows * n;
        assert_eq!(file.len(), GOLDEN_HEADER + mean_len + encoded_len + 2 * floats, "{format}");
        let mean = &file[GOLDEN_HEADER..][..mean_len];
        let encoded = &file[GOLDEN_HEADER + mean_len..][..encoded_len];
        let stored_rotated = le_floats(&file[GOLDEN_HEADER + mean_len + encoded_len..][..floats]);
        let stored_decoded = le_floats(&file[GOLDEN_HEADER + mean_len + encoded_len + floats..]);

        let (made_mean, made_rows) = encode_tile_of(format, n, &input);
        assert_eq!(made_mean, mean, "{format}: the mean");
        assert_eq!(made_rows, encoded, "{format}: the rows");
        assert_eq!(
            bits32(&decode_tile_rotated_of(format, n, mean, encoded)),
            bits32(&stored_rotated),
            "{format}"
        );
        assert_eq!(
            bits32(&decode_tile_of(format, n, mean, encoded)),
            bits32(&stored_decoded),
            "{format}"
        );
    }
}

#[test]
fn the_tile_of_the_fixture_exercises_the_cases_it_was_chosen_for() {
    let input = golden_tile_input();
    let n = GOLDEN_TILE_VALUES;
    assert_eq!(input.len(), GOLDEN_TILE_ROWS * n);
    assert_eq!(input[2 * n + GROUP + 3].to_bits(), (-0.0f32).to_bits());
    let row_wise = |format: KvFormat| -> Vec<u8> {
        input.chunks(n).flat_map(|row| encode(format, row)).collect()
    };
    for format in mean_formats() {
        let (mean, bytes) = encode_tile_of(format, n, &input);
        let patterns = mean_patterns(&mean);
        let scale = |row: usize, g: usize| {
            let at = row * format.row_bytes(n).unwrap() + g * format.group_bytes();
            u16::from_le_bytes([bytes[at], bytes[at + 1]])
        };

        // Group 0: a mean worth storing, and rows that differ from it.
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!(rms(&mean_values(&mean)[..GROUP]) > 1.0, "{format}");
        assert_ne!(bytes, row_wise(format), "{format}: the mean changes what is stored");
        // Group 1: zeros have the mean +0, and scale zero in every row.
        assert!(patterns[GROUP..2 * GROUP].iter().all(|&m| m == 0), "{format}");
        assert!((0..GOLDEN_TILE_ROWS).all(|row| scale(row, 1) == 0), "{format}");
        // Group 2: the mean saturates at the first rotated position only, and
        // what is left saturates the scale of every width but q7.
        assert_eq!(patterns[2 * GROUP], 0x7bff, "{format}");
        assert!(patterns[2 * GROUP + 1..].iter().all(|&m| m & 0x7fff < 0x7bff), "{format}");
        for row in 0..GOLDEN_TILE_ROWS {
            let saturated = scale(row, 2) & 0x7fff == 0x7bff;
            assert_eq!(saturated, bits_of(format) < 7, "{format}, row {row}");
        }
    }
}
