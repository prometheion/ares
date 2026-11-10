//! Compares the crate with ggml's own C implementation.
//!
//! The fixtures in `tests/golden/` are written by `tests/golden/generate.py`, which
//! calls `libggml-base.so` through ctypes; nothing in them comes from this crate.
//! Decoding must reproduce ggml's `f32` output bit for bit and encoding must
//! reproduce ggml's bytes exactly, so every comparison is on bit patterns (which
//! also makes NaN payloads and the sign of zero count).

use std::fs;
use std::path::PathBuf;

use ares_core::GgmlType;
use ares_core::ggml_type::ALL_TYPES;
use ares_quant::{Error, dequantize, quantize};

const HEADER_LEN: usize = 32;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn fixture_path(kind: &str, ty: GgmlType) -> PathBuf {
    golden_dir().join(format!("{kind}_{}.bin", ty.name()))
}

/// A fixture file: the header fields and the two payloads after it.
struct Fixture {
    n_blocks: usize,
    /// Dequantize fixtures: how many leading blocks came from ggml's quantizer
    /// (the rest are raw random bytes). Zero for quantize fixtures.
    n_quantized: usize,
    first: Vec<u8>,
    second: Vec<u8>,
}

impl Fixture {
    /// Reads `<kind>_<type>.bin`, or `None` when there is no such file. A file
    /// that exists but disagrees with itself or with `ares-core` panics: that
    /// means the fixture is stale, not that the crate is wrong.
    fn load(kind: &str, ty: GgmlType) -> Option<Self> {
        let path = fixture_path(kind, ty);
        let bytes = fs::read(&path).ok()?;
        let word = |i: usize| u32::from_le_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap());
        let magic: &[u8] = if kind == "dequant" { b"EQGD" } else { b"EQGQ" };
        assert_eq!(&bytes[..4], magic, "{}: wrong magic", path.display());
        assert_eq!(word(0), 1, "{}: unknown fixture version", path.display());
        assert_eq!(word(1), ty.id(), "{}: holds another type", path.display());
        assert_eq!(word(2), ty.block_values(), "{}: stale block size", path.display());
        assert_eq!(word(3), ty.block_bytes(), "{}: stale block bytes", path.display());
        let n_blocks = word(4) as usize;
        let values = n_blocks * ty.block_values() as usize;
        let blocks = n_blocks * ty.block_bytes() as usize;
        let (first_len, second_len) =
            if kind == "dequant" { (blocks, values * 4) } else { (values * 4, blocks) };
        assert_eq!(
            bytes.len(),
            HEADER_LEN + first_len + second_len,
            "{}: truncated",
            path.display()
        );
        let (first, second) = bytes[HEADER_LEN..].split_at(first_len);
        Some(Self {
            n_blocks,
            n_quantized: word(5) as usize,
            first: first.to_vec(),
            second: second.to_vec(),
        })
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Decodes every block of the fixture and returns a description of the first
/// value that differs from ggml's, if any.
fn first_dequantize_mismatch(ty: GgmlType, fx: &Fixture) -> Option<String> {
    let expected = f32s(&fx.second);
    let mut actual = vec![f32::from_bits(0xdead_beef); expected.len()];
    dequantize(ty, &fx.first, &mut actual)
        .unwrap_or_else(|e| panic!("{ty}: dequantize refused the fixture: {e}"));

    let (values, bytes) = (ty.block_values() as usize, ty.block_bytes() as usize);
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        if got.to_bits() != want.to_bits() {
            let block = i / values;
            let origin = if block < fx.n_quantized {
                "made by ggml's quantizer"
            } else {
                "raw random bytes"
            };
            return Some(format!(
                "{ty}: value {} of block {block} ({origin}): got {got:e} ({:#010x}), ggml gives {want:e} ({:#010x})\n  block bytes: {:02x?}",
                i % values,
                got.to_bits(),
                want.to_bits(),
                &fx.first[block * bytes..(block + 1) * bytes],
            ));
        }
    }
    None
}

#[test]
fn dequantize_matches_ggml_bit_for_bit() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for ty in ALL_TYPES {
        let Some(fx) = Fixture::load("dequant", ty) else {
            continue;
        };
        assert!(fx.n_blocks > 0, "{ty}: empty fixture");
        checked += 1;
        failures.extend(first_dequantize_mismatch(ty, &fx));
    }
    assert!(checked >= 29, "only {checked} dequantize fixtures found; is tests/golden/ intact?");
    assert!(
        failures.is_empty(),
        "{} type(s) differ from ggml:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Encodes the fixture's inputs and returns a description of the first block
/// whose bytes differ from ggml's, if any.
fn first_quantize_mismatch(ty: GgmlType, fx: &Fixture) -> Option<String> {
    let input = f32s(&fx.first);
    let mut actual = vec![0xa5u8; fx.second.len()];
    quantize(ty, &input, &mut actual)
        .unwrap_or_else(|e| panic!("{ty}: quantize refused the fixture: {e}"));

    let (values, bytes) = (ty.block_values() as usize, ty.block_bytes() as usize);
    for block in 0..fx.n_blocks {
        let got = &actual[block * bytes..(block + 1) * bytes];
        let want = &fx.second[block * bytes..(block + 1) * bytes];
        if got != want {
            return Some(format!(
                "{ty}: block {block} differs\n  input:        {:?}\n  got bytes:    {got:02x?}\n  ggml's bytes: {want:02x?}",
                &input[block * values..(block + 1) * values],
            ));
        }
    }
    None
}

#[test]
fn quantize_matches_ggml_byte_for_byte() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for ty in ALL_TYPES {
        let Some(fx) = Fixture::load("quantize", ty) else {
            continue;
        };
        assert!(fx.n_blocks > 0, "{ty}: empty fixture");
        checked += 1;
        failures.extend(first_quantize_mismatch(ty, &fx));
    }
    assert!(checked >= 7, "only {checked} quantize fixtures found; is tests/golden/ intact?");
    assert!(
        failures.is_empty(),
        "{} type(s) differ from ggml:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Whether a call with no data at all is accepted (`Ok`) or refused because the
/// type has no conversion (`Err(Unsupported)`). Any other error would be a bug.
fn is_supported<T>(result: Result<T, Error>, what: &str, ty: GgmlType) -> bool {
    match result {
        Ok(_) => true,
        Err(Error::Unsupported { .. }) => false,
        Err(e) => panic!("{ty}: {what} of nothing failed with {e}"),
    }
}

#[test]
fn every_type_is_covered_by_a_fixture_or_returns_unsupported() {
    // A type added to ares-core later shows up in ALL_TYPES and fails here until
    // someone either implements it and generates its fixture or makes it Unsupported.
    for ty in ALL_TYPES {
        let can_decode = is_supported(dequantize(ty, &[], &mut []), "dequantize", ty);
        let has_fixture = fixture_path("dequant", ty).exists();
        assert_eq!(
            can_decode,
            has_fixture,
            "{ty}: dequantize {} but its golden fixture {}",
            if can_decode { "works" } else { "is Unsupported" },
            if has_fixture { "exists" } else { "is missing" },
        );

        let can_encode = is_supported(quantize(ty, &[], &mut []), "quantize", ty);
        let has_fixture = fixture_path("quantize", ty).exists();
        assert_eq!(
            can_encode,
            has_fixture,
            "{ty}: quantize {} but its golden fixture {}",
            if can_encode { "works" } else { "is Unsupported" },
            if has_fixture { "exists" } else { "is missing" },
        );
    }
}

#[test]
fn the_golden_directory_holds_no_fixture_that_belongs_to_no_type() {
    let mut known: Vec<String> = Vec::new();
    for ty in ALL_TYPES {
        known.push(format!("dequant_{}.bin", ty.name()));
        known.push(format!("quantize_{}.bin", ty.name()));
    }
    for entry in fs::read_dir(golden_dir()).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if name.ends_with(".bin") {
            assert!(known.contains(&name), "{name} is not the fixture of any known type");
        }
    }
}

#[test]
fn the_ggml_unsupported_types_stay_unsupported_with_a_reason() {
    // The types that ggml cannot convert to float, or that are not floating-point
    // weights at all, must say so by name instead of failing on a length check.
    for ty in
        [GgmlType::Q8_1, GgmlType::I8, GgmlType::I16, GgmlType::I32, GgmlType::I64, GgmlType::F64]
    {
        match dequantize(ty, &[1, 2, 3], &mut [0.0; 7]) {
            Err(Error::Unsupported { ty: reported, .. }) => assert_eq!(reported, ty),
            other => panic!("{ty}: expected Unsupported, got {other:?}"),
        }
    }
}
