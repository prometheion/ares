//! CPU reference for the tensor element types of GGUF files.
//!
//! The engine runs quantized models on the GPU with its own CUDA kernels. This
//! crate is what those kernels are checked against: [`dequantize`] turns a
//! stream of blocks of any GGUF type into `f32` exactly as ggml's
//! `dequantize_row_*` functions do, and [`quantize`] produces the same bytes as
//! ggml's `quantize_row_*_ref` for the few simple types the KV cache uses. It is
//! also the fallback for anything that has no GPU path yet.
//!
//! The engine's own KV-cache formats (`f16` and the rotated `q8` to `q4`) are
//! specified and implemented in the [`kv`] module, which is the reference the
//! attention kernels are held to.
//!
//! Correctness and readability come first: plain scalar Rust, no SIMD, no
//! `unsafe`, and every function ports its ggml counterpart statement by
//! statement. It does not allocate, so decoding a whole weight matrix costs
//! only the output buffer.
//!
//! # Exactness
//!
//! Decoding is bit-for-bit equal to ggml's C code, and encoding byte-for-byte,
//! which the tests prove with fixtures that `tests/golden/generate.py` makes by
//! calling ggml itself.
//!
//! That holds because neither side fuses a multiply and an add: Rust never
//! does, and ggml's `dequantize_row_*` functions live in `ggml-base`, which the
//! default build compiles without `-mfma` or `-march` (the library the
//! fixtures came from contains no fused multiply-add instruction). A GPU kernel
//! that does fuse `d * q - m` can differ from this reference by an ulp or so per
//! fused operation, so compare kernels against it with a tolerance, not for
//! equality.
//!
//! Q8_K is decoded although ggml does not register a `to_float` for it in its
//! type table (it is an activation format that never appears in model files):
//! `dequantize_row_q8_K` exists, is exported, and is what the fixture uses.

#![forbid(unsafe_code)]

mod blocks;
mod encode;
mod error;
mod float;
mod fp4;
mod half;
mod iquants;
mod kquants;
pub mod kv;
pub mod kvhead;
pub mod layout;
mod legacy;
#[rustfmt::skip]
mod tables;
mod ternary;

use ares_core::GgmlType;

pub use error::{Error, Operation};
pub use half::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
pub use kv::{
    GROUP_VALUES as KV_GROUP_VALUES, KvFormat, KvPolicy, TILE_CELLS as KV_TILE_CELLS, decode_row,
    decode_row_rotated, decode_tile, decode_tile_rotated, encode_row, encode_tile,
    kv_bytes_per_cell, kv_mean_bytes_per_tile, rotate_groups,
};
pub use kvhead::{Attention, AttnState, KvHead, KvHeadError, KvUnit};

/// Decodes `src`, a stream of whole blocks of type `ty`, into `dst`.
///
/// `dst.len()` is the number of values; `src.len()` must be exactly
/// `ty.bytes_for(dst.len())`. Nothing is written unless the call succeeds.
///
/// Every type that ggml can convert to float is supported: the float types,
/// the legacy `Q*` types, the K-quants (Q8_K included), the i-quants, the
/// ternary types and the 4-bit float types. The integer types and F64 are not
/// weights and return [`Error::Unsupported`], as does Q8_1, for which ggml has
/// no decoder.
///
/// Block bytes that no quantizer would write (a TQ1_0 byte above 242, the
/// fourth TQ2_0 code, any E8M0 or UE4M3 scale) decode to whatever ggml's code
/// gives for them, and a NaN scale in the data gives NaN values.
pub fn dequantize(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), Error> {
    let decode = decoder(ty)?;
    check_lengths(ty, dst.len(), src.len())?;
    decode(src, dst);
    Ok(())
}

/// Encodes `src` into `dst` as blocks of type `ty`, matching ggml's
/// `quantize_row_*_ref` byte for byte.
///
/// `src.len()` must be a whole number of blocks and `dst.len()` exactly
/// `ty.bytes_for(src.len())`. Only the types the KV cache stores are supported:
/// F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0.
///
/// The input of the quantized types must be finite and its magnitude moderate
/// (below about 1e30 and not subnormal): ggml turns floats into small integers
/// with C casts, which are undefined for NaN, infinity and out-of-range values,
/// so there is no reference result to match. F16 and BF16 accept every `f32`.
pub fn quantize(ty: GgmlType, src: &[f32], dst: &mut [u8]) -> Result<(), Error> {
    let encode = encoder(ty)?;
    check_lengths(ty, src.len(), dst.len())?;
    encode(src, dst);
    Ok(())
}

type Decoder = fn(&[u8], &mut [f32]);
type Encoder = fn(&[f32], &mut [u8]);

/// The decoder of a type. The match has no wildcard arm on purpose: a type added
/// to `ares-core` does not compile here until someone decides what it does.
fn decoder(ty: GgmlType) -> Result<Decoder, Error> {
    use GgmlType::*;
    Ok(match ty {
        F32 => float::f32,
        F16 => float::f16,
        BF16 => float::bf16,
        Q1_0 => legacy::q1_0,
        Q2_0 => legacy::q2_0,
        Q4_0 => legacy::q4_0,
        Q4_1 => legacy::q4_1,
        Q5_0 => legacy::q5_0,
        Q5_1 => legacy::q5_1,
        Q8_0 => legacy::q8_0,
        Q2_K => kquants::q2_k,
        Q3_K => kquants::q3_k,
        Q4_K => kquants::q4_k,
        Q5_K => kquants::q5_k,
        Q6_K => kquants::q6_k,
        Q8_K => kquants::q8_k,
        IQ1_S => iquants::iq1_s,
        IQ1_M => iquants::iq1_m,
        IQ2_XXS => iquants::iq2_xxs,
        IQ2_XS => iquants::iq2_xs,
        IQ2_S => iquants::iq2_s,
        IQ3_XXS => iquants::iq3_xxs,
        IQ3_S => iquants::iq3_s,
        IQ4_NL => iquants::iq4_nl,
        IQ4_XS => iquants::iq4_xs,
        TQ1_0 => ternary::tq1_0,
        TQ2_0 => ternary::tq2_0,
        MXFP4 => fp4::mxfp4,
        NVFP4 => fp4::nvfp4,
        Q8_1 => {
            return Err(unsupported(
                ty,
                Operation::Dequantize,
                "ggml has no dequantizer for q8_1, which only holds activations",
            ));
        }
        I8 | I16 | I32 | I64 | F64 => {
            return Err(unsupported(
                ty,
                Operation::Dequantize,
                "integer and f64 tensors are not floating-point weights",
            ));
        }
    })
}

/// The encoder of a type; same rule about the wildcard as [`decoder`].
fn encoder(ty: GgmlType) -> Result<Encoder, Error> {
    use GgmlType::*;
    Ok(match ty {
        F16 => float::encode_f16,
        BF16 => float::encode_bf16,
        Q4_0 => encode::q4_0,
        Q4_1 => encode::q4_1,
        Q5_0 => encode::q5_0,
        Q5_1 => encode::q5_1,
        Q8_0 => encode::q8_0,
        F32 | F64 | I8 | I16 | I32 | I64 | Q8_1 | Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K
        | IQ1_S | IQ1_M | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ4_NL | IQ4_XS | TQ1_0
        | TQ2_0 | MXFP4 | NVFP4 | Q1_0 | Q2_0 => {
            return Err(unsupported(
                ty,
                Operation::Quantize,
                "only the KV cache types (f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0) can be encoded",
            ));
        }
    })
}

fn unsupported(ty: GgmlType, operation: Operation, reason: &'static str) -> Error {
    Error::Unsupported { ty, operation, reason }
}

/// Checks that `bytes` is exactly the size of `values` values of `ty`.
fn check_lengths(ty: GgmlType, values: usize, bytes: usize) -> Result<(), Error> {
    let block_values = ty.block_values();
    if !values.is_multiple_of(block_values as usize) {
        return Err(Error::PartialBlock { ty, values, block_values });
    }
    // A size that overflows u64 cannot equal the length of any slice.
    let expected = ty.bytes_for(values as u64).unwrap_or(u64::MAX);
    if expected != bytes as u64 {
        return Err(Error::ByteLength { ty, values, expected, actual: bytes });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ares_core::ggml_type::ALL_TYPES;

    use super::*;

    #[test]
    fn dequantize_refuses_a_byte_length_that_is_not_the_blocks_of_the_values() {
        let mut dst = [7.0f32; 32];
        // One Q4_0 block is 18 bytes.
        for len in [0, 17, 19, 36] {
            let err = dequantize(GgmlType::Q4_0, &vec![0; len], &mut dst).unwrap_err();
            assert_eq!(
                err,
                Error::ByteLength { ty: GgmlType::Q4_0, values: 32, expected: 18, actual: len }
            );
        }
        assert!(dst.iter().all(|&v| v == 7.0), "a refused call must not write");
    }

    #[test]
    fn dequantize_refuses_a_partial_block_of_values() {
        let mut dst = [0.0f32; 33];
        let err = dequantize(GgmlType::Q4_0, &[0; 18], &mut dst).unwrap_err();
        assert_eq!(err, Error::PartialBlock { ty: GgmlType::Q4_0, values: 33, block_values: 32 });
        // Plain types have blocks of one value and never trip this.
        assert!(dequantize(GgmlType::F32, &[0; 132], &mut dst).is_ok());
    }

    #[test]
    fn quantize_refuses_wrong_lengths() {
        let src = [1.0f32; 64];
        let mut dst = [0u8; 70];
        let err = quantize(GgmlType::Q8_0, &src, &mut dst).unwrap_err();
        assert_eq!(
            err,
            Error::ByteLength { ty: GgmlType::Q8_0, values: 64, expected: 68, actual: 70 }
        );
        let err = quantize(GgmlType::Q8_0, &src[..40], &mut dst[..34]).unwrap_err();
        assert!(matches!(err, Error::PartialBlock { values: 40, .. }));
    }

    #[test]
    fn types_without_a_decoder_say_so_instead_of_failing_on_the_length() {
        for ty in [GgmlType::I8, GgmlType::I16, GgmlType::I32, GgmlType::I64, GgmlType::F64] {
            let err = dequantize(ty, &[], &mut []).unwrap_err();
            assert!(matches!(err, Error::Unsupported { operation: Operation::Dequantize, .. }));
        }
        // Even with lengths that would also be wrong, the type is what is reported.
        let err = dequantize(GgmlType::Q8_1, &[1, 2, 3], &mut [0.0; 5]).unwrap_err();
        assert!(matches!(err, Error::Unsupported { ty: GgmlType::Q8_1, .. }));
        assert!(err.to_string().contains("q8_1"), "{err}");
    }

    #[test]
    fn only_the_kv_cache_types_can_be_encoded() {
        let encodable = [
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
        ];
        for ty in ALL_TYPES {
            let result = quantize(ty, &[], &mut []);
            if encodable.contains(&ty) {
                assert_eq!(result, Ok(()), "{ty}");
            } else {
                assert!(
                    matches!(
                        result,
                        Err(Error::Unsupported { operation: Operation::Quantize, .. })
                    ),
                    "{ty}"
                );
            }
        }
    }

    #[test]
    fn q4_0_layout_low_nibbles_first_then_high_nibbles() {
        // d = 0.5 (f16 0x3800). Byte j holds value j in the low nibble and value
        // j + 16 in the high nibble; each nibble has 8 subtracted.
        let mut block = vec![0x00, 0x38];
        block.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let mut out = [0.0f32; 32];
        dequantize(GgmlType::Q4_0, &block, &mut out).unwrap();
        for j in 0..16 {
            assert_eq!(out[j], (j as f32 - 8.0) * 0.5, "low nibble {j}");
            assert_eq!(out[j + 16], ((15 - j) as f32 - 8.0) * 0.5, "high nibble {j}");
        }
    }

    #[test]
    fn q8_0_round_trip_stays_within_half_a_step() {
        // Proves quantize and dequantize agree on the layout, independently of
        // the byte comparison with ggml in the golden tests.
        let x: Vec<f32> = (0..64).map(|i| ((i * 37 % 64) as f32 - 31.5) * 0.173).collect();
        let mut bytes = vec![0u8; GgmlType::Q8_0.bytes_for(64).unwrap() as usize];
        quantize(GgmlType::Q8_0, &x, &mut bytes).unwrap();
        let mut back = vec![0.0f32; 64];
        dequantize(GgmlType::Q8_0, &bytes, &mut back).unwrap();
        for block in 0..2 {
            let d = f16_to_f32(u16::from_le_bytes([bytes[34 * block], bytes[34 * block + 1]]));
            for i in 32 * block..32 * (block + 1) {
                assert!(
                    (x[i] - back[i]).abs() <= d * 0.5 + 1e-6,
                    "value {i}: {} vs {}",
                    x[i],
                    back[i]
                );
            }
        }
    }

    #[test]
    fn an_all_zero_block_encodes_to_zero_codes_and_decodes_to_zero() {
        for ty in [GgmlType::Q4_0, GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::Q4_1, GgmlType::Q5_1] {
            let n = ty.block_values() as usize;
            let mut bytes = vec![0xaa; ty.block_bytes() as usize];
            quantize(ty, &vec![0.0; n], &mut bytes).unwrap();
            let mut back = vec![1.0; n];
            dequantize(ty, &bytes, &mut back).unwrap();
            assert!(back.iter().all(|&v| v == 0.0), "{ty}: {back:?}");
        }
    }
}
