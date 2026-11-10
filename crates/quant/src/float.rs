//! The unquantized types: one value per "block".

use crate::half::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};

/// F32: a bit-exact copy, NaN payloads and the sign of zero included.
pub(crate) fn f32(src: &[u8], dst: &mut [f32]) {
    let (values, _) = src.as_chunks::<4>();
    for (v, bytes) in dst.iter_mut().zip(values) {
        *v = f32::from_le_bytes(*bytes);
    }
}

/// F16 to `f32`.
pub(crate) fn f16(src: &[u8], dst: &mut [f32]) {
    let (values, _) = src.as_chunks::<2>();
    for (v, bytes) in dst.iter_mut().zip(values) {
        *v = f16_to_f32(u16::from_le_bytes(*bytes));
    }
}

/// BF16 to `f32`.
pub(crate) fn bf16(src: &[u8], dst: &mut [f32]) {
    let (values, _) = src.as_chunks::<2>();
    for (v, bytes) in dst.iter_mut().zip(values) {
        *v = bf16_to_f32(u16::from_le_bytes(*bytes));
    }
}

/// `f32` to F16, as ggml's `ggml_fp32_to_fp16_row`.
pub(crate) fn encode_f16(src: &[f32], dst: &mut [u8]) {
    let (values, _) = dst.as_chunks_mut::<2>();
    for (v, bytes) in src.iter().zip(values) {
        *bytes = f32_to_f16(*v).to_le_bytes();
    }
}

/// `f32` to BF16, as ggml's `ggml_fp32_to_bf16_row_ref`.
pub(crate) fn encode_bf16(src: &[f32], dst: &mut [u8]) {
    let (values, _) = dst.as_chunks_mut::<2>();
    for (v, bytes) in src.iter().zip(values) {
        *bytes = f32_to_bf16(*v).to_le_bytes();
    }
}
