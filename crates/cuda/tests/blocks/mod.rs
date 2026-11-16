//! Weight blocks of every type for the tests of the weight kernels: the 28 weight types, and
//! blocks of random bytes with finite scales, valid in each of them. Shared by `mmvq.rs` and
//! `q4.rs`.

// Each test crate uses what it needs of this module.
#![allow(dead_code)]

use ares_core::GgmlType;

/// The types with a fast kernel, and the byte offsets of the f16 fields of a
/// block that must be finite for the test data to be meaningful (everything
/// else in a block may be any bytes).
pub const TYPES: [(GgmlType, &[usize]); 28] = [
    (GgmlType::Q4_K, &[0, 2]),
    (GgmlType::Q5_K, &[0, 2]),
    (GgmlType::Q6_K, &[208]),
    (GgmlType::Q3_K, &[108]),
    (GgmlType::IQ4_XS, &[0]),
    (GgmlType::IQ4_NL, &[0]),
    (GgmlType::IQ3_S, &[0]),
    (GgmlType::IQ3_XXS, &[0]),
    (GgmlType::IQ2_S, &[0]),
    (GgmlType::IQ2_XS, &[0]),
    (GgmlType::Q8_0, &[0]),
    // The types of the second round. The float types and the three whose scales are
    // not plain f16 fields (IQ1_M, MXFP4, NVFP4) are made by `random_blocks`.
    (GgmlType::F32, &[]),
    (GgmlType::F16, &[]),
    (GgmlType::BF16, &[]),
    (GgmlType::Q4_0, &[0]),
    (GgmlType::Q4_1, &[0, 2]),
    (GgmlType::Q5_0, &[0]),
    (GgmlType::Q5_1, &[0, 2]),
    (GgmlType::Q2_K, &[80, 82]),
    (GgmlType::IQ1_S, &[0]),
    (GgmlType::IQ1_M, &[]),
    (GgmlType::IQ2_XXS, &[0]),
    (GgmlType::TQ1_0, &[52]),
    (GgmlType::TQ2_0, &[64]),
    (GgmlType::MXFP4, &[]),
    (GgmlType::NVFP4, &[]),
    (GgmlType::Q1_0, &[0]),
    (GgmlType::Q2_0, &[0]),
];

/// The types whose weights are floats: their kernels multiply the activations exactly.
pub fn is_float(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// A value in [-1, 1), as the tests' generators make it.
fn unit(next: &mut impl FnMut() -> u64) -> f32 {
    ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// `n_blocks` blocks of `ty` of random bytes, with finite f16 scales of both
/// signs (any other bytes are valid in every one of these types).
pub fn random_blocks(next: &mut impl FnMut() -> u64, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    if is_float(ty) {
        // n_blocks values (a block of a float type is one value), of both signs
        let mut bytes = Vec::with_capacity(n_blocks * ty.block_bytes() as usize);
        for _ in 0..n_blocks {
            let v = unit(next) * 2.0;
            match ty {
                GgmlType::F32 => bytes.extend_from_slice(&v.to_le_bytes()),
                GgmlType::F16 => bytes.extend_from_slice(&ares_quant::f32_to_f16(v).to_le_bytes()),
                _ => bytes.extend_from_slice(&ares_quant::f32_to_bf16(v).to_le_bytes()),
            }
        }
        return bytes;
    }
    let (_, scales) = TYPES.iter().find(|(t, _)| *t == ty).unwrap();
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (next() >> 32) as u8).collect();
    for b in 0..n_blocks {
        let blk = &mut bytes[b * block..(b + 1) * block];
        for &offset in *scales {
            let scale = (0.002 + 0.02 * (unit(next) * 0.5 + 0.5))
                * if next().is_multiple_of(4) { -1.0 } else { 1.0 };
            blk[offset..offset + 2].copy_from_slice(&ares_quant::f32_to_f16(scale).to_le_bytes());
        }
        match ty {
            // The f16 d is spread over the top nibbles of the four u16 of the scales.
            GgmlType::IQ1_M => {
                let d = ares_quant::f32_to_f16(0.002 + 0.02 * (unit(next) * 0.5 + 0.5));
                for i in 0..4 {
                    let at = 48 + 2 * i;
                    let word = u16::from_le_bytes([blk[at], blk[at + 1]]);
                    let word = (word & 0x0FFF) | (((d >> (4 * i)) & 0xF) << 12);
                    blk[at..at + 2].copy_from_slice(&word.to_le_bytes());
                }
            }
            // An E8M0 exponent around 1 (2^-9 to 2^10), not the huge ones.
            GgmlType::MXFP4 => blk[0] = 118 + (next() % 20) as u8,
            // Four UE4M3 scales of exponent 3..10, never the NaN code.
            GgmlType::NVFP4 => {
                for byte in blk.iter_mut().take(4) {
                    *byte = ((3 + next() % 8) << 3 | (next() % 8)) as u8;
                }
            }
            _ => {}
        }
    }
    bytes
}
