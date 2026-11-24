//! Weight blocks of every type for the tests of the weight kernels: the 28 weight types, and
//! blocks of random bytes with finite scales, valid in each of them.

use ares_core::GgmlType;

use super::Rng;

/// The weight types, in the order the tests of the decode kernels go through them: first the
/// eleven of the first round of kernels, then the others.
pub const TYPES: [GgmlType; 28] = [
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q3_K,
    GgmlType::IQ4_XS,
    GgmlType::IQ4_NL,
    GgmlType::IQ3_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ2_S,
    GgmlType::IQ2_XS,
    GgmlType::Q8_0,
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q2_K,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::IQ2_XXS,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::Q2_0,
];

/// The byte offsets of the f16 scales of a block of `ty` that must be finite for the data to
/// be meaningful; everything else in a block may be any bytes. IQ1_M spreads its scale over
/// other fields, MXFP4 and NVFP4 keep theirs in other forms and the float types have none.
pub fn scale_offsets(ty: GgmlType) -> &'static [usize] {
    use GgmlType::*;
    match ty {
        Q4_K | Q5_K | Q4_1 | Q5_1 => &[0, 2],
        Q6_K => &[208],
        Q3_K => &[108],
        Q2_K => &[80, 82],
        TQ1_0 => &[52],
        TQ2_0 => &[64],
        IQ1_M | MXFP4 | NVFP4 | F32 | F16 | BF16 => &[],
        _ => &[0],
    }
}

/// The types whose weights are floats: their kernels multiply the activations exactly.
pub fn is_float(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// Spreads the f16 `d` of an IQ1_M block over the top nibbles of the four u16 of its scales.
pub fn put_iq1_m_scale(block: &mut [u8], d: u16) {
    for i in 0..4 {
        let at = 48 + 2 * i;
        let word = u16::from_le_bytes([block[at], block[at + 1]]);
        let word = (word & 0x0FFF) | (((d >> (4 * i)) & 0xF) << 12);
        block[at..at + 2].copy_from_slice(&word.to_le_bytes());
    }
}

/// `n_blocks` blocks of `ty` of random bytes, with finite f16 scales of both signs (any other
/// bytes are valid in every one of these types); for a float type, `n_blocks` values.
pub fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    if is_float(ty) {
        // n_blocks values (a block of a float type is one value), of both signs
        let mut bytes = Vec::with_capacity(n_blocks * ty.block_bytes() as usize);
        for _ in 0..n_blocks {
            let v = rng.unit() * 2.0;
            match ty {
                GgmlType::F32 => bytes.extend_from_slice(&v.to_le_bytes()),
                GgmlType::F16 => bytes.extend_from_slice(&ares_quant::f32_to_f16(v).to_le_bytes()),
                _ => bytes.extend_from_slice(&ares_quant::f32_to_bf16(v).to_le_bytes()),
            }
        }
        return bytes;
    }
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for blk in bytes.chunks_exact_mut(block) {
        for &offset in scale_offsets(ty) {
            let scale = (0.002 + 0.02 * (rng.unit() * 0.5 + 0.5))
                * if rng.next().is_multiple_of(4) { -1.0 } else { 1.0 };
            blk[offset..offset + 2].copy_from_slice(&ares_quant::f32_to_f16(scale).to_le_bytes());
        }
        match ty {
            GgmlType::IQ1_M => {
                let d = ares_quant::f32_to_f16(0.002 + 0.02 * (rng.unit() * 0.5 + 0.5));
                put_iq1_m_scale(blk, d);
            }
            // An E8M0 exponent around 1 (2^-9 to 2^10), not the huge ones.
            GgmlType::MXFP4 => blk[0] = 118 + (rng.next() % 20) as u8,
            // Four UE4M3 scales of exponent 3..10, never the NaN code.
            GgmlType::NVFP4 => {
                for byte in blk.iter_mut().take(4) {
                    *byte = ((3 + rng.next() % 8) << 3 | (rng.next() % 8)) as u8;
                }
            }
            _ => {}
        }
    }
    bytes
}

/// The scale of a block of [`random_blocks_of_one_size`]: between 0.002 and 0.022 for the
/// types whose codes are small numbers, as the blocks of real tensors have, and less by the
/// size of the codes for the types whose codes and inner scales are large (a block of Q6_K
/// decodes to up to 4096 times its scale), so that the weights of every type are of one size.
fn block_scale(rng: &mut Rng, ty: GgmlType) -> f32 {
    let codes = match ty {
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ3_S => 8.0,
        GgmlType::Q6_K | GgmlType::IQ4_XS => 24.0,
        GgmlType::IQ3_XXS => 3.0,
        _ => 1.0,
    };
    (0.002 + 0.02 * (rng.unit() * 0.5 + 0.5)) / codes
}

/// `n_blocks` random blocks of a weight type with finite scales (any other bytes are valid in
/// every one of these types), whose weights are of one size whatever the type, as the
/// experts' tests mix the types of a layer; for a float type, `n_blocks` values.
pub fn random_blocks_of_one_size(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for blk in bytes.chunks_exact_mut(block) {
        let scale = block_scale(rng, ty);
        for &at in scale_offsets(ty) {
            let signed = if rng.next().is_multiple_of(4) { -scale } else { scale };
            blk[at..at + 2].copy_from_slice(&ares_quant::f32_to_f16(signed).to_le_bytes());
        }
        match ty {
            GgmlType::IQ1_M => put_iq1_m_scale(blk, ares_quant::f32_to_f16(scale)),
            // An E8M0 exponent: 2^-7 to 2^-4.
            GgmlType::MXFP4 => blk[0] = 121 + (rng.next() % 4) as u8,
            // Four UE4M3 scales of exponent 1 to 4 (2^-7 to 2^-4), never the NaN code.
            GgmlType::NVFP4 => {
                for byte in blk.iter_mut().take(4) {
                    *byte = ((1 + rng.next() % 4) << 3 | (rng.next() % 8)) as u8;
                }
            }
            GgmlType::F32 => blk.copy_from_slice(&(rng.unit() * 0.2).to_le_bytes()),
            GgmlType::F16 => {
                blk.copy_from_slice(&ares_quant::f32_to_f16(rng.unit() * 0.2).to_le_bytes());
            }
            GgmlType::BF16 => {
                blk.copy_from_slice(&ares_quant::f32_to_bf16(rng.unit() * 0.2).to_le_bytes());
            }
            _ => {}
        }
    }
    bytes
}

/// The largest tensor of two or more dimensions of every weight type the file has: the
/// blocks a quantizer wrote, which the experts' tests take as the matrices of experts.
pub fn real_blocks(gguf: &ares_gguf::Gguf) -> Vec<(GgmlType, &[u8])> {
    let mut found: Vec<(GgmlType, &[u8])> = Vec::new();
    for tensor in gguf.tensors() {
        if tensor.dims.as_slice().len() < 2 || !tensor.ty.is_weight() {
            continue;
        }
        let data = gguf.data(tensor);
        match found.iter_mut().find(|(ty, _)| *ty == tensor.ty) {
            Some((_, largest)) if largest.len() < data.len() => *largest = data,
            Some(_) => {}
            None => found.push((tensor.ty, data)),
        }
    }
    found
}

/// A source of blocks that takes the blocks of `ty` from `data`, one stretch after the other
/// and from the start again at the end, and makes the blocks of every other type with
/// [`random_blocks_of_one_size`].
pub fn blocks_of<'a>(
    ty: GgmlType,
    data: &'a [u8],
) -> impl FnMut(&mut Rng, GgmlType, usize) -> Vec<u8> + 'a {
    let block = ty.block_bytes() as usize;
    let mut at = 0;
    move |rng, wanted, n_blocks| {
        if wanted != ty {
            return random_blocks_of_one_size(rng, wanted, n_blocks);
        }
        let mut bytes = Vec::with_capacity(n_blocks * block);
        while bytes.len() < n_blocks * block {
            if at + block > data.len() {
                at = 0;
            }
            let take = (n_blocks * block - bytes.len()).min((data.len() - at) / block * block);
            bytes.extend_from_slice(&data[at..at + take]);
            at += take;
        }
        bytes
    }
}
