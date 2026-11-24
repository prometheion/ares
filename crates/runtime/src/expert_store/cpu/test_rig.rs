//! What the tests of the CPU's expert jobs share: a generator of fixed
//! sequences, random blocks of every weight type, and the expert in exact
//! arithmetic on the decoded weights to compare a job with.

use ares_core::GgmlType;
use ares_quant::act;

/// A small generator with a fixed sequence (xorshift64*).
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub(super) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Roughly normal, with a rare value ten times as large.
    pub(super) fn activation(&mut self) -> f32 {
        let v: f32 = (0..6).map(|_| self.unit() - 0.5).sum();
        if self.next().is_multiple_of(61) { 10.0 * v } else { v }
    }
}

/// The byte offsets of the f16 scales of a block of `ty`.
pub(super) fn scale_fields(ty: GgmlType) -> &'static [usize] {
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
pub(super) fn random_matrix(rng: &mut Rng, ty: GgmlType, values: usize) -> Vec<u8> {
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

pub(super) fn decoded(ty: GgmlType, bytes: &[u8]) -> Vec<f64> {
    let mut out =
        vec![0.0f32; bytes.len() / ty.block_bytes() as usize * ty.block_values() as usize];
    ares_quant::dequantize(ty, bytes, &mut out).expect("whole blocks");
    out.iter().map(|v| f64::from(*v)).collect()
}

/// The activations the kernels see: codes times scales.
pub(super) fn dequantized_act(act: &[u8], n_in: usize, n_rows: usize) -> Vec<f64> {
    let ds_at = (n_rows * n_in).next_multiple_of(16);
    (0..n_rows * n_in)
        .map(|i| {
            let at = ds_at + 8 * (i / 32);
            let d = f32::from_le_bytes([act[at], act[at + 1], act[at + 2], act[at + 3]]);
            f64::from(act[i] as i8) * f64::from(d)
        })
        .collect()
}

/// The expert in exact arithmetic on the decoded weights. `quantized` rounds
/// the hidden values as the kernels do; without it nothing is rounded.
pub(super) fn reference(
    gate: &[f64],
    up: &[f64],
    down: &[f64],
    x: &[f64],
    n_ff: usize,
    quantized: bool,
) -> Vec<f64> {
    let n_in = x.len();
    let dot = |w: &[f64], x: &[f64]| w.iter().zip(x).map(|(w, x)| w * x).sum::<f64>();
    let mut h: Vec<f64> = (0..n_ff)
        .map(|j| {
            let g = dot(&gate[j * n_in..(j + 1) * n_in], x);
            let u = dot(&up[j * n_in..(j + 1) * n_in], x);
            if quantized {
                let (g, u) = (g as f32, u as f32);
                f64::from(g / (1.0 + (-g).exp()) * u)
            } else {
                g / (1.0 + (-g).exp()) * u
            }
        })
        .collect();
    if quantized {
        for chunk in h.as_chunks_mut::<{ act::GROUP }>().0.iter_mut() {
            let group = act::Group::of(&chunk.map(|v| v as f32));
            for (i, v) in chunk.iter_mut().enumerate() {
                *v = f64::from(group.value(i));
            }
        }
    }
    (0..n_in).map(|i| dot(&down[i * n_ff..(i + 1) * n_ff], &h)).collect()
}

pub(super) fn relative_rms(got: &[f64], want: &[f64]) -> f64 {
    let err: f64 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
    let norm: f64 = want.iter().map(|w| w * w).sum();
    (err / norm).sqrt()
}
