//! The int8 form of activations the integer weight kernels read, and its quantization: what
//! `ares_quantize_act` writes on a device (`ares-cuda`, `kernels/quantize_act.cuh`; the
//! Vulkan shader of the same name) and the CPU kernels of the expert store read.
//!
//! The values are quantized in groups of [`GROUP`]: per group `d = max|x| / 127`, the codes
//! `q = round(x / d)` (halves away from zero) as int8, and `s = d * sum(q)`, the sum of the
//! dequantized values `d * q` of the group, which the weight types with an offset per
//! sub-block multiply their offset by. A group of zeros has `d = 0` and codes of zero.
//!
//! The bytes of `n` values (the rows of a batch one after another, each whole groups) are,
//! in this order ([`Layout`]):
//!
//! 1. the `n` codes;
//! 2. from the next multiple of 16 bytes, the pairs `(d, s)` of f32, one per group;
//! 3. from the next multiple of 16 bytes, the `n` values themselves as f32, for the weight
//!    types that are floats and multiply them exactly.

/// The values that share a scale.
pub const GROUP: usize = 32;

/// Where the parts of the quantized form of a number of values lie, in bytes from its start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The start of the `(d, s)` pairs.
    pub pairs_at: u64,
    /// The end of the pairs: what a reader of the codes and the pairs needs.
    pub pairs_end: u64,
    /// The start of the f32 copy of the values.
    pub values_at: u64,
    /// The bytes of the whole form.
    pub bytes: u64,
}

impl Layout {
    /// The layout of `n_values` values (whole groups, or the last group is counted whole).
    pub const fn of(n_values: u64) -> Self {
        let pairs_at = n_values.next_multiple_of(16);
        let pairs_end = pairs_at + n_values.div_ceil(GROUP as u64) * 8;
        let values_at = pairs_end.next_multiple_of(16);
        Self { pairs_at, pairs_end, values_at, bytes: values_at + 4 * n_values }
    }
}

/// One group quantized: its codes, `d` and `s`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Group {
    pub codes: [i8; GROUP],
    pub d: f32,
    pub s: f32,
}

impl Group {
    /// Quantizes the values of a group as the kernels do, bit for bit: the codes are at most
    /// 127 in magnitude and their f32 sum is exact in any order.
    pub fn of(x: &[f32; GROUP]) -> Self {
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let mut codes = [0i8; GROUP];
        let mut sum = 0.0f32;
        if amax != 0.0 {
            for (code, v) in codes.iter_mut().zip(x) {
                let q = (v / d).round();
                *code = q as i8;
                sum += q;
            }
        }
        Self { codes, d, s: d * sum }
    }

    /// The value the kernels compute with for code `i`: `d * q`, rounded to f32.
    pub fn value(&self, i: usize) -> f32 {
        self.d * f32::from(self.codes[i])
    }
}

/// The groups of `x`, whose length must be whole groups.
pub fn groups(x: &[f32]) -> impl Iterator<Item = Group> + '_ {
    let (groups, rest) = x.as_chunks::<GROUP>();
    assert!(rest.is_empty(), "{} values are not whole groups of {GROUP}", x.len());
    groups.iter().map(Group::of)
}

/// `x` as the integer weight kernels see it: each value `d * q`, rounded to f32.
pub fn dequantized(x: &[f32]) -> Vec<f32> {
    groups(x).flat_map(|g| (0..GROUP).map(move |i| g.value(i))).collect()
}

/// The quantized form of `x` (whole groups), as `ares_quantize_act` writes it: codes, pairs
/// and the f32 copy, [`Layout::of`]`(x.len()).bytes` bytes.
pub fn quantize(x: &[f32]) -> Vec<u8> {
    let layout = Layout::of(x.len() as u64);
    let mut bytes = vec![0u8; layout.bytes as usize];
    let pairs_at = layout.pairs_at as usize;
    for (g, group) in groups(x).enumerate() {
        for (i, code) in group.codes.iter().enumerate() {
            bytes[GROUP * g + i] = *code as u8;
        }
        let at = pairs_at + 8 * g;
        bytes[at..at + 4].copy_from_slice(&group.d.to_le_bytes());
        bytes[at + 4..at + 8].copy_from_slice(&group.s.to_le_bytes());
    }
    let values_at = layout.values_at as usize;
    for (i, v) in x.iter().enumerate() {
        bytes[values_at + 4 * i..values_at + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_starts_each_part_on_16_bytes() {
        // 64 values: codes 0..64, two pairs 64..80, values 80..336.
        assert_eq!(
            Layout::of(64),
            Layout { pairs_at: 64, pairs_end: 80, values_at: 80, bytes: 336 }
        );
        // 32 values: one pair ends at 40, the values start at 48.
        assert_eq!(
            Layout::of(32),
            Layout { pairs_at: 32, pairs_end: 40, values_at: 48, bytes: 176 }
        );
        assert_eq!(Layout::of(0).bytes, 0);
    }

    #[test]
    fn a_group_rounds_halves_away_from_zero_and_zeros_stay_zero() {
        let mut x = [0.0f32; GROUP];
        assert_eq!(Group::of(&x), Group { codes: [0; GROUP], d: 0.0, s: 0.0 });
        // amax 127: d = 1, so the codes are the values rounded, halves away from zero.
        x[0] = 127.0;
        x[1] = 2.5;
        x[2] = -2.5;
        x[3] = -0.4;
        let g = Group::of(&x);
        assert_eq!(g.d, 1.0);
        assert_eq!(&g.codes[..4], &[127, 3, -3, 0]);
        assert_eq!(g.s, 127.0);
        assert_eq!(g.value(1), 3.0);
    }

    #[test]
    fn the_form_holds_codes_pairs_and_values() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.25).collect();
        let bytes = quantize(&x);
        let layout = Layout::of(64);
        assert_eq!(bytes.len() as u64, layout.bytes);
        let g = Group::of(x[32..].as_chunks::<GROUP>().0.first().unwrap());
        assert_eq!(bytes[32] as i8, g.codes[0]);
        let at = layout.pairs_at as usize + 8;
        assert_eq!(f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()), g.d);
        assert_eq!(f32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()), g.s);
        let at = layout.values_at as usize + 4 * 63;
        assert_eq!(f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()), x[63]);
        let first = Group::of(x[..GROUP].as_chunks::<GROUP>().0.first().unwrap());
        assert_eq!(dequantized(&x)[5], first.value(5));
    }
}
