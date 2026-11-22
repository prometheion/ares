//! The codebooks of the i-quants in the form the CPU kernels load them.
//!
//! The tables are not copied from anywhere: they are read out of the exact
//! reference (`ares_quant::dequantize`) once, by decoding blocks that were
//! built so that every decoded value is a codebook entry. The kernels can
//! therefore not disagree with the reference about a table.

use std::sync::OnceLock;

use ares_core::GgmlType;

/// Thirty-two bytes on a 32-byte boundary: one `ymm` load.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub(super) struct Lanes16(pub [i16; 16]);

/// Thirty-two bytes on a 32-byte boundary, as bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub(super) struct Bytes32(pub [i8; 32]);

/// What the kernels look up.
#[derive(Debug)]
pub(super) struct Tables {
    /// IQ2_S: the 8 magnitudes of each of the 1024 codebook entries, one per
    /// byte, the first value in the low byte.
    pub iq2s_grid: Box<[u64; 1024]>,
    /// IQ2_XXS: the same for its 256 entries.
    pub iq2xxs_grid: Box<[u64; 256]>,
    /// IQ2_XXS: for each 7-bit sign index, byte `j` is 0xFF when value `j`
    /// of the group is negative and 0 otherwise.
    pub iq2xxs_signs: Box<[u64; 128]>,
    /// IQ1_M: the 8 values (-1, 0 or 1 as `i8`) of each of 2048 entries.
    pub iq1_grid: Box<[u64; 2048]>,
    /// IQ3_XXS: the 4 magnitudes of each of the 256 codebook entries, one
    /// per byte, the first value in the low byte. Its signs are IQ2_XXS's
    /// (`iq2xxs_signs`).
    pub iq3xxs_grid: Box<[u32; 256]>,
    /// IQ4_NL and IQ4_XS: the 16 non-linear levels as `i8`, twice, so that
    /// one shuffle looks up 32 nibbles.
    pub iq4nl_levels: Bytes32,
    /// IQ2_S: for a scale byte, `2 s + 1` of its low nibble in lanes 0..8
    /// and of its high nibble in lanes 8..16.
    pub iq2s_scales: Box<[Lanes16; 256]>,
    /// IQ1_M: for the six scale bits of 32 values, `8 (2 s + 1)` of the
    /// first three bits in lanes 0..8 and of the last three in lanes 8..16.
    pub iq1_scales: Box<[Lanes16; 64]>,
    /// IQ1_M: for the four delta signs of 32 values, 1 or -1 in lanes 0, 4,
    /// 8 and 12 (a set bit is -1) and 0 elsewhere.
    pub iq1_deltas: Box<[Lanes16; 16]>,
}

/// The tables, built at first use.
pub(super) fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(build)
}

/// Decodes `blocks` with the reference; a table cannot be built without it.
fn decode(ty: GgmlType, blocks: &[u8]) -> Vec<f32> {
    let values = blocks.len() / ty.block_bytes() as usize * ty.block_values() as usize;
    let mut out = vec![0.0f32; values];
    // The blocks are whole by construction, which is all the reference
    // checks; a failure would leave zeros and the kernels' tests would fail.
    let _ = ares_quant::dequantize(ty, blocks, &mut out);
    out
}

/// Packs eight decoded values into the bytes of a `u64`.
fn pack(values: &[f32], byte: impl Fn(f32) -> u8) -> u64 {
    let mut bytes = [0u8; 8];
    for (b, v) in bytes.iter_mut().zip(values) {
        *b = byte(*v);
    }
    u64::from_le_bytes(bytes)
}

/// The f16 bits of 8.0: with it and a zero scale field an IQ2 group decodes
/// to its magnitudes (`d * (0.5 + 0) * 0.25 = 1`).
const F16_EIGHT: [u8; 2] = 0x4800u16.to_le_bytes();

/// The f16 bits of 4.0 and 1.0: an IQ3_XXS group with a zero scale field
/// decodes to its magnitudes with the first (`d * (0.5 + 0) * 0.5 = 1`), an
/// IQ4_NL block to its levels with the second.
const F16_FOUR: [u8; 2] = 0x4400u16.to_le_bytes();
const F16_ONE: [u8; 2] = 0x3C00u16.to_le_bytes();

fn build() -> Tables {
    // IQ2_S: { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] }, 32 groups a block.
    let mut blocks = vec![0u8; 32 * 82];
    for (n, block) in blocks.as_chunks_mut::<82>().0.iter_mut().enumerate() {
        block[..2].copy_from_slice(&F16_EIGHT);
        for group in 0..32 {
            let entry = 32 * n + group;
            block[2 + group] = entry as u8;
            block[66 + group / 4] |= ((entry >> 8) as u8) << (2 * (group % 4));
        }
    }
    let values = decode(GgmlType::IQ2_S, &blocks);
    let mut iq2s_grid = Box::new([0u64; 1024]);
    for (entry, group) in iq2s_grid.iter_mut().zip(values.as_chunks::<8>().0.iter()) {
        *entry = pack(group, |v| v as u8);
    }

    // IQ2_XXS: { f16 d; 8 x { u8 index[4]; u32 signs and scale } }.
    let mut blocks = vec![0u8; 8 * 66];
    for (n, block) in blocks.as_chunks_mut::<66>().0.iter_mut().enumerate() {
        block[..2].copy_from_slice(&F16_EIGHT);
        for group in 0..32 {
            block[2 + 8 * (group / 4) + group % 4] = (32 * n + group) as u8;
        }
    }
    let values = decode(GgmlType::IQ2_XXS, &blocks);
    let mut iq2xxs_grid = Box::new([0u64; 256]);
    for (entry, group) in iq2xxs_grid.iter_mut().zip(values.as_chunks::<8>().0.iter()) {
        *entry = pack(group, |v| v as u8);
    }
    // The signs: entry 0 of the codebook has no zero, so the sign of each
    // decoded value is the sign bit of its index.
    let mut blocks = vec![0u8; 4 * 66];
    for (n, block) in blocks.as_chunks_mut::<66>().0.iter_mut().enumerate() {
        block[..2].copy_from_slice(&F16_EIGHT);
        for chunk in 0..8 {
            let first = (32 * n + 4 * chunk) as u32;
            let signs = first | (first + 1) << 7 | (first + 2) << 14 | (first + 3) << 21;
            block[6 + 8 * chunk..10 + 8 * chunk].copy_from_slice(&signs.to_le_bytes());
        }
    }
    let values = decode(GgmlType::IQ2_XXS, &blocks);
    let mut iq2xxs_signs = Box::new([0u64; 128]);
    for (entry, group) in iq2xxs_signs.iter_mut().zip(values.as_chunks::<8>().0.iter()) {
        *entry = pack(group, |v| if v < 0.0 { 0xFF } else { 0 });
    }

    // IQ1_M: { u8 qs[32]; u8 qh[16]; u16 scales[4] }. The block's f16 scale
    // is spread over the top nibbles of the four scale words; 1.0 is 0x3C00.
    // With zero scale fields and no delta sign a value is its grid entry
    // plus an eighth.
    let mut blocks = vec![0u8; 64 * 56];
    for (n, block) in blocks.as_chunks_mut::<56>().0.iter_mut().enumerate() {
        block[53] = 0xC0;
        block[55] = 0x30;
        for group in 0..32 {
            let entry = 32 * n + group;
            block[group] = entry as u8;
            block[32 + group / 2] |= ((entry >> 8) as u8) << (4 * (group % 2));
        }
    }
    let values = decode(GgmlType::IQ1_M, &blocks);
    let mut iq1_grid = Box::new([0u64; 2048]);
    for (entry, group) in iq1_grid.iter_mut().zip(values.as_chunks::<8>().0.iter()) {
        *entry = pack(group, |v| (v - 0.125).round() as i8 as u8);
    }

    // IQ3_XXS: { f16 d; u8 index[64]; 8 x u32 signs and scale }, one index
    // per 4 values; zero signs and scales.
    let mut blocks = vec![0u8; 4 * 98];
    for (n, block) in blocks.as_chunks_mut::<98>().0.iter_mut().enumerate() {
        block[..2].copy_from_slice(&F16_FOUR);
        for group in 0..64 {
            block[2 + group] = (64 * n + group) as u8;
        }
    }
    let values = decode(GgmlType::IQ3_XXS, &blocks);
    let mut iq3xxs_grid = Box::new([0u32; 256]);
    for (entry, group) in iq3xxs_grid.iter_mut().zip(values.as_chunks::<4>().0.iter()) {
        *entry = u32::from_le_bytes(group.map(|v| v as u8));
    }

    // IQ4_NL: { f16 d; u8 qs[16] }; the low nibble of byte j is value j.
    let mut block = [0u8; 18];
    block[..2].copy_from_slice(&F16_ONE);
    for (j, byte) in block[2..].iter_mut().enumerate() {
        *byte = j as u8;
    }
    let values = decode(GgmlType::IQ4_NL, &block);
    let mut iq4nl_levels = Bytes32([0; 32]);
    for (j, level) in iq4nl_levels.0.iter_mut().enumerate() {
        *level = values[j % 16] as i8;
    }

    let mut iq2s_scales = Box::new([Lanes16([0; 16]); 256]);
    for (byte, lanes) in iq2s_scales.iter_mut().enumerate() {
        lanes.0[..8].fill(2 * (byte & 15) as i16 + 1);
        lanes.0[8..].fill(2 * (byte >> 4) as i16 + 1);
    }
    let mut iq1_scales = Box::new([Lanes16([0; 16]); 64]);
    for (bits, lanes) in iq1_scales.iter_mut().enumerate() {
        lanes.0[..8].fill(8 * (2 * (bits & 7) as i16 + 1));
        lanes.0[8..].fill(8 * (2 * (bits >> 3) as i16 + 1));
    }
    let mut iq1_deltas = Box::new([Lanes16([0; 16]); 16]);
    for (bits, lanes) in iq1_deltas.iter_mut().enumerate() {
        for group in 0..4 {
            lanes.0[4 * group] = if bits >> group & 1 != 0 { -1 } else { 1 };
        }
    }

    Tables {
        iq2s_grid,
        iq2xxs_grid,
        iq2xxs_signs,
        iq1_grid,
        iq3xxs_grid,
        iq4nl_levels,
        iq2s_scales,
        iq1_scales,
        iq1_deltas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grids_have_the_values_of_the_formats() {
        let t = tables();
        // Magnitudes of the IQ2 codebooks are odd multiples of an eighth of
        // the step, stored times eight: 8, 25, 43 (and 62 in the larger one).
        assert!(
            t.iq2s_grid.iter().flat_map(|e| e.to_le_bytes()).all(|b| [8, 25, 43, 62].contains(&b))
        );
        assert!(
            t.iq2xxs_grid.iter().flat_map(|e| e.to_le_bytes()).all(|b| [8, 25, 43].contains(&b))
        );
        assert_eq!(t.iq2xxs_grid[0], 0x0808_0808_0808_0808);
        // A 7-bit sign index stands for 8 signs with an even number of minus.
        assert_eq!(t.iq2xxs_signs[0], 0);
        assert!(t.iq2xxs_signs.iter().all(|s| (s.count_ones() / 8).is_multiple_of(2)));
        assert!(t.iq1_grid.iter().flat_map(|e| e.to_le_bytes()).all(|b| [0, 1, 0xFF].contains(&b)));
        assert_eq!(t.iq2s_scales[0x3A].0[0], 21);
        assert_eq!(t.iq2s_scales[0x3A].0[15], 7);
        assert_eq!(t.iq1_deltas[0b0101].0[..9], [-1, 0, 0, 0, 1, 0, 0, 0, -1]);
        // IQ3_XXS magnitudes are 4, 12, .. 52 and 62; entry 0 is all 4.
        assert_eq!(t.iq3xxs_grid[0], 0x0404_0404);
        assert!(
            t.iq3xxs_grid
                .iter()
                .flat_map(|e| e.to_le_bytes())
                .all(|b| (4..=62).contains(&b) && b % 2 == 0)
        );
        assert_eq!(t.iq4nl_levels.0[..16], t.iq4nl_levels.0[16..]);
        assert_eq!(
            t.iq4nl_levels.0[..16],
            [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
        );
    }
}
