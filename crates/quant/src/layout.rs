//! The device layouts of weight matrices: the reference of the rearrangement.
//!
//! A device holds a matrix of one of the types below not as it lies in the file
//! but rearranged, once at load, into the order the tensor cores read it, in the
//! same number of bytes. Both backends make the same bytes (CUDA:
//! `kernels/mmr_codes.cuh`, `ares_repack_<type>`; Vulkan: its repack shaders), and
//! this module is what they are tested against, byte for byte.
//!
//! # What is rearranged
//!
//! A matrix of `rows` rows of `cols` values; `cols` must be a multiple of 256
//! ([`has_layout`]). The rows are cut into **tiles** of 16 (the last tile is filled
//! up with rows whose codes and headers are zero bytes, so the layout takes
//! [`rearranged_bytes`]: the file's bytes of `rows` rounded up to 16) and the
//! columns into **steps** of 64 values and **slabs** `kk` of 16 values, four to a
//! step. A warp of 32 lanes multiplies one tile and one slab with one
//! `mma.m16n8k16`; lane `g = lane / 4`, `t = lane % 4` holds in the four registers
//! of its A fragment the values
//!
//! ```text
//! a0: row g,     values 2t, 2t+1        a1: row g + 8, values 2t, 2t+1
//! a2: row g,     values 2t+8, 2t+9      a3: row g + 8, values 2t+8, 2t+9
//! ```
//!
//! of the slab (rows within the tile, values within the slab), each register the
//! even value in its low and the odd one in its high half. The layout stores the
//! codes so that a lane finds exactly these, as little-endian 32-bit words.
//!
//! # The sections, in this order
//!
//! Every type is codes of `LB + HB` bits with a scale, and for some a minimum, per
//! group of 16 or 32 values ([`spec`]).
//!
//! 1. **Low plane**: the low `LB` bits of every code. For each tile, each step,
//!    each lane: `LB` words. The field of register `a_i` of slab `kk` is `LB` bits
//!    for the even value and the same 16 bits higher for the odd one, at
//!    - `LB = 4`: word `kk`, bit `4 i`
//!    - `LB = 2`: word `kk / 2`, bit `8 (kk % 2) + 2 i`
//!    - `LB = 8`: word `2 kk + i / 2`, bit `8 (i % 2)`
//! 2. **High plane** (`HB` = 1 or 2 bits above the low ones; absent for `HB = 0`):
//!    the same way, `HB` words per tile, step and lane; `HB = 1`: word 0, bit
//!    `4 kk + i`; `HB = 2`: as `LB = 2`.
//! 3. **Headers**: what the scales are made of, see [`Header`].
//!
//! So the word of lane `l` of step `s` of tile `m` in a plane of `B` bits starts
//! at word `((m * steps + s) * 32 + l) * B` of the plane, and the planes are
//! `tiles * steps * 32 * B` words long. All sections are whole multiples of 16
//! bytes.
//!
//! # The codes
//!
//! The code of a value is the unsigned number the file stores for it (Q5_K, Q6_K,
//! Q5_0, Q5_1: the low nibble with the high bits above it; Q3_K: the two low bits
//! with the bit of `hmask` above them, so that the value is `scale * (code - 4)`),
//! with two exceptions that make every code unsigned: Q8_0 stores `q + 128`, and
//! so do the int8 scales of Q6_K in the header. IQ4_XS and IQ4_NL store their
//! 4-bit index of the 16 levels unchanged.

use ares_core::GgmlType;

/// What the header section of a layout holds; `g` runs over the 8 pairs of rows
/// `(g, g + 8)` of a tile, in the order of the lanes that read them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Header {
    /// Q4_K, Q5_K: for each tile, super-block of 256 values and row of the tile, the
    /// first 16 bytes of the file's block unchanged (`d`, `dmin`, 12 bytes of 6-bit
    /// scales and minimums).
    K4,
    /// Q6_K, Q3_K, Q2_K, two sections. First, for each tile, super-block and row:
    /// `scale_words` words of scales as in the file's block at `scale_at` (Q6_K:
    /// each byte + 128). Then, for each tile, super-block and `g`: the word
    /// `d(row g) | d(row g + 8) << 16` of the f16 at `d_at`, and with `dmin` a second
    /// word made the same way of the f16 behind it.
    K16 { scale_words: usize, scale_at: usize, d_at: usize, bias: bool, dmin: bool },
    /// Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL: for each tile, step, `g` and each of the
    /// step's two blocks of 32 values: the word `d(row g) | d(row g + 8) << 16` of the
    /// f16 at the start of the file's block, and with `m` a second word made the same
    /// way of the f16 at byte 2.
    B { m: bool },
    /// IQ4_XS: for each tile, super-block and row, the first 8 bytes of the file's
    /// block unchanged (`d`, `scales_h`, `scales_l`).
    X4,
}

/// The parameters of a type's layout.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// Bits of a code in the low and in the high plane.
    pub low_bits: usize,
    pub high_bits: usize,
    pub header: Header,
    /// The code of value `k` of the file's block `b`.
    code: fn(b: &[u8], k: usize) -> u32,
}

fn nibble32(b: &[u8], at: usize, k: usize) -> u32 {
    let byte = u32::from(b[at + (k & 15)]);
    if k & 16 != 0 { byte >> 4 } else { byte & 15 }
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The layout of `ty`, or `None` for a type that keeps the file's layout.
pub fn spec(ty: GgmlType) -> Option<Spec> {
    use GgmlType::*;
    let k16 = |scale_words, scale_at, d_at, bias, dmin| Header::K16 {
        scale_words,
        scale_at,
        d_at,
        bias,
        dmin,
    };
    let (low_bits, high_bits, header, code): (_, _, _, fn(&[u8], usize) -> u32) = match ty {
        Q4_K => (4, 0, Header::K4, |b, k| {
            let byte = u32::from(b[16 + 32 * (k / 64) + (k & 31)]);
            if k & 32 != 0 { byte >> 4 } else { byte & 15 }
        }),
        Q5_K => (4, 1, Header::K4, |b, k| {
            let byte = u32::from(b[48 + 32 * (k / 64) + (k & 31)]);
            let low = if k & 32 != 0 { byte >> 4 } else { byte & 15 };
            low | ((u32::from(b[16 + (k & 31)]) >> (2 * (k / 64) + ((k >> 5) & 1))) & 1) << 4
        }),
        Q6_K => (4, 2, k16(4, 192, 208, true, false), |b, k| {
            let (half, part, l) = (k / 128, (k / 32) % 4, k % 32);
            let byte = u32::from(b[64 * half + l + 32 * (part & 1)]);
            let low = if part < 2 { byte & 15 } else { byte >> 4 };
            low | ((u32::from(b[128 + 32 * half + l]) >> (2 * part)) & 3) << 4
        }),
        Q3_K => (2, 1, k16(3, 96, 108, false, false), |b, k| {
            let (half, step, l) = (k / 128, (k / 32) % 4, k % 32);
            ((u32::from(b[32 + 32 * half + l]) >> (2 * step)) & 3)
                | ((u32::from(b[l]) >> (4 * half + step)) & 1) << 2
        }),
        Q2_K => (2, 0, k16(4, 0, 80, false, true), |b, k| {
            (u32::from(b[16 + 32 * (k / 128) + k % 32]) >> (2 * ((k / 32) % 4))) & 3
        }),
        Q4_0 => (4, 0, Header::B { m: false }, |b, k| nibble32(b, 2, k)),
        Q4_1 => (4, 0, Header::B { m: true }, |b, k| nibble32(b, 4, k)),
        Q5_0 => {
            (4, 1, Header::B { m: false }, |b, k| nibble32(b, 6, k) | ((le32(b, 2) >> k) & 1) << 4)
        }
        Q5_1 => {
            (4, 1, Header::B { m: true }, |b, k| nibble32(b, 8, k) | ((le32(b, 4) >> k) & 1) << 4)
        }
        Q8_0 => (8, 0, Header::B { m: false }, |b, k| u32::from(b[2 + k] ^ 0x80)),
        IQ4_XS => (4, 0, Header::X4, |b, k| nibble32(b, 8 + 16 * (k / 32), k % 32)),
        IQ4_NL => (4, 0, Header::B { m: false }, |b, k| nibble32(b, 2, k)),
        _ => return None,
    };
    Some(Spec { low_bits, high_bits, header, code })
}

/// Whether a matrix of `ty` with rows of `cols` values has a device layout.
pub fn has_layout(ty: GgmlType, cols: usize) -> bool {
    spec(ty).is_some() && cols != 0 && cols.is_multiple_of(256)
}

/// The bytes of the device layout of a matrix of `rows` rows of `cols` values: those
/// of the file's layout with the rows rounded up to a multiple of 16. `None` when
/// the matrix has no device layout.
pub fn rearranged_bytes(ty: GgmlType, rows: usize, cols: usize) -> Option<usize> {
    let (values, bytes) = ty.block();
    has_layout(ty, cols)
        .then(|| rows.next_multiple_of(16) * (cols / values as usize) * bytes as usize)
}

/// Of the words a lane holds of a plane of `bits` bits per code for one step: the
/// word and the bit of the even value of register `a_i` of slab `kk`.
fn plane_word(bits: usize, kk: usize, i: usize) -> usize {
    match bits {
        8 => 2 * kk + i / 2,
        4 => kk,
        2 => kk / 2,
        _ => 0,
    }
}

fn plane_shift(bits: usize, kk: usize, i: usize) -> usize {
    match bits {
        8 => 8 * (i % 2),
        4 => 4 * i,
        2 => 8 * (kk % 2) + 2 * i,
        _ => 4 * kk + i,
    }
}

/// Why [`rearrange`] refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    #[error("{ty} with rows of {cols} values has no device layout")]
    NoLayout { ty: GgmlType, cols: usize },
    #[error("{what} is {have} bytes, the matrix takes {want}")]
    Length { what: &'static str, have: usize, want: usize },
}

/// Rearranges the matrix `file` (`rows` rows of `cols` values of `ty` as in the
/// model file) into its device layout in `out` ([`rearranged_bytes`] bytes).
pub fn rearrange(
    ty: GgmlType,
    rows: usize,
    cols: usize,
    file: &[u8],
    out: &mut [u8],
) -> Result<(), LayoutError> {
    let (Some(spec), Some(want)) = (spec(ty), rearranged_bytes(ty, rows, cols)) else {
        return Err(LayoutError::NoLayout { ty, cols });
    };
    let (block_values, block_bytes) = (ty.block().0 as usize, ty.block().1 as usize);
    let row_bytes = cols / block_values * block_bytes;
    if file.len() != rows * row_bytes {
        return Err(LayoutError::Length {
            what: "the file's layout",
            have: file.len(),
            want: rows * row_bytes,
        });
    }
    if out.len() != want {
        return Err(LayoutError::Length { what: "the device layout", have: out.len(), want });
    }
    let (tiles, steps, sbs) = (rows.div_ceil(16), cols / 64, cols / 256);
    // The file's block `block` of `row`; rows past the end of the matrix are zeros.
    let block = |row: usize, block: usize| -> Option<&[u8]> {
        (row < rows).then(|| &file[row * row_bytes + block * block_bytes..][..block_bytes])
    };
    let u16_at = |row: usize, b: usize, at: usize| -> u32 {
        block(row, b).map_or(0, |b| u32::from(u16::from_le_bytes([b[at], b[at + 1]])))
    };
    let u32_at = |row: usize, b: usize, at: usize| -> u32 {
        u16_at(row, b, at) | u16_at(row, b, at + 2) << 16
    };

    let mut words = out.as_chunks_mut::<4>().0.iter_mut();
    let mut put = |word: u32| *words.next().expect("the layout's bytes") = word.to_le_bytes();

    // The planes.
    for (bits, high) in [(spec.low_bits, false), (spec.high_bits, true)] {
        for unit in 0..tiles * steps * 32 * bits {
            let (word, lane) = (unit % bits, (unit / bits) % 32);
            let (step, tile) = ((unit / bits / 32) % steps, unit / bits / 32 / steps);
            let (g, t) = (lane / 4, lane % 4);
            let mut out = 0u32;
            for kk in 0..4 {
                for i in 0..4 {
                    let row = tile * 16 + g + 8 * (i & 1);
                    if plane_word(bits, kk, i) != word {
                        continue;
                    }
                    for e in 0..2 {
                        let k = step * 64 + kk * 16 + 2 * t + 8 * (i >> 1) + e;
                        let Some(b) = block(row, k / block_values) else { continue };
                        let code = (spec.code)(b, k % block_values);
                        let field = if high {
                            code >> spec.low_bits
                        } else {
                            code & ((1 << spec.low_bits) - 1)
                        };
                        out |= field << (plane_shift(bits, kk, i) + 16 * e);
                    }
                }
            }
            put(out);
        }
    }

    // The headers.
    match spec.header {
        Header::K4 | Header::X4 => {
            let header_words = if spec.header == Header::K4 { 4 } else { 2 };
            for tile in 0..tiles {
                for sb in 0..sbs {
                    for r in 0..16 {
                        for w in 0..header_words {
                            put(u32_at(tile * 16 + r, sb, 4 * w));
                        }
                    }
                }
            }
        }
        Header::K16 { scale_words, scale_at, d_at, bias, dmin } => {
            for tile in 0..tiles {
                for sb in 0..sbs {
                    for r in 0..16 {
                        for w in 0..scale_words {
                            let scales = u32_at(tile * 16 + r, sb, scale_at + 4 * w);
                            put(if bias { scales ^ 0x8080_8080 } else { scales });
                        }
                    }
                }
            }
            for tile in 0..tiles {
                for sb in 0..sbs {
                    for g in 0..8 {
                        for w in 0..1 + usize::from(dmin) {
                            let at = d_at + 2 * w;
                            put(u16_at(tile * 16 + g, sb, at)
                                | u16_at(tile * 16 + g + 8, sb, at) << 16);
                        }
                    }
                }
            }
        }
        Header::B { m } => {
            for tile in 0..tiles {
                for step in 0..steps {
                    for g in 0..8 {
                        for sub in 0..2 {
                            for w in 0..1 + usize::from(m) {
                                let (b, at) = (2 * step + sub, 2 * w);
                                put(u16_at(tile * 16 + g, b, at)
                                    | u16_at(tile * 16 + g + 8, b, at) << 16);
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert!(words.next().is_none(), "the sections fill the layout");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TYPES: [GgmlType; 12] = [
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q3_K,
        GgmlType::Q2_K,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::IQ4_XS,
        GgmlType::IQ4_NL,
    ];

    fn random(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u8
            })
            .collect()
    }

    /// The code of value `k` of `row`, read back from the planes of a layout.
    fn code_in_layout(
        spec: &Spec,
        layout: &[u8],
        rows: usize,
        cols: usize,
        row: usize,
        k: usize,
    ) -> u32 {
        let (tiles, steps) = (rows.div_ceil(16), cols / 64);
        let (tile, r) = (row / 16, row % 16);
        let (step, kk, v) = (k / 64, (k % 64) / 16, k % 16);
        // Register a_i holds row g + 8 (i & 1), values 2t + 8 (i >> 1) and the next.
        let (g, i, t, e) = (r % 8, r / 8 + 2 * (v / 8), (v % 8) / 2, v % 2);
        let word = |at: usize| u32::from_le_bytes(layout[4 * at..4 * at + 4].try_into().unwrap());
        let unit = (tile * steps + step) * 32 + 4 * g + t;
        let field = |base: usize, bits: usize| {
            if bits == 0 {
                return 0;
            }
            let w = word(base + unit * bits + plane_word(bits, kk, i));
            (w >> (plane_shift(bits, kk, i) + 16 * e)) & ((1 << bits) - 1)
        };
        let low = field(0, spec.low_bits);
        let high = field(tiles * steps * 32 * spec.low_bits, spec.high_bits);
        low | high << spec.low_bits
    }

    #[test]
    fn a_layout_takes_the_bytes_of_the_file() {
        for ty in TYPES {
            assert_eq!(rearranged_bytes(ty, 32, 512), ty.bytes_for(32 * 512).map(|b| b as usize));
            assert_eq!(rearranged_bytes(ty, 17, 256), rearranged_bytes(ty, 32, 256));
            assert_eq!(rearranged_bytes(ty, 16, 96), None);
        }
        assert_eq!(rearranged_bytes(GgmlType::IQ3_S, 16, 256), None);
        assert!(rearrange(GgmlType::IQ3_S, 16, 256, &[0; 16 * 110], &mut [0; 16 * 110]).is_err());
        assert!(rearrange(GgmlType::Q4_K, 16, 256, &[0; 5], &mut [0; 16 * 144]).is_err());
        assert!(rearrange(GgmlType::Q4_K, 16, 256, &[0; 16 * 144], &mut [0; 5]).is_err());
    }

    /// Every code of the file is found in the planes where the specification says,
    /// the rows that fill the last tile are zero, and the header sections hold the
    /// file's header bytes of the right rows.
    #[test]
    fn the_planes_hold_every_code_at_its_place() {
        for ty in TYPES {
            let spec = spec(ty).unwrap();
            let (bv, bb) = (ty.block().0 as usize, ty.block().1 as usize);
            for (rows, cols) in [(1usize, 256usize), (16, 256), (21, 512), (48, 768)] {
                let row_bytes = cols / bv * bb;
                let file = random(rows * row_bytes, 0x9e3779b97f4a7c15 ^ (rows * cols) as u64);
                let mut layout = vec![0xAAu8; rearranged_bytes(ty, rows, cols).unwrap()];
                rearrange(ty, rows, cols, &file, &mut layout).unwrap();
                for row in 0..rows.next_multiple_of(16) {
                    for k in 0..cols {
                        let want = if row < rows {
                            let b = &file[row * row_bytes + (k / bv) * bb..][..bb];
                            (spec.code)(b, k % bv)
                        } else {
                            0
                        };
                        let have = code_in_layout(&spec, &layout, rows, cols, row, k);
                        assert_eq!(have, want, "{ty} {rows} x {cols}: row {row}, value {k}");
                    }
                }
            }
        }
    }

    /// The headers of Q4_K, the layout both backends had first: the 16 bytes of each
    /// row and super-block, tile after tile, behind the codes.
    #[test]
    fn the_headers_of_q4_k_are_the_files() {
        let (rows, cols) = (21usize, 512usize);
        let file = random(rows * 2 * 144, 7);
        let mut layout = vec![0u8; rearranged_bytes(GgmlType::Q4_K, rows, cols).unwrap()];
        rearrange(GgmlType::Q4_K, rows, cols, &file, &mut layout).unwrap();
        let codes = 32 * cols / 2;
        for row in 0..32 {
            for sb in 0..2 {
                let at = codes + (((row / 16) * 2 + sb) * 16 + row % 16) * 16;
                let want: &[u8] =
                    if row < rows { &file[(row * 2 + sb) * 144..][..16] } else { &[0; 16] };
                assert_eq!(&layout[at..at + 16], want, "row {row}, super-block {sb}");
            }
        }
    }
}
