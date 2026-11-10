//! Walking a stream of blocks without bounds checks or allocation.

use ares_core::GgmlType;

use crate::half::f16_to_f32;

/// Bytes of one block of `ty`, as the `usize` const generics need.
///
/// Taking the geometry from `ares-core` instead of repeating it here means
/// the decoders cannot disagree with the table the rest of the engine uses.
pub(crate) const fn block_bytes(ty: GgmlType) -> usize {
    ty.block_bytes() as usize
}

/// Values of one block of `ty`.
pub(crate) const fn block_values(ty: GgmlType) -> usize {
    ty.block_values() as usize
}

/// Calls `f` for every block of `src` together with the `VALUES` outputs it
/// decodes to. The callers have checked that the lengths fit; a stray tail is
/// ignored. Typed as arrays so that fixed offsets into a block compile to plain
/// loads.
#[inline]
pub(crate) fn decode_blocks<const BYTES: usize, const VALUES: usize>(
    src: &[u8],
    dst: &mut [f32],
    mut f: impl FnMut(&[u8; BYTES], &mut [f32; VALUES]),
) {
    let (blocks, _) = src.as_chunks::<BYTES>();
    let (out, _) = dst.as_chunks_mut::<VALUES>();
    for (block, values) in blocks.iter().zip(out) {
        f(block, values);
    }
}

/// Calls `f` for every `VALUES` inputs together with the `BYTES` block they
/// encode to; the mirror of [`decode_blocks`].
#[inline]
pub(crate) fn encode_blocks<const BYTES: usize, const VALUES: usize>(
    src: &[f32],
    dst: &mut [u8],
    mut f: impl FnMut(&[f32; VALUES], &mut [u8; BYTES]),
) {
    let (values, _) = src.as_chunks::<VALUES>();
    let (blocks, _) = dst.as_chunks_mut::<BYTES>();
    for (values, block) in values.iter().zip(blocks) {
        f(values, block);
    }
}

/// The little-endian `u16` at byte `at` of a block.
#[inline]
pub(crate) fn u16_at<const N: usize>(block: &[u8; N], at: usize) -> u16 {
    u16::from_le_bytes([block[at], block[at + 1]])
}

/// The little-endian `u32` at byte `at` of a block.
#[inline]
pub(crate) fn u32_at<const N: usize>(block: &[u8; N], at: usize) -> u32 {
    u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
}

/// The f16 scale stored at byte `at` of a block, widened.
#[inline]
pub(crate) fn f16_at<const N: usize>(block: &[u8; N], at: usize) -> f32 {
    f16_to_f32(u16_at(block, at))
}
