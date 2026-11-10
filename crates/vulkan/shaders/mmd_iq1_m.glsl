// IQ1_M for the f16 prefill kernels: 256 values in 56 bytes (4-byte aligned): u8 qs[32]
// (low 8 bits of 11-bit codebook indices), u8 qh[16] at 32 (two bytes per sub-block ib of 32
// values: each byte holds the high index bits of two groups (bits 0..2 and 4..6) and, in
// bits 3 and 7, the sign of their offsets), u16 scales[4] at 48 (3-bit scales, two per group
// pair: sub-block ib uses scales[ib / 2] at bit 6 (ib % 2) and 3 more bits up; the top
// nibbles of the four words are the bits of the f16 d). A group is 8 values in {-1, 0, 1};
// value = dl (grid + delta) with a delta per group and dl per pair of groups. A unit is a
// sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 56;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uint qs = ld4(blk + 4 * ib);
    const uint qhw = ld2(blk + 32 + 2 * ib);
    const uint h0 = qhw & 0xFFu;
    const uint h1 = qhw >> 8;
    const uint s01 = ld4(blk + 48);
    const uint s23 = ld4(blk + 52);
    // The f16 scale d: one nibble from the top of each 16-bit word.
    const uint dbits = ((s01 & 0xFFFFu) >> 12) | (((s01 >> 16) >> 8) & 0xF0u) |
                       (((s23 & 0xFFFFu) >> 4) & 0xF00u) | ((s23 >> 16) & 0xF000u);
    const float d = half_of(dbits);
    const uint pair = (ib >> 1) < 2 ? s01 : s23;
    const uint sc = ((ib >> 1) & 1u) == 0u ? (pair & 0xFFFFu) : (pair >> 16);
    const uint shift = 6 * (ib & 1u);
    const float dl1 = d * float(2 * ((sc >> shift) & 7u) + 1);
    const float dl2 = d * float(2 * ((sc >> (shift + 3)) & 7u) + 1);
    const uint idx[4] = uint[4](
        (qs & 0xFFu) | ((h0 << 8) & 0x700u), ((qs >> 8) & 0xFFu) | ((h0 << 4) & 0x700u),
        ((qs >> 16) & 0xFFu) | ((h1 << 8) & 0x700u), (qs >> 24) | ((h1 << 4) & 0x700u));
    const float delta[4] = float[4]((h0 & 0x08u) != 0u ? -0.125 : 0.125,
                                    (h0 & 0x80u) != 0u ? -0.125 : 0.125,
                                    (h1 & 0x08u) != 0u ? -0.125 : 0.125,
                                    (h1 & 0x80u) != 0u ? -0.125 : 0.125);
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint g = (ld4(pc.tables + TBL_IQ1S + 4 * uint64_t(idx[l] >> 1)) >> ((idx[l] & 1u) * 16u)) & 0xFFFFu;
        const float dl = l < 2 ? dl1 : dl2;
        const f16vec2 s = h2(dl);
        const f16vec2 z = h2((delta[l] - 1.0) * dl);
        const uint lo = spread_pairs(g);
        const uint hi = spread_pairs(g >> 8);
        o[4 * l] = pack2(fma(u8_01(lo), s, z));
        o[4 * l + 1] = pack2(fma(u8_23(lo), s, z));
        o[4 * l + 2] = pack2(fma(u8_01(hi), s, z));
        o[4 * l + 3] = pack2(fma(u8_23(hi), s, z));
    }
}
