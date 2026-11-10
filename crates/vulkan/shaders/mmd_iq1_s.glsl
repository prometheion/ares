// IQ1_S for the f16 prefill kernels: 256 values in 50 bytes (2-byte aligned): f16 d, u8
// qs[32] at 2 (low 8 bits of 11-bit codebook indices), u16 qh[8] at 34 (per sub-block ib of
// 32 values: three more index bits for each of its 4 groups (bits 3l .. 3l+2), the scale in
// bits 12..14 and the sign of the offset in bit 15). A group is 8 values in {-1, 0, 1},
// kept in the codebook as 2 bits per value (value + 1). value = d (2 scale + 1) (grid +
// delta), delta = +-0.125 for the sub-block. A unit is a sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 50;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uint qs = ld4_a2(blk + 2 + 4 * ib);
    const uint qh = ld2(blk + 34 + 2 * ib);
    const float dl = half_of(ld2(blk)) * float(2 * ((qh >> 12) & 7u) + 1);
    const float delta = (qh & 0x8000u) != 0u ? -0.125 : 0.125;
    const f16vec2 s = h2(dl);
    const f16vec2 z = h2((delta - 1.0) * dl);
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint index = ((qs >> (8 * l)) & 0xFFu) | (((qh >> (3 * l)) & 7u) << 8);
        const uint g = (ld4(pc.tables + TBL_IQ1S + 4 * uint64_t(index >> 1)) >> ((index & 1u) * 16u)) & 0xFFFFu;
        const uint lo = spread_pairs(g);
        const uint hi = spread_pairs(g >> 8);
        o[4 * l] = pack2(fma(u8_01(lo), s, z));
        o[4 * l + 1] = pack2(fma(u8_23(lo), s, z));
        o[4 * l + 2] = pack2(fma(u8_01(hi), s, z));
        o[4 * l + 3] = pack2(fma(u8_23(hi), s, z));
    }
}
