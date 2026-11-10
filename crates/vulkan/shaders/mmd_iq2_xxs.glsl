// IQ2_XXS for the f16 prefill kernels: 256 values in 66 bytes (2-byte aligned): f16 d; per
// sub-block ib of 32 values u8 idx[4] (indices into a codebook of 256 groups of 8
// magnitudes), then a u32 whose low 28 bits are four 7-bit sign indices and whose top 4
// bits are the scale; value = d (0.5 + scale) 0.25 magnitude sign. A unit is a sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 66;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uint idx = ld4_a2(blk + 2 + 8 * ib);
    const uint aux = ld4_a2(blk + 6 + 8 * ib);
    uint q[8];
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uvec2 g = ld8(pc.tables + TBL_IQ2XXS + 8 * uint64_t((idx >> (8 * l)) & 0xFFu));
        const uint spread = ksigns((aux >> (7 * l)) & 127u) * 0x01010101u;
        q[2 * l] = apply_sign4(g.x, sign_mask4(spread, 0x08040201u));
        q[2 * l + 1] = apply_sign4(g.y, sign_mask4(spread, 0x80402010u));
    }
    const f16vec2 s = h2(half_of(ld2(blk)) * float(2 * (aux >> 28) + 1) * 0.125);
    emit_signed(q, s, s, o);
}
