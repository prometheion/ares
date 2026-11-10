// IQ3_XXS for the f16 prefill kernels: 256 values in 98 bytes (2-byte aligned): f16 d, u8
// grid[64] at 2 (indices into a codebook of 256 groups of 4 magnitudes), u32 aux[8] at 66
// (per sub-block: four 7-bit sign indices, a 4-bit scale on top). Sub-block ib (8 groups of
// 4): grid[8 ib + 0 .. 7]; group pair l (8 values) takes grid[8 ib + 2l], grid[8 ib + 2l + 1]
// and the sign byte of 7 bits (aux >> 7l). value = d (0.5 + scale) 0.5 magnitude sign. A
// unit is a sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 98;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uvec2 idx = ld8_a2(blk + 2 + 8 * ib);
    const uint aux = ld4_a2(blk + 66 + 4 * ib);
    uint q[8];
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint word = l < 2 ? idx.x : idx.y;
        const uint i1 = (word >> (16 * (l & 1))) & 0xFFu;
        const uint i2 = (word >> (16 * (l & 1) + 8)) & 0xFFu;
        const uint signs = ksigns((aux >> (7 * l)) & 127u) * 0x01010101u;
        q[2 * l] = apply_sign4(ld4(pc.tables + TBL_IQ3XXS + 4 * uint64_t(i1)),
                               sign_mask4(signs, 0x08040201u));
        q[2 * l + 1] = apply_sign4(ld4(pc.tables + TBL_IQ3XXS + 4 * uint64_t(i2)),
                                   sign_mask4(signs, 0x80402010u));
    }
    const f16vec2 s = h2(half_of(ld2(blk)) * float(2 * (aux >> 28) + 1) * 0.25);
    emit_signed(q, s, s, o);
}
