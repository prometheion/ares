// IQ2_S for the f16 prefill kernels: 256 values in 82 bytes (2-byte aligned): f16 d, u8
// qs[64] at 2 (the first 32: low 8 bits of 10-bit codebook indices of 1024 groups of 8
// magnitudes; the last 32: sign bytes), u8 qh[8] at 66 (the top 2 bits of the indices of
// sub-block ib), u8 scales[8] at 74 (as IQ2_XS). Sub-block ib: index qs[4 ib + l] |
// (qh[ib] << (8 - 2 l)) & 0x300, signs qs[32 + 4 ib + l]. A unit is a sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 82;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uint qs = ld4_a2(blk + 2 + 4 * ib);
    const uint signs = ld4_a2(blk + 34 + 4 * ib);
    const uint qh = ld1(blk + 66 + ib);
    const uint sc = ld1(blk + 74 + ib);
    uint q[8];
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint idx = ((qs >> (8 * l)) & 0xFFu) | ((qh << (8 - 2 * l)) & 0x300u);
        const uvec2 g = ld8(pc.tables + TBL_IQ2S + 8 * uint64_t(idx));
        const uint spread = ((signs >> (8 * l)) & 0xFFu) * 0x01010101u;
        q[2 * l] = apply_sign4(g.x, sign_mask4(spread, 0x08040201u));
        q[2 * l + 1] = apply_sign4(g.y, sign_mask4(spread, 0x80402010u));
    }
    const float d = half_of(ld2(blk)) * 0.125;
    emit_signed(q, h2(d * float(2 * (sc & 0xFu) + 1)), h2(d * float(2 * (sc >> 4) + 1)), o);
}
