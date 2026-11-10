// IQ3_S for the f16 prefill kernels: 256 values in 110 bytes (2-byte aligned): f16 d, u8
// qs[64] at 2 (low 8 bits of 9-bit codebook indices of 512 groups of 4), u8 qh[8] at 66 (the
// ninth bits: byte ib for sub-block ib), u8 signs[32] at 74 (one sign bit per value), u8
// scales[4] at 106 (4 bits per sub-block). Sub-block ib: indices qs[8 ib + 2l + {0,1}] | bit
// of qh[ib], signs[4 ib + l] for l = 0..3; value = d (1 + 2 scale) magnitude sign. A unit is
// a sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 110;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uvec2 qs = ld8_a2(blk + 2 + 8 * ib);
    const uint qh = ld1(blk + 66 + ib);
    const uint signs = ld4_a2(blk + 74 + 4 * ib);
    const uint sc = ld1(blk + 106 + (ib >> 1));
    uint q[8];
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint word = l < 2 ? qs.x : qs.y;
        const uint i1 = ((word >> (16 * (l & 1))) & 0xFFu) | ((qh << (8 - 2 * l)) & 256u);
        const uint i2 = ((word >> (16 * (l & 1) + 8)) & 0xFFu) | ((qh << (7 - 2 * l)) & 256u);
        const uint spread = ((signs >> (8 * l)) & 0xFFu) * 0x01010101u;
        q[2 * l] = apply_sign4(ld4(pc.tables + TBL_IQ3S + 4 * uint64_t(i1)),
                               sign_mask4(spread, 0x08040201u));
        q[2 * l + 1] = apply_sign4(ld4(pc.tables + TBL_IQ3S + 4 * uint64_t(i2)),
                                   sign_mask4(spread, 0x80402010u));
    }
    const f16vec2 s = h2(half_of(ld2(blk)) * float(1 + 2 * ((sc >> (4 * (ib & 1u))) & 0xFu)));
    emit_signed(q, s, s, o);
}
