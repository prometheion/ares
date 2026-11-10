// IQ2_XS for the f16 prefill kernels: 256 values in 74 bytes (2-byte aligned): f16 d, u16
// qs[32] at 2 (a 9-bit codebook index of 512 groups of 8 magnitudes and a 7-bit sign index
// above it), u8 scales[8] at 66 (two 4-bit scales per sub-block: the first 16 values, the
// second 16). Sub-block ib: qs[4 ib + l], l = 0..3, one group of 8 values each.
// value = d (0.5 + scale) 0.25 magnitude sign. A unit is a sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 74;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uvec2 qs = ld8_a2(blk + 2 + 8 * ib);
    const uint sc = ld1(blk + 66 + ib);
    uint q[8];
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint word = l < 2 ? qs.x : qs.y;
        const uint c = (word >> (16 * (l & 1))) & 0xFFFFu;
        const uvec2 g = ld8(pc.tables + TBL_IQ2XS + 8 * uint64_t(c & 511u));
        const uint spread = ksigns(c >> 9) * 0x01010101u;
        q[2 * l] = apply_sign4(g.x, sign_mask4(spread, 0x08040201u));
        q[2 * l + 1] = apply_sign4(g.y, sign_mask4(spread, 0x80402010u));
    }
    const float d = half_of(ld2(blk)) * 0.125;
    emit_signed(q, h2(d * float(2 * (sc & 0xFu) + 1)), h2(d * float(2 * (sc >> 4) + 1)), o);
}
