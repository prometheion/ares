// TQ2_0 for the f16 prefill kernels: 256 values in 66 bytes (2-byte aligned): u8 qs[64], f16
// d at 64. Four 2-bit codes per byte (0, 1, 2 for -1, 0, +1; a 3 decodes to +2, as the
// reference does), in planes: in each run of 32 bytes the codes at shift 2 l of byte m are
// value 128 run + 32 l + m. A unit is a plane.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 66;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint within = u & 7u;
    const f16vec2 d = h2(half_of(ld2(blk + 64)));
    uint q[8];
    ld32_any(blk + 32 * (within >> 2), q);
    const uint l = within & 3u;
    [[unroll]] for (uint i = 0; i < 8; i++) {
        const uint c = (q[i] >> (2 * l)) & 0x03030303u;
        o[2 * i] = pack2((u8_01(c) - f16vec2(1.0hf)) * d);
        o[2 * i + 1] = pack2((u8_23(c) - f16vec2(1.0hf)) * d);
    }
}
