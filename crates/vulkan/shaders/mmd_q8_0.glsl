// Q8_0 for the f16 prefill kernels: 32 values in 34 bytes (2-byte aligned): f16 d, 32
// signed bytes; value = byte * d. A unit is a block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 34;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u) * BLOCK_BYTES;
    const f16vec2 d = h2(half_of(ld2(blk)));
    uint q[8];
    ld32_any(blk + 2, q);
    [[unroll]] for (uint i = 0; i < 8; i++) {
        o[2 * i] = pack2(s8_01(q[i]) * d);
        o[2 * i + 1] = pack2(s8_23(q[i]) * d);
    }
}
