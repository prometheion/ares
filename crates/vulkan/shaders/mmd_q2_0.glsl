// Q2_0 for the f16 prefill kernels: 64 values in 18 bytes (2-byte aligned): f16 d, 16
// bytes with four 2-bit codes each, the first value in the low bits; value = (code - 1) d.
// A unit is half a block (8 bytes).
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 64;
const uint BLOCK_BYTES = 18;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 1) * BLOCK_BYTES;
    const f16vec2 d = h2(half_of(ld2(blk)));
    [[unroll]] for (uint i = 0; i < 2; i++) {
        const uint word = ld4_a2(blk + 2 + 8 * (u & 1u) + 4 * i);
        [[unroll]] for (uint b = 0; b < 4; b++) {
            const uint c = spread_pairs(word >> (8 * b));
            o[8 * i + 2 * b] = pack2((u8_01(c) - f16vec2(1.0hf)) * d);
            o[8 * i + 2 * b + 1] = pack2((u8_23(c) - f16vec2(1.0hf)) * d);
        }
    }
}
