// Q4_0 for the f16 prefill kernels: 32 values in 18 bytes (2-byte aligned): f16 d, 16
// bytes of nibbles (low nibbles are values 0..15, high 16..31); value = (nibble - 8) * d.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 18;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u) * BLOCK_BYTES;
    const f16vec2 d = h2(half_of(ld2(blk)));
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uint q = ld4_a2(blk + 2 + 4 * i);
        const uint lo = q & 0x0F0F0F0Fu;
        const uint hi = (q >> 4) & 0x0F0F0F0Fu;
        o[2 * i] = pack2((u8_01(lo) - f16vec2(8.0hf)) * d);
        o[2 * i + 1] = pack2((u8_23(lo) - f16vec2(8.0hf)) * d);
        o[8 + 2 * i] = pack2((u8_01(hi) - f16vec2(8.0hf)) * d);
        o[8 + 2 * i + 1] = pack2((u8_23(hi) - f16vec2(8.0hf)) * d);
    }
}
