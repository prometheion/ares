// Q5_0 for the f16 prefill kernels: 32 values in 22 bytes (2-byte aligned): f16 d, u32 qh
// (bit j is the fifth bit of value j), 16 bytes of nibbles as Q4_0; value = (code - 16) * d.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 22;

// The low 4 bits of x as the bit 4 of four bytes (0x10 where the bit is set).
uint spread_to_bit4(uint x) {
    return (((x & 0xFu) * 0x00204081u) & 0x01010101u) << 4;
}

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u) * BLOCK_BYTES;
    const f16vec2 d = h2(half_of(ld2(blk)));
    const uint qh = ld4_a2(blk + 2);
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uint q = ld4_a2(blk + 6 + 4 * i);
        const uint lo = (q & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (4 * i));
        const uint hi = ((q >> 4) & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (16 + 4 * i));
        o[2 * i] = pack2((u8_01(lo) - f16vec2(16.0hf)) * d);
        o[2 * i + 1] = pack2((u8_23(lo) - f16vec2(16.0hf)) * d);
        o[8 + 2 * i] = pack2((u8_01(hi) - f16vec2(16.0hf)) * d);
        o[8 + 2 * i + 1] = pack2((u8_23(hi) - f16vec2(16.0hf)) * d);
    }
}
