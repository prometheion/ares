// Q5_1 for the f16 prefill kernels: 32 values in 24 bytes (4-byte aligned): f16 d, f16 m,
// u32 qh, 16 bytes of nibbles; value = code * d + m with the 5-bit code of Q5_0.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 24;

uint spread_to_bit4(uint x) {
    return (((x & 0xFu) * 0x00204081u) & 0x01010101u) << 4;
}

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u) * BLOCK_BYTES;
    const uint head = ld4(blk);
    const f16vec2 d = h2(half_of(head));
    const f16vec2 m = h2(half_of(head >> 16));
    const uint qh = ld4(blk + 4);
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uint q = ld4(blk + 8 + 4 * i);
        const uint lo = (q & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (4 * i));
        const uint hi = ((q >> 4) & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (16 + 4 * i));
        o[2 * i] = pack2(fma(u8_01(lo), d, m));
        o[2 * i + 1] = pack2(fma(u8_23(lo), d, m));
        o[8 + 2 * i] = pack2(fma(u8_01(hi), d, m));
        o[8 + 2 * i + 1] = pack2(fma(u8_23(hi), d, m));
    }
}
