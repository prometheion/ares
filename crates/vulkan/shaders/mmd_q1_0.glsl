// Q1_0 for the f16 prefill kernels: 128 values in 18 bytes (2-byte aligned): f16 d, 16 bytes
// of bits (bit j is the sign of value j: set is +d, clear -d). A unit is a word of bits.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 128;
const uint BLOCK_BYTES = 18;

// The 4 low bits of x as four bytes of 0 or 1.
uint spread_bits(uint x) {
    return ((x & 0xFu) * 0x00204081u) & 0x01010101u;
}

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 2) * BLOCK_BYTES;
    const f16vec2 d = h2(half_of(ld2(blk)));
    const f16vec2 d2 = d + d;
    const uint bits = ld4_a2(blk + 2 + 4 * (u & 3u));
    [[unroll]] for (uint i = 0; i < 8; i++) {
        const uint c = spread_bits(bits >> (4 * i));
        o[2 * i] = pack2(fma(u8_01(c), d2, -d));
        o[2 * i + 1] = pack2(fma(u8_23(c), d2, -d));
    }
}
