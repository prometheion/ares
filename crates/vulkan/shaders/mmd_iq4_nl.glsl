// IQ4_NL for the f16 prefill kernels: 32 values in 18 bytes (2-byte aligned): f16 d, 16
// bytes of nibbles laid out as Q4_0 (low nibbles are values 0..15, high 16..31), each a
// level of a table of 16 signed bytes; value = d * level. A unit is a block.
#define MD_TABLE16 ares_kvalues_iq4nl
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 18;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u) * BLOCK_BYTES;
    const f16vec2 d = h2(half_of(ld2(blk)));
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uint q = ld4_a2(blk + 2 + 4 * i);
        [[unroll]] for (uint k = 0; k < 4; k += 2) {
            const uint b0 = q >> (8 * k);
            const uint b1 = q >> (8 * (k + 1));
            o[2 * i + k / 2] = pack2(f16vec2(float16_t(lut16(b0 & 15u)),
                                             float16_t(lut16(b1 & 15u))) * d);
            o[8 + 2 * i + k / 2] = pack2(f16vec2(float16_t(lut16((b0 >> 4) & 15u)),
                                                 float16_t(lut16((b1 >> 4) & 15u))) * d);
        }
    }
}
