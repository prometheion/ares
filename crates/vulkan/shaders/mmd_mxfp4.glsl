// MXFP4 for the f16 prefill kernels: 32 values in 17 bytes (any alignment): u8 e (an E8M0
// scale), 16 bytes of nibbles laid out as Q4_0, each a level of the 4-bit float table
// (doubled E2M1 levels, so the scale is 2^(e - 127) / 2). A unit is a block. A scale
// outside the range of an f16 (2^-24 .. 65504) is not represented.
#define MD_TABLE16 ares_kvalues_mxfp4
#include "mmd_common.glsl"
#include "fp4_scales.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 17;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u) * BLOCK_BYTES;
    const f16vec2 d = h2(e8m0_half(ld1(blk)));
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uint q = ld4_u(blk + 1 + 4 * i);
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
