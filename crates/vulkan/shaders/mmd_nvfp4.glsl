// NVFP4 for the f16 prefill kernels: 64 values in 36 bytes (4-byte aligned): u8 d[4] (a
// UE4M3 scale per sub-block of 16 values), u8 qs[32]: sub-block s has 8 bytes of nibbles,
// the low nibbles are its first 8 values and the high ones the next 8, each a level of the
// 4-bit float table of MXFP4. A unit is two sub-blocks (half of a block).
#define MD_TABLE16 ares_kvalues_mxfp4
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 64;
const uint BLOCK_BYTES = 36;

// The UE4M3 byte (4 exponent bits with bias 7, 3 mantissa bits) as a float, halved for the
// doubled table. 0 and the NaN code 0x7F give 0.
float ue4m3_half(uint x) {
    if (x == 0u || x == 0x7Fu) {
        return 0.0;
    }
    const uint e = (x >> 3) & 0xFu;
    const float man = float(x & 7u);
    const float raw = e == 0u ? man * uintBitsToFloat(0x3b000000u)
                              : (1.0 + man / 8.0) * uintBitsToFloat((e + 120u) << 23);
    return raw * 0.5;
}

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 1) * BLOCK_BYTES;
    const uint half_ = u & 1u;
    [[unroll]] for (uint sub = 0; sub < 2; sub++) {
        const uint s = 2 * half_ + sub;
        const f16vec2 d = h2(ue4m3_half(ld1(blk + s)));
        [[unroll]] for (uint i = 0; i < 2; i++) {
            const uint q = ld4(blk + 4 + 8 * s + 4 * i);
            [[unroll]] for (uint k = 0; k < 4; k += 2) {
                const uint b0 = q >> (8 * k);
                const uint b1 = q >> (8 * (k + 1));
                o[8 * sub + 2 * i + k / 2] =
                    pack2(f16vec2(float16_t(lut16(b0 & 15u)),
                                  float16_t(lut16(b1 & 15u))) * d);
                o[8 * sub + 4 + 2 * i + k / 2] =
                    pack2(f16vec2(float16_t(lut16((b0 >> 4) & 15u)),
                                  float16_t(lut16((b1 >> 4) & 15u))) * d);
            }
        }
    }
}
