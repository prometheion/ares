// BF16 for the f16 prefill kernels: the weights widened to f32 and rounded to f16 (a bf16
// outside the f16 range, beyond 65504, becomes infinite: weights are far below it); a unit
// is 32 values in 64 bytes.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 1;
const uint BLOCK_BYTES = 2;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uvec4 v = ld16(row + 64 * uint64_t(u) + 16 * i);
        [[unroll]] for (uint k = 0; k < 4; k++) {
            const float lo = uintBitsToFloat(v[k] << 16);
            const float hi = uintBitsToFloat(v[k] & 0xFFFF0000u);
            o[4 * i + k] = pack2(f16vec2(float16_t(lo), float16_t(hi)));
        }
    }
}
