// F32 for the f16 prefill kernels: the weights themselves, rounded to f16. One value per
// "block" of 4 bytes; a unit is 32 values in 128 bytes.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 1;
const uint BLOCK_BYTES = 4;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    [[unroll]] for (uint i = 0; i < 8; i++) {
        const vec4 v = ldf4(row + 128 * uint64_t(u) + 16 * i);
        o[2 * i] = pack2(f16vec2(float16_t(v.x), float16_t(v.y)));
        o[2 * i + 1] = pack2(f16vec2(float16_t(v.z), float16_t(v.w)));
    }
}
