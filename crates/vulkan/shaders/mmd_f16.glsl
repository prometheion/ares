// F16 for the f16 prefill kernels: the weights as they are; a unit is 32 values in 64 bytes.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 1;
const uint BLOCK_BYTES = 2;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uvec4 v = ld16(row + 64 * uint64_t(u) + 16 * i);
        o[4 * i] = v.x;
        o[4 * i + 1] = v.y;
        o[4 * i + 2] = v.z;
        o[4 * i + 3] = v.w;
    }
}
