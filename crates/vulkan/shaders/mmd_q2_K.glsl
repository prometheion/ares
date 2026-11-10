// Q2_K for the f16 prefill kernels: 256 values in 84 bytes (4-byte aligned): u8 scales[16]
// (per sub-block of 16 values: scale in the low nibble, minimum in the high nibble), u8
// qs[64] at 16 (2-bit codes: each half h of 128 values has 32 bytes, byte l holds the codes
// of values 32 step + l at shift 2 step), f16 d, f16 dmin at 80.
// value = d sc code - dmin m. A unit is one step of 32 values.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 84;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint v0 = (u & 7u) * 32;
    const uint h = v0 >> 7;
    const uint step = (v0 >> 5) & 3u;
    const uint dd = ld4(blk + 80);
    const float d = half_of(dd);
    const float dmin = half_of(dd >> 16);
    const uint sa = ld1(blk + (v0 >> 4));
    const uint sb = ld1(blk + (v0 >> 4) + 1);
    const f16vec2 s_a = h2(d * float(sa & 0xFu));
    const f16vec2 s_b = h2(d * float(sb & 0xFu));
    const f16vec2 z_a = h2(dmin * float(sa >> 4));
    const f16vec2 z_b = h2(dmin * float(sb >> 4));
    [[unroll]] for (uint i = 0; i < 8; i++) {
        const uint q = ld4(blk + 16 + 32 * h + 4 * i);
        const uint w = (q >> (2 * step)) & 0x03030303u;
        const f16vec2 s = i < 4 ? s_a : s_b;
        const f16vec2 z = i < 4 ? z_a : z_b;
        o[2 * i] = pack2(fma(u8_01(w), s, -z));
        o[2 * i + 1] = pack2(fma(u8_23(w), s, -z));
    }
}
