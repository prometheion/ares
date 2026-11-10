// Q6_K for the f16 prefill kernels: 256 values in 210 bytes (2-byte aligned): u8 ql[128]
// (low 4 bits), u8 qh[64] (high 2 bits), i8 scales[16] (one per 16 values), f16 d at 208;
// value = d * scale * (code - 32). Each half h of 128 values uses ql[64h ..], qh[32h ..]
// and scales[8h ..]; for l in 0..31 the rows k = 0..3 are
//   y[l + 32 k] = (ql[64h + 32 (k & 1) + l] >> 4 (k >> 1)) & 15 | (qh[32h + l] >> 2k) & 3 << 4
// with the scale scales[8h + 2k + l / 16]. A unit is one such row of 32.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 210;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint v0 = (u & 7u) * 32;
    const uint h = v0 >> 7;
    const uint k = (v0 >> 5) & 3u;
    const float d = half_of(ld2(blk + 208));
    uint ql[8];
    uint qh[8];
    ld32_any(blk + 64 * h + 32 * (k & 1u), ql);
    ld32_any(blk + 128 + 32 * h, qh);
    const uint sc_a = ld1(blk + 192 + 8 * h + 2 * k);
    const uint sc_b = ld1(blk + 192 + 8 * h + 2 * k + 1);
    const f16vec2 s_a = h2(d * float(int(sc_a << 24) >> 24));
    const f16vec2 s_b = h2(d * float(int(sc_b << 24) >> 24));
    [[unroll]] for (uint i = 0; i < 8; i++) {
        const uint w = ((ql[i] >> (4 * (k >> 1))) & 0x0F0F0F0Fu) | (((qh[i] >> (2 * k)) & 0x03030303u) << 4);
        const f16vec2 s = i < 4 ? s_a : s_b;
        o[2 * i] = pack2((u8_01(w) - f16vec2(32.0hf)) * s);
        o[2 * i + 1] = pack2((u8_23(w) - f16vec2(32.0hf)) * s);
    }
}
