// Q3_K for the f16 prefill kernels: 256 values in 110 bytes (2-byte aligned): u8 hmask[32],
// u8 qs[64] (2 bits per value), u8 scales[12] at 96 (16 six-bit scales, offset 32: the low 4
// bits of scale i in the nibbles of bytes 0..7 (i < 8: low nibble of byte i, else high
// nibble of byte i - 8), the high 2 bits of scale i in bits 2 (i / 4) of byte 8 + i % 4),
// f16 d at 108. value = d (scale - 32) (code - 4), code = 2 low bits | high bit << 2.
// A half h (128 values) uses qs[32h ..]: byte l holds the codes of positions l, l + 32,
// l + 64, l + 96 of the half at shifts 0, 2, 4, 6; the high bit of row k is bit 4h + k of
// hmask[l]. The scale of row k is scale 8h + 2k + l / 16. A unit is one row of 32.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 110;

float scale_q3(uint64_t blk, uint i) {
    const uint lo = (ld1(blk + 96 + (i & 7u)) >> (4 * (i >> 3))) & 0xFu;
    const uint hi = (ld1(blk + 104 + (i & 3u)) >> (2 * (i >> 2))) & 3u;
    return float(int(lo | (hi << 4)) - 32);
}

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint v0 = (u & 7u) * 32;
    const uint h = v0 >> 7;
    const uint k = (v0 >> 5) & 3u;
    const float d = half_of(ld2(blk + 108));
    uint qs[8];
    uint hm[8];
    ld32_any(blk + 32 + 32 * h, qs);
    ld32_any(blk, hm);
    const f16vec2 s_a = h2(d * scale_q3(blk, 8 * h + 2 * k));
    const f16vec2 s_b = h2(d * scale_q3(blk, 8 * h + 2 * k + 1));
    [[unroll]] for (uint i = 0; i < 8; i++) {
        const uint w = ((qs[i] >> (2 * k)) & 0x03030303u) | ((((hm[i] >> (4 * h + k)) & 0x01010101u) << 2));
        const f16vec2 s = i < 4 ? s_a : s_b;
        o[2 * i] = pack2((u8_01(w) - f16vec2(4.0hf)) * s);
        o[2 * i + 1] = pack2((u8_23(w) - f16vec2(4.0hf)) * s);
    }
}
