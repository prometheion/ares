// IQ4_XS for the f16 prefill kernels: 256 values in 136 bytes (8-byte aligned): f16 d, u16
// scales_h, u8 scales_l[4], u8 qs[128]. Sub-block ib (32 values) has the scale (scales_l
// nibble | 2 bits of scales_h) - 32 and the 16 bytes qs[16 ib ..]: low nibbles are its
// values 0..15, high nibbles its values 16..31, each a level of IQ4_NL's table.
// value = d scale level. A unit is a sub-block.
#define MD_TABLE16 ares_kvalues_iq4nl
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 136;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint ib = u & 7u;
    const uvec2 hdr = ld8(blk);
    const int ls = int((((hdr.y >> (4 * (ib & 1u))) >> (8 * (ib >> 1))) & 0x0Fu) |
                       (((hdr.x >> (16 + 2 * ib)) & 3u) << 4)) - 32;
    const f16vec2 s = h2(half_of(hdr.x) * float(ls));
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const uint q = ld4(blk + 8 + 16 * ib + 4 * i);
        [[unroll]] for (uint k = 0; k < 4; k += 2) {
            const uint b0 = q >> (8 * k);
            const uint b1 = q >> (8 * (k + 1));
            o[2 * i + k / 2] = pack2(f16vec2(float16_t(lut16(b0 & 15u)),
                                             float16_t(lut16(b1 & 15u))) * s);
            o[8 + 2 * i + k / 2] = pack2(f16vec2(float16_t(lut16((b0 >> 4) & 15u)),
                                                 float16_t(lut16((b1 >> 4) & 15u))) * s);
        }
    }
}
