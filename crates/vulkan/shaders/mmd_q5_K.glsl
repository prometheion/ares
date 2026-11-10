// Q5_K for the f16 prefill kernels in the file's layout: Q4_K with a fifth bit per value.
// 256 values in 176 bytes, 16-byte aligned: header as Q4_K; at 16 u8 qh[32] (byte l holds
// bit 4 of position l of every sub-block: sub-block i at bit i); at 48 u8 qs[128] nibbles
// as Q4_K. value = d sc q - dmin m. A unit is a sub-block.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 176;

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint sub = u & 7u;
    const uint j = sub >> 1;
    const uvec4 hdr = ld16(blk);
    uint sc, mn;
    scales_k(j, hdr.yzw, sc, mn);
    const vec2 dd = unpackHalf2x16(hdr.x);
    const uint sh = 8 * (sub & 1u);
    const f16vec2 s = h2(dd.x * float((sc >> sh) & 0xFFu));
    const f16vec2 z = h2(dd.y * float((mn >> sh) & 0xFFu));
    const uint nsh = 4 * (sub & 1u);
    [[unroll]] for (uint h = 0; h < 2; h++) {
        const uvec4 q = ld16(blk + 48 + 32 * j + 16 * h);
        const uvec4 qh = ld16(blk + 16 + 16 * h);
        [[unroll]] for (uint i = 0; i < 4; i++) {
            const uint w = ((q[i] >> nsh) & 0x0F0F0F0Fu) | (((qh[i] >> sub) & 0x01010101u) << 4);
            o[8 * h + 2 * i] = pack2(fma(u8_01(w), s, -z));
            o[8 * h + 2 * i + 1] = pack2(fma(u8_23(w), s, -z));
        }
    }
}
