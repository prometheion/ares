// IQ3_S for the integer weight kernels (mmvq.glsl); `mmvq_iquants.cuh` is the model.
//
// 256 values in 110 bytes (2-byte aligned).
//   0    f16 d
//   2    u8 qs[64]      low 8 bits of 9-bit codebook indices (512 groups of 4)
//   66   u8 qh[8]       the ninth bits: byte ib for sub-block ib
//   74   u8 signs[32]   one sign bit per value
//   106  u8 scales[4]   4 bits per sub-block
// Sub-block ib: indices qs[8 ib + 2l + {0,1}] | bit of qh[ib], signs[4 ib + l] for
// l = 0..3; value = d * (1 + 2 scale) * magnitude * sign. Slot = ib.

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 110;
const uint LANES = 8;

shared uint grid[512];

struct W {
    uint w[8];
    int scale2;  // 1 + 2 * scale
    float d;
};

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    for (uint i = t; i < 512; i += ARES_MMVQ_WARPS * 32) {
        grid[i] = ld4(tables + TBL_IQ3S + 4 * i);
    }
    barrier();
}

W load_w(uint64_t blk, uint ib) {
    const uvec2 qs = ld8_a2(blk + 2 + 8 * ib);
    const uint qh = ld1(blk + 66 + ib);
    const uint signs = ld4_a2(blk + 74 + 4 * ib);
    const uint sc = ld1(blk + 106 + (ib >> 1));
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint word = l < 2 ? qs.x : qs.y;
        const uint i1 = ((word >> (16 * (l & 1))) & 0xFFu) | ((qh << (8 - 2 * l)) & 256u);
        const uint i2 = ((word >> (16 * (l & 1) + 8)) & 0xFFu) | ((qh << (7 - 2 * l)) & 256u);
        const uint spread = ((signs >> (8 * l)) & 0xFFu) * 0x01010101u;
        w.w[2 * l] = apply_sign4(grid[i1], sign_mask4(spread, 0x08040201u));
        w.w[2 * l + 1] = apply_sign4(grid[i2], sign_mask4(spread, 0x80402010u));
    }
    w.scale2 = int(1 + 2 * ((sc >> (4 * (ib & 1))) & 0xFu));
    w.d = half_of(ld2(blk));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint ib) {
    return load_sub_a(q, ds, ib);
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_ss(w.w[0], a.first.x, 0);
    s = dp_ss(w.w[1], a.first.y, s);
    s = dp_ss(w.w[2], a.first.z, s);
    s = dp_ss(w.w[3], a.first.w, s);
    s = dp_ss(w.w[4], a.second.x, s);
    s = dp_ss(w.w[5], a.second.y, s);
    s = dp_ss(w.w[6], a.second.z, s);
    s = dp_ss(w.w[7], a.second.w, s);
    acc = fma(float(s * w.scale2), w.d * a.ds.x, acc);
}
