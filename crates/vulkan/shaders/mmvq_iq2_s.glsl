// IQ2_S for the integer weight kernels (mmvq.glsl); `mmvq_iquants.cuh` is the model.
//
// 256 values in 82 bytes (2-byte aligned).
//   0   f16 d
//   2   u8 qs[64]     the first 32: low 8 bits of 10-bit codebook indices (1024
//                     groups of 8 magnitudes); the last 32: sign bytes
//   66  u8 qh[8]      the top 2 bits of the indices of sub-block ib
//   74  u8 scales[8]  as IQ2_XS
// Sub-block ib: index qs[4 ib + l] | (qh[ib] << (8 - 2l)) & 0x300, signs
// qs[32 + 4 ib + l]. Slot = ib.

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 82;
const uint LANES = 8;

shared uvec2 grid[1024];

struct W {
    uint w[8];
    int scale2[2];
    float d;
};

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    for (uint i = t; i < 1024; i += ARES_MMVQ_WARPS * 32) {
        grid[i] = ld8(tables + TBL_IQ2S + 8 * i);
    }
    barrier();
}

W load_w(uint64_t blk, uint ib) {
    const uint qs = ld4_a2(blk + 2 + 4 * ib);
    const uint signs = ld4_a2(blk + 34 + 4 * ib);
    const uint qh = ld1(blk + 66 + ib);
    const uint sc = ld1(blk + 74 + ib);
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint idx = ((qs >> (8 * l)) & 0xFFu) | ((qh << (8 - 2 * l)) & 0x300u);
        const uvec2 g = grid[idx];
        const uint spread = ((signs >> (8 * l)) & 0xFFu) * 0x01010101u;
        w.w[2 * l] = apply_sign4(g.x, sign_mask4(spread, 0x08040201u));
        w.w[2 * l + 1] = apply_sign4(g.y, sign_mask4(spread, 0x80402010u));
    }
    w.scale2[0] = int(2 * (sc & 0xFu) + 1);
    w.scale2[1] = int(2 * (sc >> 4) + 1);
    w.d = half_of(ld2(blk));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint ib) {
    return load_sub_a(q, ds, ib);
}

void dot_acc(inout float acc, W w, A a) {
    int s0 = dp_ss(w.w[0], a.first.x, 0);
    s0 = dp_ss(w.w[1], a.first.y, s0);
    s0 = dp_ss(w.w[2], a.first.z, s0);
    s0 = dp_ss(w.w[3], a.first.w, s0);
    int s1 = dp_ss(w.w[4], a.second.x, 0);
    s1 = dp_ss(w.w[5], a.second.y, s1);
    s1 = dp_ss(w.w[6], a.second.z, s1);
    s1 = dp_ss(w.w[7], a.second.w, s1);
    acc = fma(float(s0 * w.scale2[0] + s1 * w.scale2[1]), 0.125 * w.d * a.ds.x, acc);
}
