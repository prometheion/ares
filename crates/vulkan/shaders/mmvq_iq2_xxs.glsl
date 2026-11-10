// IQ2_XXS for the integer weight kernels (mmvq.glsl).
//
// 256 values in 66 bytes (2-byte aligned).
//   0   f16 d
//   2   per sub-block ib of 32 values: u8 idx[4] (indices into a codebook of 256
//       groups of 8 magnitudes), then a u32 whose low 28 bits are four 7-bit sign
//       indices and whose top 4 bits are the scale
// value = d * (0.5 + scale) * 0.25 * magnitude * sign. A lane takes one sub-block
// (slot = ib), as IQ3_XXS does.

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 66;
const uint LANES = 8;

shared uvec2 grid[256];

struct W {
    uint w[8];
    int scale2;  // 2 * scale + 1
    float d;
};

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    for (uint i = t; i < 256; i += ARES_MMVQ_WARPS * 32) {
        grid[i] = ld8(tables + TBL_IQ2XXS + 8 * i);
    }
    barrier();
}

W load_w(uint64_t blk, uint ib) {
    const uint idx = ld4_a2(blk + 2 + 8 * ib);
    const uint aux = ld4_a2(blk + 6 + 8 * ib);
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uvec2 g = grid[(idx >> (8 * l)) & 0xFFu];
        const uint spread = ksigns((aux >> (7 * l)) & 127u) * 0x01010101u;
        w.w[2 * l] = apply_sign4(g.x, sign_mask4(spread, 0x08040201u));
        w.w[2 * l + 1] = apply_sign4(g.y, sign_mask4(spread, 0x80402010u));
    }
    w.scale2 = int(2 * (aux >> 28) + 1);
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
    // (0.5 + scale) * 0.25 = (2 scale + 1) / 8
    acc = fma(float(s * w.scale2), 0.125 * w.d * a.ds.x, acc);
}
