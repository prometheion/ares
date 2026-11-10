// IQ2_XS for the integer weight kernels (mmvq.glsl); `mmvq_iquants.cuh` is the model.
//
// 256 values in 74 bytes (2-byte aligned).
//   0   f16 d
//   2   u16 qs[32]    9-bit codebook index (512 groups of 8 magnitudes) and a 7-bit
//                     sign index above it
//   66  u8 scales[8]  two 4-bit scales per sub-block: the first 16 values, the
//                     second 16
// Sub-block ib: qs[4 ib + l], l = 0..3, one group of 8 values each.
// value = d * (0.5 + scale) * 0.25 * magnitude * sign. Slot = ib.

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 74;
const uint LANES = 8;

shared uvec2 grid[512];

struct W {
    uint w[8];
    int scale2[2];  // 2 * scale + 1 of the two halves
    float d;
};

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    for (uint i = t; i < 512; i += ARES_MMVQ_WARPS * 32) {
        grid[i] = ld8(tables + TBL_IQ2XS + 8 * i);
    }
    barrier();
}

W load_w(uint64_t blk, uint ib) {
    const uvec2 qs = ld8_a2(blk + 2 + 8 * ib);
    const uint sc = ld1(blk + 66 + ib);
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint word = l < 2 ? qs.x : qs.y;
        const uint q = (word >> (16 * (l & 1))) & 0xFFFFu;
        const uvec2 g = grid[q & 511u];
        const uint spread = ksigns(q >> 9) * 0x01010101u;
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
    // (0.5 + scale) * 0.25 = (2 scale + 1) / 8
    acc = fma(float(s0 * w.scale2[0] + s1 * w.scale2[1]), 0.125 * w.d * a.ds.x, acc);
}
