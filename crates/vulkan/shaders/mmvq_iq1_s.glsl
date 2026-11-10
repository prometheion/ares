// IQ1_S for the integer weight kernels (mmvq.glsl).
//
// 256 values in 50 bytes (2-byte aligned).
//   0   f16 d
//   2   u8 qs[32]    low 8 bits of 11-bit codebook indices
//   34  u16 qh[8]    per sub-block ib of 32 values: three more index bits for each
//                    of its 4 groups (bits 3l..3l+2), the scale in bits 12..14 and
//                    the sign of the offset in bit 15
// A group is 8 values in {-1, 0, 1}. The codebook is kept as 2 bits per value (value + 1,
// 4 KB in shared memory instead of 16), so the dot product is taken with the unsigned
// g + 1 and the sum of the activations comes off: sum(g a) = sum((g + 1) a) - sum(a).
// value = d * (2 scale + 1) * (grid + delta), delta = +-0.125 for the sub-block, so
// sum(x w) = d (2 scale + 1) (dx sum((g + 1) a) + (delta - 1) s). A lane takes one
// sub-block.

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 50;
const uint LANES = 8;

shared uint grid[1024];  // two 16-bit entries each

struct W {
    uint w[8];  // grid bytes + 1 (0..2), 4 per word
    float dl;   // d * (2 scale + 1)
    float delta;
};

uint iq1_entry(uint index) {
    return (grid[index >> 1] >> ((index & 1u) * 16u)) & 0xFFFFu;
}

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    for (uint i = t; i < 1024; i += ARES_MMVQ_WARPS * 32) {
        grid[i] = ld4(tables + TBL_IQ1S + 4 * i);
    }
    barrier();
}

W load_w(uint64_t blk, uint ib) {
    const uint qs = ld4_a2(blk + 2 + 4 * ib);
    const uint qh = ld2(blk + 34 + 2 * ib);
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint g = iq1_entry(((qs >> (8 * l)) & 0xFFu) | (((qh >> (3 * l)) & 7u) << 8));
        w.w[2 * l] = spread_pairs(g);
        w.w[2 * l + 1] = spread_pairs(g >> 8);
    }
    w.dl = half_of(ld2(blk)) * float(2 * ((qh >> 12) & 7u) + 1);
    w.delta = (qh & 0x8000u) != 0u ? -0.125 : 0.125;
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint ib) {
    return load_sub_a(q, ds, ib);
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_us(w.w[0], a.first.x, 0);
    s = dp_us(w.w[1], a.first.y, s);
    s = dp_us(w.w[2], a.first.z, s);
    s = dp_us(w.w[3], a.first.w, s);
    s = dp_us(w.w[4], a.second.x, s);
    s = dp_us(w.w[5], a.second.y, s);
    s = dp_us(w.w[6], a.second.z, s);
    s = dp_us(w.w[7], a.second.w, s);
    acc = fma(float(s), w.dl * a.ds.x, acc);
    acc = fma((w.delta - 1.0) * w.dl, a.ds.y, acc);
}
