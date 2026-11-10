// IQ1_M for the integer weight kernels (mmvq.glsl).
//
// 256 values in 56 bytes (4-byte aligned).
//   0   u8 qs[32]    low 8 bits of 11-bit codebook indices
//   32  u8 qh[16]    two bytes per sub-block ib of 32 values: each byte holds the
//                    high index bits of two groups (bits 0..2 and 4..6) and, in
//                    bits 3 and 7, the sign of their offsets
//   48  u16 scales[4]  3-bit scales (two per group pair: sub-block ib uses
//                    scales[ib/2] at bit 6 (ib % 2) and 3 more bits up); the top
//                    nibbles of the four words are the bits of the f16 d
// A group is 8 values in {-1, 0, 1}; value = dl * (grid + delta) with a delta per
// group and dl per pair of groups. A lane takes one sub-block.

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 56;
const uint LANES = 8;

shared uint grid[1024];  // two 16-bit entries each, 2 bits per value (value + 1)

uint iq1_entry(uint index) {
    return (grid[index >> 1] >> ((index & 1u) * 16u)) & 0xFFFFu;
}

struct W {
    uint w[8];       // grid bytes + 1 (0..2), 4 per word
    float dl[4];     // d * (2 scale + 1) of each group
    float delta[4];  // +-0.125 of each group
};

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    for (uint i = t; i < 1024; i += ARES_MMVQ_WARPS * 32) {
        grid[i] = ld4(tables + TBL_IQ1S + 4 * i);
    }
    barrier();
}

W load_w(uint64_t blk, uint ib) {
    const uint qs = ld4(blk + 4 * ib);
    const uint qhw = ld2(blk + 32 + 2 * ib);
    const uint h0 = qhw & 0xFFu;
    const uint h1 = qhw >> 8;
    const uint s01 = ld4(blk + 48);
    const uint s23 = ld4(blk + 52);
    // The f16 scale d: one nibble from the top of each 16-bit word.
    const uint dbits = ((s01 & 0xFFFFu) >> 12) | (((s01 >> 16) >> 8) & 0xF0u)
        | (((s23 & 0xFFFFu) >> 4) & 0xF00u) | ((s23 >> 16) & 0xF000u);
    const float d = half_of(dbits);
    const uint pair = (ib >> 1) < 2 ? s01 : s23;
    const uint sc = ((ib >> 1) & 1u) == 0u ? (pair & 0xFFFFu) : (pair >> 16);
    const uint shift = 6 * (ib & 1u);
    const float dl1 = d * float(2 * ((sc >> shift) & 7u) + 1);
    const float dl2 = d * float(2 * ((sc >> (shift + 3)) & 7u) + 1);
    const uint idx[4] = uint[4](
        (qs & 0xFFu) | ((h0 << 8) & 0x700u), ((qs >> 8) & 0xFFu) | ((h0 << 4) & 0x700u),
        ((qs >> 16) & 0xFFu) | ((h1 << 8) & 0x700u), (qs >> 24) | ((h1 << 4) & 0x700u));
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint g = iq1_entry(idx[l]);
        w.w[2 * l] = spread_pairs(g);
        w.w[2 * l + 1] = spread_pairs(g >> 8);
        w.dl[l] = l < 2 ? dl1 : dl2;
    }
    w.delta[0] = (h0 & 0x08u) != 0u ? -0.125 : 0.125;
    w.delta[1] = (h0 & 0x80u) != 0u ? -0.125 : 0.125;
    w.delta[2] = (h1 & 0x08u) != 0u ? -0.125 : 0.125;
    w.delta[3] = (h1 & 0x80u) != 0u ? -0.125 : 0.125;
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint ib) {
    return load_sub_a(q, ds, ib);
}

void dot_acc(inout float acc, W w, A a) {
    const uint aw[8] = uint[8](a.first.x, a.first.y, a.first.z, a.first.w, a.second.x, a.second.y,
        a.second.z, a.second.w);
    [[unroll]] for (uint l = 0; l < 4; l++) {
        // sum((g + 1) a); the sum of the group's 8 activations turns it into the sum of g a
        // and multiplies the offset: sum((g + delta) a) = that + (delta - 1) sum(a).
        const int s = dp_us(w.w[2 * l + 1], aw[2 * l + 1], dp_us(w.w[2 * l], aw[2 * l], 0));
        const int sa = dp_ss(0x01010101u, aw[2 * l + 1], dp_ss(0x01010101u, aw[2 * l], 0));
        acc = fma(float(s) + (w.delta[l] - 1.0) * float(sa), w.dl[l] * a.ds.x, acc);
    }
}
