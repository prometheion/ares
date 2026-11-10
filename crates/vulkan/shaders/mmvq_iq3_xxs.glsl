// IQ3_XXS for the integer weight kernels (mmvq.glsl); `mmvq_iquants.cuh` is the model.
//
// 256 values in 98 bytes (2-byte aligned).
//   0   f16 d
//   2   u8 grid[64]   indices into a codebook of 256 groups of 4 magnitudes
//   66  u32 aux[8]    per sub-block: four 7-bit sign indices, 4-bit scale on top
// Sub-block ib (32 values = 8 groups of 4): grid[8 ib + 0 .. 7]; group pair l (8
// values) takes grid[8 ib + 2l], grid[8 ib + 2l + 1] and the sign byte of 7 bits
// (aux >> 7l). value = d * (0.5 + scale) * 0.5 * magnitude * sign. A lane takes one
// sub-block (slot = ib).

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 98;
const uint LANES = 8;

// The codebook in shared memory: lookups with a different index in every lane are
// conflict-limited there and would touch many cache lines in global memory.
shared uint grid[256];

struct W {
    uint w[8];   // signed values, 4 per word
    int scale2;  // 2 * scale + 1
    float d;
};

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    for (uint i = t; i < 256; i += ARES_MMVQ_WARPS * 32) {
        grid[i] = ld4(tables + TBL_IQ3XXS + 4 * i);
    }
    barrier();
}

W load_w(uint64_t blk, uint ib) {
    const uvec2 idx = ld8_a2(blk + 2 + 8 * ib);
    const uint aux = ld4_a2(blk + 66 + 4 * ib);
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const uint word = l < 2 ? idx.x : idx.y;
        const uint i1 = (word >> (16 * (l & 1))) & 0xFFu;
        const uint i2 = (word >> (16 * (l & 1) + 8)) & 0xFFu;
        const uint signs = ksigns((aux >> (7 * l)) & 127u) * 0x01010101u;
        w.w[2 * l] = apply_sign4(grid[i1], sign_mask4(signs, 0x08040201u));
        w.w[2 * l + 1] = apply_sign4(grid[i2], sign_mask4(signs, 0x80402010u));
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
    // (0.5 + scale) * 0.5 = (2 scale + 1) / 4
    acc = fma(float(s * w.scale2), 0.25 * w.d * a.ds.x, acc);
}
