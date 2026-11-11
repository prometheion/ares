// NVFP4 for the integer weight kernels (mmvq.glsl).
//
// 64 values in 36 bytes (4-byte aligned): u8 d[4] (a UE4M3 scale per sub-block of 16
// values), u8 qs[32]: sub-block s has 8 bytes of nibbles, the low nibbles are its
// first 8 values and the high ones the next 8. Nibbles are levels of the 4-bit float
// table, as MXFP4's. A lane takes one sub-block; two sub-blocks share a group of 32
// activations.

#define KV_TABLE ares_kvalues_mxfp4
#include "mmvq_iq_common.glsl"
#include "fp4_scales.glsl"

const uint BLOCK_VALUES = 64;
const uint BLOCK_BYTES = 36;
const uint LANES = 4;

struct W {
    int x[2];  // table values of the low nibbles
    int y[2];  // of the high nibbles
    float d;
};

struct A {
    uvec2 first;   // activations 16 s .. 16 s + 7
    uvec2 second;  // 16 s + 8 ..
    float dx;
};

void mmvq_init(uint t, uint64_t tables) {
    iq4_init(t);
}

W load_w(uint64_t blk, uint s) {
    const uvec2 q = uvec2(ld4(blk + 4 + 8 * s), ld4(blk + 8 + 8 * s));
    W w;
    table16(q.x, w.x[0], w.y[0]);
    table16(q.y, w.x[1], w.y[1]);
    w.d = ue4m3_half(ld1(blk + s));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    a.first = ld8(q + 16 * s);
    a.second = ld8(q + 16 * s + 8);
    a.dx = ldf1(ds + 8 * (s >> 1));
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_ss(uint(w.x[0]), a.first.x, 0);
    s = dp_ss(uint(w.x[1]), a.first.y, s);
    s = dp_ss(uint(w.y[0]), a.second.x, s);
    s = dp_ss(uint(w.y[1]), a.second.y, s);
    acc = fma(float(s), w.d * a.dx, acc);
}
