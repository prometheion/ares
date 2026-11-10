// IQ4_NL for the integer weight kernels (mmvq.glsl); `mmvq_iquants.cuh` is the model.
//
// 32 values in 18 bytes (2-byte aligned): f16 d, then 16 bytes of nibbles laid out
// as Q4_0 (low nibbles are values 0..15, high 16..31), each through the same table
// as IQ4_XS. Two lanes per block, 8 bytes each.

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 18;
const uint LANES = 2;

struct W {
    int x[2];  // values 8s .. 8s+7
    int y[2];  // values 16+8s .. 16+8s+7
    float d;
};

struct A {
    uvec2 first;   // activations 8s .. 8s+7
    uvec2 second;  // 16+8s .. 16+8s+7
    float d;
};

void mmvq_init(uint t, uint64_t tables) {
    iq4_init(t);
}

W load_w(uint64_t blk, uint s) {
    const uvec2 q = ld8_a2(blk + 2 + 8 * s);
    W w;
    table16(q.x, w.x[0], w.y[0]);
    table16(q.y, w.x[1], w.y[1]);
    w.d = half_of(ld2(blk));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    a.first = ld8(q + 8 * s);
    a.second = ld8(q + 16 + 8 * s);
    a.d = ldf1(ds);
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_ss(uint(w.x[0]), a.first.x, 0);
    s = dp_ss(uint(w.x[1]), a.first.y, s);
    s = dp_ss(uint(w.y[0]), a.second.x, s);
    s = dp_ss(uint(w.y[1]), a.second.y, s);
    acc = fma(float(s), w.d * a.d, acc);
}
