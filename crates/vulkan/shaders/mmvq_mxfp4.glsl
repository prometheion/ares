// MXFP4 for the integer weight kernels (mmvq.glsl).
//
// 32 values in 17 bytes (any alignment): u8 e (an E8M0 scale), 16 bytes of nibbles
// laid out as Q4_0, each a level of the 4-bit float table (doubled E2M1 levels, so
// the scale is 2^(e - 127) / 2). Two lanes per block, 8 bytes each, read as words
// with a funnel shift since the bytes are not word aligned. The nibbles go through
// the table in shared memory like IQ4_NL's (mmvq_iq_common.glsl).

#define KV_TABLE ares_kvalues_mxfp4
#include "mmvq_iq_common.glsl"
#include "fp4_scales.glsl"

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 17;
const uint LANES = 2;

struct W {
    int x[2];  // table values of the low nibbles: values 8s .. 8s+7
    int y[2];  // of the high nibbles: 16+8s ..
    float d;
};

struct A {
    uvec2 first;
    uvec2 second;
    float dx;
};

void mmvq_init(uint t, uint64_t tables) {
    iq4_init(t);
}

W load_w(uint64_t blk, uint s) {
    const uvec2 q = ld8_u(blk + 1 + 8 * s);
    W w;
    table16(q.x, w.x[0], w.y[0]);
    table16(q.y, w.x[1], w.y[1]);
    w.d = e8m0_half(ld1(blk));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    a.first = ld8(q + 8 * s);
    a.second = ld8(q + 16 + 8 * s);
    a.dx = ldf1(ds);
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_ss(uint(w.x[0]), a.first.x, 0);
    s = dp_ss(uint(w.x[1]), a.first.y, s);
    s = dp_ss(uint(w.y[0]), a.second.x, s);
    s = dp_ss(uint(w.y[1]), a.second.y, s);
    acc = fma(float(s), w.d * a.dx, acc);
}
