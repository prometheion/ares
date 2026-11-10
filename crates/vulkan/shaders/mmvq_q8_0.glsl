// Q8_0 for the integer weight kernels (mmvq.glsl); `mmvq_iquants.cuh` is the model.
//
// 32 values in 34 bytes (2-byte aligned): f16 d, 32 signed bytes. Two lanes per
// block, 16 bytes each.

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 34;
const uint LANES = 2;

struct W {
    uint q[4];
    float d;
};

struct A {
    uvec4 q;
    float d;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint s) {
    W w;
    [[unroll]] for (uint i = 0; i < 4; i++) {
        w.q[i] = ld4_a2(blk + 2 + 16 * s + 4 * i);
    }
    w.d = half_of(ld2(blk));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    a.q = ld16(q + 16 * s);
    a.d = ldf1(ds);
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_ss(w.q[0], a.q.x, 0);
    s = dp_ss(w.q[1], a.q.y, s);
    s = dp_ss(w.q[2], a.q.z, s);
    s = dp_ss(w.q[3], a.q.w, s);
    acc = fma(float(s), w.d * a.d, acc);
}
