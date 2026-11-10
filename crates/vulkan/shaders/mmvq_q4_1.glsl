// Q4_1 for the integer weight kernels (mmvq.glsl).
//
// 32 values in 20 bytes (4-byte aligned): f16 d, f16 m, 16 bytes of nibbles laid out
// as Q4_0; value = nibble * d + m. sum(x w) = d * dx * sum(q a) + m * s.

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 20;
const uint LANES = 2;

struct W {
    uint lo[2];
    uint hi[2];
    float d, m;
};

struct A {
    uvec2 first;
    uvec2 second;
    vec2 ds;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint s) {
    const uint head = ld4(blk);
    const uvec2 q = uvec2(ld4(blk + 4 + 8 * s), ld4(blk + 8 + 8 * s));
    W w;
    w.lo[0] = q.x & 0x0F0F0F0Fu;
    w.lo[1] = q.y & 0x0F0F0F0Fu;
    w.hi[0] = (q.x >> 4) & 0x0F0F0F0Fu;
    w.hi[1] = (q.y >> 4) & 0x0F0F0F0Fu;
    w.d = half_of(head);
    w.m = half_of(head >> 16);
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    a.first = ld8(q + 8 * s);
    a.second = ld8(q + 16 + 8 * s);
    a.ds = ldf2(ds);
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_us(w.lo[0], a.first.x, 0);
    s = dp_us(w.lo[1], a.first.y, s);
    s = dp_us(w.hi[0], a.second.x, s);
    s = dp_us(w.hi[1], a.second.y, s);
    acc = fma(float(s), w.d * a.ds.x, acc);
    acc = fma(0.5 * w.m, a.ds.y, acc);
}
