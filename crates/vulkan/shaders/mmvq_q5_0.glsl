// Q5_0 for the integer weight kernels (mmvq.glsl).
//
// 32 values in 22 bytes (2-byte aligned): f16 d, u32 qh (bit j is the fifth bit of
// value j), 16 bytes of nibbles as Q4_0; value = (code - 16) * d with the 5-bit code.

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 22;
const uint LANES = 2;

struct W {
    uint lo[2];
    uint hi[2];
    float d;
};

struct A {
    uvec2 first;
    uvec2 second;
    vec2 ds;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

// The low 4 bits of x as the bit 4 of four bytes (0x10 where the bit is set).
uint spread_to_bit4(uint x) {
    return (((x & 0xFu) * 0x00204081u) & 0x01010101u) << 4;
}

W load_w(uint64_t blk, uint s) {
    const uint qh = ld4_a2(blk + 2);
    const uvec2 q = ld8_a2(blk + 6 + 8 * s);
    W w;
    w.lo[0] = (q.x & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (8 * s));
    w.lo[1] = (q.y & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (8 * s + 4));
    w.hi[0] = ((q.x >> 4) & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (16 + 8 * s));
    w.hi[1] = ((q.y >> 4) & 0x0F0F0F0Fu) | spread_to_bit4(qh >> (20 + 8 * s));
    w.d = half_of(ld2(blk));
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
    acc = fma(-8.0 * w.d, a.ds.y, acc);
}
