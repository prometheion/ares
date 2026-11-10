// Q2_0 for the integer weight kernels (mmvq.glsl).
//
// 64 values in 18 bytes (2-byte aligned): f16 d, 16 bytes with four 2-bit codes
// each, the first value in the low bits; value = (code - 1) * d. A lane takes one
// word (4 bytes, 16 values) of codes, which it spreads to one code per byte.

const uint BLOCK_VALUES = 64;
const uint BLOCK_BYTES = 18;
const uint LANES = 4;

struct W {
    uint c[4];  // the codes of values 4i .. 4i+3 of the lane's 16, one per byte
    float d;
};

struct A {
    uvec4 q;
    float dx;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

// The four 2-bit fields of the low byte of x as four bytes, field 0 in byte 0.
uint spread_pairs(uint x) {
    x &= 0xFFu;
    x = (x | (x << 12)) & 0x000F000Fu;
    return (x | (x << 6)) & 0x03030303u;
}

W load_w(uint64_t blk, uint s) {
    const uint word = ld4_a2(blk + 2 + 4 * s);
    W w;
    [[unroll]] for (uint i = 0; i < 4; i++) {
        w.c[i] = spread_pairs(word >> (8 * i));
    }
    w.d = half_of(ld2(blk));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    a.q = ld16(q + 16 * s);
    a.dx = ldf1(ds + 8 * (s >> 1));
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_us(w.c[0], a.q.x, 0);
    s = dp_us(w.c[1], a.q.y, s);
    s = dp_us(w.c[2], a.q.z, s);
    s = dp_us(w.c[3], a.q.w, s);
    int sa = dp_ss(0x01010101u, a.q.x, 0);
    sa = dp_ss(0x01010101u, a.q.y, sa);
    sa = dp_ss(0x01010101u, a.q.z, sa);
    sa = dp_ss(0x01010101u, a.q.w, sa);
    acc = fma(float(s - sa), w.d * a.dx, acc);
}
