// Q4_0 for the integer weight kernels (mmvq.glsl).
//
// 32 values in 18 bytes (2-byte aligned): f16 d, 16 bytes of nibbles (low nibbles
// are values 0..15, high 16..31); value = (nibble - 8) * d. Two lanes per block, 8
// bytes each. The 8 comes off with the sum of the activations that the quantizer
// keeps: sum((q - 8) x) = dx * sum(q * a) - 8 * s.

const uint BLOCK_VALUES = 32;
const uint BLOCK_BYTES = 18;
const uint LANES = 2;

struct W {
    uint lo[2];  // low nibbles, one per byte: values 8s .. 8s+7
    uint hi[2];  // high nibbles: values 16+8s ..
    float d;
};

struct A {
    uvec2 first;   // activations 8s .. 8s+7
    uvec2 second;  // 16+8s .. 16+8s+7
    vec2 ds;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint s) {
    const uvec2 q = ld8_a2(blk + 2 + 8 * s);
    W w;
    w.lo[0] = q.x & 0x0F0F0F0Fu;
    w.lo[1] = q.y & 0x0F0F0F0Fu;
    w.hi[0] = (q.x >> 4) & 0x0F0F0F0Fu;
    w.hi[1] = (q.y >> 4) & 0x0F0F0F0Fu;
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
    // Each of the two lanes of the block takes half of the offset.
    acc = fma(float(s), w.d * a.ds.x, acc);
    acc = fma(-4.0 * w.d, a.ds.y, acc);
}
