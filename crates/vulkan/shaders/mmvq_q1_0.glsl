// Q1_0 for the integer weight kernels (mmvq.glsl).
//
// 128 values in 18 bytes (2-byte aligned): f16 d, 16 bytes of bits (bit j is the sign
// of value j: set is +d, clear -d). With x the dequantized activations,
// sum(+-x) = 2 sum(x where the bit is set) - sum(x): the first is a dot product with
// the bits spread to bytes of 0 and 1, the second is what the quantizer keeps.
// A lane takes the 32 values (one word of bits, one group of activations) of one
// quarter.

const uint BLOCK_VALUES = 128;
const uint BLOCK_BYTES = 18;
const uint LANES = 4;

struct W {
    uint bits;
    float d;
};

struct A {
    uvec4 first;
    uvec4 second;
    vec2 ds;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint s) {
    W w;
    w.bits = ld4_a2(blk + 2 + 4 * s);
    w.d = half_of(ld2(blk));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    a.first = ld16(q + 32 * s);
    a.second = ld16(q + 32 * s + 16);
    a.ds = ldf2(ds + 8 * s);
    return a;
}

// The 4 low bits of x as four bytes of 0 or 1.
uint spread_bits(uint x) {
    return ((x & 0xFu) * 0x00204081u) & 0x01010101u;
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_us(spread_bits(w.bits), a.first.x, 0);
    s = dp_us(spread_bits(w.bits >> 4), a.first.y, s);
    s = dp_us(spread_bits(w.bits >> 8), a.first.z, s);
    s = dp_us(spread_bits(w.bits >> 12), a.first.w, s);
    s = dp_us(spread_bits(w.bits >> 16), a.second.x, s);
    s = dp_us(spread_bits(w.bits >> 20), a.second.y, s);
    s = dp_us(spread_bits(w.bits >> 24), a.second.z, s);
    s = dp_us(spread_bits(w.bits >> 28), a.second.w, s);
    acc = fma(float(s), 2.0 * w.d * a.ds.x, acc);
    acc = fma(-w.d, a.ds.y, acc);
}
