// TQ2_0 for the integer weight kernels (mmvq.glsl).
//
// 256 values in 66 bytes (2-byte aligned): u8 qs[64], f16 d. Four 2-bit codes per
// byte (0, 1, 2 for -1, 0, +1; a 3 decodes to +2, as the reference does), in planes:
// in each run of 32 bytes the codes at shift 2 l of byte m are value 128 run +
// 32 l + m. A plane of 32 values is a group of activations. A lane takes one word
// (4 bytes) of qs: all four planes of 4 consecutive values.

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 66;
const uint LANES = 16;

struct W {
    uint c[4];
    float d;
};

struct A {
    uint a[4];
    float dx[4];
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint s) {
    const uint word = ld4_a2(blk + 4 * s);
    W w;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        w.c[l] = (word >> (2 * l)) & 0x03030303u;
    }
    w.d = half_of(ld2(blk + 64));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    const uint run = s >> 3;
    const uint m4 = 4 * (s & 7u);
    A a;
    [[unroll]] for (uint l = 0; l < 4; l++) {
        a.a[l] = ld4(q + 128 * run + 32 * l + m4);
        a.dx[l] = ldf1(ds + 8 * (4 * run + l));
    }
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    [[unroll]] for (uint l = 0; l < 4; l++) {
        const int t = dp_us(w.c[l], a.a[l], -dp_ss(0x01010101u, a.a[l], 0));
        acc = fma(float(t), w.d * a.dx[l], acc);
    }
}
