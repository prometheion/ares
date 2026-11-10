// TQ1_0 for the integer weight kernels (mmvq.glsl).
//
// 256 values in 54 bytes (2-byte aligned): u8 qs[48], u8 qh[4], f16 d. Every value
// is -1, 0 or +1 times d; five values (base-3 digits, 0..2 for -1..1) share a byte
// of qs, four share a byte of qh. The digits are read plane by plane: the digit n of
// the byte m of the first 32 bytes of qs is value 32 n + m (the 5 planes are the 5
// groups of activations), of the next 16 bytes value 160 + 16 n + m, and the digit n
// of byte 48 + j of qh is value 240 + 4 n + j.
// A digit is read by multiplying the byte by 3^n (mod 256) and taking the top of
// 3 times that, a multiplication that two bytes at a time fits in 16-bit lanes.
//
// 16 lanes per block: lanes 0..7 take a word of the first run, 8..11 a word of the
// second, 12 the word of qh, the others nothing (they add 0).

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 54;
const uint LANES = 16;

struct W {
    uint dig[5];  // digit plane n of the lane's four bytes, 0..2 per byte
    float d;
};

struct A {
    uint a[5];
    float dx[5];
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

uint tq1_digits(uint w, uint p) {
    uint even = ((w & 0x00FF00FFu) * p) & 0x00FF00FFu;
    uint odd = (((w >> 8) & 0x00FF00FFu) * p) & 0x00FF00FFu;
    even = ((even * 3u) >> 8) & 0x00030003u;
    odd = ((odd * 3u) >> 8) & 0x00030003u;
    return even | (odd << 8);
}

W load_w(uint64_t blk, uint s) {
    W w;
    uint word = 0u;
    uint planes = 0u;
    if (s < 8u) {
        word = ld4_a2(blk + 4 * s);
        planes = 5u;
    } else if (s < 12u) {
        word = ld4_a2(blk + 32 + 4 * (s - 8u));
        planes = 5u;
    } else if (s == 12u) {
        word = ld4_a2(blk + 48);
        planes = 4u;
    }
    const uint pow3[5] = uint[5](1u, 3u, 9u, 27u, 81u);
    [[unroll]] for (uint n = 0; n < 5; n++) {
        // A plane that does not exist is all zeros (digit 1), which adds nothing.
        w.dig[n] = n < planes ? tq1_digits(word, pow3[n]) : 0x01010101u;
    }
    w.d = planes == 0u ? 0.0 : half_of(ld2(blk + 52));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    A a;
    [[unroll]] for (uint n = 0; n < 5; n++) {
        uint value = 0u;  // first value of the lane's four in plane n
        bool real = true;
        if (s < 8u) {
            value = 32 * n + 4 * s;
        } else if (s < 12u) {
            value = 160 + 16 * n + 4 * (s - 8u);
        } else if (s == 12u) {
            value = 240 + 4 * n;
            real = n < 4;
        } else {
            real = false;
        }
        if (!real) {
            value = 0u;
        }
        a.a[n] = ld4(q + value);
        a.dx[n] = ldf1(ds + 8 * (value >> 5));
    }
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    [[unroll]] for (uint n = 0; n < 5; n++) {
        // sum((digit - 1) a): the 1 comes off with a dot product against ones.
        const int t = dp_us(w.dig[n], a.a[n], -dp_ss(0x01010101u, a.a[n], 0));
        acc = fma(float(t), w.d * a.dx[n], acc);
    }
}
