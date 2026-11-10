// Q3_K for the integer weight kernels (mmvq.glsl); `mmvq_kquants.cuh` is the model.
//
// 256 values in 110 bytes (2-byte aligned).
//   0    u8 hmask[32]
//   32   u8 qs[64]      2 bits per value
//   96   u8 scales[12]  16 six-bit scales (offset 32): the low 4 bits of scale i in
//                       the nibbles of bytes 0..7 (i < 8: low nibble of byte i,
//                       else high nibble of byte i-8), the high 2 bits of scale i
//                       in bits 2*(i/4) of byte 8 + i%4
//   108  f16 d
// value = d * (scale - 32) * (code - 4), code = 2 low bits | high bit << 2, the high
// bit coming from hmask. A half h (128 values) uses qs[32h .. 32h+31]: byte l holds
// the codes of positions l, l+32, l+64, l+96 of the half at shifts 0, 2, 4, 6; the
// high bit of row k is bit 4h + k of hmask[l]. Scale 8h + l/16 + 2k belongs to row
// k. A lane takes l = l0 .. l0+3 (slot = 8h + l0/4), as Q6_K does.

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 110;
const uint LANES = 16;

struct W {
    uint q[4];  // 3-bit codes of row k, one per byte
    int sc[4];
    float d;
};

struct A {
    uint q[4];
    float d[4];
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint slot) {
    const uint h = slot >> 3;
    const uint l0 = 4 * (slot & 7);
    const uint hm = ld4_a2(blk + l0);
    const uint qs = ld4_a2(blk + 32 + 32 * h + l0);
    W w;
    [[unroll]] for (uint k = 0; k < 4; k++) {
        w.q[k] = ((qs >> (2 * k)) & 0x03030303u) | (((hm >> (4 * h + k)) & 0x01010101u) << 2);
    }
    // Scales 8h + p + 2k (p = l0/16): bytes 2(k&1) + p of the words A_{2h + k/2} of
    // the unpacked scales (the same unpacking as the CPU decoder, for the two words
    // this half needs).
    const uint sw0 = ld4_a2(blk + 96);
    const uint sw1 = ld4_a2(blk + 100);
    const uint sw2 = ld4_a2(blk + 104);
    const uint s4 = 4 * h;
    const uint a_lo = ((sw0 >> s4) & 0x0F0F0F0Fu) | (((sw2 >> s4) & 0x03030303u) << 4);
    const uint a_hi = ((sw1 >> s4) & 0x0F0F0F0Fu) | (((sw2 >> (s4 + 2)) & 0x03030303u) << 4);
    const uint p8 = 8 * (l0 >> 4);
    w.sc[0] = int((a_lo >> p8) & 0xFFu) - 32;
    w.sc[1] = int((a_lo >> (16 + p8)) & 0xFFu) - 32;
    w.sc[2] = int((a_hi >> p8) & 0xFFu) - 32;
    w.sc[3] = int((a_hi >> (16 + p8)) & 0xFFu) - 32;
    w.d = half_of(ld2(blk + 108));
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint slot) {
    const uint h = slot >> 3;
    const uint l0 = 4 * (slot & 7);
    A a;
    [[unroll]] for (uint k = 0; k < 4; k++) {
        a.q[k] = ld4(q + 128 * h + 32 * k + l0);
        a.d[k] = ldf1(ds + 8 * (4 * h + k));
    }
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    [[unroll]] for (uint k = 0; k < 4; k++) {
        // (code - 4): the 4 comes off with a second dot product against four bytes
        // of -4.
        const int t = dp_us(w.q[k], a.q[k], dp_ss(0xFCFCFCFCu, a.q[k], 0));
        acc = fma(float(t * w.sc[k]), w.d * a.d[k], acc);
    }
}
