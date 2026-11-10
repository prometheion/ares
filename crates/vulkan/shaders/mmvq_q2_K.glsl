// Q2_K for the integer weight kernels (mmvq.glsl).
//
// 256 values in 84 bytes (4-byte aligned).
//   0   u8 scales[16]  per sub-block of 16 values: scale in the low nibble, minimum
//                      in the high nibble
//   16  u8 qs[64]      2-bit codes: each half h of 128 values has 32 bytes, byte l
//                      holds the codes of values 32 step + l at shift 2 step
//   80  f16 d, f16 dmin
// value = d * sc * code - dmin * m. A lane takes one sub-block (slot = 8h + 2 step +
// part, which is also the index of its scale byte): 16 bytes of qs at 16 + 32 h + 16
// part, shifted by 2 step. The minimum multiplies the sum of the 16 activations, so
// the lane takes that with four dot products against ones.

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 84;
const uint LANES = 16;

struct W {
    uint c[4];  // codes, one per byte
    float dsc;  // d * scale
    float dm;   // dmin * minimum
};

struct A {
    uvec4 q;
    float dx;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint s) {
    const uint h = s >> 3;
    const uint step = (s >> 1) & 3u;
    const uint part = s & 1u;
    const uint64_t q = blk + 16 + 32 * h + 16 * part;
    W w;
    [[unroll]] for (uint i = 0; i < 4; i++) {
        w.c[i] = (ld4(q + 4 * i) >> (2 * step)) & 0x03030303u;
    }
    const uint sc = ld1(blk + s);
    const uint head = ld4(blk + 80);
    w.dsc = half_of(head) * float(sc & 0xFu);
    w.dm = half_of(head >> 16) * float(sc >> 4);
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint s) {
    const uint h = s >> 3;
    const uint step = (s >> 1) & 3u;
    const uint part = s & 1u;
    A a;
    a.q = ld16(q + 128 * h + 32 * step + 16 * part);
    a.dx = ldf1(ds + 8 * (4 * h + step));
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int isum = dp_us(w.c[0], a.q.x, 0);
    isum = dp_us(w.c[1], a.q.y, isum);
    isum = dp_us(w.c[2], a.q.z, isum);
    isum = dp_us(w.c[3], a.q.w, isum);
    int sa = dp_ss(0x01010101u, a.q.x, 0);
    sa = dp_ss(0x01010101u, a.q.y, sa);
    sa = dp_ss(0x01010101u, a.q.z, sa);
    sa = dp_ss(0x01010101u, a.q.w, sa);
    acc = fma(float(isum), w.dsc * a.dx, acc);
    acc = fma(-float(sa), w.dm * a.dx, acc);
}
