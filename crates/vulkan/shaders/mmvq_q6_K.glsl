// Q6_K for the integer weight kernels (mmvq.glsl); `mmvq_kquants.cuh` is the model.
//
// 256 values in 210 bytes (2-byte aligned).
//   0    u8 ql[128]    low 4 bits
//   128  u8 qh[64]     high 2 bits
//   192  i8 scales[16] one per 16 values
//   208  f16 d
// value = d * scale * (code - 32), code = 6 bits. Each 128-value half h uses
// ql[64h .. 64h+63], qh[32h .. 32h+31] and scales[8h .. 8h+7]; for l in 0..31
//   y[l]    = code(ql[l]    & 15, qh[l] bits 0,1)
//   y[l+32] = code(ql[l+32] & 15, qh[l] bits 2,3)
//   y[l+64] = code(ql[l]    >> 4, qh[l] bits 4,5)
//   y[l+96] = code(ql[l+32] >> 4, qh[l] bits 6,7)
// with the scale scales[8h + l/16 + 2k] for the k-th of these four rows. A lane
// takes l = l0 .. l0+3 of one half (slot = 8h + l0/4): 16 values, four per row,
// from three words (ql[l], ql[l+32], qh[l]).

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 210;
const uint LANES = 16;

struct W {
    uint q[4];  // 6-bit codes of row k, one per byte
    int sc[4];  // scale of row k
    float d;
};

struct A {
    uint q[4];  // activations of row k at l0 .. l0+3
    float d[4];
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

W load_w(uint64_t blk, uint slot) {
    const uint h = slot >> 3;
    const uint l0 = 4 * (slot & 7);
    const uint ql_a = ld4_a2(blk + 64 * h + l0);
    const uint ql_b = ld4_a2(blk + 64 * h + 32 + l0);
    const uint qh = ld4_a2(blk + 128 + 32 * h + l0);
    W w;
    w.q[0] = (ql_a & 0x0F0F0F0Fu) | ((qh << 4) & 0x30303030u);
    w.q[1] = (ql_b & 0x0F0F0F0Fu) | ((qh << 2) & 0x30303030u);
    w.q[2] = ((ql_a >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    w.q[3] = ((ql_b >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u);
    // scales[8h + l0/16 + 2k], k = 0..3: four bytes of the 8 of this half.
    const uint s01 = ld4_a2(blk + 192 + 8 * h);
    const uint s23 = ld4_a2(blk + 192 + 8 * h + 4);
    const uint sh = 8 * (l0 >> 4);  // the low or the high byte of each pair
    w.sc[0] = bitfieldExtract(int(s01), int(sh), 8);
    w.sc[1] = bitfieldExtract(int(s01), int(16 + sh), 8);
    w.sc[2] = bitfieldExtract(int(s23), int(sh), 8);
    w.sc[3] = bitfieldExtract(int(s23), int(16 + sh), 8);
    w.d = half_of(ld2(blk + 208));
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
        // sum of (code - 32) * activation: the 32 comes off with a second dot
        // product against four bytes of -32.
        const int t = dp_us(w.q[k], a.q[k], dp_ss(0xE0E0E0E0u, a.q[k], 0));
        acc = fma(float(t * w.sc[k]), w.d * a.d[k], acc);
    }
}
