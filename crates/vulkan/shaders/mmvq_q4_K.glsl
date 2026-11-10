// Q4_K for the integer weight kernels (mmvq.glsl); `mmvq_q4_K.cuh` is the model.
//
// Layout of a block: 256 values in 144 bytes, 16-byte aligned: a 16-byte header
// (d and dmin as f16, 12 bytes of packed 6-bit scales and minimums of the 8
// sub-blocks of 32), then 128 bytes of nibbles. Of each 32 bytes the low nibbles
// are the 32 values of sub-block 2j and the high nibbles those of sub-block 2j+1.
//
// A lane takes one 16-byte half of one 32-byte group (8 lanes per block): 16 low
// nibbles (values of sub-block 2j) and the 16 high ones at the same positions
// (sub-block 2j+1). Four dot products on each give the integer products with the
// matching 16 activations; the sub-block scales and offsets are applied to the two
// integers once per lane.
//
// value = d * sc * q - dmin * m   (q in 0..15, sc and m 6 bits)
// dot   = sum over sub-blocks of d * sc * dx * sum(q * qx) - dmin * m * sx
// with dx and sx the scale and the sum of the activation group. The two lanes of a
// sub-block each add half of the offset term.

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 144;
const uint LANES = 8;

struct W {
    uint lo[4];    // low nibbles, one per byte: values of sub-block 2j
    uint hi[4];    // high nibbles left in place (byte = 16 * nibble)
    float d_lo, d_hi;  // d * scale of sub-blocks 2j and 2j+1 (d_hi / 16 for the 16 above)
    float m_lo, m_hi;  // dmin * minimum / 2 of the same sub-blocks
};

struct A {
    uvec4 lo;  // 16 activations of sub-block 2j
    uvec4 hi;  // the same positions of sub-block 2j+1
    vec4 ds;   // (d, s) of sub-block 2j, then of 2j+1
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

// The scales and minimums of sub-blocks 2j and 2j+1 (two bytes each) from the three
// words of packed scales. Sub-blocks 0..3 keep 6 bits in bytes 0..3 (scales) and
// 4..7 (minimums); sub-blocks 4..7 keep 4 bits in bytes 8..11 and the other 2 in
// the top bits of bytes 0..7. `hi` selects between the two without a branch.
void scales_k(uint j, uvec3 s, out uint sc, out uint mn) {
    const uint sh = 16 * (j & 1);
    const uint t0 = (s.x >> sh) & 0xFFFFu;
    const uint t1 = (s.y >> sh) & 0xFFFFu;
    const uint t2 = (s.z >> sh) & 0xFFFFu;
    const uint hi = 0u - (j >> 1);
    sc = ((t0 & 0x3F3Fu) & ~hi) | (((t2 & 0x0F0Fu) | ((t0 & 0xC0C0u) >> 2)) & hi);
    mn = ((t1 & 0x3F3Fu) & ~hi) | ((((t2 >> 4) & 0x0F0Fu) | ((t1 & 0xC0C0u) >> 2)) & hi);
}

W load_w(uint64_t blk, uint slot) {
    const uint j = slot >> 1;
    const uint h = slot & 1;
    const uvec4 hdr = ldw16(blk);
    const uvec4 qs = ldw16(blk + 16 + 32 * j + 16 * h);
    W w;
    w.lo[0] = qs.x & 0x0F0F0F0Fu;
    w.lo[1] = qs.y & 0x0F0F0F0Fu;
    w.lo[2] = qs.z & 0x0F0F0F0Fu;
    w.lo[3] = qs.w & 0x0F0F0F0Fu;
    w.hi[0] = qs.x & 0xF0F0F0F0u;
    w.hi[1] = qs.y & 0xF0F0F0F0u;
    w.hi[2] = qs.z & 0xF0F0F0F0u;
    w.hi[3] = qs.w & 0xF0F0F0F0u;
    uint sc, mn;
    scales_k(j, hdr.yzw, sc, mn);
    const vec2 dd = unpackHalf2x16(hdr.x);
    w.d_lo = dd.x * float(sc & 0xFFu);
    w.d_hi = dd.x * float(sc >> 8) * (1.0 / 16.0);
    w.m_lo = 0.5 * dd.y * float(mn & 0xFFu);
    w.m_hi = 0.5 * dd.y * float(mn >> 8);
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint slot) {
    const uint j = slot >> 1;
    const uint h = slot & 1;
    A a;
    a.lo = ld16(q + 64 * j + 16 * h);
    a.hi = ld16(q + 64 * j + 32 + 16 * h);
    a.ds = ldf4(ds + 16 * j);
    return a;
}

void dot_acc(inout float acc, W w, A a) {
    int lo = dp_us(w.lo[0], a.lo.x, 0);
    lo = dp_us(w.lo[1], a.lo.y, lo);
    lo = dp_us(w.lo[2], a.lo.z, lo);
    lo = dp_us(w.lo[3], a.lo.w, lo);
    // The high nibbles are 16 times too large (bytes up to 240, hence the unsigned
    // weights); d_hi already has the 1/16.
    int hi = dp_us(w.hi[0], a.hi.x, 0);
    hi = dp_us(w.hi[1], a.hi.y, hi);
    hi = dp_us(w.hi[2], a.hi.z, hi);
    hi = dp_us(w.hi[3], a.hi.w, hi);
    acc = fma(float(lo), w.d_lo * a.ds.x, acc);
    acc = fma(float(hi), w.d_hi * a.ds.z, acc);
    acc = fma(-w.m_lo, a.ds.y, acc);
    acc = fma(-w.m_hi, a.ds.w, acc);
}
