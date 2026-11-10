// Q5_K for the integer weight kernels (mmvq.glsl): Q4_K with a fifth bit per value,
// so a lane does what a Q4_K lane does and adds the bit. `mmvq_kquants.cuh` is the
// model.
//
// 256 values in 176 bytes, 16-byte aligned.
//   0   f16 d, f16 dmin, u8 scales[12]  (as Q4_K)
//   16  u8 qh[32]    byte l holds bit 4 of position l of every sub-block:
//                    sub-block i at bit i
//   48  u8 qs[128]   nibbles as Q4_K
// A lane takes one 16-byte half of a 32-byte group of qs (slot = 2j + h, as Q4_K)
// and the matching 16 bytes of qh.

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 176;
const uint LANES = 8;

struct W {
    uint lo[4];  // 5-bit codes of sub-block 2j, one per byte
    uint hi[4];  // 5-bit codes of sub-block 2j+1
    float d_lo, d_hi;  // d * scale
    float m_lo, m_hi;  // dmin * minimum / 2
};

struct A {
    uvec4 lo;
    uvec4 hi;
    vec4 ds;
};

uint mmvq_lane;
void mmvq_init(uint t, uint64_t tables) {}

// See mmvq_q4_K.glsl.
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
    const uvec4 qh = ldw16(blk + 16 + 16 * h);
    const uvec4 qs = ldw16(blk + 48 + 32 * j + 16 * h);
    W w;
    [[unroll]] for (uint i = 0; i < 4; i++) {
        w.lo[i] = (qs[i] & 0x0F0F0F0Fu) | (((qh[i] >> (2 * j)) & 0x01010101u) << 4);
        w.hi[i] = ((qs[i] >> 4) & 0x0F0F0F0Fu) | (((qh[i] >> (2 * j + 1)) & 0x01010101u) << 4);
    }
    uint sc, mn;
    scales_k(j, hdr.yzw, sc, mn);
    const vec2 dd = unpackHalf2x16(hdr.x);
    w.d_lo = dd.x * float(sc & 0xFFu);
    w.d_hi = dd.x * float(sc >> 8);
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
    int hi = dp_us(w.hi[0], a.hi.x, 0);
    hi = dp_us(w.hi[1], a.hi.y, hi);
    hi = dp_us(w.hi[2], a.hi.z, hi);
    hi = dp_us(w.hi[3], a.hi.w, hi);
    acc = fma(float(lo), w.d_lo * a.ds.x, acc);
    acc = fma(float(hi), w.d_hi * a.ds.z, acc);
    acc = fma(-w.m_lo, a.ds.y, acc);
    acc = fma(-w.m_hi, a.ds.w, acc);
}
