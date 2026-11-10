// Q5_K, Q6_K and Q3_K for the integer weight kernels (mmvq.cuh); Q4_K, the
// model of this family, is in mmvq_q4_K.cuh.
//
// Q5_K is Q4_K with a fifth bit per value, so a lane does what a Q4_K lane
// does and adds the bit. Q6_K and Q3_K keep their 6 and 3 bits in two or three
// separate planes and have a sub-block scale per 16 values; a lane takes four
// consecutive positions of each of the four "rows" a 128-value half is built
// from (see the layouts below), which makes every byte it loads whole words.
// Blocks of 210 and 110 bytes are only 2-byte aligned, hence the narrow loads.
#pragma once

#include "mmvq.cuh"

// ---------------------------------------------------------------------------
// Q5_K: 256 values in 176 bytes, 16-byte aligned.
//   0   f16 d, f16 dmin, u8 scales[12]  (as Q4_K)
//   16  u8 qh[32]    byte l holds bit 4 of position l of every sub-block:
//                    sub-block i at bit i
//   48  u8 qs[128]   nibbles as Q4_K
// A lane takes one 16-byte half of a 32-byte group of qs (slot = 2j + h, as
// Q4_K) and the matching 16 bytes of qh.
struct EngQ5K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 176;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t lo[4];    // 5-bit codes of sub-block 2j, one per byte
        uint32_t hi[4];    // 5-bit codes of sub-block 2j+1
        float d_lo, d_hi;  // d * scale
        float m_lo, m_hi;  // dmin * minimum / 2
    };
    using A = EngQ4K::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t slot, const uint8_t*) {
        const uint32_t j = slot >> 1;
        const uint32_t h = slot & 1;
        const uint4 hdr = ares_ld_stream_v4(blk);
        const uint4 qh = ares_ld_stream_v4(blk + 16 + 16 * h);
        const uint4 qs = ares_ld_stream_v4(blk + 48 + 32 * j + 16 * h);
        W w;
        const uint32_t qhw[4] = {qh.x, qh.y, qh.z, qh.w};
        const uint32_t qsw[4] = {qs.x, qs.y, qs.z, qs.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            w.lo[i] = (qsw[i] & 0x0F0F0F0Fu) | (((qhw[i] >> (2 * j)) & 0x01010101u) << 4);
            w.hi[i] = ((qsw[i] >> 4) & 0x0F0F0F0Fu) | (((qhw[i] >> (2 * j + 1)) & 0x01010101u) << 4);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            asm volatile("" : "+r"(w.lo[i]));
            asm volatile("" : "+r"(w.hi[i]));
        }
        // Scales and minimums of the sub-blocks 2j, 2j+1: see EngQ4K::load_w.
        const uint32_t sh = 16 * (j & 1);
        const uint32_t t0 = (hdr.y >> sh) & 0xFFFF;
        const uint32_t t1 = (hdr.z >> sh) & 0xFFFF;
        const uint32_t t2 = (hdr.w >> sh) & 0xFFFF;
        const uint32_t hi = 0u - (j >> 1);
        const uint32_t sc = ((t0 & 0x3F3F) & ~hi) | (((t2 & 0x0F0F) | ((t0 & 0xC0C0) >> 2)) & hi);
        const uint32_t mn = ((t1 & 0x3F3F) & ~hi) | ((((t2 >> 4) & 0x0F0F) | ((t1 & 0xC0C0) >> 2)) & hi);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x)));
        const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
        w.d_lo = d * static_cast<float>(sc & 0xFF);
        w.d_hi = d * static_cast<float>(sc >> 8);
        w.m_lo = 0.5f * dmin * static_cast<float>(mn & 0xFF);
        w.m_hi = 0.5f * dmin * static_cast<float>(mn >> 8);
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t slot) {
        return EngQ4K::load_a(q, ds, slot);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int lo = ares_dp4a_us(w.lo[0], static_cast<int>(a.lo.x), 0);
        lo = ares_dp4a_us(w.lo[1], static_cast<int>(a.lo.y), lo);
        lo = ares_dp4a_us(w.lo[2], static_cast<int>(a.lo.z), lo);
        lo = ares_dp4a_us(w.lo[3], static_cast<int>(a.lo.w), lo);
        int hi = ares_dp4a_us(w.hi[0], static_cast<int>(a.hi.x), 0);
        hi = ares_dp4a_us(w.hi[1], static_cast<int>(a.hi.y), hi);
        hi = ares_dp4a_us(w.hi[2], static_cast<int>(a.hi.z), hi);
        hi = ares_dp4a_us(w.hi[3], static_cast<int>(a.hi.w), hi);
        acc = fmaf(static_cast<float>(lo), w.d_lo * a.ds.x, acc);
        acc = fmaf(static_cast<float>(hi), w.d_hi * a.ds.z, acc);
        acc = fmaf(-w.m_lo, a.ds.y, acc);
        acc = fmaf(-w.m_hi, a.ds.w, acc);
    }
};

// ---------------------------------------------------------------------------
// Q6_K: 256 values in 210 bytes (2-byte aligned).
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
// with the scale scales[8h + l/16 + 2k] for the k-th of these four rows.
// A lane takes l = l0 .. l0+3 of one half (slot = 8h + l0/4): 16 values, four
// per row, from three words (ql[l], ql[l+32], qh[l]).
struct EngQ6K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 210;
    static constexpr int LANES = 16;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t q[4];  // 6-bit codes of row k, one per byte
        int sc[4];      // scale of row k
        float d;
    };
    struct A {
        int q[4];  // activations of row k at l0 .. l0+3
        float d[4];
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t slot, const uint8_t*) {
        const uint32_t h = slot >> 3;
        const uint32_t l0 = 4 * (slot & 7);
        const uint32_t ql_a = ares_ld_stream_u32_a2(blk + 64 * h + l0);
        const uint32_t ql_b = ares_ld_stream_u32_a2(blk + 64 * h + 32 + l0);
        const uint32_t qh = ares_ld_stream_u32_a2(blk + 128 + 32 * h + l0);
        W w;
        w.q[0] = (ql_a & 0x0F0F0F0Fu) | ((qh << 4) & 0x30303030u);
        w.q[1] = (ql_b & 0x0F0F0F0Fu) | ((qh << 2) & 0x30303030u);
        w.q[2] = ((ql_a >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
        w.q[3] = ((ql_b >> 4) & 0x0F0F0F0Fu) | ((qh >> 2) & 0x30303030u);
        // scales[8h + l0/16 + 2k], k = 0..3: four bytes of the 8 of this half.
        const uint32_t s01 = ares_ld_stream_u32_a2(blk + 192 + 8 * h);
        const uint32_t s23 = ares_ld_stream_u32_a2(blk + 192 + 8 * h + 4);
        const uint32_t half = l0 >> 4;  // 0 or 1: the low or the high byte of each pair
        const uint32_t sh = 8 * half;
        w.sc[0] = static_cast<int8_t>(s01 >> sh);
        w.sc[1] = static_cast<int8_t>(s01 >> (16 + sh));
        w.sc[2] = static_cast<int8_t>(s23 >> sh);
        w.sc[3] = static_cast<int8_t>(s23 >> (16 + sh));
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk + 208))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t slot) {
        const uint32_t h = slot >> 3;
        const uint32_t l0 = 4 * (slot & 7);
        A a;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            a.q[k] = *reinterpret_cast<const int*>(q + 128 * h + 32 * k + l0);
            a.d[k] = ds[4 * h + k].x;
        }
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            // sum of (code - 32) * activation: the 32 comes off with a second dp4a
            // against four bytes of -32.
            const int t = ares_dp4a_us(w.q[k], a.q[k], ares_dp4a(static_cast<int>(0xE0E0E0E0u), a.q[k], 0));
            acc = fmaf(static_cast<float>(t * w.sc[k]), w.d * a.d[k], acc);
        }
    }
};

// ---------------------------------------------------------------------------
// Q3_K: 256 values in 110 bytes (2-byte aligned).
//   0    u8 hmask[32]
//   32   u8 qs[64]      2 bits per value
//   96   u8 scales[12]  16 six-bit scales (offset 32): the low 4 bits of
//                       scale i in the nibbles of bytes 0..7 (i < 8: low
//                       nibble of byte i, else high nibble of byte i-8), the
//                       high 2 bits of scale i in bits 2*(i/4) of byte 8 + i%4
//   108  f16 d
// value = d * (scale - 32) * (code - 4), code = 2 low bits | high bit << 2,
// the high bit coming from hmask. A half h (128 values) uses qs[32h .. 32h+31]:
// byte l holds the codes of positions l, l+32, l+64, l+96 of the half at
// shifts 0, 2, 4, 6; the high bit of row k is bit 4h + k of hmask[l]. Scale
// 8h + l/16 + 2k belongs to row k. A lane takes l = l0 .. l0+3 (slot = 8h +
// l0/4), as Q6_K does.
struct EngQ3K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int LANES = 16;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t q[4];  // 3-bit codes of row k, one per byte
        int sc[4];
        float d;
    };
    using A = EngQ6K::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t slot, const uint8_t*) {
        const uint32_t h = slot >> 3;
        const uint32_t l0 = 4 * (slot & 7);
        const uint32_t hm = ares_ld_stream_u32_a2(blk + l0);
        const uint32_t qs = ares_ld_stream_u32_a2(blk + 32 + 32 * h + l0);
        W w;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            w.q[k] = ((qs >> (2 * k)) & 0x03030303u) | (((hm >> (4 * h + k)) & 0x01010101u) << 2);
        }
        // Scales 8h + p + 2k (p = l0/16): bytes 2(k&1) + p of the words A_{2h + k/2}
        // of the unpacked scales (the same unpacking as the CPU decoder, for the
        // two words this half needs).
        const uint32_t sw0 = ares_ld_stream_u32_a2(blk + 96);
        const uint32_t sw1 = ares_ld_stream_u32_a2(blk + 100);
        const uint32_t sw2 = ares_ld_stream_u32_a2(blk + 104);
        const uint32_t s4 = 4 * h;
        const uint32_t a_lo = ((sw0 >> s4) & 0x0F0F0F0Fu) | (((sw2 >> s4) & 0x03030303u) << 4);
        const uint32_t a_hi = ((sw1 >> s4) & 0x0F0F0F0Fu) | (((sw2 >> (s4 + 2)) & 0x03030303u) << 4);
        const uint32_t p8 = 8 * (l0 >> 4);
        w.sc[0] = static_cast<int>((a_lo >> p8) & 0xFF) - 32;
        w.sc[1] = static_cast<int>((a_lo >> (16 + p8)) & 0xFF) - 32;
        w.sc[2] = static_cast<int>((a_hi >> p8) & 0xFF) - 32;
        w.sc[3] = static_cast<int>((a_hi >> (16 + p8)) & 0xFF) - 32;
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk + 108))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t slot) {
        return EngQ6K::load_a(q, ds, slot);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            // (code - 4): the 4 comes off with a second dp4a against four bytes of -4.
            const int t = ares_dp4a_us(w.q[k], a.q[k], ares_dp4a(static_cast<int>(0xFCFCFCFCu), a.q[k], 0));
            acc = fmaf(static_cast<float>(t * w.sc[k]), w.d * a.d[k], acc);
        }
    }
};
