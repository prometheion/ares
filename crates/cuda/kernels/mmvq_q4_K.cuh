// Q4_K for the integer weight kernels (mmvq.cuh).
//
// Layout of a block (see matvec_q4_K.cuh): 256 values in 144 bytes, 16-byte
// aligned: a 16-byte header (d and dmin as f16, 12 bytes of packed 6-bit
// scales and minimums of the 8 sub-blocks of 32), then 128 bytes of nibbles.
// Of each 32 bytes the low nibbles are the 32 values of sub-block 2j and the
// high nibbles those of sub-block 2j+1 (j = 0..3, 64 values each).
//
// A lane takes one 16-byte half of one 32-byte group (8 lanes per block):
// 16 bytes are 16 low nibbles (16 values of sub-block 2j) and the 16 high
// nibbles at the same positions (16 values of sub-block 2j+1). Four dp4a on
// the low nibbles and four on the high ones give the integer dot products with
// the matching 16 activations each; the sub-block scales and offsets are applied
// to the two integers once per lane.
//
// value = d * sc * q - dmin * m   (q in 0..15, sc and m 6 bits)
// dot   = sum over sub-blocks of d * sc * dx * sum(q * qx) - dmin * m * sx
// where dx and sx are the scale and the sum of the activation group
// (quantize_act.cuh). The two lanes of a sub-block each add half of the offset
// term, so no lane has to know whether it is the first.
#pragma once

#include "mmvq.cuh"

struct EngQ4K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 144;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    // The nibbles are split once per block here, not once per activation row in
    // `dot_acc`: the compiler would otherwise redo it for every row of the batch.
    struct W {
        uint32_t lo[4];    // low nibbles, one per byte: values of sub-block 2j
        uint32_t hi[4];    // high nibbles left in place (byte = 16 * nibble)
        float d_lo, d_hi;  // d * scale of sub-blocks 2j and 2j+1 (d_hi / 16 for the 16 above)
        float m_lo, m_hi;  // dmin * minimum / 2 of the same sub-blocks
    };

    struct A {
        uint4 lo;   // 16 activations of sub-block 2j
        uint4 hi;   // the same positions of sub-block 2j+1
        float4 ds;  // (d, s) of sub-block 2j, then of 2j+1
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t slot, const uint8_t*) {
        const uint32_t j = slot >> 1;
        const uint32_t h = slot & 1;
        const uint4 hdr = ares_ld_stream_v4(blk);
        const uint4 qs = ares_ld_stream_v4(blk + 16 + 32 * j + 16 * h);
        W w;
        w.lo[0] = qs.x & 0x0F0F0F0Fu;
        w.lo[1] = qs.y & 0x0F0F0F0Fu;
        w.lo[2] = qs.z & 0x0F0F0F0Fu;
        w.lo[3] = qs.w & 0x0F0F0F0Fu;
        w.hi[0] = qs.x & 0xF0F0F0F0u;
        w.hi[1] = qs.y & 0xF0F0F0F0u;
        w.hi[2] = qs.z & 0xF0F0F0F0u;
        w.hi[3] = qs.w & 0xF0F0F0F0u;
        // Keep the split words in registers; without this the compiler recomputes
        // them for every activation row to save registers.
#pragma unroll
        for (int i = 0; i < 4; i++) {
            asm volatile("" : "+r"(w.lo[i]));
            asm volatile("" : "+r"(w.hi[i]));
        }

        // The scales and minimums of sub-blocks 2j and 2j+1 (two bytes each).
        // Sub-blocks 0..3 keep 6 bits in bytes 0..3 (scales) and 4..7 (minimums);
        // sub-blocks 4..7 keep 4 bits in bytes 8..11 and the other 2 in the top
        // bits of bytes 0..7. `hi` selects between the two without a branch.
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
        w.d_hi = d * static_cast<float>(sc >> 8) * (1.0f / 16.0f);
        w.m_lo = 0.5f * dmin * static_cast<float>(mn & 0xFF);
        w.m_hi = 0.5f * dmin * static_cast<float>(mn >> 8);
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t slot) {
        const uint32_t j = slot >> 1;
        const uint32_t h = slot & 1;
        A a;
        a.lo = ares_ld_act_v4(q + 64 * j + 16 * h);
        a.hi = ares_ld_act_v4(q + 64 * j + 32 + 16 * h);
        a.ds = *reinterpret_cast<const float4*>(ds + 2 * j);
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int lo = ares_dp4a_us(w.lo[0], static_cast<int>(a.lo.x), 0);
        lo = ares_dp4a_us(w.lo[1], static_cast<int>(a.lo.y), lo);
        lo = ares_dp4a_us(w.lo[2], static_cast<int>(a.lo.z), lo);
        lo = ares_dp4a_us(w.lo[3], static_cast<int>(a.lo.w), lo);
        // The high nibbles are 16 times too large (bytes up to 240, hence the unsigned
        // weights of dp4a.u32.s32); d_hi already has the 1/16.
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
