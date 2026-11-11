// The remaining quantized types for the integer weight kernels (mmvq.cuh):
// the legacy 4- and 5-bit types, Q2_K, IQ2_XXS, the 1.5-bit IQ1 types, the two
// ternary types, the 4-bit floats, and the 1- and 2-bit sign types. Each type
// is a lane's share of a block in the form `mmvq.cuh` describes; where a type
// could be done two ways, the comment says what was chosen and why.
//
// The common thread: whatever the codes mean, a lane turns its share into
// bytes that `dp4a` can multiply with the int8 activations, and everything that
// is constant per block or per sub-block (scales, offsets, minimums) is applied
// to the integer sum, once. An offset that every value carries (the 8 of Q4_0, the
// -1 of the ternary types, a minimum) is applied through the sum of the
// activations that the quantizer stored (`s` of `ds`, quantize_act.cuh) or
// through a `dp4a` against bytes of 1, never per value.
#pragma once

#include "fp4_scales.cuh"
#include "mmvq.cuh"
#include "mmvq_iquants.cuh"
#include "mmvq_kquants.cuh"

// ---------------------------------------------------------------------------
// Helpers

// The table lookup of `ares_table16` for any table of 16 signed bytes, given as
// four registers of four bytes (entry 0 in the least significant byte).
static __device__ __forceinline__ int2
ares_table16_with(uint32_t q4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    uint32_t tmp[2];
    const uint32_t pick = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const uint32_t sh = 16 * i;
        const uint32_t low = __byte_perm(t0, t1, q4 >> sh);
        const uint32_t high = __byte_perm(t2, t3, q4 >> sh);
        tmp[i] = __byte_perm(low, high, pick >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// Eight bytes at an address of any alignment (MXFP4 blocks are 17 bytes). Three
// aligned words and two funnel shifts; the third word is only read when the
// bytes need it.
static __device__ __forceinline__ uint2 ares_ld_stream_v2_any(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* base = reinterpret_cast<const uint32_t*>(a & ~static_cast<uintptr_t>(3));
    const uint32_t shift = 8 * static_cast<uint32_t>(a & 3);
    const uint32_t w0 = ares_ld_stream_u32(base);
    const uint32_t w1 = ares_ld_stream_u32(base + 1);
    const uint32_t w2 = shift ? ares_ld_stream_u32(base + 2) : 0;
    return make_uint2(__funnelshift_r(w0, w1, shift), __funnelshift_r(w1, w2, shift));
}

// The four bits of `n` as four bytes of 0 or 1 (bit i in byte i): the sum of
// the shifted copies puts bit i at bit 8 i and nothing else on those places.
static __device__ __forceinline__ uint32_t ares_spread_bits4(uint32_t n) {
    return (n * 0x00204081u) & 0x01010101u;
}

// The four 2-bit fields of a byte as four bytes (field i in byte i). Each field is
// moved on its own: unlike ares_spread_bits4 the shifted copies of a whole byte
// overlap, and their sums would carry into the neighbours.
static __device__ __forceinline__ uint32_t ares_spread_crumbs4(uint32_t b) {
    return (b & 0x03u) | ((b & 0x0Cu) << 6) | ((b & 0x30u) << 12) | ((b & 0xC0u) << 18);
}

static __device__ __forceinline__ float ares_f16_lo(uint32_t v) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(v)));
}

static __device__ __forceinline__ float ares_f16_hi(uint32_t v) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(v >> 16)));
}

// The same activations that the legacy 32-value blocks read: this lane's two
// halves of 8 values (see Q4_0), and the (d, s) pair of the block's group.
struct AresHalvesAct {
    uint2 first;   // activations 8s .. 8s+7
    uint2 second;  // 16+8s .. 16+8s+7
    float2 ds;
    static __device__ __forceinline__ AresHalvesAct load(const int8_t* q, const float2* ds, uint32_t s) {
        AresHalvesAct a;
        a.first = *reinterpret_cast<const uint2*>(q + 8 * s);
        a.second = *reinterpret_cast<const uint2*>(q + 16 + 8 * s);
        a.ds = ds[0];
        return a;
    }
};

// ---------------------------------------------------------------------------
// The legacy 4- and 5-bit types: 32 values in a block of an f16 d, optionally an f16
// minimum m, optionally a u32 qh with the fifth bit of every value, and 16 bytes
// of nibbles (low nibbles are values 0..15, high nibbles 16..31):
//
//             d  m  qh   bytes   value
//   Q4_0      x         18      d * (q - 8)
//   Q4_1      x  x      20      d * q + m
//   Q5_0      x     x   22      d * (q - 16)        q of 5 bits
//   Q5_1      x  x  x   24      d * q + m           q of 5 bits
//
// Value j has its fifth bit in bit j of qh (j < 16: low nibbles) and the value
// j + 16 in bit j + 16. Two lanes per block, 8 bytes of nibbles each. What every
// value carries (the -8 and -16 of the types with no minimum, the minimum of the
// others) is applied through the sum of the activations that the quantizer stored;
// each of the two lanes takes half of it. The blocks with a minimum are 4-byte
// aligned and load whole words, the others 2-byte aligned and load halves.
template <bool HAS_M, bool HAS_QH>
struct EngLegacy4 {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 2 + (HAS_M ? 2 : 0) + (HAS_QH ? 4 : 0) + 16;
    static constexpr int LANES = 2;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t lo[2], hi[2];
        float d, m;  // m: half of what every value carries, in units of the activations' sum
    };
    using A = AresHalvesAct;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t s, const uint8_t*) {
        constexpr int QH_AT = HAS_M ? 4 : 2;
        constexpr int QS_AT = QH_AT + (HAS_QH ? 4 : 0);
        W w;
        uint32_t q0, q1, qh = 0;
        if constexpr (HAS_M) {  // 4-byte aligned
            const uint32_t dm = ares_ld_stream_u32(blk);
            w.d = ares_f16_lo(dm);
            w.m = 0.5f * ares_f16_hi(dm);
            if constexpr (HAS_QH) {
                qh = ares_ld_stream_u32(blk + QH_AT);
            }
            q0 = ares_ld_stream_u32(blk + QS_AT + 8 * s);
            q1 = ares_ld_stream_u32(blk + QS_AT + 8 * s + 4);
        } else {  // 2-byte aligned
            w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
            w.m = -(HAS_QH ? 8.0f : 4.0f) * w.d;  // (8 or 16) * d, half of it
            if constexpr (HAS_QH) {
                qh = ares_ld_stream_u32_a2(blk + QH_AT);
            }
            const uint2 q = ares_ld_stream_v2_a2(blk + QS_AT + 8 * s);
            q0 = q.x;
            q1 = q.y;
        }
        const uint32_t word[2] = {q0, q1};
#pragma unroll
        for (int i = 0; i < 2; i++) {
            w.lo[i] = word[i] & 0x0F0F0F0Fu;
            w.hi[i] = (word[i] >> 4) & 0x0F0F0F0Fu;
            if constexpr (HAS_QH) {
                w.lo[i] |= ares_spread_bits4((qh >> (8 * s + 4 * i)) & 0xF) << 4;
                w.hi[i] |= ares_spread_bits4((qh >> (16 + 8 * s + 4 * i)) & 0xF) << 4;
            }
        }
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t s) {
        return A::load(q, ds, s);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a_us(w.lo[0], static_cast<int>(a.first.x), 0);
        s = ares_dp4a_us(w.lo[1], static_cast<int>(a.first.y), s);
        s = ares_dp4a_us(w.hi[0], static_cast<int>(a.second.x), s);
        s = ares_dp4a_us(w.hi[1], static_cast<int>(a.second.y), s);
        acc = fmaf(static_cast<float>(s), w.d * a.ds.x, acc);
        acc = fmaf(w.m, a.ds.y, acc);
    }
};

using EngQ40 = EngLegacy4<false, false>;
using EngQ41 = EngLegacy4<true, false>;
using EngQ50 = EngLegacy4<false, true>;
using EngQ51 = EngLegacy4<true, true>;

// ---------------------------------------------------------------------------
// MXFP4: 32 values in 17 bytes (byte aligned): an E8M0 exponent byte, then 16
// bytes of nibbles as Q4_0, each an index into the 16 levels of E2M1 (doubled,
// so that they are integers). value = level * 2^(e - 128). The blocks are 17
// bytes, so the weights are read as aligned words and shifted into place.
struct EngMXFP4 {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 17;
    static constexpr int LANES = 2;
    static constexpr int SMEM_BYTES = 0;
    static constexpr int UNROLL = 2;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        int x[2], y[2];
        float d;
    };
    using A = AresHalvesAct;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t s, const uint8_t*) {
        const uint32_t e = ares_ld_stream_u8(blk);
        const uint2 q = ares_ld_stream_v2_any(blk + 1 + 8 * s);
        const uint32_t t0 = ARES_PACK4(0, 1, 2, 3), t1 = ARES_PACK4(4, 6, 8, 12);
        const uint32_t t2 = ARES_PACK4(0, -1, -2, -3), t3 = ARES_PACK4(-4, -6, -8, -12);
        const int2 v0 = ares_table16_with(q.x, t0, t1, t2, t3);
        const int2 v1 = ares_table16_with(q.y, t0, t1, t2, t3);
        W w;
        w.x[0] = v0.x;
        w.y[0] = v0.y;
        w.x[1] = v1.x;
        w.y[1] = v1.y;
        w.d = ares_e8m0_half(e);
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t s) {
        return A::load(q, ds, s);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(w.x[0], static_cast<int>(a.first.x), 0);
        s = ares_dp4a(w.x[1], static_cast<int>(a.first.y), s);
        s = ares_dp4a(w.y[0], static_cast<int>(a.second.x), s);
        s = ares_dp4a(w.y[1], static_cast<int>(a.second.y), s);
        acc = fmaf(static_cast<float>(s), w.d * a.ds.x, acc);
    }
};

// NVFP4: 64 values in 36 bytes (4-byte aligned): four UE4M3 scale bytes, then
// four sub-blocks of 16 values in 8 bytes each (low nibbles the first 8 values,
// high nibbles the next 8), through the same 16 levels. A lane takes a sub-block.
struct EngNVFP4 {
    static constexpr int BLOCK_VALUES = 64;
    static constexpr int BLOCK_BYTES = 36;
    static constexpr int LANES = 4;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        int x[2], y[2];
        float d;
    };
    struct A {
        uint4 q;  // the 16 activations of the sub-block
        float2 ds;
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t s, const uint8_t*) {
        const uint32_t q0 = ares_ld_stream_u32(blk + 4 + 8 * s);
        const uint32_t q1 = ares_ld_stream_u32(blk + 8 + 8 * s);
        const uint32_t t0 = ARES_PACK4(0, 1, 2, 3), t1 = ARES_PACK4(4, 6, 8, 12);
        const uint32_t t2 = ARES_PACK4(0, -1, -2, -3), t3 = ARES_PACK4(-4, -6, -8, -12);
        const int2 v0 = ares_table16_with(q0, t0, t1, t2, t3);
        const int2 v1 = ares_table16_with(q1, t0, t1, t2, t3);
        W w;
        w.x[0] = v0.x;  // values 0..3 of the sub-block
        w.x[1] = v1.x;  // 4..7
        w.y[0] = v0.y;  // 8..11
        w.y[1] = v1.y;  // 12..15
        w.d = ares_ue4m3_half(ares_ld_stream_u8(blk + s));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t s) {
        A a;
        a.q = ares_ld_act_v4(q + 16 * s);
        a.ds = ds[s >> 1];
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(w.x[0], static_cast<int>(a.q.x), 0);
        s = ares_dp4a(w.x[1], static_cast<int>(a.q.y), s);
        s = ares_dp4a(w.y[0], static_cast<int>(a.q.z), s);
        s = ares_dp4a(w.y[1], static_cast<int>(a.q.w), s);
        acc = fmaf(static_cast<float>(s), w.d * a.ds.x, acc);
    }
};

// ---------------------------------------------------------------------------
// Q2_K: 256 values in 84 bytes (4-byte aligned): 16 bytes of scales (low nibble
// the scale, high nibble the minimum of a sub-block of 16), 64 bytes of 2-bit
// codes, f16 d and f16 dmin. value = d * sc * q - dmin * m.
// The layout is Q3_K's: a half (128 values) uses qs[32h .. 32h+31]; byte l holds
// the codes of positions l, l+32, l+64, l+96 of the half at shifts 0, 2, 4, 6,
// and the sub-block of row k at l is 8h + 2k + l/16. A lane takes l0 .. l0+3
// (slot = 8h + l0/4). The minimum multiplies the sum of the lane's four
// activations, one more dp4a against bytes of 1.
struct EngQ2K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 84;
    static constexpr int LANES = 16;
    static constexpr int SMEM_BYTES = 0;
    static constexpr int UNROLL = 2;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t q[4];
        float d_sc[4];   // d * scale of row k
        float dmin_m[4];  // dmin * minimum of row k
    };
    using A = EngQ6K::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t slot, const uint8_t*) {
        const uint32_t h = slot >> 3;
        const uint32_t l0 = 4 * (slot & 7);
        const uint32_t word = ares_ld_stream_u32(blk + 16 + 32 * h + l0);
        const uint32_t s01 = ares_ld_stream_u32(blk + 8 * h);
        const uint32_t s23 = ares_ld_stream_u32(blk + 8 * h + 4);
        const uint32_t dd = ares_ld_stream_u32(blk + 80);
        const float d = ares_f16_lo(dd);
        const float dmin = ares_f16_hi(dd);
        const uint32_t sh = 8 * (l0 >> 4);
        const uint32_t sc[4] = {
            (s01 >> sh) & 0xFF, (s01 >> (16 + sh)) & 0xFF, (s23 >> sh) & 0xFF, (s23 >> (16 + sh)) & 0xFF};
        W w;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            w.q[k] = (word >> (2 * k)) & 0x03030303u;
            w.d_sc[k] = d * static_cast<float>(sc[k] & 15);
            w.dmin_m[k] = dmin * static_cast<float>(sc[k] >> 4);
        }
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t slot) {
        return EngQ6K::load_a(q, ds, slot);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int t = ares_dp4a_us(w.q[k], a.q[k], 0);
            const int u = ares_dp4a(0x01010101, a.q[k], 0);
            acc = fmaf(static_cast<float>(t), w.d_sc[k] * a.d[k], acc);
            acc = fmaf(-static_cast<float>(u), w.dmin_m[k] * a.d[k], acc);
        }
    }
};

// ---------------------------------------------------------------------------
// IQ2_XXS: 256 values in 66 bytes (2-byte aligned): f16 d, then per sub-block of
// 32 values 8 bytes: four codebook indices (256 groups of 8 magnitudes) and a
// u32 with four 7-bit sign indices and a 4-bit scale on top.
// value = d * (0.5 + scale) * 0.25 * magnitude * sign. Slot = sub-block.
struct EngIQ2XXS {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 66;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 256 * 8;
    static constexpr int UNROLL = 2;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        ares_copy_table(smem, ares_iq2xxs_grid, SMEM_BYTES, t, n);
    }

    using W = EngIQ2XS::W;
    using A = EngIQ4XS::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t* smem) {
        const uint2* grid = reinterpret_cast<const uint2*>(smem);
        const uint2 v = ares_ld_stream_v2_a2(blk + 2 + 8 * ib);
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 g = grid[(v.x >> (8 * l)) & 0xFF];
            const uint32_t spread = ares_ksigns((v.y >> (7 * l)) & 127) * 0x01010101u;
            w.w[2 * l] = ares_apply_sign4(g.x, ares_sign_mask4(spread, 0x08040201u));
            w.w[2 * l + 1] = ares_apply_sign4(g.y, ares_sign_mask4(spread, 0x80402010u));
        }
        w.scale2[0] = static_cast<int>(2 * (v.y >> 28) + 1);
        w.scale2[1] = w.scale2[0];
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        return EngIQ4XS::load_a(q, ds, ib);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        EngIQ2XS::dot_acc(acc, w, a);  // two equal scales: the same sum
    }
};

// ---------------------------------------------------------------------------
// IQ1_S: 256 values in 50 bytes (2-byte aligned): f16 d, 32 bytes of low index
// bits, 8 u16 of high index bits and scales. Sub-block ib of 32 values has
// four groups of 8: index qs[4 ib + l] | (3 bits of qh[ib]) << 8 into a
// codebook of 2048 groups of bytes in {-1, 0, 1}, a scale 2 * ((qh >> 12) & 7) + 1
// and a shift delta of +-1/8 (bit 15 of qh) added to every value.
// value = d * scale * (grid + delta). The shift multiplies the sum of the
// sub-block's activations: the stored `s`.
struct EngIQ1S {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 50;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 2048 * 8;
    static constexpr int UNROLL = 4;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        ares_copy_table(smem, ares_iq1s_grid, SMEM_BYTES, t, n);
    }

    struct W {
        int w[8];
        float scale;  // d * (2 * s3 + 1)
        float delta;  // +-1/8
    };
    using A = EngIQ4XS::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t* smem) {
        const uint2* grid = reinterpret_cast<const uint2*>(smem);
        const uint32_t qs = ares_ld_stream_u32_a2(blk + 2 + 4 * ib);
        const uint32_t qh = ares_ld_stream_u16(blk + 34 + 2 * ib);
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 g = grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
            w.w[2 * l] = static_cast<int>(g.x);
            w.w[2 * l + 1] = static_cast<int>(g.y);
        }
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        w.scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
        w.delta = (qh & 0x8000) ? -0.125f : 0.125f;
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        return EngIQ4XS::load_a(q, ds, ib);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(w.w[0], static_cast<int>(a.first.x), 0);
        s = ares_dp4a(w.w[1], static_cast<int>(a.first.y), s);
        s = ares_dp4a(w.w[2], static_cast<int>(a.first.z), s);
        s = ares_dp4a(w.w[3], static_cast<int>(a.first.w), s);
        s = ares_dp4a(w.w[4], static_cast<int>(a.second.x), s);
        s = ares_dp4a(w.w[5], static_cast<int>(a.second.y), s);
        s = ares_dp4a(w.w[6], static_cast<int>(a.second.z), s);
        s = ares_dp4a(w.w[7], static_cast<int>(a.second.w), s);
        acc = fmaf(static_cast<float>(s), w.scale * a.ds.x, acc);
        acc = fmaf(w.scale * w.delta, a.ds.y, acc);
    }
};

// IQ1_M: 256 values in 56 bytes (8-byte aligned): 32 bytes of low index bits, 16
// bytes of high index bits and shift signs, 4 u16 of scales whose top nibbles
// spread the f16 d of the block. Like IQ1_S, but a scale (3 bits) for every 16
// values and a shift sign for every 8; so the sums of activations that the shift
// needs are per group of 8, found with dp4a against bytes of 1.
struct EngIQ1M {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 56;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 2048 * 8;
    static constexpr int UNROLL = 2;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        ares_copy_table(smem, ares_iq1s_grid, SMEM_BYTES, t, n);
    }

    struct W {
        int w[8];
        float scale[4];  // d * (2 * s3 + 1) of group l
        float delta[4];  // +-1/8 of group l
    };
    using A = EngIQ4XS::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t* smem) {
        const uint2* grid = reinterpret_cast<const uint2*>(smem);
        const uint32_t qs = ares_ld_stream_u32(blk + 4 * ib);
        const uint32_t qh = ares_ld_stream_u16(blk + 32 + 2 * ib);
        const uint2 sc = ares_ld_stream_v2(blk + 48);
        const uint32_t sc0 = sc.x & 0xFFFF, sc1 = sc.x >> 16, sc2 = sc.y & 0xFFFF, sc3 = sc.y >> 16;
        const uint32_t d16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(d16)));
        const uint32_t word = (ib >> 1) == 0 ? sc.x & 0xFFFF : (ib >> 1) == 1 ? sc.x >> 16 : (ib >> 1) == 2 ? sc.y & 0xFFFF : sc.y >> 16;
        const uint32_t shift = 6 * (ib & 1);
        const float dl[2] = {d * static_cast<float>(2 * ((word >> shift) & 7) + 1),
                             d * static_cast<float>(2 * ((word >> (shift + 3)) & 7) + 1)};
        const uint32_t h0 = qh & 0xFF, h1 = qh >> 8;
        const uint32_t high[4] = {(h0 << 8) & 0x700, (h0 << 4) & 0x700, (h1 << 8) & 0x700, (h1 << 4) & 0x700};
        const bool neg[4] = {(h0 & 0x08) != 0, (h0 & 0x80) != 0, (h1 & 0x08) != 0, (h1 & 0x80) != 0};
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 g = grid[((qs >> (8 * l)) & 0xFF) | high[l]];
            w.w[2 * l] = static_cast<int>(g.x);
            w.w[2 * l + 1] = static_cast<int>(g.y);
            w.scale[l] = dl[l >> 1];
            w.delta[l] = neg[l] ? -0.125f : 0.125f;
        }
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        return EngIQ4XS::load_a(q, ds, ib);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        const int av[8] = {static_cast<int>(a.first.x), static_cast<int>(a.first.y), static_cast<int>(a.first.z),
                           static_cast<int>(a.first.w), static_cast<int>(a.second.x), static_cast<int>(a.second.y),
                           static_cast<int>(a.second.z), static_cast<int>(a.second.w)};
#pragma unroll
        for (int l = 0; l < 4; l++) {
            int s = ares_dp4a(w.w[2 * l], av[2 * l], 0);
            s = ares_dp4a(w.w[2 * l + 1], av[2 * l + 1], s);
            int sum = ares_dp4a(0x01010101, av[2 * l], 0);
            sum = ares_dp4a(0x01010101, av[2 * l + 1], sum);
            acc = fmaf(w.scale[l] * a.ds.x, static_cast<float>(s) + w.delta[l] * static_cast<float>(sum), acc);
        }
    }
};

// ---------------------------------------------------------------------------
// TQ2_0: 256 values in 66 bytes (2-byte aligned): 64 bytes of 2-bit codes 0, 1, 2
// for -1, 0, +1, then f16 d. Two runs of 32 bytes; in a run, plane l (shift 2 l)
// of byte m is the value 128 * run + 32 * l + m. value = d * (code - 1). A lane
// takes four bytes of a run (slot = 8 * run + i), four planes of four values.
// The -1 is a dp4a against bytes of -1 on the same activations.
struct EngTQ20 {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 66;
    static constexpr int LANES = 16;
    static constexpr int SMEM_BYTES = 0;
    static constexpr int UNROLL = 2;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t code[4];
        float d;
    };
    struct A {
        int q[4];
        float dx[4];
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t slot, const uint8_t*) {
        const uint32_t word = ares_ld_stream_u32_a2(blk + 32 * (slot >> 3) + 4 * (slot & 7));
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            w.code[l] = (word >> (2 * l)) & 0x03030303u;
        }
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk + 64))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t slot) {
        const uint32_t run = slot >> 3;
        const uint32_t i = slot & 7;
        A a;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            a.q[l] = *reinterpret_cast<const int*>(q + 128 * run + 32 * l + 4 * i);
            a.dx[l] = ds[4 * run + l].x;
        }
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const int t = ares_dp4a_us(w.code[l], a.q[l], ares_dp4a(-1, a.q[l], 0));
            acc = fmaf(static_cast<float>(t), w.d * a.dx[l], acc);
        }
    }
};

// TQ1_0: 256 values in 54 bytes (2-byte aligned): 48 bytes of qs, 4 bytes of qh,
// f16 d. Five values to a byte (a base-3 number: the digit n of byte b is
// (b * 3^n mod 256) * 3 >> 8, in 0..2 for -1..+1), taken in planes: bytes 0..31
// give the values 32 n + m, bytes 32..47 the values 160 + 16 n + m (n < 5), and
// the four bytes of qh give 240 + 4 n + j (n < 4). value = d * (digit - 1).
// A lane takes four bytes of one of the three runs (slots 0..7, 8..11, 12; the
// other three of the 16 do nothing). The kernel is limited by its activation loads, five
// words and five scales at different places per lane (making them uniform makes the
// kernel twice as fast); a table of the digits in shared memory instead of the
// arithmetic below made no difference. A lane decodes the digits of all its planes
// four at a time (two values per 16 bits, so the products do not carry).
struct EngTQ10 {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 54;
    static constexpr int LANES = 16;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t dig[5];  // digits 0..2 of plane n, one per byte
        float d;
    };
    struct A {
        int q[5];
        float dx[5];
    };

    static __device__ __forceinline__ uint32_t digits(uint32_t word, uint32_t pow3) {
        const uint32_t even = (((word & 0x00FF00FFu) * pow3) & 0x00FF00FFu) * 3;
        const uint32_t odd = ((((word >> 8) & 0x00FF00FFu) * pow3) & 0x00FF00FFu) * 3;
        return ((even >> 8) & 0x00030003u) | (((odd >> 8) & 0x00030003u) << 8);
    }

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t slot, const uint8_t*) {
        W w;
        const uint32_t word = slot < 8    ? ares_ld_stream_u32_a2(blk + 4 * slot)
                              : slot < 12 ? ares_ld_stream_u32_a2(blk + 32 + 4 * (slot - 8))
                              : slot == 12 ? ares_ld_stream_u32_a2(blk + 48)
                                           : 0;
        const uint32_t pow3[5] = {1, 3, 9, 27, 81};
        // Idle lanes and the missing fifth plane of the qh run get the digit 1 (value
        // 0) in every byte, so that they add nothing.
        const uint32_t planes = slot < 12 ? 5 : (slot == 12 ? 4 : 0);
#pragma unroll
        for (int n = 0; n < 5; n++) {
            w.dig[n] = n < planes ? digits(word, pow3[n]) : 0x01010101u;
        }
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk + 52))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t slot) {
        A a;
#pragma unroll
        for (int n = 0; n < 5; n++) {
            // The value offset of plane n; for an idle lane or the missing fifth plane of the
            // qh run any valid one (their digits are 1: see load_w).
            uint32_t off = 0;
            if (slot < 8) {
                off = 32 * n + 4 * slot;
            } else if (slot < 12) {
                off = 160 + 16 * n + 4 * (slot - 8);
            } else if (slot == 12 && n < 4) {
                off = 240 + 4 * n;
            }
            a.q[n] = *reinterpret_cast<const int*>(q + off);
            a.dx[n] = ds[off >> 5].x;
        }
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
#pragma unroll
        for (int n = 0; n < 5; n++) {
            const int t = ares_dp4a_us(w.dig[n], a.q[n], ares_dp4a(-1, a.q[n], 0));
            acc = fmaf(static_cast<float>(t), w.d * a.dx[n], acc);
        }
    }
};

// ---------------------------------------------------------------------------
// Q1_0: 128 values in 18 bytes (2-byte aligned): f16 d, 16 bytes of signs (bit j
// set: +d, clear: -d; bit j is value j). The product is sums and differences of
// activations: with T the sum of the activations whose bit is set and S the sum
// of all, d * (2 T - S). T is a dp4a of the bit bytes (0 or 1) with the
// activations, S is the stored `s`. A lane takes a group of 32 values: 4 bytes.
struct EngQ10 {
    static constexpr int BLOCK_VALUES = 128;
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int LANES = 4;
    static constexpr int SMEM_BYTES = 0;
    static constexpr int UNROLL = 2;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t bits[8];  // value 4k + i of the group in byte i of bits[k]
        float d;
    };
    struct A {
        uint4 first, second;
        float2 ds;
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t g, const uint8_t*) {
        const uint32_t word = ares_ld_stream_u32_a2(blk + 2 + 4 * g);
        W w;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            w.bits[k] = ares_spread_bits4((word >> (4 * k)) & 0xF);
        }
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t g) {
        A a;
        a.first = ares_ld_act_v4(q + 32 * g);
        a.second = ares_ld_act_v4(q + 32 * g + 16);
        a.ds = ds[g];
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int t = ares_dp4a_us(w.bits[0], static_cast<int>(a.first.x), 0);
        t = ares_dp4a_us(w.bits[1], static_cast<int>(a.first.y), t);
        t = ares_dp4a_us(w.bits[2], static_cast<int>(a.first.z), t);
        t = ares_dp4a_us(w.bits[3], static_cast<int>(a.first.w), t);
        t = ares_dp4a_us(w.bits[4], static_cast<int>(a.second.x), t);
        t = ares_dp4a_us(w.bits[5], static_cast<int>(a.second.y), t);
        t = ares_dp4a_us(w.bits[6], static_cast<int>(a.second.z), t);
        t = ares_dp4a_us(w.bits[7], static_cast<int>(a.second.w), t);
        acc = fmaf(static_cast<float>(t), 2.0f * w.d * a.ds.x, acc);
        acc = fmaf(-w.d, a.ds.y, acc);
    }
};

// Q2_0: 64 values in 18 bytes (2-byte aligned): f16 d, 16 bytes of 2-bit codes
// 0..3 for -1, 0, +1, +2; value j is in byte j / 4 at shift 2 (j % 4).
// value = d * (code - 1). A lane takes 4 bytes (16 values); the -1 is half of the
// stored sum of the 32-value group that the lane and its neighbour share.
struct EngQ20 {
    static constexpr int BLOCK_VALUES = 64;
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int LANES = 4;
    static constexpr int SMEM_BYTES = 0;
    static constexpr int UNROLL = 2;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t code[4];  // values 4 b .. 4 b + 3 of the lane's 16 in the bytes of code[b]
        float d;
    };
    struct A {
        uint4 q;
        float2 ds;
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t i, const uint8_t*) {
        const uint32_t word = ares_ld_stream_u32_a2(blk + 2 + 4 * i);
        W w;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            w.code[b] = ares_spread_crumbs4((word >> (8 * b)) & 0xFF);
        }
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t i) {
        A a;
        a.q = ares_ld_act_v4(q + 16 * i);
        a.ds = ds[i >> 1];
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int t = ares_dp4a_us(w.code[0], static_cast<int>(a.q.x), 0);
        t = ares_dp4a_us(w.code[1], static_cast<int>(a.q.y), t);
        t = ares_dp4a_us(w.code[2], static_cast<int>(a.q.z), t);
        t = ares_dp4a_us(w.code[3], static_cast<int>(a.q.w), t);
        acc = fmaf(static_cast<float>(t), w.d * a.ds.x, acc);
        acc = fmaf(-0.5f * w.d, a.ds.y, acc);
    }
};
