// The weight types for the prefill kernels (mm.cuh) that the first module (prefill.cu)
// does not have, in f16 only: the item decoders of the codebook types are those of the
// expert kernels (expertrows_types.cuh), which decode a group of eight values straight to
// f16 pairs with the scale folded in (`decode_f16`, see AresMmDirect in mm.cuh), the
// simpler types are codes and scales as in mm_types.cuh, and the float types (F32, F16,
// BF16) have no codes at all. The 8-bit forms of mm_types.cuh would need the same decoders
// to another output; on a capped card the f16 with f16 accumulation of IQ2_S is as fast as
// its 8-bit form (src/kernels.rs), so the types here have the two f16 variants only.
#pragma once

#include "expertrows_types.cuh"
#include "mm_types.cuh"
#include "mmvq_q4_K.cuh"
#include "mmvq_more.cuh"

// The f16 pairs of the four groups of an item (sub-block of 32 values), two groups to a run.
template <class E>
static __device__ __forceinline__ void ares_mm_er_item(const typename E::Raw& raw, const uint8_t* tab, uint32_t (&out)[2][8]) {
    const AresErScales scales = E::scales(raw, tab);
#pragma unroll
    for (int l = 0; l < 4; l++) {
        uint32_t h[4];
        E::group(raw, scales, l, tab, h);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            out[l >> 1][4 * (l & 1) + j] = h[j];
        }
    }
}

// An item is a sub-block of 32 values: sb = 2 chunk + item of the row.
template <class E>
struct EngMmErItem {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = E::BLOCK_BYTES;
    static constexpr int SMEM_BYTES = E::TABLE_BYTES;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = true;
    static constexpr bool MAGIC = false;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    using Raw = typename E::Raw;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t, uint32_t) {
        E::init(smem);
    }
    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        return E::load(row, 2 * chunk + item);
    }
    static __device__ __forceinline__ void decode_f16(const Raw& raw, uint32_t, uint32_t, const uint8_t* tab, uint32_t (&out)[2][8]) {
        ares_mm_er_item<E>(raw, tab, out);
    }
};

using EngMmIQ2XXS = EngMmErItem<ErIQ2XXS>;
using EngMmIQ1M = EngMmErItem<ErIQ1M>;

// IQ1_S: 256 values in 50 bytes (2-byte aligned): f16 d, 32 bytes of low index bits, 8 u16 of
// high index bits, a 3-bit scale and a shift sign. A sub-block ib (32 values): idx qs[4 ib + l]
// | (3 bits of qh[ib]) << 8 for the four groups l of 8 values, value = d (2 s + 1) (grid + delta)
// with the delta +-1/8 of the sub-block (bit 15 of qh). The group is IQ1_M's.
struct ErIQ1S {
    static constexpr uint32_t BLOCK_VALUES = 256;
    static constexpr uint32_t BLOCK_BYTES = 50;
    static constexpr uint32_t TABLE_BYTES = ErIQ1M::TABLE_BYTES;
    using Scales = AresErScales;
    static __device__ __forceinline__ uint32_t piece(uint32_t t, uint32_t l) {
        return 4 * t + l;
    }

    struct Raw {
        uint32_t qs, qh, d;
    };

    static __device__ __forceinline__ void init(uint8_t* tab) {
        ErIQ1M::init(tab);
    }

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t sb) {
        const uint8_t* blk = row + static_cast<size_t>(sb >> 3) * BLOCK_BYTES;
        const uint32_t ib = sb & 7;
        Raw r;
        r.qs = ares_ld_stream_u32_a2(blk + 2 + 4 * ib);
        r.qh = ares_ld_stream_u16(blk + 34 + 2 * ib);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    // `sc` is d (2 s + 1) for all four groups, `nsc` the shift: the sign of the delta bit applied.
    static __device__ __forceinline__ AresErScales scales(const Raw& r, const uint8_t*) {
        AresErScales s;
        const uint32_t sc = ares_er_h2_of(static_cast<float>(2 * ((r.qh >> 12) & 7) + 1) * ares_er_f16_at(r.d));
        const uint32_t eighth = ares_er_h2_mul(sc, 0x30003000u);  // 1/8
        s.sc[0] = s.sc[1] = sc;
        s.nsc[0] = s.nsc[1] = (r.qh & 0x8000u) ? (eighth ^ 0x80008000u) : eighth;
        return s;
    }

    static __device__ __forceinline__ void group(const Raw& r, const AresErScales& s, int l, const uint8_t* tab, uint32_t (&h)[4]) {
        const uint2 g = reinterpret_cast<const uint2*>(tab)[((r.qs >> (8 * l)) & 0xFF) | (((r.qh >> (3 * l)) & 7) << 8)];
        ares_er_iq1_values(g, s.sc[0], s.nsc[0], h);
    }
};

using EngMmIQ1S = EngMmErItem<ErIQ1S>;

// ---------------------------------------------------------------------------
// Q2_0: 64 values in 18 bytes (2-byte aligned): f16 d, 16 bytes of 2-bit codes 0..3 for
// -1, 0, +1, +2, value j in byte j / 4 at shift 2 (j % 4); value = d (code - 1). The
// item is half a block: 8 bytes, a run being 4 of them.
struct EngMmQ20 {
    static constexpr int BLOCK_VALUES = 64;
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr bool MAGIC = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint2 q;
        uint32_t d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk) * BLOCK_BYTES;
        Raw r;
        r.q = ares_ld_stream_v2_a2(blk + 2 + 8 * item);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t*, AresMmDec& o) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            o.w[i] = ares_spread_crumbs4((r.q.x >> (8 * i)) & 0xFF);
            o.w[4 + i] = ares_spread_crumbs4((r.q.y >> (8 * i)) & 0xFF);
        }
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(d, d);
        o.sw[0] = o.sw[1] = d;
        o.mw[0] = o.mw[1] = d;
    }
};

// ---------------------------------------------------------------------------
// The legacy 4- and 5-bit types and the others whose blocks are 32 values with codes of 4 or
// fewer bits: the item is a block (a half block for Q1_0), its first run the low nibbles (or
// the first 16 values), the second the high ones. value = sw * code - mw as in mm_types.cuh.

// 32 values: f16 d, optionally an f16 m, optionally a u32 qh with the fifth bit of every
// value, 16 bytes of nibbles (low nibbles are values 0..15, high nibbles 16..31):
//             d  m  qh   bytes   value
//   Q4_0      x         18      d * (q - 8)
//   Q4_1      x  x      20      d * q + m
//   Q5_0      x     x   22      d * (q - 16)        q of 5 bits
//   Q5_1      x  x  x   24      d * q + m           q of 5 bits
// Value j has its fifth bit in bit j of qh, the value j + 16 in bit j + 16. The blocks with
// a minimum are 4-byte aligned and load whole words, the others 2-byte aligned and load
// halves.
template <bool HAS_M, bool HAS_QH>
struct EngMmLegacy4 {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 2 + (HAS_M ? 2 : 0) + (HAS_QH ? 4 : 0) + 16;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr bool MAGIC = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t dm, qh, q[4];
    };

    // PIECES: the large shape of the tile kernels asks for 16-byte pieces (ares_ld_words_a2).
    template <bool PIECES = false>
    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        constexpr int QH_AT = HAS_M ? 4 : 2;
        constexpr int QS_AT = QH_AT + (HAS_QH ? 4 : 0);
        const uint8_t* blk = row + static_cast<size_t>(2 * chunk + item) * BLOCK_BYTES;
        Raw r;
        r.qh = 0;
        if constexpr (HAS_M) {
            r.dm = __ldg(reinterpret_cast<const uint32_t*>(blk));
            if constexpr (HAS_QH) {
                r.qh = __ldg(reinterpret_cast<const uint32_t*>(blk + QH_AT));
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                r.q[i] = __ldg(reinterpret_cast<const uint32_t*>(blk + QS_AT) + i);
            }
        } else {
            r.dm = ares_ld_stream_u16(blk);
            if constexpr (HAS_QH) {
                r.qh = ares_ld_stream_u32_a2(blk + QH_AT);
            }
            ares_ld_words_a2<4, PIECES>(blk + QS_AT, r.q);
        }
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t*, AresMmDec& o) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            o.w[i] = r.q[i] & 0x0F0F0F0Fu;
            o.w[4 + i] = (r.q[i] >> 4) & 0x0F0F0F0Fu;
            if constexpr (HAS_QH) {
                o.w[i] |= ares_spread_bits4((r.qh >> (4 * i)) & 0xF) << 4;
                o.w[4 + i] |= ares_spread_bits4((r.qh >> (16 + 4 * i)) & 0xF) << 4;
            }
        }
        const float d = ares_mm_f16_of(r.dm);
        o.sw[0] = o.sw[1] = d;
        if constexpr (HAS_M) {
            o.mw[0] = o.mw[1] = -ares_mm_f16_of(r.dm >> 16);
        } else {
            o.mw[0] = o.mw[1] = (HAS_QH ? 16.0f : 8.0f) * d;
        }
        o.ent = make_float2(o.sw[0], o.mw[0]);
    }
};

using EngMmQ40 = EngMmLegacy4<false, false>;
using EngMmQ41 = EngMmLegacy4<true, false>;
using EngMmQ50 = EngMmLegacy4<false, true>;
using EngMmQ51 = EngMmLegacy4<true, true>;



// ---------------------------------------------------------------------------
// Q2_K: 256 values in 84 bytes (4-byte aligned): 16 bytes of scales (low nibble the scale, high
// nibble the minimum, of a group of 16 values), 64 bytes of 2-bit codes, f16 d and dmin. The
// codes are two halves of 128 values, in each four planes (shift 2 p) of 32 bytes:
// value 32 p + l of a half is bits 2 p of byte l. The item is a plane of a half, a run 16 of
// its bytes. value = d sc q - dmin mn.
struct EngMmQ2K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 84;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr bool MAGIC = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t q[8], sc, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t m = 2 * (chunk & 3) + item;  // the item of the block, 0..7
        Raw r;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            r.q[i] = __ldg(reinterpret_cast<const uint32_t*>(blk + 16 + 32 * (m >> 2)) + i);
        }
        r.sc = ares_ld_stream_u16(blk + 2 * m);
        r.d = __ldg(reinterpret_cast<const uint32_t*>(blk + 80));
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, AresMmDec& o) {
        const uint32_t plane = (2 * (chunk & 3) + item) & 3;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            o.w[i] = (r.q[i] >> (2 * plane)) & 0x03030303u;
        }
        const float d = ares_mm_f16_of(r.d), dmin = ares_mm_f16_of(r.d >> 16);
#pragma unroll
        for (int run = 0; run < 2; run++) {
            const uint32_t s = (r.sc >> (8 * run)) & 0xFF;
            o.sw[run] = d * static_cast<float>(s & 15);
            o.mw[run] = dmin * static_cast<float>(s >> 4);
        }
        o.ent = make_float2(o.sw[0], o.mw[0]);
    }
};

// TQ2_0: 256 values in 66 bytes (2-byte aligned): 64 bytes of 2-bit codes 0, 1, 2 for -1, 0, +1,
// then f16 d. Two runs of 32 bytes; in a run, plane l (shift 2 l) of byte m is the value
// 128 run + 32 l + m. The item is a plane of a run of 32 bytes, its runs 16 of them.


// MXFP4: 32 values in 17 bytes (byte aligned): an E8M0 exponent byte, then 16 bytes of nibbles
// as Q4_0, each an index into the 16 levels of E2M1 (doubled, so that they are integers).
struct EngMmMXFP4 {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 17;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = true;
    static constexpr bool MAGIC = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint2 q0, q1;
        uint32_t e;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(2 * chunk + item) * BLOCK_BYTES;
        Raw r;
        r.e = ares_ld_stream_u8(blk);
        r.q0 = ares_ld_stream_v2_any(blk + 1);
        r.q1 = ares_ld_stream_v2_any(blk + 9);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t*, AresMmDec& o) {
        const uint32_t t0 = ARES_PACK4(0, 1, 2, 3), t1 = ARES_PACK4(4, 6, 8, 12);
        const uint32_t t2 = ARES_PACK4(0, -1, -2, -3), t3 = ARES_PACK4(-4, -6, -8, -12);
        const uint32_t q[4] = {r.q0.x, r.q0.y, r.q1.x, r.q1.y};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int2 v = ares_table16_with(q[i], t0, t1, t2, t3);
            o.w[i] = static_cast<uint32_t>(v.x);
            o.w[4 + i] = static_cast<uint32_t>(v.y);
        }
        const float d = ares_e8m0_half(r.e);
        o.ent = make_float2(d, 0.0f);
        o.sw[0] = o.sw[1] = d;
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

// NVFP4: 64 values in 36 bytes (4-byte aligned): four UE4M3 scale bytes, then four sub-blocks
// of 16 values in 8 bytes each (low nibbles the first 8 values, high nibbles the next 8),
// through the same 16 levels. The item is two sub-blocks, a run each, with a scale of its own.
struct EngMmNVFP4 {
    static constexpr int BLOCK_VALUES = 64;
    static constexpr int BLOCK_BYTES = 36;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = true;
    static constexpr bool SIGNED_CODES = true;
    static constexpr bool MAGIC = false;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t scales, q[4];
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk) * BLOCK_BYTES;
        Raw r;
        r.scales = __ldg(reinterpret_cast<const uint32_t*>(blk));
#pragma unroll
        for (int i = 0; i < 4; i++) {
            r.q[i] = __ldg(reinterpret_cast<const uint32_t*>(blk + 4 + 16 * item) + i);
        }
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, AresMmDec& o) {
        const uint32_t t0 = ARES_PACK4(0, 1, 2, 3), t1 = ARES_PACK4(4, 6, 8, 12);
        const uint32_t t2 = ARES_PACK4(0, -1, -2, -3), t3 = ARES_PACK4(-4, -6, -8, -12);
#pragma unroll
        for (int run = 0; run < 2; run++) {
            const int2 a = ares_table16_with(r.q[2 * run], t0, t1, t2, t3);
            const int2 b = ares_table16_with(r.q[2 * run + 1], t0, t1, t2, t3);
            o.w[4 * run] = static_cast<uint32_t>(a.x);
            o.w[4 * run + 1] = static_cast<uint32_t>(b.x);
            o.w[4 * run + 2] = static_cast<uint32_t>(a.y);
            o.w[4 * run + 3] = static_cast<uint32_t>(b.y);
            o.sw[run] = ares_ue4m3_half((r.scales >> (8 * (2 * item + run))) & 0xFF);
            o.mw[run] = 0.0f;
        }
    }
};

// Q1_0: 128 values in 18 bytes (2-byte aligned): f16 d, 16 bytes of bits, value j in bit j % 8
// of byte j / 8; set is +d, clear -d. The item is 4 bytes, a run 2 of them.
struct EngMmQ10 {
    static constexpr int BLOCK_VALUES = 128;
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr bool MAGIC = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t bits, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 1) * BLOCK_BYTES;
        Raw r;
        r.bits = ares_ld_stream_u32_a2(blk + 2 + 4 * (2 * (chunk & 1) + item));
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t*, AresMmDec& o) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            o.w[i] = ares_spread_bits4((r.bits >> (4 * i)) & 0xF);
        }
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(2.0f * d, d);
        o.sw[0] = o.sw[1] = 2.0f * d;
        o.mw[0] = o.mw[1] = d;
    }
};


// TQ2_0: 256 values in 66 bytes (2-byte aligned): 64 bytes of 2-bit codes 0, 1, 2 for -1, 0, +1,
// then f16 d. Two runs of 32 bytes; in a run, plane l (shift 2 l) of byte m is the value
// 128 run + 32 l + m. The item is a plane of a run of 32 bytes, its runs 16 of them.
struct EngMmTQ20 {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 66;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr bool MAGIC = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t q[8], d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t m = 2 * (chunk & 3) + item;
        Raw r;
        ares_ld_words_a2<8>(blk + 32 * (m >> 2), r.q);
        r.d = ares_ld_stream_u16(blk + 64);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, AresMmDec& o) {
        const uint32_t plane = (2 * (chunk & 3) + item) & 3;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            o.w[i] = (r.q[i] >> (2 * plane)) & 0x03030303u;
        }
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(d, d);
        o.sw[0] = o.sw[1] = d;
        o.mw[0] = o.mw[1] = d;
    }
};

// TQ1_0: 256 values in 54 bytes (2-byte aligned): 48 bytes of qs, 4 of qh, f16 d at 52. Five
// base-3 digits (0, 1, 2 for -1, 0, +1) are packed in a byte of qs (four in a byte of qh),
// digit n of a byte b being (b 3^n mod 256) 3 >> 8. Values 0..159 are digit n = v / 32 of the
// bytes 0..31, 160..239 digit n = (v - 160) / 16 of the bytes 32..47, 240..255 digit
// n = (v - 240) / 4 of the bytes of qh. The items 0..4 are a digit of the 32 bytes of the first
// run, items 5, 6 and the first run of 7 digits of the 16 bytes behind (a digit a run), the
// second run of item 7 the four digits of qh.
struct EngMmTQ10 {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 54;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr bool MAGIC = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t q[8], qh, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t m = 2 * (chunk & 3) + item;
        Raw r;
        // Items 0..4 read the first 32 bytes, the others the 16 behind them.
        if (m < 5) {
            ares_ld_words_a2<8>(blk, r.q);
        } else {
            uint32_t tail[4];
            ares_ld_words_a2<4>(blk + 32, tail);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                r.q[i] = tail[i];
                r.q[4 + i] = 0;
            }
        }
        r.qh = ares_ld_stream_u32_a2(blk + 48);
        r.d = ares_ld_stream_u16(blk + 52);
        return r;
    }

    // Digit n (0..2) of the four bytes of `b`, a digit to a byte.
    static __device__ __forceinline__ uint32_t digits(uint32_t b, uint32_t pow3) {
        const uint32_t even = (((b & 0x00FF00FFu) * pow3) & 0x00FF00FFu) * 3u >> 8 & 0x00FF00FFu;
        const uint32_t odd = ((((b >> 8) & 0x00FF00FFu) * pow3) & 0x00FF00FFu) * 3u >> 8 & 0x00FF00FFu;
        return even | (odd << 8);
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, AresMmDec& o) {
        constexpr uint32_t POW3[5] = {1, 3, 9, 27, 81};
        const uint32_t m = 2 * (chunk & 3) + item;
        if (m < 5) {
            uint32_t p = 1;
#pragma unroll
            for (uint32_t n = 0; n < 5; n++) {
                if (n == m) {
                    p = POW3[n];
                }
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                o.w[i] = digits(r.q[i], p);
            }
        } else {
            // Run 0: digit 2 (m - 5) of the 16 bytes (n = 0, 2, 4 for m = 5, 6, 7), run 1: the next digit
            // (n = 1, 3) or, for m = 7, the four digits of qh.
            const uint32_t n0 = 2 * (m - 5);
            uint32_t p0 = 1, p1 = 3;
#pragma unroll
            for (uint32_t n = 0; n < 5; n++) {
                if (n == n0) {
                    p0 = POW3[n];
                    p1 = POW3[(n + 1) % 5];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                o.w[i] = digits(r.q[i], p0);
                o.w[4 + i] = digits(r.q[i], p1);
            }
            if (m == 7) {
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    o.w[4 + n] = digits(r.qh, POW3[n]);
                }
            }
        }
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(d, d);
        o.sw[0] = o.sw[1] = d;
        o.mw[0] = o.mw[1] = d;
    }
};

// ---------------------------------------------------------------------------
// The float types F32, F16 and BF16: a block is one value, the item 32 values (64 or 128
// bytes) read as 16-byte words, so the matrix must be 16-byte aligned (its rows are a
// multiple of 64 bytes long, since the step is 32 values). The tile holds the weights as
// f16 operands without a scale, converted by the round to nearest of `cvt`:
//   F16   copied; exact.
//   BF16  a bf16 is the top half of an f32, so the conversion is a shift and `cvt`; exact
//         for the values in the range of f16 normal numbers (the 8 bits of mantissa of a
//         bf16 fit the 11 of an f16).
//   F32   rounded to 11 bits of mantissa (relative error 2^-11 at most).
// A value beyond 65504 becomes infinite (the product is then infinite or NaN, not a wrong
// finite number), one below 2^-14 loses bits from the bottom (an f16 subnormal, absolute
// error at most 2^-25), and one below 2^-25 becomes zero. The variant with f16 accumulation
// adds partial sums of 64 products of a weight and a value of at most 1, which overflow at
// weights of about 1000 and more (an infinity as well). docs/ARCHITECTURE.md, "Float
// matrices", has the range of the matrices of the two models.
template <int ELEM_BYTES, bool BF16>
struct EngMmFloat {
    static constexpr int BLOCK_VALUES = 1;
    static constexpr int BLOCK_BYTES = ELEM_BYTES;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr bool MAGIC = false;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    // The 16-byte words of an item (a run is half of them).
    static constexpr int WORDS = 2 * ELEM_BYTES;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint4 q[WORDS];
    };

    // The item holds the values 64 chunk + 32 item of the row. The loads stay in L1: the
    // other item of a step reads the other half of the same lines.
    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint4* p = reinterpret_cast<const uint4*>(row + static_cast<size_t>(64 * chunk + 32 * item) * ELEM_BYTES);
        Raw r;
#pragma unroll
        for (int i = 0; i < WORDS; i++) {
            r.q[i] = __ldg(p + i);
        }
        return r;
    }

    // Two values as an f16 pair, the first in the low half.
    static __device__ __forceinline__ uint32_t pair(float first, float second) {
        const __half2 h = __floats2half2_rn(first, second);
        return *reinterpret_cast<const uint32_t*>(&h);
    }

    // The f16 pair of the two 16-bit values of a word (F16, BF16).
    static __device__ __forceinline__ uint32_t pair_of_halves(uint32_t two) {
        if constexpr (BF16) {
            return pair(__uint_as_float(two << 16), __uint_as_float(two & 0xFFFF0000u));
        } else {
            return two;
        }
    }

    static __device__ __forceinline__ void decode_f16(const Raw& r, uint32_t, uint32_t, const uint8_t*, uint32_t (&out)[2][8]) {
#pragma unroll
        for (int run = 0; run < 2; run++) {
            if constexpr (ELEM_BYTES == 4) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint4 v = r.q[4 * run + k];
                    out[run][2 * k] = pair(__uint_as_float(v.x), __uint_as_float(v.y));
                    out[run][2 * k + 1] = pair(__uint_as_float(v.z), __uint_as_float(v.w));
                }
            } else {
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const uint4 v = r.q[2 * run + k];
                    out[run][4 * k] = pair_of_halves(v.x);
                    out[run][4 * k + 1] = pair_of_halves(v.y);
                    out[run][4 * k + 2] = pair_of_halves(v.z);
                    out[run][4 * k + 3] = pair_of_halves(v.w);
                }
            }
        }
    }
};

using EngMmF32 = EngMmFloat<4, false>;
using EngMmF16 = EngMmFloat<2, false>;
using EngMmBF16 = EngMmFloat<2, true>;
