// The weight types for the prefill kernels (mm.cuh) that the first module (prefill.cu)
// does not have, in f16 only: the item decoders of the codebook types are those of the
// expert kernels (expertrows_types.cuh), which decode a group of eight values straight to
// f16 pairs with the scale folded in (`decode_f16`, see AresMmDirect in mm.cuh), the
// simpler types are codes and scales as in mm_types.cuh. The 8-bit forms of
// mm_types.cuh would need the same decoders to another output; on a capped card the
// f16 with f16 accumulation of IQ2_S is as fast as its 8-bit form (src/kernels.rs), so
// the types here have the two f16 variants only.
#pragma once

#include "expertrows_types.cuh"
#include "mm_types.cuh"
#include "mmvq_q4_K.cuh"
#include "mmvq_more.cuh"

// The f16 pairs of the four groups of an item (sub-block of 32 values), two groups to a run.
template <class E>
static __device__ __forceinline__ void ares_mm_er_item(const typename E::Raw& raw, const uint8_t* tab, uint32_t (&out)[2][8]) {
    const AresErScales scales = E::scales(raw);
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

// IQ1_S: 256 values in 50 bytes (2-byte aligned): f16 d, 32 bytes of low index bits, 8 u16 of
// high index bits, a 3-bit scale and a shift sign. A sub-block ib (32 values): idx qs[4 ib + l]
// | (3 bits of qh[ib]) << 8 for the four groups l of 8 values, value = d (2 s + 1) (grid + delta)
// with the delta +-1/8 of the sub-block (bit 15 of qh). The group is IQ1_M's.
struct ErIQ1S {
    static constexpr uint32_t BLOCK_BYTES = 50;
    static constexpr uint32_t TABLE_BYTES = ErIQ1M::TABLE_BYTES;

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
    static __device__ __forceinline__ AresErScales scales(const Raw& r) {
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
