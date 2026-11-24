// The many-row expert kernels for the weight types that have no decoder of their own in
// expertrows_types.cuh: an adapter makes one from the item decoder of the dense tile
// kernels (mm_types.cuh, mm_types_more.cuh), which exists for every weight type and gives
// 32 values of a row as f16 pairs. So an expert's gate, up and down can be stored in any
// type, with one decoder per type in the engine's tensor-core kernels, not two.
//
// What an adapted type costs next to a written one: the item is decoded whole when a step
// begins (16 words per row stream held across the step's four groups, where a written type
// decodes a group as it is used), and the down kernel, whose threads take 16 values each,
// loads and decodes the item for half of it. docs/dev/architecture/kernels.md has the measurements.
#pragma once

#include "expertrows_down.cuh"
#include "mm_types_more.cuh"

// Gate and up: thread t of a quad takes item (t & 1) of the step of 64 values (t >> 1) of
// its chunk of 128, sub-block 4 c + t counted as the tile kernels count items.
template <class M>
struct ErOfMm {
    static constexpr uint32_t BLOCK_VALUES = M::BLOCK_VALUES;
    static constexpr uint32_t BLOCK_BYTES = M::BLOCK_BYTES;
    static constexpr uint32_t TABLE_BYTES = M::SMEM_BYTES;
    static_assert(M::SMEM_BYTES % 16 == 0, "the stages behind the table are 16-byte aligned");

    struct Raw {
        typename M::Raw raw;
        uint32_t sb;
    };
    // The item's 32 values: two runs of 8 f16 pairs.
    struct Scales {
        uint32_t v[2][8];
    };

    static __device__ __forceinline__ void init(uint8_t* tab) {
        if (M::SMEM_BYTES > 0) {
            M::init(tab, threadIdx.x, ARES_ER_THREADS);
        }
    }

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t sb) {
        Raw r;
        r.raw = M::load(row, sb >> 1, sb & 1);
        r.sb = sb;
        return r;
    }

    static __device__ __forceinline__ Scales scales(const Raw& r, const uint8_t* tab) {
        Scales s;
        ares_mm_decode_item<M>(r.raw, r.sb >> 1, r.sb & 1, tab, true, s.v);
        return s;
    }

    static __device__ __forceinline__ void group(const Raw&, const Scales& s, int l, const uint8_t*, uint32_t (&h)[4]) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            h[j] = s.v[l >> 1][4 * (l & 1) + j];
        }
    }

    // Step l is half (l & 1) of run (l >> 1) of the item; where the runs of an item lie in
    // the 64 values of its step is the type's (Q4_K and Q5_K keep the low nibbles of two
    // sub-blocks in one run).
    static __device__ __forceinline__ uint32_t piece(uint32_t t, uint32_t l) {
        return 8 * (t >> 1) + ((t & 1) * M::ITEM_STRIDE + (l >> 1) * M::RUN_STRIDE) / 8 + (l & 1);
    }
};

// Down: thread t of a quad takes values 16 t to 16 t + 15 of a block of 64, which is one
// run of one item of the step.
template <class M>
struct ErDownOfMm {
    static constexpr uint32_t BLOCK_VALUES = M::BLOCK_VALUES;
    static constexpr uint32_t BLOCK_BYTES = M::BLOCK_BYTES;
    static constexpr uint32_t TABLE_BYTES = M::SMEM_BYTES;
    static_assert(M::SMEM_BYTES % 16 == 0, "the stages behind the table are 16-byte aligned");
    static_assert((M::ITEM_STRIDE == 32 && M::RUN_STRIDE == 16) || (M::ITEM_STRIDE == 16 && M::RUN_STRIDE == 32),
                  "a run is 16 consecutive values at a multiple of 16");

    // The item and the run that hold the values 16 t ...
    static __device__ __forceinline__ uint32_t item_of(uint32_t t) {
        return M::ITEM_STRIDE == 32 ? t >> 1 : t & 1u;
    }
    static __device__ __forceinline__ uint32_t run_of(uint32_t t) {
        return M::ITEM_STRIDE == 32 ? t & 1u : t >> 1;
    }

    struct Raw {
        typename M::Raw raw;
        uint32_t b, t;
    };

    static __device__ __forceinline__ void init(uint8_t* tab) {
        if (M::SMEM_BYTES > 0) {
            M::init(tab, threadIdx.x, ARES_ER_THREADS);
        }
    }

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t b, uint32_t t) {
        Raw r;
        r.raw = M::load(row, b, item_of(t));
        r.b = b;
        r.t = t;
        return r;
    }

    // The run's values come out as pairs (value 2 i, value 2 i + 1); the hidden values are
    // stored as (value j, value j + 8), so the halves of two pairs change places.
    static __device__ __forceinline__ void decode(const Raw& r, const uint8_t* tab, uint32_t (&w)[8]) {
        uint32_t v[2][8];
        ares_mm_decode_item<M>(r.raw, r.b, item_of(r.t), tab, true, v);
        const bool second = run_of(r.t) != 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t lo = second ? v[1][j >> 1] : v[0][j >> 1];
            const uint32_t hi = second ? v[1][4 + (j >> 1)] : v[0][4 + (j >> 1)];
            w[j] = __byte_perm(lo, hi, (j & 1) != 0 ? 0x7632 : 0x5410);
        }
    }
};

// The gate and up kernels and the down kernels of the type `M` of the tile kernels, named
// by `SUFFIX` (src/experts.rs makes the same names).
#define ARES_ER_MM_GATE_UP(SUFFIX, M) ARES_ER_GATE_UP_SIZES(ares_er_gate_up_##SUFFIX, ErOfMm<M>)
#define ARES_ER_MM_DOWN(SUFFIX, M) ARES_ER_DOWN_SIZES(ares_er_down_##SUFFIX, ErDownOfMm<M>)
#define ARES_ER_MM_TYPE(SUFFIX, M) \
    ARES_ER_MM_GATE_UP(SUFFIX, M)  \
    ARES_ER_MM_DOWN(SUFFIX, M)
