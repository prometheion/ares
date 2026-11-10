// The weight types for the prefill kernels (mm.cuh).
//
// An adapter decodes one item: 32 consecutive values of one weight row, in two
// runs of 16 (see the contract in mm.cuh). The step of 64 values (`chunk`)
// holds two items. The layouts of the blocks are those of the decode kernels,
// whose headers (mmvq_kquants.cuh, mmvq_iquants.cuh, mmvq_q4_K.cuh) document
// them; the codebooks, the sign handling and the 4-bit table are theirs too
// (iq_tables.cuh, ares_table16, ares_apply_sign4 ...), so the two families cannot
// disagree about a format.
//
// `load` only issues the global loads and `decode` only computes, so the loads
// of the next step can be in flight while the tensor cores work on this one.
#pragma once

#include "mm.cuh"
#include "mmvq_iquants.cuh"

// Scale and minimum of sub-block s (0..7) of a Q4_K or Q5_K block, from the
// 12 bytes of packed 6-bit values that sit in the words y, z, w of the header.
static __device__ __forceinline__ void ares_mm_scale_min_k4(const uint4& hdr, uint32_t s, uint32_t& scale, uint32_t& minimum) {
    auto byte = [&](uint32_t i) {
        const uint32_t word = (i >> 2) == 0 ? hdr.y : ((i >> 2) == 1 ? hdr.z : hdr.w);
        return (word >> (8 * (i & 3))) & 0xFF;
    };
    if (s < 4) {
        scale = byte(s) & 63;
        minimum = byte(s + 4) & 63;
    } else {
        scale = (byte(s + 4) & 0xF) | ((byte(s - 4) >> 6) << 4);
        minimum = (byte(s + 4) >> 4) | ((byte(s) >> 6) << 4);
    }
}

static __device__ __forceinline__ float ares_mm_f16_of(uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// ---------------------------------------------------------------------------
// Q4_K and Q5_K: the item is a half (16 bytes) of a 32-byte group of nibbles;
// its first run is the low nibbles (16 values of sub-block 2j), its second the
// high nibbles (the same positions of sub-block 2j+1).
struct EngMmQ4K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 144;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr int ITEM_STRIDE = 16;
    static constexpr int RUN_STRIDE = 32;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint4 hdr, qs;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        Raw r;
        // The header is read by the four steps of the block: let it stay in L1.
        r.hdr = __ldg(reinterpret_cast<const uint4*>(blk));
        r.qs = ares_ld_stream_v4(blk + 16 + 32 * (chunk & 3) + 16 * item);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, EngMmDec& o) {
        o.w[0] = r.qs.x & 0x0F0F0F0Fu;
        o.w[1] = r.qs.y & 0x0F0F0F0Fu;
        o.w[2] = r.qs.z & 0x0F0F0F0Fu;
        o.w[3] = r.qs.w & 0x0F0F0F0Fu;
        o.w[4] = (r.qs.x >> 4) & 0x0F0F0F0Fu;
        o.w[5] = (r.qs.y >> 4) & 0x0F0F0F0Fu;
        o.w[6] = (r.qs.z >> 4) & 0x0F0F0F0Fu;
        o.w[7] = (r.qs.w >> 4) & 0x0F0F0F0Fu;
        const float d = ares_mm_f16_of(r.hdr.x);
        const float dmin = ares_mm_f16_of(r.hdr.x >> 16);
        const uint32_t s = 2 * (chunk & 3);
        uint32_t sc, mn;
        // The scale entry is that of the item's own sub-block ...
        ares_mm_scale_min_k4(r.hdr, s + item, sc, mn);
        o.ent = make_float2(d * static_cast<float>(sc), dmin * static_cast<float>(mn));
        // ... the f16 tile needs the scales of both runs.
#pragma unroll
        for (int run = 0; run < 2; run++) {
            ares_mm_scale_min_k4(r.hdr, s + run, sc, mn);
            o.sw[run] = d * static_cast<float>(sc);
            o.mw[run] = dmin * static_cast<float>(mn);
        }
    }
};

// Q5_K: the fifth bit of each value comes from qh, where sub-block i keeps its
// bit at bit i of byte l.
struct EngMmQ5K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 176;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = true;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = false;
    static constexpr int ITEM_STRIDE = 16;
    static constexpr int RUN_STRIDE = 32;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint4 hdr, qh, qs;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        Raw r;
        r.hdr = __ldg(reinterpret_cast<const uint4*>(blk));
        r.qh = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 16 * item));
        r.qs = ares_ld_stream_v4(blk + 48 + 32 * (chunk & 3) + 16 * item);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, EngMmDec& o) {
        const uint32_t j = chunk & 3;
        const uint32_t qs[4] = {r.qs.x, r.qs.y, r.qs.z, r.qs.w};
        const uint32_t qh[4] = {r.qh.x, r.qh.y, r.qh.z, r.qh.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            o.w[i] = (qs[i] & 0x0F0F0F0Fu) | (((qh[i] >> (2 * j)) & 0x01010101u) << 4);
            o.w[4 + i] = ((qs[i] >> 4) & 0x0F0F0F0Fu) | (((qh[i] >> (2 * j + 1)) & 0x01010101u) << 4);
        }
        const float d = ares_mm_f16_of(r.hdr.x);
        const float dmin = ares_mm_f16_of(r.hdr.x >> 16);
        const uint32_t s = 2 * j;
        uint32_t sc, mn;
        ares_mm_scale_min_k4(r.hdr, s + item, sc, mn);
        o.ent = make_float2(d * static_cast<float>(sc), dmin * static_cast<float>(mn));
#pragma unroll
        for (int run = 0; run < 2; run++) {
            ares_mm_scale_min_k4(r.hdr, s + run, sc, mn);
            o.sw[run] = d * static_cast<float>(sc);
            o.mw[run] = dmin * static_cast<float>(mn);
        }
    }
};

// ---------------------------------------------------------------------------
// Q6_K: the step is two of the four "rows" of 32 positions a half of the block
// is built from (mmvq_kquants.cuh); the item is one row, whose two runs are the
// two halves with their own scales.
struct EngMmQ6K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 210;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = true;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t ql[8], qh[8];
        uint32_t scales, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t hh = (chunk >> 1) & 1;
        const uint32_t k = 2 * (chunk & 1) + item;
        Raw r;
        ares_ld_words_a2<8>(blk + 64 * hh + 32 * (k & 1), r.ql);
        ares_ld_words_a2<8>(blk + 128 + 32 * hh, r.qh);
        r.scales = ares_ld_stream_u16(blk + 192 + 8 * hh + 2 * k);
        r.d = ares_ld_stream_u16(blk + 208);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, EngMmDec& o) {
        const uint32_t k = 2 * (chunk & 1) + item;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t nibble = (chunk & 1) == 0 ? (r.ql[i] & 0x0F0F0F0Fu) : ((r.ql[i] >> 4) & 0x0F0F0F0Fu);
            const uint32_t top = ((r.qh[i] >> (2 * k)) & 0x03030303u) << 4;
            o.w[i] = static_cast<uint32_t>(__vsub4(static_cast<int>(nibble | top), 0x20202020));
        }
        const int s0 = static_cast<int8_t>(r.scales & 0xFF);
        const int s1 = static_cast<int8_t>((r.scales >> 8) & 0xFF);
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(d, __int_as_float((s0 & 0xFFFF) | (s1 << 16)));
        o.sw[0] = d * static_cast<float>(s0);
        o.sw[1] = d * static_cast<float>(s1);
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

// Q3_K: the same "rows"; the code is 2 bits from qs plus a high bit from hmask,
// minus 4.
struct EngMmQ3K {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = true;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t qs[8], hm[8];
        uint32_t sw[3];
        uint32_t d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t hh = (chunk >> 1) & 1;
        Raw r;
        ares_ld_words_a2<8>(blk + 32 + 32 * hh, r.qs);
        ares_ld_words_a2<8>(blk, r.hm);
        ares_ld_words_a2<3>(blk + 96, r.sw);
        r.d = ares_ld_stream_u16(blk + 108);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, EngMmDec& o) {
        const uint32_t hh = (chunk >> 1) & 1;
        const uint32_t k = 2 * (chunk & 1) + item;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t code = ((r.qs[i] >> (2 * k)) & 0x03030303u) | (((r.hm[i] >> (4 * hh + k)) & 0x01010101u) << 2);
            o.w[i] = static_cast<uint32_t>(__vsub4(static_cast<int>(code), 0x04040404));
        }
        // The 16 six-bit scales unpacked as the CPU decoder does, for the two
        // words this half of the block needs; scale 8 hh + p + 2 k is byte
        // 2 (k & 1) + p of word (k >> 1).
        const uint32_t s4 = 4 * hh;
        const uint32_t a_lo = ((r.sw[0] >> s4) & 0x0F0F0F0Fu) | (((r.sw[2] >> s4) & 0x03030303u) << 4);
        const uint32_t a_hi = ((r.sw[1] >> s4) & 0x0F0F0F0Fu) | (((r.sw[2] >> (s4 + 2)) & 0x03030303u) << 4);
        const uint32_t word = (k >> 1) != 0 ? a_hi : a_lo;
        const int s0 = static_cast<int>((word >> (16 * (k & 1))) & 0xFF) - 32;
        const int s1 = static_cast<int>((word >> (16 * (k & 1) + 8)) & 0xFF) - 32;
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(d, __int_as_float((s0 & 0xFFFF) | (s1 << 16)));
        o.sw[0] = d * static_cast<float>(s0);
        o.sw[1] = d * static_cast<float>(s1);
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

// ---------------------------------------------------------------------------
// IQ4_XS: the item is a sub-block of 32 values; low nibbles are its first 16
// values, high nibbles its last 16, each through the table of 16 levels.
struct EngMmIQ4XS {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 136;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint2 hdr, q01, q23;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t ib = 2 * (chunk & 3) + item;
        Raw r;
        r.hdr = ares_ld_stream_v2(blk);
        r.q01 = ares_ld_stream_v2(blk + 8 + 16 * ib);
        r.q23 = ares_ld_stream_v2(blk + 16 + 16 * ib);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t*, EngMmDec& o) {
        const uint32_t ib = 2 * (chunk & 3) + item;
        const uint32_t qw[4] = {r.q01.x, r.q01.y, r.q23.x, r.q23.y};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int2 v = ares_table16(qw[i]);
            o.w[i] = static_cast<uint32_t>(v.x);
            o.w[4 + i] = static_cast<uint32_t>(v.y);
        }
        const int ls = static_cast<int>(((r.hdr.y >> (4 * (ib & 1))) >> (8 * (ib >> 1)) & 0x0F) | (((r.hdr.x >> (16 + 2 * ib)) & 3) << 4)) - 32;
        const float sw = ares_mm_f16_of(r.hdr.x) * static_cast<float>(ls);
        o.ent = make_float2(sw, 0.0f);
        o.sw[0] = o.sw[1] = sw;
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

// IQ4_NL and Q8_0: blocks of 32 values with one f16 scale; the item is a block.
struct EngMmIQ4NL {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t q[4];
        uint32_t d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(2 * chunk + item) * BLOCK_BYTES;
        Raw r;
        ares_ld_words_a2<4>(blk + 2, r.q);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t*, EngMmDec& o) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int2 v = ares_table16(r.q[i]);
            o.w[i] = static_cast<uint32_t>(v.x);
            o.w[4 + i] = static_cast<uint32_t>(v.y);
        }
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(d, 0.0f);
        o.sw[0] = o.sw[1] = d;
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

struct EngMmQ80 {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 34;
    static constexpr int SMEM_BYTES = 0;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct Raw {
        uint32_t q[8];
        uint32_t d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(2 * chunk + item) * BLOCK_BYTES;
        Raw r;
        ares_ld_words_a2<8>(blk + 2, r.q);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t*, EngMmDec& o) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            o.w[i] = r.q[i];
        }
        const float d = ares_mm_f16_of(r.d);
        o.ent = make_float2(d, 0.0f);
        o.sw[0] = o.sw[1] = d;
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

// ---------------------------------------------------------------------------
// The codebook i-quants. The item is a sub-block of 32 values: groups of 4 or 8
// magnitudes looked up in a table in shared memory, signed, as signed bytes.

struct EngMmIQ3XXS {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 98;
    static constexpr int SMEM_BYTES = EngIQ3XXS::SMEM_BYTES;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        EngIQ3XXS::init(smem, t, n);
    }

    struct Raw {
        uint2 idx;
        uint32_t aux, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t ib = 2 * (chunk & 3) + item;
        Raw r;
        r.idx = ares_ld_stream_v2_a2(blk + 2 + 8 * ib);
        r.aux = ares_ld_stream_u32_a2(blk + 66 + 4 * ib);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t* smem, EngMmDec& o) {
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(smem);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t word = l < 2 ? r.idx.x : r.idx.y;
            const uint32_t i1 = (word >> (16 * (l & 1))) & 0xFF;
            const uint32_t i2 = (word >> (16 * (l & 1) + 8)) & 0xFF;
            const uint32_t signs = ares_ksigns((r.aux >> (7 * l)) & 127) * 0x01010101u;
            o.w[2 * l] = static_cast<uint32_t>(ares_apply_sign4(grid[i1], ares_sign_mask4(signs, 0x08040201u)));
            o.w[2 * l + 1] = static_cast<uint32_t>(ares_apply_sign4(grid[i2], ares_sign_mask4(signs, 0x80402010u)));
        }
        // (0.5 + scale) * 0.5 = (2 scale + 1) / 4
        const float sw = 0.25f * ares_mm_f16_of(r.d) * static_cast<float>(2 * (r.aux >> 28) + 1);
        o.ent = make_float2(sw, 0.0f);
        o.sw[0] = o.sw[1] = sw;
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

struct EngMmIQ3S {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int SMEM_BYTES = EngIQ3S::SMEM_BYTES;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = false;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        EngIQ3S::init(smem, t, n);
    }

    struct Raw {
        uint2 qs;
        uint32_t qh, signs, sc, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t ib = 2 * (chunk & 3) + item;
        Raw r;
        r.qs = ares_ld_stream_v2_a2(blk + 2 + 8 * ib);
        r.qh = ares_ld_stream_u8(blk + 66 + ib);
        r.signs = ares_ld_stream_u32_a2(blk + 74 + 4 * ib);
        r.sc = ares_ld_stream_u8(blk + 106 + (ib >> 1));
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t chunk, uint32_t item, const uint8_t* smem, EngMmDec& o) {
        const uint32_t ib = 2 * (chunk & 3) + item;
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(smem);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t word = l < 2 ? r.qs.x : r.qs.y;
            const uint32_t i1 = ((word >> (16 * (l & 1))) & 0xFF) | ((r.qh << (8 - 2 * l)) & 256);
            const uint32_t i2 = ((word >> (16 * (l & 1) + 8)) & 0xFF) | ((r.qh << (7 - 2 * l)) & 256);
            const uint32_t spread = ((r.signs >> (8 * l)) & 0xFF) * 0x01010101u;
            o.w[2 * l] = static_cast<uint32_t>(ares_apply_sign4(grid[i1], ares_sign_mask4(spread, 0x08040201u)));
            o.w[2 * l + 1] = static_cast<uint32_t>(ares_apply_sign4(grid[i2], ares_sign_mask4(spread, 0x80402010u)));
        }
        const float sw = ares_mm_f16_of(r.d) * static_cast<float>(1 + 2 * ((r.sc >> (4 * (ib & 1))) & 0xF));
        o.ent = make_float2(sw, 0.0f);
        o.sw[0] = o.sw[1] = sw;
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

// IQ2_XS and IQ2_S have a scale per 16 values: the two runs.
struct EngMmIQ2XS {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 74;
    static constexpr int SMEM_BYTES = EngIQ2XS::SMEM_BYTES;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = true;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        EngIQ2XS::init(smem, t, n);
    }

    struct Raw {
        uint2 qs;
        uint32_t sc, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t ib = 2 * (chunk & 3) + item;
        Raw r;
        r.qs = ares_ld_stream_v2_a2(blk + 2 + 8 * ib);
        r.sc = ares_ld_stream_u8(blk + 66 + ib);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t* smem, EngMmDec& o) {
        const uint2* grid = reinterpret_cast<const uint2*>(smem);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t word = l < 2 ? r.qs.x : r.qs.y;
            const uint32_t q = (word >> (16 * (l & 1))) & 0xFFFF;
            const uint2 g = grid[q & 511];
            const uint32_t spread = ares_ksigns(q >> 9) * 0x01010101u;
            o.w[2 * l] = static_cast<uint32_t>(ares_apply_sign4(g.x, ares_sign_mask4(spread, 0x08040201u)));
            o.w[2 * l + 1] = static_cast<uint32_t>(ares_apply_sign4(g.y, ares_sign_mask4(spread, 0x80402010u)));
        }
        finish(r, o);
    }

    // (0.5 + scale) * 0.25 = (2 scale + 1) / 8
    static __device__ __forceinline__ void finish(const Raw& r, EngMmDec& o) {
        const int s0 = static_cast<int>(2 * (r.sc & 0xF) + 1);
        const int s1 = static_cast<int>(2 * (r.sc >> 4) + 1);
        const float d = 0.125f * ares_mm_f16_of(r.d);
        o.ent = make_float2(d, __int_as_float(s0 | (s1 << 16)));
        o.sw[0] = d * static_cast<float>(s0);
        o.sw[1] = d * static_cast<float>(s1);
        o.mw[0] = o.mw[1] = 0.0f;
    }
};

struct EngMmIQ2S {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 82;
    static constexpr int SMEM_BYTES = EngIQ2S::SMEM_BYTES;
    static constexpr bool HAS_MIN = false;
    static constexpr bool SPLIT16 = true;
    static constexpr bool SIGNED_CODES = true;
    static constexpr int ITEM_STRIDE = 32;
    static constexpr int RUN_STRIDE = 16;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        EngIQ2S::init(smem, t, n);
    }

    struct Raw {
        uint32_t qs, signs, qh, sc, d;
    };

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t chunk, uint32_t item) {
        const uint8_t* blk = row + static_cast<size_t>(chunk >> 2) * BLOCK_BYTES;
        const uint32_t ib = 2 * (chunk & 3) + item;
        Raw r;
        r.qs = ares_ld_stream_u32_a2(blk + 2 + 4 * ib);
        r.signs = ares_ld_stream_u32_a2(blk + 34 + 4 * ib);
        r.qh = ares_ld_stream_u8(blk + 66 + ib);
        r.sc = ares_ld_stream_u8(blk + 74 + ib);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ void decode(const Raw& r, uint32_t, uint32_t, const uint8_t* smem, EngMmDec& o) {
        const uint2* grid = reinterpret_cast<const uint2*>(smem);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t idx = ((r.qs >> (8 * l)) & 0xFF) | ((r.qh << (8 - 2 * l)) & 0x300);
            const uint2 g = grid[idx];
            const uint32_t spread = ((r.signs >> (8 * l)) & 0xFF) * 0x01010101u;
            o.w[2 * l] = static_cast<uint32_t>(ares_apply_sign4(g.x, ares_sign_mask4(spread, 0x08040201u)));
            o.w[2 * l + 1] = static_cast<uint32_t>(ares_apply_sign4(g.y, ares_sign_mask4(spread, 0x80402010u)));
        }
        const int s0 = static_cast<int>(2 * (r.sc & 0xF) + 1);
        const int s1 = static_cast<int>(2 * (r.sc >> 4) + 1);
        const float d = 0.125f * ares_mm_f16_of(r.d);
        o.ent = make_float2(d, __int_as_float(s0 | (s1 << 16)));
        o.sw[0] = d * static_cast<float>(s0);
        o.sw[1] = d * static_cast<float>(s1);
        o.mw[0] = o.mw[1] = 0.0f;
    }
};
