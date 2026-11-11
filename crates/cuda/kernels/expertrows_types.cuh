// The weight types of the routed experts for the tensor-core kernels: how a thread
// turns the file's bytes into the f16 operand of `mma`.
//
// The weights are the A operand of `mma.m16n8k16` (16 weight rows by 16 values of K),
// the activations the B operand. A thread of a warp is lane = 4 g + t; it holds the
// weights of rows g and g + 8 for the K positions {2t, 2t + 1, 2t + 8, 2t + 9} of each
// slab of 16. Nothing says which values of the real K those are: if the activations
// are read in the same order, the sums are the same. So a thread takes whatever its
// decoder finds convenient:
//
// - The i-quants (gate and up, 256 values to a block): thread t of a quad owns one
//   sub-block of 32 values of every group of 128, sub-block 4 c + t of the row for
//   chunk c. One codebook entry (8 values, one `group`) is then the 8 values a
//   thread feeds to two `mma`, and the thread loads exactly the bytes its sub-block
//   uses (`Raw`). Step l of a chunk uses group l of every thread, so the 16 bytes of
//   activations it needs are those at 16 * (4 t + l) of the chunk.
// - Q2_0 (down, 64 values to a block): thread t takes values 16 t to 16 t + 15 of
//   every block, one 32-bit word of codes. The word gives the half2 (value j, value
//   j + 8) with one shift and mask, so the hidden values are stored in that order
//   (the gate and up kernel writes them so).
//
// A decoded value is the f16 `scale * magnitude`, rounded once: the codebook
// magnitudes are small integers, so they come out of the bytes exactly as f16 `1024 +
// m` (0x64 in the high byte, the trick of Marlin) and one fused multiply-add with `-1024
// * scale` makes `scale * m`.
#pragma once

#include "expertrows_layout.cuh"
#include "iq_tables.cuh"

static __device__ __forceinline__ void ares_er_copy(uint8_t* dst, const void* src, uint32_t bytes) {
    for (uint32_t i = threadIdx.x * 16; i < bytes; i += blockDim.x * 16) {
        *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(src) + i);
    }
}

// 256 entries of 16 bytes, entry s being the four f16x2 masks (0x8000 where the bit
// of s is set) of the eight signs of a sign byte s, value j in bit j.
static __device__ __forceinline__ void ares_er_sign_masks(uint8_t* dst) {
    for (uint32_t s = threadIdx.x; s < 256; s += blockDim.x) {
        uint4 m;
        m.x = ((s & 1u) << 15) | (((s >> 1) & 1u) << 31);
        m.y = (((s >> 2) & 1u) << 15) | (((s >> 3) & 1u) << 31);
        m.z = (((s >> 4) & 1u) << 15) | (((s >> 5) & 1u) << 31);
        m.w = (((s >> 6) & 1u) << 15) | (((s >> 7) & 1u) << 31);
        reinterpret_cast<uint4*>(dst)[s] = m;
    }
}

// The scales of a sub-block: the f16x2 `scale` for each half of 16 values and
// `-1024 * scale`.
struct AresErScales {
    uint32_t sc[2];
    uint32_t nsc[2];
};

static __device__ __forceinline__ uint32_t ares_er_neg1024(uint32_t sc) {
    return ares_er_h2_mul(sc, 0xE400E400u);  // -1024 as f16x2
}

static __device__ __forceinline__ float ares_er_f16_at(uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// ---------------------------------------------------------------------------
// IQ2_S: 256 values in 82 bytes. A sub-block ib (32 values): the indices qs[4 ib + l],
// the sign bytes qs[32 + 4 ib + l], qh[ib], scales[ib], l = 0..3, one group of 8 values
// each. value = d * (2 s + 1) / 8 * magnitude * sign, s the nibble of the scale byte
// for the half (l / 2).
struct ErIQ2S {
    static constexpr uint32_t BLOCK_BYTES = 82;
    static constexpr uint32_t GRID_BYTES = 1024 * 8;
    static constexpr uint32_t TABLE_BYTES = GRID_BYTES + 256 * 16;

    struct Raw {
        uint32_t qs, signs, qh, sc, d;
    };

    static __device__ __forceinline__ void init(uint8_t* tab) {
        ares_er_copy(tab, ares_iq2s_grid, GRID_BYTES);
        ares_er_sign_masks(tab + GRID_BYTES);
    }

    // Sub-block `sb` of the row (counting 32 values from its start).
    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t sb) {
        const uint8_t* blk = row + static_cast<size_t>(sb >> 3) * BLOCK_BYTES;
        const uint32_t ib = sb & 7;
        Raw r;
        r.qs = ares_ld_stream_u32_a2(blk + 2 + 4 * ib);
        r.signs = ares_ld_stream_u32_a2(blk + 34 + 4 * ib);
        r.qh = ares_ld_stream_u8(blk + 66 + ib);
        r.sc = ares_ld_stream_u8(blk + 74 + ib);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ AresErScales scales(const Raw& r) {
        const float d8 = ares_er_f16_at(r.d) * 0.125f;
        AresErScales s;
        s.sc[0] = ares_er_h2_of(static_cast<float>(2 * (r.sc & 0xF) + 1) * d8);
        s.sc[1] = ares_er_h2_of(static_cast<float>(2 * (r.sc >> 4) + 1) * d8);
        s.nsc[0] = ares_er_neg1024(s.sc[0]);
        s.nsc[1] = ares_er_neg1024(s.sc[1]);
        return s;
    }

    static __device__ __forceinline__ void group(const Raw& r, const AresErScales& s, int l, const uint8_t* tab, uint32_t (&h)[4]) {
        const uint32_t idx = ((r.qs >> (8 * l)) & 0xFF) | ((r.qh << (8 - 2 * l)) & 0x300);
        const uint2 g = reinterpret_cast<const uint2*>(tab)[idx];
        const uint4 m = reinterpret_cast<const uint4*>(tab + GRID_BYTES)[(r.signs >> (8 * l)) & 0xFF];
        const uint32_t sc = s.sc[l >> 1], nsc = s.nsc[l >> 1];
        h[0] = ares_er_h2_fma(ares_er_magic(g.x, 0x5140), sc, nsc) ^ m.x;
        h[1] = ares_er_h2_fma(ares_er_magic(g.x, 0x5342), sc, nsc) ^ m.y;
        h[2] = ares_er_h2_fma(ares_er_magic(g.y, 0x5140), sc, nsc) ^ m.z;
        h[3] = ares_er_h2_fma(ares_er_magic(g.y, 0x5342), sc, nsc) ^ m.w;
    }
};

// ---------------------------------------------------------------------------
// IQ2_XXS: 256 values in 66 bytes: f16 d, then per sub-block of 32 values 8 bytes: four
// codebook indices and a u32 with four 7-bit sign indices and a 4-bit scale on top.
// value = d * (2 s + 1) / 8 * magnitude * sign.
struct ErIQ2XXS {
    static constexpr uint32_t BLOCK_BYTES = 66;
    static constexpr uint32_t GRID_BYTES = 256 * 8;
    static constexpr uint32_t TABLE_BYTES = GRID_BYTES + 256 * 16;

    struct Raw {
        uint32_t idx, aux, d;
    };

    static __device__ __forceinline__ void init(uint8_t* tab) {
        ares_er_copy(tab, ares_iq2xxs_grid, GRID_BYTES);
        ares_er_sign_masks(tab + GRID_BYTES);
    }

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t sb) {
        const uint8_t* blk = row + static_cast<size_t>(sb >> 3) * BLOCK_BYTES;
        const uint32_t ib = sb & 7;
        Raw r;
        r.idx = ares_ld_stream_u32_a2(blk + 2 + 8 * ib);
        r.aux = ares_ld_stream_u32_a2(blk + 6 + 8 * ib);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    static __device__ __forceinline__ AresErScales scales(const Raw& r) {
        const float s = static_cast<float>(2 * (r.aux >> 28) + 1) * ares_er_f16_at(r.d) * 0.125f;
        AresErScales out;
        out.sc[0] = out.sc[1] = ares_er_h2_of(s);
        out.nsc[0] = out.nsc[1] = ares_er_neg1024(out.sc[0]);
        return out;
    }

    static __device__ __forceinline__ void group(const Raw& r, const AresErScales& s, int l, const uint8_t* tab, uint32_t (&h)[4]) {
        const uint2 g = reinterpret_cast<const uint2*>(tab)[(r.idx >> (8 * l)) & 0xFF];
        // ggml's ksigns: the eighth sign makes the number of negative values even.
        const uint32_t s7 = (r.aux >> (7 * l)) & 127;
        const uint4 m = reinterpret_cast<const uint4*>(tab + GRID_BYTES)[s7 ^ ((__popc(s7) & 1u) << 7)];
        const uint32_t sc = s.sc[0], nsc = s.nsc[0];
        h[0] = ares_er_h2_fma(ares_er_magic(g.x, 0x5140), sc, nsc) ^ m.x;
        h[1] = ares_er_h2_fma(ares_er_magic(g.x, 0x5342), sc, nsc) ^ m.y;
        h[2] = ares_er_h2_fma(ares_er_magic(g.y, 0x5140), sc, nsc) ^ m.z;
        h[3] = ares_er_h2_fma(ares_er_magic(g.y, 0x5342), sc, nsc) ^ m.w;
    }
};

// ---------------------------------------------------------------------------
// IQ1_M: 256 values in 56 bytes, 8-byte aligned: 32 bytes of low index bits, 16 of high
// index bits and delta signs, four u16 of scales whose top nibbles spread the f16 d. A
// sub-block ib (32 values): qs[4 ib + l], qh[2 ib .. 2 ib + 1], and a scale (3 bits) for
// each half of 16 values. A group of 8 values is the codebook's bytes in {-1, 0, 1}
// plus a delta of +-1/8 (the sign bit of the group), times d * (2 s + 1).
struct ErIQ1M {
    static constexpr uint32_t BLOCK_BYTES = 56;
    static constexpr uint32_t GRID_BYTES = 2048 * 8;
    static constexpr uint32_t TABLE_BYTES = GRID_BYTES;

    struct Raw {
        uint32_t qs, qh, ib;
        uint2 sc;
    };

    static __device__ __forceinline__ void init(uint8_t* tab) {
        ares_er_copy(tab, ares_iq1s_grid, GRID_BYTES);
    }

    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t sb) {
        const uint8_t* blk = row + static_cast<size_t>(sb >> 3) * BLOCK_BYTES;
        Raw r;
        r.ib = sb & 7;
        r.qs = ares_ld_stream_u32(blk + 4 * r.ib);
        r.qh = ares_ld_stream_u16(blk + 32 + 2 * r.ib);
        r.sc = ares_ld_stream_v2(blk + 48);
        return r;
    }

    // `sc` is d * (2 s + 1) of each half; `nsc` is that divided by 8, the delta.
    static __device__ __forceinline__ AresErScales scales(const Raw& r) {
        const uint32_t sc0 = r.sc.x & 0xFFFF, sc1 = r.sc.x >> 16, sc2 = r.sc.y & 0xFFFF, sc3 = r.sc.y >> 16;
        const uint32_t d16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
        const float d = ares_er_f16_at(d16);
        const uint32_t two = r.ib >> 1;
        const uint32_t word = ((two >> 1) ? r.sc.y : r.sc.x) >> (16 * (two & 1));
        const uint32_t shift = 6 * (r.ib & 1);
        AresErScales s;
        s.sc[0] = ares_er_h2_of(d * static_cast<float>(2 * ((word >> shift) & 7) + 1));
        s.sc[1] = ares_er_h2_of(d * static_cast<float>(2 * ((word >> (shift + 3)) & 7) + 1));
        s.nsc[0] = ares_er_h2_mul(s.sc[0], 0x30003000u);  // 0.125
        s.nsc[1] = ares_er_h2_mul(s.sc[1], 0x30003000u);
        return s;
    }

    static __device__ __forceinline__ void group(const Raw& r, const AresErScales& s, int l, const uint8_t* tab, uint32_t (&h)[4]) {
        const uint32_t part = (r.qh >> (8 * (l >> 1))) & 0xFF;
        const uint32_t idx = ((r.qs >> (8 * l)) & 0xFF) | ((part << (8 - 4 * (l & 1))) & 0x700);
        const uint2 g = reinterpret_cast<const uint2*>(tab)[idx];
        const uint32_t sc = s.sc[l >> 1];
        const uint32_t delta = ((part >> (3 + 4 * (l & 1))) & 1u) ? (s.nsc[l >> 1] ^ 0x80008000u) : s.nsc[l >> 1];
        const uint32_t x = g.x ^ 0x80808080u, y = g.y ^ 0x80808080u;
        h[0] = ares_er_h2_fma(ares_er_h2_sub(ares_er_magic(x, 0x5140), 0x64806480u), sc, delta);
        h[1] = ares_er_h2_fma(ares_er_h2_sub(ares_er_magic(x, 0x5342), 0x64806480u), sc, delta);
        h[2] = ares_er_h2_fma(ares_er_h2_sub(ares_er_magic(y, 0x5140), 0x64806480u), sc, delta);
        h[3] = ares_er_h2_fma(ares_er_h2_sub(ares_er_magic(y, 0x5342), 0x64806480u), sc, delta);
    }
};
