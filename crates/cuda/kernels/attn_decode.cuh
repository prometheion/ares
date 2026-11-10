// Decoding stored rows of K and V to f16 tiles in shared memory, for the mma.
//
// A unit is 8 consecutive values of one group of one row (a quarter of a group).
// Thread-wise the work is: read the scale, the 8 codes out of their planes,
// multiply (exact in f32: a code of at most 8 bits times an f16 scale) and round
// once to f16, which is what the tensor cores take. The rows are in shared memory
// as they came from global memory (cp.async), 2-byte aligned inside a row.
#pragma once

#include "attn_ptx.cuh"
#include "attn_layout.cuh"

static __device__ __forceinline__ unsigned ld_u16(const uint8_t* p) {
    return *reinterpret_cast<const unsigned short*>(p);
}

static __device__ __forceinline__ int sext(unsigned u, int bits) {
    return ((int)(u << (32 - bits))) >> (32 - bits);
}

// 8 values (rotated domain, for the qB formats) of quarter `q` of group `g` of
// the row at `row`, as 8 halves.
template <int FMT>
static __device__ __forceinline__ uint4 decode_unit(const uint8_t* row, int g, int q) {
    constexpr int fmt = FMT;
    uint4 out;
    if constexpr (fmt == FMT_F16) {
        return *reinterpret_cast<const uint4*>(row + g * 64 + q * 16);
    }
    constexpr int bits = fmt_bits(fmt) > 0 ? fmt_bits(fmt) : 8;
    const uint8_t* gp = row + g * fmt_group_bytes(fmt);
    const uint8_t* codes = gp + 2;
    uint32_t res[4];
    if constexpr (bits == 8) {
        // 8 signed bytes at gp + 2 + 8 q: two 32-bit words, built from aligned loads (the
        // codes of an odd group are 4-byte aligned, those of an even group are 2 off).
        const uint8_t* c = codes + q * 8;
        unsigned w[2];
        if (g & 1) {
            w[0] = *reinterpret_cast<const unsigned*>(c);
            w[1] = *reinterpret_cast<const unsigned*>(c + 4);
        } else {
            const unsigned* a = reinterpret_cast<const unsigned*>(c - 2);
            const unsigned a0 = a[0], a1 = a[1], a2 = a[2];
            w[0] = __funnelshift_r(a0, a1, 16);
            w[1] = __funnelshift_r(a1, a2, 16);
        }
        // byte + 128 in the low byte of a half whose high byte is 0x64 is 1024 + byte + 128 (an
        // f16 with an exact integer), so minus 1152 is the signed code; times the scale is
        // one rounding of the exact product, as converting the f32 product would be.
        const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
        const __half2 sc = __half2half2(__ushort_as_half((unsigned short)ld_u16(gp)));
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned x = w[i] ^ 0x80808080u;
            const unsigned lo = __byte_perm(x, 0x64646464u, 0x5140);
            const unsigned hi = __byte_perm(x, 0x64646464u, 0x5342);
            const __half2 a = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&lo), bias), sc);
            const __half2 b = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&hi), bias), sc);
            res[2 * i] = *reinterpret_cast<const uint32_t*>(&a);
            res[2 * i + 1] = *reinterpret_cast<const uint32_t*>(&b);
        }
    } else {
        // plane 0: bits 0-3 of the codes, two per byte; the 8 codes of this unit are 4 bytes
        // (the codes of an odd group are 4-byte aligned, those of an even group 2 off)
        unsigned n0;
        if (g & 1) {
            n0 = *reinterpret_cast<const unsigned*>(codes + 4 * q);
        } else {
            const unsigned* a = reinterpret_cast<const unsigned*>(codes - 2 + 4 * q);
            n0 = __funnelshift_r(a[0], a[1], 16);
        }
        // The codes in offset binary (the top bit flipped: code + 2^(B-1)), one per byte: the
        // even codes of the unit in `e`, the odd ones in `o`.
        unsigned e, o;
        if (bits == 4) {
            n0 ^= 0x88888888u;
        }
        e = n0 & 0x0f0f0f0fu;
        o = (n0 >> 4) & 0x0f0f0f0fu;
        if (bits >= 6) {
            unsigned hi2 = ld_u16(codes + 16 + 2 * q);  // plane 1: bits 4-5, two bits per code
            if (bits == 6) hi2 ^= 0xaaaau;
            const unsigned xe = hi2 & 0x3333u, xo = (hi2 >> 2) & 0x3333u;
            e |= ((xe & 0x0003u) << 4) | ((xe & 0x0030u) << 8) | ((xe & 0x0300u) << 12) | ((xe & 0x3000u) << 16);
            o |= ((xo & 0x0003u) << 4) | ((xo & 0x0030u) << 8) | ((xo & 0x0300u) << 12) | ((xo & 0x3000u) << 16);
        }
        if (bits == 7 || bits == 5) {
            unsigned top = bits == 7 ? codes[24 + q] : codes[16 + q];  // one bit per code
            top ^= 0xffu;                                             // the top bit, flipped
            const unsigned se = ((top & 0x55u) * 0x41041u) & 0x01010101u;
            const unsigned so = (((top >> 1) & 0x55u) * 0x41041u) & 0x01010101u;
            const int shift = bits == 7 ? 6 : 4;
            e |= se << shift;
            o |= so << shift;
        }
        const __half2 bias = __half2half2(__ushort_as_half((unsigned short)(0x6400 + (1 << (bits - 1)))));
        const __half2 sc = __half2half2(__ushort_as_half((unsigned short)ld_u16(gp)));
        __half2 h[4];
        const unsigned conv[4] = {
            __byte_perm(e, 0x64646464u, 0x5140), __byte_perm(o, 0x64646464u, 0x5140),
            __byte_perm(e, 0x64646464u, 0x5342), __byte_perm(o, 0x64646464u, 0x5342)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            h[i] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&conv[i]), bias), sc);
        }
        // h[0] = codes (0, 2), h[1] = (1, 3), h[2] = (4, 6), h[3] = (5, 7)
        const unsigned h0 = *reinterpret_cast<const unsigned*>(&h[0]), h1 = *reinterpret_cast<const unsigned*>(&h[1]);
        const unsigned h2 = *reinterpret_cast<const unsigned*>(&h[2]), h3 = *reinterpret_cast<const unsigned*>(&h[3]);
        res[0] = __byte_perm(h0, h1, 0x5410);
        res[1] = __byte_perm(h0, h1, 0x7632);
        res[2] = __byte_perm(h2, h3, 0x5410);
        res[3] = __byte_perm(h2, h3, 0x7632);
    }
    out.x = res[0];
    out.y = res[1];
    out.z = res[2];
    out.w = res[3];
    return out;
}

// Decodes the rows of one side of a chunk, `nc` cells, to the f16 tile. Thread t of the
// `nt` threads takes the units (group, quarter) = t % (D / 8) of the rows t / (D / 8),
// then every (nt / (D / 8))-th row: the unit is the same for all of its rows, so the
// address arithmetic and the choice of decoder are outside the loop.
template <int FMT, int D, int NT, int CHUNK>
static __device__ __forceinline__ void decode_side(
    const uint8_t* raw, int rb, const uint32_t* meta, int nc, __half* tile, int qs, int t) {
    constexpr int UNITS = D / 8;
    constexpr int STEP = NT / UNITS;
    const int ug = t % UNITS, first = t / UNITS;
    const int g = ug >> 2, q = ug & 3;
    // Unrolled: the rows are independent, so their loads are in flight together.
    uint4 val[(CHUNK + STEP - 1) / STEP];
#pragma unroll
    for (int i = 0; i < (CHUNK + STEP - 1) / STEP; ++i) {
        const int slot = first + i * STEP;
        val[i] = make_uint4(0, 0, 0, 0);
        if (slot < nc && meta[slot * 3] != 0xffffffffu) val[i] = decode_unit<FMT>(raw + (size_t)slot * rb, g, q);
    }
#pragma unroll
    for (int i = 0; i < (CHUNK + STEP - 1) / STEP; ++i) {
        const int slot = first + i * STEP;
        if (slot < nc) *reinterpret_cast<uint4*>(tile + slot * qs + ug * 8) = val[i];
    }
}

template <int D, int NT, int CHUNK>
static __device__ __forceinline__ void decode_side_any(
    int fmt, const uint8_t* raw, int rb, const uint32_t* meta, int nc, __half* tile, int qs, int t) {
    switch (fmt) {
        case FMT_F16: decode_side<FMT_F16, D, NT, CHUNK>(raw, rb, meta, nc, tile, qs, t); break;
        case FMT_Q8: decode_side<FMT_Q8, D, NT, CHUNK>(raw, rb, meta, nc, tile, qs, t); break;
        case FMT_Q7: decode_side<FMT_Q7, D, NT, CHUNK>(raw, rb, meta, nc, tile, qs, t); break;
        case FMT_Q6: decode_side<FMT_Q6, D, NT, CHUNK>(raw, rb, meta, nc, tile, qs, t); break;
        case FMT_Q5: decode_side<FMT_Q5, D, NT, CHUNK>(raw, rb, meta, nc, tile, qs, t); break;
        default: decode_side<FMT_Q4, D, NT, CHUNK>(raw, rb, meta, nc, tile, qs, t); break;
    }
}
