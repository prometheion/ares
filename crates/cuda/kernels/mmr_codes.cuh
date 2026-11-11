// The device layout of the types that are plain codes with a scale, and for some a
// minimum, per group of 16 or 32 values (Q4_K, Q5_K, Q6_K, Q3_K, Q2_K, Q4_0, Q4_1,
// Q5_0, Q5_1, Q8_0), and of the types whose 4-bit codes index 16 levels that are not
// evenly spaced (IQ4_XS, IQ4_NL): one template for the kernels of mmr.cuh, with the
// bits of a code, the length of a group and what a group's header holds as
// parameters. Every layout takes the bytes of the file's, with the rows rounded up
// to a multiple of 16.
//
// A value is `S * code + C` with S and C of its group, which a lane computes for its
// two rows from the group's header: code, S and C are f16, the code exactly (it is
// put into the mantissa of 1024, f16 bits 0x6400 | code, and the bias comes off), so
// one `fma` per pair of values makes the weight with one rounding.
//
//   type    code            S                C
//   Q4_K    4 bits          d * sc           -dmin * m          (sc, m 6 bits, per 32)
//   Q5_K    4 + 1 bits      d * sc           -dmin * m
//   Q6_K    4 + 2 bits      d * sc           -32 S              (sc int8, per 16)
//   Q3_K    2 + 1 bits      d * (sc - 32)    -4 S               (sc 6 bits, per 16)
//   Q2_K    2 bits          d * sc           -dmin * m          (sc, m 4 bits, per 16)
//   Q4_0    4 bits          d                -8 S               (per 32)
//   Q4_1    4 bits          d                m
//   Q5_0    4 + 1 bits      d                -16 S
//   Q5_1    4 + 1 bits      d                m
//   Q8_0    8 bits          d                -128 S             (the code is q + 128)
//   IQ4_XS  level + 128     d * (ls - 32)    -128 S             (ls 6 bits, per 32)
//   IQ4_NL  level + 128     d                -128 S
//
// The layout, for each tile of 16 rows:
//   low plane   the low LB bits (4, 2 or 8) of the codes. For each step of 64 values
//               of K: 32 lanes x LB words. Lane (g = lane / 4, t = lane % 4) holds
//               the 8 codes of each slab kk of 16 values that its registers of the
//               `mma.m16n8k16` A fragment hold:
//                 a0: row g, values 2t, 2t+1        a1: row g+8, values 2t, 2t+1
//                 a2: row g, values 2t+8, 2t+9      a3: row g+8, values 2t+8, 2t+9
//               with the even value of a_i in the low half of a word and the odd one
//               16 bits above it, so that a shift, a mask and the bias make the f16
//               pair (1024 + even, 1024 + odd) of the register:
//                 LB 4   word kk, bit 4 i
//                 LB 2   word kk / 2, bit 8 (kk % 2) + 2 i
//                 LB 8   word 2 kk + i / 2, bit 8 (i % 2)
//   high plane  the HB bits (1 or 2) above them, the same way: HB words per lane
//               and step (HB 1: bit 4 kk + i; HB 2: as LB 2).
//   headers     what S and C are made of, the 16 rows' together; see each `Hdr`.
// The planes of all tiles come first, low then high, then the headers. The steps of
// a tile follow each other, so a warp reads each plane as one stream. The scales
// stay as small as in the file (expanded to f16 they would cost half a bit per
// weight) and are multiplied with d in registers.
//
// Q4_K's layout is the one the Vulkan backend makes too (shaders/repack_q4_K.comp).
#pragma once

#include "mmr.cuh"

// ---------------------------------------------------------------------------
// The planes.

// Of the words a lane holds of a plane of `bits` bits per code for one step: the
// word and the bit of the even value of register a_i of slab kk.
static __host__ __device__ constexpr int ares_plane_word(int bits, int kk, int i) {
    return bits == 8 ? 2 * kk + i / 2 : bits == 4 ? kk : bits == 2 ? kk / 2 : 0;
}
static __host__ __device__ constexpr int ares_plane_shift(int bits, int kk, int i) {
    return bits == 8 ? 8 * (i % 2) : bits == 4 ? 4 * i : bits == 2 ? 8 * (kk % 2) + 2 * i : 4 * kk + i;
}

// N words of a plane (the lane's share of a step). The loads of 16 bytes pass L1 by
// when the weights are read once (STREAM); narrower ones are cached and evicted
// first, see mmvq.cuh.
template <int N, bool STREAM>
static __device__ __forceinline__ void ares_ld_plane(const uint32_t* p, uint32_t (&w)[N]) {
    if constexpr (N % 4 == 0) {
#pragma unroll
        for (int v = 0; v < N / 4; v++) {
            const uint4 x = STREAM ? ares_ld_stream_v4(p + 4 * v) : __ldg(reinterpret_cast<const uint4*>(p + 4 * v));
            w[4 * v] = x.x;
            w[4 * v + 1] = x.y;
            w[4 * v + 2] = x.z;
            w[4 * v + 3] = x.w;
        }
    } else if constexpr (N == 2) {
        const uint2 x = STREAM ? ares_ld_stream_v2(p) : __ldg(reinterpret_cast<const uint2*>(p));
        w[0] = x.x;
        w[1] = x.y;
    } else {
        static_assert(N == 1, "planes are 1, 2, 4 or 8 words per lane and step");
        w[0] = STREAM ? ares_ld_stream_u32(p) : __ldg(p);
    }
}

// The byte b of two words as (b0, 0x64, b1, 0x64): the f16 pair (1024 + b0, 1024 + b1).
static __device__ __forceinline__ uint32_t ares_h2_of_bytes(uint32_t w0, uint32_t w1, int b0, int b1) {
    return __byte_perm(__byte_perm(w0, w1, ((4 + b1) << 4) | b0), 0x64646464u, 0x4140);
}
// (x, x) of the low and of the high f16 of v.
static __device__ __forceinline__ uint32_t ares_h2_dup_lo(uint32_t v) {
    return __byte_perm(v, v, 0x1010);
}
static __device__ __forceinline__ uint32_t ares_h2_dup_hi(uint32_t v) {
    return __byte_perm(v, v, 0x3232);
}
// The f16 pair (v, v) of a small integer or 1024 + it, as bits.
static __host__ __device__ constexpr uint32_t ares_h2_const(int v) {
    // Integers of magnitude up to 2048 are exact in f16: sign, exponent of the top bit, mantissa.
    uint32_t a = v < 0 ? -v : v, bits = 0;
    if (a != 0) {
        int e = 0;
        while ((a >> (e + 1)) != 0) {
            e++;
        }
        bits = static_cast<uint32_t>(e + 15) << 10 | ((a << (10 - e)) & 0x3FF);
    }
    bits |= v < 0 ? 0x8000u : 0u;
    return bits | bits << 16;
}
// The bits of an f16 pair as `__half2` and back (a move), and the pair (x, x) of the low
// (H = 0) or high (H = 1) half of one: an operand of `HFMA2` that costs no instruction.
static __device__ __forceinline__ __half2 ares_h2_of(uint32_t bits) {
    return *reinterpret_cast<__half2*>(&bits);
}
static __device__ __forceinline__ uint32_t ares_h2_bits(__half2 v) {
    return *reinterpret_cast<uint32_t*>(&v);
}
template <int H>
static __device__ __forceinline__ __half2 ares_h2_lane(uint32_t bits) {
    if constexpr (H == 0) {
        return __low2half2(ares_h2_of(bits));
    } else {
        return __high2half2(ares_h2_of(bits));
    }
}
// 2^-s as an f16 pair, 0 <= s < 15.
static __host__ __device__ constexpr uint32_t ares_h2_pow2_down(int s) {
    return (static_cast<uint32_t>(15 - s) << 10) * 0x00010001u;
}

// ---------------------------------------------------------------------------
// The headers. Each `Hdr` says how the scales of a tile lie in the layout
// (`words`, `word`: the 32-bit words of its section and what each holds, for the
// rearrangement), what a lane keeps of them (`State`) and how S and C of a group
// come out of that (`scales<j, kk>`, called at the first slab of every group; j is
// the step within the super-block of 256 values). `hdr` in `open` is the section's
// start, `tile` the tile and `sb` the super-block the lane starts at; GROUP the
// values of a group.

// Q4_K, Q5_K: per super-block and row the 16 bytes of the file: d, dmin and 12 bytes
// of 6-bit scales and minimums of 8 groups of 32.
struct AresHdrK4 {
    static constexpr int GROUP = 32;
    static __host__ __device__ size_t words(size_t n_tiles, uint32_t n_in) {
        return n_tiles * (n_in / 256) * 16 * 4;
    }
    // `at(row, block, offset)` is the address of a byte of the file's block.
    template <class At>
    static __device__ uint32_t word(const At& at, uint32_t n_in, size_t idx) {
        const uint32_t n_sb = n_in / 256;
        const uint32_t wi = idx % 4, r16 = (idx / 4) % 16, sb = (idx / 64) % n_sb;
        const uint32_t tile = idx / 64 / n_sb;
        return at.u32(tile * 16 + r16, sb, 4 * wi);
    }
    struct State {
        const uint4* p;   // the header of row g of the tile, super-block 0; row g + 8 at + 8
        uint4 hd[2];      // headers of the next super-block
        uint32_t sc[2][2], mn[2][2];  // scales and minimums of the super-block in work, as bytes
        uint32_t dd2[2];              // (d, -dmin) as f16x2
    };
    static __device__ __forceinline__ void open(State& s, const uint8_t* hdr, size_t, uint32_t tile, uint32_t n_in, uint32_t lane, uint32_t sb) {
        s.p = reinterpret_cast<const uint4*>(hdr) + (static_cast<size_t>(tile) * (n_in / 256)) * 16 + (lane >> 2);
        s.hd[0] = __ldg(s.p + sb * 16);
        s.hd[1] = __ldg(s.p + sb * 16 + 8);
    }
    static __device__ __forceinline__ void super_block(State& s, uint32_t sb, uint32_t n_sb) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            ares_mm_scales_k4(s.hd[h], s.sc[h], s.mn[h]);
            s.dd2[h] = (s.hd[h].x & 0xFFFFu) | ((s.hd[h].x & 0xFFFF0000u) ^ 0x80000000u);
        }
        if (sb + 1 < n_sb) {
            s.hd[0] = __ldg(s.p + (sb + 1) * 16);
            s.hd[1] = __ldg(s.p + (sb + 1) * 16 + 8);
        }
    }
    static __device__ __forceinline__ void step(State&, uint32_t, uint32_t) {}
    template <int j, int kk>
    static __device__ __forceinline__ void scales(const State& s, uint32_t (&S)[2], uint32_t (&C)[2]) {
        constexpr int o = 2 * (j & 1) + kk / 2;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            // (1024 + scale, 1024 + minimum), then (scale * d, minimum * -dmin).
            const uint32_t x = ares_h2_of_bytes(s.sc[h][j >> 1], s.mn[h][j >> 1], o, o);
            const uint32_t v = ares_h2_mul(ares_h2_sub(x, 0x64006400u), s.dd2[h]);
            S[h] = ares_h2_dup_lo(v);
            C[h] = ares_h2_dup_hi(v);
        }
    }
    // What the few-row kernels (AresFewCodes) use instead of `scales`: a weight is
    // `S * code + C` (MIN) or `S * (code - OFFSET)`, `scale<j, kk>` keeps per row of the
    // lane (g: h = 0, g + 8: h = 1) the pair (S, C), and `weight<h>` makes the f16 pair of
    // the row's two values from the pair of codes (code - OFFSET for the types without a
    // minimum), the scale as an operand of its `HFMA2` that costs no instruction.
    static constexpr bool MIN = true;
    static constexpr int OFFSET = 0;
    struct Scale {
        uint32_t v[2];
    };
    template <int j, int kk>
    static __device__ __forceinline__ Scale scale(const State& s) {
        constexpr int o = 2 * (j & 1) + kk / 2;
        Scale r;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t x = ares_h2_of_bytes(s.sc[h][j >> 1], s.mn[h][j >> 1], o, o);
            r.v[h] = ares_h2_mul(ares_h2_sub(x, 0x64006400u), s.dd2[h]);
        }
        return r;
    }
    template <int h>
    static __device__ __forceinline__ uint32_t weight(const Scale& r, uint32_t c) {
        return ares_h2_bits(__hfma2(ares_h2_of(c), ares_h2_lane<0>(r.v[h]), ares_h2_lane<1>(r.v[h])));
    }
};

// Q6_K, Q3_K, Q2_K: a scale byte per group of 16 values, 16 per super-block and row,
// and d (Q2_K: d and dmin) per super-block and row.
//   SCALE_WORDS   words of scales per row and super-block in the layout (4; Q3_K: 3,
//                 its 16 scales of 6 bits packed as in the file)
//   SCALE_AT, D_AT  where they are in the file's block
//   SCALE_BIAS    what the byte of a scale is above the scale (Q6_K: 128, the int8
//                 scales are stored + 128; Q3_K: 32)
//   CODE_OFFSET   value = S * (code - CODE_OFFSET); Q2_K instead has a minimum in the
//                 high nibble of the scale byte (MIN_NIBBLE)
// The sections: scales (16 rows x SCALE_WORDS words per tile and super-block), then
// per tile, super-block and g the pair (d of row g, d of row g + 8), for Q2_K
// followed by the pair of dmin.
template <int SCALE_WORDS, int SCALE_AT, int D_AT, int SCALE_BIAS, int CODE_OFFSET, bool MIN_NIBBLE>
struct AresHdrK16 {
    static constexpr int GROUP = 16;
    static constexpr int DW = MIN_NIBBLE ? 2 : 1;
    static __host__ __device__ size_t scale_words(size_t n_tiles, uint32_t n_in) {
        return n_tiles * (n_in / 256) * 16 * SCALE_WORDS;
    }
    static __host__ __device__ size_t words(size_t n_tiles, uint32_t n_in) {
        return scale_words(n_tiles, n_in) + n_tiles * (n_in / 256) * 8 * DW;
    }
    template <class At>
    static __device__ uint32_t word(const At& at, uint32_t n_in, size_t idx) {
        const uint32_t n_sb = n_in / 256;
        const size_t n_scale = scale_words(at.n_tiles, n_in);
        if (idx < n_scale) {
            const uint32_t wi = idx % SCALE_WORDS, r16 = (idx / SCALE_WORDS) % 16;
            const uint32_t sb = (idx / SCALE_WORDS / 16) % n_sb, tile = idx / SCALE_WORDS / 16 / n_sb;
            const uint32_t v = at.u32(tile * 16 + r16, sb, SCALE_AT + 4 * wi);
            return SCALE_BIAS == 128 ? v ^ 0x80808080u : v;
        }
        const size_t di = idx - n_scale;
        const uint32_t wi = di % DW, g = (di / DW) % 8, sb = (di / DW / 8) % n_sb, tile = di / DW / 8 / n_sb;
        return at.u16(tile * 16 + g, sb, D_AT + 2 * wi) | at.u16(tile * 16 + g + 8, sb, D_AT + 2 * wi) << 16;
    }
    struct State {
        const uint32_t* p;   // the scale words of row g of the tile, super-block 0
        const uint32_t* pd;  // the pair of d of rows g and g + 8
        uint32_t nx[2][SCALE_WORDS], nd[DW];  // of the next super-block
        uint32_t sc[2][4];                    // the 16 scale bytes of rows g and g + 8
        uint32_t dd, ndmin;                   // (d of row g, of row g + 8); Q2_K: -dmin likewise
    };
    static __device__ __forceinline__ void load(State& s, uint32_t sb) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
#pragma unroll
            for (int v = 0; v < SCALE_WORDS; v++) {
                s.nx[h][v] = __ldg(s.p + (static_cast<size_t>(sb) * 16 + 8 * h) * SCALE_WORDS + v);
            }
        }
#pragma unroll
        for (int v = 0; v < DW; v++) {
            s.nd[v] = __ldg(s.pd + static_cast<size_t>(sb) * 8 * DW + v);
        }
    }
    static __device__ __forceinline__ void open(State& s, const uint8_t* hdr, size_t n_tiles, uint32_t tile, uint32_t n_in, uint32_t lane, uint32_t sb) {
        const uint32_t n_sb = n_in / 256, g = lane >> 2;
        const uint32_t* base = reinterpret_cast<const uint32_t*>(hdr);
        s.p = base + (static_cast<size_t>(tile) * n_sb * 16 + g) * SCALE_WORDS;
        s.pd = base + scale_words(n_tiles, n_in) + (static_cast<size_t>(tile) * n_sb * 8 + g) * DW;
        load(s, sb);
    }
    static __device__ __forceinline__ void super_block(State& s, uint32_t sb, uint32_t n_sb) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if constexpr (SCALE_WORDS == 3) {
                // Q3_K's 12 bytes: the low 4 bits of scales 0..7 and 8..15 in the nibbles of
                // words 0 and 1, the high 2 bits of all 16 in word 2.
                const uint32_t a0 = s.nx[h][0], a1 = s.nx[h][1], t = s.nx[h][2];
                s.sc[h][0] = (a0 & 0x0F0F0F0Fu) | ((t & 0x03030303u) << 4);
                s.sc[h][1] = (a1 & 0x0F0F0F0Fu) | (((t >> 2) & 0x03030303u) << 4);
                s.sc[h][2] = ((a0 >> 4) & 0x0F0F0F0Fu) | (((t >> 4) & 0x03030303u) << 4);
                s.sc[h][3] = ((a1 >> 4) & 0x0F0F0F0Fu) | (((t >> 6) & 0x03030303u) << 4);
            } else {
#pragma unroll
                for (int v = 0; v < 4; v++) {
                    s.sc[h][v] = s.nx[h][v];
                }
            }
        }
        s.dd = s.nd[0];
        if constexpr (MIN_NIBBLE) {
            s.ndmin = s.nd[1] ^ 0x80008000u;
        }
        if (sb + 1 < n_sb) {
            load(s, sb + 1);
        }
    }
    static __device__ __forceinline__ void step(State&, uint32_t, uint32_t) {}
    template <int j, int kk>
    static __device__ __forceinline__ void scales(const State& s, uint32_t (&S)[2], uint32_t (&C)[2]) {
        // Byte kk of word j of both rows: the scale of the group.
        if constexpr (MIN_NIBBLE) {
            const uint32_t b = __byte_perm(s.sc[0][j], s.sc[1][j], ((4 + kk) << 4) | kk);
            const uint32_t xs = __byte_perm(b & 0x0F0Fu, 0x64646464u, 0x4140);
            const uint32_t xm = __byte_perm((b >> 4) & 0x0F0Fu, 0x64646464u, 0x4140);
            const uint32_t vs = ares_h2_mul(ares_h2_sub(xs, 0x64006400u), s.dd);
            const uint32_t vm = ares_h2_mul(ares_h2_sub(xm, 0x64006400u), s.ndmin);
            S[0] = ares_h2_dup_lo(vs);
            S[1] = ares_h2_dup_hi(vs);
            C[0] = ares_h2_dup_lo(vm);
            C[1] = ares_h2_dup_hi(vm);
        } else {
            const uint32_t x = ares_h2_of_bytes(s.sc[0][j], s.sc[1][j], kk, kk);
            const uint32_t v = ares_h2_mul(ares_h2_sub(x, ares_h2_const(1024 + SCALE_BIAS)), s.dd);
            S[0] = ares_h2_dup_lo(v);
            S[1] = ares_h2_dup_hi(v);
            C[0] = ares_h2_mul(S[0], ares_h2_const(-CODE_OFFSET));
            C[1] = ares_h2_mul(S[1], ares_h2_const(-CODE_OFFSET));
        }
    }
    // For the few-row kernels, see AresHdrK4: (S of row g, of row g + 8), for Q2_K also
    // (C of row g, of row g + 8).
    static constexpr bool MIN = MIN_NIBBLE;
    static constexpr int OFFSET = MIN_NIBBLE ? 0 : CODE_OFFSET;
    struct Scale {
        uint32_t v[2];
    };
    template <int j, int kk>
    static __device__ __forceinline__ Scale scale(const State& s) {
        Scale r;
        if constexpr (MIN_NIBBLE) {
            const uint32_t b = __byte_perm(s.sc[0][j], s.sc[1][j], ((4 + kk) << 4) | kk);
            const uint32_t xs = __byte_perm(b & 0x0F0Fu, 0x64646464u, 0x4140);
            const uint32_t xm = __byte_perm((b >> 4) & 0x0F0Fu, 0x64646464u, 0x4140);
            r.v[0] = ares_h2_mul(ares_h2_sub(xs, 0x64006400u), s.dd);
            r.v[1] = ares_h2_mul(ares_h2_sub(xm, 0x64006400u), s.ndmin);
        } else {
            const uint32_t x = ares_h2_of_bytes(s.sc[0][j], s.sc[1][j], kk, kk);
            r.v[0] = ares_h2_mul(ares_h2_sub(x, ares_h2_const(1024 + SCALE_BIAS)), s.dd);
            r.v[1] = 0;
        }
        return r;
    }
    template <int h>
    static __device__ __forceinline__ uint32_t weight(const Scale& r, uint32_t c) {
        if constexpr (MIN_NIBBLE) {
            return ares_h2_bits(__hfma2(ares_h2_of(c), ares_h2_lane<h>(r.v[0]), ares_h2_lane<h>(r.v[1])));
        } else {
            return ares_h2_bits(__hmul2(ares_h2_of(c), ares_h2_lane<h>(r.v[0])));
        }
    }
};

// Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL: an f16 scale d, and for Q4_1 and Q5_1 an f16
// minimum m, per block of 32 values. Per tile, step of 64 values and g: for each of
// the step's two blocks the pair (d of row g, d of row g + 8), with M_AT >= 0
// followed by the pair of m. Without m, value = d * (code - CODE_OFFSET).
template <int D_AT, int M_AT, int CODE_OFFSET>
struct AresHdrB {
    static constexpr int GROUP = 32;
    static constexpr int MW = M_AT >= 0 ? 2 : 1;
    static __host__ __device__ size_t words(size_t n_tiles, uint32_t n_in) {
        return n_tiles * (n_in / 64) * 8 * 2 * MW;
    }
    template <class At>
    static __device__ uint32_t word(const At& at, uint32_t n_in, size_t idx) {
        const uint32_t n_steps = n_in / 64;
        const uint32_t wi = idx % MW, sub = (idx / MW) % 2, g = (idx / MW / 2) % 8;
        const uint32_t step = (idx / MW / 16) % n_steps, tile = idx / MW / 16 / n_steps;
        const uint32_t off = wi == 0 ? D_AT : M_AT;
        return at.u16(tile * 16 + g, 2 * step + sub, off) | at.u16(tile * 16 + g + 8, 2 * step + sub, off) << 16;
    }
    struct State {
        const uint32_t* p;                    // the words of g of the tile, step 0
        uint32_t cur[2 * MW], nx[2 * MW];     // of the step in work and of the next
    };
    static __device__ __forceinline__ void load(State& s, uint32_t step) {
        ares_ld_plane<2 * MW, false>(s.p + static_cast<size_t>(step) * 16 * MW, s.nx);
    }
    static __device__ __forceinline__ void open(State& s, const uint8_t* hdr, size_t, uint32_t tile, uint32_t n_in, uint32_t lane, uint32_t sb) {
        s.p = reinterpret_cast<const uint32_t*>(hdr) + (static_cast<size_t>(tile) * (n_in / 64) * 8 + (lane >> 2)) * 2 * MW;
        load(s, 4 * sb);
    }
    static __device__ __forceinline__ void super_block(State&, uint32_t, uint32_t) {}
    static __device__ __forceinline__ void step(State& s, uint32_t step, uint32_t n_steps) {
#pragma unroll
        for (int v = 0; v < 2 * MW; v++) {
            s.cur[v] = s.nx[v];
        }
        if (step + 1 < n_steps) {
            load(s, step + 1);
        }
    }
    template <int j, int kk>
    static __device__ __forceinline__ void scales(const State& s, uint32_t (&S)[2], uint32_t (&C)[2]) {
        const uint32_t d2 = s.cur[(kk / 2) * MW];
        S[0] = ares_h2_dup_lo(d2);
        S[1] = ares_h2_dup_hi(d2);
        if constexpr (M_AT >= 0) {
            const uint32_t m2 = s.cur[(kk / 2) * MW + 1];
            C[0] = ares_h2_dup_lo(m2);
            C[1] = ares_h2_dup_hi(m2);
        } else {
            C[0] = ares_h2_mul(S[0], ares_h2_const(-CODE_OFFSET));
            C[1] = ares_h2_mul(S[1], ares_h2_const(-CODE_OFFSET));
        }
    }
    // For the few-row kernels, see AresHdrK4: (d of row g, of row g + 8) and for Q4_1
    // and Q5_1 (m of row g, of row g + 8).
    static constexpr bool MIN = M_AT >= 0;
    static constexpr int OFFSET = M_AT >= 0 ? 0 : CODE_OFFSET;
    struct Scale {
        uint32_t v[2];
    };
    template <int j, int kk>
    static __device__ __forceinline__ Scale scale(const State& s) {
        Scale r;
        r.v[0] = s.cur[(kk / 2) * MW];
        r.v[1] = M_AT >= 0 ? s.cur[(kk / 2) * MW + 1] : 0u;
        return r;
    }
    template <int h>
    static __device__ __forceinline__ uint32_t weight(const Scale& r, uint32_t c) {
        if constexpr (M_AT >= 0) {
            return ares_h2_bits(__hfma2(ares_h2_of(c), ares_h2_lane<h>(r.v[0]), ares_h2_lane<h>(r.v[1])));
        } else {
            return ares_h2_bits(__hmul2(ares_h2_of(c), ares_h2_lane<h>(r.v[0])));
        }
    }
};

// IQ4_XS: per super-block and row the 8 bytes of the file: d, the high 2 bits of the
// 8 scales of 6 bits (scales_h), their low 4 bits (scales_l). value = d * (ls - 32) *
// level, the level + 128 being the code.
struct AresHdrX4 {
    static constexpr int GROUP = 32;
    static __host__ __device__ size_t words(size_t n_tiles, uint32_t n_in) {
        return n_tiles * (n_in / 256) * 16 * 2;
    }
    template <class At>
    static __device__ uint32_t word(const At& at, uint32_t n_in, size_t idx) {
        const uint32_t n_sb = n_in / 256;
        const uint32_t wi = idx % 2, r16 = (idx / 2) % 16, sb = (idx / 32) % n_sb, tile = idx / 32 / n_sb;
        return at.u32(tile * 16 + r16, sb, 4 * wi);
    }
    struct State {
        const uint2* p;      // the header of row g of the tile, super-block 0; row g + 8 at + 8
        uint2 nx[2];         // headers of the next super-block
        uint32_t sc[2][2];   // the 8 scales of rows g and g + 8, as bytes
        uint32_t dd;         // (d of row g, d of row g + 8)
    };
    static __device__ __forceinline__ void open(State& s, const uint8_t* hdr, size_t, uint32_t tile, uint32_t n_in, uint32_t lane, uint32_t sb) {
        s.p = reinterpret_cast<const uint2*>(hdr) + (static_cast<size_t>(tile) * (n_in / 256)) * 16 + (lane >> 2);
        s.nx[0] = __ldg(s.p + sb * 16);
        s.nx[1] = __ldg(s.p + sb * 16 + 8);
    }
    // Four nibbles to four bytes, and four pairs of bits to bits 4 and 5 of four bytes.
    static __device__ __forceinline__ uint32_t spread4(uint32_t x) {
        x = (x | (x << 8)) & 0x00FF00FFu;
        return (x | (x << 4)) & 0x0F0F0F0Fu;
    }
    static __device__ __forceinline__ uint32_t spread2(uint32_t y) {
        y = (y | (y << 12)) & 0x000F000Fu;
        return ((y | (y << 6)) & 0x03030303u) << 4;
    }
    static __device__ __forceinline__ void super_block(State& s, uint32_t sb, uint32_t n_sb) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t hi = s.nx[h].x >> 16, lo = s.nx[h].y;
            s.sc[h][0] = spread4(lo & 0xFFFFu) | spread2(hi & 0xFFu);
            s.sc[h][1] = spread4(lo >> 16) | spread2(hi >> 8);
        }
        s.dd = (s.nx[0].x & 0xFFFFu) | (s.nx[1].x << 16);
        if (sb + 1 < n_sb) {
            s.nx[0] = __ldg(s.p + (sb + 1) * 16);
            s.nx[1] = __ldg(s.p + (sb + 1) * 16 + 8);
        }
    }
    static __device__ __forceinline__ void step(State&, uint32_t, uint32_t) {}
    template <int j, int kk>
    static __device__ __forceinline__ void scales(const State& s, uint32_t (&S)[2], uint32_t (&C)[2]) {
        constexpr int ib = 2 * j + kk / 2;
        const uint32_t x = ares_h2_of_bytes(s.sc[0][ib / 4], s.sc[1][ib / 4], ib % 4, ib % 4);
        const uint32_t v = ares_h2_mul(ares_h2_sub(x, ares_h2_const(1024 + 32)), s.dd);
        S[0] = ares_h2_dup_lo(v);
        S[1] = ares_h2_dup_hi(v);
        C[0] = ares_h2_mul(S[0], ares_h2_const(-128));
        C[1] = ares_h2_mul(S[1], ares_h2_const(-128));
    }
    // For the few-row kernels, see AresHdrK4: (S of row g, of row g + 8).
    static constexpr bool MIN = false;
    static constexpr int OFFSET = 128;
    struct Scale {
        uint32_t v[2];
    };
    template <int j, int kk>
    static __device__ __forceinline__ Scale scale(const State& s) {
        constexpr int ib = 2 * j + kk / 2;
        Scale r;
        const uint32_t x = ares_h2_of_bytes(s.sc[0][ib / 4], s.sc[1][ib / 4], ib % 4, ib % 4);
        r.v[0] = ares_h2_mul(ares_h2_sub(x, ares_h2_const(1024 + 32)), s.dd);
        r.v[1] = 0;
        return r;
    }
    template <int h>
    static __device__ __forceinline__ uint32_t weight(const Scale& r, uint32_t c) {
        return ares_h2_bits(__hmul2(ares_h2_of(c), ares_h2_lane<h>(r.v[0])));
    }
};

// ---------------------------------------------------------------------------
// The types: the bits of a code in the two planes, the geometry of the file's block,
// the header, and the code of value k of the file's block at b. TABLE: the plane
// holds 4-bit indices of the 16 levels of IQ4_NL, and the code is the level + 128.

struct AresCodesQ4K {
    static constexpr int LB = 4, HB = 0, BLOCK_VALUES = 256, BLOCK_BYTES = 144;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrK4;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        const uint32_t byte = b[16 + 32 * (k / 64) + (k & 31)];
        return (k & 32) != 0 ? byte >> 4 : byte & 15;
    }
};
struct AresCodesQ5K {
    static constexpr int LB = 4, HB = 1, BLOCK_VALUES = 256, BLOCK_BYTES = 176;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrK4;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        const uint32_t byte = b[48 + 32 * (k / 64) + (k & 31)];
        const uint32_t low = (k & 32) != 0 ? byte >> 4 : byte & 15;
        return low | ((b[16 + (k & 31)] >> (2 * (k / 64) + ((k >> 5) & 1))) & 1) << 4;
    }
};
struct AresCodesQ6K {
    static constexpr int LB = 4, HB = 2, BLOCK_VALUES = 256, BLOCK_BYTES = 210;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrK16<4, 192, 208, 128, 32, false>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        const uint32_t half = k / 128, part = (k / 32) % 4, l = k % 32;
        const uint32_t byte = b[64 * half + l + 32 * (part & 1)];
        const uint32_t low = part < 2 ? byte & 15 : byte >> 4;
        return low | ((b[128 + 32 * half + l] >> (2 * part)) & 3) << 4;
    }
};
struct AresCodesQ3K {
    static constexpr int LB = 2, HB = 1, BLOCK_VALUES = 256, BLOCK_BYTES = 110;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrK16<3, 96, 108, 32, 4, false>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        const uint32_t half = k / 128, step = (k / 32) % 4, l = k % 32;
        return ((b[32 + 32 * half + l] >> (2 * step)) & 3) | ((b[l] >> (4 * half + step)) & 1) << 2;
    }
};
struct AresCodesQ2K {
    static constexpr int LB = 2, HB = 0, BLOCK_VALUES = 256, BLOCK_BYTES = 84;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrK16<4, 0, 80, 0, 0, true>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        return (b[16 + 32 * (k / 128) + k % 32] >> (2 * ((k / 32) % 4))) & 3;
    }
};
// The nibbles of a block of 32: values 0..15 in the low, 16..31 in the high ones.
template <int QS_AT>
static __device__ __forceinline__ uint32_t ares_code_nibble32(const uint8_t* b, uint32_t k) {
    const uint32_t byte = b[QS_AT + (k & 15)];
    return (k & 16) != 0 ? byte >> 4 : byte & 15;
}
struct AresCodesQ40 {
    static constexpr int LB = 4, HB = 0, BLOCK_VALUES = 32, BLOCK_BYTES = 18;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrB<0, -1, 8>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        return ares_code_nibble32<2>(b, k);
    }
};
struct AresCodesQ41 {
    static constexpr int LB = 4, HB = 0, BLOCK_VALUES = 32, BLOCK_BYTES = 20;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrB<0, 2, 0>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        return ares_code_nibble32<4>(b, k);
    }
};
struct AresCodesQ50 {
    static constexpr int LB = 4, HB = 1, BLOCK_VALUES = 32, BLOCK_BYTES = 22;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrB<0, -1, 16>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        const uint32_t qh = b[2] | b[3] << 8 | b[4] << 16 | static_cast<uint32_t>(b[5]) << 24;
        return ares_code_nibble32<6>(b, k) | ((qh >> k) & 1) << 4;
    }
};
struct AresCodesQ51 {
    static constexpr int LB = 4, HB = 1, BLOCK_VALUES = 32, BLOCK_BYTES = 24;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrB<0, 2, 0>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        const uint32_t qh = b[4] | b[5] << 8 | b[6] << 16 | static_cast<uint32_t>(b[7]) << 24;
        return ares_code_nibble32<8>(b, k) | ((qh >> k) & 1) << 4;
    }
};
struct AresCodesQ80 {
    static constexpr int LB = 8, HB = 0, BLOCK_VALUES = 32, BLOCK_BYTES = 34;
    static constexpr bool TABLE = false;
    using Hdr = AresHdrB<0, -1, 128>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        return b[2 + k] ^ 0x80u;
    }
};
struct AresCodesIQ4XS {
    static constexpr int LB = 4, HB = 0, BLOCK_VALUES = 256, BLOCK_BYTES = 136;
    static constexpr bool TABLE = true;
    using Hdr = AresHdrX4;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        return ares_code_nibble32<8>(b + 16 * (k / 32), k % 32);
    }
};
struct AresCodesIQ4NL {
    static constexpr int LB = 4, HB = 0, BLOCK_VALUES = 32, BLOCK_BYTES = 18;
    static constexpr bool TABLE = true;
    using Hdr = AresHdrB<0, -1, 128>;
    static __device__ uint32_t code(const uint8_t* b, uint32_t k) {
        return ares_code_nibble32<2>(b, k);
    }
};

// ---------------------------------------------------------------------------
// The rearrangement: one thread per 32-bit word of the layout, which gathers what
// the word holds from the file's bytes. It runs once per matrix, at load.

// The bytes of the file's layout; rows past the end of the matrix read as zero.
template <class P>
struct AresFileAt {
    const uint8_t* src;
    size_t row_bytes;
    uint32_t n_out;
    size_t n_tiles;
    __device__ const uint8_t* blk(uint32_t row, uint32_t block) const {
        return src + row * row_bytes + static_cast<size_t>(block) * P::BLOCK_BYTES;
    }
    __device__ uint32_t u16(uint32_t row, uint32_t block, uint32_t off) const {
        if (row >= n_out) {
            return 0;
        }
        const uint8_t* p = blk(row, block) + off;
        return p[0] | p[1] << 8;
    }
    __device__ uint32_t u32(uint32_t row, uint32_t block, uint32_t off) const {
        return u16(row, block, off) | u16(row, block, off + 2) << 16;
    }
};

// Words of the layout of a matrix: the grid of the rearrangement has this many threads.
template <class P>
static __host__ __device__ size_t ares_codes_words(uint32_t n_out, uint32_t n_in) {
    const size_t n_tiles = (n_out + 15) / 16;
    return n_tiles * (n_in / 64) * 32 * (P::LB + P::HB) + P::Hdr::words(n_tiles, n_in);
}

template <class P>
static __device__ __forceinline__ void ares_repack_codes(const uint8_t* src, uint8_t* dst, uint32_t n_out, uint32_t n_in) {
    const size_t n_tiles = (n_out + 15) / 16;
    const uint32_t n_steps = n_in / 64;
    const size_t units = n_tiles * n_steps * 32;
    const size_t low_words = units * P::LB, high_words = units * P::HB;
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= low_words + high_words + P::Hdr::words(n_tiles, n_in)) {
        return;
    }
    const AresFileAt<P> at{src, static_cast<size_t>(n_in / P::BLOCK_VALUES) * P::BLOCK_BYTES, n_out, n_tiles};
    uint32_t out = 0;
    if (id < low_words + high_words) {
        const bool high = id >= low_words;
        const int bits = high ? P::HB : P::LB;
        const size_t u = high ? id - low_words : id;
        const int word = static_cast<int>(u % bits);
        const uint32_t lane = (u / bits) % 32, g = lane / 4, t = lane % 4;
        const size_t unit = u / bits / 32;
        const uint32_t step = unit % n_steps, tile = unit / n_steps;
        for (int kk = 0; kk < 4; kk++) {
            for (int i = 0; i < 4; i++) {
                const uint32_t row = tile * 16 + g + 8 * (i & 1);
                if (ares_plane_word(bits, kk, i) != word || row >= n_out) {
                    continue;
                }
                for (int e = 0; e < 2; e++) {
                    const uint32_t k = step * 64 + kk * 16 + 2 * t + 8 * (i >> 1) + e;
                    const uint32_t code = P::code(at.blk(row, k / P::BLOCK_VALUES), k % P::BLOCK_VALUES);
                    const uint32_t field = high ? code >> P::LB : code & ((1u << P::LB) - 1);
                    out |= field << (ares_plane_shift(bits, kk, i) + 16 * e);
                }
            }
        }
    } else {
        out = P::Hdr::word(at, n_in, id - low_words - high_words);
    }
    reinterpret_cast<uint32_t*>(dst)[id] = out;
}

// ---------------------------------------------------------------------------
// What a lane holds of a matrix for the kernels of mmr.cuh. STREAM: the planes are
// read past L1 (for the kernels that read every weight once); DEPTH: how many steps
// ahead they are asked for.
template <class P, bool STREAM, int DEPTH>
struct AresMmrCodes {
    using H = typename P::Hdr;
    static constexpr int LW = P::LB, HW = P::HB;
    static constexpr int HN = HW > 0 ? HW : 1;

    // The lane's words of step 0 of the warp's first tile in both planes, and what
    // `open` needs to find a tile's headers.
    struct Warp {
        const uint32_t* low;
        const uint32_t* high;
        const uint8_t* hdr;
        size_t n_tiles;
        uint32_t first, n_in, lane, n_steps, n_sb;
    };
    struct Tile {
        uint32_t delta;                       // the tile is the warp's first plus this
        uint32_t lw[LW], lx[DEPTH][LW];       // low plane: the step in work, the next steps
        uint32_t hw[HN], hx[DEPTH][HN];       // high plane
        typename H::State h;
        uint32_t S[2], C[2];                  // of the group in work, rows g and g + 8
    };

    static __device__ __forceinline__ void open_warp(Warp& wp, const uint8_t* w, uint32_t n_out, uint32_t n_in, uint32_t first, uint32_t lane) {
        wp.n_tiles = (n_out + 15) / 16;
        wp.n_steps = n_in / 64;
        wp.n_sb = n_in / 256;
        wp.first = first;
        wp.n_in = n_in;
        wp.lane = lane;
        const size_t units = wp.n_tiles * wp.n_steps * 32;
        const size_t at = static_cast<size_t>(first) * wp.n_steps * 32 + lane;
        const uint32_t* base = reinterpret_cast<const uint32_t*>(w);
        wp.low = base + at * LW;
        wp.high = base + units * LW + at * HW;
        wp.hdr = w + units * (LW + HW) * 4;
    }

    static __device__ __forceinline__ void load(Tile& tl, const Warp& wp, uint32_t step, int d) {
        const size_t at = (static_cast<size_t>(tl.delta) * wp.n_steps + step) * 32;
        ares_ld_plane<LW, STREAM>(wp.low + at * LW, tl.lx[d]);
        if constexpr (HW > 0) {
            ares_ld_plane<HW, STREAM>(wp.high + at * HW, tl.hx[d]);
        }
    }

    // The tile `first + delta`, to start at super-block sb.
    static __device__ __forceinline__ void open(Tile& tl, const Warp& wp, uint32_t delta, uint32_t sb) {
        tl.delta = delta;
#pragma unroll
        for (int d = 0; d < DEPTH; d++) {
            if (4 * sb + d < wp.n_steps) {
                load(tl, wp, 4 * sb + d, d);
            }
        }
        H::open(tl.h, wp.hdr, wp.n_tiles, wp.first + delta, wp.n_in, wp.lane, sb);
    }

    static __device__ __forceinline__ void super_block(Tile& tl, const Warp& wp, uint32_t sb) {
        H::super_block(tl.h, sb, wp.n_sb);
    }

    // Takes the codes asked for before and asks for those DEPTH steps on.
    static __device__ __forceinline__ void step(Tile& tl, const Warp& wp, uint32_t step) {
#pragma unroll
        for (int v = 0; v < LW; v++) {
            tl.lw[v] = tl.lx[0][v];
        }
#pragma unroll
        for (int v = 0; v < HW; v++) {
            tl.hw[v] = tl.hx[0][v];
        }
#pragma unroll
        for (int d = 0; d + 1 < DEPTH; d++) {
#pragma unroll
            for (int v = 0; v < LW; v++) {
                tl.lx[d][v] = tl.lx[d + 1][v];
            }
#pragma unroll
            for (int v = 0; v < HW; v++) {
                tl.hx[d][v] = tl.hx[d + 1][v];
            }
        }
        if (step + DEPTH < wp.n_steps) {
            load(tl, wp, step + DEPTH, DEPTH - 1);
        }
        H::step(tl.h, step, wp.n_steps);
    }

    template <int j, int kk>
    static __device__ __forceinline__ void fragment(Tile& tl, uint32_t (&a)[4]) {
        const uint32_t kBias = 0x64006400u;
        if constexpr (kk % (H::GROUP / 16) == 0) {
            H::template scales<j, kk>(tl.h, tl.S, tl.C);
        }
        if constexpr (P::TABLE) {
            // The 16 levels of IQ4_NL + 128, four to a word. `prmt` looks four codes up
            // at once in eight of them (the low three bits of a nibble of the selector
            // pick a byte); a second one takes each byte from the low or the high eight
            // by the fourth bit of its code.
            const uint32_t t0 = 0x3F2D1801u, t1 = 0x766A5D4Fu, t2 = 0xA6998D81u, t3 = 0xF1D9C5B5u;
            const uint32_t w = tl.lw[kk];
            const uint32_t even = __byte_perm(__byte_perm(t0, t1, w), __byte_perm(t2, t3, w), 0x3210u | ((w >> 1) & 0x4444u));
            const uint32_t wo = w >> 16;
            const uint32_t odd = __byte_perm(__byte_perm(t0, t1, wo), __byte_perm(t2, t3, wo), 0x3210u | ((wo >> 1) & 0x4444u));
#pragma unroll
            for (int i = 0; i < 4; i++) {
                a[i] = ares_h2_mul(ares_h2_sub(ares_h2_of_bytes(even, odd, i, i), ares_h2_const(1024 + 128)), tl.S[i & 1]);
            }
            return;
        }
        constexpr uint32_t low_mask = ((1u << P::LB) - 1) * 0x00010001u;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int lsh = ares_plane_shift(P::LB, kk, i);
            uint32_t pair = ((tl.lw[ares_plane_word(P::LB, kk, i)] >> lsh) & low_mask) | kBias;
            if constexpr (P::HB > 0) {
                // The high bits land LB bits up.
                const int hsh = ares_plane_shift(P::HB, kk, i) - P::LB;
                constexpr uint32_t high_mask = (((1u << P::HB) - 1) * 0x00010001u) << P::LB;
                const uint32_t hw = tl.hw[ares_plane_word(P::HB, kk, i)];
                pair |= (hsh >= 0 ? hw >> (hsh >= 0 ? hsh : 0) : hw << (hsh < 0 ? -hsh : 0)) & high_mask;
            }
            a[i] = ares_h2_fma(ares_h2_sub(pair, kBias), tl.S[i & 1], tl.C[i & 1]);
        }
    }
};

// What a lane holds of a matrix for the few-row kernels (`ares_mmr_few`, mmr.cuh): one tile
// per `Warp`, the planes of the step in work and of the next steps in registers (SETS sets,
// the step j of a super-block uses set j % SETS, so that no register is ever moved), the
// header of the super-block in work, and the weights of a slab made with the fewest
// instructions that keep the arithmetic: the code of a value is cut out of its word where it
// lies (a mask, in place whenever the field and the bits above it stay in the 10 bits of an
// f16 mantissa below 1024), put into the mantissa of 1024 and brought back to the exact
// code (minus the offset of the type) by one `HFMA2` that scales by the power of two it
// was left at, then `weight` of the header makes the value.
template <class P, int SETS>
struct AresFewCodes {
    using H = typename P::Hdr;
    static constexpr int LB = P::LB, HB = P::HB, LW = P::LB, HW = P::HB;
    static constexpr int HN = HW > 0 ? HW : 1;
    static_assert(SETS == 2 || SETS == 4, "the set of a step is a constant of the 4 steps of a super-block");

    struct Warp {
        const uint32_t* low;
        const uint32_t* high;
        const uint8_t* hdr;
        size_t n_tiles;
        uint32_t tile, n_in, lane;
        // The warp's part of K ends before this super-block.
        uint32_t sb_end;
    };
    struct Tile {
        uint32_t lx[SETS][LW];
        uint32_t hx[SETS][HN];
        typename H::State h;
        typename H::Scale sc;
    };

    static __device__ __forceinline__ void open_warp(Warp& wp, const uint8_t* w, uint32_t n_out, uint32_t n_in, uint32_t tile, uint32_t lane) {
        wp.n_tiles = (n_out + 15) / 16;
        const uint32_t n_steps = n_in / 64;
        wp.tile = tile;
        wp.n_in = n_in;
        wp.lane = lane;
        wp.sb_end = n_in / 256;
        const size_t units = wp.n_tiles * n_steps * 32;
        const size_t at = static_cast<size_t>(tile) * n_steps * 32 + lane;
        const uint32_t* base = reinterpret_cast<const uint32_t*>(w);
        wp.low = base + at * LW;
        wp.high = base + units * LW + at * HW;
        wp.hdr = w + units * (LW + HW) * 4;
    }

    static __device__ __forceinline__ void load(Tile& tl, const Warp& wp, uint32_t step, int set) {
        ares_ld_plane<LW, true>(wp.low + static_cast<size_t>(step) * 32 * LW, tl.lx[set]);
        if constexpr (HW > 0) {
            ares_ld_plane<HW, true>(wp.high + static_cast<size_t>(step) * 32 * HW, tl.hx[set]);
        }
    }

    // Starts at super-block sb: its first SETS - 1 steps are asked for.
    static __device__ __forceinline__ void open(Tile& tl, const Warp& wp, uint32_t sb) {
#pragma unroll
        for (int d = 0; d < SETS - 1; d++) {
            if (4 * sb + d < 4 * wp.sb_end) {
                load(tl, wp, 4 * sb + d, d);
            }
        }
        H::open(tl.h, wp.hdr, wp.n_tiles, wp.tile, wp.n_in, wp.lane, sb);
    }

    // `__syncwarp` after a load keeps it where it was written: left to itself the compiler
    // moves the loads of a step to where their registers are first needed, and every super-block
    // then waits for its weights.
    static __device__ __forceinline__ void super_block(Tile& tl, const Warp& wp, uint32_t sb) {
        H::super_block(tl.h, sb, wp.sb_end);
        __syncwarp();
    }

    // The start of step j of the super-block: asks for the step SETS - 1 on, into the set
    // the step before used.
    template <int j>
    static __device__ __forceinline__ void begin(Tile& tl, const Warp& wp, uint32_t step) {
        if (step + SETS - 1 < 4 * wp.sb_end) {
            load(tl, wp, step + SETS - 1, (j + SETS - 1) % SETS);
        }
        H::step(tl.h, step, 4 * wp.sb_end);
        __syncwarp();
    }

    // The f16 pair of the codes of register i of slab kk, minus the offset of the type.
    template <int kk, int i>
    static __device__ __forceinline__ uint32_t code_pair(const uint32_t (&lw)[LW], const uint32_t (&hw)[HN]) {
        constexpr int OFFSET = H::OFFSET;
        constexpr int wl = ares_plane_word(LB, kk, i), sh = ares_plane_shift(LB, kk, i);
        constexpr uint32_t bias = 0x64006400u;
        if constexpr (LB == 8) {
            // Whole bytes: `prmt` puts the byte and 0x64 together.
            const uint32_t x = __byte_perm(lw[wl], 0x64646464u, (i % 2 == 0) ? 0x4240u : 0x4341u);
            return ares_h2_sub(x, ares_h2_const(1024 + OFFSET));
        } else {
            constexpr bool in_place = sh + LB + HB <= 10;
            constexpr int at = in_place ? sh : 0;
            constexpr uint32_t low_mask = ((1u << LB) - 1) * 0x00010001u;
            uint32_t x = (in_place ? (lw[wl] & (low_mask << sh)) : ((lw[wl] >> sh) & low_mask)) | bias;
            if constexpr (HB > 0) {
                constexpr int hp = ares_plane_shift(HB, kk, i), hi = ares_plane_word(HB, kk, i);
                constexpr int to = at + LB;
                constexpr int d = hp - to;
                constexpr uint32_t high_mask = (((1u << HB) - 1) * 0x00010001u) << to;
                const uint32_t moved = d >= 0 ? (hw[hi] >> (d >= 0 ? d : 0)) : (hw[hi] << (d < 0 ? -d : 0));
                x |= moved & high_mask;
            }
            if constexpr (at == 0) {
                return ares_h2_sub(x, ares_h2_const(1024 + OFFSET));
            } else {
                // x = 1024 + code 2^at: (x / 2^at) - (1024 / 2^at + OFFSET) is exact.
                return ares_h2_fma(x, ares_h2_pow2_down(at), ares_h2_const(-((1024 >> at) + OFFSET)));
            }
        }
    }

    template <int j, int kk>
    static __device__ __forceinline__ void fragment(Tile& tl, uint32_t (&a)[4]) {
        constexpr int set = j % SETS;
        if constexpr (kk % (H::GROUP / 16) == 0) {
            tl.sc = H::template scale<j, kk>(tl.h);
        }
        if constexpr (P::TABLE) {
            // The 16 levels of IQ4_NL + 128, four to a word. `prmt` looks four codes up at once
            // in eight of them (the low three bits of a nibble of the selector pick a byte); a
            // second one takes each byte from the low or the high eight by the fourth bit of
            // its code.
            const uint32_t t0 = 0x3F2D1801u, t1 = 0x766A5D4Fu, t2 = 0xA6998D81u, t3 = 0xF1D9C5B5u;
            const uint32_t w = tl.lx[set][kk];
            const uint32_t even = __byte_perm(__byte_perm(t0, t1, w), __byte_perm(t2, t3, w), 0x3210u | ((w >> 1) & 0x4444u));
            const uint32_t wo = w >> 16;
            const uint32_t odd = __byte_perm(__byte_perm(t0, t1, wo), __byte_perm(t2, t3, wo), 0x3210u | ((wo >> 1) & 0x4444u));
            // Register i is the pair (even byte i, odd byte i): two `prmt` bring the pairs of
            // registers 0, 1 and 2, 3 together, one more per register puts the 0x64 behind each
            // byte (the f16 pair 1024 + byte).
            constexpr uint32_t bias = ares_h2_const(1024 + H::OFFSET);
            const uint32_t pairs01 = __byte_perm(even, odd, 0x5140), pairs23 = __byte_perm(even, odd, 0x7362);
            a[0] = H::template weight<0>(tl.sc, ares_h2_sub(__byte_perm(pairs01, 0x64646464u, 0x4140), bias));
            a[1] = H::template weight<1>(tl.sc, ares_h2_sub(__byte_perm(pairs01, 0x64646464u, 0x4342), bias));
            a[2] = H::template weight<0>(tl.sc, ares_h2_sub(__byte_perm(pairs23, 0x64646464u, 0x4140), bias));
            a[3] = H::template weight<1>(tl.sc, ares_h2_sub(__byte_perm(pairs23, 0x64646464u, 0x4342), bias));
        } else {
            a[0] = H::template weight<0>(tl.sc, code_pair<kk, 0>(tl.lx[set], tl.hx[set]));
            a[1] = H::template weight<1>(tl.sc, code_pair<kk, 1>(tl.lx[set], tl.hx[set]));
            a[2] = H::template weight<0>(tl.sc, code_pair<kk, 2>(tl.lx[set], tl.hx[set]));
            a[3] = H::template weight<1>(tl.sc, code_pair<kk, 3>(tl.lx[set], tl.hx[set]));
        }
    }
};

// The layout of a type as the kernels of up to 32 rows read it (`Near`) and as the
// grouped blocks of a wide product do (`Wide`), and its rearrangement kernel.
#define ARES_MMR_CODES(TYPE, P)                                                                      \
    using AresMmrNear_##TYPE = AresMmrCodes<P, true, 1>;                                             \
    using AresMmrWide_##TYPE = AresMmrCodes<P, false, 2>;                                            \
    using AresMmrFew_##TYPE = AresFewCodes<P, 2>;                                                    \
    ARES_KERNEL ares_repack_##TYPE(const uint8_t* src, uint8_t* dst, uint32_t n_out, uint32_t n_in) { \
        ares_repack_codes<P>(src, dst, n_out, n_in);                                                 \
    }                                                                                                \
    ARES_MMR_KERNELS(TYPE, AresMmrNear_##TYPE, AresMmrWide_##TYPE)                                   \
    ARES_MMF_KERNELS(TYPE, AresMmrFew_##TYPE)
