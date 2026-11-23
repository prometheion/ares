// The prompt kernel: attention of the rows of a long prompt step (see src/attention.rs).
//
// One block is one ITEM of up to 128 vectors (the query heads of one KV head of up to 128 / G
// rows) over one range of cells, in entries of one kind, that starts at a multiple of 32
// cells. Eight warps, each with 16 vectors as the rows of the products, walk the cells in
// chunks of 32:
//
//   decode    the stored rows of the chunk, already in registers, to f16 tiles in shared
//             memory (K and V), or copy them there when they are f16
//   prefetch  the stored rows of the next chunk into registers, which arrive while the
//             tensor cores work on this one
//   S = Q K^T on the tensor cores with f32 accumulation,
//   the online softmax (base 2, the scores already scaled by scale * log2 e),
//   O += P V on the tensor cores with f16 accumulation over the 32 cells of the chunk, the
//             sum then added to the f32 accumulator.
//
// The queries stay in shared memory for the whole item (rotated where K is, scaled, f16) and
// the accumulator in registers. Every tile in shared memory is swizzled (16-byte piece p of
// row r at p ^ (r & 7)), so the 8 rows of an ldmatrix read hit 8 different banks.
//
// The mean of a tile (q7 to q4) never enters the tiles: the K mean's dot product with each
// vector is added to the scores of the tile's cells, and the weights of the tile's cells,
// summed per vector, times the V mean are added to the accumulator when the tile is done.
#pragma once

#include "attn_decode.cuh"

#define PROMPT_WARPS 8
#define PROMPT_VECS 128
#define PROMPT_CHUNK 32
#define PROMPT_Q_MAX 16383.0f  // the largest query code of the integer scores (15 bits)

// c += a * b with f16 accumulators: a 16x16 (row), b 16x8 (col), c two half2.
static __device__ __forceinline__ void mma_f16_f16(
    uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, "
        "{%0, %1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// c += a * b with f32 accumulators (not volatile: the compiler may schedule it).
static __device__ __forceinline__ void mma_f32(
    float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Halves offset of value `col` of row `row` in a swizzled tile of rows of D halves.
template <int D>
static __device__ __forceinline__ int swz(int row, int col) {
    return row * D + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
}

// Byte offset of byte `col` of row `row` in a swizzled tile of rows of D bytes.
template <int D>
static __device__ __forceinline__ int swzb(int row, int col) {
    return row * D + ((((col >> 4) ^ (row & 7))) << 4) + (col & 15);
}

// c += a * b in integers: a 16x32 signed bytes (row), b 32x8 signed bytes (col).
static __device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The same with a as unsigned bytes.
static __device__ __forceinline__ void mma_u8s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#define PROMPT_RAW_WORDS 9  // q8: 34 bytes, from 2 bytes before its start at most

// Loads the BYTES bytes of a quantized group that starts at `p` (2-byte aligned; every group
// size is 2 more than a multiple of 4) into `w`, from the 4-byte boundary at or before `p`:
// whole words, and the last 2 bytes alone when the group ends in the middle of a word, so
// nothing past the group is read. `prompt_realign` moves the group to byte 0 later, when
// the loads have arrived.
template <int BYTES>
static __device__ __forceinline__ void prompt_load(const uint8_t* p, uint32_t (&w)[PROMPT_RAW_WORDS]) {
    static_assert(BYTES % 4 == 2 && (BYTES + 2) / 4 <= PROMPT_RAW_WORDS, "group size");
    constexpr int F0 = (BYTES - 2) / 4;  // whole words of a group that starts at a word
    const uintptr_t at = reinterpret_cast<uintptr_t>(p);
    const uint32_t* a = reinterpret_cast<const uint32_t*>(at & ~(uintptr_t)3);
#pragma unroll
    for (int i = 0; i < PROMPT_RAW_WORDS; ++i) {
        if (i < F0) w[i] = __ldg(a + i);
        else if (i > F0) w[i] = 0u;
    }
    w[F0] = (at & 2) ? __ldg(a + F0) : (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(a + F0));
}

static __device__ __forceinline__ void prompt_load_any(int fmt, const uint8_t* p, uint32_t (&w)[PROMPT_RAW_WORDS]) {
    switch (fmt) {
        case FMT_Q8: prompt_load<34>(p, w); break;
        case FMT_Q7: prompt_load<30>(p, w); break;
        case FMT_Q6: prompt_load<26>(p, w); break;
        case FMT_Q5: prompt_load<22>(p, w); break;
        default: prompt_load<18>(p, w); break;
    }
}

// Moves a group loaded from an address whose bit 1 is `odd` to byte 0 of `w`.
static __device__ __forceinline__ void prompt_realign(uint32_t (&w)[PROMPT_RAW_WORDS], bool odd) {
    if (odd) {
#pragma unroll
        for (int i = 0; i < PROMPT_RAW_WORDS; ++i)
            w[i] = __funnelshift_r(w[i], i + 1 < PROMPT_RAW_WORDS ? w[i + 1] : 0u, 16);
    }
}

// Four signed codes (one per byte) of the quadruple `i4` (codes 4 i4 .. 4 i4 + 3) of a group
// of `bits`-bit codes whose bytes start at word 0 of `w` (byte 0 and 1 the scale).
template <int BITS>
static __device__ __forceinline__ uint32_t prompt_codes4(const uint32_t (&w)[PROMPT_RAW_WORDS], int i4) {
    if constexpr (BITS == 8) {
        return __funnelshift_r(w[i4], w[i4 + 1], 16);
    } else {
        // plane 0: bits 0-3, two codes per byte, at bytes 2 .. 17
        const int b = 2 + 2 * i4;
        const uint32_t n16 = (w[b >> 2] >> ((b & 3) * 8)) & 0xffffu;
        uint32_t x = __byte_perm(n16, n16 >> 4, 0x5140) & 0x0f0f0f0fu;
        if constexpr (BITS >= 6) {
            // plane 1: bits 4-5, four codes per byte, at bytes 18 .. 25
            const int c = 18 + i4;
            const uint32_t p1 = (w[c >> 2] >> ((c & 3) * 8)) & 0xffu;
            x |= ((p1 & 3u) | ((p1 & 0xcu) << 6) | ((p1 & 0x30u) << 12) | ((p1 & 0xc0u) << 18)) << 4;
        }
        if constexpr (BITS == 5 || BITS == 7) {
            // the top bit, one per code: 32 bits at byte 18 (q5) or 26 (q7)
            constexpr int t = BITS == 5 ? 18 : 26;
            const uint32_t plane = __funnelshift_r(w[t >> 2], w[(t >> 2) + 1], (t & 3) * 8);
            const uint32_t tb = (plane >> (4 * i4)) & 0xfu;
            x |= ((tb * 0x00204081u) & 0x01010101u) << (BITS - 1);
        }
        constexpr uint32_t s = (1u << (BITS - 1)) * 0x01010101u;
        return __vsub4(x ^ s, s);
    }
}

// Four signed codes times the scale as four halves (two half2 words): each an exact
// product rounded once, as `decode_unit` gives them.
static __device__ __forceinline__ uint2 prompt_codes_to_halves(uint32_t codes, __half2 scale) {
    const uint32_t u = codes ^ 0x80808080u;
    const uint32_t lo = __byte_perm(u, 0x64646464u, 0x5140);
    const uint32_t hi = __byte_perm(u, 0x64646464u, 0x5342);
    const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    const __half2 a = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&lo), bias), scale);
    const __half2 b = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&hi), bias), scale);
    return make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}

// Decodes a quantized group (realigned) into 32 halves at row `row`, values 32 g .. 32 g + 31
// of the swizzled tile.
template <int BITS, int D>
static __device__ __forceinline__ void prompt_decode_halves(
    const uint32_t (&w)[PROMPT_RAW_WORDS], __half* tile, int row, int g) {
    const __half2 sc = __half2half2(__ushort_as_half((unsigned short)(w[0] & 0xffffu)));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint2 a = prompt_codes_to_halves(prompt_codes4<BITS>(w, 2 * p), sc);
        const uint2 b = prompt_codes_to_halves(prompt_codes4<BITS>(w, 2 * p + 1), sc);
        *reinterpret_cast<uint4*>(tile + swz<D>(row, 32 * g + 8 * p)) = make_uint4(a.x, a.y, b.x, b.y);
    }
}

// Decodes a quantized group (realigned) into its 32 signed codes at row `row`, bytes 32 g ..
// 32 g + 31 of the swizzled byte tile, and its scale into `scale`.
template <int BITS, int D>
static __device__ __forceinline__ void prompt_decode_codes(
    const uint32_t (&w)[PROMPT_RAW_WORDS], int8_t* tile, float* scale, int row, int g) {
    *scale = h2f_bits(w[0] & 0xffffu);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const uint4 c = make_uint4(prompt_codes4<BITS>(w, 4 * p), prompt_codes4<BITS>(w, 4 * p + 1),
                                   prompt_codes4<BITS>(w, 4 * p + 2), prompt_codes4<BITS>(w, 4 * p + 3));
        *reinterpret_cast<uint4*>(tile + swzb<D>(row, 32 * g + 16 * p)) = c;
    }
}

template <int D>
static __device__ __forceinline__ void prompt_decode_halves_any(
    int fmt, const uint32_t (&w)[PROMPT_RAW_WORDS], __half* tile, int row, int g) {
    switch (fmt) {
        case FMT_Q8: prompt_decode_halves<8, D>(w, tile, row, g); break;
        case FMT_Q7: prompt_decode_halves<7, D>(w, tile, row, g); break;
        case FMT_Q6: prompt_decode_halves<6, D>(w, tile, row, g); break;
        case FMT_Q5: prompt_decode_halves<5, D>(w, tile, row, g); break;
        default: prompt_decode_halves<4, D>(w, tile, row, g); break;
    }
}

template <int D>
static __device__ __forceinline__ void prompt_decode_codes_any(
    int fmt, const uint32_t (&w)[PROMPT_RAW_WORDS], int8_t* tile, float* scale, int row, int g) {
    switch (fmt) {
        case FMT_Q8: prompt_decode_codes<8, D>(w, tile, scale, row, g); break;
        case FMT_Q7: prompt_decode_codes<7, D>(w, tile, scale, row, g); break;
        case FMT_Q6: prompt_decode_codes<6, D>(w, tile, scale, row, g); break;
        case FMT_Q5: prompt_decode_codes<5, D>(w, tile, scale, row, g); break;
        default: prompt_decode_codes<4, D>(w, tile, scale, row, g); break;
    }
}

// I8: the scores in integers (converted entries with K in q8 to q4): the codes of K against
// the query as 15-bit integers with one scale per vector, split into a signed high byte and
// an unsigned low one, two int8 products per group of 32 values, summed exactly in int32 and
// then scaled by the group's scale of K (f32). Otherwise K is an f16 tile and the query f16.
template <int D, bool I8>
static __device__ void attn_prompt_body(
    const AttnItem* items, const uint32_t* ranges, const AttnRow* rows, const uint32_t* row_ids,
    const uint32_t* tables, const uint8_t* exact_pool, const uint8_t* conv_pool, const float* q,
    const float* gate, float* out, float* st_acc, float* st_lse, uint64_t cfg, float scale_log2) {
    constexpr int NT = PROMPT_WARPS * 32;
    constexpr int C = PROMPT_CHUNK;
    constexpr int GROUPS = D / 32;
    // group decodes per side and chunk: C rows of GROUPS groups; a thread takes `PER` of them
    // over both sides (D = 256: one of K and one of V; D = 128: one of K or one of V), so the
    // side is the same for a whole warp
    constexpr int UNITS = 2 * C * GROUPS;
    constexpr int PER = UNITS / NT;
    static_assert(PER >= 1 && UNITS % NT == 0, "units per thread");
    extern __shared__ __align__(16) uint8_t smem[];
    // f16 queries [128][D] or their high and low bytes [128][D] each, then the K tile (f16 or
    // codes) and the V tile of a chunk, the scales, limits and mean dot products
    __half* Qs = reinterpret_cast<__half*>(smem);
    int8_t* Qh = reinterpret_cast<int8_t*>(smem);
    uint8_t* Ql = smem + PROMPT_VECS * D;
    uint8_t* after_q = smem + PROMPT_VECS * D * 2;
    __half* Kt = reinterpret_cast<__half*>(after_q);
    int8_t* Kb = reinterpret_cast<int8_t*>(after_q);
    __half* Vt = reinterpret_cast<__half*>(after_q + C * D * (I8 ? 1 : 2));
    float* ksc = reinterpret_cast<float*>(Vt + C * D);           // [GROUPS][C] (I8)
    float* qsc = ksc + (I8 ? GROUPS * C : 0);                    // [128] (I8)
    uint32_t* vec_lim = reinterpret_cast<uint32_t*>(qsc + (I8 ? PROMPT_VECS : 0));  // [128]
    float* mdot = reinterpret_cast<float*>(vec_lim + PROMPT_VECS);  // [128]

    const AttnItem it = items[blockIdx.x];
    const AttnGeo g = attn_geo(cfg);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tq = lane & 3;
    const bool conv = (it.flags & ITEM_CONVERTED) != 0;
    const int kf = conv ? g.kf : FMT_F16, vf = conv ? g.vf : FMT_F16;
    const int G = g.G;
    const int nvec = (int)it.n_rows * G;
    const bool k_mean = conv && fmt_has_mean(g.kf), v_mean = conv && fmt_has_mean(g.vf);
    const size_t head_c = (size_t)it.kv_head * g.head_c;
    const size_t head_e = (size_t)it.kv_head * g.head_e;
    const uint32_t start = ranges[it.ranges_off], end = start + it.total;

    // ---- vectors: limits and the scaled, possibly rotated, queries
    for (int v = tid; v < PROMPT_VECS; v += NT) {
        uint32_t lim = 0xffffffffu;
        if (v < nvec && (it.flags & ITEM_CAUSAL)) lim = rows[row_ids[it.row0 + v / G]].pos;
        vec_lim[v] = lim;
    }
    for (int v = warp; v < PROMPT_VECS; v += PROMPT_WARPS) {
        float x[GROUPS];
#pragma unroll
        for (int gi = 0; gi < GROUPS; ++gi) {
            x[gi] = 0.0f;
            if (v < nvec) {
                const size_t row = row_ids[it.row0 + v / G];
                const size_t head = (size_t)(g.head0 + it.kv_head) * G + v % G;
                x[gi] = q[(row * g.kv_total * G + head) * D + gi * 32 + lane];
                if (conv && fmt_rotated(g.kf)) x[gi] = rotate32(x[gi], lane);
            }
            x[gi] *= scale_log2;
        }
        if constexpr (I8) {
            float amax = 0.0f;
#pragma unroll
            for (int gi = 0; gi < GROUPS; ++gi) amax = fmaxf(amax, fabsf(x[gi]));
            amax = warp_max32(amax);
            const float inv = amax > 0.0f ? PROMPT_Q_MAX / amax : 0.0f;
#pragma unroll
            for (int gi = 0; gi < GROUPS; ++gi) {
                const int qi = (int)fminf(fmaxf(rintf(x[gi] * inv), -PROMPT_Q_MAX), PROMPT_Q_MAX);
                Qh[swzb<D>(v, gi * 32 + lane)] = (int8_t)(qi >> 7);
                Ql[swzb<D>(v, gi * 32 + lane)] = (uint8_t)(qi & 127);
            }
            if (lane == 0) qsc[v] = amax / PROMPT_Q_MAX;
        } else {
#pragma unroll
            for (int gi = 0; gi < GROUPS; ++gi) Qs[swz<D>(v, gi * 32 + lane)] = __float2half_rn(x[gi]);
        }
    }
    __syncthreads();
    uint32_t lim_min = 0xffffffffu;
    for (int v = lane; v < nvec; v += 32) lim_min = min(lim_min, vec_lim[v]);
#pragma unroll
    for (int h = 16; h > 0; h >>= 1) lim_min = min(lim_min, __shfl_xor_sync(0xffffffffu, lim_min, h));

    // ---- the stored rows of a chunk: which group of which side and row a thread takes
    const int n_chunks = (int)((it.total + C - 1) / C);
    auto entry_of = [&](uint32_t cell) -> uint32_t { return tables[it.table + cell / ATTN_TILE_CELLS]; };
    const int rbk = GROUPS * fmt_group_bytes(kf), rbv = GROUPS * fmt_group_bytes(vf);
    uint32_t raw[PER][PROMPT_RAW_WORDS];
    bool raw_odd[PER];
    // unit u of the chunk: side u / (C * GROUPS), row (u % (C * GROUPS)) / GROUPS, group u % GROUPS
    auto unit_of = [&](int i, int& side, int& row, int& grp) {
        const int u = tid + i * NT;
        side = u / (C * GROUPS);
        row = (u % (C * GROUPS)) / GROUPS;
        grp = u % GROUPS;
    };
    // the address of a group of a converted entry's row
    auto conv_group = [&](uint32_t ent, uint32_t cell, int side, int grp) -> const uint8_t* {
        const uint8_t* head = conv_pool + (size_t)(ent & ~ATTN_CONVERTED_BIT) * g.entry_c + head_c;
        const uint32_t r = cell % ATTN_TILE_CELLS;
        return side == 0 ? head + g.k_rows_off + (size_t)r * rbk + grp * fmt_group_bytes(kf)
                         : head + g.v_rows_off + (size_t)r * rbv + grp * fmt_group_bytes(vf);
    };
    auto prefetch = [&](int c) {
        const uint32_t c0 = start + (uint32_t)c * C;
        const uint32_t ent = entry_of(c0);
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            int side, row, grp;
            unit_of(i, side, row, grp);
            const int f = side == 0 ? kf : vf;
            const uint32_t cell = c0 + row;
            if (f != FMT_F16 && cell < end) {
                const uint8_t* p = conv_group(ent, cell, side, grp);
                raw_odd[i] = (reinterpret_cast<uintptr_t>(p) & 2) != 0;
                prompt_load_any(f, p, raw[i]);
            }
        }
    };

    // ---- state of the warp: running max (base 2), sum, accumulator, tile weights
    float o[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;
    float wA = 0.0f, wB = 0.0f;  // summed weights of the current tile's cells (V mean)
    float mdA = 0.0f, mdB = 0.0f;  // the K mean's dot products of the current tile
    uint32_t cur_ent = 0xffffffffu;
    const int vA = warp * 16 + gq, vB = vA + 8;
    const uint32_t limA = vec_lim[vA], limB = vec_lim[vB];
    const bool warp_active = warp * 16 < nvec;

    // adds the weighted V mean of the tile `ent` to the accumulator
    auto flush_mean = [&](uint32_t ent) {
        float a = wA, b = wB;
        a += __shfl_xor_sync(0xffffffffu, a, 1);
        a += __shfl_xor_sync(0xffffffffu, a, 2);
        b += __shfl_xor_sync(0xffffffffu, b, 1);
        b += __shfl_xor_sync(0xffffffffu, b, 2);
        const __half2* mv = reinterpret_cast<const __half2*>(
            conv_pool + (size_t)(ent & ~ATTN_CONVERTED_BIT) * g.entry_c + head_c + g.v_mean_off);
#pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            const float2 mm = __half22float2(mv[i * 4 + tq]);
            o[i][0] = fmaf(a, mm.x, o[i][0]);
            o[i][1] = fmaf(a, mm.y, o[i][1]);
            o[i][2] = fmaf(b, mm.x, o[i][2]);
            o[i][3] = fmaf(b, mm.y, o[i][3]);
        }
        wA = wB = 0.0f;
    };

    float qsA = 1.0f, qsB = 1.0f;  // the scales of the integer queries (I8)
    if constexpr (I8) {
        qsA = qsc[vA];
        qsB = qsc[vB];
    }
    const bool f16_sides = kf == FMT_F16 || vf == FMT_F16;

    prefetch(0);
    for (int c = 0; c < n_chunks; ++c) {
        const uint32_t c0 = start + (uint32_t)c * C;
        const uint32_t ent = entry_of(c0);
        __syncthreads();  // every warp is done with the tiles of the previous chunk
        // ---- the tiles of this chunk: decoded from registers, or copied when f16
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            int side, row, grp;
            unit_of(i, side, row, grp);
            const int f = side == 0 ? kf : vf;
            const uint32_t cell = c0 + row;
            if (I8 && side == 0) {
                // K as codes: zero past the range, I8 has quantized K only
                if (cell >= end) {
                    *reinterpret_cast<uint4*>(Kb + swzb<D>(row, 32 * grp)) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(Kb + swzb<D>(row, 32 * grp + 16)) = make_uint4(0, 0, 0, 0);
                    ksc[grp * C + row] = 0.0f;
                } else {
                    prompt_realign(raw[i], raw_odd[i]);
                    prompt_decode_codes_any<D>(f, raw[i], Kb, ksc + grp * C + row, row, grp);
                }
                continue;
            }
            __half* tile = side == 0 ? Kt : Vt;
            if (cell >= end) {
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    *reinterpret_cast<uint4*>(tile + swz<D>(row, 32 * grp + 8 * p)) = make_uint4(0, 0, 0, 0);
            } else if (f != FMT_F16) {
                prompt_realign(raw[i], raw_odd[i]);
                prompt_decode_halves_any<D>(f, raw[i], tile, row, grp);
            } else {
                const uint8_t* src;
                const uint32_t r = cell % ATTN_TILE_CELLS;
                if (conv) {
                    src = conv_group(ent, cell, side, grp);
                } else {
                    src = exact_pool + (size_t)ent * g.entry_e + head_e + (size_t)side * ATTN_TILE_CELLS * 2 * D +
                          (size_t)r * 2 * D + grp * 64;
                }
#pragma unroll
                for (int p = 0; p < 4; ++p) cp_async16(tile + swz<D>(row, 32 * grp + 8 * p), src + 16 * p);
            }
        }
        if (f16_sides) cp_async_commit();
        // ---- a new tile: the V mean of the one before, the K mean of this one
        if (ent != cur_ent) {
            if (v_mean && cur_ent != 0xffffffffu) flush_mean(cur_ent);
            if (k_mean) {
                const __half2* mk = reinterpret_cast<const __half2*>(
                    conv_pool + (size_t)(ent & ~ATTN_CONVERTED_BIT) * g.entry_c + head_c);
                // lane l takes values 8 l .. 8 l + 7 (D / 8 lanes)
                float2 mean[4];
                if (lane * 8 < D) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) mean[i] = __half22float2(mk[lane * 4 + i]);
                }
                for (int v = 0; v < 16; ++v) {
                    float acc = 0.0f;
                    if (lane * 8 < D) {
                        float qv[8];
                        if constexpr (I8) {
                            // the integer query of the scores, times its scale
                            const uint2 hb = *reinterpret_cast<const uint2*>(Qh + swzb<D>(warp * 16 + v, lane * 8));
                            const uint2 lb = *reinterpret_cast<const uint2*>(Ql + swzb<D>(warp * 16 + v, lane * 8));
                            const float sc = qsc[warp * 16 + v];
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const uint32_t hw = i < 4 ? hb.x : hb.y, lw = i < 4 ? lb.x : lb.y;
                                const int hi = (int)(int8_t)((hw >> (8 * (i & 3))) & 0xffu);
                                const int lo = (int)((lw >> (8 * (i & 3))) & 0xffu);
                                qv[i] = (float)(hi * 128 + lo) * sc;
                            }
                        } else {
                            const uint4 hv = *reinterpret_cast<const uint4*>(Qs + swz<D>(warp * 16 + v, lane * 8));
                            const __half2* qh = reinterpret_cast<const __half2*>(&hv);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const float2 b = __half22float2(qh[i]);
                                qv[2 * i] = b.x;
                                qv[2 * i + 1] = b.y;
                            }
                        }
#pragma unroll
                        for (int i = 0; i < 4; ++i) acc = fmaf(mean[i].x, qv[2 * i], fmaf(mean[i].y, qv[2 * i + 1], acc));
                    }
                    acc = ares_warp_sum(acc);
                    if (lane == 0) mdot[warp * 16 + v] = acc;
                }
                __syncwarp();
                mdA = mdot[vA];
                mdB = mdot[vB];
            }
            cur_ent = ent;
        }
        if (c + 1 < n_chunks) prefetch(c + 1);
        if (f16_sides) cp_async_wait<0>();
        __syncthreads();  // the tiles are complete

        if (!warp_active) continue;
        // ---- S = Q K^T: 16 vectors x 32 cells, four tiles of 8 cells
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
        if constexpr (I8) {
#pragma unroll
            for (int gi = 0; gi < GROUPS; ++gi) {
                uint32_t ah[4], al[4], b[4], b2[4];
                const int qrow = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, qcol = gi * 32 + (lane >> 4) * 16;
                ldmatrix_x4(ah, Qh + swzb<D>(qrow, qcol));
                ldmatrix_x4(al, Ql + swzb<D>(qrow, qcol));
                const int krow = ((lane >> 4) & 1) * 8 + (lane & 7), kcol = gi * 32 + ((lane >> 3) & 1) * 16;
                ldmatrix_x4(b, Kb + swzb<D>(krow, kcol));
                ldmatrix_x4(b2, Kb + swzb<D>(16 + krow, kcol));
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const uint32_t b0 = nt < 2 ? b[2 * nt] : b2[2 * nt - 4];
                    const uint32_t b1 = nt < 2 ? b[2 * nt + 1] : b2[2 * nt - 3];
                    int acc[4] = {0, 0, 0, 0};
                    mma_s8(acc, ah, b0, b1);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[e] <<= 7;
                    mma_u8s8(acc, al, b0, b1);
                    const float2 ks = *reinterpret_cast<const float2*>(ksc + gi * C + nt * 8 + 2 * tq);
                    s[nt][0] = fmaf((float)acc[0], ks.x, s[nt][0]);
                    s[nt][1] = fmaf((float)acc[1], ks.y, s[nt][1]);
                    s[nt][2] = fmaf((float)acc[2], ks.x, s[nt][2]);
                    s[nt][3] = fmaf((float)acc[3], ks.y, s[nt][3]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                s[nt][0] *= qsA;
                s[nt][1] *= qsA;
                s[nt][2] *= qsB;
                s[nt][3] *= qsB;
            }
        } else {
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
                uint32_t a[4], b[4], b2[4];
                ldmatrix_x4(a, Qs + swz<D>(warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, ks * 16 + (lane >> 4) * 8));
                ldmatrix_x4(b, Kt + swz<D>(((lane >> 4) & 1) * 8 + (lane & 7), ks * 16 + ((lane >> 3) & 1) * 8));
                ldmatrix_x4(b2, Kt + swz<D>(16 + ((lane >> 4) & 1) * 8 + (lane & 7), ks * 16 + ((lane >> 3) & 1) * 8));
                mma_f32(s[0], a, b[0], b[1]);
                mma_f32(s[1], a, b[2], b[3]);
                mma_f32(s[2], a, b2[0], b2[1]);
                mma_f32(s[3], a, b2[2], b2[3]);
            }
        }
        // ---- scores: the mean term, the mask where some cell may be past a row or the range
        const bool mask = c0 + C - 1 > lim_min || c0 + C > end;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float a = s[nt][j] + mdA, b = s[nt][2 + j] + mdB;
                if (mask) {
                    const uint32_t cell = c0 + nt * 8 + 2 * tq + j;
                    a = (cell < end && cell <= limA) ? a : -INFINITY;
                    b = (cell < end && cell <= limB) ? b : -INFINITY;
                }
                s[nt][j] = a;
                s[nt][2 + j] = b;
            }
        }
        // ---- online softmax over the 32 cells, rows A and B of this thread
        float cA = fmaxf(fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1])),
                         fmaxf(fmaxf(s[2][0], s[2][1]), fmaxf(s[3][0], s[3][1])));
        float cB = fmaxf(fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3])),
                         fmaxf(fmaxf(s[2][2], s[2][3]), fmaxf(s[3][2], s[3][3])));
        cA = fmaxf(cA, __shfl_xor_sync(0xffffffffu, cA, 1));
        cA = fmaxf(cA, __shfl_xor_sync(0xffffffffu, cA, 2));
        cB = fmaxf(cB, __shfl_xor_sync(0xffffffffu, cB, 1));
        cB = fmaxf(cB, __shfl_xor_sync(0xffffffffu, cB, 2));
        const float nA = fmaxf(mA, cA), nB = fmaxf(mB, cB);
        const float uA = nA == -INFINITY ? 0.0f : nA, uB = nB == -INFINITY ? 0.0f : nB;
        const float alphaA = exp2f(mA - uA), alphaB = exp2f(mB - uB);  // exp2(-inf) = 0
        mA = nA;
        mB = nB;
        float sumA = 0.0f, sumB = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - uA);
            s[nt][1] = exp2f(s[nt][1] - uA);
            s[nt][2] = exp2f(s[nt][2] - uB);
            s[nt][3] = exp2f(s[nt][3] - uB);
            sumA += s[nt][0] + s[nt][1];
            sumB += s[nt][2] + s[nt][3];
        }
        lA = lA * alphaA + sumA;
        lB = lB * alphaB + sumB;
        wA = wA * alphaA + sumA;
        wB = wB * alphaB + sumB;
        // The running maximum rarely moves once it has settled: then nothing is rescaled.
        if (!__all_sync(0xffffffffu, alphaA == 1.0f && alphaB == 1.0f)) {
#pragma unroll
            for (int i = 0; i < D / 8; ++i) {
                o[i][0] *= alphaA;
                o[i][1] *= alphaA;
                o[i][2] *= alphaB;
                o[i][3] *= alphaB;
            }
        }
        // the weights as the A operand: cells 0-15 and 16-31
        uint32_t pa[2][4];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            pa[k][0] = pack_half2(s[2 * k][0], s[2 * k][1]);
            pa[k][1] = pack_half2(s[2 * k][2], s[2 * k][3]);
            pa[k][2] = pack_half2(s[2 * k + 1][0], s[2 * k + 1][1]);
            pa[k][3] = pack_half2(s[2 * k + 1][2], s[2 * k + 1][3]);
        }
        // ---- O += P V: per 16 values, the 32 cells summed in f16, then added in f32
#pragma unroll
        for (int dt = 0; dt < D / 16; ++dt) {
            uint32_t b[4], b2[4];
            ldmatrix_x4_trans(b, Vt + swz<D>(((lane >> 3) & 1) * 8 + (lane & 7), dt * 16 + (lane >> 4) * 8));
            ldmatrix_x4_trans(b2, Vt + swz<D>(16 + ((lane >> 3) & 1) * 8 + (lane & 7), dt * 16 + (lane >> 4) * 8));
            uint32_t t0[2] = {0u, 0u}, t1[2] = {0u, 0u};
            mma_f16_f16(t0, pa[0], b[0], b[1]);
            mma_f16_f16(t1, pa[0], b[2], b[3]);
            mma_f16_f16(t0, pa[1], b2[0], b2[1]);
            mma_f16_f16(t1, pa[1], b2[2], b2[3]);
            const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&t0[0]));
            const float2 y0 = __half22float2(*reinterpret_cast<const __half2*>(&t0[1]));
            const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&t1[0]));
            const float2 y1 = __half22float2(*reinterpret_cast<const __half2*>(&t1[1]));
            o[2 * dt][0] += x0.x;
            o[2 * dt][1] += x0.y;
            o[2 * dt][2] += y0.x;
            o[2 * dt][3] += y0.y;
            o[2 * dt + 1][0] += x1.x;
            o[2 * dt + 1][1] += x1.y;
            o[2 * dt + 1][2] += y1.x;
            o[2 * dt + 1][3] += y1.y;
        }
    }
    if (v_mean && cur_ent != 0xffffffffu) flush_mean(cur_ent);
    if (!warp_active) return;

    // ---- the end of the item: normalize, un-rotate, write
    lA += __shfl_xor_sync(0xffffffffu, lA, 1);
    lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1);
    lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    const float invA = lA > 0.0f ? 1.0f / lA : 0.0f, invB = lB > 0.0f ? 1.0f / lB : 0.0f;
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
        o[i][0] *= invA;
        o[i][1] *= invA;
        o[i][2] *= invB;
        o[i][3] *= invB;
    }
    if (conv && fmt_rotated(g.vf)) {
        // rotate32 on each group of 32 values (four tiles of 8): value i of a group is
        // tile 4 gi + (i >> 3), thread tq = (i >> 1) & 3, element i & 1; the same stages and
        // operations, lower element sum, upper difference
#pragma unroll
        for (int gi = 0; gi < D / 32; ++gi) {
#pragma unroll
            for (int r = 0; r < 4; r += 2) {
                // h = 1: within the thread
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float x = o[4 * gi + j][r], y = o[4 * gi + j][r + 1];
                    o[4 * gi + j][r] = __fadd_rn(x, y);
                    o[4 * gi + j][r + 1] = __fsub_rn(x, y);
                }
                // h = 2, 4: across the threads of the quad
#pragma unroll
                for (int h = 1; h <= 2; h <<= 1) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const float v = o[4 * gi + j][r + e];
                            const float ot = __shfl_xor_sync(0xffffffffu, v, h);
                            o[4 * gi + j][r + e] = (tq & h) ? __fsub_rn(ot, v) : __fadd_rn(v, ot);
                        }
                    }
                }
                // h = 8, 16: across the tiles of the group
#pragma unroll
                for (int h = 1; h <= 2; h <<= 1) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (j & h) continue;
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const float lo = o[4 * gi + j][r + e], hi = o[4 * gi + j + h][r + e];
                            o[4 * gi + j][r + e] = __fadd_rn(lo, hi);
                            o[4 * gi + j + h][r + e] = __fsub_rn(lo, hi);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    o[4 * gi + j][r] = __fmul_rn(o[4 * gi + j][r], __uint_as_float(0x3e3504f3u));
                    o[4 * gi + j][r + 1] = __fmul_rn(o[4 * gi + j][r + 1], __uint_as_float(0x3e3504f3u));
                }
            }
        }
    }
    const int Hq = g.H * G;
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
        const int v = h2 ? vB : vA;
        if (v >= nvec) continue;
        const size_t row = row_ids[it.row0 + v / G];
        const int hl = it.kv_head * G + v % G;
        if (it.flags & ITEM_FINAL) {
            const size_t base = (row * g.kv_total * G + (size_t)(g.head0 * G + hl)) * D;
#pragma unroll
            for (int i = 0; i < D / 8; ++i) {
                const size_t at = base + i * 8 + 2 * tq;
                float x = o[i][2 * h2], y = o[i][2 * h2 + 1];
                if (gate != nullptr) {
                    const float2 gt = *reinterpret_cast<const float2*>(gate + at);
                    x *= 1.0f / (1.0f + expf(-gt.x));
                    y *= 1.0f / (1.0f + expf(-gt.y));
                }
                *reinterpret_cast<float2*>(out + at) = make_float2(x, y);
            }
        } else {
            const size_t slot = rows[row].slot_base + it.slot;
            const size_t base = (slot * Hq + hl) * D;
#pragma unroll
            for (int i = 0; i < D / 8; ++i)
                *reinterpret_cast<float2*>(st_acc + base + i * 8 + 2 * tq) = make_float2(o[i][2 * h2], o[i][2 * h2 + 1]);
            if (tq == 0) {
                const float mm = h2 ? mB : mA, ll = h2 ? lB : lA;
                st_lse[slot * Hq + hl] = ll > 0.0f ? mm + log2f(ll) : -INFINITY;
            }
        }
    }
}

#define ARES_ATTN_PROMPT(NAME, D, I8)                                                                 \
    extern "C" __global__ void __launch_bounds__(32 * PROMPT_WARPS, 1) NAME(                          \
        const AttnItem* items, const uint32_t* ranges, const AttnRow* rows, const uint32_t* row_ids, \
        const uint32_t* tables, const uint8_t* exact_pool, const uint8_t* conv_pool, const float* q, \
        const float* gate, float* out, float* st_acc, float* st_lse, uint64_t cfg, float scale_log2, \
        uint32_t rawcap) {                                                                            \
        (void)rawcap;                                                                                 \
        attn_prompt_body<D, I8>(                                                                      \
            items, ranges, rows, row_ids, tables, exact_pool, conv_pool, q, gate, out, st_acc,        \
            st_lse, cfg, scale_log2);                                                                 \
    }
