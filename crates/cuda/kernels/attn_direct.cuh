// Attention of a few query vectors (one row of a decode step) over quantized K and V,
// without the tensor cores: integer dot products on the codes for the scores, f32 products
// in registers for the output.
//
// With up to 8 vectors the two products are a matrix times a handful of vectors. The tensor
// cores need the stored rows staged (copied to shared memory, decoded there to f16 tiles, read
// back by ldmatrix), and that staging costs more than the products. Here nothing is staged:
//
//   K  lane = cell. Every lane walks the row of its own cell, 8 codes at a time, and takes the
//      dot product of the codes with the queries as 16-bit integers (`dp2a`: two 16-bit values
//      times two code bytes). The queries are rotated and quantized once per block, 15 bits and
//      a scale per vector and group of 32, so one integer sum per group and vector is scaled
//      once: by the query's scale and the scale of the group. A lane ends with the scores of
//      its cell: there is no reduction over the warp.
//   softmax  per lane; the running maximum is the warp's (one max over the lanes per unit).
//      The weights go to shared memory, [cell][vector].
//   V  lane = 8 values of every vector. The warp takes the cells of the unit one after the
//      other (two at a time for a head of 128): every lane reads the weights of the cell and
//      its own 8 codes of the row, a contiguous read for the warp, and adds weight * code *
//      scale to its accumulator in f32. That product is exact up to f32 rounding.
//
// One block is one item, as in attn_kernel.cuh. Its warps are independent: each takes a
// contiguous share of the item's cells and walks it in units of 32 cells. The rows of the next
// unit are prefetched a unit ahead, because a read that misses the caches takes longer than a
// unit's work, and a row of K is read in whole 16-byte chunks: a lane that reads its own row
// costs the load unit one access per lane and load.
//
// The mean of a tile (q7 to q4) enters once per tile: for K a product of the exact queries
// with the mean, added to the scores of the tile's cells; for V the mean times the summed
// weights. At the end the warps' states are merged (log-sum-exp), the accumulator is
// un-rotated and written as the output or as a partial state, as the tensor-core kernel does.
//
// Error against exact arithmetic: the queries carry 15 bits per group (a relative error of
// 1.5e-5 of the group's largest value at most); everything else is f32.
#pragma once

#include "attn_kernel.cuh"

// The 8 values of quarter `q` of group `g` of a row of f16 values (a tile mean), as f32.
static __device__ __forceinline__ void halves8f(const uint8_t* row, int g, int q, float (&out)[8]) {
    const uint4 v = *reinterpret_cast<const uint4*>(row + g * 64 + q * 16);
    const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i] = h2f_bits(w[i] & 0xffffu);
        out[2 * i + 1] = h2f_bits(w[i] >> 16);
    }
}

// Whether the two code words of a quarter hold the even and the odd codes (q7 to q4) or the
// first and the last four (q8).
__host__ __device__ constexpr bool fmt_interleaved(int f) { return f >= 2; }

// The 8 codes of a quarter of a group of a stored qB row, one per byte, in offset binary
// (code + 2^(B-1), so 0 to 2^B - 1), from the bytes that hold them: `a` and `b` are the
// quarter's 8 bytes of codes (q8) or `a` its 4 bytes of plane 0 (bits 0-3 of the codes, two per
// byte), `hi2` its 2 bytes of plane 1 (bits 4-5; q7 and q6) and `top` its byte of top bits (q7
// and q5). For q8, cw[0] holds the codes 0 to 3 and cw[1] the codes 4 to 7; for q7 to q4, cw[0]
// holds the even codes 0, 2, 4, 6 and cw[1] the odd ones: the order their bit planes give
// without further shuffling. A code's place in that order is its slot (`code_slot`).
template <int FMT>
static __device__ __forceinline__ void codes_of(unsigned a, unsigned b, unsigned hi2, unsigned top, uint32_t (&cw)[2]) {
    constexpr int bits = fmt_bits(FMT);
    if constexpr (bits == 8) {
        cw[0] = a ^ 0x80808080u;
        cw[1] = b ^ 0x80808080u;
        return;
    }
    unsigned n0 = a;
    if (bits == 4) n0 ^= 0x88888888u;  // the top bit flipped: offset binary
    unsigned e = n0 & 0x0f0f0f0fu;
    unsigned o = (n0 >> 4) & 0x0f0f0f0fu;
    if (bits >= 6) {
        if (bits == 6) hi2 ^= 0xaaaau;
        const unsigned xe = hi2 & 0x3333u, xo = (hi2 >> 2) & 0x3333u;
        e |= ((xe & 0x0003u) << 4) | ((xe & 0x0030u) << 8) | ((xe & 0x0300u) << 12) | ((xe & 0x3000u) << 16);
        o |= ((xo & 0x0003u) << 4) | ((xo & 0x0030u) << 8) | ((xo & 0x0300u) << 12) | ((xo & 0x3000u) << 16);
    }
    if (bits == 7 || bits == 5) {
        top ^= 0xffu;
        const unsigned se = ((top & 0x55u) * 0x41041u) & 0x01010101u;
        const unsigned so = (((top >> 1) & 0x55u) * 0x41041u) & 0x01010101u;
        const int sh = bits == 7 ? 6 : 4;
        e |= se << sh;
        o |= so << sh;
    }
    cw[0] = e;
    cw[1] = o;
}

// The code words of quarter `q` of group `g` of the row at `row`, each field with its own
// load. The codes of an odd group are 4-byte aligned, those of an even group are 2 bytes off,
// and loads have to be aligned. The address is chosen without a branch, so that lanes with
// different groups share the instructions; a load that is not needed repeats another one.
template <int FMT>
static __device__ __forceinline__ void code_words(const uint8_t* row, int g, int q, uint32_t (&cw)[2]) {
    constexpr int bits = fmt_bits(FMT);
    const uint8_t* codes = row + g * fmt_group_bytes(FMT) + 2;
    const bool odd = (g & 1) != 0;
    const unsigned shift = odd ? 0u : 16u;
    if constexpr (bits == 8) {
        const unsigned* a = reinterpret_cast<const unsigned*>(codes + q * 8 - (odd ? 0 : 2));
        const unsigned a0 = a[0], a1 = a[1], a2 = a[odd ? 1 : 2];
        codes_of<FMT>(__funnelshift_r(a0, a1, shift), __funnelshift_r(a1, a2, shift), 0, 0, cw);
        return;
    }
    const unsigned* a = reinterpret_cast<const unsigned*>(codes + 4 * q - (odd ? 0 : 2));
    const unsigned n0 = __funnelshift_r(a[0], a[odd ? 0 : 1], shift);
    unsigned hi2 = 0, top = 0;
    if (bits >= 6) hi2 = ld_u16(codes + 16 + 2 * q);
    if (bits == 7 || bits == 5) top = bits == 7 ? codes[24 + q] : codes[16 + q];
    codes_of<FMT>(n0, 0, hi2, top, cw);
}

// A group of a row in registers: the aligned chunks of the row that hold it, loaded whole
// (16 bytes each for a head of 256, whose rows are 16-byte aligned; 8 for a head of 128). A
// lane that reads its own row costs the load unit one access per lane and load, so a row is
// read with as few loads as its bytes allow: a chunk that the window `prev` of the previous
// group holds already is taken from there. The result is the group's first byte in the
// window; every offset is a constant once the loop over the groups is unrolled.
template <int FMT, int CH>
static __device__ __forceinline__ int group_window(
    const uint8_t* row, int g, uint32_t (&w)[12], const uint32_t (&prev)[12]) {
    constexpr int gb = fmt_group_bytes(FMT), WPC = CH / 4;
    const int first = gb * g / CH, at = gb * g % CH, n = (at + gb + CH - 1) / CH;
    const int prev_first = g > 0 ? gb * (g - 1) / CH : 0, prev_last = g > 0 ? (gb * g - 1) / CH : -1;
#pragma unroll
    for (int k = 0; k < 12 * 4 / CH; ++k) {
        if (k < n) {
            if (first + k == prev_last) {
#pragma unroll
                for (int j = 0; j < WPC; ++j) w[WPC * k + j] = prev[WPC * (prev_last - prev_first) + j];
            } else if constexpr (CH == 16) {
                const uint4 v = *reinterpret_cast<const uint4*>(row + (first + k) * 16);
                w[4 * k] = v.x, w[4 * k + 1] = v.y, w[4 * k + 2] = v.z, w[4 * k + 3] = v.w;
            } else {
                const uint2 v = *reinterpret_cast<const uint2*>(row + (first + k) * 8);
                w[2 * k] = v.x, w[2 * k + 1] = v.y;
            }
        }
    }
    return at;
}
// 4, 2 or 1 bytes at the (even, for 4 and 2) byte offset `off` of a window.
static __device__ __forceinline__ unsigned window32(const uint32_t (&w)[12], int off) {
    return (off & 2) ? __funnelshift_r(w[off >> 2], w[(off >> 2) + 1], 16) : w[off >> 2];
}
static __device__ __forceinline__ unsigned window16(const uint32_t (&w)[12], int off) {
    return (w[off >> 2] >> ((off & 2) * 8)) & 0xffffu;
}
static __device__ __forceinline__ unsigned window8(const uint32_t (&w)[12], int off) {
    return (w[off >> 2] >> ((off & 3) * 8)) & 0xffu;
}

// The code words of quarter `q` of the group at byte `at` of the window `w`.
template <int FMT>
static __device__ __forceinline__ void window_code_words(const uint32_t (&w)[12], int at, int q, uint32_t (&cw)[2]) {
    constexpr int bits = fmt_bits(FMT);
    const int codes = at + 2;
    if constexpr (bits == 8) {
        codes_of<FMT>(window32(w, codes + 8 * q), window32(w, codes + 8 * q + 4), 0, 0, cw);
        return;
    }
    unsigned hi2 = 0, top = 0;
    if (bits >= 6) hi2 = window16(w, codes + 16 + 2 * q);
    if (bits == 7 || bits == 5) top = window8(w, codes + (bits == 7 ? 24 : 16) + q);
    codes_of<FMT>(window32(w, codes + 4 * q), 0, hi2, top, cw);
}

static __device__ __forceinline__ void code_words_any(int fmt, const uint8_t* row, int g, int q, uint32_t (&cw)[2]) {
    switch (fmt) {
        case FMT_Q8: code_words<FMT_Q8>(row, g, q, cw); break;
        case FMT_Q7: code_words<FMT_Q7>(row, g, q, cw); break;
        case FMT_Q6: code_words<FMT_Q6>(row, g, q, cw); break;
        case FMT_Q5: code_words<FMT_Q5>(row, g, q, cw); break;
        default: code_words<FMT_Q4>(row, g, q, cw); break;
    }
}

// The slot (byte of the two code words) of code `i` of a quarter, and the code of a slot.
static __device__ __forceinline__ int code_slot(bool interleaved, int i) {
    return interleaved ? (i & 1) * 4 + (i >> 1) : i;
}
static __device__ __forceinline__ int slot_code(bool interleaved, int t) {
    return interleaved ? (t & 3) * 2 + (t >> 2) : t;
}

// c + a.lo16 * b.byte0 + a.hi16 * b.byte1 (`lo`) or the same with bytes 2 and 3 of b (`hi`):
// signed 16-bit values times unsigned bytes.
static __device__ __forceinline__ int dp2a_lo(int a, uint32_t b, int c) {
    int d;
    asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
static __device__ __forceinline__ int dp2a_hi(int a, uint32_t b, int c) {
    int d;
    asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Asks for the 32-byte sectors of the `bytes` bytes at `p` ahead of their use.
static __device__ __forceinline__ void prefetch_row(const uint8_t* p, int bytes) {
    for (int off = 0; off < bytes + 28; off += 32) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p + min(off, bytes - 4)));
    }
}

// S = Q K^T for the cell of the lane: the dot products of the row at `kp` with the NV quantized
// queries, per group one integer sum per vector (starting at the offset term), scaled by the
// query's scale and the group's. The window of the next group is loaded before the products
// of this one.
template <int FMT, int D, int NV>
static __device__ __forceinline__ void direct_scores(
    const uint8_t* kp, const short* Qh, const float* qscale, const int* qoff, float (&s)[NV]) {
    constexpr int GROUPS = D / 32, CH = D == 256 ? 16 : 8;
    uint32_t w[12], next[12];
    int at = group_window<FMT, CH>(kp, 0, w, w), next_at = 0;
#pragma unroll
    for (int gi = 0; gi < GROUPS; ++gi) {
        if (gi + 1 < GROUPS) next_at = group_window<FMT, CH>(kp, gi + 1, next, w);
        const float ks = h2f_bits(window16(w, at));
        int acc[8];
        *reinterpret_cast<int4*>(&acc[0]) = *reinterpret_cast<const int4*>(qoff + gi * 8);
        if constexpr (NV > 4) *reinterpret_cast<int4*>(&acc[4]) = *reinterpret_cast<const int4*>(qoff + gi * 8 + 4);
#pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            uint32_t cw[2];
            window_code_words<FMT>(w, at, qi, cw);
            const int4* qw = reinterpret_cast<const int4*>(Qh) + (gi * 4 + qi) * 8;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int qv[8];
                *reinterpret_cast<int4*>(&qv[0]) = qw[2 * k];
                if constexpr (NV > 4) *reinterpret_cast<int4*>(&qv[4]) = qw[2 * k + 1];
#pragma unroll
                for (int v = 0; v < NV; ++v) {
                    acc[v] = (k & 1) ? dp2a_hi(qv[v], cw[k >> 1], acc[v]) : dp2a_lo(qv[v], cw[k >> 1], acc[v]);
                }
            }
        }
        float qs[8];
        *reinterpret_cast<float4*>(&qs[0]) = *reinterpret_cast<const float4*>(qscale + gi * 8);
        if constexpr (NV > 4) *reinterpret_cast<float4*>(&qs[4]) = *reinterpret_cast<const float4*>(qscale + gi * 8 + 4);
#pragma unroll
        for (int v = 0; v < NV; ++v) s[v] = fmaf((float)acc[v] * qs[v], ks, s[v]);
        if (gi + 1 < GROUPS) {
#pragma unroll
            for (int i = 0; i < 12; ++i) w[i] = next[i];
            at = next_at;
        }
    }
}

// Shared memory of a block of the direct kernel, in bytes; src/attention.rs repeats this.
//   block: the exact queries [D][8] f32 (reused for the merged accumulator), the quantized
//          queries [D / 8 quarters][4 words][8] of two i16, their scales [D / 32][8] f32 and the
//          offset sums [D / 32][8] i32, and for the merge the maxima and sums of the warps;
//   warp:  the weights [32][8] f32 of the unit (before the softmax the same memory holds the K
//          mean products by entry of the unit), the mean products of the unit's last entry
//          [8] f32, the V row addresses [32] u64 and the entries [32] u32 of the unit.
// Three blocks of a head of 256 take 56 KB together, which leaves the SM half of its 128 KB
// for the L1 cache; with more than 64 KB of shared memory the cache is 28 KB and the rows
// that were prefetched are gone before they are read (measured: the V loop twice as slow).
#define ATTN_DIRECT_WARP_BYTES (32 * 8 * 4 + 8 * 4 + 32 * 8 + 32 * 4)
#define ATTN_DIRECT_BLOCK_BYTES(D) ((D) * 32 + (D) * 16 + (D) * 2)

// D: head size. NV: vectors the registers hold (the item has up to NV). W: warps of the block.
template <int D, int NV, int W>
static __device__ void attn_direct_body(
    const AttnItem* items, const uint32_t* ranges, const AttnRow* rows, const uint32_t* row_ids,
    const uint32_t* tables, const uint8_t* conv_pool, const uint8_t* far_pool, const float* q,
    const float* gate, float* out, float* st_acc, float* st_lse, uint64_t cfg, float scale_log2) {
    constexpr int GROUPS = D / 32;
    constexpr int CPI = 256 / D;   // cells of a step of the V loop (the lanes of a row hold 8 values each)
    constexpr int LPC = 32 / CPI;  // lanes per cell there
    static_assert(D == 256 || D == 128, "head sizes of 128 and 256");
    static_assert(NV >= 1 && NV <= 8, "up to 8 vectors");
    extern __shared__ __align__(16) uint8_t smem[];

    const AttnItem it = items[blockIdx.x];
    const AttnGeo g = attn_geo(cfg);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kf = g.kf, vf = g.vf;  // every item of this kernel is in converted entries
    const int G = g.G;
    const int nvec = (int)it.n_rows * G;
    const bool k_mean = fmt_has_mean(kf), v_mean = fmt_has_mean(vf);
    const bool k_inter = fmt_interleaved(kf), v_inter = fmt_interleaved(vf);

    float* Qf = reinterpret_cast<float*>(smem);                 // [D][8]
    short* Qh = reinterpret_cast<short*>(Qf + D * 8);           // [D / 8][4][8][2]
    float* qscale = reinterpret_cast<float*>(Qh + D * 8);       // [GROUPS][8]
    int* qoff = reinterpret_cast<int*>(qscale + GROUPS * 8);    // [GROUPS][8]
    uint8_t* wbase = smem + ATTN_DIRECT_BLOCK_BYTES(D) + (size_t)warp * ATTN_DIRECT_WARP_BYTES;
    float* Pt = reinterpret_cast<float*>(wbase);                // [32][8] weights of the unit
    float* mdot = Pt;                                           // [32][8] K mean products, until the softmax
    float* mlast = Pt + 32 * 8;                                 // [8] those of the unit's last entry
    unsigned long long* vptr = reinterpret_cast<unsigned long long*>(mlast + 8);  // [32]
    uint32_t* ents = reinterpret_cast<uint32_t*>(vptr + 32);    // [32]
    float* ml = reinterpret_cast<float*>(smem + ATTN_DIRECT_BLOCK_BYTES(D) + (size_t)W * ATTN_DIRECT_WARP_BYTES);  // [W][16]

    const size_t head_c = (size_t)it.kv_head * g.head_c;

    // ---- the queries: scaled, rotated, and quantized to 15 bits per group and vector
    for (int v = warp; v < 8; v += W) {
        for (int gi = 0; gi < GROUPS; ++gi) {
            float x = 0.0f;
            if (v < nvec) {
                const size_t row = row_ids[it.row0 + v / G];
                const size_t head = (size_t)(g.head0 + it.kv_head) * G + v % G;
                x = rotate32(q[(row * g.kv_total * G + head) * D + gi * 32 + lane], lane) * scale_log2;
            }
            Qf[(gi * 32 + lane) * 8 + v] = x;
            const float amax = warp_max32(fabsf(x));
            const float to_int = amax > 0.0f ? 32767.0f / amax : 0.0f;
            const float qi = fminf(fmaxf(rintf(x * to_int), -32767.0f), 32767.0f);
            const int slot = code_slot(k_inter, lane & 7);
            Qh[((((gi * 4 + (lane >> 3)) * 4 + (slot >> 1)) * 8 + v) << 1) + (slot & 1)] = (short)qi;
            const float sum = ares_warp_sum(qi);  // exact: at most 32 * 32767
            if (lane == 0) {
                qscale[gi * 8 + v] = amax * (1.0f / 32767.0f);
                // the codes are in offset binary: minus the offset times the sum of the queries
                qoff[gi * 8 + v] = -(int)sum * (1 << (fmt_bits(kf) - 1));
            }
        }
    }
    uint32_t lim[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        lim[v] = 0xffffffffu;
        if (v < nvec && (it.flags & ITEM_CAUSAL)) lim[v] = rows[row_ids[it.row0 + v / G]].pos;
    }
    __syncthreads();

    const uint32_t* rg = ranges + it.ranges_off;
    auto map_cell = [&](uint32_t j) -> uint32_t {
        if (it.n_ranges == 1) return rg[0] + j;
        int lo = 0, hi = (int)it.n_ranges - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (rg[3 * mid + 2] <= j) lo = mid; else hi = mid - 1;
        }
        return rg[3 * lo] + (j - rg[3 * lo + 2]);
    };

    // ---- state of the warp: running max (base 2, the same in every lane), the lane's part of
    // the sum, the accumulator of the lane's 8 values (by slot), the weights of the open V tile
    float m[NV], l[NV], o[NV][8], wsum[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        m[v] = -INFINITY;
        l[v] = 0.0f;
        wsum[v] = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) o[v][k] = 0.0f;
    }
    uint32_t k_cached = CELL_NONE;  // the entry whose mean products are in `mlast`
    uint32_t v_open = CELL_NONE;    // the entry whose weights `wsum` collects
    // the lane's values in the V layout: quarter `vq` of group `vg` of cell `vc` of a step
    const int vc = lane / LPC, vg = (lane % LPC) >> 2, vq = lane & 3;
    const float v_offset = 8388608.0f + (float)(1 << (fmt_bits(vf) - 1));
    const int gbv = fmt_group_bytes(vf);

    auto v_mean_flush = [&]() {
        if (v_open == CELL_NONE) return;
        const uint8_t* mp = conv_entry(conv_pool, far_pool, v_open, g.entry_c) + head_c + g.v_mean_off;
        float mean[8];
        halves8f(mp, vg, vq, mean);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const float mv = v_inter ? mean[(t & 3) * 2 + (t >> 2)] : mean[t];
                o[v][t] = fmaf(wsum[v], mv, o[v][t]);
            }
            wsum[v] = 0.0f;
        }
    };

    const uint32_t share = (((it.total + W - 1) / W) + 31) & ~31u;
    const uint32_t j_begin = min((uint32_t)warp * share, it.total), j_end = min(it.total, j_begin + share);
    // The lane's cell of a unit and where its rows are. The rows of the next unit are asked
    // for a unit ahead: a read that misses the caches takes longer than a unit's work.
    uint32_t n_cell = CELL_NONE, n_ent = 0;
    const uint8_t *n_kp = nullptr, *n_vp = nullptr;
    auto unit_at = [&](uint32_t j0) {
        const uint32_t j = j0 + lane;
        n_cell = CELL_NONE;
        if (j < j_end) {
            n_cell = map_cell(j);
#ifdef ATTN_SAME_ROWS
            n_cell = rg[0] + (lane & (ATTN_SAME_ROWS - 1));  // timing only: every unit reads the same rows
#endif
            n_ent = attn_entry(it, tables, n_cell) & ~ATTN_CONVERTED_BIT;
            const uint8_t* head = conv_entry(conv_pool, far_pool, n_ent, g.entry_c) + head_c;
            const uint32_t r = n_cell % ATTN_TILE_CELLS;
            n_kp = head + g.k_rows_off + (size_t)r * g.rbk;
            n_vp = head + g.v_rows_off + (size_t)r * g.rbv;
            prefetch_row(n_kp, g.rbk);
            prefetch_row(n_vp, g.rbv);
        }
    };
    unit_at(j_begin);
#ifdef ATTN_PROBE
    unsigned long long pr[4] = {0, 0, 0, 0}, t0 = clock64(), t1 = 0;
    unsigned units = 0;
#define ATTN_TICK(i) t1 = clock64(); pr[i] += t1 - t0; t0 = t1;
#else
#define ATTN_TICK(i)
#endif
    for (uint32_t j0 = j_begin; j0 < j_end; j0 += 32) {
        const uint32_t cell = n_cell, ent = n_ent;
        const uint8_t* kp = n_kp;
        const bool valid = cell != CELL_NONE;
        const int n_valid = (int)min(32u, j_end - j0);
        // A lane without a cell points at the row of lane 0 and gets no weight.
        const unsigned long long vp0 = (unsigned long long)n_vp;
        const unsigned vp_lo = __shfl_sync(0xffffffffu, (unsigned)vp0, 0);
        const unsigned vp_hi = __shfl_sync(0xffffffffu, (unsigned)(vp0 >> 32), 0);
        vptr[lane] = valid ? vp0 : ((unsigned long long)vp_hi << 32 | vp_lo);
        // (the shuffles are outside the choice: every lane has to take part in them)
        const uint32_t ent0 = __shfl_sync(0xffffffffu, ent, 0);
        ents[lane] = valid ? ent : ent0;
        unit_at(j0 + 32);
        ATTN_TICK(0)

        // ---- the K mean of every entry of the unit against every vector (lanes hold values)
        int eidx = 0;
        if (k_mean) {
            const uint32_t prev = __shfl_up_sync(0xffffffffu, ent, 1);
            const bool first = valid && (lane == 0 || ent != prev);
            const unsigned fb = __ballot_sync(0xffffffffu, first);
            eidx = __popc(fb & ((2u << lane) - 1u)) - 1;
            unsigned rem = fb;
            uint32_t entry = CELL_NONE;
            int e = 0;
            while (rem) {
                const int src = __ffs(rem) - 1;
                rem &= rem - 1;
                entry = __shfl_sync(0xffffffffu, ent, src);
                if (e == 0 && entry == k_cached) {
                    if (lane < 8) mdot[lane] = mlast[lane];
                } else {
                    const uint8_t* mp = conv_entry(conv_pool, far_pool, entry, g.entry_c) + head_c;
                    float mean[8], acc[NV];
#pragma unroll
                    for (int v = 0; v < NV; ++v) acc[v] = 0.0f;
                    if (lane < LPC) {
                        halves8f(mp, vg, vq, mean);
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const float* qd = Qf + (lane * 8 + k) * 8;
#pragma unroll
                            for (int v = 0; v < NV; ++v) acc[v] = fmaf(qd[v], mean[k], acc[v]);
                        }
                    }
#pragma unroll
                    for (int v = 0; v < NV; ++v) {
                        acc[v] = ares_warp_sum(acc[v]);
                        if (lane == 0) mdot[e * 8 + v] = acc[v];
                    }
                }
                __syncwarp();
                ++e;
            }
            k_cached = entry;
            if (lane < 8) mlast[lane] = mdot[(e - 1) * 8 + lane];
            __syncwarp();
        }

        // ---- S = Q K^T, the lane's cell against every vector
        float s[NV];
#pragma unroll
        for (int v = 0; v < NV; ++v) s[v] = 0.0f;
        if (valid) {
            switch (kf) {
                case FMT_Q8: direct_scores<FMT_Q8, D, NV>(kp, Qh, qscale, qoff, s); break;
                case FMT_Q7: direct_scores<FMT_Q7, D, NV>(kp, Qh, qscale, qoff, s); break;
                case FMT_Q6: direct_scores<FMT_Q6, D, NV>(kp, Qh, qscale, qoff, s); break;
                case FMT_Q5: direct_scores<FMT_Q5, D, NV>(kp, Qh, qscale, qoff, s); break;
                default: direct_scores<FMT_Q4, D, NV>(kp, Qh, qscale, qoff, s); break;
            }
        }

        ATTN_TICK(1)
        // ---- online softmax: the scores of the lane's cell, the maximum of the warp
        float alpha[NV], p[8];
        bool settled = true;
#pragma unroll
        for (int v = 0; v < 8; ++v) p[v] = 0.0f;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            float a = s[v];
            if (k_mean) a += mdot[eidx * 8 + v];
            a = (valid && cell <= lim[v]) ? a : -INFINITY;
            const float nm = fmaxf(m[v], warp_max32(a));
            const float u = nm == -INFINITY ? 0.0f : nm;
            alpha[v] = exp2f(m[v] - u);  // exp2(-inf) = 0
            settled = settled && alpha[v] == 1.0f;
            m[v] = nm;
            p[v] = exp2f(a - u);
            l[v] = l[v] * alpha[v] + p[v];
        }
        // The running maximum rarely moves once it has settled: then nothing is rescaled.
        if (!settled) {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                wsum[v] *= alpha[v];
#pragma unroll
                for (int k = 0; k < 8; ++k) o[v][k] *= alpha[v];
            }
        }
        *reinterpret_cast<float4*>(Pt + lane * 8) = make_float4(p[0], p[1], p[2], p[3]);
        if constexpr (NV > 4) *reinterpret_cast<float4*>(Pt + lane * 8 + 4) = make_float4(p[4], p[5], p[6], p[7]);
        __syncwarp();

        ATTN_TICK(2)
        // ---- O += P V: one cell per step (two for a head of 128), the lane's 8 codes of the
        // row, a contiguous read for the warp
        const int steps = (n_valid + CPI - 1) / CPI;
        for (int step = 0; step < steps; ++step) {
            const int i = step * CPI + vc;
            const uint8_t* vp = reinterpret_cast<const uint8_t*>(vptr[i]);
            float w[8];
            *reinterpret_cast<float4*>(&w[0]) = *reinterpret_cast<const float4*>(Pt + i * 8);
            if constexpr (NV > 4) *reinterpret_cast<float4*>(&w[4]) = *reinterpret_cast<const float4*>(Pt + i * 8 + 4);
            if (v_mean) {
                const uint32_t e = ents[i];
                if (e != v_open) {
                    v_mean_flush();
                    v_open = e;
                }
#pragma unroll
                for (int v = 0; v < NV; ++v) wsum[v] += w[v];
            }
            if (vf == FMT_Q8) {
                // The lane's codes of the row three steps on, into L1: the rows of the unit were
                // asked for a unit ago and are in L2 by now. Measured on q8 / q8 at 262144
                // cells: 1531 -> 1395 us on the RTX 4070 (3476 -> 3549 on the RTX 3060, which
                // is bound by its instructions there); for the smaller formats it only costs.
                const uint8_t* ahead = reinterpret_cast<const uint8_t*>(vptr[min(i + 3 * CPI, 31)]);
                asm volatile("prefetch.global.L1 [%0];" ::"l"(ahead + vg * gbv + 2 + vq * 8));
            }
            uint32_t cw[2];
            code_words_any(vf, vp, vg, vq, cw);
            const float scale = h2f_bits(ld_u16(vp + vg * gbv));
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                // A code in offset binary in the low byte of the float 2^23 is the float
                // 2^23 + code + 2^(B-1), exactly; minus that offset it is the code, and the
                // code times the scale is exact too.
                const float f = __uint_as_float(__byte_perm(cw[t >> 2], 0x4b000000u, 0x7440u | (t & 3)));
                const float val = (f - v_offset) * scale;
#pragma unroll
                for (int v = 0; v < NV; ++v) o[v][t] = fmaf(w[v], val, o[v][t]);
            }
        }
        ATTN_TICK(3)
#ifdef ATTN_PROBE
        ++units;
#endif
    }
#ifdef ATTN_PROBE
    if (blockIdx.x == 0 && tid == 0 && units)
        printf("PROBE units %u meta+touch %llu K %llu softmax %llu V %llu\n", units, pr[0] / units, pr[1] / units, pr[2] / units, pr[3] / units);
#endif
    if (v_mean) v_mean_flush();

    // ---- end of the item: merge the warps, normalise, un-rotate, write
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        l[v] = ares_warp_sum(l[v]);
        if (lane == 0) {
            ml[warp * 16 + v] = m[v];
            ml[warp * 16 + 8 + v] = l[v];
        }
    }
    __syncthreads();  // every warp is done with the queries; the maxima and sums are visible
    float* Of = reinterpret_cast<float*>(smem);  // [NV][D]
    float mm[NV], ll[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        float nm = -INFINITY;
        for (int w = 0; w < W; ++w) nm = fmaxf(nm, ml[w * 16 + v]);
        const float u = nm == -INFINITY ? 0.0f : nm;
        float sum = 0.0f;
        for (int w = 0; w < W; ++w) sum += ml[w * 16 + 8 + v] * exp2f(ml[w * 16 + v] - u);
        mm[v] = nm;
        ll[v] = sum;
        const float f = sum > 0.0f ? exp2f(m[v] - u) / sum : 0.0f;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            o[v][t] *= f;
            // a head of 128: the two halves of the warp hold the same values of different cells
            if constexpr (CPI == 2) o[v][t] += __shfl_xor_sync(0xffffffffu, o[v][t], 16);
        }
    }
    for (int w = 0; w < W; ++w) {
        if (warp == w && lane < LPC) {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    float* at = &Of[v * D + lane * 8 + slot_code(v_inter, t)];
                    *at = w == 0 ? o[v][t] : *at + o[v][t];
                }
            }
        }
        __syncthreads();
    }
    const int Hq = g.H * G;
    if (warp == 0 && !(it.flags & ITEM_FINAL)) {
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            if (lane == 0 && v < nvec) {
                const size_t row = row_ids[it.row0 + v / G];
                const size_t slot = rows[row].slot_base + it.slot;
                st_lse[slot * Hq + it.kv_head * G + v % G] = ll[v] > 0.0f ? mm[v] + log2f(ll[v]) : -INFINITY;
            }
        }
    }
    for (int v = warp; v < min(nvec, NV); v += W) {
        const size_t row = row_ids[it.row0 + v / G];
        const int hl = it.kv_head * G + v % G;
        for (int gi = 0; gi < GROUPS; ++gi) {
            const float x = rotate32(Of[v * D + gi * 32 + lane], lane);
            if (it.flags & ITEM_FINAL) {
                const size_t at = (row * g.kv_total * G + (size_t)(g.head0 * G + hl)) * D + gi * 32 + lane;
                out[at] = gate != nullptr ? x * (1.0f / (1.0f + expf(-gate[at]))) : x;
            } else {
                const size_t slot = rows[row].slot_base + it.slot;
                st_acc[(slot * Hq + hl) * D + gi * 32 + lane] = x;
            }
        }
    }
}

#define ARES_ATTN_DIRECT(NAME, D, NV, W, PER_SM)                                                      \
    extern "C" __global__ void __launch_bounds__(32 * (W), PER_SM) NAME(                              \
        const AttnItem* items, const uint32_t* ranges, const AttnRow* rows, const uint32_t* row_ids, \
        const uint32_t* tables, const uint8_t* conv_pool, const uint8_t* far_pool, const float* q,   \
        const float* gate, float* out, float* st_acc, float* st_lse, uint64_t cfg,                   \
        float scale_log2) {                                                                           \
        attn_direct_body<D, NV, W>(                                                                   \
            items, ranges, rows, row_ids, tables, conv_pool, far_pool, q, gate, out, st_acc, st_lse,  \
            cfg, scale_log2);                                                                         \
    }
