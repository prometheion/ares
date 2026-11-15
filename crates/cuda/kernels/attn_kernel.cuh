// Attention over the stored sequence, natively on the formats (see
// src/attention.rs for the design).
//
// One block is one ITEM: some rows of one sequence (their query heads of one KV
// head are the "vectors" of the product, up to 16 * MT of them), and a list of
// ranges of cells of the sequence, all in entries of one kind (exact or
// converted). The block walks the cells in chunks, a chunk being 16 or 32 cells:
//
//   cp.async  the raw stored rows of the next chunk into shared memory
//   decode    the raw rows of this chunk to f16 tiles (K and V), one switch on the
//             format per unit; the mean of a tile never enters the tile, it is
//             added to the scores (K) and to the accumulator (V) once per tile
//   S = Q K^T on the tensor cores (the vectors are the rows of the product),
//   the online softmax (base 2, scores already scaled by scale * log2 e),
//   O += P V.
//
// The warps are MT x CG: MT warps share the vectors (tiles of 16, or of NVW * 8 in the swapped
// orientation of decode steps), CG warps split the cells of a chunk, and XW helper warps only
// load (cp.async, from the moment the decode of the chunk is done) and decode. At the end the CG
// partial states of a tile are merged in shared memory, the accumulator is un-rotated (V in a qB
// format) and the result is written either as the final output (with the sigmoid gate as
// epilogue) or as a partial state for the merge kernel.
#pragma once

#include "attn_decode.cuh"

#define ITEM_CONVERTED 1u
#define ITEM_CAUSAL 2u
#define ITEM_FINAL 4u
#define CELL_NONE 0xffffffffu

struct AttnItem {
    uint32_t table;       // first entry of the sequence's table
    uint32_t row0;        // index in row_ids of the item's first row
    uint32_t n_rows;      // rows of the item; n_rows * G <= 16 * MT
    uint32_t kv_head;     // KV head among those in the pools
    uint32_t ranges_off;  // index in `ranges` of the item's (start, end, before) triples
    uint32_t n_ranges;
    uint32_t total;       // cells in the ranges
    uint32_t flags;       // ITEM_*
    uint32_t slot;        // piece number among the row's pieces
    uint32_t swap_tile;   // one more than the tile whose entry is swap_entry, not the table's; 0: none
    uint32_t swap_entry;
    uint32_t pad;
};

// The entry of the tile of `cell` for the item: the table's, or the one the item brings for
// one tile (a tile that the rows of this item read converted while the table still names its
// exact entry for the rows before them).
static __device__ __forceinline__ uint32_t attn_entry(const AttnItem& it, const uint32_t* tables, uint32_t cell) {
    const uint32_t tile = cell / ATTN_TILE_CELLS;
    return tile + 1 == it.swap_tile ? it.swap_entry : tables[it.table + tile];
}

// Distinct entries a chunk may touch (each needs its own mean term), by the number of
// vectors of the launch: one row against many entries (block-sparse) has few vectors.
template <int QROWS>
struct AttnShape {
    static constexpr int ME = (QROWS <= 16) ? 32 : 4;
};

// Shared memory of a launch, in bytes; src/attention.rs repeats this formula.
// NVW > 0 is the product the other way round for few vectors: the cells are the rows of the
// 16-row operand (K, V^T), the vectors the columns, in tiles of 8, NVW per warp. MT is then the
// number of warps that share the vectors. With NVW = 0 the vectors are the rows, MT tiles of 16.
template <int D, int MT, int CG, int CHUNK, int XW, int NVW>
static __device__ void attn_body(
    const AttnItem* items, const uint32_t* ranges, const AttnRow* rows, const uint32_t* row_ids,
    const uint32_t* tables, const uint8_t* exact_pool, const uint8_t* conv_pool, const float* q,
    const float* gate, float* out, float* st_acc, float* st_lse, uint64_t cfg, float scale_log2,
    uint32_t rawcap) {
    constexpr int W = MT * CG;       // warps that run the mma
    constexpr int WT = W + XW;       // all warps: the others only load and decode
    constexpr int NT = WT * 32;
    constexpr int QS = D + 8;
    constexpr bool SWAP = NVW > 0;
    constexpr int QROWS = SWAP ? (MT * NVW * 8 < 16 ? 16 : MT * NVW * 8) : 16 * MT;
    constexpr int ME = AttnShape<QROWS>::ME;
    constexpr int NV1 = SWAP ? NVW : 1;
    constexpr int PSTR = 24;  // halves per row of the weights staged for the second product
    static_assert(CG == 1 || CG == 2, "one or two cell groups");
    static_assert(CHUNK == 16 || CHUNK == 32, "chunks of 16 or 32 cells");
    extern __shared__ __align__(16) uint8_t smem[];

    const AttnItem it = items[blockIdx.x];
    const AttnGeo g = attn_geo(cfg);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const bool mma_warp = warp < W;
    const int mt = warp % MT, cg = warp / MT;
    const int gq = lane >> 2, tq = lane & 3;
    const bool conv = (it.flags & ITEM_CONVERTED) != 0;
#ifdef ATTN_FIX_Q8
    constexpr int kf = FMT_Q8, vf = FMT_Q8;
#else
    const int kf = conv ? g.kf : FMT_F16, vf = conv ? g.vf : FMT_F16;
#endif
    const int rbk = D / 32 * fmt_group_bytes(kf), rbv = D / 32 * fmt_group_bytes(vf);
    const int G = g.G;
    const int nvec = (int)it.n_rows * G;
    const bool k_mean = conv && fmt_has_mean(g.kf), v_mean = conv && fmt_has_mean(g.vf);
    int nc = (int)(rawcap / (uint32_t)max(rbk, rbv)) & ~15;
    nc = min(nc, CHUNK);

    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Kt = Qs + QROWS * QS;
    __half* Vt = Kt + CHUNK * QS;
    uint8_t* raw = reinterpret_cast<uint8_t*>(Vt + CHUNK * QS);
    uint32_t* meta = reinterpret_cast<uint32_t*>(raw + 4 * (size_t)rawcap);  // [3][CHUNK][3]
    uint32_t* elist = meta + 3 * CHUNK * 3;                                  // [3][ME + 1]
    uint32_t* vec_lim = elist + 3 * (ME + 1);                                // [QROWS]
    float* mdot = reinterpret_cast<float*>(vec_lim + QROWS);                 // [QROWS][ME]
    __half* Pst = reinterpret_cast<__half*>((reinterpret_cast<uintptr_t>(mdot + QROWS * ME) + 15) & ~(uintptr_t)15);

    const size_t head_c = (size_t)it.kv_head * g.head_c;
    const size_t head_e = (size_t)it.kv_head * g.head_e;

    // ---- vectors: limits and the scaled, possibly rotated, f16 queries
    for (int v = tid; v < QROWS; v += NT) {
        uint32_t lim = 0;
        if (v < nvec) {
            const AttnRow r = rows[row_ids[it.row0 + v / G]];
            lim = (it.flags & ITEM_CAUSAL) ? r.pos : 0xffffffffu;
        }
        vec_lim[v] = lim;
    }
    for (int v = warp; v < QROWS; v += WT) {
        for (int gi = 0; gi < D / 32; ++gi) {
            float x = 0.0f;
            if (v < nvec) {
                const size_t row = row_ids[it.row0 + v / G];
                const size_t head = (size_t)(g.head0 + it.kv_head) * G + v % G;
                x = q[(row * g.kv_total * G + head) * D + gi * 32 + lane];
                if (conv && fmt_rotated(g.kf)) x = rotate32(x, lane);
            }
            Qs[v * QS + gi * 32 + lane] = __float2half_rn(x * scale_log2);
        }
    }

    // ---- cell metadata: for each chunk the cell and entry of each slot
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
    const int n_chunks = (int)((it.total + nc - 1) / nc);
    // warp 0 loads, then stores, the metadata of one chunk
    uint32_t pend_cell = CELL_NONE, pend_ent = 0;
    auto meta_load = [&](int c) {
        pend_cell = CELL_NONE;
        pend_ent = 0;
        const uint32_t j = (uint32_t)c * nc + lane;
        if (c < n_chunks && lane < nc && j < it.total) {
            pend_cell = map_cell(j);
            pend_ent = attn_entry(it, tables, pend_cell);
        }
    };
    auto meta_store = [&](int c) {  // warp 0 only
        uint32_t* m = meta + (c % 3) * CHUNK * 3;
        const unsigned valid = __ballot_sync(0xffffffffu, pend_cell != CELL_NONE);
        const int last = valid ? 31 - __clz(valid) : 0;
        uint32_t ent = __shfl_sync(0xffffffffu, pend_ent, last);
        if (pend_cell != CELL_NONE) ent = pend_ent;
        const uint32_t prev = __shfl_up_sync(0xffffffffu, ent, 1);
        const bool first = (lane == 0) || (ent != prev);
        const unsigned fb = __ballot_sync(0xffffffffu, first && lane < nc);
        const int eidx = __popc(fb & ((2u << lane) - 1u)) - 1;
        if (lane < CHUNK) {
            m[lane * 3] = pend_cell;
            m[lane * 3 + 1] = ent & ~ATTN_CONVERTED_BIT;
            m[lane * 3 + 2] = (uint32_t)min(eidx, ME - 1);
        }
        uint32_t* el = elist + (c % 3) * (ME + 1);
        if (first && lane < nc && eidx < ME) el[eidx] = ent & ~ATTN_CONVERTED_BIT;
        if (lane == 0) el[ME] = (uint32_t)min(__popc(fb), ME);
    };
    if (warp == 0) {
        meta_load(0);
        meta_store(0);
        meta_load(1);
        meta_store(1);
    }
    __syncthreads();

    auto row_src = [&](uint32_t slot, uint32_t cell, int side) -> const uint8_t* {
        const uint32_t r = cell % ATTN_TILE_CELLS;
        if (conv) {
            const uint8_t* head = conv_pool + (size_t)slot * g.entry_c + head_c;
            return side == 0 ? head + g.k_rows_off + (size_t)r * g.rbk
                             : head + g.v_rows_off + (size_t)r * g.rbv;
        }
        return exact_pool + (size_t)slot * g.entry_e + head_e + (size_t)side * ATTN_TILE_CELLS * 2 * D +
               (size_t)r * 2 * D;
    };
    // The loads are issued by the helper warps: issuing blocks while the memory system is
    // busy, which is the time the mma warps want for the previous chunk. `by_all` is the
    // prologue, before there is a previous chunk.
    auto issue = [&](int c, bool by_all) {
        const uint32_t* m = meta + (c % 3) * CHUNK * 3;
        const int iw = by_all ? warp : warp - W;
        const int iwn = by_all ? WT : XW;
        for (int side = 0; side < 2; ++side) {
            const int rb = side == 0 ? rbk : rbv;
            const int piece = (rb % 16 == 0) ? 16 : 8;
            const int ppr = rb / piece;
            uint8_t* base = raw + (size_t)((c & 1) * 2 + side) * rawcap;
            for (int slot = iw; slot < nc; slot += iwn) {
                const uint32_t cell = m[slot * 3];
                if (cell == CELL_NONE) continue;
                if (lane < ppr) {  // ppr <= 32: one piece per lane
                    const uint8_t* src = row_src(m[slot * 3 + 1], cell, side);
                    uint8_t* dst = base + (size_t)slot * rb + lane * piece;
                    if (piece == 16) cp_async16(dst, src + lane * 16); else cp_async8(dst, src + lane * 8);
                }
            }
        }
        cp_async_commit();
    };

    // ---- state of the warp: running max (base 2), sum, accumulator
    float o[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;
    // swapped orientation: the same per vector column, accumulator [d tile of 16][vector tile of 8]
    float o_sw[SWAP ? D / 16 : 1][NV1][4];
    float mc[NV1][2], lc[NV1][2];
    if constexpr (SWAP) {
#pragma unroll
        for (int dt = 0; dt < D / 16; ++dt)
#pragma unroll
            for (int j = 0; j < NV1; ++j) o_sw[dt][j][0] = o_sw[dt][j][1] = o_sw[dt][j][2] = o_sw[dt][j][3] = 0.0f;
#pragma unroll
        for (int j = 0; j < NV1; ++j) mc[j][0] = mc[j][1] = -INFINITY, lc[j][0] = lc[j][1] = 0.0f;
    }
    uint32_t cached_ent = 0xffffffffu;

#ifdef ATTN_PROBE
    unsigned long long pr[6] = {0, 0, 0, 0, 0, 0}, t0 = clock64(), t1 = 0;
#endif
    issue(0, true);
    for (int c = 0; c < n_chunks; ++c) {
#ifdef ATTN_PROBE
        t0 = clock64();
#endif
        cp_async_wait<0>();
        __syncthreads();  // chunk c's raw rows are in shared memory; the metadata of c + 1 is visible
#ifdef ATTN_PROBE
        t1 = clock64(); pr[0] += t1 - t0; t0 = t1;
#endif
        if (warp == 0) meta_load(c + 2);

        const uint32_t* m = meta + (c % 3) * CHUNK * 3;
        const uint32_t* el = elist + (c % 3) * (ME + 1);
        const int nent = (int)el[ME];

        // decode the chunk to f16 tiles
#ifndef ATTN_NO_DECODE
        decode_side_any<D, NT, CHUNK>(kf, raw + (size_t)((c & 1) * 2) * rawcap, rbk, m, nc, Kt, QS, tid);
        decode_side_any<D, NT, CHUNK>(vf, raw + (size_t)((c & 1) * 2 + 1) * rawcap, rbv, m, nc, Vt, QS, tid);
#endif
#ifdef ATTN_PROBE
        t1 = clock64(); pr[1] += t1 - t0; t0 = t1;
#endif
        // the mean of the K tile(s) against every vector
        if (k_mean) {
            const bool same = (nent == 1 && el[0] == cached_ent);
            if (!same) {
                for (int p = warp; p < nvec * nent; p += WT) {
                    const int v = p / nent, e = p - v * nent;
                    float acc = 0.0f;
                    if (lane * 8 < D) {
                        const uint8_t* mp = conv_pool + (size_t)el[e] * g.entry_c + head_c;
                        const uint4 mv = *reinterpret_cast<const uint4*>(mp + lane * 16);
                        const uint4 qv = *reinterpret_cast<const uint4*>(Qs + v * QS + lane * 8);
                        const __half2* mh = reinterpret_cast<const __half2*>(&mv);
                        const __half2* qh = reinterpret_cast<const __half2*>(&qv);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float2 a = __half22float2(mh[i]), b = __half22float2(qh[i]);
                            acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
                        }
                    }
                    acc = ares_warp_sum(acc);
                    if (lane == 0) mdot[v * ME + e] = acc;
                }
            }
            cached_ent = (nent == 1) ? el[0] : 0xffffffffu;
        }
#ifdef ATTN_PROBE
        t1 = clock64(); pr[2] += t1 - t0; t0 = t1;
#endif
        if (mma_warp) {
            asm volatile("bar.sync 1, %0;" ::"n"(NT));
        } else {
            asm volatile("bar.arrive 1, %0;" ::"n"(NT));
#ifndef ATTN_NO_LOAD
            if (c + 1 < n_chunks) issue(c + 1, false);
#endif
        }
#ifdef ATTN_PROBE
        t1 = clock64(); pr[3] += t1 - t0; t0 = t1;
#endif

        // the slices of this chunk that belong to this warp
        if constexpr (SWAP) {
            const int vec0 = mt * NVW * 8;
            __half* Pw = Pst + (size_t)warp * (NVW * 8 * PSTR);
            for (int cb = 16 * cg; mma_warp && cb < nc; cb += 16 * CG) {
                // S = K Q^T: the 16 cells of the slice are the rows, the vectors the columns
                float s[NVW][4], s2[NVW][4];
#pragma unroll
                for (int j = 0; j < NVW; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = s2[j][0] = s2[j][1] = s2[j][2] = s2[j][3] = 0.0f;
#pragma unroll
                for (int ks = 0; ks < D / 16; ++ks) {
                    uint32_t a[4];
                    ldmatrix_x4(a, Kt + (cb + (lane & 7) + ((lane >> 3) & 1) * 8) * QS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
                    for (int jj = 0; jj < NVW; jj += 2) {
                        uint32_t b[4];
                        const __half* qp = Qs + (vec0 + jj * 8 + ((lane >> 4) & 1) * 8 + (lane & 7)) * QS + ks * 16 + ((lane >> 3) & 1) * 8;
                        if (jj + 1 < NVW) ldmatrix_x4(b, qp); else ldmatrix_x2(b, qp);
                        if (ks & 1) {
                            mma_f16_f32(s2[jj], a, b[0], b[1]);
                            if (jj + 1 < NVW) mma_f16_f32(s2[jj + 1], a, b[2], b[3]);
                        } else {
                            mma_f16_f32(s[jj], a, b[0], b[1]);
                            if (jj + 1 < NVW) mma_f16_f32(s[jj + 1], a, b[2], b[3]);
                        }
                    }
                }
                const uint32_t cellA = m[(cb + gq) * 3], cellB = m[(cb + gq + 8) * 3];
                const int eA = (int)m[(cb + gq) * 3 + 2], eB = (int)m[(cb + gq + 8) * 3 + 2];
                float alpha[NVW][2];
                float pa[NVW][4];
                bool settled = true;
#pragma unroll
                for (int j = 0; j < NVW; ++j) {
#pragma unroll
                    for (int col = 0; col < 2; ++col) {
                        const int vec = vec0 + 8 * j + 2 * tq + col;
                        const uint32_t lim = vec_lim[vec];
                        float a = s[j][col] + s2[j][col], b = s[j][2 + col] + s2[j][2 + col];
                        if (k_mean) {
                            a += mdot[vec * ME + eA];
                            b += mdot[vec * ME + eB];
                        }
                        a = (cellA != CELL_NONE && cellA <= lim && vec < nvec) ? a : -INFINITY;
                        b = (cellB != CELL_NONE && cellB <= lim && vec < nvec) ? b : -INFINITY;
                        float cmax = fmaxf(a, b);
                        cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 4));
                        cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 8));
                        cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 16));
                        const float nm = fmaxf(mc[j][col], cmax);
                        const float u = nm == -INFINITY ? 0.0f : nm;
                        alpha[j][col] = exp2f(mc[j][col] - u);
                        mc[j][col] = nm;
                        const float p0 = exp2f(a - u), p1 = exp2f(b - u);
                        pa[j][col] = p0;
                        pa[j][2 + col] = p1;
                        lc[j][col] = lc[j][col] * alpha[j][col] + p0 + p1;
#ifdef ATTN_DEBUG
                        if (blockIdx.x == 0 && warp == 0 && lane == 0 && c < 2) {
                            printf("DBG c %d cb %d j %d col %d vec %d s %g s2 %g a %g b %g cmax %g u %g alpha %g p0 %g p1 %g l %g\n", c, cb, j, col, vec, s[j][col], s2[j][col], a, b, cmax, u, alpha[j][col], p0, p1, lc[j][col]);
                        }
#endif
                        settled = settled && alpha[j][col] == 1.0f;
                        const int vl = 8 * j + 2 * tq + col;
                        Pw[vl * PSTR + gq] = __float2half_rn(p0);
                        Pw[vl * PSTR + gq + 8] = __float2half_rn(p1);
                    }
                }
                if (!__all_sync(0xffffffffu, settled)) {
#pragma unroll
                    for (int dt = 0; dt < D / 16; ++dt) {
#pragma unroll
                        for (int j = 0; j < NVW; ++j) {
                            o_sw[dt][j][0] *= alpha[j][0];
                            o_sw[dt][j][1] *= alpha[j][1];
                            o_sw[dt][j][2] *= alpha[j][0];
                            o_sw[dt][j][3] *= alpha[j][1];
                        }
                    }
                }
                __syncwarp();
                uint32_t pb[NVW][2];
#pragma unroll
                for (int jj = 0; jj < NVW; jj += 2) {
                    uint32_t b[4];
                    const __half* pp = Pw + (jj * 8 + ((lane >> 4) & 1) * 8 + (lane & 7)) * PSTR + ((lane >> 3) & 1) * 8;
                    if (jj + 1 < NVW) ldmatrix_x4(b, pp); else ldmatrix_x2(b, pp);
                    pb[jj][0] = b[0];
                    pb[jj][1] = b[1];
                    if (jj + 1 < NVW) {
                        pb[jj + 1][0] = b[2];
                        pb[jj + 1][1] = b[3];
                    }
                }
                // O^T += V^T P^T: the rows are the values of V, the columns the vectors
#pragma unroll
                for (int dt = 0; dt < D / 16; ++dt) {
                    uint32_t av[4];
                    ldmatrix_x4_trans(av, Vt + (cb + ((lane >> 4) & 1) * 8 + (lane & 7)) * QS + dt * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
                    for (int j = 0; j < NVW; ++j) mma_f16_f32(o_sw[dt][j], av, pb[j][0], pb[j][1]);
                }
                // the mean of the V tile(s), weighted by the weights of its cells
                if (v_mean) {
                    for (int e = 0; e < nent; ++e) {
                        float w[NVW][2];
#pragma unroll
                        for (int j = 0; j < NVW; ++j) {
#pragma unroll
                            for (int col = 0; col < 2; ++col) {
                                float x = (eA == e ? pa[j][col] : 0.0f) + (eB == e ? pa[j][2 + col] : 0.0f);
                                x += __shfl_xor_sync(0xffffffffu, x, 4);
                                x += __shfl_xor_sync(0xffffffffu, x, 8);
                                x += __shfl_xor_sync(0xffffffffu, x, 16);
                                w[j][col] = x;
                            }
                        }
                        const unsigned short* mv = reinterpret_cast<const unsigned short*>(
                            conv_pool + (size_t)el[e] * g.entry_c + head_c + g.v_mean_off);
#pragma unroll
                        for (int dt = 0; dt < D / 16; ++dt) {
                            const float m0 = h2f_bits(mv[dt * 16 + gq]), m1 = h2f_bits(mv[dt * 16 + gq + 8]);
#pragma unroll
                            for (int j = 0; j < NVW; ++j) {
                                o_sw[dt][j][0] = fmaf(w[j][0], m0, o_sw[dt][j][0]);
                                o_sw[dt][j][1] = fmaf(w[j][1], m0, o_sw[dt][j][1]);
                                o_sw[dt][j][2] = fmaf(w[j][0], m1, o_sw[dt][j][2]);
                                o_sw[dt][j][3] = fmaf(w[j][1], m1, o_sw[dt][j][3]);
                            }
                        }
                    }
                }
            }
        } else {
#ifdef ATTN_NO_MMA
        for (int cb = 0; false; ) {
#else
        for (int cb = 16 * cg; mma_warp && cb < nc; cb += 16 * CG) {
#endif
            // S^T tile: 16 vectors x 16 cells; even and odd k steps accumulate apart (a chain of
            // 8 dependent products instead of 16), added at the end
            float s[2][4], s2[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
                s2[i][0] = s2[i][1] = s2[i][2] = s2[i][3] = 0.0f;
            }
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
                uint32_t a[4], b[4];
                ldmatrix_x4(a, Qs + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * QS + ks * 16 + (lane >> 4) * 8);
                ldmatrix_x4(b, Kt + (cb + ((lane >> 4) & 1) * 8 + (lane & 7)) * QS + ks * 16 + ((lane >> 3) & 1) * 8);
                if (ks & 1) {
                    mma_f16_f32(s2[0], a, b[0], b[1]);
                    mma_f16_f32(s2[1], a, b[2], b[3]);
                } else {
                    mma_f16_f32(s[0], a, b[0], b[1]);
                    mma_f16_f32(s[1], a, b[2], b[3]);
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                s[i][0] += s2[i][0];
                s[i][1] += s2[i][1];
                s[i][2] += s2[i][2];
                s[i][3] += s2[i][3];
            }
            // scores: add the mean term, mask
            const int vA = mt * 16 + gq, vB = vA + 8;
            const uint32_t limA = vec_lim[vA], limB = vec_lim[vB];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int slot = cb + nt * 8 + 2 * tq + j;
                    const uint32_t cell = m[slot * 3];
                    const int e = (int)m[slot * 3 + 2];
                    float a = s[nt][j], b = s[nt][2 + j];
                    if (k_mean) {
                        a += mdot[vA * ME + e];
                        b += mdot[vB * ME + e];
                    }
                    s[nt][j] = (cell != CELL_NONE && cell <= limA && vA < nvec) ? a : -INFINITY;
                    s[nt][2 + j] = (cell != CELL_NONE && cell <= limB && vB < nvec) ? b : -INFINITY;
                }
            }
            // online softmax over the 16 cells, rows A and B of this thread
            float cA = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
            float cB = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
            cA = fmaxf(cA, __shfl_xor_sync(0xffffffffu, cA, 1));
            cA = fmaxf(cA, __shfl_xor_sync(0xffffffffu, cA, 2));
            cB = fmaxf(cB, __shfl_xor_sync(0xffffffffu, cB, 1));
            cB = fmaxf(cB, __shfl_xor_sync(0xffffffffu, cB, 2));
            const float nA = fmaxf(mA, cA), nB = fmaxf(mB, cB);
            const float uA = nA == -INFINITY ? 0.0f : nA, uB = nB == -INFINITY ? 0.0f : nB;
            const float alphaA = exp2f(mA - uA), alphaB = exp2f(mB - uB);  // exp2(-inf) = 0
            mA = nA;
            mB = nB;
            float p[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                p[nt][0] = exp2f(s[nt][0] - uA);
                p[nt][1] = exp2f(s[nt][1] - uA);
                p[nt][2] = exp2f(s[nt][2] - uB);
                p[nt][3] = exp2f(s[nt][3] - uB);
            }
            lA = lA * alphaA + (p[0][0] + p[0][1]) + (p[1][0] + p[1][1]);
            lB = lB * alphaB + (p[0][2] + p[0][3]) + (p[1][2] + p[1][3]);
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
            uint32_t pa[4];
            pa[0] = pack_half2(p[0][0], p[0][1]);
            pa[1] = pack_half2(p[0][2], p[0][3]);
            pa[2] = pack_half2(p[1][0], p[1][1]);
            pa[3] = pack_half2(p[1][2], p[1][3]);
            // O += P V
#pragma unroll
            for (int nt2 = 0; nt2 < D / 16; ++nt2) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, Vt + (cb + ((lane >> 3) & 1) * 8 + (lane & 7)) * QS + nt2 * 16 + (lane >> 4) * 8);
                mma_f16_f32(o[2 * nt2], pa, b[0], b[1]);
                mma_f16_f32(o[2 * nt2 + 1], pa, b[2], b[3]);
            }
            // the mean of the V tile(s), weighted by the weights of its cells
            if (v_mean) {
                for (int e = 0; e < nent; ++e) {
                    float wA = 0.0f, wB = 0.0f;
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            if ((int)m[(cb + nt * 8 + 2 * tq + j) * 3 + 2] == e) {
                                wA += p[nt][j];
                                wB += p[nt][2 + j];
                            }
                        }
                    }
                    wA += __shfl_xor_sync(0xffffffffu, wA, 1);
                    wA += __shfl_xor_sync(0xffffffffu, wA, 2);
                    wB += __shfl_xor_sync(0xffffffffu, wB, 1);
                    wB += __shfl_xor_sync(0xffffffffu, wB, 2);
                    const __half2* mv = reinterpret_cast<const __half2*>(
                        conv_pool + (size_t)el[e] * g.entry_c + head_c + g.v_mean_off);
#pragma unroll
                    for (int i = 0; i < D / 8; ++i) {
                        const float2 mm = __half22float2(mv[i * 4 + tq]);
                        o[i][0] = fmaf(wA, mm.x, o[i][0]);
                        o[i][1] = fmaf(wA, mm.y, o[i][1]);
                        o[i][2] = fmaf(wB, mm.x, o[i][2]);
                        o[i][3] = fmaf(wB, mm.y, o[i][3]);
                    }
                }
            }
        }
        }
#ifdef ATTN_PROBE
        t1 = clock64(); pr[4] += t1 - t0; t0 = t1;
#endif
        if (warp == 0) meta_store(c + 2);
#ifdef ATTN_PROBE
        t1 = clock64(); pr[5] += t1 - t0; t0 = t1;
#endif
        // the barrier at the top of the next iteration protects the tiles and the metadata
    }
    cp_async_wait<0>();
    __syncthreads();
#ifdef ATTN_PROBE
    if (blockIdx.x == 0 && tid == 0) {
        printf("PROBE chunks %d wait %llu decode %llu meandot %llu sync %llu mma %llu meta %llu\n", n_chunks, pr[0] / n_chunks, pr[1] / n_chunks, pr[2] / n_chunks, pr[3] / n_chunks, pr[4] / n_chunks, pr[5] / n_chunks);
    }
#endif

    // ---- end of the item: merge the cell groups, un-rotate, write
    constexpr int OS = D + 4;
    float* Of = reinterpret_cast<float*>(smem);  // [QROWS][OS]; the queries and tiles are dead now
    float* ml = Of + QROWS * OS;                 // [QROWS][2]: m and l of the second cell group
    if constexpr (SWAP) {
        const int vec0 = mt * NVW * 8;
#pragma unroll
        for (int j = 0; j < NVW; ++j) {
#pragma unroll
            for (int col = 0; col < 2; ++col) {
                lc[j][col] += __shfl_xor_sync(0xffffffffu, lc[j][col], 4);
                lc[j][col] += __shfl_xor_sync(0xffffffffu, lc[j][col], 8);
                lc[j][col] += __shfl_xor_sync(0xffffffffu, lc[j][col], 16);
            }
        }
        if (CG == 2 && mma_warp && cg == 1) {
#pragma unroll
            for (int dt = 0; dt < D / 16; ++dt)
#pragma unroll
                for (int j = 0; j < NVW; ++j)
#pragma unroll
                    for (int c = 0; c < 4; ++c)
                        Of[(vec0 + 8 * j + 2 * tq + (c & 1)) * OS + dt * 16 + gq + 8 * (c >> 1)] = o_sw[dt][j][c];
            if (gq == 0) {
#pragma unroll
                for (int j = 0; j < NVW; ++j)
#pragma unroll
                    for (int col = 0; col < 2; ++col) {
                        ml[(vec0 + 8 * j + 2 * tq + col) * 2] = mc[j][col];
                        ml[(vec0 + 8 * j + 2 * tq + col) * 2 + 1] = lc[j][col];
                    }
            }
        }
        if (CG == 2) __syncthreads();
        if (mma_warp && cg == 0) {
#pragma unroll
            for (int j = 0; j < NVW; ++j) {
#pragma unroll
                for (int col = 0; col < 2; ++col) {
                    const int vec = vec0 + 8 * j + 2 * tq + col;
                    float fa = 1.0f, fb = 0.0f;
                    if (CG == 2) {
                        const float m2 = ml[vec * 2], l2 = ml[vec * 2 + 1];
                        const float nm = fmaxf(mc[j][col], m2);
                        const float u = nm == -INFINITY ? 0.0f : nm;
                        fa = exp2f(mc[j][col] - u);
                        fb = exp2f(m2 - u);
                        lc[j][col] = lc[j][col] * fa + l2 * fb;
                        mc[j][col] = nm;
                    }
                    const float inv = lc[j][col] > 0.0f ? 1.0f / lc[j][col] : 0.0f;
#pragma unroll
                    for (int dt = 0; dt < D / 16; ++dt) {
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            float* at = &Of[vec * OS + dt * 16 + gq + 8 * hh];
                            const float x = CG == 2 ? *at : 0.0f;
                            const float v = o_sw[dt][j][2 * hh + col] * fa + x * fb;
                            *at = v * inv;
                        }
                    }
                }
            }
        }
    } else {
    lA += __shfl_xor_sync(0xffffffffu, lA, 1);
    lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1);
    lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    if (CG == 2 && mma_warp && cg == 1) {
#pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            const int col = i * 8 + 2 * tq;
            *reinterpret_cast<float2*>(&Of[(mt * 16 + gq) * OS + col]) = make_float2(o[i][0], o[i][1]);
            *reinterpret_cast<float2*>(&Of[(mt * 16 + gq + 8) * OS + col]) = make_float2(o[i][2], o[i][3]);
        }
    }
    if (CG == 2 && mma_warp && cg == 1 && tq == 0) {
        ml[(mt * 16 + gq) * 2] = mA;
        ml[(mt * 16 + gq) * 2 + 1] = lA;
        ml[(mt * 16 + gq + 8) * 2] = mB;
        ml[(mt * 16 + gq + 8) * 2 + 1] = lB;
    }
    if (CG == 2) __syncthreads();
    if (mma_warp && cg == 0) {
        if (CG == 2) {
            const float m2A = ml[(mt * 16 + gq) * 2], l2A = ml[(mt * 16 + gq) * 2 + 1];
            const float m2B = ml[(mt * 16 + gq + 8) * 2], l2B = ml[(mt * 16 + gq + 8) * 2 + 1];
            const float nA = fmaxf(mA, m2A), nB = fmaxf(mB, m2B);
            const float uA = nA == -INFINITY ? 0.0f : nA, uB = nB == -INFINITY ? 0.0f : nB;
            const float aA = exp2f(mA - uA), bA = exp2f(m2A - uA);
            const float aB = exp2f(mB - uB), bB = exp2f(m2B - uB);
#pragma unroll
            for (int i = 0; i < D / 8; ++i) {
                const int col = i * 8 + 2 * tq;
                const float2 x = *reinterpret_cast<float2*>(&Of[(mt * 16 + gq) * OS + col]);
                const float2 y = *reinterpret_cast<float2*>(&Of[(mt * 16 + gq + 8) * OS + col]);
                o[i][0] = o[i][0] * aA + x.x * bA;
                o[i][1] = o[i][1] * aA + x.y * bA;
                o[i][2] = o[i][2] * aB + y.x * bB;
                o[i][3] = o[i][3] * aB + y.y * bB;
            }
            lA = lA * aA + l2A * bA;
            lB = lB * aB + l2B * bB;
            mA = nA;
            mB = nB;
        }
        const float invA = lA > 0.0f ? 1.0f / lA : 0.0f, invB = lB > 0.0f ? 1.0f / lB : 0.0f;
#pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            const int col = i * 8 + 2 * tq;
            *reinterpret_cast<float2*>(&Of[(mt * 16 + gq) * OS + col]) = make_float2(o[i][0] * invA, o[i][1] * invA);
            *reinterpret_cast<float2*>(&Of[(mt * 16 + gq + 8) * OS + col]) = make_float2(o[i][2] * invB, o[i][3] * invB);
        }
    }
    }
    const bool unrotate = conv && fmt_rotated(g.vf);
    const int Hq = g.H * G;
    __syncthreads();
    // lse for the states: m + log2(l), -inf when the row saw nothing
    if (SWAP && mma_warp && cg == 0 && gq == 0 && !(it.flags & ITEM_FINAL)) {
        const int vec0 = mt * NVW * 8;
#pragma unroll
        for (int j = 0; j < NV1; ++j) {
#pragma unroll
            for (int col = 0; col < 2; ++col) {
                const int v = vec0 + 8 * j + 2 * tq + col;
                if (v < nvec) {
                    const size_t row = row_ids[it.row0 + v / G];
                    const size_t slot = rows[row].slot_base + it.slot;
                    st_lse[slot * Hq + it.kv_head * G + v % G] =
                        lc[j][col] > 0.0f ? mc[j][col] + log2f(lc[j][col]) : -INFINITY;
                }
            }
        }
    }
    if (!SWAP && mma_warp && cg == 0 && tq == 0 && !(it.flags & ITEM_FINAL)) {
        for (int h2 = 0; h2 < 2; ++h2) {
            const int v = mt * 16 + gq + 8 * h2;
            if (v < nvec) {
                const float mm = h2 ? mB : mA, ll = h2 ? lB : lA;
                const size_t row = row_ids[it.row0 + v / G];
                const size_t slot = rows[row].slot_base + it.slot;
                st_lse[slot * Hq + it.kv_head * G + v % G] = ll > 0.0f ? mm + log2f(ll) : -INFINITY;
            }
        }
    }
    // rows of the output, one warp per vector
    for (int v = warp; v < nvec; v += WT) {
        const size_t row = row_ids[it.row0 + v / G];
        const int hl = it.kv_head * G + v % G;
        for (int gi = 0; gi < D / 32; ++gi) {
            float x = Of[v * OS + gi * 32 + lane];
            if (unrotate) x = rotate32(x, lane);
            if (it.flags & ITEM_FINAL) {
                const size_t at = (row * g.kv_total * G + (size_t)(g.head0 * G + hl)) * D + gi * 32 + lane;
                if (gate != nullptr) x *= 1.0f / (1.0f + expf(-gate[at]));
                out[at] = x;
            } else {
                const size_t slot = rows[row].slot_base + it.slot;
                st_acc[(slot * Hq + hl) * D + gi * 32 + lane] = x;
            }
        }
    }
}

// Merges the pieces of a row (the states of its items) into the output, with the
// sigmoid gate as epilogue. Grid (rows, local query heads), block D (up to 256).
template <int D>
static __device__ void attn_merge_body(
    const AttnRow* rows, const float* st_acc, const float* st_lse, const float* gate, float* out,
    uint64_t cfg) {
    const AttnGeo g = attn_geo(cfg);
    const int Hq = g.H * g.G;
    const size_t row = blockIdx.x;
    const int hl = blockIdx.y;
    const AttnRow r = rows[row];
    if (r.n_slots == 0) return;
    float m = -INFINITY;
    for (uint32_t i = 0; i < r.n_slots; ++i) m = fmaxf(m, st_lse[(size_t)(r.slot_base + i) * Hq + hl]);
    if (m == -INFINITY) m = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f, den = 0.0f;
        for (uint32_t i = 0; i < r.n_slots; ++i) {
            const size_t s = (size_t)(r.slot_base + i) * Hq + hl;
            const float w = exp2f(st_lse[s] - m);
            acc = fmaf(w, st_acc[s * D + d], acc);
            den += w;
        }
        float x = den > 0.0f ? acc / den : 0.0f;
        const size_t at = (row * g.kv_total * g.G + (size_t)(g.head0 * g.G + hl)) * D + d;
        if (gate != nullptr) x *= 1.0f / (1.0f + expf(-gate[at]));
        out[at] = x;
    }
}

#define ARES_ATTN_KERNEL(NAME, D, MT, CG, CHUNK, XW, NVW)                                                  \
    extern "C" __global__ void __launch_bounds__(32 * ((MT) * (CG) + (XW)), 1) NAME(                  \
        const AttnItem* items, const uint32_t* ranges, const AttnRow* rows, const uint32_t* row_ids, \
        const uint32_t* tables, const uint8_t* exact_pool, const uint8_t* conv_pool, const float* q, \
        const float* gate, float* out, float* st_acc, float* st_lse, uint64_t cfg, float scale_log2, \
        uint32_t rawcap) {                                                                            \
        attn_body<D, MT, CG, CHUNK, XW, NVW>(                                                                  \
            items, ranges, rows, row_ids, tables, exact_pool, conv_pool, q, gate, out, st_acc,        \
            st_lse, cfg, scale_log2, rawcap);                                                         \
    }

#define ARES_ATTN_MERGE(NAME, D)                                                                       \
    extern "C" __global__ void NAME(                                                                   \
        const AttnRow* rows, const float* st_acc, const float* st_lse, const float* gate, float* out, \
        uint64_t cfg) {                                                                                \
        attn_merge_body<D>(rows, st_acc, st_lse, gate, out, cfg);                                      \
    }
