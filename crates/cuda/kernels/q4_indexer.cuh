// The indexer of qwen4exp's sparse attention (section 3.6 of
// docs/dev/studies/qwen4exp-architecture.md): the queries of the rows, the
// pooled keys of the blocks a step completes, the raw keys a sequence carries
// for its unfinished block, and the scores of the blocks a row sees.
//
// Where things lie on the device:
//
//   pooled keys   a key pool per attention layer. A slot holds the keys of the
//                 128 / B blocks of one entry of 128 cells, ID values each, as
//                 f32 or f16 (the format of the pool). The slot of an entry is
//                 key_table[table + cell / 128]: a u32 array parallel to the
//                 attention tables (attn_layout.cuh), same index, because an
//                 attention entry changes its pool and slot when it is
//                 converted and the keys stay where they are.
//   carried state per sequence slot and layer, `carry_stride` f32 values:
//                 [B - 1][ID] the raw keys of the cells of the unfinished block
//                 in cell order, then rot / 2 (cos, sin) pairs: the rotation of
//                 that block's first cell. Read by the window, written by the
//                 commit only.
//   rows          the cell and the table of a row come from the attention
//                 plan's row array (AttnRow: table, pos), which the K/V append
//                 reads too; the cell of a segment's first row is the `pos` of
//                 its first row and the rows of a segment are consecutive cells.
//
// The configuration word of the kernels: bits 0-15 ID, 16-23 IH (query heads),
// 24-31 B (cells per block), 32-47 rot (rotated values of a head), 48-51 the
// format of the key pool (0 f32, 1 f16).
#pragma once

#include "common.cuh"
#include "layer_step.cuh"

// The first 16 bytes of this are AttnRow of attn_layout.cuh.
struct Q4AttnRow {
    uint32_t table;  // index of the first entry of the row's sequence in the tables
    uint32_t pos;    // the cell the row writes
    uint32_t slot_base;
    uint32_t n_slots;
};

#define Q4_ENTRY_CELLS 128u
#define Q4_MAX_ID 256u
#define Q4_KEY_F16 1u

struct Q4Geo {
    uint32_t id, ih, block, rot, fmt;
    uint32_t bpe;       // blocks per entry
    size_t key_bytes;   // bytes of one pooled key in the pool
    size_t slot_bytes;  // bytes of a slot of the pool
};

static __device__ __forceinline__ Q4Geo q4_geo(uint64_t cfg) {
    Q4Geo g;
    g.id = static_cast<uint32_t>(cfg & 0xffffu);
    g.ih = static_cast<uint32_t>((cfg >> 16) & 0xffu);
    g.block = static_cast<uint32_t>((cfg >> 24) & 0xffu);
    g.rot = static_cast<uint32_t>((cfg >> 32) & 0xffffu);
    g.fmt = static_cast<uint32_t>((cfg >> 48) & 0xfu);
    g.bpe = Q4_ENTRY_CELLS / g.block;
    g.key_bytes = static_cast<size_t>(g.id) * (g.fmt == Q4_KEY_F16 ? 2 : 4);
    g.slot_bytes = g.key_bytes * g.bpe;
    return g;
}

// The pooled key of block j of the sequence whose table starts at `table`.
static __device__ __forceinline__ uint8_t* q4_key_at(const Q4Geo& g, const uint32_t* key_table, uint8_t* key_pool,
                                                      uint32_t table, uint32_t j) {
    const uint32_t slot = key_table[table + j / g.bpe];
    return key_pool + static_cast<size_t>(slot) * g.slot_bytes + static_cast<size_t>(j % g.bpe) * g.key_bytes;
}

// The rotation of a normed head staged in shared memory: element i of the
// result. Pair p < rot / 2 is (x[p], x[p + rot / 2]).
static __device__ __forceinline__ float q4_rotated(const float* x, const float2* tab, uint32_t i, uint32_t half) {
    if (i < half) {
        const float2 cs = tab[i];
        return x[i] * cs.x - x[i + half] * cs.y;
    }
    if (i < 2 * half) {
        const float2 cs = tab[i - half];
        return x[i - half] * cs.y + x[i] * cs.x;
    }
    return x[i];
}

// The indexer of one attention layer for the rows of a step, one warp per item:
//
//   items [0, n_rows * IH): a query head. u[row][e] = rope(rms(iq[row][e]) * q_norm, row)
//   then n_segs * max_new_blocks items: block b of the blocks segment s
//   completes. With c0 the cell of the segment's first row, block j = c0 / B + b
//   is complete inside the window when j < (c0 + n_rows) / B, and its key is
//       P[j] = rope(rms(mean of the raw keys of cells B j .. B j + B - 1) * k_norm, cell B j)
//   with the raw keys of cells before c0 from the carried state and the others
//   from `raw` ([row][ID]), rotated by the step's rotary table when the block
//   begins inside the window and by the carried rotation otherwise. The key is
//   written into the key pool, also for a segment whose rows may be rejected:
//   a key is read only by rows that see its whole block, and a block that is
//   cut by a rejection is completed and written again later. A replay
//   (ARES_SEG_REPLAY) writes no key: its blocks are cached.
//
// Nothing of the carried state is written. `rope` is the table of
// ares_rope_table ([row][rot / 2] pairs of cos, sin). ID is at most 256.
ARES_KERNEL ares_q4_indexer_window(const AresStep* step, const Q4AttnRow* attn_rows, const float* iq,
                                   const float* raw, const float* q_norm, const float* k_norm, const float2* rope,
                                   float* u_out, const float* carry, uint64_t carry_stride,
                                   const uint32_t* key_table, uint8_t* key_pool, uint64_t cfg, float eps,
                                   uint32_t max_new_blocks) {
    __shared__ float stage[8][Q4_MAX_ID];
    const Q4Geo g = q4_geo(cfg);
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t item = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    const size_t n_query = static_cast<size_t>(step->n_rows) * g.ih;
    const uint32_t half = g.rot >> 1;
    float* x = stage[warp];

    if (item < n_query) {
        const size_t row = item / g.ih;
        const float* src = iq + item * g.id;
        float ssq = 0.0f;
        for (uint32_t i = lane; i < g.id; i += 32) {
            const float v = src[i];
            x[i] = v;
            ssq = fmaf(v, v, ssq);
        }
        ssq = ares_warp_sum(ssq);
        const float scale = rsqrtf(ssq / static_cast<float>(g.id) + eps);
        for (uint32_t i = lane; i < g.id; i += 32) {
            x[i] = x[i] * scale * q_norm[i];
        }
        __syncwarp();
        const float2* tab = rope + row * half;
        float* dst = u_out + item * g.id;
        for (uint32_t i = lane; i < g.id; i += 32) {
            dst[i] = q4_rotated(x, tab, i, half);
        }
        return;
    }

    const size_t pool_item = item - n_query;
    const uint32_t s = static_cast<uint32_t>(pool_item / max_new_blocks);
    const uint32_t b = static_cast<uint32_t>(pool_item % max_new_blocks);
    if (s >= step->n_segs) {
        return;
    }
    const AresSeg seg = ares_step_segs(step)[s];
    // A replay's blocks are in the key pool already, which other sequences read.
    if (seg.n_rows == 0 || (seg.flags & ARES_SEG_REPLAY) != 0) {
        return;
    }
    const Q4AttnRow first_row = attn_rows[seg.row0];
    const uint32_t c0 = first_row.pos;
    const uint32_t j = c0 / g.block + b;
    if (j >= (c0 + seg.n_rows) / g.block) {
        return;
    }
    const uint32_t first = j * g.block;  // the block's first cell
    const float* carried = carry + static_cast<size_t>(seg.slot) * carry_stride;
    const uint32_t carried_first = (c0 / g.block) * g.block;
    for (uint32_t i = lane; i < g.id; i += 32) {
        float sum = 0.0f;
        for (uint32_t m = 0; m < g.block; m++) {
            const uint32_t cell = first + m;
            sum += cell < c0 ? carried[static_cast<size_t>(cell - carried_first) * g.id + i]
                             : raw[(static_cast<size_t>(seg.row0) + (cell - c0)) * g.id + i];
        }
        x[i] = sum / static_cast<float>(g.block);
    }
    float ssq = 0.0f;
    for (uint32_t i = lane; i < g.id; i += 32) {
        ssq = fmaf(x[i], x[i], ssq);
    }
    ssq = ares_warp_sum(ssq);
    const float scale = rsqrtf(ssq / static_cast<float>(g.id) + eps);
    for (uint32_t i = lane; i < g.id; i += 32) {
        x[i] = x[i] * scale * k_norm[i];
    }
    __syncwarp();
    const float2* tab = first >= c0 ? rope + (static_cast<size_t>(seg.row0) + (first - c0)) * half
                                    : reinterpret_cast<const float2*>(carried + static_cast<size_t>(g.block - 1) * g.id);
    uint8_t* dst = q4_key_at(g, key_table, key_pool, first_row.table, j);
    for (uint32_t i = lane; i < g.id; i += 32) {
        const float v = q4_rotated(x, tab, i, half);
        if (g.fmt == Q4_KEY_F16) {
            reinterpret_cast<__half*>(dst)[i] = __float2half_rn(v);
        } else {
            reinterpret_cast<float*>(dst)[i] = v;
        }
    }
}

// The commit of the carried indexer state of one attention layer: after the
// first m rows of a segment (m = accepted, or all rows of a direct segment)
// the sequence has e = c0 + m cells, and it carries the raw keys of the cells
// B * floor(e / B) .. e - 1 and the rotation of the first of them. Cells of
// that block from before the window are already in place (the block was
// unfinished then too), so only the window's rows are written, and the
// rotation when the block begins inside the window.
//   which = 0: the segments that are not direct; 1: the direct ones; 2: both.
// One block per segment, one thread per value.
ARES_KERNEL ares_q4_indexer_commit(const AresStep* step, const Q4AttnRow* attn_rows, const float* raw,
                                   const float2* rope, float* carry, uint64_t carry_stride, uint64_t cfg,
                                   uint32_t which) {
    const Q4Geo g = q4_geo(cfg);
    const AresSeg seg = ares_step_segs(step)[blockIdx.x];
    const bool direct = (seg.flags & ARES_SEG_DIRECT) != 0;
    if ((which == 0 && direct) || (which == 1 && !direct)) {
        return;
    }
    const uint32_t m = direct ? seg.n_rows : min(seg.accepted, seg.n_rows);
    if (m == 0) {
        return;
    }
    const uint32_t c0 = attn_rows[seg.row0].pos;
    const uint32_t end = c0 + m;
    const uint32_t first = (end / g.block) * g.block;
    float* carried = carry + static_cast<size_t>(seg.slot) * carry_stride;
    for (uint32_t cell = max(first, c0); cell < end; cell++) {
        const float* src = raw + (static_cast<size_t>(seg.row0) + (cell - c0)) * g.id;
        float* dst = carried + static_cast<size_t>(cell - first) * g.id;
        for (uint32_t i = threadIdx.x; i < g.id; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
    if (first >= c0 && first < end) {
        const uint32_t half = g.rot >> 1;
        const float2* src = rope + (static_cast<size_t>(seg.row0) + (first - c0)) * half;
        float2* dst = reinterpret_cast<float2*>(carried + static_cast<size_t>(g.block - 1) * g.id);
        for (uint32_t p = threadIdx.x; p < half; p += blockDim.x) {
            dst[p] = src[p];
        }
    }
}

// The scores of the blocks the rows of a launch see:
//   score[row][j] = sum_e max(0, u[row][e] . P[j])       for j < floor(n / B), n = pos + 1
// for the rows row0 .. row0 + grid.y - 1 of the step that see more than
// `dense_limit` cells (the others attend to everything and score nothing).
// Keys are read from the key pool in its format. `scores` gets `score_stride`
// f32 per row of the launch.
//
// A warp computes one block at a time: lane l holds the values l, l + 32, ...
// of the key and of the IH query heads, so a key is one coalesced read per 32
// values. A thread block takes `span` consecutive blocks. IH <= 4, ID <= 128.
#define Q4_SCORE_MAX_IH 4
#define Q4_SCORE_LANE_VALUES 4

ARES_KERNEL ares_q4_block_scores(const Q4AttnRow* attn_rows, const float* u, const uint32_t* key_table,
                                 const uint8_t* key_pool, float* scores, uint64_t cfg, uint32_t row0,
                                 uint32_t score_stride, uint32_t span, uint32_t dense_limit) {
    const Q4Geo g = q4_geo(cfg);
    const uint32_t row = row0 + blockIdx.y;
    const Q4AttnRow r = attn_rows[row];
    const uint32_t n = r.pos + 1;
    if (n <= dense_limit) {
        return;
    }
    const uint32_t nb = n / g.block;
    const uint32_t j0 = blockIdx.x * span;
    if (j0 >= nb) {
        return;
    }
    const uint32_t j1 = min(j0 + span, nb);
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const uint32_t lane_values = g.id >> 5;

    float uu[Q4_SCORE_MAX_IH][Q4_SCORE_LANE_VALUES];
    const float* ur = u + static_cast<size_t>(row) * g.ih * g.id;
#pragma unroll
    for (int e = 0; e < Q4_SCORE_MAX_IH; e++) {
#pragma unroll
        for (int k = 0; k < Q4_SCORE_LANE_VALUES; k++) {
            uu[e][k] = (e < g.ih && k < lane_values) ? ur[static_cast<size_t>(e) * g.id + k * 32 + lane] : 0.0f;
        }
    }
    float* out = scores + static_cast<size_t>(blockIdx.y) * score_stride;
    for (uint32_t j = j0 + warp; j < j1; j += warps) {
        const uint8_t* key = q4_key_at(g, key_table, const_cast<uint8_t*>(key_pool), r.table, j);
        float p[Q4_SCORE_LANE_VALUES];
#pragma unroll
        for (int k = 0; k < Q4_SCORE_LANE_VALUES; k++) {
            p[k] = 0.0f;
            if (k < lane_values) {
                p[k] = g.fmt == Q4_KEY_F16 ? __half2float(reinterpret_cast<const __half*>(key)[k * 32 + lane])
                                           : reinterpret_cast<const float*>(key)[k * 32 + lane];
            }
        }
        float score = 0.0f;
#pragma unroll
        for (int e = 0; e < Q4_SCORE_MAX_IH; e++) {
            if (e < g.ih) {
                float dot = 0.0f;
#pragma unroll
                for (int k = 0; k < Q4_SCORE_LANE_VALUES; k++) {
                    dot = fmaf(uu[e][k], p[k], dot);
                }
                score += fmaxf(ares_warp_sum(dot), 0.0f);
            }
        }
        if (lane == 0) {
            out[j] = score;
        }
    }
}
