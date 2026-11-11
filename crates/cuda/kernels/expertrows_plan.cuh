// The plan of the many-row expert kernels (expertrows_layout.cuh): from the picks of
// any number of rows to the pairs sorted by expert and the items the tile kernels
// run, built on the device.
//
// The sort is a stable counting sort: the pairs of one expert stay in the order of
// (row, pick). A block of 1024 threads takes 1024 pairs; the rank of a pair among
// the earlier ones of its block with the same expert comes from `__match_any_sync`
// within a warp and from taking the warps one after the other, so nothing depends on
// the order in which threads run.
//
// - Up to 1024 pairs (a window of drafts, or 102 rows of ten picks): one launch,
//   `ares_er_plan_small`.
// - More: `ares_er_plan_count` (a block per 1024 pairs: the rank in its block and the
//   block's count of each expert), `ares_er_plan_offsets` (one block: the prefix of
//   each expert's counts over the blocks, the first position of each expert, the
//   items) and `ares_er_plan_scatter` (the pairs to their positions).
//
// A pair whose expert id is not below n_experts is dropped: it has no position
// (`pair_pos` = ARES_ER_NO_POS) and no kernel computes it.
#pragma once

#include "expertrows_layout.cuh"

// The part of the plan that follows from the experts' counts: the first position of
// each expert, the items, the header. `count` is the calling thread's expert's count
// (thread e = expert e); `shared_base` receives the first position (an entry per
// thread, for the scatter).
static __device__ __forceinline__ void ares_er_plan_items(
    const AresErPlan& plan, uint32_t count, uint32_t n_experts, uint32_t tile, uint32_t token, uint32_t* warp_sums,
    uint32_t* shared_base) {
    const uint32_t e = threadIdx.x;
    uint32_t n_valid = 0, n_items = 0;
    const uint32_t base = ares_er_block_scan(count, warp_sums, &n_valid);
    const uint32_t n_here = (count + tile - 1) / tile;
    const uint32_t item_base = ares_er_block_scan(n_here, warp_sums, &n_items);
    if (e < n_experts) {
        plan.expert_base()[e] = base;
        plan.missing()[e] = 0;
        for (uint32_t m = 0; m < n_here; m++) {
            AresErItem item;
            item.expert = e;
            item.first = base + m * tile;
            item.count = min(tile, count - m * tile);
            item.pad = 0;
            plan.items()[item_base + m] = item;
        }
    }
    shared_base[e] = base;
    if (e == 0) {
        plan.expert_base()[n_experts] = n_valid;
        AresErHeader header;
        header.n_items = n_items;
        header.n_valid = n_valid;
        header.missing_pairs = 0;
        header.token = token;
        for (int i = 0; i < 12; i++) {
            header.reserved[i] = 0;
        }
        *plan.header() = header;
    }
    __syncthreads();
}

static __device__ __forceinline__ void ares_er_plan_place(
    const AresErPlan& plan, const float* weights, uint32_t p, uint32_t e, uint32_t pos, uint32_t n_experts, uint32_t top_k,
    uint32_t n_pairs) {
    if (p >= n_pairs) {
        return;
    }
    if (e < n_experts) {
        plan.pair_row()[pos] = p / top_k;
        plan.pair_weight()[pos] = weights[p];
        plan.pair_pos()[p] = pos;
    } else {
        plan.pair_pos()[p] = ARES_ER_NO_POS;
    }
}

// The plan of up to 1024 pairs in one block of ARES_ER_PLAN_THREADS threads.
//
//   record      the route record (a header of 16 bytes, n_pairs ids, n_pairs weights)
//   plan        the plan (expertrows_layout.cuh)
ARES_KERNEL ares_er_plan_small(
    const uint8_t* record, uint8_t* plan_bytes, uint32_t n_experts, uint32_t n_pairs, uint32_t top_k, uint32_t tile,
    uint32_t token) {
    __shared__ uint32_t hist[ARES_ER_PLAN_THREADS];
    __shared__ uint32_t sums[32];
    const AresErPlan plan(plan_bytes, n_experts, n_pairs, tile);
    const uint32_t* ids = reinterpret_cast<const uint32_t*>(record + 16);
    const float* weights = reinterpret_cast<const float*>(record + 16 + 4 * static_cast<size_t>(n_pairs));
    const uint32_t p = threadIdx.x;
    const uint32_t e = p < n_pairs ? ids[p] : 0xFFFFFFFFu;
    hist[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t rank = ares_er_block_rank(e, hist, n_experts);
    const uint32_t count = threadIdx.x < n_experts ? hist[threadIdx.x] : 0;
    __syncthreads();
    ares_er_plan_items(plan, count, n_experts, tile, token, sums, hist);
    const uint32_t pos = e < n_experts ? hist[e] + rank : 0;
    ares_er_plan_place(plan, weights, p, e, pos, n_experts, top_k, n_pairs);
}

// The first of the three launches of a plan of more than 1024 pairs: a block per 1024
// pairs. Leaves the rank of each pair in its block in `pair_pos` and the block's count
// of each expert in the block counts.
ARES_KERNEL ares_er_plan_count(
    const uint8_t* record, uint8_t* plan_bytes, uint32_t n_experts, uint32_t n_pairs, uint32_t, uint32_t tile) {
    __shared__ uint32_t hist[ARES_ER_PLAN_THREADS];
    const AresErPlan plan(plan_bytes, n_experts, n_pairs, tile);
    const uint32_t* ids = reinterpret_cast<const uint32_t*>(record + 16);
    const uint32_t p = blockIdx.x * ARES_ER_PLAN_THREADS + threadIdx.x;
    const uint32_t e = p < n_pairs ? ids[p] : 0xFFFFFFFFu;
    hist[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t rank = ares_er_block_rank(e, hist, n_experts);
    if (threadIdx.x < n_experts) {
        plan.block_counts()[static_cast<size_t>(blockIdx.x) * n_experts + threadIdx.x] = hist[threadIdx.x];
    }
    if (p < n_pairs) {
        plan.pair_pos()[p] = rank;
    }
}

// The second launch, one block: turns the block counts into the offset of each block
// inside its expert, and makes the items and the header.
ARES_KERNEL ares_er_plan_offsets(uint8_t* plan_bytes, uint32_t n_experts, uint32_t n_pairs, uint32_t tile, uint32_t token) {
    __shared__ uint32_t sums[32];
    __shared__ uint32_t base[ARES_ER_PLAN_THREADS];
    const AresErPlan plan(plan_bytes, n_experts, n_pairs, tile);
    uint32_t count = 0;
    if (threadIdx.x < n_experts) {
        uint32_t* column = plan.block_counts() + threadIdx.x;
        for (uint32_t b = 0; b < plan.l.n_blocks; b++) {
            const uint32_t c = column[static_cast<size_t>(b) * n_experts];
            column[static_cast<size_t>(b) * n_experts] = count;
            count += c;
        }
    }
    ares_er_plan_items(plan, count, n_experts, tile, token, sums, base);
}

// The third launch: a block per 1024 pairs puts each pair at its position.
ARES_KERNEL ares_er_plan_scatter(
    const uint8_t* record, uint8_t* plan_bytes, uint32_t n_experts, uint32_t n_pairs, uint32_t top_k, uint32_t tile) {
    const AresErPlan plan(plan_bytes, n_experts, n_pairs, tile);
    const uint32_t* ids = reinterpret_cast<const uint32_t*>(record + 16);
    const float* weights = reinterpret_cast<const float*>(record + 16 + 4 * static_cast<size_t>(n_pairs));
    const uint32_t p = blockIdx.x * ARES_ER_PLAN_THREADS + threadIdx.x;
    if (p >= n_pairs) {
        return;
    }
    const uint32_t e = ids[p];
    uint32_t pos = 0;
    if (e < n_experts) {
        pos = plan.expert_base()[e] + plan.block_counts()[static_cast<size_t>(blockIdx.x) * n_experts + e] + plan.pair_pos()[p];
    }
    ares_er_plan_place(plan, weights, p, e, pos, n_experts, top_k, n_pairs);
}
