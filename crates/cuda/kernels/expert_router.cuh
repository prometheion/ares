// The router of a mixture-of-experts layer: from the logits of the E experts
// of each row to the K experts it uses, their weights, and the list of
// distinct experts of the step with the rows that picked each.
//
//   prob  = softmax(logit)                      over all E
//   ids   = the K experts with the largest prob; equal values: lower index first
//   wt[j] = prob[ids[j]] / max(sum_j prob[ids[j]], 2^-14)
//
// The logits are the product of the router matrix (F32, E x N) with the rows,
// which is the float weight kernel (experts.cu instantiates it). This kernel
// does the rest in one thread block with a thread per expert: every thread
// holds its expert's probability, counts how many experts come before it, and
// the K first write themselves into the row's list.
//
// The same thread then knows which rows picked its expert, so the groups
// (expert_layout.cuh) fall out of it: a thread whose expert was picked counts
// the picked experts below it and writes its group there. The groups are
// therefore in ascending order of the expert, whatever rows the step has, and
// a row's experts are summed in the same order alone and in a batch.
//
// A step of one row does both in one launch (ares_expert_select). The rows of
// a larger step are picked side by side, a block per row (ares_expert_pick),
// which leaves each expert's weight for the row where its logit was, and a
// second launch makes the groups (ares_expert_group): the selection of a row
// is a few hundred compares per thread, and eight of them one after the
// other in one block cost more than the launch.
#pragma once

#include "common.cuh"
#include "expert_layout.cuh"

// The maximum of `v` over all threads of the block, returned to every thread.
// `partial` as in ares_block_sum.
static __device__ __forceinline__ float ares_block_max(float v, float* partial) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    const uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        partial[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    const uint32_t n_warps = blockDim.x >> 5;
    float m = lane < n_warps ? partial[lane] : -INFINITY;
    for (int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    }
    return m;
}

// Shared memory of a block that picks.
struct AresPickShared {
    float prob_of[ARES_EXPERT_MAX_EXPERTS];
    float partial[32];
    float top[ARES_EXPERT_MAX_K];
};

// Picks the experts of one row, in a block with a thread per expert: writes
// the row's `k` ids and weights (largest first) and returns this thread's
// expert's weight, or -1 when the row did not pick it.
static __device__ __forceinline__ float ares_expert_pick_row(
    const float* logits, uint32_t* ids, float* weights, uint32_t n_experts, uint32_t k,
    AresPickShared& sh) {
    const uint32_t e = threadIdx.x;
    const bool live = e < n_experts;
    const float logit = live ? logits[e] : -INFINITY;
    const float largest = ares_block_max(logit, sh.partial);
    __syncthreads();
    const float scaled = live ? expf(logit - largest) : 0.0f;
    const float total = ares_block_sum(scaled, sh.partial);
    const float prob = scaled / total;
    sh.prob_of[e] = live ? prob : -1.0f;
    __syncthreads();

    // How many experts come before this one: a larger probability, or the
    // same and a lower index.
    uint32_t rank = 0;
    for (uint32_t o = 0; o < n_experts; o++) {
        const float other = sh.prob_of[o];
        rank += (other > prob || (other == prob && o < e)) ? 1u : 0u;
    }
    const bool picked = live && rank < k;
    if (picked) {
        sh.top[rank] = prob;
    }
    __syncthreads();
    float w = -1.0f;
    if (picked) {
        // Every picked thread adds the K values in the same order.
        float sum = 0.0f;
        for (uint32_t j = 0; j < k; j++) {
            sum += sh.top[j];
        }
        w = prob / fmaxf(sum, 6.103515625e-5f);
        ids[rank] = e;
        weights[rank] = w;
    }
    __syncthreads();
    return w;
}

// Writes the groups and the header of the record, in a block with a thread
// per expert: `rows` are the rows that picked this thread's expert and
// `weight` their weights. `rows_of` is shared memory, an entry per thread.
static __device__ __forceinline__ void ares_expert_write_groups(
    uint32_t rows, const float* weight, const int32_t* slot_of, uint8_t* record,
    uint32_t n_experts, uint32_t cap, uint32_t token, uint32_t* rows_of) {
    const uint32_t e = threadIdx.x;
    AresExpertGroup* groups =
        reinterpret_cast<AresExpertGroup*>(record + sizeof(AresRouteHeader) + 8 * static_cast<size_t>(cap));
    rows_of[e] = e < n_experts ? rows : 0u;
    __syncthreads();
    if (rows != 0) {
        uint32_t index = 0;
        uint32_t first_pair = 0;
        for (uint32_t o = 0; o < e; o++) {
            const uint32_t r = rows_of[o];
            index += r != 0 ? 1u : 0u;
            first_pair += __popc(r);
        }
        AresExpertGroup group;
        group.expert = e;
        group.slot = slot_of[e];
        group.rows = rows;
        group.first_pair = first_pair;
#pragma unroll
        for (int i = 0; i < ARES_MAX_BATCH; i++) {
            group.weight[i] = weight[i];
        }
        groups[index] = group;
    }
    if (e == 0) {
        uint32_t n_groups = 0;
        uint32_t n_missing = 0;
        for (uint32_t o = 0; o < n_experts; o++) {
            if (rows_of[o] != 0) {
                n_groups++;
                n_missing += slot_of[o] < 0 ? 1u : 0u;
            }
        }
        AresRouteHeader header;
        header.token = token;
        header.n_groups = n_groups;
        header.n_missing = n_missing;
        header.n_pairs = cap;
        *reinterpret_cast<AresRouteHeader*>(record) = header;
    }
}

// The router's selection for a step of one row: one block of n_experts
// threads (rounded up to whole warps).
//
//   logits   n_experts f32
//   slot_of  the layer's slot table: n_experts entries
//   record   the route record (expert_layout.cuh), written whole
//   token    goes into the record's header
ARES_KERNEL ares_expert_select(
    const float* logits, const int32_t* slot_of, uint8_t* record, uint32_t n_experts, uint32_t k,
    uint32_t token) {
    __shared__ AresPickShared sh;
    __shared__ uint32_t rows_of[ARES_EXPERT_MAX_EXPERTS];
    uint32_t* ids = reinterpret_cast<uint32_t*>(record + sizeof(AresRouteHeader));
    float* weights = reinterpret_cast<float*>(record + sizeof(AresRouteHeader) + 4 * static_cast<size_t>(k));
    float weight[ARES_MAX_BATCH];
#pragma unroll
    for (int b = 0; b < ARES_MAX_BATCH; b++) {
        weight[b] = 0.0f;
    }
    const float w = ares_expert_pick_row(logits, ids, weights, n_experts, k, sh);
    weight[0] = w;
    ares_expert_write_groups(w >= 0.0f ? 1u : 0u, weight, slot_of, record, n_experts, k, token, rows_of);
}

// The selection of each row of a larger step: a block per row. The row's ids
// and weights go into the record, and every expert's weight for the row (-1
// when the row did not pick it) replaces its logit, for ares_expert_group.
ARES_KERNEL ares_expert_pick(float* logits, uint8_t* record, uint32_t n_experts, uint32_t k, uint32_t n_rows) {
    __shared__ AresPickShared sh;
    const uint32_t b = blockIdx.x;
    const uint32_t cap = n_rows * k;
    uint32_t* ids = reinterpret_cast<uint32_t*>(record + sizeof(AresRouteHeader)) + b * k;
    float* weights =
        reinterpret_cast<float*>(record + sizeof(AresRouteHeader) + 4 * static_cast<size_t>(cap)) + b * k;
    float* row = logits + static_cast<size_t>(b) * n_experts;
    const float w = ares_expert_pick_row(row, ids, weights, n_experts, k, sh);
    if (threadIdx.x < n_experts) {
        row[threadIdx.x] = w;
    }
}

// The groups and the header of the record of a step whose rows
// ares_expert_pick has picked: one block of n_experts threads.
ARES_KERNEL ares_expert_group(
    const float* picked, const int32_t* slot_of, uint8_t* record, uint32_t n_experts, uint32_t k,
    uint32_t n_rows, uint32_t token) {
    __shared__ uint32_t rows_of[ARES_EXPERT_MAX_EXPERTS];
    const uint32_t e = threadIdx.x;
    uint32_t rows = 0;
    float weight[ARES_MAX_BATCH];
#pragma unroll
    for (int b = 0; b < ARES_MAX_BATCH; b++) {
        weight[b] = 0.0f;
        if (b < static_cast<int>(n_rows) && e < n_experts) {
            const float w = picked[static_cast<size_t>(b) * n_experts + e];
            if (w >= 0.0f) {
                rows |= 1u << b;
                weight[b] = w;
            }
        }
    }
    ares_expert_write_groups(rows, weight, slot_of, record, n_experts, n_rows * k, token, rows_of);
}
