// The prediction of a later layer's picks: the routers of the next layers applied to the rows
// a layer's router reads, and the experts each would pick, as a list the host reads ahead of
// those layers' own routers (Pre-gated MoE, FATE, Run My AI).
//
// The logits are the float weight kernel's (mmvq_float.cuh) with a router per row of the
// grid: blockIdx.y names the router, and its logits follow those of the router before. The
// picks are a block per router with a thread per expert: a thread counts, row by row, the
// experts with a larger logit (equal: a lower index), which is the rank the router's own
// selection gives (the softmax keeps the order), and keeps its best rank over the rows. An
// expert is predicted when some row ranks it among the first `k`; the list holds the
// predicted experts by their best rank, then by number.
//
// A list is `token`, the number of experts, then the experts:
//
//   list[0]      the step's token (the host ignores a list of another step)
//   list[1]      n, at most k * n_rows
//   list[2 + i]  the i-th expert
#pragma once

#include "common.cuh"
#include "expert_layout.cuh"
#include "mmvq_float.cuh"

// The logits of two routers (`w1` may equal `w0` when one is wanted: the grid then has one
// row), for the rows of `aq` as ares_expert_logits reads them: y[r][b][e].
extern "C" __global__ void __launch_bounds__(ARES_MMVQ_WARPS * 32) ares_expert_predict_logits(
    const uint8_t* w0, const uint8_t* w1, const int8_t* aq, float* y, uint32_t n_in,
    uint32_t n_out, uint32_t n_batch) {
    const uint8_t* w = blockIdx.y == 0 ? w0 : w1;
    float* out = y + static_cast<size_t>(blockIdx.y) * n_batch * n_out;
    ares_mmvf<AresF32, ARES_MMVQ_WARPS, ARES_MMVQ_ROWS>(w, aq, out, n_in, n_out, n_batch);
}

// The predicted experts of each router: one block per router of n_experts threads (rounded up
// to whole warps). `logits` as ares_expert_predict_logits writes them; `lists` a list of
// `stride` words per router.
ARES_KERNEL ares_expert_predict_pick(
    const float* logits, uint32_t* lists, uint32_t stride, uint32_t n_experts, uint32_t k,
    uint32_t n_rows, uint32_t token) {
    __shared__ float row[ARES_EXPERT_MAX_EXPERTS];
    __shared__ uint32_t best_of[ARES_EXPERT_MAX_EXPERTS];
    __shared__ uint32_t count;
    const uint32_t e = threadIdx.x;
    const bool live = e < n_experts;
    const float* mine = logits + static_cast<size_t>(blockIdx.x) * n_rows * n_experts;
    if (e == 0) {
        count = 0;
    }
    uint32_t best = 0xffffffffu;
    for (uint32_t b = 0; b < n_rows; b++) {
        if (live) {
            row[e] = mine[static_cast<size_t>(b) * n_experts + e];
        }
        __syncthreads();
        if (live) {
            const float v = row[e];
            uint32_t rank = 0;
            for (uint32_t o = 0; o < n_experts; o++) {
                const float other = row[o];
                rank += (other > v || (other == v && o < e)) ? 1u : 0u;
            }
            best = min(best, rank);
        }
        __syncthreads();
    }
    const bool picked = live && best < k;
    if (live) {
        best_of[e] = picked ? best : 0xffffffffu;
    }
    __syncthreads();
    uint32_t* list = lists + static_cast<size_t>(blockIdx.x) * stride;
    if (picked) {
        uint32_t at = 0;
        for (uint32_t o = 0; o < n_experts; o++) {
            const uint32_t other = best_of[o];
            at += (other < best || (other == best && o < e)) ? 1u : 0u;
        }
        list[2 + at] = e;
        atomicAdd(&count, 1u);
    }
    __syncthreads();
    if (e == 0) {
        list[0] = token;
        list[1] = count;
    }
}
