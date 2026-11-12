// The last stage of the many-row expert kernels with the ordered sum (expertrows_down.cuh).
#pragma once

#include "expertrows_layout.cuh"

// The sum of the experts' outputs of each row in the order of the router, added to the
// row of `out` (the Ordered way). A block per row.
ARES_KERNEL ares_er_combine(
    uint8_t* plan_bytes, const float* scratch, float* out, uint32_t n_experts, uint32_t n_pairs, uint32_t tile, uint32_t top_k,
    uint32_t n_embd) {
    const AresErPlan plan(plan_bytes, n_experts, n_pairs, tile);
    const uint32_t row = blockIdx.x;
    const uint32_t* pos = plan.pair_pos() + static_cast<size_t>(row) * top_k;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < top_k; j++) {
            const uint32_t p = pos[j];
            if (p != ARES_ER_NO_POS) {
                sum += scratch[static_cast<size_t>(p) * n_embd + e];
            }
        }
        out[static_cast<size_t>(row) * n_embd + e] += sum;
    }
}
