// What drafts that stay on the device need besides the block that computes
// them: a draft chain's picks go straight into the records the next kernels
// read (the next link's token, the verify window's token, the list the host
// reads at the end of the step), and the rows of hidden state a chain starts
// from are gathered from those of the step.
#pragma once

#include "sampler_math.cuh"

// One block. The tokens of `n` picks: pick `c` goes to `next_tokens[c]` (for
// `c < n_next`; the token array of the next link's record), to
// `record_tokens[record_rows[c]]` (the step record's token of the row that
// verifies it) and to `drafts[draft_at[c]]` (what the host reads).
ARES_KERNEL ares_draft_put(const AresSampleOut* picks, uint32_t n, uint32_t* next_tokens,
                           uint32_t n_next, uint32_t* record_tokens, const uint32_t* record_rows,
                           uint32_t* drafts, const uint32_t* draft_at) {
    for (uint32_t c = threadIdx.x; c < n; c += blockDim.x) {
        const uint32_t token = picks[c].token;
        if (c < n_next) {
            next_tokens[c] = token;
        }
        record_tokens[record_rows[c]] = token;
        drafts[draft_at[c]] = token;
    }
}

// Grid (rows): `dst[r] = src[index[r]]`, rows of `d` values.
ARES_KERNEL ares_draft_rows(float* dst, const float* src, const uint32_t* index, uint32_t d) {
    const float* from = src + (size_t)index[blockIdx.x] * d;
    float* to = dst + (size_t)blockIdx.x * d;
    for (uint32_t j = threadIdx.x; j < d; j += blockDim.x) {
        to[j] = from[j];
    }
}
