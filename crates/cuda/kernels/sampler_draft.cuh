// What drafts that stay on the device need besides the block that computes
// them: a draft chain's picks go straight into the records the next kernels
// read (the next link's token, the verify window's token, the list the host
// reads at the end of the step), and the hidden state a chain starts from is
// kept per sequence from step to step.
//
// The "carry" of a sequence is `carry_rows` rows of `d` values. After a step
// it holds the final hidden state of the sequence's rows of that step: of the
// last row in row 0 when all rows were certain, of every row of a window that
// verified drafts otherwise; the commit of the window then moves the row of
// the last accepted token to row 0. So before a step's kernels, row 0 is the
// hidden state the sequence's newest token was picked from.
#pragma once

#include "layer_step.cuh"
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

// Grid (max_seg_rows, n_segs): for every row of the step the final hidden
// state of the position before it: of the row before in its segment, or the
// sequence's carry for a segment's first row (zeros at position 0). `hidden`
// has the final hidden state of every row of the step.
ARES_KERNEL ares_draft_shift(const AresStep* step, const float* hidden, const float* carry,
                             uint32_t carry_rows, float* out, uint32_t d) {
    const AresSeg seg = ares_step_segs(step)[blockIdx.y];
    const uint32_t i = blockIdx.x;
    if (i >= seg.n_rows) {
        return;
    }
    const uint32_t r = seg.row0 + i;
    const float* from = i == 0 ? carry + (size_t)seg.slot * carry_rows * d : hidden + (size_t)(r - 1) * d;
    const bool none = i == 0 && seg.pos0 == 0;
    float* to = out + (size_t)r * d;
    for (uint32_t j = threadIdx.x; j < d; j += blockDim.x) {
        to[j] = none ? 0.0f : from[j];
    }
}

// Grid (max_seg_rows, n_segs): keeps the final hidden state of the step's
// rows in the carries (see the top of this file).
ARES_KERNEL ares_draft_keep(const AresStep* step, const float* hidden, float* carry,
                            uint32_t carry_rows, uint32_t d) {
    const AresSeg seg = ares_step_segs(step)[blockIdx.y];
    const uint32_t i = blockIdx.x;
    if (i >= seg.n_rows) {
        return;
    }
    uint32_t at;
    if (seg.flags & ARES_SEG_DIRECT) {
        if (i + 1 != seg.n_rows) {
            return;
        }
        at = 0;
    } else {
        if (i >= carry_rows) {
            return;
        }
        at = i;
    }
    const float* from = hidden + (size_t)(seg.row0 + i) * d;
    float* to = carry + ((size_t)seg.slot * carry_rows + at) * d;
    for (uint32_t j = threadIdx.x; j < d; j += blockDim.x) {
        to[j] = from[j];
    }
}

// Grid (n_segs) of a commit record: the hidden state of the last accepted row
// of each window becomes row 0 of its sequence's carry.
ARES_KERNEL ares_draft_commit(const AresStep* commit, float* carry, uint32_t carry_rows, uint32_t d) {
    const AresSeg seg = ares_step_segs(commit)[blockIdx.x];
    if ((seg.flags & ARES_SEG_DIRECT) || seg.accepted < 2 || seg.accepted > carry_rows) {
        return;
    }
    float* base = carry + (size_t)seg.slot * carry_rows * d;
    const float* from = base + (size_t)(seg.accepted - 1) * d;
    for (uint32_t j = threadIdx.x; j < d; j += blockDim.x) {
        base[j] = from[j];
    }
}
