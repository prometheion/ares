// The corrections of a row's logits that come before everything else: the
// entries (a bias, a repetition penalty, a presence or frequency penalty, DRY)
// and the mask. They change the logits in place; the sampler's other kernels
// read the logits as they are afterwards.
#pragma once

#include "sampler_math.cuh"

// Grid (ceil(entries / ARES_SAMPLE_THREADS), rows), ARES_SAMPLE_THREADS
// threads. Applies the entries of each row: `l = l + bias`, then `l = l <= 0 ?
// l * rep : l / rep`, then `l = l - sub` (`Entry::apply` in
// crates/sampler/src/row.rs). The column of a token is `token - first`, or
// `inverse[token]` when the columns are not in the order of the tokens. An
// entry for a token this device does not hold is left alone. The logit the
// kernel finds is written into the entry's `bias`, so that it is known later.
ARES_KERNEL ares_sample_scatter(float* logits, uint32_t stride, uint32_t n_local, uint32_t first,
                                const uint32_t* inverse, uint32_t vocab,
                                const AresSampleParams* params, AresEntry* entries) {
    const uint32_t r = blockIdx.y;
    const AresSampleParams p = params[r];
    const uint32_t i = blockIdx.x * ARES_SAMPLE_THREADS + threadIdx.x;
    if (i >= p.entries_count) {
        return;
    }
    AresEntry* e = entries + p.entries_first + i;
    const uint32_t token = e->token;
    uint32_t col;
    if (inverse != nullptr) {
        col = token < vocab ? inverse[token] : 0xffffffffu;
    } else {
        col = token - first;
    }
    if (col >= n_local) {
        return;
    }
    float* at = logits + (size_t)r * stride + col;
    const float found = *at;
    float l = __fadd_rn(found, e->bias);
    l = l <= 0.0f ? __fmul_rn(l, e->rep) : __fdiv_rn(l, e->rep);
    *at = __fsub_rn(l, e->sub);
    e->bias = found;
}

// Grid (chunks, rows), ARES_SAMPLE_THREADS threads. Makes the logits of the
// tokens a row's mask does not allow minus infinity. The mask is `mask_words`
// words per row, a bit per token, set where the token is allowed.
ARES_KERNEL ares_sample_mask(float* logits, uint32_t stride, uint32_t n_local, const uint32_t* tokens,
                             uint32_t first, uint32_t chunk, const AresSampleParams* params,
                             const uint32_t* masks, uint32_t mask_words) {
    const uint32_t r = blockIdx.y;
    const AresSampleParams p = params[r];
    if (!(p.flags & ARES_SAMPLE_F_MASK)) {
        return;
    }
    const uint32_t* mask = masks + (size_t)p.mask_slot * mask_words;
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    float* row = logits + (size_t)r * stride;
    for (uint32_t i = begin + threadIdx.x; i < end; i += ARES_SAMPLE_THREADS) {
        const uint32_t token = tokens ? tokens[first + i] : first + i;
        const bool allowed = (token >> 5) < mask_words && ((mask[token >> 5] >> (token & 31u)) & 1u);
        if (!allowed) {
            row[i] = -__uint_as_float(0x7f800000u);
        }
    }
}

// Grid (ceil(max(vocab, n_local) / ARES_SAMPLE_THREADS)), ARES_SAMPLE_THREADS
// threads. `inverse[tokens[c]] = c` for the columns `c` of a device that holds
// a token list; the host filled the table with 0xffffffff first.
ARES_KERNEL ares_sample_invert(const uint32_t* tokens, uint32_t n_local, uint32_t* inverse,
                               uint32_t vocab) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_local && tokens[c] < vocab) {
        inverse[tokens[c]] = c;
    }
}
