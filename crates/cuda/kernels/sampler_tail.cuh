// What a row reports after its pick: the probabilities of the pick and of the
// best alternatives, and for mirostat the surprise of the pick.
//
// The probability of a token is `exp(-x) / sum`, with `x` the token's scaled
// distance below the best token of the set and `sum` the weights of the set,
// so its logarithm is `-x - ln(sum)`, worked out in double precision from the
// integer sum (ares_ln_fixed). For the model's own probabilities the set is
// every token, at temperature 1, from the best logit the model gave.
#pragma once

#include "sampler_pick.cuh"

// Grid (rows), 32 threads. The logit of a row's pick on the device that holds
// its column, into this device's slot of `exchange`
// (ARES_SAMPLE_PASS_VALUES values per device and row: whether it is here, the
// logit after the corrections, the logit before them) and zeros elsewhere.
ARES_KERNEL ares_lp_pick_logit(const float* logits, uint32_t stride, uint32_t n_local, uint32_t first,
                               const uint32_t* inverse, uint32_t vocab,
                               const AresSampleParams* params, const AresEntry* entries,
                               const AresSampleOut* out, float* exchange, uint32_t devices,
                               uint32_t rank) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    const AresSampleParams p = params[r];
    if (!(p.flags & (ARES_SAMPLE_F_LP_RAW | ARES_SAMPLE_F_LP_POST | ARES_SAMPLE_F_MIROSTAT))) {
        return;
    }
    const uint32_t token = out[r].token;
    float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_PASS_VALUES;
    for (uint32_t d = 0; d < devices; ++d) {
        for (int k = 0; k < 3; ++k) {
            slots[d * ARES_SAMPLE_PASS_VALUES + k] = 0.0f;
        }
    }
    if (token == 0xffffffffu) {
        return;
    }
    uint32_t col;
    if (inverse != nullptr) {
        col = token < vocab ? inverse[token] : 0xffffffffu;
    } else {
        col = token - first;
    }
    if (col >= n_local) {
        return;
    }
    bool bad = false;
    const float corrected = ares_sample_logit(logits[(size_t)r * stride + col], &bad);
    float raw = corrected;
    for (uint32_t i = 0; i < p.entries_count; ++i) {
        const AresEntry e = entries[p.entries_first + i];
        if (e.token == token) {
            raw = ares_sample_logit(e.bias, &bad);
            break;
        }
    }
    float* slot = slots + rank * ARES_SAMPLE_PASS_VALUES;
    slot[0] = 1.0f;
    slot[1] = corrected;
    slot[2] = raw;
}

// The logarithm of the probability of a token at distance `x`.
static __device__ __forceinline__ float ares_log_probability(float x, double ln_sum) {
    return __double2float_rn(__dsub_rn(-(double)x, ln_sum));
}

// Grid (rows), 32 threads. The pick's logit from all devices, then the
// probabilities and the surprise.
ARES_KERNEL ares_lp_finish(const float* exchange, uint32_t devices, const AresSampleParams* params,
                           AresSampleState* state, AresSampleOut* out, AresLogprobOut* lp) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    const AresSampleParams p = params[r];
    AresSampleState* s = state + r;
    if (!(p.flags & (ARES_SAMPLE_F_LP_RAW | ARES_SAMPLE_F_LP_POST | ARES_SAMPLE_F_MIROSTAT))) {
        return;
    }
    const bool raw_mode = (p.flags & ARES_SAMPLE_F_LP_RAW) != 0;
    const bool post_mode = (p.flags & ARES_SAMPLE_F_LP_POST) != 0;
    const bool has_token = out[r].token != 0xffffffffu;
    float corrected = 0.0f, raw = 0.0f;
    bool found = false;
    const float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_PASS_VALUES;
    for (uint32_t d = 0; d < devices && has_token; ++d) {
        const float* slot = slots + d * ARES_SAMPLE_PASS_VALUES;
        if (slot[0] != 0.0f) {
            found = true;
            corrected = slot[1];
            raw = slot[2];
        }
    }
    const double lnz_set = (!(s->status & ARES_SAMPLE_S_EMPTY) && (post_mode || (p.flags & ARES_SAMPLE_F_MIROSTAT)))
                               ? ares_ln_fixed(s->sums[1])
                               : 0.0;
    if (p.flags & ARES_SAMPLE_F_MIROSTAT) {
        if (found) {
            float d, x;
            ares_sample_x(*s, ares_sample_key(s->l_max, corrected), &d, &x);
            out[r].aux = __double2float_rn(
                __ddiv_rn(__dadd_rn(lnz_set, (double)x), 0.693147180559945309417232));
        }
    }
    if (!raw_mode && !post_mode) {
        return;
    }
    lp[r].chosen = -__uint_as_float(0x7f800000u);
    if (post_mode) {
        for (uint32_t c = 0; c < lp[r].n; ++c) {
            float d, x;
            ares_sample_x(*s, ares_sample_key(s->l_max, lp[r].logprob[c]), &d, &x);
            lp[r].logprob[c] = ares_log_probability(x, lnz_set);
        }
        if (found) {
            float d, x;
            ares_sample_x(*s, ares_sample_key(s->l_max, corrected), &d, &x);
            lp[r].chosen = ares_log_probability(x, lnz_set);
        }
    } else {
        for (uint32_t c = 0; c < lp[r].n; ++c) {
            const float d = __fsub_rn(s->raw_lmax, lp[r].logprob[c]);
            lp[r].logprob[c] = ares_log_probability(d > 0.0f ? d : 0.0f, s->raw_lnz);
        }
        // A token without a finite logit has no probability: minus infinity
        // stays (the distance from a row's best would be NaN when no token
        // has one).
        if (found && ares_sample_present(raw)) {
            const float d = __fsub_rn(s->raw_lmax, raw);
            lp[r].chosen = ares_log_probability(d > 0.0f ? d : 0.0f, s->raw_lnz);
        }
    }
}
