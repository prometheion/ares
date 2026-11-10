// The pick of a sampled row: among the tokens the cuts left, the one with
// the largest `logit / temperature + noise`, where the noise is the Gumbel
// number of (seed, position, token). That is a draw from the distribution of
// those tokens at that temperature, and because a token's noise depends on
// nothing else, the same row gives the same pick alone or in any batch, on
// one device or on several.
#pragma once

#include "sampler_math.cuh"

// f32 values per device and row in the exchange of the picks.
#define ARES_SAMPLE_PICK_VALUES 2

// Grid (chunks, rows), ARES_SAMPLE_THREADS threads. Raises `state[r].best`
// to the best packed score of this chunk.
ARES_KERNEL ares_sample_pick(const float* logits, uint32_t stride, uint32_t n_local,
                             const uint32_t* tokens, uint32_t first, uint32_t chunk,
                             const AresSampleParams* params, AresSampleState* state) {
    __shared__ unsigned long long partial[ARES_SAMPLE_THREADS / 32];
    const uint32_t r = blockIdx.y;
    const AresSampleParams p = params[r];
    if (!(p.flags & ARES_SAMPLE_F_SAMPLE)) {
        return;
    }
    const float l_max = state[r].l_max;
    const uint32_t cut = state[r].cut_key;
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    const float* row = logits + (size_t)r * stride;
    unsigned long long best = 0ull;
    for (uint32_t i = begin + threadIdx.x; i < end; i += ARES_SAMPLE_THREADS) {
        const float l = __fadd_rn(row[i], 0.0f);
        if (ares_sample_key(l_max, l) > cut) {
            continue;
        }
        const uint32_t token = tokens ? tokens[first + i] : first + i;
        const float g = ares_sample_gumbel(p.seed_lo, p.seed_hi, p.position, token);
        const float score = __fadd_rn(__fmul_rn(l, p.inv_temperature), g);
        const unsigned long long v = ares_sample_pack(score, token);
        best = v > best ? v : best;
    }
    best = ares_block_max_u64(best, partial);
    if (threadIdx.x == 0 && best != 0ull) {
        atomicMax(&state[r].best, best);
    }
}

// Grid (rows), 32 threads, several devices only: this device's best into its
// slot of `exchange` (ARES_SAMPLE_PICK_VALUES values per device and row: the
// score, and the token or -1 for none), zeros into the other slots.
ARES_KERNEL ares_sample_pick_export(const AresSampleState* state, const AresSampleParams* params,
                                    float* exchange, uint32_t devices, uint32_t rank) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    if (!(params[r].flags & ARES_SAMPLE_F_SAMPLE)) {
        return;
    }
    const unsigned long long best = state[r].best;
    float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_PICK_VALUES;
    for (uint32_t d = 0; d < devices; ++d) {
        const bool mine = d == rank;
        slots[d * ARES_SAMPLE_PICK_VALUES] =
            mine && best != 0ull ? ares_sample_unsortable((uint32_t)(best >> 32)) : 0.0f;
        slots[d * ARES_SAMPLE_PICK_VALUES + 1] =
            mine ? (best != 0ull ? (float)(0xffffffffu - (uint32_t)best) : -1.0f) : 0.0f;
    }
}

// Grid (rows), 32 threads: the pick of a sampled row into its record.
ARES_KERNEL ares_sample_finish(const AresSampleState* state, const float* exchange, uint32_t devices,
                               const AresSampleParams* params, AresSampleOut* out) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    if (!(params[r].flags & ARES_SAMPLE_F_SAMPLE)) {
        return;
    }
    unsigned long long best = 0ull;
    if (devices == 1) {
        best = state[r].best;
    } else {
        const float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_PICK_VALUES;
        for (uint32_t d = 0; d < devices; ++d) {
            const float token = slots[d * ARES_SAMPLE_PICK_VALUES + 1];
            if (token >= 0.0f) {
                const unsigned long long v =
                    ares_sample_pack(slots[d * ARES_SAMPLE_PICK_VALUES], (uint32_t)token);
                best = v > best ? v : best;
            }
        }
    }
    if (best == 0ull) {
        out[r].token = 0xffffffffu;
        out[r].status = state[r].status | ARES_SAMPLE_S_EMPTY;
        return;
    }
    out[r].token = 0xffffffffu - (uint32_t)best;
    out[r].value = ares_sample_unsortable((uint32_t)(best >> 32));
    out[r].status = state[r].status;
}
