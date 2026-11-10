// The best logits of each row: the greedy pick, the reference point of every
// threshold (the best logit) and the candidates a grammar may fall back to.
//
// Three kernels. `ares_sample_top` finds the best of each chunk of the
// vocabulary, `ares_sample_top_merge` the best of a device's chunks, and when
// several devices each hold a slice of the vocabulary `ares_sample_top_join`
// merges their lists after the devices exchanged them (as a sum: each device
// writes its own slot of the exchange buffer and zeros elsewhere).
#pragma once

#include "sampler_math.cuh"

// Entries per chunk in the list `ares_sample_top` writes: the candidates and
// one word of flags.
#define ARES_SAMPLE_LIST (ARES_SAMPLE_CANDIDATES + 1)
// f32 values per device and row in the exchange of the candidates: logit and
// token of each, and the status.
#define ARES_SAMPLE_TOP_VALUES (2 * ARES_SAMPLE_CANDIDATES + 2)

// `rounds` times: the largest proposal of the block, taken from the thread
// that made it. `mine` is the thread's own list in descending order with
// `n_mine` entries. Thread 0 gets the block's list in `out`.
static __device__ __forceinline__ void ares_sample_take_best(const unsigned long long* mine,
                                                             uint32_t n_mine,
                                                             unsigned long long* partial,
                                                             unsigned long long* out) {
    uint32_t next = 0;
    for (int c = 0; c < ARES_SAMPLE_CANDIDATES; ++c) {
        const unsigned long long proposal = next < n_mine ? mine[next] : 0ull;
        const unsigned long long best = ares_block_max_u64(proposal, partial);
        if (best != 0ull && proposal == best) {
            next += 1;
        }
        if (threadIdx.x == 0) {
            out[c] = best;
        }
        __syncthreads();
    }
}

// Grid (chunks, rows), ARES_SAMPLE_THREADS threads. Chunk `b` of row `r` is
// the logits `b * chunk ..` of the row; its best candidates go to
// `lists[(r * chunks + b) * ARES_SAMPLE_LIST ..]`, packed, best first, then a
// word that is 1 when the chunk has a logit that is NaN or plus infinity.
//
// `tokens` maps a column of the logits to its token (`tokens[first + i]`);
// null means the token is `first + i`.
ARES_KERNEL ares_sample_top(const float* logits, uint32_t stride, uint32_t n_local,
                            const uint32_t* tokens, uint32_t first, uint32_t chunk,
                            unsigned long long* lists) {
    __shared__ unsigned long long partial[ARES_SAMPLE_THREADS / 32];
    __shared__ unsigned long long merged[ARES_SAMPLE_CANDIDATES];
    __shared__ uint32_t bad_any;
    const uint32_t r = blockIdx.y;
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    const float* row = logits + (size_t)r * stride;
    if (threadIdx.x == 0) {
        bad_any = 0;
    }
    __syncthreads();

    unsigned long long mine[ARES_SAMPLE_CANDIDATES];
    uint32_t n_mine = 0;
    bool bad = false;
    for (uint32_t i = begin + threadIdx.x; i < end; i += ARES_SAMPLE_THREADS) {
        // Adding zero turns a negative zero into a positive one, so that
        // equal logits have equal bits.
        float l = __fadd_rn(row[i], 0.0f);
        if (!(l <= 3.402823466e+38f)) {
            bad = true;
            l = -__uint_as_float(0x7f800000u);
        }
        const uint32_t token = tokens ? tokens[first + i] : first + i;
        unsigned long long v = ares_sample_pack(l, token);
        // Insert into the descending list; the smallest falls off the end.
        for (uint32_t j = 0; j < n_mine; ++j) {
            if (v > mine[j]) {
                const unsigned long long t = mine[j];
                mine[j] = v;
                v = t;
            }
        }
        if (n_mine < ARES_SAMPLE_CANDIDATES) {
            mine[n_mine++] = v;
        }
    }
    if (bad) {
        bad_any = 1;
    }
    ares_sample_take_best(mine, n_mine, partial, merged);
    if (threadIdx.x == 0) {
        unsigned long long* out = lists + ((size_t)r * gridDim.x + blockIdx.x) * ARES_SAMPLE_LIST;
        for (int c = 0; c < ARES_SAMPLE_CANDIDATES; ++c) {
            out[c] = merged[c];
        }
        out[ARES_SAMPLE_CANDIDATES] = bad_any;
    }
}

// A row's candidates are known: sets up its state and, for a greedy row,
// writes the pick. `best` is the row's list over all devices, best first.
static __device__ void ares_sample_join(const unsigned long long* best, uint32_t bad,
                                        const AresSampleParams* params, AresSampleState* state,
                                        AresSampleOut* out) {
    uint32_t status = bad ? ARES_SAMPLE_S_NONFINITE : 0u;
    for (int c = 0; c < ARES_SAMPLE_CANDIDATES; ++c) {
        const bool have = best[c] != 0ull;
        out->cand_token[c] = have ? 0xffffffffu - (uint32_t)best[c] : 0xffffffffu;
        out->cand_logit[c] =
            have ? ares_sample_unsortable((uint32_t)(best[c] >> 32)) : -__uint_as_float(0x7f800000u);
    }
    const float l_max = out->cand_logit[0];
    if (!(l_max > -3.402823466e+38f)) {
        // No token has a finite logit.
        status |= ARES_SAMPLE_S_EMPTY;
    }
    state->l_max = l_max;
    state->status = status;
    state->prefix_k = 0;
    state->rem_k = params->top_k;
    state->z_k = 0;
    state->prefix_p = 0;
    state->cut_key = min(params->min_p_key, ARES_SAMPLE_KEY_MAX);
    state->rem_p = 0;
    state->best = 0ull;
    const bool sampled = (params->flags & ARES_SAMPLE_F_SAMPLE) != 0;
    out->token = sampled ? 0xffffffffu : out->cand_token[0];
    out->value = l_max;
    out->status = status;
    out->reserved = 0;
}

// Grid (rows), ARES_SAMPLE_THREADS threads, `chunks` at most that many. With
// one device the row is finished here; with several, this device's list goes
// into its slot of `exchange` (ARES_SAMPLE_TOP_VALUES values per device and
// row) and zeros into the other slots, for the sum across devices.
ARES_KERNEL ares_sample_top_merge(const unsigned long long* lists, uint32_t chunks,
                                  uint32_t devices, uint32_t rank, float* exchange,
                                  const AresSampleParams* params, AresSampleState* state,
                                  AresSampleOut* out) {
    __shared__ unsigned long long partial[ARES_SAMPLE_THREADS / 32];
    __shared__ unsigned long long merged[ARES_SAMPLE_CANDIDATES];
    __shared__ uint32_t bad_any;
    const uint32_t r = blockIdx.x;
    if (threadIdx.x == 0) {
        bad_any = 0;
    }
    __syncthreads();
    unsigned long long mine[ARES_SAMPLE_CANDIDATES];
    uint32_t n_mine = 0;
    if (threadIdx.x < chunks) {
        const unsigned long long* list = lists + ((size_t)r * chunks + threadIdx.x) * ARES_SAMPLE_LIST;
        for (int c = 0; c < ARES_SAMPLE_CANDIDATES; ++c) {
            mine[c] = list[c];
            n_mine += list[c] != 0ull;
        }
        if (list[ARES_SAMPLE_CANDIDATES] != 0ull) {
            bad_any = 1;
        }
    }
    ares_sample_take_best(mine, n_mine, partial, merged);
    if (threadIdx.x != 0) {
        return;
    }
    if (devices == 1) {
        ares_sample_join(merged, bad_any, params + r, state + r, out + r);
        return;
    }
    float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_TOP_VALUES;
    for (uint32_t d = 0; d < devices; ++d) {
        float* slot = slots + d * ARES_SAMPLE_TOP_VALUES;
        for (int c = 0; c < ARES_SAMPLE_CANDIDATES; ++c) {
            const bool have = d == rank && merged[c] != 0ull;
            // A token as a float is exact below 2^24; -1 marks "none". The
            // other devices' slots hold zeros, so the sum leaves this one.
            slot[2 * c] = have ? ares_sample_unsortable((uint32_t)(merged[c] >> 32)) : 0.0f;
            slot[2 * c + 1] =
                d != rank ? 0.0f : (have ? (float)(0xffffffffu - (uint32_t)merged[c]) : -1.0f);
        }
        slot[2 * ARES_SAMPLE_CANDIDATES] = d == rank ? (float)bad_any : 0.0f;
        slot[2 * ARES_SAMPLE_CANDIDATES + 1] = 0.0f;
    }
}

// Grid (rows), 32 threads: merges the lists of all devices after the sum.
ARES_KERNEL ares_sample_top_join(const float* exchange, uint32_t devices,
                                 const AresSampleParams* params, AresSampleState* state,
                                 AresSampleOut* out) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    const float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_TOP_VALUES;
    unsigned long long best[ARES_SAMPLE_CANDIDATES];
    for (int c = 0; c < ARES_SAMPLE_CANDIDATES; ++c) {
        best[c] = 0ull;
    }
    uint32_t bad = 0;
    for (uint32_t d = 0; d < devices; ++d) {
        const float* slot = slots + d * ARES_SAMPLE_TOP_VALUES;
        bad |= slot[2 * ARES_SAMPLE_CANDIDATES] != 0.0f;
        for (int c = 0; c < ARES_SAMPLE_CANDIDATES; ++c) {
            if (!(slot[2 * c + 1] >= 0.0f)) {
                continue;
            }
            unsigned long long v = ares_sample_pack(slot[2 * c], (uint32_t)slot[2 * c + 1]);
            for (int j = 0; j < ARES_SAMPLE_CANDIDATES; ++j) {
                if (v > best[j]) {
                    const unsigned long long t = best[j];
                    best[j] = v;
                    v = t;
                }
            }
        }
    }
    ares_sample_join(best, bad, params + r, state + r, out + r);
}
