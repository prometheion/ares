// The best logits of each row: the greedy pick, the reference point of every
// threshold (the best logit) and the candidates a grammar may fall back to.
// The same kernels, with more candidates, find the alternatives a row reports
// probabilities for.
//
// Three kernels per list. `ares_sample_top` finds the best of each chunk of the
// vocabulary, `ares_sample_top_merge` the best of a device's chunks, and when
// several devices each hold a slice of the vocabulary `ares_sample_top_join`
// merges their lists after the devices exchanged them (as a sum: each device
// writes its own slot of the exchange buffer and zeros elsewhere).
#pragma once

#include "sampler_stage.cuh"

// Entries per chunk in the list a top kernel writes for `N` candidates: the
// candidates and one word of flags.
#define ARES_TOP_LIST(N) ((N) + 1)
// f32 values per device and row in the exchange of the candidates: logit and
// token of each, and the status.
#define ARES_TOP_VALUES(N) (2 * (N) + 2)
// A thread of a top kernel sees at most this many logits of its chunk (a chunk
// is 4096 logits, a block 256 threads), so its own list needs no more places.
#define ARES_TOP_MINE 16

// `N` times: the largest proposal of the block, taken from the thread that
// made it. `mine` is the thread's own list in descending order with `n_mine`
// entries. Thread 0 gets the block's list in `out`.
template <int N>
static __device__ __forceinline__ void ares_sample_take_best(const unsigned long long* mine,
                                                             uint32_t n_mine,
                                                             unsigned long long* partial,
                                                             unsigned long long* out) {
    uint32_t next = 0;
    for (int c = 0; c < N; ++c) {
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

// Which tokens a top kernel looks at.
#define ARES_TOP_ALL 0
// The tokens of the set the chain left (the probabilities after the chain).
#define ARES_TOP_SET 1

// Grid (chunks, rows), ARES_SAMPLE_THREADS threads. Chunk `b` of row `r` is
// the logits `b * chunk ..` of the row; its best candidates go to
// `lists[(r * chunks + b) * ARES_TOP_LIST(N) ..]`, packed, best first, then a
// word that is 1 when the chunk has a logit that is NaN or plus infinity.
//
// `tokens` maps a column of the logits to its token (`tokens[first + i]`);
// null means the token is `first + i`.
template <int N, int WHICH>
static __device__ __forceinline__ void ares_sample_top_body(
    const float* logits, uint32_t stride, uint32_t n_local, const uint32_t* tokens, uint32_t first,
    uint32_t chunk, unsigned long long* lists, const AresSampleState* state) {
    __shared__ unsigned long long partial[ARES_SAMPLE_THREADS / 32];
    __shared__ unsigned long long merged[N];
    __shared__ uint32_t bad_any;
    const uint32_t r = blockIdx.y;
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    const float* row = logits + (size_t)r * stride;
    if (threadIdx.x == 0) {
        bad_any = 0;
    }
    __syncthreads();

    constexpr int MINE = N < ARES_TOP_MINE ? N : ARES_TOP_MINE;
    unsigned long long mine[MINE];
    uint32_t n_mine = 0;
    bool bad = false;
    for (uint32_t i = begin + threadIdx.x; i < end; i += ARES_SAMPLE_THREADS) {
        // Adding zero turns a negative zero into a positive one, so that
        // equal logits have equal bits.
        const float l = ares_sample_logit(row[i], &bad);
        if (WHICH == ARES_TOP_SET) {
            if (!ares_sample_present(l) || !ares_in_set(state[r], ares_sample_key(state[r].l_max, l))) {
                continue;
            }
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
        if (n_mine < MINE) {
            mine[n_mine++] = v;
        }
    }
    if (bad) {
        bad_any = 1;
    }
    ares_sample_take_best<N>(mine, n_mine, partial, merged);
    if (threadIdx.x == 0) {
        unsigned long long* out = lists + ((size_t)r * gridDim.x + blockIdx.x) * ARES_TOP_LIST(N);
        for (int c = 0; c < N; ++c) {
            out[c] = merged[c];
        }
        out[N] = bad_any;
    }
}

ARES_KERNEL ares_sample_top(const float* logits, uint32_t stride, uint32_t n_local,
                            const uint32_t* tokens, uint32_t first, uint32_t chunk,
                            unsigned long long* lists) {
    ares_sample_top_body<ARES_SAMPLE_CANDIDATES, ARES_TOP_ALL>(logits, stride, n_local, tokens, first,
                                                               chunk, lists, nullptr);
}

// The same for the rows that report the model's own probabilities, on the
// logits as the model gave them: 20 candidates.
ARES_KERNEL ares_lp_top_raw(const float* logits, uint32_t stride, uint32_t n_local,
                            const uint32_t* tokens, uint32_t first, uint32_t chunk,
                            unsigned long long* lists, const AresSampleParams* params) {
    if (!(params[blockIdx.y].flags & ARES_SAMPLE_F_LP_RAW)) {
        return;
    }
    ares_sample_top_body<ARES_LOGPROBS_MAX, ARES_TOP_ALL>(logits, stride, n_local, tokens, first, chunk,
                                                          lists, nullptr);
}

// And for the rows that report the probabilities after the chain: the
// candidates of the set that is left.
ARES_KERNEL ares_lp_top_post(const float* logits, uint32_t stride, uint32_t n_local,
                             const uint32_t* tokens, uint32_t first, uint32_t chunk,
                             unsigned long long* lists, const AresSampleParams* params,
                             const AresSampleState* state) {
    const uint32_t r = blockIdx.y;
    if (!(params[r].flags & ARES_SAMPLE_F_LP_POST) || (state[r].status & ARES_SAMPLE_S_EMPTY)) {
        return;
    }
    ares_sample_top_body<ARES_LOGPROBS_MAX, ARES_TOP_SET>(logits, stride, n_local, tokens, first, chunk,
                                                          lists, state);
}

// The best `N` candidates of the chunks of row `r` in `lists`, and whether
// any chunk saw a logit that is not a number. Run by every thread of the
// block, which has at least as many threads as chunks.
template <int N>
static __device__ __forceinline__ void ares_sample_merge_chunks(
    const unsigned long long* lists, uint32_t r, uint32_t chunks, unsigned long long* partial,
    unsigned long long* merged, uint32_t* bad_any) {
    const unsigned long long* list =
        threadIdx.x < chunks ? lists + ((size_t)r * chunks + threadIdx.x) * ARES_TOP_LIST(N) : nullptr;
    if (list != nullptr && list[N] != 0ull) {
        *bad_any = 1;
    }
    uint32_t taken = 0;
    for (int c = 0; c < N; ++c) {
        const unsigned long long proposal = (list != nullptr && taken < (uint32_t)N) ? list[taken] : 0ull;
        const unsigned long long best = ares_block_max_u64(proposal, partial);
        if (best != 0ull && proposal == best) {
            taken += 1;
        }
        if (threadIdx.x == 0) {
            merged[c] = best;
        }
        __syncthreads();
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
    const bool sampled = (params->flags & ARES_SAMPLE_F_SAMPLE) != 0;
    state->l_max = l_max;
    state->status = status;
    state->lo = 0;
    state->hi = sampled ? ARES_SAMPLE_KEY_MAX : 0u;
    state->inv_t = 1.0f;
    state->cursor = 0;
    state->k = params->top_k;
    state->prefix = 0;
    state->rem = 0;
    state->cut = 0xffffffffu;
    state->cut100 = 0xffffffffu;
    state->target = 0;
    state->best = 0ull;
    state->n_gather = 0;
    out->token = sampled ? 0xffffffffu : out->cand_token[0];
    out->value = l_max;
    out->status = status;
    out->aux = 0.0f;
    if (sampled && !(status & ARES_SAMPLE_S_EMPTY)) {
        ares_sample_advance(*params, state);
    }
}

// Grid (rows), ARES_SAMPLE_THREADS threads, `chunks` at most that many. With
// one device the row is finished here; with several, this device's list goes
// into its slot of `exchange` (ARES_TOP_VALUES values per device and row) and
// zeros into the other slots, for the sum across devices.
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
    ares_sample_merge_chunks<ARES_SAMPLE_CANDIDATES>(lists, r, chunks, partial, merged, &bad_any);
    if (threadIdx.x != 0) {
        return;
    }
    if (devices == 1) {
        ares_sample_join(merged, bad_any, params + r, state + r, out + r);
        return;
    }
    float* slots = exchange + (size_t)r * devices * ARES_TOP_VALUES(ARES_SAMPLE_CANDIDATES);
    for (uint32_t d = 0; d < devices; ++d) {
        float* slot = slots + d * ARES_TOP_VALUES(ARES_SAMPLE_CANDIDATES);
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

// The lists of all devices, merged: the best `N` of them. One thread.
template <int N>
static __device__ __forceinline__ void ares_sample_join_lists(const float* slots, uint32_t devices,
                                                              unsigned long long* best,
                                                              uint32_t* bad) {
    for (int c = 0; c < N; ++c) {
        best[c] = 0ull;
    }
    *bad = 0;
    for (uint32_t d = 0; d < devices; ++d) {
        const float* slot = slots + d * ARES_TOP_VALUES(N);
        *bad |= slot[2 * N] != 0.0f;
        for (int c = 0; c < N; ++c) {
            if (!(slot[2 * c + 1] >= 0.0f)) {
                continue;
            }
            unsigned long long v = ares_sample_pack(slot[2 * c], (uint32_t)slot[2 * c + 1]);
            for (int j = 0; j < N; ++j) {
                if (v > best[j]) {
                    const unsigned long long t = best[j];
                    best[j] = v;
                    v = t;
                }
            }
        }
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
    unsigned long long best[ARES_SAMPLE_CANDIDATES];
    uint32_t bad;
    ares_sample_join_lists<ARES_SAMPLE_CANDIDATES>(
        exchange + (size_t)r * devices * ARES_TOP_VALUES(ARES_SAMPLE_CANDIDATES), devices, best, &bad);
    ares_sample_join(best, bad, params + r, state + r, out + r);
}

// The list of a row that reports probabilities is known: its tokens and
// logits go to the record, as many as the row asks for, and the best logit to
// the state when it is the model's own (`raw`).
static __device__ void ares_lp_take(const unsigned long long* best, const AresSampleParams& p,
                                    AresSampleState* state, AresLogprobOut* out, bool raw) {
    uint32_t n = 0;
    const uint32_t want = min(p.logprobs, (uint32_t)ARES_LOGPROBS_MAX);
    for (uint32_t c = 0; c < want; ++c) {
        if (best[c] == 0ull || !ares_sample_present(ares_sample_unsortable((uint32_t)(best[c] >> 32)))) {
            break;
        }
        out->token[n] = 0xffffffffu - (uint32_t)best[c];
        // The logit; the finishing kernel turns it into a probability.
        out->logprob[n] = ares_sample_unsortable((uint32_t)(best[c] >> 32));
        n += 1;
    }
    out->n = n;
    if (raw) {
        state->raw_lmax = best[0] == 0ull ? -__uint_as_float(0x7f800000u)
                                          : ares_sample_unsortable((uint32_t)(best[0] >> 32));
    }
}

// Whether a row takes part in the list of the probabilities of kind `raw`.
static __device__ __forceinline__ bool ares_lp_row(const AresSampleParams& p, const AresSampleState& s,
                                                   bool raw) {
    return raw ? (p.flags & ARES_SAMPLE_F_LP_RAW) != 0
               : ((p.flags & ARES_SAMPLE_F_LP_POST) != 0 && !(s.status & ARES_SAMPLE_S_EMPTY));
}

// Grid (rows), ARES_SAMPLE_THREADS threads: like ares_sample_top_merge for the
// 20 candidates of a row that reports probabilities. `raw` picks the kind of
// row (the model's own probabilities, or those after the chain).
ARES_KERNEL ares_lp_top_merge(const unsigned long long* lists, uint32_t chunks, uint32_t devices,
                              uint32_t rank, float* exchange, const AresSampleParams* params,
                              AresSampleState* state, AresLogprobOut* out, uint32_t raw) {
    __shared__ unsigned long long partial[ARES_SAMPLE_THREADS / 32];
    __shared__ unsigned long long merged[ARES_LOGPROBS_MAX];
    __shared__ uint32_t bad_any;
    const uint32_t r = blockIdx.x;
    if (!ares_lp_row(params[r], state[r], raw != 0)) {
        return;
    }
    if (threadIdx.x == 0) {
        bad_any = 0;
    }
    __syncthreads();
    ares_sample_merge_chunks<ARES_LOGPROBS_MAX>(lists, r, chunks, partial, merged, &bad_any);
    if (threadIdx.x != 0) {
        return;
    }
    if (devices == 1) {
        ares_lp_take(merged, params[r], state + r, out + r, raw != 0);
        return;
    }
    float* slots = exchange + (size_t)r * devices * ARES_TOP_VALUES(ARES_LOGPROBS_MAX);
    for (uint32_t d = 0; d < devices; ++d) {
        float* slot = slots + d * ARES_TOP_VALUES(ARES_LOGPROBS_MAX);
        for (int c = 0; c < ARES_LOGPROBS_MAX; ++c) {
            const bool have = d == rank && merged[c] != 0ull;
            slot[2 * c] = have ? ares_sample_unsortable((uint32_t)(merged[c] >> 32)) : 0.0f;
            slot[2 * c + 1] =
                d != rank ? 0.0f : (have ? (float)(0xffffffffu - (uint32_t)merged[c]) : -1.0f);
        }
        slot[2 * ARES_LOGPROBS_MAX] = 0.0f;
        slot[2 * ARES_LOGPROBS_MAX + 1] = 0.0f;
    }
}

// Grid (rows), 32 threads: merges the lists of all devices after the sum.
ARES_KERNEL ares_lp_top_join(const float* exchange, uint32_t devices, const AresSampleParams* params,
                             AresSampleState* state, AresLogprobOut* out, uint32_t raw) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    if (!ares_lp_row(params[r], state[r], raw != 0)) {
        return;
    }
    unsigned long long best[ARES_LOGPROBS_MAX];
    uint32_t bad;
    ares_sample_join_lists<ARES_LOGPROBS_MAX>(
        exchange + (size_t)r * devices * ARES_TOP_VALUES(ARES_LOGPROBS_MAX), devices, best, &bad);
    ares_lp_take(best, params[r], state + r, out + r, raw != 0);
}
