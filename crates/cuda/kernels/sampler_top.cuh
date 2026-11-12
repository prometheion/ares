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
// The logits of its chunk a thread of a top kernel reads: a chunk is at most
// this many logits per thread of the block. They stay in the thread's
// registers, so nothing is read twice.
#define ARES_TOP_PER 16

// Which tokens a top kernel looks at.
#define ARES_TOP_ALL 0
// The tokens of the set the chain left (the probabilities after the chain).
#define ARES_TOP_SET 1

// Grid (chunks, rows), up to ARES_SAMPLE_THREADS threads (a multiple of 32, at
// least `N`), `chunk` at most ARES_TOP_PER logits per thread. Chunk `b` of row
// `r` is the logits `b * chunk ..` of the row; its best candidates go to
// `lists[(r * chunks + b) * ARES_TOP_LIST(N) ..]`, packed, best first, zero
// where the chunk has fewer, then a word that is 1 when the chunk has a logit
// that is NaN or plus infinity.
//
// `tokens` maps a column of the logits to its token (`tokens[first + i]`);
// null means the token is `first + i`.
//
// Most logits are far below the best, so the kernel first finds a bar that
// the `N` best of the chunk reach, from one number per thread, and only then
// orders the few logits that reach it:
//
// 1. Every thread finds its own best. A column is the threads with the same
//    lane; the bar is the `N`th best of the 32 columns' bests. `N` different
//    logits reach it, so the `N` best of the chunk do.
// 2. The threads whose best reaches the bar are few. Each counts how many of
//    them have a better best: fewer than `N` for a thread that holds one of
//    the `N` best of the chunk, and for at most `N` threads.
// 3. Those threads put their logits that reach the bar on one list, at most
//    `N * ARES_TOP_PER` entries, and every entry finds its place by counting
//    the entries above it.
//
// Packed values order by logit and then by token, and no two are equal, so
// the result does not depend on the order in which threads add to the lists.
template <int N, int WHICH>
static __device__ __forceinline__ void ares_sample_top_body(
    const float* logits, uint32_t stride, uint32_t n_local, const uint32_t* tokens, uint32_t first,
    uint32_t chunk, unsigned long long* lists, const AresSampleState* state) {
    static_assert(N <= 32, "the bar is one of the 32 columns' bests");
    // The best of every thread; after the bar is known, the bests that reach it.
    __shared__ unsigned long long thread_best[ARES_SAMPLE_THREADS];
    __shared__ unsigned long long column_best[32];
    __shared__ unsigned long long candidates[N * ARES_TOP_PER];
    __shared__ uint32_t n_strong;
    __shared__ uint32_t n_candidates;
    __shared__ uint32_t bad_any;
    const uint32_t tid = threadIdx.x;
    const uint32_t r = blockIdx.y;
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    const float* row = logits + (size_t)r * stride;
    if (tid == 0) {
        n_strong = 0;
        n_candidates = 0;
        bad_any = 0;
    }

    // Bit `k` of `valid`: the thread's logit `k` is one the kernel looks at.
    float v[ARES_TOP_PER];
    uint32_t valid = 0;
    bool bad = false;
#pragma unroll
    for (int k = 0; k < ARES_TOP_PER; ++k) {
        const uint32_t i = begin + k * blockDim.x + tid;
        v[k] = 0.0f;
        if (i < end) {
            // Adding zero turns a negative zero into a positive one, so that
            // equal logits have equal bits.
            v[k] = ares_sample_logit(row[i], &bad);
            valid |= 1u << k;
        }
    }
    if (WHICH == ARES_TOP_SET) {
        const float l_max = state[r].l_max;
        const uint32_t lo = state[r].lo;
        const uint32_t hi = state[r].hi;
#pragma unroll
        for (int k = 0; k < ARES_TOP_PER; ++k) {
            const uint32_t key = ares_sample_key(l_max, v[k]);
            if (!ares_sample_present(v[k]) || key < lo || key > hi) {
                valid &= ~(1u << k);
            }
        }
    }

    // The thread's best: the largest logit, and the lowest token among equal
    // ones. Columns in the order of the tokens need no token for that.
    float best_logit = 0.0f;
    int best_k = -1;
    bool tie = false;
#pragma unroll
    for (int k = 0; k < ARES_TOP_PER; ++k) {
        if (valid & (1u << k)) {
            if (best_k < 0 || v[k] > best_logit) {
                best_logit = v[k];
                best_k = k;
                tie = false;
            } else if (v[k] == best_logit) {
                tie = true;
            }
        }
    }
    unsigned long long mine = 0ull;
    if (best_k >= 0) {
        const uint32_t i = begin + best_k * blockDim.x + tid;
        uint32_t token = tokens ? tokens[first + i] : first + i;
        if (tie && tokens) {
#pragma unroll
            for (int k = 0; k < ARES_TOP_PER; ++k) {
                if ((valid & (1u << k)) && v[k] == best_logit) {
                    token = min(token, tokens[first + begin + k * blockDim.x + tid]);
                }
            }
        }
        mine = ares_sample_pack(best_logit, token);
    }
    thread_best[tid] = mine;
    __syncthreads();
    if (bad) {
        bad_any = 1;
    }

    // The bar: the `N`th best of the columns, zero when fewer columns have a
    // logit. Columns without one are equal; the lane orders them.
    const uint32_t lane = tid & 31;
    const uint32_t n_warps = blockDim.x >> 5;
    unsigned long long column = 0ull;
    for (uint32_t w = 0; w < n_warps; ++w) {
        const unsigned long long other = thread_best[w * 32 + lane];
        column = other > column ? other : column;
    }
    if (tid < 32) {
        column_best[lane] = column;
    }
    __syncthreads();
    uint32_t columns_above = 0;
#pragma unroll
    for (uint32_t j = 0; j < 32; ++j) {
        const unsigned long long other = column_best[j];
        columns_above += (other > column || (other == column && j < lane)) ? 1u : 0u;
    }
    const unsigned long long bar = ares_warp_max_u64(columns_above == N - 1 ? column : 0ull);

    const bool strong = mine != 0ull && mine >= bar;
    if (strong) {
        thread_best[atomicAdd(&n_strong, 1u)] = mine;
    }
    __syncthreads();
    uint32_t threads_above = N;
    if (strong) {
        threads_above = 0;
        const uint32_t n = n_strong;
        for (uint32_t j = 0; j < n; ++j) {
            threads_above += thread_best[j] > mine ? 1u : 0u;
        }
    }
    if (threads_above < N) {
#pragma unroll
        for (int k = 0; k < ARES_TOP_PER; ++k) {
            if (valid & (1u << k)) {
                const uint32_t i = begin + k * blockDim.x + tid;
                const unsigned long long packed =
                    ares_sample_pack(v[k], tokens ? tokens[first + i] : first + i);
                if (packed >= bar) {
                    const uint32_t at = atomicAdd(&n_candidates, 1u);
                    if (at < N * ARES_TOP_PER) {
                        candidates[at] = packed;
                    }
                }
            }
        }
    }
    __syncthreads();

    // Every candidate to its place in the chunk's list. Equal entries, which
    // only a token list that names a token twice gives, keep their order.
    const uint32_t n = min(n_candidates, (uint32_t)(N * ARES_TOP_PER));
    unsigned long long* out = lists + ((size_t)r * gridDim.x + blockIdx.x) * ARES_TOP_LIST(N);
    for (uint32_t c = tid; c < n; c += blockDim.x) {
        const unsigned long long packed = candidates[c];
        uint32_t above = 0;
        for (uint32_t j = 0; j < n; ++j) {
            const unsigned long long other = candidates[j];
            above += (other > packed || (other == packed && j < c)) ? 1u : 0u;
        }
        if (above < N) {
            out[above] = packed;
        }
    }
    if (tid >= n && tid < N) {
        out[tid] = 0ull;
    }
    if (tid == 0) {
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
