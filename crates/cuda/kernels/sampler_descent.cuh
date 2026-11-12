// The exact cut-offs of top-k, top-p and typical sampling, and the hundredth
// token of mirostat 1: a radix descent over keys.
//
// A token's key is the bit pattern of a distance that orders like its
// probability (for top-k and top-p the distance below the best logit; for
// typical sampling the distance of the scaled logit from its mean), so "the k
// most probable tokens" or "the most probable tokens that hold the share p"
// are both "the tokens whose key is at most K". K is found from the top bits
// down, in three levels of 11, 11 and 10 bits: a level counts, per value of
// its bits, the tokens that agree with the bits found so far (and sums their
// weights), and the first value at which the running total reaches the target
// is the next part of K. Only the tokens of the row's set count.
//
// A level is three kernels. `ares_sample_hist` builds the histogram of a
// device's slice of the vocabulary, `ares_sample_export` turns it into f32
// values that add exactly (so the histograms of several devices are joined by
// the sum the devices have anyway), and `ares_sample_decide` reads the total
// and moves the row's state one level down.
//
// Tokens tied at the cut are all kept: the cut is a threshold on the
// probability, not a number of tokens.
#pragma once

#include "sampler_stage.cuh"

// Which descent a level belongs to.
#define ARES_DESC_TOPK 0u
#define ARES_DESC_TOPP 1u
#define ARES_DESC_TYPICAL 2u
#define ARES_DESC_K100 3u

static __device__ __forceinline__ bool ares_desc_runs(const AresSampleParams& p,
                                                      const AresSampleState& s, uint32_t desc,
                                                      uint32_t position) {
    if (!(p.flags & ARES_SAMPLE_F_SAMPLE) || (s.status & ARES_SAMPLE_S_EMPTY) || s.cursor != position) {
        return false;
    }
    const uint32_t stage = ares_stage_at(p, position);
    switch (desc) {
        case ARES_DESC_TOPK:
            return stage == ARES_STAGE_TOPK || stage == ARES_STAGE_MIRO1;
        case ARES_DESC_TOPP:
            return stage == ARES_STAGE_TOPP;
        case ARES_DESC_TYPICAL:
            return stage == ARES_STAGE_TYPICAL;
        default:
            return stage == ARES_STAGE_MIRO1;
    }
}

// The bins of a level: level 0 takes the top 11 bits of the key, which never
// exceed 1020 for a finite or infinite distance.
static __device__ __forceinline__ uint32_t ares_sample_level_bins(uint32_t level) {
    return level == 1 ? 2048u : 1024u;
}

// A row with at most this many tokens for a histogram (as far as the kernels
// before know, see `AresSampleState::set_tokens`) adds them to it one by one.
#define ARES_HIST_FEW 256

// Adds the tokens of the set among this thread's `n` logits `v` to the
// histogram at `bins`: the count of each token, and its weight unless the
// descent goes by count.
static __device__ __forceinline__ void ares_hist_chunk(const float* v, uint32_t n,
                                                       const AresSampleState& s, uint32_t desc,
                                                       uint32_t level, bool by_count, uint32_t* bins) {
    const uint32_t prefix = s.prefix;
#pragma unroll
    for (int k = 0; k < ARES_SAMPLE_PER; ++k) {
        if ((uint32_t)k >= n) {
            break;
        }
        bool bad = false;
        const float l = ares_sample_logit(v[k], &bad);
        if (!ares_sample_present(l)) {
            continue;
        }
        const uint32_t token_key = ares_sample_key(s.l_max, l);
        if (!ares_in_set(s, token_key)) {
            continue;
        }
        float d, x;
        ares_sample_x(s, token_key, &d, &x);
        // The key the descent orders by.
        const uint32_t key =
            desc == ARES_DESC_TYPICAL ? __float_as_uint(fabsf(__fsub_rn(x, s.mean_x))) : token_key;
        uint32_t bin;
        if (level == 0) {
            bin = key >> 21;
        } else if (level == 1) {
            if ((key >> 21) != prefix) {
                continue;
            }
            bin = (key >> 10) & 0x7ffu;
        } else {
            if ((key >> 10) != prefix) {
                continue;
            }
            bin = key & 0x3ffu;
        }
        uint32_t* at = bins + bin * ARES_SAMPLE_BIN_WORDS;
        atomicAdd(at, 1u);
        if (by_count) {
            continue;
        }
        const uint64_t w = ares_sample_weight(__float_as_uint(x));
        if (w != 0) {
            atomicAdd(at + 1, (uint32_t)(w & 0x3fffu));
            atomicAdd(at + 2, (uint32_t)((w >> 14) & 0x3fffu));
            atomicAdd(at + 3, (uint32_t)(w >> 28));
        }
    }
}

// Grid (chunks, rows), ARES_SAMPLE_THREADS threads, `chunk` at most
// ARES_SAMPLE_PER logits per thread. Adds the tokens of the set
// in chunk `blockIdx.x` of row `blockIdx.y` to the row's histogram
// `hist[(r * ARES_SAMPLE_BINS + bin) * ARES_SAMPLE_BIN_WORDS ..]`, which must be zero
// before the first chunk adds to it. A descent by count (top-k, the
// hundredth token) reads no weights, so its histograms hold none.
//
// A chunk builds its histogram in shared memory, where the same few bins take
// most of the tokens, and adds the bins it used to the row's. After a cut,
// and below the first level, a row has few tokens that count; the kernel that
// decided the level before says how many, and then the chunks skip their own
// histograms and add the tokens to the row's directly.
ARES_KERNEL ares_sample_hist(const float* logits, uint32_t stride, uint32_t n_local, uint32_t chunk,
                             const AresSampleParams* params, const AresSampleState* state,
                             uint32_t* hist, uint32_t desc, uint32_t level, uint32_t position) {
    __shared__ uint32_t local[ARES_SAMPLE_BINS * ARES_SAMPLE_BIN_WORDS];
    const uint32_t r = blockIdx.y;
    if (!ares_desc_runs(params[r], state[r], desc, position)) {
        return;
    }
    const AresSampleState s = state[r];
    const bool by_count = desc == ARES_DESC_TOPK || desc == ARES_DESC_K100;
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    float v[ARES_SAMPLE_PER];
    const uint32_t n = ares_sample_read(logits + (size_t)r * stride, begin, end, v);
    uint32_t* global = hist + (size_t)r * ARES_SAMPLE_BINS * ARES_SAMPLE_BIN_WORDS;
    if ((level == 0 ? s.set_tokens : s.level_tokens) <= ARES_HIST_FEW) {
        ares_hist_chunk(v, n, s, desc, level, by_count, global);
        return;
    }
    // The words of a bin that this descent uses: the count alone, or all.
    const uint32_t words = ares_sample_level_bins(level) * ARES_SAMPLE_BIN_WORDS;
    const uint32_t step = by_count ? ARES_SAMPLE_BIN_WORDS : 1u;
    for (uint32_t i = threadIdx.x * step; i < words; i += ARES_SAMPLE_THREADS * step) {
        local[i] = 0;
    }
    __syncthreads();
    ares_hist_chunk(v, n, s, desc, level, by_count, local);
    __syncthreads();
    for (uint32_t i = threadIdx.x * step; i < words; i += ARES_SAMPLE_THREADS * step) {
        const uint32_t v = local[i];
        if (v != 0) {
            atomicAdd(global + i, v);
        }
    }
}

// Grid (bins / ARES_SAMPLE_THREADS, rows), ARES_SAMPLE_THREADS threads. Moves
// the histogram of a level into `exchange` as four f32 values per bin (the
// count and the weight in limbs of 20 bits: sums of up to 16 devices stay
// exact) and clears the histogram for the next level.
ARES_KERNEL ares_sample_export(uint32_t* hist, float* exchange, const AresSampleParams* params,
                               const AresSampleState* state, uint32_t desc, uint32_t position) {
    const uint32_t r = blockIdx.y;
    if (!ares_desc_runs(params[r], state[r], desc, position)) {
        return;
    }
    const size_t at = ((size_t)r * ARES_SAMPLE_BINS + blockIdx.x * ARES_SAMPLE_THREADS + threadIdx.x) *
                      ARES_SAMPLE_BIN_WORDS;
    const uint64_t w = (uint64_t)hist[at + 1] + ((uint64_t)hist[at + 2] << 14) + ((uint64_t)hist[at + 3] << 28);
    exchange[at] = (float)hist[at];
    exchange[at + 1] = (float)(uint32_t)(w & 0xfffffu);
    exchange[at + 2] = (float)(uint32_t)((w >> 20) & 0xfffffu);
    exchange[at + 3] = (float)(uint32_t)(w >> 40);
    hist[at] = 0;
    hist[at + 1] = 0;
    hist[at + 2] = 0;
    hist[at + 3] = 0;
}

static __device__ __forceinline__ uint64_t ares_sample_bin_count(const float* bin) {
    return (uint64_t)bin[0];
}

static __device__ __forceinline__ uint64_t ares_sample_bin_weight(const float* bin) {
    return (uint64_t)bin[1] + ((uint64_t)bin[2] << 20) + ((uint64_t)bin[3] << 40);
}

// Grid (rows), ARES_SAMPLE_THREADS threads. Reads the histogram of a level
// (summed over all devices) and takes the row one level down. After the last
// level the cut is known: top-k and top-p cap the set and are done, typical
// sampling and mirostat 1 keep the key for the stage's next kernel.
ARES_KERNEL ares_sample_decide(const float* exchange, const AresSampleParams* params,
                               AresSampleState* state, uint32_t desc, uint32_t level,
                               uint32_t position) {
    __shared__ uint64_t counts[ARES_SAMPLE_THREADS];
    __shared__ uint64_t weights[ARES_SAMPLE_THREADS];
    const uint32_t r = blockIdx.x;
    const AresSampleParams p = params[r];
    AresSampleState* s = state + r;
    if (!ares_desc_runs(p, *s, desc, position)) {
        return;
    }
    const uint32_t bins = ares_sample_level_bins(level);
    const uint32_t per_thread = bins / ARES_SAMPLE_THREADS;
    const float* row = exchange + (size_t)r * ARES_SAMPLE_BINS * ARES_SAMPLE_BIN_WORDS;
    const float* mine = row + (size_t)threadIdx.x * per_thread * ARES_SAMPLE_BIN_WORDS;

    // The thread's bins, all read before one is used.
    float own[ARES_SAMPLE_BINS / ARES_SAMPLE_THREADS * ARES_SAMPLE_BIN_WORDS];
#pragma unroll
    for (uint32_t j = 0; j < ARES_SAMPLE_BINS / ARES_SAMPLE_THREADS * ARES_SAMPLE_BIN_WORDS; ++j) {
        own[j] = j < per_thread * ARES_SAMPLE_BIN_WORDS ? mine[j] : 0.0f;
    }
    uint64_t c = 0, w = 0;
#pragma unroll
    for (uint32_t j = 0; j < ARES_SAMPLE_BINS / ARES_SAMPLE_THREADS; ++j) {
        c += ares_sample_bin_count(own + j * ARES_SAMPLE_BIN_WORDS);
        w += ares_sample_bin_weight(own + j * ARES_SAMPLE_BIN_WORDS);
    }
    counts[threadIdx.x] = c;
    weights[threadIdx.x] = w;
    // Running totals over the threads, the thread's own bins included.
    for (uint32_t offset = 1; offset < ARES_SAMPLE_THREADS; offset <<= 1) {
        __syncthreads();
        const uint64_t pc = threadIdx.x >= offset ? counts[threadIdx.x - offset] : 0;
        const uint64_t pw = threadIdx.x >= offset ? weights[threadIdx.x - offset] : 0;
        __syncthreads();
        counts[threadIdx.x] += pc;
        weights[threadIdx.x] += pw;
    }
    __syncthreads();

    const bool by_count = desc == ARES_DESC_TOPK || desc == ARES_DESC_K100;
    __shared__ uint64_t shared_target;
    if (threadIdx.x == 0) {
        uint64_t target;
        if (level > 0) {
            target = s->rem;
        } else if (by_count) {
            target = desc == ARES_DESC_K100 ? (uint64_t)ARES_SAMPLE_M : (uint64_t)s->k;
        } else {
            // The share of the weight of the set.
            const uint64_t z = weights[ARES_SAMPLE_THREADS - 1];
            if (desc == ARES_DESC_TOPP) {
                target = ares_sample_share(z, p.top_p_q32);
                target = target == 0 ? 1 : target;
            } else {
                target = ares_sample_share(z, p.typical_q32) + 1;
            }
        }
        shared_target = target;
    }
    __syncthreads();
    const uint64_t target = shared_target;
    const uint64_t* totals = by_count ? counts : weights;
    const uint64_t before = threadIdx.x ? totals[threadIdx.x - 1] : 0;
    const uint32_t shift = level == 0 ? 0u : (level == 1 ? 11u : 10u);
    if (threadIdx.x == ARES_SAMPLE_THREADS - 1 && totals[threadIdx.x] < target) {
        // Fewer tokens than the target asks for: no cut. A prefix no key has
        // keeps the deeper levels empty.
        s->prefix = 0xffffffffu;
        s->rem = 0;
        s->level_tokens = 0;
    } else if (before < target && target <= totals[threadIdx.x]) {
        uint64_t run = before;
        uint32_t bin = 0;
        // The tokens in the bins before the one the target falls in, and in it.
        uint64_t tokens_below = threadIdx.x ? counts[threadIdx.x - 1] : 0;
        uint64_t tokens_in = 0;
        for (uint32_t j = 0; j < per_thread; ++j) {
            const float* b = mine + j * ARES_SAMPLE_BIN_WORDS;
            tokens_in = ares_sample_bin_count(b);
            const uint64_t v = by_count ? tokens_in : ares_sample_bin_weight(b);
            bin = threadIdx.x * per_thread + j;
            if (run + v >= target) {
                break;
            }
            run += v;
            tokens_below += tokens_in;
        }
        s->prefix = level == 0 ? bin : (s->prefix << shift) | bin;
        s->rem = target - run;
        s->below_tokens = (level == 0 ? 0u : s->below_tokens) + (uint32_t)tokens_below;
        s->level_tokens = (uint32_t)tokens_in;
    }
    __syncthreads();
    if (threadIdx.x != 0 || level != 2) {
        return;
    }
    const uint32_t cut = s->prefix;
    switch (desc) {
        case ARES_DESC_TOPK:
        case ARES_DESC_TOPP:
            if (cut != 0xffffffffu) {
                s->hi = min(s->hi, cut);
                // The tokens of the set up to the cut are what is left of it.
                s->set_tokens = min(s->set_tokens, s->below_tokens + s->level_tokens);
            }
            ares_stage_done(p, s, position);
            break;
        case ARES_DESC_TYPICAL:
            s->cut = cut;
            break;
        default:
            s->cut100 = cut;
            break;
    }
}
