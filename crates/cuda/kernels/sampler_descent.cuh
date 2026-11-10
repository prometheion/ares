// The exact cut-offs of top-k and top-p: a radix descent over the keys.
//
// A token's key is the bit pattern of its distance below the best logit, so
// keys order like probabilities and "the k most probable tokens" or "the most
// probable tokens that hold the share p" are both "the tokens whose key is at
// most K". K is found from the top bits down, in three levels of 11, 11 and 10
// bits: a level counts, per value of its bits, the tokens that agree with the
// bits found so far (and sums their weights), and the first value at which
// the running total reaches the target is the next part of K.
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

#include "sampler_math.cuh"

// Which cut a level works on.
#define ARES_SAMPLE_TOP_K 0u
#define ARES_SAMPLE_TOP_P 1u

// Words per bin of a histogram: the count, and the sum of the weights in
// three limbs of 14 bits (each limb's sum over a whole vocabulary of up to
// 2^18 tokens stays below 2^32).
#define ARES_SAMPLE_BIN_WORDS 4

static __device__ __forceinline__ bool ares_sample_takes_part(uint32_t flags, uint32_t descent) {
    const uint32_t need =
        ARES_SAMPLE_F_SAMPLE | (descent == ARES_SAMPLE_TOP_K ? ARES_SAMPLE_F_TOP_K : ARES_SAMPLE_F_TOP_P);
    return (flags & need) == need;
}

// The bins of a level: level 0 takes the top 11 bits of the key, which never
// exceed 1020 for a finite or infinite distance.
static __device__ __forceinline__ uint32_t ares_sample_level_bins(uint32_t level) {
    return level == 1 ? 2048u : 1024u;
}

// Grid (chunks, rows), ARES_SAMPLE_THREADS threads. Adds the tokens of chunk
// `blockIdx.x` of row `blockIdx.y` to the row's histogram
// `hist[(r * ARES_SAMPLE_BINS + bin) * ARES_SAMPLE_BIN_WORDS ..]`, which must be zero
// before the first chunk adds to it.
ARES_KERNEL ares_sample_hist(const float* logits, uint32_t stride, uint32_t n_local, uint32_t chunk,
                             const AresSampleParams* params, const AresSampleState* state,
                             uint32_t* hist, uint32_t descent, uint32_t level) {
    __shared__ uint32_t local[ARES_SAMPLE_BINS * ARES_SAMPLE_BIN_WORDS];
    const uint32_t r = blockIdx.y;
    if (!ares_sample_takes_part(params[r].flags, descent)) {
        return;
    }
    const uint32_t bins = ares_sample_level_bins(level);
    for (uint32_t i = threadIdx.x; i < bins * ARES_SAMPLE_BIN_WORDS; i += ARES_SAMPLE_THREADS) {
        local[i] = 0;
    }
    __syncthreads();

    const float l_max = state[r].l_max;
    const uint32_t prefix = descent == ARES_SAMPLE_TOP_K ? state[r].prefix_k : state[r].prefix_p;
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    const float* row = logits + (size_t)r * stride;
    for (uint32_t i = begin + threadIdx.x; i < end; i += ARES_SAMPLE_THREADS) {
        const uint32_t key = ares_sample_key(l_max, row[i]);
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
        const uint64_t w = ares_sample_weight(key);
        uint32_t* at = local + bin * ARES_SAMPLE_BIN_WORDS;
        atomicAdd(at, 1u);
        if (w != 0) {
            atomicAdd(at + 1, (uint32_t)(w & 0x3fffu));
            atomicAdd(at + 2, (uint32_t)((w >> 14) & 0x3fffu));
            atomicAdd(at + 3, (uint32_t)(w >> 28));
        }
    }
    __syncthreads();
    uint32_t* global = hist + (size_t)r * ARES_SAMPLE_BINS * ARES_SAMPLE_BIN_WORDS;
    for (uint32_t i = threadIdx.x; i < bins * ARES_SAMPLE_BIN_WORDS; i += ARES_SAMPLE_THREADS) {
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
                               uint32_t descent) {
    const uint32_t r = blockIdx.y;
    if (!ares_sample_takes_part(params[r].flags, descent)) {
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
// (summed over all devices) and takes the row one level down.
ARES_KERNEL ares_sample_decide(const float* exchange, const AresSampleParams* params,
                               AresSampleState* state, uint32_t descent, uint32_t level) {
    __shared__ uint64_t counts[ARES_SAMPLE_THREADS];
    __shared__ uint64_t weights[ARES_SAMPLE_THREADS];
    const uint32_t r = blockIdx.x;
    const AresSampleParams p = params[r];
    if (!ares_sample_takes_part(p.flags, descent)) {
        return;
    }
    const uint32_t bins = ares_sample_level_bins(level);
    const uint32_t per_thread = bins / ARES_SAMPLE_THREADS;
    const float* row = exchange + (size_t)r * ARES_SAMPLE_BINS * ARES_SAMPLE_BIN_WORDS;
    const float* mine = row + (size_t)threadIdx.x * per_thread * ARES_SAMPLE_BIN_WORDS;

    uint64_t c = 0, w = 0;
    for (uint32_t j = 0; j < per_thread; ++j) {
        c += ares_sample_bin_count(mine + j * ARES_SAMPLE_BIN_WORDS);
        w += ares_sample_bin_weight(mine + j * ARES_SAMPLE_BIN_WORDS);
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

    AresSampleState* s = state + r;
    const bool by_count = descent == ARES_SAMPLE_TOP_K;
    uint64_t target;
    if (by_count) {
        target = s->rem_k;
    } else if (level == 0) {
        // The share top_p of what top-k left, or of everything.
        const uint64_t z = (p.flags & ARES_SAMPLE_F_TOP_K) ? s->z_k : weights[ARES_SAMPLE_THREADS - 1];
        const uint64_t lo = (z & 0xffffffffull) * p.top_p_q32;
        const uint64_t hi = (z >> 32) * p.top_p_q32;
        target = hi + (lo >> 32);
        target = target == 0 ? 1 : target;
    } else {
        target = s->rem_p;
    }
    const uint64_t* totals = by_count ? counts : weights;
    const uint64_t before = threadIdx.x ? totals[threadIdx.x - 1] : 0;
    const uint64_t w_before_mine = threadIdx.x ? weights[threadIdx.x - 1] : 0;
    if (threadIdx.x == ARES_SAMPLE_THREADS - 1 && totals[threadIdx.x] < target) {
        // Fewer tokens than top-k asks for: no cut. A prefix no key has
        // keeps the deeper levels empty.
        if (by_count) {
            s->prefix_k = 0xffffffffu;
            s->rem_k = 0;
            s->z_k += weights[threadIdx.x];
        } else {
            s->prefix_p = 0xffffffffu;
            s->rem_p = 0;
        }
        return;
    }
    if (!(before < target && target <= totals[threadIdx.x])) {
        return;
    }
    uint64_t run = before;
    uint64_t w_run = w_before_mine;
    uint32_t bin = 0;
    uint64_t w_bin = 0;
    for (uint32_t j = 0; j < per_thread; ++j) {
        const float* b = mine + j * ARES_SAMPLE_BIN_WORDS;
        const uint64_t v = by_count ? ares_sample_bin_count(b) : ares_sample_bin_weight(b);
        w_bin = ares_sample_bin_weight(b);
        bin = threadIdx.x * per_thread + j;
        if (run + v >= target) {
            break;
        }
        run += v;
        w_run += w_bin;
    }
    const uint32_t shift = level == 0 ? 0u : (level == 1 ? 11u : 10u);
    const bool last = level == 2;
    if (by_count) {
        s->prefix_k = (s->prefix_k << shift) | bin;
        s->rem_k = (uint32_t)(target - run);
        s->z_k += w_run + (last ? w_bin : 0);
        if (last) {
            s->cut_key = min(s->cut_key, s->prefix_k);
        }
    } else {
        s->prefix_p = (s->prefix_p << shift) | bin;
        s->rem_p = target - run;
        if (last) {
            s->cut_key = min(s->cut_key, s->prefix_p);
        }
    }
}
