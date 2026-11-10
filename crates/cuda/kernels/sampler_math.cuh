// The arithmetic of the sampler, written so that the host can repeat it bit
// for bit (src/sampler.rs has the same functions in Rust).
//
// Rules that make that possible:
// - Floating point goes through `__fmul_rn`, `__fadd_rn`, `__fsub_rn` and
//   `__fdiv_rn` only: IEEE operations in round-to-nearest that the compiler
//   never fuses into a multiply-add. No `expf`, no `logf`: the exponential and
//   the logarithm below are polynomials of those operations.
// - Everything that is summed over tokens is an integer, so the order of the
//   sum (threads, blocks, devices) does not matter.
#pragma once

#include "common.cuh"

// The candidates kept per row: the best tokens by logit, for the greedy pick
// and for a grammar that refuses the pick.
#define ARES_SAMPLE_CANDIDATES 8
// Threads per block of every sampler kernel.
#define ARES_SAMPLE_THREADS 256
// Bins of a level of the radix descent, at most.
#define ARES_SAMPLE_BINS 2048

// Row flags (AresSampleParams::flags).
#define ARES_SAMPLE_F_SAMPLE 1u
#define ARES_SAMPLE_F_TOP_K 2u
#define ARES_SAMPLE_F_TOP_P 4u

// Row status (AresSampleOut::status).
#define ARES_SAMPLE_S_NONFINITE 1u
#define ARES_SAMPLE_S_EMPTY 2u

// The largest key a token may have and still be picked: the largest finite
// distance from the best logit. A logit of minus infinity has the key of plus
// infinity, one above.
#define ARES_SAMPLE_KEY_MAX 0x7f7fffffu

// What the host sets per row (32 bytes).
struct AresSampleParams {
    uint32_t flags;
    float inv_temperature;
    uint32_t top_k;
    // top_p as a fraction of 2^32.
    uint32_t top_p_q32;
    // The key of -ln(min_p): tokens above it are cut. ARES_SAMPLE_KEY_MAX
    // when min-p is off.
    uint32_t min_p_key;
    uint32_t seed_lo;
    uint32_t seed_hi;
    uint32_t position;
};

// A row's state between the kernels of one pick (64 bytes).
struct AresSampleState {
    float l_max;
    uint32_t status;
    uint32_t prefix_k;
    uint32_t rem_k;
    uint64_t z_k;
    uint32_t prefix_p;
    uint32_t cut_key;
    uint64_t rem_p;
    // The best score so far and its token, packed (see ares_sample_pack).
    unsigned long long best;
    uint64_t reserved[3];
};

// What reaches the host per row (80 bytes).
struct AresSampleOut {
    uint32_t token;
    uint32_t status;
    // The logit of a greedy pick, the score (logit / temperature + noise) of
    // a sampled one.
    float value;
    uint32_t reserved;
    uint32_t cand_token[ARES_SAMPLE_CANDIDATES];
    float cand_logit[ARES_SAMPLE_CANDIDATES];
};

static __device__ __forceinline__ uint32_t ares_f32_bits(float v) {
    return __float_as_uint(v);
}

// A float as an unsigned number with the same order (minus infinity lowest).
static __device__ __forceinline__ uint32_t ares_sample_sortable(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

static __device__ __forceinline__ float ares_sample_unsortable(uint32_t s) {
    return __uint_as_float((s & 0x80000000u) ? (s & 0x7fffffffu) : ~s);
}

// A value and its token as one number whose order is "higher value first,
// lower token on a tie". Zero is below every packed pair and means "none".
static __device__ __forceinline__ unsigned long long ares_sample_pack(float value, uint32_t token) {
    return ((unsigned long long)ares_sample_sortable(value) << 32) | (0xffffffffu - token);
}

// The key of a token: the bits of its distance below the best logit, which
// order like the distance itself. Smaller key, larger probability.
static __device__ __forceinline__ uint32_t ares_sample_key(float l_max, float logit) {
    const float d = __fsub_rn(l_max, logit);
    return d > 0.0f ? __float_as_uint(d) : 0u;
}

// exp(-d) for d >= 0 as a fixed-point number with 40 fractional bits, 2^40
// for d = 0 and 0 from about d = 28 on. The same polynomial as
// `weight_fixed` in src/sampler.rs.
static __device__ __forceinline__ uint64_t ares_sample_weight(uint32_t key) {
    const float d = __uint_as_float(key);
    const float t = __fmul_rn(d, 1.44269504088896340736f);
    if (!(t < 41.0f)) {
        return 0;
    }
    const uint32_t n = (uint32_t)t;
    const float f = __fsub_rn(t, (float)n);
    const float r = __fmul_rn(f, 0.693147180559945309417f);
    // exp(-r) for r in [0, ln 2): the Taylor series to r^9, by Horner.
    float p = -1.0f / 362880.0f;
    p = __fadd_rn(__fmul_rn(p, r), 1.0f / 40320.0f);
    p = __fadd_rn(__fmul_rn(p, r), -1.0f / 5040.0f);
    p = __fadd_rn(__fmul_rn(p, r), 1.0f / 720.0f);
    p = __fadd_rn(__fmul_rn(p, r), -1.0f / 120.0f);
    p = __fadd_rn(__fmul_rn(p, r), 1.0f / 24.0f);
    p = __fadd_rn(__fmul_rn(p, r), -1.0f / 6.0f);
    p = __fadd_rn(__fmul_rn(p, r), 0.5f);
    p = __fadd_rn(__fmul_rn(p, r), -1.0f);
    p = __fadd_rn(__fmul_rn(p, r), 1.0f);
    const uint64_t m = (uint64_t)__fmul_rn(p, 1099511627776.0f);
    return m >> n;
}

// ln(x) for a positive normal x.
static __device__ __forceinline__ float ares_sample_ln(float x) {
    uint32_t b = __float_as_uint(x);
    int e = (int)(b >> 23) - 127;
    uint32_t mant = b & 0x007fffffu;
    // Bring the mantissa into [sqrt(1/2), sqrt(2)).
    if (mant > 0x003504f3u) {
        e += 1;
        b = mant | 0x3f000000u;
    } else {
        b = mant | 0x3f800000u;
    }
    const float m = __uint_as_float(b);
    const float s = __fdiv_rn(__fsub_rn(m, 1.0f), __fadd_rn(m, 1.0f));
    const float z = __fmul_rn(s, s);
    // 2 atanh(s) = 2 s (1 + z/3 + z^2/5 + z^3/7 + z^4/9).
    float p = 1.0f / 9.0f;
    p = __fadd_rn(__fmul_rn(p, z), 1.0f / 7.0f);
    p = __fadd_rn(__fmul_rn(p, z), 1.0f / 5.0f);
    p = __fadd_rn(__fmul_rn(p, z), 1.0f / 3.0f);
    p = __fadd_rn(__fmul_rn(p, z), 1.0f);
    const float lm = __fmul_rn(__fmul_rn(2.0f, s), p);
    return __fadd_rn(__fmul_rn((float)e, 0.693147180559945309417f), lm);
}

// Philox 4x32 with 10 rounds (Salmon et al., "Parallel random numbers: as
// easy as 1, 2, 3"): a counter-based generator, so the number for a counter
// does not depend on any other number drawn. Returns the first output word.
static __device__ __forceinline__ uint32_t ares_sample_philox(uint32_t c0, uint32_t c1, uint32_t k0,
                                                              uint32_t k1) {
    uint32_t c2 = 0, c3 = 0;
    for (int round = 0; round < 10; ++round) {
        const uint64_t p0 = (uint64_t)0xD2511F53u * c0;
        const uint64_t p1 = (uint64_t)0xCD9E8D57u * c2;
        const uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
        const uint32_t n1 = (uint32_t)p1;
        const uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
        const uint32_t n3 = (uint32_t)p0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}

// The Gumbel noise of `token` at `position` for a seed: -ln(-ln(u)) with u
// uniform in (0, 1) from the generator above.
static __device__ __forceinline__ float ares_sample_gumbel(uint32_t seed_lo, uint32_t seed_hi,
                                                           uint32_t position, uint32_t token) {
    const uint32_t x = ares_sample_philox(token, position, seed_lo, seed_hi);
    const float u = __fmul_rn(__fadd_rn((float)(x >> 9), 0.5f), 1.0f / 8388608.0f);
    const float e = -ares_sample_ln(u);
    return -ares_sample_ln(e);
}

// The largest `v` of the 32 lanes of a warp, returned to every lane.
static __device__ __forceinline__ unsigned long long ares_warp_max_u64(unsigned long long v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, v, offset);
        v = o > v ? o : v;
    }
    return v;
}

// The largest `v` of all threads of the block, returned to every thread.
// `partial` is shared memory with one value per warp; a caller that reuses it
// must __syncthreads() before the next call.
static __device__ __forceinline__ unsigned long long ares_block_max_u64(unsigned long long v,
                                                                        unsigned long long* partial) {
    v = ares_warp_max_u64(v);
    const uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        partial[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    const uint32_t n_warps = blockDim.x >> 5;
    return ares_warp_max_u64(lane < n_warps ? partial[lane] : 0ull);
}
