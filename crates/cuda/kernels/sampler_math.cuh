// The arithmetic of the sampler, written so that the host can repeat it bit
// for bit (crates/sampler/src/arith.rs has the same functions in Rust).
//
// Rules that make that possible:
// - Floating point goes through `__fmul_rn`, `__fadd_rn`, `__fsub_rn`,
//   `__fdiv_rn` and their double precision twins only: IEEE operations in
//   round-to-nearest that the compiler never fuses into a multiply-add. No
//   `expf`, no `logf`, no `pow`: the exponential and the logarithm below are
//   polynomials of those operations.
// - Everything that is summed over tokens is an integer, so the order of the
//   sum (threads, blocks, devices) does not matter.
#pragma once

#include "common.cuh"

// The candidates kept per row: the best tokens by logit, for the greedy pick
// and for a grammar that refuses the pick.
#define ARES_SAMPLE_CANDIDATES 8
// The alternatives a row reports next to the token it picked.
#define ARES_LOGPROBS_MAX 20
// Threads per block of every sampler kernel.
#define ARES_SAMPLE_THREADS 256
// Bins of a level of the radix descent, at most.
#define ARES_SAMPLE_BINS 2048
// Words per bin of a histogram: the count, and the sum of the weights in
// three limbs of 14 bits (each limb's sum over a whole vocabulary of up to
// 2^18 tokens stays below 2^32).
#define ARES_SAMPLE_BIN_WORDS 4
// f32 values per device and row in the exchange of a pass.
#define ARES_SAMPLE_PASS_VALUES 24
#define ARES_SAMPLE_MAX_STAGES 16
// The tokens mirostat 1 estimates its exponent from.
#define ARES_SAMPLE_M 100

// Row flags (AresSampleParams::flags).
#define ARES_SAMPLE_F_SAMPLE 1u
#define ARES_SAMPLE_F_MASK 2u
#define ARES_SAMPLE_F_LP_RAW 4u
#define ARES_SAMPLE_F_LP_POST 8u
#define ARES_SAMPLE_F_MIROSTAT 16u

// Row status (AresSampleOut::status).
#define ARES_SAMPLE_S_NONFINITE 1u
#define ARES_SAMPLE_S_EMPTY 2u

// The stages of a row's program (four bits each).
#define ARES_STAGE_NSIGMA 1u
#define ARES_STAGE_TOPK 2u
#define ARES_STAGE_TYPICAL 3u
#define ARES_STAGE_TOPP 4u
#define ARES_STAGE_MINP 5u
#define ARES_STAGE_XTC 6u
#define ARES_STAGE_TEMP 7u
#define ARES_STAGE_MIRO1 8u
#define ARES_STAGE_MIRO2 9u

// The largest key a token may have and still be picked: the largest finite
// distance from the best logit. A logit of minus infinity has the key of plus
// infinity, one above.
#define ARES_SAMPLE_KEY_MAX 0x7f7fffffu

// The smallest temperature that scales the logits.
#define ARES_SAMPLE_TEMPERATURE_MIN 1e-6f

// What the host sets per row (128 bytes, `SampleRow`).
struct AresSampleParams {
    uint32_t flags;
    uint32_t seed_lo;
    uint32_t seed_hi;
    uint32_t position;
    // The stages, four bits each: the first eight, the last eight.
    uint32_t program_lo;
    uint32_t program_hi;
    uint32_t top_k;
    // Shares as fractions of 2^32.
    uint32_t top_p_q32;
    // -ln(min_p).
    float min_p_c;
    uint32_t typical_q32;
    float n_sigma;
    float xtc_prob;
    uint32_t xtc_thr_q32;
    float temp;
    // 1 / temp, or 0 when temp is below the smallest temperature.
    float inv_temp;
    float dyn_delta;
    float dyn_exp;
    float mu;
    uint32_t vocab;
    uint32_t entries_first;
    uint32_t entries_count;
    // The mask of the row, or 0xffffffff.
    uint32_t mask_slot;
    uint32_t logprobs;
    uint32_t reserved[9];
};
static_assert(sizeof(AresSampleParams) == 128, "SampleRow is 32 words");

// A correction of one token's logit (`Entry`). After the correction the
// device leaves the logit it found in `bias`, so that the raw logit of a
// token is known.
struct AresEntry {
    uint32_t token;
    float bias;
    float rep;
    float sub;
};
static_assert(sizeof(AresEntry) == 16, "Entry is 16 bytes");

// A row's state between the kernels of one pick (256 bytes).
struct AresSampleState {
    float l_max;
    uint32_t status;
    // The tokens still in the running have a key from lo to hi.
    uint32_t lo;
    uint32_t hi;
    float inv_t;
    // The next position of the program to run.
    uint32_t cursor;
    // The k of the top-k that runs (a mirostat 1 sets it).
    uint32_t k;
    // The radix descent.
    uint32_t prefix;
    uint64_t rem;
    // The result of the last descent and the target of the scan after it.
    uint32_t cut;
    uint32_t cut100;
    uint64_t target;
    unsigned long long best;
    // The sums of a pass, accumulated by its blocks, and the same after the
    // devices added them up.
    unsigned long long acc[5];
    uint64_t sums[5];
    float mean_x;
    uint32_t n_gather;
    // The raw logits' best, for probabilities of the model's own softmax.
    float raw_lmax;
    uint32_t pick_token;
    float pick_raw;
    float pick_cor;
    // The logarithm of the weight of all tokens, for the model's own
    // probabilities.
    double raw_lnz;
    uint32_t reserved[20];
};
static_assert(sizeof(AresSampleState) == 256, "the host sizes the scratch by 256 bytes a row");

// What reaches the host per row (80 bytes, `SamplePick`).
struct AresSampleOut {
    uint32_t token;
    uint32_t status;
    // The logit of a greedy pick, the score (logit * inv_t + noise) of a
    // sampled one.
    float value;
    // The surprise of a mirostat row's pick, in bits.
    float aux;
    uint32_t cand_token[ARES_SAMPLE_CANDIDATES];
    float cand_logit[ARES_SAMPLE_CANDIDATES];
};
static_assert(sizeof(AresSampleOut) == 80, "SamplePick is 80 bytes");

// The probabilities a row reports (`LogprobOut`).
struct AresLogprobOut {
    uint32_t n;
    float chosen;
    uint32_t token[ARES_LOGPROBS_MAX];
    float logprob[ARES_LOGPROBS_MAX];
};
static_assert(sizeof(AresLogprobOut) == 168, "LogprobOut is 168 bytes");

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

// A logit as the sampler sees it: negative zero made positive, and a NaN or
// plus infinity (reported through `bad`) made minus infinity, which no stage
// keeps.
static __device__ __forceinline__ float ares_sample_logit(float v, bool* bad) {
    const float l = __fadd_rn(v, 0.0f);
    if (!(l <= 3.402823466e+38f)) {
        *bad = true;
        return -__uint_as_float(0x7f800000u);
    }
    return l;
}

static __device__ __forceinline__ bool ares_sample_present(float l) {
    return l > -3.402823466e+38f;
}

// The key of a token: the bits of its distance below the best logit, which
// order like the distance itself. Smaller key, larger probability.
static __device__ __forceinline__ uint32_t ares_sample_key(float l_max, float logit) {
    const float d = __fsub_rn(l_max, logit);
    return d > 0.0f ? __float_as_uint(d) : 0u;
}

// exp(-x) for x >= 0 as a fixed-point number with 40 fractional bits, 2^40
// for x = 0 and 0 from about x = 28 on. The same polynomial as `weight_fixed`
// in crates/sampler/src/arith.rs. `key` is the bit pattern of x.
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
static __device__ __forceinline__ uint32_t ares_sample_philox(uint32_t c0, uint32_t c1, uint32_t c2,
                                                              uint32_t k0, uint32_t k1) {
    uint32_t c3 = 0;
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

// A uniform number in (0, 1) from a word of the generator.
static __device__ __forceinline__ float ares_sample_unit(uint32_t x) {
    return __fmul_rn(__fadd_rn((float)(x >> 9), 0.5f), 1.0f / 8388608.0f);
}

// The Gumbel noise of `token` at `position` for a seed: -ln(-ln(u)) with u
// uniform in (0, 1) from the generator above.
static __device__ __forceinline__ float ares_sample_gumbel(uint32_t seed_lo, uint32_t seed_hi,
                                                           uint32_t position, uint32_t token) {
    const float e =
        -ares_sample_ln(ares_sample_unit(ares_sample_philox(token, position, 0, seed_lo, seed_hi)));
    return -ares_sample_ln(e);
}

// The uniform number that decides whether the XTC cut applies.
static __device__ __forceinline__ float ares_sample_xtc_draw(uint32_t seed_lo, uint32_t seed_hi,
                                                             uint32_t position) {
    return ares_sample_unit(ares_sample_philox(0, position, 1, seed_lo, seed_hi));
}

// --- double precision --------------------------------------------------------

#define ARES_LN2_HI 6.93147180369123816490e-01
#define ARES_LN2_LO 1.90821492927058770002e-10

// exp(x) for x between -700 and 700.
static __device__ __forceinline__ double ares_exp_d(double x) {
    x = x < -700.0 ? -700.0 : (x > 700.0 ? 700.0 : x);
    const double k = floor(__dadd_rn(__dmul_rn(x, 1.44269504088896340736), 0.5));
    const double r = __dsub_rn(__dsub_rn(x, __dmul_rn(k, ARES_LN2_HI)), __dmul_rn(k, ARES_LN2_LO));
    double p = 1.0 / 6227020800.0;
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 479001600.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 39916800.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 3628800.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 362880.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 40320.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 5040.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 720.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 120.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 24.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0 / 6.0);
    p = __dadd_rn(__dmul_rn(p, r), 0.5);
    p = __dadd_rn(__dmul_rn(p, r), 1.0);
    p = __dadd_rn(__dmul_rn(p, r), 1.0);
    return __dmul_rn(p, __longlong_as_double(((long long)k + 1023ll) << 52));
}

// ln(x) for a positive normal x.
static __device__ __forceinline__ double ares_ln_d(double x) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(x);
    long long e = (long long)((b >> 52) & 0x7ffull) - 1023ll;
    const unsigned long long mant = b & 0x000fffffffffffffull;
    double m;
    if (mant > 0x0006a09e667f3bcdull) {
        e += 1;
        m = __longlong_as_double((long long)(mant | 0x3fe0000000000000ull));
    } else {
        m = __longlong_as_double((long long)(mant | 0x3ff0000000000000ull));
    }
    const double s = __ddiv_rn(__dsub_rn(m, 1.0), __dadd_rn(m, 1.0));
    const double z = __dmul_rn(s, s);
    double p = 1.0 / 23.0;
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 21.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 19.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 17.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 15.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 13.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 11.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 9.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 7.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 5.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0 / 3.0);
    p = __dadd_rn(__dmul_rn(p, z), 1.0);
    const double lm = __dmul_rn(__dmul_rn(2.0, s), p);
    return __dadd_rn(__dmul_rn((double)e, ARES_LN2_HI),
                     __dadd_rn(__dmul_rn((double)e, ARES_LN2_LO), lm));
}

// x to the power y for x >= 0.
static __device__ __forceinline__ double ares_pow_d(double x, double y) {
    if (x == 0.0) {
        return y > 0.0 ? 0.0 : (y == 0.0 ? 1.0 : __longlong_as_double(0x7ff0000000000000ll));
    }
    return ares_exp_d(__dmul_rn(y, ares_ln_d(x)));
}

// The natural logarithm of a fixed-point sum of weights (40 fractional bits).
static __device__ __forceinline__ double ares_ln_fixed(uint64_t sum) {
    return __dsub_rn(ares_ln_d(__ull2double_rn(sum)), __dmul_rn(40.0, 0.693147180559945309417232));
}

// floor(sum * q / 2^32) for a fraction q of 2^32.
static __device__ __forceinline__ uint64_t ares_sample_share(uint64_t sum, uint32_t q) {
    const uint64_t lo = (sum & 0xffffffffull) * q;
    const uint64_t hi = (sum >> 32) * q;
    return hi + (lo >> 32);
}

// --- the program and the distances ----------------------------------------------

static __device__ __forceinline__ uint32_t ares_stage_at(const AresSampleParams& p, uint32_t i) {
    if (i >= ARES_SAMPLE_MAX_STAGES) {
        return 0;
    }
    const uint32_t word = i < 8 ? p.program_lo : p.program_hi;
    return (word >> (4 * (i & 7))) & 15u;
}

static __device__ __forceinline__ bool ares_in_set(const AresSampleState& s, uint32_t key) {
    return key >= s.lo && key <= s.hi;
}

// The distance of a token below the best token of the set, and the same
// scaled by the temperature.
static __device__ __forceinline__ void ares_sample_x(const AresSampleState& s, uint32_t key,
                                                     float* d, float* x) {
    *d = __fsub_rn(__uint_as_float(key), __uint_as_float(s.lo));
    *x = __fmul_rn(*d, s.inv_t);
}

// The terms one token adds to the sums of a pass: see `Sums::of` in
// crates/sampler/src/arith.rs.
static __device__ __forceinline__ void ares_sample_terms(float d, float x, unsigned long long* t) {
    const uint64_t w = ares_sample_weight(__float_as_uint(x));
    const float dc = fminf(d, 4096.0f);
    t[0] = 1;
    t[1] = w;
    t[2] = (unsigned long long)__fmul_rn(__ull2float_rn(w), x);
    t[3] = (unsigned long long)__fmul_rn(dc, 16777216.0f);
    t[4] = (unsigned long long)__fmul_rn(__fmul_rn(dc, dc), 65536.0f);
}

// --- reductions --------------------------------------------------------------------

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

// The sum of `v` over the 32 lanes of a warp, returned to every lane.
static __device__ __forceinline__ unsigned long long ares_warp_sum_u64(unsigned long long v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// The sum of `v` over all threads of the block, returned to every thread.
static __device__ __forceinline__ unsigned long long ares_block_sum_u64(unsigned long long v,
                                                                        unsigned long long* partial) {
    v = ares_warp_sum_u64(v);
    const uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        partial[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    const uint32_t n_warps = blockDim.x >> 5;
    return ares_warp_sum_u64(lane < n_warps ? partial[lane] : 0ull);
}
