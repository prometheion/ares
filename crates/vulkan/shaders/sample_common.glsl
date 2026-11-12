// Shared declarations of the sampler kernels (`sample_*`, `draft_*`): the records, the
// arithmetic, and the block reductions. This is `crates/cuda/kernels/sampler_math.cuh` for
// GLSL, and the arithmetic is the same operation for operation, so that a pick is the same on
// either backend and the host can repeat it bit for bit (`ares_cuda::sample_reference`):
//
// - Floating point is IEEE operations in round-to-nearest that the compiler may not fuse
//   (`precise`), the division is `div_rn`, and the exponential and the logarithm are the
//   polynomials of the specification, not `exp` and `log`.
// - Everything that is summed over tokens is an integer, so the order of the sum (invocations,
//   workgroups, devices) does not matter.
// - Special values are tested on the bits: a shader compiler may assume no NaN or infinity.
#ifndef ARES_SAMPLE_COMMON
#define ARES_SAMPLE_COMMON

#include "layer_common.glsl"

// The candidates kept per row: the best tokens by logit.
const uint CANDIDATES = 8u;
// Invocations per workgroup of every sampler kernel that reduces.
const uint THREADS = 256u;
// Bins of a level of the radix descent, at most.
const uint BINS = 2048u;
// Words per bin of a histogram: the count, and the sum of the weights in three limbs of 14 bits.
const uint BIN_WORDS = 4u;
// Entries per chunk in the list `sample_top` writes: the candidates and one word of flags.
const uint LIST = CANDIDATES + 1u;
// f32 values per device and row in the exchange of the candidates.
const uint TOP_VALUES = 2u * CANDIDATES + 2u;
// f32 values per device and row in the exchange of the picks.
const uint PICK_VALUES = 2u;

const uint F_SAMPLE = 1u;
const uint F_TOP_K = 2u;
const uint F_TOP_P = 4u;
const uint S_NONFINITE = 1u;
const uint S_EMPTY = 2u;
// The largest key a token may have and still be picked.
const uint KEY_MAX = 0x7f7fffffu;
const uint DESCENT_TOP_K = 0u;
const uint DESCENT_TOP_P = 1u;

const float NEG_INF = uintBitsToFloat(0xff800000u);

// What the host sets per row (32 bytes).
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ParamsRef {
    uint flags;
    float inv_temperature;
    uint top_k;
    uint top_p_q32;
    uint min_p_key;
    uint seed_lo;
    uint seed_hi;
    uint position;
};

// A row's state between the kernels of one pick (64 bytes).
layout(buffer_reference, std430, buffer_reference_align = 16) buffer StateRef {
    float l_max;
    uint status;
    uint prefix_k;
    uint rem_k;
    uint64_t z_k;
    uint prefix_p;
    uint cut_key;
    uint64_t rem_p;
    uint64_t best;
    uint64_t reserved;
};

// What reaches the host per row (80 bytes).
layout(buffer_reference, std430, buffer_reference_align = 16) buffer OutRef {
    uint token;
    uint status;
    float value;
    uint reserved;
    uint cand_token[CANDIDATES];
    float cand_logit[CANDIDATES];
};

layout(buffer_reference, std430, buffer_reference_align = 8) buffer U64Ref { uint64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer U32Ref { uint v[]; };

const uint64_t STATE_BYTES = 64ul;
const uint64_t PARAMS_BYTES = 32ul;
const uint64_t OUT_BYTES = 80ul;

ParamsRef params_of(uint64_t params, uint r) { return ParamsRef(params + PARAMS_BYTES * r); }
StateRef state_of(uint64_t state, uint r) { return StateRef(state + STATE_BYTES * r); }
OutRef out_of(uint64_t out_, uint r) { return OutRef(out_ + OUT_BYTES * r); }

// A float as an unsigned number with the same order (minus infinity lowest).
uint sortable(float v) {
    const uint b = floatBitsToUint(v);
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

float unsortable(uint s) {
    return uintBitsToFloat((s & 0x80000000u) != 0u ? (s & 0x7fffffffu) : ~s);
}

// A value and its token as one number whose order is "higher value first, lower token on a
// tie". Zero is below every packed pair and means "none".
uint64_t pack_pick(float value, uint token) {
    return (uint64_t(sortable(value)) << 32) | uint64_t(0xffffffffu - token);
}

// A logit as the sampler sees it: negative zero is positive zero (so that equal logits have
// equal bits), and a NaN or plus infinity is minus infinity, with `bad` set.
float canonical_logit(float x, out bool bad) {
    uint b = floatBitsToUint(x);
    bad = (b & 0x7f800000u) == 0x7f800000u && b != 0xff800000u;
    if (bad) {
        b = 0xff800000u;
    } else if (b == 0x80000000u) {
        b = 0u;
    }
    return uintBitsToFloat(b);
}

// The key of a token: the bits of its distance below the best logit, which order like the
// distance itself. Smaller key, larger probability.
uint sample_key(float l_max, float logit) {
    precise float d = l_max - logit;
    return d > 0.0 ? floatBitsToUint(d) : 0u;
}

// exp(-d) for d >= 0 (the distance with bit pattern `key`) as a fixed-point number with 40
// fractional bits, 2^40 for d = 0 and 0 from about d = 28 on: the Taylor series to r^9 of
// exp(-r), r in [0, ln 2), by Horner, shifted right by the integer part of d / ln 2.
uint64_t sample_weight(uint key) {
    const float d = uintBitsToFloat(key);
    precise float t = d * 1.44269504088896340736;
    if (!(t < 41.0)) {
        return 0ul;
    }
    const uint n = uint(t);
    precise float f = t - float(n);
    precise float r = f * 0.693147180559945309417;
    float p = -1.0 / 362880.0;
    const float coef[9] = float[9](
        1.0 / 40320.0, -1.0 / 5040.0, 1.0 / 720.0, -1.0 / 120.0, 1.0 / 24.0, -1.0 / 6.0, 0.5,
        -1.0, 1.0);
    [[unroll]] for (uint i = 0u; i < 9u; ++i) {
        precise float scaled = p * r;
        precise float next = scaled + coef[i];
        p = next;
    }
    precise float fixed_point = p * 1099511627776.0;
    return uint64_t(fixed_point) >> n;
}

// ln(x) for a positive normal x.
float sample_ln(float x) {
    uint b = floatBitsToUint(x);
    int e = int(b >> 23) - 127;
    const uint mant = b & 0x007fffffu;
    // Bring the mantissa into [sqrt(1/2), sqrt(2)).
    if (mant > 0x003504f3u) {
        e += 1;
        b = mant | 0x3f000000u;
    } else {
        b = mant | 0x3f800000u;
    }
    const float m = uintBitsToFloat(b);
    precise float num = m - 1.0;
    precise float den = m + 1.0;
    const float s = div_rn(num, den);
    precise float z = s * s;
    // 2 atanh(s) = 2 s (1 + z/3 + z^2/5 + z^3/7 + z^4/9).
    float p = 1.0 / 9.0;
    const float coef[4] = float[4](1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        precise float scaled = p * z;
        precise float next = scaled + coef[i];
        p = next;
    }
    precise float two_s = 2.0 * s;
    precise float lm = two_s * p;
    precise float whole = float(e) * 0.693147180559945309417;
    precise float result = whole + lm;
    return result;
}

// Philox 4x32 with 10 rounds: a counter-based generator, so the number for a counter does not
// depend on any other number drawn. Returns the first output word.
uint sample_philox(uint c0, uint c1, uint k0, uint k1) {
    uint c2 = 0u, c3 = 0u;
    for (uint round = 0u; round < 10u; ++round) {
        uint hi0, lo0, hi1, lo1;
        umulExtended(0xD2511F53u, c0, hi0, lo0);
        umulExtended(0xCD9E8D57u, c2, hi1, lo1);
        const uint n0 = hi1 ^ c1 ^ k0;
        const uint n1 = lo1;
        const uint n2 = hi0 ^ c3 ^ k1;
        const uint n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}

// The Gumbel noise of `token` at `position` for a seed: -ln(-ln(u)) with u uniform in (0, 1)
// from the generator above.
float sample_gumbel(uint seed_lo, uint seed_hi, uint position, uint token) {
    const uint x = sample_philox(token, position, seed_lo, seed_hi);
    precise float biased = float(x >> 9) + 0.5;
    precise float u = biased * (1.0 / 8388608.0);
    const float e = -sample_ln(u);
    return -sample_ln(e);
}

// --- reductions over 64-bit numbers --------------------------------------------------------
//
// A subgroup operation on a 64-bit number needs a feature not every device has, so the maximum
// is taken in two steps on 32-bit halves: the highest high word, then the highest low word
// among those that have it.

uint64_t cluster_max_u64(uint64_t v) {
    const uint hi = uint(v >> 32), lo = uint(v);
    const uint max_hi = subgroupClusteredMax(hi, 32);
    const uint max_lo = subgroupClusteredMax(hi == max_hi ? lo : 0u, 32);
    return (uint64_t(max_hi) << 32) | uint64_t(max_lo);
}

// One partial per cluster of the workgroup (at most 8 clusters of 256 invocations).
shared uint64_t block_partial[THREADS / 32u];

// The largest `v` of all invocations of the workgroup, returned to every invocation. A caller
// that calls it again must `barrier()` in between: the partials are reused.
uint64_t block_max_u64(uint64_t v) {
    v = cluster_max_u64(v);
    const uint lane = gl_SubgroupInvocationID & 31u;
    if (lane == 0u) {
        block_partial[gl_LocalInvocationID.x >> 5] = v;
    }
    barrier();
    const uint n_warps = THREADS >> 5;
    return cluster_max_u64(lane < n_warps ? block_partial[lane] : 0ul);
}

#endif
