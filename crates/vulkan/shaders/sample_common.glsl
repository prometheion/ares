// Shared declarations of the sampler kernels (`sample_*`, `lp_*`): the records, the arithmetic,
// the stages that need no pass, and the block reductions. This is
// `crates/cuda/kernels/sampler_math.cuh` and `sampler_stage.cuh` for GLSL, operation for
// operation, so that a pick is the same on either backend and the host repeats it bit for bit
// (`ares_sampler::arith`, `ares_sampler::sample_reference`):
//
// - Floating point is IEEE operations in round-to-nearest that the compiler may not fuse
//   (`precise`: the helpers `fadd`, `fmul`, ... and `dadd`, `dmul`, ... below), the division is
//   correctly rounded by a fused correction, and the exponential and the logarithm are the
//   polynomials of the specification, not `exp` and `log`.
// - Everything that is summed over tokens is an integer, so the order of the sum (invocations,
//   workgroups, devices) does not matter.
// - Special values are tested on the bits: a shader compiler may assume no NaN or infinity.
//
// The double precision functions need `shaderFloat64`; the host refuses a device without it
// (`SamplerKernels::load`).
#ifndef ARES_SAMPLE_COMMON
#define ARES_SAMPLE_COMMON

#include "common.glsl"

// The candidates kept per row: the best tokens by logit.
const uint CANDIDATES = 8u;
// The alternatives a row reports next to the token it picked.
const uint LOGPROBS_MAX = 20u;
// Invocations per workgroup of every sampler kernel that reduces.
const uint THREADS = 256u;
// Bins of a level of the radix descent, at most.
const uint BINS = 2048u;
// Words per bin of a histogram: the count, and the sum of the weights in three limbs of 14 bits.
const uint BIN_WORDS = 4u;
// f32 values per device and row in the exchange of a pass.
const uint PASS_VALUES = 24u;
const uint MAX_STAGES = 16u;
// The tokens mirostat 1 estimates its exponent from.
const uint SAMPLE_M = 100u;
// f32 values per device and row in the exchange of the picks.
const uint PICK_VALUES = 2u;
// f32 values per device and row in the exchange of the first hundred keys of mirostat 1.
const uint GATHER_VALUES = 1u + 2u * SAMPLE_M;

// Row flags (`Params::flags`).
const uint F_SAMPLE = 1u;
const uint F_MASK = 2u;
const uint F_LP_RAW = 4u;
const uint F_LP_POST = 8u;
const uint F_MIROSTAT = 16u;

// Row status (`Out::status`).
const uint S_NONFINITE = 1u;
const uint S_EMPTY = 2u;

// The stages of a row's program (four bits each).
const uint STAGE_NSIGMA = 1u;
const uint STAGE_TOPK = 2u;
const uint STAGE_TYPICAL = 3u;
const uint STAGE_TOPP = 4u;
const uint STAGE_MINP = 5u;
const uint STAGE_XTC = 6u;
const uint STAGE_TEMP = 7u;
const uint STAGE_MIRO1 = 8u;
const uint STAGE_MIRO2 = 9u;

// The largest key a token may have and still be picked: the largest finite distance from the
// best logit. A logit of minus infinity has the key of plus infinity, one above.
const uint KEY_MAX = 0x7f7fffffu;

// The smallest temperature that scales the logits.
const float TEMPERATURE_MIN = 1e-6;

// A built-in call is not a constant expression, so this cannot be a global
// const: strict glslang rejects it. A macro is evaluated where it is used.
#define NEG_INF uintBitsToFloat(0xff800000u)

// What the host sets per row (128 bytes, `ares_sampler::SampleRow`).
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Params {
    uint flags;
    uint seed_lo;
    uint seed_hi;
    uint position;
    // The stages, four bits each: the first eight, the last eight.
    uint program_lo;
    uint program_hi;
    uint top_k;
    // Shares as fractions of 2^32.
    uint top_p_q32;
    // -ln(min_p).
    float min_p_c;
    uint typical_q32;
    float n_sigma;
    float xtc_prob;
    uint xtc_thr_q32;
    float temp;
    // 1 / temp, or 0 when temp is below the smallest temperature.
    float inv_temp;
    float dyn_delta;
    float dyn_exp;
    float mu;
    uint vocab;
    uint entries_first;
    uint entries_count;
    // The mask of the row, or 0xffffffff.
    uint mask_slot;
    uint logprobs;
    uint reserved[9];
};

// A correction of one token's logit (16 bytes, `ares_sampler::Entry`). After the correction
// the device leaves the logit it found in `bias`, so that the raw logit of a token is known.
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Entry {
    uint token;
    float bias;
    float rep;
    float sub;
};

// A row's state between the kernels of one pick (128 bytes).
layout(buffer_reference, std430, buffer_reference_align = 16) buffer State {
    float l_max;
    uint status;
    // The tokens still in the running have a key from lo to hi.
    uint lo;
    uint hi;
    float inv_t;
    // The next position of the program to run.
    uint cursor;
    // The k of the top-k that runs (a mirostat 1 sets it).
    uint k;
    // The radix descent.
    uint prefix;
    uint64_t rem;
    // The result of the last descent and the target of the scan after it.
    uint cut;
    uint cut100;
    uint64_t target;
    // The sums of a pass after the devices added them up.
    uint64_t sums[5];
    float mean_x;
    uint n_gather;
    // The raw logits' best, for probabilities of the model's own softmax.
    float raw_lmax;
    // The logarithm of the weight of all tokens, for the model's own probabilities.
    double raw_lnz;
    uint reserved[2];
};

// What reaches the host per row (80 bytes, `ares_sampler::SamplePick`).
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Out {
    uint token;
    uint status;
    // The logit of a greedy pick, the score (logit * inv_t + noise) of a sampled one.
    float value;
    // The surprise of a mirostat row's pick, in bits.
    float aux;
    uint cand_token[CANDIDATES];
    float cand_logit[CANDIDATES];
};

// The probabilities a row reports (168 bytes, `ares_sampler::LogprobOut`).
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Logprobs {
    uint n;
    float chosen;
    uint token[LOGPROBS_MAX];
    float logprob[LOGPROBS_MAX];
};

layout(buffer_reference, std430, buffer_reference_align = 8) buffer U64Ref { uint64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer UintRef { uint v[]; };

const uint64_t PARAMS_BYTES = 128ul;
const uint64_t STATE_BYTES = 128ul;
const uint64_t ENTRY_BYTES = 16ul;
const uint64_t OUT_BYTES = 80ul;
const uint64_t LOGPROBS_BYTES = 168ul;

Params params_of(uint64_t base, uint r) { return Params(base + PARAMS_BYTES * r); }
State state_of(uint64_t base, uint r) { return State(base + STATE_BYTES * r); }
Out out_of(uint64_t base, uint r) { return Out(base + OUT_BYTES * r); }
Logprobs logprobs_of(uint64_t base, uint r) { return Logprobs(base + LOGPROBS_BYTES * r); }

float ldf(uint64_t a) { return FloatRefRO(a).v[0]; }
uint ldu(uint64_t a) { return WordRef(a).v[0]; }

// The token of column `i` of a device's slice: `tokens[first + i]`, or `first + i` without a
// token list.
uint token_of(uint64_t tokens, uint first, uint i) {
    return tokens != 0ul ? ldu(tokens + 4ul * (first + i)) : first + i;
}

// --- IEEE operations that are not fused ---------------------------------------------------

float fadd(float a, float b) { precise float r = a + b; return r; }
float fsub(float a, float b) { precise float r = a - b; return r; }
float fmul(float a, float b) { precise float r = a * b; return r; }
// a / b correctly rounded (the division of GLSL may be 2.5 ulp off).
float fdiv(float a, float b) {
    precise float r = 1.0 / b;
    precise float e = fma(-b, r, 1.0);
    r = fma(e, r, r);
    precise float q = a * r;
    precise float rem = fma(-q, b, a);
    return fma(rem, r, q);
}
double dadd(double a, double b) { precise double r = a + b; return r; }
double dsub(double a, double b) { precise double r = a - b; return r; }
double dmul(double a, double b) { precise double r = a * b; return r; }
double ddiv(double a, double b) {
    precise double r = 1.0LF / b;
    precise double e = fma(-b, r, 1.0LF);
    r = fma(e, r, r);
    precise double q = a * r;
    precise double rem = fma(-q, b, a);
    return fma(rem, r, q);
}

// --- the keys of tokens ---------------------------------------------------------------------

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

// A logit as the sampler sees it: negative zero made positive, and a NaN or plus infinity
// (reported through `bad`) made minus infinity, which no stage keeps.
float sample_logit(float v, out bool bad) {
    uint b = floatBitsToUint(v);
    bad = (b & 0x7f800000u) == 0x7f800000u && b != 0xff800000u;
    if (bad) {
        b = 0xff800000u;
    } else if (b == 0x80000000u) {
        b = 0u;
    }
    return uintBitsToFloat(b);
}

bool present(float l) { return l > -3.402823466e+38; }

// The key of a token: the bits of its distance below the best logit, which order like the
// distance itself. Smaller key, larger probability.
uint sample_key(float l_max, float logit) {
    const float d = fsub(l_max, logit);
    return d > 0.0 ? floatBitsToUint(d) : 0u;
}

// exp(-x) for x >= 0 as a fixed-point number with 40 fractional bits, 2^40 for x = 0 and 0 from
// about x = 28 on. The same polynomial as `weight_fixed` in crates/sampler/src/arith.rs. `key`
// is the bit pattern of x.
uint64_t sample_weight(uint key) {
    const float d = uintBitsToFloat(key);
    const float t = fmul(d, 1.44269504088896340736);
    if (!(t < 41.0)) {
        return 0ul;
    }
    const uint n = uint(t);
    const float f = fsub(t, float(n));
    const float r = fmul(f, 0.693147180559945309417);
    // exp(-r) for r in [0, ln 2): the Taylor series to r^9, by Horner.
    float p = -1.0 / 362880.0;
    const float coef[9] = float[9](
        1.0 / 40320.0, -1.0 / 5040.0, 1.0 / 720.0, -1.0 / 120.0, 1.0 / 24.0, -1.0 / 6.0, 0.5,
        -1.0, 1.0);
    [[unroll]] for (uint i = 0u; i < 9u; ++i) {
        p = fadd(fmul(p, r), coef[i]);
    }
    return uint64_t(fmul(p, 1099511627776.0)) >> n;
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
    const float s = fdiv(fsub(m, 1.0), fadd(m, 1.0));
    const float z = fmul(s, s);
    // 2 atanh(s) = 2 s (1 + z/3 + z^2/5 + z^3/7 + z^4/9).
    float p = 1.0 / 9.0;
    const float coef[4] = float[4](1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0);
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        p = fadd(fmul(p, z), coef[i]);
    }
    const float lm = fmul(fmul(2.0, s), p);
    return fadd(fmul(float(e), 0.693147180559945309417), lm);
}

// Philox 4x32 with 10 rounds (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3"): a
// counter-based generator, so the number for a counter does not depend on any other number
// drawn. Returns the first output word.
uint sample_philox(uint c0, uint c1, uint c2, uint k0, uint k1) {
    uint c3 = 0u;
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

// A uniform number in (0, 1) from a word of the generator.
float sample_unit(uint x) {
    return fmul(fadd(float(x >> 9), 0.5), 1.0 / 8388608.0);
}

// The Gumbel noise of `token` at `position` for a seed: -ln(-ln(u)) with u uniform in (0, 1)
// from the generator above.
float sample_gumbel(uint seed_lo, uint seed_hi, uint position, uint token) {
    const float e = -sample_ln(sample_unit(sample_philox(token, position, 0u, seed_lo, seed_hi)));
    return -sample_ln(e);
}

// The uniform number that decides whether the XTC cut applies.
float sample_xtc_draw(uint seed_lo, uint seed_hi, uint position) {
    return sample_unit(sample_philox(0u, position, 1u, seed_lo, seed_hi));
}

// --- double precision ------------------------------------------------------------------------

const double LN2_HI = 6.93147180369123816490e-01LF;
const double LN2_LO = 1.90821492927058770002e-10LF;
const double LN2 = 0.693147180559945309417232LF;

// The double whose bits are `v`.
double double_of(uint64_t v) { return packDouble2x32(uvec2(uint(v), uint(v >> 32))); }

uint64_t bits_of(double x) {
    const uvec2 w = unpackDouble2x32(x);
    return (uint64_t(w.y) << 32) | uint64_t(w.x);
}

// exp(x) for x between -700 and 700.
double sample_exp_d(double x) {
    x = x < -700.0LF ? -700.0LF : (x > 700.0LF ? 700.0LF : x);
    const double k = floor(dadd(dmul(x, 1.44269504088896340736LF), 0.5LF));
    const double r = dsub(dsub(x, dmul(k, LN2_HI)), dmul(k, LN2_LO));
    double p = 1.0LF / 6227020800.0LF;
    const double coef[13] = double[13](
        1.0LF / 479001600.0LF, 1.0LF / 39916800.0LF, 1.0LF / 3628800.0LF, 1.0LF / 362880.0LF,
        1.0LF / 40320.0LF, 1.0LF / 5040.0LF, 1.0LF / 720.0LF, 1.0LF / 120.0LF, 1.0LF / 24.0LF,
        1.0LF / 6.0LF, 0.5LF, 1.0LF, 1.0LF);
    [[unroll]] for (uint i = 0u; i < 13u; ++i) {
        p = dadd(dmul(p, r), coef[i]);
    }
    return dmul(p, packDouble2x32(uvec2(0u, uint(int(k) + 1023) << 20)));
}

// ln(x) for a positive normal x.
double sample_ln_d(double x) {
    const uint64_t b = bits_of(x);
    int64_t e = int64_t((b >> 52) & 0x7ffUL) - 1023L;
    const uint64_t mant = b & 0x000fffffffffffffUL;
    double m;
    if (mant > 0x0006a09e667f3bcdUL) {
        e += 1L;
        m = double_of(mant | 0x3fe0000000000000UL);
    } else {
        m = double_of(mant | 0x3ff0000000000000UL);
    }
    const double s = ddiv(dsub(m, 1.0LF), dadd(m, 1.0LF));
    const double z = dmul(s, s);
    double p = 1.0LF / 23.0LF;
    const double coef[11] = double[11](
        1.0LF / 21.0LF, 1.0LF / 19.0LF, 1.0LF / 17.0LF, 1.0LF / 15.0LF, 1.0LF / 13.0LF,
        1.0LF / 11.0LF, 1.0LF / 9.0LF, 1.0LF / 7.0LF, 1.0LF / 5.0LF, 1.0LF / 3.0LF, 1.0LF);
    [[unroll]] for (uint i = 0u; i < 11u; ++i) {
        p = dadd(dmul(p, z), coef[i]);
    }
    const double lm = dmul(dmul(2.0LF, s), p);
    const double ed = double(e);
    return dadd(dmul(ed, LN2_HI), dadd(dmul(ed, LN2_LO), lm));
}

// x to the power y for x >= 0.
double sample_pow_d(double x, double y) {
    if (bits_of(x) << 1 == 0ul) {
        return y > 0.0LF ? 0.0LF : (y == 0.0LF ? 1.0LF : double_of(0x7ff0000000000000UL));
    }
    return sample_exp_d(dmul(y, sample_ln_d(x)));
}

// The natural logarithm of a fixed-point sum of weights (40 fractional bits).
double sample_ln_fixed(uint64_t sum) {
    return dsub(sample_ln_d(double(sum)), dmul(40.0LF, LN2));
}

// floor(sum * q / 2^32) for a fraction q of 2^32.
uint64_t sample_share(uint64_t sum, uint q) {
    const uint64_t lo = (sum & 0xffffffffUL) * uint64_t(q);
    const uint64_t hi = (sum >> 32) * uint64_t(q);
    return hi + (lo >> 32);
}

// --- the program and the distances ----------------------------------------------------------

uint stage_at(Params p, uint i) {
    if (i >= MAX_STAGES) {
        return 0u;
    }
    const uint word = i < 8u ? p.program_lo : p.program_hi;
    return (word >> (4u * (i & 7u))) & 15u;
}

bool in_set(State s, uint key) { return key >= s.lo && key <= s.hi; }

// The distance of a token below the best token of the set, and the same scaled by the
// temperature.
void sample_x(State s, uint key, out float d, out float x) {
    d = fsub(uintBitsToFloat(key), uintBitsToFloat(s.lo));
    x = fmul(d, s.inv_t);
}

// The terms one token adds to the sums of a pass: see `Sums::of` in
// crates/sampler/src/arith.rs.
void sample_terms(float d, float x, out uint64_t t[5]) {
    const uint64_t w = sample_weight(floatBitsToUint(x));
    const float dc = min(d, 4096.0);
    t[0] = 1ul;
    t[1] = w;
    t[2] = uint64_t(fmul(float(w), x));
    t[3] = uint64_t(fmul(dc, 16777216.0));
    t[4] = uint64_t(fmul(fmul(dc, dc), 65536.0));
}

// --- the stages that need no pass over the row -----------------------------------------------
//
// A row's state has a cursor, the next position of its program. A stage that only changes the
// set or the temperature from numbers the state holds (min-p, a fixed temperature) runs as soon
// as the stage before it is done, in the kernel that finishes that one, so that it costs no
// launch.

// Runs the stages from the cursor on that need no pass over the row, up to the first that does.
void sample_advance(Params p, State s) {
    for (;;) {
        const uint stage = stage_at(p, s.cursor);
        if (stage == STAGE_MINP) {
            // Keeps the tokens at least min_p times as probable as the best of the set:
            // d' <= -ln(min_p) / inv_t.
            const float lim = fadd(uintBitsToFloat(s.lo), fdiv(p.min_p_c, s.inv_t));
            s.hi = min(s.hi, min(floatBitsToUint(lim), KEY_MAX));
        } else if (stage == STAGE_TEMP && !(p.dyn_delta > 0.0)) {
            if (p.temp < TEMPERATURE_MIN) {
                s.hi = s.lo;
            } else {
                s.inv_t = fmul(s.inv_t, p.inv_temp);
            }
        } else {
            return;
        }
        s.cursor += 1u;
    }
}

// The stage at `position` is done: the next one runs.
void sample_stage_done(Params p, State s, uint position) {
    s.cursor = position + 1u;
    sample_advance(p, s);
}

// Whether a sampled row, with something left to pick, is at `position` of its program with the
// stage `kind` there.
bool stage_runs(Params p, State s, uint position, uint kind) {
    return (p.flags & F_SAMPLE) != 0u && (s.status & S_EMPTY) == 0u && s.cursor == position
        && stage_at(p, position) == kind;
}

// Whether the XTC cut applies at this row's position: its uniform draw is at most the
// probability.
bool xtc_applies(Params p) {
    return sample_xtc_draw(p.seed_lo, p.seed_hi, p.position) <= p.xtc_prob;
}

// --- reductions ---------------------------------------------------------------------------------
//
// A workgroup of THREADS invocations reduces in two steps: each subgroup by subgroup operations,
// then the subgroups' results through shared memory, one value per subgroup (a subgroup has at
// least 8 invocations, so at most THREADS / 8 = 32 of them). A subgroup operation on a 64-bit
// number needs a feature not every device has, so the maximum is taken in two steps on 32-bit
// halves and the sum in four limbs of 16 bits.

shared uint64_t block_partial[THREADS / 8u];

uint64_t subgroup_max_u64(uint64_t v) {
    const uint hi = uint(v >> 32), lo = uint(v);
    const uint max_hi = subgroupMax(hi);
    const uint max_lo = subgroupMax(hi == max_hi ? lo : 0u);
    return (uint64_t(max_hi) << 32) | uint64_t(max_lo);
}

// The largest `v` of all invocations of the workgroup, returned to every invocation. A caller
// that calls a reduction again must `barrier()` in between: the partials are reused.
uint64_t block_max_u64(uint64_t v) {
    v = subgroup_max_u64(v);
    if (subgroupElect()) {
        block_partial[gl_SubgroupID] = v;
    }
    barrier();
    uint64_t best = 0ul;
    for (uint i = 0u; i < gl_NumSubgroups; ++i) {
        best = max(best, block_partial[i]);
    }
    return best;
}

uint64_t subgroup_sum_u64(uint64_t v) {
    uint64_t total = 0ul;
    [[unroll]] for (uint k = 0u; k < 4u; ++k) {
        total += uint64_t(subgroupAdd(uint(v >> (16u * k)) & 0xffffu)) << (16u * k);
    }
    return total;
}

// The sum of `v` over all invocations of the workgroup, returned to every invocation (modulo
// 2^64); the same rule for a second call as above.
uint64_t block_sum_u64(uint64_t v) {
    v = subgroup_sum_u64(v);
    if (subgroupElect()) {
        block_partial[gl_SubgroupID] = v;
    }
    barrier();
    uint64_t total = 0ul;
    for (uint i = 0u; i < gl_NumSubgroups; ++i) {
        total += block_partial[i];
    }
    return total;
}

#endif
