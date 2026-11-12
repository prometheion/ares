// The stages that need numbers about the row's set: its sums (top-n-sigma,
// typical sampling, XTC, a dynamic temperature, mirostat), the tokens above a
// threshold (XTC), the keys inside a radius (typical sampling) and the first
// hundred keys (mirostat 1).
//
// A stage is a pass over the row (`ares_sample_pass`), the export of what the
// blocks accumulated into the exchange buffer (`ares_sample_pass_export`), the
// sum across devices, and a kernel that reads the total and moves the row's
// state (`ares_sample_stage_decide`). Every number a device sends is an
// integer split into four limbs of 16 bits, so the sums of up to 16 devices
// are exact in f32.
#pragma once

#include "sampler_stage.cuh"

#define ARES_PASS_STATS 0u
#define ARES_PASS_XTC 1u
#define ARES_PASS_TYPICAL 2u
#define ARES_PASS_GATHER 3u
#define ARES_PASS_TAIL 4u
#define ARES_PASS_RAW 5u

// What a decide kernel does with the totals of a pass.
#define ARES_PHASE_NSIGMA 0u
#define ARES_PHASE_TYPICAL_MEAN 1u
#define ARES_PHASE_TYPICAL_SET 2u
#define ARES_PHASE_XTC_TARGET 3u
#define ARES_PHASE_XTC_CUT 4u
#define ARES_PHASE_TEMP 5u
#define ARES_PHASE_MIRO2 6u
#define ARES_PHASE_MIRO1 7u
#define ARES_PHASE_TAIL 8u
#define ARES_PHASE_RAW 9u

// f32 values per device and row in the exchange of the first hundred keys of
// mirostat 1: the count, and each key as two halves of 16 bits.
#define ARES_GATHER_VALUES (1 + 2 * ARES_SAMPLE_M)

// Whether a row takes part in a pass.
static __device__ __forceinline__ bool ares_pass_runs(const AresSampleParams& p,
                                                      const AresSampleState& s, uint32_t mode,
                                                      uint32_t position) {
    switch (mode) {
        case ARES_PASS_STATS: {
            if (!(p.flags & ARES_SAMPLE_F_SAMPLE) || (s.status & ARES_SAMPLE_S_EMPTY) ||
                s.cursor != position) {
                return false;
            }
            const uint32_t stage = ares_stage_at(p, position);
            return stage == ARES_STAGE_NSIGMA || stage == ARES_STAGE_TYPICAL ||
                   stage == ARES_STAGE_MIRO2 || (stage == ARES_STAGE_XTC && ares_xtc_applies(p)) ||
                   (stage == ARES_STAGE_TEMP && p.dyn_delta > 0.0f);
        }
        case ARES_PASS_XTC:
            return ares_stage_runs(p, s, position, ARES_STAGE_XTC) && ares_xtc_applies(p) &&
                   s.target != 0;
        case ARES_PASS_TYPICAL:
            return ares_stage_runs(p, s, position, ARES_STAGE_TYPICAL);
        case ARES_PASS_GATHER:
            return ares_stage_runs(p, s, position, ARES_STAGE_MIRO1);
        case ARES_PASS_TAIL:
            return (p.flags & (ARES_SAMPLE_F_LP_POST | ARES_SAMPLE_F_MIROSTAT)) &&
                   !(s.status & ARES_SAMPLE_S_EMPTY);
        default:
            return (p.flags & ARES_SAMPLE_F_LP_RAW) && ares_sample_present(s.raw_lmax);
    }
}

// Grid (chunks, rows), ARES_SAMPLE_THREADS threads, `chunk` at most
// ARES_SAMPLE_PER logits per thread. Adds the tokens of the
// set in chunk `blockIdx.x` of row `blockIdx.y` to the row's accumulators:
//
// - STATS, TAIL, RAW: the five sums of `Sums` (RAW is over all tokens at
//   temperature 1, from the model's best logit);
// - XTC: acc[0] counts the tokens whose weight reaches the target, acc[1] is
//   the largest key among them plus one;
// - TYPICAL: acc[0] is the smallest key among the tokens within the radius
//   (as 2^32 minus the key, for a maximum), acc[1] the largest plus one;
// - GATHER: the keys below the hundredth go to the row's list in `gather`.
ARES_KERNEL ares_sample_pass(const float* logits, uint32_t stride, uint32_t n_local, uint32_t chunk,
                             const AresSampleParams* params, AresSampleState* state,
                             uint32_t* gather, uint32_t gather_stride, uint32_t mode,
                             uint32_t position) {
    __shared__ unsigned long long partial[ARES_SAMPLE_THREADS / 32];
    const uint32_t r = blockIdx.y;
    const AresSampleParams p = params[r];
    const AresSampleState s = state[r];
    if (!ares_pass_runs(p, s, mode, position)) {
        return;
    }
    const uint32_t begin = blockIdx.x * chunk;
    const uint32_t end = min(begin + chunk, n_local);
    float v[ARES_SAMPLE_PER];
    const uint32_t n = ares_sample_read(logits + (size_t)r * stride, begin, end, v);
    unsigned long long t[5] = {0, 0, 0, 0, 0};
    unsigned long long min_enc = 0, max_key = 0;
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
        float d, x;
        uint32_t key;
        if (mode == ARES_PASS_RAW) {
            d = __fsub_rn(s.raw_lmax, l);
            x = d;
            key = 0;
        } else {
            key = ares_sample_key(s.l_max, l);
            if (!ares_in_set(s, key)) {
                continue;
            }
            ares_sample_x(s, key, &d, &x);
        }
        switch (mode) {
            case ARES_PASS_XTC:
                if (ares_sample_weight(__float_as_uint(x)) >= s.target) {
                    t[0] += 1;
                    max_key = max(max_key, (unsigned long long)key + 1);
                }
                break;
            case ARES_PASS_TYPICAL:
                if (__float_as_uint(fabsf(__fsub_rn(x, s.mean_x))) <= s.cut) {
                    min_enc = max(min_enc, (unsigned long long)(0xffffffffu - key) + 1);
                    max_key = max(max_key, (unsigned long long)key + 1);
                }
                break;
            case ARES_PASS_GATHER:
                if (key < s.cut100) {
                    const uint32_t at = atomicAdd(&state[r].n_gather, 1u);
                    if (at < ARES_SAMPLE_M) {
                        gather[(size_t)r * gather_stride + at] = key;
                    }
                }
                break;
            default: {
                unsigned long long terms[5];
                ares_sample_terms(d, x, terms);
                for (int k = 0; k < 5; ++k) {
                    t[k] += terms[k];
                }
            }
        }
    }
    if (mode == ARES_PASS_GATHER) {
        return;
    }
    if (mode == ARES_PASS_XTC || mode == ARES_PASS_TYPICAL) {
        const unsigned long long a =
            mode == ARES_PASS_XTC ? ares_block_sum_u64(t[0], partial) : ares_block_max_u64(min_enc, partial);
        __syncthreads();
        const unsigned long long b = ares_block_max_u64(max_key, partial);
        if (threadIdx.x == 0) {
            if (mode == ARES_PASS_XTC) {
                atomicAdd(&state[r].acc[0], a);
            } else {
                atomicMax(&state[r].acc[0], a);
            }
            atomicMax(&state[r].acc[1], b);
        }
        return;
    }
    // The five sums of every warp, then of the chunk: thread `k` adds up sum
    // `k`. A warp, and a chunk, without a token of the set has nothing to add.
    __shared__ unsigned long long warp_sums[ARES_SAMPLE_THREADS / 32][5];
    if (__ballot_sync(0xffffffffu, t[0] != 0) != 0) {
        for (int k = 0; k < 5; ++k) {
            t[k] = ares_warp_sum_u64(t[k]);
        }
    }
    if ((threadIdx.x & 31) == 0) {
        for (int k = 0; k < 5; ++k) {
            warp_sums[threadIdx.x >> 5][k] = t[k];
        }
    }
    __syncthreads();
    if (threadIdx.x < 5) {
        unsigned long long total = 0;
        for (uint32_t w = 0; w < (blockDim.x >> 5); ++w) {
            total += warp_sums[w][threadIdx.x];
        }
        if (total != 0) {
            atomicAdd(&state[r].acc[threadIdx.x], total);
        }
    }
}

// Grid (rows), 32 threads. Moves the accumulators of a pass into this
// device's slot of `exchange` (ARES_SAMPLE_PASS_VALUES values per device and
// row, four limbs per number) and zeros into the other slots, and clears the
// accumulators.
ARES_KERNEL ares_sample_pass_export(AresSampleState* state, const AresSampleParams* params,
                                    float* exchange, uint32_t devices, uint32_t rank, uint32_t mode,
                                    uint32_t position) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    if (!ares_pass_runs(params[r], state[r], mode, position)) {
        return;
    }
    const uint32_t n = (mode == ARES_PASS_XTC || mode == ARES_PASS_TYPICAL) ? 2u : 5u;
    float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_PASS_VALUES;
    for (uint32_t d = 0; d < devices; ++d) {
        float* slot = slots + d * ARES_SAMPLE_PASS_VALUES;
        for (uint32_t k = 0; k < n; ++k) {
            const unsigned long long v = d == rank ? state[r].acc[k] : 0ull;
            for (uint32_t limb = 0; limb < 4; ++limb) {
                slot[k * 4 + limb] = (float)(uint32_t)((v >> (16 * limb)) & 0xffffull);
            }
        }
    }
    for (uint32_t k = 0; k < n; ++k) {
        state[r].acc[k] = 0;
    }
}

// The number `k` of a device's slot.
static __device__ __forceinline__ unsigned long long ares_slot_number(const float* slot, uint32_t k) {
    unsigned long long v = 0;
    for (uint32_t limb = 0; limb < 4; ++limb) {
        v |= (unsigned long long)(uint32_t)slot[k * 4 + limb] << (16 * limb);
    }
    return v;
}

// The five sums, added up over the devices.
static __device__ __forceinline__ void ares_slot_sums(const float* slots, uint32_t devices,
                                                      unsigned long long* sums) {
    for (uint32_t k = 0; k < 5; ++k) {
        sums[k] = 0;
    }
    for (uint32_t d = 0; d < devices; ++d) {
        for (uint32_t k = 0; k < 5; ++k) {
            sums[k] += ares_slot_number(slots + d * ARES_SAMPLE_PASS_VALUES, k);
        }
    }
}

// The standard deviation of the distances of a set, as a float.
static __device__ __forceinline__ float ares_sigma(const unsigned long long* sums) {
    if (sums[0] == 0) {
        return 0.0f;
    }
    const double n = __ull2double_rn(sums[0]);
    const double mean = __ddiv_rn(__ddiv_rn(__ull2double_rn(sums[3]), 16777216.0), n);
    const double var =
        __dsub_rn(__ddiv_rn(__ddiv_rn(__ull2double_rn(sums[4]), 65536.0), n), __dmul_rn(mean, mean));
    return __double2float_rn(var > 0.0 ? __dsqrt_rn(var) : 0.0);
}

// The entropy of the distribution of a set, in nats.
static __device__ __forceinline__ double ares_entropy(const unsigned long long* sums) {
    if (sums[1] == 0) {
        return 0.0;
    }
    return __dadd_rn(ares_ln_fixed(sums[1]),
                     __ddiv_rn(__ull2double_rn(sums[2]), __ull2double_rn(sums[1])));
}

// Whether the token with key `key` is kept by mirostat 2 at surprise limit
// `limit` (in nats of the scaled distance).
static __device__ __forceinline__ bool ares_miro2_keeps(const AresSampleState& s, uint32_t key,
                                                        double limit) {
    float d, x;
    ares_sample_x(s, key, &d, &x);
    return (double)x <= limit;
}

// Grid (rows), 32 threads. Reads the totals of the pass before it and takes
// the row's stage one step on (see ARES_PHASE_*).
ARES_KERNEL ares_sample_stage_decide(const float* exchange, const AresSampleParams* params,
                                     AresSampleState* state, uint32_t devices, uint32_t phase,
                                     uint32_t position) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    const AresSampleParams p = params[r];
    AresSampleState* s = state + r;
    const float* slots = exchange + (size_t)r * devices * ARES_SAMPLE_PASS_VALUES;
    unsigned long long sums[5];
    switch (phase) {
        case ARES_PHASE_NSIGMA: {
            if (!ares_stage_runs(p, *s, position, ARES_STAGE_NSIGMA)) {
                return;
            }
            ares_slot_sums(slots, devices, sums);
            if (sums[0] > 1) {
                const float lim = __fadd_rn(__uint_as_float(s->lo), __fmul_rn(p.n_sigma, ares_sigma(sums)));
                s->hi = min(s->hi, min(__float_as_uint(lim), (uint32_t)ARES_SAMPLE_KEY_MAX));
            }
            ares_stage_done(p, s, position);
            return;
        }
        case ARES_PHASE_TYPICAL_MEAN: {
            if (!ares_stage_runs(p, *s, position, ARES_STAGE_TYPICAL)) {
                return;
            }
            ares_slot_sums(slots, devices, sums);
            s->mean_x = sums[1] == 0 ? 0.0f
                                     : __double2float_rn(__ddiv_rn(__ull2double_rn(sums[2]),
                                                                   __ull2double_rn(sums[1])));
            return;
        }
        case ARES_PHASE_TYPICAL_SET: {
            if (!ares_stage_runs(p, *s, position, ARES_STAGE_TYPICAL)) {
                return;
            }
            unsigned long long min_enc = 0, max_key = 0;
            for (uint32_t d = 0; d < devices; ++d) {
                const float* slot = slots + d * ARES_SAMPLE_PASS_VALUES;
                min_enc = max(min_enc, ares_slot_number(slot, 0));
                max_key = max(max_key, ares_slot_number(slot, 1));
            }
            if (max_key != 0) {
                s->lo = (uint32_t)(0x100000000ull - min_enc);
                s->hi = (uint32_t)(max_key - 1);
            }
            ares_stage_done(p, s, position);
            return;
        }
        case ARES_PHASE_XTC_TARGET: {
            if (!ares_stage_runs(p, *s, position, ARES_STAGE_XTC) || !ares_xtc_applies(p)) {
                return;
            }
            ares_slot_sums(slots, devices, sums);
            // ceil(z * thr / 2^32), at least 1.
            const uint64_t lo = (sums[1] & 0xffffffffull) * p.xtc_thr_q32;
            const uint64_t hi = (sums[1] >> 32) * p.xtc_thr_q32;
            uint64_t target = hi + (lo >> 32) + ((lo & 0xffffffffull) != 0 ? 1 : 0);
            s->target = target == 0 ? 1 : target;
            return;
        }
        case ARES_PHASE_XTC_CUT: {
            if (!ares_stage_runs(p, *s, position, ARES_STAGE_XTC)) {
                return;
            }
            if (ares_xtc_applies(p) && s->target != 0) {
                unsigned long long count = 0, max_key = 0;
                for (uint32_t d = 0; d < devices; ++d) {
                    const float* slot = slots + d * ARES_SAMPLE_PASS_VALUES;
                    count += ares_slot_number(slot, 0);
                    max_key = max(max_key, ares_slot_number(slot, 1));
                }
                if (count >= 2) {
                    s->lo = (uint32_t)(max_key - 1);
                }
                s->target = 0;
            }
            ares_stage_done(p, s, position);
            return;
        }
        case ARES_PHASE_TEMP: {
            if (!ares_stage_runs(p, *s, position, ARES_STAGE_TEMP) || !(p.dyn_delta > 0.0f)) {
                return;
            }
            ares_slot_sums(slots, devices, sums);
            if (sums[0] > 1) {
                const double t0 = (double)p.temp;
                const double delta = (double)p.dyn_delta;
                const double low = __dsub_rn(t0, delta);
                const double min_t = low > 0.0 ? low : 0.0;
                const double max_t = __dadd_rn(t0, delta);
                double normalized = __ddiv_rn(ares_entropy(sums), ares_ln_d(__ull2double_rn(sums[0])));
                normalized = normalized < 0.0 ? 0.0 : (normalized > 1.0 ? 1.0 : normalized);
                const double dynamic = __dadd_rn(
                    min_t, __dmul_rn(__dsub_rn(max_t, min_t), ares_pow_d(normalized, (double)p.dyn_exp)));
                const float t = __double2float_rn(dynamic);
                if (t < ARES_SAMPLE_TEMPERATURE_MIN) {
                    s->hi = s->lo;
                } else {
                    s->inv_t = __fmul_rn(s->inv_t, __fdiv_rn(1.0f, t));
                }
            }
            ares_stage_done(p, s, position);
            return;
        }
        case ARES_PHASE_MIRO2: {
            if (!ares_stage_runs(p, *s, position, ARES_STAGE_MIRO2)) {
                return;
            }
            ares_slot_sums(slots, devices, sums);
            const double limit = __dsub_rn(__dmul_rn((double)p.mu, 0.693147180559945309417232),
                                           ares_ln_fixed(sums[1]));
            // The largest key whose token is kept, found from where the
            // threshold falls and corrected for the rounding of the
            // multiplications: the surprise grows with the key.
            uint32_t hi = s->lo;
            if (ares_miro2_keeps(*s, s->lo, limit)) {
                const float lo_f = __uint_as_float(s->lo);
                const float guess = __fadd_rn(
                    lo_f, __double2float_rn(__ddiv_rn(limit, (double)s->inv_t)));
                hi = max(s->lo, min(__float_as_uint(guess), (uint32_t)ARES_SAMPLE_KEY_MAX));
                for (int i = 0; i < 4096 && hi < ARES_SAMPLE_KEY_MAX && ares_miro2_keeps(*s, hi + 1, limit); ++i) {
                    hi += 1;
                }
                for (int i = 0; i < 4096 && hi > s->lo && !ares_miro2_keeps(*s, hi, limit); ++i) {
                    hi -= 1;
                }
            }
            s->hi = min(s->hi, hi);
            ares_stage_done(p, s, position);
            return;
        }
        case ARES_PHASE_TAIL:
        case ARES_PHASE_RAW: {
            if (!ares_pass_runs(p, *s, phase == ARES_PHASE_TAIL ? ARES_PASS_TAIL : ARES_PASS_RAW, position)) {
                return;
            }
            ares_slot_sums(slots, devices, sums);
            if (phase == ARES_PHASE_TAIL) {
                for (int k = 0; k < 5; ++k) {
                    s->sums[k] = sums[k];
                }
            } else {
                s->raw_lnz = ares_ln_fixed(sums[1]);
            }
            return;
        }
        default:
            return;
    }
}

// Grid (rows), 32 threads. Moves the keys the row's blocks gathered into this
// device's slot of `exchange` (ARES_GATHER_VALUES values per device and row)
// and zeros into the other slots, and empties the list.
ARES_KERNEL ares_sample_gather_export(AresSampleState* state, const AresSampleParams* params,
                                      const uint32_t* gather, uint32_t gather_stride, float* exchange,
                                      uint32_t devices, uint32_t rank, uint32_t position) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    if (!ares_pass_runs(params[r], state[r], ARES_PASS_GATHER, position)) {
        return;
    }
    const uint32_t n = min(state[r].n_gather, (uint32_t)ARES_SAMPLE_M);
    float* slots = exchange + (size_t)r * devices * ARES_GATHER_VALUES;
    for (uint32_t d = 0; d < devices; ++d) {
        float* slot = slots + d * ARES_GATHER_VALUES;
        slot[0] = d == rank ? (float)n : 0.0f;
        for (uint32_t i = 0; i < ARES_SAMPLE_M; ++i) {
            const uint32_t key = (d == rank && i < n) ? gather[(size_t)r * gather_stride + i] : 0u;
            slot[1 + 2 * i] = (float)(key >> 16);
            slot[2 + 2 * i] = (float)(key & 0xffffu);
        }
    }
    state[r].n_gather = 0;
}

// Grid (rows), 32 threads. Mirostat 1: the `k` that the hundred most probable
// tokens imply. The keys below the hundredth come from all devices; the rest
// of the hundred are the hundredth key itself.
ARES_KERNEL ares_sample_miro1_decide(const float* exchange, const AresSampleParams* params,
                                     AresSampleState* state, uint32_t devices, uint32_t position) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint32_t r = blockIdx.x;
    const AresSampleParams p = params[r];
    AresSampleState* s = state + r;
    if (!ares_stage_runs(p, *s, position, ARES_STAGE_MIRO1)) {
        return;
    }
    const float* slots = exchange + (size_t)r * devices * ARES_GATHER_VALUES;
    // The keys below the hundredth, in ascending order.
    uint32_t keys[ARES_SAMPLE_M];
    uint32_t n = 0;
    for (uint32_t d = 0; d < devices; ++d) {
        const float* slot = slots + d * ARES_GATHER_VALUES;
        const uint32_t count = (uint32_t)slot[0];
        for (uint32_t i = 0; i < count && n < ARES_SAMPLE_M; ++i) {
            const uint32_t key = ((uint32_t)slot[1 + 2 * i] << 16) | (uint32_t)slot[2 + 2 * i];
            uint32_t at = n++;
            while (at > 0 && keys[at - 1] > key) {
                keys[at] = keys[at - 1];
                --at;
            }
            keys[at] = key;
        }
    }
    // With fewer than a hundred tokens in the set all of them are listed.
    const uint32_t m = s->cut100 == 0xffffffffu ? n : (uint32_t)ARES_SAMPLE_M;
    uint32_t k = 1;
    if (m >= 2) {
        double x[ARES_SAMPLE_M];
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t key = i < n ? keys[i] : s->cut100;
            float d, xf;
            ares_sample_x(*s, key, &d, &xf);
            x[i] = (double)xf;
        }
        double sum_tb = 0.0, sum_tt = 0.0;
        for (uint32_t i = 0; i + 1 < m; ++i) {
            const double t = ares_ln_d(__ddiv_rn((double)(i + 2), (double)(i + 1)));
            sum_tb = __dadd_rn(sum_tb, __dmul_rn(t, __dsub_rn(x[i + 1], x[i])));
            sum_tt = __dadd_rn(sum_tt, __dmul_rn(t, t));
        }
        const double s_hat = __ddiv_rn(sum_tb, sum_tt);
        if (isfinite(s_hat) && s_hat > 0.0) {
            const double eps = __dsub_rn(s_hat, 1.0);
            const double n_vocab = (double)p.vocab;
            const double two_mu = ares_exp_d(__dmul_rn((double)p.mu, 0.693147180559945309417232));
            const double base = __ddiv_rn(
                __dmul_rn(eps, two_mu),
                __dsub_rn(1.0, ares_exp_d(__dmul_rn(-eps, ares_ln_d(n_vocab)))));
            if (isfinite(base) && base > 0.0) {
                const double value = ares_pow_d(base, __ddiv_rn(1.0, s_hat));
                if (isfinite(value) && value >= 1.0) {
                    k = value < 4294967295.0 ? (uint32_t)value : 0xffffffffu;
                }
            }
        }
    }
    s->k = k;
}
