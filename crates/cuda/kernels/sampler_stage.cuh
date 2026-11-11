// The stages of a row's program that need no pass over the row, and what
// decides whether a row takes part in a kernel.
//
// A row's state has a cursor, the next position of its program. A stage that
// only changes the set or the temperature from numbers the state holds (min-p,
// a fixed temperature) runs as soon as the stage before it is done, in the
// kernel that finishes that one, so that it costs no launch.
#pragma once

#include "sampler_math.cuh"

// Runs the stages from the cursor on that need no pass over the row, up to
// the first that does.
static __device__ void ares_sample_advance(const AresSampleParams& p, AresSampleState* s) {
    for (;;) {
        const uint32_t stage = ares_stage_at(p, s->cursor);
        if (stage == ARES_STAGE_MINP) {
            // Keeps the tokens at least min_p times as probable as the best of
            // the set: d' <= -ln(min_p) / inv_t.
            const float lim = __fadd_rn(__uint_as_float(s->lo), __fdiv_rn(p.min_p_c, s->inv_t));
            s->hi = min(s->hi, min(__float_as_uint(lim), (uint32_t)ARES_SAMPLE_KEY_MAX));
        } else if (stage == ARES_STAGE_TEMP && !(p.dyn_delta > 0.0f)) {
            if (p.temp < ARES_SAMPLE_TEMPERATURE_MIN) {
                s->hi = s->lo;
            } else {
                s->inv_t = __fmul_rn(s->inv_t, p.inv_temp);
            }
        } else {
            return;
        }
        s->cursor += 1;
    }
}

// The stage at `position` is done: the next one runs.
static __device__ void ares_stage_done(const AresSampleParams& p, AresSampleState* s,
                                       uint32_t position) {
    s->cursor = position + 1;
    ares_sample_advance(p, s);
}

// Whether a sampled row, with something left to pick, is at `position` of its
// program with the stage `kind` there.
static __device__ __forceinline__ bool ares_stage_runs(const AresSampleParams& p,
                                                       const AresSampleState& s,
                                                       uint32_t position, uint32_t kind) {
    return (p.flags & ARES_SAMPLE_F_SAMPLE) && !(s.status & ARES_SAMPLE_S_EMPTY) &&
           s.cursor == position && ares_stage_at(p, position) == kind;
}

// Whether the XTC cut applies at this row's position: its uniform draw is at
// most the probability.
static __device__ __forceinline__ bool ares_xtc_applies(const AresSampleParams& p) {
    return ares_sample_xtc_draw(p.seed_lo, p.seed_hi, p.position) <= p.xtc_prob;
}
