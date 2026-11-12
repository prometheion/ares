// The stages that need numbers about the row's set: its sums (top-n-sigma, typical sampling,
// XTC, a dynamic temperature, mirostat), the tokens above a threshold (XTC), the keys inside a
// radius (typical sampling) and the first hundred keys (mirostat 1).
//
// A stage is a pass over the row (`sample_pass`), the export of what the chunks added up into
// the exchange buffer (`sample_pass_export`), the sum across devices, and a shader that reads
// the total and moves the row's state (`sample_stage_decide`). Every number a device sends is an
// integer split into four limbs of 16 bits, so the sums of up to 16 devices are exact in f32.
//
// The chunks of a pass write their partial results to the row's slots of the `lists` scratch
// (five 64-bit numbers a chunk, `PARTIALS` words each) and the export adds them up: CUDA adds
// them in the row's state with 64-bit atomics, an optional feature here.
#include "sample_common.glsl"

const uint PASS_STATS = 0u;
const uint PASS_XTC = 1u;
const uint PASS_TYPICAL = 2u;
const uint PASS_GATHER = 3u;
const uint PASS_TAIL = 4u;
const uint PASS_RAW = 5u;

// What a decide shader does with the totals of a pass.
const uint PHASE_NSIGMA = 0u;
const uint PHASE_TYPICAL_MEAN = 1u;
const uint PHASE_TYPICAL_SET = 2u;
const uint PHASE_XTC_TARGET = 3u;
const uint PHASE_XTC_CUT = 4u;
const uint PHASE_TEMP = 5u;
const uint PHASE_MIRO2 = 6u;
const uint PHASE_MIRO1 = 7u;
const uint PHASE_TAIL = 8u;
const uint PHASE_RAW = 9u;

// 64-bit numbers per chunk in the partial results of a pass.
const uint PARTIALS = 5u;

// Whether a row takes part in a pass.
bool pass_runs(Params p, State s, uint mode, uint position) {
    switch (mode) {
    case PASS_STATS: {
        if ((p.flags & F_SAMPLE) == 0u || (s.status & S_EMPTY) != 0u || s.cursor != position) {
            return false;
        }
        const uint stage = stage_at(p, position);
        return stage == STAGE_NSIGMA || stage == STAGE_TYPICAL || stage == STAGE_MIRO2
            || (stage == STAGE_XTC && xtc_applies(p)) || (stage == STAGE_TEMP && p.dyn_delta > 0.0);
    }
    case PASS_XTC:
        return stage_runs(p, s, position, STAGE_XTC) && xtc_applies(p) && s.target != 0ul;
    case PASS_TYPICAL:
        return stage_runs(p, s, position, STAGE_TYPICAL);
    case PASS_GATHER:
        return stage_runs(p, s, position, STAGE_MIRO1);
    case PASS_TAIL:
        return (p.flags & (F_LP_POST | F_MIROSTAT)) != 0u && (s.status & S_EMPTY) == 0u;
    default:
        return (p.flags & F_LP_RAW) != 0u && present(s.raw_lmax);
    }
}

// The number `k` of a device's slot of the exchange, at byte address `slot`.
uint64_t slot_number(uint64_t slot, uint k) {
    uint64_t v = 0ul;
    for (uint limb = 0u; limb < 4u; ++limb) {
        v |= uint64_t(uint(ldf(slot + 4ul * (k * 4u + limb)))) << (16u * limb);
    }
    return v;
}
