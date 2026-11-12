// A row's candidates are known: sets up its state and, for a greedy row, writes the pick.
// `best` is the row's list over all devices, best first. One invocation.
#include "sample_top_lists.glsl"

void join_row(uint64_t best[CANDIDATES], uint bad, Params p, State s, Out o) {
    uint status = bad != 0u ? S_NONFINITE : 0u;
    [[unroll]] for (uint c = 0u; c < CANDIDATES; ++c) {
        const bool have = best[c] != 0ul;
        o.cand_token[c] = have ? 0xffffffffu - uint(best[c]) : 0xffffffffu;
        o.cand_logit[c] = have ? unsortable(uint(best[c] >> 32)) : NEG_INF;
    }
    const float l_max = o.cand_logit[0];
    if (!(l_max > -3.402823466e+38)) {
        // No token has a finite logit.
        status |= S_EMPTY;
    }
    const bool sampled = (p.flags & F_SAMPLE) != 0u;
    s.l_max = l_max;
    s.status = status;
    s.lo = 0u;
    s.hi = sampled ? KEY_MAX : 0u;
    s.inv_t = 1.0;
    s.cursor = 0u;
    s.k = p.top_k;
    s.prefix = 0u;
    s.rem = 0ul;
    s.cut = 0xffffffffu;
    s.cut100 = 0xffffffffu;
    s.target = 0ul;
    s.n_gather = 0u;
    o.token = sampled ? 0xffffffffu : o.cand_token[0];
    o.value = l_max;
    o.status = status;
    o.aux = 0.0;
    if (sampled && (status & S_EMPTY) == 0u) {
        sample_advance(p, s);
    }
}
