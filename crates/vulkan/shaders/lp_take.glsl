// The list of a row that reports probabilities is known: its tokens and logits go to the record,
// as many as the row asks for, and the best logit to the state when it is the model's own
// (`raw`). One invocation. The including shader defines `TOP_N` as `LOGPROBS_MAX`.
#include "sample_top_lists.glsl"

// Whether a row takes part in the list of the probabilities of kind `raw`.
bool lp_row(Params p, State s, bool raw) {
    return raw ? (p.flags & F_LP_RAW) != 0u
               : ((p.flags & F_LP_POST) != 0u && (s.status & S_EMPTY) == 0u);
}

void lp_take(uint64_t best[LOGPROBS_MAX], Params p, State s, Logprobs o, bool raw) {
    uint n = 0u;
    const uint want = min(p.logprobs, LOGPROBS_MAX);
    for (uint c = 0u; c < want; ++c) {
        if (best[c] == 0ul || !present(unsortable(uint(best[c] >> 32)))) {
            break;
        }
        o.token[n] = 0xffffffffu - uint(best[c]);
        // The logit; the finishing shader turns it into a probability.
        o.logprob[n] = unsortable(uint(best[c] >> 32));
        n += 1u;
    }
    o.n = n;
    if (raw) {
        s.raw_lmax = best[0] == 0ul ? NEG_INF : unsortable(uint(best[0] >> 32));
    }
}
