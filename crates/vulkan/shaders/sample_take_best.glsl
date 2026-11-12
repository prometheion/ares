// `CANDIDATES` times: the largest proposal of the workgroup, taken from the invocation that
// made it. `mine` is the invocation's own list in descending order with `n_mine` entries.
// Invocation 0 gets the workgroup's list in `merged` (declared by the including shader).
void take_best(uint64_t mine[CANDIDATES], uint n_mine) {
    uint next = 0u;
    for (uint c = 0u; c < CANDIDATES; ++c) {
        uint64_t proposal = 0ul;
        [[unroll]] for (uint j = 0u; j < CANDIDATES; ++j) {
            if (j == next && next < n_mine) {
                proposal = mine[j];
            }
        }
        const uint64_t best = block_max_u64(proposal);
        if (best != 0ul && proposal == best) {
            next += 1u;
        }
        if (gl_LocalInvocationID.x == 0u) {
            merged[c] = best;
        }
        barrier();
    }
}

// A row's candidates are known: sets up its state and, for a greedy row, writes the pick.
// `best` is the row's list over all devices, best first.
void join_row(uint64_t best[CANDIDATES], uint bad, ParamsRef p, StateRef state, OutRef out_) {
    uint status = bad != 0u ? S_NONFINITE : 0u;
    [[unroll]] for (uint c = 0u; c < CANDIDATES; ++c) {
        const bool have = best[c] != 0ul;
        out_.cand_token[c] = have ? 0xffffffffu - uint(best[c]) : 0xffffffffu;
        out_.cand_logit[c] = have ? unsortable(uint(best[c] >> 32)) : NEG_INF;
    }
    const float l_max = out_.cand_logit[0];
    // No token has a finite logit.
    if (!(l_max > -3.402823466e+38)) {
        status |= S_EMPTY;
    }
    state.l_max = l_max;
    state.status = status;
    state.prefix_k = 0u;
    state.rem_k = p.top_k;
    state.z_k = 0ul;
    state.prefix_p = 0u;
    state.cut_key = min(p.min_p_key, KEY_MAX);
    state.rem_p = 0ul;
    state.best = 0ul;
    const bool sampled = (p.flags & F_SAMPLE) != 0u;
    out_.token = sampled ? 0xffffffffu : out_.cand_token[0];
    out_.value = l_max;
    out_.status = status;
    out_.reserved = 0u;
}
