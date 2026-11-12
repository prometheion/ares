// The best logits of each chunk of each row: the greedy pick, the reference point of every
// threshold (the best logit) and the candidates a grammar may fall back to. The same shader with
// more candidates finds the alternatives a row reports probabilities for. The wrappers
// (`sample_top.comp`, `lp_top_raw.comp`, `lp_top_post.comp`) define `TOP_N` and `TOP_ROWS`:
//
// - `TOP_ROWS` 0: every row, all tokens;
// - 1: the rows that report the model's own probabilities, all tokens;
// - 2: the rows that report the probabilities after the chain, the tokens of the set the chain
//   left.
//
// Grid (chunks, rows), THREADS invocations. Chunk `b` of row `r` is the logits `b * chunk ..` of
// the row; its best candidates go to `lists[(r * chunks + b) * LIST ..]`, packed, best first,
// then a word that is 1 when the chunk has a logit that is NaN or plus infinity. `tokens` maps a
// column of the logits to its token (`tokens[first + i]`); null means the token is `first + i`.
#include "sample_top_lists.glsl"

layout(local_size_x = 256) in;

layout(push_constant, std430) uniform Push {
    uint64_t logits;
    uint64_t tokens;
    uint64_t lists;
    uint64_t params;
    uint64_t state;
    uint stride;
    uint n_local;
    uint first;
    uint chunk;
} pc;

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint r = gl_WorkGroupID.y;
#if TOP_ROWS == 1
    if ((params_of(pc.params, r).flags & F_LP_RAW) == 0u) {
        return;
    }
#elif TOP_ROWS == 2
    State st = state_of(pc.state, r);
    if ((params_of(pc.params, r).flags & F_LP_POST) == 0u || (st.status & S_EMPTY) != 0u) {
        return;
    }
#endif
    const uint begin = gl_WorkGroupID.x * pc.chunk;
    const uint end = min(begin + pc.chunk, pc.n_local);
    const uint64_t row = pc.logits + 4ul * uint64_t(r) * pc.stride;
    if (tid == 0u) {
        bad_any = 0u;
    }
    barrier();

    uint64_t mine[MINE];
    [[unroll]] for (uint j = 0u; j < MINE; ++j) {
        mine[j] = 0ul;
    }
    uint n_mine = 0u;
    bool any_bad = false;
    for (uint i = begin + tid; i < end; i += THREADS) {
        bool bad;
        const float l = sample_logit(ldf(row + 4ul * i), bad);
        any_bad = any_bad || bad;
#if TOP_ROWS == 2
        if (!present(l) || !in_set(st, sample_key(st.l_max, l))) {
            continue;
        }
#endif
        uint64_t v = pack_pick(l, token_of(pc.tokens, pc.first, i));
        // Insert into the descending list; the smallest falls off the end.
        [[unroll]] for (uint j = 0u; j < MINE; ++j) {
            if (j < n_mine && v > mine[j]) {
                const uint64_t t = mine[j];
                mine[j] = v;
                v = t;
            }
        }
        if (n_mine < MINE) {
            [[unroll]] for (uint j = 0u; j < MINE; ++j) {
                if (j == n_mine) {
                    mine[j] = v;
                }
            }
            n_mine++;
        }
    }
    if (any_bad) {
        bad_any = 1u;
    }
    take_best(mine, n_mine);
    if (tid == 0u) {
        const uint64_t out_list =
            pc.lists + 8ul * (uint64_t(r) * gl_NumWorkGroups.x + gl_WorkGroupID.x) * LIST;
        for (uint c = 0u; c < TOP_N; ++c) {
            U64Ref(out_list + 8ul * c).v[0] = merged[c];
        }
        U64Ref(out_list + 8ul * TOP_N).v[0] = uint64_t(bad_any);
    }
}
