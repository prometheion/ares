// The lists of best tokens, for the shaders that find them (`sample_top*`, `lp_top*`). The
// including shader defines `TOP_N`, the candidates a list holds.
//
// A list is packed values (`pack_pick`), best first, zero where it has fewer. A chunk of the
// vocabulary writes its list followed by one word that is 1 when the chunk has a logit that is
// not a number; the chunks' lists of a device are merged into the device's list, and the lists
// of several devices after the exchange between them (the sum: each device writes its own slot
// of the exchange buffer and zeros elsewhere).
#include "sample_common.glsl"

// Entries per chunk in the list a top shader writes: the candidates and one word of flags.
const uint LIST = TOP_N + 1u;
// f32 values per device and row in the exchange of the candidates: logit and token of each, and
// the status.
const uint TOP_VALUES = 2u * TOP_N + 2u;

shared uint64_t merged[TOP_N];
shared uint bad_any;

// `TOP_N` times: the largest proposal of the workgroup, taken from the invocation that made it.
// `mine` is the invocation's own list in descending order with `n_mine` entries (at most
// `MINE`). Invocation 0 gets the workgroup's list in `merged`.
#define MINE (TOP_N < 16u ? TOP_N : 16u)
void take_best(uint64_t mine[MINE], uint n_mine) {
    uint next = 0u;
    for (uint c = 0u; c < TOP_N; ++c) {
        uint64_t proposal = 0ul;
        [[unroll]] for (uint j = 0u; j < MINE; ++j) {
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

// The best `TOP_N` candidates of the chunks of row `r` in `lists`, and whether any chunk saw a
// logit that is not a number. Run by every invocation of the workgroup, which has at least as
// many invocations as there are chunks. `bad_any` must be zero before, and a barrier between.
void merge_chunks(uint64_t lists, uint r, uint chunks) {
    const uint tid = gl_LocalInvocationID.x;
    const bool listed = tid < chunks;
    const uint64_t list = listed ? lists + 8ul * (uint64_t(r) * chunks + tid) * LIST : 0ul;
    if (listed && U64Ref(list + 8ul * TOP_N).v[0] != 0ul) {
        bad_any = 1u;
    }
    uint taken = 0u;
    for (uint c = 0u; c < TOP_N; ++c) {
        const uint64_t proposal = (listed && taken < TOP_N) ? U64Ref(list + 8ul * taken).v[0] : 0ul;
        const uint64_t best = block_max_u64(proposal);
        if (best != 0ul && proposal == best) {
            taken += 1u;
        }
        if (tid == 0u) {
            merged[c] = best;
        }
        barrier();
    }
}

// This device's list into its slot of `exchange` (`TOP_VALUES` values per device and row) and
// zeros into the other slots, for the sum across devices. One invocation.
void export_list(uint64_t exchange, uint r, uint devices, uint rank, uint bad) {
    const uint64_t slots = exchange + 4ul * uint64_t(r) * devices * TOP_VALUES;
    for (uint d = 0u; d < devices; ++d) {
        const uint64_t slot = slots + 4ul * d * TOP_VALUES;
        for (uint c = 0u; c < TOP_N; ++c) {
            const bool have = d == rank && merged[c] != 0ul;
            // A token as a float is exact below 2^24; -1 marks "none". The other devices'
            // slots hold zeros, so the sum leaves this one.
            FloatRef(slot + 8ul * c).v[0] = have ? unsortable(uint(merged[c] >> 32)) : 0.0;
            FloatRef(slot + 8ul * c + 4ul).v[0] =
                d != rank ? 0.0 : (have ? float(0xffffffffu - uint(merged[c])) : -1.0);
        }
        FloatRef(slot + 8ul * TOP_N).v[0] = d == rank ? float(bad) : 0.0;
        FloatRef(slot + 8ul * TOP_N + 4ul).v[0] = 0.0;
    }
}

// The lists of all devices after the sum, merged: the best `TOP_N` of them. One invocation.
void join_lists(uint64_t exchange, uint r, uint devices, out uint64_t best[TOP_N], out uint bad) {
    const uint64_t slots = exchange + 4ul * uint64_t(r) * devices * TOP_VALUES;
    [[unroll]] for (uint c = 0u; c < TOP_N; ++c) {
        best[c] = 0ul;
    }
    bad = 0u;
    for (uint d = 0u; d < devices; ++d) {
        const uint64_t slot = slots + 4ul * d * TOP_VALUES;
        if (ldf(slot + 8ul * TOP_N) != 0.0) {
            bad = 1u;
        }
        for (uint c = 0u; c < TOP_N; ++c) {
            const float token = ldf(slot + 8ul * c + 4ul);
            if (!(token >= 0.0)) {
                continue;
            }
            uint64_t v = pack_pick(ldf(slot + 8ul * c), uint(token));
            [[unroll]] for (uint j = 0u; j < TOP_N; ++j) {
                if (v > best[j]) {
                    const uint64_t t = best[j];
                    best[j] = v;
                    v = t;
                }
            }
        }
    }
}
