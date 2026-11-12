// The best score of a row's chunks, for the shaders that follow `sample_pick` (THREADS
// invocations, `chunks` at most that many). A row with nothing left has none.
#include "sample_common.glsl"

uint64_t pick_best(uint64_t lists, uint r, uint chunks, State s) {
    uint64_t best = 0ul;
    if ((s.status & S_EMPTY) == 0u && gl_LocalInvocationID.x < chunks) {
        best = U64Ref(lists + 8ul * (uint64_t(r) * chunks + gl_LocalInvocationID.x)).v[0];
    }
    return block_max_u64(best);
}
