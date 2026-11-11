// The sampler: the pick of each row from its logits, on the device. One fat
// binary (build.rs), loaded by `SamplerKernels` (src/sampler.rs). The
// definitions it is held to, and the same arithmetic for the host, are in
// crates/sampler.

#include "common.cuh"
#include "sampler_math.cuh"
#include "sampler_stage.cuh"
#include "sampler_top.cuh"
#include "sampler_correct.cuh"
#include "sampler_descent.cuh"
#include "sampler_pass.cuh"
#include "sampler_pick.cuh"
#include "sampler_tail.cuh"
#include "sampler_draft.cuh"
