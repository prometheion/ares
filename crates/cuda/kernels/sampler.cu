// The sampler: the pick of each row from its logits, on the device. One fat
// binary (build.rs), loaded by `SamplerKernels` (src/sampler.rs), which also
// holds the same arithmetic for the host.

#include "common.cuh"
#include "sampler_math.cuh"
#include "sampler_top.cuh"
#include "sampler_descent.cuh"
#include "sampler_pick.cuh"
#include "sampler_draft.cuh"
