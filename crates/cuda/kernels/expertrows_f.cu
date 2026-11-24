// Many-row expert kernels of more weight types (see expertrows.cu and expertrows_mm.cuh):
// the 4-bit float types, Q1_0 and the floats. A module of its own so that the compiler runs side by side with the
// others; built like expertrows.cu and loaded by `ExpertRowKernels` (src/experts.rs).

#include "common.cuh"
#include "expertrows_mm.cuh"

ARES_ER_MM_TYPE(mxfp4, EngMmMXFP4)
ARES_ER_MM_TYPE(nvfp4, EngMmNVFP4)
ARES_ER_MM_TYPE(q1_0, EngMmQ10)
ARES_ER_MM_TYPE(f32, EngMmF32)
ARES_ER_MM_TYPE(f16, EngMmF16)
ARES_ER_MM_TYPE(bf16, EngMmBF16)
