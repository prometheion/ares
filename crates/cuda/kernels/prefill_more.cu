// The second prefill module: the tensor-core weight kernels (mm.cuh) of the types that
// prefill.cu does not have, f16 and f16 with f16 accumulation only (mm_types_more.cuh).
// The Rust side makes the same names from the types of src/kernels/types.rs.

#include "common.cuh"
#include "mm_shapes.cuh"
#include "mm_types_more.cuh"

#define ARES_MM_MORE(NAME, T)                       \
    ARES_MM_F16_LARGE(ares_mm_f16_##NAME, T)        \
    ARES_MM_F16_MEDIUM(ares_mm_f16m_##NAME, T)      \
    ARES_MM_F16_SMALL(ares_mm_f16s_##NAME, T)       \
    ARES_MM_H_LARGE(ares_mm_h_##NAME, T)            \
    ARES_MM_H_MEDIUM(ares_mm_hm_##NAME, T)          \
    ARES_MM_H_SMALL(ares_mm_hs_##NAME, T)

ARES_MM_MORE(iq2_xxs, EngMmIQ2XXS)
ARES_MM_MORE(iq1_s, EngMmIQ1S)
ARES_MM_MORE(iq1_m, EngMmIQ1M)
ARES_MM_MORE(q2_0, EngMmQ20)
ARES_MM_MORE(q2_K, EngMmQ2K)
ARES_MM_MORE(q4_0, EngMmQ40)
ARES_MM_MORE(q4_1, EngMmQ41)
ARES_MM_MORE(q5_0, EngMmQ50)
ARES_MM_MORE(q5_1, EngMmQ51)
ARES_MM_MORE(tq1_0, EngMmTQ10)
ARES_MM_MORE(tq2_0, EngMmTQ20)
ARES_MM_MORE(mxfp4, EngMmMXFP4)
ARES_MM_MORE(nvfp4, EngMmNVFP4)
ARES_MM_MORE(q1_0, EngMmQ10)
ARES_MM_MORE(f32, EngMmF32)
ARES_MM_MORE(f16, EngMmF16)
ARES_MM_MORE(bf16, EngMmBF16)
