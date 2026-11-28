// The kernels of the mixture of experts: the router and the routed experts of
// a decode step, addressed through a slot table. One fat binary (build.rs),
// loaded by `ExpertKernels` (src/experts.rs).

#include "common.cuh"
#include "mmvq.cuh"
#include "mmvq_q4_K.cuh"
#include "mmvq_kquants.cuh"
#include "mmvq_iquants.cuh"
#include "mmvq_more.cuh"
#include "mmvq_float.cuh"
#include "expert_layout.cuh"
#include "expert_router.cuh"
#include "expert_ffn.cuh"
#include "expert_predict.cuh"

// The router's logits: its matrix is F32 and multiplies the rows exactly.
ARES_MMVF_KERNEL(ares_expert_logits, AresF32)

// Gate and up, and down, for every weight type: an expert's matrices may be stored in any
// of them (a row must be whole blocks of the type). The Rust side (src/experts.rs) makes
// the same names from the types of src/kernels/types.rs. Each for steps of up to 8 rows,
// and `_r1`, `_r2`, `_r4` for steps of up to 1, 2 and 4 (expert_ffn.cuh, `ares_expert_dots`).
#define ARES_EXPERT_ROWS(SUFFIX, T, NB)                                      \
    ARES_EXPERT_GATE_UP_KERNEL(ares_expert_gate_up_##SUFFIX, T, NB)          \
    ARES_EXPERT_DOWN_KERNEL(ares_expert_down_##SUFFIX, T, NB)
#define ARES_EXPERT_TYPE(SUFFIX, T)                  \
    ARES_EXPERT_ROWS(SUFFIX, T, ARES_MAX_BATCH)      \
    ARES_EXPERT_ROWS(SUFFIX##_r1, T, 1)              \
    ARES_EXPERT_ROWS(SUFFIX##_r2, T, 2)              \
    ARES_EXPERT_ROWS(SUFFIX##_r4, T, 4)

ARES_EXPERT_TYPE(q4_K, EngQ4K)
ARES_EXPERT_TYPE(q5_K, EngQ5K)
ARES_EXPERT_TYPE(q6_K, EngQ6K)
ARES_EXPERT_TYPE(q3_K, EngQ3K)
ARES_EXPERT_TYPE(q2_K, EngQ2K)
ARES_EXPERT_TYPE(iq4_xs, EngIQ4XS)
ARES_EXPERT_TYPE(iq4_nl, EngIQ4NL)
ARES_EXPERT_TYPE(iq3_s, EngIQ3S)
ARES_EXPERT_TYPE(iq3_xxs, EngIQ3XXS)
ARES_EXPERT_TYPE(iq2_s, EngIQ2S)
ARES_EXPERT_TYPE(iq2_xs, EngIQ2XS)
ARES_EXPERT_TYPE(iq2_xxs, EngIQ2XXS)
ARES_EXPERT_TYPE(iq1_s, EngIQ1S)
ARES_EXPERT_TYPE(iq1_m, EngIQ1M)
ARES_EXPERT_TYPE(q8_0, EngQ80)
ARES_EXPERT_TYPE(q4_0, EngQ40)
ARES_EXPERT_TYPE(q4_1, EngQ41)
ARES_EXPERT_TYPE(q5_0, EngQ50)
ARES_EXPERT_TYPE(q5_1, EngQ51)
ARES_EXPERT_TYPE(tq1_0, EngTQ10)
ARES_EXPERT_TYPE(tq2_0, EngTQ20)
ARES_EXPERT_TYPE(mxfp4, EngMXFP4)
ARES_EXPERT_TYPE(nvfp4, EngNVFP4)
ARES_EXPERT_TYPE(q1_0, EngQ10)
ARES_EXPERT_TYPE(q2_0, EngQ20)
ARES_EXPERT_TYPE(f32, EngExpertFloat<AresF32>)
ARES_EXPERT_TYPE(f16, EngExpertFloat<AresF16>)
ARES_EXPERT_TYPE(bf16, EngExpertFloat<AresBF16>)
