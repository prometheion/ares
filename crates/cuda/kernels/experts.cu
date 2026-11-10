// The kernels of the mixture of experts: the router and the routed experts of
// a decode step, addressed through a slot table. One fat binary (build.rs),
// loaded by `ExpertKernels` (src/experts.rs).

#include "common.cuh"
#include "mmvq.cuh"
#include "mmvq_q4_K.cuh"
#include "mmvq_iquants.cuh"
#include "mmvq_more.cuh"
#include "mmvq_float.cuh"
#include "expert_layout.cuh"
#include "expert_router.cuh"
#include "expert_ffn.cuh"

// The router's logits: its matrix is F32 and multiplies the rows exactly.
ARES_MMVF_KERNEL(ares_expert_logits, AresF32)

// Gate and up, one kernel per type the routed experts of a model use. The
// Rust side (src/experts.rs) lists the same names.
ARES_EXPERT_GATE_UP_KERNEL(ares_expert_gate_up_iq2_s, EngIQ2S)
ARES_EXPERT_GATE_UP_KERNEL(ares_expert_gate_up_iq2_xxs, EngIQ2XXS)
ARES_EXPERT_GATE_UP_KERNEL(ares_expert_gate_up_iq1_m, EngIQ1M)

// Down.
ARES_EXPERT_DOWN_KERNEL(ares_expert_down_q2_0, EngQ20)
