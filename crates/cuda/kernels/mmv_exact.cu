// The weight kernels that multiply unquantized activations with a quantized matrix
// (mmv_exact.cuh): one per quantized weight type, named after the type as ggml spells it.
// The float types have theirs in module.cu (`ares_mmvq_f32` and so on), which read the same
// activations. A module of its own (build.rs), so that it compiles beside module.cu. The
// Rust side (src/kernels.rs) lists the same names.

#include "mmv_exact.cuh"

ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q4_0, 2)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q4_1, 3)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q5_0, 6)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q5_1, 7)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q8_0, 8)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q2_K, 10)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q3_K, 11)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q4_K, 12)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q5_K, 13)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q6_K, 14)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq2_xxs, 16)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq2_xs, 17)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq3_xxs, 18)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq1_s, 19)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq4_nl, 20)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq3_s, 21)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq2_s, 22)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq4_xs, 23)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_iq1_m, 29)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_tq1_0, 34)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_tq2_0, 35)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_mxfp4, 39)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_nvfp4, 40)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q1_0, 41)
ARES_MMV_EXACT_KERNEL(ares_mmv_exact_q2_0, 42)
