// The kernel module of the engine: every kernel is compiled into one fat
// binary from this file.

#include "common.cuh"
#include "basic.cuh"
#include "matvec_q4_K.cuh"
#include "quantize_act.cuh"
#include "mmvq_q4_K.cuh"
#include "mmvq_kquants.cuh"
#include "mmvq_iquants.cuh"
#include "mmvq_more.cuh"
#include "mmvq_float.cuh"
#include "sum_peer.cuh"

// One weight kernel per type, named after the type as ggml spells it. The
// Rust side (src/kernels.rs) lists the same names.
ARES_MMVQ_KERNEL(ares_mmvq_q4_K, EngQ4K)
ARES_MMVQ_KERNEL(ares_mmvq_q5_K, EngQ5K)
ARES_MMVQ_KERNEL(ares_mmvq_q6_K, EngQ6K)
ARES_MMVQ_KERNEL(ares_mmvq_q3_K, EngQ3K)
ARES_MMVQ_KERNEL(ares_mmvq_iq4_xs, EngIQ4XS)
ARES_MMVQ_KERNEL(ares_mmvq_iq4_nl, EngIQ4NL)
ARES_MMVQ_KERNEL(ares_mmvq_iq3_s, EngIQ3S)
ARES_MMVQ_KERNEL(ares_mmvq_iq3_xxs, EngIQ3XXS)
ARES_MMVQ_KERNEL(ares_mmvq_iq2_s, EngIQ2S)
ARES_MMVQ_KERNEL(ares_mmvq_iq2_xs, EngIQ2XS)
ARES_MMVQ_KERNEL(ares_mmvq_q8_0, EngQ80)

// The types that completed the set.
ARES_MMVQ_KERNEL(ares_mmvq_q4_0, EngQ40)
ARES_MMVQ_KERNEL(ares_mmvq_q4_1, EngQ41)
ARES_MMVQ_KERNEL(ares_mmvq_q5_0, EngQ50)
ARES_MMVQ_KERNEL(ares_mmvq_q5_1, EngQ51)
ARES_MMVQ_KERNEL(ares_mmvq_mxfp4, EngMXFP4)
ARES_MMVQ_KERNEL(ares_mmvq_nvfp4, EngNVFP4)
ARES_MMVQ_KERNEL(ares_mmvq_q2_K, EngQ2K)
ARES_MMVQ_KERNEL(ares_mmvq_iq2_xxs, EngIQ2XXS)
ARES_MMVQ_KERNEL(ares_mmvq_iq1_s, EngIQ1S)
ARES_MMVQ_KERNEL(ares_mmvq_iq1_m, EngIQ1M)
ARES_MMVQ_KERNEL(ares_mmvq_tq1_0, EngTQ10)
ARES_MMVQ_KERNEL(ares_mmvq_tq2_0, EngTQ20)
ARES_MMVQ_KERNEL(ares_mmvq_q1_0, EngQ10)
ARES_MMVQ_KERNEL(ares_mmvq_q2_0, EngQ20)
ARES_MMVF_KERNEL(ares_mmvq_f32, AresF32)
ARES_MMVF_KERNEL(ares_mmvq_f16, AresF16)
ARES_MMVF_KERNEL(ares_mmvq_bf16, AresBF16)
