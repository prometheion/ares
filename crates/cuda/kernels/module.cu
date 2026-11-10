// The kernel module of the engine: every kernel is compiled into one fat
// binary from this file.

#include "common.cuh"
#include "basic.cuh"
#include "matvec_q4_K.cuh"
#include "quantize_act.cuh"
#include "mmvq_q4_K.cuh"
#include "mmvq_kquants.cuh"
#include "mmvq_iquants.cuh"
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
