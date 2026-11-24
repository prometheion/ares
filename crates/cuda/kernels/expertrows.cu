// The many-row expert kernels: the plan of a layer's picks and the tile kernels that
// compute the experts' products for any number of rows on the tensor cores. A module
// of its own (see build.rs); it uses `mma.m16n8k16` with f16 inputs and `cp.async`, as
// the prefill module does.
// Loaded by `ExpertRowKernels` (src/experts.rs); the Rust side lists the same names.

#include "common.cuh"
#include "expertrows_plan.cuh"
#include "expertrows_down.cuh"
#include "mm_types_more.cuh"

// Gate and up of the types that have a decoder written for these kernels
// (expertrows_types.cuh; IQ1_S's is next to its tile kernel in mm_types_more.cuh), and down
// of Q2_0: the types of the model the kernels were made for. Every other type, and these in
// the other role, come through the decoders of the dense tile kernels
// (expertrows_k.cu, expertrows_i.cu, expertrows_l.cu, expertrows_f.cu).
ARES_ER_GATE_UP_SIZES(ares_er_gate_up_iq2_s, ErIQ2S)
ARES_ER_GATE_UP_SIZES(ares_er_gate_up_iq2_xxs, ErIQ2XXS)
ARES_ER_GATE_UP_SIZES(ares_er_gate_up_iq1_m, ErIQ1M)
ARES_ER_GATE_UP_SIZES(ares_er_gate_up_iq1_s, ErIQ1S)

ARES_ER_DOWN_SIZES(ares_er_down_q2_0, ErQ20)
