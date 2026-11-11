// The many-row expert kernels: the plan of a layer's picks and the tile kernels that
// compute the experts' products for any number of rows on the tensor cores. A module
// of its own, compiled for compute capability 8.0 and newer only (see build.rs): it
// uses `mma.m16n8k16` with f16 inputs and `cp.async`, as the prefill module does.
// Loaded by `ExpertRowKernels` (src/experts.rs); the Rust side lists the same names.

#include "common.cuh"
#include "expertrows_plan.cuh"
#include "expertrows_down.cuh"

// The gate and up kernels by tile (8 NB pairs): the stages of activations in shared memory.
// Three, and two for the largest tile, which would otherwise leave room for one block on a
// multiprocessor (src/experts.rs lists the same numbers). Deeper pipelines of the small
// tiles (more stages, weights loaded several chunks ahead) were measured slower on the real
// weights: more registers and shared memory per block, fewer blocks on a multiprocessor.
#define ARES_ER_GATE_UP_SIZES(PREFIX, T)                          \
    ARES_ER_GATE_UP_KERNEL(PREFIX##_n1, T, 1, 3)                  \
    ARES_ER_GATE_UP_KERNEL(PREFIX##_n2, T, 2, 3)                  \
    ARES_ER_GATE_UP_KERNEL(PREFIX##_n4, T, 4, 3)                  \
    ARES_ER_GATE_UP_KERNEL(PREFIX##_n8, T, 8, 2)

ARES_ER_GATE_UP_SIZES(ares_er_gate_up_iq2_s, ErIQ2S)
ARES_ER_GATE_UP_SIZES(ares_er_gate_up_iq2_xxs, ErIQ2XXS)
ARES_ER_GATE_UP_SIZES(ares_er_gate_up_iq1_m, ErIQ1M)

#define ARES_ER_DOWN_SIZES(PREFIX)                                \
    ARES_ER_DOWN_KERNEL(PREFIX##_ordered_n1, 1, 4, false)         \
    ARES_ER_DOWN_KERNEL(PREFIX##_ordered_n2, 2, 4, false)         \
    ARES_ER_DOWN_KERNEL(PREFIX##_ordered_n4, 4, 4, false)         \
    ARES_ER_DOWN_KERNEL(PREFIX##_ordered_n8, 8, 4, false)         \
    ARES_ER_DOWN_KERNEL(PREFIX##_atomic_n1, 1, 4, true)           \
    ARES_ER_DOWN_KERNEL(PREFIX##_atomic_n2, 2, 4, true)           \
    ARES_ER_DOWN_KERNEL(PREFIX##_atomic_n4, 4, 4, true)           \
    ARES_ER_DOWN_KERNEL(PREFIX##_atomic_n8, 8, 4, true)

ARES_ER_DOWN_SIZES(ares_er_down_q2_0)
