// Many-row expert kernels of more weight types (see expertrows.cu and expertrows_mm.cuh):
// the i-quants of 3 bits and fewer (gate and up of four of them are in expertrows.cu, so only their down is here) and gate and up of Q2_0. A module of its own so that the compiler runs side by side with the
// others; built like expertrows.cu and loaded by `ExpertRowKernels` (src/experts.rs), which
// lists the same names.

#include "common.cuh"
#include "expertrows_mm.cuh"

ARES_ER_MM_TYPE(iq3_s, EngMmIQ3S)
ARES_ER_MM_TYPE(iq3_xxs, EngMmIQ3XXS)
ARES_ER_MM_TYPE(iq2_xs, EngMmIQ2XS)
ARES_ER_MM_DOWN(iq2_s, EngMmIQ2S)
ARES_ER_MM_DOWN(iq2_xxs, EngMmIQ2XXS)
ARES_ER_MM_DOWN(iq1_s, EngMmIQ1S)
ARES_ER_MM_DOWN(iq1_m, EngMmIQ1M)
ARES_ER_MM_GATE_UP(q2_0, EngMmQ20)
