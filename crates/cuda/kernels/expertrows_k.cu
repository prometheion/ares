// Many-row expert kernels of more weight types (see expertrows.cu and expertrows_mm.cuh):
// the K-quants, Q8_0 and the 4-bit i-quants. A module of its own so that the compiler runs side by side with the
// others; built like expertrows.cu and loaded by `ExpertRowKernels` (src/experts.rs), which
// lists the same names.

#include "common.cuh"
#include "expertrows_mm.cuh"

ARES_ER_MM_TYPE(q4_K, EngMmQ4K)
ARES_ER_MM_TYPE(q5_K, EngMmQ5K)
ARES_ER_MM_TYPE(q6_K, EngMmQ6K)
ARES_ER_MM_TYPE(q3_K, EngMmQ3K)
ARES_ER_MM_TYPE(q2_K, EngMmQ2K)
ARES_ER_MM_TYPE(q8_0, EngMmQ80)
ARES_ER_MM_TYPE(iq4_xs, EngMmIQ4XS)
ARES_ER_MM_TYPE(iq4_nl, EngMmIQ4NL)
