// Many-row expert kernels of more weight types (see expertrows.cu and expertrows_mm.cuh):
// the legacy 4- and 5-bit types and the ternary ones. A module of its own so that the compiler runs side by side with the
// others; built like expertrows.cu and loaded by `ExpertRowKernels` (src/experts.rs).

#include "common.cuh"
#include "expertrows_mm.cuh"

ARES_ER_MM_TYPE(q4_0, EngMmQ40)
ARES_ER_MM_TYPE(q4_1, EngMmQ41)
ARES_ER_MM_TYPE(q5_0, EngMmQ50)
ARES_ER_MM_TYPE(q5_1, EngMmQ51)
ARES_ER_MM_TYPE(tq1_0, EngMmTQ10)
ARES_ER_MM_TYPE(tq2_0, EngMmTQ20)
