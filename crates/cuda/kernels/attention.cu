// The attention module: writing K and V into the cache in the engine's own
// formats, and attention that reads them natively. One fat binary (build.rs),
// compute capability 8.0 and newer (cp.async, ldmatrix and the f16 mma of Ampere),
// loaded by `AttentionKernels` (src/attention.rs).

#include "common.cuh"
#include "attn_codec.cuh"
#include "attn_layout.cuh"
#include "attn_write.cuh"
#include "attn_kernel.cuh"

// Attention kernels: D, m-tiles of 16 vectors (MT), cell groups (CG), cells per chunk.
//   m1c2k32  up to 16 vectors (one row of a decode step), two warps per tile; xN: N more warps that only load and decode
//   m3c2k32  up to 48 vectors (eight rows of a decode step)
//   m4c1k16  up to 64 vectors, the wide step
ARES_ATTN_KERNEL(ares_attn_d256_m1c2k32x6, 256, 1, 2, 32, 6)
ARES_ATTN_KERNEL(ares_attn_d256_m3c2k32x2, 256, 3, 2, 32, 2)
ARES_ATTN_KERNEL(ares_attn_d256_m4c1k16x4, 256, 4, 1, 16, 4)
ARES_ATTN_KERNEL(ares_attn_d128_m1c2k32x6, 128, 1, 2, 32, 6)
ARES_ATTN_KERNEL(ares_attn_d128_m3c2k32x2, 128, 3, 2, 32, 2)
ARES_ATTN_KERNEL(ares_attn_d128_m4c1k16x4, 128, 4, 1, 16, 4)
ARES_ATTN_MERGE(ares_attn_merge_d256, 256)
ARES_ATTN_MERGE(ares_attn_merge_d128, 128)
