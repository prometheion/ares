// The attention module: writing K and V into the cache in the engine's own
// formats, and attention that reads them natively. One fat binary (build.rs),
// compute capability 8.0 and newer (cp.async, ldmatrix and the f16 mma of Ampere),
// loaded by `AttentionKernels` (src/attention.rs).

#include "common.cuh"
#include "attn_codec.cuh"
#include "attn_layout.cuh"
#include "attn_write.cuh"
#include "attn_kernel.cuh"

// Attention kernels (kernels/attn_kernel.cuh). Per head size: s8, s16 and s48 are the decode
// shapes (up to 8, 16 and 48 vectors, i.e. query heads of the rows of a window), w64 the wide
// step (up to 64 vectors). The arguments of the macro: head size, warps that share the
// vectors, cell groups, cells per chunk, helper warps that only load and decode, and tiles of
// 8 vectors per warp (0: the vectors are the 16 rows of the product).
// The wide step has the vectors as the rows of the product; decode has the cells as the rows
// and the vectors as columns in tiles of 8 (half the tensor-core work for one row of 6 heads).
ARES_ATTN_KERNEL(ares_attn_d256_s8, 256, 1, 2, 32, 6, 1)
ARES_ATTN_KERNEL(ares_attn_d256_s16, 256, 1, 2, 32, 6, 2)
ARES_ATTN_KERNEL(ares_attn_d256_s48, 256, 3, 2, 32, 2, 2)
ARES_ATTN_KERNEL(ares_attn_d256_w64, 256, 4, 1, 16, 4, 0)
ARES_ATTN_KERNEL(ares_attn_d128_s8, 128, 1, 2, 32, 6, 1)
ARES_ATTN_KERNEL(ares_attn_d128_s16, 128, 1, 2, 32, 6, 2)
ARES_ATTN_KERNEL(ares_attn_d128_s48, 128, 3, 2, 32, 2, 2)
ARES_ATTN_KERNEL(ares_attn_d128_w64, 128, 4, 1, 16, 4, 0)
ARES_ATTN_MERGE(ares_attn_merge_d256, 256)
ARES_ATTN_MERGE(ares_attn_merge_d128, 128)
