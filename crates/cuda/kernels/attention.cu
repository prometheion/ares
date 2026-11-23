// The attention module: writing K and V into the cache in the engine's own
// formats, and attention that reads them natively. One fat binary (build.rs),
// compute capability 8.0 and newer (cp.async, ldmatrix and the f16 mma of Ampere),
// loaded by `AttentionKernels` (src/attention.rs).

#include "common.cuh"
#include "attn_codec.cuh"
#include "attn_layout.cuh"
#include "attn_write.cuh"
#include "attn_kernel.cuh"
#include "attn_direct.cuh"
#include "attn_prompt.cuh"

// The tensor-core attention kernels (kernels/attn_kernel.cuh). Per head size: s8, s16 and s48
// are the decode shapes (up to 8, 16 and 48 vectors, i.e. query heads of the rows of a window), w64 the wide
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
// The prompt kernel (kernels/attn_prompt.cuh): items of up to 128 vectors over one range of
// cells that starts at a multiple of 32, 8 warps. The `i` kernels take converted entries with
// K in q8 to q4 and compute the scores in integers; the others take every item.
ARES_ATTN_PROMPT(ares_attn_d256_p128, 256, false)
ARES_ATTN_PROMPT(ares_attn_d256_p128i, 256, true)
ARES_ATTN_PROMPT(ares_attn_d128_p128, 128, false)
ARES_ATTN_PROMPT(ares_attn_d128_p128i, 128, true)
ARES_ATTN_MERGE(ares_attn_merge_d256, 256)
ARES_ATTN_MERGE(ares_attn_merge_d128, 128)

// The direct kernels (kernels/attn_direct.cuh), for items of up to 8 vectors over quantized K
// and V: integer dot products for the scores, f32 products in registers for the output, no
// tensor cores. Per head size and number of vectors the registers hold (the launch takes the
// smallest that fits). Blocks of 4 warps, 3 blocks per SM: src/attention.rs has the same 4.
#define ATTN_DIRECT_WARPS 4
#define ATTN_DIRECT_PER_SM 3
ARES_ATTN_DIRECT(ares_attn_d256_r2, 256, 2, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
ARES_ATTN_DIRECT(ares_attn_d256_r4, 256, 4, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
ARES_ATTN_DIRECT(ares_attn_d256_r6, 256, 6, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
ARES_ATTN_DIRECT(ares_attn_d256_r8, 256, 8, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
ARES_ATTN_DIRECT(ares_attn_d128_r2, 128, 2, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
ARES_ATTN_DIRECT(ares_attn_d128_r4, 128, 4, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
ARES_ATTN_DIRECT(ares_attn_d128_r6, 128, 6, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
ARES_ATTN_DIRECT(ares_attn_d128_r8, 128, 8, ATTN_DIRECT_WARPS, ATTN_DIRECT_PER_SM)
