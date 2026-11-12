// The shapes of the prefill kernels (mm.cuh) as launch macros, shared by the modules that
// instantiate them (prefill.cu, prefill_more.cu); the Rust side (src/kernels.rs) lists the
// same sizes.
#pragma once

#include "mm.cuh"

// The shapes: weight rows x activation rows per block, warps, blocks per SM.
// Found by measurement on an RTX 4070 and an RTX 3060 (see mm.cuh); the Rust side
// (src/kernels.rs) picks one by the number of activation rows and lists the same
// sizes, including the shared memory each needs.
//
//   large   128 x 128   4 x 4 warps (512 threads), 1 block per SM    more than 64 rows
//   medium   64 x  64   2 x 2 warps (128 threads), 3 (int8) or 2 (f16) blocks per SM   33 to 64
//   small    64 x  32   2 x 2 warps (128 threads), 3 blocks per SM    up to 32 rows
#define ARES_MM_I8_LARGE(NAME, T) ARES_MM_KERNEL(NAME, 512, 1, ares_mm_i8, T, 128, 128, 4, 4, 64, int8_t)
#define ARES_MM_I8_MEDIUM(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_i8, T, 64, 64, 2, 2, 64, int8_t)
#define ARES_MM_I8_SMALL(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_i8, T, 64, 32, 2, 2, 64, int8_t)
#define ARES_MM_F16_LARGE(NAME, T) ARES_MM_KERNEL(NAME, 512, 1, ares_mm_f16, T, 128, 128, 4, 4, 64, __half)
#define ARES_MM_F16_MEDIUM(NAME, T) ARES_MM_KERNEL(NAME, 128, 2, ares_mm_f16, T, 64, 64, 2, 2, 64, __half)
#define ARES_MM_F16_SMALL(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_f16, T, 64, 32, 2, 2, 64, __half)

// The f16 variant with f16 accumulation, flushed to f32 every ARES_MM_FLUSH mma of 16
// values of K (see mm.cuh): ares_mm_h<|m|s>_<type>. The GeForce cards run it at twice
// the rate of the f32 accumulation.
#define ARES_MM_FLUSH 4
#define ARES_MM_H_LARGE(NAME, T) ARES_MM_KERNEL(NAME, 512, 1, ares_mm_f16, T, 128, 128, 4, 4, 64, __half, ARES_MM_FLUSH)
#define ARES_MM_H_MEDIUM(NAME, T) ARES_MM_KERNEL(NAME, 128, 2, ares_mm_f16, T, 64, 64, 2, 2, 64, __half, ARES_MM_FLUSH)
#define ARES_MM_H_SMALL(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_f16, T, 64, 32, 2, 2, 64, __half, ARES_MM_FLUSH)
