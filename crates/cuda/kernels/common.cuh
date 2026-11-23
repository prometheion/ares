// Shared helpers of the kernels.
//
// Conventions:
// - Every kernel is `extern "C"` and prefixed `ares_`, so the host finds it by
//   its plain name.
// - Kernels take sizes as arguments. Nothing about a batch is compiled in, so
//   a change of batch shape costs nothing on the device.
// - Block sizes are multiples of 32: the reductions below shuffle within full
//   warps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#define ARES_KERNEL extern "C" __global__ void

// The most rows of activations one launch of a weight kernel handles.
#define ARES_MAX_BATCH 8

// The sum of `v` over the 32 lanes of a warp, returned to every lane.
static __device__ __forceinline__ float ares_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// The sum of `v` over all threads of the block, returned to every thread.
// `partial` is shared memory with one float per warp. A caller that reuses
// `partial` must __syncthreads() before the next call.
static __device__ __forceinline__ float ares_block_sum(float v, float* partial) {
    v = ares_warp_sum(v);
    const uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        partial[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    const uint32_t n_warps = blockDim.x >> 5;
    return ares_warp_sum(lane < n_warps ? partial[lane] : 0.0f);
}

static __device__ __forceinline__ float ares_half_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// c + the dot product of the four signed bytes of a with the four signed bytes
// of b.
static __device__ __forceinline__ int ares_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

// The same with the four bytes of `a` taken as unsigned (0..255) and those of
// `b` as signed. Quantized weights are unsigned codes, activations are signed,
// and `__dp4a` has no CUDA C++ overload for the mix, so this is the PTX
// instruction itself.
static __device__ __forceinline__ int ares_dp4a_us(uint32_t a, int b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
