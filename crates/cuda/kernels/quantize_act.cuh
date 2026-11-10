// Quantization of activations for the integer weight kernels (mmvq_*.cuh).
//
// A decode step reads every weight once, so the weight kernels are bound by
// memory and should spend as little as possible per weight byte. They do that
// with integer dot products (`dp4a`: four int8 products and a sum per
// instruction), which needs the activations as int8 too. The activations are
// few (one row of n_in values per sequence) and are quantized once per input
// of a group of matrices, not once per matrix.
//
// Format, per activation row of n_in values (n_in a multiple of 32):
//   q   n_in int8 values
//   ds  n_in / 32 pairs of f32: ds[g] = (d, s) for the group g of 32 values
//       x[32 g + i] ~ x^[32 g + i] = d * q[32 g + i], with d = max|x| / 127 of
//       the group, q = round-half-away-from-zero(x / d), and s = d * sum(q), the
//       sum of the x^ of the group. Types whose values have an offset per
//       sub-block (Q4_K, Q5_K) multiply the offset by s, so the offset cancels
//       the offset of the codes exactly and a weight kernel computes W x^, the
//       product of the exact weights with the dequantized activations, up to
//       f32 rounding. (The sum of the unquantized x would look more exact but
//       measures worse: on a Q4_K matrix of Qwen3.8-27B with synthetic
//       heavy-tailed activations it gave 1.4 times the error at 1 row and 2.6
//       times at 4, because it no longer cancels the rounding error that the
//       codes' mean makes.)
// The rows of a batch follow each other in both arrays, and the pairs start at
// the next multiple of 16 bytes after the values (ares_act_ds_offset).
//
// The scale is f32, not f16: the whole activation set is small next to the
// weights, and the rounding of an f16 scale would cost accuracy for no speed.
#pragma once

#include "common.cuh"

// Byte offset of the (d, s) pairs behind `n_values` = n_batch * n_in int8 values.
static __host__ __device__ __forceinline__ size_t ares_act_ds_offset(size_t n_values) {
    return (n_values + 15) & ~static_cast<size_t>(15);
}

// One thread per value; a group of 32 values is one warp (blocks are multiples
// of 32 threads and n_values a multiple of 32), so the group's maximum and sum
// of codes are warp reductions. Threads past n_values stay in the warp with a
// zero so the shuffles are complete.
ARES_KERNEL ares_quantize_act(const float* x, int8_t* q, uint32_t n_values) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n_values ? x[i] : 0.0f;

    float amax = fabsf(v);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }

    const float d = amax / 127.0f;
    const float code = amax == 0.0f ? 0.0f : roundf(v / d);
    // The codes are integers of at most 127 in magnitude, so this f32 sum is exact
    // in any order.
    const float sum_codes = ares_warp_sum(i < n_values ? code : 0.0f);
    if (i < n_values) {
        q[i] = static_cast<int8_t>(code);
        if ((threadIdx.x & 31) == 0) {
            float2* ds = reinterpret_cast<float2*>(reinterpret_cast<uint8_t*>(q) + ares_act_ds_offset(n_values));
            ds[i >> 5] = make_float2(d, d * sum_codes);
        }
    }
}
