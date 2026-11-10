// Scalar functions of the layer kernels, in f32 with the full-precision expf.
#pragma once

#include "common.cuh"

static __device__ __forceinline__ float ares_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float ares_silu(float x) {
    return x / (1.0f + expf(-x));
}

// ln(1 + e^x), x itself above 20 (where the difference is below f32 rounding).
static __device__ __forceinline__ float ares_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}
