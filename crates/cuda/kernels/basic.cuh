// Small kernels over f32 rows.
#pragma once

#include "common.cuh"

// y[i] += x[i] for i < n. One thread per value.
//
// This is the combine step of a sum across devices: a device adds the partial
// result it received from its peer to its own.
ARES_KERNEL ares_add_f32(float* y, const float* x, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += x[i];
    }
}

// out[i] = a[i] + b[i] for i < n. One thread per value. out may be a or b.
ARES_KERNEL ares_add_to_f32(float* out, const float* a, const float* b, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// The longest row, in values per thread, that ares_rms_norm_f32 keeps in
// registers (a row of 256 * 24 = 6144 values with the launch of 256 threads).
#define ARES_NORM_RESIDENT 24

// RMS normalisation of rows: y = x / sqrt(mean(x^2) + eps) * weight.
// `weight` may be null for a plain normalisation. One block per row.
//
// A normalisation is latency, not work: one block reads a row, reduces it and
// writes it back. The straightforward version reads the row twice, with the
// reduction in between, which is two dependent round trips to memory; for a row
// that fits, each thread therefore loads its values once, all loads in flight
// together, and keeps them in registers for the scaling. (Measured for 5120
// values: 6.5 us -> 4.4 us on an RTX 4070 and 10.6 -> 5.3 us on an RTX 3060.) The
// sums are formed in the same order as in the loop for longer rows, so the
// result is the same bit for bit.
ARES_KERNEL ares_rms_norm_f32(const float* x, const float* weight, float* y, uint32_t n, float eps) {
    __shared__ float partial[32];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
    float* yr = y + static_cast<size_t>(blockIdx.x) * n;

    if (n <= blockDim.x * ARES_NORM_RESIDENT) {
        float v[ARES_NORM_RESIDENT];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < ARES_NORM_RESIDENT; k++) {
            const uint32_t i = threadIdx.x + k * blockDim.x;
            v[k] = i < n ? xr[i] : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < ARES_NORM_RESIDENT; k++) {
            sum += v[k] * v[k];
        }
        sum = ares_block_sum(sum, partial);
        const float scale = rsqrtf(sum / static_cast<float>(n) + eps);
#pragma unroll
        for (int k = 0; k < ARES_NORM_RESIDENT; k++) {
            const uint32_t i = threadIdx.x + k * blockDim.x;
            if (i < n) {
                yr[i] = v[k] * scale * (weight != nullptr ? weight[i] : 1.0f);
            }
        }
        return;
    }

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += xr[i] * xr[i];
    }
    sum = ares_block_sum(sum, partial);
    const float scale = rsqrtf(sum / static_cast<float>(n) + eps);

    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = xr[i] * scale * (weight != nullptr ? weight[i] : 1.0f);
    }
}

// out[g] = the sum of x[32 g .. 32 g + 31] for g < n_groups. One thread per
// group. Quantized types with an offset per sub-block need these sums once
// per activation row, not once per weight row.
ARES_KERNEL ares_sum32_f32(const float* x, float* out, uint32_t n_groups) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n_groups) {
        const float* xg = x + static_cast<size_t>(g) * 32;
        float sum = 0.0f;
        for (int l = 0; l < 32; l++) {
            sum += xg[l];
        }
        out[g] = sum;
    }
}
