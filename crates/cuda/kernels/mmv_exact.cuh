// The weight kernel that multiplies unquantized activations with a matrix of any
// quantized type: `y[b] = W x[b]`, the weights decoded as ggml decodes them, the sums f32.
//
// The decode kernels of the quantized types (mmvq_*.cuh) multiply int8 activations, about
// 1 % rms from the exact product. Some matrices take values that are not quantized at
// all: those of qwen4exp's stream mixers and n-gram block, which the engine keeps exact
// (the reference multiplies them in f32), and which a file may still store in a quantized
// type. This kernel is the float kernel (mmvq_float.cuh) with a decoder per value in
// place of the float loads: each lane decodes the 8 weights it multiplies
// (weight_value.cuh) and the order of the sums is that of the float kernel, so a matrix
// gives the same bits as its decoded copy in F32 there. It reads the f32 section of the
// prepared activations, as the float kernel does.
//
// Such a matrix is read once per step, like any other, and its decoding is a handful of
// instructions a weight: these matrices are small (the largest, an n-gram key projection,
// is 27 MiB in Q8_0), so the decode costs time only where it is many rows.
#pragma once

#include "mmvq.cuh"
#include "weight_value.cuh"

// y[b][r] = sum over c of W[r][c] * x[b][c] for a matrix of type `TY` (rows of whole
// blocks); the arguments are those of ares_mmvq, `aq` the prepared activations (the f32
// copy is what is read).
template <uint32_t TY, int WARPS, int ROWS>
static __device__ __forceinline__ void ares_mmv_exact(const uint8_t* __restrict__ w, const int8_t* __restrict__ aq,
                                                      float* __restrict__ y, uint32_t n_in, uint32_t n_out,
                                                      uint32_t n_batch) {
    using T = AresWeight<TY>;
    const float* __restrict__ x =
        reinterpret_cast<const float*>(aq + ares_act_x_offset(static_cast<size_t>(n_batch) * n_in));
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const size_t row_bytes = static_cast<size_t>(n_in / T::VALUES) * T::BYTES;
    const uint32_t n_tiles = (n_out + WARPS * ROWS - 1) / (WARPS * ROWS);

    for (uint32_t tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const uint32_t row0 = (tile * WARPS + warp) * ROWS;
        const uint8_t* wrow[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            wrow[r] = w + static_cast<size_t>(min(row0 + r, n_out - 1)) * row_bytes;
        }
        float acc[ROWS][ARES_MAX_BATCH];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                acc[r][b] = 0.0f;
            }
        }
        // Two chunks of 8 weights per lane and step, as the float kernel's vector path: every
        // block of every type is a whole number of chunks of 8.
        for (uint32_t e0 = 8 * lane; e0 < n_in; e0 += 512) {
            const bool second = e0 + 256 < n_in;
            float wv[2][ROWS][8];
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    wv[0][r][k] = T::value(wrow[r], e0 + k);
                    wv[1][r][k] = second ? T::value(wrow[r], e0 + 256 + k) : 0.0f;
                }
            }
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                if (b < static_cast<int>(n_batch)) {
                    const float* xb = x + static_cast<size_t>(b) * n_in;
                    const float4 xa = __ldg(reinterpret_cast<const float4*>(xb + e0));
                    const float4 xc = __ldg(reinterpret_cast<const float4*>(xb + e0 + 4));
                    const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w};
#pragma unroll
                    for (int r = 0; r < ROWS; r++) {
#pragma unroll
                        for (int k = 0; k < 8; k++) {
                            acc[r][b] = fmaf(wv[0][r][k], xv[k], acc[r][b]);
                        }
                    }
                    if (second) {
                        const float4 ya = __ldg(reinterpret_cast<const float4*>(xb + e0 + 256));
                        const float4 yc = __ldg(reinterpret_cast<const float4*>(xb + e0 + 260));
                        const float yv[8] = {ya.x, ya.y, ya.z, ya.w, yc.x, yc.y, yc.z, yc.w};
#pragma unroll
                        for (int r = 0; r < ROWS; r++) {
#pragma unroll
                            for (int k = 0; k < 8; k++) {
                                acc[r][b] = fmaf(wv[1][r][k], yv[k], acc[r][b]);
                            }
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                if (b < static_cast<int>(n_batch)) {
                    const float total = ares_warp_sum(acc[r][b]);
                    if (lane == 0 && row0 + r < n_out) {
                        y[static_cast<size_t>(b) * n_out + row0 + r] = total;
                    }
                }
            }
        }
    }
}

#define ARES_MMV_EXACT_KERNEL(NAME, TY)                                                        \
    extern "C" __global__ void __launch_bounds__(ARES_MMVQ_WARPS * 32) NAME(                  \
        const uint8_t* w, const int8_t* aq, float* y, uint32_t n_in, uint32_t n_out,          \
        uint32_t n_batch) {                                                                    \
        ares_mmv_exact<TY, ARES_MMVQ_WARPS, ARES_MMVQ_ROWS>(w, aq, y, n_in, n_out, n_batch);  \
    }
