// The weight kernel for the float types (F32, F16, BF16).
//
// A float weight needs no codebook, no scale and no integer arithmetic, and the
// activations need no quantizing: the kernel multiplies them exactly, from the
// f32 copy that `ares_quantize_act` leaves behind (quantize_act.cuh), so the
// product has none of the error of the int8 kernels, only f32 rounding. Such a
// matrix is two to four bytes per weight, which is also the time it takes to
// read it: a fused multiply-add per weight and batch row is a fraction of what
// the GPU can do while the memory delivers a weight.
//
// Structure as the integer kernels (mmvq.cuh): a warp computes ROWS rows, each lane
// loads 16 bytes of each row (8 halves or 4 floats; two such loads per row in
// flight), the activations are read through L1 (they are the same for every
// row of the grid), and the lanes' sums meet in a warp reduction. A matrix whose
// rows are not 16-byte aligned (a row length that is not a multiple of 8, or a
// matrix at an odd address) takes a scalar path that reads one weight per lane
// and step.
#pragma once

#include "mmvq.cuh"

struct AresF32 {
    static constexpr int ELEM_BYTES = 4;
    // 8 weights from 32 bytes
    static __device__ __forceinline__ void load8(const uint8_t* p, float v[8]) {
        const uint4 a = ares_ld_stream_v4(p);
        const uint4 b = ares_ld_stream_v4(p + 16);
        v[0] = __uint_as_float(a.x);
        v[1] = __uint_as_float(a.y);
        v[2] = __uint_as_float(a.z);
        v[3] = __uint_as_float(a.w);
        v[4] = __uint_as_float(b.x);
        v[5] = __uint_as_float(b.y);
        v[6] = __uint_as_float(b.z);
        v[7] = __uint_as_float(b.w);
    }
    static __device__ __forceinline__ float one(const uint8_t* p) {
        return __uint_as_float(ares_ld_stream_u32(p));
    }
};

struct AresF16 {
    static constexpr int ELEM_BYTES = 2;
    static __device__ __forceinline__ void load8(const uint8_t* p, float v[8]) {
        const uint4 a = ares_ld_stream_v4(p);
        const uint32_t word[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[2 * i] = __half2float(__ushort_as_half(static_cast<unsigned short>(word[i])));
            v[2 * i + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(word[i] >> 16)));
        }
    }
    static __device__ __forceinline__ float one(const uint8_t* p) {
        return __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(p))));
    }
};

struct AresBF16 {
    static constexpr int ELEM_BYTES = 2;
    static __device__ __forceinline__ void load8(const uint8_t* p, float v[8]) {
        const uint4 a = ares_ld_stream_v4(p);
        const uint32_t word[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[2 * i] = __uint_as_float(word[i] << 16);
            v[2 * i + 1] = __uint_as_float(word[i] & 0xFFFF0000u);
        }
    }
    static __device__ __forceinline__ float one(const uint8_t* p) {
        return __uint_as_float(ares_ld_stream_u16(p) << 16);
    }
};

// y[b][r] = sum over c of W[r][c] * x[b][c]; the arguments are those of ares_mmvq,
// `aq` the prepared activations (the f32 copy is what is read).
template <class T, int WARPS, int ROWS>
static __device__ __forceinline__ void ares_mmvf(
    const uint8_t* __restrict__ w, const int8_t* __restrict__ aq, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch) {
    const float* __restrict__ x =
        reinterpret_cast<const float*>(aq + ares_act_x_offset(static_cast<size_t>(n_batch) * n_in));
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const size_t row_bytes = static_cast<size_t>(n_in) * T::ELEM_BYTES;
    const bool vector_path = (n_in % 8 == 0) && ((reinterpret_cast<uintptr_t>(w) | row_bytes) & 15) == 0;
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

        if (vector_path) {
            // Two chunks of 8 weights per lane and step: all their loads are issued first.
            for (uint32_t e0 = 8 * lane; e0 < n_in; e0 += 512) {
                const bool second = e0 + 256 < n_in;
                float wv[2][ROWS][8];
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    T::load8(wrow[r] + static_cast<size_t>(e0) * T::ELEM_BYTES, wv[0][r]);
                    if (second) {
                        T::load8(wrow[r] + static_cast<size_t>(e0 + 256) * T::ELEM_BYTES, wv[1][r]);
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
        } else {
            for (uint32_t e = lane; e < n_in; e += 32) {
                float wv[ROWS];
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    wv[r] = T::one(wrow[r] + static_cast<size_t>(e) * T::ELEM_BYTES);
                }
#pragma unroll
                for (int b = 0; b < ARES_MAX_BATCH; b++) {
                    if (b < static_cast<int>(n_batch)) {
                        const float xv = __ldg(x + static_cast<size_t>(b) * n_in + e);
#pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            acc[r][b] = fmaf(wv[r], xv, acc[r][b]);
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

#define ARES_MMVF_KERNEL(NAME, T)                                                              \
    extern "C" __global__ void __launch_bounds__(ARES_MMVQ_WARPS * 32) NAME(                  \
        const uint8_t* w, const int8_t* aq, float* y, uint32_t n_in, uint32_t n_out,          \
        uint32_t n_batch) {                                                                    \
        ares_mmvf<T, ARES_MMVQ_WARPS, ARES_MMVQ_ROWS>(w, aq, y, n_in, n_out, n_batch);        \
    }
