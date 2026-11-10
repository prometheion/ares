// Matrix times vectors for Q4_K weights, with f32 activations.
//
// This is the first weight kernel and the reference for the ones to come: it
// is written to be obviously right and is verified against the CPU decoder.
// It reads each weight byte once and keeps the activations in f32. The tuned
// decode kernels (integer dot products over quantized activations, tables per
// GPU generation) and the prefill path (tensor-core GEMM with on-the-fly
// dequantisation) replace it where they are faster, and are held to its
// results.
#pragma once

#include "common.cuh"

// A Q4_K block holds 256 values in 144 bytes:
//   0   f16 d      scale of the sub-block scales
//   2   f16 dmin   scale of the sub-block minimums
//   4   u8[12]     8 scales and 8 minimums, 6 bits each
//   16  u8[128]    4 bits per value: of each 32 bytes the low nibbles are
//                  values 0..31 of a 64-value unit, the high nibbles 32..63
#define ARES_Q4K_BLOCK_BYTES 144

// The 6-bit scale and minimum of sub-block j (0..7), as ggml packs them.
static __device__ __forceinline__ void ares_q4k_scale_min(
    uint32_t j, const uint8_t* s, uint32_t& scale, uint32_t& minimum) {
    if (j < 4) {
        scale = s[j] & 63;
        minimum = s[j + 4] & 63;
    } else {
        scale = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        minimum = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// y[b][r] = sum over c of W[r][c] * x[b][c].
//
//   w     n_out rows of n_in / 256 blocks, 16-byte aligned
//   x     n_batch rows of n_in values
//   xsum  n_batch rows of n_in / 32 sums (ares_sum32_f32 of x)
//   y     n_batch rows of n_out values
//
// One block per output row; n_batch <= ARES_MAX_BATCH. A thread takes units of
// 64 values: it decodes the unit's 32 weight bytes once and applies them to
// every activation row, so the weights are read once however many rows of
// activations there are.
ARES_KERNEL ares_matvec_q4_K(
    const uint8_t* w, const float* x, const float* xsum, float* y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch) {
    __shared__ float partial[32];

    const uint32_t row = blockIdx.x;
    const uint32_t blocks_per_row = n_in >> 8;
    const uint32_t n_units = blocks_per_row * 4;
    const uint8_t* wr = w + static_cast<size_t>(row) * blocks_per_row * ARES_Q4K_BLOCK_BYTES;

    float acc[ARES_MAX_BATCH];
    for (uint32_t b = 0; b < ARES_MAX_BATCH; b++) {
        acc[b] = 0.0f;
    }

    for (uint32_t u = threadIdx.x; u < n_units; u += blockDim.x) {
        const uint8_t* block = wr + static_cast<size_t>(u >> 2) * ARES_Q4K_BLOCK_BYTES;
        const uint32_t j = u & 3;

        const float d = ares_half_at(block);
        const float dmin = ares_half_at(block + 2);
        uint32_t s_lo, m_lo, s_hi, m_hi;
        ares_q4k_scale_min(2 * j, block + 4, s_lo, m_lo);
        ares_q4k_scale_min(2 * j + 1, block + 4, s_hi, m_hi);
        const float d_lo = d * static_cast<float>(s_lo);
        const float d_hi = d * static_cast<float>(s_hi);
        const float o_lo = dmin * static_cast<float>(m_lo);
        const float o_hi = dmin * static_cast<float>(m_hi);

        const uint32_t* q = reinterpret_cast<const uint32_t*>(block + 16 + j * 32);
        uint32_t words[8];
        for (int k = 0; k < 8; k++) {
            words[k] = q[k];
        }

        const uint32_t col = u * 64;
        for (uint32_t b = 0; b < n_batch; b++) {
            const float* xb = x + static_cast<size_t>(b) * n_in + col;
            float lo = 0.0f;
            float hi = 0.0f;
            for (int k = 0; k < 8; k++) {
                const uint32_t word = words[k];
                for (int t = 0; t < 4; t++) {
                    const uint32_t byte = (word >> (8 * t)) & 0xFF;
                    lo += static_cast<float>(byte & 0xF) * xb[k * 4 + t];
                    hi += static_cast<float>(byte >> 4) * xb[32 + k * 4 + t];
                }
            }
            const float* sums = xsum + static_cast<size_t>(b) * (n_in >> 5) + (col >> 5);
            acc[b] += d_lo * lo + d_hi * hi - o_lo * sums[0] - o_hi * sums[1];
        }
    }

    for (uint32_t b = 0; b < n_batch; b++) {
        const float total = ares_block_sum(acc[b], partial);
        if (threadIdx.x == 0) {
            y[static_cast<size_t>(b) * n_out + row] = total;
        }
        __syncthreads();
    }
}
