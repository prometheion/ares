// The stream mixer of qwen4exp ("hyper-connection", section 3.3 of
// docs/dev/studies/qwen4exp-architecture.md): what a read and a write do
// between their three matrix products.
//
//   read:   xn[c] = rms(R[c]) * norm[c]                 ares_q4_stream
//           lo    = silu((down @ xn) / hc)              ares_q4_squash
//           m     = (1/hc) sum_c xn[c] * sigmoid(gate[c])   with gate = up @ lo    ares_q4_mix_quant
//           inj   = inject @ xn
//   write:  R[c] += out * 2 * sigmoid(inj[c] / hc)      ares_q4_stream
//
// The residual R of a row is hc streams of n values, stream-major. A write is
// always followed by the norm of the next read, so one kernel does both in one
// pass over the residual.
#pragma once

#include "common.cuh"
#include "layer_math.cuh"
#include "quantize_act.cuh"

// The write of a block's output and the norm of the next read, either or both.
// One thread block per (row, stream).
//   block_out != null:  r[row][c][i] += block_out[row][i] * 2 * sigmoid(inj[row][c] / hc)
//   norm != null:       xn[row][c][i] = r[row][c][i] / sqrt(mean_i(r[row][c]^2) + eps) * norm[c][i]
// `norm` has a weight per value and stream ([hc][n]), shared by all rows.
// n is a multiple of 4 and every buffer is 16-byte aligned.
ARES_KERNEL ares_q4_stream(float* r, const float* block_out, const float* inj, const float* norm, float* xn,
                           uint32_t rows, uint32_t hc, uint32_t n, float eps) {
    __shared__ float partial[32];
    const uint32_t row = blockIdx.x / hc, c = blockIdx.x % hc;
    if (row >= rows) {
        return;
    }
    const size_t at = (static_cast<size_t>(row) * hc + c) * n;
    float4* r4 = reinterpret_cast<float4*>(r + at);
    const uint32_t chunks = n >> 2;
    const bool write = block_out != nullptr;
    float factor = 0.0f;
    const float4* o4 = nullptr;
    if (write) {
        factor = 2.0f * ares_sigmoid(inj[static_cast<size_t>(row) * hc + c] / static_cast<float>(hc));
        o4 = reinterpret_cast<const float4*>(block_out + static_cast<size_t>(row) * n);
    }
    float ssq = 0.0f;
    for (uint32_t k = threadIdx.x; k < chunks; k += blockDim.x) {
        float4 v = r4[k];
        if (write) {
            const float4 o = o4[k];
            v.x = fmaf(o.x, factor, v.x);
            v.y = fmaf(o.y, factor, v.y);
            v.z = fmaf(o.z, factor, v.z);
            v.w = fmaf(o.w, factor, v.w);
            r4[k] = v;
        }
        ssq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    }
    if (norm == nullptr) {
        return;
    }
    ssq = ares_block_sum(ssq, partial);
    const float scale = rsqrtf(ssq / static_cast<float>(n) + eps);
    const float4* w4 = reinterpret_cast<const float4*>(norm + static_cast<size_t>(c) * n);
    float4* y4 = reinterpret_cast<float4*>(xn + at);
    for (uint32_t k = threadIdx.x; k < chunks; k += blockDim.x) {
        // The thread wrote this chunk itself a few instructions ago.
        const float4 v = r4[k];
        const float4 w = w4[k];
        y4[k] = make_float4(v.x * scale * w.x, v.y * scale * w.y, v.z * scale * w.z, v.w * scale * w.w);
    }
}

// The low-rank activation of a read: lo = silu(down / hc), the division inside
// the silu. One thread per value.
ARES_KERNEL ares_q4_squash(const float* down, float* lo, uint32_t n_values, float hc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_values) {
        lo[i] = ares_silu(down[i] / hc);
    }
}

// The scalar gate of the shared expert of the mixture of experts (section 3.7):
// out[row][i] = y[row][i] * sigmoid(s[row]) for rows of n values, one thread
// per value. In place (out == y) is allowed.
ARES_KERNEL ares_q4_gate_scale(const float* y, const float* s, float* out, uint32_t rows, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * n) {
        out[i] = y[i] * ares_sigmoid(s[i / n]);
    }
}

// The largest of / the sum over the 8 lanes of an aligned group of lanes.
static __device__ __forceinline__ float q4_group8_max(float v) {
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}
static __device__ __forceinline__ float q4_group8_sum(float v) {
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// The gated mean of a read, written as quantized activations for the products
// that follow (the layout of ares_quantize_act over rows * n values: int8
// codes, a (scale, scale * sum of codes) pair per 32 values, the f32 values):
//   m[row][i] = (1/hc) * sum_c xn[row][c][i] * sigmoid(gate[row][c][i])
// One lane per float4 of m; a group of 32 values is 8 consecutive lanes, so n
// is a multiple of 32 and the block size a multiple of 32.
ARES_KERNEL ares_q4_mix_quant(const float* xn, const float* gate, int8_t* act, uint32_t rows, uint32_t hc,
                              uint32_t n) {
    const size_t chunk = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t per_row = n >> 2;
    const size_t n_values = static_cast<size_t>(rows) * n;
    const bool live = chunk < static_cast<size_t>(rows) * per_row;
    float4 m = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (live) {
        const size_t row = chunk / per_row;
        const uint32_t k = static_cast<uint32_t>(chunk % per_row);
        const float4* x4 = reinterpret_cast<const float4*>(xn + row * hc * n);
        const float4* g4 = reinterpret_cast<const float4*>(gate + row * hc * n);
        for (uint32_t c = 0; c < hc; c++) {
            const float4 x = x4[static_cast<size_t>(c) * per_row + k];
            const float4 g = g4[static_cast<size_t>(c) * per_row + k];
            m.x = fmaf(x.x, ares_sigmoid(g.x), m.x);
            m.y = fmaf(x.y, ares_sigmoid(g.y), m.y);
            m.z = fmaf(x.z, ares_sigmoid(g.z), m.z);
            m.w = fmaf(x.w, ares_sigmoid(g.w), m.w);
        }
        const float count = static_cast<float>(hc);
        m = make_float4(m.x / count, m.y / count, m.z / count, m.w / count);
    }
    // The arithmetic of ares_quantize_act, 4 values per lane.
    const float amax = q4_group8_max(fmaxf(fmaxf(fabsf(m.x), fabsf(m.y)), fmaxf(fabsf(m.z), fabsf(m.w))));
    const float d = amax / 127.0f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    if (amax != 0.0f) {
        c0 = roundf(m.x / d);
        c1 = roundf(m.y / d);
        c2 = roundf(m.z / d);
        c3 = roundf(m.w / d);
    }
    // The codes are integers of at most 127 in magnitude: their f32 sum is exact in any order.
    const float sum_codes = q4_group8_sum((c0 + c1) + (c2 + c3));
    if (live) {
        const uint32_t packed = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c0))) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c1))) << 8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c2))) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c3))) << 24);
        uint8_t* base = reinterpret_cast<uint8_t*>(act);
        reinterpret_cast<uint32_t*>(base)[chunk] = packed;
        reinterpret_cast<float4*>(base + ares_act_x_offset(n_values))[chunk] = m;
        if ((threadIdx.x & 7) == 0) {
            reinterpret_cast<float2*>(base + ares_act_ds_offset(n_values))[chunk >> 3] =
                make_float2(d, d * sum_codes);
        }
    }
}
