// Norms, the rotary table, the preparation of q, k and the gate for attention,
// the gated norm and the SwiGLU with their activation quantization, and the
// gather of the rows whose logits are wanted.
//
// The quantization writes exactly the layout of `ares_quantize_act`
// (quantize_act.cuh): int8 codes, then one (scale, scale * sum of codes) pair of
// f32 per group of 32 values, then a copy of the activations in f32, each array
// at the next multiple of 16 bytes (ares_act_ds_offset, ares_act_x_offset), so
// the weight kernels cannot tell a fused preparation from a separate one
// (the tests compare the bytes).
#pragma once

#include <cstring>

#include "common.cuh"
#include "layer_math.cuh"
#include "layer_step.cuh"
#include "quantize_act.cuh"

// Sum of `v` over the `n` lanes of an aligned group of lanes (n = 2, 4, 8, ...).
template <int N>
static __device__ __forceinline__ float ares_group_sum(float v) {
#pragma unroll
    for (int offset = N / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <int N>
static __device__ __forceinline__ float ares_group_max(float v) {
#pragma unroll
    for (int offset = N / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Quantizes one float4 chunk (4 consecutive values) of a buffer of `n_values`
// values. Groups of 32 values are 8 consecutive chunks held by 8 consecutive
// lanes of an aligned group, so every lane of the warp must call this with the
// same control flow (lanes without data pass zeros and `live` = false).
// `chunk` is the index of the float4 in the whole buffer. The arithmetic is
// that of ares_quantize_act.
static __device__ __forceinline__ void ares_quant_chunk(float4 v, bool live, size_t chunk, int8_t* q,
                                                        size_t n_values) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = ares_group_max<8>(amax);
    const float d = amax / 127.0f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    if (amax != 0.0f) {
        c0 = roundf(v.x / d);
        c1 = roundf(v.y / d);
        c2 = roundf(v.z / d);
        c3 = roundf(v.w / d);
    }
    // The codes are integers of at most 127 in magnitude: exact in any order.
    const float sum_codes = ares_group_sum<8>((c0 + c1) + (c2 + c3));
    if (live) {
        const uint32_t packed = (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c0)))) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c1))) << 8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c2))) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(c3))) << 24);
        reinterpret_cast<uint32_t*>(q)[chunk] = packed;
        uint8_t* base = reinterpret_cast<uint8_t*>(q);
        reinterpret_cast<float4*>(base + ares_act_x_offset(n_values))[chunk] = v;
        if ((threadIdx.x & 7) == 0) {
            reinterpret_cast<float2*>(base + ares_act_ds_offset(n_values))[chunk >> 3] =
                make_float2(d, d * sum_codes);
        }
    }
}

// ---------------------------------------------------------------------------
// RMS norm (or L2 norm) of rows of `w` values, one warp per row, an optional
// weight shared by all rows. w is a multiple of 4 and at most 1024.
//   mean != 0: y = x / sqrt(mean(x^2) + eps) * weight       (rms norm)
//   mean == 0: y = x / sqrt(sum(x^2) + eps) * weight        (the q/k norm of the delta rule)
// In place is allowed (y == x).
#define ARES_HEAD_NORM_MAX_CHUNKS 8

ARES_KERNEL ares_head_norm(const float* x, const float* weight, float* y, uint32_t rows, uint32_t w, float eps,
                           uint32_t mean) {
    const uint32_t lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) {
        return;
    }
    const uint32_t chunks = w >> 2;
    const float4* xr = reinterpret_cast<const float4*>(x + row * w);
    float4* yr = reinterpret_cast<float4*>(y + row * w);
    float4 v[ARES_HEAD_NORM_MAX_CHUNKS];
    float ssq = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_HEAD_NORM_MAX_CHUNKS; k++) {
        const uint32_t c = lane + 32 * k;
        v[k] = c < chunks ? xr[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        ssq += v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
    }
    ssq = ares_warp_sum(ssq);
    const float scale = rsqrtf((mean != 0 ? ssq / static_cast<float>(w) : ssq) + eps);
#pragma unroll
    for (int k = 0; k < ARES_HEAD_NORM_MAX_CHUNKS; k++) {
        const uint32_t c = lane + 32 * k;
        if (c < chunks) {
            float4 o = make_float4(v[k].x * scale, v[k].y * scale, v[k].z * scale, v[k].w * scale);
            if (weight != nullptr) {
                const float4 wt = reinterpret_cast<const float4*>(weight)[c];
                o.x *= wt.x;
                o.y *= wt.y;
                o.z *= wt.z;
                o.w *= wt.w;
            }
            yr[c] = o;
        }
    }
}

// ---------------------------------------------------------------------------
// The rows whose logits are wanted: y[i] = norm(x[out_rows[i]]) for i < n_out,
// with an RMS norm by `weight` when it is not null and a plain copy otherwise.
// One block per output row.
ARES_KERNEL ares_gather_rows(const AresStep* step, const float* x, const float* weight, float* y, uint32_t d,
                             float eps) {
    __shared__ float partial[32];
    const uint32_t i = blockIdx.x;
    if (i >= step->n_out) {
        return;
    }
    const float* xr = x + static_cast<size_t>(ares_step_out_rows(step)[i]) * d;
    float* yr = y + static_cast<size_t>(i) * d;
    if (weight == nullptr) {
        for (uint32_t j = threadIdx.x; j < d; j += blockDim.x) {
            yr[j] = xr[j];
        }
        return;
    }
    float sum = 0.0f;
    for (uint32_t j = threadIdx.x; j < d; j += blockDim.x) {
        sum += xr[j] * xr[j];
    }
    sum = ares_block_sum(sum, partial);
    const float scale = rsqrtf(sum / static_cast<float>(d) + eps);
    for (uint32_t j = threadIdx.x; j < d; j += blockDim.x) {
        yr[j] = xr[j] * scale * weight[j];
    }
}

// ---------------------------------------------------------------------------
// The rotary table of a step: for every row and every rotated pair p < rot / 2,
// (cos, sin) of pos * freq_base^(-2p / rot), with the position kind of the pair
// chosen by the sections as the architecture document defines (interleaved
// layout: sector = p mod (s0 + s1 + s2 + s3); kind 0 (time) for sector % 3 == 0
// below 3 * s0, kind 1 (height) below 3 * s1, kind 2 (width) below 3 * s2, the
// extra position otherwise). The angle is computed in f64: positions reach
// 262144 and an f32 angle would be off by 1e-2 radians there. The table is
// the same for every attention layer, so a step builds it once.
// One block per row, one thread per pair.
ARES_KERNEL ares_rope_table(const AresStep* step, float2* table, uint32_t rot, float freq_base, uint32_t s0,
                            uint32_t s1, uint32_t s2, uint32_t s3) {
    const uint32_t row = blockIdx.x;
    const uint32_t p = threadIdx.x;
    const uint32_t half = rot >> 1;
    if (row >= step->n_rows || p >= half) {
        return;
    }
    const uint32_t total = s0 + s1 + s2 + s3;
    const uint32_t sector = p % total;
    const uint32_t kind = sector % 3;
    int stream = 3;
    if (kind == 0 && sector < 3 * s0) {
        stream = 0;
    } else if (kind == 1 && sector < 3 * s1) {
        stream = 1;
    } else if (kind == 2 && sector < 3 * s2) {
        stream = 2;
    }
    const double position = static_cast<double>(ares_step_pos(step)[static_cast<size_t>(row) * 4 + stream]);
    const double freq = pow(static_cast<double>(freq_base), -2.0 * static_cast<double>(p) / static_cast<double>(rot));
    double s, c;
    sincos(position * freq, &s, &c);
    table[static_cast<size_t>(row) * half + p] = make_float2(static_cast<float>(c), static_cast<float>(s));
}

// ---------------------------------------------------------------------------
// Preparation of the attention input of a step, one warp per (row, head), q
// heads first and then k heads:
//   q head:  the query is the first `hd` values of the head's `2 hd` of the
//            projection `qg`, the gate the second `hd`; query = rope(rms(query) * q_norm)
//   k head:  rope(rms(k) * k_norm)
//   v is not touched here.
// The gate is copied to `gate_out` unless that is null (attention may then read
// it from `qg` with a stride). hd is a multiple of 32 and at most 256; the first
// `rot` values of a head are rotated, in pairs (p, p + rot / 2).
#define ARES_PREP_MAX_HD 256

ARES_KERNEL ares_attn_prep(const AresStep* step, const float* qg, const float* k_in, const float* q_norm,
                           const float* k_norm, const float2* table, float* q_out, float* gate_out,
                           float* k_out, uint32_t n_head, uint32_t n_kv_head, uint32_t hd, uint32_t rot,
                           float eps) {
    __shared__ float stage[8][ARES_PREP_MAX_HD];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t item = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    const uint32_t slots = n_head + n_kv_head;
    const size_t row = item / slots;
    const uint32_t slot = static_cast<uint32_t>(item % slots);
    if (row >= step->n_rows) {
        return;
    }
    const bool is_q = slot < n_head;
    const float* src = is_q ? qg + row * n_head * 2 * hd + static_cast<size_t>(slot) * 2 * hd
                            : k_in + row * n_kv_head * hd + static_cast<size_t>(slot - n_head) * hd;
    const float* weight = is_q ? q_norm : k_norm;
    float* dst = is_q ? q_out + row * n_head * hd + static_cast<size_t>(slot) * hd
                      : k_out + row * n_kv_head * hd + static_cast<size_t>(slot - n_head) * hd;
    float ssq = 0.0f;
    for (uint32_t i = lane; i < hd; i += 32) {
        const float v = src[i];
        stage[warp][i] = v;
        ssq += v * v;
    }
    ssq = ares_warp_sum(ssq);
    const float scale = rsqrtf(ssq / static_cast<float>(hd) + eps);
    __syncwarp();
    // Normalise in place in the stage so that the rotation reads normalised pairs.
    for (uint32_t i = lane; i < hd; i += 32) {
        stage[warp][i] = stage[warp][i] * scale * weight[i];
    }
    __syncwarp();
    const uint32_t half = rot >> 1;
    for (uint32_t i = lane; i < hd; i += 32) {
        float o = stage[warp][i];
        if (i < rot) {
            const uint32_t p = i < half ? i : i - half;
            const float2 cs = table[row * half + p];
            const float a = stage[warp][p], b = stage[warp][p + half];
            o = i < half ? a * cs.x - b * cs.y : a * cs.y + b * cs.x;
        }
        dst[i] = o;
    }
    if (is_q && gate_out != nullptr) {
        float* g = gate_out + row * n_head * hd + static_cast<size_t>(slot) * hd;
        for (uint32_t i = lane; i < hd; i += 32) {
            g[i] = src[hd + i];
        }
    }
}

// ---------------------------------------------------------------------------
// The gated norm of the recurrent block with the quantization of what follows:
// for every (row, head) of `w` values (w a multiple of 32, at most 256):
//   y = o / sqrt(mean(o^2) + eps) * weight * gate(z),  gate = silu (gate_kind 0) or sigmoid (1)
// and y quantized for the output projection into `q` in the layout of
// ares_quantize_act over all rows * heads * w values. One warp per item.
ARES_KERNEL ares_gated_norm_quant(const float* o, const float* z, const float* weight, int8_t* q,
                                  uint32_t items, uint32_t w, float eps, uint32_t gate_kind) {
    const uint32_t lane = threadIdx.x & 31;
    const size_t item = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    // Whole warps stay in the shuffles; a warp past the end does nothing at all.
    if (item >= items) {
        return;
    }
    const uint32_t chunks = w >> 2;
    const size_t n_values = static_cast<size_t>(items) * w;
    const float4* orow = reinterpret_cast<const float4*>(o + item * w);
    const float4* zrow = reinterpret_cast<const float4*>(z + item * w);
    float4 v[2];
    float ssq = 0.0f;
#pragma unroll
    for (int k = 0; k < 2; k++) {
        const uint32_t c = lane + 32 * k;
        v[k] = c < chunks ? orow[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        ssq += v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
    }
    ssq = ares_warp_sum(ssq);
    const float scale = rsqrtf(ssq / static_cast<float>(w) + eps);
#pragma unroll
    for (int k = 0; k < 2; k++) {
        const uint32_t c = lane + 32 * k;
        float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (c < chunks) {
            const float4 zz = zrow[c];
            const float4 wt = reinterpret_cast<const float4*>(weight)[c];
            const float4 vv = v[k];
            if (gate_kind == 0) {
                y = make_float4(vv.x * scale * wt.x * ares_silu(zz.x), vv.y * scale * wt.y * ares_silu(zz.y),
                                vv.z * scale * wt.z * ares_silu(zz.z), vv.w * scale * wt.w * ares_silu(zz.w));
            } else {
                y = make_float4(vv.x * scale * wt.x * ares_sigmoid(zz.x), vv.y * scale * wt.y * ares_sigmoid(zz.y),
                                vv.z * scale * wt.z * ares_sigmoid(zz.z), vv.w * scale * wt.w * ares_sigmoid(zz.w));
            }
        }
        ares_quant_chunk(y, c < chunks, item * chunks + c, q, n_values);
    }
}

// ---------------------------------------------------------------------------
// SwiGLU with the quantization of its result for the down projection:
// y = silu(gate) * up over n_values (a multiple of 32), quantized into `q` in
// the layout of ares_quantize_act. One thread per 4 values.
ARES_KERNEL ares_swiglu_quant(const float* gate, const float* up, int8_t* q, uint32_t n_values) {
    const size_t chunk = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool live = chunk < (n_values >> 2);
    float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (live) {
        const float4 g = reinterpret_cast<const float4*>(gate)[chunk];
        const float4 u = reinterpret_cast<const float4*>(up)[chunk];
        y = make_float4(ares_silu(g.x) * u.x, ares_silu(g.y) * u.y, ares_silu(g.z) * u.z, ares_silu(g.w) * u.w);
    }
    ares_quant_chunk(y, live, chunk, q, n_values);
}
