// The fused layer kernels that write f16 activations: the gated norm of the
// recurrent block and the SwiGLU of the feed-forward part for a product that
// reads f16 (a matrix in its device layout, or the f16 tile kernels of the
// file's layout), and what the merges of a tensor split need to write the same
// form (layer_merge.cuh).
//
// The form is the one of `ares_convert_f16` (mm.cuh), byte for byte (the tests
// compare the bytes): row `b` of `n_in` values as f16 at
// `b * ares_f16_act_pitch(n_in)` values, each row divided by the power of two
// that brings its largest magnitude below 1, and behind the rows, at the next
// multiple of 16 bytes, the f32 power of two of every row. A fused kernel
// writes it in the launch that computes the values, where a separate
// conversion is one more launch per product input: 3 to 5 microseconds of a
// decode step each on an RTX 4070, 128 times per step of Qwen3.8-27B.
//
// The row's power of two needs the largest magnitude of the whole row before
// any value is written, so these kernels run one block per row, where the
// kernels that quantize to int8 (a scale per 32 values) take any grid.
#pragma once

#include "common.cuh"
#include "layer_math.cuh"
#include "quantize_act.cuh"

// The f16 values from one row of f16 activations to the next. This is
// `ares_act_f16_pitch` of mm.cuh, which the layer kernels' binary does not
// include; the two must stay equal (tests/layers.rs compares what the fused
// kernels write with what `ares_convert_f16` writes).
static __host__ __device__ __forceinline__ size_t ares_f16_act_pitch(uint32_t n_in) {
    const size_t lines = (static_cast<size_t>(n_in) * 2 + 127) / 128 + 1;
    return (lines | 1) * 64;
}

// Where the f32 factors of the rows are, behind `n_rows` rows of `n_in` values.
static __device__ __forceinline__ float* ares_f16_act_factors(__half* out, uint32_t n_in, uint32_t n_rows) {
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(out) +
                                    ares_act_ds_offset(static_cast<size_t>(n_rows) * ares_f16_act_pitch(n_in) * 2));
}

// The power of two a row whose largest magnitude is `m` is divided by (`*up`)
// and its inverse (`*down`): the arithmetic of ares_convert_f16.
static __device__ __forceinline__ void ares_f16_act_scale(float m, float* down, float* up) {
    int e = 0;
    if (m > 0.0f && isfinite(m)) {
        frexpf(m, &e);
        e = max(-100, min(100, e));
    }
    *up = ldexpf(1.0f, e);
    *down = ldexpf(1.0f, -e);
}

// Four consecutive values of a row, times `down`, as f16 at float4 `chunk` of
// the row that starts at `row`.
static __device__ __forceinline__ void ares_f16_act_store(__half* row, uint32_t chunk, float4 v, float down) {
    const uint64_t a = __half_as_ushort(__float2half_rn(v.x * down));
    const uint64_t b = __half_as_ushort(__float2half_rn(v.y * down));
    const uint64_t c = __half_as_ushort(__float2half_rn(v.z * down));
    const uint64_t e = __half_as_ushort(__float2half_rn(v.w * down));
    reinterpret_cast<uint64_t*>(row)[chunk] = a | (b << 16) | (c << 32) | (e << 48);
}

static __device__ __forceinline__ float ares_f16_act_max4(float4 v) {
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}

// The maximum of `v` over all threads of the block, returned to every thread.
// `partial` as for ares_block_sum.
static __device__ __forceinline__ float ares_f16_act_block_max(float v, float* partial) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    const uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        partial[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    const uint32_t n_warps = blockDim.x >> 5;
    float m = lane < n_warps ? partial[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    }
    return m;
}

// The gated norm of one (row, head) item of `w` values, for the lane's one or
// two float4 chunks: y = o / sqrt(mean(o^2) + eps) * weight * gate(z). The
// arithmetic of ares_gated_norm_quant. Every lane of the warp must call it.
static __device__ __forceinline__ void ares_gated_norm_item(const float4* orow, const float4* zrow,
                                                            const float* weight, uint32_t chunks, uint32_t w,
                                                            float eps, uint32_t gate_kind, float4* y) {
    const uint32_t lane = threadIdx.x & 31;
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
        y[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (c < chunks) {
            const float4 zz = zrow[c];
            const float4 wt = reinterpret_cast<const float4*>(weight)[c];
            const float4 vv = v[k];
            if (gate_kind == 0) {
                y[k] = make_float4(vv.x * scale * wt.x * ares_silu(zz.x), vv.y * scale * wt.y * ares_silu(zz.y),
                                   vv.z * scale * wt.z * ares_silu(zz.z), vv.w * scale * wt.w * ares_silu(zz.w));
            } else {
                y[k] =
                    make_float4(vv.x * scale * wt.x * ares_sigmoid(zz.x), vv.y * scale * wt.y * ares_sigmoid(zz.y),
                                vv.z * scale * wt.z * ares_sigmoid(zz.z), vv.w * scale * wt.w * ares_sigmoid(zz.w));
            }
        }
    }
}

// Heads of a row a warp of ares_gated_norm_f16 keeps in registers. A block of
// 1024 threads leaves a thread 64 registers, which two heads of two float4
// chunks and the arithmetic fit.
#define ARES_GATED_NORM_F16_HEADS 2

// ---------------------------------------------------------------------------
// The gated norm of the recurrent block as f16 activations for the output
// projection: for each of `n_rows` rows its `n_heads` heads of `w` values (w a
// multiple of 32, at most 256),
//   y = o / sqrt(mean(o^2) + eps) * weight * gate(z),  gate = silu (gate_kind 0) or sigmoid (1)
// written as `n_rows` rows of `n_heads * w` f16 values. One block per row; a
// warp takes the heads `warp`, `warp + warps`, ... and keeps them in registers
// between the row's largest magnitude and the write, so every value is loaded
// once: `n_heads <= ARES_GATED_NORM_F16_HEADS * blockDim.x / 32`.
extern "C" __global__ void __launch_bounds__(1024) ares_gated_norm_f16(const float* o, const float* z,
                                                                       const float* weight, __half* out,
                                                                       uint32_t n_rows, uint32_t n_heads, uint32_t w,
                                                                       float eps, uint32_t gate_kind) {
    __shared__ float partial[32];
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_warps = blockDim.x >> 5;
    const uint32_t row = blockIdx.x;
    const uint32_t chunks = w >> 2;
    const uint32_t n_in = n_heads * w;
    const float4* orow = reinterpret_cast<const float4*>(o + static_cast<size_t>(row) * n_in);
    const float4* zrow = reinterpret_cast<const float4*>(z + static_cast<size_t>(row) * n_in);

    float4 y[ARES_GATED_NORM_F16_HEADS][2];
    float m = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_GATED_NORM_F16_HEADS; k++) {
        const uint32_t h = warp + k * n_warps;
        y[k][0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        y[k][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        // The same for every lane of the warp, which the item's sum needs.
        if (h < n_heads) {
            ares_gated_norm_item(orow + static_cast<size_t>(h) * chunks, zrow + static_cast<size_t>(h) * chunks,
                                 weight, chunks, w, eps, gate_kind, y[k]);
            m = fmaxf(m, fmaxf(ares_f16_act_max4(y[k][0]), ares_f16_act_max4(y[k][1])));
        }
    }
    m = ares_f16_act_block_max(m, partial);
    float down, up;
    ares_f16_act_scale(m, &down, &up);

    __half* dst = out + static_cast<size_t>(row) * ares_f16_act_pitch(n_in);
#pragma unroll
    for (int k = 0; k < ARES_GATED_NORM_F16_HEADS; k++) {
        const uint32_t h = warp + k * n_warps;
        if (h < n_heads) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const uint32_t c = lane + 32 * j;
                if (c < chunks) {
                    ares_f16_act_store(dst, h * chunks + c, y[k][j], down);
                }
            }
        }
    }
    if (threadIdx.x == 0) {
        ares_f16_act_factors(out, n_in, n_rows)[row] = up;
    }
}

// float4 chunks a thread of ares_swiglu_f16 keeps in registers.
#define ARES_SWIGLU_F16_RESIDENT 6

// ---------------------------------------------------------------------------
// SwiGLU as f16 activations for the down projection: y = silu(gate) * up for
// `n_rows` rows of `n_in` values (a multiple of 4), written as rows of f16. One
// block per row, which keeps the row in registers between its largest
// magnitude and the write: `n_in <= 16 * ARES_SWIGLU_F16_RESIDENT * blockDim.x / 4`.
// A thread waits for each of its loads, so the host gives a row as many
// threads as a block has before it gives a thread a second chunk.
extern "C" __global__ void __launch_bounds__(1024) ares_swiglu_f16(const float* gate, const float* up_in,
                                                                   __half* out, uint32_t n_rows, uint32_t n_in) {
    __shared__ float partial[32];
    const uint32_t row = blockIdx.x;
    const uint32_t chunks = n_in >> 2;
    const float4* g = reinterpret_cast<const float4*>(gate + static_cast<size_t>(row) * n_in);
    const float4* u = reinterpret_cast<const float4*>(up_in + static_cast<size_t>(row) * n_in);

    float4 y[ARES_SWIGLU_F16_RESIDENT];
    float m = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_SWIGLU_F16_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        y[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (c < chunks) {
            const float4 gg = g[c];
            const float4 uu = u[c];
            y[k] = make_float4(ares_silu(gg.x) * uu.x, ares_silu(gg.y) * uu.y, ares_silu(gg.z) * uu.z,
                               ares_silu(gg.w) * uu.w);
        }
        m = fmaxf(m, ares_f16_act_max4(y[k]));
    }
    m = ares_f16_act_block_max(m, partial);
    float down, up;
    ares_f16_act_scale(m, &down, &up);

    __half* dst = out + static_cast<size_t>(row) * ares_f16_act_pitch(n_in);
#pragma unroll
    for (int k = 0; k < ARES_SWIGLU_F16_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        if (c < chunks) {
            ares_f16_act_store(dst, c, y[k], down);
        }
    }
    if (threadIdx.x == 0) {
        ares_f16_act_factors(out, n_in, n_rows)[row] = up;
    }
}
