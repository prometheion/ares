// The row kernels of the image tower (`qwen3vl_merger`, src/tower.rs): everything
// between its matrix products and its attention.
//
// The tower is a vision transformer: layer norms with a bias, biases on every
// product, the tanh GELU, a learned position table that is resized to the grid
// of the image, and a rotary rotation of query and key by the patch's row and
// column. Its residual reaches values in the thousands next to values near one,
// so the residual and the statistics of the norms are f32 throughout; what goes
// into a product is written in the f16 form of `ares_convert_f16` (mm.cuh), which
// divides each row by a power of two and keeps its precision whatever its size.
//
// Patches are in the order of the merger: the 2 x 2 groups row by row, inside a
// group (0,0), (0,1), (1,0), (1,1). Patch `p` of a grid of `nx` groups per line
// lies at row 2 (p / 4 / nx) + (p % 4) / 2 and column 2 (p / 4 % nx) + p % 2.
#pragma once

#include "common.cuh"

// The f16 activations of the tile kernels, as `ares_convert_f16` (mm.cuh) writes
// them: row `b` of `n_in` values at `b * ares_tower_act_pitch(n_in)` halves, each
// row divided by the power of two that brings its largest magnitude below 1, and
// behind the rows, at the next multiple of 16 bytes, the f32 power of two of every
// row. The same helpers are in mm.cuh and layer_act_f16.cuh, which this module does
// not include; tests/tower.rs compares what these kernels write with what
// `ares_convert_f16` writes, byte for byte.
static __device__ __forceinline__ size_t ares_tower_act_pitch(uint32_t n_in) {
    const size_t lines = (static_cast<size_t>(n_in) * 2 + 127) / 128 + 1;
    return (lines | 1) * 64;
}

static __device__ __forceinline__ float* ares_tower_act_factors(__half* out, uint32_t n_in, uint32_t n_rows) {
    const size_t bytes = static_cast<size_t>(n_rows) * ares_tower_act_pitch(n_in) * 2;
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(out) + ((bytes + 15) & ~static_cast<size_t>(15)));
}

// The power of two a row whose largest magnitude is `m` is divided by (`*down`
// is its inverse, `*up` the factor written behind the rows).
static __device__ __forceinline__ void ares_tower_act_scale(float m, float* down, float* up) {
    int e = 0;
    if (m > 0.0f && isfinite(m)) {
        frexpf(m, &e);
        e = max(-100, min(100, e));
    }
    *up = ldexpf(1.0f, e);
    *down = ldexpf(1.0f, -e);
}

// The maximum of `v` over the block, returned to every thread; `partial` holds a
// float per warp and must not be in use.
static __device__ __forceinline__ float ares_tower_block_max(float v, float* partial) {
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

// Where patch `p` lies in the grid of patches: (row, column).
static __device__ __forceinline__ void ares_tower_patch_at(uint32_t p, uint32_t nx, uint32_t* row, uint32_t* col) {
    const uint32_t group = p / 4;
    *row = 2 * (group / nx) + (p % 4) / 2;
    *col = 2 * (group % nx) + p % 2;
}

// Layer norm of rows written as f16 activations: `n_rows` rows of the output,
// each made of `group` consecutive input rows of `n` values side by side (the
// merger reads four patches as one row). Each input row is normalised on its own,
// `(x - mean) / sqrt(variance + eps) * weight + bias`, with its mean and variance
// in f32; the output row is then divided by the power of two of its largest value
// and written as f16 rows of `n * group` values, with the factors behind them.
// One block per output row; the row stays in shared memory (`n * group` floats,
// the launch's dynamic shared memory). `n` must be a multiple of 4 and `group`
// at most 8.
ARES_KERNEL ares_tower_norm_f16(const float* x, const float* weight, const float* bias, __half* out, uint32_t n,
                                uint32_t group, uint32_t n_rows, float eps) {
    extern __shared__ __align__(16) float row[];
    __shared__ float partial[32];
    __shared__ float mean_of[8], scale_of[8];
    const uint32_t width = n * group;
    const float4* src = reinterpret_cast<const float4*>(x + static_cast<size_t>(blockIdx.x) * width);
    float4* values = reinterpret_cast<float4*>(row);
    for (uint32_t c = threadIdx.x; c < width / 4; c += blockDim.x) {
        values[c] = src[c];
    }
    __syncthreads();
    for (uint32_t s = 0; s < group; s++) {
        const float* sub = row + s * n;
        float sum = 0.0f;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            sum += sub[i];
        }
        const float mean = ares_block_sum(sum, partial) / static_cast<float>(n);
        __syncthreads();
        float squares = 0.0f;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float d = sub[i] - mean;
            squares += d * d;
        }
        const float variance = ares_block_sum(squares, partial) / static_cast<float>(n);
        __syncthreads();
        if (threadIdx.x == 0) {
            mean_of[s] = mean;
            scale_of[s] = 1.0f / sqrtf(variance + eps);
        }
    }
    __syncthreads();
    float peak = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        const uint32_t s = i / n, k = i - s * n;
        const float v = (row[i] - mean_of[s]) * scale_of[s] * weight[k] + bias[k];
        row[i] = v;
        peak = fmaxf(peak, fabsf(v));
    }
    peak = ares_tower_block_max(peak, partial);
    float down, up;
    ares_tower_act_scale(peak, &down, &up);
    __half* o = out + static_cast<size_t>(blockIdx.x) * ares_tower_act_pitch(width);
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        o[i] = __float2half_rn(row[i] * down);
    }
    if (threadIdx.x == 0) {
        ares_tower_act_factors(out, width, n_rows)[blockIdx.x] = up;
    }
}

// 0.5 x (1 + tanh(sqrt(2 / pi) x (1 + 0.044715 x^2))), in f32.
static __device__ __forceinline__ float ares_tower_gelu(float x) {
    const float inner = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// The tanh GELU of `y + bias` written as f16 activations: rows of `n` values in,
// rows of `n_pad` values out, the values from `n` on zero (a product whose inputs
// are not a multiple of 32 reads its matrix padded to `n_pad` columns with zeros).
// One block per row, the row in shared memory (`n_pad` floats).
ARES_KERNEL ares_tower_gelu_f16(const float* y, const float* bias, __half* out, uint32_t n, uint32_t n_pad,
                                uint32_t n_rows) {
    extern __shared__ __align__(16) float row[];
    __shared__ float partial[32];
    const float* src = y + static_cast<size_t>(blockIdx.x) * n;
    float peak = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_pad; i += blockDim.x) {
        const float v = i < n ? ares_tower_gelu(src[i] + bias[i]) : 0.0f;
        row[i] = v;
        peak = fmaxf(peak, fabsf(v));
    }
    peak = ares_tower_block_max(peak, partial);
    float down, up;
    ares_tower_act_scale(peak, &down, &up);
    __half* o = out + static_cast<size_t>(blockIdx.x) * ares_tower_act_pitch(n_pad);
    for (uint32_t i = threadIdx.x; i < n_pad; i += blockDim.x) {
        o[i] = __float2half_rn(row[i] * down);
    }
    if (threadIdx.x == 0) {
        ares_tower_act_factors(out, n_pad, n_rows)[blockIdx.x] = up;
    }
}

// The residual: x += y + bias, rows of `n` values (`n` a multiple of 4), the bias
// added to the product before the product is added to the residual. One thread
// per four values.
ARES_KERNEL ares_tower_residual(float* x, const float* y, const float* bias, uint32_t n, uint32_t n_values) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_values / 4) {
        return;
    }
    const float4 b = reinterpret_cast<const float4*>(bias)[c % (n / 4)];
    const float4 v = reinterpret_cast<const float4*>(y)[c];
    float4 r = reinterpret_cast<float4*>(x)[c];
    r.x = __fadd_rn(r.x, __fadd_rn(v.x, b.x));
    r.y = __fadd_rn(r.y, __fadd_rn(v.y, b.y));
    r.z = __fadd_rn(r.z, __fadd_rn(v.z, b.z));
    r.w = __fadd_rn(r.w, __fadd_rn(v.w, b.w));
    reinterpret_cast<float4*>(x)[c] = r;
}

// Columns of a result that was computed as part of a wider one, put in their
// place with their bias: dst[r][at + i] = src[r][i] + bias[i] for rows of `width`
// values in `src` and of `dst_width` in `dst` (`bias` is the part's, and may be
// null). One thread per value.
ARES_KERNEL ares_tower_place(const float* src, float* dst, const float* bias, uint32_t n_rows, uint32_t width,
                             uint32_t dst_width, uint32_t at) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(n_rows) * width) {
        return;
    }
    const size_t r = i / width;
    const uint32_t k = static_cast<uint32_t>(i - r * width);
    const float v = src[i];
    dst[r * dst_width + at + k] = bias != nullptr ? __fadd_rn(v, bias[k]) : v;
}

// The two cells of the position table around output cell `i` of a side of `out`
// cells, and the weight of the second: ggml's bilinear resize with aligned
// corners, in f32 as it computes it (`sf` is the host's (out - 1) / (side - 1)).
static __device__ __forceinline__ void ares_tower_around(uint32_t i, float sf, uint32_t side, uint32_t* lo,
                                                         uint32_t* hi, float* w) {
    const float pos = __fdiv_rn(static_cast<float>(i), sf);
    const int f = static_cast<int>(floorf(pos));
    const int last = static_cast<int>(side) - 1;
    const int a = min(max(f, 0), last), b = min(max(f + 1, 0), last);
    *lo = static_cast<uint32_t>(a);
    *hi = static_cast<uint32_t>(b);
    *w = fminf(fmaxf(__fsub_rn(pos, static_cast<float>(a)), 0.0f), 1.0f);
}

// The patches' bias and learned positions: x = (x + patch_bias) + position, where
// x holds the patch product of `n_patches` patches of `n` values and the position
// of a patch is the `side` x `side` table resized bilinearly to the `pw` x `ph`
// grid (or the table's own cell when the grid is the table's size). Every
// operation is the reference's, in its order and without contraction, so the
// positions are its bits. One block per patch.
ARES_KERNEL ares_tower_positions(float* x, const float* patch_bias, const float* table, uint32_t n, uint32_t side,
                                 uint32_t pw, uint32_t ph, float sfx, float sfy) {
    const uint32_t p = blockIdx.x;
    uint32_t py, px;
    ares_tower_patch_at(p, pw / 2, &py, &px);
    float* row = x + static_cast<size_t>(p) * n;
    if (pw == side && ph == side) {
        const float* cell = table + (static_cast<size_t>(py) * side + px) * n;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            row[i] = __fadd_rn(__fadd_rn(row[i], patch_bias[i]), cell[i]);
        }
        return;
    }
    uint32_t y0, y1, x0, x1;
    float dy, dx;
    ares_tower_around(py, sfy, side, &y0, &y1, &dy);
    ares_tower_around(px, sfx, side, &x0, &x1, &dx);
    const float* a = table + (static_cast<size_t>(y0) * side + x0) * n;
    const float* b = table + (static_cast<size_t>(y0) * side + x1) * n;
    const float* c = table + (static_cast<size_t>(y1) * side + x0) * n;
    const float* d = table + (static_cast<size_t>(y1) * side + x1) * n;
    const float ex = __fsub_rn(1.0f, dx), ey = __fsub_rn(1.0f, dy);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float pos = __fmul_rn(__fmul_rn(a[i], ex), ey);
        pos = __fadd_rn(pos, __fmul_rn(__fmul_rn(b[i], dx), ey));
        pos = __fadd_rn(pos, __fmul_rn(__fmul_rn(c[i], ex), dy));
        pos = __fadd_rn(pos, __fmul_rn(__fmul_rn(d[i], dx), dy));
        row[i] = __fadd_rn(__fadd_rn(row[i], patch_bias[i]), pos);
    }
}

// Query, key and value of `n_rows` patches from the product of their norm
// (`qkv`: rows of three times `n_head * head_dim` values, query | key | value) and
// its bias, as the attention reads them: f16 rows of `pitch` values per head and
// patch, `heads[part][head][patch]` for the parts query, key, value, with the
// values from `head_dim` on zero. Query and key are rotated: a head has
// `head_dim / 2` pairs (k, k + head_dim / 2); the first half of the pairs turns by
// the patch's row, the second by its column, pair `j` of a half by
// `position * step^j`, the angle built as the reference builds it, by multiplying
// on in f32 (`step` is the host's 10000^(-4 / head_dim)). The patches are
// `p0 .. p0 + n_rows` of `n_patches`, in a grid of `nx` groups per line. One
// block per patch.
ARES_KERNEL ares_tower_rotate_heads(const float* qkv, const float* bias, __half* heads, uint32_t n_head,
                                    uint32_t head_dim, uint32_t pitch, uint32_t n_patches, uint32_t p0, uint32_t nx,
                                    float step) {
    __shared__ float2 turn[128];
    const uint32_t pairs = head_dim / 2, half = pairs / 2;
    const uint32_t p = p0 + blockIdx.x;
    uint32_t py, px;
    ares_tower_patch_at(p, nx, &py, &px);
    if (threadIdx.x < pairs) {
        const uint32_t j = threadIdx.x;
        float theta = static_cast<float>(j < half ? py : px);
        for (uint32_t k = 0; k < (j < half ? j : j - half); k++) {
            theta = __fmul_rn(theta, step);
        }
        turn[j] = make_float2(cosf(theta), sinf(theta));
    }
    __syncthreads();
    const uint32_t width = n_head * head_dim;
    const float* row = qkv + static_cast<size_t>(blockIdx.x) * 3 * width;
    const size_t part_stride = static_cast<size_t>(n_head) * n_patches * pitch;
    for (uint32_t i = threadIdx.x; i < 3 * n_head * pitch; i += blockDim.x) {
        const uint32_t part = i / (n_head * pitch);
        const uint32_t rest = i - part * n_head * pitch;
        const uint32_t head = rest / pitch, k = rest - head * pitch;
        float v = 0.0f;
        if (k < head_dim) {
            const uint32_t at = part * width + head * head_dim;
            v = __fadd_rn(row[at + k], bias[at + k]);
            if (part < 2) {
                const uint32_t j = k < pairs ? k : k - pairs;
                const uint32_t other = k < pairs ? k + pairs : k - pairs;
                const float o = __fadd_rn(row[at + other], bias[at + other]);
                const float2 cs = turn[j];
                // lo' = lo cos - hi sin, hi' = hi cos + lo sin.
                v = k < pairs ? fmaf(v, cs.x, -o * cs.y) : fmaf(v, cs.x, o * cs.y);
            }
        }
        heads[part * part_stride + (static_cast<size_t>(head) * n_patches + p) * pitch + k] = __float2half_rn(v);
    }
}
