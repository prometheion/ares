// The gated delta rule of the recurrent block over the rows of segments, and
// its commit.
//
// For a value head h with key head j = h mod n_k_heads (tiled), the matrix
// S_h of d x d values stored as [m][i] (row m = value index, i = key index,
// contiguous), and for each row t of a segment:
//   decay = exp(g),  g = softplus(alpha + dt_bias[h]) * a[h]     (g <= 0)
//   beta  = sigmoid(beta_in)
//   sk[m] = decay * sum_i S[m][i] k[i]
//   d[m]  = beta * (v[m] - sk[m])
//   S[m][i] = decay * S[m][i] + k[i] d[m]
//   o[m]  = sum_i S[m][i] q[i] / sqrt(d)
// The rows m of S are independent of each other: a thread group owns one row
// (LPR lanes, each holding d / LPR values of it in registers) and walks the
// rows of the segment in order; nothing is exchanged between groups except the
// inputs of the row (k, q, v, g, beta), which a block stages in shared memory
// TC rows at a time. A block owns RB rows of S of one head of one segment:
// the grid is (n_v_heads * d / RB, n_segs). Segments of different lengths
// share a launch; a block of a short segment is done early.
//
// Window: reads the committed state of the segment's slot, writes the outputs,
// and writes the state only for direct segments (all rows certain to be
// accepted). Commit: replays the first `accepted` rows into the state without
// outputs, for the segments that are not direct.
//
// The state is f32 or f16 in memory (ST); the arithmetic is f32 either way.
// With f16 in memory the matrix is rounded to f16 after every row, in the
// registers too: a step of one row stores the rounded matrix and the next
// step loads it, and the rows of a window, a prompt or a commit must go on
// from the same values, or a row would depend on the step it was computed in.
#pragma once

#include "common.cuh"
#include "layer_math.cuh"
#include "layer_step.cuh"

static __device__ __forceinline__ float4 ares_delta_load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
static __device__ __forceinline__ float4 ares_delta_load4(const __half* p) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
static __device__ __forceinline__ void ares_delta_store4(float* p, float4 v) {
    *reinterpret_cast<float4*>(p) = v;
}
static __device__ __forceinline__ void ares_delta_store4(__half* p, float4 v) {
    const __half2 a = __floats2half2_rn(v.x, v.y);
    const __half2 b = __floats2half2_rn(v.z, v.w);
    uint2 raw;
    raw.x = *reinterpret_cast<const uint32_t*>(&a);
    raw.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = raw;
}

// The values as the state holds them between two rows: themselves for f32,
// rounded to f16 for f16 (what a store and a load would give).
static __device__ __forceinline__ float4 ares_delta_held(const float*, float4 v) {
    return v;
}
static __device__ __forceinline__ float4 ares_delta_held(const __half*, float4 v) {
    const float2 a = __half22float2(__floats2half2_rn(v.x, v.y));
    const float2 b = __half22float2(__floats2half2_rn(v.z, v.w));
    return make_float4(a.x, a.y, b.x, b.y);
}

// DK: head width d (k and v heads have the same width), LPR: lanes per row of S,
// RB: rows of S per block, TC: rows of the segment staged at a time.
// WINDOW: true for the window kernel (outputs, state written only if direct),
// false for the commit (no outputs, deferred segments only).
template <int DK, int LPR, int RB, int TC, typename ST, bool WINDOW>
static __device__ __forceinline__ void ares_delta_body(const AresStep* step, const float* q, const float* kv,
                                                       const float* alpha, const float* beta_in,
                                                       const float* dt_bias, const float* ssm_a, ST* state,
                                                       uint64_t slot_stride, float* out, uint32_t n_k_heads,
                                                       uint32_t n_v_heads) {
    constexpr int THREADS = RB * LPR;
    constexpr int CHUNKS = DK / 4;          // float4 chunks per row of S
    constexpr int PER_LANE = CHUNKS / LPR;  // chunks held by one lane
    static_assert(CHUNKS % LPR == 0, "a row of S must split evenly over its lanes");
    static_assert(32 % LPR == 0, "the lanes of a row must lie in one warp");

    __shared__ __align__(16) float ks[TC][DK];
    __shared__ __align__(16) float qs[WINDOW ? TC : 1][DK];
    __shared__ float vs[TC][RB];
    __shared__ float os[WINDOW ? TC : 1][RB];
    __shared__ float decays[TC], betas[TC];

    const AresSeg seg = ares_step_segs(step)[blockIdx.y];
    const bool direct = (seg.flags & ARES_SEG_DIRECT) != 0;
    uint32_t n_rows;
    if (WINDOW) {
        n_rows = seg.n_rows;
    } else {
        if (direct) {
            return;
        }
        n_rows = min(seg.accepted, seg.n_rows);
    }
    if (n_rows == 0) {
        return;
    }
    const bool write_state = !WINDOW || direct;

    const uint32_t slices = DK / RB;
    const uint32_t h = blockIdx.x / slices;
    const uint32_t slice = blockIdx.x % slices;
    const uint32_t kh = h % n_k_heads;
    const uint32_t key_dim = n_k_heads * DK;
    const uint32_t kv_stride = key_dim + n_v_heads * DK;
    const uint32_t value_dim = n_v_heads * DK;
    const uint32_t tid = threadIdx.x;
    const uint32_t r = tid / LPR;   // the row of S inside the slice
    const uint32_t sub = tid % LPR;  // the lane inside the row

    ST* row_state = state + static_cast<size_t>(seg.slot) * slot_stride +
                    (static_cast<size_t>(h) * DK + static_cast<size_t>(slice) * RB + r) * DK;
    float4 s[PER_LANE];
#pragma unroll
    for (int j = 0; j < PER_LANE; j++) {
        s[j] = ares_delta_load4(row_state + 4 * (sub + LPR * j));
    }

    const float scale = rsqrtf(static_cast<float>(DK));
    const float dt = dt_bias[h], a = ssm_a[h];

    for (uint32_t t0 = 0; t0 < n_rows; t0 += TC) {
        const uint32_t tc = min(static_cast<uint32_t>(TC), n_rows - t0);
        // Stage the inputs of rows t0 .. t0 + tc - 1.
        for (uint32_t idx = tid; idx < tc * CHUNKS; idx += THREADS) {
            const uint32_t t = idx / CHUNKS, c = idx % CHUNKS;
            const size_t row = static_cast<size_t>(seg.row0) + t0 + t;
            reinterpret_cast<float4*>(ks[t])[c] =
                *reinterpret_cast<const float4*>(kv + row * kv_stride + kh * DK + 4 * c);
            if (WINDOW) {
                reinterpret_cast<float4*>(qs[t])[c] =
                    *reinterpret_cast<const float4*>(q + row * key_dim + kh * DK + 4 * c);
            }
        }
        for (uint32_t idx = tid; idx < tc * RB; idx += THREADS) {
            const uint32_t t = idx / RB, m = idx % RB;
            const size_t row = static_cast<size_t>(seg.row0) + t0 + t;
            vs[t][m] = kv[row * kv_stride + key_dim + h * DK + slice * RB + m];
        }
        if (tid < tc) {
            const size_t row = static_cast<size_t>(seg.row0) + t0 + tid;
            decays[tid] = expf(ares_softplus(alpha[row * n_v_heads + h] + dt) * a);
            betas[tid] = ares_sigmoid(beta_in[row * n_v_heads + h]);
        }
        __syncthreads();

        for (uint32_t t = 0; t < tc; t++) {
            const float decay = decays[t];
            const float4* kt = reinterpret_cast<const float4*>(ks[t]);
            float sk0 = 0.0f, sk1 = 0.0f;
#pragma unroll
            for (int j = 0; j < PER_LANE; j++) {
                const float4 k4 = kt[sub + LPR * j];
                sk0 = fmaf(s[j].x, k4.x, sk0);
                sk1 = fmaf(s[j].y, k4.y, sk1);
                sk0 = fmaf(s[j].z, k4.z, sk0);
                sk1 = fmaf(s[j].w, k4.w, sk1);
            }
            float sk = sk0 + sk1;
#pragma unroll
            for (int offset = LPR / 2; offset > 0; offset >>= 1) {
                sk += __shfl_xor_sync(0xffffffffu, sk, offset);
            }
            const float d = betas[t] * (vs[t][r] - decay * sk);
            if (WINDOW) {
                const float4* qt = reinterpret_cast<const float4*>(qs[t]);
                float o0 = 0.0f, o1 = 0.0f;
#pragma unroll
                for (int j = 0; j < PER_LANE; j++) {
                    const float4 k4 = kt[sub + LPR * j];
                    const float4 q4 = qt[sub + LPR * j];
                    s[j].x = fmaf(decay, s[j].x, k4.x * d);
                    s[j].y = fmaf(decay, s[j].y, k4.y * d);
                    s[j].z = fmaf(decay, s[j].z, k4.z * d);
                    s[j].w = fmaf(decay, s[j].w, k4.w * d);
                    o0 = fmaf(s[j].x, q4.x, o0);
                    o1 = fmaf(s[j].y, q4.y, o1);
                    o0 = fmaf(s[j].z, q4.z, o0);
                    o1 = fmaf(s[j].w, q4.w, o1);
                }
                float o = o0 + o1;
#pragma unroll
                for (int offset = LPR / 2; offset > 0; offset >>= 1) {
                    o += __shfl_xor_sync(0xffffffffu, o, offset);
                }
                if (sub == 0) {
                    os[t][r] = o * scale;
                }
            } else {
#pragma unroll
                for (int j = 0; j < PER_LANE; j++) {
                    const float4 k4 = kt[sub + LPR * j];
                    s[j].x = fmaf(decay, s[j].x, k4.x * d);
                    s[j].y = fmaf(decay, s[j].y, k4.y * d);
                    s[j].z = fmaf(decay, s[j].z, k4.z * d);
                    s[j].w = fmaf(decay, s[j].w, k4.w * d);
                }
            }
#pragma unroll
            for (int j = 0; j < PER_LANE; j++) {
                s[j] = ares_delta_held(row_state, s[j]);
            }
        }
        if (WINDOW) {
            __syncthreads();
            for (uint32_t idx = tid; idx < tc * RB; idx += THREADS) {
                const uint32_t t = idx / RB, m = idx % RB;
                out[(static_cast<size_t>(seg.row0) + t0 + t) * value_dim + h * DK + slice * RB + m] = os[t][m];
            }
        }
        __syncthreads();
    }

    if (write_state) {
#pragma unroll
        for (int j = 0; j < PER_LANE; j++) {
            ares_delta_store4(row_state + 4 * (sub + LPR * j), s[j]);
        }
    }
}

// One extern "C" kernel per (state type, window or commit); the shape template
// arguments are chosen here, the sizes of the model are arguments.
#define ARES_DELTA_KERNELS(SUFFIX, DK, LPR, RB, TC)                                                            \
    ARES_KERNEL ares_delta_window_f32_##SUFFIX(const AresStep* step, const float* q, const float* kv,        \
                                               const float* alpha, const float* beta_in, const float* dt_bias, \
                                               const float* ssm_a, float* state, uint64_t slot_stride,       \
                                               float* out, uint32_t n_k_heads, uint32_t n_v_heads) {          \
        ares_delta_body<DK, LPR, RB, TC, float, true>(step, q, kv, alpha, beta_in, dt_bias, ssm_a, state,   \
                                                      slot_stride, out, n_k_heads, n_v_heads);               \
    }                                                                                                        \
    ARES_KERNEL ares_delta_window_f16_##SUFFIX(const AresStep* step, const float* q, const float* kv,        \
                                               const float* alpha, const float* beta_in, const float* dt_bias, \
                                               const float* ssm_a, __half* state, uint64_t slot_stride,      \
                                               float* out, uint32_t n_k_heads, uint32_t n_v_heads) {          \
        ares_delta_body<DK, LPR, RB, TC, __half, true>(step, q, kv, alpha, beta_in, dt_bias, ssm_a, state,  \
                                                       slot_stride, out, n_k_heads, n_v_heads);              \
    }                                                                                                        \
    ARES_KERNEL ares_delta_commit_f32_##SUFFIX(const AresStep* step, const float* kv, const float* alpha,    \
                                               const float* beta_in, const float* dt_bias,                   \
                                               const float* ssm_a, float* state, uint64_t slot_stride,       \
                                               uint32_t n_k_heads, uint32_t n_v_heads) {                      \
        ares_delta_body<DK, LPR, RB, TC, float, false>(step, nullptr, kv, alpha, beta_in, dt_bias, ssm_a,   \
                                                       state, slot_stride, nullptr, n_k_heads, n_v_heads);   \
    }                                                                                                        \
    ARES_KERNEL ares_delta_commit_f16_##SUFFIX(const AresStep* step, const float* kv, const float* alpha,    \
                                               const float* beta_in, const float* dt_bias,                   \
                                               const float* ssm_a, __half* state, uint64_t slot_stride,      \
                                               uint32_t n_k_heads, uint32_t n_v_heads) {                      \
        ares_delta_body<DK, LPR, RB, TC, __half, false>(step, nullptr, kv, alpha, beta_in, dt_bias, ssm_a,  \
                                                        state, slot_stride, nullptr, n_k_heads, n_v_heads);  \
    }
