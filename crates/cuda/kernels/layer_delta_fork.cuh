// The design of the llama.cpp fork's gated delta rule kernel
// (ggml/src/ggml-cuda/gated_delta_net.cu, `gated_delta_net_cuda`, the
// non-KDA path), ported to this engine's inputs and step record so that the two
// designs can be timed on the same data. It is a baseline for comparison, not
// an engine kernel.
//
// The design: one block of 32 x 4 threads per (value head, sequence, 16
// columns of the matrix); a warp holds 4 columns of the matrix (one value index
// each), 8 lanes per column with 16 values of the column in registers (key
// indices r * 8 + lane); every row of the sequence reads k and q for its lane's
// indices straight from global memory, reduces over the 8 lanes with shuffles,
// and writes its output value; nothing is staged in shared memory and the state
// is written once at the end.
#pragma once

#include "common.cuh"
#include "layer_math.cuh"
#include "layer_step.cuh"

#define ARES_FORK_S 128
#define ARES_FORK_COLS_PER_WARP 4

extern "C" __global__ void __launch_bounds__(32 * 4, 2)
    ares_delta_window_fork_f32(const AresStep* step, const float* q, const float* kv, const float* alpha,
                               const float* beta_in, const float* dt_bias, const float* ssm_a, float* state,
                               uint64_t slot_stride, float* out, uint32_t n_k_heads, uint32_t n_v_heads,
                               uint32_t write_state) {
    constexpr int S_v = ARES_FORK_S;
    constexpr int lanes_per_col = 32 / ARES_FORK_COLS_PER_WARP;  // 8
    constexpr int rows_per_lane = S_v / lanes_per_col;           // 16
    const AresSeg seg = ares_step_segs(step)[blockIdx.y];
    const uint32_t h = blockIdx.x;
    const uint32_t kh = h % n_k_heads;
    const int lane = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * ARES_FORK_COLS_PER_WARP + col_in_warp;
    const uint32_t key_dim = n_k_heads * S_v, kv_stride = key_dim + n_v_heads * S_v, value_dim = n_v_heads * S_v;
    float* s_col = state + static_cast<size_t>(seg.slot) * slot_stride + (static_cast<size_t>(h) * S_v + col) * S_v;
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = s_col[r * lanes_per_col + lane_in_col];
    }
    const float scale = rsqrtf(static_cast<float>(S_v));
    const float dt = dt_bias[h], a = ssm_a[h];
    for (uint32_t t = 0; t < seg.n_rows; t++) {
        const size_t row = static_cast<size_t>(seg.row0) + t;
        const float* q_t = q + row * key_dim + kh * S_v;
        const float* k_t = kv + row * kv_stride + kh * S_v;
        const float* v_t = kv + row * kv_stride + key_dim + h * S_v;
        const float beta_val = ares_sigmoid(beta_in[row * n_v_heads + h]);
        const float g_val = expf(ares_softplus(alpha[row * n_v_heads + h] + dt) * a);
        float k_reg[rows_per_lane], q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
#pragma unroll
        for (int offset = lanes_per_col / 2; offset > 0; offset >>= 1) {
            kv_shard += __shfl_xor_sync(0xffffffffu, kv_shard, offset);
        }
        const float delta_col = (v_t[col] - g_val * kv_shard) * beta_val;
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
#pragma unroll
        for (int offset = lanes_per_col / 2; offset > 0; offset >>= 1) {
            attn_partial += __shfl_xor_sync(0xffffffffu, attn_partial, offset);
        }
        if (lane_in_col == 0) {
            out[row * value_dim + h * S_v + col] = attn_partial * scale;
        }
    }
    if (write_state) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_col[r * lanes_per_col + lane_in_col] = s_shard[r];
        }
    }
}
