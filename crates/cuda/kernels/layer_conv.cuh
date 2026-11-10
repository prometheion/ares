// The causal convolution of the recurrent block over the kept window of a
// sequence, with SiLU, the split into q / k / v and the normalisation of q and
// k per head; and the commit of the window.
//
// Per channel c of the projection `qkv` (conv_dim channels) the convolution has
// K taps w[c][0..K-1], tap K-1 on the newest input:
//   conv_t[c] = silu(sum_i w[c][i] * e_{t-(K-1)+i}[c])
// where e_j are the raw projections of the rows and, before the first row, the
// K-1 inputs the sequence's slot keeps (`conv_state`, [slot][channel][K-1],
// oldest first). The window kernel reads that state and never writes it.
//
// Output: the first key_dim channels (queries) go to `q_out` [rows][key_dim],
// the rest (keys, then values) to `kv_out` [rows][conv_dim - key_dim]; the
// channels of queries and keys are normalised per head of 128 channels when
// `norm_eps >= 0`: x / sqrt(sum(x^2) + eps) with the epsilon inside the root.
// `kv_out` is what the commit of the delta rule replays; `qkv` what the
// commit of the window replays.
//
// A warp handles 128 channels (a head) of a chunk of consecutive rows of one
// segment, 4 channels per lane. Rows are independent, so a chunk begins
// from the raw inputs before it (or from the state for the first K-1 rows of
// the segment); the chunks of a long prompt run in parallel.
#pragma once

#include "common.cuh"
#include "layer_math.cuh"
#include "layer_step.cuh"

#define ARES_CONV_MAX_TAPS 8

ARES_KERNEL ares_conv_window(const AresStep* step, const float* qkv, const float* taps, const float* conv_state,
                             uint64_t slot_stride, float* q_out, float* kv_out, uint32_t conv_dim,
                             uint32_t key_dim, uint32_t kernel, uint32_t chunk_rows, float norm_eps) {
    const AresSeg seg = ares_step_segs(step)[blockIdx.z];
    const uint32_t t0 = blockIdx.y * chunk_rows;
    if (t0 >= seg.n_rows) {
        return;
    }
    const uint32_t t1 = min(t0 + chunk_rows, seg.n_rows);
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t tile = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t c0 = tile * 128 + lane * 4;
    if (c0 >= conv_dim) {
        return;  // a whole warp: tiles are 128 channels and conv_dim is a multiple of 128
    }
    const uint32_t kk = kernel - 1;
    const bool normed = norm_eps >= 0.0f && c0 < 2 * key_dim;

    float w[4][ARES_CONV_MAX_TAPS];
#pragma unroll
    for (int ch = 0; ch < 4; ch++) {
#pragma unroll
        for (int i = 0; i < ARES_CONV_MAX_TAPS; i++) {
            w[ch][i] = i < kernel ? taps[static_cast<size_t>(c0 + ch) * kernel + i] : 0.0f;
        }
    }
    // win[j][ch] is the input j + 1 places before the first of the chunk's tap window.
    float win[ARES_CONV_MAX_TAPS - 1][4];
    const float* state = conv_state + static_cast<size_t>(seg.slot) * slot_stride;
#pragma unroll
    for (int j = 0; j < ARES_CONV_MAX_TAPS - 1; j++) {
#pragma unroll
        for (int ch = 0; ch < 4; ch++) {
            float value = 0.0f;
            if (j < kk) {
                // Input number (t0 - kk + j) of the segment; negative numbers are in the state.
                const int64_t at = static_cast<int64_t>(t0) - kk + j;
                value = at >= 0 ? qkv[(static_cast<size_t>(seg.row0) + at) * conv_dim + c0 + ch]
                                : state[static_cast<size_t>(c0 + ch) * kk + (at + kk)];
            }
            win[j][ch] = value;
        }
    }

#pragma unroll 4
    for (uint32_t t = t0; t < t1; t++) {
        const size_t row = static_cast<size_t>(seg.row0) + t;
        const float4 x4 = *reinterpret_cast<const float4*>(qkv + row * conv_dim + c0);
        float x[4] = {x4.x, x4.y, x4.z, x4.w};
        float y[4];
#pragma unroll
        for (int ch = 0; ch < 4; ch++) {
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < ARES_CONV_MAX_TAPS; i++) {
                const float input = i < kk ? win[i][ch] : (i == kk ? x[ch] : 0.0f);
                acc = fmaf(w[ch][i], input, acc);
            }
            y[ch] = ares_silu(acc);
            // Shift the window: the oldest input leaves, this one enters.
#pragma unroll
            for (int i = 0; i < ARES_CONV_MAX_TAPS - 1; i++) {
                if (i + 1 < kk) {
                    win[i][ch] = win[i + 1][ch];
                } else if (i + 1 == kk) {
                    win[i][ch] = x[ch];
                }
            }
        }
        if (normed) {
            float ssq = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
            ssq = ares_warp_sum(ssq);
            const float scale = rsqrtf(ssq + norm_eps);
            y[0] *= scale;
            y[1] *= scale;
            y[2] *= scale;
            y[3] *= scale;
        }
        const float4 out = make_float4(y[0], y[1], y[2], y[3]);
        if (c0 < key_dim) {
            *reinterpret_cast<float4*>(q_out + row * key_dim + c0) = out;
        } else {
            *reinterpret_cast<float4*>(kv_out + row * (conv_dim - key_dim) + (c0 - key_dim)) = out;
        }
    }
}

// The commit of the convolution window: after the first `m` rows of a segment
// the window is the last K-1 of (old window, then the first m raw inputs).
//   which = 0: the segments that are not direct, with m = accepted (the deferred commit)
//   which = 1: the direct segments, with m = n_rows (right after their window)
//   which = 2: both
// One thread per channel; the window is read completely before it is written,
// so the update is in place.
ARES_KERNEL ares_conv_commit(const AresStep* step, const float* qkv, float* conv_state, uint64_t slot_stride,
                             uint32_t conv_dim, uint32_t kernel, uint32_t which) {
    const AresSeg seg = ares_step_segs(step)[blockIdx.y];
    const bool direct = (seg.flags & ARES_SEG_DIRECT) != 0;
    if ((which == 0 && direct) || (which == 1 && !direct)) {
        return;
    }
    const uint32_t m = direct ? seg.n_rows : min(seg.accepted, seg.n_rows);
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (m == 0 || c >= conv_dim) {
        return;
    }
    const uint32_t kk = kernel - 1;
    float* state = conv_state + static_cast<size_t>(seg.slot) * slot_stride + static_cast<size_t>(c) * kk;
    float old[ARES_CONV_MAX_TAPS - 1];
#pragma unroll
    for (int i = 0; i < ARES_CONV_MAX_TAPS - 1; i++) {
        old[i] = i < kk ? state[i] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < ARES_CONV_MAX_TAPS - 1; i++) {
        if (i < kk) {
            const uint32_t at = m + i;
            float kept = 0.0f;
#pragma unroll
            for (int j = 0; j < ARES_CONV_MAX_TAPS - 1; j++) {
                if (j == at) {
                    kept = old[j];
                }
            }
            state[i] = at < kk ? kept : qkv[(static_cast<size_t>(seg.row0) + (at - kk)) * conv_dim + c];
        }
    }
}
