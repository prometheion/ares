// The device part of the hashed n-gram embedding of qwen4exp (section 3.4 of
// docs/reference/qwen4exp-architecture.md). The host hashes the tokens and
// reads the table rows from the file; the device gets, per row of the step,
// the stored bytes of the rows of all heads and does the rest:
//
//   emb     = the fetched rows decoded, side by side                  ares_q4_ngram_decode
//   key     = ple_key @ emb,  value = ple_value @ emb                 (weight kernels)
//   g[c]    = sigmoid(sign(s) sqrt(max(|s|, 1e-6))),
//             s = (rms(key[c]) nk[c] . rms(R[c]) nq[c]) / sqrt(N)
//   gated[c] = value * g[c],  nrm[c] = rms(gated[c]) * nc[c]          ares_q4_ngram_gate
//   conv    = dilated causal convolution over the history and the rows' nrm
//   R[c]   += gated[c] + silu(conv[c])                                ares_q4_ngram_window
//   history <- the nrm of the accepted rows                           ares_q4_ngram_commit
//
// The history of a sequence slot is [back][D] f32, oldest first, with
// back = (kernel - 1) * dilation: the nrm vectors of its newest cells, zero
// before the start of the sequence. The window reads it and never writes it.
#pragma once

#include "common.cuh"
#include "layer_math.cuh"
#include "layer_step.cuh"

#define Q4_NGRAM_MAX_TAPS 8
#define Q4_NGRAM_MAX_BACK 32

static __device__ const int8_t q4_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                      1,    13,   25,  38,  53,  69,  89,  113};

// Decodes the fetched table rows: IQ4_NL blocks of 18 bytes (an f16 scale,
// then 16 bytes where byte j holds value j in its low nibble and value j + 16
// in its high one). `fetched` holds, per row of the step, the rows of all
// heads one after the other, so block number b of the buffer is values
// 32 b .. 32 b + 31 of `emb`. One warp per block.
ARES_KERNEL ares_q4_ngram_decode(const uint8_t* fetched, float* emb, uint32_t n_blocks) {
    const size_t block = static_cast<size_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (block >= n_blocks) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31;
    const uint8_t* b = fetched + block * 18;
    const float d = ares_half_at(b);
    const uint32_t byte = b[2 + (lane & 15)];
    const uint32_t nibble = lane < 16 ? (byte & 15u) : (byte >> 4);
    emb[block * 32 + lane] = d * static_cast<float>(q4_iq4nl_values[nibble]);
}

// The sum of `v` over all threads of the block in f64. `partial` has one
// double per warp.
static __device__ __forceinline__ double q4_block_sum_f64(double v, double* partial) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    const uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        partial[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    double sum = 0.0;
    const uint32_t warps = blockDim.x >> 5;
    for (uint32_t w = 0; w < warps; w++) {
        sum += partial[w];
    }
    return sum;
}

// The gate, the gated value and its norm, one thread block per (row, stream).
//   key    [row][hc][n]   value [row][n]   r [row][hc][n]   (the residual, read only)
//   norm_key, norm_query, norm_conv  [hc][n]
//   gated, nrm  [row][hc][n];  gate_out [row][hc] or null
// The four sums of a stream are taken in f64: the gate is the sigmoid of a
// square root of a dot product that can be close to zero, where an f32 sum
// would be visible in the result.
ARES_KERNEL ares_q4_ngram_gate(const float* key, const float* value, const float* r, const float* norm_key,
                               const float* norm_query, const float* norm_conv, float* gated, float* nrm,
                               float* gate_out, uint32_t rows, uint32_t hc, uint32_t n, float eps) {
    __shared__ double partial[4][32];
    const uint32_t row = blockIdx.x / hc, c = blockIdx.x % hc;
    if (row >= rows) {
        return;
    }
    const size_t at = (static_cast<size_t>(row) * hc + c) * n;
    const size_t wat = static_cast<size_t>(c) * n;
    const float* v = value + static_cast<size_t>(row) * n;
    double sk = 0.0, sq = 0.0, sd = 0.0, sv = 0.0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float k = key[at + i], q = r[at + i], x = v[i];
        sk += static_cast<double>(k) * k;
        sq += static_cast<double>(q) * q;
        sv += static_cast<double>(x) * x;
        sd += static_cast<double>(k * norm_key[wat + i]) * static_cast<double>(q * norm_query[wat + i]);
    }
    sk = q4_block_sum_f64(sk, partial[0]);
    sq = q4_block_sum_f64(sq, partial[1]);
    sv = q4_block_sum_f64(sv, partial[2]);
    sd = q4_block_sum_f64(sd, partial[3]);
    const double width = static_cast<double>(n);
    const double scale_k = 1.0 / sqrt(sk / width + static_cast<double>(eps));
    const double scale_q = 1.0 / sqrt(sq / width + static_cast<double>(eps));
    const double s = sd * scale_k * scale_q / sqrt(width);
    const double sign = s > 0.0 ? 1.0 : (s < 0.0 ? -1.0 : 0.0);
    const float g = static_cast<float>(1.0 / (1.0 + exp(-sign * sqrt(fmax(fabs(s), 1e-6)))));
    // rms(value * g): the mean of squares is g^2 times the one of the value.
    const float scale_g =
        static_cast<float>(1.0 / sqrt(static_cast<double>(g) * g * sv / width + static_cast<double>(eps)));
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = v[i] * g;
        gated[at + i] = x;
        nrm[at + i] = x * scale_g * norm_conv[wat + i];
    }
    if (gate_out != nullptr && threadIdx.x == 0) {
        gate_out[static_cast<size_t>(row) * hc + c] = g;
    }
}

// The dilated convolution and the residual update over the rows of the step,
// from the histories of the sequences' slots (never written here):
//   conv[t][ch] = sum_k taps[ch][k] * in(t - (kernel - 1 - k) * dilation)[ch]
//   r[t][ch]   += gated[t][ch] + silu(conv[t][ch])
// where in(q) is the nrm of row q of the segment, and for q < 0 the history.
// `taps` is ple_conv1d as the file stores it, [D][kernel] in f16 (taps_f16 != 0)
// or f32. One thread per channel and chunk of rows of a segment.
ARES_KERNEL ares_q4_ngram_window(const AresStep* step, float* r, const float* gated, const float* nrm,
                                 const void* taps, uint32_t taps_f16, const float* history, uint64_t slot_stride,
                                 uint32_t d, uint32_t kernel, uint32_t dilation, uint32_t chunk_rows) {
    const AresSeg seg = ares_step_segs(step)[blockIdx.z];
    const uint32_t t0 = blockIdx.y * chunk_rows;
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (t0 >= seg.n_rows || ch >= d) {
        return;
    }
    const uint32_t t1 = min(t0 + chunk_rows, seg.n_rows);
    const uint32_t back = (kernel - 1) * dilation;
    const float* hist = history + static_cast<size_t>(seg.slot) * slot_stride;
    float w[Q4_NGRAM_MAX_TAPS];
#pragma unroll
    for (int k = 0; k < Q4_NGRAM_MAX_TAPS; k++) {
        w[k] = 0.0f;
        if (k < kernel) {
            const size_t at = static_cast<size_t>(ch) * kernel + k;
            w[k] = taps_f16 != 0 ? __half2float(static_cast<const __half*>(taps)[at])
                                 : static_cast<const float*>(taps)[at];
        }
    }
    for (uint32_t t = t0; t < t1; t++) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < Q4_NGRAM_MAX_TAPS; k++) {
            if (k < kernel) {
                // The index of the input in "history, then the rows of the segment".
                const uint32_t in = t + back - (kernel - 1 - k) * dilation;
                const float x = in < back ? hist[static_cast<size_t>(in) * d + ch]
                                          : nrm[(static_cast<size_t>(seg.row0) + (in - back)) * d + ch];
                acc = fmaf(w[k], x, acc);
            }
        }
        const size_t at = (static_cast<size_t>(seg.row0) + t) * d + ch;
        r[at] = r[at] + gated[at] + ares_silu(acc);
    }
}

// The commit of the history: after the first m rows of a segment it is the
// last `back` of (the old history, then the nrm of those rows).
//   which = 0: the segments that are not direct, m = accepted; 1: the direct
//   ones, m = all their rows; 2: both.
// One thread per channel, which reads its old column before it writes.
ARES_KERNEL ares_q4_ngram_commit(const AresStep* step, const float* nrm, float* history, uint64_t slot_stride,
                                 uint32_t d, uint32_t back, uint32_t which) {
    const AresSeg seg = ares_step_segs(step)[blockIdx.y];
    const bool direct = (seg.flags & ARES_SEG_DIRECT) != 0;
    if ((which == 0 && direct) || (which == 1 && !direct)) {
        return;
    }
    const uint32_t m = direct ? seg.n_rows : min(seg.accepted, seg.n_rows);
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (m == 0 || ch >= d) {
        return;
    }
    float* hist = history + static_cast<size_t>(seg.slot) * slot_stride;
    float old[Q4_NGRAM_MAX_BACK];
    for (uint32_t i = 0; i < back; i++) {
        old[i] = hist[static_cast<size_t>(i) * d + ch];
    }
    for (uint32_t i = 0; i < back; i++) {
        const uint32_t in = m + i;
        hist[static_cast<size_t>(i) * d + ch] =
            in < back ? old[in] : nrm[(static_cast<size_t>(seg.row0) + (in - back)) * d + ch];
    }
}
