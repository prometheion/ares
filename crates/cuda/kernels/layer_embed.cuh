// Embedding gather: the row `token_embd[token]` decoded to f32.
//
// The rows come from two places: rows the device keeps, and rows the host staged for the step.
// The rows are stored as GGUF stores them (q6_K in the 0.8B, q3_K in the 27B,
// iq4_xs in qwen4exp, iq2_xxs in the 2-bit file of the 27B), and the layouts
// are those of ggml's dequantize_row_* functions, which `ares_quant::dequantize`
// reproduces; every weight type has a decoder (weight_value.cuh). A gather
// touches a few rows per step, so the decoder is a plain function of (row,
// value index) and each thread decodes the values it writes: coalesced stores,
// a handful of byte loads per value, no shared memory.
#pragma once

#include "common.cuh"
#include "layer_step.cuh"
#include "weight_value.cuh"

// The place of a token's row among the rows kept on the device: the position of `token` in
// `tokens` (ascending, `count` of them), or `token` itself when there is no list (the rows
// are those of the first `count` tokens); `count` when the device keeps no row of it.
static __device__ __forceinline__ uint32_t ares_resident_row(const uint32_t* tokens, uint32_t count,
                                                             uint32_t token) {
    if (tokens == nullptr) {
        return token < count ? token : count;
    }
    uint32_t lo = 0;
    uint32_t hi = count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (tokens[mid] < token) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (lo < count && tokens[lo] == token) ? lo : count;
}

// One block per gathered row; the tokens come from the step record. A row is read from the
// rows kept on the device when they hold its token, and otherwise from the row the host
// staged for it (row r of the step is `staged[r]`). A token past the table, and a token that
// neither holds, gives zeros (the host checks ids; the kernel does not trap).
ARES_KERNEL ares_embed_gather(const AresStep* step, const uint8_t* resident, const uint32_t* resident_tokens,
                              uint32_t n_resident, const uint8_t* staged, uint32_t ty, uint64_t row_bytes,
                              uint32_t n_vocab, uint32_t n_embd, float* out) {
    const uint32_t r = blockIdx.x;
    if (r >= step->n_rows) {
        return;
    }
    const uint32_t token = ares_step_tokens(step)[r];
    float* dst = out + static_cast<size_t>(r) * n_embd;
    const uint8_t* src = nullptr;
    if (token < n_vocab) {
        const uint32_t row = ares_resident_row(resident_tokens, n_resident, token);
        if (row < n_resident) {
            src = resident + static_cast<size_t>(row) * row_bytes;
        } else if (staged != nullptr) {
            src = staged + static_cast<size_t>(r) * row_bytes;
        }
    }
    if (src == nullptr) {
        for (uint32_t j = threadIdx.x; j < n_embd; j += blockDim.x) {
            dst[j] = 0.0f;
        }
        return;
    }
    for (uint32_t j = threadIdx.x; j < n_embd; j += blockDim.x) {
        dst[j] = ares_embed_value(src, ty, j);
    }
}
