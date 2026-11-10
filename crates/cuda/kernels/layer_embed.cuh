// Embedding gather: the row `token_embd[token]` decoded to f32.
//
// The table is stored as GGUF stores it (q6_K in the 0.8B, q3_K in the 27B,
// iq4_xs in qwen4exp), and the layouts are those of ggml's dequantize_row_*
// functions, which `ares_quant::dequantize` reproduces. A gather touches a few
// rows per step, so the decoder is a plain function of (row, value index) and
// each thread decodes the values it writes: coalesced stores, a handful of
// byte loads per value, no shared memory. Types whose decoding needs a
// codebook (the iq1/iq2/iq3 families, ternary, mxfp4) are not here; the host
// wrapper refuses them by name.
#pragma once

#include <cstring>

#include "common.cuh"
#include "layer_step.cuh"
#include "iq_tables.cuh"

static __device__ __forceinline__ float ares_f16_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// The 6-bit scale of sub-block s (0..15) of a Q3_K block: four low bits from
// `scales[s % 8]` (low nibble for s < 8, high nibble after) and two high bits
// from `scales[8 + s % 4]`.
static __device__ __forceinline__ int ares_q3k_scale(const uint8_t* scales, int s) {
    const int low = (s < 8) ? (scales[s] & 0xF) : (scales[s - 8] >> 4);
    const int high = (scales[8 + (s & 3)] >> (2 * (s >> 2))) & 3;
    return (low | (high << 4)) - 32;
}

// Scale and minimum of sub-block j (0..7) of a Q4_K or Q5_K block (ggml's get_scale_min_k4).
static __device__ __forceinline__ void ares_k4_scale_min(const uint8_t* q, int j, int& sc, int& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Value `j` of a row stored in type `ty` (a ggml type id). `row` is the first
// byte of the row; `j` < the row's number of values.
static __device__ float ares_embed_value(const uint8_t* row, uint32_t ty, uint32_t j) {
    switch (ty) {
    case 0:  // F32
        return reinterpret_cast<const float*>(row)[j];
    case 1:  // F16
        return __half2float(reinterpret_cast<const __half*>(row)[j]);
    case 30:  // BF16
        return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(row)[j]) << 16);
    case 2: {  // Q4_0: 18 bytes, d, 16 bytes of nibbles (low nibbles are values 0..15)
        const uint8_t* b = row + (j >> 5) * 18;
        const uint32_t jj = j & 31;
        const uint32_t q = b[2 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        return static_cast<float>(nib - 8) * ares_f16_at(b);
    }
    case 3: {  // Q4_1: 20 bytes, d, m
        const uint8_t* b = row + (j >> 5) * 20;
        const uint32_t jj = j & 31;
        const uint32_t q = b[4 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        return static_cast<float>(nib) * ares_f16_at(b) + ares_f16_at(b + 2);
    }
    case 6: {  // Q5_0: 22 bytes, d, qh (32 bits), 16 bytes of nibbles
        const uint8_t* b = row + (j >> 5) * 22;
        const uint32_t jj = j & 31;
        uint32_t qh;
        memcpy(&qh, b + 2, 4);
        const uint32_t q = b[6 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        const int hi = (qh >> jj) & 1;
        return static_cast<float>((nib | (hi << 4)) - 16) * ares_f16_at(b);
    }
    case 7: {  // Q5_1: 24 bytes, d, m, qh, nibbles
        const uint8_t* b = row + (j >> 5) * 24;
        const uint32_t jj = j & 31;
        uint32_t qh;
        memcpy(&qh, b + 4, 4);
        const uint32_t q = b[8 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        const int hi = (qh >> jj) & 1;
        return static_cast<float>(nib | (hi << 4)) * ares_f16_at(b) + ares_f16_at(b + 2);
    }
    case 8: {  // Q8_0: 34 bytes, d, 32 int8
        const uint8_t* b = row + (j >> 5) * 34;
        return static_cast<float>(static_cast<int8_t>(b[2 + (j & 31)])) * ares_f16_at(b);
    }
    case 10: {  // Q2_K: 84 bytes, scales[16], qs[64], d, dmin
        const uint8_t* b = row + (j >> 8) * 84;
        const uint32_t jj = j & 255;
        const uint32_t half = jj >> 7, within = jj & 127;
        const uint32_t group = within >> 5, l = within & 31;
        const uint32_t sc = b[half * 8 + group * 2 + (l >> 4)];
        const uint32_t q = (b[16 + half * 32 + l] >> (2 * group)) & 3;
        const float d = ares_f16_at(b + 80), dmin = ares_f16_at(b + 82);
        return d * static_cast<float>(sc & 0xF) * static_cast<float>(q) - dmin * static_cast<float>(sc >> 4);
    }
    case 11: {  // Q3_K: 110 bytes, hmask[32], qs[64], scales[12], d
        const uint8_t* b = row + (j >> 8) * 110;
        const uint32_t jj = j & 255;
        const uint32_t half = jj >> 7, within = jj & 127;
        const uint32_t group = within >> 5, l = within & 31;  // shift = 2 * group, mask bit = half * 4 + group
        const int s = static_cast<int>(half * 8 + group * 2 + (l >> 4));
        const float dl = ares_f16_at(b + 108) * static_cast<float>(ares_q3k_scale(b + 96, s));
        const int low = (b[32 + half * 32 + l] >> (2 * group)) & 3;
        const int high = (b[l] >> (half * 4 + group)) & 1;
        return dl * static_cast<float>(low - (high ? 0 : 4));
    }
    case 12: {  // Q4_K: 144 bytes, d, dmin, scales[12], qs[128]
        const uint8_t* b = row + (j >> 8) * 144;
        const uint32_t jj = j & 255;
        const uint32_t chunk = jj >> 6, within = jj & 63;
        const bool high = within >= 32;
        int sc, mn;
        ares_k4_scale_min(b + 4, static_cast<int>(2 * chunk + (high ? 1 : 0)), sc, mn);
        const uint32_t q = b[16 + chunk * 32 + (within & 31)];
        const float v = static_cast<float>(high ? (q >> 4) : (q & 0xF));
        return ares_f16_at(b) * static_cast<float>(sc) * v - ares_f16_at(b + 2) * static_cast<float>(mn);
    }
    case 13: {  // Q5_K: 176 bytes, d, dmin, scales[12], qh[32], qs[128]
        const uint8_t* b = row + (j >> 8) * 176;
        const uint32_t jj = j & 255;
        const uint32_t chunk = jj >> 6, within = jj & 63;
        const bool high = within >= 32;
        const uint32_t l = within & 31;
        int sc, mn;
        ares_k4_scale_min(b + 4, static_cast<int>(2 * chunk + (high ? 1 : 0)), sc, mn);
        const uint32_t q = b[48 + chunk * 32 + l];
        const uint32_t bit = (b[16 + l] >> (2 * chunk + (high ? 1 : 0))) & 1;
        const float v = static_cast<float>((high ? (q >> 4) : (q & 0xF)) + (bit ? 16 : 0));
        return ares_f16_at(b) * static_cast<float>(sc) * v - ares_f16_at(b + 2) * static_cast<float>(mn);
    }
    case 14: {  // Q6_K: 210 bytes, ql[128], qh[64], scales[16] (int8), d
        const uint8_t* b = row + (j >> 8) * 210;
        const uint32_t jj = j & 255;
        const uint32_t half = jj >> 7, within = jj & 127;
        const uint32_t part = within >> 5, l = within & 31;  // part 0..3: values l, l+32, l+64, l+96
        const uint8_t* ql = b + half * 64;
        const uint8_t* qh = b + 128 + half * 32;
        const uint32_t lo = (part & 1) ? ql[l + 32] : ql[l];
        const uint32_t nib = (part >= 2) ? (lo >> 4) : (lo & 0xF);
        const uint32_t hi = (qh[l] >> (2 * part)) & 3;
        const int q = static_cast<int>(nib | (hi << 4)) - 32;
        const int sc = static_cast<int8_t>(b[192 + half * 8 + (l >> 4) + 2 * part]);
        return ares_f16_at(b + 208) * static_cast<float>(sc) * static_cast<float>(q);
    }
    case 20: {  // IQ4_NL: 18 bytes, d, 16 bytes of nibbles through a table
        const uint8_t* b = row + (j >> 5) * 18;
        const uint32_t jj = j & 31;
        const uint32_t q = b[2 + (jj & 15)];
        const uint32_t nib = jj < 16 ? (q & 0xF) : (q >> 4);
        return ares_f16_at(b) * static_cast<float>(ares_kvalues_iq4nl[nib]);
    }
    case 23: {  // IQ4_XS: 136 bytes, d, scales_h (16 bits), scales_l[4], qs[128]
        const uint8_t* b = row + (j >> 8) * 136;
        const uint32_t jj = j & 255;
        const uint32_t ib = jj >> 5, jb = jj & 31;
        const uint32_t scales_h = *reinterpret_cast<const uint16_t*>(b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const uint32_t q = b[8 + ib * 16 + (jb & 15)];
        const uint32_t nib = jb < 16 ? (q & 0xF) : (q >> 4);
        return ares_f16_at(b) * static_cast<float>(ls - 32) * static_cast<float>(ares_kvalues_iq4nl[nib]);
    }
    default:
        return __int_as_float(0x7fc00000);  // NaN: the host refuses unsupported types
    }
}

// One block per gathered row; the rows come from the step record's tokens.
// A token past the table writes zeros (the host checks ids; the kernel does not trap).
ARES_KERNEL ares_embed_gather(const AresStep* step, const uint8_t* table, uint32_t ty, uint64_t row_bytes,
                              uint32_t n_vocab, uint32_t n_embd, float* out) {
    const uint32_t r = blockIdx.x;
    if (r >= step->n_rows) {
        return;
    }
    const uint32_t token = ares_step_tokens(step)[r];
    float* dst = out + static_cast<size_t>(r) * n_embd;
    if (token >= n_vocab) {
        for (uint32_t j = threadIdx.x; j < n_embd; j += blockDim.x) {
            dst[j] = 0.0f;
        }
        return;
    }
    const uint8_t* src = table + static_cast<size_t>(token) * row_bytes;
    for (uint32_t j = threadIdx.x; j < n_embd; j += blockDim.x) {
        dst[j] = ares_embed_value(src, ty, j);
    }
}
