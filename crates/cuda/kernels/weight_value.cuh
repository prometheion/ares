// The values of a weight row in every weight type a GGUF file can carry, one value at a time.
//
// `AresWeight<TY>` (TY a ggml type id) has the size of a block of the type and the decoder
// of value `j` of a row: the layouts are those of ggml's dequantize_row_* functions, which
// `ares_quant::dequantize` reproduces. The decoders are plain functions of (row, value
// index), made for kernels that touch a few values per thread: the embedding gather
// (layer_embed.cuh), the selectors of the DFlash drafter (dflash.cu), the fused read of the
// stream mixer (q4_fused.cuh) and the weight kernel that multiplies unquantized activations
// with any type (mmv_exact.cuh). A kernel that knows the type at compile time takes the
// template; `ares_embed_value` and `ares_weight4_any` choose by a type id at run time.
// The codebooks are those of the weight kernels (iq_tables.cuh) and the scales of the
// 4-bit floats are theirs too (fp4_scales.cuh).
//
// The decoders compute as ggml does, except that the compiler may fuse a multiply and an
// add (`d * q + m`) where the CPU rounds twice: a value can differ from the CPU's decode in
// its last bit. Types whose value is one product (Q8_0, Q4_0, the float types) decode
// exactly.
#pragma once

#include <cstring>

#include "common.cuh"
#include "fp4_scales.cuh"
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

// Four bytes at any alignment, little endian (the blocks of most types start 2 bytes off a
// word boundary).
static __device__ __forceinline__ uint32_t ares_u32_at(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static __device__ __forceinline__ uint32_t ares_u16_at(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
}

// Byte `k` (0..7) of a codebook entry, as a magnitude.
static __device__ __forceinline__ float ares_grid_byte(uint64_t entry, uint32_t k) {
    return static_cast<float>(static_cast<uint32_t>(entry >> (8 * k)) & 0xFF);
}

// -1 or +1 by bit `k` of the sign byte (ggml's kmask_iq2xs).
static __device__ __forceinline__ float ares_sign_of(uint32_t signs, uint32_t k) {
    return ((signs >> k) & 1) ? -1.0f : 1.0f;
}

// The 4-bit scale of the 16 values `l` (0..3) of a 32-value sub-block of IQ2_XS and IQ2_S:
// two scales in a byte, the low one for the first 16 values.
static __device__ __forceinline__ float ares_iq2_scale(float d, uint32_t scales, uint32_t l) {
    return d * (0.5f + static_cast<float>((scales >> (4 * (l >> 1))) & 0xF)) * 0.25f;
}

// The weight type with ggml type id `TY`: `VALUES` values in a block of `BYTES` bytes, and
// `value(row, j)`, value `j` of a row that starts at `row` (`j` < the row's number of
// values). The float types are blocks of one value.
template <uint32_t TY>
struct AresWeight;

// F32
template <>
struct AresWeight<0> {
    static constexpr uint32_t VALUES = 1;
    static constexpr uint32_t BYTES = 4;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        return reinterpret_cast<const float*>(row)[j];
    }
};

// F16
template <>
struct AresWeight<1> {
    static constexpr uint32_t VALUES = 1;
    static constexpr uint32_t BYTES = 2;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        return __half2float(reinterpret_cast<const __half*>(row)[j]);
    }
};

// BF16
template <>
struct AresWeight<30> {
    static constexpr uint32_t VALUES = 1;
    static constexpr uint32_t BYTES = 2;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(row)[j]) << 16);
    }
};

// Q4_0: 18 bytes, d, 16 bytes of nibbles (low nibbles are values 0..15)
template <>
struct AresWeight<2> {
    static constexpr uint32_t VALUES = 32;
    static constexpr uint32_t BYTES = 18;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 5) * 18;
        const uint32_t jj = j & 31;
        const uint32_t q = b[2 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        return static_cast<float>(nib - 8) * ares_f16_at(b);
    }
};

// Q4_1: 20 bytes, d, m
template <>
struct AresWeight<3> {
    static constexpr uint32_t VALUES = 32;
    static constexpr uint32_t BYTES = 20;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 5) * 20;
        const uint32_t jj = j & 31;
        const uint32_t q = b[4 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        return static_cast<float>(nib) * ares_f16_at(b) + ares_f16_at(b + 2);
    }
};

// Q5_0: 22 bytes, d, qh (32 bits), 16 bytes of nibbles
template <>
struct AresWeight<6> {
    static constexpr uint32_t VALUES = 32;
    static constexpr uint32_t BYTES = 22;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 5) * 22;
        const uint32_t jj = j & 31;
        uint32_t qh;
        memcpy(&qh, b + 2, 4);
        const uint32_t q = b[6 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        const int hi = (qh >> jj) & 1;
        return static_cast<float>((nib | (hi << 4)) - 16) * ares_f16_at(b);
    }
};

// Q5_1: 24 bytes, d, m, qh, nibbles
template <>
struct AresWeight<7> {
    static constexpr uint32_t VALUES = 32;
    static constexpr uint32_t BYTES = 24;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 5) * 24;
        const uint32_t jj = j & 31;
        uint32_t qh;
        memcpy(&qh, b + 4, 4);
        const uint32_t q = b[8 + (jj & 15)];
        const int nib = jj < 16 ? (q & 0xF) : (q >> 4);
        const int hi = (qh >> jj) & 1;
        return static_cast<float>(nib | (hi << 4)) * ares_f16_at(b) + ares_f16_at(b + 2);
    }
};

// Q8_0: 34 bytes, d, 32 int8
template <>
struct AresWeight<8> {
    static constexpr uint32_t VALUES = 32;
    static constexpr uint32_t BYTES = 34;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 5) * 34;
        return static_cast<float>(static_cast<int8_t>(b[2 + (j & 31)])) * ares_f16_at(b);
    }
};

// Q2_K: 84 bytes, scales[16], qs[64], d, dmin
template <>
struct AresWeight<10> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 84;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 84;
        const uint32_t jj = j & 255;
        const uint32_t half = jj >> 7, within = jj & 127;
        const uint32_t group = within >> 5, l = within & 31;
        const uint32_t sc = b[half * 8 + group * 2 + (l >> 4)];
        const uint32_t q = (b[16 + half * 32 + l] >> (2 * group)) & 3;
        const float d = ares_f16_at(b + 80), dmin = ares_f16_at(b + 82);
        return d * static_cast<float>(sc & 0xF) * static_cast<float>(q) - dmin * static_cast<float>(sc >> 4);
    }
};

// Q3_K: 110 bytes, hmask[32], qs[64], scales[12], d
template <>
struct AresWeight<11> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 110;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
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
};

// Q4_K: 144 bytes, d, dmin, scales[12], qs[128]
template <>
struct AresWeight<12> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 144;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
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
};

// Q5_K: 176 bytes, d, dmin, scales[12], qh[32], qs[128]
template <>
struct AresWeight<13> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 176;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
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
};

// Q6_K: 210 bytes, ql[128], qh[64], scales[16] (int8), d
template <>
struct AresWeight<14> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 210;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
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
};

// IQ4_NL: 18 bytes, d, 16 bytes of nibbles through a table
template <>
struct AresWeight<20> {
    static constexpr uint32_t VALUES = 32;
    static constexpr uint32_t BYTES = 18;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 5) * 18;
        const uint32_t jj = j & 31;
        const uint32_t q = b[2 + (jj & 15)];
        const uint32_t nib = jj < 16 ? (q & 0xF) : (q >> 4);
        return ares_f16_at(b) * static_cast<float>(ares_kvalues_iq4nl[nib]);
    }
};

// IQ4_XS: 136 bytes, d, scales_h (16 bits), scales_l[4], qs[128]
template <>
struct AresWeight<23> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 136;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 136;
        const uint32_t jj = j & 255;
        const uint32_t ib = jj >> 5, jb = jj & 31;
        const uint32_t scales_h = *reinterpret_cast<const uint16_t*>(b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const uint32_t q = b[8 + ib * 16 + (jb & 15)];
        const uint32_t nib = jb < 16 ? (q & 0xF) : (q >> 4);
        return ares_f16_at(b) * static_cast<float>(ls - 32) * static_cast<float>(ares_kvalues_iq4nl[nib]);
    }
};

// IQ2_XXS: 66 bytes, d, per 32 values 4 codebook indices and a word of 4 sign indices + scale
template <>
struct AresWeight<16> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 66;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 66;
        const uint32_t jj = j & 255, ib32 = jj >> 5, l = (jj >> 3) & 3, k = jj & 7;
        const uint32_t aux = ares_u32_at(b + 6 + 8 * ib32);
        const float db = ares_f16_at(b) * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
        const float g = ares_grid_byte(ares_iq2xxs_grid[b[2 + 8 * ib32 + l]], k);
        return db * g * ares_sign_of(ares_ksigns_iq2xs[(aux >> (7 * l)) & 127], k);
    }
};

// IQ2_XS: 74 bytes, d, 32 x (9-bit index, 7-bit sign index), 8 scale bytes
template <>
struct AresWeight<17> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 74;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 74;
        const uint32_t jj = j & 255, ib32 = jj >> 5, l = (jj >> 3) & 3, k = jj & 7;
        const uint32_t q = ares_u16_at(b + 2 + 2 * (4 * ib32 + l));
        const float db = ares_iq2_scale(ares_f16_at(b), b[66 + ib32], l);
        return db * ares_grid_byte(ares_iq2xs_grid[q & 511], k) * ares_sign_of(ares_ksigns_iq2xs[q >> 9], k);
    }
};

// IQ2_S: 82 bytes, d, qs[64] (indices, then sign bytes), qh[8], scales[8]
template <>
struct AresWeight<22> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 82;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 82;
        const uint32_t jj = j & 255, ib32 = jj >> 5, l = (jj >> 3) & 3, k = jj & 7;
        const uint32_t idx = b[2 + 4 * ib32 + l] | ((static_cast<uint32_t>(b[66 + ib32]) << (8 - 2 * l)) & 0x300);
        const float db = ares_iq2_scale(ares_f16_at(b), b[74 + ib32], l);
        return db * ares_grid_byte(ares_iq2s_grid[idx], k) * ares_sign_of(b[34 + 4 * ib32 + l], k);
    }
};

// IQ3_XXS: 98 bytes, d, qs[64] (indices), per 32 values a word of 4 sign indices + scale
template <>
struct AresWeight<18> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 98;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 98;
        const uint32_t jj = j & 255, ib32 = jj >> 5, l = (jj >> 3) & 3, k = jj & 7;
        const uint32_t aux = ares_u32_at(b + 66 + 4 * ib32);
        const float db = ares_f16_at(b) * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
        const uint32_t grid = ares_iq3xxs_grid[b[2 + 8 * ib32 + 2 * l + (k >> 2)]];
        const float g = static_cast<float>((grid >> (8 * (k & 3))) & 0xFF);
        return db * g * ares_sign_of(ares_ksigns_iq2xs[(aux >> (7 * l)) & 127], k);
    }
};

// IQ3_S: 110 bytes, d, qs[64], qh[8], signs[32], scales[4]
template <>
struct AresWeight<21> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 110;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 110;
        const uint32_t jj = j & 255, ib32 = jj >> 5, l = (jj >> 3) & 3, k = jj & 7, part = k >> 2;
        const uint32_t scale = (b[106 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xF;
        const float db = ares_f16_at(b) * static_cast<float>(1 + 2 * scale);
        const uint32_t idx = b[2 + 8 * ib32 + 2 * l + part] | ((static_cast<uint32_t>(b[66 + ib32]) << (8 - 2 * l - part)) & 256);
        const float g = static_cast<float>((ares_iq3s_grid[idx] >> (8 * (k & 3))) & 0xFF);
        return db * g * ares_sign_of(b[74 + 4 * ib32 + l], k);
    }
};

// IQ1_S: 50 bytes, d, qs[32], qh[8] (u16: 4 x 3 index bits, 3-bit scale, delta sign)
template <>
struct AresWeight<19> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 50;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 50;
        const uint32_t jj = j & 255, ib = jj >> 5, l = (jj >> 3) & 3, k = jj & 7;
        const uint32_t qh = ares_u16_at(b + 34 + 2 * ib);
        const float dl = ares_f16_at(b) * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
        const uint32_t idx = b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
        return dl * (static_cast<float>(static_cast<int8_t>(ares_iq1s_grid[idx] >> (8 * k))) + delta);
    }
};

// IQ1_M: 56 bytes, qs[32], qh[16], scales[8] (four u16 that also carry the f16 d)
template <>
struct AresWeight<29> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 56;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 56;
        const uint32_t jj = j & 255, ib = jj >> 5, l = (jj >> 3) & 3, k = jj & 7;
        const uint32_t sc0 = ares_u16_at(b + 48), sc1 = ares_u16_at(b + 50), sc2 = ares_u16_at(b + 52),
                       sc3 = ares_u16_at(b + 54);
        const uint32_t d_bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000);
        const uint32_t sc = ares_u16_at(b + 48 + 2 * (ib >> 1));
        const float dl = __half2float(__ushort_as_half(static_cast<unsigned short>(d_bits))) *
                         static_cast<float>(2 * ((sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1);
        const uint32_t h = b[32 + 2 * ib + (l >> 1)];
        const uint32_t idx = b[4 * ib + l] | (((l & 1) ? (h << 4) : (h << 8)) & 0x700);
        const float delta = (h & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
        return dl * (static_cast<float>(static_cast<int8_t>(ares_iq1s_grid[idx] >> (8 * k))) + delta);
    }
};

// TQ1_0: 54 bytes, qs[48] (5 base-3 digits a byte), qh[4] (4 digits a byte), d
template <>
struct AresWeight<34> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 54;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 54;
        const uint32_t jj = j & 255;
        // The digit planes: 5 of 32 bytes (values 0..159), 5 of 16 bytes (160..239), 4 of 4 bytes.
        uint32_t digit, byte;
        if (jj < 160) {
            digit = jj >> 5;
            byte = b[jj & 31];
        } else if (jj < 240) {
            digit = (jj - 160) >> 4;
            byte = b[32 + ((jj - 160) & 15)];
        } else {
            digit = (jj - 240) >> 2;
            byte = b[48 + ((jj - 240) & 3)];
        }
        uint32_t pow3 = 1;
        for (uint32_t i = 0; i < digit; i++) {
            pow3 *= 3;
        }
        // Times 3^digit moves the digit to the top of the byte's fraction; * 3 >> 8 reads it.
        const int trit = static_cast<int>(((byte * pow3) & 0xFF) * 3 >> 8) - 1;
        return static_cast<float>(trit) * ares_f16_at(b + 52);
    }
};

// TQ2_0: 66 bytes, qs[64] (4 codes a byte, in planes of 32 bytes), d
template <>
struct AresWeight<35> {
    static constexpr uint32_t VALUES = 256;
    static constexpr uint32_t BYTES = 66;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 8) * 66;
        const uint32_t jj = j & 255;
        const uint32_t code = (b[32 * (jj >> 7) + (jj & 31)] >> (2 * ((jj >> 5) & 3))) & 3;
        return static_cast<float>(static_cast<int>(code) - 1) * ares_f16_at(b + 64);
    }
};

// MXFP4: 17 bytes, E8M0 scale, 16 bytes of nibbles through the E2M1 levels
template <>
struct AresWeight<39> {
    static constexpr uint32_t VALUES = 32;
    static constexpr uint32_t BYTES = 17;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 5) * 17;
        const uint32_t jj = j & 31;
        const uint32_t q = b[1 + (jj & 15)];
        const uint32_t nib = jj < 16 ? (q & 0xF) : (q >> 4);
        return static_cast<float>(ares_kvalues_mxfp4[nib]) * ares_e8m0_half(b[0]);
    }
};

// NVFP4: 36 bytes, 4 UE4M3 scales, 4 sub-blocks of 16 values in 8 bytes of nibbles
template <>
struct AresWeight<40> {
    static constexpr uint32_t VALUES = 64;
    static constexpr uint32_t BYTES = 36;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 6) * 36;
        const uint32_t jj = j & 63, s = jj >> 4, r = jj & 15;
        const uint32_t q = b[4 + 8 * s + (r & 7)];
        const uint32_t nib = r < 8 ? (q & 0xF) : (q >> 4);
        return static_cast<float>(ares_kvalues_mxfp4[nib]) * ares_ue4m3_half(b[s]);
    }
};

// Q1_0: 18 bytes, d, 128 sign bits (set is +d)
template <>
struct AresWeight<41> {
    static constexpr uint32_t VALUES = 128;
    static constexpr uint32_t BYTES = 18;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 7) * 18;
        const uint32_t jj = j & 127;
        const float d = ares_f16_at(b);
        return ((b[2 + (jj >> 3)] >> (jj & 7)) & 1) ? d : -d;
    }
};

// Q2_0: 18 bytes, d, 64 codes of 2 bits (-1, 0, 1, 2 times d)
template <>
struct AresWeight<42> {
    static constexpr uint32_t VALUES = 64;
    static constexpr uint32_t BYTES = 18;
    static __device__ __forceinline__ float value(const uint8_t* row, uint32_t j) {
        const uint8_t* b = row + (j >> 6) * 18;
        const uint32_t jj = j & 63;
        const uint32_t code = (b[2 + (jj >> 2)] >> (2 * (jj & 3))) & 3;
        return static_cast<float>(static_cast<int>(code) - 1) * ares_f16_at(b);
    }
};

// Value `j` of a row stored in type `ty` (a ggml type id). `row` is the first
// byte of the row; `j` < the row's number of values.
static __device__ float ares_embed_value(const uint8_t* row, uint32_t ty, uint32_t j) {
    switch (ty) {
    case 0:
        return AresWeight<0>::value(row, j);
    case 1:
        return AresWeight<1>::value(row, j);
    case 30:
        return AresWeight<30>::value(row, j);
    case 2:
        return AresWeight<2>::value(row, j);
    case 3:
        return AresWeight<3>::value(row, j);
    case 6:
        return AresWeight<6>::value(row, j);
    case 7:
        return AresWeight<7>::value(row, j);
    case 8:
        return AresWeight<8>::value(row, j);
    case 10:
        return AresWeight<10>::value(row, j);
    case 11:
        return AresWeight<11>::value(row, j);
    case 12:
        return AresWeight<12>::value(row, j);
    case 13:
        return AresWeight<13>::value(row, j);
    case 14:
        return AresWeight<14>::value(row, j);
    case 20:
        return AresWeight<20>::value(row, j);
    case 23:
        return AresWeight<23>::value(row, j);
    case 16:
        return AresWeight<16>::value(row, j);
    case 17:
        return AresWeight<17>::value(row, j);
    case 22:
        return AresWeight<22>::value(row, j);
    case 18:
        return AresWeight<18>::value(row, j);
    case 21:
        return AresWeight<21>::value(row, j);
    case 19:
        return AresWeight<19>::value(row, j);
    case 29:
        return AresWeight<29>::value(row, j);
    case 34:
        return AresWeight<34>::value(row, j);
    case 35:
        return AresWeight<35>::value(row, j);
    case 39:
        return AresWeight<39>::value(row, j);
    case 40:
        return AresWeight<40>::value(row, j);
    case 41:
        return AresWeight<41>::value(row, j);
    case 42:
        return AresWeight<42>::value(row, j);
    default:
        return __int_as_float(0x7fc00000);  // NaN: the host refuses unsupported types
    }
}

// Values `j` to `j + 3` of a row of type `TY`; `j` is a multiple of 4, so the four are in
// one block of every type.
template <uint32_t TY>
static __device__ __forceinline__ float4 ares_weight4(const uint8_t* row, uint32_t j) {
    return make_float4(AresWeight<TY>::value(row, j), AresWeight<TY>::value(row, j + 1),
                       AresWeight<TY>::value(row, j + 2), AresWeight<TY>::value(row, j + 3));
}

// `ares_weight4` for a type known at run time (a ggml type id); NaNs for a type that holds
// no weights. Not inlined: a call costs little next to a decoder of any type, and inlined
// into an unrolled loop the decoders of every type made a kernel take a minute to compile.
static __device__ __noinline__ float4 ares_weight4_any(const uint8_t* row, uint32_t ty, uint32_t j) {
    switch (ty) {
    case 0:
        return ares_weight4<0>(row, j);
    case 1:
        return ares_weight4<1>(row, j);
    case 30:
        return ares_weight4<30>(row, j);
    case 2:
        return ares_weight4<2>(row, j);
    case 3:
        return ares_weight4<3>(row, j);
    case 6:
        return ares_weight4<6>(row, j);
    case 7:
        return ares_weight4<7>(row, j);
    case 8:
        return ares_weight4<8>(row, j);
    case 10:
        return ares_weight4<10>(row, j);
    case 11:
        return ares_weight4<11>(row, j);
    case 12:
        return ares_weight4<12>(row, j);
    case 13:
        return ares_weight4<13>(row, j);
    case 14:
        return ares_weight4<14>(row, j);
    case 20:
        return ares_weight4<20>(row, j);
    case 23:
        return ares_weight4<23>(row, j);
    case 16:
        return ares_weight4<16>(row, j);
    case 17:
        return ares_weight4<17>(row, j);
    case 22:
        return ares_weight4<22>(row, j);
    case 18:
        return ares_weight4<18>(row, j);
    case 21:
        return ares_weight4<21>(row, j);
    case 19:
        return ares_weight4<19>(row, j);
    case 29:
        return ares_weight4<29>(row, j);
    case 34:
        return ares_weight4<34>(row, j);
    case 35:
        return ares_weight4<35>(row, j);
    case 39:
        return ares_weight4<39>(row, j);
    case 40:
        return ares_weight4<40>(row, j);
    case 41:
        return ares_weight4<41>(row, j);
    case 42:
        return ares_weight4<42>(row, j);
    default: {
        const float nan = __int_as_float(0x7fc00000);
        return make_float4(nan, nan, nan, nan);
    }
    }
}

// The bytes of a row of `n` values of type `ty` (a ggml type id; `n` whole blocks of it); 0
// for a type that holds no weights.
static __device__ __forceinline__ size_t ares_weight_row_bytes(uint32_t ty, size_t n) {
    switch (ty) {
    case 0:
        return n / AresWeight<0>::VALUES * AresWeight<0>::BYTES;
    case 1:
        return n / AresWeight<1>::VALUES * AresWeight<1>::BYTES;
    case 30:
        return n / AresWeight<30>::VALUES * AresWeight<30>::BYTES;
    case 2:
        return n / AresWeight<2>::VALUES * AresWeight<2>::BYTES;
    case 3:
        return n / AresWeight<3>::VALUES * AresWeight<3>::BYTES;
    case 6:
        return n / AresWeight<6>::VALUES * AresWeight<6>::BYTES;
    case 7:
        return n / AresWeight<7>::VALUES * AresWeight<7>::BYTES;
    case 8:
        return n / AresWeight<8>::VALUES * AresWeight<8>::BYTES;
    case 10:
        return n / AresWeight<10>::VALUES * AresWeight<10>::BYTES;
    case 11:
        return n / AresWeight<11>::VALUES * AresWeight<11>::BYTES;
    case 12:
        return n / AresWeight<12>::VALUES * AresWeight<12>::BYTES;
    case 13:
        return n / AresWeight<13>::VALUES * AresWeight<13>::BYTES;
    case 14:
        return n / AresWeight<14>::VALUES * AresWeight<14>::BYTES;
    case 20:
        return n / AresWeight<20>::VALUES * AresWeight<20>::BYTES;
    case 23:
        return n / AresWeight<23>::VALUES * AresWeight<23>::BYTES;
    case 16:
        return n / AresWeight<16>::VALUES * AresWeight<16>::BYTES;
    case 17:
        return n / AresWeight<17>::VALUES * AresWeight<17>::BYTES;
    case 22:
        return n / AresWeight<22>::VALUES * AresWeight<22>::BYTES;
    case 18:
        return n / AresWeight<18>::VALUES * AresWeight<18>::BYTES;
    case 21:
        return n / AresWeight<21>::VALUES * AresWeight<21>::BYTES;
    case 19:
        return n / AresWeight<19>::VALUES * AresWeight<19>::BYTES;
    case 29:
        return n / AresWeight<29>::VALUES * AresWeight<29>::BYTES;
    case 34:
        return n / AresWeight<34>::VALUES * AresWeight<34>::BYTES;
    case 35:
        return n / AresWeight<35>::VALUES * AresWeight<35>::BYTES;
    case 39:
        return n / AresWeight<39>::VALUES * AresWeight<39>::BYTES;
    case 40:
        return n / AresWeight<40>::VALUES * AresWeight<40>::BYTES;
    case 41:
        return n / AresWeight<41>::VALUES * AresWeight<41>::BYTES;
    case 42:
        return n / AresWeight<42>::VALUES * AresWeight<42>::BYTES;
    default:
        return 0;
    }
}
