// The KV formats of the engine on the device (attention module, attention.cu).
//
// This is crates/quant/src/kv.rs written for a warp: one warp encodes one group
// of 32 values, lane i holding value i. The specification fixes the order of
// every float operation so that the bytes are the same as the CPU's, and so
// every operation here is an explicit round-to-nearest intrinsic that the
// compiler may not fuse (`__fadd_rn`, `__fmul_rn`, `__fdiv_rn`), and the
// reductions are the exchanges the specification names (`sum32`: the pairs at
// distance 16 first, then 8, 4, 2, 1, which an xor butterfly gives to every
// lane with the same bits, because float addition is commutative).
//
// Format numbers are the order of `KvFormat::ALL`: f16 q8 q7 q6 q5 q4.
#pragma once

#include "common.cuh"

#define ATTN_TILE_CELLS 128

enum AttnFormat { FMT_F16 = 0, FMT_Q8 = 1, FMT_Q7 = 2, FMT_Q6 = 3, FMT_Q5 = 4, FMT_Q4 = 5 };

__host__ __device__ constexpr int fmt_bits(int f) { return f == 0 ? 0 : 9 - f; }
__host__ __device__ constexpr int fmt_group_bytes(int f) { return f == 0 ? 64 : 2 + 4 * (9 - f); }
__host__ __device__ constexpr bool fmt_has_mean(int f) { return f >= 2; }
__host__ __device__ constexpr bool fmt_rotated(int f) { return f >= 1; }

// Bits of the status word the writing kernels or-in (read back by the host now
// and then; the reference returns these as errors).
#define ATTN_STATUS_NONFINITE 1u    // an input value was NaN or infinite
#define ATTN_STATUS_OVERFLOW 2u     // the rotation of a group overflowed f32
#define ATTN_STATUS_DIRECT_MEAN 4u  // a direct write into an entry of a format with a mean

static __device__ __forceinline__ unsigned f2h_bits(float x) {
    return __half_as_ushort(__float2half_rn(x));
}
static __device__ __forceinline__ float h2f_bits(unsigned b) {
    return __half2float(__ushort_as_half((unsigned short)b));
}

static __device__ __forceinline__ float warp_max32(float v) {
#pragma unroll
    for (int h = 16; h > 0; h >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, h));
    return v;
}

// The sum of 32 values in the specified order; every lane gets the same bits.
static __device__ __forceinline__ float sum32(float v) {
#pragma unroll
    for (int h = 16; h > 0; h >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, h));
    return v;
}

// The 32-point Walsh-Hadamard transform of the group, scaled by 1/sqrt(32):
// stage widths 1 to 16, the lower element gets the sum, the upper the
// difference (old values), then one multiply.
static __device__ __forceinline__ float rotate32(float v, int lane) {
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, h);
        v = (lane & h) ? __fsub_rn(o, v) : __fadd_rn(v, o);
    }
    return __fmul_rn(v, __uint_as_float(0x3e3504f3u));
}

// The stored scale (an f16 pattern) of a group whose largest rotated magnitude
// is `amax`: the smallest positive f16 not below amax / qmax.
static __device__ __forceinline__ unsigned scale_bits(float amax, int bits) {
    if (amax == 0.0f) return 0;
    const float q = __fdiv_rn(amax, (float)((1 << (bits - 1)) - 1));
    const unsigned nearest = f2h_bits(q);
    if (nearest >= 0x7c00u) return 0x7bffu;
    const unsigned scale = (nearest == 0 || h2f_bits(nearest) < q) ? nearest + 1 : nearest;
    return scale < 0x7bffu ? scale : 0x7bffu;
}

// Writes the 32 codes (lane i holds code i, a `bits`-bit two's complement
// number) in bit planes: bits 0 to 3, then bits 4 and 5, then the top bit.
// `out` is 4 * bits bytes at any 1-byte alignment.
static __device__ __forceinline__ void pack_codes(int code, int bits, uint8_t* out, int lane) {
    const unsigned u = (unsigned)code;
    if (bits == 8) {
        out[lane] = (uint8_t)u;
        return;
    }
    {
        const unsigned f = u & 0xfu;
        const unsigned lo = __shfl_sync(0xffffffffu, f, (lane * 2) & 31);
        const unsigned hi = __shfl_sync(0xffffffffu, f, (lane * 2 + 1) & 31);
        if (lane < 16) out[lane] = (uint8_t)(lo | (hi << 4));
    }
    if (bits >= 6) {
        const unsigned f = (u >> 4) & 3u;
        const unsigned a = __shfl_sync(0xffffffffu, f, (lane * 4) & 31);
        const unsigned b = __shfl_sync(0xffffffffu, f, (lane * 4 + 1) & 31);
        const unsigned c = __shfl_sync(0xffffffffu, f, (lane * 4 + 2) & 31);
        const unsigned d = __shfl_sync(0xffffffffu, f, (lane * 4 + 3) & 31);
        if (lane < 8) out[16 + lane] = (uint8_t)(a | (b << 2) | (c << 4) | (d << 6));
        if (bits == 7) {
            const unsigned w = __ballot_sync(0xffffffffu, (u >> 6) & 1u);
            if (lane < 4) out[24 + lane] = (uint8_t)(w >> (8 * lane));
        }
    } else if (bits == 5) {
        const unsigned w = __ballot_sync(0xffffffffu, (u >> 4) & 1u);
        if (lane < 4) out[16 + lane] = (uint8_t)(w >> (8 * lane));
    }
}

// Encodes the rotated (and, for a tile, mean-subtracted) group `r` (lane i
// holds r[i]) into `out`, 2 + 4 * bits bytes: steps 3 and 4 of the
// specification, the rule of q8 or the search of q7 to q4.
static __device__ void encode_group(float r, int bits, uint8_t* out, int lane) {
    const float amax = warp_max32(fabsf(r));
    unsigned scale = 0;
    int code = 0;
    if (amax == 0.0f) {
        // scale +0, codes 0
    } else if (bits == 8) {
        scale = scale_bits(amax, 8);
        const float c = rintf(__fdiv_rn(r, h2f_bits(scale)));
        code = (int)fminf(fmaxf(c, -127.0f), 127.0f);
    } else {
        const float qmax = (float)((1 << (bits - 1)) - 1);
        const float pmax = warp_max32(fmaxf(r, 0.0f));
        const float nmax = warp_max32(fmaxf(-r, 0.0f));
        const float sign = (nmax >= pmax) ? 1.0f : -1.0f;
        const unsigned sign_bit = (sign < 0.0f) ? 0x8000u : 0u;
        const int steps = 1 << (8 - bits);
        float best_error = 0.0f;
        unsigned best_scale = 0;
        int best_code = 0;
        for (int j = 0; j <= steps; ++j) {
            const float candidate = __fmul_rn(amax, (float)(64 - j) / 64.0f);
            const unsigned base = scale_bits(candidate, bits);
            const float grid = sign * h2f_bits(base);
            float c = rintf(__fdiv_rn(r, grid));
            c = fminf(fmaxf(c, -qmax - 1.0f), qmax);
            const int ci = (int)c;
            const float product = __fmul_rn(r, c);
            const int squares = __reduce_add_sync(0xffffffffu, ci * ci);
            unsigned s_bits;
            if (squares == 0) {
                s_bits = base;
            } else {
                const float p = sum32(product);
                const float fitted = fminf(__fdiv_rn(fabsf(p), (float)squares), 65504.0f);
                unsigned fh = f2h_bits(fitted);
                fh = fh < 1u ? 1u : fh;
                s_bits = fh | sign_bit;
            }
            const float d = __fsub_rn(r, __fmul_rn((float)ci, h2f_bits(s_bits)));
            const float error = sum32(__fmul_rn(d, d));
            if (j == 0 || error < best_error) {
                best_error = error;
                best_scale = s_bits;
                best_code = ci;
            }
        }
        scale = best_scale;
        code = best_code;
    }
    if (lane == 0) {
        out[0] = (uint8_t)(scale & 0xffu);
        out[1] = (uint8_t)(scale >> 8);
    }
    pack_codes(code, bits, out + 2, lane);
}

// One f32 value of a row to its f16 pattern, saturating at +-65504.
static __device__ __forceinline__ unsigned to_f16_saturating(float v) {
    return f2h_bits(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

// Encodes one group of 32 values (lane i holds value i, unrotated) in `fmt`.
// Flags a non-finite input or an overflowing rotation in `status`.
static __device__ __forceinline__ void encode_group_of(
    int fmt, float v, uint8_t* out, int lane, uint32_t* status) {
    if (!isfinite(v)) atomicOr(status, ATTN_STATUS_NONFINITE);
    if (fmt == FMT_F16) {
        *reinterpret_cast<unsigned short*>(out + 2 * lane) = (unsigned short)to_f16_saturating(v);
        return;
    }
    const float r = rotate32(v, lane);
    if (!isfinite(r) && isfinite(v)) atomicOr(status, ATTN_STATUS_OVERFLOW);
    encode_group(r, fmt_bits(fmt), out, lane);
}
