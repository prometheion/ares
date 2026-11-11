// The scales of the 4-bit float types, shared by the weight kernels (mmvq_more.cuh)
// and the embedding gather (layer_embed.cuh), so that they cannot disagree about
// a format. Both return the scale halved, because the levels of the E2M1 table
// (`ares_kvalues_mxfp4`) are doubled to make them integers.
#pragma once

#include "common.cuh"

// The scale of an MXFP4 block, 2^(e - 127) / 2, for its E8M0 byte `e`: e = 0 is
// the subnormal 2^-128 and e = 255, which the OCP specification reserves for
// NaN, is taken as the large finite 2^127 (as ggml's `ggml_e8m0_to_fp32_half`
// and the CPU reference do).
static __device__ __forceinline__ float ares_e8m0_half(uint32_t e) {
    return __uint_as_float(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
}

// The scale of an NVFP4 sub-block for its UE4M3 byte `x` (4 exponent bits with
// bias 7, 3 mantissa bits; the top bit is not read): zero and the NaN code 0x7f
// give 0.
static __device__ __forceinline__ float ares_ue4m3_half(uint32_t x) {
    if (x == 0 || x == 0x7f) {
        return 0.0f;
    }
    const uint32_t exp = (x >> 3) & 0xF;
    const float man = static_cast<float>(x & 7);
    const float raw = exp == 0 ? man * __uint_as_float(0x3b000000u)
                               : (1.0f + man * 0.125f) * __uint_as_float((exp + 120) << 23);
    return raw * 0.5f;
}
