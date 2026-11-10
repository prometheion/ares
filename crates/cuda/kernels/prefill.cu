// The prefill module: the tensor-core weight kernels (mm.cuh) and what they
// need. It is a module of its own, compiled only for compute capability 8.0 and
// newer (see build.rs): the integer and f16 `mma` shapes and `cp.async` used
// here start with Ampere, while the decode kernels in module.cu run on every
// GPU the toolkit builds for.

#include "common.cuh"
#include "mm_types.cuh"
#include "mmr.cuh"
#include "mmr_codes.cuh"

// The shapes: weight rows x activation rows per block, warps, blocks per SM.
// Found by measurement on an RTX 4070 and an RTX 3060 (see mm.cuh); the Rust side
// (src/kernels.rs) picks one by the number of activation rows and lists the same
// sizes, including the shared memory each needs.
//
//   large   128 x 128   4 x 4 warps (512 threads), 1 block per SM    more than 64 rows
//   medium   64 x  64   2 x 2 warps (128 threads), 3 (int8) or 2 (f16) blocks per SM   33 to 64
//   small    64 x  32   2 x 2 warps (128 threads), 3 blocks per SM    up to 32 rows
#define ARES_MM_I8_LARGE(NAME, T) ARES_MM_KERNEL(NAME, 512, 1, ares_mm_i8, T, 128, 128, 4, 4, 64, int8_t)
#define ARES_MM_I8_MEDIUM(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_i8, T, 64, 64, 2, 2, 64, int8_t)
#define ARES_MM_I8_SMALL(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_i8, T, 64, 32, 2, 2, 64, int8_t)
#define ARES_MM_F16_LARGE(NAME, T) ARES_MM_KERNEL(NAME, 512, 1, ares_mm_f16, T, 128, 128, 4, 4, 64, __half)
#define ARES_MM_F16_MEDIUM(NAME, T) ARES_MM_KERNEL(NAME, 128, 2, ares_mm_f16, T, 64, 64, 2, 2, 64, __half)
#define ARES_MM_F16_SMALL(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_f16, T, 64, 32, 2, 2, 64, __half)

// One kernel per type, variant and shape, named after the type as ggml spells it:
// ares_mm_<i8|f16><|m|s>_<type>. The Rust side lists the same names.
ARES_MM_I8_LARGE(ares_mm_i8_q4_K, EngMmQ4K)
ARES_MM_I8_LARGE(ares_mm_i8_q5_K, EngMmQ5K)
ARES_MM_I8_LARGE(ares_mm_i8_q6_K, EngMmQ6K)
ARES_MM_I8_LARGE(ares_mm_i8_q3_K, EngMmQ3K)
ARES_MM_I8_LARGE(ares_mm_i8_iq4_xs, EngMmIQ4XS)
ARES_MM_I8_LARGE(ares_mm_i8_iq4_nl, EngMmIQ4NL)
ARES_MM_I8_LARGE(ares_mm_i8_iq3_s, EngMmIQ3S)
ARES_MM_I8_LARGE(ares_mm_i8_iq3_xxs, EngMmIQ3XXS)
ARES_MM_I8_LARGE(ares_mm_i8_iq2_s, EngMmIQ2S)
ARES_MM_I8_LARGE(ares_mm_i8_iq2_xs, EngMmIQ2XS)
ARES_MM_I8_LARGE(ares_mm_i8_q8_0, EngMmQ80)

ARES_MM_I8_MEDIUM(ares_mm_i8m_q4_K, EngMmQ4K)
ARES_MM_I8_MEDIUM(ares_mm_i8m_q5_K, EngMmQ5K)
ARES_MM_I8_MEDIUM(ares_mm_i8m_q6_K, EngMmQ6K)
ARES_MM_I8_MEDIUM(ares_mm_i8m_q3_K, EngMmQ3K)
ARES_MM_I8_MEDIUM(ares_mm_i8m_iq4_xs, EngMmIQ4XS)
ARES_MM_I8_MEDIUM(ares_mm_i8m_iq4_nl, EngMmIQ4NL)
ARES_MM_I8_MEDIUM(ares_mm_i8m_iq3_s, EngMmIQ3S)
ARES_MM_I8_MEDIUM(ares_mm_i8m_iq3_xxs, EngMmIQ3XXS)
ARES_MM_I8_MEDIUM(ares_mm_i8m_iq2_s, EngMmIQ2S)
ARES_MM_I8_MEDIUM(ares_mm_i8m_iq2_xs, EngMmIQ2XS)
ARES_MM_I8_MEDIUM(ares_mm_i8m_q8_0, EngMmQ80)

ARES_MM_I8_SMALL(ares_mm_i8s_q4_K, EngMmQ4K)
ARES_MM_I8_SMALL(ares_mm_i8s_q5_K, EngMmQ5K)
ARES_MM_I8_SMALL(ares_mm_i8s_q6_K, EngMmQ6K)
ARES_MM_I8_SMALL(ares_mm_i8s_q3_K, EngMmQ3K)
ARES_MM_I8_SMALL(ares_mm_i8s_iq4_xs, EngMmIQ4XS)
ARES_MM_I8_SMALL(ares_mm_i8s_iq4_nl, EngMmIQ4NL)
ARES_MM_I8_SMALL(ares_mm_i8s_iq3_s, EngMmIQ3S)
ARES_MM_I8_SMALL(ares_mm_i8s_iq3_xxs, EngMmIQ3XXS)
ARES_MM_I8_SMALL(ares_mm_i8s_iq2_s, EngMmIQ2S)
ARES_MM_I8_SMALL(ares_mm_i8s_iq2_xs, EngMmIQ2XS)
ARES_MM_I8_SMALL(ares_mm_i8s_q8_0, EngMmQ80)

ARES_MM_F16_LARGE(ares_mm_f16_q4_K, EngMmQ4K)
ARES_MM_F16_LARGE(ares_mm_f16_q5_K, EngMmQ5K)
ARES_MM_F16_LARGE(ares_mm_f16_q6_K, EngMmQ6K)
ARES_MM_F16_LARGE(ares_mm_f16_q3_K, EngMmQ3K)
ARES_MM_F16_LARGE(ares_mm_f16_iq4_xs, EngMmIQ4XS)
ARES_MM_F16_LARGE(ares_mm_f16_iq4_nl, EngMmIQ4NL)
ARES_MM_F16_LARGE(ares_mm_f16_iq3_s, EngMmIQ3S)
ARES_MM_F16_LARGE(ares_mm_f16_iq3_xxs, EngMmIQ3XXS)
ARES_MM_F16_LARGE(ares_mm_f16_iq2_s, EngMmIQ2S)
ARES_MM_F16_LARGE(ares_mm_f16_iq2_xs, EngMmIQ2XS)
ARES_MM_F16_LARGE(ares_mm_f16_q8_0, EngMmQ80)

ARES_MM_F16_MEDIUM(ares_mm_f16m_q4_K, EngMmQ4K)
ARES_MM_F16_MEDIUM(ares_mm_f16m_q5_K, EngMmQ5K)
ARES_MM_F16_MEDIUM(ares_mm_f16m_q6_K, EngMmQ6K)
ARES_MM_F16_MEDIUM(ares_mm_f16m_q3_K, EngMmQ3K)
ARES_MM_F16_MEDIUM(ares_mm_f16m_iq4_xs, EngMmIQ4XS)
ARES_MM_F16_MEDIUM(ares_mm_f16m_iq4_nl, EngMmIQ4NL)
ARES_MM_F16_MEDIUM(ares_mm_f16m_iq3_s, EngMmIQ3S)
ARES_MM_F16_MEDIUM(ares_mm_f16m_iq3_xxs, EngMmIQ3XXS)
ARES_MM_F16_MEDIUM(ares_mm_f16m_iq2_s, EngMmIQ2S)
ARES_MM_F16_MEDIUM(ares_mm_f16m_iq2_xs, EngMmIQ2XS)
ARES_MM_F16_MEDIUM(ares_mm_f16m_q8_0, EngMmQ80)

ARES_MM_F16_SMALL(ares_mm_f16s_q4_K, EngMmQ4K)
ARES_MM_F16_SMALL(ares_mm_f16s_q5_K, EngMmQ5K)
ARES_MM_F16_SMALL(ares_mm_f16s_q6_K, EngMmQ6K)
ARES_MM_F16_SMALL(ares_mm_f16s_q3_K, EngMmQ3K)
ARES_MM_F16_SMALL(ares_mm_f16s_iq4_xs, EngMmIQ4XS)
ARES_MM_F16_SMALL(ares_mm_f16s_iq4_nl, EngMmIQ4NL)
ARES_MM_F16_SMALL(ares_mm_f16s_iq3_s, EngMmIQ3S)
ARES_MM_F16_SMALL(ares_mm_f16s_iq3_xxs, EngMmIQ3XXS)
ARES_MM_F16_SMALL(ares_mm_f16s_iq2_s, EngMmIQ2S)
ARES_MM_F16_SMALL(ares_mm_f16s_iq2_xs, EngMmIQ2XS)
ARES_MM_F16_SMALL(ares_mm_f16s_q8_0, EngMmQ80)


// The f16 variant with f16 accumulation, flushed to f32 every ARES_MM_FLUSH mma of 16
// values of K (see mm.cuh): ares_mm_h<|m|s>_<type>. The GeForce cards run it at twice
// the rate of the f32 accumulation.
#define ARES_MM_FLUSH 4
#define ARES_MM_H_LARGE(NAME, T) ARES_MM_KERNEL(NAME, 512, 1, ares_mm_f16, T, 128, 128, 4, 4, 64, __half, ARES_MM_FLUSH)
#define ARES_MM_H_MEDIUM(NAME, T) ARES_MM_KERNEL(NAME, 128, 2, ares_mm_f16, T, 64, 64, 2, 2, 64, __half, ARES_MM_FLUSH)
#define ARES_MM_H_SMALL(NAME, T) ARES_MM_KERNEL(NAME, 128, 3, ares_mm_f16, T, 64, 32, 2, 2, 64, __half, ARES_MM_FLUSH)
ARES_MM_H_LARGE(ares_mm_h_q4_K, EngMmQ4K)
ARES_MM_H_LARGE(ares_mm_h_q5_K, EngMmQ5K)
ARES_MM_H_LARGE(ares_mm_h_q6_K, EngMmQ6K)
ARES_MM_H_LARGE(ares_mm_h_q3_K, EngMmQ3K)
ARES_MM_H_LARGE(ares_mm_h_iq4_xs, EngMmIQ4XS)
ARES_MM_H_LARGE(ares_mm_h_iq4_nl, EngMmIQ4NL)
ARES_MM_H_LARGE(ares_mm_h_iq3_s, EngMmIQ3S)
ARES_MM_H_LARGE(ares_mm_h_iq3_xxs, EngMmIQ3XXS)
ARES_MM_H_LARGE(ares_mm_h_iq2_s, EngMmIQ2S)
ARES_MM_H_LARGE(ares_mm_h_iq2_xs, EngMmIQ2XS)
ARES_MM_H_LARGE(ares_mm_h_q8_0, EngMmQ80)

ARES_MM_H_MEDIUM(ares_mm_hm_q4_K, EngMmQ4K)
ARES_MM_H_MEDIUM(ares_mm_hm_q5_K, EngMmQ5K)
ARES_MM_H_MEDIUM(ares_mm_hm_q6_K, EngMmQ6K)
ARES_MM_H_MEDIUM(ares_mm_hm_q3_K, EngMmQ3K)
ARES_MM_H_MEDIUM(ares_mm_hm_iq4_xs, EngMmIQ4XS)
ARES_MM_H_MEDIUM(ares_mm_hm_iq4_nl, EngMmIQ4NL)
ARES_MM_H_MEDIUM(ares_mm_hm_iq3_s, EngMmIQ3S)
ARES_MM_H_MEDIUM(ares_mm_hm_iq3_xxs, EngMmIQ3XXS)
ARES_MM_H_MEDIUM(ares_mm_hm_iq2_s, EngMmIQ2S)
ARES_MM_H_MEDIUM(ares_mm_hm_iq2_xs, EngMmIQ2XS)
ARES_MM_H_MEDIUM(ares_mm_hm_q8_0, EngMmQ80)

ARES_MM_H_SMALL(ares_mm_hs_q4_K, EngMmQ4K)
ARES_MM_H_SMALL(ares_mm_hs_q5_K, EngMmQ5K)
ARES_MM_H_SMALL(ares_mm_hs_q6_K, EngMmQ6K)
ARES_MM_H_SMALL(ares_mm_hs_q3_K, EngMmQ3K)
ARES_MM_H_SMALL(ares_mm_hs_iq4_xs, EngMmIQ4XS)
ARES_MM_H_SMALL(ares_mm_hs_iq4_nl, EngMmIQ4NL)
ARES_MM_H_SMALL(ares_mm_hs_iq3_s, EngMmIQ3S)
ARES_MM_H_SMALL(ares_mm_hs_iq3_xxs, EngMmIQ3XXS)
ARES_MM_H_SMALL(ares_mm_hs_iq2_s, EngMmIQ2S)
ARES_MM_H_SMALL(ares_mm_hs_iq2_xs, EngMmIQ2XS)
ARES_MM_H_SMALL(ares_mm_hs_q8_0, EngMmQ80)

// The kernels over the device layouts (mmr.cuh, mmr_codes.cuh): ares_mmr_<shape>_<type>
// and the rearrangement ares_repack_<type>. The Rust side (src/kernels/layout.rs)
// lists the same types.
ARES_MMR_CODES(q4_K, AresCodesQ4K)
ARES_MMR_CODES(q5_K, AresCodesQ5K)
ARES_MMR_CODES(q6_K, AresCodesQ6K)
ARES_MMR_CODES(q3_K, AresCodesQ3K)
ARES_MMR_CODES(q2_K, AresCodesQ2K)
ARES_MMR_CODES(q4_0, AresCodesQ40)
ARES_MMR_CODES(q4_1, AresCodesQ41)
ARES_MMR_CODES(q5_0, AresCodesQ50)
ARES_MMR_CODES(q5_1, AresCodesQ51)
ARES_MMR_CODES(q8_0, AresCodesQ80)
ARES_MMR_CODES(iq4_xs, AresCodesIQ4XS)
ARES_MMR_CODES(iq4_nl, AresCodesIQ4NL)

// The probe of Kernels::load: how fast the tensor cores run `mma` with f16
// inputs and f32 accumulation against f16 accumulation. GeForce cards run the
// first at half the rate of the second; cards without that cap (professional
// cards, or GeForce cards where the cap is lifted) at the same rate. Each warp
// issues `iters` rounds of 8 independent `mma` with the same operands.
template <bool F32_ACC>
static __device__ __forceinline__ void ares_probe_mma(float* out, uint32_t iters) {
    const uint32_t one = 0x3c003c00u;  // two f16 ones
    float acc32[8][4];
    uint32_t acc16[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            acc32[j][e] = 0.0f;
        }
        acc16[j][0] = acc16[j][1] = 0;
    }
    for (uint32_t i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (F32_ACC) {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
                             : "+f"(acc32[j][0]), "+f"(acc32[j][1]), "+f"(acc32[j][2]), "+f"(acc32[j][3])
                             : "r"(one));
            } else {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%2,%2,%2}, {%2,%2}, {%0,%1};"
                             : "+r"(acc16[j][0]), "+r"(acc16[j][1])
                             : "r"(one));
            }
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        sum += acc32[j][0] + acc32[j][1] + acc32[j][2] + acc32[j][3] + static_cast<float>(acc16[j][0] ^ acc16[j][1]);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

ARES_KERNEL ares_probe_mma_f32(float* out, uint32_t iters) {
    ares_probe_mma<true>(out, iters);
}

ARES_KERNEL ares_probe_mma_f16(float* out, uint32_t iters) {
    ares_probe_mma<false>(out, iters);
}
