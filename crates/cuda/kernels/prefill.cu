// The prefill module: the tensor-core weight kernels (mm.cuh) and what they
// need. It is a module of its own, compiled only for compute capability 8.0 and
// newer (see build.rs): the integer and f16 `mma` shapes and `cp.async` used
// here start with Ampere, while the decode kernels in module.cu run on every
// GPU the toolkit builds for.

#include "common.cuh"
#include "mm_types.cuh"

// One kernel per type and variant, named after the type as ggml spells it. The
// Rust side (src/kernels.rs) lists the same names.
ARES_MM_I8_KERNEL(ares_mm_i8_q4_K, EngMmQ4K)
ARES_MM_I8_KERNEL(ares_mm_i8_q5_K, EngMmQ5K)
ARES_MM_I8_KERNEL(ares_mm_i8_q6_K, EngMmQ6K)
ARES_MM_I8_KERNEL(ares_mm_i8_q3_K, EngMmQ3K)
ARES_MM_I8_KERNEL(ares_mm_i8_iq4_xs, EngMmIQ4XS)
ARES_MM_I8_KERNEL(ares_mm_i8_iq4_nl, EngMmIQ4NL)
ARES_MM_I8_KERNEL(ares_mm_i8_iq3_s, EngMmIQ3S)
ARES_MM_I8_KERNEL(ares_mm_i8_iq3_xxs, EngMmIQ3XXS)
ARES_MM_I8_KERNEL(ares_mm_i8_iq2_s, EngMmIQ2S)
ARES_MM_I8_KERNEL(ares_mm_i8_iq2_xs, EngMmIQ2XS)
ARES_MM_I8_KERNEL(ares_mm_i8_q8_0, EngMmQ80)

ARES_MM_I8S_KERNEL(ares_mm_i8s_q4_K, EngMmQ4K)
ARES_MM_I8S_KERNEL(ares_mm_i8s_q5_K, EngMmQ5K)
ARES_MM_I8S_KERNEL(ares_mm_i8s_q6_K, EngMmQ6K)
ARES_MM_I8S_KERNEL(ares_mm_i8s_q3_K, EngMmQ3K)
ARES_MM_I8S_KERNEL(ares_mm_i8s_iq4_xs, EngMmIQ4XS)
ARES_MM_I8S_KERNEL(ares_mm_i8s_iq4_nl, EngMmIQ4NL)
ARES_MM_I8S_KERNEL(ares_mm_i8s_iq3_s, EngMmIQ3S)
ARES_MM_I8S_KERNEL(ares_mm_i8s_iq3_xxs, EngMmIQ3XXS)
ARES_MM_I8S_KERNEL(ares_mm_i8s_iq2_s, EngMmIQ2S)
ARES_MM_I8S_KERNEL(ares_mm_i8s_iq2_xs, EngMmIQ2XS)
ARES_MM_I8S_KERNEL(ares_mm_i8s_q8_0, EngMmQ80)

ARES_MM_F16_KERNEL(ares_mm_f16_q4_K, EngMmQ4K)
ARES_MM_F16_KERNEL(ares_mm_f16_q5_K, EngMmQ5K)
ARES_MM_F16_KERNEL(ares_mm_f16_q6_K, EngMmQ6K)
ARES_MM_F16_KERNEL(ares_mm_f16_q3_K, EngMmQ3K)
ARES_MM_F16_KERNEL(ares_mm_f16_iq4_xs, EngMmIQ4XS)
ARES_MM_F16_KERNEL(ares_mm_f16_iq4_nl, EngMmIQ4NL)
ARES_MM_F16_KERNEL(ares_mm_f16_iq3_s, EngMmIQ3S)
ARES_MM_F16_KERNEL(ares_mm_f16_iq3_xxs, EngMmIQ3XXS)
ARES_MM_F16_KERNEL(ares_mm_f16_iq2_s, EngMmIQ2S)
ARES_MM_F16_KERNEL(ares_mm_f16_iq2_xs, EngMmIQ2XS)
ARES_MM_F16_KERNEL(ares_mm_f16_q8_0, EngMmQ80)

ARES_MM_F16S_KERNEL(ares_mm_f16s_q4_K, EngMmQ4K)
ARES_MM_F16S_KERNEL(ares_mm_f16s_q5_K, EngMmQ5K)
ARES_MM_F16S_KERNEL(ares_mm_f16s_q6_K, EngMmQ6K)
ARES_MM_F16S_KERNEL(ares_mm_f16s_q3_K, EngMmQ3K)
ARES_MM_F16S_KERNEL(ares_mm_f16s_iq4_xs, EngMmIQ4XS)
ARES_MM_F16S_KERNEL(ares_mm_f16s_iq4_nl, EngMmIQ4NL)
ARES_MM_F16S_KERNEL(ares_mm_f16s_iq3_s, EngMmIQ3S)
ARES_MM_F16S_KERNEL(ares_mm_f16s_iq3_xxs, EngMmIQ3XXS)
ARES_MM_F16S_KERNEL(ares_mm_f16s_iq2_s, EngMmIQ2S)
ARES_MM_F16S_KERNEL(ares_mm_f16s_iq2_xs, EngMmIQ2XS)
ARES_MM_F16S_KERNEL(ares_mm_f16s_q8_0, EngMmQ80)

// out[i] = x[i] as f16 for i < n. The activations of the f16 variant.
ARES_KERNEL ares_convert_f16(const float* x, __half* out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half_rn(x[i]);
    }
}

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
