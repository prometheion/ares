// The layer kernels of the engine: everything a model needs between the matrix
// products and attention. One fat binary (build.rs), loaded next to the others
// by `LayerKernels` (src/layers.rs).

#include "common.cuh"
#include "layer_step.cuh"
#include "layer_math.cuh"
#include "layer_embed.cuh"
#include "layer_norm.cuh"
#include "layer_conv.cuh"
#include "layer_delta.cuh"
#include "layer_delta_fork.cuh"
#include "layer_act_f16.cuh"
#include "layer_merge.cuh"

// The delta rule for the head width of both architectures (128). The default
// shape: two lanes per row of the matrix (64 values each in registers), 64 rows
// per block, 16 rows of the segment staged at a time. Measured best of the
// shapes below for prompt rows on both development GPUs (see
// examples/layers_bench.rs), equal for decode rows.
ARES_DELTA_KERNELS(d128, 128, 2, 64, 16)

// Other shapes of the same kernel, selectable by name for tuning (src/layers.rs).
ARES_DELTA_KERNELS(lpr4_rb64, 128, 4, 64, 16)
ARES_DELTA_KERNELS(lpr8_rb64, 128, 8, 64, 16)
ARES_DELTA_KERNELS(lpr4_rb32, 128, 4, 32, 16)
ARES_DELTA_KERNELS(lpr4_rb128, 128, 4, 128, 16)
ARES_DELTA_KERNELS(lpr2_rb32, 128, 2, 32, 16)
ARES_DELTA_KERNELS(lpr8_rb32, 128, 8, 32, 16)
ARES_DELTA_KERNELS(lpr1_rb64, 128, 1, 64, 16)
ARES_DELTA_KERNELS(lpr2_rb64_tc8, 128, 2, 64, 8)
ARES_DELTA_KERNELS(lpr2_rb16, 128, 2, 16, 16)

// Rows of a result that was computed as some columns of a wider one, put in
// their place: dst[r][at + i] = src[r][i] for r < n_rows, i < width, with
// rows of `dst_width` values in `dst`. One thread per value. (A matrix that
// lies on the device in two parts is multiplied part by part; with one row
// each part writes its columns itself and nothing is placed.)
ARES_KERNEL ares_place_rows(const float* src, float* dst, uint32_t n_rows, uint32_t width, uint32_t dst_width,
                            uint32_t at) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(n_rows) * width) {
        return;
    }
    const size_t r = i / width;
    dst[r * dst_width + at + (i - r * width)] = src[i];
}
