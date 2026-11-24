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
#include "layer_act_f16.cuh"
#include "layer_merge.cuh"

// The delta rule for the head width of both architectures (128): two lanes per
// row of the matrix (64 values each in registers), 64 rows per block, 16 rows
// of the segment staged at a time. Of the shapes measured (1 to 8 lanes per
// row, 16 to 128 rows per block, 8 or 16 rows staged; examples/layers_bench.rs
// before 2026-11-24) the best for prompt rows on both development GPUs, equal
// for decode rows. src/layers.rs launches it with these numbers.
ARES_DELTA_KERNELS(d128, 128, 2, 64, 16)

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
