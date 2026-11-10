// Writing K and V into the cache: appending the rows of a step, and converting
// an exact entry (128 cells as f16) into the formats of the policy. Both produce
// the bytes of `encode_tile` (crates/quant/src/kv.rs), see attn_codec.cuh.
#pragma once

#include "attn_layout.cuh"

// Encodes one row of D f32 values (lane-wise: lane i of each group) in `fmt`.
static __device__ __forceinline__ void encode_row_f32(
    int fmt, const float* src, uint8_t* dst, int D, int lane, uint32_t* status) {
    const int gb = fmt_group_bytes(fmt);
    for (int g = 0; g < D / 32; ++g) {
        encode_group_of(fmt, src[g * 32 + lane], dst + g * gb, lane, status);
    }
}

// Appends the K and V rows of a step. Grid (n_rows, H), block 64: warp 0 writes
// K, warp 1 V. A row goes to the cell `pos` of its sequence: into an exact entry
// as f16, or, for a policy without a mean (f16 and q8, no exact windows), straight
// into the format in a converted entry. Draft rows are ordinary rows here; what is
// pending is decided by the table the host builds (their entries are exact).
//
// k_in, v_in: [row][kv_total][D] f32, the row stride being kv_total * D.
ARES_KERNEL ares_attn_append(
    const AttnRow* rows, const float* k_in, const float* v_in, const uint32_t* tables,
    uint8_t* exact_pool, uint8_t* conv_pool, uint64_t cfg, uint32_t* status) {
    const AttnGeo g = attn_geo(cfg);
    const int row = blockIdx.x, h = blockIdx.y;
    const int side = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const AttnRow ri = rows[row];
    const uint32_t ref = tables[ri.table + ri.pos / ATTN_TILE_CELLS];
    const int cell = (int)(ri.pos % ATTN_TILE_CELLS);
    const float* src = (side == 0 ? k_in : v_in) + ((size_t)row * g.kv_total + g.head0 + h) * g.D;
    const size_t slot = ref & ~ATTN_CONVERTED_BIT;
    if (ref & ATTN_CONVERTED_BIT) {
        if (fmt_has_mean(g.kf) || fmt_has_mean(g.vf)) {
            if (threadIdx.x == 0) atomicOr(status, ATTN_STATUS_DIRECT_MEAN);
            return;
        }
        uint8_t* head = conv_pool + slot * g.entry_c + (size_t)h * g.head_c;
        const int fmt = side == 0 ? g.kf : g.vf;
        const int rb = side == 0 ? g.rbk : g.rbv;
        uint8_t* dst = head + (side == 0 ? g.k_rows_off : g.v_rows_off) + (size_t)cell * rb;
        encode_row_f32(fmt, src, dst, g.D, lane, status);
    } else {
        uint8_t* dst = exact_pool + slot * g.entry_e + (size_t)h * g.head_e +
                       (size_t)side * ATTN_TILE_CELLS * 2 * g.D + (size_t)cell * 2 * g.D;
        encode_row_f32(FMT_F16, src, dst, g.D, lane, status);
    }
}

// Converts the `rows` rows of one side of one tile into `fmt`, with the mean of
// the rotated rows for q7 to q4. `Src::at(row, col)` gives the stored value.
// Block of 256 threads (8 warps), all of them call this.
template <class Src>
static __device__ void convert_side(
    const Src& src, int rows, int D, int fmt, int use_mean, uint8_t* mean_out, uint8_t* rows_out,
    float* mean_s, uint32_t* status) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
    const int groups = D / 32, gb = fmt_group_bytes(fmt);
    const int rb = groups * gb;
    const bool mean = fmt_has_mean(fmt) && use_mean;
    if (mean) {
        for (int g = warp; g < groups; g += nwarps) {
            float sum = 0.0f;
            for (int i = 0; i < rows; ++i) {
                const float v = src.at(i, g * 32 + lane);
                const float r = rotate32(v, lane);
                sum = __fadd_rn(sum, r);
            }
            const float m = fminf(fmaxf(__fdiv_rn(sum, (float)rows), -65504.0f), 65504.0f);
            const unsigned bits = f2h_bits(m);
            *reinterpret_cast<unsigned short*>(mean_out + 2 * (g * 32 + lane)) = (unsigned short)bits;
            mean_s[g * 32 + lane] = h2f_bits(bits);
        }
        __syncthreads();
    }
    for (int i = warp; i < rows; i += nwarps) {
        for (int g = 0; g < groups; ++g) {
            const float v = src.at(i, g * 32 + lane);
            uint8_t* dst = rows_out + (size_t)i * rb + g * gb;
            if (fmt == FMT_F16) {
                *reinterpret_cast<unsigned short*>(dst + 2 * lane) = (unsigned short)to_f16_saturating(v);
                continue;
            }
            float r = rotate32(v, lane);
            if (mean) r = __fsub_rn(r, mean_s[g * 32 + lane]);
            encode_group(r, fmt_bits(fmt), dst, lane);
        }
    }
}

struct ExactSrc {
    const unsigned short* base;  // 128 rows of D halves
    int D;
    __device__ float at(int row, int col) const {
        return __half2float(__ushort_as_half(base[(size_t)row * D + col]));
    }
};

struct F32Src {
    const float* base;
    int D;
    __device__ float at(int row, int col) const { return base[(size_t)row * D + col]; }
};

// Converts exact entries into converted ones. jobs: pairs (exact slot, converted
// slot). Grid (n_jobs, H, 2): z = 0 is K, 1 is V. Block 256.
ARES_KERNEL ares_attn_convert(
    const uint32_t* jobs, const uint8_t* exact_pool, uint8_t* conv_pool, uint64_t cfg,
    uint32_t* status) {
    extern __shared__ float mean_s[];
    const AttnGeo g = attn_geo(cfg);
    const size_t from = jobs[2 * blockIdx.x], to = jobs[2 * blockIdx.x + 1];
    const int h = blockIdx.y, side = blockIdx.z;
    const uint8_t* e = exact_pool + from * g.entry_e + (size_t)h * g.head_e +
                       (size_t)side * ATTN_TILE_CELLS * 2 * g.D;
    uint8_t* head = conv_pool + to * g.entry_c + (size_t)h * g.head_c;
    const int fmt = side == 0 ? g.kf : g.vf;
    ExactSrc src{reinterpret_cast<const unsigned short*>(e), g.D};
    convert_side(
        src, ATTN_TILE_CELLS, g.D, fmt, 1, head + (side == 0 ? 0 : g.v_mean_off),
        head + (side == 0 ? g.k_rows_off : g.v_rows_off), mean_s, status);
}

// The codec on f32 input, for tests: encodes `rows` rows of D values as one tile
// (with the mean of the format) or, with use_mean = 0, each row on its own
// (`encode_row`). Grid 1, block 256, D * 4 bytes of shared memory.
ARES_KERNEL ares_attn_encode_tile_f32(
    const float* src, uint32_t rows, uint32_t D, uint32_t fmt, uint32_t use_mean,
    uint8_t* mean_out, uint8_t* rows_out, uint32_t* status) {
    extern __shared__ float mean_s[];
    F32Src s{src, (int)D};
    convert_side(s, (int)rows, (int)D, (int)fmt, (int)use_mean, mean_out, rows_out, mean_s, status);
}
