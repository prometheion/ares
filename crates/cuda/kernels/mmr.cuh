// The prefill weight kernels that read a rearranged copy of the weights and decode
// them in registers (Marlin's way, vLLM csrc/quantization/marlin; see mm.cuh for
// the kernels that read the file's layout and decode to a tile in shared memory).
//
// Why: the file's layout puts the 144 bytes of one block of one row 2880 bytes
// from the next row's, so a step of 64 values asks the memory system for 128
// different lines per thread block. That, not arithmetic, was what bound the tile
// kernels (an ablation with every row reading the same address was 35 % faster):
// the load/store unit is shared with `ldmatrix`, and 16 lines per warp-wide load
// keep it busy. Here a warp reads 512 contiguous bytes with one load per step and
// holds exactly the weights its `mma` instructions need, so nothing is staged:
// the decoded weights go from registers to the tensor cores, and only the
// activations pass through shared memory.
//
// The rearranged form of a Q4_K matrix (same number of bytes as the file's, with
// the rows rounded up to a multiple of 16):
//   codes   for each tile of 16 rows and each step of 64 values of K: 32 lanes x 16
//           bytes. Lane (g = lane / 4, t = lane % 4) holds four 32-bit words, one
//           per slab of 16 values (kk); nibble i (0..3) of a word is the code of the
//           even and nibble i + 4 that of the odd value of the pair that register
//           a_i of the `mma.m16n8k16` A fragment holds:
//             a0: row g, values 2t, 2t+1        a1: row g+8, values 2t, 2t+1
//             a2: row g, values 2t+8, 2t+9      a3: row g+8, values 2t+8, 2t+9
//           so that `lop3(w >> 4i, 0x000F000F, 0x64006400)` is the f16 pair
//           (1024 + even, 1024 + odd) of register a_i.
//   headers for each tile of 16 rows and super-block of 256 values: the 16 rows'
//           16-byte headers (d, dmin, the 12 bytes of 6-bit scales) as in the file.
// The codes come first (n_out16 * n_in / 2 bytes), then the headers.
#pragma once

#include <type_traits>

#include "mm_types.cuh"

// Bytes of the rearranged form: the same as the file's, with n_out rounded up to a
// multiple of 16 rows.
static __host__ __device__ __forceinline__ size_t ares_mmr_q4k_codes_bytes(uint32_t n_out, uint32_t n_in) {
    return static_cast<size_t>((n_out + 15) / 16) * 16 * n_in / 2;
}

// One thread per (tile of 16 rows, step of 64 values, lane) for the codes, then one
// per (row, super-block) for the headers. The file's block of a row is 144 bytes:
// header (16), then four groups of 32 bytes whose low nibbles are the 32 values of
// sub-block 2j and the high nibbles those of sub-block 2j + 1.
ARES_KERNEL ares_repack_q4_K(const uint8_t* src, uint8_t* dst, uint32_t n_out, uint32_t n_in) {
    const uint32_t n_tiles = (n_out + 15) / 16, n_steps = n_in / 64, n_sb = n_in / 256;
    const size_t row_bytes = static_cast<size_t>(n_sb) * 144;
    const size_t n_codes = static_cast<size_t>(n_tiles) * n_steps * 32;
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < n_codes) {
        const uint32_t lane = id % 32;
        const size_t unit = id / 32;
        const uint32_t ks = unit % n_steps, mt = unit / n_steps;
        const uint32_t g = lane / 4, t = lane % 4;
        const uint32_t sb = ks / 4, j = ks % 4;
        uint32_t word[4];
        for (int kk = 0; kk < 4; kk++) {
            uint32_t w = 0;
            for (int i = 0; i < 4; i++) {
                const uint32_t row = mt * 16 + g + 8 * (i & 1);
                uint32_t q[2] = {0, 0};
                if (row < n_out) {
                    const uint8_t* blk = src + row * row_bytes + static_cast<size_t>(sb) * 144;
                    for (int e = 0; e < 2; e++) {
                        const uint32_t kp = kk * 16 + 2 * t + 8 * (i >> 1) + e;  // 0..63 within the step
                        const uint8_t byte = blk[16 + 32 * j + (kp & 31)];
                        q[e] = (kp >= 32) ? (byte >> 4) : (byte & 0xF);
                    }
                }
                w |= (q[0] << (4 * i)) | (q[1] << (4 * i + 16));
            }
            word[kk] = w;
        }
        *reinterpret_cast<uint4*>(dst + (unit * 32 + lane) * 16) = make_uint4(word[0], word[1], word[2], word[3]);
    } else {
        const size_t h = id - n_codes;  // (tile, super-block, row of the tile)
        if (h < static_cast<size_t>(n_tiles) * n_sb * 16) {
            const uint32_t r16 = h % 16;
            const size_t ts = h / 16;
            const uint32_t sb = ts % n_sb, mt = ts / n_sb;
            const uint32_t row = mt * 16 + r16;
            uint4 hdr = make_uint4(0, 0, 0, 0);
            if (row < n_out) {
                hdr = *reinterpret_cast<const uint4*>(src + row * row_bytes + static_cast<size_t>(sb) * 144);
            }
            *reinterpret_cast<uint4*>(dst + ares_mmr_q4k_codes_bytes(n_out, n_in) + h * 16) = hdr;
        }
    }
}

// The grid of the repack kernel: threads to launch.
static __host__ __device__ __forceinline__ size_t ares_repack_q4_K_threads(uint32_t n_out, uint32_t n_in) {
    const size_t n_tiles = (n_out + 15) / 16;
    return n_tiles * (n_in / 64) * 32 + n_tiles * (n_in / 256) * 16;
}

// Dynamic shared memory of ares_mmr_q4_K: STAGES tiles of J activation rows.
template <int J, int STAGES>
struct AresMmrLayout {
    static constexpr int PITCH = 64 * 2 + 16;
    static constexpr uint32_t BYTES = STAGES * J * PITCH;
};

// y[b][r] = sum over k of W[r][k] * x[b][k], W in the rearranged form of Q4_K, x
// the f16 activations of ares_convert_f16 (row-scaled; the scale is put back at
// the end). I x J outputs per block, WM x WN warps; a warp computes 16 rows x
// J / WN columns, so the A fragment it decodes serves J / WN / 8 mma per slab.
// FLUSH as in ares_mm_f16 (0: f32 accumulation).
template <int I, int J, int WM, int WN, int FLUSH, int STAGES = 2>
static __device__ __forceinline__ void ares_mmr_q4_K(
    const uint8_t* __restrict__ w, const __half* __restrict__ ax, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch, uint32_t group) {
    constexpr int NT = WM * WN * 32;
    constexpr int TN = J / WN, NTL = TN / 8;
    constexpr int PITCH = AresMmrLayout<J, STAGES>::PITCH;
    constexpr int PB = 8;  // pieces of 16 bytes per activation row and step (64 f16 values)
    constexpr int BI = (J * PB + NT - 1) / NT;
    static_assert(I == 16 * WM && J % (16 * WN) == 0, "a warp is 16 rows");
    static_assert((4 % (FLUSH == 0 ? 1 : FLUSH)) == 0, "flush ends with a step");
    ares_check_smem(AresMmrLayout<J, STAGES>::BYTES);

    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t wm = warp / WN, wn = warp % WN;
    const uint32_t g = lane >> 2, t = lane & 3;
    uint32_t wt, bt;
    ares_mm_tile((n_out + I - 1) / I, (n_batch + J - 1) / J, group, wt, bt);
    const uint32_t col0 = bt * J;
    const uint32_t n_tiles = (n_out + 15) / 16, n_steps = n_in / 64, n_sb = n_in / 256;
    const uint32_t mt = min(wt * WM + wm, n_tiles - 1);  // this warp's tile of 16 rows
    const uint32_t row0 = (wt * WM + wm) * 16;

    const uint4* __restrict__ codes = reinterpret_cast<const uint4*>(w) + (static_cast<size_t>(mt) * n_steps) * 32 + lane;
    const uint4* __restrict__ hdrs =
        reinterpret_cast<const uint4*>(w + ares_mmr_q4k_codes_bytes(n_out, n_in)) + (static_cast<size_t>(mt) * n_sb) * 16 + g;

    uint8_t* const bs = ares_dyn_smem;  // stage s: bs + s * J * PITCH
    const __half* b_src[BI];
    uint32_t b_dst[BI];
#pragma unroll
    for (int u = 0; u < BI; u++) {
        const uint32_t item = tid + u * NT;
        const uint32_t c = item / PB, s = item % PB;
        b_src[u] = ax + static_cast<size_t>(min(col0 + c, n_batch - 1)) * n_in + 8 * s;
        b_dst[u] = c * PITCH + 16 * s;
    }
    auto load_acts = [&](uint32_t step, int stage) {
#pragma unroll
        for (int u = 0; u < BI; u++) {
            if (tid + u * NT < J * PB) {
                ares_cp_async16(bs + stage * J * PITCH + b_dst[u], b_src[u] + 64 * step, 16u);
            }
        }
    };

    float acc[NTL][4];
    uint32_t hacc[NTL][2];
#pragma unroll
    for (int ni = 0; ni < NTL; ni++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[ni][e] = 0.0f;
        }
    }

    uint4 cw[2];     // codes of this and the next step
    uint4 hd_next[2];  // headers (rows g and g + 8) of the next super-block
    uint32_t sc[2][2], mn[2][2];  // scales and minimums as bytes, rows g and g + 8
    uint32_t dd2[2];              // (d, -dmin) as f16x2, rows g and g + 8

#pragma unroll
    for (int p = 0; p < STAGES - 1; p++) {
        if (p < n_steps) {
            load_acts(p, p);
        }
        ares_cp_commit();
    }
    cw[0] = __ldg(codes);
    hd_next[0] = __ldg(hdrs);
    hd_next[1] = __ldg(hdrs + 8);

    const uint32_t kBias = 0x64006400u, kMagic = 0x64646464u;
    for (uint32_t sb = 0; sb < n_sb; sb++) {
        // The headers of this super-block: expand the 6-bit scales to bytes.
#pragma unroll
        for (int h = 0; h < 2; h++) {
            ares_mm_scales_k4(hd_next[h], sc[h], mn[h]);
            dd2[h] = (hd_next[h].x & 0xFFFFu) | ((hd_next[h].x & 0xFFFF0000u) ^ 0x80000000u);
        }
        if (sb + 1 < n_sb) {
            hd_next[0] = __ldg(hdrs + (sb + 1) * 16);
            hd_next[1] = __ldg(hdrs + (sb + 1) * 16 + 8);
        }
        auto step_fn = [&](auto jc) {
            constexpr int j = decltype(jc)::value;
            const uint32_t step = sb * 4 + j;
            const int stage = step % STAGES;
            ares_cp_wait_group<STAGES - 2>();
            __syncthreads();
            if (step + STAGES - 1 < n_steps) {
                load_acts(step + STAGES - 1, (step + STAGES - 1) % STAGES);
            }
            ares_cp_commit();
            if (step + 1 < n_steps) {
                cw[(j + 1) & 1] = __ldg(codes + (static_cast<size_t>(step) + 1) * 32);
            }

            // The scales of the two sub-blocks (2j, 2j+1) of rows g and g+8, as f16 pairs
            // (sw, sw) and (-mw, -mw).
            uint32_t sw2[2][2], nmw2[2][2];  // [row][sub-block]
#pragma unroll
            for (int h = 0; h < 2; h++) {
                constexpr int o = 2 * (j & 1);
                const uint32_t pair = __byte_perm(sc[h][j >> 1], mn[h][j >> 1],
                                                  ((4 + o + 1) << 12) | ((o + 1) << 8) | ((4 + o) << 4) | o);
#pragma unroll
                for (int sub = 0; sub < 2; sub++) {
                    const uint32_t x = __byte_perm(pair, kMagic, sub == 0 ? 0x4140 : 0x4342);
                    const uint32_t v = ares_h2_mul(ares_h2_sub(x, kBias), dd2[h]);
                    sw2[h][sub] = __byte_perm(v, v, 0x1010);
                    nmw2[h][sub] = __byte_perm(v, v, 0x3232);
                }
            }

            const uint4 c4 = cw[j & 1];
            const uint32_t words[4] = {c4.x, c4.y, c4.z, c4.w};
            const uint8_t* const b_base = bs + stage * (J * PITCH);
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                // Decode the A fragment: a0, a2 are row g; a1, a3 row g + 8.
                const int sub = kk >> 1;
                uint32_t a[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const uint32_t pair_codes = ((words[kk] >> (4 * i)) & 0x000F000Fu) | kBias;
                    a[i] = ares_h2_fma(ares_h2_sub(pair_codes, kBias), sw2[i & 1][sub], nmw2[i & 1][sub]);
                }
                uint32_t b[NTL][2];
#pragma unroll
                for (int nj = 0; nj < NTL / 2; nj++) {
                    uint32_t r4[4];
                    ares_ldmatrix_x4(r4, b_base + (wn * TN + nj * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * PITCH + kk * 32 + ((lane >> 3) & 1) * 16);
                    b[2 * nj][0] = r4[0];
                    b[2 * nj][1] = r4[1];
                    b[2 * nj + 1][0] = r4[2];
                    b[2 * nj + 1][1] = r4[3];
                }
#pragma unroll
                for (int ni = 0; ni < NTL; ni++) {
                    if constexpr (FLUSH == 0) {
                        ares_mma_f16_k16(acc[ni], a, b[ni][0], b[ni][1]);
                    } else {
                        if (kk % FLUSH == 0) {
                            ares_mma_f16acc_k16<false>(hacc[ni], a, b[ni][0], b[ni][1]);
                        } else {
                            ares_mma_f16acc_k16<true>(hacc[ni], a, b[ni][0], b[ni][1]);
                        }
                        if (kk % FLUSH == FLUSH - 1) {
                            ares_flush_f16(acc[ni], hacc[ni]);
                        }
                    }
                }
            }
        };
        step_fn(std::integral_constant<int, 0>{});
        step_fn(std::integral_constant<int, 1>{});
        step_fn(std::integral_constant<int, 2>{});
        step_fn(std::integral_constant<int, 3>{});
    }

    // The rows were divided by their scale (ares_convert_f16): put it back.
    const float* const row_scale =
        reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(ax) + ares_act_ds_offset(static_cast<size_t>(n_batch) * n_in * 2));
#pragma unroll
    for (int ni = 0; ni < NTL; ni++) {
        const uint32_t c0 = col0 + wn * TN + ni * 8 + 2 * t;
        const float s0 = c0 < n_batch ? row_scale[c0] : 0.0f;
        const float s1 = c0 + 1 < n_batch ? row_scale[c0 + 1] : 0.0f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const uint32_t r = row0 + g + (e >> 1) * 8;
            const uint32_t c = c0 + (e & 1);
            if (r < n_out && c < n_batch) {
                y[static_cast<size_t>(c) * n_out + r] = acc[ni][e] * ((e & 1) ? s1 : s0);
            }
        }
    }
}

#define ARES_MMR_KERNEL(NAME, TYPE, I, J, WM, WN, FLUSH, ...)                                               \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, 1) NAME(                                      \
        const uint8_t* w, const __half* act, float* y, uint32_t n_in, uint32_t n_out, uint32_t n_batch,     \
        uint32_t group) {                                                                                     \
        ares_mmr_##TYPE<I, J, WM, WN, FLUSH, ##__VA_ARGS__>(w, act, y, n_in, n_out, n_batch, group);                       \
    }
