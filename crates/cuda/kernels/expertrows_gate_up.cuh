// Gate and up of the items of a plan on the tensor cores:
//
//   hidden[pair] = silu(gate_e x) * (up_e x)     for the pairs of every item
//
// A thread block takes one item (an expert and up to 8 NB of its pairs) and 64 of the
// expert's hidden values, a warp 16 of them: 16 rows of the gate matrix and the same 16
// of the up matrix, against all the pairs of the item. The weights are read once for
// all of them, from the address the table gives for the expert (the matrices of an
// expert are `up_offset` apart, see ExpertShape).
//
// - A operand (weights): decoded in registers from the file's bytes (expertrows_types.cuh).
//   Four row streams per thread (gate g, gate g + 8, up g, up g + 8) with their raw
//   words loaded one chunk of 128 values ahead.
// - B operand (activations): the rows of the item gathered by cp.async (`pair_row`)
//   into shared memory, a chunk of 128 values per stage, STAGES stages. The 16-byte
//   pieces are placed so that the 8 lanes of a quarter warp that read at the same
//   time (two rows, four threads) fall into different banks.
// - Epilogue: the accumulators times the activations' row scales, silu and the
//   product in f32, and the hidden values stored as f16. They are divided by a power
//   of two chosen per pair and per 64 hidden values (the largest of the block), which
//   goes to `hscale`: nothing can overflow an f16 and the down kernel multiplies it
//   back. The hidden values of a thread's rows g and g + 8 are stored side by side,
//   which is the order the down kernel reads (expertrows_types.cuh, Q2_0).
//
// An expert without an address is not read: the item's pairs are added to
// `missing_pairs` of the plan's header and the expert is flagged.
//
// What was measured (RTX 4070 / RTX 3060, the recorded routes of Qwen3.8-Flash-Next)
// ------------------------------------------------------------------------------------
// - All NB blocks of 8 columns are computed, however few pairs the item has: branching
//   around the unused ones was 8 % slower at 4096 rows (the compiler cannot overlap the
//   decoding of a step with the `mma` of the step before across the branches), and the
//   tile is chosen so that the waste is small (ExpertRowKernels::tile_for).
// - Stages and register use decide the occupancy: with three stages of 8 KB the 32 pair
//   tile keeps 2 blocks on a multiprocessor, the 64 pair tile with two stages 3 instead of
//   1 (gate and up 10.3 to 7.6 ms at 4096 rows). Loading the weights 2 to 4 chunks ahead
//   with 4 to 6 stages for the small tiles was 30 to 40 % slower (gate and up of 1 row 51
//   to 69 us, of 8 rows 219 to 309 us), accumulating in f16 and flushing every 64 values
//   14 % faster at 4096 rows for 8e-4 more error, activations resident in shared memory
//   for the 8 pair tile no faster.
// - Not the arithmetic bounds it. The kernels reach 33 TFLOPS of useful products at
//   4096 rows (40 with the padding of the last tile of an expert) where a loop of `mma`
//   alone reaches 58.6; removing the decoding, the loads of weights, the barriers or the
//   epilogue one at a time changed nothing measurable. Below 1024 rows the weights'
//   loads bound it: a lane reads 2 and 4 bytes of 41 bytes of a row per chunk, eight rows
//   at a time, and the same bytes read that way (and nothing else) stream at 189 GB/s on the
//   RTX 4070 and 149 on the RTX 3060, where a warp reading whole rows in order gets 473
//   and 337, the limit of the memory. A lane owning two adjacent sub-blocks (the quad reads
//   the whole 82-byte block of a row at once) reached 256 and 257, rows of 88 byte
//   pieces staged by a warp 273 and 218; L2 prefetch hints made it worse (145, 103 GB/s).
//   A layout of the lanes that reads rows in order, or the weights in a form that makes
//   them contiguous, is what would take the kernels from 230 GB/s towards the limit.
#pragma once

#include "expertrows_plan.cuh"
#include "expertrows_types.cuh"
#include "layer_math.cuh"

// The physical position, in a row of 16 pieces of 16 bytes, of the piece 4 t + l that
// thread t reads at step l, for a row of parity `par` (the quarter warp that reads
// together has rows 2k and 2k + 1).
static __device__ __forceinline__ uint32_t ares_er_phys(uint32_t t, uint32_t l, uint32_t par) {
    return ((t >> 1) << 3) | ((t & 1u) << 2) | (l ^ ((t >> 1) << 1) ^ par);
}

// The same for a piece `q` of the row (q = 4 t + l).
static __device__ __forceinline__ uint32_t ares_er_phys_of(uint32_t q, uint32_t par) {
    return ares_er_phys(q >> 2, q & 3u, par);
}

template <class T, int NB, int STAGES>
static __device__ __forceinline__ void ares_er_gate_up(
    const uint64_t* __restrict__ expert_addr, uint32_t up_offset, uint8_t* plan_bytes, const __half* __restrict__ xh,
    const float* __restrict__ xscale, __half* __restrict__ hidden, float* __restrict__ hscale, uint32_t n_embd, uint32_t n_ff,
    uint32_t n_experts, uint32_t n_pairs) {
    constexpr uint32_t NMAX = 8 * NB;
    constexpr uint32_t STAGE_BYTES = NMAX * 256;
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* tab = smem;
    uint8_t* stages = smem + T::TABLE_BYTES;
    uint32_t* rows_sm = reinterpret_cast<uint32_t*>(stages + STAGES * STAGE_BYTES);
    float* redmax = reinterpret_cast<float*>(rows_sm + NMAX);

    const AresErPlan plan(plan_bytes, n_experts, n_pairs, NMAX);
    if (blockIdx.x >= plan.header()->n_items) {
        return;
    }
    const AresErItem item = plan.items()[blockIdx.x];
    const uint64_t addr = expert_addr[item.expert];
    if (addr == 0) {
        if (blockIdx.y == 0 && threadIdx.x == 0) {
            atomicAdd(&plan.header()->missing_pairs, item.count);
            plan.missing()[item.expert] = 1;
        }
        return;
    }

    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t g = lane >> 2, t = lane & 3;
    const uint32_t count = item.count;
    const uint32_t nb_act = (count + 7) >> 3;
    const uint32_t tile = blockIdx.y;
    const uint32_t n_chunks = n_embd >> 7;
    const size_t pitch = ares_er_act_pitch(n_embd);

    for (uint32_t i = threadIdx.x; i < NMAX; i += ARES_ER_THREADS) {
        rows_sm[i] = i < count ? plan.pair_row()[item.first + i] : 0;
    }
    T::init(tab);
    __syncthreads();

    // The activations of chunk `chunk` into stage `stage`.
    auto issue = [&](uint32_t chunk, uint32_t stage) {
        uint8_t* dst = stages + stage * STAGE_BYTES;
#pragma unroll
        for (uint32_t m = 0; m < NMAX / 8; m++) {
            const uint32_t idx = threadIdx.x + ARES_ER_THREADS * m;
            const uint32_t r = idx >> 4, q = idx & 15;
            const bool live = r < count;
            const __half* src = xh + static_cast<size_t>(rows_sm[r]) * pitch + chunk * 128 + 8 * q;
            ares_er_cp16(dst + r * 256 + 16 * ares_er_phys_of(q, r & 1), live ? src : xh, live ? 16 : 0);
        }
    };

    const uint8_t* gate = reinterpret_cast<const uint8_t*>(addr);
    const uint8_t* up = gate + up_offset;
    const size_t row_bytes = static_cast<size_t>(n_embd >> 8) * (T::BLOCK_BYTES);
    const uint32_t row0 = tile * 64 + warp * 16;
    const uint8_t* wrow[4] = {
        gate + (row0 + g) * row_bytes,
        gate + (row0 + g + 8) * row_bytes,
        up + (row0 + g) * row_bytes,
        up + (row0 + g + 8) * row_bytes,
    };
    uint32_t b_ofs[4];
#pragma unroll
    for (uint32_t l = 0; l < 4; l++) {
        b_ofs[l] = g * 256 + 16 * ares_er_phys(t, l, g & 1);
    }

    float acc_g[NB][4], acc_u[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            acc_g[nb][i] = 0.0f;
            acc_u[nb][i] = 0.0f;
        }
    }

#pragma unroll
    for (uint32_t s = 0; s < STAGES - 1; s++) {
        if (s < n_chunks) {
            issue(s, s);
        }
        ares_er_cp_commit();
    }
    typename T::Raw raw[4];
#pragma unroll
    for (int s = 0; s < 4; s++) {
        raw[s] = T::load(wrow[s], t);
    }

    for (uint32_t c = 0; c < n_chunks; c++) {
        ares_er_cp_wait<STAGES - 2>();
        __syncthreads();
        if (c + STAGES - 1 < n_chunks) {
            issue(c + STAGES - 1, (c + STAGES - 1) % STAGES);
        }
        ares_er_cp_commit();

        typename T::Raw next[4];
        if (c + 1 < n_chunks) {
#pragma unroll
            for (int s = 0; s < 4; s++) {
                next[s] = T::load(wrow[s], 4 * (c + 1) + t);
            }
        }
        AresErScales sc[4];
#pragma unroll
        for (int s = 0; s < 4; s++) {
            sc[s] = T::scales(raw[s]);
        }
        const uint8_t* stage = stages + (c % STAGES) * STAGE_BYTES;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            uint32_t h[4][4];
#pragma unroll
            for (int s = 0; s < 4; s++) {
                T::group(raw[s], sc[s], l, tab, h[s]);
            }
#pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                const uint4 b = ares_er_lds128(stage + nb * 2048 + b_ofs[l]);
                ares_er_mma(acc_g[nb], h[0][0], h[1][0], h[0][1], h[1][1], b.x, b.y);
                ares_er_mma(acc_g[nb], h[0][2], h[1][2], h[0][3], h[1][3], b.z, b.w);
                ares_er_mma(acc_u[nb], h[2][0], h[3][0], h[2][1], h[3][1], b.x, b.y);
                ares_er_mma(acc_u[nb], h[2][2], h[3][2], h[2][3], h[3][3], b.z, b.w);
            }
        }
        if (c + 1 < n_chunks) {
#pragma unroll
            for (int s = 0; s < 4; s++) {
                raw[s] = next[s];
            }
        }
    }
    ares_er_cp_wait<0>();

    // The hidden values: silu(gate) * up of the rows g and g + 8, for the columns
    // 2 t and 2 t + 1 of each block of 8.
#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        if (nb < nb_act) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const uint32_t col = nb * 8 + 2 * t + e;
                const float s = col < count ? xscale[rows_sm[col]] : 0.0f;
                const float h0 = ares_silu(acc_g[nb][e] * s) * (acc_u[nb][e] * s);
                const float h1 = ares_silu(acc_g[nb][2 + e] * s) * (acc_u[nb][2 + e] * s);
                acc_g[nb][e] = h0;
                acc_g[nb][2 + e] = h1;
                float m = fmaxf(fabsf(h0), fabsf(h1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
                if (g == 0) {
                    redmax[warp * NMAX + col] = m;
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        if (nb < nb_act) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const uint32_t col = nb * 8 + 2 * t + e;
                if (col < count) {
                    float m = fmaxf(fmaxf(redmax[col], redmax[NMAX + col]), fmaxf(redmax[2 * NMAX + col], redmax[3 * NMAX + col]));
                    int ex = 0;
                    if (m > 0.0f && isfinite(m)) {
                        frexpf(m, &ex);
                        ex = max(-100, min(100, ex));
                    }
                    const float inv = ldexpf(1.0f, -ex);
                    const uint32_t pos = item.first + col;
                    const __half2 pair = __floats2half2_rn(acc_g[nb][e] * inv, acc_g[nb][2 + e] * inv);
                    *reinterpret_cast<__half2*>(hidden + static_cast<size_t>(pos) * n_ff + row0 + 2 * g) = pair;
                    if (warp == 0 && g == 0) {
                        hscale[static_cast<size_t>(tile) * n_pairs + pos] = ldexpf(1.0f, ex);
                    }
                }
            }
        }
    }
}

// Defines the gate and up kernel `NAME` for the type `T`, `NB` blocks of 8 pairs and
// `STAGES` stages of activations.
//
// Grid: (items of the plan or more, n_ff / 64); block: ARES_ER_THREADS; dynamic shared
// memory: ares_er_gate_up_shared<T, NB, STAGES>().
#define ARES_ER_GATE_UP_KERNEL(NAME, T, NB, STAGES)                                                        \
    extern "C" __global__ void __launch_bounds__(ARES_ER_THREADS) NAME(                                   \
        const uint64_t* expert_addr, uint32_t up_offset, uint8_t* plan, const __half* xh, const float* xscale, \
        __half* hidden, float* hscale, uint32_t n_embd, uint32_t n_ff, uint32_t n_experts, uint32_t n_pairs) { \
        ares_er_gate_up<T, NB, STAGES>(expert_addr, up_offset, plan, xh, xscale, hidden, hscale, n_embd, n_ff, \
                                       n_experts, n_pairs);                                                \
    }
