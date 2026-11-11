// Down of the items of a plan on the tensor cores, with the router weights and the sum
// into the rows' outputs:
//
//   out[row] += sum over the row's picks of weight * (down_e hidden[pair])
//
// A thread block takes an item and 64 of the n_embd output values, a warp 16 of them
// (rows g and g + 8 of its slab). The weights are the A operand again, Q2_0 decoded in
// registers: thread t of a quad takes values 16 t to 16 t + 15 of every block of 64
// as one word of codes, and the hidden values (the B operand, a block of 64 per stage,
// 128 bytes a row) are stored in the order that makes the word's half2 (value j, value
// j + 8) the pair of the operand registers (the gate and up kernel writes them so).
//
// The block of 64 values is also what a power of two scale of the hidden values
// covers, so the four `mma` of a block start from zero and their sum is multiplied by
// the column's scale as it is added to the accumulators. The weights carry their own
// block scale d: the decoded value (code - 1) * d is exact in f16.
//
// Two ways to add the experts' outputs to the rows (Accumulate in src/experts.rs):
//
// - Atomic: `red.add` of weight * value into the row of `out`. No scratch, but the
//   order in which a row's experts are added depends on the schedule.
// - Ordered: the value goes to `scratch[pair][n_embd]`, and ares_er_combine adds a
//   row's picks in the order of the router, so the row's bits do not depend on the
//   other rows or on the schedule.
#pragma once

#include "expertrows_gate_up.cuh"

// Q2_0: 64 values in 18 bytes: f16 d, 16 bytes of 2-bit codes, value j in byte j / 4 at
// shift 2 (j % 4); value = d * (code - 1).
struct ErQ20 {
    static constexpr uint32_t BLOCK_BYTES = 18;

    struct Raw {
        uint32_t codes, d;
    };

    // The words of block `b` of the row that thread `t` of a quad decodes.
    static __device__ __forceinline__ Raw load(const uint8_t* row, uint32_t b, uint32_t t) {
        const uint8_t* blk = row + static_cast<size_t>(b) * BLOCK_BYTES;
        Raw r;
        r.codes = ares_ld_stream_u32_a2(blk + 2 + 4 * t);
        r.d = ares_ld_stream_u16(blk);
        return r;
    }

    // The 16 values as 8 half2: (value j, value 8 + j).
    static __device__ __forceinline__ void decode(const Raw& r, uint32_t (&w)[8]) {
        const uint32_t d2 = r.d | (r.d << 16);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint32_t h = ((r.codes >> (2 * j)) & 0x00030003u) | 0x64006400u;
            w[j] = ares_er_h2_mul(ares_er_h2_sub(h, 0x64016401u), d2);
        }
    }
};

template <int NB, int STAGES, bool ATOMIC>
static __device__ __forceinline__ void ares_er_down(
    const uint64_t* __restrict__ expert_addr, uint32_t down_offset, uint8_t* plan_bytes, const __half* __restrict__ hidden,
    const float* __restrict__ hscale, float* __restrict__ out, float* __restrict__ scratch, uint32_t n_ff, uint32_t n_embd,
    uint32_t n_experts, uint32_t n_pairs) {
    constexpr uint32_t NMAX = 8 * NB;
    constexpr uint32_t STAGE_BYTES = NMAX * 128;
    extern __shared__ __align__(16) uint8_t smem[];
    uint32_t* rows_sm = reinterpret_cast<uint32_t*>(smem + STAGES * STAGE_BYTES);
    float* weight_sm = reinterpret_cast<float*>(rows_sm + NMAX);

    const AresErPlan plan(plan_bytes, n_experts, n_pairs, NMAX);
    if (blockIdx.x >= plan.header()->n_items) {
        return;
    }
    const AresErItem item = plan.items()[blockIdx.x];
    const uint64_t addr = expert_addr[item.expert];
    const uint32_t tile = blockIdx.y;
    const uint32_t count = item.count;
    if (addr == 0) {
        // The sum of the rows that wanted this expert gets nothing from it.
        if (!ATOMIC) {
            for (uint32_t i = threadIdx.x; i < count * 64; i += ARES_ER_THREADS) {
                scratch[static_cast<size_t>(item.first + i / 64) * n_embd + tile * 64 + (i & 63)] = 0.0f;
            }
        }
        return;
    }

    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t g = lane >> 2, t = lane & 3;
    const uint32_t nb_act = (count + 7) >> 3;
    const uint32_t n_blocks = n_ff >> 6;

    for (uint32_t i = threadIdx.x; i < NMAX; i += ARES_ER_THREADS) {
        rows_sm[i] = i < count ? plan.pair_row()[item.first + i] : 0;
        weight_sm[i] = i < count ? plan.pair_weight()[item.first + i] : 0.0f;
    }
    __syncthreads();

    // Block `b` of the item's hidden values into stage `stage`.
    auto issue = [&](uint32_t b, uint32_t stage) {
        uint8_t* dst = smem + stage * STAGE_BYTES;
        for (uint32_t idx = threadIdx.x; idx < NMAX * 8; idx += ARES_ER_THREADS) {
            const uint32_t r = idx >> 3, q = idx & 7;
            const bool live = r < count;
            const __half* src = hidden + static_cast<size_t>(item.first + r) * n_ff + 64 * b + 8 * q;
            ares_er_cp16(dst + r * 128 + 16 * (q ^ (r & 1)), live ? src : hidden, live ? 16 : 0);
        }
    };

    const uint8_t* down = reinterpret_cast<const uint8_t*>(addr) + down_offset;
    const size_t row_bytes = static_cast<size_t>(n_blocks) * ErQ20::BLOCK_BYTES;
    const uint32_t row0 = tile * 64 + warp * 16;
    const uint8_t* wrow[2] = {down + (row0 + g) * row_bytes, down + (row0 + g + 8) * row_bytes};

    float acc[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            acc[nb][i] = 0.0f;
        }
    }

#pragma unroll
    for (uint32_t s = 0; s < STAGES - 1; s++) {
        if (s < n_blocks) {
            issue(s, s);
        }
        ares_er_cp_commit();
    }
    ErQ20::Raw raw[2];
    float2 scale[NB];
#pragma unroll
    for (int s = 0; s < 2; s++) {
        raw[s] = ErQ20::load(wrow[s], 0, t);
    }
#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        const uint32_t col = nb * 8 + 2 * t;
        scale[nb] = make_float2(col < count ? hscale[item.first + col] : 0.0f, col + 1 < count ? hscale[item.first + col + 1] : 0.0f);
    }

    for (uint32_t b = 0; b < n_blocks; b++) {
        ares_er_cp_wait<STAGES - 2>();
        __syncthreads();
        if (b + STAGES - 1 < n_blocks) {
            issue(b + STAGES - 1, (b + STAGES - 1) % STAGES);
        }
        ares_er_cp_commit();

        ErQ20::Raw next[2];
        float2 next_scale[NB];
        if (b + 1 < n_blocks) {
#pragma unroll
            for (int s = 0; s < 2; s++) {
                next[s] = ErQ20::load(wrow[s], b + 1, t);
            }
#pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                const uint32_t col = nb * 8 + 2 * t;
                const float* hs = hscale + static_cast<size_t>(b + 1) * n_pairs + item.first + col;
                next_scale[nb] = make_float2(col < count ? hs[0] : 0.0f, col + 1 < count ? hs[1] : 0.0f);
            }
        }
        uint32_t w0[8], w1[8];
        ErQ20::decode(raw[0], w0);
        ErQ20::decode(raw[1], w1);
        const uint8_t* stage = smem + (b % STAGES) * STAGE_BYTES;
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            if (nb < static_cast<int>(nb_act)) {
                const uint8_t* row = stage + (nb * 8 + g) * 128;
                const uint4 p = ares_er_lds128(row + 16 * ((2 * t) ^ (g & 1)));
                const uint4 q = ares_er_lds128(row + 16 * ((2 * t + 1) ^ (g & 1)));
                float tmp[4];
                ares_er_mma0(tmp, w0[0], w1[0], w0[1], w1[1], p.x, p.y);
                ares_er_mma(tmp, w0[2], w1[2], w0[3], w1[3], p.z, p.w);
                ares_er_mma(tmp, w0[4], w1[4], w0[5], w1[5], q.x, q.y);
                ares_er_mma(tmp, w0[6], w1[6], w0[7], w1[7], q.z, q.w);
                acc[nb][0] = fmaf(tmp[0], scale[nb].x, acc[nb][0]);
                acc[nb][1] = fmaf(tmp[1], scale[nb].y, acc[nb][1]);
                acc[nb][2] = fmaf(tmp[2], scale[nb].x, acc[nb][2]);
                acc[nb][3] = fmaf(tmp[3], scale[nb].y, acc[nb][3]);
            }
        }
        if (b + 1 < n_blocks) {
#pragma unroll
            for (int s = 0; s < 2; s++) {
                raw[s] = next[s];
            }
#pragma unroll
            for (int nb = 0; nb < NB; nb++) {
                scale[nb] = next_scale[nb];
            }
        }
    }
    ares_er_cp_wait<0>();

#pragma unroll
    for (int nb = 0; nb < NB; nb++) {
        if (nb < static_cast<int>(nb_act)) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const uint32_t col = nb * 8 + 2 * t + e;
                if (col < count) {
                    const float w = weight_sm[col];
                    const float va = acc[nb][e] * w, vb = acc[nb][2 + e] * w;
                    if (ATOMIC) {
                        float* dst = out + static_cast<size_t>(rows_sm[col]) * n_embd + row0 + g;
                        atomicAdd(dst, va);
                        atomicAdd(dst + 8, vb);
                    } else {
                        float* dst = scratch + static_cast<size_t>(item.first + col) * n_embd + row0 + g;
                        dst[0] = va;
                        dst[8] = vb;
                    }
                }
            }
        }
    }
}

// The sum of the experts' outputs of each row in the order of the router, added to the
// row of `out` (the Ordered way). A block per row.
ARES_KERNEL ares_er_combine(
    uint8_t* plan_bytes, const float* scratch, float* out, uint32_t n_experts, uint32_t n_pairs, uint32_t tile, uint32_t top_k,
    uint32_t n_embd) {
    const AresErPlan plan(plan_bytes, n_experts, n_pairs, tile);
    const uint32_t row = blockIdx.x;
    const uint32_t* pos = plan.pair_pos() + static_cast<size_t>(row) * top_k;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < top_k; j++) {
            const uint32_t p = pos[j];
            if (p != ARES_ER_NO_POS) {
                sum += scratch[static_cast<size_t>(p) * n_embd + e];
            }
        }
        out[static_cast<size_t>(row) * n_embd + e] += sum;
    }
}

// Defines the down kernel `NAME`: `NB` blocks of 8 pairs, `STAGES` stages of hidden
// values, ATOMIC or Ordered.
//
// Grid: (items of the plan or more, n_embd / 64); block: ARES_ER_THREADS; dynamic shared
// memory: STAGES * 8 NB * 128 + 8 NB * 8 bytes.
#define ARES_ER_DOWN_KERNEL(NAME, NB, STAGES, ATOMIC)                                                          \
    extern "C" __global__ void __launch_bounds__(ARES_ER_THREADS) NAME(                                        \
        const uint64_t* expert_addr, uint32_t down_offset, uint8_t* plan, const __half* hidden, const float* hscale, \
        float* out, float* scratch, uint32_t n_ff, uint32_t n_embd, uint32_t n_experts, uint32_t n_pairs) {   \
        ares_er_down<NB, STAGES, ATOMIC>(expert_addr, down_offset, plan, hidden, hscale, out, scratch, n_ff, n_embd, \
                                         n_experts, n_pairs);                                                  \
    }
