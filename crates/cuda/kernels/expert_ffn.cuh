// The routed experts of a layer for the rows of a decode step:
//
//   f(e)   = down_e @ (silu(gate_e @ m) * (up_e @ m))
//   out[b] += sum over the experts e of row b of wt[b][e] * f(e)
//
// in two launches that read each picked expert's weights once, however many
// rows picked it. The weights are the file's quantized bytes in the expert's
// slot; the products are the integer dot products of the decode family
// (mmvq.cuh) with the decoders of the weight types (`T::load_w`, `T::load_a`,
// `T::dot_acc`), so a type that has a decode kernel can be an expert's type.
//
// 1. ares_expert_gate_up_<type>: a thread block per picked expert and 32
//    values of its inner width. A warp reads row j of the gate and row j of
//    the up matrix together, both against the same activations, and has
//    silu(gate) * up of every row of the step that picked the expert. The
//    block then quantizes its 32 values per row for the down product, in the
//    format of quantize_act.cuh (int8 with a scale per 32), so no kernel sits
//    between the two.
// 2. ares_expert_down_<type>: the layout of the decode kernels (a block per 8
//    output values, a warp per 2). Each warp walks the picked experts in
//    ascending order, multiplies its rows of every down matrix with the
//    hidden values of the rows that picked the expert, and adds the weighted
//    results to the rows' outputs. A row's experts are therefore always added
//    in the same order, whatever else the step computes.
//
// An expert that was picked and has no slot yet is waited for in step 1, by
// the blocks of that expert only, on the device (see expert_layout.cuh); the
// blocks of the experts that are there compute meanwhile. The wait ends with
// the expert's slot, or with the host's claim of the group: then the CPU
// computes the expert, both kernels leave it out, and the down kernel adds
// the host's sum for the rows at its end, waiting for it if it must. A record
// without a missing expert never looks at the host share.
#pragma once

#include "common.cuh"
#include "expert_layout.cuh"
#include "layer_math.cuh"
#include "mmvq.cuh"

// The dot products of `ROWS` weight rows with the activation rows named in
// `mask`, added to `acc`: the inner loop of ares_mmvq, with the activation
// row of batch entry b at index `arow[b]` of the arrays at `q0` and `ds0`.
template <class T, int ROWS>
static __device__ __forceinline__ void ares_expert_dots(
    const uint8_t* const* wrow, const int8_t* __restrict__ q0, const float2* __restrict__ ds0,
    const uint32_t* arow, uint32_t mask, uint32_t n_in, uint32_t lane, const uint8_t* smem,
    float (&acc)[ROWS][ARES_MAX_BATCH]) {
    const uint32_t blocks_per_row = n_in / T::BLOCK_VALUES;
    const uint32_t groups_per_row = n_in >> 5;
    const uint32_t slot = lane % T::LANES;
    const uint32_t kb_step = 32 / T::LANES;

    constexpr int U = AresUnroll<T>::value;
    uint32_t kb = lane / T::LANES;
    if constexpr (U > 1) {
        for (; kb + (U - 1) * kb_step < blocks_per_row; kb += U * kb_step) {
            typename T::W wr[U][ROWS];
#pragma unroll
            for (int u = 0; u < U; u++) {
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    wr[u][r] = T::load_w(wrow[r] + static_cast<size_t>(kb + u * kb_step) * T::BLOCK_BYTES, slot, smem);
                }
            }
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                if ((mask >> b) & 1u) {
#pragma unroll
                    for (int u = 0; u < U; u++) {
                        const uint32_t kk = kb + u * kb_step;
                        const typename T::A a = T::load_a(
                            q0 + static_cast<size_t>(arow[b]) * n_in + static_cast<size_t>(kk) * T::BLOCK_VALUES,
                            ds0 + static_cast<size_t>(arow[b]) * groups_per_row +
                                static_cast<size_t>(kk) * (T::BLOCK_VALUES / 32),
                            slot);
#pragma unroll
                        for (int r = 0; r < ROWS; r++) {
                            T::dot_acc(acc[r][b], wr[u][r], a);
                        }
                    }
                }
            }
        }
    }
    for (; kb < blocks_per_row; kb += kb_step) {
        typename T::W wr[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            wr[r] = T::load_w(wrow[r] + static_cast<size_t>(kb) * T::BLOCK_BYTES, slot, smem);
        }
#pragma unroll
        for (int b = 0; b < ARES_MAX_BATCH; b++) {
            if ((mask >> b) & 1u) {
                const typename T::A a = T::load_a(
                    q0 + static_cast<size_t>(arow[b]) * n_in + static_cast<size_t>(kb) * T::BLOCK_VALUES,
                    ds0 + static_cast<size_t>(arow[b]) * groups_per_row +
                        static_cast<size_t>(kb) * (T::BLOCK_VALUES / 32),
                    slot);
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    T::dot_acc(acc[r][b], wr[r], a);
                }
            }
        }
    }
}

// The launch shape of the gate and up kernel: 8 warps, each computes 4 of the
// block's 32 hidden values one after the other. Warp w then quantizes row w.
#define ARES_EXPERT_GU_WARPS 8
#define ARES_EXPERT_GU_VALUES 4

static_assert(ARES_EXPERT_GU_WARPS == ARES_MAX_BATCH, "a warp quantizes one row of the step");
static_assert(ARES_EXPERT_GU_WARPS * ARES_EXPERT_GU_VALUES == 32, "a block is one group of 32 values");

// Grid: (n_ff / 32, groups): block (t, g) computes the hidden values
// [32 t, 32 t + 32) of group g of the record. Blocks of groups past the
// record's count return at once, so the launch may be sized for the most
// groups a step can have (n_rows * k) when the host does not know the count.
//
//   pool, stride, up_offset   the layer's slots (expert_layout.cuh)
//   slot_of                   its slot table
//   record                    the route record of the layer and step
//   aq                        the rows' activations (quantize_act.cuh), n_rows of n_in
//   hq, hds                   out: the hidden values of every pair, quantized:
//                             pair p has n_ff int8 at hq + p * n_ff and n_ff / 32
//                             (d, s) at hds + p * (n_ff / 32)
//   host                      the layer's host share, or null: without one a
//                             missing expert can only arrive in a slot
template <class T>
static __device__ __forceinline__ void ares_expert_gate_up(
    const uint8_t* __restrict__ pool, uint64_t stride, uint32_t up_offset, const int32_t* slot_of,
    const uint8_t* __restrict__ record, const int8_t* __restrict__ aq, int8_t* __restrict__ hq,
    float2* __restrict__ hds, const AresHostShare* host, uint32_t n_in, uint32_t n_ff, uint32_t n_rows,
    uint32_t k) {
    static_assert(32 % T::LANES == 0, "a block is shared by whole lanes of a warp");
    __shared__ __align__(16) uint8_t smem[T::SMEM_BYTES > 0 ? T::SMEM_BYTES : 1];
    __shared__ float hidden[ARES_MAX_BATCH][32];
    __shared__ int32_t slot_found;

    if (blockIdx.y >= ares_route_header(record)->n_groups) {
        return;
    }
    const AresExpertGroup* group = ares_route_groups(record, n_rows * k) + blockIdx.y;
    const uint32_t expert = group->expert;
    const uint32_t mask = group->rows;
    const uint32_t first_pair = group->first_pair;

    if (threadIdx.x == 0) {
        // The expert's slot, waited for if the expert is still on its way,
        // until it has one or the host says that it computes the expert.
        const volatile int32_t* entry = reinterpret_cast<const volatile int32_t*>(&slot_of[expert]);
        int32_t slot = *entry;
        if (slot < 0) {
            const uint32_t mine = ares_route_header(record)->token + 1u;
            const volatile uint32_t* claim =
                host != nullptr ? reinterpret_cast<const volatile uint32_t*>(&host->claim[blockIdx.y]) : nullptr;
            uint32_t polls = 0;
            while (slot < 0 && ++polls != ARES_EXPERT_MAX_POLLS) {
                if (claim != nullptr && *claim == mine) {
                    slot = ARES_EXPERT_ON_HOST;
                    break;
                }
                slot = *entry;
            }
        }
        slot_found = slot;
    }
    if (T::SMEM_BYTES > 0) {
        T::init(smem, threadIdx.x, ARES_EXPERT_GU_WARPS * 32);
    }
    __syncthreads();
    const int32_t slot = slot_found;
    if (slot == ARES_EXPERT_ON_HOST) {
        // The host computes this expert: nothing of it is needed here.
        return;
    }

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t tile = blockIdx.x;

    if (slot >= 0) {
        const uint8_t* gate = pool + static_cast<size_t>(slot) * stride;
        const uint8_t* up = gate + up_offset;
        const size_t row_bytes = static_cast<size_t>(n_in / T::BLOCK_VALUES) * T::BLOCK_BYTES;
        const float2* ad =
            reinterpret_cast<const float2*>(aq + ares_act_ds_offset(static_cast<size_t>(n_rows) * n_in));
        uint32_t arow[ARES_MAX_BATCH];
#pragma unroll
        for (int b = 0; b < ARES_MAX_BATCH; b++) {
            arow[b] = b;
        }

        for (uint32_t v = 0; v < ARES_EXPERT_GU_VALUES; v++) {
            const uint32_t in_tile = warp * ARES_EXPERT_GU_VALUES + v;
            const uint32_t j = min(tile * 32 + in_tile, n_ff - 1);
            const uint8_t* wrow[2] = {gate + j * row_bytes, up + j * row_bytes};
            float acc[2][ARES_MAX_BATCH];
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                acc[0][b] = 0.0f;
                acc[1][b] = 0.0f;
            }
            ares_expert_dots<T, 2>(wrow, aq, ad, arow, mask, n_in, lane, smem, acc);
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                if ((mask >> b) & 1u) {
                    const float g = ares_warp_sum(acc[0][b]);
                    const float u = ares_warp_sum(acc[1][b]);
                    if (lane == 0) {
                        hidden[b][in_tile] = ares_silu(g) * u;
                    }
                }
            }
        }
    }
    __syncthreads();

    // Warp b quantizes the block's 32 values of row b.
    if (((mask >> warp) & 1u) && tile * 32 + lane < n_ff) {
        // An expert that never arrived poisons the rows that wanted it.
        const float v = slot >= 0 ? hidden[warp][lane] : __int_as_float(0x7fc00000);
        float amax = fabsf(v);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
        }
        const float d = amax / 127.0f;
        const float code = amax > 0.0f ? roundf(v / d) : 0.0f;
        const float sum_codes = ares_warp_sum(code);
        const size_t pair = first_pair + ares_pair_of_row(mask, warp);
        hq[pair * n_ff + tile * 32 + lane] = static_cast<int8_t>(code);
        if (lane == 0) {
            hds[pair * (n_ff >> 5) + tile] = make_float2(d, d * sum_codes);
        }
    }
}

// Grid: ceil(n_out / (ARES_MMVQ_WARPS * ARES_MMVQ_ROWS)) blocks of
// ARES_MMVQ_WARPS warps.
//
//   pool, stride, down_offset, slot_of, record   as above
//   hq, hds   the pairs' hidden values, as ares_expert_gate_up wrote them
//   out       n_rows rows of n_out f32; the experts' weighted sum is added
//   host      the layer's host share, or null
template <class T>
static __device__ __forceinline__ void ares_expert_down(
    const uint8_t* __restrict__ pool, uint64_t stride, uint32_t down_offset, const int32_t* slot_of,
    const uint8_t* __restrict__ record, const int8_t* __restrict__ hq, const float2* __restrict__ hds,
    float* __restrict__ out, const AresHostShare* host, uint32_t n_ff, uint32_t n_out, uint32_t n_rows,
    uint32_t k) {
    static_assert(32 % T::LANES == 0, "a block is shared by whole lanes of a warp");
    constexpr int ROWS = ARES_MMVQ_ROWS;
    __shared__ __align__(16) uint8_t smem[T::SMEM_BYTES > 0 ? T::SMEM_BYTES : 1];
    if (T::SMEM_BYTES > 0) {
        T::init(smem, threadIdx.x, ARES_MMVQ_WARPS * 32);
        __syncthreads();
    }

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t row0 = (blockIdx.x * ARES_MMVQ_WARPS + warp) * ROWS;
    const size_t row_bytes = static_cast<size_t>(n_ff / T::BLOCK_VALUES) * T::BLOCK_BYTES;
    const uint32_t n_groups = ares_route_header(record)->n_groups;
    const AresExpertGroup* groups = ares_route_groups(record, n_rows * k);

    float acc[ROWS][ARES_MAX_BATCH];
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
#pragma unroll
        for (int b = 0; b < ARES_MAX_BATCH; b++) {
            acc[r][b] = 0.0f;
        }
    }

    // Whether the host computes any of the groups: only a group without a
    // slot can say so, so a record whose experts are all there reads nothing
    // of the host share.
    const uint32_t mine = ares_route_header(record)->token + 1u;
    bool on_host = false;

    for (uint32_t g = 0; g < n_groups; g++) {
        const AresExpertGroup group = groups[g];
        const int32_t slot = slot_of[group.expert];
        const uint32_t mask = group.rows;
        if (slot < 0 && host != nullptr && host->claim[g] == mine) {
            on_host = true;
            continue;
        }
        float part[ROWS][ARES_MAX_BATCH];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                // No slot even now: the gate and up kernel gave up waiting.
                part[r][b] = slot >= 0 ? 0.0f : __int_as_float(0x7fc00000);
            }
        }
        if (slot >= 0) {
            const uint8_t* down = pool + static_cast<size_t>(slot) * stride + down_offset;
            const uint8_t* wrow[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                wrow[r] = down + static_cast<size_t>(min(row0 + r, n_out - 1)) * row_bytes;
            }
            uint32_t arow[ARES_MAX_BATCH];
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                arow[b] = group.first_pair + ares_pair_of_row(mask, b);
            }
            ares_expert_dots<T, ROWS>(wrow, hq, hds, arow, mask, n_ff, lane, smem, part);
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
#pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                if ((mask >> b) & 1u) {
                    acc[r][b] = fmaf(group.weight[b], part[r][b], acc[r][b]);
                }
            }
        }
    }

    // The host's sums, waited for like a slot: by the one thread of the warp
    // that writes the outputs. Sums that never arrive poison the rows.
    const float* host_sum = nullptr;
    bool host_lost = false;
    if (on_host && lane == 0) {
        const volatile uint32_t* done = reinterpret_cast<const volatile uint32_t*>(&host->done);
        uint32_t polls = 0;
        while (*done != mine && ++polls != ARES_EXPERT_MAX_POLLS) {
        }
        host_lost = *done != mine;
        host_sum = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(host) + ARES_HOST_SUM_OFFSET);
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
#pragma unroll
        for (int b = 0; b < ARES_MAX_BATCH; b++) {
            if (b < static_cast<int>(n_rows)) {
                float total = ares_warp_sum(acc[r][b]);
                if (lane == 0 && row0 + r < n_out) {
                    if (host_sum != nullptr) {
                        total += host_lost ? __int_as_float(0x7fc00000)
                                           : host_sum[static_cast<size_t>(b) * n_out + row0 + r];
                    }
                    out[static_cast<size_t>(b) * n_out + row0 + r] += total;
                }
            }
        }
    }
}

// Defines the gate and up kernel `NAME` for the weight type `T`.
#define ARES_EXPERT_GATE_UP_KERNEL(NAME, T)                                                     \
    extern "C" __global__ void __launch_bounds__(ARES_EXPERT_GU_WARPS * 32) NAME(              \
        const uint8_t* pool, uint64_t stride, uint32_t up_offset, const int32_t* slot_of,       \
        const uint8_t* record, const int8_t* aq, int8_t* hq, float2* hds,                       \
        const AresHostShare* host, uint32_t n_in, uint32_t n_ff, uint32_t n_rows, uint32_t k) { \
        ares_expert_gate_up<T>(pool, stride, up_offset, slot_of, record, aq, hq, hds, host,     \
                               n_in, n_ff, n_rows, k);                                          \
    }

// Defines the down kernel `NAME` for the weight type `T`.
#define ARES_EXPERT_DOWN_KERNEL(NAME, T)                                                        \
    extern "C" __global__ void __launch_bounds__(ARES_MMVQ_WARPS * 32) NAME(                   \
        const uint8_t* pool, uint64_t stride, uint32_t down_offset, const int32_t* slot_of,     \
        const uint8_t* record, const int8_t* hq, const float2* hds, float* out,                 \
        const AresHostShare* host, uint32_t n_ff, uint32_t n_out, uint32_t n_rows, uint32_t k) {\
        ares_expert_down<T>(pool, stride, down_offset, slot_of, record, hq, hds, out, host,     \
                            n_ff, n_out, n_rows, k);                                            \
    }
