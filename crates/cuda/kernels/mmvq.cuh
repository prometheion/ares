// The skeleton of the fast weight kernels: quantized matrix times 1 to 8
// activation rows, with integer dot products over activations that
// ares_quantize_act (quantize_act.cuh) has quantized to int8.
//
// Why it looks like this
// ----------------------
// Decode reads every weight once per token, so the time of a matrix is its
// bytes divided by the memory bandwidth, if the kernel keeps enough loads in
// flight and spends little per byte. The structure serves that:
//
// - A warp works on `ROWS` output rows; its 32 lanes split the row's blocks
//   between them. A type says how many lanes share one block (`T::LANES`) and
//   what a lane does with its share (`load_w`, `load_a`, `dot`), so adding a
//   type is writing those three and nothing else.
// - Every weight byte is loaded by exactly one lane, with the widest load its
//   alignment allows, straight from memory into registers (no smem staging of
//   weights: they are used once). The loads of all `ROWS` rows of an iteration
//   are issued before any is used, which is what keeps bytes in flight.
// - The activations of the batch are read through the L1 cache; they are tiny
//   next to the weights and every block of the grid reads the same ones. One
//   activation load serves all `ROWS` rows.
// - The batch size is a kernel argument. The loop over batch rows is unrolled
//   to the largest batch and each step is skipped by a branch that is the same
//   for all threads, so the accumulators stay in registers and the weights are
//   still read once however many rows there are.
//
// A type `T` provides:
//   BLOCK_VALUES, BLOCK_BYTES    the geometry of a block (GGUF layout)
//   LANES                        lanes per block; divides 32
//   SMEM_BYTES, init(smem, t, n) a codebook the type wants in shared memory
//                                (0 for none) and how thread t of n copies it
//   struct W, A                  the registers a lane holds for one block
//   W load_w(blk, slot, smem)    loads and prepares the lane's share of one
//                                weight block; `slot` is the lane's index
//                                within the block (0..LANES-1)
//   A load_a(q, ds, slot)        loads the lane's share of the activations of
//                                the same block (q: the block's int8 values,
//                                ds: its (d, s) pairs)
//   void dot_acc(acc, W, A)      adds the lane's contribution to the dot product
//                                of the block with the activations to `acc`
#pragma once

#include "common.cuh"
#include <type_traits>

#include "quantize_act.cuh"

// Loads of weights. Every weight byte is read once, so caching it in L1 would only
// evict the activations, which are used again and again. The loads differ in how
// they leave L1 alone:
//
// - 16-byte loads (Q4_K, Q5_K: a lane reads whole 32-byte sectors in two of them)
//   are not allocated in L1 at all (`no_allocate`).
// - Smaller loads (2, 4 or 8 bytes: a lane needs part of a sector, and the rest of
//   it comes with the next load of the same lane or its neighbour) are allocated but
//   marked `evict_first`. With `no_allocate` every one of them asks L2 for its
//   sector again. Measured on 17408 x 5120 matrices on an RTX 4070 (1 / 8 rows, GB/s):
//     2 bytes  (Q6_K, Q3_K, Q8_0, ...)   360 -> 468 at 1 row, 228 -> 333 at 8
//     4 bytes  (Q4_1, Q5_1)              396 -> 473 at 1 row, 206 -> 408 at 8 (Q5_1)
//     8 bytes  (IQ4_XS)                  322 -> 426 at 8 rows
//   and the same or more on an RTX 3060. For the 16-byte loads the change is neutral
//   (Q4_K 8 rows 429 -> 424, Q5_K 427 -> 443), so they stay as they were.
// Plain caching (neither hint) made 8 rows much worse for every type: the weights
// push the activations out of L1.

static __device__ __forceinline__ uint4 ares_ld_stream_v4(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint2 ares_ld_stream_v2(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::evict_first.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ares_ld_stream_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::evict_first.u32 {%0}, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ares_ld_stream_u8(const uint8_t* p) {
    return __ldg(p);
}

static __device__ __forceinline__ uint32_t ares_ld_stream_u16(const void* p) {
    uint16_t v;
    asm volatile("ld.global.nc.L1::evict_first.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

// 32 bits from an address that is only 2-byte aligned: the blocks of Q6_K,
// Q3_K and the i-quants are 110, 210, 82 ... bytes, so half of them (or all,
// for some types) start 2 bytes off a 4-byte boundary.
static __device__ __forceinline__ uint32_t ares_ld_stream_u32_a2(const uint8_t* p) {
    return ares_ld_stream_u16(p) | (ares_ld_stream_u16(p + 2) << 16);
}

// 64 bits from an address that is only 2-byte aligned.
static __device__ __forceinline__ uint2 ares_ld_stream_v2_a2(const uint8_t* p) {
    return make_uint2(ares_ld_stream_u32_a2(p), ares_ld_stream_u32_a2(p + 4));
}

// Activation loads: cached, since the whole grid reads the same few bytes.
static __device__ __forceinline__ uint4 ares_ld_act_v4(const void* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

// How many blocks of a row a lane loads before it uses any: a type can set `UNROLL`
// (default 1). The loads of the weights are what the memory system works on, so a
// type that needs few bytes per lane step (the 1- and 2-bit types) keeps more bytes in
// flight per warp by issuing the loads of several steps together.
template <class T, class = void>
struct AresUnroll {
    static constexpr int value = 1;
};
template <class T>
struct AresUnroll<T, std::void_t<decltype(T::UNROLL)>> {
    static constexpr int value = T::UNROLL;
};

// y[b][r] = sum over c of W[r][c] * x[b][c], for b < n_batch, r < n_out.
//
//   w   n_out rows of n_in / BLOCK_VALUES blocks, 16-byte aligned
//   aq  the int8 values of the quantized activations (n_batch rows of n_in)
//   y   n_batch rows of n_out f32 values
//
// Grid: up to ceil(n_out / (WARPS * ROWS)) blocks of WARPS * 32 threads; with
// fewer, each block takes several tiles of rows.
template <class T, int WARPS, int ROWS>
static __device__ __forceinline__ void ares_mmvq(
    const uint8_t* __restrict__ w, const int8_t* __restrict__ aq, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch) {
    static_assert(32 % T::LANES == 0, "a block is shared by whole lanes of a warp");

    // A codebook goes to shared memory once per thread block: lookups with a
    // different index in every lane are conflict-limited there and would touch
    // many cache lines in global memory.
    __shared__ __align__(16) uint8_t smem[T::SMEM_BYTES > 0 ? T::SMEM_BYTES : 1];
    if (T::SMEM_BYTES > 0) {
        T::init(smem, threadIdx.x, WARPS * 32);
        __syncthreads();
    }

    const float2* __restrict__ ad =
        reinterpret_cast<const float2*>(aq + ares_act_ds_offset(static_cast<size_t>(n_batch) * n_in));

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;

    // Thread blocks walk the tiles of rows (WARPS * ROWS rows each) in steps of the
    // grid, so that a grid smaller than the matrix amortises the codebook copy.
    const uint32_t n_tiles = (n_out + WARPS * ROWS - 1) / (WARPS * ROWS);
    const auto rows_of_tile = [&](const uint32_t tile) {
        const uint32_t row0 = (tile * WARPS + warp) * ROWS;
        const uint32_t blocks_per_row = n_in / T::BLOCK_VALUES;
        const size_t row_bytes = static_cast<size_t>(blocks_per_row) * T::BLOCK_BYTES;
        const uint32_t groups_per_row = n_in >> 5;
        const uint32_t slot = lane % T::LANES;
        const uint32_t kb_step = 32 / T::LANES;

        // The rows past the end of the matrix repeat the last one; their results
        // are not written.
        const uint8_t* wrow[ROWS];
    #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            wrow[r] = w + static_cast<size_t>(min(row0 + r, n_out - 1)) * row_bytes;
        }

        float acc[ROWS][ARES_MAX_BATCH];
    #pragma unroll
        for (int r = 0; r < ROWS; r++) {
    #pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                acc[r][b] = 0.0f;
            }
        }

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
                    if (b < static_cast<int>(n_batch)) {
#pragma unroll
                        for (int u = 0; u < U; u++) {
                            const uint32_t k = kb + u * kb_step;
                            const typename T::A a = T::load_a(
                                aq + static_cast<size_t>(b) * n_in + static_cast<size_t>(k) * T::BLOCK_VALUES,
                                ad + static_cast<size_t>(b) * groups_per_row +
                                    static_cast<size_t>(k) * (T::BLOCK_VALUES / 32),
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
                if (b < static_cast<int>(n_batch)) {
                    const typename T::A a = T::load_a(
                        aq + static_cast<size_t>(b) * n_in + static_cast<size_t>(kb) * T::BLOCK_VALUES,
                        ad + static_cast<size_t>(b) * groups_per_row + static_cast<size_t>(kb) * (T::BLOCK_VALUES / 32),
                        slot);
    #pragma unroll
                    for (int r = 0; r < ROWS; r++) {
                        T::dot_acc(acc[r][b], wr[r], a);
                    }
                }
            }
        }

        // Sum over the lanes of the warp; lane 0 writes.
    #pragma unroll
        for (int r = 0; r < ROWS; r++) {
    #pragma unroll
            for (int b = 0; b < ARES_MAX_BATCH; b++) {
                if (b < static_cast<int>(n_batch)) {
                    const float total = ares_warp_sum(acc[r][b]);
                    if (lane == 0 && row0 + r < n_out) {
                        y[static_cast<size_t>(b) * n_out + row0 + r] = total;
                    }
                }
            }
        }
    };
    // Only the types with a large codebook walk several tiles per block (the launch of
    // the others has a block for every tile, and the loop would cost them registers).
    if constexpr (T::SMEM_BYTES >= 16384) {
        for (uint32_t tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
            rows_of_tile(tile);
        }
    } else {
        rows_of_tile(blockIdx.x);
    }
}

// The launch shape of the weight kernels of every type: 4 warps per thread
// block, 2 output rows per warp (8 rows per block). Measured on an RTX 4070
// and an RTX 3060 for 1 to 8 activation rows, this is within a few percent of
// the best of the shapes tried (1, 2 or 4 rows per warp, 2, 4 or 8 warps) for
// matrices from 2048 to 17408 rows, and the choice does not depend on the
// batch. Splitting K over the warps of a block, which helps matrices of a few
// hundred rows by about a microsecond (below the cost of a launch), was tried
// as a kernel argument and dropped: it changed the code of every kernel for
// the worse (q3_K and Q6_K lost 10 to 15% on large matrices) and the
// whole-model pass did not gain.
#define ARES_MMVQ_WARPS 4
#define ARES_MMVQ_ROWS 2

// Defines the kernel `NAME` for the weight type `T`: y = W x for the quantized
// matrix `w` (n_out rows of n_in values) and n_batch rows of activations that
// ares_quantize_act produced at `aq`.
#define ARES_MMVQ_KERNEL(NAME, T)                                                              \
    extern "C" __global__ void __launch_bounds__(ARES_MMVQ_WARPS * 32) NAME(                  \
        const uint8_t* w, const int8_t* aq, float* y, uint32_t n_in, uint32_t n_out,         \
        uint32_t n_batch) {                                                                   \
        ares_mmvq<T, ARES_MMVQ_WARPS, ARES_MMVQ_ROWS>(w, aq, y, n_in, n_out, n_batch);          \
    }
