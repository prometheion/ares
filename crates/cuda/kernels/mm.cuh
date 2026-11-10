// The prefill weight kernels: quantized matrix times a batch of activation rows
// on the tensor cores, with the weights dequantised tile by tile on the fly.
//
// Why a second family
// -------------------
// The decode kernels (mmvq*.cuh) read each weight once and spend a dp4a per
// four weights per activation row, which is the right shape up to 8 rows: the
// time is the weights' bytes over the memory bandwidth. Past a few rows the
// work is arithmetic, and the tensor cores are two orders of magnitude ahead
// of the CUDA cores at it. The price of using them is that a weight must be
// available as a tile in shared memory in the format the instruction reads,
// so the kernels here dequantise a tile of the weights per step and amortise
// that over every activation row of the thread block.
//
// Two variants, one set of weight decoders
// ----------------------------------------
// - int8 (`ares_mm_i8_*`): the tile holds the weights' integer codes (Q4_K's
//   0..15, a table value of an i-quant, ...), the activations are the int8
//   values that ares_quantize_act produced for the decode kernels (same buffer,
//   so one quantization serves both families), and the instruction is the
//   integer `mma.m16n8k32` (`k16` for types with a scale per 16 values). The
//   sub-block scales (and Q4_K's minimums) are applied to the integer result
//   of each 32 values in f32.
// - f16 (`ares_mm_f16_*`): the tile holds the dequantised weights as f16, the
//   activations are f16, the instruction is `mma.m16n8k16` with f32
//   accumulation, and nothing is applied per step. It is the direct, more exact
//   product: no rounding of the activations to 8 bits.
// Which one runs is decided per device (Kernels::prefill_path). GeForce cards
// halve the rate of FP16 `mma` with f32 accumulation (but not of the integer
// one) unless that cap is lifted; the int8 variant is the fast choice on a
// capped card, and the f16 variant the exact one, which a card without the cap
// runs at its full rate. Kernels::load measures which case a device is.
//
// Structure of a thread block
// ---------------------------
// A block computes a tile of I weight rows by J activation rows. The weights are
// the A operand of `mma` (16 rows), the activations the B operand (8 rows):
// every result element then is `y[b][r]` with consecutive r in consecutive lanes
// of a quad, which stores 32-byte segments without staging through shared
// memory. The K dimension goes through in steps of 64 values.
//
//   - Activations: `cp.async` straight into shared memory, double buffered; the
//     rows of a tile are contiguous in global memory (n_in bytes per row).
//   - Weights: a thread loads the raw bytes of its share of the next step into
//     registers while the tensor cores work on the current one, then decodes
//     them (T::decode) and writes the tile (int8 codes or f16) for the step
//     after. One `__syncthreads` per step.
//   - Shared memory per block is below 48 KB: more would need the launch
//     attribute CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, which the
//     Function wrapper does not set. Two blocks of that size share an SM, which
//     is what hides the latency of one block's barriers.
//   - The grid is 1-D and ordered so that the blocks running at the same time
//     share weight tiles: a group of `group` activation tiles takes all weight
//     tiles in turn. The weights then come from DRAM once per group instead of
//     once per block (a Q4_K matrix of 17408 x 5120 with 1024 rows on an RTX
//     4070: 4.45 ms -> 2.64 ms in the first version of the kernel), and the
//     activations of a group stay in L2.
//
// What bounds the kernels, and how that is known
// ----------------------------------------------
// Measured with ablations (a kernel with the weight path, the activation path
// or the scaling of the results removed in turn), `cuobjdump` instruction
// counts and the probe in prefill.cu; `ncu` counters are not available on the
// development machine.
//
// - Large batches (256 rows and more): no single resource. On the RTX 4070 the
//   integer `mma` and `ldmatrix` alone reach 87% of the tensor cores' peak
//   (204 of 234 TOPS). Adding the scaling of the results (a conversion, a
//   multiply and two fused multiply-adds per output and 32 values of K: 260 of
//   the 500 to 620 instructions a warp issues per step) leaves 115 TOPS; the
//   full kernel with the weight path removed reaches 89, with the activation
//   path removed 83, and the whole kernel about 65. The costs add instead of
//   overlapping because a step is short (a few microseconds) and every warp of
//   a block waits for the same barrier.
//   Deeper prefetch of the weights (lookahead 2 and 3), more blocks per SM and
//   a prefetch of weight lines into L2 did not change the time or made it worse;
//   removing a fifth of the instructions (hoisting the address arithmetic out of
//   the loop) did not change it either, so it is not the number of instructions
//   but the latency chain of a step. The f16 variant reaches 85 to 88% of the
//   capped rate of f16 `mma` with f32 accumulation on both GPUs (49.6 of 58.6 and
//   22.5 of 25.6 TFLOPS for Q4_K at 1024 rows); with f16 accumulation, the rate
//   of an uncapped card, it takes the same time (49.1 TFLOPS on the RTX 4070), so
//   something other than the rate of the tensor cores bounds it, and why it
//   gains nothing from the doubled rate is not settled.
// - Small batches (up to 64 rows): the weights. The kernels read 32 to 70
//   bytes of each of 64 to 128 rows per step, which GDDR6 serves at about a
//   third of its bandwidth (the same kernel with every block reading the same
//   128 rows from L2 takes 0.20 ms instead of 0.37 ms for 64 rows). A loader that
//   reads whole blocks per row at once would need 20 to 100 more registers per
//   thread; it is the next step for this range.
// - Types with 2-byte aligned blocks (Q6_K, Q3_K, Q8_0, IQ4_NL): their words
//   are read from aligned addresses and shifted, several per sector; with the
//   loads kept in L1 (ares_ld_words_a2) Q6_K is 2.6 times as fast as with the
//   non-allocating loads of the decode kernels.
//
// A type `T` provides (mm_types.cuh):// A type `T` provides (mm_types.cuh):
//   BLOCK_VALUES, BLOCK_BYTES, SMEM_BYTES, init   as in the decode kernels
//   HAS_MIN                the values have an offset per 32: v = sw * code - mw
//   SPLIT16                the scale changes every 16 values (an integer scale
//                          per half of a 32-value item)
//   SIGNED_CODES           the codes are signed bytes (table values, grid values
//                          with signs, Q6_K's code - 32), not 0..255
//   ITEM_STRIDE, RUN_STRIDE   where an item's two runs of 16 codes go in the
//                          64-value row of the tile: item * ITEM_STRIDE + run *
//                          RUN_STRIDE
//   Raw load(row, chunk, item)       the global loads of an item
//   void decode(raw, chunk, item, smem, AresMmDec&)   the decoding
// A step of 64 values has two items per weight row (32 values each).
#pragma once

#include "common.cuh"
#include "mmvq.cuh"

// What an item decodes to.
struct AresMmDec {
    uint32_t w[8];  // 32 codes, one per byte: run 0 = w[0..3], run 1 = w[4..7]
    float sw[2];    // f16 variant: value of a code = sw[run] * code - mw[run]
    float mw[2];
    // int8 variant: the scale entry of the item's 32 values. Types with one
    // scale per 32: (scale, minimum), with value = scale * code - minimum. Types
    // with a scale per 16: (d, two signed 16-bit integer scales packed as the
    // bits of a float), with value = d * scale * code.
    float2 ent;
};

// Values per step along K, and the row pitch of the tiles in shared memory in
// bytes: the data (64 bytes of int8, 128 of f16) plus 16, so that the eight rows
// of one `ldmatrix` matrix fall into different banks.
#define ARES_MM_KC 64
#define ARES_MM_PITCH8 80
#define ARES_MM_PITCH16 144

static __device__ __forceinline__ uint32_t ares_smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// `cp.async` of 16 bytes, of which `bytes` (0 or 16) are read: the rest is
// zero, which is how the tail of a row past n_in is filled.
static __device__ __forceinline__ void ares_cp_async16(void* dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(ares_smem_addr(dst)), "l"(src), "r"(bytes));
}

static __device__ __forceinline__ void ares_cp_async8(void* dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(ares_smem_addr(dst)), "l"(src), "r"(bytes));
}

static __device__ __forceinline__ void ares_cp_commit() {
    asm volatile("cp.async.commit_group;");
}

static __device__ __forceinline__ void ares_cp_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// Four 8x8 matrices of 16-bit values (8 rows of 16 bytes each) from shared
// memory into the fragment layout of `mma`: lane l supplies the address of row
// l % 8 of matrix l / 8.
static __device__ __forceinline__ void ares_ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(ares_smem_addr(p)));
}

// The integer products: 16x8 outputs, 32 (or 16) values of K, accumulated from
// zero.
static __device__ __forceinline__ void ares_mma_s8_k32(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int init) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(init));
}

static __device__ __forceinline__ void ares_mma_s8_k16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// 16x8 outputs, 16 values of K, f16 inputs, f32 accumulation (added to c).
static __device__ __forceinline__ void ares_mma_f16_k16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// N consecutive 32-bit words from an address that is only 2-byte aligned (the
// blocks of Q6_K, Q3_K, Q8_0 ... start at multiples of 210, 110, 34 bytes):
// aligned loads plus a funnel shift. The loads stay in L1: the lanes of a warp
// read 4 bytes each from many rows, so a sector is asked for again by the next
// word (and Q6_K's 4 steps of a half read the same high bits). The word past the end is only read when
// the address is not 4-byte aligned, so no load goes beyond the data.
template <int N>
static __device__ __forceinline__ void ares_ld_words_a2(const uint8_t* p, uint32_t (&out)[N]) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* base = reinterpret_cast<const uint32_t*>(a & ~static_cast<uintptr_t>(3));
    const uint32_t shift = (a & 2) * 8;
    uint32_t w[N + 1];
#pragma unroll
    for (int i = 0; i < N; i++) {
        w[i] = __ldg(base + i);
    }
    w[N] = shift != 0 ? __ldg(base + N) : 0u;
#pragma unroll
    for (int i = 0; i < N; i++) {
        out[i] = __funnelshift_r(w[i], w[i + 1], shift);
    }
}

// The block of the 1-D grid: which weight tile and which activation tile. A
// group of `group` activation tiles takes every weight tile in turn, so the
// blocks that run together share their weights (and the activations of the group
// stay in L2).
static __device__ __forceinline__ void ares_mm_tile(uint32_t n_wt, uint32_t n_bt, uint32_t group, uint32_t& wt, uint32_t& bt) {
    const uint32_t per_group = group * n_wt;
    const uint32_t g = blockIdx.x / per_group;
    const uint32_t rem = blockIdx.x - g * per_group;
    const uint32_t size = min(group, n_bt - g * group);
    wt = rem / size;
    bt = g * group + rem % size;
}

// ---------------------------------------------------------------------------
// The shared memory of a block, in both variants: two stages (the weight tile,
// the activation tile and, for the int8 variant, the scales of both), then the
// table of the type. It is dynamic shared memory, so a launch must ask for
// `BYTES` of it (Kernels does, from the same formula).
//
//   T            the weight type (its table)
//   I, J         weight rows and activation rows of the block
//   KC           values of K per step (a multiple of 64)
//   ELEM_BYTES   1 for the int8 tiles, 2 for f16
template <class T, int I, int J, int KC, int ELEM_BYTES>
struct AresMmLayout {
    static constexpr int SUBS = KC / 32;
    static constexpr int PITCH = KC * ELEM_BYTES + 16;
    static constexpr uint32_t A_TILE = I * PITCH;
    static constexpr uint32_t B_TILE = J * PITCH;
    static constexpr uint32_t A_SC = ELEM_BYTES == 1 ? SUBS * I * 8 : 0;
    static constexpr uint32_t B_SC = ELEM_BYTES == 1 ? SUBS * J * 8 : 0;
    static constexpr uint32_t STAGE = A_TILE + B_TILE + A_SC + B_SC;
    static constexpr uint32_t BYTES = 2 * STAGE + T::SMEM_BYTES;
};

extern __shared__ __align__(16) uint8_t ares_dyn_smem[];

// A launch that asked for less shared memory than the kernel lays out would
// read and write beyond its block; stop it instead.
static __device__ __forceinline__ void ares_check_smem(uint32_t needed) {
    uint32_t have;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(have));
    if (have < needed) {
        __trap();
    }
}

// Asks L2 for the line that holds `p`.
static __device__ __forceinline__ void ares_prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// 1.5 * 2^23. A 32-bit integer c with |c| < 2^22 placed in the mantissa of this
// number (the bits of M + c, which `mma` gives when it starts from the bits of M)
// is the float M + c exactly; M comes off again exactly.
#define ARES_MM_MAGIC_BITS 0x4B400000
#define ARES_MM_MAGIC 12582912.0f

// ---------------------------------------------------------------------------
// The int8 variant.
//
//   w    n_out rows of n_in / BLOCK_VALUES blocks
//   aq   n_batch rows of n_in int8 values, then the (d, s) pairs
//        (quantize_act.cuh)
//   y    n_batch rows of n_out f32 values
//
// Per 32 values, for a weight row r and an activation row b:
//   types with a scale per 32 (and maybe a minimum):
//     y += sw[r] * dx[b] * (sum of code * q)  -  mw[r] * s[b]
//   types with a scale per 16:
//     y += d[r] * dx[b] * (s0[r] * (sum over the first 16) + s1[r] * (sum over the last 16))
// where the sums are the integer results of `mma`, converted to f32. This is
// the arithmetic of the decode kernels, so the two families agree up to f32
// rounding.
//
// The conversion of the integer sum c to f32 is where the instructions go (one
// per output and 32 values of K, and `I2F` is not a full-rate instruction), so
// where |c| is below 2^22 (T::MAGIC; the bounds are with the types) it is done
// by the mantissa trick: `mma` starts from the bits of M = 1.5 * 2^23, which
// makes its result the bits of the float M + c, and
//     u = fma(M + c, sw, -(M * sw))
// is sw * c rounded once, exactly as sw * (float)c would be, because the scale
// of every type is an f16 times an integer of at most 6 bits (17 bits), so
// M * sw is exact. One fused multiply-add takes the place of the conversion and
// the multiplication; y += u * dx follows.
template <class T, int I, int J, int WM, int WN, int KC>
static __device__ __forceinline__ void ares_mm_i8(
    const uint8_t* __restrict__ w, const int8_t* __restrict__ aq, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch, uint32_t group, uint32_t prefetch) {
    using Lay = AresMmLayout<T, I, J, KC, 1>;
    constexpr int NT = WM * WN * 32;
    constexpr int TM = I / WM, TN = J / WN, MT = TM / 16, NTL = TN / 8;
    constexpr int SUBS = Lay::SUBS, PITCH = Lay::PITCH;
    constexpr int AITEMS = (I * SUBS + NT - 1) / NT;
    constexpr int PB = KC / 16;
    constexpr int BI = (J * PB + NT - 1) / NT;
    constexpr int SI = (J * SUBS + NT - 1) / NT;
    static_assert(I % (16 * WM) == 0 && J % (16 * WN) == 0, "warp tile");
    static_assert(KC % 64 == 0, "a step holds whole chunks of 64 values");
    constexpr bool MAGIC = T::MAGIC;
    ares_check_smem(Lay::BYTES);

    uint8_t* const table = ares_dyn_smem + 2 * Lay::STAGE;
    auto a_s = [&](int stage) { return ares_dyn_smem + stage * Lay::STAGE; };
    auto b_s = [&](int stage) { return ares_dyn_smem + stage * Lay::STAGE + Lay::A_TILE; };
    auto a_sc = [&](int stage) {
        return reinterpret_cast<float2*>(ares_dyn_smem + stage * Lay::STAGE + Lay::A_TILE + Lay::B_TILE);
    };
    auto b_sc = [&](int stage) { return a_sc(stage) + SUBS * I; };

    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t wm = warp / WN, wn = warp % WN;
    const uint32_t g = lane >> 2, t = lane & 3;
    uint32_t wt, bt;
    ares_mm_tile((n_out + I - 1) / I, (n_batch + J - 1) / J, group, wt, bt);
    const uint32_t row0 = wt * I, col0 = bt * J;

    const uint32_t n_steps = (n_in + KC - 1) / KC;
    const uint32_t groups_per_row = n_in >> 5;
    const size_t row_bytes = static_cast<size_t>(n_in / T::BLOCK_VALUES) * T::BLOCK_BYTES;
    const float2* __restrict__ ds =
        reinterpret_cast<const float2*>(aq + ares_act_ds_offset(static_cast<size_t>(n_batch) * n_in));

    if (T::SMEM_BYTES > 0) {
        T::init(table, tid, NT);
        __syncthreads();
    }

    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int j = 0; j < NTL; j++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[i][j][e] = 0.0f;
            }
        }
    }

    typename T::Raw raw[AITEMS];

    // Everything that does not change from step to step is computed once.
    //
    // The items of the weight tile this thread decodes (32 values of one row):
    // the row of the matrix, which 32 values of the step, where they go in the tile.
    const uint8_t* w_row[AITEMS];
    uint32_t a_off[AITEMS];
    uint32_t sc_off[AITEMS];
    uint32_t a_sub[AITEMS];
    bool a_on[AITEMS];
#pragma unroll
    for (int u = 0; u < AITEMS; u++) {
        const uint32_t item = tid + u * NT;
        const uint32_t r = item / SUBS, sub = item % SUBS;
        a_on[u] = item < I * SUBS;
        a_sub[u] = sub;
        w_row[u] = w + static_cast<size_t>(min(row0 + r, n_out - 1)) * row_bytes;
        a_off[u] = r * PITCH + (sub >> 1) * 64 + (sub & 1) * T::ITEM_STRIDE;
        sc_off[u] = sub * I + r;
    }
    // The pieces of the activation tile this thread copies: 16 bytes of one row, and
    // the (d, s) pair of one group of 32 values of one row.
    const int8_t* b_src[BI];
    uint32_t b_dst[BI];
    uint32_t b_k[BI];
    const float2* s_src[SI];
    uint32_t s_dst[SI];
    uint32_t s_sub[SI];
#pragma unroll
    for (int u = 0; u < BI; u++) {
        const uint32_t item = tid + u * NT;
        const uint32_t c = item / PB, s = item % PB;
        b_src[u] = aq + static_cast<size_t>(min(col0 + c, n_batch - 1)) * n_in + 16 * s;
        b_dst[u] = c * PITCH + 16 * s;
        b_k[u] = 16 * s;
    }
#pragma unroll
    for (int u = 0; u < SI; u++) {
        const uint32_t item = tid + u * NT;
        const uint32_t c = item % J, s = item / J;
        s_src[u] = ds + static_cast<size_t>(min(col0 + c, n_batch - 1)) * groups_per_row + s;
        s_dst[u] = s * J + c;
        s_sub[u] = 32 * s;
    }
    // Whether every step is whole: only rows of 32-value blocks can end in the middle of one.
    const bool whole = (n_in & (KC - 1)) == 0;

    // The raw bytes of the items this thread decodes for step `step`.
    auto load_weights = [&](uint32_t step, typename T::Raw (&dst)[AITEMS]) {
#pragma unroll
        for (int u = 0; u < AITEMS; u++) {
            const uint32_t g32 = step * SUBS + a_sub[u];
            if (a_on[u] && (whole || 32 * g32 < n_in)) {
                dst[u] = T::load(w_row[u], g32 >> 1, g32 & 1);
                if (prefetch != 0) {
                    // Once per group of four steps and row, ask L2 for the bytes of the
                    // next group: the loads of a step are 32 to 70 bytes of each row, which
                    // GDDR6 serves badly; a request for the whole group makes it read lines.
                    const uint32_t chunk = g32 >> 1;
                    const uint32_t ahead = 64 * (chunk + 4);
                    if ((chunk & 3) == 0 && ahead < n_in) {
                        const uint8_t* start = w_row[u] + static_cast<size_t>(ahead / T::BLOCK_VALUES) * T::BLOCK_BYTES;
                        constexpr int GROUP_BYTES = 256 / T::BLOCK_VALUES * T::BLOCK_BYTES;
                        // The two threads of a row share the lines: first, middle and last.
                        if ((g32 & 1) == 0) {
                            ares_prefetch_l2(start);
                            if (GROUP_BYTES > 128) {
                                ares_prefetch_l2(start + GROUP_BYTES - 1);
                            }
                        } else if (GROUP_BYTES > 128) {
                            ares_prefetch_l2(start + 128);
                        }
                    }
                }
            }
        }
    };
    // Decodes them into the tile of `stage`.
    auto store_weights = [&](uint32_t step, int stage, const typename T::Raw (&src)[AITEMS]) {
#pragma unroll
        for (int u = 0; u < AITEMS; u++) {
            const uint32_t g32 = step * SUBS + a_sub[u];
            if (a_on[u]) {
                AresMmDec dec;
                if (whole || 32 * g32 < n_in) {
                    T::decode(src[u], g32 >> 1, g32 & 1, table, dec);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        dec.w[i] = 0;
                    }
                    dec.ent = make_float2(0.0f, 0.0f);
                }
                uint8_t* tile = a_s(stage) + a_off[u];
                *reinterpret_cast<uint4*>(tile) = make_uint4(dec.w[0], dec.w[1], dec.w[2], dec.w[3]);
                *reinterpret_cast<uint4*>(tile + T::RUN_STRIDE) = make_uint4(dec.w[4], dec.w[5], dec.w[6], dec.w[7]);
                a_sc(stage)[sc_off[u]] = dec.ent;
            }
        }
    };
    // The activations of step `step`: KC bytes per row in pieces of 16, and the
    // (d, s) pairs of the row.
    auto load_acts = [&](uint32_t step, int stage) {
#pragma unroll
        for (int u = 0; u < BI; u++) {
            if (tid + u * NT < J * PB) {
                const bool on = whole || KC * step + b_k[u] < n_in;
                ares_cp_async16(b_s(stage) + b_dst[u], b_src[u] + KC * step, on ? 16u : 0u);
            }
        }
#pragma unroll
        for (int u = 0; u < SI; u++) {
            if (tid + u * NT < J * SUBS) {
                const bool on = whole || KC * step + s_sub[u] < n_in;
                ares_cp_async8(b_sc(stage) + s_dst[u], s_src[u] + SUBS * step, on ? 8u : 0u);
            }
        }
    };

    load_acts(0, 0);
    ares_cp_commit();
    load_weights(0, raw);
    store_weights(0, 0, raw);

    for (uint32_t step = 0; step < n_steps; step++) {
        const int stage = step & 1;
        ares_cp_wait_all();
        __syncthreads();
        if (step + 1 < n_steps) {
            load_acts(step + 1, stage ^ 1);
            load_weights(step + 1, raw);
        }
        ares_cp_commit();

        const uint8_t* const a_base = a_s(stage);
        const uint8_t* const b_base = b_s(stage);
        const float2* const asc = a_sc(stage);
        const float2* const bsc = b_sc(stage);
#pragma unroll
        for (int sub = 0; sub < SUBS; sub++) {
            uint32_t a[MT][4];
#pragma unroll
            for (int mi = 0; mi < MT; mi++) {
                ares_ldmatrix_x4(a[mi], a_base + (wm * TM + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * PITCH + sub * 32 + (lane >> 4) * 16);
            }
            uint32_t b[NTL][2];
#pragma unroll
            for (int nj = 0; nj < NTL / 2; nj++) {
                uint32_t r4[4];
                ares_ldmatrix_x4(r4, b_base + (wn * TN + nj * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * PITCH + sub * 32 + ((lane >> 3) & 1) * 16);
                b[2 * nj][0] = r4[0];
                b[2 * nj][1] = r4[1];
                b[2 * nj + 1][0] = r4[2];
                b[2 * nj + 1][1] = r4[3];
            }
            float2 sa[MT][2];
            float neg_m[MT][2];
#pragma unroll
            for (int mi = 0; mi < MT; mi++) {
                sa[mi][0] = asc[sub * I + wm * TM + mi * 16 + g];
                sa[mi][1] = asc[sub * I + wm * TM + mi * 16 + g + 8];
                neg_m[mi][0] = -ARES_MM_MAGIC * sa[mi][0].x;
                neg_m[mi][1] = -ARES_MM_MAGIC * sa[mi][1].x;
            }
            int s0[MT][2], s1[MT][2];
            if (T::SPLIT16) {
#pragma unroll
                for (int mi = 0; mi < MT; mi++) {
#pragma unroll
                    for (int hi = 0; hi < 2; hi++) {
                        const int packed = __float_as_int(sa[mi][hi].y);
                        s0[mi][hi] = static_cast<int>(static_cast<int16_t>(packed & 0xFFFF));
                        s1[mi][hi] = packed >> 16;
                    }
                }
            }
#pragma unroll
            for (int ni = 0; ni < NTL; ni++) {
                // (d, s) of the two activation rows this thread holds in column pair ni.
                const float4 sb = *reinterpret_cast<const float4*>(&bsc[sub * J + wn * TN + ni * 8 + 2 * t]);
#pragma unroll
                for (int mi = 0; mi < MT; mi++) {
                    if (T::SPLIT16) {
                        int c0[4], c1[4];
                        ares_mma_s8_k16(c0, a[mi][0], a[mi][1], b[ni][0]);
                        ares_mma_s8_k16(c1, a[mi][2], a[mi][3], b[ni][1]);
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            const int hi = e >> 1;
                            const float dx = (e & 1) ? sb.z : sb.x;
                            if (MAGIC) {
                                const int total = s1[mi][hi] * c1[e] + (s0[mi][hi] * c0[e] + ARES_MM_MAGIC_BITS);
                                const float u = fmaf(__int_as_float(total), sa[mi][hi].x, neg_m[mi][hi]);
                                acc[mi][ni][e] = fmaf(u, dx, acc[mi][ni][e]);
                            } else {
                                const int total = s0[mi][hi] * c0[e] + s1[mi][hi] * c1[e];
                                acc[mi][ni][e] = fmaf(static_cast<float>(total), sa[mi][hi].x * dx, acc[mi][ni][e]);
                            }
                        }
                    } else {
                        int c[4];
                        ares_mma_s8_k32(c, a[mi], b[ni][0], b[ni][1], MAGIC ? ARES_MM_MAGIC_BITS : 0);
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            const int hi = e >> 1;
                            const float dx = (e & 1) ? sb.z : sb.x;
                            if (MAGIC) {
                                const float u = fmaf(__int_as_float(c[e]), sa[mi][hi].x, neg_m[mi][hi]);
                                acc[mi][ni][e] = fmaf(u, dx, acc[mi][ni][e]);
                            } else {
                                acc[mi][ni][e] = fmaf(static_cast<float>(c[e]), sa[mi][hi].x * dx, acc[mi][ni][e]);
                            }
                            if (T::HAS_MIN) {
                                acc[mi][ni][e] = fmaf(-sa[mi][hi].y, (e & 1) ? sb.w : sb.y, acc[mi][ni][e]);
                            }
                        }
                    }
                }
            }
        }
        if (step + 1 < n_steps) {
            store_weights(step + 1, stage ^ 1, raw);
        }
    }

#pragma unroll
    for (int mi = 0; mi < MT; mi++) {
#pragma unroll
        for (int ni = 0; ni < NTL; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t r = row0 + wm * TM + mi * 16 + g + (e >> 1) * 8;
                const uint32_t c = col0 + wn * TN + ni * 8 + 2 * t + (e & 1);
                if (r < n_out && c < n_batch) {
                    y[static_cast<size_t>(c) * n_out + r] = acc[mi][ni][e];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The f16 variant.
//
//   ax   n_batch rows of n_in f16 values (ares_convert_f16)
//
// The tile of the weights holds `sw * code - mw` rounded to f16, the same
// rounding the CPU emulation in the tests applies to `ares_quant::dequantize`,
// and the products are summed in f32.
template <class T, int I, int J, int WM, int WN, int KC>
static __device__ __forceinline__ void ares_mm_f16(
    const uint8_t* __restrict__ w, const __half* __restrict__ ax, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch, uint32_t group, uint32_t prefetch) {
    using Lay = AresMmLayout<T, I, J, KC, 2>;
    constexpr int NT = WM * WN * 32;
    constexpr int TM = I / WM, TN = J / WN, MT = TM / 16, NTL = TN / 8;
    constexpr int SUBS = Lay::SUBS, PITCH = Lay::PITCH;
    constexpr int AITEMS = (I * SUBS + NT - 1) / NT;
    constexpr int PB = KC / 8;
    constexpr int BI = (J * PB + NT - 1) / NT;
    static_assert(I % (16 * WM) == 0 && J % (16 * WN) == 0, "warp tile");
    static_assert(KC % 64 == 0, "a step holds whole chunks of 64 values");
    ares_check_smem(Lay::BYTES);

    uint8_t* const table = ares_dyn_smem + 2 * Lay::STAGE;
    auto a_s = [&](int stage) { return ares_dyn_smem + stage * Lay::STAGE; };
    auto b_s = [&](int stage) { return ares_dyn_smem + stage * Lay::STAGE + Lay::A_TILE; };

    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t wm = warp / WN, wn = warp % WN;
    const uint32_t g = lane >> 2, t = lane & 3;
    uint32_t wt, bt;
    ares_mm_tile((n_out + I - 1) / I, (n_batch + J - 1) / J, group, wt, bt);
    const uint32_t row0 = wt * I, col0 = bt * J;

    const uint32_t n_steps = (n_in + KC - 1) / KC;
    const size_t row_bytes = static_cast<size_t>(n_in / T::BLOCK_VALUES) * T::BLOCK_BYTES;

    if (T::SMEM_BYTES > 0) {
        T::init(table, tid, NT);
        __syncthreads();
    }

    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int j = 0; j < NTL; j++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[i][j][e] = 0.0f;
            }
        }
    }

    typename T::Raw raw[AITEMS];

    // Everything that does not change from step to step is computed once (see
    // ares_mm_i8).
    const uint8_t* w_row[AITEMS];
    uint32_t a_off[AITEMS];
    uint32_t a_sub[AITEMS];
    bool a_on[AITEMS];
#pragma unroll
    for (int u = 0; u < AITEMS; u++) {
        const uint32_t item = tid + u * NT;
        const uint32_t r = item / SUBS, sub = item % SUBS;
        a_on[u] = item < I * SUBS;
        a_sub[u] = sub;
        w_row[u] = w + static_cast<size_t>(min(row0 + r, n_out - 1)) * row_bytes;
        a_off[u] = r * PITCH + 2 * ((sub >> 1) * 64 + (sub & 1) * T::ITEM_STRIDE);
    }
    const __half* b_src[BI];
    uint32_t b_dst[BI];
    uint32_t b_k[BI];
#pragma unroll
    for (int u = 0; u < BI; u++) {
        const uint32_t item = tid + u * NT;
        const uint32_t c = item / PB, s = item % PB;
        b_src[u] = ax + static_cast<size_t>(min(col0 + c, n_batch - 1)) * n_in + 8 * s;
        b_dst[u] = c * PITCH + 16 * s;
        b_k[u] = 8 * s;
    }
    const bool whole = (n_in & (KC - 1)) == 0;

    auto load_weights = [&](uint32_t step, typename T::Raw (&dst)[AITEMS]) {
#pragma unroll
        for (int u = 0; u < AITEMS; u++) {
            const uint32_t g32 = step * SUBS + a_sub[u];
            if (a_on[u] && (whole || 32 * g32 < n_in)) {
                dst[u] = T::load(w_row[u], g32 >> 1, g32 & 1);
                if (prefetch != 0) {
                    // Once per group of four steps and row, ask L2 for the bytes of the
                    // next group: the loads of a step are 32 to 70 bytes of each row, which
                    // GDDR6 serves badly; a request for the whole group makes it read lines.
                    const uint32_t chunk = g32 >> 1;
                    const uint32_t ahead = 64 * (chunk + 4);
                    if ((chunk & 3) == 0 && ahead < n_in) {
                        const uint8_t* start = w_row[u] + static_cast<size_t>(ahead / T::BLOCK_VALUES) * T::BLOCK_BYTES;
                        constexpr int GROUP_BYTES = 256 / T::BLOCK_VALUES * T::BLOCK_BYTES;
                        // The two threads of a row share the lines: first, middle and last.
                        if ((g32 & 1) == 0) {
                            ares_prefetch_l2(start);
                            if (GROUP_BYTES > 128) {
                                ares_prefetch_l2(start + GROUP_BYTES - 1);
                            }
                        } else if (GROUP_BYTES > 128) {
                            ares_prefetch_l2(start + 128);
                        }
                    }
                }
            }
        }
    };
    // A code as f32: the byte is placed in the mantissa of 2^23 (unsigned) or
    // 1.5 * 2^23 (signed), which makes the exact integer, and the offset comes
    // off again. No integer-to-float conversion is needed.
    auto store_weights = [&](uint32_t step, int stage, const typename T::Raw (&src)[AITEMS]) {
#pragma unroll
        for (int u = 0; u < AITEMS; u++) {
            const uint32_t g32 = step * SUBS + a_sub[u];
            if (a_on[u]) {
                AresMmDec dec;
                if (whole || 32 * g32 < n_in) {
                    T::decode(src[u], g32 >> 1, g32 & 1, table, dec);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        dec.w[i] = 0;
                    }
                    dec.sw[0] = dec.sw[1] = dec.mw[0] = dec.mw[1] = 0.0f;
                }
                uint8_t* tile = a_s(stage) + a_off[u];
#pragma unroll
                for (int run = 0; run < 2; run++) {
                    uint8_t* dst = tile + 2 * run * T::RUN_STRIDE;
#pragma unroll
                    for (int q = 0; q < 2; q++) {
                        // 8 codes: words 4 * run + 2 * q and + 1.
                        uint32_t out[4];
#pragma unroll
                        for (int half_word = 0; half_word < 2; half_word++) {
                            const uint32_t word = dec.w[4 * run + 2 * q + half_word];
#pragma unroll
                            for (int pair = 0; pair < 2; pair++) {
                                const uint32_t b0 = (word >> (16 * pair)) & 0xFF;
                                const uint32_t b1 = (word >> (16 * pair + 8)) & 0xFF;
                                float f0, f1;
                                if (T::SIGNED_CODES) {
                                    f0 = __int_as_float(0x4B400000 + static_cast<int>(static_cast<int8_t>(b0))) - 12582912.0f;
                                    f1 = __int_as_float(0x4B400000 + static_cast<int>(static_cast<int8_t>(b1))) - 12582912.0f;
                                } else {
                                    f0 = __uint_as_float(0x4B000000u | b0) - 8388608.0f;
                                    f1 = __uint_as_float(0x4B000000u | b1) - 8388608.0f;
                                }
                                const __half2 v = __floats2half2_rn(fmaf(dec.sw[run], f0, -dec.mw[run]), fmaf(dec.sw[run], f1, -dec.mw[run]));
                                out[2 * half_word + pair] = *reinterpret_cast<const uint32_t*>(&v);
                            }
                        }
                        *reinterpret_cast<uint4*>(dst + 16 * q) = make_uint4(out[0], out[1], out[2], out[3]);
                    }
                }
            }
        }
    };
    // KC f16 values per row in pieces of 16 bytes.
    auto load_acts = [&](uint32_t step, int stage) {
#pragma unroll
        for (int u = 0; u < BI; u++) {
            if (tid + u * NT < J * PB) {
                const bool on = whole || KC * step + b_k[u] < n_in;
                ares_cp_async16(b_s(stage) + b_dst[u], b_src[u] + KC * step, on ? 16u : 0u);
            }
        }
    };

    load_acts(0, 0);
    ares_cp_commit();
    load_weights(0, raw);
    store_weights(0, 0, raw);

    for (uint32_t step = 0; step < n_steps; step++) {
        const int stage = step & 1;
        ares_cp_wait_all();
        __syncthreads();
        if (step + 1 < n_steps) {
            load_acts(step + 1, stage ^ 1);
            load_weights(step + 1, raw);
        }
        ares_cp_commit();

        const uint8_t* const a_base = a_s(stage);
        const uint8_t* const b_base = b_s(stage);
#pragma unroll
        for (int kk = 0; kk < KC / 16; kk++) {
            uint32_t a[MT][4];
#pragma unroll
            for (int mi = 0; mi < MT; mi++) {
                ares_ldmatrix_x4(a[mi], a_base + (wm * TM + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * PITCH + kk * 32 + (lane >> 4) * 16);
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
#pragma unroll
                for (int mi = 0; mi < MT; mi++) {
                    ares_mma_f16_k16(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
                }
            }
        }
        if (step + 1 < n_steps) {
            store_weights(step + 1, stage ^ 1, raw);
        }
    }

#pragma unroll
    for (int mi = 0; mi < MT; mi++) {
#pragma unroll
        for (int ni = 0; ni < NTL; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t r = row0 + wm * TM + mi * 16 + g + (e >> 1) * 8;
                const uint32_t c = col0 + wn * TN + ni * 8 + 2 * t + (e & 1);
                if (r < n_out && c < n_batch) {
                    y[static_cast<size_t>(c) * n_out + r] = acc[mi][ni][e];
                }
            }
        }
    }
}

// One kernel of a type and a shape. `THREADS` and `BLOCKS` (blocks per SM) go
// to `__launch_bounds__`; the shapes are in prefill.cu.
#define ARES_MM_KERNEL(NAME, THREADS, BLOCKS, VARIANT, T, I, J, WM, WN, KC, ACT)                \
    extern "C" __global__ void __launch_bounds__(THREADS, BLOCKS) NAME(                         \
        const uint8_t* w, const ACT* act, float* y, uint32_t n_in, uint32_t n_out,              \
        uint32_t n_batch, uint32_t group, uint32_t prefetch) {                                   \
        VARIANT<T, I, J, WM, WN, KC>(w, act, y, n_in, n_out, n_batch, group, prefetch);         \
    }
