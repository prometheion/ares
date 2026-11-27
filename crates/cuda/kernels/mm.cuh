// The prefill weight kernels that read the file's layout: quantized matrix times a
// batch of activation rows on the tensor cores, with the weights dequantised tile
// by tile into shared memory. They take any type through a small per-type piece
// (mm_types.cuh) and are the general path; mmr.cuh has the kernels that read a
// rearranged copy of the weights and decode in registers, which are faster where
// they exist (Q4_K so far).
//
// Why a second family
// -------------------
// The decode kernels (mmvq*.cuh) read each weight once and spend a dp4a per four
// weights per activation row, which is the right shape up to 8 rows: the time is
// the weights' bytes over the memory bandwidth. Past a few rows the work is
// arithmetic, and the tensor cores are two orders of magnitude ahead of the CUDA
// cores at it.
//
// Three variants, one set of weight decoders
// ------------------------------------------
// - int8 (`ares_mm_i8_*`): the tile holds the weights' integer codes, the
//   activations are the int8 values ares_quantize_act produced for the decode
//   kernels, the instruction is the integer `mma.m16n8k32` (`k16` for types with a
//   scale per 16 values), and the sub-block scales (and Q4_K's minimums) are applied
//   to the integer result of each 32 values in f32. Error against the exact f32
//   product: about 1.1 % rms (the 8-bit activations).
// - f16 (`ares_mm_f16_*`): the tile holds the dequantised weights as f16, made from
//   the codes and the f16 scales with packed f16x2 arithmetic (ares_mm_codes_to_f16);
//   the activations are f16, divided per row by a power of two so that the largest
//   value is below 1 (ares_convert_f16: no sum can overflow, any magnitude keeps its
//   precision); `mma.m16n8k16` with f32 accumulation. 0.03 % rms.
// - f16 with f16 accumulation (`ares_mm_h*_*`, template argument FLUSH = 4): the same
//   tiles, but `mma` accumulates in f16 and the partial sum of every 4 `mma` (64
//   values of K, at most 64 products of a weight and a value below 1) is added to
//   f32 registers. GeForce cards run that at twice the rate of f32 accumulation.
//   0.05 % rms, 0.12 % at worst on the test data.
// Which one runs is measured per type on every card (Kernels::prefill_path_for and
// src/kernels/paths.rs: int8 is the fast one for most types on a capped card, as the
// measurements below show).
//
// Structure of a thread block
// ---------------------------
// A block computes a tile of I weight rows by J activation rows. The weights are the
// A operand of `mma` (16 rows), the activations the B operand (8 rows): every
// result element then is `y[b][r]` with consecutive r in consecutive lanes of a
// quad, which stores 32-byte segments without staging through shared memory. The K
// dimension goes through in steps of 64 values.
//
//   - Activations: `cp.async` straight into shared memory, double buffered; the rows
//     of a tile are contiguous in global memory.
//   - Weights: a thread loads the raw bytes of a share of a step into registers two
//     steps before they are used; in the f16 variants the decoding of the next
//     step's tile happens inside the loop of `mma` (after the second slab of 16
//     values) so that it overlaps the tensor cores, and all warps take part (a unit
//     is one run of 16 values; a warp decodes one run). One `__syncthreads` per step.
//   - Dynamic shared memory (the launch asks for AresMmLayout::BYTES): 49 KB (int8)
//     to 74 KB (f16) for the large shape, which owns an SM; the medium and small
//     shapes (64 x 64 and 64 x 32, blocks of 4 warps) are for batches of up to 64
//     rows.
//   - The grid is 1-D and ordered so that the blocks running at the same time share
//     weight tiles: a group of `group` activation tiles takes all weight tiles in
//     turn (ares_mm_tile).
//
// What bounds the kernels, and how that is known (RTX 4070 / RTX 3060, Q4_K,
// 17408 x 5120, 1024 rows; no hardware counters are available, so by ablation:
// the same kernel with a part removed)
// ------------------------------------------------------------------------------
// - int8: 73 to 80 TOPS (of 234) / 33.6 (of 102). The loop of a warp is 416
//   instructions per step of which 192 scale the integer results (3 FFMA per output
//   and 32 values), so the four warps of a scheduler issue 1664 instructions while
//   the tensor cores need 1024 cycles: the kernel cannot reach the tensor cores'
//   rate whatever the overlap, and the per-group float scaling is why. Removing the
//   scaling gives 111 TOPS, removing the weight path 90, the activation path 90, an
//   empty skeleton of `mma`, `ldmatrix` and barriers 167.
// - f16 with f32 accumulation: 85 to 88 % of the capped rate of the tensor cores
//   (49 of 58.6 TFLOPS / 19.8 of 25.6). With f16 accumulation (the rate an
//   uncapped card has for f32 accumulation) the same tile kernel got 65 TFLOPS on
//   the RTX 4070, 28 on the RTX 3060; the skeleton alone reaches 103 / 44 of 117 /
//   51.
// - What held the tile kernels back in that form was not arithmetic: with every
//   weight row reading the same address (all loads hit L1) the kernel is 35 %
//   faster. The file's layout puts the 36 to 70 bytes a row needs per step 2880
//   bytes from the next row's, so one load of a warp touches 16 lines, and the load
//   unit, which `ldmatrix` shares, is busy with them. That is what the rearranged
//   form of mmr.cuh removes. (An L2 prefetch of the weights, loading three steps
//   ahead instead of two, and a different order of the decoding changed nothing.)
// - Types with 2-byte aligned blocks (Q6_K, Q3_K, Q8_0, IQ4_NL): their words are
//   read from aligned addresses and shifted, several per sector; with the loads
//   kept in L1 (ares_ld_words_a2) Q6_K is 2.6 times as fast as with the
//   non-allocating loads of the decode kernels. Q4_K's quants are loaded into L1
//   for the same reason (the second warp that decodes the other run of an item
//   reads them again): on the RTX 3060 that is 22.9 against 28.7 TFLOPS.
// - On a card without the cap (an RTX 4060 Ti with gpu-unlock, 1024 rows) the f16
//   path is bound by the same load unit: loading the weights once instead of every
//   step takes Q6_K from 36 to 65 TFLOPS and Q8_0 from 45 to 63 (the decoding and
//   the tile stores still every step), every lane reading the same row 59 and 65,
//   the same loads all hitting L1 no faster for Q6_K. Its ldmatrix, tile stores and
//   activations move twice the int8 path's bytes through that unit, so the types
//   with many load instructions per item are slower on it than on int8 there (1024
//   rows: IQ3_S 23 %, IQ3_XXS 15 %, Q8_0 12 %, IQ4_NL 10 %, IQ4_XS 9 %; Q6_K and Q3_K
//   were 12 and 11 % until their words were read as 16-byte pieces, ares_ld_words_a2), the
//   others as fast (docs/dev/architecture/kernels.md, "Prefill weight kernels"; the path is
//   measured per type, src/kernels/paths.rs).
//
// A type `T` provides (mm_types.cuh):
//   BLOCK_VALUES, BLOCK_BYTES, SMEM_BYTES, init   as in the decode kernels
//   HAS_MIN                the values have an offset per 32: v = sw * code - mw
//   SPLIT16                the scale changes every 16 values (an integer scale
//                          per half of a 32-value item)
//   SIGNED_CODES           the codes are signed bytes (table values, grid values
//                          with signs, Q6_K's code - 32), not 0..255
//   MAGIC                  the integer sums of an item stay below 2^22 (the int8
//                          scaling may use the mantissa trick)
//   ITEM_STRIDE, RUN_STRIDE   where an item's two runs of 16 codes go in the
//                          64-value row of the tile: item * ITEM_STRIDE + run *
//                          RUN_STRIDE
//   Raw load(row, chunk, item)       the global loads of an item
//   void decode(raw, chunk, item, smem, AresMmDec&)   the decoding: 32 codes, one
//                          per byte, the scale entry of the int8 variant and the
//                          scales (sw, mw) of each run for the f16 variants
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
// The f16 variant decodes the weights of the next step after this many of the 4 slabs
// of 16 values of the current one (the loads were asked for a step ago).
#define ARES_MM_DECODE_AT 1
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

// Waits until at most N of the groups committed by this thread are pending.
template <int N>
static __device__ __forceinline__ void ares_cp_wait_group() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
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

// 16x8 outputs, 16 values of K, f16 inputs and f16 accumulation: the result is
// d (two registers of two f16) = a * b (+ c when `ACCUM`). The tensor cores of
// the GeForce cards run this at twice the rate of the f32 accumulation.
template <bool ACCUM>
static __device__ __forceinline__ void ares_mma_f16acc_k16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if (ACCUM) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                     : "+r"(d[0]), "+r"(d[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};"
                     : "=r"(d[0]), "=r"(d[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
    }
}

// c += the four f16 values of d, in f32.
static __device__ __forceinline__ void ares_flush_f16(float (&c)[4], const uint32_t (&d)[2]) {
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&d[0]));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&d[1]));
    c[0] += lo.x;
    c[1] += lo.y;
    c[2] += hi.x;
    c[3] += hi.y;
}

// N consecutive 32-bit words from an address that is only 2-byte aligned (the
// blocks of Q6_K, Q3_K, Q8_0 ... start at multiples of 210, 110, 34 bytes).
//
// PIECES = false: aligned 4-byte loads plus a funnel shift. The loads stay in L1: the
// lanes of a warp read 4 bytes each from many rows, so a sector is asked for again by
// the next word (and Q6_K's 4 steps of a half read the same high bits). The word past
// the end is only read when the address is not 4-byte aligned, so no load goes beyond
// the data.
//
// PIECES = true: the 16-byte aligned pieces that hold the words, then the shift. What
// costs in the large shape of the tile kernels is not the loads' bytes but the lines a
// load of a warp touches: the lanes read from 16 rows, so every load instruction is 16
// lines or more for the load unit (which `ldmatrix` shares), whether a lane reads 4
// bytes or 16. Three 16-byte loads instead of nine 4-byte ones for the 32 codes of an
// item took that shape of Q8_0 from 44 to 48.5 TFLOPS (f16) and 50 to 56.5 TOPS (int8)
// on an RTX 4060 Ti whose tensor cores run f32 accumulation at full rate, Q6_K from 35.5
// to 40.6 and 40.8 to 42.3, Q3_K from 40 to 44 and 44.4 to 45.6, IQ4_NL from 50.5 to
// 52.5 and 56.5 to 59.4. Where the time is that of reading the weights from memory (the
// medium and small shapes) and in the expert kernels, which decode an item in two lanes,
// the pieces were slower for some types (Q3_K at 64 rows 28 against 22 TFLOPS, the down
// kernels of Q3_K 10 to 15 %), so only the large shape asks for them
// (ares_mm_load). For the one or two words of the i-quants, which replace two or four
// 2-byte loads, the pieces made the f16 path of IQ2_S 14 % slower: those keep their
// loads. Only pieces that hold a wanted byte are read, so no load goes beyond the
// 16-byte granule of the data's last byte.
template <int N, bool PIECES = false>
static __device__ __forceinline__ void ares_ld_words_a2(const uint8_t* p, uint32_t (&out)[N]) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if constexpr (!PIECES) {
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
    } else {
        // The pieces the words can reach from an offset of up to 14 bytes into the first.
        constexpr int COUNT = (14 + 4 * N + 15) / 16;
        const uint4* base = reinterpret_cast<const uint4*>(a & ~static_cast<uintptr_t>(15));
        const uint32_t off = a & 15;
        // The words of the pieces, and three more so that every word an offset selects
        // exists.
        uint32_t win[4 * COUNT + 3];
#pragma unroll
        for (int c = 0; c < COUNT; c++) {
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c == 0 || 16 * c < off + 4 * N) {
                v = __ldg(base + c);
            }
            win[4 * c] = v.x;
            win[4 * c + 1] = v.y;
            win[4 * c + 2] = v.z;
            win[4 * c + 3] = v.w;
        }
        win[4 * COUNT] = win[4 * COUNT + 1] = win[4 * COUNT + 2] = 0;
        // The first word is `off / 4` words in: selected without indexing registers by a
        // value (which would put them in local memory), then shifted by the odd two bytes.
        const uint32_t first = off >> 2, shift = (off & 2) * 8;
        uint32_t v[N + 1];
#pragma unroll
        for (int i = 0; i <= N; i++) {
            const uint32_t lo = (first & 1) != 0 ? win[i + 1] : win[i];
            const uint32_t hi = (first & 1) != 0 ? win[i + 3] : win[i + 2];
            v[i] = (first & 2) != 0 ? hi : lo;
        }
#pragma unroll
        for (int i = 0; i < N; i++) {
            out[i] = __funnelshift_r(v[i], v[i + 1], shift);
        }
    }
}

// Whether a type can read its 2-byte aligned words as 16-byte pieces
// (`T::template load<true>`, see ares_ld_words_a2).
template <class T, class = void>
struct AresMmPieces : std::false_type {};
template <class T>
struct AresMmPieces<T, std::void_t<decltype(&T::template load<true>)>> : std::true_type {};

// The global loads of an item: as 16-byte pieces where the type can and `PIECES` (the
// large shape) asks for them, else the type's plain loads.
template <class T, bool PIECES>
static __device__ __forceinline__ typename T::Raw ares_mm_load(const uint8_t* row, uint32_t chunk, uint32_t item) {
    if constexpr (PIECES && AresMmPieces<T>::value) {
        return T::template load<true>(row, chunk, item);
    } else {
        return T::load(row, chunk, item);
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

// The f16 activations (ares_convert_f16): the f16 values from one row to the next.
// A row is rounded up to whole lines of 128 bytes and one or two more, so that the
// pitch is an odd number of lines: the kernels read the same place of 8 to 64 rows
// together, and with rows a power of two apart (4096 values are 8192 bytes) those
// places compete for the same sets of the caches. Measured on an RTX 4070 with a
// 20480 x 4096 Q4_K matrix: 16 rows 0.123 ms with rows 8192 bytes apart, against
// 0.097 ms for the same bytes of a 17408 x 5120 matrix.
static __host__ __device__ __forceinline__ size_t ares_act_f16_pitch(uint32_t n_in) {
    const size_t lines = (static_cast<size_t>(n_in) * 2 + 127) / 128 + 1;
    return (lines | 1) * 64;
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
    // The large shape reads 2-byte aligned words as 16-byte pieces (ares_ld_words_a2).
    constexpr bool PIECES = I * J >= 128 * 128;
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
                dst[u] = ares_mm_load<T, PIECES>(w_row[u], g32 >> 1, g32 & 1);
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
//   ax   n_batch rows of n_in f16 values, ares_act_f16_pitch(n_in) values apart, then
//        (16-byte aligned) one f32 per row:
//        the factor 2^e the row was divided by so that its largest value is at
//        most 1 (ares_convert_f16)
//
// The tile of the weights holds `sw * code - mw` as f16, made by packed f16x2
// arithmetic straight from the codes (below); the products are summed either in
// f32 (FLUSH = 0, the rate of the tensor cores on a card without the GeForce cap)
// or in f16 for FLUSH mma of 16 values of K, after which the f16 partial sums are
// added to f32 registers (the capped cards run that at twice the rate). A partial
// sum is at most 16 * FLUSH products of a weight and a value of at most 1.

// Two f16 values with the bits 0x64XX are 1024 + XX; twice that many bytes to the
// f16 pair a `prmt` makes of two codes.
static __device__ __forceinline__ uint32_t ares_h2_sub(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
static __device__ __forceinline__ uint32_t ares_h2_mul(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
static __device__ __forceinline__ uint32_t ares_h2_fma(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// The 16 values of one run: codes cw[0..3] (four per word, one per byte) as
// f16 pairs out[0..7]: out = sw * code - mw, in order.
template <bool SIGNED>
static __device__ __forceinline__ void ares_mm_codes_to_f16(const uint32_t (&cw)[4], float sw, float mw, uint32_t (&out)[8]) {
    const __half2 s2h = __float2half2_rn(sw), m2h = __float2half2_rn(-mw);
    const uint32_t s2 = *reinterpret_cast<const uint32_t*>(&s2h), m2 = *reinterpret_cast<const uint32_t*>(&m2h);
    const uint32_t bias = SIGNED ? 0x64806480u : 0x64006400u;  // 1152 or 1024
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t w = SIGNED ? (cw[i] ^ 0x80808080u) : cw[i];
        const uint32_t p0 = __byte_perm(w, 0x64646464u, 0x4140);
        const uint32_t p1 = __byte_perm(w, 0x64646464u, 0x4342);
        out[2 * i] = ares_h2_fma(ares_h2_sub(p0, bias), s2, m2);
        out[2 * i + 1] = ares_h2_fma(ares_h2_sub(p1, bias), s2, m2);
    }
}

// Whether a type decodes an item straight to f16 (`T::decode_f16(raw, chunk, item, table,
// out)`, the 32 values as 2 runs of 8 f16 pairs) instead of through the codes and
// scales of `decode`: the types whose f16 decoders are the ones of the expert kernels
// (expertrows_types.cuh), which have no 8-bit form, and the float types.
template <class T, class = void>
struct AresMmDirect : std::false_type {};
template <class T>
struct AresMmDirect<T, std::void_t<decltype(T::decode_f16)>> : std::true_type {};

// The decoding of one item (32 values): the f16 values of its two runs.
template <class T>
static __device__ __forceinline__ void ares_mm_decode_item(
    const typename T::Raw& src, uint32_t chunk, uint32_t item, const uint8_t* table, bool valid, uint32_t (&out)[2][8]) {
    if constexpr (AresMmDirect<T>::value) {
        if (valid) {
            T::decode_f16(src, chunk, item, table, out);
        } else {
#pragma unroll
            for (int run = 0; run < 2; run++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    out[run][i] = 0;
                }
            }
        }
    } else {
        AresMmDec dec;
        if (valid) {
            T::decode(src, chunk, item, table, dec);
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                dec.w[i] = 0;
            }
            dec.sw[0] = dec.sw[1] = dec.mw[0] = dec.mw[1] = 0.0f;
        }
#pragma unroll
        for (int run = 0; run < 2; run++) {
            const uint32_t cw[4] = {dec.w[4 * run], dec.w[4 * run + 1], dec.w[4 * run + 2], dec.w[4 * run + 3]};
            ares_mm_codes_to_f16<T::SIGNED_CODES>(cw, dec.sw[run], dec.mw[run], out[run]);
        }
    }
}

template <class T, int I, int J, int WM, int WN, int KC, int FLUSH = 0>
static __device__ __forceinline__ void ares_mm_f16(
    const uint8_t* __restrict__ w, const __half* __restrict__ ax, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch, uint32_t group, uint32_t prefetch) {
    using Lay = AresMmLayout<T, I, J, KC, 2>;
    constexpr int NT = WM * WN * 32;
    constexpr int TM = I / WM, TN = J / WN, MT = TM / 16, NTL = TN / 8;
    constexpr int SUBS = Lay::SUBS, PITCH = Lay::PITCH;
    // The items of the weight tile this thread decodes (32 values of one row, in two
    // runs of 16); thread tid takes items tid, tid + NT ...
    constexpr int ITEMS = I * SUBS;
    constexpr int AUNITS = (ITEMS + NT - 1) / NT;
    constexpr int PB = KC / 8;
    constexpr int BI = (J * PB + NT - 1) / NT;
    static_assert(I % (16 * WM) == 0 && J % (16 * WN) == 0, "warp tile");
    static_assert(KC % 64 == 0, "a step holds whole chunks of 64 values");
    
    static_assert((KC / 16) % (FLUSH == 0 ? 1 : FLUSH) == 0, "the partial sums end with a step");
    // The large shape reads 2-byte aligned words as 16-byte pieces (ares_ld_words_a2).
    constexpr bool PIECES = I * J >= 128 * 128;
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
    uint32_t hacc[MT][NTL][2];

    typename T::Raw raw[AUNITS];

    // Everything that does not change from step to step is computed once.
    const uint8_t* w_row[AUNITS];
    uint32_t a_off[AUNITS];
    uint32_t a_sub[AUNITS];
    bool a_on[AUNITS];
#pragma unroll
    for (int u = 0; u < AUNITS; u++) {
        const uint32_t unit = tid + u * NT;
        const uint32_t item = unit;
        const uint32_t r = item / SUBS, sub = item % SUBS;
        a_on[u] = unit < ITEMS;
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
        b_src[u] = ax + static_cast<size_t>(min(col0 + c, n_batch - 1)) * ares_act_f16_pitch(n_in) + 8 * s;
        b_dst[u] = c * PITCH + 16 * s;
        b_k[u] = 8 * s;
    }
    const bool whole = (n_in & (KC - 1)) == 0;

    auto load_weights = [&](uint32_t step, typename T::Raw (&dst)[AUNITS]) {
#pragma unroll
        for (int u = 0; u < AUNITS; u++) {
            const uint32_t g32 = step * SUBS + a_sub[u];
            if (a_on[u] && (whole || 32 * g32 < n_in)) {
                dst[u] = ares_mm_load<T, PIECES>(w_row[u], g32 >> 1, g32 & 1);
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
    // Decodes the runs loaded for step `step` into the tile of `stage`.
    auto store_weights = [&](uint32_t step, int stage, const typename T::Raw (&src)[AUNITS]) {
#pragma unroll
        for (int u = 0; u < AUNITS; u++) {
            const uint32_t g32 = step * SUBS + a_sub[u];
            if (a_on[u]) {
                const bool valid = whole || 32 * g32 < n_in;
                uint32_t out[2][8];
                ares_mm_decode_item<T>(src[u], g32 >> 1, g32 & 1, table, valid, out);
                uint8_t* dst = a_s(stage) + a_off[u];
#pragma unroll
                for (int run = 0; run < 2; run++) {
                    *reinterpret_cast<uint4*>(dst + 2 * run * T::RUN_STRIDE) = make_uint4(out[run][0], out[run][1], out[run][2], out[run][3]);
                    *reinterpret_cast<uint4*>(dst + 2 * run * T::RUN_STRIDE + 16) = make_uint4(out[run][4], out[run][5], out[run][6], out[run][7]);
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

    // The raw bytes of the weights are loaded two steps ahead of their use: those of
    // step + 2 are asked for at the start of step, those of step + 1 (asked for one
    // step ago) are decoded in the middle of it into the tile of the next step.
    typename T::Raw raw_b[AUNITS];
    load_acts(0, 0);
    ares_cp_commit();
    load_weights(0, raw);
    store_weights(0, 0, raw);
    if (n_steps > 1) {
        load_weights(1, raw);
    }

    auto step_body = [&](uint32_t step, int stage, typename T::Raw (&rd)[AUNITS], typename T::Raw (&ld)[AUNITS]) {
        ares_cp_wait_all();
        __syncthreads();
        if (step + 1 < n_steps) {
            load_acts(step + 1, stage ^ 1);
        }
        if (step + 2 < n_steps) {
            load_weights(step + 2, ld);
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
                    if constexpr (FLUSH == 0) {
                        ares_mma_f16_k16(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
                    } else {
                        if (kk % FLUSH == 0) {
                            ares_mma_f16acc_k16<false>(hacc[mi][ni], a[mi], b[ni][0], b[ni][1]);
                        } else {
                            ares_mma_f16acc_k16<true>(hacc[mi][ni], a[mi], b[ni][0], b[ni][1]);
                        }
                        if (kk % FLUSH == FLUSH - 1) {
                            ares_flush_f16(acc[mi][ni], hacc[mi][ni]);
                        }
                    }
                }
            }
            if (kk == ARES_MM_DECODE_AT && step + 1 < n_steps) {
                store_weights(step + 1, stage ^ 1, rd);
            }
        }
    };
    for (uint32_t step = 0; step < n_steps; step += 2) {
        step_body(step, 0, raw, raw_b);
        if (step + 1 < n_steps) {
            step_body(step + 1, 1, raw_b, raw);
        }
    }

    // The rows were divided by their scale (ares_convert_f16): put it back.
    const float* const row_scale =
        reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(ax) + ares_act_ds_offset(static_cast<size_t>(n_batch) * ares_act_f16_pitch(n_in) * 2));
#pragma unroll
    for (int ni = 0; ni < NTL; ni++) {
        const uint32_t c0 = col0 + wn * TN + ni * 8 + 2 * t;
        const float s0 = c0 < n_batch ? row_scale[c0] : 0.0f;
        const float s1 = c0 + 1 < n_batch ? row_scale[c0 + 1] : 0.0f;
#pragma unroll
        for (int mi = 0; mi < MT; mi++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t r = row0 + wm * TM + mi * 16 + g + (e >> 1) * 8;
                const uint32_t c = c0 + (e & 1);
                if (r < n_out && c < n_batch) {
                    y[static_cast<size_t>(c) * n_out + r] = acc[mi][ni][e] * ((e & 1) ? s1 : s0);
                }
            }
        }
    }
}

// One kernel of a type and a shape. `THREADS` and `BLOCKS` (blocks per SM) go
// to `__launch_bounds__`; the shapes are in prefill.cu.
#define ARES_MM_KERNEL(NAME, THREADS, BLOCKS, VARIANT, T, I, J, WM, WN, KC, ACT, ...)           \
    extern "C" __global__ void __launch_bounds__(THREADS, BLOCKS) NAME(                         \
        const uint8_t* w, const ACT* act, float* y, uint32_t n_in, uint32_t n_out,              \
        uint32_t n_batch, uint32_t group, uint32_t prefetch) {                                   \
        VARIANT<T, I, J, WM, WN, KC, ##__VA_ARGS__>(w, act, y, n_in, n_out, n_batch, group, prefetch); \
    }

// The activations of the f16 variant: row b of n_in values as f16, divided by a
// power of two chosen so that the row's largest value is below 1 (then no sum of
// the kernel can overflow an f16, and rows of any magnitude keep their precision),
// and, behind the values (at the next multiple of 16 bytes), the f32 factor of each
// row. One block per row, of up to 1024 threads: a thread waits for each of its
// loads, so with 256 threads a row of 17408 values took 12 microseconds on an RTX
// 4070, which a decode step pays once per input of every layer.
ARES_KERNEL ares_convert_f16(const float* x, __half* out, uint32_t n_in, uint32_t n_batch) {
    __shared__ float peak[32];
    const uint32_t b = blockIdx.x;
    const float* row = x + static_cast<size_t>(b) * n_in;
    float m = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        m = fmaxf(m, fabsf(row[i]));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    }
    if ((threadIdx.x & 31) == 0) {
        peak[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    m = peak[0];
    for (uint32_t i = 1; i < blockDim.x / 32; i++) {
        m = fmaxf(m, peak[i]);
    }
    int e = 0;
    if (m > 0.0f && isfinite(m)) {
        frexpf(m, &e);
        e = max(-100, min(100, e));
    }
    const float scale = ldexpf(1.0f, e), inv = ldexpf(1.0f, -e);
    __half* const o = out + static_cast<size_t>(b) * ares_act_f16_pitch(n_in);
    for (uint32_t i = threadIdx.x; i < n_in; i += blockDim.x) {
        o[i] = __float2half_rn(row[i] * inv);
    }
    if (threadIdx.x == 0) {
        reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(out) + ares_act_ds_offset(static_cast<size_t>(n_batch) * ares_act_f16_pitch(n_in) * 2))[b] = scale;
    }
}
