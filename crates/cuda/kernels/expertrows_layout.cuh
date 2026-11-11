// What the many-row expert kernels share: the plan of a layer and its device
// layout, and the small PTX wrappers. One module (expertrows.cu), loaded by
// `ExpertRowKernels` (src/experts.rs).
//
// The plan
// --------
// The input is a route record of any number of rows (the first part of the record
// of expert_layout.cuh: a 16-byte header, `ids[n_pairs]` and `weights[n_pairs]`,
// pair p = row * k + j being the j-th pick of the row). The plan sorts the pairs by
// expert, stably, and cuts each expert's run of pairs into **items** of at most
// `tile` pairs:
//
//   header          64 bytes: n_items, n_valid, missing_pairs, token (u32 words)
//   expert_base     (n_experts + 1) u32: the first sorted position of each expert
//   items           max_items * 16 bytes: AresErItem {expert, first, count, 0}
//   pair_row        n_pairs u32: the row of the pair at each sorted position
//   pair_weight     n_pairs f32: its router weight
//   pair_pos        n_pairs u32: the sorted position of each pair (row * k + j),
//                   or ARES_ER_NO_POS for a pair whose expert id was out of range
//   missing         n_experts u32: 1 for an expert that a kernel found without an
//                   address
//   block counts    ceil(n_pairs / 1024) * n_experts u32: scratch of the multi-block
//                   plan (not used when n_pairs <= 1024)
//
// The kernels get the plan's base pointer and the three numbers that fix its layout.
#pragma once

#include "common.cuh"
#include "mmvq.cuh"

#define ARES_ER_THREADS 128
#define ARES_ER_PLAN_THREADS 1024
#define ARES_ER_NO_POS 0xFFFFFFFFu

struct AresErItem {
    uint32_t expert;
    uint32_t first;  // sorted position of the item's first pair
    uint32_t count;  // pairs of the item, 1 to tile
    uint32_t pad;
};

struct AresErHeader {
    uint32_t n_items;
    uint32_t n_valid;
    uint32_t missing_pairs;
    uint32_t token;
    uint32_t reserved[12];
};

static_assert(sizeof(AresErItem) == 16, "the host reads items by offsets");
static_assert(sizeof(AresErHeader) == 64, "the host reads the header by offsets");

static __host__ __device__ __forceinline__ size_t ares_er_align16(size_t x) {
    return (x + 15) & ~static_cast<size_t>(15);
}

// The most items a plan of `n_pairs` pairs over `n_experts` experts can have: one
// per expert that has a pair, and one more for every full tile.
static __host__ __device__ __forceinline__ uint32_t ares_er_max_items(uint32_t n_experts, uint32_t n_pairs, uint32_t tile) {
    return (n_pairs < n_experts ? n_pairs : n_experts) + n_pairs / tile;
}

struct AresErLayout {
    size_t base, items, row, weight, pos, missing, blocks, total;
    uint32_t n_blocks;
};

static __host__ __device__ __forceinline__ AresErLayout ares_er_layout(uint32_t n_experts, uint32_t n_pairs, uint32_t tile) {
    AresErLayout l;
    l.n_blocks = (n_pairs + ARES_ER_PLAN_THREADS - 1) / ARES_ER_PLAN_THREADS;
    l.base = sizeof(AresErHeader);
    l.items = ares_er_align16(l.base + 4 * (static_cast<size_t>(n_experts) + 1));
    l.row = l.items + 16 * static_cast<size_t>(ares_er_max_items(n_experts, n_pairs, tile));
    l.weight = l.row + 4 * static_cast<size_t>(n_pairs);
    l.pos = l.weight + 4 * static_cast<size_t>(n_pairs);
    l.missing = l.pos + 4 * static_cast<size_t>(n_pairs);
    l.blocks = ares_er_align16(l.missing + 4 * static_cast<size_t>(n_experts));
    l.total = l.blocks + (n_pairs > ARES_ER_PLAN_THREADS ? 4 * static_cast<size_t>(l.n_blocks) * n_experts : 0);
    return l;
}

// A plan seen from a kernel.
struct AresErPlan {
    uint8_t* bytes;
    AresErLayout l;

    __device__ __forceinline__ AresErPlan(uint8_t* plan, uint32_t n_experts, uint32_t n_pairs, uint32_t tile)
        : bytes(plan), l(ares_er_layout(n_experts, n_pairs, tile)) {}
    __device__ __forceinline__ AresErHeader* header() const { return reinterpret_cast<AresErHeader*>(bytes); }
    __device__ __forceinline__ uint32_t* expert_base() const { return reinterpret_cast<uint32_t*>(bytes + l.base); }
    __device__ __forceinline__ AresErItem* items() const { return reinterpret_cast<AresErItem*>(bytes + l.items); }
    __device__ __forceinline__ uint32_t* pair_row() const { return reinterpret_cast<uint32_t*>(bytes + l.row); }
    __device__ __forceinline__ float* pair_weight() const { return reinterpret_cast<float*>(bytes + l.weight); }
    __device__ __forceinline__ uint32_t* pair_pos() const { return reinterpret_cast<uint32_t*>(bytes + l.pos); }
    __device__ __forceinline__ uint32_t* missing() const { return reinterpret_cast<uint32_t*>(bytes + l.missing); }
    __device__ __forceinline__ uint32_t* block_counts() const { return reinterpret_cast<uint32_t*>(bytes + l.blocks); }
};

// The f16 activations as Kernels::prepare_prefill_activations writes them
// (PrefillPath::F16, `ares_convert_f16` in mm.cuh): the f16 values of row b start at
// halves b * pitch, and behind the rows, at the next multiple of 16 bytes, is one f32
// per row: the power of two the row was divided by. The pitch is an odd number of
// 128-byte lines; this is `ares_act_f16_pitch` of mm.cuh.
static __host__ __device__ __forceinline__ size_t ares_er_act_pitch(uint32_t n_in) {
    const size_t lines = (static_cast<size_t>(n_in) * 2 + 127) / 128 + 1;
    return (lines | 1) * 64;
}

// ---------------------------------------------------------------------------
// PTX: cp.async, mma, packed f16.

static __device__ __forceinline__ uint32_t ares_er_smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes from global to shared memory, of which `bytes` (16 or 0) are read; the
// rest is zero.
static __device__ __forceinline__ void ares_er_cp16(void* dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(ares_er_smem_addr(dst)), "l"(src), "r"(bytes));
}

static __device__ __forceinline__ void ares_er_cp_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
static __device__ __forceinline__ void ares_er_cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

static __device__ __forceinline__ uint4 ares_er_lds128(const void* p) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(ares_er_smem_addr(p)));
    return v;
}

// 16 x 8 outputs, 16 values of K: f16 inputs, f32 accumulation added to c.
static __device__ __forceinline__ void ares_er_mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The same starting from zero.
static __device__ __forceinline__ void ares_er_mma0(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}

static __device__ __forceinline__ uint32_t ares_er_h2_fma(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

static __device__ __forceinline__ uint32_t ares_er_h2_mul(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

static __device__ __forceinline__ uint32_t ares_er_h2_sub(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// Two f16 with the same value, from an f32.
static __device__ __forceinline__ uint32_t ares_er_h2_of(float v) {
    const uint32_t h = __half_as_ushort(__float2half_rn(v));
    return h | (h << 16);
}

// The bytes of `bytes` selected by `sel` (a PRMT selector) with 0x64 between them:
// two f16 of the form 0x64XX, which are 1024 + XX.
static __device__ __forceinline__ uint32_t ares_er_magic(uint32_t bytes, uint32_t sel) {
    return __byte_perm(bytes, 0x64646464u, sel);
}

// ---------------------------------------------------------------------------
// Block-wide helpers of the plan.

// The exclusive prefix sum of `v` over the threads of the block (a multiple of 32
// threads); `total` receives the sum. `warp_sums` is 32 words of shared memory.
static __device__ __forceinline__ uint32_t ares_er_block_scan(uint32_t v, uint32_t* warp_sums, uint32_t* total) {
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t x = v;
    for (uint32_t d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) {
            x += y;
        }
    }
    if (lane == 31) {
        warp_sums[warp] = x;
    }
    __syncthreads();
    uint32_t offset = 0, sum = 0;
    for (uint32_t w = 0; w < (blockDim.x >> 5); w++) {
        const uint32_t s = warp_sums[w];
        offset += w < warp ? s : 0;
        sum += s;
    }
    __syncthreads();
    *total = sum;
    return offset + x - v;
}

// The rank of the calling thread's pair among the earlier pairs of the block (thread
// order) with the same expert, and the count of each expert of the block in `hist`
// (n_experts words, zeroed by the caller before). `e` is the pair's expert, or
// n_experts or more for a thread without a valid pair. Warps are taken one after the
// other, so the ranks do not depend on the order in which they run.
static __device__ __forceinline__ uint32_t ares_er_block_rank(uint32_t e, uint32_t* hist, uint32_t n_experts) {
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t rank = 0;
    for (uint32_t w = 0; w < (blockDim.x >> 5); w++) {
        __syncthreads();
        if (warp == w) {
            const uint32_t same = __match_any_sync(0xffffffffu, e);
            const uint32_t leader = __ffs(same) - 1;
            uint32_t base = 0;
            if (lane == leader && e < n_experts) {
                base = hist[e];
                hist[e] = base + __popc(same);
            }
            base = __shfl_sync(0xffffffffu, base, leader);
            rank = base + __popc(same & ((1u << lane) - 1u));
        }
    }
    __syncthreads();
    return rank;
}
