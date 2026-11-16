// The attention of the image tower: every patch attends every patch of the
// image, head by head, with no mask and no cache (src/tower.rs).
//
// The language models' attention reads stored pages through tables and wants
// heads of a multiple of 32 values; the tower's heads have 72 and its keys are
// the image's patches, all present at once, up to 16384 of them. So this is a
// kernel of its own, in the shape of FlashAttention 2:
//
//   a block holds 128 queries of one head (8 warps of 16), their fragments in
//   registers, and walks the keys in chunks of 64, the next chunk's keys and
//   values copied into shared memory (cp.async) while this one is computed:
//   S = Q K^T on the tensor cores (f16 operands, f32 sums), the softmax online in
//   f32 (base 2, the scores scaled by scale * log2 e), the probabilities rounded
//   to f16 and summed as rounded, O += P V (f32 sums), and O / l at the end.
//
// Queries, keys and values are f16 rows of `PITCH` values (the head's values,
// then zeros up to a multiple of 16), `heads[part][head][patch]` as
// ares_tower_rotate_heads writes them. Keys past the last patch are masked;
// queries past it are computed and not written.
#pragma once

#include "common.cuh"
#include "attn_ptx.cuh"

#define ARES_TOWER_ATT_WARPS 8
#define ARES_TOWER_ATT_QUERIES (16 * ARES_TOWER_ATT_WARPS)
#define ARES_TOWER_ATT_KEYS 64

// Two 8x8 b16 matrices, transposed; lanes 0 to 15 give the addresses of their rows.
static __device__ __forceinline__ void ares_tower_ldmatrix_x2_trans(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_addr(p))
                 : "memory");
}

// A copy of 16 bytes into shared memory that writes zeros when `valid` is false.
static __device__ __forceinline__ void ares_tower_cp16(void* dst, const void* src, bool valid) {
    const uint32_t bytes = valid ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(bytes)
                 : "memory");
}

// `rows` rows of PITCH halves from `src` (rows `first ..` of `n_valid`) into
// shared memory rows of `PITCH + 8` halves; rows from `n_valid` on are zero.
template <int PITCH>
static __device__ __forceinline__ void ares_tower_load_rows(__half* dst, const __half* src, uint32_t first,
                                                            uint32_t n_valid, int rows) {
    constexpr int CHUNKS = PITCH / 8;  // 16-byte pieces of a row
    for (int i = threadIdx.x; i < rows * CHUNKS; i += blockDim.x) {
        const int r = i / CHUNKS, c = i - r * CHUNKS;
        const bool valid = first + r < n_valid;
        const __half* from = src + (valid ? static_cast<size_t>(first + r) * PITCH + 8 * c : 0);
        ares_tower_cp16(dst + r * (PITCH + 8) + 8 * c, from, valid);
    }
}

// D: values of a head; PITCH: D rounded up to 16. `heads` as above, `out` rows of
// `n_head * D` f32 values per patch. Grid: (patches / 128 rounded up, heads).
template <int D, int PITCH>
static __device__ void ares_tower_attention_body(const __half* heads, float* out, uint32_t n_patches,
                                                 uint32_t n_head, float scale_log2) {
    static_assert(PITCH % 16 == 0 && PITCH >= D && PITCH - D < 16, "PITCH is D rounded up to 16");
    constexpr int SP = PITCH + 8;                 // halves per row in shared memory
    constexpr int KS = PITCH / 16;                // k-steps of Q K^T
    constexpr int NT = D / 8;                     // tiles of 8 values of the output
    constexpr int KEYS = ARES_TOWER_ATT_KEYS;
    constexpr int STAGE = 2 * KEYS * SP;          // halves of one stage: keys, then values
    static_assert(D % 8 == 0, "the output is written in tiles of 8 values");
    static_assert(ARES_TOWER_ATT_QUERIES * SP <= STAGE, "the queries are staged in the second stage");
    extern __shared__ __align__(16) __half smem[];

    const uint32_t head = blockIdx.y, q0 = blockIdx.x * ARES_TOWER_ATT_QUERIES;
    const size_t part = static_cast<size_t>(n_head) * n_patches * PITCH;
    const __half* Q = heads + static_cast<size_t>(head) * n_patches * PITCH;
    const __half* K = Q + part;
    const __half* V = K + part;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const uint32_t n_chunks = (n_patches + KEYS - 1) / KEYS;

    // The first chunk into stage 0, the queries into stage 1, then the queries'
    // fragments into registers.
    ares_tower_load_rows<PITCH>(smem, K, 0, n_patches, KEYS);
    ares_tower_load_rows<PITCH>(smem + KEYS * SP, V, 0, n_patches, KEYS);
    cp_async_commit();
    ares_tower_load_rows<PITCH>(smem + STAGE, Q, q0, n_patches, ARES_TOWER_ATT_QUERIES);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    uint32_t qf[KS][4];
#pragma unroll
    for (int s = 0; s < KS; s++) {
        ldmatrix_x4(qf[s], smem + STAGE + (warp * 16 + (lane & 15)) * SP + s * 16 + (lane >> 4) * 8);
    }
    __syncthreads();

    float o[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) {
        o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (uint32_t c = 0; c < n_chunks; c++) {
        if (c + 1 < n_chunks) {
            __half* next = smem + ((c + 1) & 1) * STAGE;
            ares_tower_load_rows<PITCH>(next, K, (c + 1) * KEYS, n_patches, KEYS);
            ares_tower_load_rows<PITCH>(next + KEYS * SP, V, (c + 1) * KEYS, n_patches, KEYS);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const __half* Ks = smem + (c & 1) * STAGE;
        const __half* Vs = Ks + KEYS * SP;

        // S = Q K^T: 8 tiles of 8 keys.
        float s[KEYS / 8][4];
#pragma unroll
        for (int j = 0; j < KEYS / 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
#pragma unroll
            for (int j = 0; j < KEYS / 8; j += 2) {
                uint32_t b[4];
                const int mi = lane >> 3, r = lane & 7;
                ldmatrix_x4(b, Ks + (8 * j + (mi >> 1) * 8 + r) * SP + ks * 16 + (mi & 1) * 8);
                mma_f16_f32(s[j], qf[ks], b[0], b[1]);
                mma_f16_f32(s[j + 1], qf[ks], b[2], b[3]);
            }
        }

        // The online softmax of rows g and g + 8 of the warp's tile.
        const bool last = (c + 1) * KEYS > n_patches;
        float top0 = -INFINITY, top1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < KEYS / 8; j++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[j][e] * scale_log2;
                if (last && c * KEYS + 8 * j + 2 * t + (e & 1) >= n_patches) {
                    v = -INFINITY;
                }
                s[j][e] = v;
            }
            top0 = fmaxf(top0, fmaxf(s[j][0], s[j][1]));
            top1 = fmaxf(top1, fmaxf(s[j][2], s[j][3]));
        }
        top0 = fmaxf(top0, __shfl_xor_sync(0xffffffffu, top0, 1));
        top0 = fmaxf(top0, __shfl_xor_sync(0xffffffffu, top0, 2));
        top1 = fmaxf(top1, __shfl_xor_sync(0xffffffffu, top1, 1));
        top1 = fmaxf(top1, __shfl_xor_sync(0xffffffffu, top1, 2));
        const float n0 = fmaxf(m0, top0), n1 = fmaxf(m1, top1);
        const float a0 = exp2f(m0 - n0), a1 = exp2f(m1 - n1);
        m0 = n0;
        m1 = n1;
        uint32_t p[KEYS / 8][2];
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int j = 0; j < KEYS / 8; j++) {
            const __half2 lo = __floats2half2_rn(exp2f(s[j][0] - n0), exp2f(s[j][1] - n0));
            const __half2 hi = __floats2half2_rn(exp2f(s[j][2] - n1), exp2f(s[j][3] - n1));
            const float2 lf = __half22float2(lo), hf = __half22float2(hi);
            sum0 += lf.x + lf.y;
            sum1 += hf.x + hf.y;
            p[j][0] = *reinterpret_cast<const uint32_t*>(&lo);
            p[j][1] = *reinterpret_cast<const uint32_t*>(&hi);
        }
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            o[j][0] *= a0;
            o[j][1] *= a0;
            o[j][2] *= a1;
            o[j][3] *= a1;
        }

        // O += P V: 4 k-steps of 16 keys, NT tiles of 8 values.
#pragma unroll
        for (int kk = 0; kk < KEYS / 16; kk++) {
            const uint32_t a[4] = {p[2 * kk][0], p[2 * kk][1], p[2 * kk + 1][0], p[2 * kk + 1][1]};
            const int mi = lane >> 3, r = lane & 7;
#pragma unroll
            for (int j = 0; j + 1 < NT; j += 2) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, Vs + (16 * kk + (mi & 1) * 8 + r) * SP + 8 * j + (mi >> 1) * 8);
                mma_f16_f32(o[j], a, b[0], b[1]);
                mma_f16_f32(o[j + 1], a, b[2], b[3]);
            }
            if (NT % 2 == 1) {
                uint32_t b[4];
                ares_tower_ldmatrix_x2_trans(b, Vs + (16 * kk + (mi & 1) * 8 + r) * SP + 8 * (NT - 1));
                mma_f16_f32(o[NT - 1], a, b[0], b[1]);
            }
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const uint32_t r0 = q0 + warp * 16 + g, r1 = r0 + 8;
    const size_t width = static_cast<size_t>(n_head) * D;
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const uint32_t col = head * D + 8 * j + 2 * t;
        if (r0 < n_patches) {
            *reinterpret_cast<float2*>(out + r0 * width + col) = make_float2(o[j][0] / l0, o[j][1] / l0);
        }
        if (r1 < n_patches) {
            *reinterpret_cast<float2*>(out + r1 * width + col) = make_float2(o[j][2] / l1, o[j][3] / l1);
        }
    }
}

// The shared memory of a launch: two stages of 64 keys and 64 values.
#define ARES_TOWER_ATT_SMEM(PITCH) (2 * 2 * ARES_TOWER_ATT_KEYS * ((PITCH) + 8) * 2)

#define ARES_TOWER_ATTENTION(NAME, D, PITCH)                                                                \
    extern "C" __global__ void __launch_bounds__(32 * ARES_TOWER_ATT_WARPS, 2)                              \
        NAME(const __half* heads, float* out, uint32_t n_patches, uint32_t n_head, float scale_log2) {     \
        ares_tower_attention_body<D, PITCH>(heads, out, n_patches, n_head, scale_log2);                    \
    }
