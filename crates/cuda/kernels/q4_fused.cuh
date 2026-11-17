// Fused kernels of the stream mixer of qwen4exp, and what a model that is
// spread over two devices exchanges between them.
//
// A read of the stream mixer (q4_mixer.cuh has the formulas) is two launches
// here instead of six:
//
//   ares_q4_read_down   the write of the last block's output, the norm of the
//                       streams, and `down @ xn` and `inject @ xn` as one
//                       partial sum per stream
//   ares_q4_read_up     lo = silu(down / hc), gate = up @ lo, the gated mean
//                       and its quantization
//
// The three matrices of a mixer multiply unquantized values. They may be of
// any weight type: each thread decodes the values it multiplies (weight_value.cuh)
// and the sums are those of the same matrix in F32. The two launches come in one
// kernel per reader of `down` and `up`: `ares_q4_read_down` and `ares_q4_read_up`
// for the float types, `_<type>` for the common quantized types (q4_reads.cu) and
// `_any` for every type by its id at run time. `inject` (a few rows, F32 in the
// files) is a float type for all of them but `_any`, which takes it in any type:
// a quantized `inject` takes the `_any` read whatever `down` is. (A call to the
// run-time decoder inside the unrolled read keeps many values live across it:
// the first launch took 110 registers instead of 72 with it, too many for 640
// threads; the `_any` launch is bounded to 1024 threads and spills.)
//
// **Two devices.** One device, the mixer, holds the residual streams and runs
// every read. The other, the worker, gets the mixed input of each half block
// and returns its partial output:
//
//   mixer   ares_q4_read_up writes the activations into the worker's memory
//           as well, and a flag per block behind them
//   worker  ares_q4_swap waits for the flags, computes nothing itself; the
//           worker's products follow it; the next ares_q4_swap writes the
//           worker's partial output into the mixer's memory, then a flag
//   mixer   the next ares_q4_read_down waits for that flag and adds the
//           worker's partial to its own while it writes the streams
//
// The router runs on the mixer. ares_q4_route_split cuts its record into one
// record per device (the expert kernels walk the groups of a record, so each
// device computes the experts of its record) and writes the worker's into the
// worker's memory; ares_q4_route_recv waits for it there.
//
// Everything that crosses is written into the peer's memory by the sender
// (kernels can write peer memory, not read it) and announced by a flag: a u32
// that holds the number of the exchange, or a later one. The data is written,
// and fenced, before its flag.
#pragma once

#include "common.cuh"
#include "expert_layout.cuh"
#include "layer_math.cuh"
#include "quantize_act.cuh"
#include "weight_value.cuh"

// The float types of a mixer's matrices (the `ty` arguments are ggml type ids).
#define ARES_Q4_F32 0u
#define ARES_Q4_F16 1u
#define ARES_Q4_BF16 30u

// The most rows of `down` one block of ares_q4_read_down computes.
#define ARES_Q4_TILE 32
// The most residual streams.
#define ARES_Q4_MAX_STREAMS 8
// float4 chunks of a stream a thread keeps in registers.
#define ARES_Q4_RESIDENT 2
// The largest rank of a mixer.
#define ARES_Q4_MAX_RANK 1024
// As ARES_MERGE_PUSH_MAX_POLLS: a peer that never arrives gives NaN instead
// of a device that hangs for good.
#define ARES_Q4_MAX_POLLS 0xF0000000u

static __device__ __forceinline__ uint32_t ares_q4_value_bytes(uint32_t ty) {
    return ty == ARES_Q4_F32 ? 4u : 2u;
}

// Four consecutive values of a row of a float matrix. `p` is aligned to the
// bytes of the four. (Loads of eight values past the cache, four issued
// before the first is used, as the weight kernels read, were measured too:
// a read in the model took 9 % longer on an RTX 4070 and 16 % longer on an
// RTX 3060 than with these.)
static __device__ __forceinline__ float4 ares_q4_load4(const uint8_t* p, uint32_t ty) {
    if (ty == ARES_Q4_F32) {
        return *reinterpret_cast<const float4*>(p);
    }
    const uint64_t b = *reinterpret_cast<const uint64_t*>(p);
    if (ty == ARES_Q4_BF16) {
        return make_float4(__uint_as_float(static_cast<uint32_t>(b & 0xffffu) << 16),
                           __uint_as_float(static_cast<uint32_t>((b >> 16) & 0xffffu) << 16),
                           __uint_as_float(static_cast<uint32_t>((b >> 32) & 0xffffu) << 16),
                           __uint_as_float(static_cast<uint32_t>(b >> 48) << 16));
    }
    return make_float4(__half2float(__ushort_as_half(static_cast<unsigned short>(b))),
                       __half2float(__ushort_as_half(static_cast<unsigned short>(b >> 16))),
                       __half2float(__ushort_as_half(static_cast<unsigned short>(b >> 32))),
                       __half2float(__ushort_as_half(static_cast<unsigned short>(b >> 48))));
}

// How the read kernels find the values of a matrix of rows of `row_values` values: `at(m, r,
// row_values, first, ty)` is a cursor at value `first` of row `r`, and `load4(cursor, k, ty)`
// gives the values `first + k` to `first + k + 3` (`first` and `k` multiples of 4). The
// offsets are integers of the caller's width, so that the float reader computes its addresses
// as the read did before it took other types (the same instructions).
//
// The float types, by their type at run time.
struct AresQ4Float {
    using Cursor = const uint8_t*;
    static __device__ __forceinline__ Cursor at(const uint8_t* m, size_t r, size_t row_values, size_t first,
                                                uint32_t ty) {
        return m + (r * row_values + first) * ares_q4_value_bytes(ty);
    }
    template <class K>
    static __device__ __forceinline__ float4 load4(Cursor c, K k, uint32_t ty) {
        return ares_q4_load4(c + k * ares_q4_value_bytes(ty), ty);
    }
};

// A type of blocks known at compile time (ggml type id `TY`): the cursor is the row and the
// first value.
template <uint32_t TY>
struct AresQ4Blocks {
    struct Cursor {
        const uint8_t* row;
        uint32_t first;
    };
    static __device__ __forceinline__ Cursor at(const uint8_t* m, size_t r, size_t row_values, size_t first,
                                                uint32_t) {
        return {m + r * (row_values / AresWeight<TY>::VALUES) * AresWeight<TY>::BYTES,
                static_cast<uint32_t>(first)};
    }
    template <class K>
    static __device__ __forceinline__ float4 load4(Cursor c, K k, uint32_t) {
        return ares_weight4<TY>(c.row, c.first + static_cast<uint32_t>(k));
    }
};

// Any weight type, known at run time.
struct AresQ4Any {
    struct Cursor {
        const uint8_t* row;
        uint32_t first;
    };
    static __device__ __forceinline__ bool is_float(uint32_t ty) {
        return ty == ARES_Q4_F32 || ty == ARES_Q4_F16 || ty == ARES_Q4_BF16;
    }
    static __device__ __forceinline__ Cursor at(const uint8_t* m, size_t r, size_t row_values, size_t first,
                                                uint32_t ty) {
        if (is_float(ty)) {
            return {AresQ4Float::at(m, r, row_values, first, ty), 0};
        }
        return {m + r * ares_weight_row_bytes(ty, row_values), static_cast<uint32_t>(first)};
    }
    template <class K>
    static __device__ __forceinline__ float4 load4(Cursor c, K k, uint32_t ty) {
        return is_float(ty) ? AresQ4Float::load4(c.row, k, ty)
                            : ares_weight4_any(c.row, ty, c.first + static_cast<uint32_t>(k));
    }
};

static __device__ __forceinline__ float ares_q4_dot4(float4 a, float4 b) {
    return (a.x * b.x + a.y * b.y) + (a.z * b.z + a.w * b.w);
}

// Waits until `flag` holds `token` or a later number. False when it never did.
static __device__ __forceinline__ bool ares_q4_wait(const uint32_t* flag, uint32_t token) {
    const volatile uint32_t* f = reinterpret_cast<const volatile uint32_t*>(flag);
    uint32_t polls = 0;
    while (*f - token > 0x7fffffffu) {
        if (++polls == ARES_Q4_MAX_POLLS) {
            return false;
        }
    }
    return true;
}

// The first launch of a read. Grid (hc * tiles, rows): block (c * tiles + t,
// row) has stream c of the row and tile t of the rows of `down`, a tile being
// ceil(rank / tiles) rows, at most ARES_Q4_TILE.
// `n / 4 <= ARES_Q4_RESIDENT * blockDim.x`, `blockDim.x >= 64`; n is a
// multiple of 32.
//
//   write (block_out or peer_out not null):
//     r_dst[row][c] = r_src[row][c] + (block_out[row] + peer_out[row]) * 2 * sigmoid(inj[row][c] / hc)
//     `peer_out` is waited for: flag `row` of `peer_flags` must hold `token`.
//     With a norm and more than one tile, r_dst must not be r_src: every tile
//     reads the stream and tile 0 writes it.
//   norm (norm not null):
//     xn[row][c] = rms(r[row][c]) * norm[c]
//     downp[row][c][j] = sum_i down[j][c * n + i] * xn[row][c][i]        j < rank
//     injp[row][c][k]  = sum_i inject[k][c * n + i] * xn[row][c][i]      k < hc, when inject is not null
//   The sums over c of `downp` and `injp` are the two products of the read;
//   ares_q4_read_up adds them.
//
// `downp` is followed by `injp` (at rows * hc * rank values). `shape` holds
// the type of `down` in its lowest byte, the type of `inject` in the next (ggml
// type ids), then hc, then tiles; the grid's height is the number of rows. `W`
// reads `down` and `I` reads `inject`.
template <class W, class I>
static __device__ __forceinline__ void ares_q4_read_down_t(const float* r_src, float* r_dst, const float* block_out,
                                                           const float* peer_out, const uint32_t* peer_flags,
                                                           uint32_t token, const float* inj, const float* norm,
                                                           float* xn, const uint8_t* down, const uint8_t* inject,
                                                           float* downp, uint32_t shape, uint32_t n, uint32_t rank,
                                                           float eps) {
    const uint32_t rows = gridDim.y;
    const uint32_t hc = (shape >> 16) & 0xffu;
    const uint32_t tiles = shape >> 24;
    float* injp = downp + static_cast<size_t>(rows) * hc * rank;
    __shared__ float partial[32];
    __shared__ float sums[32][ARES_Q4_TILE + ARES_Q4_MAX_STREAMS];
    __shared__ uint32_t arrived;
    const uint32_t row = blockIdx.y;
    const uint32_t c = blockIdx.x / tiles;
    const uint32_t tile = blockIdx.x % tiles;
    const uint32_t chunks = n >> 2;
    const size_t at = (static_cast<size_t>(row) * hc + c) * chunks;
    const bool write = block_out != nullptr || peer_out != nullptr;

    bool ok = true;
    if (peer_out != nullptr) {
        if (threadIdx.x == 0) {
            const bool there = ares_q4_wait(&peer_flags[row], token);
            __threadfence();
            arrived = there ? 1u : 0u;
        }
        __syncthreads();
        ok = arrived != 0;
    }
    const float factor =
        write ? 2.0f * ares_sigmoid(inj[static_cast<size_t>(row) * hc + c] / static_cast<float>(hc)) : 0.0f;
    const float4* src4 = reinterpret_cast<const float4*>(r_src) + at;
    const float4* own4 = reinterpret_cast<const float4*>(block_out) + static_cast<size_t>(row) * chunks;
    const float4* peer4 = reinterpret_cast<const float4*>(peer_out) + static_cast<size_t>(row) * chunks;

    float4 v[ARES_Q4_RESIDENT];
    float ssq = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_Q4_RESIDENT; k++) {
        const uint32_t i = threadIdx.x + k * blockDim.x;
        v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < chunks) {
            float4 x = src4[i];
            if (write) {
                float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (block_out != nullptr) {
                    o = own4[i];
                }
                if (peer_out != nullptr) {
                    const float4 p = peer4[i];
                    o = make_float4(o.x + p.x, o.y + p.y, o.z + p.z, o.w + p.w);
                }
                x = make_float4(fmaf(o.x, factor, x.x), fmaf(o.y, factor, x.y), fmaf(o.z, factor, x.z),
                                fmaf(o.w, factor, x.w));
                if (!ok) {
                    const float nan = __int_as_float(0x7fc00000);
                    x = make_float4(nan, nan, nan, nan);
                }
                if (tile == 0) {
                    (reinterpret_cast<float4*>(r_dst) + at)[i] = x;
                }
            }
            v[k] = x;
            ssq += (x.x * x.x + x.y * x.y) + (x.z * x.z + x.w * x.w);
        }
    }
    if (norm == nullptr) {
        return;
    }
    ssq = ares_block_sum(ssq, partial);
    const float scale = rsqrtf(ssq / static_cast<float>(n) + eps);
    const float4* w4 = reinterpret_cast<const float4*>(norm) + static_cast<size_t>(c) * chunks;
#pragma unroll
    for (int k = 0; k < ARES_Q4_RESIDENT; k++) {
        const uint32_t i = threadIdx.x + k * blockDim.x;
        if (i < chunks) {
            const float4 w = w4[i];
            v[k] = make_float4(v[k].x * scale * w.x, v[k].y * scale * w.y, v[k].z * scale * w.z,
                               v[k].w * scale * w.w);
            if (tile == 0) {
                (reinterpret_cast<float4*>(xn) + at)[i] = v[k];
            }
        }
    }

    // The tile's rows of `down`, and in tile 0 the rows of `inject`, against
    // this stream's values: every thread has its values in registers and
    // reads its columns of each row.
    const uint32_t down_ty = shape & 0xffu;
    const uint32_t inject_ty = (shape >> 8) & 0xffu;
    // The rows of a tile: the host cuts `down` into tiles of at most
    // ARES_Q4_TILE rows.
    const uint32_t tile_rows = (rank + tiles - 1) / tiles;
    const uint32_t j0 = tile * tile_rows;
    const uint32_t nj = j0 >= rank ? 0u : (rank - j0 < tile_rows ? rank - j0 : tile_rows);
    const uint32_t nk = tile == 0 && inject != nullptr ? hc : 0u;
    const size_t row_values = static_cast<size_t>(hc) * n;
    const size_t first = static_cast<size_t>(c) * n;
    float acc[ARES_Q4_TILE + ARES_Q4_MAX_STREAMS];
#pragma unroll
    for (int j = 0; j < ARES_Q4_TILE; j++) {
        float a = 0.0f;
        if (static_cast<uint32_t>(j) < nj) {
            const auto w = W::at(down, j0 + j, row_values, first, down_ty);
#pragma unroll
            for (int k = 0; k < ARES_Q4_RESIDENT; k++) {
                const uint32_t i = threadIdx.x + k * blockDim.x;
                if (i < chunks) {
                    a += ares_q4_dot4(W::load4(w, static_cast<size_t>(i) * 4, down_ty), v[k]);
                }
            }
        }
        acc[j] = a;
    }
#pragma unroll
    for (int j = 0; j < ARES_Q4_MAX_STREAMS; j++) {
        float a = 0.0f;
        if (static_cast<uint32_t>(j) < nk) {
            const auto w = I::at(inject, j, row_values, first, inject_ty);
#pragma unroll
            for (int k = 0; k < ARES_Q4_RESIDENT; k++) {
                const uint32_t i = threadIdx.x + k * blockDim.x;
                if (i < chunks) {
                    a += ares_q4_dot4(I::load4(w, static_cast<size_t>(i) * 4, inject_ty), v[k]);
                }
            }
        }
        acc[ARES_Q4_TILE + j] = a;
    }
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < ARES_Q4_TILE + ARES_Q4_MAX_STREAMS; j++) {
        const float s = ares_warp_sum(acc[j]);
        if (lane == 0) {
            sums[warp][j] = s;
        }
    }
    __syncthreads();
    const uint32_t j = threadIdx.x;
    const bool is_down = j < nj;
    const bool is_inject = j >= ARES_Q4_TILE && j < ARES_Q4_TILE + nk;
    if (is_down || is_inject) {
        const uint32_t n_warps = blockDim.x >> 5;
        float s = 0.0f;
        for (uint32_t w = 0; w < n_warps; w++) {
            s += sums[w][j];
        }
        const size_t stream = static_cast<size_t>(row) * hc + c;
        if (is_down) {
            downp[stream * rank + j0 + j] = s;
        } else {
            injp[stream * hc + (j - ARES_Q4_TILE)] = s;
        }
    }
}

// The second launch of a read. Grid (n / 32, rows): block (g, row) computes
// the values [32 g, 32 g + 32) of the row's mixed input and quantizes them as
// one group. blockDim.x = 32 * hc * tpr: `tpr` threads share a row of `up`
// (a power of two up to 32; `rank` is a multiple of 4 * tpr).
//
//   lo[j]      = silu(sum_c downp[row][c][j] / hc)
//   gate[c][i] = sum_j up[c * n + i][j] * lo[j]
//   m[i]       = (1 / hc) * sum_c xn[row][c][i] * sigmoid(gate[c][i])
//   act        the activations of all rows in the layout of ares_quantize_act
//              over rows * n values
//   inj[row][k] = sum_c injp[row][c][k]        when inj is not null
//   peer_act   the same activations once more, in the peer's memory, with
//              flag `row * (n / 32) + g` of `peer_flags` set to `token` behind
//              them; null for none
// `W` reads `up`, of type `up_ty`.
template <class W>
static __device__ __forceinline__ void ares_q4_read_up_t(const float* xn, const float* downp, const uint8_t* up,
                                                         int8_t* act, float* inj, int8_t* peer_act,
                                                         uint32_t* peer_flags, uint32_t token, uint32_t up_ty,
                                                         uint32_t hc, uint32_t n, uint32_t rank, uint32_t tpr) {
    const uint32_t rows = gridDim.y;
    const float* injp = downp + static_cast<size_t>(rows) * hc * rank;
    __shared__ float lo[ARES_Q4_MAX_RANK];
    __shared__ float mixed[ARES_Q4_MAX_STREAMS][32];
    const uint32_t row = blockIdx.y;
    const uint32_t group = blockIdx.x;
    const float count = static_cast<float>(hc);
    for (uint32_t j = threadIdx.x; j < rank; j += blockDim.x) {
        float d = 0.0f;
        for (uint32_t c = 0; c < hc; c++) {
            d += downp[(static_cast<size_t>(row) * hc + c) * rank + j];
        }
        lo[j] = ares_silu(d / count);
    }
    if (group == 0 && inj != nullptr && threadIdx.x < hc) {
        float s = 0.0f;
        for (uint32_t c = 0; c < hc; c++) {
            s += injp[(static_cast<size_t>(row) * hc + c) * hc + threadIdx.x];
        }
        inj[static_cast<size_t>(row) * hc + threadIdx.x] = s;
    }
    __syncthreads();

    // Thread (c, i, part) reads its part of row c * n + 32 g + i of `up`.
    const uint32_t part = threadIdx.x % tpr;
    const uint32_t u = threadIdx.x / tpr;
    const uint32_t i = u & 31;
    const uint32_t c = u >> 5;
    const uint32_t value = group * 32 + i;
    const uint32_t span = rank / tpr;
    const auto w = W::at(up, static_cast<size_t>(c) * n + value, rank, part * span, up_ty);
    const float* l = lo + part * span;
    float g = 0.0f;
    for (uint32_t k = 0; k < span; k += 4) {
        const float4 d = W::load4(w, k, up_ty);
        g += (d.x * l[k] + d.y * l[k + 1]) + (d.z * l[k + 2] + d.w * l[k + 3]);
    }
    for (uint32_t offset = tpr >> 1; offset > 0; offset >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, offset);
    }
    if (part == 0) {
        mixed[c][i] = xn[(static_cast<size_t>(row) * hc + c) * n + value] * ares_sigmoid(g);
    }
    __syncthreads();
    if (threadIdx.x >= 32) {
        return;
    }

    // The first warp has the group: lane i its value i. The arithmetic of
    // ares_quantize_act.
    const uint32_t lane = threadIdx.x;
    float m = 0.0f;
    for (uint32_t s = 0; s < hc; s++) {
        m += mixed[s][lane];
    }
    m /= count;
    float amax = fabsf(m);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float d = amax / 127.0f;
    const float code = amax != 0.0f ? roundf(m / d) : 0.0f;
    const float sum_codes = ares_warp_sum(code);
    // Four codes in a word, written by the first lane of the four.
    uint32_t packed = static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(code))) << (8 * (lane & 3));
    packed |= __shfl_xor_sync(0xffffffffu, packed, 1);
    packed |= __shfl_xor_sync(0xffffffffu, packed, 2);
    const size_t n_values = static_cast<size_t>(rows) * n;
    const size_t index = static_cast<size_t>(row) * n + group * 32 + lane;
#pragma unroll
    for (int target = 0; target < 2; target++) {
        uint8_t* base = reinterpret_cast<uint8_t*>(target == 0 ? act : peer_act);
        if (base != nullptr) {
            if ((lane & 3) == 0) {
                reinterpret_cast<uint32_t*>(base)[index >> 2] = packed;
            }
            reinterpret_cast<float*>(base + ares_act_x_offset(n_values))[index] = m;
            if (lane == 0) {
                reinterpret_cast<float2*>(base + ares_act_ds_offset(n_values))[index >> 5] =
                    make_float2(d, d * sum_codes);
            }
        }
    }
    if (peer_act != nullptr) {
        __syncwarp();
        if (lane == 0) {
            // The data must be on its way before the flag that announces it.
            __threadfence_system();
            *reinterpret_cast<volatile uint32_t*>(&peer_flags[static_cast<size_t>(row) * (n >> 5) + group]) = token;
        }
    }
}

// The two launches of a read for one reader of `down` and `up` (`W`) and one of `inject`
// (`I`): `SUFFIX` names them (`ares_q4_read_down##SUFFIX`, `ares_q4_read_up##SUFFIX`), and
// `BOUNDS` is empty or the launch bounds of the first.
#define ARES_Q4_READS(SUFFIX, W, I, BOUNDS)                                                                   \
    extern "C" __global__ void BOUNDS ares_q4_read_down##SUFFIX(                                              \
        const float* r_src, float* r_dst, const float* block_out, const float* peer_out,                     \
        const uint32_t* peer_flags, uint32_t token, const float* inj, const float* norm, float* xn,           \
        const uint8_t* down, const uint8_t* inject, float* downp, uint32_t shape, uint32_t n, uint32_t rank,  \
        float eps) {                                                                                          \
        ares_q4_read_down_t<W, I>(r_src, r_dst, block_out, peer_out, peer_flags, token, inj, norm, xn, down,  \
                                  inject, downp, shape, n, rank, eps);                                        \
    }                                                                                                         \
    ARES_KERNEL ares_q4_read_up##SUFFIX(const float* xn, const float* downp, const uint8_t* up, int8_t* act,  \
                                        float* inj, int8_t* peer_act, uint32_t* peer_flags, uint32_t token,   \
                                        uint32_t up_ty, uint32_t hc, uint32_t n, uint32_t rank,               \
                                        uint32_t tpr) {                                                       \
        ares_q4_read_up_t<W>(xn, downp, up, act, inj, peer_act, peer_flags, token, up_ty, hc, n, rank, tpr);  \
    }

// The worker's side of an exchange, either half or both. One block per row,
// `n / 4 <= ARES_Q4_RESIDENT * blockDim.x`.
//
//   push (peer_dst not null):  peer_dst[row] = a[row] + b[row] * sigmoid(s[row])
//     in the peer's memory, then flag `row` of `peer_flags` = token_out.
//     `a` or `b` may be null (a zero); without `s` the factor is 1.
//   wait (own_flags not null): until the `n_flags` flags hold `token_in`.
//     Then, each when its output is not null:
//     zero[row] = 0                          (n values)
//     s_out[row] = sum_i inp[i] * m[row][i]   (the scalar gate of the shared expert)
ARES_KERNEL ares_q4_swap(const float* a, const float* b, const float* s, float* peer_dst, uint32_t* peer_flags,
                         uint32_t token_out, const uint32_t* own_flags, uint32_t n_flags, uint32_t token_in,
                         const float* inp, const float* m, float* s_out, float* zero, uint32_t rows,
                         uint32_t n) {
    __shared__ float partial[32];
    __shared__ uint32_t arrived;
    const uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const uint32_t chunks = n >> 2;
    const size_t at = static_cast<size_t>(row) * chunks;
    if (peer_dst != nullptr) {
        const float gate = s != nullptr ? ares_sigmoid(s[row]) : 1.0f;
        for (uint32_t i = threadIdx.x; i < chunks; i += blockDim.x) {
            float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (a != nullptr) {
                x = (reinterpret_cast<const float4*>(a) + at)[i];
            }
            if (b != nullptr) {
                const float4 y = (reinterpret_cast<const float4*>(b) + at)[i];
                x = make_float4(fmaf(y.x, gate, x.x), fmaf(y.y, gate, x.y), fmaf(y.z, gate, x.z),
                                fmaf(y.w, gate, x.w));
            }
            (reinterpret_cast<float4*>(peer_dst) + at)[i] = x;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            *reinterpret_cast<volatile uint32_t*>(&peer_flags[row]) = token_out;
        }
    }
    if (own_flags == nullptr) {
        return;
    }
    if (threadIdx.x == 0) {
        arrived = 1u;
    }
    __syncthreads();
    bool mine = true;
    for (uint32_t f = threadIdx.x; f < n_flags; f += blockDim.x) {
        mine = ares_q4_wait(&own_flags[f], token_in) && mine;
    }
    if (!mine) {
        arrived = 0u;
    }
    __threadfence();
    __syncthreads();
    const bool ok = arrived != 0;
    if (zero != nullptr) {
        for (uint32_t i = threadIdx.x; i < chunks; i += blockDim.x) {
            (reinterpret_cast<float4*>(zero) + at)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    if (s_out != nullptr) {
        float sum = 0.0f;
        for (uint32_t i = threadIdx.x; i < chunks; i += blockDim.x) {
            sum += ares_q4_dot4(reinterpret_cast<const float4*>(inp)[i], (reinterpret_cast<const float4*>(m) + at)[i]);
        }
        sum = ares_block_sum(sum, partial);
        if (threadIdx.x == 0) {
            s_out[row] = ok ? sum : __int_as_float(0x7fc00000);
        }
    }
}

// Cuts the route record of a layer (expert_layout.cuh) into the record of
// this device and the record of its peer: group g goes to the peer when
// `owner[expert]` is not 0. Both records keep the picks and weights per row;
// their groups are the device's own, with `first_pair` counted over them.
// The peer's record is written into the peer's memory with -1 for every slot
// (its own table says where its experts are: ares_q4_route_recv), then
// `peer_flag` = token. One block.
//
//   zero   `zero_values` f32 (a multiple of 4) that are cleared first, or
//          null: the buffer the device's experts add their outputs to
//   cap    n_rows * k of the record
ARES_KERNEL ares_q4_route_split(const uint8_t* record, const uint8_t* owner, uint8_t* own, uint8_t* peer,
                                uint32_t* peer_flag, uint32_t token, float* zero, uint32_t zero_values,
                                uint32_t cap) {
    if (zero != nullptr) {
        for (uint32_t i = threadIdx.x; i < zero_values >> 2; i += blockDim.x) {
            reinterpret_cast<float4*>(zero)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    const uint32_t* src = reinterpret_cast<const uint32_t*>(record + sizeof(AresRouteHeader));
    for (uint32_t i = threadIdx.x; i < 2 * cap; i += blockDim.x) {
        const uint32_t word = src[i];
        reinterpret_cast<uint32_t*>(own + sizeof(AresRouteHeader))[i] = word;
        reinterpret_cast<uint32_t*>(peer + sizeof(AresRouteHeader))[i] = word;
    }
    if (threadIdx.x == 0) {
        const AresRouteHeader head = *ares_route_header(record);
        const AresExpertGroup* groups = ares_route_groups(record, cap);
        AresExpertGroup* mine = const_cast<AresExpertGroup*>(ares_route_groups(own, cap));
        AresExpertGroup* theirs = const_cast<AresExpertGroup*>(ares_route_groups(peer, cap));
        uint32_t n_mine = 0, n_theirs = 0, pairs_mine = 0, pairs_theirs = 0, missing = 0;
        for (uint32_t g = 0; g < head.n_groups; g++) {
            AresExpertGroup group = groups[g];
            const uint32_t pairs = __popc(group.rows);
            if (owner[group.expert] != 0) {
                group.first_pair = pairs_theirs;
                group.slot = -1;
                theirs[n_theirs++] = group;
                pairs_theirs += pairs;
            } else {
                group.first_pair = pairs_mine;
                missing += group.slot < 0 ? 1u : 0u;
                mine[n_mine++] = group;
                pairs_mine += pairs;
            }
        }
        AresRouteHeader out = head;
        out.n_groups = n_mine;
        out.n_missing = missing;
        *reinterpret_cast<AresRouteHeader*>(own) = out;
        out.n_groups = n_theirs;
        out.n_missing = 0;
        *reinterpret_cast<AresRouteHeader*>(peer) = out;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t*>(peer_flag) = token;
    }
}

// Waits for the record ares_q4_route_split wrote here, then gives every
// group its slot from this device's table and counts the experts without
// one. A record that never arrives becomes a record without groups. One
// block; one thread works.
ARES_KERNEL ares_q4_route_recv(uint8_t* record, const int32_t* slot_of, const uint32_t* flag, uint32_t token,
                               uint32_t cap) {
    if (threadIdx.x != 0) {
        return;
    }
    const bool ok = ares_q4_wait(flag, token);
    __threadfence();
    AresRouteHeader* head = reinterpret_cast<AresRouteHeader*>(record);
    if (!ok) {
        head->n_groups = 0;
        head->n_missing = 0;
        head->n_pairs = cap;
        return;
    }
    AresExpertGroup* groups = const_cast<AresExpertGroup*>(ares_route_groups(record, cap));
    uint32_t missing = 0;
    for (uint32_t g = 0; g < head->n_groups; g++) {
        const int32_t slot = slot_of[groups[g].expert];
        groups[g].slot = slot;
        missing += slot < 0 ? 1u : 0u;
    }
    head->n_missing = missing;
}
