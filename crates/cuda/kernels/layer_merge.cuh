// The merge of a tensor split: what a device does with the partial results of
// a half block.
//
// Under a tensor split every device computes a slice of each half of a block
// (the mixer, the feed-forward part) and gets rows of `d` values that are a
// partial sum. The residual stream of every device then needs the sum of all
// devices' partials, and the next half its norm. These kernels do that in one
// launch per device: add the partials, add the sum to the residual stream,
// normalise it for the next half, and where the next products take 8-bit
// activations quantize it for them. (TensorRT-LLM fuses the residual and the
// norm into its sum across GPUs the same way.)
//
// **On the wire a partial is f16.** A row crosses the link as `d` f16 values
// and one f32 factor: the row is multiplied by a power of two before it is
// rounded, so that its largest magnitude is below 32768, and the factor puts
// it back. The factor is 1 for every row that fits f16 as it is, which is the
// normal case; it exists so that no finite f32 partial can overflow. Every
// device adds the same rounded values in the same order (device order), its
// own included, so all devices hold the bitwise same residual stream. The
// residual stream itself stays f32.
//
// The wire form of `n_rows` rows: the rows' f16 values, then (at the next
// multiple of 4 bytes) one f32 factor per row.
//
// The *wide* wire form carries the rows as f32 instead (and factors of 1, in
// the same place behind them): twice the bytes and the exact sum. It is for
// the model's exact mode, which is a reference and not a way to serve.
//
// Two ways to cross:
// - `ares_partial_to_wire` + a copy by the copy engine + `ares_merge_wire`:
//   for many rows. The copy leaves the processors free, so the device can
//   compute other rows while a partial travels.
// - `ares_merge_push`: for a few rows, with peer access. The push of the
//   llama.cpp fork (see sum_peer.cuh): the kernel writes its rows straight
//   into the peer's memory, then a token, waits for the peer's token in its
//   own memory, and merges. No copy and no event.
//
// All three run one block per row with a thread per four values, which the
// row must be a multiple of 32 of (the groups of the quantized activations),
// and keep the row in registers: `d <= 16 * ARES_MERGE_RESIDENT * blockDim.x`.
#pragma once

#include "common.cuh"
#include "quantize_act.cuh"
#include "layer_norm.cuh"

// float4 chunks a thread keeps in registers.
#define ARES_MERGE_RESIDENT 6

// The most devices whose partials one merge adds.
#define ARES_MERGE_MAX_DEVICES 8

// What a merge does after the sum (the `mode` argument).
#define ARES_MERGE_SUM 0u    // nothing
#define ARES_MERGE_NORM 1u   // the RMS norm of the new rows, as f32 rows
#define ARES_MERGE_QUANT 2u  // the RMS norm, quantized for the weight kernels
#define ARES_MERGE_MODE 3u   // the bits of `mode` that hold one of the above
#define ARES_MERGE_WIDE 256u // the bit of `mode` for the wide wire form

// As ARES_SUM_PEER_MAX_POLLS (sum_peer.cuh): a peer that never arrives gives
// NaN instead of a device that hangs for good.
#define ARES_MERGE_PUSH_MAX_POLLS 0xF0000000u

// Byte offset of the factors behind the values of `n_rows` rows.
static __device__ __forceinline__ size_t ares_wire_factors(uint32_t n_rows, uint32_t d, bool wide) {
    const size_t values = static_cast<size_t>(n_rows) * d;
    return wide ? values * 4 : (values * 2 + 3) & ~static_cast<size_t>(3);
}

// The powers of two for a row whose largest magnitude is `m`: `down` brings
// the row below 32768, `up` puts it back. Both are 1 unless the row would
// overflow f16. A row that is not finite gets a pair that keeps it not finite.
static __device__ __forceinline__ void ares_wire_scale(float m, float* down, float* up) {
    // m = f * 2^e with f in [0.5, 1).
    const int e = static_cast<int>((__float_as_uint(m) >> 23) & 0xffu) - 126;
    const int k = e > 15 ? e - 15 : 0;
    *down = __uint_as_float(static_cast<uint32_t>(127 - k) << 23);
    *up = __uint_as_float(static_cast<uint32_t>(127 + k) << 23);
}

// Four values as f16, in one word.
static __device__ __forceinline__ uint64_t ares_wire_pack(float4 v, float down) {
    const uint64_t a = __half_as_ushort(__float2half_rn(v.x * down));
    const uint64_t b = __half_as_ushort(__float2half_rn(v.y * down));
    const uint64_t c = __half_as_ushort(__float2half_rn(v.z * down));
    const uint64_t e = __half_as_ushort(__float2half_rn(v.w * down));
    return a | (b << 16) | (c << 32) | (e << 48);
}

static __device__ __forceinline__ float4 ares_wire_unpack(uint64_t bits, float up) {
    return make_float4(__half2float(__ushort_as_half(static_cast<unsigned short>(bits))) * up,
                       __half2float(__ushort_as_half(static_cast<unsigned short>(bits >> 16))) * up,
                       __half2float(__ushort_as_half(static_cast<unsigned short>(bits >> 32))) * up,
                       __half2float(__ushort_as_half(static_cast<unsigned short>(bits >> 48))) * up);
}

// Writes float4 `chunk` of a wire form and returns the values as a reader of
// the wire form gets them.
static __device__ __forceinline__ float4 ares_wire_store(uint8_t* wire, size_t chunk, float4 v, float down,
                                                         float up, bool wide) {
    if (wide) {
        reinterpret_cast<float4*>(wire)[chunk] = v;
        return v;
    }
    const uint64_t bits = ares_wire_pack(v, down);
    reinterpret_cast<uint64_t*>(wire)[chunk] = bits;
    return ares_wire_unpack(bits, up);
}

// float4 `chunk` of a wire form whose row has the factor `up`.
static __device__ __forceinline__ float4 ares_wire_load(const uint8_t* wire, size_t chunk, float up, bool wide) {
    if (wide) {
        return reinterpret_cast<const float4*>(wire)[chunk];
    }
    return ares_wire_unpack(reinterpret_cast<const uint64_t*>(wire)[chunk], up);
}

static __device__ __forceinline__ float ares_max4(float4 v) {
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}

static __device__ __forceinline__ float ares_sumsq4(float4 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
}

// The maximum of `v` over all threads of the block, returned to every thread.
// `partial` as for ares_block_sum.
static __device__ __forceinline__ float ares_block_max(float v, float* partial) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    const uint32_t lane = threadIdx.x & 31;
    if (lane == 0) {
        partial[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    const uint32_t n_warps = blockDim.x >> 5;
    float m = lane < n_warps ? partial[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    }
    return m;
}

// The end of a merge: the norm of the row `v` (its sum of squares over this
// thread's values in `ssq`), written as `mode` says. `chunk0` is the index of
// the row's first float4 in the buffers of all rows.
static __device__ __forceinline__ void ares_merge_norm(const float4* v, float ssq, float* partial, uint32_t chunks,
                                                       size_t chunk0, uint32_t d, size_t n_values,
                                                       const float* weight, float* xn, int8_t* act, float eps,
                                                       uint32_t mode) {
    ssq = ares_block_sum(ssq, partial);
    const float scale = rsqrtf(ssq / static_cast<float>(d) + eps);
#pragma unroll
    for (int k = 0; k < ARES_MERGE_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        const bool live = c < chunks;
        float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (live) {
            const float4 wt = reinterpret_cast<const float4*>(weight)[c];
            y = make_float4(v[k].x * scale * wt.x, v[k].y * scale * wt.y, v[k].z * scale * wt.z,
                            v[k].w * scale * wt.w);
        }
        if (mode == ARES_MERGE_QUANT) {
            ares_quant_chunk(y, live, chunk0 + c, act, n_values);
        } else if (live) {
            reinterpret_cast<float4*>(xn)[chunk0 + c] = y;
        }
    }
}

// The wire form of `n_rows` rows of `d` f32 values at `p`, wide or not. One
// block per row.
ARES_KERNEL ares_partial_to_wire(const float* p, uint8_t* wire, uint32_t n_rows, uint32_t d, uint32_t wide) {
    __shared__ float partial[32];
    const uint32_t row = blockIdx.x;
    const uint32_t chunks = d >> 2;
    const size_t chunk0 = static_cast<size_t>(row) * chunks;
    const float4* pr = reinterpret_cast<const float4*>(p) + chunk0;

    float4 a[ARES_MERGE_RESIDENT];
    float m = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_MERGE_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        a[k] = c < chunks ? pr[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        m = fmaxf(m, ares_max4(a[k]));
    }
    m = ares_block_max(m, partial);
    float down = 1.0f, up = 1.0f;
    if (!wide) {
        ares_wire_scale(m, &down, &up);
    }
#pragma unroll
    for (int k = 0; k < ARES_MERGE_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        if (c < chunks) {
            ares_wire_store(wire, chunk0 + c, a[k], down, up, wide != 0);
        }
    }
    if (threadIdx.x == 0) {
        reinterpret_cast<float*>(wire + ares_wire_factors(n_rows, d, wide != 0))[row] = up;
    }
}

// x += the sum of the devices' partials, all in wire form, and what `mode`
// says. One block per row.
//
//   own          this device's partial
//   peers        the other devices' partials, in device order, `peer_stride`
//                bytes apart
//   own_rank     this device's place in the device order
//   weight       the weight of the norm (modes NORM and QUANT)
//   xn           the normalised rows (mode NORM)
//   act          the quantized activations of all rows, in the layout of
//                ares_quantize_act over n_rows * d values (mode QUANT)
ARES_KERNEL ares_merge_wire(float* x, const uint8_t* own, const uint8_t* peers, uint64_t peer_stride,
                            uint32_t n_peers, uint32_t own_rank, uint32_t n_rows, uint32_t d, const float* weight,
                            float* xn, int8_t* act, float eps, uint32_t mode) {
    __shared__ float partial[32];
    const uint32_t row = blockIdx.x;
    const uint32_t chunks = d >> 2;
    const size_t chunk0 = static_cast<size_t>(row) * chunks;
    const bool wide = (mode & ARES_MERGE_WIDE) != 0;
    mode &= ARES_MERGE_MODE;
    const size_t factors = ares_wire_factors(n_rows, d, wide);

    const uint8_t* src[ARES_MERGE_MAX_DEVICES];
    float up[ARES_MERGE_MAX_DEVICES];
    const uint32_t n = n_peers + 1 < ARES_MERGE_MAX_DEVICES ? n_peers + 1 : ARES_MERGE_MAX_DEVICES;
    for (uint32_t r = 0; r < n; r++) {
        src[r] = r == own_rank ? own : peers + static_cast<size_t>(r < own_rank ? r : r - 1) * peer_stride;
        up[r] = reinterpret_cast<const float*>(src[r] + factors)[row];
    }

    float4* xr = reinterpret_cast<float4*>(x) + chunk0;
    float4 v[ARES_MERGE_RESIDENT];
    float ssq = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_MERGE_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (c < chunks) {
            float4 s = ares_wire_load(src[0], chunk0 + c, up[0], wide);
            for (uint32_t r = 1; r < n; r++) {
                const float4 t = ares_wire_load(src[r], chunk0 + c, up[r], wide);
                s = make_float4(s.x + t.x, s.y + t.y, s.z + t.z, s.w + t.w);
            }
            const float4 y0 = xr[c];
            const float4 y = make_float4(y0.x + s.x, y0.y + s.y, y0.z + s.z, y0.w + s.w);
            xr[c] = y;
            v[k] = y;
            ssq += ares_sumsq4(y);
        }
    }
    if (mode == ARES_MERGE_SUM) {
        return;
    }
    ares_merge_norm(v, ssq, partial, chunks, chunk0, d, static_cast<size_t>(n_rows) * d, weight, xn, act, eps, mode);
}

// x += p + (the peer's p), both rounded through the wire form, and what
// `mode` says: the merge of two devices with peer access, as one kernel per
// device. One block per row; each block has its own flag.
//
//   p            this device's partial, f32
//   peer_recv    the peer's receive area for this call (peer memory)
//   peer_flags   the peer's flags, n_flags of them (peer memory)
//   own_recv     this device's receive area for this call
//   own_flags    this device's flags
//   token        the number of this call; it grows by one per call
//   own_first    whether this device is the first in device order
//
// A flag counts as arrived when it holds this call's token or a later one, as
// in ares_sum_peer_f32. The flags of rows this call does not have are set to
// the token as well, so that no flag falls 2^31 calls behind and then looks
// ahead. `blockDim.x >= n_flags`.
ARES_KERNEL ares_merge_push(const float* p, float* x, uint8_t* peer_recv, uint32_t* peer_flags,
                            const uint8_t* own_recv, const uint32_t* own_flags, uint32_t n_flags, uint32_t token,
                            uint32_t own_first, uint32_t n_rows, uint32_t d, const float* weight, float* xn,
                            int8_t* act, float eps, uint32_t mode) {
    __shared__ float partial[32];
    __shared__ uint32_t arrived;
    const uint32_t row = blockIdx.x;
    const uint32_t chunks = d >> 2;
    const size_t chunk0 = static_cast<size_t>(row) * chunks;
    const bool wide = (mode & ARES_MERGE_WIDE) != 0;
    mode &= ARES_MERGE_MODE;
    const size_t factors = ares_wire_factors(n_rows, d, wide);
    const float4* pr = reinterpret_cast<const float4*>(p) + chunk0;

    float4 a[ARES_MERGE_RESIDENT];
    float m = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_MERGE_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        a[k] = c < chunks ? pr[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        m = fmaxf(m, ares_max4(a[k]));
    }
    m = ares_block_max(m, partial);
    float down = 1.0f, up = 1.0f;
    if (!wide) {
        ares_wire_scale(m, &down, &up);
    }
#pragma unroll
    for (int k = 0; k < ARES_MERGE_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        if (c < chunks) {
            // This device adds what the peer will read, not what it computed.
            a[k] = ares_wire_store(peer_recv, chunk0 + c, a[k], down, up, wide);
        }
    }
    if (threadIdx.x == 0) {
        reinterpret_cast<float*>(peer_recv + factors)[row] = up;
    }
    if (row == 0 && threadIdx.x >= n_rows && threadIdx.x < n_flags) {
        *reinterpret_cast<volatile uint32_t*>(&peer_flags[threadIdx.x]) = token;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // The data must be on its way before the token that announces it.
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t*>(&peer_flags[row]) = token;

        uint32_t polls = 0;
        uint32_t ok = 1;
        while (*reinterpret_cast<const volatile uint32_t*>(&own_flags[row]) - token > 0x7fffffffu) {
            if (++polls == ARES_MERGE_PUSH_MAX_POLLS) {
                ok = 0;
                break;
            }
        }
        __threadfence();
        arrived = ok;
    }
    __syncthreads();

    const bool ok = arrived != 0;
    const float peer_up = reinterpret_cast<const float*>(own_recv + factors)[row];
    float4* xr = reinterpret_cast<float4*>(x) + chunk0;
    float ssq = 0.0f;
#pragma unroll
    for (int k = 0; k < ARES_MERGE_RESIDENT; k++) {
        const uint32_t c = threadIdx.x + k * blockDim.x;
        if (c < chunks) {
            const float4 b = ares_wire_load(own_recv, chunk0 + c, peer_up, wide);
            const float4 lo = own_first ? a[k] : b;
            const float4 hi = own_first ? b : a[k];
            const float4 y0 = xr[c];
            float4 y = make_float4(y0.x + (lo.x + hi.x), y0.y + (lo.y + hi.y), y0.z + (lo.z + hi.z),
                                   y0.w + (lo.w + hi.w));
            if (!ok) {
                const float nan = __int_as_float(0x7fc00000);
                y = make_float4(nan, nan, nan, nan);
            }
            xr[c] = y;
            a[k] = y;
            ssq += ares_sumsq4(y);
        }
    }
    if (mode == ARES_MERGE_SUM) {
        return;
    }
    ares_merge_norm(a, ssq, partial, chunks, chunk0, d, static_cast<size_t>(n_rows) * d, weight, xn, act, eps, mode);
}
