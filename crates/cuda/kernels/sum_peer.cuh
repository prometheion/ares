// The sum of two devices' partial results over peer access, as one kernel per
// device and no copy.
//
// This is the peer-to-peer AllReduce of Piggidragon's llama.cpp fork
// (ggml-cuda/allreduce-p2p.cu), with f32 on the link so both devices get the
// exact sum. On consumer cards with patched peer access a kernel can write
// into the other device's memory but cannot read from it, so the data is
// pushed: each device writes its partial into the other's receive buffer and
// then a token into the other's flag, waits for the token the other device
// writes into its own flag, and adds. Only local memory is polled, and the
// data crosses the link once, as posted writes.
#pragma once

#include "common.cuh"

// The number of blocks of every launch. Each block has its own flag.
#define ARES_SUM_PEER_BLOCKS 8

// How long a block waits for the peer before it gives up, in polls of the
// flag: minutes, far beyond any real wait for the other device's layer. A
// peer that never arrives is a bug in the caller; the result is then NaN
// everywhere instead of a device that hangs for good.
#define ARES_SUM_PEER_MAX_POLLS 0xF0000000u

// out = data + (the peer's data), n values, grid of ARES_SUM_PEER_BLOCKS blocks.
//
//   data, out    on this device; out may be data
//   peer_recv    the peer's receive buffer for this call's slot (peer memory)
//   peer_flags   the peer's flags (peer memory)
//   own_recv     this device's receive buffer for this call's slot
//   own_flags    this device's flags
//   token        the number of this call; it grows by one per call
//
// A flag counts as arrived when it holds this call's token or a later one
// (the peer may already be a call ahead): in unsigned arithmetic, when it is
// at most 2^31 - 1 ahead of the token.
ARES_KERNEL ares_sum_peer_f32(
    const float* data, float* out, float* peer_recv, uint32_t* peer_flags,
    const float* own_recv, const uint32_t* own_flags, uint32_t n, uint32_t token) {
    __shared__ uint32_t arrived;

    const uint32_t per = (n + gridDim.x - 1) / gridDim.x;
    const uint32_t i0 = blockIdx.x * per;
    const uint32_t i1 = i0 + per < n ? i0 + per : n;

    for (uint32_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        peer_recv[i] = data[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // The data must be on its way before the token that announces it.
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t*>(&peer_flags[blockIdx.x]) = token;

        uint32_t polls = 0;
        uint32_t ok = 1;
        while (*reinterpret_cast<const volatile uint32_t*>(&own_flags[blockIdx.x]) - token > 0x7fffffffu) {
            if (++polls == ARES_SUM_PEER_MAX_POLLS) {
                ok = 0;
                break;
            }
        }
        __threadfence();
        arrived = ok;
    }
    __syncthreads();

    const bool ok = arrived != 0;
    for (uint32_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        out[i] = ok ? data[i] + own_recv[i] : __int_as_float(0x7fc00000);
    }
}
