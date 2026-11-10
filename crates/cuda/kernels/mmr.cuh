// The weight kernels over a device layout: one family for every number of activation
// rows, from one copy of the matrix.
//
// A device layout (mmr_codes.cuh) is a matrix
// rearranged once at load, in the same number of bytes as the file's, into the order
// the tensor cores read it: the codes of a tile of 16 rows and a step of 64 values of
// K lie together, each lane's share in the order of the registers of its `mma`
// A fragment, and the scales of the 16 rows lie together beside them. A warp reads
// its weights with one load per lane, tile and step, decodes them in registers
// (Marlin's way, vLLM csrc/quantization/marlin) and hands them to `mma`; nothing of
// the weights passes through shared memory. The file's layout cannot do that: there
// the 144 bytes of one Q4_K block of one row lie 2880 bytes from the next row's, so
// the load of a warp touches 16 lines.
//
// The kernel is `mma.m16n8k16` with the weights as the 16-row operand and 8
// activation rows as the other (so one to eight rows pad nothing), f16 inputs. What
// changes with the number of rows is only how many fragments of activations a warp
// multiplies its weights with and how they reach it:
//
//   rows     block               a warp computes          activations
//   1..8     4 warps,  32 x 8    2 tiles x 1 fragment     read by each lane (L1)
//   9..16    4 warps,  32 x 16   2 tiles x 2 fragments    read by each lane (L1)
//   17..32   4 warps, 128 x 32   2 tiles x 4 fragments    cp.async to shared, ldmatrix
//   33..     4 warps, 128 x 64   2 tiles x 8 fragments    cp.async to shared, ldmatrix
//
// In the two shapes of up to 16 rows the four warps of a block compute the same two
// tiles, each a quarter of K, and add their parts at the end. A warp waits for one
// load per step, so a tile computed by one warp over all of K takes the latency of a
// load times the steps of K however little else the device has to do: a pass over
// the matrices of Qwen3.8-27B with one row took 34.9 ms that way (its matrices of
// 5120 rows by 17408 values are 272 steps long) against 22.0 ms with the integer
// kernels, which split K over the lanes of a warp.
//
// Up to 16 rows the kernel waits for the weights: it runs at the speed of the memory
// like the integer kernels of mmvq.cuh, with f32 accumulation, so its result is the
// exact product of the f16 weights and f16 activations. Above, the tensor cores
// bound it; there each shape exists with f32 accumulation (the rate of a card
// without the GeForce cap) and with f16 accumulation added to f32 registers after
// every 4 `mma` (64 values of K), which the capped cards run at twice the rate.
// Which shape serves which number of rows is measured per device (src/kernels.rs).
//
// What was measured on the way (RTX 4070 / RTX 3060, Q4_K 17408 x 5120):
// - Two tiles per warp, each fragment of activations multiplied with both: half the
//   `ldmatrix` per `mma` of one tile per warp, 86 against 78 TFLOPS at 1024 rows.
//   Four tiles per warp need more registers than a thread has.
// - Activations: `cp.async` in three stages beat 16-byte loads into registers that
//   are stored to shared memory a step later (86 against 81).
// - Rows past the end of the batch are filled with zeros (`cp.async` of size 0), not
//   read again from the last row: 24 rows took 0.120 ms instead of 0.098 otherwise.
// - Up to 16 rows the codes are read past L1 (`no_allocate`): the weights are used
//   once and would evict the activations (0.109 -> 0.096 ms at one row).
// - Blocks of a wide product that share a weight tile (`group`, see mm.cuh) read
//   their codes two steps ahead through L1.
// - The kernel of 1 to 8 rows is held to 102 registers (5 blocks of 4 warps per SM):
//   more resident warps keep more loads in flight on the RTX 3060 (0.154 -> 0.152 ms
//   at one row); fewer registers than that spill.
// - A block computes its tile over all of K, and the grid has a block per tile. When
//   there are more blocks than the device holds at once, the ones that do not fit
//   run when the others are done, each at the speed of one stream of loads and not
//   of the memory: a 20480 x 4096 Q4_K matrix at 16 rows (320 blocks, 276 at once
//   on the RTX 4070) takes 0.123 ms where a matrix whose blocks all fit takes 0.092
//   for the same bytes. Blocks that walk several tiles with a grid that is an even
//   share of them fixed that (0.095) and cost the kernels that the tensor cores
//   bound a quarter of their speed (fewer blocks at once), so it was not kept. What
//   would fix both is Marlin's stripes: blocks that walk K in pieces and add their
//   parts. Not built.
// - When the f16 sums are added to f32 (right after each `mma`, after the slab, or a
//   period later from a second set of registers) made no difference.
#pragma once

#include <type_traits>

#include "mm_types.cuh"

static __device__ __forceinline__ void ares_ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(ares_smem_addr(p)));
}

// The dynamic shared memory of a block: three stages of J activation rows of 64 f16
// values, each row 16 bytes longer so that `ldmatrix` finds no bank conflicts. None
// when the lanes read the activations themselves.
template <int J, bool DIRECT>
struct AresMmrShared {
    static constexpr int STAGES = 3;
    static constexpr int PITCH = 64 * 2 + 16;
    static constexpr uint32_t BYTES = DIRECT ? 0 : STAGES * J * PITCH;
};

// y[b][r] = sum over k of W[r][k] * x[b][k].
//
//   w    the matrix in the device layout of `T`
//   ax   n_batch rows of n_in f16 values, ares_act_f16_pitch(n_in) values apart, then
//        (16-byte aligned) one f32 per row: the power of two the row was divided by
//        (ares_convert_f16)
//   y    n_batch rows of n_out f32 values
//
// A block is WM warps and computes NB * 8 activation rows; a warp TA tiles of 16
// weight rows by all NB fragments. With the activations in shared memory the warps
// of a block compute different tiles (WM * TA * 16 weight rows) in step; DIRECT, they
// compute the same TA tiles, warp w the super-blocks of K from w / WM to (w + 1) / WM
// of them, and the block adds their parts. FLUSH = 0: f32 accumulation; else f16
// accumulation, added to f32 after every FLUSH `mma`.
// Grid: ceil(n_out / I) * ceil(n_batch / J) blocks, in groups of `group` activation
// tiles that take every weight tile in turn (ares_mm_tile).
//
// A layout `T` provides, for what one lane holds:
//   struct Warp                 where the lane's share of the warp's first tile is
//   struct Tile                 the registers of one tile
//   open_warp(Warp&, w, n_out, n_in, first, lane)
//   open(Tile&, const Warp&, delta, sb)    the tile `first + delta`, to start at
//                                          super-block sb
//   super_block(Tile&, const Warp&, sb)    before step 4 sb
//   step(Tile&, const Warp&, step)         before the products of a step
//   fragment<j, kk>(Tile&, a)   the A fragment of slab kk (16 values of K) of step j
//                               of the super-block, as four f16x2 registers
template <class T, int WM, int TA, int NB, int FLUSH, bool DIRECT>
static __device__ __forceinline__ void ares_mmr(
    const uint8_t* __restrict__ w, const __half* __restrict__ ax, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch, uint32_t group) {
    constexpr int NT = WM * 32;
    constexpr int I = (DIRECT ? 1 : WM) * TA * 16, J = NB * 8;
    constexpr int STAGES = AresMmrShared<J, DIRECT>::STAGES;
    constexpr int PITCH = AresMmrShared<J, DIRECT>::PITCH;
    constexpr int PIECES = J * 8;  // pieces of 16 bytes of activations per step
    constexpr int BI = (PIECES + NT - 1) / NT;
    static_assert(FLUSH == 0 || 4 % FLUSH == 0, "a flush ends with a step");
    static_assert(NB == 1 || NB % 2 == 0, "fragments are loaded in pairs");
    static_assert(NT % 8 == 0, "a thread moves the same part of each of its rows");
    if constexpr (!DIRECT) {
        ares_check_smem(AresMmrShared<J, DIRECT>::BYTES);
    }

    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3;
    uint32_t wt, bt;
    ares_mm_tile((n_out + I - 1) / I, (n_batch + J - 1) / J, group, wt, bt);
    const uint32_t col0 = bt * J;
    const uint32_t n_tiles = (n_out + 15) / 16, n_steps = n_in / 64, n_sb = n_in / 256;
    const uint32_t tile0 = (DIRECT ? wt : wt * WM + warp) * TA;
    // The super-blocks of K this warp computes.
    const uint32_t sb_lo = DIRECT ? warp * n_sb / WM : 0;
    const uint32_t sb_hi = DIRECT ? (warp + 1) * n_sb / WM : n_sb;
    uint8_t* const bs = ares_dyn_smem;
    const size_t pitch = ares_act_f16_pitch(n_in);

    // The tiles past the end of the matrix repeat the last one; their results are not
    // written.
    typename T::Warp wp;
    typename T::Tile tl[TA];
    const uint32_t first = min(tile0, n_tiles - 1);
    T::open_warp(wp, w, n_out, n_in, first, lane);
    if (sb_lo < sb_hi) {
#pragma unroll
        for (int a = 0; a < TA; a++) {
            T::open(tl[a], wp, min(tile0 + a, n_tiles - 1) - first, sb_lo);
        }
    }

    // The activations of a step are PIECES pieces of 16 bytes, piece i the part i % 8 of
    // row i / 8. Thread tid moves the pieces tid + u NT: the same part of rows NT / 8
    // apart. Rows past the end of the batch are filled with zeros.
    const __half* b_src[BI];
    uint32_t b_dst0 = 0, b_valid = 0;
    const __half* arow[NB];
    if constexpr (DIRECT) {
        // Lane (g, t) holds of fragment ni the values 2t, 2t + 1, 2t + 8, 2t + 9 of each
        // slab of row 8 ni + g.
#pragma unroll
        for (int ni = 0; ni < NB; ni++) {
            arow[ni] = ax + static_cast<size_t>(min(col0 + ni * 8 + g, n_batch - 1)) * pitch + 2 * t;
        }
    } else {
        b_dst0 = (tid / 8) * PITCH + 16 * (tid % 8);
#pragma unroll
        for (int u = 0; u < BI; u++) {
            const uint32_t c = (tid + u * NT) / 8;
            b_src[u] = ax + static_cast<size_t>(min(col0 + c, n_batch - 1)) * pitch + 8 * (tid % 8);
            b_valid |= (tid + u * NT < PIECES && col0 + c < n_batch) ? (1u << u) : 0u;
        }
    }
    auto load_acts = [&](uint32_t step, int stage) {
#pragma unroll
        for (int u = 0; u < BI; u++) {
            if (tid + u * NT < PIECES) {
                ares_cp_async16(bs + stage * J * PITCH + b_dst0 + u * (NT / 8) * PITCH, b_src[u] + 64 * step, ((b_valid >> u) & 1) * 16u);
            }
        }
    };

    float acc[TA][NB][4];
    uint32_t hacc[TA][NB][2];
#pragma unroll
    for (int a = 0; a < TA; a++) {
#pragma unroll
        for (int ni = 0; ni < NB; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[a][ni][e] = 0.0f;
            }
        }
    }

    if constexpr (!DIRECT) {
#pragma unroll
        for (int p = 0; p < STAGES - 1; p++) {
            if (p < n_steps) {
                load_acts(p, p);
            }
            ares_cp_commit();
        }
    }

    for (uint32_t sb = sb_lo; sb < sb_hi; sb++) {
#pragma unroll
        for (int a = 0; a < TA; a++) {
            T::super_block(tl[a], wp, sb);
        }
        // One step of 64 values of K; j, its place in the super-block, is a constant so
        // that a layout selects its scales with constant shifts.
        auto step_fn = [&](auto jc) {
            constexpr int j = decltype(jc)::value;
            const uint32_t step = sb * 4 + j;
            if constexpr (!DIRECT) {
                ares_cp_wait_group<STAGES - 2>();
                __syncthreads();
                if (step + STAGES - 1 < n_steps) {
                    load_acts(step + STAGES - 1, (step + STAGES - 1) % STAGES);
                }
                ares_cp_commit();
            }
#pragma unroll
            for (int a = 0; a < TA; a++) {
                T::step(tl[a], wp, step);
            }
            const uint8_t* const b_base = bs + (step % STAGES) * (J * PITCH);
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                // The B fragments of the slab: register 0 holds the values 2t, 2t + 1 and
                // register 1 the values 2t + 8, 2t + 9 of activation row g of the fragment.
                uint32_t b[NB][2];
                if constexpr (DIRECT) {
#pragma unroll
                    for (int ni = 0; ni < NB; ni++) {
                        const uint32_t* p = reinterpret_cast<const uint32_t*>(arow[ni] + step * 64 + kk * 16);
                        b[ni][0] = __ldg(p);
                        b[ni][1] = __ldg(p + 4);
                    }
                } else if constexpr (NB == 1) {
                    ares_ldmatrix_x2(b[0], b_base + (lane & 7) * PITCH + kk * 32 + ((lane >> 3) & 1) * 16);
                } else {
#pragma unroll
                    for (int nj = 0; nj < NB / 2; nj++) {
                        uint32_t r4[4];
                        ares_ldmatrix_x4(r4, b_base + (nj * 16 + (lane & 7) + ((lane >> 4) & 1) * 8) * PITCH + kk * 32 + ((lane >> 3) & 1) * 16);
                        b[2 * nj][0] = r4[0];
                        b[2 * nj][1] = r4[1];
                        b[2 * nj + 1][0] = r4[2];
                        b[2 * nj + 1][1] = r4[3];
                    }
                }
#pragma unroll
                for (int a = 0; a < TA; a++) {
                    uint32_t af[4];
                    if (kk == 0) {
                        T::template fragment<j, 0>(tl[a], af);
                    } else if (kk == 1) {
                        T::template fragment<j, 1>(tl[a], af);
                    } else if (kk == 2) {
                        T::template fragment<j, 2>(tl[a], af);
                    } else {
                        T::template fragment<j, 3>(tl[a], af);
                    }
#pragma unroll
                    for (int ni = 0; ni < NB; ni++) {
                        if constexpr (FLUSH == 0) {
                            ares_mma_f16_k16(acc[a][ni], af, b[ni][0], b[ni][1]);
                        } else {
                            if (kk % FLUSH == 0) {
                                ares_mma_f16acc_k16<false>(hacc[a][ni], af, b[ni][0], b[ni][1]);
                            } else {
                                ares_mma_f16acc_k16<true>(hacc[a][ni], af, b[ni][0], b[ni][1]);
                            }
                            if (kk % FLUSH == FLUSH - 1) {
                                ares_flush_f16(acc[a][ni], hacc[a][ni]);
                            }
                        }
                    }
                }
            }
        };
        step_fn(std::integral_constant<int, 0>{});
        step_fn(std::integral_constant<int, 1>{});
        step_fn(std::integral_constant<int, 2>{});
        step_fn(std::integral_constant<int, 3>{});
    }

    // The parts of K that the warps of the block computed, added in the first.
    if constexpr (DIRECT && WM > 1) {
        __shared__ float parts[WM][TA * NB * 4][32];
#pragma unroll
        for (int a = 0; a < TA; a++) {
#pragma unroll
            for (int ni = 0; ni < NB; ni++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    parts[warp][(a * NB + ni) * 4 + e][lane] = acc[a][ni][e];
                }
            }
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
#pragma unroll
        for (int a = 0; a < TA; a++) {
#pragma unroll
            for (int ni = 0; ni < NB; ni++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
#pragma unroll
                    for (int v = 1; v < WM; v++) {
                        acc[a][ni][e] += parts[v][(a * NB + ni) * 4 + e][lane];
                    }
                }
            }
        }
    }

    // The rows were divided by their scale (ares_convert_f16): put it back. Element e of
    // an accumulator is weight row g + 8 (e / 2), activation row 2t + e % 2 of the fragment.
    const float* const row_scale =
        reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(ax) + ares_act_ds_offset(static_cast<size_t>(n_batch) * pitch * 2));
#pragma unroll
    for (int ni = 0; ni < NB; ni++) {
        const uint32_t c0 = col0 + ni * 8 + 2 * t;
        const float s0 = c0 < n_batch ? row_scale[c0] : 0.0f;
        const float s1 = c0 + 1 < n_batch ? row_scale[c0 + 1] : 0.0f;
#pragma unroll
        for (int a = 0; a < TA; a++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t r = (tile0 + a) * 16 + g + (e >> 1) * 8;
                const uint32_t c = c0 + (e & 1);
                if (r < n_out && c < n_batch) {
                    y[static_cast<size_t>(c) * n_out + r] = acc[a][ni][e] * ((e & 1) ? s1 : s0);
                }
            }
        }
    }
}

// One kernel: the layout `T`, the shape and at least `BLOCKS` blocks per SM.
#define ARES_MMR_KERNEL(NAME, T, WM, TA, NB, FLUSH, DIRECT, BLOCKS)                                         \
    extern "C" __global__ void __launch_bounds__(WM * 32, BLOCKS) NAME(                                      \
        const uint8_t* w, const __half* act, float* y, uint32_t n_in, uint32_t n_out, uint32_t n_batch,     \
        uint32_t group) {                                                                                     \
        ares_mmr<T, WM, TA, NB, FLUSH, DIRECT>(w, act, y, n_in, n_out, n_batch, group);                      \
    }

// The six kernels of a layout, `ares_mmr_<shape>_<type>`: the shapes of the table
// above with f32 accumulation (f), and the two wide ones with f16 accumulation (h).
// `NEAR` is the layout as the kernels of up to 32 rows read it, `WIDE` as the grouped
// blocks of a wide product do.
#define ARES_MMR_FLUSH 4
#define ARES_MMR_KERNELS(TYPE, NEAR, WIDE)                                          \
    ARES_MMR_KERNEL(ares_mmr_r8f_##TYPE, NEAR, 4, 2, 1, 0, true, 5)                 \
    ARES_MMR_KERNEL(ares_mmr_r16f_##TYPE, NEAR, 4, 2, 2, 0, true, 3)                \
    ARES_MMR_KERNEL(ares_mmr_r32f_##TYPE, NEAR, 4, 2, 4, 0, false, 3)               \
    ARES_MMR_KERNEL(ares_mmr_r32h_##TYPE, NEAR, 4, 2, 4, ARES_MMR_FLUSH, false, 3)  \
    ARES_MMR_KERNEL(ares_mmr_r64f_##TYPE, WIDE, 4, 2, 8, 0, false, 2)               \
    ARES_MMR_KERNEL(ares_mmr_r64h_##TYPE, WIDE, 4, 2, 8, ARES_MMR_FLUSH, false, 2)
