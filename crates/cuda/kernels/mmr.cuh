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
//   rows     kernel            block                 a warp computes        activations
//   1..8     ares_mmr_few      4 or 8 warps          1 tile x 1 fragment    read by each lane (L1)
//   9..16    ares_mmr_few      4 or 8 warps          1 tile x 2 fragments   cp.async per warp, ldmatrix
//   17..32   ares_mmr          4 warps, 128 x 32     2 tiles x 4 fragments  cp.async to shared, ldmatrix
//   33..     ares_mmr          4 warps, 128 x 64     2 tiles x 8 fragments  cp.async to shared, ldmatrix
//
// Up to 16 rows (a decode step, a window of drafts) the kernel waits for the weights: it
// runs at the speed of the memory like the integer kernels of mmvq.cuh, with f32
// accumulation, so its result is the exact product of the f16 weights and f16
// activations, summed in the order `ares_mmr_few` says: the same bits whatever the other
// rows of the batch are and however the kernel is launched. Above, the tensor cores bound
// the kernels; there each shape exists with f32 accumulation (the rate of a card without
// the GeForce cap) and with f16 accumulation added to f32 registers after every 4 `mma`
// (64 values of K), which the capped cards run at twice the rate. Which shape serves which
// number of rows above 16 is measured per device (src/kernels/layout.rs), and which
// launch the few-row kernels take per matrix shape.
//
// What was measured on the way (RTX 4070 / RTX 3060; Q4_K 17408 x 5120 unless said; the
// products of the walks over matrices larger than L2 and of a model's shapes, against the
// integer kernels of the file's layout, one row):
// - Two tiles per warp, each fragment of activations multiplied with both: half the
//   `ldmatrix` per `mma` of one tile per warp, 86 against 78 TFLOPS at 1024 rows.
//   Four tiles per warp need more registers than a thread has.
// - Activations: `cp.async` in three stages beat 16-byte loads into registers that
//   are stored to shared memory a step later (86 against 81).
// - Rows past the end of the batch are filled with zeros (`cp.async` of size 0), not
//   read again from the last row: 24 rows took 0.120 ms instead of 0.098 otherwise.
// - The codes are read past L1 (`no_allocate`): the weights are used once and would evict
//   the activations (0.109 -> 0.096 ms at one row).
// - Blocks of a wide product that share a weight tile (`group`, see mm.cuh) read
//   their codes two steps ahead through L1.
// - When the f16 sums are added to f32 (right after each `mma`, after the slab, or a
//   period later from a second set of registers) made no difference.
// - Up to 16 rows the first kernel of this family computed 2 tiles per warp over a quarter
//   of K each (4 warps on the same 2 tiles), one load ahead. It was 1 to 8 % slower than
//   the integer kernels on mid-sized matrices and 1.6 to 2 times slower on small ones
//   (Q4_K 256 x 5120: 11.8 against 6.3 microseconds on the RTX 3060), because a block
//   was one stream of loads of 20 steps, the number of blocks was not an even share of
//   the device (a matrix of 320 tiles made 160 blocks of which 140 fit on an RTX 3060: a
//   second wave of one stream of loads, 114 %), and the decode took 7 % of the time on
//   the RTX 3060 (without it 97 %; a streaming read of the same bytes reaches 330 GB/s
//   over that card's 28 multiprocessors, 460 over the RTX 4070's 46). What
//   `ares_mmr_few` does about each:
//   - K is cut into parts, whatever the launch (`ARES_MMF_PARTS`, below), and the warps of
//     a tile take 1, 2, 4 or 8 of them: a tile of a small matrix is then the work of 8
//     warps and a chain of 2.5 super-blocks (Q5_K 512 x 5120 on the RTX 3060: 14.0
//     microseconds before, 8.5 with 8 warps on the tile, the integer kernel 8.4); of a big
//     one a warp, with no sum between warps. Which is the fastest is a matter of the shape and the
//     device, and measured (`Kernels::tune_layout`; a rule by tiles per multiprocessor in
//     its place made a decode step with every matrix in the device layout 26.5 ms against
//     24.7 on the 27B).
//   - The loads of the next step are asked for at the start of a step, into the other of
//     two sets of registers that the steps of a super-block use in turn (so that no
//     register is moved), and kept there by `__syncwarp`: ptxas moves a load to where
//     its registers are first needed otherwise, and every super-block waits for its
//     weights (loads after the step, static sets: 118 to 146 % on the RTX 3060). With 4
//     sets (three steps ahead) 102 to 139 %; the weights through `cp.async` into a ring
//     of shared memory per warp (a super-block per stage, read back with a 16-byte
//     `ld.shared` per lane) 114 % with 2 stages and 127 % with 3 on the RTX 3060 (the
//     shared memory cuts the warps resident), 101 % on the RTX 4070.
//   - The code of a value is cut out where it lies: a mask, in place whenever the field
//     and the bits above it stay in the 10 bits of an f16 mantissa below 1024, and the
//     exact code (minus the offset of the type) comes back with one `HFMA2` that scales
//     by the power of two it was left at; the scale is an operand of the `HFMA2` that makes
//     the value (`.H0_H0`, `.H1_H1`: no `prmt` to duplicate it). Fewer integer
//     instructions per weight: the RTX 3060, Q4_K 6400 x 5120, 104 -> 100 % at one row.
//     (A scale divided by 16 for the codes that stay times 16 is a subnormal for small
//     scales and rounds.)
//   - Prefetching the activations of the next step (no gain, 104 -> 109 % on the smallest
//     matrices), cutting K into 20 parts, one per super-block, instead of 8 (Q5_K
//     512 x 5120 on the RTX 3060 104 -> 123 %), a second tile per warp with K cut the same
//     way for every launch (0 to 40 % slower than one).
//
#pragma once

#include <type_traits>

#include "mm_types.cuh"

static __device__ __forceinline__ void ares_ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(ares_smem_addr(p)));
}

// The dynamic shared memory of a block: three stages of J activation rows of 64 f16
// values, each row 16 bytes longer so that `ldmatrix` finds no bank conflicts.
template <int J>
struct AresMmrShared {
    static constexpr int STAGES = 3;
    static constexpr int PITCH = 64 * 2 + 16;
    static constexpr uint32_t BYTES = STAGES * J * PITCH;
};

// y[b][r] = sum over k of W[r][k] * x[b][k], 17 or more rows.
//
//   w    the matrix in the device layout of `T`
//   ax   n_batch rows of n_in f16 values, ares_act_f16_pitch(n_in) values apart, then
//        (16-byte aligned) one f32 per row: the power of two the row was divided by
//        (ares_convert_f16)
//   y    n_batch rows of n_out f32 values
//
// A block is WM warps and computes NB * 8 activation rows; a warp TA tiles of 16
// weight rows by all NB fragments, the warps of a block different tiles (WM * TA * 16
// weight rows) in step. FLUSH = 0: f32 accumulation; else f16 accumulation, added to f32
// after every FLUSH `mma`.
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
template <class T, int WM, int TA, int NB, int FLUSH>
static __device__ __forceinline__ void ares_mmr(
    const uint8_t* __restrict__ w, const __half* __restrict__ ax, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch, uint32_t group) {
    constexpr int NT = WM * 32;
    constexpr int I = WM * TA * 16, J = NB * 8;
    constexpr int STAGES = AresMmrShared<J>::STAGES;
    constexpr int PITCH = AresMmrShared<J>::PITCH;
    constexpr int PIECES = J * 8;  // pieces of 16 bytes of activations per step
    constexpr int BI = (PIECES + NT - 1) / NT;
    static_assert(FLUSH == 0 || 4 % FLUSH == 0, "a flush ends with a step");
    static_assert(NB == 1 || NB % 2 == 0, "fragments are loaded in pairs");
    static_assert(NT % 8 == 0, "a thread moves the same part of each of its rows");
    ares_check_smem(AresMmrShared<J>::BYTES);

    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3;
    uint32_t wt, bt;
    ares_mm_tile((n_out + I - 1) / I, (n_batch + J - 1) / J, group, wt, bt);
    const uint32_t col0 = bt * J;
    const uint32_t n_tiles = (n_out + 15) / 16, n_steps = n_in / 64, n_sb = n_in / 256;
    const uint32_t tile0 = (wt * WM + warp) * TA;
    uint8_t* const bs = ares_dyn_smem;
    const size_t pitch = ares_act_f16_pitch(n_in);

    // The tiles past the end of the matrix repeat the last one; their results are not
    // written.
    typename T::Warp wp;
    typename T::Tile tl[TA];
    const uint32_t first = min(tile0, n_tiles - 1);
    T::open_warp(wp, w, n_out, n_in, first, lane);
#pragma unroll
    for (int a = 0; a < TA; a++) {
        T::open(tl[a], wp, min(tile0 + a, n_tiles - 1) - first, 0);
    }

    // The activations of a step are PIECES pieces of 16 bytes, piece i the part i % 8 of
    // row i / 8. Thread tid moves the pieces tid + u NT: the same part of rows NT / 8
    // apart. Rows past the end of the batch are filled with zeros.
    const __half* b_src[BI];
    uint32_t b_dst0 = 0, b_valid = 0;
    b_dst0 = (tid / 8) * PITCH + 16 * (tid % 8);
#pragma unroll
    for (int u = 0; u < BI; u++) {
        const uint32_t c = (tid + u * NT) / 8;
        b_src[u] = ax + static_cast<size_t>(min(col0 + c, n_batch - 1)) * pitch + 8 * (tid % 8);
        b_valid |= (tid + u * NT < PIECES && col0 + c < n_batch) ? (1u << u) : 0u;
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

#pragma unroll
    for (int p = 0; p < STAGES - 1; p++) {
        if (p < n_steps) {
            load_acts(p, p);
        }
        ares_cp_commit();
    }

    for (uint32_t sb = 0; sb < n_sb; sb++) {
#pragma unroll
        for (int a = 0; a < TA; a++) {
            T::super_block(tl[a], wp, sb);
        }
        // One step of 64 values of K; j, its place in the super-block, is a constant so
        // that a layout selects its scales with constant shifts.
        auto step_fn = [&](auto jc) {
            constexpr int j = decltype(jc)::value;
            const uint32_t step = sb * 4 + j;
            ares_cp_wait_group<STAGES - 2>();
            __syncthreads();
            if (step + STAGES - 1 < n_steps) {
                load_acts(step + STAGES - 1, (step + STAGES - 1) % STAGES);
            }
            ares_cp_commit();
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
                if constexpr (NB == 1) {
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

// y[b][r] = sum over k of W[r][k] * x[b][k], for up to 8 * NB activation rows: what a
// decode step and a verify window run.
//
// K is cut into ARES_MMF_PARTS parts of whole super-blocks (part p: those from p n_sb / 8 to
// (p + 1) n_sb / 8), each summed from zero by the `mma` of one warp, and the result is the
// parts added in the order of K. That is the arithmetic of the product, whatever else the
// launch is: a tile of 16 weight rows is computed by `1 << log2_warps` of the BW warps of a
// block, each taking the same number of consecutive parts (the parts of the other warps
// reach it through shared memory, `ares_mmf_smem_bytes`). So the result of a row is a
// function of the matrix and the row's own activations: not of the other rows of the batch,
// nor of the kernel (rows, BW) and warps per tile that compute it, which is chosen by what
// is fastest (src/kernels/layout.rs).
// Grid: ceil(n_tiles / (BW >> log2_warps)) blocks.
//
// The layout `T` (AresFewCodes) holds the weights of the next steps in registers. With one
// fragment of activations (up to 8 rows) each lane reads its 2 x 4 bytes of each row of a slab
// (L1). With two, the activations of a step (64 values of each row) are copied by `cp.async`
// into a ring of ARES_MMF_STAGES stages that each warp has to itself in shared memory, one
// step ahead, and read from there with `ldmatrix`: the registers it gives are the ones the
// lanes would have loaded, so the result has the same bits. A load of a warp that reads 4
// bytes of 8 rows asks L1 for 8 lines, and with 16 rows the 16 such loads of a step bound the
// kernel, not the weights (RTX 3060, Q4_K 17408 x 5120: 231 to 247 us for 16 rows against 152
// for one, 160 to 169 with every lane of a load reading one row; RTX 4070 113 against 107).
// A copy of 16 bytes of 4 rows and an `ldmatrix` of 8 rows ask for 4 lines or 4 banks of
// shared memory: a quarter of the wavefronts (with the ring, 173 us on the RTX 3060 in the
// launch of 4 tiles a block; a launch of one tile a block reads the activations once per tile,
// 250 us, which is why the launch of this kernel is measured with 16 rows). With one fragment
// the ring was 2 to 5 % slower for the small matrices of the RTX 3060 at one row, where the
// lanes of a load read one row and L1 serves it at once. Rows past the batch are not copied:
// what their stage holds reaches only the outputs of those rows, which are not written.
#define ARES_MMF_PARTS 8
#define ARES_MMF_STAGES 2

// A stage of the activations of a warp: NB * 8 rows of 128 bytes, the 16-byte piece c of
// row r at r * 128 + (c ^ (r & 7)) * 16, so that an `ldmatrix` of a piece of 8 rows meets no
// bank twice.
static __host__ __device__ constexpr uint32_t ares_mmf_stage_bytes(int nb) {
    return static_cast<uint32_t>(nb * 8 * 128);
}

// The ring of a block of BW warps with NB fragments: none with one.
static __host__ __device__ constexpr uint32_t ares_mmf_ring_bytes(int bw, int nb) {
    return nb == 1 ? 0u : static_cast<uint32_t>(bw * ARES_MMF_STAGES) * ares_mmf_stage_bytes(nb);
}

// The dynamic shared memory of a block of BW warps with 1 << log2_warps warps per tile: the
// activations of every warp, then the parts of K the warps of a tile hand to each other.
static __host__ __device__ constexpr uint32_t ares_mmf_smem_bytes(int bw, int nb, int log2_warps) {
    return ares_mmf_ring_bytes(bw, nb) +
           (log2_warps == 0 ? 0u : static_cast<uint32_t>((bw >> log2_warps) * ARES_MMF_PARTS * nb * 4 * 32 * 4));
}

// `cp.async` of 16 bytes that keeps them in L1: the warps of a block that compute different
// tiles over the same parts of K copy the same activations.
static __device__ __forceinline__ void ares_cp_async16_l1(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(ares_smem_addr(dst)), "l"(src));
}

template <class T, int BW, int NB>
static __device__ __forceinline__ void ares_mmr_few(
    const uint8_t* __restrict__ w, const __half* __restrict__ ax, float* __restrict__ y,
    uint32_t n_in, uint32_t n_out, uint32_t n_batch, uint32_t log2_warps) {
    const uint32_t warps = 1u << log2_warps;
    const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t g = lane >> 2, t = lane & 3;
    const uint32_t slot = warp >> log2_warps, kw = warp & (warps - 1);
    const uint32_t n_tiles = (n_out + 15) / 16, n_sb = n_in / 256;
    const uint32_t tile = blockIdx.x * (BW >> log2_warps) + slot;
    // The parts this warp computes.
    const uint32_t pa = kw * (ARES_MMF_PARTS >> log2_warps), pb = pa + (ARES_MMF_PARTS >> log2_warps);
    const size_t pitch = ares_act_f16_pitch(n_in);

    // A tile past the end of the matrix repeats the last one; its results are not written.
    typename T::Warp wp;
    typename T::Tile tl;
    T::open_warp(wp, w, n_out, n_in, min(tile, n_tiles - 1), lane);
    wp.sb_end = pb * n_sb / ARES_MMF_PARTS;
    const uint32_t first_step = 4 * (pa * n_sb / ARES_MMF_PARTS), end_step = 4 * wp.sb_end;
    if (first_step < end_step) {
        T::open(tl, wp, pa * n_sb / ARES_MMF_PARTS);
    }
    // The activations: of fragment ni lane (g, t) holds the values 2t, 2t + 1, 2t + 8, 2t + 9
    // of each slab of row 8 ni + g. With one fragment it reads them itself; with two, lane l
    // copies the piece l % 8 of the rows l / 8 + 4 u of a step into the warp's ring and
    // `ldmatrix` gives them.
    constexpr bool RING = NB > 1;
    constexpr int J = NB * 8;
    constexpr uint32_t STAGE = ares_mmf_stage_bytes(NB);
    const __half* const arow = ax + static_cast<size_t>(min(g, n_batch - 1)) * pitch + 2 * t;
    uint8_t* const ring = ares_dyn_smem + static_cast<size_t>(warp) * ARES_MMF_STAGES * STAGE;
    const uint32_t piece = lane & 7, row0 = lane >> 3;
    const __half* const src0 = ax + static_cast<size_t>(row0) * pitch + 8 * piece;
    auto copy_step = [&](uint32_t step) {
        uint8_t* const stage = ring + (step % ARES_MMF_STAGES) * STAGE;
#pragma unroll
        for (int u = 0; u < J / 4; u++) {
            const uint32_t r = row0 + 4 * u;
            if (r < n_batch) {
                ares_cp_async16_l1(stage + r * 128 + ((piece ^ (r & 7)) << 4), src0 + static_cast<size_t>(4 * u) * pitch + 64 * step);
            }
        }
    };
    if constexpr (RING) {
        if (first_step < end_step) {
            copy_step(first_step);
        }
        ares_cp_commit();
    }
    // The rows were divided by their scale (ares_convert_f16): put it back at the end.
    // Element e of an accumulator is weight row g + 8 (e / 2), activation row 2t + e % 2 of
    // the fragment.
    const float* const row_scale =
        reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(ax) + ares_act_ds_offset(static_cast<size_t>(n_batch) * pitch * 2));

    float total[NB][4];
    float* const sums = reinterpret_cast<float*>(ares_dyn_smem + ares_mmf_ring_bytes(BW, NB)) +
                        static_cast<size_t>(slot) * ARES_MMF_PARTS * NB * 4 * 32;
    for (uint32_t p = pa; p < pb; p++) {
        float acc[NB][4];
#pragma unroll
        for (int ni = 0; ni < NB; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[ni][e] = 0.0f;
            }
        }
        for (uint32_t sb = p * n_sb / ARES_MMF_PARTS; sb < (p + 1) * n_sb / ARES_MMF_PARTS; sb++) {
            T::super_block(tl, wp, sb);
            auto step_fn = [&](auto jc) {
                constexpr int j = decltype(jc)::value;
                const uint32_t step = sb * 4 + j;
                T::template begin<j>(tl, wp, step);
                // The next step's activations go into the stage the step before read (every
                // lane is past its `ldmatrix` of it); this step's have landed once the copies
                // of every lane have.
                if constexpr (RING) {
                    if (step + 1 < end_step) {
                        copy_step(step + 1);
                    }
                    ares_cp_commit();
                    ares_cp_wait_group<1>();
                    __syncwarp();
                }
                const uint8_t* const stage = ring + (step % ARES_MMF_STAGES) * STAGE;
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    // The B fragments of the slab: register 0 holds the values 2t, 2t + 1 and
                    // register 1 the values 2t + 8, 2t + 9 of activation row g of the fragment.
                    // Lanes 8 m to 8 m + 7 name the rows of matrix m: rows 8 (m / 2) + 0..7, the
                    // piece 2 kk + m % 2.
                    uint32_t b[NB][2];
                    const uint32_t c = 2 * kk + ((lane >> 3) & 1);
                    if constexpr (!RING) {
                        const uint32_t* q = reinterpret_cast<const uint32_t*>(arow + step * 64 + kk * 16);
                        b[0][0] = __ldg(q);
                        b[0][1] = __ldg(q + 4);
                    } else {
#pragma unroll
                        for (int nj = 0; nj < NB / 2; nj++) {
                            const uint32_t r = nj * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                            uint32_t r4[4];
                            ares_ldmatrix_x4(r4, stage + r * 128 + ((c ^ (r & 7)) << 4));
                            b[2 * nj][0] = r4[0];
                            b[2 * nj][1] = r4[1];
                            b[2 * nj + 1][0] = r4[2];
                            b[2 * nj + 1][1] = r4[3];
                        }
                    }
                    uint32_t af[4];
                    if (kk == 0) {
                        T::template fragment<j, 0>(tl, af);
                    } else if (kk == 1) {
                        T::template fragment<j, 1>(tl, af);
                    } else if (kk == 2) {
                        T::template fragment<j, 2>(tl, af);
                    } else {
                        T::template fragment<j, 3>(tl, af);
                    }
#pragma unroll
                    for (int ni = 0; ni < NB; ni++) {
                        ares_mma_f16_k16(acc[ni], af, b[ni][0], b[ni][1]);
                    }
                }
                // Every lane has read this stage before any lane copies into it again.
                if constexpr (RING) {
                    __syncwarp();
                }
            };
            step_fn(std::integral_constant<int, 0>{});
            step_fn(std::integral_constant<int, 1>{});
            step_fn(std::integral_constant<int, 2>{});
            step_fn(std::integral_constant<int, 3>{});
        }
        // With one warp per tile the parts are added in registers, in order; otherwise they
        // are kept apart for the sum below.
#pragma unroll
        for (int ni = 0; ni < NB; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                if (log2_warps == 0) {
                    total[ni][e] = p == 0 ? acc[ni][e] : total[ni][e] + acc[ni][e];
                } else {
                    sums[((p * NB + ni) * 4 + e) * 32 + lane] = acc[ni][e];
                }
            }
        }
    }

    if (log2_warps > 0) {
        // Output o of a tile is element o / 32 of the accumulators of lane o % 32; the warps
        // of the tile take every `warps`-th group of 32.
        constexpr int OUT = NB * 4 * 32;
        __syncthreads();
        for (uint32_t o = kw * 32 + lane; o < OUT; o += warps * 32) {
            float sum = sums[o];
            for (uint32_t p = 1; p < ARES_MMF_PARTS; p++) {
                sum += sums[p * OUT + o];
            }
            const uint32_t el = o / 32, e = el % 4, ni = el / 4;
            const uint32_t r = tile * 16 + ((o % 32) >> 2) + (e >> 1) * 8;
            const uint32_t c = ni * 8 + 2 * (o % 4) + (e & 1);
            if (r < n_out && c < n_batch) {
                y[static_cast<size_t>(c) * n_out + r] = sum * row_scale[c];
            }
        }
        return;
    }
#pragma unroll
    for (int ni = 0; ni < NB; ni++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const uint32_t r = tile * 16 + g + (e >> 1) * 8;
            const uint32_t c = ni * 8 + 2 * t + (e & 1);
            if (r < n_out && c < n_batch) {
                y[static_cast<size_t>(c) * n_out + r] = total[ni][e] * row_scale[c];
            }
        }
    }
}

// A few-row kernel: the layout `T`, BW warps per block, NB fragments of activations, at
// least `BLOCKS` blocks per SM.
#define ARES_MMF_KERNEL(NAME, T, BW, NB, BLOCKS)                                                          \
    extern "C" __global__ void __launch_bounds__(BW * 32, BLOCKS) NAME(                                  \
        const uint8_t* w, const __half* act, float* y, uint32_t n_in, uint32_t n_out, uint32_t n_batch, \
        uint32_t log2_warps) {                                                                           \
        ares_mmr_few<T, BW, NB>(w, act, y, n_in, n_out, n_batch, log2_warps);                            \
    }

// The four kernels of a layout for up to 8 and up to 16 rows, `ares_mmf_<rows>_w<warps>_<type>`.
// `FEW` is the layout as they read it.
#define ARES_MMF_KERNELS(TYPE, FEW)                              \
    ARES_MMF_KERNEL(ares_mmf_8_w4_##TYPE, FEW, 4, 1, 6)          \
    ARES_MMF_KERNEL(ares_mmf_8_w8_##TYPE, FEW, 8, 1, 3)          \
    ARES_MMF_KERNEL(ares_mmf_16_w4_##TYPE, FEW, 4, 2, 4)         \
    ARES_MMF_KERNEL(ares_mmf_16_w8_##TYPE, FEW, 8, 2, 2)

// One kernel: the layout `T`, the shape and at least `BLOCKS` blocks per SM.
#define ARES_MMR_KERNEL(NAME, T, WM, TA, NB, FLUSH, BLOCKS)                                                 \
    extern "C" __global__ void __launch_bounds__(WM * 32, BLOCKS) NAME(                                      \
        const uint8_t* w, const __half* act, float* y, uint32_t n_in, uint32_t n_out, uint32_t n_batch,     \
        uint32_t group) {                                                                                     \
        ares_mmr<T, WM, TA, NB, FLUSH>(w, act, y, n_in, n_out, n_batch, group);                              \
    }

// The four kernels of a layout for 17 rows and more, `ares_mmr_<shape>_<type>`: the shapes
// of the table above with f32 accumulation (f), and the two wide ones with f16
// accumulation (h). `NEAR` is the layout as the kernels of up to 32 rows read it, `WIDE` as
// the grouped blocks of a wide product do.
#define ARES_MMR_FLUSH 4
#define ARES_MMR_KERNELS(TYPE, NEAR, WIDE)                                      \
    ARES_MMR_KERNEL(ares_mmr_r32f_##TYPE, NEAR, 4, 2, 4, 0, 3)                  \
    ARES_MMR_KERNEL(ares_mmr_r32h_##TYPE, NEAR, 4, 2, 4, ARES_MMR_FLUSH, 3)     \
    ARES_MMR_KERNEL(ares_mmr_r64f_##TYPE, WIDE, 4, 2, 8, 0, 2)                  \
    ARES_MMR_KERNEL(ares_mmr_r64h_##TYPE, WIDE, 4, 2, 8, ARES_MMR_FLUSH, 2)
