#version 450
// The weight kernel over a device layout for many activation rows: a matrix in the layout
// of `ares_quant::layout` (mml_types.glsl, mml_codes.glsl) times up to 32 (ARES_NBT = 2) or
// 64 (ARES_NBT = 4) rows of f16 activations per workgroup, on the tensor cores. The
// counterpart of the shapes r32f, r32h, r64f and r64h of CUDA's `kernels/mmr.cuh`; build.rs
// compiles it per type and shape into `mml_r32f_<type>.spv` and so on.
//
//   ARES_ACC32 = 0   f16 accumulation, added to f32 registers after every 64 values of K
//   ARES_ACC32 = 1   f32 accumulation
//
// A workgroup (4 subgroups) computes 128 weight rows by 16 ARES_NBT activation rows:
// subgroup s the rows 32 s .. 32 s + 31 as 2 tiles of 16, against all activation rows
// (ARES_NBT operands of 16). Per step of 64 values of K a lane reads its words of the
// planes, decodes the 4 slabs of 16 values of each tile into A operands in registers (the
// weights never pass through shared memory; `coop_layout_check.comp` verifies the element
// order when the device is loaded) and multiplies them with the activations of the step,
// which the workgroup shares through shared memory (registers while the products of the
// step before run). One barrier per step.
//
//   w    the matrix in its device layout, 16-byte aligned
//   act  the prepared activations (convert_act_f16.comp)
//   y    n_rows rows of n_out f32 values
// Grid: ceil(n_out / 128) * ceil(n_rows / (16 ARES_NBT)) workgroups, in groups of `group`
// activation tiles that take all weight tiles in turn: the weights come from memory once
// per group and the activations of a group stay in L2.

#include "mmvq_common.glsl"

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define MAT_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MAT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MAT_H coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define MAT_F coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

#include "mmd_common.glsl"
#include "mml_types.glsl"
#ifndef LT_STREAM
#define LT_STREAM 0
#endif
#include "mml_codes.glsl"

layout(local_size_x = 128) in;

layout(push_constant, std430) uniform Push {
    uint64_t w;
    uint64_t act;
    uint64_t y;
    uint n_in;
    uint n_out;
    uint n_rows;
    uint group;
} pc;

const uint MT = 128;              // weight rows per workgroup
const uint NBT = ARES_NBT;        // B operands of 16 activation rows
const uint NT = 16 * ARES_NBT;    // activation rows per workgroup
// Rows of the shared tile are 72 halves = 9 vectors of 16 bytes: 8 for the 64 values and
// one more so that neither the lanes' writes nor the matrix loads meet in the same banks.
const uint ROW16 = 9;
// Pieces of 16 bytes of activations per step, and how many of them an invocation moves.
const uint PIECES = NT * 8;
const uint MOVES = PIECES / 128;

shared uvec4 sb_[2][NT * ROW16];  // activations, row n at sb_[s][n * 9]
shared float stage[4][256];       // one f32 matrix per subgroup, for the result

void main() {
    const uint n_wt = (pc.n_out + MT - 1) / MT;
    const uint n_bt = (pc.n_rows + NT - 1) / NT;
    const uint id = uint(ares_group());
    const uint g_all = pc.group * n_wt;
    const uint gi = id / g_all;
    const uint g_eff = min(pc.group, n_bt - gi * pc.group);
    const uint within = id - gi * g_all;
    const uint wt = within / g_eff;
    const uint bt = gi * pc.group + within % g_eff;
    if (wt >= n_wt) {
        return;
    }
    const uint tid = gl_LocalInvocationID.x;
#if LT_TABLE
    lt_init(tid, 128);
    barrier();
#endif
    const uint sg = tid >> 5;
    const uint lane = tid & 31u;
    const uint n_steps = pc.n_in >> 6;
    const uint n_sb = pc.n_in >> 8;
    const uint n_tiles16 = (pc.n_out + 15) / 16;
    const uint tile0 = wt * 8 + sg * 2;

    // The two tiles of 16 weight rows of this subgroup. A tile past the end of the matrix
    // repeats the last one; its results are not written.
    LtTile tl[2];
    [[unroll]] for (uint a = 0; a < 2; a++) {
        lt_open(tl[a], pc.w, uint64_t(n_tiles16), min(tile0 + a, n_tiles16 - 1), pc.n_in, lane,
                0);
    }

    MAT_F accf[2][NBT];
#if ARES_ACC32
#define acc accf
#else
    MAT_H acc[2][NBT];
#endif
    [[unroll]] for (uint a = 0; a < 2; a++) {
        [[unroll]] for (uint b = 0; b < NBT; b++) {
            accf[a][b] = MAT_F(0.0);
#if !ARES_ACC32
            acc[a][b] = MAT_H(0.0hf);
#endif
        }
    }

    // The activations of a step: piece i (16 bytes) is the part i % 8 of row i / 8; an
    // invocation moves the pieces tid + 128 u. A row past the end of the batch repeats the
    // last one.
    uvec4 xa[MOVES];
#define LOAD_ACT(s)                                                                         \
    [[unroll]] for (uint u = 0; u < MOVES; u++) {                                           \
        const uint idx = tid + 128 * u;                                                     \
        const uint row = min(bt * NT + (idx >> 3), pc.n_rows - 1);                          \
        xa[u] = ld16(pc.act + 2 * (uint64_t(row) * pc.n_in + uint64_t(s) * 64) +            \
                     16 * (idx & 7u));                                                      \
    }
#define STORE_ACT(st)                                                                       \
    [[unroll]] for (uint u = 0; u < MOVES; u++) {                                           \
        const uint idx = tid + 128 * u;                                                     \
        sb_[st][(idx >> 3) * ROW16 + (idx & 7u)] = xa[u];                                   \
    }

    LOAD_ACT(0u)
    STORE_ACT(0u)
    barrier();

    for (uint sb = 0; sb < n_sb; sb++) {
        [[unroll]] for (uint a = 0; a < 2; a++) {
            lt_super_block(tl[a], sb);
        }
        [[unroll]] for (uint j = 0; j < 4; j++) {
            const uint ks = 4 * sb + j;
            const uint cur = ks & 1u;
            const bool more = ks + 1 < n_steps;
            // The codes of this step were asked for a step ago; the next step's and its
            // activations are requested now.
            [[unroll]] for (uint a = 0; a < 2; a++) {
                lt_step(tl[a], ks, j);
            }
            if (more) {
                LOAD_ACT(ks + 1)
            }

            // The A operands of this step: 2 tiles x 4 slabs of 16 values.
            MAT_A ma[2][4];
            [[unroll]] for (uint a = 0; a < 2; a++) {
                [[unroll]] for (uint kk = 0; kk < 4; kk++) {
                    lt_fragment(ma[a][kk], tl[a], j, kk);
                }
            }

            [[unroll]] for (uint kk = 0; kk < 4; kk++) {
                MAT_B mb[NBT];
                [[unroll]] for (uint b = 0; b < NBT; b++) {
                    coopMatLoad(mb[b], sb_[cur], (16 * b) * ROW16 + 2 * kk, ROW16,
                                gl_CooperativeMatrixLayoutColumnMajor);
                }
                [[unroll]] for (uint a = 0; a < 2; a++) {
                    [[unroll]] for (uint b = 0; b < NBT; b++) {
                        acc[a][b] = coopMatMulAdd(ma[a][kk], mb[b], acc[a][b]);
                    }
                }
            }
#if !ARES_ACC32
            // The partial sums of 64 products go to f32.
            [[unroll]] for (uint a = 0; a < 2; a++) {
                [[unroll]] for (uint b = 0; b < NBT; b++) {
                    accf[a][b] += MAT_F(acc[a][b]);
                    acc[a][b] = MAT_H(0.0hf);
                }
            }
#endif
            if (more) {
                STORE_ACT(cur ^ 1u)
            }
            barrier();
        }
    }

    // The result: stage each matrix in shared memory (column-major: element (m, n) at
    // n * 16 + m) and write rows of 16 consecutive outputs, scaled by the factor of the
    // activation row.
    const uint64_t scales = pc.act + act_ds_offset(uint64_t(pc.n_rows) * pc.n_in * 2);
    [[unroll]] for (uint a = 0; a < 2; a++) {
        [[unroll]] for (uint b = 0; b < NBT; b++) {
            coopMatStore(accf[a][b], stage[sg], 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            [[unroll]] for (uint i = 0; i < 8; i++) {
                const uint idx = lane + 32 * i;
                const uint n = bt * NT + 16 * b + (idx >> 4);
                const uint m = (tile0 + a) * 16 + (idx & 15u);
                if (m < pc.n_out && n < pc.n_rows) {
                    FloatRef(pc.y + 4 * (uint64_t(n) * pc.n_out + m)).v[0] =
                        stage[sg][idx] * ldf1(scales + 4 * uint64_t(n));
                }
            }
            subgroupMemoryBarrierShared();
            subgroupBarrier();
        }
    }
}
