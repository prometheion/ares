// Prefill for Q4_K on the tensor cores from the rearranged form (`repack_q4_K.comp`) with
// the weights never in shared memory: each lane decodes the codes it reads into the elements
// of its part of the A operand, in registers (Marlin's way, and the way CUDA's `mmr.cuh`
// feeds `mma`). It needs the elements of a cooperative matrix of use A to be held in the
// order of an `mma` fragment, which `coop_layout_check.comp` verifies when the device is
// loaded (`Kernels::cooperative_operand_in_registers`); on a device where they are not,
// `mm_*_q4_K_r.comp` does the same with a tile in shared memory.
//
//   ARES_ACC32 = 0   f16 accumulation, added to f32 registers after every 64 values of K
//   ARES_ACC32 = 1   f32 accumulation
//
// A workgroup (4 subgroups) computes 128 weight rows by 64 activation rows: subgroup s the
// rows 32 s .. 32 s + 31 as 2 tiles of 16, against all 64 activation rows (4 matrices of 16).
// Per step of 64 values of K a lane reads 2 words of codes (512 contiguous bytes per tile and
// subgroup), decodes the 4 slabs of 16 values of each tile into A operands and multiplies
// them with the activations of the step, which the workgroup shares through shared memory
// (the only data that passes there, registers while the products of the step before run).
// One barrier per step.
//
//   w    the rearranged matrix, 16-byte aligned
//   act  the prepared activations (convert_act_f16.comp)
//   y    n_rows rows of n_out f32 values
// Grid: ceil(n_out / 128) * ceil(n_rows / 64) workgroups, in groups of `group` activation
// tiles that take all weight tiles in turn (as mm_f16_core.glsl).

#include "mmvq_common.glsl"
#include "mmd_common.glsl"

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 128) in;

layout(push_constant, std430) uniform Push {
    uint64_t w;
    uint64_t act;
    uint64_t y;
    uint64_t tables;
    uint n_in;
    uint n_out;
    uint n_rows;
    uint group;
} pc;

const uint MT = 128;  // weight rows per workgroup
const uint NT = 64;   // activation rows per workgroup
// Rows of the shared tile are 72 halves = 9 vectors of 16 bytes.
const uint ROW16 = 9;

shared uvec4 sb[2][NT * ROW16];  // activations, row n at sb[s][n * 9]
shared float stage[4][256];      // one f32 matrix per subgroup, for the result

#define MAT_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MAT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MAT_H coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define MAT_F coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

// Writes the 8 f16 of one slab of the lane (word cw of its tile: 4 nibble pairs) into the
// elements of `m`. The pair of register a_i of an `mma` fragment (row g + 8 (i & 1), columns
// 2t + 8 (i >> 1) and the next; scale s0 / minimum z0 for row g, s1 / z1 for g + 8) is held in
// elements 4 (i & 1) + 2 (i >> 1) and the next, in the order of Vulkan's operand
// (coop_layout_check.comp).
void decode_word(inout MAT_A m, uint cw, f16vec2 s0, f16vec2 z0, f16vec2 s1, f16vec2 z1) {
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const f16vec2 nib =
            unpackFloat2x16(((cw >> (4 * i)) & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0hf);
        const f16vec2 v = (i & 1u) == 0u ? fma(nib, s0, -z0) : fma(nib, s1, -z1);
        m[4 * (i & 1) + 2 * (i >> 1)] = v.x;
        m[4 * (i & 1) + 2 * (i >> 1) + 1] = v.y;
    }
}

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
    const uint sg = tid >> 5;
    const uint lane = tid & 31u;
    const uint g = lane >> 2;
    const uint t = lane & 3u;
    const uint n_steps = pc.n_in >> 6;
    const uint n_sb = pc.n_in >> 8;
    const uint n_tiles16 = (pc.n_out + 15) / 16;
    const uint64_t codes_bytes = uint64_t(n_tiles16) * 16 * pc.n_in / 2;

    // The two tiles of 16 weight rows of this subgroup.
    uint64_t code_base[2];
    uint64_t hdr_base[2];
    [[unroll]] for (uint a = 0; a < 2; a++) {
        const uint mt16 = min(wt * 8 + sg * 2 + a, n_tiles16 - 1);
        code_base[a] = pc.w + uint64_t(mt16) * n_steps * 512 + uint64_t(lane) * 16;
        hdr_base[a] = pc.w + codes_bytes + (uint64_t(mt16) * n_sb * 16 + g) * 16;
    }

    MAT_F accf[2][4];
#if ARES_ACC32
#define acc accf
#else
    MAT_H acc[2][4];
#endif
    [[unroll]] for (uint a = 0; a < 2; a++) {
        [[unroll]] for (uint b = 0; b < 4; b++) {
            accf[a][b] = MAT_F(0.0);
#if !ARES_ACC32
            acc[a][b] = MAT_H(0.0hf);
#endif
        }
    }

    uvec4 craw[2];
    uvec4 xa[4];
    uvec4 hd[4];  // headers of rows g and g + 8 of both tiles, for the super-block of the step

#define LOAD_STEP(s)                                                                        \
    [[unroll]] for (uint a = 0; a < 2; a++) {                                               \
        craw[a] = ld16(code_base[a] + uint64_t(s) * 512);                                   \
    }                                                                                       \
    [[unroll]] for (uint i = 0; i < 4; i++) {                                               \
        const uint idx = tid + 128 * i;                                                     \
        const uint row = min(bt * NT + (idx >> 3), pc.n_rows - 1);                          \
        xa[i] = ld16(pc.act + 2 * (uint64_t(row) * pc.n_in + uint64_t(s) * 64) +            \
                     16 * (idx & 7u));                                                      \
    }
#define STORE_ACT(st)                                                                       \
    [[unroll]] for (uint i = 0; i < 4; i++) {                                               \
        const uint idx = tid + 128 * i;                                                     \
        sb[st][(idx >> 3) * ROW16 + (idx & 7u)] = xa[i];                                    \
    }

    LOAD_STEP(0u)
    STORE_ACT(0u)
    barrier();

    for (uint ks = 0; ks < n_steps; ks++) {
        const uint cur = ks & 1u;
        const uint nxt = cur ^ 1u;
        const bool more = ks + 1 < n_steps;
        // Headers of the super-block, at its first step.
        if ((ks & 3u) == 0u) {
            [[unroll]] for (uint a = 0; a < 2; a++) {
                hd[2 * a] = ld16(hdr_base[a] + uint64_t(ks >> 2) * 256);
                hd[2 * a + 1] = ld16(hdr_base[a] + uint64_t(ks >> 2) * 256 + 128);
            }
        }
        // The codes of this step are in craw; the next step's are requested now.
        uvec4 cw[2] = craw;
        if (more) {
            LOAD_STEP(ks + 1)
        }

        // --- decode: the A operands of this step, 2 tiles x 4 slabs of 16 values ---------
        MAT_A ma[2][4];
        const uint j = ks & 3u;
        [[unroll]] for (uint a = 0; a < 2; a++) {
            uint sc0, mn0, sc1, mn1;
            scales_k(j, hd[2 * a].yzw, sc0, mn0);
            scales_k(j, hd[2 * a + 1].yzw, sc1, mn1);
            const vec2 d0 = unpackHalf2x16(hd[2 * a].x);
            const vec2 d1 = unpackHalf2x16(hd[2 * a + 1].x);
            [[unroll]] for (uint half_ = 0; half_ < 2; half_++) {
                const uint sh = 8 * half_;
                const f16vec2 s0 = f16vec2(float16_t(d0.x * float((sc0 >> sh) & 0xFFu)));
                const f16vec2 z0 = f16vec2(float16_t(d0.y * float((mn0 >> sh) & 0xFFu)));
                const f16vec2 s1 = f16vec2(float16_t(d1.x * float((sc1 >> sh) & 0xFFu)));
                const f16vec2 z1 = f16vec2(float16_t(d1.y * float((mn1 >> sh) & 0xFFu)));
                decode_word(ma[a][2 * half_], cw[a][2 * half_], s0, z0, s1, z1);
                decode_word(ma[a][2 * half_ + 1], cw[a][2 * half_ + 1], s0, z0, s1, z1);
            }
        }

        // --- the products: 4 slabs of 16 values ----------------------------------------------
        [[unroll]] for (uint kk = 0; kk < 4; kk++) {
            MAT_B mb[4];
            [[unroll]] for (uint b = 0; b < 4; b++) {
                coopMatLoad(mb[b], sb[cur], (16 * b) * ROW16 + 2 * kk, ROW16,
                            gl_CooperativeMatrixLayoutColumnMajor);
            }
            [[unroll]] for (uint a = 0; a < 2; a++) {
                [[unroll]] for (uint b = 0; b < 4; b++) {
                    acc[a][b] = coopMatMulAdd(ma[a][kk], mb[b], acc[a][b]);
                }
            }
        }
#if !ARES_ACC32
        // The partial sums of 64 products go to f32.
        [[unroll]] for (uint a = 0; a < 2; a++) {
            [[unroll]] for (uint b = 0; b < 4; b++) {
                accf[a][b] += MAT_F(acc[a][b]);
                acc[a][b] = MAT_H(0.0hf);
            }
        }
#endif
        if (more) {
            STORE_ACT(nxt)
        }
        barrier();
    }

    // --- the result: stage each matrix in shared memory (column-major: element (m, n) at
    // n * 16 + m) and write rows of 16 consecutive outputs, scaled by the factor of the
    // activation row.
    const uint64_t scales = pc.act + act_ds_offset(uint64_t(pc.n_rows) * pc.n_in * 2);
    [[unroll]] for (uint a = 0; a < 2; a++) {
        [[unroll]] for (uint b = 0; b < 4; b++) {
            coopMatStore(accf[a][b], stage[sg], 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            [[unroll]] for (uint i = 0; i < 8; i++) {
                const uint idx = lane + 32 * i;
                const uint n = bt * NT + 16 * b + (idx >> 4);
                const uint m = wt * MT + 32 * sg + 16 * a + (idx & 15u);
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
