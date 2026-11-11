#version 450
// The weight kernel over a device layout for few activation rows: a matrix in the layout of
// `ares_quant::layout` (mml_types.glsl, mml_codes.glsl) times 1 to 8 (ARES_NB = 1) or up to
// 16 (ARES_NB = 2) rows of f16 activations, on the tensor cores with f32 accumulation. It
// serves decode from the same copy of the matrix the prefill kernel (mml_wide.glsl) reads, at
// the speed of the memory, and its result is the exact product of the f16 weights and f16
// activations. (CUDA's counterpart is `ares_mmr_few` in `kernels/mmr.cuh`.)
// build.rs compiles it per type and shape into `mml_r8f_<type>.spv` and `mml_r16f_<type>.spv`.
//
// A workgroup computes 32 weight rows (2 tiles of 16) by 8 ARES_NB activation rows, and its
// ARES_KS subgroups of 32 each take a stripe of K (a share of the super-blocks of 256
// values) of those same rows: their partial sums are added through shared memory at the
// end. A lane's loop over K is a chain (the codes of a step are asked for one step ahead,
// so a step waits for memory once), and a matrix with few rows has few such chains in
// flight: with the rows of a workgroup spread over two subgroups instead (64 rows, no
// stripes) a 2048 x 17408 matrix took 104 us on an RTX 4070 where the integer kernel,
// which splits a row over its lanes, takes 51, and a 256 x 5120 one 30 against 10. The
// stripes cut the chain to a quarter and put four times as many loads in flight.
//
// Nothing of the matrix or the activations passes through shared memory (the types of 16
// levels keep their table there): a lane decodes its codes into the elements of the A
// operand, reads its elements of the B operand (16 values of K by 8 activation rows)
// straight from the activations and holds its elements of the accumulators. That needs
// the element order of the three kinds of operand, which `coop_layout_check.comp` and
// `coop_layout_check8.comp` verify when the device is loaded.
//
//   w    the matrix in its device layout, 16-byte aligned
//   act  the prepared activations (convert_act_f16.comp): n_rows rows of n_in f16, then (at
//        the next multiple of 16 bytes) one f32 factor per row
//   y    n_rows rows of n_out f32 values
// Grid: ceil(n_out / 32) * ceil(n_rows / (8 ARES_NB)) workgroups, in groups of `group`
// activation tiles that take all weight tiles in turn.

#include "mmvq_common.glsl"

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define MAT_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MAT_B coopmat<float16_t, gl_ScopeSubgroup, 16, 8, gl_MatrixUseB>
#define MAT_C coopmat<float, gl_ScopeSubgroup, 16, 8, gl_MatrixUseAccumulator>

#include "mmd_common.glsl"
#include "mml_types.glsl"
// The planes are read through L1 like everything else: loads that pass it by (`coherent`
// under the Vulkan memory model, the nearest thing to CUDA's `no_allocate`, which gains 12 %
// there) were 3 to 7 % slower here at one row on an RTX 4070 and no different on an RTX 3060.
#ifndef LT_STREAM
#define LT_STREAM 0
#endif
#include "mml_codes.glsl"

// Stripes of K per workgroup (see above); the host's shape table assumes 32 weight rows per
// workgroup whatever this is.
#ifndef ARES_KS
#define ARES_KS 4
#endif

layout(local_size_x = 32 * ARES_KS) in;

layout(push_constant, std430) uniform Push {
    uint64_t w;
    uint64_t act;
    uint64_t y;
    uint n_in;
    uint n_out;
    uint n_rows;
    uint group;
} pc;

const uint MT = 32;            // weight rows per workgroup
const uint NB = ARES_NB;       // B operands of 8 activation rows
const uint NT = 8 * ARES_NB;   // activation rows per workgroup
const uint KS = ARES_KS;       // stripes of K

#if ARES_KS > 1
// The partial sums: for each stripe and lane the 4 elements of its 2 NB accumulators.
shared float part[KS * 32 * 8 * NB];
#endif

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
    lt_init(tid, 32 * KS);
    barrier();
#endif
    const uint sg = tid >> 5;
    const uint lane = tid & 31u;
    const uint g = lane >> 2;
    const uint t = lane & 3u;
    const uint n_sb = pc.n_in >> 8;
    const uint n_tiles16 = (pc.n_out + 15) / 16;
    const uint tile0 = wt * 2;
    // The super-blocks of this subgroup's stripe.
    const uint sb0 = (sg * n_sb) / KS;
    const uint sb1 = ((sg + 1) * n_sb) / KS;

    // The two tiles of 16 weight rows of this subgroup. A tile past the end of the matrix
    // repeats the last one; its results are not written.
    LtTile tl[2];
    [[unroll]] for (uint a = 0; a < 2; a++) {
        lt_open(tl[a], pc.w, uint64_t(n_tiles16), min(tile0 + a, n_tiles16 - 1), pc.n_in, lane,
                sb0);
    }

    // Lane (g, t) holds of B operand ni the values 2t, 2t + 1, 2t + 8, 2t + 9 of each slab
    // of activation row 8 ni + g. A row past the end of the batch repeats the last one.
    uint64_t arow[NB];
    [[unroll]] for (uint ni = 0; ni < NB; ni++) {
        const uint row = min(bt * NT + ni * 8 + g, pc.n_rows - 1);
        arow[ni] = pc.act + 2 * (uint64_t(row) * pc.n_in + 2 * t);
    }

    MAT_C acc[2][NB];
    [[unroll]] for (uint a = 0; a < 2; a++) {
        [[unroll]] for (uint ni = 0; ni < NB; ni++) {
            acc[a][ni] = MAT_C(0.0);
        }
    }

    for (uint sb = sb0; sb < sb1; sb++) {
        [[unroll]] for (uint a = 0; a < 2; a++) {
            lt_super_block(tl[a], sb);
        }
        [[unroll]] for (uint j = 0; j < 4; j++) {
            const uint step = 4 * sb + j;
            [[unroll]] for (uint a = 0; a < 2; a++) {
                lt_step(tl[a], step, j);
            }
            [[unroll]] for (uint kk = 0; kk < 4; kk++) {
                MAT_B mb[NB];
                [[unroll]] for (uint ni = 0; ni < NB; ni++) {
                    const uint64_t p = arow[ni] + 2 * (uint64_t(step) * 64 + kk * 16);
                    const f16vec2 lo = unpackFloat2x16(ld4(p));
                    const f16vec2 hi = unpackFloat2x16(ld4(p + 16));
                    mb[ni][0] = lo.x;
                    mb[ni][1] = lo.y;
                    mb[ni][2] = hi.x;
                    mb[ni][3] = hi.y;
                }
                [[unroll]] for (uint a = 0; a < 2; a++) {
                    MAT_A ma;
                    lt_fragment(ma, tl[a], j, kk);
                    [[unroll]] for (uint ni = 0; ni < NB; ni++) {
                        acc[a][ni] = coopMatMulAdd(ma, mb[ni], acc[a][ni]);
                    }
                }
            }
        }
    }

    // Element e of an accumulator is weight row g + 8 (e / 2), activation row 2t + e % 2 of
    // its operand, the same in every stripe: lane l of stripe 0 adds what lane l of each
    // stripe holds. The rows were divided by their factor (convert_act_f16.comp): put it back.
#if ARES_KS > 1
    [[unroll]] for (uint ni = 0; ni < NB; ni++) {
        [[unroll]] for (uint a = 0; a < 2; a++) {
            [[unroll]] for (uint e = 0; e < 4; e++) {
                part[(sg * 32 + lane) * (8 * NB) + (2 * ni + a) * 4 + e] = acc[a][ni][e];
            }
        }
    }
    barrier();
    if (sg != 0u) {
        return;
    }
#endif
    const uint64_t scales = pc.act + act_ds_offset(uint64_t(pc.n_rows) * pc.n_in * 2);
    [[unroll]] for (uint ni = 0; ni < NB; ni++) {
        const uint c0 = bt * NT + ni * 8 + 2 * t;
        const float s0 = c0 < pc.n_rows ? ldf1(scales + 4 * uint64_t(c0)) : 0.0;
        const float s1 = c0 + 1 < pc.n_rows ? ldf1(scales + 4 * uint64_t(c0 + 1)) : 0.0;
        [[unroll]] for (uint a = 0; a < 2; a++) {
            [[unroll]] for (uint e = 0; e < 4; e++) {
                const uint r = (tile0 + a) * 16 + g + 8 * (e >> 1);
                const uint c = c0 + (e & 1u);
#if ARES_KS > 1
                float sum = 0.0;
                [[unroll]] for (uint s = 0; s < KS; s++) {
                    sum += part[(s * 32 + lane) * (8 * NB) + (2 * ni + a) * 4 + e];
                }
#else
                const float sum = acc[a][ni][e];
#endif
                if (r < pc.n_out && c < pc.n_rows) {
                    FloatRef(pc.y + 4 * (uint64_t(c) * pc.n_out + r)).v[0] =
                        sum * ((e & 1u) != 0u ? s1 : s0);
                }
            }
        }
    }
}
