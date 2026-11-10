// The shared part of the f16 prefill kernels (mm_*.comp): the declarations and the
// tile geometry. A kernel file includes this, then a weights piece (`mmw_*.glsl`: how the
// weights of the step are fetched and decoded into the shared tile), then `mm_f16_core.glsl`
// (the loop over K and the tensor-core products).
//
//   ARES_ACC32 = 0   f16 accumulation, added to f32 registers after every 64 values of K
//                    (PrefillPath::F16Flush; the activations are divided per row by a power
//                    of two so that no sum of 64 products can overflow)
//   ARES_ACC32 = 1   f32 accumulation (PrefillPath::F16)
//
// A workgroup (4 subgroups) computes 64 weight rows by 64 activation rows, each subgroup a
// 32 x 32 corner as 2 x 2 matrices of 16 x 16, in steps of 64 values of K. The decoded
// weights and the f16 activations of a step are plain row-major f16 tiles in shared memory
// (two stages, one barrier per step); cooperative matrices are opaque, so nothing but the
// decoding knows which element is which.
//
//   act  the prepared activations (convert_act_f16.comp): n_rows rows of n_in f16, then (at
//        the next multiple of 16 bytes) one f32 factor per row
//   y    n_rows rows of n_out f32 values
// Grid: ceil(n_out / 64) * ceil(n_rows / 64) workgroups, in groups of `group` activation
// tiles that take all weight tiles in turn: the weights come from memory once per group and
// the activations of a group stay in L2.

#include "mmvq_common.glsl"

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

const uint MT = 64;  // weight rows per workgroup
const uint NT = 64;  // activation rows per workgroup
// Rows of the shared tiles are 72 halves = 9 vectors of 16 bytes: 8 for the 64 values and
// one more so that neither the lanes' writes nor the matrix loads meet in the same banks.
const uint ROW16 = 9;

shared uint sa[2][MT * ROW16 * 4];  // decoded weights as f16 pairs, row r at sa[s][r * 36]
shared uvec4 sb[2][NT * ROW16];  // activations, row n at sb[s][n * 9]
shared float stage[4][256];      // one f32 matrix per subgroup, for the result

// Where this invocation is, set by main() before the weights piece runs.
uint tid;     // in the workgroup
uint sg;      // subgroup
uint lane;    // in the subgroup
uint wt;      // weight tile of the workgroup
uint n_steps; // steps of 64 values of K
