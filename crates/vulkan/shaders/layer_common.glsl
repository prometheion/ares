// Shared declarations of the layer kernels (`layer_*.comp`): the step record as the
// kernels read it, the scalar functions, and the quantization of activations fused into
// the kernels that feed a matrix product.
//
// The step record is the one of the CUDA layer (`kernels/layer_step.cuh`), byte for byte:
//   header   32 bytes: n_rows, n_segs, n_out, reserved, then the byte offsets of the
//            segments, tokens, positions and output rows
//   segment  32 bytes: row0, n_rows, slot, accepted, pos0, flags, two reserved words
//   uint     [n_rows]      token of each row
//   int      [n_rows][4]   position of each row: time, height, width, extra
//   uint     [n_out]       the rows whose logits are wanted
//
// Lanes: the kernels give work to clusters of 8 or 32 consecutive invocations of a
// workgroup and reduce over them with clustered subgroup operations, as the weight kernels
// do. The host loads them only on a device whose subgroups are a multiple of 32 wide
// (`LayerKernels::load`).
// (A kernel that includes mmvq_common.glsl first, for its narrow loads, defines
// ARES_COMMON_INCLUDED: common.glsl has no include guard.)
#ifndef ARES_COMMON_INCLUDED
#include "common.glsl"
#endif
#extension GL_KHR_shader_subgroup_clustered : require

layout(buffer_reference, std430, buffer_reference_align = 16) buffer LVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer LVec4RO { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer LVec2 { vec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer LHalf4 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer LWord { uint v[]; };

vec4 ldv4(uint64_t a) { return LVec4RO(a).v[0]; }
float ldf(uint64_t a) { return FloatRefRO(a).v[0]; }
uint ldu(uint64_t a) { return WordRef(a).v[0]; }

// Segment flag: all rows of the segment will be accepted.
const uint SEG_DIRECT = 1u;

struct Seg {
    uint row0;
    uint n_rows;
    uint slot;
    uint accepted;
    uint flags;
};

uint step_n_rows(uint64_t step) { return ldu(step); }
uint step_n_out(uint64_t step) { return ldu(step + 8); }
Seg step_seg(uint64_t step, uint i) {
    const uint64_t a = step + ldu(step + 16) + 32 * uint64_t(i);
    return Seg(ldu(a), ldu(a + 4), ldu(a + 8), ldu(a + 12), ldu(a + 20));
}
uint step_token(uint64_t step, uint row) { return ldu(step + ldu(step + 20) + 4 * uint64_t(row)); }
int step_pos(uint64_t step, uint row, uint kind) {
    return int(ldu(step + ldu(step + 24) + 16 * uint64_t(row) + 4 * kind));
}
uint step_out_row(uint64_t step, uint i) { return ldu(step + ldu(step + 28) + 4 * uint64_t(i)); }

// The scalar functions, in f32.
float l_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
float l_silu(float x) { return x / (1.0 + exp(-x)); }
vec4 l_sigmoid(vec4 x) { return 1.0 / (1.0 + exp(-x)); }
vec4 l_silu(vec4 x) { return x / (1.0 + exp(-x)); }
// ln(1 + e^x), x itself above 20. GLSL has no log1p: ln(u) * e / (u - 1) with u = 1 + e
// is exact to rounding where 1 + e loses the low bits of e (Kahan's form).
float l_softplus(float x) {
    if (x > 20.0) {
        return x;
    }
    precise float e = exp(x);
    precise float u = 1.0 + e;
    if (u == 1.0) {
        return e;
    }
    precise float r = log(u) * e / (u - 1.0);
    return r;
}

// --- quantization of activations, the arithmetic and the bytes of quantize_act.comp ------
//
// Behind the int8 codes and the (scale, sum) pairs, which are what the weight kernels read
// (`Kernels::quantized_activation_bytes`), the fused kernels keep the f32 values they
// computed, at the next multiple of 16 bytes: the weight kernels of the float types and
// the conversion to f16 for the tensor cores read those.
uint64_t act_ds_at(uint64_t n_values) { return (n_values + 15) & ~uint64_t(15); }
uint64_t act_x_at(uint64_t n_values) {
    return (act_ds_at(n_values) + (n_values / 32) * 8 + 15) & ~uint64_t(15);
}

float round_away(float x) {
    const float t = trunc(x);
    return abs(x - t) >= 0.5 ? t + sign(x) : t;
}
// a / b correctly rounded (see quantize_act.comp).
float div_rn(float a, float b) {
    precise float r = 1.0 / b;
    precise float e = fma(-b, r, 1.0);
    r = fma(e, r, r);
    precise float q = a * r;
    precise float rem = fma(-q, b, a);
    return fma(rem, r, q);
}

// Quantizes chunk `chunk` (4 consecutive values) of a buffer of `n_values` values at `q`.
// A group of 32 values is 8 consecutive chunks held by a cluster of 8 invocations, so every
// invocation of the cluster calls this in the same control flow; one without data passes
// zeros and `live` = false.
void quant_chunk(vec4 v, bool live, uint64_t chunk, uint64_t q, uint64_t n_values) {
    float amax = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    amax = subgroupClusteredMax(amax, 8);
    const float d = div_rn(amax, 127.0);
    vec4 code = vec4(0.0);
    if (amax != 0.0) {
        code = vec4(
            round_away(div_rn(v.x, d)), round_away(div_rn(v.y, d)), round_away(div_rn(v.z, d)),
            round_away(div_rn(v.w, d)));
    }
    // Integers of at most 127 in magnitude: the sum is exact in any order.
    const float sum_codes = subgroupClusteredAdd(code.x + code.y + code.z + code.w, 8);
    if (live) {
        const ivec4 c = ivec4(code);
        LWord(q + 4 * chunk).v[0] =
            (uint(c.x) & 0xFFu) | ((uint(c.y) & 0xFFu) << 8) | ((uint(c.z) & 0xFFu) << 16) | (uint(c.w) << 24);
        LVec4(q + act_x_at(n_values) + 16 * chunk).v[0] = v;
        if ((gl_SubgroupInvocationID & 7u) == 0u) {
            LVec2(q + act_ds_at(n_values) + 8 * (chunk >> 3)).v[0] = vec2(d, d * sum_codes);
        }
    }
}
