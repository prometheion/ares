// The rate of the tensor cores through VK_KHR_cooperative_matrix: a subgroup multiplies
// matrices of the shape CM_M x CM_K by CM_K x CM_N into a CM_M x CM_N accumulator, in a
// loop on registers only (no memory traffic), eight independent accumulators to cover the
// latency. The example `vk_coop_rate` times it. A wrapper file defines the shape and the
// element types.

#pragma use_vulkan_memory_model
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 128) in;

layout(push_constant, std430) uniform Push {
    uint64_t sink;
    uint iterations;
} pc;

layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer Sink { float v[]; };

void main() {
    // Operands that differ from element to element and from invocation to invocation, so
    // that nothing folds to a constant and the two halves of a wide product (two hardware
    // instructions) are not the same instruction twice, which the compiler would run once.
    coopmat<TA, gl_ScopeSubgroup, CM_M, CM_K, gl_MatrixUseA> a;
    coopmat<TA, gl_ScopeSubgroup, CM_K, CM_N, gl_MatrixUseB> b;
    for (uint i = 0; i < a.length(); i++) {
        a[i] = TA((gl_LocalInvocationID.x + i) & 3u);
    }
    for (uint i = 0; i < b.length(); i++) {
        b[i] = TA((gl_LocalInvocationID.x * 3u + i * 5u) & 3u);
    }
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c0 =
        coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator>(TC(0));
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c1 = c0;
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c2 = c0;
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c3 = c0;
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c4 = c0;
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c5 = c0;
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c6 = c0;
    coopmat<TC, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> c7 = c0;
    for (uint i = 0; i < pc.iterations; i++) {
        c0 = coopMatMulAdd(a, b, c0);
        c1 = coopMatMulAdd(a, b, c1);
        c2 = coopMatMulAdd(a, b, c2);
        c3 = coopMatMulAdd(a, b, c3);
        c4 = coopMatMulAdd(a, b, c4);
        c5 = coopMatMulAdd(a, b, c5);
        c6 = coopMatMulAdd(a, b, c6);
        c7 = coopMatMulAdd(a, b, c7);
    }
    // Every element of every accumulator is used, or the compiler drops the hardware
    // instructions that only feed the unused ones (a wide product is several).
    float total = 0.0;
    for (uint i = 0; i < c0.length(); i++) {
        total += float(c0[i]) + float(c1[i]) + float(c2[i]) + float(c3[i]) + float(c4[i])
            + float(c5[i]) + float(c6[i]) + float(c7[i]);
    }
    // Written only if it is an impossible value, so the loop cannot be dropped.
    if (total == 12345.678) {
        Sink(pc.sink).v[gl_GlobalInvocationID.x] = total;
    }
}
