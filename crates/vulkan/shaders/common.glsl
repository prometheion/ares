// Shared declarations of the kernels.
//
// Conventions:
// - Every kernel is a compute shader whose entry point is `main`, in a file
//   named after the kernel (`add_f32.comp`). The host finds it by that name.
// - Buffers are never bound. A kernel receives the device addresses of its
//   buffers in push constants (VK_KHR_buffer_device_address), next to its
//   sizes, exactly like the CUDA kernels receive pointers. No descriptor set
//   exists, so a launch costs one push and one dispatch whatever it touches.
// - Kernels take sizes as arguments. Nothing about a batch is compiled in, so
//   a change of batch shape costs nothing on the device. The one compiled-in
//   number is the workgroup size, which the host picks from a fixed set of
//   pipelines made at load.
// - Push constants are laid out as std430: addresses first (8 bytes, 8-byte
//   aligned), then 4-byte values. `Args` on the host packs them the same way.
// - Workgroup sizes are multiples of 32 and the subgroup is at least 8
//   invocations wide, so a workgroup has at most 32 subgroups.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

// The most rows of activations one launch of a weight kernel handles.
#define ARES_MAX_BATCH 8

// Pointers to f32 values, to 32-bit words and to 16-byte units, as buffer
// references. The alignment is a promise to the compiler: a reference to
// 16-byte units must be 16-byte aligned, and one to floats 4-byte aligned.
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatRef { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatRefRO { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WordRef { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Vec4Ref { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer FloatVec4Ref { vec4 v[]; };

// The index of the workgroup in a launch of one logical dimension. A device
// may cap the x dimension of a dispatch below what a long vector needs; the
// host then folds the launch into x and y, and this undoes the fold.
uint64_t ares_group() {
    return uint64_t(gl_WorkGroupID.y) * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}
