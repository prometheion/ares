// Declarations shared by the quantized weight kernels (mmvq_*.comp) and the
// activation quantizer (quantize_act.comp).
//
// These kernels need more of the device than the first four: integer dot products
// (VK_KHR_shader_integer_dot_product), 8- and 16-bit buffer access for the blocks
// of the types whose size is not a multiple of 4, and clustered subgroup
// operations for the sum over a warp. The host enables all of it where the device
// has it and loads these kernels only there (`Kernels::has_quantized_kernels`).

// The Vulkan memory model: under it the `coherent` qualifier of the weight loads
// (see `ldw*` below) makes the driver read them past the L1 cache; under the default
// memory model it is only a decoration that NVIDIA's compiler ignores.
#pragma use_vulkan_memory_model

#include "common.glsl"

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_clustered : require

// Read-only references of every width a block can be read at. A block of a type
// starts at a multiple of its size, so only 16-byte aligned references are made
// to addresses that are; the others say so.
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer P16 { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer P8 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer P4 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer P2 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) readonly buffer P1 { uint8_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer PF4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer PF2 { vec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer PF1 { float v[]; };

// The same for the weights, which every launch reads once: a nontemporal load is
// the streaming access (CUDA's `ld.global.nc.L1::no_allocate`) that keeps them from
// pushing the activations, which the whole grid reads again and again, out of L1.
layout(buffer_reference, std430, buffer_reference_align = 16) readonly coherent buffer W16 { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly coherent buffer W8 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly coherent buffer W4 { uint v[]; };

uvec4 ldw16(uint64_t a) { return W16(a).v[0]; }
uvec2 ldw8(uint64_t a) { return W8(a).v[0]; }
uint ldw4(uint64_t a) { return W4(a).v[0]; }

uvec4 ld16(uint64_t a) { return P16(a).v[0]; }
uvec2 ld8(uint64_t a) { return P8(a).v[0]; }
uint ld4(uint64_t a) { return P4(a).v[0]; }
uint ld2(uint64_t a) { return uint(P2(a).v[0]); }
uint ld1(uint64_t a) { return uint(P1(a).v[0]); }
vec4 ldf4(uint64_t a) { return PF4(a).v[0]; }
vec2 ldf2(uint64_t a) { return PF2(a).v[0]; }
float ldf1(uint64_t a) { return PF1(a).v[0]; }

// 32 and 64 bits from an address that is only 2-byte aligned: the blocks of Q6_K,
// Q3_K and the i-quants are 210, 110, 82 ... bytes, so half of them (or all, for
// some types) start 2 bytes off a 4-byte boundary. Two 16-bit loads, as the CUDA
// kernels do.
uint ld4_a2(uint64_t a) { return ld2(a) | (ld2(a + 2) << 16); }
uvec2 ld8_a2(uint64_t a) { return uvec2(ld4_a2(a), ld4_a2(a + 4)); }

// 32 bits from an address of any alignment: two aligned words and a funnel shift.
// The second word is only read when the address is not aligned, so nothing past the
// end of a block that ends on a word boundary is touched.
uint ld4_u(uint64_t a) {
    const uint64_t base = a & ~uint64_t(3);
    const uint shift = uint(a & uint64_t(3)) * 8u;
    const uint lo = ld4(base);
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (ld4(base + 4) << (32u - shift));
}
uvec2 ld8_u(uint64_t a) { return uvec2(ld4_u(a), ld4_u(a + 4)); }

// The f16 whose bits are the low 16 of `bits`, as f32 (exact).
float half_of(uint bits) { return unpackHalf2x16(bits).x; }

// c + the dot product of the four signed bytes of a with the four signed bytes of
// b (CUDA's __dp4a), and with the four bytes of a taken as unsigned (0..255) and
// those of b as signed (CUDA's dp4a.u32.s32): quantized weights are unsigned
// codes, activations are signed. Both are single instructions where the device
// reports integerDotProduct4x8BitPacked{Signed,MixedSignedness}Accelerated.
int dp_ss(uint a, uint b, int c) { return c + dotPacked4x8EXT(int(a), int(b)); }
int dp_us(uint a, uint b, int c) { return c + dotPacked4x8EXT(a, int(b)); }

// The most rows of activations one launch of a weight kernel handles
// (ARES_MAX_BATCH in common.glsl).
//
// Byte offset of the (d, s) pairs behind `n_values` = n_batch * n_in int8 values,
// as in quantize_act.cuh.
uint64_t act_ds_offset(uint64_t n_values) { return (n_values + 15) & ~uint64_t(15); }
