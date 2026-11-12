// Shared declarations of the attention kernels (`attn_*`): the geometry of the pools, the
// formats of the stored K and V rows, and the codec that writes them.
//
// This is `crates/cuda/kernels/attn_layout.cuh` and `attn_codec.cuh` for a cluster of 32
// invocations: one cluster encodes one group of 32 values, invocation i holding value i. The
// specification (`ares_quant::kv`) fixes the order of every float operation, so that the bytes
// are the CPU's and CUDA's. Here every operation of it is a `precise` one (no fused
// multiply-add), the divisions are correctly rounded (`div_rn`), and the reductions are the
// butterflies the specification names (`sum32`: the pairs at distance 16 first, then 8, 4, 2,
// 1). The float to half conversion is done on the bits, because the rounding of
// `packHalf2x16` is not specified by Vulkan.
//
// Layout of a sequence on the device, see `src/attention.rs`:
//
//   exact pool      slot = H heads of [K: 128 rows of D halves][V: 128 rows of D halves]
//   converted pool  slot = H heads of [K mean][K rows][V mean][V rows]
//
// Clusters: invocation i of a workgroup is lane i % 32 of cluster i / 32, and a subgroup is a
// multiple of 32 wide (the host refuses a device where it is not: `AttentionKernels::load`).
#ifndef ARES_ATTN_COMMON
#define ARES_ATTN_COMMON

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_shuffle : require
#include "layer_common.glsl"

layout(buffer_reference, std430, buffer_reference_align = 1) buffer B1 { uint8_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer B2 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer B4 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer B8 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer B16 { uvec4 v[]; };

uint ld1(uint64_t a) { return uint(B1(a).v[0]); }
uint ld2(uint64_t a) { return uint(B2(a).v[0]); }
uint ld4(uint64_t a) { return B4(a).v[0]; }
uvec2 ld8(uint64_t a) { return B8(a).v[0]; }
uvec4 ld16(uint64_t a) { return B16(a).v[0]; }

const uint FMT_F16 = 0u;
const uint FMT_Q8 = 1u;
const uint TILE_CELLS = 128u;
const uint CONVERTED_BIT = 0x80000000u;

// Bits of the status word the writing kernels or-in.
const uint STATUS_NONFINITE = 1u;
const uint STATUS_OVERFLOW = 2u;
const uint STATUS_DIRECT_MEAN = 4u;

uint fmt_bits(uint f) { return f == 0u ? 0u : 9u - f; }
uint fmt_group_bytes(uint f) { return f == 0u ? 64u : 2u + 4u * (9u - f); }
bool fmt_has_mean(uint f) { return f >= 2u; }
bool fmt_rotated(uint f) { return f >= 1u; }

// The geometry of the pools for one launch, decoded from the 64-bit configuration word:
//   bits 0-3 K format, 4-7 V format, 8-15 H (kv heads in the pools), 16-23 kv_total (kv
//   heads of the model), 24-31 head0 (first kv head of this device), 32-39 G (query heads
//   per kv head), 40-51 D (head size).
struct Geo {
    uint kf, vf, H, kv_total, head0, G, D;
    uint rbk, rbv, kmean, vmean;
    uint k_rows_off, v_mean_off, v_rows_off;
    uint head_c;
    uint64_t head_e, entry_e, entry_c;
};

Geo attn_geo(uint64_t cfg) {
    Geo g;
    const uint lo = uint(cfg & 0xffffffffUL), hi = uint(cfg >> 32);
    g.kf = lo & 15u;
    g.vf = (lo >> 4) & 15u;
    g.H = (lo >> 8) & 255u;
    g.kv_total = (lo >> 16) & 255u;
    g.head0 = (lo >> 24) & 255u;
    g.G = hi & 255u;
    g.D = (hi >> 8) & 4095u;
    const uint groups = g.D / 32u;
    g.rbk = groups * fmt_group_bytes(g.kf);
    g.rbv = groups * fmt_group_bytes(g.vf);
    g.kmean = fmt_has_mean(g.kf) ? 2u * g.D : 0u;
    g.vmean = fmt_has_mean(g.vf) ? 2u * g.D : 0u;
    g.k_rows_off = g.kmean;
    g.v_mean_off = g.kmean + TILE_CELLS * g.rbk;
    g.v_rows_off = g.v_mean_off + g.vmean;
    g.head_c = g.v_rows_off + TILE_CELLS * g.rbv;
    g.head_e = uint64_t(TILE_CELLS) * 2ul * g.D * 2ul;
    g.entry_e = g.head_e * g.H;
    g.entry_c = uint64_t(g.head_c) * g.H;
    return g;
}

// One row of a step (16 bytes): the sequence's table, the position of the row's own cell, and
// where its partial states go.
struct AttnRowRec {
    uint table;
    uint pos;
    uint slot_base;
    uint n_slots;
};

AttnRowRec attn_row(uint64_t rows, uint i) {
    const uvec4 w = ld16(rows + 16ul * i);
    return AttnRowRec(w.x, w.y, w.z, w.w);
}

// --- the cluster of 32 -----------------------------------------------------------------

uint lane32() { return gl_SubgroupInvocationID & 31u; }
// The subgroup index of lane `i` of the invocation's cluster.
uint cluster_lane(uint i) { return (gl_SubgroupInvocationID & ~31u) | i; }

float cl_max(float v) { return subgroupClusteredMax(v, 32); }

// The sum of 32 values in the specified order; every lane gets the same bits.
float sum32(float v) {
    [[unroll]] for (uint h = 16u; h > 0u; h >>= 1) {
        precise float s = v + subgroupShuffleXor(v, h);
        v = s;
    }
    return v;
}

// The 32-point Walsh-Hadamard transform of the group, scaled by 1/sqrt(32): stage widths 1 to
// 16, the lower element gets the sum, the upper the difference (old values), then one
// multiply.
float rotate32(float v, uint lane) {
    [[unroll]] for (uint h = 1u; h < 32u; h <<= 1) {
        const float o = subgroupShuffleXor(v, h);
        precise float r = ((lane & h) != 0u) ? (o - v) : (v + o);
        v = r;
    }
    precise float scaled = v * uintBitsToFloat(0x3e3504f3u);
    return scaled;
}

// --- half precision, on the bits ------------------------------------------------------

float h2f(uint bits) { return unpackHalf2x16(bits & 0xffffu).x; }

// The f16 pattern nearest to x (ties to even), as `__float2half_rn`: overflow gives infinity.
uint f2h(float x) {
    const uint u = floatBitsToUint(x);
    const uint sign = (u >> 16) & 0x8000u;
    const uint e = (u >> 23) & 0xffu;
    uint m = u & 0x7fffffu;
    if (e == 255u) {
        return sign | 0x7c00u | (m != 0u ? (0x200u | (m >> 13)) : 0u);
    }
    const int ex = int(e) - 127 + 15;
    if (ex >= 31) {
        return sign | 0x7c00u;
    }
    if (ex <= 0) {
        // Subnormal or zero: the 24-bit significand shifted into the 10 bits below 2^-14.
        if (ex < -10) {
            return sign;
        }
        m |= 0x800000u;
        const uint shift = uint(14 - ex);
        uint half_bits = m >> shift;
        const uint rem = m & ((1u << shift) - 1u);
        const uint mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (half_bits & 1u) != 0u)) {
            half_bits++;
        }
        return sign | half_bits;
    }
    uint half_bits = (uint(ex) << 10) | (m >> 13);
    const uint rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_bits & 1u) != 0u)) {
        half_bits++;
    }
    return sign | half_bits;
}

bool is_finite_f(float v) { return (floatBitsToUint(v) & 0x7f800000u) != 0x7f800000u; }

// One f32 value of a row to its f16 pattern, saturating at +-65504.
uint to_f16_saturating(float v) { return f2h(min(max(v, -65504.0), 65504.0)); }

// The stored scale (an f16 pattern) of a group whose largest rotated magnitude is `amax`: the
// smallest positive f16 not below amax / qmax.
uint scale_bits(float amax, uint bits) {
    if (amax == 0.0) {
        return 0u;
    }
    const float q = div_rn(amax, float((1u << (bits - 1u)) - 1u));
    const uint nearest = f2h(q);
    if (nearest >= 0x7c00u) {
        return 0x7bffu;
    }
    const uint scale = (nearest == 0u || h2f(nearest) < q) ? nearest + 1u : nearest;
    return min(scale, 0x7bffu);
}

// --- the codes in bit planes -----------------------------------------------------------

// Writes the 32 codes (invocation i holds code i, a `bits`-bit two's complement number) in bit
// planes: bits 0 to 3, then bits 4 and 5, then the top bit. `dst` is 4 * bits bytes at an even
// address. A plane is gathered into words by clustered ORs (the byte j of a word holds the
// fields of the codes 2j, 2j + 1 or 4j .. 4j + 3, little endian), and the first invocation of
// each cluster of the word's size writes it.
void pack_codes(int code, uint bits, uint64_t dst) {
    const uint u = uint(code);
    const uint lane = lane32();
    if (bits == 8u) {
        B1(dst + lane).v[0] = uint8_t(u & 0xffu);
        return;
    }
    {
        const uint w = subgroupClusteredOr((u & 0xfu) << (4u * (lane & 7u)), 8u);
        if ((lane & 7u) == 0u) {
            B2(dst + 4ul * (lane >> 3)).v[0] = uint16_t(w & 0xffffu);
            B2(dst + 4ul * (lane >> 3) + 2ul).v[0] = uint16_t(w >> 16);
        }
    }
    if (bits >= 6u) {
        const uint w = subgroupClusteredOr(((u >> 4) & 3u) << (2u * (lane & 15u)), 16u);
        if ((lane & 15u) == 0u) {
            B2(dst + 16ul + 4ul * (lane >> 4)).v[0] = uint16_t(w & 0xffffu);
            B2(dst + 16ul + 4ul * (lane >> 4) + 2ul).v[0] = uint16_t(w >> 16);
        }
        if (bits == 7u) {
            const uint t = subgroupClusteredOr(((u >> 6) & 1u) << lane, 32u);
            if (lane == 0u) {
                B2(dst + 24ul).v[0] = uint16_t(t & 0xffffu);
                B2(dst + 26ul).v[0] = uint16_t(t >> 16);
            }
        }
    } else if (bits == 5u) {
        const uint t = subgroupClusteredOr(((u >> 4) & 1u) << lane, 32u);
        if (lane == 0u) {
            B2(dst + 16ul).v[0] = uint16_t(t & 0xffffu);
            B2(dst + 18ul).v[0] = uint16_t(t >> 16);
        }
    }
}

// Encodes the rotated (and, for a tile, mean-subtracted) group `r` (invocation i holds r[i])
// into `dst`, 2 + 4 * bits bytes: steps 3 and 4 of the specification, the rule of q8 or the
// search of q7 to q4.
void encode_group(float r, uint bits, uint64_t dst) {
    const float amax = cl_max(abs(r));
    uint scale = 0u;
    int code = 0;
    if (amax == 0.0) {
        // scale +0, codes 0
    } else if (bits == 8u) {
        scale = scale_bits(amax, 8u);
        const float c = roundEven(div_rn(r, h2f(scale)));
        code = int(min(max(c, -127.0), 127.0));
    } else {
        const float qmax = float((1u << (bits - 1u)) - 1u);
        const float pmax = cl_max(max(r, 0.0));
        const float nmax = cl_max(max(-r, 0.0));
        const float sgn = (nmax >= pmax) ? 1.0 : -1.0;
        const uint sign_bit = (sgn < 0.0) ? 0x8000u : 0u;
        const int steps = 1 << (8u - bits);
        float best_error = 0.0;
        uint best_scale = 0u;
        int best_code = 0;
        for (int j = 0; j <= steps; ++j) {
            precise float candidate = amax * (float(64 - j) / 64.0);
            const uint base = scale_bits(candidate, bits);
            const float grid = sgn * h2f(base);
            float c = roundEven(div_rn(r, grid));
            c = min(max(c, -qmax - 1.0), qmax);
            const int ci = int(c);
            precise float product = r * c;
            const int squares = subgroupClusteredAdd(ci * ci, 32);
            uint s_bits;
            if (squares == 0) {
                s_bits = base;
            } else {
                const float p = sum32(product);
                const float fitted = min(div_rn(abs(p), float(squares)), 65504.0);
                s_bits = max(f2h(fitted), 1u) | sign_bit;
            }
            precise float scaled = float(ci) * h2f(s_bits);
            precise float d = r - scaled;
            precise float dd = d * d;
            const float error = sum32(dd);
            if (j == 0 || error < best_error) {
                best_error = error;
                best_scale = s_bits;
                best_code = ci;
            }
        }
        scale = best_scale;
        code = best_code;
    }
    if (lane32() == 0u) {
        B2(dst).v[0] = uint16_t(scale);
    }
    pack_codes(code, bits, dst + 2ul);
}

// Encodes one group of 32 values (invocation i holds value i, unrotated) in `fmt` at `dst`.
// Flags a non-finite input or an overflowing rotation in `status`.
void encode_group_of(uint fmt, float v, uint64_t dst, uint64_t status) {
    if (!is_finite_f(v)) {
        atomicOr(B4(status).v[0], STATUS_NONFINITE);
    }
    if (fmt == FMT_F16) {
        B2(dst + 2ul * lane32()).v[0] = uint16_t(to_f16_saturating(v));
        return;
    }
    const float r = rotate32(v, lane32());
    if (!is_finite_f(r) && is_finite_f(v)) {
        atomicOr(B4(status).v[0], STATUS_OVERFLOW);
    }
    encode_group(r, fmt_bits(fmt), dst);
}

// Encodes one row of D f32 values at `src` in `fmt` at `dst`.
void encode_row_f32(uint fmt, uint64_t src, uint64_t dst, uint D, uint64_t status) {
    const uint gb = fmt_group_bytes(fmt);
    for (uint g = 0u; g < D / 32u; ++g) {
        encode_group_of(fmt, ldf(src + 4ul * (g * 32u + lane32())), dst + uint64_t(g * gb), status);
    }
}

#endif
