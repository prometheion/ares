// The conversion of the rows of one side of one tile into a format (attn_convert.comp and
// attn_encode_tile_f32.comp): the mean of the rotated rows for q7 to q4, then every row's
// groups encoded. A workgroup of 256 invocations (8 clusters) calls it, all of them.
#include "attn_common.glsl"

// The largest head size the including shader converts (it sizes the shared mean).
#ifndef ATTN_MAX_D
#define ATTN_MAX_D 256
#endif
shared float mean_s[ATTN_MAX_D];

// `value(row, col)` is the stored value to convert: the including shader defines it, from
// the base address and row length it sets in these two before calling `convert_side`.
uint64_t src_base;
uint src_d;
float value(uint row, uint col);
void convert_side(
    uint rows, uint D, uint fmt, bool use_mean, uint64_t mean_out, uint64_t rows_out,
    uint64_t status) {
    const uint warp = gl_LocalInvocationID.x >> 5, lane = lane32(), nwarps = 8u;
    const uint groups = D / 32u, gb = fmt_group_bytes(fmt), rb = groups * gb;
    const bool mean = fmt_has_mean(fmt) && use_mean;
    if (mean) {
        for (uint g = warp; g < groups; g += nwarps) {
            float sum = 0.0;
            for (uint i = 0u; i < rows; ++i) {
                const float r = rotate32(value(i, g * 32u + lane), lane);
                precise float next = sum + r;
                sum = next;
            }
            const float m = min(max(div_rn(sum, float(rows)), -65504.0), 65504.0);
            const uint bits = f2h(m);
            B2(mean_out + 2ul * (g * 32u + lane)).v[0] = uint16_t(bits);
            mean_s[g * 32u + lane] = h2f(bits);
        }
        barrier();
    }
    for (uint i = warp; i < rows; i += nwarps) {
        for (uint g = 0u; g < groups; ++g) {
            const float v = value(i, g * 32u + lane);
            const uint64_t dst = rows_out + uint64_t(i) * rb + uint64_t(g) * gb;
            if (!is_finite_f(v)) {
                atomicOr(B4(status).v[0], STATUS_NONFINITE);
            }
            if (fmt == FMT_F16) {
                B2(dst + 2ul * lane).v[0] = uint16_t(to_f16_saturating(v));
                continue;
            }
            float r = rotate32(v, lane);
            if (!is_finite_f(r) && is_finite_f(v)) {
                atomicOr(B4(status).v[0], STATUS_OVERFLOW);
            }
            if (mean) {
                precise float centred = r - mean_s[g * 32u + lane];
                r = centred;
            }
            encode_group(r, fmt_bits(fmt), dst);
        }
    }
}
