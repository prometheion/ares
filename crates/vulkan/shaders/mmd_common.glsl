// Helpers of the weight decoders (`mmd_<type>.glsl`) of the f16 prefill kernels in the
// file's layout. A decoder defines the block geometry (BLOCK_VALUES, BLOCK_BYTES) and
//
//   void md_unit(uint64_t row, uint u, out uint o[16])
//
// which decodes the 32 values 32u .. 32u+31 of the weight row at address `row` into 16
// words of two f16 each, the lower value in the low half. The kernel runs it for the two
// units of every step of 64 values, ahead of the step's products.
//
// Codes come out of the blocks as bytes, four to a word in the order of the values; an
// f16 holds any integer below 2048 exactly, so a pair of bytes becomes a pair of f16 with
// one `or` of 0x6400 (the f16 1024 + code) and one subtraction, and the scale is applied
// with a fused multiply-add on pairs.

// (mmvq_common.glsl is included by mm_f16_head.glsl, ahead of this file.)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#include "iq_tables.glsl"

// Bytes 0 and 1 of w, and bytes 2 and 3, as f16 pairs (unsigned, 0..255).
f16vec2 u8_01(uint w) {
    return unpackFloat2x16(((w & 0xFFu) | ((w << 8) & 0x00FF0000u)) | 0x64006400u) -
           f16vec2(1024.0hf);
}
f16vec2 u8_23(uint w) {
    return unpackFloat2x16((((w >> 16) & 0xFFu) | ((w >> 8) & 0x00FF0000u)) | 0x64006400u) -
           f16vec2(1024.0hf);
}
// The same for signed bytes.
f16vec2 s8_01(uint w) { return u8_01(w ^ 0x80808080u) - f16vec2(128.0hf); }
f16vec2 s8_23(uint w) { return u8_23(w ^ 0x80808080u) - f16vec2(128.0hf); }

// A decoder of a type whose 4-bit codes index a table of 16 signed bytes defines MD_TABLE16
// (the table, a const int[16]) before including this file. The table goes to shared memory,
// 16 entries by 32 lanes, entry n of lane l at n * 32 + l: lane l always reads bank l, so
// any 32 indices are conflict free (a constant array indexed differently by each lane is
// read one distinct index at a time). The kernel calls md_init() once and puts a barrier
// behind it.
#ifdef MD_TABLE16
shared uint kvlut[16 * 32];
void md_init() {
    for (uint i = tid; i < 16 * 32; i += 128) {
        kvlut[i] = uint(MD_TABLE16[i >> 5]) & 0xFFu;
    }
}
// The table entry n (0..15) as a signed integer.
int lut16(uint n) {
    return int(kvlut[(n << 5) | (tid & 31u)] << 24) >> 24;
}
#else
void md_init() {}
#endif

uint pack2(f16vec2 v) { return packFloat2x16(v); }
f16vec2 h2(float x) { return f16vec2(float16_t(x)); }

// Eight words (32 bytes) from an address that is only 2-byte aligned: nine aligned words
// and funnel shifts; the ninth is only read when the address is not word aligned.
void ld32_any(uint64_t a, out uint w[8]) {
    const uint64_t base = a & ~uint64_t(3);
    const uint shift = uint(a & uint64_t(3)) * 8u;
    if (shift == 0u) {
        [[unroll]] for (uint i = 0; i < 8; i++) {
            w[i] = ld4(base + 4 * i);
        }
    } else {
        uint r[9];
        [[unroll]] for (uint i = 0; i < 9; i++) {
            r[i] = ld4(base + 4 * i);
        }
        [[unroll]] for (uint i = 0; i < 8; i++) {
            w[i] = (r[i] >> shift) | (r[i + 1] << (32u - shift));
        }
    }
}

// The scales and minimums of sub-blocks 2j and 2j+1 (two bytes each) from the three
// words of packed scales. Sub-blocks 0..3 keep 6 bits in bytes 0..3 (scales) and
// 4..7 (minimums); sub-blocks 4..7 keep 4 bits in bytes 8..11 and the other 2 in
// the top bits of bytes 0..7. `hi` selects between the two without a branch.
void scales_k(uint j, uvec3 s, out uint sc, out uint mn) {
    const uint sh = 16 * (j & 1);
    const uint t0 = (s.x >> sh) & 0xFFFFu;
    const uint t1 = (s.y >> sh) & 0xFFFFu;
    const uint t2 = (s.z >> sh) & 0xFFFFu;
    const uint hi = 0u - (j >> 1);
    sc = ((t0 & 0x3F3Fu) & ~hi) | (((t2 & 0x0F0Fu) | ((t0 & 0xC0C0u) >> 2)) & hi);
    mn = ((t1 & 0x3F3Fu) & ~hi) | ((((t2 >> 4) & 0x0F0Fu) | ((t1 & 0xC0C0u) >> 2)) & hi);
}


// --- helpers of the i-quants (the same as mmvq_iq_common.glsl) -----------------------------

// The sign bits of a group of 8 values from a 7-bit index: bit 7 makes the number of
// negative values even (ggml's ksigns_iq2xs, without the table).
uint ksigns(uint v7) {
    return v7 ^ ((bitCount(v7) & 1u) << 7);
}

// Sign masks (0xFF in the bytes to negate) for four values: `spread` has the sign byte in each
// of its four bytes, `select` the bit of each value in the byte.
uint sign_mask4(uint spread, uint select) {
    return (((spread & select) + 0x7F7F7F7Fu) >> 7 & 0x01010101u) * 255u;
}

// The four magnitudes (bytes, all in 1..127) with the signs of `mask` applied.
uint apply_sign4(uint magnitudes, uint mask) {
    return (magnitudes ^ mask) + (mask & 0x01010101u);
}

// The four 2-bit fields of the low byte of x as four bytes, field 0 in byte 0.
uint spread_pairs(uint x) {
    x &= 0xFFu;
    x = (x | (x << 12)) & 0x000F000Fu;
    return (x | (x << 6)) & 0x03030303u;
}

// Writes the 32 signed bytes q[0..7] (four to a word) scaled by the f16 pairs s_a (the first
// 16 values) and s_b (the others) as f16 pairs.
void emit_signed(uint q[8], f16vec2 s_a, f16vec2 s_b, out uint o[16]) {
    [[unroll]] for (uint i = 0; i < 8; i++) {
        const f16vec2 s = i < 4 ? s_a : s_b;
        o[2 * i] = pack2(s8_01(q[i]) * s);
        o[2 * i + 1] = pack2(s8_23(q[i]) * s);
    }
}
