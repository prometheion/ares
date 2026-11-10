// Helpers of the i-quant types (mmvq_iq*.glsl); `mmvq_iquants.cuh` is the model.
//
// IQ4_XS and IQ4_NL map each nibble through a table of 16 signed bytes; the other
// i-quants look 4 or 8 magnitudes up in a codebook (iq_tables.glsl) by 8 to 10 bits
// of the block and give them signs, so both end as signed bytes that the dot
// product multiplies with the int8 activations.
//
// All scales of these types are small integers (or integers plus a half) times the
// f16 d of the block, so the integer dot products are scaled in integers and the
// float multiplication happens once per sub-block:
//   dot = d * dx * (integer scale) * sum(w * qx) * constant
// exactly, with no rounding before the final products.
//
// What CUDA does with one instruction each, GLSL has no operation for, so:
// - the byte permutes (PRMT) that look 8 nibbles up in the 16-entry table are
//   lookups in a table in shared memory, laid out so that a lookup never has a bank
//   conflict (see `table16`);
// - the signs: the per-byte compare (VCMPNE4) and subtract (VSUB4) are a few
//   integer operations on the whole word, exact because no grid byte is 0.

#include "iq_tables.glsl"

// Which of the 32 lanes of its warp the invocation is (the subgroup invocation
// modulo 32). A table whose entries every lane reads at its own index is laid out
// by lane.
uint mmvq_lane;

// The sign bits of a group of 8 values from a 7-bit index: bit 7 makes the number
// of negative values even (ggml's ksigns_iq2xs, without the table).
uint ksigns(uint v7) {
    return v7 ^ ((bitCount(v7) & 1u) << 7);
}

// Sign masks (0xFF in the bytes to negate) for four values: `spread` has the sign
// byte in each of its four bytes, `select` the bit of each value in the byte. The
// bytes of `spread & select` are 0 or a single bit; adding 0x7F to such a byte
// carries into bit 7 exactly when it is not 0, and nothing carries out of the byte.
uint sign_mask4(uint spread, uint select) {
    return (((spread & select) + 0x7F7F7F7Fu) >> 7 & 0x01010101u) * 255u;
}

// The four magnitudes (bytes, all in 1..127) with the signs of `mask` applied:
// negation per byte is invert and add one, and no byte of a magnitude is 0, so the
// addition never carries into the next byte.
uint apply_sign4(uint magnitudes, uint mask) {
    return (magnitudes ^ mask) + (mask & 0x01010101u);
}

// The four 2-bit fields of the low byte of x as four bytes, field 0 in byte 0.
uint spread_pairs(uint x) {
    x &= 0xFFu;
    x = (x | (x << 12)) & 0x000F000Fu;
    return (x | (x << 6)) & 0x03030303u;
}

// The 16 levels of IQ4_NL (KVALUES_IQ4NL in crates/quant/src/tables.rs) for the 4-bit
// indices in the bytes of `q4`: x gets the table entries of the low nibbles of the 4
// bytes, y those of the high nibbles.
//
// The table lives in shared memory, 16 entries by 32 lanes, entry n of lane l at
// n * 32 + l: lane l always reads bank l, so any 32 indices are conflict free, and
// the compiler can address it with the lane's constant offset.
shared uint kvlut[16 * 32];

// IQ4_NL's levels, unless the including file says which 16 signed bytes it maps
// nibbles to (the 4-bit float types do).
#ifndef KV_TABLE
#define KV_TABLE ares_kvalues_iq4nl
#endif

void iq4_init(uint t) {
    for (uint i = t; i < 16 * 32; i += ARES_MMVQ_WARPS * 32) {
        kvlut[i] = uint(KV_TABLE[i >> 5]) & 0xFFu;
    }
    barrier();
}

void table16(uint q4, out int x, out int y) {
    uint lows = 0u;
    uint highs = 0u;
    [[unroll]] for (uint k = 0; k < 4; k++) {
        const uint nl = bitfieldExtract(q4, int(8 * k), 4);
        const uint nh = bitfieldExtract(q4, int(8 * k + 4), 4);
        lows = bitfieldInsert(lows, kvlut[(nl << 5) | mmvq_lane], int(8 * k), 8);
        highs = bitfieldInsert(highs, kvlut[(nh << 5) | mmvq_lane], int(8 * k), 8);
    }
    x = int(lows);
    y = int(highs);
}

// What a lane holds of the activations of a sub-block of 32 values.
struct SubA {
    uvec4 first;   // activations 0..15
    uvec4 second;  // 16..31
    vec2 ds;
};

SubA load_sub_a(uint64_t q, uint64_t ds, uint ib) {
    SubA a;
    a.first = ld16(q + 32 * ib);
    a.second = ld16(q + 32 * ib + 16);
    a.ds = ldf2(ds + 8 * ib);
    return a;
}
