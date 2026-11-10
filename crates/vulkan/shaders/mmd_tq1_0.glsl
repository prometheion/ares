// TQ1_0 for the f16 prefill kernels: 256 values in 54 bytes (2-byte aligned): u8 qs[48], u8
// qh[4], f16 d at 52. Every value is -1, 0 or +1 times d; five values (base-3 digits, 0..2
// for -1..1) share a byte of qs, four share a byte of qh. The digit n of byte m of the first
// 32 bytes of qs is value 32 n + m, of the next 16 bytes value 160 + 16 n + m, and the digit n
// of byte 48 + j of qh is value 240 + 4 n + j. A digit is read by multiplying the byte by
// 3^n (mod 256) and taking the top of 3 times that. A unit is 32 values: units 0..4 are the
// planes of the first run, 5 and 6 two planes of the second each, 7 the last plane of the
// second run and the four planes of qh.
#include "mmd_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 54;

// The digit p of each of the four bytes of w (p = 3^n), as bytes of 0 .. 2.
uint tq1_digits(uint w, uint p) {
    uint even = ((w & 0x00FF00FFu) * p) & 0x00FF00FFu;
    uint odd = (((w >> 8) & 0x00FF00FFu) * p) & 0x00FF00FFu;
    even = ((even * 3u) >> 8) & 0x00030003u;
    odd = ((odd * 3u) >> 8) & 0x00030003u;
    return even | (odd << 8);
}

uint pow3(uint n) {
    return n == 0u ? 1u : n == 1u ? 3u : n == 2u ? 9u : n == 3u ? 27u : 81u;
}

void md_unit(uint64_t row, uint u, out uint o[16]) {
    const uint64_t blk = row + uint64_t(u >> 3) * BLOCK_BYTES;
    const uint within = u & 7u;
    const f16vec2 d = h2(half_of(ld2(blk + 52)));
    uint dig[8];
    if (within < 5u) {
        uint q[8];
        ld32_any(blk, q);
        [[unroll]] for (uint i = 0; i < 8; i++) {
            dig[i] = tq1_digits(q[i], pow3(within));
        }
    } else if (within < 7u) {
        // Two planes of the next 16 bytes: n = 2 (within - 5) and the one after.
        const uint n = 2 * (within - 5u);
        [[unroll]] for (uint i = 0; i < 4; i++) {
            const uint w = ld4_a2(blk + 32 + 4 * i);
            dig[i] = tq1_digits(w, pow3(n));
            dig[4 + i] = tq1_digits(w, pow3(n + 1));
        }
    } else {
        // The last plane (n = 4) of the next 16 bytes, then the four planes of qh.
        [[unroll]] for (uint i = 0; i < 4; i++) {
            dig[i] = tq1_digits(ld4_a2(blk + 32 + 4 * i), 81u);
        }
        const uint qh = ld4_a2(blk + 48);
        [[unroll]] for (uint n = 0; n < 4; n++) {
            dig[4 + n] = tq1_digits(qh, pow3(n));
        }
    }
    [[unroll]] for (uint i = 0; i < 8; i++) {
        o[2 * i] = pack2((u8_01(dig[i]) - f16vec2(1.0hf)) * d);
        o[2 * i + 1] = pack2((u8_23(dig[i]) - f16vec2(1.0hf)) * d);
    }
}
