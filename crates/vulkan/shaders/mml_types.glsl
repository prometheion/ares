// The device layouts of the weight types that have one (`ares_quant::layout`, the same
// bytes as CUDA's `kernels/mmr_codes.cuh`): what a kernel or the rearrangement needs to
// know of a type, selected by LT_TYPE (given by build.rs).
//
//   LT_LB, LT_HB       bits of a code in the low and in the high plane
//   LT_HDR             what the header section holds (LT_HDR_K4, _K16, _B, _X4)
//   LT_TABLE           the code is a 4-bit index of the 16 levels of IQ4_NL
//   LT_BLOCK_VALUES, LT_BLOCK_BYTES   the file's block
//   K16: LT_SCALE_WORDS words of scales per row and super-block (at LT_SCALE_AT of the
//        file's block), d at LT_D_AT, a scale byte is the scale + LT_SCALE_BIAS, the value
//        S * (code - LT_CODE_OFFSET); LT_MIN_NIBBLE: the high nibble of the scale byte is a
//        minimum instead (Q2_K)
//   B:   LT_M: a minimum m per block behind d; else the value is d * (code - LT_CODE_OFFSET)
//
// A value is S * code + C with S and C of its group of 16 or 32 values:
//   Q4_K, Q5_K   d * sc, -dmin * m           Q6_K   d * sc, -32 S       Q3_K   d * (sc - 32), -4 S
//   Q2_K         d * sc, -dmin * m           Q4_0   d, -8 S             Q4_1, Q5_1   d, m
//   Q5_0         d, -16 S                    Q8_0   d, -128 S (the code is q + 128)
//   IQ4_XS       d * (ls - 32) * level       IQ4_NL d * level

#define LT_HDR_K4 0
#define LT_HDR_K16 1
#define LT_HDR_B 2
#define LT_HDR_X4 3

#define LT_Q4_K 0
#define LT_Q5_K 1
#define LT_Q6_K 2
#define LT_Q3_K 3
#define LT_Q2_K 4
#define LT_Q4_0 5
#define LT_Q4_1 6
#define LT_Q5_0 7
#define LT_Q5_1 8
#define LT_Q8_0 9
#define LT_IQ4_XS 10
#define LT_IQ4_NL 11

#define LT_HB 0
#define LT_TABLE 0
#define LT_M 0
#define LT_MIN_NIBBLE 0
#define LT_SCALE_BIAS 0
#define LT_CODE_OFFSET 0
#define LT_SCALE_WORDS 4
#define LT_SCALE_AT 0
#define LT_D_AT 0

#if LT_TYPE == LT_Q4_K
#define LT_LB 4
#define LT_HDR LT_HDR_K4
#define LT_BLOCK_VALUES 256
#define LT_BLOCK_BYTES 144
#elif LT_TYPE == LT_Q5_K
#define LT_LB 4
#undef LT_HB
#define LT_HB 1
#define LT_HDR LT_HDR_K4
#define LT_BLOCK_VALUES 256
#define LT_BLOCK_BYTES 176
#elif LT_TYPE == LT_Q6_K
#define LT_LB 4
#undef LT_HB
#define LT_HB 2
#define LT_HDR LT_HDR_K16
#undef LT_SCALE_AT
#define LT_SCALE_AT 192
#undef LT_D_AT
#define LT_D_AT 208
#undef LT_SCALE_BIAS
#define LT_SCALE_BIAS 128
#undef LT_CODE_OFFSET
#define LT_CODE_OFFSET 32
#define LT_BLOCK_VALUES 256
#define LT_BLOCK_BYTES 210
#elif LT_TYPE == LT_Q3_K
#define LT_LB 2
#undef LT_HB
#define LT_HB 1
#define LT_HDR LT_HDR_K16
#undef LT_SCALE_WORDS
#define LT_SCALE_WORDS 3
#undef LT_SCALE_AT
#define LT_SCALE_AT 96
#undef LT_D_AT
#define LT_D_AT 108
#undef LT_SCALE_BIAS
#define LT_SCALE_BIAS 32
#undef LT_CODE_OFFSET
#define LT_CODE_OFFSET 4
#define LT_BLOCK_VALUES 256
#define LT_BLOCK_BYTES 110
#elif LT_TYPE == LT_Q2_K
#define LT_LB 2
#define LT_HDR LT_HDR_K16
#undef LT_D_AT
#define LT_D_AT 80
#undef LT_MIN_NIBBLE
#define LT_MIN_NIBBLE 1
#define LT_BLOCK_VALUES 256
#define LT_BLOCK_BYTES 84
#elif LT_TYPE == LT_Q4_0
#define LT_LB 4
#define LT_HDR LT_HDR_B
#undef LT_CODE_OFFSET
#define LT_CODE_OFFSET 8
#define LT_BLOCK_VALUES 32
#define LT_BLOCK_BYTES 18
#elif LT_TYPE == LT_Q4_1
#define LT_LB 4
#define LT_HDR LT_HDR_B
#undef LT_M
#define LT_M 1
#define LT_BLOCK_VALUES 32
#define LT_BLOCK_BYTES 20
#elif LT_TYPE == LT_Q5_0
#define LT_LB 4
#undef LT_HB
#define LT_HB 1
#define LT_HDR LT_HDR_B
#undef LT_CODE_OFFSET
#define LT_CODE_OFFSET 16
#define LT_BLOCK_VALUES 32
#define LT_BLOCK_BYTES 22
#elif LT_TYPE == LT_Q5_1
#define LT_LB 4
#undef LT_HB
#define LT_HB 1
#define LT_HDR LT_HDR_B
#undef LT_M
#define LT_M 1
#define LT_BLOCK_VALUES 32
#define LT_BLOCK_BYTES 24
#elif LT_TYPE == LT_Q8_0
#define LT_LB 8
#define LT_HDR LT_HDR_B
#undef LT_CODE_OFFSET
#define LT_CODE_OFFSET 128
#define LT_BLOCK_VALUES 32
#define LT_BLOCK_BYTES 34
#elif LT_TYPE == LT_IQ4_XS
#define LT_LB 4
#define LT_HDR LT_HDR_X4
#undef LT_TABLE
#define LT_TABLE 1
#define LT_BLOCK_VALUES 256
#define LT_BLOCK_BYTES 136
#elif LT_TYPE == LT_IQ4_NL
#define LT_LB 4
#define LT_HDR LT_HDR_B
#undef LT_TABLE
#define LT_TABLE 1
#define LT_BLOCK_VALUES 32
#define LT_BLOCK_BYTES 18
#else
#error "LT_TYPE names no type with a device layout"
#endif

// Of the words a lane holds of a plane of `bits` bits per code for one step: the word and
// the bit of the even value of register a_i of slab kk (`ares_quant::layout`).
uint lt_plane_word(uint bits, uint kk, uint i) {
    return bits == 8u ? 2u * kk + i / 2u : bits == 4u ? kk : bits == 2u ? kk / 2u : 0u;
}
uint lt_plane_shift(uint bits, uint kk, uint i) {
    return bits == 8u ? 8u * (i % 2u)
                      : bits == 4u ? 4u * i : bits == 2u ? 8u * (kk % 2u) + 2u * i : 4u * kk + i;
}
