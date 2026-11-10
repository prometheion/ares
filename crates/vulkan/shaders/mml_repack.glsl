#version 450
#include "mmvq_common.glsl"
#include "mml_types.glsl"

// The rearrangement of a matrix from the file's layout into its device layout
// (`ares_quant::layout::rearrange` is the reference, byte for byte; CUDA's
// `ares_repack_<type>` makes the same bytes): one invocation per 32-bit word of the layout,
// which gathers what the word holds from the file's bytes. It runs once per matrix, at
// load. build.rs compiles it once per type (LT_TYPE) into `mml_repack_<type>.spv`.
//
//   src  the matrix as the file stores it
//   dst  the layout: the bytes of the file's with the rows rounded up to a multiple of 16
// Grid: one invocation per word, 256 to a workgroup.

layout(local_size_x = 256) in;

layout(push_constant, std430) uniform Push {
    uint64_t src;
    uint64_t dst;
    uint n_out;
    uint n_in;
} pc;

layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer Out4 { uint v[]; };

uint64_t row_bytes;

// The address of block `block` of row `row` in the file's layout.
uint64_t blk(uint row, uint block) {
    return pc.src + uint64_t(row) * row_bytes + uint64_t(block) * LT_BLOCK_BYTES;
}

// 16 and 32 bits of the file's block; rows past the end of the matrix read as zero.
uint u16_at(uint row, uint block, uint off) {
    if (row >= pc.n_out) {
        return 0u;
    }
    const uint64_t p = blk(row, block) + off;
    return ld1(p) | (ld1(p + 1) << 8);
}
uint u32_at(uint row, uint block, uint off) {
    return u16_at(row, block, off) | (u16_at(row, block, off + 2) << 16);
}

// The nibbles of a block of 32 values at `at`: values 0..15 in the low, 16..31 in the high.
uint nibble32(uint64_t b, uint at, uint k) {
    const uint byte = ld1(b + at + (k & 15u));
    return (k & 16u) != 0u ? byte >> 4 : byte & 15u;
}

// The code of value k of the file's block at b.
uint code_of(uint64_t b, uint k) {
#if LT_TYPE == LT_Q4_K
    const uint byte = ld1(b + 16 + 32 * (k / 64) + (k & 31u));
    return (k & 32u) != 0u ? byte >> 4 : byte & 15u;
#elif LT_TYPE == LT_Q5_K
    const uint byte = ld1(b + 48 + 32 * (k / 64) + (k & 31u));
    const uint low = (k & 32u) != 0u ? byte >> 4 : byte & 15u;
    return low | (((ld1(b + 16 + (k & 31u)) >> (2 * (k / 64) + ((k >> 5) & 1u))) & 1u) << 4);
#elif LT_TYPE == LT_Q6_K
    const uint half_ = k / 128, part = (k / 32) % 4, l = k % 32;
    const uint byte = ld1(b + 64 * half_ + l + 32 * (part & 1u));
    const uint low = part < 2 ? byte & 15u : byte >> 4;
    return low | (((ld1(b + 128 + 32 * half_ + l) >> (2 * part)) & 3u) << 4);
#elif LT_TYPE == LT_Q3_K
    const uint half_ = k / 128, step = (k / 32) % 4, l = k % 32;
    return ((ld1(b + 32 + 32 * half_ + l) >> (2 * step)) & 3u) |
           (((ld1(b + l) >> (4 * half_ + step)) & 1u) << 2);
#elif LT_TYPE == LT_Q2_K
    return (ld1(b + 16 + 32 * (k / 128) + k % 32) >> (2 * ((k / 32) % 4))) & 3u;
#elif LT_TYPE == LT_Q4_0 || LT_TYPE == LT_IQ4_NL
    return nibble32(b, 2, k);
#elif LT_TYPE == LT_Q4_1
    return nibble32(b, 4, k);
#elif LT_TYPE == LT_Q5_0
    const uint qh = ld1(b + 2) | (ld1(b + 3) << 8) | (ld1(b + 4) << 16) | (ld1(b + 5) << 24);
    return nibble32(b, 6, k) | (((qh >> k) & 1u) << 4);
#elif LT_TYPE == LT_Q5_1
    const uint qh = ld1(b + 4) | (ld1(b + 5) << 8) | (ld1(b + 6) << 16) | (ld1(b + 7) << 24);
    return nibble32(b, 8, k) | (((qh >> k) & 1u) << 4);
#elif LT_TYPE == LT_Q8_0
    return ld1(b + 2 + k) ^ 0x80u;
#elif LT_TYPE == LT_IQ4_XS
    return nibble32(b + 16 * (k / 32), 8, k % 32);
#endif
}

// Word `idx` of the header section.
uint header_word(uint64_t idx, uint64_t n_tiles) {
    const uint n_sb = pc.n_in / 256;
    const uint n_steps = pc.n_in / 64;
#if LT_HDR == LT_HDR_K4 || LT_HDR == LT_HDR_X4
    const uint HW = LT_HDR == LT_HDR_K4 ? 4 : 2;
    const uint wi = uint(idx % HW), r16 = uint((idx / HW) % 16);
    const uint sb = uint((idx / (HW * 16)) % n_sb), tile = uint(idx / (HW * 16) / n_sb);
    return u32_at(tile * 16 + r16, sb, 4 * wi);
#elif LT_HDR == LT_HDR_K16
    const uint DW = 1 + LT_MIN_NIBBLE;
    const uint64_t n_scale = n_tiles * n_sb * 16 * LT_SCALE_WORDS;
    if (idx < n_scale) {
        const uint wi = uint(idx % LT_SCALE_WORDS), r16 = uint((idx / LT_SCALE_WORDS) % 16);
        const uint sb = uint((idx / LT_SCALE_WORDS / 16) % n_sb);
        const uint tile = uint(idx / LT_SCALE_WORDS / 16 / n_sb);
        const uint v = u32_at(tile * 16 + r16, sb, LT_SCALE_AT + 4 * wi);
        return LT_SCALE_BIAS == 128 ? v ^ 0x80808080u : v;
    }
    const uint64_t di = idx - n_scale;
    const uint wi = uint(di % DW), g = uint((di / DW) % 8);
    const uint sb = uint((di / DW / 8) % n_sb), tile = uint(di / DW / 8 / n_sb);
    return u16_at(tile * 16 + g, sb, LT_D_AT + 2 * wi) |
           (u16_at(tile * 16 + g + 8, sb, LT_D_AT + 2 * wi) << 16);
#else
    const uint MW = 1 + LT_M;
    const uint wi = uint(idx % MW), sub = uint((idx / MW) % 2), g = uint((idx / MW / 2) % 8);
    const uint step = uint((idx / MW / 16) % n_steps), tile = uint(idx / MW / 16 / n_steps);
    return u16_at(tile * 16 + g, 2 * step + sub, 2 * wi) |
           (u16_at(tile * 16 + g + 8, 2 * step + sub, 2 * wi) << 16);
#endif
}

void main() {
    const uint64_t n_tiles = (uint64_t(pc.n_out) + 15) / 16;
    const uint n_steps = pc.n_in / 64;
    row_bytes = uint64_t(pc.n_in / LT_BLOCK_VALUES) * LT_BLOCK_BYTES;
    const uint64_t units = n_tiles * n_steps * 32;
    const uint64_t low_words = units * LT_LB, high_words = units * LT_HB;
    const uint64_t total = n_tiles * 16 * row_bytes / 4;
    const uint64_t id = ares_group() * 256 + gl_LocalInvocationID.x;
    if (id >= total) {
        return;
    }
    uint word_out = 0u;
    if (id < low_words + high_words) {
        const bool high = id >= low_words;
        const uint bits = high ? uint(LT_HB) : uint(LT_LB);
        const uint64_t u = high ? id - low_words : id;
        const uint word = uint(u % bits);
        const uint lane = uint((u / bits) % 32), g = lane / 4, t = lane % 4;
        const uint64_t unit = u / bits / 32;
        const uint step = uint(unit % n_steps), tile = uint(unit / n_steps);
        for (uint kk = 0; kk < 4; kk++) {
            for (uint i = 0; i < 4; i++) {
                const uint row = tile * 16 + g + 8 * (i & 1u);
                if (lt_plane_word(bits, kk, i) != word || row >= pc.n_out) {
                    continue;
                }
                for (uint e = 0; e < 2; e++) {
                    const uint k = step * 64 + kk * 16 + 2 * t + 8 * (i >> 1) + e;
                    const uint code = code_of(blk(row, k / LT_BLOCK_VALUES), k % LT_BLOCK_VALUES);
                    const uint field = high ? code >> LT_LB : code & ((1u << LT_LB) - 1u);
                    word_out |= field << (lt_plane_shift(bits, kk, i) + 16 * e);
                }
            }
        }
    } else {
        word_out = header_word(id - low_words - high_words, n_tiles);
    }
    Out4(pc.dst + 4 * id).v[0] = word_out;
}
