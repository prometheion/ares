// The weights piece of the f16 prefill kernel for Q4_K in the rearranged form
// (`repack_q4_K.comp`, CUDA's `mmr.cuh`): each subgroup decodes one tile of 16 weight rows
// per step. A lane reads one 16-byte word of codes of the form (the subgroup's 32 lanes
// read 512 contiguous bytes), turns every nibble pair into an f16 pair with one `or`, one
// subtract of 1024 and one fused multiply-add with the sub-block's scale and minimum, and
// writes the pair where the A matrix is loaded from. The layout of a lane's words is that
// of an `mma` m16n8k16 A fragment, which is only used here, for knowing which element is
// which.

#include "mmd_common.glsl"

uint64_t code_base;  // this lane's codes of step 0
uint64_t hdr0;       // header of the lane's row g of the tile, super-block 0
uint64_t hdr1;       // and of row g + 8
uvec4 craw;          // the codes of the step to come

void weights_setup() {
    const uint g = lane >> 2;
    const uint n_sb = pc.n_in >> 8;
    const uint n_tiles16 = (pc.n_out + 15) / 16;
    const uint64_t codes_bytes = uint64_t(n_tiles16) * 16 * pc.n_in / 2;
    const uint mt16 = min(wt * 4 + sg, n_tiles16 - 1);
    code_base = pc.w + uint64_t(mt16) * n_steps * 512 + uint64_t(lane) * 16;
    // Header of row (16 mt16 + g + 8 h) of super-block sb.
    hdr0 = pc.w + codes_bytes + (uint64_t(mt16) * n_sb * 16 + g) * 16;
    hdr1 = hdr0 + 8 * 16;
}

void weights_load(uint s) {
    craw = ld16(code_base + uint64_t(s) * 512);
}

// Decodes the codes of one word `cw` (16 values of the lane's rows g and g + 8) with the
// sub-block's factors (scale s, minimum z of the rows g and g + 8) and writes the f16 pairs.
void decode_word(uint stage_i, uint row_base, uint k0, uint cw, f16vec2 s0, f16vec2 z0,
                 f16vec2 s1, f16vec2 z1) {
    [[unroll]] for (uint i = 0; i < 4; i++) {
        const f16vec2 nib =
            unpackFloat2x16(((cw >> (4 * i)) & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0hf);
        const f16vec2 v = (i & 1u) == 0u ? fma(nib, s0, -z0) : fma(nib, s1, -z1);
        const uint row = row_base + (i & 1u) * 8;
        const uint at = (row * ROW16 * 8 + k0 + 8 * (i >> 1)) >> 1;  // in 32-bit words
        sa[stage_i][at] = packFloat2x16(v);
    }
}

void weights_store(uint s, uint st) {
    const uint g = lane >> 2;
    const uint t = lane & 3u;
    const uint sb_i = s >> 2;
    const uint j = s & 3u;
    const uvec4 h0 = ld16(hdr0 + uint64_t(sb_i) * 256);
    const uvec4 h1 = ld16(hdr1 + uint64_t(sb_i) * 256);
    uint sc0, mn0, sc1, mn1;
    scales_k(j, h0.yzw, sc0, mn0);
    scales_k(j, h1.yzw, sc1, mn1);
    const vec2 d0 = unpackHalf2x16(h0.x);
    const vec2 d1 = unpackHalf2x16(h1.x);
    const uint row_base = sg * 16 + g;
    // Words 0 and 1 hold values 0..31 of the step (sub-block 2j), words 2 and 3 values
    // 32..63 (sub-block 2j + 1).
    [[unroll]] for (uint half_ = 0; half_ < 2; half_++) {
        const uint sh = 8 * half_;
        const f16vec2 s0 = f16vec2(float16_t(d0.x * float((sc0 >> sh) & 0xFFu)));
        const f16vec2 z0 = f16vec2(float16_t(d0.y * float((mn0 >> sh) & 0xFFu)));
        const f16vec2 s1 = f16vec2(float16_t(d1.x * float((sc1 >> sh) & 0xFFu)));
        const f16vec2 z1 = f16vec2(float16_t(d1.y * float((mn1 >> sh) & 0xFFu)));
        decode_word(st, row_base, (2 * half_) * 16 + 2 * t, craw[2 * half_], s0, z0, s1, z1);
        decode_word(st, row_base, (2 * half_ + 1) * 16 + 2 * t, craw[2 * half_ + 1], s0, z0, s1,
                    z1);
    }
}
