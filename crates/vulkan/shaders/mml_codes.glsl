// What a lane holds of a matrix in a device layout (mml_types.glsl) and how it turns that
// into the elements of its part of the A operand, in registers: the counterpart of
// `AresMmrCodes` in CUDA's `kernels/mmr_codes.cuh`, for the kernels `mml_near.glsl` and
// `mml_wide.glsl`.
//
// The kernel includes mmvq_common.glsl and mml_types.glsl first and defines
//   LT_STREAM   1: the 16-byte loads of the planes pass L1 by (the kernels that read every
//               weight once); 0: they are cached
//   MAT_A       the cooperative matrix type of the weights (f16, 16 x 16, use A)
//
// A lane (g = lane / 4, t = lane % 4) of a subgroup works on tiles of 16 weight rows. For
// each tile it keeps a `LtTile`: where its words of the planes and of the headers are, the
// words of the step of 64 values in work and of the next one (asked for a step ahead), and
// the scale S and the constant C of the group of values in work for its two rows g and
// g + 8. The calls, in the order of the loop over K:
//
//   lt_init(tid, n_threads)    once per workgroup and a barrier behind it (LT_TABLE only)
//   lt_open(tl, tile, sb)      once; also asks for the codes and headers of step 4 sb
//   lt_super_block(tl, sb)     before step 4 sb
//   lt_step(tl, step, j)       before the products of a step (j = step % 4)
//   lt_fragment(m, tl, j, kk)  the A operand of slab kk (16 values) of the step
//
// The elements of the operand are written in the order `coop_layout_check.comp` verifies:
// the pair of register a_i of an `mma` fragment (row g + 8 (i & 1), values 2t + 8 (i >> 1)
// and the next) is elements 4 (i & 1) + 2 (i >> 1) and the next.

// How many steps ahead a lane asks for its codes: a step then waits for memory only a
// DEPTH-th of the time, which is what bounds a matrix with few rows (few lanes at work).
#ifndef LT_DEPTH
#define LT_DEPTH 1
#endif
#define LT_HN (LT_HB > 0 ? LT_HB : 1)
#define LT_DW (1 + LT_MIN_NIBBLE)
#define LT_MW (1 + LT_M)

#if LT_STREAM
#define lt_ld16 ldw16
#else
#define lt_ld16 ld16
#endif

struct LtTile {
    uint64_t low;   // the lane's words of step 0 of the tile in the low plane
    uint64_t high;  // and in the high plane
    uint64_t hdr;   // its words of the first header section, super-block or step 0
    uint64_t hdr2;  // K16: the pair of d of rows g and g + 8, super-block 0
    uint lw[LT_LB];           // the low plane: the step in work
    uint lx[LT_DEPTH][LT_LB]; // and the next steps
    uint hw[LT_HN];           // the high plane
    uint hx[LT_DEPTH][LT_HN];
#if LT_HDR == LT_HDR_K4
    uvec4 nx[2];    // the headers of rows g and g + 8, next super-block
    uint sc[2][2];  // the scales of the 8 groups of 32 of the super-block in work, as bytes
    uint mn[2][2];  // and the minimums
    f16vec2 dd[2];  // (d, -dmin)
#elif LT_HDR == LT_HDR_K16
    uint nx[2][LT_SCALE_WORDS];
    uint nd[LT_DW];
    uint sc[2][4];  // the 16 scale bytes of the super-block of rows g and g + 8
    f16vec2 d;      // d of row g and of row g + 8
    f16vec2 ndmin;  // -dmin likewise
#elif LT_HDR == LT_HDR_B
    uint nx[LT_DEPTH][2 * LT_MW];
    uint cur[2 * LT_MW];
#else
    uvec2 nx[2];
    uint sl[2];     // scales_l and scales_h of rows g and g + 8
    uint sh[2];
    f16vec2 d;      // d of row g and of row g + 8
#endif
    f16vec2 S[2];   // of the group in work, rows g and g + 8
    f16vec2 C[2];
};

// The matrix `w` has `n_tiles` tiles of 16 rows; `tile` is clamped by the caller.
uint lt_n_steps, lt_n_sb, lt_lane;

// Asks for the lane's words of both planes of `step`, into place `d` of the steps ahead.
void lt_load(inout LtTile tl, uint step, uint d) {
#if LT_LB == 4
    const uvec4 x = lt_ld16(tl.low + uint64_t(step) * 512);
    tl.lx[d][0] = x.x;
    tl.lx[d][1] = x.y;
    tl.lx[d][2] = x.z;
    tl.lx[d][3] = x.w;
#elif LT_LB == 2
    const uvec2 x = ld8(tl.low + uint64_t(step) * 256);
    tl.lx[d][0] = x.x;
    tl.lx[d][1] = x.y;
#else
    const uvec4 x = lt_ld16(tl.low + uint64_t(step) * 1024);
    const uvec4 z = lt_ld16(tl.low + uint64_t(step) * 1024 + 16);
    tl.lx[d][0] = x.x;
    tl.lx[d][1] = x.y;
    tl.lx[d][2] = x.z;
    tl.lx[d][3] = x.w;
    tl.lx[d][4] = z.x;
    tl.lx[d][5] = z.y;
    tl.lx[d][6] = z.z;
    tl.lx[d][7] = z.w;
#endif
#if LT_HB == 1
    tl.hx[d][0] = ld4(tl.high + uint64_t(step) * 128);
#elif LT_HB == 2
    const uvec2 h = ld8(tl.high + uint64_t(step) * 256);
    tl.hx[d][0] = h.x;
    tl.hx[d][1] = h.y;
#endif
}

// Asks for the headers of super-block `at` (K4, K16, X4) or of step `at` (B, into place `d`
// of the steps ahead).
void lt_load_header(inout LtTile tl, uint at, uint d) {
#if LT_HDR == LT_HDR_K4
    tl.nx[0] = ld16(tl.hdr + uint64_t(at) * 256);
    tl.nx[1] = ld16(tl.hdr + uint64_t(at) * 256 + 128);
#elif LT_HDR == LT_HDR_K16
    [[unroll]] for (uint h = 0; h < 2; h++) {
        const uint64_t p = tl.hdr + (uint64_t(at) * 16 + 8 * h) * (4 * LT_SCALE_WORDS);
#if LT_SCALE_WORDS == 4
        const uvec4 x = ld16(p);
        tl.nx[h][0] = x.x;
        tl.nx[h][1] = x.y;
        tl.nx[h][2] = x.z;
        tl.nx[h][3] = x.w;
#else
        [[unroll]] for (uint v = 0; v < LT_SCALE_WORDS; v++) {
            tl.nx[h][v] = ld4(p + 4 * v);
        }
#endif
    }
    [[unroll]] for (uint v = 0; v < LT_DW; v++) {
        tl.nd[v] = ld4(tl.hdr2 + uint64_t(at) * (32 * LT_DW) + 4 * v);
    }
#elif LT_HDR == LT_HDR_B
#if LT_M
    const uvec4 x = ld16(tl.hdr + uint64_t(at) * 128);
    tl.nx[d][0] = x.x;
    tl.nx[d][1] = x.y;
    tl.nx[d][2] = x.z;
    tl.nx[d][3] = x.w;
#else
    const uvec2 x = ld8(tl.hdr + uint64_t(at) * 64);
    tl.nx[d][0] = x.x;
    tl.nx[d][1] = x.y;
#endif
#else
    tl.nx[0] = ld8(tl.hdr + uint64_t(at) * 128);
    tl.nx[1] = ld8(tl.hdr + uint64_t(at) * 128 + 64);
#endif
}

// Opens tile `tile` of the matrix at `w` (`n_tiles` tiles of 16 rows, rows of `n_in`
// values) for lane `lane`, to start at super-block `sb`.
void lt_open(out LtTile tl, uint64_t w, uint64_t n_tiles, uint tile, uint n_in, uint lane,
             uint sb) {
    lt_n_steps = n_in >> 6;
    lt_n_sb = n_in >> 8;
    lt_lane = lane;
    const uint g = lane >> 2;
    const uint64_t units = n_tiles * lt_n_steps * 32;
    const uint64_t at = uint64_t(tile) * lt_n_steps * 32 + lane;
    tl.low = w + at * (4 * LT_LB);
    tl.high = w + units * (4 * LT_LB) + at * (4 * LT_HB);
    const uint64_t hdr = w + units * (4 * (LT_LB + LT_HB));
    tl.hdr2 = hdr;
#if LT_HDR == LT_HDR_K4
    tl.hdr = hdr + (uint64_t(tile) * lt_n_sb * 16 + g) * 16;
#elif LT_HDR == LT_HDR_K16
    tl.hdr = hdr + (uint64_t(tile) * lt_n_sb * 16 + g) * (4 * LT_SCALE_WORDS);
    tl.hdr2 = hdr + n_tiles * lt_n_sb * (16 * 4 * LT_SCALE_WORDS) +
              (uint64_t(tile) * lt_n_sb * 8 + g) * (4 * LT_DW);
#elif LT_HDR == LT_HDR_B
    tl.hdr = hdr + (uint64_t(tile) * lt_n_steps * 8 + g) * (8 * LT_MW);
#else
    tl.hdr = hdr + (uint64_t(tile) * lt_n_sb * 16 + g) * 8;
#endif
    [[unroll]] for (uint d = 0; d < LT_DEPTH; d++) {
        if (4 * sb + d < lt_n_steps) {
            lt_load(tl, 4 * sb + d, d);
#if LT_HDR == LT_HDR_B
            lt_load_header(tl, 4 * sb + d, d);
#endif
        }
    }
#if LT_HDR != LT_HDR_B
    lt_load_header(tl, sb, 0);
#endif
}

// Before step 4 sb: takes the headers of the super-block and asks for the next one's.
void lt_super_block(inout LtTile tl, uint sb) {
#if LT_HDR == LT_HDR_K4
    // The 12 bytes of 6-bit scales and minimums (ggml's get_scale_min_k4): groups 0..3 keep
    // 6 bits in bytes 0..3 (scales) and 4..7 (minimums); groups 4..7 keep 4 bits in bytes
    // 8..11 and the other 2 in the top bits of bytes 0..7.
    [[unroll]] for (uint h = 0; h < 2; h++) {
        const uvec4 x = tl.nx[h];
        tl.sc[h][0] = x.y & 0x3F3F3F3Fu;
        tl.mn[h][0] = x.z & 0x3F3F3F3Fu;
        tl.sc[h][1] = (x.w & 0x0F0F0F0Fu) | ((x.y >> 2) & 0x30303030u);
        tl.mn[h][1] = ((x.w >> 4) & 0x0F0F0F0Fu) | ((x.z >> 2) & 0x30303030u);
        tl.dd[h] = unpackFloat2x16(x.x ^ 0x80000000u);
    }
#elif LT_HDR == LT_HDR_K16
    [[unroll]] for (uint h = 0; h < 2; h++) {
#if LT_SCALE_WORDS == 3
        // Q3_K's 12 bytes: the low 4 bits of scales 0..7 and 8..15 in the nibbles of words 0
        // and 1, the high 2 bits of all 16 in word 2.
        const uint a0 = tl.nx[h][0], a1 = tl.nx[h][1], t2 = tl.nx[h][2];
        tl.sc[h][0] = (a0 & 0x0F0F0F0Fu) | ((t2 & 0x03030303u) << 4);
        tl.sc[h][1] = (a1 & 0x0F0F0F0Fu) | (((t2 >> 2) & 0x03030303u) << 4);
        tl.sc[h][2] = ((a0 >> 4) & 0x0F0F0F0Fu) | (((t2 >> 4) & 0x03030303u) << 4);
        tl.sc[h][3] = ((a1 >> 4) & 0x0F0F0F0Fu) | (((t2 >> 6) & 0x03030303u) << 4);
#else
        tl.sc[h] = tl.nx[h];
#endif
    }
    tl.d = unpackFloat2x16(tl.nd[0]);
#if LT_MIN_NIBBLE
    tl.ndmin = unpackFloat2x16(tl.nd[1] ^ 0x80008000u);
#endif
#elif LT_HDR == LT_HDR_X4
    [[unroll]] for (uint h = 0; h < 2; h++) {
        tl.sh[h] = tl.nx[h].x >> 16;
        tl.sl[h] = tl.nx[h].y;
    }
    tl.d = unpackFloat2x16((tl.nx[0].x & 0xFFFFu) | (tl.nx[1].x << 16));
#endif
#if LT_HDR != LT_HDR_B
    if (sb + 1 < lt_n_sb) {
        lt_load_header(tl, sb + 1, 0);
    }
#endif
}

// Before the products of `step` (j = step % 4): takes the codes asked for LT_DEPTH steps
// ago and asks for those LT_DEPTH steps on.
void lt_step(inout LtTile tl, uint step, uint j) {
    tl.lw = tl.lx[0];
    tl.hw = tl.hx[0];
#if LT_HDR == LT_HDR_B
    tl.cur = tl.nx[0];
#endif
    [[unroll]] for (uint d = 0; d + 1 < LT_DEPTH; d++) {
        tl.lx[d] = tl.lx[d + 1];
        tl.hx[d] = tl.hx[d + 1];
#if LT_HDR == LT_HDR_B
        tl.nx[d] = tl.nx[d + 1];
#endif
    }
    if (step + LT_DEPTH < lt_n_steps) {
        lt_load(tl, step + LT_DEPTH, LT_DEPTH - 1);
#if LT_HDR == LT_HDR_B
        lt_load_header(tl, step + LT_DEPTH, LT_DEPTH - 1);
#endif
    }
}

// The f16 pair of two small unsigned integers (below 1024): they go into the mantissa of
// 1024 and the bias comes off, both exactly.
f16vec2 lt_pair(uint lo, uint hi) {
    return unpackFloat2x16(lo | (hi << 16) | 0x64006400u) - f16vec2(1024.0hf);
}

// S and C of the group that slab kk of step j starts. They are computed in f16 pairs, the
// way CUDA's kernel does (one packed multiplication makes two of them): a scale is a small
// integer, exact in f16, so its product with d is rounded once.
void lt_scales(inout LtTile tl, uint j, uint kk) {
#if LT_HDR == LT_HDR_K4
    // Group 2j + kk / 2 of the super-block: byte 2 (j & 1) + kk / 2 of word j / 2.
    const uint sh = 8u * (2u * (j & 1u) + (kk >> 1));
    [[unroll]] for (uint h = 0; h < 2; h++) {
        const f16vec2 v =
            lt_pair((tl.sc[h][j >> 1] >> sh) & 0xFFu, (tl.mn[h][j >> 1] >> sh) & 0xFFu) * tl.dd[h];
        tl.S[h] = f16vec2(v.x);
        tl.C[h] = f16vec2(v.y);
    }
#elif LT_HDR == LT_HDR_K16
    // Byte kk of word j of both rows: the scale of the group.
    const uint b0 = (tl.sc[0][j] >> (8u * kk)) & 0xFFu;
    const uint b1 = (tl.sc[1][j] >> (8u * kk)) & 0xFFu;
#if LT_MIN_NIBBLE
    const f16vec2 v = lt_pair(b0 & 15u, b1 & 15u) * tl.d;
    const f16vec2 c = lt_pair(b0 >> 4, b1 >> 4) * tl.ndmin;
#else
    const f16vec2 v = (lt_pair(b0, b1) - f16vec2(float16_t(LT_SCALE_BIAS))) * tl.d;
    const f16vec2 c = v * float16_t(-LT_CODE_OFFSET);
#endif
    tl.S[0] = f16vec2(v.x);
    tl.S[1] = f16vec2(v.y);
    tl.C[0] = f16vec2(c.x);
    tl.C[1] = f16vec2(c.y);
#elif LT_HDR == LT_HDR_B
    const f16vec2 d2 = unpackFloat2x16(tl.cur[(kk >> 1) * LT_MW]);
    tl.S[0] = f16vec2(d2.x);
    tl.S[1] = f16vec2(d2.y);
#if LT_M
    const f16vec2 m2 = unpackFloat2x16(tl.cur[(kk >> 1) * LT_MW + 1]);
    tl.C[0] = f16vec2(m2.x);
    tl.C[1] = f16vec2(m2.y);
#else
    tl.C[0] = tl.S[0] * float16_t(-LT_CODE_OFFSET);
    tl.C[1] = tl.S[1] * float16_t(-LT_CODE_OFFSET);
#endif
#else
    // The 6-bit scale of group 2j + kk / 2: 4 bits of scales_l, 2 of scales_h.
    const uint ib = 2u * j + (kk >> 1);
    const uint ls0 = ((tl.sl[0] >> (4u * ib)) & 15u) | (((tl.sh[0] >> (2u * ib)) & 3u) << 4);
    const uint ls1 = ((tl.sl[1] >> (4u * ib)) & 15u) | (((tl.sh[1] >> (2u * ib)) & 3u) << 4);
    const f16vec2 v = (lt_pair(ls0, ls1) - f16vec2(32.0hf)) * tl.d;
    tl.S[0] = f16vec2(v.x);
    tl.S[1] = f16vec2(v.y);
#endif
}

#if LT_TABLE
// The level of IQ4_NL of a 4-bit index, as an integer: the 16 levels + 128 are the bytes
// of four words (the constants of CUDA's kernel, which looks four up at once with `prmt`).
int lt_level(uint c) {
    const uint lo = (c & 4u) != 0u ? 0x766A5D4Fu : 0x3F2D1801u;
    const uint hi = (c & 4u) != 0u ? 0xF1D9C5B5u : 0xA6998D81u;
    const uint w = (c & 8u) != 0u ? hi : lo;
    return int((w >> (8u * (c & 3u))) & 0xFFu) - 128;
}
// GLSL has no byte permutation, and selecting and shifting as above costs about ten
// instructions per value. So the workgroup keeps a table in shared memory of the 256 pairs
// of levels as f16 pairs, indexed by the two 4-bit indices of a register's pair of values:
// one load makes the pair. `lt_init` fills it (invocation `tid` of `n_threads`); the kernel
// puts a barrier behind it.
shared uint lt_pairs[256];
void lt_init(uint tid, uint n_threads) {
    for (uint i = tid; i < 256; i += n_threads) {
        lt_pairs[i] = packFloat2x16(f16vec2(ivec2(lt_level(i & 15u), lt_level(i >> 4))));
    }
}
#endif

// The lane's 8 values of slab kk of the step of tile `tl`, as elements of `m`.
void lt_fragment(inout MAT_A m, inout LtTile tl, uint j, uint kk) {
    if ((kk & uint(LT_HDR == LT_HDR_K16 ? 0 : 1)) == 0u) {
        lt_scales(tl, j, kk);
    }
    [[unroll]] for (uint i = 0; i < 4; i++) {
#if LT_TABLE
        const uint w = tl.lw[kk];
        const uint pair = ((w >> (4u * i)) & 0xFu) | ((w >> (4u * i + 12u)) & 0xF0u);
        const f16vec2 v = unpackFloat2x16(lt_pairs[pair]) * tl.S[i & 1u];
#else
        const uint low_mask = ((1u << LT_LB) - 1u) * 0x00010001u;
        uint pair =
            (tl.lw[lt_plane_word(LT_LB, kk, i)] >> lt_plane_shift(LT_LB, kk, i)) & low_mask;
#if LT_HB > 0
        const uint high_mask = ((1u << LT_HB) - 1u) * 0x00010001u;
        pair |= ((tl.hw[lt_plane_word(LT_HB, kk, i)] >> lt_plane_shift(LT_HB, kk, i)) & high_mask)
                << LT_LB;
#endif
        const f16vec2 code = unpackFloat2x16(pair | 0x64006400u) - f16vec2(1024.0hf);
        const f16vec2 v = fma(code, tl.S[i & 1u], tl.C[i & 1u]);
#endif
        m[4u * (i & 1u) + 2u * (i >> 1)] = v.x;
        m[4u * (i & 1u) + 2u * (i >> 1) + 1u] = v.y;
    }
}
