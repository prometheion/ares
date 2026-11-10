// The i-quants and Q8_0 for the integer weight kernels (mmvq.cuh).
//
// IQ4_XS and IQ4_NL map each nibble through a table of 16 signed bytes; the
// other i-quants look 4 or 8 magnitudes up in a codebook (iq_tables.cuh) by
// 8 to 10 bits of the block and give them signs, so both end as signed bytes
// that dp4a multiplies with the int8 activations. Blocks are 2-byte aligned
// (4-byte for none but IQ4_XS), hence the narrow loads.
//
// All scales of these types are small integers (or integers plus a half) times
// the f16 d of the block, so the integer dot products are scaled in integers
// and the float multiplication happens once per sub-block:
//   dot = d * dx * (integer scale) * sum(w * qx) * constant
// exactly, with no rounding before the final products.
#pragma once

#include "mmvq.cuh"
#include "iq_tables.cuh"

#define ARES_PACK4(a, b, c, d)                                                                  \
    (static_cast<uint32_t>(static_cast<uint8_t>(a)) | (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) | \
     (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) | (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24))

// The bytes of four 4-bit indices into the table of IQ4_NL / IQ4_XS: x holds
// the table entries of the low nibbles of the 4 bytes of q4, y those of the
// high nibbles. PRMT selects bytes of the two registers of an 8-byte table
// half by the low 3 bits of a nibble; the fourth bit picks the half.
static __device__ __forceinline__ int2 ares_table16(uint32_t q4) {
    // The 16 levels of IQ4_NL (KVALUES_IQ4NL in crates/quant/src/tables.rs), four
    // bytes to a register, entry 0 in the least significant byte.
    const uint32_t t0 = ARES_PACK4(-127, -104, -83, -65);
    const uint32_t t1 = ARES_PACK4(-49, -35, -22, -10);
    const uint32_t t2 = ARES_PACK4(1, 13, 25, 38);
    const uint32_t t3 = ARES_PACK4(53, 69, 89, 113);
    uint32_t tmp[2];
    const uint32_t pick = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const uint32_t sh = 16 * i;
        const uint32_t low = __byte_perm(t0, t1, q4 >> sh);
        const uint32_t high = __byte_perm(t2, t3, q4 >> sh);
        tmp[i] = __byte_perm(low, high, pick >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// Sign masks (0xFF in the bytes to negate) for four values from four bits of
// `sign_bits` (bit i for byte i), and the application to four magnitudes.
static __device__ __forceinline__ int ares_sign_mask4(uint32_t spread, uint32_t select) {
    return __vcmpne4(spread & select, 0);
}

static __device__ __forceinline__ int ares_apply_sign4(uint32_t magnitudes, int mask) {
    return __vsub4(static_cast<int>(magnitudes) ^ mask, mask);
}

// The 8 sign bits of a 7-bit index: bit 7 makes the number of negative values
// even (ggml's ksigns_iq2xs, without the table).
static __device__ __forceinline__ uint32_t ares_ksigns(uint32_t v7) {
    return v7 ^ ((__popc(v7) & 1u) << 7);
}

// ---------------------------------------------------------------------------
// IQ4_XS: 256 values in 136 bytes (8-byte aligned).
//   0  f16 d, u16 scales_h
//   4  u8 scales_l[4]
//   8  u8 qs[128]
// Sub-block ib (32 values) has the scale (scales_l nibble | 2 bits of scales_h)
// - 32 and the 16 bytes qs[16 ib ..]: low nibbles are its values 0..15, high
// nibbles its values 16..31. value = d * scale * table[nibble]. A lane takes
// one sub-block (slot = ib).
struct EngIQ4XS {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 136;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        int x[4];  // table values of the low nibbles: values 0..15 of the sub-block
        int y[4];  // of the high nibbles: values 16..31
        int ls;    // scale - 32
        float d;
    };
    struct A {
        uint4 first;   // activations 0..15
        uint4 second;  // 16..31
        float2 ds;
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t*) {
        const uint2 hdr = ares_ld_stream_v2(blk);
        const uint2 q01 = ares_ld_stream_v2(blk + 8 + 16 * ib);
        const uint2 q23 = ares_ld_stream_v2(blk + 16 + 16 * ib);
        const uint32_t qw[4] = {q01.x, q01.y, q23.x, q23.y};
        W w;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int2 v = ares_table16(qw[i]);
            w.x[i] = v.x;
            w.y[i] = v.y;
        }
        w.ls = static_cast<int>(((hdr.y >> (4 * (ib & 1))) >> (8 * (ib >> 1)) & 0x0F) |
                                (((hdr.x >> (16 + 2 * ib)) & 3) << 4)) - 32;
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x)));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        A a;
        a.first = ares_ld_act_v4(q + 32 * ib);
        a.second = ares_ld_act_v4(q + 32 * ib + 16);
        a.ds = ds[ib];
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(w.x[0], static_cast<int>(a.first.x), 0);
        s = ares_dp4a(w.x[1], static_cast<int>(a.first.y), s);
        s = ares_dp4a(w.x[2], static_cast<int>(a.first.z), s);
        s = ares_dp4a(w.x[3], static_cast<int>(a.first.w), s);
        s = ares_dp4a(w.y[0], static_cast<int>(a.second.x), s);
        s = ares_dp4a(w.y[1], static_cast<int>(a.second.y), s);
        s = ares_dp4a(w.y[2], static_cast<int>(a.second.z), s);
        s = ares_dp4a(w.y[3], static_cast<int>(a.second.w), s);
        acc = fmaf(static_cast<float>(s * w.ls), w.d * a.ds.x, acc);
    }
};

// ---------------------------------------------------------------------------
// IQ4_NL: 32 values in 18 bytes (2-byte aligned): f16 d, then 16 bytes of
// nibbles laid out as Q4_0 (low nibbles are values 0..15, high 16..31), each
// through the same table as IQ4_XS. Two lanes per block, 8 bytes each.
struct EngIQ4NL {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 18;
    static constexpr int LANES = 2;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        int x[2];  // values 8s .. 8s+7
        int y[2];  // values 16+8s .. 16+8s+7
        float d;
    };
    struct A {
        uint2 first;   // activations 8s .. 8s+7
        uint2 second;  // 16+8s .. 16+8s+7
        float d;
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t s, const uint8_t*) {
        const uint2 q = ares_ld_stream_v2_a2(blk + 2 + 8 * s);
        W w;
        const int2 v0 = ares_table16(q.x);
        const int2 v1 = ares_table16(q.y);
        w.x[0] = v0.x;
        w.y[0] = v0.y;
        w.x[1] = v1.x;
        w.y[1] = v1.y;
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t s) {
        A a;
        a.first = *reinterpret_cast<const uint2*>(q + 8 * s);
        a.second = *reinterpret_cast<const uint2*>(q + 16 + 8 * s);
        a.d = ds[0].x;
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(w.x[0], static_cast<int>(a.first.x), 0);
        s = ares_dp4a(w.x[1], static_cast<int>(a.first.y), s);
        s = ares_dp4a(w.y[0], static_cast<int>(a.second.x), s);
        s = ares_dp4a(w.y[1], static_cast<int>(a.second.y), s);
        acc = fmaf(static_cast<float>(s), w.d * a.d, acc);
    }
};

// ---------------------------------------------------------------------------
// Q8_0: 32 values in 34 bytes (2-byte aligned): f16 d, 32 signed bytes. Two
// lanes per block, 16 bytes each.
struct EngQ80 {
    static constexpr int BLOCK_VALUES = 32;
    static constexpr int BLOCK_BYTES = 34;
    static constexpr int LANES = 2;
    static constexpr int SMEM_BYTES = 0;
    static __device__ __forceinline__ void init(uint8_t*, uint32_t, uint32_t) {}

    struct W {
        uint32_t q[4];
        float d;
    };
    struct A {
        uint4 q;
        float d;
    };

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t s, const uint8_t*) {
        W w;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            w.q[i] = ares_ld_stream_u32_a2(blk + 2 + 16 * s + 4 * i);
        }
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t s) {
        A a;
        a.q = ares_ld_act_v4(q + 16 * s);
        a.d = ds[0].x;
        return a;
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(static_cast<int>(w.q[0]), static_cast<int>(a.q.x), 0);
        s = ares_dp4a(static_cast<int>(w.q[1]), static_cast<int>(a.q.y), s);
        s = ares_dp4a(static_cast<int>(w.q[2]), static_cast<int>(a.q.z), s);
        s = ares_dp4a(static_cast<int>(w.q[3]), static_cast<int>(a.q.w), s);
        acc = fmaf(static_cast<float>(s), w.d * a.d, acc);
    }
};

// Copies `bytes` bytes of a device table to shared memory, 16 bytes per thread
// and step. Tables and destination are 16-byte aligned.
static __device__ __forceinline__ void ares_copy_table(uint8_t* dst, const void* src, uint32_t bytes, uint32_t t, uint32_t n) {
    for (uint32_t i = t * 16; i < bytes; i += n * 16) {
        *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(src) + i);
    }
}

// ---------------------------------------------------------------------------
// IQ3_XXS: 256 values in 98 bytes (2-byte aligned).
//   0   f16 d
//   2   u8 grid[64]   indices into a codebook of 256 groups of 4 magnitudes
//   66  u32 aux[8]    per sub-block: four 7-bit sign indices, 4-bit scale on top
// Sub-block ib (32 values = 8 groups of 4): grid[8 ib + 0 .. 7]; group pair l
// (8 values) takes grid[8 ib + 2l], grid[8 ib + 2l + 1] and the sign byte of
// 7 bits (aux >> 7l). value = d * (0.5 + scale) * 0.5 * magnitude * sign. A
// lane takes one sub-block (slot = ib).
struct EngIQ3XXS {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 98;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 256 * 4;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        ares_copy_table(smem, ares_iq3xxs_grid, SMEM_BYTES, t, n);
    }

    struct W {
        int w[8];  // signed values, 4 per int
        int scale2;  // 2 * scale + 1
        float d;
    };
    using A = EngIQ4XS::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t* smem) {
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(smem);
        const uint2 idx = ares_ld_stream_v2_a2(blk + 2 + 8 * ib);
        const uint32_t aux = ares_ld_stream_u32_a2(blk + 66 + 4 * ib);
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t word = l < 2 ? idx.x : idx.y;
            const uint32_t i1 = (word >> (16 * (l & 1))) & 0xFF;
            const uint32_t i2 = (word >> (16 * (l & 1) + 8)) & 0xFF;
            const uint32_t signs = ares_ksigns((aux >> (7 * l)) & 127) * 0x01010101u;
            w.w[2 * l] = ares_apply_sign4(grid[i1], ares_sign_mask4(signs, 0x08040201u));
            w.w[2 * l + 1] = ares_apply_sign4(grid[i2], ares_sign_mask4(signs, 0x80402010u));
        }
        w.scale2 = static_cast<int>(2 * (aux >> 28) + 1);
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        return EngIQ4XS::load_a(q, ds, ib);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(w.w[0], static_cast<int>(a.first.x), 0);
        s = ares_dp4a(w.w[1], static_cast<int>(a.first.y), s);
        s = ares_dp4a(w.w[2], static_cast<int>(a.first.z), s);
        s = ares_dp4a(w.w[3], static_cast<int>(a.first.w), s);
        s = ares_dp4a(w.w[4], static_cast<int>(a.second.x), s);
        s = ares_dp4a(w.w[5], static_cast<int>(a.second.y), s);
        s = ares_dp4a(w.w[6], static_cast<int>(a.second.z), s);
        s = ares_dp4a(w.w[7], static_cast<int>(a.second.w), s);
        // (0.5 + scale) * 0.5 = (2 scale + 1) / 4
        acc = fmaf(static_cast<float>(s * w.scale2), 0.25f * w.d * a.ds.x, acc);
    }
};

// ---------------------------------------------------------------------------
// IQ3_S: 256 values in 110 bytes (2-byte aligned).
//   0    f16 d
//   2    u8 qs[64]      low 8 bits of 9-bit codebook indices (512 groups of 4)
//   66   u8 qh[8]       the ninth bits: byte ib for sub-block ib
//   74   u8 signs[32]   one sign bit per value
//   106  u8 scales[4]   4 bits per sub-block
// Sub-block ib: indices qs[8 ib + 2l + {0,1}] | bit of qh[ib], signs[4 ib + l]
// for l = 0..3; value = d * (1 + 2 scale) * magnitude * sign. Slot = ib.
struct EngIQ3S {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 110;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 512 * 4;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        ares_copy_table(smem, ares_iq3s_grid, SMEM_BYTES, t, n);
    }

    struct W {
        int w[8];
        int scale2;  // 1 + 2 * scale
        float d;
    };
    using A = EngIQ4XS::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t* smem) {
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(smem);
        const uint2 qs = ares_ld_stream_v2_a2(blk + 2 + 8 * ib);
        const uint32_t qh = ares_ld_stream_u8(blk + 66 + ib);
        const uint32_t signs = ares_ld_stream_u32_a2(blk + 74 + 4 * ib);
        const uint32_t sc = ares_ld_stream_u8(blk + 106 + (ib >> 1));
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t word = l < 2 ? qs.x : qs.y;
            const uint32_t i1 = ((word >> (16 * (l & 1))) & 0xFF) | ((qh << (8 - 2 * l)) & 256);
            const uint32_t i2 = ((word >> (16 * (l & 1) + 8)) & 0xFF) | ((qh << (7 - 2 * l)) & 256);
            const uint32_t spread = ((signs >> (8 * l)) & 0xFF) * 0x01010101u;
            w.w[2 * l] = ares_apply_sign4(grid[i1], ares_sign_mask4(spread, 0x08040201u));
            w.w[2 * l + 1] = ares_apply_sign4(grid[i2], ares_sign_mask4(spread, 0x80402010u));
        }
        w.scale2 = static_cast<int>(1 + 2 * ((sc >> (4 * (ib & 1))) & 0xF));
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        return EngIQ4XS::load_a(q, ds, ib);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s = ares_dp4a(w.w[0], static_cast<int>(a.first.x), 0);
        s = ares_dp4a(w.w[1], static_cast<int>(a.first.y), s);
        s = ares_dp4a(w.w[2], static_cast<int>(a.first.z), s);
        s = ares_dp4a(w.w[3], static_cast<int>(a.first.w), s);
        s = ares_dp4a(w.w[4], static_cast<int>(a.second.x), s);
        s = ares_dp4a(w.w[5], static_cast<int>(a.second.y), s);
        s = ares_dp4a(w.w[6], static_cast<int>(a.second.z), s);
        s = ares_dp4a(w.w[7], static_cast<int>(a.second.w), s);
        acc = fmaf(static_cast<float>(s * w.scale2), w.d * a.ds.x, acc);
    }
};

// ---------------------------------------------------------------------------
// IQ2_XS: 256 values in 74 bytes (2-byte aligned).
//   0   f16 d
//   2   u16 qs[32]    9-bit codebook index (512 groups of 8 magnitudes) and a
//                     7-bit sign index above it
//   66  u8 scales[8]  two 4-bit scales per sub-block: the first 16 values, the
//                     second 16
// Sub-block ib: qs[4 ib + l], l = 0..3, one group of 8 values each.
// value = d * (0.5 + scale) * 0.25 * magnitude * sign. Slot = ib.
struct EngIQ2XS {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 74;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 512 * 8;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        ares_copy_table(smem, ares_iq2xs_grid, SMEM_BYTES, t, n);
    }

    struct W {
        int w[8];
        int scale2[2];  // 2 * scale + 1 of the two halves
        float d;
    };
    using A = EngIQ4XS::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t* smem) {
        const uint2* grid = reinterpret_cast<const uint2*>(smem);
        const uint2 qs = ares_ld_stream_v2_a2(blk + 2 + 8 * ib);
        const uint32_t sc = ares_ld_stream_u8(blk + 66 + ib);
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t word = l < 2 ? qs.x : qs.y;
            const uint32_t q = (word >> (16 * (l & 1))) & 0xFFFF;
            const uint2 g = grid[q & 511];
            const uint32_t spread = ares_ksigns(q >> 9) * 0x01010101u;
            w.w[2 * l] = ares_apply_sign4(g.x, ares_sign_mask4(spread, 0x08040201u));
            w.w[2 * l + 1] = ares_apply_sign4(g.y, ares_sign_mask4(spread, 0x80402010u));
        }
        w.scale2[0] = static_cast<int>(2 * (sc & 0xF) + 1);
        w.scale2[1] = static_cast<int>(2 * (sc >> 4) + 1);
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        return EngIQ4XS::load_a(q, ds, ib);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        int s0 = ares_dp4a(w.w[0], static_cast<int>(a.first.x), 0);
        s0 = ares_dp4a(w.w[1], static_cast<int>(a.first.y), s0);
        s0 = ares_dp4a(w.w[2], static_cast<int>(a.first.z), s0);
        s0 = ares_dp4a(w.w[3], static_cast<int>(a.first.w), s0);
        int s1 = ares_dp4a(w.w[4], static_cast<int>(a.second.x), 0);
        s1 = ares_dp4a(w.w[5], static_cast<int>(a.second.y), s1);
        s1 = ares_dp4a(w.w[6], static_cast<int>(a.second.z), s1);
        s1 = ares_dp4a(w.w[7], static_cast<int>(a.second.w), s1);
        // (0.5 + scale) * 0.25 = (2 scale + 1) / 8
        acc = fmaf(static_cast<float>(s0 * w.scale2[0] + s1 * w.scale2[1]), 0.125f * w.d * a.ds.x, acc);
    }
};

// ---------------------------------------------------------------------------
// IQ2_S: 256 values in 82 bytes (2-byte aligned).
//   0   f16 d
//   2   u8 qs[64]     the first 32: low 8 bits of 10-bit codebook indices
//                     (1024 groups of 8 magnitudes); the last 32: sign bytes
//   66  u8 qh[8]      the top 2 bits of the indices of sub-block ib
//   74  u8 scales[8]  as IQ2_XS
// Sub-block ib: index qs[4 ib + l] | (qh[ib] << (8 - 2l)) & 0x300, signs
// qs[32 + 4 ib + l]. Slot = ib.
struct EngIQ2S {
    static constexpr int BLOCK_VALUES = 256;
    static constexpr int BLOCK_BYTES = 82;
    static constexpr int LANES = 8;
    static constexpr int SMEM_BYTES = 1024 * 8;
    static __device__ __forceinline__ void init(uint8_t* smem, uint32_t t, uint32_t n) {
        ares_copy_table(smem, ares_iq2s_grid, SMEM_BYTES, t, n);
    }

    using W = EngIQ2XS::W;
    using A = EngIQ4XS::A;

    static __device__ __forceinline__ W load_w(const uint8_t* blk, uint32_t ib, const uint8_t* smem) {
        const uint2* grid = reinterpret_cast<const uint2*>(smem);
        const uint32_t qs = ares_ld_stream_u32_a2(blk + 2 + 4 * ib);
        const uint32_t signs = ares_ld_stream_u32_a2(blk + 34 + 4 * ib);
        const uint32_t qh = ares_ld_stream_u8(blk + 66 + ib);
        const uint32_t sc = ares_ld_stream_u8(blk + 74 + ib);
        W w;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t idx = ((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
            const uint2 g = grid[idx];
            const uint32_t spread = ((signs >> (8 * l)) & 0xFF) * 0x01010101u;
            w.w[2 * l] = ares_apply_sign4(g.x, ares_sign_mask4(spread, 0x08040201u));
            w.w[2 * l + 1] = ares_apply_sign4(g.y, ares_sign_mask4(spread, 0x80402010u));
        }
        w.scale2[0] = static_cast<int>(2 * (sc & 0xF) + 1);
        w.scale2[1] = static_cast<int>(2 * (sc >> 4) + 1);
        w.d = __half2float(__ushort_as_half(static_cast<unsigned short>(ares_ld_stream_u16(blk))));
        return w;
    }

    static __device__ __forceinline__ A load_a(const int8_t* q, const float2* ds, uint32_t ib) {
        return EngIQ4XS::load_a(q, ds, ib);
    }

    static __device__ __forceinline__ void dot_acc(float& acc, const W& w, const A& a) {
        EngIQ2XS::dot_acc(acc, w, a);
    }
};
