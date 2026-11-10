// IQ4_XS for the integer weight kernels (mmvq.glsl); `mmvq_iquants.cuh` is the model.
//
// 256 values in 136 bytes (8-byte aligned).
//   0  f16 d, u16 scales_h
//   4  u8 scales_l[4]
//   8  u8 qs[128]
// Sub-block ib (32 values) has the scale (scales_l nibble | 2 bits of scales_h) - 32
// and the 16 bytes qs[16 ib ..]: low nibbles are its values 0..15, high nibbles its
// values 16..31. value = d * scale * table[nibble]. A lane takes one sub-block
// (slot = ib).

#include "mmvq_iq_common.glsl"

const uint BLOCK_VALUES = 256;
const uint BLOCK_BYTES = 136;
const uint LANES = 8;

struct W {
    int x[4];  // table values of the low nibbles: values 0..15 of the sub-block
    int y[4];  // of the high nibbles: values 16..31
    int ls;    // scale - 32
    float d;
};

#define A SubA

void mmvq_init(uint t, uint64_t tables) {
    iq4_init(t);
}

W load_w(uint64_t blk, uint ib) {
    const uvec2 hdr = ldw8(blk);
    const uvec2 q01 = ldw8(blk + 8 + 16 * ib);
    const uvec2 q23 = ldw8(blk + 16 + 16 * ib);
    const uint qw[4] = uint[4](q01.x, q01.y, q23.x, q23.y);
    W w;
    [[unroll]] for (uint i = 0; i < 4; i++) {
        table16(qw[i], w.x[i], w.y[i]);
    }
    w.ls = int((((hdr.y >> (4 * (ib & 1))) >> (8 * (ib >> 1))) & 0x0Fu)
               | (((hdr.x >> (16 + 2 * ib)) & 3u) << 4)) - 32;
    w.d = half_of(hdr.x);
    return w;
}

A load_a(uint64_t q, uint64_t ds, uint ib) {
    return load_sub_a(q, ds, ib);
}

void dot_acc(inout float acc, W w, A a) {
    int s = dp_ss(uint(w.x[0]), a.first.x, 0);
    s = dp_ss(uint(w.x[1]), a.first.y, s);
    s = dp_ss(uint(w.x[2]), a.first.z, s);
    s = dp_ss(uint(w.x[3]), a.first.w, s);
    s = dp_ss(uint(w.y[0]), a.second.x, s);
    s = dp_ss(uint(w.y[1]), a.second.y, s);
    s = dp_ss(uint(w.y[2]), a.second.z, s);
    s = dp_ss(uint(w.y[3]), a.second.w, s);
    acc = fma(float(s * w.ls), w.d * a.ds.x, acc);
}
