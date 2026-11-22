// Where the K and V rows of a sequence lie on the device (see src/attention.rs
// for the full description).
//
// A sequence is a table of u32 entries, one per 128 cells (one tile). Bit 31
// says which pool the entry is in, the low 31 bits are the slot in it:
//
//   exact pool      slot = H heads of [K: 128 rows of D halves][V: 128 rows of D halves]
//   converted pool  slot = H heads of [K mean][K rows][V mean][V rows]
//
// where a row is `format.row_bytes(D)` bytes and a mean (only for q7 to q4) is
// D halves. The rows are the bytes `KvHead` stores. A pool is one array of
// slots per attention layer and device; the table serves every layer.
//
// A converted entry with bit 30 set lies in the far pool: the same slots in
// page-locked host memory, which the device reads and writes over its link (the
// K/V of a context beyond VRAM). Bits 0-29 are then the slot in that pool.
#pragma once

#include "attn_codec.cuh"

#define ATTN_CONVERTED_BIT 0x80000000u
#define ATTN_FAR_BIT 0x40000000u

// Where converted slot `slot` (bit 31 cleared) starts: in the pool in VRAM, or in
// the far pool when bit 30 is set.
template <class T>
static __device__ __forceinline__ T* conv_entry(T* near, T* far, uint32_t slot, size_t entry_c) {
    return (slot & ATTN_FAR_BIT) ? far + (size_t)(slot & ~ATTN_FAR_BIT) * entry_c
                                 : near + (size_t)slot * entry_c;
}

// One row of a step: the sequence's table, the position of the row's own cell,
// and where its partial states go.
struct AttnRow {
    uint32_t table;      // index of the first entry of the sequence's table
    uint32_t pos;        // position of the cell the row writes (it sees cells <= pos when causal)
    uint32_t slot_base;  // first state slot of the row (merge reads slot_base .. slot_base + n_slots)
    uint32_t n_slots;    // pieces the row has (0: the row is finished by its only item)
};

// The geometry of the pools for one launch, decoded from the 64-bit config word:
//   bits 0-3 K format, 4-7 V format, 8-15 H (kv heads in the pools), 16-23
//   kv_total (kv heads of the model), 24-31 head0 (first kv head of this device),
//   32-39 G (query heads per kv head), 40-51 D (head size).
struct AttnGeo {
    int kf, vf, H, kv_total, head0, G, D;
    int rbk, rbv, kmean, vmean;
    int k_rows_off, v_mean_off, v_rows_off;
    int head_c;
    size_t head_e, entry_e, entry_c;
};

static __device__ __forceinline__ AttnGeo attn_geo(uint64_t cfg) {
    AttnGeo g;
    g.kf = (int)(cfg & 15);
    g.vf = (int)((cfg >> 4) & 15);
    g.H = (int)((cfg >> 8) & 255);
    g.kv_total = (int)((cfg >> 16) & 255);
    g.head0 = (int)((cfg >> 24) & 255);
    g.G = (int)((cfg >> 32) & 255);
    g.D = (int)((cfg >> 40) & 4095);
    const int groups = g.D / 32;
    g.rbk = groups * fmt_group_bytes(g.kf);
    g.rbv = groups * fmt_group_bytes(g.vf);
    g.kmean = fmt_has_mean(g.kf) ? 2 * g.D : 0;
    g.vmean = fmt_has_mean(g.vf) ? 2 * g.D : 0;
    g.k_rows_off = g.kmean;
    g.v_mean_off = g.kmean + ATTN_TILE_CELLS * g.rbk;
    g.v_rows_off = g.v_mean_off + g.vmean;
    g.head_c = g.v_rows_off + ATTN_TILE_CELLS * g.rbv;
    g.head_e = (size_t)ATTN_TILE_CELLS * 2 * g.D * 2;
    g.entry_e = g.head_e * g.H;
    g.entry_c = (size_t)g.head_c * g.H;
    return g;
}
