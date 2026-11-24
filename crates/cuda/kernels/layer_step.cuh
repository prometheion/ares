// The step record, as the layer kernels (layers.cu) read it.
//
// A step carries rows of any mix of sequences: for each sequence one segment of
// consecutive rows. The conductor writes one record per step into a pinned
// buffer and uploads it with one copy ("A step" in docs/dev/architecture/overview.md);
// the kernels below get a pointer to it and read what they need. Nothing about
// a step's shape is compiled in or passed per launch except the sizes that
// decide the grid.
//
// Layout, all little endian, every array 16-byte aligned:
//   AresStep   header (32 bytes) with the counts and the byte offsets of the arrays
//   AresSeg    [n_segs]            the segments, 32 bytes each
//   uint32_t   [n_rows]            token id of each row
//   int32_t    [n_rows][4]         position of each row: time, height, width, extra
//                                  (a text token has (p, p, p, 0))
//   uint32_t   [n_out]             the rows whose logits are wanted, in output order
//
// The rows of a segment are consecutive: segment s owns rows
// [row0, row0 + n_rows). Segments do not overlap; they need not cover all rows
// in order (a launch only reads the segments it is given).
#pragma once

#include "common.cuh"

// Segment flag: all rows of the segment will be accepted (a prompt slice, or a
// generation row without drafts). The delta-rule window kernel then writes the
// recurrent state itself and the deferred commit skips the segment.
#define ARES_SEG_DIRECT 1u
// Segment flag: the rows are a replay (always direct too): their cells are
// cached already and they write none. The runtime's record sets it (its flag 2
// is the model's own drafts, which no kernel here reads).
#define ARES_SEG_REPLAY 4u

struct AresSeg {
    uint32_t row0;      // first row of the segment among the rows of the step
    uint32_t n_rows;    // rows of the segment (the length of its window)
    uint32_t slot;      // the sequence's slot in the recurrent state arrays
    uint32_t accepted;  // rows to commit, <= n_rows; written after verification
    int32_t pos0;       // position of the first row (informational for the kernels: rows carry their own)
    uint32_t flags;     // ARES_SEG_*
    uint32_t reserved0;
    uint32_t reserved1;
};

struct AresStep {
    uint32_t n_rows;
    uint32_t n_segs;
    uint32_t n_out;
    uint32_t reserved;
    uint32_t off_segs;    // byte offsets from the start of the record
    uint32_t off_tokens;
    uint32_t off_pos;
    uint32_t off_out;
};

static __device__ __forceinline__ const AresSeg* ares_step_segs(const AresStep* s) {
    return reinterpret_cast<const AresSeg*>(reinterpret_cast<const uint8_t*>(s) + s->off_segs);
}
static __device__ __forceinline__ const uint32_t* ares_step_tokens(const AresStep* s) {
    return reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(s) + s->off_tokens);
}
static __device__ __forceinline__ const int32_t* ares_step_pos(const AresStep* s) {
    return reinterpret_cast<const int32_t*>(reinterpret_cast<const uint8_t*>(s) + s->off_pos);
}
static __device__ __forceinline__ const uint32_t* ares_step_out_rows(const AresStep* s) {
    return reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(s) + s->off_out);
}
