// What the kernels of the mixture of experts share: the route record that the
// router writes for a layer of a step, and the slot of an expert.
//
// A layer's experts are not all on the device. Those that are sit in slots of
// a pool: `stride` bytes each, the three matrices of an expert one behind the
// other in the file's quantized bytes (gate at 0, up at `up_offset`, down at
// `down_offset`; src/experts.rs `ExpertShape` computes them). The layer's
// slot table says where each expert is: `slot_of[e]` is its slot, or -1.
//
// The host changes the table while steps run (an upload ends with the write
// of its table entry), under two rules that the kernels rely on:
//
// - An entry appears only after the expert's bytes are in the slot.
// - An entry is removed, and its slot overwritten, only while no kernel of
//   the layer can be reading the slot: between the steps, or for an expert
//   that the layer's router did not pick in the step that is running.
//
// So an entry that a kernel finds stays valid while the kernel runs, and an
// expert that the router picked and that has no entry yet will get one: the
// kernel waits for it on the device (expert_ffn.cuh). The host never has to
// synchronise with the step to repair a miss.
#pragma once

#include "common.cuh"

// The most experts a layer may have: the router's selection runs in one
// thread block with a thread per expert.
#define ARES_EXPERT_MAX_EXPERTS 1024

// The most experts the router picks per row.
#define ARES_EXPERT_MAX_K 16

// How long a block waits for an expert that has no slot before it gives up,
// in polls of the table entry: minutes. An expert that never arrives is a bug
// of the caller; the rows that wanted it then get NaN instead of a device
// that hangs for good.
#define ARES_EXPERT_MAX_POLLS 0xF0000000u

// The head of a route record. The record of a layer and step is, in this
// order (cap = n_rows * k):
//   AresRouteHeader
//   uint32_t ids[cap]          row b's picks at [b * k, (b + 1) * k), largest first
//   float    weights[cap]      their renormalised weights
//   AresExpertGroup groups[cap]  the picked experts, each once, ascending
struct AresRouteHeader {
    uint32_t token;      // the caller's number for the record
    uint32_t n_groups;   // distinct experts picked by the rows
    uint32_t n_missing;  // of those, how many had no slot when the router ran
    uint32_t n_pairs;    // n_rows * k
};

// One picked expert and the rows that picked it.
struct AresExpertGroup {
    uint32_t expert;
    int32_t slot;         // the table's entry when the router ran, -1 for none
    uint32_t rows;        // bit b: row b picked the expert
    uint32_t first_pair;  // pairs (row, expert) of the groups before this one
    float weight[ARES_MAX_BATCH];  // row b's weight for the expert
};

static_assert(sizeof(AresRouteHeader) == 16, "the host reads the record by offsets");
static_assert(sizeof(AresExpertGroup) == 48, "the host reads the record by offsets");

static __device__ __forceinline__ const AresRouteHeader* ares_route_header(const uint8_t* record) {
    return reinterpret_cast<const AresRouteHeader*>(record);
}

static __device__ __forceinline__ const AresExpertGroup* ares_route_groups(const uint8_t* record, uint32_t cap) {
    return reinterpret_cast<const AresExpertGroup*>(record + sizeof(AresRouteHeader) + 8 * static_cast<size_t>(cap));
}

// The index, among the pairs of a group, of row `b` (which must be in `rows`).
static __device__ __forceinline__ uint32_t ares_pair_of_row(uint32_t rows, uint32_t b) {
    return __popc(rows & ((1u << b) - 1u));
}
