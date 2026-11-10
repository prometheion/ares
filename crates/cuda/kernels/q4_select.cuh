// The selection of the cells a row of qwen4exp attends to (section 3.6 of
// docs/reference/qwen4exp-architecture.md) and its hand-over to the attention
// kernels as lists of cell ranges.
//
// A row that sees n = pos + 1 cells has floor(n / B) blocks and a tail of
// n % B cells. Up to top_k + B - 1 cells it attends to all of them. Past that
// it takes its tail and the best blocks by score, equal scores by the lower
// block first:
//   rule 0 (cell budget):  until top_k + B - 1 cells are selected in all; the
//                          last block taken gives only its first cells when
//                          the budget ends inside it
//   rule 1 (whole blocks): the best top_k / B blocks whole
//
// The result of a row is two ascending lists of ranges, one for the cells in
// converted entries of the attention tables and one for those in exact
// entries, as (start, end, cells of the list before this range) triples: the
// form the ranges of an AttnItem have (attn_kernel.cuh). A range never
// crosses from an entry of one kind into one of the other.
#pragma once

#include "common.cuh"
#include "q4_indexer.cuh"

// The threads of a block of the select kernel. A constant: with it the compiler unrolls the
// passes over the scores, which keeps several loads in flight; the same passes with the block
// size read at run time took twice as long.
#define Q4_SEL_THREADS 256u
// Blocks of a row at most: 2^16 (262 144 cells of 4), one bit each, and the bit of the tail.
#define Q4_SEL_MAX_WORDS 2049u
#define Q4_SEL_NONE 0xffffffffu
#define Q4_MAX_PIECES 64u

// The exclusive prefix sum of `v` over the threads of the block in thread
// order; `*total` gets the sum over all threads. `warp_totals` is shared
// memory with one word per warp.
static __device__ __forceinline__ uint32_t q4_block_scan(uint32_t v, uint32_t* warp_totals, uint32_t* total) {
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t inclusive = v;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t below = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= static_cast<uint32_t>(offset)) {
            inclusive += below;
        }
    }
    __syncthreads();  // the previous use of warp_totals is over
    if (lane == 31) {
        warp_totals[warp] = inclusive;
    }
    __syncthreads();
    uint32_t before = 0, all = 0;
    const uint32_t warps = blockDim.x >> 5;
    for (uint32_t w = 0; w < warps; w++) {
        const uint32_t t = warp_totals[w];
        before += w < warp ? t : 0;
        all += t;
    }
    *total = all;
    return before + inclusive - v;
}

// The order of scores as unsigned integers. Scores are sums of rectified
// values, so they are not negative; anything with the sign bit ranks lowest.
static __device__ __forceinline__ uint32_t q4_score_key(float score) {
    const uint32_t bits = __float_as_uint(score);
    return (bits >> 31) != 0 ? 0u : bits;
}

// Selection for the rows row0 .. row0 + grid.x - 1 of the step, one thread
// block of 256 threads per row. Every pass over the scores waits for memory:
// a row is bound by how many loads its block keeps in flight.
//
//   scores   [row of the launch][score_stride] f32 from ares_q4_block_scores
//            (not read for a row within the dense limit)
//   tables   the attention tables (bit 31 of an entry: converted), or null when
//            all entries are of one kind: then everything is in the first list
//   lists    [row of the launch][cap] triples; the converted list first, the
//            exact list behind it
//   counts   [row of the launch][4]: ranges and cells of the converted list,
//            ranges and cells of the exact list. A row whose lists need more
//            than `cap` triples gets the counts and only the first `cap` triples.
//
// The K-th largest score is found by a radix select over the bits of the
// scores (eight rounds of 4 bits; the threads count digits in registers, so
// no round waits on shared counters whatever the scores are); blocks
// above it are taken, and of the blocks equal to it the lowest indices, by a
// prefix count over the words of a mask of the equal ones. The chosen blocks
// are a bitmap in shared memory, from which every thread writes the ranges of
// its words at offsets given by prefix sums. Every pass over the scores reads
// 32 consecutive ones per warp.
// B is 1, 2 or 4, so that an entry of 128 cells is whole words of the bitmap.
ARES_KERNEL ares_q4_select(const Q4AttnRow* attn_rows, const float* scores, const uint32_t* tables,
                           uint32_t* lists, uint32_t* counts, uint64_t cfg, uint32_t row0, uint32_t score_stride,
                           uint32_t top_k, uint32_t rule, uint32_t cap) {
    __shared__ uint32_t bitmap[Q4_SEL_MAX_WORDS + 1];
    __shared__ uint32_t ties[Q4_SEL_MAX_WORDS + 1];
    __shared__ uint32_t hist[16];
    __shared__ uint32_t warp_totals[Q4_SEL_THREADS / 32];
    __shared__ uint32_t shared[4];

    const Q4Geo g = q4_geo(cfg);
    const uint32_t tid = threadIdx.x, threads = Q4_SEL_THREADS;
    const Q4AttnRow r = attn_rows[row0 + blockIdx.x];
    const uint32_t n = r.pos + 1;
    const uint32_t B = g.block;
    const uint32_t nb = n / B, tail = n % B;
    const uint32_t words = nb / 32 + 1;
    const float* s = scores + static_cast<size_t>(blockIdx.x) * score_stride;
    // Each thread owns a run of consecutive blocks that is whole words.
    const uint32_t per = max(((nb + threads - 1) / threads + 31) & ~31u, 32u);
    const uint32_t j_lo = min(tid * per, nb), j_hi = min(j_lo + per, nb);
    const uint32_t dense_limit = top_k + B - 1;
    const bool dense = n <= dense_limit;

    for (uint32_t w = tid; w < words; w += threads) {
        bitmap[w] = 0;
    }
    if (tid == 0) {
        shared[2] = Q4_SEL_NONE;
    }
    uint32_t rem = 0;
    if (dense) {
        __syncthreads();
        for (uint32_t j = j_lo; j < j_hi; j += 32) {
            bitmap[j >> 5] = j_hi - j >= 32 ? 0xffffffffu : (1u << (j_hi - j)) - 1;
        }
    } else {
        const uint32_t budget = rule == 0 ? dense_limit - tail : top_k;
        rem = budget % B;
        // The row sees more blocks than it takes (n > top_k + B - 1), so the K-th exists.
        uint32_t k_rem = budget / B + (rem != 0 ? 1 : 0);
        uint32_t prefix = 0, mask = 0;
        const uint32_t lane = tid & 31, warp = tid >> 5, warps = threads / 32;
        for (int shift = 28; shift >= 0; shift -= 4) {
            if (tid < 16) {
                hist[tid] = 0;
            }
            __syncthreads();
            // A thread counts the 16 digits of its scores (every 256th, so a warp reads 32
            // consecutive ones) in registers, 16 bits per digit: at most 256 scores each.
            uint64_t count[4] = {0, 0, 0, 0};
            for (uint32_t j = tid; j < nb; j += threads) {
                const uint32_t key = q4_score_key(s[j]);
                if ((key & mask) == prefix) {
                    const uint32_t digit = (key >> shift) & 15u;
                    const uint64_t one = 1ull << ((digit & 3u) * 16);
#pragma unroll
                    for (uint32_t word = 0; word < 4; word++) {
                        count[word] += (digit >> 2) == word ? one : 0ull;
                    }
                }
            }
            // The sums of a warp still fit 16 bits (at most 8192); its first lane adds them
            // to the histogram of the block.
#pragma unroll
            for (uint32_t word = 0; word < 4; word++) {
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    count[word] += __shfl_xor_sync(0xffffffffu, count[word], offset);
                }
            }
            if (lane == 0) {
#pragma unroll
                for (uint32_t digit = 0; digit < 16; digit++) {
                    atomicAdd(&hist[digit],
                              static_cast<uint32_t>((count[digit >> 2] >> ((digit & 3u) * 16)) & 0xffffull));
                }
            }
            __syncthreads();
            if (tid == 0) {
                uint32_t above = 0;
                int bin = 15;
                for (; bin > 0; bin--) {
                    if (above + hist[bin] >= k_rem) {
                        break;
                    }
                    above += hist[bin];
                }
                shared[0] = prefix | (static_cast<uint32_t>(bin) << shift);
                shared[1] = k_rem - above;
            }
            __syncthreads();
            prefix = shared[0];
            k_rem = shared[1];
            mask |= 15u << shift;
        }
        // Blocks above `prefix` are taken, and the k_rem lowest of the blocks equal to it.
        // One pass writes, per word of 32 blocks, which are above and which are equal.
        for (uint32_t w = warp; w * 32 < nb; w += warps) {
            const uint32_t j = w * 32 + lane;
            const uint32_t key = j < nb ? q4_score_key(s[j]) : 0u;
            const uint32_t above = __ballot_sync(0xffffffffu, j < nb && key > prefix);
            const uint32_t equal = __ballot_sync(0xffffffffu, j < nb && key == prefix);
            if (lane == 0) {
                bitmap[w] = above;
                ties[w] = equal;
            }
        }
        __syncthreads();
        // The rank of a tie is the number of ties before it: a prefix sum over the words,
        // which a thread owns in runs.
        const uint32_t tie_words = (nb + 31) / 32;
        const uint32_t run = (tie_words + threads - 1) / threads;
        const uint32_t t_lo = min(tid * run, tie_words), t_hi = min(t_lo + run, tie_words);
        uint32_t mine = 0;
        for (uint32_t w = t_lo; w < t_hi; w++) {
            mine += __popc(ties[w]);
        }
        uint32_t all_ties;
        uint32_t rank = q4_block_scan(mine, warp_totals, &all_ties);
        for (uint32_t w = t_lo; w < t_hi; w++) {
            uint32_t left = ties[w];
            const uint32_t count = __popc(left);
            if (rank < k_rem && count != 0) {
                const uint32_t want = min(k_rem - rank, count);
                uint32_t take = 0, last = 0;
                for (uint32_t i = 0; i < want; i++) {
                    last = left & (0u - left);
                    take |= last;
                    left ^= last;
                }
                bitmap[w] |= take;
                if (rem != 0 && rank + want == k_rem) {
                    shared[2] = 32 * w + (__ffs(last) - 1);  // the last block in rank order gives `rem` cells
                }
            }
            rank += count;
        }
    }
    __syncthreads();
    if (tid == 0 && tail != 0) {
        bitmap[nb >> 5] |= 1u << (nb & 31);  // the tail, as one more block of `tail` cells
    }
    __syncthreads();
    const uint32_t part = shared[2];  // the block that gives only `rem` cells, if any

    // Ranges. A thread owns consecutive words; list 0 is converted, list 1 exact.
    const uint32_t wpe = g.bpe / 32;  // words per entry
    const uint32_t wper = (words + threads - 1) / threads;
    const uint32_t w_lo = min(tid * wper, words), w_hi = min(w_lo + wper, words);
    auto kind_of = [&](uint32_t w) -> uint32_t {
        return (tables == nullptr || (tables[r.table + w / wpe] >> 31) != 0) ? 0u : 1u;
    };
    // The start and end bits of the ranges in word w and the cells its blocks give.
    auto scan_word = [&](uint32_t w, uint32_t& starts, uint32_t& ends, uint32_t& cells) -> uint32_t {
        const uint32_t S = bitmap[w];
        const uint32_t kind = kind_of(w);
        uint32_t prev = 0, next = 0;
        if (w > 0 && part != 32 * w - 1 && kind_of(w - 1) == kind) {
            prev = bitmap[w - 1] >> 31;
        }
        if (w + 1 < words && kind_of(w + 1) == kind) {
            next = bitmap[w + 1] & 1u;
        }
        starts = S & ~((S << 1) | prev);
        ends = S & ~((S >> 1) | (next << 31));
        cells = B * __popc(S);
        if (part != Q4_SEL_NONE && (part >> 5) == w) {
            const uint32_t b = part & 31;
            ends |= 1u << b;
            if (b < 31) {
                starts |= S & (1u << (b + 1));
            }
            cells -= B - rem;
        }
        if (tail != 0 && (nb >> 5) == w) {
            cells -= B - tail;
        }
        return kind;
    };
    // Counts per thread, the two lists packed as 16 bits each (a list has at most 65535
    // ranges and cells: top_k is far below that).
    uint32_t n_starts = 0, n_ends = 0, n_cells = 0;
    for (uint32_t w = w_lo; w < w_hi; w++) {
        uint32_t starts, ends, cells;
        const uint32_t shift = 16 * scan_word(w, starts, ends, cells);
        n_starts += static_cast<uint32_t>(__popc(starts)) << shift;
        n_ends += static_cast<uint32_t>(__popc(ends)) << shift;
        n_cells += cells << shift;
    }
    uint32_t all_starts, all_ends, all_cells;
    const uint32_t start_base = q4_block_scan(n_starts, warp_totals, &all_starts);
    const uint32_t end_base = q4_block_scan(n_ends, warp_totals, &all_ends);
    const uint32_t cell_base = q4_block_scan(n_cells, warp_totals, &all_cells);
    const uint32_t n_conv = all_starts & 0xffffu;
    uint32_t si[2] = {start_base & 0xffffu, n_conv + (start_base >> 16)};
    uint32_t ei[2] = {end_base & 0xffffu, n_conv + (end_base >> 16)};
    uint32_t cb[2] = {cell_base & 0xffffu, cell_base >> 16};
    uint32_t* list = lists + static_cast<size_t>(blockIdx.x) * cap * 3;
    for (uint32_t w = w_lo; w < w_hi; w++) {
        uint32_t starts, ends, cells;
        const uint32_t kind = scan_word(w, starts, ends, cells);
        const uint32_t S = bitmap[w];
        while (starts != 0) {
            const uint32_t b = __ffs(starts) - 1;
            starts &= starts - 1;
            uint32_t before = cb[kind] + B * __popc(S & ((1u << b) - 1));
            if (part != Q4_SEL_NONE && (part >> 5) == w && (part & 31) < b) {
                before -= B - rem;
            }
            const uint32_t at = si[kind]++;
            if (at < cap) {
                list[3 * at] = B * (32 * w + b);
                list[3 * at + 2] = before;
            }
        }
        while (ends != 0) {
            const uint32_t b = __ffs(ends) - 1;
            ends &= ends - 1;
            const uint32_t j = 32 * w + b;
            const uint32_t len = j == part ? rem : (j == nb ? tail : B);
            const uint32_t at = ei[kind]++;
            if (at < cap) {
                list[3 * at + 1] = B * j + len;
            }
        }
        cb[kind] += cells;
    }
    if (tid == 0) {
        uint32_t* count = counts + static_cast<size_t>(blockIdx.x) * 4;
        count[0] = n_conv;
        count[1] = all_cells & 0xffffu;
        count[2] = all_starts >> 16;
        count[3] = all_cells >> 16;
    }
}

// What the attention plan reserved for a row whose lists are made on the
// device (32 bytes).
struct Q4Hand {
    uint32_t ranges_off;    // index in the plan's `ranges` (in u32) of the row's lists
    uint32_t item_conv;     // first of the items for the converted list: [piece][KV head]
    uint32_t pieces_conv;   // pieces reserved for it (0: the row has no converted entries)
    uint32_t item_exact;    // the same for the exact list
    uint32_t pieces_exact;
    uint32_t kv_heads;      // items per piece
    uint32_t reserved0;
    uint32_t reserved1;
};

// The fields of AttnItem (attn_kernel.cuh, 48 bytes) the hand-over writes.
#define Q4_ITEM_WORDS 12u
#define Q4_ITEM_RANGES_OFF 4u
#define Q4_ITEM_N_RANGES 5u
#define Q4_ITEM_TOTAL 6u

// The hand-over of the lists of ares_q4_select to the attention kernels, one
// thread block per row of the launch. The select kernel wrote the row's lists
// into the plan's `ranges` at `hand.ranges_off` (its `lists` argument is that
// address for the first row and `cap` the distance between rows). Each list
// is cut into its reserved pieces of about equal numbers of cells, at range
// boundaries: the `before` of every range is made relative to its piece, and
// the items of piece i get their `ranges_off`, `n_ranges` and `total`. A piece
// that gets no range has `total = 0`.
ARES_KERNEL ares_q4_hand_over(const Q4Hand* hands, const uint32_t* counts, uint32_t* ranges, uint32_t* items) {
    __shared__ uint32_t first[Q4_MAX_PIECES + 1];
    __shared__ uint32_t base[Q4_MAX_PIECES + 1];
    const Q4Hand hand = hands[blockIdx.x];
    const uint32_t* count = counts + static_cast<size_t>(blockIdx.x) * 4;
    const uint32_t tid = threadIdx.x;
    uint32_t list_at = 0;  // triples of the row's lists before this one
    for (uint32_t kind = 0; kind < 2; kind++) {
        const uint32_t n = count[2 * kind], cells = count[2 * kind + 1];
        const uint32_t pieces = min(kind == 0 ? hand.pieces_conv : hand.pieces_exact, Q4_MAX_PIECES);
        const uint32_t item0 = kind == 0 ? hand.item_conv : hand.item_exact;
        uint32_t* list = ranges + hand.ranges_off + 3 * static_cast<size_t>(list_at);
        list_at += n;
        if (pieces == 0) {
            continue;
        }
        for (uint32_t i = tid; i <= pieces; i += blockDim.x) {
            first[i] = Q4_SEL_NONE;
            base[i] = 0;
        }
        __syncthreads();
        // The piece of a range is where its first cell falls among the cells of the list.
        auto piece_of = [&](uint32_t before) -> uint32_t {
            return static_cast<uint32_t>(static_cast<uint64_t>(before) * pieces / max(cells, 1u));
        };
        for (uint32_t i = tid; i < n; i += blockDim.x) {
            const uint32_t before = list[3 * i + 2];
            const uint32_t piece = piece_of(before);
            if (i == 0 || piece_of(list[3 * (i - 1) + 2]) != piece) {
                first[piece] = i;
                base[piece] = before;
            }
        }
        __syncthreads();
        if (tid == 0) {
            first[pieces] = n;
            base[pieces] = cells;
            for (uint32_t i = pieces; i-- > 0;) {
                if (first[i] == Q4_SEL_NONE) {
                    first[i] = first[i + 1];
                    base[i] = base[i + 1];
                }
            }
        }
        __syncthreads();
        for (uint32_t i = tid; i < n; i += blockDim.x) {
            const uint32_t before = list[3 * i + 2];
            list[3 * i + 2] = before - base[piece_of(before)];
        }
        const uint32_t list_off = hand.ranges_off + 3 * (list_at - n);
        for (uint32_t i = tid; i < pieces * hand.kv_heads; i += blockDim.x) {
            const uint32_t piece = i / hand.kv_heads;
            uint32_t* item = items + static_cast<size_t>(item0 + i) * Q4_ITEM_WORDS;
            item[Q4_ITEM_RANGES_OFF] = list_off + 3 * first[piece];
            item[Q4_ITEM_N_RANGES] = first[piece + 1] - first[piece];
            item[Q4_ITEM_TOTAL] = base[piece + 1] - base[piece];
        }
        __syncthreads();
    }
}
