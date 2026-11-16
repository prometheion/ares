// The kernels of the DFlash drafter (src/dflash.rs, `DflashKernels`): what its
// inject and its block pass need besides the weight products (the decode and
// tile kernels of the main modules), the RMS norm and the sums.
//
// - ares_dflash_tap: a tap of the model, the mean of `n` streams of rows (one
//   stream: a copy), as rows the next product reads;
// - ares_dflash_conv: the dynamic convolution over the rows of each block;
// - ares_dflash_head_norm_rope: the RMS norm of each head (optional) and the
//   rotary turn of its pairs (i, i + head_dim / 2) by the row's position;
// - ares_dflash_ring_write: the keys and values of rows into the ring of
//   their sequence, in f16, at the cell of their position;
// - ares_dflash_attend: each row of a block over the ring's cells of the last
//   `window` positions before it and over every row of the block (not causal);
// - ares_dflash_swiglu: silu(gate) * up;
// - ares_dflash_top: the best k columns of each row of logits;
// - ares_dflash_select: the selector's scores between the candidates of
//   neighbouring positions and the greedy walk through them: the drafts.
//
// Everything is f32 but the ring. Each kernel is a plain loop: the drafter's
// rows are few (a block of 8 per sequence) and its time is in its weights.

#include "common.cuh"
#include "layer_embed.cuh"

#include <cfloat>

#define ARES_DFLASH_WARP 32

static __device__ __forceinline__ float ares_dflash_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

static __device__ __forceinline__ float ares_dflash_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// dst[r * dst_pitch + i] = (1 / n) * sum over s of src[s * stream_pitch + r * row_pitch + i].
// Grid (ceil(d / 256), rows), 256 threads.
extern "C" __global__ void ares_dflash_tap(const float* src, uint32_t n, uint64_t stream_pitch,
                                           uint32_t row_pitch, float* dst, uint32_t dst_pitch,
                                           uint32_t d) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t r = blockIdx.y;
    if (i >= d) return;
    float sum = 0.0f;
    for (uint32_t s = 0; s < n; ++s) sum += src[s * stream_pitch + uint64_t(r) * row_pitch + i];
    dst[uint64_t(r) * dst_pitch + i] = n == 1 ? sum : sum / float(n);
}

// The dynamic convolution of side `side` (0 before the part, 1 after) over
// rows that come in blocks of `block`: y[t][c] = sum over taps j < min(kernel,
// block) with j <= t of (dyn[t][g(c) + groups * (j + kernel * side)] +
// base[c + d * (j + kernel * side)]) * x[t - j][c], g(c) = c / group.
// Grid (ceil(d / 256), rows), 256 threads.
extern "C" __global__ void ares_dflash_conv(const float* x, const float* dyn, const float* base,
                                            float* y, uint32_t d, uint32_t group,
                                            uint32_t kernel, uint32_t side, uint32_t block) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (c >= d) return;
    const uint32_t groups = d / group;
    const uint32_t values = groups * kernel * 2;
    const uint32_t t = row % block;
    const uint32_t g = c / group;
    float sum = 0.0f;
    for (uint32_t j = 0; j < kernel && j < block; ++j) {
        if (j > t) break;
        const uint32_t at = j + kernel * side;
        const float w = dyn[uint64_t(row) * values + g + groups * at] + base[c + uint64_t(d) * at];
        sum += w * x[uint64_t(row - j) * d + c];
    }
    y[uint64_t(row) * d + c] = sum;
}

// Each head of each row: with `norm`, x = x / rms(x) * w (eps inside the
// root); then the pairs (i, i + dim / 2) turn by positions[row] * base^(-2i/dim).
// Grid (heads, rows), dim / 2 threads (dim at most 512).
extern "C" __global__ void ares_dflash_head_norm_rope(float* x, const float* w, uint32_t heads,
                                                      uint32_t dim, const uint32_t* positions,
                                                      float base, float eps, uint32_t norm) {
    __shared__ float part[ARES_DFLASH_WARP];
    const uint32_t half = dim / 2;
    const uint32_t i = threadIdx.x;
    float* h = x + (uint64_t(blockIdx.y) * heads + blockIdx.x) * dim;
    float a = h[i], b = h[i + half];
    if (norm) {
        float sq = ares_dflash_warp_sum(a * a + b * b);
        const uint32_t warp = i / ARES_DFLASH_WARP, lane = i % ARES_DFLASH_WARP;
        if (lane == 0) part[warp] = sq;
        __syncthreads();
        sq = 0.0f;
        for (uint32_t k = 0; k < (half + ARES_DFLASH_WARP - 1) / ARES_DFLASH_WARP; ++k) sq += part[k];
        const float scale = rsqrtf(sq / float(dim) + eps);
        a *= scale * w[i];
        b *= scale * w[i + half];
    }
    const float theta = float(positions[blockIdx.y]) * powf(base, -2.0f * float(i) / float(dim));
    float s, c;
    sincosf(theta, &s, &c);
    h[i] = a * c - b * s;
    h[i + half] = a * s + b * c;
}

// Row r's keys and values (kv values each) into its sequence's ring, as f16:
// the ring of slot s, layer l holds the keys of `cells` cells, then their
// values, at ((s * layers + l) * 2 + part) * cells * kv halves; position p is
// cell p % cells. A row whose slot is ~0 is skipped. Grid (rows), 256 threads.
extern "C" __global__ void ares_dflash_ring_write(const float* k, const float* v, __half* ring,
                                                  const uint32_t* slots,
                                                  const uint32_t* positions, uint32_t layer,
                                                  uint32_t layers, uint32_t cells, uint32_t kv) {
    const uint32_t r = blockIdx.x;
    const uint32_t slot = slots[r];
    if (slot == 0xffffffffu) return;
    const uint64_t cell = positions[r] % cells;
    __half* keys = ring + ((uint64_t(slot) * layers + layer) * 2) * cells * kv + cell * kv;
    __half* values = keys + uint64_t(cells) * kv;
    for (uint32_t i = threadIdx.x; i < kv; i += blockDim.x) {
        keys[i] = __float2half(k[uint64_t(r) * kv + i]);
        values[i] = __float2half(v[uint64_t(r) * kv + i]);
    }
}

// One sequence's block in the attention: its slot, the position of its first
// row, and the first position of its ring that holds a cell of it.
struct AresDflashSeq {
    uint32_t slot;
    uint32_t position;
    uint32_t from;
    uint32_t unused;
};

// Attention of the rows of each block. Row r of sequence q (rows q * block +
// r of `qv`, `kb`, `vb`) at position P = position + r, head h, reads the ring
// cells of positions [max(from, P - window + 1), position) and the keys and
// values of every row of its block, head h / (heads / kv_heads); out =
// softmax(q k * scale) v. Head size 128. Grid (sequences, heads), one warp
// per row of the block (block * 32 threads).
extern "C" __global__ void ares_dflash_attend(const float* qv, const float* kb, const float* vb,
                                              const __half* ring, float* out,
                                              const AresDflashSeq* seqs, uint32_t layer,
                                              uint32_t layers, uint32_t cells, uint32_t window,
                                              uint32_t block, uint32_t heads, uint32_t kv_heads,
                                              float scale) {
    constexpr uint32_t DIM = 128;
    constexpr uint32_t PER = DIM / ARES_DFLASH_WARP;
    __shared__ float qs[16][DIM];
    const uint32_t q = blockIdx.x, h = blockIdx.y;
    const uint32_t r = threadIdx.x / ARES_DFLASH_WARP, lane = threadIdx.x % ARES_DFLASH_WARP;
    if (r >= block) return;
    const AresDflashSeq seq = seqs[q];
    const uint32_t kh = h / (heads / kv_heads);
    const uint32_t kv = kv_heads * DIM;
    const uint64_t row = uint64_t(q) * block + r;
    const float* qrow = qv + row * heads * DIM + h * DIM;
    for (uint32_t i = lane; i < DIM; i += ARES_DFLASH_WARP) qs[r][i] = qrow[i] * scale;
    __syncwarp();
    const int64_t at = int64_t(seq.position) + r;
    int64_t lo = at - int64_t(window) + 1;
    if (lo < int64_t(seq.from)) lo = seq.from;
    if (lo < 0) lo = 0;
    const int64_t hi = seq.position;
    const __half* keys = ring + ((uint64_t(seq.slot) * layers + layer) * 2) * cells * kv;
    const __half* values = keys + uint64_t(cells) * kv;
    float m = -FLT_MAX, sum = 0.0f;
    float acc[PER];
#pragma unroll
    for (uint32_t e = 0; e < PER; ++e) acc[e] = 0.0f;
    // The ring's cells, 32 at a time: lane l scores cell `first + l`.
    for (int64_t first = lo; first < hi; first += ARES_DFLASH_WARP) {
        const int64_t mine = first + lane;
        float s = -FLT_MAX;
        if (mine < hi) {
            const __half* kr = keys + uint64_t(mine % cells) * kv + kh * DIM;
            float dot = 0.0f;
            for (uint32_t i = 0; i < DIM; i += 8) {
                const uint4 raw = *reinterpret_cast<const uint4*>(kr + i);
                const __half2* p = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const float2 f = __half22float2(p[e]);
                    dot += f.x * qs[r][i + 2 * e] + f.y * qs[r][i + 2 * e + 1];
                }
            }
            s = dot;
        }
        const float mn = fmaxf(m, ares_dflash_warp_max(s));
        const float p = mine < hi ? __expf(s - mn) : 0.0f;
        const float fix = __expf(m - mn);
        sum = sum * fix + ares_dflash_warp_sum(p);
#pragma unroll
        for (uint32_t e = 0; e < PER; ++e) acc[e] *= fix;
        m = mn;
        const int64_t count = hi - first < ARES_DFLASH_WARP ? hi - first : ARES_DFLASH_WARP;
        for (int64_t j = 0; j < count; ++j) {
            const float wj = __shfl_sync(0xffffffffu, p, uint32_t(j));
            const __half* vr = values + uint64_t((first + j) % cells) * kv + kh * DIM + lane * PER;
#pragma unroll
            for (uint32_t e = 0; e < PER; ++e) acc[e] += wj * __half2float(vr[e]);
        }
    }
    // The rows of the block, all of them.
    {
        float s = -FLT_MAX;
        if (lane < block) {
            const float* kr = kb + (uint64_t(q) * block + lane) * kv + kh * DIM;
            float dot = 0.0f;
            for (uint32_t i = 0; i < DIM; ++i) dot += kr[i] * qs[r][i];
            s = dot;
        }
        const float mn = fmaxf(m, ares_dflash_warp_max(s));
        const float p = lane < block ? __expf(s - mn) : 0.0f;
        const float fix = __expf(m - mn);
        sum = sum * fix + ares_dflash_warp_sum(p);
#pragma unroll
        for (uint32_t e = 0; e < PER; ++e) acc[e] *= fix;
        for (uint32_t j = 0; j < block; ++j) {
            const float wj = __shfl_sync(0xffffffffu, p, j);
            const float* vr = vb + (uint64_t(q) * block + j) * kv + kh * DIM + lane * PER;
#pragma unroll
            for (uint32_t e = 0; e < PER; ++e) acc[e] += wj * vr[e];
        }
    }
    float* o = out + row * heads * DIM + h * DIM + lane * PER;
#pragma unroll
    for (uint32_t e = 0; e < PER; ++e) o[e] = acc[e] / sum;
}

// out[i] = silu(gate[i]) * up[i]. Grid ceil(n / 256), 256 threads.
extern "C" __global__ void ares_dflash_swiglu(const float* gate, const float* up, float* out,
                                              uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = g / (1.0f + __expf(-g)) * up[i];
}

// Whether (v, c) comes before (bv, bc): the larger value first, equal values
// by the smaller column.
static __device__ __forceinline__ bool ares_dflash_before(float v, uint32_t c, float bv,
                                                          uint32_t bc) {
    return v > bv || (v == bv && c < bc);
}

// The best k columns of each row of logits (n columns, rows `stride` apart),
// the best first, and their values. Round by round, the best of what comes
// after the pick of the round before. Grid (rows), 256 threads.
extern "C" __global__ void ares_dflash_top(const float* logits, uint32_t n, uint32_t stride,
                                           uint32_t k, uint32_t* cols, float* vals) {
    __shared__ float bv[8];
    __shared__ uint32_t bc[8];
    __shared__ float last_v;
    __shared__ uint32_t last_c;
    const float* row = logits + uint64_t(blockIdx.x) * stride;
    const uint32_t warp = threadIdx.x / ARES_DFLASH_WARP, lane = threadIdx.x % ARES_DFLASH_WARP;
    if (threadIdx.x == 0) {
        last_v = FLT_MAX;
        last_c = 0;
    }
    __syncthreads();
    for (uint32_t round = 0; round < k; ++round) {
        const float lv = last_v;
        const uint32_t lc = last_c;
        const bool first = round == 0;
        float v = -FLT_MAX;
        uint32_t c = 0xffffffffu;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const float x = row[i];
            // Only what comes after the last pick.
            if (!first && !ares_dflash_before(lv, lc, x, i)) continue;
            if (c == 0xffffffffu || ares_dflash_before(x, i, v, c)) {
                v = x;
                c = i;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const uint32_t oc = __shfl_xor_sync(0xffffffffu, c, o);
            if (oc != 0xffffffffu && (c == 0xffffffffu || ares_dflash_before(ov, oc, v, c))) {
                v = ov;
                c = oc;
            }
        }
        if (lane == 0) {
            bv[warp] = v;
            bc[warp] = c;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (uint32_t w = 1; w < blockDim.x / ARES_DFLASH_WARP; ++w) {
                if (bc[w] != 0xffffffffu && (c == 0xffffffffu || ares_dflash_before(bv[w], bc[w], v, c))) {
                    v = bv[w];
                    c = bc[w];
                }
            }
            if (c == 0xffffffffu) c = 0;
            cols[blockIdx.x * k + round] = c;
            vals[blockIdx.x * k + round] = v;
            last_v = v;
            last_c = c;
        }
        __syncthreads();
    }
}

// One sequence's pass for the selector: its newest token, the first draft row
// of its window in the step record (or ~0), the drafts the window takes, and
// where its drafts go in the list the host reads.
struct AresDflashSelect {
    uint32_t newest;
    uint32_t record_row;
    uint32_t wanted;
    uint32_t place;
};

// The selector of the drafter and the walk. For sequence q (grid x) and draft
// position j = 0 .. positions - 1 (rows q * positions + j of `cols`, `vals`
// and `gate`): the candidates are the head columns `cols` (tokens through
// `head_tokens`, or the columns themselves when it is null), and the score of
// candidate c after candidate p of the position before (after the newest
// token at j = 0) is vals[c] + sum over the rank of succ[token c][i] *
// pred[token p][i] * gate[i], the two tables rows of `rank` values in type
// `ty` (row_bytes each). The scores go to `lattice` (per sequence and
// position: k tokens as floats, then k * k scores, the rows after p = 0 of
// position 0 equal to it); the walk takes at each position the best score
// after its pick before (the first of equal ones) and its share of the
// softmax of that row. The first `wanted` drafts go into the record's token
// array at `record_row` + j, and with their shares into the host's lists at
// `place` + j. Grid (sequences), 256 threads; k at most 16, rank at most 256.
extern "C" __global__ void ares_dflash_select(
    const uint32_t* cols, const float* vals, const float* gate, const uint32_t* head_tokens,
    const uint8_t* succ, const uint8_t* pred, uint32_t ty, uint32_t row_bytes, uint32_t rank,
    uint32_t k, uint32_t positions, const AresDflashSelect* seqs, float* lattice,
    uint32_t* record_tokens, uint32_t* drafts, float* probs) {
    __shared__ float s_rows[16][256];
    __shared__ float p_rows[16][256];
    __shared__ uint32_t tokens[2][16];
    __shared__ float scores[16][16];
    __shared__ uint32_t picked;
    const uint32_t q = blockIdx.x;
    const AresDflashSelect seq = seqs[q];
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = k + k * k;
    if (tid == 0) picked = 0;
    for (uint32_t j = 0; j < positions; ++j) {
        const uint32_t at = q * positions + j;
        const uint32_t now = j & 1, before = now ^ 1;
        if (tid < k) {
            const uint32_t c = cols[at * k + tid];
            tokens[now][tid] = head_tokens ? head_tokens[c] : c;
        }
        __syncthreads();
        // The successor rows of this position's candidates, the predecessor
        // rows of the position before's (or of the newest token), each times
        // the gate.
        const uint32_t n_pred = j == 0 ? 1 : k;
        for (uint32_t i = tid; i < k * rank; i += blockDim.x) {
            const uint32_t c = i / rank, e = i % rank;
            s_rows[c][e] = ares_embed_value(succ + uint64_t(tokens[now][c]) * row_bytes, ty, e);
        }
        for (uint32_t i = tid; i < n_pred * rank; i += blockDim.x) {
            const uint32_t p = i / rank, e = i % rank;
            const uint32_t t = j == 0 ? seq.newest : tokens[before][p];
            p_rows[p][e] = ares_embed_value(pred + uint64_t(t) * row_bytes, ty, e) *
                           gate[uint64_t(at) * rank + e];
        }
        __syncthreads();
        for (uint32_t i = tid; i < k * k; i += blockDim.x) {
            const uint32_t p = i / k, c = i % k;
            float dot = 0.0f;
            if (p < n_pred) {
                for (uint32_t e = 0; e < rank; ++e) dot += s_rows[c][e] * p_rows[p][e];
            }
            scores[p][c] = vals[at * k + c] + dot;
        }
        __syncthreads();
        if (j == 0) {
            for (uint32_t i = tid; i < k * k; i += blockDim.x) scores[i / k][i % k] = scores[0][i % k];
            __syncthreads();
        }
        float* out = lattice + uint64_t(at) * stride;
        for (uint32_t i = tid; i < k; i += blockDim.x) out[i] = float(tokens[now][i]);
        for (uint32_t i = tid; i < k * k; i += blockDim.x) out[k + i] = scores[i / k][i % k];
        if (tid == 0) {
            const float* row = scores[picked];
            uint32_t best = 0;
            for (uint32_t c = 1; c < k; ++c) {
                if (row[c] > row[best]) best = c;
            }
            float sum = 0.0f;
            for (uint32_t c = 0; c < k; ++c) sum += __expf(row[c] - row[best]);
            if (j < seq.wanted) {
                const uint32_t token = tokens[now][best];
                if (seq.record_row != 0xffffffffu) record_tokens[seq.record_row + j] = token;
                drafts[seq.place + j] = token;
                probs[seq.place + j] = 1.0f / sum;
            }
            picked = best;
        }
        __syncthreads();
    }
}
