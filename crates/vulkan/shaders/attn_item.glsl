// Attention of the rows of one item over the cells of its ranges: one workgroup is one item
// (`crates/cuda/src/attention/plan.rs` describes items, `src/attention/plan.rs` here).
//
// Compiled per head size (`ATTN_D`: 128 or 256) and capacity in vectors (`ATTN_NV`: 8 or 16;
// the vectors of an item are the query heads of its rows), and specialized per format of K and
// V (`KF`, `VF`, the numbers of `CacheFormat`). The arithmetic is f32 throughout; there is no
// tensor core and no staging of rows, because with a handful of vectors the products wait for
// memory, not for the arithmetic:
//
//   scores   invocation = cell. A chunk is 128 cells. Every invocation walks the row of its
//            own cell, group by group (32 values), decodes the group to f32 and adds its
//            products with all vectors to the vector's score; the queries are in shared
//            memory (rotated and scaled once per item), read as broadcasts.
//   softmax  the scores stay in registers. The maximum of a vector over the chunk is the
//            maximum of the clusters' maxima (one shared exchange), the weights are
//            exp2(score - maximum), their sum likewise, and every invocation keeps the running
//            maximum and sum of every vector (all compute them alike).
//   V        invocation = D / 128 columns of the output for all vectors. The invocations read
//            the rows of the chunk's cells one after the other, decode their own columns and
//            add weight * value to their accumulators.
//
// In the stored formats with a mean (q7 to q4) the decoded value of a code is
// `code * scale + mean`, which is the rotated value; the reference splits the sum into the
// product with the codes and the product with the mean, and the two are equal up to rounding.
// A tile's rows are read through the mean of their own tile (the mean pointers are per cell).
//
// The scores are in the base-2 domain (the queries carry scale * log2(e)), so the softmax is
// exp2, and a partial state keeps m + log2(l) as the CUDA kernels do. At the end the
// accumulators are un-rotated (rotated V), divided by the sum of the weights and written as
// the output with the gate applied, or as a partial state in the scratch for the merge.
#version 450
#include "attn_common.glsl"

#ifndef ATTN_D
#error "ATTN_D (the head size) is not defined"
#endif
#ifndef ATTN_NV
#error "ATTN_NV (the capacity in vectors) is not defined"
#endif


layout(local_size_x = 128) in;

layout(constant_id = 0) const uint KF = 1;
layout(constant_id = 1) const uint VF = 1;

layout(push_constant, std430) uniform Push {
    uint64_t items;
    uint64_t ranges;
    uint64_t rows;
    uint64_t row_ids;
    uint64_t tables;
    uint64_t exact_pool;
    uint64_t conv_pool;
    uint64_t q;
    uint64_t gate;
    uint64_t dst;
    uint64_t st_acc;
    uint64_t st_lse;
    uint64_t cfg;
    float scale_log2;
} pc;

const uint D = ATTN_D;
const uint NV = ATTN_NV;
const uint GROUPS = D / 32u;
// Columns of the output an invocation holds.
const uint DPT = D / 128u;
// Cells per chunk: one per invocation.
const uint CH = 128u;
const float NEG_INF = uintBitsToFloat(0xff800000u);

const uint ITEM_CONVERTED = 1u;
const uint ITEM_CAUSAL = 2u;
const uint ITEM_FINAL = 4u;

// The queries [v][D] as vec4, then (at the end) the accumulators in the same place.
shared vec4 Qs[NV * D / 4u];
// The weights of the chunk [cell][v].
shared vec4 Ps[CH * NV / 4u];
// Where the V row of a cell and the mean of its tile are.
shared uint64_t VRow[CH];
shared uint64_t VMean[CH];
// The maxima and sums of the four clusters, [cluster][v].
shared float WM[4u * NV];
shared float WS[4u * NV];
// The last visible cell of every vector.
shared uint Lim[NV];
// The running maximum and sum of the weights of every vector. One invocation per vector
// updates them between the chunks: in registers they would be 2 * NV more of the registers
// that the accumulators and the scores need.
shared float MS[NV];
shared float LS[NV];

// The 32 values of the group at `grp` of a row in a `qB` format as f32, in the rotated domain:
// code * scale, plus the group's slice of the tile mean at `mean` for the formats that have
// one.
void decode_group(uint fmt, uint64_t grp, uint64_t mean, out float val[32]) {
    const uint bits = fmt_bits(fmt);
    const float sc = h2f(ld2(grp));
    const uint64_t cp = grp + 2ul;
    if (bits == 8u) {
        [[unroll]] for (uint i = 0u; i < 16u; ++i) {
            const uint w = ld2(cp + 2ul * i);
            val[2u * i] = float(int(w << 24) >> 24) * sc;
            val[2u * i + 1u] = float(int(w << 16) >> 24) * sc;
        }
        return;
    }
    uint p0[4];
    [[unroll]] for (uint i = 0u; i < 4u; ++i) {
        p0[i] = ld2(cp + 4ul * i) | (ld2(cp + 4ul * i + 2ul) << 16);
    }
    uint p1[2] = uint[2](0u, 0u);
    if (bits >= 6u) {
        [[unroll]] for (uint i = 0u; i < 2u; ++i) {
            p1[i] = ld2(cp + 16ul + 4ul * i) | (ld2(cp + 18ul + 4ul * i) << 16);
        }
    }
    uint pt = 0u;
    if (bits == 7u || bits == 5u) {
        const uint64_t at = cp + (bits == 7u ? 24ul : 16ul);
        pt = ld2(at) | (ld2(at + 2ul) << 16);
    }
    [[unroll]] for (uint i = 0u; i < 32u; ++i) {
        uint u = (p0[i >> 3] >> (4u * (i & 7u))) & 0xfu;
        if (bits >= 6u) {
            u |= ((p1[i >> 4] >> (2u * (i & 15u))) & 3u) << 4;
        }
        if (bits == 7u || bits == 5u) {
            u |= ((pt >> i) & 1u) << (bits - 1u);
        }
        val[i] = float(int(u << (32u - bits)) >> (32u - bits)) * sc;
    }
    if (fmt_has_mean(fmt)) {
        [[unroll]] for (uint i = 0u; i < 4u; ++i) {
            const uvec4 w = ld16(mean + 16ul * i);
            const vec2 a = unpackHalf2x16(w.x), b = unpackHalf2x16(w.y);
            const vec2 c = unpackHalf2x16(w.z), d = unpackHalf2x16(w.w);
            val[8u * i] += a.x;
            val[8u * i + 1u] += a.y;
            val[8u * i + 2u] += b.x;
            val[8u * i + 3u] += b.y;
            val[8u * i + 4u] += c.x;
            val[8u * i + 5u] += c.y;
            val[8u * i + 6u] += d.x;
            val[8u * i + 7u] += d.y;
        }
    }
}

// The scores of the cell whose K row is at `krow` (and its tile's mean at `kmean`) with all
// vectors, added to s.
void scores(bool conv, uint64_t krow, uint64_t kmean, inout float s[NV]) {
    if (conv && KF != FMT_F16) {
        const uint gb = fmt_group_bytes(KF);
        for (uint gi = 0u; gi < GROUPS; ++gi) {
            float val[32];
            decode_group(KF, krow + uint64_t(gi * gb), kmean + 64ul * gi, val);
            [[unroll]] for (uint i = 0u; i < 8u; ++i) {
                const vec4 kv = vec4(val[4u * i], val[4u * i + 1u], val[4u * i + 2u], val[4u * i + 3u]);
                [[unroll]] for (uint v = 0u; v < NV; ++v) {
                    s[v] += dot(Qs[v * (D / 4u) + gi * 8u + i], kv);
                }
            }
        }
    } else {
        for (uint gi = 0u; gi < GROUPS; ++gi) {
            [[unroll]] for (uint i = 0u; i < 4u; ++i) {
                const uvec4 w = ld16(krow + 64ul * gi + 16ul * i);
                const vec4 a = vec4(unpackHalf2x16(w.x), unpackHalf2x16(w.y));
                const vec4 b = vec4(unpackHalf2x16(w.z), unpackHalf2x16(w.w));
                [[unroll]] for (uint v = 0u; v < NV; ++v) {
                    s[v] += dot(Qs[v * (D / 4u) + gi * 8u + 2u * i], a)
                        + dot(Qs[v * (D / 4u) + gi * 8u + 2u * i + 1u], b);
                }
            }
        }
    }
}

// The DPT values of the columns `col0 ..` of the V row at `vrow` (the mean of its tile at
// `vmean`), as f32 in the stored domain.
void decode_v(bool conv, uint64_t vrow, uint64_t vmean, uint col0, out float val[DPT]) {
    if (!conv || VF == FMT_F16) {
        if (DPT == 2u) {
            const vec2 h = unpackHalf2x16(ld4(vrow + 2ul * col0));
            val[0] = h.x;
            val[DPT - 1u] = h.y;
        } else {
            val[0] = h2f(ld2(vrow + 2ul * col0));
        }
        return;
    }
    const uint bits = fmt_bits(VF);
    const uint gi = col0 / 32u, io = col0 % 32u;
    const uint64_t gp = vrow + uint64_t(gi * fmt_group_bytes(VF));
    const float sc = h2f(ld2(gp));
    const uint64_t cp = gp + 2ul;
    int c0, c1 = 0;
    if (bits == 8u) {
        if (DPT == 2u) {
            const uint w = ld2(cp + io);
            c0 = int(w << 24) >> 24;
            c1 = int(w << 16) >> 24;
        } else {
            c0 = int(ld1(cp + io) << 24) >> 24;
        }
    } else {
        // The fields of the codes io and io + 1 (io is even when DPT is 2): a nibble, two
        // bits for q7 and q6, and a top bit for q7 and q5.
        const uint sh = io & 1u;
        uint u0 = (ld1(cp + (io >> 1)) >> (4u * sh)) & 0xfu;
        uint u1 = (ld1(cp + (io >> 1)) >> (4u * (sh + 1u))) & 0xfu;
        if (bits >= 6u) {
            const uint b = ld1(cp + 16ul + (io >> 2));
            u0 |= ((b >> (2u * (io & 3u))) & 3u) << 4;
            u1 |= ((b >> (2u * ((io + 1u) & 3u))) & 3u) << 4;
        }
        if (bits == 7u || bits == 5u) {
            const uint b = ld1(cp + (bits == 7u ? 24ul : 16ul) + (io >> 3));
            u0 |= ((b >> (io & 7u)) & 1u) << (bits - 1u);
            u1 |= ((b >> ((io + 1u) & 7u)) & 1u) << (bits - 1u);
        }
        c0 = int(u0 << (32u - bits)) >> (32u - bits);
        c1 = int(u1 << (32u - bits)) >> (32u - bits);
    }
    val[0] = float(c0) * sc;
    if (DPT == 2u) {
        val[DPT - 1u] = float(c1) * sc;
    }
    if (fmt_has_mean(VF)) {
        if (DPT == 2u) {
            const vec2 h = unpackHalf2x16(ld4(vmean + 2ul * col0));
            val[0] += h.x;
            val[DPT - 1u] += h.y;
        } else {
            val[0] += h2f(ld2(vmean + 2ul * col0));
        }
    }
}

float of_get(uint v, uint col) { return Qs[(v * D + col) >> 2][col & 3u]; }

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint warp = tid >> 5, lane = lane32();
    const uint64_t item = ares_group();
    const uvec4 w0 = ld16(pc.items + 48ul * item);
    const uvec4 w1 = ld16(pc.items + 48ul * item + 16ul);
    const uvec4 w2 = ld16(pc.items + 48ul * item + 32ul);
    const uint it_table = w0.x, row0 = w0.y, n_rows = w0.z, kv_head = w0.w;
    const uint ranges_off = w1.x, n_ranges = w1.y, total = w1.z, flags = w1.w;
    const uint it_slot = w2.x;
    const Geo g = attn_geo(pc.cfg);
    const uint G = g.G;
    const uint nvec = n_rows * G;
    const bool conv = (flags & ITEM_CONVERTED) != 0u;
    const bool causal = (flags & ITEM_CAUSAL) != 0u;
    const bool is_final = (flags & ITEM_FINAL) != 0u;
    const bool krot = conv && fmt_rotated(KF);
    const bool vrot = conv && fmt_rotated(VF);

    // The queries: rotated when K is stored rotated, then scaled. A vec4 of shared memory is
    // written whole by one invocation of four (a component is not written on its own: that is
    // a read and a write of the vec4 on some devices).
    for (uint v = warp; v < NV; v += 4u) {
        for (uint gi = 0u; gi < GROUPS; ++gi) {
            float x = 0.0;
            if (v < nvec) {
                const uint64_t row = ld4(pc.row_ids + 4ul * (row0 + v / G));
                const uint64_t head = uint64_t(g.head0 + kv_head) * G + v % G;
                x = ldf(pc.q + 4ul * ((row * g.kv_total * G + head) * D + gi * 32u + lane));
                if (krot) {
                    x = rotate32(x, lane);
                }
                x *= pc.scale_log2;
            }
            const float x1 = subgroupShuffle(x, cluster_lane((lane + 1u) & 31u));
            const float x2 = subgroupShuffle(x, cluster_lane((lane + 2u) & 31u));
            const float x3 = subgroupShuffle(x, cluster_lane((lane + 3u) & 31u));
            if ((lane & 3u) == 0u) {
                Qs[v * (D / 4u) + gi * 8u + (lane >> 2)] = vec4(x, x1, x2, x3);
            }
        }
    }
    if (tid < NV) {
        uint lim = 0xffffffffu;
        if (causal && tid < nvec) {
            lim = attn_row(pc.rows, ld4(pc.row_ids + 4ul * (row0 + tid / G))).pos;
        }
        Lim[tid] = lim;
    }

    float acc[NV][DPT];
    [[unroll]] for (uint v = 0u; v < NV; ++v) {
        [[unroll]] for (uint k = 0u; k < DPT; ++k) {
            acc[v][k] = 0.0;
        }
    }
    if (tid < NV) {
        MS[tid] = NEG_INF;
        LS[tid] = 0.0;
    }
    const uint col0 = tid * DPT;
    barrier();

    for (uint j0 = 0u; j0 < total; j0 += CH) {
        const uint j = j0 + tid;
        const bool valid = j < total;
        // The cell of the item's j-th: ranges are (start, end, cells before) triples. Past
        // the end the last cell is read, so that every address exists; its weight is 0.
        const uint jj = valid ? j : total - 1u;
        uint r = 0u;
        if (n_ranges > 1u) {
            uint hi = n_ranges - 1u;
            while (r < hi) {
                const uint mid = (r + hi + 1u) >> 1;
                if (ld4(pc.ranges + 4ul * (ranges_off + 3u * mid + 2u)) <= jj) {
                    r = mid;
                } else {
                    hi = mid - 1u;
                }
            }
        }
        const uint cell = ld4(pc.ranges + 4ul * (ranges_off + 3u * r))
            + (jj - ld4(pc.ranges + 4ul * (ranges_off + 3u * r + 2u)));
        const uint entry = ld4(pc.tables + 4ul * (it_table + (cell >> 7)));
        const uint64_t slot = entry & ~CONVERTED_BIT;
        const uint rr = cell & 127u;
        uint64_t krow, kmean = 0ul;
        if (conv) {
            const uint64_t head = pc.conv_pool + slot * g.entry_c + uint64_t(kv_head) * g.head_c;
            krow = head + g.k_rows_off + uint64_t(rr) * g.rbk;
            kmean = head;
            VRow[tid] = head + g.v_rows_off + uint64_t(rr) * g.rbv;
            VMean[tid] = head + g.v_mean_off;
        } else {
            const uint64_t head =
                pc.exact_pool + slot * g.entry_e + uint64_t(kv_head) * g.head_e;
            krow = head + uint64_t(rr) * (2u * D);
            VRow[tid] = head + uint64_t(TILE_CELLS * 2u * D) + uint64_t(rr) * (2u * D);
            VMean[tid] = 0ul;
        }

        float s[NV];
        [[unroll]] for (uint v = 0u; v < NV; ++v) {
            s[v] = 0.0;
        }
        scores(conv, krow, kmean, s);
        [[unroll]] for (uint v = 0u; v < NV; ++v) {
            s[v] = (valid && cell <= Lim[v]) ? s[v] : NEG_INF;
            const float cluster_max = cl_max(s[v]);
            if (lane == 0u) {
                WM[warp * NV + v] = cluster_max;
            }
        }
        barrier();

        float alpha[NV];
        [[unroll]] for (uint v = 0u; v < NV; ++v) {
            const float chunk_max =
                max(max(WM[v], WM[NV + v]), max(WM[2u * NV + v], WM[3u * NV + v]));
            const float m_old = MS[v];
            const float next = max(m_old, chunk_max);
            float p = 0.0;
            alpha[v] = 1.0;
            if (next != NEG_INF) {
                alpha[v] = exp2(m_old - next);
                p = exp2(s[v] - next);
            }
            s[v] = p;
            const float cluster_sum = subgroupClusteredAdd(p, 32);
            if (lane == 0u) {
                WS[warp * NV + v] = cluster_sum;
            }
        }
        [[unroll]] for (uint vq = 0u; vq < NV / 4u; ++vq) {
            Ps[tid * (NV / 4u) + vq] = vec4(s[4u * vq], s[4u * vq + 1u], s[4u * vq + 2u], s[4u * vq + 3u]);
        }
        barrier();

        // The vector `tid` takes the chunk into its running maximum and sum (the other
        // invocations are in the loop over the cells below).
        if (tid < NV) {
            const float chunk_max = max(
                max(WM[tid], WM[NV + tid]), max(WM[2u * NV + tid], WM[3u * NV + tid]));
            const float m_old = MS[tid];
            const float next = max(m_old, chunk_max);
            if (next != NEG_INF) {
                LS[tid] = LS[tid] * exp2(m_old - next)
                    + ((WS[tid] + WS[NV + tid]) + (WS[2u * NV + tid] + WS[3u * NV + tid]));
                MS[tid] = next;
            }
        }
        [[unroll]] for (uint v = 0u; v < NV; ++v) {
            [[unroll]] for (uint k = 0u; k < DPT; ++k) {
                acc[v][k] *= alpha[v];
            }
        }
        // Eight cells at a time: their rows are asked for before any is used. (Four cells, and m
        // and l in registers, were measured on the same plan: within 5 % for q8 and q4, 13 % slower
        // for f16 with 8 rows over 262144 cells on the RTX 4070; the kernel for 16 vectors takes
        // 143 registers with m and l in shared memory against 165.)
        const uint cells = (min(CH, total - j0) + 7u) & ~7u;
        for (uint c = 0u; c < cells; c += 8u) {
            [[unroll]] for (uint u = 0u; u < 8u; ++u) {
                float val[DPT];
                decode_v(conv, VRow[c + u], VMean[c + u], col0, val);
                [[unroll]] for (uint vq = 0u; vq < NV / 4u; ++vq) {
                    const vec4 p = Ps[(c + u) * (NV / 4u) + vq];
                    [[unroll]] for (uint k = 0u; k < DPT; ++k) {
                        acc[4u * vq][k] += p.x * val[k];
                        acc[4u * vq + 1u][k] += p.y * val[k];
                        acc[4u * vq + 2u][k] += p.z * val[k];
                        acc[4u * vq + 3u][k] += p.w * val[k];
                    }
                }
            }
        }
        barrier();
    }

    // The accumulators to shared memory (the queries are no longer needed), then one cluster
    // per vector: un-rotate, divide by the sum of the weights, write.
    [[unroll]] for (uint v = 0u; v < NV; ++v) {
        // Four consecutive columns are one vec4, written by the invocation that holds the first.
        const uint next = cluster_lane((lane + 1u) & 31u);
        vec4 c;
        if (DPT == 2u) {
            c = vec4(
                acc[v][0], acc[v][DPT - 1u], subgroupShuffle(acc[v][0], next),
                subgroupShuffle(acc[v][DPT - 1u], next));
        } else {
            c = vec4(
                acc[v][0], subgroupShuffle(acc[v][0], next),
                subgroupShuffle(acc[v][0], cluster_lane((lane + 2u) & 31u)),
                subgroupShuffle(acc[v][0], cluster_lane((lane + 3u) & 31u)));
        }
        if (((tid * DPT) & 3u) == 0u) {
            Qs[(v * D + col0) >> 2] = c;
        }
    }
    barrier();
    const uint hq = g.H * G;
    for (uint v = warp; v < nvec; v += 4u) {
        const uint64_t row = ld4(pc.row_ids + 4ul * (row0 + v / G));
        const uint hl = kv_head * G + v % G;
        const float lv = LS[v];
        const float inv = lv > 0.0 ? 1.0 / lv : 0.0;
        const uint64_t piece =
            uint64_t(attn_row(pc.rows, uint(row)).slot_base + it_slot) * hq + hl;
        for (uint gi = 0u; gi < GROUPS; ++gi) {
            float x = of_get(v, gi * 32u + lane);
            if (vrot) {
                x = rotate32(x, lane);
            }
            x *= inv;
            if (is_final) {
                const uint64_t at = (row * g.kv_total * G + uint64_t(g.head0 * G + hl)) * D
                    + gi * 32u + lane;
                if (pc.gate != 0ul) {
                    x *= 1.0 / (1.0 + exp(-ldf(pc.gate + 4ul * at)));
                }
                FloatRef(pc.dst + 4ul * at).v[0] = x;
            } else {
                FloatRef(pc.st_acc + 4ul * (piece * D + gi * 32u + lane)).v[0] = x;
            }
        }
        if (!is_final && lane == 0u) {
            FloatRef(pc.st_lse + 4ul * piece).v[0] = lv > 0.0 ? MS[v] + log2(lv) : NEG_INF;
        }
    }
}
