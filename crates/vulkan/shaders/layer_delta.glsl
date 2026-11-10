// The gated delta rule of the recurrent block over the rows of segments, and its commit.
// The body of `layer_delta_{window,commit}_{f32,f16}*.comp`, which define
//   DELTA_WINDOW  1 for the window kernel (outputs; the state is written only for direct
//                 segments), 0 for the commit (no outputs, deferred segments only)
//   DELTA_F16     1 when the state is stored as f16 (the arithmetic is f32 either way)
// and may define the shape: DELTA_LPR (invocations per row of the matrix: 1, 2, 4 or 8),
// DELTA_RB (rows of the matrix per workgroup), DELTA_TC (rows of the segment staged at a
// time).
//
// For a value head h with key head j = h mod n_k_heads, the matrix S_h of d x d values
// stored as [m][i] (row m = value index, i = key index, contiguous), and for each row t of
// a segment:
//   decay = exp(g),  g = softplus(alpha + dt_bias[h]) * a[h]     (g <= 0)
//   beta  = sigmoid(beta_in)
//   sk[m] = decay * sum_i S[m][i] k[i]
//   d[m]  = beta * (v[m] - sk[m])
//   S[m][i] = decay * S[m][i] + k[i] d[m]
//   o[m]  = sum_i S[m][i] q[i] / sqrt(d)
// The rows m of S are independent of each other: a cluster of DELTA_LPR invocations owns
// one row (each holding d / DELTA_LPR values of it in registers) and walks the rows of the
// segment in order; nothing is exchanged between clusters except the inputs of the row
// (k, q, v, g, beta), which a workgroup stages in shared memory DELTA_TC rows at a time.
// A workgroup owns DELTA_RB rows of S of one head of one segment.
// Grid: (n_v_heads * d / DELTA_RB, n_segs). Segments of different lengths share a launch.
//
// The default shape keeps the shared memory (12.4 KiB) inside the 16 KiB every Vulkan
// device has.
#include "layer_common.glsl"

#ifndef DELTA_LPR
#define DELTA_LPR 2
#endif
#ifndef DELTA_RB
#define DELTA_RB 64
#endif
#ifndef DELTA_TC
#define DELTA_TC 8
#endif

const uint DK = 128;          // head width
const uint CHUNKS = DK / 4;   // chunks of 4 values per row of S
const uint PER_LANE = CHUNKS / DELTA_LPR;
const uint THREADS = DELTA_RB * DELTA_LPR;

layout(local_size_x = DELTA_RB * DELTA_LPR) in;

layout(push_constant, std430) uniform Push {
    uint64_t step;
    uint64_t q;        // null for the commit
    uint64_t kv;
    uint64_t alpha;
    uint64_t beta_in;
    uint64_t dt_bias;
    uint64_t ssm_a;
    uint64_t state;
    uint64_t slot_stride;
    uint64_t out_;     // null for the commit
    uint n_k_heads;
    uint n_v_heads;
} pc;

shared vec4 ks[DELTA_TC][CHUNKS];
shared float vs[DELTA_TC][DELTA_RB];
shared float decays[DELTA_TC];
shared float betas[DELTA_TC];
#if DELTA_WINDOW
shared vec4 qs[DELTA_TC][CHUNKS];
shared float os[DELTA_TC][DELTA_RB];
#endif

#if DELTA_F16
const uint64_t CHUNK_BYTES = 8;
vec4 load_state(uint64_t a) {
    const uvec2 raw = LHalf4(a).v[0];
    return vec4(unpackHalf2x16(raw.x), unpackHalf2x16(raw.y));
}
void store_state(uint64_t a, vec4 v) {
    LHalf4(a).v[0] = uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#else
const uint64_t CHUNK_BYTES = 16;
vec4 load_state(uint64_t a) { return ldv4(a); }
void store_state(uint64_t a, vec4 v) { LVec4(a).v[0] = v; }
#endif

void main() {
    const Seg seg = step_seg(pc.step, gl_WorkGroupID.y);
    const bool direct = (seg.flags & SEG_DIRECT) != 0u;
    // Every invocation of a workgroup takes the same way out here, so the barriers below
    // are reached by all or by none.
#if DELTA_WINDOW
    const uint n_rows = seg.n_rows;
    const bool write_state = direct;
#else
    if (direct) {
        return;
    }
    const uint n_rows = min(seg.accepted, seg.n_rows);
    const bool write_state = true;
#endif
    if (n_rows == 0u) {
        return;
    }

    const uint slices = DK / DELTA_RB;
    const uint h = gl_WorkGroupID.x / slices;
    const uint slice = gl_WorkGroupID.x % slices;
    const uint kh = h % pc.n_k_heads;
    const uint key_dim = pc.n_k_heads * DK;
    const uint value_dim = pc.n_v_heads * DK;
    const uint kv_stride = key_dim + value_dim;
    const uint tid = gl_LocalInvocationID.x;
    const uint r = tid / DELTA_LPR;    // the row of S inside the slice
    const uint sub = tid % DELTA_LPR;  // the invocation inside the row

    const uint64_t row_state =
        pc.state + (CHUNK_BYTES / 4) * (uint64_t(seg.slot) * pc.slot_stride +
                                        (uint64_t(h) * DK + slice * DELTA_RB + r) * DK);
    vec4 s[PER_LANE];
    [[unroll]] for (uint j = 0; j < PER_LANE; j++) {
        s[j] = load_state(row_state + CHUNK_BYTES * (sub + DELTA_LPR * j));
    }

    const float scale = inversesqrt(float(DK));
    const float dt = ldf(pc.dt_bias + 4 * uint64_t(h));
    const float a = ldf(pc.ssm_a + 4 * uint64_t(h));

    for (uint t0 = 0; t0 < n_rows; t0 += DELTA_TC) {
        const uint tc = min(uint(DELTA_TC), n_rows - t0);
        // Stage the inputs of rows t0 .. t0 + tc - 1.
        for (uint idx = tid; idx < tc * CHUNKS; idx += THREADS) {
            const uint t = idx / CHUNKS;
            const uint c = idx % CHUNKS;
            const uint64_t row = uint64_t(seg.row0) + t0 + t;
            ks[t][c] = ldv4(pc.kv + 4 * (row * kv_stride + kh * DK + 4 * c));
#if DELTA_WINDOW
            qs[t][c] = ldv4(pc.q + 4 * (row * key_dim + kh * DK + 4 * c));
#endif
        }
        for (uint idx = tid; idx < tc * DELTA_RB; idx += THREADS) {
            const uint t = idx / DELTA_RB;
            const uint m = idx % DELTA_RB;
            const uint64_t row = uint64_t(seg.row0) + t0 + t;
            vs[t][m] = ldf(pc.kv + 4 * (row * kv_stride + key_dim + h * DK + slice * DELTA_RB + m));
        }
        if (tid < tc) {
            const uint64_t row = uint64_t(seg.row0) + t0 + tid;
            decays[tid] = exp(l_softplus(ldf(pc.alpha + 4 * (row * pc.n_v_heads + h)) + dt) * a);
            betas[tid] = l_sigmoid(ldf(pc.beta_in + 4 * (row * pc.n_v_heads + h)));
        }
        barrier();

        for (uint t = 0; t < tc; t++) {
            const float decay = decays[t];
            float sk0 = 0.0;
            float sk1 = 0.0;
            [[unroll]] for (uint j = 0; j < PER_LANE; j++) {
                const vec4 k4 = ks[t][sub + DELTA_LPR * j];
                sk0 = fma(s[j].x, k4.x, sk0);
                sk1 = fma(s[j].y, k4.y, sk1);
                sk0 = fma(s[j].z, k4.z, sk0);
                sk1 = fma(s[j].w, k4.w, sk1);
            }
            float sk = sk0 + sk1;
#if DELTA_LPR > 1
            sk = subgroupClusteredAdd(sk, DELTA_LPR);
#endif
            const float d = betas[t] * (vs[t][r] - decay * sk);
#if DELTA_WINDOW
            float o0 = 0.0;
            float o1 = 0.0;
            [[unroll]] for (uint j = 0; j < PER_LANE; j++) {
                const vec4 k4 = ks[t][sub + DELTA_LPR * j];
                const vec4 q4 = qs[t][sub + DELTA_LPR * j];
                s[j] = fma(vec4(decay), s[j], k4 * d);
                o0 = fma(s[j].x, q4.x, o0);
                o1 = fma(s[j].y, q4.y, o1);
                o0 = fma(s[j].z, q4.z, o0);
                o1 = fma(s[j].w, q4.w, o1);
            }
            float o = o0 + o1;
#if DELTA_LPR > 1
            o = subgroupClusteredAdd(o, DELTA_LPR);
#endif
            if (sub == 0u) {
                os[t][r] = o * scale;
            }
#else
            [[unroll]] for (uint j = 0; j < PER_LANE; j++) {
                const vec4 k4 = ks[t][sub + DELTA_LPR * j];
                s[j] = fma(vec4(decay), s[j], k4 * d);
            }
#endif
        }
#if DELTA_WINDOW
        barrier();
        for (uint idx = tid; idx < tc * DELTA_RB; idx += THREADS) {
            const uint t = idx / DELTA_RB;
            const uint m = idx % DELTA_RB;
            FloatRef(pc.out_ + 4 * ((uint64_t(seg.row0) + t0 + t) * value_dim + h * DK +
                                    slice * DELTA_RB + m))
                .v[0] = os[t][m];
        }
#endif
        barrier();
    }

    if (write_state) {
        [[unroll]] for (uint j = 0; j < PER_LANE; j++) {
            store_state(row_state + CHUNK_BYTES * (sub + DELTA_LPR * j), s[j]);
        }
    }
}
