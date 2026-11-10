// The loop over K and the tensor-core products of the f16 prefill kernels; see
// `mm_f16_head.glsl`. The weights piece before it defines
//   void weights_setup()              once, after tid, sg, lane, wt and n_steps are known
//   void weights_load(uint s)         requests the data of step s (into registers)
//   void weights_store(uint s, uint st)  decodes it and writes the A tile of stage st
// The steps overlap: while the products of step s run, the data of step s + 1 is loaded.

void main() {
    const uint n_wt = (pc.n_out + MT - 1) / MT;
    const uint n_bt = (pc.n_rows + NT - 1) / NT;
    // The workgroups of one group of activation tiles take all weight tiles in turn.
    const uint id = uint(ares_group());
    const uint g_all = pc.group * n_wt;
    const uint gi = id / g_all;
    const uint g_eff = min(pc.group, n_bt - gi * pc.group);
    const uint within = id - gi * g_all;
    wt = within / g_eff;
    const uint bt = gi * pc.group + within % g_eff;
    if (wt >= n_wt) {
        return;
    }
    tid = gl_LocalInvocationID.x;
    sg = tid >> 5;
    lane = tid & 31u;
    n_steps = pc.n_in >> 6;

    // This subgroup's 32 x 32 corner of the tile, for the products.
    const uint m_base = (sg >> 1) * 32;
    const uint n_base = (sg & 1u) * 32;

#if ARES_ACC32
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[2][2];
#else
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[2][2];
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> accf[2][2];
#endif
    [[unroll]] for (uint a = 0; a < 2; a++) {
        [[unroll]] for (uint b = 0; b < 2; b++) {
#if ARES_ACC32
            acc[a][b] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#else
            acc[a][b] =
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0hf);
            accf[a][b] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
#endif
        }
    }

    weights_setup();
    barrier();

    // The activation tile of step s: 64 rows of 64 halves = 8 vectors of 16 bytes each,
    // 4 per invocation.
    uvec4 xa[4];

#define LOAD_ACT(s)                                                                         \
    [[unroll]] for (uint i = 0; i < 4; i++) {                                               \
        const uint idx = tid + 128 * i;                                                     \
        const uint row = min(bt * NT + (idx >> 3), pc.n_rows - 1);                          \
        xa[i] = ld16(pc.act + 2 * (uint64_t(row) * pc.n_in + uint64_t(s) * 64) +            \
                     16 * (idx & 7u));                                                      \
    }
#define STORE_ACT(st)                                                                       \
    [[unroll]] for (uint i = 0; i < 4; i++) {                                               \
        const uint idx = tid + 128 * i;                                                     \
        sb[st][(idx >> 3) * ROW16 + (idx & 7u)] = xa[i];                                    \
    }

    weights_load(0u);
    LOAD_ACT(0u)
    weights_store(0u, 0u);
    STORE_ACT(0u)
    barrier();

    for (uint ks = 0; ks < n_steps; ks++) {
        const uint cur = ks & 1u;
        const uint nxt = cur ^ 1u;
        const bool more = ks + 1 < n_steps;
        if (more) {
            weights_load(ks + 1);
            LOAD_ACT(ks + 1)
        }
        // --- the products of step ks: 4 slabs of 16 values --------------------------------
        [[unroll]] for (uint kk = 0; kk < 4; kk++) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ma[2];
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mb[2];
            [[unroll]] for (uint a = 0; a < 2; a++) {
                coopMatLoad(ma[a], sa[cur], (m_base + 16 * a) * ROW16 * 4 + 8 * kk, ROW16 * 4,
                            gl_CooperativeMatrixLayoutRowMajor);
                coopMatLoad(mb[a], sb[cur], (n_base + 16 * a) * ROW16 + 2 * kk, ROW16,
                            gl_CooperativeMatrixLayoutColumnMajor);
            }
            [[unroll]] for (uint a = 0; a < 2; a++) {
                [[unroll]] for (uint b = 0; b < 2; b++) {
                    acc[a][b] = coopMatMulAdd(ma[a], mb[b], acc[a][b]);
                }
            }
        }
#if !ARES_ACC32
        // The partial sums of 64 products go to f32.
        [[unroll]] for (uint a = 0; a < 2; a++) {
            [[unroll]] for (uint b = 0; b < 2; b++) {
                accf[a][b] +=
                    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc[a][b]);
                acc[a][b] =
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0hf);
            }
        }
#endif
        if (more) {
            weights_store(ks + 1, nxt);
            STORE_ACT(nxt)
        }
        barrier();
    }

    // --- the result: stage each matrix in shared memory (column-major: element (m, n) at
    // n * 16 + m) and write rows of 16 consecutive outputs. The factor of each activation
    // row (convert_act_f16.comp) is put back here.
    const uint64_t scales = pc.act + act_ds_offset(uint64_t(pc.n_rows) * pc.n_in * 2);
    [[unroll]] for (uint a = 0; a < 2; a++) {
        [[unroll]] for (uint b = 0; b < 2; b++) {
#if ARES_ACC32
            coopMatStore(acc[a][b], stage[sg], 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
#else
            coopMatStore(accf[a][b], stage[sg], 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
#endif
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            [[unroll]] for (uint i = 0; i < 8; i++) {
                const uint idx = lane + 32 * i;
                const uint n = bt * NT + n_base + 16 * b + (idx >> 4);
                const uint m = wt * MT + m_base + 16 * a + (idx & 15u);
                if (m < pc.n_out && n < pc.n_rows) {
                    FloatRef(pc.y + 4 * (uint64_t(n) * pc.n_out + m)).v[0] =
                        stage[sg][idx] * ldf1(scales + 4 * uint64_t(n));
                }
            }
            subgroupMemoryBarrierShared();
            subgroupBarrier();
        }
    }
}
