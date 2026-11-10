// The skeleton of the quantized weight kernels: a quantized matrix times 1 to 8
// activation rows, with integer dot products over activations that
// quantize_act.comp has quantized to int8. The GLSL counterpart of `mmvq.cuh`, with
// the same structure and the same arithmetic; read its comment for why it looks
// like this.
//
// A kernel file defines a type (a `mmvq_<type>.glsl`) and includes this file:
//   BLOCK_VALUES, BLOCK_BYTES    the geometry of a block (GGUF layout)
//   LANES                        lanes per block; divides 32
//   struct W, A                  what a lane holds for one block
//   void mmvq_init(uint t, uint64_t tables)
//                                copies the type's codebook to shared memory from the
//                                table buffer (or does nothing); every invocation calls
//                                it, with its index in the workgroup, before anything
//                                else
//   W load_w(uint64_t blk, uint slot)
//   A load_a(uint64_t q, uint64_t ds, uint slot)
//   void dot_acc(inout float acc, W w, A a)
//
// y[b][r] = sum over c of W[r][c] * x[b][c], for b < n_batch, r < n_out.
//   w   n_out rows of n_in / BLOCK_VALUES blocks, aligned for the widest load of
//       the type (16 bytes for Q4_K and Q5_K, 8 for IQ4_XS, 2 for the rest)
//   aq  the quantized activations (quantize_act.comp), 16-byte aligned
//   y   n_batch rows of n_out f32 values
//   tables  the codebooks of the i-quants (iq_tables.bin)
//
// A workgroup is 4 warps of 32 lanes and computes 8 output rows, 2 to a warp. On a
// device whose subgroup is 64 wide a subgroup is two warps. The warp is a logical
// unit here (the 32 lanes of a cluster of the subgroup); the sum over its lanes is
// a clustered reduction.

layout(local_size_x = ARES_MMVQ_WARPS * 32) in;

layout(push_constant, std430) uniform Push {
    uint64_t w;
    uint64_t aq;
    uint64_t y;
    uint64_t tables;
    uint n_in;
    uint n_out;
    uint n_batch;
} pc;

const uint WARPS = ARES_MMVQ_WARPS;
const uint ROWS = ARES_MMVQ_ROWS;

void main() {
    const uint64_t group = ares_group();
    // A whole workgroup returns here, before any barrier.
    if (group * (WARPS * ROWS) >= pc.n_out) {
        return;
    }
    mmvq_init(gl_LocalInvocationID.x, pc.tables);

    const uint lane = gl_SubgroupInvocationID & 31u;
    const uint warp = gl_SubgroupID * (gl_SubgroupSize >> 5) + (gl_SubgroupInvocationID >> 5);
    mmvq_lane = lane;
    const uint row0 = uint(group) * (WARPS * ROWS) + warp * ROWS;

    const uint blocks_per_row = pc.n_in / BLOCK_VALUES;
    const uint64_t row_bytes = uint64_t(blocks_per_row) * BLOCK_BYTES;
    const uint groups_per_row = pc.n_in >> 5;
    const uint slot = lane % LANES;
    const uint kb_step = 32 / LANES;
    const uint64_t ad = pc.aq + act_ds_offset(uint64_t(pc.n_batch) * pc.n_in);

    // The rows past the end of the matrix repeat the last one; their results are
    // not written.
    uint64_t wrow[ROWS];
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        wrow[r] = pc.w + uint64_t(min(row0 + r, pc.n_out - 1)) * row_bytes;
    }

    float acc[ROWS][ARES_MAX_BATCH];
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        [[unroll]] for (uint b = 0; b < ARES_MAX_BATCH; b++) {
            acc[r][b] = 0.0;
        }
    }

    for (uint kb = lane / LANES; kb < blocks_per_row; kb += kb_step) {
        W wr[ROWS];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            wr[r] = load_w(wrow[r] + uint64_t(kb) * BLOCK_BYTES, slot);
        }
        [[unroll]] for (uint b = 0; b < ARES_MAX_BATCH; b++) {
            if (b >= pc.n_batch) {
                break;
            }
            const A a = load_a(
                pc.aq + uint64_t(b) * pc.n_in + uint64_t(kb) * BLOCK_VALUES,
                ad + 8 * (uint64_t(b) * groups_per_row + uint64_t(kb) * (BLOCK_VALUES / 32)),
                slot);
            [[unroll]] for (uint r = 0; r < ROWS; r++) {
                dot_acc(acc[r][b], wr[r], a);
            }
        }
    }

    // Sum over the lanes of the warp; lane 0 writes.
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        [[unroll]] for (uint b = 0; b < ARES_MAX_BATCH; b++) {
            if (b >= pc.n_batch) {
                break;
            }
            const float total = subgroupClusteredAdd(acc[r][b], 32);
            if (lane == 0 && row0 + r < pc.n_out) {
                FloatRef(pc.y + 4 * (uint64_t(b) * pc.n_out + row0 + r)).v[0] = total;
            }
        }
    }
}
