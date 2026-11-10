// The weight kernel for float weights (F32, F16, BF16): the matrix times 1 to 8 rows of
// f32 activations, in f32. There is nothing to quantize: the activations are read as
// they are, the weights are widened to f32 on load, and the products are summed with
// fused multiply-adds, so the result is the exact product up to the rounding of the f32
// sums (not of an int8 quantization). The structure is that of mmvq.glsl (a warp
// computes 2 rows, a workgroup 8; the lanes of a warp take 8 consecutive values each,
// 256 per step; the batch is unrolled to 8 and cut by a uniform branch).
//
// The type file defines `load8(uint64_t address, out float v[8])`, which reads 8 values
// of its type from a 16-byte aligned address, and BYTES_PER_VALUE.
//
//   w   n_out rows of n_in values, 16-byte aligned, n_in a multiple of 8
//   x   n_batch rows of n_in f32 values, 16-byte aligned
//   y   n_batch rows of n_out f32 values

layout(local_size_x = ARES_MMVQ_WARPS * 32) in;

layout(push_constant, std430) uniform Push {
    uint64_t w;
    uint64_t x;
    uint64_t y;
    uint n_in;
    uint n_out;
    uint n_batch;
} pc;

// The number of activation rows when the kernel is built for one (ARES_NB, 1 to 8), else
// 0: the loop over the rows is then cut by a branch on the push constant.
const uint SPEC_BATCH = ARES_NB;
const uint NB = SPEC_BATCH == 0 ? ARES_MAX_BATCH : SPEC_BATCH;

const uint WARPS = ARES_MMVQ_WARPS;
const uint ROWS = ARES_MMVQ_ROWS;

void main() {
    const uint64_t group = ares_group();
    if (group * (WARPS * ROWS) >= pc.n_out) {
        return;
    }
    const uint lane = gl_SubgroupInvocationID & 31u;
    const uint warp = gl_SubgroupID * (gl_SubgroupSize >> 5) + (gl_SubgroupInvocationID >> 5);
    const uint row0 = uint(group) * (WARPS * ROWS) + warp * ROWS;
    const uint chunks = pc.n_in >> 3;
    const uint64_t row_bytes = uint64_t(pc.n_in) * BYTES_PER_VALUE;

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

    for (uint c = lane; c < chunks; c += 32) {
        float wv[ROWS][8];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            float v[8];
            load8(wrow[r] + uint64_t(c) * (8 * BYTES_PER_VALUE), v);
            [[unroll]] for (uint k = 0; k < 8; k++) {
                wv[r][k] = v[k];
            }
        }
        [[unroll]] for (uint b = 0; b < NB; b++) {
            if (SPEC_BATCH == 0 && b >= pc.n_batch) {
                break;
            }
            const uint64_t at = pc.x + 4 * (uint64_t(b) * pc.n_in + uint64_t(c) * 8);
            const vec4 x0 = ldf4(at);
            const vec4 x1 = ldf4(at + 16);
            [[unroll]] for (uint r = 0; r < ROWS; r++) {
                float a = acc[r][b];
                a = fma(wv[r][0], x0.x, a);
                a = fma(wv[r][1], x0.y, a);
                a = fma(wv[r][2], x0.z, a);
                a = fma(wv[r][3], x0.w, a);
                a = fma(wv[r][4], x1.x, a);
                a = fma(wv[r][5], x1.y, a);
                a = fma(wv[r][6], x1.z, a);
                a = fma(wv[r][7], x1.w, a);
                acc[r][b] = a;
            }
        }
    }

    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        [[unroll]] for (uint b = 0; b < NB; b++) {
            if (SPEC_BATCH == 0 && b >= pc.n_batch) {
                break;
            }
            const float total = subgroupClusteredAdd(acc[r][b], 32);
            if (lane == 0 && row0 + r < pc.n_out) {
                FloatRef(pc.y + 4 * (uint64_t(b) * pc.n_out + row0 + r)).v[0] = total;
            }
        }
    }
}
