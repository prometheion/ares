// The image tower's module: the row kernels (tower_rows.cuh) and the attention
// (tower_attention.cuh) of the `qwen3vl_merger` projector, loaded by
// `TowerKernels` (src/tower.rs). Compute capability 8.0 and newer (the
// attention's cp.async, ldmatrix and f16 mma); the tower's products are the
// tile kernels of the prefill module, which need the same.

#include "common.cuh"
#include "tower_rows.cuh"
#include "tower_attention.cuh"

// The attention per head size: the head's values and that rounded up to 16.
ARES_TOWER_ATTENTION(ares_tower_attention_d72, 72, 80)
