// The kernels of the `qwen4exp` architecture (Qwen3.8-Flash-Next, the preview
// of Qwen4) that the layer kernels and the attention kernels do not have: the
// stream mixer, the device part of the hashed n-gram embedding, the indexer of
// the sparse attention with its block scores and the selection of the cells a
// row attends to. One fat binary (build.rs), loaded by `Q4Kernels`
// (src/q4.rs). The recurrent layers of the architecture run on layers.cu, its
// attention on attention.cu, its experts on experts.cu.

#include "common.cuh"
#include "layer_step.cuh"
#include "layer_math.cuh"
#include "q4_mixer.cuh"
#include "q4_indexer.cuh"
#include "q4_select.cuh"
#include "q4_ngram.cuh"
