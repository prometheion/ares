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
#include "q4_fused.cuh"

// The fused read of the stream mixer for float matrices, and for every weight type by its
// type at run time; q4_reads.cu has it for the common quantized types with their decoders
// inlined.
ARES_Q4_READS(, AresQ4Float)
ARES_Q4_READS(_any, AresQ4Any)
