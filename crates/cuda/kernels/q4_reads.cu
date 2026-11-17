// The fused read of the stream mixer of qwen4exp (q4_fused.cuh) for matrices of the
// quantized types files use for the mixers (Q8_0 in Unsloth's UD-Q4_K_XL, the 32-value
// types a matrix of rows of 320 values can have, the K-quants of 4 to 6 bits): one pair of
// kernels per type of `down` and `up`, named after the type as ggml spells it, each with
// its decoder inlined. Every other weight type takes the pair in q4.cu that decodes by the
// type at run time (`ares_q4_read_down_any`), which is slower (a call per four values) and
// compiles once for all of them: inlined, a pair takes some 12 seconds of nvcc. A module of
// its own (build.rs) so that it compiles beside q4.cu. The Rust side (src/q4.rs) lists the
// same names.

#include "q4_fused.cuh"

ARES_Q4_READS(_q8_0, AresQ4Blocks<8>, AresQ4Float, )
ARES_Q4_READS(_q4_0, AresQ4Blocks<2>, AresQ4Float, )
ARES_Q4_READS(_q4_1, AresQ4Blocks<3>, AresQ4Float, )
ARES_Q4_READS(_q5_0, AresQ4Blocks<6>, AresQ4Float, )
ARES_Q4_READS(_q5_1, AresQ4Blocks<7>, AresQ4Float, )
ARES_Q4_READS(_iq4_nl, AresQ4Blocks<20>, AresQ4Float, )
ARES_Q4_READS(_q4_K, AresQ4Blocks<12>, AresQ4Float, )
ARES_Q4_READS(_q5_K, AresQ4Blocks<13>, AresQ4Float, )
ARES_Q4_READS(_q6_K, AresQ4Blocks<14>, AresQ4Float, )
