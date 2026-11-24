# The `qwen4exp` architecture (Qwen3.8-Flash-Next)

This is the specification of the architecture that the test model
Qwen3.8-Flash-Next uses, written so that the forward pass can be implemented
from this document. The sibling document for the other test architecture is
[qwen35-architecture.md](qwen35-architecture.md).

Written on 2026-11-10 from source and from the model file's header. Nothing
was run on a GPU and no engine was built. What was run, on the CPU only: the
`inspect` example of `ares-gguf` on both files, Strata's
`tools/gguf_reader.py` for the full metadata arrays, a read of a few small
F32 tensors to see their value ranges, and a Python transcription of the
n-gram hash of section 3.4, which reproduces Strata's oracle rows.

## 0. Sources and conventions

| Short | What | Where, state |
| --- | --- | --- |
| **S** | Strata, the reference for this architecture | `~/Services/strata`, commit `504b1433`, source of engine 0.1.59. The engine binary installed on this machine (`engine/strata`) is the release 0.1.48 |
| **F** | The user's llama.cpp fork, the second description | `~/Services/llama.cpp-native`, branch `prod/native`, commit `f8f346786` |
| model | "Swift Flash Next" (Strata's "Swift 1.5"; base model Qwen3.8 Flash Next by its metadata), IQ2_XS mix | `~/Services/Strata-data/models/swift-IQ2_XS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{1,2}-of-00002.gguf`, 63.47 GiB |
| MTP | The draft block, packed by Strata | `~/Services/Strata-data/mtp/mtp-q2_0.gguf`, 847.8 MiB |

Citations name the file and the function, `S:` or `F:` in front, paths
relative to the repository. A statement without a citation is either a plain
reading of the file header or arithmetic done here, and says so where it
matters. "Not verified" means exactly that.

**The rule.** Strata is the reference. Where Strata and the fork differ,
section 5 names the difference and what this document follows.

**Shapes** are in GGUF order, `[ne0, ne1, ...]`, with `ne0` contiguous. A
matrix `W [n_in, n_out]` maps a vector `x` of `n_in` values to `n_out`
values, `y[r] = sum_i W[r][i] * x[i]`; row `r` is stored contiguously and is
quantized in blocks along the row. A stacked expert tensor
`[n_in, n_out, n_expert]` holds expert `e` as the `e`-th contiguous slab of
`n_out` rows.

**Symbols** (values of this model in brackets):

| Symbol | Meaning |
| --- | --- |
| `N` | width of one residual stream, `embedding_length` [2560] |
| `hc` | residual streams, `hyper_connection.count` [4]; `D = hc * N` [10240] |
| `LR` | rank of the stream mixer, `hyper_connection.low_rank` [320] |
| `L` | layers [48] |
| `E`, `K`, `FF` | experts per layer [512], experts per token [10], expert width [640] |
| `H`, `HKV`, `HD` | query heads [24], key/value heads [2], head width [256] |
| `ROT` | rotated dimensions of a head, `rope.dimension_count` [64] |
| `IH`, `ID` | indexer query heads [4], indexer key width [128] |
| `B`, `TOPK` | cells per indexer block [4], selection budget in cells [2048] |
| `S`, `GK`, `GV` | recurrent head width [128], key heads [16], value heads [48] |
| `CG` | recurrent channels, `2*GK*S + GV*S` [10240] |
| `V` | vocabulary [248320] |

**Functions.** `eps = 1e-6` (`attention.layer_norm_rms_epsilon`).

- `rms(x)` over a vector of `n` values: `x / sqrt(mean(x^2) + eps)`
  (F: `ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_rms_norm_f32`).
- `l2(x)`: `x / sqrt(sum(x^2) + eps)`, no division by the width
  (S: `include/strata/kernels/gdn.hpp`, `gdn_l2_norm`; F:
  `src/models/models.h`, `build_gdn_l2_norm`).
- `sigmoid`, `silu(x) = x * sigmoid(x)`, `softplus(x) = x` for `x > 20`,
  else `ln(1 + e^x)` (F: `ggml/src/ggml-cpu/unary-ops.cpp`, `op_softplus`;
  S: `docs/GDN_CONTRACT.md`).

A **cell** is one position of a sequence in the attention state; cell `p` is
the `p`-th token of the sequence. For text the rotary position equals the
cell index; for images it does not (section 3.9).

## 1. The pass at a glance

```
x      = token_embd[token]                          (N)
R[c]   = x  for each stream c                       (hc x N)
for l in 0 .. L-1:
    if l == 1:  R = ple(R, rows(token, two tokens before))        3.4
    m, inj = read(R, hc_attn_*[l])                                3.3
    a      = recurrent(m)  if l % 4 != 3  else  attention(m)      3.5, 3.6
    R      = write(R, a, inj)
    m, inj = read(R, hc_ffn_*[l])
    f      = moe(m)                                               3.7
    R      = write(R, f, inj)
m      = read(R, output_hc_*)        (no inject)                  3.8
logits = output @ m                  (V)
```

S: `include/strata/core/layer.hpp` (the comment above `block_layer`),
`src/core/layer.cpp` `block_layer_pre`, `block_layer_post`; F:
`src/models/qwen4exp.cpp`, `llama_model_qwen4exp::graph::graph`.

There is no layer norm anywhere: the read mixer carries the norm. 36 layers
are recurrent (Gated DeltaNet) and 12 are attention (`l = 3, 7, ..., 47`).
Every layer has the mixture of experts. Layer 1 alone has the hashed n-gram
embedding ("PLE"). The MTP draft block is a 49th block of the attention kind
in a file of its own (section 4).

## 2. The files

### 2.1 Layout

The model is one GGUF version 3 in two parts with 1224 tensors: part 1 holds
the metadata, the six global tensors and the layers up to
`blk.13.attn_qkv.weight` (342 tensors); part 2 holds the rest (882 tensors,
`blk.13.ffn_down_exps.weight` to `blk.47.indexer.q_proj.weight`) and only
the three `split.*` keys. Tensor data starts at offset 10967808 in part 1
and 56320 in part 2 (default alignment 32).

By type: 388 BF16 (1.31 GiB), 388 F32 (250 MiB), 1 F16, 181 IQ4_XS, 68
IQ2_S (17.0 GiB), 53 Q2_0 (10.6 GiB), 44 IQ3_S, 37 IQ4_NL (26.9 GiB, almost
all of it the n-gram table), 29 Q6_K, 22 IQ2_XXS (4.4 GiB), 7 Q8_0, 6 IQ1_M
(1.0 GiB). Q2_0 is ggml type 42 (F: `ggml/include/ggml.h`,
`GGML_TYPE_Q2_0 = 42`): 64 values per block of 18 bytes, an f16 scale `d`
and 2-bit codes, `value = (code - 1) * d` (S: `tools/canonical_xcheck.py`,
`tools/mtp_pack.py` `q2_0`).

### 2.2 Metadata (75 keys)

Architecture keys, prefix `qwen4exp.`:

| Key | Value | What it controls |
| --- | --- | --- |
| `block_count` | 48 | `L` |
| `context_length` | 262144 | trained context |
| `embedding_length` | 2560 | `N` |
| `attention.head_count` | 24 | `H` |
| `attention.head_count_kv` | 2 | `HKV`; query head `a` reads KV head `a / (H/HKV)` |
| `attention.key_length`, `attention.value_length` | 256, 256 | `HD` for K and V |
| `attention.layer_norm_rms_epsilon` | 1e-6 | `eps` of every norm and of `l2` |
| `rope.dimension_count` | 64 | `ROT`: the first 64 of the 256 (attention) or 128 (indexer) values rotate |
| `rope.dimension_sections` | [11, 11, 10, 0] | which position stream each rotated pair reads (section 3.6) |
| `rope.freq_base` | 1e7 | rotary base |
| `full_attention_interval` | 4 | layer `l` is attention when `(l + 1) % 4 == 0` |
| `attention.recurrent_layers` | 48 booleans, `[true, true, true, false] x 12` | the same statement per layer |
| `attention.compress_ratios` | 48 integers, 4 on attention layers, 0 elsewhere | `B`, and which layers have an indexer |
| `attention.indexer.head_count` | 4 | `IH`, indexer QUERY heads; the key has one head |
| `attention.indexer.key_length` | 128 | `ID` |
| `attention.indexer.top_k` | 2048 | `TOPK`, the budget in cells, i.e. 512 blocks |
| `ssm.state_size` | 128 | `S` |
| `ssm.group_count` | 16 | `GK` |
| `ssm.time_step_rank` | 48 | `GV` |
| `ssm.inner_size` | 6144 | `GV * S` |
| `ssm.conv_kernel` | 4 | taps of the recurrent layers' convolution |
| `hyper_connection.count` | 4 | `hc` |
| `hyper_connection.low_rank` | 320 | `LR` |
| `expert_count` | 512 | `E` |
| `expert_used_count` | 10 | `K` |
| `expert_feed_forward_length` | 640 | `FF` of routed experts |
| `expert_shared_feed_forward_length` | 640 | `FF` of the shared expert |
| `ple.layers` | [1] | the layer with the n-gram embedding (0-based) |
| `ple.ngram_size` | 3 | tokens hashed; also the dilation of the PLE convolution |
| `ple.heads_per_ngram` | 8 | table rows per n-gram order; `(3 - 1) * 8 = 16` rows per token |
| `ple.conv_kernel` | 4 | taps of the PLE convolution |
| `ple.eos_token_id` | 248044 | the token that cuts the hash window (`<\|endoftext\|>`, not the chat end token) |
| `ple.image_token_id` | 248056 | the token hashed for image cells (`<\|image_pad\|>`) |
| `embedding_length_per_layer_input` | 160 | values per table row; `16 * 160 = N` |
| `ple.layer_multipliers` | [23703573157769, 20109073645365, 8052911324071] | hash multipliers, one per window slot |
| `ple.head_vocab_sizes` | 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 | rows of each of the 16 heads |
| `ple.head_offsets` | 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 | first table row of each head (running sum of the sizes; the sizes sum to 320001446) |

Who reads them: F reads all of them in `src/models/qwen4exp.cpp`
`load_arch_hparams` (it prefers `attention.recurrent_layers` and falls back
to the interval; it derives `B` from `compress_ratios` and refuses a ratio
of 1 or one that does not divide `top_k`). S reads the dimensions in
`include/strata/artifact/gguf_reader.hpp` `read_geometry` and refuses a
value its kernels are not compiled for (`geometry_unsupported` pins the
layer count, the width, the head counts and widths, the rotated width, the
recurrent dimensions, the indexer's dimensions with `B` and `top_k`, the
mixer's two numbers, `expert_feed_forward_length` and the epsilon; not the
expert counts). It reads the three PLE arrays in `src/program/generate.cpp`
(the block that calls `ple_consts_from_arrays`). S does not read
`attention.recurrent_layers`.

The other keys:

- `general.*` (21): `architecture = "qwen4exp"`, `type`, `name = "Swift
  Flash Next"`, `size_label = "512x56B"`, licence and base-model fields,
  `tags`, `sampling.top_k = 20`, `sampling.top_p = 0.95`, `sampling.temp =
  1` (the model's suggested sampling), `quantization_version = 2`,
  `file_type = 20`.
- `tokenizer.ggml.*` and `tokenizer.chat_template` (9): `model = "gpt2"`
  (byte-level BPE), `pre = "qwen35"`, 248320 `tokens`, `token_type`
  (248044 normal, 27 control, 6 user-defined, 243 unused), 247587 `merges`,
  `bos_token_id = 248044`, `eos_token_id = 248046` (`<|im_end|>`),
  `padding_token_id = 248044`, a chat template of 8952 characters. There is
  no `add_bos_token` key. Special tokens used below: 248044
  `<|endoftext|>`, 248045 `<|im_start|>`, 248046 `<|im_end|>`, 248053 and
  248054 vision start and end, 248056 `<|image_pad|>`, 248068 and 248069
  `<think>` and `</think>`.
- `quantize.imatrix.*` (4) and `split.*` (3): provenance and the split.

### 2.3 Tensors (1224)

Global (6):

| Name | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `token_embd.weight` | [2560, 248320] | IQ4_XS | token embedding, one row of `N` per token (1360 bytes) |
| `output.weight` | [2560, 248320] | IQ4_XS | output head |
| `output_hc_norm.weight` | [10240] | F32 | final mixer: norm weights |
| `output_hc_down.weight` | [10240, 320] | BF16 | final mixer: down |
| `output_hc_up.weight` | [320, 10240] | BF16 | final mixer: up |
| `per_layer_token_embd.weight` | [160, 320001536] | IQ4_NL | n-gram table, one row of 160 values in 90 bytes; 26.82 GiB |

Every layer `blk.<l>.` (16 tensors, 48 layers):

| Suffix | Shape | Types in this file | Meaning |
| --- | --- | --- | --- |
| `hc_attn_norm.weight`, `hc_ffn_norm.weight` | [10240] | F32 | mixer norm weights, stream `c` at `[c*N, (c+1)*N)` |
| `hc_attn_down.weight`, `hc_ffn_down.weight` | [10240, 320] | BF16 | mixer down |
| `hc_attn_up.weight`, `hc_ffn_up.weight` | [320, 10240] | BF16 | mixer up |
| `hc_attn_inject.weight`, `hc_ffn_inject.weight` | [10240, 4] | BF16 | write-back weights per stream |
| `ffn_gate_inp.weight` | [2560, 512] | F32 | router |
| `ffn_gate_exps.weight` | [2560, 640, 512] | IQ2_S x34, IQ2_XXS x11, IQ1_M x3 | routed experts: gate |
| `ffn_up_exps.weight` | [2560, 640, 512] | same type as the layer's gate | routed experts: up |
| `ffn_down_exps.weight` | [640, 2560, 512] | Q2_0 x48 | routed experts: down |
| `ffn_gate_inp_shexp.weight` | [2560] | F32 | scalar gate of the shared expert |
| `ffn_gate_shexp.weight` | [2560, 640] | IQ4_XS x31, IQ3_S x9, Q6_K x8 | shared expert: gate |
| `ffn_up_shexp.weight` | [2560, 640] | IQ4_XS x26, IQ3_S x14, Q6_K x8 | shared expert: up |
| `ffn_down_shexp.weight` | [640, 2560] | IQ4_NL x36, Q8_0 x7, Q2_0 x5 | shared expert: down |

Gate and up of a layer always have the same type: IQ1_M in layers 8, 13,
37; IQ2_XXS in 1, 4, 9, 10, 11, 14, 18, 19, 27, 29, 30; IQ2_S in the other
34. One expert is 524800 + 524800 + 460800 bytes with IQ2_S (1.44 MiB),
422400 per matrix with IQ2_XXS, 358400 with IQ1_M (arithmetic from the block
sizes: 82, 66 and 56 bytes per 256 values, 18 bytes per 64 for Q2_0).

Recurrent layers (9 more tensors, 36 layers, `l % 4 != 3`):

| Suffix | Shape | Types | Meaning |
| --- | --- | --- | --- |
| `attn_qkv.weight` | [2560, 10240] | IQ4_XS x35, IQ3_S x1 | rows: q `[0, 2048)`, k `[2048, 4096)`, v `[4096, 10240)`, heads of 128 |
| `attn_gate.weight` | [2560, 6144] | IQ4_XS x22, IQ3_S x11, Q6_K x3 | output gate `z`, 48 heads of 128 |
| `ssm_conv1d.weight` | [4, 10240] | F32 | convolution, tap `i` of channel `ch` at `i + 4*ch` |
| `ssm_alpha.weight`, `ssm_beta.weight` | [2560, 48] | BF16 | decay and write strength per value head |
| `ssm_dt.bias` | [48] | F32 | bias of the decay |
| `ssm_a` | [48] | F32 | decay base, negative (here -158 to -0.028) |
| `ssm_norm.weight` | [128] | F32 | output norm weights, shared by all heads |
| `ssm_out.weight` | [6144, 2560] | IQ4_XS x29, IQ3_S x6, Q6_K x1 | output projection |

Attention layers (10 more tensors, 12 layers, `l % 4 == 3`):

| Suffix | Shape | Types | Meaning |
| --- | --- | --- | --- |
| `attn_q.weight` | [2560, 12288] | IQ4_XS x11, IQ3_S x1 | per head 512 rows: query `[0, 256)`, output gate `[256, 512)` |
| `attn_k.weight` | [2560, 512] | IQ4_XS x8, Q6_K x3, IQ3_S x1 | 2 key heads of 256 |
| `attn_v.weight` | [2560, 512] | IQ4_XS x7, Q6_K x5 | 2 value heads of 256 |
| `attn_output.weight` | [6144, 2560] | IQ4_XS x10, IQ3_S x1, Q6_K x1 | output projection, input head-major |
| `attn_q_norm.weight`, `attn_k_norm.weight` | [256] | F32 | norm weights of query and key heads |
| `indexer.q_proj.weight` | [2560, 512] | BF16 | 4 indexer query heads of 128 |
| `indexer.k_proj.weight` | [2560, 128] | BF16 | the one indexer key head |
| `indexer.q_norm.weight`, `indexer.k_norm.weight` | [128] | F32 | norm weights of indexer query and pooled key |

Layer 1 only (6 tensors):

| Suffix | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `ple_key.weight` | [2560, 10240] | BF16 | gathered rows to a key per stream |
| `ple_value.weight` | [2560, 2560] | BF16 | gathered rows to a value |
| `ple_norm_key.weight`, `ple_norm_query.weight`, `ple_norm_conv.weight` | [10240] | F32 | norm weights per stream |
| `ple_conv1d.weight` | [4, 10240] | F16 | convolution, tap `k` of channel `ch` at `k + 4*ch` |

Count: `6 + 48*16 + 36*9 + 12*10 + 6 = 1224`. The names and shapes agree
with S: `include/strata/core/family_layout.hpp` (`layer_tensors`,
`global_tensors`; the PLE tensors are checked by the loaders, not by that
table) and F: `src/models/qwen4exp.cpp` `load_arch_tensors`.

### 2.4 What the converter did to the weights

These facts decide how the tensors are used; they come from the converter F:
`conversion/qwen4exp.py` and `conversion/qwen.py`.

- **Norm weights hold `1 + w`.** The checkpoint's norms scale by `1 + w`;
  the converter stores `w + 1` for the norm weights (`Qwen3NextModel.
  modify_tensors`, and `Qwen4ExpTextModel.modify_tensors` for the PLE and
  indexer norms). `ssm_norm.weight` is the exception: it is stored as it is.
  Either way the pass multiplies by the stored value and adds nothing
  (S: `include/strata/kernels/gr.hpp`, comment on `gr_read`;
  `include/strata/kernels/ple.hpp`, comment on `PleWeights`). Which
  converter wrote this file is not recorded in it; the value ranges read
  from it fit stored `1 + w`: `indexer.*_norm` mean 0.96, the PLE norms
  mean 0.89, `ssm_norm` mean 0.97, `attn_q_norm` mean 1.28, where the MTP
  file's raw indexer norms have mean 0.08 and 0.10.
- **`ssm_a = -exp(A_log)`**, `ssm_dt.bias` is the checkpoint's `dt_bias`,
  the convolution kernels are squeezed to two dimensions.
- **Value heads are reordered** from the checkpoint's grouped order to tiled
  order (`_LinearAttentionVReorderBase`), in the v rows of `attn_qkv`, in
  `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, the v
  channels of `ssm_conv1d` and the columns of `ssm_out`. In the file, value
  head `h` belongs to key head `h % GK`. A comparison with the original
  checkpoint or another runtime needs that permutation.
- **The indexer's q and k projections** come from one checkpoint tensor
  `index_qk_proj`, the first `IH * ID` rows the query, the rest the key.
- **The n-gram table** is the concatenation of the checkpoint's table
  shards in index order. It has 320001536 rows; the 16 heads address
  320001446 of them. The checkpoint's own table is FP8 with one scale (S:
  `include/strata/kernels/ngram.hpp`, comment at `fp8_e4m3_dequant_row`);
  this file re-quantized it to IQ4_NL.

### 2.5 The MTP file

`general.architecture = "qwen4exp-mtp"`, four `strata.mtp.*` provenance keys
(source "Qwen/Qwen3.8-Flash-Next BF16 checkpoint, mtp.* tensors",
`expert_format = "q2_0"`), 31 tensors with the checkpoint's own names
(prefix `mtp.`, layer part `layers.0.`):

| Name (without `mtp.`) | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `pre_fc_norm_embedding.weight` | [2560] | F32 | norm of the token embedding |
| `pre_fc_norm_hidden.weight` | [10240] | F32 | norm of the trunk's residual |
| `fc_embedding.weight`, `fc_hidden.weight` | [2560, 2560] | BF16 | input projections |
| `layers.0.attn_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight}` | [10240], [10240, 320], [320, 10240], [10240, 4] | F32, BF16 x3 | as `hc_attn_{norm, down, up, inject}` |
| `layers.0.mlp_hyper_connection.{...}` | the same | the same | as `hc_ffn_*` |
| `layers.0.self_attn.{q_proj, k_proj, v_proj, o_proj}.weight` | [2560, 12288], [2560, 512], [2560, 512], [6144, 2560] | BF16 | as `attn_{q, k, v, output}` |
| `layers.0.self_attn.{q_norm, k_norm}.weight` | [256] | F32 | as `attn_{q, k}_norm` |
| `layers.0.self_attn.indexer.index_qk_proj.weight` | [2560, 640] | BF16 | indexer query (rows 0..511) and key (rows 512..639) in one tensor |
| `layers.0.self_attn.indexer.{q_layernorm, k_layernorm}.weight` | [128] | F32 | as `indexer.{q, k}_norm` |
| `layers.0.mlp.gate.weight` | [2560, 512] | BF16 | router |
| `layers.0.mlp.experts.gate_up_proj` | [2560, 1280, 512] | Q2_0 | per expert rows 0..639 gate, 640..1279 up |
| `layers.0.mlp.experts.down_proj` | [640, 2560, 512] | Q2_0 | per expert down |
| `layers.0.mlp.shared_expert.{gate_proj, up_proj, down_proj}.weight` | [2560, 640] x2, [640, 2560] | BF16 | shared expert |
| `layers.0.mlp.shared_expert_gate.weight` | [2560, 1] | BF16 | scalar gate of the shared expert |
| `hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}` | [10240], [10240, 320], [320, 10240] | F32, BF16 x2 | the block's own final mixer |

S: `tools/mtp_pack.py` (the writer), `tools/mtp_rt.py` (the reader that
makes Strata's run-time files).

Two differences from the main file must be handled by the loader:

- **The nine F32 norm weights are raw** (the checkpoint's `w`): the pass
  multiplies by `1 + w` (S: `tools/mtp_rt.py`, the branch for
  `tensor_type == 0`). The value ranges fit: the indexer norms have mean
  0.08 and 0.10 here against 0.96 in the main file.
- Gate and up of the routed experts are one tensor, and the indexer's two
  projections are one tensor.

The value heads' reordering of 2.4 does not apply: the block has no
recurrent layer.

The fork cannot read this file: it knows the architecture name `qwen4exp`
only (F: `src/llama-arch.cpp`) and expects the block as `blk.48.*` inside
the model file, with `nextn.eh_proj` (the two `fc` matrices side by side),
`nextn.enorm`, `nextn.hnorm`, `nextn.hc_head_{norm, down, up}` (F:
`src/models/qwen4exp.cpp` `load_arch_tensors`; `conversion/qwen4exp.py`
`modify_tensors`).

The draft head's token list is not in either GGUF: Strata reads
`~/Services/Strata-data/mtp/rt/draft_vocab.bin`, 106299
token ids as little-endian int32, not sorted (S: `tools/draft_vocab.py`).

## 3. The forward pass

One row is one token of one sequence at one cell. All formulas are per row;
rows of different sequences never interact, and a row reads only its own
sequence's state.

### 3.1 Embedding

`x = token_embd[token]` (the `token`-th row, `N` values, no scaling), and
`R[c] = x` for the `hc` streams (S: `include/strata/core/layer.hpp`
`embed_row`; F: `graph::graph`, nodes `model.input_embed` and `hc_init`).
An image cell takes its `N` values from the vision projector instead.

### 3.2 Layer order

Section 1. The n-gram embedding is applied to `R` at the start of layer 1,
before that layer's first read (S: `src/core/layer.cpp` `block_layer_pre`;
F: `graph::graph`).

### 3.3 The stream mixer ("hyper-connection")

`read(R, norm, down, up, inject) -> (m, inj)`:

```
xn[c]  = rms(R[c]) * norm[c]                  per stream, rms over N values     (hc x N)
lo     = silu((down @ xn) / hc)               xn taken as one vector of D       (LR)
gate   = up @ lo                                                                (D)
m      = (1/hc) * sum_c xn[c] * sigmoid(gate[c])                                (N)
inj    = inject @ xn                                                            (hc)
```

The division by `hc` is inside the `silu`. The final mixer before the
output head has no `inject`.

`write(R, out, inj)`: `R[c] += out * 2 * sigmoid(inj[c] / hc)` for every
stream; the block output is the same for all streams, only the factor
differs.

S: `include/strata/kernels/gr.hpp` (`gr_read`, `gr_write`); F:
`src/models/qwen4exp.cpp` `build_hc_mix`, `build_hc_combine`. The two agree.

### 3.4 The hashed n-gram embedding (layer 1)

**Row ids, on the host.** For the token `t0` at a cell, with `t1` the token
one cell before and `t2` two cells before:

```
ctx[0] = t0
cut = false
for s in 1, 2:
    cut = cut or (t_s is missing) or (t_s == ple.eos_token_id)
    ctx[s] = cut ? ple.eos_token_id : t_s
mixed_2 = (ctx[0]*m[0]) xor (ctx[1]*m[1])                        unsigned 64-bit, wrapping
mixed_3 = mixed_2 xor (ctx[2]*m[2])
row[h]  = mixed_2 mod vocab[h] + offset[h]     for h = 0 .. 7
row[h]  = mixed_3 mod vocab[h] + offset[h]     for h = 8 .. 15
```

`m`, `vocab`, `offset` are the three `ple.*` arrays. A predecessor before
the start of the sequence is "missing". An end-of-text token in the window
replaces itself and everything older by end-of-text; the token's own value
never cuts. An image cell hashes as `ple.image_token_id`. S:
`src/kernels/ngram.cpp` `ngram_rows`; F: `src/models/qwen4exp.cpp`
`llm_graph_input_qwen4exp_ple::set_input`. The two are the same code in
effect.

Test vectors (S: `src/kernels/ple_oracle_vectors.inc`; reproduced here by a
Python transcription of the lines above):

- token 11, no predecessors: 13555483, 24877159, 54278900, 63631058,
  98394898, 117447587, 136774765, 157274238, 173487882, 192109366,
  213364329, 236577563, 258758149, 262164274, 283572184, 312653920.
- token 44 after 22 (one back) and 33 (two back): 12721672, 27113445,
  44309803, 60385209, 97021817, 115900810, 134219364, 151977589, 161183764,
  181257609, 201344488, 221596875, 241635416, 261674107, 281687144,
  301751935.

**Gather.** `emb` (`N` values) is the 16 rows side by side, head `h` at
`[h*160, (h+1)*160)`. A row is 5 IQ4_NL blocks of 18 bytes at file offset
`tensor offset + row * 90`: an f16 scale `d`, then 16 bytes where byte `j`
holds element `j` in its low nibble and element `j + 16` in its high one;
`value = d * table[nibble]` with the table -127, -104, -83, -65, -49, -35,
-22, -10, 1, 13, 25, 38, 53, 69, 89, 113 (S: `src/kernels/ngram.cpp`
`iq4nl_dequant_row`; F: `build_inp_ple`).

**The block.**

```
key    = ple_key @ emb                                           (D)
value  = ple_value @ emb                                         (N)
kn[c]  = rms(key[c]) * ple_norm_key[c]                           per stream
qn[c]  = rms(R[c])   * ple_norm_query[c]
s[c]   = (kn[c] . qn[c]) / sqrt(N)
g[c]   = sigmoid(sign(s[c]) * sqrt(max(|s[c]|, 1e-6)))
gated[c] = value * g[c]                                          (hc x N)
nrm    = rms(gated[c]) * ple_norm_conv[c], as one vector         (D)
conv[ch] = sum_{k=0..3} w[k][ch] * nrm_at(cell - (3-k)*3)[ch]    (D)
R[c]   = R[c] + gated[c] + silu(conv[c])
```

`nrm_at(q)` is the `nrm` vector of cell `q` of the same sequence, zero
before the sequence's start. The convolution is causal, depthwise, with
dilation `ple.ngram_size`: it reads this cell and the cells 3, 6 and 9
back. The state a sequence carries is the `nrm` vectors of its last 9
cells. S: `include/strata/kernels/ple.hpp` (file comment, `ple_block`),
`src/core/layer.cpp` `block_layer_pre`; F: `build_ple`,
`build_conv_state_at`. The two agree.

### 3.5 The recurrent layer (Gated DeltaNet)

```
u     = attn_qkv @ m                                   (CG)
z     = attn_gate @ m                                  (GV x S)
alpha = ssm_alpha @ m                                  (GV)
beta  = sigmoid(ssm_beta @ m)                          (GV)
g     = softplus(alpha + ssm_dt.bias) * ssm_a          (GV), <= 0
c[ch] = sum_{i=0..3} kw[ch][i] * u_at(cell - 3 + i)[ch]          4 taps, tap 3 is this cell
h     = silu(c)                                        the whole vector, before the split
q[j]  = l2(h[j*S ...])           j = 0 .. GK-1         key heads
k[j]  = l2(h[GK*S + j*S ...])
v[n]  = h[2*GK*S + n*S ...]      n = 0 .. GV-1         value heads, not normalised
```

`u_at(q)` is the `u` vector of cell `q`, zero before the start. Then per
value head `n`, with key head `j = n % GK` and the head's state `M[n]`, an
`S x S` matrix indexed `[i][o]`:

```
M[n]     = M[n] * exp(g[n])
sk[o]    = sum_i M[n][i][o] * k[j][i]
d[o]     = (v[n][o] - sk[o]) * beta[n]
M[n][i][o] += k[j][i] * d[o]
y0[n][o] = sum_i M[n][i][o] * q[j][i] / sqrt(S)        reads the updated state
```

and the output

```
y[n]  = rms(y0[n]) * ssm_norm * sigmoid(z[n])          rms per head over S values
out   = ssm_out @ y                                    y as one vector of GV*S     (N)
```

The decay is applied before the rank-one update. The output gate is a
sigmoid (Qwen3.5 used SiLU there). The state a sequence carries per
recurrent layer is `M` (`GV * S * S` values) and the `u` vectors of its last
3 cells.

S: `include/strata/kernels/gdn.hpp` (`gdn_step`, `gdn_conv_step`,
`gdn_l2_norm`, `gdn_out_norm`), `src/core/layer.cpp` `gdn_layer`,
`src/kernels/cuda/elementwise.cu` `gdn_gate_kernel`, `docs/GDN_CONTRACT.md`;
F: `src/models/qwen4exp.cpp` `build_layer_attn_linear`, `build_qkvz`,
`build_norm_gated`, `src/models/delta-net-base.cpp`
`build_delta_net_autoregressive`. The two agree. (S stores the state as
`[i][n][o]`, a choice of memory layout only.)

For many consecutive rows of one sequence the recurrence is the same steps
in order; S also has a chunked form that solves 64 cells at once and
differs in the last bits (S: `include/strata/prefill/gdn_chunk.hpp`).

### 3.6 The attention layer (sparse attention through an indexer)

**Projections.**

```
qg    = attn_q @ m          per head a: q[a] = qg[a*512 .. +256), gate[a] = qg[a*512+256 .. +256)
kk    = attn_k @ m          key heads k[b], b = 0, 1                              (HKV x HD)
vv    = attn_v @ m          value heads v[b]
q[a]  = rope(rms(q[a]) * attn_q_norm, pos)        rms per head over HD values
k[b]  = rope(rms(k[b]) * attn_k_norm, pos)
```

`v` gets neither norm nor rotation. The cell's `k` and `v` are stored
(what is stored is the key after norm and rotation) before the row attends,
so a row can attend to itself.

**Rotation.** `rope(x, pos)` rotates the first `ROT = 64` values of `x` and
leaves the rest. Pair `i` (0..31) is `(x[i], x[i + 32])`:

```
theta = P(i) * base^(-2*i / ROT)
x[i], x[i+32] = x[i]*cos(theta) - x[i+32]*sin(theta),  x[i]*sin(theta) + x[i+32]*cos(theta)
```

`pos` is three numbers `(t, h, w)` and `P(i)` is `t` for `i % 3 == 0`, `h`
for `i % 3 == 1`, `w` for `i % 3 == 2`; this is what the sections
`[11, 11, 10, 0]` mean for the 32 pairs. For text `t = h = w =` the cell
index. S: `src/kernels/cuda/rope.cu` `build_rope_table` (the table is built
in float64), `include/strata/kernels/rope.hpp` `rope_neox_pair`,
`include/strata/kernels/mrope.hpp` `mrope_pos`; F:
`ggml/src/ggml-cuda/rope.cu` `rope_multi` with `is_imrope`, selected by
`src/llama-model.cpp` (`LLAMA_ROPE_TYPE_IMROPE`). The two agree.

**The indexer.**

```
r     = indexer.k_proj @ m                                   raw key of the cell     (ID)
u[e]  = rope(rms(iq[e]) * indexer.q_norm, pos)               iq = indexer.q_proj @ m, heads e = 0..3
```

The raw key is never normed or rotated on its own. Cells are grouped into
blocks of `B = 4` in sequence order: block `j` is cells `4j .. 4j+3`. When
the last cell of block `j` has been processed the block gets one pooled
key:

```
P[j]  = rope(rms(mean(r of cells 4j .. 4j+3)) * indexer.k_norm, pos of cell 4j)
```

A row at cell `p` (so `n = p + 1` cells exist, its own included) sees the
complete blocks `j < floor(n / 4)`, which includes the block the row itself
completes when `n % 4 == 0`, and its **tail**, the `n % 4` newest cells
`4*floor(n/4) .. p`. It scores each visible block

```
score[j] = sum_{e=0..3} max(0, u[e] . P[j])
```

with the rectifier per head, before the sum over heads. There is no softmax
in the indexer.

**Selection.** If `n <= TOPK + B - 1 = 2051` the row attends to all `n`
cells. Otherwise, as this document specifies it (section 5, difference 1):

- the tail cells are always selected;
- the remaining budget `2051 - (n % 4)` cells is filled from the blocks in
  the order of their score, highest first, equal scores by lower block
  index first; a block gives its 4 cells, and the last block taken gives
  only its first cells if the budget ends inside it.

So the row attends to exactly 2051 cells: with a tail of 3 cells that is
512 whole blocks; with a tail of 0, 1 or 2 cells it is 512 whole blocks and
the first 3, 2 or 1 cells of the block ranked 513th. When `n % 4 == 0` the
row's own cell is in an ordinary block and is attended only if that block
is selected.

**Attention** over the selected cells `Sel`, for query head `a` with KV
head `b = floor(a / (H / HKV))` (heads 0..11 read KV head 0, 12..23 read KV
head 1):

```
w[c]   = softmax over c in Sel of (q[a] . k_c[b]) / sqrt(HD)
att[a] = sum_{c in Sel} w[c] * v_c[b]                                  (HD)
att[a] = att[a] * sigmoid(gate[a])
out    = attn_output @ att            att as one vector of H*HD, head-major      (N)
```

S: `src/core/layer.cpp` `qsa_layer` (the order of operations, steps 1 to
9), `include/strata/kernels/qsa.hpp` (the layouts and
`qsa_selection_width`), `src/kernels/cuda/qsa_select.cu`
`block_scores_kernel` and `block_topk_kernel`,
`src/kernels/cuda/qsa_decode_attn.cu` `attn_chunk_body`; F:
`src/models/qwen4exp.cpp` `build_layer_attn`, `build_qsa_sel`,
`build_attn_qsa`, `src/llama-memory-hybrid-idx.cpp` `set_input_kpool`
(which blocks and which tail a token sees).

In a step that holds several rows of one sequence, row `i` sees the cells,
raw keys and pooled keys of the rows before it in the step; the visibility
rule by `n` is what bounds each row (S: `src/core/verify.cpp`
`record_window`; `src/prefill/prefill.cpp`, the block marked `QSA`).

### 3.7 The mixture of experts

```
logit = ffn_gate_inp @ m                                   (E)
prob  = softmax(logit)                                     over all E
ids   = the K experts with the largest prob                equal values: lower index first
wt[j] = prob[ids[j]] / max(sum_j prob[ids[j]], 2^-14)
f(e)  = down_e @ (silu(gate_e @ m) * (up_e @ m))           inner width FF          (N)
sh    = sigmoid(ffn_gate_inp_shexp . m)                    one scalar
out   = sum_j wt[j] * f(ids[j])  +  sh * (down_sh @ (silu(gate_sh @ m) * (up_sh @ m)))
```

No bias, no group selection, no clamp, no scale on the weights. The shared
expert is added, not weighted against the routed ones. Router, routed
experts and shared expert all read the same `m`.

S: `src/core/layer.cpp` `moe_route`, `moe_shared`, `moe_combine_parts`,
`include/strata/kernels/router_top10.hpp`,
`include/strata/kernels/shared_expert.hpp`; F: `src/models/qwen4exp.cpp`
`build_layer_ffn`, `src/llama-graph.cpp` `build_moe_ffn` (softmax gating,
`norm_w = true`, clamp `6.103515625e-5`; `expert_weights_scale` is 0 for
this architecture and no swiglu clamp is loaded). The two agree; only S
states the order of equal values.

### 3.8 The output head

`m = read(R, output_hc_norm, output_hc_down, output_hc_up)` without
inject, `logits = output @ m`, 248320 values (S:
`include/strata/core/layer.hpp` `lm_head`; F: `graph::graph`, nodes
`result_norm` and `result_output`).

### 3.9 Images

Not specified here beyond what the text pass must leave room for (S:
`include/strata/kernels/mrope.hpp`; F:
`llm_graph_input_qwen4exp_ple::set_input`): an image cell has its embedding
from the vision projector (a separate file, on this machine
`Strata-data/models/mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf`; the
converter calls the vision tower an unmodified Qwen3-VL one, F:
`conversion/qwen4exp.py` `Qwen4ExpVisionModel`), hashes as
`ple.image_token_id`, and has a rotary position
`(t, h, w) = (p, p + y, p + x)` for grid cell `(x, y)` of an image that
starts at rotary position `p`; the text after an image continues at
`p + max(nx, ny)`. Blocks of the indexer stay groups of 4 cells in sequence
order, and a pooled key is rotated to the position triple of its first cell
(F: `set_input_kpool`, `new_pool_pos`).

## 4. The MTP draft block

One block of the attention kind that predicts one token further than the
trunk. Cell `i` of the block pairs the trunk's residual after the last
layer at cell `i`, `Rt` (`hc x N`, taken before the final mixer), with the
token at cell `i + 1`, `tok`; its output is a guess for the token at cell
`i + 2`.

```
e     = fc_embedding @ (rms(token_embd[tok]) * (1 + pre_fc_norm_embedding))           (N)
hn[c] = rms(Rt[c]) * (1 + pre_fc_norm_hidden[c])       every stream by itself         (hc x N)
h[c]  = fc_hidden @ hn[c]                                                             (hc x N)
R[c]  = h[c] + e
m, inj = read(R, attn_hyper_connection)
k, v  = k_proj @ m, v_proj @ m;  k[b] = rope(rms(k[b]) * (1 + k_norm), i);  store at cell i of the block's own K/V
q, gate from q_proj @ m as in 3.6;  q[a] = rope(rms(q[a]) * (1 + q_norm), i)
att   = attention of 3.6 over the block's own cells 0 .. i, with the block's own indexer
R     = write(R, o_proj @ (att * sigmoid(gate)), inj)
m, inj = read(R, mlp_hyper_connection)
R     = write(R, moe(m), inj)                         3.7 with the block's router, experts and shared expert
m     = read(R, hyper_connection_mixer)               the block's own final mixer, no inject
logits = rows of the trunk's output.weight @ m        all rows, or the rows of the draft token list
```

The draft is the largest logit; its probability under the softmax of the
computed logits is what a caller thresholds. A chain continues at cell
`i + 1` with `Rt` replaced by this step's `R` (after the second write) and
`tok` replaced by the draft. Every norm weight of the MTP file takes the
`1 +` (2.5); the mixers' `hc_norm` too.

The trunk's cells that were accepted must each be put through the block's
first half (up to the store of `k` and `v`), so that the block's K/V covers
every cell before a chain starts; a prompt does this for all its cells.

S: `include/strata/core/mtp.hpp` (file comment), `src/core/mtp.cpp`
`record_front`, `record_rest`, `src/kernels/cuda/verify_kernels.cu`
`window_ids_kernel`; F: `src/models/qwen4exp.cpp` `graph_mtp`,
`common/speculative.cpp` `common_speculative_impl_draft_mtp::process`.
Section 5 lists three places where the two differ (8, 9, 10).

**The draft token list.** Strata computes the block's logits over a subset
of the vocabulary: the rows of `output.weight` for the ids in
`draft_vocab.bin` (106299 of 248320 here), and maps the pick back to a
token id (S: `src/core/mtp.cpp` `record_rest`, `map_ids`). A token outside
the list can never be drafted. The trunk's head is never restricted.

## 5. Where Strata and the fork differ

| # | Subject | Strata | Fork | This document |
| --- | --- | --- | --- | --- |
| 1 | Selection past 2051 cells | A budget of 2051 CELLS: the tail, then cells of the best blocks, the last block possibly in part (S: `qsa_select.cu` `block_topk_kernel`, the `eq_budget` loop; `qsa.hpp` `qsa_selection_width`) | The best `min(visible, 512)` whole BLOCKS plus the tail, so `2048 + n % 4` cells (F: `build_qsa_sel`, `n_top_pool`; `set_input_kpool`) | Strata's rule, in 3.6. See below |
| 2 | Scale of the block score | none in the main engine | times `1/sqrt(ID)` (F: `build_qsa_sel`, `weights`) | none; a positive factor does not change the order |
| 3 | Equal scores | lower index first, for blocks and for experts | not specified (`ggml_top_k` "in no particular order", `ggml_argsort_top_k`) | Strata's |
| 4 | Precision of indexer keys | pooled keys and the raw tail in f32 (S: `qsa.hpp`, "INDEXER KEYS: FP32") | raw key and pooled key per cell in the indexer cache, in the cache's type (F: `build_qsa_sel`, `packed`) | f32 as the definition; the format in pages is the engine's choice and can flip near-ties |
| 5 | How attention is restricted | a list of cell ids read through the page table | a mask of minus infinity over all cells, then dense attention (F: `build_qsa_sel`, `build_attn_qsa`) | the list; same values |
| 6 | K/V storage | f16, int8 with an f16 scale per 64 values, 4-bit after a 256-point Hadamard rotation, or K 8-bit and V 4-bit (S: `include/strata/core/layer.hpp` `QsaState`) | the KV cache's type, optionally rotated (F: `build_attn_qsa`, `self_k_rot`) | the engine's own formats |
| 7 | Undoing rejected drafts | nothing recurrent is written before acceptance; a commit replays the accepted rows (S: `include/strata/kernels/verify_kernels.hpp`) | snapshots of the recurrent state for the last `n_rs_seq` tokens (F: `delta-net-base.cpp` `build_recurrent_attn`, `qwen4exp.cpp` `build_conv_state_at`) | Strata's, which is also what Ares plans |
| 8 | MTP: norm of the trunk residual | one `rms` over all `D` values (default; `--mtp-hnorm stream` gives the other) (S: `mtp.cpp` `record_front`) | `rms` per stream (F: `graph_mtp`, `mtp_hnorm`) | the fork's, `rms` per stream, as the mixers norm the streams. Measured: see below |
| 9 | MTP: what its attention reads | all of its last 32768 cells, no indexer; the block's indexer tensors are unused (S: `mtp.hpp`, `record_rest`) | the block's own indexer and selection when the context has an indexer memory (F: `graph_mtp`, `src/llama-model.cpp` `filter_idx` for the MTP context) | the fork's: the block is an attention layer like the trunk's, with its own indexer. Same result as Strata's up to 2051 cells |
| 10 | MTP: rotary position of cell `i` | `i` (S: `mtp.hpp`) | `i + 1`, the position of the paired token (F: `speculative.cpp` `process`) | Strata's. A shift of all cells by one leaves `q . k` unchanged up to rounding |
| 11 | MTP file | separate file, checkpoint names, raw norm weights, fused tensors | `blk.48.*` in the model file, converted | this machine has Strata's file only |
| 12 | Which layers are recurrent | from `full_attention_interval`, checked against `compress_ratios` | from `attention.recurrent_layers`, else the interval | either; they agree for this file |
| 13 | Shapes | compiled in, a file of another shape is refused | read from the metadata | from the metadata |
| 14 | Draft token list | always the subset when the file exists | a subset that falls back to the whole vocabulary for a sequence whose recent text is often outside it (F: `speculative.cpp`, `vocab_in`, `seq_miss`) | not part of the architecture |

**On difference 8.** The block's file has no outputs to compare with, so
the two norms were measured by what the block is for (2026-11-15, the CPU
reference, `tests/qwen4exp_real.rs`; the 99 tokens of the English trace of
section 8, of which the model wrote 80): fed the trunk's residual of a cell
and the token behind it, the block's largest logit was the token the trunk
wrote one further in 63 of 80 cells with a norm per stream and in 58 of 80
with one norm over all streams; chains of three links from ten cells were
accepted 9, 6 and 3 times of 10, 9 and 6 (per stream) and 9, 7 and 4 times
of 10, 9 and 7 (one norm). Strata's notes give 0.60 and 0.63 for its two
settings on its own measure (S: `docs/DGX_SPARK_LINUX.md`), and the header
of its drafter 0.89, 0.86 and 0.85 for the three links of its probe (S:
`mtp.hpp`), which was not available here.

**On difference 1.** It is the only difference that changes the trunk's
numbers, by at most 3 cells of the 513th-ranked block and only past 2051
cells. Strata is not at one with itself here: its second engine for this
model, the all-cards split, keeps "the QSA_TOP highest" blocks and the
tail, as the fork does (S: `src/split/src/qsa.cuh`, file comment and the
comment above the select kernel). The fork's comment calls whole blocks
plus tail the reference's behaviour (F: `load_arch_hparams`, "the reference
groups the visible tokens in cache order and always keeps the tail").
Strata's main engine says it transcribed its rule from the llama.cpp graph
of its time: per-cell scores with a bias of 1e9 on the tail cells "matching
llama.cpp's causal decode input", then "the `width` largest cells" (S:
`qsa.hpp`, comments on `qsa_index`, `topk_512` and `qsa_selection_width`);
that older graph was not read here. The original implementation was not
available to settle it. This document follows the main engine because it is
the rule's reference and because its outputs are what an implementation can
be compared with on this machine (section 8); the budget rule should be one
parameter of the selection kernel, since the other rule is the same kernel
with a budget of `2048 + n % 4`.

Everything else in sections 3.1 to 3.8 is the same in both.

**Not an architecture difference, but visible in numbers:** Strata
multiplies quantized weights with activations rounded to 8 bits per value
in blocks of 32 on its token and window paths, and with f16 activations and
weights dequantized to f16 on its prompt path (S: `src/core/layer.cpp`
`gdn_layer` and `qsa_layer`, the `quantize_q8_*` calls;
`include/strata/core/layer.hpp`, "THE ACTIVATIONS FOLLOW THE CONTRACT";
`include/strata/prefill/prefill.hpp`, file comment). Its outputs therefore
depend on which path a row took. Strata describes its window path as
matching llama.cpp's CUDA arithmetic for single tokens (S:
`include/strata/kernels/gr.hpp`, "the pinned llama.cpp CUDA single-token
BF16 MMVF input precision"); the fork's kernels were not read for this.
An exact match with either engine is not defined by the architecture.

## 6. For the engine: the state of a sequence

In the terms of [`architecture/sequence-state.md`](../architecture/sequence-state.md), "Sequence state".

### 6.1 Per position: pages

| Part | Per cell | Notes |
| --- | --- | --- |
| K of attention layer `l` (12 layers) | 2 heads x 256 values | after norm and rotation |
| V of attention layer `l` | 2 heads x 256 values | as projected |
| K and V of the MTP block | 2 x 256 values each | the block's own cache; one more "layer" |

That is 12288 values per cell for the trunk: 24576 bytes as f16, 13056
bytes with the engine's `q8` (8.5 bits per value), which is what the layout
in the architecture document already assumes. The MTP block adds 1024 values, 1088
bytes in `q8`. For a split over devices, Strata's all-cards engine divides
the cells among the cards by block and keeps every pooled key on every card
(S: `src/split/src/kernels.hpp`, "The K/V cache is split by position").

### 6.2 Per block of 4 cells: a per-page part

One pooled key of 128 values per block and attention layer, written when
the block's last cell is processed. In f32: 512 bytes per block and layer,
1536 bytes per cell over the 12 layers. The page size must be a multiple of
4 so that a block never straddles pages (and of 128 for the tiled KV
formats). The MTP block needs no pooled keys as specified here
(difference 9).

A pooled key is read only by rows whose `n` makes its block visible, so a
key written by a row that is later rejected is never read: after the
truncation the block is incomplete again and its slot is rewritten when an
accepted row completes it.

### 6.3 Recurrent: not addressable by position

| Part | Size | Advances |
| --- | --- | --- |
| `M` of each recurrent layer (36) | 48 x 128 x 128 f32, 3.0 MiB | every cell |
| `u` history of each recurrent layer | 3 x 10240 f32, 120 KiB | every cell |
| PLE history (layer 1) | 9 x 10240 f32, 360 KiB | every cell |
| Raw indexer keys of the incomplete block, per attention layer (12) | up to 3 x 128 f32 | every cell; empty when the cell count is a multiple of 4 |
| The two tokens before the next cell | 2 ids, host | every cell |
| Next rotary position | 1 number, host | differs from the cell count only after images |

Together 112.2 MiB for the recurrent layers plus 0.4 MiB. This is what a
checkpoint holds; Strata's checkpoint holds the same device state next to
the token ids (S: `include/strata/core/conversation_cache.hpp`
`ConversationCheckpoint`: `ids`, `gdn`, `ple`, `tails`; its `dead` and
`block_pos` belong to its kernels and are not needed). A sequence can resume at cell `n` only where a
checkpoint for `n` exists and the pages hold the cells below `n`. A
checkpoint at a cell count that is a multiple of 4, and so at every page
end, needs no indexer part.

The MTP block has no recurrent state: its K/V is per position, and as
specified here it has no indexer.

### 6.4 What a step keeps, and how it ends

For a sequence with a window of rows at cells `p0 .. p0 + T - 1` (the
newest token and drafts, or a piece of a prompt):

| Part | During the window | Commit of the first `a` rows | Rejected rows |
| --- | --- | --- | --- |
| K and V | appended at each row's cell before the row attends | page table cut after cell `p0 + a - 1` | cells above the cut are rewritten later, never read meanwhile |
| Pooled keys | written for blocks completed inside the window, from the carried raw keys and the window's | nothing | 6.2 |
| Raw indexer keys | carried ones are read; the window's rows are kept per row and layer (128 f32) | the carried set becomes the raw keys of the cells of the incomplete block after cell `p0 + a - 1` | dropped |
| Recurrent layers | rows computed in order from the carried `M` and `u` history, neither written; kept per row and layer: `u`, `g` and `beta` (the last two depend on the row's input), and `q`, `k`, `v` unless the commit recomputes them from `u` and the history | the first `a` rows are replayed into `M`; the history becomes the last 3 `u` | dropped |
| PLE | rows computed from the carried history; `nrm` kept per row | the history takes the first `a` rows' `nrm` | dropped |
| Hash window, position | host | advance by `a` tokens | nothing |

Strata keeps `u`, the vector after convolution and norms, `g` and `beta`
per row and recurrent layer, 80 KiB per row and layer, 2.8 MiB per row over
the 36 layers (S: `include/strata/core/verify.hpp`, members `qkv_L_`,
`h_L_`, `gate_L_`, `beta_L_`; `src/core/verify.cpp` `capture_commit`). It
appends the window's raw keys to the carried tail during the window and
restores the tail from a snapshot at the commit; keeping the tail untouched
and pooling from "carried plus window" is the same result (Strata's split
engine does that: S: `src/split/src/qsa.cuh` `k_idx_prep_pool`).

A fork inside a page copies that page before it writes (the planner's copy
on write): besides K and V, the pooled key of the block that contains the
fork point will differ between the two continuations.

## 7. For the engine: kernels and host pieces that are missing

Ares has: quantized matrix times rows for eleven types (Q4_K, Q5_K, Q6_K,
Q3_K, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, Q8_0) for one to
thousands of rows, RMS norm, add, activation quantization, sums between
devices; expert tiers and waves as bookkeeping; reads of experts and table
rows from the file; the planner with a layer-by-layer pass for prompts.
Missing, for this architecture:

**GPU kernels**

1. **Weight types.** Matrix times rows for IQ2_XXS (22 tensors, the routed
   gate and up of 11 layers), IQ1_M (6, of 3 layers), Q2_0 (53: every
   routed down, 5 shared downs), BF16 (388: mixers, indexer, `ssm_alpha`
   and `ssm_beta`, `ple_key`, `ple_value`; all of the MTP block's dense
   tensors) and F32 (48 routers of 512 x 2560, 48 gate vectors). Covered
   already: IQ4_XS, IQ3_S, Q6_K, IQ4_NL, IQ2_S, Q8_0.
2. **Experts by id.** For the (row, expert) pairs of a step grouped by
   expert: gate and up on the same input, `silu(gate) * up`, down, and the
   sum into the row's output with the router's weight; each expert's three
   matrices addressed where the tier holds them. Gate and up share a type
   per layer, down is Q2_0.
3. **Router.** Softmax over `E`, the `K` largest with the lower index
   first, the renormalised weights; the ids and weights go to the expert
   kernel and, for the tiers' heat and for missing experts, to the host.
4. **Embedding gather.** The token's row of `token_embd` (IQ4_XS)
   dequantized, copied to the `hc` streams.
5. **Stream mixer.** Per-stream RMS norm with a weight per value;
   `silu(x / hc)`; the gated mean over streams; the write
   `R[c] += out * 2 * sigmoid(inj[c] / hc)`. The three projections are
   item 1.
6. **Elementwise.** Sigmoid, SiLU, `softplus(a + b) * c`, the output gate
   `x * sigmoid(g)`.
7. **Recurrent layer.** (a) the 4-tap depthwise convolution over history
   plus rows, with SiLU; (b) `l2` per head for q and k; (c) the recurrence
   over a run of rows from the carried state, not writing it; (d) the
   commit, replaying the first `a` rows and writing `M` and the history;
   (e) the same recurrence for the thousands of rows of a prompt piece;
   (f) the output norm per head with `ssm_norm` and `sigmoid(z)`.
8. **Norm and rotation.** RMS norm per head with weights, then the partial
   rotation with three position streams, for q, k, the indexer queries and
   the pooled key.
9. **K/V append.** Encode K and V rows in the page's format and write them
   through the page table; also for the MTP block.
10. **Indexer keys.** Pool completed blocks (mean of 4 raw keys from the
    carried ones and the step's rows, norm, rotation at the first cell)
    into the page's per-block part; update the carried raw keys at commit.
11. **Block scores.** Per row, the scores of its visible blocks: four dot
    products of 128 with a rectifier each, summed; keys read through the
    page table; the number of visible blocks differs per row.
12. **Selection.** Per row, the best blocks under the cell budget and the
    tail, as an ascending list of cell ids with its length; the budget rule
    a parameter (difference 1).
13. **Attention over an id list.** 12 query heads per KV head, softmax over
    the listed cells, K and V read through the page table in the page's
    format, for a few rows (up to 2051 cells each) and for the many rows of
    a prompt piece. Ares has the KV formats and no attention kernel yet.
14. **PLE block.** Per-stream norms, the per-stream dot product with the
    signed square root and sigmoid, the dilated convolution over history
    plus rows with the F16 kernel, the residual update; the history's
    commit.
15. **Output.** The head is item 1 over 248320 rows (322 MiB of weights
    per pass); picking a token from 248320 logits; a head over a list of
    rows for drafts.
16. **MTP block.** The items above with its own weights; the one new piece
    is the norm over all `D` values.

**Host**

1. The n-gram hash of 3.4 for every row, drafts included (a draft row's
   window ends in the drafts before it).
2. The 16 table rows per row: ids to `ares-io` row reads, IQ4_NL decoding
   (90 bytes to 160 values), 2560 values per row on the device before
   layer 1. Strata measured 623 rows per second of cold direct reads on
   this machine's drive for this table (its log line, 2026-11-07), so the
   row cache and reading ahead for drafts and prompts matter.
3. Three position numbers per row in the step record (equal for text).
4. The way from the router's picks to experts that are not on the device,
   inside a step: the picks of layer `l` are known only after the layers
   before it. Strata stops the device in the middle of the step on a flag
   the host raises (S: `include/strata/core/layer.hpp` `Doorbell`); the
   pass for prompts in `ares-sched` is the layer-by-layer answer. For
   decode rows this piece is open in Ares.
5. The loader: tensors by the names of section 2, shape checks against the
   metadata, the two conventions of the MTP file, the draft token list.
6. Checkpoints of 6.3 with the PLE history and the carried raw keys.
7. The tokenizer (byte-level BPE, pre-tokenizer `qwen35`) and the chat
   template; not specified here.

## 8. What an implementation can be checked against

The machine: two GPUs with 12 GB each, 31 GB of RAM; the model is 63.47
GiB, of which 26.82 GiB is the n-gram table (read by rows) and about 33
GiB the routed experts. Nothing below was run for this document; other
work was using the GPUs.

### 8.1 Strata, with this model

It runs here: `strata-swift-iq2_xs.json` and `strata-swift-iq2_xs.log` in
the Strata directory are a configuration and a log of 2026-11-07 with the
0.1.48 engine on both GPUs (layers 0 to 25 on one, 26 to 47 on the other).
From that log: the three largest loading steps take 6.9, 16.3 and 11.1
seconds; a 22 to 34 token prompt is read at 10 to 34 tokens per second; 128
tokens are generated at 12 to 62 tokens per second with drafts; 2072 prompt
tokens are read at 12 to 130 and 8216 at 34 to 201 tokens per second, with
that much spread between runs.

What it can write out, by reading the source (0.1.59) and the strings of
the installed binary (0.1.48), not by running it:

| Output | How | Limits |
| --- | --- | --- |
| The residual after every layer, per position (`(L + 1) x hc x N` f32), halves, routed ids | `--dump-layers`, `--dump-halves`, `--dump-routing`, `--dump-logits` | **Not available for this model.** These are written by the token loop, and a pack with i-quant experts ("native") skips the token loop (S: `src/program/generate.cpp`, `if (native_pack) { spec_pos = pos; break; }` and "a native pack runs verify windows only") |
| Layer outputs for chosen layers 0..46, per row of every verify window: position, token, then per layer `N` f32, the MEAN of the four streams of the residual after the layer. **Correction (2026-11-10, run with 0.1.48):** the file starts with a header that this row did not mention: `"STAP"`, `u32` version 1, `u32` `N`, `u32` number of tapped layers, then the tapped layers as `i32` each; the records follow | `STRATA_TAP_FILE=path`, `STRATA_TAP_LAYERS=0,1,...` | the whole model in one verifier, so one GPU (S: `src/core/verify.cpp` `Verifier::init`, `tap_write`) |
| Logits of the first window's row (the prompt's last token), `V` f32 | `STRATA_DUMP_FIRST_LOGITS=path` | one row per request |
| Log-probabilities per row | `STRATA_LOGPOS=path` (and `_TOPK`) | text lines |
| A hash of the final residual rows and the picks of every window | `--window-hashes path` | equality only |
| Reading given tokens through windows instead of the prompt path | the engine without `--serve`: a one-token prompt in `--tokens` and the rest in `--spec-follow path` (teacher forcing); under `--serve`, `--short-read N` | how a prompt of several tokens behaves without `--prefill` on such a pack was not checked |

So Strata gives, for this model, logits and the stream-mean of every layer
but the last for rows that went through verify windows, on one GPU. A run
on one 12 GB card holds fewer experts in VRAM than the logged run; its
speed was not measured. One tap row is 10 KiB per layer, 470 KiB for all
47.

**Run on 2026-11-10 (section 10):** without `--serve` the 0.1.48 engine
refuses `--layer-split` ("needs --serve"), so such a run is on one card
(`--gpu 0`); it takes 36 seconds for a 22 token prompt and 8 tokens, and the
logits and the taps come out as described.

What does not match by construction when comparing with it: difference 1
is absent (the document follows Strata); the activation rounding of section
5's last paragraph is present, so agreement is to a tolerance, not bit for
bit; Strata's prompt path is not bit-identical to its own windows.

### 8.2 The fork, with this model

The fork is built (`build/bin`, with `llama-debug`, `llama-eval-callback`,
`test-llama-archs`) and knows every tensor type in the file, including
Q2_0. Whether it loads and runs this 63.47 GiB file within 31 GB of RAM and
two 12 GB cards was **not verified**: the n-gram table is read by rows
(F: `load_arch_tensors`, `TENSOR_READ_LAZY`), the experts would have to be
split between the cards and mapped host memory, and no configuration for
this model exists in `llama-configs`.

If it runs, it is the only program here that exposes every intermediate
tensor: the graph names them (`hc_init`, per layer `hc_mixed`, `hc_inject`,
`linear_attn_out`, `attn_output`, `indexer_score`, `indexer_top_k`,
`ffn_moe_topk`, `ffn_out`, `l_last`, and `result_norm`, `result_output`; F:
the `cb(...)` calls in `qwen4exp.cpp`), and an eval callback sees each one.
`llama-debug` prints only the first and last values and the sum of each
tensor and saves the last token's logits (`--save-logits`). Full tensors
need a small program with its own callback; S: `tools/qwen35/
q36_llama_dump.cpp` is that program for the other architecture (it dumps
`l_out-N`; for this one the node is `l_last-N`, `[N, hc, rows]`, 1.97 MiB
per position for all layers). Fused operations in the fork remove some
intermediate nodes. The MTP block cannot be checked with the fork here
(2.5).

Differences 1 to 5 apply when comparing with the fork.

### 8.3 A small synthetic model

Most checking should not need the big file. What a synthetic model of this
architecture needs:

- **A writer.** Every tensor of 2.3 at small sizes with seeded random
  values, and the metadata of 2.2. Unquantized types (F32, BF16) are enough
  to check the architecture; the quantized products are checked at the
  kernel level against `ares-quant`. The values must keep activations
  finite over many layers (S: `tools/q4geo_synth.py` scales its blocks for
  that and notes that a random router packs its logits so closely that two
  implementations pick different experts, which it answers with a router
  scale).
- **Small numbers that exercise every rule:** at least 8 layers so that
  two are attention; an indexer budget of a few blocks (for example
  `top_k = 8`) and sequences several times longer, with lengths in all
  four residues modulo 4, so that selection, the tail, and the partial
  block of difference 1 occur; value heads a multiple of key heads greater
  than one, so that `n % GK` is tested; `H / HKV > 1`; more experts than
  `K`; a PLE layer with real hash constants, a small table and sequences
  that contain the cut token; sequences longer than 9 cells for the PLE
  history and longer than 3 for the recurrent one.
- **A second implementation to compare with.** The fork reads shapes from
  the metadata and has a CPU backend, so it can run such a file without a
  GPU; its own generator, `test-llama-archs -a qwen4exp -o DIR` (F:
  `tests/test-llama-archs.cpp` `get_gguf_ctx`), writes a random model of 2
  layers (one recurrent with the PLE, one attention) with width 256. Its
  indexer budget is 131072 cells, which never selects ("a realistic top-k
  here makes the results unstable and hard to match between CPU and GPU"),
  it has 2 experts and uses both, and as many value heads as key heads: it
  leaves the block selection, the experts' top-k and the head pairing
  untested. **Correction (2026-11-10):** on this machine that generator
  does not run: the prebuilt `build/bin/test-llama-archs` is older than the
  libraries next to it and stops with "key not found in model:
  qwen4exp.ssm.group_count". Constraints the fork puts on a file: one compress ratio above
  1 that divides `top_k`; the indexer key not narrower than the rotated
  width; the PLE layer must be a recurrent layer; `hyper_connection.count`
  above 1. Whether a file with a small `top_k` runs stably on the fork's
  CPU backend was not tried. **Settled (2026-11-10):** it does, see
  section 10.
- **Strata cannot run a small model.** Its kernels are compiled for one
  shape and fix `S = 128`, `GK = 16`, `HD = 256`, `ROT = 64`, the indexer
  at 4 x 128, `B = 4`, `hc = 4`, a width of 1024 to 4096 in multiples of
  256 (S: `include/strata/core/geometry_checks.hpp`), and the loader pins
  the layer count and `top_k` too (S: `gguf_reader.hpp`
  `geometry_unsupported`). A reduced shape needs Strata built from source
  with a geometry header (`-DSTRATA_GEOMETRY_FILE`, S:
  `cmake/strata_geometry.cmake`, `tools/q4geo_synth.py --header`), which
  does not exist on this machine. So the synthetic model is checked against
  the fork and an own reference, and Strata is the check on the real file.
- **An own reference.** A plain double-precision implementation of section
  3 on the CPU, compared once with the fork on the synthetic file (with
  the selection rule switched to whole blocks, difference 1) and then used
  as the oracle for the kernels. Strata's comments describe the same
  method, a Python reference (`ref/gdn.py`, `ref/qsa.py`, `ref/model.py`,
  ...) written before the kernels; of it only `ref/load.py` is in this
  checkout.
- **Kernel-level vectors that exist already:** the hash rows above (S:
  `src/kernels/ple_oracle_vectors.inc`), and Strata's parity programs name
  the properties worth pinning (S: `include/strata/kernels/qsa.hpp`,
  `gdn.hpp`, `ngram.hpp`: the rectifier per head, rotation at the block's
  first cell, the tail, `h % GK`, decay before update, SiLU before the
  split, the cut of the hash window).

## 9. Open points

1. **The selection rule past 2051 cells** (difference 1): Strata's main
   engine against its split engine and the fork. The original
   implementation was not read.
2. **The MTP block's norm of the trunk residual** (difference 8): one RMS
   over 10240 values (Strata's default) or one per stream (the fork).
   Decided by measurement on 2026-11-15 for one per stream ("On difference
   8" in section 5): 63 against 58 of 80 guesses on one text. One text of
   80 tokens is a small sample.
3. **The MTP block's attention past 2051 cells** (difference 9): the
   checkpoint ships an indexer for the block and the converter gives the
   block a compress ratio (F: `conversion/qwen4exp.py`
   `set_gguf_parameters`); Strata ignores it on purpose. The reference and
   the engine use the block's indexer; nothing was measured past 2051
   cells. Only draft quality depends on it.
4. **The `1 +` on the MTP file's norm weights** rests on `tools/mtp_rt.py`
   and on value ranges, and since 2026-11-15 on a run: with it the block
   guesses 63 of 80 of the trunk's tokens (section 5).
5. **Whether the fork runs this file on this machine, and how fast**
   (8.2): not tried.
6. **Strata's tap output on one GPU** (8.1): read from source and from the
   binary's strings, not run; the 0.1.48 binary is older than the source
   read.
7. **Tolerances.** Neither reference defines exact values (section 5, last
   paragraph). What difference from Strata's taps and logits counts as
   agreement has to be set from a measurement, for example Strata against
   itself with different window sizes.
8. **Images** (3.9) are described only as far as the text pass must allow
   for them.
9. `tools/q4geo_synth.py` says in its header that it does not write the
   n-gram table, while its code has a branch that writes a sparse one. Not
   resolved; it does not matter unless Strata is rebuilt for another shape.
10. [strata-notes.md](strata-notes.md) says Strata pins everything "except
    expert count, top-k and vocabulary" at compile time. The loader pins
    `attention.indexer.top_k` as well (S: `gguf_reader.hpp`
    `geometry_unsupported`).

## 10. Checked on 2026-11-10 with the reference in `ares-model`

The reference of 8.3 exists (`crates/model/src/qwen4exp/`), with the writer
of the small model (`synthetic.rs`), the dump program for the fork
(`crates/model/tools/q4capture/`, whose README has the commands and the full
numbers) and a program that runs the real file (`tools/q4trace.rs`). What
running them showed, by the open point of section 9 it bears on:

- **The formulas of section 3** (not an open point, but the premise of the
  rest). On the small model with every tensor `f32`, `top_k = 8`, blocks of
  4, 12 experts of which 3 are used, 6 value heads on 2 key heads and 3
  streams, the fork's CPU backend and the reference with the selection rule
  set to whole blocks agree to rounding in every layer: at most 1.6e-5
  relative RMS in the worst row of the residual and of the logits over three
  seeds and 48 to 200 tokens, the same routed experts in every row and
  layer, the same blocks kept in all 335 rows that select, whether the
  sequence is decoded in one call, token by token or in calls of five or
  seven. The hash rows of 3.4 are Strata's oracle rows. On the real file the
  reference generates sensible text under greedy picks (a German answer
  with its reasoning, cut at 160 tokens, and an English answer that the
  model ends itself after 80).
- **Point 1, the selection rule.** Not settled: which rule the original
  uses is still unread. Confirmed as described: the fork keeps whole blocks,
  and the reference with the cell budget differs from it in exactly the rows
  whose tail has fewer than three cells. Read from S for this
  (`qsa_select.cu` `block_topk_kernel`): the main engine selects by a
  threshold over cells, takes every cell above it and the cells at it
  lowest index first, which is the rule of 3.6 including its order of equal
  scores. What the choice does to the real model was measured once: 32 rows
  at cells 2080 to 2111 of a 2178 token text, run from the same state under
  both rules, differ by 0.8e-2 to 1.9e-2 relative RMS in the logits on
  average (8e-2 at worst) and in 52 of 15360 expert picks, with the same
  greedy pick in all 32 rows. That is no larger than the distance between
  Strata and the exact arithmetic (point 7), so logits will not tell the two
  rules apart.
- **Point 5, the fork with the real file.** Not tried. A small file with a
  realistic `top_k` runs stably on the fork's CPU backend (above).
- **Point 6, Strata's tap output.** Settled: the installed 0.1.48 engine
  writes the taps and the first logits on one card without `--serve`
  (corrections in 8.1).
- **Point 7, tolerances.** Measured for one German prompt of 22 tokens and
  the 64 tokens after it, Strata 0.1.48 on one card with `--kv fp16`
  against the exact reference: the 64 greedy tokens are the same; the
  logits of the last prompt token have the same largest value (probability
  1.0 in both), nine of the ten largest in common, a relative RMS difference
  of 0.12 over all 248320 values and a KL divergence of 2.2e-6 nats. (An
  English prompt of 19 tokens: the same 80 greedy tokens up to the end of
  the answer, the ten largest logits the same, 0.058 and 1.9e-8 nats.) The
  mean of the streams after a layer differs by 6.7e-3 relative RMS after
  layer 0 (median of 64 cells), 1.0e-2 after layer 11, 2.4e-2 after layer
  23, 3.3e-2 after layer 35 and 2.3e-2 after layer 46, with 1.3e-1 as the
  worst cell; the last prompt token, whose state Strata's prompt path left,
  is further off (up to 2.0e-1 in the middle layers). The growth is smooth,
  with no step at the n-gram layer or at an attention layer. Strata gives
  the same logits bit for bit when run twice. That the gap is activation
  rounding (section 5, last paragraph) is supported, not proven: rounding the
  reference's activations as ggml's `Q8_0` does (blocks of 32) in front of
  every quantized matrix brings the difference after layer 0 from 7.2e-3 to
  2.3e-3; Strata's own rounding contract was not reproduced, and the
  remaining 2.3e-3 is not attributed. So "agreement with Strata" for a GPU
  implementation is, by this measurement, the same greedy tokens and stream
  means within about 1e-2 after the first layers and 3e-2 to 1e-1 further
  up; a tighter bound needs Strata against itself on its two paths.
- **Not an open point, but a trap when comparing with the fork on the
  CPU:** llama.cpp's CPU matrix product converts the activations to the
  dot-product type of the weights, which for BF16 weights is BF16 (8 bits of
  mantissa) and for F16 weights F16. With the real file's unquantized types
  in the small model the fork is 1e-3 away after layer 0 for that reason; a
  model with every tensor `f32` has no such rounding.

Points 2, 3 and 4 (the MTP block), 8 (images), 9 and 10 were not touched.
