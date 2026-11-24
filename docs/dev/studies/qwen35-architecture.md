# The `qwen35` architecture

This is the specification of the model family that `general.architecture = "qwen35"`
names in a GGUF file: Qwen3.5 and Qwen3.8 dense models. It is written so that the
forward pass can be implemented from this document alone, and it is what the CPU
reference in `crates/model` (`ares_model::qwen35`) implements and what the GPU
implementation will be held to.

Two models are the test targets; the numbers of their configurations below are read from their files:

| | file | size |
| --- | --- | --- |
| **0.8B** | `Qwen3.5-0.8B-Q4_K_M.gguf` | 507.8 MiB, 24 blocks, width 1024 |
| **27B** | `Qwen3.8-27B-UD-Q4_K_S.gguf` | 14.30 GiB, 64 blocks and a next-token block, width 5120 |

How to read the claims. The definition of the architecture is the code of the user's
llama.cpp fork (`~/Services/llama.cpp-native`, commit `f8f346786`),
cited as `file: function`. A statement marked **read** comes from that source or from
the files' headers. A statement marked **measured** was observed by running something
(the reference, llama.cpp on the CPU, or the tools in `crates/model`); the section
"Proof against llama.cpp" has the details. What could not be established is in
"What was not verified".

## 1. In one paragraph

`qwen35` is a decoder-only transformer whose blocks come in two kinds. Three of every
four blocks are **recurrent**: a short causal convolution over a projection of the
input, then a gated delta rule (a linear-attention recurrence with a matrix state per
head that is multiplied by a decay and corrected by the error of its own prediction at
every token), then a gated RMS norm and an output projection. Every fourth block
(blocks 3, 7, 11, ...) is full softmax **attention** with grouped key/value heads, a
query/key RMS norm, rotary embedding on a quarter of each head, and a sigmoid gate on
the attention output that is computed from the same projection as the query. Every
block ends in the same SwiGLU feed-forward part, and both parts sit inside pre-norm
residual connections. The output is an RMS norm and a projection to 248 320 logits,
shared with the embedding table in the 0.8B and separate in the 27B. Only the attention
blocks keep state per position (keys and values); the recurrent blocks keep a fixed
size state per sequence, 3 MiB per block in the 27B.

## 2. The files

### 2.1 Metadata

`ares_model::qwen35::Config::from_gguf` reads exactly these keys. A missing key, a key
under `qwen35.` that is not in the table, or values that do not describe a model are
errors that name the key. `general.*`, `tokenizer.*` and `quantize.*` are not read.

| key | 0.8B | 27B | what it controls |
| --- | ---: | ---: | --- |
| `general.architecture` | `qwen35` | `qwen35` | selects this implementation |
| `qwen35.block_count` | 24 | 65 | blocks in the file, including the next-token block when `nextn_predict_layers` is 1 |
| `qwen35.nextn_predict_layers` | absent (0) | 1 | number of next-token-prediction blocks stored after the model's blocks (0 or 1) |
| `qwen35.context_length` | 262144 | 262144 | trained context; the reference refuses positions beyond it |
| `qwen35.embedding_length` | 1024 | 5120 | width `d` of the hidden state |
| `qwen35.feed_forward_length` | 3584 | 17408 | width of the feed-forward part |
| `qwen35.attention.head_count` | 8 | 24 | query heads |
| `qwen35.attention.head_count_kv` | 2 | 4 | key/value heads; each serves `head_count / head_count_kv` (4 and 6) consecutive query heads |
| `qwen35.attention.key_length` | 256 | 256 | width of a query/key head |
| `qwen35.attention.value_length` | 256 | 256 | width of a value head; must equal `key_length` |
| `qwen35.attention.layer_norm_rms_epsilon` | 1e-6 | 1e-6 | epsilon of every RMS norm and of the query/key normalisation of the recurrent blocks |
| `qwen35.rope.dimension_count` | 64 | 64 | number of leading values of a head that are rotated (a quarter of 256) |
| `qwen35.rope.dimension_sections` | [11, 11, 10, 0] | [11, 11, 10, 0] | split of the 32 rotated pairs among position kinds (see 4.4) |
| `qwen35.rope.freq_base` | 1e7 | 1e7 | base of the rotation frequencies |
| `qwen35.ssm.conv_kernel` | 4 | 4 | width `K` of the causal convolution |
| `qwen35.ssm.state_size` | 128 | 128 | width `d_k` of a query/key head of the delta rule, and of a value head |
| `qwen35.ssm.group_count` | 16 | 16 | number of query/key heads `H_k` |
| `qwen35.ssm.time_step_rank` | 16 | 48 | number of value heads `H_v`, one state matrix each |
| `qwen35.ssm.inner_size` | 2048 | 6144 | `H_v * d_k`; checked |
| `qwen35.full_attention_interval` | 4 | 4 | block `i` is attention when `(i + 1) % interval == 0` |

Block kinds follow from the interval alone (**read**: `llama_model_qwen35::load_arch_hparams`
looks for an optional per-block array of recurrent blocks first; neither file has it, and
the reference would refuse the key as unknown). Attention blocks are 3, 7, 11, 15, 19, 23 in the 0.8B and 3, 7, ..., 63 in
the 27B. No key is read for scaling the RoPE (`rope.scaling.*`, YaRN): neither file has
one, so the frequencies are the plain ones (**read**: `llama-context.cpp` sets
`yarn_ext_factor = 0` unless the scaling type is YaRN, and the default type is linear
with factor 1). The attention scale is `1 / sqrt(key_length)`, the key
`attention.scale` is absent (`hparams.f_attention_scale == 0`).

Further keys exist in the files and are not read: the tokenizer (`gpt2` byte-level BPE
with `pre = qwen35`, 248 320 tokens, EOS 248046, padding 248055, BOS 248044 in the 27B
only), the chat template, `general.sampling.{top_k,top_p,temp}` (27B: 20, 0.95, 1),
and the quantization record (`quantize.imatrix.*`, `general.file_type` 15 and 14).

### 2.2 Tensors

`d = n_embd`, `F = n_ff`, `H = n_head`, `G = n_head_kv`, `h = head_dim`,
`H_k, H_v, d_k` as in the metadata, `key_dim = H_k d_k`, `value_dim = H_v d_k`,
`conv_dim = 2 key_dim + value_dim`, `K = conv_kernel`. Dimensions are in GGUF order:
the first one is contiguous, so `[cols, rows]` is a matrix of `rows` outputs of `cols`
inputs (the first dimension is the number of values a quantized block row spans). `N` is the block index.
The reference accepts any type `ares_quant::dequantize` can decode for every tensor
and requires exactly these dimensions; a tensor that is not listed is an error.

Every block:

| name | dims | role | 0.8B types | 27B types |
| --- | --- | --- | --- | --- |
| `blk.N.attn_norm.weight` | `[d]` | RMS norm weight before the recurrent or attention part | f32 | f32 |
| `blk.N.post_attention_norm.weight` | `[d]` | RMS norm weight before the feed-forward part | f32 | f32 |
| `blk.N.ffn_gate.weight` | `[d, F]` | gate | q4_K | iq2_xs, iq3_xxs, iq3_s, iq4_xs, q3_K, q4_K, q5_K |
| `blk.N.ffn_up.weight` | `[d, F]` | up | q4_K | iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs, q3_K, q4_K, q5_K, q6_K |
| `blk.N.ffn_down.weight` | `[F, d]` | down | q4_K, q6_K | iq3_xxs, iq3_s, iq4_nl, iq4_xs, q3_K, q4_K, q5_K |

Recurrent blocks (18 in the 0.8B, 48 in the 27B):

| name | dims | role | 0.8B | 27B |
| --- | --- | --- | --- | --- |
| `blk.N.attn_qkv.weight` | `[d, conv_dim]` | query, key and value in one projection | q5_K | iq4_nl, iq4_xs, q4_K |
| `blk.N.attn_gate.weight` | `[d, value_dim]` | the gate `z` of the output norm | q4_K | iq4_xs, q4_K, q5_K |
| `blk.N.ssm_conv1d.weight` | `[K, conv_dim]` | convolution, `K` taps per channel, channel-major | f32 | f32 |
| `blk.N.ssm_dt.bias` | `[H_v]` | bias of the decay input | f32 | f32 |
| `blk.N.ssm_a` | `[H_v]` | `-exp(A_log)`, negative (no `.weight` suffix) | f32 | f32 |
| `blk.N.ssm_beta.weight` | `[d, H_v]` | input of the update strength `beta` | q8_0 | q8_0 |
| `blk.N.ssm_alpha.weight` | `[d, H_v]` | input of the decay | q8_0 | q8_0 |
| `blk.N.ssm_norm.weight` | `[d_k]` | weight of the gated RMS norm over a value head | f32 | f32 |
| `blk.N.ssm_out.weight` | `[value_dim, d]` | output projection | q5_K | iq4_nl, iq4_xs, q4_K, q5_K, q6_K |

Attention blocks (6 in the 0.8B, 16 in the 27B):

| name | dims | role | 0.8B | 27B |
| --- | --- | --- | --- | --- |
| `blk.N.attn_q.weight` | `[d, 2 H h]` | query and output gate, per head the query then the gate | q4_K | iq4_nl, iq4_xs, q4_K |
| `blk.N.attn_k.weight` | `[d, G h]` | key | q4_K | q4_K, q5_K, q6_K |
| `blk.N.attn_v.weight` | `[d, G h]` | value | q4_K, q6_K | q4_K, q5_K, q6_K, q8_0 |
| `blk.N.attn_output.weight` | `[H h, d]` | output projection | q4_K | iq4_nl, iq4_xs, q4_K, q5_K |
| `blk.N.attn_q_norm.weight` | `[h]` | RMS norm weight over each query head | f32 | f32 |
| `blk.N.attn_k_norm.weight` | `[h]` | RMS norm weight over each key head | f32 | f32 |

Model:

| name | dims | 0.8B | 27B |
| --- | --- | --- | --- |
| `token_embd.weight` | `[d, n_vocab]` | q6_K | q3_K |
| `output_norm.weight` | `[d]` | f32 | f32 |
| `output.weight` | `[d, n_vocab]` | absent: the head is `token_embd.weight` | q6_K |

The next-token block (27B only, `blk.64`, the block index equals `n_layer`): the
attention-block tensors above (q6_K and q8_0), the feed-forward tensors and both norms
(q6_K), and `blk.64.nextn.eh_proj.weight [2d, d]` (q6_K), `blk.64.nextn.enorm.weight [d]`,
`blk.64.nextn.hnorm.weight [d]`, `blk.64.nextn.shared_head_norm.weight [d]` (f32). The
loader (**read**: `load_arch_tensors`, `load_block_mtp`) also accepts
`nextn.embed_tokens.weight` and `nextn.shared_head_head.weight`; the file has neither,
so the block uses the model's embeddings and head.

Two conversions are baked into the tensors (**read**: `conversion/qwen.py`,
`Qwen3NextModel.modify_tensors`), so the stored values are the ones to use directly:

- every norm weight except `ssm_norm` has 1 added at conversion (the HF checkpoint
  stores `w - 1` and computes `x * (1 + w)`); the file holds `1 + w` and the norm
  multiplies by it. `ssm_norm.weight` is stored as is. (Measured: `attn_norm` of block 0
  lies in [0.88, 2.0] in the 0.8B, `ssm_norm` in [0.35, 1.19].)
- `ssm_a` holds `-exp(A_log)`. Measured ranges: -10.6 to -0.001 in the 0.8B,
  -60.0 to -0.004 in the 27B, always negative; `ssm_dt.bias` reaches 19.25 in the 27B
  (so `softplus` meets its linear region).

And the value heads are stored in **tiled** order (**read**: `_LinearAttentionVReorderBase`
reorders the Hugging Face "grouped" order for the ggml broadcast): value head `h` of
`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_a`, `ssm_dt.bias`, the value
channels of `ssm_conv1d`, and the columns of `ssm_out` pairs with query/key head
`h mod H_k`. In the 27B (`H_v = 48`, `H_k = 16`) value heads 0, 16, 32 use key head 0.

### 2.3 What differs between the two models besides sizes

| | 0.8B | 27B |
| --- | --- | --- |
| value heads per key head | 1 (16 and 16) | 3 (48 and 16): the tiling of 2.2 matters |
| query heads per key/value head | 4 (8 and 2) | 6 (24 and 4) |
| output head | tied to `token_embd` | own `output.weight` |
| next-token block | none | `blk.64`, see 2.5 |
| token embedding / quantization mix | q6_K; q4_K, q5_K, q6_K, q8_0 | q3_K; eleven types (iq2_xs to q8_0) |
| BOS token in the tokenizer | none | 248044 |

Everything else of the architecture is equal: `head_dim` 256, `d_k` 128, `K` 4, the
rope settings, the epsilon, the vocabulary and the block pattern.

### 2.4 What the files contain that text inference does not use

- **Vision.** Neither file has vision tensors or vision keys (**measured**: the tensor
  lists have only the names of 2.2; no `v.*`, `mm.*`, no `clip.*` or
  `qwen35.vision.*` metadata). The 0.8B file carries the tag `image-text-to-text`
  only. The vision projector of the 27B is a separate file in the same directory,
  `mmproj-BF16.gguf` (888 MB). The position encoding is the one place the text path
  touches multimodal design: the rotary embedding has sections for time, height and
  width (4.4), and a text token uses its position for all three.
- The chat template, the sampling defaults and the quantization record (2.1).
- The next-token block of the 27B (2.5), unless a drafter uses it.

### 2.5 Multi-token prediction

The 27B has one next-token-prediction block and the 0.8B has none. Its weights are the
last block of the file, `blk.64.*` (index `n_layer`), listed in 2.2; the trunk is
`blk.0` to `blk.63`. The fork drafts with it (`common/speculative.cpp`,
`common_speculative_impl_draft_mtp`; the graph is `llama_model_qwen35::graph_mtp`).
**Read**, per drafted token at position `p+1`, with `h_p` the output of the final
norm of the trunk at position `p` (`h_nextn` in the graph: after `output_norm`) and
`e` the embedding of the token `x_{p+1}` (from `token_embd`):

```
e'   = rms_norm(e) * enorm          h'  = rms_norm(h_p) * hnorm
x    = eh_proj [e' ; h']            (concatenation, embedding first: 2d inputs, d outputs)
x    = x + attention_block(rms_norm(x) * attn_norm)           (the attention of 4.4, with its own K/V cache)
x    = x + ffn(rms_norm(x) * post_attention_norm)             (4.5)
logits = head( rms_norm(x) * (shared_head_norm, or output_norm when absent) )
```

The block's keys and values at a position depend on that position's input only, so it is
run in two ways (**read**: `common_speculative_impl_draft_mtp`): over every position the
model has computed, with the model's own `h` (zeros before position 0), which fills its
cache; and as a chain that drafts, where link 0 reads the newest token and the `h` it was
picked from, and each further link reads the token the link before picked and, in place
of the model's `h`, the block's own output after its head norm. The reference runs both
(`reference/mtp.rs`); on the hidden states of a llama.cpp run of the 27B its outputs are
within 3 % (relative L2, what llama.cpp's 8-bit activations cost) of what the fork's block
computes from the same hidden states, and its chains are the fork's token for token.

## 3. Conventions

`n` rows are consecutive positions `p_0 .. p_0 + n - 1` of one sequence. A "head" index
`i` of a flat vector of `H x w` values is the slice `[i w, (i + 1) w)`. `x @ W` below
means "the vector of `W.rows` dot products of the rows of `W` with `x`". `rms_norm(x; w)`
is defined in 4.2, `silu(x) = x / (1 + e^-x)`, `sigmoid(x) = 1 / (1 + e^-x)`,
`softplus(x) = ln(1 + e^x)`. `eps` is `layer_norm_rms_epsilon`.

## 4. The forward pass

### 4.1 The model

```
x_t      = token_embd[token_t]                              [d]   (decoded to f32; no scaling)
for block i in 0 .. n_layer:
    u_t  = rms_norm(x_t; attn_norm_i)
    y_t  = (recurrent_i | attention_i)(u rows, state_i)     [d]   (4.3 or 4.4)
    x_t  = x_t + y_t
    v_t  = rms_norm(x_t; post_attention_norm_i)
    x_t  = x_t + ffn_i(v_t)                                 (4.5)
    hidden_i[t] = x_t                                       (the hidden state "after block i")
f_t      = rms_norm(x_t; output_norm)                       (4.6)
logits_t = head @ f_t                                       [n_vocab]
```

Block `i` is attention when `(i + 1) % full_attention_interval == 0`. The residual stream
is never normalised or scaled between blocks. (**Read**: `llama_model_qwen35::graph::graph`.)

### 4.2 RMS norm

```
rms_norm(x; w)_j = x_j / sqrt( mean(x^2) + eps ) * w_j
```

over the whole vector (`d` values; `head_dim` or `d_k` values per head where 4.3 and
4.4 say so). ggml sums the squares in `f64` and the quotient in `f32`
(`ggml_compute_forward_rms_norm_f32`).

### 4.3 The recurrent block

Input: the normed rows `u_t` (`[d]`). State: the convolution window `C` (the last
`K - 1` inputs of the convolution of every channel) and one matrix `S_h` per value
head. All shapes use `d_k = 128`.

Projections (matrix products):

```
qkv_t = u_t @ attn_qkv     [conv_dim]      z_t = u_t @ attn_gate   [value_dim]
b_t   = u_t @ ssm_beta     [H_v]           a_t = u_t @ ssm_alpha   [H_v]
```

Per-head scalars (`h` over value heads):

```
beta_t[h]  = sigmoid(b_t[h])
g_t[h]     = softplus(a_t[h] + ssm_dt.bias[h]) * ssm_a[h]        (<= 0)
decay_t[h] = exp(g_t[h])                                          in (0, 1]
```

Causal convolution, per channel `c` of `qkv` (`conv_dim` channels), taps
`w[c][0..K-1]` (`ssm_conv1d.weight`, tap `K-1` multiplies the newest input), over the
inputs `e_j[c] = qkv_j[c]` and, before the window, the stored inputs of `C`:

```
conv_t[c] = silu( sum_{i=0}^{K-1} w[c][i] * e_{t-(K-1)+i}[c] )
```

`C` holds `e_{-(K-1)} .. e_{-1}`: the raw projections, before the convolution, oldest
first. A sequence starts with zeros.

Split `conv_t` into `q~` (first `key_dim` values), `k~` (next `key_dim`) and `v~` (last
`value_dim`), as `H_k` heads of `d_k` and `H_v` heads of `d_k`. Normalise queries and
keys per head, with the epsilon **inside** the root:

```
q^_{t,j} = q~_{t,j} / sqrt( sum(q~_{t,j}^2) + eps )       j over key heads, same for k^
```

(ggml spells it `rms_norm(x, eps / d_k)` times `1 / sqrt(d_k)`: `build_gdn_l2_norm` in
`src/models/models.h`.) The gated delta rule, for each value head `h`, with the key
head `j = h mod H_k` (tiled, see 2.2), and `S_h` a `d_k x d_k` matrix (rows indexed by
the key index `i`, columns by the value index `m`):

```
S_h   = decay_t[h] * S_h                                   decay the whole matrix
pred  = S_h^T k^_{t,j}                          [d_k]      what the state predicts for this key
delta = beta_t[h] * ( v~_{t,h} - pred )         [d_k]      the correction
S_h   = S_h + k^_{t,j} delta^T                             rank-one update
o_{t,h} = S_h^T q^_{t,j} / sqrt(d_k)            [d_k]      read out with the query
```

(**Read**, and what the CPU computes: `ggml_compute_forward_gated_delta_net_one_chunk`;
the fork always uses this fused operator, `cparams.fused_gdn_ar = fused_gdn_ch = true`
in `llama-context.cpp`; it is sequential over the rows of a sequence for both one
token and many. The chunked algorithm of `build_delta_net_chunking` is not used.)

Output of the block:

```
y_{t,h} = rms_norm(o_{t,h}; ssm_norm) * silu( z_{t,h} )    per value head, over its d_k values
out_t   = (y_t concatenated over h) @ ssm_out             [d]
```

After row `t` the state is `C = e_{t-K+2} .. e_t` and the matrices `S_h`. Memory layout
of the matrices in llama.cpp and in the reference: per head, `d_k` rows of `d_k` values
where row `m` (value index) holds the key index contiguously, i.e. element `[h][m][i]`
(**read**: "state is stored transposed" in the kernel).

### 4.4 The attention block

Input: the normed rows `u_t`; state: the keys and values of all earlier positions.

```
[q|gate]_t = u_t @ attn_q      [H * 2h]   head i: values [i*2h, i*2h + h) are the query, [i*2h + h, (i+1)*2h) the gate
k_t        = u_t @ attn_k      [G * h]
v_t        = u_t @ attn_v      [G * h]
q_{t,i}    = rms_norm(q_{t,i}; attn_q_norm)     per head, over h = 256 values
k_{t,g}    = rms_norm(k_{t,g}; attn_k_norm)
q, k       = rope(q, p_t), rope(k, p_t)         first rope_dim = 64 values of each head
```

The key as it enters the cache is the normed and rotated one, the value is unchanged.

**Rotary embedding** (**read**: `ggml_compute_forward_rope_flt`, `ggml_mrope_cache_init`
in `ops.cpp`, mode `GGML_ROPE_TYPE_IMROPE`, selected by `llama_model_rope_type`). Only
the first `R = 64` values of a head are rotated, in pairs `(x_p, x_{p+R/2})` for
`p = 0 .. R/2 - 1` (the "half-split" pairing, not neighbouring values); the other 192
values pass through. Pair `p` rotates by the angle

```
angle_p = pos_p * freq_base^( -2p / R )         pos_p = the token position for the three position kinds, 0 for the extra kind
x_p'       = x_p * cos(angle_p) - x_{p+R/2} * sin(angle_p)
x_{p+R/2}' = x_p * sin(angle_p) + x_{p+R/2} * cos(angle_p)
```

The sections decide which position a pair uses (interleaved, "IMRoPE"): with
`sector = p mod sum(sections)`, a pair uses the time position when
`sector mod 3 == 0 and sector < 3 s_0`, the height position when `sector mod 3 == 1 and
sector < 3 s_1`, the width position when `sector mod 3 == 2 and sector < 3 s_2`, and
the extra position otherwise. A text token has `(t, h, w, e) = (p, p, p, 0)` (**read**:
`llama-batch.cpp`, "expand [p] to [p, p, p, 0]"), so with the sections [11, 11, 10, 0]
all 32 pairs rotate by `p * freq_base^(-2i/64)` for pair `i`, with the position `p` the
0-based index of the token in the sequence. The reference refuses sections that do not
add up to `R / 2`. (ggml accumulates the frequencies by repeated `f32` multiplication,
`theta *= theta_scale`; the reference uses the closed form in `f64`, and the two agree
to the rounding of the comparison in 8.1.)

**Attention**, causal, per query head `i` with key/value head `g = i div (H / G)`
(consecutive query heads share a group: 4 per group in the 0.8B, 6 in the 27B):

```
s_{t,i,j} = q_{t,i} . k_{j,g} / sqrt(h)          for positions j <= p_t     (sqrt(256) = 16)
a_{t,i}   = softmax_j(s_{t,i,j}) . v_{j,g}       [h]
o_{t,i}   = a_{t,i} * sigmoid( gate_{t,i} )      elementwise over the h values
out_t     = (o_t concatenated over i) @ attn_output       [d]
```

No sliding window, no attention sink, no soft-capping, no bias, no position-dependent
scaling. (**Read**: `llama_model_qwen35::graph::build_layer_attn`.)

### 4.5 The feed-forward part

```
ffn(v) = ( silu(v @ ffn_gate) * (v @ ffn_up) ) @ ffn_down          [F] then [d]
```

The gate goes through `silu`, the up projection does not (**read**: `build_ffn` with
`LLM_FFN_SILU`, `LLM_FFN_PAR` and the fused `ggml_swiglu_split(gate, up)`). No bias; the
optional clamp of `swiglu_clamp_shexp` is zero in these models.

### 4.6 Output

`f_t = rms_norm(x_t; output_norm)`, `logits_t = f_t @ head` where `head` is
`output.weight` or, when the file has none, `token_embd.weight` (rows of the same
`[d, n_vocab]` matrix). Only the rows that are sampled need the final norm and the head.
The output of the norm is also what the next-token block takes as `h_p` (2.5).

### 4.7 Activation functions

| function | where | definition | in ggml (CPU) |
| --- | --- | --- | --- |
| `silu` | convolution output, `z` gate of the recurrent output norm, feed-forward gate | `x / (1 + e^-x)` | vectorised `ggml_v_silu` (a polynomial `exp`) |
| `sigmoid` | `beta`, attention output gate | `1 / (1 + e^-x)` | `1 / (1 + expf(-x))` |
| `softplus` | decay input | `ln(1 + e^x)` | `x > 20 ? x : logf(1 + expf(x))` |
| `exp` | decay `exp(g)`, softmax | | `expf` |

### 4.8 Where f16 or f32 matters

What the model needs (the reference): hidden states, norms, the convolution, the delta
rule and the softmax in `f32` with `f64` sums; the recurrent state in `f32`. What
llama.cpp does on its CPU path, which decides how far the two can be compared
(**read**, and **measured** where stated):

| quantity | llama.cpp | note |
| --- | --- | --- |
| hidden state, residual stream | f32 | the 0.8B stays small (rms 0.04 to 0.4, max 11 on an English text, **measured**), the 27B was not run |
| conv window `C` and matrices `S_h` | f32 (`create_memory`: `recurrent_type_r = recurrent_type_s = GGML_TYPE_F32`, `llama-model.cpp`) | per sequence, not per position |
| K and V in the cache | f16 by default (`cache_type_k = GGML_TYPE_F16`) | **measured**: `-ctk f32 -ctv f32` with flash attention gives bit-identical results, because `build_attn_mha` casts an f32 K/V to f16 for `ggml_flash_attn_ext` |
| flash attention, many rows | Q kept in f32, K and V widened from f16, f32 accumulators (`ggml_compute_forward_flash_attn_ext_tiled`) | |
| flash attention, one row (generation) | Q rounded to f16, the weighted sum of V accumulated in **f16** (`..._f16_one_chunk`, `VKQ16`); with 512 or more cache cells split over the threads (`use_split_kv_path`) | **read**; 8.3 measures that two thread counts give different text |
| activations that meet a quantized weight | rounded to the activation type of the weight: `Q8_K` (blocks of 256) for q4_K, q5_K, q6_K, q3_K and the iq types, `Q8_0` (blocks of 32) for q8_0 and iq4_nl (`vec_dot_type` in `ggml-cpu.c`) | **measured**: this is the larger part of the difference to an exact computation |
| dot products of weights and activations | integer within a block, f32 across blocks | a different summation order than the reference |
| softmax, `exp` in the delta rule | f32, `expf` | |

Rotation angles, norms and softmax denominators need `f32` at least; a GPU
implementation that stores K and V in the engine's formats (`f16`, `q8` and below)
should compare against this reference with the same K/V formats applied, not against
the exact values.

## 5. State per sequence, in Ares' terms

### 5.1 What is where

| state | blocks | size | goes into | per |
| --- | --- | ---: | --- | --- |
| keys and values `k_{j,g}`, `v_{j,g}` (after the norm and the rotation) | attention | `2 G h` values per position and block | **pages**, position-addressable | position |
| convolution window `C` | recurrent | `(K-1) conv_dim` values | per-sequence slot | sequence |
| delta-rule matrices `S_h` | recurrent | `H_v d_k d_k` values | per-sequence slot | sequence |

| | 0.8B | 27B |
| --- | ---: | ---: |
| attention blocks | 6 | 16 |
| K and V per position, all blocks (values) | 6 144 | 32 768 |
| ... at f16 | 12 KiB | 64 KiB |
| ... for the full 262 144 positions at f16 | 3.0 GiB | 16.0 GiB |
| `C` per recurrent block (f32) | 72 KiB | 120 KiB |
| `S` per recurrent block (f32) | 1 MiB | 3 MiB |
| recurrent state per sequence, all blocks (f32) | 19.3 MiB | 149.6 MiB |

Pages hold the first row of the table and `PageLayout` has to give `2 G h * n_attention`
values per position and device (the blocks of a device only). The recurrent state does
not grow with the context and does not fit pages: it is one record per sequence. A
**checkpoint** (`ares-state`) of a prefix is the record above, 150 MiB in the 27B, so the
checkpoint policy has to be sparse (a checkpoint per few thousand positions at most,
and it is what lets a cached prefix resume at all). Whether the state may be stored in
less than f32 was not measured; llama.cpp keeps f32.

### 5.2 Verify, then commit

A step runs a **window** of rows of a sequence: the newest token followed by draft
tokens, or a slice of a prompt. The rows are computed from the committed state, and
the accepted prefix is committed afterwards (`docs/dev/architecture/overview.md`, "A step"). For
this architecture:

- **Attention blocks**: the window's K and V rows are appended to the sequence's
  pages before the attention of the window runs (row `t` attends to the committed
  positions and to the earlier rows of the window). Rejected rows are cut with
  `Sequence::truncate`. Nothing else is needed.
- **Recurrent blocks** must not change `C` or `S_h` for rows that may be rejected,
  because the state cannot be cut. The window reads `C` and `S_h` as they were at its
  start and **keeps its inputs**. Committing `m` accepted rows replays them. What a row
  has to keep, per recurrent block:

  | kept | values | needed for |
  | --- | ---: | --- |
  | `k^_{t,j}` (normalised keys) | `key_dim` | the update of `S_h` |
  | `v~_{t,h}` | `value_dim` | the update |
  | `g_t`, `beta_t` | `2 H_v` | the update |
  | `qkv_t` (the raw projection) | `conv_dim` | the new `C` (only the last `K-1` accepted rows) |

  `q^` is not needed: it only reads the state. For the 27B that is 18 528 values
  (74 KiB at f32) per row and block, 3.5 MiB per row over 48 blocks. A snapshot of
  `S_h` after each row, the other possible design (the fork has it as `n_rs_seq`),
  costs 3 MiB per row *and block*, 150 MiB per row.
- Replay is cheap: `3 d_k^2` multiply-adds per value head per row, 113 million per row
  over all blocks of the 27B, and it reads and writes `S` once per commit, not per row,
  if the replay kernel loops over the accepted rows with the matrix in registers.
- The rows that are certain to be accepted (a generation row without drafts, the rows
  of a prompt) may be written into the state directly by the same kernel that computes
  them: only rows after a draft boundary need the deferred commit. `ares_model`'s
  `Reference::window` and `commit` are the specification of the deferred form and
  `forward` of the direct one; they give the same bits.
- **A window may be longer than the convolution** (`K = 4`): row `t` of a window reads
  the committed `C` for its first `K-1` rows and the raw projections of earlier rows
  after that; the same holds for the committed `C` after a commit of fewer than `K-1`
  rows (the window shifts, the older inputs stay). The reference tests commits at every
  length of a 14-token prompt, so every case of 1, 2, 3 and more rows is covered.

`SequenceState` in the reference is exactly 5.1: `conv(block)`, `delta(block)` for the
recurrent blocks, `kv(block)` for attention blocks, and the position.

## 6. GPU kernels that Ares does not have yet

Ares has: quantized matrix times 1 to thousands of rows (eleven types), RMS norm, add,
activation quantization. Shapes below are for the 27B; "rows" is `n`, the rows of one
step, any mix of sequences (one segment of consecutive rows per sequence, each with its
position and its state slot from the step record).

| # | kernel | in -> out | batch structure |
| ---: | --- | --- | --- |
| 1 | **embedding gather**: decode the row `token_embd[token]` | `n` ids -> `[n, 5120]` | one row per token; the table is q3_K (27B) or q6_K (0.8B), so a gather that decodes a row of any type |
| 2 | **RMS norm over heads**: `rms_norm` with row length 256 and a weight of 256 shared by all heads, and per-head variants for the recurrent output | `[n, 24, 256]` and `[n, 4, 256]` | rows = `n * heads`; this may be the existing kernel if it takes the row length and the weight length |
| 3 | **causal conv1d with state** (width 4, then `silu`) | `qkv [n, 10240]` + `C [10240, 3]` -> `[n, 10240]` | per segment: reads `C` of its sequence, row `t` sees `C` and the raw inputs before it; one thread per channel, loop over the segment's rows |
| 4 | **q/k normalisation** per head (`x / sqrt(sum x^2 + eps)`, 128 values) | `[n, 16, 128]` twice | fuse into 3 or 5 |
| 5 | **gated delta rule** | `q^, k^ [n, 16, 128]`, `v~ [n, 48, 128]`, `g, beta [n, 48]`, `S [slot][48][128][128]` f32 -> `o [n, 48, 128]` | grid = value heads (48) x segments; sequential over the rows of a segment; one block holds a head's matrix (64 KiB f32) with `d_k` columns spread over the warps, as the fork's `ggml-cuda/gated_delta_net.cu` does (one block per `(head, sequence)`, `S` in registers, a reduction over the key index per column); value head `h` reads key head `h mod 16`; ragged segments (llama.cpp asserts equal lengths, Ares cannot) |
| 6 | **commit**: advance `C` and `S` over the first `m` rows of a segment from the kept inputs | kept rows -> `C`, `S` | per segment `m` from the step record (0 for fully rejected); the same arithmetic as 5 without outputs |
| 7 | **gated RMS norm**: `rms_norm(o; ssm_norm) * silu(z)` over 128 values per head, followed by the activation quantization for `ssm_out` | `o, z [n, 48, 128]` -> `[n, 6144]` | rows = `n * 48`; fuse the quantization |
| 8 | **q/k prepare for attention**: split the query and gate halves of each head, per-head RMS norm, rotary on the first 64 values with the positions of the rows, write `k, v` into the pages in the K/V format of the engine | `qg [n, 24*512]`, `k, v [n, 1024]` -> `q [n, 24, 256]`, `gate [n, 24, 256]`, pages | the position of each row is in the step record; pages from the page table; the rotary is half-split pairs (4.4) with `cos`/`sin` per `(position, pair)` |
| 9 | **paged attention** over the pages of the sequence | `q`, pages -> `o [n, 24, 256]` times `sigmoid(gate)` | head dim 256 for K and V, 6 query heads per key/value head, causal inside the window, one page table per sequence, rows of different sequences and different lengths in one launch; the gate multiplication is the epilogue; reads the engine's K/V formats (the rotated formats of `ares-quant::kv`) |
| 10 | **SwiGLU with activation quantization**: `silu(gate) * up` then quantize for `ffn_down` | `[n, 17408]` twice -> int8 blocks | elementwise; the existing quantization kernel with a prologue |
| 11 | **row gather for the head**: pick the rows that are sampled (the last row of each sequence's window, all draft rows), then norm and head | `[n_out, 5120]` -> `[n_out, 248320]` | the head is the existing matrix product with `n_out` rows (up to thousands for scoring decisions) |
| 12 | **matrix products that share an input** (not new kernels, a scheduling request): `attn_qkv + attn_gate + ssm_beta + ssm_alpha` (10240 + 6144 + 48 + 48 outputs, one quantization of the input), `attn_q + attn_k + attn_v`, `ffn_gate + ffn_up` | | `ssm_beta` and `ssm_alpha` have only 48 outputs of q8_0: alone they are too small for a launch |

Sums across devices for a tensor split are the existing ones, twice per block (after
the recurrent or attention part and after the feed-forward part): with the split of the
recurrent block over value heads (each device holds whole heads and their `S`), the
only cross-device quantity is the `[n, 5120]` output of `ssm_out`; splitting a
head's `d_k` columns over devices would need a sum inside kernel 5 and is not
recommended.

## 7. The reference implementation

`crates/model` (`ares-model`), module `qwen35`:

| item | what it is |
| --- | --- |
| `Config::from_gguf` | the strict configuration of 2.1 |
| `Weights::load` | the exact table of 2.2; matrices stay as the stored blocks and are decoded row by row with `ares_quant::dequantize` |
| `SequenceState` | the state of 5.1 |
| `Reference::window(state, tokens, LogitRows)` | rows from the committed state; the state is untouched; returns the hidden state after every block, the output of the final norm and the logits of the asked rows, and the kept inputs of 5.2 |
| `Reference::commit(state, window, accepted)` | the accepted prefix of a window into the state |
| `Reference::forward` | a window committed whole |
| `Reference::block(layer, state, x)` | one block alone on given input rows: the unit for comparing a GPU block with this one |

The arithmetic is 4.1 to 4.6 with `f32` storage of every intermediate result and `f64`
for sums, `exp`, `ln`, `sqrt`, `cos` and `sin`, one rounding to `f32` per result. Every
sum has a fixed order that depends only on the length of the vector, so a row has
the same bits alone, in a batch, split at any point, or computed with another number
of threads (tests: `a_batch_of_rows_gives_the_bits_of_the_rows_one_at_a_time`,
`splitting_a_prompt_anywhere_gives_the_same_bits`,
`the_number_of_threads_does_not_change_a_bit`, `rejected_rows_leave_no_trace`).
It does not imitate llama.cpp's rounding (K and V in f16, quantized activations).

Speed on the 0.8B on the development machine (16 threads, **measured**, other work
was running on the machine): 32 to 44 tokens per second for a prompt of 230 tokens
and 86 to 165 ms per generated token, depending on the load. It decodes every weight row for every call, so a
generated token costs a full decode of the 0.5 GB of weights and of the 254 M values
of the head.

## 8. Proof against llama.cpp

Everything in this section is **measured** unless it says otherwise. llama.cpp-native
`f8f346786`, CPU only (`CUDA_VISIBLE_DEVICES=""`, `-ngl 0 -dev none`), 6 threads at
`nice -n 10`, with the program `crates/model/tools/capture` (README there). The prompts
are `tools/capture/prompts/{english,german,code}.txt`: 231, 253 and 241 tokens (ids from the
fork's tokenizer: the prebuilt `llama-tokenize` of the fork crashes in
`common_params_parser_init` because it is older than its libraries, so it was compiled
from the fork's `tools/tokenize/tokenize.cpp` against the current libraries). For each
prompt llama.cpp ran the prompt in one batch with every row an output, then picked 40
tokens greedily, one per call, and the capture program saved the output of every block
for every row, the output of the final norm and the logits. Captured data
(811 MB, outside the repository): `/var/tmp/ares-model-capture/`.

The graph of a generation step was listed (`CAP_LIST=1`): 18 `SSM_CONV`, 18
`GATED_DELTA_NET`, 6 `FLASH_ATTN_EXT`, 24 `GLU`, 115 `RMS_NORM`; so the fused delta
rule and flash attention are the paths that ran.

The measure is the relative difference `|reference - llama| / |llama|` over the hidden
state of a block (all values of all rows, L2).

### 8.1 Exactness: the made-up model

A model with f32 weights (`Spec::default()` of `tests/common`, written by
`examples/synthetic`): eight blocks of width 64, three value heads per key head,
two query heads per key/value head, partial rotary with sections [3, 3, 2, 0], an own
output head, 48 prompt tokens and 40 generated. llama.cpp quantizes no activation on f32
weights, and `-fa off -ctk f32 -ctv f32` keeps attention in f32. Then:

| | relative difference |
| --- | ---: |
| each block alone, fed with llama.cpp's own input, recurrent blocks | 1.1e-7 to 4.5e-7 |
| each block alone, attention blocks | 9.1e-8 and 7.4e-8 |
| the chain, block 0 to block 7 (all rows) | 4.5e-7 growing to 5.4e-6 |
| logits, 41 steps | 2.8e-6 to 1.4e-5 (max over steps 3.9e-5) |
| greedy picks of the 48 prompt rows that differ | 0 |
| 40 greedy tokens | identical |

So the equations of section 4, the tiling of the value heads, the grouping of query
heads, the rotary embedding (including partial rotation and the sections), the
convolution, the delta rule, the gates and the head agree with llama.cpp to rounding.
With flash attention on and f16 K/V (the default) the two attention blocks differ by
7.8e-5 and 1.1e-4 instead (**measured**): that is the f16 K/V. The values of this run
are constants in the test
`the_made_up_model_gives_what_llama_cpp_computed_for_it`, so `cargo test` holds the
reference to llama.cpp without llama.cpp.

### 8.2 The 0.8B

Relative difference of the hidden state after each block. "Alone": the reference runs
that block on the hidden state llama.cpp gave it (`Reference::block`), so the numbers
do not accumulate. "Chain": the reference runs the whole prompt itself. "Steps": the
reference is fed the tokens llama.cpp generated, one at a time, and the 40 steps are
averaged. Columns are English / German / code.

| block | kind | alone | chain (prompt) | chain (steps, mean) |
| ---: | --- | --- | --- | --- |
| 0 | delta | 4.5e-2 / 4.4e-2 / 4.7e-2 | 4.5e-2 / 4.4e-2 / 4.7e-2 | 7.4e-2 / 6.8e-2 / 7.1e-2 |
| 1 | delta | 2.5e-2 / 2.3e-2 / 2.4e-2 | 4.9e-2 / 4.9e-2 / 4.8e-2 | 7.2e-2 / 7.2e-2 / 6.9e-2 |
| 2 | delta | 2.7e-2 / 2.5e-2 / 2.7e-2 | 4.7e-2 / 4.9e-2 / 5.0e-2 | 7.7e-2 / 7.2e-2 / 6.6e-2 |
| 3 | attn | 1.8e-2 / 1.9e-2 / 2.0e-2 | 4.8e-2 / 4.9e-2 / 5.1e-2 | 7.0e-2 / 6.9e-2 / 6.8e-2 |
| 4 | delta | 1.6e-2 / 1.7e-2 / 1.9e-2 | 4.4e-2 / 4.7e-2 / 4.7e-2 | 6.6e-2 / 6.3e-2 / 6.1e-2 |
| 5 | delta | 1.6e-2 / 1.6e-2 / 1.9e-2 | 4.1e-2 / 4.5e-2 / 4.6e-2 | 5.6e-2 / 6.2e-2 / 6.0e-2 |
| 6 | delta | 1.4e-2 / 1.3e-2 / 1.6e-2 | 3.6e-2 / 4.1e-2 / 4.1e-2 | 5.2e-2 / 6.0e-2 / 5.3e-2 |
| 7 | attn | 1.6e-2 / 1.5e-2 / 1.7e-2 | 3.9e-2 / 4.3e-2 / 4.5e-2 | 6.1e-2 / 6.3e-2 / 5.1e-2 |
| 8 | delta | 1.2e-2 / 1.3e-2 / 1.4e-2 | 3.9e-2 / 4.3e-2 / 4.5e-2 | 6.5e-2 / 5.9e-2 / 5.0e-2 |
| 9 | delta | 1.0e-2 / 9.6e-3 / 1.2e-2 | 3.9e-2 / 4.2e-2 / 4.4e-2 | 6.0e-2 / 5.7e-2 / 5.0e-2 |
| 10 | delta | 1.1e-2 / 1.0e-2 / 1.2e-2 | 3.6e-2 / 3.9e-2 / 4.1e-2 | 5.2e-2 / 5.4e-2 / 4.9e-2 |
| 11 | attn | 1.3e-2 / 1.2e-2 / 1.4e-2 | 3.9e-2 / 4.2e-2 / 4.5e-2 | 5.3e-2 / 5.7e-2 / 5.0e-2 |
| 12 | delta | 1.1e-2 / 1.1e-2 / 1.3e-2 | 3.8e-2 / 4.1e-2 / 4.4e-2 | 5.2e-2 / 5.5e-2 / 4.6e-2 |
| 13 | delta | 1.1e-2 / 1.2e-2 / 1.3e-2 | 3.7e-2 / 4.0e-2 / 4.3e-2 | 4.7e-2 / 5.5e-2 / 4.3e-2 |
| 14 | delta | 1.3e-2 / 1.4e-2 / 1.5e-2 | 3.4e-2 / 3.7e-2 / 4.0e-2 | 4.6e-2 / 5.4e-2 / 4.3e-2 |
| 15 | attn | 1.5e-2 / 1.5e-2 / 1.7e-2 | 3.6e-2 / 4.0e-2 / 4.3e-2 | 5.3e-2 / 5.4e-2 / 4.2e-2 |
| 16 | delta | 1.2e-2 / 1.5e-2 / 1.7e-2 | 3.6e-2 / 4.0e-2 / 4.2e-2 | 5.1e-2 / 5.3e-2 / 4.2e-2 |
| 17 | delta | 1.0e-2 / 1.2e-2 / 1.3e-2 | 3.5e-2 / 3.9e-2 / 4.2e-2 | 5.1e-2 / 5.2e-2 / 4.2e-2 |
| 18 | delta | 9.2e-3 / 1.2e-2 / 1.3e-2 | 3.3e-2 / 3.6e-2 / 4.0e-2 | 5.1e-2 / 5.1e-2 / 3.9e-2 |
| 19 | attn | 1.5e-2 / 1.6e-2 / 1.5e-2 | 3.6e-2 / 4.0e-2 / 4.2e-2 | 6.4e-2 / 6.6e-2 / 4.0e-2 |
| 20 | delta | 8.4e-3 / 1.5e-2 / 9.9e-3 | 3.5e-2 / 3.9e-2 / 4.1e-2 | 6.2e-2 / 6.8e-2 / 4.1e-2 |
| 21 | delta | 7.8e-3 / 1.3e-2 / 1.0e-2 | 3.5e-2 / 3.9e-2 / 4.1e-2 | 5.8e-2 / 6.5e-2 / 4.2e-2 |
| 22 | delta | 1.0e-2 / 1.4e-2 / 1.2e-2 | 3.7e-2 / 4.2e-2 / 4.4e-2 | 6.3e-2 / 5.9e-2 / 4.5e-2 |
| 23 | attn | 1.4e-2 / 1.6e-2 / 1.5e-2 | 3.9e-2 / 4.7e-2 / 4.9e-2 | 6.3e-2 / 5.7e-2 / 5.5e-2 |

The final norm differs by 4.2e-2 / 5.2e-2 / 5.3e-2 over all prompt rows (English /
German / code) and by 3.9e-2 / 4.4e-2 / 4.3e-2 at the last row. The logits of the last
prompt token differ by 4.1e-2 / 5.4e-2 / 4.5e-2; over the 41 logit rows (the last
prompt token and the 40 steps) the worst relative difference is 6.7e-2 / 5.7e-2 /
6.4e-2.

**Tokens.** The prompts have 231, 253 and 241 tokens and 40 tokens were generated
(at least 32 were asked for).

- With the tokens llama.cpp generated fed in one at a time, the reference picks the same
  next token as llama.cpp at every step for English and code. For German it picks
  another one at two steps, 6 and 21, where llama.cpp's own two best logits are 0.021
  and 0.044 apart.
- The 40 tokens the reference generates by itself are identical to llama.cpp's for
  English and code (40 of 40). For German they are identical for 6 tokens and differ at
  step 6 (reference 5810, llama.cpp 187083, margin 0.021 in llama.cpp's logits), after
  which the two texts differ, as expected. That margin is below the noise of llama.cpp
  against itself (8.3, point 4): the same llama.cpp with 3 threads instead of 6 picks
  5810, the reference's token, at step 6 and then follows another text. This is a tie
  that rounding decides, not a finding.
- On the prompt itself, the greedy pick after a row differs from llama.cpp's in 4 of
  231 rows (English), 15 of 253 (German) and 2 of 241 (code).

### 8.3 What the 4 % is, and how it is known

The reference computes the architecture exactly; llama.cpp on the CPU does not. The
differences of 8.2 are accounted for as follows.

1. **The arithmetic is the same.** On the made-up model, where llama.cpp has nothing to
   approximate, the two agree to 1e-7 per block (8.1).
2. **Quantized activations are the larger part.** In llama.cpp an activation row that
   meets a q4_K, q5_K or q6_K weight is first rounded to `Q8_K` (256 values, one scale
   per block, 127 levels), one that meets q8_0 to `Q8_0` (**read**: `vec_dot_type` in
   `ggml-cpu.c`; **measured**: the node list shows the `MUL_MAT`s with quantized
   weights; the quantizers are `quantize_row_q8_K_ref`/`quantize_row_q8_0`). A
   temporary patch of the reference that applied exactly this rounding before every
   matrix product (not kept; README, section 6) showed the effect of it alone, with
   llama.cpp's own tensors as input: the feed-forward part of block 0 (input
   `attn_post_norm-0` of llama.cpp) differs from llama.cpp by 2.7e-2 without the
   rounding and by 2.1e-7 (median prompt row) with it; the first matrix product of the
   recurrent part, by 4e-3 and 7e-8 (median row). The rows that did not reach rounding
   level were rows where one element sat on a rounding boundary of the quantizer, which
   moves the whole row by about 1e-3. Measured against the update that a block adds
   (`theirs - input`) instead of the whole hidden state, the isolated difference of a
   block is 3.3e-2 to 4.5e-2 for block 0 to 3 (English), so the size is not an
   artefact of a large residual stream.
3. **f16 K and V and flash attention** add to the attention blocks: with flash attention
   off and f32 K/V llama.cpp's block 3 changes by 8.3e-3 / 8.5e-3 / 9.6e-3 and block 23
   by 3.9e-2 / 4.4e-2 / 5.2e-2 (chain, the two llama.cpp runs against each other), while the
   reference's isolated difference to the f32 run is the same as to the default run
   (block 3: 1.85e-2 against 1.84e-2). The f16 effect is the smaller one.
   With `-ctk f32 -ctv f32` and flash attention on, llama.cpp's output is bit-identical to the
   default for all three prompts (the cast of 4.8).
4. **llama.cpp is not equal to itself.** Prefill is bit-identical between 6 and 3
   threads, but generation is not: blocks 0 to 2 are identical and from the first
   attention block on the difference grows, with a mean over the 40 steps of 0.5 to
   0.7 % at block 3 and 1.4 to 2.6 % at block 23 (English and code), a worst step of 4.9 to
   6.0 % at block 23, and
   a relative logit difference up to 4.5e-2 (English) and 9.0e-2 (code); the German
   text is different after step 6 (a 0.021 margin). The source suggests why (**read**,
   not isolated by an experiment): in a generation step the flash attention rounds Q to f16 and accumulates the
   weighted V in f16, and splits the cache over the threads once it has 512 cells
   (the cache is padded), so the result depends on the thread count. This is the same
   size as the reference's difference in the "steps" column, which is why a token that
   differs between the reference and llama.cpp at a margin below that noise is not
   a finding.
5. What is left after 1 to 4 is summation order and the `expf`/`silu` approximations of
   ggml: 1e-7 relative, below everything above.

A GPU implementation can therefore expect the same size of difference to llama.cpp
(1 to 5 % of a block's hidden state, 3 to 7 % of the logits) if it quantizes activations
as llama.cpp does, and 1e-4 per attention block with f16 K and V and exact activations
(8.1). It should be judged against this reference, where the numbers are rounding.

## 9. What was not verified

- **The 27B was not run.** Its configuration and weight table parse (opt-in test), but no
  forward pass was compared. What the 27B adds to the 0.8B is checked on the made-up
  model, which has the same ratio of value heads to key heads (3) and grouped query
  heads (2 per group; the 27B has 6), but not the size, the mix of quantization types
  (the iq2 and iq3 decoders are `ares_quant`'s, tested there against ggml) or the 64
  blocks. The statement that the `softplus`
  input reaches the linear region is from the weights (`dt_bias` up to 19.25), not from
  a run.
- **Positions with different time, height and width** (images) are implemented as the
  general rule of 4.4 with the extra kind at zero, and compared with llama.cpp only
  for text positions. The extra section (`s_3 > 0`) is not exercised by any file.
- **Contexts past 512 cache cells** were not captured (the longest comparison has 271
  positions); the reference has no length-dependent behaviour, llama.cpp's flash
  attention has (4.8).
- **The next-token block** is parsed and its use is read from the fork, not run.
- **f16 K and V on a GPU** were not compared with the reference: only llama.cpp's CPU
  path was.
- **Generation speed of other implementations** and VRAM use were not part of this.

## 10. Sources

| what | where |
| --- | --- |
| hyper-parameters, tensor list, block kinds | `src/models/qwen35.cpp`: `load_arch_hparams`, `load_arch_tensors` |
| the block loop, residuals, output | `qwen35.cpp`: `llama_model_qwen35::graph::graph` |
| recurrent block | `qwen35.cpp`: `build_layer_attn_linear`, `build_qkvz`, `build_norm_gated`; `src/models/delta-net-base.cpp`: `build_conv_state`, `build_recurrent_attn`, `build_delta_net_fused`; `src/models/models.h`: `build_gdn_l2_norm` |
| attention block | `qwen35.cpp`: `build_layer_attn`; `src/llama-graph.cpp`: `build_attn_mha`, `build_norm`, `build_ffn` |
| next-token block | `qwen35.cpp`: `graph_mtp`; `common/speculative.cpp`: `common_speculative_impl_draft_mtp` |
| CPU arithmetic | `ggml/src/ggml-cpu/ops.cpp`: `ggml_compute_forward_gated_delta_net_one_chunk`, `..._ssm_conv_f32`, `..._rms_norm_f32`, `..._l2_norm_f32`, `..._rope_flt`, `ggml_mrope_cache_init`, `..._flash_attn_ext_f16_one_chunk`, `..._tiled`; `unary-ops.cpp`: `op_softplus`, `op_sigmoid`; `ggml-cpu.c`: type traits (`vec_dot_type`) |
| positions of text tokens | `src/llama-batch.cpp` (`[p, p, p, 0]`), `src/llama-graph.cpp`: `llm_graph_input_pos::set_input` |
| rope type, state types | `src/llama-model.cpp`: `llama_model_rope_type`, the hybrid memory (`recurrent_type_r/s`); `src/llama-context.cpp`: `fused_gdn_*`, flash attention |
| what is baked into the tensors | `conversion/qwen.py`: `Qwen3NextModel`, `_LinearAttentionVReorderBase`, `Qwen3_5TextModel` |
| the GPU delta rule of the fork | `ggml/src/ggml-cuda/gated_delta_net.cu`, `gated-delta-net-mma.cu` |
