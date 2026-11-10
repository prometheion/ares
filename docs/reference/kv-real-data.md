# KV formats on real attention data

Measured on 2026-11-09. The queries, keys and values that Qwen3.5-0.8B (all 6
attention layers) and Qwen3.8-27B (5 of its 16 attention layers) feed to
attention were captured from llama.cpp while it read one text of 3 686 tokens.
On them, attention was computed exactly and with the K and V rows that the
engine's formats, kvarn's quantizer and several ideas the engine left out
would store. Before this, the formats had been chosen on the synthetic rows of
`crates/quant/tests/kv_techniques.rs` only.

Not validated (details in section 11): the text is one text of 3.7k tokens and
the models are one family; the measure is the error of attention (output,
weights, scores), not logits, perplexity or task scores; the engine was
measured through its CPU reference codec and kvarn through a re-implementation
that was checked against its CPU reference routines (the CUDA kernels were
read, not run); the exact suffix of kvarn is modelled; nothing about speed was
measured.

## Summary

- **The engine's ladder holds on real data.** Against the old absolute-maximum
  scale, with nothing exact, the output error of attention is 12%, 15%, 18%
  lower at 6, 5, 4 bits, the KL of the weights 26%, 31%, 37% lower, the SNR
  0.8, 1.0, 1.6 dB higher, the same SNR gain as on the synthetic rows. It wins
  in every layer, every part of the context and every kind of text, and the
  stress case ("a large mean in every group") does not occur: no group of 32
  has even half of its energy in its mean (fewer than 0.01% of the groups in
  either model).
- **Exact first and newest cells are the largest effect, and the newest matter
  most.** At q4/q4 the `KvPolicy` defaults cut the output error by 76% and the
  KL by 66%. Of the gain from the absolute-maximum scale without exact cells to
  the engine or kvarn with kvarn's tail (6 to 4 bits), 83% to 92% of the output
  error and 49% to 66% of the KL come from the exact cells; of the error of the
  scores, 17% to 24% at 4 bits. 45% to 59% of the attention weight sits on the
  newest 128 cells. The first 4 cells give 87% to 94% of the output-error gain
  of the first 128; the newest window keeps paying (each doubling from 128
  cells lowers the q4 error by 28%, 41%, 57%, on a context of only 3.7k).
- **kvarn's quantizer is better on K and worse on V, and does not beat the
  engine's on the attention output.** On K it gains SNR +1.1 dB and 15% KL at 4
  bits (same tail); on V it loses 2.2 dB. Together the output error is within 6%
  of the engine's at equal widths and 12% to 14% higher when V is narrower than
  K; only the weights are better (KL 12% to 34% lower, score error 10% to 20%
  lower).
- **Of the ideas left out, only a mean per channel is worth its cost; the
  two-axis normalisation is not.** The normalisation (tiles of 128 x 128, 16
  Sinkhorn iterations, +0.25 bit) changes the engine's SNR by 0.0 to -0.1 dB
  and the KL by +3% to +4% on the 0.8B (+2% to +18% on the 27B, SNR down by up
  to 0.7 dB). A static mean per channel from the first 256 cells (no storage,
  no read work per cell) lowers the KL by 17% and the output error by 16% at
  q4/q4. A mean per tile of 128 cells (0.125 bit per value; one read term per
  tile) lowers the KL by 49% and the output error by 31% at q4/q4, 48% and 26%
  at q5/q5; on the 27B 44% and 24% at q4/q4. The mean per tile has since been
  built into `ares-quant` (`q7` to `q4`); section 13 measures it as built:
  under the engine's tile policy it lowers the output error by 25% to 32% and
  the KL by 48% to 49% (K6/V6 to K4/V4, 0.8B) against the same exact cells with
  rows, and makes no layer, depth or text kind worse.
- **kvarn's gain on K is mostly its per-channel offset.** The static mean
  matches kvarn's K-side KL gain (50.8 against 52.1 at q4/q4 with its tail) and
  the mean per tile beats it by far (31.2).

## 1. What was measured

**Models.** `Qwen3.5-0.8B-Q4_K_M.gguf` (architecture `qwen35`: 24 layers, every
fourth one attention; 8 query heads, 2 KV heads, head size 256; captured layers
3, 7, 11, 15, 19, 23) and `Qwen3.8-27B-UD-Q4_K_S.gguf` (`qwen35`, 64 layers, 16
attention; 24 query heads, 4 KV heads, head size 256; captured layers 3, 15,
31, 47, 63). RoPE rotates 64 of the 256 values. Both ran on the CPU of the
build machine, no GPU layers and no CUDA device visible.

**Capture.** A small program against `libllama` of
`~/Services/llama.cpp-native` (read only, built outside the
checkout) sets the scheduler's eval callback and copies, per attention layer,
the graph nodes that feed `build_attn` (`src/models/qwen35.cpp`): `Qcur-N` and
`Kcur-N` (the RoPE nodes, after projection, RMS norm and rotation) and
`Vcur-N` (the reshaped projection), as f32 per position and head. It also
copies `attn_pregate-N`, the model's own attention output, for a check. The
text was prefilled in physical batches of 512 tokens with an f16 KV cache; the
captured tensors are the graph's, before they enter the cache.

**Text.** One sequence of 3 686 tokens in the ChatML format of the models,
written for this test: a system prompt (tokens 0 to 97), English prose and its
summary (98 to 963), German prose and its summary (964 to 1703), Python and
Rust source with a review (1704 to 2606), short German and English chat turns
(2607 to 3008), an English post-mortem draft with an answer and a last
question (3009 to 3685). The first tokens are a system prompt and the newest
tokens are ordinary text of all four kinds. The 27B read the same tokens (the
tokenizer is shared); the CPU run took 6 minutes, so a longer text than "a few
hundred tokens" was affordable.

**Queries.** For every captured layer, query head and every second position
from 512 on (below 512 the policies leave little to quantize), the exact
attention is computed over all earlier positions with the captured f32 Q, K and
V: plain softmax, scale 1/sqrt(256), query head `h` on KV head `h / (n_head /
n_kv)`. Samples per candidate: 76 176 for the 0.8B (1 587 positions x 8 heads x
6 layers), 95 280 for the 27B (794 x 24 x 5, every fourth position).

**Metrics**, per (layer, head, position), then averaged over all samples (or
the 99th percentile taken):

- *rel L2*: `|o' - o| / |o|` for the output `o` of one head, before the output
  gate and projection.
- *KL*: KL(p || p') of the attention weights over the cache, in nats. K only.
- *score*: the standard deviation over the keys of the error of the scaled
  scores `q.k / 16`, in nats of log-weight (a shift common to all keys does not
  count). K only.
- *SNR*: 10 log10 of the energy of the rows over the energy of the error of the
  stored rows, on whole groups of 128 rows, per side, as the mean of the
  per-layer values.

**Candidates.** K and V at 8, 6, 5 and 4 bits and the mixed pairs K6/V5 and
K5/V4 (written `Kb/Vb`).

- *engine*: `encode_row`/`decode_row` of `ares-quant` as shipped (rotation of
  32, scale search, fitted scale, sign of the scale, `2 + 4B` bytes per group).
- *absmax*: the engine's rotation and layout with the old absolute-maximum scale
  (`BASE` of `kv_techniques.rs`: scale from the largest magnitude rounded up to
  f16, symmetric codes). At 8 bits it is `q8`, bit for bit. *absmax, no
  rotation* also leaves out the rotation of 32: a plain `q8_0`/`q5_0`-like type
  without llama.cpp's own rotation.
- *kvarn quantizer*: the quantizer of beellama.cpp `1cef3bf`, re-implemented
  (section 2): one Walsh-Hadamard transform over the head (256), values rounded
  to f16 as the stage holds them, tiles of 128 cells x 128 values (K as
  `[channel][cell]`, V as `[cell][channel]`), 16 Sinkhorn iterations dividing
  columns and rows by their standard deviations (the iterate with the least
  max/min spread is kept), per row a grid from its minimum to its maximum with
  `2^B` levels, three f16 vectors per tile; `B + 0.375` bits per value.
- *what-ifs on the engine* (section 6): static channel scale, static channel
  mean, kvarn's two-axis tile normalisation in front of the engine's codec (in
  both orders), a midrange offset per group, a mean per tile.

**Which cells are exact** (kept as f16; for kvarn as f16 of the rotated head):

| name | first cells | newest cells |
| --- | --- | --- |
| nothing exact | 0 | 0 |
| first 128 + open group | 128 | the group of 128 being filled, 1 to 128 cells: kvarn's intrinsic exact suffix (a group is quantized when its last cell is written; group 0 never is) |
| first 128 + 2 groups | 128 | the open group and the one before it, 129 to 256 cells |
| first 128 + newest 128 | 128 | 128, sliding (the semantics of the engine's `KvPolicy`) |
| `KvPolicy` defaults | 128, 64, 32 at q4, q5, q6 | 256, 128, 64, sliding; a mixed pair takes the larger numbers of its two formats; none at q8 |

"Nothing exact" for kvarn (marked (*) in the tables) means that every whole
tile is quantized from statistics that include cells newer than the query.
kvarn cannot do that (an open tile cannot be normalised), so these entries show
the quantizer alone and are slightly favourable to it; the entries with the
open group are causal.
Exact cells apply to K and V together, as in `KvPolicy`.

## 2. Checks of the tool

| check | result |
| --- | --- |
| captured Q, K, V reproduce the model's own attention output (`attn_pregate`) with query head `h` on KV head `h / group` (this also checks the reference attention: scale, grouping, causal mask) | relative L2 difference 0.9e-4 to 4.0e-4 on average per layer, at most 1.8e-3 (the model reads an f16 cache); with the grouping `h % n_kv` it is 0.66 to 1.08, so the grouping is right (both models) |
| `absmax` at 8 bits against `ares-quant` `q8` | 943 616 of 943 616 values bit-identical, for K and for V |
| kvarn's head transform (256 values) against `apply_reference_kvarn_wht_head` of beellama's tests, on 64 real K rows | 16 384 of 16 384 values bit-identical |
| kvarn's tile quantizer against `llama_kvarn_quantize_*_tile` and `llama_kvarn_dequantize_*_tile`, compiled unchanged from `src/llama-kvarn.cpp` and run on 384 real tiles (K and V; 4, 5, 6, 8 bits; layers 3, 15, 23 of the 0.8B) | all 6.3 million codes, all 3 x 128 f16 side values per tile and all dequantized values bit-identical |

The original code compiled and ran on the CPU, so the re-implementation is
validated against it. The production path is the CUDA kernel `kvarn.cu`; its
`kvarn_quantize_tile` was read and follows the same steps (f16 stage, identity
scales as the first candidate, same clamps and rounding), but was not run.

## 3. The real data against the synthetic assumptions

Statistics per KV head, averaged over the heads and then over the layers
(range over the layers in brackets for the 0.8B). "Flat" is the value of a
spectrum without structure.

| statistic | K, 0.8B | V, 0.8B | K, 27B | V, 27B |
| --- | --- | --- | --- | --- |
| rms of the values | 1.52 | 0.92 (0.50 to 2.29) | 1.46 | 1.76 |
| share of the energy in the per-channel means over the sequence | 31% (21 to 36) | 37% (18 to 91) | 27% | 26% |
| share of the energy in the 8 loudest channels of 256 (flat 3.1%) | 18% | 21% | 19% | 15% |
| loudest channel rms / median channel rms | 4.4 | 5.4 | 4.6 | 3.9 |
| overlap of the 16 loudest channels of the first 256 cells with those of the rest | 70% | 70% | 64% | 69% |
| correlation of the log channel rms, first 256 cells against the rest | 0.87 | 0.86 | 0.84 | 0.82 |
| coefficient of variation of the row norm | 1.8% | 10.8% | 1.9% | 16.7% |
| norm of cell 0 / median row norm | 1.03 | 1.81 (0.65 to 3.0) | 0.96 | 1.37 |
| excess kurtosis of the values: raw / after the rotation of 32 | 2.3 / 0.19 | 6.9 / 0.27 | 2.6 / 0.39 | 5.7 / 0.56 |
| groups of 32 whose largest value holds over half the energy: raw / rotated | 2.6% / 0.01% | 8.3% / 0.00% | 4.9% / 0.01% | 2.7% / 0.01% |
| groups with over half / over a quarter of the energy in their mean (rotated value 0) | 0% / 0.3% | 0% / 0.2% | 0% / 0.2% | 0% / 0.3% |
| mean energy share of the rotated value 0 (flat 3.1%) | 3.2% | 3.1% | 3.0% | 3.1% |
| lag-1 correlation of a channel (mean removed) between neighbouring cells | 0.55 | 0.38 | 0.51 | 0.36 |

What this says for the assumptions of `kv.rs`:

- **The "large mean in every group" stress case does not occur.** No group of
  32 has half of its energy in its mean in any layer of either model; the mean
  has 3.0% to 3.2% of a group's energy, which is what a flat spectrum gives.
  The eight group means carry 1.2% to 2.2% of the variance of the scores over
  the keys, the single rotated value with the largest share 1.8% to 5.0%: no
  score is dominated by one rotated value or by a common offset. The engine's
  scale search never meets the case in which it was known to lose, and it does
  not lose in any layer (section 5).
- **Values are less peaky than the synthetic outlier rows.** The rotation is
  worth 1.7 to 2.4 dB here (8 bits: K 43.8 against 45.6 dB, V 43.3 against
  45.7), against 7 to 9 dB on "Gaussian with outliers". Raw groups of V are
  peaky (kurtosis 6.9; in layer 19 of the 0.8B 26% of the groups have over half
  of their energy in one value) and the rotation makes them Gaussian (kurtosis
  0.2 to 0.6, no group dominated by one value).
- **The gain of the ladder over the old scale is as on synthetic rows.** SNR of
  engine over absmax: 0.0, 0.8, 1.0, 1.6 dB at 8, 6, 5, 4 bits for K and V; the
  module documentation has 0.0, 0.8, 1.2, 1.6. Absolute SNR is about 1.7 dB below
  the synthetic Gaussian rows (engine, K: 45.6, 34.2, 28.1, 22.0 against 47.3,
  35.8, 29.9, 23.7).
- **A few channels are persistently loud, and much of the energy sits in
  channel means.** 15% to 21% of the energy is in the 8 loudest of 256
  channels, 23% to 30% in the 16 loudest. Their identity is mostly stable (64%
  to 70% of the loudest 16 of the first 256 cells are still among the loudest
  16 of the rest; the log rms profiles correlate at 0.82 to 0.87). In K the
  loud channels lean towards the 64 RoPE-rotated values (38% to 40% of the
  loudest 16, against a quarter of the channels). 26% to 37% of the energy of K
  and V is the mean of their channels, and neighbouring cells are correlated
  (lag-1 0.36 to 0.55 with the mean removed), so the mean also drifts locally.
- **K and V differ.** K rows have a constant norm (RMS norm: variation 2%), V
  rows do not (11% to 17%; cell 0 of V is 2.6 to 3.0 times the median row norm in
  three of the six layers of the 0.8B and 2.0 to 2.2 times in two of the five
  of the 27B). V is heavier-tailed before the rotation, K more correlated
  between neighbouring cells. In the last attention
  layer of the 0.8B 91% of V's energy is its channel means, in the last one of
  the 27B 81%.
- **Where the attention weight goes** (exact attention, queries from 512): the
  newest 128 cells hold 45% to 59% of it in every captured layer of both
  models, the newest 1 024 cells 79% to 89%; the first 128 cells hold 4% to 14%
  and, in most layers, nearly all of that is in the first 1 to 4 cells (cell 0
  alone: 0.3% to 11%).

0.8B, per layer (all queries from 512):

| layer | weight on the first 128 cells | weight on the newest 128 cells | entropy (nats) |
|---|---|---|---|
| L03 | 0.043 | 0.572 | 4.35 |
| L07 | 0.076 | 0.582 | 3.45 |
| L11 | 0.090 | 0.474 | 3.97 |
| L15 | 0.094 | 0.492 | 3.90 |
| L19 | 0.056 | 0.488 | 4.30 |
| L23 | 0.066 | 0.537 | 3.69 |

## 4. Qwen3.5-0.8B: engine, absolute maximum and kvarn

Mean rel L2 (x 1e-3), lower is better. First the quantizers with nothing exact,
then with exact cells.

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| absmax, no rotation; nothing exact | 5.83 | 23.22 | 43.87 | 89.51 | 34.22 | 67.25 |
| absmax; nothing exact | 3.62 | 14.99 | 31.10 | 65.18 | 22.95 | 48.96 |
| engine; nothing exact | 3.62 | 13.15 | 26.43 | 53.64 | 20.19 | 41.40 |
| kvarn quantizer; nothing exact (*) | 3.65 | 14.58 | 29.44 | 59.43 | 23.87 | 49.00 |
| absmax; first 128 + open group | 1.47 | 5.85 | 12.13 | 26.39 | 8.75 | 18.64 |
| engine; first 128 + open group | 1.47 | 5.16 | 10.48 | 23.09 | 7.78 | 16.30 |
| kvarn quantizer; first 128 + open group (kvarn as built) | 1.39 | 5.40 | 10.99 | 22.81 | 8.85 | 18.27 |
| engine; first 128 + 2 groups | 1.00 | 3.39 | 6.95 | 15.36 | 5.08 | 10.71 |
| kvarn quantizer; first 128 + 2 groups | 0.93 | 3.54 | 7.18 | 14.97 | 5.76 | 11.86 |
| absmax; first 128 + newest 128 | 1.15 | 4.50 | 9.32 | 20.36 | 6.72 | 14.27 |
| engine; first 128 + newest 128 | 1.15 | 3.98 | 8.13 | 17.95 | 5.98 | 12.58 |
| kvarn quantizer; first 128 + newest 128 | 1.08 | 4.15 | 8.44 | 17.55 | 6.78 | 13.98 |
| absmax; KvPolicy defaults | 3.62 | 5.60 | 9.55 | 14.47 | 6.90 | 9.99 |
| engine; KvPolicy defaults | 3.62 | 4.94 | 8.34 | 12.85 | 6.14 | 8.86 |
| kvarn quantizer; KvPolicy defaults | 3.65 | 5.17 | 8.67 | 12.40 | 6.97 | 9.73 |

Mean KL of the attention weights (x 1e-4). It does not depend on V's width, so
K6/V5 equals K6/V6 and K5/V4 equals K5/V5.

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| absmax, no rotation; nothing exact | 0.86 | 14.38 | 58.20 | 260.86 | 14.38 | 58.20 |
| absmax; nothing exact | 0.51 | 8.57 | 36.75 | 166.15 | 8.57 | 36.75 |
| engine; nothing exact | 0.51 | 6.32 | 25.44 | 104.10 | 6.32 | 25.44 |
| kvarn quantizer; nothing exact (*) | 0.33 | 5.26 | 21.81 | 91.92 | 5.26 | 21.81 |
| absmax; first 128 + open group | 0.29 | 4.82 | 20.60 | 93.86 | 4.82 | 20.60 |
| engine; first 128 + open group | 0.29 | 3.60 | 14.55 | 61.53 | 3.60 | 14.55 |
| kvarn quantizer; first 128 + open group (kvarn as built) | 0.19 | 2.94 | 12.14 | 52.08 | 2.94 | 12.14 |
| engine; first 128 + 2 groups | 0.19 | 2.44 | 9.91 | 42.01 | 2.44 | 9.91 |
| kvarn quantizer; first 128 + 2 groups | 0.13 | 1.99 | 8.17 | 35.20 | 1.99 | 8.17 |
| absmax; first 128 + newest 128 | 0.23 | 3.78 | 16.16 | 73.72 | 3.78 | 16.16 |
| engine; first 128 + newest 128 | 0.23 | 2.84 | 11.53 | 48.78 | 2.84 | 11.53 |
| kvarn quantizer; first 128 + newest 128 | 0.15 | 2.31 | 9.53 | 41.02 | 2.31 | 9.53 |
| absmax; KvPolicy defaults | 0.51 | 4.76 | 16.76 | 53.36 | 3.92 | 11.69 |
| engine; KvPolicy defaults | 0.51 | 3.57 | 11.97 | 35.59 | 2.95 | 8.40 |
| kvarn quantizer; KvPolicy defaults | 0.33 | 2.92 | 9.94 | 29.68 | 2.40 | 6.87 |

99th percentile of rel L2 (x 1e-3):

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| absmax; nothing exact | 10.33 | 43.68 | 92.88 | 184.77 | 50.44 | 108.77 |
| engine; nothing exact | 10.33 | 37.32 | 74.54 | 144.22 | 43.07 | 87.01 |
| kvarn quantizer; nothing exact (*) | 10.26 | 42.64 | 82.78 | 158.05 | 53.00 | 108.80 |
| absmax; first 128 + open group | 5.94 | 24.84 | 51.10 | 109.67 | 37.57 | 79.64 |
| engine; first 128 + open group | 5.94 | 21.10 | 41.73 | 87.74 | 32.65 | 65.26 |
| kvarn quantizer; first 128 + open group (kvarn as built) | 5.65 | 22.60 | 45.75 | 92.86 | 38.78 | 79.99 |
| absmax; KvPolicy defaults | 10.33 | 24.26 | 46.39 | 86.30 | 35.54 | 64.78 |
| engine; KvPolicy defaults | 10.33 | 20.54 | 38.67 | 69.88 | 30.83 | 52.98 |
| kvarn quantizer; KvPolicy defaults | 10.26 | 22.28 | 43.24 | 77.62 | 37.04 | 64.66 |

Mean error of the scaled scores (x 1e-3 nats):

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| absmax; nothing exact | 10.88 | 44.56 | 91.96 | 197.35 | 44.56 | 91.96 |
| engine; nothing exact | 10.88 | 40.18 | 80.64 | 162.69 | 40.18 | 80.64 |
| kvarn quantizer; nothing exact (*) | 8.74 | 34.81 | 70.60 | 146.44 | 34.81 | 70.60 |
| absmax; first 128 + open group | 10.24 | 41.92 | 86.52 | 185.63 | 41.92 | 86.52 |
| engine; first 128 + open group | 10.24 | 37.80 | 75.98 | 153.42 | 37.80 | 75.98 |
| kvarn quantizer; first 128 + open group (kvarn as built) | 8.18 | 32.63 | 66.15 | 137.25 | 32.63 | 66.15 |
| absmax; KvPolicy defaults | 10.88 | 43.24 | 86.34 | 171.66 | 41.83 | 80.02 |
| engine; KvPolicy defaults | 10.88 | 39.00 | 75.82 | 142.26 | 37.74 | 70.38 |
| kvarn quantizer; KvPolicy defaults | 8.74 | 33.75 | 66.14 | 126.90 | 32.64 | 61.14 |

### 4.1 The engine's formats against the absolute maximum

With nothing exact the engine's scale search lowers the output error by 12%
(6 bits), 15% (5) and 18% (4) against the same codec with the absolute-maximum
scale, the 99th percentile of the output error by 15%, 20%, 22%, the KL by 26%,
31%, 37% and the score error by 10%, 12%, 18%. Without the rotation of 32 (but
with the absolute-maximum scale) the output error at 4 bits is 89.5 against
65.2 and the KL 261 against 166, so the rotation is worth more than the search
at 4 bits. The search wins in every layer, every part of the context and every
kind of text (section 5), and also with exact cells (11% to 13% lower output
error with `KvPolicy` defaults). On this data the search never loses to the
absolute-maximum scale in either measure.

### 4.2 kvarn's quantizer against the engine's

- With nothing exact the output error of kvarn is 11% higher at 6 to 4 bits
  (18% higher for K6/V5, K5/V4) and its KL 17%, 14%, 12% lower.
- With kvarn's tail (first 128 + open group) the output error of kvarn is 5%
  lower at 8 bits, 5% higher at 6 and 5 bits, 1% lower at 4 bits (22.8 against
  23.1), and 14% and 12% higher for K6/V5 and K5/V4; the KL is 34%, 18%, 17%,
  15% lower at 8, 6, 5, 4 bits.
- The two sides differ (section 7). On K kvarn's quantizer is better than the
  engine's, by 1.1 dB SNR and 15% KL at 4 bits; on V it is worse, by 2.2 dB SNR
  and 6% output error at 4 bits.
- The score error of kvarn is 10% to 20% below the engine's at every width, with
  and without tail.

The author's figures say that most of kvarn's gain at 4 bits comes from the
exact cells. For the attention output that holds here, for the weights only in
part. Of the log-gain from absmax without exact cells to the engine or kvarn
with kvarn's tail, the share from the exact cells and from the quantizer is

| pair | absmax, nothing exact | engine, kvarn tail | kvarn, kvarn tail | engine: share of the log-gain from exact cells / from the quantizer | kvarn: the same |
|---|---|---|---|---|---|
| K8/V8 | 3.62 | 1.47 | 1.39 | 100% / 0% | 94% / 6% |
| K6/V6 | 14.99 | 5.16 | 5.40 | 88% / 12% | 92% / 8% |
| K5/V5 | 31.10 | 10.48 | 10.99 | 87% / 13% | 91% / 9% |
| K4/V4 | 65.18 | 23.09 | 22.81 | 87% / 13% | 86% / 14% |
| K6/V5 | 22.95 | 7.78 | 8.85 | 89% / 11% | 101% / -1% |
| K5/V4 | 48.96 | 16.30 | 18.27 | 88% / 12% | 98% / 2% |

for the output error, and

| pair | absmax, nothing exact | engine, kvarn tail | kvarn, kvarn tail | engine: share of the log-gain from exact cells / from the quantizer | kvarn: the same |
|---|---|---|---|---|---|
| K8/V8 | 0.51 | 0.29 | 0.19 | 100% / 0% | 57% / 43% |
| K6/V6 | 8.57 | 3.60 | 2.94 | 66% / 34% | 54% / 46% |
| K5/V5 | 36.75 | 14.55 | 12.14 | 62% / 38% | 52% / 48% |
| K4/V4 | 166.15 | 61.53 | 52.08 | 57% / 43% | 49% / 51% |
| K6/V5 | 8.57 | 3.60 | 2.94 | 66% / 34% | 54% / 46% |
| K5/V4 | 36.75 | 14.55 | 12.14 | 62% / 38% | 52% / 48% |

for the KL. For the score error the exact cells give 17% to 24% at 4 bits
(the quantizer, K above all, the rest). The weights are mostly improved by a
better K, the output mostly by the exact cells because half of the weight sits
on the newest 128 cells, whose V values are then exact.

### 4.3 What exact cells buy

Output error (x 1e-3) of the engine's q4/q4 as a function of the first and the
newest cells kept exact (sliding window; `KvPolicy` default at q4: 128 and 256):

| first cells exact \ newest cells exact | 0 | 32 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| 0 | 53.64 | 33.61 | 28.24 | 24.42 | 20.76 | 14.99 |
| 4 | 49.69 | 25.76 | 18.74 | 13.74 | 8.58 | 4.01 |
| 16 | 49.63 | 25.67 | 18.65 | 13.63 | 8.46 | 3.90 |
| 64 | 49.54 | 25.51 | 18.47 | 13.44 | 8.25 | 3.72 |
| 128 | 49.24 | 25.04 | 17.95 | 12.85 | 7.61 | 3.30 |
| 256 | 48.84 | 24.38 | 17.18 | 11.97 | 6.87 | 3.04 |

KL (x 1e-4), same grid:

| first cells exact \ newest cells exact | 0 | 32 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| 0 | 104.10 | 72.96 | 55.23 | 42.22 | 28.34 | 15.18 |
| 4 | 101.34 | 69.61 | 51.69 | 38.57 | 24.58 | 12.03 |
| 16 | 101.09 | 69.33 | 51.40 | 38.27 | 24.28 | 11.82 |
| 64 | 100.38 | 68.54 | 50.60 | 37.46 | 23.47 | 11.23 |
| 128 | 98.77 | 66.77 | 48.78 | 35.59 | 21.67 | 10.17 |
| 256 | 96.10 | 63.86 | 45.77 | 32.50 | 19.41 | 9.22 |

- Exact cells are the largest single effect. q4/q4 with the `KvPolicy` default
  (128 first, 256 newest) has an output error of 12.85 against 53.6 with nothing
  exact (-76%) and a KL of 35.6 against 104.1 (-66%). At q5 and q6 the defaults
  (64/128, 32/64) cut the output error by 68% and 62%. At q8 the default is no
  exact cells; 128 + 128 would take the output error from 3.6 to 1.2 (x 1e-3).
- **The first cells matter, but 4 give most of it.** With the newest 128 cells
  exact, the first 4 cells cut the output error from 28.2 to 18.7, 128 cells to
  17.95, 256 to 17.2. The first 4 cells give 92% to 94% of the gain of the
  first 128 for newest windows of 128 to 1 024. The cells 5 to 128 are worth
  4% of the remaining error with a window of 128 and 18% with a window of
  1 024. The KL asks for more first cells: with the newest 128 exact, 4 cells
  give 55% of the KL gain of the first 128, and the KL still falls by 4% from 64
  to 128 first cells and by 6% from 128 to 256.
- **The newest cells matter more and keep paying.** With the first 128 cells
  exact the output error is 25.0, 17.95, 12.85, 7.61, 3.30 for windows of 32,
  128, 256, 512, 1 024 (28%, 41%, 57% less for each doubling from 128). This
  context is short: at position 2 000 a window of 1 024 covers half the cache, so
  the large windows show the error of what is left, not a rate for long
  contexts.
- A cell kept exact costs, per attention layer, 2 048 bytes for the 0.8B (2 KV
  heads of 256, K and V in f16) against 576 at q4/q4, and for the 27B (4 KV
  heads) 4 096 against 1 152. 128 first cells cost 6 MiB per sequence for the
  27B (16 attention layers); 1 024 newest cells 46 MiB (for the 0.8B: 1.1 and 8.6
  MiB). For comparison 64k cells of q4/q4 in one layer of the 0.8B are 36 MiB.
- kvarn's tail (first 128, open group) is a weaker tail than the engine's
  defaults: at q4/q4 the output error is 23.1 against 12.9 (the open group is
  64 cells wide on average; with two groups 15.4).

## 5. Where the differences are: layers, depth, text

Output error (x 1e-3) at q4/q4 per layer:

| candidate | L03 | L07 | L11 | L15 | L19 | L23 | all |
|---|---|---|---|---|---|---|---|
| absmax; nothing exact | 61.02 | 70.58 | 73.06 | 83.44 | 64.34 | 38.64 | 65.18 |
| engine; nothing exact | 50.87 | 58.44 | 59.15 | 66.50 | 51.53 | 35.36 | 53.64 |
| kvarn; nothing exact | 50.64 | 68.74 | 69.77 | 70.50 | 56.86 | 40.04 | 59.43 |
| engine; first 128 + open group | 19.32 | 20.92 | 27.84 | 30.74 | 29.62 | 10.14 | 23.09 |
| kvarn; first 128 + open group | 18.57 | 22.11 | 26.80 | 28.38 | 31.07 | 9.91 | 22.81 |

KL (x 1e-4) at q4/q4 per layer:

| candidate | L03 | L07 | L11 | L15 | L19 | L23 | all |
|---|---|---|---|---|---|---|---|
| absmax; nothing exact | 190.67 | 162.42 | 177.80 | 191.85 | 164.40 | 109.79 | 166.15 |
| engine; nothing exact | 126.37 | 99.80 | 106.93 | 118.92 | 104.75 | 67.80 | 104.10 |
| kvarn; nothing exact | 94.33 | 104.53 | 104.26 | 98.31 | 86.15 | 63.94 | 91.92 |
| engine; first 128 + open group | 67.14 | 53.18 | 66.94 | 72.19 | 65.57 | 44.13 | 61.53 |
| kvarn; first 128 + open group | 49.94 | 53.84 | 60.32 | 55.96 | 51.61 | 40.83 | 52.08 |

By position of the query and by kind of text, q4/q4, output error:

| candidate | t512-1023 | t1024-2047 | t2048+ | english | german | code |
|---|---|---|---|---|---|---|
| absmax; nothing exact | 65.79 | 68.63 | 62.83 | 63.13 | 66.27 | 66.83 |
| engine; nothing exact | 54.53 | 56.21 | 51.76 | 52.45 | 54.71 | 54.12 |
| kvarn; nothing exact | 60.70 | 62.99 | 56.80 | 57.21 | 60.58 | 61.23 |
| engine; first 128 + open group | 21.53 | 21.77 | 24.41 | 23.23 | 22.14 | 23.97 |
| kvarn; first 128 + open group | 21.36 | 21.43 | 24.12 | 22.80 | 21.50 | 24.27 |
| engine; KvPolicy defaults | 11.83 | 10.87 | 14.42 | 12.51 | 12.66 | 13.55 |

KL:

| candidate | t512-1023 | t1024-2047 | t2048+ | english | german | code |
|---|---|---|---|---|---|---|
| absmax; nothing exact | 149.63 | 166.04 | 171.39 | 163.32 | 172.22 | 163.35 |
| engine; nothing exact | 94.14 | 102.97 | 107.91 | 103.94 | 107.73 | 100.26 |
| kvarn; nothing exact | 84.75 | 92.72 | 93.66 | 88.97 | 95.71 | 91.81 |
| engine; first 128 + open group | 49.97 | 57.14 | 67.88 | 62.69 | 63.37 | 57.85 |
| kvarn; first 128 + open group | 41.32 | 49.15 | 57.28 | 51.87 | 54.70 | 49.47 |
| engine; KvPolicy defaults | 26.44 | 30.73 | 41.49 | 36.04 | 37.22 | 33.14 |

- In every layer the engine beats absmax in output error and in KL. With
  nothing exact kvarn has a higher output error than the engine in five of six
  layers and a lower KL in five. With its tail kvarn is ahead in output error in
  layers 3, 11, 15, 23 and behind in 7 and 19, by at most 8% either way.
- With nothing exact the output error is flat over the context (54.5, 56.2,
  51.8 for the engine); with a tail it grows from 21.5 at positions 512 to 1 023
  to 24.4 from 2 048 on; the KL grows with depth (engine with kvarn's tail 50.0, 57.1,
  67.9) because a smaller share of the cells is exact.
- English, German and code differ by 4% to 13% in output error and by 5% to 12%
  in KL for the six candidates of the tables above; with other tails by up to
  27%, since where a segment sits in the context decides how many cells are
  exact. No comparison of engine, absmax and kvarn changes sign between the
  three kinds of text.

## 6. What the engine left out: ideas on top of its format

Each idea is applied to K and V and put in front of the engine's codec
(rotation, scale search, group layout unchanged); the exact cells are kvarn's
(first 128, open group) so that tile ideas are comparable. "Static" statistics
come from the first 256 cells of a head (earlier cells are stored plain, so the
idea is causal), "tile" statistics from the cells of the tile (known when it is
full). Storage in bits per value on top of `B + 0.5`: scale 0, mean 0, tile
normalisation 0.25, group offset 0.5, tile mean 16 x 256 / (cells x 256) =
`16 / cells` (0.125 for 128, 0.5 for 32, 0.03 for 512). A dash in a table is a
candidate that was not run for that pair.

Mean rel L2 (x 1e-3):

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| engine | 1.47 | 5.16 | 10.48 | 23.09 | 7.78 | 16.30 |
| kvarn quantizer | 1.39 | 5.40 | 10.99 | 22.81 | 8.85 | 18.27 |
| + static channel scale (first 256 cells) | 1.69 | 6.01 | 12.30 | 25.91 | 8.71 | 18.08 |
| + static channel scale^0.5 | 1.41 | 4.86 | 9.97 | 22.00 | 7.33 | 15.37 |
| + static channel mean (first 256 cells) | 1.31 | 4.60 | 9.29 | 19.39 | 6.81 | 13.89 |
| + static channel mean (first 128 cells) | 1.33 | 4.68 | 9.52 | 19.74 | 6.89 | 14.17 |
| + static channel mean (first 512 cells) | 1.30 | 4.59 | 9.23 | 19.54 | 6.83 | 13.95 |
| + static channel mean (whole sequence; not causal) | 1.19 | 4.17 | 8.43 | 17.69 | 6.33 | 12.90 |
| + static mean and scale^0.5 | 1.27 | 4.44 | 8.95 | 18.74 | 6.68 | 13.63 |
| + two-axis tile normalisation, original domain | 1.50 | 5.30 | 10.86 | 23.57 | 7.90 | 16.58 |
| + two-axis tile normalisation, after group rotation | 1.49 | 5.21 | 10.69 | 23.04 | 7.82 | 16.45 |
| + group midrange offset | 1.31 | 4.84 | 10.15 | 22.83 | 7.37 | 15.98 |
| + mean of the previous 128 cells | - | 4.28 | 8.60 | 17.75 | - | 13.08 |
| + mean of the tile, tiles of 512 | - | 3.99 | 8.04 | 16.59 | - | 12.42 |
| + mean of the tile, tiles of 256 | - | 3.92 | 7.89 | 16.24 | - | 12.22 |
| + mean of the tile, tiles of 128 | - | 3.85 | 7.72 | 15.95 | - | 12.03 |
| + mean of the tile, tiles of 64 | - | 3.73 | 7.48 | 15.34 | - | 11.73 |
| + mean of the tile, tiles of 32 | - | 3.58 | 7.17 | 14.71 | - | 11.31 |

Mean KL (x 1e-4):

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| engine | 0.29 | 3.60 | 14.55 | 61.53 | 3.60 | 14.55 |
| kvarn quantizer | 0.19 | 2.94 | 12.14 | 52.08 | 2.94 | 12.14 |
| + static channel scale (first 256 cells) | 0.43 | 5.60 | 22.76 | 93.38 | 5.60 | 22.76 |
| + static channel scale^0.5 | 0.27 | 3.25 | 13.22 | 55.60 | 3.25 | 13.22 |
| + static channel mean (first 256 cells) | 0.23 | 3.03 | 12.26 | 50.80 | 3.03 | 12.26 |
| + static channel mean (first 128 cells) | 0.24 | 3.19 | 13.05 | 53.13 | 3.19 | 13.05 |
| + static channel mean (first 512 cells) | 0.23 | 3.00 | 12.16 | 50.54 | 3.00 | 12.16 |
| + static channel mean (whole sequence; not causal) | 0.18 | 2.38 | 9.61 | 39.87 | 2.38 | 9.61 |
| + static mean and scale^0.5 | 0.21 | 2.74 | 11.03 | 46.01 | 2.74 | 11.03 |
| + two-axis tile normalisation, original domain | 0.30 | 3.76 | 15.27 | 64.20 | 3.76 | 15.27 |
| + two-axis tile normalisation, after group rotation | 0.30 | 3.73 | 15.25 | 63.29 | 3.73 | 15.25 |
| + group midrange offset | 0.22 | 3.19 | 13.52 | 60.99 | 3.19 | 13.52 |
| + mean of the previous 128 cells | - | 2.47 | 9.95 | 40.99 | - | 9.95 |
| + mean of the tile, tiles of 512 | - | 2.09 | 8.44 | 34.79 | - | 8.44 |
| + mean of the tile, tiles of 256 | - | 1.97 | 8.00 | 32.90 | - | 8.00 |
| + mean of the tile, tiles of 128 | - | 1.87 | 7.54 | 31.21 | - | 7.54 |
| + mean of the tile, tiles of 64 | - | 1.75 | 7.03 | 28.80 | - | 7.03 |
| + mean of the tile, tiles of 32 | - | 1.57 | 6.33 | 26.10 | - | 6.33 |

SNR of the stored rows, dB (mean over the layers), K:

| codec | 8 bits | 6 bits | 5 bits | 4 bits |
|---|---|---|---|---|
| absmax, no rotation | 43.8 | 31.6 | 25.3 | 18.7 |
| absmax | 45.6 | 33.4 | 27.1 | 20.4 |
| engine | 45.6 | 34.2 | 28.1 | 22.0 |
| kvarn quantizer | 47.6 | 35.5 | 29.4 | 23.1 |
| + static scale | 44.3 | 32.8 | 26.8 | 20.7 |
| + static scale^0.5 | 46.0 | 34.6 | 28.5 | 22.4 |
| + static mean (256) | 46.3 | 34.8 | 28.7 | 22.7 |
| + static mean and scale^0.5 | 46.5 | 35.1 | 29.0 | 22.9 |
| + tile normalisation (orig.) | 45.6 | 34.1 | 28.0 | 22.0 |
| + tile normalisation (rot.) | 45.5 | 34.1 | 28.0 | 22.0 |
| + group offset | 46.8 | 34.8 | 28.6 | 22.1 |
| + tile mean 512 | 47.8 | 36.3 | 30.2 | 24.2 |
| + tile mean 128 | 48.3 | 36.8 | 30.8 | 24.7 |
| + tile mean 32 | 49.1 | 37.6 | 31.6 | 25.5 |
| + previous-tile mean 128 | 47.2 | 35.7 | 29.7 | 23.6 |

V:

| codec | 8 bits | 6 bits | 5 bits | 4 bits |
|---|---|---|---|---|
| absmax, no rotation | 43.3 | 31.1 | 24.8 | 18.2 |
| absmax | 45.7 | 33.5 | 27.2 | 20.6 |
| engine | 45.7 | 34.3 | 28.2 | 22.2 |
| kvarn quantizer | 44.6 | 32.5 | 26.3 | 20.0 |
| + static scale | 45.1 | 33.6 | 27.6 | 21.5 |
| + static scale^0.5 | 46.5 | 35.0 | 29.0 | 22.9 |
| + static mean (256) | 47.9 | 36.4 | 30.4 | 24.3 |
| + static mean and scale^0.5 | 47.9 | 36.4 | 30.4 | 24.3 |
| + tile normalisation (orig.) | 45.7 | 34.3 | 28.2 | 22.1 |
| + tile normalisation (rot.) | 45.7 | 34.3 | 28.2 | 22.1 |
| + group offset | 46.9 | 34.9 | 28.7 | 22.3 |
| + tile mean 512 | 49.1 | 37.6 | 31.6 | 25.5 |
| + tile mean 128 | 49.4 | 37.9 | 31.9 | 25.8 |
| + tile mean 32 | 49.8 | 38.3 | 32.3 | 26.2 |
| + previous-tile mean 128 | 48.5 | 37.1 | 31.0 | 24.9 |

- **Two-axis normalisation (kvarn's core) adds nothing to the engine's format.**
  Tiles of 128 x 128 normalised in both directions (16 Sinkhorn iterations, f16
  scales, +0.25 bit) change the SNR by 0.0 to -0.1 dB, the output error by +2%
  (normalised before the rotation of 32) or 0% (kvarn's order, after it) and the
  KL by +4% and +3% at q4. The engine's group scale already follows the size of
  each group of each cell, so the scale per cell is redundant (K rows have a
  constant norm anyway). A scale per channel does not help either: the full
  static scale is worse than none (output error +12%, KL +52% at q4); its square
  root helps a little (-5% output error, -10% KL; SNR +0.4 dB on K, +0.7 on V). A
  scale that flattens the channel energies moves error to the loud channels,
  which carry the scores.
- **A mean per channel is what carries kvarn's gain on K, and more.** The static
  mean from the first 256 cells lowers the KL by 17% (50.8 against 61.5 at q4;
  kvarn's quantizer 52.1) and the output error by 16% (19.4 against 23.1), at no
  storage cost. It is a constant for the head: on K it cancels in the softmax,
  on V it is added once to the finished output (the weights sum to 1). It goes
  stale when the sequence changes: the mean of the whole sequence (which needs
  the future) gives KL 39.9 and output error 17.7, while means from the first
  128, 256 or 512 cells are within 5% of each other.
- **A mean per tile is better still.** An f16 vector per tile of 128 cells
  (0.125 bit per value at head size 256) takes the KL to 31.2 (-49%) and the
  output error to 15.95 (-31%) at q4/q4, to 7.54 (-48%) and 7.72 (-26%) at q5/q5,
  to 1.87 (-48%) and 3.85 (-25%) at q6/q6. SNR at 4 bits, layer means: K +2.7 dB, V +3.6 dB (+1.3 to +2.8 dB in
  layers 3 to 19, +12.0 dB in layer 23, whose V is mostly its mean); the same
  at every width. Smaller tiles are better (KL 26.1, 28.8, 31.2, 32.9, 34.8 at q4
  for tiles of 32, 64, 128, 256, 512 cells) and the gain falls slowly: tiles
  of 512 (0.03 bit) keep 88% of the KL gain and 91% of the output-error gain of
  tiles of 128. The mean of the previous 128 cells, which does not make a cell
  wait for its tile to fill, keeps two thirds of the KL gain (41.0) and three
  quarters of the error gain (17.75).
- **A midrange offset per rotated group** (+0.5 bit) is worth -1% output error
  and -1% KL at q4, -11% output error at 8 bits; the SNR gain at 4 bits is 0.1
  dB. It does not pay for its half bit (about 3 dB).

SNR of V per layer at 4 bits (dB); K does not change from layer to layer
(engine 22.0 to 22.1 in all six, kvarn 22.4 to 23.7, tile mean 24.0 to 25.2):

| codec | L03 | L07 | L11 | L15 | L19 | L23 |
|---|---|---|---|---|---|---|
| absmax | 20.3 | 20.6 | 20.5 | 20.7 | 21.1 | 20.3 |
| engine | 21.9 | 22.1 | 22.1 | 22.2 | 22.7 | 21.9 |
| kvarn quantizer | 20.0 | 20.1 | 20.1 | 20.1 | 20.2 | 19.7 |
| + static mean (256) | 23.7 | 22.6 | 22.4 | 22.8 | 24.4 | 29.8 |
| + tile mean 128 | 24.7 | 24.1 | 23.6 | 23.5 | 24.8 | 33.9 |

## 7. K against V

One side only (kvarn's tail; only the side named uses the idea, the other side
is the engine):

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| engine | 1.47 | 5.16 | 10.48 | 23.09 | 7.78 | 16.30 |
| kvarn on K only | 1.30 | 4.90 | 9.95 | 20.76 | 7.59 | 15.88 |
| kvarn on V only | 1.55 | 5.65 | 11.43 | 24.53 | 9.00 | 18.49 |
| kvarn on both | 1.39 | 5.40 | 10.99 | 22.81 | 8.85 | 18.27 |
| static mean on K only | 1.39 | 4.94 | 10.01 | 21.21 | 7.63 | 15.95 |
| static mean on V only | 1.38 | 4.84 | 9.79 | 21.37 | 6.98 | 14.27 |
| static mean on both | 1.31 | 4.60 | 9.29 | 19.39 | 6.81 | 13.89 |
| static scale^0.5 on K only | 1.45 | 5.01 | 10.27 | 22.65 | 7.68 | 16.15 |
| static scale^0.5 on V only | 1.43 | 5.01 | 10.19 | 22.45 | 7.44 | 15.52 |
| static scale^0.5 on both | 1.41 | 4.86 | 9.97 | 22.00 | 7.33 | 15.37 |
| tile normalisation (orig.) on K only | 1.50 | 5.30 | 10.85 | 23.55 | 7.89 | 16.56 |
| tile normalisation (orig.) on V only | 1.46 | 5.16 | 10.49 | 23.13 | 7.80 | 16.31 |
| tile normalisation (orig.) on both | 1.50 | 5.30 | 10.86 | 23.57 | 7.90 | 16.58 |
| group offset on K only | 1.38 | 4.99 | 10.38 | 23.01 | 7.66 | 16.23 |
| group offset on V only | 1.40 | 5.01 | 10.25 | 22.90 | 7.50 | 16.06 |
| group offset on both | 1.31 | 4.84 | 10.15 | 22.83 | 7.37 | 15.98 |
| tile mean 128 on K only | - | 4.40 | 8.85 | 18.66 | - | 15.15 |
| tile mean 128 on V only | - | 4.67 | 9.46 | 20.64 | - | 13.31 |
| tile mean 128 on both | - | 3.85 | 7.72 | 15.95 | - | 12.03 |
| previous-tile mean 128 on K only | - | 4.69 | 9.48 | 19.91 | - | 15.58 |
| previous-tile mean 128 on V only | - | 4.77 | 9.66 | 21.04 | - | 13.86 |
| previous-tile mean 128 on both | - | 4.28 | 8.60 | 17.75 | - | 13.08 |

The KL depends on K only: it is the engine's whenever the idea is on V only,
and equals the "both" entry whenever it is on K.

- kvarn's quantizer is a K quantizer. On K it lowers the KL at every width (15%
  to 34%) and the output error by 10% at q4. On V it raises the output error
  (+5% at 8 bits, +9% at 6 and 5, +6% at 4) and its SNR is 2.2 dB below the
  engine's. In kvarn a V row is a cell and the grid is the minimum and maximum
  of 128 rotated values; the engine's grid is searched on 32 values.
- A mean per channel helps both: on K through the scores (KL -17% static, -49%
  per tile), on V through the output (output error -7% static, -11% per tile).
  The V gain in SNR is larger (the mean is a large part of V's energy in the
  last layer).
- The other ideas matter for one side only or not at all: the square root of the
  scale helps K and V a little and about equally; the tile normalisation does
  nothing on either.

## 8. Qwen3.8-27B (5 of 16 attention layers)

The same text and candidates, a smaller set, queries every fourth position from
512. Mean rel L2 (x 1e-3) and KL (x 1e-4):

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| absmax; nothing exact | 3.47 | 14.18 | 29.16 | 63.09 | - | 47.43 |
| engine; nothing exact | 3.47 | 12.36 | 24.84 | 50.81 | - | 40.37 |
| kvarn quantizer; nothing exact (*) | 3.44 | 13.82 | 27.98 | 58.38 | - | 47.78 |
| absmax; first 128 + open group | 1.60 | 6.33 | 13.09 | 28.46 | - | 20.24 |
| engine; first 128 + open group | 1.60 | 5.54 | 11.29 | 24.29 | - | 17.55 |
| kvarn quantizer; first 128 + open group (kvarn as built) | 1.50 | 5.87 | 11.92 | 24.75 | - | 19.69 |
| absmax; KvPolicy defaults | 3.47 | 6.06 | 10.35 | 15.82 | - | 11.01 |
| engine; KvPolicy defaults | 3.47 | 5.32 | 8.96 | 13.70 | - | 9.64 |
| kvarn quantizer; KvPolicy defaults | 3.44 | 5.62 | 9.44 | 13.67 | - | 10.69 |

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| absmax; nothing exact | 0.42 | 7.08 | 30.07 | 138.13 | - | 30.07 |
| engine; nothing exact | 0.42 | 5.11 | 20.23 | 83.54 | - | 20.23 |
| kvarn quantizer; nothing exact (*) | 0.28 | 4.45 | 18.28 | 78.22 | - | 18.28 |
| absmax; first 128 + open group | 0.24 | 4.01 | 17.14 | 79.01 | - | 17.14 |
| engine; first 128 + open group | 0.24 | 2.98 | 11.98 | 50.63 | - | 11.98 |
| kvarn quantizer; first 128 + open group (kvarn as built) | 0.16 | 2.54 | 10.43 | 44.81 | - | 10.43 |
| absmax; KvPolicy defaults | 0.42 | 4.00 | 14.05 | 44.77 | - | 9.65 |
| engine; KvPolicy defaults | 0.42 | 2.98 | 9.86 | 29.34 | - | 6.87 |
| kvarn quantizer; KvPolicy defaults | 0.28 | 2.54 | 8.62 | 25.59 | - | 5.95 |

What-ifs (kvarn's tail), output error and KL:

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| engine | 1.60 | 5.54 | 11.29 | 24.29 | - | 17.55 |
| kvarn quantizer | 1.50 | 5.87 | 11.92 | 24.75 | - | 19.69 |
| + static channel scale (first 256 cells) | 1.84 | 6.52 | 13.19 | 27.54 | - | 19.27 |
| + static channel scale^0.5 | 1.58 | 5.43 | 11.13 | 23.81 | - | 17.11 |
| + static channel mean (first 256 cells) | 1.50 | 5.27 | 10.63 | 22.14 | - | 16.09 |
| + static channel mean (whole sequence; not causal) | - | 4.85 | 9.75 | 20.29 | - | 15.08 |
| + two-axis tile normalisation, original domain | 1.73 | 6.09 | 12.30 | 26.22 | - | 18.85 |
| + two-axis tile normalisation, after group rotation | 1.61 | 5.56 | 11.31 | 24.13 | - | 17.48 |
| + group midrange offset | 1.40 | 5.19 | 10.72 | 24.03 | - | 17.08 |
| + mean of the previous 128 cells | - | 4.97 | 10.01 | 20.55 | - | 15.34 |
| + mean of the tile, tiles of 512 | - | 4.67 | 9.38 | 19.25 | - | 14.61 |
| + mean of the tile, tiles of 128 | - | 4.53 | 9.04 | 18.56 | - | 14.17 |
| + mean of the tile, tiles of 32 | - | 4.26 | 8.54 | 17.39 | - | 13.44 |

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 |
|---|---|---|---|---|---|---|
| engine | 0.24 | 2.98 | 11.98 | 50.63 | - | 11.98 |
| kvarn quantizer | 0.16 | 2.54 | 10.43 | 44.81 | - | 10.43 |
| + static channel scale (first 256 cells) | 0.37 | 4.79 | 19.26 | 78.71 | - | 19.26 |
| + static channel scale^0.5 | 0.24 | 2.89 | 11.77 | 49.05 | - | 11.77 |
| + static channel mean (first 256 cells) | 0.21 | 2.68 | 10.81 | 45.04 | - | 10.81 |
| + static channel mean (whole sequence; not causal) | - | 2.14 | 8.64 | 35.71 | - | 8.64 |
| + two-axis tile normalisation, original domain | 0.28 | 3.54 | 14.19 | 59.51 | - | 14.19 |
| + two-axis tile normalisation, after group rotation | 0.25 | 3.05 | 12.31 | 51.80 | - | 12.31 |
| + group midrange offset | 0.18 | 2.62 | 10.93 | 49.79 | - | 10.93 |
| + mean of the previous 128 cells | - | 2.22 | 9.00 | 36.99 | - | 9.00 |
| + mean of the tile, tiles of 512 | - | 1.88 | 7.59 | 31.41 | - | 7.59 |
| + mean of the tile, tiles of 128 | - | 1.73 | 6.88 | 28.16 | - | 6.88 |
| + mean of the tile, tiles of 32 | - | 1.46 | 5.96 | 24.21 | - | 5.96 |

Exact cells, q4/q4, first by newest cells, output error:

| first cells exact \ newest cells exact | 0 | 32 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| 0 | 50.81 | 33.34 | 27.08 | 23.05 | 19.04 | 13.60 |
| 4 | 47.24 | 27.24 | 19.79 | 14.88 | 9.88 | 5.33 |
| 16 | 47.15 | 27.14 | 19.68 | 14.76 | 9.74 | 5.21 |
| 64 | 47.03 | 26.97 | 19.49 | 14.56 | 9.52 | 5.02 |
| 128 | 46.50 | 26.26 | 18.70 | 13.70 | 8.57 | 4.28 |
| 256 | 45.98 | 25.55 | 17.86 | 12.70 | 7.73 | 3.86 |

The 27B agrees with the 0.8B on every comparison above, with these numbers:

- The search against the absolute maximum, nothing exact: output error 13%,
  15%, 20% lower at 6, 5, 4 bits, KL 28%, 33%, 40% lower; SNR +0.7, +1.1, +1.6 dB.
- kvarn against the engine: with nothing exact the output error is higher by
  12% (6 bits), 13% (5), 15% (4) and the KL lower by 13% (6), 10% (5), 6% (4);
  with kvarn's tail the output error is 6% higher at 6 and 5 bits, 2% higher at
  4 bits and 12% higher at K5/V4, the KL 15%, 13%, 12% lower (6, 5, 4 bits); SNR
  +0.9 dB on K and -2.0 dB on V at 4 bits.
- The exact cells give 83% to 86% of the log-gain in output error at 4 to 6 bits
  and 56% to 65% of the KL (engine with kvarn's tail).
- The first 4 cells give 87% to 89% of the first-128 gain in output error and
  51% of it in KL; a window of 1 024 newest cells takes the output error at q4/q4
  from 50.8 to 4.3.
- Static mean (256 cells): KL -11%, output error -9% at q4/q4. Mean per tile of 128:
  KL -44%, output error -24%; tiles of 512: -38% and -21%; previous 128 cells:
  -27% and -15%. SNR of the tile mean at 4 bits: K +2.5 dB, V +2.4 dB (V +8.4
  dB in the last layer).
- Tile normalisation: worse before the rotation (output error +8%, KL +18%),
  neutral after it (-1%, +2%). Group offset: -1% and -2%. Scale^0.5: -2% and -3%.

## 9. Read-side cost of each idea, in the engine's terms

The design rule: reads stay cheap, at most one extra term per group. Numbers on
the right are q4/q4 with kvarn's tail on the 0.8B (output error x 1e-3 / KL x
1e-4).

| idea | reads | writes | stored | result |
| --- | --- | --- | --- | --- |
| engine as shipped | none | scale search, 17 candidates at q4 | B + 0.5 bits | 23.1 / 61.5 |
| absolute-maximum scale | none | cheapest | B + 0.5 | 26.4 / 93.9 (worse) |
| kvarn's quantizer | per value: unpack and `(code * scale + zero) * column` for V (K folds the column scale into the query); per tile: K's channel scale into the query, row scale per score; V's row scale and zero point into the weights and the channel scale per output channel | Sinkhorn per tile; waits for 128 cells | B + 0.375 | 22.8 / 52.1 |
| first / newest cells exact | none per value; a pool of f16 pages beside the quantized ones | convert a page when it leaves the window | f16 for those cells | section 4.3 |
| static channel scale | none per cell: the query (K) or the finished output (V) is multiplied once per head and token | divide by a stored vector | none | 25.9 / 93.4 (full scale, worse); sqrt 22.0 / 55.6 |
| static channel mean | none per cell: cancels in the softmax (K); one add of a stored vector per head and token (V) | subtract a stored vector | none (a vector per head) | 19.4 / 50.8 |
| mean per tile | one term per tile: `q.mean` for K (256 multiply-adds per 128 cells), a vector add per tile weighted by the tile's total softmax weight for V | needs the tile, or the previous tile's mean | `16 / cells` bit: 0.125 (128), 0.03 (512) | 15.95 / 31.2 (previous tile 17.75 / 41.0; 512 cells 16.6 / 34.8) |
| two-axis tile normalisation | per tile: query scaling; per cell and slice a multiply on the score (K) and on the weight (V); in the order before the rotation V needs an un-rotation per tile | Sinkhorn per tile; waits for the tile | +0.25 bit | 23.6 / 64.2 (no gain) |
| group midrange offset | one term per group and cell (the sum of the rotated query over the group times the offset) | min and max per group | +0.5 bit | 22.8 / 61.0 |

## 10. Recommendation

Sections 1 to 9 and this recommendation were written before anything was
built; they are readings of the data. Item 3's mean per tile was then built into
`ares-quant` and is measured as built in section 13; items 1 and 2 speak about
the formats and the `KvPolicy` defaults as they were before that change (the
new defaults are described in section 13).

1. **Keep the formats as they are.** The ladder is as good on real K and V as on
   the synthetic rows (+1.6 dB at q4 over the old scale; 12% to 18% lower output
   error and 26% to 37% lower KL with nothing exact), no layer loses, and the
   stress case that worried the module documentation does not occur.
2. **Change the shape of the exact cells in `KvPolicy`.** The newest window is
   where the memory pays: the defaults of 256 cells at q4 are far from the knee
   (1 024 cells take the q4/q4 output error from 12.9 to 3.3 on this text), and
   1 024 cells cost 46 MiB per sequence on the 27B. The first cells matter less:
   4 give 87% to 94% of the gain of 128 in output error, but the KL keeps improving
   slowly with more, and 128 first cells cost 6 MiB per sequence on the 27B, so
   the first-cell default can stay (or fall to 16 to 64) while the newest window
   grows. The right size is a question of the VRAM budget and of a longer
   context than this data has.
3. **If one idea is added to the formats, it should be a mean per channel, not
   kvarn's normalisation.** In order of cost:
   - a static mean per head and side, from the first cells of a sequence or from
     a calibration: no storage, no work per cell; KL -17% and output error -16%
     on the 0.8B at q4/q4, -11% and -9% on the 27B, that is a quarter to a third
     of the KL gain and 40% to 50% of the output-error gain of the mean per
     tile;
   - a mean per tile (an f16 vector per 128 to 512 cells, subtracted before the
     group rotation): KL -43% to -49% and output error -28% to -31% on the 0.8B
     at q4/q4 (-38% to -44% and -21% to -24% on the 27B) for 0.03 to 0.125 bit per
     value and a per-tile read term. Next to the value of a bit of width (q4 to
     q5 lowers the output error by 55% for one bit) this is a good price. It
     needs the tile to be full, or the mean of the previous tile (about two
     thirds of the KL gain and three quarters of the output gain).
   - the square root of a static channel scale is free too (-5% output error,
     -10% KL at q4 on the 0.8B; -2% and -3% on the 27B) but small next to the
     mean.
4. **Do not add** kvarn's two-axis normalisation, a per-channel scale, a midrange
   offset per group, or kvarn's quantizer. Its K advantage is the channel
   offset, which the mean gives without per-value work, and it is worse on V.

## 11. Limits and what was not validated

- One text of 3 686 tokens and one model family. The 0.8B and the 27B share the
  architecture; other attention layouts (all layers attending, sliding windows,
  MLA) were not looked at. The context stops at 3 686 tokens: with tails the
  share of exact cells shrinks with depth, so the numbers with tails do not
  predict 64k, and large windows (1 024) cover much of this context.
- The measure is the error of one attention head's output, weights and scores.
  It was not carried through the output projection, the following layers or the
  logits, so KLD of logits, perplexity and task scores were not measured. How
  an error of this size in one layer adds up in the network is not known.
- The 27B: 5 of its 16 attention layers; the 0.8B: all 6. Positions below 512 were
  not queried.
- The engine was measured through its CPU reference codec, not its kernels.
  kvarn was measured through the re-implementation that was validated against
  beellama's CPU reference; the CUDA kernels were read, not run. The exact
  suffix of kvarn depends on batch sizes and stage groups in the runtime; the
  model here is 1 or 2 groups, and kvarn as it runs may be in between. The
  author's own benchmark (KLD against bf16 KV at 64k, a different model) was not
  reproduced.
- The static and tile statistics come from the sequence they are applied to
  (the first 256 cells, or the tile); a calibration from other text was not
  tried. The mean over the whole sequence was (it is not causal and is an upper
  bound for a static mean).
- The tile mean of sections 6 to 9 is evaluated with tiles aligned to 128 cells
  and kvarn's exact cells; section 13 measures the built codec with the
  engine's tile-granular policy. What no section studied: how tiles map onto the
  engine's pages and onto prefix sharing (a tile that straddles two sequences),
  the conversion of a tile when it leaves the newest window, the attention
  kernels' per-tile terms, and contexts much longer than 3.7k.
- Nothing about speed was measured; the costs in section 9 come from the
  definitions of the formats.

## 12. How it was done

Everything ran on the CPU, six threads at low priority (a full comparison
takes about 14 minutes; the 27B capture 6 minutes), no GPU, no server. The
sources of the tool are in `crates/quant/tools/kv_real/`: `capture/capture.cpp`
(against `libllama` of llama.cpp-native), `kvarn_ref/` (the harness that links
`llama-kvarn.o` compiled unchanged from a beellama checkout), `corpus/assemble.py`
(the text), and a Rust program that uses `ares-quant` as a library (`src/`,
commands `run`, `stats`, `validate`; `tables.py` and `decomp.py` print tables
from the result files). It is not a workspace member. Captured data (1.5 GB) and
the clone of beellama were deleted after the first round (section 13 captured
the data again with the same commands). To repeat: build `capture.cpp` and
run it with `KVCAP_OUT` set on the text; build `kvarn_ref` from a beellama
checkout; `cargo build --release` in the tool directory; `kvreal validate --data
DIR --kvarn-ref PATH --work DIR`; `kvreal stats --data DIR`; `kvreal run --data
DIR --stride 2 --out FILE` (`--set extra|sweep|tilemean` for the further
candidate sets).

## 13. The mean per tile as built

Added 2026-11-10. After sections 1 to 12 the mean per tile was built into
`ares-quant` for `q7` to `q4`, and this section measures the built codec on
the same data. The data were captured a second time with the same commands
(`crates/quant/tools/kv_real/README.md`); the three candidates that were run
again to check this (`engine|K4V4|kvarn`, `engine|K4V4|policy`,
`engine+tile-mean(128)|K4V4|kvarn`, 130 result lines and 448 SNR lines on the
0.8B) came out identical to the numbers of sections 4 to 9 as printed.

**What is built** (`crates/quant/src/kv.rs`): `encode_tile` and `decode_tile`
store the rows of one head (K or V) for consecutive cells a tile at a time:
the mean of the tile's rotated rows as f16 (`16 / cells` bits per value, 0.125
for the 128 cells of `KV_TILE_CELLS`) and every row relative to the stored
mean. `q8` and `f16` have no mean. Tiles are aligned to the start of the
sequence. `KvPolicy` is tile-granular as soon as K or V is `q7` to `q4`: the
cells stored in the formats (`quantized_cells(n)`) are the whole tiles between
its two windows, a tile being converted when it is full and none of its cells
is in a window; every other cell is `f16`. The defaults are 128 first cells
for `q7` to `q4` and newest windows of at least 32, 64, 128, 256 cells
(`q7` to `q4`), which with the tiles are 256 to 383 cells wide at `q4`. These
differ from the sliding policy of sections 4 to 9 (first 128, 64, 32 cells at
`q4`, `q5`, `q6`, and exactly 256, 128, 64 newest cells; the tool keeps that
policy as the tail `policy` so that those numbers stay reproducible).

**Candidates run** (pairs K8/V8, K6/V6, K5/V5, K4/V4, K6/V5, K5/V4, and K8/V4
and K4/V8 for the last two):

- `built|KbVb|kvarn`: the built codec on tiles of 128 with the tail of the
  what-ifs (first 128 cells and the open group, as `engine+tile-mean(128)`).
- `built|KbVb|policy`: the engine as it is now. For a query that attends over
  `n = t + 1` cells, the cells in `KvPolicy::new(k, v).quantized_cells(n)` are
  read from tiles encoded once with `encode_tile`, all others as `f16`.
- `rows|KbVb|tile-policy`: the row codec (`encode_row`/`decode_row`, no mean)
  with exactly the same exact cells as the previous one. The difference
  between the two is what the mean adds under the engine's real policy; the
  difference between this and the old sliding policy is what the changed
  windows add.

References are the stored numbers of the earlier runs, not recomputed: the old
sliding policy with the old defaults (`engine|..|policy`), the kvarn quantizer
as built by kvarn (`kvarn|..|kvarn`), the engine's rows with the same tail
(`engine|..|kvarn`) and the absolute maximum without exact cells. A dash is a
number that was not run (the earlier runs had no K8/V4 or K4/V8 and, on the
27B, no K6/V5). Queries as before: 0.8B every second position from 512 (76 176 samples), 27B every
fourth (95 280).

### 13.1 The built codec against the what-if

`kvreal validate`, layer 3 of the 0.8B: at 8 bits the built codec is the row
codec bit for bit (943 616 of 943 616 values, K and V). At 6, 5, 4 bits its SNR
is that of the what-if of section 6 to 0.001 dB (K 37.135, 31.102, 25.018 dB; V
34.920, 28.881, 22.798; the rows 34.131, 28.086, 21.995 for K) while the two
reconstructions are not the same numbers: their rms difference is 27%, 18%,
12% of the rms quantization error for K at 6, 5, 4 bits (19%, 13%, 8% for V),
and 40% to 55% of the values differ by more than 1e-4 of the rms.

The same comparison on attention, candidate 1 against the what-if with the
same tail (mean x 1e-3 for the output error and the scores, x 1e-4 for the KL;
K8/V8 has no mean, so its two columns are both the row codec):

0.8B:

| pair | measure | what-if (mean in the original domain) | built (mean in the rotated domain) | difference |
|---|---|---|---|---|
| K8/V8 | rel | 1.4652 | 1.4652 | +0.000% |
| K8/V8 | kl | 0.2862 | 0.2862 | +0.000% |
| K8/V8 | rel99 | 5.9446 | 5.9446 | +0.000% |
| K8/V8 | sc | 10.2361 | 10.2361 | +0.000% |
| K6/V6 | rel | 3.8523 | 3.8534 | +0.028% |
| K6/V6 | kl | 1.8735 | 1.8747 | +0.064% |
| K6/V6 | rel99 | 16.8745 | 16.8190 | -0.329% |
| K6/V6 | sc | 26.9374 | 26.9407 | +0.013% |
| K5/V5 | rel | 7.7224 | 7.7234 | +0.012% |
| K5/V5 | kl | 7.5359 | 7.5371 | +0.016% |
| K5/V5 | rel99 | 33.5687 | 33.6372 | +0.204% |
| K5/V5 | sc | 53.9634 | 53.9394 | -0.045% |
| K4/V4 | rel | 15.9545 | 15.9586 | +0.026% |
| K4/V4 | kl | 31.2131 | 31.2304 | +0.055% |
| K4/V4 | rel99 | 69.9404 | 69.7564 | -0.263% |
| K4/V4 | sc | 108.8235 | 108.8144 | -0.008% |
| K5/V4 | rel | 12.0309 | 12.0303 | -0.005% |
| K5/V4 | kl | 7.5359 | 7.5371 | +0.016% |
| K5/V4 | rel99 | 55.5505 | 55.4675 | -0.149% |
| K5/V4 | sc | 53.9634 | 53.9394 | -0.045% |

27B:

| pair | measure | what-if (mean in the original domain) | built (mean in the rotated domain) | difference |
|---|---|---|---|---|
| K8/V8 | rel | 1.5991 | 1.5991 | +0.000% |
| K8/V8 | kl | 0.2399 | 0.2399 | +0.000% |
| K8/V8 | rel99 | 6.1203 | 6.1203 | +0.000% |
| K8/V8 | sc | 9.3990 | 9.3990 | +0.000% |
| K6/V6 | rel | 4.5253 | 4.5205 | -0.107% |
| K6/V6 | kl | 1.7269 | 1.7182 | -0.501% |
| K6/V6 | rel99 | 18.2755 | 18.2123 | -0.345% |
| K6/V6 | sc | 25.5413 | 25.5378 | -0.014% |
| K5/V5 | rel | 9.0421 | 9.0428 | +0.008% |
| K5/V5 | kl | 6.8786 | 6.8786 | +0.001% |
| K5/V5 | rel99 | 36.5435 | 36.4278 | -0.317% |
| K5/V5 | sc | 51.2558 | 51.2655 | +0.019% |
| K4/V4 | rel | 18.5557 | 18.5535 | -0.012% |
| K4/V4 | kl | 28.1572 | 28.1681 | +0.039% |
| K4/V4 | rel99 | 73.6272 | 73.6869 | +0.081% |
| K4/V4 | sc | 103.3384 | 103.3376 | -0.001% |
| K5/V4 | rel | 14.1720 | 14.1743 | +0.016% |
| K5/V4 | kl | 6.8786 | 6.8786 | +0.001% |
| K5/V4 | rel99 | 60.2321 | 60.2272 | -0.008% |
| K5/V4 | sc | 51.2558 | 51.2655 | +0.019% |

They agree to 0.03% in output error and 0.06% in KL on the 0.8B, and to 0.1%
and 0.5% on the 27B (K6/V6, the worst), with the 99th percentile of the output
error within 0.35%. The differences have both signs and the SNR is the same,
so they are noise, not a loss. The cause is inferred, not measured: the what-if
rounds the mean to f16 in the original domain and the built codec in the
rotated domain, so the rows relative to the stored mean differ by a rounding of
the mean (up to 2^-11 of its size), which moves every value a little and some
codes by a step.

### 13.2 Output error, KL, p99 and score error

0.8B:

output error, mean (x 1e-3)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 3.62 | 14.99 | 31.10 | 65.18 | 22.95 | 48.96 | - | - |
| engine, old sliding policy (128/256 at q4) | 3.62 | 4.94 | 8.34 | 12.85 | 6.14 | 8.86 | - | - |
| kvarn quantizer, kvarn's tail | 1.39 | 5.40 | 10.99 | 22.81 | 8.85 | 18.27 | - | - |
| engine rows, kvarn's tail | 1.47 | 5.16 | 10.48 | 23.09 | 7.78 | 16.30 | - | - |
| built tiles, kvarn's tail | 1.47 | 3.85 | 7.72 | 15.96 | 5.95 | 12.03 | - | - |
| rows, tile policy | 3.62 | 4.11 | 6.95 | 10.93 | 5.08 | 7.46 | 6.22 | 8.39 |
| built tiles, tile policy | 3.62 | 3.07 | 5.06 | 7.42 | 3.87 | 5.42 | 4.68 | 5.54 |

KL, mean (x 1e-4)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 0.51 | 8.57 | 36.75 | 166.15 | 8.57 | 36.75 | - | - |
| engine, old sliding policy (128/256 at q4) | 0.51 | 3.57 | 11.97 | 35.59 | 2.95 | 8.40 | - | - |
| kvarn quantizer, kvarn's tail | 0.19 | 2.94 | 12.14 | 52.08 | 2.94 | 12.14 | - | - |
| engine rows, kvarn's tail | 0.29 | 3.60 | 14.55 | 61.53 | 3.60 | 14.55 | - | - |
| built tiles, kvarn's tail | 0.29 | 1.87 | 7.54 | 31.23 | 1.87 | 7.54 | - | - |
| rows, tile policy | 0.51 | 2.91 | 9.91 | 30.74 | 2.44 | 7.25 | 0.14 | 30.74 |
| built tiles, tile policy | 0.51 | 1.51 | 5.10 | 15.57 | 1.27 | 3.74 | 0.14 | 15.57 |

output error, 99th percentile (x 1e-3)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 10.33 | 43.68 | 92.88 | 184.77 | 50.44 | 108.77 | - | - |
| engine, old sliding policy (128/256 at q4) | 10.33 | 20.54 | 38.67 | 69.88 | 30.83 | 52.98 | - | - |
| kvarn quantizer, kvarn's tail | 5.65 | 22.60 | 45.75 | 92.86 | 38.78 | 79.99 | - | - |
| engine rows, kvarn's tail | 5.94 | 21.10 | 41.73 | 87.74 | 32.65 | 65.26 | - | - |
| built tiles, kvarn's tail | 5.94 | 16.82 | 33.64 | 69.76 | 27.50 | 55.47 | - | - |
| rows, tile policy | 10.33 | 19.62 | 36.53 | 65.20 | 29.01 | 48.61 | 42.46 | 52.71 |
| built tiles, tile policy | 10.33 | 15.80 | 29.36 | 52.27 | 23.87 | 40.60 | 36.80 | 38.51 |

score error, mean (x 1e-3 nats)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 10.88 | 44.56 | 91.96 | 197.35 | 44.56 | 91.96 | - | - |
| engine, old sliding policy (128/256 at q4) | 10.88 | 39.00 | 75.82 | 142.26 | 37.74 | 70.38 | - | - |
| kvarn quantizer, kvarn's tail | 8.18 | 32.63 | 66.15 | 137.25 | 32.63 | 66.15 | - | - |
| engine rows, kvarn's tail | 10.24 | 37.80 | 75.98 | 153.42 | 37.80 | 75.98 | - | - |
| built tiles, kvarn's tail | 10.24 | 26.94 | 53.94 | 108.81 | 26.94 | 53.94 | - | - |
| rows, tile policy | 10.88 | 36.88 | 72.37 | 138.23 | 35.99 | 68.36 | 9.19 | 138.23 |
| built tiles, tile policy | 10.88 | 26.27 | 51.29 | 97.59 | 25.62 | 48.35 | 9.19 | 97.59 |

27B:

output error, mean (x 1e-3)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 3.47 | 14.18 | 29.16 | 63.09 | - | 47.43 | - | - |
| engine, old sliding policy (128/256 at q4) | 3.47 | 5.32 | 8.96 | 13.70 | - | 9.64 | - | - |
| kvarn quantizer, kvarn's tail | 1.50 | 5.87 | 11.92 | 24.75 | - | 19.69 | - | - |
| engine rows, kvarn's tail | 1.60 | 5.54 | 11.29 | 24.29 | - | 17.55 | - | - |
| built tiles, kvarn's tail | 1.60 | 4.52 | 9.04 | 18.55 | 7.02 | 14.17 | - | - |
| rows, tile policy | 3.47 | 4.37 | 7.40 | 11.80 | 5.47 | 8.26 | 6.95 | 8.96 |
| built tiles, tile policy | 3.47 | 3.56 | 5.88 | 8.84 | 4.52 | 6.52 | 5.62 | 6.54 |

KL, mean (x 1e-4)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 0.42 | 7.08 | 30.07 | 138.13 | - | 30.07 | - | - |
| engine, old sliding policy (128/256 at q4) | 0.42 | 2.98 | 9.86 | 29.34 | - | 6.87 | - | - |
| kvarn quantizer, kvarn's tail | 0.16 | 2.54 | 10.43 | 44.81 | - | 10.43 | - | - |
| engine rows, kvarn's tail | 0.24 | 2.98 | 11.98 | 50.63 | - | 11.98 | - | - |
| built tiles, kvarn's tail | 0.24 | 1.72 | 6.88 | 28.17 | 1.72 | 6.88 | - | - |
| rows, tile policy | 0.42 | 2.38 | 8.06 | 25.48 | 2.00 | 5.98 | 0.12 | 25.48 |
| built tiles, tile policy | 0.42 | 1.38 | 4.66 | 14.26 | 1.16 | 3.48 | 0.12 | 14.26 |

output error, 99th percentile (x 1e-3)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 8.73 | 35.66 | 73.04 | 161.73 | - | 99.03 | - | - |
| engine, old sliding policy (128/256 at q4) | 8.73 | 20.82 | 39.55 | 73.06 | - | 56.09 | - | - |
| kvarn quantizer, kvarn's tail | 5.83 | 23.16 | 47.11 | 97.39 | - | 80.89 | - | - |
| engine rows, kvarn's tail | 6.12 | 21.17 | 42.45 | 87.64 | - | 67.69 | - | - |
| built tiles, kvarn's tail | 6.12 | 18.21 | 36.43 | 73.69 | 29.79 | 60.23 | - | - |
| rows, tile policy | 8.73 | 19.89 | 37.44 | 67.66 | 29.93 | 50.91 | 44.72 | 53.44 |
| built tiles, tile policy | 8.73 | 17.10 | 31.95 | 56.82 | 26.31 | 44.85 | 40.38 | 41.56 |

score error, mean (x 1e-3 nats)

| candidate | K8/V8 | K6/V6 | K5/V5 | K4/V4 | K6/V5 | K5/V4 | K8/V4 | K4/V8 |
|---|---|---|---|---|---|---|---|---|
| absmax, nothing exact | 9.99 | 40.91 | 84.57 | 180.94 | - | 84.57 | - | - |
| engine, old sliding policy (128/256 at q4) | 9.99 | 35.79 | 69.48 | 130.41 | - | 64.47 | - | - |
| kvarn quantizer, kvarn's tail | 7.60 | 30.37 | 61.60 | 127.37 | - | 61.60 | - | - |
| engine rows, kvarn's tail | 9.40 | 34.72 | 69.65 | 140.82 | - | 69.65 | - | - |
| built tiles, kvarn's tail | 9.40 | 25.54 | 51.27 | 103.34 | 25.54 | 51.27 | - | - |
| rows, tile policy | 9.99 | 33.88 | 66.33 | 126.72 | 33.06 | 62.65 | 8.44 | 126.72 |
| built tiles, tile policy | 9.99 | 24.90 | 48.79 | 92.77 | 24.29 | 46.05 | 8.44 | 92.77 |

### 13.3 What the mean adds under the engine's policy, and what the windows add

From the old sliding policy with rows, to the tile policy with rows (only the
windows change), to the built tiles (the mean). First the 0.8B, then the 27B.

output error (x 1e-3)

| pair | old sliding policy, rows | tile policy, rows | change (wider windows) | tile policy, built tiles | change (mean) | total |
|---|---|---|---|---|---|---|
| K8/V8 | 3.62 | 3.62 | +0% | 3.62 | +0% | +0% |
| K6/V6 | 4.94 | 4.11 | -17% | 3.07 | -25% | -38% |
| K5/V5 | 8.34 | 6.95 | -17% | 5.06 | -27% | -39% |
| K4/V4 | 12.85 | 10.93 | -15% | 7.42 | -32% | -42% |
| K6/V5 | 6.14 | 5.08 | -17% | 3.87 | -24% | -37% |
| K5/V4 | 8.86 | 7.46 | -16% | 5.42 | -27% | -39% |
| K8/V4 | - | 6.22 | - | 4.68 | -25% | - |
| K4/V8 | - | 8.39 | - | 5.54 | -34% | - |

KL (x 1e-4)

| pair | old sliding policy, rows | tile policy, rows | change (wider windows) | tile policy, built tiles | change (mean) | total |
|---|---|---|---|---|---|---|
| K8/V8 | 0.51 | 0.51 | +0% | 0.51 | +0% | +0% |
| K6/V6 | 3.57 | 2.91 | -19% | 1.51 | -48% | -58% |
| K5/V5 | 11.97 | 9.91 | -17% | 5.10 | -49% | -57% |
| K4/V4 | 35.59 | 30.74 | -14% | 15.57 | -49% | -56% |
| K6/V5 | 2.95 | 2.44 | -17% | 1.27 | -48% | -57% |
| K5/V4 | 8.40 | 7.25 | -14% | 3.74 | -48% | -56% |
| K8/V4 | - | 0.14 | - | 0.14 | +0% | - |
| K4/V8 | - | 30.74 | - | 15.57 | -49% | - |

output error (x 1e-3)

| pair | old sliding policy, rows | tile policy, rows | change (wider windows) | tile policy, built tiles | change (mean) | total |
|---|---|---|---|---|---|---|
| K8/V8 | 3.47 | 3.47 | +0% | 3.47 | +0% | +0% |
| K6/V6 | 5.32 | 4.37 | -18% | 3.56 | -19% | -33% |
| K5/V5 | 8.96 | 7.40 | -17% | 5.88 | -21% | -34% |
| K4/V4 | 13.70 | 11.80 | -14% | 8.84 | -25% | -35% |
| K6/V5 | - | 5.47 | - | 4.52 | -17% | - |
| K5/V4 | 9.64 | 8.26 | -14% | 6.52 | -21% | -32% |
| K8/V4 | - | 6.95 | - | 5.62 | -19% | - |
| K4/V8 | - | 8.96 | - | 6.54 | -27% | - |

KL (x 1e-4)

| pair | old sliding policy, rows | tile policy, rows | change (wider windows) | tile policy, built tiles | change (mean) | total |
|---|---|---|---|---|---|---|
| K8/V8 | 0.42 | 0.42 | +0% | 0.42 | +0% | +0% |
| K6/V6 | 2.98 | 2.38 | -20% | 1.38 | -42% | -54% |
| K5/V5 | 9.86 | 8.06 | -18% | 4.66 | -42% | -53% |
| K4/V4 | 29.34 | 25.48 | -13% | 14.26 | -44% | -51% |
| K6/V5 | - | 2.00 | - | 1.16 | -42% | - |
| K5/V4 | 6.87 | 5.98 | -13% | 3.48 | -42% | -49% |
| K8/V4 | - | 0.12 | - | 0.12 | +0% | - |
| K4/V8 | - | 25.48 | - | 14.26 | -44% | - |

Average exact cells per query (the 0.8B; the sampled queries attend over 2 099
cells on average; the 27B is the same within one cell):

| pair | old sliding policy | tile policy | kvarn's tail | of the cells attended (mean) |
|---|---|---|---|---|
| K8/V8 | 0 | 0 | 192 | 2099 |
| K6/V6 | 96 | 256 | 192 | 2099 |
| K5/V5 | 192 | 320 | 192 | 2099 |
| K4/V4 | 384 | 448 | 192 | 2099 |
| K6/V5 | 192 | 320 | 192 | 2099 |
| K5/V4 | 384 | 448 | 192 | 2099 |
| K8/V4 | 384 | 448 | 192 | 2099 |
| K4/V8 | 384 | 448 | 192 | 2099 |

### 13.4 Per layer, depth and text at K4/V4

Under the tile policy, rows against built tiles. 0.8B:

| candidate | L03 | L07 | L11 | L15 | L19 | L23 | all |
|---|---|---|---|---|---|---|---|
| output error: rows, tile policy | 9.36 | 9.18 | 13.19 | 14.15 | 14.57 | 5.12 | 10.93 |
| output error: built tiles, tile policy | 6.55 | 6.82 | 9.23 | 9.32 | 11.10 | 1.52 | 7.42 |
| output error: change | -30% | -26% | -30% | -34% | -24% | -70% | -32% |
| KL: rows, tile policy | 33.23 | 24.52 | 32.75 | 36.12 | 33.65 | 24.16 | 30.74 |
| KL: built tiles, tile policy | 16.26 | 14.66 | 17.43 | 16.26 | 15.60 | 13.19 | 15.57 |
| KL: change | -51% | -40% | -47% | -55% | -54% | -45% | -49% |

| candidate | t512-1023 | t1024-2047 | t2048+ | english | german | code |
|---|---|---|---|---|---|---|
| output error: rows, tile policy | 9.85 | 9.39 | 12.22 | 11.06 | 11.50 | 10.11 |
| output error: built tiles, tile policy | 6.64 | 6.26 | 8.40 | 7.36 | 7.76 | 7.12 |
| output error: change | -33% | -33% | -31% | -33% | -32% | -30% |
| KL: rows, tile policy | 21.40 | 26.97 | 36.01 | 31.96 | 33.35 | 26.12 |
| KL: built tiles, tile policy | 10.08 | 13.47 | 18.59 | 15.74 | 17.13 | 13.59 |
| KL: change | -53% | -50% | -48% | -51% | -49% | -48% |

27B:

| candidate | L03 | L15 | L31 | L47 | L63 | all |
|---|---|---|---|---|---|---|
| output error: rows, tile policy | 10.74 | 16.73 | 11.48 | 14.23 | 5.83 | 11.80 |
| output error: built tiles, tile policy | 7.77 | 13.97 | 8.66 | 11.11 | 2.68 | 8.84 |
| output error: change | -28% | -16% | -25% | -22% | -54% | -25% |
| KL: rows, tile policy | 25.88 | 25.07 | 22.05 | 26.44 | 27.97 | 25.48 |
| KL: built tiles, tile policy | 13.12 | 17.30 | 12.46 | 13.85 | 14.55 | 14.26 |
| KL: change | -49% | -31% | -44% | -48% | -48% | -44% |

| candidate | t512-1023 | t1024-2047 | t2048+ | english | german | code |
|---|---|---|---|---|---|---|
| output error: rows, tile policy | 10.00 | 10.62 | 13.10 | 11.53 | 12.54 | 11.35 |
| output error: built tiles, tile policy | 7.49 | 7.94 | 9.82 | 8.56 | 9.42 | 8.58 |
| output error: change | -25% | -25% | -25% | -26% | -25% | -24% |
| KL: rows, tile policy | 16.33 | 22.46 | 30.23 | 26.29 | 26.72 | 22.97 |
| KL: built tiles, tile policy | 8.84 | 12.63 | 16.96 | 14.55 | 15.22 | 12.76 |
| KL: change | -46% | -44% | -44% | -45% | -43% | -44% |

### 13.5 SNR of the stored rows

Mean over the layers, dB, whole tiles of 128 rows. The what-if is the codec of
section 6. 0.8B:

| codec | side | 8 bits | 6 bits | 5 bits | 4 bits |
|---|---|---|---|---|---|
| rows | K | 45.6 | 34.2 | 28.1 | 22.0 |
| rows | V | 45.7 | 34.3 | 28.2 | 22.2 |
| built tiles | K | 45.6 | 36.8 | 30.8 | 24.7 |
| built tiles | V | 45.7 | 37.9 | 31.9 | 25.8 |
| what-if tile mean 128 | K | 48.3 | 36.8 | 30.8 | 24.7 |
| what-if tile mean 128 | V | 49.4 | 37.9 | 31.9 | 25.8 |

27B:

| codec | side | 8 bits | 6 bits | 5 bits | 4 bits |
|---|---|---|---|---|---|
| rows | K | 45.6 | 34.1 | 28.1 | 22.0 |
| rows | V | 45.5 | 34.1 | 28.0 | 22.0 |
| built tiles | K | 45.6 | 36.6 | 30.6 | 24.5 |
| built tiles | V | 45.5 | 36.6 | 30.5 | 24.4 |
| what-if tile mean 128 | K | 48.1 | 36.6 | 30.6 | 24.5 |
| what-if tile mean 128 | V | 48.0 | 36.6 | 30.5 | 24.4 |

### 13.6 Reading

Measured:

- **The built mean does what the what-if said.** With kvarn's tail the built
  codec has 15.96 / 31.23 (output error x 1e-3 / KL x 1e-4) at K4/V4 on the
  0.8B against 15.95 / 31.21 for the what-if and 23.09 / 61.53 for the rows;
  on the 27B 18.55 / 28.17 against 18.56 / 28.16 and 24.29 / 50.63.
- **Under the engine's real policy the mean lowers the output error by 24% to
  34% on the 0.8B (17% to 27% on the 27B) and, where K has a mean, the KL by
  48% to 49% (42% to 44%)**, in every pair that has a mean on at least one
  side, against the same exact cells with rows. At K4/V4: output error 10.93 to
  7.42 and KL 30.74 to 15.57 on the 0.8B, 11.80 to 8.84 and 25.48 to 14.26 on
  the 27B. A mean on one side alone works too: K8/V4 (V has the mean) -25% /
  -19% output error, K4/V8 (K has the mean) -34% / -27% output error and -49%
  / -44% KL (0.8B / 27B); the KL of K8/V4 is unchanged, as it only sees K.
- **No layer, depth bin or text kind is worse with the mean, and neither is
  any of the 416 (0.8B) and 384 (27B) combinations of pair, scope and measure
  (output error, its 99th percentile, KL, score error).** At K4/V4 the gain
  in output error is smallest in layer 19 of the 0.8B (-24%) and layer 15 of
  the 27B (-16%, KL -31%), largest in the last attention layer of each model
  (-70% and -54%), where V is mostly its mean. Over depth and text the gain is
  flat: -30% to -33% and -48% to -53% (0.8B), -24% to -26% and -43% to -46% (27B)
  in output error and KL.
- **The windows account for a part of the change from the old policy.** The
  tile policy keeps 256, 320, 448 cells exact on average at q6, q5, q4 (0.8B)
  against 96, 192, 384 for the old sliding policy. With rows, that is 14% to
  18% less output error and 13% to 20% less KL. The mean takes a further 25%
  to 32% (0.8B) and 19% to 25% (27B) of the output error at K6/V6 to K4/V4, and
  42% to 49% of the KL. Together the engine as it is now has 37% to 42% less
  output error and 56% to 58% less KL than the sliding policy with rows on the
  0.8B, 32% to 35% and 49% to 54% on the 27B.
- **SNR**: built tiles gain 2.6 to 2.7 dB (K) and 3.6 to 3.7 dB (V) over rows on
  the 0.8B at 6 to 4 bits, 2.5 dB on both sides on the 27B. At 8 bits nothing
  changes.

Inferred, not measured here:

- The mean costs 0.125 bit per value on the tile cells (4.625 bits at `q4`),
  and the tile policy keeps up to 64 more cells exact on average at `q4` than
  the old one (more at `q6`, `q5`). This is a cost in memory that the
  comparison does not charge. A gain of a third of the output error and half
  of the KL for 2.8% more bytes in the tiles is a good price (one more bit of
  width, 22% more bytes, lowered the output error by 55% at `q4` in
  section 4), but the exact-cell memory was not compared at equal bytes.
- These are 3.7k-token contexts. With a longer context the exact cells are a
  smaller share of the cells attended and the mean applies to all the others,
  so the gain from the mean should hold and the contribution of the windows
  should shrink; neither was measured.
- The tool measures the codec on tiles aligned to the start of a sequence and
  the policy as `KvPolicy` states it. How tiles map onto pages, onto prefix
  sharing, and what attention kernels do with the per-tile terms was not
  looked at.

## 14. The reference attention over the stored sequence

Added 2026-11-10. The reference that the attention kernels will be held to is
`ares_quant::kvhead` (`crates/quant/src/kvhead/`): `KvHead` stores the cells of
one KV head under a `KvPolicy` (exact cells as `f16`, tiles converted as the
policy says, from the stored `f16` values, since the originals are gone), and
`KvHead::attend` computes the attention over a list of ranges of cells the way a
kernel does, as the merge of the states (log-sum-exp and normalised V sum) of the
ranges. This section runs the captured data through it and compares it with
`built|KbVb|policy` of section 13. The two differ in exactly two ways: the
tiles of the stored sequence are encoded from the `f16` values of the cells and
those of the arrays from the `f32` originals, and the attention is the `f64`
reference over states against `f32` arithmetic over arrays. The policy, the
exact cells and the codec are the same. The tool candidate is `head|KbVb|policy`
(`--set head`, README section 7): the head is fed cell by cell, and at each
sampled position `t` the query attends over cells `0..=t`.

Mean values over all samples (output error x 1e-3, KL x 1e-4, p99 x 1e-3, score
error x 1e-3 nats). The last column is the relative L2 distance between the output
of the stored sequence and that of the arrays for the same query, over all queries.
0.8B (stride 2):

| pair | output error: arrays / head / change | KL: arrays / head / change | p99 output error: arrays / head | score error: arrays / head | distance of the outputs, query by query: mean / p99 / max |
|---|---|---|---|---|---|
| K8/V8 | 3.618 / 3.618 / +0.00% | 0.506 / 0.506 / +0.00% | 10.333 / 10.333 | 10.877 / 10.877 | 5.0e-07 / 2.9e-06 / 1.3e-05 |
| K6/V6 | 3.072 / 3.075 / +0.10% | 1.510 / 1.509 / -0.04% | 15.796 / 15.795 | 26.272 / 26.266 | 7.8e-04 / 4.9e-03 / 2.0e-02 |
| K5/V5 | 5.061 / 5.060 / -0.01% | 5.097 / 5.100 / +0.06% | 29.360 / 29.316 | 51.286 / 51.305 | 8.3e-04 / 6.1e-03 / 1.9e-02 |
| K4/V4 | 7.422 / 7.425 / +0.04% | 15.567 / 15.582 / +0.10% | 52.269 / 52.320 | 97.595 / 97.605 | 6.5e-04 / 5.5e-03 / 3.7e-02 |
| K6/V5 | 3.872 / 3.877 / +0.13% | 1.268 / 1.268 / +0.07% | 23.865 / 23.995 | 25.625 / 25.619 | 7.2e-04 / 5.2e-03 / 2.1e-02 |
| K5/V4 | 5.425 / 5.426 / +0.02% | 3.738 / 3.742 / +0.10% | 40.600 / 40.470 | 48.353 / 48.374 | 6.2e-04 / 5.0e-03 / 3.6e-02 |
| K8/V4 | 4.683 / 4.683 / +0.00% | 0.141 / 0.141 / +0.02% | 36.804 / 36.783 | 9.195 / 9.203 | 4.3e-04 / 3.6e-03 / 3.7e-02 |
| K4/V8 | 5.540 / 5.544 / +0.07% | 15.567 / 15.582 / +0.10% | 38.509 / 38.624 | 97.595 / 97.605 | 5.0e-04 / 4.2e-03 / 3.0e-02 |

27B (stride 4):

| pair | output error: arrays / head / change | KL: arrays / head / change | p99 output error: arrays / head | score error: arrays / head | distance of the outputs, query by query: mean / p99 / max |
|---|---|---|---|---|---|
| K8/V8 | 3.474 / 3.474 / -0.00% | 0.420 / 0.420 / -0.00% | 8.728 / 8.728 | 9.987 / 9.987 | 5.5e-07 / 3.4e-06 / 1.1e-05 |
| K6/V6 | 3.557 / 3.562 / +0.15% | 1.379 / 1.382 / +0.28% | 17.103 / 17.141 | 24.905 / 24.916 | 8.8e-04 / 5.2e-03 / 1.7e-02 |
| K5/V5 | 5.877 / 5.876 / -0.01% | 4.657 / 4.654 / -0.07% | 31.954 / 31.863 | 48.787 / 48.765 | 8.8e-04 / 6.3e-03 / 2.7e-02 |
| K4/V4 | 8.838 / 8.839 / +0.02% | 14.256 / 14.260 / +0.03% | 56.818 / 56.695 | 92.766 / 92.802 | 7.7e-04 / 6.5e-03 / 4.9e-02 |
| K6/V5 | 4.516 / 4.519 / +0.07% | 1.163 / 1.167 / +0.32% | 26.306 / 26.385 | 24.293 / 24.304 | 8.0e-04 / 5.5e-03 / 2.1e-02 |
| K5/V4 | 6.523 / 6.523 / +0.00% | 3.477 / 3.474 / -0.09% | 44.849 / 44.698 | 46.050 / 46.030 | 6.9e-04 / 5.6e-03 / 3.3e-02 |
| K8/V4 | 5.618 / 5.621 / +0.05% | 0.117 / 0.118 / +0.63% | 40.381 / 40.396 | 8.437 / 8.446 | 5.3e-04 / 4.3e-03 / 3.3e-02 |
| K4/V8 | 6.535 / 6.538 / +0.04% | 14.256 / 14.260 / +0.03% | 41.561 / 41.424 | 92.766 / 92.802 | 5.7e-04 / 4.9e-03 / 4.9e-02 |

Measured:

- **Converting from `f16` costs nothing measurable.** The mean output error of the
  stored sequence is between -0.01% and +0.13% (0.8B) and -0.01% and +0.15% (27B)
  of the arrays' error, the KL between -0.09% and +0.32% (and +0.63% for K8/V4 on
  the 27B, where the KL itself is 0.1e-4), the 99th percentile of the output
  error between -0.34% and +0.54%, the score error within 0.1%. The signs are
  mixed. At K4/V4: 7.425 against 7.422 on the 0.8B, 8.839 against 8.838 on the
  27B. Of the 416 (0.8B) and 384 (27B) combinations of pair, scope and measure
  nine and five differ by more than 1%; all are a 99th percentile in a single
  layer or part of the text (a tail statistic of 2 000 to 19 000 samples, signs
  mixed, up to 3%), or the tiny KL of K8/V4.
- **Query by query the outputs are not the same numbers.** The relative L2
  distance is 4e-4 to 9e-4 on average (99th percentile 4e-3 to 7e-3, maximum
  up to 5e-2) at 6 to 4 bits, mixed pairs included: converting from the rounded values moves some codes by a
  step. That is about a tenth of the output error against the exact attention
  (7e-3 at K4/V4), of random sign, and it does not change the average error.
- **At K8/V8 the distance is 5e-7.** There, no cell is converted from `f16` (a
  `q8` cell without exact windows is encoded straight from the `f32` row, as the
  codec's tests and `KvHead` say), so what is left is the `f32` arrays against the
  `f64` reference and the merge: they compute the same attention.

Inferred, not measured separately: the `f16` rounding (2^-11 of a value) is small
next to the step of `q6` to `q4`, and shows only where the format is fine, in the
KL of a `q8` K tile (K8/V4, +0.6% on the 27B). The conclusion for the engine:
encoding a tile from its stored `f16` rows loses nothing that the policy's
measurements in sections 4 to 13 did not already contain.

What was tested of the reference itself is in `crates/quant/tests/kvhead.rs`
(33 tests): the stored bytes against a model written from the policy and the
codec for every pair of formats at the lengths where tiles are converted, the
conversion from `f16` and the one `q8` exception, truncation and the pending
protocol, the kernel way against the plain way for every pair across tile and
window boundaries and selections that cut tiles and exact runs, any split of the
visible cells into pieces (per tile, per page of 16 or 64, exact cells apart from
tiles, halves) merged in four orders, pieces far below the others and empty
pieces, and a naive `f32` kernel (whole and in pieces) within the documented
tolerances.
