# Notes on KVarN (beellama.cpp) and TurboQuant KV formats

Read on 2026-11-09: https://github.com/Anbeeld/beellama.cpp at `1cef3bf6d`
(2026-11-08, after v0.4.9): `src/llama-kvarn.cpp`, the graph and cache code,
the CUDA kernels `kvarn.cu` and `fattn-mma-kvarn-decode.cuh`, the Vulkan
shader `kvarn_flash_attn.comp`, README, docs, CHANGELOG, issues #109, #140,
#145, #163, and the author's two linked articles (through a summarising fetch,
so second-hand). Also
`ggml-turbo-quant.c` and the block definitions of thecodacus/llama.cpp
`fable5/turboquant` at `12a02027d` (2026-07-18). Nothing was built or run
except a CPU simulation on synthetic data. Statements are facts from these
sources unless marked **Judgement** or *not verified*.

## Format

KVarN is Huawei's calibration-free quantizer (arXiv 2606.03458 according to the
article). Widths `kvarn2/3/4/5/6/8`, separate for K and V, plus one separate
pair for sliding-window layers. No per-layer, per-head or per-channel widths.

- **Rotation.** One normalised Walsh-Hadamard transform over the whole head
  (64 to 512 values), no sign flips. Query rotated once, output un-rotated once.
- **Tile.** The rotated head is cut into 128-wide slices. A record is one slice
  of one KV head over 128 consecutive tokens: 128 x 128 values.
- **Normalisation.** 16 Sinkhorn iterations divide rows and columns by their
  standard deviations; the iterate with the smallest max/min spread is kept.
- **Grid.** Each row of the normalised tile gets a uniform B-bit grid from its
  minimum to its maximum. K rows are channels, V rows are tokens.
- **Stored.** Codes packed at exactly B bits, row-major (K is channel-major),
  and three f16 vectors of 128: row scale, row zero point, column scale.
  `value = (code * scale[row] + zero[row]) * other[col]`; B + 0.375 bits per
  value. No codebook, no residual.
- **Exact rows.** The first 128 tokens and the newest 128 to 255 stay f16;
  `--kv-tail-tokens` enlarges the suffix. A group is quantized when full.

Other KV types: llama.cpp's `q8_0 q5_0 q5_1 q4_0 q4_1 iq4_nl`, the fork's
`q6_0 q6_1 q3_0 q3_1 q2_0 q2_1` (32 values, f16 scale; `_1` adds an f16
minimum), and an f16/bf16 "precision tail" for all; `turbo2/3/4` now redirect
to kvarn. Standard quantized types are Hadamard-rotated too (K over the head,
V in 64-point blocks, default on): kvarn-versus-qN compares rotation with
rotation.

## Read path and what makes it slower

CUDA decode, one thread block per 64 or 128 tokens:

1. Copy the tile's three vectors to shared memory. For K, multiply the query
   by the channel scale and sum query times zero point into one scalar.
2. Unpack K codes from global memory (word loads and shifts), tensor-core
   product, then `score = token_scale * (sum + scalar)`.
3. For V, unpack and evaluate the full formula per value (two multiplies, one
   add, three shared reads), then the tensor-core product.
4. Write partial sums; a second kernel combines them.

A llama.cpp type needs per value one code extraction and one multiply by a
scale that is constant for 32 contiguous values of a token row.

Causes named in the source comments (contributors' measurements on an RTX
3090): the kernel waits on global loads of the packed payload (6.1 of 15.2
cycles; one load touches four record rows 160 bytes apart); registers limit it
to 4 blocks of 8 warps per multiprocessor; each block has a fixed setup cost.
Folding V's scales as K's are folded gave nothing (39.79 against 39.75 tok/s):
the limit is the packed tile layout, not the arithmetic. Before a fix in
v0.4.5, host-side cache planning cost 7 to 11 ms per token (#140). A precision
tail is rotated again at every step. Vulkan applies the full formula per value
for K and V.

Write path: per token one transform into an f16 stage; per completed group one
Sinkhorn kernel per tile.

## Numbers in the repository

Median KLD against bf16 KV, Qwen 3.6 27B Q5_K_S, 64k context, Wikitext-2, RTX
3090; the author's. The first and last columns are in the README, the others
mostly only in the articles.

| Bits | qN_0 tail 0 | tail 128 | tail 1024 | kvarn tail 128 | tail 1024 |
|---|---|---|---|---|---|
| 8 | 0.000909 | 0.000901 | 0.000897 | 0.000871 | 0.000871 |
| 6 | 0.000960 | 0.000913 | 0.000908 | 0.000889 | 0.000879 |
| 5 | 0.001154 | 0.000970 | 0.000938 | 0.000927 | 0.000897 |
| 5 (`q5_1`) | 0.001087 | 0.000960 | 0.000926 | | |
| 4 | 0.001846 | 0.001206 | 0.001057 | 0.001111 | 0.000994 |

Mixed: `q5_0/q4_0` 0.001516 (tail 0); `kvarn5/kvarn4` 0.000936 (tail 1024).
Gemma 4 31B, 16k, tail 128: `q4_0` 0.0801, `kvarn4` 0.0582, `q5_0` 0.0461,
`kvarn5` 0.0413. No perplexity or task scores exist. Issue #109 (a user, the
author agrees): at 4 bits the 128 exact tokens give most of the gain (35 %),
the quantizer 8 %.

Speed. Prefill is 5 to 7 % below same-width standard types on Qwen, 7 to 14 %
on Gemma (author). Decode at depth 131k, a 35B MoE model on two RTX 3090
(author, #140): `kvarn5/kvarn4` 32.8 tok/s with the host cost, 77.5 without
(a contributor's build), `q6_0/q5_0` 49.8. A user later reports `kvarn3` 36.4
against `q3_0` 29.9 at 82k. The fork's standard types run on llama.cpp's
vector kernels, so this does not contradict native q5 kernels beating kvarn5.

## What is write-only

Free at read: rotation, Sinkhorn, the range choice, the exact rows. Per tile
or per token when folded: K's channel scale and zero point (query rescale, one
scalar), K's token scale (one multiply per score), V's token scale and zero
point (into the softmax weight, one scalar), V's channel scale (one multiply
per channel per tile). Per value: unpacking, and in the fork V's formula.

## TurboQuant (thecodacus)

`turbo2_0/3_0/4_0`: per 128 values a sign-flipped Hadamard transform, L2
normalisation, each value mapped to the nearest of 4, 8 or 16 fixed Lloyd-Max
centroids of a Gaussian, one f16 norm (B + 0.125 bits). Reading is a table
lookup times the norm. Nothing is shared across tokens, except an optional
calibrated K channel scale folded into the query (`TURBO_INNERQ`, header
comment only). Against KVarN: a fixed non-uniform grid instead of
data-dependent two-axis scales.

## Judgement

Simulation (mine; synthetic Gaussian values; signal-to-error in dB at 4/5/6
bits): the engine's grid of 32 values with an absolute-maximum scale
20.3/26.9/33.2; with a clip search 20.8/27.2/33.3; KVarN's row grid over 128
values 20.0/26.3/32.5; Lloyd-Max over 128, 20.4/26.1. On well-behaved data the
engine's grid equals both; KVarN gains by making real data well-behaved and by
the exact rows.

Formats are per page, so f16 pages, `qB` pages and a later variant can be mixed
in one sequence. KVarN's own records are not worth porting.

| Idea | Read cost | Verdict |
|---|---|---|
| f16 first and newest pages, encoded to `qB` when a page fills | none per value; a small f16 page pool | add; largest measured gain |
| Clip search for the scale | none | add; about 0.5 dB at `q4` |
| Rotation over 64 or the head, sign flips | none | measure; llama.cpp keeps V at 64 as "beneficial" (*not verified*) |
| Widths per layer | none | later; needs calibration |
| Fixed per-head channel scale and K mean, calibrated per model | none: folded into the query and the final un-rotation; a constant K offset cancels in the softmax | try before any new format |
| Per-page channel scale and zero point (KVarN's core) | per page: query rescale, two scalars; encoding waits for the full page | second variant `qBn`; about 0.4 to 0.75 bit by the fork's tables |
| Non-uniform grid | a lookup per value | no |

Defaults. **Fast:** K `q5`, V `q4`, today's format, first and newest page f16.
**Quality:** K `q6`, V `q5`, same format, about 1024 newest tokens f16. A
normalised `q4n` is for memory-bound contexts, not for speed. K above V follows
the fork's ladder; its tail-0 data gives V's fifth bit slightly more (24 %
against 18 %), so measure `q4`/`q5` too.

Not verified: the ideas' effect on the engine, the paper, the fork's numbers.
