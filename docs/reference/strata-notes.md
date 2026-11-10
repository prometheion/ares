# Notes from reading Strata

Strata (https://github.com/niko1221/strata, MIT) runs Qwen3.8-Flash-Next, the
GGUF architecture `qwen4exp`, which the model file itself calls "a preview of
the Qwen4 architecture". These notes record what was read from the Strata
source at commit `504b1433` (engine 0.1.59) on 2026-11-09. Nothing was built
or run, so they describe the source, not measured behaviour, and paths are
relative to the Strata repository. They are the reference for the parts of
this engine that must fit Qwen4 when it is released.

## 1. The `qwen4exp` model

### Hyperparameters (GGUF prefix `qwen4exp.`)

| Key | Value |
| --- | --- |
| `block_count` | 48 |
| `embedding_length` | 2560 |
| vocabulary (`tokenizer.ggml.tokens`) | 248320 |
| `context_length` | 262144 |
| `full_attention_interval` | 4 (layer `l` is attention when `l % 4 == 3`: 12 attention layers, 36 recurrent) |
| `attention.compress_ratios` | 48 entries, 4 on attention layers, 0 elsewhere |
| `attention.head_count` / `head_count_kv` | 24 / 2 |
| `attention.key_length` = `value_length` | 256 |
| `rope.dimension_count`, `freq_base`, `dimension_sections` | 64, 1e7, [11, 11, 10, 0] |
| `attention.indexer.head_count` / `key_length` / `top_k` | 4 / 128 / 2048 cells |
| `ssm.state_size`, `group_count`, `time_step_rank`, `conv_kernel`, `inner_size` | 128, 16, 48, 4, 6144 |
| `hyper_connection.count` / `low_rank` | 4 / 320 |
| `expert_count` / `expert_used_count` | 512 (Coder variant 256) / 10 |
| `expert_feed_forward_length` = shared | 640 |
| `ple.layers`, `ngram_size`, `heads_per_ngram`, `conv_kernel` | [1], 3, 8, 4 |
| `embedding_length_per_layer_input` | 160 |

Strata pins everything except expert count, top-k and vocabulary at compile
time (`include/strata/core/geometry.hpp`). (The later, closer reading for
[qwen4exp-architecture.md](qwen4exp-architecture.md) found that its loader
does pin `attention.indexer.top_k`; where the two documents differ, that one
is the more careful.) Its split engine also carries a
dense variant `FNX_27B` ("Qwen3.8-27B, dense": width 5120, 64 layers, 24 query
and 4 KV heads, dense FFN 17408, a draft layer in the file, no indexer). Both
shapes are candidates for Qwen4 27B and Qwen4 Flash.

### Layer structure

- The residual is four streams of width 2560 ("hyper connections"), started as
  four copies of the token embedding. There are no layer norms. Each layer is
  `read -> mixer -> write`, then `read -> MoE -> write`, where read mixes the
  streams through a low-rank gate (RMS norm per stream, rank 320) and write
  adds the block output back to each stream with a per-stream gain.
- Recurrent layer (Gated DeltaNet, `l % 4 != 3`): one projection to q, k, v,
  a depthwise causal convolution over the last 4 positions, a delta-rule
  state update per head, and an output projection.
- Attention layer (`l % 4 == 3`): 24 query heads over 2 KV heads of width 256,
  an output gate, and an indexer that selects which cells to attend (below).
- MoE on every layer: 512 experts of width 640, top 10, plus a shared expert
  that is always on. Router: softmax over all 512, top-k with ties to the
  lowest index, weights renormalised over the k.
- Layer 1 adds the PLE ("per-layer embedding"): the last 3 tokens are hashed
  to 16 rows of a table with 320 001 536 rows of 160 values (28.8 GB), which
  form one vector of width 2560 after a convolution over 9 history rows.
- Output: a final read mix, then `output.weight`. No output norm.
- MTP draft block (one layer, 31 `mtp.*` tensors, shipped apart from the
  published GGUFs): pairs the trunk's final residual at position `i` with
  token `i + 1` and predicts token `i + 2`; it has its own K/V.

In GGUF the expert weights are stacked 3-D tensors with the expert on the last
axis (`ffn_gate_exps`, `ffn_up_exps` [2560, 640, 512], `ffn_down_exps`
[640, 2560, 512]), so one expert is three contiguous byte ranges at
`offset + e * stride`.

### State a sequence carries

| State | Size | Addressable by position |
| --- | --- | --- |
| K/V of the 12 attention layers, 2 heads x 256 per cell | fp16: 24 576 B per token; int8 with an fp16 scale per 64 values: 12 672 B; rotated q4_0: 6 912 B | yes |
| Pooled indexer keys, f32 [128] per block of 4 cells and attention layer | 1 536 B per token | yes, by block |
| Indexer tail (raw keys of the open block) and the constant "dead" key | 3 x 128 + 128 f32 per attention layer | no |
| Recurrent state, per recurrent layer [128, 48, 128] f32 plus conv history [3, 10240] f32 | 3.27 MB per layer, 117.7 MB per sequence | no |
| PLE history | 2 token ids and 9 rows x 10240 f32 | no |
| Draft block K/V | one layer | yes |

K is stored after normalisation and RoPE. A K/V page in Strata is 4 cells,
which is exactly one indexer block.

### Sparse attention

Per attention layer and per query token:

1. Append the cell: K, V, and the raw indexer key (`indexer.k_proj`, 128).
2. When a block of 4 cells completes, its pooled key is
   `rope(rms_norm(mean of the 4 raw keys))` at the block's first position.
   Pooled keys are append-only; raw keys are not kept.
3. Score every completed block: the sum over 4 indexer heads of
   `relu(q_h . pooled[b])`. The open tail block always wins.
4. If the sequence has at most `top_k + 3` cells the selection is everything.
   Otherwise take the top blocks until `top_k` cells are selected (radix
   select, ties to the lowest index) and emit their cell ids ascending.
5. Attention reads only the selected cells, through the id list and the page
   table. One list serves all 24 heads. Nothing is shared between layers.

So decode reads about 2 051 cells per attention layer at any context depth,
and what the selection needs on the GPU at all times is the pooled keys
(1.5 KB per token), not the K/V. That is what makes host-resident K/V with a
VRAM page cache viable ("KV streaming" in Strata: the authoritative copy in
pinned host memory, VRAM slots, CLOCK eviction, a resolve kernel in front of
the attention).

Open point: at the cut-off Strata's two engines differ. The core kernels emit
exactly `top_k + 3` cells and cut the threshold block; the split engine keeps
512 whole blocks plus the tail. Which one matches the reference model must be
checked against the real Qwen4.

## 2. How Strata's engine runs it

- **Execution.** No graph IR. Kernel calls are captured into CUDA graphs: one
  per verify-window width, a commit graph, batch graphs per row layout. All
  values that change per token live in fixed-address pinned buffers and a
  device step buffer, so a captured graph is reused without changes.
- **Verify window and commit.** A window is the last token plus the drafts,
  run through all layers at once. It appends K/V and indexer keys but leaves
  the recurrent state untouched and stores each recurrent layer's inputs.
  After acceptance a commit pass replays only the accepted prefix into the
  recurrent state and restores the indexer tail and PLE history from a small
  snapshot. Rejected K/V cells are simply overwritten later. So there is no
  rollback of 118 MB of recurrent state.
- **Sampling on the GPU.** Greedy argmax inside the window; for sampled
  requests a kernel applies penalties, top-k up to 64, top-p, min-p and
  temperature. The random number for a position comes from a counter-based
  generator keyed by (seed, position), so a draft is accepted exactly when it
  equals what plain decoding would have sampled.
- **Drafting.** The MTP block (dense attention over its last 32 768 cells, a
  draft head over a vocabulary subset) and a suffix drafter (trigram index
  over prompt and output, longest match of at least 3). Draft length follows
  a policy that compares expected tokens per millisecond from the measured
  acceptance and step times.
- **Prefill.** Chunks of 2 048 to 8 192 tokens through all layers with
  tensor-core GEMMs and on-the-fly dequantisation, experts grouped by expert,
  non-resident experts streamed through a pinned ring. Generating slots run
  between chunks.
- **Batching.** Each sequence owns its state; a batch is one window with a
  row per conversation, at most 8 rows.
- **Conversation cache.** A checkpoint holds the non-addressable state (about
  118 MB of host RAM) while K/V stays in place. Checkpoints are made at turn
  boundaries, every 16 384 prompt tokens and at the root; the root is pinned,
  the rest LRU, optionally thinned geometrically toward the root. Parked
  conversations keep K/V pages and pooled keys in host RAM.
- **Multi-GPU.** The layer split hands the residual over once per window
  through pinned host memory. The all-cards split engine puts every layer on
  every card and splits recurrent heads, attention query heads, KV positions
  (dealt to cards by block), each expert's width and the vocabulary rows, with
  two sums across cards per layer. The NVIDIA build exchanges through pinned
  host memory, without peer access.

## 3. Expert cache, SSD streaming and the hashed-row tables

- **Unit.** A whole expert (gate, up, down) keyed by (layer, expert).
- **Tiers.** VRAM slots per layer (a slot table per layer). Host RAM: all
  experts pinned when they fit, else an arena of expert slots filled from
  disk, else the page cache. SSD: read from the GGUF in place at
  `tensor offset + e * stride`, one read per matrix, `O_DIRECT` with 4 KiB
  alignment from a thread pool with two queues (demand before guesses).
- **Heat.** A routing profile shipped with the model seeds per-layer counts;
  every routed pick adds 1; counts decay (x0.9 every 256 tokens).
- **Swaps.** Between windows and within a layer, the hottest non-resident
  expert replaces the coldest resident one when `heat_in > 1.25 * heat_out + 1`,
  at most 8 in flight. The Qwen engine does it every 4 windows with a gain
  threshold and a cap of 96 swaps.
- **Prediction.** Applying layer `L + 1`'s router to layer `L`'s MoE input
  predicts the next layer's experts (hit rate by rank 97 / 91 / 81 / 70 / 57 /
  41 %). Guesses start low-priority reads; a wrong guess is a wasted read.
  For prefill, experts that the profile says will be hit with probability
  at least 0.9 in the chunk are read ahead.
- **Misses.** In decode the GPU computes the shared and resident experts,
  publishes the routing, and the CPU computes the others and returns their
  weighted sum; some misses go to GPU staging slots over PCIe when a measured
  cost model says that is faster. In prefill every non-resident expert is
  streamed to the GPU. No token is replayed.
- **Pruning that changes answers** (opt-in): dropping low-weight experts that
  are not in RAM, or running the top 5 of 6.
- **Hashed-row tables** (DeepSeek "Engram", Qwen "PLE"): row ids are computed
  from the last tokens on the host, so the reads are issued when the token is
  known and collected just before the layer that needs them. `O_DIRECT` 4 KiB
  reads, pages deduplicated and sorted, a row cache (up to about 82 % of
  reads served), next chunk's rows read ahead during prefill.
- **Numbers Strata reports.** 663.6 MB of expert bytes read per token is
  16.2 ms of a 53 ms token. A static profile hits 64 % at 4 105 slots; with
  adaptive swaps 71 to 78 %.
