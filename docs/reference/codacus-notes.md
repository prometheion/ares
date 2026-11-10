# Notes from reading TheCodacus' llama.cpp fork

https://github.com/thecodacus/llama.cpp, default branch `perf`, read on
2026-11-09. Open pull requests form a stack: #19 `fable5/prefill-8k-chunks`,
#20 `fable5/moe-dyn-cache`, #21 `fable5/deepseek41`, #22
`fable5/parallel-readahead`. Every number below is quoted from that
repository; none was reproduced here.

## Expert cache in VRAM

- **Unit.** Whole experts, `S` slots per layer, for the layers whose experts
  live in host memory. Per layer the three expert tensors get a "hot" copy with
  the same type and `S` experts, one contiguous slab per expert copied byte for
  byte, plus two maps of expert id to hot slot and to cold id (-1 on the other
  side).
- **Dual path.** The router's top-k is unchanged. The ids go through both maps;
  each side runs the full expert chain and an id of -1 yields a zero row, then
  the two results are added. The cold side runs on the CPU, overlapped with the
  GPU side by a worker thread.
- **Profile.** A trace of routed experts per position and layer; the cache takes
  the top `S` per layer by decode count.
- **Measured on an RTX 3060** (decode tokens/s): Qwen3.6-35B 42.3 -> 51.3, with
  MTP and overlap 74.2; Qwen3.8-Flash-Next with MTP 16.6 -> 24.4. Break-even is
  about 25 % routing coverage.
- **Borrowing.** Batches above a size free the cache, run every expert on the
  GPU, and upload the cache again (about 2 s). Two RTX 3060: prefill 61 ->
  437-597 tokens/s.

## Dynamic cache and masking (#20)

- `--moe-cache-dyn N`: demand is counted per layer and expert inside the graph.
  Every `N` decoded tokens `score = 0.9 * score + demand` and the top `S` are
  elected again. Experts that stay keep their slot; replaced slots leave the
  maps at once; a worker uploads the new ones and the maps are published after
  the decode in which the upload finished. Lossless, works without a profile.
  Locality: an expert used now was used in the last 64 tokens 89 % of the time.
- `--moe-cache-mask N`: the selection scores of experts that are not cached get
  a large negative bias, except the router's top `N` raw picks, which are
  protected; then the top-k is taken. Picks beyond `N` that are not cached are
  replaced by cached experts, and the count stays `k`. This is lossy: on
  Qwen3.6 protecting the top 2 kept GSM8K at 55/60 against 56/60 and HumanEval
  at 49/50 against 50/50, while a hard mask (no protection) fell to 25/60.
- Together on Qwen3.8-Flash-Next (one RTX 3060, MTP, 59 slots): static 27.9,
  dyn 32 gives 32.0, static with mask 2 gives 36.9, dyn 32 with mask 2 gives
  38.7 decode tokens/s.

This pair is the closest thing in the fork to "dynamic expert pruning"
(removing the experts a run does not need, re-decided every x tokens, with a
bounded quality loss). Nothing in the fork runs fewer than `k` experts or drops
experts by a weight threshold.

## Expert prediction (`fable5/expert-prediction`, not in the stack)

A layer's router is run on the layer input before the attention, the predicted
misses are copied into a round-robin ring of slots on a second stream through
a pinned staging buffer, and the maps are republished before the lookup. Single
token decode only; no throughput result is stated.

## Host memory and SSD

- Registering the mapped weights as page-locked read-only memory
  (`cudaHostRegister`) lifts host-to-device copies from 6-7 to about 20 GB/s.
  Where registration fails, the weights are copied into pinned buffers.
- Prefetching whole expert tensors for large batches on a second stream into
  event-ordered staging slots: prefill 1 383 -> 1 880 tokens/s with 3 slots.
- Read-ahead threads that touch one byte per page ahead of the compute: 16.1 ->
  38.2 tokens/s. Page faults read at about 1.7 GB/s, `pread` with 16 threads at
  5-6 GB/s.
- `--layer-major N`: `N` layers run over all micro-batches before the next `N`,
  so each layer group's weights are read from disk once per prompt instead of
  once per micro-batch (DeepSeek-V4.1, 16.7k prompt: 16 -> about 300
  tokens/s).
- No direct-I/O tier: cold experts come through the page cache. The hashed-row
  tables (111 GB) are read one row per token by up to 32 threads.

## "Jev mode"

Jev is a proprietary decision model of TypeSafe AI: it answers a request by
choosing among fixed options instead of writing tokens. "Jev mode" is the same
idea applied to any model: score each option of a schema from one cached
prefix, by reading the logits after the prompt as probabilities over the
allowed choices, and stop before autoregressive decoding. The fork implements
it as `/v1/decision` (`tools/parallel-decision/`, `--decision-seqs`), scoring
the branches of multi-token options in one pass; its external pull requests
carry the name (`jev-hybrid-decision`, `jev-mtp-decision-fix`). A third-party
measurement with Qwen3.6-27B on an RTX 3090 reports 1.3x at four tokens and
5.5x at fifty against generating the answer: the gain is the tokens that are
never written.
