# Techniques of other engines

A survey made on 2026-11-09 of what is worth taking from engines other than
vLLM, llama.cpp, Strata and TheCodacus' fork (those have their own notes).
"Read" means read in the source or documentation of the repository at that
date; "claimed" means the project's own statement, not checked. Nothing was
built or run, and every number is the named project's own measurement.

## SGLang

- **Radix cache eviction** (read): only unlocked leaves are evictable, so a
  branch shrinks from its tip. The score is pluggable (LRU, LFU, segmented
  LRU, priority) and one policy keeps only as much of a conversation's tail as
  its next prompt needs for a time-to-first-token budget.
- **Cache-aware scheduling** (read): the waiting queue is ordered by matched
  prefix, or by waited tokens over uncached tokens, or shortest prompt first.
  A queued request that shares at least 32 tokens with another queued one is
  held back so the first fills the cache. Cheap, and useful for parallel
  agents that share a system prompt.
- **Overlap scheduler** (read): step N+1 is launched before step N's results
  are processed; the next input ids are gathered on the GPU. Sampling is
  deferred when a grammar mask needs the previous result.
- **Adaptive speculation** (read): a moving average of the accepted length
  picks among prepared draft lengths, with hysteresis.
- **Paged experts** (read): K of E experts in GPU slots, all in pinned RAM; a
  GPU kernel takes the eviction decision and a second kernel copies from
  pinned host memory through its device address, so paging needs no host
  round trip. Prefill splits experts into waves of at most K. On a 16 GB
  card, Qwen3-30B-A3B int4: 208 tokens/s at K = 63 of 128, 231 at K = 86.
- **HiSparse** (read): for sparse-attention models the full KV lives in pinned
  RAM and each request has a fixed GPU buffer; a kernel does hit detection,
  eviction and the copy of misses per step, and the miss plan of one layer is
  replayed for the following layers on a side stream.
- **Quantized KV** (read): plain 4-bit float KV costs accuracy on hard
  reasoning (aime25: Qwen3-235B 0.773 to 0.600, GPT-OSS-120B 0.753 to 0.353)
  while gsm8k is unchanged.
- **Hybrid state** (read): recurrent state is tracked every 256 tokens; cached
  checkpoints can be stored as int8 with a scale per head and channel.
- **KV on reserved virtual memory** (read): address space is reserved up front
  and physical pages are committed later, so addresses stay stable.

## TokenSpeed

github.com/lightseekorg/tokenspeed (LightSeek Foundation, MIT, announced May
2026): a datacenter engine for agentic workloads. From its design documents
(read):

- **State caches for recurrent layers**: two blocks per request (input
  checkpoint and output). State is published only at natural checkpoints
  (prompt end, finish, suspension) under the same chained hash as KV plus a
  hash of the tail, so a prefix hit can land at any token count. KV hits are
  bitwise; a recurrent resume differs slightly because kernels group tokens
  differently.
- **Suspension as a byte copy**: under memory pressure a running request is
  written to host memory as an image (all state plus sampler and drafter
  state) and restored bitwise. Forcing a suspension every N steps is the test
  oracle for state handling.
- **One decode path**: plain decode is a verify window of one row, and greedy
  is top-k of 1.
- **Draft trees**: up to 64 nodes per verify, with the accepted path compacted
  afterwards. Sampling noise is keyed by (seed, position).
- **Replay for recurrent layers**: verify never writes recurrent state; it
  stores a row per token and replays the accepted ones at commit.
- **Decode monokernel**: one persistent launch per layer with stages on fixed
  blocks. No numbers; high complexity.

## AirLLM

Layer-by-layer loading from disk with one prefetch thread and optional
compression to shrink reads. Every forward pass re-reads all touched weights,
so tokens per second are bounded by disk throughput over bytes per pass. Its
per-expert loading reads only routed experts. Nothing to take for decode
beyond what the expert tiers already do.

## ExLlamaV3

- **KV quantization** (read): each group of 32 values gets a Hadamard
  rotation, an f16 scale, a 2 to 8 bit grid and bit-plane packing; K and V
  have independent widths and the attention kernels decode in place.
- **Page table** (read): 256-token pages keyed by a chained hash, shared
  between batch members; eviction takes empty pages, then orphaned chains,
  then leaves of the least recently used trees.
- **RAM and disk tiers** (read): evicted pages go to pinned slots and are
  written through to self-describing files with a fingerprint of weights and
  configuration; a restore is capped at the last usable recurrent checkpoint.
- **Fused GPU sampler** (read): one pass with Gumbel noise from a counter-based
  generator; top-k and top-p by a two-level histogram select instead of a
  sort; the grammar bitmask is applied in the kernel.
- **Graph parameter patching** (read): recorded kernel parameters of a CUDA
  graph are patched only when they change, so pointers and lengths vary
  without new graphs.
- **Draft confidence calibration** (read): maps the drafter's confidence to
  observed acceptance and truncates the draft.

## Others

- **TensorRT-LLM**: KV retention by priority and time to live; pages are
  re-encoded to 4 bits only in the host and disk tiers; a GPU suffix automaton
  overrides the draft head on matches of at least 4 tokens, behind an
  acceptance gate; Gumbel sampling sharded by vocabulary per GPU, which fits a
  tensor split; 8-bit payloads for sums across GPUs; a sampled-threshold top-k
  for sparse attention (5.05x over radix select, claimed); a prefix cache for
  tokenization.
- **LMDeploy**: int8 KV with per-head ranges matches fp16; int4 loses 3.3
  points on gsm8k.
- **mistral.rs**: a transition log to roll back recurrent layers after a
  verify; per-GPU autotuning cached on disk; a detokenizer that emits the
  longest valid UTF-8 prefix.
- **MoE kernels in several engines**: (token, expert) pairs sorted by expert,
  then one grouped matrix product per expert.
- **Tokenizers**: the `bpe` crate is reported about 10x faster than the
  common implementation (claimed).

## Grammar engines

- **llguidance** (MIT, Rust): an Earley parser with a derivative-based lexer,
  no table per grammar, compile in about a millisecond, a mask in 50 to 140
  microseconds typically (its maintainers' benchmark). It can check a list of
  candidate tokens or a whole draft without advancing, and roll back.
- **xgrammar** (Apache-2.0, C++): lowest median when its cache hits, compile
  from 10 ms to 1 s, Rust only through community bindings.
- **outlines-core** (Apache-2.0, Rust): regular expressions to automata; no
  context-free grammars.

The fit for this engine is llguidance: validate the GPU's top candidates on
the CPU and upload a mask only when none of them is allowed.

## What this engine takes, in order of value

1. A checkpoint policy for recurrent state: checkpoints at natural points,
   named by the same chained hash as pages, with compressed storage. Without
   it prefix caching does not work for hybrid models.
2. Replay at commit for recurrent layers (already the plan; several engines
   confirm it).
3. Rotated low-bit KV with separate K and V widths, decoded in the attention
   kernel. The largest VRAM lever; un-rotated 4-bit float visibly breaks
   reasoning.
4. A fixed GPU working set for the KV of sparse-attention models, with the
   miss plan of one layer reused for the next.
5. Expert paging decided on the device.
6. Overlapped steps with the next token gathered on the GPU, and the verify
   window of one row as the only decode path.
7. Reserved virtual memory for state arenas.
8. Three-tier page cache with fingerprinted pages on disk and a 4-bit
   re-encoding for cold tiers; forced suspension as the test oracle.
9. Draft control: confidence calibration, a suffix automaton as a second
   drafter.
10. llguidance with candidate validation, and a sort-free sampler with the
    grammar mask applied in the kernel.

## References noted in OPEN.md (moved here on 2026-11-25)

Engines the owner or an agent pointed to after this survey, as they were noted in
[`OPEN.md`](../../OPEN.md), verbatim; each is to be read in its code before the piece it names is
built (the owner's rule of 2026-11-24: take code, not ideas).

### Strata-HotRod (2026-11-24)

Reference (2026-11-24, the owner's link): Strata-HotRod, https://github.com/sangharshadhyeta/Strata-HotRod,
MIT, a fork of Strata 0.1.45 for Flash-Next IQ3_S on an RTX 3060 12 GB with 30 GB RAM and NVMe on
PCIe 3.0; one request at a time; the author's own measurements, many single runs. Claims 17 ->
50-70 tokens/s decode on short answers (38-54 after a 45k prompt), UD-Q4_K_XL 15-25. Where it
comes from: (1) stand-in experts, the big lever: a missing expert is replaced by the best expert
on the card when router weight x logit gap is within a self-tuned budget (maths 47/50 against
48/50, code 30/30 both; aggressive settings broke words and JSON and are off). A candidate for
the owner's dynamic pruning (item 6), measured for quality against the reference (the owner's
rule: differ for speed, never wrong). (2) A learned correction on the routers' look-ahead: of the
needed experts not on the card, the plain router held 77-79 / 66-69 / 55-58 % at 1 / 2 / 4
layers, the learned one 85-88 / 76-79 / 66-71 %; but read ahead only 29 % of the guesses were
used (95 % unused as read-ahead rows): precision is the problem, as Ares measured (3-8 % in
decode). (3) K/V from RAM past 64k with 32k on the card (as Ares's far pages); two I/O threads,
next layer first; a pinned RAM tier with PCIe balance (+9 %); swaps across layers (one run). For
Inkling 2 Mini it points to another fork, UnbiasedStrata (not read). To be read in its code
before stand-ins are built; measure in the SSD comparison (item 3).

### TokenWing (2026-11-17)

Reference now in the open (2026-11-17): Run My AI's engine TokenWing, MIT, built on llama.cpp,
https://github.com/runmyai/tokenwing (`docs/how-it-works.md`, `docs/architecture.md`,
`docs/settings.md`); it runs Qwen3.8 Flash-Next (IQ3_S and UD-Q4_K_XL). Its README: expert reads
with O_DIRECT through 8 I/O threads and a priority queue; the routers of layers L+1 and L+2 on
layer L's hidden state (75.7 % of picks predicted one layer ahead, 67.2 % two, on Ling 3.0
Flash); a RAM cache per layer (`--moe-stream-window`), a VRAM cache that follows the text (up to
64 experts swapped every 8 tokens, `--moe-stream-vram-follow`); use counts halved every 64
tokens, the lowest leaves first; prompts layer by layer, so each expert is read about once per
prompt (0.98 to 1.00 reads per distinct expert). To be read in its code before items 3 above
(large prompt chunks) and the prediction's next step are built.

### Basalt (2026-11-18)

Reference to read (2026-11-18): Basalt, https://github.com/jesdga95/basalt (MIT, a fork of Strata,
kernels from Infernix; NVIDIA Blackwell only, so it cannot run on these cards): Qwen3.8 Flash-Next
with MTP, Qwen3.8-27B with DFlash 2 and DeepSeek-V4.1-Flash with DSpark, drafting on by default;
OpenAI and Anthropic routes, a router, images, documents (PDF, text) and video (NVDEC for
Flash-Next); experts that do not fit in VRAM in RAM; K/V in int8. Its README gives no speeds.
