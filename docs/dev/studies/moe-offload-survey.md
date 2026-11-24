# MoE offload in the engines Run My AI names

Read on 2026-11-10, for the question whether Ares should take more from the
engines that https://runmyai.org/how-it-works credits. Read: that page again
in full (the HTML, its data script `assets/data.js`, and the FAQ and roadmap
of the same site); ik_llama.cpp (README, `docs/parameters.md` and about 30
pull requests, `main` of that day); Fiddler (README and the paper, arXiv
2402.07033v3); ktransformers (README, `kt-kernel/README.md`, the expert
scheduling and native precision tutorials, and the abstract of its SOSP '25
paper on the authors' page; the paper itself was not read, and the LMSYS post
about it only through a summarising fetch); BELLS (README of `bells-next`);
AI-DER (README 1.5.2, `engine/c/tier.h`, `engine/c/qwen38_tier.h`; the
analysis document its header cites is not in the repository); llama.cpp pull
requests 29887 and 30112; TheCodacus' pull request 22; the papers FATE (arXiv
2502.12224v2), ProMoE (2410.22134v3) and Pre-gated MoE (2308.12066v3) from
their HTML versions, by section. Nothing was unreachable. Nothing was built
or run: every number is the named source's own measurement ("stated").

## What "layer chunking" is

No source uses that term. What is meant is the prefill order that reads each
layer's experts once per prompt instead of once per batch of tokens:

- **Run My AI**, section 11, "Layer-major prefill": "The whole prompt passes
  through one layer before the next starts, so each layer's experts are read
  about once per prompt (0.98 to 1.00 reads per distinct expert)." Within a
  layer, "Expert waves": "If a layer needs more distinct experts than it has
  slots, they're processed in waves that fit: ceil(min(n_expert, n_tokens × k)
  / n_slots) waves, each expert read at most once." The FAQ says "Prompts run
  layer by layer".
- **TheCodacus' fork**, pull request 22, "Layer-major prompt processing
  (`--layer-major N`)": "each layer's weights are read once per prompt,
  whatever the ubatch size, and the GPU computes that layer for every ubatch
  before moving on". `N` is the number of layers in a group, so this is the
  variant that literally works on chunks of layers.
- **ktransformers**: "Layerwise Prefill". **BELLS**: "prompt staging".
- Different things with "chunk" in the name: `--chunked-prefill-size`
  (SGLang and ktransformers: the most tokens per prefill batch), ProMoE's
  "chunked prefetching" (a guessed read cut into pieces, which `ares-io`
  has), and Strata's prefill in chunks of 2 048 to 8 192 tokens through all
  layers.

Ares does not have it. In `ares-sched` a step runs its rows through every
layer once, and the architecture document plans read-ahead for prefill but no layer-major
order and no waves. It is recorded only in runmyai-notes.md and
codacus-notes.md.

## Run My AI (TokenWing), the page again

The source is still closed (it opens on 2026-11-17). The page still gives no
tokens per second: the site's script can render tables of reply and prompt
speed against llama.cpp, but the data file holds none, and section 14 lists
only the test laptop (Ryzen AI 9 HX 370, 32 GB LPDDR5X-8000, RTX 5070 Ti
Laptop 12 GB, WD SN5000S). Stated effects without a table: reading ahead "cut
the SSD wait per token by about two-thirds", layer-major prefill made long
prompts "about 4x faster".

### Who it names, and for what

| Source | What TokenWing says it took |
| --- | --- |
| llama.cpp | The base: GGUF, quantization formats, kernels, server |
| Pre-gated MoE, FATE, ProMoE (papers) | Predicting a layer's experts with that layer's own router on an earlier hidden state |
| ktransformers, BELLS | The hottest experts in VRAM, computed by the GPU while the CPU computes the rest |
| Strata | Filling the expert cache per layer from routing counted on real text |
| Fiddler | Computing an expert on the CPU where it sits instead of copying it to the GPU |
| AI-DER | The rule for swapping VRAM experts as the text moves |
| ik_llama.cpp | Grouped expert selection as one GPU op; gate, up and activation as one CPU step with one work queue over all experts of a layer; updating the linear-attention state in place |
| ktransformers | The idea of a multi-row CPU expert kernel with software prefetch; kernel validation against a scalar reference with odd shapes and byte-identical output |

The FAQ also compares with Strata (stated: reads ahead only on Linux and only
when RAM holds at least 55 % of the experts; drops low-ranked experts that are
not in memory by default) and MoE4All (stated: reads experts only on demand).
Neither claim was checked.

### What runmyai-notes.md does not have

- **Correction.** The fall of demand reads from 0.87 to 0.44 per token is not
  the effect of prediction as a whole. It is the effect of predicting the
  first streamed layer of a token from the up to two non-streamed layers
  before it (dense layers, layers kept whole in VRAM).
- **Prediction input.** Adding the layer's shared-expert output to the hidden
  state "cut misses by 11 %" on Ling. The predictions for L+1 and L+2 are one
  fused op. Width stays at top-k: a prediction 64 wide reaches only 90.8 %
  recall and "barely moved the SSD wait". The distance is a setting; two
  layers beat one and three on the test laptop, and the reason given is that
  one layer computes faster there than one expert read takes.
- **Slots for guesses are bound late.** A prediction waits on a list sorted by
  distance and rank. It gets a RAM slot only when an I/O thread has no demand
  read and no started prefetch to work on, "so predictions don't evict cached
  experts early". An expert predicted in this token for a layer that has not
  run is never evicted by a prefetch, and by a demand read only as a last
  resort. A prefetched expert nobody used has value 0.
- **Caps.** At most 2 prefetch chunks in service while a demand read is, at
  most 4 otherwise. Demand reads are not split: splitting was slower, because
  each request has a fixed cost in the I/O stack.
- **Routing reaches the CPU without a sync.** Inside the launch that routes,
  the GPU writes the selected ids and the predictions into pinned host memory:
  the data, a memory fence, then a counter. The CPU watches the counter.
- **Gate and up before down.** Each projection has its own ready flag; gate
  and up are read first and computed while down is still being read.
- **Eviction evidence.** On recorded routing, perfect hindsight (Belady) would
  cut plain LRU's misses to 0.57x, and no cheap rule came close. During a
  prompt the cache uses LRU, and each prompt batch ages the counts by its
  token count.
- **Prefill details.** Experts outside the current wave point to all-zero
  slices and are skipped. Large waves are computed on the GPU: the wave's
  experts are copied from RAM to VRAM, and the next step's experts into a
  second set of VRAM places meanwhile. From 4 096 tokens the most-used experts
  of the next 2 streamed layers are read ahead from a usage table (98.5 % of
  them turn out to be needed). The prefill compute buffer can borrow VRAM from
  the expert cache; on small GPUs that made long prompts slower, so it is a
  switch.
- **Hand-over to decode.** The last prompt batch orders its waves by recency
  (half-life 128 tokens) so the experts the end of the prompt used stay in
  RAM. After 500 ms without a request, empty RAM slots are filled from the
  usage table.
- **Memory.** RAM slots are sized from free RAM minus 0.5 GB minus the
  engine's other memory. A prompt's working memory counts against the limit:
  the cache gives back slots (empty first, then coldest) and takes them back
  at the first decode token. Each GPU has its own expert cache for the layers
  it holds.
- **Known limits.** First reply tokens are slow after prompts of 16k tokens
  and more. One SSD only.

## ik_llama.cpp

A llama.cpp fork (MIT), CPU and CUDA only. Its placement of experts is static
or a byte-budget LRU cache; it streams from disk through the page cache. A
search of its pull requests finds no direct I/O, no router prediction and no
layer-major prefill.

- **What TokenWing took.** Grouped expert routing for the Ling models as one
  CPU kernel (PR 836) and as CUDA kernels (PR 838, "a few percent" slower
  than plain top-k). Fused gate, up and activation (PR 229, `-fmoe`): the
  three expert products each searched the routed experts and quantized the
  same input; fused, DeepSeek-Lite gains about 26 % prefill and 7 % decode on
  CUDA, 3-4 % and 1-2 % on the CPU (ikawrakow, RTX 4080). `-muge` (PR 1137)
  also merges the gate and up tensors into one at load: 6-7 % more prefill on
  a TQ2_T "Qwen-3.8-Next" on two RTX 3090 (PR 2822).
- **Placement.** Tensor overrides, `--cpu-moe`, `--n-cpu-moe N`. With one GPU
  the experts of the last N layers go to the CPU instead of the first N,
  about 10 % more prefill at the same VRAM (PR 2262).
- **GPU cache of host experts** (`--moe-cache-mib`, PR 2644, a port of
  llama.cpp 29887 and 30112): LRU. Qwen3.8-Flash-Next on one RTX 3090 at
  equal VRAM: 39.0 to 48.3 tokens/s against whole layers; +19 % on two cards
  with a layer split; +1 % on Qwen3.6-35B with the tensor split.
- **When expert products go to the GPU** (PR 520): only if the batch has at
  least `32 * experts / active experts` tokens, that is 32 rows per expert on
  average. DeepSeek-Lite prefill at batch 128: 345 to 723 tokens/s.
- **Only the activated experts are copied** for such a batch (`-ooae`,
  PR 698): prefill 1.33-1.68x on GPT-OSS-120B, nothing on DeepSeek-Lite, where
  a large batch activates every expert.
- **From disk.** `--defer-experts` leaves expert pages unmapped at load.
  `--prefetch-experts` (PR 2101, 2667) has `pread` workers warm the page cache:
  the routed slices in decode, the whole expert tensors of the next 3 graph
  splits in prefill, marked cold afterwards. Models larger than RAM: prefill
  1.9-2.7x (64 GB RAM, RTX 4080 SUPER), CPU-only decode 2.79 to 5.47 tokens/s.
  `--defer-ple` keeps the hashed-row table out of resident memory (23.4 GiB
  less at load on Qwen3.8-Flash-Next).
- **CUDA expert kernel** (PR 2831): one block per row tile and expert, each
  running all column tiles of its expert, instead of a stream-k split with a
  fix-up kernel: prefill -1.9 to +18.5 % over eleven MoE models (GB10, RTX
  4070).
- **CPU kernels.** Weights left in RAM can be repacked at load into
  row-interleaved types (4, 8 or 16 rows per block, `-rtr`). The 16-row
  AVX-512 layouts give 21-29 % more prompt speed than the 8-row ones for Q4_0,
  Q5_0, Q6_0, IQ4_NL and MXFP4 (PR 2807, Xeon 6248R). The README warns that
  repacked tensors can no longer be sent to the GPU for large batches where
  the type has no CUDA kernel.
- **`qwen4exp`.** Prefill attention reads a list of selected rows instead of a
  dense mask (PR 2738, used once the cache holds 8 times the selection width;
  PR 2801, 8-9 % more prefill at 192k-256k on GB10). The indexer's top-k is a
  radix select of four 8-bit passes instead of a full sort, which was about
  7 % of a batch at 64k (PR 2767).
- **Several GPUs.** Split mode `graph` puts every layer on every card with f16
  exchange and, as an option, a thread per device; "much better" with a driver
  that allows peer-to-peer access (PR 1089). GLM-4.5-AIR on four RTX 3090 at
  100k context: decode 2.4x the layer split (PR 1067).
- **Lossy.** `-ser Kmin,t` always runs the top `Kmin` experts and the others
  only if their probability exceeds `t` times the top one's: 5-7 % more decode
  at `4,0.2` or `5,0.4` on DeepSeek-Lite (PR 239).

## Fiddler

A PyTorch prototype for unquantized Mixtral-8x7B (ICLR 2025, last change
2024-11). Its content is one decision rule.

- An expert that is not in VRAM is either computed on the CPU (the activations
  travel) or copied to the GPU (the weights travel). Copying one expert takes
  2 to 5 times as long as computing it on the GPU; copying activations is
  under 1 % of the latency.
- The choice is made per expert from the number of rows routed to it: GPU time
  is modelled as a constant, CPU time as linear in the rows, both measured at
  start. One row goes to the CPU, a long prompt's rows go to the GPU.
- Experts are placed in VRAM by popularity from an offline profile. On Mixtral
  that is worth little: 56 of 256 experts give an expected hit rate of 25.2 %
  against 18.7 % for the worst choice.
- Stated against llama.cpp b2956: 1.26x for single requests, 1.30x for long
  prefill, 11.57x for beam search.

## ktransformers

CPU kernels for experts (AMX, AVX-512, AVX2) under SGLang, for servers with
hundreds of GB of RAM (Apache-2.0). Stated in the paper's abstract: prefill
4.62-19.74x and decode 1.25-4.09x over earlier hybrid systems.

- **Expert deferral** (`--kt-max-deferred-experts-per-token`): some of a
  token's CPU experts are finished later and their output is added at a later
  point of the residual, so the CPU works while the GPU runs the next
  attention (the mechanism is from the LMSYS post, second-hand). Stated: CPU
  utilisation from below 75 % to almost 100 %, up to 1.45x throughput, at most
  0.5 % average accuracy drop; the README calls 1-4 deferred experts the
  useful range and warns of "noticeable accuracy loss" at 5-7. It changes the
  output.
- **Two prefill paths**: below a token threshold the CPU and GPU share the
  work; above it, "Layerwise Prefill" runs on the GPU alone and transfers the
  CPU's weights layer by layer, which needs one MoE layer of extra VRAM (about
  3.6 GB for MiniMax-M2.1, 9 GB for Kimi-K2-Thinking).
- **Dynamic expert update**: the routing counted during a layerwise prefill
  re-decides which experts are in VRAM. Qwen3-Next-80B on four RTX 4090, 10 %
  of the experts on the GPU: 56.6 tokens/s with a uniform choice, 58.6 by
  profile, 70.2 with the update; at 20 %: 60.3, 61.9, 74.7. From 70 % on the
  static profile is ahead (107.2 against 95.0 at 90 %).
- **NUMA**: one thread pool per node, each expert's weights split over the
  nodes' local memory (up to 63 % more decode on two sockets, second-hand
  from the LMSYS post).

## BELLS

A llama.cpp fork (MIT) with a per-layer VRAM cache of experts. By default
every routed expert is computed on the GPU and a miss is uploaded first.

- **Unequal slots per layer** (`BELLS_LAYER_PROFILE`): the slots are divided
  according to recorded lookup counts per expert instead of equally; "Layers
  with diffuse routing get more slots". No numbers.
- **Prompt staging**: a large prompt batch bypasses the cache. Each layer's
  full expert stack is placed in a staging buffer just before the layer runs,
  the next layer's on a second thread meanwhile; experts already cached are
  copied inside VRAM. It costs two buffers of one layer's experts (2 x 0.66
  GiB for Qwen3.8-Flash-Next Q2_0). Stated on an A10G: about 200 to 700-850
  prompt tokens/s.
- **Second GPU as a victim cache** (`--bells-l2-slots`): an expert evicted
  from the computing card moves to an idle card and comes back by a
  device-to-device copy.
- **Split** (`--bells-split K`): K of a token's experts on the GPU, the rest
  on the CPU at the same time.
- **Co-occurrence predictor** (`BELLS_GLOBAL_PREFETCH`): learns online which
  expert of a later layer follows which expert of an earlier one, up to 4
  layers ahead, and uploads candidates above 90 % confidence; links whose
  measured precision is under 80 % are dropped. No numbers.
- Also: adaptive swaps every N tokens with decaying counts, slots sized per
  GPU by probing real allocations, cached experts stored in a smaller
  quantization (lossy, experimental).
- Stated: Qwen3.6-35B Q4 on an RTX 3060, 32.5 to 60.9 tokens/s against all
  experts on the CPU; behind Strata on a 24 GB card in its own comparison.

## AI-DER

A gateway over five engines with an own CUDA engine for Qwen3.8-Flash-Next
(no license file). On the same model the own engine is behind llama.cpp and
Strata in the project's own table (27-33 against 35 and 56-73 tokens/s).

- **Swap rule** (`tier.h`): promote when `hot > cold + cold/4 + 4`; counts
  halve at each decay. Eviction score is frequency first, recency only as a
  tie-break (`heat << 8 | recency up to 255`).
- **Decode**: the GPU runs the chain for the resident subset of the routed
  experts and returns a bit mask; the CPU computes the others meanwhile.
- **Prefetch ensemble** (`qwen38_tier.h`): a routed expert that is not
  resident is queued for a low-priority upload at once, ahead of the 16-token
  swap tick; the 16 hottest experts per layer are pinned; and a table from
  the hash of the last two tokens to the likely experts of layers 0 to 7 is
  consulted as soon as the next token is known, before its layer 0 runs.
  Stated for the speculative uploads (the first and the third): +6 % on a 2k
  prompt of a different domain, -2 to -4.5 % in warm decode, so they are on
  for prefill and off for decode. These are uploads over PCIe, not SSD reads.

## llama.cpp

Mainline has the GPU cache for host experts since 2026-10-07 (PR 29887, from
qvac-fabric; several GPUs in PR 30112): LRU, for batches of at most 32 tokens,
"10 % of the total expert size is good". Qwen3.8-Flash-Next Q4_0 on an RTX
5090: 30.8 tokens/s with whole expert layers in VRAM, 54.5 with fewer layers
plus a 6.4 GB cache, 67.8 with no whole layers and an 18.6 GB cache (hit rate
89 %). On two RTX 4090 the cache alone gives 1.85x decode but 0.73x prefill;
batches above 32 tokens take the regular offload path, and then every expert
is on the host.

## The papers

- **Pre-gated MoE** trains a new gate that selects the experts of the next
  block, during fine-tuning. It changes the model.
- **FATE.** The next layer's gate on the current gate input predicts 78.8 % of
  the experts in decode and close to 100 % in prefill, where the tokens'
  predictions cover each other; fetching every expert above the 75th
  percentile of the predicted weights reaches 97.15 %. Prediction is worst in
  shallow layers, where routing is flat, so the cache gives them more room:
  with all experts of layers 0-3 cached the hit rate is 99.08 %. Eviction is
  ARC. In prefill the experts are transferred in
  descending order of the tokens routed to them and computed in that order.
  In decode the number of experts to prefetch is the layer's compute time
  over one expert's transfer time. In prefill, unpopular experts are
  transferred as int2 and the others as int4 (lossy). Stated: up to 4.5x
  prefill and 4.1x decode over load on demand.
- **ProMoE.** A two-layer MLP per layer (about 2 million parameters, trained
  in 1-2 hours) predicts 84.7 % on average, where the next layer's own gate
  reaches only 66.9 % on one of the two models; predicting one layer further
  costs it 5 %. Prefetches are cut into chunks (one per matrix) so a miss
  waits for one chunk at most; a miss is issued the moment the gate has run
  ("early preemption"); and the experts of a layer are computed in the order
  cached, being fetched, not yet started ("reordered inference"). In its
  llama.cpp integration most of the prefill gain comes from the reordering.
  Stated: 2.20x prefill and 2.07x decode on average over existing offloading.

## Seen in passing

- **MoE4All** (github.com/Headmaster218/MoE4All): a Vulkan engine with VRAM,
  RAM and SSD tiers whose GPU decides residency itself. Only its README was
  read. It is the nearest relative of Ares' Vulkan backend and deserves notes
  of its own.
- **Strata** has documents newer than strata-notes.md or not covered there:
  keeping a helper GPU's experts out of the primary cache's candidates (about
  1 900-2 000 duplicate slots observed), batched expert uploads through
  `cudaMemcpyBatchAsync` (CUDA 13), an expert cache in segments that can be
  given back.

## For Ares

### Already in Ares

- Hot experts in VRAM, GPU and CPU at the same time, results summed (Fiddler,
  ktransformers, BELLS split, AI-DER): `docs/dev/architecture/weights.md`, "Weights", `ares-weights`.
- CPU or upload by a measured cost model (Fiddler's rule, ik_llama.cpp PR
  520): `docs/dev/architecture/weights.md`, "Weights". The detail to keep: the variable is the rows
  routed to one expert, with GPU time constant and CPU time linear.
- Counts with decay, swaps with a margin and a cap (AI-DER's rule, BELLS,
  ktransformers' update, the LRU caches of llama.cpp and ik_llama.cpp):
  `ExpertHeat`, `SlotSet::elect`, `SwapRule`. Seeding from a profile:
  `ExpertHeat::seed`.
- A cache over all layers instead of whole layers on one side (llama.cpp's
  numbers confirm it for decode): no layer setting exists.
- Only routed experts move (`-ooae`): by design.
- Demand before prefetch, chunked prefetch, cancel and promote (ProMoE,
  TokenWing), direct I/O in place from the GGUF: `ares-io`. Reading mirrored
  model files from several drives goes beyond TokenWing, which reads one.
- Router prediction of the next layer (FATE, ProMoE's baseline, TokenWing):
  planned in `docs/dev/architecture/weights.md`, "Weights".
- Hashed-row tables read on demand (`--defer-ple`): `ares-io` rows.
- Page-locked staging and mapped weights (`--cpu-moe-pinned`,
  `GGML_CUDA_NO_PINNED_WEIGHTS`): `ares-cuda`.
- Tensor split with peer-to-peer sums and a thread per device (split mode
  `graph`): `ares-runtime`.
- Sparse attention from a row list without a dense mask, radix select for the
  indexer: planned (strata-notes.md, engines-survey.md).
- One RAM limit under which the expert cache gives way to a prompt:
  `ares-core` broker.
- Lossy pruning as an option: `SlotSet::write_route_bias`. ik_llama.cpp's
  threshold rule is another form of the same trade.
- Kernels checked against a scalar reference: every kernel has one.

### Worth taking

1. **Layer-major prefill with expert waves.** A prompt step that covers a
   range of layers for all prompt rows, instead of all layers for a range of
   rows. Within a MoE layer the routed experts are grouped into waves that
   fit the staging space, each expert read or uploaded once, most-routed
   first. This matters without an SSD tier too: with experts in RAM, a prompt
   in row chunks uploads most of every layer's experts once per chunk over
   PCIe. Where: a second step shape in `ares-sched` and the conductor, a wave
   planner in `ares-weights` (routed counts and slots in, waves out), no
   change in `ares-io`. Cost: the residual of all prompt rows is kept between
   layer groups (TheCodacus keeps it on the device when it fits, else in
   pinned memory); recurrent state and K/V advance per layer over the whole
   prompt; generating requests get their turn between layer groups; two
   staging sets of one layer's experts in VRAM, to be booked in the ledger or
   lent by the expert tier. It belongs with the hand-over to decode: an
   election from the prompt's own routing, weighted to its end, before the
   first reply token. Stated: 954 GiB read for a 4k prompt in batches of 16
   tokens without it and 0.98-1.00 reads per expert with it, "about 4x" (Run
   My AI); 16 to about 300 tokens/s on a 16.7k prompt (TheCodacus); 200 to
   700-850 (BELLS); +20 % at 10-20 % of experts in VRAM from the re-election
   (ktransformers).
2. **The RAM tier as a cache in front of the SSD.** `SlotSet` only elects: it
   has no way to admit an expert whose read just completed, it evicts at the
   moment of the decision, and it cannot protect an expert that this token
   will need. For SSD streaming it needs admission on arrival with the
   coldest unprotected member as the victim, a victim chosen when a guessed
   read starts and not when the guess is made, protection of experts
   predicted for layers of the current token, and value 0 for a guess nobody
   used. Where: `ares-weights`, small. No isolated numbers; Run My AI states
   that the best possible eviction would have 0.57x the misses of LRU, which
   is the room there is.
3. **Order inside a layer.** Compute the experts that are present first, then
   those arriving; issue a miss the moment the router has run; read gate and
   up before down and start on them when they land. Where: the expert pipeline
   of the runtime; `ares-io` already reads one matrix per request. Cost:
   small. Stated: most of ProMoE's prefill gain in llama.cpp (2.20x in all),
   part of FATE's 4.5x.
4. **Slots per layer by need, and a token table for the first layers.**
   Divide a tier's bytes over the layers by the gain of the next slot from the
   heat profile instead of equally; early layers route flatly and are
   predicted worst, so they get more (FATE: 99.08 % hits with layers 0-3 fully
   cached; BELLS gives layers with diffuse routing more slots; ik_llama.cpp
   keeps the early layers' experts on the GPU since PR 2262). The router
   prediction has no input for the first MoE layers of a token; a table from
   the last two token ids to their likely experts fills that gap and can be
   consulted when the token is sampled, like the hashed-row reads (AI-DER:
   +6 % prefill, -2 to -4.5 % decode over PCIe; Run My AI: predicting the
   first streamed layer halves its demand reads). Where: `ares-weights`
   (allocation, small) and a profile artefact per model (the table). The
   table must be measured on the SSD tier before it is kept.
5. **Details of the planned prediction.** Add the shared expert's output to
   the predictor's input (11 % fewer misses, Run My AI); make the distance a
   measured setting with two layers as the start; keep the width at top-k.
   Publish routing and predictions from the router kernel into mapped pinned
   memory with a counter, so the I/O and CPU side see them without a stream
   sync. The way back (the CPU's sum entering a step that was queued without
   waiting) is an open point of the step design that the first MoE model must
   settle. Cost: small on CUDA; on Vulkan the visibility of a host-visible
   buffer during a pass needs a test.
6. **Expert kernels.** Gate and up of an expert adjacent in its slot and
   computed in one pass with the input prepared once (ik_llama.cpp: 26 %
   prefill and 7 % decode on CUDA from the fusion, 6-7 % more from merged
   tensors); for prefill one block per expert and row tile (up to 18 %). For
   the CPU side, ik_llama.cpp's row-interleaved layouts are the reference,
   but a repacked copy is not the bytes the GPU reads, so this is a
   measurement to make when the CPU kernel exists, not a decision. Cost: the
   slot layout is free if decided before the kernels are written.

### Not applicable

- Pre-gated MoE, ProMoE's learned predictors, TokenWing's correction file:
  they need training per model; the router itself is the predictor.
- Expert deferral: it changes the output. If the CPU ever limits decode it is
  an opt-in of the same class as route-bias pruning.
- int2 or re-quantized experts in the cache (FATE, BELLS): Ares computes with
  the file's weights.
- Co-occurrence predictors (BELLS): FATE and ProMoE measure path-based
  prediction below the router on the previous hidden state.
- Page-cache streaming (`--prefetch-experts`, `--defer-experts`, mmap in
  BELLS and AI-DER): replaced by direct I/O.
- Layer counts and solvers for placement (`--n-cpu-moe`, `--fit`): no layer
  setting exists.
- A second GPU as a victim cache: every card computes.
- NUMA pools and mirrored weights, AMX kernels: server CPUs; the CPU here
  only computes missed experts.
- TokenWing's own file layout with 4 KiB alignment: `ares-io` widens reads
  on the GGUF instead.
- Reloading tensors from changed files, observing a subset of layers to save
  graph splits, beam search: tools or workarounds for llama.cpp's graph.
