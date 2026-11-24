# How the serving engines build their fast parts

A second survey, made on 2026-11-10, of vLLM, SGLang, FlashInfer and the
engines around them, this time in their code: which data structure, which
kernel, which file. The first survey ([engines-survey.md](engines-survey.md))
lists features; what it already says is not repeated here, and its errors are
corrected at the end. Engines that stream MoE experts are in
[moe-offload-survey.md](moe-offload-survey.md) and are left out.

**How to read it.**

- *(read)* means read in the source at the commit named below, by the author
  of this survey or by a reader working for it. Each engine's section says
  which. *(stated)* means the project's own documentation, comments or paper
  say so. *(calc)* is arithmetic done for this survey from stated or measured
  numbers. Nothing was built or run: **no number here is verified**, and every
  number is its authors' unless it is marked *(calc)*.
- Each technique has the same fields: what it does, **where** it lives, the
  **numbers** its authors give, what it costs in **VRAM**, and the verdict for
  **Ares**: *have* (with the crate), *take* (where it goes, what it replaces,
  what it costs) or *leave* (why).
- The verdicts are held against Ares' fixed points: GGUF files as input with
  the weights kept at their quantized size in VRAM (rearranged at load where
  that helps, never expanded); quantized K and V in pages from one pool and
  nothing preallocated per request; CUDA and Vulkan; one to a few consumer
  GPUs; recurrent hybrid models with dense or block-sparse attention, MoE and
  dense.

**Commits read.**

| Repository | Commit | Read by |
| --- | --- | --- |
| vLLM (`~/Services/vllm-src`) | 8530a77 | Scheduler, prefix cache, memory budget, attention call, executors: directly. Kernels, speculation, grammar, sampler: delegated |
| IST-DASLab/marlin | 1f25790 | Delegated |
| sgl-project/sglang | b2616ca | Delegated; scheduler loop and policies also directly (at 67ff3b3) |
| flashinfer-ai/flashinfer | 7752881 (0.7.3) | Delegated |
| NVIDIA/TensorRT-LLM | d9e44e9 | Delegated |
| mlc-ai/mlc-llm, with mlc-ai/relax 629cc65 and apache/tvm-tirx 198a045 | 44fd06d | Delegated |
| InternLM/lmdeploy | cf52797 | Delegated |
| turboderp-org/exllamav3 | bd1f519 | Delegated; cache and sampler files also directly |
| EricLBuehler/mistral.rs | fb4cdc3 | Delegated |
| huggingface/text-generation-inference | b4adbf2 | Delegated |
| aphrodite-engine/aphrodite-engine | 95f3174 | Delegated |
| lightseekorg/tokenspeed | e2b673a | Delegated; design documents also directly |

The file and constant references of the delegated readers were spot-checked
(Marlin's `dequant.h` constants and `m_block_size_8`, ExLlamaV3's
`AUTO_RECONSTRUCT_THRESHOLD`, TensorRT-LLM's `fusedSampling/`, LMDeploy's
`convert.cuh`, FlashInfer's `qsa_nvfp4_attention.cuh`, TokenSpeed's
`tail_table.h`, MLC's `batch_jumpforward.cc`); all were where the reports
said. What nobody read is listed at the end.

## vLLM

Paths are relative to the vLLM checkout.

### Scheduler and continuous batching

**One loop without phases** *(read)*. A request has `num_computed_tokens` and
`num_tokens_with_spec`; every step hands out one token budget so that the
first catches up with the second. Running requests are served first, then
waiting ones. A long prompt's share of a step is capped
(`long_prefill_token_threshold`), and the cap is lifted when it is the only
request. Chunked prefill, prefix hits and drafts are all the same thing to
this loop: tokens that are not computed yet.

- Where: `vllm/v1/core/sched/scheduler.py` `Scheduler.schedule` (698; running
  requests from 792, waiting ones from 1048, the cap at 753-767, 843 and
  1331). Queue policies are `fcfs` and `priority` only
  (`vllm/v1/core/sched/request_queue.py:13`).
- Numbers: none stated. Defaults: 2048 tokens per step
  (`vllm/config/scheduler.py:41`).
- VRAM: none.
- Ares: **have** (`ares_sched::plan` is this loop with decision rows, drafts
  and a time model on top).

**Blocks are taken per step, and the newest request pays for a shortage**
*(read)*. `allocate_slots` takes exactly the blocks this step writes (plus a
lookahead for drafts). Nothing is reserved for a request's future output.
When the pool is empty, the last request of the running list (or the one of
lowest priority) is preempted: all its blocks are freed, its
`num_computed_tokens` is set to 0 and it goes back to the head of the queue,
to be recomputed from whatever the prefix cache still holds.

- Where: `scheduler.py:905-960` (the loop around `allocate_slots`),
  `_preempt_request` (1798, the reset at 1821).
- Numbers: none stated.
- VRAM: none; the cost is recomputation after a preemption.
- Ares: **take** the first half into `ares-runtime` and `ares-sched`: pages
  are claimed per step, not reserved per request at submission. It replaces
  `Runtime::worst_case_pages` (`crates/runtime/src/engine.rs`), which reserves
  prompt plus `max_tokens` (the whole context when the client names no limit)
  and answers `Overloaded` otherwise. **Leave** the recompute: a sequence that
  loses is spilled to RAM or suspended, which Ares already plans. The
  admission rule to go with it is SGLang's, below.

### Prefix caching

**A chain of block hashes, registered while the request runs** *(read)*. A
block of 16 tokens is named by a hash over its parent's hash, its tokens and
extra keys (adapter, images, a salt). A block is registered the moment it is
full, so a second request with the same prompt shares it at once. Freed
blocks keep their hash and wait in a free list that is threaded through the
blocks themselves (two pointers per block, no other structure), in the order
least recently used first and, within one request, the tail of the chain
first. A cached block is given up only when the free list hands it out
again.

- Where: `vllm/v1/core/kv_cache_utils.py` `hash_block_tokens` (709),
  `FreeKVCacheBlockQueue` (260); `vllm/v1/core/block_pool.py`
  `cache_full_blocks` (230), `get_cached_block` (202), `touch` (792),
  `_maybe_evict_cached_block` (769).
- Numbers: none stated.
- VRAM: none (host: a hash and two pointers per block).
- Ares: **have** the chain and the leaf-first order (`ares-state`). **Take**
  the registration when a page fills. `PrefixIndex::insert` can be called at
  any time, but ARCHITECTURE.md only says that the index "holds pages after
  their sequence ended" and the runtime has no prefix cache yet; if pages
  enter at the end, two requests that start with the same prompt at the same
  time share nothing. SGLang does the same after every prompt chunk (below).

**Hits inside a block** *(read)*. For models with recurrent state stored at
aligned positions, a block can be made reachable from a boundary inside it:
the entry is extra lookup metadata that points at the existing block, without
a copy.

- Where: `block_pool.py` `cache_partial_block` (486);
  `kv_cache_utils.py` `partial_hash_hits_enabled` (764).
- Numbers: none stated.
- VRAM: none.
- Ares: **take** the idea as tail entries; TokenSpeed's table is the better
  design for it (below).

### The attention path

**The page table goes into the kernel** *(read)*. There is no gather on the
host and no contiguous copy of a sequence: the attention call takes the whole
K and V pools, a block table per request and the used lengths. Scales for
8-bit K/V are arguments of the same call; a split of long contexts over
several blocks of the grid comes from a plan passed along.

- Where: `vllm/v1/attention/backends/flash_attn.py`, the call at 1404-1428
  (`block_table`, `seqused_k`, `q_descale`/`k_descale`/`v_descale`,
  `num_splits`, `scheduler_metadata`); writes in `csrc/cache_kernels.cu` and
  `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`.
- Numbers: none in tree.
- VRAM: one int32 per block and request for the table.
- Ares: **have** by design (principle 3). The kernels themselves are under
  FlashInfer and LMDeploy below.

**Cascade attention is gone** *(read)*. vLLM once computed a prefix shared by
the whole batch once and merged it with each request's own part. The code
says "Deprecated: cascade attention has been removed"
(`vllm/v1/worker/gpu_model_runner.py:539`,
`vllm/v1/core/kv_cache_manager.py:700`).

- Ares: the merge of attention states is in `ares_quant::kvhead` as algebra.
  That vLLM dropped the batch-wide use of it is a reason to **leave** it out
  of the first attention kernels and to measure before adding it; see
  FlashInfer for where its authors find a gain.

### Quantized weight kernels: Marlin

This is the part of vLLM that matters most for Ares, because it is the
opposite of how Ares' weight kernels are built today.

**Weights rearranged at load into the order the tensor cores read them**
*(read)*. The 4-bit codes are cut into tiles of 16 by 64 and permuted so that
one 16-byte load of one thread holds exactly the codes of that thread's
`mma` fragments, with the nibbles interleaved for a two-instruction decode.
The scales are a separate, aligned f16 array, permuted the same way; AWQ's
zero points likewise. The rearrangement is itself a GPU kernel.

- Where: `csrc/quantization/marlin/gptq_marlin_repack.cu` `repack_tile`
  (79-177), `awq_marlin_repack.cu`;
  `vllm/model_executor/layers/quantization/utils/marlin_utils.py`
  `get_scale_perms` (514), `marlin_permute_scales` (524),
  `marlin_zero_points` (566). In the original: `marlin/__init__.py`
  `_get_perms`, `Layer.pack`.
- Numbers: "0.125 bits storage overhead of the group scales" at groups of 128
  *(stated, README)*.
- VRAM: the same bits as the checkpoint.
- Ares: **take**, as a load step in `ares-cuda` (and its mirror in
  `ares-vulkan`). Today the device holds the GGUF blocks as they are in the
  file and every kernel decodes that layout (`BLOCK_VALUES, BLOCK_BYTES: the
  geometry of a block (GGUF layout)`, `crates/cuda/kernels/mmvq.cuh`), down to
  funnel shifts for 17-byte blocks. The fixed points allow the rearrangement.
  For the K quantizations: a plane of codes in fragment order (the high bits
  of q5_K, q6_K and q3_K permuted alike) and a plane with the 6-bit scales
  and minimums unchanged plus the f16 `d` and `dmin`, aligned. Expanding
  scale and minimum to f16 per 32 weights would cost 0.5 bit per weight,
  about 1.5 GiB on the 13.8 GiB of Qwen3.8-27B *(calc)*, so they stay 6-bit
  and are multiplied in registers. Cost: a transform per type at load; the
  permutation crosses 64 output rows, so an expert read in place from the
  GGUF (`ares-io`) needs the transform when it arrives, or the RAM tier holds
  experts in device order and only the SSD tier in file order.

**Dequantization in registers, weights never decoded into shared memory**
*(read)*. Packed weight bytes and f16 activations go from global to shared
memory through a ring of four `cp.async` stages (two on Turing). A thread
turns a word of codes into f16 pairs in registers: `lop3` puts each code into
the mantissa of 1024.0 (`(q & 0x000f000f) | 0x64006400`), one `__hsub2`
removes the bias, the high nibbles take one `__hfma2`; the group scale is one
`__hmul2`. The instruction is `mma.m16n8k16` on f16 with f32 accumulation.

- Where: `csrc/quantization/marlin/marlin_template.h` `fetch_to_shared` (756),
  `matmul` (981-1089); `dequant.h` (110-142); `marlin_mma.h` `mma` (7).
- Numbers *(stated, README and paper abstract)*: "close to ideal (4x)
  speedups up to batchsizes of 16-32" against f16, "64-128 with gradually
  decreasing" speedup, "up to 2.8x" end to end in vLLM. Read off the README's
  plot for an A10: about 3.87x for 1 to 32 rows, 3.1x at 64, 1.55x at 128.
- VRAM: none beyond the weights; 48 to 64 KB of shared memory per block.
- Ares: **take**; it replaces the weight path of `mm.cuh` (load raw bytes,
  `T::decode`, write a tile to shared memory, `ldmatrix`, one barrier per
  step). That path is what `mm.cuh` itself names as its bound below 64 rows
  ("32 to 70 bytes of each of 64 to 128 rows per step ... a third of its
  bandwidth"). The sub-blocks of the K quantizations fit the template's
  existing shapes: groups of 32 values (q4_K, q5_K) and of 16 (q6_K, q3_K),
  and q4_K's minimum is the AWQ form `fma(w, s, -zp * s)`. For a 17408 x
  5120 q4_K matrix the memory bound is 0.105 ms on the RTX 4070 and 0.147 ms
  on the RTX 3060; at the capped rate of f16 `mma` with f32 accumulation a
  row costs 3.0 and 7.0 microseconds, so the kernel stays memory-bound up to
  about 34 and 21 rows *(calc)*, where today's prefill kernel takes about
  0.3 ms. What does not carry over: the grid codebooks of iq2 and iq3 (they
  keep a decoder with a table in shared memory), and above the crossover the
  GeForce cap halves this path (see ExLlamaV3 for the way around it).

**One family from 1 to 64 rows** *(read)*. For 1 to 8 rows the operands are
swapped: two weight fragments are the 16-row operand and the activations the
8-row one, so nothing is padded to 16. For 9 to 64 rows the activations are
1 to 4 blocks of 16. Above 64 the same problem is run in parallel slices.

- Where: `csrc/quantization/marlin/marlin.cu` (76-90, `marlin_mm` 305-417);
  `marlin_mma.h` `mma_trans` (138); `generate_kernels.py`.
- Numbers: as above; none for the 8-row variant alone.
- VRAM: none.
- Ares: **take** the 8-row variant in place of the integer dot products at 5
  to 8 rows, where `mmvq` falls to 26 % of the memory limit, and one kernel
  up to about 64 rows. **Keep** `mmvq` for the few rows where it is at the
  memory limit (a pass over Qwen3.8-27B takes 21.7 ms with one row and 23.2
  ms with three) and the grouped grid of `mm.cuh` for very wide steps.

**One block per processor, in stripes** *(read)*. Exactly one block per SM is
launched. Each walks a stripe of tiles down a slice of columns and continues
into the next slice, so every SM gets the same work whatever the shape.
Blocks that share a slice add their parts in slice order behind a lock in
global memory.

- Where: `marlin_template.h` (comment at 242-252, `init_part2_slice` 400,
  `thread_block_reduce` 1202, `global_reduce_fp32` 1378, `barrier_acquire`
  151).
- Numbers: "good SM utilization on most matrix shapes" *(stated)*.
- VRAM: `sms * 64 * 256` f32 of temporary, 3 MB at 46 SMs, allocated per
  call.
- Ares: **take** the persistent block per SM for 1 to 64 rows (a sequential
  stream of weights per SM, no launch per tile); the temporary belongs in the
  step scratch. The stripes matter for small matrices (experts, attention
  projections).

**The integer path** *(read)*. With int8 activations the codes become int8 in
registers, the instruction is `mma.m16n8k32` on int8, and the group scale is
applied as an integer (`frag_c += frag_c_tmp * scale`) with one float
multiply per output at the end.

- Where: `dequant.h` (493-502); `marlin_template.h` `matmul_a8` (1091-1196);
  `marlin_utils.py` `marlin_act_int8_process_scales` (544).
- Numbers: none stated.
- VRAM: none.
- Ares: **take** for the int8 prefill path on capped cards. The sub-block
  scales of the K quantizations are 6-bit integers already, so the integer
  accumulation is exact, and the f32 scaling that is 260 of the 500 to 620
  instructions of a step in `mm.cuh` runs once per 256 values instead of once
  per 32 *(calc: an int32 holds eight sub-blocks)*.

**The MoE form** *(read)*. Pairs of token and expert are sorted by expert and
padded to blocks of 8 to 64; each block is one tile whose activations the
loader gathers through the sorted list and whose weight pointer is offset by
the expert.

- Where: `csrc/moe/marlin_moe_wna16/marlin_template.h` (880-892);
  `vllm/model_executor/layers/fused_moe/` `moe_align_block_size`.
- Numbers: none stated.
- VRAM: the sorted list and a temporary.
- Ares: **take** the layout for the kernels that compute a wave of experts
  (one launch for all experts of a layer, the gather in the loader).

**GGUF in vLLM** *(stated; the plugin read through a summary only)*. It is not
in the tree any more: support moved to an out-of-tree plugin and is called
"highly experimental and under-optimized"
(`docs/features/quantization/gguf.md`). The plugin keeps the blocks packed and
runs ports of llama.cpp's own kernels.

- Ares: **leave**; there is nothing in it that llama.cpp does not have.

### K and V quantization

| Format | What it does | Where | Their numbers | Ares |
| --- | --- | --- | --- | --- |
| `fp8` with one scale per tensor *(read)* | 8-bit float; scales from a calibrated checkpoint, else 1.0; read fused in the attention kernels | `vllm/model_executor/layers/quantization/kv_cache.py`; `flash_attn.py:1288-1309` | None stated | **Leave**: `q8` with a scale per group covers it, and an uncalibrated global scale is the weak point |
| Scales per token and head, `int8`, `fp8`, `int4` *(read)* | The writer takes the largest magnitude of a head's vector as its scale; 4-bit after one randomised Hadamard rotation | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` (169); `ops/int4_per_token_head.py`; read in `triton_unified_attention.py` | None stated | **Have** in a finer form (`ares-quant`) |
| TurboQuant, 3 to 4 bits *(read: headers)* | Hadamard rotation, a trained codebook for K, uniform V, a norm correction | `vllm/model_executor/layers/quantization/turboquant/`; `backends/turboquant_attn.py` | k8v4: 2.6x smaller, +1.17 % perplexity; 4-bit: 3.8x, +2.71 %; 3-bit: 4.9x, +20.59 %; without sparing the first and last two layers "GSM8K drops ~30 points on Qwen3-4B" *(stated)* | See next row |
| Octave *(read: design document)* | 64 rotary dimensions at 4 bits and the rest at 3; an int8 codebook so the product of query and key stays an integer dot product; attention runs in the rotated space; the scale is chosen so that the product of a vector with its reconstruction equals its squared norm | `backends/rocm_octave_attn.py`; `docs/design/rocm_octave.md` | 3.06 to 3.88 bits per value; "25-30% below TurboQuant at equal size" in attention-output error on Qwen3.8-27B; a least-squares scale "shrinks every reconstructed vector (by about 3% at 3 bits)" and cost "2-3x more NLL" at 380k tokens *(stated)* | **Take** three rules into the pending long-context measurement of `ares-quant`: the norm-preserving scale against the least-squares fit Ares uses for `q7` to `q4`; more bits for the rotary dimensions of the Qwen heads; an int8 codebook so the integer path serves the product of query and key on capped cards |

VRAM of all of them: the K/V pool at the format's bits; TurboQuant about 230
bytes per token and KV head at 256 dimensions *(calc from its formula)*.

### Speculative decoding

**One interface, many drafters** *(read)*. `propose(...)` returns draft ids
on the GPU; a cache carries proposal distributions where a drafter samples.
Present: n-gram on the CPU (a search on the reversed tokens) and on the GPU,
suffix decoding, a draft model, EAGLE and EAGLE-3 heads, MTP, DFlash, DFlash2,
DSpark and others. There are no draft trees.

- Where: `vllm/v1/worker/gpu/spec_decode/speculator.py` (67);
  `vllm/v1/spec_decode/ngram_proposer.py` (209).
- Numbers: a qualitative table only
  (`docs/features/speculative_decoding/README.md`).
- VRAM: per drafter.
- Ares: **have** (`ares-sched` drafters).

**Suffix decoding** *(read: the wrapper)*. A suffix tree per request over
prompt and output, and one tree over up to 10 000 past requests, on the CPU
in an external library; the draft is as long as a factor times the match and
is cut where the estimated probability falls below 0.1.

- Where: `vllm/v1/spec_decode/suffix_decoding.py`.
- Numbers: none in the docs.
- VRAM: none.
- Ares: **take** the tree across requests as the second lookup drafter in
  `ares-sched` (agents repeat themselves across requests, not only within
  one).

**Verification** *(read)*. Drafts are extra positions of the same request in
the same forward pass. Rejected ones are undone by lowering
`num_computed_tokens`; their K/V slots are overwritten later. A recurrent
layer keeps one state block per draft position and picks one after
acceptance.

- Where: `scheduler.py:2339-2357`;
  `vllm/v1/attention/backends/mamba_attn.py` (122-133, 484);
  `MambaSpec.num_speculative_blocks`.
- Numbers: none.
- VRAM: one recurrent state per request and draft position.
- Ares: **have**, cheaper (window and commit keep no state per draft).

**The rejection sampler** *(read in outline)*. Greedy: the target's pick
equals the draft. A greedy draft under sampling: the target samples with
Gumbel noise hashed from seed, position and token, and the draft is accepted
when it is that pick. Sampled drafts: the usual ratio test and a resample from
the residual.

- Where: `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`,
  `rejection_sampler_utils.py`; `vllm/v1/worker/gpu/sample/gumbel.py`.
- Numbers: a lossy variant ("FLy") gives 1.04 to 1.31x over exact
  verification at 96.3 % or more of the accuracy *(stated,
  `docs/features/speculative_decoding/fly.md`)*.
- VRAM: f32 rows of the vocabulary, in chunks of 1 GiB.
- Ares: **have** (principle 6 is their second case). **Leave** the lossy
  variant and block verification while drafts are not sampled.

**A budget of draft rows across requests** *(read: headers and docs)*. Every
draft position of every request is scored by the probability that it
survives (the product of the per-position estimates); the best positions
across all requests are verified, up to a budget from a cost profile.

- Where: `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`,
  `acceptance_estimator.py`.
- Numbers: none.
- VRAM: one float per request and position.
- Ares: **taken** in `ares_sched::choose`: the rows of all windows are ranked
  by their chance and the step takes the number of them that gives the best
  tokens per millisecond by the price of a step measured at load; the
  measurement that a ninth row costs a second launch at eight requests is this
  trade.

### Structured output

**Where the time goes** *(read)*. Grammars are compiled on a thread pool and
cached. The mask of a step is filled on the engine thread while the forward
pass runs (the pass is started without blocking, then the mask, then the
sample); a pool of threads is used only above 128 constrained requests. The
mask is a row of `vocab / 32` words, uploaded on its own stream and applied
by a kernel. Jump-forward is a TODO in both main backends.

- Where: `vllm/v1/structured_output/__init__.py` (`grammar_init` 134,
  `grammar_bitmask` 388); `backend_xgrammar.py` (216);
  `vllm/v1/engine/core.py` (696-704).
- Numbers: none.
- VRAM: about 1 MB for 32 rows at a vocabulary of 248k.
- Ares: **take** the overlap (grammar work during the device's step). The
  mask stays the fallback behind the check of the GPU's candidates.

**Drafts and thinking under a grammar** *(read)*. Drafts are cut at the first
token the grammar refuses, on the CPU, before they are verified; rows inside
a reasoning section are not constrained; a finished grammar gives a full
mask.

- Where: `vllm/v1/structured_output/__init__.py` (`trim_spec_tokens` 363,
  457-508, `_get_constraint_bounds` 284).
- Numbers: none.
- VRAM: one mask row per draft position.
- Ares: **take** both with llguidance: cutting a draft saves rows, and the
  gate for thinking is needed for every reasoning model.

### Sampler

**Sorts, with sort-free paths beside them** *(read)*. The default pick is a
Gumbel-max kernel. Top-p in plain PyTorch is a full sort; a Triton kernel
searches a pivot instead (below, "Qrita"), and FlashInfer's sampler is used
when no row is greedy or seeded. Penalties read a table of counts of
`max requests x vocabulary` int32, of which a comment says it "can be very
large, taking up GBs".

- Where: `vllm/v1/worker/gpu/sample/sampler.py` (278-360), `gumbel.py`,
  `penalties.py`; `vllm/v1/sample/ops/topk_topp_sampler.py` (527-590, 634),
  `topk_topp_triton.py`.
- Numbers: none in tree.
- VRAM: an f32 copy of rows times vocabulary whenever a row needs
  processing; the count table.
- Ares: **leave** the tables; penalties come from the sequence's own token
  ids. The kernel to take is TensorRT-LLM's, below.

### Several GPUs

**A process per rank** *(read)*. Every tensor-parallel or pipeline rank is
its own worker process with its own driver context, PyTorch runtime, CUDA
graphs and K/V pool. Sums go through the first implementation that applies:
NCCL with symmetric memory, FlashInfer's or vLLM's own all-reduce over shared
buffers (it needs peer access, and full connectivity above two GPUs), else
NCCL. Pipeline stages hand their activations to the next rank one after the
other.

- Where: `vllm/v1/executor/multiproc_executor.py` `MultiprocExecutor` (125);
  `vllm/distributed/device_communicators/cuda_communicator.py` `all_reduce`
  (384); `custom_all_reduce.py` (`_can_p2p` 86, the limits at 138 and 267);
  `vllm/v1/worker/gpu_model_runner.py:4205` (`send_tensor_dict`).
- Numbers: none stated.
- VRAM: everything outside the weights is held once per rank.
- Ares: **have** (`Mesh`, `Lane`, one process). Nothing here addresses two
  unequal cards.

### Where vLLM's VRAM goes

Each cause, where it is decided, and what it costs. Where the source gives no
figure the cost is a formula.

| Cause | Where it is decided | Cost | In Ares |
| --- | --- | --- | --- |
| **The K/V pool takes everything that is left.** After the weights, the measured peak of a dummy pass and the estimate for CUDA graphs, all of `gpu_memory_utilization` (default 0.92) times the card goes to K/V at start, whether a request needs it or not, and never shrinks | `vllm/v1/worker/gpu_worker.py` `determine_available_memory` (655-830); `vllm/config/cache.py:215`; `gpu_model_runner.py` `initialize_kv_cache_tensors` (6904) | The rest of the card. Nothing else (a second model, experts, a larger batch) can use it | One pool too, but sized by the ledger from the configured context, shared with checkpoints and, later, movable (see TensorRT-LLM's address ranges) |
| **K/V in f16.** The default type is the model's | `vllm/config/cache.py:235` (`cache_dtype: "auto"`) | 16 bits per value. For a `qwen4exp`-shaped model and 262 144 tokens: 6.7 GiB against 3.56 GiB at `q8` and 1.94 GiB at `q4` *(calc from ARCHITECTURE.md's 3.56 GiB)* | `q8` by default, `q4` with exact cells |
| **Weights at the checkpoint's width.** 16 bits unless the checkpoint is quantized; embedding and output head usually stay f16 | Model loading; `docs/features/speculative_decoding/mtp.md` gives a head of 248k x 2560 as "0.6 or 0.35 GiB" quantized | 1.27 GB per f16 table of that size *(calc)* | Everything at the GGUF's width |
| **A CUDA graph per batch size.** Sizes 1, 2, 4, then every 8 up to 256, then every 16 up to 512 by default, bounded by the request limit | `vllm/config/compilation.py:695-707`; `profile_cudagraph_memory` in `gpu_worker.py:727` | Measured at start and taken from K/V; no figure in the docs. SGLang reserves `2 MB x max batch` for decode graphs *(stated)*; the llama.cpp fork was measured at 26 MiB per shape and GPU | None (principle 1) |
| **The peak of a dummy pass, kept for good.** A pass at the largest step (2048 tokens) is run and its peak, with what the allocator could not give back, is reserved | `gpu_worker.py:697-760`; `vllm/utils/mem_utils.py` `memory_profiling` (230). A comment at 704 describes a freed multi-GiB block that stays pinned by a small allocation | Whatever PyTorch peaked at | One scratch area per step, in the ledger |
| **Rows of the vocabulary.** f32 copies of the logits for processing; a count table for penalties; one row set per draft position | `sampler.py`, `penalties.py`, `rejection_sampler.py` (chunks of 1 GiB) | About 1 MB per row at 248k; the count table is 254 MB at 256 requests *(calc)*, "GBs" by its own comment | Picks per row on the device, penalties from token ids |
| **A recurrent state per draft position** | `MambaSpec.num_speculative_blocks` | (1 + K) states per request | None (commit replays) |
| **A process per rank** | `MultiprocExecutor` | A driver context, a PyTorch runtime, graphs, communication buffers and a K/V pool per GPU; measured as "non-torch" memory at start, no figure stated | One process, one context per device |

What is fast in vLLM does not depend on any of these: the scheduling loop,
the page table in the attention kernel and Marlin's weight kernels can be had
without the pool that fills the card, the graphs and the per-rank processes.

## SGLang

Paths are relative to `python/sglang/srt/` unless they start with `docs/`,
`rust/` or `kernels/` (which is `python/sglang/kernels/`).

### Prefix cache

**The radix tree as code** *(read)*. A node holds a slice of tokens, one K/V
slot index per token, its children keyed by their first page of tokens, a
lock count and the fields the eviction policies read. Matching compares
slices in doubling steps and then bisects. A match that ends inside a node
splits the node, so a lookup changes the tree. With pages larger than one
token the keys are cut to whole pages and the rest stays with the request, so
hits are token-exact only at a page size of 1.

- Where: `mem_cache/radix_cache.py` (60-248 `TreeNode` and `RadixKey`,
  251-279, 658-707 `_split_node`, 712-767).
- Numbers: none measured. The cost shows in a guard: cache-aware ordering is
  switched off above 128 queued requests ("Turn off the expensive prefix
  matching and sorting", `managers/schedule_policy.py:130, 310-321`).
- VRAM: none (host: 8 bytes per cached token and a Python object per node).
- Ares: **have** (`ares-state`): a chained page hash gives the same
  page-aligned hits without splits or per-token indices.

**Pages enter the cache while the request runs** *(read)*. One function,
`checkpoint(req, up_to)`, runs after every chunk of a prompt and at its end:
it inserts the pages, frees the duplicates the insert exposed, and rewrites
the request's row to the tree's slots. Output tokens enter at the finish.

- Where: `mem_cache/radix_cache.py` (455-550); `mem_cache/common.py`
  (233-242, 379-426);
  `managers/scheduler_components/batch_result_processor.py` (512, 623).
- Numbers: none.
- VRAM: none.
- Ares: **take** into `ares_state::PrefixIndex`, called from commit: a page
  is published when it fills, not when its sequence ends.

**Locks and eviction** *(read)*. A lock walks from a node to the root; a node
can go only when it is unlocked and has no resident child. Every call to
`evict` builds a new heap over all such leaves. The policy for multi-turn
tails (T-LRU) computes `budget = max(history + next_prompt_estimate -
threshold, 0)` and sorts first what lies beyond that budget.

- Where: `mem_cache/radix_cache.py` (559-623, 797-810);
  `mem_cache/evict_policy.py` (49-100).
- Numbers: none.
- VRAM: none; a heap over all leaves per eviction call.
- Ares: **have** the leaf-first order. **Leave** T-LRU: with `ares-store` a
  miss in VRAM is a read from disk, not a recomputation.

**One tree for K/V, sliding windows and recurrent state** *(read: READMEs and
the node type; the 6.4k-line core not)*. Each node carries data per component
and tier. A match advances only where every component is valid. On interior
nodes recurrent state is evicted before window state before K/V; a leaf goes
whole. The walks, locks and eviction order are a Rust core by default; Python
keeps the I/O.

- Where: `mem_cache/unified_cache/components/README.md`;
  `rust/sglang-radix-tree/` (`node.rs:137-211`);
  `mem_cache/unified_cache/unified_tree_core.py`.
- Numbers: none; no reason for the Rust core is stated beyond releasing the
  interpreter lock.
- VRAM: none.
- Ares: **have** in outline (checkpoints attach to index entries). **Take**
  the order: on entries inside a chain, checkpoints of recurrent state go
  before K/V pages.

### Admission

**The queue** *(read)*. The default order is first come, first served. The
cache-aware orders (longest matched prefix; waited tokens over uncached
tokens; shortest uncached prompt) are options. Under them a request with at
most 32 cached tokens that shares at least 32 with an earlier queued one is
sorted last, by matching the queue against a throwaway tree in every pass.

- Where: `managers/schedule_policy.py` (100-112, 206-277, 352-402, 405-497);
  `arg_groups/fields/schedule.py:102`.
- Numbers: none.
- VRAM: none.
- Ares: **take** into `ares_sched::plan`: prompts ordered by uncached rows,
  and a prompt held back while another admitted prompt is computing the page
  it would miss first. With hashes that is one lookup per waiting prompt.

**The admission rule** *(read)*. A ledger in tokens that reserves the output
that running requests are expected to still produce:

```
avail     = free slots + evictable slots        (the candidate's own prefix locked first)
reserved  = sum over running of min(max_new - produced, 4096) * ratio
admit if    extend + min(max_new, 4096) + page < avail - reserved
ratio     = 0.7, falling to 0.14 over 600 decode steps;
            after a retraction: min(1, (sum produced + 20 n) / (sum max_new + 1))
```

A prompt longer than the step's budget is cut to whole pages and its output
is charged when its last chunk is admitted; one prompt is chunked at a time.

- Where: `managers/schedule_policy.py` `PrefillAdder` (807-816, 1559-1877);
  `mem_cache/prefill_budget.py` (134-266);
  `managers/scheduler_components/new_token_ratio_tracker.py`.
- Numbers: defaults only.
- VRAM: none.
- Ares: **take** into `ares-sched`, counted in pages, together with vLLM's
  claim per step. It replaces the reservation of prompt plus `max_tokens` at
  submission. The ratio corrects itself, which a fixed worst case cannot.

**Retraction** *(read)*. Every decode step checks the pool. When it is short,
the request with the fewest generated tokens goes first (ties: the longest
prompt); its own K/V is freed, its cached prefix stays, and it is queued again
to be recomputed.

- Where: `managers/schedule_batch.py` (1796-1811, 2926-3083);
  `managers/scheduler.py` (4996-5086).
- Numbers: none.
- VRAM: none.
- Ares: **leave** the recompute; **take** the order (fewest generated tokens
  first) for choosing whose pages spill to RAM.

Against vLLM, SGLang is the weaker reference for the step itself: a prompt
batch is a separate forward pass that takes the step (mixing is off by
default, `managers/scheduler.py:4292-4305`), where vLLM and Ares put prompt
chunks next to decode rows. Its admission rule is the better one.

### The overlapped loop

**What is launched before what** *(read)*. Per iteration: take requests in;
plan batch N from host state that is one result old; enqueue the forward pass
and the sample of N; process the result of N-1. The input ids of N are not on
the host: the sample kernel of N-1 wrote its picks into a device buffer
indexed by request slot, and the embedding of N gathers from it. The picks
are copied to the host on a separate stream. One wait on the device remains
per step: for the copy of step N-1.

- Where: `managers/scheduler.py` `event_loop_overlap` (2160-2251), `run_batch`
  (5191-5321); `managers/overlap_utils.py` `FutureMap` (304-372, 744-819).
- Numbers *(stated, the v0.4 release note, read through a summary)*: "a 1.1x
  speedup against its previous version", "most significant on small models
  and large tensor parallelism sizes".
- VRAM: two int64 per request slot.
- Ares: **take the mechanism, not the motive**. Ares' conductor costs
  microseconds per step, so the 1.1x will not transfer. What transfers is the
  table on the device: the pick is written by the sample kernel and read by
  the next step's first kernel, so a step can be enqueued before the host has
  seen the previous picks, and with a layer split the first device can start
  on other sequences while the second finishes. It goes into `ares-runtime`
  and the step record; 8 bytes per slot and device.

**The rules that make a stale plan safe** *(read)*.

- A finish by length is known ahead and is filtered before the launch. An end
  token or stop string is seen one step late: the row runs once more and its
  result is dropped. Stop strings force no wait.
- Frees and rewrites of rows are ordered after the reads of the step that is
  in flight (an event the scheduling stream waits for).
- Where: `managers/scheduler.py` (2107-2118, 4984-4994);
  `batch_result_processor.py` (315-320).
- Ares: **take** into `ares_state::PagePool`: pages freed by the commit of
  step N can be reused only when the step enqueued before that commit has
  ended. LMDeploy states the same rule ("delayed cleanup").

**Grammar and speculation on the loop** *(read)*. With a grammar the forward
pass is enqueued and the sample held back until the previous result has
advanced the matcher; then the mask is filled and the sample enqueued. With a
drafter the accepted lengths are computed on the device, but the scheduler
still waits for them before it plans: only the drafter's catch-up and the
result processing overlap.

- Where: `managers/tp_worker.py` (734-756); `managers/scheduler.py`
  (2232-2261, 5252-5267); `managers/overlap_utils.py` (682-742).
- Numbers: none.
- VRAM: none.
- Ares: **take** the split of window and sample into two enqueues. The
  accepted length is a real wait per step even in SGLang; the plan must
  count on it.

### K/V across GPU, RAM and disk

Read by a delegated reader; three places were checked.

**Copies without threads, loads layer by layer** *(read)*. Copies between the
tiers are queued by the scheduler thread onto their own streams and their end
is an event that is polled once per step. A restore is issued per layer with
one event each, and the forward stream waits for layer L's event at layer L's
attention: the restore of later layers overlaps the computation of earlier
ones.

- Where: `managers/cache_controller.py` (72-116, 195-205, 1021-1056,
  1164-1211); `mem_cache/l2_transfer.py` (83-101, 237-261);
  `mem_cache/memory_pool.py` (2928-2955).
- Numbers: none. A warning: a large direct copy "can queue ahead of the
  forward's small metadata host-to-device copies and stall the forward"
  (`l2_transfer.py:31-38`).
- VRAM: none.
- Ares: **take** both for the restore path (`ares-runtime`, events of
  `ares-cuda`): the time to the first token on a hit becomes the larger of
  restore and compute instead of their sum. Keep the step record off any
  stream that carries bulk copies. Vulkan has one queue per device and gets
  no overlap.

**A hit on a lower tier does not block the queue** *(read)*. A request whose
pages are being fetched is skipped by the scheduler and others are admitted;
the fetch starts when the request is queued, for hits of 256 tokens or more.
A restore is all or nothing, is skipped under 10 tokens, and falls back to
recomputation when the pool is short after one eviction pass.

- Where: `managers/cache_controller.py` (1297-1531); `managers/scheduler.py`
  (4616-4627); `mem_cache/unified_radix_cache.py` (2333-2423).
- Numbers: none in the repository.
- VRAM: none.
- Ares: **take** skip-not-block into `plan`; derive the threshold from the
  step-time model instead of a constant.

**Write policies, the disk format, kernel I/O** *(read)*. A page is copied to
RAM at its first insert by default (or at its second hit, or at eviction).
On disk the key is a chained SHA-256 over the previous digest and the page's
tokens; the file backend writes one raw file per page without a header (a
TODO says so). A kernel that reads registered host memory and gathers by
index exists because SGLang's K/V is addressed by token and scattered.

- Where: `mem_cache/unified_radix_cache.py` (622-640);
  `mem_cache/hicache_storage.py` (809-1119);
  `mem_cache/cpp_utils/hash_binding.cpp` (62-81);
  `kernels/aot/csrc/kvcacheio/transfer.cu` (20-65, 342-432).
- Numbers: the kernel is "up to 3x" faster than plain copies *(stated,
  `docs/docs/advanced_features/hicache_design.mdx:153`; no data in the
  repository)*. RAM: the size of K and V in f16 per token; the default RAM
  pool is twice the device pool.
- VRAM: staging for 64 pages.
- Ares: **have**, better (`ares-store`: records with headers, written once
  when full). **Leave** the kernel for the prompt cache, whose pages are
  contiguous; it stays the tool for the sparse K/V working set.

### Speculation

Read by a delegated reader.

**The seed of the next round stays on the device** *(read)*. A round is
draft, verify, publish the accepted lengths, catch the drafter up. The bonus
token, the candidates and the hidden state of the next round stay in buffers
per request slot.

- Where: `speculative/eagle_worker_v2.py` (669-757, 1121-1313, 1469-1603);
  `speculative/eagle_worker_common.py` (756-1108).
- Numbers *(stated,
  `docs/docs/advanced_features/speculative_decoding.mdx:141-150`)*:
  Llama-3.1-8B on one H100, 158 tokens/s plain, 244 with EAGLE-2, 373 with
  EAGLE-3.
- VRAM: K/V slots for twice the draft window per request.
- Ares: **take** for drafters that are models: their tokens are referenced
  from a device buffer and are not written into the host's step record.

**Recurrent state across a verify** *(read)*. The default stores a snapshot
of the state for every verified row and scatters the right one into place
after acceptance. Replaying the accepted inputs at commit is an option, for
chains only. When the accepted span crosses the checkpoint grid, the state
there is kept for the prefix cache.

- Where: `speculative/spec_utils.py` (1101-1432); `mem_cache/memory_pool.py`
  (888-1002).
- Numbers: the snapshots are "the dominant spec scratch, ~46x the conv
  state", "~9GB" in one configuration (`memory_pool.py:894-900`) *(stated)*.
- VRAM: as stated.
- Ares: **have** (principle 5). **Take** one rule into `ares-state`: the
  replay at commit emits the checkpoint when it crosses the grid.

**Other pieces.**

| Piece | What it does *(read)* | Where | Ares |
| --- | --- | --- | --- |
| Draft trees | Candidates expanded per level; a kernel emits mask, positions and links; the accepted path's K/V is copied to the front, one indexed copy per layer | `speculative/eagle_utils.py` (162-296); `kernels/jit/csrc/speculative/tree.cuh` | **Leave**: SGLang's own defaults for recent models are chains, and the copy fights pages that hold whole tiles |
| A drafter without K/V | The MTP head reads the target's pool at a fixed position and recurs on token and hidden state | `speculative/frozen_kv_mtp_worker_v2.py` (541-693) | Comes with the model; costs no draft K/V |
| Unequal draft lengths in one pass | For one drafter only, with a graph per bucket of total rows | `speculative/ragged_verify.py` | **Have** (`plan`) |
| Adaptive length | Per bucket of batch size, a moving average of the accepted length picks among prepared lengths | `speculative/adaptive_spec_params.py` (25-264) | **Have**, per sequence |
| N-gram | A trie with an anchor per request that advances one token at a time; breadth-first by recency or best-first by frequency; 10 M nodes preallocated; the lookup blocks the scheduler | `kernels/jit/csrc/ngram_corpus/trie.cpp` (23-331); `speculative/ngram_worker.py` (245-315) | **Take** the incremental anchors and one corpus for all sequences (`ares-sched`); see the suffix automaton under TensorRT-LLM |

### Structured output

**Jump-forward has no caller** *(read)*. `try_jump_forward`,
`jump_forward_str_state` and `jump_and_retokenize` are defined in five files
under `constrained/` and are called from nowhere else in the repository. The
first survey found the same one commit earlier.

**What runs** *(read)*. The mask is filled on the scheduler thread at sample
time (llguidance fills the rows of a batch in parallel, xgrammar one by one),
uploaded without blocking and applied by a kernel. Compiled grammars are
cached by kind and text; a miss compiles on a thread pool while the request
waits in its own queue and the others go on. A wrapper holds the grammar off
until the end of the thinking section. Tool calls compile to xgrammar's
structural tags.

- Where: `constrained/base_grammar_backend.py` (84-93, 201-316),
  `grammar_manager.py` (154-340), `reasoner_grammar_backend.py` (43-314),
  `llguidance_backend.py` (163-175).
- Numbers: "up to 10x faster than other open-source solutions for JSON
  decoding tasks" *(stated, v0.4 note)*.
- VRAM: a bit per row and token of the vocabulary.
- Ares: **take** the queue for requests whose grammar is compiling, the gate
  for thinking, and tool calls as grammars that start at a trigger. For
  forced tokens see MLC's jump-forward below.

### Several GPUs, sampler, memory

Read by a delegated reader.

- **Tensor parallelism is a process per rank**, each with a full scheduler
  that plans identically; every rank has its own context, graphs and K/V
  pool, and the pool is sized by the smallest card.
  (`entrypoints/engine.py:959-999`, `distributed/parallel_state.py`.) A sum
  over PCIe by shared memory is given as 5.3 microseconds for 12 KiB against
  205 for NCCL *(stated, `distributed/device_communicators/pcie_ipc_ar.py`)*,
  the class of Ares' 8 microseconds for 10 KiB. Ares: **have**.
- **Pipeline stages keep several micro-batches in flight** and choose the
  chunk of a prompt from a quadratic fit of latency against length,
  `f(l) = a l^2 + b l + c`, solved for a target time
  (`managers/scheduler_pp_mixin.py`,
  `managers/scheduler_components/dynamic_chunk_sizer.py:37-110`). A comment
  says the pipeline does not work with the overlapped loop. Ares: **take**
  the fit (the context term of the step price, `ContextFit`) and the rule of as many groups
  of sequences in flight as devices in a layer chain.
- **The sampler** falls back to `torch.topk` or a full sort unless FlashInfer
  is used; penalties are dense rows of the vocabulary
  (`layers/sampler.py:212-227, 402-505`). Ares: **leave**.
- **Memory**: about `512 MB + 1.5 MB x max(chunk, 2048) + graphs` is held
  back and the K/V pool takes the rest after the weights
  (`arg_groups/memory_hook.py:49-71, 292-400`); decode graphs are captured per
  bucket of batch size. Ares: **leave**; it is vLLM's pattern.

## FlashInfer

Paths are relative to the repository; `inc/` is `include/flashinfer/`. Read
by a delegated reader. Ares has no attention kernel and no sampler yet, so
this section is the one to read before writing them.

**Pages as a table per request** *(read)*. K and V are pools addressed by
page, token and head. A batch of unequal requests is three arrays: an offset
per request into a list of page ids, and the fill of each last page. Page
size is free. An index past the end maps to offset 0, so loads issued ahead
need no branch.

- Where: `inc/page.cuh` (37-270).
- Numbers: "page size has little effect" *(stated)*.
- VRAM: 4 bytes per page and request in the step.
- Ares: **have** (`ares-state`, step record). **Take** the harmless offset
  for loads ahead, and an order inside a page in which the codes of one head
  and tile are contiguous.

**Writing K and V** *(read)*. One kernel appends rows addressed by batch
index and position. Variants quantize while appending (a scale per 16 values,
a second candidate kept when its error is lower) and fuse the query/key
normalisation and the rotary embedding into the same launch.

- Where: `inc/page.cuh` (382-727); `inc/pos_enc.cuh` (473, 897).
- Numbers: none.
- VRAM: none.
- Ares: **take** the shape: one writer kernel per step does normalisation,
  rotary embedding, Hadamard rotation, scale and packing, addressed from the
  step record. For the formats with a tile mean it is two kernels: new cells
  go to `f16`, and a tile of 128 is converted when it leaves the exact
  window.

**Decode attention on the plain cores** *(read)*. One block per request (or
piece of one) and KV head. Each thread loads 16 bytes of a K row into shared
memory through two `cp.async` stages, the dot product is reduced across lanes
with shuffles, and the query heads that share a KV head are rows of threads
over the same tile. No `mma`. The running result is a state: output, maximum
and sum, rescaled as the maximum grows.

- Where: `inc/attention/decode.cuh` (62-193, 406-624, 759-845);
  `inc/attention/state.cuh`.
- Numbers *(stated, its introduction)*: memory bandwidth "close to 100% for
  long sequences" for single heads; "40%+" with grouped heads on the plain
  cores.
- VRAM: none.
- Ares: **take as the Vulkan attention kernel** for every row count (nothing
  in it needs tensor cores; shuffles become subgroup operations and
  `cp.async` plain loads) and as the CUDA path for a group of one head. With
  Ares' formats: an integer dot product of codes with the query that was
  rotated once, reduced inside each group of 32, times the group's scale,
  then across groups; the output accumulated rotated; one extra product with
  the mean per tile.

**Decode on the tensor cores for grouped heads** *(read)*. Decode goes
through the prompt kernel with the query rows packed as row times group: the
heads that share a KV head fill the 16 rows of the tile.

- Where: `inc/attention/scheduler.cuh` (557-625); `inc/utils.cuh` (411).
- Numbers *(stated)*: "2-3x" for grouped heads on A100 and H100.
- VRAM: none.
- Ares: **take** as the CUDA decode path for layers with grouped heads. For a
  layer with 12 query heads per KV head and 256 dimensions, one row at `q8`
  is about 23 operations per byte, 11 TFLOPS at 479 GB/s against the capped
  58.6: the cap does not bind for one row. It binds from about 5 rows at
  `q8` and 3 at `q4` on the RTX 4070 (3 and 2 on the RTX 3060) *(calc)*;
  above that the products need f16 accumulation per tile.

**The prompt kernel** *(read)*. The FlashAttention-2 loop: the query tile is
loaded once; K and V come through `cp.async` into shared memory laid out to
avoid bank conflicts, then `ldmatrix` and `mma.m16n8k16` with f32
accumulation. The causal mask is applied only on the tiles at the diagonal.
A batch of unequal requests is a list of work items, one block each, built on
the host. A path that accumulates the product of query and key in f16 exists
and is switched off at compile time.

- Where: `inc/attention/prefill.cuh` (336-516, 690-852, 1630-1769,
  4279-4950); `inc/mma.cuh`.
- Numbers *(stated)*: f16 accumulation of the query-key product "could bring
  50% speedup on RTX 4090"; paging costs about 1 % in decode and 10 % in
  prompts against contiguous K/V.
- VRAM: none; up to 101 KB of shared memory per block.
- Ares: **take** the loop for verify windows and prompts on CUDA. On capped
  cards both products accumulate in f16 per tile and are added into f32
  registers. A K/V tile that divides 128 lets the row sum the softmax needs
  serve as the summed weight for the tile-mean term of V.

**Quantized K/V inside the kernel, three ways** *(read)*. (a) Convert each
fragment after `ldmatrix`, with a multiply by the scale before the `mma`.
(b) Unpack the packed tile once in shared memory into a 16-bit staging tile
and run the ordinary loop; the comments call this cheaper. (c) Factor a scale
per token and head out of the product and apply it afterwards. Scales are
pools of their own with their own loads. There is no int8 `mma` for
attention.

- Where: `inc/attention/prefill.cuh` (142-202, 888-1160, 1429-1539,
  1659-1766).
- Numbers *(stated, the paper)*: in vLLM with 8-bit K/V the time per token
  fell from 12.56 to 10.92 ms.
- VRAM: none; the staging tile is shared memory.
- Ares: **take (b)**: unpack B-bit codes times the f16 scale per 32 into an
  f16 staging tile; rotate the query tile once when it is loaded; un-rotate
  the output once at the end, or in the merge when the work was split. The
  format must be a switch per tile and not a template parameter as here,
  because one sequence mixes `f16` pages (the exact cells) with `qB` pages.
  ExLlamaV3 measured the same choice (below).

**The work list and the merge** *(read)*. Before every launch the host
searches the smallest piece of K/V (at least 128 tokens) for which the number
of blocks fits the processors, and writes a list of (request, query tile, K/V
piece) into a page-locked buffer that is uploaded with one copy. Each block
writes a partial state; a second kernel merges the states of a row by their
log-sums. There is no cache of plans. A newer planner assigns work items of
prompts and of decode rows to the least loaded block and runs them all in
one persistent launch.

- Where: `inc/attention/scheduler.cuh` (73-130, 348-508, 544-690, 770-936,
  981-1552); `inc/attention/cascade.cuh` (376-478, 696-747);
  `inc/attention/persistent.cuh`.
- Numbers *(stated)*: with inputs of 4096 to 16384 tokens, 8.63 ms per token
  with the balancing and 13.89 without. For a single sequence on an RTX 4090
  the split "does not improve performance". The paper admits "heavy Python
  overhead" in vLLM and plans to "move the scheduler to device".
- VRAM: a partial state per block, about 1 MB for 92 blocks.
- Ares: **take** the list and the merge, built by `ares-sched` into the step
  record (no padding and no mask of unused blocks, because there are no
  graphs). Rotated partial outputs are merged rotated and un-rotated there.
  On two small cards the split is what keeps all processors busy for one long
  sequence; their own RTX 4090 note says to measure it first.

**A shared prefix computed once** *(read)*. The attention over a union of
sets of cells is the merge of the states over each set. A wrapper runs a
prefix that all rows share as one pass with many query rows, each row's own
part separately, and merges.

- Where: `inc/attention/state.cuh`; `flashinfer/cascade.py` (226-575).
- Numbers *(stated)*: "up to 31x" over an old vLLM with a prefix of 32768 and
  a large batch. From the paper's table, suffix 128: at batch 16 and prefix
  8192, 88.7 against 226.6 microseconds; at prefix 1024 and batch 16, 45.2
  against 46.5 (no gain). End to end with four continuations per prompt: 14
  to 17 % less time per token.
- VRAM: a state per row and level.
- Ares: **have** the algebra (`ares_quant::kvhead`). **Take** later, as a
  rule of the planner and not as a wrapper: rows whose page tables share a
  run of pages (forks, the options of a decision, the drafts of one
  sequence) get one work item for the shared run. The gain exists only for
  long shared runs and many rows.

**Block-sparse attention** *(read)*. The older wrapper maps a block-sparse
matrix onto the paged kernel (a block is a page, the column indices are the
page list) but plans on the host, which is useless when an indexer on the
device selects per layer. The newer entry for the Qwen models takes device
arrays: a list of selected blocks per row with its count, the block table and
the row-to-request map, without a host plan. Its reference kernel uses only
Ampere instructions: one block of four warps per row, KV head and piece;
tiles of 64 cells; each lane owns a cell and reads the selection three tiles
and the block table two tiles ahead; K is decoded straight into `mma`
fragments; the query is staged in fragment order; the grouped heads are the
rows of the `mma`. A second kernel merges the pieces.

- Where: `flashinfer/sparse.py` (363-560, 1173-1200, 2297-2427);
  `flashinfer/experimental/qsa_nvfp4/csrc/qsa_nvfp4_attention.cuh` (31-860).
  Also there: a fused indexer score and exact top-k by a histogram cut with a
  tie list (`flashinfer/experimental/sm120_dsa_indexer/`), and a radix top-k
  (`inc/topk.cuh`).
- Numbers *(stated)*: block-sparse against dense at a budget of 32768: 22 to
  68 microseconds against 1712. For the Qwen reference kernel: none.
- VRAM: `rows x (width + 1) x 4` bytes of id list.
- Ares: **take the reference kernel as the template for the Qwen4 attention
  kernel**; it already is "id list and page table". Changes: Ares' codes and
  scales in place of the 4-bit float, a rotated query. The tile mean is the
  awkward part: a gathered tile of 64 selected cells can touch up to 16
  storage tiles, so the means of the touched tiles are gathered as extra
  rows, one product with the query and one summed weight each. **Take** the
  histogram cut for the indexer's selection.

**Sampling without a sort** *(read)*. One block of 1024 threads per row.
Top-k, top-p and both together are rejection rounds over the probabilities:
draw, take the drawn token's probability as a pivot (and a second pivot
halfway to the upper bound), sum mass and count above both in one pass,
accept or narrow. Min-p is a maximum pass, a mass pass and one draw. A pick
straight from logits is one pass with Gumbel noise. The generator is
counter-based, but its stream is selected by the row's index in the batch.

- Where: `inc/sampling.cuh` (585-657, 682-792, 861-1356, 1359-1740,
  2320-2505).
- Numbers *(stated)*: "reduce the overall sampling time by more than 50%" in
  vLLM.
- VRAM: a row of probabilities per sampled row.
- Ares: **take** the Gumbel pick as the default, with noise keyed by seed,
  position and token and never by the row's place in the batch; then
  verifying a draft stays a comparison of picks. For the filters the
  threshold kernel of TensorRT-LLM is the better fit (below). **Leave** the
  speculative sampling kernel: it needs full probability rows of draft and
  target and buys nothing for drafters that propose one token.

**Also in the repository.**

- Kernels for the gated delta rule with replay: the verify leaves the state
  frozen and keeps the raw inputs; a commit kernel replays the accepted
  prefix (`flashinfer/gdn_decode.py:1873`, `flashinfer/gdn_kernels/`) *(read:
  docstrings)*. Ares: **have** the design; these are the kernels to read when
  writing the recurrent ones.
- Whole-block kernels for the Qwen models and the attention, matrix and MoE
  kernels for Blackwell are binaries without source
  (`flashinfer/artifacts.py`). Ares: **leave**.
- The compile-time variants (`inc/attention/variants.cuh`) and the JIT that
  builds one module per head size and type. Ares: **leave** the JIT; model
  constants are compile-time arguments in `build.rs`.
- Portable to GLSL: the page-table gather, the state and its merge, the work
  list, the plain-core decode kernel, the generator, the sampling passes and
  the histogram select. Bound to CUDA: the `cp.async` pipeline, `ldmatrix`,
  `mma`.

## TensorRT-LLM

Paths are relative to the repository. Read by a delegated reader. Its newest
attention, matrix and MoE kernels for Blackwell are binaries; what follows is
what has source.

**A sampler in one kernel without a row of scratch** *(read)*. One block per
row. After a pass for the maximum, `w = exp(l / T - max)` is `p / p_max`, so
min-p, top-k and top-p are all thresholds on `w`. A row that only needs a
token draws by rejection: sample from a superset, measure count and mass
above the candidate in one sweep, narrow on a rejection (up to 32 rounds).
Exact cut-offs come from a radix descent over the bits of the floats (four
passes of 8 bits). Parameters are tensors per row; only token ids and finish
reasons go to the host.

- Where: `cpp/tensorrt_llm/kernels/fusedSampling/fusedSamplingKernelsCommon.cuh`,
  `fusedSamplingKernels.h`;
  `tensorrt_llm/_torch/pyexecutor/sampler/sampler.py` (`_sample_fused`).
- Numbers: none on speed.
- VRAM: no rows-times-vocabulary scratch; about 48 KB of shared memory.
- Ares: **take** as the sampler of `ares-cuda` and its GLSL mirror: 2 to 3
  sweeps of a row of logits. It replaces the histogram select that was
  planned from ExLlamaV3 where exact cut-offs are wanted. The generator is
  keyed by seed and position.

**A pick across a head that is split by vocabulary** *(read)*. Each GPU
reduces its part of the logits to one (value, index) pair per row, with
Gumbel noise keyed per vocabulary entry, and the pairs are exchanged.

- Where: `tensorrt_llm/_torch/speculative/vocab_parallel.py`.
- Numbers: none.
- VRAM: two floats per row and GPU.
- Ares: **take** when the output head is tensor-split (two values per row
  through `Mesh`). Top-p needs an exchange of mass first.

**4-bit weights converted in registers** *(read)*. At load the nibbles are
reordered inside each word and the rows permuted in groups of 32 so that the
load instruction puts narrow data into the right lanes. In the loop, eight
nibbles become eight f16 with four `lop3`, a shift, two subtractions and two
fused multiply-adds; the group scales are a third operand loaded once per
group.

- Where: `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp`;
  `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h`;
  `.../gemm/threadblock/dq_mma_multistage_finegrained.h`.
- Numbers: none stated.
- VRAM: the weights at 4 bits plus scales.
- Ares: **take**; it is the same answer as Marlin's from a second code base.
  A kernel on the plain cores is instantiated per exact row count up to 16
  (`cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemm.cu`), where
  Ares stops at 8.

**Sums across GPUs** *(read)*. One-shot: each rank pushes its part into every
peer's buffer and then sums its own. The handshake needs no token: receive
buffers are filled with -0.0 beforehand, and each thread spins on its own 16
bytes until they change. The residual add and the norm that follow are fused
into the same kernel. A low-precision variant sends int8 or 8-bit float with
a scale per warp and is chosen only without a fast link, for messages of 2 MB
or more.

- Where: `cpp/tensorrt_llm/kernels/customAllReduceKernels.cu`
  (`oneShotAllReduceKernel`, `block_barrier`);
  `cpp/tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.cu`;
  `cpp/tensorrt_llm/thop/allreduceOp.cpp` (`selectImplementation`).
- Numbers: none stated.
- VRAM: a workspace of ranks times the largest message.
- Ares: **take** two things into `Mesh`: the sentinel per vector instead of a
  token (write-only peer access is enough for it, which is all the patched
  driver gives), and the residual and norm inside the sum kernel, one kernel
  fewer for each of the 96 to 130 sums of a token. **Take** the low-precision
  payload only for the wide sums that go to the copy engine at 3.3 GB/s.
  **Leave** the two-round form.

**The prefix index and its eviction** *(read)*. A trie of blocks whose keys
are SHA-256 digests chained from the previous block; a match that ends inside
a block is copied into a private page instead of shared. Eviction is by
priority class and then least recent use, with a time to live that resets the
priority; pages below a priority are dropped instead of moved to RAM.

- Where: `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h`,
  `page.h`, `evictionController.h`, `config.h`.
- Numbers: none.
- VRAM: none.
- Ares: **have** the index. **Take** the copy of a tail: a prompt that ends
  inside a page copies its cells instead of recomputing them, which is valid
  while they are still `f16` exact cells or `q8` rows. **Take** a priority
  class ahead of the leaf-first order, and a rule which pages never spill.

**Pools on reserved address ranges** *(read)*. Each pool is a reserved range
backed by chunks of 2 to 32 MiB mapped at its top; pools with different slot
sizes trade chunks toward a ratio that is recomputed every 100 requests.

- Where: `.../kv_cache_manager_v2/cudaVirtMem.h` (`VirtMem::extend`,
  `shrink`), `config.h` (`PoolRebalanceConfig`).
- Numbers: page-locked memory on the wrong NUMA node costs 1.6x to 2.9x
  *(stated in the source)*.
- VRAM: at most one chunk of slack per pool.
- Ares: **take** into `ares-cuda` and the ledger: the page pool, the
  checkpoints and the expert slots as ranges whose chunks the ledger moves.
  It replaces sizes fixed at load. Mapping must stay off the step.

**Page moves: the copy engine beats a kernel on PCIe** *(read)*. Moves
between host and device run either as a kernel that reads registered host
memory or as one batched copy; the kernel is enabled only on coherent links.

- Where: `.../kv_cache_manager_v2/batchedPageCopy.{h,cu}`.
- Numbers *(stated in the source)*: H100 on PCIe 4, to the host: 8.56 GiB/s
  with the kernel, 17.45 with the copy engine; from the host 23.9 both.
- VRAM: 8 bytes per page for the list.
- Ares: **leave** the kernel and **take** the finding: page residency issues
  one batched copy per step and uses no processors for it.

**The working set for sparse attention** *(stated: design document; kernels
not read line by line)*. The sparse buffers live in host pages; a request
leases an area of cells, 2 to 4 times the selection width. Per launch a fetch
kernel builds a hash in shared memory over the selected ordinals, streams the
resident set past it and evicts by an array of last uses; cells above the
write mark are pinned; layers that share an indexer share one resident set.

- Where: `.../kv_cache_manager_v2/SPARSE_ATTENTION_DESIGN.md`,
  `sparseCacheKernels.cu`, `config.h` (`SparseCacheConfig`).
- Numbers: none.
- VRAM: cells times unit bytes per request and selection group.
- Ares: **take** for the residency layer of `ares-state` for block-sparse
  models: the hash sized by the selection and the shared resident set are the
  mechanics SGLang's guide does not spell out. The cells are leased from the
  pool, not reserved per slot.

**Other pieces** *(read unless marked)*.

| Piece | What it does | Where | Ares |
| --- | --- | --- | --- |
| Suffix automaton | Per request, one relocatable buffer of nodes, tokens and a hash area; built on the CPU for the prompt, extended and queried on the GPU; overrides the draft head at matches of 4 or more ("up to 40%" higher acceptance, *stated*); about 240 bytes per token | `cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomaton.h`; `tensorrt_llm/_torch/speculative/sa_enhancer.py` | **Take** as a host drafter in `ares-sched` in place of the `NgramDrafter`'s hash of the last n tokens; **leave** it on the GPU |
| Decode attention ("XQA") | The query heads of a group are one side of the `mma`; long sequences are split over blocks and merged; 8-bit K/V become f16 in registers; one global scale | `cpp/kernels/xqa/mha.cu`, `defines.h` | The structure is the same as LMDeploy's, which is the closer source |
| Indexer top-k | Radix histograms over the bits of the scores (11 bits of the f16 pattern, then 11, 11 and 10 of f32), a sort inside the threshold bin | `cpp/tensorrt_llm/kernels/indexerTopK.cu` | **Take** for block selection (portable to GLSL) |
| Recurrent state under speculation | Every per-step state is stored during the verify and one is selected at commit | `cpp/tensorrt_llm/kernels/mamba2MTPSSMCache/` | **Leave** |
| Memory budget | A dummy request of the largest size is run and the K/V quota is what is left of the peak, times a fraction | `tensorrt_llm/_torch/pyexecutor/_util.py` (`_cal_max_memory`) | **Leave**; the ledger is stricter |
| Tokenization cache *(stated)* | A prompt that extends a cached one is tokenized from a few tokens before the cached end and spliced only if the ids at the seam agree | `docs/source/features/prefix-tokenization-cache.md` | **Take** into `ares-tokenizer` |

## MLC-LLM

Paths are relative to the repository; `tvm/` is the pinned TVM. Read by a
delegated reader.

**Jump-forward as an engine action** *(read)*. An action that runs no model:
the string the grammar forces is appended, the last 10 tokens plus the string
are tokenized again, the first differing index gives a rollback of rows that
are already in K/V, a last token that is a prefix of another is dropped, the
tokens are committed with probability 1, and the next decode step feeds
several tokens for that sequence.

- Where: `cpp/serve/engine_actions/batch_jumpforward.cc`; `batch_decode.cc`
  (100-150).
- Numbers: none stated.
- VRAM: none.
- Ares: **take** into `ares-sched` as a kind of row: forced rows, next to the
  newest token and the drafts, that are neither sampled nor verified;
  `Sequence::truncate` does the healing at the seam. One tokenization on the
  host per jump. It is the only engine here in which jump-forward runs.

**An explicit memory budget** *(read)*. Tokens of K/V = (VRAM x 0.85 -
parameters - twice the largest compiled temporary - auxiliary buffers -
workspaces) / bytes per token, with closed formulas for each term and three
modes (one sequence, four, all that fits).

- Where: `cpp/serve/config.cc` (`EstimateMemoryUsageOnMode`, 652-800);
  `python/mlc_llm/compiler_pass/estimate_memory_usage.py`.
- Numbers: none.
- VRAM: a workspace for logits of sequences x vocabulary x 16 bytes; a fixed
  128 MiB buffer of indices.
- Ares: **have** (the ledger).

**The sampler and penalties** *(read)*. The cut-off for top-p is found
without a sort by testing several pivots per sweep; top-k falls back to a
full sort; penalties and biases are sparse scatters over lists built on the
host.

- Where: `python/mlc_llm/op/top_p_pivot.py`;
  `python/mlc_llm/compiler_pass/attach_logit_processor.py`.
- Ares: **leave** the sampler; **take** the sparse lists for penalties (a
  list in the step record).

**What TVM's Vulkan path does, as a list of what to avoid** *(read)*. A
pipeline is created on first use and there is no pipeline cache. A global
memory barrier follows every dispatch. Every read back to the host is a copy
to staging, a submit and a wait on a fence. Every buffer is its own
allocation.

- Where: `tvm/src/backend/vulkan/runtime/vulkan_wrapped_func.cc`,
  `vulkan_stream.cc`, `vulkan_device_api.cc`, `vulkan_buffer.cc`.
- Numbers: none; MLC publishes no comparison of Vulkan with CUDA.
- Ares: **have**, and ahead of it (addresses in push constants, batched
  submits). It confirms two rules: sampling stays on the device on Vulkan,
  and a barrier belongs only where a kernel reads what the previous one
  wrote.

**Tensor cores and integer dot products from SPIR-V** *(read)*. The code
generator knows `VK_KHR_shader_integer_dot_product` and
`VK_NV_cooperative_matrix`, and lowers matrix intrinsics to
`OpCooperativeMatrix{Load,MulAdd,Store}NV`. MLC's own Vulkan target enables
neither.

- Where: `tvm/src/backend/vulkan/codegen/spirv_support.h`, `codegen_spirv.cc`
  (538-621).
- Ares: the probe for cooperative matrices is already in the working tree
  (`crates/vulkan/examples/vk_coop_probe.rs`). The integer dot product is the
  Vulkan form of `dp4a` and belongs in the same probe.

**One function with three bodies** *(read)*. A matrix product is compiled
with schedules for up to 2 rows, up to 8 rows and above, behind a branch that
is the same for all threads (`python/mlc_llm/compiler_pass/low_batch_specialization.py`).
Ares: **have** (`decode_kernel_rows`).

## LMDeploy (TurboMind)

Paths are relative to the repository. Read by a delegated reader. The block
manager and batch loop of older versions no longer exist; the engine is in
`src/turbomind/engine/` and `memory/`.

**Decode attention with the dequantization inside** *(read)*. This is the
closest existing code to the attention kernel Ares has to write. K and V of a
token and head are 8-bit or 4-bit codes with an f16 scale and zero from the
vector's own range. A block of 64 tokens is walked backward through the block
pointers, and a tile never crosses a block. K is the 16-row operand of the
`mma` and the query heads of a group (rounded to 8 or 16) the other side. One
`ldmatrix` of packed codes yields the operands of several `mma`; the codes
become f16 in registers (a byte permutation for 8 bits, `lop3` for 4), then
one fused multiply-add per value applies scale and zero. Scales are a second
operand with their own asynchronous loads; K and V tiles alternate through
one pipeline of five stages.

- Where: `src/turbomind/kernels/attention/quantization.h` (`ConvertKvCache`),
  `block.h`, `block_iterator.h`, `impl_81616.h` (`StateQK::Load`,
  `Transform`, `ComputeQK`), `mainloop_sm80.h`,
  `kernel/decoding_sm80_128.cu`.
- Numbers *(stated)*: "2x faster KV8 kernels" (README); 1.27x and 1.39x
  requests per second with 8 and 4 bits, from fitting more blocks
  (`docs/en/quantization/kv_quant.md`).
- VRAM: per token, head and K or V: `D x bits / 8 + 4` bytes.
- Ares: **take as the skeleton** of the decode attention kernel on CUDA: the
  group as one side of the `mma`, backward tiles within a page, the expansion
  of packed operands, scales as an operand. One change: Ares' scale is
  symmetric per 32 rotated values, so one integer `mma.m16n8k32` per group
  into a fresh accumulator, scaled afterwards with four multiply-adds per
  thread, replaces sixteen per-value multiply-adds before the `mma` and also
  avoids the cap on f32 accumulation.

**One source for Turing, Ampere and newer** *(read)*. The kernel is a
template over architecture, main loop and cache iterator. The Turing variant
reuses the Ampere `mma` code, emits `m16n8k16` as two `m16n8k8` on the same
fragments, and replaces `cp.async` by staging in registers: the next tile is
loaded before a compute phase and stored to shared memory after it.

- Where: `src/turbomind/kernels/attention/mainloop_sm70.h`,
  `kernel/decoding_sm75_128.cu`; `src/turbomind/kernels/core/mma.h`
  (126-144).
- Ares: **take**; this is the Turing path that ARCHITECTURE.md lists as open
  for the prefill kernels, and the attention kernels need it too.

**Prompts do not read quantized blocks** *(read)*. For a prompt the history
is dequantized into a contiguous f16 buffer and a plain f16 attention runs
over it (`src/turbomind/models/llama/unified_attention_layer.cc:468-632`).
Ares: **leave**; a transient f16 copy of the context breaks the ledger.

**A 4-bit matrix product whose layout comes from the kernel itself** *(read)*.
At load, a GPU kernel runs the product's own path from global memory through
shared memory to registers once and stores what each lane holds; the stored
layout is therefore the fragment order by construction, and the loop needs no
`ldmatrix`. There are about 30 kernels per family (tile shapes, stages, cache
policy, split). Per matrix shape a cost model ranks them, the best are timed,
and the winner per shape and batch bucket goes into a table that is saved to
disk and loaded again.

- Where: `src/turbomind/kernels/gemm/convert.cuh` (`ConvertOperand`,
  `convert_kernel`), `kernel/sm80_16816_4.cu`, `gemm.cu` (`Find`,
  `Measure`), `dispatch_cache.cu`, `tuner/`.
- Numbers *(stated)*: 206.4 tokens/s for Llama-2-7B at 4 bits on an RTX 4090
  (`docs/en/quantization/w4a16.md`); "2.4x faster than FP16" (README).
- VRAM: the weights at 4 bits plus scale and zero per group.
- Ares: **take** both ideas: derive each type's device layout from the
  kernel's own load path instead of writing a permutation by hand per type,
  and replace the fixed row thresholds (`decode_kernel_rows`) by a measured
  table per device, stored on disk. Cost: seconds of tuning per device.

**Memory as transactions** *(read)*. One device region is allocated once and
divided by a page allocator and slabs per kind of object. A scheduling pass
replays its allocations and evictions on a copy of the capacity state and
commits only what fits.

- Where: `src/turbomind/memory/page.h`, `slab.h`, `object.h`;
  `src/turbomind/engine/README.md`.
- Ares: **take** the dry run: a step's claims on pages and checkpoints are
  planned against a copy of the counters and committed whole. TokenSpeed does
  the same (below).

**Resuming a model with recurrent layers** *(stated in the engine's README;
configuration read)*. Checkpoints are published at least 4096 tokens apart,
at the prompt's length minus a small skip (the last token of a prompt is
volatile), and optionally at the end of generation. The point from which
recurrent state is rebuilt is separate from the number of blocks whose K/V is
taken from the cache: between them the rows are computed again to rebuild the
state while their K/V stays read-only and is not written again.

- Where: `src/turbomind/engine/README.md`;
  `src/turbomind/models/llama/GatedDeltaNetLayer.cc`;
  `lmdeploy/messages.py` (337-360).
- VRAM: one recurrent state per checkpoint.
- Ares: **take** into the checkpoint policy of `ares-state`: a minimum
  spacing, the checkpoint a little before the prompt's end, and a replay over
  cached pages without K/V writes. The last saves the scale search of the
  low-bit formats and keeps the pages shared.

**A low rung below 4 bits** *(read)*. In the PyTorch engine: a rotation of
the whole vector, K as a 3-bit trained level plus one sign bit of the
residual with two norms per token and head, V as a 2-bit level plus a norm
(`lmdeploy/pytorch/kernels/cuda/turbo_quant.py`). Stated: "~5x" less memory,
7.3 % less throughput; no accuracy table. Ares: an experiment for
`ares-quant` (trained levels after the rotation, a V rung below `q4`), to be
judged by the existing measurement.

## ExLlamaV3: what is new at code level

Paths are relative to `exllamav3/` in the repository; `ext/` is
`exllamav3/exllamav3_ext/`. Read by delegated readers. The format (trellis
codes) cannot be used; the kernel engineering can.

**Tensor cores for 1 to 8 rows, weights decoded in registers** *(read)*. A
lane loads two words of the weight bit stream, one funnel shift gives eight
codes, and the codebook is arithmetic (a multiply, a byte sum by `dp4a`, one
fused multiply-add). The rows of the batch are the rows of one `mma`, so
eight rows cost the instructions of one. The accumulators are f16 and are
folded into f32 every 2 to 4 slices.

- Where: `ext/quant/exl3_gemv_kernel.cuh` (1-22, 201-204);
  `ext/quant/exl3_dq.cuh`; `ext/quant/codebook.cuh` (38-112).
- Numbers *(stated, `ext/quant/exl3_gemv.cu:22-27`)*: on an RTX 3090 at 4
  bits and up to 8 rows it "wins 15-60% at attention-projection sizes"
  against their pipelined kernel.
- VRAM: none.
- Ares: **take**: it confirms Marlin's 8-row form from an engine built for
  GeForce cards, and it shows the way around the cap (f16 accumulators,
  folded often).

**Integer dot products only up to 2 rows, and an exact form of them**
*(read)*. The int8 path is used for at most 2 rows. A residual mode quantizes
the rounding error of the activations and runs a second `dp4a` on the same
decoded word.

- Where: `ext/quant/exl3_gemv_int8_kernel.cuh` (20-50, 362, 811-823);
  `ext/quant/exl3_gemv_int8.cu` (19-21, 44-58).
- Numbers *(stated)*: the residual mode gives "~15-16 bit effective
  activation precision (KL at parity with the fp16 kernel or better)"; the
  plain mode "~0.9% output RMS deviation".
- VRAM: a 16 MiB workspace per device.
- Ares: **take** the residual `dp4a` as the exact path for 1 to 2 rows. It
  replaces `ares_matvec_q4_K` as the exact kernel and removes the reason to
  accept 1 % rms where the kernel waits for memory anyway.

**Above 144 rows: decode the matrix once, then a dense product** *(read)*.
The weights are decoded once into a transient f16 matrix and a dense product
runs on it, with f16 accumulators that are flushed into f32 every 32 elements
along the inner dimension.

- Where: `modules/quant/exl3.py` (10 `AUTO_RECONSTRUCT_THRESHOLD = 144`,
  140); `ext/hgemm_f16acc.cu` (1-36).
- Numbers *(stated, `doc/env_vars.md:581`)*: f16 accumulation "measured 2.00x
  on the 3090, 4090 and 5090; 1.00x on the RTX PRO 6000".
- VRAM: one decoded matrix, 170 MiB for 17408 x 5120 *(calc)*.
- Ares: **take** for wide steps, in place of decoding tiles inside the `mma`
  loop: the scales are folded into the values once, so the scaling that is
  half of today's loop is gone, and the 2.00x is exactly the GeForce cap.
  The decoded matrix is scratch in the ledger, not a weight; it can be
  decoded a slab of rows at a time to bound it. Aphrodite's successor of
  Marlin does the same as its third tier (below).

**Recurrent layers: two state rows and a flip** *(read)*. One launch per
layer handles all rows, heads and tokens. A sequence has two state rows: a
speculative pass reads the base row and writes the other. When everything is
accepted, a flag is flipped and no kernel runs; when part is accepted, the
accepted prefix is scanned again from staged inputs.

- Where: `ext/gdn.cu` (528, 2293-2430); `cache/recurrent_util.py` (61-105).
- Numbers: none.
- VRAM: 1.0 MiB per row and layer for 32 heads of 128 x 128 in f16, two rows
  per sequence; 16.6 KB of staged input per token and layer *(read, calc)*.
- Ares: a trade to **measure, not to take blindly**. As principle 5 is
  written, the state of every recurrent layer is read for the window and read
  and written again at commit, for every accepted token; with two rows plain
  decode and a fully accepted window need no replay. For Qwen3.8-27B that
  saves one read of 144 MiB of state per token, about 0.3 ms at the RTX
  4070's bandwidth against a step of 28 ms and more *(calc from
  qwen35-architecture.md: 48 blocks of 3 MiB)*, and costs 150 MiB more VRAM
  per sequence. It pays for one or two sequences and not for many. What is
  free is the log for the partial case: mistral.rs keeps the rank-1 updates
  of each drafted row, about 1.5 KiB per head and row, instead of staged
  inputs (`mistralrs-core/src/kv_cache/hybrid_cache.rs:36-93`).

**Checkpoints and the prompt of a recurrent model** *(read)*. The prompt runs
in ranges of 2048 tokens (stated: 18 % faster than one pass of 16k; 1024
costs 38 % more). Checkpoints are taken only at page boundaries: every 2048
tokens while generating, every 32768 in a prompt, and always at the prompt's
last page boundary. They are keyed by the page's chain hash and live
unquantized in RAM.

- Where: `modules/gated_delta_net_fn/gated_delta_rule.py` (89-118);
  `generator/job.py` (1778-1858); `cache/recurrent.py` (159-349).
- VRAM: about 230 MiB transient per layer for a range.
- Ares: **take** the intervals as the starting policy of `set_checkpoint`,
  and the range as a bound for the ledger.

**Attention over the rotated K/V** *(read)*. The algebra is Ares'
(`crates/quant/src/kv.rs`): the query rotated once, K never un-rotated, V
accumulated rotated. What differs:

- Codes sit in bit planes of power-of-two width (5 bits are a 4-bit and a
  1-bit plane), so nothing straddles a word and every plane unpacks with
  aligned shifts. Ares stores `q5` to `q7` as a bit stream of 5, 6 or 7 bits
  per code (`pack_codes`, `crates/quant/src/kv.rs`).
- Decode is split: a program per KV head, block of rows and piece of the
  context, whose rows are up to 32 pairs of query position and sibling head,
  so a tile is unpacked once for all of them. The tile is `max(16, 8192 /
  head size)` keys; the pieces are `min(2 x SMs / programs, length / tile,
  128)`; a second kernel combines.
- From 256 query rows the selected window is decoded once.
- Where: `modules/attention_fn/triton_paged.py` (716-721, 812, 902, 1151,
  1289-1297, 2030-2046); `ext/cache/q_cache_kernels.cuh` (9-24).
- Numbers *(stated)*: unpacking inside the prompt kernel costs "roughly
  5-25%"; staged, it is "within ~1-3% of fp16". A kernel that spills to local
  memory makes CUDA size "the context's local-memory pool for full device
  occupancy ... and never returns it".
- VRAM: scratch for the decoded window, 128 MiB at 4 heads of 128.
- Ares: **take** the bit planes while no kernel reads the formats yet (same
  size, new golden files), the packing of rows and the rule for the pieces,
  and a check at build time that no kernel uses local memory, which lies
  outside the ledger.

**Other pieces** *(read)*.

| Piece | What it does | Where | Ares |
| --- | --- | --- | --- |
| Exact selection for sparse attention | A radix select per query row on a 16-bit key of the f16 score, two histograms of 256 bins, ties by ascending index | `ext/dsa_topk.cu` (75-275) | **Take**, with its tie order as a rule for the open cut-off question |
| Decode-shaped MoE | Two launches per layer over (token, expert) slots; slots of one expert share an `mma` in runs of up to 8; the block that arrives last runs the activation and the sum | `ext/quant/exl3_moe_coop_kernel.cuh` (3-22) | **Take** the shape for MoE decode |
| Launch tuning | Tile shape and number of blocks per matrix shape by elimination rounds, with twice the L2 overwritten before each sample | `ext/quant/coop_autotune.cu` (42-58, 334-377) | **Take** the L2 eviction between samples for Ares' benchmarks; see LMDeploy for the table |
| Tensor split | A process per GPU; sums through page-locked host memory with a CPU thread adding; no peer transfers; weights split by free VRAM | `model/model_tp.py`; `ext/parallel/all_reduce_cpu.cu` | **Leave**; Ares' push kernel is ahead |
| Hand-off to CPU experts | `cuStreamWaitValue32` and `cuStreamWriteValue32` instead of kernels that spin | `ext/cpu/moe_handoff.cu` (65-71, 103-106) | **Take** for the CPU expert pipeline and the host route |
| Stop strings | A matcher with three results (none, partial at the end, full); a full match cuts pages and restores recurrent state from the nearest checkpoint | `ext/generator/strings.cpp`; `generator/job.py` (1024-1129) | **Take** the three-way match |

## mistral.rs

Paths are relative to the repository. Read by a delegated reader. It is the
largest engine in Rust and runs GGUF files, so it is the reference for "how
is this done in Rust", for better and worse.

**GGUF weights** *(read)*. The blocks stay quantized, and the kernels are
ports of llama.cpp's two families, with its constant of 8 rows
(`mistralrs-quant/src/gguf/fast_mmvq.rs:70`). One quantization of the
activations serves several projections, and the query, key and value
projections (and gate with up) are fused launches. Stated: 1 240 to 1 950
tokens/s of prompt against llama.cpp's 772 to 1 211 on one machine
(`releases/v0.9.5/report.md`). Ares: **have** the kernels; **take** the fused
launches.

**The tokenizer from a GGUF** *(read)*. `convert_gguf_metadata_to_hf_tokenizer`
builds a `tokenizers::Tokenizer` from the file's metadata: the `llama` kind
becomes a unigram model with a decoder chain, `gpt2` a BPE model from
vocabulary and merges with a table from `tokenizer.ggml.pre` to a split
expression, and an unknown pre-tokenizer is an error ("refusing to guess").
Chat templates run in minijinja with Python-compatible methods. One table of
token bytes serves llguidance, streaming and stop strings.

- Where: `mistralrs-core/src/gguf/gguf_tokenizer.rs` (119-742);
  `mistralrs-core/src/pipeline/chat_template.rs` (372-448);
  `mistralrs-core/src/pipeline/llg.rs` (9-58).
- Ares: **take** as the recipe to check `ares-tokenizer` against: the
  dispatch on the model kind, the table of pre-tokenizers, the refusal to
  guess, one token-byte table for all three users.

**Grammar and sampling** *(read)*. The sampler picks first and asks
llguidance `validate_tokens(&[tok])`; on a refusal it computes a full mask.
The GPU path returns `2k + 2` floats per row (values, ids, the softmax
denominator, the maximum) (`mistralrs-core/src/sampler.rs:2040-2123`,
`pipeline/sampling.rs:1450-1617`). Ares: **take** the record of candidates;
validate all k and upload a bitmask on a miss.

**The scheduler** runs one bucket of sequences of equal length per step with
a graph per padded batch size
(`mistralrs-core/src/scheduler/default_scheduler.rs:90-118`). Ares:
**leave**; it is what principles 1 and 7 avoid.

## Text Generation Inference

In maintenance mode by its own README. Read by a delegated reader. Two rules
are worth having:

- Admission is charged for the suffix after the prefix match, not for the
  whole prompt (`backends/v3/src/queue.rs:237-504`), and a request that does
  not fit waits in the queue. Ares: **take** both;
  `Runtime::worst_case_pages` charges the whole prompt and `submit` answers
  `Overloaded`.
- Chat templates in minijinja are tested against 28 expected strings
  (`router/src/infer/chat_template.rs:28-136`). Ares: **take** the table of
  cases.

Its trie for the prefix cache (`backends/v3/src/radix.rs`) is what
`PrefixIndex` already is; there is no detokenizer in Rust in the repository.

## Aphrodite, now Sonar

The package only raises "Aphrodite has been renamed to Sonar"; the code is a
fork of vLLM (`sonar/`, `csrc/`, `rust/`). Read by a delegated reader, less
deeply than the others.

- **No mode for a single user and nothing for low VRAM**: K/V is sized as in
  vLLM (`sonar/v1/worker/gpu_worker.py:642`). Its GGUF kernels are in the
  tree but not built. Ares: nothing to take.
- **"Swordfish"**, its successor of Marlin, has a third tier in which "the
  weight dequantizes once into a transient dense buffer and cuBLAS takes the
  GEMM" (`csrc/quantization/swordfish/swordfish_dense_tier.cu:1-5`). It
  confirms ExLlamaV3's decode-once path.
- **A copy-engine operation between two kernels costs a switch of engines**:
  "~90 us per decoded token on B200" for a memset node (stated in its
  source). Ares: keep copy-engine operations out of the step's stream.
- **Samplers**: the long list (DRY, XTC, mirostat and others) is PyTorch on
  rows of the vocabulary, several with a sort. DRY has a host form that keeps
  positions by token and returns a sparse map from token to match length
  (`sonar/v1/sample/ops/dry.py:144-236`). Ares: **take** that form for the
  conductor, as a sparse list of (id, penalty) in the step record.
- **Rust crates** for tokenization (a fast BPE encoder first, `tokenizers`
  as fallback), chat templates and tool-call parsers (`rust/`, Apache-2.0).
  Worth reading before `ares-tokenizer` and tool calls are finished.

## TokenSpeed: what is new at code level

Paths are relative to `tokenspeed-scheduler/csrc/` unless noted. Read by a
delegated reader; the bodies of the admission and capacity code were not.

**Tails: resuming inside a page** *(read)*. A boundary inside a page has its
own hash, computed over the page's tokens up to that boundary and the prior
page's hash. A table maps the hash of the last full page to at most 8 tails
(their tokens and hash), least recently used first; a probe hashes its own
tokens up to each tail's offset.

- Where: `cache/prefix/prefix_hasher.h` (89); `cache/prefix/tail_table.h`
  (49-51, `kMaxTailsPerPrior = 8`).
- VRAM: none.
- Ares: **take** into `PrefixIndex`: a prompt resumes at the real end of the
  cached text and not at the last page boundary, which for a page that holds
  whole tiles of 128 cells is the difference of up to a page of recomputed
  rows on every turn of a conversation.

**Which checkpoints to keep** *(read)*. A checkpoint is ranked by its
distance to the newest checkpoint below it, up to 5; the far ones are evicted
first, and those at branch points keep their rank.

- Where: `cache/prefix/checkpoint_tree.h` (38-48).
- Ares: **take** as the eviction order of checkpoints in `PrefixIndex`.

**Admission as a plan with a refund** *(read)*. The pool's free counts are
copied, tentative victims are removed until the request fits, and a last pass
returns the victims that were not needed.

- Where: `cache/coordinator/shadow_occupancy.h`, `eviction_plan.h`.
- Ares: **take** with LMDeploy's dry run; they are the same idea.

**States own their resources** *(read)*. The request state is a variant
(`Submitted`, `Prefilling`, `Decoding`, `Retracted`, ...) and an event is a
function object with one overload per legal source state; pages, slot and the
count of results in flight travel as one move-only value
(`fsm/states.h:33`, `fsm/forward_states.h:158`). Ares: **have** in spirit
(plain data owned by the conductor); the count of results in flight becomes
necessary once steps overlap.

**A sampler in two stages** *(read; the kernels are a vendored library)*.
Stage one, per tile of 2k to 16k vocabulary entries: grammar mask, penalties
and temperature, each logit packed as a 16-bit key above its index, the
tile's best 128 kept. Stage two merges, applies top-p and min-p to at most
128 candidates and draws with a generator keyed by seed and step. A variant
sends each GPU's best 128 for a head split by vocabulary.

- Where: `tokenspeed-kernel/python/tokenspeed_kernel/ops/sampling/sonic.py`
  (20-36); `tokenspeed-kernel/thirdparty/sonic_sampler/`.
- Numbers *(stated)*: "3.6x to 11x faster than the FlashInfer backend" on one
  datacenter machine.
- VRAM: 128 words per tile and row.
- Ares: the cut at 128 candidates is a real choice (it changes what top-p
  can keep). **Take** the shape only if the exact kernel of TensorRT-LLM
  proves too slow; the best-128 record is the same thing as the candidates
  for the grammar.

**Grammar** is xgrammar behind a C++ bridge, with a full mask per row and
step (`tokenspeed-engine/crates/ts-grammar/src/lib.rs:36-80`). Ares:
**leave**.

## Others

- **Qrita** (arXiv 2602.01518; read as code in vLLM and Sonar): exact top-k
  and top-p by a pivot guessed from the row's mean and deviation, a gathered
  candidate set and a four-way search. Stated: "up to 2x throughput" against
  sort-based kernels. Ares: the second candidate for exact cut-offs next to
  the radix descent.
- **fastokens** (github.com/crusoecloud/fastokens, Apache-2.0; README read):
  a BPE-only encoder with parallel chunks and two caches (prefixes, and
  segments between special tokens). Stated: "10x+ faster" than `tokenizers`.
  Ares: use it if it accepts the tokenizer built from the GGUF; otherwise
  the two caches are the idea.
- **InstantTensor** (a loader with direct I/O; stated: 35 GB/s on a
  datacenter machine) and **CacheGen** (K/V as a compressed bit stream,
  stated 3.5 to 4.3x smaller): looked at and left. Ares is bound by its SSDs
  and the RTX 3060's link, and restores from `ares-store` are not yet
  measured as bound by the disk.

## Take now

Ranked by expected gain. Each item names the part of Ares it changes. The
gains are expectations from the sources' numbers and from arithmetic on
Ares' own measurements; none is measured.

1. **Weight kernels: a device layout per class of type, decoded in
   registers.** At load, rearrange each matrix into the order the tensor
   cores read it, with scales and minimums in their own aligned arrays
   (Marlin's `repack_tile`; TensorRT-LLM's preprocessor; LMDeploy derives the
   layout by running the kernel's own load path once). Then one kernel from a
   few rows to about 64 that decodes in registers with `lop3` and feeds `mma`
   directly, with the 8-row form in which the weights are the 16-row operand
   (Marlin's `mma_trans`, ExLlamaV3's GEMV), a persistent block per SM, and
   f16 accumulators folded into f32 every few slices on capped cards
   (ExLlamaV3). Changes: `ares-cuda` (`mm.cuh`, and `mmvq.cuh` above the few
   rows where it is at the memory limit), the load path, `ares-io` for
   experts. Expected: 5 to 8 rows from 26 % of the
   memory limit toward the limit (a pass over Qwen3.8-27B takes 38.0 ms with
   eight rows and 21.7 ms with one today), and a large matrix at 9 to 34 rows
   from about 0.3 ms toward 0.105 ms on the RTX 4070 *(calc)*. Cost: a
   transform per type at load, a second layout for experts in RAM, kernels
   per class instead of per type.

2. **Weight kernels, wide steps: decode a matrix once, then a dense
   product.** Above a row count to be measured (ExLlamaV3 uses 144), decode
   the matrix, or a slab of it, into f16 scratch and run a dense product with
   f16 accumulators flushed into f32 (ExLlamaV3 `hgemm_f16acc.cu`: 2.00x on
   GeForce cards, stated; Sonar's dense tier). For the int8 path, apply the
   6-bit sub-block scales as integers and convert once per 256 values
   (Marlin's `matmul_a8`). Changes: `ares-cuda` prefill family and the step
   scratch in the ledger. Expected: this is where Ares is behind the fork
   (weights alone at 1054 rows per second against the fork's whole pass at
   1180); the scaling that is half of today's loop disappears and the cap is
   avoided. Cost: up to 170 MiB of scratch per device for the largest matrix,
   less in slabs.

3. **Attention kernels: write the first ones from three sources, not from
   the fork.** Decode on CUDA from LMDeploy's kernel (the head group as one
   side of the `mma`, backward tiles inside a page, packed operands expanded
   in registers, one integer `mma` per group of 32 with the scale applied
   afterwards). Block-sparse from FlashInfer's Qwen reference kernel (device
   id list, block table, loads issued tiles ahead, means of the touched tiles
   gathered as extra rows). Vulkan from FlashInfer's plain-core decode
   kernel. Wide steps: unpack into an f16 staging tile and run the ordinary
   loop, the format a switch per tile (FlashInfer's variant (b); ExLlamaV3:
   1 to 3 % from f16 against 5 to 25 % for unpacking in the loop). One writer
   kernel per step. The Turing variant from LMDeploy's one source. Changes:
   `ares-cuda`, `ares-vulkan`, the step record. Expected: these kernels
   decide the speed at long context; there is no baseline in Ares to compare
   with. Cost: none beyond writing them.

4. **K/V formats, before any kernel reads them.** Bit planes of
   power-of-two width for `q5` to `q7` in place of the bit stream
   (ExLlamaV3); in the pending long-context measurement, the norm-preserving
   scale against the least-squares fit (vLLM's Octave document reports the
   least-squares scale costing 2 to 3 times the loss at 380k tokens, stated),
   and more bits for the rotary dimensions. Changes: `ares-quant` and its
   golden files. Expected: cheaper unpacking at the same size; possibly the
   difference between a `q4` that holds at 262k tokens and one that does
   not. Cost: the formats change once more; now is when it is cheap.

5. **Planner and conductor: admission.** Claim pages per step (vLLM), keep a
   reservation for the output that running requests are expected to produce,
   with SGLang's self-correcting ratio, charge a new request only for the
   rows after its prefix match and let it wait when it does not fit (TGI),
   plan a step's claims against a copy of the counters and commit them whole
   (LMDeploy, TokenSpeed), and when the pool is short spill the sequence with
   the fewest generated tokens (SGLang's order). Changes: `ares-sched`,
   `ares-runtime` (`worst_case_pages`, the `Overloaded` answer). Expected:
   concurrency. Today a request without `max_tokens` reserves a whole context
   length, so with a pool of one context a second request is refused while
   the pool is nearly empty. Cost: the spill and suspend paths must exist
   before the reservation can be optimistic.

6. **State and caches: the prefix index.** Publish a page when it fills
   (vLLM, SGLang); tails so a prompt resumes at the real end of the cached
   text (TokenSpeed's table of up to 8 tails per page, with a copy of the
   cells, which is valid while they are exact `f16` cells: TensorRT-LLM,
   LMDeploy); hold a prompt back while another is computing the page it
   misses; a priority class ahead of leaf-first eviction. Changes:
   `ares-state::PrefixIndex`, `ares-sched::plan`. Expected: on every turn of
   a conversation up to a page of rows less to compute, and sharing between
   parallel agents from the first step. Cost: host bookkeeping.

7. **State and caches: the checkpoint policy for recurrent state.** A
   checkpoint of Qwen3.8-27B is 150 MiB, so where and how they are kept
   decides whether prefix caching is affordable at all. Take: intervals as in
   ExLlamaV3 (dense while generating, sparse in a prompt, always at the
   prompt's last page boundary); the checkpoint a little before the prompt's
   end and a replay over cached pages without K/V writes (LMDeploy); the
   checkpoint emitted by the commit when it crosses the grid (SGLang);
   eviction by distance to the next checkpoint below and before K/V pages on
   interior entries (TokenSpeed, SGLang); stored compressed (SGLang's int8
   with a scale per head and channel is a quarter of f32). Changes:
   `ares-state`, `ares-store`. Expected: prefix hits that do not recompute
   back to a far checkpoint. Cost: RAM per checkpoint.

8. **Sampler.** One kernel per row in which every filter is a threshold on
   `p / p_max` and exact cut-offs come from a radix descent (TensorRT-LLM's
   fused sampler); the pick by Gumbel noise keyed by seed, position and token
   (vLLM, FlashInfer); a record of the best candidates per row for the
   grammar (mistral.rs' `2k + 2` floats); penalties, DRY included, as sparse
   lists from the host (MLC, Sonar); a pair per device when the head is split
   by vocabulary (TensorRT-LLM). Changes: `ares-cuda`, `ares-vulkan`, the
   step record, `Mesh`. Expected: 2 to 3 sweeps of a row of logits and no
   rows of the vocabulary in VRAM. Cost: none beyond writing it.

9. **Planner and conductor: steps that overlap, restores that overlap.**
   The pick stays in a device table that the next step reads (SGLang's
   `FutureMap`); a finish by end token is seen one step late and its row
   dropped; pages freed by a commit are reusable only after the step in
   flight; a restore is issued per layer with an event the forward stream
   waits for at that layer, and a request in restore is skipped, not waited
   for (SGLang). A context term in the step-time model (SGLang's quadratic
   fit). A budget of draft rows across requests by survival probability
   (vLLM). Changes: `ares-runtime`, `ares-sched` (`ContextFit`, `plan`),
   `ares-state::PagePool`. Expected: small on one device (the conductor
   already costs microseconds); real for the layer split with several
   sequences, for the time to the first token on a cache hit, and at eight
   requests where a draft row costs a launch.

10. **Speculation.** One lookup drafter built on a suffix automaton or an
    incremental trie, shared across requests (TensorRT-LLM, SGLang, vLLM's
    suffix decoding), in place of the three n-gram variants; drafts of model
    drafters referenced on the device and not carried through the host
    (SGLang); drafts cut at the first token the grammar refuses before they
    are verified (vLLM). Changes: `ares-sched`, `ares-runtime`. Expected:
    longer matches, matches across requests, fewer wasted rows.

11. **Structured output.** llguidance on the GPU's candidates as planned,
    plus: forced tokens as rows that are neither sampled nor verified, with
    the seam tokenized again (MLC's jump-forward, the only one that runs);
    the mask filled while the device computes (vLLM); requests whose grammar
    is compiling parked in their own queue, the grammar held off during
    thinking, tool calls as grammars with a trigger (SGLang). Changes:
    `ares-runtime`, `ares-sched`, `ares-api`. Expected: forced runs of JSON
    cost no sampling; nothing waits for a compile.

12. **Loading and the tokenizer.** The layout transform as a GPU kernel at
    load and a measured table of which kernel serves which shape and row
    count, kept on disk (LMDeploy); the tokenizer checked against
    mistral.rs' recipe from GGUF metadata, with one table of token bytes for
    grammar, streaming and stop strings; a tokenization cache verified at the
    seam (TensorRT-LLM); chat templates in minijinja with a table of expected
    strings (TGI). Changes: the load path of `ares-cuda`, `ares-tokenizer`.

Outside the eight parts, smaller: the sentinel per vector and the residual
and norm inside the sum kernel (TensorRT-LLM) for `Mesh`; integer dot
products next to cooperative matrices in the Vulkan probe; pools on reserved
address ranges with chunks the ledger can move (TensorRT-LLM, SGLang); a
build check that no kernel uses local memory (ExLlamaV3).

## Where Ares is still llama.cpp-shaped without a reason

What follows copies llama.cpp, or Piggidragon's fork of it, where another
engine does the same thing better and nothing in Ares' fixed points asks for
the copy. Keeping the weights and the K/V small is not on this list: that is
the part of llama.cpp Ares wants.

1. **The device holds the file's layout.** Every weight kernel decodes GGUF
   blocks as they lie in the file: `BLOCK_VALUES, BLOCK_BYTES: the geometry of
   a block (GGUF layout)`, loads from 2-byte aligned blocks, funnel shifts for
   17-byte blocks, sub-block scales and minimums interleaved with the codes.
   That is ggml's premise (the file is mapped and used as it is), and it is
   the stated reason for what bounds `mm.cuh`. The fixed point is the size in
   VRAM, not the layout; the scope section already allows a rearrangement.
   Marlin, TensorRT-LLM, LMDeploy and ExLlamaV3 all store weights in the
   order their kernel reads them. **Instead:** take item 1 of the list above.

2. **A kernel per type, for 28 types, on two backends.** It follows from
   point 1: when the device layout is the file's, every type needs its own
   loads. Most types are the same thing with different numbers. **Instead:**
   device layouts per class. (a) Plain codes with a scale, and for some a
   minimum, per group: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K, the 1-
   and 2-bit and ternary types with fixed levels. One template with the bits
   per code, the group length and the presence of a minimum as parameters,
   as Marlin's is. (b) Sixteen levels that are not evenly spaced: IQ4_NL,
   IQ4_XS, MXFP4, NVFP4. The same template with a 16-entry table in
   registers. (c) Grid codebooks: the IQ1, IQ2 and IQ3 types and TQ1_0 keep
   their own decoders. (d) Floats. The file-order kernels that exist stay as
   the fallback for a type whose class kernel is not written.

3. **Two kernel families split at 8 rows.** `mmvq` for 1 to 8 rows and `mm`
   above is ggml-cuda's split, with its constant (`MMVQ_MAX_BATCH_SIZE`;
   mistral.rs copied it too). Ares' own numbers say the split is in the wrong
   place: integer dot products reach 89 to 98 % of the memory limit at one
   row and 26 % at 5 to 8. ExLlamaV3 stops them at 2 rows; Marlin has one
   tensor-core kernel from 1 to 64. **Instead:** integer dot products only
   while they are at the memory limit (up to 3 or 4 rows by Ares' own
   numbers), the register-decode tensor-core kernel from there to about 64,
   the decode-once path above; the boundaries from a measured table per
   device, not from constants.

4. **The prefill kernel decodes tiles inside the `mma` loop.** That is
   llama.cpp's structure for wide products, engineered better. `mm.cuh` says
   what it costs: a barrier per step, the scaling at 260 of 500 to 620
   instructions, weight loads at a third of the bandwidth below 64 rows.
   **Instead:** items 1 and 2 above.

5. **One percent of error is accepted because llama.cpp has it.** Principle
   8 sets the bar at "about 1 % rms per product from 8-bit activations in
   llama.cpp's quantized kernels", and the activations are quantized in
   ggml's own format (int8 with a scale and a sum per 32). Marlin multiplies
   f16 activations exactly; ExLlamaV3 runs a second integer dot product on
   the rounding error and reports parity with f16 where the kernel waits for
   memory anyway. **Instead:** the exact form is the default wherever it is
   free (the few rows of the integer kernels by the residual, the rows above
   them by f16 activations), and the 1 % stays only for the int8 path on
   capped cards at wide steps.

6. **Admission is a slot's reservation under another name.** The pages are
   not tied to a slot, but `RuntimeConfig` has `max_slots`, and `submit`
   reserves every request's "longest possible length" (prompt plus
   `max_tokens`, a whole context when the client names none) and answers
   `Overloaded` when that does not fit. That is llama.cpp's `n_ctx` per slot
   with a counter in place of memory. vLLM claims per step; SGLang reserves
   what is expected and corrects the expectation; TGI queues. **Instead:**
   item 5 above.

7. **The lookup drafters are llama.cpp's three.** ARCHITECTURE.md plans "the
   full drafter stack of llama.cpp": the simple lookup, then the map and
   cache variants. The simple one keeps the last position of every run of n
   tokens of one sequence. **Instead:** one suffix automaton or incremental
   trie, with a corpus shared by all sequences (item 10).

8. **Drafts travel through the host.** "All drafters produce the same thing,
   a list of proposed tokens", written into the step record. That is right
   for lookup drafters and is how llama.cpp drives every drafter. For a draft
   head or a draft model it puts a copy to the host and back between two
   kernels of the same device, every step. **Instead:** a drafter that is a
   model leaves its tokens in a device table that the window reads (SGLang);
   the host learns them with the picks.

9. **The loop is synchronous.** "The next step is planned only when the
   current one has ended." It is listed under next steps as "overlapped
   steps", but without the rules that make it work. **Instead:** item 9
   above; the rules are the content.

10. **The step cost is the fork's line.** `base + per_row x rows` was fitted
    on the fork, where it describes weight reads. Attention grows with the
    context, and a prompt chunk at 100k tokens costs more than one at 1k.
    The planner's cap on prompt rows next to generation comes from this
    line. **Instead:** a term for the context (SGLang fits a quadratic).

11. **K/V codes as a bit stream.** `q5`, `q6` and `q7` store 32 codes as a
    stream of 5, 6 or 7 bits each, least significant bit first. That is
    ggml's habit of packing by position, and it is worse than ggml's own
    Q5_K and Q6_K, which split low and high bits. **Instead:** planes (item
    4), before the first attention kernel is written against the stream.

12. **Prefix hits end at page boundaries, and it is not said when pages
    enter the index.** llama.cpp caches a slot's prompt when the slot is
    done; the description of `PrefixIndex` reads the same way. **Instead:**
    item 6.

Checked and **not** on this list, because no other engine does them better:
the server's flags and model presets, the split planner for unequal cards
(no engine here has one; ExLlamaV3 splits by free VRAM only), the push sums
between devices (the same class as SGLang's and TensorRT-LLM's best), direct
I/O for loading, the page pool and its ledger, verify and commit, and the
exact cells of the low-bit K/V formats.

## Corrections to the first survey

- **SGLang's queue**: a queued request that shares a prefix with another is
  sorted last, not held; this happens only under the cache-aware orders, the
  default is first come, first served, and it is off above 128 queued
  requests.
- **SGLang's recurrent state under speculation**: snapshots per verified row
  are the default; replay at commit is an option.
- **SGLang's 8-bit float K/V** also loses accuracy in its own table (aime25
  on Qwen3-235B: 0.7733 at 16 bits, 0.7333 at 8, 0.6000 at 4).
- **SGLang's paged experts** are for one GPU and refuse speculation and GGUF.
- **vLLM** has removed cascade attention, and its GGUF support is an
  out-of-tree plugin.
- **Aphrodite** is now called Sonar and has nothing for low VRAM.
- **ExLlamaV3's tensor split** is a process per GPU with sums through host
  memory; the first survey did not look at it.

## What was not read

- **Everything delegated** was read by other readers at the commits in the
  table; their file references were spot-checked, not re-read. Inside the
  SGLang, ExLlamaV3, mistral.rs and TGI parts some topics were read by
  readers of those readers (SGLang's tiers, speculation, several GPUs and
  sampler; ExLlamaV3's recurrent layers, tensor split, CPU experts, sparse
  selection and loader; all of mistral.rs and TGI).
- **Binaries without source**: TensorRT-LLM's and FlashInfer's attention,
  matrix and MoE kernels for Blackwell, cuDNN's sparse attention for the Qwen
  models.
- **Papers and blogs** only through abstracts, appendix tables or
  summarising fetches: the Marlin paper's body, the FlashInfer paper's
  figures and its three blog posts, SGLang's release notes. Values read off
  Marlin's plots are by eye.
- **vLLM**: the out-of-tree GGUF plugin (a summary only); the bodies of the
  rejection sampler, the pivot-search sampler, the adaptive verification and
  the low-bit K/V kernels (headers and documents only); the suffix tree,
  xgrammar and llguidance themselves (external packages); the asynchronous
  scheduler.
- **SGLang**: the Rust tree core beyond its README and node type; the
  paged-experts and sparse working-set sources; the drafters for block
  drafts. The clone has one commit, so there is no history to say when or
  why jump-forward lost its caller.
- **FlashInfer**: the generated kernels for the Qwen models, the tail of the
  reference kernel beyond its header comments, the kernels for Hopper and
  Blackwell, the bodies of the top-k and the gated-delta-rule kernels.
- **TensorRT-LLM**: the first K/V manager, the sparse working-set kernels
  beyond their design document, the fused MoE kernels, the kernel for relaxed
  acceptance.
- **MLC-LLM**: the draft and verify actions, the attention as TIR; upstream
  TVM (only the commits MLC pins).
- **LMDeploy**: the scheduler's source (its README was used), the merge of
  split products, the checkpoint code of its PyTorch engine.
- **ExLlamaV3**: the inner pipeline of its wide matrix product, the prompt
  attention kernel's body.
- **TokenSpeed**: the bodies of admission, coordinator and capacity model;
  its recurrent kernels.
- **Sonar**: its sampler kernel, its prefill kernel beyond the header, the
  Rust crates beyond the tokenizer's interface.
- **Not reached at all**: NVIDIA Dynamo's Rust crates (a directory listing
  only), fastsafetensors, Run:ai's streamer, tensorizer, any new engine in
  Rust or on Vulkan.
