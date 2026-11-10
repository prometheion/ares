# Architecture of Ares

This document describes how the engine is built, why, and how far each part
is. It is the place to start before changing anything.

The goal ([issue 1](https://github.com/prometheion/ares/issues/1)) is one
inference server that takes the best technique of every existing engine and
drops everything kept only for compatibility: maximum speed and maximum VRAM
efficiency for the newest models, on Linux, on one to a few consumer GPUs,
from GGUF files with llama.cpp's quantization for weights and for the KV
cache.

Scope decisions:

- **Current models have recurrent layers.** Every architecture the engine
  targets mixes recurrent layers with attention that is either dense (Spark)
  or sparse (Flash-Next, Qwen4). State that is not addressable by position is
  therefore the normal case, not an extension: checkpoints, verify-then-commit
  and resuming from a cached prefix are designed around it, and nothing is
  built that only helps pure-attention models.

- **No compatibility, with anything.** This is a new engine and carries no
  legacy: no old file format versions, no flags kept for habit, no code paths
  for hardware or models that are too old. Being familiar to llama.cpp users
  is welcome where it costs nothing and is never a reason to keep something.
  A GPU that cannot run the current kernels is not supported, and a model
  architecture is removed when it is superseded.

- **Server only.** One binary, `ares`. No command line tools, no GUI.
- **Linux only.** That allows direct I/O, `io_uring` and huge pages without
  a portability layer.
- **Newest models only.** An architecture is supported while it is current
  and removed when it is superseded. There is no generic operator graph to
  keep old models running.
- **CUDA and Vulkan.** Everything CUDA 13 runs on is supported and tuned for;
  every other GPU goes through the generic Vulkan backend. Both device layers
  exist from the start and are held to the same tests; everything above them
  is written once.
- **No CPU inference.** A model does not run on the CPU, and no layer of a
  dense model is offloaded to it. The CPU computes in one place only: the
  experts of a MoE model that are not on the GPU (see [Weights](#weights)).
- **Rust**, with kernels in CUDA C++ and GLSL.
- **No opt-in for improvements.** An optimisation without a downside is
  simply how the engine works and has no switch. Options exist only for real
  choices, where something is traded: quality for memory (expert pruning,
  KV format), generation speed for prompt speed.

## Principles

Each of these answers a cost that was measured in llama.cpp on the target
machine (RTX 4070 + RTX 3060, Qwen3.8-27B) or read from the engines listed in
[What is taken from where](#what-is-taken-from-where).

1. **Shapes are data.** There is no graph that is built or instantiated per
   batch shape. A model's forward pass is ordinary code that enqueues kernels,
   and every kernel takes its sizes as arguments. A step may carry any mix of
   rows (one token each of three conversations, five drafts of one of them, a
   slice of a prompt) at no setup cost. In llama.cpp a change of batch shape
   under tensor split costs about 18 ms and every cached shape about 26 MiB of
   VRAM per GPU.
2. **A step is enqueued whole.** Each device has a stream that is filled for
   the entire step. Where partial results cross between devices, a stream
   waits for the other device's event, on the device. The host never
   synchronises between layers. In llama.cpp the per-layer sums across two
   GPUs cost about 4.6 ms of a 41 ms step in waits.
3. **State lives in pages, not in slots.** K/V rows and indexer keys are
   stored in pages drawn from one pool. No memory is reserved per slot,
   sequences share the pages of a common prefix, and attention reads through a
   page table, so sparse attention reads only what it selects and pages can
   live on another tier.
4. **Memory is planned, not requested.** Every byte of VRAM and pinned RAM is
   reserved in a ledger before the model runs. There are three kinds of device
   memory and none of them is a general allocator: weights (placed once),
   pools of fixed-size units (pages, expert slots), and one scratch area per
   step. A configuration that does not fit is refused at load, with the list
   of who holds what; nothing can run out during a request.
5. **Verify, then commit.** Drafts are verified in a window that appends to
   position-addressable state and leaves recurrent state untouched; the
   accepted prefix is then committed. Rejected rows are cut from the page
   table. There is no snapshot and restore of recurrent state per step.
6. **Sample on the device.** Logits stay on the GPU; penalties, top-k, top-p,
   min-p and the pick run there. The random number for a position comes from a
   counter-based generator keyed by (seed, position), so verifying a draft is
   an exact comparison with what plain decoding would have produced, and
   results do not depend on batch composition.
7. **One thread decides.** All bookkeeping (pages, plans, expert tiers) is
   plain data owned by the step loop, with no locks. Other threads do work
   that waits: one per device to enqueue, a pool for file I/O, the HTTP
   runtime. This is the answer to "actually good multithreading": threads
   never contend for the engine's state because only one of them owns it.

## Crates

| Crate | What it holds | State |
| --- | --- | --- |
| `ares-core` | Tensor element types and block geometry, byte units, memory ledgers | Done, unit tested |
| `ares-gguf` | Zero-copy GGUF reader (version 3, split files, validation at open), a small writer for tests | Done; tested against synthetic, truncated and corrupted files and real models |
| `ares-quant` | CPU reference decoder for every GGUF type; the specification and reference codec of the engine's own KV formats (`f16`, `q8` to `q4`, tiles with a mean, the policy of exact cells) | Done; bit-identical to ggml's C code on fixtures generated from it; the KV formats are pinned by golden files and were measured on the K and V of real models |
| `ares-cuda` | Driver API bound at run time, devices, memory, streams, events, modules, the decode and prefill weight kernels for eleven quantization types, the mesh that sums across devices over peer access or the host | Device layer and decode kernels done; prefill kernels work and are verified but are not yet faster than the fork; tested on both GPUs, with and without peer access |
| `ares-vulkan` | The same device-level surface on Vulkan, the same first kernels as compute shaders | Done; 23 device tests pass on both GPUs |
| `ares-state` | Page pool, page tables with copy on write, prefix index with checkpoints, page layouts | Done; randomised model check |
| `ares-store` | State on disk: pages and checkpoints by content hash in segment files, per fingerprint, with a size limit | Done as a store of opaque bytes, tested against restarts, the limit and damaged files; not yet connected to devices |
| `ares-io` | Reads of weights straight from the model file: io_uring with direct I/O, demand before prefetch, cancel and promote, copies of a file on several drives with the reads spread over them, a map from experts to file ranges, row reads behind a cache | Done, tested against real files and on two drives; 1.4 to 1.7 GiB/s of random expert-sized reads per SSD of the development machine and 2.9 to 3.0 GiB/s over both; not yet connected to devices |
| `ares-sched` | Step planner (generation, decision rows and prompts in one step), the order of a pass for long prompts of a model with experts, step time model, draft length choice, n-gram drafter | Done, unit tested |
| `ares-weights` | Expert heat, tier election with two-phase swaps, the waves of experts of a pass, routing bias for pruning | Done, unit tested |
| `ares-api` | Request, event, decision and error types; the `Engine` trait between HTTP and the runtime (`submit` for generation, `decide` for decisions) | Done, unit tested |
| `ares-server` | OpenAI-compatible HTTP server, `POST /v1/systemone` (wire-compatible with llama.cpp's, served for any model) and the `ares` binary | Done as a front end, tested against a mock engine; answers that no model architecture is implemented until the runtime exists |
| `ares-runtime` | The conductor thread and one thread per device, the step loop that connects plan, state and kernels, speculation with the n-gram drafter, the `Model` trait a model implements, a synthetic model that drives the real kernels | Done as a skeleton; 71 tests and 10 GPU tests on one device, tensor split and layer split; implements `Engine::submit` (not `decide`), greedy picks only, no prefix cache, no real model |

Not started: `ares-model` (architecture implementations), `ares-tokenizer`.

## A step

```
requests ──► plan ──► upload step record ──► window ──► sample ──► commit ──► events
              │        (rows, positions,      (all layers,  (on device)  (accepted prefix)
              │         page tables)           all devices)
              └── what each sequence wants: prompt rows, or newest token + drafts
```

1. **Plan** (`ares_sched::plan`). This is continuous batching as vLLM does
   it: prompts and generation share steps, and there is no separate mode for
   reading a prompt next to a generating request. Every generating sequence gets a row for
   its newest token, then drafts while rows are left; prompts take the rest,
   shortest first, starved ones before that. While anyone generates, the
   prompt share is capped at the rows that fit in an allowed slowdown
   (`prefill_rows_next_to`, from the measured cost line `base + per_row *
   rows`).
2. **Step record.** Everything that varies per step (row to sequence and
   position, page-table changes, draft tokens) is written to one pinned
   buffer and uploaded with one copy per device. Kernels read it there, so
   launches carry only pointers and sizes.
3. **Window.** The model enqueues its kernels on every device's stream.
   Position-addressable state is appended; recurrent layers read their base
   state and keep their inputs.
4. **Sample.** A kernel per sequence row picks the token. Only the picks, and
   on request a few candidates for grammar-constrained sampling, go to the
   host.
5. **Commit.** For each sequence the accepted prefix is replayed into its
   recurrent state and the page table is cut after the last accepted row
   (`Sequence::truncate`). Drafters and their acceptance statistics are
   updated.

### Threads

`ares-runtime` runs this loop with one **conductor** thread and one thread
per device. The conductor owns everything that is bookkeeping (sequences,
page tables, the plan, the time model, drafters, request state) as plain data
without locks. A device thread owns its stream and its part of the model. Per
step the conductor hands every device thread one job and gets one result
back; the only place a host thread waits for a device is the device thread's
wait for the end of its step.

A model plugs in through the `Model` trait (what it is, its split over
devices, its page layout, and `open` on each device's thread) and
`DeviceModel::enqueue`, which queues the kernels of a step and never waits.
Until a real model exists a synthetic one drives the real kernels: 64 layers
of two q4_K matrices of 5120 x 17408 each, with sums between devices for the
tensor split and a hand-over per group of rows for the layer split. Its next
token is a pure function of a sequence's own tokens, so the tests can check
that a request gets the same text alone and next to 1, 7 and 31 others, with
speculation on and off, on every placement.

Measured 2026-11-10 with that model (RTX 4070 and RTX 3060, 128 tokens per
request, medians):

- **The host is not in the way.** The conductor's own work is a few
  microseconds per step against steps of 18 to 105 ms, and all devices idle
  for 1 to 5 microseconds between the end of a step and the next dispatch
  (0.01 to 0.03 % of the step). Under `nsys`, 99 % of consecutive kernels
  start within a microsecond of each other and the host path makes no system
  call per step.
- What it took: a conductor that parks while it waits costs 20 to 50
  microseconds per step for the wake-up, so it spins through the step
  instead. Pinning threads to cores, polling the device and the driver's
  wait flags were measured and changed nothing; they are not in the code.
- **One device:** 50.5 tokens/s for one request, 272 for eight, and flat
  above that (292 for 32), because the decode kernels take at most eight rows
  per launch and the runtime does not yet switch to the prefill kernels for
  wider steps.
- **Tensor split** (peer access): 67.4 tokens/s for one request against 50.5
  on one device, 250 for eight; slower than one device from eight requests
  on, where the sums cost more than the halved weight reads save. Without
  peer access each sum goes through host memory and the split is slower
  still (56.4 and 193).
- **Layer split:** 43.2 tokens/s for one request, where each device idles
  half of the time, and 320 for 32, where the two halves overlap. Steps do
  not overlap yet: the next step is planned only when the current one has
  ended.
- **Speculation** with the n-gram drafter on repeating text doubles one
  request (205 to 425 tokens/s on a 16-layer model) and gives nothing at
  eight requests, where a ninth row costs a second launch.
- Through HTTP with 32 streaming clients the single device gives 300
  tokens/s, the same as without the server.

These are numbers of a synthetic model without attention; they say what the
threads and the hand-overs cost, not how fast a real model will be.

## Sequence state

A page holds the state of a fixed number of consecutive positions for every
part of the model that keeps state per position, on every device; devices keep
one array per part, indexed by page number.

- `PagePool`: fixed capacity from the VRAM ledger, reference counts.
- `Sequence`: a page table. `append` returns at most one `CopyOnWrite`, issued
  when the sequence is about to write into a page others also read.
  `truncate` removes rejected drafts. `fork` and `fork_prefix` share pages,
  which makes several continuations of one prompt (branch scoring, parallel
  tool calls, n-best) cost only their own tails.
- `PrefixIndex`: every full page is named by a chain hash of the context and
  all tokens up to its end. A new prompt walks its pages from the start and
  takes the cached ones. The index holds pages after their sequence ended and
  gives them up oldest first, from the ends of chains, when the pool runs
  short. This one mechanism replaces per-slot prompt caches, slot similarity
  matching and a separate conversation cache, and it is the key under which
  pages will be written to disk.
- Checkpoints: state that is not addressable by position (recurrent layers)
  can only be resumed where a checkpoint exists, so checkpoints attach to
  index entries and `PrefixMatch::last_checkpoint` says how far a prompt can
  resume.
- Decisions ("Jev mode"): choosing among fixed options (a label, a tool, the
  fields of a schema) needs no generation. The options are scored from the
  logits after the shared prompt, and options longer than one token are
  forks of the prompt's sequence that run as rows of one step. TheCodacus'
  fork does this for llama.cpp on `/v1/decision`; here it is a use of `fork`
  and the planner, on the official endpoint, `POST /v1/systemone`. The
  endpoint is wire-compatible with llama.cpp's and is served for any model,
  not only for native decision models. Decision rows share steps with
  generation: the planner takes them as `Want::Score`, after the newest token
  of every generating sequence and before drafts and prompts, so neither kind
  of request waits for the other. The endpoint and `Engine::decide` exist and
  are tested against a mock engine; the scoring comes with the runtime.
- `PageLayout`: bytes per page per device, from which the pool's capacity
  follows. For a `qwen4exp`-shaped model the full 262 144-token context is
  3.47 GiB with K and V as Strata stores them (int8, a scale per 64 values)
  and 3.56 GiB with the engine's `q8` (a scale per 32), shared by all
  sequences.

**KV formats.** K and V are stored in one ladder of formats with plain
names, chosen separately for K and for V: `f16`, then `q8`, `q7`, `q6`, `q5`
and `q4`, where `qB` is B bits per value plus an f16 scale per group of 32
values (B + 0.5 bits per value; `q4` is 18 bytes per 32 values). Each group is
rotated with a 32-point Hadamard transform before it is quantized, which
spreads outliers over the group. The rotation costs nothing when reading: a
dot product is unchanged by it, so attention rotates the query once and never
un-rotates K, and V is accumulated rotated and un-rotated once at the end. The
reference encoder and decoder live in `ares-quant`; the attention kernels of
both backends are held to them.

`q8` is the default for K and V: against `f16` it loses hardly anything and
it is the fast choice. Below it the formats work harder the smaller they get:

- `q7` to `q4` search the scale of each group when writing (3, 5, 9 and 17
  candidates, each with a least-squares fit of the scale).
- They store their rows a **tile** of 128 cells at a time, relative to the
  mean of the tile: one f16 vector per tile, 0.125 bit per value. This is the
  one technique that costs something when reading: one extra term per tile
  and query (for K a dot product of the query with the mean, for V the mean
  times the summed weights of the tile).
- A `KvPolicy` keeps the first tile and the newest cells of a sequence in
  `f16` (none at `q8`; at `q4` the first 128 cells and the newest 256 to
  383), which is what kvarn (beellama.cpp) gets most of its gain from. A tile
  is converted when it is full and leaves the newest window, so its mean is
  known by then.

Any K format combines with any V format. A page has to hold whole tiles: the
stored rows of a tile depend on all of its cells, and a page is shared
between sequences by the tokens up to its own end.

The formats were chosen on synthetic rows and then measured on the K and V of
two real models (Qwen3.5-0.8B and Qwen3.8-27B, one text of 3.7k tokens; the
measure is the error of attention, not of logits;
`docs/reference/kv-real-data.md`):

- The scale search lowers the error of the attention output by 12 %, 15 % and
  18 % at `q6`, `q5` and `q4` against the plain scale of `q8`, in every layer.
- The exact cells are the largest effect: 76 % less output error at `q4` with
  128 first and 256 newest cells. The newest cells carry most of it.
- kvarn's quantizer is better on K and worse on V; on the attention output it
  is within 6 % of the engine's. Its two-axis normalisation adds nothing on
  top of the engine's format.
- The mean of a tile came out of this measurement: a quarter to a third of the
  energy of real K and V is the mean of each value over neighbouring cells.
  With the built codec and the engine's policy it lowers the output error by
  25 %, 27 % and 32 % at `q6`, `q5` and `q4` on the 0.8B (19 %, 21 % and 25 %
  on the 27B) and the KL divergence of the attention weights by 48 to 49 %
  (42 to 44 %). No layer, part of the context or kind of text gets worse.
- All of it together, `q4` for K and V on the 0.8B: the output error is 6.5 %
  with a plain scale and nothing exact and 0.74 % as built.

What is still missing is a measurement on logits at a long context: with 3.7k
tokens the exact cells are an eighth to a fifth of what a query reads.

**On disk.** `ares-store` keeps full pages, and the checkpoint that belongs
to a page's end, under the same chain hash in a directory per fingerprint
(model, quantization, state formats, page size; nothing that leaves the state
unchanged, such as drafters or the split over devices). A page is written
once, when it is full, so nothing is written at unload, a shared prefix is
stored once, and a restart finds everything by reading record headers. Space
is reclaimed a segment at a time, least recently used first, after the stores
of other fingerprints. It is a cache: it does not sync, and a torn or damaged
record costs only itself. Moving page bytes between the devices and the store
comes with the runtime.

A randomised test drives sequences through appends, rollbacks, forks, cache
hits and evictions against a simulated device and checks after every
operation that each sequence reads exactly the state its own tokens produce,
that nothing is written into a shared page, and that no page leaks.

### What this prepares for Qwen4

The reference is Strata's implementation of Qwen3.8-Flash-Next (`qwen4exp`);
see [reference/strata-notes.md](reference/strata-notes.md). The items of the
Qwen4 handover map as follows.

| Handover item | In this engine |
| --- | --- |
| Sparse attention on index lists (the indexer's selection should be what attention reads, not a mask over all cells) | Attention reads through an id list and the page table by design. The page size is a multiple of the indexer block, and the pooled key of a block is a per-page part of the layout. The selection and attention kernels come with the model. |
| Sparse KV offload (K/V in pinned RAM, keys and a page cache on the GPU, fetch only missing selected pages) | Pages are the unit of residency. The selection needs only the pooled keys on the GPU (1.5 KB per token for `qwen4exp`), so K/V pages can live in pinned RAM behind a VRAM page cache. The residency layer is the next piece of `ares-state`. |
| Ragged one-pass verification (drafts of unequal length in one batch) | `plan` puts windows of any lengths in one step. Recurrent kernels take per-sequence segments from the step record. |
| Graph cache for batch shapes | Not needed: there is no per-shape graph. |
| Adaptive draft length | `best_draft_len` maximises expected tokens per millisecond from measured acceptance per position (or the draft head's confidence) and the measured step cost. The same number chooses between drafters and accounts for the rows a draft costs other sequences. |
| Rollback rows independent of draft length | Verify and commit removes rollback: nothing recurrent is modified before acceptance. |
| Draft vocabulary without extra VRAM | Planned in weight placement: the output head's rows are ordered by frequency at load, so a draft head is a prefix of the same tensor. |
| VRAM as the binding limit | Ledgers and the three kinds of device memory. |
| Prefill next to generation | `prefill_rows_next_to`. |

## Speculation

The engine carries the full drafter stack of llama.cpp, with the selection of
Piggidragon's fork on top:

| Part | What it is | State |
| --- | --- | --- |
| N-gram drafters | Continue an earlier occurrence of the current text. The simple lookup exists; the map and cache variants follow | `NgramDrafter` done |
| MTP | The model's own draft head or block | With the model |
| DFlash, DSpark, DFlash2 | A small draft model that reads the target's hidden states at a few layers and proposes a block of tokens at once | With the runtime: it needs taps on the target's layers and a second model on the same devices |
| Drafter selection | Per sequence and step, the drafter with the highest expected tokens per millisecond, from measured acceptance and cost; drafters that were not chosen keep learning from what was accepted | `best_draft_len` and `Acceptance` done; the selector object that holds them per drafter is next |
| Adaptive draft length | The length that maximises the same number | `best_draft_len` done |
| Verification | Windows of any length for any number of sequences in one step, then commit | Planner done; kernels with the model |

All drafters produce the same thing, a list of proposed tokens with an
optional confidence per token, so selection and verification do not depend on
where a draft comes from.

Adaptive draft length did not pay off in the llama.cpp fork (a few percent up
for English and code, a few percent down for German). Two of its costs there
do not exist here: a new batch shape built a new graph (about 18 ms, and 26
MiB of VRAM per GPU for each cached shape), and a rejected draft past the
rollback rows forced a replay. Whether the gain is real must be measured
again on this engine with a real model, in German as well as English.

## Weights

There are two ways a model's weights are held, and no third:

- **Dense: everything in VRAM.** All weights of a dense model are on the
  GPUs, placed on one device or split across several when the model is
  loaded; the split planner (next in `ares-runtime`) works from the memory
  ledgers, the measured memory bandwidth of each device and the measured cost
  of the link between them. There is no offload of dense weights or layers to
  the CPU. A dense model that does not fit is the wrong model for the
  machine; what can give way is its context (see KV offload below), or a MoE
  model is used.
- **MoE: experts in tiers.** The dense parts are in VRAM; the experts are
  spread over VRAM, pinned RAM and SSD and follow the routing, as described
  next. Experts that a token needs and that are not on the GPU are computed
  on the CPU or streamed to the GPU, whichever a measured cost model says is
  faster, in a pipeline next to the GPU's work. There is no setting for how
  many layers keep their experts on the CPU: the tiers are sized from the
  memory budgets and filled from what the router picks.

**KV offload** exists for both, and is adaptive: K and V pages that do not
fit in VRAM live in pinned RAM, as many as fit stay resident, and the rest are
brought in as attention needs them. For attention that selects blocks
(Qwen4-style) only the selected pages move. For dense attention every page is
read at every token, so the stream of pages from RAM is bound by the PCIe
link and is prefetched layer by layer through a ring, as in
RaymondHuang210129's adaptive KV streaming for llama.cpp, which is the
reference for this part. It is the next piece of `ares-state` (residency of
pages) and comes with the runtime.

- **MoE tiers.** Experts live on VRAM, pinned RAM or SSD. `ExpertHeat` counts
  routed picks with decay (seeded from a routing profile when one ships with
  the model); a `SlotSet` per layer and tier elects the hottest experts and
  re-elects every few tokens with a margin against thrashing and a cap on
  swaps. A swap hides the evicted expert at once and shows the new one when
  its data has arrived. On a synthetic workload whose topic changes every 500
  tokens, re-election serves 79 % of the picks from a tier of a quarter of the
  experts where a tier filled once serves 29 %.
- **Dynamic expert pruning.** `SlotSet::write_route_bias` turns a tier into a
  routing constraint. With a large penalty the router only uses experts that
  are present, except for its strongest raw picks, which the router kernel
  exempts; re-electing the tier every x tokens keeps the experts a run needs
  and drops the rest. This trades a bounded loss of quality for never
  fetching cold experts and is off by default.
- **Reading from SSD** (`ares-io`; not yet connected to the tiers): experts
  are read in place from the GGUF at `offset + e * stride` with direct I/O
  through `io_uring` into buffers the caller provides (pinned staging buffers
  of a GPU backend), demand reads before read-ahead, which is cut into chunks
  and can be cancelled or promoted. Next-layer router prediction and
  profile-based read-ahead for prefill, which start reads early, are planned.
- **Several SSDs.** A model can have copies on more than one drive (a mirror
  directory per extra drive), and the parts of a split model can be on
  different drives. Every drive has its own queue depth, and each chunk of a
  mirrored file goes to the drive expected to deliver it soonest. Each drive
  is described by two measured numbers, the time of a chunk that is alone
  there and the time per chunk when several are, so a lone read goes to the
  drive with the shorter access time and a burst is shared by throughput; a
  drive that has not been measured for a second is tried again. Measured
  2026-11-10 with reads of 1.44 MiB (one expert of `qwen4exp`) from a 14 GiB
  file mirrored on the two NVMe drives of the development machine: 1.66 and
  1.36 GiB/s for each drive alone with 8 to 64 reads in flight, 2.9 to 3.0
  GiB/s for both; with 8 in flight the median read takes 3.8 ms against 6.1
  and 8.3 ms; a lone read takes 1.4 ms, the time of the faster drive. The
  copies must be identical: sizes are compared and contents sampled at open.
  The price is the disk space of the copy.
- **Prefill of a MoE model, layer by layer** (the order and the waves are
  built, running them is not). A long prompt uses
  most experts of every layer. Run in chunks of rows through the whole
  model, it fetches them again for every chunk, from the SSD or over PCIe
  from RAM. The other order fetches each expert about once per prompt: a
  group of layers is computed for all rows of the prompt before the next
  group starts, and within a layer the routed experts are taken in waves
  that fit the space for them, most used first. Every engine that streams
  experts found this to be the largest effect on long prompts (Run My AI:
  954 GiB read for a 4k prompt in batches of 16 tokens without it, about one
  read per expert with it; TheCodacus: 16 to about 300 tokens/s on a 16.7k
  prompt; none of these numbers was reproduced here).

  `ares_sched::Pass` is that order: all rows of the prompt, in chunks that
  fit a step, up to the router of the first layer with experts; then that
  layer's experts for all rows; then all rows on to the next router. A
  prompt runs as a pass when the model's experts are fetched and the prompt
  does not fit into one step (`runs_as_pass`); `pass_rows_between` sizes the
  pieces so that generating sequences, which get their ordinary steps
  between the pieces, are slowed by the chosen factor and no more.
  `ares_weights::ExpertWaves` divides the experts a layer needs into waves
  that fit the room for them: what is in host memory before what is on disk,
  the most used first, each expert in one wave. At the end of a prompt the
  tiers are elected from the prompt's own routing, its end weighing most
  (`ExpertHeat::record_counts` with `recency_weight`). On a synthetic prompt
  of 4096 tokens over 256 experts with room for 32 at a time, a layer
  fetches 224 experts once when it sees the whole prompt, and 5243 times one
  (23 times each) when the prompt comes in chunks of 64 rows.

  What is missing is everything that computes: the conductor that runs the
  pieces and keeps the hidden states of all prompt rows between them, the
  kernels that compute a wave for the rows routed to it, and a model with
  experts.
- **More from the engines that stream experts** (planned; details and
  numbers in [reference/moe-offload-survey.md](reference/moe-offload-survey.md)):
  the RAM tier as a cache in front of the SSD (admission when a read
  arrives, a victim chosen when a guessed read starts, protection of experts
  predicted for the current token); inside a layer, experts that are present
  first and a miss issued the moment the router has run, gate and up before
  down; slots per layer by what the next slot gains instead of equally, and
  a table from the last tokens to likely experts for the first MoE layers,
  which the router prediction cannot reach; gate and up of an expert
  adjacent in its slot and computed in one pass.
- **Hashed-row tables**: for models with n-gram embedding tables (28.8 GB for
  `qwen4exp`) the row ids follow from the last tokens on the host, so the
  reads are issued when the token is known and collected before the layer
  that needs them, behind a row cache (`ares-io`; not yet connected).

## Devices

`ares-cuda` binds the driver API (`libcuda.so.1`) at run time, so the CUDA
toolkit is needed to build the kernels and not to run. Kernels are compiled by
`build.rs` into two fat binaries embedded in the binary: the decode kernels
and everything else with machine code for Turing, Ampere, Ada and Blackwell
and PTX as the baseline, and the prefill kernels for Ampere and newer only.

- `devices()` and `peer_access()` describe the hardware without opening it.
- `Device` holds the primary context; `DeviceBuffer`, `PinnedBuffer` and
  `HostRegistration` (page-locking mapped model files in place) own memory.
- `Stream` and `Event` order work on and between devices.
- `Kernels` gives one typed launch per kernel.

**Decode weight kernels.** A quantized matrix times 1 to 8 rows of
activations, which is what decides decode speed. The activations are
quantized once per input to int8 with a scale per group of 32, and the kernels
compute integer dot products (`__dp4a`) straight on the weight codes, for the
eleven types of the test model (iq4_xs, q4_K, q5_K, q6_K, q3_K, iq3_s,
iq4_nl, iq3_xxs, iq2_s, iq2_xs, q8_0). Each is verified tightly against a CPU
emulation of the same arithmetic and loosely against the exact f32 product
(about 1 % rms from quantizing the activations).

Measured 2026-11-09 against the memory limit of each card (479 GB/s on the
RTX 4070, 341 GB/s on the RTX 3060, measured with a plain reading kernel), at
17408 x 5120:

- One row: 89 to 98 % of the limit on the 4070 for every type; on the 3060 97
  to 99 % for seven types and 62 to 82 % for q3_K and the iq2/iq3 types, whose
  decoding costs more instructions than that card has to spare.
- Five to eight rows are much weaker (down to 26 % on the 3060): there the
  work is arithmetic, and a tensor-core path is the fix. That is the prefill
  kernel family below, which takes over from 7 to 21 rows depending on the
  type.
- Hardware counters (`ncu`) are blocked for non-root users on the development
  machine, so what bounds a kernel was established by ablation and `nsys`.

A pass over all weight matrices of Qwen3.8-27B (13.8 GiB, 96 % of the file,
split 0.625 / 0.375 over the two cards) takes 21.7 ms with one row, 23.2 ms
with three and 38.0 ms with eight. That is a lower bound for a decode step
(no attention, recurrence, sums or sampling); the llama.cpp fork needs 28 ms
for the whole step. Uploading those weights took 1.9 s with the file in the
page cache and 9.5 to 12 s from a cold disk, through the simple path;
freeing them 4 to 8 ms. Launching a kernel costs the host under a
microsecond and nothing in the path synchronises.

**Prefill weight kernels.** A quantized matrix times many rows of
activations, on the tensor cores, for the same eleven types. There are two
paths, and `Kernels::prefill_path` picks one per device (see the cap below):

- **int8**: `mma` on int8 tiles of the weight codes, reading the same int8
  activations the decode kernels use, so one quantization of the activations
  serves both families. About 1.1 % rms from the exact f32 product, like the
  decode kernels.
- **f16**: the weights decoded to f16 tiles, f16 `mma` with f32 accumulation.
  This is the exact path: 0.03 % rms from the exact product.

`Kernels::decode_kernel_rows(ty)` is the number of rows up to which the decode
kernels are the faster family (6 to 20, by type); above it `matmul` is used.
A row gives the same bits alone and inside a batch.

Measured 2026-11-10, 17408 x 5120 q4_K, 1024 rows: 3.06 ms on the RTX 4070
and 6.91 ms on the RTX 3060 with int8 (60 and 26 TOPS), 3.68 and 8.10 ms
with f16. The other types take 1.0 to 1.7 times as long. That is a quarter
of the integer peak of the tensor cores (234 and 102 TOPS): the loop is not
bound by them but by its own instructions (about half of them scale the
integer results) and by the weight loads; below 64 rows the weights alone
bound it at about 0.3 ms per matrix.

A pass over all weight matrices of Qwen3.8-27B at the split 0.625 / 0.375
takes 108 ms with 64 rows, 281 ms with 256, 980 ms with 1024 and 1943 ms with
2048: 1054 rows per second, with the RTX 4070 idle for 29 % of the pass. At a
split of 0.7 / 0.3, which balances the compute, it is 1319 rows per second.
The llama.cpp fork prefills about 1180 tokens per second at 2048 tokens for
its whole forward pass, so these kernels are **not yet ahead of the fork**:
weights alone take about as long as the fork's whole pass. What is open:
the scaling part of the loop, weight access for small batches, what limits
the f16 kernel, kernels for Turing (the prefill binary needs compute
capability 8.0; `matmul` refuses below it), and more than 48 KB of shared
memory per block (the launcher cannot ask for it yet).

**The tensor-core cap of GeForce cards.** GeForce cards run FP16, BF16 and
TF32 tensor-core math with FP32 accumulation at half rate; professional cards
with the same silicon do not, and gpu-unlock
(https://github.com/Zagitta/gpu-unlock) removes the cap on RTX 20, 30 and 40
at boot. Kernels can work around the cap (integer tiles, or accumulating in
FP16, as the llama.cpp fork does for attention), at the price of a small loss
of exactness. Without the cap the direct path is both the fast one and the
exact one, so the workaround and its divergence are simply gone.

There is no build option for this. Each device is probed for a few
milliseconds when the kernels are loaded and the result is reported at start:

- **Tensor cores at full rate:** the direct path is used. The workaround is
  not available.
- **Capped:** the workaround is the default, because it is faster there, and
  the report says so and names the remedy. The direct path can be chosen
  instead, as the exact but slower setting.

The development machine's cards are capped, and the probe shows it exactly:
f16 `mma` with f32 accumulation runs at 58.6 TFLOPS on the RTX 4070 and 25.6
on the RTX 3060, with f16 accumulation at 117.0 and 51.2, and int8 at 234 and
102 TOPS. The probe takes about 10 ms in `Kernels::load`
(`tensor_cores_full_rate()`). For the prefill kernels the workaround is the
int8 path and the direct path is f16: `prefill_path(Precision::Fast)` gives
f16 on a full-rate device and int8 on a capped one, `Precision::Exact` always
f16.

One thing is not as the rule assumes. No uncapped card was at hand, so the
f16 kernel was timed with f16 accumulation, which runs at the rate an uncapped
card has for f32 accumulation, and it got no faster (49.1 against 49.4
TFLOPS): today's f16 kernel is not bound by the tensor cores. On an uncapped
card it would therefore still be 20 to 30 % slower than int8 from 256 rows
on. The policy is in place; the f16 kernel has to get faster before the
direct path is free.

`ares_matvec_q4_K` stays as the exact f32 reference. It is slow because each
lane reads activations 256 bytes apart, not because of its arithmetic.

One thing the device tests caught is worth knowing: a "synchronous" upload
from ordinary memory returns when the data is in the driver's staging buffer,
not when it is on the device, and a kernel on a stream does not wait for it.
`DeviceBuffer::write` therefore waits for the device, and the hot path only
uses page-locked memory on streams.

### Several devices

A model can be spread over devices in two ways, and the engine supports both,
per model and even per part of a model:

- **Tensor split.** Every layer is on every device, each holding a slice of
  the weights (rows of the FFN, heads of the attention, experts). Each device
  computes a partial result and the partials are summed, twice per layer. All
  devices work on the same token at the same time, which is what a single
  generating sequence needs, since it is bound by reading the weights. The
  cost is the sums: 96 to 130 per token.
- **Layer split.** Each device holds a contiguous range of layers and the
  activations are handed over once per boundary. A single sequence then uses
  one device at a time, but the hand-over does not make the sender wait, so
  with independent work in flight (the chunks of a prompt, the rows of
  several sequences, verify windows of different requests) the devices form a
  pipeline and all stay busy. This is the better split for prompt processing
  and for throughput with many sequences, and the only one that makes sense
  over a slow link or with very unequal cards.

Which split a model gets is decided by the split planner from measurements
(each device's memory bandwidth, the link, the VRAM ledgers), and the two can
be mixed: for example a tensor split for the layers that both cards can hold
slices of and a layer range for what only fits on one.

The VRAM cost of multi-GPU is what the exchange buffers take and nothing
else. The engine is one process with one driver context per device; there is
no framework runtime per GPU, no communication library with its own buffers,
no graph per batch shape and no KV reserved per device beyond the shared page
pool. A sum needs one receive buffer per device, a lane a ring of a few
parcels.

`Mesh::sum_f32` enqueues the sum of the devices' partial results. `Lane` is a
one-way path with a ring of slots on the receiver's side: `send_f32` writes a
parcel into the next slot and lets the sender's stream go on, `receive` makes
the receiver's stream wait for it, `release` frees the slot. The sender is
held back, on the device, only when it is a whole ring ahead. In both, the
order between devices is kept on the devices; the host does not wait. Two
routes:

- **Peer**: directly between the devices over PCIe. Consumer cards need
  patched NVIDIA kernel modules and the matching driver library, which is
  selected with `ARES_LIBCUDA=<path>/libcuda.so.1`. The mesh uses this route
  whenever every pair of devices has peer access.
- **Host**: through a page-locked buffer.

Measured on RTX 4070 (PCIe x16) + RTX 3060 (PCIe 3.0 x4), `examples/linkbench`,
2026-11-09:

| | Host route | Peer route |
| --- | ---: | ---: |
| Copy of 40 KiB, 4070 to 3060 / back | 26 / 37 us | 12 / 22 us |
| Copy of 64 MiB | 2.6 GB/s | 3.3 GB/s |
| Sum of 2 560 values across both, chained | 31 us | 8 us |
| Sum of 10 240 values | 47 us | 18 us |
| Sum of 40 960 values | 121 us | 60 us |
| Sum of 163 840 values | 420 us | 264 us |

On the peer route, sums of up to 131 072 values between two devices use the
push kernel of Piggidragon's llama.cpp fork: one kernel per device writes its
partial and a token straight into the other device's memory, waits for the
other's token in its own memory, and adds. No copy and no event is involved,
which is why a small sum costs 8 us instead of 31. (With the patched driver a
kernel can write to the peer's memory but reads from it return garbage, so
the data is pushed, never pulled.) Larger sums go to the copy engine, which is
bound by the link (the 3060's PCIe 3.0 x4) and leaves the processors free.
Every route gives every device the bitwise same f32 sum; sending fewer bits
over the link is a later, measured option.

### Vulkan

`ares-vulkan` gives the same surface (devices, buffers, ordered work,
events, kernels) on Vulkan 1.2 or newer, with the same first kernels as GLSL
compute shaders held to the same tests. It is the generic backend for every
GPU that CUDA 13 does not run on.

- Kernels take buffer addresses and sizes in push constants, exactly as the
  CUDA kernels take pointers. There are no descriptor sets, so a launch needs
  no pipeline and no descriptor write: shapes are data here too.
- A stream records commands into a command buffer and submits them in
  batches. Recording a launch costs under a microsecond; submitting every
  launch on its own would cost about ten.
- A device has one queue, so streams of one device do not run concurrently,
  and separate devices have no events between them: data between devices goes
  through the host.
- Its Q4_K kernel is not a port of the CUDA baseline but already restructured
  (a warp reads one block with consecutive weight and activation loads, a
  workgroup computes four output rows): 17408 x 5120 with one activation row
  takes 0.14 ms on the RTX 4070 (363 GB/s of weights) and 0.26 ms on the RTX
  3060 (194 GB/s). The straight port of the CUDA baseline's layout measures
  69 and 30 GB/s on Vulkan, which points at that layout, not at the backend,
  as what holds the CUDA baseline back.

## What is taken from where

| Source | What | Where |
| --- | --- | --- |
| vLLM | Paged KV, continuous batching, prefix caching by block hash | `ares-state`, `ares-sched` |
| llama.cpp | GGUF, weight and KV quantization types, n-gram speculation | `ares-gguf`, `ares-quant`, `ares-sched` |
| Piggidragon's llama.cpp fork | Quantized-KV attention kernels, tensor split with peer-to-peer sums, prefill next to decode, logits top-k on device, draft vocabulary as a view, drafter selection by measured gain | Kernels and runtime (planned); planner and draft policy (done) |
| Strata | Expert cache with RAM and SSD tiers, verify window and commit, sparse attention on index lists, GPU sampling with a counter-based generator, checkpoints at prefix boundaries, the `qwen4exp` reference | `ares-weights`, `ares-state`, runtime (planned) |
| TheCodacus' fork | Hot expert slots with a dual path, dynamic re-election, router masking, page-locked mapped weights, read-ahead | `ares-weights`, `ares-cuda::HostRegistration` |
| SGLang | Leaf-first cache eviction, cache-aware queue order, overlapped steps, a fixed GPU working set for sparse-attention KV, expert paging decided on the device | `ares-state` (eviction done); planner and runtime (planned) |
| TokenSpeed | Checkpoints of recurrent state under the chained hash, suspension of a request as a byte copy (also the test oracle), one decode path | `ares-state` (checkpoints attach to hashes; policy planned), runtime (planned) |
| ExLlamaV3 | Rotated low-bit KV with separate K and V widths, pages on RAM and disk tiers with fingerprints, a sort-free GPU sampler | Kernels and `ares-state` residency (planned) |
| TensorRT-LLM, LMDeploy, mistral.rs | 4-bit re-encoding of cold pages, sampling sharded per GPU, int8 KV with per-head ranges, transition-log rollback | Planned |
| llguidance | Grammar-constrained output by validating the GPU's top candidates on the CPU | Planned |
| AirLLM | Nothing beyond the expert tiers: reading whole layers per pass is bounded by disk throughput | - |
| Run My AI, ik_llama.cpp, Fiddler, ktransformers, BELLS, AI-DER, FATE, ProMoE | Direct reads with demand before prefetch; prefill layer by layer in waves of experts; the RAM tier as a cache with admission and protection; the order of work inside a MoE layer; slots per layer by need; fused gate and up of an expert | `ares-io` (reads done); the rest planned, see [Weights](#weights) |

Notes on each source are under [reference/](reference/).

## Status

State on 2026-11-10, on an RTX 4070 and an RTX 3060 with driver 615.78.08:

- No model runs yet. What exists is the backbone in the table under
  [Crates](#crates).
- The CUDA tests pass on both GPUs, with the stock driver library (host
  route) and with the patched one (peer routes): 11 of the device layer, 5 of
  the decode kernels, 6 of the prefill kernels, the last two also on real
  tensors of Qwen3.8-27B.
- Decode weight kernels run one row at 89 to 98 % of the memory limit of the
  RTX 4070. Prefill weight kernels work for all eleven types but reach only a
  quarter of the tensor cores' integer peak and are not yet ahead of the
  llama.cpp fork (see [Devices](#devices)).
- The KV formats are specified, pinned by golden files and measured on the K
  and V of real models; the attention kernels that read them do not exist yet.
- The server crates and the quantization reference were verified through
  their own test suites.
- Real model files are test data and stay outside the repository. The reader
  opens Qwen3.8-27B (14.3 GiB, architecture `qwen35`) in 22 ms and the split
  Qwen3.8-Flash-Next (63.5 GiB in two parts, `qwen4exp`, 1 224 tensors) in
  28 ms; the latter's metadata matches
  [reference/strata-notes.md](reference/strata-notes.md).

## Next steps

1. `ares-runtime` beyond the skeleton: wider steps through the prefill
   kernels (and more than eight rows per decode launch), sums and hand-overs
   that a device thread can queue for its own half without a rendezvous on
   the host, overlapped steps, decisions (`Engine::decide`), the prefix
   cache, sampling on the device, and the Vulkan backend.
2. Kernels that every current model needs: faster prefill weight kernels (the
   open points are listed under [Devices](#devices), a Turing path among
   them), attention over quantized K/V read through page tables (rotated
   formats, separate K and V widths, the mean of a tile once per tile, exact
   cells in `f16` pages), the kernels that write K and V in those formats, a
   sort-free GPU sampler, and the Vulkan mirrors of the weight kernels.
3. Weight placement and the split planner; connecting `ares-io` and
   `ares-store` to the devices.
4. Tokenizer and chat templates; structured output with llguidance.
   Multimodal input (the projector files that ship next to the models) comes
   with the first model that has one.
5. The first model. Architecture-specific work waits for the Qwen4 release;
   until then `qwen35` (Qwen3.8-27B, the current production model) and
   `qwen4exp` are the test targets.
6. Residency of pages (a fixed GPU working set over K/V in pinned RAM),
   writing pages to and restoring them from `ares-store` in the
   background, the checkpoint policy for recurrent
   state, and suspension of a request as a byte copy, which doubles as the
   test oracle for all state handling.

Every kernel is written for both backends and verified against the same CPU
reference.

## Host memory

There is one setting for RAM: the most the whole engine may use. Inside that
limit the users of memory make room for each other; nobody is given a fixed
share.

| User | Kind | When memory is short |
| --- | --- | --- |
| K/V pages of running sequences that do not fit in VRAM (KV offload) | Required | Never gives way; everything below shrinks for it |
| RAM tier of MoE experts | Elastic, high | Shrinks to the working set the running requests need |
| Row caches and staging | Elastic, middle | Shrink to a small floor |
| Prompt cache in RAM (pages of finished conversations) | Elastic, low | Goes first; its pages are on SSD as well |

A broker in `ares-core` keeps the books: a user asks to grow, and when the
memory is not free the broker names who must give back how much, lowest
priority first and never below a floor. KV offload of a running request can
therefore only fail when the limit is too small for the requests themselves,
not because a cache holds the memory. VRAM is handled the same way in spirit
but is planned ahead (ledgers), because moving things there is costly.

Nothing that costs speed happens silently:

- **At load**, the engine states what the configured context means: how many
  tokens of state fit in VRAM, and, if the context is larger than that, that
  K/V will spill to RAM, from which point, and how much RAM that can take at
  most. A context that fits entirely in VRAM never offloads.
- **When K/V first spills** for a running request, that is reported, with the
  amount.
- **When the expert tier in RAM has to shrink**, that is a warning, because
  generation gets slower: experts that were in RAM now come from the SSD.
- **When the prompt cache in RAM shrinks**, that is routine and only counted;
  its pages are on SSD.

The same facts are available as counters to whoever monitors the server.

## Loading and unloading

Switching models must take seconds, bound by the hardware and not by the
engine. What that needs:

- **Nothing to start.** The server process, the device contexts and the
  kernels stay loaded while models come and go. The CUDA kernels are
  precompiled machine code; the Vulkan pipelines want a pipeline cache on
  disk.
- **Weights at the speed of the slowest link.** Tensors are read with direct
  I/O (or taken from the page cache when the file is already there) into
  page-locked staging and copied to all devices at once, each device on its
  own stream. The limits are then the SSD and each card's PCIe link, about
  3.3 GB/s for the RTX 3060 on its x4 slot.
- **State is reserved, not prepared.** The page pool is a few large device
  allocations. Pages are never cleared, because a page table decides what is
  valid, and there are no warm-up passes, because there are no graphs to
  build. Page-locked host memory, which is slow to lock, is taken in chunks
  in the background and only as far as offload needs it.
- **Unloading frees and returns.** Nothing is written at unload: state on
  disk is written as it is produced.

Load and unload times are measured with the first model; there are no numbers
yet.

## Server

The server is used like llama.cpp's: flags, and a router mode driven by a
preset file (`models.ini`, one section per model) that loads and unloads
models on request and keeps as many loaded as fit. Today the binary takes
`--host`, `--port` and `--model`; presets and the router come with the
runtime.

The set of flags is the engine's own and starts from nothing. A flag exists
only for something that is worth setting: a real choice (KV formats, context
budget, the prompt share next to generation, expert pruning) or a fact about
the machine. There are no flags without effect and none that switch on what
is always on; an unknown flag or preset key is an error that names it.

## Open questions

- **Indexer cut-off.** Strata's two engines differ in which cells the sparse
  attention keeps at the cut-off. This must be settled against the real
  Qwen4.
- **Pascal cards (GTX 10xx).** Settled: the CUDA backend covers what a CUDA
  13 toolkit builds for, which starts at Turing (GTX 16, RTX 20). Older cards
  go through the Vulkan backend. (A driver of the 13 level does run on
  Pascal, but only with kernels built by a 12.x toolkit, which is outside the
  engine's scope.)
