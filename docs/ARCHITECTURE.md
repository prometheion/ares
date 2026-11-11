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

- **Two test cases stand for Qwen4.** Until Qwen4 is released the engine is
  built and measured on its two closest relatives, one for each way weights
  are held: Qwen3.8-27B (`qwen35`), dense and entirely on the GPUs, on both
  cards; and Qwen3.8-Flash-Next (`qwen4exp`), a mixture of experts with the
  experts in tiers and streamed from SSD. A part of the engine is done when
  it has run in both. On the development machine (two cards with 12 GB, 31
  GB of RAM, a 63.5 GiB file) Flash-Next can only run with streaming, so the
  case without it is not measured there.

- **Nothing is tuned to one machine.** The CUDA backend serves every GPU
  that CUDA 13 supports, the Vulkan backend every other. Which kernel, block
  shape or row-count boundary a device gets may depend on its class (its
  compute capability on CUDA: "Ampere and Ada use this"; what the device
  reports on Vulkan) or on something measured on the device at load (the
  tensor cores' rate with `f32` against `f16` accumulation, memory
  bandwidth, the link to another device), never on a device name, a vendor
  string, an ordinal, or a constant that is only the result of a measurement
  on the development machine. The numbers in this document come from an RTX
  4070 and an RTX 3060; a class that could not be run there is compiled and
  marked untested.

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

- **Every GGUF weight type.** A model file may carry any of the 28 types
  that hold weights (the float types, the legacy and K quantizations, the
  i-quants, the ternary and 1-bit types, the 4-bit float types), and each has
  its kernels on both backends. Both backends have a decode kernel for every
  type. The kernels for many rows cover the eleven types of Qwen3.8-27B, the
  first reference file, and are being extended.

- **No engine is the model for this one.** llama.cpp, vLLM, SGLang, Strata
  and the others are references, each for what it does best, and a design is
  taken from whichever has the better one. Two things are fixed points and
  not up for trade: GGUF files are the input, and VRAM is not wasted. Weights
  stay in VRAM at their quantized size (rearranged for the kernels at load
  where that helps, never expanded), K and V are quantized and paged from
  one pool, and nothing is preallocated per request. That is what llama.cpp
  gets right and the serving engines do not; how fast the rest is done owes
  it nothing.

- **Server only.** One binary, `ares`. No command line tools, no GUI.
- **Linux only.** That allows direct I/O, `io_uring` and huge pages without
  a portability layer.
- **Newest models only.** An architecture is supported while it is current
  and removed when it is superseded. There is no generic operator graph to
  keep old models running.
- **CUDA and Vulkan.** Everything CUDA 13 runs on is supported and tuned for;
  every other GPU goes through the generic Vulkan backend. Both device layers
  exist from the start and are held to the same tests; everything above them
  is written once. Every kernel family exists on both as far as Vulkan
  allows, computing on the quantized data itself on both; transfers between
  devices use whatever direct path a system has, on either backend.
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
   counter-based generator keyed by (seed, position), so the randomness of a
   pick does not depend on what else is in the step, and verifying a draft
   compares it with the pick the model makes for that position.
7. **One thread decides.** All bookkeeping (pages, plans, expert tiers) is
   plain data owned by the step loop, with no locks. Other threads do work
   that waits: one per device to enqueue, a pool for file I/O, the HTTP
   runtime. This is the answer to "actually good multithreading": threads
   never contend for the engine's state because only one of them owns it.
8. **Quality is held, agreement is not.** The engine does not reproduce
   llama.cpp's numbers, and its own kernels need not agree with each other
   to the last bit: a step of three rows and a step of three thousand use
   different kernels, a capped GeForce card accumulates differently from an
   uncapped one, and where a different arithmetic is faster it is taken.
   What every kernel is held to is its error against the exact arithmetic of
   the model, measured and written down next to its speed. Exact arithmetic
   is the default wherever it costs nothing: at the few rows where a kernel
   waits for memory anyway, activations stay `f16` (Marlin) or the rounding
   error of 8-bit activations is removed by a second product (ExLlamaV3).
   An error is accepted only where it buys speed, and then at most what the
   engines this one replaces have (about 1 % rms per product from 8-bit
   activations in llama.cpp's quantized kernels). That number is a ceiling,
   not a target. Output is therefore not guaranteed to be identical between
   runs that batch differently, as it is not in the engines this one learns
   from. The CPU references in `ares-quant` and `ares-model` are the exact
   arithmetic that kernels are compared with.

## Crates

| Crate | What it holds | State |
| --- | --- | --- |
| `ares-core` | Tensor element types and block geometry, byte units, memory ledgers | Done, unit tested |
| `ares-gguf` | Zero-copy GGUF reader (version 3, split files, validation at open), a small writer for tests | Done; tested against synthetic, truncated and corrupted files and real models |
| `ares-quant` | CPU reference decoder for every GGUF type; the specification and reference codec of the engine's own KV formats (`f16`, `q8` to `q4`, tiles with a mean, the policy of exact cells) | Done; bit-identical to ggml's C code on fixtures generated from it; the KV formats are pinned by golden files and were measured on the K and V of real models |
| `ares-cuda` | Driver API bound at run time, devices, memory, streams, events, modules, decode weight kernels for all 28 weight types, prefill weight kernels for eleven, the layer kernels of `qwen35`, attention over the stored K/V formats, the mesh that sums across devices over peer access or the host | Device layer, decode, layer and attention kernels done; prefill kernels verified, their pass over the weights 1.3 times as fast as the fork's whole prefill; tested on both GPUs, with and without peer access |
| `ares-vulkan` | The same device-level surface on Vulkan; decode and prefill weight kernels for all 28 weight types as compute shaders, the prefill ones on cooperative matrices; sums between devices | Device layer and weight kernels done (23 device, 11 kernel and 9 mesh tests on both GPUs); no layer or attention kernels and no model yet |
| `ares-state` | Page pool, page tables with copy on write, the prefix index (pages published as they fill, tails, announcements, priority classes), the checkpoint policy for recurrent state, page layouts | Done as bookkeeping; randomised model checks; the conductor's prefix cache uses them (see `ares-runtime`) |
| `ares-store` | State on disk: pages and checkpoints by content hash in segment files, per fingerprint, with a size limit | Done as a store of opaque bytes, tested against restarts, the limit and damaged files; not yet connected to devices |
| `ares-io` | Reads of weights straight from the model file: io_uring with direct I/O, demand before prefetch, cancel and promote, copies of a file on several drives with the reads spread over them, a map from experts to file ranges, row reads behind a cache | Done, tested against real files and on two drives; 1.4 to 1.7 GiB/s of random expert-sized reads per SSD of the development machine and 2.9 to 3.0 GiB/s over both; the expert store and the n-gram rows of `qwen4exp` read through it |
| `ares-sched` | Step planner (generation, decision rows and prompts in one step), admission (pages claimed per step, a reservation that corrects itself, a queue instead of refusals), the order of a pass for long prompts of a model with experts, step time model with a term for the context, draft length choice, the lookup drafter on a suffix automaton shared by all sequences | Done; unit and model tests; the conductor uses the planner, admission (without spill), the lookup drafter and the step cost |
| `ares-weights` | Expert heat, tiers that are elected and fed by reads (admission, protection, guesses bound late, slots given back), slots per layer by need, the waves of experts of a pass, a token table for the first layers, the lookahead distance, routing bias for pruning | Done, unit tested; the expert store uses the slot sets, the heat and the split of slots; guesses, the token table and pruning are not used by a running model yet |
| `ares-api` | Request, event, decision and error types; the `Engine` trait between HTTP and the runtime (`submit` for generation, `decide` for decisions, `stats`) | Done, unit tested |
| `ares-server` | The `ares` binary: loads a model on the devices and serves it over an OpenAI-compatible HTTP API, with the plan of devices, split, shares and context made from what the devices report | Serves `qwen35` models (chat and plain completions, streamed and not, timings, metrics, staged shutdown); 96 tests without a GPU and 9 that start the real binary; the prefix cache is on, with `--ram` for the engine's RAM; no router mode, no multimodal input, no tool calls, `/v1/systemone` answers 501 |
| `ares-runtime` | The conductor thread and one thread per device, the step loop that connects plan, state and kernels, admission, speculation with the lookup drafter and with a model's own drafts, the `Model` trait a model implements, a synthetic model that drives the real kernels, and the expert store (VRAM slots, RAM tier, reads, the CPU computing what is not on a GPU) | Drives `qwen35` on one device, in tensor split and in layer split, with picks made on the device; implements `Engine::submit` (not `decide`); the prefix cache is in the conductor, checked against the host executor as an oracle and carried out by `qwen35` on the devices (checkpoints in page-locked RAM within one RAM limit, `PinnedRam`), and there is no spill of pages to RAM (a request that gives way stays in the pool's cache); the expert store is driven by `qwen4exp`'s own session, not by the conductor |
| `ares-model` | Model architectures: configuration and weight table from a GGUF, a CPU reference of the forward pass with explicit state (window and commit) that GPU implementations are compared with block by block, and the implementation on the devices | `qwen35` (Qwen3.5-0.8B, Qwen3.8-27B) runs through the runtime and the server, with the model's draft block. `qwen4exp` (Qwen3.8-Flash-Next) runs on one or two cards through a session of its own, without drafts. See [Several devices](#several-devices) |
| `ares-tokenizer` | Text to token ids and back for byte-level BPE (`gpt2` with the `qwen35` pre-tokenizer) as llama.cpp does it, malformed UTF-8 included; control tokens in text only on request; encoding on several threads for long prompts; a streaming decoder that never cuts a character; the chat templates of the model file (Jinja, a dialect of its own that errors on what it does not support) with tools and the generation prompt | Done; token ids equal llama.cpp's on millions of generated and real texts for the three Qwen test models, chat prompts equal on thousands of conversations; about 20 times llama.cpp's encode speed on one core |

Both test architectures are specified under [reference/](reference/):
[qwen35](reference/qwen35-architecture.md) and
[qwen4exp](reference/qwen4exp-architecture.md).

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
   rows`). `StepTimeModel` now also fits a cost per attended cell, because
   attention grows with the context and a prompt chunk at 100 000 tokens
   costs more than one at 1 000 (`prefill_rows_at`); the conductor still
   calls the older form.
2. **Step record.** Everything that varies per step (row to sequence and
   position, page-table changes, draft tokens) is written to one pinned
   buffer and uploaded with one copy per device. Kernels read it there, so
   launches carry only pointers and sizes. A prompt that starts from the
   prefix cache adds replay segments, copies of recurrent state (saved at a
   position, restored into a slot) and copies of pages (the copy on write of
   a shared page); the record documents the order they run in.
3. **Window.** The model enqueues its kernels on every device's stream.
   Position-addressable state is appended; recurrent layers read their base
   state and keep their inputs.
4. **Sample.** The sampler's kernels pick the token of every wanted row on
   the device (`ares_cuda::SamplerKernels`): the most probable token, or
   temperature, top-k, top-p and min-p as exact thresholds on `p / p_max`
   (a radix descent over integer histograms) and a pick by Gumbel noise of
   (seed, position, token) from a counter-based generator. Its arithmetic is
   repeated bit for bit on the host (`sample_reference`), which is what its
   tests compare with. Under a tensor split every device holds the logits of
   a slice of the vocabulary and the devices add up small partial results
   between the sampler's steps. Only the picks and the eight best candidates
   of each row (80 bytes per row) go to the host.
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
A model whose device side carries out the step record's copies and replay
segments says so (`Model::prefix_cache`, with the bytes of one checkpoint on
each device) and takes part in the prefix cache; one that does not has none.
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
  takes the cached ones. This one mechanism replaces per-slot prompt caches,
  slot similarity matching and a separate conversation cache, and it is the
  key under which pages will be written to disk.
  - A page enters the index when it fills, not when its sequence ends
    (`publish`, called from the commit with the number of settled cells), so
    two requests that start with the same prompt at the same time share from
    the first full page. A sequence that finds its page cached already takes
    the cached one and frees its own.
  - A sequence announces the pages its prompt will fill. A second prompt
    whose first missing page is announced is told so and is held back until
    the page is published or the announcement withdrawn, instead of computing
    the same rows next to the first. Only running sequences announce, so two
    prompts cannot wait for each other.
  - Tails: the partly used last page of a sequence is published under the
    hash of the page before it, with its tokens; at most 8 per page, least
    recently used first. A prompt that continues a tail starts at the real
    end of the cached text and copies the tail's page before its first write,
    which the page table does for any shared partly used page. A prompt may
    take any leading part of a tail, so a tail may hold only cells whose
    stored form depends on no later cell: any row of `f16` and `q8`, and with
    `q7` to `q4` exact cells only. A page is one tile there, and a tile is
    converted only when it is full and has left the newest window, so the
    partly used last page is always exact and is published at the commit
    that ends in it. A full page that was converted cannot serve a part of
    itself: its rows hang on the tile's mean and cannot be continued.
  - The last token of a prompt is always computed, because its logits are
    needed: the prompt is looked up without it.
  - Eviction takes pages nobody holds from the ends of chains, a lower
    priority class before a higher one whatever their age, and within a
    class the least recently used. `evictable` counts what can go.
- Checkpoints: state that is not addressable by position (recurrent layers)
  can only be resumed where a copy of it was kept, and a copy is large (75
  MiB as `f16` for Qwen3.8-27B), so `CheckpointPolicy` decides where copies
  go and which are dropped. The conductor keeps them as places within a budget
  of bytes (`RuntimeConfig::checkpoint_budget`); the copies themselves are the
  device's part (`qwen35` does them, see below). The state is `f32` today: 150
  MiB for Qwen3.8-27B over both cards (94 and 56 MiB at shares of 0.625 /
  0.375), plus 20 KiB for the draft block's carry.
  - Where: only at page boundaries and at the end of a tail. Inside a prompt
    at least 32 768 cells apart, while generating at least 2 048, measured
    from the last checkpoint on the sequence's own path; always at the
    prompt's anchor, the last page boundary at least one row before the
    prompt's end, where the next turn of a conversation resumes; and the live
    state at the end of a request becomes the checkpoint of its tail. The
    commit takes a checkpoint when it crosses one of these positions.
  - What a prompt replays: K and V are taken from every cached page and from
    a tail, however far the last checkpoint is behind them. The rows between
    that checkpoint and the end of the cached cells run again to rebuild the
    recurrent state; they read K and V and write none. That keeps the pages
    shared and saves the writes (and the scale search of the low-bit
    formats), not the rows' time: `Resume` names the checkpoint, the replayed
    rows and the computed rows.
  - Which go when the byte budget is full: first the checkpoints that later
    requests have left behind (more than a set number of later requests have
    checkpoints below them, on one branch; a checkpoint where texts branch
    is never such), then the one with the least distance to the next
    checkpoint toward the start of the text, which is the rows a resume
    replays more without it, then the least recently used. A checkpoint goes
    without its K/V page; a page evicted from the end of a chain takes its
    checkpoint along.
- Admission (`ares_sched::Admission`): nothing is reserved for a request's
  longest possible length. Pages are claimed per step. A reservation covers
  the rest of the running prompts and the output they are expected to
  produce: `min(tokens left, 4096)` times a ratio that every finished request
  moves toward the share of its allowance it really used and that a shortage
  raises. A new request is charged for the pages of its prompt after the
  prefix match and its clipped allowance in full, and waits when that does
  not fit; nothing is refused for load, and a request that is alone always
  starts. Waiting requests are tried with the fewest uncached rows first; one
  that was passed over too often goes first and holds the others back. A
  step's claims are settled against a copy of the counters: cached pages are
  evicted first, draft rows are cut before anyone gives way, then the
  sequence with the fewest generated tokens is spilled, and victims that
  turn out not to be needed stay. The object decides and reports; the
  conductor carries it out.
- In the conductor (`crates/runtime/src/conductor.rs`, `cache.rs`), for a
  model that takes part (`Model::prefix_cache`):
  - A request is looked up when it arrives and again when it starts. It
    shares the cached pages and a tail, restores the state from the last
    checkpoint on its path, replays the rows from there to the end of the
    cached cells (a segment of its own with the replay flag; they read the
    cells and write none) and computes the rest. A prompt whose first missing
    page a sequence that already runs has announced is put back in line,
    held, until the page is published or the announcement withdrawn; a
    request that arrives in the same pass as the one that computes the page
    is held after it started.
  - A page enters the index at the commit that fills it. A prefill step ends
    at the next position where the policy wants a checkpoint, and so does the
    window of a generating sequence (its drafts are cut), so the state is
    exactly there when the save is queued; the save runs with the next
    record. Replayed rows and rows that write are separate steps.
  - When a request ends, is cancelled or gives way, its partly used last page
    is published as a tail and its live state becomes that tail's checkpoint
    (or the last full page's, on a page boundary). Tails are not published
    while a request runs: the index would hold a reference that makes it copy
    its last page at every step. A request that gives way comes back as a
    prefix hit.
  - When a step's claims need room, cached pages go from the ends of chains
    after the requests that start have taken what they share; checkpoints go
    when their budget is full and with their page.
  - Per request the runtime reports the prompt tokens that came from the
    cache (`Usage::cached_prompt_tokens`) and the rows it replayed and
    computed (`Snapshot::requests`); the counters add them up.
  - The host executor plays a device as an oracle: it keeps the cells of every
    page and the state of every slot as hashes of all tokens in order, and a
    row that reads a wrong one picks another token.
- On the devices (`qwen35`, `crates/model/src/qwen35/device/part/resume.rs`),
  in the order the record's documentation gives (the commit of the step
  before, the saves, the page copies, the restores, the rows):
  - **A checkpoint** of a device part is its slot of the recurrent state (the
    convolution windows and delta-rule matrices of its blocks) and, on the
    device that holds the draft block, one row of the block's carry. It lies
    in **page-locked host memory**, not in VRAM: a save is a copy from the
    device and a restore a copy to it, queued on the step's stream, so the
    stream orders them against the kernels. The places are locked in chunks
    (at least 32 MiB) when one is first used, within the engine's RAM limit
    (see "Host memory").
  - **A page copy** copies the whole entry of the page, for every attention
    block and for the draft block's own array (the cells of an entry lie in
    planes per head, and cells past those the sequence has are written before
    they are read).
  - **A replay** is a prompt step whose rows append nothing: the K/V append of
    every attention block and of the draft block's follow of the model runs
    per run of rows that write, and the rows of a replay are left out. All
    else runs as for a prompt, which advances the recurrent state and leaves
    the carry at the last row.
  - **The draft block's state per sequence** is its carry (the final hidden
    state of the sequence's last row: the first row of a segment reads it for
    the block's keys and values at its position, and the next chain of drafts
    starts from it) and its keys and values, which lie in the pool's pages
    (one more array of entries), written by the follow of every row of every
    step. Both are therefore in a checkpoint and in a page copy, and a
    replay must not write the cells.
  - **Measured** 2026-11-11 with `qwen35_run --conversation` (the second turn
    of a conversation whose history is the given number of tokens, with the
    chat template's thinking off; a conversation of the same size that the
    engine has never seen as the cold control), Qwen3.8-27B, tensor split
    0.625 / 0.375, 150 MiB of checkpoint over both cards:

    | History | Prompt of the second turn | Cold, first token after | From the cache, first token after | Cached / replayed / computed rows |
    | --- | --- | --- | --- | --- |
    | 8k | 8278 tokens | 7070 and 7042 ms (1167 tokens/s) | 138 and 111 ms | 8255 / 0 / 23 |
    | 24k | 24 662 tokens | 25 239 and 25 258 ms (975 tokens/s) | 150 and 122 ms | 24 639 / 0 / 23 |

    The checkpoint of the first turn's end is exactly where the second turn
    starts, so no row is replayed. A save of one checkpoint takes 3.7 ms on the
    RTX 4070 (93.5 MiB, 25 GB/s) and 17.6 to 17.9 ms on the RTX 3060 (56.1
    MiB, PCIe 3.0 x4, 3.2 GB/s); a restore 3.9 and 17.7 ms; a copy of a page
    0.03 to 0.05 ms. The cards work in step, so a step with a save or a restore
    takes the 3060's 17.7 ms longer, once per checkpoint: a request that ends
    leaves one save (and the anchor of a long prompt another), a request that
    starts from a checkpoint one restore. Against the seconds a start from
    nothing costs this is no problem, and a staging area in VRAM
    (which the copies could use to overlap with the step) is not built. The
    first use of a place waits for its lock: 21 to 22.5 ms for the RTX 4070's
    93.5 MiB and 11.8 ms for the RTX 3060's 56.1 MiB, once per place (the
    first two places of a run cost 52 and 49 ms of waiting on the host). RAM
    locked: 748 MiB after the first conversation (5 places) and 1048 MiB after
    the second (7 places). The same second turn of 8192 tokens of the 0.8B on
    each card alone (8274 prompt tokens, 8220 cached, 28 replayed, 54
    computed): RTX 4070 cold 361 and 357 ms, from the cache 21 and 17 ms, save
    0.77 ms, restore 0.82 ms (19.3 MiB); RTX 3060 cold 810 and 806 ms, from
    the cache 41 and 37 ms, save 6.15 ms, restore 6.11 ms. Generation is as
    fast as before: with the same flags on the same day a prompt of 2048
    tokens reads at 1228 to 1241 tokens per second before and 1227 to 1241
    after, a decode step without drafts at 1000 tokens takes 26.0 and 25.9 ms
    before and 26.0 and 26.1 ms after (the 24.8 ms of an earlier day is not
    this machine's today), and with the model's own drafts (3 per window, the
    fork's draft vocabulary, 300 tokens, most probable token) German, English
    and code give 79.5 / 76.2, 73.3 / 76.5 and 75.9 / 74.4 tokens per second
    before and 77.8 / 77.6, 76.5 / 80.3 and 78.3 / 81.5 after (78.4, 75.2 and
    80.8 with 16 checkpoints kept), a step of 34.2 to 34.6 ms throughout.
  - **Rounding.** A row's result depends on the size of the step that
    computes it, also with exact products (measured on the 0.8B, one prompt of
    480 tokens in steps of several sizes against steps of 128 rows: steps of
    32 and 64 rows give the same logits bit for bit; steps of 1, 7 and 100
    rows give logits that differ by 2.0e-3, 1.6e-3 and 1.2e-3 of the logit at
    most, with 2, 1 and 0 of the 480 top tokens different). The cache changes
    the steps that compute a prompt (rows replayed from a checkpoint, the last
    row alone), so a run from the cache is not bit for bit the run without it:
    the logits of 40 generated tokens differ by 0.7e-3 to 0.9e-3 of the logit
    at most (one card, each card alone, tensor split, layer split), the tokens
    are the same. Texts are the same in every test that copies a text, whose
    words follow with a wide margin; a text with near ties (a list whose order
    the model is unsure of) can part at one, as can two requests that decode
    together instead of alone.
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

**The reference for attention.** `ares_quant::kvhead` is what the attention
kernels of both backends are held to. `KvHead` is the stored sequence of one
KV head under a policy, byte for byte as a kernel must write it: new cells as
`f16`, tiles converted from the stored `f16` values when the policy says so
(on the real data that costs nothing measurable against converting from the
originals). Draft cells are appended as pending, which converts nothing, so
rejected ones can always be cut and accepted ones are settled. Attention is
defined the way FlashInfer builds it: the attention of a query over a range of
cells is a state (the log of the summed exponentials of its scores, and the
weighted sum of V), and the states of disjoint ranges merge into the state of
their union, in any order. Dense attention is one range, block-sparse
attention the selected blocks. This is what lets a kernel compute pages
independently, lets K and V of a sequence lie on two devices with only states
crossing, and lets a prefix shared by many rows of a step be attended to
once. A plain f32 implementation agrees with the reference to 2e-6; the
tolerance for kernels is ten times that.

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

The engine has every kind of drafter that llama.cpp and Piggidragon's fork
have, with the fork's selection on top:

| Part | What it is | State |
| --- | --- | --- |
| Lookup drafter | Continues the longest earlier occurrence of the current text, in this request or another: one suffix automaton over a corpus that all sequences share (`LookupCorpus`, `LookupDrafter`; after TensorRT-LLM, SGLang and vLLM's suffix decoding), with a confidence per proposed token and a bound on memory. It replaces the n-gram lookup per sequence; llama.cpp's map and cache variants are not built | Done and measured on text; the runtime still uses `NgramDrafter` |
| MTP | The model's own draft head or block | Done for `qwen35` on one device and under a tensor split: the block runs on the device with the embedding table, its chain and its picks stay on the device (`device/mtp.rs`); not under a layer split |
| DFlash, DSpark, DFlash2 | A small draft model that reads the target's hidden states at a few layers and proposes a block of tokens at once | With the runtime: it needs taps on the target's layers and a second model on the same devices |
| Drafter selection | Per sequence and step, the drafter with the highest expected tokens per millisecond, from measured acceptance and cost; drafters that were not chosen keep learning from what was accepted | `best_draft_len` and `Acceptance` done; the selector object that holds them per drafter is next |
| Adaptive draft length | The length that maximises the same number | `best_draft_len` done |
| Verification | Windows of any length for any number of sequences in one step, then commit | Planner done; kernels with the model |

All drafters produce the same thing, a list of proposed tokens with an
optional confidence per token, so selection and verification do not depend on
where a draft comes from. A lookup draft is written into the step record by
the host. The draft of a head or a draft model is to stay on the device, in a
table the verify window reads (SGLang), and reach the host together with the
picks; carrying it through the host would put a copy out and back between two
kernels of the same device at every step. A draft is cut at the first token
the grammar refuses before it is verified (vLLM).

**The lookup drafter.** An automaton and not a trie of runs, because the
longest match of a sequence's end is then not searched for at all: it is
where the suffix links of the sequence's own state lead, an insertion is
constant work, and a token costs two states and three transitions whatever
the match length, where a trie of runs up to `d` tokens costs `d` nodes and
`d` lookups per token and never matches more than `d`. An automaton cannot
forget, so the corpus is two of them: when the young one is full the old one
is dropped and the young one becomes the old. Text is found for one to two
automata's worth of tokens (64 MiB hold 260 000 to 520 000). Each state
keeps where its most recent occurrence went on and how many of its
occurrences went on, and the confidence of a proposed token is read from a
table by the length of the matched context and by how its earlier
occurrences went on (once; always this way; mostly; seldom). The table
starts from measured acceptance and follows what verification reports.

Measured 2026-11-10 by replaying this repository's own text (153 000 tokens
of documentation, 629 000 of Rust, 598 of German; vocabulary of Qwen3.5), a
proposal counted as accepted as far as it equals the text. No model ran, and
the tokens per second are computed from the step cost of Qwen3.8-27B on the
development machine (33 ms and 1.1 ms per row, 29.3 tokens/s without
drafts), drafts of at most 16 tokens:

| Text and replay | n-gram of 3, per request | Lookup, shared corpus, length by confidence |
| --- | --- | --- |
| English, written from nothing | 0.19 proposals per token, 0.98 accepted per proposal, 2.78 wasted rows per token, 32.3 tokens/s | 0.32, 0.72, 0.79, 36.7 tokens/s |
| English, the prompt written again with edits | 0.14, 5.91, 1.37, 121 tokens/s | 0.08, 11.7, 0.24, 223 tokens/s |
| English, a file an earlier request wrote | 32.0 tokens/s (it knows one request) | 223 tokens/s |
| Rust, written from nothing | 0.16, 1.65, 2.23, 36.1 tokens/s | 0.41, 1.07, 1.41, 46.9 tokens/s |
| Rust, the prompt written again with edits | 0.11, 7.72, 0.89, 147 tokens/s | 0.08, 12.1, 0.25, 223 tokens/s |
| German, the prompt written again with edits | 0.06, 15.1, 0.03, 240 tokens/s | 0.07, 13.2, 0.06, 238 tokens/s |

The gain in copied text comes from taking the right place: the last three
tokens often end several places, and the n-gram lookup continues the most
recent one. In text written from nothing the automaton proposes more often
and is wrong more often per proposal; it pays only because the confidence
cuts those drafts short (with every draft at full length it gives 30.5
tokens/s on English, less than the n-gram lookup). The German sample is two
short files and says nothing about German: there the confidence, which was
measured on English and code, cuts drafts that would have been accepted (238
against 256 tokens/s at full length). On the host a token costs about 0.2
microseconds to read and a draft 0.4, eight and thirty times the n-gram
lookup's cost; a long prompt is therefore read into the corpus as its rows
are committed, not at once.

**The model's own drafts** (`qwen35`, the next-token-prediction block of the
27B). A step with them is still enqueued whole: the block's chain runs first,
on the device that holds the embedding table, each link's pick made by the
sampler with the settings and the noise of the row that will verify it and
written straight into the token array of the step record on the device; then
the verify window, whose embedding reads those tokens; then, for every row of
the step, the block's own keys and values from the hidden states the model
just computed. The host learns the drafts with the picks, counts the accepted
ones and puts the commit into the next record. The conductor chooses per
sequence and step between a lookup draft and the model's own by expected
tokens per millisecond (`best_draft_len`), with the cost of a link measured on
the device.

Measured 2026-11-10 on Qwen3.8-27B (tensor split 0.625 / 0.375, RTX 4070 and
RTX 3060, at most three drafts, the fork's draft vocabulary of 66 343 tokens,
300 tokens per request through the chat template with the fork's
`reasoning_effort`): 79, 76 and 79 tokens/s for German prose, English prose
and code with the most probable token, 76, 72 and 83 sampled (temperature 1,
top-k 20, top-p 0.95), against 40.5 without drafts and 62 to 67 for the fork
with its production flags on the same questions. A link of the chain costs
1.46 ms (it reads 335 MiB of the block and 278 MiB of the head), a step 34 ms
for 2.5 to 2.9 tokens; the first draft is accepted in 74 to 83 % of the
steps. The block takes 335 MiB of weights and 2.1 MiB of K and V per 1000
tokens on its device, and nothing for its head, which is rows of the output
head. With exact products the text with drafts equals the text without, for
the most probable token and for sampled tokens with the same seed
(`tests/qwen35_mtp_runtime.rs`); with the fast kernels a window of several
rows rounds differently from single rows and the texts part after 150 to 650
characters (the margins of the picks where they part were not measured).

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
- **More from the engines that stream experts** (details and the sources'
  numbers in [reference/moe-offload-survey.md](reference/moe-offload-survey.md)).
  Built as bookkeeping in `ares-weights`, with no runtime using it yet; the
  numbers are from a synthetic workload with the properties the sources
  report (locality in time, topics that change, early layers that route by
  the token), not from a model:
  - *The RAM tier as a cache in front of the SSD.* `SlotSet` takes reads as
    well as elections: a demand read gets a slot at once, a guessed read
    (`GuessQueue`) gets one only when it is started, experts routed or
    predicted for the current token are protected, and slots can be given
    back to the memory broker and taken again. Against a tier that is only
    re-elected, the model starts 34 % fewer reads per token and waits 20 %
    less with 4 reads in flight (58 % and 44 % with 8). Nearly all of that
    is reading guesses at all; protection removes the evictions of experts in
    use, binding the victim late is worth about 1 %, and admission on its own
    gains nothing over re-election.
  - *Slots per layer by need* (`split_slots`): a tier's bytes go to the
    layers by what the next slot gains, with a margin and a cap on moves so
    that slots do not go back and forth. It raises the share served from the
    tier from 0.70 to 0.72 at 32 slots per layer and does nothing at 16; at
    small budgets it takes slots from the early layers, which is the opposite
    of what the sources recommend for them, so a weight for how badly a layer
    is predicted is still to be added.
  - *A table from the last tokens to likely experts* (`TokenTable`) for the
    first MoE layers, which the router prediction cannot reach: 104 bytes per
    entry for 4 layers of 8 experts, built from recorded routing. On the
    synthetic workload the first layers are served 0.69 instead of 0.39 with
    527 entries. It needs a recording tool and must be measured on a real
    model before it is kept.
  - *The lookahead distance as a measured setting* (`Lookahead`): from the
    read time, the layer time and the recall observed per distance it picks
    the distance with the least expected waiting, starting at two layers.
  Still planned, with the first MoE model: inside a layer, experts that are
  present first and a miss issued the moment the router has run, gate and up
  before down; gate and up of an expert adjacent in its slot and computed in
  one pass; the shared expert's output in the predictor's input.
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

**Decode weight kernels.** A matrix times 1 to 8 rows of activations, which
is what decides decode speed, for all 28 weight types (`kernels/mmvq.cuh`,
`mmvq_more.cuh`, `mmvq_float.cuh`). For the quantized types the activations
are quantized once per input to int8 with a scale per group of 32, and the
kernels compute integer dot products (`__dp4a`) straight on the weight codes;
scales and offsets are applied once to the integer sum, never per value. The
float types (f32, f16, bf16) multiply the exact activations, of which the
prepared buffer holds an f32 copy. Each kernel is verified tightly against a
CPU emulation of the same arithmetic and loosely against the exact f32
product: about 1 % rms for the quantized types, from quantizing the
activations, and nothing beyond f32 rounding for the float types. Real tensors
of the test models cover 17 types. The other 11 (q4_0, q4_1, q5_0, q5_1,
q2_K, iq1_s, tq1_0, tq2_0, mxfp4, nvfp4, q1_0) are tested on generated blocks
only, because no model at hand carries them.

Measured 2026-11-10 against the memory limit of each card (479 GB/s on the
RTX 4070, 341 GB/s on the RTX 3060, measured with a plain reading kernel), at
17408 x 5120, in percent of the limit:

| Types | RTX 4070, 1 row | 8 rows | RTX 3060, 1 row | 8 rows |
| --- | --- | --- | --- | --- |
| 4 bits and more, floats (q4_K, q5_K, q6_K, q8_0, iq4_xs, iq4_nl, q4_0, q4_1, q5_0, q5_1, mxfp4, nvfp4, f32, f16, bf16) | 95 to 100 | 71 to 98 | 93 to 100 | 46 to 92 |
| 3 bits (q3_K, iq3_s, iq3_xxs) | 87 to 96 | 39 to 63 | 63 to 82 | 26 to 44 |
| 2 bits and fewer (q2_K, iq2_s, iq2_xs, iq2_xxs, iq1_s, iq1_m, tq2_0, q1_0, q2_0) | 65 to 94 | 29 to 57 | 47 to 88 | 19 to 38 |
| tq1_0 | 14 | 10 | 9 | 7 |

- One row of Q4_K takes 0.107 ms on the 4070 and 0.150 ms on the 3060.
- The types of 4 bits and more hold 90 % or more up to four rows on the 4070.
  The low-bit types fall off from two rows on, because their work is integer
  arithmetic (a `dp4a` per four weights and row), not memory. They need the
  tensor cores from about three rows on, which is the register kernel of the
  prefill family below.
- A weight load narrower than a 32-byte sector must not bypass the L1 cache
  on these cards, or it requests its sector from L2 again with every
  instruction. Fixing that took iq4_xs at eight rows from 322 to 428 GB/s on
  the 4070 and from 163 to 244 on the 3060.
- TQ1_0 is bound by the loads of its activations (five scattered words and
  five scales per lane) and is the one type far from the limit at one row.
- Hardware counters (`ncu`) are blocked for non-root users on the development
  machine, so what bounds a kernel was established by ablation and `nsys`.

A pass over all weight matrices of Qwen3.8-27B (13.8 GiB, 96 % of the file,
split 0.625 / 0.375 over the two cards) takes 22.0 ms with one row, 23.2 ms
with three, 24.4 ms with four and 34.6 ms with eight (38.0 ms before the load
fix). That is a lower bound for a decode step
(no attention, recurrence, sums or sampling); the llama.cpp fork needs 28 ms
for the whole step. Uploading those weights took 1.9 s with the file in the
page cache and 9.5 to 12 s from a cold disk, through the simple path;
freeing them 4 to 8 ms. Launching a kernel costs the host under a
microsecond and nothing in the path synchronises.

**Prefill weight kernels.** A quantized matrix times many rows of
activations, on the tensor cores, for the same eleven types
(`kernels/mm.cuh`, `mm_types.cuh`, `mmr.cuh`, `prefill.cu`). Three paths,
which `Kernels::prefill_path_for(ty, precision)` chooses per type and device:

- **int8**: `mma` on int8 tiles of the weight codes, reading the same int8
  activations the decode kernels use. The integer result of every 32 values
  is scaled in f32. 1.05 % rms from the exact f32 product (1.5 % at worst) on
  the test data.
- **f16**: the weights decoded to f16 tiles (packed f16x2 arithmetic from the
  codes and the f16 scales), f16 `mma` with f32 accumulation. The activations
  are divided per row by a power of two so that their largest value is below
  1, and the factor is put back at the end. 0.03 % rms.
- **f16 with f16 accumulation** (`PrefillPath::F16Flush`): the same tiles, but
  `mma` accumulates in f16 and the partial sum of every 4 `mma` (64 values of
  K: at most 64 products of a weight and a value below 1, so no overflow) is
  added to f32 registers. GeForce cards run that at twice the rate of f32
  accumulation. 0.05 % rms, 0.12 % at worst.

On a capped card (below) `Precision::Fast` takes F16Flush for Q4_K, IQ2_S and
IQ2_XS, where it is as fast as int8 (1.02 to 1.18 times, measured on both
cards at 1024 rows) and 20 times more exact, and int8 for the other types
(F16Flush is 0.74 to 0.91 times as fast there); `Precision::Exact` is always
f16. A row gives the same bits alone and inside a batch within a path.

Measured 2026-11-10, 17408 x 5120, TOPS (2 * n_out * n_in * rows per
second), RTX 4070 (peaks: int8 234, f16 with f16 accumulation 117, with f32
58.6) / RTX 3060 (102, 51.2, 25.6):

|              | 64 rows   | 256       | 1024      | 2048      |
|--------------|-----------|-----------|-----------|-----------|
| q4_K int8    | 45 / 24   | 76.5 / 33.7 | 78.1 / 34.3 | 78.0 / 34.5 |
| q4_K f16     | 46 / 22   | 54.8 / 23.0 | 55.2 / 23.7 | 55.3 / 23.7 |
| q4_K F16Flush | 46 / 22  | 75.4 / 33.5 | 77.9 / 34.1 | 77.7 / 34.1 |
| iq4_xs int8  | 45 / 21   | 81.8 / 36.2 | 81.9 / 37.1 | 81.6 / 36.9 |
| iq4_xs F16Flush | 44 / 22 | 69.8 / 30.2 | 69.7 / 30.8 | 69.5 / 30.8 |
| q6_K int8    | 28 / 13   | 54.2 / 23.8 | 55.7 / 24.8 | 55.9 / 24.8 |
| q6_K F16Flush | 30 / 14  | 43.3 / 19.4 | 44.2 / 20.1 | 44.6 / 20.1 |

(First-round values at 1024 rows: q4_K int8 62.4 / 26.4, f16 49.3 / 22.5;
iq4_xs int8 52.8 / 22.8; q6_K int8 33.4 / 14.7. The f16 path on the RTX
3060 was slower than in the first round after the large 128 x 128 shape
(19.5 against 22.5); it is 23.7 now.) Which type is slowest depends on the
path: q6_K, in both.

What bounds them, by ablation (the same kernel with a part removed; the
hardware counters are not available): int8 issues 416 instructions per warp
and step of which 192 scale the integer results, four warps per scheduler
1664 for 1024 cycles of tensor work, so it cannot get near the tensor cores
whatever the overlap; removing the scaling gives 111 TOPS. The f16 kernels
have no such cost, and an empty skeleton of `mma`, `ldmatrix` and barriers
reaches 103 of 117 TFLOPS with f16 accumulation (RTX 4070; 44 of 51.2 on the
RTX 3060). What held the first version of them back was the weights' loads:
with every weight row reading the same address the kernel was 35 % faster.
The file's layout puts the 36 to 70 bytes a row needs per step 2880 bytes from
the next row's, so one load of a warp touches 16 lines, and the load unit
(which `ldmatrix` shares) is busy with them. Decoding the next tile inside the
loop of `mma`, loading the quants into L1, and decoding with packed f16x2
instructions (Marlin's `prmt`/`hsub2`/`hfma2`) took Q4_K from 49 to 55 TFLOPS
with f32 accumulation and to 78 with f16 accumulation. Now the activations
bound them: without their `cp.async` the kernel reaches 98 TFLOPS (RTX 4070).

**Rearranged Q4_K** (`Kernels::rearrange`, `matmul_rearranged`,
`kernels/mmr.cuh`). A Q4_K matrix can be rearranged once at upload into the
order the tensor cores read it (codes in `mma` fragment order, 16 bytes per
lane and 64 values; the 16-byte block headers of 16 rows together): the same
number of bytes (rows rounded up to 16), 0.3 ms for a 17408 x 5120 matrix on
the RTX 4070 (about 80 ms for the 13.8 GiB of the 27B model), and a kernel that
decodes in registers like Marlin's, with no weight tile in shared memory. It
is correct and as fast as the tile kernel from 256 rows on (78.7 / 33.9
TFLOPS at 256 to 2048 rows), faster at 128 rows on the RTX 4070 (76.7 against
60.1), and slower below (29 at 64 rows). So Marlin's layout and register
decoding did not beat the improved tile kernel at prompt sizes, because
both are bound by the activations now; the form is kept for 128 rows and as the
base for the next step. A rearranged matrix is read by nothing else: the decode
kernels and the other prefill kernels need the file's form, so a model that
decodes with the same matrix cannot keep only the rearranged one (until a decode
kernel reads it).

`Kernels::decode_kernel_rows(ty)` is the number of rows up to which the decode
kernels are the faster family (6 to 20, by type); above it `matmul` is used.
Re-measured with all paths: Q4_K 16, IQ4_XS 12, Q6_K 8 to 12 on the RTX 4070
(unchanged); the RTX 3060 keeps decode ahead up to about 16 for Q6_K.

A pass over all weight matrices of Qwen3.8-27B (Q4_K_S file, 13.8 GiB) at
the split 0.625 / 0.375 takes 92 ms with 64 rows, 221 ms with 256, 694 ms
with 1024 and 1339 ms with 2048: 1529 rows per second at 2048 rows (first
round: 1054), at a split of 0.7 / 0.3 1787 rows per second at 1024 rows (the
2048-row pass no longer fits the RTX 4070's memory since the prepared
activations carry an f32 copy). The llama.cpp fork prefills about 1180 tokens
per second at 2048 tokens for its whole forward pass, so the weights alone are
now about 1.3 to 1.5 times as fast as the fork's whole pass; attention and the
rest still come on top. The f32 copy is of use only to the decode kernels of
the float types and is to leave the buffer the prefill kernels size.

**Layer kernels** (`kernels/layers.cu`, `ares_cuda::LayerKernels`).
Everything a block of `qwen35` does between its matrix products: the
embedding gather (15 weight types), the norms per head, the causal
convolution with its window of earlier rows, the gated delta rule as a
window that reads the state and a commit that writes the accepted prefix,
the gated output norm and SwiGLU fused with the quantization of the next
product's activations, the rotary table and the preparation of queries and
keys. The rotary table is built once per step in f64, because an f32 angle
is off by about 0.01 rad at position 262k. A step record, uploaded once per
step, tells the kernels which rows belong to which sequence and state slot,
their positions, how many rows were accepted, and which rows are certain: a
prompt row or a row without a draft writes its state directly and needs no
commit. The same kernels serve `qwen4exp`, whose recurrent layers differ
only in the gate of the output norm.

Against the f64 CPU reference every kernel is within 7e-7. A chain of all 18
recurrent blocks of Qwen3.5-0.8B on the GPU, with the weight kernels in
between, is within 1.0e-3 to 1.5e-3 of the reference with exact products and
within 1.6e-2 to 1.9e-2 with 8-bit activations; the second number is the
rounding of the weight kernels.

Measured 2026-11-10 at the shapes of Qwen3.8-27B. A decode row costs a
recurrent block about 21 µs on the RTX 4070 outside its products, of which
the delta rule takes 10 to 14: it moves the 3 MiB of state of the block and
is bound by memory. A prompt of 2048 rows costs the delta rule 1.16 ms per
block (2.44 ms on the RTX 3060), which is 2.3 times as fast as the thread
mapping of the llama.cpp fork's kernel in the same harness. The rule is
sequential in the rows; a chunked form on the tensor cores is not built.

The state of the delta rule can be held in `f16` (`StateFormat::F16`), which
halves the 150 MiB per sequence of the 27B and takes the delta rule of four
sequences from 55 to 32 µs. Measured on the CPU reference with the 0.8B over
1000 tokens of real text, rounding the state to `f16` after every step moves
the logits by a KL divergence of 1e-6 to 3e-6 and never changes the most
likely token, but the error grows with the length (the hidden state is off
by 7e-4 over the first 250 tokens and by 1.8e-3 over the last). Rounded
once, as a checkpoint would be, it is 1e-8 for `f16` and 1e-4 to 2e-4 for
int8 with a scale per row; int8 with a scale per head is unusable. `f32`
stays the default for the live state until `f16` is measured on the 27B and
on long sequences; checkpoints can be `f16`.

**Attention kernels** (`kernels/attention.cu`, `ares_cuda::AttentionKernels`,
`attention::Planner`). They read K and V in the stored formats, with no
copy in another type. On the device a sequence is a table of entries of 128
cells, the same for every layer; an entry is either exact (K and V rows in
`f16`) or converted (per KV head the mean and the rows of K, then of V, in
the bytes `ares_quant::kvhead` stores). Formats without a mean (`f16`,
`q8`) are written straight into converted entries. For `q7` to `q4` the
first tile and the newest tiles are exact entries, and a full tile is
converted with its mean when the policy says so. One kernel template serves
all 36 pairs of K and V formats (a switch per chunk), in shapes for up to 8,
16, 48 and 64 query vectors, over dense windows and over lists of ranges,
for several sequences in a step. A planner cuts a step into items of one
block each, and their results are merged as log-sum-exp states, so a head
can be computed over any split of its cells.

Writing is bit-exact against the golden files of the formats and against
`KvHead` for appends, drafts and conversions. Reading is within about 1e-3
of the largest output element of `KvHead::attend` on the same stored bytes
(6e-3 at worst), which is the `f16` operands of the tensor cores; it does
not depend on the format, and `q4` / `q4` on captured rows of Qwen3.8-27B
is as close as `f16` / `f16`.

Measured 2026-11-10 per attention layer (Qwen3.8-27B has 16), one sequence,
RTX 4070 / RTX 3060:

| K / V | 1 row, 4096 cells | 32768 cells | 262144 cells | 8 rows, 262144 cells | prompt of 2048 rows |
| --- | --- | --- | --- | --- | --- |
| `q8` / `q8` | 50 / 90 µs | 0.28 / 0.46 ms | 2.19 / 3.48 ms | 3.14 / 5.39 ms | 2.73 / 5.95 ms |
| `q4` / `q4` | 49 / 92 µs | 0.23 / 0.51 ms | 2.11 / 3.62 ms | 3.47 / 6.08 ms | 2.41 / 5.22 ms |
| `f16` / `f16` | 54 / 97 µs | 0.36 / 0.53 ms | 2.76 / 3.98 ms | 3.44 / 5.84 ms | 2.43 / 5.26 ms |

At 262144 cells `f16` reads at 77 % of the memory limit, `q8` at 52 % and
`q4` at 29 %: the quantized formats are bound by the work of decoding, not
by memory, so they are not yet faster than `f16` although they read a half
or a quarter of the bytes. Open: that decode work, accumulation in `f16`
with a flush to `f32` on capped cards (every path accumulates in `f32`
now, at half the rate), the item for a prefix that several sequences share,
and a Turing path.

**The tensor-core cap of GeForce cards.** GeForce cards run FP16, BF16 and
TF32 tensor-core math with FP32 accumulation at half rate; professional cards
with the same silicon do not, and gpu-unlock
(https://github.com/Zagitta/gpu-unlock) removes the cap on RTX 20, 30 and 40
at boot. Kernels can work around the cap (integer tiles, or accumulating in
FP16 and flushing to FP32, as above), at the price of a small loss of
exactness. Without the cap the direct path is both the fast one and the exact
one, so the workaround and its divergence are simply gone.

There is no build option for this. Each device is probed for a few
milliseconds when the kernels are loaded and the result is reported at start:

- **Tensor cores at full rate:** the direct path (f16, f32 accumulation) is
  used.
- **Capped:** the workaround is the default, because it is faster there, and
  the report says so and names the remedy. The exact path can be chosen
  instead.

The development machine's cards are capped, and the probe shows it exactly:
f16 `mma` with f32 accumulation runs at 58.6 TFLOPS on the RTX 4070 and 25.6
on the RTX 3060, with f16 accumulation at 117.0 and 51.2, and int8 at 234 and
102 TOPS. The probe takes about 10 ms in `Kernels::load`
(`tensor_cores_full_rate()`). No uncapped card was at hand, so the f16 kernel
with f32 accumulation could not be timed at the full rate; with f16
accumulation (the rate an uncapped card has for f32) it is the F16Flush path
above.

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

- **Peer**: directly between the devices. The mesh takes this route whenever
  the driver reports peer access for every pair of devices, however the
  system comes by it: an NVLink bridge, cards and drivers that allow
  peer-to-peer over PCIe as they are, or consumer cards with patched NVIDIA
  kernel modules and the matching driver library (selected with
  `ARES_LIBCUDA=<path>/libcuda.so.1`). Nothing in the engine is specific to
  the patched driver, but that is the only kind of peer access the
  development machine has: NVLink and unpatched peer-to-peer are untested.
- **Host**: through a page-locked buffer.

Peer transfers are not a CUDA matter alone. The Vulkan backend, which is
what AMD and Intel cards run on, has no mesh yet; it gets the same two
things with the routes Vulkan offers (memory of one device imported by
another, device groups, shared page-locked host memory), chosen by probing.
Still to do on the CUDA side: a route per pair of devices instead of one for
the whole mesh, the kind of link in the start-up report, and a probe of
whether kernels can read peer memory (they can with real peer-to-peer, which
would allow pulling instead of pushing).

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

### Qwen3.8-27B on the two cards

Measured 2026-11-10 with `crates/model/examples/qwen35_run.rs`, one
sequence, 8-bit activations, K and V in `q8`, the peer route. The llama.cpp
fork on the same cards: about 1180 prompt tokens per second at 2048 tokens,
28 ms per token without drafts (35.7 tokens per second), and 62 to 67 tokens
per second with drafts from the MTP block and its production flags.

| | Tensor split 0.625 / 0.375 | Tensor split 0.7 / 0.3 | Layer split 0.62 / 0.38 | Fork |
| --- | --- | --- | --- | --- |
| Prompt of 2048 tokens | 1218 tokens/s | 1461 to 1474 | 1348 | about 1180 |
| Decode without drafts, 1000 tokens of context | 24.8 ms (40.4 tokens/s) | 26.2 ms | 41.4 ms | 28 ms |
| With the model's drafts, German / English / code | 79.3 / 75.6 / 79.0 tokens/s | | not built | 62 to 67 |
| The same, sampled | 76.4 / 72.4 / 83.4 tokens/s | | | 63 to 66 |
| VRAM taken at a context of 4096 | 9500 / 5410 MiB, with the draft block 10064 / 5540 | 10590 / 4362 MiB | 9582 / 5940 MiB | |

The fork's numbers with drafts were measured on the same day on the same
questions by running it with a copy of its production preset; the other fork
numbers are older measurements of the same setup.

**The tensor split.** Each card holds a share of every matrix and the cards
add their partial results twice per block. In the first version that made
prompt processing half as fast as the fork (554 tokens per second): 128 sums
per step, each moving rows x 5120 `f32` values in both directions over the
RTX 3060's PCIe 3.0 x4 link, with nothing overlapped. Now:

- The sum, the add to the residual stream, the next norm and its 8-bit
  quantization are one kernel per half block, where there were four
  launches.
- Partials travel as `f16`, each row with a power-of-two factor so that no
  finite value overflows. Every card adds the same rounded values in the
  same order, so all hold the same residual stream bit for bit. With exact
  products the partials travel as `f32`.
- A step of more than 256 rows is cut into groups; one group's partials
  travel while the next group computes. That overlap alone took the rate at
  0.7 / 0.3 from 964 to 1464 tokens per second.
- Few rows (a decode step) are pushed inside the merge kernel and need no
  order from the host.

Prompt processing is bound by the tensor cores now (76 to 79 % of each
card's time at 0.7 / 0.3); the link carries 1.9 of about 3.3 GB/s. Without
peer access prompts are as fast (1213 and 1461 tokens per second through
host memory) and a decode step takes 26.8 ms. One share has to serve both:
0.625 costs prompts 17 %, 0.7 costs decode 6 %. The shares are parameters;
choosing them from a measurement at load is not built.

**A decode step** without drafts is 24.8 ms, of which the weight products
are 21 (the floor the memory bandwidth of the two cards sets is 22.0 ms for
a pass over all weights). With the model's drafts a step is 34.1 ms for 2.5
to 2.7 tokens: the draft block's chain takes about 4 ms of it (1.46 ms per
draft token), the sampler 0.06 ms on the RTX 4070 and 0.28 ms on the RTX
3060, the host 0.02 ms. At 32000 tokens of context a step without drafts
was 28.2 ms *(measured by the agent that built the tensor split)*.

Against the reference, on Qwen3.5-0.8B on one card: the most likely token
is the reference's in 725 of 725 prompt rows with exact products and in 715
of 725 with 8-bit activations (the widest margin missed is 0.106 logits),
and 40 greedy tokens are equal except at one near tie in German. llama.cpp
itself agrees with the reference in 704 of 725 rows.

The weights are loaded in about 4 s from a cached file. The recurrent state
takes 150 MiB per sequence over both cards, K and V 33 MiB per 1000 tokens.

**Device layouts in the model.** At load the model probes its own products
in both layouts, for every type and shape, at the rows it decodes with and
at a prompt group, and gives each group of matrices the layout that does not
slow a decode step and is fastest for prompts after that. Measured
2026-11-11 at shares of 0.625 / 0.375, every matrix in the file's layout
against the measured choice: prompts at 1234 to 1243 against 1285 to 1291
tokens per second, a decode step without drafts 24.7 against 24.8 ms, with
the model's drafts 81.0 / 76.7 / 78.7 against 81.8 / 84.2 / 85.5 tokens per
second (a step of 34.2 against 32.4 ms). The gain is in windows of drafts
and in prompts: at one row the device-layout kernels are slower than the
integer kernels on most matrices, so without drafts most of the model stays
in the file's layout. With every matrix in the device layout prompts run at
1360 tokens per second and a decode step is 7 % slower *(measured by the
agent that built the choice)*.

### Qwen3.8-Flash-Next on the two cards

Measured 2026-11-11 with `crates/model/examples/qwen4exp_run.rs`, one
sequence, no drafts, a missing expert uploaded. Strata on the same machine
(its log of 2026-11-07): 128 tokens generated at 12 to 76 tokens per second,
median 56, with drafts; prompts read at 42 to 201 tokens per second.

The model is 63.5 GiB on a machine with 24 GB of VRAM and 31 GB of RAM. The
dense parts (3.4 GiB) are in VRAM, 14 100 of the 24 576 experts have a VRAM
slot (19.4 GiB), a RAM tier holds some thousands more, and the rest is read
from the file when the router picks it. On recorded text 99 % of the picks
find their expert in a slot.

| Placement | Decode step at 1000 tokens, median | Fastest | At 8192 tokens |
| --- | --- | --- | --- |
| By layers (0..25 on one card, 25..48 on the other) | 25.2 ms | 21.8 ms | 31.1 ms *(agent)* |
| Both cards on every layer | 22.3 ms (19.6 on a quiet drive *(agent)*) | 16.5 ms | 21.8 ms *(agent)* |

In the first placement the cards work one after the other. In the second
one card mixes the residual streams and runs the router, attention and the
head, the other computes its share of the recurrent heads and of the experts
of every layer, and one sum per half block joins them; no weight is on both.
A step that misses no expert takes about 17 ms. The median is above that
because an expert that is in no slot now holds both cards while it is
fetched, and an upload to the RTX 3060 takes 0.4 ms over its link.

Against the CPU reference the largest logit of a prompt's last token is the
reference's in every run, 9 or 10 of the 10 largest agree, and the next
token is the reference's in 159 or 160 of 160 German and 79 of 80 English
rows; the rows that differ are near ties. The model runs through a session
of its own, with at most eight rows per step: prompts are read at 54 to 73
tokens per second *(agent)*, bound by experts that have to be fetched.

An expert that is in no slot can be computed on the CPU instead of being
uploaded, chosen per expert by measured cost. On the recorded routing with
few slots that took the expert path from 33 to 44 steps per second; with 99
% of the picks in a slot it makes no measurable difference to the model.

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
- The twelve weight types that have a device layout on CUDA have it on
  Vulkan too, rearranged by a shader into the same bytes
  (`ares_quant::layout` is the specification both are tested against byte
  for byte), so one upload format serves both backends. One family of
  kernels on cooperative matrices serves every row count from that copy,
  from the first row, with a table per device from a measurement at load
  and the same API as CUDA's. The other types run their file-layout kernels
  (integer dot products for few rows, cooperative matrices through a decoder
  per type for many). Measured 2026-11-11, a pass over all weights of
  Qwen3.8-27B on the two cards (split 0.625 / 0.375), device layout against
  file layout: one row 21.9 against 21.6 ms, eight rows 24.0 against 40.7,
  sixteen 33.6 against 96.8, sixty-four 68.1 against 102.4. CUDA's device
  layout takes 23.4, 25.4, 31.0 and 69.7 ms for the same passes. At 1024
  rows a Q4_K matrix runs at 88.5 / 39.9 TOPS on the RTX 4070 / RTX 3060
  and Q6_K at 75.5 / 34.3 *(measured by the agent that built the kernels;
  CUDA: 88.8 / 39.3 and 74.5)*.
- The layer kernels exist with CUDA's semantics and are held to the same
  references. Attention, the expert and sampler kernels, the kernels of
  `qwen4exp` and a model on Vulkan do not exist yet.
- No direct route between the two cards works on this driver: an exported
  allocation is imported by the other card without an error and carries no
  data. Sums go through host pages that both cards import, at the speed of
  CUDA's host route (55 against 47 µs for 10240 values; CUDA's peer route
  takes 18). Direct routes are built and untested for lack of hardware.
- The NVIDIA driver (615.78) has crashed at process exit in both
  directions: with devices still alive, and while destroying a device whose
  shader compiler thread was still working after hundreds of pipelines had
  been created. The tests leave without destroying their devices; the
  engine needs the same care at shutdown.

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
| Marlin, ExLlamaV3, LMDeploy, TensorRT-LLM (their kernels) | Weights rearranged at load into the order the tensor cores read them and decoded in registers; `f16` accumulators folded into `f32` on capped cards; one decode of a matrix and a dense product for wide steps; the shape of the attention kernels; bit planes for K/V codes | `ares-quant` (planes done); `ares-cuda` prefill and attention kernels (in progress) |
| FlashInfer | Attention as states that merge (`lse`, `v`), so a head is computed over any split of its cells; block-sparse attention over a block table | `ares-quant::kvhead` (reference done); kernels in progress |
| vLLM, SGLang, TGI, TokenSpeed, LMDeploy, ExLlamaV3 (their schedulers and caches) | Pages claimed per step with a reservation for the output still expected; a request waits when it does not fit; prefix pages published when they fill, with tails; checkpoint intervals and eviction for recurrent state; a context term in the step cost | `ares-sched`, `ares-state` (done as bookkeeping, not wired into the conductor) |
| TensorRT-LLM, SGLang, vLLM (suffix decoding) | One lookup drafter on a suffix automaton over a corpus shared by all requests | `ares-sched` (done, not wired) |
| MiniJinja | The interpreter of chat templates | `ares-tokenizer` |
| llguidance | Grammar-constrained output by validating the GPU's top candidates on the CPU | Planned |
| AirLLM | Nothing beyond the expert tiers: reading whole layers per pass is bounded by disk throughput | - |
| Run My AI, ik_llama.cpp, Fiddler, ktransformers, BELLS, AI-DER, FATE, ProMoE | Direct reads with demand before prefetch; prefill layer by layer in waves of experts; the RAM tier as a cache with admission and protection; the order of work inside a MoE layer; slots per layer by need; fused gate and up of an expert | `ares-io` (reads done); the rest planned, see [Weights](#weights) |

Notes on each source are under [reference/](reference/). The reading of the
other engines' code, with a ranked list of what to take and a list of where
Ares still copies llama.cpp without a reason, is
[reference/serving-engines-code-survey.md](reference/serving-engines-code-survey.md).

## Status

State on 2026-11-11, on an RTX 4070 and an RTX 3060 with driver 615.78.08.
Work is paused here until Qwen4 is released; nothing below is in progress.

- **Qwen3.8-27B** runs on both cards through the runtime and the server and
  is ahead of the llama.cpp fork in tensor split: a prompt of 2048 tokens at
  1285 to 1291 tokens per second against about 1180, a decode step without
  drafts in 24.8 ms against 28, and 82 to 86 tokens per second with the
  model's own drafts against 62 to 67. Through HTTP it is as fast as without.
  See [Several devices](#several-devices).
- **Qwen3.8-Flash-Next** runs on both cards, agrees with the CPU reference
  and answers in German and English. A decode step at 1000 tokens of context
  takes a median of 22.3 ms without drafts (about 45 tokens per second; a
  step that misses no expert takes 17 ms). Strata generates at a median of
  56 tokens per second with drafts on this machine, so this model is not
  ahead of its reference engine yet: it has no drafts, and prompts are read
  at 54 to 73 tokens per second in steps of eight rows.
- CUDA has every kernel family the two models need: weights for all 28
  types (twelve with a device layout that serves every row count from one
  copy), layers, attention over the stored K/V formats, sampler, sums
  between devices, router and experts, the kernels of `qwen4exp`. The CUDA
  tests pass on both cards, the kernel ones also on real tensors.
- Vulkan has the weight kernels (the same device layouts, byte for byte) and
  the layer kernels, at CUDA's speed on these cards. It has no attention, no
  expert, sampler or `qwen4exp` kernels and no model.
- The KV formats are specified, pinned by golden files and measured on the K
  and V of real models; the CUDA kernels write them bit-exactly and read
  them within 1e-3 of the CPU reference. The dense model uses `q8` for K
  and V; the formats with a tile mean (`q7` to `q4`) are not wired into a
  model.
- The prefix cache runs on the devices for `qwen35`, in every placement: a
  conversation's next turn starts from the pages and the checkpoint of the
  last one (on Qwen3.8-27B a second turn after a history of 24k tokens gives
  its first token after about 0.14 s instead of 25 s; see "Sequence state"),
  the usage of an answer says how much came from the cache, and the engine's
  RAM is one setting (`--ram`).
- Known defects: a warm-up step that fails hangs the start of the runtime
  instead of reporting; the first token of a prompt that arrives next to a
  drafting sequence takes about 0.4 s; with the fast kernels the text with
  drafts is not always the text without (with exact products it is, for
  steps of the same size: see "Rounding" under "Sequence state"); a prompt
  that shares only the start of the chat template with an earlier request
  (three tokens) now runs a step of its own to replay them before its rows
  (one more step in the counts of `--engine`, not timed); a model that only
  runs on Turing has no prefill and no attention path.
- Real model files are test data and stay outside the repository.

## Next steps

Nothing is being worked on until Qwen4 is released. What is open, in the
order it would be taken up; the source of each idea is in
[reference/serving-engines-code-survey.md](reference/serving-engines-code-survey.md).

1. Qwen4 itself: its architecture next to `qwen4exp`, and whatever its
   files need.
2. The mixture-of-experts case ahead of Strata: drafts from the model's
   draft block; the prompt path (experts in waves, tensor-core kernels for
   the experts' types, steps of more than eight rows); the model behind the
   runtime's `Model` with flow control by route records, so the server can
   serve it; owners of experts that follow the heat.
3. Attention for models that attend a few thousand cells through lists: the
   kernel chosen by a measurement on the device, staged loads where they
   pay, one launch per row, items of twelve query heads, rows of a prompt
   that share cells; and the K/V formats with a tile mean in a model.
4. The prefix cache beyond the first pass: checkpoints at the message
   boundaries the server names (the anchor is a page boundary before the end
   of the prompt now), a warm pass in the background, spill of a sequence to
   RAM, pages and checkpoints that move between VRAM, RAM and `ares-store`
   (and so survive a restart), locking the RAM of the next chunk of places
   ahead of its first use, and `qwen4exp` with a cache.
5. A kernel for one to three rows on the device layout that is not slower
   than the integer kernels, so the dense model can hold every matrix in it;
   shares of a tensor split from a measurement at load; device layouts for
   the remaining types.
6. Vulkan: attention, experts, sampler, the `qwen4exp` kernels, and the
   models; a faster upload.
7. The sampler's penalties and grammars (llguidance), tool calls, the
   router mode with `models.ini`, multimodal input, `Engine::decide` and
   `/v1/systemone` on the real engine.
8. A path for Turing; a run on any card that is not one of the two here.

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

**What is built.** The broker is `PinnedRam` (`crates/runtime/src/ram.rs`),
shared by the executor and the device threads. It has two users so far: the
step record and the read-back buffers of the devices (required, booked at
start: `required_ram`) and the checkpoints of recurrent state of the prefix
cache (elastic, the lowest priority, floor 0), which a device part locks in
chunks of at least 32 MiB when a place is first used and keeps locked while
the engine runs. The limit is `GpuConfig::ram_limit`, which the server sets
from `--ram` (default: half of what the machine reports as available when it
starts; the log states it and the reason); what the limit leaves after the
required memory is the checkpoint budget of the conductor
(`RuntimeConfig::checkpoint_budget`), so the places the conductor hands out
never lock more than the limit. Nobody asks the checkpoints to give memory
back yet, as nothing competes with them; a lock that does not fit fails the
step with the books in the message. The first use of a place waits for the
lock on the device's thread (21 to 22.5 ms for 93.5 MiB, 11.8 ms for 56.1
MiB, measured with the 27B); locking ahead in the background is not built. `GET /metrics` reports the
bytes locked (`ares_checkpoint_ram_bytes`).

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
  its pages are on SSD. (Pages are not in RAM yet; the checkpoints are the
  prompt cache's only RAM today.)

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
  in the background and only as far as offload needs it. (Today the only such
  memory is the checkpoints' places, which a device thread locks in chunks
  when one is first used, not in the background; see "Host memory".)
- **Unloading frees and returns.** Nothing is written at unload: state on
  disk is written as it is produced.

Load and unload times are measured with the first model; there are no numbers
yet.

## Server

`ares --model <file.gguf>` loads a model on the devices and serves it.
Without options it plans for itself: the first device that holds the weights
and a context, or else a tensor split over all devices with shares from the
memory bandwidth each device reports, and a context from the VRAM that is
left; the start-up report says what was chosen, why, and where the VRAM
went. Devices, split, shares, context, K/V format, precision, drafts and the
RAM the engine may use (`--ram`) can be given instead. The RAM defaults to half
of what the machine reports as available when the server starts; the
engine's step buffers are taken from it, and the rest is the budget of the
checkpoints that let a conversation continue from what was computed (see
"Host memory").

It answers `POST /v1/chat/completions` and `/v1/completions`, streamed and
not, `GET /v1/models`, `/health` and `/metrics`, with timings in every answer
(prompt tokens and time, generated tokens and time, drafts proposed and
accepted). The prefix cache is on for `qwen35` without a flag: a prompt that
starts with text the engine has computed reads the cached part from the pages
and the checkpoint instead of computing it, the answer's `usage` says how much
(`prompt_tokens_details.cached_tokens`, as OpenAI does), `timings.prompt_n` counts
the tokens that were computed, and `/metrics` has the counters (starts from the
cache, cells taken from it, rows replayed, checkpoints saved, restored and
dropped, pages copied, pages cached, RAM locked for checkpoints). A field of a
request that the server does not hand to the engine
is refused by name, not ignored: grammars, logit bias, penalties, tools and
images among them today. A client that disconnects frees its sequence. The
first signal lets running requests finish, the second ends them, and the
devices are released before the process exits.

Measured 2026-11-11 with Qwen3.8-27B on both cards: 79.2, 75.3 and 78.8
tokens per second for German, English and code through HTTP, which is what
the same model gives without HTTP; the first token after 76 to 121 ms; two
requests at once at 63 tokens per second each.

A router mode driven by a preset file (`models.ini`, one section per model)
that loads and unloads models on request, and multimodal input, are not
built.

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
