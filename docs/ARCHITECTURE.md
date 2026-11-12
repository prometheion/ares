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

- **Two supported models until Qwen4.** Qwen3.8-27B (`qwen35`), dense and
  entirely on the GPUs, and Qwen3.8-Flash-Next (`qwen4exp`), a mixture of
  experts with the experts in tiers and streamed from SSD, are the models
  the engine supports. They are also the two cases every part is built and
  measured on, one for each way weights are held: a part of the engine is
  done when it has run in both. Both are deprecated when Qwen4 is released
  and removed when it has replaced them. On the development machine (two
  cards with 12 GB, 31 GB of RAM, a 63.5 GiB file) Flash-Next can only run
  with streaming, so the case without it is not measured there.

- **The first release is complete.** There is no release before the engine
  has everything: what the llama.cpp fork it replaces has and what the other
  engines have, inside the limits listed here, ready for use, with the same
  features on CUDA and on Vulkan. That includes drafts from block drafters
  with draft trees, image input, the decision endpoint, a path for Turing
  cards and the prompt cache on SSD. The list of features, who has each and
  how far Ares is, is
  [reference/feature-checklist.md](reference/feature-checklist.md); a
  release date is not a reason to leave a row out.

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
  type, and the embedding gather decodes every type on both. The kernels for
  many rows cover every type on both backends.

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
6. **Sample on the device.** Logits stay on the GPU; the corrections of a
   request (penalties, DRY, logit bias, a mask), the samplers it chooses
   (top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature and a dynamic
   one, mirostat), the pick, and the probabilities it asks for all run there.
   The random number for a position comes from a counter-based generator keyed
   by (seed, position, token), so the randomness of a pick does not depend on
   what else is in the step, and verifying a draft compares it with the pick the
   model makes for that position. What a pick depends on that only the host
   knows, the tokens before the row, goes to the device per row, so every row
   of a window is picked as if the drafts before it were the text.
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
| `ares-sampler` | The definitions of the sampler: the settings of a row, the arithmetic every backend repeats bit for bit, the pick of one row written from its definitions (`sample_reference`), the penalties, DRY and logit bias the host works out, token masks | Done; the reference is checked against llama.cpp's samplers written again in double precision, the penalties against their definition; both backends' kernels are held to it bit for bit; see [Sampling](#sampling) |
| `ares-cuda` | Driver API bound at run time, devices, memory, streams, events, modules, decode weight kernels for all 28 weight types, prefill weight kernels for eleven, the layer kernels of `qwen35`, attention over the stored K/V formats, the sampler's kernels, the mesh that sums across devices over peer access or the host | Device layer, decode, layer and attention kernels done; prefill kernels verified, their pass over the weights 1.3 times as fast as the fork's whole prefill; tested on both GPUs, with and without peer access |
| `ares-grammar` | The grammar engine on the host: GBNF grammars, JSON schemas, Lark and regular expressions turned into the set of tokens that may come next (llguidance behind a vocabulary of bytes per token), the reasoning gate, the cut of forbidden drafts | Done; GPU-free tests of the engine, and through the runtime and the server; see [Sampling](#sampling) |
| `ares-vulkan` | The same device-level surface on Vulkan; decode and prefill weight kernels for all 28 weight types as compute shaders, the prefill ones on cooperative matrices; the layer kernels; attention over the stored K/V formats; the sampler's kernels; sums between devices | Device layer and weight, layer, attention and sampler kernels done (23 device, 9 kernel, 8 layer, 4 layout, 9 mesh, 9 attention and 8 sampler tests on both GPUs; the sampler's records equal `ares-sampler`'s reference bit for bit); no expert or `qwen4exp` kernels and no model yet |
| `ares-state` | Page pool, page tables with copy on write, the prefix index (pages published as they fill, tails, announcements, priority classes), the checkpoint policy for recurrent state, page layouts | Done as bookkeeping; randomised model checks; the conductor's prefix cache uses them (see `ares-runtime`) |
| `ares-store` | State on disk: pages and checkpoints by content hash in segment files, per fingerprint, with a size limit | Done as a store of opaque bytes, tested against restarts, the limit and damaged files; not yet connected to devices |
| `ares-io` | Reads of weights straight from the model file: io_uring with direct I/O, demand before prefetch, cancel and promote, copies of a file on several drives with the reads spread over them, a map from experts to file ranges, row reads behind a cache | Done, tested against real files and on two drives; 1.4 to 1.7 GiB/s of random expert-sized reads per SSD of the development machine and 2.9 to 3.0 GiB/s over both; the expert store and the n-gram rows of `qwen4exp` read through it |
| `ares-sched` | Step planner (generation, decision rows and prompts in one step), admission (pages claimed per step, a reservation that corrects itself, a queue instead of refusals), the order of a pass for long prompts of a model with experts, step time model with a term for the context, draft length choice, the lookup drafter on a suffix automaton shared by all sequences | Done; unit and model tests; the conductor uses the planner, admission (without spill), the lookup drafter and the step cost |
| `ares-weights` | Expert heat, tiers that are elected and fed by reads (admission, protection, guesses bound late, slots given back), slots per layer by need, the waves of experts of a pass, a token table for the first layers, the lookahead distance, routing bias for pruning | Done, unit tested; the expert store uses the slot sets, the heat and the split of slots; guesses, the token table and pruning are not used by a running model yet |
| `ares-api` | Request (with every sampling field), event (text and probabilities), decision and error types; the `Engine` trait between HTTP and the runtime (`submit` for generation, `decide` for decisions, `stats`, `chat_format` for what a model's chat template declares about thinking) | Done, unit tested |
| `ares-server` | The `ares` binary: loads a model on the devices and serves it over an OpenAI-compatible HTTP API, with the plan of devices, split, shares and context made from what the devices report | Serves `qwen35` and `qwen4exp` models (chat and plain completions, streamed and not, every sampling field and the probabilities of the tokens, `n` completions, timings, metrics, staged shutdown, the reasoning of a reply apart from its content and in the history, with llama-server's controls for thinking, tool calls held to the tools' schemas by a grammar); 192 tests without a GPU, 17 that start the real binary (some need the 27B or the files of Flash-Next) and 25 that run the engine in the test's process (the 7 of tool calls have not been run); for `qwen35` the prefix cache is on, with `--ram` for the engine's RAM, for `qwen4exp` the RAM is mostly the tier of experts; no router mode, no multimodal input, `/v1/systemone` answers 501 |
| `ares-runtime` | The conductor thread and one thread per device, the step loop that connects plan, state and kernels, admission, speculation with the lookup drafter and with a model's own drafts, the `Model` trait a model implements, a synthetic model that drives the real kernels, and the expert store (VRAM slots, RAM tier, reads, the CPU computing what is not on a GPU) | Drives `qwen35` on one device, in tensor split and in layer split, with picks made on the device; implements `Engine::submit` (not `decide`); the prefix cache is in the conductor, checked against the host executor as an oracle and carried out by `qwen35` on the devices (checkpoints in page-locked RAM within one RAM limit, `PinnedRam`), and there is no spill of pages to RAM (a request that gives way stays in the pool's cache); the expert store is driven by the device threads of `qwen4exp` (see [Qwen3.8-Flash-Next through the runtime](#qwen38-flash-next-through-the-runtime)) |
| `ares-model` | Model architectures: configuration and weight table from a GGUF, a CPU reference of the forward pass with explicit state (window and commit) that GPU implementations are compared with block by block, and the implementation on the devices | `qwen35` (Qwen3.5-0.8B, Qwen3.8-27B) runs through the runtime and the server, with the model's draft block. `qwen4exp` (Qwen3.8-Flash-Next) runs through the runtime and the server on one or two cards, without speculation (it has no draft head built, and the lookup in earlier text takes part only next to a model drafter), and through a session of its own that tests and measurements use. See [Several devices](#several-devices) |

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

1. **Plan** (`ares_sched`). Prompts and generation take turns
   (`ares_sched::Share`). A step is a *round* of generation (the newest
   token of every generating sequence, with the drafts that pay: below) or a
   *piece* of prompt (up to the rows at which the model's prompt kernels
   reach their rate, 256 for `qwen35`, shortest prompt first and starved ones
   before that, with the newest token of every generating sequence riding
   along without drafts, so that they go on at a token a step). A round that
   verifies drafts keeps the inputs of its rows for the commit and has room
   for the rows of windows and no more, which is why prompt rows never share
   a step with draft rows. The one setting is the *slowdown* `s`
   (`prefill_slowdown`, 2 by default): while prompts and generation both
   wait, generation gets `1 / s` of the time. A step of prompt that takes `t`
   ms leaves `t / (s - 1)` ms of rounds owed before the next one, and a round
   pays back what it takes; a round that overshoots is a credit, so the
   share holds over time whatever the size of a round. A prompt that a piece
   completes does not wait for the debt of a long one, as long as the debt
   is no more than one whole piece leaves, so the first token of a short
   prompt comes after the step in flight and one step of its own; a burst of
   short prompts is read in one piece. While nobody generates a prompt takes
   the widest step. (The scheme is TensorFold's lanes, whose "a long prompt's
   step owes the decoders" this takes; it replaces the cap of the rows of a
   prompt next to windows at `prefill_rows_at`, under which a prompt next to
   a drafting sequence trickled in a few rows a step.)
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
   the device (`ares_cuda::SamplerKernels`, or `ares_vulkan::SamplerKernels` with the
   same steps): the logits are corrected first
   (the row's entries and mask, written in the step record), then the most
   probable token is taken, or the stages of the row's program cut the tokens
   and scale the temperature as exact thresholds (radix descents over integer
   histograms, sums of integers) and the pick is made by Gumbel noise of
   (seed, position, token) from a counter-based generator. Its arithmetic is
   repeated bit for bit on the host (`ares_sampler::sample_reference`), which
   is what its tests compare with; see [Sampling](#sampling). Under a tensor
   split every device holds the logits of a slice of the vocabulary and the
   devices add up small partial results between the sampler's steps. Only the
   picks and the eight best candidates of each row (80 bytes per row), and the
   probabilities a row asks for (168 bytes), go to the host.
5. **Commit.** For each sequence the accepted prefix is replayed into its
   recurrent state and the page table is cut after the last accepted row
   (`Sequence::truncate`). Drafters and their acceptance statistics are
   updated.

**A prompt next to sequences that generate** (one or two sequences generating
with drafts, a prompt of 77 to 110 tokens or of 2,000 tokens arriving beside
them; first token seen by the client, medians of 8 and of 3 requests;
measured 2026-11-12 at a load of 2 to 5, `--parallel 3`, before and after back
to back):

| First token of | alone | next to 1 generating | next to 2 generating |
| --- | --- | --- | --- |
| two cards (Q4_K_S, tensor split 0.625 / 0.375): | | | |
| a short prompt, before (the step capped at the draft window) | 154 ms | 490 ms | 653 ms |
| a short prompt, now | 162 ms | 189 ms | 186 ms |
| a prompt of 2,000 tokens, before | 1.64 s | 9.6 s | 12.2 s |
| a prompt of 2,000 tokens, now | 1.68 s | 4.2 s | 4.0 s |
| one card (the 4070, IQ2_XXS): | | | |
| a short prompt, before | 327 ms | 674 ms | 850 ms |
| a short prompt, now | 342 ms | 379 ms | 388 ms |
| a prompt of 2,000 tokens, before | 7.2 s | 17.6 s | 17.2 s |
| a prompt of 2,000 tokens, now | 7.5 s | 13.7 s | 12.9 s |

A short prompt waits for the step in flight (35 ms at most on two cards) and
then runs whole in a step of its own, so it is 24 to 46 ms behind what the
cards need for it alone. The generating sequences pay for it in a pause: on
two cards the longest gap in their stream went from 57 to 75 ms to 184 ms for
a short prompt (a step of its own, about 100 ms, with the newest token of each
riding along) and to 280 ms for the pieces of a long one (a piece of 256 rows
is a step of 200 to 300 ms); on one card, where a piece of 256 rows takes 0.9 s
of the IQ2_XXS weights, the gap is 0.4 s for a short prompt and 0.95 s for the
pieces of a long one. Their rate while a long prompt is read fell to 48 and
49 % of what it was on two cards, which is the slowdown of 2 that is set
(before: 75 and 69 % of it, for six and seven times as long), and to 41 and 47 % on
one card. A long prompt takes 2.4 to 2.5 times what it takes alone on two
cards and 1.7 to 1.8 times on one.

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
`DeviceModel::enqueue`, which queues the kernels of a step and never waits. A
model whose step needs the host to go on while the device runs, because a layer
waits for an expert that the host brings, also serves that in
`DeviceModel::wait`, where the thread would otherwise block in the driver for the
end of its step.
A model whose device side carries out the step record's copies and replay
segments says so (`Model::prefix_cache`, with the bytes of one checkpoint on
each device) and takes part in the prefix cache; one that does not has none.
Until a real model exists a synthetic one drives the real kernels: 64 layers
of two q4_K matrices of 5120 x 17408 each, with sums between devices for the
tensor split and a hand-over per group of rows for the layer split. Its next
token is a pure function of a sequence's own tokens, so the tests can check
that a request gets the same text alone and next to 1, 7 and 31 others, with
speculation on and off, on every placement.

**When a device fails.** The error reaches the caller with the device's index,
name and driver number and the cause (`device 1: NVIDIA GeForce RTX 3060 (CUDA
1): ...`), whether the part could not be built, a step returned an error or a
thread panicked (where panics unwind; the release profile aborts). The device
that failed is the one named, not the ones that were left waiting for it. What
used to hang the start, and could hang a step, was a kernel of the push
path: one thread per device enqueues its half of a merge and each device's
kernel waits on the device for the other's flag, so a thread that failed
before it enqueued its half left the other's stream with a kernel that waits
for a flag nobody sets (it gives up after four billion polls), and with it
the end of the step and the wait for the thread to stop. A thread that fails
now writes every flag ahead of any token from the host, over a stream of
its own and a copy engine that the waiting kernel does not hold
(`Mesh::release_waiters`, called by `Links::abort`), and the thread that
failed repeats that until the stream it had enqueued on is idle, because the
pushes it had queued before the failure still run and each writes the other
device's flag with a token that is behind the later kernels
(`Mesh::release_waiters_until_idle`); the released kernels add up meaningless
values in a step nobody reads. A part that fails to build
tells the model (`Model::abandon`) so that the parts that wait for its
measurements (`qwen35`'s layout choice, which waited two minutes) go on at
once. A thread that has not stopped ten seconds after it was told to is left
behind and the shutdown ends. `ares_runtime::testing::Faulty` makes a model
fail on purpose, and `qwen35_failure` runs it on two cards.

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
- **Speculation** with a drafter that copies from earlier text doubles one
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

**Where the tile formats keep their exact cells** (`qwen35`, `part/forms.rs`
and `part/kv.rs`). The engine's pool holds converted entries only: page `p` is
slot `p` of the converted pool of every attention layer, 128 cells in the
format. The exact entries (a tile in `f16`) are a second pool on each device,
for the attention blocks and for the draft block separately, which a page
holds a slot of while it is exact: from its first write until it is full and
has left the newest cells of every sequence that reads it. The conductor knows
nothing of this beyond one thing, the pages the pool freed, which every step
record names (see "Sequence state" below); the device decides the rest, the
same way on every device, from the segments of the record:

- At the start of a step, for every sequence in it, the tiles that left its
  window are converted into their pages (`AttentionKernels::convert`), the
  pages its rows write get exact slots, and the group's page table is brought
  up to date (entries of exact tiles name the slot, the others the page). The
  step's own rows then append to exact tiles and attend to both kinds.
- A page shared by two sequences is converted for both, when either's window
  has left it; each finds its table brought up to date when it next runs. A
  sequence that starts from the cache therefore reads the cached tiles in the
  form they have: converted where the sequence that computed them went on,
  exact where it ended.
- The pool has `slots * (2 + ceil(newest / 128)) + ceil(rows / 128) + slots +
  tails` slots (`exact_slots`): the first tile and the window of every
  sequence that runs, the tiles a step begins, and the tails of the prefix
  cache. When it is short the full page that was used least recently and that
  no sequence of the step reads is converted for room (a cached page loses its
  exactness, never its text); a partly used page never is, which is why the
  cache keeps at most one tail per slot (`PrefixCache::tails_per_slot`).
- Page copies copy the exact entry; a page that is copied is always partly
  used, hence exact.

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
| Adaptive draft length | `choose` maximises expected tokens per millisecond of the whole step from measured acceptance per position (or the lookup's confidence) and the measured price of a step by its rows. The same number chooses between drafters and accounts for the rows a draft costs other sequences. |
| Rollback rows independent of draft length | Verify and commit removes rollback: nothing recurrent is modified before acceptance. |
| Draft vocabulary without extra VRAM | Planned in weight placement: the output head's rows are ordered by frequency at load, so a draft head is a prefix of the same tensor. |
| VRAM as the binding limit | Ledgers and the three kinds of device memory. |
| Prefill next to generation | Pieces of prompt between rounds of generation, paid back by time (`Share`). |

## Speculation

The engine has every kind of drafter that llama.cpp and Piggidragon's fork
have, with the fork's selection on top, and the selection is not a setting:
it is how the engine drafts.

| Part | What it is | State |
| --- | --- | --- |
| Lookup drafter | Continues the longest earlier occurrence of the current text, in this request or another: one suffix automaton over a corpus that all sequences share (`LookupCorpus`, `LookupDrafter`; after TensorRT-LLM, SGLang and vLLM's suffix decoding), with a confidence per proposed token and a bound on memory. It replaces the n-gram lookup per sequence; llama.cpp's map and cache variants are not built | Done and measured on text; the conductor uses it, next to a model drafter and never alone |
| MTP | The model's own draft head or block | Done for `qwen35` on one device and under a tensor split: the block runs on the device with the embedding table, its chain and its picks stay on the device (`device/mtp.rs`); not under a layer split. Chains of up to the window's rows less one (7 by default) |
| DFlash, DSpark, DFlash2 | A small draft model that reads the target's hidden states at a few layers and proposes a block of tokens at once | Not built: it needs taps on the target's layers and a second model on the same devices. `--drafters FILE` is refused with "not built yet" (the file is not opened); the choice of a window takes any number of drafters |
| Drafter selection | Per step and for all sequences together, whose drafts go into each window, by expected tokens per millisecond of the whole step, from the price of a step and the acceptance measured per drafter, position and sequence; a proposal that was not taken learns from what turned out right | Done: `ares_sched::choose` |
| Adaptive draft length | The length that maximises the same number | Done, in the same call |
| Draft trees | Several branches verified in one window | Not built; the choice is made over rows ranked by their chance, which is a tree's ranking (see `crates/sched/src/window.rs`) |
| Verification | Windows of any length for any number of sequences in one step, then commit | Done |

All drafters produce the same thing, a list of proposed tokens with a chance
that each is accepted, so selection and verification do not depend on where a
draft comes from. A lookup draft is written into the step record by the host.
The draft of a head or a draft model stays on the device, in a table the
verify window reads (SGLang), and reaches the host together with the picks;
carrying it through the host would put a copy out and back between two
kernels of the same device at every step. A draft is cut at the first token
the grammar refuses before it is verified (vLLM).

**The policy.** Four decisions, made as the user decided them and none of them
a flag or a constant of this machine:

- *Drafting is always on.* The operator chooses which **model** drafters take
  part (`--drafters`: the model's own draft block, `mtp`, by default if the
  file has one; `none`; later a block drafter from a file of its own) and the
  room for windows (`--window`). The lookup in earlier text costs nothing to
  ask and takes part as soon as at least one model drafter does, never on its
  own: a server with no model drafter (`--drafters none`, a file without a
  draft block, a layer split, which has no draft block) does not speculate,
  and says so in the plan.
- *The price of a step is a table*, by its rows, because the weight kernels
  read their rows in launches of eight and a ninth row is a second launch
  (25.4 ms of weights for 8 rows of the 27B, 31.0 for 16). `Conductor::warm_up`
  measures it on the devices at load: steps of every row count a window can
  have, shaped as the steps are (windows of several rows that keep their
  inputs, prompt rows that are certain), then the doublings up to a piece of
  prompt, the fastest of two (one above 24 rows), and windows whose draft
  rows the block fills, from 2 rows to the longest window, for the time of its
  chain by length. Every step that runs corrects the entry for its rows by a
  fifth of the difference, so a device that slows down or a context that gets
  deep is followed; a context term (per cell a row reads) is fitted
  separately (`ares_sched::PriceTable`).
- *Acceptance is measured per drafter, per position and per sequence*
  (`ares_sched::Acceptance`): position `i` is accepted *given* that the ones
  before it were, so a draft that failed at position 2 says nothing about
  position 3. A sequence that is new starts from what the engine has seen of
  that drafter at that position (and that from a prior of 0.6 before it has
  seen any), and its own counts, which forget gradually, outweigh the start
  as they come. The lookup's chances are its confidence per token
  (match length and how earlier occurrences went on), which verification
  corrects as it always did.
- *The choice is one number*, tokens per millisecond of the whole step: every
  draft row is worth the chance that it and the rows before it are accepted,
  rows are ranked by that chance across all windows of the step, and the
  step takes the first `m` of them, for the `m` at which the rate is highest.
  The price has edges, so every `m` is tried. A drafter that runs on the
  devices pays its chain once for all windows that use it, however long the
  longest is; which drafter a sequence uses is tried one sequence at a time
  and kept when the rate of the step rises, and lengths are then tried one
  shorter and one longer. In 300 random cases of two sequences, two drafters
  and chains of up to five rows it is the best of all choices in 293 and
  never below 97.9 % of it.

**What limited the chain.** In order: the default of three drafts
(`--drafts`), which was a cap on every sequence's window whatever the text;
the step cost as a line, `base + per_row * rows`, which cannot see that the
ninth row costs a launch and so could not decide where a longer chain stops
paying; the whole step capped at the rows of the windows while anyone drafted,
which is also what made a prompt crawl; and the room the model reserves per
row of a window (the inputs it keeps for the commit, 3.6 MB; the logits,
1 MB; a link of the chain's scratch and the carry of the hidden states: about
5 MiB for the 27B together, for every sequence), which is the one real limit
and so the one option, `--window`. Its default is the rows of one launch of
the few-row weight kernels (`ares_cuda::MAX_BATCH`, 8): inside a launch a
longer window costs rows and no weights, and the table measured at load says
what a second launch costs, so a window longer than a launch is only worth
reserving for text that is copied for many tokens in a row. Everything below
it is the policy's: a window of 8 rows is a possible chain of 7 links, and a
step uses as many as pay.

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
on the device that embeds (which keeps the rows of the tokens the chain can
pick, see "The token embedding of `qwen35`"), each link's pick made by the
sampler with the settings and the noise of the row that will verify it and
written straight into the token array of the step record on the device; then
the verify window, whose embedding reads those tokens; then, for every row of
the step, the block's own keys and values from the hidden states the model
just computed. The host learns the drafts with the picks, counts the accepted
ones and puts the commit into the next record. The conductor chooses per
sequence and step between a lookup draft and the model's own by expected
tokens per millisecond (`ares_sched::choose`), with the cost of a link measured
on the device.

Measured 2026-11-10 on Qwen3.8-27B (tensor split 0.625 / 0.375, RTX 4070 and
RTX 3060, at most three drafts, the fork's draft vocabulary of 66 343 tokens,
300 tokens per request through the chat template with the fork's
`reasoning_effort`): 79, 76 and 79 tokens/s for German prose, English prose
and code with the most probable token, 76, 72 and 83 sampled (temperature 1,
top-k 20, top-p 0.95), against 40.5 without drafts and 62 to 67 for the fork
with its production flags on the same questions. A link of the chain costs
1.46 ms (it reads 335 MiB of the block and 278 MiB of the head), a step 34 ms
for 2.5 to 2.9 tokens; the first draft is accepted in 74 to 83 % of the
steps. The block takes 335 MiB of weights and, with its K and V in `q4` (the only format
it has: 11 000 drafts each with the block's K and V in `q8` and in `q4` were
accepted at the same rate, 67.4 % against 67.1 % for the first draft of a
window and 64.7 % against 65.8 % for the second, greedy, and 70.3 % against
70.3 % and 64.5 % against 66.6 % sampled, `z` below 1.3, and 74.5 / 76.1
against 74.8 / 76.8 tokens per second), 1.1 MiB per 1000 tokens on its device
where `q8` took 2.1 (the same on the `Q4_K_S` file over both cards, 3000
drafts each: first draft 69.9 % against 67.8 % greedy and 63.2 % against 64.9 %
sampled), and nothing for its head, which is rows of the output head. With
exact products the text with drafts equals the text without, for
the most probable token and for sampled tokens with the same seed
(`tests/qwen35_mtp_runtime.rs`, and over HTTP by the hash of the reply's tokens,
`ares-server`'s `tests/real.rs` on the 27B and both cards). With the fast
kernels the texts are not guaranteed to agree: a window of several rows rounds
differently from single rows. Before the few-row kernels the texts parted after
150 to 650 characters (the margins of the picks where they part were not
measured; `tests/real.rs` had two of three questions parting after 300 and 430
characters on one card). Measured 2026-11-12 on the same questions (three
drafts, 300 tokens, the most probable token and sampled with seed 20261110):
with every matrix in the device layout all six texts are the same token for token
(23 and 22 matrices, the i-quants of 2 and 3 bits among them, have no device
layout and run through the integer decode kernels), with every matrix in the
file's layout two of six are, and with the measured choice two of six in one
load and four of six in another, which are not the same model: the choice of
layouts differs from load to load, and more matrices stay in the file's layout
when the model has its draft block (96 and 100 on the RTX 4070 against 37 and
40; the cause was not found).

The cause is not a matrix kernel. A row of a matrix has the same bits alone
and in a step of up to eight rows (sixteen in the device layout), in a launch
of any shape, and so have the norms, merges and activations: rows of different
sequences in one step give the top logit of the rows alone bit for bit, in
either layout, on one card and on two, and launching every product of the
file's layout one row at a time changes no bit. The rows of *one* sequence in
a step do change each other's result: the top logit of a row in a window of
four differs from the row alone by 1.7e-4 of the logit on average in the
device layout, 1.1e-3 in the file's layout and 8.8e-5 with exact products
(Qwen3.8-27B, 16 rows each; 6.3e-4, 2.6e-3 and 2.2e-4 on the 0.8B). It does
not grow with the context (at 8, 100 and 700 tokens: 2.5e-4, 1.7e-4 and 1.8e-4
in the device layout, 3.0e-3, 1.1e-3 and 1.3e-3 in the file's), so it is how
the recurrent or the attention layers take several rows of one sequence (not
isolated further), not the reading of a long cache. The activations are
rounded to f16 or to 8 bits before the next product, which turns a difference
in the last bits of a float into a step of that rounding now and then; the
8-bit step is larger, so the file's layout carries the difference four to
seven times further than the device layout and picks another token more often.
In the device layout the six texts are the same because the difference is
below the margin of their picks, not because it is zero.

Measured 2026-11-11 with the policy on Qwen3.8-27B (tensor split 0.625 / 0.375,
`qwen35_run --engine`, 400 tokens per request through the chat template with
the fork's `reasoning_effort`, the fork's draft vocabulary, most probable
token and sampled with temperature 1, top-k 20, top-p 0.95; the machine at a
load of 12 to 20 from other work). "Today" is the build before this work with
at most three drafts, which is the policy with a window of 4 rows less the
price table and the per-position estimates:

| Tokens per second | no drafts | today | window 4 | window 8 | window 16 |
| --- | --- | --- | --- | --- | --- |
| German prose, greedy / sampled | 38.4 / 40.3 | 78.9 / 77.1 | 79.0 / 78.8 | 79.1 / 85.7 | 76.6 / 75.4 |
| English prose | 40.7 / 40.3 | 73.4 / 74.5 | 83.3 / 67.5 | 86.2 / 70.6 | 87.3 / 69.4 |
| New code | 40.6 / 40.1 | 83.8 / 79.1 | 80.2 / 76.8 | 78.8 / 78.3 | 80.1 / 80.3 |
| A rewrite of a given file | 40.5 / 39.9 | - | 111.0 / 108.7 | 142.3 / 147.8 | 137.7 / 144.5 |

(The numbers without drafts are through HTTP, the others from the tool, which
runs the same engine; the run to run spread of a cell is about 5 %.) The
windows of 4 and 8 rows differ where text is copied: the rewrite verifies
chains of 7 links in six steps of ten and of 6 in three (the draft block is
right 0.85 to 1.00 at each of the seven positions there, the lookup as well)
and gains 28 to 36 % over 3 links, with 5.6 tokens a step against 3.6. Prose
and code gain nothing from a longer chain: their draft block is right 0.6 to
0.8 at each position, so the chance of a whole chain falls with its length
(0.7 to the fifth is 0.17), and the policy chooses chains of 3 to 6 from the
price it measured (a link of the chain costs 1.5 ms, a row a fraction of a
millisecond, next to the 25 ms of the weights). English prose gains 13 to 17 %
greedy over three links; the sampled cells are within the spread. A window of
16 rows is no better than one of 8, so the second launch of the weight
kernels that a ninth row costs does not pay for the room it takes (VRAM for
another 8 rows of every sequence); this is the measurement behind the default
of 8. The study's prediction of +20 % on new code from chains of 7 links did
not appear: the code question's drafts are right 0.72, 0.74, 0.65, 0.67 and
0.70 at the first five positions, and the policy stops at 5.

Adaptive draft length did not pay off in the llama.cpp fork (a few percent up
for English and code, a few percent down for German). Two of its costs there
do not exist here: a new batch shape built a new graph (about 18 ms, and 26
MiB of VRAM per GPU for each cached shape), and a rejected draft past the
rollback rows forced a replay. Measured on this engine, the gain over three
links is where text is copied (the rewrite) and for English prose, and zero
or inside the spread for German and code; nothing is lost where it does not
pay, because the policy drafts what the price and the acceptance say pay.

## Sampling

A request can set what the model's logits go through before a token is drawn
from them, and ask for the probabilities of what was drawn. All of it runs on
the device, in the kernels of `ares-cuda` (`kernels/sampler_*.cuh`) or of
`ares-vulkan` (`shaders/sample_*.comp`, see [Vulkan](#vulkan)), and is
defined by `ares-sampler`, a crate with no device in it: the settings of a
row (`SampleParams`, `SampleRow`), the arithmetic every backend repeats
operation by operation (`arith`), the pick of one row written from its
definitions (`sample_reference`) and the corrections the host works out
(`penalties`). Every test of the kernels compares the records they write with
the reference's bit for bit; Vulkan's are held to the same.

**What a row carries.** Each wanted row of a step has a sampler row in the step
record (128 bytes: position, seed, the stages of its program and their
settings, where its corrections and its mask are, what it reports), and may
have corrections (16 bytes each, at most 4096 a row) and a mask (a bit per
token). The logits are corrected in place before anything reads them. A model
that drafts for itself on the device picks its drafts from the logits as they
are (`SampleRow::for_draft`): a draft is a guess that the verifying row
confirms.

**Corrections.** An entry changes one token: `l = l + bias`, then
`l = l <= 0 ? l * rep : l / rep` (llama.cpp's repetition penalty, which
lowers a logit on either side of zero), then `l = l - sub`; a token the mask
does not allow becomes minus infinity. The host emits them from the row's own
path, the prompt and the reply up to the row's input token and, for a row of a
window that verifies drafts, the drafts before it. That is what makes a window
pick what the sequence picks one token at a time. They are:

- the repetition, presence and frequency penalties of llama.cpp over the last
  `repeat_last_n` tokens of the prompt and the reply (64 unless set, at most
  1024, 0 off). OpenAI counts presence and frequency over the generated tokens
  only; this server counts the window, as llama-server does.
- DRY, ported from llama.cpp's algorithm (the Z algorithm on the reversed window
  and the sequence breakers, which are found once per list of strings by a
  pass over the vocabulary's text) with the same defaults (multiplier 0, base
  1.75, allowed length 2, window 64, breakers `\n : " *`). The penalty is
  added to `sub`, so it is subtracted after the repetition penalty has
  scaled the logit where llama.cpp subtracts twice; the difference is one
  rounding.
- a logit bias: a token and a number, `false` for a token that must never be
  picked, or a text, of which every token gets the bias; `ignore_eos` is a ban
  on the tokens that end a generation.

A request with corrections that depend on the path (penalties, DRY) or a mask
takes no drafts from the model's own draft block: those are made on the device
while the step runs, and the host cannot correct a row for drafts it does not
have yet. It takes the lookup drafts, which the host has, wherever the model has
a draft block (the lookup joins next to a model drafter and never runs alone).

**The chain.** A sampled row starts with the set of every token that has a
finite logit, ordered by the distance below the best logit (a token's *key* is
the bits of that distance, which order like it). The row's program is a list
of at most 16 stages; each cuts the set to the tokens with keys from `lo` to
`hi`, or scales the temperature `T` the logits are read at. `x` below is the
distance below the best token of the set divided by `T`; a token's probability
within the set is `exp(-x)` over the sum of that.

| Stage | Keeps | Found on the device by |
| --- | --- | --- |
| top-n-sigma `n` | `d <= n * sigma`, sigma the standard deviation of the distances `d` of the set | sums of `d` and `d^2` as integers |
| top-k `k` | the `k` most probable and every token tied with the last | radix descent over the keys, by count |
| typical `p` | the tokens within the smallest radius of `x` around the mean of `x` (weighted by probability) that holds more than `p` | sums, then a descent over the radii by weight, then the least and greatest key inside |
| top-p `p` | the most probable that hold at least `p` of the set | radix descent by weight |
| min-p `p` | at least `p` times as probable as the best of the set | one threshold, no pass |
| XTC `prob`, `thr` | with probability `prob`: removes the tokens more probable than the least probable that has `thr` or more, if there are two | a draw, sums, a scan |
| temperature `t` | sets `T = t * T`; a `t` below 1e-6 keeps the tokens tied for the best | arithmetic; a dynamic temperature (range `r`, exponent `e`) first sums the entropy `H` of the set and takes `t` between `max(0, t-r)` and `t+r` by `(H / ln n)^e` |
| mirostat 2 | the tokens whose surprise `-log2 p` is at most `mu`, or the best | sums, then a search of the key |
| mirostat 1 | the top-`k` of the `k` that the hundred most probable tokens and `mu` imply | descent for the hundredth key, a gather of the keys below it, arithmetic, descent for `k` |

The order of the program is the request's `samplers` (llama.cpp's default
order: penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC,
temperature); a stage that is not listed is not applied, one that is listed
twice is applied twice. Penalties and DRY correct the logits and so come
first or not at all. A stage whose setting switches it off is left out of the
program. Mirostat replaces the chain, as it does in llama.cpp, which then
applies neither penalties nor DRY; the logit bias applies to every chain.

The pick is the token of the final set with the largest `logit / T + noise`,
the noise being Gumbel of (seed, position, token). A row with temperature 0
(and no dynamic temperature or mirostat) takes the most probable token of the
corrected logits, the lower token on a tie, whatever the stages are.

Where this differs from llama.cpp: tokens tied at a cut are all kept (llama.cpp
keeps the first by its sort order); the random numbers are not its generator's,
so a seed does not give the text llama.cpp gives, only the same text every time
here; a temperature of 0 takes the best token even where a stage before the
temperature would have removed it; the probabilities are fixed-point sums, so
a cut can fall one token away from where llama.cpp's float sums put it when a
token sits within 1e-6 of it; `min_keep` above 1 is refused (a cut would need
the count of the tokens it left); adaptive-p is refused (see below); a window
longer than 1024 is refused.

**Accuracy.** Everything that is summed over tokens is an integer (a count, the
weights as fixed-point numbers with 40 fractional bits, the entropy term, the
sums for the deviation), so the order in which threads, blocks and devices add
does not matter; the exponential and the logarithm are polynomials of IEEE
operations, in single precision per token and in double precision per row. The
cross-device sums are numbers split into limbs of 16 or 20 bits that add
exactly as `f32`. That is why the device's records equal the host's bit for
bit.

**Mirostat and drafts.** Mirostat's `mu` moves with every token by the
probability of the token picked, which a row of a window cannot know before the
row before it is picked. A request with mirostat therefore has windows of one
row; its `mu` is the host's and the device reports the surprise of each pick.
Adaptive-p has the same dependence and is not built.

**Drafts and the text.** Every feature gives the same tokens, text and
probabilities with drafts as without them (a request that says `"draft": false`
is the one to compare with), and a request does not change what the requests in
the same steps get. The synthetic model of `ares-runtime`, whose device part
drafts what it has seen the model pick, proves it bit for bit on every
placement, with a rule about the tokens as well
(`every_sampler_feature_gives_the_same_tokens_with_and_without_drafts_on_every_placement`
and `requests_with_different_settings_in_one_step_do_not_affect_each_other_on_every_placement`
in `crates/runtime/tests/gpu.rs`), and `crates/runtime/tests/sampling.rs` checks
the records of the windows against the definitions. The lookup in earlier text
drafts only next to a model drafter, so the tests on a real model use the 27B and
its draft block, on both cards with exact products. There the texts were equal
with drafts and without in all 14 settings through the server
(`crates/server/tests/real_sampling.rs`: 437 draft rows verified, 425 of them
right), equal under two rules about the tokens, and equal for requests that run
next to each other. A model is not exact in general: the logits of a row depend
on the size of the step that computed it by up to 2e-3 ("Rounding" under
"Sequence state"), so on a model or a placement where that shows, a sampled
text can part where two tokens are that close; the tests assert that the
texts agree for a long stretch and that the probabilities agree on it within
the model's error.

**Probabilities.** A request asks for the probability of each generated token
and of the `top` most probable alternatives (at most 20). Two distributions:

- *the model's own*: the softmax of the logits as the model gave them, before
  the corrections and the chain (OpenAI's `logprobs`, llama.cpp's `n_probs`;
  vLLM's default). The device takes the best twenty and the sum of the
  weights from the uncorrected logits before it corrects them, and the pick's
  uncorrected logit from the entry that corrected it.
- *after the chain*: the probability within the tokens the chain left, at the
  temperature it ended with (llama.cpp's `post_sampling_probs`). A row that
  takes the most probable token has the tokens tied for it.

OpenAI says nothing of a distribution; vLLM and llama.cpp take the model's own
by default, and so does this server. A probability whose logarithm is below
-100 is not reported. The probabilities of a token go out with the text of
the token, once the characters are whole and no stop string holds them back; a
reply that ends at its token limit inside a character reports the tokens of the
half character with the last delivery, so that it reports as many tokens as it
has (one that a stop string cuts does not report the tokens of the string).

**Cost.** The time of the sampler's kernels for one step with 248 320 logits, from
the launch to the end of the stream, in microseconds (medians of 40 runs, the
first number for 1 row, then 4 and 16; `the_time_of_a_pick` in
`crates/cuda/tests/sampler.rs` prints them). The first three lines were also
measured on the sampler before this piece, in the same minutes and with the
same load; the others have no earlier figure. Corrections are 64 entries a
row, a mask allows a third of the tokens, probabilities are the five best.

| Row asks for | RTX 4070 before | RTX 4070 now | RTX 3060 before | RTX 3060 now |
| --- | --- | --- | --- | --- |
| most probable token | 38 / 81 / 241 | 40 / 82 / 245 | 56 / 138 / 386 | 58 / 140 / 391 |
| top-k 20, top-p 0.95 | 124 / 211 / 556 | 125 / 201 / 546 | 156 / 384 / 1138 | 155 / 380 / 1133 |
| top-p 0.95 | 88 / 158 / 442 | 90 / 156 / 448 | 130 / 276 / 832 | 130 / 294 / 876 |
| most probable token, penalties | | 40 / 84 / 240 | | 58 / 142 / 390 |
| most probable token, a mask | | 42 / 86 / 260 | | 64 / 160 / 452 |
| top-k, top-p, penalties and a mask | | 124 / 202 / 541 | | 163 / 406 / 1190 |
| most probable token, probabilities | | 94 / 209 / 624 | | 146 / 417 / 1232 |
| top-k, top-p, probabilities after the chain | | 142 / 233 / 610 | | 211 / 481 / 1405 |
| top-n-sigma, typical, top-k, top-p | | 172 / 275 / 700 | | 244 / 581 / 1732 |

A request that asks for nothing of the new features costs what it did: the
most probable token is 1 to 5 percent slower than before (2 microseconds for
one row, the size of the noise of the measurement), top-k with top-p is the
same or faster, and top-p alone is 5 to 7 percent slower on the RTX 3060 for 4
and 16 rows (not isolated; the histogram kernel does more for each token than
the old one, a test of the set it works on and the scaling by the temperature
that a chain with a temperature before the cut needs). Penalties are free. A mask costs 6 percent on the RTX 4070 and up to 16 percent on the
RTX 3060 for the most probable token. Probabilities of the model's own
distribution add a pass over the logits: 54 microseconds for one row and 380
for 16 on the RTX 4070 (88 and 840 on the RTX 3060). Top-n-sigma and typical
together add 47 microseconds for one row and 154 for 16 on the RTX 4070.

**Requests.** The server reads the fields under the names of the OpenAI API and
of llama.cpp's server: `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`
(`typical`), `top_n_sigma`, `xtc_probability`, `xtc_threshold`,
`dynatemp_range`, `dynatemp_exponent`, `mirostat`, `mirostat_tau`,
`mirostat_eta`, `repeat_penalty` (`repetition_penalty`), `repeat_last_n`,
`presence_penalty`, `frequency_penalty`, `dry_multiplier`, `dry_base`,
`dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`,
`logit_bias` (both forms), `ignore_eos`, `samplers` (names or letters), `seed`,
`stop`, `n` (`n_cmpl`) and `max_tokens` (`max_completion_tokens`, `n_predict`);
for probabilities `logprobs` and `top_logprobs` (chat), `logprobs` (completions,
OpenAI's older form of `tokens` and `top_logprobs`), and `n_probs` with
`post_sampling_probs`. Unset means the model's default for the numbers the
model owns (temperature, top-k, top-p, min-p), and llama.cpp's for the rest.
`n` completions are `n` requests with the seeds `seed`, `seed + 1`, ...; the
engine reads the prompt once and the other requests start from what it
cached. What is refused, by name: `min_keep` above 1, `repeat_last_n` or
`dry_penalty_last_n` above 1024 or -1, more than 1024 biased tokens, more than
20 alternatives, `n` above 16, `adaptive_target`, `adaptive_decay`, the sampler
names `infill` and `adaptive_p`, and a `samplers` list that puts penalties or
DRY after a sampler that cuts. Structured output is read from `grammar`,
`json_schema` and `response_format` (below), and the calls of tools are held
to a grammar the server writes (see [Server](#server), "Tool calls").

**The hook.** A request can carry a `TokenFilter`
(`Runtime::submit_filtered`): before each step the conductor asks it, for every
wanted row, which tokens may follow the reply and the drafts before the row, and
writes the answer as a mask into the record; it may also say how many of the
drafts it allows, and that it has failed. Grammars are one such filter.

**Structured output.** A request can carry a `Structure` (`ares-api`): a rule
for the text of the reply, a grammar in GBNF or a JSON schema, and where in the
reply the rule starts to bind. The engine turns it into the set of tokens that
may come next at every row, writes that set as a mask into the step record, and
the sampler applies it before the draw like any other mask (above), so a reply
follows the rule from the first token to the last or the request ends in an
error. `Runtime::submit_filtered` still takes any `TokenFilter`; a request with
a `Structure` gets one made by the runtime.

*The engine.* `ares-grammar` wraps llguidance (MIT, crates.io, version 1.9;
Cargo.toml names it with the thread pool of its C interface left out; it builds
offline once fetched): an Earley parser over lexemes that a lexer built on
derivatives of regular expressions finds, with the token mask computed by
walking the vocabulary's trie. It is suitable because it takes the vocabulary
as bytes per token, which is what `Model::piece` gives, and needs nothing but
that: the tokens a control token stands for are empty and no grammar allows
them, and the end tokens are allowed exactly where the grammar can end. Reading
the vocabulary of 248,320 tokens takes about a second; the runtime does it on a
thread of its own when it starts, and a request with a structure waits for it
if it comes sooner. Compiling a grammar takes one to two milliseconds.

*Kinds.* `grammar` is GBNF as llama.cpp reads it (`root`, literals and
classes, groups, `*`, `+`, `?`, `{m,n}`, `.`, `<[id]>` and `!<[id]>` tokens,
comments, the same rules for where a line ends a rule); it is translated to
the engine's Lark syntax, with the position of a syntax error (line and column)
in the message. Left recursion, which llama.cpp refuses, works. `json_schema`
and the schemas of `response_format` go to the engine's JSON compiler, in
llama.cpp's way: keywords it does not know are left out (the engine's default
refuses them), white space between the parts is llama.cpp's own (one space, or
line breaks and an indent of up to 20, so a model cannot fill its reply with
blanks), `$ref` is followed inside the schema and a remote one is refused (the
server fetches nothing). An empty schema is any object, as llama-server reads
it.

*Where the rule starts.* A model that thinks is free while it thinks. The
server knows the format from the chat template (`ThinkingFormat`) and sets
`Start` for the request: from the first token when the template has no
thinking or the prompt has closed the block (thinking off); after the closing
tag when the prompt has opened the block (`<think>\n`); and for a template that
writes nothing into the prompt, either the opening tag followed by reasoning
or the rule from the first token. llama-server does this for `response_format`
and `json_schema` (it builds the grammar around the reasoning parser) and binds
a plain `grammar` to the first token, which keeps a thinking model from
thinking; this server binds all three after the reasoning, and
`enable_thinking: false` gives the other behaviour. The answer starts at the
token after the closing tag: the blank line the template writes there is not
allowed to the model, and a closing tag that shares a token with the start of
the answer is not checked against the rule for the rest of that token.

*A rule on a trigger.* A `Structure` may name a trigger: the answer is then
free from where the rule would start until its text contains the trigger, and
the rule binds what follows it; a reply that never writes the trigger stays
free to its end, and the rows before the trigger carry no mask. The rule is
checked from the token after the one that completes the trigger, as it is after
the closing tag of the reasoning: what the same token holds after the trigger is
not checked, and the tags in question are tokens of their own in the
vocabularies of the supported models. This is what llama.cpp calls a lazy
grammar and what the calls of tools use; the third kind of rule, a grammar in
the engine's own Lark syntax (`Shape::Lark`), is the one the server writes for
them. No field of a request sets a trigger.

*Drafts.* The text with drafts is the text without. Every row of a window gets
the mask of its own path, so a draft the rule forbids is never the pick of the
row before it and is rejected whatever happens. On top of that the conductor
cuts a forbidden draft, and the ones after it, from the lookup's proposal
before the window is chosen (`TokenFilter::admitted`), so the rows are not paid
for: a test with runs of one letter has every draft verified accepted, and 74 of
119 without the cut. A request with a rule takes no drafts from the model's own
draft block (their tokens are not known when the record is written, as for
penalties). Tokens the rule forces (one token allowed) are not offered as
drafts: the lookup is the drafter that joins next to a draft block, and a
forced token would be a third source with a chance of one; it is not built.

*Cost.* Host time, on the vocabulary of the 0.8B (248,320 tokens), one core of
the machine (`cargo run --release -p ares-grammar --example mask_cost`): reading
the vocabulary 0.5 s, once per runtime; compiling a JSON schema of moderate size
(eight properties, a pattern, enums, a nested object, an array) 0.7 ms; the mask
of a step along random walks through it a median of 1 microsecond, 61 at the 90th
percentile and 1.9 milliseconds at the longest (a step that opens something the
engine has not seen); `json_object` a median of 2 microseconds. A GBNF grammar
is read one character at a time, as in llama.cpp, and costs more: llama.cpp's
`json.gbnf` takes 6.8 milliseconds a mask (compiling 1.2 ms), a grammar of a few
literals 3 microseconds. The mask is worked out on the conductor's thread before
the step is launched, one row after the other, so it adds to the step; working it
out during the device's step (vLLM) is not built, and would hide the 6.8
milliseconds of a character grammar behind a step of a large model. Merging the
rules that are one lexeme (a string in quotes) into regular expressions was
tried and gave nothing for `json.gbnf`, whose string rule ends in optional white
space. The device applies the mask in the sampler, which costs what the section
on masks above says (6 to 16 percent of the draw of the most probable token).

*Errors.* A grammar or schema the engine refuses is a 400 that names the field
and the problem; both `grammar` and `json_schema` (or `response_format`) in one
request is a 400, where llama-server takes one of them silently. If the engine
fails in the middle of a reply (it ran out of the fuel it gives a grammar, or
the grammar allows no token and the model has no end token) the request ends
with an error and the text is not sent.

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
- **Reading from SSD** (`ares-io`; the expert store and the n-gram rows of
  `qwen4exp` read through it): experts
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

  What is missing is the conductor that runs the pieces and keeps the hidden
  states of all prompt rows between them, and a model with experts. The
  kernels that compute a wave for the rows routed to it are built (see
  "Many-row expert kernels" under "Devices").
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
  that needs them, behind a row cache (`ares-io`; `qwen4exp` does this).

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

**Few rows on the device layout** (`ares_mmr_few` in `kernels/mmr.cuh`, called by
`Kernels::product` for up to 16 rows of a matrix in the device layout). One
tile of 16 weight rows at a time on `mma` with f32 accumulation, the codes
decoded in registers and the next step's weights asked for one step ahead.
Its arithmetic is a function of the matrix and of the row's own activations
and nothing else: K is cut into 8 parts of whole super-blocks (fewer
non-empty ones for a matrix of fewer than 8), each part is summed from zero by
`mma`, and the parts are added in the order of K. A tile is computed by 1, 2,
4 or 8 warps, each taking an equal share of the parts (the others' parts reach
the first warp through shared memory), a block holds 4 or 8 warps, and which
of the four launches is the fastest depends on the shape and the card:
`Kernels::tune_layout` times them for each shape at load (the model's chooser
does it over its probe buffer, as large as the weights, because a small matrix
timed over a few dozen megabytes finds its address translations and the
launches that wait for them look better than they are). None of the launches
changes a bit: a row has the same bits alone, in a window of up to 16 rows, with
other rows beside it, in each launch, tuned or not
(`tests/layout.rs::a_row_is_the_same_bits_in_any_batch_and_launch`). 0.04 %
rms from the exact f32 product, as before (the f16 operands). Against the
integer decode kernels of the file's layout, one row, in the model's probes
(cold address translations, 59 shapes per card), the sum over the matrices of
a step is 98.7 % (RTX 4070) and 95.4 % (RTX 3060) of the integer kernels'
time, and single shapes run from 89 % to 111 %: a few i-quant and Q5_K shapes
of the RTX 3060 and the matrices of 256 to 768 rows are 2 to 11 % slower;
the walk of `examples/few_rows_bench.rs` (one shape, rows 1 to 16, `--plans`
for each launch) is the tool for a single shape. Cutting K the same way for
every launch costs nothing measurable: the same launch with a cut per
launch was 0 to 5 % slower at the median.

**Prefill weight kernels.** A quantized matrix times many rows of
activations, on the tensor cores, for eleven types (`kernels/mm.cuh`,
`mm_types.cuh`, `mmr.cuh`, `prefill.cu`) and, in a second module with the f16
paths only (`prefill_more.cu`, `mm_types_more.cuh`; below), fourteen more.
Three paths,
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

**The second module** (`prefill_more.cu`, `mm_types_more.cuh`): iq2_xxs, iq1_s, iq1_m,
q2_0, q2_K, q4_0, q4_1, q5_0, q5_1, tq1_0, tq2_0, mxfp4, nvfp4 and q1_0. The same tile
kernels over the file's layout, with the f16 and the f16-accumulation paths and no
int8 one: for IQ2_S the f16 path with f16 accumulation is as fast as int8 (73.4
against 65.5 TOPS on the RTX 4070, 31.4 against 29.2 on the RTX 3060) and 20 times
more exact, so an 8-bit form of these types was not built. IQ2_XXS, IQ1_S and IQ1_M
hand the tile kernel their decoder as `decode_f16` (`AresMmDirect` in `mm.cuh`):
the item's 32 values come out as f16 pairs straight from the codebook entry, the sign
masks and the scale, with the decoders of the many-row expert kernels
(`expertrows_types.cuh`; IQ1_S shares the group of IQ1_M, `ares_er_iq1_values`),
where the others go through codes and scales as the first module does (`ares_mm_codes_to_f16`). Measured
2026-11-12 (`prefill_bench`, 17408 x 5120, 2048 rows, TOPS, RTX 4070 / RTX 3060; load
from other jobs on the machine does not touch a GPU kernel):

|          | f16 (f32 accumulation) | f16 with f16 accumulation | decode kernels, 8 rows |
|----------|------------------------|---------------------------|------------------------|
| iq2_xxs  | 52.4 / 23.0            | 71.7 / 31.1               | 14.1 / 6.7             |
| iq1_s    | 53.2 / 22.9            | 81.4 / 34.3               | 12.6 / 5.7             |
| (iq2_s)  | 52.0 / 22.5            | 73.4 / 31.4               |                        |
| (q4_K)   | 55.3 / 23.5            | 77.9 / 33.9               |                        |

The other twelve, f16 with f16 accumulation at 1024 rows (RTX 4070 / RTX 3060):
iq1_m 84.1 / 35.3, q2_0 81.9 / 34.9, q2_K 76.4 / 32.9, q4_0 73.7 / 32.2, q4_1 73.8 / 32.1,
q5_0 66.9 / 29.5, q5_1 66.3 / 29.1, tq1_0 65.1 / 27.9, tq2_0 79.1 / 33.7, mxfp4 75.2 / 32.5, nvfp4 74.6 / 32.4, q1_0 81.6 / 34.8.

IQ2_XXS and IQ1_S are in the range of the types that had kernels (5.1 times as fast as the
decode kernels in launches of eight rows for IQ2_XXS, 6.5 for IQ1_S), and no device
layout was tried for them: the rearranged form of Q4_K above was not faster than the
tile kernel at prompt sizes and these kernels are within 8 % of Q4_K's rate, so there is
no gap for a layout to close, and Vulkan has nothing to mirror (the layouts stay those
of the file). Their rows of `DECODE_ROWS` (the decode kernels win up to 16 rows for IQ2_XXS, Q2_0 and Q4_0,
up to 12 for IQ1_S and IQ1_M; Q2_K and Q5_1 have none, since theirs win only up to about 10
rows, nor have the types not measured, so above eight rows those take the tile kernels) are
from the same bench at 8 to 32 rows.

The 2-bit file of Qwen3.8-27B (`Qwen3.8-27B-UD-IQ2_XXS.gguf`, 2048 tokens, one card,
`qwen35_run --prefill 2048`): the prompt went from 267 to 1154 tokens per second on the
RTX 4070 and from 128 to 536 on the RTX 3060 (the llama.cpp fork: 1011 to 1108 and 563);
the decode step at 1000 tokens of context did not change (21.03 and 21.06 ms, 36.53 and
36.51 ms), the top-1 agreement with llama.cpp on the German and English prompts is
96.4 / 97.8 % (RTX 4070) and 96.8 / 98.7 % (RTX 3060) after, 96.0 to 99.1 % before. With
only IQ2_XXS and IQ1_S the prompt ran at 1104 and 521 tokens per second; the other 4 %
and 3 % are the three attention blocks with q2_K matrices, which took the whole group of
matrices that read their input through the decode kernels. The Q4_K_S file on both
cards (0.625 / 0.375) took 1253 and 1278 tokens per second, 24.72 and 24.81 ms per decode
step, which is the noise of the layout probes. On the RTX 3060 the weights are 88 % of
the pass (profile of the first version per step of 512 rows: weights 860 ms, recurrent
layers 86, attention 34), at the 31 to 34 TOPS that every tile kernel reaches there (Q4_K
34), IQ2_XXS being about 43 % of that time and IQ1_S 28 % by their rates and numbers of
values; the llama.cpp fork's 563 tokens per second would need about 5 % less time than
the 536. On the RTX 4070 the weights take 1.62 of the 1.87 seconds.

**Float matrices** (f32, f16, bf16: `EngMmFloat` in `kernels/mm_types_more.cuh`, the second
module). They have the two f16 variants and no int8 one, since they have no integer codes,
and make the last three of the 28 types with a tile kernel. An item of 32 values is read as
16-byte words (the matrix must be 16-byte aligned, `n_in` a multiple of 32) and converted as
the tile is made: f16 is copied, bf16 is shifted to the top of an f32 and rounded by `cvt`,
f32 rounded by `cvt`; there is no scale. Before this, a dense model with such a matrix
multiplied it, and every matrix that read the same input, with the decode kernels in
launches of eight rows.

*Range.* An f16 holds 6.1e-5 to 65504 in full precision, down to 6e-8 with fewer bits. The
2-D float tensors that a product reads, in the models at hand (every value of every tensor
measured): the Q4_K_S file of Qwen3.8-27B has none, only vectors (norms, `ssm_a`,
`ssm_dt.bias`) and the convolution taps; its IQ4_XS file has 96 f32 matrices, `ssm_alpha`
and `ssm_beta` of 5120 x 48, whose values are f16 values (largest 0.06 to 0.33, converting
them to f16 changes nothing). Qwen3.8-Flash-Next has 436: 388 bf16
(`hc_{attn,ffn}_{up,down,inject}`, the indexer's `q_proj` and `k_proj`, `ssm_alpha`,
`ssm_beta`, `ple_key`, `ple_value`, `output_hc_*`) and 48 f32 (`ffn_gate_inp`, the router),
whose values are bf16 values too (their low 16 bits are zero). The largest value of any of
them is 8.56 (per matrix 0.16 to 8.56), 7600 times below the overflow. In a matrix 0.06 to
2.6 % of the values are below 6.1e-5, and 0 to 0.5 % (all but `ssm_alpha` below 0.005 %)
below 3e-8, which become zero. Converting straight to f16 changes the weights by 1e-9 to
1.9e-7 of their rms: bf16 values in the normal range are exact in f16 (8 bits of mantissa in
11), so only those small ones are not. A scale per matrix (a power of two that puts the
largest value at 0.5 to 1) was measured and lost: these matrices are already that size, and
it moved the error from 4e-9..1.9e-7 to 4e-9..3.7e-7 (it pushes the matrices whose largest
value is 4 to 8 below the range of normal numbers by a factor of 4 to 8). A scale per row of
weights would take the worst to 3e-8 (the rows of `ssm_alpha` differ in size by up to
3.5e5), at the price of a vector of exponents per matrix and a pass over the matrix to make
it; it is not worth that for an error this far below the activations' rounding. So the
kernels take no scale and are never silently wrong by range: a weight beyond 65504 is an
infinity in the tile, so the outputs that read it are infinite or NaN (tested for bf16 and
f32), and the variant with f16 accumulation, which adds partial sums of 64 products of a
weight and a value of at most 1, does the same from weights of about 1000. What a matrix
loses below 6.1e-5 is absolute: at most 2^-25 per weight, which the test
`tiny_float_weights_are_off_by_at_most_the_spacing_of_f16_subnormals` bounds on weights from
1e-7 to 1e-4 (3.1e-6 measured on 512 values against the bound 2.0e-5). On other models a
float matrix whose values do not fit has to be looked at before the tile kernels are used on
it; nothing in the engine checks that on load.

*Error against the exact f32 product* (`tests/prefill.rs`, weights of rms 0.05 with
outliers, activations of rms 1 with outliers, shapes from 1 x 32 to 200 x 1024): the f16
path 0.02 to 0.03 % rms for f16 and bf16 weights (the rounding of the activations; the
weights are exact), 0.02 to 0.04 % for f32 weights; the path with f16 accumulation 0.04 to
0.06 %. Against the same product with the operands rounded to f16 and summed exactly, the
f16 path is within 3e-6 of the magnitude of the sum and the flush path within 1e-3.

*Speed* (`examples/prefill_bench`, a 17408 x 5120 matrix, ms, RTX 4070 / RTX 3060, machine
load 29 to 38 from other builds; the last column is the decode kernels in launches of eight
rows, which is what the model did before):

| rows | f16 path | f16 accumulation | eight-row launches |
|---:|---|---|---|
| 16 | 0.40 / 0.63 | 0.40 / 0.63 | 0.78 / 1.41 |
| 64 | 0.43 / 0.65 | 0.43 / 0.63 | 3.13 / 5.64 |
| 256 | 0.94 / 2.19 | 0.73 / 1.57 | 12.4 / 22.5 |
| 1024 | 3.71 / 8.52 | 2.80 / 6.32 | 49.8 / 90.5 |
| 4096 | 14.7 / 34.2 | 11.0 / 25.3 | 199 / 362 |

(bf16; f16 is within 3 % of it everywhere.) f32: 0.83 / 1.18 at 16 rows, 1.41 / 2.98 at
256, 5.29 / 11.9 at 1024 and 20.6 / 47.8 at 4096 with the f16 path; 5.73 / 12.4 and 22.3 /
50.2 with f16 accumulation, which is not faster for f32 (so the tensor cores are not what
bounds it; the large shape of f32 uses all 128 registers a thread may have, with 64 of them
holding the raw bytes of the next two items); eight-row launches 1.53 / 2.29 at 16 rows, 24.3 /
36.6 at 256, 97 / 147 at 1024 and 389 / 589 at 4096. In TOPS at 4096 rows: bf16 and f16 50 / 21
(f16 path) and 66 / 29 (f16 accumulation), f32 35 / 15 and 33 / 15, of 58.6 / 25.6 that the
capped tensor cores run. Up to 8 rows the decode kernels are as fast or faster (0.39 / 0.72 ms
for bf16 at 8 rows against 0.42 / 0.76; 0.056 against 0.107 ms for 320 x 10240) and compute
the exact product; from 9 rows on the tile
kernels are faster for all three types (0.40 against 0.77 for two launches), so the float
types have no row of `DECODE_ROWS`. Flash-Next's mixers (bf16, 512 rows):
`hc_{attn,ffn}_down` 320 x 10240 3.10 / 4.02 ms in launches of eight rows, 0.31 / 0.44 ms on
the tile kernels' f16 path and 0.21 / 0.30 with f16 accumulation; `hc_{attn,ffn}_up` 10240 x
320 1.21 / 2.55 ms, 0.097 / 0.22 and 0.081 / 0.18 ms.

A model calls them through `Kernels::product` (matrix in `WeightLayout::File`, with the
activations in the form `Kernels::activation_form` names and `prepare_activations` makes for
the same arguments): up to 8 rows with `Precision::Fast` it is the decode kernel, above that
the tile kernel of `prefill_path_for`, and with `Precision::Exact` always the f16 path.

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

**Many-row expert kernels** (`kernels/expertrows*.cuh`,
`ares_cuda::ExpertRowKernels`). The experts of a MoE layer for any number of rows,
on the tensor cores, for the types of Qwen3.8-Flash-Next (gate and up IQ2_S,
IQ2_XXS, IQ1_M; down Q2_0). The decode kernels above take eight rows; these take
as many as there is memory for, and are meant for prompts and windows of drafts.
Four stages, all on the device, none needing the host to read the picks:

- **Plan** (`plan_rows`). From a route record of any number of rows (the ids and
  weights the router writes, `n_rows * top_k` of each) a stable counting sort by
  expert: the pairs (row, pick) of an expert follow each other in the order of
  their numbers, and each expert's run is cut into items of at most `tile` pairs
  (8, 16, 32 or 64). One launch for up to 1024 pairs, three above (a block per
  1024 pairs counts and ranks with `__match_any_sync`, one block turns the counts
  into positions and items, a block per 1024 pairs scatters). The plan lists the
  first position of each expert, the items `(expert, first, count)`, the row and
  router weight at each position and the position of each pair; a pair whose id
  is out of range is dropped. The kernels are launched for the most items a plan
  can have and the surplus blocks return at once, so nothing waits for the count.
- **Gate and up** (`rows_gate_up`). A block takes an item and 64 of the expert's
  hidden values, a warp 16: 16 rows of gate and the same 16 of up against all the
  pairs of the item, so the weights are read once for all of them, from the
  address a table of one `u64` per expert gives (0: not present). The weights are
  the A operand of `mma.m16n8k16` and are decoded in registers from the file's
  bytes into f16 (the codebook magnitudes come out of the bytes as `1024 + m`
  with the trick of Marlin and one fused multiply-add with the scale; the signs
  are XORed into the sign bits from a table of masks in shared memory), the
  activations the B operand, gathered by `cp.async` from the f16 rows of
  `Kernels::prepare_prefill_activations`. A thread of a quad takes one sub-block of
  32 values of every 128, which is what its two `mma` of a step need, since the
  order of K inside an `mma` is free if both operands use the same. The epilogue
  makes `silu(gate) * up` in f32, divides it by a power of two per pair and 64
  values (so nothing overflows an f16) and stores f16.
- **Down** (`rows_down`). The same with Q2_0: a word of 16 codes gives the pairs
  (value j, value j + 8) of the operand with one shift and mask, so the hidden
  values are stored in that order. The product of a block of 64 is multiplied by
  the pair's scale as it is added to the accumulators.
- **Sum** (`Accumulate`). `Atomic` adds the weighted output to the row with
  `red.add` (no scratch; the order in which a row's picks are added depends on the
  schedule). `Ordered` writes it to a scratch of `top_k * n_embd` f32 per row
  (102 KB a row for Flash-Next) and `rows_combine` adds a row's picks in the order
  of the router: a row then has the same bits alone, in a batch and in any other
  place of a batch (tested), as the decode kernels guarantee.

An expert of the plan without an address is not read: its pairs add nothing, are
counted in the plan (`missing_pairs`) and the expert is flagged, so the caller
knows what to fetch. One plan serves several calls with different tables, which is
how the waves of a layer run.

Needs compute capability 8.0 (the module is built like the prefill module; a
Turing card has no tile kernels for experts either).

These kernels do not wait for an expert on the device and have no host share: they skip
an expert without an address and report it. The decode kernels therefore stay for the
steps of a generating sequence, which depend on both, and for the windows below the
break-even (which is at most eight rows, the decode kernels' own limit, so every step of
more than eight rows is above it). The slices of a prompt of Qwen3.8-Flash-Next run on the
new kernels (see "Wide steps" under "Qwen3.8-Flash-Next through the runtime"); the host
gives each round its address table and brings what a table lacked.

Against the exact result (f64 on the dequantized weights and the f32 activations) the
output is 0.05 to 0.07 % rms off, and the hidden values the same, on generated blocks of
all four types and on the real tensors of one layer of each type (the same figures on both
cards). Rows scaled by 1e-4 to 1e5 within one call stay within 0.2 %. The decode
kernels, which quantize the activations and the hidden values to int8, are at about 1 % in
the matrices of their family (their expert test allows 5 %).

**f16 or int8 activations.** f16, with f32 accumulation. The reasons, measured: the dense
kernels of the same cards and the same type (`prefill_bench`, IQ2_S, 17408 x 5120, 1024
rows) reach 66.5 TOPS with int8 and 52.0 TFLOPS with f16 and f32 accumulation on the RTX
4070 (29.5 and 22.6 on the RTX 3060), and 73.4 (31.5) with f16 accumulation, so int8 is not
the fast path for this type. For the experts it would also need the scale of every 16 values
(IQ2_S) or 32 (IQ2_XXS) applied to the integer sum of each `mma`: either the four threads of
a quad share one sub-block and each loads the same words (four times the load instructions
of the layout above), or the scale is applied to every result at about three instructions
per output. Accumulating in f16 and flushing to f32 every 64 values was built and measured
on the experts: gate and up at 4096 rows 14 % faster (8.1 against 7.0 ms on the RTX 4070,
synthetic weights), the output 8e-4 off from the f32 accumulation; it was not kept, since
the kernels are not bound by the tensor cores (a loop of `mma` alone gets 58.6 of 58.6
TFLOPS here; the kernels get 33 useful, 40 counting the padding of the last item of an
expert).

Measured 2026-11-11 on the recorded routes of `qwen4exp` (`examples/expert_rows_bench.rs`:
`long`, `mixed`, `german` and `english` of `/var/tmp/ares-qwen4exp-trace`, token after token),
the real experts of one layer, all that the rows pick in device memory, random activations.
Microseconds for one layer's block (IQ2_S gate and up, Q2_0 down), "block" being plan, gate
and up, down and, for the ordered sum, the combine, run one after the other; the tile is
the best of 8, 16, 32 and 64. "Decode kernels" are `ExpertKernels::experts` in launches of
eight rows (the router and the host-built record are not in their time). GB/s are the weights
read once (every expert the rows pick) over the block time, TFLOPS are the useful products
of the three matrices over it; the cards read 473 and 337 GB/s in a loop of coalesced
reads and compute 58.6 and 25.7 TFLOPS in a loop of `mma` (f16 with f32 accumulation).

RTX 4070, IQ2_S gate and up (blk.0):

| rows | tile | plan | gate and up | down | combine | block, atomic | block, ordered | rows/s | GB/s | TFLOPS | decode kernels | times faster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 7.4 | 62 | 19 | 10 | 83 | 94 | 12106 | 183 | 1.2 | 62 | 0.76 |
| 2 | 8 | 6.7 | 105 | 32 | 9 | 144 | 153 | 13896 | 199 | 1.4 | 101 | 0.70 |
| 4 | 8 | 6.8 | 178 | 54 | 10 | 242 | 252 | 16512 | 231 | 1.6 | 213 | 0.88 |
| 8 | 8 | 7.0 | 288 | 80 | 10 | 390 | 403 | 20506 | 244 | 2.0 | 361 | 0.93 |
| 16 | 8 | 7.3 | 503 | 175 | 10 | 690 | 713 | 23182 | 241 | 2.3 | 737 | 1.07 |
| 64 | 16 | 9.4 | 1101 | 409 | 10 | 1537 | 1651 | 41634 | 208 | 4.1 | 2901 | 1.89 |
| 256 | 16 | 16.2 | 1904 | 772 | 22 | 2606 | 2774 | 98243 | 191 | 9.7 | 11814 | 4.53 |
| 1024 | 32 | 18.4 | 3571 | 1583 | 280 | 4849 | 5368 | 211178 | 145 | 20.8 | 49310 | 10.17 |
| 4096 | 32 | 26.3 | 8153 | 6587 | 1101 | 13772 | 15663 | 297426 | 53 | 29.2 | 196319 | 14.26 |

RTX 3060, the same:

| rows | tile | plan | gate and up | down | combine | block, atomic | block, ordered | rows/s | GB/s | TFLOPS | decode kernels | times faster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 10.0 | 126 | 40 | 12 | 172 | 185 | 5827 | 88 | 0.6 | 144 | 0.84 |
| 2 | 8 | 9.3 | 214 | 65 | 10 | 292 | 307 | 6845 | 98 | 0.7 | 231 | 0.79 |
| 4 | 8 | 9.5 | 407 | 121 | 11 | 542 | 560 | 7377 | 103 | 0.7 | 438 | 0.81 |
| 8 | 8 | 9.7 | 676 | 216 | 12 | 897 | 921 | 8920 | 106 | 0.9 | 759 | 0.85 |
| 16 | 8 | 10.2 | 1190 | 384 | 12 | 1566 | 1608 | 10219 | 106 | 1.0 | 1540 | 0.98 |
| 64 | 16 | 13.9 | 2282 | 881 | 30 | 3074 | 3212 | 20822 | 104 | 2.0 | 6014 | 1.96 |
| 256 | 16 | 21.7 | 3943 | 1849 | 101 | 5706 | 5916 | 44867 | 87 | 4.4 | 24704 | 4.33 |
| 1024 | 32 | 23.9 | 7847 | 3828 | 395 | 11538 | 12089 | 88750 | 61 | 8.7 | 103300 | 8.95 |
| 4096 | 64 | 44.4 | 18968 | 11911 | 1523 | 30940 | 32339 | 132386 | 24 | 13.0 | 409562 | 13.24 |

The other two types are faster in proportion to their bytes. Block time with the best tile and
the atomic sum, RTX 4070 / RTX 3060, in microseconds: IQ2_XXS (blk.1) 73 / 132 at 1 row, 350 / 656
at 8, 1855 / 4243 at 256, 12998 / 28900 at 4096; IQ1_M (blk.8) 66 / 114, 194 / 404, 1398 / 3551,
12934 / 28763. The decode kernels at 4096 rows take 183 and 148 ms (RTX 4070), 383 and 296 ms
(RTX 3060) for the two.

- **From 16 rows on they are as fast or faster than the decode kernels**: 1.9 times as fast at
  64 rows, 4.3 to 4.5 at 256, 9 to 10 at 1024, 13 to 14 at 4096 (IQ2_S; the others more). Below
  16 rows the decode kernels still win for IQ2_S (0.7 to 0.93 times as fast on the RTX 4070, 0.8
  to 0.85 on the RTX 3060 up to 8 rows) and for IQ2_XXS at 1 and 2 rows on the RTX 4070; IQ1_M is
  ahead from 2 rows, and from 1 row on the RTX 3060. So the family cannot be chosen by a constant:
  `ExpertRowKernels::few_rows_break_even` times both on the device for the shape (about 100 ms, once
  at load) and says up to how many rows the decode kernels are faster. The test run found 8 rows
  for IQ2_S on both cards, 5 (RTX 4070) and 1 (RTX 3060) for IQ2_XXS and none for IQ1_M. A verify
  window of 2 to 16 rows is therefore not faster than today's kernels for the 34 IQ2_S layers; for
  the 14 others it is up to 1.4 (IQ2_XXS) to 2 (IQ1_M) times as fast at 16 rows.
- **What bounds them.** Below 1024 rows the weights' loads: gate and up read 190 to 245 GB/s of
  the 473 the card can (RTX 3060 90 to 106 of 337), because each lane of a warp reads 2 and 4
  bytes of 41 bytes of a row per step, eight rows at a time, a pattern that streams at 189 GB/s
  (149 on the RTX 3060) in a loop of those loads alone, where a warp reading rows in order reads
  473 (337). Above that the product: 33 TFLOPS of useful products in gate and up (RTX 4070, 4096
  rows), 20 in down, of the 58.6 of the tensor cores. Branching around unused columns, deeper
  pipelines, activations resident in shared memory, L2 prefetch hints, and accumulating in f16
  (14 % faster for 8e-4 more error) were tried and are in the comments of the kernels with
  their numbers; none closed the gap. The layout of the loads is the open lead.
- **Atomic or ordered.** `red.add` into the output is 3 to 15 % faster than the scratch and the
  combine (more above 1024 rows, where the scratch is 100 KB a row), and needs no scratch; the
  ordered sum is what gives a row the same bits alone, in a batch and in any place of a batch.
- **The plan** costs 7 us for one row, 9 us at 64, 16 at 256, 26 at 4096 (RTX 4070; 10, 14, 22, 44
  on the RTX 3060): a few percent of a block at 256 rows and above.
- **What a prompt pass could reach.** Experts only, all resident, 48 layers (34 of IQ2_S, 11 of
  IQ2_XXS, 3 of IQ1_M) at the best tile: RTX 4070 650 ms for 4096 rows (6300 rows per second),
  220 ms for 1024 rows (4640); RTX 3060 1456 ms (2810) and 536 ms (1910); the decode kernels in
  batches of eight take 9.1 s and 19 s for 4096 rows, 449 and 215 rows per second. Two cards on a
  layer split can add up to about 9100 rows per second for the experts. The rest of a layer is
  not in these numbers. The dense parts of the model are 3.4 GiB, which by their bytes is
  roughly 7 to 13 GFLOP of work per row against the 4.7 of the experts, so a whole pass would take
  about twice as long at the dense kernels' rates (an estimate, not a measurement). The experts that
  are not in memory, which the pass has to stream, come on top.
- **Memory a call needs**, per row for Qwen3.8-Flash-Next (ten picks): 5.2 KB of f16
  activations, 13.2 KB of hidden values, 102 KB of scratch for the ordered sum (none for the
  atomic one), 0.12 KB of plan and 10 KB of output; the plan is 0.6 MB for 4096 rows.

**Layer kernels** (`kernels/layers.cu`, `ares_cuda::LayerKernels`).
Everything a block of `qwen35` does between its matrix products: the
embedding gather (all 28 weight types), the norms per head, the causal
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

**The token embedding of `qwen35`.** The embedding table (`token_embd`,
248 320 rows of 5120 values in the file's types: 313 MiB in the small test
file, 680 MiB at Q4_K_S) does not lie on the device. A step needs the rows of
its own tokens, a few kilobytes each in the file's form, and the gather kernel
decodes them wherever they lie (`kernels/layer_embed.cuh`: all 28 weight
types). Where they come from (`device/embed.rs`):

- **The host.** The table lies in page-locked host memory of the device that
  embeds (the first one), booked as required memory of the engine's one RAM
  limit (`Model::pinned_ram`). A step copies the rows of its tokens into a
  page-locked staging buffer, one row per row of the step, and uploads them
  on the step's stream in one copy right after the record; no event, no wait.
  A prompt step of 512 rows uploads 1.4 MiB at Q4_K.
- **The device itself, for the draft chain.** The model's own draft block
  picks tokens inside the step, on the device, and embeds them (the chain's
  next link, the verify window, the block following the model) before the
  host knows them. So the device that holds the draft block keeps the rows of
  every token the chain can pick, which are the rows of the draft head: with
  a draft vocabulary those tokens in ascending order (the same list that
  orders the head, so the gather finds a row by binary search and nothing
  extra is stored), without one the first rows of the head on the device (the
  whole table when the device holds the whole head, and then the host stages
  nothing). A device without a draft block keeps no row. The newest token of
  a chain, which the host knows, is staged with the draft block's records.
  The gather takes a row from the kept rows when they hold its token and
  from the staged row of its position otherwise.
- **A tied head** (a file without `output.weight`, whose head is the table)
  is a head like any other: its rows are laid out as the head's rows, in the
  layout that measures faster, and the embedding adds nothing. With a draft
  block on its device the head is one matrix in the file's layout instead,
  and the rows of the draft head, which are its first rows, are the rows the
  gather keeps: still nothing twice on the device.

A layer split no longer counts the table in the first device's share.

**Device layouts in the model.** At load the model first measures how the
few-row kernels launch for every shape of a matrix that can have the device
layout (`Kernels::tune_layout`, four launches timed over the probe buffer: a
second and a half of load on each card), then probes its own products in both
layouts, for every type and shape, at one row and at a prompt group, and gives
each group of matrices the layout that does not slow a decode step by more
than the noise of a probe (2 % of the group) and is fastest for prompts after
that. The rows of the decode probe are one whether or not the model drafts: a
choice that depended on the window could change the text with drafts.
Measured 2026-11-12 at shares of 0.625 / 0.375, prompt of 2048 tokens and a
decode step at 1000 tokens of context, with the binary of the commit before
the few-row kernels beside it, back to back (that commit in brackets):

| | every matrix in the file's layout | every matrix in the device layout | measured choice |
| --- | --- | --- | --- |
| Prompt, tokens per second | 1232 to 1234 (1232 to 1233) | 1356 to 1359 (1311 to 1322) | 1351 to 1366 (1267 to 1283) |
| Decode step | 24.76 ms (24.74) | 24.75 ms (26.5 to 28.7) | 24.75 to 24.78 ms (24.81 to 24.86) |

The measured choice puts 458 to 461 of 498 matrices in the device layout on
the RTX 4070 and 432 to 469 of 497 on the RTX 3060, a little different in
every load: the probes of shapes within 2 % of each other go either way. Of
the ones it leaves in the file's layout, 22 on each card are the i-quants of 2
and 3 bits, which have no device layout, and the rest are shapes for which one
row is still slower in the device layout: 512 and 768 rows of Q4_K and Q5_K
(+6 to +13 % on the RTX 4070, +4 to +5 % on the RTX 3060 for 256 and 512
rows), IQ4_XS and IQ4_NL of 3072 and 4608 columns (+3 to +6 %), Q4_K of 3072
columns on the RTX 3060 (+4 %), Q5_K of 10752 rows on the RTX 4070 (+4 %) and
IQ4_XS of 6400 and 6656 rows on the RTX 3060 (+1 to +3 %).

With the model's own drafts (three, the fork's draft vocabulary, 300 tokens)
German, English and code give, in two loads each, 85.2 / 78.2 / 89.8 and 86.6
/ 80.5 / 86.6 tokens per second for the most probable token and 81.6 / 77.3 /
93.9 and 82.4 / 76.8 / 91.5 sampled; the start commit gave 82.0 / 81.1 / 82.2
and 81.4 / 80.4 / 79.5, and 78.4 / 76.2 / 88.0 and 82.6 / 74.7 / 87.7 sampled.
That is +5 % for German, -2 % for English (inside the difference between two
loads) and +9 % for code on the averages. A step with drafts takes 31.2 to
31.3 ms against 32.4; the tokens per step (2.40 to 2.78 against 2.50 to 2.65)
depend on which matrices the load left in the file's layout, so the rates of
two loads differ by up to 4 %. Qwen3.5-0.8B on one card reads prompts at 26
700 to 29 700 tokens per second on the RTX 4070 (21 300 to 26 700 at the start
commit) and 13 100 to 14 500 on the RTX 3060 (11 900 to 12 400); a decode step
takes 2.41 to 2.42 ms of device time on the RTX 4070 (2.30 to 2.31: 5 % more,
with 156 to 168 of 187 matrices in the device layout) and 3.30 to 3.33 ms on
the RTX 3060 (3.28 to 3.35). The machine ran other builds during these
measurements (load average 3 to 5 of 16 threads, 7 to 13 for the 0.8B).

The same two runs with the branch rebased onto the tile kernels for more weight
types, the draft policy and Flash-Next, against the binary of the commit under
it (two loads each, the measured choice): the 27B in Q4_K_S on both cards reads
prompts at 1347 to 1366 tokens per second against 1257 to 1282, and a decode
step takes 24.73 and 24.77 ms against 25.06 and 24.87. The IQ2_XXS file of the
27B on the RTX 4070 alone (context 32768), where the tile kernels for the 2-bit
types and the few-row kernels meet, reads prompts at 1168 to 1171 against 1157
to 1160 and a decode step takes 20.90 and 20.91 ms against 21.03 and 21.00. The
choice puts 99 of the 498 matrices in the device layout there and leaves 399 in
the file's; the commit under it left all 498 in the file's. The 2-bit i-quants
have no device layout, and their few rows run on the integer kernels as before.

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
rows; the rows that differ are near ties. The tool steps the model through a
session of its own, with eight rows per step unless `--rows` says more: prompts
are read at 54 to 73 tokens per second *(agent)* in steps of eight rows, bound by
experts that have to be fetched, and at 357 to 388 (1000 tokens) in slices of 512
rows on the many-row kernels (see "Wide steps" below). The server runs it through
the runtime (see below).

An expert that is in no slot can be computed on the CPU instead of being
uploaded, chosen per expert by measured cost. On the recorded routing with
few slots that took the expert path from 33 to 44 steps per second; with 99
% of the picks in a slot it makes no measurable difference to the model.

### Qwen3.8-Flash-Next through the runtime

`ares --model <first part of the file>` serves Qwen3.8-Flash-Next like the
27B: the same endpoints, timings and metrics, cancellation, a request that
gives way and comes back, several sequences in a step. The model is
`ares_model::qwen4exp::device::Qwen4Exp`, its parts `Qwen4ExpDevice`; `Session`
stays as the tool's and the tests' way to step it by hand (and for the probes
that compare layer by layer).

**It does not speculate.** The draft policy gives a model drafts only when it
has a drafter of its own (`Model::own_drafts`): the lookup in earlier text
takes part as soon as one does and never alone (see [Speculation](#speculation)).
Flash-Next has no draft head built, so its windows are one row, its parts
reserve no room for windows, and the server refuses `--drafters mtp` (and a
drafter from a file), `--window` and `--draft-vocab` by name; `--drafters
none` is what it does anyway. The measurements below that speak of drafts were
taken before the policy, when the lookup ran by itself on this model, and
were not repeated: the column without drafts is what the server does now.

**A step that has to wait for experts.** The experts a layer needs are known
when its router has run, so a step cannot be enqueued whole and left alone:
an expert that is in no slot is brought (uploaded, or computed by the CPU)
while the device waits for it, and the host that brings it must still be
running. What was chosen:

- A device thread enqueues its part layer by layer and stays at most six
  layers ahead of the route records the store has handled. A host that went
  on would fill the driver's queue, block inside a launch and then no longer be
  able to bring the expert its device waits for; the session found that, and
  the number is the session's.
- The store has one owner at a time and sits behind a lock that no thread waits
  for in a step. Whoever finds it free serves it for all parts (hands over the
  records that arrived, polls reads and copies), and whoever does not goes on.
  A thread serves it whenever it would wait: for the lead, for the rows of the
  n-gram table, and for the end of its step. For that last one the runtime has
  a hook, `DeviceModel::wait`, where it used to block in the driver; the
  default still blocks, and `picked` tells the device with the head what was
  picked, so that the rows of the n-gram table for the next token are read
  while the conductor still decides.
- The first part to reach a step begins it for the store, the last to end it
  ends it. A part that fails makes the others give up, and a step that stands
  still for 30 s ends with an error that says what it waited for.
- The conductor sees none of this. It dispatches a step and collects it when
  every device has ended; a step that misses experts is a longer step, and
  cancellation and new requests are handled between steps as for any model.
  The only protection of other sequences is the planner's: a prompt is read
  in pieces of up to `Model::prompt_rows` rows (for this model its whole step,
  `step_rows`: 256 rows on this machine), each with the newest token of every
  generating sequence riding along, and the share owes the generating
  sequences rounds for the time a piece took. A piece of 256 rows takes 1 to
  2.5 s (a prompt of 1000 tokens, four pieces, 4.6 to 10.1 s to the first
  token; in steps of eight rows it was 125 steps of about 70 ms, 8.7 s) against
  22 ms for a step of one row, because every row routes to experts of its own,
  so the token that rides along with a piece comes after that time, and the
  rounds owed between pieces go at a token each; whether generation gets the
  half of the time that the share promises was not measured for this model.
  Taking the missing experts of a step's rows apart, to let the rows that have
  theirs run first, is not built.

Both placements work through the same code. By layers, the second device
receives the residual through the runtime's links; with both cards on every
layer the two parts exchange partial results in memory they write to each
other, and the runtime builds no path (`Split::Shared`). Page tables follow the
conductor's updates (a part keeps a mirror of them in page-locked memory and
copies the ranges that changed), K and V and the keys of the indexer live in
the pages the runtime allocates (`Frame`'s layout, 13.9 MiB per 1000 cells on
the card with the attention layers), and the logits of the wanted rows stay on
the device for the sampler.

**Wide steps: a prompt on the many-row kernels.** A step of up to eight rows is as
described above, and a round of generation stays there: one row for each sequence, since the
model takes no drafts. The piece of a prompt that the conductor sends between rounds, up to
`Model::prompt_rows` rows, which for this model is `Model::max_rows` (`step_rows`, below),
with the newest token of every generating sequence riding along, runs every layer at that
width (`device/part/wide.rs`, `device/misses.rs`):

- *Mixers, n-gram block, recurrent layers, attention* take the rows as they are: the
  kernels were written for the 27B's prompts. The dense products go through the tile
  kernels (`Kernels::product`) in the activation form the device and the type ask for:
  the int8 rows the fused kernels already leave (`mix_quant`, `gated_norm_quant`,
  `swiglu_quant`), or f16 rows (`prepare_mixed`, `gated_norm_f16`, `swiglu_f16`) when the
  device runs the f16 tensor cores at full rate or `Options::precision` is `Exact`. The
  float matrices (the mixers', the router, the indexer projections, the recurrent gates,
  the n-gram block's projections) have no tile kernel and are multiplied eight rows at a
  time by the decode kernel, which reads the f32 section of the activations
  (`Part::wide_product`); the chunks run on four streams, since a launch of eight rows
  has a few dozen blocks and takes about 90 microseconds whatever it multiplies. The
  fused read of a stream mixer is made for a few rows, so a wide step reads its mixers
  with the separate kernels.
- *The router* is `ExpertKernels::logits_rows` and `pick_rows`: ids and weights of every
  row's picks in the form `ExpertRowKernels::plan_rows` reads, without the groups of the
  decode kernels, which hold eight rows.
- *The routed experts* run in **rounds**. The many-row kernels do not wait on the device
  for an expert that has no slot, so in a wide step the host takes part in every layer
  (`Misses`). When the layer is enqueued it gives the first round a table of the address of
  every expert that has a slot in VRAM (`ExpertStore::vram_slot`), and the device computes
  them while the route record travels to the host. The record gives, per expert, how many
  pairs picked it. The experts without a slot are the layer's **misses**, most picked
  first. Once the first round has ended on the device (nothing reads a slot any more), the
  store is asked for as many misses as the layer has slots (`ExpertStore::routed`, which
  protects the batch and evicts what is not needed), and the batch that has arrived is the
  next round's table, whose outputs `red.add` into the same rows. The plan of the layer is
  made once. The host never waits inside a call: the caller serves the store between the
  steps of `Misses::advance`. A step that misses nothing has one round and the host only
  waits for the record, while the first round runs. The experts are not computed on the
  CPU in wide steps; the heat of the store gets the layer's counts of rows per expert
  (`record_counts`), so what a prompt used is what the tiers hold when it ends.
- *Both devices on every layer*: the mixer routes. The worker's host thread reads the
  record from the mixer's board (the mixer has enqueued the layer when `Flow::enqueued`
  says so), uploads it, and builds its table from the experts it owns
  (`Placement::owner`). The mixer pushes the mixed input of both halves to the worker as
  f32 rows with a flag per row; the worker quantizes them. The worker's exchange is two
  launches in a wide step: its kernel has a block per row, and the blocks that wait for
  the mixer's input must not keep from running the blocks that push the partial output
  the mixer needs before it sends that input (with more rows than blocks that fit the
  device at once the single launch stood still).
- *What stays at eight rows*: the host share and the buffers of the decode kernels, the
  per-layer copies of the layer's input rows that the CPU reads, and the store's reserve
  (`StoreConfig::max_rows`). Only the scratch of a step scales with the rows.

`step_rows` decides the rows a model opens with: the largest power of two up to 4096
whose scratch (`Frame::step_bytes`) is at most a thirty-second of the smaller device's
memory, and 8 on a device below compute capability 8.0. The scratch is about 0.7 MiB a
row for Flash-Next, and every byte of it is a byte less for expert slots, which the
generating steps miss.

**Measured through the server** (2026-11-11, RTX 4070 and RTX 3060, the
engine started with `--chat-var enable_thinking=false` and otherwise its
defaults, greedy, 128 tokens, a prompt of about 1000 tokens: 1000 German, 1070
English, 1034 Rust; load average 12 to 18 from other agents' builds unless
said; the numbers are of the second and third of three rounds of the same
three prompts, after the warm-up and a round that fills the slots):

| | Both on every layer, lookup drafts | The same, no drafts (`--drafts 0` then; what the server does now) | A range of layers each, lookup drafts |
| --- | --- | --- | --- |
| Generation, German / English / code | 46 and 45 / 44 and 110 / 70 and 73 tokens/s | 44 and 46 / 48 and 43 / 42 and 42 | 27 and 45 / 16 and 42 / 56 and 74 |
| A short question (25 tokens), German / English / code | 59 and 68 / 45 and 60 / 54 and 57 | 39 and 37 / 42 and 40 / 33 and 33 | 29 and 69 / 61 and 46 / 38 and 55 |
| Time to the first token, prompt of 1000 tokens | 8.4 to 15.8 s | 8.7 to 13.4 s | 10.1 to 20.6 s |
| Time to the first token, short question | 0.30 to 0.43 s | 0.30 to 0.46 s | 0.29 to 0.52 s |
| Prompt, 1000 tokens | 63 to 128 tokens/s | 77 to 123 | 50 to 106 |
| Prompt, 2000 tokens | 126 tokens/s | 108 | 72 |
| The tool, same machine, load 17 to 23 | decode step at 1000 tokens: 23.1 ms median (21.5 to 23.1 at lower loads); prompt of 2000 tokens: 71 tokens/s in steps of eight rows (before the wide steps below) | | |

Without drafts a token takes 20.7 to 23.9 ms, the tool's step (21.5 to 23.1
ms), so the server adds nothing to it. The median over the six warm runs with
the lookup drafts was 58 tokens per second, and 48 and 51 over the twelve runs of each of
the two servers measured again below, against Strata's 56 with drafts: level
within the spread of the runs, which is wider than the difference (44 to 110
in the first series; what the drafts gain depends on the text, as everywhere:
English prose at 110 accepted 95 of 95). Prompts were read at 63 to 139 tokens per second, in steps
of eight rows, where the tool's earlier measurements were 54 to 73 and Strata's
42 to 201 (slices of a prompt are wider now, see "Wide steps, measured").

Where the experts were found, per decode step without drafts (480 picks, 10
experts of 48 layers), second and third round: 6.0 to 15.6 outside a slot for
English and German prose (1.3 to 3.3 %) and 34 to 37 for code (7 to 8 %), of
which 0.7 to 2.0 from the file and 5 to 35 from the RAM tier (each miss is
then computed by the CPU or uploaded, 4.5 to 32 and 2.0 to 17 per step); the
layers waited 2.3 to 6.6 ms per step in all. Prompt steps (eight rows) pick
about 40 distinct experts per layer and miss 4 to 15 % of them: 80 to 300 per
step, 7 to 21 of them from the file. A run of the engine takes 11.4 GiB of each
card (11 689 and 11 663 MiB in use of 12 GiB: the dense parts are 2.9 GiB on the
first card and 0.4 on the second, 13 824 expert slots, 5 808 and 8 016, are 56 %
of the experts, K and V are 13.9 MiB per 1000 tokens, 18 432 tokens in all) and
10.6 to 11.9 GB of RAM (10.2 GiB of them the tier of 7 488 slots, 30 % of the
experts), so 87 % of the experts have a slot. Starting takes 11 to 15 s to load and 41 to 44 s
for the warm-up; the first request after a start without the warm-up took 77
s for a prompt of 1000 tokens (13 tokens/s), with it 13 to 23 s in four
sessions and 53 s in one at a load of 15 to 18; a text the warm-up did not
resemble keeps being slow until its experts have slots.

Measured again on 2026-11-12, after the work was put on the main line (the
embedding gather, the prefix cache, reasoning in the server, the many-row expert
kernels, none of which the model's path uses): the server of before and the one
of after, started alternately twice, three rounds of the same three prompts at
128 tokens, rounds two and three, load average 10 to 15. Generation was 30 to 89
(median over the twelve runs 47.7) tokens per second before and 31 to 84
(median 50.8) after: German 30 to 89 and 31 to 84, English 20 to 55 and 43 to
53, code 32 to 81 and 48 to 71. Prompts of 1000 tokens were read at 37 to 138
and 65 to 138 tokens per second, the time to the first token was 7.8 to 27.8 s
and 7.8 to 15.9 s. The two servers do not differ beyond the spread between runs
of one of them, which comes from what the text accepts of its drafts and from
the load. The cards held 11 679 and 11 661 MiB, the process 9.4 to 11.4 GB of RAM
(once 14.7 GB right after the start). A first run of the server after the work
at a load of 23 to 25, with other agents' builds and benchmarks on the same
drive and cores, gave 21 to 43 tokens per second and prompts at 35 to 52: the
machine's load decides what a step that waits for the drive costs, so a
comparison needs the two servers measured next to each other.

Against the tool's chat on its two questions the server writes the same
English text (64 tokens, the second time; the first time it parted after 30
tokens at a near tie) and, in German, a text of the same kind that starts with
"Das Sonnenlicht" where the tool's starts with "Sonnenlicht" (a near tie: the
reference's own text, in `/var/tmp/ares-qwen4exp-trace/german.txt`, has "Das
Sonnenlicht"). The engine's own test pins the first sentence of the English
answer to the reference's.

**What the plan decides.** Where the layers go (`--split tensor` or `layer`, by
default both cards on every layer when each can write the other's memory), the
context, and the RAM. The dense parts take 2.9 GiB on the first card and 0.4
on the second; what the cards have left is slots for experts. The context
costs slots, so without `--context` the K and V of the pool take a
thirty-second of the room for slots (17 408 to 18 432 tokens for two sequences
on this machine) where the 27B gets what is left. Options that mean nothing for the
model are errors that name them: `--shares`, `--precision exact`,
`--draft-vocab`, K/V formats with a tile mean, more sequences than the eight rows of a
generating step (`--parallel`), and the options of drafting (`--drafters mtp`,
`--window`).

**The RAM.** The tier of experts is a user of the one RAM limit (`--ram`,
`PinnedRam`): elastic, with a priority above the checkpoints, so the limit
minus what the engine needs to run (the step buffers and the parts'
page-locked buffers, 19 MiB) is the tier, as far as the experts that have no
VRAM slot can use it. The number of RAM slots is what that gives: with the
default limit (half of the RAM the machine has available when the server
starts) and 22 GiB available, 10.9 GiB, 7 970 experts. Before this change the
tier was a third of the available RAM, at most 8 GiB, whatever the RAM limit
said, and the page-locked buffers of the parts were outside the books.

**What the gap between 17 and 22 ms is made of, and what the size of the tier
did.** The tool counts, for every decode step at 1000 tokens, the experts the
routers picked that were in no slot, and splits the steps by what they missed
(on the recorded routing, 192 steps per run, five runs, load average 6 to 23
from other agents' builds, so the milliseconds differ between runs and the
classes within a run are what to read; the ranges leave out the run at a load
of 14, whose medians were 25.1, 31.5 and 42.6 ms):

| Steps | Share | Median |
| --- | --- | --- |
| No expert outside a slot | 2 to 6 % | 17.3 to 20.4 ms |
| Misses served from the RAM tier or computed by the CPU only (3 to 5 of them) | 23 to 31 % | 17.5 to 19.1 ms |
| At least one expert read from the file (2 to 3 of them, among 6 to 17 misses) | 62 to 72 % | 24.2 to 29.5 ms |

A miss served from RAM costs an upload of 0.1 ms at the median and the CPU
computes the others while the device runs, so neither shows in the step: the
steps with such misses take as long as the steps with none. The 5 to 12 ms
above that belong to the experts read from the file, 2 to 3 per step at 2.4 to
3.3 ms at the median and 3.7 to 11 ms at the 99th percentile each, which the
device waits for in the layer that needs them, both cards at once. Most steps
have one, which is why the median (21 to 23 ms) is far above the 17 ms of the
steps that have none and the mean (24 to 30 ms) above the median. The rows of
the n-gram table were not part of it (0.07 to 1.0 ms per step below a load of
9, read ahead from the picks).

Those reads are experts used for the first time in a run, or used again after
a tier let them go, so the size of the tier decides how many of them there are
only when the text comes back to experts it has left. With the profile loaded
the tool reads the same 3.0 experts per step with a tier of 5 472 slots and
with 8 064 (22.6 and 22.8 ms at the median in two runs back to back at a load
of 6 to 7; the second pair, 21.5 and 37.0 ms, ran at loads of 8 and 14 and
says only that the load decides). Through the server, with a new text every
request, the size decides a great deal. The same
prompts in a server per tier, load average 12 to 13 (15 to 18 for the
second), round 1 of 2 or 3, tokens per second with the lookup drafts the
server had then:

| Tier (`--ram`) | RAM slots | Experts with a slot | File reads per step, prompt rows | File reads per step, decode | Generation: German / English / code | Prompt, 1000 tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 7.5G, the third of the available RAM the tier took before | 5 472 | 78.5 % | 20.8 / 14.1 / 46.1 | 4.7 / 2.1 / 13.5 | 37.7 / 43.1 / 51.2 | 69 / 101 / 57 |
| default, half of the available RAM as the limit | 7 488 | 86.7 % | 11.0 / 6.9 / 20.6 | 3.3 / 0.9 / 6.1 | 46.4 / 43.9 / 69.6 | 63 / 102 / 81 |
| 16G | 11 712 | all | 0.2 / 0.1 / 0.7 | 0.6 / 0.2 / 0.2 | 75.5 / 44.2 / 103.9 | 139 / 137 / 127 |

The default now takes the limit less the engine's buffers (10.3 GiB of 20.65
GiB available when the server started) where it took a third of the available
RAM, 7.5 GiB, outside the books of the limit; that is 37 % more slots and about
half the reads from the file. With a limit that lets the tier hold every
expert the VRAM has no slot for (about 16 GiB on this machine), reads from the
file stop after the first use of an expert. Raising `--ram` is the lever, and
how much RAM is free is the user's to say; the limit is not raised by itself.
The tier does not shrink if another user of the machine needs the memory.

**Taking part in the prefix cache** (`prefix_cache()` is `None`: every prompt
is computed from its first token, a request that gives way comes back as a
prompt of all its tokens) needs the device side to carry out what the step
record asks: the saves and restores of the recurrent state of a slot (the
convolution windows and delta matrices of 36 layers, the n-gram history and the
carried indexer state of the attention layers: 112 MiB over both cards per
slot, which would be a checkpoint), the copies of a page's entry (K, V and the
keys of every attention layer), and replay segments, rows that rebuild
recurrent state from cached cells and append nothing (the K/V append, the
indexer's window and the n-gram commit would skip them). The host side needs
the n-gram feed to be told what the cached prefix's last tokens were, because
it keeps them per sequence for the hash of the rows of the table. The
conductor's part, and the places for checkpoints in `PinnedRam`, exist.

**Wide steps, measured** (2026-11-12, RTX 4070 and RTX 3060, the engine of before (`0a43601`,
steps of eight rows) and the one with wide steps started one after the other, both with the
lookup drafts the server made before the draft policy, greedy, 128 tokens, prompts of about 1000 tokens (German, Rust) and 4700 (the long one: prose,
code and German), rounds one to three of the same prompts, the numbers of round three; load
average 2.4 to 9.4 for the layers and 2.6 to 10 for the two on every layer, other agents'
builds in the background; `--chat-var enable_thinking=false`, otherwise the defaults):

| | Layers split, before | Layers split, wide steps | Both on every layer, before | Both on every layer, wide steps |
| --- | --- | --- | --- | --- |
| Prompt, German 1000: tokens/s, time to the first token | 110, 9.1 s | 170, 5.9 s | 141, 7.1 s | 217, 4.6 s |
| Prompt, Rust 1000 | 83, 12.5 s | 103, 10.1 s | 100, 10.4 s | 123, 8.4 s |
| Prompt, 4700 tokens | 100, 50.4 s | 132, 38.1 s | 126, 39.8 s | 166, 30.4 s |
| Generation, German / Rust / long prompt: tokens/s | 42 / 68 / 41 | 53 / 46 / 36 | 70 / 80 / 43 | 58 / 53 / 34 |
| VRAM in use (RTX 4070, RTX 3060) | 11 703, 11 695 MiB | 11 707, 11 695 | 11 683, 11 665 | 11 697, 11 663 |
| RAM, anonymous | 11.80 GB | 11.81 GB | 11.71 GB | 11.65 GB |

The step has 256 rows (`step_rows` with a thirty-second of the memory: 221 and 179 MiB of
step buffers where eight rows took 53 and 45), which costs 0.16 GiB of expert slots on the
RTX 4070 (9.20 to 9.04 GiB). Wide steps alone make the prompts 1.2 to 1.6 times as fast, not the
five to ten times that the kernels alone would allow, and they cost the generating steps
that follow: in this series they miss 1.2 to 1.5 times as many experts (outside a slot, of the picks of
a decode step: 4.1, 12.7 and 7.3 % before and 6.2, 15.0 and 10.7 % after for the layers split;
4.3, 12.6 and 7.8 % and 5.7, 15.5 and 10.5 % for the other), and the generation speeds in the table
lie lower in five of six pairs, within a spread of the runs that is as wide as the difference
(30 to 90 tokens per second by text and run, as before). A step of 2048 rows (an eighth of the
memory, 1.4 GiB of step buffers, 1.1 to 1.2 GiB fewer slots; the layers split) read the prompts 1.3 to 1.8 times
as fast as eight rows, and generated at 47, 33, 34, 29 and 33 tokens per second where eight rows
gave 77, 41, 68, 37 and 69: the slots are worth more than the rows.

What a wide step misses (the same runs, per layer and step of 256 rows, the layers split
and the two on every layer alike): 184 to 243 distinct experts picked of 512, of which 34 to
57 (19 to 29 %) had no slot, against 39 to 45 picked and 2 to 5 of them missed in a step of eight
rows; the layers waited 31 to 62 ms per layer and step for them (the long prompt: 55 to
56), where 126 to 631 steps of eight rows waited 0.3 to 0.9 ms per layer each. In the tool, with the
same 1000 tokens read again (the experts of the text are in the tiers), 512 rows read at 388
tokens per second on the layers split and 357 on the two on every layer; 4000 tokens in slices
of 512 rows read at 112 and 145 tokens per second warm and 67 and 63 cold, in slices of 2048 at
147 and 129 warm (one layer of a slice then waits 160 to 350 ms for 110 to 300 missed experts).

What bounds the prompt now:

- *Bringing the missed experts.* A slice of a long prompt picks 40 to 80 % of the experts of
  every layer, the tiers hold 56 % in VRAM and 30 % in RAM, so every slice misses a fifth to
  two thirds of what it picks, and the next slice misses them again. The store delivers an expert
  from the RAM tier in 1 to 2 ms, from the file in 8 to 130 ms, one request after the other:
  about a gigabyte per second where the link takes ten or more. That is 30 to 60 percent of a
  slice's time, and what the layer-major pass with waves must fix: fetch each expert of a layer
  once for the whole prompt, with many requests in flight and the next layer's experts read
  while this one computes.
- *The dense part of a layer.* The host waits 18 to 40 ms per layer and slice of 512 rows for the
  router's picks; that is the device's time through the mixers, the projections and the
  recurrent layer. The mixers' matrices are BF16, which has no tile kernel; eight rows at a
  time over four streams they take 4 ms a read on the RTX 4070 and 7 on the RTX 3060 at 512
  rows (the fused read took 7.5 and 24). A tile kernel for the float types would be the
  next step there.
- *The host in every layer.* The first round runs while the picks travel to the host, but the
  next layer cannot be enqueued before the host has the misses of this one: a layer without
  misses costs one round trip of the record (about 0.1 ms); the waves would take that out too.

**Where a faster prompt path goes next.** The prompt pass that streams whole layers
(`ares_sched::Pass` and `ExpertStore::plan_waves` / `wave`, built and unused) goes where
`Misses` is now: it would hand the store the layer's counts for all rows of the prompt
instead of a slice's, enqueue the waves, and keep the hidden states of the prompt's rows in a
buffer of their own on the device; the conductor would run the pass's pieces between ordinary
steps. `Misses` already has the part that matters for it: a round computes the experts of a
table of addresses, whatever slot or transit memory they are in.

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
  layout took 23.4, 25.4, 31.0 and 69.7 ms for the same passes (before its
  few-row kernels). At 1024
  rows a Q4_K matrix runs at 88.5 / 39.9 TOPS on the RTX 4070 / RTX 3060
  and Q6_K at 75.5 / 34.3 *(measured by the agent that built the kernels;
  CUDA: 88.8 / 39.3 and 74.5)*.
- The layer kernels exist with CUDA's semantics and are held to the same
  references.
- Attention exists for every stored K/V format (`f16`, `q8` and the formats
  with a tile mean down to `q4`, separately for K and V), over dense ranges,
  lists of ranges and prefixes that several sequences share, with the kernels
  that append new cells and convert exact tiles. The bytes it writes are the
  CPU reference's (`ares_quant::kv`, which CUDA's are held to as well),
  checked byte for byte; its
  output equals `KvHead::attend` within 1.1e-5 of the largest output element
  on random rows, on the captured rows of Qwen3.8-27B and for every pair of
  formats and both head sizes, on both cards (the reference's tolerance for
  an `f32` kernel is 2e-5; CUDA's tensor-core kernels are held to 8e-3).
  There is no tensor-core instruction in it. An item (some rows of one KV head
  over a list of ranges, as the CUDA plan cuts them) is one workgroup that
  takes a cell per invocation for the scores and a slice of the columns per
  invocation for V, in `f32`, with the rows decoded in registers; it holds up
  to 16 vectors (the query heads of its rows; 8 on a device with less than
  28 KiB of shared memory) where CUDA's hold 64. The planner aims for three
  workgroups for each multiprocessor, as CUDA's does; Vulkan does not report
  their count, so it is measured at load, by how many workgroups of 1024
  invocations the device runs at once (46 and 27 on these cards, the same in
  every load). The device needs what the quantized weight kernels need and
  shuffles in a subgroup.
  Measured 2026-11-12 on the shape of Qwen3.8-27B (head size 256, 4 KV heads
  of 6 query heads), one attention layer, the best of two runs, Vulkan next
  to CUDA on the same card, `ARES_LIBCUDA` from the p2p bench:

  | case (per layer) | RTX 4070 Vulkan us | RTX 4070 CUDA us | ratio | RTX 3060 Vulkan us | RTX 3060 CUDA us | ratio |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | f16/f16 1 rows over 1024 | 27 | 20 | 1.36 | 41 | 47 | 0.87 |
  | f16/f16 8 rows over 1024 | 62 | 34 | 1.84 | 121 | 109 | 1.11 |
  | f16/f16 64 rows over 1024 | 229 | 174 | 1.32 | 611 | 335 | 1.82 |
  | f16/f16 1 rows over 32768 | 358 | 307 | 1.16 | 494 | 526 | 0.94 |
  | f16/f16 8 rows over 32768 | 943 | 415 | 2.27 | 2,336 | 818 | 2.86 |
  | f16/f16 64 rows over 32768 | 7,023 | 3,229 | 2.17 | 16,779 | 6,101 | 2.75 |
  | f16/f16 1 rows over 262144 | 2,568 | 2,267 | 1.13 | 3,504 | 3,957 | 0.89 |
  | f16/f16 8 rows over 262144 | 7,928 | 2,728 | 2.91 | 18,885 | 5,822 | 3.24 |
  | f16/f16 64 rows over 262144 | 58,219 | 24,983 | 2.33 | 137,887 | 47,610 | 2.90 |
  | f16/f16 list of 2051 over 32768 | 38 | 32 | 1.19 | 76 | 92 | 0.83 |
  | q8/q8 1 rows over 1024 | 55 | 28 | 2.01 | 83 | 49 | 1.68 |
  | q8/q8 8 rows over 1024 | 108 | 40 | 2.74 | 226 | 115 | 1.97 |
  | q8/q8 64 rows over 1024 | 361 | 206 | 1.75 | 1,048 | 382 | 2.75 |
  | q8/q8 1 rows over 32768 | 493 | 217 | 2.27 | 888 | 480 | 1.85 |
  | q8/q8 8 rows over 32768 | 1,776 | 382 | 4.65 | 4,175 | 782 | 5.34 |
  | q8/q8 64 rows over 32768 | 45,696 | 3,378 | 13.53 | 32,521 | 6,335 | 5.13 |
  | q8/q8 1 rows over 262144 | 3,463 | 1,376 | 2.52 | 6,688 | 3,552 | 1.88 |
  | q8/q8 8 rows over 262144 | 15,954 | 2,447 | 6.52 | 32,716 | 5,396 | 6.06 |
  | q8/q8 64 rows over 262144 | 103,061 | 25,989 | 3.97 | 260,029 | 49,401 | 5.26 |
  | q8/q8 list of 2051 over 32768 | 71 | 39 | 1.81 | 147 | 72 | 2.06 |
  | q4/q4 1 rows over 1024 | 60 | 46 | 1.29 | 78 | 74 | 1.05 |
  | q4/q4 8 rows over 1024 | 106 | 41 | 2.55 | 200 | 105 | 1.90 |
  | q4/q4 64 rows over 1024 | 375 | 210 | 1.79 | 860 | 391 | 2.20 |
  | q4/q4 1 rows over 32768 | 426 | 196 | 2.17 | 818 | 445 | 1.84 |
  | q4/q4 8 rows over 32768 | 1,655 | 411 | 4.03 | 4,146 | 934 | 4.44 |
  | q4/q4 64 rows over 32768 | 12,787 | 4,208 | 3.04 | 31,231 | 7,988 | 3.91 |
  | q4/q4 1 rows over 262144 | 3,056 | 930 | 3.29 | 6,193 | 2,587 | 2.39 |
  | q4/q4 8 rows over 262144 | 12,978 | 2,701 | 4.80 | 33,379 | 6,082 | 5.49 |
  | q4/q4 64 rows over 262144 | 102,631 | 33,304 | 3.08 | 251,630 | 63,071 | 3.99 |
  | q4/q4 list of 2051 over 32768 | 80 | 61 | 1.30 | 137 | 117 | 1.17 |

  Where Vulkan is behind and what bounds it. With one row of `f16` over
  32768 or more cells it is within 16 % of CUDA on the RTX 4070 (418 GB/s of
  the card's 504 over 262144 cells) and 6 to 11 % faster on the RTX 3060:
  memory bound. With one row of `q8` or `q4` it takes 1.05 to 3.3 times as
  long on both cards: each invocation walks the row of its own cell through
  2-byte loads (the groups of the formats are 2-byte aligned) and unpacks the
  bit planes into `f32` before the products, so the kernel is bound by
  instructions and not by memory (165 and 99 GB/s on the RTX 4070 against
  CUDA's 413 and 321 for one row over 262144 cells). Windows of 8 and 64
  rows take 1.1 to 6.5 times as long, mostly 2 to 5: an item holds 16
  vectors, two rows of six heads, so a window of 8 rows is four items per KV
  head that each read every cell (CUDA: one item of 48 vectors on tensor
  cores), and 64 rows read them 32 times. A kernel for wide windows on the
  cooperative matrices that the weight kernels already use is not built.
  A list of 2051 cells takes 0.83 to 2.1 times as long as CUDA's.
  One entry stands out: `q8`, 64 rows over 32768 cells, on the RTX 4070 takes
  45.7 ms in both runs of the whole table, 13.5 times CUDA's. Alone in a
  process it takes 12.6 ms (3.7 times CUDA's) with a planner target of 78, 121
  or 138 workgroups, as it did in four earlier runs of the whole table, so it
  is not the plan; it depends on what ran before or where the pools lie, and
  the cause is not found.
- The sampler (`SamplerKernels`, `shaders/sample_*.comp`, `lp_*.comp`,
  `draft_*.comp`) has every step, record and entry point of CUDA's:
  corrections and masks, the token map and slices of the vocabulary as
  several devices hold them, the radix descents of top-k, top-p and typical
  sampling, top-n-sigma, min-p, XTC, temperature and its dynamic form,
  mirostat 1 and 2, the draw, the probabilities before and after the chain,
  and the draft kernels (picks into the next records, hidden-state gather,
  shift, keep and commit of the carry). `tests/sampler.rs` is the twin of
  CUDA's: 1312 rows of 41 settings, four shapes of rows and vocabularies
  from one token to Qwen's equal the records of `sample_reference` bit for
  bit on both cards, and so do rows with corrections and masks, with a token
  map, with the vocabulary cut into two or three slices whose exchange is
  added on the host, and in batches of different sizes that share one
  scratch area. A second test holds each function of the arithmetic (the
  weight, both logarithms, the exponential, the power, the Gumbel noise, the
  terms of a token, the division and the square root in double precision, the
  conversions of 64-bit integers) to `ares_sampler::arith` over 20 000
  random inputs from every range the sampler reaches, bit for bit on both
  cards. The code is CUDA's operation for operation, with these differences:
  - No 64-bit atomic (an optional feature on Vulkan, which the first
    mirror already avoided): every chunk of the vocabulary writes its best
    score, or its sums, into the scratch area, and the kernel that follows
    takes the maximum or adds them. The row's state is 128 bytes.
  - Subgroup operations of whatever width the device reports (at least 8)
    and shared memory for one value per subgroup; a maximum or a sum of
    64-bit numbers goes through 32-bit halves or 16-bit limbs, so that the
    64-bit subgroup types are not needed.
  - Division in single and double precision is corrected by fused
    multiply-adds to the correctly rounded quotient (the division of a
    shader is only within a few ulp), and every operation is `precise`, so
    that nothing is fused into a multiply-add.
  - **Double precision.** The reference uses IEEE doubles where a row needs a
    handful of numbers: the entropy and the deviation of a set, the dynamic
    temperature, the limit of mirostat 2, the exponent that mirostat 1
    estimates, the logarithm of the sum that a probability is divided by.
    The kernels repeat them with `double`, which needs the device feature
    `shaderFloat64`; both cards have it, and their addition, subtraction,
    multiplication, division (the corrected form),
    square root, and conversions between 64-bit integers, `f32` and `f64`
    equal Rust's in every bit tested. A device without `shaderFloat64` is
    refused at `SamplerKernels::load` with that reason; no other precision is
    substituted, since a sampler that diverged from the reference in the
    last bit would no longer give a seed the same text on both backends.
    (A soft-float library in 64-bit integers would give the bits on every
    device; it is not built.) Denormal results, and the order of NaNs and
    infinities in arithmetic, are left to the device: the shaders test the
    special logits on their bits, and a logit in the denormal range is not
    in the tests.
  - A device is also refused for less than 32 KiB of shared memory (the
    histograms of the descent) or workgroups of fewer than 256 invocations,
    with the numbers it reports. Nothing asks for a device's name.
- The expert and `qwen4exp` kernels and a model on Vulkan do not exist yet.
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
| TensorRT-LLM, SGLang, vLLM (suffix decoding) | One lookup drafter on a suffix automaton over a corpus shared by all requests | `ares-sched`, used by the conductor |
| MiniJinja | The interpreter of chat templates | `ares-tokenizer` |
| llguidance | Grammar-constrained output: the grammar engine on the host gives a mask per row, which the sampler applies on the device (the mask and its hook are built, see [Sampling](#sampling)) | Done (`ares-grammar`): GBNF and JSON schema on the sampler's mask, after the reasoning of a model that thinks, and a forbidden draft is cut; Lark and regular expressions are in the engine and have no request field |
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
  Drafts are chosen for every step from a price of a step measured at load
  and acceptance measured while running (see [Speculation](#speculation)):
  chains of 7 links where text is copied (142 to 163 tokens per second for a
  rewrite of a file), of 3 to 6 on prose and code. A prompt that arrives next
  to generating sequences gets its first token about 25 ms after what the
  cards need alone (see "A step"). See [Several devices](#several-devices).
- **Qwen3.8-Flash-Next** runs on both cards, agrees with the CPU reference
  and answers in German and English, and is served by the `ares` binary like
  the 27B, in both placements, without speculation: it has no draft head
  built, and the lookup in earlier text takes part only next to a model
  drafter. 43 to 48 tokens per second at the median of a series of runs at
  about 1000 tokens of context (a decode step takes 22 to 23 ms, 17 ms when it
  misses no expert). With the lookup drafts it had before the draft policy,
  which have not been measured since, it was at 48 to 58 (three series; 20 to
  110 by text and run), level with Strata's median of 56 with drafts on this
  machine within the spread; without them it is behind that until its draft
  head is built. Prompts are read in pieces of 256 rows on the many-row expert
  kernels, at 103 to 217 tokens per second for 1000 tokens and 132 to 166 for
  4700 (before: eight rows, 83 to 141 and 100 to 126; measured before the draft
  policy, with the slices the engine then sent); bound by the experts the
  slices miss (see [Qwen3.8-Flash-Next through the
  runtime](#qwen38-flash-next-through-the-runtime)).
- CUDA has every kernel family the two models need: weights for all 28
  types (twelve with a device layout that serves every row count from one
  copy), layers, attention over the stored K/V formats, sampler, sums
  between devices, router and experts, the kernels of `qwen4exp`. The CUDA
  tests pass on both cards, the kernel ones also on real tensors.
- The sampler serves everything a request can set on the device, in the
  names of the OpenAI API and of llama.cpp's server: penalties, DRY, logit
  bias, a mask per row (the hook the grammars will use), the probabilities of
  the tokens, the samplers of llama.cpp's chain in any order, mirostat, `n`
  completions and a seed. Its kernels equal the reference of `ares-sampler`
  bit for bit on both cards, the reference equals llama.cpp's definitions, and
  with drafts the tokens are the tokens without them (bit for bit on the
  synthetic model, and equal in all 14 settings on the 27B). A request that asks
  for none of it costs what the old sampler did. See [Sampling](#sampling).
  Vulkan's kernels (same steps and records, no 64-bit atomics, `shaderFloat64`
  required) equal the same reference bit for bit on both cards; no model runs
  them yet.
- Vulkan has the weight kernels (the same device layouts, byte for byte) and
  the layer kernels, at CUDA's speed on these cards, and the attention kernels
  over all stored K/V formats. Attention is within 20 % of
  CUDA's for one row of `f16`, 1.3 to 3.3 times as slow for one row of `q8`
  or `q4` and 2 to 6.5 times for windows of 8 and 64 rows (no tensor cores,
  items of 16 vectors). It has the sampler and no expert or `qwen4exp`
  kernels and no model.
- The KV formats are specified, pinned by golden files and measured on the K
  and V of real models; the CUDA kernels write them bit-exactly and read
  them within 1e-3 of the CPU reference. The dense model serves all of
  them for K and V apart, in every placement and with the prefix cache
  (`q8` is the default); the formats with a tile mean (`q7` to `q4`) keep a
  second, small pool of exact entries (see "Sequence state").
- The prefix cache runs on the devices for `qwen35`, in every placement: a
  conversation's next turn starts from the pages and the checkpoint of the
  last one (on Qwen3.8-27B a second turn after a history of 24k tokens gives
  its first token after about 0.14 s instead of 25 s; see "Sequence state"),
  the usage of an answer says how much came from the cache, and the engine's
  RAM is one setting (`--ram`).
- Known defects: a long prompt next to a generating Flash-Next sequence slows
  it by the misses of the prompt's rows (nothing separates the two inside a
  step); with the fast kernels the text with drafts is not always
  the text without (the rows of one sequence in a step change each other's
  logits by 2e-4 to 1e-3 of the logit, more in the file's layout than in the
  device layout: see "Speculation"; with exact products it is the same, for
  steps of the same size: see "Rounding" under "Sequence state"; the hash of a
  reply's tokens, which the server returns, and `"draft": false` check it over
  HTTP, and the test of that for the fast kernels is switched off until they
  agree); a prompt
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
   draft block (and with it the lookup in earlier text, which the model had
   before the draft policy and has not while it has no drafter); the prompt
   path's waves (steps of more than eight rows run on the many-row expert
   kernels; what the layer-major pass with waves has left to fix is described
   under "Qwen3.8-Flash-Next through the runtime"); reading the
   experts the next layers will pick before their routers have run; owners of
   experts that follow the heat; the tier of experts giving memory back.
3. Attention for models that attend a few thousand cells through lists: the
   kernel chosen by a measurement on the device, staged loads where they
   pay, one launch per row, items of twelve query heads, rows of a prompt
   that share cells; and the K/V formats with a tile mean in `qwen4exp`.
4. The prefix cache beyond the first pass: checkpoints at the message
   boundaries the server names (the anchor is a page boundary before the end
   of the prompt now), a warm pass in the background, spill of a sequence to
   RAM, pages and checkpoints that move between VRAM, RAM and `ares-store`
   (and so survive a restart), locking the RAM of the next chunk of places
   ahead of its first use, and `qwen4exp` with a cache (what that needs is
   under "Qwen3.8-Flash-Next through the runtime").
5. Shares of a tensor split from a measurement at load; device layouts for
   the remaining types (the i-quants of 2 and 3 bits stay in the file's layout,
   and with them the integer kernels for few rows); a choice of layouts that
   does not change with the draft block or from one load to the next; a row
   of a draft window computed with the bits it has alone, in the recurrent
   and attention layers too, so that drafts cannot change the text.
6. Vulkan: experts, the `qwen4exp` kernels, and the models (the sampler's
   kernels wait for a model that calls them); a sampler on devices without
   `shaderFloat64`, with the double-precision functions written in 64-bit
   integers; attention for
   windows of many rows on the cooperative matrices, and a faster decode of
   the `q8` and `q4` rows; a faster upload.
7. The router mode with `models.ini`, multimodal input, `Engine::decide` and
   `/v1/systemone` on the real engine. For tool calls: the tests on the real
   models, the model's own drafts for the free part of a reply with tools, and
   a format for templates that write calls otherwise.
8. A path for Turing; a run on any card that is not one of the two here.

## Host memory

There is one setting for RAM: the most the whole engine may use. Inside that
limit the users of memory make room for each other; nobody is given a fixed
share.

| User | Kind | When memory is short |
| --- | --- | --- |
| K/V pages of running sequences that do not fit in VRAM (KV offload) | Required | Never gives way; everything below shrinks for it |
| RAM tier of MoE experts | Elastic, high | Shrinks to the working set the running requests need (it does not shrink yet: nobody competes with it) |
| Row caches and staging | Elastic, middle | Shrink to a small floor |
| Prompt cache in RAM (pages of finished conversations) | Elastic, low | Goes first; its pages are on SSD as well |

A broker in `ares-core` keeps the books: a user asks to grow, and when the
memory is not free the broker names who must give back how much, lowest
priority first and never below a floor. KV offload of a running request can
therefore only fail when the limit is too small for the requests themselves,
not because a cache holds the memory. VRAM is handled the same way in spirit
but is planned ahead (ledgers), because moving things there is costly.

**What is built.** The broker is `PinnedRam` (`crates/runtime/src/ram.rs`),
shared by the executor and the device threads. It has three users so far:

- the step record and the read-back buffers of the devices, with the
  page-locked buffers a model's parts hold (required, booked at start:
  `required_ram`, `Model::pinned_ram`); for `qwen35` that includes the token
  embedding table (313 MiB in the small test file, 680 MiB in the 27B at
  Q4_K_S) and the staging buffer of the rows a step uploads from it, unless
  the device keeps the whole table (see "The token embedding of `qwen35`");
- the tier of experts of a mixture-of-experts model (elastic, priority 100,
  floor 0), which the model locks once at load: as much as the limit leaves
  and the experts that have no VRAM slot can use. It does not give anything
  back yet, since nothing ranks above it that could ask;
- the checkpoints of recurrent state of the prefix cache (elastic, the lowest
  priority, floor 0), which a device part locks in chunks of at least 32 MiB
  when a place is first used and keeps locked while the engine runs. They
  give way to the tier when both are asked for.

The limit is `GpuConfig::ram_limit`, which the server sets from `--ram`
(default: half of what the machine reports as available when it starts; the
log states it and the reason). For a model with a prefix cache, what the limit
leaves after the required memory is the checkpoint budget of the conductor
(`RuntimeConfig::checkpoint_budget`), so the places the conductor hands out
never lock more than the limit; for a mixture of experts it is the tier. Nobody
asks the checkpoints to give memory back yet, as nothing competes with them; a
lock that does not fit fails the step with the books in the message. The first
use of a place waits for the lock on the device's thread (21 to 22.5 ms for
93.5 MiB, 11.8 ms for 56.1 MiB, measured with the 27B); locking ahead in the
background is not built. `GET /metrics` reports the bytes locked
(`ares_checkpoint_ram_bytes`) and the counters of the expert store
(`ares_expert_*`).

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
went. Devices, split, shares, context, K/V format, precision, the model
drafters that take part (`--drafters`) and the room for their windows
(`--window`), and the RAM the engine may use (`--ram`) can be given instead. The RAM defaults to half
of what the machine reports as available when the server starts; the
engine's step buffers are taken from it, and the rest is the budget of the
checkpoints that let a conversation continue from what was computed (see
"Host memory").

It answers `POST /v1/chat/completions` and `/v1/completions`, streamed and
not, `GET /v1/models`, `/health` and `/metrics`, with timings in every answer
(prompt tokens and time, generated tokens and time, drafts proposed and
accepted, and `token_hash`, a hash of the reply's token ids in the final message
and in the last event of a stream). A request can switch drafting off for itself
with `"draft": false`; the two replies of one request, with and without drafts,
are then compared by their hashes over HTTP (they are equal with
`--precision exact`; with the fast kernels they are not always, see Status).
The prefix cache is on for `qwen35` without a flag: a prompt that
starts with text the engine has computed reads the cached part from the pages
and the checkpoint instead of computing it, the answer's `usage` says how much
(`prompt_tokens_details.cached_tokens`, as OpenAI does), `timings.prompt_n` counts
the tokens that were computed, and `/metrics` has the counters (starts from the
cache, cells taken from it, rows replayed, checkpoints saved, restored and
dropped, pages copied, pages cached, RAM locked for checkpoints). The sampling
fields of the OpenAI API and of llama.cpp's server are read under their names
(see [Sampling](#sampling)), `n` completions of a prompt are `n` requests that
read the prompt once, and the probabilities of the tokens come back in the
answer, streamed or not, and `grammar`, `json_schema` and `response_format` make the
reply follow a GBNF grammar or a JSON schema (see [Sampling](#sampling), "Structured
output"; the same fields on chat and on completions, and a model that thinks is free
while it thinks). A field of a request that the server does not hand to
the engine is refused by name, not ignored: images among
them today, and the sampler's few limits (a history of more than 1024 tokens
for the penalties, more than 20 alternatives, `min_keep` above 1, adaptive-p).
A client that disconnects frees its sequence. The
first signal lets running requests finish, the second ends them, and the
devices are released before the process exits.

**Thinking.** Qwen3.8 thinks before it answers: the template ends the prompt
with `<think>\n`, and the model writes its reasoning, `\n</think>\n\n` and the
answer. The server returns the reasoning in `reasoning_content` of the message
(and of the deltas of a stream, as their own chunks) and the answer in `content`;
a reply that `max_tokens` cuts inside the thinking has reasoning and an empty
content. An assistant message of a request may carry `reasoning_content` back,
and the chat template writes it into the prompt. What marks the reasoning is
read off the model's own template, not looked up by the name of a model: when a
model is set up, `ReasoningFormat::detect` (`ares-tokenizer`) renders small
conversations and compares the texts, as llama.cpp's autoparser does. An
assistant message with and without `reasoning_content` gives the pair of tags
and the white space the template writes around them; the end of the prompt with
`enable_thinking` on and off says whether the model continues inside the block
or the template closes an empty one; earlier turns with `preserve_reasoning` and
a few values of `reasoning_effort` say which of the two variables the template
reads. The result travels as `ares_api::ChatFormat` (`Model::chat_format`,
`Engine::chat_format`) and the server's `Splitter` sorts the text with it as it
streams: it holds back what may still be a tag and the white space the template
writes around the reasoning, and gives the same text however the pieces are cut.
A template with no pair of tags around the reasoning has no thinking, and its
replies stay whole. The newline the template writes before the closing tag is
not part of `reasoning_content` (llama-server leaves it in, which a template
that does not trim would write twice).

The controls are llama-server's, under its names and defaults:

| Control | Start | Request | Here |
| --- | --- | --- | --- |
| Thinking on, off or left to the template | `--reasoning on\|off\|auto` (`-rea`) | `chat_template_kwargs.enable_thinking`; `reasoning_effort: "none"` is off | the variable `enable_thinking`; `auto` sets nothing |
| Format of the reply | `--reasoning-format` (default `auto`) | `reasoning_format` | `none` leaves the text whole in `content`, the end of the prompt from the opening tag on included, as llama-server does; `auto`, `deepseek` and `deepseek-legacy` extract (llama-server parses the three the same way) |
| Effort | `--reasoning-effort` | `reasoning_effort` | the variable `reasoning_effort`; refused if the template does not read it |
| Thinking of earlier turns stays in the prompt | `--reasoning-preserve` (default on), `--no-reasoning-preserve` | `chat_template_kwargs.preserve_reasoning` | the variable `preserve_reasoning` |
| Any other variable of the template | `--chat-var` | `chat_template_kwargs` | the variable; `messages`, `tools`, `add_generation_prompt`, `bos_token` and `eos_token` are the server's |
| A budget of thinking tokens and the message injected when it is spent | `--reasoning-budget`, `--reasoning-budget-message` | `reasoning_budget_tokens`, `thinking_budget_tokens`, `reasoning_budget_message`, `reasoning_control` | not built: only -1, an empty message and `false` are accepted |

llama-server keeps the budget in its sampler, which forces the tokens of the
message and of the closing tag into the sequence once the budget is spent. The
engine cannot put a token into a sequence that the model did not sample, so a
request that asks for a budget is refused by name and is not answered without
one. A request is refused, too, for an effort the template does not read, for
`reasoning_effort: "none"` when `enable_thinking` does not change the prompt, and
for `reasoning_format`, `reasoning_effort` and `chat_template_kwargs` on
`/v1/completions`, which has no template.

**A conversation that continues from the cache.** The prefix cache continues a
conversation at the end of the last reply only if the next prompt, rendered from
what the client sends back, begins with the prompt and the tokens the model
generated. That is why the reasoning has to come back: the template of
Qwen3.8 writes `<think>\n`, the reasoning, `\n</think>\n\n` and the content
into the prompt for an assistant message, and without `reasoning_content` it
writes an empty block where the model had begun to think, so that the new
prompt parts from the cached sequence at the last token of the old prompt
(the newline of `<think>\n` is one token with the next, `<think>\n\n`). The
server does not cut or reshape what it returns for this: the reasoning is the
text between the tags without the layout the template writes itself, and the
content is the rest, both exactly as generated, so that the template writes the
turn back as the model generated it. `reasoning/templates.rs` in `ares-server`
plays that through the real templates and vocabularies of Qwen3.8-27B and of
Qwen3.8-Flash-Next (their templates declare the same and give the same
numbers) and counts the tokens of the sequence that the next prompt does not
repeat:

- None, for a reply with thinking (German with umlauts and emoji, code blocks,
  blank lines, tags that are only text), for a reply without thinking (the
  template closes an empty block in the prompt), for a reply the limit of
  tokens cut inside the answer or inside the thinking (also right after a full
  stop), and over four turns in a row, one with thinking off.
- Where the template trims or lays out the text other than the model did:
  a model that thinks of nothing (`\n</think>` right after `<think>\n`) parts at
  the last token of the prompt and the reply is computed again (the two
  newlines are two tokens for the model and one for the tokenizer); a newline
  after the answer costs it and `<|im_end|>` (2 tokens), a newline before the
  reasoning the whole reply, an empty line at the end of the reasoning 6
  tokens, an extra empty line or a single newline after the closing tag 4, a
  closing tag without the newline before it 5 (the replies of the test, 66 to
  68 tokens in all).
- Tokens the model sampled that are not the tokenizer's tokens for their own
  text: a sampled reply of 1500 tokens in the test through the server wrote
  `...` and `"?` where the tokenizer makes `..."` and `?`, and the next prompt
  parted there (at token 1128 of 1867, so the engine cached the 1024 of the
  whole pages before it). Greedy replies did not do it in the tests; a
  test with the real vocabularies (`tokens_that_are_not_the_tokenizers_own...`)
  holds the arithmetic of such a parting.
- Qwen3.5-0.8B's own template is another one: it writes the thinking of a turn
  into the prompt only after the last user message, `preserve_reasoning` or
  not, so no reply is ever repeated: the new prompt parts 2 tokens before the
  end of the old one (4 with thinking off), whatever the client sends back.

Through the server, with the 0.8B on a GPU and served with the template of
Qwen3.8-27B (read from its file; `tests/reasoning_cache.rs`, asked greedily), a client that sends
`reasoning_content` back gets this from the engine's report of each request
(`Snapshot::requests`) and `/metrics`:

| Conversation | Turn | Prompt tokens | Cached | Computed | Replayed |
| --- | --- | --- | --- | --- | --- |
| Short questions, 314, 259 and 158 tokens generated | 1 | 66 | 0 | 66 | 0 |
| | 2 | 406 | 379 | 27 | 0 |
| | 3 | 691 | 664 | 27 | 0 |
| A system message first; the first two replies cut by the limit of 600 tokens | 1 | 405 | 0 | 405 | 0 |
| | 2 | 1035 | 1004 | 31 | 0 |
| | 3 | 1665 | 1634 | 31 | 0 |

The cached tokens of a turn are the prompt and the tokens generated in the turn
before but the last, which is generated and never computed (the end token, or
the token that the limit cut off); what is computed is that token, the line
break after the reply, the new message and the start of the next reply. The same
long conversation with a client that sends back only `content` (turn 2: 434
prompt tokens, 384 cached, 50 computed; turn 3: 463, 384, 79 computed; a
sampled run had 128 rows replayed in turn 3) reads the cached whole pages
before the last token of the old prompt and computes the rest again, up to a
page, with a replay of the rows between the last checkpoint and the cached
cells when the two do not meet. Measured
through the server before the reasoning came back, Qwen3.8-27B with a history
of 28 000 tokens computed 80 to 165 rows again in every turn, the first token
after 0.34 to 0.55 s where the continued sequence gives about 0.15 s. A resume
point at the end of the prompt, which would shrink that cost for a client that
drops the reasoning, is another piece of work and is not built here.

Measured 2026-11-11 with Qwen3.8-27B on both cards: 79.2, 75.3 and 78.8
tokens per second for German, English and code through HTTP, which is what
the same model gives without HTTP; the first token after 76 to 121 ms; two
requests at once at 63 tokens per second each.

**Tool calls.** A chat request can bring `tools`. The chat template writes
them into the prompt, and the model calls one by writing the call into its
reply in the form the template writes the calls of earlier turns. That form is
read off the template, not looked up by the name of a model: when a model is
set up, `ToolFormat::detect` (`ares-tokenizer`) renders an assistant message
that calls made-up tools with made-up arguments, takes the texts between the
names, the keys and the values as the pieces of the format, and accepts them
only if they write the rendered calls again exactly. One form is understood,
the one the templates of Qwen3.8-27B, Qwen3.8-Flash-Next and Qwen3.5-0.8B
write (llama.cpp knows it as Qwen3-Coder's):

```text
<tool_call>
<function=NAME>
<parameter=KEY>
VALUE
</parameter>
</function>
</tool_call>
```

A value that is a string stands as it is, every other value is JSON. After the
text of a message the template writes an empty line before the first call, and
a line break between two calls. The format travels in `ares_api::ChatFormat`
next to the thinking. A template that writes calls in another form (one JSON
object between the tags, as Hermes' does) has no format here, and a request that
brings tools for it is refused with a 400 that says so; a format for such
templates is not built.

The reply is held to the calls by a grammar and not trusted to be one.
`ares_grammar::calls` writes it in the engine's Lark syntax, after the grammar
llama-server builds for this format (`common/parsers/qwen3-coder.cpp`): the name
is one of the tools of the request; the arguments are the properties of the
tool's schema, the required ones first, each once, in any order (in the order of
the schema when there are more than six, llama.cpp's limit), then the others in
any order; a value whose schema allows no string is JSON that the schema accepts
(through the engine's JSON compiler, with the definitions of the tool's schema
for its references); a value that can only be a string is text up to the closing
tag, or one of the strings of its `enum` or `const`; a value that may be either
is text up to the closing tag. A call may be followed by white space and, when
several calls are allowed, by another call; then the reply ends. That holds
whether a tool says `strict` or not. A string is written raw, so the
`pattern`, `minLength`, `maxLength` and `format` of a string are not enforced.

| `tool_choice` | The reply |
| --- | --- |
| `auto` (the default) | free until the model writes `<tool_call>` (the rule's trigger); from there calls, and nothing after them |
| `required` | calls from the first token of the answer, after the empty line the template writes there: at least one |
| a named function | the same, and every call is of that function; the prompt still lists all tools |
| `none` | text: the tools are in the prompt, no rule binds the reply, and a call the model writes stays in `content`, as with llama-server |

`parallel_tool_calls: false` allows one call. The reasoning of a model that
thinks is free in every case, and the rule (or the wait for the trigger) starts
after its closing tag; a call the model writes inside its thinking is reasoning.

The server reads the calls out of the answer as it comes (`CallSplitter`, on
what the reasoning's splitter calls content): it holds back what may still be
markup, so that the pieces of a stream are the same however the tokens cut the
text and no markup reaches `content`; the white space between the content and a
call, and after the last call, is the template's layout and belongs to neither.
A call comes back in `tool_calls` with an id (`call_`, the request's id and the
place of the call), the name and the arguments as the text of one JSON object,
written as llama-server writes it (`{"city":"Köln","days":3}`): a string of the
format as a JSON string, any other value as the JSON the model wrote, and for an
argument that may be either, JSON if it is JSON of a kind the schema allows and
a string otherwise. A stream sends OpenAI's deltas: one with the place, the id,
the type and the name of a call, then fragments of its arguments. `content` of a
reply that is only calls is the empty text, as llama-server writes it.
`finish_reason` is `tool_calls` when the reply ended by itself (or at a stop
string) and has a call. Assistant messages with `tool_calls` and messages of
the role `tool` are checked (a name, arguments that are a JSON object) and
handed to the template, which writes them into the prompt.

A reply that the limit of tokens cuts inside a call ends with `length` and
returns the call as far as it came: the name if it was whole, and arguments that
are not whole JSON (`{"city":"Köl`); a call whose name was cut is not returned.
llama-server and OpenAI do the same. A stop string that cuts a call gives the
same call with `tool_calls` as the reason, as llama-server does.

The text with drafts is the text without, as for every rule (host tests on the
synthetic model: `crates/runtime/tests/tools.rs` under the rule, and
`crates/server/tests/tools_engine.rs` from the request to the answer, where the
reply is also the same whole and streamed). A request with tools that may be
called has a rule, and a request with a rule takes no drafts from the model's own
draft block (see [Sampling](#sampling)): that holds for its whole reply, the
thinking and the free text before a call included, where no mask binds. To give
those rows the model's drafts the conductor would have to stop accepting them at
the token that makes the rule bind; that is not built.

On the vocabulary of the 0.8B (248,320 tokens) the grammar of a tool compiles in
about a millisecond, and the mask of a step along random walks through it takes a
median of 2 microseconds inside the text of a string argument (3 at the 90th
percentile, 1.8 milliseconds at the longest) and 3 inside JSON arguments (5, and
53 microseconds at the longest); host time on one core, with the machine under
load (`mask_cost` of `ares-grammar`, which has the two grammars).

Where this is not llama-server: `required` there lets text come before the first
call, here the answer is a call from its first token; a named function is read
as `auto` there (it reads `tool_choice` as a string), here it is enforced; a
`response_format` or `json_schema` next to tools wins over them without a word
there and is refused here, as `grammar` is by both (with `tool_choice: none`
the other rule applies and the tools are only in the prompt); llama-server takes
`<tool_call>` inside the thinking as the end of the thinking and the start of a
call, here it is reasoning; the strings of an `enum` are enforced here and any
text is accepted there; the white space between the content and a call is not
part of `content` here. The legacy `functions` and `function_call` are refused.
The tests with real models (`crates/server/tests/real_tools.rs`: the 0.8B on
one card, the 27B on two) are written and have not been run.

For a mixture of experts (`qwen4exp`) the plan is another one: where the
layers go (`--split tensor`: both devices on every layer, `layer`: a range
each, by default the first when each device can write the other's memory), the
context (without `--context`, what a thirty-second of the room for expert slots
holds), and the RAM (`--ram`, whose limit less what the engine needs is the
tier of experts). The server reads a short text of German, English and code
before it listens, so that the first request does not find every slot empty. It
has no prefix cache, no `--shares`, no `--precision exact`, no `--draft-vocab`,
no `--window` and no model drafter (`--drafters mtp`), and it does not
speculate; asking for one of the options is an error that names it. Thinking is handled as for the
27B: the model declares what its chat template says (`Model::chat_format`), so
the reasoning of a reply comes apart from the content, in streams too, and
`--reasoning` and the controls of a request work the same way; a reply that
`max_tokens` cuts inside the thinking has reasoning and no content.

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
