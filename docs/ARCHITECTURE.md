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
| `ares-store` | State on disk: pages and checkpoints by content hash in segment files, per fingerprint, with a size limit, read and written with direct I/O outside of its index | Done as a store of opaque bytes, tested against restarts, the limit and damaged files; the prefix cache of `qwen35` keeps its level on SSD in it (`--cache`) |
| `ares-io` | Reads of weights straight from the model file: io_uring with direct I/O, demand before prefetch, cancel and promote, copies of a file on several drives with the reads spread over them, a map from experts to file ranges, row reads behind a cache | Done, tested against real files and on two drives; 1.4 to 1.7 GiB/s of random expert-sized reads per SSD of the development machine and 2.9 to 3.0 GiB/s over both; the expert store and the n-gram rows of `qwen4exp` read through it |
| `ares-sched` | Step planner (generation, decision rows and prompts in one step), admission (pages claimed per step, a reservation that corrects itself, a queue instead of refusals), the order of a pass for long prompts of a model with experts, step time model with a term for the context, draft length choice, the lookup drafter on a suffix automaton shared by all sequences | Done; unit and model tests; the conductor uses the planner, admission (without spill), the lookup drafter and the step cost |
| `ares-weights` | Expert heat, tiers that are elected and fed by reads (admission, protection, guesses bound late, slots given back), slots per layer by need, the waves of experts of a pass, a token table for the first layers, the lookahead distance, routing bias for pruning | Done, unit tested; the expert store uses the slot sets, the heat, the swap rule and the split of slots; the guess queue, the token table, the waves and pruning are not used by a running model yet (the lookahead distance only by a guess of `qwen4exp` that is switched off) |
| `ares-api` | Request (with every sampling field), event (text and probabilities), decision and error types; the `Engine` trait between HTTP and the runtime (`submit` for generation, `decide` for decisions, `stats`, `chat_format` for what a model's chat template declares about thinking) | Done, unit tested |
| `ares-server` | The `ares` binary: loads a model on the devices and serves it over an OpenAI-compatible HTTP API, with the plan of devices, split, shares and context made from what the devices report | Serves `qwen35` and `qwen4exp` models (chat and plain completions, streamed and not, every sampling field and the probabilities of the tokens, `n` completions, timings, metrics, staged shutdown, the reasoning of a reply apart from its content and in the history, with llama-server's controls for thinking, tool calls held to the tools' schemas by a grammar, also when the model opens the call inside its thinking); 213 tests without a GPU, 17 that start the real binary (some need the 27B or the files of Flash-Next) and 26 that run the engine in the test's process (the 8 of tool calls have not been run); for `qwen35` the prefix cache is on, with `--ram` for the engine's RAM, for `qwen4exp` the RAM is mostly the tier of experts; a router mode that serves several models from a preset file, loads them on request and unloads them for room or when idle (host tests of the preset reader and of the state machine on a scripted loader; its 2 tests on real devices have not been run); no multimodal input, `/v1/systemone` answers 501 |
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
- A row reads a tile converted from the position at which the tile has left
  the newest cells (`newest + 128 * (tile + 1)`), whatever step computes the
  row. A draft window that reaches across such a position has the tile both
  ways: it is converted at the start of the step and keeps its exact entry;
  the rows from the position on read the converted entry (the plan of their
  attention names it for that one tile, the table does not), the rows before
  it the exact one, and a step that starts past the position gives the exact
  entry up. So a row of a window reads the bytes it would read in a step of
  its own. The tile must have been full before the window and only one tile
  may leave inside it, so with these formats a draft window has at most
  `min(newest, 128)` rows (32 with `q7`, 64 with `q6`, 128 with `q5` and
  `q4`); a longer one is refused by the device part.

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

**On SSD.** The prefix cache has a level on SSD (`ares --cache <dir>`;
`crates/runtime/src/cache/ssd.rs` over `ares-store`). What a sequence
computes is written to a store as it is computed, and a prompt whose start is
in the store and not in the devices has it brought back before it starts. A
conversation therefore resumes after its pages left the devices, after its
model was unloaded (router mode), and after a restart of the server. A model
takes part when its device side moves state to the host and back
(`Model::durable`, `DeviceModel::transfer`); `qwen35` does, `qwen4exp` has no
prefix cache yet.

- **The store** (`ares-store`) keeps two kinds of records under the chain
  hash of a page, the page and the checkpoint of recurrent state at its end,
  in a directory per fingerprint. A record is a header block and one part per
  device, each padded to 4096 bytes, so parts are written from and read into
  aligned buffers with direct I/O (through the page cache on a filesystem
  without it). Pages and checkpoints lie in segment files of their own (256
  MiB): a conversation's pages sit together, and its checkpoints, which later
  turns leave behind, age and go without taking pages along. The index is
  rebuilt from the record headers when the store is opened. One process uses
  a store at a time (a lock file); a second server on the same directory and
  fingerprint runs without the level and says so.
- **Bytes move outside of steps.** The conductor calls
  `Executor::transfer` between steps (`Cache::pump`, at the top of its loop):
  the device threads copy pages and checkpoint places to or from buffers in
  host memory and wait for their streams. One writer thread writes the
  buffers to the store and syncs the drive a second after its last record
  (and every ten seconds while records keep coming); four reader threads
  read and check what comes back. The step loop does not wait for the drive:
  pages that the drive takes slower than they are computed stay queued (they
  are in the pool), and a page the pool gives up before its turn is not
  written. The one exception is a checkpoint that must leave its place when
  three are with the writer already: the loop waits up to 250 ms for a
  buffer, and gives that checkpoint up after that. The store's index is locked to reserve or
  find a record, never while bytes move. The buffers (16 pages and up to 3
  checkpoints, made when first used) are ordinary memory aligned for direct
  I/O, not page-locked, so a copy to or from them blocks its device thread,
  which is what a transfer does anyway.
- **What is written when.** A full page is queued at the commit after which
  it has the form it keeps: at once with `f16` and `q8` and no draft block,
  and else when the sequence's last step began past it by the newest cells
  the device keeps exact (`Durable::settle_cells`; a device converts the
  tiles behind them at the start of a step). The rest is queued when the
  sequence ends, gives way or is cancelled. A
  checkpoint at a page boundary (the anchor of a prompt, the intervals in a
  prompt and while generating) is copied out of its place in the pump that
  follows the step whose record carried its save, while the place still holds
  it. A record the store has is not written again. Nothing is written when a
  model is unloaded or the server stops beyond what was already queued.
  The partly used last page of a sequence and the state at its end are not
  written: after a restart the next turn restores the checkpoint before them
  (the anchor of the last prompt, or a later one from the reply) and computes
  from there, which costs the rows of the last reply once.
- **The rule against wear.** Nothing of a text is written before its
  sequence holds 2048 settled cells; then its pages are written from the
  first on, and its checkpoints from that depth on. Below that a text is
  computed again in about a second and a half on the 27B, and what would be
  written is mostly its checkpoint, which is as large as the pages of several
  thousand cells. Pages of a short prompt that a long one later starts with
  are written then, by the long one.
- **A hit.** `Cache::probe` asks the store for the pages after the ones the
  index has. For a model with recurrent state the run ends at its deepest
  checkpoint in the store (the pages behind it would be replayed row by row,
  which takes as long as computing them), and without a checkpoint nothing is
  brought back. The run is announced in the index under an owner of its own,
  so the prompt, and any other prompt that shares it, is held as for a page
  that another sequence computes. The readers bring the pages and the
  checkpoint into host memory, the conductor gives each page a page of the
  pool (cached pages nobody holds make room), the devices take the bytes, and
  the pages enter the index in order, the last one with the checkpoint, which
  went into a checkpoint place. The prompt then starts as from any cached
  prefix. A record that cannot be read or fails its checksum ends its run and
  is forgotten; the prompt computes from there.
- **Forms.** With `q7` to `q4`, and for the draft block always, a page is an
  exact entry while it is the first tile or among the newest cells and a
  converted one later. A page goes to the store in the form it has, with a
  tag; the delay above lets most pages be written converted. A page that
  comes back exact takes an exact slot when it is the first tile or lies at
  the end of the run, and is converted on the way in otherwise, as the window
  of a sequence that went on would have converted it.
- **The fingerprint** covers everything besides the tokens that changes the
  stored bytes: the model's files (their size, their first 64 KiB and 16
  samples of 4 KiB spread over each), the formats of K and of V, the draft
  block and its formats when the model drafts, the format of the recurrent
  state, how the products round (`--precision`), the page size, and which
  heads of every block each device holds (the bytes are stored per device, so
  another split or other shares are another store). The names of the pages
  start from the fingerprint, not from the model's id, so a model keeps its
  store under another name. State of another fingerprint is never read; it
  counts against the size limit and is the first to go.
- **The size limit** (`--cache-size`, 32 GiB by default) is for everything in
  the directory. When it is passed, stores of other fingerprints that no
  process has open go first, then segments none of whose records a prompt can
  still reach (a page is reachable while every page before it in its chain is
  stored, a checkpoint while its page is), then the segment used least
  recently; the order of use survives a restart in the files' modification
  times. Several models that are loaded at once keep within the limit each.
- **Failure.** The level is a cache and never reaches a request: a record
  that cannot be written is dropped and counted, a full drive gives up old
  segments and the write is tried once more, a directory that vanished is made
  again (what was written between its loss and the next sync is lost), and a
  damaged or missing file costs the runs that needed it.
- **RAM.** A checkpoint goes to the drive and comes back through a
  checkpoint place, so one place is what the level needs of the RAM limit
  (the server counts it as required when `--cache` is given); with more room
  the places keep the hot checkpoints as before.
- **Measured** 2026-11-15 with the server and `--cache`: Qwen3.8-27B Q4_K_S
  on the RTX 4070 and the RTX 3060, tensor split 0.625 / 0.375, two sequence
  slots, the model's draft block on. A conversation whose first prompt has
  29 322 tokens (a list of 1900 lines and a question) and whose second turn
  adds the reply and another question (29 354 tokens); requests on
  `/v1/completions`, most probable token, time to the first token of the
  stream. The store lies on the btrfs volume (an NVMe drive, compressed with
  zstd) that the model files are read from, and other work used the machine
  and that drive throughout: the load average is given with every time, and
  `dd` with direct I/O and blocks of 4 MiB wrote 1.0 to 1.2 GB/s and read 1.9
  GB/s when the drive was free, and read 160 to 730 MB/s when it was not.

  | First token after | K and V in `q8` | K and V in `q5` |
  | --- | --- | --- |
  | Computed (nothing cached) | 30.39 s (load 11); without `--cache` 30.73 s (load 17) | 34.77 s (load 19) |
  | Second turn, from the devices | 0.150 s (load 11); without `--cache` 0.144 s | 0.191 s (load 19) |
  | Second turn after the pool gave up 92 of its 229 pages and their checkpoint | 0.59 s (load 23) | not measured: the pool of `q5` held the conversation |
  | Second turn in a new process, all 229 pages and the checkpoint from SSD | 1.15 s (load 17); 2.1 to 3.0 s in four more runs (load 12 to 40, the drive busy) | 2.90 s (load 18, the drive busy) |

  The replies were the same in every row. What the store took: with `q8` a
  page of 128 cells is 4.62 MB (36.1 MB per 1000 tokens), with `q5` 3.17 MB
  on average (24.8 MB per 1000 tokens; the pages a request leaves behind its
  end are written as exact entries, and the first page always), and a
  checkpoint is 157 MB, one for every turn whose prompt ends on a new page.
  The first turn with `q8` wrote 229 pages and a checkpoint, 1.21 GB, in 1.28
  s of the writer's time (0.95 GB/s); over a whole run the writer needed 44 s
  for 3.9 GB, since a sync took seconds whenever other work wrote to the
  volume. The step loop does not wait for the writer: reading the prompt
  took as long with the store as without it, and of that time the loop spent
  0.30 s in transfers (1.0 %). A request with a prompt of 2970 tokens that
  then writes 600 tokens (27 pages and a checkpoint to the store) took 5.27 s
  with the store and 5.01 s without it, eight runs each at loads of 14 to
  21; the transfers were 0.06 to 0.32 s of it, and its tokens came at 218
  per second against 233. A page costs a transfer of 3 to 5 ms and a
  checkpoint one of 30 to 60 ms, each once, so a long generation pays about
  one percent. In the run of 1.15 s the four readers read the 1.21 GB in
  0.89 s each (1.36 GB/s together, against the 1.9 GB/s of `dd`), the
  devices took the bytes in 0.25 s, and the rest is the 42 rows after the
  checkpoint; with the drive busy a reader needed 1.0 to 2.5 s. The buffers
  in transit are ordinary memory: with the machine at a load of 33 to 40 the
  copies to the devices took 0.9 to 1.5 s instead of 0.25 s, which
  page-locked buffers would not.
- **Across loads.** The state that comes back is bit for bit the state that
  was written, so a resume from SSD in the same load equals a resume from the
  devices where the same checkpoint is used. A later load may choose other
  kernel layouts for its products (they are measured at load), whose rounding
  differs from the load that wrote the state: the reply after a restart is
  the reply a server gives that continues that state, which is not bit for
  bit what the new load would have computed for the prefix itself. The texts
  of the tests, which copy from their prompts, are the same.

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
| MTP | The model's own draft head or block | Done for `qwen35` on one device and under a tensor split: the block runs on the device with the embedding table, its chain and its picks stay on the device (`device/mtp.rs`); not under a layer split. Chains of up to the window's rows less one (7 by default). Done for `qwen4exp` in every placement, from a file of the block (`--drafters FILE`): the block runs on the device with the output head and the host reads every pick (see "Qwen3.8-Flash-Next through the runtime") |
| DFlash, DSpark, DFlash2 | A small draft model that reads the target's hidden states at a few layers and proposes a block of tokens at once | Not built: it needs taps on the target's layers and a second model on the same devices. `--drafters FILE` is refused with "not built yet" for a `qwen35` model (the file is not opened; for a `qwen4exp` model the file is the model's own draft block); the choice of a window takes any number of drafters |
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
sampled), and nothing for its head, which is rows of the output head.

**The text with drafts is the text without**, token for token, with the fast
kernels as with exact products, for the most probable token and for sampled
tokens with the same seed. A step verifies a window of rows of a sequence, and
row `k` of it gives the logits that a step of that row alone gives after the
`k` rows before it were committed, bit for bit: the weight kernels give a row
the same product in any launch of a decode step, the recurrent blocks carry
their state from row to row as single steps store it, and attention plans
every generated row as if it were the only row of its step (see "Attention
kernels" under [Devices](#devices)). The tests: every kind of block, every
format of K and V, windows with rejected drafts, on made-up models
(`crates/model/tests/qwen35_windows.rs`: every logit, the recurrent state and
the stored K and V); the 0.8B on one card, in a tensor split and in a layer
split with every format (`a_row_of_a_draft_window_picks_what_it_picks_alone_on_every_placement`
in `tests/qwen35_device_runtime.rs`); the kernels and the plan in `ares-cuda`
(`a_row_of_a_window_has_the_bits_it_has_alone`); and the 27B over HTTP by the
hash of a reply's tokens (`ares-server`'s `tests/real.rs`).

It was not so before 2026-11-15, with the fast kernels. Measured then with
`qwen35_run --engine 8 --undrafted` (German prose, English prose and code, 300
tokens each, the most probable token and sampled with seed 20261110; every
request with drafts, with `"draft": false` and with drafts again in one
runtime): with K and V in `q8`, the default, 6 of 6 replies of the `IQ2_XXS`
file on the RTX 4070 parted from the reply without drafts, after 213 to 845 of
1100 to 1500 characters, and 4 of 6 of the `Q4_K_S` file over both cards, after
285 to 867; with K and V in `f16` none of 6 parted on either. The second reply
with drafts was the first in all 24. After the change, the same day, with
`q8`: none of 6 parts on the RTX 4070 and none of 6 over both cards.

The cause was attention and nothing else. A probe ran the blocks of a step
one by one (`Harness::blocks`) and fed every block of a window the rows that
the single rows had fed it (Qwen3.5-0.8B, windows of 4 and 8 rows after 200 to
700 tokens): every recurrent block, with its feed-forward half, its norms and
its products, gave a row of a window the bits of the row alone (all 216 and
432 rows) and left the same state after the commit; every attention block
gave other bits (none of 72 and 144 rows, off by up to 7e-3 of the block's
largest output) while the stored K and V were the same bytes. With K or V in
`f16` every row was equal, and with `q4` as long as every tile was still an
exact entry. What depended on the step was the read:

- *The kernel.* One row over quantized K and V in converted entries ran on
  the direct kernel (queries of 15 bits, integer products), a window on the
  tensor cores (f16 operands). The two differ by about 1e-3.
- *The pieces.* A row's cells are cut into pieces whose states are merged in
  rounded sums; the cut followed the last row of the window and every other
  row of the step, so a window that began a new table entry cut its earlier
  rows otherwise than a step of their own would (once a sequence has more
  entries than a row has pieces).
- *With `q7` to `q4`, when a tile is converted.* A step converted the tiles
  that had left the newest cells as of its first row; a later row of the
  window read a tile exact that it reads converted in a step of its own.

The earlier measurements fit this: rows of different sequences in one step
were equal to the rows alone (each is an item of its own, on the direct
kernel), rows of one sequence were not, and the difference did not grow with
the context.

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

A request with corrections that depend on the path (penalties, DRY) takes no
drafts from the model's own draft block: those are made on the device while the
step runs, and the host cannot correct a row for drafts it does not have yet. It
takes the lookup drafts, which the host has, wherever the model has a draft block
(the lookup joins next to a model drafter and never runs alone). A request with
a mask is in the same position only while its rule binds: where the rule leaves
the reply free it takes the model's drafts like any other request (see "Drafts"
under "Structured output" below).

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
placement, with a rule about the tokens and with a grammar that ends its reply
as well
(`every_sampler_feature_gives_the_same_tokens_with_and_without_drafts_on_every_placement`
and `requests_with_different_settings_in_one_step_do_not_affect_each_other_on_every_placement`
in `crates/runtime/tests/gpu.rs`), and `crates/runtime/tests/sampling.rs` checks
the records of the windows against the definitions. The lookup in earlier text
drafts only next to a model drafter, so the tests on a real model use the 27B and
its draft block, on both cards with exact products. There the texts were equal
with drafts and without in all 14 settings through the server
(`crates/server/tests/real_sampling.rs`: 437 draft rows verified, 425 of them
right), equal under two rules about the tokens, and equal for requests that run
next to each other. Since 2026-11-15 a row of a draft window has the logits of
the row alone bit for bit with the fast kernels too (see "The text with drafts
is the text without" under [Speculation](#speculation)), so the same holds as
a model is served. What a row still depends on is the steps that computed its
prompt and, past 16 generated rows in a step, the size of the step ("Rounding"
under "Sequence state"): where that shows, a sampled text can part where two
tokens are that close, and the tests of requests next to each other assert
that the texts agree for a long stretch and that the probabilities agree on it
within the model's error.

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

**Cost.** The time of the sampler's kernels for one step with 248 320 logits a
row, from the launch to the end of the stream, in microseconds: the median of
40 picks, for 1 / 4 / 16 rows. `vk_sampler_bench` (an example of `ares-vulkan`)
runs both backends on the same card with the same rows and the same method;
`the_time_of_a_pick` in `crates/cuda/tests/sampler.rs` prints CUDA's alone.
Corrections are 64 entries a row, a mask allows a third of the tokens,
probabilities are the five best. Measured on 2026-11-12, each backend alone on
one card: three runs of the benchmark with CUDA's kernels as they were
(commit 6c27759) and three with the kernels described below, in pairs back to
back, while other work compiled and other processes used the cards between
the runs (load average 4 to 10). A figure is the median of its runs (six
for Vulkan, whose sampler is the same in all); the run furthest from it was
35 percent away for Vulkan, 31 for CUDA before and 30 for CUDA now.

RTX 4070:

| Row asks for | Vulkan | CUDA before | CUDA now | now / Vulkan |
| --- | --- | --- | --- | --- |
| most probable token | 32 / 49 / 128 | 36 / 76 / 244 | 23 / 29 / 57 | 0.71 / 0.59 / 0.45 |
| top-k 20, top-p 0.95 | 139 / 196 / 535 | 123 / 203 / 545 | 98 / 127 / 228 | 0.71 / 0.65 / 0.43 |
| top-p 0.95 | 93 / 138 / 394 | 84 / 148 / 431 | 72 / 107 / 251 | 0.77 / 0.77 / 0.64 |
| most probable token, penalties | 36 / 50 / 111 | 41 / 82 / 245 | 27 / 32 / 57 | 0.76 / 0.65 / 0.51 |
| most probable token, a mask | 37 / 56 / 150 | 44 / 91 / 266 | 30 / 39 / 75 | 0.81 / 0.70 / 0.50 |
| top-k, top-p, penalties and a mask | 145 / 200 / 498 | 129 / 207 / 553 | 105 / 137 / 258 | 0.72 / 0.68 / 0.52 |
| most probable token, probabilities | 95 / 171 / 520 | 103 / 225 / 682 | 60 / 92 / 209 | 0.63 / 0.54 / 0.40 |
| top-k, top-p, probabilities after the chain | 180 / 262 / 692 | 163 / 260 / 691 | 123 / 160 / 292 | 0.68 / 0.61 / 0.42 |
| top-n-sigma, typical, top-k, top-p | 228 / 342 / 865 | 192 / 304 / 799 | 175 / 239 / 484 | 0.77 / 0.70 / 0.56 |

RTX 3060:

| Row asks for | Vulkan | CUDA before | CUDA now | now / Vulkan |
| --- | --- | --- | --- | --- |
| most probable token | 80 / 110 / 240 | 70 / 151 / 411 | 42 / 77 / 119 | 0.52 / 0.70 / 0.50 |
| top-k 20, top-p 0.95 | 240 / 459 / 1154 | 170 / 407 / 1169 | 137 / 243 / 602 | 0.57 / 0.53 / 0.52 |
| top-p 0.95 | 184 / 316 / 842 | 132 / 290 / 897 | 107 / 201 / 543 | 0.58 / 0.64 / 0.64 |
| most probable token, penalties | 70 / 108 / 223 | 64 / 145 / 409 | 37 / 63 / 110 | 0.53 / 0.59 / 0.49 |
| most probable token, a mask | 87 / 127 / 318 | 76 / 170 / 483 | 48 / 90 / 186 | 0.55 / 0.71 / 0.59 |
| top-k, top-p, penalties and a mask | 209 / 420 / 1165 | 167 / 422 / 1217 | 141 / 269 / 676 | 0.67 / 0.64 / 0.58 |
| most probable token, probabilities | 178 / 366 / 978 | 156 / 448 / 1267 | 97 / 183 / 451 | 0.54 / 0.50 / 0.46 |
| top-k, top-p, probabilities after the chain | 286 / 554 / 1505 | 215 / 508 / 1437 | 160 / 320 / 745 | 0.56 / 0.58 / 0.50 |
| top-n-sigma, typical, top-k, top-p | 346 / 692 / 1894 | 252 / 613 / 1772 | 227 / 478 / 1178 | 0.66 / 0.69 / 0.62 |

*Why CUDA's kernels were the slower ones.* The two backends ran the same
algorithm, statement for statement, and Vulkan's was up to 2.2 times faster
where a step picks for several rows (the table: 111 against 245 microseconds
for the most probable token of 16 rows with penalties on the RTX 4070). The
device time of each part of a pick (`sampler_steps`, an example of
`ares-cuda`, records an event on the stream after every part) showed where it
went: of the 244 microseconds of the most probable token of 16 rows on the RTX
4070, 225 were the kernel that finds the best eight logits of each chunk of
4096, sixteen times what reading the rows takes (a kernel that only reads the
16 rows and takes one maximum per chunk needs 14 microseconds, 51 on the RTX
3060). It was instructions and barriers, not memory: every thread kept a
sorted list of its best eight among 16 logits, which the CUDA compiler puts
into the thread's local memory, and the block took its best eight in eight
rounds of a reduction over all threads, 17 barriers and 160 shuffles a thread.
The kernel with twenty candidates (the probabilities) took 281 microseconds
the same way. The levels of a radix descent came next: each built a histogram
of 1024 or 2048 bins in shared memory in every chunk, cleared before and added
to the row's after, 38 microseconds a level for 16 rows however few tokens the
level had. What the Vulkan compiler makes of the same statements is faster;
where exactly was not looked for, because both did work that is not needed.

*What CUDA's kernels do now*, for the same records bit for bit:

- **The best of a chunk.** A thread reads its 16 logits into registers and
  finds its own best. The threads with the same lane are a column; the eighth
  best of the 32 columns' bests (the twentieth, for the probabilities) is a
  bar that at least eight logits reach, so the best eight of the chunk do.
  Few threads have a best that reaches the bar; each counts how many of them
  have a better one, and the at most eight with fewer than eight above them
  put their logits that reach the bar on one list, where every entry finds
  its place by counting the entries above it. Four barriers, no list per
  thread. The kernel alone, 16 rows: 225 to 39 microseconds on the RTX 4070
  and 354 to 86 on the RTX 3060; with twenty candidates 281 to 40 and 591 to
  97. Rows made to be hard for it (every logit equal, descending, the best
  tokens all in the columns of one thread) take 36 to 53 microseconds on the
  RTX 4070, less than before in every case, and are in the tests
  (`rows_whose_best_tokens_crowd_together_equal_the_host`).
- **A thread's logits at once.** A kernel that passes over a row read a
  logit, used it and read the next, so a thread waited for the memory sixteen
  times. The top, histogram, pass and draw kernels read the thread's 16
  logits before they use one. (The mask kernel got slower that way, 21 to 31
  microseconds for 16 rows on the RTX 4070, and is as it was.)
- **Histograms of few tokens.** The kernel that decides a level leaves in the
  row's state how many tokens the bin it chose holds, which are the tokens of
  the next level, and after the last level how many the set still has. A row
  with at most 256 tokens for a histogram adds them to the row's histogram
  directly, with no histogram per chunk; after top-k 20 that is every level
  that follows. The numbers choose how a histogram is built and nothing
  else. A descent by count (top-k, the hundredth token of mirostat 1) holds
  no weights. The six levels of top-k 20 with top-p 0.95 for 16 rows: 254 to
  176 microseconds with the direct histograms and 175 to 115 with the reads
  on the RTX 4070 (two pairs of runs, each back to back), 673 to 540 and 539
  to 390 on the RTX 3060.
- **The sums of a pass** meet at one barrier, not ten: 28 to 17 microseconds
  for the pass of the probabilities after the chain, 16 rows, RTX 4070.

Two things that were tried made nothing faster and are not in: the running
totals of the kernel that decides a level by shuffles in place of its 16
barriers (35 microseconds for six of them either way), and a count of the
level's tokens per chunk in front of every histogram (the dense histograms
got 5 to 10 percent slower).

*Where the time of a pick is now.* Device time of the parts, 16 rows, RTX 4070
/ RTX 3060, in microseconds (`sampler_steps`, one run):
| Part | RTX 4070 before | RTX 4070 now | RTX 3060 before | RTX 3060 now |
| --- | --- | --- | --- | --- |
| the best eight logits | 206 to 230 | 35 to 41 | 358 to 369 | 78 to 81 |
| the best twenty of the model's own logits | 289 | 44 | 637 | 101 |
| the best twenty of the set after the chain | 81 | 34 | 194 | 74 |
| the six levels of top-k 20 with top-p 0.95 | 230 | 130 | 669 | 388 |
| the three levels of top-p 0.95 alone | 143 | 124 | 389 | 306 |
| deciding six levels | 35 | 42 | 55 | 72 |
| the draw after the cuts | 17 | 16 | 61 | 55 |
| the pass of the probabilities after the chain | 28 | 17 | 80 | 59 |
| the pass that weighs every token | 90 | 90 | 209 | 212 |
| a mask | 22 | 22 | 75 | 74 |

A level with few tokens takes 19 to 25 microseconds on the RTX 4070 and 63 to
71 on the RTX 3060; the first level of top-p alone, where every token has a
weight, 69 and 140. Deciding a level is the one part that got slower: 6 to 8
microseconds now on the RTX 4070 and 9 to 24 on the RTX 3060, where the second
level of a descent by weight takes 24. The kernel got the count of the chosen
bin to write and reads its bins at once; which of the two costs the time was
not found.

A kernel that passes over the rows is now within 1.3 to 3 times of reading
them, and a pick with cuts is the number of its passes: eight for top-k with
top-p. After the first cut the set is a few dozen tokens, and the later
levels, passes and the draw still read every logit to find them; a list of
the set's tokens, written when the first cut is made, would make them
independent of the vocabulary. That is not built. For one row a pick is its
launches: 3 microseconds each on the RTX 4070 and 5 on the RTX 3060, 22 of
them for top-k with top-p.

*In a decode step.* The 27B with its draft block and the vocabulary list on
both cards (`qwen35_run --engine 8 --generate 300 --sample`, seven prompts,
windows of up to 16 rows, the logits of a row split between the cards; the
kernels before and after in runs back to back, two pairs, load average 4 to 21):
the sampler's part of a step that picks the most probable tokens went from
0.077 to 0.036 ms on the RTX 4070 and from 0.51 to 0.47 ms on the RTX 3060
(the median over the prompts of the median of their steps), and a link of
the draft chain, which ends in a pick over the vocabulary list, from 1.49 to
1.48 ms. A step takes 31 to 41 ms by the number of links it drafts, and the
median of a prompt's steps moves between those values from run to run; for
the five times a prompt's steps drafted the same number of links in both
runs of a pair it was 0.16 to 0.25 ms lower in four (40.63 to 40.38, 31.44
to 31.28, 35.88 to 35.68, 33.77 to 33.54 ms) and 2 ms lower in the fifth, a
jump between two of those values.
That is 0.5 to 0.7 percent of a step, of which the sampler was 1.5 percent;
tokens per second did not change measurably (97.8 and 98.2 on average). With
top-k 20 and top-p 0.95 the sampler's part depends on the rows of the
window: for the three prompts with the same links in both runs it went from
0.61, 0.63 and 0.94 ms to 0.53, 0.53 and 0.82 ms on the RTX 4070 and from
1.03, 1.05 and 1.37 to 0.95, 0.96 and 1.25 ms on the RTX 3060. Most of that
part is not the kernels, which take 0.23 ms for 16 rows of the whole
vocabulary on one card: between the parts of a pick the two cards add up
their partial results, 8192 values a row for every level of a descent. What
that exchange costs was not taken apart here.

A request that sets nothing pays for the best eight logits of its row and
nothing else. Penalties cost a launch. A mask is a pass that writes (22
microseconds for 16 rows on the RTX 4070, 74 on the RTX 3060). The model's own
probabilities add the best twenty and a pass that weighs every token (44 and
90 microseconds for 16 rows on the RTX 4070, 101 and 212 on the RTX 3060).

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
the answer is not checked against the rule for the rest of that token. With the
0.8B (`crates/server/tests/real_structured.rs`) the thinking under a
`response_format` is the thinking of the same request without one, character
for character, and the answer after it obeys the schema. When the thinking
ends is the model's business: taking the most probable token, the 0.8B thinks
about the question of that test in a loop to any limit, with the rule and
without, so the test puts a bias on the token of the closing tag, the least
of a ladder with which the reply ends by itself (1 for the 0.8B).

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

*A text that ends the reasoning.* A model that thinks may write what belongs
after its reasoning without closing the reasoning first: the marker that opens
the call of a tool. A `Structure` may name an interrupt (`Interrupt`): a text,
and the rule for what follows it. While a reply thinks the guide looks for the
closing tag and for that text; at whichever comes first the reasoning ends, and
after the text the interrupt's rule binds, from the token after the one that
completes it, as after a trigger. After the closing tag the structure's own
rule and trigger apply as without an interrupt, and a trigger that begins in
the token of the closing tag is seen. A structure with an interrupt and without
a trigger binds every reply, through one rule or the other, so such a reply may
not end while it thinks: the rows of its reasoning carry a mask of every token
but the ones that end a reply, and a draft that would end it there is cut. With
a trigger the reasoning carries no mask, and a reply may end in it. The guide
compiles the interrupt's rule next to the structure's (one grammar when the two
are the same), and refuses an interrupt whose text is empty. No field of a
request sets an interrupt; the server writes one for the tools of a request
(see [Server](#server), "Tool calls").

*Drafts.* The text with drafts is the text without. Every row of a window gets
the mask of its own path, so a draft the rule forbids is never the pick of the
row before it and is rejected whatever happens. On top of that the conductor
cuts a forbidden draft, and the ones after it, from the lookup's proposal
before the window is chosen (`TokenFilter::admitted`), so the rows are not paid
for: a test with runs of one letter has every draft verified accepted, and 74 of
119 without the cut. Tokens the rule forces (one token allowed) are not offered
as drafts: the lookup is the drafter that joins next to a draft block, and a
forced token would be a third source with a chance of one; it is not built.

*The model's own drafts.* The tokens of the draft block's drafts are not known
when the record is written, so the rows of a window that takes them carry no
mask. For the rows a rule leaves free that is exact: no mask applies there with
drafts or without. Most rules are free for most of a reply (the reasoning before
a `response_format`, a schema or a grammar; with tools that may be called, the
reasoning and the text before the trigger), so a request with a rule takes the
model's drafts whenever the next token is free, as a request without a rule
does, and takes the lookup's alone while the rule is bound. Where the rule
starts to bind inside such a window is found when the picks are back
(`TokenFilter::free_rows`, which a `Structure` answers from the phase of the
reply, `Guide::free_rows`): the token that makes it bind (the last token of the
closing tag, of the trigger) was picked by a free row and is taken; the rows
after it needed a mask, so their picks are not the request's and their cells are
cut, as those of rejected drafts are. That token is then the newest token of the
next window, whose first row has its mask; its row is computed a second time,
which is the cost of a binding token inside a window, once per phase. The
devices are not involved: they get the count of accepted rows from the host as
for every window, and a window of this kind looks like any window of the model's
drafts. A filter that does not say which rows it leaves free is taken to bind
every row.

On the host (`crates/runtime/tests/free_drafts.rs`, the synthetic model with
text that never repeats and a draft block that is right nine times in ten, a
window of 8 rows, a step priced at 3 ms and 0.03 ms a row and 0.3 ms a link of
the chain): a reply of 310 free tokens and 6 bound ones had 255 of the block's
drafts accepted and took 60 steps, and one of 310 free tokens and a call of 42
took 89; the same 310 tokens without a rule had 255 accepted in 55 steps, and
the two replies without drafts took 316 and 352 steps. The same file puts the
binding token at every pick of a window in turn, for both kinds of rule, and has
a filter that is free and bound in turn, after which the drafts come back.

On the devices the synthetic model shows the same on every placement
(`a_rule_that_is_free_in_stretches_takes_the_models_drafts_on_every_placement`
in `crates/runtime/tests/gpu.rs`: a rule that binds the last three of every 7,
10 or 13 tokens, on a second runtime of a model whose draft block has seen the
reply): of a reply of 150 tokens, 22 to 30 windows took the block's drafts and
56 to 92 of its drafts were accepted, and the reply is the one without drafts.
On the 27B over both cards (`the_27b_calls_the_same_with_drafts_and_without`
in `crates/server/tests/real_tools.rs`) a request with tools whose reply thinks
and then calls (100 tokens) had 57 of the block's drafts verified and 40
accepted, next to 32 and 29 of the lookup's, and was the reply without drafts
token for token; the server reported 84 tokens a second for it and 31 for the
same request without drafts. That is one run of each, with a lookup that had
the text of two calls before it, and no benchmark. A reply that is a call from
its first token (the same test, thinking off) takes none of the block's
drafts, as the rule binds all of it: 20 of 27 and 37 of 37 of the lookup's
were accepted there.

Under a bound rule the model's drafts are not taken. A window verified without
masks could be kept as far as each pick is a token the mask of its row allows,
which the host can work out afterwards; that is exact for a row that takes the
most probable token and not for a sampled one, whose cuts (top-k, top-p and the
others) are made within the tokens the mask leaves. It is not built.

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
Through the server, the 0.8B on the RTX 4070 wrote 400 tokens under a JSON
schema (an array of integers) at 412 tokens a second and 400 tokens of free
text at 415 (`the_grammar_costs_what_the_mask_costs_per_step` in
`crates/server/tests/real_structured.rs`, the best of three runs each).

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
  and can be cancelled or promoted. The wide steps of `qwen4exp` submit all
  reads of a layer at once, into staging slots next to the RAM tier (see
  "Wide steps" under "Qwen3.8-Flash-Next through the runtime"). Reading ahead
  of the router is built in the store and has no guess worth reading yet:
  next-layer router prediction, which starts reads early, is planned.
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
kernels take no scale.

*Weights beyond the range.* A weight beyond 65504 is an infinity in the tile, so the outputs
that read it are infinite or NaN (tested for bf16 and f32), and the variant with f16
accumulation, which adds partial sums of 64 products of a weight and a value of at most 1,
gives an infinity from weights of about 1000 (tested with a run of 64 weights of 8000). A
`qwen35` model does not get there. `WeightRange::of` reads the largest magnitude of a float
matrix, and the loader asks it once for every float matrix a product reads (one more read of
its bytes; the result is `Mat::range`). A matrix whose largest weight is at most 512 takes any
path: 64 products of at most 512 stay below 2^15, half the range of f16 (tested at 512 with
activations just below 1). Above that it takes the f16 path, whose sums are f32, where another
float matrix takes f16 accumulation. Above 65504, or with an infinity or a NaN in the file, it
takes no tile kernel: the matrices that read the same input are multiplied by the decode
kernels in launches of eight rows, which compute with the weights as stored (and, for a
quantized matrix among them, with int8 activations, also where `Precision::Exact` was asked
for). The test `float_weights_beyond_f16_are_multiplied_by_the_kernels_that_hold_them` runs a
made-up model with a row of weights of 61440 and two weights of 7e4: its logits are within
7.1e-3 of the reference, as without such weights (6.0e-3); with the rule taken out the logits
of a prompt of nine rows are not numbers and the step after it fails.
The layout probes time such a matrix by its type, on the tile kernels. `qwen4exp` reads the
range of its float matrices at load as well (`Mat::range` in its `device/load.rs`) and its
wide steps ask it before every float product; its quantized matrices go through
`Kernels::product`, which knows no range and needs none.

*Small weights.* What a matrix loses below 6.1e-5 is absolute, at most 2^-25 per weight. The
test `tiny_float_weights_are_off_by_at_most_the_spacing_of_f16_subnormals` derives the bound of
an output from the roundings of each step and holds both paths to it on weights from 1e-7 to
1e-4 (512 values, 2026-11-12, both cards): the f16 path is off by at most 3.4e-6 to 5.0e-6
against a bound of 1.7e-5. The path with f16 accumulation loses more there, and not through
the weights. The tensor cores round its f16 sum after every 8 products (measured: doing
exactly that on the CPU gives the kernel's bits for 99.7 to 99.8 % of the outputs on both
cards, rounding after every 16 for 14 %), and the sum is in the units of the activation row
divided by its factor, the power of two above the row's largest magnitude. A partial sum
below 1.2e-4 is therefore rounded to a multiple of 2^-24 of that factor, so an output is off
by up to `n_in / 8` times that whatever its own size: 3.3e-5 to 3.5e-5 in the test (bound
2.6e-4), ten times the f16 path. That is felt by an output whose row of weights is small next
to the activations: on Flash-Next's `ssm_alpha` (bf16, 48 x 2560, whose rows differ in size by
up to 3.5e5) the worst output with f16 accumulation was 3.3 % of the sum of the magnitudes of
its products away from the product of the rounded operands (8 rows, RTX 4070; 2e-6 on the f16
path), while the error over all outputs was 0.045 % rms. The tests take that absolute term off
before they compare the path with its emulation.

*Error against the exact f32 product* (`tests/prefill.rs`, weights of rms 0.05 with
outliers, activations of rms 1 with outliers, shapes from 1 x 32 to 200 x 1024): the f16
path 0.02 to 0.03 % rms for f16 and bf16 weights (the rounding of the activations; the
weights are exact), 0.02 to 0.04 % for f32 weights; the path with f16 accumulation 0.04 to
0.06 %. Against the same product with the operands rounded to f16 and summed exactly, the
f16 path is within 3e-6 of the magnitude of the sum and the flush path within 1e-3.
On the real float tensors (`prefill_matmul_on_real_tensors`, 2026-11-12, both cards, 8, 9 and
70 activation rows of a normal distribution with outliers, the first 200 rows of the first
matrix of each kind): the 14 kinds of bf16 matrix of Flash-Next 0.014 to 0.024 % rms on the
f16 path and 0.042 to 0.061 % with f16 accumulation, its f32 `ffn_gate_inp` 0.020 to 0.022 %
and 0.041 to 0.053 %, and the f32 `ssm_alpha` and `ssm_beta` of the IQ4_XS file of
Qwen3.8-27B 0.017 to 0.023 % and 0.046 to 0.054 %. The decode kernels, which take eight rows
of them, show no error in the four digits printed. Both cards give the same numbers.

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

*In a model* (`qwen35_run`, Qwen3.8-27B on both cards at 0.625 / 0.375, a prompt of 2048
tokens in steps of 512 rows, two passes, then 96 decode steps at 1000 tokens of context;
2026-11-12; before is the commit before these kernels, after is with them and with the ten
other pieces integrated at the same time; each pair back to back, two rounds, with the
machine's load average in brackets):

| file | | prompt, tokens per second | decode step, ms (median) |
|---|---|---|---|
| Q4_K_S, no float matrix | before | 1366, 1363 [2.7]; 1354, 1354 [13] | 24.76; 24.80 |
| | after | 1347, 1345 [2.7]; 1342, 1352 [12] | 24.75; 24.76 |
| IQ4_XS, 96 f32 matrices | before | 871, 866 [2.6 to 7.6]; 874, 878 [21 to 16] | 28.89; 25.47 |
| | after | 1283, 1295 [7.6 to 15]; 1306, 1310 [16 to 12] | 26.59; 25.50 |

The Q4_K_S file is unchanged within the 1.4 % its runs differ by. The IQ4_XS file reads a
prompt 1.5 times as fast: its `ssm_alpha` and `ssm_beta` (f32, 5120 x 48) had taken the
matrices that read the same input with them through launches of eight rows in every recurrent
block. Its decode step is the same (the fastest step of the first round was 25.43 before and
25.31 after; the medians of that round were taken while the load rose).

*The rows of a step.* A file of float matrices reads a prompt in the steps of 512 rows that
every file takes; wide steps are what the tile kernels are for. Measured on one card (RTX
4070, 2026-11-12, a prompt of 2048 tokens, two passes, each run twice, steps of 128 against
512 rows by `--max-rows`): a copy of Qwen3.5-0.8B with all 187 matrices in bf16 (1434 MiB, in
the file's layout) reads 12342, 13079, 13677 and 13688 tokens per second in steps of 128 rows
and 27625, 27548, 27597 and 27536 in steps of 512 (load 8). The first eight blocks of
Qwen3.8-27B with their 62 matrices in bf16 (made from the Q4_K_S file; 5806 MiB in the file's
layout, 17408 x 5120 the largest) read 5596, 5699, 5700 and 5695 tokens per second in steps
of 128 rows and 7810, 8068, 8082 and 8062 in steps of 512 (load 10).

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
on the tensor cores, for every weight type (see "Experts in every weight type" below; the
kernels were written for the types of Qwen3.8-Flash-Next, gate and up IQ2_S, IQ2_XXS and
IQ1_M and down Q2_0, and what is measured in this section is measured on those). The decode
kernels above take eight rows; these take
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
- **Down** (`rows_down`). The same with the type of down, for Q2_0: a word of 16 codes gives the pairs
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
the four types the kernels were written for and on the real tensors of one layer of each
type (the same figures on both cards); over all 28 types, as gate and up and as down, 0.04
to 0.10 % (the test allows 0.2 %). Rows scaled by 1e-4 to 1e5 within one call stay within 0.2 %. The decode
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

**Experts in every weight type** (`kernels/experts.cu`, `kernels/expertrows*.cu`,
`ares_runtime::expert_store::cpu`). The three matrices of an expert may be stored in any of
the 28 types a model file holds weights in. Gate and up of a layer share one type (one kernel
computes both rows against the same activations; the loader refuses a layer in which they
differ), down has its own, and the types may differ from layer to layer. A row must be whole
blocks of its type and whole groups of 32 values (`ExpertShape::check`), so a type with blocks
of 256 values cannot hold a down matrix whose rows are the 640 hidden values of Flash-Next;
the many-row kernels also take the width in steps of 128 and the inner width in steps of 64
(`ExpertShape::check_rows`, asked when a model opens for wide steps). The router
(`ffn_gate_inp`) must be F32.

- *Decode kernels.* `ares_expert_gate_up_<type>` and `ares_expert_down_<type>` are the
  templates of `expert_ffn.cuh` over the decode traits of the weight kernels, instantiated
  for all 25 quantized types. The float types have a trait of the same interface
  (`EngExpertFloat`): the weights as stored times the int8 activations, so that an expert of
  any type computes `W x^` for the rounded activations `x^` and the host can compute the
  same.
- *Many-row kernels.* IQ2_S, IQ2_XXS, IQ1_M and IQ1_S as gate and up and Q2_0 as down have a
  decoder written for these kernels. Every other type and role goes through an adapter
  (`expertrows_mm.cuh`) that makes a gate and up or a down decoder of the item decoder of the
  dense tile kernels (32 values of a row as f16 pairs), so the engine has one tensor-core
  decoder per type. An adapted type decodes an item whole when a step begins and holds its
  16 words per row stream across the step, where a written decoder makes each group as it is
  used; the down kernel, whose threads take 16 values each, decodes an item for half of it.
  The kernels are in five modules (`expertrows.cu`, `expertrows_{k,i,l,f}.cu`) that compile
  side by side, and whether the sum is atomic or ordered is an argument of the down kernel,
  not a kernel of its own.
- *On the CPU* (`cpu/kernels.rs`). IQ2_S, IQ2_XXS and IQ1_M as gate and up and Q2_0 as down
  have kernels that multiply the file's bytes. Every other type and role is decoded row by
  row with the reference decoder (`ares_quant::dequantize`) and multiplied in f32 with the
  same rounded activations, once for all the rows of the step that use the expert. The
  decoding is what such an expert costs (table below).
- *The cost model* (`CostModel` in `expert_store/hybrid.rs`) times CPU jobs per pair of
  types (gate and up, down): each pair a model has is calibrated by itself when the store
  gets its CPU half, and a job's time is recorded under its layer's pair. Before, every gate
  type but IQ2_S and IQ2_XXS shared one entry and the type of down was not looked at, which
  would have timed a decoded type as IQ1_M. A pair whose jobs take longer than an upload is
  uploaded, by the measured numbers: the split takes whichever of the CPU's job and the last
  upload ends first.

*Tested* (2026-11-12, both cards). Decode kernels, every type as gate and up and as down, on
generated blocks at rows of 512 and 128 values and at the real widths (2560 and 640, or 768
for a down type with blocks of 256): the hidden values are within 0.50 of their own rounding
step of `silu(gate x^) * (up x^)` on the reference decoder's weights (the test allows 0.51),
the output within 1.5e-7 of the exact sum over the device's own hidden values, the layer
within 1.0 to 3.4 % of the exact arithmetic on unrounded activations (8-bit activations and
hidden values, as for the four types before). Many-row kernels, the same types and widths,
every tile and both sums: 0.04 to 0.10 % rms against the exact result. The blocks of real
tensors as experts (`ARES_TEST_GGUF=<any model file>`, the largest tensor of each type the
file has, at the real widths), on the Q4_K_S and the IQ2_XXS file of Qwen3.8-27B, a
Q6_K file with BF16 and F16 tensors and Flash-Next's own, 19 types between them (all but
Q4_0, Q4_1, Q5_0, Q5_1, which the made-up models hold as encoded values, and TQ1_0, TQ2_0,
MXFP4, NVFP4 and Q1_0, of which there is no file here: those five are tested on generated
blocks only): the decode kernels within 0.50 steps and 2e-7, the many-row kernels within
0.06 to 0.13 %. The CPU against the device, layer by layer in
made-up models of 28 pairs of types with half of the experts missing: 2.2e-7 to 4.6e-7 where
no hidden code falls on a rounding boundary and up to 2.3e-3 where one does (the decoded
path sums in another order than the device's integer sums; the test allows 1e-2). Whole
made-up models of 28 pairs of types against the model's CPU reference, layer by layer and row
by row: the experts of a row within 1.4e-2 to 2.8e-2 of the reference's in
steps of up to eight rows and 5.9e-3 to 1.3e-2 in wide steps, in the 764 to 768 of a model's
768 rows of layers that made the reference's choices (on one card and on two); four pairs through the runtime in every
placement (`models_of_other_expert_types_answer_as_the_reference_in_every_placement`).

*Speed of the kernels by type* (`examples/expert_types_bench`, 2026-11-12, load average 6 to
9 from other work; microseconds, RTX 4070 / RTX 3060; an expert of 2560 x 640, or 768 hidden
values for a down type with blocks of 256, 64 experts of random bytes, eight picks a row and
no two rows of a step of eight the same ones; the type is that of gate and up with Q2_0 as
down, or that of down with IQ2_S as gate and up; the decode kernels at 1 and 8 rows, the
many-row kernels with their plan and the atomic sum at 8, 64 and 512 rows):

| type | bits | gate and up: decode kernels, 1 row | 8 rows | many-row kernels, 8 rows | 64 | 512 | down: decode kernels, 1 row | 8 rows | many-row kernels, 8 rows | 64 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q4_K | 4.50 | 56 / 113 | 443 / 811 | 575 / 761 | 530 / 656 | 1021 / 2447 | 44 / 88 | 339 / 650 | 663 / 1140 | 650 / 1194 | 1218 / 2817 |
| q5_K | 5.50 | 47 / 116 | 497 / 846 | 591 / 780 | 635 / 783 | 1115 / 2462 | 46 / 94 | 371 / 701 | 691 / 1220 | 673 / 1296 | 1284 / 2936 |
| q6_K | 6.56 | 58 / 126 | 559 / 929 | 916 / 972 | 936 / 1139 | 1400 / 3433 | 56 / 107 | 416 / 814 | 784 / 1509 | 776 / 1550 | 1477 / 3414 |
| q3_K | 3.44 | 51 / 113 | 410 / 806 | 469 / 787 | 455 / 830 | 1151 / 3386 | 52 / 110 | 376 / 806 | 724 / 1480 | 735 / 1524 | 1440 / 3248 |
| q2_K | 2.62 | 49 / 109 | 382 / 796 | 377 / 558 | 331 / 594 | 1013 / 2377 | 54 / 118 | 398 / 865 | 593 / 1169 | 594 / 1184 | 1279 / 2864 |
| iq4_xs | 4.25 | 45 / 109 | 431 / 785 | 540 / 751 | 520 / 686 | 1013 / 2386 | 44 / 91 | 336 / 666 | 584 / 1233 | 590 / 1249 | 1265 / 2911 |
| iq4_nl | 4.50 | 46 / 106 | 444 / 764 | 565 / 705 | 562 / 659 | 1027 / 2348 | 42 / 91 | 327 / 667 | 584 / 1060 | 618 / 1071 | 1164 / 2647 |
| iq3_s | 3.44 | 50 / 114 | 405 / 838 | 439 / 1013 | 370 / 768 | 1096 / 2623 | 46 / 98 | 327 / 705 | 607 / 1349 | 653 / 1593 | 1408 / 3206 |
| iq3_xxs | 3.06 | 48 / 106 | 376 / 754 | 395 / 725 | 371 / 732 | 1027 / 2477 | 47 / 97 | 324 / 697 | 576 / 1329 | 612 / 1461 | 1385 / 3218 |
| iq2_s | 2.56 | 47 / 103 | 344 / 739 | 401 / 911 | 399 / 882 | 1011 / 2422 | 46 / 96 | 320 / 689 | 571 / 1364 | 582 / 1378 | 1304 / 3151 |
| iq2_xs | 2.31 | 46 / 103 | 340 / 749 | 421 / 954 | 394 / 845 | 1045 / 2505 | 46 / 96 | 323 / 694 | 570 / 1270 | 590 / 1364 | 1346 / 3145 |
| iq2_xxs | 2.06 | 49 / 110 | 367 / 794 | 373 / 710 | 365 / 707 | 995 / 2364 | 46 / 96 | 316 / 687 | 531 / 1216 | 544 / 1251 | 1322 / 3047 |
| iq1_s | 1.56 | 43 / 97 | 336 / 710 | 290 / 579 | 298 / 622 | 1146 / 2744 | 49 / 97 | 354 / 695 | 531 / 1222 | 575 / 1328 | 1263 / 3141 |
| iq1_m | 1.75 | 45 / 108 | 379 / 790 | 267 / 579 | 303 / 652 | 1132 / 2731 | 51 / 105 | 400 / 772 | 527 / 1193 | 557 / 1277 | 1204 / 2869 |
| q8_0 | 8.50 | 89 / 141 | 662 / 1071 | 872 / 1066 | 748 / 971 | 1135 / 2473 | 42 / 99 | 406 / 750 | 692 / 1098 | 702 / 1154 | 1156 / 2670 |
| q4_0 | 4.50 | 42 / 105 | 445 / 770 | 608 / 755 | 566 / 703 | 1007 / 2298 | 41 / 88 | 319 / 645 | 607 / 987 | 598 / 1019 | 1100 / 2536 |
| q4_1 | 5.00 | 38 / 110 | 470 / 807 | 635 / 770 | 588 / 726 | 1020 / 2317 | 40 / 86 | 323 / 637 | 578 / 954 | 560 / 986 | 1029 / 2361 |
| q5_0 | 5.50 | 41 / 115 | 497 / 841 | 654 / 789 | 671 / 767 | 1063 / 2376 | 41 / 91 | 341 / 668 | 614 / 1041 | 612 / 1069 | 1155 / 2618 |
| q5_1 | 6.00 | 43 / 119 | 523 / 880 | 635 / 764 | 606 / 719 | 1049 / 2368 | 44 / 92 | 357 / 698 | 582 / 960 | 583 / 1022 | 1047 / 2412 |
| tq1_0 | 1.69 | 77 / 200 | 738 / 1519 | 548 / 1260 | 589 / 1361 | 1212 / 2891 | 87 / 204 | 838 / 1532 | 674 / 1462 | 668 / 1489 | 1437 / 3609 |
| tq2_0 | 2.06 | 39 / 88 | 319 / 635 | 415 / 591 | 329 / 591 | 963 / 2295 | 50 / 105 | 372 / 769 | 579 / 1168 | 578 / 1217 | 1301 / 3040 |
| mxfp4 | 4.25 | 44 / 104 | 424 / 775 | 658 / 916 | 486 / 700 | 1041 / 2399 | 47 / 107 | 350 / 776 | 472 / 1043 | 505 / 1076 | 1142 / 2664 |
| nvfp4 | 4.50 | 49 / 108 | 424 / 823 | 820 / 796 | 762 / 668 | 1107 / 2431 | 45 / 96 | 349 / 715 | 597 / 1084 | 601 / 1093 | 1074 / 2448 |
| q1_0 | 1.12 | 38 / 85 | 284 / 604 | 262 / 484 | 272 / 551 | 931 / 2218 | 37 / 77 | 256 / 530 | 407 / 967 | 414 / 977 | 1033 / 2471 |
| q2_0 | 2.25 | 42 / 95 | 329 / 672 | 417 / 532 | 392 / 647 | 973 / 2312 | 46 / 103 | 344 / 740 | 403 / 911 | 398 / 882 | 1010 / 2420 |
| f32 | 32.00 | 274 / 377 | 1946 / 2875 | 2963 / 4127 | 3103 / 5090 | 3424 / 9744 | 142 / 222 | 1062 / 1701 | 1286 / 1951 | 1287 / 2029 | 1573 / 3340 |
| f16 | 16.00 | 144 / 214 | 1072 / 1647 | 1043 / 1481 | 1030 / 1465 | 1424 / 2639 | 84 / 134 | 599 / 1045 | 815 / 1329 | 819 / 1399 | 1162 / 2774 |
| bf16 | 16.00 | 145 / 214 | 1072 / 1647 | 1043 / 1484 | 1027 / 1464 | 1475 / 2778 | 84 / 134 | 600 / 1046 | 815 / 1327 | 819 / 1397 | 1117 / 2722 |

The types of 2 to 6.6 bits lie within 1.3 times of the median of their column, with these
exceptions. *The float types and Q8_0* are bound by their bytes: a step of one row reads eight
experts once, 104 MiB for F32 as gate and up, 54 for F16 and BF16 and 30 for Q8_0, at 350 to
430 GB/s on the RTX 4070 of the 473 the card reads in a loop of loads. *TQ1_0* is 1.6 to 2.4
times as slow as its neighbours in the decode kernels (120 to 150 GB/s at one row): its
bytes hold five values each in planes 32 values apart, so a lane loads five words of
activations and five scales from different places, which is what bounds the dense decode
kernel of the type too (`mmvq_more.cuh`). *Q6_K* is 1.3 to 1.8 times as slow as the median
in the many-row kernels: its item is 18 raw words per row stream, which the adapter holds
for four streams and for the step ahead. *NVFP4* as gate and up is 1.35 to 1.45 times as slow as the other 4-bit types in the
many-row kernels at 8 and 64 rows on the RTX 4070 (820 and 750 microseconds against 565 to
610 and 550 to 570, the same in two runs) and not on the RTX 3060 or at 512 rows; the cause
was not found.

*The four types of the real file, before and after* (`examples/expert_rows_bench` and
`expert_layer_bench` on the real experts and the recorded routes at 1, 8, 64, 512 and 4096
rows, the build of before and this one started alternately, twice on each card, the better
of each; load average 2 to 6): the decode kernels within 1 % at every point on both cards
(they are the same instantiations). Gate and up of the many-row kernels within 4 % on the RTX
4070, faster at most points, and within 1 % on the RTX 3060 but for IQ2_XXS, 7 % slower at 8
rows (453 to 485 microseconds) and 4 % faster at 4096. Down of the many-row kernels the same
at 1 and 4096 rows and faster between: 6 to 18 % on the RTX 4070 (1251 to 1058 microseconds
for IQ2_S's layer at 512 rows), 10 to 24 % on the RTX 3060.

*Speed on the CPU by type* (`ares-runtime`'s `examples/expert_cpu_types`, 2026-11-12, nine
workers as the engine places them on this machine, load average 3 to 5 from other work;
the median time of a job of one expert that 1 or 8 rows use, in microseconds, from the
creation of the job to its last task, which is what the cost model times; the same shapes as
above):

| type | gate and up: 1 row | 8 rows | down: 1 row | 8 rows |
| --- | ---: | ---: | ---: | ---: |
| q4_K | 147 | 399 | 105 | 325 |
| q5_K | 214 | 468 | 143 | 368 |
| q6_K | 267 | 520 | 162 | 385 |
| q3_K | 507 | 757 | 267 | 471 |
| q2_K | 473 | 728 | 254 | 467 |
| iq4_xs | 271 | 517 | 165 | 404 |
| iq4_nl | 265 | 517 | 139 | 326 |
| iq3_s | 273 | 521 | 167 | 395 |
| iq3_xxs | 277 | 533 | 171 | 396 |
| iq2_s | 48 | 195 | 222 | 446 |
| iq2_xs | 379 | 647 | 226 | 462 |
| iq2_xxs | 55 | 210 | 229 | 451 |
| iq1_s | 282 | 547 | 178 | 419 |
| iq1_m | 71 | 262 | 188 | 449 |
| q8_0 | 131 | 390 | 91 | 279 |
| q4_0 | 208 | 472 | 122 | 304 |
| q4_1 | 231 | 478 | 131 | 312 |
| q5_0 | 298 | 562 | 157 | 336 |
| q5_1 | 310 | 572 | 163 | 339 |
| tq1_0 | 226 | 485 | 144 | 383 |
| tq2_0 | 141 | 395 | 110 | 347 |
| mxfp4 | 256 | 508 | 134 | 319 |
| nvfp4 | 296 | 551 | 149 | 336 |
| q1_0 | 513 | 770 | 247 | 415 |
| q2_0 | 514 | 774 | 48 | 194 |
| f32 | 338 | 474 | 177 | 359 |
| f16 | 471 | 700 | 213 | 383 |
| bf16 | 163 | 401 | 96 | 276 |

An expert of the types with a written kernel takes 48 to 71 microseconds when one row uses
it and 195 to 262 when eight do; a decoded type takes
131 to 514 as gate and up (2.7 to 11
times the 48 of IQ2_S) and 390 to 774 for eight rows, 91 to 267 and 276 to 471 as down (Q2_0:
48 and 194). An upload of one expert takes 0.1 ms at the median on the RTX 4070 and 0.4 ms
over the link of the RTX 3060 (measured with the real file, see "Qwen3.8-Flash-Next through
the runtime"). So the CPU is sooner for a single miss of the written types on both cards; a
decoded type is sooner only against the slower link, and not when several rows use the
expert. The store decides that per layer and step from the times it measured for the layer's
pair of types and for the device's uploads; nothing in it names a type as slow.

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

A row of a draft window has the bits of the row alone in these kernels: the
convolution takes its window from the raw inputs of the rows before it, the
delta rule carries the matrix from row to row in the registers exactly as a
step of one row stores and reloads it, and the commit replays the accepted
rows with the same operations (`crates/model/tests/qwen35_windows.rs` holds
the outputs, the state and the logits to that, bit for bit).

The state of the delta rule can be held in `f16` (`StateFormat::F16`), which
halves the 150 MiB per sequence of the 27B and takes the delta rule of four
sequences from 55 to 32 µs (measured before the rounding per row that
follows). The matrix is then rounded to `f16` after every
row, in the registers of a window, a prompt and a commit as well, because a
step of one row stores the rounded matrix: a row must not depend on the step
that computed it. Measured on the CPU reference with the 0.8B over
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

*Generated rows* (`Planner::window`). The row of a decode step and the rows of
a draft window must give the same bits alone and together, or a reply would
depend on what was drafted for it. Three things decide the bits of a row, and
for these rows each is fixed by the row alone:

- The pieces are those of a step that holds nothing but the row: the whole
  budget of blocks (three per multiprocessor, by the KV heads of the device,
  at most 64 pieces), cut over the entries the row reads. Rows of a window
  that read the same entries have the same pieces and share their items, so
  the cells are read once for all of them; a window that begins a new entry
  is two tiles of rows in that step. Other sequences change nothing, and
  such rows read no prefix together with other sequences.
- The kernel is the tensor-core kernel in its decode shapes (8, 16 and 48
  vectors), for one row as for eight: a vector has the same arithmetic in
  each of them, whatever shares the item, and the cells past a row's own are
  masked to exact zeros. The direct kernel, which is faster for one row over
  quantized K and V, is not taken for these rows: a window of rows has no
  kernel with its arithmetic that reads the cells once (its work is per
  vector, eight rows would cost it eight times one). It still serves the rows
  that see a list of ranges (`qwen4exp`) and the links of the draft block's
  chain, whose results are proposals.
- The entries are the caller's: with `q7` to `q4` the rows of a window past
  the position at which a tile leaves the newest cells read it converted (see
  "Where the tile formats keep their exact cells").

The rows of a prompt are planned as before (tiles of up to 64 vectors, pieces
by the step), and are not the rows a single step computes.

It costs nothing that was measured. 2026-11-15, the build before against the
build after, back to back, without the profile, K and V in `q8`, the machine
at a load of 15 to 27 from other work. A decode step without drafts, which is
the step that changed kernels: on the RTX 4070 with the `IQ2_XXS` file 20.94
ms before and 20.78 ms after at 1000 tokens of context, 21.95 and 21.90 ms at
8000, 24.03 and 24.02 ms at 32000 (attention over the 16 layers took 0.66 ms
of a step on the direct kernel and 0.52 ms on the tensor cores at 1000 tokens,
1.52 and 1.50 ms at 8000); over both cards with the `Q4_K_S` file 24.83 and
24.54 ms at 1000 tokens. A prompt of 2048 tokens: 1167 and 1171 tokens per
second before, 1170 and 1170 after on the one card; 1350 and 1355 before, 1358
and 1363 after on both. With drafts (`qwen35_run --engine 8 --generate 300`,
the draft vocabulary, the most probable token, two runs of seven requests
with each build) a window's step is the step it was, as far as a loaded
machine shows: over both cards the step median of a request was 31.41 to 31.47
ms before in the quietest stretch (three requests; 34 to 39 ms in the others,
as the load rose) and 31.42 to 31.77 ms after (the seven requests of the
quieter run; 32.7 to 37.2 ms in the other), and the tokens per second follow
the load and the text (59 to 82 before and 64 to 80 after for prose and code,
109 and 127 before and 120 and 138 after for the rewrite). On the one card a
run where the policy chose one link both times had 130 steps of 23.92 ms
before and 130 of 24.00 ms after for the same 300 tokens (91.6 and 91.0 tokens
per second); where it chose two links in one run and one in the other the
steps cannot be compared. The
scratch of a step grew by 1.6 MiB on the one card and by 1.2 and 1.0 MiB on
the two (the states of the pieces of rows planned alone).

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
248 320 rows of 5120 values in the file's types: 313 MiB in the IQ2_XXS test
file of the 27B, 521 MiB in its UD-Q4_K_S file, whose table is Q3_K) does not
lie on the device. A step needs the rows of
its own tokens, a few kilobytes each in the file's form, and the gather kernel
decodes them wherever they lie (`kernels/layer_embed.cuh`: all 28 weight
types). Where they come from (`device/embed.rs`):

- **The host.** The table lies in page-locked host memory of the device that
  embeds (the first one), booked as required memory of the engine's one RAM
  limit (`Model::pinned_ram`). A step copies the rows of its tokens into a
  page-locked staging buffer, one row per row of the step, and uploads them
  on the step's stream in one copy right after the record; no event, no wait.
  A prompt step of 512 rows uploads 1.1 MiB of the UD-Q4_K_S file's table.
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

The table is required memory of the RAM limit, as the step buffers are. The
default limit always holds it. A `--ram` below the sum is refused, and the
refusal names every share and the sum in bytes, which `--ram` takes as it
stands (`plan::too_little_ram`; the 0.8B test file: "--ram 4.0 MiB is less
than the 199.5 MiB the engine needs to run (209235952 bytes): 212.0 KiB for
its step buffers, 199.3 MiB for the token embedding table and the rows of a
step, which lie in page-locked RAM and not on a device"). The table does not
go back to a device when the limit is small. VRAM is what bounds the context,
and `--ram` is set low for other reasons, to switch the checkpoints off for
one: a limit that also moved the table would take context away without being
asked to, and what the devices hold would depend on it. The price is paid by
an operator with little RAM and much VRAM, who must give the table its RAM
(0.2 GiB to 0.5 GiB for the files above) and cannot choose the device
instead. That choice is not built; the gather reads a whole table on the
device already (the draft block without a vocabulary), so it would be a
decision of the placement.

Measured on 2026-11-12 with `qwen35_run` on an RTX 4070 and an RTX 3060, a
context of 32 768 tokens, K and V in `q5`, against the commit before
(92aedff, the table on the first device). The columns are the VRAM the
process took on each card, in MiB, the page-locked RAM the model books and
what the kernel counted more for the process (`RssShmem`, where it counts
what the driver locks; `VmLck` and `VmPin` stay at zero):

| File and placement | VRAM before | VRAM now | RAM booked | Kernel |
| --- | --- | --- | --- | --- |
| IQ2_XXS, one card, no drafts | 8026 | 7716 | 313 | +315 |
| IQ2_XXS, one card, drafts with a vocabulary of 66 343 tokens | 9422 | 9194 | 313 | +313 |
| IQ2_XXS, one card, drafts without a vocabulary | 9424 | 9424 | 0 | 0 |
| UD-Q4_K_S, tensor split 0.625 / 0.375, drafts with the vocabulary | 10948, 6046 | 10568, 6048 | 522 | +523 |
| UD-Q4_K_S, tensor split 0.625 / 0.375, no drafts | 9918, 5680 | 9400, 5682 | 522 | +521 |
| IQ2_XXS, layer split 0.5 / 0.5 | 4168, 4098 | 4050, 3906 | 313 | +312 |

The weights of the first card shrink by the table less the rows it keeps for
the draft chain (84 MiB and 139 MiB for the vocabulary), its step scratch
grows by the staged rows (0.6 MiB and 1.1 MiB), and every card holds 2 MiB
more outside its ledger, the larger kernel module. The layer split moved two
blocks from the second card to the first, which no longer counts the table.

The same day, same cards, the speed of three trees: the commit before
(92aedff), the change as it was first merged (6c27759) and the tree with the
step scratch put right (below), run in turn two to six times each while
other builds ran on the machine (load 3 to 25). Staging a prompt step of 512
rows was timed by itself: 0.12 ms to 0.31 ms for the host to copy the rows
and 0.10 ms to 0.19 ms for the upload, of a step of 380 ms.

| Run | Before | As first merged | Now |
| --- | --- | --- | --- |
| IQ2_XXS, one card: prompt of 2048 tokens, ms | 1749 to 1751 | 1749 to 1757 | 1748 to 1752 |
| IQ2_XXS, one card: decode step, ms | 20.88, 20.94 | 20.92, 20.93 | 20.89, 20.95 |
| IQ2_XXS, two cards: prompt of 2048 tokens, ms | 1693 to 1711 | 1717 to 1737 | 1693 to 1727 |
| UD-Q4_K_S, two cards: prompt of 2048 tokens, ms | 1502 to 1520 | 1520 to 1538 | 1501 to 1529 |
| UD-Q4_K_S, two cards: decode step, ms | 24.76 to 24.83 | 24.70 to 24.79 | 24.73 to 24.77 |
| UD-Q4_K_S, two cards, drafts with the vocabulary: tokens per second, mean of seven prompts of 300 tokens | 97.8 to 100.4 | 97.7 to 98.9 | 98.1 to 98.9 |
| the same: a link of the draft chain, ms | 1.43 to 1.46 | 1.48 to 1.51 | 1.48 to 1.51 |

On one card nothing changed, and a prompt on the tensor split takes what it
took. A link of the draft chain takes 0.05 ms longer than before, for a
reason of its own (the last paragraph of this section). Of the tokens per
second one run of each of the two later columns is left out, 89.7 and 92.9,
taken while the load rose from 5 to 17. On a layer split of the IQ2_XXS file
(shares 0.5 and 0.5), where the change moved two blocks to the first card,
the same prompt takes 2157 ms to 2171 ms before, 2085 ms to 2095 ms as first
merged and 2064 ms to 2075 ms now (load 3 to 6), a decode step 27.38 ms and
27.44 ms, 26.93 ms and 26.97 ms, 26.92 ms and 26.95 ms.

**What the tensor split lost at first.** As first merged, the change made a
prompt on two cards 0.7 % to 1.8 % slower, in steps of 128 rows 0.2 % to
0.4 %, and the profile of a prompt step showed the time in the weight
products of the second card, which holds the bytes it held. It was neither
the table in page-locked memory nor its upload, the exchange between the
cards or the layouts: it was where the second card's step scratch begins.
The stage of the staged rows is the first buffer of the scratch, and a part
that stages nothing still took 64 bytes for it, so on the second card every
buffer behind it began 64 bytes later than before. Among them are the f16
activations of the weight products. Their rows lie a whole number of lines of
128 bytes apart, counted from the start of their buffer, and a step of the
tile kernels reads 64 values of each row: that is one line when the buffer
starts on a line and two when it starts between two. On the first card the
stage is a multiple of 128 bytes in both files, which is why one card showed
nothing.

Found with a probe build that moves one buffer of the second card's scratch
at a time (2026-11-12, UD-Q4_K_S on the same split, prompt of 2048 tokens in
steps of 512 rows, both passes of three runs in turn with the others at load
8 to 15; the two rows marked * are from a batch of three runs whose first
ran while the load rose from 5 to 18 and is left out, 1524 ms and 1531 ms,
1514 ms and 1544 ms):

| The second card's step scratch | Prompt, ms |
| --- | --- |
| as before the change: nothing in front of the first buffer | 1501 to 1508 |
| as first merged: every buffer 64 bytes later | 1514 to 1523 |
| only the f16 activations 64 bytes later | 1516 to 1531 |
| only the 8-bit activations 64 bytes later (the f16 ones 128) * | 1497 to 1513 |
| every buffer but the prepared activations 64 bytes later | 1500 to 1510 |
| every buffer 128 bytes later * | 1501 to 1510 |

So the step scratch now starts every buffer on a line
(`Kernels::ACTIVATION_LINE`, which is also the 128 of the pitch of the rows),
and a part that stages nothing has no stage (`Carve` in `device/part.rs`).
Before, the activations lay on a line only because every size in front of
them happened to be a multiple of 128 bytes; now where a buffer starts does
not depend on the sizes of the others. The VRAM is what the table above
says: the first card holds 518 MiB less without drafts (10102 MiB to
9584 MiB taken at a context of 32 768 tokens in the default K/V format) and
the second what it held (5790 MiB, 5792 MiB).

**The link of the draft chain** takes 1.48 ms to 1.51 ms where it took
1.43 ms to 1.46 ms (UD-Q4_K_S on the split, drafts with the vocabulary),
which is 0.2 ms to 0.3 ms of a step with four drafts (its median 33.27 ms to
33.36 ms before, 33.55 ms to 33.59 ms now) and less in tokens per second
than two runs of one binary differ by. The chain runs on the first
card only, and neither the gather nor the staged row of the newest token
costs it: with the same layout forced for every matrix the link is the same
before and after the change (1.48 ms to 1.50 ms against 1.48 ms to 1.51 ms in
the device layout, 1.45 ms to 1.48 ms against 1.45 ms to 1.46 ms in the
file's). What changed is the choice of the layouts on the first card ("Device
layouts in the model"): before, the draft head (q6_K, 66 343 rows) and the
two wide matrices of the draft block's feed-forward part (q6_K, 17 408 rows)
took the file's layout, now the device layout. The chooser times each layout
in three rounds on a buffer as large as the device's weights and takes the
fastest. For these matrices at one row every round of the file's layout
measures 0.1434 ms to 0.1441 ms and every round of the device layout
0.1423 ms to 0.1427 ms, except that with a buffer of the old size (9470 MiB)
one round of the file's layout comes out at 0.1381 ms and 0.1367 ms (the
result repeats to 0.0005 ms from load to load) and decided for it. With the
buffer the table no longer takes part of (9088 MiB) there is no such round.
Giving the probe of the present tree a buffer of the old size, and nothing
else, brings back the old choice and the old link (1.43 ms to 1.44 ms). The
old choice was the faster one for the chain, but it was not made on what the
probe usually measures: the probe has the two layouts within one percent of
each other for these matrices, and the chain runs 0.05 ms faster per link
with the file's. A round that is 2 % or more faster than the other two of
its product occurs in 7 to 9 of the 232 products timed on the first card, in
other products with another size of the buffer. This is left as it is:
getting the faster link back for a reason would take a probe that reads the
draft block's matrices as a chain reads them (several times a step, a few
hundred megabytes apart) and a choice that does not rest on one round of
three, and either changes the layouts of other matrices on both cards, which
has to be measured as a change of its own.

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

**It drafts with its draft block, from a file of the block.** The model's draft
head is a block of its own (section 4 of the
[specification](reference/qwen4exp-architecture.md)): one layer of the attention
kind with its own mixture of 512 experts, two projections that make its input
from the trunk's residual after the last layer and the embedding of the token
behind it, and its own final mixer in front of the model's output head. The model
file carries none; `--drafters FILE` names the file Strata packs it in (889 MB for
Flash-Next: 175 MiB of dense weights, the experts in Q2_0), and the server refuses
a file that is no block of this model before a device is touched. `--drafters mtp`
is refused (a block inside the model file, as the llama.cpp fork lays it out, is
not read: there is no such file here), and so is `--draft-vocab`: the block picks
from the whole output head. Without a block the model does not speculate: the
lookup in earlier text takes part as soon as a model drafter does and never alone
(see [Speculation](#speculation)).

- *Where it runs.* On the device that holds the output head, in every placement
  (one card; the last card of a split by layers; the card that mixes when both
  work on every layer), as a second part of one layer (`device/draft.rs`,
  `device/part/block.rs`) that reads the model's output head where it is. Its keys,
  values and indexer keys are pages of its own, addressed through the model's
  page tables.
- *What it does in a step.* The block is one token behind the model: its cell `i`
  pairs the trunk's residual of cell `i` with the token of cell `i + 1`. In front
  of every step of the model it (1) follows the rows of the step before that
  count, which the record of this step tells (rows that only write keys and
  values, eight to a launch), and keeps the residual of the last of them; (2)
  runs the newest cell of every sequence, whose token is the first row of the
  step; (3) for the windows that take its drafts goes on from its own residual
  and its own pick, link after link. The keys of the indexer a sequence carries
  are saved behind the first link and put back behind the last, and the keys and
  values of a chain's cells are overwritten when the model has passed them.
- *The host reads every pick.* The embedding of a row and the rows of the n-gram
  table are host work in this model, so a link ends with a read of its pick
  (where the 27B's chain stays on the device), and the device threads that do not
  hold the block wait for the drafts before they enqueue the step that verifies
  them. The verify window is the model's ordinary step of up to eight rows; the
  rows that are rejected leave no trace in the recurrent layers, the n-gram
  history and the indexer's keys (`Part::enqueue_commit`, which existed), and
  their keys and values are overwritten.
- *Its experts go through the expert store*, as layer 0 in front of the model's
  layers (a step routes the block before any layer of the model), read from the
  block's file, and brought into its slots when the store is built. The block
  gets the slots of a layer: 111 in VRAM and 73 in RAM of its 512 experts when
  both cards share the layers (where a layer has about half of its experts on
  each card), 260 and 146 with a range of layers each. A division by the experts
  a layer has on the device was tried through the plan's profile and changed
  nothing; the store moves slots between layers by need as it runs.
- *The policy is the conductor's, unchanged, and on this machine it chooses no
  window yet.* The price of a step by its rows and of a chain by its links is
  measured when the engine starts, and those steps find no expert in a slot: with
  the block loaded they gave 35.5, 49.4, 93.0, 77.0, 97.5, 133.5, 111.3 and
  122.5 ms for one to eight rows and 15.6, 3.3, 4.7, 6.3, 7.7, 9.0 and 10.1 ms
  for one to seven links (2026-11-15, both cards on every layer, load average 20;
  a step of one row takes 22 ms once the tiers hold a text's experts). A price
  is corrected only by a step of its size that runs, so the one-row price comes
  down to what it is and the others stay where they were measured, and a window
  of two rows at 49 ms and a link never pays against a row at 22 ms. Started
  without that measurement (so that the table fills from the steps that run) the
  server did draft: the block's links were accepted 115 times of 176, 51 of 73
  and 21 of 29 for the first three on German and English prose and Rust (the
  fourth: 7 of 18), a link took 7.5 to 8.1 ms while most of the block's experts
  were in no slot, and the rows that verify waited for experts of their own (135
  outside a slot and 167 ms waited a step with windows of eight rows, where a
  step of one row misses 6 to 37). It was not faster for it, the measured steps
  no longer filled the tiers with what a first text finds there, and a first
  chain that ran while experts were missing left a price that kept every later
  one from being chosen. So the server measures at the start as for any model.
  What this model needs of the conductor, and does not have: prices of its
  windows taken when the tiers are filled, or tried again after they were.
- *The window* is four rows unless `--window` says otherwise (2 to 8): a row of
  this model routes to experts of its own, so a rejected draft costs what a token
  costs.

*Drafts change no token.* On made-up models, in every placement, a sequence
writes with the block what it writes without, token for token: greedy, with
drafts of the lookup that are accepted, with `"draft": false`, and sampled with
a seed (`tests/qwen4exp_runtime.rs`); the block's chains are the CPU reference's
from the same cells (first links 33 of 33, 18 of 18 and 32 of 34 in one run of
the three placements). Two sequences that share their steps part from what each
writes alone at a near tie, with the block and without it (the reference has the
two tokens 0.003 apart at a spread of 1.5): what a row's sums are depends on the
shape of its step, drafts or not. On the real file the comparison is made with
every miss uploaded and none computed by the CPU, whose arithmetic is not the
device's: 64 greedy tokens and 48 sampled with a seed were the same with and
without the block on both cards (2026-11-15; 110 drafts verified, 73 accepted,
the links 29 of 32, 18 of 29 and 12 of 18). With the CPU serving misses, as the
server runs, two runs of the same request differ at near ties without any draft,
so nothing can be said there beyond the first sentence of an answer, which the
server's test pins.

*Measured* (2026-11-15, RTX 4070 and RTX 3060, `--ram 10G`, `--chat-var
enable_thinking=false`, greedy, 128 tokens after prompts of about 1000 tokens of
German prose, English prose and Rust; the machine was shared with three other
jobs the whole day, the load average is given, and a step that waits for the
drive or the CPU takes what the load leaves it, so the columns are not one
measurement of two engines):

| | Generation, first request: German / English / Rust | The same requests again |
| --- | --- | --- |
| Both cards on every layer, no drafter (load 9 to 11) | 30.9 / 19.4 / 19.4 tokens/s | 33.3 / 44.2 / 33.2 |
| The same (load 12 to 22, a build running) | 19.3 / 24.3 / 16.9 | 27.9 / 24.6 / 13.3 |
| Both cards on every layer, the block, prices learned from the steps that run, windows of 8 rows (load 11 to 18) | 10.7 / 24.4 / 17.2 | 63.7 / 28.9 / 50.7 |
| A range of layers each, the block, prices measured at the start, windows of 4 rows (load 6 to 8) | 23.7 / 30.8 / 27.8 | 39.9 / 37.6 / 47.6 |
| A range of layers each, no drafter (load 28 to 33) | 15.4 / 22.6 / 8.8 | 19.9 / 25.7 / 10.4 |
| One card (RTX 4070), the block, prices measured at the start, windows of 4 rows (load 16 to 22) | 11.4 / 13.5 / 5.9 | 4.1 / 15.8 / 9.6 |

In the fourth row the conductor did choose windows (its measured prices came out
lower on a quieter machine): 255 links in 87 steps, the links accepted 59 times of
87, 36 of 59 and 21 of 36, 5.4 ms a link on the RTX 3060 (15.7 ms for the chain of
a step that drafts), 535 to 1080 picks a step where a step of one row has 480, and
21 to 166 of them outside a slot. On one card, which holds a third of the experts
in VRAM and reads the rest from RAM and the file, windows were chosen as well: 326
links in 182 steps, accepted 117 times of 181, 50 of 82 and 3 of 5, 6.1 to 6.9 ms
a link. The second time a text is asked the lookup has
it, which is where the 63.7 and the 47.6 come from (37 and 15 steps for 128
tokens). The block takes 175 MiB of dense weights, 21 to 49 MiB of keys and values
and 35 to 48 MiB of buffers on its card. No pair of runs with and
without the block at the same load exists, so whether the block makes this model
faster is not shown; the first requests of the third row say that windows of eight
rows on tiers that do not hold the text make it slower.

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
  `step_rows`: 512 rows on this machine), each with the newest token of every
  generating sequence riding along, and the share owes the generating
  sequences rounds for the time a piece took. A piece of 512 rows of a long
  prompt takes 2 to 2.5 s (3.5 to 4.5 s before the misses went through transit,
  see "Transit, measured") against 22 ms for a step of one row, because every
  row routes to experts of its own; 0.9 s of it is the layers before any
  expert. So the token that rides along with a piece comes after that time, a
  request that arrives while a lone prompt is read waits that long for its
  first step, and the rounds owed between pieces go at a token each; whether
  generation gets the half of the time that the share promises was not
  measured for this model.
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
described above, and a round of generation stays there: one row for each sequence, and with
a draft block the drafts they verify, eight rows for all of them together. The piece of a prompt that the conductor sends between rounds, up to
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
  the n-gram block's projections) go through the float tile kernels (`Part::wide_float`):
  the f32 section of the activations is converted to f16 rows in a scratch of its own,
  whatever form the quantized matrices of the step read, and the sums are f32 whatever the
  step's precision is (see "Float matrices of a wide step" below for why). The range of a
  float matrix's weights is read when the model is loaded (`Mat::range`); a matrix with a
  weight beyond f16, or of a width the tile kernels do not take, is multiplied eight rows
  at a time by the decode kernel on four streams, as all of them were before (a launch of
  eight rows has a few dozen blocks and takes about 90 microseconds whatever it
  multiplies). The fused read of a stream mixer is made for a few rows, so a wide step
  reads its mixers with the separate kernels.
- *The router* is such a float product (`ffn_gate_inp` is F32) and `pick_rows`: ids and
  weights of every row's picks in the form `ExpertRowKernels::plan_rows` reads, without the
  groups of the decode kernels, which hold eight rows.
- *The routed experts* run in **rounds**. The many-row kernels do not wait on the device
  for an expert that has no slot, so in a wide step the host takes part in every layer
  (`Misses`). When the layer is enqueued it gives the first round a table of the address of
  every expert that has a slot in VRAM (`ExpertStore::vram_slot`), and the device computes
  them while the route record travels to the host. The record gives, per expert, how many
  pairs picked it. The experts without a slot are the layer's **misses**, most picked first.
  A step that misses nothing has one round and the host only waits for the record, while the
  first round runs. The plan of the layer is made once, and the host never waits inside a
  call: the caller serves the store between the steps of `Misses::advance`. The experts are
  not computed on the CPU in wide steps; the heat of the store gets the layer's counts of
  rows per expert (`record_counts`).
- *The misses pass through transit* (`expert_store/transit.rs`). A slice of a prompt uses far
  more experts of a layer than the layer has slots, each for all its rows at once and then
  no more, so a missed expert takes no slot: the store uploads it into one of a small pool
  of transit slots on its device, which belongs to no layer, from its RAM slot (which keeps
  it) or, when no tier holds it, from the file through staging slots in host memory
  (`ExpertStore::transit_want`). Nothing is evicted and no table entry is written, so the
  uploads start with the record, next to the first round, and every read of the layer is
  submitted at once. Whenever the device has finished a round and half of the transit slots
  hold experts that have arrived (or all that are left, or nothing of the layer is on its
  way any more), those are the next round's table, whose outputs `red.add` into the same
  rows; when it has ended, its transit slots are free for the next ones.
- *What a prompt used takes slots by its heat, on the device.* When the round that read an
  expert from transit has ended (`transit_release`), the heat has the counts of the step. An
  expert that finds a free slot of the layer in VRAM, or that is hotter than the member that
  would leave (by the margin of a swap, `SwapRule`, when that member has to be copied down to
  RAM to be kept), takes that slot: its bytes are copied from
  the transit slot to the slot on the device and the table entry follows them
  (`SlotDevice::adopt`, in the order of the copies down, so that no upload waits for it), at
  most half of the transit slots at a time. That is how the warm-up fills empty tiers and how
  VRAM follows a prompt, without an upload and without evicting what is hotter. An expert
  that came from the file and that is hotter than what the RAM tier would give up for it is
  copied there from staging; the others are forgotten. What the adopted experts evict goes down to RAM like every
  evicted expert.
- *The sizes* come from the budgets at load (`plan_transit`): a device's transit holds half
  of the experts of a layer that the device computes and that have no slot in VRAM (56 to 58
  slots per device on this machine with both cards on every layer, 1.44 MiB each, taken
  from the bytes for slots: a slot less for every layer), staging twice the experts of a layer that no tier holds (165 to 250
  slots here, by the RAM the tier got), at most an eighth of each tier. A tier that has only the slots a decode step needs has no
  transit: its misses come into the layer's slots in batches, each giving way to the next
  (`ExpertStore::routed`), as all misses did before transit.
- *No read is started on a guess.* The store reads guesses ahead into half of staging as
  prefetches and promotes them when a record names them (`transit_guess`), and `Feed` can
  guess, for the layers ahead (`ares_weights::Lookahead` gives the distance), what the wide
  step before missed there and no tier holds. That guess is off (`SessionOptions::
  read_ahead`), because it is mostly wrong and costs more than it gives: see "Transit,
  measured". A guess worth reading is the next layer's router applied to this layer's
  input, which is not built.
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
generating step (`--parallel`), `--drafters mtp` (the model file carries no draft block;
`--drafters FILE` names the block's file), and `--window` without a drafter or beyond
the eight rows of a step that verifies.

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

**Float matrices of a wide step, measured** (2026-11-12, RTX 4070 and RTX 3060; the engine
of before (`b8b0271`, the float matrices in launches of eight rows) and the one with the
float tile kernels started one after the other, two or three times; the tool, `qwen4exp_run --prefill 1000
--rows 512 --repeat 4 --breakdown` on the first 1000 tokens of this document, of which the
first reading fills the tiers and is left out; load average 3 to 22 from other work):

| | before | after |
| --- | --- | --- |
| Prompt, layers split: tokens per second (three pairs of runs) | 258, 276, 296; 256, 274, 273; 266, 274, 276 | 335, 428, 434; 415, 459, 462; 402, 450, 457 |
| Prompt, both cards on every layer | 288, 314, 314; 292, 312, 315 | 384, 422, 421; 374, 413, 416 |
| Device time of the mixers per step, layers split (25 layers on the RTX 4070, 23 on the RTX 3060) | 207.5 / 344 ms | 52.9 / 78.7 ms |
| The other dense products of the same step | 66.2 / 118 ms | 45.0 / 84.8 ms |
| The host's wait for a layer's picks, layers split | 17.1 to 17.8 ms | 6.5 to 7.0 ms |
| The same, both cards on every layer | 17.7 to 18.4 ms | 9.5 to 9.8 ms |

A layer's two mixers took 8.3 ms on the RTX 4070 and 15 ms on the RTX 3060 and take 2.1 and
3.4 ms; as kernels alone the four products of a layer's mixers are 8.6 / 13.1 ms in launches
of eight rows and 0.8 / 1.3 ms on the tile kernels (the numbers under "Float matrices"
above), so what a mixer costs now is mostly its other kernels. The router alone is worth 5
to 8 % of the prompt: with the router left on its launches of eight rows the same runs read
351 to 425 tokens per second on the layers split and the host waited 7.8 to 8.4 ms for a
layer's picks.

*What it costs in precision.* The float products of a wide step were exact f32 and now read
f16 operands (0.02 to 0.04 % rms a product, see "Float matrices"). The sums are f32 whatever
the step's precision asks for the quantized matrices: with f16 sums, which is what
`Precision::Fast` picks for a float type on these cards, the layers of the made-up model
were up to four times as far from the CPU reference (`every_layer_of_wide_steps_matches_the_reference`:
attention core 6.2e-2 against 1.8e-2 with quantized dense matrices, experts 5.3e-2 against
1.65e-2 with float ones; f16 sums lose a row of small weights among larger ones, and the
recurrent gates have such rows). With f32 sums and float dense matrices the recurrent core
is 7.4e-4 from the reference where it was 3.6e-7, and everything else is where it was
(experts 1.65e-2 after 1.62e-2, attention core 7.7e-3 after 7.9e-3; with quantized dense
matrices 1.59e-2, 0.216 and 1.84e-2 after 1.57e-2, 0.216 and 1.65e-2). What moves are the
discrete choices at near ties: the router of the made-up model with float dense matrices
picked the reference's experts in 2804 of 2816 rows of layers where it had in 2814 (2810
with the router left exact; with quantized dense matrices, whose 8-bit activations move the
router's input a hundred times as far, 2746 after 2734), and in 2 of 28 made-up models one
row of a wide step selected another block in an attention layer, which none did with exact
float products. The tests of the expert types compare the rows that made the reference's
choices and count the others (`ExpertRows` in `tests/qwen4exp_device.rs`). 

*The real file* against its recorded CPU reference (`qwen4exp_run --check` on the German and
the English trace, both cards on every layer, one run each before and after). With the given
tokens read in one wide step, the logits of the last one are 0.023 and 0.032 from the
reference's (relative RMS; 0.126 and 0.061 before) and, fed the reference's tokens, the next
token is the reference's in 160 of 160 German and 79 of 80 English rows (158 and 78 before).
In steps of eight rows, which the float tile kernels do not touch, it is 158 and 80 after 160
and 80. The rows that differ are near ties (the two logits 0.01 to 0.15 apart), and which
rows they are changes from run to run with the misses the CPU serves, as before.

**Transit, measured** (2026-11-12, RTX 4070 and RTX 3060, the engine of before (`b8b0271`:
every miss of a wide step takes a slot of its layer) and the one with transit started one
after the other, `--chat-var enable_thinking=false`, otherwise the defaults (steps of 512
rows, no drafts), greedy, 128 tokens; per round a short German question, German prose of
1000 tokens, Rust of 1034, the short question, the long prompt of 5033 tokens (prose, code
and German), the short question; the numbers of the second of two rounds; other agents'
builds ran on the machine, the load average is given):

| | Both on every layer, before (load 12 to 16) | The same, transit (load 6 to 7) | Layers split, before (load 6 to 7) | Layers split, transit (load 3.5 to 4) |
| --- | --- | --- | --- | --- |
| Prompt, German 1000: tokens/s, time to the first token | 185, 5.4 s | 255, 3.9 s | 165, 6.1 s | 231, 4.3 s |
| Prompt, Rust 1034 | 104, 10.0 s | 174, 6.0 s | 41 (77 in the first round), 25.2 s | 152, 6.8 s |
| Prompt, 5033 tokens | 141, 35.9 s | 224, 22.5 s | 112, 45.1 s | 198, 25.4 s |
| Generation after the prompt, German / Rust / long: tokens/s | 24 / 21 / 23 | 32 / 36 / 30 | 26 / 11 / 23 | 36 / 32 / 32 |
| The short question before and after the long prompt | 24, 32 | 36, 32 | 8, 28 | 34, 33 |
| Read from the file for the long prompt | 23.1 GiB | 7.7 GiB | 25.1 GiB | 9.0 GiB |

The engine of before ran at a higher load in the first pair and was disturbed in the Rust
request of the second (and the short question behind it), so its generation numbers there are
too low: at a load of 5 to 6 it generated at 35, 28 and 23 tokens per second after the three
prompts and answered the short question at 32 and 30 (prompts then: 188, 98 and 130 tokens per
second). The transit columns are of two builds of this work (`6213d28`, `e6f12f6`); the last
rules (`373b0b3`) were measured against the build before them, the tier fixed with `--ram
10G`, at a load of 3 to 4: the long prompt at 211 and 208 tokens per second the first time a
server sees it and 248 and 245 the second, German prose as the first request at 123 and 117,
generation at 34 to 40. What a server reads first after its warm-up is bound by the file and
gains least: German prose 120 before and 107 with transit in one pair, 62 and 117 in the
other; the long prompt 123 and 213, 112 and 180. Run again with the final rules and the tier
fixed, on a machine at a load of 14 to 19, German and Rust were 173 and 85 before and 222 and
152 with transit, and the long prompt 126 and 118: the load and the drive decide a run, so
the engines were always measured next to each other.

What moved (the tool, both cards on every layer, the tiers filled from the recorded routing
of other texts, then 4700 tokens of the long text read three times in steps of 256 rows, the
tier fixed at 9 GiB, load 17 to 31): before, every reading uploaded 53 000 to 57 000 experts
into slots, evicted as many, copied 29 000 to 35 000 of them down to RAM and read 19 000 to
24 000 from the file (26 to 33 GiB), the third reading of the same text like the first; the
layers waited 32 to 80 ms each (per part) for their misses, 74 to 89 % of a layer's time,
and it read at 37, 64 and 90 tokens per second. With transit the three readings read 25, 14
and 14 GiB from the file, 20 000, 600 and 80 experts took a VRAM slot, the layers waited 67,
15 and 15 ms (86, 60 and 60 %), and it read at 45, 168 and 169 tokens per second. A slice of
256 rows misses 29 to 35 of the 124 experts it picks per layer and part either way: the slots
hold what they hold. A miss whose bytes are in RAM costs the link and no more (0.25 to 0.3
ms an expert over both cards, 0.07 to 0.12 ms on the RTX 4070 and 0.5 to 0.7 on the RTX
3060, against 0.9 to 2 ms before), so the churn was the cost and not the link; the RTX
3060's link (8 GT/s, four lanes) is what the misses of its layers wait for now. The first
reading of a short prompt on tiers of other texts is the case transit does not help and
sometimes hurts (1000 tokens: 86 before and 35 with transit at that load, 55 and 49, 107 and
84 on a quieter machine): nearly every miss takes a slot then and most of what they evict is
copied down; why that costs more than the old path's own copies is not understood.
Generation right after a reading, in the tool: 41.5 and 15.3 tokens per second before (after
1000 and after 4700 tokens, the second at a load of 29), 33.9 and 24.1 with transit.

- *The transit slots.* 16, 32, 64 and 128 slots per device read a second reading of 2500
  tokens at 200, 207, 215 and 223 tokens per second (steps of 512 rows, load 11 to 15; the
  first reading is bound by the file and shows the load: 86 to 98) with 4.3, 2.5, 1.45 and
  1.02 rounds after the first per layer. The plan's rule (half of the experts of a layer
  that have no VRAM slot: 56 to 58 here, 82 MiB per device) is where the rate stops growing
  for the memory.
- *Reading ahead on a guess.* With the guess (what the wide step before missed in the layer
  and no tier holds) a reading of 2500 tokens read 14.0 to 15.0 GiB from the file against
  12.3 to 12.4 without and a second reading 4.0 to 5.7 against 2.9, and ran at 69 and 96
  tokens per second against 102 and 107, and 221 and 214 against 259 and 277 (on, off, on,
  off; load 25, 12, 9, 5). Of the guessed reads 48 % were used in a first reading and 8 to
  14 % in a second: what a step read and the heat keeps is in RAM afterwards, so what is
  left to guess is mostly what one row used.
- *The rows of a step.* 4700 tokens in steps of 256, 512 and 1024 rows (the tool, a second
  reading, load 4 to 6): 100, 112 and 126 tokens per second, while the layers of a step take
  0.41, 0.86 and 1.77 s before any expert (2.9 s at 2048 rows). More rows bring each expert
  for more rows at once, 12 % of rate for twice the rows, and keep every other request
  waiting twice as long.

What bounds the prompt now:

- *Bringing the missed experts.* A slice of a long prompt picks 40 to 80 % of the experts of
  every layer, the tiers hold 55 % in VRAM and 30 % in RAM, so every slice misses a quarter
  to a third of what it picks, and the next slice misses most of them again: they are
  uploaded again at the link's speed (3 GiB/s on the RTX 3060), which is 60 % of a layer's
  time when their bytes are in RAM. What no tier holds is read from the file when the
  record names it, all reads of a layer at once, and the layer waits for the last: the
  file gave 0.7 to 1.0 GiB/s on this machine while others used the drive, and a first
  reading takes 4 to 13 GiB from it.
- *The dense part of a layer.* The host waited 18 to 40 ms per layer and slice of 512 rows for the
  router's picks; that is the device's time through the mixers, the projections and the
  recurrent layer. The mixers' matrices are BF16; eight rows at a time over four streams
  they took 4 ms a read on the RTX 4070 and 7 on the RTX 3060 at 512 rows (the fused read
  took 7.5 and 24). On the float tile kernels a read takes 1.1 and 1.7 ms and the host waits
  7 to 10 ms for the picks (see "Float matrices of a wide step, measured" above).
- *The host in every layer.* The first round runs while the picks travel to the host, but the
  next layer cannot be enqueued before the host has the misses of this one: a layer without
  misses costs one round trip of the record (about 0.1 ms); the waves would take that out too.

**Where a faster prompt path goes next.**

- *A guess worth reading ahead.* The reads of a layer start when its record arrives and the
  layer waits for the slowest of them; what a first reading takes from the file is what
  bounds it. The store can read ahead of the router and promote what the record names; what
  it lacks is a guess that is right. The next layer's router applied to this layer's input
  (FATE reports close to all experts of a prompt's rows found that way) needs a second
  product and a second record per layer on the part that routes, and the record's way to the
  host; with it the reads of layer L+1 run while layer L computes.
- *The pass.* A step of more rows brings each expert of a layer once for more rows, and a
  pass (`ares_sched::Pass`: all rows up to a layer's router in chunks that fit the scratch,
  then the layer's experts for all rows) takes the scratch out of that. A pass inside one
  step was designed and written but never compiled (the residual, the pending output and
  the experts' input of every row kept on the device, 56 KiB a row; rounds over the chunks;
  only for both cards on every layer), and is not in the tree: a step is what a request
  that arrives waits for, and a step of 512 rows already takes more than a second on this
  machine (see "Transit, measured"), so a pass of more rows inside one step keeps every
  other request waiting for seconds. What an interruptible pass needs instead: the
  conductor runs the pieces of `Pass` between ordinary steps (`pass_rows_between` sizes
  them), so a pass is a state of the model that outlives a step: the kept rows on every
  part and what `Misses` knows of the layer (plan made or not, which experts were computed)
  must survive ordinary steps that run between two pieces, which need the scratch and the
  transit slots meanwhile; the executor and `DeviceModel` need a piece as a unit of work
  that ends without its rows having reached the end of the model; a sequence in a pass has
  rows at a point inside the model, which the page tables, the recurrent state (its windows
  advance chunk by chunk and layer by layer, not row by row through all layers) and
  cancellation must know; and a split by layers must hand the kept rows of all chunks over
  between the devices at the boundary layer, in pieces the link buffers hold.

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
  the 27B, in both placements. Its draft block (`--drafters FILE`) runs in
  every placement and changes no token, but no window is chosen for it on
  this machine yet (see "It drafts with its draft block"), so it generates
  without speculation: 43 to 48 tokens per second at the median of a series of runs at
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
  step); a row of a draft window has the bits of the row alone only while
  the step has no more rows than the few-row kernels of the device layout take
  (16: three sequences that draft at once with windows of 8, or a window above
  16 rows, go past it, and their rows then round as the rows of a prompt do);
  a row still depends on the steps that computed its prompt (see "Rounding"
  under "Sequence state"), so a reply from the prefix cache is not always the
  reply computed from nothing; a prompt
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
   RAM, the partly used last page and the state at a request's end on SSD
   (pages and checkpoints at page boundaries are, see "On SSD"), a layout of
   stored state that does not depend on the split, locking the RAM of the
   next chunk of places
   ahead of its first use, and `qwen4exp` with a cache (what that needs is
   under "Qwen3.8-Flash-Next through the runtime").
5. Shares of a tensor split from a measurement at load; device layouts for
   the remaining types (the i-quants of 2 and 3 bits stay in the file's layout,
   and with them the integer kernels for few rows); a choice of layouts that
   does not change with the draft block or from one load to the next; the
   few-row kernels of the device layout in launches, so that a step of more
   than 16 generated rows keeps every row's bits; rows of a prompt that
   compute as single rows do, so that the prefix cache cannot change a text.
6. Vulkan: experts, the `qwen4exp` kernels, and the models (the sampler's
   kernels wait for a model that calls them); a sampler on devices without
   `shaderFloat64`, with the double-precision functions written in 64-bit
   integers; attention for
   windows of many rows on the cooperative matrices, and a faster decode of
   the `q8` and `q4` rows; a faster upload.
7. Multimodal input, `Engine::decide` and
   `/v1/systemone` on the real engine. For tool calls: the tests on the real
   models (the model's own drafts in the free part of a reply with tools
   among them, which only the host has run), and a format for templates that
   write calls otherwise.
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
| Prompt cache in RAM (pages of finished conversations) | Elastic, low | Goes first; its pages are on SSD as well (with `--cache`; the checkpoints are its RAM today, and one place is required for the way to the drive) |

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
  embedding table (313 MiB and 521 MiB in the two files of the 27B that were
  measured) and the staging buffer of the rows a step uploads from it, unless
  the device keeps the whole table (see "The token embedding of `qwen35`",
  also for what a limit below it does: the server refuses it and names the
  shares);
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
  prompt cache's only RAM today, and with `--cache` the ones at page
  boundaries are on SSD as well.)

The same facts are available as counters to whoever monitors the server.

## Loading and unloading

Switching models must take seconds, bound by the hardware and not by the
engine. What that needs:

- **Nothing to start.** The server process, the device contexts and the
  kernels stay loaded while models come and go. The CUDA kernels are
  precompiled machine code; the Vulkan pipelines want a pipeline cache on
  disk. (Today the server process stays, and a device's driver context
  stays while a model is on the device: it is opened with the first model
  there and closed with the last, so that a server with nothing loaded holds
  no device memory and a model whose device failed is loaded again on a
  fresh context. A model that takes the place of the only other one on its
  devices therefore opens them again; what that costs is part of its load
  time and is not measured apart.)
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
- **Unloading frees and returns.** Nothing is written at unload beyond what
  was already queued for the drive: state on SSD is written as it is
  produced (see "On SSD" under "Sequence state"). With `--cache` a model that
  is loaded again finds its conversations in its store; without it the
  prefix cache of a model goes with it. The
  unload ends the engine's requests, joins its threads, drops the engine and
  compares what every device reports as free before and after with what the
  model took at its load; the log has both and the time.

Models are loaded and unloaded while the server runs by its router mode (see
"Several models" under Server). Load and unload times of the real files
through it are not measured yet.

## Server

`ares --model <file.gguf>` loads a model on the devices and serves it;
`ares --models-preset <file.ini>` serves several, each loaded when a request
names it (see "Several models" below).
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
"Host memory"). `--cache <dir>` gives the prompt cache its level on SSD and
`--cache-size` the most the directory may take (32G when it is not given);
without a directory nothing is written to any drive (see "On SSD" under
"Sequence state"). In a preset file the keys are `cache` and `cache-size`,
for all models in `[*]` or for one.

It answers `POST /v1/chat/completions` and `/v1/completions`, streamed and
not, Anthropic's `POST /v1/messages` and `/v1/messages/count_tokens` (see
"The Messages API" below), OpenAI's `POST /v1/responses` without state, with
`/v1/responses/input_tokens` and `/v1/chat/completions/input_tokens`, which
count a request and generate nothing (see "The Responses API" below),
`GET /v1/models`, `/health` and `/metrics`, with
timings in every answer of the OpenAI routes
(prompt tokens and time, generated tokens and time, drafts proposed and
accepted, and `token_hash`, a hash of the reply's token ids in the final message
and in the last event of a stream). A request can switch drafting off for itself
with `"draft": false`; the two replies of one request, with and without drafts,
are then compared by their hashes over HTTP (they are equal, with the fast kernels and with `--precision exact`: see
"The text with drafts is the text without" under [Speculation](#speculation)).
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
| `auto` (the default) | free until the model writes `<tool_call>` (the rule's trigger), in its answer or inside its thinking; from there calls, and nothing after them |
| `required` | calls from the first token of the answer, after the empty line the template writes there, or from `<tool_call>` inside the thinking: at least one |
| a named function | the same, and every call is of that function; the prompt still lists all tools |
| `none` | text: the tools are in the prompt, no rule binds the reply, and a call the model writes stays in `content`, as with llama-server |

`parallel_tool_calls: false` allows one call. The reasoning of a model that
thinks is free in every case, and the rule (or the wait for the trigger) starts
after its closing tag, or at a call the model opens before it.

*A call inside the thinking.* These models sometimes write `<tool_call>` while
they still think, without `</think>` before it. When the tools of a request may
be called (`auto`, `required`, a named function), the marker ends the reasoning
where it stands, as llama-server has it for these templates
(`thinking_end_tags` and the parser of `common/parsers/qwen3-coder.cpp`). The
marker is the trigger of the detected `ToolFormat` and the tags are those of
the detected thinking; nothing is known by the name of a model.

- The engine: the structure of the request names the marker as an interrupt
  (see [Sampling](#sampling)), with the rule of a call whose marker is written.
  From the token after the marker the reply is held as a call after `</think>`
  is: a name among the tools, arguments by the schema, the closing tags, then
  white space, another call if several are allowed, or the end. With `required`
  or a named function the end tokens are not allowed while the reply thinks, so
  it cannot end as reasoning alone: it ends with a call or at the limit of
  tokens. llama-server reaches the same by a grammar over the whole reply.
- The reply: the reasoning's splitter ends the reasoning at the marker
  (`Splitter::ends_at`, set by `Sorter::new` when the request has a splitter of
  calls) and passes the marker on as the first text of the content, where
  `CallSplitter` reads the call. `reasoning_content` is the text before the
  marker without the white space between the two (llama-server leaves it in),
  `content` is empty, and no part of the marker reaches either, however a
  stream cuts the text. White space at the end of the reasoning is held back
  for one piece of a stream, until the next shows whether the marker follows. A
  raw reply (`reasoning_format: none`) keeps its thinking in `content` and has
  the call too: `CallSplitter` reads the first marker of a reply wherever it
  stands.
- `</think>` after such a call cannot be written: after a call the rule allows
  white space, another call and the end, and no text. A reply that never writes
  it lacks nothing. When the turn comes back, the template writes `<think>`, the
  reasoning, `</think>`, an empty line and the call, the same prompt as for a
  reply that closed its thinking; checked with the templates and vocabularies of
  Qwen3.5-0.8B, Qwen3.8-27B and Qwen3.8-Flash-Next (`tools/templates.rs` in
  `ares-server`). The cached sequence is repeated up to the end of the
  reasoning and the call is computed again (41 of the 56 generated tokens in
  the test), where a reply that closed its thinking before the call is repeated
  whole.
- With `tool_choice: "none"`, or without tools, nothing binds the reply and
  nothing reads calls: the marker inside the thinking is reasoning like any
  other text. llama-server ends the reasoning at the marker there too and
  returns the rest of the thinking as `content`.

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
reply is also the same whole and streamed; both with a call after the thinking
and with one inside it). A request with tools that may be
called has a rule. With `auto` the rule is free for the thinking and for the
text before a call: there the request takes the model's own drafts like a
request without tools, and the conductor stops taking what a window gives at
the token that makes the rule bind (the trigger, also when it stands inside the
thinking; see "Drafts" under [Sampling](#sampling)). Inside a call it takes the
lookup's drafts alone. A request that has to call (`required`, a named
function) may not end while it thinks, so its thinking rows carry a mask and
are not free: it takes the lookup's drafts alone for its whole reply.

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
`<tool_call>` inside the thinking as the end of the thinking with and without
tools, here only when the tools may be called; the strings of an `enum` are
enforced here and any text is accepted there; the white space between the
content or the reasoning and a call is part of neither here. The legacy
`functions` and `function_call` are refused.
The tests with real models (`crates/server/tests/real_tools.rs`: the 0.8B on
one card, the 27B on two) pass. Whether a model opens a call inside its
thinking is the model's business, so the test of that does not wait for it: it
bans the token of `</think>` with a `logit_bias` and puts a bias on the token
of the marker, the least of a ladder with which the reply has a call, and
offers tools without an argument of free text (a string may hold the marker,
and the bias would write it there). Under the ban alone the 0.8B thinks on to
the limit and never writes the marker. With a bias of 6 on the marker (8 for a
sampled reply) it thinks, opens the call inside its thinking, and the call
comes back with the reasoning before it, the same raw, streamed and without
drafts. The last step of the ladder is more than two logits are apart, so a
model that follows no smaller one has the marker as its first token.

**The Messages API.** `POST /v1/messages` is Anthropic's Messages API. A
request is served as the chat request that asks the same: the `anthropic`
module of `ares-server` writes its body over into a chat body, and from there
it is the chat route's path (the same reader, the same engine request, the same
prefix cache, the same rule for calls, the same end when the client leaves).
The reply comes back as a message with content blocks, or as Anthropic's
events.

| Field | Here |
| --- | --- |
| `model`, `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stream` | as on the chat route |
| `system` | a string or text blocks: one system message |
| `messages` | turns of `user` and `assistant`, a string or blocks. User: `text`, and `tool_result` (a string or text blocks), which becomes a message of the role `tool` where it stands. Assistant: `text`, `thinking`, `redacted_thinking`, `tool_use` |
| `tools` | `name`, `description`, `input_schema` (and `type: custom`, `strict`, `eager_input_streaming`, which ask for what is always done) |
| `tool_choice` | `auto`, `any` (the chat route's `required`), `tool` with a name, `none`; `disable_parallel_tool_use` allows one call |
| `thinking` | `enabled` and `adaptive` switch thinking on, `disabled` off; a request without it leaves it to `--reasoning` |
| `stop_sequences` | the stop strings |
| `output_config` | `format` with a JSON schema is the chat route's `response_format`; `effort` is its `reasoning_effort` |
| `metadata`, `service_tier` | bookkeeping: accepted, without effect |
| `cache_control` | accepted on the body, on blocks and on tools; without effect (below) |

The text blocks of one message are joined with a line break. A result that says
`is_error` has `Error: ` before its text, because the chat template has no
other place for it. Everything else is refused with a 400 that names it, in
Anthropic's shape of an error (`{"type": "error", "error": {"type",
"message"}}`, with the status the chat route gives): `image` and `document`
blocks (input other than text is not built), tools of a `type` that Anthropic's
service defines or runs (`web_search_...`, `bash_...`) and the blocks they
leave in a history, `defer_loading`, `mcp_servers`, `container`,
`context_management`, citations, a last message of the assistant (a prefill:
continuing a message is not built), and any member the module does not know.
`x-api-key`, `anthropic-version` and `anthropic-beta` are not looked at: the
server has no keys yet and one version of the API, and a field of a beta is
served or refused by its name in the body.

- **A budget for thinking** (`thinking.budget_tokens`) is checked and not
  enforced. The engine has none, and Claude Code sends one with every request
  that thinks, so a refusal would refuse the client. Thinking is on and as
  long as the model makes it; `max_tokens` bounds thinking and answer
  together, as at Anthropic, where the budget is a target too.
- **Signatures.** A thinking block of a reply carries the fixed signature
  `unsigned`: the server has no secret to sign with, and an empty signature
  may make a client drop the block. A thinking block that comes back is the
  reasoning of its message whatever its signature says. A reply's blocks sent
  back unchanged are the turn the chat route gets back with
  `reasoning_content`, so the template writes what the model generated and the
  prefix cache continues (see "A conversation that continues from the cache").
  `redacted_thinking` is accepted and nothing of it enters the prompt: only
  Anthropic's service can read it.
- **Cache markers.** The server caches every prefix by itself, so a
  `cache_control` marker changes nothing and its `ttl` is no promise. The
  usage has `cache_read_input_tokens` (the part of the prompt the cache gave),
  `input_tokens` (the rest: the two add up to the prompt, as at Anthropic) and
  `cache_creation_input_tokens`, which is 0.
- **Why a reply ended.** `end_turn`, `max_tokens` or `tool_use`. The engine
  reports one reason for the end token and for a stop string, so a reply that
  a stop sequence ended says `end_turn` with `stop_sequence: null`. Telling
  them apart needs the conductor to say which stop string matched.
- **The billing line of Claude Code.** Claude Code begins its system prompt
  with `x-anthropic-billing-header: ...; cch=a5145;`, whose `cch` is another
  one in every request. Left alone it would part every prompt from the cached
  one at its first line, so its five characters are overwritten, as
  llama-server does. Nothing else of a prompt is changed.

A stream is `message_start`, then for every block `content_block_start`, its
deltas (`thinking_delta`, `text_delta`, `input_json_delta`; a thinking block
gets a `signature_delta` before it closes) and `content_block_stop`, then
`message_delta` with the reason and the usage, and `message_stop`. The blocks
are in the order the model wrote them and numbered from 0; a whole message is
the same events put together, so the two cannot differ. The engine counts the
prompt when the reply ends, so the usage of `message_start` is zeros and
`message_delta` has all of it. While the engine has nothing to say the stream
sends `ping`. A failure after the stream began is an `error` event, and the
stream ends without `message_stop`. The input of a `tool_use` block is the
JSON text the model wrote, byte for byte; a call that `max_tokens` cut has the
fragments in a stream and `{}` in a whole message. Call ids are `toolu_`, the
request's id and the place of the call.

`POST /v1/messages/count_tokens` reads a body as `/v1/messages` does (without
`max_tokens` if the client leaves it out) and returns `{"input_tokens": N}`,
the tokens of the prompt the engine would have been given; nothing is
generated. The count is `Engine::count_tokens` of `ares-api`, which renders and
tokenizes a prompt; the runtime answers it with the model's own encoding of the
prompt, so the count is the prompt a run of the same request reports (computed
plus cached), which a host test checks through the real runtime. An engine that
cannot count answers 501.

Both routes run on the core of the chat route in `routes.rs`: `prepare`
applies what a parsed chat request says about thinking and tools and returns
the sorter of its reply, `Reply::start` hands the request to the engine, and
`Reply::next` gives the reply as pieces (reasoning, content, the start of a
call, fragments of its arguments) and its end (reason, usage, timings). A
route of another API writes its body over into the chat body, calls these, and
writes the pieces in its own format.

Where this is not llama-server: it concatenates system blocks with nothing
between them, moves the text of a user turn before its tool results, reads
`tool_choice` `tool` as `any` and does not read `none`, drops
`thinking: disabled`, `is_error`, `disable_parallel_tool_use` and every member
it does not know, passes images on, and enforces `budget_tokens` in its
sampler. Its `stop_reason` is `end_turn` for a stop string as well, but it
names the string in `stop_sequence`. The tests with a real model
(`crates/server/tests/real_messages.rs`, the 0.8B on one card) pass.

**The Responses API.** `POST /v1/responses` is OpenAI's Responses API,
without state. A request is served as the chat request that asks the same, as
a Messages request is: the `responses` module of `ares-server` writes its body
over into a chat body, and from there it is the chat route's path (the same
reader, the same engine request, the same prefix cache, the same rule for
calls, the same end when the client leaves). The reply comes back as a
response object with its output items, or as the typed events of the API.

| Field | Here |
| --- | --- |
| `model`, `temperature`, `top_p`, `stream` | as on the chat route |
| `instructions` | a system message |
| `input` | a string (one message of the user), or items: messages (`user`, `assistant`, `system`, `developer`; a string or `input_text` and `output_text` parts), `function_call`, `function_call_output` (a string or `input_text` parts), `reasoning` |
| `tools` | functions in the shape of this API: `type: "function"`, `name`, `description`, `parameters`, `strict` (the arguments are held to the schema either way) |
| `tool_choice` | `auto`, `none`, `required`, `{"type": "function", "name"}` |
| `parallel_tool_calls` | as on the chat route |
| `max_output_tokens` | the limit of tokens; a reply it cuts has the status `incomplete` and `incomplete_details.reason: "max_output_tokens"` |
| `reasoning` | `effort` is the chat route's `reasoning_effort` (`none` switches thinking off, any other value is the template's variable); `summary: "auto"` is accepted and no summary is written |
| `text.format` | `text`, `json_object`, `json_schema`: the chat route's `response_format` |
| `include` | `reasoning.encrypted_content` is accepted and adds nothing (below) |
| `truncation` | `disabled`, which is what the server does |
| `store`, `background` | `false` (below) |
| `metadata`, `user`, `safety_identifier`, `service_tier`, `client_metadata` | bookkeeping: accepted, without effect; `metadata` and `user` are said back |
| `prompt_cache_key`, `prompt_cache_retention`, `prompt_cache_options` | accepted, without effect, as `cache_control` is on `/v1/messages`: the server caches every prefix by itself and drops what it kept by its own budget, so a retention is no promise |

The parts of one message are concatenated, as the chat route does it. The
items one reply returned for a turn (a reasoning item, a message, calls) are
one message of the assistant again; an item that does not continue a turn in
that order begins another message. `instructions` and the messages of the
roles `system` and `developer` that stand before the conversation are one
system message, joined by an empty line: Codex puts its instructions at the
head of `input` and sends no `instructions` (read in its source, not tried),
and the chat template of Qwen3.5-0.8B refuses a system message that is not
the first one (`System message must be at the beginning.`, read in the model
file). Such a message inside a conversation stays where it stands, and the
template decides about it: that template refuses it.

Everything else is refused with a 400 that names it, in OpenAI's shape of an
error, as on the chat route: `input_image`, `input_file` and `input_audio`
parts (input other than text is not built), tools that are not functions
(`web_search`, `file_search`, `code_interpreter`, `computer_use`, `mcp`,
`local_shell`, `custom` and the like), the items such tools leave in a
history and a `tool_choice` that names one or a subset (`allowed_tools`),
`prompt` templates, `top_logprobs` and `include:
["message.output_text.logprobs"]` (the pieces of a reply are not told apart
into the tokens of the answer, of the reasoning and of calls),
`text.verbosity`, `reasoning.summary` `concise` and `detailed`,
`truncation: "auto"`, `max_tool_calls`, `stream_options.include_obfuscation:
true`, and any member the module does not know. Only members of this API are
read: the sampler's fields of the chat route (`top_k`, `min_p`, `seed`, `stop`
and the others) are refused here, as on `/v1/messages`.

- **No state** (open decision 20 of the feature checklist). The route keeps
  no responses, as llama-server, ollama and Strata. A response says
  `"store": false`, also when the request left `store` out, where OpenAI's
  default is `true`. A request that sets `store: true` is refused: it asks
  for something the server does not do, and the clients that work without
  stored responses say so themselves. Codex sends `store: false` with every
  request (read in its source, `codex-rs/core/src/client.rs`, on 2026-11-12);
  the OpenAI SDKs send the member only when their caller sets it.
  `previous_response_id`, `conversation`, `background: true`, an
  `item_reference` in `input` and a `prompt` template are refused by name,
  and `GET` and `DELETE /v1/responses/{id}`, `cancel`, `input_items` and
  `compact` answer 404 and say that no response is kept. Stored responses are
  row A05 of the checklist and are not built.
- **Reasoning items.** The reasoning of a reply is one item of the type
  `reasoning`, before the message. Its `content` is the reasoning as the
  model wrote it, in a part of the type `reasoning_text`, which is what the
  API has for models whose reasoning is not hidden; its `summary` is empty,
  because the server writes none; and it has no `encrypted_content`, because
  the server has nothing to hide and no secret to encrypt with. A reasoning
  item that comes back is the reasoning of its turn: the text of its
  `content`, whatever its id says. The item of a reply sent back unchanged is
  therefore the turn the chat route gets back with `reasoning_content`, so the
  template writes what the model generated, on every turn the same, and the
  prefix cache continues (see "A conversation that continues from the
  cache"). A `summary` and an `encrypted_content` of an item are accepted and
  nothing of them enters the prompt: a summary is not the reasoning, and
  encrypted content is text only the service that wrote it can read. A client
  that sends back only those two (and not `content`) gets the same answers and
  pays what a chat client pays that drops `reasoning_content`: the last turn
  is computed again.
- **`include`.** `reasoning.encrypted_content` asks for reasoning items that
  a client without stored responses can send back, and Codex asks for it with
  every request. Every reasoning item of this server is such an item, so the
  value is accepted and adds nothing. The other values are refused by name.
- **`truncation`.** `disabled` is what the server does: an input that does
  not fit the context is the error `context_length_exceeded`. `auto` would
  drop items from the conversation, which the server does not do.
- **The usage.** `input_tokens` is the whole prompt,
  `input_tokens_details.cached_tokens` the part of it the prefix cache gave,
  `output_tokens` what was generated. There are no `output_tokens_details`:
  the engine does not count the tokens of the reasoning apart
  (`ares_api::Usage` has one count for the reply), and a number that is not
  measured is not written. The response has the `timings` of the chat route
  next to the usage.
- **`output_text`.** The response has no member of that name. The official
  SDKs put it together from the items; llama-server does not send it either,
  and its own test reads it from the Python SDK.
- **What a response says back.** `instructions`, `max_output_tokens`,
  `temperature`, `top_p`, `tools`, `tool_choice`, `parallel_tool_calls`,
  `text`, `reasoning`, `metadata` and `user` as the request gave them (`null`
  where it gave nothing: the engine's own defaults are not known to the
  route), and `store: false`, `background: false`, `truncation: "disabled"`,
  `previous_response_id: null`.

The output is in the order the model wrote it: the reasoning item, the
message (one `output_text` part), then one `function_call` item for every
call, with `call_id` (`call_`, the request's id and the place of the call),
`name` and `arguments` as the JSON text the model wrote. The ids of items are
`rs_`, `msg_` or `fc_`, the request's id and the place in the output. An item
the limit of tokens cut has the status `incomplete`, and a call it cut has its
arguments as far as they came.

A stream is `response.created` and `response.in_progress`, then for every
item `response.output_item.added`, its events and `response.output_item.done`,
then `response.completed` or `response.incomplete` with the response as it is
returned whole; every event has its `sequence_number`, counted from 0. A
reasoning item and a message have `response.content_part.added`, their deltas
(`response.reasoning_text.delta`, `response.output_text.delta`), the text once
more (`response.reasoning_text.done`, `response.output_text.done`) and
`response.content_part.done`; a call has
`response.function_call_arguments.delta` and
`response.function_call_arguments.done`. The stream and the whole response are
made from the same thing (`Output` in `responses/reply.rs`, which takes the
pieces of a reply and keeps the items), so the two cannot differ. The usage is
in the last event only, since the engine counts the prompt when the reply
ends. A failure after the stream began is an `error` event (`code`, `message`,
`param`), and the stream ends with it. There is no `[DONE]`.

`POST /v1/responses/input_tokens` reads a body as `/v1/responses` does, and
`POST /v1/chat/completions/input_tokens` one as `/v1/chat/completions` does;
both return `{"object": "response.input_tokens", "input_tokens": N}`, the
shape OpenAI and llama-server have, and generate nothing. The count is
`Engine::count_tokens`, as for `/v1/messages/count_tokens`, so it is the
prompt a run of the same request reports (host tests on the scripted engine
and through the real runtime).

Where this is not llama-server: it drops tools that are not functions,
`store`, `background`, `truncation`, `include`, `text` and every member it
does not know without a word, passes the chat route's fields through, and
reads a `tool_choice` that names a function as `auto`; here each is served or
refused. It requires a reasoning item to have a `summary` and a `content`
that is not empty, and writes `encrypted_content: ""`. Its stream has no
`sequence_number`, no `response.reasoning_text.done`, no
`response.function_call_arguments.done` and no events for the part of a
reasoning item, it numbers the items by what kinds it has seen, and it calls
every response `completed`, also one the limit of tokens cut.

What is not there: the count of reasoning tokens (it needs the engine), a
switch that turns thinking on for a template whose default is off and that
does not read `reasoning_effort` (the route has `effort` alone, and
`--reasoning` decides for the server), the probabilities of tokens, custom
tools (a tool that takes free text), and a `text.format` next to tools that
may be called (the reply follows one rule, as on the chat route). What Codex
sends was read in its source and not tried against a running Codex. The
tests with a real model (`crates/server/tests/real_responses.rs`, the 0.8B on
one card) pass.

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

Multimodal input is not built.

**Several models (router mode).** `ares --models-preset <file.ini>` serves the
models of a preset file. A model is loaded when a request names it, stays
loaded while the devices and the RAM hold it next to the others, and is
unloaded when another model needs its room or when it was idle for its time.
`ares --model <file.gguf>` is the same server with one preset that is loaded
before the server listens and never unloaded; a load that fails there ends
the start, as before.

*The preset file* is an INI file (`crates/server/src/preset.rs`). The section
`[*]` holds what is the same for every model; every other section is one
model, and its name is the id requests use and `/v1/models` lists.

```ini
[*]
ram = 24G
idle-unload = 30m

[qwen-27b]
model = /models/Qwen3.8-27B-UD-Q4_K_S.gguf
aliases = default, gpt-4o
reasoning-effort = medium
load-on-startup = true

[small]
model = /models/Qwen3.5-0.8B-Q4_K_M.gguf
devices = 0
context = 16384
ram = 2G
```

A key is an option of the command line for one model under its name without
the dashes, with the value the option takes (`context`, `kv`, `devices`,
`chat-var`, which may stand several times; a switch such as
`reasoning-preserve` takes `true` or `false`). The command line and the file
are read by the same code (`options.rs`), so there is one vocabulary and one
check of a value. Three keys exist only in the file:

| Key | Meaning | Default |
| --- | --- | --- |
| `aliases` | more ids the model answers to, separated by commas | none |
| `load-on-startup` | `true`: loaded when the server starts, in the background, not at its first request | `false` |
| `idle-unload` | how long the model stays loaded without a request: seconds, or a number with `s`, `m` or `h`; `never` | `never` |

What holds for a model is, from the strongest: the command line (an option of
a model given next to `--models-preset` holds for every model), the model's
section, the section `[*]`, the built-in defaults. `ram` is the one key that
means something else by where it stands: on the command line and in `[*]` it
is the most RAM of the server, for all loaded models together (default: half
of what the machine has available when the server starts); in a model's
section it is that model's part of it, and a model without one takes what the
limit has left when it is loaded. `host` and `port` are the server's and stand
on the command line only; `--model` and `--models-preset` exclude each other.

Nothing in the file is ignored. The server does not start, and says the file,
the line and the reason, for: a key it does not know, a value an option does
not take, a key before the first section, a key or a section that stands
twice, a section without `model`, a model file that does not exist, an id or
alias that two sections answer to, `model` or `aliases` in `[*]`, and a
model's `ram` above the server's. An option of llama-server is not accepted
under its name: the error names the option of Ares that does the job where
there is one (`ctx-size`: `context`; `tensor-split`: `shares`; `ctk` and
`ctv`: `kv`; `chat-template-kwargs`: `chat-var`; `sleep-idle-seconds`:
`idle-unload`; `alias`: `aliases`, and the others of `other_name` in
`preset.rs`). What only the devices or the model file can say (a split the
devices cannot do, a context that fits nowhere) is found when the model is
loaded, and is then the reason of a failed load.

*A request and its model* (`crates/server/src/models.rs`). The `model` of a
request is an id or an alias; anything else is a 404 `model_not_found` that
lists the ids. The request takes a hold on the model for as long as it runs.
If the model is not loaded the request waits until it is, behind a load that
already runs, and the answer names the model by its id whatever alias the
request used. A client that goes away while it waits gives its place up: a
load nobody waits for is not started, and one that has begun is finished (the
way of the weights to the devices is not interrupted) and leaves the model
loaded and idle. One task does every load and every unload, one at a time, so
that what the devices report as free is what the next load is planned
against.

*Making room.* Nothing counts models. The loader plans a model as on the
command line, against what the devices report as free and what the server's
RAM limit has left, and says "no room" before it reads the weights when the
plan does not fit: the estimate of the model with the context its preset
names is more than a device has free; or, for a preset that names no context,
less than the least context a device is chosen for (8192 tokens, or the
model's own) fits; or the RAM left is less than what the engine needs to run.
Then the model that has been idle the longest is unloaded and the load is
planned again, until it fits or no model is left, in which case the load
fails with the plan's words. A preset that names no context gets what the
devices have free when it is loaded, as on the command line, which next to
other models is less than alone: presets that are to share a device name
their `context`, and their `ram`.

A model that is serving requests is never unloaded under them. When a load
finds no room and only such models are loaded, the one used least recently
gives way: it gets no new requests (they wait in line, behind the load, and
load it again afterwards), and it is unloaded when its running requests have
ended. The wait of the new request is therefore bounded by the requests that
run when it arrives, so it waits, and no request is answered 503 because
another model is busy (open decision 16 of the feature checklist: Ares waits
here as it does for a full queue). `/v1/models` shows such a model as
`loading` with the reason `waiting for room: ...`. What this costs: with
requests for two models that cannot be loaded together arriving all the
time, the server changes between them at the pace of its loads.

*Idle unload.* A model whose preset has an `idle-unload` is unloaded when no
request has held it for that long; the time counts from the end of its last
request. The next request loads it again. The time is one setting of a
preset, with `[*]` as the default for all (open decision 19).

*What an unload releases.* The engine of the model is shut down (its
requests ended, its threads joined) and dropped, with the pages of sequence
state, the weights and the page-locked RAM it holds; a device that no other
model is on is closed with its driver context, so that a server with nothing
loaded holds no device memory. The unload compares what every device reports
as free before and after with what the model had taken at its load, logs
both, and warns when less came back than was taken.

The prefix cache does not outlive its model. Its cells are pages of K and V
in VRAM and its checkpoints are page-locked RAM, both the engine's, and a
checkpoint without the pages it belongs to resumes nothing; keeping either
across an unload would keep the memory the unload is for. A model that is
loaded again starts with an empty cache, and the first request of a
conversation computes its prompt again. Keeping a conversation across an
unload needs the pages on the SSD, written as they are produced (see
"Loading and unloading"), which is not built.

*Endpoints.*

| Endpoint | What it does |
| --- | --- |
| `GET /v1/models` | every preset in the order of the file: `id`, `aliases`, `status` (`{"value": "unloaded" \| "loading" \| "loaded" \| "unloading" \| "failed", "reason"}`), and `created` and `context_length` once the model was loaded |
| `POST /models/load` `{"model": id}` | puts the model in the line of loads and answers `{"success": true}` at once |
| `POST /models/unload` `{"model": id}` | the model gets no new requests and is unloaded when those that run have ended; answers `{"success": true}` at once |
| `GET /models/sse` | Server-Sent Events: the state of every model as it is, then every change, as `{"model", "event": "model_status", "data": {"status", "reason"}}` |
| `GET /metrics?model=id` | the statistics of one loaded model; without `model` when the server has one |

The bodies and the answers of the two `POST` routes and the events are
llama-server's. Where they differ: a failure is a state of its own with its
reason (llama-server writes `unloaded` with `failed: true` and an exit code,
since its models are processes); a member of a body other than `model` is
refused; the event stream begins with the state of every model, so that a
client that connects late knows where it is; and there are no `args`,
`progress` or download events. A load asked through the endpoint is not
waited for by the answer, as in llama-server: whether it succeeded is in the
list and in the events. An unload does not end running requests; a way to
end them by force is not built.

*Failures and the end.* A load that fails leaves its model `failed` with the
reason, which every request that waited for it gets as a 503
`model_load_failed`; the other models are not touched, and the next request
for the model tries the load again. An engine whose device fails says so
(`Engine::failure`; the runtime's conductor sets it when a step fails on a
device): its model gets no new requests, is unloaded when the requests the
failure ended have let go, and is listed as `failed` with the engine's reason
until a request loads it again; models on other devices serve on. On the
first signal the server stops as before, and then unloads every model before
the process exits; a load that runs then is finished first.

*What is not there.* The plan of a load cannot be asked without the load
(the mixture-of-experts model's placement reads the devices itself), so the
router finds the models that have to go by trying: a model that fits
nowhere has unloaded the idle models, and waited for the busy ones, before
it fails. A model is planned against free memory only, so a preset without a
`context` that is loaded next to others gets less than alone. The RAM limits
of loaded models are parts of the server's limit that are booked at load;
the models do not give RAM to each other while they run. A model option
given on the command line needs text that is UTF-8, paths included.

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
