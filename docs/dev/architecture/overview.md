# Architecture of Ares

This document describes how the engine is built and why. It is the place to
start before changing anything. What runs today is in [`user/features.md`](../../user/features.md),
what is open in [`OPEN.md`](../../OPEN.md).

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
  with draft trees, image input, the decision endpoint and the prompt cache
  on SSD. The list of features, who has each and
  how far Ares is, is
  [reference/feature-checklist.md](../studies/feature-checklist.md); a
  release date is not a reason to leave a row out.

- **Nothing is tuned to one machine.** The CUDA backend serves the GPU
  classes it was tested on (today Ampere 8.6 and Ada 8.9, see
  [Devices](devices.md)), the Vulkan backend every other. Which kernel, block
  shape or row-count boundary a device gets may depend on its class (its
  compute capability on CUDA: "Ampere and Ada use this"; what the device
  reports on Vulkan) or on something measured on the device at load (the
  tensor cores' rate with `f32` against `f16` accumulation, memory
  bandwidth, the link to another device), never on a device name, a vendor
  string, an ordinal, or a constant that is only the result of a measurement
  on the development machine. The numbers in this document come from an RTX
  4070, an RTX 3060 and an RTX 4060 Ti; a class that was not run is not
  built.

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
  experts of a MoE model that are not on the GPU (see [Weights](weights.md)).
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
| `ares-sampler` | The definitions of the sampler: the settings of a row, the arithmetic every backend repeats bit for bit, the pick of one row written from its definitions (`sample_reference`), the penalties, DRY and logit bias the host works out, token masks | Done; the reference is checked against llama.cpp's samplers written again in double precision, the penalties against their definition; both backends' kernels are held to it bit for bit; see [Sampling](sampling.md) |
| `ares-cuda` | Driver API bound at run time, devices, memory, streams, events, modules, decode weight kernels for all 28 weight types, prefill weight kernels for eleven, the layer kernels of `qwen35`, attention over the stored K/V formats, the sampler's kernels, the image tower's kernels besides its products, the mesh that sums across devices over peer access or the host | Device layer, decode, layer and attention kernels done; prefill kernels verified, their pass over the weights 1.3 times as fast as the fork's whole prefill; tested on both GPUs, with and without peer access |
| `ares-grammar` | The grammar engine on the host: GBNF grammars, JSON schemas and Lark grammars turned into the set of tokens that may come next (llguidance behind a vocabulary of bytes per token), the reasoning gate, the cut of forbidden drafts | Done; GPU-free tests of the engine, and through the runtime and the server; see [Sampling](sampling.md) |
| `ares-vulkan` | The same device-level surface on Vulkan; decode and prefill weight kernels for all 28 weight types as compute shaders, the prefill ones on cooperative matrices; the layer kernels; attention over the stored K/V formats; the sampler's kernels; sums between devices | Device layer and weight, layer, attention and sampler kernels done (23 device, 9 kernel, 8 layer, 4 layout, 9 mesh, 9 attention and 8 sampler tests on both GPUs; the sampler's records equal `ares-sampler`'s reference bit for bit); no expert or `qwen4exp` kernels and no model yet |
| `ares-state` | Page pool, page tables with copy on write, the prefix index (pages published as they fill, tails, announcements, priority classes), the checkpoint policy for recurrent state, page layouts | Done as bookkeeping; randomised model checks; the conductor's prefix cache uses them (see `ares-runtime`) |
| `ares-store` | State on disk: pages and checkpoints by content hash in segment files, per fingerprint, with a size limit, read and written with direct I/O outside of its index | Done as a store of opaque bytes, tested against restarts, the limit and damaged files; the prefix cache of `qwen35` keeps its level on SSD in it (`--cache`) |
| `ares-io` | Reads of weights straight from the model file: io_uring with direct I/O, demand before prefetch, cancel and promote, copies of a file on several drives with the reads spread over them, a map from experts to file ranges, row reads behind a cache | Done, tested against real files and on two drives; 1.4 to 1.7 GiB/s of random expert-sized reads per SSD of the development machine and 2.9 to 3.0 GiB/s over both; the expert store and the n-gram rows of `qwen4exp` read through it |
| `ares-sched` | Step planner (generation and prompts in one step), the share of time between generation and prompts, admission (pages claimed per step, a reservation that corrects itself, a queue instead of refusals), the price of a step by its rows with a term for the context, draft length choice and the acceptance of drafts, the draft policy of the rounds that ties them together (`Drafting`: the windows a round verifies, the bars of a chain's links, the worth of a token), the n-gram drafter on a suffix automaton shared by all sequences | Done; unit and model tests; the conductor uses all of it (admission without spill to RAM) |
| `ares-weights` | Expert heat, tiers that are elected and fed by reads (admission, protection, guesses bound late, slots given back), slots per layer by need, a token table for the first layers, the lookahead distance, which experts a tier holds (for the routers' pruning) | Done, unit tested; the expert store uses the slot sets, the heat, the swap rule, the split of slots and the guess queue (its reads ahead of a router); the token table is not used by a running model yet, and elections only by the simulations of `tests/premises` (the lookahead distance only by a guess of `qwen4exp` that is switched off) |
| `ares-api` | Request (with every sampling field), event (text and probabilities), decision and error types; the `Engine` trait between HTTP and the runtime (`submit` for generation, `decide` for decisions, `stats`, `chat_format` for what a model's chat template declares about thinking and tool calls, `count_tokens`, `tokenize`, `pieces`, `failure`; the `Structure` of a reply and the `Json` value of requests) | Done, unit tested |
| `ares-tokenizer` | Text to token ids and back for the byte-level BPE vocabularies of GGUF models, as llama.cpp does; the chat templates (the Jinja dialect of the supported models) and what a template declares about thinking and tool calls | Done; tests against `llama-tokenize` and `llama-server --jinja` |
| `ares-server` | The `ares` binary: loads a model on the devices and serves it over an OpenAI-compatible HTTP API, with the plan of devices, split, shares and context made from what the devices report | Serves `qwen35` and `qwen4exp` models (chat and plain completions, streamed and not, every sampling field and the probabilities of the tokens, `n` completions, timings, metrics, staged shutdown, the reasoning of a reply apart from its content and in the history, with llama-server's controls for thinking, tool calls held to the tools' schemas by a grammar, also when the model opens the call inside its thinking); unit tests in `src` and route suites against a scripted engine without a GPU, suites that start the real binary (some need the 27B or the files of Flash-Next) and suites that run the engine in the test's process (the suites of the routes on the 0.8B, or on any file the server loads with `ARES_TEST_API_MODEL`: they pass on Flash-Next); for `qwen35` the prefix cache is on, with `--ram` for the engine's RAM, for `qwen4exp` the RAM is mostly the tier of experts; a router mode that serves several models from a preset file, loads them on request and unloads them for room or when idle (host tests of the preset reader and of the state machine on a scripted loader, and `tests/real_router.rs` on real devices); images through a projector (`--mmproj`, see [Images in a prompt](images.md#images-in-a-prompt)), decisions on `/v1/systemone` as Flide serves them |
| `ares-runtime` | The conductor thread and one thread per device, the step loop that connects plan, state and kernels, admission, speculation with the n-gram drafter and with a model's drafts, the `Model` trait a model implements, a synthetic model that drives the real kernels, the prefix cache with its levels in RAM and on SSD, and the expert store (VRAM slots, RAM tier, reads, the CPU computing what is not on a GPU) | Drives `qwen35` and `qwen4exp` on one device, in tensor split and in layer split, with picks made on the device; implements `Engine::submit` and `Engine::decide` (decisions are a parent prompt and its forks through the prefix cache, see "Decisions" in [sequence-state.md](sequence-state.md)); the prefix cache is in the conductor, checked against the host executor as an oracle and carried out by both models on the devices (checkpoints in page-locked RAM within one RAM limit, `PinnedRam`; pages that leave the devices go to RAM and, with `--cache`, to the SSD), and a request that gives way is not spilled with its pages: what it computed stays in the prefix cache, and it computes again what the cache no longer holds; the expert store is driven by the device threads of `qwen4exp` (see [Qwen3.8-Flash-Next through the runtime](flash-next.md)) |
| `ares-model` | Model architectures: configuration and weight table from a GGUF, a CPU reference of the forward pass with explicit state (window and commit) that GPU implementations are compared with block by block, and the implementation on the devices | `qwen35` (Qwen3.5-0.8B, Qwen3.8-27B) runs through the runtime and the server, with the model's MTP block. `qwen4exp` (Qwen3.8-Flash-Next) runs through the runtime and the server on one or two cards, without speculation (it has no draft head built, and the n-gram drafter takes part only next to a model drafter), and through a session of its own that tests and measurements use. See [Several devices](devices.md#several-devices). The image tower `qwen3vl` (a CPU reference and the pass on one device, its weights in VRAM or in page-locked RAM) feeds both models' image cells (see [Images in a prompt](images.md#images-in-a-prompt)) |

Both test architectures are specified under [studies/](../studies/):
[qwen35](../studies/qwen35-architecture.md) and
[qwen4exp](../studies/qwen4exp-architecture.md). The projector for images is
described under [The image tower](images.md).

### The shared model layer

Ares supports a few models, and each gets its maximum through a launch plan of
its own; what can be shared is shared in `ares-model`'s module `blocks`. A
**block** is a struct that owns one kind of state and every rule about it:
where the state lives, which rows of a step touch it how, and what a draft
window, a commit, a replay or a copy does to it. It has no graph and no list
of operators; a model calls its methods in the model's own order, with device
pointers that the model's own products wrote, and a block adds no launch to
what the model enqueued before it existed.

**The rule.** A part of a model becomes a shared block when it owns state that
the runtime's contracts touch (pages, slot state, checkpoints, windows,
commits, replays, trees) or when it drafts. Everything between two pieces of
state stays with the model, in its launch plan: the products, the norms and
gates, the activation forms, which kernel runs at which row count, fused
launches and the exchanges between devices. A model may replace a launch of a
block with a fused kernel of its own as long as the block's tests pass with
it; it may not keep state outside a block. A stateless part (a norm, an FFN)
is shared only when a second model uses it as it is.

The blocks so far:

- **The K/V group** (`blocks/kv.rs`, with the tile forms of the exact pool in
  `blocks/forms.rs`): attention layers whose K and V share one pair of
  formats, with their entries in the engine's pages, the exact pool of the
  formats with a tile mean and the group's table. It appends a step's rows
  (not those of a replay), plans them (a prompt's together, a generated row
  as the only row of a step, with the tile that leaves the newest cells
  inside a draft window read converted from its position on), prepares a
  step (conversions, exact slots, the table), settles a draft window at its
  commit, and copies, sends to the host and takes back a page in the form it
  has. The 27B has two groups: its attention blocks, and its MTP block.
  Flash-Next's attention layers are a group too, and so is its MTP block
  (a part of one layer), whose K and V are `q4` whatever the model's. Its
  table is the attention half of the part's tables, which hold the
  conductor's entries; the group writes the entries of exact tiles over them
  at the start of a step. Where every row is within the indexer's dense
  limit the group plans the rows as for the 27B (a generated row as the only
  row of a step: a row is twelve vectors there, so it reads the tensor-core
  kernels either way, and a window's rows share their tiles). Past the limit
  the indexer's selection kernel fills each row's lists on the device, cut by
  the kind of entry the table names (`Q4Kernels::select` with the tables): a
  list of cells in converted entries, read on the direct kernel, and one of
  cells in exact entries, read on the tensor cores, each with its pieces. A
  generated row has the pieces it has as the only row of a step, the rows of
  a prompt share the step's. The selection is Flash-Next's; the group owns
  which entries the lists name. Measured 2026-11-16 (IQ2_XS on both cards):
  K, V and the indexer's keys take 13.92 MiB per 1000 cells with `q8`, 8.24
  with `q4`; at 131 072 cells placed by layers that is 912, 636 and 540 MiB
  per card with `q8`, `q5` and `q4`, and the expert slots grow by 538 and
  726 MiB (about 515 slots with `q4`). With `q8` a decode row is planned as
  before (the same bits; the attention takes 0.51 ms of a step on the first
  card before and after).
- **The recurrent block** (`blocks/recurrent.rs`): the layers of the gated
  delta rule with their extents in a sequence's slot (a convolution window
  and the delta-rule matrices per layer) and the inputs a draft window keeps.
  It decides where a step's inputs go (kept for a window, the shared scratch
  otherwise), runs the window (convolution, delta rule, and the commit of the
  convolution windows of the direct rows) and the deferred commit of a
  window's accepted rows. Both models' recurrent layers run on it; what feeds
  it (the 27B's norm and products, Flash-Next's stream mixer) and what follows
  it (the gated norm in each model's form, the output product) stay theirs.
- **The slot arena and its checkpoints** (`blocks/slots.rs`): the state a
  device part keeps per sequence, and what the prefix cache does with it and
  with the pages. The arena holds a slot per sequence slot; the blocks that
  keep state per sequence declare their extents in it one after the other
  (`SlotLayout`: the recurrent block its windows and matrices, Flash-Next's
  indexers their carried keys and its n-gram block its history), and a
  sequence that starts in a slot starts from zeros. A **checkpoint** is the
  extents a model declares (the slot, and state outside it: an MTP block's
  carry, the slot of an MTP block that is a part of its own) and bytes the
  host keeps of the sequence itself (Flash-Next's n-gram feed: its position
  and last tokens), in a place in page-locked RAM (`Places`, locked in chunks
  as they come into use). A **page** is every K/V group of the device (tag bit
  `1 << i` in transit for the `i`-th, which says whether its entries are exact)
  and every array kept per page besides them (an indexer's pooled keys): a
  page copy copies all of it, a page goes to the host and back with all of it.
  The copies of a record run in its order (`enqueue_copies`: saves, page
  copies, restores), and what the prefix cache is told (`PrefixCache`,
  `Durable`) is computed in one place from what each device declares before
  it opens (`Declared`: the bytes of a checkpoint, the groups with their
  formats and geometries, the bytes of the arrays); the fingerprint stays the
  model's. Both models run on it: the 27B's checkpoint is its slot and its
  MTP block's carry row, Flash-Next's is described in "Qwen3.8-Flash-Next
  through the runtime". Moved from the 27B as it was (its five fingerprint
  cases bit for bit, the same copies and kernels in the same order; measured
  2026-11-16 in alternation, three runs each at loads of 6 to 28: a decode
  step at 1000 tokens of the tensor split 25.73 ms before and after, a prompt
  of 2048 tokens 1309 and 1313 tokens/s).
- **Token rows** (`blocks/rows.rs`): where the rows of a token embedding table
  come from. The table's home on the host is page-locked memory (the 27B, read
  once at load and counted in the one RAM limit) or the mapped model file
  (Flash-Next, whose RAM is the tier of its experts); a step copies the rows
  of its tokens in the file's form into a stage and uploads them with one
  copy, and the gather decodes them on the device (`embed_gather`). A device
  keeps the rows of the tokens a drafter picks there (the head view's), which
  the host cannot stage. A second source of rows (`splice_rows`: rows another
  computation leaves in the model's width, the image tower's later) is copied
  over the gathered ones before the first layer reads them. The 27B moved onto
  it as it was; Flash-Next no longer decodes and uploads every row as floats
  (10 240 bytes a row) but stages its IQ4_XS rows (1 360) and gathers them on
  the device, whose decode is the host's operation for operation (the same
  bits). Its step inputs no longer upload the section of the rows' values,
  which only the device writes.
- **The head view** (`blocks/head.rs`): the order of the output head's rows by
  a draft vocabulary (its tokens first, ascending) and the rows of the draft
  head, which every drafter that projects to the vocabulary reads, the MTP
  block and the DFlash drafter on both models (`--draft-vocab`). The model's
  logits come in the same order, and a list on the device names each column's
  token for the sampler.
- **The MTP driver** (`blocks/mtp.rs`): one for every model, described under
  "The MTP driver" in [Speculation](speculation.md). A model gives it its MTP
  layer and where its hidden state is tapped.
- **The step's inputs and drafts** (`blocks/step.rs`): what a part reads from
  the step record (tokens, segments, replays, rotary positions, the windows
  whose drafts the MTP block or the DFlash drafter makes, the table updates,
  the outcome of the windows before) with one set of checks, the bounds of a
  step's segments and the K/V group's limit of a window, and how the drafters
  of a part hand out a step's drafts (one list, their probabilities, their
  device time).
- **The arena of weights** (`blocks/arena.rs`): a stateless part both models
  use as it is: every tensor at a multiple of 64 bytes, the stored bytes, rows
  or columns gathered into a matrix of their own, or values the host made.
  The bytes are uploaded from the file's mapping where its pages are in the
  page cache (one `mincore`); the others are read from the file with ordinary
  reads a piece at a time and uploaded from there, because the kernel stops
  reading around the faults of a mapping that keeps missing the page cache,
  and every fault then reads one page (until 2026-11-27 only on a machine short
  of memory for a load; a reload after the kernel had taken part of the file's
  pages back read 5 to 20 MB per second, GitHub issue #66).
  The 27B keeps its layout chooser on top.

What every model reads from its files the same way is in the module `files`:
the strict reader of an architecture's metadata keys (the architecture check,
the refusal of a key it does not know, counts, floats and arrays), the weight
table (each tensor by name with the dimensions the configuration gives it, a
matrix's rows checked to be whole blocks whose first one decodes, a tensor no
one took refused), the stored `Matrix`, and one `FileError` that names the key
or tensor and that the errors of `qwen35`, `qwen4exp`, `qwen3vl` and `dflash`
wrap. Each model keeps its list of keys and tensors and its own checks of how
the values fit together.

What the CPU references share is the module `reference`: the dot product and
the matrix product with their fixed orders of summation, the threads over
ranges of rows, the scalar functions and norms, the delta rule and the
convolution of a recurrent layer, and what a window takes besides its tokens.
Each model's reference keeps its own blocks and names in its `pieces` the
functions its kernels are held to. (The image tower's and the DFlash drafter's
references sum in other orders, on purpose.) What a model is to the text of a
request (its name and context, tokenizer, chat options and format, the
projector images are read through) is `chat::ModelText`, which both models'
`Model` impls take their text half from.

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
   reach their rate, `Model::prompt_rows`: for `qwen35` under a split two
   groups of `pipeline_rows`, 512, since a step of one group runs without the
   overlap of a second; the better priority first, then
   starved ones, then the shortest, with the newest token of every generating sequence riding
   along without drafts, so that they go on at a token a step). A round that
   verifies drafts keeps the inputs of its rows for the commit and has room
   for the rows of windows and no more, which is why prompt rows never share
   a step with draft rows. The one setting is the *floor* of generation
   (`--generation-floor`, `RuntimeConfig::generation_floor`, 20 tokens per
   second by default): while prompts wait, every generating sequence keeps
   that rate and the prompts get the rest of the time. The share measures what
   rounds give a generating sequence (`r` tokens per millisecond of rounds, the
   mean over the sequences, each round weighing 0.95 of the one after it), and
   a step of prompt that takes `t` ms leaves `(f t - riding) / (r - f)` ms of
   rounds owed (`riding`: the token each sequence got in the step), never more
   than `t`: generation gets at most half of the time, which is what a floor
   the rounds cannot reach (a slow model, many sequences) and a share that has
   measured no round yet leave. While a request waits in line for a place,
   generation gets half of the time whatever the floor: the generating requests
   give places back, and a prompt's request does not end before its reply (at
   the floor a request waiting beside a generation and a 32k prompt got its
   first token after 68 to 73 s, at half after 39 to 41). A round pays back what it takes; a round that
   overshoots is a credit, so the share holds over time whatever the size of a
   round. A prompt that a piece completes does not wait for the debt of a long
   one, as long as the debt is no more than one whole piece leaves, so the
   first token of a short prompt comes after the step in flight and one step of
   its own; a burst of short prompts is read in one piece. A prompt read alone
   (nobody generates, no other prompt has started) takes the widest step,
   `--batch-size` rows, which for a mixture of experts may run on VRAM its
   expert slots lend (see "Large prompt steps on lent VRAM" under
   Qwen3.8-Flash-Next); as soon as another request waits, steps of prompt are
   pieces in the model's own memory (`Model::own_rows`, `--ubatch-size`),
   halved until the measured price of the piece is at most
   `RuntimeConfig::piece_ms` (500 ms). A model that pauses its wide steps
   (`Model::pauses_wide_steps`: Flash-Next reading a prompt layer by layer)
   reads a long prompt beside generating sequences in such a step as well: the
   prompt's rows alone, paused once the time of a piece has passed, the rounds
   running between its pieces, so that every expert is read once for the step's
   rows and not once a piece. Such a step is at most half of `Model::max_rows`
   (the most rows the expert slots can lend for): the rounds between its pieces
   do without the slots it borrows, and a loan of all of them stalled them.
   (The scheme is TensorFold's lanes, whose "a long prompt's
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
   is what its tests compare with; see [Sampling](sampling.md). Under a tensor
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

With the floor of generation and pieces of two groups (2026-11-23, the 27B at
the owner's production settings, main b8bb506, before and after alternating,
load 2 to 33; the fork's numbers from the parallel comparison in `OPEN.md`):

| Beside a generating request | before | now | fork |
| --- | --- | --- | --- |
| an 8k prompt, first token | 21.0-21.5 s | 10.6-12.9 s | 15.2-15.5 s |
| a 32k prompt, first token | 124.7-127.5 s | 82-100 s (71-73 s with pieces of up to 1 s) | 59.8-59.9 s |
| the generating request meanwhile | 26-35 tokens/s | 17-20 tokens/s | 21-26 tokens/s |
| its longest gap | 0.38-0.56 s | 0.49-0.94 s (0.56-1.04 s) | 0.51-0.66 s |
| on main acb2587 (the new prompt attention, two rounds), 8k / 32k | 18.6-18.8 / 84.1-87.2 s | 9.5-15.2 / 57.3-59.9 s | |

The prompt and the generation used 0.99 of their two rates alone (the fork 0.96
to 0.97; its own measurement of concurrent prefill: 0.40 of the decode rate and
0.61 of the prompt rate), so running a piece and a round at once on two
streams, as the fork's concurrent prefill does, has at most a few per cent to
give over taking turns: not built. Steps of 512 rows (two groups) cost 0.96,
1.36, 1.77 and 2.16 ms a row at 0, 8, 16 and 24k cells, one group of 256 1.55,
1.74 and 2.21: with `piece_ms` at 500 ms the pieces fall to one group from about
8k cells on that attention, which is where the pieces of up to 1 s gain.

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
Beside the real models, a synthetic one drives the real kernels for the
runtime's own tests and bench: 64 layers
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
once, and a part whose step failed tells its model too (`DeviceModel::step_failed`,
also for a panic the model did not see), so that Flash-Next's other part gives
its waits up at once. A host thread of a step never waits in the driver for
work that may wait for another device: Flash-Next's pass waited for the inbox of
its next micro-batch so, with both devices on every layer, and after the other
device failed it stood there for good, its device at 100 % (the hang after a
failed test, #67); it now waits where a failed part or a step that stands still
ends the wait (`Flow::wait_for`). A thread that has not stopped ten seconds
after it was told to is left behind and the shutdown ends. `ares_runtime::testing::Faulty` makes a model
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
| TensorRT-LLM, SGLang, vLLM (suffix decoding) | One n-gram drafter on a suffix automaton over a corpus shared by all requests | `ares-sched`, used by the conductor |
| MiniJinja | The interpreter of chat templates | `ares-tokenizer` |
| Flide | The System One API of `/v1/systemone` (request, answers, limits, errors, several requests in one body), how a chat model is asked a question and read at its labels (`next_token`, style `vllm`), exact state forking of a request's questions | `ares-api` (`decision.rs`, `answer.rs`), `ares-server` (`systemone.rs`), the conductor's decisions (through the prefix cache) |
| llguidance | Grammar-constrained output: the grammar engine on the host gives a mask per row, which the sampler applies on the device (the mask and its hook are built, see [Sampling](sampling.md)) | Done (`ares-grammar`): GBNF and JSON schema on the sampler's mask, after the reasoning of a model that thinks, and a forbidden draft is cut; Lark grammars are in the engine for the tool calls and have no request field |
| AirLLM | Nothing beyond the expert tiers: reading whole layers per pass is bounded by disk throughput | - |
| Run My AI, ik_llama.cpp, Fiddler, ktransformers, BELLS, AI-DER, FATE, ProMoE | Direct reads with demand before prefetch; prefill layer by layer in waves of experts; the RAM tier as a cache with admission and protection; the order of work inside a MoE layer; slots per layer by need; fused gate and up of an expert | `ares-io` (reads done); the rest planned, see [Weights](weights.md) |

Notes on each source are under [reference/](../studies/). The reading of the
other engines' code, with a ranked list of what to take and a list of where
Ares still copies llama.cpp without a reason, is
[studies/serving-engines-code-survey.md](../studies/serving-engines-code-survey.md).

## Status and what is open

What the engine can do today is listed in [`user/features.md`](../../user/features.md), what is still
open in [`OPEN.md`](../../OPEN.md); both are brought up to date at every merge into `main`.
This document describes how the parts work and why. Numbers in it carry the date they
were measured; the current ones are in `docs/user/features.md`.

## Open questions

- **Indexer cut-off.** Strata's two engines differ in which cells the sparse
  attention keeps at the cut-off. This must be settled against the real
  Qwen4.
- **Pascal and Turing cards (GTX 10, GTX 16, RTX 20).** Settled: CUDA runs
  only on the tested classes (see [Devices](devices.md)); these and every other
  untested GPU go through the Vulkan backend.
