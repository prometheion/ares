# Devices

`ares-cuda` binds the driver API (`libcuda.so.1`) at run time, so the CUDA
toolkit is needed to build the kernels and not to run. Kernels are compiled by
`build.rs` into fat binaries embedded in the binary, one per module, with
machine code for the GPU classes the engine was tested and tuned on and for
nothing else.

**GPU classes.** CUDA runs only on tested classes (the owner, 2026-11-23):
today Ampere (8.6, RTX 30) and Ada (8.9, RTX 40); Blackwell (12.0, RTX 50)
joins after the RTX 5060 Ti's tests. Every other GPU is to run through the
Vulkan backend. A build has machine code for its classes only and no PTX, so
no other GPU compiles the kernels on its first load and runs code nobody ran.
`ARES_CUDA_ARCHS` (for example `86,89,120`) builds other classes on purpose:
each named class is built, the build warns about every untested one, and a
class below 8.0 (the kernels use the `mma` shapes, `cp.async` and `ldmatrix`
of Ampere) or one the toolkit does not list is a build error. The classes a
build has are `ares_cuda::BUILT_CLASSES`, which `build.rs` writes.

A device is run only if its compute capability is one of those exactly:
`Device::open` refuses any other (`DeviceInfo::check_class`), also an 8.7
Jetson Orin, which could run the 8.6 machine code but was never tested. The
refusal names the device, its capability and the classes of the build, and
says that other GPUs are to run through the Vulkan backend. The server's
device choice (`--devices`, or every device the driver reports) skips such a
device with a log line while another remains and refuses with the refusal
when none does. Since every built class is Ampere or newer, every device that
opens has the tensor-core kernels (prefill, attention, many-row experts, the
image tower); there is no path for a device without them.

- `devices()` and `peer_access()` describe the hardware without opening it.
- `Device` holds the primary context; `DeviceBuffer`, `PinnedBuffer` and
  `HostRegistration` (page-locking mapped model files in place) own memory.
- `Stream` and `Event` order work on and between devices.
- `Kernels` gives one typed launch per kernel.

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
to 2.7 tokens: the MTP block's chain takes about 4 ms of it (1.46 ms per
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
types). Where they come from (`blocks/rows.rs`):

- **The host.** The table lies in page-locked host memory of the device that
  embeds (the first one), booked as required memory of the engine's one RAM
  limit (`Model::pinned_ram`). A step copies the rows of its tokens into a
  page-locked staging buffer, one row per row of the step, and uploads them
  on the step's stream in one copy right after the record; no event, no wait.
  A prompt step of 512 rows uploads 1.1 MiB of the UD-Q4_K_S file's table.
- **The device itself, for the draft chain.** The model's MTP block
  picks tokens inside the step, on the device, and embeds them (the chain's
  next link, the verify window, the block following the model) before the
  host knows them. So the device that holds the MTP block keeps the rows of
  every token the chain can pick, which are the rows of the draft head: with
  a draft vocabulary those tokens in ascending order (the same list that
  orders the head, so the gather finds a row by binary search and nothing
  extra is stored), without one the first rows of the head on the device (the
  whole table when the device holds the whole head, and then the host stages
  nothing). A device without an MTP block keeps no row. The newest token of
  a chain, which the host knows, is staged with the MTP block's records.
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
device already (the MTP block without a vocabulary), so it would be a
decision of the placement.

Measured on 2026-11-12 with `qwen35_run` on an RTX 4070 and an RTX 3060, a
context of 32 768 tokens, K and V in `q5`, against the commit before
(b245771, the table on the first device). The columns are the VRAM the
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

With the model's drafts (three, the fork's draft vocabulary, 300 tokens)
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

### What the second card is for

The two cards of the development machine differ in speed (479 against 341 GB/s),
in their link (PCIe 4.0 x16 against 3.0 x4, 3.3 GB/s) and not in memory (12 GiB
each). How the slower one is used best was measured on 2026-11-22 for both
models (the owner's question of that day; the roles of 2026-11-17 in
[`OPEN.md`](../../OPEN.md)).

**The fork's attention split.** The owner's llama.cpp fork has `--attn-split`
(`-as`, with `--split-mode tensor` only; commits 553a934 to ff1a44e): every
tensor whose split is counted in the heads of `blk.N.attn_output.weight`
follows it instead of `--tensor-split`. Those are the four projections of the
full-attention layers (query with its gate, key, value, output) and the K/V
cache of those layers (`cache_k_lN`, `cache_v_lN`); the recurrent layers (whose
split is counted against `ssm_out`), the feed-forward parts and the head keep
`--tensor-split`. Each card holds K and V of its heads for every cell, so the
context is shared by the heads: with `-as 1,3` the 4070 holds a quarter of K/V
and the 3060 three quarters, and the context ends when the first card is full.
The rounding to whole heads is carried from layer to layer (`-as 0.3,0.7` gives
20 of 68 heads). The production preset is `-ts 0.625,0.375 -as 1,3`: the faster
card holds more weights, the slower one more context, and two sequences of
131 072 cells fit.

**The same in Ares** (`Placement::tensor_with_attention`,
`Spread::TensorWithAttention`): the KV heads of every attention block follow
shares of their own, with their query heads, their slices of the projections,
their K and V and their attention work; the recurrent blocks, the feed-forward
parts and the head follow the shares. The cut of a block is at whole heads and
its rounding is carried into the next attention block, so over all of them a
device holds its share within one head. A device may hold no heads at all
(its blocks are then empty mixers, which the parts had already).

**The partial states of the attention plans** (`PlanConfig::state_slots_bound`). A
row whose cells are cut into several pieces takes a state slot per piece, and a
device reserves the slots its plans can take. The bound kept since the first
version (twenty tiles of rows times the budget of pieces, and the rows alone)
was passed in two ways: with a tile mean (`q7` to `q4`) a prompt row past the
first tiles reads the exact first tile, converted tiles and exact newest ones,
three pieces at least, since a piece is of one kind; and a row of a sequence
that reads its prefix together with another one has the pieces of both its
tiles. The owner's settings (`q5`) failed at the first long prompt: a step of 256
rows at cell 512 needed 768 slots on the RTX 3060, 680 were reserved. The bound
now follows the planner (a tile's rows per piece of the budget, two tiles per
sequence, two slots a row, five with a tile mean, and the rows alone with two
more each with one), and a test checks it against random steps of prompts,
shared prefixes and rows alone (`plan.rs`). It costs the devices tens of MiB
of scratch (the 3060 of the owner's settings 62 MiB where it reserved 12); the
server's estimate counts it (`attention_state_bytes`), and a device that is
over gives up at least a whole head. `tests/real_devices.rs` loads the 27B three
ways and compares the planned pages and the text: two sequences, K and V in
`q8`, no drafters, 890 pages with the heads following the weights, 1770 with a
quarter of them on the 4070, and 1946 (248 832 tokens: the model's 262 144 do
not fit at `q8`, so the heads go by room) by the context; the same German
answer in all three (2026-11-22).

**Flash-Next: the second card as an expert tier.** With both cards on every
layer, the card that mixes (`TensorChoice::mixer`) holds the residual streams,
every stream mixer, the routers, the attention layers with all their keys and
values, and the head; the worker holds a share of the recurrent heads
(`core_share`), maybe the shared expert, and the experts it owns, and computes
them. With `core_share` 1 and the shared expert on the mixer the worker holds
experts and nothing else: the main card runs the model, the second card is one
more tier of experts between the main card's slots and RAM, and it computes
what it holds (the rows of a layer go over, the sums come back, as for any
worker). Lending the bytes instead (copying an expert from the second card to
the first when it is picked) is not built: on this machine the second card's
link (3.3 GB/s) is slower than an upload from page-locked RAM to the first
card over its own link (an expert of the IQ2_XS file in 0.1 ms against 0.4 ms
over the second card's link), so a copy from the second card would be the
slowest tier.

**The second card's role is timed at load** (`device/roles.rs`, the owner's rule
of 2026-11-22: no option). When both cards work on every layer and neither the
share nor the shared expert is given (`Spread::Tensor` with both `None`, or
`Spread::Auto`), the placement times the two roles on the cards at hand: the
worker computing a share of the recurrent heads and the shared experts (the
mixer's share the worker's read time over the sum of both, as before) and the
worker holding experts only (`core_share` 1, the shared expert on the mixer).
Each choice is placed as the load would place it (one that does not fit the
cards is refused), its parts are opened with a sequence of a few thousand cells
and every expert in a slot of zeros (slots of four times the card's L2 cache, so
the experts' kernels read from memory), and a decode step of one row (the median
of nine after three) and a wide step of the parts' own rows (up to 512, after one
of sixteen rows that warms the kernels up) run as in a model that misses nothing, the exchanges between
the cards included; then the parts are closed, before anything of the model is
placed for good. The choice whose decode step times wide step is the least is
kept: each kind of step counts by how much faster it is, where milliseconds alone
would let the decode step, some fifty times a prompt row, decide what prompts pay
for. What the trial does not see is the experts a step misses, which the two
roles hold about equally (experts only gives the worker the bytes of its dense
share for slots, and the mixer as many fewer). The trial runs in the first
placement of a load (`place`, `Session::open`); later placements of the same load
pin its choice, and with `--cache` the server keeps it next to the stores as
before. A worker that holds experts only holds no state that a draft window
changes, so it keeps nothing for windows and their commit does nothing there (it
had refused windows, which took Flash-Next down with its MTP block once the trial
chose that role; `qwen4exp_runtime` `a_worker_of_experts_only_runs_draft_windows`). The server's log says what each role took and which was kept;
`qwen4exp_run` prints it.

**Drafters on the second card** are not built. The step that only drafts runs
the chain on the card that holds the drafter while every other card waits, so
a chain on the slower card is slower by what that card is slower and gains
nothing beside it; what the drafter's memory gives the first card back is room
for context or slots, which the attention heads (27B) and the expert slots
(Flash-Next) of the second card give already. A drafter on the second card
pays only with a pipeline that drafts the next window there while the first
card verifies this one (Strata's `--pipeline-windows`), which Ares does not
have.
