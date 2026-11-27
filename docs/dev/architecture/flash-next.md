# Qwen3.8-Flash-Next through the runtime

`ares --model <first part of the file>` serves Qwen3.8-Flash-Next like the
27B: the same endpoints, timings and metrics, cancellation, a request that
gives way and comes back, several sequences in a step. The model is
`ares_model::qwen4exp::device::Qwen4Exp`, its parts `Qwen4ExpDevice`.

There is one opening path and one way to drive the parts. Opening goes through
the stages of `device/open.rs` (a part with its token rows, the room each device
has for expert slots, the store's plan with transit, the feed with the pruning
mask and the owners of the experts, the peers and the owner mask): the runtime's
device threads each run them for their own part and meet twice in between
(`runtime/parts.rs`: the plan needs every device's room, the store every
device's slots), the roles' trial runs the first and the last of them on parts
without a store. A step is driven through `Flow` (`device/flow.rs`): the store
behind a lock no thread waits for, the n-gram rows, the lead of at most six
layers over the route records, the rounds of a wide step's missed experts, the
waits. `Session` steps the model by hand for the tests and the tools through the
same stages and the same flow, from one host thread for every part; it stays a
face of its own, not one over the executor as the 27B's session is, because its
probe stops both devices at every layer and reads both parts (the runtime's
threads would need a barrier for that inside the production step) and the tests
read the logits, which the runtime samples on the device. What only it does: the
tier in RAM locked whole, the experts of two devices divided by a recorded
routing (`SessionOptions::routing`, for measurements), sequences that own their
pages. The model's options are `ModelOptions` (`device/options.rs`), what the
server builds.

**It drafts with its MTP block, from the model file or a file of the block.**
The model's draft head is a block of its own (section 4 of the
[specification](../studies/qwen4exp-architecture.md)): one layer of the attention
kind with its own mixture of 512 experts, two projections that make its input
from the trunk's residual after the last layer and the embedding of the token
behind it, and its own final mixer in front of the model's output head. The
IQ2_XS file carries none; `--drafters FILE` names the file Strata packs it in
(architecture `qwen4exp-mtp`, 889 MB for Flash-Next: 175 MiB of dense weights,
the experts in Q2_0), and `--drafters mtp` reads the block from a model file
that carries its `mtp.*` tensors. The server refuses a file that is no block of
this model before a device is touched. `--draft-vocab` is the head view of
every drafter of the model (`blocks/head.rs`): the output head's rows of its
tokens come first, the model's logits in that order. Without a block the model
does not speculate: the n-gram drafter takes part as soon as a model drafter
does and never alone (see [Speculation](speculation.md)).

- *Where it runs.* On the device that holds the output head, in every placement
  (one card; the last card of a split by layers; the card that mixes when both
  work on every layer), as the layer of the shared MTP driver (`blocks/mtp.rs`;
  the layer is `device/mtp.rs` and a part of one layer, `device/part/mtp.rs`)
  that reads the model's output head where it is. Its keys, values and indexer
  keys are pages of its own, addressed through the model's page tables.
- *What it does in a step* is the driver's (see "The MTP driver" under
  [Speculation](speculation.md)), with a lag of one: its cell `i` pairs the trunk's
  residual of cell `i` with the token of cell `i + 1`. In front of every step of
  the model it (1) follows the rows of the step before that count, eight to a
  launch; (2) writes the newest cell of every sequence; (3) for the windows that
  take its drafts goes on from its own residual and its own pick, link after
  link. The keys of the indexer a sequence carries are saved behind the first
  link and put back behind the last.
- *The chain stays on the device.* A link's pick is made by the sampler on the
  device and written into the next link's record, whose embedding the device
  gathers from the rows of the embedding table it keeps (the head view's
  tokens: the whole table without a draft vocabulary, 337 MiB; the vocabulary's
  rows with one); the newest tokens' rows come staged from the mapped file. The
  part of the block sends its steps' records and plans from a ring of buffers,
  so the host enqueues a chain whole and reads the drafts once, when the step
  that only drafts has ended (until 2026-11-17 the host waited for every link's
  pick, because it decoded the next link's embedding). The chains run in that
  step (steps 2 and 3 above, for the sequences that draft; the other
  device threads have nothing to do in it), and the round that verifies them
  is the model's ordinary step of up to eight rows with the drafts as tokens;
  the block does not write again the newest cells its chains wrote. The step
  that only drafts is no step of the expert store (its routing is the block's,
  whose experts are resident, so the store reads none of its route records),
  and when it has ended the host asks for the rows of the n-gram table that the
  round's rows will need (the newest token and the drafts made), so that they
  are read while the conductor decides. The rows
  that are rejected leave no trace in the recurrent layers, the n-gram
  history and the indexer's keys (`Part::enqueue_commit`, which existed), and
  their keys and values are overwritten.
- *Its experts are resident in VRAM*, as layer 0 of the expert store in front of
  the model's layers, read from the block's file when the store is built: all
  512 of them, in a pool of their own outside the tiers (`LayerSpec::resident`),
  675 MiB on the card with the head that the tiers do not get. Until 2026-11-16
  the block had the slots of a layer (111 in VRAM and 73 in RAM of its 512
  experts with both cards on every layer, 260 and 146 with a range of layers
  each), and a link took 5.4 to 8.1 ms while most of them were in no slot.
- *The policy is the conductor's.* The price of a step by its rows and of a
  chain by its links is
  measured when the engine starts, and those steps find no expert in a slot: with
  the block loaded they gave 35.5, 49.4, 93.0, 77.0, 97.5, 133.5, 111.3 and
  122.5 ms for one to eight rows and 15.6, 3.3, 4.7, 6.3, 7.7, 9.0 and 10.1 ms
  for one to seven links (2026-11-15, both cards on every layer, load average 20;
  a step of one row takes 22 ms once the tiers hold a text's experts). Until
  2026-11-15 a price was corrected only by a step of its size that ran, so the
  one-row price came down to what it is and the others stayed where they were
  measured, and a window of two rows at 49 ms and a link never paid against a
  row at 22 ms; since then a size that has not run for 256 steps is tried
  again at the price of the row that runs (see "Prices that stay true").
  Started
  without that measurement (so that the table fills from the steps that run) the
  server did draft: the block's links were accepted 115 times of 176, 51 of 73
  and 21 of 29 for the first three on German and English prose and Rust (the
  fourth: 7 of 18), a link took 7.5 to 8.1 ms while most of the block's experts
  were in no slot, and the rows that verify waited for experts of their own (135
  outside a slot and 167 ms waited a step with windows of eight rows, where a
  step of one row misses 6 to 37). It was not faster for it, the measured steps
  no longer filled the tiers with what a first text finds there, and a first
  chain that ran while experts were missing left a price that kept every later
  one from being chosen. So the server measures at the start as for any model,
  and the windows' prices are taken again when the tiers are filled.
- *The window* is four rows unless `--window` says otherwise (2 to 8): a row of
  this model routes to experts of its own, so a rejected draft costs what a token
  costs.

*Drafts change no token.* On made-up models, in every placement, a sequence
writes with the block what it writes without, token for token: greedy, with
drafts of the n-gram drafter that are accepted, with `"draft": false`, and sampled with
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
  still for 30 s ends with an error that says what it waited for (`Flow::check`:
  30 s since its wait began and since anything of the steps last moved, a layer
  enqueued or followed, an upload or a read the store finished, a pass's rows
  handed on, rows of the n-gram table read; the lend waits hold the store and
  count from their start).
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
  first step, and the rounds owed between pieces go at a token each.
  Where the model reads a prompt layer by layer (`Model::pauses_wide_steps`),
  a long prompt beside generating sequences is read as such a pass instead of
  in pieces: the prompt's rows alone, at most half as many as the slots can
  lend for, paused between two micro-batches once `RuntimeConfig::piece_ms`
  has passed, the rounds owed to the generating sequences running between its
  pieces (the pass keeps its loan of VRAM meanwhile, so the rounds find fewer
  expert slots: a loan leaves a layer a row's picks and the free slots,
  `ExpertStore::scratch_floor`). A pass of all the rows the slots could lend
  for (21k) stalled the rounds between its pieces in 3 runs of 4 ("a step
  stood still ... the store waits for 7 to 9 experts"), half as wide in none
  of 4, the prompt of 21k rows in 38.5 to 40 s beside the generating request. Measured on the test machine (one RTX 4060 Ti, IQ2_XS, `--ram 24G`,
  2026-11-23, load below 8, two rounds each): an 8k prompt beside a generating
  request 61.8 and 62.1 s before (alone 10.2 s; pieces that read the experts
  again each), 16.6 and 16.8 s with the pass (alone 10.4 and 9.8 s), the
  generation keeping 19.4 and 19.5 tokens/s (the floor of 20; 26 before, of
  40 to 53 alone), its longest gap 1.0 to 1.2 s; a 16k prompt 113 s before,
  29 to 31 s with it. The pass's pieces took 510 to 560 ms, the rounds between
  them 330 to 500 ms; the rounds' rate fell from 0.059 to 0.040 tokens a
  millisecond while the pass held its loan.
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

**What the plan decides.** Where the layers go (`--split tensor` or `layer`, by
default both cards on every layer when each can write the other's memory), the
context, and the RAM. The dense parts take 2.9 GiB on the first card and 0.4
on the second; what the cards have left is slots for experts. The context
costs slots; without `--context` it is the model's own all the same (the
owner's rule of 2026-11-22, for both models), 262 144 cells, and what of it
does not fit in VRAM overflows into RAM (see "K and V beyond VRAM" below): the
indexer's keys of every page stay in VRAM, the K and V of the first pages up
to a thirty-second of the room for slots (`plan::near_pages`; what the context
in VRAM cost before 2026-11-22, 17 408 to 18 432 tokens for two sequences on
this machine), those of the rest in page-locked RAM as sequences grow into
them. A context named with `--context` (for all sequences together, so it may
be larger than the model's own: four sequences of 131 072 cells are
`--context 524288`; a request still holds at most the model's own) is placed
the same way (the owner, 2026-11-24). The plan says what was wanted and how
much of it is in VRAM (`plan::ContextPlan::overflowing`). Options that mean nothing for the
model are errors that name them: `--shares`, `--draft-vocab`, more sequences than the eight rows of a
generating step (`--parallel`), `--drafters mtp` (the model file carries no MTP block;
`--drafters FILE` names the block's file), and `--window` without a drafter or beyond
the eight rows of a step that verifies.

**K and V beyond VRAM** (`blocks/far.rs`, 2026-11-22). Flash-Next's attention
reads what its indexer picks: past 2051 cells a row reads 512 blocks of 4
cells and its tail, 2.2 MB of K and V a layer with `q8`, whatever the length of
the context. So the K and V of a long context can lie in RAM and cost a step
only the cells picked from there, where every page of K and V in VRAM is
expert slots fewer: the model's own context took 3.56 GiB, a fifth of the
slots, and decode was 11 to 22 % slower for it ("What the second card is
for"). One mechanism serves both the context beyond VRAM and the VRAM it gave
the tier: the K and V of a context never take more VRAM than the plan's share,
so nothing idle in VRAM needs lending.

- *Where a page is.* The K/V group of the attention layers keeps the converted
  entries of pages `0..near` in the pool on the device as before, those of the
  pages from `near` on in a **far pool** in host memory: one range of address
  space per device part, an array of slots per layer, page `p` in slot
  `p - near`. The device's table names such a page with bit 30
  (`ares_cuda::FAR_ENTRY`) and that slot; the conductor's entries stay page
  numbers, and the part translates them (`kv::Group::device_entry`). The
  attention, append and convert kernels take the far pool of a layer next to
  its converted pool and read the same bytes from either (`conv_entry` in
  `attn_layout.cuh`): the arithmetic is the same, so is the output. The
  indexer's pooled keys of every page stay in VRAM (the block scores read all
  of them), and so do the exact entries of the tile formats and the MTP
  block's K and V (one layer, `q4`).
- *Which pages are far.* The pool hands out its lowest free page first
  (`PagePool`), so a short sequence lies in VRAM and a long one overflows from
  its `near`-th page on. Its newest cells are then written over the link
  (12 x 1088 bytes a decode row: nothing).
- *Memory.* The far pool is address space (`MAP_NORESERVE`), of which pieces
  of about 64 MiB over all layers are locked and mapped for the device
  (`cuMemHostRegister` with the device map; a device must read registered
  memory at its host address, which `PinnedRam::context_region` checks) as the
  sequences reach them, and given back when no page in use lies in a piece any more,
  keeping one piece spare. They are booked in `--ram` as required memory ("K/V
  of the context beyond VRAM"), above the tier of experts: a running sequence
  cannot do without its cells. Far pages
  that only the prompt cache holds rank as prompt cache, the lowest (the
  owner's answer of 2026-11-22; see "Host memory"): they leave the devices first
  when it must give memory back, into RAM where it has room and otherwise
  nowhere, and a piece is unlocked once no page in use lies in it. Such a page
  never waits for the drive: a long request leaves hundreds of them at once
  (490 after a prompt of 80 507 tokens), and with a wait for the writer to free
  a buffer of RAM for each (up to a quarter of a second) the conductor ran no
  step and answered no command for half a minute while the next request
  waited (seen in the merge check of 2026-11-23; `cache/tests.rs`
  `far_pages_that_find_no_room_in_ram_are_dropped_without_waiting_for_the_drive`).
  The flow
  keeps the books of which pages the tables name (and which a transfer
  brings in), and the conductor locks the pieces the step reaches before it
  hands the step over (`Model::before_step`, `Flow::before_step`,
  `Flow::lock_far`): page-locked memory does not change inside a step ("When
  page-locked memory changes" in [`host-memory.md`](host-memory.md)); a step
  with far pools that was not prepared so fails. When the
  limit has no room the tier's allowance already leaves it
  (`PinnedRam::set_context_target`) and the tier gives pieces up then, with no
  step running.
- *Prompts.* Every row of a prompt reads the cells of its own pick, so a wide
  step that read the far pool in place would cross the link once per row. A
  step on lent VRAM (a prompt read alone, "Large prompt steps on lent VRAM")
  copies a layer's far entries that its sequences reach into the loan behind
  the layer's append (`Scratch::far_stage`, up to the far pool's size, which
  the slots lend with the step's scratch) and reads them there; each byte
  crosses the link once per layer and step. A prompt read layer by layer
  (a pass, "Flash-Next's prompts layer by layer") keeps the copy in its loan,
  first in the largest piece, sized for the far slots the pages in use reach
  (`PassLayout::new`; the most rows of a pass are those that fit beside a copy
  of every far slot): each micro-batch brings it up to date behind its append,
  the first of a layer with every far slot its rows reach, each later one from
  the first page its own rows wrote on (`Part::pass_stage`), so each byte
  crosses the link about once per layer and pass. Measured on the two local
  cards (2026-11-23, IQ2_XS, `--ram 8G`, the model's own context, 146 pages
  near, a prompt of 80 507 tokens in passes of up to 32 768 rows, load 4 to
  8): 97.0 s with the copy, 202.4 s reading the far pool in place. Steps of the
  parts' own scratch (pieces of prompt while another request waits, decode)
  read in place.
- *Scratch.* With the model's own context a row's block scores take 4 bytes
  per block of 262 144 cells, 256 KiB; the scratch held them for every row of
  the widest step, so a step on lent VRAM cost a quarter of a GiB more per
  thousand rows and the slots lent for fewer rows (2048 against 4096 with a
  context of 66 560 on the two development cards). A step scores and selects
  its rows 256 at a time (`SCORE_ROWS`) into one buffer since.
- *The prefix cache.* Pages in the pool are pages, near or far: a copy on
  write, a page that leaves for RAM or the drive and one that comes back move
  their far entries with the driver's copy between any two addresses
  (`Stream::copy_any`). The RAM level copies a far page; adopting its memory
  without the copy is open.

**The RAM.** The tier of experts is a user of the one RAM limit (`--ram`,
`PinnedRam`): elastic, with a priority above the checkpoints, so the limit
minus what the engine needs to run (the step buffers and the parts'
page-locked buffers, 19 MiB) is the tier, as far as the experts that have no
VRAM slot can use it. The number of RAM slots is what that gives: with the
default limit (half of the RAM the machine has available when the server
starts) and 22 GiB available, 10.9 GiB, 7 970 experts. Before this change the
tier was a third of the available RAM, at most 8 GiB, whatever the RAM limit
said, and the page-locked buffers of the parts were outside the books.

**Files in any type.** Some matrices of the model are multiplied with values that are
not quantized: the stream mixers' (down, up and inject of every half block, and the
mixer in front of the head) and the n-gram block's key and value projections, which the
reference multiplies in f32. Files store them in any of the 28 weight types: Swift's
IQ2_XS in BF16, Unsloth's UD-Q4_K_XL in Q8_0 (the 194 mixer matrices of 10240 by 320 are
644 MiB there, 2425 MiB as F32). They stay on the card as the file stores them, and every
product with them decodes each weight as ggml does and sums in f32
(`kernels/weight_value.cuh` has a decoder per type):

- *Steps of up to eight rows.* The fused read of a mixer (`ares_q4_read_down`,
  `ares_q4_read_up`) decodes in the thread that multiplies: the float types as before, nine
  quantized types with their decoder inlined (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, Q4_K,
  Q5_K, Q6_K: `kernels/q4_reads.cu`, which takes nvcc some 100 seconds), every other type
  through a decoder chosen by the type at run time (inlined for all 25 types the module took
  five minutes to compile). `inject` has few rows and always decodes by its type. The
  separate kernels and the n-gram block take `Kernels::matvec_exact`, the float decode
  kernel with a decoder per weight: its sums run in the order of the float kernel, so a
  matrix gives the bits its decoded copy in F32 gives, except for the last bit of a weight
  of a type with an offset (`d * q + m`, which the device may fuse; Q4_1, Q5_1, Q2_K, Q4_K,
  Q5_K).
- *Wide steps.* The tile kernel of the matrix's type with the rows converted to f16, as the
  float matrices of a wide step have it (`Part::wide_float`).
- *The router* stays F32 as the files store it; a router in another type is decoded to F32
  once at load (the draft block's file has BF16), four bytes a weight.
- *The n-gram table* may be of any type whose block divides its rows of 160 values (the
  floats and the 32-value types): the device decodes the fetched rows value by value.

**A layer with fewer slots in VRAM than a step picks.** A decode step of eight rows can
pick 80 experts of a layer, and the store used to give every layer that many slots in
VRAM and eight free, so that every miss of a step could be uploaded without evicting
another pick. With the experts of UD-Q4_K_XL (3.0 MiB a slot in VRAM) that is 12.3 GiB
per card on one card or with both cards on every layer (each card holds a share of all 48
layers), and the plan refused. Where the CPU computes misses (every `--serve` but `stream`), a card
whose bytes do not hold that many now gives each of its layers as many slots as they hold
(at least a row's ten picks and the eight free), and the RAM tier keeps the rest of a step
for the CPU: each layer's floor in RAM is the step's picks and the free slots less its
VRAM slots, so the two tiers of a layer together hold a step. During a step an upload
takes only a slot whose expert the step did not pick (`SlotSet::room`), the misses beyond
those go to the CPU whatever the cost model said, and the VRAM tier never evicts a pick
(`SlotSet::keep_picks`; it used to, as a last resort that the old floor never reached).
The refusals name the limit: the card, the slots its bytes hold per layer and what a
layer needs, or the RAM the floors need (then `--ram`).

**The prefix cache** (`crates/model/src/qwen4exp/device/resume.rs`, on the
shared slot block). Flash-Next takes part as the 27B does, with the levels in
RAM and on SSD; what differs is what a checkpoint and a page hold:

- **A checkpoint** of a device is the slot of its part (the convolution
  windows and delta-rule matrices of its recurrent layers, the carried keys of
  its indexers, the n-gram history); on the device of the MTP block, the
  block's slot (its indexer's carried keys) and its carry (the residual of the
  sequence's last row that counts, which the block's next cell is paired
  with); and on the device that holds the n-gram layer, what the n-gram feed
  knows of the sequence: its position and its last two tokens, from which the
  rows of the table that its next tokens need are found. That part is the
  host's: it is written into the place when the save is queued and read from
  it when the restore is, so it travels with the place to RAM and to the drive.
- **A page** is the K/V entries and the pooled keys of the part's attention
  layers and of the MTP block (whose pools are its own buffer, with the
  same page numbers): a page copy copies all of it, a page goes to the host
  and back with all of it, each group's entries in the form they have.
- **A replay** writes neither K and V nor pooled keys: the K/V append skips its
  rows, and the indexer kernel leaves out the blocks of a segment with the
  record's replay flag (bit 4, the runtime record's; `StepRecord::mark_replay`
  for the records the part builds itself). Its rows advance the recurrent
  state, the indexers' carried keys, the n-gram history and the feed as a
  prompt's do. The MTP block's cells of the replayed rows are cached too (the
  sequence that computed them had the same tokens up to the end of the
  replay), so the block follows a replay with its carry alone.
- **The MTP block one token behind.** The block's cell `i` pairs the
  residual of cell `i` with the token of cell `i + 1`, so the last cell of a
  page is written one step after the page is full, and only when the sequence
  goes on. A sequence that stops at the end of a page leaves the block's last
  cell of it unwritten, and the block settles that page all the same (else it
  would keep an exact slot of `q4` for as long as the page is cached); a
  sequence that resumes at a page boundary writes that cell into the cached
  page while its tile is exact, and reads it without writing once the tile is
  converted (a block row whose tile is converted reads and writes nothing).
  Only drafts can differ by it; the model's own cells are never written behind
  its sequence's end.
- **A DFlash drafter after a replay.** The drafter keeps nothing in a
  checkpoint or a page: its ring (the keys and values of the fused taps of a
  slot's last positions) is its own and is not saved. A replay runs the
  model's layers, so its rows have their taps (the mean of the four streams
  before each target layer), and the drafter injects every row of a step into
  the slot's ring, replay rows included; it reads no segment flag. A segment
  that does not continue the cells the ring holds (a new sequence, a sequence
  resumed from the cache) starts the ring again at its first row
  (`Drafter::begin_step`). So after a hit the drafter drafts from the replayed
  rows and those after them, which are right; the cached rows before the
  checkpoint, which nothing replays, are not in its context until the ring has
  filled again (keeping the ring in the checkpoints is open). Only drafts
  depend on it.
- **The order** in a record: the commit of the step before (with the n-gram
  feed's settling of the windows), the MTP block's following of that step,
  the freed pages and the tables, then the saves, the page copies and the
  restores, then the chains and the rows; a record that only drafts does its
  copies too.
- **The split of two devices on every layer** is measured at each load (the
  read times of a stream mixer decide the mixer and its share of the
  recurrent heads), and the state of a sequence is laid out by it, so the
  fingerprint of the state on SSD covers it. With `--cache` the server keeps
  the split it measured for a model on its devices next to the stores
  (`<model>.split`) and places the model with it again; without that a
  restart measured another split and found nothing on the drive. Within one
  load every placement takes the split the first one measured.
- **Memory.** A checkpoint of the IQ2_XS file is about 112 MiB over the two
  cards in `f32` (36 recurrent layers of 48 value heads of 128 x 128; the
  MTP block adds its carry of 40 KiB and its indexer's 1.8 KiB). The places
  are the prompt cache's, the lowest user of the engine's page-locked RAM:
  the tier of experts has the allowance first, and the cache in RAM holds what
  the tier leaves. With a full tier the cache has nothing beyond its floor:
  one checkpoint place, booked before the tier (the server's plan counts it
  among what the engine needs, and refuses a limit too small for it), which
  holds the last checkpoint of a conversation, so the next turn resumes from it
  and replays only the rows after it; with `--cache` also 16 pages that state
  passes through on its way to the drive and back. (Before the owner's
  decision of 2026-11-16 the floor was there only with `--cache`, and without
  it a hit replayed every cached row.)
- **Measured** 2026-11-16 with the server (`real_cache.rs`): Swift IQ2_XS on the
  RTX 4070 and the RTX 3060, both on every layer, `--ram 8G --context 20480
  --parallel 2 --cache <dir>`, a conversation of 660 facts and two questions
  (9851 prompt tokens in its second turn), time to the first token of the
  second turn; two runs, loads 6 to 10 and 5 to 7:

  | Second turn | First token after | From the cache |
  | --- | --- | --- |
  | Computed from nothing (a server that never saw it) | 25.1 s; 31.8 s | 0 |
  | From the cards | 0.65 s; 0.68 s | 9827 tokens |
  | After two other conversations took the pool | 1.15 s; 1.18 s | 9728 tokens, 68 pages from SSD, none from RAM |
  | After a restart of the server | 2.28 s | 9728 tokens from the drive |

  Every reply was the one of the server that never saw the conversation. RAM
  held what the tier left the cache, its floor: one checkpoint place (118 MB)
  and 1.9 MB of pages, so the pages that left the cards went on to the drive
  and came back from there. With `--ram` large enough for the tier and the
  cache, they come back from RAM (`qwen4exp_prefix.rs` shows it on a made-up
  model).
