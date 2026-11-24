# Prompts of Flash-Next

**Wide steps: a prompt on the many-row kernels.** A step of up to eight rows is as
described above, and a round of generation stays there: one row for each sequence, and with
an MTP block the drafts they verify, eight rows for all of them together. The piece of a prompt that the conductor sends between rounds, up to
`Model::prompt_rows` rows, which for this model is `Model::max_rows` (`step_rows`, below),
with the newest token of every generating sequence riding along, runs every layer at that
width (`device/part/wide.rs`, `device/misses.rs`):

- *Mixers, n-gram block, recurrent layers, attention* take the rows as they are: the
  kernels were written for the 27B's prompts. The dense products go through the tile
  kernels (`Kernels::product`) in the activation form the device and the type ask for:
  the int8 rows the fused kernels already leave (`mix_quant`, `gated_norm_quant`,
  `swiglu_quant`), or f16 rows (`prepare_mixed`, `gated_norm_f16`, `swiglu_f16`) when the
  device runs the f16 tensor cores at full rate or `Options::precision` is `Exact`. Each
  quantized type takes its own form (its measured path at the options' precision, as the
  27B's chooser takes per matrix; the owner's decision of 2026-11-22): where the types of a
  part's dense matrices differ (a type with no int8 tile kernel, such as the Q2_0 of three
  shared-expert downs of the IQ2_XS file, or the measured paths of a full-rate card, Q6_K,
  Q3_K and Q5_K f16 against IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS and Q8_0 int8 on the RTX 4060
  Ti), the kernels that write a product's input write both forms (`Part::wide_f16`,
  `Part::wide_int8`); until then every dense matrix took the exact f16 path there. The
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
  call: the caller serves the store between the steps of `Misses::advance`. The heat of the
  store counts an expert once a step of the part's own rows (`record_counts`), as a decode step
  does: a wider step (a prompt read alone on lent VRAM, or a pass) counts as the steps of own
  rows it stands for, at most once a row that picked the expert. (Counted once a step, a pass
  of a whole prompt weighed as one token, the tiers kept what came before it, and the next turn
  of the conversation, which replays the prompt's last rows, missed them: 4.4 to 4.9 s to its
  first token on the local cards against main's 2.8 to 3.3 s, see "Long prompts layer by
  layer".)
- *The CPU takes the misses few rows picked* (`expert_store/hybrid/wide.rs`, 2026-11-22). A
  step of a few rows (a turn whose prompt the cache holds but for its last rows, the end of
  a prompt, the rows a replay rebuilds the state with) misses hundreds of experts per layer
  that one or two rows picked, and an upload costs the same whatever rows use the expert:
  0.43 ms on the x4 link of the RTX 4060 Ti, where the CPU computes a row of one in tens of
  microseconds. So in a step of up to `Options::cpu_rows` rows (2048) the part copies the
  layer's mixed input, its int8 codes and scales, to page-locked memory on the side stream
  when it enqueues the layer (`Part::cpu_act`); when the record has arrived the store splits
  the misses in the RAM tier between the CPU and the link by the cost model of the decode
  steps (`CostModel::split_by`: the fewest rows to the CPU, behind the uploads of the misses
  only the file holds, with a time per pair measured for this work apart from the decode
  steps' jobs), the CPU computes its share in one job over the rows that picked one of them
  (`LayerJob::with_rows`: an expert reads at most eight rows at a time, so one that more
  rows picked is given in pieces), the experts held in their RAM slots, and the part adds the
  sums to the layer's output behind the rounds (`Part::add_cpu_sums`). The layer ends when
  the rounds and the CPU have. The arithmetic of those experts is the decode steps' (8-bit
  activations), not the f16 of the many-row kernels.
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
  (`ExpertStore::routed`), as all misses did before transit. A device with room for a slot
  for every expert it owns has bytes for slots cut to its experts and a 64th; its transit
  comes on top of that, as far as its room goes (taken from it, the 64th held three of the
  six transit slots a split by layers of the test model got, and every layer a slot fewer
  than its experts: `qwen4exp_runtime`'s test of steps of 196 rows found 376 slots for 384
  experts on two cards, 2026-11-22).
- *No read is started on a guess.* The store reads guesses ahead into half of staging as
  prefetches and promotes them when a record names them (`transit_guess`), and `Feed` can
  guess, for the layers ahead (`ares_weights::Lookahead` gives the distance), what the wide
  step before missed there and no tier holds. That guess is off (`ModelOptions::
  read_ahead`), because it is mostly wrong and costs more than it gives: see "Transit,
  measured". The next layers' routers applied to this layer's input are a better guess;
  the steps of eight rows make it on the device (see "Reading experts ahead of their
  router" below), a wide step does not yet.
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

`step_rows` decides the rows a model keeps scratch for (`--ubatch-size` says otherwise): the largest power of two up to 4096
whose scratch (`Frame::step_bytes`) is at most a thirty-second of the smaller device's
memory. The scratch is about 0.7 MiB a
row for Flash-Next, and every byte of it is a byte less for expert slots, which the
generating steps miss.

**Where a faster prompt path goes next.**

- *A guess worth reading ahead.* The reads of a layer start when its record arrives and the
  layer waits for the slowest of them; what a first reading takes from the file is what
  bounds it. The store can read ahead of the router and promote what the record names; what
  it lacks is a guess that is right. The next layer's router applied to this layer's input
  (FATE reports close to all experts of a prompt's rows found that way) needs a second
  product and a second record per layer on the part that routes, and the record's way to the
  host; with it the reads of layer L+1 run while layer L computes.
- *The pass* (built 2026-11-23: "Long prompts layer by layer" below). A step of more rows
  brings each expert of a layer once for more rows, and a pass (all rows up to a layer's router
  in micro-batches that fit the scratch, then the layer's experts for all rows) takes the
  scratch out of that. A pass inside one step that cannot stop was designed and written but
  never compiled (the residual, the pending output and the experts' input of every row kept on
  the device, 56 KiB a row; rounds over the chunks; only for both cards on every layer), and is
  not in the tree: a step is what a request that arrives waits for, and a step of 512 rows
  already takes more than a second on this machine (see "Transit, measured"), so a pass of more
  rows inside one step keeps every other request waiting for seconds. The pass that was built
  pauses between micro-batches instead and goes on in the conductor's next record.

**Large prompt steps on lent VRAM** (`expert_store/scratch.rs`, `Part::lay_out`, `Flow::lend`;
2026-11-22). A step reads every expert its rows pick once, so a long prompt read in steps of
512 rows reads the experts of the file again for every 512 rows (each slice picks 40 to 80 %
of every layer). Strata reads prompts in chunks of 8192 and pays the chunk's buffers with
slots of its expert cache that it lends for the chunk and refills after (Strata
`DETAILS.md:1330`, `BATCHING.md:241`); Ares does the same:

- *Two sizes of a step.* `--ubatch-size` (`Options::own_rows`, `Model::own_rows`; by default
  `step_rows`, 512 on two cards of 12 GiB) is what the parts keep scratch for; `--batch-size`
  (`Options::max_rows`, `Model::max_rows`; 8192 by default) is the widest step, which only a
  prompt read alone takes. The plan sizes it by what the pools of each device's slots would
  lend (`batch_rows`), and once the store is built the parts take the most rows up to it, in
  multiples of 256 (`widest_rows`; of the own rows where those are fewer), whose scratch
  every device's slots lend in pieces (`ExpertStore::lendable_pieces`, `Frame::new_in`); the
  model's `max_rows` is that, and the server's log says it. (Until 2026-11-22 the rows were
  halved from 8192 and a loan came from the largest pool only: 2048 rows for UD-Q3_K_XL on the
  RTX 4060 Ti at 40G, whose largest pool holds 21 of its 48 layers.) A wider step's scratch
  (486 KiB a row on the part with the streams of the IQ2_XS file: the buffers of the read of
  a stream mixer, of the recurrent or attention half and of the n-gram block lie in one place
  on a loan, as a layer uses them one after the other; 725 KiB with them apart, as the own
  scratch keeps them, where a draft window keeps the n-gram block's `nrm` until its commit)
  lies in VRAM the expert slots of the device lend; everything else a step needs (the step's inputs and
  plans, the exchange of two cards, the predictions: about 50 KiB a row, and 24 KiB a row of
  page-locked RAM) is sized for the widest step for good. `Part::lay_out` points the scratch
  of each step at its own buffer or at the loan (the two layouts of `Frame`: `own` and `s`),
  and writes the zeros and the status word a step reads there.
- *The loan.* Always the last slots of the device's pools, the largest first, a piece of each
  the scratch needs (the slots of a pool lie in one allocation, the pools apart; the scratch
  is laid out so that no buffer is cut by the end of a piece, and `Pieces` maps an offset to
  its piece), so it lies at the same addresses every time: the mixer knows where the worker's
  lent scratch takes its input (`Ports::lent_raw`) from the load on. The store lends piece by
  piece (`ExpertStore::lend_pieces`) and keeps what it lent when a slot of a later pool was
  busy, until the next try. The slots belong to every layer of the pool, so the
  layers give their coldest slots in proportion to what they have above the floor of a loan
  (a row's picks, 10 for Flash-Next, `ExpertStore::scratch_floor`; until 2026-11-22 the free
  slots as well, 18, which only made the loan smaller: the misses of a wide step go through
  transit, not into the tier's free slots); what
  those held is evicted as any evicted expert is (copied down to RAM when the RAM tier keeps
  it), the experts that lie in the lent slots and stay move on the device into the slots
  given (`SlotDevice::adopt`), and the numbers change hands: the same bookkeeping as the RAM
  tier's for the pieces it gives back (`Tier::vacate`; each moves the bytes in its own
  memory). A lent number is set aside in its `SlotSet` (`SlotSet::set_aside`): nothing takes
  it again until the loan ends (`Tier::reopen`). (This loan of scratch is not the slot a miss
  is uploaded into "on loan" of `fix/mtp-and-slots`; the names keep them apart:
  `ExpertStore::lend_pieces`, `StoreStats::scratch_loans`.) The first part to begin a wide step lends on every device, waits
  until nothing touches the slots of the device (no read, upload or adoption on its way) and
  the moves and copies down have run, and only then does the step begin for the store.
- *The way back.* The slots go back, empty, at the end of the first step that does not run
  on them: that step's start still reads what the last wide step left there (the MTP block
  follows the rows of the step before from the part's residual). Empty slots fill the way
  they always do: the hottest experts of RAM (`promote_hot`, every free slot at once, started
  as the slots come back and at the end of every step), the fill while idle, what steps
  upload.
- *Who gets the wide step.* A prompt read alone; a generating sequence never shares a step
  with a loan (it rides along in pieces of the own scratch only), so nothing it holds in
  VRAM is lent while it generates. A request that arrives while a wide step runs waits for
  that step, then for at most one piece.
- *Exact?* The rows of a wide step are computed by the same kernels as in steps of the own
  scratch; what differs is the shape of the step: the attention plans' pieces and the order
  in which a wide step's experts add into a row (`red.add`, which differs from run to run
  already, see "Wide steps"). The test `a_prompt_read_alone_runs_on_vram_the_expert_slots_lend`
  (`tests/qwen4exp_runtime.rs`) checks the answer against the CPU reference in every
  placement, as the other wide-step tests do. In one run the lent steps' answer and the own
  scratch's agreed in all 16 tokens on one card.

The estimate (`batch_rows`) counts what the drafters take of the bytes for slots on the
device with the head (the MTP block's experts stay resident there) since 2026-11-22: without
it, on the 16 GB card with the MTP block, the parts were opened for 8192 rows (their buffers
for good 160 MiB larger, 48 slots fewer) and the store then lent 4096 all the same.

**Long prompts layer by layer** (`device/runtime/pass.rs`, `device/part/pass.rs`; 2026-11-23,
after TokenWing's layer-major prompts). A lent step reads each expert once for its rows, so a
long prompt in lent steps of 4096 rows still reads the file's experts once per 4096 rows, its
scratch costs 0.5 to 0.7 MiB a row of the slots, and a request that arrives waits for the step
in flight. Where the model can (`reads_layer_by_layer`: micro-batches the many-row kernels
take, K/V without an exact pool, no DFlash drafter; every placement), a step of one prompt
wider than the parts' own scratch runs as a **pass**:

- *The order.* The step's rows are cut into **micro-batches** of the own rows (512 here; the
  last two share what is left when the last would be narrower than 16 rows). For each layer,
  every micro-batch runs the layer up to its router and shared expert, as a step of its own
  rows would (its record and attention plan, made when the pass starts, go to the inbox from a
  ring of four page-locked buffers; its rows read and write the scratch of the own rows); then
  the layer's routed experts run once for every row of the step (`Misses` over all rows, its
  rounds over the chunks below); then the next layer. Each micro-batch's K/V and recurrent
  state of a layer is written before the next micro-batch's of that layer and after its own of
  the layer before, so every row sees what a step of all rows would: the same kernels, the
  sums of the dense products as in steps of the own rows; only the order in which the
  experts' outputs add into a row (`red.add`) and the attention plan's pieces (cut by the rows
  of a step) differ, as between any two steps of other widths.
- *What a row keeps* between layers lies on VRAM the expert slots lend for the pass, in
  **chunks** of up to sixteen micro-batches or 8192 rows (`chunk_rows`), each buffer of a chunk
  in one range (the many-row kernels index their input and output by row) and in one piece of
  the loan: the residual streams (40 KiB a row on Flash-Next), the output of the layer's mixture
  of experts (the shared expert's, scaled by its gate, and the routed experts' added to it) and
  its inject values, which the next read writes to the streams, the experts' f16 input, the
  router's picks (`ExpertKernels::pick_rows_into` writes a micro-batch's into its chunk's
  record) and the plan of the many-row kernels; the hidden values of one chunk's experts are
  shared by the chunks. About 56 KiB a row and 100 MB with a worker's 15 KiB a row (it keeps no
  streams), against 486 to 725 KiB a row of a lent step's scratch. While a micro-batch runs,
  `Part::at` maps the scratch's residual, MoE output and inject values to its rows of the chunk;
  everything else is the own scratch.
- *The loan* is sized for the step's rows (`Flow::lend_pass`, a piece of each pool as the lent
  steps' scratch), so a prompt of 17 000 rows lends about 0.9 GiB where a lent step of 4096 rows
  took 2 GiB, and a short prompt little. A pass that follows one that has read its rows takes
  its loan over when its rows fit in it; any other step gives it back before it begins. The
  last micro-batch's residual is copied to the own scratch, where the MTP block's following
  at the next step's start reads it, so no step after the pass reads the loan. The widest pass
  is what every device's slots can lend for (`pass_span`, the model's `max_rows`), at most
  `--batch-size` (32 768 by default).
- *Two devices.* By layers, each device runs its layers for every micro-batch, and the first
  hands the residual over per micro-batch in the lane's two parcels (the second waits on the
  host for them, or for the first to pause). With both devices on every layer, the exchanges of
  a piece are numbered by the pieces in the order they run, a block of eight each (the output of
  a layer's experts crosses in the next layer's piece of the same micro-batch, first, and is
  numbered again when a paused pass goes on), so that every flag only grows; the mixer decides
  for both whether a piece runs; the worker folds its shared expert's gated output into its
  experts' output, keeps its experts' input per row, and takes the mixer's picks of every row
  (with their weights) from the host, tagged by layer so that its thread never takes the layer
  before's.
- *Interruptible.* Before every micro-batch but the first of a step, the pass may **pause**
  (`DeviceModel::paused`, `StepOutput::paused`): when the engine was asked something since the
  conductor last took its commands (`StepContext::asks`, which the runtime raises on every
  request and cancellation; the record says what the conductor had seen), or when the record's
  time for the step has passed (`RecordView::pause_after`). Nothing of a paused step is
  committed. The conductor keeps its part, plans the other sequences without it, and writes its
  rows again when it is its turn: alone when nothing else runs (no time bound), else as a piece
  of prompt in the share's turns, after the other prompts' pieces, for about `piece_ms` (500
  ms). The model goes on where it stopped; its sequence never gives way, neither to a shortage
  of pages nor to a request of a better priority (`Claims::keep`: its state is neither before
  nor after its rows), and when it is cancelled nothing of it is kept and the next record gives
  the step up (`RecordView::gives_up_paused`). A request that arrives therefore waits for at
  most a layer's experts of the pass in flight and a micro-batch, then runs its own steps.
- *Images.* A prompt with an image is read layer by layer as any other (see "Images in a
  prompt"): each micro-batch's record carries its rows' rotary positions (the span's record all
  of them), the part that embeds copies the staged rows of the micro-batch's image cells over
  its gathered rows (`rows::splices_within` cuts the step's image runs to the micro-batch), and
  the MTP block follows each micro-batch at its rotary positions. A micro-batch may embed
  after the pass paused, so every record that goes on with a paused step stages the same rows
  again; the conductor holds the rows of the images among a paused step's rows
  (`paused_images`) so that the image store cannot let them go before the step is done, and
  the tower looks for a paused sequence's missing images only after the paused rows. In
  planning, a step is cut at an image whose rows are not there (and at the stage's room)
  before the checkpoints due inside it are named, so that they lie inside the rows it keeps;
  a checkpoint cut after them that leaves one too close to the step's end, or the step no
  wider than the own rows, ends the step at the first of them instead.
  (`a_prompt_with_an_image_read_layer_by_layer_gives_the_reference_answer`: a prompt of 400
  rows with an image of 8 x 5 made-up rows against the CPU reference, whose window takes
  input rows and rotary positions; `runtime/tests/images.rs`
  `a_paused_step_stages_its_image_rows_again_after_the_store_let_them_go`.)
- *Not in a pass:* the predictions of the next layer's picks of a wide step (a pass reads the
  next layer's experts that no tier holds into staging while a layer's micro-batches run).
- *Exact?* `a_prompt_read_layer_by_layer_gives_the_reference_answer` (`tests/qwen4exp_runtime.rs`)
  reads a prompt of 400 rows in micro-batches of 16 and 32 rows (one chunk and several) in
  every placement: the first tokens are the reference's and the same prompt's in steps of the
  own rows (all 16 tokens in five of the six runs); `a_request_cuts_in_between_the_pieces_of_a_long_prompt`
  sends a request 5 ms after a prompt of 480 rows: it gets its first token after 16 ms, the long
  prompt after 61 to 78 ms, and both answer as alone, token for token the reference's, in every
  placement. The conductor's side has tests of its own with a host executor that pauses
  (`tests/cache/prefix.rs`: a paused step runs again until its rows are done; a request runs between
  the pieces; a cancelled request gives its paused step up), and the oracle checks that no step
  between reads, saves or restores the paused sequence's state.
- *The CPU's share* of a pass of up to `cpu_rows` rows (2048): each micro-batch copies its
  rows' mixed input to the host where they lie among the pass's rows, the misses are split
  between the CPU and the link from the pass's picks as a wide step's are, and the CPU's sums
  go into their chunks' rows (`a_prompt_read_layer_by_layer_has_the_cpu_compute_its_misses`).
- *Checkpoints inside a pass.* A step ends where a checkpoint of the prefix cache is due (every
  16 384 fresh tokens, at the anchor: the last page boundary before a prompt's end or its last
  `<|im_start|>`, at the roots), and the rows after such a cut were a step of their own that
  picked nearly every expert again. A model that saves inside its steps
  (`Model::saves_inside_steps`, at least 16 rows from either end) gets those checkpoints named
  in the record instead (saves inside the step: slot, place, position; the conductor takes the
  places before the step and names their pages at its commit, `Cache::reserve`/`attach`), and the
  step goes on. The pass cuts its micro-batches at those rows and, as each layer goes past one,
  copies that layer's part of the slot (`Frame::layer_slot`: its recurrent state, its indexer's
  carried keys, the n-gram history at its layer) into the place (`Places::save_part`); the MTP
  block's state goes there once the block has followed the rows up to the position, and what
  the n-gram feed keeps of the sequence there is written at once. The place holds, when the pass
  is done, what a save at the end of a step at that position would hold.
  (`a_prompt_read_layer_by_layer_leaves_checkpoints_inside_it`, `tests/cache/prefix.rs`
  `checkpoints_inside_a_step_serve_a_later_prompt`.) The tail behind the last checkpoint that
  is too close to the end is a step of its own as before. The places of the saves inside a
  step are locked where a step's saves lock theirs, after the step's copies and before any
  layer (`Resume::hold`): a lock of page-locked memory waits for the device's work, and between
  two micro-batches of a pass on both devices that work spins on the other device's exchanges,
  which wait for the thread that locks (the pass stood still, 30 s, and the engine failed).
  Until the commit names them, the reserved places count among the locks the next record will
  make (`Cache::pending_locks`), so that no other place is handed the same room of the limit;
  they are taken before what leaves the devices is settled, since taking one may give other
  checkpoints up. They are taken the latest first (when the memory holds fewer than are due,
  the ones nearest the end, which the next turn goes on from, get them), and the commit names
  each in order and makes room for it as a save at a step's end does, so that a later one
  pushes an earlier one out as after cuts: taken all at once, a first turn of 9821 rows kept
  the checkpoint at 8192 and lost its anchor, and the next turn replayed 1635 rows instead of
  99 (`real_cache`'s Flash-Next test asserts fewer than 256; `tests/cache/prefix.rs`
  `a_turn_after_saves_inside_a_step_replays_no_more_than_after_cut_steps`).
