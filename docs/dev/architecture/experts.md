# Experts of Flash-Next: profile, tiers, reading ahead, pruning

**The usage profile of the experts.** The tiers start empty, and what they learn
while the model serves, the store's heat, went with the model. A start filled
them by reading a text of German, English and code (41 to 44 s on 2026-11-11, 20
to 32 s on 2026-11-16), and the first texts unlike it waited for the drive. Now the heat outlives the
model, as Strata's `expert_profile_save` and Run My AI's table of usage do
(`expert_store/profile.rs`, `expert_store/fill.rs`, `qwen4exp/device/usage.rs`):

- *What it is.* The heat of every expert of every layer of the model as the store
  has it: one for every expert a step picks, whatever rows pick it (a wide step's
  as a decode step's; until 2026-11-22 a wide step counted every row, so a prompt
  outweighed everything generated after it and VRAM filled from such a profile held
  what long prompts used: 16 to 23 % of a decode's picks on the test machine, see
  below; profiles of that kind are format 1 and are not used), times 0.9 every 32
  steps and nothing while idle, so a count in which recent picks weigh more (half of
  it is from the last 210 steps or so; the ratios survive the decay, so it ranks by
  the whole history). It is per layer of the model and expert, whatever the
  placement: with both cards on every layer an expert's heat is the larger of the
  two halves of the store's layer (a decode step counts an expert on its owner's
  half, a wide step on both), and it is seeded on its owner's half. The draft
  block's experts are not in it.
- *Where.* A file of 96 KiB for Flash-Next (`<fingerprint>.experts`) in the
  `--cache` directory when one is named (a file next to the prompt cache's stores,
  which are directories) and else in the user's cache directory
  (`$XDG_CACHE_HOME/ares`, `~/.cache/ares`); no option of its own. The name is a
  fingerprint of the model's files (size, header and samples, as the prompt cache's
  store has it) and of the shape of its experts; the file says again what it was
  made for and ends in a checksum, so a profile of another file, a damaged one or
  one of another format is ignored and the log says which (`Ignored`). It is
  written through a temporary file in the same directory, made durable and renamed
  over the old one: a reader finds the old profile or the new one, whole.
- *When it is written.* Right after the warm-up of a start that had none; once a
  minute while steps run (a thread of the model, `ares-expert-usage`, compares the
  store's count of steps with the one of the last write); and when the model is
  dropped, at an unload and at a clean stop.
- *The start* (`Qwen4Exp::serving`, which the server calls where it read the
  warm-up text, after the runtime measured the price of a step): the heat becomes
  the profile's, so that the steps of the measurement, which route token 0 a
  thousand times, do not enter it, and the tiers are filled from it
  (`ExpertStore::plan_fill`, `advance_fill`). For every layer VRAM is made to hold
  the hottest experts: every free slot is filled, and a member gives way to an expert
  hotter than itself (the measurement's steps leave their experts in the slots, and
  until 2026-11-22 only members without heat gave way, so on the test machine the
  fill brought 1 267 to 1 862 experts into 3 888 slots and left the rest to what the
  measurement had brought); behind them the next hottest as far as the RAM tier has
  room, a free slot or a member worth nothing there (one without heat, a copy of an
  expert VRAM holds): nothing in RAM that is worth something is evicted for a fill,
  and nothing without heat is brought. The experts are read layer by layer in the order
  of the file, in each layer those for VRAM before those for RAM (the ones for VRAM
  pass through RAM slots, which those for RAM would otherwise have taken first), each
  group as it lies in the layer's tensors, 32 at a time as demand reads (three reads
  of about 0.5 MiB each), each into its RAM slot and the ones for VRAM from there into
  their VRAM slot, the table entry behind the bytes, after which the copy in RAM is the
  first to give way. A filled expert is a member worth its heat, not a guess that
  nobody used. The store's lock is let go between the turns, so the watch of the
  machine's memory can shrink the tier meanwhile. Without a profile the start reads
  the warm-up text as before, writes the profile at once and fills the tiers from
  the heat the warm-up left in the same way (`Qwen4Exp::fill_from_heat`).
  The same fill takes a list instead of the heat (`ExpertStore::plan_listed_fill`:
  the MTP block's experts before the first chains, a session's or a test's recorded
  routing or every expert), with the same rules but no member giving way. A fill that
  finds an expert on its way into RAM for someone else (a read ahead of a router)
  takes what that read brings, and reads the expert itself when it was cancelled.
  (Until 2026-11-24 a list went through an older way in, `ExpertStore::fill` and
  `predicted`, as guesses; it is gone.)
- *While nothing runs* (no step for a second): the same thread fills what is empty
  or colder, VRAM's free slots and members colder than an expert outside, and the
  RAM tier's room, from the heat, those for VRAM first and
  the hottest first over all layers, two experts at a time as prefetches (which the
  reader keeps behind every demand read: a demand read waits at most for the four
  chunks of 512 KiB a drive takes of prefetches, see `ares_io::Config`) and one upload
  per device. A step that begins stops it at once: the store does nothing of a fill
  while a step runs, what is on its way takes its slot when it arrives, and the plan
  is made again when the engine is idle the next time. It follows the elastic tier:
  it reads only into the slots the tier holds now, and into nothing while the tier
  gives memory back or holds more than it may (the reads it has on their way then
  are all a release waits for), and the pieces the tier takes again are filled.

**Members, loans and promotions** (`expert_store/hybrid.rs`, `ares_weights::SlotSet`). While
steps run, what VRAM holds changes slowly and by heat, and the link still serves misses
beside the CPU where that is sooner:

- *A miss* is served by the CPU or over the link, per layer and step by the measured costs
  (`CostModel`: a CPU job by the types of the layer's experts and its pairs, timed at load and
  by every job; an upload by device, timed by every upload), so that the later of the two ends
  first; on a slow link the CPU takes more. An upload becomes a member only when it takes a
  free slot or beats the coldest member the step did not pick by the rule of a promotion
  (`SwapRule`: more than 1.25 times its heat and one pick more). Otherwise it goes up *on
  loan* (`SlotSet::lend`): into the slot of a member on loan, or of the coldest member, which
  is how the slots on loan come to be. A member on loan is worth nothing, also when a step
  uses it again: its slot is the first any upload takes, so the slots on loan pass from miss
  to miss and the members stay. Until 2026-11-22 every upload of a miss took the coldest
  member's slot for good: VRAM followed the last tokens instead of the text (63 to 171 uploads
  a token on the test machine, a third to a half of the picks in VRAM).
- *Promotions*, between steps (`ExpertStore::promote_hot`): per layer the hottest experts
  outside and on loan against the coldest members, as long as the rule holds. A member on
  loan that beats one becomes a member and that one goes on loan, without a copy
  (`SlotSet::keep_loan`); an expert in RAM is uploaded into a free slot or the slot that
  leaves, its entry written when the bytes are there and no step is at the layer. As many
  swaps may be on the link as it does in the time of a step beside the uploads steps wait for,
  by the measured times of a step and of an upload (one while either was never measured),
  where there were eight at most before; a free slot evicts nothing and fills at once
  (2026-11-22: a loan of VRAM gives thousands back empty, and filled a step's room at a time
  they left the reply after a long prompt at 10 to 13 tokens/s against 36 on the local cards;
  the split of the steps' misses counts every upload on the link, so it gives the CPU what
  would wait behind the fill).
- *At the start and while nothing runs* the fill makes VRAM the hottest of every layer
  without that margin (see the usage profile above): no step waits then.

**Reading experts ahead of their router.** A pick that no tier holds is read from the file
when its layer's record reaches the host, and the layer waits for it. The routers of the
next layers applied to the rows the router of this layer reads name most of what those
layers will pick (Pre-gated MoE, FATE, ProMoE; Run My AI applies the routers of layers L+1
and L+2 on the GPU and prefetches in chunks of 512 KiB, at most two while a demand read
runs), so their reads can start a layer or two before their records
(`device/part/predict.rs`, `expert_store/ahead.rs`, `ares_cuda::ExpertKernels::predict`):

- *On the device.* In a step of the decode kernels (up to eight rows), right after the
  router of layer L, the part that runs the router (one card, a range of layers, or the
  mixer when both cards work on every layer) applies the routers of layers L+1 and L+2 of
  its range (`Options::predict`, 2) to the rows the router of L read: the float weight
  kernel with a router per row of its grid, and a block per router that ranks every
  expert by its logit in each row (the router's own order) and lists those that some row
  ranks among the first ten, by their best rank. Two launches and a copy of the lists
  (a few hundred bytes) to the board in page-locked memory, on a stream of the
  predictions that waits for the router's event; the step's stream never waits for it in
  the step, and the predictions write only their own buffers (the logits and the lists),
  so what the model computes is the same bit for bit. The next step waits for the last
  prediction of the step before it writes the rows they read. The router's input is
  the input of layer L's experts, without the shared expert's output, which is on the
  other card when the worker computes it (`--shared worker`).
- *On the host.* The feed hands the lists of a layer to the store behind the layer's
  route record (`Feed::follow_predictions`; a list carries the step's token, and one of
  another step is ignored), with both cards on every layer each expert to its owner's half
  of the store layer. `ExpertStore::ahead` takes them for a layer the step has not routed:
  what a tier holds or what is on its way is left alone; the others are read into the RAM
  tier as guesses, the likeliest first and behind every demand read. A read ahead takes a
  slot only when its read starts, and only room that nobody uses (`Want::Spare`): a free
  slot, a guess nobody used, a copy of an expert that VRAM holds. It never takes the place
  of a member a later step may need: when it did (as any guess does, the coldest member),
  the generating steps after a prompt read with reads ahead read 14 experts a step from the
  file where the build of before read 6 to 8 (one card, and two cards with `--ram 4G`).
- *Only what is picked more often than not.* When the tiers hold what a text uses, an
  expert that no tier holds is one the text has not used, and a prediction that names it is
  mostly wrong: on two cards with `--ram 8G` a read ahead of every such expert was picked 3 %
  of the time (0.11 used and 3.59 wasted of 3.7 reads a step, and 31 more that no bound let
  start), as the first attempt had found on the host. So the store counts, per distance of
  the prediction and place in its list (the best rank among the rows), how often an expert
  named there that no tier held was then picked, read or not, and reads ahead only at the
  places where that has been at least half the time (16 samples first; the counts are
  halved beyond 512, so they follow the text and the tiers): then a read for nothing costs
  the drive no more than the reads that are used save. Where the tiers hold little (one
  card, a small RAM tier, a larger file) more places qualify.
- *Bounds from the drive and the step.* At most as many experts on their way ahead as the
  drives take of prefetches at once (`Reader::prefetch_room`: four chunks of 512 KiB per
  drive, two experts of Flash-Next), and per step at most what the drive could read in the
  time of a step beside the step's own demand reads (the drive's measured pace and means of
  the steps that predicted). The reader keeps every prefetch behind every demand read and
  takes only half its prefetch depth while a demand chunk is at the drive, so a demand
  read shares the drive with at most two chunks of 512 KiB.
- *Dropped when the record arrives.* The guesses for the layer that have not started are
  forgotten, the read of an expert the record did not pick is cancelled (what is at the
  drive comes back and is dropped), and a pick whose read is on its way is promoted to a
  demand read, as for any guess. Whoever else asked for the expert while the cancelled read
  was on its way (a fill of the tiers between steps, a prediction of a step) took it as
  coming and started no read of its own: when the read fails, the store reads the expert
  again for it. (Without that, a fill right after a step left such an expert in no tier,
  and the next step that picked it read it from the file.)
- *Wide steps* (a slice of a prompt): at layer L the next layer's router is applied to
  the rows as a float product on the f16 tile kernels, on the predictions' stream (the
  step's stream waits only for the f16 copy of the rows, which the next layer's read
  overwrites), and every row's picks go to the board. The feed counts them per expert and
  hands them, the most picked first, to the store as the guesses of layer L+1
  (`ExpertStore::transit_guess`): what no tier holds is read into staging as prefetches, in
  at most half of it, and the record of L+1 promotes what it names (`transit_want`) and
  drops the rest (`transit_forget`). A layer of a wide step takes tens of milliseconds, so
  one layer is lead enough. The prediction is made only at the layers where the wide step
  before read the next layer's experts from the file (the board carries the feed's word to
  the part), so a prompt whose experts are in the tiers pays nothing for it.
  Since 2026-11-22 (`flash/ssd-stream`) two things differ where the experts stream from
  the SSD. *While VRAM is lent* (a prompt is read alone then, whose pass only other
  requests' steps interrupt) the guesses go as demand reads, behind the misses asked for
  before them: as prefetches the
  reader kept them to a few chunks of 512 KiB beside a demand read, and through a 32k prompt
  on UD-Q4_K_XL the drive read 1.65 to 1.85 GB/s where it gives 2.4, two thirds of it these
  guesses. And *a guess is read only where its class is named at least as often as not*
  (`GuessOdds` in `transit.rs`; classes: the step before's misses, and a prediction's by the
  rows it named the expert for, 1, 2, 3-4, ... 65 and more): such a step is bound by the
  drive, and 24 % of the guesses read through that prompt were not named by their layer's
  record. Every guess is counted when its record comes, read or not, so a class that
  stopped being read comes back; the server's log says the counts every 20 s. On that
  prompt every class was named 61 to 82 % of the time, so none was left unread, and the
  drive's rate did not rise: such a step is bound by the link (below). *The guesses that the
  RAM tier holds go up into transit slots* the layer before leaves free, while nothing named
  waits for one, in at most half of them (`Transit::upload_guesses`): a layer's misses went
  over the link only once its record had named them, so from a layer's last upload until the
  next layer's record (its first kernels for 4096 rows, ~37 ms) and through its last round
  the link had nothing to do. A record that names an uploaded guess finds it there; a layer
  done with gives the slots of the others back. A named miss that waits for a slot, from RAM
  or read into staging, keeps the guesses out: when only those RAM held did, the guesses took
  the slots the staged ones waited for, and where the drive is the bound (IQ2_XS, `--ram 8G`,
  the RTX 3060 on PCIe 3.0 x4, 2026-11-22) a prompt of 8k tokens took 71.0 s against 47.7 and
  46.6 without the uploads; with the rule, 45.0 s (2k: 29.9 against 29.3 and 30.3), 88 to 93 %
  of the uploaded guesses named. On the link-bound 16 GB card it is not measured yet.
  *A pass* (a prompt read layer by layer) guesses the whole next layer of the part while its
  micro-batches run (`Flow::guess_layer`): every expert counts as a prediction's for the rows
  that pick one of the layer's experts on average (the pass's rows times `top_k` over the
  experts, the top class at the widths of a pass), so its record counts those odds, what no
  tier holds is read into staging and what RAM holds goes up into the free transit slots.
- *The text is unchanged.* The fingerprint of the tool (`qwen4exp_run --fingerprint`: the
  logits of a prompt of 600 tokens, 24 generated tokens and windows of drafts, bit for bit;
  both cards with the split pinned, `--core-share 0.76 --shared worker`, every miss
  uploaded, `--serve stream`) is the same with the predictions and the reads ahead as
  without them, with `--ram-gib 8` and with `--ram-gib 4` (2026-11-16: `ffbe225f5e70e94b`
  for all four; the runs with the predictions read 4.2 and 4.6 experts ahead per step, of
  which 1.1 and 1.2 were picked, in the prompt's steps of eight rows and the windows). With the prompt in wide steps (`--rows 512`) no comparison holds bit for
  bit: two runs of the build of before differ themselves (the many-row kernels add a row's
  experts with `red.add`, `Accumulate::Atomic`, in the order of the schedule), and the
  picks of a run with the predictions were those of one of them.
- *Counted* (`StoreStats`, `/metrics`): experts predicted and of them those picked
  (`ares_expert_predicted_total`, `ares_expert_predicted_picked_total`), reads ahead and
  their bytes, those the record picked and those it did not (`ares_expert_ahead_*`, the
  wide steps' guesses included). A layer's wait for its misses in a step of up to eight
  rows (from its record to the last miss in a slot or delivered) is also counted by what
  came last (`StoreStats::wait_last_ms`, `ares_expert_wait_{cpu,ram,file,on_its_way}_microseconds_total`):
  the CPU's sums of experts in RAM, an upload from RAM, an expert read from the file
  (uploaded, or computed by the CPU once its bytes landed), one on its way already; so a
  profile of decode says which source the device waited for (`qwen4exp_run`'s `waits`
  line).

**Pruning as a fourth level** (`device/prune.rs`, `ExpertKernels::route_pruned`,
`kernels/expert_router.cuh`). Next to VRAM, RAM and the file, a pick may also be
*pruned*: when its expert only the file holds, the router can leave it out or let an
expert a tier holds stand in for it, and the step does not wait for the drive. It is
off unless asked for, because it changes the output. Three rules, behind one dial:

- *skip* (Strata's tail skip): the pick contributes nothing.
- *router*: the router's best expert that a tier holds and that the row did not pick
  stands in for it (the j-th touched pick of a row gets the j-th best such expert). This
  is what a routing bias with the strongest picks exempt gives, which the store described
  and never had (`SlotSet::write_route_bias`, removed).
- *similar* (the owner's idea): the first of the pick's most similar experts that a tier
  holds and that the row did not pick stands in for it; a pick without one stays. How
  alike two experts are is measured once per model file and kept beside the usage profile
  (`ExpertNeighbours`, `<fingerprint>.near`): see "The table of similar experts" below.

The dial says which picks a rule may touch: those from a rank on (`from_rank`, 0 the
strongest of the row's ten), with a weight below a bound, and for *similar* a stand-in
of at least a closeness. The weights of what is computed are the router's (a stand-in
takes the weight of the pick it stands in for; a pick left out leaves the others as they
are, as Strata does), renormalised over what is computed (every computed expert its own
probability over their sum), or, for *similar*, the router's times the table's scale.

*Where.* On the device, right after the top-k, before the groups of the route record are
made (`ares_expert_select_pruned`, `ares_expert_pick_pruned`, new entry points beside
the plain ones, which stay as they were). The record keeps the router's picks and weights
(`RouteRecord::ids`, `weights`); its groups, which the expert kernels, the split between
two cards and the store work from, are what is computed. So the store never sees a pruned
pick, and a row with no touched pick computes what it computes without pruning, bit for
bit. Which experts a tier holds is a byte per expert of every layer of the model that the
host writes at the start of every step from the store (`PruneMask`: VRAM on any card, the
RAM tier, or on its way to either; `ExpertStore::mark_present`), two page-locked buffers
taking turns; every part that routes copies its layers' bytes onto its card on its stream
before its first layer (`PartPrune::begin`). The router's slot table could not say it: with
both cards on every layer the mixer's table holds only its own experts, and none says what
RAM holds. An expert that arrives during the step counts from the next one. Only steps of
up to eight rows are pruned (generation, windows of drafts); a slice of a prompt is read
whole. The decision is per row, so a row's result does not depend on what the other rows
of a window routed (Strata decides per expert over the window: an expert that some row
needs strongly is read for all).

*The heat* still counts what the router wanted: the store gets the picks that were left
out or stood in for as heat of their own (`Feed`, `ExpertStore::record_counts`), so the
tiers learn a text's experts between steps as they would without pruning, and a pruned
expert that is used often enough comes in and is computed from then on. The stand-ins
count as picks as well.

*The table of similar experts* (`ExpertNeighbours`; made by `qwen4exp_prune calibrate`
and `near`, `crates/model/examples/qwen4exp_prune/`). What matters of a stand-in is how
near its output is to the output of the pick it stands in for, on the rows the router
sends to that pick. So the table is measured on such rows: a calibration (texts of German,
English and code other than the ones the rules are measured on, read in steps of eight
rows) keeps every row's input of every layer's experts as the kernels read it (int8 codes
and their scales) and the router's picks. For every expert X of a layer, the rows that
picked it (the largest weight first, at most eight; the rows whose router score for X is
highest when fewer than four did) are run through X and through each candidate Y on the
CPU, with the engine's own CPU kernels, which compute what the device does. The closeness
of Y to X is the squared cosine of the two outputs, each row weighing the square of X's
weight there, and 0 where they point apart: the share of X's output that Y, scaled best,
gives. Its least-squares scale is kept with it. The candidates are the 48 nearest by the
outputs on 16 rows that all experts share and the 16 nearest by the router's rows; every
expert keeps its 16 nearest.

*Recommendation.* Offer `skip` and leave it off by default (it changes the output, and
the operator chooses that). As the value to suggest: `skip:8` (Strata's tail) where the
model's quality matters most (KL 0.008 on the 2-bit file, 0.026 on Q4), `skip:4` where speed
from the SSD matters more (KL 0.011 on the 2-bit file for 1.45 times the speed with 8 GiB,
0.077 on Q4 for 2.5 times). `router` and `similar` stay as rules for measurement: they cost
more at every rank that leaves most picks alone.
