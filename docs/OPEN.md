# What is still open

Updated 2026-11-22, `main` at the prefill path measured per type on full-rate cards (after the
shared layer's steps 5 and 6: one MTP driver, drafter kinds `ngram`, `mtp`, `dflash`; nothing
pushed). What is there is listed in
[`user/features.md`](user/features.md).
This file is brought up to date at every merge into `main`. It lists work, defects and
measurements that are owed; what a feature must do in detail is in
[`dev/studies/feature-checklist.md`](dev/studies/feature-checklist.md) (223 rows: 70 done,
42 partly, 88 not started, 23 outside the frame; its rows K01, K02, K06 and P14 are out
of date: reasoning is built, and `q7` to `q4` run in the 27B).

GPU work runs on the test machine (RTX 4060 Ti 16 GB, gpu-unlock) from
2026-11-22 on; the two local cards are production and are used only when a test needs two
cards, with the owner told first.

## Why Ares (the owner, 2026-11-23)
1. Control over the features: the engine does what the owner wants (Jev mode, for example, is a
   feature the owner values highly).
2. Speed, always.
3. Parallel requests: llama.cpp is very poor at them and vLLM needs a lot of VRAM.
4. Efficient use of VRAM and RAM.
5. Large mixtures of experts streamed from the SSD, better than Strata: a Q4 or Q5 of a model
   larger than RAM without a large loss of speed.
6. Room for experiments: new prefetchers, drafting and prefill tricks, dynamic pruning.
Every piece is weighed against these.

Order of work (the owner, 2026-11-23): no phases any more. Every piece is placed by the reasons
above, in the order under "Decided by the owner" (prefill, the parallel-request fixes, large
mixtures of experts from the SSD, Jev mode, the few-row expert kernels, then Inkling 2 Mini and
trees). What is left of "CUDA complete" (the list below) and of "Optimising" is a store to draw
from where it serves one of the reasons. Vulkan parity stays last.

## The order of 2026-11-17 (after the gap study; items 1 to 7 done, the rest is a store)
For "CUDA complete", the order in which the items below are taken up; each item stays in its
section further down.
8. Then the rest of CUDA complete: image input pieces 4 to 8, gate and up in two types, the
   roles of the cards, the open defects, the server items.
Later than their place suggests: the classic draft model (VRAM; better with an aux card),
Jev mode, stored responses, TLS. DFlash2 trees (steps 9 and 10) with the experts of a tree's
tokens read ahead ("Speculation that pays"): the owner, 2026-11-22, expects much of them for
decode; measured first (how many picks a token table and route memory predict, how much longer a
tree is accepted than a chain), then built; started when the pieces in work leave room, one
piece at a time.

## In work now

| Piece | State |
| --- | --- |
| Parallel requests against the owner's fork at production settings (the owner, 2026-11-23: "the whole reason I am making the engine"): two generating, one generating beside a long prompt, two agent sessions, 3 to 8 at once, priorities | measured (see "Parallel requests against the owner's production"); the fixes merged 2026-11-24 (above). Open: the expert store stalling under a full loan, a floor that rises for a request near its end, the `--parallel` default (question below) |
| DFlash2 trees and the experts of a tree's tokens read ahead, phase 1 | measured (`flash/trees-study`, instruments only, not merged): Flash-Next +1 to 3 % with a tree of 4, larger trees lose, reading ahead from token tables or route memory +1 % (they name 1.4 to 6 % of a round's missing experts); 27B +9 % with trees of at most 8 rows; the MTP chain with its own candidates as siblings is as good as a DFlash tree of 8 nodes. The lever the data point at: a distinct expert in a few-row step costs 16.5 µs (about 80 GB/s of the cards' 360 / 500); at 5 µs chains reach about 120 tokens/s and trees then pay +15 %. Owner's decision on what to build |

## Shared model layer (the owner's rule, 2026-11-16)
Generic for Ares, not like llama.cpp: a few models, shared code where it can be, each model
still at its maximum through its own launch plan. A part becomes a shared block when it owns
state the runtime touches (pages, slot state, checkpoints, windows, commits, replays, trees)
or when it drafts; products, norms, kernel choice per row count, fused launches and exchanges
between cards stay with the model. Steps 0 to 8 made the K/V group, the recurrent block, the
slot arena with its checkpoints, the token rows, the head view and the MTP driver shared
(`crates/model/src/blocks`; the rule is in `dev/architecture/overview.md`, "The shared model layer"). Design
study: `/var/tmp/ares-survey/shared-model-layer.md`. Each step lands green on its own.
- What trees (9, 10) find from 8: the lattice of every pass is kept (`Drafter::lattice`: 16
  candidates and a 16 x 16 score table per position, tokens already mapped through the head
  view); the walk is one function of it (`Lattice::chain`, kernel `ares_dflash_select`), which a
  tree fill replaces; the drafts list has a place per window, which becomes a place per node;
  the inject is per row and position, so a tree's kept path needs only its rows' positions
  (a rejected branch's rows are never read, as now).
- [ ] 9. Trees in the blocks, verified once for every drafter and both models. Later (the owner, 2026-11-17).
- [ ] 10. Tree fill from the DFlash lattice, MTP links mixed into a tree. Later.
- [ ] Flash-Next past the dense limit: a draft window's row that reaches across the point where
      a tile leaves the newest cells reads that tile exact, the row alone reads it converted
      (`Group::ahead` is planned for rows within the limit only; the lists are made on the
      device from the table, which still names the exact entry). A shadow row of the table for
      such rows would close it.
- [ ] Flash-Next with several sequences: a generated row is planned alone now (no prefix read
      together with other sequences); its cost with three sequences was not measured.

## Qwen3.8 Flash-Next (`qwen4exp`)

### Found against Strata on the test machine (2026-11-22; RTX 4060 Ti 16 GB, 62 GiB RAM, UD-Q3_K_XL)
- [ ] Ares is behind: decode with drafts 40-55 % of Strata's (15.9-31.0 against 40.0-43.0
      tokens/s), without 43-51 %; prompts 4 to 10 times slower (Strata 398 / 829 / 1257 tokens/s
      at 2k / 8k / 32k, Ares 94-121); the first token of a short repeated prompt 3 to 15 times
      slower (no checkpoint at the prompt's end: item 2 of the order). Ahead: load 3 times faster
      (35-50 s against 96-149), 12 GiB of RAM left free against Strata's 2-3, a long conversation
      turn asked again 2.7-2.9 s against 9.3 s (Strata keeps only the newest path).
- [ ] Against Strata 0.2.19 (engine 0.1.65, installed 2026-11-22 evening) at 40 GiB: not measured
      yet (the test machine switched off at 21:00); the runs are queued to start when it is back
      (`~/work/flash-cpu/requeue.log`, tables in `results/summary-final.txt`). Strata 0.1.60 at 40
      GiB decoded at 29.9 tokens/s with drafts (mean); Ares now 33.6 to 36.4. Without drafts the
      device's 27 ms a token outside the experts (below) bounds Ares near 30 tokens/s here.
- [ ] Without uploads of misses there are no loans, and the share of picks in VRAM falls (70
      against 80 % at the start of qwen4exp_run's decode at 4000 tokens, 53-56 against 80 %
      later in it). Uploading every miss the CPU computed on loan as well, behind the step
      (118 a step), won it back only to 75 % and made the steps slower (42.8 against 40.6 ms at
      the median, the dense part on the device 19.3 against 15.6 ms): on this x4 link the
      uploads hold up what a step reads from the host and the layers' sums. A policy for the
      slots that keeps a token's misses for the next tokens without the link: open.
- [ ] A decode step of UD-Q3_K_XL on the RTX 4060 Ti takes about 27 ms on the device outside
      the experts (qwen4exp_run --breakdown at 4000 tokens: dense 15.6 ms, stream mixers 7.8,
      exchange 1.1, recurrent 1.0, attention 0.9); with the CPU's misses about 40 ms a token
      without drafts. The dense weights (3.85 GiB) at the card's 288 GB/s are 14 ms alone: the
      floor of decode on this card is there, not in the experts.
- [ ] Prompts on the test machine at 40 GiB: 3 to 4 times behind Strata 0.1.60 at 8k and 32k
      tokens (Ares 250 and 281 tokens/s, Strata 787 and 1192). A prompt read alone took steps of
      2048 rows on the VRAM the slots lend on this card (Strata: chunks of 8192), and in such a
      step nearly every expert is picked, so the ~20 800 experts without a VRAM slot cross the
      x4 link once per step: about 45 GB at 6.4 GB/s, 7 s per 2048 rows, 290 tokens/s, which is
      what was measured. Built on `flash/prompt` (2026-11-22): loans from every pool of slots in
      pieces, steps in multiples of 256 rows, the read, core and n-gram buffers in one place on
      a loan (725 -> 486 KiB a row), the CPU's share of steps of up to 2048 rows, a dense
      activation form per type. To measure there against Strata 0.1.65. What is left beside
      them: the uploads of a layer start only with its record (`flash/ssd-stream` uploads the
      next layer's guesses into free transit slots); the CPU takes no part in steps of more than
      2048 rows; about 40 KiB a row of VRAM (the step's staged and decoded rows, plans,
      predictions) and 28 KiB of page-locked RAM are sized for the widest step for good, which
      bounds `--batch-size` and costs decode its slots.
- [ ] `qwen4exp_runtime`'s `flash_next_writes_with_drafts_what_it_writes_without` fails on the two
      local cards in some runs (2026-11-23, IQ2_XS with the MTP block): the greedy reply with
      drafts parts from the one without at a near tie (token 16, "," against "."), with the whole
      head or with the draft vocabulary. Main failed it in one of two runs, `fix/turn-cell` in
      three of three, whose change is not reached there (the test keeps no checkpoints); it
      passed on the test machine's one card. A test of one arithmetic for both replies would
      need the step shapes of both to be the same.
- [ ] A cached prompt's replayed rows and its computed rows are steps of their own (and a
      prompt step is cut at a root): two passes over the experts where one would do; with the
      CPU's share of small steps cheaper, not built.
- [ ] Transit adoption (`transit_adopt`, the prompt chunks' experts taking VRAM slots after a
      wide step) asks only "hotter than the coldest member" (with the swap margin when the member
      has to be copied down), not the rule of a promotion; left as it was for the work on prompt
      chunks that borrow expert slots.

### Defect seen on the local cards (2026-11-22, `flash/ssd-stream`)
- [ ] `qwen4exp_runtime` `the_mtp_block_follows_wide_steps_that_start_at_any_cell` fails on the
      RTX 4070 + RTX 3060 in its second placement (`Layers`): `warm()` asserts "every expert has
      a slot" (the made-up model's slots with steps of 196 rows do not hold every expert there).
      The same with `verify/stack` 7816385 itself, so it is the test's or the stack's, not this
      branch's; the placement `One` passes.

### Defects seen in the merge check of 2026-11-17 (intermittent, slower but not wrong)
- [ ] `ares-server --test real_flash_next` `flash_next_drafts_with_its_mtp_block` sometimes verifies no
      draft at all (0 of 0) for a whole run; passes in other runs. Suspected: the conductor's
      prices measured cold at the start shut the windows out while warm-up is off with drafts.
- [ ] `ares-server --test real_cache` `flash_next_resumes_a_conversation_from_the_cards_from_ram_and_from_ssd`
      failed once in the full suite under load (after two other conversations took the pool,
      4096 or fewer of 9851 tokens came from the cache); alone it passes (9728 from SSD).
      Suspected: a page on its way from RAM to the SSD is found in neither while the write runs.
- [ ] `ares-model --test qwen4exp_runtime` `the_mtp_block_drafts_what_the_reference_drafts_in_every_placement`
      failed once in the merge check of 2026-11-22 (first links), passed three times alone. In
      one of those runs the policy kept asking the made-up block for 21 chains (78 drafts, none
      accepted) where it otherwise stops after one; then the allowance of one parting chain in
      five is at its edge. Why the policy keeps asking a block that is never right is open.

### Speculation that pays (output unchanged)
- [ ] Experts read ahead from the drafts (the owner's idea, 2026-11-18; no engine has it:
      TensorFold's drafts and routing do not interact, Strata and TokenWing predict within one
      token only). The drafts name the next tokens a round before they are verified; reading
      their experts ahead (SSD to RAM, RAM to VRAM) during the drafting step and the round
      makes window rows cheap, and with them trees on a mixture of experts. A token alone does
      not name its experts, so: (1) route memory: for the `ngram` drafter's copied text, the
      picks recorded when the copied tokens ran before (about 1 KiB a token); (2) a table of
      each token's usual experts per layer, kept by decayed counts, for `mtp` and `dflash`
      drafts; (3) a tree's branches weighted by their chance, within the drive's measured pace
      (the read-ahead's caps); (4) a tree priced by the tiers its rows' predicted experts are
      in. Measure first: how many picks (1) and (2) predict, and how much time a round leaves
      for reads; then decide on trees for Flash-Next again (the gap study put them later).
- [ ] A window's price by what its rows read: on one card a round with drafts reads 100 to
      260 experts from the file against 45 to 100 for a row, and what those reads push out of
      the tiers and hold up for the next steps is charged to no step, so the conductor keeps
      choosing windows that cost more than they bring. A price per row by the tier its
      experts come from (the study's `dense(rows) + chain(links) + misses x cost per tier`),
      or the block not resident where VRAM holds a third of the experts, are the leads.
- [ ] With the shared MTP driver (links of 1.4 ms, 0.9 with the draft vocabulary, where they took
      1.7 to 2.0 with the host's wait) Flash-Next answered at 39.0 tokens/s on the mean of twelve
      requests against 41.1 before (2026-11-17, alternating, loads 2 to 6; requests from 31 to
      59): the conductor verifies more of the cheaper drafts at a lower rate (the first link's
      54 to 58 % against 63 to 71 %). A window's price by what its rows read (below) is the lead.
- [ ] Drafts in the measuring tool (`qwen4exp_run` has none; the server and the runtime's
      tests measure them).

### Experts from the SSD
- [ ] **Prompts are bound by the link.** A wide step uploads every expert it picks that VRAM
      does not hold (the loan leaves 20-40 a layer), from RAM or staging, once per step: the
      32k prompt uploaded 428 GB in 96 s (4.46 GB/s, ~0.7 ms an expert; ~363 a layer = the
      ~250 ms a layer took), and read 170 GB from the drive beside it (1.74 GB/s of 2.4). Steps
      of 4096 rows make 8 passes; Strata reads it in 2 chunks of 21 248. With the MTP block's
      675 MiB of resident experts the 16 GB card cannot lend 8192 rows even at the loan's floor
      of a row's picks (without the block it can). The levers: a CPU share of the experts a
      wide step picks for few rows (Strata's prefill does this up to 3072-token chunks), less
      scratch per row (0.67 MiB), or the layer-major pass below; all model work, the owner's.
- [ ] **A short wide step costs what a long one does.** The cost of a step from the SSD is the
      experts it touches, not its rows: a prompt of 69 tokens (one step) took 3.2-4.0 s, a
      replay of 111 rows 18 s. Steps of 8 rows (the CPU serves what RAM holds) may be cheaper
      for short prompts; the conductor's prices of wide steps are measured at the start on one
      token repeated and say 270 ms for 64 rows. Being measured (`round6`).
- [ ] **The second ask replays the prompt.** Checkpoints of recurrent state stand only on page
      boundaries (128 cells) and a message boundary 1024 cells after the last one, so a prompt
      of fewer than 129 tokens resumes from 0 (3 cells) and a 623-token one from 512; from the
      SSD every replayed slice is a wide step (3 to 30 s). Strata resumes 7 tokens before the
      end (its newest path). A checkpoint at a prompt's exact end, or the newest path kept, is
      the cache's (`cache/checkpoints`) and the owner's decision.
- [ ] Strata uses the page cache beyond its budget: 12.2 GiB of the model's pages were cached
      at the end of its decode session (MemAvailable 12.8 GiB while loaded, Ares 20.7 GiB at
      `--ram 40G`). Equal budgets are not equal memory; Ares at `--ram 52G` is owed.
- [ ] Decode waits for the drive: 2-7 % of the picks come from the file (17-25 reads a token
      at 3.3-4.6 ms each, 27-90 MB a token), and the layers waited 64 ms a token of 77 on
      German prose. The read-ahead of predictions was used 50-63 % where it read, but few
      places qualify: of the experts in no tier a prediction named one layer ahead, the first
      places were picked 64, 58, 48, 39 % of the time (one row) and 55-61 % (windows of
      drafts), two layers ahead 4-15 %; and with experts of 3 MiB only one was read ahead at
      a time (two since a7c0375). A drive this slow at one or two reads (1.5 ms an expert
      alone, 3.7 in decode beside the others) and a lead of one layer (~1.5 ms) hide little;
      more RAM (see the page cache above) or pruning (opt-in) remove misses.

Reference (2026-11-24, the owner's link): Strata-HotRod, https://github.com/sangharshadhyeta/Strata-HotRod,
MIT, a fork of Strata 0.1.45 for Flash-Next IQ3_S on an RTX 3060 12 GB with 30 GB RAM and NVMe on
PCIe 3.0; one request at a time; the author's own measurements, many single runs. Claims 17 ->
50-70 tokens/s decode on short answers (38-54 after a 45k prompt), UD-Q4_K_XL 15-25. Where it
comes from: (1) stand-in experts, the big lever: a missing expert is replaced by the best expert
on the card when router weight x logit gap is within a self-tuned budget (maths 47/50 against
48/50, code 30/30 both; aggressive settings broke words and JSON and are off). A candidate for
the owner's dynamic pruning (item 6), measured for quality against the reference (the owner's
rule: differ for speed, never wrong). (2) A learned correction on the routers' look-ahead: of the
needed experts not on the card, the plain router held 77-79 / 66-69 / 55-58 % at 1 / 2 / 4
layers, the learned one 85-88 / 76-79 / 66-71 %; but read ahead only 29 % of the guesses were
used (95 % unused as read-ahead rows): precision is the problem, as Ares measured (3-8 % in
decode). (3) K/V from RAM past 64k with 32k on the card (as Ares's far pages); two I/O threads,
next layer first; a pinned RAM tier with PCIe balance (+9 %); swaps across layers (one run). For
Inkling 2 Mini it points to another fork, UnbiasedStrata (not read). To be read in its code
before stand-ins are built; measure in the SSD comparison (item 3).

Reference now in the open (2026-11-17): Run My AI's engine TokenWing, MIT, built on llama.cpp,
https://github.com/runmyai/tokenwing (`docs/how-it-works.md`, `docs/architecture.md`,
`docs/settings.md`); it runs Qwen3.8 Flash-Next (IQ3_S and UD-Q4_K_XL). Its README: expert reads
with O_DIRECT through 8 I/O threads and a priority queue; the routers of layers L+1 and L+2 on
layer L's hidden state (75.7 % of picks predicted one layer ahead, 67.2 % two, on Ling 3.0
Flash); a RAM cache per layer (`--moe-stream-window`), a VRAM cache that follows the text (up to
64 experts swapped every 8 tokens, `--moe-stream-vram-follow`); use counts halved every 64
tokens, the lowest leaves first; prompts layer by layer, so each expert is read about once per
prompt (0.98 to 1.00 reads per distinct expert). To be read in its code before items 3 above
(large prompt chunks) and the prediction's next step are built.
- [ ] A start on two cards that is much faster than the warm-up: reads of several neighbouring
      experts at once into a buffer (the sorted single reads reach 1.25 GiB/s where random ones
      reach 0.7 to 0.8; the gain of larger ones could not be measured on the shared drive), or
      answering once VRAM is filled and filling the RAM tier behind it until the first request
      (which then misses what RAM would have held).
- [ ] The MTP block's experts in the usage profile (they are filled in the order of their
      numbers at the start, as before).
- [ ] The slots per layer from the profile when the store is planned (`StorePlan::new` takes
      a heat; the runtime passes none, and the slots move between layers by need as it runs).
      Tried 2026-11-22 (`flash/ssd-stream`, reverted): VRAM by each layer's hottest and RAM by
      the heat beyond them, from the profile the warm-up writes; on UD-Q4_K_XL with 40 GiB
      decode then read 1.5 to 2 times as much from the file a token (87 against 145 MB on
      German prose): a short text's profile names too few experts per layer to divide the
      slots by. The layers do differ (95 % of a layer's picks take 166 to 345 experts), and RAM's
      slots never move between layers while the engine runs: a rebalance of RAM's slots by the
      heat as it runs, as VRAM's every 256 steps, is the form to try.
- [ ] The heat counts the steps that measure the price of a step at the start (token 0, about a
      thousand rows); the server's start drops it (`Qwen4Exp::serving`), the runtime itself
      still counts it for a model without a profile.
- [ ] Measurements owed (no turn of the shared GPU was left): the Q4_K_XL file (104 GiB), base
      against the predictions; prompts on the real file in wide steps (`--prefill 4000 --rows
      512`); more than one pair per configuration on a quiet machine.
- [ ] The prediction made only where it can pay: in steps of one row on these tiers no place
      qualifies, and the card still applies two routers per layer (not seen in the step
      times). A step could predict only now and then while no place qualifies, to keep the
      counts, through the board as wide steps are asked.
      (A first attempt with the prediction on the host was rejected: it changed the text and
      halved generation on two cards; the branch was deleted on 2026-11-23, the numbers are here.)
- [ ] The shared expert's output in the prediction's input (Run My AI: 11 % fewer misses on
      Ling): the part that routes has it only when it computes the shared expert itself
      (not with `--shared worker`), and the routed experts add into the same buffer, so it
      needs a copy on the step's stream. Not built; the prediction reads the router's input.
- [ ] Passes, what is left: with both devices on every layer the MTP block's chains after a
      pass agree with the reference's for 1 to 3 links where after steps of the own rows 5 to 7
      (the trunk's tokens are the reference's): not found. Uploads of the next layer's
      RAM-held experts while a layer's micro-batches run (only file reads go ahead now); a
      pause inside a layer's experts (a request waits for at most one layer's experts); the
      lent wide steps are still there for tile-mean K/V and DFlash, which passes do not take.
      TokenWing's form, read 2026-11-22
      (`src/llama-context.cpp` `lm_layer_major_ok`, `src/llama-moe-stream.cpp`
      `lm_lend_place`): only for a call of one sequence and more than one micro-batch; each
      micro-batch runs a layer's attention and router (its picks, weights and the normed
      expert input kept per row), then one expert graph runs the layer's experts for all rows
      in chunks of the micro-batch, and the layer's K/V is rewound and replayed per micro-batch;
      what a call keeps lives in places lent by the VRAM expert pool (about 60 KiB a row for
      qwen4exp's four streams). For Ares that is about 80 KiB a row (the four streams of the
      residual, the experts' input, their hidden values and output, the picks) against the 0.5
      to 0.7 MiB of a step's scratch that the lent steps take now, so a loan of the same size
      would hold eight times the rows; it needs the part's layers to loop over micro-batches
      with their own records and plans, and the mixer's and worker's exchanges per
      micro-batch. A step that a request can interrupt between layers needs the same state.
- [ ] A missed expert waits 2.3 to 3.2 ms for the file where the drive needs 1.4 ms: find
      the rest. Measured 2026-11-16 (`qwen4exp_run --decode-at`, its `drive:` line; two cards
      `--ram-gib 8`, load 7 to 12, the drive shared with other agents' builds and models):
      a demand read of one matrix took 2.0 to 2.1 ms at the median from its submission to the
      reader to its completion (8.9 to 9.4 at the 99th percentile), an expert's three reads
      2.8 to 3.6 ms on the mean, the miss from the record to its slot 2.3 to 3.2 ms, and one
      from RAM 0.1 ms (the upload). So the rest is in the reads themselves, not in the
      store: the three matrices of an expert are three requests (their tensors lie apart in
      the file), several misses of a step share the drive, and so do other users of it. On
      one card (8.6 reads a step) a matrix took 2.7 ms at the median and 28.6 ms at the 99th
      percentile. Not resolved further on the shared drive.
- [ ] Pruning, what is left: (1) the table for `similar` is made by the tool
      (`qwen4exp_prune calibrate` and `near`, minutes of GPU and CPU); the server only reads
      it, and refuses `similar` without it; (2) slices of a prompt (wide steps) are not
      pruned; (3) a decision per expert over a window (Strata's) instead of per row was not
      measured; (4) where a third of all picks are touched (Q4, `skip:1`), stand-ins with the
      table's scale (KL 0.42) and the router's renormalised (0.46) beat skip (0.50): skip
      renormalised at that rank was not measured; (5) the speeds were measured with every
      miss uploaded (`--serve stream`, for runs that repeat), not with the CPU serving them.
- [ ] Runs that should give the same bits do not (found while measuring pruning, IQ2_XS, two
      cards, the tool `qwen4exp_prune`, 1000 tokens teacher-forced after a prompt of 256):
      (1) with the CPU computing misses (`--serve model`, the engine's way) two runs of the same
      text differ by KL 0.0063, top-1 96.8 %, as large as Strata's tail skip; (2) with every
      miss uploaded (`--serve stream`) a prompt read in a wide step still differs (KL 0.0065):
      the rounds of a wide step `red.add` into the rows in the order their experts arrive;
      (3) with the prompt in steps of eight rows the next texts in the same slot after
      `Session::restart` differ (KL 0.0097 English, 0.042 code, read after German), and even
      the first text of a process differs a little where many misses are waited for (KL
      0.0001 German, 0 English, 0.0025 code, top-1 99.0 %). Owed: find what (3) reads that
      depends on the run (tier state, a slot's leftovers, the order in which waited experts
      land), and say in the architecture (`dev/architecture/`) which paths are reproducible; (1) and (2) are choices
      for speed that a test of quality must know.
- [ ] The first reading of a short prompt after other texts is slower with transit
      (35 against 86 tokens/s in one run); not understood.

### Host memory
- [ ] A release that does not wait for the running step, so the headroom stays at a few
      hundred MiB (it sits at its cap of 1.95 GiB after every load).
- [ ] A MoE model and a dense model on one device under memory pressure: the dense
      checkpoints are still freed on the watch's thread, the path that deadlocked.
- [ ] A log line when the tier shrinks.

### Cache, tests, kernels
- [ ] Device tests that do not depend on machine load
      2026-11-24: `qwen4exp_runtime` `a_request_cuts_in_between_the_pieces_of_a_long_prompt`
      failed in 2 suite runs under load 8 to 21 (once the long answer began 73 instead of 41),
      on builds with the runtime's cleanup phase 2; alone it passed 16 of 16 with every token
      the reference's, and in 6 alternating suite runs (before that cleanup 3, after it 3) it
      passed every time. Not tied to a change. The test should print the long answer's logits
      against the reference's when it fails, and judge a near tie by them (the owner's rule).
      (`steps_wider_than_eight_rows_match_the_reference`,
      `every_layer_of_wide_steps_matches_the_reference`,
      `experts_of_every_weight_type_on_the_cpu_change_nothing_*` fail now and then under load).
      2026-11-23, RTX 4070 alone at load 12 to 17: `steps_wider_than_eight_rows_match_the_reference`
      failed in 2 of 4 runs, on main's build as often as on the merged branches': in the Fast
      precision the last row's logits after the 256-row step differ by 0.13 to 0.215 (relative
      RMS) from run to run against the bound 0.2, with 882 or 883 experts picked and 50 to 61
      more rounds. The spread is the step's own, not the load's; whether the integer prompt
      kernel raised it is not measured. The bound or the spread is to be settled.
- [ ] The step scratch starts its buffers on a 128-byte line (as fixed for the 27B).
- [ ] Float experts beyond the f16 range are not caught in the many-row kernels.
- [ ] CPU path for the types without a written kernel (Q2_K, Q3_K, Q6_K, IQ2_XS, IQ3_S, IQ1_S,
      IQ4_NL and Q4_0 to Q8_0 as gate and up, the 256-value types as down, TQ*, MXFP4, NVFP4,
      Q1_0, the floats) is 3 to 8 times slower than the written ones; none is in a Flash-Next
      file at hand.
- [ ] NVFP4 many-row gate and up 1.4 times its neighbours on the RTX 4070; IQ2_XXS many-row
      at 8 rows 7 % slower on the RTX 3060 since the kernels for all types.

### Parity with the 27B (the owner's rule: what one model can, the other must)
Everything `user/features.md` lists for the 27B and not for Flash-Next is owed:
- Not owed (the owner, 2026-11-16): replies that are bit for bit the same from run to run.
  They differ slightly when the CPU computes missed experts, and that is accepted.

## Qwen3.8 27B (`qwen35`)
- [ ] Seen 2026-11-17 (main 4bec4b9 as well): with `max_tokens` 40 the first greedy reply to
      real_drafts.rs' German question after a start (IQ2_XXS, one card, fast kernels) and the same
      request again, its prefix from the cache, are different tokens (hashes 4d9a1bef2efe92d9
      and c1f7796b6753a9e5), with drafts and without; at 160 tokens all agree. Not looked into.
- [ ] The draft chain's cut costs a wait of the host per link where a bar can end it (the
      device thread reads the link's picks before it enqueues the next); since 2026-11-16 the
      chain is a step of its own and the round verifies only what pays of it, so the cut
      saves links only. A cut on the device (the links of a chain as kernels that end
      early) would keep the saving without the waits.
- [ ] A generating step of more than 16 rows still changes kernel family (three sequences
      drafting at once): the last gap in "text with drafts is the text without" (which holds
      since 2026-11-16 for single sequences in every placement and K/V format). Texts under
      several concurrent requests were not compared; the replay from the prefix cache rounds
      differently from single rows.
- [ ] Layout choice that is stable: it rests on the fastest of three probe rounds; a link of
      the draft chain costs 1.49 ms where 1.44 ms is possible; the choice changes between
      loads and with the MTP block; the 0.8B decodes 5 % slower than it could.
- [ ] `qwen35_run` aborts a direct prompt run with `--kv q5,q5` (reserve for partial states).
- [ ] Float matrices under the fast precision use f16 sums (0.04 to 0.06 % rms) where the
      F32 matrices of an IQ4_XS file were exact before: decide with a KL measurement.
- [ ] VRAM and quality against the fork at equal settings: never measured.
- [ ] Prompt on the RTX 3060 alone is 3 % behind the fork.
- [ ] The model's drafts for the thinking rows of a request that has to call a tool;
      drafts while a rule is bound.
- [ ] Sampler: a row with no finite logit reports `-0.0` where the reference says minus
      infinity; deciding a level got slower (35 to 42 us); a list of the set's tokens after
      the first cut; on two cards most of the sampler's time on the RTX 3060 is the exchange.
- [ ] F32 tile kernel is 2.5 times slower than BF16 (untuned).

## Drafters (every model)
- [ ] Blackwell (the owner, 2026-11-22: an RTX 5060 Ti is ordered for the test machine, next to
      its RTX 4060 Ti; an MCIO Gen5 splitter makes the x16 slot two x8: the 5060 Ti at Gen5 x8, the
      4060 Ti at Gen4 x8, twice the link it has today, so the link-bound numbers measured there at
      x4 must be taken again). Two cards of two generations there: placements across Ada and
      Blackwell, peer access only if that machine's driver allows it. Both in place in about a
      month; until then the owner can swap the 5060 Ti in for the 4060 Ti for a while to test. A
      default build has no sm_120 code and refuses the card (only tested classes, 2026-11-23). When
      the card is in: a build with `ARES_CUDA_ARCHS=89,120` there and the whole GPU suite on it;
      when it passes, 120 joins `TESTED` in `crates/cuda/build.rs` (and the names in
      `dev/architecture/devices.md` "GPU classes", `user/features.md`, `README.md`); then Blackwell's own paths
      where they pay, measured: FP4 tensor cores
      for NVFP4 and MXFP4 weights (Strata 0.1.65 names `dense_fp4_tc` and `moe_fp4_tc`), FP8; the
      prefill path per type measured there as on every card; and Ares against Basalt (Blackwell
      only, so the first chance to compare) and Strata on the same card.
Reference to read (2026-11-18): Basalt, https://github.com/jesdga95/basalt (MIT, a fork of Strata,
kernels from Infernix; NVIDIA Blackwell only, so it cannot run on these cards): Qwen3.8 Flash-Next
with MTP, Qwen3.8-27B with DFlash 2 and DeepSeek-V4.1-Flash with DSpark, drafting on by default;
OpenAI and Anthropic routes, a router, images, documents (PDF, text) and video (NVDEC for
Flash-Next); experts that do not fit in VRAM in RAM; K/V in int8. Its README gives no speeds.
- [ ] A classic draft model as a fourth drafter kind (llama.cpp's `-md`): a small model with the
      same vocabulary drafts with its own forward passes and its own K/V; chosen by the same price
      rule as `ngram`, `mtp` and `dflash` (the owner, 2026-11-17). Later (the owner, 2026-11-17).
- [ ] Roles of the cards (the owner, 2026-11-17; Strata has Main, Helper and Off per card): main
      cards hold the model and its expert cache; an aux card holds drafters (a draft model, DFlash,
      an MTP block) and, for a mixture of experts, its free VRAM as one more expert tier between
      the main cards and RAM. The option's name and form go to the owner before it is built.
      Measured 2026-11-22 (`dev/architecture/devices.md`, "What the second card is for"): for the 27B the
      second card is worth most as room for K and V (the attention heads by shares of their own,
      as the fork's `--attn-split`: 2.1 times the context, 3 % slower decode at 1000 cells and
      9 % at 32 000). Built as how the plan works, with no option (the owner, 2026-11-22):
      `AttentionShares::ByContext`, heads with the weights as far as the wanted context
      (`--context`, or the model's own) fits, moved only as far as it needs. The rest of the
      roles item (what the aux card is called, drafters on it) waits for the owner.
- [ ] Flash-Next's second card (measured 2026-11-22, `dev/architecture/devices.md`, "What the second card is
      for"): on every layer it is worth most (twice the VRAM tier); as an expert tier only (no
      dense share, `core_share` 1) it decodes as fast and reads warm prompts 27 % faster than the
      share chosen today from read times. Proposed: the plan measures at load a wide and a decode
      step with and without the dense share and keeps the faster; goes to the owner first (it is
      the card-roles design), not built.
- [ ] Drafters on the second card: not built. A chain that runs on the slower card is slower by
      what that card is slower, and nothing runs beside it in a step that only drafts; it pays
      only with a pipeline that drafts the next window while the main card verifies this one
      (Strata's `--pipeline-windows`).
- [ ] DFlash2 trees (steps 9, 10): tree verification in the blocks, tree fill from the lattice,
      node allocation over sequences. Study: `/var/tmp/ares-dflash-study/REPORT.md`.
- [ ] DFlash2 on Flash-Next split by layers: the first device's taps (or its share of the
      fusion, `rows x n_embd`) must reach the head's device; the hand-over lane could carry it.
- [ ] The DFlash ring in the prefix cache's checkpoints (42 MiB a slot in f16): a sequence that
      resumes from the cache drafts from the resume point's context only until its ring fills.
- [ ] Sampled DFlash drafts: the walk is greedy; keyed noise in the scores (as the MTP block's
      picks have) would raise sampled acceptance (study: 2 to 8 %).
- [ ] The 27B with the DFlash drafter at the default split and a 32k context in `qwen35_run`:
      the tool sizes its pool without the drafter and the device's ledger refuses the drafter
      (the server's plan counts it).

## Prompt cache (every model; RAM first, then SSD, as in the owner's fork)
- [ ] Reuse of K/V blocks away from the prefix, as CacheBlend does, opt-in like pruning (the owner,
      2026-11-18; approximate: the output diverges, measured and bounded by a dial, off by
      default). For a conversation whose earlier part changed (an older tool result cut, an
      edited message, a compaction): the attention layers' pages of the unchanged blocks are
      taken again, K turned to its new position (RoPE is relative); Flash-Next's QSA indexes
      blocks already, so its pooled keys go with them. The hard part is the recurrent state (three
      layers in four): take the old checkpoint after the change (the gates forget old tokens),
      or recompute a bridge of N tokens after the change, or recompute the tokens whose K/V
      deviate most (CacheBlend: about 15 %). Measure first: an early message changed, exact
      recompute against each way, KL of the next tokens and how far the reply parts; then a dial.
- Not wanted (the owner, 2026-11-22): saving and loading a conversation's state by name
  (llama.cpp's `slot-save-path`); the prompt cache covers it.
- [ ] Locking a page's buffers costs about 0.5 ms per MB when RAM grows (a page of the 27B is
      4.6 MB): lock ahead in the background while a step runs.
- [ ] A checkpoint that later requests left behind at deeper page boundaries is not written
      when it leaves the index; one that leaves with its page, after the deeper ones of its
      text left, is.
- [ ] An unload or a stop waits for the drive to take what only RAM and the devices hold (on
      the btrfs volume here, compressed and busy: 20 to 100 s for 1.5 GB). What only RAM holds
      could be written after the devices are released, by a writer that outlives the engine.
- [ ] A host test of a place whose memory a device keeps locked after it was given up: the host
      executor frees it at once, so the fault where pages in RAM left no room in `--ram` for a
      checkpoint's lock was found on the devices only.
- [ ] One size limit for all models on a directory; a test of the full drive; a layout that
      does not depend on the split.
- [ ] Cache policy: resume point at the prompt's end, memory of reply tokens, marks, warm pass.
- [ ] On `/v1/responses` a second turn took 1 of 35 prompt tokens from the cache where
      `/v1/messages` took 18: the history is rendered differently somewhere. (2026-11-22: on
      Flash-Next and on the 27B the two routes take the same, 83 of 99 and 89 of 105 in
      `real_responses` and `real_messages`; the 0.8B was not run again.)

## Server and API
- [ ] Tool calls: a generic format for other templates; recovery of imperfect calls (T05);
      string constraints `pattern`, `minLength`, `maxLength`, `format`; forced tokens as
      drafts; a guard for `ignore_eos` with a complete grammar; the end-token rule for a
      schema with thinking; llama.cpp's `json.gbnf` costs 6.8 ms per mask.
- [ ] Responses: stored responses (A05), WebSocket (A06); the chat route's sampler fields
      are refused there.
- [ ] Router: load and unload times in the docs; "no room" for Flash-Next is recognised by
      message text; a plan cannot be asked without loading; two elastic models on real devices.
- [ ] A VRAM reserve for other
      programs (V04), container image and service unit (X09), key roles and a reload of the
      key file (X02), TLS (X03).

### Claude Code against Ares (2026-11-22, Claude Code 2.1.301, print mode)
- [ ] Flash-Next with both devices on every layer (the driver with peer access, 2026-11-22
      14:27, load 2 to 6) read the same first prompt at 40.9 tokens/s (the split by layers 61
      at load 13) and answered at 3 to 6 tokens/s, with 5 of 55 and 7 of 127 drafts accepted;
      the MTP block's first link took 1 of 6 and 3 of 12 where the split by layers took 25 of
      41, 29 of 45 and 56 of 78 on the same prompts. Few samples, but the MTP block looks badly
      fed in that placement; not looked into. The made-up model's MTP test that failed once in
      the merge check of 2026-11-22 drafted badly in the same placement (`Shared`).
- [ ] A new session on Flash-Next reads the whole 16 thousand tokens again: the root checkpoint
      ("Next", item 2). The template writes the tools first (12.7 thousand tokens) and the
      system prompt after them, whose first line is Claude Code's billing line: 2.1.301 writes no
      `cch=` there but `cc_version=2.1.301.<3 hex>`, another suffix for another first message
      (`66d`, `e64`), so on the 27B a new session took 12672 of 16287 tokens from the cache and
      computed the rest. A stamp that is pinned must include that suffix (item 1).
- [ ] Claude Code gives a tool's input back with the defaults of its schema added
      (`{"replace_all": false, ...}` first, where the model wrote `file_path`, `old_string`,
      `new_string`), so the history parts from the cached sequence inside the last reply. On
      Flash-Next with one checkpoint place (`--ram 8G` without `--cache`) that turn replayed all
      16512 cached rows: 373 s to the first token for 242 new tokens. A checkpoint at every
      message boundary ("Next", item 2) brings it to the rows of the reply. Writing a call
      that comes back without the arguments that equal their schema's `default`, in the
      schema's order, would match what a model usually writes; it changes the history the
      client sent, so it is the owner's decision (not built).

## Inkling 2 Mini (`inkling2`; the owner, 2026-11-22: worth supporting, and looped transformers matter more and more)
Thinking Machines' Inkling-2-Mini (Apache 2.0), GGUF by Unsloth: `unsloth/Inkling-2-Mini-GGUF`
(UD-IQ1_S 7.2 GB to UD-Q8_K_XL 21.3 GB, UD-Q4_K_XL 16.4 GB; vision tower `mmproj-*`; its own
DSpark drafter `dspark-Inkling-2-Mini-Q8_0.gguf`, 0.4 GB). Notes: `/var/tmp/ares-survey/inkling/`.
- What it is: 24 decoder layers with a mixture of experts (8 of 128 per token); the middle 12
  layers run three times with shared weights (48 layer applications per token); attention a
  hybrid of local and global layers; 26.2B parameters, 3.4B active (5.4B with the loops); context
  up to 1M tokens; images through a vision transformer into the same hidden space; NVFP4 and
  MXFP8 numerics (the experts carry per-expert NVFP4 scales).
- Two layouts of the same tensors: the root files in Unsloth's looped layout, `Mainline_Support/`
  in mainline llama.cpp's `inkling2_loop` layout (llama.cpp PR 32694; PR 32704 keeps the
  per-expert scales on the second and third passes, without it perplexity 130 instead of 7.65).
- Its drafter is a 3-layer block drafter (block size 8) with a Markov head, in DSpark's format
  (`general.architecture = dflash`); Ares refuses DSpark by name today (shared layer step 8).
- [ ] Study first: the two layouts, the loop (K/V and recurrent state per pass or shared, how a
      pass is told apart), local and global attention, the expert scales, what of the shared layer
      and Flash-Next's expert store it takes as it is; then the model, its drafter (DSpark), its
      vision tower. In the order (the owner, 2026-11-22): after the Flash-Next pieces in work
      (CPU kernels, SSD streaming), one piece at a time.

## Whole features not started
- [ ] Documents as input (PDF, text), as Basalt takes them next to images; video stays backlog.
- [ ] Audio, video and Gemma 4 are backlog: none of them blocks the first release (the owner,
      2026-11-16).
- [ ] Jev mode: `/v1/systemone` exists and answers 501; `Engine::decide` on the real runtime. Later.
      Reference (2026-11-18): Flide, https://github.com/OmniJev/Flide (Apache-2.0, Python and
      Triton, a server for the Jev `/v1/systemone` API, paper arXiv:2610.42148): "exact state
      forking", one state forked into the questions, each answered in one forward pass, the
      answer the one the model gives the question alone; a gated delta-rule kernel (from
      flash-linear-attention), so recurrent state is forked as in our models; Qwen3.5 and Gemma 4
      chat models and the OneJev / Open-Jev decision models; several requests in one body, up
      to 256 options a question, a session cache (`--session-cache-gb`), `--batch-invariant`.
      Its numbers (one H200, OneJev1.6-0.8B): one email of eight questions in 10.0 ms, 10,000
      in 28 s; "up to 9.5 times faster than vLLM and SGLang".
      Its groundwork in the runtime, kept by the cleanup for it (2026-11-24): the decisions of
      `ares-sched` (`Want::Score`, `PartKind::Score`, the option rows first in `plan`), the
      conductor's arms for them, and `Engine::decide`, which answers Unsupported today.
- [ ] Untested GPU classes go to Vulkan (the owner, 2026-11-23; was "Turing. Later."). CUDA builds
      and opens only the tested classes, Ampere (8.6) and Ada (8.9) (`cuda/tested-archs`); Turing,
      Pascal, Hopper, data-centre Blackwell, Jetson and every other GPU are to run through the
      Vulkan backend once it runs models. What was not tuned on CUDA goes there too.
- [x] K/V of a running sequence beyond VRAM (P16). The owner, 2026-11-22: without `--context` a
      model gets its own context always, and what of it does not fit in VRAM overflows into RAM.
      Built for Flash-Next (2026-11-22, `kv/overflow`; "K and V beyond VRAM" in `dev/architecture/flash-next.md`).
      Open:
  - [ ] The 27B: not built. Its 16 dense layers read every cell, so K and V in RAM cross the link
        once per token: 5.4 ms per 1000 cells on the test machine's x4 link, about 1.4 ms on a x16
        link (`far_attention_bench`); its context beyond VRAM would run at a few tokens per second.
        On the development machine 104.5 ms a step at 32 768 cells from RAM on the 4070, 435 ms on
        the 3060. Its heads move to the second card first, and with `q4` two cards hold 482 707
        tokens.
  - [ ] Steps of the parts' own scratch (pieces of a prompt while another request waits) read the
        far entries in place, row by row; only steps on lent VRAM and passes (in their loan) read a
        copy. A copy in the own scratch would cost VRAM for good (the far pool's share of one
        layer, 254 MiB at 262 144 cells).
  - [ ] The RAM level of the prompt cache copies a far page when it leaves the devices; it could
        take over the page's memory instead.
  - [ ] The share of the room for K and V in VRAM (a thirty-second, `plan::NEAR_SHARE`) is the old
        rule's, not measured against other shares.
  - [ ] Pages are near or far by their number: the newest pages of a long sequence are the far
        ones, so its newest cells (often picked by the indexer) are read over the link. Keeping the
        newest pages in VRAM needs pages that move between the pools.
  - [ ] **Lent steps with the CPU's share** (found by `kvfix`, 2026-11-24; main's, not the far
        pages'): on the made-up model (`prompt(300, 7)`, the BF16 test file, steps of 32 rows on a
        loan, placement One) the first token's log-probabilities are 58 -1.80..-1.85 and 22
        -2.20..-2.25 in most runs and 58 -2.14, 22 -2.10 in the others, the same values every time
        (the token after it follows: 39 at -1.89 or -1.74): two states, not a spread. Main 852cc02
        8 of 12 runs (7 answered 22), main ae836d2 One 6 of 8 and by layers 1 of 8, this branch
        alike with and without far pages; with every miss uploaded (`Serve::Stream`) 0 of 16, in wide steps of the
        own scratch (about 32 experts on the CPU a step, against 370 on a loan) 0 of 24, with every
        miss the CPU can take on it (`Serve::Cpu`) 5 of 16. The rows' int8 input of the CPU against
        the GPU's f16 moves the first token by a few hundredths, not by 0.3: something of the CPU's
        share on lent VRAM is computed differently in some runs. The reading of `misses.rs`,
        `hybrid/wide.rs`, `cpu/mod.rs` and `transit.rs` found no cause; a diagnostic that prints the
        first tokens' log-probabilities per run (`zz_dbg_lent_gaps`, with the serve mode and a log
        line per wide layer's CPU share) is in `/var/tmp/ares-kvfix/` (`exp5-all.patch`).

## Weight types (`moe/q4-load`, 2026-11-17)
Every tensor of both models takes every weight type the shape allows (the audit of every place
that refused a type is in the branch's notes; `dev/architecture/flash-next.md` "Files in any type" and "A layer
with fewer slots in VRAM than a step picks"). Left:
- [ ] Gate and up of a layer's routed experts in two types are refused (the expert kernels read
      both with one decoder): the owner wants it (2026-11-17): two launches where the types differ, the fused kernel where they agree.
- [ ] A router of a type other than F32 is decoded to F32 at load, four bytes a weight (files
      keep it F32); its kernels (decode, prediction, wide steps) would take the type.
- [ ] Where VRAM holds fewer slots than a step picks, every layer's slots are at its floor, so
      the heat no longer moves slots between layers; the floors in RAM cost `--ram`: UD-Q4_K_XL
      needs 9G on one card and 10G with both cards on every layer (8G by layers).
- [ ] The DFlash drafter's float matrices take the f16 tile path above eight rows without the
      range check of Flash-Next's (`WeightRange`).
- [ ] Shapes, not types: rows whose width a type's block does not divide cannot be stored in it
      (Flash-Next's expert down projections of 640 values, the n-gram table's rows of 160, the
      image tower's `ffn_down` of 4304); the 27B's tensor split cuts in units of 256.

## Optimising (a store of levers; taken by the owner's reasons, not as a phase)
Measured per phase on a quiet machine against the fork and Strata, the largest first. Known levers:
- [ ] A device layout for the codebook i-quants (IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_*):
      in the file's layout their tile kernels stay bound by the load unit (RTX 4060 Ti, 1024 rows:
      IQ3_S int8 56 / f16 44 TOPS, IQ3_XXS 58 / 49) where the device layout's kernels take 66 to 70
      TFLOPS for the types that have one. The 27B UD-Q4_K_S has 0.75 GiB of them, Flash-Next
      UD-Q3_K_XL 22.7 GiB of IQ3_XXS experts (the expert kernels read the same file layout).
- [ ] The 16-byte pieces of `ares_ld_words_a2` elsewhere: only the large shape of the tile
      kernels takes them. On the RTX 4060 Ti they made the medium shape of Q3_K and Q6_K slower
      (64 rows: 28 against 22 TFLOPS), and in the expert kernels (`expert_types_bench`) they took
      gate and up of Q6_K, Q8_0, Q5_0 and TQ2_0 5 to 17 % faster but down of Q3_K, Q6_K, Q8_0,
      Q4_0, Q5_0 4 to 15 % slower. A choice per kernel, measured on the capped cards as well.
- [ ] Flash-Next's wide steps take one form of activations for every dense matrix; where the
      types' forms differ (a type with no int8 tile kernel, or measured paths that differ) every
      dense matrix runs f16. The owner (2026-11-22): a form per matrix, int8 on a capped card as
      the 27B's ("Decided by the owner").
- [ ] Attention over long contexts: llama.cpp PR 32407 (merged 2026-11-17) rotates the order in
      which blocks read K/V tiles so that blocks sharing K/V hit L2 together: prompts of the 27B
      on an RTX 5090 at 131072 cells 1290 -> 1822 tokens/s (1.41x), 1.25x at 65536, 1.07x at
      32768, nothing below. Ares' attention for prompts and wide windows at long contexts.
      Prompts: the new prompt kernel (see "Against the owner's production") is bound by its
      instructions, not by K/V traffic (about 50 GB/s of the RTX 3060's 360 at 98k), so the order
      of reads is not its lever; wide windows (drafts) still run on the decode shapes.
- [ ] VRAM on loan to the expert tier: the MTP block's 675 MiB while a long prompt runs (gap
      study `/var/tmp/ares-survey/gap-study-2026-11-17.md`). K/V pages a conversation does not
      need yet (Strata's `--kv-grow`) need no loan since `kv/overflow`: the K and V of a context
      take at most a thirty-second of the room for slots in VRAM, the rest lies in RAM and is
      locked as sequences reach it.
- [ ] Checkpoints of the recurrent state in int8 (SGLang's `mamba_checkpoint_pool.py`): a quarter
      of Flash-Next's 112 MiB, which matters because a full tier leaves one place; after a KL
      measurement (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
- [ ] Steps that overlap the host's scheduling with the GPU (vLLM's default now), CUDA graphs:
      first measure the gaps between kernels and the host waits (grammar mask, draft chain) (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
- [ ] The lookup's corpus kept over a restart; the lookup's confidence calibrated on German text;
      what `cache_prompt: true` should do (the owner to decide) (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
- [ ] The fused read of a Q8_0 mixer (UD-Q4_K_XL) takes 43.8 us on the RTX 4070 where BF16 takes
      24.9: it decodes byte by byte (a load of four codes at once would cut it); sixteen rare types
      read through the decoder chosen at run time; the float read is 2 % slower than before the
      types (5400 against 5112 instructions).
- [ ] An expert from the SSD is three reads (its matrices lie apart in the file): 2.8 to 3.6 ms
      where the drive needs 1.4 ms per read. One read per expert (a layout of the SSD level).
- [x] Long prompts layer by layer (2026-11-23). [ ] Their read-ahead; the profile's fill in
      large ordered reads.
- [ ] Few-row kernels for windows and trees; attention for wide windows; the F32 tile kernel
      (2.5 times BF16); NVFP4 and IQ2_XXS gaps; the image tower's row kernels fused.
- [ ] The sampler's exchange on the second card; a layout choice that is stable between loads;
      the MTP link (1.49 against 1.44 ms possible); DFlash's drafts-only step (4 to 5 ms).

## Vulkan (last)
- [ ] A backend-free crate and the `ares-gpu` facade; expert and Flash-Next kernels; both
      models on Vulkan in every placement; attention for wide windows (2 to 6.5 times slower
      than CUDA today); the delta rule's f16 state rounded per row; the sampler's `-0.0`.
- [ ] Vulkan's draft carry (`SamplerKernels::{shift,keep,commit}_hidden`) has no CUDA twin any
      more (no caller there, removed on `cleanup/cuda`): drop it or give both a caller at parity.

## Measurements owed on a quiet machine
- [ ] On the test machine (RTX 4060 Ti 16 GB with gpu-unlock, 62 GB RAM, a
      Samsung 990, Strata 0.2.15 installed, good internet for downloads): Flash-Next against Strata
      with the same file (UD-Q3_K_XL there), a large RAM tier, one card of another class.
- [ ] Real models with 1, 2 and 4 concurrent clients against the fork and Strata (Strata saw four
      requests on a 12 GB card with experts outside VRAM 11 to 24 % slower in sum: parallel
      requests buy waiting time there, throughput on the 27B) (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
- [ ] Flash-Next with the owner's real limit (a RAM tier of 24 to 26 GiB), one card and two.
- [ ] The SSD cache's times to the first token (taken under load so far).
- [ ] Flash-Next with its MTP block once speculation pays; target above 56 tokens/s.
- [ ] Before a release: both models against the fork, Strata and TokenWing (Flash-Next), speed,
      VRAM, quality.

## Against the owner's production (2026-11-23, the 27B UD-Q4_K_S on the RTX 4070 + RTX 3060)
- [ ] Long prompts: 32k 31 % and 98k 57 % behind the fork. About 70 % of the heads sit on the RTX
      3060; the fork puts 75 % there and still reads 98k at 752 tokens/s, so Ares' attention over
      long contexts on that card is the gap (its flash attention against ours; llama.cpp PR 32407 is
      in "Optimising").
      Built (`attn/long-context`): a prompt kernel of 128 vectors (`kernels/attn_prompt.cuh`) with
      integer scores over quantized K (and the tile mean's dot products from the same int8
      products), P V summed in f16 per chunk of 32 cells, and phases in which decode and the tensor
      cores overlap; the planner cuts a prompt's tiles into pieces that fill the kernel's waves and
      orders them so the blocks running at once share K/V in the L2 cache. One attention layer, a
      step of 256 rows at the end of 98 304 cells, K and V in `q5` (the fork `q5_1`): RTX 3060 with
      3 KV heads 79.9 ms before, 21.0 ms now, the fork 13.5 ms; RTX 4070 with 1 head 11.3, 3.0 and
      2.1 ms. At 32 768 cells 26.6, 7.4 and 4.6 ms on the 3060. Output closer to the reference
      than before (integer scores). What is left: the kernel reaches about 45 % of the int8/f16
      tensor rate of the capped cards, the fork about 67 % (its O in f16 over the whole range and
      the query in registers; Ares keeps O in f32, 128 of 255 registers, two warps per scheduler
      whose latencies do not overlap). O in f16 over the whole range was measured: 7 % faster, 8 to
      10 times the error already at 1.4k cells, growing with the length (the owner's call, not
      built). Tried and dropped: a single K tile (2.5 % slower), the decode's lanes on rows (q8
      3 %), cells shared out over one wave of blocks reading several pieces (15 % slower: no L2
      sharing), P V with f32 sums (11 % slower), 8 vectors per warp in 16 warps (no faster).
      End to end at these settings (the bench of `/var/tmp/ares-tools/prod-ab/`, prompt suite, two
      rounds alternating with main, load 1 to 5): 2k / 32k / 98k 1131 / 1000-1002 / 709-710 tokens/s
      against main's 1082-1103 / 653-655 / 324 and the fork's 840 / 944 / 752: 32k is reached, 98k is
      6 % short. The state slots of the longer prompt tiles cost about 10 MB of VRAM on the 4070 and
      6.5 MB on the 3060 (the 27B three ways: 874 / 1754 / 1938 pages against 890 / 1770 / 1946).
      Levers left, the owner's call: the 3060's 78 pieces run in 2.8 waves; 5 pieces a tile (7.0
      waves) would save about 6 % of its attention for some 25 MB of state slots on each card, the
      VRAM-tight 4070 included; heads by speed need room on the 4070 these settings do not leave
      (other `--shares`); O in f16 (above). Within the kernel: a design with warps that only decode
      and the others only multiply (more warps than the f32 accumulator leaves registers for today).
- [ ] Images on these settings, and a day of the owner's own use as a second llama-swap entry.

## Parallel requests against the owner's production (2026-11-23, `bench/parallel`)
What Ares does better: three or more requests (a place each, at the same context), priorities, the
system prompt two agents send at once (the second waits for the first's announced pages and shares
them), cached turns, one request alone. What it does worse, and why:
- [ ] Two prompts at once are read one after the other (shortest first, the order that gives the
      lower mean wait); the second then takes turns with the first's generation, now at the
      floor and not at half. The agents' turn with a 5.8k tool result sent at once (main ae836d2, two
      rounds, cold and warm): the second agent's first token 17.0 / 17.4 / 17.0 / 19.2 s before, 16.6
      / 14.7 / 18.5 / 14.9 s now (fork 9-17); the first agent's turn ends 3 to 5 s later (its 200 tokens
      now at the floor), so the two turns' ends together are even to worse: for replies this short
      the old half was as good. Whether the floor should rise for a request near its end (its tokens
      wanted are known) is open. Sharing a piece between two prompts would delay the first
      without bringing the second forward.
- [ ] A turn that continues at 17-23k reads 576 tokens/s (fork 880; Ares 1020 from the start of a
      prompt): the attention at depth (the long-context piece).
- [ ] Two generating sequences share a step's draft rows by their chance (`window::choose`) up to
      the 8-row edge of the price table (8 rows 31.0 ms, 9 rows 37.1, 16 rows 47.8): the step stays
      at 32-35 ms (alone 32), one request gets 1.6 tokens a step and the other 2.2 to 3.5. The fork's
      step grows from 33.6 to 40-44 ms and keeps each slot's drafts (2.0-2.7 tokens a step each).
      Ares' slower request is as fast as the fork's slower one and the other is faster (quiet machine:
      49 + 67, 52 + 68, 85 + 58 against 49 + 50, 52 + 55, 57 + 59 tokens/s), so this is not behind;
      a cheaper 9-to-16-row step would lift the slower one.
- [ ] Flash-Next (IQ2_XS, both cards, `--ram 8G`, 40960 cells; the fork has none): two at once
      give 44 to 54 tokens/s in sum (alone 19 to 40), but 3 or 4 at once stay at 32 to 35 with
      `--parallel 2` and with `--parallel 4` (more sequences pick more experts that RAM and the SSD
      bring). An 8k prompt beside a generating request takes 295 s against 25.6 s alone (11.5
      times), the generation keeps 20.6 of 28.1 tokens/s, its longest gap 3.0 s: alone the prompt
      runs in 9 wide steps on lent VRAM, beside it in pieces in the model's own memory that each
      read the experts again (5.3 million expert picks against 51 thousand, 240 s of expert waits
      against 32). The layer-major passes now on main (574d741) read that prompt alone in 17.7 s
      against 26.6 (same machine state) but not beside a generating request (523 s against 496 s,
      machine load 20 to 55 from builds; Flash-Next's decode fell to 9 to 15 tokens/s under it):
      the pieces next to generation are still narrow and read the experts again.
      Done for the prompt (2026-11-23, `sched/parallel`): a long prompt beside generation is read as
      a pass that pauses for the rounds (`Model::pauses_wide_steps`), at most half as wide as the
      slots can lend for. On the test machine (one RTX 4060 Ti, IQ2_XS, `--ram 24G`): 8k beside a
      generating request 61.8 / 62.1 s before, 16.6 / 16.8 s now (alone 10.2); 16k 113 / 117 s
      before, 29 / 31 s now; the generation keeps 14 to 19.5 tokens/s (26 before), its longest gap
      0.7 to 1.2 s. On the two cards at `--ram 8G`: not measured (the local cards were busy).
- [ ] Flash-Next: rounds between the pieces of a paused pass can stall the expert store when the
      pass borrows all the slots can lend ("a step stood still for 30s while the host held back the
      next layer ... the store waits for 7 to 9 experts and is not settled": a 21k-row pass opened
      beside a generating request, 3 runs of 4 on one card; a pass opened alone and then joined by
      generation did not stall in 2 runs). A loan leaves a layer `top_k + vram_reserve` slots
      (`scratch_floor`); windows of 4 rows and the MTP chain then need more distinct experts than
      that. The conductor now lends a pass beside generation at most half; the store itself should
      bring a step's experts through transit or the CPU whatever the loan holds.
- [ ] Flash-Next's greedy text differs from run to run of the same binary (test machine, IQ2_XS: 4
      requests of 4 differed between two runs of main, alone and beside), so a change's text cannot
      be compared by hash on this model; presumably the CPU and GPU paths of the experts, which
      timing chooses.

## Cleanup (the owner, 2026-11-23: the code must be clean although agents wrote it)
The first push to GitHub comes after the whole cleanup (phase 1 and phase 2) and the restructured
docs (the owner, 2026-11-23); then the unpushed history is scrubbed of the test machine's address
and home paths (2 and 6 commits) and pushed without force.
Audit of main e6d7676 by area, reports and the plan in `/var/tmp/ares-audit/` (`PLAN.md`). The core
designs are sound (the shared blocks, one reader and core for all routes, one record writer and
reader, an independent host oracle); the debt is where the code grew: copied bookkeeping beside
each new path, replaced designs left behind for a bench or a test, hand-kept lists, test helpers
copied into every file. About 4,000 to 5,000 lines can go without changing behaviour.
  - [ ] At Vulkan parity: Vulkan's draft carry (`draft_shift`/`keep`/`commit`, which CUDA no
        longer has, and its test), its delta-rule shape sweep, and its copies of the activation
        layout (`vulkan/src/layers.rs`) and of the quantization (`vulkan/tests/mmvq.rs`): use
        `ares_quant::act` there.
  - [ ] The owner to decide: whether `examples/launch_cost.rs` and `q4_bench.rs` (results named
        nowhere in the docs) are retired, and `matvec_bench` folded into `few_rows_bench` and
        `weight_pass`.
  - [ ] Small, not done: the examples' own copies of `Rng`, `f32_bytes` and `gpus` (attention,
        expert layer, expert rows, expert types, sampler steps); the runtime's `store_hidden`
        quantizes 32 hidden values as `ares_quant::act::Group` does but keeps its own sums per
        16 (a hot CPU path, left as it is); the kernel side's type lists per module (an X-macro
        list, optional).
  - [ ] In the pass over all kernels: the many-row down kernel's branch of the ordered sum, which
        the host no longer takes: without it nvcc gives ErQ20's kernel of 16 pairs 64 registers
        instead of 70 and it was 5 % slower (`kernels/expertrows_down.cuh`), so it stays until a
        layout without it is measured.
  - [ ] Moved to phase 2: one `time_best` for the five timing loops at load (they decide
        measured choices, so a merge is a change of measurements), the names (`Eng*`/`EngMm*`/
        `Er*`/`AresCodes*` decoders, `q4` for qwen4exp beside the Q4 types, tile names `_n1` for
        8 rows) and the splits of `experts.rs`, `kernels.rs`, `mm.cuh` and `mmr_codes.cuh`.
  - [ ] The loaders (finding 1: `load.rs` and `qwen4exp.rs`, their copied tail, the qwen35
        memory estimate in the server), with them the names `qwen4exp::Split` (a saved spread)
        and `load::watch` beside `Models::watch`: after the model crate's phase 2 merges.
  - [ ] Other crates' parts of the audit (`server.md`), not this branch's crates: `ToolFormat`
        and the thinking format written in `ares-tokenizer` and `ares-api` (and copied in
        `ares-model`'s `chat.rs`), three JSON value types, tools as raw JSON across the API,
        the order rule of the samplers and the mirostat range checked in the runtime as well,
        `LOGPROBS_MAX`/`BIAS_MAX` beside the API's limits; the mask-bit test three times
        (grammar, sampler), llama.cpp's defaults in the sampler and the runtime; test-only
        public code (`Breakers::of_tokens`, `keeps_history`/`opens` computed and dropped).

## Owed on two cards (the owner's cards are production; run when the owner frees them)
- [ ] The trial of the second card's role costs 3.2 to 3.7 s at load (4.1 to 4.2 s of trial, of
      which 1.6 s opening the first role's parts with the prefill paths measured): each role opens its
      parts and runs a wide step of 512 rows. Cheaper: a narrower wide step, the parts of the
      kept role handed to the load, or the choice remembered per model and cards without
      `--cache` as with it (the owner to decide whether a remembered choice is still "measured
      at load").
- [ ] Flash-Next's K/V group sizes its partial states with the bound of before (twenty tiles of
      the budget): a prompt that reads its prefix with another sequence, or past its first tiles
      with K and V in `q7` to `q4`, can take more (the 27B's failure of 2026-11-23). To size it
      with `PlanConfig::state_slots_bound` as the 27B does, measured for the slots it costs.
- [ ] `qwen4exp_device` with both cards on every layer (`ARES_TEST_SPREAD=tensor`, the share 0.5
      and experts only alike), 2026-11-23 on main e33779a: `every_layer_of_wide_steps_matches_the_reference`
      (attention core 5.16e-2 against a bound of 3e-2 for the shipped types) and
      `experts_of_every_weight_type_on_the_cpu_change_nothing_4` (f32 and q2_K: 4 of 480 misses
      not computed on the CPU) fail; they passed on the evening's main (c5d27e9).
- [ ] Rust 1.100 (stable on the test machine) crashes compiling `ares-model --test
      qwen35_failure` (an internal compiler error); 1.99 builds it, and the test passes there.
- [ ] `qwen4exp_runtime` with the default test threads hung for over 18 minutes once (two device
      threads spinning; the `moe/prompt-chunks` agent); with `--test-threads=1` it passes.
- [ ] `qwen4exp_device` with several tests at once on the same cards (2026-11-23, both local
      cards, `cuda/tested-archs` and its base e6d7676 alike): with four test threads
      `every_layer_matches_the_reference`, `every_layer_of_wide_steps_matches_the_reference` and
      `a_long_step_that_misses_experts_ends` fail and the tests with experts on the CPU or
      uploaded run past 15 minutes; one at a time each passes in seconds, and the whole suite
      with `--test-threads=1` passes, 25 of 25 in 183 s.

## Decided by the owner (2026-11-22)
- Push day (the owner, 2026-11-24): when the cleanup and the docs are done, the first push and
  the switch of the owner's production to Ares come together. Before it: the libcuda deadlock
  of page-locked frees fixed (it reaches the 27B too, through the prompt cache's checkpoints).
  Then: llama-swap's 27B entry runs Ares at the owner's settings with its own prompt-cache
  directory; the owner's llama.cpp trees go once every branch, stash and local commit is on
  GitHub or in a bundle, right away with no fallback period (the owner) (Ares's own uses moved first: the draft vocabulary, the tools' default
  paths, the fork comparisons from `/var/tmp/ares-survey/src/`); llama-swap's configuration is
  cleaned up and the program updated (restarts and installs are the owner's, with sudo).
- Take code, not ideas (the owner, 2026-11-24): where an open engine has what Ares needs, its
  code is taken and adapted instead of rebuilt from its docs. Allowed: MIT, BSD, Apache-2.0
  (Ares is Apache-2.0); not GPL, AGPL or code with no license. Kernels come over nearly as they
  are, host logic (C++, Python) is translated into Rust from the source. Every piece names its
  origin (repository, file, commit) in a header comment and in THIRD-PARTY-NOTICES with the
  license text. A/B and the owner's rules apply as to any change. Taken first, then improved
  (the owner): the port is its own commit, adapted to Ares's structures, and every improvement
  after it is a commit of its own measured against the port.
- A named `--context` overflows like the model's own (the owner, 2026-11-24: an operator always
  names it, e.g. 4 sequences of 128k = `--context 524288` for all together). For Flash-Next the
  indexer's keys and the near pages stay in VRAM and the rest lies in RAM, whether the context
  was named or not; the 27B's (dense attention) stays all in VRAM. In work on `cleanup2/server`
  after the loaders.
- The cleanup comes first (the owner, 2026-11-24): the SSD comparison (item 3, UD-Q4_K_XL and
  UD-Q5_K_XL against the newest Strata) starts after cleanup phases 1 and 2, not beside them.
- `--parallel` defaults to 4 (the owner, 2026-11-24; was 2). At the owner's settings (27B,
  262144 cells, two cards) four cost +80 / +440 MiB on the RTX 4070 / 3060 and gave 133 / 127 /
  125 tokens/s for 3 / 4 / 8 requests at once with every first token within 0.33 s; eight gave
  136 / 150 / 210 tokens/s for +170 / +1340 MiB (the RTX 3060 at 11,425 MiB).
- Greedy output need not be the same from run to run (the owner, 2026-11-23): Strata's is not
  either; a divergence for speed is fine as long as the output is not wrong. So the expert rows
  keep their atomic sums, and the unused ordered sum (`Accumulate::Ordered`, `rows_combine`) goes
  in the cleanup. Tests that compare greedy text with a reference must accept a different token
  at a near tie (the reference's logits rank both close) and fail only on a token the reference
  ranks clearly worse: `flash_next_writes_with_drafts_what_it_writes_without`, the MTP links
  after a pass and the wide-step bound fail now at such ties.
- Inkling 2 Mini on Strata is set aside (the owner, 2026-11-23): Strata serves it with thinking
  forced off (and without prefix reuse for requests with tools), so its quality against the 27B
  there (code 80 against 91 %, harder agent tasks 8/12 against 12/12) says little about the model.
  Inkling in Ares stays after the 27B and Flash-Next are on par.
- CUDA runs only on tested classes (the owner, 2026-11-23): today Ampere (8.6) and Ada (8.9);
  Blackwell (12.0) after the RTX 5060 Ti's tests. Every other GPU goes through the Vulkan backend
  once it runs models; no untested machine code and no PTX fallback (`cuda/tested-archs`).
- Order from 2026-11-23 (the owner): 1 prefill (the 27B's long prompts, Flash-Next's prompts and
  cached agent turns), 2 the fixes the parallel-request comparison finds, 3 large mixtures of
  experts from the SSD (UD-Q4_K_XL and UD-Q5_K_XL against the newest Strata), 4 Jev mode, 5 a pass over
  all kernels (the owner, 2026-11-23: every kernel looked at once, improved where it is needed),
  led by a profile of real work on both models: first the kernels for steps of a few rows (the
  27B's dense matrices, Flash-Next's experts), so that a step of 8 to 16 rows costs little more than
  one, the prompt attention on the RTX 3060 (45 % of its tensor rate against the fork's 67 %; a
  kernel with warps that only decode and warps that only multiply), and right after them draft
  trees, measured with those costs (TensorFold's tree gains rest on 16 rows costing 1.05 to 1.2
  times one row; here 1.46); then Inkling 2 Mini.
- Inkling 2 Mini in Ares comes after the 27B and Flash-Next are at least as fast as the fork and
  Strata in everything (the owner, 2026-11-23); the test of Inkling on Strata goes on only as a
  judgement of the model.
- Ares stays (the owner, 2026-11-23, after the comparison with Strata 0.1.65 on the test machine:
  Flash-Next UD-Q3_K_XL at 40 GiB decodes 34.4 / 25.3 tokens/s with / without drafts against
  Strata's 22.0 / 20.2, but reads prompts at 178 / 260 / 291 against 241 / 796 / 1159 tokens/s and
  takes 6 to 31 s for a cached agent turn against 0.3 to 0.5 s). Prefill is the problem to solve
  first, for both models; Inkling 2 Mini and Flash-Next are tested on the newest Strata meanwhile.
- K/V beyond VRAM for dense attention (the 27B) is not wanted: Qwen4 will not have dense
  attention and Inkling 2 Mini's K/V is small; the 27B keeps the context its devices hold.
- K/V beyond VRAM (`kv/overflow`): the far K/V pages a finished request leaves behind rank as
  prompt cache, the lowest user of `--ram`: under pressure they give way before the tier of
  experts (to the SSD level with `--cache`, else dropped); a running sequence's far K/V stays
  required.
- Large prompt steps stay as built. The interruptible layer-major pass (TokenWing's order: the
  whole prompt through one layer, then the next, so that each expert crosses the link once per
  prompt; a waiting request can cut in between layers) is moved forward (the owner, 2026-11-22):
  the next Flash-Next piece after `flash/prompt`, not the optimising phase.
- Flash-Next on two cards: the plan measures at load whether the second card computes a dense
  share or holds experts only, and keeps the faster; no option. Built (`devices/load-choice`,
  `device/roles.rs`): on the two development cards it keeps experts only (see "Owed on two
  cards" for the runs and what the trial costs at load).
- Checkpoints: both rules as built (a message boundary 1024 cells or more after the last; with
  one place a request's end state does not push out its own anchor).
- A tool call a client gives back with its schema's defaults added is left as the client sent
  it; the checkpoint at every message boundary makes that turn cheap (63 to 82 rows on
  Flash-Next, measured).
- Flash-Next's dense products take int8 on a capped card as the 27B's do. The cause given in the
  question was wrong: the float matrices are not in the vote (`dense_types` leaves them out since
  2026-11-12). What forces the exact f16 path is a quantized type with no int8 tile kernel
  (Q2_0, Q4_0, Q5_0, Q2_K, IQ2_XXS, IQ1_*; the IQ2_XS file has Q2_0 in three shared-expert
  downs): one such matrix makes the forms differ and every dense matrix takes f16, at half rate on
  a capped card. On the full-rate RTX 4060 Ti the measured choices differ by type (Q6_K, Q3_K,
  Q5_K f16; IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, Q8_0 int8), so there too every dense matrix runs f16.
  To build: a form per matrix (both forms prepared where a layer's types differ), as the 27B's
  chooser does.
- Capped cards measure their prefill path per type as full-rate cards do, in place of the rule
  kept from the two development cards (`FLUSH_AS_FAST`). Built (`devices/load-choice`); the
  rule stays only for a type nobody measured (tests and benches).

## Questions to the owner
- None open.
