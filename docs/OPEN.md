# What is still open

Updated 2026-11-16, `main` at the shared layer's step 3, the speculation, the server basics
and the image tower on the device (nothing pushed). What is there is listed in
[`FEATURES.md`](FEATURES.md).
This file is brought up to date at every merge into `main`. It lists work, defects and
measurements that are owed; what a feature must do in detail is in
[`reference/feature-checklist.md`](reference/feature-checklist.md) (223 rows: 70 done,
42 partly, 88 not started, 23 outside the frame; its rows K01, K02, K06 and P14 are out
of date: reasoning is built, and `q7` to `q4` run in the 27B).

Order of work the owner has set: CUDA complete for both models first, Vulkan parity last.

## In work now

| Piece | State |
| --- | --- |
| Shared model layer step 4: slot state and checkpoints as one contract, Flash-Next's prefix cache with RAM and SSD | agent working (`model/prefix-flash`) |
| Shared model layer steps 7 and 8: drafters as kinds of the runtime, DFlash2 on both models (chains; trees are 9 and 10) | agent working (`spec/dflash`) |
| Flash-Next: the next layers' routers predict picks on the device, capped read-ahead from the SSD | agent working (`moe/predict`) |
| Flash-Next: pruning as a fourth level, three rules compared (skip, router's next best, most similar), `--expert-prune RULE:THRESHOLD`, off by default | agent working (`moe/pruning`) |

## Shared model layer (the owner's rule, 2026-11-16)
Generic for Ares, not like llama.cpp: a few models, shared code where it can be, each model
still at its maximum through its own launch plan. A part becomes a shared block when it owns
state the runtime touches (pages, slot state, checkpoints, windows, commits, replays, trees)
or when it drafts; products, norms, kernel choice per row count, fused launches and exchanges
between cards stay with the model. Today the two models share almost nothing inside
`crates/model`, which is why every parity item below is owed twice. Design study:
`/var/tmp/ares-survey/shared-model-layer.md`. Each step lands green on its own.
- [x] 0. A `blocks` module; the 27B's K/V code moved into it as it is.
- [x] 1. The K/V group as a block, the 27B onto it (bit for bit, same launches).
- [x] 2. The recurrent (delta-rule) block, both models onto it (bit for bit).
- [x] 3. Flash-Next onto the K/V group: K and V in `q7` to `q4`, windows as if alone.
- [x] 4. Slot state and checkpoints as one contract: Flash-Next's prefix cache with RAM and SSD
      (2026-11-16, `blocks/slots.rs`; the 27B moved onto it bit for bit).
- [ ] 5. Token rows (embedding on the host, in the mapped file or on the card; the image rows).
- [ ] 6. One driver for the draft block: Flash-Next's draft vocabulary and picks on the card,
      `--drafters mtp`; the 27B's drafts under a layer split. After the speculation piece.
- [x] 7. Drafters as kinds of the runtime (lookup, draft block, DFlash): `Model::drafters`, the
      drafter's id in word 6 of a segment, a price curve, acceptance and confidence per drafter.
- [x] 8. DFlash2 chains on both models: each model gives only its tap (the 27B the residual `x`,
      Flash-Next the mean of its four streams), its embedding rows and its draft head; the loader
      reads the drafter's shape from the file and refuses DSpark, DFlash 1 and the other variants
      by name. Open: Flash-Next under a split by layers (taps would cross devices), the ring in
      the prefix cache's checkpoints, sampled drafts (the walk is greedy).
- What trees (9, 10) find from 8: the lattice of every pass is kept (`Drafter::lattice`: 16
  candidates and a 16 x 16 score table per position, tokens already mapped through the head
  view); the walk is one function of it (`Lattice::chain`, kernel `ares_dflash_select`), which a
  tree fill replaces; the drafts list has a place per window, which becomes a place per node;
  the inject is per row and position, so a tree's kept path needs only its rows' positions
  (a rejected branch's rows are never read, as now).
- [ ] 9. Trees in the blocks, verified once for every drafter and both models.
- [ ] 10. Tree fill from the DFlash lattice, MTP links mixed into a tree.

## Qwen3.8 Flash-Next (`qwen4exp`)

### Speculation that pays (output unchanged)
The draft block is in `main` (`--drafters <file>`): exact. Measured 2026-11-16 on two cards
with `--ram 8G` (the runs are in "What made the draft block of Flash-Next pay" in
`ARCHITECTURE.md`): with the block 21 to 35 tokens per second on new text and 27 to 64 the
second time, against 13 to 22 and 17 to 29 without it at about the same load; one server
asked with and without drafts in turn gave 12 to 35 % more with them on the mean. On one
card (the RTX 4070, a third of the experts in VRAM) the block makes it 7 to 34 % slower.
What is left is the file: at `--ram 8G` a quarter of the experts are only on the SSD, and a
row of new text reads up to sixteen of them a token (Strata's 56 tokens per second run with
the file in the page cache). Measured again after the rebase onto the shared model layer
(machine load 15 to 22 from other builds): one server asked with and without drafts in turn,
twelve requests each, gave 13 % more with them on the mean (German 8, English 5, code 30);
single requests ranged from 46 % slower to 157 % faster, and servers started one after the
other differed more by the load than by the drafts.
- [x] Window prices that follow the tiers: a size's price is the mean of its last eight
      steps without the largest, and a size that has not run for 256 steps is tried again
      at the price of the nearest smaller size that runs (`ares_sched::Curve`).
- [x] All 512 experts of the draft block resident in VRAM (675 MiB, a pool of their own
      outside the tiers, `LayerSpec::resident`): a link costs 1.4 to 1.6 ms where it took
      5 to 9.
- [x] Chain cut per token by the draft's own probability: the links. The block reports the
      probability of each draft, the conductor counts acceptance by probability and sets
      a bar per link from the step's rate and the link's price, and the chain ends at it.
- [x] The rows of the drafts that do not pay leave the step: the chain runs in a step that
      only drafts, and the conductor chooses what the round verifies of it by the
      probability of every draft (both models, `Conductor::chains_first`).
- [x] The price of a window by what it misses: the plain mean for a model whose steps wait
      for what they fetch (`Model::steps_wait_for_fetches`).
- [x] A draft row waits for the file only while it is worth it (the record's row worth; the
      store leaves a row whose reads cost more than it is worth, and the conductor accepts
      nothing from it on). "Never wait" as first built left nearly every draft row at
      `--ram 8G` and took the gain away.
- [x] What a row was left without is read ahead for the next round (`StoreConfig::read_left`):
      without it leaving only moved the read to the next round's first row.
- [ ] Draft vocabulary (German, English, code) for the draft block: the block's head is the
      model's whole output head (248 320 rows) on the device with the head; a subset of its
      rows gathered next to it, and the block's sampler mapping back to token ids, as the
      27B does it (`--draft-vocab`). Not started.
- [ ] A window's price by what its rows read: on one card a round with drafts reads 100 to
      260 experts from the file against 45 to 100 for a row, and what those reads push out of
      the tiers and hold up for the next steps is charged to no step, so the conductor keeps
      choosing windows that cost more than they bring. A price per row by the tier its
      experts come from (the study's `dense(rows) + chain(links) + misses x cost per tier`),
      or the block not resident where VRAM holds a third of the experts, are the leads.
- [ ] `--drafters mtp` for a model file that carries its block; drafts in the measuring tool.

### Experts from the SSD
- [x] Usage profile: the server keeps its expert heat in a file, loads it at start in place
      of the canned warm-up text, keeps it current, and fills empty slots from it when idle
      (branch `moe/expert-profile`). Measured 2026-11-16: the first requests after a start from
      a profile of use miss as few experts as a warm server's (2.7 to 4.5 reads from the file
      per step against 6 to 15.6 after the warm-up); the fill takes 12.6 to 13.1 s on one card
      (warm-up 25 to 32 s) but 20 to 24 s on two (warm-up 21.6 to 27.8 s, no longer 41 to 44):
      both read 20 to 25 GiB and are bound by the drive.
- [ ] A start on two cards that is much faster than the warm-up: reads of several neighbouring
      experts at once into a buffer (the sorted single reads reach 1.25 GiB/s where random ones
      reach 0.7 to 0.8; the gain of larger ones could not be measured on the shared drive), or
      answering once VRAM is filled and filling the RAM tier behind it until the first request
      (which then misses what RAM would have held).
- [ ] The draft block's experts in the usage profile (they are filled in the order of their
      numbers at the start, as before).
- [ ] The slots per layer from the profile when the store is planned (`StorePlan::new` takes
      a heat; the runtime passes none, and the slots move between layers by need as it runs).
- [ ] The heat counts the steps that measure the price of a step at the start (token 0, about a
      thousand rows); the server's start drops it (`Qwen4Exp::serving`), the runtime itself
      still counts it for a model without a profile.
- [x] Prediction on the device (branch `moe/predict`): right after a layer's router, on a
      stream of its own, the routers of the next two layers on the rows it read; the store
      reads ahead what no tier holds where such reads have been picked at least half the
      time (per kind of step, distance and place), into RAM nobody uses, at most what the
      drives take of prefetches at once and per step what the drive could read beside the
      step's demand reads; a drive takes two prefetch chunks of 512 KiB beside a demand chunk
      (four alone). Wide steps predict the next layer where the step before read from the
      file, read ahead through transit's guesses. Fingerprint unchanged. Measured: on the
      IQ2_XS file with `--ram 8G` or `4G`, one card or two, a generated row's predicted
      experts in no tier are picked 3 to 8 % of the time, so nothing is read ahead in
      generation; rows of prompts in steps of eight read some (a quarter used). Through the
      server on one card (50 to 100 reads from the file a step) 60 % of the reads ahead were
      picked and the reads on demand fell from 65 to 54 a step (one pair of runs); on two
      cards no difference beside the spread of runs on the shared machine.
- [ ] Measurements owed (no turn of the shared GPU was left): the Q4_K_XL file (104 GiB), base
      against the predictions; prompts on the real file in wide steps (`--prefill 4000 --rows
      512`); more than one pair per configuration on a quiet machine.
- [ ] The prediction made only where it can pay: in steps of one row on these tiers no place
      qualifies, and the card still applies two routers per layer (not seen in the step
      times). A step could predict only now and then while no place qualifies, to keep the
      counts, through the board as wide steps are asked.
      (A first attempt with the prediction on the host was rejected: it changed the text and
      halved generation on two cards; branch `moe/read-ahead` is kept as a record.)
- [ ] The shared expert's output in the prediction's input (Run My AI: 11 % fewer misses on
      Ling): the part that routes has it only when it computes the shared expert itself
      (not with `--shared worker`), and the routed experts add into the same buffer, so it
      needs a copy on the step's stream. Not built; the prediction reads the router's input.
- [ ] Long prompts layer by layer over the whole prompt, interruptible, with read-ahead of
      the most used experts of the next two layers. (Wide steps read the next layer's
      predicted picks ahead now, within a step.)
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
- [x] Pruning as a fourth level next to VRAM, RAM and SSD (branch `moe/pruning`):
      `--expert-prune skip|router|similar:RANK`, off by default, on the device right after
      the router's top-k; the tool `qwen4exp_prune` measures it. Skip is the rule to use
      (least KL for the reads it saves at every rank on both files); a stand-in by
      similarity does not pay because the experts' outputs are nearly orthogonal (the most
      similar one carries well under 1 % of a missed expert's output). IQ2_XS, 8 GiB:
      `skip:4` 0.7 reads per token instead of 4.5, 45 tokens/s instead of 31, KL 0.011;
      Q4_K_XL from the SSD: `skip:8` 5.0 tokens/s instead of 3.5 at KL 0.026, `skip:1` 17.5
      at KL 0.50. Tables in ARCHITECTURE.md ("Pruning as a fourth level").
- [ ] Pruning, what is left: (1) the table for `similar` is made by the tool
      (`qwen4exp_prune calibrate` and `near`, minutes of GPU and CPU); the server only reads
      it, and refuses `similar` without it; (2) slices of a prompt (wide steps) are not
      pruned; (3) a decision per expert over a window (Strata's) instead of per row was not
      measured; (4) where a third of all picks are touched (Q4, `skip:1`), stand-ins with the
      table's scale (KL 0.42) and the router's renormalised (0.46) beat skip (0.50): skip
      renormalised at that rank was not measured; (5) the speeds were measured with every
      miss uploaded (`--serve stream`, for runs that repeat), not with the CPU serving them.
- [ ] Defect: Flash-Next's float-path tensors (stream mixers `hc_attn`/`hc_ffn` down and up,
      `output_hc` down and up, `ple_key`/`ple_value`) are refused in a quantized type
      (`device/load.rs` `float_matrix`; UD-Q4_K_XL has them in Q8_0, the IQ2_XS file in BF16).
      Owed: take them in every type without a VRAM cost beyond the file's (a quantized kernel
      with float activations, or a measured choice), exact where the type allows. Sizes: the
      194 mixer matrices of 10240 x 320 are 644 MiB as Q8_0 and 2425 MiB as F32 (1212 as BF16),
      `ple_key` (2560 x 10240) 26.6 against 100 MiB, `ple_value` (2560 x 2560) 6.6 against 25
      MiB; all of them dense weights in VRAM (the mixers on the card that holds the streams,
      the n-gram block's on the card of layer 1), so every MiB more is expert slots less. The
      pruning measurement used a copy with them dequantized to F32 (exact for Q8_0), 1.9 GiB of
      slots less than the file would need.
- [ ] Defect: the UD-Q4_K_XL file (as an F32 copy of the float path, see above) does not load
      with both cards on every layer, the default: the card that holds the streams has room
      for 4.94 GiB of slots, and the store asks for 88 slots of about 3.6 MiB per layer of the
      experts it owns (eight rows of ten picks and eight free), 15 GiB; the error then says to
      name a larger `--ram`, which is not the cause. `--split layer` loads (7.61 and 7.22 GiB
      of slots, 21 and 20 % of the experts each card runs). Owed: a floor of slots per layer
      and device that follows what a step can pick there (with the experts split between two
      cards, a device's share of a step's picks), and the right error.
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
      land), and say in ARCHITECTURE.md which paths are reproducible; (1) and (2) are choices
      for speed that a test of quality must know.
- [ ] The first reading of a short prompt after other texts is slower with transit
      (35 against 86 tokens/s in one run); not understood.
- [ ] A request that arrives behind a lone long prompt waits for one piece of 512 rows,
      2 to 2.5 s. Needs a bound in time. **Question to the owner: how long may it wait?**

### Host memory
- [ ] A release that does not wait for the running step, so the headroom stays at a few
      hundred MiB (it sits at its cap of 1.95 GiB after every load).
- [ ] A MoE model and a dense model on one device under memory pressure: the dense
      checkpoints are still freed on the watch's thread, the path that deadlocked.
- [ ] A log line when the tier shrinks.

### Cache, tests, kernels
- [x] K and V in `q7` to `q4` for this model (2026-11-16: the shared K/V group, see
      "Parity with the 27B" below).
- [x] Prefix cache for this model, with the RAM and SSD levels (2026-11-16, on the shared slot
      block; see "Parity with the 27B").
- [ ] Device tests that do not depend on machine load
      (`steps_wider_than_eight_rows_match_the_reference`,
      `every_layer_of_wide_steps_matches_the_reference`,
      `experts_of_every_weight_type_on_the_cpu_change_nothing_*` fail now and then under load).
- [ ] The step scratch starts its buffers on a 128-byte line (as fixed for the 27B).
- [ ] Float experts beyond the f16 range are not caught in the many-row kernels.
- [ ] CPU path for the 24 types without a hand-written kernel is 3 to 11 times slower.
- [ ] NVFP4 many-row gate and up 1.4 times its neighbours on the RTX 4070; IQ2_XXS many-row
      at 8 rows 7 % slower on the RTX 3060 since the kernels for all types.

### Parity with the 27B (the owner's rule: what one model can, the other must)
Everything `FEATURES.md` lists for the 27B and not for Flash-Next is owed:
- [x] K and V in `q7` to `q4`, separately for K and V; the draft block's K and V in `q4`
      (2026-11-16, the shared K/V group). Open: a row of a draft window past the dense limit
      that reaches across the point where a tile leaves the newest cells reads that tile exact,
      not converted as the row alone would (rows within the limit read it as alone); the
      share of drafts accepted with the block's `q4` was not measured on this model.
- [x] The prefix cache with its RAM and SSD levels (2026-11-16). Open: the draft block's last
      cell of a page waits for the next token (the block is one token behind), so a page that is
      cached at a sequence's end has it unwritten and a sequence that resumes there writes it
      into the shared page while its tile is exact; only drafts can differ. With `--ram` small
      enough that the tier of experts takes it all, the cache in RAM has its floor alone: one
      checkpoint place, booked before the tier, also without `--cache` (the owner, 2026-11-16:
      a conversation resumes from its last checkpoint), and with `--cache` 16 pages besides.
      Open: the one place holds the newest checkpoint, the end of a turn; a template that writes
      the reply into the next prompt otherwise than it was generated (the 0.8B's) parts before
      that end, and the next turn replays every cached row. With one place the cache could keep
      the last checkpoint at a page boundary instead of a tail (Flash-Next's template continues
      exactly: 1416 of 1436 tokens cached, nothing replayed, at `--ram 8G` without `--cache`).
      On the made-up Flash-Next under a limit the tier takes whole the place kept its first
      checkpoint over three turns (1 saved, 2 restored, 242 of 498 found rows replayed): a later
      checkpoint did not take the place over there; not looked into.
- [ ] The draft block read from the model file (`--drafters mtp`) and a draft vocabulary.
- Not owed (the owner, 2026-11-16): replies that are bit for bit the same from run to run.
  They differ slightly when the CPU computes missed experts, and that is accepted.
- [ ] Structured output, tool calls and the Messages and Responses routes have never run on
      the real Flash-Next file (the runtime and server code is shared; a smoke script for
      tools exists and was not run).

## Qwen3.8 27B (`qwen35`)
- [ ] Drafts under a layer split (Flash-Next drafts in every placement; the 27B not under
      `--split layer`): parity owed to the 27B, step 6 of the shared layer.
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
      loads and with the draft block; the 0.8B decodes 5 % slower than it could.
- [ ] `qwen35_run` aborts a direct prompt run with `--kv q5,q5` (reserve for partial states).
- [ ] Float matrices under the fast precision use f16 sums (0.04 to 0.06 % rms) where the
      F32 matrices of an IQ4_XS file were exact before: decide with a KL measurement.
- [ ] VRAM and quality against the fork at equal settings: never measured.
- [ ] Prompt on the RTX 3060 alone is 3 % behind the fork.
- [ ] The model's own drafts for the thinking rows of a request that has to call a tool;
      drafts while a rule is bound.
- [ ] Sampler: a row with no finite logit reports `-0.0` where the reference says minus
      infinity; deciding a level got slower (35 to 42 us); a list of the set's tokens after
      the first cut; on two cards most of the sampler's time on the RTX 3060 is the exchange.
- [ ] F32 tile kernel is 2.5 times slower than BF16 (untuned).

## Shared model layer (every model; the study `shared-model-layer.md`, steps 0 to 4 done)
The K/V group, the recurrent block and the slot arena with its checkpoints are shared
(`crates/model/src/blocks`); the rule for what
becomes a block is in `ARCHITECTURE.md` ("The shared model layer"). Left, in the study's order:
- [x] Slot arena and checkpoints as a block (extents from the blocks, save, restore, transfer,
      page copy over all groups); then the prefix cache of Flash-Next on it (2026-11-16).
- [ ] Token rows (table at home on the host, in the mapped file or on the device; kept rows of
      a head view; a second row source for images); Flash-Next's embedding from staged rows.
- [ ] One draft-block driver for both models (chain on the device, picks by the sampler); the
      27B's drafts under a layer split; Flash-Next's draft vocabulary and `--drafters mtp`.
- [x] Drafter kinds in the runtime, taps and DFlash (chains).
- [ ] Trees in the blocks, tree fill.
- [ ] Flash-Next past the dense limit: a draft window's row that reaches across the point where
      a tile leaves the newest cells reads that tile exact, the row alone reads it converted
      (`Group::ahead` is planned for rows within the limit only; the lists are made on the
      device from the table, which still names the exact entry). A shadow row of the table for
      such rows would close it.
- [ ] Flash-Next with several sequences: a generated row is planned alone now (no prefix read
      together with other sequences); its cost with three sequences was not measured.

## Drafters (every model)
- [x] DFlash2 as a kind of drafter the runtime knows, chains, for both models (steps 7, 8):
      `ares_model::dflash`, `--drafters <file>`. The 27B's file and Flash-Next's (trained on the
      mean of its four hyper-connection streams, which llama.cpp cannot drive) both draft.
- [ ] DFlash2 trees (steps 9, 10): tree verification in the blocks, tree fill from the lattice,
      node allocation over sequences. Study: `/var/tmp/ares-dflash-study/REPORT.md`.
- [ ] DFlash2 on Flash-Next split by layers: the first device's taps (or its share of the
      fusion, `rows x n_embd`) must reach the head's device; the hand-over lane could carry it.
- [ ] The DFlash ring in the prefix cache's checkpoints (42 MiB a slot in f16): a sequence that
      resumes from the cache drafts from the resume point's context only until its ring fills.
- [ ] Sampled DFlash drafts: the walk is greedy; keyed noise in the scores (as the draft block's
      picks have) would raise sampled acceptance (study: 2 to 8 %).
- [ ] The 27B with the DFlash drafter at the default split and a 32k context in `qwen35_run`:
      the tool sizes its pool without the drafter and the device's ledger refuses the drafter
      (the server's plan counts it).

## Prompt cache (every model; RAM first, then SSD, as in the owner's fork)
- [x] RAM level for the K/V state that leaves VRAM; write to SSD when the RAM level evicts
      (full or shrinking), at unload and at a clean stop (branch `cache/ram-level`).
- [x] The same for Flash-Next (2026-11-16: its prefix cache on the shared slot block; no
      runtime change was needed, also with both cards on every layer).
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
      `/v1/messages` took 18: the history is rendered differently somewhere.

## Server and API
- [x] API keys (X01); configuration file and environment variables (M08); CORS (X04);
      Host and Origin checks (X05).
- [x] Tokenize and detokenize (A09); model properties (A16); template override (F07);
      continuing the last assistant message (F09).
- [ ] Truncation and context shift (F18).
- [ ] `/v1/completions` reads the text of special tokens (`<|im_start|>`) as text, where
      llama-server parses them; `/tokenize` reads a text the same way.
- [ ] Thinking budget (K03; Anthropic's `budget_tokens` is accepted and not enforced);
      loop guard (K05).
- [x] Reasoning tokens in the usage (K07); `stop_reason: stop_sequence` and the stop string
      that ended a reply.
- [ ] Tool calls: a generic format for other templates; recovery of imperfect calls (T05);
      string constraints `pattern`, `minLength`, `maxLength`, `format`; forced tokens as
      drafts; a guard for `ignore_eos` with a complete grammar; the end-token rule for a
      schema with thinking; llama.cpp's `json.gbnf` costs 6.8 ms per mask.
- [ ] Responses: stored responses (A05), WebSocket (A06); the chat route's sampler fields
      are refused there. Messages: never run against Claude Code.
- [ ] Router: load and unload times in the docs; "no room" for Flash-Next is recognised by
      message text; a plan cannot be asked without loading; two elastic models on real devices.
- [ ] A request with tools or an effort for an unknown model gets a 400 where a 404 belongs.
- [x] Latency histograms (O01), logs (O03, O04), a bounded queue that says when to retry
      (S04), timeouts (S23).
- [ ] Priorities (S06), a watchdog for a silent engine (S23), a VRAM reserve for other
      programs (V04), container image and service unit (X09), key roles and a reload of the
      key file (X02), TLS (X03).
- [ ] `/props` of a model that is not loaded (today: 503 `model_not_loaded`, as `/metrics`).

## Whole features not started
- [ ] Image input (F11; the owner's preset sets `mmproj`). Design: `/var/tmp/ares-survey/image-input-design.md`
      (2026-11-15). Both projector files are the same tower (`qwen3vl_merger`: patches of
      16 x 16, 27 blocks of width 1152, a 2 x 2 merger to the language model's width; 888
      and 866 MiB), with different weights. An image becomes (W/32) x (H/32) rows: 256 for
      512 x 512, 1024 for 1024 x 1024. Pieces 0 to 3 done:
  - [x] 0. Reference numbers from the fork's `llama-mtmd-debug` on the CPU: rows of both
        projectors for eleven square patterns of 4 to 1024 rows, with the sum of every node of
        the graph. The program reads no image file, so no own image, no wide or tall grid and
        no resize went through the fork; the resize is checked against Pillow instead.
  - [x] 1. CPU reference of the tower (`ares-model::qwen3vl`): config, size rule, resize, patch
        rows, the pass. See "The image tower" in `ARCHITECTURE.md`.
  - [x] 2. The tower on the device (`qwen3vl::DeviceTower`, `ares_cuda::TowerKernels`): the
        products on the f16 tile kernels with f32 sums (`ffn_down` padded to 4320 inputs),
        dense attention for heads of 72, norms with bias, bias and GELU, the 2-D rotation and
        the position table's resize as kernels of their own; residual and norm statistics in
        `f32`. Held to the exact reference per image (see "The image tower" in
        `ARCHITECTURE.md`). Open: the row kernels between the products take about a third of
        the time outside the attention (fusing the residual with the norm after it, a bias in
        the products' epilogue); the tower on Vulkan. Not taken: f16 sums flushed every 64
        values in the products other than `ffn_down` (5 to 9 % faster, up to twice as far
        from the exact rows; output must not get worse).
  - [x] 3. Both homes for the tower's weights (`qwen3vl::Home`, a load option): resident in
        VRAM, or in page-locked RAM and uploaded unit by unit into two slots of VRAM while the
        unit before computes (no VRAM held between images beyond the two slots, 58 MiB); the
        same rows bit for bit. Open: the server's key (with piece 7; the owner chose
        `--mmproj-weights ram|vram`, `ram` by default, 2026-11-16) and adding `DeviceTower::pinned_bytes`
        to the `pinned_ram` of the model that carries the tower (with pieces 5 and 6).
  - [ ] 4. The tower as a job the conductor plans (between steps, on the device whose link and
        tensor cores are faster; its scratch reserved in the ledger at load). Cells and positions
        in the runtime: image spans next to tokens, positions (t, h, w) per row, staging and a
        store of image rows.
  - [ ] 5. The splice in `qwen35` (a row source in the embedding gather; positions in the draft chain's links).
  - [ ] 6. The splice in `qwen4exp` (host rows; the n-gram feed's image token).
  - [ ] 7. Routes (`image_url`, Anthropic `image`, Responses `input_image`), limits, errors, `--mmproj`,
        `--mmproj-weights ram|vram` (`ram` by default; the owner, 2026-11-16).
        Embedded images only; fetching remote URLs is backlog. At most 4096 rows per image by
        default (as the fork), with a key to lower it.
  - [ ] 8. Caching: cells named by a digest of the resized pixels, 64-bit names through the index and the SSD level.
- [ ] Audio, video and Gemma 4 are backlog: none of them blocks the first release (the owner,
      2026-11-16).
- [ ] Jev mode: `/v1/systemone` exists and answers 501; `Engine::decide` on the real runtime.
- [ ] Turing.
- [ ] K/V of a running sequence beyond VRAM (P16; an exception, low priority).

## Vulkan (last)
- [ ] A backend-free crate and the `ares-gpu` facade; expert and Flash-Next kernels; both
      models on Vulkan in every placement; attention for wide windows (2 to 6.5 times slower
      than CUDA today); the delta rule's f16 state rounded per row; the sampler's `-0.0`.

## Measurements owed on a quiet machine
- [ ] Flash-Next with the owner's real limit (a RAM tier of 24 to 26 GiB), one card and two.
- [ ] The SSD cache's times to the first token (taken under load so far).
- [ ] Flash-Next with its draft block once speculation pays; target above 56 tokens/s.
- [ ] Before a release: both models against the fork and against Strata, speed, VRAM, quality.

## Questions to the owner
- How long may a request wait behind a long prompt on Flash-Next?
