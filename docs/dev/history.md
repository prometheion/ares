# History

The items of [`OPEN.md`](../OPEN.md) that are done, moved here verbatim on 2026-11-24,
newest first. An item stands under the latest date it names (items that name none are at
the end), below the section of `OPEN.md` it stood in. Branch names, commits and paths are as
they were written then; an open remainder of an item is listed in `OPEN.md` again.

## 2026-11-25

Done on `chore/leftovers` (2026-11-25), each item verbatim and then what was done:

### From "Decided by the owner (2026-11-22)"

From the item "Push day done 2026-11-25 ... Left from it" (its other lines stay in `OPEN.md`):

  - [x] `ares --version` does not exist; a deployed binary is named by its commit only.
        Done: `ares --version` and `-V` print `ares <version> (<commit>, <date>)`, which is
        also the log's first line and `build_info` of `/props`; `crates/server/build.rs` takes
        the commit from git in a checkout, else from `ARES_BUILD_COMMIT`, else `unknown`.
  - [x] Dev tools whose default paths name the removed `~/Services/llama.cpp-native`
        (`crates/quant/tests/golden/generate.py`, `crates/quant/tools/gen_tables.py`, the
        capture tools' READMEs): take the llama.cpp tree as an argument; comparisons against
        the fork need a build from a fresh clone.
        Done: the three generators (with `crates/tokenizer/tools/gen_unicode.py`, which also
        wrote a fixed revision) take the tree as an argument or `LLAMA_CPP_DIR`, with no
        default; the READMEs of `capture`, `q4capture` and `kv_real` build it from a fresh
        clone on the CPU. Checked against a fresh clone of the fork (3f38177): the golden
        fixtures the same bytes, the tables the same but for the revision they name, the
        capture programs built and one ran.
  - [x] The owner's small calls of the cleanup (2026-11-25, as proposed): `examples/launch_cost.rs`
        and `q4_bench.rs` go, `matvec_bench` folds into `few_rows_bench` and `weight_pass`; the
        tools `qwen4exp_run` and `qwen4exp_prune` stay on `Session`.
        Done: the three examples are gone; `few_rows_bench` times every type with a fast kernel
        (`all`, and `all:17408x5120` by default) with GB/s, which is what `matvec_bench` had and
        the other two did not (`weight_pass` needed nothing).

### From "Small fixes / Model, runtime, kernels"

- [x] Dev-only strings with runs of spaces (found 2026-11-24 by `fix/help-docs`; 4 in 3 files:
      `crates/model/tests/qwen4exp_device.rs`, `crates/runtime/examples/expert_pipe_bench.rs`,
      `crates/weights/tests/premises/cache.rs`).
      Done: the four were format strings continued over a line without `\` and continue with
      `\` now. A scan of every string literal found no others (`expert_bench`'s
      `file {i}     {}`, named with them before, lines up with the labels around it).

Fixed on `fix/model-sweep` (2026-11-25), one commit per item; each item verbatim, then what fixed it.

### From "Defects", "Model, runtime, kernels"

- [x] Device memory is not behind the gate of page-locked memory: `cuMemFree` "may perform
      implicit synchronization" too (the libcuda deadlock's class). An engine allocates it when its
      model opens and frees it when the model goes, but for the 27B's attention plan larger than
      its buffer (`Plan::upload` inside a step) and an engine that unloads beside another that
      serves. Files: `crates/cuda/src/memory.rs` (`DeviceBuffer::drop`), `crates/cuda/src/gate.rs`,
      `crates/model/src/qwen35/device/part.rs`.

  *Fixed on `fix/model-sweep`, 2026-11-25:* `DeviceBuffer` is freed through `gate::retire` (`DeviceFree`): at once when no job runs, otherwise when the last job ends; `Device::mem_info` and an allocation that finds no room first free what is due (`gate::settle`, which waits for the running jobs like `gate::change`), so a placement after an unload sees the freed memory. Unit test `settling_frees_what_a_step_gave_back_once_the_step_ends` (`crates/cuda/src/gate.rs`). Commit "ares-cuda: device memory is freed where no job runs".

- [x] The 30-s checks of a step that waits count from the start of the wait, not from the last
      progress, and then fail the step (`Flow::check`): seen 2026-11-24 as "a step stood still for
      30s while rows of the n-gram table were read" at load 27 to 34 (main as well), beside the
      hang after a long far request (the libcuda deadlock, fixed 2026-11-24). Timed from the start:
      the n-gram wait, the throttle, `follow`, the lend waits and the pass's waits; only `wait_for`
      and `complete` restart on progress. Files: `crates/model/src/qwen4exp/device/flow.rs`,
      `qwen4exp/device/runtime/pass.rs`.

  *Fixed on `fix/model-sweep`, 2026-11-25:* `Flow::check` counts 30 s from the later of the wait's start and the last time anything of the steps moved (`Flow::moved`: what `Flow::progress` counts, a pass's rows handed on, taken or let run, rows of the n-gram table read; `NgramFeed::completed`), looked at once a wait passes a second, so every wait (the n-gram wait, the throttle, `follow`, the pass's waits) restarts on progress; the lend waits hold the store and still count from their start. Unit test `a_wait_stands_still_from_its_start_or_the_last_move_whichever_is_later`. Commit "ares-model: a step that waits stands still only when nothing has moved for 30 s".

### From "Small fixes", "Model, runtime, kernels"

- [x] A log line when the tier shrinks (a shrink is counted only; the runtime has no log, so the
      server needs a callback). Files: `crates/runtime/src/expert_store/elastic.rs`,
      `crates/runtime/src/ram.rs`, the server's loader.

  *Fixed on `fix/model-sweep`, 2026-11-25:* the store tells a callback of every shrink of the RAM tier (`ExpertStore::on_shrink`, `RamShrink`: pieces, bytes, what it holds and may hold, at its floor), Flash-Next turns it into a line (`Qwen4Exp::with_log`), and the server logs it at info. Test in `crates/runtime/tests/expert_store.rs` (the shrink to half). Commit "ares-runtime: a log line when the tier of experts in RAM shrinks".

- [x] A loan of VRAM stays while the engine is idle after a request that ended with its prompt
      (it goes back at the next step only). Files: `crates/model/src/qwen4exp/device/flow.rs`,
      `qwen4exp/device/runtime/pass.rs`.

  *Fixed on `fix/model-sweep`, 2026-11-25:* the conductor tells the model when nothing runs and nobody waits (`Model::idle`), and Flash-Next gives back the VRAM lent for wide steps and the loan of a pass that has read its rows there (`Flow::idle`); a pass that has not read all its rows keeps its loan. Not covered by a host test (the GPU suites run it). Commit "ares-runtime: Flash-Next gives lent VRAM back when the engine idles".

- [x] The heat counts the steps that measure the price of a step at the start (token 0, about a
      thousand rows); the server's start drops it (`Qwen4Exp::serving`), the runtime itself
      still counts it for a model without a profile. Files:
      `crates/runtime/src/conductor/measure.rs`, `crates/model/src/qwen4exp/device/usage.rs`.

  *Fixed on `fix/model-sweep`, 2026-11-25:* the conductor tells the model when the steps that measure are done (`Model::measured`, at the end of the warm-up), and Flash-Next forgets the heat they left (`ExpertStore::forget_heat`), with a usage profile or without. Test `the_model_hears_once_that_the_steps_that_measure_are_done` (`crates/runtime/tests/conductor.rs`). Commit "ares-runtime: Flash-Next forgets the heat of the steps that measure".

- [x] Sampler: a row with no finite logit reports `-0.0` where the reference says minus infinity
      (the test leaves such rows out). Files: `crates/cuda/kernels/sampler*.cuh`,
      `crates/cuda/src/sampler.rs`, `crates/cuda/tests/sampler.rs`.

  *Fixed on `fix/model-sweep`, 2026-11-25:* the finishing kernel leaves minus infinity for a pick without a finite logit (`ares_lp_finish`, the model's own probabilities: the distance from the row's best was NaN and became 0); `crates/cuda/tests/sampler.rs` checks such rows now. Commit "ares-cuda: a pick without a finite logit has the probability minus infinity".

- [x] Flash-Next's step scratch starts its buffers on a 64-byte line, not the 128-byte line fixed
      for the 27B (`Kernels::ACTIVATION_LINE`). File: `crates/model/src/qwen4exp/device/part/frame.rs`
      (`PieceCarve`).

  *Fixed on `fix/model-sweep`, 2026-11-25:* `PieceCarve` starts every buffer on `Kernels::ACTIVATION_LINE` (128 bytes), as the 27B's carve; a piece of a loan starts at an expert slot, and a real file's slots are multiples of 256 bytes. Unit test `the_carve_starts_every_buffer_on_a_line_and_in_one_piece`. Commit "ares-model: Flash-Next's step scratch starts its buffers on 128-byte lines".

- [x] The randomized test of admission plays one priority. File: `crates/sched/tests/admission.rs`.

  *Fixed on `fix/model-sweep`, 2026-11-25:* the test plays one priority in nine runs and two or three in four (shortage victims by priority, a running request outranked only by a better newcomer, a spilled one in line holds back its own and every worse priority, overtaking counted by the same or a worse priority). It found a defect of admission: a newcomer that made worse ones give way counted what they let go of as room, the pages of its own cached prefix among them, which its charge had left out already (`Admission::room` takes those of the prefixes that running sequences hold off now). Commit "ares-sched: the admission test plays several priorities, and a newcomer's prefix is not counted twice".

Found done or stale by the triage of 2026-11-25 (branch `docs/triage`, every open item checked
against `main` 4c834979): each item verbatim, then what did it or what changed. Items that only
moved or were merged with a twin are not here; what is left of a partly done item is in
[`OPEN.md`](../OPEN.md) again, in its new section.

### From the head of `OPEN.md`: "The order of 2026-11-17"

**The order of 2026-11-17 (after the gap study; items 1 to 7 done, the rest is a store)**
For "CUDA complete", the order in which the items below are taken up; each item stays in its
section further down.
Items 1 to 7 are done (2026-11-22; their texts are in [`dev/history.md`](history.md), what is left of them
is in the sections below).
8. Then the rest of CUDA complete: image input pieces 4 to 8, gate and up in two types, the
   roles of the cards, the open defects, the server items.
Later than their place suggests: the classic draft model (VRAM; better with an aux card),
Jev mode, stored responses, TLS. DFlash2 trees (steps 9 and 10) with the experts of a tree's
tokens read ahead ("Speculation that pays"): the owner, 2026-11-22, expects much of them for
decode; measured first (how many picks a token table and route memory predict, how much longer a
tree is accepted than a chain), then built; started when the pieces in work leave room, one
piece at a time.

*Triage, 2026-11-25:* superseded by the order of 2026-11-23 (no phases; `OPEN.md`, "Why Ares" and "Decided by the owner"); image input pieces 4 to 8 merged 2026-11-23; what is left of item 8 is in `OPEN.md`'s sections. The owner's expectation of the trees is kept with them under "Features".

### From "Shared model layer"

- [x] Left of step 5, the token rows (steps 0 to 8 are done, see [`dev/history.md`](history.md)): no image
      tower uses the second row source yet; DFlash's newest token and mask rows are still
      decoded on the host for every pass (both models).

  *Triage, 2026-11-25:* Half done: the image tower uses the second row source (`rows::splice`, ae1280ce, 2026-11-22). DFlash's rows are in `OPEN.md`, "Small fixes".

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Found against Strata on the test machine"

- [x] Ares is behind: decode with drafts 40-55 % of Strata's (15.9-31.0 against 40.0-43.0
      tokens/s), without 43-51 %; prompts 4 to 10 times slower (Strata 398 / 829 / 1257 tokens/s
      at 2k / 8k / 32k, Ares 94-121); the first token of a short repeated prompt 3 to 15 times
      slower (no checkpoint at the prompt's end: item 2 of the order). Ahead: load 3 times faster
      (35-50 s against 96-149), 12 GiB of RAM left free against Strata's 2-3, a long conversation
      turn asked again 2.7-2.9 s against 9.3 s (Strata keeps only the newest path).

  *Triage, 2026-11-25:* Stale: Strata 0.1.65 was measured on 2026-11-23 (the owner's entry "Ares stays", 2026-11-23, in `OPEN.md` under "Decided by the owner": decode 34.4 / 25.3 tokens/s with / without drafts against Strata's 22.0 / 20.2, prompts 178 / 260 / 291 against 241 / 796 / 1159 tokens/s, a cached agent turn 6 to 31 s against 0.3 to 0.5 s); the anchor fix and the passes (2026-11-23) came after it. The comparison with the newest Strata is in `OPEN.md`, "Measurements owed".

- [x] Against Strata 0.2.19 (engine 0.1.65, installed 2026-11-22 evening) at 40 GiB: not measured
      yet (the test machine switched off at 21:00); the runs are queued to start when it is back
      (`~/work/flash-cpu/requeue.log`, tables in `results/summary-final.txt`). Strata 0.1.60 at 40
      GiB decoded at 29.9 tokens/s with drafts (mean); Ares now 33.6 to 36.4. Without drafts the
      device's 27 ms a token outside the experts (below) bounds Ares near 30 tokens/s here.

  *Triage, 2026-11-25:* Done 2026-11-23: measured against engine 0.1.65 at 40 GiB (the numbers in the item above, from `OPEN.md`, "Decided by the owner").

- [x] Prompts on the test machine at 40 GiB: 3 to 4 times behind Strata 0.1.60 at 8k and 32k
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

  *Triage, 2026-11-25:* Stale: the passes (`flash/layer-major`, 2026-11-23) read a long prompt layer by layer, each expert once per pass, and the RAM-held guesses go up into free transit slots (24fc9cd4). Left in `OPEN.md`: the comparison ("Measurements owed"), the CPU's share above 2048 rows and the buffers sized for the widest step ("Store of levers").

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Defect seen on the local cards"

- [x] `qwen4exp_runtime` `the_mtp_block_follows_wide_steps_that_start_at_any_cell` fails on the
      RTX 4070 + RTX 3060 in its second placement (`Layers`): `warm()` asserts "every expert has
      a slot" (the made-up model's slots with steps of 196 rows do not hold every expert there).
      The same with `verify/stack` 7816385 itself, so it is the test's or the stack's, not this
      branch's; the placement `One` passes.

  *Triage, 2026-11-25:* Done: aa12d2b9 (2026-11-22, transit slots on top of the budget cut to the device's experts); also recorded under 2026-11-23, "Owed on two cards".

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Speculation that pays"

- [x] Experts read ahead from the drafts (the owner's idea, 2026-11-18; no engine has it:
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

  *Triage, 2026-11-25:* Measured by the DFlash2 trees study, phase 1 (`flash/trees-study`): reading ahead from token tables or route memory gave +1 % (they name 1.4 to 6 % of a round's missing experts); what to build is the owner's (`OPEN.md`, "Questions to the owner"). The idea stays with the trees in `OPEN.md`, "Features".

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Experts from the SSD"

- [x] **Prompts are bound by the link.** A wide step uploads every expert it picks that VRAM
      does not hold (the loan leaves 20-40 a layer), from RAM or staging, once per step: the
      32k prompt uploaded 428 GB in 96 s (4.46 GB/s, ~0.7 ms an expert; ~363 a layer = the
      ~250 ms a layer took), and read 170 GB from the drive beside it (1.74 GB/s of 2.4). Steps
      of 4096 rows make 8 passes; Strata reads it in 2 chunks of 21 248. With the MTP block's
      675 MiB of resident experts the 16 GB card cannot lend 8192 rows even at the loan's floor
      of a row's picks (without the block it can). The levers: a CPU share of the experts a
      wide step picks for few rows (Strata's prefill does this up to 3072-token chunks), less
      scratch per row (0.67 MiB), or the layer-major pass below; all model work, the owner's.

  *Triage, 2026-11-25:* Stale: the passes (2026-11-23) are the layer-major form named here, and the RAM-held guesses go up into free transit slots (24fc9cd4). Left in `OPEN.md`: a CPU share of the experts a wide step picks for few rows and less scratch per row ("Store of levers"), the comparison ("Measurements owed").

- [x] Passes, what is left: with both devices on every layer the MTP block's chains after a
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

  *Triage, 2026-11-25:* One point done: the next layer's RAM-held guesses go up into free transit slots while a layer's micro-batches run (24fc9cd4, `Flow::guess_layer`); TokenWing's form, read here, is what the passes built. The rest is in `OPEN.md`: the MTP chains after a pass ("Defects"), a pause inside a layer's experts and the lent steps that remain ("Store of levers").

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Parity with the 27B"

- [x] Left of the prefix cache with its RAM and SSD levels (done 2026-11-16): the MTP block's
      last cell of a page waits for the next token (the block is one token behind), so a page
      that is cached at a sequence's end has it unwritten and a sequence that resumes there
      writes it into the shared page while its tile is exact; only drafts can differ. The one
      place holds the newest checkpoint, the end of a turn; a template that writes the reply
      into the next prompt otherwise than it was generated (the 0.8B's) parts before that end,
      and the next turn replays every cached row. With one place the cache could keep the last
      checkpoint at a page boundary instead of a tail (Flash-Next's template continues exactly:
      1416 of 1436 tokens cached, nothing replayed, at `--ram 8G` without `--cache`). On the
      made-up Flash-Next under a limit the tier takes whole the place kept its first checkpoint
      over three turns (1 saved, 2 restored, 242 of 498 found rows replayed): a later
      checkpoint did not take the place over there; not looked into.

  *Triage, 2026-11-25:* Stale: since bc9744fb (2026-11-22) a request's end state no longer pushes out its anchor in the one place, which the owner kept ("Checkpoints: both rules as built"); the MTP block's cell one token behind is by design (`architecture/flash-next.md`). The made-up model's three turns were not run again.

### From "Qwen3.8 27B (`qwen35`)"

- [x] `qwen35_run` aborts a direct prompt run with `--kv q5,q5` (reserve for partial states).

  *Triage, 2026-11-25:* Fixed in the code by 5d5aee5f (2026-11-23): the 27B's part reserves its partial states by `PlanConfig::state_slots_bound`, which covers K and V in `q5` and every opening, `qwen35_run`'s included; `qwen35_run` was not run again.

### From "Drafters (every model)"

- [x] Flash-Next's second card (measured 2026-11-22, `dev/architecture/devices.md`, "What the second card is
      for"): on every layer it is worth most (twice the VRAM tier); as an expert tier only (no
      dense share, `core_share` 1) it decodes as fast and reads warm prompts 27 % faster than the
      share chosen today from read times. Proposed: the plan measures at load a wide and a decode
      step with and without the dense share and keeps the faster; goes to the owner first (it is
      the card-roles design), not built.

  *Triage, 2026-11-25:* Done: 7eff7fcd (2026-11-22, `devices/load-choice`): the plan times both roles at load and keeps the faster, no option (`crates/model/src/qwen4exp/device/roles.rs`); also recorded under 2026-11-23. What the trial costs at load is in `OPEN.md`, "Questions to the owner".

### From "Server and API / Claude Code against Ares"

- [x] Flash-Next with both devices on every layer (the driver with peer access, 2026-11-22
      14:27, load 2 to 6) read the same first prompt at 40.9 tokens/s (the split by layers 61
      at load 13) and answered at 3 to 6 tokens/s, with 5 of 55 and 7 of 127 drafts accepted;
      the MTP block's first link took 1 of 6 and 3 of 12 where the split by layers took 25 of
      41, 29 of 45 and 56 of 78 on the same prompts. Few samples, but the MTP block looks badly
      fed in that placement; not looked into. The made-up model's MTP test that failed once in
      the merge check of 2026-11-22 drafted badly in the same placement (`Shared`).

  *Triage, 2026-11-25:* Not seen again: on 2026-11-22 evening a server session in that placement accepted 626 of 1078 drafts (2026-11-23 above, "Owed on two cards"). The made-up model's test is in `OPEN.md`, "Defects".

- [x] A new session on Flash-Next reads the whole 16 thousand tokens again: the root checkpoint
      ("Next", item 2). The template writes the tools first (12.7 thousand tokens) and the
      system prompt after them, whose first line is Claude Code's billing line: 2.1.301 writes no
      `cch=` there but `cc_version=2.1.301.<3 hex>`, another suffix for another first message
      (`66d`, `e64`), so on the 27B a new session took 12672 of 16287 tokens from the cache and
      computed the rest. A stamp that is pinned must include that suffix (item 1).

  *Triage, 2026-11-25:* Stale: the roots at the end of the tool list and of the system prompt are built (19d5dce7, 2026-11-22); the `cc_version` suffix is left as upstream leaves it (2026-11-22 above), only `cch=` is pinned. What is left (with one place the root goes to the drive and is not brought back) is in `OPEN.md`, "Small fixes".

- [x] Claude Code gives a tool's input back with the defaults of its schema added
      (`{"replace_all": false, ...}` first, where the model wrote `file_path`, `old_string`,
      `new_string`), so the history parts from the cached sequence inside the last reply. On
      Flash-Next with one checkpoint place (`--ram 8G` without `--cache`) that turn replayed all
      16512 cached rows: 373 s to the first token for 242 new tokens. A checkpoint at every
      message boundary ("Next", item 2) brings it to the rows of the reply. Writing a call
      that comes back without the arguments that equal their schema's `default`, in the
      schema's order, would match what a model usually writes; it changes the history the
      client sent, so it is the owner's decision (not built).

  *Triage, 2026-11-25:* Decided by the owner (`OPEN.md`): the call stays as the client sent it; the checkpoint at every message boundary (19d5dce7) makes that turn 63 to 82 rows on Flash-Next.

### From "Whole features not started": K/V of a running sequence beyond VRAM

- K/V of a running sequence beyond VRAM (P16). The owner, 2026-11-22: without `--context` a
    model gets its own context always, and what of it does not fit in VRAM overflows into RAM.
    Built for Flash-Next (2026-11-22, `kv/overflow`; "K and V beyond VRAM" in `dev/architecture/flash-next.md`;
    what was done and fixed since is in [`dev/history.md`](history.md)).
    A context named with `--context` overflows the same way (the owner, 2026-11-24; on
    `fix/far-free`, the commit of the former `cleanup2/server`; the owner names it for all
    sequences, four of 128k are `--context 524288`, a request holds at most the model's own):
    IQ2_XS on both local cards with `--context 524288 --parallel 4 --ram 16G` loads with
    18 176 tokens of K and V in VRAM and answers a 96k prompt and a short one, where main
    refuses it (no VRAM left for expert slots). Open:

*Triage, 2026-11-25:* built and merged 2026-11-24 (`kv/overflow`; the named context with 28e64535). What is left of it is in `OPEN.md`, "Store of levers", "K and V beyond VRAM".

- [x] The 27B: not built. Its 16 dense layers read every cell, so K and V in RAM cross the link
      once per token: 5.4 ms per 1000 cells on the test machine's x4 link, about 1.4 ms on a x16
      link (`far_attention_bench`); its context beyond VRAM would run at a few tokens per second.
      On the development machine 104.5 ms a step at 32 768 cells from RAM on the 4070, 435 ms on
      the 3060. Its heads move to the second card first, and with `q4` two cards hold 482 707
      tokens.

  *Triage, 2026-11-25:* Decided by the owner (2026-11-23, `OPEN.md`): K/V beyond VRAM for dense attention (the 27B) is not wanted; the 27B keeps the context its devices hold.

### From "Optimising"

- [x] Flash-Next's wide steps take one form of activations for every dense matrix; where the
      types' forms differ (a type with no int8 tile kernel, or measured paths that differ) every
      dense matrix runs f16. The owner (2026-11-22): a form per matrix, int8 on a capped card as
      the 27B's ("Decided by the owner").

  *Triage, 2026-11-25:* Done: 6227d864 (2026-11-22): each type takes the form its measured path gives; where the forms differ, the kernels that write a product's input write both.

### From "Measurements owed on a quiet machine"

- [x] On the test machine (RTX 4060 Ti 16 GB with gpu-unlock, 62 GB RAM, a
      Samsung 990, Strata 0.2.15 installed, good internet for downloads): Flash-Next against Strata
      with the same file (UD-Q3_K_XL there), a large RAM tier, one card of another class.

  *Triage, 2026-11-25:* Done 2026-11-22 and 2026-11-23: UD-Q3_K_XL and UD-Q4_K_XL at 40 GiB against Strata 0.1.60 and 0.1.65 (`experiments/against-strata.md`).

### From "Parallel requests against the owner's production"

- [x] Flash-Next (IQ2_XS, both cards, `--ram 8G`, 40960 cells; the fork has none): two at once
      give 44 to 54 tokens/s in sum (alone 19 to 40), but 3 or 4 at once stay at 32 to 35 with
      `--parallel 2` and with `--parallel 4` (more sequences pick more experts that RAM and the SSD
      bring). An 8k prompt beside a generating request takes 295 s against 25.6 s alone (11.5
      times), the generation keeps 20.6 of 28.1 tokens/s, its longest gap 3.0 s: alone the prompt
      runs in 9 wide steps on lent VRAM, beside it in pieces in the model's own memory that each
      read the experts again (5.3 million expert picks against 51 thousand, 240 s of expert waits
      against 32). The layer-major passes now on main (3b8d526) read that prompt alone in 17.7 s
      against 26.6 (same machine state) but not beside a generating request (523 s against 496 s,
      machine load 20 to 55 from builds; Flash-Next's decode fell to 9 to 15 tokens/s under it):
      the pieces next to generation are still narrow and read the experts again.
      Done for the prompt (2026-11-23, `sched/parallel`): a long prompt beside generation is read as
      a pass that pauses for the rounds (`Model::pauses_wide_steps`), at most half as wide as the
      slots can lend for. On the test machine (one RTX 4060 Ti, IQ2_XS, `--ram 24G`): 8k beside a
      generating request 61.8 / 62.1 s before, 16.6 / 16.8 s now (alone 10.2); 16k 113 / 117 s
      before, 29 / 31 s now; the generation keeps 14 to 19.5 tokens/s (26 before), its longest gap
      0.7 to 1.2 s. On the two cards at `--ram 8G`: not measured (the local cards were busy).

  *Triage, 2026-11-25:* The prompt's part is built and merged (930d0be7, 2026-11-23; merged 2026-11-24). Left in `OPEN.md`: 3 or 4 sequences at once ("Store of levers") and the run on two cards at `--ram 8G` ("Measurements owed").

- [x] Flash-Next's greedy text differs from run to run of the same binary (test machine, IQ2_XS: 4
      requests of 4 differed between two runs of main, alone and beside), so a change's text cannot
      be compared by hash on this model; presumably the CPU and GPU paths of the experts, which
      timing chooses.

  *Triage, 2026-11-25:* Decided by the owner (2026-11-23, `OPEN.md`): greedy output need not repeat; the ordered sum is gone (a241dec7, 2026-11-24). The tests that must judge near ties are in `OPEN.md`, "Defects".

### From "Cleanup"

The first push to GitHub comes after the whole cleanup (phase 1 and phase 2) and the restructured
docs (the owner, 2026-11-23); then the unpushed history is scrubbed of the test machine's address
and home paths (2 and 6 commits) and pushed without force.
Audit of main ae3a334 by area, reports and the plan in `/var/tmp/ares-audit/` (`PLAN.md`). The core
designs are sound (the shared blocks, one reader and core for all routes, one record writer and
reader, an independent host oracle); the debt is where the code grew: copied bookkeeping beside
each new path, replaced designs left behind for a bench or a test, hand-kept lists, test helpers
copied into every file. About 4,000 to 5,000 lines can go without changing behaviour.

*Triage, 2026-11-25:* done: phases 1 and 2 merged 2026-11-24, the history scrubbed and pushed on 2026-11-25 (push day). What is left of each phase is in `OPEN.md`, "Features", the cleanup.

- [x] Runtime: the running means of the store (their weights differ: one would change what
      the store decides early on); the transit sizing that lives in the model
      (`plan_transit`) and the store config built twice (the model's files); `write_record`
      and `step` of the conductor in smaller parts; the host executor's places on
      `PlaceChunks` (its oracle plays the strictest device, which the chunks are not); the
      record's windows and the text half of `impl Model` in both models (the model's phase
      2); the cache's transfer counter kept twice (a metric's name).

  *Triage, 2026-11-25:* Done parts: the store config built in one place (`open::store_config`, c0a74eaf), the record's windows (3172c036) and the text half (`ModelText`, e68729b5). The rest is in `OPEN.md`: the cleanup under "Features", the host executor's places and the transfer counter under "Small fixes".

- [x] Model: the tools (`qwen4exp_run`, `qwen4exp_prune`) measure the session, not the
      runtime (moving them drops `--owners routing` and the host time of a step: the owner's
      call); `roles::Runner` stays the trial's own timing loop (no store, every expert in
      zeros); Flash-Next's embedding staging written twice (`part.rs`, `part/pass.rs`), the
      profile marks of both parts, `Balance` beside `Meeting`, the record's token offset read
      by hand (an `ares-cuda` accessor), the server's draft vocabulary reader (server phase),
      the large functions left (`enqueue_pass`, Flash-Next's `Part::assemble`,
      `Config::from_gguf`, `Drafter::pass_layers`; `Frame::widest` and the 27B's
      `open_balanced` went down to 277 and 222 lines in stages) and the names of the audit's
      finding 13.

  *Triage, 2026-11-25:* Done parts: the server's draft vocabulary reader uses the model's (a951a913); the tools stay on `Session` (the owner, 2026-11-25). The rest is in `OPEN.md`, "Features", the cleanup.

## 2026-11-24

### From "In work now"

Merged 2026-11-24 (every GPU suite passed on the two local cards): item 6, K/V beyond VRAM into RAM for Flash-Next (row below).
Merged 2026-11-24 (verified on the two local cards, every GPU suite they touch): the parallel-request fixes (generation keeps `--generation-floor`, 20 tokens/s; at the owner's settings a 32k prompt beside generation 57 to 60 s against 84 to 87 before and the fork's 60, 98k 233 s against 754 and the fork's 209 to 219; Flash-Next's long prompt as a paused pass, 8k beside generation 62 -> 16.7 s on one card; `run_on_devices` closes the watchdog's blind waits), the SSD streaming work without the K/V overflow (measurement tools, a lower loan floor, RAM-held guesses into free transit slots), cleanup phase 1 of the server and the model crates (about 2,570 lines fewer, the router test's cache and port, `--help`, the test wrapper's price estimate). The tests that fail at near ties fail as often on main's build as on the merged one (alternating suite runs).

| Piece | State |
| --- | --- |
| Flash-Next UD-Q4_K_XL (104 GiB, larger than RAM) streamed from the SSD against Strata, then Ares's reading path | merged 2026-11-24 without the K/V overflow; the comparison with the newest Strata at Q4/Q5 is item 3 of the order. What it brought: decode ahead of Strata 0.1.60 on prose, prompts behind (the x4 link); measurement tools, a lower loan floor, RAM-held guesses into free transit slots |
| Cleanup phase 1, CUDA | merged 2026-11-24 (every GPU suite passed on the two local cards; about 1,800 lines fewer): one table of weight types drives every type list (the hand-kept decode list could leave rows unwritten), the dead kernels (draft carry, nine delta-rule shapes, the Q4_K reference, the ordered sum per the owner, the break-even probe, the attention's experiment switches), `tests/common`, `ares_quant::act`; the 1,562 kept kernels' machine code is unchanged, 27B decode the same |
| Cleanup phase 2, runtime | merged 2026-11-24 (every GPU suite passed on the two local cards; about 1,700 lines fewer): one policy for page-locked memory (the unused broker of `ares-core` gone), budgets and floors in one place with the server's RAM default, the cache's RAM and SSD imports one way, route picks classified once, the conductor split into seven files with the draft policy in `ares_sched::Drafting`, the old expert fill replaced by `plan_listed_fill`; 27B decode at the owner's settings 168.6 against main's 168.9 tokens/s over three alternating rounds |
| Cleanup phase 2, server (items 1 to 7) | merged 2026-11-24 (every server GPU suite passed on the two local cards; the 27B's chat, Messages and Responses bodies identical to main's): one harness for the binary, one option table that prints `--help` and one INI reader, files named by what they hold (`openai`, `generate`, `metrics`, `sorter`), one way to serve (`Models::of_engine`), one segmenter, `real.rs` split, one test variable per model file (README "Tests") |
| Cleanup phase 2, model | merged 2026-11-24 (every model and server GPU suite passed on the two local cards and the test machine): one opening path and one flow for Flash-Next's parts (`device/open.rs`, `device/flow.rs`; `Session` a thin face for the tests and tools), the step record decoded once (`blocks/step.rs`), the arena, the reference and the text half shared by both models, `runtime.rs` split; the arena uploads from the mapping unless memory is short (27B warm upload 5.9 -> 3.2 s, Flash-Next unchanged) |
| Cleanup phase 2, the server's two loaders and the leftovers across crates | merged 2026-11-24 without the named-context commit (every server GPU suite passed; the start-up logs and VRAM after load identical to main's for the 27B at the owner's settings, with no options and on one card, Flash-Next on both cards and on one, and on the test machine): the loaders in `load/` in phases with one tail and one RAM budget, the 27B's estimate in the model (the server reads no tensors), system messages as the template declares, the formats read once, the samplers' order rule once, `Qwen35::new` given its tokenizer. The named `--context` for Flash-Next waits on `fix/far-free` with the libcuda deadlock fix |
| Cleanup phase 1, runtime | merged 2026-11-24 (every GPU suite passed on the two local cards; about 1,300 lines fewer): the old pass in waves, unreached API, one bookkeeping for loans and the RAM tier's give-back, one tracker of the thinking; the old fill moved to phase 2; the cache's RAM files' items now free since kv merged (see "Cleanup") |
| 6: K/V beyond VRAM into RAM (Flash-Next) | merged 2026-11-24 after every GPU suite passed on the two local cards: 13,056 expert slots (53 %) against main's 10,752 (44 %) at the model's own context, an 80k prompt in 103 s; finished requests' far pages rank as prompt cache and are dropped, never written page by page, when RAM has no room (that wait stalled the conductor after an 80k far prompt). Open: the hang seen twice in the merge check did not come back in 10 runs (cause not found); main's lent steps with the CPU's share give one of two first tokens at a near tie (not the far pages') |

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Host memory"

- [x] A MoE model and a dense model on one device under memory pressure: the dense
      checkpoints were freed on the watch's thread, the path that deadlocked. Since `fix/far-free`
      (2026-11-24) the gate of page-locked memory is the process's: every engine's frees wait
      for the steps of all of them (not tried with two models).

### From "Whole features not started"

  - [x] **A deadlock in libcuda between the RAM thread and a device thread** (found 2026-11-24 by
        the `real_cache` test `flash_next_resumes_a_conversation_from_the_cards_from_ram_and_from_ssd`,
        whose `--context 20480` overflows since a named context does; 2 of 2 in the suite, 2 of 3 of
        the same flow outside it, `/var/tmp/ares-c2load/hang/`): `ares-ram` frees page-locked
        checkpoint chunks (`PinnedRam::sweep_checkpoints` from `Machine::look`) with
        `cuMemFreeHost`, which synchronizes with the devices and holds the driver's lock, while
        `ares-device-1` in a step waits for that lock in `cuMemcpyHtoDAsync` (`CudaSlots::set_entry`
        from `ExpertStore::poll_cpu`), and the step waits for it: the engine stands still, and
        does not stop on SIGINT. Very likely the hang seen twice in the merge check of item 6.
        Fixed on `fix/far-free` (2026-11-24): page-locked memory changes only while no device runs
        a job, for every thread of the process (`ares_cuda::gate`; "When page-locked memory
        changes" in `dev/architecture/host-memory.md`): a free or an unlock asked for during a step
        is done when the step ends, a lock or an allocation waits for the end or fails inside a
        step, and what a step needs locked (checkpoint chunks, far pieces, a tier piece) is locked
        before the hand-over. The cause: a step's kernels spin on the device for what only the
        device's host thread brings (an expert's slot entry), and that thread needs the driver's
        process-wide lock to enqueue it; `cuMemFreeHost` from another thread takes the lock and
        waits for the devices' queued work, so neither moves. Proof on the two local cards: the
        test's requests alone (`repro.py`) 10 of 10 with the same answers, `real_cache` 5 of 5
        (4 of 4 each), the merge check's bench flow (an 80k prompt beyond VRAM at `--ram 8G`,
        then a short request) 5 of 5 without a hang on this branch and 5 of 5 on main be0023db
        (no hang on main either); every GPU suite of the server (`real_cache` among them,
        and `real_elastic` once more with a helper that takes 4 GiB: the tier gave memory back
        while generating), `qwen4exp_runtime` (the known near tie of
        `flash_next_writes_with_drafts_what_it_writes_without` once), `qwen4exp_device`,
        `qwen4exp_prefix` and every binary of `ares-runtime` passed; alternating A/B of Flash-Next's decode and an 8k
        prompt on both cards at `--ram 8G` and of the 27B at the owner's settings: the same as
        main's (`dev/experiments/host-memory.md`, "Page-locked memory between steps"). Notes:
        `/var/tmp/ares-farfree/notes.md`.
  - [x] K and V beyond VRAM on two cards: not run with the real file (the made-up model's test
        passes on the RTX 4070 + RTX 3060). Run on 2026-11-24 (`fix/far-free`): Flash-Next IQ2_XS
        on the RTX 4070 + RTX 3060 with K and V beyond VRAM, `real_cache`'s resume test (a
        named context of 20 480 tokens, 18 432 of them in VRAM) 5 of 5 and its requests alone 10
        of 10, and the merge check's 80k prompt beyond VRAM 10 of 10.
  - [x] **The hang of a request after a long far request** (merge check 2026-11-23): the
        deadlock above, found and fixed 2026-11-24 (`fix/far-free`): the bench flow of the merge
        check ran 5 of 5 without a hang there and on main.
  - [x] **Lent wide steps with K and V beyond VRAM "fail since the prompt kernel"** (merge check
        2026-11-23, `integrate2`): neither the far pages' nor the prompt kernel's (`kvfix`,
        2026-11-24). The failing case compared the first token of a prompt read in lent steps (32
        rows on a loan) with the CPU reference, which ranks it 0.43 logits above the next (58 over
        22). Lent steps answer 22 in about half the runs on a quiet machine, with every page in VRAM
        as often as with K and V beyond the first page, and on main without this branch: see "Lent
        steps with the CPU's share" below. The far test's lent case now uploads every miss and
        asserts that the far pages change no token (`k_and_v_beyond_vram_change_no_token`, 5 of 5
        runs in all three placements; the old case failed in 3 of 4 runs of the branch on acb2587).
  - [x] **A request after a long far request stalls or hangs** (merge check 2026-11-23,
        `integrate2`; `kvfix` 2026-11-24, two local cards, IQ2_XS, `--ram 8G`, the model's own
        context, the bench's flow: a short request, a prompt of 80 507 tokens, an unrelated short
        request, `/metrics` around each). The stall is explained and fixed: when the long request
        ends its far pages (490) rank as prompt cache and the next trim takes them all off the
        devices; the RAM level had no room at `--ram 8G`, and each page of a text long enough for
        the drive waited for the writer to free a buffer (up to 250 ms) while the conductor ran no
        step and answered no command. Counted with an instrumented build: that trim took 827 ms with
        490 far pages, 463 waits for the writer and 468 transfer calls (the writer kept up that
        time; at the merge check's load it was half a minute without an answer to `/metrics`). Far
        pages now go into RAM when it has room and are dropped otherwise (`cache.rs`,
        `far_pages_that_find_no_room_in_ram_are_dropped_without_waiting_for_the_drive`). The hang
        did not come back in 10 runs of the flow (on 2340ca0 under gdb: this branch, an instrumented
        build and the merge check's very binary; without gdb: before and after the fix; rebased on
        main 93d11310: five with the fix; loads 3 to 56): its cause is not found, and the engine's
        step watchdog (600 s) stays the guard. Seen on the way, neither of them the far pages': the
        decode right after the long prompt is slow when the machine is busy (3.0 to 15.9 tokens/s
        with the load; the pass's loan gives its slots back empty and the CPU computes the misses
        beside other programs; main 93d11310 7.5 after its own 80k prompt, this branch 7.5 and 7.8
        at like loads); and "a step stood still for 30s while rows of the n-gram table were read"
        (one of the five runs with the fix, at load 27 to 34), which main 93d11310 gives as well
        (its 80k prompt at load 9 to 23; its warm-up at load 56 stood still on a pass's experts):
        the 30-s checks count from the start of a wait, not from the last progress. Main against
        this branch on the same cards: 10 752 VRAM slots (43.8 %) against 13 056 (53.1 %), the 80k
        prompt in 105.5 s against 104.6 s (load 9 to 27); with the fix at load 5 to 8 102.6 s
        (97.0 s before the rebase, 202.4 s without the pass's copy).

### From "Cleanup (the owner, 2026-11-23: the code must be clean although agents wrote it)"

- [x] The loaders (finding 1: `load.rs` and `qwen4exp.rs`, their copied tail, the qwen35
      memory estimate in the server), with them the names `qwen4exp::Split` (a saved spread)
      and `load::watch` beside `Models::watch`: after the model crate's phase 2 merges.
      Done (merged 2026-11-24, cleanup phase 2 of the loaders): `load/qwen35.rs` and
      `load/qwen4exp.rs` in phases with one tail (`load/start.rs`) and one RAM budget
      (`load/ram.rs`), the 27B's estimate in the model; `SavedSpread` and `load::machine`.
- [x] Other crates' parts of the audit (`server.md`), not this branch's crates: `ToolFormat`
      and the thinking format written in `ares-tokenizer` and `ares-api` (and copied in
      `ares-model`'s `chat.rs`), three JSON value types, tools as raw JSON across the API,
      the order rule of the samplers and the mirostat range checked in the runtime as well,
      `LOGPROBS_MAX`/`BIAS_MAX` beside the API's limits; the mask-bit test three times
      (grammar, sampler), llama.cpp's defaults in the sampler and the runtime; test-only
      public code (`Breakers::of_tokens`, `keeps_history`/`opens` computed and dropped).
      Done (merged 2026-11-24): the formats are the API's types, read once by the tokenizer;
      `SamplerKind::order_problem` and the limits const-asserted equal; the mask bit through
      `mask_allows`; the runtime's defaults from the sampler's constants; `Breakers::of_tokens`
      the tests' own, `keeps_history` gone, `opens` a test helper. Typed tools, one JSON
      value type and the mirostat range stay open in `OPEN.md`.
- [x] Found on the way (main c1927ac8 as well): `ares-tokenizer --test model`
      `renders_the_prompts_of_llama_cpp` fails with `ARES_TEST_QWEN35_0_8B` set (request 26,
      a message of two text parts, renders nothing here where llama.cpp renders them joined);
      `real_flash_next` `flash_next_answers_in_both_placements` asks for `--split tensor` on a
      machine of one card and fails instead of saying it is skipped. Both fixed on `fix/far-free`
      (2026-11-24): the rendering was right (the server joins a message's text parts as they
      are, llama.cpp's prompt is "part1part2"); the test joins them now. The placements test
      is skipped with a note when one device is named (both splits need two).
- [ ] Phase 2 (after `kv/overflow`, `flash/ssd-stream`, `attn/long-context`, `sched/parallel`):
      one driver of Flash-Next's parts, the step record decoded once, one memory policy, budgets in
      one place, the conductor split, the two model loaders of the server, shared device helpers.
      Runtime part done on `cleanup2/runtime`: the leftovers of phase 1 (the cache's four oracle
      suites are one test binary with one rig, `tests/cache/`; `testing::TestMachine`); one policy
      for page-locked memory (the books count and rank in one place, `Books::allowances`; the
      broker of `ares-core` is gone); budgets in one place (the server's RAM default in the load
      alone, `Durable::page_room`, `SsdConfig::floor_pages`, `StoreConfig::slots_for`, the store's
      rounds and resident bytes, the store's commit says its bytes); the cache's RAM and SSD
      levels bring a run back through the same helpers, and the cache is split into files; the
      picks of a route record classified once (`routed_rows` in three parts); the draft policy in
      `ares_sched::Drafting` and the conductor split into files by concern (`RecordKind` for what
      a record is for); the old expert fill gone (`plan_listed_fill`, the bench reads ahead
      through `ahead`); admission without the restore of spilled pages (a spilled sequence
      computes again, as the runtime does; the model test's world too); the store's step clock,
      odds, power-of-two classes, read submission and key files once; the expert store's, the CPU
      kernels' and the paused steps' test rigs shared (`testing::Pauses`). Left: the running
      means of the store (their weights differ: one would change what the store decides early
      on); the transit sizing that lives in the model (`plan_transit`) and the store config built
      twice (the model's files); one host budget for both server loaders (the server's phase 2
      merges them); `write_record` and `step` of the conductor in smaller parts; the host
      executor's places on `PlaceChunks` (its oracle plays the strictest device, which the
      chunks are not); the record's windows and the text half of `impl Model` in both models
      (the model's phase 2); the cache's transfer counter kept twice (a metric's name).
      Checked on the two local cards (2026-11-24, on main 8493ed1e): every GPU suite of the
      runtime, `qwen35_device_runtime`, `qwen4exp_device`, `qwen4exp_prefix`, `real_devices`,
      `real_cache` and `real` pass; `qwen4exp_runtime` 30 of 30 but once
      `a_request_cuts_in_between_the_pieces_of_a_long_prompt` (the split by layers: the long
      prompt's first token another), which passed alone three times on this branch and on main
      and in the whole suite again. Speed, VRAM and the plans as on main (27B at the owner's
      settings, Flash-Next IQ2_XS at `--ram 8G`, alternating).
      Server part on `cleanup2/server` (all but the loaders): one harness that starts the `ares`
      binary in every test (`tests/common/process.rs`, `socket.rs`); one table of options
      (`options::OPTIONS`: names, spellings, scope, help, and the defaults' readers) that
      `--help`, the environment and both files come from, and one reader of the two files
      (`ini.rs`); `wire.rs`/`routes.rs` split into `openai.rs` with `openai/{request,reply}.rs`,
      `generate.rs`, `metrics.rs`, `routes.rs`, `crate::reply` renamed `sorter`; one way to
      serve (`Models::fixed`, `router`, `router_with`, `serve`, `serve_until` and two `Config`
      fields gone: tests serve their engine as presets); one segmenter for the blocks of
      Messages and the items of Responses (`answer::Segments`); the in-process GPU suites'
      runtimes, router and gate once (`tests/gpu`, `common::model_file`); `real.rs` split into
      `real`, `real_drafts`, `real_flash_next`. Left, with the reason:
  - [x] One variable name per test model file across the workspace (the owner, 2026-11-24):
        the table in README.md's "Tests" (`ARES_TEST_QWEN35_0_8B`, `ARES_TEST_QWEN38_27B`,
        `ARES_TEST_QWEN38_27B_IQ2`, `ARES_TEST_FLASH_NEXT`, `ARES_TEST_FLASH_NEXT_MTP`; the
        kernels' `ARES_TEST_GGUF` is any file). `Question::option_texts` is used by its tests
        only (System One's groundwork, kept).
      Model part done on `cleanup2/model` (2026-11-24): one opening path and one flow for
      Flash-Next (`device/open.rs`, `device/flow.rs`: the runtime's device threads, the session and
      the roles' trial open through the same stages, the threads and the session step through the
      same `Flow`; `runtime.rs` split into `runtime/{parts,meeting,device,plan}.rs` and
      `part/frame.rs` out of `part.rs`; `ModelOptions` for what the server builds), the step record
      read and the drafts handed out once for both models (`blocks/step.rs`), their text half
      (`chat::ModelText`), one arena of weights (`blocks/arena.rs`: Flash-Next now uploads through
      the 27B's file reads), the references' operations (`crate::reference`), the made-up
      Flash-Next on the GGUF builder, small copies (`MTP_KV`, `invalid`, the draft vocabulary
      reader, the segment checks). `Session` stays a face of its own, not one over the executor as
      the 27B's: its per-layer probe stops both devices at every layer and reads both parts, which
      the runtime's device threads could only do with a barrier inside the production step, and
      the tests read the logits on the host, which the runtime samples on the device. Left for the
      model: the tools (`qwen4exp_run`, `qwen4exp_prune`) measure the session, not the runtime
      (moving them drops `--owners routing` and the host time of a step: the owner's call);
      `roles::Runner` stays the trial's own timing loop (no store, every expert in zeros);
      Flash-Next's embedding staging written twice (`part.rs`, `part/pass.rs`), the profile marks
      of both parts, `Balance` beside `Meeting`, the record's token offset read by hand (an
      `ares-cuda` accessor), the server's draft vocabulary reader (server phase), `Qwen35::new`
      reading its tokenizer while `Qwen4Exp::new` is given one, the large functions left
      (`enqueue_pass`, Flash-Next's `Part::assemble`, `Config::from_gguf`, `Drafter::pass_layers`;
      `Frame::widest` and the 27B's `open_balanced` went down to 277 and 222 lines in stages) and
      the names of the audit's finding 13. Verified on the two local cards (2026-11-24, rebased on
      8493ed1e): every GPU suite of both models and the server's `real`, `real_cache`,
      `real_devices` as on main (the known near-bound and tensor-split failures only); alternating
      A/B against main: Flash-Next and the 27B decode, an 8k prompt and two at once as fast, the
      same plan and the same role kept at load. The shared arena uploads from the file's mapping
      and reads with ordinary reads only on a machine short of memory for the load (less than
      twice its bytes available), the 27B's reason for them; neither model loads slower than on
      main (alternating loads, warm and cold page cache, medians): the 27B on the two local cards
      warm 3.2 s against 5.9 s for the weights (ready 9.6 against 12.2 s), cold 13.3 against 14.2 s
      (ready 23.4 against 22.7 s, means 24.1 and 23.8: noise); on the test machine's RTX 4060 Ti
      (the 27B at IQ2_XXS) warm 1.9 against 2.3 s (ready 7.1 against 7.6), cold 6.8 against 7.4 s
      (ready 14.4 against 15.1). Flash-Next uploads its 3.3 GiB of dense weights as main does: on
      the local cards 0.3 s warm and 0.6 s cold, on the test machine 0.9 and 2.0 s, time to ready
      and the role trial alike.

## 2026-11-23

### From "Against the owner's production (2026-11-23, the 27B UD-Q4_K_S on the RTX 4070 + RTX 3060)"

- [x] Long prompts against the fork: the prompt attention kernel (from the open item "Long
      prompts" of that section, which stays in `OPEN.md`).
      Built (`attn/long-context`): a prompt kernel of 128 vectors (`kernels/attn_prompt.cuh`) with
      integer scores over quantized K (and the tile mean's dot products from the same int8
      products), P V summed in f16 per chunk of 32 cells, and phases in which decode and the tensor
      cores overlap; the planner cuts a prompt's tiles into pieces that fill the kernel's waves and
      orders them so the blocks running at once share K/V in the L2 cache. One attention layer, a
      step of 256 rows at the end of 98 304 cells, K and V in `q5` (the fork `q5_1`): RTX 3060 with
      3 KV heads 79.9 ms before, 21.0 ms now, the fork 13.5 ms; RTX 4070 with 1 head 11.3, 3.0 and
      2.1 ms. At 32 768 cells 26.6, 7.4 and 4.6 ms on the 3060. Output closer to the reference
      than before (integer scores).
      Tried and dropped: a single K tile (2.5 % slower), the decode's lanes on rows (q8
      3 %), cells shared out over one wave of blocks reading several pieces (15 % slower: no L2
      sharing), P V with f32 sums (11 % slower), 8 vectors per warp in 16 warps (no faster).
      End to end at these settings (the bench of `/var/tmp/ares-tools/prod-ab/`, prompt suite, two
      rounds alternating with main, load 1 to 5): 2k / 32k / 98k 1131 / 1000-1002 / 709-710 tokens/s
      against main's 1082-1103 / 653-655 / 324 and the fork's 840 / 944 / 752: 32k is reached, 98k is
      6 % short. The state slots of the longer prompt tiles cost about 10 MB of VRAM on the 4070 and
      6.5 MB on the 3060 (the 27B three ways: 874 / 1754 / 1938 pages against 890 / 1770 / 1946).

### From "In work now"

Merged 2026-11-23 (verified on the two local cards): CUDA built and run only on the tested classes (Ampere 8.6 and Ada 8.9; a device of another class is refused and is to run through Vulkan), the Flash-Next agent-turn failure at turn 2 (a checkpoint given up while its restore waits keeps its save of the same record), the public README with the thanks, THIRD-PARTY-NOTICES and the index of the documents.
Merged 2026-11-23: the prompt attention kernel (scores over quantized K in integers, L2-shared tile order); at the owner's production settings the 27B reads 2k / 8k / 32k / 98k at about 1130 / 1130 / 990 / 705 tokens/s against the fork's 840 / 994 / 944 / 752.
Merged 2026-11-23: Flash-Next's prompts layer by layer (TokenWing's order, interruptible between micro-batches, two devices, checkpoints inside a pass, images in passes), the CPU share of wide steps, forms per dense type, loans in pieces, the anchor fix for cached turns, the heat of a pass counted per row; a 17k prompt 2.6 times as fast locally, a short request during it first token 0.8 s against 13 to 16 s.
Merged 2026-11-23 (the two local cards): the second card's role timed at load, capped cards measuring their prefill path, the transit budget (the 196-row test), the 27B at the owner's production settings planned and run (the attention's partial states bounded by the planner).
Merged 2026-11-23 (verified on the two local cards): the CPU kernels for every expert type, image input pieces 4 to 8, request priorities and the server items, the slot-number crash fix.

| Piece | State |
| --- | --- |
| Inkling 2 Mini and Flash-Next on the newest Strata on the test machine (the owner, 2026-11-23: Inkling could replace the 27B), with Inkling's quality against the 27B for the owner's use | done: Flash-Next under "Found against Strata"; Inkling on Strata set aside (the owner: Strata forces its thinking off) |

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Found against Strata on the test machine (2026-11-22; RTX 4060 Ti 16 GB, 62 GiB RAM, UD-Q3_K_XL)"

- [x] The agent suite on the test machine (main ae3a334, 2026-11-23, UD-Q3_K_XL at 40G on the 4060
      Ti): the second turn failed every time in its first step ("the rows of slot 0 start at cell
      650 and its sequence is at cell 512") and the model was taken down. The first turn's end
      state was saved with the next record into the one place; the second turn restored it in
      that record, and the places for the checkpoints inside its pass, taken before the record
      was written, gave the end state up: its save was dropped and the restore read the first
      turn's anchor (fixed on `fix/turn-cell`: the save stays while a restore of the record reads
      the place; tests in `runtime/tests/cache/prefix.rs`, `cache.rs` and `qwen4exp_runtime`, the last
      failing as the suite did before the fix). With the fix, two sessions on the test machine
      passed all six turns (first ask 622/1 cached 5.5 and 6.5 s, 6832/648 12.6 and 12.8 s, the
      third turn from the second's end state 1.8 s; asked again 622/512 4.3 and 4.8 s, 6832/6784
      15.3 and 15.5 s, 6915 or 6974 tokens 1.5 and 2.0 s). Seen on the way: with one place the
      second turn's checkpoints inside its pass find none (the place waits for the restore), and
      the end state it restores is given up for nothing, so the turn keeps only its end state;
      the first turn's anchor at 512 was a root as well, so its end state pushed it out (the end
      state keeps its anchor only from a plain one); and the conversation asked again replays
      6784 rows from the start (15 s against Strata's 0.33 to 0.47 s): the one place holds
      another state, and the tail at 648 that has one lies off a path through full pages.
      Locally (IQ2_XS, `--ram 8G`) main failed the same way on the RTX 4070 alone and on both
      cards, and the fix passed all six turns on each; `real_cache` passed on both cards.

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Experts from the SSD"

- [x] Long prompts layer by layer over the whole prompt, interruptible (branch
      `flash/layer-major`, 2026-11-23; `architecture/flash-next-prompts.md` "Long prompts layer by layer"): a step
      of one prompt wider than the own scratch is a pass of micro-batches of the own rows per
      layer and the layer's experts once for all rows, about 56 KiB a row kept on a loan sized
      for the prompt, in every placement; it pauses between two micro-batches when a request
      or a cancellation comes (or its time as a piece is up) and goes on where it stopped.
      Tested against the reference in every placement and for a request that cuts in (its
      first token after 16 ms where the prompt took 61 to 78 ms, made-up model).
      Measured locally (IQ2_XS, `--ram 8G`, 17k prompt, A/B against `flash/prompt`, with the
      checkpoints inside the pass): the prompt 2.1 times as fast on two cards and 3.0 on one,
      the tokens after it 1.2 to 1.3 times, a request that comes during it its first token
      after 0.8 to 1.8 s instead of 21 to 49 s (`experiments/flash-next-prompts.md`). **Not measured yet:** the
      test machine against Strata (2k/8k/32k, UD-Q3_K_XL and UD-Q4_K_XL; it was off). Built with it: the CPU's share of a pass of up to 2048 rows;
      the next layer's experts that no tier holds read into staging while a layer's
      micro-batches run.
      Checkpoints due inside a pass are saved there layer by layer (the step is no longer cut
      at every 16 384 tokens and at the anchor).
      Rebased onto main 2026-11-23 (with images and priorities): a pass takes an image's
      staged rows and every row's rotary position (tested against the CPU reference with a
      made-up image in every placement), a paused pass keeps its image rows and its place
      (it gives way neither to a shortage nor to a better priority), and the record's header
      grew to 44 words (images 38/39, the pause 40, what was asked 41, saves inside 42/43).
      Found in the move: the GPU test of checkpoints inside a pass ran without them (its
      model did not forward `saves_inside_steps`); with them, a pass on both devices could stand
      still when a save locked its place's memory between two micro-batches (fixed: the places
      are locked before the pass), and `real_cache`'s Flash-Next test at `--ram 8G` failed a
      step when places reserved for saves inside a step were handed the same room of the limit
      (fixed: they count as pending locks, and are taken before what leaves the devices is
      settled). The second turn of that test, resumed from the cards (9827 of 9851 tokens
      cached), took 10 to 12.7 s against main's 2.5 to 3.6 s: the first turn's pass reserved
      places for the checkpoint at 8192 and the anchor at once and lost the anchor (1635 rows
      replayed instead of 99; fixed: places the latest first, room made at each attach), and the
      heat counted the whole pass as one step, so the tiers kept what came before the prompt
      (fixed: a wide step counts as the steps of own rows it stands for). Now 2.9 s against
      main's 2.5 to 3.3 s, alternating, and the test asserts fewer than 256 rows replayed.

### From "Parallel requests against the owner's production (2026-11-23, `bench/parallel`)"

- [x] A long prompt next to a generating sequence took 2.5 times its time alone (the fork 1.6 to
      1.8): `Share` gave generation half of the time (the slowdown of 2) and the prompt ran in
      pieces of 256 rows, one group of the tensor split, which runs without the overlap of a second
      (2026-11-23, `sched/parallel`). Now generation keeps a floor rate (`--generation-floor`, 20
      tokens/s by default, measured from its rounds, at most half of the time) and a piece of the
      27B under a split is two groups (512 rows). Measured (old main b8bb506, alternating, load 2 to
      33): 8k beside a generating request 21.0-21.5 s before, 10.6-12.9 s now (fork 15.2-15.5);
      32k 124.7-127.5 s before, 70.9-73.3 s with pieces of up to a second (82-100 s with half a
      second); the generation keeps 17-20 tokens/s (fork 21-26), its longest gap 0.5-1.0 s (fork
      0.5-0.6 at 32k). On main acb2587 with the new prompt attention
      (two rounds, load 10 to 42): 8k 18.6-18.8 s before, 9.5-15.2 s now (fork 15.2-15.5), 32k
      84.1-87.2 s before, 57.3-59.9 s now (fork 59.8-59.9), the generation 30-31 tokens/s before and 18-20 now, its longest gap 0.31-0.40
      s before and 0.51-0.62 now (fork 0.51-0.66). Priorities stay: a short request of priority 0
      beside a generation and a 32k prompt 0.18-0.30 s (before 0.29-0.40), of the same priority
      40.9-41.0 s (before 37-47: it waits for a place, and while a request waits generation gets half,
      `prompts_ran`'s `crowded`; at the floor it waited 68-73 s on the old attention). A 98k prompt
      beside a generating request 233 s (old main 754-763 s, fork 209-219), the generation at 20.8
      tokens/s, its longest gap 0.52 s. Two at once and 3 / 4 at once unchanged within one round
      each (pair sums 110.6 / 111.7 / 124.6 against 108.5 / 108.3 / 127.9; many 3 106.0 against 106.2;
      many 4 97.7 against 115.1, whose queued requests' texts differ run to run and copy by n-grams
      differently).
- [ ] `--parallel` (the owner's call; measured with this branch on main acb2587, 262144 cells): at 4,
      3 / 4 / 8 at once 133 / 127 / 125 tokens/s at load 37 to 45 (the parallel comparison at 4: 117-120
      / 139-149 / 143-150), every first token of 4 within 0.33 s, a 32k prompt beside generation
      56.7 s; VRAM 11 523 / 10 523 MiB (+80 / +440 over 2). At 8: 136 / 150 / 210 tokens/s, every first
      token of 8 within 0.51 s, the 32k prompt 57.3 s; VRAM 11 617 / 11 425 MiB (+170 / +1 340). At 2
      a third request waits for a place (5 to 8 s here, 19-55 s beside long work), and while it waits
      generation gets half of the time. Recommended for agents with sub-agents: 4 (the third and
      fourth get places at once for 0.5 GiB of the RTX 3060); 8 when the 3060's last GiB is free.
- [x] The floor by setting (27B UD-IQ2_XXS on one RTX 4060 Ti, no drafter, generation alone 31
      tokens/s, a 16k prompt alone 21.5 s, beside a generating request): floor 0 23.2 s
      (generation 3.0 tokens/s, the token a piece), 10: 30.5 s (10.1), 20 and 30: 45.4 s (17.2: 31
      tokens/s alone cannot keep 20 in half of the time, so half), the old split 45.5 s (17.2).
      The default of 20 is the fork's generation beside a prompt; the owner's call.

### From "Cleanup (the owner, 2026-11-23: the code must be clean although agents wrote it)"

- [ ] Defects first: the watchdog's blind device waits (in `sched/parallel`); a server test that
      writes the user's `~/.cache/ares`; a type list that must match a kernel by hand; Flash-Next
      tests scheduling with another price estimate than production; two options missing in `--help`.
      Done on `cleanup/server-model`: the cache, the Flash-Next tests' estimate, `--help`. Done on
      `cleanup/cuda`: the type list (the host reads from each decode kernel whether it walks
      several tiles of rows; a test fails a row a kernel leaves unwritten).
- [ ] Phase 1 (no conflict with the branches in flight): server test helpers and handler copies,
      `Models::Fixed`, dead code; one table of weight types in `ares-cuda`, dead kernels, test
      helpers; the replaced "pass in waves" design, copied quota code, unused public API in the
      runtime; the file layer written four times and test copies in `ares-model`. System One stays.
      Server and model part done on `cleanup/server-model`; moved to phase 2 from it:
      `Models::Fixed` (mock engines answer to several names, which a router of one preset does
      not: decide the names first), `plan::default_ram` and `parse_available_ram` (`plan.rs`,
      `load.rs`), one segmenter for the blocks of Messages and the items of Responses.
      Runtime part done on `cleanup/runtime` (about 1,300 lines fewer): the pass in waves and its
      docs, the copied give-back of the loans and the RAM tier (`Tier::vacate`), the thinking
      followed once (`watch::Tracker`), unreached API in the store, sched, state and store
      crates, the store's small copies, the drafting suites' rig, docs that drifted. Left for
      after `kv/overflow` (its files): `PinnedRam::machine`, `RamLevel::shed`'s argument, the
      `on_cache` hook's allowance, `PrefixIndex::lookup` beside `lookup_chain`, the runtime's
      re-exports nobody outside uses (`lib.rs`), the cache oracle rigs of `ssd.rs`, `ssd_ram.rs`
      and `ram_cache.rs` (with `prefix.rs`'s), the module docs of `cache.rs` and `ram.rs`.
      Moved to phase 2: the old expert fill (`fill` / `predicted`) is not the MTP layer's alone:
      `Session::fill` and `Runtime::fill` (13 sites in the Flash-Next GPU tests, `qwen4exp_run`)
      fill from a list through it, and the bench's predictors read through `predicted` between
      steps, where `ahead` does nothing; one way in (a `TierFill` planned from a list) changes
      what those tests and the bench do. Admission's restore of spilled pages (the conductor
      always says 0 pages; the model test of admission still simulates it). The rest of the
      store's small copies (running means, odds, the step clock, the profile and table files,
      the CPU kernels' test rig, the read submission of `read` and `transit_read`), the expert
      store's test rig and the fake executor of `tests/images.rs`.
      `ares-cuda` part done on `cleanup/cuda`: one table of weight types (`kernels/types.rs`), the
      dead kernels (draft carry, nine delta-rule shapes and the fork's design, the Q4_K reference,
      the ordered sum of the expert rows, the break-even probe, after `kv/overflow` the attention
      kernels' hand-build switches `ATTN_PROBE` and the like), `tests/common` (the attention test
      included), `ares_quant::act` for the int8 activations, `invalid`/`check_align` once. Left,
      with the reason:

### From "Owed on two cards (the owner's cards are production; run when the owner frees them)"

Run on 2026-11-22 evening (main 753dcb1, RTX 4070 + RTX 3060, peer access, load about 30 from
the agents' builds):
- [x] The two-device suites: `real_cache` 4 of 4, `real_devices`, `qwen35_device_runtime` (the
      attention-split placements) 4 of 4, `qwen4exp_device` with `ARES_TEST_SPREAD=tensor` 23 of
      23 and with `ARES_TEST_CORE_SHARE=1 ARES_TEST_SHARED=mixer` 23 of 23; `real` 21 of 22 (the
      known intermittent `flash_next_drafts_with_its_mtp_block` that verifies no draft at all).
- [x] The MTP block's first links with both cards on every layer: a server session (IQ2_XS,
      `--ram 8G`, MTP) accepted 626 of 1078 drafts, decode 17 to 52 tokens/s; the poor first
      links the server agent saw are not seen again.
- [x] `qwen4exp_runtime` `the_mtp_block_follows_wide_steps_that_start_at_any_cell`: passed with one
      device, failed in the split by layers on two: the warm-up found 376 VRAM slots for 8 layers x
      48 experts. The cause: the transit slots of wide steps (two rows' picks, six here) came out
      of a device's bytes for slots, which the plans cut to its experts and a 64th (three slots of
      192), so each layer had 47 slots for 48 experts (the 64-row tests do not fill every slot).
      Transit now comes on top of that budget as far as the device's room goes (`plan_transit`,
      `devices/load-choice`); the test passes in all three placements on the two cards.
- [x] The 27B at the owner's production settings (`--split tensor --shares 0.625,0.375 --context
      262144 --parallel 2 --kv q5 --drafters mtp --draft-vocab`, `--ram 16G`): the heads by the
      context planned the RTX 4070 above its free memory (the MTP block's K and V and whole heads
      were not counted) and the load died reserving the recurrent state. Fixed
      (`devices/load-choice`, the test in `load.rs`); an asked context the devices cannot hold is
      refused by estimate. On the cards (2026-11-23): 0.319 of the heads on the 4070, the plan
      11.22 of 11.52 GiB there; loaded in 8 to 16 s and answered, the 4070 at 11 523 MiB with
      177 MiB free beyond the 256 MiB left unreserved, the 3060 at 9 931 MiB.
- [x] Flash-Next's second card: the role timed at load (`devices/load-choice`): on the two cards
      the trial keeps experts only, and the runs agree (warm prompts 20 to 29 % faster than
      with the share, decode as fast; `architecture/devices.md`, "The second card's role is timed at load").

## 2026-11-22

### From the head of `OPEN.md`

Updated 2026-11-22, `main` at the prefill path measured per type on full-rate cards (after the
shared layer's steps 5 and 6: one MTP driver, drafter kinds `ngram`, `mtp`, `dflash`; nothing
pushed).

### From "The order of 2026-11-17 (after the gap study; items 1 to 7 done, the rest is a store)"

1. [x] Pin Claude Code's billing stamp at the start of its system prompt as upstream llama.cpp does
   (the owner, 2026-11-17: on `/v1/messages` the five characters after `cch=` become `fffff`).
   Done 2026-11-22 (branch `cache/checkpoints`): it was there since 2026-11-12 with a stricter
   rule (letters and digits only); now upstream's to the byte.
2. [x] A root checkpoint at the end of the system prompt and the tool list, a checkpoint at every
   message boundary, and checkpoints at fixed distances inside a long prompt, for both models:
   a change in the middle of a conversation (an older tool result cut, an edited message) then
   costs only the rows from the last checkpoint before it (prompt cache; the owner, 2026-11-18).
   Done 2026-11-22 (branch `cache/checkpoints`, see "Prompt cache" below for what is left).
3. [x] Large prompt chunks for Flash-Next with expert slots borrowed in VRAM, batch and micro-batch
   adjustable, adaptive when another request waits (experts from the SSD). Done 2026-11-22
   (`moe/prompt-chunks`); a step cannot be cut, so a request that comes during a long lone step
   waits for it (owner's question below).
4. Done 2026-11-22 (`server/flash-api`): tools, structured output and the Messages and Responses
   routes on the real Flash-Next, and Claude Code itself against Ares on both models ("Server and API").
6. [x] K/V of a running sequence beyond VRAM for Flash-Next at long contexts (whole features). The
   owner's rule for the context (2026-11-22): without `--context` the model's own maximum (262144
   for Qwen3.8) on both models; what does not fit in VRAM overflows into RAM. On the 27B the
   attention heads move to the second card as far as the context needs (built on `devices/roles`).
   Built 2026-11-22 for Flash-Next (branch `kv/overflow`, "K and V beyond VRAM" in
   `architecture/flash-next.md`): the indexer's keys of every page and the K and V of the first pages (a
   thirty-second of the room for slots) in VRAM, the rest in page-locked RAM read over the link.
   The 27B's is not built: measured, its dense layers read every cell over the link (220 ms a step
   at 32k cells on a x4 link), see "Qwen3.8 27B".
7. [x] Priorities of requests (server). Done 2026-11-22 (branch `server/priorities`): measured on
   the test machine, a request of priority 0 next to busy requests of priority 5 has its first
   token in 21 ms where one of priority 5 waits 382 ms to 10 s (see "Priorities of requests" in
   `architecture/server.md`). The server part of item 8 is on the same branch.

### From "In work now"

Merged 2026-11-22 after the test machine's one card passed every GPU suite (`fix/mtp-and-slots`,
`moe/prompt-chunks`, `cache/checkpoints`, `devices/roles`); what needs two cards is owed below
("Owed on two cards").

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Found against Strata on the test machine (2026-11-22; RTX 4060 Ti 16 GB, 62 GiB RAM, UD-Q3_K_XL)"

- [x] **Crash** with the MTP block (`--drafters` the q2 block): "writing K and V reported status 0x4
      in the step before" (a row written straight into a converted entry of a tile-mean format;
      only the MTP block's K/V is `q4`) at the end of a long prompt after a few requests, 6 of 6
      times; the engine reloads (28-51 s). Not with drafts off. Cause: `blocks/mtp.rs` decided
      which rows of all launches that only write keys are a replay before the first launch, while
      each launch of Flash-Next's block prepares its K/V group: a prompt slice's tiles that no
      launch had given an exact slot yet looked converted, so the block wrote only the first cell
      of each tile of a prompt (its drafts read the rest unwritten), and a slice that starts at a
      cell that is no multiple of eight (behind the 3 cached tokens of a chat's start) had a launch
      that began in such a tile, was marked a replay as a whole and still wrote the next tile's
      first row, into its converted entry. Fixed 2026-11-22: each launch is decided when the ones
      before it are enqueued (`Driver::key_launch`), and a sequence's rows that write behind rows
      that do not begin a launch of their own. The exact pool's size was right (each launch is a
      step of eight rows for the group). The 27B's block prepares once per model step, before it
      follows: not affected. Test: `qwen4exp_runtime`
      `the_mtp_block_follows_wide_steps_that_start_at_any_cell`.
- [x] Why, from the logs: GPU hit rate of the experts 37-74 % against Strata's 62-83 % with
      about the same slots: Ares fills only 1267-1862 of about 4000 slots at the start and then
      moves 63-171 experts a token over the card's x4 link (33-102 ms a token waited); Strata
      fills every slot from a ranked profile, changes them slowly, and computes every expert
      missing from the GPU on the CPU from page-locked RAM (Ares computes 14-42 % there and
      uploads the rest). Strata also skips missed experts of rank 7 and lower by default (not
      like for like; Ares' `--expert-prune skip:8` is that). Built 2026-11-22 (see "Members,
      loans and promotions" in `architecture/experts.md`): the fill makes VRAM the hottest of every layer
      (at the start, after the warm-up, when idle); members change only by the rule of a
      promotion, at most what the link does in a step; a miss goes to the CPU or the link by the
      measured costs, and an upload that is not worth a member's place goes up on loan. Also
      found: the profile counted every row of a prompt, so it ranked what long prompts used (a
      decode's hit rate 16-23 % when VRAM was filled from the comparison's profile); a wide step
      now counts an expert once (profile format 2). Measured against the build of before, both
      without a profile (`experiments/flash-next.md`, "Members, loans and promotions, measured (2026-11-22)"): with the MTP
      block 20.0-37.6 tokens/s against 17.2-34.8 at 56G (mean +11 %), without it the same
      (16.4-21.9), at 24G +13 to 14 %; VRAM hit 59-72 % in both builds (Strata 62-83 %), 36-53
      uploads a token against 51-58, the CPU 18-26 % of the picks. Started from a profile of use
      (format 2) VRAM is full of the profile's hottest (3 473 brought by the fill) and the hit
      rate is the same, 59 % and 70 %: on this card the size of VRAM, not its filling, bounds it.
      Still behind Strata (40.0/43.0 with drafts, 27.2/28.6 without): the misses cost 26-56 ms a
      token, see the next item.
- [x] The CPU computed the types of UD-Q3_K_XL slowly (decoded row by row, 170-410 us per
      expert and row on the test machine, about an upload). Built 2026-11-22 (branch
      `flash/cpu-experts`): kernels for the types of every Flash-Next file at hand, as gate and
      up IQ3_XXS, IQ4_XS, Q4_K, Q5_K and Q2_0 (the MTP block), as down Q4_0, Q4_1, Q5_0, Q5_1,
      Q8_0 and IQ4_NL, with AVX2 and AVX-VNNI picked at run time and the decoded path as the
      portable fallback (a CPU without x86-64-v3 no longer refuses the CPU half); a job's down
      products start expert by expert. A job of one expert and row now takes 37-45 us on the test
      machine, three experts of a row 97 us (74 GB/s, near the memory's rate); the cost model
      sends nearly every miss to the CPU. See "The kernels for the types of the Flash-Next files"
      in `architecture/kernels.md`. Decode against the build of before (40G, the bench of the Strata
      comparison): with the MTP block 33.6 to 36.4 tokens/s against 26.0 to 26.1 (mean of six
      asks), without it 24.9 against 15.0; the wait for misses 18 ms a token against 28 to 41.
- [x] The agent suite on the test machine (b-a-on, 2026-11-22): a turn whose prompt is almost
      all cached was read as if it were not (6832 tokens with 6784 cached: 31.2 s, the same as
      uncached; 622 with 512 cached: 12.0 s); Strata 0.33-0.47 s. Two causes: the anchor of the
      first ask lost its place to its end state where the roots held all places but one (fixed,
      `flash/prompt` ebe5061, test in `runtime/tests/cache/prefix.rs`), and every small wide step paid
      a pass of uploads (the CPU's share of wide steps). To measure there.
- [x] A loan of VRAM after the tier had moved slots between layers (rebalance) could take a
      layer's given-up number for a slot's owner: the owner kept the slot inside the loan (its
      expert read lent memory) and a later loan stopped the engine ("as many slots were given
      as are lent", the base build on the local cards, the third loan of a session). Fixed
      (a36d379 on main; `flash/prompt`'s d6acafe is the same fix): a moved slot leaves no
      number behind.
- [x] The default context on that card is 23,552 tokens ("a thirty-second of the expert room"):
      too small for a 32k prompt. Since 2026-11-22 (`kv/overflow`) it is the model's 262 144, the
      K and V beyond 28 544 tokens in RAM on that card.

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Experts from the SSD"

- [x] Prompt batch and micro-batch adjustable as in llama.cpp (`--batch-size`/`-b`,
      `--ubatch-size`/`-ub`, 2026-11-22, branch `moe/prompt-chunks`): a prompt read alone takes
      steps of up to `--batch-size` rows (8192 for Flash-Next as far as its slots can lend the
      scratch), whose scratch lies in VRAM the expert slots lend for the step and take back
      after it (`expert_store/scratch.rs`); when another request waits, steps of prompt are pieces
      in the parts' own memory (`--ubatch-size`, 512 here) halved until their price is at most
      500 ms (`RuntimeConfig::piece_ms`). Measured (both cards, IQ2_XS, `--ram 8G`, 4096-row
      lent steps, which is what these cards' slots lend): prompts of 8k and 32k tokens 1.6 to
      1.9 times as fast as in steps of 512 rows (310 to 395 tokens/s against 191 to 239), 38 %
      fewer bytes read from the file (84 GiB against 139 for 32k). **Not met:** a request that
      arrives while a prompt is read alone waits for the lent step in flight (its first token
      after 13.5 s against 2.7 s with steps of 512 rows): see "Large prompt steps on lent VRAM"
      in `architecture/flash-next-prompts.md`. **Owner's decision:** keep the large lone step, bound it in time
      (fewer rows, less gain) until the step can be interrupted, or build the interruptible
      layer-major pass next. Left: TokenWing's order (each layer over the whole
      prompt, an expert read once per prompt, about 80 KiB a row instead of 0.5 to 0.7 MiB of
      scratch; the item "Long prompts layer by layer" below), a step on loan that a request
      can interrupt, the parcels of a split by layers sized for the widest step (80 KiB a row
      of VRAM on the receiving card and as much page-locked RAM, for good), the step's inputs
      in page-locked RAM sized for it (about 20 KiB a row a part), a loan that stays while the
      engine is idle after a request that ended with its prompt, both cards on every layer not
      measured (the machine had no driver with peer access).
      What was asked: a large micro-batch by default for a mixture of experts (Strata reads
      prompts in chunks of 8192: each expert is read once per chunk, not once per 512 rows);
      adaptive: a lone prompt takes the large chunk, and when another request waits the pieces
      shrink to a bound in time (the owner, 2026-11-17).
      Several requests at once ("parallel calling") belongs to the optimising phase.
      The large chunks need what Strata pays them with: the prompt borrows expert slots in
      VRAM and refills them afterwards (Strata `DETAILS.md:1383-1385`); Strata reads IQ2_XS
      prompts at 1250 to 2090 tokens/s on a 12 GB card with every expert in RAM, Ares here 40
      to 210 from the SSD (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`). Strata's trade for a short request behind a long prompt: 5.4 s
      against 1.8 s of waiting, 6.0 s against 8.0 s for the long prompt.
- [x] A request that arrives behind a lone long prompt: once it waits, steps of prompt are
      pieces of at most 500 ms by their measured price (`RuntimeConfig::piece_ms`, 2026-11-22).
      It still waits for the step in flight, which for a prompt read alone is a step of up to
      `--batch-size` rows on lent VRAM (see "Large prompt steps on lent VRAM"; measured
      there).

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Parity with the 27B (the owner's rule: what one model can, the other must)"

- [x] Structured output, tool calls and the Messages and Responses routes on the real
      Flash-Next file (2026-11-22, branch `server/flash-api`): the suites `real_tools`,
      `real_structured`, `real_messages` and `real_responses` run on any file the server loads
      (`ARES_TEST_API_MODEL`, flags in `ARES_TEST_API_FLAGS`) and pass on Flash-Next (two
      cards, `--ram 8G`; run before the driver with peer access was back, so on a split by
      layers) and on the 27B (two cards, tensor split, `--precision exact`, where a stream is
      held to be its whole reply). Flash-Next's replies are not repeatable, so two replies to
      one request (a stream and the whole reply, with drafts and without) are compared where
      they are the same tokens and checked each on its own otherwise. Claude Code against
      Ares: see "Claude Code against Ares" under "Server and API".

### From "Drafters (every model)"

- [x] Flash-Next served with the model's own context (262 144 cells, the default since
      2026-11-22) gives up a fifth of the VRAM slots (3.56 GiB of K/V and keys, all on the mixing
      card): decode 11 to 22 % slower, warm prompts half as fast. Since `kv/overflow` the context
      takes its keys and a thirty-second of the room in VRAM (743 MiB on the test machine's card)
      and the rest of its K and V in RAM.

### From "Prompt cache (every model; RAM first, then SSD, as in the owner's fork)"

- [x] A root checkpoint at the end of the system prompt and of the tool list, a checkpoint at
      every message boundary and at fixed distances inside a long prompt (the owner, 2026-11-18),
      for both models (2026-11-22, branch `cache/checkpoints`; "Sequence state" in
      `architecture/sequence-state.md`). Open: a root that comes back from the drive is a plain checkpoint
      again (the store keeps no mark); message checkpoints that later requests left behind are
      not written to the drive, so with one checkpoint place in RAM (Flash-Next at `--ram 8G`)
      an edit far back resumes from the root or from the drive's deepest checkpoint before it;
      with one place (measured 2026-11-22, Flash-Next `--ram 8G --cache`) the roots do not pay
      yet: the root leaves the place for later checkpoints and goes to the drive, but a prompt
      asks the drive only for pages the index does not hold, so a new conversation with the
      same 12k-token system prompt still replayed 18 304 rows (221 s); wanted: a checkpoint
      brought back from the drive for a page the index holds (`cache/ssd.rs` imports runs of
      pages with their deepest state only), or more places (int8 checkpoints);
      a message boundary is 1024 cells from the last checkpoint at least, and a prompt step
      that ends a piece before it takes it there (not measured: what a cut step costs
      Flash-Next with the large prompt chunks of item 3).
- [x] Claude Code writes a new billing stamp (`x-anthropic-billing-header ... cch=...`) at the
      start of its system prompt on every request, so every turn misses the cache from there;
      upstream llama.cpp, the owner's fork and Strata pin it (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
      Pinned on `/v1/messages` as upstream does (2026-11-22). Upstream (master, 2026-11-22)
      pins only `cch=`; Claude Code 2.1.301 writes no `cch=` but a `cc_version=2.1.301.<3 hex>`
      whose suffix changes with the first message (the server agent's capture), which upstream
      leaves alone, so Ares does too (the owner: as upstream). The root at the end of the tool
      list (written before the system prompt) keeps such a session from replaying the tools.

### From "Server and API"

- [x] Truncation and context shift (F18), what fits models with recurrent state:
      `truncation: "auto"` on `/v1/responses` drops the oldest whole turns until the
      conversation fits (2026-11-22, `server/priorities`). Context shift is not possible for
      them (a delta-rule state cannot lose a token; llama.cpp switches it off for such models
      too); a cut of a prompt to its last N tokens (vLLM, TGI, SGLang) is not built, see
      "The Responses API" in `architecture/api.md`.
- [x] `/v1/completions` and `/tokenize` read the text of control tokens (`<|im_start|>`) as
      those tokens, as llama-server does (2026-11-22, `server/priorities`); `/tokenize` takes
      `parse_special: true` and refuses `false` and `add_special`.
- [x] Thinking budget (K03): `reasoning_budget_tokens` and its other names, Anthropic's
      `budget_tokens`, model defaults; enforced as llama-server does, by a mask of one token per
      forced row (2026-11-22, `server/priorities`). Loop guard (K05): a cycle of at most 64
      tokens over the last 512 of the thinking ends the thinking, one token 256 times ends an
      answer (`length`). Open: the thresholds are fixed (no option); a passage loop of longer
      cycles (Strata's opt-in #728) is not looked for; `reasoning_control` is not built.
- [x] Chat and Responses routes: a system or developer message inside a conversation is a turn
      of the user, as on the Messages route (one rule in the reader of the chat body,
      `openai::request::place_system_messages`); an effort the template refuses is detected with its
      thinking (`refused_efforts`: Flash-Next `minimal` and `high`, the 27B `minimal`), `high`
      is then the template's default and another one a 400 that says so (2026-11-22,
      `server/priorities`).
- [x] Messages: `thinking.display: omitted` as Anthropic's API (an empty block whose signature
      carries the thinking back); `context_management.applied_edits` in the message and the last
      `message_delta`; `clear_tool_uses_20250919` served with all its members; the turns of
      `clear_thinking_20251015` counted as Anthropic counts them (2026-11-22,
      `server/priorities`). The placeholder of a cleared result is Ares' own text: Anthropic
      does not publish its.
- [x] A request with tools or an effort for an unknown model got a 400 where a 404 belongs (a
      server of one engine without presets): the name is checked before the template is asked
      (2026-11-22, `server/priorities`).
- [x] Priorities (S06): `priority` and `x-ares-priority`, the line, the places, the pages, the
      rows of a step and a full `--queue` by priority, preemption of a worse priority to the
      prefix cache (2026-11-22, `server/priorities`; see "Priorities of requests" in
      `architecture/server.md`). Open: a prompt of a worse priority still takes its share of the time
      next to generation of a better one (the floor of generation, not the priority, bounds it); a
      threshold of the gap before a running request gives way (SGLang's, 10 by default) is not
      there (any gap is enough); the randomized test of admission plays one priority.
- [x] A watchdog for a silent engine (S23): a step that runs longer than 600 s fails the engine
      (`RuntimeConfig::watchdog`), its requests end with the reason, the router unloads it and
      leaves its stuck threads behind (2026-11-22, `server/priorities`). Open: no option for the
      limit (a field of `RuntimeConfig`); the steps of the warm-up are not watched; the memory a
      stuck device holds stays taken until the process ends.
- [x] `/props` of a model that is not loaded: read from its preset and its file without a load,
      with `"loaded": false` (2026-11-22, `server/priorities`); `/metrics` of one is still 503.

### From "Server and API / Claude Code against Ares (2026-11-22, Claude Code 2.1.301, print mode)"

- [x] Claude Code could not talk to Ares at all: `context_management`, `thinking.display` and
      system turns between the others were refused, and effort `high` would have reached
      Flash-Next's template, which refuses it. Served since (see "The Messages API" in
      `architecture/api.md`).
- [x] A stream's response started only with the first token: Claude Code gave up after six
      minutes while Flash-Next computed its first prompt and sent it again. It now starts
      after 10 s without it (`Config::stream_wait`).

### From "Whole features not started"

- [x] Image input (F11; the owner's preset sets `mmproj`). Design: `/var/tmp/ares-survey/image-input-design.md`
      (2026-11-15). Both projector files are the same tower (`qwen3vl_merger`: patches of
      16 x 16, 27 blocks of width 1152, a 2 x 2 merger to the language model's width; 888
      and 866 MiB), with different weights. An image becomes (W/32) x (H/32) rows: 256 for
      512 x 512, 1024 for 1024 x 1024. Pieces 0 to 3 done:
  - [x] 0. Reference numbers from the fork's `llama-mtmd-debug` on the CPU: rows of both
        projectors for eleven square patterns of 4 to 1024 rows, with the sum of every node of
        the graph. The program reads no image file, so no own image, no wide or tall grid and
        no resize went through the fork; the resize is checked against Pillow instead.
  - [x] 1. CPU reference of the tower (`ares-model::qwen3vl`): config, size rule, resize, patch
        rows, the pass. See "The image tower" in `architecture/images.md`.
  - [x] 2. The tower on the device (`qwen3vl::DeviceTower`, `ares_cuda::TowerKernels`): the
        products on the f16 tile kernels with f32 sums (`ffn_down` padded to 4320 inputs),
        dense attention for heads of 72, norms with bias, bias and GELU, the 2-D rotation and
        the position table's resize as kernels of their own; residual and norm statistics in
        `f32`. Held to the exact reference per image (see "The image tower" in
        `architecture/images.md`). Open: the row kernels between the products take about a third of
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
  - [x] 4. The tower as a job the conductor plans (2026-11-22, `image/input`): between steps on
        the device the server chooses (the fastest link to the host for weights in RAM, the most
        multiprocessors for weights in VRAM), as prompt work charged to the share, a range of
        units per job while sequences generate; its VRAM taken off that device before anything is
        planned. Image spans next to tokens, the rotary position `(t, h, w)` of every row in the
        record (the cell in the row extras), the image stage (page-locked, 1024 rows at most) and a
        store of image rows (256 MiB of ordinary memory, least recently used first). See "Images in
        a prompt" in `architecture/images.md`. Open: which device computes a block faster is not
        measured; the share counts a piece with a tower job as one piece.
  - [x] 5. The splice in `qwen35` (checked against the fork on the 27B on both cards: the same
        prompt tokens, the same 80 tokens of a reply with f16 K/V and exact products; see
        "Images in a prompt"): the staged rows copied over the gathered ones on the part that
        embeds (one card, layer split, tensor split), the records of a split's groups at the rows'
        rotary positions, and the MTP block's keys and chain links at them too. Open: the MTP
        block reads an image cell as the pad token's embedding (llama.cpp gives the draft context
        the image's rows), DFlash takes cells as positions; drafts after an image may be accepted
        less often, the output is the main model's.
  - [x] 6. The splice in `qwen4exp`: the same copy on the part that embeds, Flash-Next's own step
        record at the rows' rotary positions; the n-gram feed hashes the pad token, which the
        model checks is `ple.image_token_id`. Run on IQ2_XS on both local cards (right answers
        on the three routes, a resend from the cache); not compared with a reference engine.
  - [x] 7. Routes: `image_url` (data URL or base64 alone), Anthropic `image` (base64, in user turns
        and tool results), Responses `input_image` (data URL); remote URLs and `file_id` refused
        with a 400 by name; JPEG, PNG, GIF, WebP, BMP; 16384 pixels a side and 64 megapixels at
        most. `--mmproj`, `--mmproj-weights ram|vram` (`ram` by default), `--image-max-tokens`
        (8 to 4096, 4096 by default); `DeviceTower::pinned_bytes` in the model's `pinned_ram`.
        Open: fetching remote URLs (backlog). Images in a system message are refused, as the
        models' chat template refuses them.
  - [x] 8. Caching: image cells named by a digest of the resized pixels and the projector
        (`cell_name`), 64-bit names through the prefix index, the RAM level and the store on SSD
        (text names unchanged, so text-only pages stay valid). Open: the store of image rows is
        not kept on SSD (after a restart a resumed conversation runs the tower once per image).
  - [x] Pages a finished request leaves in the device pool (the prompt cache on the devices): the
        owner's answer (2026-11-22): their far K and V rank as prompt cache, the lowest user of
        `--ram`; when memory gets tight they give way first, before the tier of experts, to the
        level on SSD with `--cache` or nowhere without it. A running sequence's far K and V stay
        required. Built (`Cache::trim` takes such pages off the devices while the prompt cache
        holds more than its allowance, `PinnedRam::set_context_cached`; a far piece is unlocked
        once no page in use lies in it). Tests: `ares-runtime --test cache
        far_pages_of_a_finished_request_give_way_when_the_tier_of_experts_wants_the_memory`, the
        books' `far_k_and_v_of_finished_requests_counts_as_prompt_cache_and_not_against_the_tier`.
  - [x] The tokens right after a long prompt decode slower than before (test machine, 65 100
        tokens: 11.6 to 12.0 tokens/s against 14.2 to 15.5 in three of four runs of the build
        before) while the same context decodes faster once the tiers are warm. Not the overflow
        (control run on the two development cards, 2026-11-22, load 9 to 46, the same build
        with `--context 66560`, all K and V in VRAM, against the overflow, B, C, B, C): right
        after the 64k prompt B decoded 12.5 and 8.8 tokens/s, C 2.9 and 5.6; again from the
        cache B 16.5 and 6.2, C 5.2 and 11.1. C lent steps of 4096 rows, B of 2048: a prompt
        read alone lends expert slots for its steps, and they come back empty, so the tokens
        right after it miss what the slots held; the bigger the loan, the slower they are.
        Found with it: the block scores of every row of a wide step took 256 KiB a row at the
        model's own 262 144 cells, which made B's loans smaller per row than they need be;
        rows are scored 256 at a time since (`SCORE_ROWS`). Open (for the prompt chunks): the
        slots a loan gave back filled with the hottest experts of RAM before the first token
        after the prompt; the same no-loan control (`-b 512`) on a quiet machine; the A/B of the
        build with the rows scored 256 at a time (scripts in `/var/tmp/ares-kv/scripts`).
  - [x] Two cards (a split by layers: a far pool per part; both cards on every layer: the mixer's):
        `qwen4exp_runtime k_and_v_beyond_vram_change_no_token` passes on the RTX 4070 + RTX 3060
        (2026-11-22). Not run with the real file on two cards.

### From "Optimising (a store of levers; taken by the owner's reasons, not as a phase)"

- [x] The prefill path on a card whose tensor cores run f32 accumulation at full rate (the test
      machine's RTX 4060 Ti with gpu-unlock; `probe_full_rate` detects it, 2026-11-22): measured
      per type at load (`Kernels::measure_prefill_path`, the exact path within 5 % of the
      fastest), by the qwen35 chooser and Flash-Next's wide steps. What bounds the f16 path there is
      the load unit (the file layout's loads touch 16 lines per instruction, and the f16 path's
      ldmatrix, tile stores and activations double the int8 path's traffic there), not the
      decoding; the 2-byte aligned words are now read as 16-byte pieces in the large shape
      (Q8_0, Q6_K, Q3_K f16 +10 to 14 %) and the activation tiles grouped by L2 (512 rows, the
      prompt step: +12 to 19 % on the f16 path). Q6_K, Q3_K and Q5_K now take the exact path,
      IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS and Q8_0 int8. Qwen3.8-27B UD-Q4_K_S, 2048 tokens: 1000 ->
      1025 tokens/s. `docs/dev/architecture/kernels.md`, "Prefill weight kernels".
- [x] Capped cards measure their prefill path per type too (the owner, 2026-11-22; built in
      `devices/load-choice`): on the 27B the prompt reads as fast as before (1342 to 1358
      tokens/s against 1346 to 1361, three alternating pairs on the two development cards), the
      same text, the load 1.0 to 1.1 s longer per device; Q5_K on the RTX 4070 takes F16Flush
      where the kept rule took int8.

## 2026-11-17

### From "The order of 2026-11-17 (after the gap study; items 1 to 7 done, the rest is a store)"

5. [x] Flash-Next's draft vocabulary (step 6 of the shared layer).

### From "Shared model layer (the owner's rule, 2026-11-16)"

- [x] 5. Token rows (embedding on the host, in the mapped file or on the card; the image rows):
      `blocks/rows.rs`, 2026-11-17. The 27B moved onto it bit for bit; Flash-Next stages its
      IQ4_XS rows from the mapped file and the card decodes them, bit for bit as the host did.
      Open: no image tower uses the second row source yet; DFlash's newest token and mask rows
      are still decoded on the host for every pass (both models).
- [x] 6. One driver for MTP (`blocks/mtp.rs`, `blocks/head.rs`, 2026-11-17), with the owner's rules (2026-11-16/17): as little per model as can
      be (the model gives its layer and its tap); one set of names for the drafter kinds,
      `ngram`, `mtp`, `dflash`, in types, files, metrics, options and docs; MTP from a
      separate file for every model (`--drafters mtp` for the block in the model file,
      `--drafters <file>` for a separate MTP or DFlash file); the draft vocabulary as one
      head view for every drafter that projects to the vocabulary, on every model. Brings
      Flash-Next's draft vocabulary and picks on the card, the 27B's drafts under a layer split.

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Speculation that pays (output unchanged)"

- [x] Draft vocabulary (German, English, code) for the MTP block and the DFlash drafter: the
      head view of every drafter (`--draft-vocab`, 2026-11-17; shared layer step 6).
- [x] `--drafters mtp` for a model file that carries its block (2026-11-17).

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Experts from the SSD"

- [x] (fixed 2026-11-17, `moe/q4-load`) Defect: Flash-Next's float-path tensors (stream mixers `hc_attn`/`hc_ffn` down and up,
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
- [x] (fixed 2026-11-17, `moe/q4-load`) Defect: the UD-Q4_K_XL file (as an F32 copy of the float path, see above) does not load
      with both cards on every layer, the default: the card that holds the streams has room
      for 4.94 GiB of slots, and the store asks for 88 slots of about 3.6 MiB per layer of the
      experts it owns (eight rows of ten picks and eight free), 15 GiB; the error then says to
      name a larger `--ram`, which is not the cause. `--split layer` loads (7.61 and 7.22 GiB
      of slots, 21 and 20 % of the experts each card runs). Owed: a floor of slots per layer
      and device that follows what a step can pick there (with the experts split between two
      cards, a device's share of a step's picks), and the right error.

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Parity with the 27B (the owner's rule: what one model can, the other must)"

- [x] The MTP block read from the model file (`--drafters mtp`) and a draft vocabulary
      (2026-11-17).

### From "Qwen3.8 27B (`qwen35`)"

- [x] Drafts under a layer split (2026-11-17, step 6 of the shared layer: the MTP block on the
      last device, with the head).

### From "Measurements owed on a quiet machine"

- [x] Flash-Next IQ2_XS decode step before and after `moe/q4-load` (2026-11-17, four alternating
      pairs at load 5 to 10): medians 23.3 to 25.2 ms before, 21.8 to 24.7 after, fastest 18.4 both.

## 2026-11-16

### From "Shared model layer (the owner's rule, 2026-11-16)"

- [x] 0. A `blocks` module; the 27B's K/V code moved into it as it is.
- [x] 1. The K/V group as a block, the 27B onto it (bit for bit, same launches).
- [x] 2. The recurrent (delta-rule) block, both models onto it (bit for bit).
- [x] 3. Flash-Next onto the K/V group: K and V in `q7` to `q4`, windows as if alone.
- [x] 4. Slot state and checkpoints as one contract: Flash-Next's prefix cache with RAM and SSD
      (2026-11-16, `blocks/slots.rs`; the 27B moved onto it bit for bit).
- [x] 7. Drafters as kinds of the runtime (ngram, MTP block, DFlash): `Model::drafters`, the
      drafter's id in word 6 of a segment, a price curve, acceptance and confidence per drafter.
- [x] 8. DFlash2 chains on both models: each model gives only its tap (the 27B the residual `x`,
      Flash-Next the mean of its four streams), its embedding rows and its draft head; the loader
      reads the drafter's shape from the file and refuses DSpark, DFlash 1 and the other variants
      by name. Open: Flash-Next under a split by layers (taps would cross devices), the ring in
      the prefix cache's checkpoints, sampled drafts (the walk is greedy).

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Experts from the SSD"

- [x] Usage profile: the server keeps its expert heat in a file, loads it at start in place
      of the canned warm-up text, keeps it current, and fills empty slots from it when idle
      (branch `moe/expert-profile`). Measured 2026-11-16: the first requests after a start from
      a profile of use miss as few experts as a warm server's (2.7 to 4.5 reads from the file
      per step against 6 to 15.6 after the warm-up); the fill takes 12.6 to 13.1 s on one card
      (warm-up 25 to 32 s) but 20 to 24 s on two (warm-up 21.6 to 27.8 s, no longer 41 to 44):
      both read 20 to 25 GiB and are bound by the drive.

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Cache, tests, kernels"

- [x] K and V in `q7` to `q4` for this model (2026-11-16: the shared K/V group, see
      "Parity with the 27B" below).
- [x] Prefix cache for this model, with the RAM and SSD levels (2026-11-16, on the shared slot
      block; see "Parity with the 27B").

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Parity with the 27B (the owner's rule: what one model can, the other must)"

- [x] K and V in `q7` to `q4`, separately for K and V; the MTP block's K and V in `q4`
      (2026-11-16, the shared K/V group). Open: a row of a draft window past the dense limit
      that reaches across the point where a tile leaves the newest cells reads that tile exact,
      not converted as the row alone would (rows within the limit read it as alone); the
      share of drafts accepted with the block's `q4` was not measured on this model.
- [x] The prefix cache with its RAM and SSD levels (2026-11-16). Open: the MTP block's last
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

### From "Prompt cache (every model; RAM first, then SSD, as in the owner's fork)"

- [x] The same for Flash-Next (2026-11-16: its prefix cache on the shared slot block; no
      runtime change was needed, also with both cards on every layer).

## Undated

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Speculation that pays (output unchanged)"

- [x] Window prices that follow the tiers: a size's price is the mean of its last eight
      steps without the largest, and a size that has not run for 256 steps is tried again
      at the price of the nearest smaller size that runs (`ares_sched::Curve`).
- [x] All 512 experts of the MTP block resident in VRAM (675 MiB, a pool of their own
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

### From "Qwen3.8 Flash-Next (`qwen4exp`) / Experts from the SSD"

- [x] The reader is not the limit: on this drive (btrfs zstd, kyber) it reaches 2.3-2.7 GB/s
      for random experts at 2 to 64 in flight with any queue depth from 4 to 64 and whole
      demand reads (pieces of 256 or 128 KiB are slower); file order does not matter.

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
- [x] Pruning as a fourth level next to VRAM, RAM and SSD (branch `moe/pruning`):
      `--expert-prune skip|router|similar:RANK`, off by default, on the device right after
      the router's top-k; the tool `qwen4exp_prune` measures it. Skip is the rule to use
      (least KL for the reads it saves at every rank on both files); a stand-in by
      similarity does not pay because the experts' outputs are nearly orthogonal (the most
      similar one carries well under 1 % of a missed expert's output). IQ2_XS, 8 GiB:
      `skip:4` 0.7 reads per token instead of 4.5, 45 tokens/s instead of 31, KL 0.011;
      Q4_K_XL from the SSD: `skip:8` 5.0 tokens/s instead of 3.5 at KL 0.026, `skip:1` 17.5
      at KL 0.50. Tables in `experiments/flash-next.md` ("Pruning, measured (2026-11-16)").

### From "Drafters (every model)"

- [x] DFlash2 as a kind of drafter the runtime knows, chains, for both models (steps 7, 8):
      `ares_model::dflash`, `--drafters <file>`. The 27B's file and Flash-Next's (trained on the
      mean of its four hyper-connection streams, which llama.cpp cannot drive) both draft.

### From "Prompt cache (every model; RAM first, then SSD, as in the owner's fork)"

- [x] RAM level for the K/V state that leaves VRAM; write to SSD when the RAM level evicts
      (full or shrinking), at unload and at a clean stop (branch `cache/ram-level`).

### From "Server and API"

- [x] API keys (X01); configuration file and environment variables (M08); CORS (X04);
      Host and Origin checks (X05).
- [x] Tokenize and detokenize (A09); model properties (A16); template override (F07);
      continuing the last assistant message (F09).
- [x] Reasoning tokens in the usage (K07); `stop_reason: stop_sequence` and the stop string
      that ended a reply.
- [x] Latency histograms (O01), logs (O03, O04), a bounded queue that says when to retry
      (S04), timeouts (S23).
