# What is still open

Updated 2026-11-22, `main` at the prefill path measured per type on full-rate cards (after the
shared layer's steps 5 and 6: one MTP driver, drafter kinds `ngram`, `mtp`, `dflash`; nothing
pushed). What is there is listed in
[`FEATURES.md`](FEATURES.md).
This file is brought up to date at every merge into `main`. It lists work, defects and
measurements that are owed; what a feature must do in detail is in
[`reference/feature-checklist.md`](reference/feature-checklist.md) (223 rows: 70 done,
42 partly, 88 not started, 23 outside the frame; its rows K01, K02, K06 and P14 are out
of date: reasoning is built, and `q7` to `q4` run in the 27B).

GPU work runs on the test machine `admin@test-machine` (RTX 4060 Ti 16 GB, gpu-unlock) from
2026-11-22 on; the two local cards are production and are used only when a test needs two
cards, with the owner told first.

Order of work the owner has set: CUDA complete for both models first, then optimising (better
kernels, better prefetching: measured per phase on a quiet machine against the fork and Strata),
Vulkan parity last.

## Next, in this order (the owner, 2026-11-17, after the gap study)
For "CUDA complete", the order in which the items below are taken up; each item stays in its
section further down.
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
5. [x] Flash-Next's draft vocabulary (step 6 of the shared layer).
6. K/V of a running sequence beyond VRAM for Flash-Next at long contexts (whole features). The
   owner's rule for the context (2026-11-22): without `--context` the model's own maximum (262144
   for Qwen3.8) on both models; what does not fit in VRAM overflows into RAM. Until the overflow
   exists, the most that fits, with a log line. On the 27B the attention heads move to the second
   card as far as the context needs (built on `devices/roles`).
7. Priorities of requests (server).
8. Then the rest of CUDA complete: image input pieces 4 to 8, gate and up in two types, the
   roles of the cards, the open defects, the server items.
Later than their place suggests: the classic draft model (VRAM; better with an aux card), Turing,
Jev mode, stored responses, TLS. DFlash2 trees (steps 9 and 10) with the experts of a tree's
tokens read ahead ("Speculation that pays"): the owner, 2026-11-22, expects much of them for
decode; measured first (how many picks a token table and route memory predict, how much longer a
tree is accepted than a chain), then built; started when the pieces in work leave room, one
piece at a time.

## In work now
Merged 2026-11-22 after the test machine's one card passed every GPU suite (`fix/mtp-and-slots`,
`moe/prompt-chunks`, `cache/checkpoints`, `devices/roles`); what needs two cards is owed below
("Owed on two cards").

| Piece | State |
| --- | --- |
| Flash-Next decodes UD-Q3_K_XL at least as fast as Strata on the test machine (the owner, 2026-11-22): CPU kernels for every expert type, then the next lever by measurement | handed back, being verified (`flash/cpu-experts`): with drafts 26 -> 34 to 36 tokens/s at 40G, ahead of Strata 0.1.60 (29.9); Strata 0.1.65 not measured yet |
| Flash-Next prompts at least as fast as Strata (176 / 250 / 281 against 198 / 787 / 1192 tokens/s at 2k / 8k / 32k), and a nearly cached turn read as if uncached (31 s for 6832 tokens with 6784 cached; Strata 0.4 s) | agent working (`flash/prompt`) |
| Flash-Next UD-Q4_K_XL (104 GiB, larger than RAM) streamed from the SSD against Strata, then Ares's reading path | agent working on the test machine (`flash/ssd-stream`) |
| 6: K/V beyond VRAM into RAM (Flash-Next) | handed back, continuing: finished requests' far pages as prompt cache (the owner), the control run for decode after a long prompt (`kv/overflow`) |
| 7 and the server part of 8: priorities of requests, system messages and efforts on the chat and Responses routes, Messages items, thinking budget, special tokens, truncation, small ones | handed back, being verified (`server/priorities`) |
| Image input, pieces 4 to 8 (the owner, 2026-11-22: next, for production): the tower as a conductor job, the splices in both models, the routes, caching | handed back; with the CPU kernels and the priorities in one line of commits (`server/priorities`), being verified on the local cards |
| Flash-Next's second card measured at load (experts only or a dense share), capped cards measuring their prefill path per type, the MTP test that fails split by layers | agent working on the local cards (`devices/load-choice`) |
| DFlash2 trees and the experts of a tree's tokens read ahead, phase 1 (the owner, 2026-11-22: pulled forward if it brings a lot): acceptance of trees against chains, the extra experts a tree costs, how well draft tokens predict their experts, the estimate | agent measuring (`flash/trees-study`) |

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
- [x] 7. Drafters as kinds of the runtime (ngram, MTP block, DFlash): `Model::drafters`, the
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
- [ ] 9. Trees in the blocks, verified once for every drafter and both models. Later (the owner, 2026-11-17).
- [ ] 10. Tree fill from the DFlash lattice, MTP links mixed into a tree. Later.

## Qwen3.8 Flash-Next (`qwen4exp`)

### Found against Strata on the test machine (2026-11-22; RTX 4060 Ti 16 GB, 62 GiB RAM, UD-Q3_K_XL)
Strata moves fast (0.2.19 / engine 0.1.65 on 2026-11-22, 0.2.20 / 0.1.66 on 2026-11-23): 0.1.66
names fused expert kernels for all 13 Unsloth type pairs (MoE decode 1.04 to 1.67 times), INT8
tensor-core dense formats (DFlash2 verify and chunked prefill 1.8 to 3.3 times), faster AVX2 CPU
kernels for cold experts (Q8_0 +55 %, IQ4_XS +33 %, Q4_K +27 %), host K/V offload from 32k on
mid-size cards, a GPU+SSD-only mode, and a fix for NaN outputs on prompts of 24k to 45k. The test
machine's Strata is updated to the newest engine before every comparison; numbers against older
engines below are history.
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
- [ ] Ares is behind: decode with drafts 40-55 % of Strata's (15.9-31.0 against 40.0-43.0
      tokens/s), without 43-51 %; prompts 4 to 10 times slower (Strata 398 / 829 / 1257 tokens/s
      at 2k / 8k / 32k, Ares 94-121); the first token of a short repeated prompt 3 to 15 times
      slower (no checkpoint at the prompt's end: item 2 of the order). Ahead: load 3 times faster
      (35-50 s against 96-149), 12 GiB of RAM left free against Strata's 2-3, a long conversation
      turn asked again 2.7-2.9 s against 9.3 s (Strata keeps only the newest path).
- [x] Why, from the logs: GPU hit rate of the experts 37-74 % against Strata's 62-83 % with
      about the same slots: Ares fills only 1267-1862 of about 4000 slots at the start and then
      moves 63-171 experts a token over the card's x4 link (33-102 ms a token waited); Strata
      fills every slot from a ranked profile, changes them slowly, and computes every expert
      missing from the GPU on the CPU from page-locked RAM (Ares computes 14-42 % there and
      uploads the rest). Strata also skips missed experts of rank 7 and lower by default (not
      like for like; Ares' `--expert-prune skip:8` is that). Built 2026-11-22 (see "Members,
      loans and promotions" in ARCHITECTURE.md): the fill makes VRAM the hottest of every layer
      (at the start, after the warm-up, when idle); members change only by the rule of a
      promotion, at most what the link does in a step; a miss goes to the CPU or the link by the
      measured costs, and an upload that is not worth a member's place goes up on loan. Also
      found: the profile counted every row of a prompt, so it ranked what long prompts used (a
      decode's hit rate 16-23 % when VRAM was filled from the comparison's profile); a wide step
      now counts an expert once (profile format 2). Measured against the build of before, both
      without a profile (ARCHITECTURE.md, after "Members, loans and promotions"): with the MTP
      block 20.0-37.6 tokens/s against 17.2-34.8 at 56G (mean +11 %), without it the same
      (16.4-21.9), at 24G +13 to 14 %; VRAM hit 59-72 % in both builds (Strata 62-83 %), 36-53
      uploads a token against 51-58, the CPU 18-26 % of the picks. Started from a profile of use
      (format 2) VRAM is full of the profile's hottest (3 473 brought by the fill) and the hit
      rate is the same, 59 % and 70 %: on this card the size of VRAM, not its filling, bounds it.
      Still behind Strata (40.0/43.0 with drafts, 27.2/28.6 without): the misses cost 26-56 ms a
      token, see the next item.
- [ ] The CPU computes the types of UD-Q3_K_XL slowly: IQ3_XXS and IQ4_XS as gate and up, Q5_0
      and IQ4_NL as down have no kernel of their own and are decoded row by row
      (`cpu/kernels.rs`): a job takes 170-410 us per expert and row on the test machine where an
      upload takes 410-490 us, so the cost model sends a third to a half of the misses over the
      link. Strata computes these types with ggml's AVX2 kernels at several times the rate.
      Kernels for them are the next lever of decode on such a machine.
- [ ] Transit adoption (`transit_adopt`, the prompt chunks' experts taking VRAM slots after a
      wide step) asks only "hotter than the coldest member" (with the swap margin when the member
      has to be copied down), not the rule of a promotion; left as it was for the work on prompt
      chunks that borrow expert slots.
- [ ] The default context on that card is 23,552 tokens ("a thirty-second of the expert room"):
      too small for a 32k prompt; the rule needs a floor or the owner's choice.

### Defects seen in the merge check of 2026-11-17 (intermittent, slower but not wrong)
- [ ] `ares-server --test real` `flash_next_drafts_with_its_mtp_block` sometimes verifies no
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
The MTP block is in `main` (`--drafters <file>`): exact. Measured 2026-11-16 on two cards
with `--ram 8G` (the runs are in "What made the MTP block of Flash-Next pay" in
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
- [x] Draft vocabulary (German, English, code) for the MTP block and the DFlash drafter: the
      head view of every drafter (`--draft-vocab`, 2026-11-17; shared layer step 6).
- [ ] A window's price by what its rows read: on one card a round with drafts reads 100 to
      260 experts from the file against 45 to 100 for a row, and what those reads push out of
      the tiers and hold up for the next steps is charged to no step, so the conductor keeps
      choosing windows that cost more than they bring. A price per row by the tier its
      experts come from (the study's `dense(rows) + chain(links) + misses x cost per tier`),
      or the block not resident where VRAM holds a third of the experts, are the leads.
- [x] `--drafters mtp` for a model file that carries its block (2026-11-17).
- [ ] With the shared MTP driver (links of 1.4 ms, 0.9 with the draft vocabulary, where they took
      1.7 to 2.0 with the host's wait) Flash-Next answered at 39.0 tokens/s on the mean of twelve
      requests against 41.1 before (2026-11-17, alternating, loads 2 to 6; requests from 31 to
      59): the conductor verifies more of the cheaper drafts at a lower rate (the first link's
      54 to 58 % against 63 to 71 %). A window's price by what its rows read (below) is the lead.
- [ ] Drafts in the measuring tool (`qwen4exp_run` has none; the server and the runtime's
      tests measure them).

### Experts from the SSD
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
      in ARCHITECTURE.md. **Owner's decision:** keep the large lone step, bound it in time
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
- [ ] The MTP block's experts in the usage profile (they are filled in the order of their
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
      predicted picks ahead now, within a step.) TokenWing's form, read 2026-11-22
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
- [x] A request that arrives behind a lone long prompt: once it waits, steps of prompt are
      pieces of at most 500 ms by their measured price (`RuntimeConfig::piece_ms`, 2026-11-22).
      It still waits for the step in flight, which for a prompt read alone is a step of up to
      `--batch-size` rows on lent VRAM (see "Large prompt steps on lent VRAM"; measured
      there).

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
- [x] The MTP block read from the model file (`--drafters mtp`) and a draft vocabulary
      (2026-11-17).
- Not owed (the owner, 2026-11-16): replies that are bit for bit the same from run to run.
  They differ slightly when the CPU computes missed experts, and that is accepted.
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

## Qwen3.8 27B (`qwen35`)
- [ ] Seen 2026-11-17 (main 4bec4b9 as well): with `max_tokens` 40 the first greedy reply to
      real.rs's German question after a start (IQ2_XXS, one card, fast kernels) and the same
      request again, its prefix from the cache, are different tokens (hashes 4d9a1bef2efe92d9
      and c1f7796b6753a9e5), with drafts and without; at 160 tokens all agree. Not looked into.
- [x] Drafts under a layer split (2026-11-17, step 6 of the shared layer: the MTP block on the
      last device, with the head).
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

## Shared model layer (every model; the study `shared-model-layer.md`, steps 0 to 8 done)
The K/V group, the recurrent block and the slot arena with its checkpoints are shared
(`crates/model/src/blocks`); the rule for what
becomes a block is in `ARCHITECTURE.md` ("The shared model layer"). Left, in the study's order:
- [x] Slot arena and checkpoints as a block (extents from the blocks, save, restore, transfer,
      page copy over all groups); then the prefix cache of Flash-Next on it (2026-11-16).
- [x] Token rows (table at home on the host, in the mapped file or on the device; kept rows of
      a head view; a second row source for images); Flash-Next's embedding from staged rows.
- [x] One MTP driver for both models (chain on the device, picks by the sampler); the
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
- [ ] Blackwell (the owner, 2026-11-22: an RTX 5060 Ti is ordered for the test machine, next to
      its RTX 4060 Ti; an MCIO Gen5 splitter makes the x16 slot two x8: the 5060 Ti at Gen5 x8, the
      4060 Ti at Gen4 x8, twice the link it has today, so the link-bound numbers measured there at
      x4 must be taken again). Two cards of two generations there: placements across Ada and
      Blackwell, peer access only if that machine's driver allows it. Both in place in about a
      month; until then the owner can swap the 5060 Ti in for the 4060 Ti for a while to test. Ares
      compiles sm_120 code (`crates/cuda/build.rs`) but has never run it. When the card is in: the
      whole GPU suite on it; then Blackwell's own paths where they pay, measured: FP4 tensor cores
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
      Measured 2026-11-22 (`ARCHITECTURE.md`, "What the second card is for"): for the 27B the
      second card is worth most as room for K and V (the attention heads by shares of their own,
      as the fork's `--attn-split`: 2.1 times the context, 3 % slower decode at 1000 cells and
      9 % at 32 000). Built as how the plan works, with no option (the owner, 2026-11-22):
      `AttentionShares::ByContext`, heads with the weights as far as the wanted context
      (`--context`, or the model's own) fits, moved only as far as it needs. The rest of the
      roles item (what the aux card is called, drafters on it) waits for the owner.
- [ ] Flash-Next's second card (measured 2026-11-22, `ARCHITECTURE.md`, "What the second card is
      for"): on every layer it is worth most (twice the VRAM tier); as an expert tier only (no
      dense share, `core_share` 1) it decodes as fast and reads warm prompts 27 % faster than the
      share chosen today from read times. Proposed: the plan measures at load a wide and a decode
      step with and without the dense share and keeps the faster; goes to the owner first (it is
      the card-roles design), not built.
- [ ] Flash-Next served with the model's own context (262 144 cells, the default since
      2026-11-22) gives up a fifth of the VRAM slots (3.56 GiB of K/V and keys, all on the mixing
      card): decode 11 to 22 % slower, warm prompts half as fast. K/V beyond VRAM in RAM (below)
      would let the slots stay.
- [ ] Drafters on the second card: not built. A chain that runs on the slower card is slower by
      what that card is slower, and nothing runs beside it in a step that only drafts; it pays
      only with a pipeline that drafts the next window while the main card verifies this one
      (Strata's `--pipeline-windows`).
- [x] DFlash2 as a kind of drafter the runtime knows, chains, for both models (steps 7, 8):
      `ares_model::dflash`, `--drafters <file>`. The 27B's file and Flash-Next's (trained on the
      mean of its four hyper-connection streams, which llama.cpp cannot drive) both draft.
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
- [x] A root checkpoint at the end of the system prompt and of the tool list, a checkpoint at
      every message boundary and at fixed distances inside a long prompt (the owner, 2026-11-18),
      for both models (2026-11-22, branch `cache/checkpoints`; "Sequence state" in
      `ARCHITECTURE.md`). Open: a root that comes back from the drive is a plain checkpoint
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
- Not wanted (the owner, 2026-11-22): saving and loading a conversation's state by name
  (llama.cpp's `slot-save-path`); the prompt cache covers it.
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
      `/v1/messages` took 18: the history is rendered differently somewhere. (2026-11-22: on
      Flash-Next and on the 27B the two routes take the same, 83 of 99 and 89 of 105 in
      `real_responses` and `real_messages`; the 0.8B was not run again.)

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
      are refused there.
- [ ] Chat and Responses routes: a system or developer message inside a conversation reaches
      the template, and the templates of Qwen3.8 refuse it (the Messages route gives it to the
      template as a user turn); `reasoning_effort: high` reaches Flash-Next's template, which
      knows `xhigh`, `medium` and `low` and refuses it (the Messages route maps Anthropic's
      `high`, its default, to the template's default).
- [ ] Messages: `thinking.display: omitted` gets the thinking whole (to omit it and still have
      it in the next prompt, the signature would have to carry it); a response does not list
      `context_management.applied_edits`; `clear_tool_uses_20250919` is refused.
- [ ] Router: load and unload times in the docs; "no room" for Flash-Next is recognised by
      message text; a plan cannot be asked without loading; two elastic models on real devices.
- [ ] A request with tools or an effort for an unknown model gets a 400 where a 404 belongs.
- [x] Latency histograms (O01), logs (O03, O04), a bounded queue that says when to retry
      (S04), timeouts (S23).
- [ ] Priorities (S06), a watchdog for a silent engine (S23), a VRAM reserve for other
      programs (V04), container image and service unit (X09), key roles and a reload of the
      key file (X02), TLS (X03).
- [ ] `/props` of a model that is not loaded (today: 503 `model_not_loaded`, as `/metrics`).

### Claude Code against Ares (2026-11-22, Claude Code 2.1.301, print mode)
Claude Code in a sandbox of its own (own home, a throwaway project, `ANTHROPIC_BASE_URL` on
Ares), three small tasks: read a file and judge a function, fix it with Edit and run the tests
with Bash. Scripts and logs in `/var/tmp/ares-flashapi` (`notes.md`).
- [x] Claude Code could not talk to Ares at all: `context_management`, `thinking.display` and
      system turns between the others were refused, and effort `high` would have reached
      Flash-Next's template, which refuses it. Served since (see "The Messages API" in
      `ARCHITECTURE.md`).
- [x] A stream's response started only with the first token: Claude Code gave up after six
      minutes while Flash-Next computed its first prompt and sent it again. It now starts
      after 10 s without it (`Config::stream_wait`).
- Flash-Next (two cards split by layers, the driver with peer access being missing; `--ram
  8G`, its MTP block, machine load 8 to 27 from other builds):
  the tasks are done right, in 3 and 4 turns. Claude Code's first prompt is 16.3 thousand
  tokens (the 20 tools of its default set; 3.5 thousand with `--tools Read,Edit,Bash`), read at
  25 to 61 tokens/s: 3.5 to 4.5 minutes to the first token. A turn that continues the cached
  conversation has its first token in 3 to 9 s; replies come at 5 to 9 tokens/s with 50 to 90 %
  of the drafts accepted.
- The 27B (two cards, tensor split, its MTP block, peer access): the same tasks in 2 and 4
  turns, 37 s and 22 s. The first prompt at 1052 tokens/s (15.5 s), a continued turn's first
  token in 0.2 to 0.5 s, replies at 65 to 120 tokens/s.
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
- [ ] Turing. Later.
- [ ] K/V of a running sequence beyond VRAM (P16; an exception, low priority). The owner,
      2026-11-22: without `--context` a model gets its own context always, and what of it does not
      fit in VRAM overflows into RAM. The plan says what was wanted and what of it is in VRAM
      (`plan::ContextPlan { wanted, in_vram }`); until the overflow is built the context is what
      VRAM holds, with a log line.

## Weight types (`moe/q4-load`, 2026-11-17)
Every tensor of both models takes every weight type the shape allows (the audit of every place
that refused a type is in the branch's notes; ARCHITECTURE.md "Files in any type" and "A layer
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

## Optimising (after CUDA is complete, before Vulkan; the owner, 2026-11-16)
Measured per phase on a quiet machine against the fork and Strata, the largest first. Known levers:
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
      1025 tokens/s. `docs/ARCHITECTURE.md`, "Prefill weight kernels".
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
- [ ] Capped cards measure their prefill path per type too (the owner, 2026-11-22): one
      condition in `kernels/paths.rs`, verified on capped cards.
- [ ] Attention over long contexts: llama.cpp PR 32407 (merged 2026-11-17) rotates the order in
      which blocks read K/V tiles so that blocks sharing K/V hit L2 together: prompts of the 27B
      on an RTX 5090 at 131072 cells 1290 -> 1822 tokens/s (1.41x), 1.25x at 65536, 1.07x at
      32768, nothing below. Ares' attention for prompts and wide windows at long contexts.
- [ ] VRAM on loan to the expert tier: K/V pages a conversation does not need yet (Strata's
      `--kv-grow`), the MTP block's 675 MiB while a long prompt runs (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
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
- [ ] Long prompts layer by layer with read-ahead; the profile's fill in large ordered reads.
- [ ] Few-row kernels for windows and trees; attention for wide windows; the F32 tile kernel
      (2.5 times BF16); NVFP4 and IQ2_XXS gaps; the image tower's row kernels fused.
- [ ] The sampler's exchange on the second card; a layout choice that is stable between loads;
      the MTP link (1.49 against 1.44 ms possible); DFlash's drafts-only step (4 to 5 ms).

## Vulkan (last)
- [ ] A backend-free crate and the `ares-gpu` facade; expert and Flash-Next kernels; both
      models on Vulkan in every placement; attention for wide windows (2 to 6.5 times slower
      than CUDA today); the delta rule's f16 state rounded per row; the sampler's `-0.0`.

## Measurements owed on a quiet machine
- [ ] On the test machine `admin@test-machine` (RTX 4060 Ti 16 GB with gpu-unlock, 62 GB RAM, a
      Samsung 990, Strata 0.2.15 installed, good internet for downloads): Flash-Next against Strata
      with the same file (UD-Q3_K_XL there), a large RAM tier, one card of another class.
- [ ] Real models with 1, 2 and 4 concurrent clients against the fork and Strata (Strata saw four
      requests on a 12 GB card with experts outside VRAM 11 to 24 % slower in sum: parallel
      requests buy waiting time there, throughput on the 27B) (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
- [x] Flash-Next IQ2_XS decode step before and after `moe/q4-load` (2026-11-17, four alternating
      pairs at load 5 to 10): medians 23.3 to 25.2 ms before, 21.8 to 24.7 after, fastest 18.4 both.
- [ ] Flash-Next with the owner's real limit (a RAM tier of 24 to 26 GiB), one card and two.
- [ ] The SSD cache's times to the first token (taken under load so far).
- [ ] Flash-Next with its MTP block once speculation pays; target above 56 tokens/s.
- [ ] Before a release: both models against the fork, Strata and TokenWing (Flash-Next), speed,
      VRAM, quality.

## Owed on two cards (the owner's cards are production; run when the owner frees them)
Run on 2026-11-22 evening (main c5d27e9, RTX 4070 + RTX 3060, peer access, load about 30 from
the agents' builds):
- [x] The two-device suites: `real_cache` 4 of 4, `real_devices`, `qwen35_device_runtime` (the
      attention-split placements) 4 of 4, `qwen4exp_device` with `ARES_TEST_SPREAD=tensor` 23 of
      23 and with `ARES_TEST_CORE_SHARE=1 ARES_TEST_SHARED=mixer` 23 of 23; `real` 21 of 22 (the
      known intermittent `flash_next_drafts_with_its_mtp_block` that verifies no draft at all).
- [x] The MTP block's first links with both cards on every layer: a server session (IQ2_XS,
      `--ram 8G`, MTP) accepted 626 of 1078 drafts, decode 17 to 52 tokens/s; the poor first
      links the server agent saw are not seen again.
- [ ] `qwen4exp_runtime` `the_mtp_block_follows_wide_steps_that_start_at_any_cell`: passes with one
      device, fails in the split by layers on two: the warm-up finds 376 VRAM slots for 8 layers x
      48 experts (steps of 196 rows; the other tests take 8 or 64 rows and pass). Something keeps
      8 slots from the store in that placement at that step width.
- [ ] Rust 1.100 (stable on the test machine) crashes compiling `ares-model --test
      qwen35_failure` (an internal compiler error); 1.99 builds it, and the test passes there.
- [ ] `qwen4exp_runtime` with the default test threads hung for over 18 minutes once (two device
      threads spinning; the `moe/prompt-chunks` agent); with `--test-threads=1` it passes.

## Decided by the owner (2026-11-22)
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
  share or holds experts only, and keeps the faster; no option. To build; it needs the owner's
  two cards to verify ("Owed on two cards").
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
  kept from the two development cards (`FLUSH_AS_FAST`). To build with the item above; the
  measurement on capped cards needs the owner's cards.

## Questions to the owner
- None open.
