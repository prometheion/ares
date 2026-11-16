# What is still open

Updated 2026-11-16, `main` at elastic host memory and the image tower's CPU reference
(nothing pushed). What is there is listed in [`FEATURES.md`](FEATURES.md).
This file is brought up to date at every merge into `main`. It lists work, defects and
measurements that are owed; what a feature must do in detail is in
[`reference/feature-checklist.md`](reference/feature-checklist.md) (223 rows: 70 done,
42 partly, 88 not started, 23 outside the frame; its rows K01, K02, K06 and P14 are out
of date: reasoning is built, and `q7` to `q4` run in the 27B).

Order of work the owner has set: CUDA complete for both models first, Vulkan parity last.

## In work now

| Piece | State |
| --- | --- |
| Speculation that pays on Flash-Next: prices that follow the tiers (built), rows of unmade links leave the step, the draft block's experts resident, no window row waits for a file read; built in the conductor and the expert store, not in one model | agent working (branch `sched/live-prices`) |
| RAM level of the prompt cache, written to the SSD only when evicted, at unload and at a stop | agent working (`cache/ram-level`) |
| Shared model layer, steps 0 to 3 (see below); step 3 brings Flash-Next K and V in `q7` to `q4` | agent working (`model/shared-blocks`) |
| Image tower on the device, weights resident in VRAM or in RAM and uploaded per image (pieces 2 and 3) | agent working (`vision/tower-device`) |
| Test files: `Qwen3.8-Flash-Next-UD-Q4_K_XL` (104 GiB), then Flash-Next's DFlash2 drafter (3.1 GiB) | downloading to `/mnt/data/ares-models`, 6 MB/s |

## Shared model layer (the owner's rule, 2026-11-16)
Generic for Ares, not like llama.cpp: a few models, shared code where it can be, each model
still at its maximum through its own launch plan. A part becomes a shared block when it owns
state the runtime touches (pages, slot state, checkpoints, windows, commits, replays, trees)
or when it drafts; products, norms, kernel choice per row count, fused launches and exchanges
between cards stay with the model. Today the two models share almost nothing inside
`crates/model`, which is why every parity item below is owed twice. Design study:
`/var/tmp/ares-survey/shared-model-layer.md`. Each step lands green on its own.
- [ ] 0. A `blocks` module; the 27B's K/V code moved into it as it is.
- [ ] 1. The K/V group as a block, the 27B onto it (bit for bit, same launches).
- [ ] 2. The recurrent (delta-rule) block, both models onto it (bit for bit).
- [ ] 3. Flash-Next onto the K/V group: K and V in `q7` to `q4`, windows as if alone.
- [ ] 4. Slot state and checkpoints as one contract: Flash-Next's prefix cache with RAM and SSD.
- [ ] 5. Token rows (embedding on the host, in the mapped file or on the card; the image rows).
- [ ] 6. One driver for the draft block: Flash-Next's draft vocabulary and picks on the card,
      `--drafters mtp`; the 27B's drafts under a layer split. After the speculation piece.
- [ ] 7. Drafters as kinds of the runtime (lookup, draft block, DFlash). After the speculation piece.
- [ ] 8. DFlash on both models: each model gives only its tap (the 27B a copy of the residual,
      Flash-Next the mean of its four streams); the loader reads the drafter's shape from the file.
- [ ] 9. Trees in the blocks, verified once for every drafter and both models.
- [ ] 10. Tree fill from the DFlash lattice, MTP links mixed into a tree.

## Qwen3.8 Flash-Next (`qwen4exp`)

### Speculation that pays (output unchanged)
The draft block is in `main` (`--drafters <file>`): exact, and no faster yet. Measured
2026-11-15 on two cards with `--ram 8G` under load: 13 to 41 tokens/s without it and 10 to 41
with it; the runs are too noisy to tell the two apart.
- [ ] Window prices that follow the tiers: measured warm, re-measured, stale sizes tried
      again. Today they are measured once at load on cold tiers, so no window is chosen on
      two cards. The same defect lets a late step poison the 27B's price table under CPU load.
- [ ] All 512 experts of the draft block resident in VRAM (about 700 MiB): a link costs
      5 to 9 ms today, 1.5 ms when its experts are there.
- [ ] Chain cut per token by the draft's own probability.
- [ ] Speculation never waits for a file read: a window row whose expert is not cached is
      not accepted; the rows before it stay exact.
- [ ] Draft vocabulary (German, English, code) for the draft block.
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
- [ ] Prediction on the device: the routers of the next two layers applied to the current
      state in the router's own launch; read-ahead in chunks of 512 KiB, at most two while
      a demand read runs. (A first attempt with the prediction on the host was rejected:
      it changed the text and halved generation on two cards; branch `moe/read-ahead`
      is kept as a record.)
- [ ] Long prompts layer by layer over the whole prompt, interruptible, with read-ahead of
      the most used experts of the next two layers.
- [ ] A missed expert waits 2.3 to 3.2 ms for the file where the drive needs 1.4 ms: find
      the rest.
- [ ] Pruning as a fourth level next to VRAM, RAM and SSD: a picked expert that is only on
      the SSD is fetched only when the pick is strong; else it is skipped (tail skip) or
      replaced by the most similar cached expert (the owner's idea; no engine has it). A
      threshold as the quality dial, off by default; measured per rule and threshold on
      the IQ2_XS file and on the Q4_K_XL file (KL, agreement, reads per token, speed).
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
- [ ] K and V in `q7` to `q4` for this model (it takes `f16` and `q8`; the smaller formats
      need the pool of exact entries and the tile conversion the 27B has). At 13.9 MiB per
      1000 tokens in `q8`, a context of 131072 holds 1.8 GiB that could be expert slots.
- [ ] Prefix cache for this model (it has none), with the RAM and SSD levels.
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
- [ ] K and V in `q7` to `q4`, separately for K and V (see above); the draft block's K and V in `q4`.
- [ ] The prefix cache with its RAM and SSD levels (see above).
- [ ] The draft block read from the model file (`--drafters mtp`) and a draft vocabulary.
- Not owed (the owner, 2026-11-16): replies that are bit for bit the same from run to run.
  They differ slightly when the CPU computes missed experts, and that is accepted.
- [ ] Structured output, tool calls and the Messages and Responses routes have never run on
      the real Flash-Next file (the runtime and server code is shared; a smoke script for
      tools exists and was not run).

## Qwen3.8 27B (`qwen35`)
- [ ] Drafts under a layer split (Flash-Next drafts in every placement; the 27B not under
      `--split layer`): parity owed to the 27B, step 6 of the shared layer.
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

## Drafters (every model)
- [ ] DFlash2 with draft trees, as a kind of drafter the runtime knows, for every model that
      has a DFlash2 drafter file (dense and MoE alike), not as a feature of one model: the
      drafter model on the device, a tree instead of a chain (fill, node allocation), tree
      verification in the windows, the policy choosing among lookup, MTP and DFlash2 or all
      of them. Built on the shared layer (steps 7 to 10 above), for both models at once.
      Study: `/var/tmp/ares-dflash-study/REPORT.md`. Drafter files: the 27B's
      (`~/Services/models/dflash-test/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`; the 27B's DSpark file
      has the same architecture name with another shape: block of 7, other taps, no selector);
      Flash-Next's from `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF` (downloading; trained on the
      mean of the model's four hyper-connection streams, which llama.cpp cannot drive yet).

## Prompt cache (every model; RAM first, then SSD, as in the owner's fork)
- [ ] RAM level for the K/V state that leaves VRAM; write to SSD when the RAM level evicts
      (full or shrinking), at unload and at a clean stop. Today pages are written through
      as they are computed and only checkpoints lie in RAM.
- [ ] The same for Flash-Next.
- [ ] One size limit for all models on a directory; the transit buffers inside `--ram`;
      a test of the full drive; a layout that does not depend on the split.
- [ ] Cache policy: resume point at the prompt's end, memory of reply tokens, marks, warm pass.
- [ ] On `/v1/responses` a second turn took 1 of 35 prompt tokens from the cache where
      `/v1/messages` took 18: the history is rendered differently somewhere.

## Server and API
- [ ] API keys (X01); configuration file and environment variables (M08); CORS (X04);
      Host and Origin checks (X05).
- [ ] Tokenize and detokenize (A09); model properties (A16); template override (F07);
      continuing the last assistant message (F09); truncation and context shift (F18).
- [ ] Thinking budget (K03; Anthropic's `budget_tokens` is accepted and not enforced);
      loop guard (K05); reasoning tokens in the usage (K07).
- [ ] `stop_reason: stop_sequence` (the engine does not tell a stop string from the end token).
- [ ] Tool calls: a generic format for other templates; recovery of imperfect calls (T05);
      string constraints `pattern`, `minLength`, `maxLength`, `format`; forced tokens as
      drafts; a guard for `ignore_eos` with a complete grammar; the end-token rule for a
      schema with thinking; llama.cpp's `json.gbnf` costs 6.8 ms per mask.
- [ ] Responses: stored responses (A05), WebSocket (A06); the chat route's sampler fields
      are refused there. Messages: never run against Claude Code; no prefill.
- [ ] Router: load and unload times in the docs; "no room" for Flash-Next is recognised by
      message text; a plan cannot be asked without loading; two elastic models on real devices.
- [ ] A request with tools or an effort for an unknown model gets a 400 where a 404 belongs.
- [ ] Latency histograms (O01), logs (O03, O04), a bounded queue that says when to retry
      (S04), priorities (S06), timeouts (S23), a VRAM reserve for other programs (V04),
      container image and service unit (X09).

## Whole features not started
- [ ] Image input (F11; the owner's preset sets `mmproj`). Design: `/var/tmp/ares-survey/image-input-design.md`
      (2026-11-15). Both projector files are the same tower (`qwen3vl_merger`: patches of
      16 x 16, 27 blocks of width 1152, a 2 x 2 merger to the language model's width; 888
      and 866 MiB), with different weights. An image becomes (W/32) x (H/32) rows: 256 for
      512 x 512, 1024 for 1024 x 1024. Pieces, 0 and 1 done:
  - [x] 0. Reference numbers from the fork's `llama-mtmd-debug` on the CPU: rows of both
        projectors for eleven square patterns of 4 to 1024 rows, with the sum of every node of
        the graph. The program reads no image file, so no own image, no wide or tall grid and
        no resize went through the fork; the resize is checked against Pillow instead.
  - [x] 1. CPU reference of the tower (`ares-model::qwen3vl`): config, size rule, resize, patch
        rows, the pass. See "The image tower" in `ARCHITECTURE.md`.
  - [ ] 2. The tower on the device: dense attention for heads of 72, layer norm with bias,
        bias and GELU, the tower's rotary rotation, the position table's resize; `ffn_down`
        padded from 4304 to 4320 inputs. The residual reaches 5400 to 8400, so `ffn_down`,
        the residual add and the norms' statistics stay in `f32`; the device is held to the
        exact reference (`Arithmetic::Exact`). Reference dumps of the fork:
        `/var/tmp/ares-qwen3vl-ref` (with the notes of pieces 0 and 1).
  - [ ] 3. Both homes for the tower's weights, as the owner requires (llama's `mmproj` with and
        without `no-mmproj-offload`): resident in VRAM, or held in RAM and uploaded block
        by block for each image so that no VRAM is taken for good; a key chooses. The tower
        as a job the conductor plans. (Computed on the GPU in both cases.)
  - [ ] 4. Cells and positions in the runtime: image spans next to tokens, positions (t, h, w)
        per row, staging and a store of image rows.
  - [ ] 5. The splice in `qwen35` (a row source in the embedding gather; positions in the draft chain's links).
  - [ ] 6. The splice in `qwen4exp` (host rows; the n-gram feed's image token).
  - [ ] 7. Routes (`image_url`, Anthropic `image`, Responses `input_image`), limits, errors, `--mmproj`.
        Embedded images only; fetching remote URLs is backlog. At most 4096 rows per image by
        default (as the fork), with a key to lower it.
  - [ ] 8. Caching: cells named by a digest of the resized pixels, 64-bit names through the index and the SSD level.
- [ ] Audio, video and Gemma 4 are on the waiting list.
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
- Should the SSD cache be on by default with a default directory? (Today: only with `--cache`.)
- Do audio, video and Gemma 4 block the first release?
