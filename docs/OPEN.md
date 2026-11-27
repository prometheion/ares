# What is still open

What Ares does is listed in [`user/features.md`](user/features.md), what was done, by date, in
[`dev/history.md`](dev/history.md). This file is brought up to date at every merge into `main`. It lists work, defects and
measurements that are owed; what a feature must do in detail is in
[`dev/studies/feature-checklist.md`](dev/studies/feature-checklist.md) (223 rows: 70 done,
42 partly, 88 not started, 23 outside the frame; its rows K01, K02, K06 and P14 are out
of date: reasoning is built, and `q7` to `q4` run in the 27B).

GPU work runs on the test machine (RTX 4060 Ti 16 GB, gpu-unlock) from
2026-11-22 on; the two local cards are production and are used only when a test needs two
cards, with the owner told first.

Sorted on 2026-11-25 by kind (every item checked against `main` 4c834979; what was found done
or stale is in [`dev/history.md`](dev/history.md), 2026-11-25): the owner's reasons, the work in progress and the
owner's decisions first; then defects, small fixes, measurements owed and the questions to the
owner; then features and the store of levers. A defect or small fix names its files, and "GPU"
marks a line that only a GPU run can settle.

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
trees). What is left of "CUDA complete" and of "Optimising" is the store of levers below, drawn
from where it serves one of the reasons. Vulkan parity stays last.

## In work now
What was merged, and when, is in [`dev/history.md`](dev/history.md).


| Piece | State |
| --- | --- |
| Parallel requests against the owner's fork at production settings (the owner, 2026-11-23: "the whole reason I am making the engine"): two generating, one generating beside a long prompt, two agent sessions, 3 to 8 at once, priorities | measured (see "Parallel requests against the owner's production"); the fixes merged 2026-11-24 (above). Open: the expert store stalling under a full loan ("Defects"), a floor that rises for a request near its end ("Features"); `--parallel` defaults to 4 (the owner, 2026-11-24) |
| DFlash2 trees and the experts of a tree's tokens read ahead, phase 1 | measured (`flash/trees-study`, instruments only, not merged): Flash-Next +1 to 3 % with a tree of 4, larger trees lose, reading ahead from token tables or route memory +1 % (they name 1.4 to 6 % of a round's missing experts); 27B +9 % with trees of at most 8 rows; the MTP chain with its own candidates as siblings is as good as a DFlash tree of 8 nodes. The lever the data point at: a distinct expert in a few-row step costs 16.5 µs (about 80 GB/s of the cards' 360 / 500); at 5 µs chains reach about 120 tokens/s and trees then pay +15 %. Owner's decision on what to build ("Questions to the owner") |
| Jev mode (`feat/jev`, 2026-11-25) | built and measured (what was done in [`dev/history.md`](dev/history.md), the numbers in [`dev/experiments/decisions.md`](dev/experiments/decisions.md)); what is left under "Features", "Jev mode" |
| The push day's leftovers (`chore/leftovers`, 2026-11-25) | in work (under "Decided by the owner", the push day) |
| Prompt progress as llama.cpp's server sends it, and a default log for a person (`feat/progress-logs`, the owner, 2026-11-26) | built (see [`dev/history.md`](dev/history.md), 2026-11-26), not merged; what is left is under "Features", "Server and API" |

## Decided by the owner (2026-11-22)
- The triage's seven questions, answered 2026-11-25:
  - `--generation-floor` stays 20 tokens/s (the fork's generation beside a prompt); looked at again
    after the kernel pass makes steps cheaper.
  - A checkpoint at the end of every request (a place per request end), so a repeated or continued
    prompt resumes at its end, not at the last page or message boundary (to build: `fix/server-sweep`).
  - The RTX 3060's prompt attention takes 5 pieces a tile (about 6 % of its attention for some 25 MB
    of state slots a card); O in f16 is not taken (its error grows with the length). In the kernel pass.
  - The roles of the cards are built and tested now on the local RTX 4070 + RTX 3060, not after the
    RTX 5060 Ti: Flash-Next on these two cards with only the machine's 32 GB of RAM, the rest of its
    experts from the SSD, must be usable (slower, but working well). The option's name and form go to
    the owner before it is built.
  - The second card's role is remembered per model, cards, driver and binary, and measured again
    when one of them changes (built on `fix/model-sweep`, 2026-11-25: `roles.rs`, `<fingerprint>.roles`
    beside the usage profiles; "The second card's role" in `dev/architecture/devices.md`).
  - `cache_prompt` is accepted with its meaning: `true` is the normal behaviour, `false` makes the
    request neither read nor write the prompt cache (to build: `fix/server-sweep`).
  - DFlash2 trees come after the few-row kernels and are decided with those costs.
- Push day done 2026-11-25: the first push (prometheion/ares main at the scrubbed history) and
  the owner's production switched to Ares (llama-swap's two 27B entries run Ares from
  `~/Services/ares`, presets mapped from the fork's; both checked in a trial run and through
  the gateway); the llama.cpp trees, Spark, SGLang, vLLM and the fork's prompt caches are
  deleted (archived locally first). Left from it:
  - [ ] llama-swap updated to upstream with the owner's evict-lock patch (agent working).
  - [ ] The unit's stop timeout (CachyOS defaults to 10 s; Ares writes its prompt cache when it
        stops): the new unit is ready for the owner to copy (sudo).
- Push day (the owner, 2026-11-24): when the cleanup and the docs are done, the first push and
  the switch of the owner's production to Ares come together. Before it: the libcuda deadlock
  of page-locked frees fixed (it reaches the 27B too, through the prompt cache's checkpoints).
  Then: llama-swap's 27B entry runs Ares at the owner's settings with its own prompt-cache
  directory; the owner's llama.cpp trees go once every branch, stash and local commit is on
  GitHub or in a bundle, right away with no fallback period (the owner) (Ares's own uses moved first: the draft vocabulary, the tools' default
  paths, the fork comparisons from `/var/tmp/ares-survey/src/`); llama-swap's configuration is
  cleaned up and the program updated (restarts and installs are the owner's, with sudo).
  The history scrub, measured 2026-11-24: of 900 unpushed commits (GitHub has one), 3 to 7 hold
  the test machine's address, `/home/<user>` or a host name of the owner's network in a diff, none
  in a message, none in
  today's tree. Plan: rewrite origin/main..main with `git filter-branch` (nothing to install),
  replacing those with "the test machine", `~` and `chat.example.org`; nearly every hash
  changes, so a script then maps the commit hashes cited in today's docs and comments to the
  new ones in one commit; push without force, on the owner's word.
  Done 2026-11-25: the 911 unpushed commits rewritten (none of them holds those any more; today's
  tree unchanged but for this paragraph; messages, authors and dates the same; a bundle of the
  history before is kept locally), and the 40 commit hashes cited in the docs mapped.
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
  was named or not; the 27B's (dense attention) stays all in VRAM. Built 2026-11-24 (28e64535).
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
  after a pass and the wide-step bound fail now at such ties (under "Defects").
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
  the next Flash-Next piece after `flash/prompt`, not the optimising phase. Built 2026-11-23
  (`flash/layer-major`, the passes).
- Flash-Next on two cards: the plan measures at load whether the second card computes a dense
  share or holds experts only, and keeps the faster; no option. Built (`devices/load-choice`,
  `device/roles.rs`): on the two development cards it keeps experts only (the runs: [`dev/history.md`](dev/history.md),
  2026-11-23; what the trial costs at load is under "Questions to the owner").
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
  chooser does. Built 2026-11-22 (6227d864): each type takes the form its measured path gives, and
  where the forms differ the kernels that write a product's input write both.
- Capped cards measure their prefill path per type as full-rate cards do, in place of the rule
  kept from the two development cards (`FLUSH_AS_FAST`). Built (`devices/load-choice`); the
  rule stays only for a type nobody measured (tests and benches).
- Shared model layer (the owner's rule, 2026-11-16): generic for Ares, not like llama.cpp: a few
  models, shared code where it can be, each model still at its maximum through its own launch
  plan. A part becomes a shared block when it owns state the runtime touches (pages, slot state,
  checkpoints, windows, commits, replays, trees) or when it drafts; products, norms, kernel
  choice per row count, fused launches and exchanges between cards stay with the model. Steps 0
  to 8 made the K/V group, the recurrent block, the slot arena with its checkpoints, the token
  rows, the head view and the MTP driver shared (`crates/model/src/blocks`; the rule is in
  `dev/architecture/overview.md`, "The shared model layer"). Design study:
  `/var/tmp/ares-survey/shared-model-layer.md`. Each step lands green on its own; steps 9 and 10
  (trees) are under "Features".
- Not owed (the owner, 2026-11-16): replies that are bit for bit the same from run to run.
  They differ slightly when the CPU computes missed experts, and that is accepted.
- Not wanted (the owner, 2026-11-22): saving and loading a conversation's state by name
  (llama.cpp's `slot-save-path`); the prompt cache covers it.

## Defects
Wrong behaviour, a crash, a wrong number, a test that fails. Each line names its files; "GPU"
marks a line that only a GPU run can settle or check.

| Area | Defects | Of them GPU |
| --- | --- | --- |
| Server, API, cache | 5 | 1 |
| Model, runtime, kernels | 13 | 12 |

### From a user's report (2026-11-25; Flash-Next on one RTX 4080 16 GB, `--ram 40G`/`52G`, 4 sequences, opencode and Claude-Code-style requests): each takes the model down
- [ ] Two long prompts arriving together become one step of 8007 rows in 2 segments, which
      Flash-Next's layer pass refuses ("a step wider than 512 rows reads one prompt",
      `qwen4exp/device/runtime/pass.rs` `pass_of`): both requests get 500 and the model is unloaded.
      In work: `fix/flash-crashes`.
- [ ] Far K/V finds no RAM room ("the tier of experts is at its floor and the prompt cache holds the
      rest", `qwen4exp/device/flow.rs` `FAR_WAIT`) at `--ram 40G` beside a 39,810-token request and at
      `--ram 52G` repeating a 50k prompt: the prompt cache (lowest rank) does not give way, the tier's
      floor latches for good (`runtime/src/ram.rs` `experts_floor`), and the engine takes the model
      down instead of ending one request. In work: `fix/flash-crashes`.
- [ ] The same 49,965-token prompt sent again (one sequence, `--ram 52G`) was read again from the
      start instead of taken from the prompt cache, while the first request's far pages still held
      RAM as prompt cache; the second read's new far pages then hit the RAM failure above. A repeat
      must reuse the finished sequence's pages and checkpoints. In work: `fix/flash-crashes`.
- [ ] SIGTERM ends in SIGABRT (release `panic = "abort"`): core dumps up to 32 GB and minutes of
      hanging shutdown; llama-swap stops Ares with SIGTERM on every restart. In work: `fix/server-sweep`.

### Server, API, cache
- [ ] Chat and completion bodies pass over unknown members nested in a message, a tool or
      `stream_options`, where Messages and Responses refuse them; the server's rule is that
      nothing is dropped silently (found 2026-11-24 by `fix/help-docs`, documented as it is in
      `docs/user/api.md`). Files: `crates/server/src/openai/request.rs` (`message()`,
      `stream_options`), `crates/server/src/tools.rs` (`tool()`), `docs/user/api.md`.
- [ ] Two names of one limit or count with different values let one win silently: `n_predict`
      over `max_completion_tokens` and `max_tokens`, `n_cmpl` over `n`, `n_probs` over
      `logprobs`; only `typical` and `repetition_penalty` against their twins are a 400
      (`f32_alias`). Found 2026-11-24 by `fix/help-docs`. Files:
      `crates/server/src/openai/request.rs`, `docs/user/api.md`.
- [ ] A negative `min_keep` is accepted (only values above 1 are refused). Found 2026-11-24 by
      `fix/help-docs`. File: `crates/server/src/openai/request.rs`.
- [ ] With API keys the Origin check never runs, even when `--allowed-hosts` is named (written so
      in `access.rs`; with the option named it is not what the option says). Found 2026-11-24 by
      `fix/help-docs`. Files: `crates/server/src/access.rs`, `docs/user/options.md`.
- [ ] GPU. `ares-server --test real_cache`
      `flash_next_resumes_a_conversation_from_the_cards_from_ram_and_from_ssd` failed once in the
      full suite under load (2026-11-17: after two other conversations took the pool, 4096 or
      fewer of 9851 tokens came from the cache); alone it passes (9728 from SSD). Suspected, and
      still possible in today's code: a page on its way from RAM to the SSD is found in neither (a
      RAM page whose bytes are with the writer is not `ready`, and the drive reports stored
      records only). Since 2026-11-24 the test runs with K and V beyond VRAM (`--context 20480`).
      Files: `crates/runtime/src/cache/moves.rs` (`import_from_ram`), `cache/ram.rs`,
      `cache/ssd/mod.rs`, `cache/ssd/pump.rs`.

### Model, runtime, kernels
Tests that fail:
- [ ] GPU. `qwen4exp_runtime`'s `flash_next_writes_with_drafts_what_it_writes_without` fails on
      the two local cards in some runs (2026-11-23, IQ2_XS with the MTP block): the greedy reply
      with drafts parts from the one without at a near tie (token 16, "," against "."), with the
      whole head or with the draft vocabulary; main failed it in one of two runs, `fix/turn-cell`
      in three of three; it passed on the test machine's one card. The test still asserts equal
      text; by the owner's rule (2026-11-23) it must judge the first differing token by the
      logits, as `drafts_change_no_token_in_any_placement` does (`reference_gap`). This model has
      no CPU reference: the plain run's log-probabilities would serve. File:
      `crates/model/tests/qwen4exp_runtime.rs`.
- [ ] GPU. Device tests that fail under machine load at near ties, unchanged since:
      `qwen4exp_runtime` `a_request_cuts_in_between_the_pieces_of_a_long_prompt` (2026-11-24: 2
      suite runs at load 8 to 21, once the long answer began 73 instead of 41; alone 16 of 16 with
      every token the reference's, and 6 alternating suite runs passed; not tied to a change);
      `qwen4exp_device` `steps_wider_than_eight_rows_match_the_reference` (2026-11-23, RTX 4070
      alone at load 12 to 17: 2 of 4 runs, main's build as often; in the Fast precision the last
      row's logits after the 256-row step differ by 0.13 to 0.215 relative RMS from run to run
      against the bound 0.2, with 882 or 883 experts picked and 50 to 61 more rounds: the spread
      is the step's own, not the load's), `every_layer_of_wide_steps_matches_the_reference`,
      `experts_of_every_weight_type_on_the_cpu_change_nothing_*`. They should print the answer's
      logits against the reference's when they fail and judge a near tie by them (the owner's
      rule); the wide step's bound or its spread is to be settled (whether the integer prompt
      kernel raised the spread is not measured). Files: `crates/model/tests/qwen4exp_runtime.rs`,
      `crates/model/tests/qwen4exp_device.rs`.
      *Seen on `fix/model-sweep` (2026-11-26, both local cards, load 10 to 30):* in the whole
      `qwen4exp_runtime` suite `a_prompt_read_layer_by_layer_leaves_checkpoints_inside_it` failed
      (its message was lost: the run was stopped) and the next test,
      `a_prompt_read_layer_by_layer_reads_k_and_v_beyond_vram_from_a_copy`, did not end in 20
      minutes (device 0 at 100 %, its device thread spinning, device 1 idle), as if a kernel the
      failed test left behind still waited. Alone both passed (10 s and 3 s), and the suite again
      31 of 31 at load 8 to 15. A test that fails should not leave the cards waiting for the
      next. The same day `qwen4exp_device`
      `experts_of_every_weight_type_on_the_cpu_change_nothing_3` failed once at load 16 to 20 on
      "every expert has a slot" (353 of 384 experts got one, so the device had less room than at
      other times) and passed alone at load 5 to 10.
- [ ] GPU (two cards). `qwen4exp_device` with both cards on every layer
      (`ARES_TEST_SPREAD=tensor`, the share 0.5 and experts only alike), 2026-11-23 on main a36d379:
      `every_layer_of_wide_steps_matches_the_reference` (attention core 5.16e-2 against a bound of
      3e-2 for the shipped types) and `experts_of_every_weight_type_on_the_cpu_change_nothing_4`
      (f32 and q2_K: 4 of 480 misses not computed on the CPU) fail; they passed on the evening's
      main (753dcb1). Neither test changed since. Files: `crates/model/tests/qwen4exp_device.rs`,
      `crates/runtime/src/expert_store/cpu/`, `expert_store/hybrid.rs`.
- [ ] GPU. Several test threads on the same cards: `qwen4exp_device` with four test threads
      (2026-11-23, both local cards, `cuda/tested-archs` and its base ae3a334 alike):
      `every_layer_matches_the_reference`, `every_layer_of_wide_steps_matches_the_reference` and
      `a_long_step_that_misses_experts_ends` fail, and the tests with experts on the CPU or
      uploaded run past 15 minutes; one at a time each passes in seconds, the suite with
      `--test-threads=1` 25 of 25 in 183 s. `qwen4exp_runtime` with the default test threads hung
      for over 18 minutes once (two device threads spinning; the `moe/prompt-chunks` agent);
      with `--test-threads=1` it passes (the hang may be the libcuda deadlock fixed 2026-11-24; not
      run again). Fix: one process-wide lock for the GPU tests, or `--test-threads=1` written in
      the files' heads. Files: `crates/model/tests/common/gpu.rs`, the two test files.
- [ ] Rust 1.100 (stable on the test machine) crashes compiling `ares-model --test
      qwen35_failure` (an internal compiler error); 1.99 builds it, and the test passes there.
      `rust-toolchain.toml` names `stable`, not a version, so each machine takes its own; the
      test file changed since (not known whether the crash remains). Fix: pin the toolchain, or
      change the test. Files: `rust-toolchain.toml`, `crates/model/tests/qwen35_failure.rs`.
- [ ] GPU. `ares-server --test real_flash_next` `flash_next_drafts_with_its_mtp_block` sometimes
      verifies no draft at all (0 of 0) for a whole run (2026-11-17, and 2026-11-22 in `real` 21 of
      22); passes in other runs. The suspicion of 2026-11-17 (warm-up off with drafts) does not fit
      today's code: the server keeps the conductor's warm-up, which measures the draft windows.
      Files: `crates/sched/src/drafting.rs`, `crates/runtime/src/conductor/measure.rs`,
      `crates/sched/src/price.rs`.
- [ ] GPU. `ares-model --test qwen4exp_runtime`
      `the_mtp_block_drafts_what_the_reference_drafts_in_every_placement` failed once in the merge
      check of 2026-11-22 (first links), passed three times alone. In one of those runs the policy
      kept asking the made-up block for 21 chains (78 drafts, none accepted) where it otherwise
      stops after one; then the allowance of one parting chain in five is at its edge. Why the
      policy keeps asking a block that is never right is open (the policy only moved, into
      `ares_sched::Drafting`). Files: `crates/sched/src/drafting.rs`,
      `crates/sched/src/confidence.rs`, the test.

Wrong behaviour:
- [ ] GPU. Flash-Next: rounds between the pieces of a paused pass can stall the expert store when
      the pass borrows all the slots can lend ("a step stood still for 30s while the host held
      back the next layer ... the store waits for 7 to 9 experts and is not settled": a 21k-row
      pass opened beside a generating request, 3 runs of 4 on one card; a pass opened alone and
      then joined by generation did not stall in 2 runs). Windows of 4 rows and the MTP chain then
      need more distinct experts than the loan leaves a layer. The conductor lends a pass beside
      generation at most half; the store itself should bring a step's experts through transit or
      the CPU whatever the loan holds. Since a6d73b00 a loan leaves a layer `top_k.min(floor)`
      slots (the item said `top_k + vram_reserve`), lower than when the stall was seen: check it
      again first. Files: `crates/runtime/src/expert_store/scratch.rs`, `hybrid.rs`,
      `transit.rs`, `crates/runtime/src/conductor/mod.rs`.
      *Checked on `fix/model-sweep` (2026-11-25), not reproduced:* the local RTX 4070 alone,
      IQ2_XS with the MTP block, `--ram 8G`, `--context 40960`; a fresh prompt of 31.1k to 31.2k
      tokens read alone as a pass (steps of up to 32768 rows on lent VRAM), joined 3 s later by a
      request that generates 400 tokens (`/var/tmp/ares-fixmodel/ab/d4.sh`): main a8920d61 4 runs
      of 4 without a stall (prompt 62 to 69 s, the 400 tokens 104 to 114 s), the branch 2 of 2 and
      4 more with the prompt cached, load 6 to 36. A pass opened beside generation is half as wide
      (the conductor), where it never stalled. What is left is the robustness the item names (the
      store bringing a step's experts whatever a loan holds), untested.
- [ ] GPU. A step fails when the tier of experts cannot grow by a piece (found 2026-11-25 on
      `fix/model-sweep`, 1 of 8 runs of the check above on the branch, none of 4 on main, not tied
      to a change of the branch): `Flow::before_step` grows the RAM tier by a piece
      (`grow_ram(1)`) and passes a refused booking on as the step's error, so the engine fails and
      the model is taken down ("the RAM limit of 8.00 GiB has no room for 50.2 MiB more of the tier
      of experts: the step record and read-back hold 41.2 MiB, the K/V beyond VRAM 285.3 MiB, the
      tier of experts 7.54 GiB, the prompt cache 112.6 MiB"; a request generating beside a 31k
      prompt's pass). The grow is optional; its refusal should leave the tier as it is (as
      `Flow::follow_ram` does). Near `fix/flash-crashes`' far K/V item. Files:
      `crates/model/src/qwen4exp/device/flow.rs` (`before_step`),
      `crates/runtime/src/expert_store/elastic.rs` (`grow_ram`), `crates/runtime/src/ram.rs`.
- [ ] GPU. **Lent steps with the CPU's share** (found by `kvfix`, 2026-11-24; main's, not the far
      pages'): on the made-up model (`prompt(300, 7)`, the BF16 test file, steps of 32 rows on a
      loan, placement One) the first token's log-probabilities are 58 -1.80..-1.85 and 22
      -2.20..-2.25 in most runs and 58 -2.14, 22 -2.10 in the others, the same values every time
      (the token after it follows: 39 at -1.89 or -1.74): two states, not a spread. Main 2340ca0
      8 of 12 runs (7 answered 22), main acb2587 One 6 of 8 and by layers 1 of 8, alike with and
      without far pages; with every miss uploaded (`Serve::Stream`) 0 of 16, in wide steps of the
      own scratch (about 32 experts on the CPU a step, against 370 on a loan) 0 of 24, with every
      miss the CPU can take on it (`Serve::Cpu`) 5 of 16. The rows' int8 input of the CPU against
      the GPU's f16 moves the first token by a few hundredths, not by 0.3: something of the CPU's
      share on lent VRAM is computed differently in some runs. The reading of `misses.rs`,
      `hybrid/wide.rs`, `cpu/mod.rs` and `transit.rs` found no cause; a diagnostic that prints the
      first tokens' log-probabilities per run (`zz_dbg_lent_gaps`, with the serve mode and a log
      line per wide layer's CPU share) is in `/var/tmp/ares-kvfix/` (`exp5-all.patch`).
      *Classified on `fix/model-sweep` (2026-11-25): a defect, not a near tie.* The CPU reference
      (the patch's `zz_dbg_reference_margins`, same file and prompt) puts 58 first with a gap of
      0.433 to 22 (logits); the common state's log-probabilities (-1.80..-1.85 against
      -2.20..-2.25, a gap of about 0.40) agree with it, the other state's (-2.14 against -2.10, 22
      ahead by 0.04) is off by about 0.47. The cause is not found. Files:
      `crates/model/src/qwen4exp/device/misses.rs`, `crates/runtime/src/expert_store/hybrid/wide.rs`,
      `expert_store/cpu/mod.rs`, `expert_store/transit.rs`.
- [ ] GPU. With both devices on every layer the MTP block's chains after a pass agree with the
      reference's for 1 to 3 links where after steps of the own rows 5 to 7 (the trunk's tokens
      are the reference's): cause not found. Files: `crates/model/src/qwen4exp/device/part/pass.rs`,
      `qwen4exp/device/runtime/pass.rs`, `qwen4exp/device/part/mtp.rs`.
- [ ] GPU. Runs that should give the same bits do not (found while measuring pruning, IQ2_XS, two
      cards, the tool `qwen4exp_prune`, 1000 tokens teacher-forced after a prompt of 256): with the
      prompt in steps of eight rows the next texts in the same slot after `Session::restart`
      differ (KL 0.0097 English, 0.042 code, read after German), and even the first text of a
      process differs a little where many misses are waited for (KL 0.0001 German, 0 English,
      0.0025 code, top-1 99.0 %). Owed: find what it reads that depends on the run (tier state, a
      slot's leftovers, the order in which waited experts land), and say in
      `dev/architecture/flash-next.md` which paths are reproducible. The CPU computing misses
      (two runs differ by KL 0.0063, top-1 96.8 %) and a wide step's `red.add` in arrival order
      (KL 0.0065) are accepted by the owner's rule (2026-11-23). Files:
      `crates/model/src/qwen4exp/device/session.rs`, `crates/model/examples/qwen4exp_prune/`.
- [ ] GPU. The 27B, seen 2026-11-17 (main 6bcc626 as well): with `max_tokens` 40 the first greedy
      reply to `real_drafts.rs`' German question after a start (IQ2_XXS, one card, fast kernels) and
      the same request again, its prefix from the cache, are different tokens (hashes
      4d9a1bef2efe92d9 and c1f7796b6753a9e5), with drafts and without; at 160 tokens all agree.
      Not looked into (a cache replay runs as wide rows, which round differently from single rows,
      but then 160 tokens should differ too). Files: `crates/model/src/qwen35/device/`,
      `crates/runtime/src/cache/`, `crates/server/tests/real_drafts.rs`.
      *Looked at on `fix/model-sweep` (2026-11-25, main a8920d61, the local RTX 4070, one card,
      no drafts, `/var/tmp/ares-fixmodel/ab/d8.sh`): a near tie, not a defect by the owner's rule.*
      The two replies of 40 tokens (the same hashes as above) part at their last token only: the
      first, its prompt of 66 tokens read as one step, has " " -1.411 before ' "' -1.475; the second,
      from the cache (`prompt_n` 1: one row computed after the state at cell 65), ' "' -1.428 before
      " " -1.481. The first reply of 40 is the first 40 tokens of the reply of 160. At 160 all agree
      because none came from the cache: each of the three requests read all 66 tokens (the reply
      ends with its end token at 148 tokens, and the next request found no state to resume from;
      after a reply cut at 40 it did). That difference of the cache is the server's
      (`fix/server-sweep`: a checkpoint at the end of every request).

## Small fixes
Gaps a few hours close: a missing option, a limit, a test, a doc, a leftover of a done piece.

| Area | Small fixes |
| --- | --- |
| Server, API, cache | 10 |
| Model, runtime, kernels | 5 |

### Server, API, cache
- [ ] An MTP block inside Flash-Next's own model file is used only when `--drafters mtp` names
      it; the 27B uses its block by default (found 2026-11-24 by `fix/help-docs`). By the owner's
      rules (parity, and a pure improvement on by default) Flash-Next should too, once it is
      measured to pay there (the block's VRAM is taken from the expert slots). Files:
      `crates/server/src/load/qwen4exp.rs` (`in_model`), `crates/server/src/options.rs` (help),
      `docs/user/options.md`.
- [ ] Larger. The roots and left-behind checkpoints with one place in RAM (Flash-Next at
      `--ram 8G`): a root that comes back from the drive is a plain checkpoint again (the store
      keeps no mark); message checkpoints that later requests left behind are not written to the
      drive, and a checkpoint left behind at deeper page boundaries is not written when it leaves
      the index (one that leaves with its page, after the deeper ones of its text left, is); so an
      edit far back resumes from the root or from the drive's deepest checkpoint before it. With
      one place (measured 2026-11-22, Flash-Next `--ram 8G --cache`) the roots do not pay yet: the
      root leaves the place for later checkpoints and goes to the drive, but a prompt asks the
      drive only for pages the index does not hold, so a new conversation with the same 12k-token
      system prompt still replayed 18 304 rows (221 s), and a new Claude Code session reads all its
      16 thousand tokens again. Wanted: a checkpoint brought back from the drive for a page the
      index holds (the SSD level imports runs of pages with their deepest state only), or more
      places (int8 checkpoints, "Store of levers", "Checkpoints"). Files: `crates/runtime/src/cache/ssd/mod.rs`
      (`Cache::probe`), `cache/ssd/pump.rs` (`drain`), `cache/places.rs`, `crates/state/src/checkpoint.rs`
      (`left_behind_on_pages`), `crates/store/src/lib.rs`.
- [ ] The host executor's places: on `PlaceChunks`, and a place whose memory a device keeps locked
      after it was given up. The host executor frees it at once (its oracle plays the strictest
      device, which the chunks are not), so the fault where pages in RAM left no room in `--ram`
      for a checkpoint's lock was found on the devices only. Files: `crates/runtime/src/testing.rs`,
      `testing/slots.rs`, `crates/runtime/tests/cache/ram.rs`.
- [ ] A test of the full drive for the SSD level (running out of room is handled, never tested).
      Files: `crates/runtime/src/cache/ssd/mod.rs`, `crates/runtime/tests/cache/ssd.rs`.
- [ ] Left of truncation (F18, done 2026-11-22): a cut of a prompt to its last N tokens (vLLM,
      TGI, SGLang) is not built, see "The Responses API" in `dev/architecture/api.md`. Files:
      `crates/server/src/openai/request.rs`, `crates/server/src/generate.rs`, the docs.
- [ ] Left of the watchdog (S23, done 2026-11-22): no option for the limit (a field of
      `RuntimeConfig`, 600 s); the steps of the warm-up are not watched (the watch starts after
      it); the memory a stuck device holds stays taken until the process ends. Files:
      `crates/server/src/options.rs`, `crates/server/src/plan.rs`, `crates/runtime/src/engine.rs`.
- [ ] `/metrics` of a model that is not loaded is still 503 (its `/props` is read from its
      preset and file since 2026-11-22). Files: `crates/server/src/metrics.rs`,
      `crates/server/src/load/mod.rs` (`Described`), `docs/user/api.md`.
- [ ] Router: "no room" for Flash-Next is recognised by the message's text ("does not fit",
      "memory"), not by a kind of error; the load and unload times are not in the user docs. Files:
      `crates/server/src/load/qwen4exp.rs`, `load/start.rs`, `crates/server/src/models.rs`,
      `docs/user/`.
- [ ] The cache's transfer time is counted twice, exported as `cache_ram_transfer_nanoseconds_total`
      and `cache_ssd_transfer_nanoseconds_total` (cleanup phase 2: a metric's name). Files:
      `crates/runtime/src/cache/moves.rs`, `crates/server/src/metrics.rs`.
- [ ] The mirostat range is checked twice, by `Sampling::validate` and by the runtime's request
      sampler. Files: `crates/api/src/sampling.rs`, `crates/runtime/src/sampling.rs`.

### Model, runtime, kernels
- [ ] Left of step 5 of the shared layer, the token rows: DFlash's newest token and mask rows are
      still decoded on the host for every pass (both models). Files:
      `crates/model/src/dflash/device.rs` (`pass_layers`), `qwen35/device/part/dflash.rs`,
      `qwen4exp/device/part/dflash.rs`, `crates/model/src/blocks/rows.rs`.
      *Not done on `fix/model-sweep` (2026-11-25):* what it takes: `pass_layers` writes a pass's
      `block` rows per sequence on the host (the newest token's row decoded through the `embed`
      callback, then the mask row) and uploads them, 16 rows of `n_embd` f32 a sequence (320 KiB
      for the 27B). On the device: the mask row uploaded once at open and copied into rows 1 to
      `block - 1`, the newest token's row gathered from the table's staged rows (`rows::source`,
      `embed_gather`) into a row of its own and copied to row 0; the drafter's device test against
      the reference (`crates/model/tests/dflash.rs`, the 27B's drafter file) checks it.
- [ ] Flash-Next past the dense limit: a draft window's row that reaches across the point where
      a tile leaves the newest cells reads that tile exact, the row alone reads it converted
      (`Group::ahead` is planned for rows within the limit only; the lists are made on the
      device from the table, which still names the exact entry). A shadow row of the table for
      such rows would close it. Files: `crates/model/src/blocks/kv.rs`, `blocks/forms.rs`,
      `crates/model/src/qwen4exp/device/part.rs` (`build_plan`). *Not done on `fix/model-sweep`
      (2026-11-25).*
- [ ] Transit adoption (`transit_adopt`, the prompt chunks' experts taking VRAM slots after a
      wide step) asks only "hotter than the coldest member" (with the swap margin when the member
      has to be copied down), not the rule of a promotion (`promote_hot`); it was left for the work
      on prompt chunks that borrow expert slots, which is merged. Files:
      `crates/runtime/src/expert_store/transit.rs`, `expert_store/hybrid.rs`.
      *Looked at on `fix/model-sweep` (2026-11-25), not changed:* the two rules differ where the
      costs do. A promotion uploads the newcomer over the link, so it asks the margin and the bonus
      of `SwapRule` (against trading places back and forth, an upload each time); an adoption finds
      the newcomer on the device already (a copy inside VRAM), and asks the same margin exactly
      when the swap costs a transfer (the member copied down to RAM). When RAM holds the member,
      the swap moves nothing over the link, and a promotion back still needs the margin, so no
      upload can trade places back and forth. The promotion's rule there would keep a colder member
      in VRAM for nothing. Open only as a measurement (adoptions per prompt and the decode after it
      with the margin on both), if someone doubts it.
- [ ] The MTP block's experts are not in the usage profile (they are filled in the order of their
      numbers at the start). Files: `crates/model/src/qwen4exp/device/usage.rs`,
      `qwen4exp/device/feed.rs` (`fill_mtp`), `crates/runtime/src/expert_store/profile.rs`.
      *Looked at on `fix/model-sweep` (2026-11-25), not a gap:* the MTP block's layer is resident
      (`feed.rs`, `spec_of(.., true)`; its bytes, every expert, come out of the device's budget
      before the tiers), and `fill_mtp` brings all of its experts when the parts open, before the
      first step. A profile could only change the order of a fill that ends before the engine
      serves; nothing of the block is chosen by heat. It matters only if the block stops being
      resident (a block larger than VRAM), which nothing plans.
- [ ] Float experts beyond the f16 range are not caught in the many-row kernels (`WeightRange` is
      applied to the dense matrices and the router only). Files:
      `crates/model/src/qwen4exp/device/load.rs`, `qwen4exp/device/part/wide.rs`,
      `crates/cuda/src/experts.rs`.
      *Looked at on `fix/model-sweep` (2026-11-25), not done:* the many-row kernels turn a float
      expert's weights into f16 (`EngMmFloat`, `mm_types_more.cuh`; beyond 65504 an infinity). The
      dense matrices are read at load, the experts are not (a streamed model's experts would all
      have to be read), so the check belongs where the store reads an expert (float shapes only, a
      pass over its bytes), and a wide step needs a way for a layer with such an expert that does
      not exist yet (the decode kernels eight rows at a time, or the CPU). Only files with float
      experts are concerned; none of the owner's are.

### Decode under offload (a user's runs, 2026-11-25): the aim of item 3
One RTX 4080 16 GB, Flash-Next UD-IQ4_XS (87 GiB), `--ram 52G`, `--context 131072`, `--parallel 1`, MTP:
prompt 1,240 / 1,167 / 1,212 tokens/s at 50k / 100k / 113k (Strata's level), decode 12.9 / 13.1 /
13.8 tokens/s there and 18 to 22 at short contexts. Its log: 13.1 % of the experts in VRAM slots, 90.7 %
in the RAM tier; a step's price measured at load 17.4 ms for one row but 126.9 ms for two (another row
picks other experts, so drafts barely pay: 12 links in 6 steps), and real steps of 45 to 75 ms: the
misses (a CPU job of 327 us per IQ4_XS expert and row, or an upload of 1.24 ms) dominate. Strata-HotRod
reports 15 to 25 tokens/s for UD-Q4_K_XL on an RTX 3060 with 30 GB RAM, mostly from its stand-in experts.
Item 3 starts here (on the test machine, the closest setup): a profile of a decode step under offload
(VRAM hits, RAM uploads, CPU jobs, drive waits, attention and indexer at long context), then the
stand-ins taken from HotRod's code (quality against the reference, the owner's rule), the VRAM hit
rate, the miss path, and the few-row kernels so drafts pay again.

## Measurements owed
Runs on a GPU with no code to write; the place each result goes is the experiments file named.

### Flash-Next against Strata and from the SSD (the test machine)
- [ ] The SSD comparison (item 3 of the order): Ares's passes (prompts layer by layer, built
      2026-11-23) against the newest Strata on the test machine, 2k / 8k / 32k, UD-Q3_K_XL,
      UD-Q4_K_XL and UD-Q5_K_XL, at equal budgets. The last numbers (2026-11-23, engine 0.1.65,
      UD-Q3_K_XL at 40 GiB, before the passes): decode 34.4 / 25.3 tokens/s with / without drafts
      against 22.0 / 20.2; prompts 178 / 260 / 291 against 241 / 796 / 1159; a cached agent turn 6
      to 31 s against 0.3 to 0.5 s. Log: `dev/experiments/against-strata.md`.
- [ ] The Q4_K_XL file (104 GiB), base against the predictions; prompts on the real file in wide
      steps (`--prefill 4000 --rows 512`); more than one pair per configuration on a quiet machine.
- [ ] **A short wide step costs what a long one does.** The cost of a step from the SSD is the
      experts it touches, not its rows: a prompt of 69 tokens (one step) took 3.2-4.0 s, a
      replay of 111 rows 18 s. Steps of 8 rows (the CPU serves what RAM holds) may be cheaper
      for short prompts; the conductor's prices of wide steps are measured at the start on one
      token repeated and say 270 ms for 64 rows. The run `round6` (2026-11-22) left no result in
      the docs.
- [ ] Strata uses the page cache beyond its budget: 12.2 GiB of the model's pages were cached
      at the end of its decode session (MemAvailable 12.8 GiB while loaded, Ares 20.7 GiB at
      `--ram 40G`). Equal budgets are not equal memory; Ares at `--ram 52G` is owed.
- [ ] Pruning: (3) a decision per expert over a window (Strata's) instead of per row; (4) where a
      third of all picks are touched (Q4, `skip:1`), stand-ins with the table's scale (KL 0.42)
      and the router's renormalised (0.46) beat skip (0.50): skip renormalised at that rank; (5)
      the speeds were measured with every miss uploaded (`--serve stream`, for runs that repeat),
      not with the CPU serving them. Strata-HotRod's stand-in experts are a candidate
      (`dev/studies/engines-survey.md`).
- [ ] The first reading of a short prompt after other texts was slower with transit (35 against
      86 tokens/s in one run, 2026-11-17); not understood, and the transit path changed much
      since: run it again before looking for a cause.
- [ ] Flash-Next with 1, 2 and 4 concurrent clients against Strata (Strata saw four requests on a
      12 GB card with experts outside VRAM 11 to 24 % slower in sum: parallel requests buy waiting
      time there; gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`). The 27B against the
      fork was measured 2026-11-23 (`dev/experiments/against-the-fork.md`).
- [ ] Flash-Next with the owner's real limit (a RAM tier of 24 to 26 GiB), one card and two.
- [ ] The SSD cache's times to the first token (taken under load so far).
- [ ] Flash-Next with its MTP block once speculation pays; target above 56 tokens/s.

### Flash-Next, other
- [ ] Flash-Next with several sequences: a generated row is planned alone (no prefix read
      together with other sequences, `alone_pieces`); its cost with three sequences was not
      measured. Then, if it pays: `crates/model/src/qwen4exp/device/part.rs` (`build_plan`),
      `blocks/kv.rs` (`plan_rows`).
- [ ] Left of K and V in `q7` to `q4` (done 2026-11-16): the share of drafts accepted with the
      MTP block's `q4` K and V was not measured on Flash-Next (`blocks/mtp.rs`, `MTP_KV`).
- [ ] The share of the room for K and V in VRAM (a thirty-second, `plan::NEAR_SHARE`) is the old
      rule's, not measured against other shares.
- [ ] Left of the slower tokens right after a long prompt (2026-11-22, explained in
      [`dev/history.md`](dev/history.md)): the returned slots now fill as they come back (605b484f,
      39153798), but only after the first step off the loan; the same no-loan control (`-b 512`)
      on a quiet machine; the A/B of the build with the rows scored 256 at a time.
- [ ] Passes and lent wide steps with both cards on every layer (the machine had no driver with
      peer access when they were built; the local cards have it since 2026-11-22).
- [ ] Flash-Next's long prompt beside generation as a paused pass on the two local cards at
      `--ram 8G` (measured on the test machine only: 8k 61.8 -> 16.7 s).

### The 27B against the owner's former production (the fork)
The fork is archived since 2026-11-25: comparisons use a build from a clone under
`/var/tmp/ares-survey/src/`. The settings and tables: [`dev/experiments/against-the-fork.md`](dev/experiments/against-the-fork.md).
- [ ] Quality against the fork at equal settings (KL or top-1); VRAM was measured 2026-11-23
      (11.5 / 10.1 against 11.6 / 11.7 GiB).
- [ ] Float matrices on a capped card use f16 sums (0.04 to 0.06 % rms) where the
      F32 matrices of an IQ4_XS file were exact before: decide with a KL measurement (the 27B's
      chooser; Flash-Next's float products keep f32 sums).
- [ ] A turn that continues at 17-23k read 576 tokens/s (fork 880; Ares 1020 from the start of a
      prompt), measured before the prompt attention kernel merged (2026-11-23): run it again.
- [ ] Images at the owner's production settings (production runs on Ares since 2026-11-25).
- [ ] Texts under several concurrent requests against the reference (a generating step of more
      than 16 rows changes kernel family, three sequences drafting at once; near ties allowed by
      the owner's rule of 2026-11-23); the replay from the prefix cache rounds differently from
      single rows.
- [ ] On `/v1/responses` a second turn took 1 of 35 prompt tokens from the cache where
      `/v1/messages` took 18 (the 0.8B): on Flash-Next and the 27B the two routes take the same
      (2026-11-22: 83 of 99 and 89 of 105 in `real_responses` and `real_messages`); the 0.8B was
      not run again.
- [ ] Two elastic models on real devices (the router; `real_elastic` runs one).
- [ ] Before a release: both models against the fork, Strata and TokenWing (Flash-Next), speed,
      VRAM, quality.

## Questions to the owner
- None open (the seven of the triage answered 2026-11-25, under "Decided by the owner").

## Features
Pieces of real work, days each, in the order of work where the owner placed them.

### Speculation and drafters
- [ ] DFlash2 trees (steps 9 and 10 of the shared layer): tree verification in the blocks, verified
      once for every drafter and both models; tree fill from the DFlash lattice, MTP links mixed
      into a tree; node allocation over sequences (study `/var/tmp/ares-dflash-study/REPORT.md`).
      With the experts of a tree's tokens read ahead (the owner's idea, 2026-11-18; no engine has
      it): route memory for the `ngram` drafter's copied text (about 1 KiB a token), a table of each
      token's usual experts per layer by decayed counts for `mtp` and `dflash`, a tree's branches
      weighted by their chance within the drive's measured pace, a tree priced by the tiers its
      rows' predicted experts are in. The owner (2026-11-22) expects much of them for decode; phase
      1 is measured ("In work now"). What trees find from step 8: the lattice of every pass is kept
      (`Drafter::lattice`: 16 candidates and a 16 x 16 score table per position, tokens already
      mapped through the head view); the walk is one function of it (`Lattice::chain`, kernel
      `ares_dflash_select`), which a tree fill replaces; the drafts list has a place per window,
      which becomes a place per node; the inject is per row and position, so a tree's kept path
      needs only its rows' positions (a rejected branch's rows are never read, as now).
- [ ] DFlash2 on Flash-Next split by layers (refused at load today): the first device's taps (or
      its share of the fusion, `rows x n_embd`) must reach the head's device; the hand-over lane
      could carry it.
- [ ] A classic draft model as a fourth drafter kind (llama.cpp's `-md`): a small model with the
      same vocabulary drafts with its own forward passes and its own K/V; chosen by the same price
      rule as `ngram`, `mtp` and `dflash`. Later (the owner, 2026-11-17).
- [ ] The roles of the cards (the owner, 2026-11-17; the question above): main cards hold the model
      and its expert cache; an aux card holds drafters (a draft model, DFlash, an MTP block) and,
      for a mixture of experts, its free VRAM as one more expert tier. Built of it (2026-11-22, no
      option): for the 27B the second card is worth most as room for K and V
      (`AttentionShares::ByContext`, heads with the weights as far as the wanted context fits; 2.1
      times the context, 3 % slower decode at 1000 cells and 9 % at 32 000); Flash-Next's second
      card's role is timed at load. Drafters on the second card are not built: a chain on the
      slower card is slower by what that card is slower, and nothing runs beside it in a step that
      only drafts; it pays only with a pipeline that drafts the next window while the main card
      verifies this one (Strata's `--pipeline-windows`).
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
      only, so the first chance to compare; `dev/studies/engines-survey.md`) and Strata on the same card.

### The pass over all kernels (item 5 of the order)
- [ ] Few-row kernels for windows and trees, so that a step of 8 to 16 rows costs little more than
      one (here 16 rows cost 1.46 times one; TensorFold's tree gains rest on 1.05 to 1.2): the
      27B's dense matrices, Flash-Next's experts (a distinct expert in a few-row step costs 16.5 µs,
      about 80 GB/s of the cards' 360 / 500; at 5 µs chains reach about 120 tokens/s and trees then
      pay +15 %); attention for wide windows (drafts still run on the decode shapes).
- [ ] The prompt attention on the RTX 3060 (about 45 % of its tensor rate against the fork's 67 %:
      the fork keeps O in f16 over the whole range and the query in registers; Ares keeps O in f32,
      128 of 255 registers, two warps per scheduler whose latencies do not overlap): a design with
      warps that only decode and others that only multiply (more warps than the f32 accumulator
      leaves registers for today). The RTX 3060 alone reads a 2048-token IQ2_XXS prompt at 536
      tokens/s against the fork's 563 (`dev/architecture/kernels.md`).

### Flash-Next
- [ ] Gate and up of a layer's routed experts in two types are refused (the expert kernels read
      both with one decoder): the owner wants it (2026-11-17): two launches where the types
      differ, the fused kernel where they agree.
- [ ] Pruning, what is left to build: (1) the table for `similar` is made by the tool
      (`qwen4exp_prune calibrate` and `near`, minutes of GPU and CPU); the server only reads it,
      and refuses `similar` without it; (2) slices of a prompt (wide steps) are not pruned.

### Prompt cache
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
- [ ] An unload or a stop waits for the drive to take what only RAM and the devices hold (on
      the btrfs volume here, compressed and busy: 20 to 100 s for 1.5 GB; the wait is capped at
      120 s, where a unit stops a service after 10 s by default). What only RAM holds could be
      written after the devices are released, by a writer that outlives the engine.
- [ ] One size limit for all models on a directory (each engine keeps its own `--cache-size`); a
      layout that does not depend on the split (another split is another store today).
- [ ] Cache policy: memory of reply tokens, marks, warm pass (the resume point at the prompt's end
      is a question to the owner).

### Server and API
- [ ] Left of the prompt progress and the log (`feat/progress-logs`, 2026-11-26): a step of
      Flash-Next's prompt on lent VRAM that is not a pass, and the 27B's prompt steps, report
      after each step only (the share inside a step comes from the pass alone); the acceptance
      of drafts by position is in `/metrics`, not in a request's lines (llama.cpp prints it at
      trace).
- [ ] Left of the thinking budget and the loop guard (K03, K05, done 2026-11-22): the
      thresholds are fixed (no option); a passage loop of longer cycles (Strata's opt-in #728)
      is not looked for.
- [ ] Left of priorities (S06, done 2026-11-22): a prompt of a worse priority still takes its
      share of the time next to generation of a better one (the floor of generation, not the
      priority, bounds it); a threshold of the gap before a running request gives way (SGLang's,
      10 by default) is not there (any gap is enough).
- [ ] Two prompts at once are read one after the other (shortest first, the order that gives the
      lower mean wait); the second then takes turns with the first's generation, at the floor.
      The agents' turn with a 5.8k tool result sent at once (main acb2587, two rounds, cold and
      warm): the second agent's first token 17.0 / 17.4 / 17.0 / 19.2 s before, 16.6 / 14.7 /
      18.5 / 14.9 s now (fork 9-17); the first agent's turn ends 3 to 5 s later (its 200 tokens now
      at the floor), so the two turns' ends together are even to worse: for replies this short
      the old half was as good. Whether the floor should rise for a request near its end (its
      tokens wanted are known) is open. Sharing a piece between two prompts would delay the first
      without bringing the second forward.
- [ ] Tool calls: a generic format for other templates; recovery of imperfect calls (T05);
      string constraints `pattern`, `minLength`, `maxLength`, `format`; forced tokens as
      drafts; a guard for `ignore_eos` with a complete grammar; the end-token rule for a
      schema with thinking; llama.cpp's `json.gbnf` costs 6.8 ms per mask.
- [ ] Responses: stored responses (A05), WebSocket (A06); the chat route's sampler fields
      are refused there (only `temperature` and `top_p` are served).
- [ ] Router: a plan cannot be asked without loading.
- [ ] A VRAM reserve for other programs (V04), container image and service unit (X09), key roles
      and a reload of the key file (X02), TLS (X03).

### Input beyond text
Image input (F11) is built, pieces 0 to 8 (2026-11-15 to 2026-11-22; [`dev/history.md`](dev/history.md)); what
is left of it to speed up is in "Store of levers".
- [ ] Fetching remote image URLs (backlog; refused with a 400 today).
- [ ] Documents as input (PDF, text), as Basalt takes them next to images; video stays backlog.
- [ ] Audio, video and Gemma 4 are backlog: none of them blocks the first release (the owner,
      2026-11-16).

### Inkling 2 Mini (`inkling2`; the owner, 2026-11-22: worth supporting, and looped transformers matter more and more)
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
      vision tower. In the order (the owner, 2026-11-23): after the 27B and Flash-Next are at least
      as fast as the fork and Strata in everything, one piece at a time.

### Jev mode (`feat/jev`, 2026-11-25)
- [ ] Forks as rows of one step whatever `--parallel` is: a fork today is a sequence with a slot
      of its own (150 MiB of recurrent state on the 27B) that restores the parent's checkpoint
      from RAM, so with the owner's `--parallel 2` the questions of a decision run two at a time
      (1.3 to 2.1 s a decision of 8 questions on the 27B, 2026-11-26,
      [`dev/experiments/decisions.md`](dev/experiments/decisions.md)).
      Flide's way is a fork that reads the parent's state and writes none: a segment flag for the
      delta-rule and convolution kernels (state read from the parent's slot), page tables and
      exact entries for fork slots that have no recurrent state, the MTP block and DFlash
      skipping fork rows, and the conductor seating forks in those slots. It touches both models'
      device parts, the kernels and the executor's tables.
- [ ] Images in a decision (Flide puts a request's `images` ahead of the text of the user turn,
      for a model that reads them): refused with Flide's 422 today.
- [ ] A decision model's own calibration (Flide's `template` format: temperatures, label sets,
      readout heads in the model's files) and the formats of native decision models (OneJev,
      Open-Jev and the rest): only chat models are asked, as Flide's `next_token` asks them.
- [ ] Flide's `--batch-invariant` (the same bits whatever a request is batched with): a fork's
      logits depend on the size of the steps its rows run in, as every prompt's do ("Rounding"
      in [`dev/architecture/sequence-state.md`](dev/architecture/sequence-state.md)).

### Vulkan (last)
- [ ] A backend-free crate and the `ares-gpu` facade; expert and Flash-Next kernels; both
      models on Vulkan in every placement; attention for wide windows (2 to 6.5 times slower
      than CUDA today); the delta rule's f16 state rounded per row; the sampler's `-0.0`; the
      image tower on Vulkan.
- [ ] Untested GPU classes go to Vulkan (the owner, 2026-11-23; was "Turing. Later."). CUDA builds
      and opens only the tested classes, Ampere (8.6) and Ada (8.9) (`cuda/tested-archs`); Turing,
      Pascal, Hopper, data-centre Blackwell, Jetson and every other GPU are to run through the
      Vulkan backend once it runs models. What was not tuned on CUDA goes there too.
      **Turing on CUDA after all** (the owner, 2026-11-26): after Blackwell's tests on the RTX 5060 Ti.
      Someone interested may give access to a Turing card for the tests. Turing (7.5) lacks
      several things the kernels use:
      - `cp.async` (13 kernel files);
      - `mma` m16n8k16 (9 files); Turing has m16n8k8 for f16 and m8n8k16 for int8;
      - bf16 in hardware (5 files);
      - more than 64 KiB of shared memory per block.

      llama.cpp's `mma.cuh` has Turing's paths (MIT). Its cards have 8 to 11 GiB, the T4 16 and
      the Titan RTX 24.
- [ ] Left of cleanup phase 1, at Vulkan parity: Vulkan's draft carry
      (`SamplerKernels::{shift,keep,commit}_hidden`, shaders `draft_{shift,keep,commit}.comp`),
      which CUDA no longer has (no caller there, removed on `cleanup/cuda`), and its test: drop it
      or give both a caller at parity; its delta-rule shape sweep (24 shaders); its copies of the
      activation layout (`vulkan/src/layers.rs`) and of the quantization (`vulkan/tests/mmvq.rs`):
      use `ares_quant::act` there.

### Cleanup, what phases 1 and 2 left (the owner, 2026-11-23: the code must be clean although agents wrote it)
The audit, its reports and the plan: `/var/tmp/ares-audit/` (`PLAN.md`); what was done:
[`dev/history.md`](dev/history.md), 2026-11-24.
- [ ] CUDA (moved to phase 2, not done): one `time_best` for the five timing loops at load (they
      decide measured choices, so a merge is a change of measurements); the names
      (`Eng*`/`EngMm*`/`Er*`/`AresCodes*` decoders, `q4` for qwen4exp beside the Q4 types, tile
      names `_n1` for 8 rows); the splits of `experts.rs` (1937 lines), `kernels.rs` (1196),
      `mm.cuh` (1100) and `mmr_codes.cuh` (1006).
- [ ] Runtime: the running means of the store (their weights differ: one would change what the
      store decides early on); the transit sizing that lives in the model (`plan_transit`);
      `write_record` (290 lines) and `step` (335) of the conductor in smaller parts.
- [ ] Model: Flash-Next's embedding staging written twice (`part.rs`, `part/pass.rs`), the profile
      marks of both parts, `Balance` beside `Meeting`, the record's token offset read by hand (an
      `ares-cuda` accessor), the large functions left (`enqueue_pass` 233 lines, Flash-Next's
      `Part::assemble` 233, `Config::from_gguf` 193, `Drafter::pass_layers` 203; `Frame::widest`
      and the 27B's `open_balanced` went down to 277 and 222 lines in stages) and the names of the
      audit's finding 13 (`Spread::Layer`/`Layers`, `MtpBytes` against `MtpReport`, `Rows` twice);
      `roles::Runner` stays the trial's own timing loop (no store, every expert in zeros).
- [ ] The other crates' part of the audit (`server.md`, finding 7): typed tools and tool calls in
      `ares-api` instead of raw OpenAI JSON across the API, one JSON value type in the server
      (three today: `ares_api::Json`, `ares_tokenizer::Json`, `serde_json::Value`), with the rule
      for integers above `i64::MAX` decided once.

### LoRA adapters, and a scheduler that picks one per task (the owner, 2026-11-26: far back, an experiment and a new feature, after the important work is done)
The owner's idea:
- **Adapters per task.** Each adapter is trained by RL for one task and harness, and carries a tag.
- **A decision before each request.** The same model, in Jev mode, decides which adapter fits.
  Ares loads that adapter and computes only what the adapter changes, where it can; the task
  goes on with it. That gives many fine-tunes, each chosen by its task.
- **A learned scheduler** instead of Jev mode is the other way.
- **Training stays outside Ares.** Unsloth or another existing trainer does it; Ares gets a script
  or a pipe that drives it, not a trainer of its own.

The work:
- [ ] **LoRA adapters.** GGUF adapters as llama.cpp loads them (`--lora`, the request's `lora`
      field, `/lora-adapters`). Several adapters in one step, by their rows (the batched kernels of
      Punica or S-LoRA). For the 27B and Flash-Next (parity). The survey's row:
      [`dev/studies/feature-checklist.md`](dev/studies/feature-checklist.md) M10.
- [ ] **What a switch must compute again.** An adapter that acts on every token changes K, V and
      the recurrent state of the whole prompt, so a switch means a new prefill.
      **Decided by the owner (2026-11-26): adapters on the upper layers only.** A switch computes
      those layers again for the prompt, and keeps the layers below; Flash-Next's pass already
      reads a prompt layer by layer. The owner's reasons:
      - Unsloth trains adapters that way too.
      - The adapters must be trainable on the owner's own cards, and for Flash-Next only this way
        fits. The backward pass stops at the lowest adapted layer, so the layers below keep no
        activations for it.

      Not taken:
      - IBM's Activated LoRA (arXiv 2504.12397): adapters that act only from a point in the
        prompt on.
      - A decision for a whole task, with a prompt cache keyed by adapter.
- [ ] **The scheduler.**
      - First a decision in Jev mode, rendered after the request's own prompt as a fork. It then
        shares its prefix with the request and costs only the rows of its question.
      - Later a learned one: a probe on the hidden state of the prompt's last row, which is
        computed anyway. It is trained on the RL runs' results and on the decisions made so far.
- [ ] **The RL pipe.**
      - Ares serves the rollouts (generation is most of an RL step) and loads each new adapter
        while it runs.
      - The trainer (Unsloth, TRL) takes the gradient steps, and the harness gives the reward.
      - The adapters are trained on the owner's own cards, with the upper layers only (above).

## Store of levers
Optimisations, taken by the owner's reasons, not as a phase; measured on a quiet machine against
the fork and Strata, the largest first. Grouped by model and area.

### Experts from the SSD
Flash-Next's expert tiers, the drive and the link. The runs: [`dev/experiments/against-strata.md`](dev/experiments/against-strata.md),
[`dev/experiments/flash-next.md`](dev/experiments/flash-next.md); the engines to read first (Strata-HotRod,
TokenWing): [`dev/studies/engines-survey.md`](dev/studies/engines-survey.md).
- [ ] Prompts are bound by the link: a wide step uploads every expert it picks that VRAM does not
      hold, once per step (2026-11-22, UD-Q4_K_XL: the 32k prompt uploaded 428 GB in 96 s at 4.46
      GB/s, about 0.7 ms an expert). The passes (2026-11-23) take each expert once per pass; left:
      a CPU share of the experts a wide step picks for few rows (Strata's prefill does this up to
      3072-token chunks), less scratch per row (0.67 MiB). With the MTP block's 675 MiB of resident
      experts the 16 GB card cannot lend 8192 rows even at the loan's floor (without the block it
      can).
- [ ] The CPU takes no part in steps of more than 2048 rows; the per-row buffers are sized for the
      widest step for good, which bounds `--batch-size` and costs decode its slots: about 40 KiB a
      row of VRAM (the step's staged and decoded rows, plans, predictions) and 28 KiB of page-locked
      RAM; the parcels of a split by layers (80 KiB a row of VRAM on the receiving card and as much
      page-locked RAM); the step's inputs in page-locked RAM (about 20 KiB a row a part). Files:
      `qwen4exp/device/runtime.rs`, `runtime/plan.rs`, `part.rs` (`inbox_host`).
- [ ] Decode waits for the drive: 2-7 % of the picks come from the file (17-25 reads a token
      at 3.3-4.6 ms each, 27-90 MB a token), and the layers waited 64 ms a token of 77 on
      German prose. The read-ahead of predictions was used 50-63 % where it read, but few
      places qualify: of the experts in no tier a prediction named one layer ahead, the first
      places were picked 64, 58, 48, 39 % of the time (one row) and 55-61 % (windows of
      drafts), two layers ahead 4-15 %; and with experts of 3 MiB only one was read ahead at
      a time (two since a7c0375). A drive this slow at one or two reads (1.5 ms an expert
      alone, 3.7 in decode beside the others) and a lead of one layer (~1.5 ms) hide little;
      more RAM (see the page cache under "Measurements owed") or pruning (opt-in) remove misses.
- [ ] An expert from the SSD is three reads (its matrices lie apart in the file): a missed expert
      waits 2.3 to 3.2 ms for the file where the drive needs 1.4 ms. Measured 2026-11-16
      (`qwen4exp_run --decode-at`, its `drive:` line; two cards `--ram-gib 8`, load 7 to 12, the
      drive shared): a demand read of one matrix took 2.0 to 2.1 ms at the median from its
      submission to its completion (8.9 to 9.4 at the 99th percentile), an expert's three reads
      2.8 to 3.6 ms on the mean, the miss from the record to its slot 2.3 to 3.2 ms, one from RAM
      0.1 ms (the upload); on one card (8.6 reads a step) a matrix took 2.7 ms at the median and
      28.6 ms at the 99th percentile. So the rest is in the reads (three requests, several misses
      of a step and other users sharing the drive), not in the store. The lever: one read per
      expert (a layout of the SSD level).
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
- [ ] A start on two cards that is much faster than the warm-up: reads of several neighbouring
      experts at once into a buffer (the sorted single reads reach 1.25 GiB/s where random ones
      reach 0.7 to 0.8; the gain of larger ones could not be measured on the shared drive), the
      profile's fill in large ordered reads, or answering once VRAM is filled and filling the RAM
      tier behind it until the first request (which then misses what RAM would have held).
- [ ] A layer's resident picks computed while its misses are still being read. ProMoE calls
      this "reordered inference"; LES (OpenBMB, 2026-10) states 1.94x together with its reads
      ahead, on a phone; see [`dev/studies/moe-offload-survey.md`](dev/studies/moe-offload-survey.md).
      Not checked yet: whether a decode layer's resident experts run before a read from the
      file arrives (experts.md: "the layer waits for it"). The sums must not depend on the
      order (kernels.md), so the late experts would add into a buffer of their own. Here a read
      (1.4 to 4 ms) is far longer than the resident experts' work: measure what it hides first.
      The same source repacks the experts so that each lies whole in its file. In GGUF, gate,
      up and down are tensors of their own, so an expert takes three reads. That would be a file
      of Ares's own, beside the neighbouring reads above.
- [ ] The slots per layer from the profile when the store is planned (`StorePlan::new` takes
      a heat; the runtime passes none, and the slots move between layers by need as it runs).
      Tried 2026-11-22 (`flash/ssd-stream`, reverted): VRAM by each layer's hottest and RAM by
      the heat beyond them, from the profile the warm-up writes; on UD-Q4_K_XL with 40 GiB
      decode then read 1.5 to 2 times as much from the file a token (87 against 145 MB on
      German prose): a short text's profile names too few experts per layer to divide the
      slots by. The layers do differ (95 % of a layer's picks take 166 to 345 experts), and RAM's
      slots never move between layers while the engine runs: a rebalance of RAM's slots by the
      heat as it runs, as VRAM's every 256 steps, is the form to try.
- [ ] Where VRAM holds fewer slots than a step picks, every layer's slots are at its floor, so
      the heat no longer moves slots between layers; the floors in RAM cost `--ram`: UD-Q4_K_XL
      needs 9G on one card and 10G with both cards on every layer (8G by layers).
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
- [ ] Passes, what is left: a pause inside a layer's experts (a request waits for at most one
      layer's experts); read-ahead deeper than the next layer (today the next layer's experts that
      no tier holds go into staging and its RAM-held guesses into transit slots); the lent wide
      steps are still there for tile-mean K/V and DFlash, which passes do not take.
- [ ] A cached prompt's replayed rows and its computed rows are steps of their own: two passes
      over the experts where one would do (a prompt step is still cut where a checkpoint is due,
      except inside a pass since 3af7b602); with the CPU's share of small steps cheaper, not built.
- [ ] Flash-Next with 3 or 4 requests at once stays at 32 to 35 tokens/s in sum with `--parallel
      2` and with `--parallel 4` (two give 44 to 54, one 19 to 40; IQ2_XS, both cards, `--ram 8G`,
      40960 cells): more sequences pick more experts that RAM and the SSD bring.
- [ ] VRAM on loan to the expert tier: the MTP block's 675 MiB while a long prompt runs (gap
      study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).
- [ ] CPU path for the types without a written kernel (Q2_K, Q3_K, Q6_K, IQ2_XS, IQ3_S, IQ1_S,
      IQ4_NL and Q4_0 to Q8_0 as gate and up, the 256-value types as down, TQ*, MXFP4, NVFP4,
      Q1_0, the floats) is 3 to 8 times slower than the written ones; none is in a Flash-Next
      file at hand.

### Speculation on Flash-Next
The MTP block of Flash-Next, measured: [`dev/experiments/speculation.md`](dev/experiments/speculation.md).
- [ ] A window's price by what its rows read: on one card a round with drafts reads 100 to
      260 experts from the file against 45 to 100 for a row, and what those reads push out of
      the tiers and hold up for the next steps is charged to no step, so the conductor keeps
      choosing windows that cost more than they bring. A price per row by the tier its
      experts come from (the study's `dense(rows) + chain(links) + misses x cost per tier`),
      or the block not resident where VRAM holds a third of the experts, are the leads. With the
      shared MTP driver (links of 1.4 ms, 0.9 with the draft vocabulary, where they took 1.7 to 2.0
      with the host's wait) Flash-Next answered at 39.0 tokens/s on the mean of twelve requests
      against 41.1 before (2026-11-17, alternating, loads 2 to 6; requests from 31 to 59): the
      conductor verifies more of the cheaper drafts at a lower rate (the first link's 54 to 58 %
      against 63 to 71 %).
- [ ] Drafts in the measuring tool (`qwen4exp_run` has none; the server and the runtime's
      tests measure them; the tools stay on `Session`, the owner 2026-11-25).

### K and V beyond VRAM
Built for Flash-Next (2026-11-22 to 2026-11-24, `kv/overflow`, the named `--context`; "K and V
beyond VRAM" in `dev/architecture/flash-next.md`).
- [ ] Steps of the parts' own scratch read the far entries in place, row by row; only steps on
      lent VRAM and passes (in their loan) read a copy. Since the paused pass (930d0be7) a long
      prompt beside generation reads the copy; in-place pieces remain for prompts beside other
      prompts, short remainders and models without passes. A copy in the own scratch would cost
      VRAM for good (the far pool's share of one layer, 254 MiB at 262 144 cells).
- [ ] The RAM level of the prompt cache copies a far page when it leaves the devices; it could
      take over the page's memory instead.
- [ ] Pages are near or far by their number: the newest pages of a long sequence are the far
      ones, so its newest cells (often picked by the indexer) are read over the link. Keeping the
      newest pages in VRAM needs pages that move between the pools.

### Host memory
- [ ] A release that does not wait for the running step, so the headroom stays at a few
      hundred MiB (it sits at its cap, a sixteenth of the machine's RAM, 1.95 GiB, after every
      load); it needs memory the device does not read during a step. Since 2026-11-24 page-locked
      memory changes only while no device runs a job (`dev/architecture/host-memory.md`).
- [ ] Locking a page's buffers costs about 0.5 ms per MB when RAM grows (a page of the 27B is
      4.6 MB): lock ahead, between steps (page-locked memory may not change during one since
      2026-11-24), for example while the engine is idle.

### Weight types
Every tensor of both models takes every weight type the shape allows (`moe/q4-load`, 2026-11-17;
`dev/architecture/flash-next.md` "Files in any type").
- [ ] A router of a type other than F32 is decoded to F32 at load, four bytes a weight (files
      keep it F32); its kernels (decode, prediction, wide steps) would take the type.
- [ ] Shapes, not types: rows whose width a type's block does not divide cannot be stored in it
      (Flash-Next's expert down projections of 640 values, the n-gram table's rows of 160, the
      image tower's `ffn_down` of 4304); the 27B's tensor split cuts in units of 256.

### Kernels
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
- [ ] Attention over long contexts: llama.cpp PR 32407 (merged 2026-11-17) rotates the order in
      which blocks read K/V tiles so that blocks sharing K/V hit L2 together: prompts of the 27B
      on an RTX 5090 at 131072 cells 1290 -> 1822 tokens/s (1.41x), 1.25x at 65536, 1.07x at
      32768, nothing below. The new prompt kernel is bound by its instructions, not by K/V traffic
      (about 50 GB/s of the RTX 3060's 360 at 98k), so the order of reads is not its lever; wide
      windows (drafts) still run on the decode shapes.
- [ ] The fused read of a Q8_0 mixer (UD-Q4_K_XL) takes 43.8 us on the RTX 4070 where BF16 takes
      24.9: it decodes value by value (a load of four codes at once would cut it); sixteen rare
      types read through the decoder chosen at run time; the float read is 2 % slower than before
      the types (5400 against 5112 instructions).
- [ ] The F32 tile kernel: 1.4 times BF16 at 4096 rows and 2.1 times at 16 rows (its large shape
      uses all 128 registers; `dev/architecture/kernels.md`).
- [ ] NVFP4 many-row gate and up 1.4 times its neighbours on the RTX 4070; IQ2_XXS many-row
      at 8 rows 7 % slower on the RTX 3060 since the kernels for all types.
- [ ] The many-row down kernel's branch of the ordered sum, which the host no longer takes:
      without it nvcc gives ErQ20's kernel of 16 pairs 64 registers instead of 70 and it was 5 %
      slower (`kernels/expertrows_down.cuh`), so it stays until a layout without it is measured
      (the pass over all kernels).
- [ ] Steps that overlap the host's scheduling with the GPU (vLLM's default now), CUDA graphs:
      first measure the gaps between kernels and the host waits (grammar mask, draft chain) (gap
      study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).

### The 27B and its drafts
- [ ] The draft chain's cut costs a wait of the host per link where a bar can end it (the
      device thread reads the link's picks before it enqueues the next); since 2026-11-16 the
      chain is a step of its own and the round verifies only what pays of it, so the cut
      saves links only. A cut on the device (the links of a chain as kernels that end
      early) would keep the saving without the waits.
- [ ] Layout choice that is stable: it rests on the fastest of three probe rounds; a link of
      the draft chain costs 1.49 ms where 1.44 ms is possible; the choice changes between
      loads and with the MTP block; the 0.8B decodes 5 % slower than it could.
- [ ] Two generating sequences share a step's draft rows by their chance (`window::choose`) up to
      the 8-row edge of the price table (8 rows 31.0 ms, 9 rows 37.1, 16 rows 47.8): the step stays
      at 32-35 ms (alone 32), one request gets 1.6 tokens a step and the other 2.2 to 3.5. The fork's
      step grows from 33.6 to 40-44 ms and keeps each slot's drafts (2.0-2.7 tokens a step each).
      Ares' slower request is as fast as the fork's slower one and the other is faster (quiet machine:
      49 + 67, 52 + 68, 85 + 58 against 49 + 50, 52 + 55, 57 + 59 tokens/s), so this is not behind;
      a cheaper 9-to-16-row step would lift the slower one.
- [ ] The model's drafts for the thinking rows of a request that has to call a tool (a reply that
      must reach a rule takes none while it thinks); drafts while a rule is bound (only the
      n-gram drafter's today).
- [ ] Sampler: deciding a level got slower (35 to 42 us); a list of the set's tokens after the
      first cut; on two cards most of the sampler's time on the RTX 3060 is the exchange.

### Drafters
- [ ] The DFlash ring in the prefix cache's checkpoints (42 MiB a slot in f16): a sequence that
      resumes from the cache drafts from the resume point's context only until its ring fills.
- [ ] Sampled DFlash drafts: the walk is greedy; keyed noise in the scores (as the MTP block's
      picks have) would raise sampled acceptance (study: 2 to 8 %).
- [ ] DFlash's drafts-only step (4 to 5 ms).
- [ ] The lookup's corpus kept over a restart; the lookup's confidence calibrated on German text
      (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).

### Checkpoints
- [ ] Checkpoints of the recurrent state in int8 (SGLang's `mamba_checkpoint_pool.py`): a quarter
      of Flash-Next's 112 MiB, which matters because a full tier leaves one place; after a KL
      measurement (gap study `/var/tmp/ares-survey/gap-study-2026-11-17.md`).

### Images
- [ ] The row kernels between the tower's products take about a third of the time outside
      the attention (fusing the residual with the norm after it, a bias in the products'
      epilogue).
- [ ] Which device computes a block of the tower faster is not measured (the server picks it from
      reported facts); the share counts a piece with a tower job as one piece.
- [ ] The MTP block reads an image cell as the pad token's embedding (llama.cpp gives the
      draft context the image's rows), DFlash takes cells as positions; drafts after an image
      may be accepted less often, the output is the main model's.
- [ ] The store of image rows is not kept on SSD (after a restart a resumed conversation runs
      the tower once per image).
