# Prefetching experts from the SSD

A study made on 2026-11-27, for the owner's word of that day (translated): "for SSD streaming
we have plenty of references, TokenWing or Strata-HotRod for example, and our own experimenting;
I think SSD streaming is well doable with extreme prefetching." It asks what the references do
in their code, what Ares's own read-ahead lacks, how prefetching should spend the drive's spare
time, which predictors to compare, and in what order to build. A section on stand-in experts follows
the owner's two decisions of the same day: they are an opt-in rule of `--expert-prune`, and the
rule is cost-aware.

Read, in the code, each at the commit named and under the licence checked there:

- **Strata-HotRod** (https://github.com/sangharshadhyeta/Strata-HotRod) at fc1e6d53. MIT,
  "Niko1221 and the Strata contributors".
- **TokenWing** (https://github.com/runmyai/tokenwing) at 23976328. MIT, "The ggml authors" and
  "Harin Mani Karri"; the TokenWing files carry no headers of their own, and the top-level licence
  covers them.
- **Strata** (https://github.com/niko1221/strata) at c2347740. MIT.
- **The papers** as noted in [`moe-offload-survey.md`](moe-offload-survey.md): FATE, ProMoE and
  Pre-gated MoE.

The text in those repositories was read as data. "Stated" marks a number of the named source
that was not checked. Nothing ran on a GPU or on the test machine. One measurement ran on the
host: the precision of the predictors that need only recorded routes, from the routing traces of
2026-11-10. See "Measured offline" below.

## Where Ares stands

From [`ssd-decode-profile.md`](../experiments/ssd-decode-profile.md) and the comment of
2026-11-27 on #16. The setup: Flash-Next UD-Q4_K_XL (104 GiB, an expert 3.2 MB) on the RTX 4060
Ti 16 GB, with 3120 VRAM slots (65 a layer). The RAM tier is 4800 slots at `--ram 16G` and
13 056 at `--ram 40G`.

| Decode step, one sequence | `--ram 40G` | `--ram 16G` |
| --- | ---: | ---: |
| Step, median | 67.5 ms | 158.6 ms |
| Device's own work | 33 ms | 33 ms |
| Waits for the CPU's sums | 11.9 ms | 3.8 ms |
| Waits for the file | 25.2 ms (6.9 layers, 3.6 ms each) | 142.5 ms (28.5 layers, 5.0 ms each) |
| Reads of the file a step | 8.6 (27.3 MB) | 61.8 (195 MB) |
| With a core free for the I/O thread (12 or 10 workers) | 61.9 ms, file 15.4 to 17.2 | 121 to 125 ms, file 101 to 108 |
| Named by the predictions and in no tier, a step | 40, of which 11 % picked | 151, of which 23 % picked |

The drive gives 1.53 GB/s with one expert in flight (2.0 ms for one), 1.99 GB/s with two
(3.05 ms each) and 1.98 GB/s with four.

Ares's read-ahead was built on 2026-11-16 (`moe/predict`) and is described in
[`experts.md`](../architecture/experts.md), "Reading experts ahead of their router". It has these
parts:

- **On the device** (`device/part/predict.rs`, `Part::enqueue_prediction`). In a decode step of
  up to eight rows, right after the router of layer L, the routers of L+1 and L+2 are applied to
  the rows the router of L read. The input is layer L's own expert input. It does not go through
  the target layer's stream mixer, and it leaves out the shared expert's output. The cost is two
  F32 router products a layer, 1.7 ms a step (`--predict 0` takes it off the dense part).
- **On the host** (`expert_store/ahead.rs`, `ExpertStore::ahead`). The lists go to the store
  behind each layer's route record. An expert that no tier holds is queued as a guess. Its read
  starts only at a class (distance, place in the list, one row or several) whose named experts
  have been picked at least half the time (`Odds::worth`). It goes into room nobody uses
  (`Want::Spare`). The record cancels what it did not pick and promotes what it did.
- **Bounds.** Experts on their way ahead are limited to `Reader::prefetch_room` over the expert's
  bytes plus one. With the reader's four chunks of 512 KiB and an expert of 3.2 MB, that is
  2 lanes. A step may read ahead at most the drive's measured pace times the step's mean time,
  minus the step's mean demand bytes (`ahead_end_step`). That is already the budget this study
  argues for.
- **The reader** (`ares-io`, `sched.rs`, `Drive::has_room`). Every prefetch is kept behind every
  demand read. A drive takes four prefetch chunks, or two while a demand chunk is there.

What the profile found:

- The half-picked rule lets almost nothing through: 0.1 to 0.3 reads ahead a step at 40G, 2.5
  at 16G.
- A prefetch waits 170 to 250 ms at the reader. Part of the cause is the I/O thread waiting
  200 µs for a core at each wake-up, which agent `iocore` is fixing.
- With `--predict 0` the step is the same and 1.7 ms of dense work goes away.

So the plumbing exists. What is missing is a predictor precise enough on the experts the tiers do
not hold, enough lead, and a reader path that serves prefetches in time.

## The references, in their code

### Strata-HotRod

**Prediction.** HotRod has a learned, low-rank correction on the routers' look-ahead, in
`src/kernels/cuda/native_router.cu`, lines 1867-2904.

- **The form.** For every source layer s and distance d in `kLaDist[] = {1, 2, 4}`, the guess
  for layer s+d is `W_router[s+d] · x_s + B (A · x_s)`.
  - x_s is layer s's MoE input, the same input Ares uses.
  - Four slots compete for each pair: the plain router, and corrections of rank 4, 8 and 16.
  - A starts at noise of 1e-3 and B at zero, so a slot starts as the plain router.
- **Training.**
  - It trains online while serving, on decode windows of up to eight rows, on a side stream,
    with AdamW (`kLaLr = 1e-3`).
  - The target is the real logits of layer s+d in the same window. The loss is a centred MSE over
    the true top 32 plus a hinge on the rest (`la_b_grad_k`).
  - A slot takes the lead when the Wilson 90 % lower bound of its advantage is above 0 after 200
    picks (`la_decide_one`). The others freeze after 2000 picks.
- **State and cost.**
  - The state is saved per model in `data/predictor-<fp>.bin`, fp16, a few MB.
  - In VRAM it takes about 171 MB, with the Adam state.
  - Every window it computes all three ranks for 137 (s, d) pairs, about 12 M multiply-adds a
    token and 47 MB of F32 reads.
  - Stated, `docs/SPEC-adaptive-la.md` (8 Nov): 32.3 tokens/s with the learner, 34.0 without.
- **Its numbers.**
  - Stated (README, `docs/HOTROD-DETAILS.md`): of the needed experts not on the card, the learned
    correction finds 85 to 88 % at d = 1, 76 to 79 % at d = 2 and 66 to 71 % at d = 4. The plain
    router finds 77 to 79 / 66 to 69 / 55 to 58 %.
  - These are counters (`la_b_score_k`). A pick counts as found when it is among the 6 best
    non-resident guesses, for picks of rank below 3 only. So they are a recall on the strong
    picks, not the share of reads that is used.
  - No logs of them are in the repository.

**Prefetch.** At each layer's dispatch, `expert_pool_dispatch_multi` (`src/core/expert_source.cpp`,
lines 6887-6976) queues reads for l+1, l+2 and l+4, in that order.

- **Which reads.** Rank 0 of every row first, then rank 1, and so on. Experts on the card are
  skipped. Ranks 0 to 2 are always read. The other ranks are read only while the I/O workers have
  room. `pred_tail_read_warp` drops beforehand every guess a stand-in would serve.
- **The I/O threads.** Two by default (`set_io_prefetch`; 8 with `STRATA_IO_PF_THREADS=auto`).
  They take the smallest target layer first (`STRATA_IO_EDF`, on). Guesses from an earlier window
  or for a layer already routed are dropped.
- **Limits.** Demand reads do not go through these threads by default; the decode thread reads
  its misses itself. A read in flight is never cancelled. There is no byte budget a token.
- **Where the bytes land.** Guesses go into pageable stage buffers, or straight into a free or
  cold RAM-tier place (`promote_read`, with the RAM overlay).
- **PCIe balance.** The RAM tier is page-locked in place (`STRATA_PIN_SEGS`). `PcieModel`
  balances each layer's misses between uploads from it and the CPU, by two least-squares fits of
  the two sides' times.
- **Stated numbers.**
  - 29 % of what is read ahead is used (`docs/PREDICTOR-EXPERIMENTS.md`, 24 Oct: about 15 500
    read ahead, 4 500 used). `docs/perf-ideas.md` says 9 to 18 %.
  - The I/O prefetch took decode from 17.2 to 27.2 tokens/s (README).

**Stand-ins**: see the section below.

**What Ares has of it:**

- The routers of later layers on the same input.
- Next layer first: guesses are queued per layer and dropped at the record.
- Cancelling at the record, which HotRod does not do for reads in flight.
- No learned correction, no d = 4, and no PCIe model. Ares's `CostModel` splits misses between
  the CPU and uploads by measured costs.

### TokenWing

The mechanisms are mostly off in the engine's defaults. The "Run My AI" command lines of
`docs/getting-started.md` turn them on. Its numbers are for Ling 3.0 Flash. TokenWing runs
`qwen4exp`, but it has no preset and no number for it.

- **The guess input** (`src/models/qwen4exp.cpp`, lines 656-689; `--moe-stream-pred-shared`).
  For a target layer X, the guess reads what X's router would read if layer L's routed experts
  added nothing.
  - It takes the wide residual after L's attention, plus L's gated shared expert, combined as L's
    write does (`build_hc_combine`).
  - It then mixes that with X's own FFN stream mixer (`build_hc_mix` with `hc_ffn_*` of X), and
    applies X's router.
  - The routers of several distances run as one product (`--moe-stream-guess-combined`).
  - Stated: adding the shared expert cut misses by 11 % (`docs/how-it-works.md`).
- **The learned correction** (`--moe-stream-correction`; `tokenwing-tools/correction/`, 952
  lines of Python; `src/llama-moe-correction.cpp`).
  - A ridge regression from the guess input to the real logits, per target and lead, fitted
    offline from recordings. It is folded into the guess router, `W' = W_X + Bᵀ`, so it costs
    nothing at run time beyond the router copy.
  - A pair is written only if, on held-out recordings, it finds at least as many of the needed
    experts as the plain guess. The needed experts are those picked and in no tier. The check
    runs for every kind of text (maths, code, chat, JSON, story).
  - Stated (`src/llama-moe-stream.h`): lead 2 with the correction finds 0.760, the plain lead 1
    0.757.
- **I/O.** `io_loop_t` in `src/llama-moe-stream.cpp`, lines 6014-6499.
  - Eight reader threads, each with a synchronous O_DIRECT `preadv` into the RAM slot. There is
    no io_uring and no staging.
  - The deques are served in a fixed order: demand, then guesses that have a slot, then the
    waiting list (late binding: a guess takes a victim slot only when its read starts).
  - At most 2 guess pieces of 512 KiB run while a demand read is in service, and 4 otherwise.
  - When the target layer starts, guesses it did not pick are dropped or cancelled, and picked
    ones are promoted.
  - A guess that a nearer guess no longer names is moved back (`la_back`).
  - Far guesses (d ≥ 2) add at most one new entry per target (`--moe-stream-guess-pace 1`), so
    most reads come from one layer ahead.
  - There is no byte budget a token.
- **Caches.**
  - The RAM slots are equal per layer. Eviction takes the lowest decayed use count (half-life 64
    words), and a landed guess nobody used is worth 0.
  - A VRAM cache follows the text, up to 64 swaps every 8 reply words.
- **Measurement in the engine.**
  - `--moe-stream-pred-recall` counts recall at 8.
  - `--moe-stream-record` writes the router logits of leads 1 to 4, for `check_correction.py`.
  - The stated 75.7 / 67.2 / 61.1 % (one, two, three layers ahead, Ling) have no script that
    reproduces them.
- **Drafts and previous tokens.** No prefetch from draft tokens, and none from the previous
  token's routes. Stated (`docs/design-decisions.md`): consecutive tokens share about a third of a
  layer's experts.

**What Ares has of it:**

- Demand first, prefetches cut into 512 KiB pieces, two of them beside a demand read, cancel and
  promote at the record.
- Late binding (`Want::Spare` takes its room only when the read starts).
- An unused guess is worth nothing.
- Not the guess input through the target's mixer with the shared expert, and not the correction.

### Strata

Strata has **no prefetch of experts from draft tokens.** What draft tokens drive is the reads of
the n-gram table's rows:

- `prefetch_ple` in `engine/src/core/mtp.cpp`, lines 1920-2025, runs for every draft as soon as
  the host sees it.
- `engine/src/core/verify.cpp`, lines 3851-3860, runs for every token of a window before its
  launch.
- Rejected drafts are not cancelled.
- The MTP block's experts stay in VRAM and its routes are not exported.

For experts, Strata reads ahead from later layers' routers in three ways:

- **`RouterLookahead`** (`engine/src/core/expert_source.cpp`, lines 3532-3820).
  - A host thread holds BF16 copies of every router and applies those of L+1 and L+2 to layer L's
    MoE input of up to 8 rows, top 10 per row.
  - The confidence of a guess is `(k - j)/k × 0.5^dj`.
  - An opt-in scheduled admission (`STRATA_IO_PF_SCHED`, `warm_scored`, lines 1246-1277) reads
    a guess only above a bar `0.5 + 0.5 × pressure`. A window has a byte budget of half the
    drive's measured rate times the window's time. Late jobs are dropped, and the scale follows a
    usefulness target of 0.5.
  - A gate closes the read-ahead when it is useful less than 10 % of the time (lines 2275-2377).
- **A prefetch into the L3 cache** (`ExpertPool::prefetch_next`). The CPU workers do it for the
  next layer's top 4 experts in RAM.
- **Decayed counts** that move experts between VRAM and RAM every 4 windows (decay 0.7, gain 1.5,
  96 swaps).

Stated:

- Routing prefetch from the SSD on UD-Q4_K_XL: +14 % (6.9 to 7.9 tokens/s), about half of the
  SSD reads predicted (`docs/UNSLOTH_Q4.md`).
- A router look-ahead on the PCIe share: recall 27 to 34 %, precision about 8 %, "about 12x
  wasted bandwidth", closed (`docs/bench-results/2026-10-04-community-rtx-5070ti-windows`).
- Useless read-ahead "18 GiB RAM tier, 0 % used, -9 to -15 % decode" (a comment at
  `expert_source.cpp:2353`).
- Tail skip at rank 7: 1.29 times, KL 0.0126 (`DETAILS.md`).

**What Ares has of it:**

- The same predictor, on the device instead of a host thread.
- The byte budget per step.
- Not the bar that follows the pressure, not the usefulness gate, and not the deadline drop.
- Ares's tail skip is `--expert-prune skip:<rank>`.

### The papers

- **FATE.** The next layer's gate on the current gate input finds 78.8 % of the picks in decode.
  Fetching every expert above the 75th percentile of the predicted weights finds 97 %. Shallow
  layers are predicted worst and get more cache.
- **ProMoE.** A two-layer MLP per layer, about 2 M parameters trained in 1 to 2 hours, finds
  84.7 % where the next gate finds 66.9 % on one of its models. Each further layer of distance
  costs it about 5 points.
- **Pre-gated MoE** changes the model and is out of scope.

Neither paper measures Flash-Next.

## The drive's budget

A prefetch that its layer uses costs the drive nothing extra: it replaces the demand read the
layer would have made. A prefetch that is not used costs its bytes. Every prefetch also delays a
demand read that arrives while its chunk is at the drive, by up to the two chunks the reader
lets stay beside it: 1 MiB, about 0.5 ms. So the budget limits wasted bytes, not reads:

`spare = pace × step time − demand bytes`, and expected waste `Σ (1 − pᵢ) × bytesᵢ ≤ spare`

Here pᵢ is the measured chance that a guess of its class is picked.

| Per step, at 1.99 GB/s | 40G now | 40G, I/O core fixed | 16G now | 16G, I/O core fixed |
| --- | ---: | ---: | ---: | ---: |
| Step | 67.5 ms | 61.9 ms | 158.6 ms | ~123 ms |
| Drive time of the demand reads | 13.7 ms | 13.7 ms | 98 ms | 98 ms |
| Spare | 107 MB (33 experts) | 96 MB (30) | 121 MB (38) | ~50 MB (16) |

Two regimes follow.

**At 40G the drive has room and the misses are few.** There are 8.6 reads a step against room for
30. The predictor's precision on experts in no tier hardly binds, so the bar can be low (p of
0.1 to 0.2). The limit is lead.

- A layer of the step takes 0.69 ms of device time, and the layers at 40G wait little apart from
  their reads.
- A guess one layer ahead hides about a third of a read of 2.5 ms; two layers hide half.
- A read is hidden only from three to four layers ahead.
- So at 40G: wide and deep. Every named expert in no tier goes up to the spare, at L+1, L+2 and
  L+4.

**At 16G the drive is the bound once the I/O thread has its core.** The demand reads alone are
98 ms of drive time in a step of ~123 ms. The floor of the step is that drive time, unless the
bytes shrink.

- What prefetching can win is overlap. The reads are issued one layer after another at the
  records, so the drive sees one or two experts at a time. A queue kept at two experts reaches
  the drive's full 1.99 GB/s. The prize is therefore at most ~25 ms, from about 123 to about
  98 ms.
- Wasted bytes come straight out of the 16 experts of spare, so the bar must be high (p of 0.35
  to 0.5, as today).
- Lead matters less: at 16G a layer waits 3.5 to 5 ms, so a guess one layer ahead has about a
  read's time.
- So at 16G: narrow and shallow. L+1 and L+2, the best places only, and enough in flight to keep
  two experts at the drive.

## A design: prefetching under the drive's budget

Nothing here changes the output: a prefetch only moves bytes. The fingerprint of
`qwen4exp_run --fingerprint` must stay the same with it and without it, as it did on 2026-11-16.

1. **Candidates and their value.** A candidate comes from a predictor (below) and has:
   - a target layer, with a deadline: the expected time of that layer's record, from the store's
     own record times of the steps before;
   - a class, as `Odds` has it (predictor, distance, place in its list, one row or several), with
     a measured chance p of being picked;
   - its bytes.

   Its value per byte is p × the time it saves (the part of a read its lead can hide) / bytes.
2. **The bar follows the spare** instead of a fixed half.
   - Candidates are taken by value. A candidate starts while the step's expected waste
     `Σ (1 − p) × bytes` is within its spare.
   - Strata's `warm_scored` raises its bar `0.5 + 0.5 × pressure` in the same way.
   - The class counts stay as they are. Every named expert is counted when its record arrives,
     read or not, so a class that is not read keeps its p measured.
3. **Demand first, always.** The reader keeps every demand read ahead of every prefetch, and two
   prefetch chunks beside a demand chunk (`Drive::has_room`).
   - The prefetch depth at the drive should be the depth that gives the drive's rate: two experts
     here.
   - That replaces the bound of lanes from `prefetch_room`, two experts, with one from the
     drive's measured curve (`expert_bench`: 2 in flight give 1.99 GB/s, 4 give no more).
   - More may wait queued in the store, never at the drive.
4. **Order: the earliest deadline first, then value.** This is HotRod's `STRATA_IO_EDF` and
   TokenWing's `la_less_b`.
   - A guess that cannot reach the drive before its deadline (queue bytes over pace, more than
     its lead) is not started.
   - When a nearer prediction for the same layer no longer names a queued guess, the guess goes
     back, as TokenWing's `la_back` does.
5. **Cancelled as soon as the record shows it is not needed.** This is today's rule.
   - The layer's record forgets its unstarted guesses, cancels the reads of experts it did not
     pick, and promotes the reads of those it did.
   - At the end of a step the guesses for layers that did not run are forgotten.
   - A read that completes for nothing stays a guess worth nothing.
6. **Where the bytes land: the RAM tier, as a guess.**
   - Decode on this card serves its misses from RAM: by the CPU, or uploaded from there by the
     cost model. So a prefetch lands where the miss would have been served from. Staging and
     transit are for wide steps.
   - `Want::Spare` takes only room nobody uses: a free slot, an unused guess, or a copy of a VRAM
     expert. It keeps members safe.
   - At 16G that room may run out. Then a small guess pool of the RAM tier's bytes (twice the
     experts the budget allows a step, 32 to 64 slots, 100 to 200 MB at 16G) keeps the
     read-ahead from stopping for want of room.
   - The ledger counts the pool. It costs the tier 0.7 to 1.3 % of its slots.
7. **Off where nothing qualifies.** When no class has qualified for a while, the device skips
   the prediction (lever 5 of #16: 1.7 ms a step). It probes again now and then to keep the
   counts measured.
8. **Depth and width by the regime.** These come from the budget and the counts, not from fixed
   numbers:
   - at 40G the spare lets classes with p of 0.1 to 0.2 in, and the lead favours L+2 and L+4;
   - at 16G only the classes above about 0.4 qualify, at L+1 and L+2.

   The same rules give both.
9. **Drafts and several sequences.** A window of drafts, or a step of several sequences, already
   predicts every row and merges the lists (up to eight rows), with odds of their own ("several
   rows").
   - With drafts, the drafted tokens are known while the chain is drafted, before the verify. A
     table of token to experts (below) can then start reads for the window's first layers.
   - Those layers have no prediction inside the step. At 16G they hold 15 % of the misses in the
     traces.
   - Without drafts, the next token is known only when the step that follows begins, and the lead
     to layer 0 is nothing.

## Predictors

What each would name for layer T before T's router runs. "Precision" means, of the named experts
that no tier holds, the share that was picked: the share of read-ahead bytes that is used.

| Predictor | Input and lead | Cost on the device a step | Precision and recall on Flash-Next | Offline? |
| --- | --- | --- | --- | --- |
| 1. Routers of L+1 and L+2 on L's expert input (today) | after L's router; 1 or 2 layers | 1.7 ms (two F32 routers a layer, 0.5 GB) | UD-Q4_K_XL, profile: 11 % (40G), 23 % (16G). IQ2_XS, 2026-11-12: of picks only on the file, 40 to 45 % named at 10 (L+1), 30 to 36 % (L+2); precision 14 to 22 % (L+1), 7.9 to 12.6 % (L+2) | needs hidden states |
| 1b. The same through the target's FFN stream mixer, plus L's shared expert (TokenWing) | the residual in the middle of L; 1 or 2 layers | the mixer of T per distance: 6.7 MB Q8_0 a layer, +1.1 ms at the card's bandwidth, +3.9 ms with today's mixer kernel (#18) | IQ2_XS, 2026-11-12, without the shared expert: 52 % of the file-only picks named at 10 (L+1); precision 33 to 38 %; places 1 to 6 picked 46 to 85 % of the time (German, one card); all picks 80.7 % at 10 against 69.5 % | needs hidden states |
| 2. HotRod's learned low-rank correction, online | 1, 2, 4 layers | about 12 M multiply-adds and 47 MB a token, a training batch per window; stated 5 % of decode, 171 MB VRAM | stated recall of the strong picks off the card 85 to 88 / 76 to 79 / 66 to 71 % against 77 to 79 / 66 to 69 / 55 to 58; read-ahead used 29 % | needs hidden states and logits |
| 2b. TokenWing's ridge correction folded into a guess router, offline | 1, 2 (4) layers | none beyond a copy of the router per (target, lead): 250 MB in F16 for two leads, or a low-rank B of a few MB | stated on Ling: lead 2 corrected as good as plain lead 1 | needs hidden states and logits |
| 3. A ProMoE-style MLP | 1 layer (2: -5 points) | about 4 MB a layer in F16 (190 MB), one small MLP a layer | stated 84.7 % on other models; none on Flash-Next | needs hidden states |
| 4. The drafts' tokens, the target's routers on the MTP block's hidden rows | the chain's time | small | trees study (2026-11-23, IQ2_XS): 5 to 11 % of a drafted row's picks at 10 (chance 2 %) | measured |
| 4b. The drafts' tokens, a table token → experts per layer | the chain's time; layers 0 to 2 | host only | below: of the misses of layer 0 52 % named at 16 (16G), of layers 1 and 2 36 %; over all layers 30 % of 20 reads used | yes, done |
| 5. The next token from the last tokens' routes (row before, last four) | a step | host | below: 0 to 6 % of the misses: what the last tokens picked is in the tiers | yes, done |
| 5b. A table of the last two tokens → experts (AI-DER) | a step | host | below: 26 to 44 % precision, 3 to 13 % of the misses | yes, done |
| 6. Co-occurrence of routes within a token (BELLS) | after L's router; 1 to 4 layers | host | below: 1 to 5 % of the misses at 40G, 4 to 19 % at 16G, precision 11 to 36 % | yes, done |

**The input is worth more than the predictor.** Two findings point the same way.

- *The study of 2026-11-12 (IQ2_XS).* Applying the target's mixer to the residual (1b) raises
  precision on file-only experts from 14-22 % to 33-38 %. It also lifts the best places above the
  half that Ares's gate asks for.
- *HotRod.* Its correction on the plain input mostly learns what that mixer does.

1b, then 2b for longer lead, is the order. 3 is only worth building if 2b falls short. 2 costs
5 % of decode and its VRAM, where 2b costs nothing at run time.

**What can be measured offline.**

- *`qwen4exp_predict`* (`crates/model/examples/`) exists. `record` needs a GPU: it writes per row
  and layer the residual behind the layer, the output of its first half, the picks, and the tier
  of every expert. `analyze` is host only: it computes predictors 1 and 1b at distances 1 to 3,
  the row before, the same token and the most used, against the picks and the tiers, by place.
  - The recordings of 2026-11-12 (IQ2_XS, 2.8 GiB) are gone. UD-Q4_K_XL needs a new recording on
    the test machine.
  - To compare 2b and 3, `record` must also write the real router logits of every layer.
  - TokenWing's `make_correction.py` is the model for the fit: ridge, folds by recording, and a
    gate per kind of text.
- *The routing traces of 2026-11-10* (`/var/tmp/ares-qwen4exp-trace`, `q4trace` files). They hold
  routes and weights only, from the CPU reference on IQ2_XS. They serve predictors 4b, 5, 5b and 6,
  and a simulation of the tiers. That ran here.
- *The trees probe* (`flash/trees-study`, not merged). It holds routes of draft paths with tier
  snapshots.

## Measured offline: predictors from routes alone

- **The data.** The four traces of 2026-11-10, read in one sequence (long, mixed, German,
  English): 3570 tokens, 304 of them generated, IQ2_XS, CPU reference.
- **The tier model.** A cache per layer with decayed heat (0.9 every 32 tokens). A miss is
  admitted and the coldest member the token did not pick leaves.
- **Capacities.** 337 experts a layer, as at 40G (65 VRAM and 272 RAM), and 165, as at 16G. They
  miss 6.7 and 48.0 of a token's 480 picks after a warm-up of 300 tokens. The profile has 8.6 and
  62 on UD-Q4_K_XL.
- **The scripts.** `crates/model/tools/prefetch/`: `routes_predict.py` for the table (about
  5 minutes a capacity), `routes_lift.py` for the lift and the layers, `routes_slots.py` for slots
  by need.

Every table is learned online from the tokens before. Precision and misses named are at 10 named
a layer. "Budget" takes the best B non-resident named experts of a token over all layers.

| Predictor | 40G: misses named | 40G: precision | 40G: budget 20: used, misses covered | 16G: misses named | 16G: precision | 16G: budget 20: used, misses covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| row before | 0 % | 0 % | 0, 0 | 0 % | 0 % | 0, 0 |
| last four rows | 0 % | 5 % | 3 %, 0.5 % | 0.9 % | 8 % | 5 %, 1.9 % |
| same token last time | 7.1 % | 31 % | 32 %, 6.1 % | 17.4 % | 28 % | 32 %, 8.2 % |
| token table (decayed) | 7.6 % | 34 % | 9 %, 9.3 % | 17.3 % | 34 % | 30 %, 8.7 % |
| last two tokens | 6.3 % | 43 % | 28 %, 5.7 % | 11.5 % | 44 % | 44 %, 5.1 % |
| co-occurrence, L-1 → L | 1.4 % | 35 % | 2.3 %, 6.6 % | 5.8 % | 29 % | 16 %, 6.8 % |
| co-occurrence, L-4 → L | 1.0 % | 36 % | 1.6 %, 4.4 % | 4.4 % | 25 % | 13 %, 5.4 % |
| co-occurrence L-1 and token table | 4.8 % | 51 % | 3.9 %, 11.3 % | 12.0 % | 44 % | 25 %, 10.5 % |

- **Routes alone do not name the misses.**
  - The last tokens' picks are in the tiers by definition: 0 % of the misses.
  - The best route-only predictor covers 10 to 15 % of the misses at 20 to 40 reads a token.
  - The router-based predictors name 40 to 52 %.
  - Scoring the co-occurrence by lift instead of counts gave no more (5.4 % at 20 reads, 16G).
- **The token table pays at the first layers.** These have no prediction inside the step and
  miss most: 3.4, 2.1 and 1.9 misses a token at layers 0, 1 and 2 against 1.0 on average (16G).
  The table, at 16 named, names 52 % of layer 0's misses and 36 % of layers 1 and 2; with the
  co-occurrence, scored by lift, added, 57 % of layers 1 and 2. At 40G: 35, 12 and 14 %. Its lead
  is the drafts' (above).
- **Slots by need would save misses without any prediction.** The same total of slots, given to
  the layers by the number of distinct experts each used in the first 1000 tokens:
  - 48.0 → 44.7 misses a token at 16G (-7 %), with 228, 213 and 203 slots for layers 0 to 2;
  - 6.7 → 5.6 at 40G (-17 %).
  - Ares moves VRAM slots between layers by heat (`ExpertStore::rebalance`, every 256 steps) but
    not RAM slots. The simulation pools both, so it bounds the gain.

**Caveats.**

- The traces are IQ2_XS, short, and mostly prompt tokens read as decode rows.
- The tier model is a cache per layer, not Ares's two tiers with loans and promotions.
- The numbers rank the predictors. They are not the gains on UD-Q4_K_XL.

## Stand-in experts: a cost-aware rule of `--expert-prune`

Stand-ins change the output. They are outside the prefetch plan, which must not. The owner's
decisions of 2026-11-27:

- they belong to the dynamic pruning and stay opt-in;
- the rule must be balanced: when the read costs no speed, nothing is replaced; under the load of
  parallel calls, many distinct experts, it should win against the quality it costs.

### HotRod's rule, in its code

`route_resident_warp` (`src/kernels/cuda/native_router.cu`, lines 417-741) runs one warp per
token, right after the router:

```
gap  = l[e] - bv            // the pick's raw logit minus the best resident expert's
take = rank <  lo ? w * max(gap, 0) <= budget        // only if budget > 0
     : rank >= lo ? w * max(gap, 0) <= budget_tail
     :              gap <= edge                      // rank lo-1
```

- **The terms.**
  - `w` is the pick's renormalised top-10 weight.
  - The stand-in is the resident expert with the highest raw logit that the token does not use
    already, earlier swaps included. Only router logits count; there is no measure of similarity.
  - After the swaps the ten weights are the softmax over the computed experts' own logits.
  - "Resident" is on the card: HotRod also stands in for experts its RAM holds.
- **The defaults** (`setup.py`, Qwen IQ3_S; `route_budget_window` in `src/core/verify.cpp`,
  lines 5816-5944):
  - ranks 0 and 1 are never replaced;
  - rank 2 only within 1 logit (`STRATA_ROUTE_RESIDENT_EDGE=1`);
  - ranks 3 to 9 when `w × gap ≤ 0.1`, a fixed budget.
- **The self-tuning** needs `STRATA_FRAGILE_RECHECK=1`, opt-in since 9 Nov. The comment at
  `verify.cpp:496` says why: 42.4 to 43.6 tokens/s with the re-check off against 31.0 to 33.7 with
  it on, all answers clean in both. When it is on:
  - a row whose output-head margin (top 1 minus top 2) falls below k × the stand-ins' accumulated
    `w × gap` is decided again exactly;
  - two budgets alternate in runs of 8 windows, judged by tokens/s in paired blocks, with a veto
    on the rate of flips;
  - a brake per request halves the budget on flips.
  - `data/standin-tune.txt` keeps the result, and decode without the re-check reads it.
- **Stated as broken** (`docs/HOTROD-DETAILS.md`, comments in `verify.cpp`):
  - a fixed budget of 0.25 broke words in 2 of 3 prose answers;
  - the speed-only search climbed to 0.95 and broke prose;
  - any stand-in at ranks 3 to 9 broke JSON and code;
  - the budget at ranks 0 to 2 broke 4 of 11 emoji;
  - replacing rank 0 or 1 within 2 to 8 logits gave dropped tokens, wrong sums and collapse.
- **The maths 47/50 against 48/50 and code 30/30** (`SPEC.md` §18, 10 Oct: GSM8K 50, HumanEval
  30, greedy) were measured on the older rule (ranks 2 to 9, any stand-in), not on `w × gap`.
  Later checks: 12/12 JSON and code answers valid, MMLU-Pro 87.1 %.

### The rule in Ares

**`--expert-prune standin:<b>`**, a fourth rule beside `skip`, `router` and `similar`. `b` is the
ceiling of the quality budget, HotRod's 0.1 to start. It reuses the pruning's paths
(`ares_cuda::Prune`, `ares_expert_pick_row_pruned` in `kernels/expert_router.cuh`,
`device/prune.rs`):

- **What it may touch.** A pick may be touched only when its expert is in no tier and not on its
  way in time. That is the `PruneMask` byte the host writes each step (`mark_present`), with one
  refinement: an expert on its way counts as present only when its read is expected to land
  before its layer's record. One that will not stays 0, and its read goes on for later steps.
  - Ranks 0 and 1 are never touched.
  - Rank 2 only within 1 logit.
  - Ranks 3 and above when `w × max(gap, 0) ≤ θ_l`.
  - RAM-held picks stay exact, as the owner's "only what would hold the step" says. HotRod stands
    in for them too; that would be a later rule against the CPU's 0.31 ms a layer, measured the
    same way.
- **The stand-in** is the `router` rule's candidate: the best expert by router probability that a
  tier holds and that the row does not use. The block already ranks those candidates for rule 2.
  - What changes is that the touch condition depends on the gap to the candidate. So thread 0
    walks the eligible picks in rank order and gives each the next best unused candidate. It
    keeps the swap only if the pick's loss is within θ_l; otherwise the pick stays and the
    candidate stays free. That is at most ten turns of a serial loop.
  - The weights are the router's own probabilities of what is computed, renormalised
    (`PruneWeights::Renormalise`, as HotRod's softmax over the computed logits).
  - A variant on `similar` would take the loss `w × (1 − closeness)` with the table's stand-in.
    It is worth measuring next to it, not first.
- **The cheapest in quality first.** The kernel writes the loss `w × gap` of every file-only pick
  of the row into the route record, touched or not, beside the picks the record keeps. The host
  keeps their distribution per layer, as `Odds` keeps its counts. A threshold θ_l on the loss is
  the same as taking the cheapest picks first.
- **Only what would hold the step, and following the load.** The host sets θ_l for each layer at
  the start of every step, written with the mask into the same page-locked buffer (48 floats).
  It reads:
  - `StoreStats::wait_last_ms` and `waits_last` at "file" (`WAITED_FOR`): what the layers
    waited for the file, by what came last (on main since 372763fb);
  - the file picks a step, and the drive's measured pace (`DriveStats::shared_ns_per_mib`).
  - The number of picks to replace is about what the step waits for the file, in reads:
    `n = W_file × pace / bytes of an expert`, so each replaced pick is a read the step would wait
    for. θ_l is set so that about n picks a step fall below it, and never above the ceiling b.
  - When the reads are hidden, by prefetch or because the drive keeps up, `W_file` is near 0,
    θ = 0, and nothing is replaced.
  - With 2 or 4 sequences each step picks more distinct experts, waits more, and θ rises toward
    b.
  - Per layer: a layer whose misses all have reads on their way in time has nothing to touch.
    θ_l can be 0 for layers whose waits the counters show to be short, so the replacements go to
    the layers that hold the step.
- **Not reproducible bit for bit.** The rule depends on time. The fingerprint runs with it off,
  and its quality is measured as a distribution.

**What to measure before it is offered.**

- *Against the exact path* (nothing pruned), on the same prompts, greedy, alternating runs:
  - at 1, 2 and 4 sequences;
  - at `--ram 16G` and `40G`;
  - with prefetching as it will then be, since prefetch lowers `W_file` and with it what the rule
    touches.
- *Quality*, with `qwen4exp_prune measure` extended with `--rule standin` and the budget:
  - the KL divergence and top-1 agreement per step against the reference's 256 most likely
    tokens;
  - task checks of the kinds HotRod broke:
    - maths: 50 GSM8K-style problems, the answer checked;
    - code: 30 problems with tests;
    - JSON: answers parsed and checked against a schema, with and without a grammar;
    - German prose: broken words counted against a word list, and the KL on German text.
  - Several sequences go through the server's bench, since the tool runs one.
- *Speed*: tokens/s per sequence and in all, the file waits a step, replaced picks a token, and
  the mean loss.
- *To offer it*, a ceiling whose task results match the exact path within their noise on every
  kind, and whose KL is below that of `skip:8`: 0.026 on Q4 (see `experts.md`). The suggested
  value then is the ceiling that holds at 4 sequences and 16G.

## The plan, ranked

The gains are estimates from the profile and the offline numbers, against the profile's steps.
Those of 1 to 5 come on top of 0, and at 16G together they cannot pass the drive's floor of
~98 ms. "AB" means alternating runs of
`qwen4exp_run --decode-at 4000 --decode-steps 256 --breakdown --serve model` at
`--ram-gib 39.5` and `16`, three pairs. The counters to read are `StoreStats`: reads ahead,
used, wasted, `ahead_miss_ms`, an expert's reads, and the waits by what came last.

| # | What | Port (source, licence) | Gain at 40G (67.5 ms) | Gain at 16G (158.6 ms) | The A/B that proves it | Needs `iocore` first |
| --- | --- | --- | ---: | ---: | --- | --- |
| 0 | A core for the I/O thread (agent `iocore`) | — | -6 to -11 ms (measured) | -35 to -42 ms (measured) | the profile's | — |
| 1 | The read-ahead path under the budget: the bar from the spare instead of a half, in flight by the drive's curve instead of 2 lanes, the earliest deadline first, demote when a nearer list drops a guess, off where nothing qualifies | ideas only: HotRod's EDF (`expert_source.cpp:1052-1067`), TokenWing's `la_less_b` / `la_back`, Strata's `warm_scored` bar (all MIT) | -2 to -4 ms (3 of the 8.6 misses read 1 to 1.4 ms early; -1.7 ms where nothing qualifies) | -8 to -15 ms (6 to 8 of 62 reads overlapped) | AB new against today, and `--predict 0`; the prefetch latency must fall from 170-250 ms to under a layer's time | yes |
| 2 | The guess input through the target's FFN mixer plus L's shared expert | TokenWing `src/models/qwen4exp.cpp:656-689` (`build_hc_combine`, `build_hc_mix`), MIT; Ares has both mixer kernels | -1 to -5 ms net (about 4.5 of the 8.6 misses read early instead of 3; cost +1.1 ms at bandwidth, +3.9 ms until #18) | -15 to -25 ms with 1, to the drive's floor of ~98 ms | first `qwen4exp_predict record` and `analyze` on UD-Q4_K_XL (host part), then AB with and without | yes (through 1) |
| 3 | RAM slots by need between layers (`rebalance` for the RAM tier) | none (FATE's finding; Ares's own `rebalance`) | -2 to -4 ms (-17 % misses offline, an upper bound) | -5 to -10 ms (-7 %) | AB with and without; misses from the file a step | no |
| 4 | Lead from a learned correction, L+2 and L+4 | TokenWing `tokenwing-tools/correction/{make_correction,correction_fit,recording}.py` and `src/llama-moe-correction.cpp` (MIT) for the fit and the gate; HotRod's online learner (`native_router.cu:1867-2904`, MIT) only as the alternative | -2 to -5 ms (covered misses hide their whole read) | 0 to -5 ms | offline first (the record with logits; recall on the file-only picks by distance), then AB of d = 1,2 against 1,2,4 | yes |
| 5 | The drafts' tokens: a token table for layers 0 to 2, reads started as the chain drafts | none (AI-DER's table has no licence file: the idea, not code) | about -1 ms a verify step | -2 to -5 ms a verify step | AB with MTP drafts, with and without; misses of layers 0 to 2 | yes |
| 6 | Stand-in experts, `--expert-prune standin:<b>`, cost-aware (separate, changes the output) | HotRod `route_resident_warp` (`native_router.cu:417-741`) and `route_budget_window` (`verify.cpp:5816-5944`), MIT, as the rule's reference | not known: 0 at one sequence when 1 to 4 hide the reads | not known; the most where several sequences wait for the drive | the quality and speed matrix above | no, but measure after 1 and 2 |

**Not ranked.**

- *A ProMoE-style MLP.* It is only worth building if 2 and 4 leave the precision short; it needs
  the recordings of 4 and 190 MB of VRAM.
- *HotRod's online learner.* It costs 5 % of decode and 171 MB where the ridge fit costs nothing
  at run time. It is a fallback if a fit offline does not carry from text to text.

**What comes first.**

- 0, then 1, measured alone: without a path that serves a prefetch in time no predictor shows
  what it is worth. That is the profile's lesson of a prefetch waiting 170 to 250 ms.
- 3 can go in parallel: it touches only the store's rebalance.
- 2 is the largest prefetch lever at 16G, and its first step needs no new code, only a recording
  on the test machine.

## What could not be settled without runs

- **The reader.** How long a prefetch waits at the reader once the I/O thread has its core. The
  170 to 250 ms are with the starved thread.
- **Collisions.** What a prefetch chunk at the drive costs a demand read on the Samsung 990, in
  the engine, beside the CPU workers.
- **Precision on the real file.** The precision of 1b, 2b and 3 on UD-Q4_K_XL. All router-based
  numbers here are IQ2_XS, one card, from 2026-11-12, or stated by others on other models.
- **Lead in wall time.** How long a layer takes from its record to the next record at 40G and 16G
  after the I/O core fix. That decides how many layers of lead hide a read.
- **The mixer's cost.** What the target's mixer costs as a side product. It depends on #18's
  kernel.
- **Room at 16G.** Whether `Want::Spare` finds room at 16G or the guess pool is needed.
- **Slots by need.** How much of the offline -7 to -17 % survives Ares's two tiers, VRAM
  rebalancing and loans.
- **The drafts' table.** Its precision with real drafts, with acceptance of 54 to 77 % on first
  asks, and how much lead a chain gives.
- **Stand-ins.** Their quality and speed: no number exists for the `w × gap` rule on maths and
  code, even in HotRod.
