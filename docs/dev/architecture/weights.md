# Weights

There are two ways a model's weights are held, and no third:

- **Dense: everything in VRAM.** All weights of a dense model are on the
  GPUs, placed on one device or split across several when the model is
  loaded; the split planner (next in `ares-runtime`) works from the memory
  ledgers, the measured memory bandwidth of each device and the measured cost
  of the link between them. There is no offload of dense weights or layers to
  the CPU. A dense model that does not fit is the wrong model for the
  machine; what can give way is its context (see KV offload below), or a MoE
  model is used.
- **MoE: experts in tiers.** The dense parts are in VRAM; the experts are
  spread over VRAM, pinned RAM and SSD and follow the routing, as described
  next. Experts that a token needs and that are not on the GPU are computed
  on the CPU or streamed to the GPU, whichever a measured cost model says is
  faster, in a pipeline next to the GPU's work. There is no setting for how
  many layers keep their experts on the CPU: the tiers are sized from the
  memory budgets and filled from what the router picks.

**KV offload** exists for both, and is adaptive: K and V pages that do not
fit in VRAM live in pinned RAM, as many as fit stay resident, and the rest are
brought in as attention needs them. For attention that selects blocks
(Qwen4-style) only the selected pages move. For dense attention every page is
read at every token, so the stream of pages from RAM is bound by the PCIe
link and is prefetched layer by layer through a ring, as in
RaymondHuang210129's adaptive KV streaming for llama.cpp, which is the
reference for this part. It is the next piece of `ares-state` (residency of
pages) and comes with the runtime.

- **MoE tiers.** Experts live on VRAM, pinned RAM or SSD. `ExpertHeat` counts
  routed picks with decay (seeded from the usage profile an earlier run of the
  model wrote, which also fills the tiers at the start and, while nothing runs,
  their empty slots: see "The usage profile of the experts" under
  "Qwen3.8-Flash-Next through the runtime"); a `SlotSet` per layer and tier
  elects the hottest experts and re-elects every few tokens with a margin
  against thrashing and a cap on swaps. A swap hides the evicted expert at once
  and shows the new one when its data has arrived. On a synthetic workload whose topic changes every 500
  tokens, re-election serves 79 % of the picks from a tier of a quarter of the
  experts where a tier filled once serves 29 %.
- **The experts of a drafter stay resident.** A layer of the expert store can
  be resident (`LayerSpec::resident`): every one of its experts has a VRAM
  slot in a pool of its own for as long as the store lives, outside the
  tiers. Its bytes come out of the device's budget before the tiers get
  theirs (a budget that does not hold them is refused with what they need),
  a few RAM slots pass its experts through from the file when the store is
  filled, and nothing evicts them or moves its slots. A model's MTP block
  is such a layer: its router runs in every link of every chain, and a link
  that waits for an expert costs what a token costs. Flash-Next's block has
  512 experts in Q2_0, 675 MiB on the device with the output head.
- **A row that verifies a draft waits for the file only while it is worth
  it.** The model tells the store the windows of a step that verify drafts
  and what each row of the step is worth (`ExpertStore::set_windows`, from
  the record's row worth: for row `k` of a window, the chance that the
  drafts up to its own are right times what a token costs at the rate the
  round was chosen for). A window's first row is its sequence's newest token
  and gets every expert it picks, wherever it is; a row after it verifies a
  draft that the next step can verify again. An expert that no tier but the
  file holds (or whose read is on its way) and that only such rows picked
  costs its layer a read, unless the layer waits for the file anyway; it is
  read while the rows a missing expert would leave behind (in each window
  the lowest row that picked it and the rows after it) are worth more than
  the read, which is then charged to them, so a row that would wait for many
  reads is left after a few. An expert that is not read is claimed by the
  CPU, which adds nothing to the rows' sums. Those rows and every row of
  their window after them are *left*: their results are not what the rows
  would give alone, the model says so (`DeviceModel::rows_not_exact`), and
  the conductor accepts no draft from the first left row of a window on.
  The rows before it stay exact, because a row depends only on the rows
  before it, and the text is the text without drafts. An expert whose rows
  are all left needs no work at all, wherever it is. The rule is in the
  store's miss path for every model with experts in tiers
  (`expert_store/windows.rs`); a store without a CPU half leaves nothing,
  and neither does a step whose record says nothing of its rows' worth.
  What a row was left without is read into RAM as a guess for the steps
  after (`StoreConfig::read_left`): the token of a left row is most often
  the next round's first, which routes to the same experts and would
  otherwise wait for them there, alone in its layer. See "Speculation" for
  what was measured.
- **Dynamic expert pruning, a fourth level.** A pick of the router whose
  expert only the file holds may be left out, or another expert that a tier
  holds may stand in for it, so that the step does not wait for the drive;
  the router of a decode step does it on the device right after its top-k
  (`ExpertKernels::route_pruned`, `ares_cuda::Prune`), off unless asked.
  See "Pruning as a fourth level" under "Qwen3.8-Flash-Next through the
  runtime". Until 2026-11-16 this line described `SlotSet::write_route_bias`,
  a bias for the router's scores with the strongest picks exempt; nothing
  called it and the router kernel had no input for it, so it was removed;
  the rule `router` is what it set out to be.
- **Reading from SSD** (`ares-io`; the expert store and the n-gram rows of
  `qwen4exp` read through it): experts
  are read in place from the GGUF at `offset + e * stride` with direct I/O
  through `io_uring` into buffers the caller provides (pinned staging buffers
  of a GPU backend), demand reads before read-ahead, which is cut into chunks
  and can be cancelled or promoted. The wide steps of `qwen4exp` submit all
  reads of a layer at once, into staging slots next to the RAM tier (see
  "Wide steps" under "Qwen3.8-Flash-Next through the runtime"). A drive takes
  at most four prefetch chunks of 512 KiB at once, and two while a demand
  chunk is there (`ares_io::Config::prefetch_depth`, Run My AI's rule). The
  steps of eight rows of `qwen4exp` read ahead of the router what the routers
  of the next layers, applied on the device, predict (see "Reading experts
  ahead of their router" under "Qwen3.8-Flash-Next through the runtime").
- **Several SSDs.** A model can have copies on more than one drive (a mirror
  directory per extra drive), and the parts of a split model can be on
  different drives. Every drive has its own queue depth, and each chunk of a
  mirrored file goes to the drive expected to deliver it soonest. Each drive
  is described by two measured numbers, the time of a chunk that is alone
  there and the time per chunk when several are, so a lone read goes to the
  drive with the shorter access time and a burst is shared by throughput; a
  drive that has not been measured for a second is tried again. Measured
  2026-11-10 with reads of 1.44 MiB (one expert of `qwen4exp`) from a 14 GiB
  file mirrored on the two NVMe drives of the development machine: 1.66 and
  1.36 GiB/s for each drive alone with 8 to 64 reads in flight, 2.9 to 3.0
  GiB/s for both; with 8 in flight the median read takes 3.8 ms against 6.1
  and 8.3 ms; a lone read takes 1.4 ms, the time of the faster drive. The
  copies must be identical: sizes are compared and contents sampled at open.
  The price is the disk space of the copy.
- **Prefill of a MoE model, layer by layer.** A long prompt uses most experts
  of every layer. Run in chunks of rows through the whole model, it fetches
  them again for every chunk, from the SSD or over PCIe from RAM. The other
  order fetches each expert about once per prompt: a layer is computed for
  all rows of the prompt before the next layer starts. Every engine that
  streams experts found this to be the largest effect on long prompts (Run
  My AI: 954 GiB read for a 4k prompt in batches of 16 tokens without it,
  about one read per expert with it; TheCodacus: 16 to about 300 tokens/s on
  a 16.7k prompt; none of these numbers was reproduced here). Flash-Next
  reads its long prompts in that order, as a pass of the model that can
  pause between micro-batches; a layer's misses come through transit there
  (see "Long prompts layer by layer" under Flash-Next). It replaced a first
  design of pieces planned in `ares-sched` and waves of a layer's experts
  through the tiers, which is gone.
- **More from the engines that stream experts** (details and the sources'
  numbers in [reference/moe-offload-survey.md](reference/moe-offload-survey.md)).
  Built as bookkeeping in `ares-weights`, with no runtime using it yet; the
  numbers are from a synthetic workload with the properties the sources
  report (locality in time, topics that change, early layers that route by
  the token), not from a model:
  - *The RAM tier as a cache in front of the SSD.* `SlotSet` takes reads as
    well as elections: a demand read gets a slot at once, a guessed read
    (`GuessQueue`) gets one only when it is started, experts routed or
    predicted for the current token are protected, and slots can be given
    back to the memory books and taken again. Against a tier that is only
    re-elected, the model starts 34 % fewer reads per token and waits 20 %
    less with 4 reads in flight (58 % and 44 % with 8). Nearly all of that
    is reading guesses at all; protection removes the evictions of experts in
    use, binding the victim late is worth about 1 %, and admission on its own
    gains nothing over re-election.
  - *Slots per layer by need* (`split_slots`): a tier's bytes go to the
    layers by what the next slot gains, with a margin and a cap on moves so
    that slots do not go back and forth. It raises the share served from the
    tier from 0.70 to 0.72 at 32 slots per layer and does nothing at 16; at
    small budgets it takes slots from the early layers, which is the opposite
    of what the sources recommend for them, so a weight for how badly a layer
    is predicted is still to be added.
  - *A table from the last tokens to likely experts* (`TokenTable`) for the
    first MoE layers, which the router prediction cannot reach: 104 bytes per
    entry for 4 layers of 8 experts, built from recorded routing. On the
    synthetic workload the first layers are served 0.69 instead of 0.39 with
    527 entries. It needs a recording tool and must be measured on a real
    model before it is kept.
  - *The lookahead distance as a measured setting* (`Lookahead`): from the
    read time, the layer time and the recall observed per distance it picks
    the distance with the least expected waiting, starting at two layers.
  Still planned, with the first MoE model: inside a layer, experts that are
  present first and a miss issued the moment the router has run, gate and up
  before down; gate and up of an expert adjacent in its slot and computed in
  one pass; the shared expert's output in the predictor's input.
- **Hashed-row tables**: for models with n-gram embedding tables (28.8 GB for
  `qwen4exp`) the row ids follow from the last tokens on the host, so the
  reads are issued when the token is known and collected before the layer
  that needs them, behind a row cache (`ares-io`; `qwen4exp` does this).
