# Sequence state

A page holds the state of a fixed number of consecutive positions for every
part of the model that keeps state per position, on every device; devices keep
one array per part, indexed by page number.

- `PagePool`: fixed capacity from the VRAM ledger, reference counts.
- `Sequence`: a page table. `append` returns at most one `CopyOnWrite`, issued
  when the sequence is about to write into a page others also read.
  `truncate` removes rejected drafts. `fork` and `fork_prefix` share pages,
  which makes several continuations of one prompt (branch scoring, parallel
  tool calls, n-best) cost only their own tails.
- `PrefixIndex`: every full page is named by a chain hash of the context and
  all tokens up to its end. A new prompt walks its pages from the start and
  takes the cached ones. This one mechanism replaces per-slot prompt caches,
  slot similarity matching and a separate conversation cache, and it is the
  key under which pages will be written to disk.
  - A page enters the index when it fills, not when its sequence ends
    (`publish`, called from the commit with the number of settled cells), so
    two requests that start with the same prompt at the same time share from
    the first full page. A sequence that finds its page cached already takes
    the cached one and frees its own.
  - A sequence announces the pages its prompt will fill. A second prompt
    whose first missing page is announced is told so and is held back until
    the page is published or the announcement withdrawn, instead of computing
    the same rows next to the first. Only running sequences announce, so two
    prompts cannot wait for each other.
  - Tails: the partly used last page of a sequence is published under the
    hash of the page before it, with its tokens; at most 8 per page, least
    recently used first. A prompt that continues a tail starts at the real
    end of the cached text and copies the tail's page before its first write,
    which the page table does for any shared partly used page. A prompt may
    take any leading part of a tail, so a tail may hold only cells whose
    stored form depends on no later cell: any row of `f16` and `q8`, and with
    `q7` to `q4` exact cells only. A page is one tile there, and a tile is
    converted only when it is full and has left the newest window, so the
    partly used last page is always exact and is published at the commit
    that ends in it. A full page that was converted cannot serve a part of
    itself: its rows hang on the tile's mean and cannot be continued.
  - The last token of a prompt is always computed, because its logits are
    needed: the prompt is looked up without it.
  - Eviction takes pages nobody holds from the ends of chains, a lower
    priority class before a higher one whatever their age, and within a
    class the least recently used. `evictable` counts what can go.
- Checkpoints: state that is not addressable by position (recurrent layers)
  can only be resumed where a copy of it was kept, and a copy is large (75
  MiB as `f16` for Qwen3.8-27B), so `CheckpointPolicy` decides where copies
  go and which are dropped. The conductor keeps them as places within a budget
  of bytes (`RuntimeConfig::cache_ram`, which the pages in RAM share); the
  copies themselves are the device's part (`qwen35` does them, see below).
  The state is `f32` today: 150
  MiB for Qwen3.8-27B over both cards (94 and 56 MiB at shares of 0.625 /
  0.375), plus 20 KiB for the MTP block's carry.
  - Where: only at page boundaries and at the end of a tail. A prompt of a
    conversation names where its parts end (`Boundary`): the chat template
    knows it, and the model reads it off the template
    (`Model::encode_with_boundaries`, see "Where the parts of a conversation
    end" below). A later prompt that changes the conversation after such a
    place (an edited message, an older tool result cut, a new conversation
    with the same system prompt) shares everything before it. So a checkpoint
    is kept at the **roots**, the page boundary at or before the end of the
    tool list and of the system prompt, always (the llama.cpp fork's
    `--checkpoint-root`); at **every message boundary**, the page boundary at
    or before it, when it lies at least 1024 cells after the last checkpoint
    (a conversation of short messages would copy the state at every page
    otherwise); inside a prompt at least 8192 cells apart (the fork's
    spacing; ExLlamaV3 has 32 768), while generating at least 2 048, measured
    from the last checkpoint on the sequence's own path; always at the
    prompt's anchor, the last page boundary at least one row before the
    prompt's end, where the next turn of a conversation resumes; and the live
    state at the end of a request becomes the checkpoint of its tail. The
    commit takes a checkpoint when it crosses one of these positions, and a
    step ends there. A step cut short costs a pass over the weights (for
    Flash-Next over its experts), so a prompt step that ends on a page
    boundary at most a piece of prompt (`Model::prompt_rows`) before a message
    boundary that is due takes that boundary's checkpoint where it ends: a
    prompt read from its start in pieces of whole pages is never cut for a
    message boundary, its checkpoints lie up to a piece before them, and two
    of them lie a piece apart at least (for the 27B's pieces of 256 rows the
    1024 cells decide; Flash-Next's pieces are larger).
  - What a prompt replays: K and V are taken from every cached page and from
    a tail, however far the last checkpoint is behind them. The rows between
    that checkpoint and the end of the cached cells run again to rebuild the
    recurrent state; they read K and V and write none. That keeps the pages
    shared and saves the writes (and the scale search of the low-bit
    formats), not the rows' time: `Resume` names the checkpoint, the replayed
    rows and the computed rows.
  - Which go when the byte budget is full: first the aged ones. A checkpoint
    on one branch of the text is aged when a later request has checkpoints
    below it (TokenSpeed), or when three or more checkpoints lie below it: a
    conversation keeps its newest three (the fork keeps the newest three of a
    cached prompt), so a long prompt with many message boundaries does not
    push the newest checkpoints of other conversations out. A checkpoint where
    texts branch is never aged. The roots are never aged either and go after
    every other checkpoint: every new conversation of an agent starts there.
    Among the rest the one with the least distance to the next checkpoint
    toward the start of the text goes first, which is the rows a resume
    replays more without it (the fork's `--checkpoint-spread`: the kept ones
    thin out evenly), then the least recently used. A checkpoint goes without
    its K/V page; a page evicted from the end of a chain takes its checkpoint
    along. With room for one checkpoint besides the roots (Flash-Next at
    `--ram 8G` or 40G on one card: the tier of experts takes the rest, and
    the roots, which go last, hold every place but one), a request's end
    state does not push out its own anchor: the next turn often parts from
    the cache inside the reply (Claude Code sends a call of its Edit tool
    back with `"replace_all": false` the model never wrote; Qwen's templates
    leave out an answer's thinking once a new question follows; a client
    asks the same prompt again), and from the end state it would replay
    every row it shares after the roots, from the anchor only the reply's.
    (Until 2026-11-22 the rule looked at a budget of one place only: on the
    agent suite of the Strata comparison, UD-Q3_K_XL at 40G on the 4060 Ti,
    the same prompt asked a second time, 6784 of its 6832 tokens cached,
    took 31.2 s against 30.3 s for the first ask: it replayed from the root.
    The host executor's test of the suite replayed 672 of 736 cached rows.)
    A root keeps its mark when its page goes to RAM and comes back, is
    never "left behind" for the drive (it is written when it leaves RAM, if
    its text is long enough for the drive), and comes back from the drive as
    a plain checkpoint.
- **Where the parts of a conversation end** (`ares-model`, `chat.rs`). The
  engine renders the template, so it renders a conversation a second time,
  with a mark (a character of the private use area and the message's number)
  before and after the text of every message that is not empty, inside the
  white space at either end of the text, which templates trim. The places of
  the marks in the text without them are the boundaries, and they count only
  when that text is the prompt to the byte: a template that reads a message's
  text otherwise than writing it (a test whether it is empty, a split of it)
  renders something else with the marks, and such a prompt has no boundaries.
  A mark the template writes twice is left out. A byte place becomes a token
  place through the lengths of the tokens' pieces; a place inside a token
  counts where the token starts. The roots are the start and the end of the
  system prompt and the start of the first message after it: Qwen3.5's and
  Qwen3.8's templates write the tools before the system prompt, so its start
  is the end of the tool list; a template that writes them after it ends them
  at the first message. Measured on the three model files' templates (a
  conversation with tools, a call with reasoning, its result, an answer):
  two boundaries per message, three roots. Plain text (`/completion`) has
  none.
  - **Measured** 2026-11-22 on the Messages route (Qwen3.8-27B Q4_K_S, two
    cards, `--ram 8G`, load 4 to 11; before is `main` without boundaries, A,
    B, A, B): a conversation with a system prompt of about 12 000 tokens, 12
    tools (about 3000) and a task of about 1500 tokens goes on for three
    turns; a second and a third conversation with the same system prompt and
    tools and other tasks then took 20.3 to 20.6 s to their first token
    before (18 304 rows replayed: the first conversation's checkpoints lie
    after its task) and 2.54 to 2.63 s after (nothing replayed; what is left
    is the new task's rows). The cold first prompt of 19 903 tokens took
    20.59 and 20.61 s before and 20.77 and 20.88 s after (five saves against
    two). A conversation of five turns (user messages of about 300 tokens)
    whose first message is edited replayed 8064 rows (9.13 s) before and none
    (2.33 s) after; with its second message (the first answer) edited, both
    restore the first turn's anchor (nothing replayed, 2.3 s). Claude Code's
    billing stamp: the next turn with another `cch` took 8950 of 8970 tokens
    from the cache (0.18 s). Flash-Next at `--ram 8G` (with `--cache`) has
    one checkpoint place, and there the roots do not pay yet: the root leaves
    the place for the later checkpoints of the same prompt and goes to the
    drive, but a prompt asks the drive only for the pages the index does not
    hold, and the index still holds the system prompt's pages, so the second
    conversation replayed 18 304 rows (221 s against 166 s for the cold first
    one). What would make it pay is in `OPEN.md`. The one place does keep a
    request's anchor over its end state now: with Claude Code's rewritten Edit
    call sent back (a forced call per turn, `"replace_all": false` put in
    front, Flash-Next without `--cache`, load 16 to 22), the next turns
    replayed 8146 and 8255 rows before (136 and 64 s) and 82 and 63 after
    (12.2 and 10.6 s).
- Admission (`ares_sched::Admission`): nothing is reserved for a request's
  longest possible length. Pages are claimed per step. A reservation covers
  the rest of the running prompts and the output they are expected to
  produce: `min(tokens left, 4096)` times a ratio that every finished request
  moves toward the share of its allowance it really used and that a shortage
  raises. A new request is charged for the pages of its prompt after the
  prefix match and its clipped allowance in full, and waits when that does
  not fit; nothing is refused for load, and a request that is alone always
  starts. Waiting requests are tried with the fewest uncached rows first; one
  that was passed over too often goes first and holds the others back. A
  step's claims are settled against a copy of the counters: cached pages are
  evicted first, draft rows are cut before anyone gives way, then the
  sequence with the fewest generated tokens is spilled, and victims that
  turn out not to be needed stay. All of this holds among the requests of
  one priority: the line is taken by priority first, a shortage spills the
  worst priority first, and a request that does not fit makes running ones
  of a worse priority give way (see "Priorities of requests" under
  [Server](server.md)). The object decides and reports; the
  conductor carries it out.
- In the conductor (`crates/runtime/src/conductor/`, `cache/`), for a
  model that takes part (`Model::prefix_cache`):
  - A request is looked up when it arrives and again when it starts. It
    shares the cached pages and a tail, restores the state from the last
    checkpoint on its path, replays the rows from there to the end of the
    cached cells (a segment of its own with the replay flag; they read the
    cells and write none) and computes the rest. A prompt whose first missing
    page a sequence that already runs has announced is put back in line,
    held, until the page is published or the announcement withdrawn; a
    request that arrives in the same pass as the one that computes the page
    is held after it started.
  - A page enters the index at the commit that fills it. A prefill step ends
    at the next position where the policy wants a checkpoint, and so does the
    window of a generating sequence (its drafts are cut), so the state is
    exactly there when the save is queued; the save runs with the next
    record. Replayed rows and rows that write are separate steps.
  - When a request ends, is cancelled or gives way, its partly used last page
    is published as a tail and its live state becomes that tail's checkpoint
    (or the last full page's, on a page boundary). Tails are not published
    while a request runs: the index would hold a reference that makes it copy
    its last page at every step. A request that gives way comes back as a
    prefix hit.
  - When a step's claims need room, cached pages go from the ends of chains
    after the requests that start have taken what they share; checkpoints go
    when their budget is full and with their page. A checkpoint that goes
    while a restore of the next record reads it keeps its place until that
    record has run, and a save into it that waits for the same record stays:
    the record saves before it restores, so the restore gets the state the
    checkpoint was taken of. (An agent's next turn restores the end state of
    the turn before, which is saved with the same record; when that turn is
    read in one wide step, the places of its saves inside are taken before
    the record is written, and with one place that gives the end state up.
    Until 2026-11-23 its save went with it, the restore read what the place
    held before, the first turn's anchor, and Flash-Next failed the step:
    "the rows of slot 0 start at cell 650 and its sequence is at cell 512".
    `tests/cache/prefix.rs`
    `a_turn_that_restores_an_end_state_still_saved_with_its_record_gets_that_state`, and on
    the GPU `qwen4exp_runtime`'s
    `a_turn_read_layer_by_layer_from_an_end_state_saved_with_its_record_answers_as_the_reference`,
    which failed the same way before.)
  - Per request the runtime reports the prompt tokens that came from the
    cache (`Usage::cached_prompt_tokens`) and the rows it replayed and
    computed (`Snapshot::requests`); the counters add them up.
  - The host executor plays a device as an oracle: it keeps the cells of every
    page and the state of every slot as hashes of all tokens in order, and a
    row that reads a wrong one picks another token.
- On the devices (the shared slot block `crates/model/src/blocks/slots.rs`;
  `qwen35` in `crates/model/src/qwen35/device/part/resume.rs`, `qwen4exp` in
  `crates/model/src/qwen4exp/device/resume.rs`, see "Qwen3.8-Flash-Next
  through the runtime"), in the order the record's documentation gives (the
  commit of the step before, the saves, the page copies, the restores, the
  rows):
  - **A checkpoint** of a device part is its slot of the recurrent state (the
    convolution windows and delta-rule matrices of its blocks) and, on the
    device that holds the MTP block, one row of the block's carry. It lies
    in **page-locked host memory**, not in VRAM: a save is a copy from the
    device and a restore a copy to it, queued on the step's stream, so the
    stream orders them against the kernels. The places are locked in chunks
    (at least 32 MiB) when one is first used, within the engine's RAM limit
    (see "Host memory").
  - **A page copy** copies the whole entry of the page, for every attention
    block and for the MTP block's own array (the cells of an entry lie in
    planes per head, and cells past those the sequence has are written before
    they are read).
  - **A replay** is a prompt step whose rows append nothing: the K/V append of
    every attention block and of the MTP block's follow of the model runs
    per run of rows that write, and the rows of a replay are left out. All
    else runs as for a prompt, which advances the recurrent state and leaves
    the carry at the last row.
  - **The MTP block's state per sequence** is its carry (the final hidden
    state of the sequence's last row: the first row of a segment reads it for
    the block's keys and values at its position, and the next chain of drafts
    starts from it) and its keys and values, which lie in the pool's pages
    (one more array of entries), written by the follow of every row of every
    step. Both are therefore in a checkpoint and in a page copy, and a
    replay must not write the cells.
  - **Measured** 2026-11-11 with `qwen35_run --conversation` (the second turn
    of a conversation whose history is the given number of tokens, with the
    chat template's thinking off; a conversation of the same size that the
    engine has never seen as the cold control), Qwen3.8-27B, tensor split
    0.625 / 0.375, 150 MiB of checkpoint over both cards:

    | History | Prompt of the second turn | Cold, first token after | From the cache, first token after | Cached / replayed / computed rows |
    | --- | --- | --- | --- | --- |
    | 8k | 8278 tokens | 7070 and 7042 ms (1167 tokens/s) | 138 and 111 ms | 8255 / 0 / 23 |
    | 24k | 24 662 tokens | 25 239 and 25 258 ms (975 tokens/s) | 150 and 122 ms | 24 639 / 0 / 23 |

    The checkpoint of the first turn's end is exactly where the second turn
    starts, so no row is replayed. A save of one checkpoint takes 3.7 ms on the
    RTX 4070 (93.5 MiB, 25 GB/s) and 17.6 to 17.9 ms on the RTX 3060 (56.1
    MiB, PCIe 3.0 x4, 3.2 GB/s); a restore 3.9 and 17.7 ms; a copy of a page
    0.03 to 0.05 ms. The cards work in step, so a step with a save or a restore
    takes the 3060's 17.7 ms longer, once per checkpoint: a request that ends
    leaves one save (and the anchor of a long prompt another), a request that
    starts from a checkpoint one restore. Against the seconds a start from
    nothing costs this is no problem, and a staging area in VRAM
    (which the copies could use to overlap with the step) is not built. The
    first use of a place waits for its lock (on the conductor's thread before
    the step since 2026-11-24, on the device's before): 21 to 22.5 ms for the RTX 4070's
    93.5 MiB and 11.8 ms for the RTX 3060's 56.1 MiB, once per place (the
    first two places of a run cost 52 and 49 ms of waiting on the host). RAM
    locked: 748 MiB after the first conversation (5 places) and 1048 MiB after
    the second (7 places). The same second turn of 8192 tokens of the 0.8B on
    each card alone (8274 prompt tokens, 8220 cached, 28 replayed, 54
    computed): RTX 4070 cold 361 and 357 ms, from the cache 21 and 17 ms, save
    0.77 ms, restore 0.82 ms (19.3 MiB); RTX 3060 cold 810 and 806 ms, from
    the cache 41 and 37 ms, save 6.15 ms, restore 6.11 ms. Generation is as
    fast as before: with the same flags on the same day a prompt of 2048
    tokens reads at 1228 to 1241 tokens per second before and 1227 to 1241
    after, a decode step without drafts at 1000 tokens takes 26.0 and 25.9 ms
    before and 26.0 and 26.1 ms after (the 24.8 ms of an earlier day is not
    this machine's today), and with the model's drafts (3 per window, the
    fork's draft vocabulary, 300 tokens, most probable token) German, English
    and code give 79.5 / 76.2, 73.3 / 76.5 and 75.9 / 74.4 tokens per second
    before and 77.8 / 77.6, 76.5 / 80.3 and 78.3 / 81.5 after (78.4, 75.2 and
    80.8 with 16 checkpoints kept), a step of 34.2 to 34.6 ms throughout.
  - **Rounding.** A row's result depends on the size of the step that
    computes it, also with exact products (measured on the 0.8B, one prompt of
    480 tokens in steps of several sizes against steps of 128 rows: steps of
    32 and 64 rows give the same logits bit for bit; steps of 1, 7 and 100
    rows give logits that differ by 2.0e-3, 1.6e-3 and 1.2e-3 of the logit at
    most, with 2, 1 and 0 of the 480 top tokens different). The cache changes
    the steps that compute a prompt (rows replayed from a checkpoint, the last
    row alone), so a run from the cache is not bit for bit the run without it:
    the logits of 40 generated tokens differ by 0.7e-3 to 0.9e-3 of the logit
    at most (one card, each card alone, tensor split, layer split), the tokens
    are the same. Texts are the same in every test that copies a text, whose
    words follow with a wide margin; a text with near ties (a list whose order
    the model is unsure of) can part at one, as can two requests that decode
    together instead of alone.
- **Decisions** ("Jev mode"): typed questions about a state answered by choosing
  among fixed options, with no generation, on `POST /v1/systemone` for every
  model. The request, the answers and the way a chat model is asked are
  Flide's (https://github.com/OmniJev/Flide, its `next_token` format in style
  `vllm`, the rendering of vLLM's endpoint): every question is a prompt of its
  own, one user turn with the state, a blank line and the question with its
  options labeled (A to Z, then the two-letter labels that are one token;
  `yes` and `no`; 0 to 9), thinking off, and the answer is the softmax of the
  labels' logits at the first token of the reply (`ares_api::Question::render`
  and `answer`). The questions of a request share the state as the start of
  their prompts, and Flide's exact state forking is done with the prefix
  cache (`conductor/decisions.rs`):
  - **The parent**: the shared start runs as a request of its own that reads
    its prompt and ends. As every request does, it leaves its pages, its
    partly used last page and its recurrent state at its end in the cache
    (`Conductor::keep`); a state read before is found there, as any prompt.
  - **The forks**: then every question is a request whose prompt is the whole
    question. It shares the parent's pages, copies the partly used page
    before its first write, restores the checkpoint (an exact copy of the
    delta-rule matrices and the convolution windows at the end of the shared
    start, the fork of Flide's Gated DeltaNet layers) and computes only its
    own rows. The forks are prompts like any other: they share steps with each
    other and with generation, as many at once as there are slots, shortest
    first.
  - **The read**: a fork's last row is wanted, and the step hands back the
    logits of the question's label tokens there, before the sampler changes
    any (the record's *scores*, gathered on the device that holds each token's
    column, see `record.rs`). The fork then ends and keeps nothing but the full
    pages it published (a checkpoint of its state would cost a copy and serve
    nobody).
  A question alone gets the same logits up to the rounding of its rows' steps
  (see "Rounding" above): the parent and the forks are prompts computed from
  the same pages and state a request of the whole prompt would read. When the
  cache cannot hold what the parent left, a fork restores an earlier
  checkpoint and replays, or computes from its first token: the answers are
  the same and only the work is not shared. A decision of one question has no
  parent. The parts of a decision are requests the conductor makes (ids with
  the top bit set); the counts of requests and the accounting of what
  requests produce leave them out.
- `PageLayout`: bytes per page per device, from which the pool's capacity
  follows. For a `qwen4exp`-shaped model the full 262 144-token context is
  3.47 GiB with K and V as Strata stores them (int8, a scale per 64 values)
  and 3.56 GiB with the engine's `q8` (a scale per 32), shared by all
  sequences.

**KV formats.** K and V are stored in one ladder of formats with plain
names, chosen separately for K and for V: `f16`, then `q8`, `q7`, `q6`, `q5`
and `q4`, where `qB` is B bits per value plus an f16 scale per group of 32
values (B + 0.5 bits per value; `q4` is 18 bytes per 32 values). Each group is
rotated with a 32-point Hadamard transform before it is quantized, which
spreads outliers over the group. The rotation costs nothing when reading: a
dot product is unchanged by it, so attention rotates the query once and never
un-rotates K, and V is accumulated rotated and un-rotated once at the end. The
reference encoder and decoder live in `ares-quant`; the attention kernels of
both backends are held to them.

`q8` is the default for K and V: against `f16` it loses hardly anything and
it is the fast choice. Below it the formats work harder the smaller they get:

- `q7` to `q4` search the scale of each group when writing (3, 5, 9 and 17
  candidates, each with a least-squares fit of the scale).
- They store their rows a **tile** of 128 cells at a time, relative to the
  mean of the tile: one f16 vector per tile, 0.125 bit per value. This is the
  one technique that costs something when reading: one extra term per tile
  and query (for K a dot product of the query with the mean, for V the mean
  times the summed weights of the tile).
- A `KvPolicy` keeps the first tile and the newest cells of a sequence in
  `f16` (none at `q8`; at `q4` the first 128 cells and the newest 256 to
  383), which is what kvarn (beellama.cpp) gets most of its gain from. A tile
  is converted when it is full and leaves the newest window, so its mean is
  known by then.

Any K format combines with any V format. A page has to hold whole tiles: the
stored rows of a tile depend on all of its cells, and a page is shared
between sequences by the tokens up to its own end.

**Where the tile formats keep their exact cells** (the K/V group of
`ares-model`, `blocks/kv.rs` and `blocks/forms.rs`). The engine's pool holds
converted entries only: page `p` is slot `p` of the converted pool of every
attention layer, 128 cells in the format. The exact entries (a tile in `f16`) are a second pool on each device,
for the attention blocks and for the MTP block separately, which a page
holds a slot of while it is exact: from its first write until it is full and
has left the newest cells of every sequence that reads it. The conductor knows
nothing of this beyond one thing, the pages the pool freed, which every step
record names (see "Sequence state" below); the device decides the rest, the
same way on every device, from the segments of the record:

- At the start of a step, for every sequence in it, the tiles that left its
  window are converted into their pages (`AttentionKernels::convert`), the
  pages its rows write get exact slots, and the group's page table is brought
  up to date (entries of exact tiles name the slot, the others the page). The
  step's own rows then append to exact tiles and attend to both kinds.
- A page shared by two sequences is converted for both, when either's window
  has left it; each finds its table brought up to date when it next runs. A
  sequence that starts from the cache therefore reads the cached tiles in the
  form they have: converted where the sequence that computed them went on,
  exact where it ended.
- The pool has `slots * (2 + ceil(newest / 128)) + ceil(rows / 128) + slots +
  tails` slots (`exact_slots`): the first tile and the window of every
  sequence that runs, the tiles a step begins, and the tails of the prefix
  cache. When it is short the full page that was used least recently and that
  no sequence of the step reads is converted for room (a cached page loses its
  exactness, never its text); a partly used page never is, which is why the
  cache keeps at most one tail per slot (`PrefixCache::tails_per_slot`).
- Page copies copy the exact entry; a page that is copied is always partly
  used, hence exact.
- A row reads a tile converted from the position at which the tile has left
  the newest cells (`newest + 128 * (tile + 1)`), whatever step computes the
  row. A draft window that reaches across such a position has the tile both
  ways: it is converted at the start of the step and keeps its exact entry;
  the rows from the position on read the converted entry (the plan of their
  attention names it for that one tile, the table does not), the rows before
  it the exact one, and a step that starts past the position gives the exact
  entry up. So a row of a window reads the bytes it would read in a step of
  its own. The tile must have been full before the window and only one tile
  may leave inside it, so with these formats a draft window has at most
  `min(newest, 128)` rows (32 with `q7`, 64 with `q6`, 128 with `q5` and
  `q4`); a longer one is refused by the device part.

The formats were chosen on synthetic rows and then measured on the K and V of
two real models (Qwen3.5-0.8B and Qwen3.8-27B, one text of 3.7k tokens; the
measure is the error of attention, not of logits;
`docs/dev/studies/kv-real-data.md`):

- The scale search lowers the error of the attention output by 12 %, 15 % and
  18 % at `q6`, `q5` and `q4` against the plain scale of `q8`, in every layer.
- The exact cells are the largest effect: 76 % less output error at `q4` with
  128 first and 256 newest cells. The newest cells carry most of it.
- kvarn's quantizer is better on K and worse on V; on the attention output it
  is within 6 % of the engine's. Its two-axis normalisation adds nothing on
  top of the engine's format.
- The mean of a tile came out of this measurement: a quarter to a third of the
  energy of real K and V is the mean of each value over neighbouring cells.
  With the built codec and the engine's policy it lowers the output error by
  25 %, 27 % and 32 % at `q6`, `q5` and `q4` on the 0.8B (19 %, 21 % and 25 %
  on the 27B) and the KL divergence of the attention weights by 48 to 49 %
  (42 to 44 %). No layer, part of the context or kind of text gets worse.
- All of it together, `q4` for K and V on the 0.8B: the output error is 6.5 %
  with a plain scale and nothing exact and 0.74 % as built.

What is still missing is a measurement on logits at a long context: with 3.7k
tokens the exact cells are an eighth to a fifth of what a query reads.

**The reference for attention.** `ares_quant::kvhead` is what the attention
kernels of both backends are held to. `KvHead` is the stored sequence of one
KV head under a policy, byte for byte as a kernel must write it: new cells as
`f16`, tiles converted from the stored `f16` values when the policy says so
(on the real data that costs nothing measurable against converting from the
originals). Draft cells are appended as pending, which converts nothing, so
rejected ones can always be cut and accepted ones are settled. Attention is
defined the way FlashInfer builds it: the attention of a query over a range of
cells is a state (the log of the summed exponentials of its scores, and the
weighted sum of V), and the states of disjoint ranges merge into the state of
their union, in any order. Dense attention is one range, block-sparse
attention the selected blocks. This is what lets a kernel compute pages
independently, lets K and V of a sequence lie on two devices with only states
crossing, and lets a prefix shared by many rows of a step be attended to
once. A plain f32 implementation agrees with the reference to 2e-6; the
tolerance for kernels is ten times that.

### What this prepares for Qwen4

The reference is Strata's implementation of Qwen3.8-Flash-Next (`qwen4exp`);
see [reference/strata-notes.md](../studies/strata-notes.md). The items of the
Qwen4 handover map as follows.

| Handover item | In this engine |
| --- | --- |
| Sparse attention on index lists (the indexer's selection should be what attention reads, not a mask over all cells) | Attention reads through an id list and the page table by design. The page size is a multiple of the indexer block, and the pooled key of a block is a per-page part of the layout. The selection and attention kernels come with the model. |
| Sparse KV offload (K/V in pinned RAM, keys and a page cache on the GPU, fetch only missing selected pages) | Built for `qwen4exp` (2026-11-22, "K and V beyond VRAM"): the pooled keys of every page stay on the GPU, the K and V of the first pages too, those of the rest lie in page-locked RAM and the attention reads the picked cells there over the link; a wide step and a pass read a copy of a layer's far entries in lent VRAM. No page cache: a page is near or far by its number. |
| Ragged one-pass verification (drafts of unequal length in one batch) | `plan` puts windows of any lengths in one step. Recurrent kernels take per-sequence segments from the step record. |
| Graph cache for batch shapes | Not needed: there is no per-shape graph. |
| Adaptive draft length | `choose` maximises expected tokens per millisecond of the whole step from measured acceptance per position (or the n-gram drafter's confidence) and the measured price of a step by its rows. The same number chooses between drafters and accounts for the rows a draft costs other sequences. |
| Rollback rows independent of draft length | Verify and commit removes rollback: nothing recurrent is modified before acceptance. |
| Draft vocabulary without extra VRAM | Planned in weight placement: the output head's rows are ordered by frequency at load, so a draft head is a prefix of the same tensor. |
| VRAM as the binding limit | Ledgers and the three kinds of device memory. |
| Prefill next to generation | Pieces of prompt between rounds of generation, paid back by time (`Share`). |
