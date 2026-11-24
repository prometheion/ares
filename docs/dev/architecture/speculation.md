# Speculation

The engine has every kind of drafter that llama.cpp and Piggidragon's fork
have, with the fork's selection on top, and the selection is not a setting:
it is how the engine drafts.

| Part | What it is | State |
| --- | --- | --- |
| N-gram drafter | Continues the longest earlier occurrence of the current text, in this request or another: one suffix automaton over a corpus that all sequences share (`NgramCorpus`, `NgramDrafter`; after TensorRT-LLM, SGLang and vLLM's suffix decoding), with a confidence per proposed token and a bound on memory. It replaces a match of the last tokens per sequence; llama.cpp's map and cache variants are not built | Done and measured on text; the conductor uses it, next to a model drafter and never alone |
| MTP | The model's own draft block (`mtp`: one more layer of its own kind) | Done for both models on one driver (`blocks/mtp.rs`, see "The MTP driver" below) in every placement, from the model file (`--drafters mtp`) or a file of the block (`--drafters FILE`, architecture `qwen35-mtp` or `qwen4exp-mtp`): the block runs on the device with the first rows of the output head, its chain and its picks stay on the device, and a draft vocabulary is the head view of every drafter. Both make their chains, of up to the window's rows less one (7 by default), in a step that only drafts, and the round verifies what pays of them |
| DFlash, DSpark, DFlash2 | A small draft model that reads the target's hidden states at a few layers and proposes a block of tokens at once | DFlash2 built as a kind of drafter for both models, chains only (`--drafters FILE` with a file of architecture `dflash`; see "Drafters as kinds" below). DSpark and DFlash 1 files are refused by name |
| Drafter selection | Per step and for all sequences together, whose drafts go into each window, by expected tokens per millisecond of the whole step, from the price of a step and the acceptance measured per drafter, position and sequence; a proposal that was not taken learns from what turned out right | Done: `ares_sched::choose` |
| Adaptive draft length | The length that maximises the same number | Done, in the same call; and a chain ends as it is made where one more link does not raise that number, by the probability the drafter gave its drafts (see "The chain is cut by the drafter's own probability") |
| Draft trees | Several branches verified in one window | Not built; the choice is made over rows ranked by their chance, which is a tree's ranking (see `crates/sched/src/window.rs`) |
| Verification | Windows of any length for any number of sequences in one step, then commit | Done |

All drafters produce the same thing, a list of proposed tokens with a chance
that each is accepted, so selection and verification do not depend on where a
draft comes from. An n-gram draft is written into the step record by the host.
The draft of a head or a draft model stays on the device, in a table the
verify window reads (SGLang), and reaches the host together with the picks;
carrying it through the host would put a copy out and back between two
kernels of the same device at every step. A draft is cut at the first token
the grammar refuses before it is verified (vLLM).

**The policy.** Four decisions, made as the user decided them and none of them
a flag or a constant of this machine:

- *Drafting is always on.* The operator chooses which **model** drafters take
  part (`--drafters`: the model's MTP block, `mtp`, by default if the
  file has one; `none`; later a block drafter from a file of its own) and the
  room for windows (`--window`). The n-gram drafter costs nothing to
  ask and takes part as soon as at least one model drafter does, never on its
  own: a server with no model drafter (`--drafters none`, a file without a
  MTP block, a layer split, which has no MTP block) does not speculate,
  and says so in the plan.
- *The price of a step is a table*, by its rows, because the weight kernels
  read their rows in launches of eight and a ninth row is a second launch
  (25.4 ms of weights for 8 rows of the 27B, 31.0 for 16). `Conductor::warm_up`
  measures it on the devices at load: steps of every row count a window can
  have, shaped as the steps are (windows of several rows that keep their
  inputs, prompt rows that are certain), then the doublings up to a piece of
  prompt, the fastest of two (one above 24 rows), and windows whose draft
  rows the block fills, from 2 rows to the longest window, for the time of its
  chain by length. The steps that run keep the table true (see "Prices that
  stay true"); a context term (per cell a row reads) is fitted separately
  (`ares_sched::PriceTable`).
- *Acceptance is measured per drafter, per position and per sequence*
  (`ares_sched::Acceptance`): position `i` is accepted *given* that the ones
  before it were, so a draft that failed at position 2 says nothing about
  position 3. A sequence that is new starts from what the engine has seen of
  that drafter at that position (and that from a prior of 0.6 before it has
  seen any), and its own counts, which forget gradually, outweigh the start
  as they come. The n-gram drafter's chances are its confidence per token
  (match length and how earlier occurrences went on), which verification
  corrects as it always did.
- *The choice is one number*, tokens per millisecond of the whole step: every
  draft row is worth the chance that it and the rows before it are accepted,
  rows are ranked by that chance across all windows of the step, and the
  step takes the first `m` of them, for the `m` at which the rate is highest.
  The price has edges, so every `m` is tried. A drafter that runs on the
  devices pays its chain once for all windows that use it, however long the
  longest is; which drafter a sequence uses is tried one sequence at a time
  and kept when the rate of the step rises, and lengths are then tried one
  shorter and one longer. In 300 random cases of two sequences, two drafters
  and chains of up to five rows it is the best of all choices in 293 and
  never below 97.9 % of it.

**Prices that stay true.** A price measured once goes stale, and a price that
is too high keeps its own size from running, so nothing corrects it: the
first table of Flash-Next is measured before its tiers hold any expert and
kept every window from being chosen, and on the 27B one late step under CPU
load moved the price of its row count a fifth of the way to a value ten to a
hundred times too high. The table (`ares_sched::Curve`, for the rows of a step
and for the links of a chain) is therefore built on two rules.

- *The price of a size is the mean of its last eight measurements without the
  largest* (the smallest of up to three). Other work on the machine only adds
  to a step, so one late step in eight moves nothing. It is a mean and not a
  median because part of what a step takes is its own and comes unevenly: a
  step of Flash-Next that misses experts takes several times as long as one
  that does not, the more often the more rows it has, and a size gives its
  tokens over its mean time (a first version priced a size at the second
  smallest of five, and the server then chose windows whose usual step was
  cheap and whose mean was not). A size that changed is followed as its
  measurements are replaced, wholly after eight; the fit of the context term
  takes no step that took more than twice its price. A model whose steps
  wait for what they fetch (`Model::steps_wait_for_fetches`: Flash-Next,
  whose experts in no VRAM tier are brought while the step waits) is priced
  by the plain mean of the eight (`ares_sched::Estimate::Mean`): there the
  dearest step is not late, it is the one that fetched, and what a size
  fetches is what it costs.
- *A size that has not run for 256 measurements of the table is old.* Its
  price is held between the prices of the nearest sizes on either side that
  are current (fewer rows do not cost more, more rows do not cost less), and
  the choice of a window is made at the price to *try* it at
  (`Curve::trial_ms`): that of the nearest smaller size that is current, as if
  its extra rows were free. It is then taken whenever its rows could pay at
  all, the step measures it, and the measurement stands beside the old price,
  the smaller of the two being the price; it is tried a second time, so that
  one late step among the two does not waste the turn.

No size can be locked out by this. One late step moves nothing. A price that
is too high for any other reason (two late steps close together, a cold
start, a busy stretch, a text that missed its experts) is old after 256 steps
in which the size did not run, whatever it is, and one step at the trial
price replaces it. A size that is
not taken even at the trial price cannot pay at any price, so its price does
not matter. Trying costs at most two steps per size in 256, and they are
not lost: their rows verify drafts. The share of prompt and generation
(`Share`) reads the plain prices, so a trial price never shortens a prompt's
turn.

The measurement at load stays. It gives every size a price before the first
request, taken together under the same conditions, which is the shape of the
table (where a launch ends), and it pays the first use of every kernel. Its
level is not trusted: on cold tiers it is too high for every size, the one row
that runs comes down with its first steps, and the windows are tried again
after 256 steps (`windows_priced_out_on_a_cold_start_are_tried_again_and_chosen_when_they_pay`
in `crates/runtime/tests/conductor.rs` states exactly that device).

**The chain is cut by the drafter's own probability.** The acceptance per
position is an average: it says that the third draft is right six times in
ten, not whether this one is. The drafter knows more. The probability it gave
a draft (`SamplePick::confidence`: the share of the pick among the row's eight
best logits, which the sampler reports for every row) is counted against what
verification found (`ares_sched::Confidence`, sixteen ranges of probability,
starting from the probability itself and moving to what was observed), and
that gives the chance of a draft the drafter has just made.

A chain is made link by link, so the rule is applied where the chain is made.
One more link is worth the chance that its draft counts and costs its time,
and it is made when that raises the rate of the step, the number the windows
are chosen by: `reach * mean >= rate * ms`, where the reach is the product
of the chances of the drafts made so far, `mean` the chance of a draft whose
probability is not known yet (the average over the drafts verified), `rate`
the tokens per millisecond the step was chosen for and `ms` what the curve
says the link takes. The conductor writes the table of chances and, for every
draft row, the reach its link needs (`ares_sched::need`) into the step record
("draft bars"); the drafter multiplies the chances of its picks and ends the
chain at the first link whose bar the reach misses. There is no threshold of
its own in this: a link that costs nothing is always made, and one that costs
more than a token never.

A link that is not made comes back as no draft (`u32::MAX`). The choice of
the chain's length knows it: a chain is priced by the links expected to be
made (the share of drafts that leave a chain above each bar), and a draft row
counts only as often as its link is made behind a draft that was right, so
chains are asked longer than the average draft would carry and the chain
decides per token how far it goes.

**The chain runs in a step of its own, and the round verifies what pays of
it.** A round in which sequences take the model's drafts is two records
(`Conductor::chains_first`). The first only drafts (a flag of the record): its
segments are the windows whose drafts the model makes, as long as their
chains are asked, with the sampler rows their links pick with and the draft
bars; the model makes the chains and nothing else, no row of the model runs
and nothing is left to commit. What the record carries besides (the commit of
the step before, the copies, the page tables, the freed pages) is carried out
as in any record. The drafts come back with what the drafter said of each.
Then the conductor knows every draft, and chooses how many of each chain the
round verifies by the same rate it chooses windows by (`Drafting::shorten`
over `ares_sched::choose`, the chain paid for): a draft is worth the chance of its probability
(`ares_sched::Confidence`) times the chances of the drafts before it, and a
draft the request's rule forbids is cut with the drafts after it, as an
n-gram draft is. The rows of the drafts it does not take are not in the step,
and their cells leave the table. The round verifies the rest as tokens, with
the masks the drafts before a row give it, like the n-gram drafter's drafts: so a
request with a rule takes the model's drafts through the token that makes
the rule bind and is masked from there on (`tests/free_drafts.rs`), and a
request with penalties takes them too, each row with the corrections of its
path (until 2026-11-16 such a request took only the n-gram drafter's drafts). The
round's record also says what each draft row is worth (the record's row
worth: the chance that the drafts up to its own are right times what a token
costs at the rate the round was chosen for), which a model whose rows may
have to wait for the file uses to decide whether a row is worth the wait (see
"A row that verifies a draft waits for the file only while it is worth it"
under [Weights](weights.md)). A row the model leaves (`StepOutput::inexact`)
gives no draft from it on: the conductor accepts at most the drafts the rows
before it tested.

The decision of a window's shape is in one place for every model, the
conductor with `ares_sched::Drafting` (which keeps what the engine knows of the
model's drafters: their acceptance and confidence, the price of their chains,
the bars of their links and the worth of a token, and chooses a round's
windows from it); a model only makes chains (the 27B on the device, link after link
with the picks kept there; Flash-Next with every pick read on the host) and
runs the step it is given. A tree of drafts extends the same choice: its
rows are ranked by the chances along their path (see `crates/sched/src/window.rs`).
The price of a chain is what the step that only drafts took, so the round
trip of the extra step is in it; the price of the round's rows is the round's.

What each drafter does with the bars in the step that only drafts:

- *The MTP driver* (both models) enqueues its chain on the device and copies
  every link's picks to the host. Where a bar can end the chain (the least
  chance in the table would not carry the reach over it) the device thread
  waits for those picks before it enqueues the next link; where no pick could
  end it, nothing is waited for. (Until 2026-11-17 Flash-Next's block read every
  pick on the host.)

**What limited the chain.** In order: the default of three drafts
(`--drafts`), which was a cap on every sequence's window whatever the text;
the step cost as a line, `base + per_row * rows`, which cannot see that the
ninth row costs a launch and so could not decide where a longer chain stops
paying; the whole step capped at the rows of the windows while anyone drafted,
which is also what made a prompt crawl; and the room the model reserves per
row of a window (the inputs it keeps for the commit, 3.6 MB; the logits,
1 MB; a link of the chain's scratch and the carry of the hidden states: about
5 MiB for the 27B together, for every sequence), which is the one real limit
and so the one option, `--window`. Its default is the rows of one launch of
the few-row weight kernels (`ares_cuda::MAX_BATCH`, 8): inside a launch a
longer window costs rows and no weights, and the table measured at load says
what a second launch costs, so a window longer than a launch is only worth
reserving for text that is copied for many tokens in a row. Everything below
it is the policy's: a window of 8 rows is a possible chain of 7 links, and a
step uses as many as pay.

**The n-gram drafter.** An automaton and not a trie of runs, because the
longest match of a sequence's end is then not searched for at all: it is
where the suffix links of the sequence's own state lead, an insertion is
constant work, and a token costs two states and three transitions whatever
the match length, where a trie of runs up to `d` tokens costs `d` nodes and
`d` lookups per token and never matches more than `d`. An automaton cannot
forget, so the corpus is two of them: when the young one is full the old one
is dropped and the young one becomes the old. Text is found for one to two
automata's worth of tokens (64 MiB hold 260 000 to 520 000). Each state
keeps where its most recent occurrence went on and how many of its
occurrences went on, and the confidence of a proposed token is read from a
table by the length of the matched context and by how its earlier
occurrences went on (once; always this way; mostly; seldom). The table
starts from measured acceptance and follows what verification reports.

**The MTP driver** (`blocks/mtp.rs`), one for every model. A model gives it
its MTP layer (`MtpLayer`: the launch plan of a link, and of rows that only
write keys and values; the lag between a pair and the block's cell; the
per-sequence state links change) and the rows where its hidden state is tapped
(the 27B: the final hidden state, `output_norm` of the residual after the last
block, in the step's `xn`; Flash-Next: the residual streams after the last
layer). Everything else is the driver's:

- *Pairs and cells.* The block reads a token and the model's hidden state of
  the position before it; the pair of the token at position `p` is the block's
  cell `p - lag` (the 27B 0, Flash-Next 1). The **carry** of a slot is the
  hidden state of the sequence's last row that counts, with a row of zeros
  behind the slots for a sequence's first pair.
- *Following the model.* At the start of a step, before its copies of the
  prefix cache and before any tile is converted, the rows of the model's last
  step that count (all rows of a prompt slice or a single row, the accepted rows
  of a window, which the record tells) give the block their pairs behind the
  first: rows that only write keys and values, in launches of the layer's size.
  The hidden state of the last row that counts becomes the carry; a replay's
  cells are cached, so it moves only the carry. Rows that were rejected are
  never followed, so nothing the block writes is undone and a layer with state
  per sequence (Flash-Next's indexer) needs nothing else. Until 2026-11-17 the
  27B's block followed every row of a step, rejected ones included, in the step
  itself, and kept every row of a window in its carry until the commit chose
  one; now its cell of a row is written a step later.
- *The newest cell* of every sequence of a step that does not draft only writes
  keys and values; for one that drafts it is link 0 of its chain.
- *Launches that only write keys* (following and the newest cells) are decided
  one at a time, each when the ones before it are enqueued (`Driver::key_launch`):
  a launch of Flash-Next's block prepares its K/V group like a step (the tiles
  its rows write get exact slots, the tiles its sequences left are converted),
  so whether a row's tile is held converted (and the row a replay that writes
  nothing) is only known then. A row behind a row of its sequence that writes in
  the same launch writes too, and the rows of a sequence that write behind rows
  that do not begin a launch of their own, so that every segment is a replay as
  a whole: the K/V group prepares a segment by its first row. Until 2026-11-22
  every row of a step was decided before the first launch: the tiles of a prompt
  slice that no launch had given a slot yet looked converted, so the block wrote
  only the first cell of each tile of a prompt, and where a slice started at a
  cell that is no multiple of eight (behind a hit in the prompt cache) a launch
  that began in such a tile and reached the next wrote that tile's first row into
  its converted entry, which the device reported as status `0x4` and the engine
  ended on. The 27B's block prepares its group once per model step, before it
  follows, so the order changes nothing there.
- *The chain.* Longest chains first, so that the rows of a link are the first
  rows of the link before; the records of every link (the record, the sampler's
  settings, where each pick goes, the layer's plan) are uploaded with one copy.
  Link `j` pairs the pick of link `j - 1` with the block's own output of that
  link; each pick is made by the device sampler from the draft head (the head
  view's first rows of the output head) with the settings and the noise of the
  row that will verify it, and written into the step record's token array, the
  next link's record (whose rows of the embedding table the device keeps) and
  the step's list of drafts. The host reads the picks at the end of the step,
  and before a link only where a bar may end a chain. The cells of links after
  the first are provisional: the layer's per-sequence state is saved behind
  link 0 and put back behind the last, and their keys and values are written
  again when the model has passed them. A bar can end a chain before links the
  driver laid out run, so what a layer keeps for the links of a step (the
  27B's attention plans) is dropped when the next step's links are laid out
  (`MtpLayer::forget_links`); until 2026-11-17 the next chain's first link took
  such a plan and its attention read past its tables (`CUDA_ERROR_ILLEGAL_ADDRESS`
  in the server after a few requests; `qwen35_mtp_device.rs` now cuts a chain
  and runs it again).

The round that verifies the drafts is an ordinary step whose windows carry the
drafts as tokens. The conductor chooses per sequence and step between an n-gram
draft and a model drafter's by expected tokens per millisecond
(`ares_sched::choose`), with the price of a chain taken from the steps that
only draft. The block is on the device with the first rows of the output head:
alone or under a tensor split the device that embeds, under a layer split the
last one (the 27B drafts there too since 2026-11-17), whose final hidden states
it follows; the table on the host is shared by the parts that stage rows from
it. Its weights come from the model file (`--drafters mtp`) or a file of its
own (`--drafters FILE`: `qwen35-mtp` for the 27B, the block's tensors under the
model file's names `blk.<n_layer>.*`; `qwen4exp-mtp` for Flash-Next).

Measured 2026-11-10 on Qwen3.8-27B (tensor split 0.625 / 0.375, RTX 4070 and
RTX 3060, at most three drafts, the fork's draft vocabulary of 66 343 tokens,
300 tokens per request through the chat template with the fork's
`reasoning_effort`): 79, 76 and 79 tokens/s for German prose, English prose
and code with the most probable token, 76, 72 and 83 sampled (temperature 1,
top-k 20, top-p 0.95), against 40.5 without drafts and 62 to 67 for the fork
with its production flags on the same questions. A link of the chain costs
1.46 ms (it reads 335 MiB of the block and 278 MiB of the head), a step 34 ms
for 2.5 to 2.9 tokens; the first draft is accepted in 74 to 83 % of the
steps. The block takes 335 MiB of weights and, with its K and V in `q4` (the only format
it has: 11 000 drafts each with the block's K and V in `q8` and in `q4` were
accepted at the same rate, 67.4 % against 67.1 % for the first draft of a
window and 64.7 % against 65.8 % for the second, greedy, and 70.3 % against
70.3 % and 64.5 % against 66.6 % sampled, `z` below 1.3, and 74.5 / 76.1
against 74.8 / 76.8 tokens per second), 1.1 MiB per 1000 tokens on its device
where `q8` took 2.1 (the same on the `Q4_K_S` file over both cards, 3000
drafts each: first draft 69.9 % against 67.8 % greedy and 63.2 % against 64.9 %
sampled), and nothing for its head, which is rows of the output head.

**The text with drafts is the text without**, token for token, with the fast
kernels as with exact products, for the most probable token and for sampled
tokens with the same seed. A step verifies a window of rows of a sequence, and
row `k` of it gives the logits that a step of that row alone gives after the
`k` rows before it were committed, bit for bit: the weight kernels give a row
the same product in any launch of a decode step, the recurrent blocks carry
their state from row to row as single steps store it, and attention plans
every generated row as if it were the only row of its step (see "Attention
kernels" under [Devices](devices.md)). The tests: every kind of block, every
format of K and V, windows with rejected drafts, on made-up models
(`crates/model/tests/qwen35_windows.rs`: every logit, the recurrent state and
the stored K and V); the 0.8B on one card, in a tensor split and in a layer
split with every format (`a_row_of_a_draft_window_picks_what_it_picks_alone_on_every_placement`
in `tests/qwen35_device_runtime.rs`); the kernels and the plan in `ares-cuda`
(`a_row_of_a_window_has_the_bits_it_has_alone`); and the 27B over HTTP by the
hash of a reply's tokens (`ares-server`'s `tests/real_drafts.rs`).

It was not so before 2026-11-15, with the fast kernels. Measured then with
`qwen35_run --engine 8 --undrafted` (German prose, English prose and code, 300
tokens each, the most probable token and sampled with seed 20261110; every
request with drafts, with `"draft": false` and with drafts again in one
runtime): with K and V in `q8`, the default, 6 of 6 replies of the `IQ2_XXS`
file on the RTX 4070 parted from the reply without drafts, after 213 to 845 of
1100 to 1500 characters, and 4 of 6 of the `Q4_K_S` file over both cards, after
285 to 867; with K and V in `f16` none of 6 parted on either. The second reply
with drafts was the first in all 24. After the change, the same day, with
`q8`: none of 6 parts on the RTX 4070 and none of 6 over both cards.

The cause was attention and nothing else. A probe ran the blocks of a step
one by one (`Harness::blocks`) and fed every block of a window the rows that
the single rows had fed it (Qwen3.5-0.8B, windows of 4 and 8 rows after 200 to
700 tokens): every recurrent block, with its feed-forward half, its norms and
its products, gave a row of a window the bits of the row alone (all 216 and
432 rows) and left the same state after the commit; every attention block
gave other bits (none of 72 and 144 rows, off by up to 7e-3 of the block's
largest output) while the stored K and V were the same bytes. With K or V in
`f16` every row was equal, and with `q4` as long as every tile was still an
exact entry. What depended on the step was the read:

- *The kernel.* One row over quantized K and V in converted entries ran on
  the direct kernel (queries of 15 bits, integer products), a window on the
  tensor cores (f16 operands). The two differ by about 1e-3.
- *The pieces.* A row's cells are cut into pieces whose states are merged in
  rounded sums; the cut followed the last row of the window and every other
  row of the step, so a window that began a new table entry cut its earlier
  rows otherwise than a step of their own would (once a sequence has more
  entries than a row has pieces).
- *With `q7` to `q4`, when a tile is converted.* A step converted the tiles
  that had left the newest cells as of its first row; a later row of the
  window read a tile exact that it reads converted in a step of its own.

The earlier measurements fit this: rows of different sequences in one step
were equal to the rows alone (each is an item of its own, on the direct
kernel), rows of one sequence were not, and the difference did not grow with
the context.

### Drafters as kinds

A drafter is a kind the runtime knows, not a feature of one model. The three
kinds have one name each everywhere (types, modules, files, options,
statistics, logs): `ngram`, `mtp` and `dflash`. The n-gram drafter is the
runtime's own and has id 0 (`ares_sched::NgramCorpus` and `NgramDrafter`: the
longest earlier occurrence of the text, on a suffix automaton); a model lists
the drafters it runs on its devices, its model drafters (`Model::drafters`: a
kind and the most drafts it makes per sequence and step), and the record names
them by their place in that list plus one. The model drafters' drafts are the
model drafts (`DeviceModel::model_drafts`). Two kinds of model drafter exist:

- **`mtp`** (`DrafterKind::Mtp`): the MTP block, one more layer of the model's
  own kind (from the model file, `--drafters mtp`, or a file of its own), a
  chain of links on the MTP driver, each of which costs device time and can end
  the chain by the drafter's probability (the draft bars).
- **`dflash`** (`DrafterKind::Dflash`): a drafter model from a file of its own
  that reads taps of the model; one pass makes every draft of a window, so a
  draft more costs no device time once the pass runs.

`--drafters` names them comma-separated: `mtp`, or a file name, which the
server takes as an MTP block or a DFlash drafter by the architecture the file
names (and refuses otherwise); `--draft-vocab` is the head view of every model
drafter.

A window whose drafts a model drafter makes has the model-drafts flag in its
segment and the drafter's id in word 6. The conductor keeps per drafter a curve
of what its drafts cost (measured at warm-up for 1 to the longest window, then
by the steps that only draft), an acceptance per position (per sequence and of
the engine) and a confidence by the drafter's probability. `want` offers every
drafter's chances for every sequence, `ares_sched::choose` takes the drafter
and the length that raise the rate of the whole round most, and one step that
only drafts makes the drafts of every window that takes a model drafter (a
step in which two sequences took two drafters prices both curves with the
step's time). The device part keeps one list of drafts per step, written at
each window's place in segment order, whichever drafter made them; the
statistics count drafts by source (ngram, mtp, dflash).

### The DFlash drafter

`ares_model::dflash` is the drafter for every model that has a DFlash2 file.
What it is (from the fork's `src/models/dflash.cpp`, `common/speculative.cpp`
and the DFlash study `/var/tmp/ares-dflash-study/REPORT.md`): five layers of
the model's width with no embedding table and no head, which are the model's.
For every computed token the model hands over its residual stream before the
target layers (the *taps*); the drafter fuses them (`fc`, a norm) and stores
every layer's key and value of the result at the token's position. A pass
takes the block `[newest token, mask x 7]` at the next positions, runs it
through its layers (RMS norm, a dynamic convolution of two taps over the
block before and after attention and the feed-forward part, attention that
is not causal inside the block over the stored cells of the last 2048
positions and the block itself, SwiGLU), normalises, multiplies by the model's
draft head, keeps the 16 best tokens of rows 1 to 7, and scores each candidate
after each candidate of the position before through the selector (a gate of
256 values per row, rows of two tables of 256 values per token). The chain is
the greedy walk through that lattice: at each position the best score after
the pick before; its probability is the pick's share of the softmax of that
row, which the conductor reads like the MTP block's.

**What is shared and what is the model's.** The loader reads the shape from
the file (block size, taps, selector, convolution, rotary base and sections,
window, mask token) and refuses a variant it does not run by its name: DSpark
(a Markov head, yarn rotary, drafts from the anchor), DFlash 1 (no selector),
LiLiCorr, the DeepSeek-V4 and Mistral Large 4 drafters, gated attention, a
vocabulary or head of its own, a norm per tap. Everything stateful is the
drafter's: the fusion, the ring, the pass, the selector, the walk, the list.
A model gives three things:

| | the 27B (`qwen35/device/part/dflash.rs`) | Flash-Next (`qwen4exp/device/part/dflash.rs`) |
| --- | --- | --- |
| tap | the residual `x` before each target block; under a tensor split every device holds it after the merge, the drafter's device reads its own | the mean of the four hyper-connection streams before each target layer: the residual after the read of the layer's first mixer (the read applies the write before it), `hc` streams of `n` values per row |
| embedding of a token | its row of the table in page-locked host memory, decoded on the host | its row decoded on the host from the mapped file |
| draft head | the draft plan's (the first rows of the output head on the device that embeds, the draft vocabulary when there is one), in its own layout and products | the output head in the file's layout, the drafter's products |
| device | the draft plan's device: one device, or a tensor split | the part that holds the streams and the head: one device, or the mixer of two devices on every layer |

**Inject.** A tap goes through its share of `fc` as soon as the model's layer
gives it (the fusion is split into one matrix per tap at load), so no tap is
kept; after the model's last layer the sum is normed and every layer's key
(per-head norm, rotary turn) and value are written into the ring of the row's
sequence. Every row of every step is injected: prompt rows, replayed rows, and
a window's draft rows too. The ring has `2048 + max(block, window rows)`
cells per layer and slot, position `p` in cell `p mod cells`, so a rejected
row's cell lies at a position at or past the newest token, which the next
pass does not read, and maps to no cell the pass reads; the row accepted there
later overwrites it. This is the study's "inject the kept rows at the commit"
without keeping taps across steps, at the cost of fusing the rejected rows
(the weights are read once for all rows of a step either way). The host keeps
for each slot which positions its ring holds since the sequence took it; a
segment that does not continue them (a new sequence, a sequence that resumed
from the prefix cache) starts it again, and a pass reads no cell before that.

**Pass.** In the step that only drafts: the newest token's row and the mask
token's row (decoded once at load) are uploaded with the pass's records, the
layers run on the decode kernels (8 rows a sequence) with 8-bit activations
like the model's own decode products, the head is the model's, and the best
16 per row, the selector's gate, the scores and the walk are kernels of
`kernels/dflash.cu` (one block per sequence for the selector, which decodes
the table rows it needs). The drafts go into the step record's token array and
the step's list at the window's place.

**Memory.** On the drafter's device, in its ledger before the pool is sized:
the weights as the file stores them (27B: 1080 MiB; Flash-Next: 335 MiB, from its `Q4_K_M` file), the
ring (27B: 5 layers x 2064 cells x 8 heads x 128 x K and V in f16 = 42 MiB per
sequence slot; Flash-Next the same), the scratch (the pass of every slot at
once, the logits of the head for 7 rows per slot, the fusion of a step of the
most rows). The server counts all of it in the plan of that device: a start
that cannot fit is refused by the ledger with what holds the device.

**Not built.** Trees (steps 9, 10): the lattice is computed and kept per pass
(`Drafter::lattice`), the walk takes its first-choice path only. The ring in
the prefix cache's checkpoints: a sequence that resumes starts its ring at the
resume point. Flash-Next under a split by layers (the first device's taps
would have to cross to the head's device). Sampled drafts (the walk is greedy;
the target's sampled picks verify them as any other drafts).

The drafter's own pass on the device is held to the CPU reference (`tests/dflash.rs`): with the
reference's activations rounded as the weight kernels round them, queries, keys and values are equal
to 3e-5, attention to 4e-4 (the ring is f16), the input of the feed-forward part to 2e-3, and after
five layers the hidden states to 1.4 % (27B) and 3 % (Flash-Next), every draft of a pass is the
reference's and 14 to 16 of the 16 candidates per position are shared.
