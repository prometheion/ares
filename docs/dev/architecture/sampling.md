# Sampling

A request can set what the model's logits go through before a token is drawn
from them, and ask for the probabilities of what was drawn. All of it runs on
the device, in the kernels of `ares-cuda` (`kernels/sampler_*.cuh`) or of
`ares-vulkan` (`shaders/sample_*.comp`, see [Vulkan](#vulkan)), and is
defined by `ares-sampler`, a crate with no device in it: the settings of a
row (`SampleParams`, `SampleRow`), the arithmetic every backend repeats
operation by operation (`arith`), the pick of one row written from its
definitions (`sample_reference`) and the corrections the host works out
(`penalties`). Every test of the kernels compares the records they write with
the reference's bit for bit; Vulkan's are held to the same.

**What a row carries.** Each wanted row of a step has a sampler row in the step
record (128 bytes: position, seed, the stages of its program and their
settings, where its corrections and its mask are, what it reports), and may
have corrections (16 bytes each, at most 4096 a row) and a mask (a bit per
token). The logits are corrected in place before anything reads them. A model
that drafts for itself on the device picks its drafts from the logits as they
are (`SampleRow::for_draft`): a draft is a guess that the verifying row
confirms.

**Corrections.** An entry changes one token: `l = l + bias`, then
`l = l <= 0 ? l * rep : l / rep` (llama.cpp's repetition penalty, which
lowers a logit on either side of zero), then `l = l - sub`; a token the mask
does not allow becomes minus infinity. The host emits them from the row's own
path, the prompt and the reply up to the row's input token and, for a row of a
window that verifies drafts, the drafts before it. That is what makes a window
pick what the sequence picks one token at a time. They are:

- the repetition, presence and frequency penalties of llama.cpp over the last
  `repeat_last_n` tokens of the prompt and the reply (64 unless set, at most
  1024, 0 off). OpenAI counts presence and frequency over the generated tokens
  only; this server counts the window, as llama-server does.
- DRY, ported from llama.cpp's algorithm (the Z algorithm on the reversed window
  and the sequence breakers, which are found once per list of strings by a
  pass over the vocabulary's text) with the same defaults (multiplier 0, base
  1.75, allowed length 2, window 64, breakers `\n : " *`). The penalty is
  added to `sub`, so it is subtracted after the repetition penalty has
  scaled the logit where llama.cpp subtracts twice; the difference is one
  rounding.
- a logit bias: a token and a number, `false` for a token that must never be
  picked, or a text, of which every token gets the bias; `ignore_eos` is a ban
  on the tokens that end a generation.

A request with corrections that depend on the path (penalties, DRY) takes no
drafts from the model's MTP block: those are made on the device while the
step runs, and the host cannot correct a row for drafts it does not have yet. It
takes the n-gram drafts, which the host has, wherever the model has an MTP block
(the n-gram drafter joins next to a model drafter and never runs alone). A request with
a mask is in the same position only while its rule binds: where the rule leaves
the reply free it takes the model's drafts like any other request (see "Drafts"
under "Structured output" below).

**The chain.** A sampled row starts with the set of every token that has a
finite logit, ordered by the distance below the best logit (a token's *key* is
the bits of that distance, which order like it). The row's program is a list
of at most 16 stages; each cuts the set to the tokens with keys from `lo` to
`hi`, or scales the temperature `T` the logits are read at. `x` below is the
distance below the best token of the set divided by `T`; a token's probability
within the set is `exp(-x)` over the sum of that.

| Stage | Keeps | Found on the device by |
| --- | --- | --- |
| top-n-sigma `n` | `d <= n * sigma`, sigma the standard deviation of the distances `d` of the set | sums of `d` and `d^2` as integers |
| top-k `k` | the `k` most probable and every token tied with the last | radix descent over the keys, by count |
| typical `p` | the tokens within the smallest radius of `x` around the mean of `x` (weighted by probability) that holds more than `p` | sums, then a descent over the radii by weight, then the least and greatest key inside |
| top-p `p` | the most probable that hold at least `p` of the set | radix descent by weight |
| min-p `p` | at least `p` times as probable as the best of the set | one threshold, no pass |
| XTC `prob`, `thr` | with probability `prob`: removes the tokens more probable than the least probable that has `thr` or more, if there are two | a draw, sums, a scan |
| temperature `t` | sets `T = t * T`; a `t` below 1e-6 keeps the tokens tied for the best | arithmetic; a dynamic temperature (range `r`, exponent `e`) first sums the entropy `H` of the set and takes `t` between `max(0, t-r)` and `t+r` by `(H / ln n)^e` |
| mirostat 2 | the tokens whose surprise `-log2 p` is at most `mu`, or the best | sums, then a search of the key |
| mirostat 1 | the top-`k` of the `k` that the hundred most probable tokens and `mu` imply | descent for the hundredth key, a gather of the keys below it, arithmetic, descent for `k` |

The order of the program is the request's `samplers` (llama.cpp's default
order: penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC,
temperature); a stage that is not listed is not applied, one that is listed
twice is applied twice. Penalties and DRY correct the logits and so come
first or not at all. A stage whose setting switches it off is left out of the
program. Mirostat replaces the chain, as it does in llama.cpp, which then
applies neither penalties nor DRY; the logit bias applies to every chain.

The pick is the token of the final set with the largest `logit / T + noise`,
the noise being Gumbel of (seed, position, token). A row with temperature 0
(and no dynamic temperature or mirostat) takes the most probable token of the
corrected logits, the lower token on a tie, whatever the stages are.

Where this differs from llama.cpp: tokens tied at a cut are all kept (llama.cpp
keeps the first by its sort order); the random numbers are not its generator's,
so a seed does not give the text llama.cpp gives, only the same text every time
here; a temperature of 0 takes the best token even where a stage before the
temperature would have removed it; the probabilities are fixed-point sums, so
a cut can fall one token away from where llama.cpp's float sums put it when a
token sits within 1e-6 of it; `min_keep` above 1 is refused (a cut would need
the count of the tokens it left); adaptive-p is refused (see below); a window
longer than 1024 is refused.

**What a field is when a request leaves it out.** The model's default, if
the server was given one for that field (`--temperature`, `--top-p` and the
others: "The defaults of a model's requests" under [Server](#server)), and
the engine's own value otherwise: temperature 0, which takes the most probable
token; no top-k, top-p 1, min-p 0, typical 1 and top-n-sigma 0, which cut
nothing; XTC and the dynamic temperature off (threshold 0.1, exponent 1); the
three penalties off (1, 0, 0) over a window of 64; DRY off (multiplier 0, base
1.75, allowed length 2, window 64); no mirostat (tau 5, eta 0.1 when a request
switches it on); llama.cpp's order; a seed of its own for every request. The
engine reads a request in which a field is either a value or open
(`ares_api::Sampling`), and the server fills the open fields from the model's
defaults before it hands the request over, so the engine does not know where a
value came from.

**Accuracy.** Everything that is summed over tokens is an integer (a count, the
weights as fixed-point numbers with 40 fractional bits, the entropy term, the
sums for the deviation), so the order in which threads, blocks and devices add
does not matter; the exponential and the logarithm are polynomials of IEEE
operations, in single precision per token and in double precision per row. The
cross-device sums are numbers split into limbs of 16 or 20 bits that add
exactly as `f32`. That is why the device's records equal the host's bit for
bit.

**Mirostat and drafts.** Mirostat's `mu` moves with every token by the
probability of the token picked, which a row of a window cannot know before the
row before it is picked. A request with mirostat therefore has windows of one
row; its `mu` is the host's and the device reports the surprise of each pick.
Adaptive-p has the same dependence and is not built.

**Drafts and the text.** Every feature gives the same tokens, text and
probabilities with drafts as without them (a request that says `"draft": false`
is the one to compare with), and a request does not change what the requests in
the same steps get. The synthetic model of `ares-runtime`, whose device part
drafts what it has seen the model pick, proves it bit for bit on every
placement, with a rule about the tokens and with a grammar that ends its reply
as well
(`every_sampler_feature_gives_the_same_tokens_with_and_without_drafts_on_every_placement`
and `requests_with_different_settings_in_one_step_do_not_affect_each_other_on_every_placement`
in `crates/runtime/tests/gpu.rs`), and `crates/runtime/tests/sampling.rs` checks
the records of the windows against the definitions. The n-gram drafter
drafts only next to a model drafter, so the tests on a real model use the 27B and
its MTP block, on both cards with exact products. There the texts were equal
with drafts and without in all 14 settings through the server
(`crates/server/tests/real_sampling.rs`: 437 draft rows verified, 425 of them
right), equal under two rules about the tokens, and equal for requests that run
next to each other. Since 2026-11-15 a row of a draft window has the logits of
the row alone bit for bit with the fast kernels too (see "The text with drafts
is the text without" under [Speculation](#speculation)), so the same holds as
a model is served. What a row still depends on is the steps that computed its
prompt and, past 16 generated rows in a step, the size of the step ("Rounding"
under "Sequence state"): where that shows, a sampled text can part where two
tokens are that close, and the tests of requests next to each other assert
that the texts agree for a long stretch and that the probabilities agree on it
within the model's error.

**Probabilities.** A request asks for the probability of each generated token
and of the `top` most probable alternatives (at most 20). Two distributions:

- *the model's own*: the softmax of the logits as the model gave them, before
  the corrections and the chain (OpenAI's `logprobs`, llama.cpp's `n_probs`;
  vLLM's default). The device takes the best twenty and the sum of the
  weights from the uncorrected logits before it corrects them, and the pick's
  uncorrected logit from the entry that corrected it.
- *after the chain*: the probability within the tokens the chain left, at the
  temperature it ended with (llama.cpp's `post_sampling_probs`). A row that
  takes the most probable token has the tokens tied for it.

OpenAI says nothing of a distribution; vLLM and llama.cpp take the model's own
by default, and so does this server. A probability whose logarithm is below
-100 is not reported. The probabilities of a token go out with the text of
the token, once the characters are whole and no stop string holds them back; a
reply that ends at its token limit inside a character reports the tokens of the
half character with the last delivery, so that it reports as many tokens as it
has (one that a stop string cuts does not report the tokens of the string).

**Cost.** The time of the sampler's kernels for one step with 248 320 logits a
row, from the launch to the end of the stream, in microseconds: the median of
40 picks, for 1 / 4 / 16 rows. `vk_sampler_bench` (an example of `ares-vulkan`)
runs both backends on the same card with the same rows and the same method;
`the_time_of_a_pick` in `crates/cuda/tests/sampler.rs` prints CUDA's alone.
Corrections are 64 entries a row, a mask allows a third of the tokens,
probabilities are the five best. Measured on 2026-11-12, each backend alone on
one card: three runs of the benchmark with CUDA's kernels as they were
(commit 6c27759) and three with the kernels described below, in pairs back to
back, while other work compiled and other processes used the cards between
the runs (load average 4 to 10). A figure is the median of its runs (six
for Vulkan, whose sampler is the same in all); the run furthest from it was
35 percent away for Vulkan, 31 for CUDA before and 30 for CUDA now.

RTX 4070:

| Row asks for | Vulkan | CUDA before | CUDA now | now / Vulkan |
| --- | --- | --- | --- | --- |
| most probable token | 32 / 49 / 128 | 36 / 76 / 244 | 23 / 29 / 57 | 0.71 / 0.59 / 0.45 |
| top-k 20, top-p 0.95 | 139 / 196 / 535 | 123 / 203 / 545 | 98 / 127 / 228 | 0.71 / 0.65 / 0.43 |
| top-p 0.95 | 93 / 138 / 394 | 84 / 148 / 431 | 72 / 107 / 251 | 0.77 / 0.77 / 0.64 |
| most probable token, penalties | 36 / 50 / 111 | 41 / 82 / 245 | 27 / 32 / 57 | 0.76 / 0.65 / 0.51 |
| most probable token, a mask | 37 / 56 / 150 | 44 / 91 / 266 | 30 / 39 / 75 | 0.81 / 0.70 / 0.50 |
| top-k, top-p, penalties and a mask | 145 / 200 / 498 | 129 / 207 / 553 | 105 / 137 / 258 | 0.72 / 0.68 / 0.52 |
| most probable token, probabilities | 95 / 171 / 520 | 103 / 225 / 682 | 60 / 92 / 209 | 0.63 / 0.54 / 0.40 |
| top-k, top-p, probabilities after the chain | 180 / 262 / 692 | 163 / 260 / 691 | 123 / 160 / 292 | 0.68 / 0.61 / 0.42 |
| top-n-sigma, typical, top-k, top-p | 228 / 342 / 865 | 192 / 304 / 799 | 175 / 239 / 484 | 0.77 / 0.70 / 0.56 |

RTX 3060:

| Row asks for | Vulkan | CUDA before | CUDA now | now / Vulkan |
| --- | --- | --- | --- | --- |
| most probable token | 80 / 110 / 240 | 70 / 151 / 411 | 42 / 77 / 119 | 0.52 / 0.70 / 0.50 |
| top-k 20, top-p 0.95 | 240 / 459 / 1154 | 170 / 407 / 1169 | 137 / 243 / 602 | 0.57 / 0.53 / 0.52 |
| top-p 0.95 | 184 / 316 / 842 | 132 / 290 / 897 | 107 / 201 / 543 | 0.58 / 0.64 / 0.64 |
| most probable token, penalties | 70 / 108 / 223 | 64 / 145 / 409 | 37 / 63 / 110 | 0.53 / 0.59 / 0.49 |
| most probable token, a mask | 87 / 127 / 318 | 76 / 170 / 483 | 48 / 90 / 186 | 0.55 / 0.71 / 0.59 |
| top-k, top-p, penalties and a mask | 209 / 420 / 1165 | 167 / 422 / 1217 | 141 / 269 / 676 | 0.67 / 0.64 / 0.58 |
| most probable token, probabilities | 178 / 366 / 978 | 156 / 448 / 1267 | 97 / 183 / 451 | 0.54 / 0.50 / 0.46 |
| top-k, top-p, probabilities after the chain | 286 / 554 / 1505 | 215 / 508 / 1437 | 160 / 320 / 745 | 0.56 / 0.58 / 0.50 |
| top-n-sigma, typical, top-k, top-p | 346 / 692 / 1894 | 252 / 613 / 1772 | 227 / 478 / 1178 | 0.66 / 0.69 / 0.62 |

*Why CUDA's kernels were the slower ones.* The two backends ran the same
algorithm, statement for statement, and Vulkan's was up to 2.2 times faster
where a step picks for several rows (the table: 111 against 245 microseconds
for the most probable token of 16 rows with penalties on the RTX 4070). The
device time of each part of a pick (`sampler_steps`, an example of
`ares-cuda`, records an event on the stream after every part) showed where it
went: of the 244 microseconds of the most probable token of 16 rows on the RTX
4070, 225 were the kernel that finds the best eight logits of each chunk of
4096, sixteen times what reading the rows takes (a kernel that only reads the
16 rows and takes one maximum per chunk needs 14 microseconds, 51 on the RTX
3060). It was instructions and barriers, not memory: every thread kept a
sorted list of its best eight among 16 logits, which the CUDA compiler puts
into the thread's local memory, and the block took its best eight in eight
rounds of a reduction over all threads, 17 barriers and 160 shuffles a thread.
The kernel with twenty candidates (the probabilities) took 281 microseconds
the same way. The levels of a radix descent came next: each built a histogram
of 1024 or 2048 bins in shared memory in every chunk, cleared before and added
to the row's after, 38 microseconds a level for 16 rows however few tokens the
level had. What the Vulkan compiler makes of the same statements is faster;
where exactly was not looked for, because both did work that is not needed.

*What CUDA's kernels do now*, for the same records bit for bit:

- **The best of a chunk.** A thread reads its 16 logits into registers and
  finds its own best. The threads with the same lane are a column; the eighth
  best of the 32 columns' bests (the twentieth, for the probabilities) is a
  bar that at least eight logits reach, so the best eight of the chunk do.
  Few threads have a best that reaches the bar; each counts how many of them
  have a better one, and the at most eight with fewer than eight above them
  put their logits that reach the bar on one list, where every entry finds
  its place by counting the entries above it. Four barriers, no list per
  thread. The kernel alone, 16 rows: 225 to 39 microseconds on the RTX 4070
  and 354 to 86 on the RTX 3060; with twenty candidates 281 to 40 and 591 to
  97. Rows made to be hard for it (every logit equal, descending, the best
  tokens all in the columns of one thread) take 36 to 53 microseconds on the
  RTX 4070, less than before in every case, and are in the tests
  (`rows_whose_best_tokens_crowd_together_equal_the_host`).
- **A thread's logits at once.** A kernel that passes over a row read a
  logit, used it and read the next, so a thread waited for the memory sixteen
  times. The top, histogram, pass and draw kernels read the thread's 16
  logits before they use one. (The mask kernel got slower that way, 21 to 31
  microseconds for 16 rows on the RTX 4070, and is as it was.)
- **Histograms of few tokens.** The kernel that decides a level leaves in the
  row's state how many tokens the bin it chose holds, which are the tokens of
  the next level, and after the last level how many the set still has. A row
  with at most 256 tokens for a histogram adds them to the row's histogram
  directly, with no histogram per chunk; after top-k 20 that is every level
  that follows. The numbers choose how a histogram is built and nothing
  else. A descent by count (top-k, the hundredth token of mirostat 1) holds
  no weights. The six levels of top-k 20 with top-p 0.95 for 16 rows: 254 to
  176 microseconds with the direct histograms and 175 to 115 with the reads
  on the RTX 4070 (two pairs of runs, each back to back), 673 to 540 and 539
  to 390 on the RTX 3060.
- **The sums of a pass** meet at one barrier, not ten: 28 to 17 microseconds
  for the pass of the probabilities after the chain, 16 rows, RTX 4070.

Two things that were tried made nothing faster and are not in: the running
totals of the kernel that decides a level by shuffles in place of its 16
barriers (35 microseconds for six of them either way), and a count of the
level's tokens per chunk in front of every histogram (the dense histograms
got 5 to 10 percent slower).

*Where the time of a pick is now.* Device time of the parts, 16 rows, RTX 4070
/ RTX 3060, in microseconds (`sampler_steps`, one run):
| Part | RTX 4070 before | RTX 4070 now | RTX 3060 before | RTX 3060 now |
| --- | --- | --- | --- | --- |
| the best eight logits | 206 to 230 | 35 to 41 | 358 to 369 | 78 to 81 |
| the best twenty of the model's own logits | 289 | 44 | 637 | 101 |
| the best twenty of the set after the chain | 81 | 34 | 194 | 74 |
| the six levels of top-k 20 with top-p 0.95 | 230 | 130 | 669 | 388 |
| the three levels of top-p 0.95 alone | 143 | 124 | 389 | 306 |
| deciding six levels | 35 | 42 | 55 | 72 |
| the draw after the cuts | 17 | 16 | 61 | 55 |
| the pass of the probabilities after the chain | 28 | 17 | 80 | 59 |
| the pass that weighs every token | 90 | 90 | 209 | 212 |
| a mask | 22 | 22 | 75 | 74 |

A level with few tokens takes 19 to 25 microseconds on the RTX 4070 and 63 to
71 on the RTX 3060; the first level of top-p alone, where every token has a
weight, 69 and 140. Deciding a level is the one part that got slower: 6 to 8
microseconds now on the RTX 4070 and 9 to 24 on the RTX 3060, where the second
level of a descent by weight takes 24. The kernel got the count of the chosen
bin to write and reads its bins at once; which of the two costs the time was
not found.

A kernel that passes over the rows is now within 1.3 to 3 times of reading
them, and a pick with cuts is the number of its passes: eight for top-k with
top-p. After the first cut the set is a few dozen tokens, and the later
levels, passes and the draw still read every logit to find them; a list of
the set's tokens, written when the first cut is made, would make them
independent of the vocabulary. That is not built. For one row a pick is its
launches: 3 microseconds each on the RTX 4070 and 5 on the RTX 3060, 22 of
them for top-k with top-p.

*In a decode step.* The 27B with its MTP block and the vocabulary list on
both cards (`qwen35_run --engine 8 --generate 300 --sample`, seven prompts,
windows of up to 16 rows, the logits of a row split between the cards; the
kernels before and after in runs back to back, two pairs, load average 4 to 21):
the sampler's part of a step that picks the most probable tokens went from
0.077 to 0.036 ms on the RTX 4070 and from 0.51 to 0.47 ms on the RTX 3060
(the median over the prompts of the median of their steps), and a link of
the draft chain, which ends in a pick over the vocabulary list, from 1.49 to
1.48 ms. A step takes 31 to 41 ms by the number of links it drafts, and the
median of a prompt's steps moves between those values from run to run; for
the five times a prompt's steps drafted the same number of links in both
runs of a pair it was 0.16 to 0.25 ms lower in four (40.63 to 40.38, 31.44
to 31.28, 35.88 to 35.68, 33.77 to 33.54 ms) and 2 ms lower in the fifth, a
jump between two of those values.
That is 0.5 to 0.7 percent of a step, of which the sampler was 1.5 percent;
tokens per second did not change measurably (97.8 and 98.2 on average). With
top-k 20 and top-p 0.95 the sampler's part depends on the rows of the
window: for the three prompts with the same links in both runs it went from
0.61, 0.63 and 0.94 ms to 0.53, 0.53 and 0.82 ms on the RTX 4070 and from
1.03, 1.05 and 1.37 to 0.95, 0.96 and 1.25 ms on the RTX 3060. Most of that
part is not the kernels, which take 0.23 ms for 16 rows of the whole
vocabulary on one card: between the parts of a pick the two cards add up
their partial results, 8192 values a row for every level of a descent. What
that exchange costs was not taken apart here.

A request that sets nothing pays for the best eight logits of its row and
nothing else. Penalties cost a launch. A mask is a pass that writes (22
microseconds for 16 rows on the RTX 4070, 74 on the RTX 3060). The model's own
probabilities add the best twenty and a pass that weighs every token (44 and
90 microseconds for 16 rows on the RTX 4070, 101 and 212 on the RTX 3060).

**Requests.** The server reads the fields under the names of the OpenAI API and
of llama.cpp's server: `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`
(`typical`), `top_n_sigma`, `xtc_probability`, `xtc_threshold`,
`dynatemp_range`, `dynatemp_exponent`, `mirostat`, `mirostat_tau`,
`mirostat_eta`, `repeat_penalty` (`repetition_penalty`), `repeat_last_n`,
`presence_penalty`, `frequency_penalty`, `dry_multiplier`, `dry_base`,
`dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`,
`logit_bias` (both forms), `ignore_eos`, `samplers` (names or letters), `seed`,
`stop`, `n` (`n_cmpl`) and `max_tokens` (`max_completion_tokens`, `n_predict`);
for probabilities `logprobs` and `top_logprobs` (chat), `logprobs` (completions,
OpenAI's older form of `tokens` and `top_logprobs`), and `n_probs` with
`post_sampling_probs`. Unset means the model's default for the numbers the
model owns (temperature, top-k, top-p, min-p), and llama.cpp's for the rest.
`n` completions are `n` requests with the seeds `seed`, `seed + 1`, ...; the
engine reads the prompt once and the other requests start from what it
cached. What is refused, by name: `min_keep` above 1, `repeat_last_n` or
`dry_penalty_last_n` above 1024 or -1, `dry_allowed_length` above 1024, more than 1024 biased tokens, more than
20 alternatives, `n` above 16, `adaptive_target`, `adaptive_decay`, the sampler
names `infill` and `adaptive_p`, and a `samplers` list that puts penalties or
DRY after a sampler that cuts, names one of them twice or more than 16 samplers
that cut or scale (`SamplerKind::order_problem` in `ares-api`, the one rule
that requests and the server's `--samplers` are held to). Structured output is read from `grammar`,
`json_schema` and `response_format` (below), and the calls of tools are held
to a grammar the server writes (see [Server](#server), "Tool calls").
