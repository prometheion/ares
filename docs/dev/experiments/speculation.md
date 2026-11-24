# Experiments: speculation

Dated measurement logs, moved here verbatim from `ARCHITECTURE.md` and `OPEN.md` on 2026-11-24. They are not kept current: each names its date, build and machine.

## The chain in a step of its own, measured (2026-11-15 and 2026-11-16)

Measured 2026-11-15, the build before this work against the build with it,
back to back, on a machine that other work kept at the load averages given;
every number is an indication.

*Qwen3.8-27B* (tensor split 0.625 / 0.375, `qwen35_run --engine 8 --generate
300`, the fork's draft vocabulary, most probable token; tokens per second for
German prose, English prose, new code, the rewrite of a file, and a second
question of each of the first three):

| Run (load average) | before | with |
| --- | --- | --- |
| 1 (17 to 29) | 87.4, 93.8, 88.1, 139.6, 91.4, 89.7, 96.5 | 88.1, 94.6, 85.1, 139.2, 92.5, 94.5, 101.0 |
| 2 (13 to 45) | 71.4, 75.7, 74.9, 131.1, 77.2, 76.4, 84.9 | 67.6, 72.4, 72.6, 108.3, 73.6, 80.8, 91.3 |
| 3 (10 to 18) | 86.4, 92.3, 87.0, 143.9, 91.5, 91.7, 96.8 | 86.3, 94.4, 87.5, 150.0, 91.5, 91.2, 98.9 |

(Runs 1 and 2 priced a size at the second smallest of its last five steps,
run 3 is the rule above.) The two builds are not told apart. What changed is
inside: the windows verify 1 to 7 drafts where they verified 3 to 5 (German,
run 1: 4, 14, 16, 12, 26, 22 and 1 windows of 1 to 7 drafts, before 72 of 4
and 21 of 5), a step gives 3 % more tokens (3.06, 3.23 and 2.86 against 2.97,
3.16 and 2.75) and takes 1 to 7 % longer, because the device thread waits for
the picks of a link before it enqueues the next. Of the 290 to 490 links
asked for in a request 11 to 26 % were not made. The block's probability says what
the average cannot: of the drafts it gave 0.94 and more, 140 of 146 were
accepted (German) and 148 of 154 (the second code question), of those
between 0.25 and 0.56 25 of 67 and 20 of 49. The gain is not there until a link can end without a wait
of the host.

With the tool's process stopped for 250 ms once a second while it generates
(32 to 39 stops a run, one step in thirty late by ten times its length), the
build before this work verifies windows of 1 to 3 drafts where it verified 4
and 5 (German: 27, 43, 38, 8 and 6 windows of 1 to 5 drafts) and gives 59.6,
68.6, 64.0, 104.3, 71.0, 61.5 and 76.6 tokens per second (load 8); the build
with it keeps its windows (German: 8, 10, 23, 38, 5, 5 and 9 of 1 to 7) and
its table (28.4 to 35.9 ms for 1 to 8 rows after the run, 24.6 to 30.3
undisturbed) and gives 58.2, 49.2, 55.6, 83.0, 58.6, 58.9 and 63.0 at a load
that rose to 22 during the run: its steps were 10 ms longer, which the load
or the waits for the links under it account for, not the table. An earlier
pair at loads of 17 to 45 gave 64.6, 53.2, 62.7, 103.3, 64.5, 69.2, 73.1
before and 67.5, 65.7, 58.0, 96.3, 69.1, 67.0, 74.2 with.

Measured 2026-11-16 with the chain in a step of its own (and the round cut
by the drafter's probabilities), against `main` before it (`bc26683`), the
same tool and settings, three pairs back to back at loads of 1 to 14; tokens
per second for the same seven requests:

| Run | `main` | with the step that only drafts |
| --- | --- | --- |
| 1 | 86.4, 94.4, 86.8, 143.8, 92.2, 90.5, 96.8 | 88.3, 92.9, 88.0, 138.6, 91.4, 96.9, 100.6 |
| 2 | 85.6, 81.8, 87.0, 133.9, 91.8, 89.8, 94.6 | 89.0, 94.7, 87.2, 149.8, 96.7, 97.2, 102.0 |
| 3 | 87.3, 91.8, 87.2, 139.4, 91.9, 90.9, 97.5 | 89.8, 93.1, 88.4, 139.4, 92.0, 95.3, 103.9 |

The means are 1 to 7 % higher (German 89.0 against 86.4, English 93.6 against
89.3, the rewrite 142.6 against 139.0, the second code question 102.2 against
96.3), within the spread of single runs for most requests and above it for
none on the side of `main`. The tokens per step are those of `main` (2.7 to
3.7, 4.8 to 5.7 for the rewrite) with shorter steps: the drafter made 3 to 6
links a round where `main` made 4 to 6, 4 to 26 % of the links asked were not
made, and the round verified all but 1 to 5 % of the drafts made. A chain is
priced at 2.0 ms for one link and 10.3 for seven, the step that only drafts
included.

*Qwen3.8-Flash-Next* through the server, on 2026-11-15, before the chain was a
step of its own and the block's experts resident (both cards on every layer,
`--ram 8G`, `--drafters` the block's file, three questions of 300 tokens asked
three times over, thinking off): the server chose windows in every request,
from the first on, of 1.7 to 3.7 tokens a step, and 47 to 98 % of their drafts
were accepted. It was slower for them:

| Tokens per second, German, English, code | round 1 | round 2 | round 3 |
| --- | --- | --- | --- |
| with the MTP block (load 15 to 19) | 17.9, 18.5, 15.9 | 24.0, 23.8, 21.9 | 55.1, 27.0, 23.1 |
| without (load 9) | 34.8, 29.0, 19.9 | 38.5, 35.1, 23.3 | 39.7, 39.3, 26.0 |

A step with a window reads 9 to 56 experts from the file where a step of one
row reads 3.5 to 16, and a window step of 1.7 tokens takes 96 ms where a row
takes 29. Two things the price of a window does not say here: the dearest of
its last eight steps is left out, and on this model that step is not late,
it is the one that fetched; and the experts a window brings into the tiers
push out what the next steps need, which no step is charged for. The build
before this work chose windows in seven of nine requests of the same run and
was no faster (28.9, 14.1, 8.6; 11.8, 12.6, 10.6; 12.9, 18.6, 14.6 at loads
of 20 to 42). What came of it on 2026-11-16 follows.

**What made the MTP block of Flash-Next pay** (2026-11-16, RTX 4070 and RTX
3060, both on every layer, `--ram 8G`, thinking off, greedy, 200 tokens of
German prose, English prose and code, each prompt twice; the machine was
shared with other agents' builds and servers all night, at load averages of
5 to 45 that changed within a run, so only what ran side by side is
compared; the runs are in `/var/tmp/ares-prices/out2` of the work's notes):

- *The block's experts resident.* A link of the chain takes 1.4 to 1.8 ms (the
  prices at load: 1.4, 2.7 and 4.1 ms for one to three links), where it took
  5.4 to 8.1 with the block's experts in the tiers. They take 675 MiB of the
  RTX 4070: 13 344 slots of the model's experts without the block, 12 816
  with it.
- *A draft row that never waits for the file*, as first built, left nearly
  every draft row: at `--ram 8G` a quarter of the experts are only in the
  file, a row reads one to three of them a token on known text and up to
  sixteen on new text, and a row that is left takes no draft. The block's
  first drafts were accepted 2 times of 106 (they are right about nine times
  in ten), and the server gave 3.3 to 11.7 tokens per second.
- *Never leaving a row* brought the acceptance back (284 of 329 first drafts,
  1.3 to 3.4 tokens a round), and was slower than no drafts (7.9, 8.0 and
  5.5 tokens per second on new text at a load of 27 against 15.1, 13.4 and
  8.9 at 36) until the step that only drafts stopped doing the store's work
  between steps (the heat's decay, the RAM tier that grows a piece, the
  promotions) a second time per round, and read the rows of the n-gram
  table the round needs while the conductor decides: then 27, 30 and 18
  tokens per second on new text and 35, 43 and 27
  the second time, at a load of 15, where the server without the block gave
  23, 21 and 16, and 29, 28 and 23 at a load of 17.
- *Leaving a row without reading ahead* what it was left without only moved
  the read to the next round, whose first row is most often the left row's
  token and then waits for the same experts alone in its layer: rounds of
  1.6 to 2.6 tokens took 100 to 150 ms. With the read started as a guess the
  rounds are as fast as without drafts while rows are left.
- *The rule as it stands* (a row waits while it is worth it, what it is left
  without is read ahead): one server with the block, every prompt asked with
  drafts and with `"draft": false` one after the other, the order turned every
  round (so that the load is the same and each comes first as often), load 5
  to 9: with drafts 32.1, 35.6, 26.0 / 46.1, 66.6, 52.3 / 43.3, 45.8, 33.5 /
  52.2, 52.0, 62.7 tokens per second, without 39.9, 41.7, 37.6 / 32.2, 30.6,
  24.5 / 43.8, 45.0, 41.3 / 38.3, 39.0, 25.9 (German, English, code; rounds
  1 and 3 asked with drafts first). The mean over the rounds is 43.4, 50.0 and
  43.6 with drafts and 38.6, 39.1 and 32.3 without: 12 to 35 % more. Whichever
  request comes second finds the tiers filled by the first, and from the
  second round on the n-gram drafter has the whole reply, which only
  the request with drafts uses; the first round is the plainest case, and
  there the request with drafts, coming first, was the slower.

Two servers of the final build started one after the other, with the block and
without it (200 tokens, the three prompts twice; the load is the run's; the
first pair's server without the block ran while another agent's build took
the machine to a load of 30 and is shown for what it is):

| Tokens per second, German, English, code | first time | second time |
| --- | --- | --- |
| without the block, pair 1 (load 30 to 33) | 8.9, 9.5, 7.7 | 19.7, 15.4, 13.9 |
| with the block, pair 1 (load 12) | 30.4, 35.2, 25.0 | 63.8, 29.6, 29.7 |
| without the block, pair 2 (load 12 to 15) | 21.9, 20.3, 13.1 | 29.3, 23.4, 16.9 |
| with the block, pair 2 (load 16 to 17) | 21.2, 30.9, 16.4 | 30.0, 27.4, 27.9 |
| without the block, earlier build, same rule (load 12) | 32.8, 27.3, 16.7 | 30.2, 33.6, 22.4 |

With the block a round gives 1.3 to 1.9 tokens on a text the first time and
1.8 to 3.6 the second, when the n-gram drafter has the earlier reply; 36 to 90 % of
the drafts verified are accepted, fewer than the block is right (about 85 %
of its first drafts), because a row the store leaves is not accepted. The
second pair, the only one at about the same load, gives with the block 3 %
less to 52 % more the first time (German, English, code: -3, +52, +25 %) and
2 to 65 % more the second, and the first pair's server with the block is the
fastest run of all.

*One card* (the RTX 4070 alone, the same build and settings, one pair at a
load of 7 to 10): without the block 9.6, 8.9 and 6.2 tokens per second the
first time and 11.4, 11.0 and 7.2 the second, with it 7.5, 7.8 and 5.2 and
10.6, 7.3 and 5.7: 7 to 34 % slower. A third of the experts have a VRAM slot
there, a step of one row reads 45 to 100 experts from the file, a round with
drafts 100 to 260, and the block's experts take 675 MiB of the 11 GiB. What
the conductor's prices do not see is what a window's reads cost the steps
after it (the experts it brings push out others, and its reads ahead queue
before theirs); see `docs/OPEN.md`.

*How far from Strata.* Strata's log on this machine has 12 to 76 tokens per
second with drafts, median 55, for 128 tokens (`/var/tmp/ares-survey/strata-speculation.md`).
The runs above with the block lie between 16 and 64, the second request of a
text at 28 to 64. What separates the two is the file: Strata runs with no
RAM copy but 23.6 GiB of page cache for a file set of 28.5 GiB, so nearly
every expert that is in no VRAM slot is in RAM, while `--ram 8G` leaves a
quarter of the experts only on the SSD, where a read takes 2.5 to 10 ms on a
busy machine; a row of new text reads one to sixteen of them a token. Strata
also skips the weakest missed experts (tail skip, on by default, which
changes the text) and computes misses on the CPU from the page cache
without a copy. With a tier that holds every expert that has no VRAM slot
(about 16 GiB, see "What the gap between 17 and 22 ms is made of") the file
is read only for an expert's first use, and a draft row then never waits.

## The n-gram drafter, measured (2026-11-10)

Measured 2026-11-10 by replaying this repository's own text (153 000 tokens
of documentation, 629 000 of Rust, 598 of German; vocabulary of Qwen3.5), a
proposal counted as accepted as far as it equals the text. No model ran, and
the tokens per second are computed from the step cost of Qwen3.8-27B on the
development machine (33 ms and 1.1 ms per row, 29.3 tokens/s without
drafts), drafts of at most 16 tokens:

| Text and replay | last 3 tokens, per request | n-gram drafter, shared corpus, length by confidence |
| --- | --- | --- |
| English, written from nothing | 0.19 proposals per token, 0.98 accepted per proposal, 2.78 wasted rows per token, 32.3 tokens/s | 0.32, 0.72, 0.79, 36.7 tokens/s |
| English, the prompt written again with edits | 0.14, 5.91, 1.37, 121 tokens/s | 0.08, 11.7, 0.24, 223 tokens/s |
| English, a file an earlier request wrote | 32.0 tokens/s (it knows one request) | 223 tokens/s |
| Rust, written from nothing | 0.16, 1.65, 2.23, 36.1 tokens/s | 0.41, 1.07, 1.41, 46.9 tokens/s |
| Rust, the prompt written again with edits | 0.11, 7.72, 0.89, 147 tokens/s | 0.08, 12.1, 0.25, 223 tokens/s |
| German, the prompt written again with edits | 0.06, 15.1, 0.03, 240 tokens/s | 0.07, 13.2, 0.06, 238 tokens/s |

The gain in copied text comes from taking the right place: the last three
tokens often end several places, and a match of the last three continues the most
recent one. In text written from nothing the automaton proposes more often
and is wrong more often per proposal; it pays only because the confidence
cuts those drafts short (with every draft at full length it gives 30.5
tokens/s on English, less than the match of the last three). The German sample is two
short files and says nothing about German: there the confidence, which was
measured on English and code, cuts drafts that would have been accepted (238
against 256 tokens/s at full length). On the host a token costs about 0.2
microseconds to read and a draft 0.4, eight and thirty times what the match of
the last three costs; a long prompt is therefore read into the corpus as its rows
are committed, not at once.

## The MTP driver, measured (2026-11-17)

Measured 2026-11-17, `main` before the driver against the driver, in alternation,
loads 2 to 6. The 27B on two cards (tensor split, no draft vocabulary, the seven
requests of `qwen35_run --engine 8` with 200 tokens each): 75.1 and 74.6 tokens/s
on the mean of two runs each; a link takes 1.48 ms on the device (1.50 with exact
products) and 2.5 to 2.6 ms per link of a step that drafts in the engine on both
builds (the driver's step that drafts also follows the accepted rows of the round
before). Under a layer split without a draft vocabulary a link reads the whole
output head on the last card: 4.70 ms a link through the runtime on the RTX 3060.
Flash-Next (two cards, `--ram 8G`, the server, twelve requests of German, English
and code each): a link takes 1.38 to 1.40 ms where it took 1.65 to 2.02 with the
host's wait for every pick, and 0.85 to 0.88 ms with the draft vocabulary of
66 343 tokens; 39.0 tokens/s on the mean against 41.1 before (requests from 31 to
59), 39.3 with the vocabulary. With cheaper links the conductor verifies more of
the block's drafts, at a lower rate (the first link's: 54 to 58 % against 63 to
71 %). The DFlash drafter's step that only drafts takes 5.19 ms with the whole
head and 3.03 ms with the vocabulary (`qwen4exp_runtime`); the replies with
drafts are the replies without, with the vocabulary too.

## The draft policy on the 27B, measured (2026-11-11)

Measured 2026-11-11 with the policy on Qwen3.8-27B (tensor split 0.625 / 0.375,
`qwen35_run --engine`, 400 tokens per request through the chat template with
the fork's `reasoning_effort`, the fork's draft vocabulary, most probable
token and sampled with temperature 1, top-k 20, top-p 0.95; the machine at a
load of 12 to 20 from other work). "Today" is the build before this work with
at most three drafts, which is the policy with a window of 4 rows less the
price table and the per-position estimates:

| Tokens per second | no drafts | today | window 4 | window 8 | window 16 |
| --- | --- | --- | --- | --- | --- |
| German prose, greedy / sampled | 38.4 / 40.3 | 78.9 / 77.1 | 79.0 / 78.8 | 79.1 / 85.7 | 76.6 / 75.4 |
| English prose | 40.7 / 40.3 | 73.4 / 74.5 | 83.3 / 67.5 | 86.2 / 70.6 | 87.3 / 69.4 |
| New code | 40.6 / 40.1 | 83.8 / 79.1 | 80.2 / 76.8 | 78.8 / 78.3 | 80.1 / 80.3 |
| A rewrite of a given file | 40.5 / 39.9 | - | 111.0 / 108.7 | 142.3 / 147.8 | 137.7 / 144.5 |

(The numbers without drafts are through HTTP, the others from the tool, which
runs the same engine; the run to run spread of a cell is about 5 %.) The
windows of 4 and 8 rows differ where text is copied: the rewrite verifies
chains of 7 links in six steps of ten and of 6 in three (the MTP block is
right 0.85 to 1.00 at each of the seven positions there, the n-gram drafter as well)
and gains 28 to 36 % over 3 links, with 5.6 tokens a step against 3.6. Prose
and code gain nothing from a longer chain: their MTP block is right 0.6 to
0.8 at each position, so the chance of a whole chain falls with its length
(0.7 to the fifth is 0.17), and the policy chooses chains of 3 to 6 from the
price it measured (a link of the chain costs 1.5 ms, a row a fraction of a
millisecond, next to the 25 ms of the weights). English prose gains 13 to 17 %
greedy over three links; the sampled cells are within the spread. A window of
16 rows is no better than one of 8, so the second launch of the weight
kernels that a ninth row costs does not pay for the room it takes (VRAM for
another 8 rows of every sequence); this is the measurement behind the default
of 8. The study's prediction of +20 % on new code from chains of 7 links did
not appear: the code question's drafts are right 0.72, 0.74, 0.65, 0.67 and
0.70 at the first five positions, and the policy stops at 5.

Adaptive draft length did not pay off in the llama.cpp fork (a few percent up
for English and code, a few percent down for German). Two of its costs there
do not exist here: a new batch shape built a new graph (about 18 ms, and 26
MiB of VRAM per GPU for each cached shape), and a rejected draft past the
rollback rows forced a replay. Measured on this engine, the gain over three
links is where text is copied (the rewrite) and for English prose, and zero
or inside the spread for German and code; nothing is lost where it does not
pay, because the policy drafts what the price and the acceptance say pay.

## The DFlash drafter, measured (2026-11-16)

**Measured** (2026-11-16, tensor split 0.625 / 0.375, `Q4_K_S`, `q5` K and V, a context of 16384,
the draft vocabulary, windows of 8, `qwen35_run --engine` with the speculation's seven requests of
300 tokens, thinking on; load 4 to 22; `/var/tmp/ares-drafters/out`, means of the runs in brackets):

| request | ngram + MTP block (3 runs, load 4 to 6) | ngram + MTP block + DFlash (3, load 11 to 22) | ngram + DFlash (3, load 5 to 21) |
| --- | ---: | ---: | ---: |
| german | 89.4 (2.92) | 85.9 (3.07) | 90.6 (3.07) |
| english | 86.3 (2.81) | 106.7 (3.83) | 113.2 (3.83) |
| code | 88.5 (2.87) | 88.0 (3.11) | 94.0 (3.15) |
| rewrite | 148.6 (5.51) | 164.4 (6.30) | 177.6 (6.43) |
| german2 | 94.5 (3.18) | 94.2 (3.30) | 98.2 (3.30) |
| english2 | 94.5 (3.05) | 90.8 (3.14) | 95.3 (3.18) |
| code2 | 102.6 (3.61) | 120.1 (4.27) | 123.9 (4.27) |

Tokens per second (tokens per step). The second column holds a run whose load rose from 11 to 33
while it ran (74.7 to 144.2 tokens per second); without it the column is within 2 % of the third.
Over the MTP block alone the drafter gains +1, +31, +6, +20, +4, +1 and +21 % (tokens per step
+5 to +36 %), and loses nowhere: the policy takes it where it pays. The DFlash drafter's drafts are
accepted 0.78, 0.78, 0.74, 0.74, 0.81, 0.81 and 0.85 of the times each position is reached (1303 of
1666 first drafts).

- The drafter is right where the study counted it: tokens per step on these reasoning texts
  3.07 / 3.83 / 3.15 (German, English, code question) against the study's 3.25 / 4.27 / 3.32 for its chain of 7 on the fork's texts
  (`REPORT.md` 3.4, the three questions); the first draft is accepted 0.72 to 0.82 of the time,
  every later one 0.6 to 0.9 of the times it is reached. With both drafters the policy took the
  DFlash drafter for nearly every window (the G-text test: 38 of 38 drafting rounds).
- The step that only drafts costs 4.2 to 5.5 ms with the drafter (its warm-up curve), against the
  study's estimate of 3.9 ms and the fork's 12.5 ms; the inject adds its products to every step
  on the drafter's device (98 MiB of weights for the 27B's file, about 0.3 ms).
- VRAM on the RTX 4070 (the drafter's device): weights 1079.6 MiB, rings 80.6 MiB (two slots),
  scratch 107.9 MiB, records 0.3 MiB: 1.24 GiB. At a context of 32768 `qwen35_run` (which sizes
  its pool without the drafter) is refused by the device's ledger: "cannot reserve 1.05 GiB for
  'dflash drafter: weights': 976.4 MiB free of 11.23 GiB"; the server's plan counts the drafter
  and shrinks the pool instead.
- Exact: the 27B's text with DFlash drafts is the text without them, alone and next to the draft
  block, greedy and sampled (`qwen35_mtp_runtime.rs` exact products; the server's
  `with_the_dflash_drafter_the_reply_with_drafts_is_the_reply_without` with the fast kernels, three
  questions of 300 tokens). Flash-Next (two cards, both on every layer, every miss uploaded) writes
  with them what it writes without, 3.0 tokens a step, the first draft accepted 14 of 18.
- Flash-Next on two cards with `--ram 8G` (the server, German, English and code of 200 tokens, two
  rounds; tokens per second of generation, round 1 / round 2, load 4 to 15):

  | request, round | MTP block (2 runs) | MTP block + DFlash (2) | DFlash (2) |
  | --- | ---: | ---: | ---: |
  | German, 1 / 2 | 27.2 / 36.8 | 26.4 / 41.4 | 31.1 / 37.4 |
  | English, 1 / 2 | 32.5 / 38.2 | 31.9 / 45.9 | 33.3 / 51.4 |
  | code, 1 / 2 | 19.9 / 31.6 | 22.4 / 32.2 | 24.9 / 38.5 |

  The runs with the MTP block alone ran at a load of 11 to 13, those with the drafter alone at 4
  to 5. The runs
  spread as the speculation's did (the tiers warm in round 1, and other agents' builds moved
  the load by a factor of two within a run); the order of the means is the measurement, not the
  per-request numbers.

## The MTP block of Flash-Next on two cards, measured (2026-11-16 and after the rebase)

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
