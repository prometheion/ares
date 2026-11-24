# Experiments: Flash-Next decode, experts and tiers

Dated measurement logs, moved here verbatim from `ARCHITECTURE.md` and `OPEN.md` on 2026-11-24. They are not kept current: each names its date, build and machine.

## The MTP block of Flash-Next, measured (2026-11-15)

*Measured* (2026-11-15, RTX 4070 and RTX 3060, `--ram 10G`, `--chat-var
enable_thinking=false`, greedy, 128 tokens after prompts of about 1000 tokens of
German prose, English prose and Rust; the machine was shared with three other
jobs the whole day, the load average is given, and a step that waits for the
drive or the CPU takes what the load leaves it, so the columns are not one
measurement of two engines):

| | Generation, first request: German / English / Rust | The same requests again |
| --- | --- | --- |
| Both cards on every layer, no drafter (load 9 to 11) | 30.9 / 19.4 / 19.4 tokens/s | 33.3 / 44.2 / 33.2 |
| The same (load 12 to 22, a build running) | 19.3 / 24.3 / 16.9 | 27.9 / 24.6 / 13.3 |
| Both cards on every layer, the block, prices learned from the steps that run, windows of 8 rows (load 11 to 18) | 10.7 / 24.4 / 17.2 | 63.7 / 28.9 / 50.7 |
| A range of layers each, the block, prices measured at the start, windows of 4 rows (load 6 to 8) | 23.7 / 30.8 / 27.8 | 39.9 / 37.6 / 47.6 |
| A range of layers each, no drafter (load 28 to 33) | 15.4 / 22.6 / 8.8 | 19.9 / 25.7 / 10.4 |
| One card (RTX 4070), the block, prices measured at the start, windows of 4 rows (load 16 to 22) | 11.4 / 13.5 / 5.9 | 4.1 / 15.8 / 9.6 |

In the fourth row the conductor did choose windows (its measured prices came out
lower on a quieter machine): 255 links in 87 steps, the links accepted 59 times of
87, 36 of 59 and 21 of 36, 5.4 ms a link on the RTX 3060 (15.7 ms for the chain of
a step that drafts), 535 to 1080 picks a step where a step of one row has 480, and
21 to 166 of them outside a slot. On one card, which holds a third of the experts
in VRAM and reads the rest from RAM and the file, windows were chosen as well: 326
links in 182 steps, accepted 117 times of 181, 50 of 82 and 3 of 5, 6.1 to 6.9 ms
a link. The second time a text is asked the n-gram drafter has
it, which is where the 63.7 and the 47.6 come from (37 and 15 steps for 128
tokens). The block takes 175 MiB of dense weights, 21 to 49 MiB of keys and values
and 35 to 48 MiB of buffers on its card. No pair of runs with and
without the block at the same load exists, so whether the block makes this model
faster is not shown; the first requests of the third row say that windows of eight
rows on tiers that do not hold the text make it slower.

**Measured through the server** (2026-11-11, RTX 4070 and RTX 3060, the
engine started with `--chat-var enable_thinking=false` and otherwise its
defaults, greedy, 128 tokens, a prompt of about 1000 tokens: 1000 German, 1070
English, 1034 Rust; load average 12 to 18 from other agents' builds unless
said; the numbers are of the second and third of three rounds of the same
three prompts, after the warm-up and a round that fills the slots):

| | Both on every layer, n-gram drafts | The same, no drafts (`--drafts 0` then; what the server does now) | A range of layers each, n-gram drafts |
| --- | --- | --- | --- |
| Generation, German / English / code | 46 and 45 / 44 and 110 / 70 and 73 tokens/s | 44 and 46 / 48 and 43 / 42 and 42 | 27 and 45 / 16 and 42 / 56 and 74 |
| A short question (25 tokens), German / English / code | 59 and 68 / 45 and 60 / 54 and 57 | 39 and 37 / 42 and 40 / 33 and 33 | 29 and 69 / 61 and 46 / 38 and 55 |
| Time to the first token, prompt of 1000 tokens | 8.4 to 15.8 s | 8.7 to 13.4 s | 10.1 to 20.6 s |
| Time to the first token, short question | 0.30 to 0.43 s | 0.30 to 0.46 s | 0.29 to 0.52 s |
| Prompt, 1000 tokens | 63 to 128 tokens/s | 77 to 123 | 50 to 106 |
| Prompt, 2000 tokens | 126 tokens/s | 108 | 72 |
| The tool, same machine, load 17 to 23 | decode step at 1000 tokens: 23.1 ms median (21.5 to 23.1 at lower loads); prompt of 2000 tokens: 71 tokens/s in steps of eight rows (before the wide steps below) | | |

Without drafts a token takes 20.7 to 23.9 ms, the tool's step (21.5 to 23.1
ms), so the server adds nothing to it. The median over the six warm runs with
the n-gram drafts was 58 tokens per second, and 48 and 51 over the twelve runs of each of
the two servers measured again below, against Strata's 56 with drafts: level
within the spread of the runs, which is wider than the difference (44 to 110
in the first series; what the drafts gain depends on the text, as everywhere:
English prose at 110 accepted 95 of 95). Prompts were read at 63 to 139 tokens per second, in steps
of eight rows, where the tool's earlier measurements were 54 to 73 and Strata's
42 to 201 (slices of a prompt are wider now, see "Wide steps, measured").

Where the experts were found, per decode step without drafts (480 picks, 10
experts of 48 layers), second and third round: 6.0 to 15.6 outside a slot for
English and German prose (1.3 to 3.3 %) and 34 to 37 for code (7 to 8 %), of
which 0.7 to 2.0 from the file and 5 to 35 from the RAM tier (each miss is
then computed by the CPU or uploaded, 4.5 to 32 and 2.0 to 17 per step); the
layers waited 2.3 to 6.6 ms per step in all. Prompt steps (eight rows) pick
about 40 distinct experts per layer and miss 4 to 15 % of them: 80 to 300 per
step, 7 to 21 of them from the file. A run of the engine takes 11.4 GiB of each
card (11 689 and 11 663 MiB in use of 12 GiB: the dense parts are 2.9 GiB on the
first card and 0.4 on the second, 13 824 expert slots, 5 808 and 8 016, are 56 %
of the experts, K and V are 13.9 MiB per 1000 tokens, 18 432 tokens in all) and
10.6 to 11.9 GB of RAM (10.2 GiB of them the tier of 7 488 slots, 30 % of the
experts), so 87 % of the experts have a slot. Starting takes 11 to 15 s to load and 41 to 44 s
for the warm-up (20 to 32 s on 2026-11-16, and a start from a usage profile fills the tiers
instead: see "The usage profile of the experts" below); the first request after a start without the warm-up took 77
s for a prompt of 1000 tokens (13 tokens/s), with it 13 to 23 s in four
sessions and 53 s in one at a load of 15 to 18; a text the warm-up did not
resemble keeps being slow until its experts have slots.

Measured again on 2026-11-12, after the work was put on the main line (the
embedding gather, the prefix cache, reasoning in the server, the many-row expert
kernels, none of which the model's path uses): the server of before and the one
of after, started alternately twice, three rounds of the same three prompts at
128 tokens, rounds two and three, load average 10 to 15. Generation was 30 to 89
(median over the twelve runs 47.7) tokens per second before and 31 to 84
(median 50.8) after: German 30 to 89 and 31 to 84, English 20 to 55 and 43 to
53, code 32 to 81 and 48 to 71. Prompts of 1000 tokens were read at 37 to 138
and 65 to 138 tokens per second, the time to the first token was 7.8 to 27.8 s
and 7.8 to 15.9 s. The two servers do not differ beyond the spread between runs
of one of them, which comes from what the text accepts of its drafts and from
the load. The cards held 11 679 and 11 661 MiB, the process 9.4 to 11.4 GB of RAM
(once 14.7 GB right after the start). A first run of the server after the work
at a load of 23 to 25, with other agents' builds and benchmarks on the same
drive and cores, gave 21 to 43 tokens per second and prompts at 35 to 52: the
machine's load decides what a step that waits for the drive costs, so a
comparison needs the two servers measured next to each other.

Against the tool's chat on its two questions the server writes the same
English text (64 tokens, the second time; the first time it parted after 30
tokens at a near tie) and, in German, a text of the same kind that starts with
"Das Sonnenlicht" where the tool's starts with "Sonnenlicht" (a near tie: the
reference's own text, in `/var/tmp/ares-qwen4exp-trace/german.txt`, has "Das
Sonnenlicht"). The engine's own test pins the first sentence of the English
answer to the reference's.

## K and V beyond VRAM, measured (2026-11-22)

*Measured* (2026-11-22, `far_attention_bench`, the RTX 4060 Ti of the test
machine on a PCIe 4.0 x4 link, 6.4 GB/s for a copy; one decode row, one layer,
K and V `q8`): Flash-Next's 2048 picked cells take 31.5 to 33 us in VRAM and
428 to 429 us from RAM at 4k to 262k cells (5.2 GB/s, the link): a decode step
of 12 layers pays at most 4.8 ms when every picked cell is far, about 1.2 ms
on a x16 link. The 27B's dense layers read every cell: 32 768 cells take 328 us
in VRAM and 13.7 ms from RAM (220 ms a step of 16 layers), 262 144 cells 2.2
and 109 ms (1.74 s a step), so its K and V beyond VRAM are not built: its heads
move to the second card first (`devices/roles`), and with `q4` two cards hold
482 707 tokens.
On the development machine's cards (the same bench, 2026-11-22, load 12 to
16): the RTX 4070 on PCIe 4.0 x16 (19 to 25 GB/s) reads Flash-Next's picked
cells from RAM in 126 to 147 us a layer (34 to 36 in VRAM, 1.5 to 1.8 ms a
step), the RTX 3060 on PCIe 3.0 x4 (3.3 GB/s) in 850 us (10.2 ms a step);
the 27B's dense layers at 32 768 cells take 104.5 ms a step on the 4070 and
435 ms on the 3060 from RAM, so with the attention split (three quarters of
the heads on the 3060) every 1000 cells beyond VRAM would cost about 8 ms a
token. With two cards on every layer Flash-Next's attention is the mixer's.
The made-up model's test (`k_and_v_beyond_vram_change_no_token`) passes on
one card, by layers (a far pool per part) and with both cards on every layer.

*The real file, A/B* (2026-11-22, the test machine: IQ2_XS on the RTX 4060 Ti, PCIe 4.0 x4,
`--ram 16G`, no `--context`, no drafters, every run from a fresh usage profile so that the
warm-up fills the tiers alike; A is the build before, which serves the model's 262 144 cells
all in VRAM, B this one; runs A1, B1, A2, B3, A3, B4, A4, B5 in that order at a load of 4 to
19; B1 failed, see below). What the expert tier gets back: A's context took 3.57 GiB of the
11.24 GiB for slots and left 5376 VRAM slots (21.9 % of the experts), B's takes 743 MiB (the
keys of 2050 pages and the K and V of 225 pages, 28 544 tokens) and leaves 7440 (30.3 %), 38 %
more; the RAM tier is the same (11 313 and 11 355 slots of 15.8 GiB).

| | A (all in VRAM) | B (beyond 28 544 tokens in RAM) |
| --- | --- | --- |
| short chat (267 tokens), 128 tokens, first time | 15.7, 16.7, 17.1, 17.7 tokens/s | 20.3, 17.1, 19.5 |
| the same again (from the prefix cache) | 28.1, 31.7, 29.0, 28.0 | 29.5, 24.9, 32.5 |
| a prompt of 65 100 tokens | 258.8, 258.0, 259.7, 254.9 tokens/s | 555.3, 553.6, 554.6 |
| 64 tokens right after it | 15.5, 14.2, 15.5, 9.4 tokens/s | 12.0, 11.6, 11.9 |
| the prompt again (cached) and 192 tokens | 14.8, 9.4 | 16.1, 17.8 |

The prompt reads 2.1 times as fast: with more slots its wide steps upload a third of the
experts (268 000 against 806 000 in A4 and B5) and wait 76 s instead of 206. Decode with the
tiers warm at 65 000 cells is as fast or faster (the far K and V cost at most 2.6 ms a token
there by the bench), and at short contexts B is as fast or faster. The 64 tokens right after the
long prompt are slower in B in all three runs (11.6 to 12.0 against 14.2 to 15.5 in three of
four A runs). It is not the overflow: on the two development cards (2026-11-22, load 9 to 46)
the same build with `--context 66560`, all K and V in VRAM, decoded the 64 tokens right after the
prompt at 2.9 and 5.6 tokens/s against the overflow's 12.5 and 8.8. A prompt read alone lends
expert slots for its steps ("Large prompt steps on lent VRAM") and they come back empty, so the
tokens right after it miss what they held; that build lent steps of 4096 rows, the overflow of
2048, and the build before, with 3.57 GiB less room for slots, lent none on those cards (its
loans on the test machine were not looked at). B1's long prompt failed at the first step past the pages in
VRAM: the copy of a layer's far entries into the loan crossed from one locked piece to the next,
which the driver refuses; fixed by copying a piece at a time.

## Files in any type, measured (2026-11-17)

Measured 2026-11-17 on both cards (`tests/mmvq.rs`, `tests/q4.rs`): the decode kernel of
every one of the 28 types gave the bits of its decoded copy in F32 on every output (9207
per type), 3e-8 to 2e-7 of the sums' magnitude from the f64 product; the fused read of the
mixer with all three matrices in a type gave the bits of the F32 copy as well. The fused read
first took 110 registers a thread instead of 72, too many for the 640 threads of a stream of
2560 (a launch that failed): a call to the decoder chosen at run time inside the unrolled
read keeps many values live across it. So `inject` is a float type in every read but the one
that decodes everything by its type, which is bounded to 1024 threads; a quantized `inject`,
and a block wider than a read's registers allow on the device (`Function::max_threads`),
take that one. The float read is the code of before (72 registers), the inlined ones take 72
to 85. The IQ2_XS file's fingerprint is unchanged (`ffbe225f5e70e94b`). A made-up model
per type (`tests/qwen4exp_device.rs`, the float path in each of the 28 types, the mixers of
rank 256) matches the reference layer by layer on one card and with both cards on every
layer, in steps of up to eight rows and in a wide step, as the model with BF16 does.

The cost, measured 2026-11-17 (`qwen4exp_run --read-time`, one row, the fastest of five
times 64, load 11 to 13): a read of the IQ2_XS file's BF16 mixer takes 24.9 us on the RTX
4070 and 73.9 us on the RTX 3060, against 24.4 and 73.1 before the types (the templated read
has some more address arithmetic); the Q8_0 mixer of UD-Q4_K_XL takes 43.8 and 97.0 us. It
reads half the bytes of BF16, and decoding each weight is what it costs: on the mixer's card
of the two cards on every layer, 97 reads a step, about 1.9 ms more than BF16 would take, for
568 MiB of VRAM less (1.8 GiB less than F32). The IQ2_XS file's decode step at 1000 tokens
on two cards (eight pairs of runs of the build before and after, load 10 to 19): medians of
22.6 to 27.0 ms before and 23.9 to 29.7 after, means 25.1 and 26.7, with the store's numbers
equal; the spread of the loaded machine is larger than the difference.

## UD-Q4_K_XL through the server (2026-11-17)

UD-Q4_K_XL through the server, 2026-11-17 (load average 16 to 30 from other builds on the
machine; a German question, 91 tokens, a code question, 200, a story, 256; greedy; the
three placements gave the same text, correct):

| Placement, `--ram` | VRAM slots per layer | RAM tier | German | Code | Story (tokens/s) |
| --- | --- | --- | --- | --- | --- |
| By layers, 8G | 105 (RTX 4070), 110 (RTX 3060) | 7.81 GiB, 2349 slots | 4.12 | 1.75 | 3.60 |
| One card (RTX 4070), 9G | 37 | 8.87 GiB, 2666 slots | 2.93 | 1.79 | 2.77 |
| Both cards on every layer, 10G | 46 (mixer), 69 (worker) | 9.84 GiB, 2956 slots | 4.37 | 2.55 | 4.31 |

With pruning (`--expert-prune`, the same three questions, tokens/s German / code / story):

| Placement, `--ram` | Off | `skip:8` | `skip:4` |
| --- | --- | --- | --- |
| By layers, 8G | 4.12 / 1.75 / 3.60 | 6.49 / 3.93 / 7.68 | 12.54 / 8.73 / 13.08 |
| One card, 9G | 2.93 / 1.79 / 2.77 | 4.93 / 2.88 / 4.88 | 10.72 / 6.63 / 11.17 |
| Both cards on every layer, 10G | 4.37 / 2.55 / 4.31 | 7.96 / 4.47 / 7.48 | 15.22 / 10.58 / 15.42 |

The pruned answers differ in their words and are still right.

With `--ram 8G` the last two are refused by the RAM their floors need: one card holds 37
slots per layer and needs 51 in RAM (7345 MiB, the tier had 7211); with both cards on every
layer the mixer's card holds 47 and the two cards' layers need 8785 MiB in RAM (the tier had
7195). The refusal says so and names `--ram`.

The tool's decode steps at 1000 tokens of the corpus (`qwen4exp_run --serve model`, one row
a step, 64 steps, load 4 to 13): by layers with 8 GiB 96.9 ms (median; 87 % of the picks in
a slot, 62 misses a step of which the CPU computed 38), one card with 9 GiB 225 ms (58 %, 200
misses, 76 on the CPU), both cards on every layer with 10 GiB 79.1 ms (86 %, 65 misses, 45 on
the CPU). The mixers took 2.5 ms a step on the RTX 4070 and 4.7 ms on the RTX 3060 by layers,
4.9 ms on one card: the drive and the experts are the step.

## The usage profile, measured (2026-11-16)

*Measured* (2026-11-16, RTX 4070 and RTX 3060, `--ram 8G`, `--chat-var
enable_thinking=false`; the server of before, bc26683, and the one with the profile started
one after the other; three questions of 35 to 44 tokens, German, English and code, 128 tokens
each, greedy, sent as soon as the server answered, then the same three again. "Profile of the
warm-up" is the one a start without a profile writes right after its warm-up, so both start
from the same text; "profile after use" is the one such a server wrote at its stop after the
six requests. The drive was shared with other agents' builds and runs the whole time and gave
random reads of experts 0.2 to 1.0 GB/s against 1.3 to 1.7 when quiet, which decides every
phase of a start that reads, and the load fell during the series, so the runs of before had
the higher load in the first pairs):

| Cards | Server | Warm-up, or fill from the profile | Answers after | First round, tokens/s: German / English / code | Reads from the file per step, first round | Second round, tokens/s | Load average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two | before | warm-up 27.6 s (25.7 GiB read since the start) | 68.2 s | 11.6 / 8.8 / 8.2 | 10.8 / 7.3 / 15.6 | 14.1 / 12.4 / 10.5 | 25 |
| two | profile of the warm-up | 20.2 s, 11 303 into VRAM and 3 947 into RAM, 20.6 GiB | 35.5 s | 21.5 / 27.0 / 19.7 | 12.9 / 8.8 / 15.1 | 20.3 / 16.8 / 8.2 | 12 |
| two | before | warm-up 27.8 s (25.0 GiB) | 96.8 s | 11.9 / 12.1 / 5.8 | 11.1 / 7.5 / 15.5 | 15.3 / 14.3 / 9.7 | 24 to 29 |
| two | profile of the warm-up | 21.8 s, 20.6 GiB | 38.8 s | 26.6 / 31.5 / 16.2 | 11.7 / 6.6 / 15.1 | 35.1 / 43.0 / 27.8 | 8 |
| two | profile after use | 23.8 s, 10 795 and 5 420, 21.9 GiB | 42.9 s | 29.3 / 31.2 / 25.8 | 2.9 / 2.7 / 4.5 | 36.4 / 28.3 / 19.7 | 14 |
| two | before | warm-up 21.6 s (25.0 GiB) | 39.9 s | 27.7 / 31.1 / 17.2 | 11.4 / 6.0 / 15.1 | 31.8 / 40.0 / 22.6 | 9 to 10 |
| two | profile after use | 23.0 s, 22.0 GiB | 38.9 s | 32.3 / 37.7 / 31.5 | 3.9 / 3.1 / 4.5 | 41.1 / 39.4 / 34.3 | 6 |
| one (RTX 4070) | before | warm-up 31.9 s (29.1 GiB) | 40.5 s | 5.7 / 7.3 / 5.3 | 70 / 68 / 101 | 10.2 / 9.6 / 6.2 | 8 to 12 |
| one | profile of the warm-up | 12.6 s, 3 064 and 5 231, 11.2 GiB | 27.8 s | 7.1 / 7.3 / 4.4 | 58 / 55 / 100 | 7.1 / 8.3 / 5.4 | 16 |
| one | before | warm-up 25.0 s (27.3 GiB) | 39.4 s | 7.7 / 6.9 / 4.0 | 71 / 79 / 99 | 4.9 / 10.2 / 2.0 | 14 to 17 |
| one | profile of the warm-up | 13.1 s, 11.2 GiB | 21.7 s | 11.0 / 10.4 / 6.3 | 56 / 59 / 97 | 13.6 / 16.0 / 7.5 | 6 to 8 |

- *The start.* What changed is the warm-up against the fill; loading the weights and measuring
  the price of a step, the same in both, took 15 to 69 s by what the drive gave. On one card the
  fill took 12.6 and 13.1 s against a warm-up of 31.9 and 25.0 s (VRAM holds 5 040 experts there,
  so it reads 11.2 GiB). On two cards it took 20 to 24 s against 21.6 to 27.8 s: it reads 20.6 to
  22 GiB, about as much as the warm-up does (25 GiB with the price measurement), and both are
  bound by the drive; at a load of 3 to 7 the engine's own test of the real file took 20.7 s for
  the warm-up and 17.0 s for the fill (21.3 GiB, 1.25 GiB/s). The warm-up is not the 41 to 44 s
  of 2026-11-11 any more, so on two cards a start from the profile is as fast as one with the
  warm-up, not much faster.
- *The first requests* after a start from a profile of what the server was used for missed 2.7 to
  4.5 experts per step from the file, as a warm server does in its second round (2.5 to 6.9),
  where a start with the warm-up missed 6.0 to 15.6, and generated at 26 to 38 tokens per
  second, the second round's speed. With the profile of the warm-up they missed what the
  warm-up start misses, as they should; the speeds of the first pairs differ by the load more
  than by anything else.
- *The reads.* Sorted by their place in the file the fill's reads reached 1.25 GiB/s where
  random reads of experts reached 0.67 to 0.80 GiB/s on the same drive in the same hour.
  Reads of several neighbouring experts at once (into a buffer, copied into their slots) are
  not built; how much more they would give on this drive could not be measured, as the drive
  was never quiet.
- *The idle fill* (two cards, a start without a profile, nothing asked for 90 s, the tiers'
  gauges every half second): the warm-up left 13 328 of 13 344 VRAM slots and 4 573 of 5 568 RAM
  slots with experts; a second after the start the idle fill began and had the RAM tier at 5 558
  after 5.5 s (2.4 GiB read, 0.43 GiB/s at two experts in flight), a short request at 6 s
  stopped it (its first token after 1.2 s, the request a new text with 74 reads from the file per
  step), and a second after the request it went on and had both tiers full at 11.4 s (2.5 GiB
  read by the fill in all). What a request that arrives finds on its way is at most two
  experts' reads, as prefetches that the reader keeps behind its demand reads (at most four
  chunks of 512 KiB at the drive), and an upload per device; that bound is the store's
  (`tests/expert_fill.rs`) and was not resolved by a measurement here, whose times are those of
  the request's own misses.

## Members, loans and promotions, measured (2026-11-22)

*Measured* (2026-11-22, the test machine: RTX 4060 Ti 16 GB on a PCIe 4.0 x4 link, Core Ultra 5
235HX, 62 GiB RAM; UD-Q3_K_XL, `--context 40960`, the bench of the Strata comparison: German,
English and code, 300 tokens greedy, each asked twice; the build of before, 641a8c7, and this
one alternating, two rounds each, every session from a fresh cache directory without a usage
profile, so both fill the tiers from the warm-up; load average 4 to 12 at the starts, from the
engine before). Tokens per second by the client, rounds 1/2; the expert columns are means over
the decode requests from `/metrics`:

| `--ram`, drafts | Build | German 1st | 2nd | English 1st | 2nd | Code 1st | 2nd | VRAM hit % | Uploads a token | CPU % | Wait ms a token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 56G, MTP block | before | 20.3/17.2 | 26.5/26.7 | 22.5/23.7 | 26.8/26.8 | 23.3/24.0 | 32.3/34.8 | 61 | 58 | 27 | 28 |
| 56G, MTP block | loans | 20.0/21.3 | 35.0/32.0 | 24.8/25.9 | 37.6/30.7 | 23.1/23.9 | 32.5/32.6 | 61 | 53 | 26 | 26 |
| 56G, none | before | 16.1/16.4 | 19.8/21.0 | 18.6/20.2 | 21.2/23.2 | 17.1/18.4 | 20.3/22.4 | 70 | 51 | 21 | 29 |
| 56G, none | loans | 16.4/16.6 | 21.1/21.9 | 19.3/18.8 | 21.5/21.9 | 17.1/16.8 | 21.5/21.1 | 72 | 50 | 18 | 30 |
| 24G, MTP block | before | 12.2/11.3 | 16.1/15.4 | 15.3/14.8 | 19.6/25.3 | 13.7/13.2 | 20.9/24.5 | 59 | 56 | 26 | 51 |
| 24G, MTP block | loans | 13.8/14.0 | 21.0/23.6 | 17.3/16.3 | 21.8/22.0 | 13.6/13.7 | 26.8/26.2 | 62 | 52 | 25 | 45 |
| 24G, none | before | 10.3/10.3 | 13.9/13.6 | 12.9/12.7 | 16.2/15.8 | 11.1/10.7 | 15.8/15.2 | 71 | 57 | 19 | 56 |
| 24G, none | loans | 12.3/12.9 | 16.7/17.7 | 13.6/15.5 | 17.2/18.1 | 11.0/12.0 | 17.0/14.6 | 71 | 36 | 22 | 48 |
| IQ2_XS, 56G, MTP block (one round) | before | 32.6 | 52.2 | 45.9 | 78.2 | 47.1 | 76.7 | 71 | 16 | 25 | 10 |
| IQ2_XS, 56G, MTP block (one round) | loans | 27.8 | 54.2 | 44.9 | 51.9 | 46.0 | 86.6 | 76 | 18 | 21 | 9 |

- Over the six cells of a row, the mean is 11 % higher with the MTP block at 56G, the same
  without it, and 13 to 14 % higher at 24G; the second asks gain most. The IQ2_XS file, one round
  each, is 6 % lower in the mean, most of it one cell (English asked again, where the n-gram
  drafter's luck decides most; without it 2 % higher).
- Started without a profile, both builds fill VRAM from the warm-up, and the hit rate is the
  same (59 to 72 %): what was behind the 37 to 49 % of the comparison was the profile that counted
  prompts by their rows and the fill that left two thirds of VRAM to the price measurement. Loans
  keep the members, so a text asked again finds what it used. Started from a profile of use
  (format 2, written by an earlier session of the same bench) the fill brought 3 473 experts
  into VRAM and the hit rate was the same, 59 % with the block and 70 % without (21.2 to 32.5
  and 17.0 to 21.9 tokens/s, one round each): on this card the size of VRAM bounds it.
- What a token waits for is the misses, 26 to 56 ms: on this machine a CPU job takes 170 to 410
  us per expert and row for the types of this file (decoded row by row) and an upload 410 to 490
  us, so neither side is much faster and the cost model keeps both busy (see OPEN.md).

*With kernels for the file's types* (2026-11-22, branch `flash/cpu-experts`; the test machine,
`--ram 40G`, `--context 40960`, no profile, the same bench; the build of before is `verify/stack`;
sessions one after the other, load average at their start in the table; tokens per second by the
client, German / English / code, each asked twice):

| Build, drafts | German 1st/2nd | English 1st/2nd | Code 1st/2nd | Mean | VRAM hit % | CPU % of picks | Uploads a token | Wait ms a token | Load |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| before, MTP block | 18.9/26.7 | 24.3/34.7 | 21.9/29.5 | 26.0 | 61 | 27 | 49.7 | 28.3 | 2.1 |
| kernels (215eaef), MTP block | 22.6/50.9 | 29.8/40.5 | 30.3/44.3 | 36.4 | 57 | 37 | 24.2 | 18.4 | 4.0 |
| before, MTP block | 16.2/33.7 | 21.4/33.8 | 21.3/30.0 | 26.1 | 61 | 25 | 57.6 | 30.6 | 8.9 |
| all of the branch (5aa8aa4), MTP block | 23.5/36.3 | 31.8/39.5 | 30.9/39.5 | 33.6 | 57 | 37 | 25.0 | 18.9 | 6.3 |
| before, none | 12.4/17.5 | 13.8/17.1 | 12.8/16.5 | 15.0 | 72 | 17 | 55.9 | 41.0 | 2.3 (a build ran beside it) |
| kernels and job order (1c2dac3), none | 20.4/26.5 | 25.1/27.7 | 22.4/27.1 | 24.9 | 69 | 28 | 21.9 | 18.2 | 5.6 |

- Decode is 29 to 40 % faster with the MTP block and 66 % without it in these sessions; the wait for
  misses halves (18 against 28 to 41 ms a token) and the uploads halve, the CPU computing nearly
  every miss. The prompts carry the session's name, so the texts differ from session to session.
- Where a step's time goes (`qwen4exp_run --decode-at 4000 --decode-steps 256 --breakdown`,
  `--ram-gib 39.4`, one row a step): 59.7 ms a step before (load 7 to 9), 39.7 to 40.6 ms with
  the branch (load 5 to 7); the wait for misses summed over the layers 27.5 ms before, 8.2 to 9.4
  ms now. A layer whose misses the CPU computes (2.9 experts at the median) has its job made 9 to
  10 us after its route record, the job's last task ends at 160 to 175 us, and the sums are on
  their way 6 to 10 us later: the job is the wait, and three experts' 7 MB from the RAM tier in
  about 150 us is close to what the memory gives. On the device, with timed events: dense parts
  15.6 ms, stream mixers 7.8, exchange 1.1, recurrent layers 1.0, attention 0.9, the experts
  (with their wait) 15 ms; about 27 ms a token are outside the experts on this card.
- Without uploads of misses there are no loans, and VRAM's share of the picks is lower: 70 %
  against 80 % at the start of that decode, 55 % against 80 % later in it (the server's bench:
  57 against 61 % with the MTP block, 69 against 72 % without). Uploading on loan every miss the
  CPU computed, behind the step (118 a step as far as the link had room), raised it to 75 % and
  made the steps slower (42.8 against 40.6 ms at the median, the dense part 19.3 against 15.6
  ms): on this x4 link the uploads hold up what a step reads from the host and the layers'
  sums. That is not in the engine (OPEN.md).
- Against Strata: its baseline at the same budget (0.2.15, engine 0.1.60, 40 GiB) decoded at 24.2 to
  38.5 tokens/s with drafts (mean 29.9) and read prompts at 198 / 787 / 1192 tokens/s (2k / 8k /
  32k; Ares 176 / 250 / 281, see OPEN.md); its runs without drafts were disturbed by a build. Strata
  0.2.19 (engine 0.1.65, decode said to be 20 to 59 % faster on RTX 40 cards) was installed during
  this work and is still to be measured against this branch.

## Reading experts ahead, measured (2026-11-16)

*Measured* (2026-11-16, the IQ2_XS file, `qwen4exp_run --decode-at 1000 --decode-steps 128`
on the first 1000 tokens of this document, read in steps of eight rows, then 128 generated;
the engine's serving (`--serve model`); the build of before and the one with the predictions
run one after the other in one turn of the shared GPU, the load given; the machine and the
drive were shared with three other agents the whole time, and the same build gave 15 to 41
tokens per second from run to run):

| Run | Cards, `--ram-gib` | Build | Decode: median, mean | Reads from the file per step | Experts predicted in no tier per step, of them picked | Read ahead per step | Load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | two, 8 | before | 26.86 ms, 33.5 tokens/s | 4.0 | | | 9 |
| 1 | two, 8 | every prediction read (first form) | 23.57 ms, 40.2 | 2.9 | 35, 3 % of the reads used | 3.7 (0.11 used) | 7 |
| 2 | two, 8 | the odds, one table | 26.19 ms, 30.9 | 5.5 | 53, 4.6 % | 1.1 (0.04 used) | 12 |
| 2 | one, 8 | the odds, one table | 46.46 ms, 18.8 | 8.6 | 53, 7.8 % | 3.3 (0.47 used) | 13 |
| 2 | one, 8 | before | 37.35 ms, 24.2 | 8.8 | | | 14 |
| 3 | two, 8 | odds per kind of step | 26.30 ms, 29.6 | 7.3 | 61, 5.3 % | 0 | 7 |
| 3 | two, 8 | before | 49.05 ms, 15.0 | 8.4 | | | 11 |
| 3 | one, 8 | before | 30.19 ms, 30.3 | 8.0 | | | 9 |
| 3 | one, 8 | odds per kind of step | 39.71 ms, 19.7 | 14.2 | 82, 8.0 % | 0 | 10 |
| 3 | two, 4 | odds per kind of step | 38.12 ms, 22.2 | 14.0 | 84, 7.6 % | 0 | 11 |
| 3 | two, 4 | before | 29.63 ms, 30.6 | 6.3 | | | 12 |
| 4 | two, 8 | room nobody uses (`Want::Spare`) | 25.46 ms, 31.6 | 5.6 | 56, 4.6 % | 0 | 10 |
| 4 | two, 8 | before | 22.68 ms, 40.9 | 2.8 | | | 12 |

- *A generated row's picks that no tier holds are what a prediction gets wrong.* The routers
  of the next two layers name about 650 experts a step of one row, half of which the routers
  then pick; but of the 35 to 84 among them that no tier holds only 3 to 8 % are picked, and
  no place of the lists reaches a half. So the store reads nothing ahead in steps of one row
  on these tiers, and the fastest steps (17.8 to 18.5 ms) are the same with the predictions
  as without: what they cost the card is not seen beside the spread of the runs.
- *Rows of a prompt in steps of eight rows* name experts in no tier that are picked a fifth
  of the time (20 % in the fingerprint's runs), and some places of their lists more often:
  the store reads 3.5 to 4.6 experts ahead a step there, of which a quarter are used.
- *Through the server* (2026-11-16 19:25 to 19:37, `--ram 8G`, the same copy of a usage
  profile at every start, three questions of 25 to 35 tokens, German, English and code, 128
  tokens each, greedy, twice; the build of before and the one with the predictions one
  after the other in one turn of the GPU; the counters per step are over the prompts' wide
  steps and the generating steps together):

  | Cards | Build | Generation, round 1: German / English / code | Round 2 | Prompt, round 2 | Reads from the file per step | Read ahead per step, used | Wait per step | Load |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | two | with predictions | 28.3 / 33.7 / 18.5 tokens/s | 38.0 / 45.7 / 29.4 | 103 / 106 / 122 tokens/s | 6.2 | 2.4, 0.9 | 21.0 ms | 10 to 11 |
  | two | before | 31.7 / 33.9 / 19.5 | 40.2 / 34.7 / 31.1 | 118 / 114 / 121 | 7.7 | | 20.6 ms | 8 to 9 |
  | one (RTX 4070) | before | 11.7 / 8.6 / 5.5 | 13.4 / 10.0 / 2.5 | 14 / 13 / 15 | 65.1 | | 163 ms | 8 to 22 |
  | one | with predictions | 5.1 / 6.9 / 5.8 | 16.0 / 12.9 / 7.4 | 20 / 19 / 19 | 53.5 | 15.8, 9.5 | 140 ms | 25 to 12 |

  On two cards the two builds do not differ beyond the spread of the runs. On one card,
  where a step reads 50 to 100 experts from the file, 60 % of the reads ahead were picked,
  the reads on demand fell from 65 to 54 a step and the wait for the misses from 163 to 140
  ms a step; the second round generated at 16.0 / 12.9 / 7.4 tokens per second against
  13.4 / 10.0 / 2.5 and read its prompts at 19 to 20 against 13 to 15 (the first round of
  the build with the predictions ran at a load of 25 against 8, and lost). One pair each,
  on a machine shared with three other agents: an indication, not a measurement of the
  difference.
- *Wide steps.* On the made-up model of `steps_wider_than_eight_rows_match_the_reference`
  (half of the experts in slots, steps of 9, 64, 256 and 3 rows) the next layer's
  predictions read 162 to 167 experts ahead into staging, and the steps read 57 to 61 from
  the file on demand where the build of before read 192; with four transit slots, 179 on
  demand and 210 ahead where it read 336. On the real file this was not measured (the
  shared GPU gave no turn for it).
- *What reading ahead cost.* Reads for nothing take the drive from the reads the steps wait
  for (run 2 on one card: a miss from the file waited 3.7 ms at the median and 39.7 at the
  99th percentile, against 2.6 and 8.0 before), and as long as they took the slot of the
  coldest member of the RAM tier (run 3), the steps after the prompt read 14 experts a step
  from the file where the build of before read 6 to 8. Hence the odds per kind of step and
  the room nobody uses.

**What the gap between 17 and 22 ms is made of, and what the size of the tier
did.** The tool counts, for every decode step at 1000 tokens, the experts the
routers picked that were in no slot, and splits the steps by what they missed
(on the recorded routing, 192 steps per run, five runs, load average 6 to 23
from other agents' builds, so the milliseconds differ between runs and the
classes within a run are what to read; the ranges leave out the run at a load
of 14, whose medians were 25.1, 31.5 and 42.6 ms):

| Steps | Share | Median |
| --- | --- | --- |
| No expert outside a slot | 2 to 6 % | 17.3 to 20.4 ms |
| Misses served from the RAM tier or computed by the CPU only (3 to 5 of them) | 23 to 31 % | 17.5 to 19.1 ms |
| At least one expert read from the file (2 to 3 of them, among 6 to 17 misses) | 62 to 72 % | 24.2 to 29.5 ms |

A miss served from RAM costs an upload of 0.1 ms at the median and the CPU
computes the others while the device runs, so neither shows in the step: the
steps with such misses take as long as the steps with none. The 5 to 12 ms
above that belong to the experts read from the file, 2 to 3 per step at 2.4 to
3.3 ms at the median and 3.7 to 11 ms at the 99th percentile each, which the
device waits for in the layer that needs them, both cards at once. Most steps
have one, which is why the median (21 to 23 ms) is far above the 17 ms of the
steps that have none and the mean (24 to 30 ms) above the median. The rows of
the n-gram table were not part of it (0.07 to 1.0 ms per step below a load of
9, read ahead from the picks).

Those reads are experts used for the first time in a run, or used again after
a tier let them go, so the size of the tier decides how many of them there are
only when the text comes back to experts it has left. With the profile loaded
the tool reads the same 3.0 experts per step with a tier of 5 472 slots and
with 8 064 (22.6 and 22.8 ms at the median in two runs back to back at a load
of 6 to 7; the second pair, 21.5 and 37.0 ms, ran at loads of 8 and 14 and
says only that the load decides). Through the server, with a new text every
request, the size decides a great deal. The same
prompts in a server per tier, load average 12 to 13 (15 to 18 for the
second), round 1 of 2 or 3, tokens per second with the n-gram drafts the
server had then:

| Tier (`--ram`) | RAM slots | Experts with a slot | File reads per step, prompt rows | File reads per step, decode | Generation: German / English / code | Prompt, 1000 tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 7.5G, the third of the available RAM the tier took before | 5 472 | 78.5 % | 20.8 / 14.1 / 46.1 | 4.7 / 2.1 / 13.5 | 37.7 / 43.1 / 51.2 | 69 / 101 / 57 |
| default, half of the available RAM as the limit | 7 488 | 86.7 % | 11.0 / 6.9 / 20.6 | 3.3 / 0.9 / 6.1 | 46.4 / 43.9 / 69.6 | 63 / 102 / 81 |
| 16G | 11 712 | all | 0.2 / 0.1 / 0.7 | 0.6 / 0.2 / 0.2 | 75.5 / 44.2 / 103.9 | 139 / 137 / 127 |

The default now takes the limit less the engine's buffers (10.3 GiB of 20.65
GiB available when the server started) where it took a third of the available
RAM, 7.5 GiB, outside the books of the limit; that is 37 % more slots and about
half the reads from the file. With a limit that lets the tier hold every
expert the VRAM has no slot for (about 16 GiB on this machine), reads from the
file stop after the first use of an expert. Raising `--ram` is the lever, and
how much RAM is free is the user's to say. (Since 2026-11-15 the default limit
is all the machine has and the tier follows what is available, giving memory
back when another user of the machine needs it; see "Host memory" and the
measurement there.)

## Pruning, measured (2026-11-16)

*How alike the experts are, measured* (IQ2_XS, 3296 calibration rows; UD-Q4_K_XL, 640;
2026-11-16). Hardly at all. The nearest expert of a layer gives a median closeness of 0.001
to 0.017 in layers 1 to 46 (90 % of the experts below 0.003 to 0.074), 0.070 in layer 0
and 0.013 in layer 47; the Q4 file gives the same picture (median over the layers 0.003).
The outputs of two experts on the rows that pick one of them are nearly orthogonal: the
best stand-in carries well under 1 % of a missed expert's output in most layers, so a
stand-in that takes the missed pick's weight adds an error about as large as the one it
takes away. Which measure finds the nearest (64 experts per layer measured against all
511 others on their own rows, layers 0, 12, 24, 36 and 47 of IQ2_XS): the router's rows
find a stand-in with 42 to 70 % of the best closeness, the outputs on 16 shared rows 21
to 74 %, the experts' weights (a sketch of all three matrices) 2 to 7 %. Weights say
nothing (the inner units of two experts are in no common order); the router's rows and
outputs on any rows say something, outputs on the expert's own rows say it best, and that
is what the table keeps.

*The rules, measured* (2026-11-16, RTX 4070 and RTX 3060, `qwen4exp_prune measure`; per
text a process of its own that opens the model as the server does, fills the tiers from the
usage profile its warm-up wrote, reads 256 tokens of the text as a prompt in steps of eight
rows without pruning and then 1000 tokens of it one a step, as generation does, with the
text's tokens as the tokens generated; KL is the divergence of each step's distribution from
the same run without pruning over its 256 most likely tokens and the rest as one, top-1 how
often the most likely token is the same; every miss uploaded (`--serve stream`), because with
the CPU computing misses two runs without pruning already differ by KL 0.006; texts German
prose, English prose and Rust; load average 3 to 15). IQ2_XS, both cards on every layer,
the split pinned:

| `--ram` | Rule | From rank | KL (German / English / code) | Top-1 | Reads from the SSD per token | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| 8G | none | - | 0.0008 (0.0001 / 0.0000 / 0.0025): two runs without pruning | 99.6 % | 4.52 | 31.2 |
| 8G | skip | 8 | 0.0081 (0.0073 / 0.0105 / 0.0066) | 96.3 % | 2.33 | 37.9 |
| 8G | router | 8 | 0.0085 (0.0085 / 0.0101 / 0.0070) | 96.2 % | 2.33 | 37.7 |
| 8G | similar | 8 | 0.0095 (0.0088 / 0.0116 / 0.0081) | 95.8 % | 2.34 | 36.7 |
| 8G | skip | 6 | 0.0094 (0.0107 / 0.0106 / 0.0067) | 95.9 % | 1.37 | 41.6 |
| 8G | skip | 4 | 0.0111 (0.0108 / 0.0134 / 0.0091) | 95.5 % | 0.71 | 45.2 |
| 8G | router | 4 | 0.0141 (0.0120 / 0.0143 / 0.0160) | 95.4 % | 0.72 | 42.0 |
| 8G | similar | 4 | 0.0141 (0.0127 / 0.0166 / 0.0128) | 95.2 % | 0.72 | 38.6 |
| 8G | similar, scaled | 4 | 0.0108 (0.0111 / 0.0126 / 0.0086) | 95.7 % | 0.71 | 40.7 |
| 8G | similar, renormalised | 4 | 0.0128 (0.0132 / 0.0146 / 0.0105) | 95.6 % | 0.71 | 43.8 |
| 8G | skip, renormalised | 4 | 0.0123 (0.0127 / 0.0135 / 0.0106) | 95.7 % | 0.72 | 41.3 |
| 8G | router, weight kept | 4 | 0.0142 (0.0154 / 0.0149 / 0.0122) | 94.8 % | 0.72 | 33.0 |
| 8G | skip | 1 | 0.0276 (0.0245 / 0.0255 / 0.0328) | 93.5 % | 0.00 | 46.1 |
| 8G | router | 1 | 0.0313 (0.0277 / 0.0275 / 0.0387) | 93.2 % | 0.00 | 45.3 |
| 8G | similar | 1 | 0.0380 (0.0315 / 0.0338 / 0.0486) | 92.9 % | 0.00 | 44.1 |
| 8G | skip, weight below 0.05 | 1 | 0.0065 (0.0062 / 0.0086 / 0.0048) | 96.4 % | 3.91 | 23.0 (a busy minute) |
| 8G | skip, weight below 0.1 | 1 | 0.0111 (0.0115 / 0.0123 / 0.0094) | 95.6 % | 0.80 | 42.5 |
| 4G | none | - | - | - | 10.13 | 15.0 |
| 4G | skip | 8 | 0.0098 (0.0088 / 0.0118 / 0.0089) | 95.5 % | 5.14 | 23.3 |
| 4G | router | 8 | 0.0105 (0.0098 / 0.0122 / 0.0094) | 95.6 % | 5.15 | 26.1 |
| 4G | similar | 8 | 0.0115 (0.0105 / 0.0135 / 0.0106) | 94.8 % | 5.19 | 23.5 |
| 4G | skip | 1 | 0.0516 (0.0495 / 0.0490 / 0.0564) | 91.9 % | 0.00 | 49.0 |
| 4G | skip, weight below 0.05 | 1 | 0.0073 (0.0076 / 0.0094 / 0.0050) | 96.2 % | 8.75 | 17.1 |

The UD-Q4_K_XL file (as a copy with its float path in F32, see OPEN.md; a range of layers
per card, 4 344 experts in VRAM and 2 400 in a RAM tier of 8 GiB, 27 % of all, 23 % with 4 GiB;
a prompt of 32 tokens and 200 tokens one a step per text):

| `--ram` | Rule | From rank | KL (German / English / code) | Top-1 | Reads from the SSD per token | Tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| 8G | none | - | - | - | 81.0 | 3.5 |
| 8G | skip | 8 | 0.0258 (0.0383 / 0.0261 / 0.0129) | 93.2 % | 47.1 | 5.0 |
| 8G | router | 8 | 0.0301 (0.0381 / 0.0318 / 0.0206) | 90.0 % | 46.9 | 4.8 |
| 8G | similar | 8 | 0.0345 (0.0465 / 0.0393 / 0.0177) | 89.7 % | 48.0 | 4.7 |
| 8G | skip | 4 | 0.0773 (0.1110 / 0.0858 / 0.0350) | 86.3 % | 16.7 | 8.7 |
| 8G | skip | 1 | 0.5005 (0.7379 / 0.6505 / 0.1129) | 69.3 % | 0.0 | 17.5 |
| 8G | router | 1 | 0.4632 (0.6963 / 0.5884 / 0.1049) | 71.7 % | 0.0 | 13.5 |
| 8G | skip, weight below 0.05 | 1 | 0.0104 (0.0097 / 0.0139 / 0.0076) | 93.8 % | 72.5 | 3.8 |
| 8G | similar, scaled | 1 | 0.4240 (0.6139 / 0.5830 / 0.0752) | 73.3 % | 1.7 | 11.8 |
| 4G | none | - | - | - | 98.5 | 3.0 |
| 4G | skip | 8 | 0.0283 (0.0415 / 0.0296 / 0.0138) | 92.2 % | 58.5 | 4.5 |
| 4G | skip | 1 | 0.6015 (0.8440 / 0.7742 / 0.1865) | 68.8 % | 0.0 | 20.8 |

What this says:

- *Skip is the rule.* At every rank and on both files it costs the least for the same reads
  saved, with one exception where a third of all picks are touched (Q4, every pick from rank
  1): there the stand-ins by similarity with the table's scale (0.42) and the router's,
  renormalised (0.46), beat skip (0.50), all three far beyond use. A stand-in by similarity with
  the missed pick's weight is the worst of the three (it adds an output that has almost
  nothing in common with the missed one); with the table's scale it is as good as skip
  (0.0108 against 0.0111) and costs more time, because its stand-ins are computed. The
  router's next best expert is in between: its probability is small, and renormalising
  shifts weight onto the picks that stay. The owner's idea does not pay on this model,
  because its experts do not resemble each other.
- *The rank is the dial, not the weight.* A bound on the pick's weight does as well as a
  rank, not better: below 0.1 it touches about what the 4th rank on does (0.80 reads per
  token against 0.71, KL 0.0111 against 0.0111), below 0.05 it saves little (3.91 reads
  against 4.52) for KL 0.0065. The rank needs no number the operator has to guess.
- *What it gives.* On IQ2_XS with 8 GiB of RAM, where 71 % of the experts have a slot
  (12 768 in VRAM, 4 602 in RAM; 62 % with 4 GiB),
  skipping from the 4th pick on takes the reads from the SSD from 4.5 to 0.7 per token and
  generation from 31 to 45 tokens per second for KL 0.011 (95.5 % of the tokens the same);
  from the 8th on (Strata's tail) 2.3 reads, 38 tokens per second, KL 0.008. With 4 GiB
  (10.1 reads per token, 15 tokens per second) the gain is larger: from the 8th, 23 tokens per
  second at KL 0.010; every pick, 49 tokens per second at KL 0.052. KL here is against the same
  file without pruning; Strata measured 0.003 to 0.006 for its tail skip on 2 to 3 bit files,
  with a decision per expert over a window where Ares decides per row.
- *The Q4 file from the SSD.* It reads 81 experts per token from the SSD without pruning
  (3.5 tokens per second). Pruning makes it faster in proportion to what it leaves out, and
  its quality falls faster than on the 2-bit file: from the 8th pick on 5.0 tokens per
  second at KL 0.026, from the 4th 8.7 at 0.077, every pick 17.5 at 0.50 (69 % of the tokens
  the same: another text). For scale, the IQ2_XS file without pruning is KL 0.29 from the Q4
  file on the same 600 positions (0.37 / 0.37 / 0.11, top-1 75 %; Strata's IQ2_XS pack, a
  different quantization, so a measure of distance, not of which is right). So a Q4 file
  read from the SSD with `skip:4` (8.7 tokens per second) stays far nearer to itself than the
  2-bit file is, at a quarter of the 2-bit file's speed with most of its experts in slots;
  with `skip:1` it is farther than the 2-bit file. Pruning makes the Q4 file usable from the
  SSD at 5 to 9 tokens per second, not fast.
