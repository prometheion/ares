# Experiments: Flash-Next prompts

Dated measurement logs, moved here verbatim on 2026-11-24 from the architecture document (now [`../architecture/`](../architecture/overview.md)) and from [`OPEN.md`](../../OPEN.md). They are not kept current: each names its date, build and machine.

## Wide steps, measured (2026-11-12)

**Wide steps, measured** (2026-11-12, RTX 4070 and RTX 3060, the engine of before (`a7cde70`,
steps of eight rows) and the one with wide steps started one after the other, both with the
n-gram drafts the server made before the draft policy, greedy, 128 tokens, prompts of about 1000 tokens (German, Rust) and 4700 (the long one: prose,
code and German), rounds one to three of the same prompts, the numbers of round three; load
average 2.4 to 9.4 for the layers and 2.6 to 10 for the two on every layer, other agents'
builds in the background; `--chat-var enable_thinking=false`, otherwise the defaults):

| | Layers split, before | Layers split, wide steps | Both on every layer, before | Both on every layer, wide steps |
| --- | --- | --- | --- | --- |
| Prompt, German 1000: tokens/s, time to the first token | 110, 9.1 s | 170, 5.9 s | 141, 7.1 s | 217, 4.6 s |
| Prompt, Rust 1000 | 83, 12.5 s | 103, 10.1 s | 100, 10.4 s | 123, 8.4 s |
| Prompt, 4700 tokens | 100, 50.4 s | 132, 38.1 s | 126, 39.8 s | 166, 30.4 s |
| Generation, German / Rust / long prompt: tokens/s | 42 / 68 / 41 | 53 / 46 / 36 | 70 / 80 / 43 | 58 / 53 / 34 |
| VRAM in use (RTX 4070, RTX 3060) | 11 703, 11 695 MiB | 11 707, 11 695 | 11 683, 11 665 | 11 697, 11 663 |
| RAM, anonymous | 11.80 GB | 11.81 GB | 11.71 GB | 11.65 GB |

The step has 256 rows (`step_rows` with a thirty-second of the memory: 221 and 179 MiB of
step buffers where eight rows took 53 and 45), which costs 0.16 GiB of expert slots on the
RTX 4070 (9.20 to 9.04 GiB). Wide steps alone make the prompts 1.2 to 1.6 times as fast, not the
five to ten times that the kernels alone would allow, and they cost the generating steps
that follow: in this series they miss 1.2 to 1.5 times as many experts (outside a slot, of the picks of
a decode step: 4.1, 12.7 and 7.3 % before and 6.2, 15.0 and 10.7 % after for the layers split;
4.3, 12.6 and 7.8 % and 5.7, 15.5 and 10.5 % for the other), and the generation speeds in the table
lie lower in five of six pairs, within a spread of the runs that is as wide as the difference
(30 to 90 tokens per second by text and run, as before). A step of 2048 rows (an eighth of the
memory, 1.4 GiB of step buffers, 1.1 to 1.2 GiB fewer slots; the layers split) read the prompts 1.3 to 1.8 times
as fast as eight rows, and generated at 47, 33, 34, 29 and 33 tokens per second where eight rows
gave 77, 41, 68, 37 and 69: the slots are worth more than the rows.

What a wide step misses (the same runs, per layer and step of 256 rows, the layers split
and the two on every layer alike): 184 to 243 distinct experts picked of 512, of which 34 to
57 (19 to 29 %) had no slot, against 39 to 45 picked and 2 to 5 of them missed in a step of eight
rows; the layers waited 31 to 62 ms per layer and step for them (the long prompt: 55 to
56), where 126 to 631 steps of eight rows waited 0.3 to 0.9 ms per layer each. In the tool, with the
same 1000 tokens read again (the experts of the text are in the tiers), 512 rows read at 388
tokens per second on the layers split and 357 on the two on every layer; 4000 tokens in slices
of 512 rows read at 112 and 145 tokens per second warm and 67 and 63 cold, in slices of 2048 at
147 and 129 warm (one layer of a slice then waits 160 to 350 ms for 110 to 300 missed experts).

## Float matrices of a wide step, measured (2026-11-12)

**Float matrices of a wide step, measured** (2026-11-12, RTX 4070 and RTX 3060; the engine
of before (`c4e0662`, the float matrices in launches of eight rows) and the one with the
float tile kernels started one after the other, two or three times; the tool, `qwen4exp_run --prefill 1000
--rows 512 --repeat 4 --breakdown` on the first 1000 tokens of this document, of which the
first reading fills the tiers and is left out; load average 3 to 22 from other work):

| | before | after |
| --- | --- | --- |
| Prompt, layers split: tokens per second (three pairs of runs) | 258, 276, 296; 256, 274, 273; 266, 274, 276 | 335, 428, 434; 415, 459, 462; 402, 450, 457 |
| Prompt, both cards on every layer | 288, 314, 314; 292, 312, 315 | 384, 422, 421; 374, 413, 416 |
| Device time of the mixers per step, layers split (25 layers on the RTX 4070, 23 on the RTX 3060) | 207.5 / 344 ms | 52.9 / 78.7 ms |
| The other dense products of the same step | 66.2 / 118 ms | 45.0 / 84.8 ms |
| The host's wait for a layer's picks, layers split | 17.1 to 17.8 ms | 6.5 to 7.0 ms |
| The same, both cards on every layer | 17.7 to 18.4 ms | 9.5 to 9.8 ms |

A layer's two mixers took 8.3 ms on the RTX 4070 and 15 ms on the RTX 3060 and take 2.1 and
3.4 ms; as kernels alone the four products of a layer's mixers are 8.6 / 13.1 ms in launches
of eight rows and 0.8 / 1.3 ms on the tile kernels (the numbers under "Float matrices"
above), so what a mixer costs now is mostly its other kernels. The router alone is worth 5
to 8 % of the prompt: with the router left on its launches of eight rows the same runs read
351 to 425 tokens per second on the layers split and the host waited 7.8 to 8.4 ms for a
layer's picks.

*What it costs in precision.* The float products of a wide step were exact f32 and now read
f16 operands (0.02 to 0.04 % rms a product, see "Float matrices"). The sums are f32 whatever
the step's precision asks for the quantized matrices: with f16 sums, which is what
`Precision::Fast` picks for a float type on these cards, the layers of the made-up model
were up to four times as far from the CPU reference (`every_layer_of_wide_steps_matches_the_reference`:
attention core 6.2e-2 against 1.8e-2 with quantized dense matrices, experts 5.3e-2 against
1.65e-2 with float ones; f16 sums lose a row of small weights among larger ones, and the
recurrent gates have such rows). With f32 sums and float dense matrices the recurrent core
is 7.4e-4 from the reference where it was 3.6e-7, and everything else is where it was
(experts 1.65e-2 after 1.62e-2, attention core 7.7e-3 after 7.9e-3; with quantized dense
matrices 1.59e-2, 0.216 and 1.84e-2 after 1.57e-2, 0.216 and 1.65e-2). What moves are the
discrete choices at near ties: the router of the made-up model with float dense matrices
picked the reference's experts in 2804 of 2816 rows of layers where it had in 2814 (2810
with the router left exact; with quantized dense matrices, whose 8-bit activations move the
router's input a hundred times as far, 2746 after 2734), and in 2 of 28 made-up models one
row of a wide step selected another block in an attention layer, which none did with exact
float products. The tests of the expert types compare the rows that made the reference's
choices and count the others (`ExpertRows` in `tests/qwen4exp_device.rs`). 

*The real file* against its recorded CPU reference (`qwen4exp_run --check` on the German and
the English trace, both cards on every layer, one run each before and after). With the given
tokens read in one wide step, the logits of the last one are 0.023 and 0.032 from the
reference's (relative RMS; 0.126 and 0.061 before) and, fed the reference's tokens, the next
token is the reference's in 160 of 160 German and 79 of 80 English rows (158 and 78 before).
In steps of eight rows, which the float tile kernels do not touch, it is 158 and 80 after 160
and 80. The rows that differ are near ties (the two logits 0.01 to 0.15 apart), and which
rows they are changes from run to run with the misses the CPU serves, as before.

## Transit, measured (2026-11-12)

**Transit, measured** (2026-11-12, RTX 4070 and RTX 3060, the engine of before (`c4e0662`:
every miss of a wide step takes a slot of its layer) and the one with transit started one
after the other, `--chat-var enable_thinking=false`, otherwise the defaults (steps of 512
rows, no drafts), greedy, 128 tokens; per round a short German question, German prose of
1000 tokens, Rust of 1034, the short question, the long prompt of 5033 tokens (prose, code
and German), the short question; the numbers of the second of two rounds; other agents'
builds ran on the machine, the load average is given):

| | Both on every layer, before (load 12 to 16) | The same, transit (load 6 to 7) | Layers split, before (load 6 to 7) | Layers split, transit (load 3.5 to 4) |
| --- | --- | --- | --- | --- |
| Prompt, German 1000: tokens/s, time to the first token | 185, 5.4 s | 255, 3.9 s | 165, 6.1 s | 231, 4.3 s |
| Prompt, Rust 1034 | 104, 10.0 s | 174, 6.0 s | 41 (77 in the first round), 25.2 s | 152, 6.8 s |
| Prompt, 5033 tokens | 141, 35.9 s | 224, 22.5 s | 112, 45.1 s | 198, 25.4 s |
| Generation after the prompt, German / Rust / long: tokens/s | 24 / 21 / 23 | 32 / 36 / 30 | 26 / 11 / 23 | 36 / 32 / 32 |
| The short question before and after the long prompt | 24, 32 | 36, 32 | 8, 28 | 34, 33 |
| Read from the file for the long prompt | 23.1 GiB | 7.7 GiB | 25.1 GiB | 9.0 GiB |

The engine of before ran at a higher load in the first pair and was disturbed in the Rust
request of the second (and the short question behind it), so its generation numbers there are
too low: at a load of 5 to 6 it generated at 35, 28 and 23 tokens per second after the three
prompts and answered the short question at 32 and 30 (prompts then: 188, 98 and 130 tokens per
second). The transit columns are of two builds of this work (`6213d28`, `e6f12f6`); the last
rules (`373b0b3`) were measured against the build before them, the tier fixed with `--ram
10G`, at a load of 3 to 4: the long prompt at 211 and 208 tokens per second the first time a
server sees it and 248 and 245 the second, German prose as the first request at 123 and 117,
generation at 34 to 40. What a server reads first after its warm-up is bound by the file and
gains least: German prose 120 before and 107 with transit in one pair, 62 and 117 in the
other; the long prompt 123 and 213, 112 and 180. Run again with the final rules and the tier
fixed, on a machine at a load of 14 to 19, German and Rust were 173 and 85 before and 222 and
152 with transit, and the long prompt 126 and 118: the load and the drive decide a run, so
the engines were always measured next to each other.

What moved (the tool, both cards on every layer, the tiers filled from the recorded routing
of other texts, then 4700 tokens of the long text read three times in steps of 256 rows, the
tier fixed at 9 GiB, load 17 to 31): before, every reading uploaded 53 000 to 57 000 experts
into slots, evicted as many, copied 29 000 to 35 000 of them down to RAM and read 19 000 to
24 000 from the file (26 to 33 GiB), the third reading of the same text like the first; the
layers waited 32 to 80 ms each (per part) for their misses, 74 to 89 % of a layer's time,
and it read at 37, 64 and 90 tokens per second. With transit the three readings read 25, 14
and 14 GiB from the file, 20 000, 600 and 80 experts took a VRAM slot, the layers waited 67,
15 and 15 ms (86, 60 and 60 %), and it read at 45, 168 and 169 tokens per second. A slice of
256 rows misses 29 to 35 of the 124 experts it picks per layer and part either way: the slots
hold what they hold. A miss whose bytes are in RAM costs the link and no more (0.25 to 0.3
ms an expert over both cards, 0.07 to 0.12 ms on the RTX 4070 and 0.5 to 0.7 on the RTX
3060, against 0.9 to 2 ms before), so the churn was the cost and not the link; the RTX
3060's link (8 GT/s, four lanes) is what the misses of its layers wait for now. The first
reading of a short prompt on tiers of other texts is the case transit does not help and
sometimes hurts (1000 tokens: 86 before and 35 with transit at that load, 55 and 49, 107 and
84 on a quieter machine): nearly every miss takes a slot then and most of what they evict is
copied down; why that costs more than the old path's own copies is not understood.
Generation right after a reading, in the tool: 41.5 and 15.3 tokens per second before (after
1000 and after 4700 tokens, the second at a load of 29), 33.9 and 24.1 with transit.

- *The transit slots.* 16, 32, 64 and 128 slots per device read a second reading of 2500
  tokens at 200, 207, 215 and 223 tokens per second (steps of 512 rows, load 11 to 15; the
  first reading is bound by the file and shows the load: 86 to 98) with 4.3, 2.5, 1.45 and
  1.02 rounds after the first per layer. The plan's rule (half of the experts of a layer
  that have no VRAM slot: 56 to 58 here, 82 MiB per device) is where the rate stops growing
  for the memory.
- *Reading ahead on a guess.* With the guess (what the wide step before missed in the layer
  and no tier holds) a reading of 2500 tokens read 14.0 to 15.0 GiB from the file against
  12.3 to 12.4 without and a second reading 4.0 to 5.7 against 2.9, and ran at 69 and 96
  tokens per second against 102 and 107, and 221 and 214 against 259 and 277 (on, off, on,
  off; load 25, 12, 9, 5). Of the guessed reads 48 % were used in a first reading and 8 to
  14 % in a second: what a step read and the heat keeps is in RAM afterwards, so what is
  left to guess is mostly what one row used.
- *The rows of a step.* 4700 tokens in steps of 256, 512 and 1024 rows (the tool, a second
  reading, load 4 to 6): 100, 112 and 126 tokens per second, while the layers of a step take
  0.41, 0.86 and 1.77 s before any expert (2.9 s at 2048 rows). More rows bring each expert
  for more rows at once, 12 % of rate for twice the rows, and keep every other request
  waiting twice as long.

What bounds the prompt now:

- *Bringing the missed experts.* A slice of a long prompt picks 40 to 80 % of the experts of
  every layer, the tiers hold 55 % in VRAM and 30 % in RAM, so every slice misses a quarter
  to a third of what it picks, and the next slice misses most of them again: they are
  uploaded again at the link's speed (3 GiB/s on the RTX 3060), which is 60 % of a layer's
  time when their bytes are in RAM. What no tier holds is read from the file when the
  record names it, all reads of a layer at once, and the layer waits for the last: the
  file gave 0.7 to 1.0 GiB/s on this machine while others used the drive, and a first
  reading takes 4 to 13 GiB from it.
- *The dense part of a layer.* The host waited 18 to 40 ms per layer and slice of 512 rows for the
  router's picks; that is the device's time through the mixers, the projections and the
  recurrent layer. The mixers' matrices are BF16; eight rows at a time over four streams
  they took 4 ms a read on the RTX 4070 and 7 on the RTX 3060 at 512 rows (the fused read
  took 7.5 and 24). On the float tile kernels a read takes 1.1 and 1.7 ms and the host waits
  7 to 10 ms for the picks (see "Float matrices of a wide step, measured" above).
- *The host in every layer.* The first round runs while the picks travel to the host, but the
  next layer cannot be enqueued before the host has the misses of this one: a layer without
  misses costs one round trip of the record (about 0.1 ms); the waves would take that out too.

## Large prompt steps on lent VRAM, measured (2026-11-22)

*Measured* (2026-11-22, the IQ2_XS file, the server, `--ram 8G`, `--context 40960`, both cards
on every layer through the driver with peer access; a prompt of 8016 tokens and one of 33 323
of this document read with `max_tokens` 1, each server started from the same usage profile;
the old path is the same build with `-b 512`, which takes the steps a prompt took before; each
pair of runs in one turn of the GPU, the load given; files read are the store's bytes from the
file, uploads its copies into VRAM and transit slots):

| Pair, load | Path | 8k: tokens/s, read, uploads | 32k: tokens/s, read, uploads |
| --- | --- | --- | --- |
| 1, 5.6 to 6 | steps of 512 rows | 191.4, 36.7 GiB, 52 625 | 239.3, 138.6 GiB, 205 885 |
| 1 | lent steps of 4096 rows | 309.6, 22.9 GiB, 22 901 | 298.2, 84.2 GiB, 92 840 |
| 2, 4.4 to 4.6 | lent steps of 4096 rows | 342.7, 22.9 GiB, 22 855 | 395.1, 84.4 GiB, 92 794 |
| 2 | steps of 512 rows | 203.9, 36.8 GiB, 52 712 | 210.8, 139.1 GiB, 205 959 |

On one card (the RTX 4070, the same file and `--ram 8G`, load 3.9 to 5.2) the prompt of 8016
tokens read at 49.1 tokens/s in steps of 512 rows, 148.5 GiB from the file and 151 328 uploads,
and at 206.2 tokens/s in lent steps of 4096 rows, 42.5 GiB and 39 199 uploads (a second pair,
lent steps first, at a load of 17 to 33: 199.3 against 40.1 tokens/s, the same bytes): where the
tiers hold little, every step of 512 rows reads most of the file again.

What a prompt reads from the file by the rows of its lone steps (both cards, a prompt of
31 992 tokens; the bytes do not depend on the load, the speeds do): 512 rows about 139 GiB,
2048 rows 98.3 GiB, 4096 rows 81.5 GiB.

The plan's estimate took 4096 rows on these cards (8192 would need 5.7 GiB of the mixer's 8.2
GiB of slots, more than its largest pool lends above the floors). Prompts read 1.6 to 1.7
times as fast at 8k and 1.25 to 1.9 times at 32k, with 38 % fewer bytes from the file and 55
to 57 % fewer uploads; what is still read is what the lent slots held (the loan takes about
half of the mixer's slots for the prompt) and what a 4096-row step picks that the tiers do not
hold, again for each step. Without the driver's peer access (the layers split, the run of the
same morning) the old path read the same prompts at 56 to 125 and 136 to 172 tokens/s,
reading 37.9 and 145.7 GiB.

*A request behind a prompt read alone* (the 32k prompt; 20 s after it, a request of 40 tokens
for 8 tokens, streamed; the same server pair, load 4.7 to 5.5): with steps of 512 rows its first
token came after 2.68 s and the long prompt took 134.6 s (241 tokens/s); with lent steps of 4096
rows after 13.47 s, the long prompt 85.4 s (380 tokens/s). The request waits for the lent step
in flight, about 11 s here, and then for a piece of its own (with lone steps of 1024 and
2048 rows, measured while other work loaded the machine to 13 to 23, it waited 9.5 and 12.3 s,
the long prompt read at 147 and 208 tokens/s); a step cannot be cut, so what the
owner asked for (no request waits seconds behind a long prompt) needs the interruptible pass
of "Long prompts layer by layer" in `../architecture/flash-next-prompts.md`, or a lone step bounded in time.

## Long prompts layer by layer, measured (2026-11-23)

*Measured* (2026-11-23, local, the IQ2_XS file through the server, `--ram 8G`, `--context
40960`, a fresh cache and no usage profile; a prompt of 17 000 tokens of this document with 64
tokens after it, then the same length of other text with a request of about 40 tokens for 8
tokens sent 10 s after it; the base is `flash/prompt` (a44474e, lent steps of 4096 rows on two
cards), both built alike; alternating, the load 2 to 6):

| Cards, round | Build | 17k prompt | Read from the file | Uploads | Tokens/s after it (2-16, 17-32, 33-64) | The short request's first token | The long prompt meanwhile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Two, 1 | passes | 45.5 s | 53.8 GiB | 43 151 | 11.9, 17.6, 15.5 | 2.45 s | 60.9 s |
| Two, 1 | base | 55.7 s | 64.2 GiB | 50 628 | 7.5, 9.2, 13.9 | 20.7 s | 51.2 s |
| Two, 2 | passes | 44.6 s | 54.1 GiB | 43 335 | 11.8, 17.2, 15.0 | 0.91 s | 42.0 s |
| Two, 2 | base | 52.3 s | 60.2 GiB | 47 171 | 9.9, 15.8, 12.9 | 58.0 s | 104.1 s |
| One, 1 | passes | 53.5 s | 72.9 GiB | 57 517 | 8.8, 13.3, 9.8 | 1.68 s | 55.9 s |
| One, 1 | base | 83.8 s | 78.8 GiB | 63 992 | 4.8, 9.9, 8.9 | 27.7 s | 66.2 s |
| One, 2 | passes | 54.9 s | 73.5 GiB | 58 186 | 8.7, 12.7, 9.4 | 1.72 s | 55.4 s |
| One, 2 | base | 67.1 s | 79.0 GiB | 63 873 | 5.9, 10.5, 9.3 | 27.4 s | 65.9 s |

These passes were still cut where checkpoints were due (16 384 and the anchor), and the two
steps after the cuts read most experts again. With the checkpoints saved inside the pass (the
same day, one round each, the load 1.5 to 3.6):

| Cards | Build | 17k prompt | Read from the file | Uploads | Tokens/s after it (2-16, 17-32, 33-64) | The short request's first token | The long prompt meanwhile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Two | passes | 26.0 s | 13.3 GiB | 13 420 | 15.2, 18.6, 16.4 | 0.77 s | 35.5 s |
| Two | base | 54.6 s | 63.5 GiB | 50 277 | 9.4, 14.1, 13.7 | 20.7 s | 68.5 s |
| One | passes | 20.9 s | 25.5 GiB | 22 797 | 9.7, 11.0, 9.8 | 1.76 s | 23.0 s |
| One | base | 62.2 s | 79.4 GiB | 64 861 | 5.3, 10.3, 9.3 | 48.8 s | 136.2 s |

The prompt reads 2.1 times as fast on two cards and 3.0 times on one, with a fifth to a third
of the reads on demand (the reads ahead of the next layer are not in that column) and of the
uploads; the tokens right after it are 1.2 to 1.3 times as fast (the loan of a pass of 17 000
rows is about 0.9 GiB where a lent step of 4096 rows took 2 GiB of slots), and a request that
comes during the prompt gets its first token after 0.8 to 1.8 s instead of 21 to 49 s.
