# Experiments: two cards

Dated measurement logs, moved here verbatim on 2026-11-24 from the architecture document (now [`../architecture/`](../architecture/overview.md)) and from [`OPEN.md`](../../OPEN.md). They are not kept current: each names its date, build and machine.

## Qwen3.8-27B on the two cards, measured (2026-11-10)

Measured 2026-11-10 with `crates/model/examples/qwen35_run.rs`, one
sequence, 8-bit activations, K and V in `q8`, the peer route. The llama.cpp
fork on the same cards: about 1180 prompt tokens per second at 2048 tokens,
28 ms per token without drafts (35.7 tokens per second), and 62 to 67 tokens
per second with drafts from the MTP block and its production flags.

| | Tensor split 0.625 / 0.375 | Tensor split 0.7 / 0.3 | Layer split 0.62 / 0.38 | Fork |
| --- | --- | --- | --- | --- |
| Prompt of 2048 tokens | 1218 tokens/s | 1461 to 1474 | 1348 | about 1180 |
| Decode without drafts, 1000 tokens of context | 24.8 ms (40.4 tokens/s) | 26.2 ms | 41.4 ms | 28 ms |
| With the model's drafts, German / English / code | 79.3 / 75.6 / 79.0 tokens/s | | not built | 62 to 67 |
| The same, sampled | 76.4 / 72.4 / 83.4 tokens/s | | | 63 to 66 |
| VRAM taken at a context of 4096 | 9500 / 5410 MiB, with the MTP block 10064 / 5540 | 10590 / 4362 MiB | 9582 / 5940 MiB | |

The fork's numbers with drafts were measured on the same day on the same
questions by running it with a copy of its production preset; the other fork
numbers are older measurements of the same setup.

## The tensor split's step scratch and draft link, measured (2026-11-12)

The same day, same cards, the speed of three trees: the commit before
(b245771), the change as it was first merged (e19be3b) and the tree with the
step scratch put right (below), run in turn two to six times each while
other builds ran on the machine (load 3 to 25). Staging a prompt step of 512
rows was timed by itself: 0.12 ms to 0.31 ms for the host to copy the rows
and 0.10 ms to 0.19 ms for the upload, of a step of 380 ms.

| Run | Before | As first merged | Now |
| --- | --- | --- | --- |
| IQ2_XXS, one card: prompt of 2048 tokens, ms | 1749 to 1751 | 1749 to 1757 | 1748 to 1752 |
| IQ2_XXS, one card: decode step, ms | 20.88, 20.94 | 20.92, 20.93 | 20.89, 20.95 |
| IQ2_XXS, two cards: prompt of 2048 tokens, ms | 1693 to 1711 | 1717 to 1737 | 1693 to 1727 |
| UD-Q4_K_S, two cards: prompt of 2048 tokens, ms | 1502 to 1520 | 1520 to 1538 | 1501 to 1529 |
| UD-Q4_K_S, two cards: decode step, ms | 24.76 to 24.83 | 24.70 to 24.79 | 24.73 to 24.77 |
| UD-Q4_K_S, two cards, drafts with the vocabulary: tokens per second, mean of seven prompts of 300 tokens | 97.8 to 100.4 | 97.7 to 98.9 | 98.1 to 98.9 |
| the same: a link of the draft chain, ms | 1.43 to 1.46 | 1.48 to 1.51 | 1.48 to 1.51 |

On one card nothing changed, and a prompt on the tensor split takes what it
took. A link of the draft chain takes 0.05 ms longer than before, for a
reason of its own (the last paragraph of this section). Of the tokens per
second one run of each of the two later columns is left out, 89.7 and 92.9,
taken while the load rose from 5 to 17. On a layer split of the IQ2_XXS file
(shares 0.5 and 0.5), where the change moved two blocks to the first card,
the same prompt takes 2157 ms to 2171 ms before, 2085 ms to 2095 ms as first
merged and 2064 ms to 2075 ms now (load 3 to 6), a decode step 27.38 ms and
27.44 ms, 26.93 ms and 26.97 ms, 26.92 ms and 26.95 ms.

**What the tensor split lost at first.** As first merged, the change made a
prompt on two cards 0.7 % to 1.8 % slower, in steps of 128 rows 0.2 % to
0.4 %, and the profile of a prompt step showed the time in the weight
products of the second card, which holds the bytes it held. It was neither
the table in page-locked memory nor its upload, the exchange between the
cards or the layouts: it was where the second card's step scratch begins.
The stage of the staged rows is the first buffer of the scratch, and a part
that stages nothing still took 64 bytes for it, so on the second card every
buffer behind it began 64 bytes later than before. Among them are the f16
activations of the weight products. Their rows lie a whole number of lines of
128 bytes apart, counted from the start of their buffer, and a step of the
tile kernels reads 64 values of each row: that is one line when the buffer
starts on a line and two when it starts between two. On the first card the
stage is a multiple of 128 bytes in both files, which is why one card showed
nothing.

Found with a probe build that moves one buffer of the second card's scratch
at a time (2026-11-12, UD-Q4_K_S on the same split, prompt of 2048 tokens in
steps of 512 rows, both passes of three runs in turn with the others at load
8 to 15; the two rows marked * are from a batch of three runs whose first
ran while the load rose from 5 to 18 and is left out, 1524 ms and 1531 ms,
1514 ms and 1544 ms):

| The second card's step scratch | Prompt, ms |
| --- | --- |
| as before the change: nothing in front of the first buffer | 1501 to 1508 |
| as first merged: every buffer 64 bytes later | 1514 to 1523 |
| only the f16 activations 64 bytes later | 1516 to 1531 |
| only the 8-bit activations 64 bytes later (the f16 ones 128) * | 1497 to 1513 |
| every buffer but the prepared activations 64 bytes later | 1500 to 1510 |
| every buffer 128 bytes later * | 1501 to 1510 |

So the step scratch now starts every buffer on a line
(`Kernels::ACTIVATION_LINE`, which is also the 128 of the pitch of the rows),
and a part that stages nothing has no stage (`Carve` in `device/part.rs`).
Before, the activations lay on a line only because every size in front of
them happened to be a multiple of 128 bytes; now where a buffer starts does
not depend on the sizes of the others. The VRAM is what the table above
says: the first card holds 518 MiB less without drafts (10102 MiB to
9584 MiB taken at a context of 32 768 tokens in the default K/V format) and
the second what it held (5790 MiB, 5792 MiB).

**The link of the draft chain** takes 1.48 ms to 1.51 ms where it took
1.43 ms to 1.46 ms (UD-Q4_K_S on the split, drafts with the vocabulary),
which is 0.2 ms to 0.3 ms of a step with four drafts (its median 33.27 ms to
33.36 ms before, 33.55 ms to 33.59 ms now) and less in tokens per second
than two runs of one binary differ by. The chain runs on the first
card only, and neither the gather nor the staged row of the newest token
costs it: with the same layout forced for every matrix the link is the same
before and after the change (1.48 ms to 1.50 ms against 1.48 ms to 1.51 ms in
the device layout, 1.45 ms to 1.48 ms against 1.45 ms to 1.46 ms in the
file's). What changed is the choice of the layouts on the first card ("Device
layouts in the model"): before, the draft head (q6_K, 66 343 rows) and the
two wide matrices of the MTP block's feed-forward part (q6_K, 17 408 rows)
took the file's layout, now the device layout. The chooser times each layout
in three rounds on a buffer as large as the device's weights and takes the
fastest. For these matrices at one row every round of the file's layout
measures 0.1434 ms to 0.1441 ms and every round of the device layout
0.1423 ms to 0.1427 ms, except that with a buffer of the old size (9470 MiB)
one round of the file's layout comes out at 0.1381 ms and 0.1367 ms (the
result repeats to 0.0005 ms from load to load) and decided for it. With the
buffer the table no longer takes part of (9088 MiB) there is no such round.
Giving the probe of the present tree a buffer of the old size, and nothing
else, brings back the old choice and the old link (1.43 ms to 1.44 ms). The
old choice was the faster one for the chain, but it was not made on what the
probe usually measures: the probe has the two layouts within one percent of
each other for these matrices, and the chain runs 0.05 ms faster per link
with the file's. A round that is 2 % or more faster than the other two of
its product occurs in 7 to 9 of the 232 products timed on the first card, in
other products with another size of the buffer. This is left as it is:
getting the faster link back for a reason would take a probe that reads the
MTP block's matrices as a chain reads them (several times a step, a few
hundred megabytes apart) and a choice that does not rest on one round of
three, and either changes the layouts of other matrices on both cards, which
has to be measured as a change of its own.

## Qwen3.8-Flash-Next on the two cards, measured (2026-11-11)

Measured 2026-11-11 with `crates/model/examples/qwen4exp_run.rs`, one
sequence, no drafts, a missing expert uploaded. Strata on the same machine
(its log of 2026-11-07): 128 tokens generated at 12 to 76 tokens per second,
median 56, with drafts; prompts read at 42 to 201 tokens per second.

The model is 63.5 GiB on a machine with 24 GB of VRAM and 31 GB of RAM. The
dense parts (3.4 GiB) are in VRAM, 14 100 of the 24 576 experts have a VRAM
slot (19.4 GiB), a RAM tier holds some thousands more, and the rest is read
from the file when the router picks it. On recorded text 99 % of the picks
find their expert in a slot.

| Placement | Decode step at 1000 tokens, median | Fastest | At 8192 tokens |
| --- | --- | --- | --- |
| By layers (0..25 on one card, 25..48 on the other) | 25.2 ms | 21.8 ms | 31.1 ms *(agent)* |
| Both cards on every layer | 22.3 ms (19.6 on a quiet drive *(agent)*) | 16.5 ms | 21.8 ms *(agent)* |

In the first placement the cards work one after the other. In the second
one card mixes the residual streams and runs the router, attention and the
head, the other computes its share of the recurrent heads and of the experts
of every layer, and one sum per half block joins them; no weight is on both.
A step that misses no expert takes about 17 ms. The median is above that
because an expert that is in no slot now holds both cards while it is
fetched, and an upload to the RTX 3060 takes 0.4 ms over its link.

Against the CPU reference the largest logit of a prompt's last token is the
reference's in every run, 9 or 10 of the 10 largest agree, and the next
token is the reference's in 159 or 160 of 160 German and 79 of 80 English
rows; the rows that differ are near ties. The tool steps the model through a
session of its own, with eight rows per step unless `--rows` says more: prompts
are read at 54 to 73 tokens per second *(agent)* in steps of eight rows, bound by
experts that have to be fetched, and at 357 to 388 (1000 tokens) in slices of 512
rows on the many-row kernels (see "Wide steps" below). The server runs it through
the runtime (see below).

An expert that is in no slot can be computed on the CPU instead of being
uploaded, chosen per expert by measured cost. On the recorded routing with
few slots that took the expert path from 33 to 44 steps per second; with 99
% of the picks in a slot it makes no measurable difference to the model.

## The 27B's attention heads by context, measured (2026-11-22)

**The 27B, measured** (2026-11-22, `qwen35_run --attention-shares`, UD-Q4_K_S,
K and V in `q8`, no drafts, one sequence opened with 33 024 cells, runs in turn
T, A, L, B, T, A, L, B, then T, M, T, M; load average 4 to 10 for the rows
given). *All of it through page-locked host memory*: the machine had no driver
library with peer access that day, so a decode step of the plain tensor split
takes 26.3 ms where it took 24.8 ms over the peer route; the differences
between the rows are what this measures. Context is what the devices' free
memory holds in all with 256 MiB left on each (`RES room`); a prompt is tokens
per second (2048 tokens from nothing, and reading 32 000); a decode step is the
median in ms of 64 steps at 1000 and at 32 000 cells of context.

| Placement (weights; heads) | Context | Prompt 2048 | Decode 1k | Read 32k | Decode 32k |
| --- | --- | --- | --- | --- | --- |
| T: tensor split 0.625 / 0.375; heads with the weights | 126 825 (the 4070 is full) | 1349, 1358 | 26.39, 26.28 | 943, 949 | 29.22, 29.17 |
| A: 0.625 / 0.375; heads 1 : 3, as the fork's production | 234 513 (the 3060 is full) | 1264 | 27.57, 27.45 (+4 %) | 753, 756 (-20 %) | 32.69, 32.63 (+12 %) |
| M: 0.625 / 0.375; heads by room, 0.33 : 0.67 | 265 114 (both full) | 1278, 1270 | 27.21, 27.19 (+3 %) | 787, 782 (-17 %) | 31.81, 31.85 (+9 %) |
| B: 0.7 / 0.3; heads 1 : 3 | 242 228 (both nearly full) | 1474, 1480 | 28.57, 28.52 (+8 %) | 828 (-12 %) | 33.65 (+15 %) |
| L: layer split 0.62 / 0.38 | 113 742 (the 4070 is full) | 1228, 1232 | 41.17, 41.21 (+56 %) | 855 | 45.26, 45.32 (+55 %) |

The same with peer access, once it was back that afternoon (one run each in
turn, load average 13 to 24):

| Placement, over the peer route | Context | Prompt 2048 | Decode 1k | Read 32k | Decode 32k |
| --- | --- | --- | --- | --- | --- |
| T | 126 729 | 1353 | 24.54 | 948 | 27.39 |
| A | 234 433 | 1266 | 25.39 (+3.5 %) | 756 (-20 %) | 30.54 (+11.5 %) |
| M | 265 025 | 1275 | 25.22 (+2.8 %) | 787 (-17 %) | 29.72 (+8.5 %) |

K and V take 20.75 / 12.45 MiB per 1000 tokens on the 4070 / 3060 in T, 8.30 /
24.90 in A and B, 10.89 / 22.31 in M and 22.83 / 10.38 in L. With K and V in
`q4` T holds 228 562 tokens and M 482 707: the model's own context of 262 144
fits there with a few heads moved. The 4070 holds
62.5 % of the weights and, with the heads following them, 62.5 % of K and V:
it is full with 2.2 GiB to give while the 3060 has 6 GiB nobody uses. Moving
heads to the 3060 doubles the context (M: 2.1 times), at a price that grows
with the length of the sequence, because the 3060 then reads most of K and V
of every step: 3 to 4 % per decode step at 1000 cells and 9 to 12 % at
32 000, and 17 to 20 % for reading a long prompt. Shifting weights to the
4070 as well (B) makes short prompts 10 % faster and decode slower. The layer
split is no use for one sequence: its decode step is half as fast again and
its context is the smallest.

So the heads are a choice between context and the speed of long sequences,
and the room decides how far they must move: with the most context as the
goal, heads by room (`attention_by_room`: device `i` gets the share of
the heads that its room without heads is of all rooms, which makes every
device full at the same page: M's 0.33 is what the rule gives from the rooms
measured in A, and there both cards had room for 232 000 to 236 000 more
tokens),
and for a context that fits with fewer moved, only as many as it needs
(`heads_for_context`: the weights' shares lowered where a device's room is
short, `fit_shares`; all three in `ares_model::qwen35::device`, `plan.rs`, with
the model's estimate of what each device takes, `Estimate`). That is how the
plan places the heads
(`AttentionShares::ByContext`), with no option of its own (the owner,
2026-11-22): the context steers it, `--context` or, without it, the model's
own, and when that does not fit wherever the heads are, the heads go by room
and the context is what VRAM holds; a `--context` the devices cannot hold is
refused by estimate before anything is loaded, with the context they do hold.
The room of a device is what it has once it holds everything but the heads,
the K and V of the MTP block included (it has heads of its own, on the device
with the first rows of the head: 296 MiB for 262 144 tokens in `q5`), and the
heads' cost counts the scratch that grows with their weights. The heads of a
block are whole, so a device can hold up to a head more than its share (about
100 MiB of the 27B's at that context): the plan checks the shares against the
footprint of the placement they make and moves heads off a device that is
over until none is. (The owner's production settings, `--shares 0.625,0.375
--context 262144 --parallel 2 --kv q5` with the MTP block, planned the RTX
4070 20 MiB above its free memory with 0.363 of the heads before, and the load
died; the plan now gives it 0.320 and 11 491 of its 11 796 MiB free,
`qwen35/device/plan.rs` `the_27b_at_the_owners_settings_plans_within_free_memory`.) Shares
given (`AttentionShares::Given`) are for tests and measurements.

## Flash-Next's second card as an expert tier, measured (2026-11-22)

*Measured* (2026-11-22, `qwen4exp_run`, the IQ2_XS file, `--ram-gib 8`, 4096
cells, the peer route, two rounds in turn FT, FX, FL, FO at a load average of
4 to 7; a decode step at 1000 cells of a technical English text, 128 steps; a
prompt of 1000 tokens in steps of 512 rows, read the second time, when the RAM
tier holds its experts):

| Placement | VRAM slots, 4070 + 3060 | Prompt, tokens/s | Decode, median ms | Mean | Fastest | Picks in a slot | Per step from RAM / the file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FT: both cards on every layer as measured at load (`core_share` 0.75, the shared expert on the worker) | 5712 + 7872 | 494, 500 | 22.7, 22.9 | 24.2, 28.4 | 17.4, 17.3 | 97.7, 96.4 % | 7.9 / 3.1, 11.8 / 5.3 |
| FX: the second card holds experts only (`core_share` 1, the shared expert on the mixer) | 5424 + 8208 | 624, 633 | 20.6, 23.4 | 21.9, 28.5 | 18.0, 18.1 | 98.8, 96.7 % | 3.9 / 1.8, 10.7 / 5.1 |
| FL: a range of layers each (0..25, 25..48) | 6825 + 6578 | 604, 612 | 31.3, 28.2 | 36.9, 31.6 | 23.8, 23.7 | 96.2, 98.0 % | 12.5 / 5.6, 6.8 / 2.6 |
| FO: the 4070 alone | 5280 | 84, 88 | 43.8, 48.9 | 48.4, 51.7 | 21.3, 23.9 | 80.3, 82.7 % | 80.6 / 13.9, 67.5 / 15.5 |

The second card is worth most on every layer: its VRAM is a tier of 8000
experts (picks from RAM and the file fall to a seventh) and it computes what it
holds, where by layers a sequence waits for one card after the other. Whether
it computes a share of the dense part as well hardly matters for decode, which
the misses decide (the step that misses nothing is 4 % faster with the share);
without it the 3060 is a pure expert tier, holds 336 experts more, skips the
exchange of every mixer half, and a wide step is 27 % faster, because the
slower card no longer computes recurrent heads for 512 rows.

## The second card's role timed at load, measured (2026-11-23)

*Measured* on the two development cards (2026-11-23, `qwen4exp_run`, the IQ2_XS file,
`--ram-gib 8`, 4096 cells, the peer route, as in the table above; R lets the trial choose,
FT and FX give the roles; three rounds in turn, the load average in brackets):

| Run | Trial: decode, wide step of 512 rows (the share; experts only) | Opened | Prompt, warm | Decode median / mean / fastest | In a slot |
| --- | --- | --- | --- | --- | --- |
| R (3.5) | 16.43 ms, 604.5 ms; 17.24 ms, 494.9 ms: experts only kept | 8.8 s | 607 | 22.8 / 24.2 / 18.1 | 98.0 % |
| FT (3.5) | | 2.7 s | 488 | 22.5 / 26.4 / 17.3 | 97.5 % |
| FX (3.7) | | 3.8 s | 603 | 23.6 / 26.4 / 18.0 | 97.8 % |
| R (8 to 6) | 16.30 ms, 704.8 ms; 17.07 ms, 486.6 ms: experts only kept | 5.9 s | 595 | 22.9 / 24.4 / 18.1 | 98.3 % |
| FT (6 to 4) | | 4.8 s | 480 | 24.6 / 31.7 / 17.3 | 96.6 % |
| FX (4) | | 2.2 s | 621 | 22.6 / 26.1 / 18.0 | 97.7 % |
| R (5) | 20.53 ms, 745.3 ms; 17.11 ms, 487.0 ms: experts only kept | 5.9 s | 605 | 20.7 / 22.3 / 18.1 | 98.5 % |
| FT (4) | | 2.7 s | 502 | 21.0 / 22.0 / 17.4 | 98.4 % |
| FX (3) | | 2.2 s | 604 | 21.8 / 22.7 / 18.0 | 98.5 % |

The trial sees what the runs show: with the share a decode step that misses nothing is 5 %
faster (16.3 against 17.1 to 17.2 ms in two of the three; the runs' fastest steps 17.3
against 18.0), and a wide step 22 to 53 % slower; experts only reads a warm prompt 20 to 29 %
faster, and its decode is as fast (medians within a millisecond either way, which the misses
decide), so the choice is right on these cards. Every trial so far kept experts only, also two at a load
average of 20 to 28 whose times were 20 to 70 % longer. *What it costs:* 4.1 and 4.2 s in
the quiet rounds (the first role 2.6 to 2.7 s, of which 1.6 s opening its parts with the
paths of the prefill kernels measured on both cards, which the load's parts then take
without measuring again; the second role 1.4 s), 7.0 s in the first round (with the
full-width wide step that warmed up, since replaced by one of sixteen rows); the load opens
3.2 to 3.7 s later than with the role given (5.9 s against 2.2 to 2.7). That is more than the second or two expected: each role opens its
parts, uploads its dense weights (3.4 GiB on the mixer) and runs a wide step of 512 rows.

The model's own context (262 144 cells, what the plan serves without
`--context` since 2026-11-22) cost a fifth of the VRAM slots until its K and
V overflowed into RAM (see "K and V beyond VRAM"; now the keys and a
thirty-second of the room stay in VRAM): its K/V and
indexer keys take 13.92 MiB per 1000 cells, 3.56 GiB, all on the card that
mixes when both cards work on every layer (the 4070 keeps 2976 of its 5712
slots in FT, 2688 of 5424 in FX) and half on each by layers. The same runs
with 262 144 cells (one round, load average 8 to 16): FT 10 800 slots, a
decode step of 27.7 ms (median; mean 32.7), prompts 263 tokens/s; FX 10 896
slots, 24.9 ms (31.5), 333; FL 10 573 slots, 33.1 ms (35.4), 310. Decode is 11
to 22 % slower and a warm prompt half as fast as with 4096 cells.
