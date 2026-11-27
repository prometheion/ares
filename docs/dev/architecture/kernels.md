# Kernels

**One table of weight types** (`src/kernels/types.rs`). What the kernels of each of the 28
weight types are and need is one row of `TYPES`, which every list of types in the crate is
read from: the alignment of its decode kernel and of its exact kernel, whether its tile
kernels have the int8 variant (those that do are in `prefill.cu`, the others in
`prefill_more.cu`), their alignment and table bytes, up to how many rows the decode kernels
are the faster family, whether the f16 path with f16 accumulation was as fast as int8 on the
capped cards, the table bytes of a gate and up decoder written for the many-row expert
kernels, and whether qwen4exp's fused read has its decoder inlined. A kernel's name is the
type's as ggml spells it (`ares_mmvq_q4_K`, `ares_er_down_q2_0_n1`), the types with a device
layout are those `ares_quant::layout` lays out, and two facts are read from the kernels when
they are loaded, so that the host and a kernel cannot disagree: whether a decode kernel walks
several tiles of rows per thread block (its static shared memory, the codebook, is at least
16 KiB: IQ1_S and IQ1_M; its launch then has fewer blocks than the matrix has tiles), and
which module of the many-row expert kernels holds a type's kernels.

**Decode weight kernels.** A matrix times 1 to 8 rows of activations, which
is what decides decode speed, for all 28 weight types (`kernels/mmvq.cuh`,
`mmvq_more.cuh`, `mmvq_float.cuh`). For the quantized types the activations
are quantized once per input to int8 with a scale per group of 32, and the
kernels compute integer dot products (`__dp4a`) straight on the weight codes;
scales and offsets are applied once to the integer sum, never per value. The
float types (f32, f16, bf16) multiply the exact activations, of which the
prepared buffer holds an f32 copy. Each kernel is verified tightly against a
CPU emulation of the same arithmetic and loosely against the exact f32
product: about 1 % rms for the quantized types, from quantizing the
activations, and nothing beyond f32 rounding for the float types. Real tensors
of the test models cover 17 types. The other 11 (q4_0, q4_1, q5_0, q5_1,
q2_K, iq1_s, tq1_0, tq2_0, mxfp4, nvfp4, q1_0) are tested on generated blocks
only, because no model at hand carries them.

Measured 2026-11-10 against the memory limit of each card (479 GB/s on the
RTX 4070, 341 GB/s on the RTX 3060, measured with a plain reading kernel), at
17408 x 5120, in percent of the limit, with the decode kernels alone (the tool
now: `examples/few_rows_bench.rs all:17408x5120`, which times what a model runs,
the tile kernels above a type's decode rows among it):

| Types | RTX 4070, 1 row | 8 rows | RTX 3060, 1 row | 8 rows |
| --- | --- | --- | --- | --- |
| 4 bits and more, floats (q4_K, q5_K, q6_K, q8_0, iq4_xs, iq4_nl, q4_0, q4_1, q5_0, q5_1, mxfp4, nvfp4, f32, f16, bf16) | 95 to 100 | 71 to 98 | 93 to 100 | 46 to 92 |
| 3 bits (q3_K, iq3_s, iq3_xxs) | 87 to 96 | 39 to 63 | 63 to 82 | 26 to 44 |
| 2 bits and fewer (q2_K, iq2_s, iq2_xs, iq2_xxs, iq1_s, iq1_m, tq2_0, q1_0, q2_0) | 65 to 94 | 29 to 57 | 47 to 88 | 19 to 38 |
| tq1_0 | 14 | 10 | 9 | 7 |

- One row of Q4_K takes 0.107 ms on the 4070 and 0.150 ms on the 3060.
- The types of 4 bits and more hold 90 % or more up to four rows on the 4070.
  The low-bit types fall off from two rows on, because their work is integer
  arithmetic (a `dp4a` per four weights and row), not memory. They need the
  tensor cores from about three rows on, which is the register kernel of the
  prefill family below.
- A weight load narrower than a 32-byte sector must not bypass the L1 cache
  on these cards, or it requests its sector from L2 again with every
  instruction. Fixing that took iq4_xs at eight rows from 322 to 428 GB/s on
  the 4070 and from 163 to 244 on the 3060.
- TQ1_0 is bound by the loads of its activations (five scattered words and
  five scales per lane) and is the one type far from the limit at one row.
- Hardware counters (`ncu`) are blocked for non-root users on the development
  machine, so what bounds a kernel was established by ablation and `nsys`.

A pass over all weight matrices of Qwen3.8-27B (13.8 GiB, 96 % of the file,
split 0.625 / 0.375 over the two cards) takes 22.0 ms with one row, 23.2 ms
with three, 24.4 ms with four and 34.6 ms with eight (38.0 ms before the load
fix). That is a lower bound for a decode step
(no attention, recurrence, sums or sampling); the llama.cpp fork needs 28 ms
for the whole step. Uploading those weights took 1.9 s with the file in the
page cache and 9.5 to 12 s from a cold disk, through the simple path;
freeing them 4 to 8 ms. Launching a kernel costs the host under a
microsecond and nothing in the path synchronises.

**Few rows on the device layout** (`ares_mmr_few` in `kernels/mmr.cuh`, called by
`Kernels::product` for up to 16 rows of a matrix in the device layout). One
tile of 16 weight rows at a time on `mma` with f32 accumulation, the codes
decoded in registers and the next step's weights asked for one step ahead.
Its arithmetic is a function of the matrix and of the row's own activations
and nothing else: K is cut into 8 parts of whole super-blocks (fewer
non-empty ones for a matrix of fewer than 8), each part is summed from zero by
`mma`, and the parts are added in the order of K. A tile is computed by 1, 2,
4 or 8 warps, each taking an equal share of the parts (the others' parts reach
the first warp through shared memory), a block holds 4 or 8 warps, and which
of the four launches is the fastest depends on the shape and the card:
`Kernels::tune_layout` times them for each shape at load (the model's chooser
does it over its probe buffer, as large as the weights, because a small matrix
timed over a few dozen megabytes finds its address translations and the
launches that wait for them look better than they are). None of the launches
changes a bit: a row has the same bits alone, in a window of up to 16 rows, with
other rows beside it, in each launch, tuned or not
(`tests/layout.rs::a_row_is_the_same_bits_in_any_batch_and_launch`). 0.04 %
rms from the exact f32 product, as before (the f16 operands). Against the
integer decode kernels of the file's layout, one row, in the model's probes
(cold address translations, 59 shapes per card), the sum over the matrices of
a step is 98.7 % (RTX 4070) and 95.4 % (RTX 3060) of the integer kernels'
time, and single shapes run from 89 % to 111 %: a few i-quant and Q5_K shapes
of the RTX 3060 and the matrices of 256 to 768 rows are 2 to 11 % slower;
the walk of `examples/few_rows_bench.rs` (one shape, rows 1 to 16, `--plans`
for each launch) is the tool for a single shape. Cutting K the same way for
every launch costs nothing measurable: the same launch with a cut per
launch was 0 to 5 % slower at the median.

**Prefill weight kernels.** A quantized matrix times many rows of
activations, on the tensor cores, for eleven types (`kernels/mm.cuh`,
`mm_types.cuh`, `mmr.cuh`, `prefill.cu`) and, in a second module with the f16
paths only (`prefill_more.cu`, `mm_types_more.cuh`; below), fourteen more.
Three paths,
which `Kernels::prefill_path_for(ty, precision)` chooses per type and device:

- **int8**: `mma` on int8 tiles of the weight codes, reading the same int8
  activations the decode kernels use. The integer result of every 32 values
  is scaled in f32. 1.05 % rms from the exact f32 product (1.5 % at worst) on
  the test data.
- **f16**: the weights decoded to f16 tiles (packed f16x2 arithmetic from the
  codes and the f16 scales), f16 `mma` with f32 accumulation. The activations
  are divided per row by a power of two so that their largest value is below
  1, and the factor is put back at the end. 0.03 % rms.
- **f16 with f16 accumulation** (`PrefillPath::F16Flush`): the same tiles, but
  `mma` accumulates in f16 and the partial sum of every 4 `mma` (64 values of
  K: at most 64 products of a weight and a value below 1, so no overflow) is
  added to f32 registers. GeForce cards run that at twice the rate of f32
  accumulation. 0.05 % rms, 0.12 % at worst.

On a capped card (below) the f16 path runs at half the rate, and F16Flush is as
fast as int8 for some types and 20 times more exact: on the two development
cards at 1024 rows for Q4_K, IQ2_S and IQ2_XS (1.02 to 1.18 times), not for the
others (0.74 to 0.91 times). The engine takes the path measured per type on the
device at load (below), on every card, capped or not (`Precision::Fast`);
`Precision::Exact`, a tool of the tests, is always f16. A row gives the same bits
alone and inside a batch within a path.

On a card without the cap the f16 path costs the tensor cores no more than int8's
work, and whether the tile kernels run it as fast depends on the type; on a capped
card the same holds for F16Flush. So the path is measured per type, on every card,
when a model is loaded (`Kernels::measure_prefill_path`,
`kernels/paths.rs`): a probe matrix of the type with rows of 4096 values, as many rows
as make it larger than L2, 512 activation rows (a prompt step: four activation tiles of
the large shape in a group), every path timed in turn, and of the paths within 5 % of
the fastest the most exact kept for `prefill_path_for` (f16, then F16Flush, then
int8). 50 to 150 ms per type, once per device and process: kernels of a device loaded
again (Flash-Next's parts opened for the trial of the second card's role and then for
good, a model loaded again) take the first measurement and the same path. The qwen35
chooser measures every type of its groups before its probes (so they time the path that
will run) and the type of a head that keeps the file's layout beside them, Flash-Next's
wide steps its quantized dense types. A type that was not measured (in tests and
benches, and a type of Flash-Next whose probe found no memory) takes the rule kept from
the development cards, on every card: f16 for Q4_K, IQ2_S, IQ2_XS and the types without
an int8 kernel, summed in f32 where the tensor cores run that at full rate and in f16
(F16Flush) where they are capped, int8 else. (On the RTX 4060 Ti with gpu-unlock that
rule's int8 is within 5 % of f16 at 512 rows for every type of the table below and up
to 27 % faster, IQ3_S; f16 for every type, the rule for such a card before 2026-11-27,
was up to 21 % slower.) The load log names, per device, whether its tensor cores are
capped, the path each type takes with the probe's time of each path, and the kernels
the device layout's table took for each type (`ares-server`, `load/products.rs`). Measured on the two capped
development cards (2026-11-22, Qwen3.8-27B UD-Q4_K_S, tensor split 0.625 / 0.375,
three alternating pairs at a load average of 26 to 13): the RTX 4070 takes F16Flush for
IQ2_S, IQ2_XS, Q4_K and Q5_K and int8 for IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q3_K, Q6_K and
Q8_0, the RTX 3060 the same but int8 for Q5_K (F16Flush 1.022 ms against 0.919): only
Q5_K on the 4070 differs from the kept rule, towards the more exact path. A prompt of
2048 tokens reads at 1342 to 1358 tokens/s against 1346 to 1361 before (the same within
the noise), the decode step and the text are the same, and the load takes 1.0 s (4070)
and 1.1 s (3060) longer: eleven types measured on each device, at the same time. The probe's shape matters: with two activation tiles
the f16 path of Q5_K and Q6_K is 4 to 9 % behind int8 where from four on it is as fast,
and with rows of 14080 values (the first probe's, sized by its bytes) an f16 activation
tile holds 3.6 MB, a group two tiles against int8's four, and Q5_K's f16 path was 17 %
behind. On the test machine's RTX 4060 Ti with gpu-unlock (2026-11-22, 17408 x 5120,
TOPS, int8 / f16; the path the probe chose; the f16 path before the changes below):

| type | path | 64 rows | 256 | 512 | 1024 | 4096 | f16 before, 512 / 1024 |
|---|---|---|---|---|---|---|---|
| iq4_xs | int8 | 39.4 / 36.8 | 61.4 / 55.6 | 62.2 / 57.9 | 63.2 / 57.6 | 62.5 / 57.7 | 51.4 / 56.3 |
| q4_K | f16 | 37.5 / 38.1 | 58.5 / 58.1 | 58.5 / 60.6 | 59.7 / 61.0 | 59.0 / 61.3 | 53.6 / 60.4 |
| q5_K | f16 | 31.9 / 32.6 | 53.8 / 51.5 | 53.5 / 55.4 | 54.4 / 55.7 | 54.5 / 55.9 | 47.1 / 54.1 |
| q6_K | f16 | 22.9 / 24.5 | 42.8 / 41.4 | 43.4 / 43.4 | 43.9 / 43.6 | 44.3 / 44.1 | 34.8 / 37.0 |
| q3_K | f16 | 26.4 / 28.0 | 46.4 / 46.1 | 47.2 / 46.3 | 47.8 / 46.4 | 47.9 / 46.6 | 40.2 / 41.4 |
| iq3_s | int8 | 30.3 / 31.9 | 54.3 / 42.8 | 55.8 / 44.0 | 57.3 / 44.2 | 56.7 / 44.8 | 42.8 / 43.6 |
| iq4_nl | int8 | 38.3 / 35.4 | 56.4 / 53.8 | 60.8 / 56.3 | 62.8 / 56.7 | 62.1 / 57.5 | 48.4 / 53.5 |
| iq3_xxs | int8 | 34.5 / 33.2 | 54.5 / 48.1 | 56.0 / 49.6 | 57.5 / 48.9 | 57.4 / 49.0 | 47.2 / 47.9 |
| iq2_s | f16 | 31.8 / 34.7 | 48.3 / 53.0 | 50.0 / 53.6 | 50.4 / 53.3 | 49.7 / 53.3 | 52.1 / 52.0 |
| iq2_xs | f16 | 34.6 / 35.8 | 48.7 / 55.4 | 50.0 / 55.9 | 50.6 / 56.1 | 50.2 / 55.9 | 54.9 / 55.5 |
| q8_0 | int8 | 29.4 / 27.4 | 53.6 / 48.6 | 57.7 / 53.1 | 60.1 / 53.1 | 60.1 / 53.3 | 39.4 / 44.9 |
| q2_K | f16 | - / 38.7 | - / 59.6 | - / 59.5 | - / 59.3 | - / 59.2 | 58.8 / 59.0 |
| q4_0 | f16 | - / 36.0 | - / 54.5 | - / 57.6 | - / 57.9 | - / 59.0 | 48.9 / 54.1 |
| q4_1 | f16 | - / 39.4 | - / 54.5 | - / 55.7 | - / 56.0 | - / 56.3 | 51.7 / 55.5 |
| q5_0 | f16 | - / 34.3 | - / 50.0 | - / 51.9 | - / 52.4 | - / 52.4 | 43.2 / 47.6 |
| q5_1 | f16 | - / 37.5 | - / 49.4 | - / 50.7 | - / 51.0 | - / 51.2 | 47.4 / 50.6 |
| iq2_xxs | f16 | - / 38.2 | - / 50.7 | - / 54.5 | - / 54.5 | - / 54.7 | 52.2 / 53.1 |
| iq1_s | f16 | - / 27.0 | - / 59.6 | - / 60.4 | - / 60.3 | - / 60.3 | 59.3 / 59.7 |
| iq1_m | f16 | - / 30.5 | - / 61.0 | - / 61.5 | - / 61.4 | - / 61.4 | 60.8 / 61.1 |
| q2_0 | f16 | - / 40.7 | - / 58.6 | - / 59.6 | - / 59.4 | - / 59.5 | 58.5 / 58.8 |
| tq1_0 | f16 | - / 30.7 | - / 50.6 | - / 52.1 | - / 52.2 | - / 53.4 | 51.3 / 51.8 |
| tq2_0 | f16 | - / 38.7 | - / 59.2 | - / 59.9 | - / 60.1 | - / 59.9 | 59.1 / 59.7 |
| mxfp4 | f16 | - / 36.7 | - / 52.2 | - / 54.6 | - / 54.9 | - / 54.8 | 50.1 / 53.8 |
| nvfp4 | f16 | - / 38.6 | - / 54.0 | - / 55.8 | - / 56.2 | - / 56.6 | 50.5 / 55.5 |
| q1_0 | f16 | - / 43.2 | - / 61.8 | - / 62.5 | - / 62.3 | - / 62.4 | 61.5 / 62.2 |
| bf16 | f16 | - / 16.1 | - / 49.1 | - / 53.7 | - / 54.0 | - / 54.1 | 45.4 / 51.9 |
| f16 | f16 | - / 16.1 | - / 49.9 | - / 54.8 | - / 54.7 | - / 55.3 | 46.0 / 52.7 |
| f32 | f16 | - / 8.3 | - / 26.7 | - / 33.3 | - / 33.5 | - / 33.8 | 25.4 / 30.7 |

The types of the second module have no int8 path; between f16 and F16Flush the probe
took f16 for every one of them (F16Flush was at most 4 % faster, Q5_0).

What bounds the f16 path there, by ablation in a copy of the kernel (`ncu` needs
root on that machine; RTX 4060 Ti, Q8_0 / Q6_K / IQ4_XS / Q4_K / IQ3_S at 1024 rows,
TFLOPS of the copy: 45 / 36 / 50 / 58 / 42): a loop of `mma` alone reaches 92, the
skeleton of `mma`, `ldmatrix` and barriers 79. Loading the weights once instead of
every step (the decoding and the tile stores still every step) gives 63 / 65 / 65 /
66 / 60; leaving the decoding out with the loads in gives 47 / 42 / 54 / 61 / 45. So
the loads of the file's layout bound it, and not their bytes or latency: the same
load instructions all hitting L1 (every step reading the bytes of the first) are as
slow for Q6_K (36) and only partly faster for Q8_0 (55), while every lane of a warp
reading from the same row (one line per load instruction) gives 59 / 65. A load
instruction of a warp reads from 16 rows and costs the load unit 16 lines or more
whatever its width, and `ldmatrix`, the tile stores and the `cp.async` of the
activations go through the same unit; the f16 path moves twice the bytes the int8
path moves there (two-byte tiles and activations), which is why the types with many
load instructions per item (Q6_K 20, Q3_K 23, Q8_0 10, the codebook i-quants 6 to 9;
Q4_K and IQ4_XS 2 and 3) fall behind on it and the others do not. Reading the
2-byte aligned words as the 16-byte pieces that hold them (three loads for the 32
codes of a Q8_0 item instead of nine, `ares_ld_words_a2`) gained the large shape of
Q8_0 11 %, of Q6_K 14 %, Q3_K 10 %, IQ4_NL 4 % on the f16 path and 3 to 12 % on int8.
The medium and small shapes, bound by reading the weights from memory, and the expert
kernels' down products lost with the pieces (Q3_K at 64 rows 28 against 22 TFLOPS)
and keep the 4-byte loads; so do the one or two words of the i-quants (IQ2_S lost
14 %). What
did not help: an L2 prefetch of the next steps' weights (large shape), blocks of 8
warps with 64 x 32 or 32 x 64 warp tiles (fewer `ldmatrix` per `mma`, 5 to 8 %
slower: too few warps to hide the loads). The cure for the rest is the device layout
(below), which reads a tile of 16 rows with one load per lane: on the same card its
f16 kernels take 66 to 70 TFLOPS for Q4_K, Q6_K, Q8_0 and IQ4_XS at 256 to 1024
rows. The codebook types (IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS ...) have none yet.

The activation tiles of the tile kernels are grouped as those of the device layout's
kernels (a quarter of L2, a divisor of the tiles when one is near): a fixed 4 MiB
made groups of 3 on that card, so the 4 tiles of a prompt step of 512 rows ran as a
group of 3 and one alone (Q4_K f16 51 TFLOPS against 57 at 1024 rows); in groups of
4, 57.

On real prompts (2048 tokens in steps of 512 rows, the same card, alternating runs of the
build before and after, load 1 to 3): Qwen3.8-27B UD-Q4_K_S keeps 38 to 41 of its 497
matrices in the file's layout (the IQ3_S, IQ3_XXS, IQ2_S and IQ2_XS ones, which have no
device layout, 1.5 GiB): 2046 and 2048 ms before, 1998 and 1999 after (1000 against 1025
tokens/s; the weight products 425 against 412 ms), the IQ3 types now on int8. Flash-Next
UD-Q3_K_XL (`--ram-gib 32`) reads such a prompt in 13.7 s before and after (150 tokens/s):
its time is that of 69 GiB of expert uploads; of the device's time its dense products
(Q6_K and Q8_0 on the f16 path, which its wide steps take for all dense matrices) went from
111 to 96 ms.

Measured 2026-11-10, 17408 x 5120, TOPS (2 * n_out * n_in * rows per
second), RTX 4070 (peaks: int8 234, f16 with f16 accumulation 117, with f32
58.6) / RTX 3060 (102, 51.2, 25.6):

|              | 64 rows   | 256       | 1024      | 2048      |
|--------------|-----------|-----------|-----------|-----------|
| q4_K int8    | 45 / 24   | 76.5 / 33.7 | 78.1 / 34.3 | 78.0 / 34.5 |
| q4_K f16     | 46 / 22   | 54.8 / 23.0 | 55.2 / 23.7 | 55.3 / 23.7 |
| q4_K F16Flush | 46 / 22  | 75.4 / 33.5 | 77.9 / 34.1 | 77.7 / 34.1 |
| iq4_xs int8  | 45 / 21   | 81.8 / 36.2 | 81.9 / 37.1 | 81.6 / 36.9 |
| iq4_xs F16Flush | 44 / 22 | 69.8 / 30.2 | 69.7 / 30.8 | 69.5 / 30.8 |
| q6_K int8    | 28 / 13   | 54.2 / 23.8 | 55.7 / 24.8 | 55.9 / 24.8 |
| q6_K F16Flush | 30 / 14  | 43.3 / 19.4 | 44.2 / 20.1 | 44.6 / 20.1 |

(First-round values at 1024 rows: q4_K int8 62.4 / 26.4, f16 49.3 / 22.5;
iq4_xs int8 52.8 / 22.8; q6_K int8 33.4 / 14.7. The f16 path on the RTX
3060 was slower than in the first round after the large 128 x 128 shape
(19.5 against 22.5); it is 23.7 now.) Which type is slowest depends on the
path: q6_K, in both.

What bounds them, by ablation (the same kernel with a part removed; the
hardware counters are not available): int8 issues 416 instructions per warp
and step of which 192 scale the integer results, four warps per scheduler
1664 for 1024 cycles of tensor work, so it cannot get near the tensor cores
whatever the overlap; removing the scaling gives 111 TOPS. The f16 kernels
have no such cost, and an empty skeleton of `mma`, `ldmatrix` and barriers
reaches 103 of 117 TFLOPS with f16 accumulation (RTX 4070; 44 of 51.2 on the
RTX 3060). What held the first version of them back was the weights' loads:
with every weight row reading the same address the kernel was 35 % faster.
The file's layout puts the 36 to 70 bytes a row needs per step 2880 bytes from
the next row's, so one load of a warp touches 16 lines, and the load unit
(which `ldmatrix` shares) is busy with them. Decoding the next tile inside the
loop of `mma`, loading the quants into L1, and decoding with packed f16x2
instructions (Marlin's `prmt`/`hsub2`/`hfma2`) took Q4_K from 49 to 55 TFLOPS
with f32 accumulation and to 78 with f16 accumulation. Now the activations
bound them: without their `cp.async` the kernel reaches 98 TFLOPS (RTX 4070).

**Rearranged Q4_K** (`Kernels::rearrange`, `kernels/mmr.cuh`; today the device layout of twelve
types, multiplied through `Kernels::product`, see "Few rows on the device layout"). The first
measurements of it, with Q4_K alone: a Q4_K matrix can be rearranged once at upload into the
order the tensor cores read it (codes in `mma` fragment order, 16 bytes per
lane and 64 values; the 16-byte block headers of 16 rows together): the same
number of bytes (rows rounded up to 16), 0.3 ms for a 17408 x 5120 matrix on
the RTX 4070 (about 80 ms for the 13.8 GiB of the 27B model), and a kernel that
decodes in registers like Marlin's, with no weight tile in shared memory. It
is correct and as fast as the tile kernel from 256 rows on (78.7 / 33.9
TFLOPS at 256 to 2048 rows), faster at 128 rows on the RTX 4070 (76.7 against
60.1), and slower below (29 at 64 rows). So Marlin's layout and register
decoding did not beat the improved tile kernel at prompt sizes, because
both are bound by the activations now; the form is kept for 128 rows and as the
base for the next step. A rearranged matrix is read by nothing else: the decode
kernels and the other prefill kernels need the file's form, so a model that
decodes with the same matrix cannot keep only the rearranged one (until a decode
kernel reads it).

`Kernels::decode_kernel_rows(ty)` is the number of rows up to which the decode
kernels are the faster family (6 to 20, by type); above it `matmul` is used.
Re-measured with all paths: Q4_K 16, IQ4_XS 12, Q6_K 8 to 12 on the RTX 4070
(unchanged); the RTX 3060 keeps decode ahead up to about 16 for Q6_K.

**The second module** (`prefill_more.cu`, `mm_types_more.cuh`): iq2_xxs, iq1_s, iq1_m,
q2_0, q2_K, q4_0, q4_1, q5_0, q5_1, tq1_0, tq2_0, mxfp4, nvfp4 and q1_0. The same tile
kernels over the file's layout, with the f16 and the f16-accumulation paths and no
int8 one: for IQ2_S the f16 path with f16 accumulation is as fast as int8 (73.4
against 65.5 TOPS on the RTX 4070, 31.4 against 29.2 on the RTX 3060) and 20 times
more exact, so an 8-bit form of these types was not built. IQ2_XXS, IQ1_S and IQ1_M
hand the tile kernel their decoder as `decode_f16` (`AresMmDirect` in `mm.cuh`):
the item's 32 values come out as f16 pairs straight from the codebook entry, the sign
masks and the scale, with the decoders of the many-row expert kernels
(`expertrows_types.cuh`; IQ1_S shares the group of IQ1_M, `ares_er_iq1_values`),
where the others go through codes and scales as the first module does (`ares_mm_codes_to_f16`). Measured
2026-11-12 (`prefill_bench`, 17408 x 5120, 2048 rows, TOPS, RTX 4070 / RTX 3060; load
from other jobs on the machine does not touch a GPU kernel):

|          | f16 (f32 accumulation) | f16 with f16 accumulation | decode kernels, 8 rows |
|----------|------------------------|---------------------------|------------------------|
| iq2_xxs  | 52.4 / 23.0            | 71.7 / 31.1               | 14.1 / 6.7             |
| iq1_s    | 53.2 / 22.9            | 81.4 / 34.3               | 12.6 / 5.7             |
| (iq2_s)  | 52.0 / 22.5            | 73.4 / 31.4               |                        |
| (q4_K)   | 55.3 / 23.5            | 77.9 / 33.9               |                        |

The other twelve, f16 with f16 accumulation at 1024 rows (RTX 4070 / RTX 3060):
iq1_m 84.1 / 35.3, q2_0 81.9 / 34.9, q2_K 76.4 / 32.9, q4_0 73.7 / 32.2, q4_1 73.8 / 32.1,
q5_0 66.9 / 29.5, q5_1 66.3 / 29.1, tq1_0 65.1 / 27.9, tq2_0 79.1 / 33.7, mxfp4 75.2 / 32.5, nvfp4 74.6 / 32.4, q1_0 81.6 / 34.8.

IQ2_XXS and IQ1_S are in the range of the types that had kernels (5.1 times as fast as the
decode kernels in launches of eight rows for IQ2_XXS, 6.5 for IQ1_S), and no device
layout was tried for them: the rearranged form of Q4_K above was not faster than the
tile kernel at prompt sizes and these kernels are within 8 % of Q4_K's rate, so there is
no gap for a layout to close, and Vulkan has nothing to mirror (the layouts stay those
of the file). Their decode rows (`decode_rows` in `kernels/types.rs`: the decode kernels win up to 16 rows for IQ2_XXS, Q2_0 and Q4_0,
up to 12 for IQ1_S and IQ1_M; Q2_K and Q5_1 have none, since theirs win only up to about 10
rows, nor have the types not measured, so above eight rows those take the tile kernels) are
from the same bench at 8 to 32 rows.

The 2-bit file of Qwen3.8-27B (`Qwen3.8-27B-UD-IQ2_XXS.gguf`, 2048 tokens, one card,
`qwen35_run --prefill 2048`): the prompt went from 267 to 1154 tokens per second on the
RTX 4070 and from 128 to 536 on the RTX 3060 (the llama.cpp fork: 1011 to 1108 and 563);
the decode step at 1000 tokens of context did not change (21.03 and 21.06 ms, 36.53 and
36.51 ms), the top-1 agreement with llama.cpp on the German and English prompts is
96.4 / 97.8 % (RTX 4070) and 96.8 / 98.7 % (RTX 3060) after, 96.0 to 99.1 % before. With
only IQ2_XXS and IQ1_S the prompt ran at 1104 and 521 tokens per second; the other 4 %
and 3 % are the three attention blocks with q2_K matrices, which took the whole group of
matrices that read their input through the decode kernels. The Q4_K_S file on both
cards (0.625 / 0.375) took 1253 and 1278 tokens per second, 24.72 and 24.81 ms per decode
step, which is the noise of the layout probes. On the RTX 3060 the weights are 88 % of
the pass (profile of the first version per step of 512 rows: weights 860 ms, recurrent
layers 86, attention 34), at the 31 to 34 TOPS that every tile kernel reaches there (Q4_K
34), IQ2_XXS being about 43 % of that time and IQ1_S 28 % by their rates and numbers of
values; the llama.cpp fork's 563 tokens per second would need about 5 % less time than
the 536. On the RTX 4070 the weights take 1.62 of the 1.87 seconds.

**Float matrices** (f32, f16, bf16: `EngMmFloat` in `kernels/mm_types_more.cuh`, the second
module). They have the two f16 variants and no int8 one, since they have no integer codes,
and make the last three of the 28 types with a tile kernel. An item of 32 values is read as
16-byte words (the matrix must be 16-byte aligned, `n_in` a multiple of 32) and converted as
the tile is made: f16 is copied, bf16 is shifted to the top of an f32 and rounded by `cvt`,
f32 rounded by `cvt`; there is no scale. Before this, a dense model with such a matrix
multiplied it, and every matrix that read the same input, with the decode kernels in
launches of eight rows.

*Range.* An f16 holds 6.1e-5 to 65504 in full precision, down to 6e-8 with fewer bits. The
2-D float tensors that a product reads, in the models at hand (every value of every tensor
measured): the Q4_K_S file of Qwen3.8-27B has none, only vectors (norms, `ssm_a`,
`ssm_dt.bias`) and the convolution taps; its IQ4_XS file has 96 f32 matrices, `ssm_alpha`
and `ssm_beta` of 5120 x 48, whose values are f16 values (largest 0.06 to 0.33, converting
them to f16 changes nothing). Qwen3.8-Flash-Next has 436: 388 bf16
(`hc_{attn,ffn}_{up,down,inject}`, the indexer's `q_proj` and `k_proj`, `ssm_alpha`,
`ssm_beta`, `ple_key`, `ple_value`, `output_hc_*`) and 48 f32 (`ffn_gate_inp`, the router),
whose values are bf16 values too (their low 16 bits are zero). The largest value of any of
them is 8.56 (per matrix 0.16 to 8.56), 7600 times below the overflow. In a matrix 0.06 to
2.6 % of the values are below 6.1e-5, and 0 to 0.5 % (all but `ssm_alpha` below 0.005 %)
below 3e-8, which become zero. Converting straight to f16 changes the weights by 1e-9 to
1.9e-7 of their rms: bf16 values in the normal range are exact in f16 (8 bits of mantissa in
11), so only those small ones are not. A scale per matrix (a power of two that puts the
largest value at 0.5 to 1) was measured and lost: these matrices are already that size, and
it moved the error from 4e-9..1.9e-7 to 4e-9..3.7e-7 (it pushes the matrices whose largest
value is 4 to 8 below the range of normal numbers by a factor of 4 to 8). A scale per row of
weights would take the worst to 3e-8 (the rows of `ssm_alpha` differ in size by up to
3.5e5), at the price of a vector of exponents per matrix and a pass over the matrix to make
it; it is not worth that for an error this far below the activations' rounding. So the
kernels take no scale.

*Weights beyond the range.* A weight beyond 65504 is an infinity in the tile, so the outputs
that read it are infinite or NaN (tested for bf16 and f32), and the variant with f16
accumulation, which adds partial sums of 64 products of a weight and a value of at most 1,
gives an infinity from weights of about 1000 (tested with a run of 64 weights of 8000). A
`qwen35` model does not get there. `WeightRange::of` reads the largest magnitude of a float
matrix, and the loader asks it once for every float matrix a product reads (one more read of
its bytes; the result is `Mat::range`). A matrix whose largest weight is at most 512 takes any
path: 64 products of at most 512 stay below 2^15, half the range of f16 (tested at 512 with
activations just below 1). Above that it takes the f16 path, whose sums are f32, where another
float matrix takes f16 accumulation. Above 65504, or with an infinity or a NaN in the file, it
takes no tile kernel: the matrices that read the same input are multiplied by the decode
kernels in launches of eight rows, which compute with the weights as stored (and, for a
quantized matrix among them, with int8 activations, also where `Precision::Exact` was asked
for). The test `float_weights_beyond_f16_are_multiplied_by_the_kernels_that_hold_them` runs a
made-up model with a row of weights of 61440 and two weights of 7e4: its logits are within
7.1e-3 of the reference, as without such weights (6.0e-3); with the rule taken out the logits
of a prompt of nine rows are not numbers and the step after it fails.
The layout probes time such a matrix by its type, on the tile kernels. `qwen4exp` reads the
range of its float matrices at load as well (`Mat::range`, read once per matrix by
`blocks/arena.rs` for both models) and its wide steps ask it before every float product; its quantized matrices go through
`Kernels::product`, which knows no range and needs none.

*Small weights.* What a matrix loses below 6.1e-5 is absolute, at most 2^-25 per weight. The
test `tiny_float_weights_are_off_by_at_most_the_spacing_of_f16_subnormals` derives the bound of
an output from the roundings of each step and holds both paths to it on weights from 1e-7 to
1e-4 (512 values, 2026-11-12, both cards): the f16 path is off by at most 3.4e-6 to 5.0e-6
against a bound of 1.7e-5. The path with f16 accumulation loses more there, and not through
the weights. The tensor cores round its f16 sum after every 8 products (measured: doing
exactly that on the CPU gives the kernel's bits for 99.7 to 99.8 % of the outputs on both
cards, rounding after every 16 for 14 %), and the sum is in the units of the activation row
divided by its factor, the power of two above the row's largest magnitude. A partial sum
below 1.2e-4 is therefore rounded to a multiple of 2^-24 of that factor, so an output is off
by up to `n_in / 8` times that whatever its own size: 3.3e-5 to 3.5e-5 in the test (bound
2.6e-4), ten times the f16 path. That is felt by an output whose row of weights is small next
to the activations: on Flash-Next's `ssm_alpha` (bf16, 48 x 2560, whose rows differ in size by
up to 3.5e5) the worst output with f16 accumulation was 3.3 % of the sum of the magnitudes of
its products away from the product of the rounded operands (8 rows, RTX 4070; 2e-6 on the f16
path), while the error over all outputs was 0.045 % rms. The tests take that absolute term off
before they compare the path with its emulation.

*Error against the exact f32 product* (`tests/prefill.rs`, weights of rms 0.05 with
outliers, activations of rms 1 with outliers, shapes from 1 x 32 to 200 x 1024): the f16
path 0.02 to 0.03 % rms for f16 and bf16 weights (the rounding of the activations; the
weights are exact), 0.02 to 0.04 % for f32 weights; the path with f16 accumulation 0.04 to
0.06 %. Against the same product with the operands rounded to f16 and summed exactly, the
f16 path is within 3e-6 of the magnitude of the sum and the flush path within 1e-3.
On the real float tensors (`prefill_matmul_on_real_tensors`, 2026-11-12, both cards, 8, 9 and
70 activation rows of a normal distribution with outliers, the first 200 rows of the first
matrix of each kind): the 14 kinds of bf16 matrix of Flash-Next 0.014 to 0.024 % rms on the
f16 path and 0.042 to 0.061 % with f16 accumulation, its f32 `ffn_gate_inp` 0.020 to 0.022 %
and 0.041 to 0.053 %, and the f32 `ssm_alpha` and `ssm_beta` of the IQ4_XS file of
Qwen3.8-27B 0.017 to 0.023 % and 0.046 to 0.054 %. The decode kernels, which take eight rows
of them, show no error in the four digits printed. Both cards give the same numbers.

*Speed* (`examples/prefill_bench`, a 17408 x 5120 matrix, ms, RTX 4070 / RTX 3060, machine
load 29 to 38 from other builds; the last column is the decode kernels in launches of eight
rows, which is what the model did before):

| rows | f16 path | f16 accumulation | eight-row launches |
|---:|---|---|---|
| 16 | 0.40 / 0.63 | 0.40 / 0.63 | 0.78 / 1.41 |
| 64 | 0.43 / 0.65 | 0.43 / 0.63 | 3.13 / 5.64 |
| 256 | 0.94 / 2.19 | 0.73 / 1.57 | 12.4 / 22.5 |
| 1024 | 3.71 / 8.52 | 2.80 / 6.32 | 49.8 / 90.5 |
| 4096 | 14.7 / 34.2 | 11.0 / 25.3 | 199 / 362 |

(bf16; f16 is within 3 % of it everywhere.) f32: 0.83 / 1.18 at 16 rows, 1.41 / 2.98 at
256, 5.29 / 11.9 at 1024 and 20.6 / 47.8 at 4096 with the f16 path; 5.73 / 12.4 and 22.3 /
50.2 with f16 accumulation, which is not faster for f32 (so the tensor cores are not what
bounds it; the large shape of f32 uses all 128 registers a thread may have, with 64 of them
holding the raw bytes of the next two items); eight-row launches 1.53 / 2.29 at 16 rows, 24.3 /
36.6 at 256, 97 / 147 at 1024 and 389 / 589 at 4096. In TOPS at 4096 rows: bf16 and f16 50 / 21
(f16 path) and 66 / 29 (f16 accumulation), f32 35 / 15 and 33 / 15, of 58.6 / 25.6 that the
capped tensor cores run. Up to 8 rows the decode kernels are as fast or faster (0.39 / 0.72 ms
for bf16 at 8 rows against 0.42 / 0.76; 0.056 against 0.107 ms for 320 x 10240) and compute
the exact product; from 9 rows on the tile
kernels are faster for all three types (0.40 against 0.77 for two launches), so the float
types have no decode rows. Flash-Next's mixers (bf16, 512 rows):
`hc_{attn,ffn}_down` 320 x 10240 3.10 / 4.02 ms in launches of eight rows, 0.31 / 0.44 ms on
the tile kernels' f16 path and 0.21 / 0.30 with f16 accumulation; `hc_{attn,ffn}_up` 10240 x
320 1.21 / 2.55 ms, 0.097 / 0.22 and 0.081 / 0.18 ms.

*In a model* (`qwen35_run`, Qwen3.8-27B on both cards at 0.625 / 0.375, a prompt of 2048
tokens in steps of 512 rows, two passes, then 96 decode steps at 1000 tokens of context;
2026-11-12; before is the commit before these kernels, after is with them and with the ten
other pieces integrated at the same time; each pair back to back, two rounds, with the
machine's load average in brackets):

| file | | prompt, tokens per second | decode step, ms (median) |
|---|---|---|---|
| Q4_K_S, no float matrix | before | 1366, 1363 [2.7]; 1354, 1354 [13] | 24.76; 24.80 |
| | after | 1347, 1345 [2.7]; 1342, 1352 [12] | 24.75; 24.76 |
| IQ4_XS, 96 f32 matrices | before | 871, 866 [2.6 to 7.6]; 874, 878 [21 to 16] | 28.89; 25.47 |
| | after | 1283, 1295 [7.6 to 15]; 1306, 1310 [16 to 12] | 26.59; 25.50 |

The Q4_K_S file is unchanged within the 1.4 % its runs differ by. The IQ4_XS file reads a
prompt 1.5 times as fast: its `ssm_alpha` and `ssm_beta` (f32, 5120 x 48) had taken the
matrices that read the same input with them through launches of eight rows in every recurrent
block. Its decode step is the same (the fastest step of the first round was 25.43 before and
25.31 after; the medians of that round were taken while the load rose).

*The rows of a step.* A file of float matrices reads a prompt in the steps of 512 rows that
every file takes; wide steps are what the tile kernels are for. Measured on one card (RTX
4070, 2026-11-12, a prompt of 2048 tokens, two passes, each run twice, steps of 128 against
512 rows by `--max-rows`): a copy of Qwen3.5-0.8B with all 187 matrices in bf16 (1434 MiB, in
the file's layout) reads 12342, 13079, 13677 and 13688 tokens per second in steps of 128 rows
and 27625, 27548, 27597 and 27536 in steps of 512 (load 8). The first eight blocks of
Qwen3.8-27B with their 62 matrices in bf16 (made from the Q4_K_S file; 5806 MiB in the file's
layout, 17408 x 5120 the largest) read 5596, 5699, 5700 and 5695 tokens per second in steps
of 128 rows and 7810, 8068, 8082 and 8062 in steps of 512 (load 10).

A model calls them through `Kernels::product` (matrix in `WeightLayout::File`, with the
activations in the form `Kernels::activation_form` names and `prepare_activations` makes for
the same arguments): up to 8 rows with `Precision::Fast` it is the decode kernel, above that
the tile kernel of `prefill_path_for`, and with `Precision::Exact` always the f16 path.

A pass over all weight matrices of Qwen3.8-27B (Q4_K_S file, 13.8 GiB) at
the split 0.625 / 0.375 takes 92 ms with 64 rows, 221 ms with 256, 694 ms
with 1024 and 1339 ms with 2048: 1529 rows per second at 2048 rows (first
round: 1054), at a split of 0.7 / 0.3 1787 rows per second at 1024 rows (the
2048-row pass no longer fits the RTX 4070's memory since the prepared
activations carry an f32 copy). The llama.cpp fork prefills about 1180 tokens
per second at 2048 tokens for its whole forward pass, so the weights alone are
now about 1.3 to 1.5 times as fast as the fork's whole pass; attention and the
rest still come on top. The f32 copy is of use only to the decode kernels of
the float types and is to leave the buffer the prefill kernels size.

**Many-row expert kernels** (`kernels/expertrows*.cuh`,
`ares_cuda::ExpertRowKernels`). The experts of a MoE layer for any number of rows,
on the tensor cores, for every weight type (see "Experts in every weight type" below; the
kernels were written for the types of Qwen3.8-Flash-Next, gate and up IQ2_S, IQ2_XXS and
IQ1_M and down Q2_0, and what is measured in this section is measured on those). The decode
kernels above take eight rows; these take
as many as there is memory for, and are meant for prompts and windows of drafts.
Three stages, all on the device, none needing the host to read the picks:

- **Plan** (`plan_rows`). From a route record of any number of rows (the ids and
  weights the router writes, `n_rows * top_k` of each) a stable counting sort by
  expert: the pairs (row, pick) of an expert follow each other in the order of
  their numbers, and each expert's run is cut into items of at most `tile` pairs
  (8, 16, 32 or 64). One launch for up to 1024 pairs, three above (a block per
  1024 pairs counts and ranks with `__match_any_sync`, one block turns the counts
  into positions and items, a block per 1024 pairs scatters). The plan lists the
  first position of each expert, the items `(expert, first, count)`, the row and
  router weight at each position and the position of each pair; a pair whose id
  is out of range is dropped. The kernels are launched for the most items a plan
  can have and the surplus blocks return at once, so nothing waits for the count.
- **Gate and up** (`rows_gate_up`). A block takes an item and 64 of the expert's
  hidden values, a warp 16: 16 rows of gate and the same 16 of up against all the
  pairs of the item, so the weights are read once for all of them, from the
  address a table of one `u64` per expert gives (0: not present). The weights are
  the A operand of `mma.m16n8k16` and are decoded in registers from the file's
  bytes into f16 (the codebook magnitudes come out of the bytes as `1024 + m`
  with the trick of Marlin and one fused multiply-add with the scale; the signs
  are XORed into the sign bits from a table of masks in shared memory), the
  activations the B operand, gathered by `cp.async` from the f16 rows of
  `Kernels::prepare_prefill_activations`. A thread of a quad takes one sub-block of
  32 values of every 128, which is what its two `mma` of a step need, since the
  order of K inside an `mma` is free if both operands use the same. The epilogue
  makes `silu(gate) * up` in f32, divides it by a power of two per pair and 64
  values (so nothing overflows an f16) and stores f16.
- **Down** (`rows_down`). The same with the type of down, for Q2_0: a word of 16 codes gives the pairs
  (value j, value j + 8) of the operand with one shift and mask, so the hidden
  values are stored in that order. The product of a block of 64 is multiplied by
  the pair's scale as it is added to the accumulators, and the weighted output is added
  to the row with `red.add`: the order in which a row's picks are added depends on the
  schedule, so a row's sum can differ in its last bits from run to run and between a
  row alone and in a batch; its hidden values do not (tested). An ordered sum (a
  scratch of `top_k * n_embd` f32 per row and a kernel adding a row's picks in the order
  of the router) gave the same bits everywhere; nothing used it, and since the owner's
  rule of 2026-11-23 (greedy output may differ from run to run, as long as it is not
  wrong) the host has it no more (2026-11-24). The down kernel keeps its branch: without
  it nvcc lays out the kernel of 16 pairs of Q2_0 in 64 registers instead of 70, and it
  was 5 % slower.

An expert of the plan without an address is not read: its pairs add nothing, are
counted in the plan (`missing_pairs`) and the expert is flagged, so the caller
knows what to fetch. One plan serves several calls with different tables, which is
how the waves of a layer run.

These kernels do not wait for an expert on the device and have no host share: they skip
an expert without an address and report it. The decode kernels therefore stay for the
steps of a generating sequence, which depend on both, and for windows of up to eight rows
(the decode kernels' own limit). The slices of a prompt of Qwen3.8-Flash-Next run on the
new kernels (see "Wide steps" under "Qwen3.8-Flash-Next through the runtime"); the host
gives each round its address table and brings what a table lacked.

Against the exact result (f64 on the dequantized weights and the f32 activations) the
output is 0.05 to 0.07 % rms off, and the hidden values the same, on generated blocks of
the four types the kernels were written for and on the real tensors of one layer of each
type (the same figures on both cards); over all 28 types, as gate and up and as down, 0.04
to 0.10 % (the test allows 0.2 %). Rows scaled by 1e-4 to 1e5 within one call stay within 0.2 %. The decode
kernels, which quantize the activations and the hidden values to int8, are at about 1 % in
the matrices of their family (their expert test allows 5 %).

**f16 or int8 activations.** f16, with f32 accumulation. The reasons, measured: the dense
kernels of the same cards and the same type (`prefill_bench`, IQ2_S, 17408 x 5120, 1024
rows) reach 66.5 TOPS with int8 and 52.0 TFLOPS with f16 and f32 accumulation on the RTX
4070 (29.5 and 22.6 on the RTX 3060), and 73.4 (31.5) with f16 accumulation, so int8 is not
the fast path for this type. For the experts it would also need the scale of every 16 values
(IQ2_S) or 32 (IQ2_XXS) applied to the integer sum of each `mma`: either the four threads of
a quad share one sub-block and each loads the same words (four times the load instructions
of the layout above), or the scale is applied to every result at about three instructions
per output. Accumulating in f16 and flushing to f32 every 64 values was built and measured
on the experts: gate and up at 4096 rows 14 % faster (8.1 against 7.0 ms on the RTX 4070,
synthetic weights), the output 8e-4 off from the f32 accumulation; it was not kept, since
the kernels are not bound by the tensor cores (a loop of `mma` alone gets 58.6 of 58.6
TFLOPS here; the kernels get 33 useful, 40 counting the padding of the last item of an
expert).

Measured 2026-11-11 on the recorded routes of `qwen4exp` (`examples/expert_rows_bench.rs`:
`long`, `mixed`, `german` and `english` of `/var/tmp/ares-qwen4exp-trace`, token after token),
the real experts of one layer, all that the rows pick in device memory, random activations.
Microseconds for one layer's block (IQ2_S gate and up, Q2_0 down), "block" being plan, gate
and up, down and, for the ordered sum (removed since), the combine, run one after the other; the tile is
the best of 8, 16, 32 and 64. "Decode kernels" are `ExpertKernels::experts` in launches of
eight rows (the router and the host-built record are not in their time). GB/s are the weights
read once (every expert the rows pick) over the block time, TFLOPS are the useful products
of the three matrices over it; the cards read 473 and 337 GB/s in a loop of coalesced
reads and compute 58.6 and 25.7 TFLOPS in a loop of `mma` (f16 with f32 accumulation).

RTX 4070, IQ2_S gate and up (blk.0):

| rows | tile | plan | gate and up | down | combine | block, atomic | block, ordered | rows/s | GB/s | TFLOPS | decode kernels | times faster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 7.4 | 62 | 19 | 10 | 83 | 94 | 12106 | 183 | 1.2 | 62 | 0.76 |
| 2 | 8 | 6.7 | 105 | 32 | 9 | 144 | 153 | 13896 | 199 | 1.4 | 101 | 0.70 |
| 4 | 8 | 6.8 | 178 | 54 | 10 | 242 | 252 | 16512 | 231 | 1.6 | 213 | 0.88 |
| 8 | 8 | 7.0 | 288 | 80 | 10 | 390 | 403 | 20506 | 244 | 2.0 | 361 | 0.93 |
| 16 | 8 | 7.3 | 503 | 175 | 10 | 690 | 713 | 23182 | 241 | 2.3 | 737 | 1.07 |
| 64 | 16 | 9.4 | 1101 | 409 | 10 | 1537 | 1651 | 41634 | 208 | 4.1 | 2901 | 1.89 |
| 256 | 16 | 16.2 | 1904 | 772 | 22 | 2606 | 2774 | 98243 | 191 | 9.7 | 11814 | 4.53 |
| 1024 | 32 | 18.4 | 3571 | 1583 | 280 | 4849 | 5368 | 211178 | 145 | 20.8 | 49310 | 10.17 |
| 4096 | 32 | 26.3 | 8153 | 6587 | 1101 | 13772 | 15663 | 297426 | 53 | 29.2 | 196319 | 14.26 |

RTX 3060, the same:

| rows | tile | plan | gate and up | down | combine | block, atomic | block, ordered | rows/s | GB/s | TFLOPS | decode kernels | times faster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 10.0 | 126 | 40 | 12 | 172 | 185 | 5827 | 88 | 0.6 | 144 | 0.84 |
| 2 | 8 | 9.3 | 214 | 65 | 10 | 292 | 307 | 6845 | 98 | 0.7 | 231 | 0.79 |
| 4 | 8 | 9.5 | 407 | 121 | 11 | 542 | 560 | 7377 | 103 | 0.7 | 438 | 0.81 |
| 8 | 8 | 9.7 | 676 | 216 | 12 | 897 | 921 | 8920 | 106 | 0.9 | 759 | 0.85 |
| 16 | 8 | 10.2 | 1190 | 384 | 12 | 1566 | 1608 | 10219 | 106 | 1.0 | 1540 | 0.98 |
| 64 | 16 | 13.9 | 2282 | 881 | 30 | 3074 | 3212 | 20822 | 104 | 2.0 | 6014 | 1.96 |
| 256 | 16 | 21.7 | 3943 | 1849 | 101 | 5706 | 5916 | 44867 | 87 | 4.4 | 24704 | 4.33 |
| 1024 | 32 | 23.9 | 7847 | 3828 | 395 | 11538 | 12089 | 88750 | 61 | 8.7 | 103300 | 8.95 |
| 4096 | 64 | 44.4 | 18968 | 11911 | 1523 | 30940 | 32339 | 132386 | 24 | 13.0 | 409562 | 13.24 |

The other two types are faster in proportion to their bytes. Block time with the best tile and
the atomic sum, RTX 4070 / RTX 3060, in microseconds: IQ2_XXS (blk.1) 73 / 132 at 1 row, 350 / 656
at 8, 1855 / 4243 at 256, 12998 / 28900 at 4096; IQ1_M (blk.8) 66 / 114, 194 / 404, 1398 / 3551,
12934 / 28763. The decode kernels at 4096 rows take 183 and 148 ms (RTX 4070), 383 and 296 ms
(RTX 3060) for the two.

- **From 16 rows on they are as fast or faster than the decode kernels**: 1.9 times as fast at
  64 rows, 4.3 to 4.5 at 256, 9 to 10 at 1024, 13 to 14 at 4096 (IQ2_S; the others more). Below
  16 rows the decode kernels still win for IQ2_S (0.7 to 0.93 times as fast on the RTX 4070, 0.8
  to 0.85 on the RTX 3060 up to 8 rows) and for IQ2_XXS at 1 and 2 rows on the RTX 4070; IQ1_M is
  ahead from 2 rows, and from 1 row on the RTX 3060. So the family cannot be chosen by a constant:
  a probe that timed both on the device for the shape (about 100 ms; no production code called
  it, and it was removed on 2026-11-24) found that the decode kernels are faster up to 8 rows
  for IQ2_S on both cards, 5 (RTX 4070) and 1 (RTX 3060) for IQ2_XXS and none for IQ1_M. A verify
  window of 2 to 16 rows is therefore not faster than today's kernels for the 34 IQ2_S layers; for
  the 14 others it is up to 1.4 (IQ2_XXS) to 2 (IQ1_M) times as fast at 16 rows.
- **What bounds them.** Below 1024 rows the weights' loads: gate and up read 190 to 245 GB/s of
  the 473 the card can (RTX 3060 90 to 106 of 337), because each lane of a warp reads 2 and 4
  bytes of 41 bytes of a row per step, eight rows at a time, a pattern that streams at 189 GB/s
  (149 on the RTX 3060) in a loop of those loads alone, where a warp reading rows in order reads
  473 (337). Above that the product: 33 TFLOPS of useful products in gate and up (RTX 4070, 4096
  rows), 20 in down, of the 58.6 of the tensor cores. Branching around unused columns, deeper
  pipelines, activations resident in shared memory, L2 prefetch hints, and accumulating in f16
  (14 % faster for 8e-4 more error) were tried and are in the comments of the kernels with
  their numbers; none closed the gap. The layout of the loads is the open lead.
- **Atomic or ordered.** `red.add` into the output was 3 to 15 % faster than the scratch and the
  combine of the ordered sum (more above 1024 rows, where the scratch is 100 KB a row), and needs
  no scratch; the ordered sum, which gave a row the same bits alone, in a batch and in any place
  of a batch, is gone (above).
- **The plan** costs 7 us for one row, 9 us at 64, 16 at 256, 26 at 4096 (RTX 4070; 10, 14, 22, 44
  on the RTX 3060): a few percent of a block at 256 rows and above.
- **What a prompt pass could reach.** Experts only, all resident, 48 layers (34 of IQ2_S, 11 of
  IQ2_XXS, 3 of IQ1_M) at the best tile: RTX 4070 650 ms for 4096 rows (6300 rows per second),
  220 ms for 1024 rows (4640); RTX 3060 1456 ms (2810) and 536 ms (1910); the decode kernels in
  batches of eight take 9.1 s and 19 s for 4096 rows, 449 and 215 rows per second. Two cards on a
  layer split can add up to about 9100 rows per second for the experts. The rest of a layer is
  not in these numbers. The dense parts of the model are 3.4 GiB, which by their bytes is
  roughly 7 to 13 GFLOP of work per row against the 4.7 of the experts, so a whole pass would take
  about twice as long at the dense kernels' rates (an estimate, not a measurement). The experts that
  are not in memory, which the pass has to stream, come on top.
- **Memory a call needs**, per row for Qwen3.8-Flash-Next (ten picks): 5.2 KB of f16
  activations, 13.2 KB of hidden values, 0.12 KB of plan and 10 KB of output; the plan is
  0.6 MB for 4096 rows.

**Experts in every weight type** (`kernels/experts.cu`, `kernels/expertrows*.cu`,
`ares_runtime::expert_store::cpu`). The three matrices of an expert may be stored in any of
the 28 types a model file holds weights in. Gate and up of a layer share one type (one kernel
computes both rows against the same activations; the loader refuses a layer in which they
differ), down has its own, and the types may differ from layer to layer. A row must be whole
blocks of its type and whole groups of 32 values (`ExpertShape::check`), so a type with blocks
of 256 values cannot hold a down matrix whose rows are the 640 hidden values of Flash-Next;
the many-row kernels also take the width in steps of 128 and the inner width in steps of 64
(`ExpertShape::check_rows`, asked when a model opens for wide steps). The router
(`ffn_gate_inp`) must be F32.

- *Decode kernels.* `ares_expert_gate_up_<type>` and `ares_expert_down_<type>` are the
  templates of `expert_ffn.cuh` over the decode traits of the weight kernels, instantiated
  for all 25 quantized types. The float types have a trait of the same interface
  (`EngExpertFloat`): the weights as stored times the int8 activations, so that an expert of
  any type computes `W x^` for the rounded activations `x^` and the host can compute the
  same.
- *Many-row kernels.* IQ2_S, IQ2_XXS, IQ1_M and IQ1_S as gate and up and Q2_0 as down have a
  decoder written for these kernels. Every other type and role goes through an adapter
  (`expertrows_mm.cuh`) that makes a gate and up or a down decoder of the item decoder of the
  dense tile kernels (32 values of a row as f16 pairs), so the engine has one tensor-core
  decoder per type. An adapted type decodes an item whole when a step begins and holds its
  16 words per row stream across the step, where a written decoder makes each group as it is
  used; the down kernel, whose threads take 16 values each, decodes an item for half of it.
  The kernels are in five modules (`expertrows.cu`, `expertrows_{k,i,l,f}.cu`) that compile
  side by side, and whether the sum is atomic or ordered is an argument of the down kernel,
  not a kernel of its own.
- *On the CPU* (`cpu/kernels.rs`). The types of the Flash-Next files have kernels that
  multiply the file's bytes with exact integer sums per 32 values, as the device does: as
  gate and up IQ2_S, IQ2_XXS, IQ1_M, IQ3_XXS (UD-Q3_K_XL), IQ4_XS (UD-Q3_K_XL), Q4_K and
  Q5_K (UD-Q4_K_XL) and Q2_0 (the MTP block); as down Q2_0 and the types of blocks of 32
  values Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and IQ4_NL (`Block32`). IQ3_XXS looks its codebook up
  by a gather or by eight loads (each worker measures which is faster on its core, as for
  the IQ2 types), IQ4_XS and IQ4_NL their 16 levels by a shuffle; Q4_K and Q5_K multiply
  their unsigned codes and take the minimum from the sum of the activations, which the
  activations carry per 32 values; Q2_0 as gate and up runs the down kernel of Q2_0 on the
  activations laid out in the order of its bit planes once per job (`Planes`). The down
  kernels of `Block32` multiply hidden codes stored in the order of their values (Q2_0's in
  the order of its planes, `Order`), with `maddubs` or AVX-VNNI's `dpbusd`, as signed
  codes (Q4_0 and Q5_0 less their offset) or with the minimum added from the hidden codes'
  sums (Q4_1, Q5_1). Every other type and role is decoded row by row
  with the reference decoder (`ares_quant::dequantize`) and multiplied in f32 with the same
  rounded activations, once for all the rows of the step that use the expert; the decoding
  is what such an expert costs (table below). The written kernels need x86-64-v3 (AVX2,
  FMA, F16C, BMI2); on a CPU without it every type goes through the decoded path in
  portable code (`Features::vector`), which before refused the CPU half altogether.
- *A job* (`LayerJob` in `cpu/mod.rs`) is a layer's experts for one step, cut into tasks
  that the workers claim in order: the gate and up products of an expert in pieces of 32
  hidden values, and its down product in pieces of 64 output values, which wait for the
  expert's hidden values. An expert's down pieces come after the gate and up pieces of the
  next expert, so the down products, which read their bytes as fast as the memory gives
  them, run beside the gate and up pieces, which the cores bound (decoding codebooks); the
  parts of a chunk of the output are added in the order of the experts, so the sums do not
  depend on which worker ran what. Before 2026-11-22 every down piece waited for the gate
  and up pieces of all the job's experts.
- *The cost model* (`CostModel` in `expert_store/hybrid.rs`) times CPU jobs per pair of
  types (gate and up, down): each pair a model has is calibrated by itself when the store
  gets its CPU half, and a job's time is recorded under its layer's pair. Before, every gate
  type but IQ2_S and IQ2_XXS shared one entry and the type of down was not looked at, which
  would have timed a decoded type as IQ1_M. A pair whose jobs take longer than an upload is
  uploaded, by the measured numbers: the split takes whichever of the CPU's job and the last
  upload ends first.

*Tested* (2026-11-12, both cards). Decode kernels, every type as gate and up and as down, on
generated blocks at rows of 512 and 128 values and at the real widths (2560 and 640, or 768
for a down type with blocks of 256): the hidden values are within 0.50 of their own rounding
step of `silu(gate x^) * (up x^)` on the reference decoder's weights (the test allows 0.51),
the output within 1.5e-7 of the exact sum over the device's own hidden values, the layer
within 1.0 to 3.4 % of the exact arithmetic on unrounded activations (8-bit activations and
hidden values, as for the four types before). Many-row kernels, the same types and widths,
every tile and both sums: 0.04 to 0.10 % rms against the exact result. The blocks of real
tensors as experts (`ARES_TEST_GGUF=<any model file>`, the largest tensor of each type the
file has, at the real widths), on the Q4_K_S and the IQ2_XXS file of Qwen3.8-27B, a
Q6_K file with BF16 and F16 tensors and Flash-Next's own, 19 types between them (all but
Q4_0, Q4_1, Q5_0, Q5_1, which the made-up models hold as encoded values, and TQ1_0, TQ2_0,
MXFP4, NVFP4 and Q1_0, of which there is no file here: those five are tested on generated
blocks only): the decode kernels within 0.50 steps and 2e-7, the many-row kernels within
0.06 to 0.13 %. The CPU against the device, layer by layer in
made-up models of 28 pairs of types with half of the experts missing: 2.2e-7 to 4.6e-7 where
no hidden code falls on a rounding boundary and up to 2.3e-3 where one does (the decoded
path sums in another order than the device's integer sums; the test allows 1e-2). Whole
made-up models of 28 pairs of types against the model's CPU reference, layer by layer and row
by row: the experts of a row within 1.4e-2 to 2.8e-2 of the reference's in
steps of up to eight rows and 5.9e-3 to 1.3e-2 in wide steps, in the 764 to 768 of a model's
768 rows of layers that made the reference's choices (on one card and on two); four pairs through the runtime in every
placement (`models_of_other_expert_types_answer_as_the_reference_in_every_placement`).

*Speed of the kernels by type* (`examples/expert_types_bench`, 2026-11-12, load average 6 to
9 from other work; microseconds, RTX 4070 / RTX 3060; an expert of 2560 x 640, or 768 hidden
values for a down type with blocks of 256, 64 experts of random bytes, eight picks a row and
no two rows of a step of eight the same ones; the type is that of gate and up with Q2_0 as
down, or that of down with IQ2_S as gate and up; the decode kernels at 1 and 8 rows, the
many-row kernels with their plan and the atomic sum at 8, 64 and 512 rows):

| type | bits | gate and up: decode kernels, 1 row | 8 rows | many-row kernels, 8 rows | 64 | 512 | down: decode kernels, 1 row | 8 rows | many-row kernels, 8 rows | 64 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q4_K | 4.50 | 56 / 113 | 443 / 811 | 575 / 761 | 530 / 656 | 1021 / 2447 | 44 / 88 | 339 / 650 | 663 / 1140 | 650 / 1194 | 1218 / 2817 |
| q5_K | 5.50 | 47 / 116 | 497 / 846 | 591 / 780 | 635 / 783 | 1115 / 2462 | 46 / 94 | 371 / 701 | 691 / 1220 | 673 / 1296 | 1284 / 2936 |
| q6_K | 6.56 | 58 / 126 | 559 / 929 | 916 / 972 | 936 / 1139 | 1400 / 3433 | 56 / 107 | 416 / 814 | 784 / 1509 | 776 / 1550 | 1477 / 3414 |
| q3_K | 3.44 | 51 / 113 | 410 / 806 | 469 / 787 | 455 / 830 | 1151 / 3386 | 52 / 110 | 376 / 806 | 724 / 1480 | 735 / 1524 | 1440 / 3248 |
| q2_K | 2.62 | 49 / 109 | 382 / 796 | 377 / 558 | 331 / 594 | 1013 / 2377 | 54 / 118 | 398 / 865 | 593 / 1169 | 594 / 1184 | 1279 / 2864 |
| iq4_xs | 4.25 | 45 / 109 | 431 / 785 | 540 / 751 | 520 / 686 | 1013 / 2386 | 44 / 91 | 336 / 666 | 584 / 1233 | 590 / 1249 | 1265 / 2911 |
| iq4_nl | 4.50 | 46 / 106 | 444 / 764 | 565 / 705 | 562 / 659 | 1027 / 2348 | 42 / 91 | 327 / 667 | 584 / 1060 | 618 / 1071 | 1164 / 2647 |
| iq3_s | 3.44 | 50 / 114 | 405 / 838 | 439 / 1013 | 370 / 768 | 1096 / 2623 | 46 / 98 | 327 / 705 | 607 / 1349 | 653 / 1593 | 1408 / 3206 |
| iq3_xxs | 3.06 | 48 / 106 | 376 / 754 | 395 / 725 | 371 / 732 | 1027 / 2477 | 47 / 97 | 324 / 697 | 576 / 1329 | 612 / 1461 | 1385 / 3218 |
| iq2_s | 2.56 | 47 / 103 | 344 / 739 | 401 / 911 | 399 / 882 | 1011 / 2422 | 46 / 96 | 320 / 689 | 571 / 1364 | 582 / 1378 | 1304 / 3151 |
| iq2_xs | 2.31 | 46 / 103 | 340 / 749 | 421 / 954 | 394 / 845 | 1045 / 2505 | 46 / 96 | 323 / 694 | 570 / 1270 | 590 / 1364 | 1346 / 3145 |
| iq2_xxs | 2.06 | 49 / 110 | 367 / 794 | 373 / 710 | 365 / 707 | 995 / 2364 | 46 / 96 | 316 / 687 | 531 / 1216 | 544 / 1251 | 1322 / 3047 |
| iq1_s | 1.56 | 43 / 97 | 336 / 710 | 290 / 579 | 298 / 622 | 1146 / 2744 | 49 / 97 | 354 / 695 | 531 / 1222 | 575 / 1328 | 1263 / 3141 |
| iq1_m | 1.75 | 45 / 108 | 379 / 790 | 267 / 579 | 303 / 652 | 1132 / 2731 | 51 / 105 | 400 / 772 | 527 / 1193 | 557 / 1277 | 1204 / 2869 |
| q8_0 | 8.50 | 89 / 141 | 662 / 1071 | 872 / 1066 | 748 / 971 | 1135 / 2473 | 42 / 99 | 406 / 750 | 692 / 1098 | 702 / 1154 | 1156 / 2670 |
| q4_0 | 4.50 | 42 / 105 | 445 / 770 | 608 / 755 | 566 / 703 | 1007 / 2298 | 41 / 88 | 319 / 645 | 607 / 987 | 598 / 1019 | 1100 / 2536 |
| q4_1 | 5.00 | 38 / 110 | 470 / 807 | 635 / 770 | 588 / 726 | 1020 / 2317 | 40 / 86 | 323 / 637 | 578 / 954 | 560 / 986 | 1029 / 2361 |
| q5_0 | 5.50 | 41 / 115 | 497 / 841 | 654 / 789 | 671 / 767 | 1063 / 2376 | 41 / 91 | 341 / 668 | 614 / 1041 | 612 / 1069 | 1155 / 2618 |
| q5_1 | 6.00 | 43 / 119 | 523 / 880 | 635 / 764 | 606 / 719 | 1049 / 2368 | 44 / 92 | 357 / 698 | 582 / 960 | 583 / 1022 | 1047 / 2412 |
| tq1_0 | 1.69 | 77 / 200 | 738 / 1519 | 548 / 1260 | 589 / 1361 | 1212 / 2891 | 87 / 204 | 838 / 1532 | 674 / 1462 | 668 / 1489 | 1437 / 3609 |
| tq2_0 | 2.06 | 39 / 88 | 319 / 635 | 415 / 591 | 329 / 591 | 963 / 2295 | 50 / 105 | 372 / 769 | 579 / 1168 | 578 / 1217 | 1301 / 3040 |
| mxfp4 | 4.25 | 44 / 104 | 424 / 775 | 658 / 916 | 486 / 700 | 1041 / 2399 | 47 / 107 | 350 / 776 | 472 / 1043 | 505 / 1076 | 1142 / 2664 |
| nvfp4 | 4.50 | 49 / 108 | 424 / 823 | 820 / 796 | 762 / 668 | 1107 / 2431 | 45 / 96 | 349 / 715 | 597 / 1084 | 601 / 1093 | 1074 / 2448 |
| q1_0 | 1.12 | 38 / 85 | 284 / 604 | 262 / 484 | 272 / 551 | 931 / 2218 | 37 / 77 | 256 / 530 | 407 / 967 | 414 / 977 | 1033 / 2471 |
| q2_0 | 2.25 | 42 / 95 | 329 / 672 | 417 / 532 | 392 / 647 | 973 / 2312 | 46 / 103 | 344 / 740 | 403 / 911 | 398 / 882 | 1010 / 2420 |
| f32 | 32.00 | 274 / 377 | 1946 / 2875 | 2963 / 4127 | 3103 / 5090 | 3424 / 9744 | 142 / 222 | 1062 / 1701 | 1286 / 1951 | 1287 / 2029 | 1573 / 3340 |
| f16 | 16.00 | 144 / 214 | 1072 / 1647 | 1043 / 1481 | 1030 / 1465 | 1424 / 2639 | 84 / 134 | 599 / 1045 | 815 / 1329 | 819 / 1399 | 1162 / 2774 |
| bf16 | 16.00 | 145 / 214 | 1072 / 1647 | 1043 / 1484 | 1027 / 1464 | 1475 / 2778 | 84 / 134 | 600 / 1046 | 815 / 1327 | 819 / 1397 | 1117 / 2722 |

The types of 2 to 6.6 bits lie within 1.3 times of the median of their column, with these
exceptions. *The float types and Q8_0* are bound by their bytes: a step of one row reads eight
experts once, 104 MiB for F32 as gate and up, 54 for F16 and BF16 and 30 for Q8_0, at 350 to
430 GB/s on the RTX 4070 of the 473 the card reads in a loop of loads. *TQ1_0* is 1.6 to 2.4
times as slow as its neighbours in the decode kernels (120 to 150 GB/s at one row): its
bytes hold five values each in planes 32 values apart, so a lane loads five words of
activations and five scales from different places, which is what bounds the dense decode
kernel of the type too (`mmvq_more.cuh`). *Q6_K* is 1.3 to 1.8 times as slow as the median
in the many-row kernels: its item is 18 raw words per row stream, which the adapter holds
for four streams and for the step ahead. *NVFP4* as gate and up is 1.35 to 1.45 times as slow as the other 4-bit types in the
many-row kernels at 8 and 64 rows on the RTX 4070 (820 and 750 microseconds against 565 to
610 and 550 to 570, the same in two runs) and not on the RTX 3060 or at 512 rows; the cause
was not found.

*The four types of the real file, before and after* (`examples/expert_rows_bench` and
`expert_layer_bench` on the real experts and the recorded routes at 1, 8, 64, 512 and 4096
rows, the build of before and this one started alternately, twice on each card, the better
of each; load average 2 to 6): the decode kernels within 1 % at every point on both cards
(they are the same instantiations). Gate and up of the many-row kernels within 4 % on the RTX
4070, faster at most points, and within 1 % on the RTX 3060 but for IQ2_XXS, 7 % slower at 8
rows (453 to 485 microseconds) and 4 % faster at 4096. Down of the many-row kernels the same
at 1 and 4096 rows and faster between: 6 to 18 % on the RTX 4070 (1251 to 1058 microseconds
for IQ2_S's layer at 512 rows), 10 to 24 % on the RTX 3060.

*Speed on the CPU by type* (`ares-runtime`'s `examples/expert_cpu_types`, 2026-11-12, nine
workers as the engine places them on this machine, load average 3 to 5 from other work;
the median time of a job of one expert that 1 or 8 rows use, in microseconds, from the
creation of the job to its last task, which is what the cost model times; the same shapes as
above):

| type | gate and up: 1 row | 8 rows | down: 1 row | 8 rows |
| --- | ---: | ---: | ---: | ---: |
| q4_K | 147 | 399 | 105 | 325 |
| q5_K | 214 | 468 | 143 | 368 |
| q6_K | 267 | 520 | 162 | 385 |
| q3_K | 507 | 757 | 267 | 471 |
| q2_K | 473 | 728 | 254 | 467 |
| iq4_xs | 271 | 517 | 165 | 404 |
| iq4_nl | 265 | 517 | 139 | 326 |
| iq3_s | 273 | 521 | 167 | 395 |
| iq3_xxs | 277 | 533 | 171 | 396 |
| iq2_s | 48 | 195 | 222 | 446 |
| iq2_xs | 379 | 647 | 226 | 462 |
| iq2_xxs | 55 | 210 | 229 | 451 |
| iq1_s | 282 | 547 | 178 | 419 |
| iq1_m | 71 | 262 | 188 | 449 |
| q8_0 | 131 | 390 | 91 | 279 |
| q4_0 | 208 | 472 | 122 | 304 |
| q4_1 | 231 | 478 | 131 | 312 |
| q5_0 | 298 | 562 | 157 | 336 |
| q5_1 | 310 | 572 | 163 | 339 |
| tq1_0 | 226 | 485 | 144 | 383 |
| tq2_0 | 141 | 395 | 110 | 347 |
| mxfp4 | 256 | 508 | 134 | 319 |
| nvfp4 | 296 | 551 | 149 | 336 |
| q1_0 | 513 | 770 | 247 | 415 |
| q2_0 | 514 | 774 | 48 | 194 |
| f32 | 338 | 474 | 177 | 359 |
| f16 | 471 | 700 | 213 | 383 |
| bf16 | 163 | 401 | 96 | 276 |

An expert of the types with a written kernel takes 48 to 71 microseconds when one row uses
it and 195 to 262 when eight do; a decoded type takes
131 to 514 as gate and up (2.7 to 11
times the 48 of IQ2_S) and 390 to 774 for eight rows, 91 to 267 and 276 to 471 as down (Q2_0:
48 and 194). An upload of one expert takes 0.1 ms at the median on the RTX 4070 and 0.4 ms
over the link of the RTX 3060 (measured with the real file, see "Qwen3.8-Flash-Next through
the runtime"). So the CPU is sooner for a single miss of the written types on both cards; a
decoded type is sooner only against the slower link, and not when several rows use the
expert. The store decides that per layer and step from the times it measured for the layer's
pair of types and for the device's uploads; nothing in it names a type as slow.

*The kernels for the types of the Flash-Next files* (2026-11-22, the test machine: Core Ultra 5
235HX, 6 performance and 8 efficiency cores, AVX2 and AVX-VNNI, DDR5; `expert_cpu_types` with 13
workers as the engine places them, load average 1.6 to 2.8, no engine running; the same shapes as
the table above; microseconds per job of one expert, the build of before, `verify/stack`, and this
one one after the other):

| type | as | before: 1 row | 8 rows | now: 1 row | 2 rows | 4 rows | 8 rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iq3_xxs | gate and up | 157 | 306 | 37 | 47 | 75 | 144 |
| iq4_xs | gate and up | 156 | 307 | 28 | 39 | 68 | 127 |
| q4_K | gate and up | 108 | 257 | 33 | 40 | 71 | 128 |
| q5_K | gate and up | 115 | 289 | 32 | 49 | 103 | 170 |
| q2_0 | gate and up | 261 | 411 | 21 | 24 | 39 | 70 |
| q5_0 | down | 71 | 176 | 36 | 52 | 80 | 157 |
| iq4_nl | down | 72 | 189 | 33 | 48 | 79 | 154 |
| q4_0 | down | 58 | 164 | 32 | 48 | 79 | 153 |
| q4_1 | down | 62 | 183 | 32 | 48 | 80 | 156 |
| q5_1 | down | 73 | 178 | 37 | 51 | 85 | 163 |
| q8_0 | down | 44 | 150 | 33 | 49 | 79 | 150 |

The pairs of types the files have, as their layers do (now; the same run):

| gate and up / down | pool, 1 row | 2 rows | 4 rows | 8 rows | one thread, 1 row | its rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iq3_xxs / q5_0 (UD-Q3_K_XL) | 45 | 57 | 88 | 179 | 282 | 8.4 GB/s |
| iq3_xxs / iq4_nl (UD-Q3_K_XL) | 44 | 55 | 87 | 169 | 242 | 9.0 GB/s |
| iq4_xs / iq4_nl (UD-Q3_K_XL) | 37 | 48 | 81 | 152 | 168 | 15.9 GB/s |
| iq2_s / q2_0 (IQ2_XS) | 30 | 41 | 66 | 128 | 140 | 10.8 GB/s |
| q4_K / q8_0 (UD-Q4_K_XL) | 39 | 49 | 87 | 151 | 408 | 8.8 GB/s |
| q5_K / q5_1 (UD-Q4_K_XL) | 49 | 62 | 101 | 189 | 441 | 7.9 GB/s |

- A job of one expert for one row of UD-Q3_K_XL's types took 268 to 373 us at load before
  (the server's calibration); it takes 37 to 45 now. Jobs of several experts, as a layer of a
  decode step has them (IQ3_XXS / Q5_0, `expert_cpu_types layers`): 2 experts 63 us, 3 97,
  6 217, 10 430 for one row; 3 experts of 4 rows 201, 10 of 8 rows 1307. Three experts of one
  row read 7.2 MB in 97 us, 74 GB/s: the pool is near what the memory gives.
- On one core the kernels are bound by decoding, not by the bytes: 8 to 16 GB/s on one
  performance core. Gate and up of IQ3_XXS spend most of it on the codebook (a gather, or
  eight loads where they are faster on the core); the down products of a single pair were
  bound by the chain of their 20 block sums, so a down task takes four rows of the matrix at
  a time for a pair that is alone in its call (on an i5-13400F's performance core the down
  product of one row went from 185 to 124 us for Q5_0 and from 177 to 104 for IQ4_NL).
  Computing IQ3_XXS's signs from the parity of their index instead of a table was slower on
  an efficiency core and is not done.
- The order of a job's tasks (above): jobs of several rows 16 to 23 % sooner than with every
  down piece behind all gate and up pieces (3 experts of 4 rows 195 to 206 us against 260 to
  261, alternating twice at load 15), jobs of one row about the same (7 % slower at 10
  experts).

**Layer kernels** (`kernels/layers.cu`, `ares_cuda::LayerKernels`).
Everything a block of `qwen35` does between its matrix products: the
embedding gather (all 28 weight types), the norms per head, the causal
convolution with its window of earlier rows, the gated delta rule as a
window that reads the state and a commit that writes the accepted prefix,
the gated output norm and SwiGLU fused with the quantization of the next
product's activations, the rotary table and the preparation of queries and
keys. The rotary table is built once per step in f64, because an f32 angle
is off by about 0.01 rad at position 262k. A step record, uploaded once per
step, tells the kernels which rows belong to which sequence and state slot,
their positions, how many rows were accepted, and which rows are certain: a
prompt row or a row without a draft writes its state directly and needs no
commit. The same kernels serve `qwen4exp`, whose recurrent layers differ
only in the gate of the output norm.

Against the f64 CPU reference every kernel is within 7e-7. A chain of all 18
recurrent blocks of Qwen3.5-0.8B on the GPU, with the weight kernels in
between, is within 1.0e-3 to 1.5e-3 of the reference with exact products and
within 1.6e-2 to 1.9e-2 with 8-bit activations; the second number is the
rounding of the weight kernels.

Measured 2026-11-10 at the shapes of Qwen3.8-27B. A decode row costs a
recurrent block about 21 µs on the RTX 4070 outside its products, of which
the delta rule takes 10 to 14: it moves the 3 MiB of state of the block and
is bound by memory. A prompt of 2048 rows costs the delta rule 1.16 ms per
block (2.44 ms on the RTX 3060), which is 2.3 times as fast as the thread
mapping of the llama.cpp fork's kernel in the same harness. The shape (two lanes
a row of the matrix, 64 rows a block, 16 rows of the segment staged at a time) was
the best of ten measured for prompt rows on both cards; the others and the port of
the fork's design were removed on 2026-11-24 (`kernels/layers.cu`). The rule is
sequential in the rows; a chunked form on the tensor cores is not built.

A row of a draft window has the bits of the row alone in these kernels: the
convolution takes its window from the raw inputs of the rows before it, the
delta rule carries the matrix from row to row in the registers exactly as a
step of one row stores and reloads it, and the commit replays the accepted
rows with the same operations (`crates/model/tests/qwen35_windows.rs` holds
the outputs, the state and the logits to that, bit for bit).

The state of the delta rule can be held in `f16` (`StateFormat::F16`), which
halves the 150 MiB per sequence of the 27B and takes the delta rule of four
sequences from 55 to 32 µs (measured before the rounding per row that
follows). The matrix is then rounded to `f16` after every
row, in the registers of a window, a prompt and a commit as well, because a
step of one row stores the rounded matrix: a row must not depend on the step
that computed it. Measured on the CPU reference with the 0.8B over
1000 tokens of real text, rounding the state to `f16` after every step moves
the logits by a KL divergence of 1e-6 to 3e-6 and never changes the most
likely token, but the error grows with the length (the hidden state is off
by 7e-4 over the first 250 tokens and by 1.8e-3 over the last). Rounded
once, as a checkpoint would be, it is 1e-8 for `f16` and 1e-4 to 2e-4 for
int8 with a scale per row; int8 with a scale per head is unusable. `f32`
stays the default for the live state until `f16` is measured on the 27B and
on long sequences; checkpoints can be `f16`.

**Attention kernels** (`kernels/attention.cu`, `ares_cuda::AttentionKernels`,
`attention::Planner`). They read K and V in the stored formats, with no
copy in another type. On the device a sequence is a table of entries of 128
cells, the same for every layer; an entry is either exact (K and V rows in
`f16`) or converted (per KV head the mean and the rows of K, then of V, in
the bytes `ares_quant::kvhead` stores). Formats without a mean (`f16`,
`q8`) are written straight into converted entries. For `q7` to `q4` the
first tile and the newest tiles are exact entries, and a full tile is
converted with its mean when the policy says so. One kernel template serves
all 36 pairs of K and V formats (a switch per chunk), in shapes for up to 8,
16, 48 and 64 query vectors, over dense windows and over lists of ranges,
for several sequences in a step; the rows of a prompt have a kernel of their
own of up to 128 vectors (below). A planner cuts a step into items of one
block each, and their results are merged as log-sum-exp states, so a head
can be computed over any split of its cells.

Writing is bit-exact against the golden files of the formats and against
`KvHead` for appends, drafts and conversions. Reading is within about 1e-3
of the largest output element of `KvHead::attend` on the same stored bytes
(6e-3 at worst), which is the `f16` operands of the tensor cores; it does
not depend on the format, and `q4` / `q4` on captured rows of Qwen3.8-27B
is as close as `f16` / `f16`.

*Generated rows* (`Planner::window`). The row of a decode step and the rows of
a draft window must give the same bits alone and together, or a reply would
depend on what was drafted for it. Three things decide the bits of a row, and
for these rows each is fixed by the row alone:

- The pieces are those of a step that holds nothing but the row: the whole
  budget of blocks (three per multiprocessor, by the KV heads of the device,
  at most 64 pieces), cut over the entries the row reads. Rows of a window
  that read the same entries have the same pieces and share their items, so
  the cells are read once for all of them; a window that begins a new entry
  is two tiles of rows in that step. Other sequences change nothing, and
  such rows read no prefix together with other sequences.
- The kernel is the tensor-core kernel in its decode shapes (8, 16 and 48
  vectors), for one row as for eight: a vector has the same arithmetic in
  each of them, whatever shares the item, and the cells past a row's own are
  masked to exact zeros. The direct kernel, which is faster for one row over
  quantized K and V, is not taken for these rows: a window of rows has no
  kernel with its arithmetic that reads the cells once (its work is per
  vector, eight rows would cost it eight times one). It still serves the rows
  that see a list of ranges (`qwen4exp`) and the links of the MTP block's
  chain, whose results are proposals.
- The entries are the caller's: with `q7` to `q4` the rows of a window past
  the position at which a tile leaves the newest cells read it converted (see
  "Where the tile formats keep their exact cells").

*Rows of a prompt* (`AttnShape::Prompt128`, `kernels/attn_prompt.cuh`, 2026-11-23). A prompt's
rows are cut into tiles of up to 128 vectors (21 rows of the 27B's 6 query heads per KV head,
the tiles of a step as equal as they can be), and each tile into the same number of pieces: the
count that fills the waves of the prompt kernel best (one block per multiprocessor, its
registers fill one; `PlanConfig::prompt_blocks`), within the tiles' share of the budget of pieces.
The items run piece by piece and KV head, so the blocks that run at once read the same cells and
the L2 cache serves them all. (Sharing the cells out evenly over one wave instead, blocks reading
several pieces one after the other, was built and measured: 15 % slower on both cards, because the
blocks that run at once then read different cells.) A block has 8 warps of 16 vectors and walks
its range in chunks of 32 cells; the queries stay in shared memory, the accumulator in registers
(f32), and the stored rows of the next chunk are loaded into registers while the tensor cores
work on this one.

- *Scores in integers* (converted entries with K in `q8` to `q4`, `ares_attn_d*_p128i`). The
  codes of K are integers: the query becomes a 15-bit integer per value (rotated, scaled, one
  scale per vector) split into a signed high byte and an unsigned low one, two int8 products
  (`m16n8k32`) per group of 32 values sum it exactly in int32, and the group's scale of K is
  applied in f32. On GeForce cards f16 products with f32 sums run at half rate; the int8 ones
  run at four times that, so the scores cost what f16 sums would, and they are closer to the
  reference than f16 operands (the wide window of 300 rows: `q8`/`q8` 5e-4 to 8e-4 of the largest
  output against 1.9e-3 to 3.3e-3). The tile mean's dot products with the queries come from the
  same products: the K mean as a 15-bit integer per value (a scale per group) in two rows of a
  small tile whose other rows read a row of zeros, built by one warp per group in the phase before
  the tile begins and read where the warps decode (as f32 sums over the integer queries' bytes they
  took 15 % of the kernel with `q5`; they now take about 1 %).
- *P V* in f16 with f16 sums over the 32 cells of a chunk, added to the f32 accumulator after each
  chunk (one rounding of a sum of 32 products, the size of the weights' own f16 rounding; summing
  in f16 over the whole range, as llama.cpp does, would grow with the length).
- *Phases.* A chunk has two phases between barriers, each with work for the tensor cores and work
  for the other pipes: the scores with the decode of V, then the weights and P V with the decode
  of the next chunk's K; warps 4 to 7 compute first and warps 0 to 3 decode first, so the two
  warps of a scheduler overlap.
- Exact entries and `f16` K run on the same kernel with f16 scores (`ares_attn_d*_p128`); the
  wide kernel (64 vectors) is left for items of a list of ranges.

It costs nothing that was measured. 2026-11-15, the build before against the
build after, back to back, without the profile, K and V in `q8`, the machine
at a load of 15 to 27 from other work. A decode step without drafts, which is
the step that changed kernels: on the RTX 4070 with the `IQ2_XXS` file 20.94
ms before and 20.78 ms after at 1000 tokens of context, 21.95 and 21.90 ms at
8000, 24.03 and 24.02 ms at 32000 (attention over the 16 layers took 0.66 ms
of a step on the direct kernel and 0.52 ms on the tensor cores at 1000 tokens,
1.52 and 1.50 ms at 8000); over both cards with the `Q4_K_S` file 24.83 and
24.54 ms at 1000 tokens. A prompt of 2048 tokens: 1167 and 1171 tokens per
second before, 1170 and 1170 after on the one card; 1350 and 1355 before, 1358
and 1363 after on both. With drafts (`qwen35_run --engine 8 --generate 300`,
the draft vocabulary, the most probable token, two runs of seven requests
with each build) a window's step is the step it was, as far as a loaded
machine shows: over both cards the step median of a request was 31.41 to 31.47
ms before in the quietest stretch (three requests; 34 to 39 ms in the others,
as the load rose) and 31.42 to 31.77 ms after (the seven requests of the
quieter run; 32.7 to 37.2 ms in the other), and the tokens per second follow
the load and the text (59 to 82 before and 64 to 80 after for prose and code,
109 and 127 before and 120 and 138 after for the rewrite). On the one card a
run where the policy chose one link both times had 130 steps of 23.92 ms
before and 130 of 24.00 ms after for the same 300 tokens (91.6 and 91.0 tokens
per second); where it chose two links in one run and one in the other the
steps cannot be compared. The
scratch of a step grew by 1.6 MiB on the one card and by 1.2 and 1.0 MiB on
the two (the states of the pieces of rows planned alone).

Measured 2026-11-10 per attention layer (Qwen3.8-27B has 16), one sequence,
RTX 4070 / RTX 3060:

| K / V | 1 row, 4096 cells | 32768 cells | 262144 cells | 8 rows, 262144 cells | prompt of 2048 rows |
| --- | --- | --- | --- | --- | --- |
| `q8` / `q8` | 50 / 90 µs | 0.28 / 0.46 ms | 2.19 / 3.48 ms | 3.14 / 5.39 ms | 2.73 / 5.95 ms |
| `q4` / `q4` | 49 / 92 µs | 0.23 / 0.51 ms | 2.11 / 3.62 ms | 3.47 / 6.08 ms | 2.41 / 5.22 ms |
| `f16` / `f16` | 54 / 97 µs | 0.36 / 0.53 ms | 2.76 / 3.98 ms | 3.44 / 5.84 ms | 2.43 / 5.26 ms |

At 262144 cells `f16` reads at 77 % of the memory limit, `q8` at 52 % and
`q4` at 29 %: the quantized formats are bound by the work of decoding, not
by memory, so they are not yet faster than `f16` although they read a half
or a quarter of the bytes. Open: that decode work, accumulation in `f16`
with a flush to `f32` on capped cards (every path accumulates in `f32`
now, at half the rate), and the item for a prefix that several sequences
share.

**The tensor-core cap of GeForce cards.** GeForce cards run FP16, BF16 and
TF32 tensor-core math with FP32 accumulation at half rate; professional cards
with the same silicon do not, and gpu-unlock
(https://github.com/Zagitta/gpu-unlock) removes the cap on RTX 20, 30 and 40
at boot. Kernels can work around the cap (integer tiles, or accumulating in
FP16 and flushing to FP32, as above), at the price of a small loss of
exactness. Without the cap the direct path is as fast at the tensor cores and the
exact one; whether it is as fast in the tile kernels of the file's layout depends
on the type (below), so there it is measured per type.

There is no build option for this. Each device is probed for a few
milliseconds when the kernels are loaded and the result is in the load log
(`ares-server`, `load/products.rs`):

- **Tensor cores at full rate:** the direct path (f16, f32 accumulation) is
  used wherever it is as fast; the tile kernels of the file's layout take, per
  type, the path a probe at load found fastest (`Kernels::measure_prefill_path`,
  "Prefill weight kernels" above), the exact one on ties.
- **Capped:** the workaround runs, because it is faster there, and the log
  says so and names the remedy: per type, the path a probe at load found
  fastest of int8 and f16 with f16 accumulation (the more exact on ties).

There is no option for the exact path either (`--precision` went on
2026-11-26, issue #54): the owner decided that exact products everywhere are
not a trade-off worth a choice, since the text of the faster path differs
from theirs only slightly (8-bit activations: 1.05 % rms per product, f16
sums 0.05 %, against the f16 operands' 0.03 %), so what is faster runs, on a
capped card and an unlocked one. What exact products bought was a text that
does not depend on what else runs in a step; the fast path gives that for
generated rows (a single row and a window of drafts compute the same bits in
any decode step, so the text with drafts is the text without), and a prompt
row's rounding depends on the size of its step, so a text next to others or
from the prompt cache may part at a near tie. `Precision::Exact` stays as a
tool of the tests and benches, which hold a device to the CPU reference with
it; no option of the server reaches it.

The development machine's cards are capped, and the probe shows it exactly:
f16 `mma` with f32 accumulation runs at 58.6 TFLOPS on the RTX 4070 and 25.6
on the RTX 3060, with f16 accumulation at 117.0 and 51.2, and int8 at 234 and
102 TOPS. The probe takes about 10 ms in `Kernels::load`
(`tensor_cores_full_rate()`). The test machine's RTX 4060 Ti with gpu-unlock
(34 multiprocessors, 32 MiB of L2, a power limit of 165 W) runs a loop of f16
`mma` at 96.7 TFLOPS with f32 accumulation and 96.9 with f16, int8 at 194 TOPS
(2790 MHz; under the tile kernels the card sits at its power limit at 2655 MHz),
and the probe reports full rate.

The first weight kernel, an f32 Q4_K kernel kept as a reference (`ares_matvec_q4_K`), was
removed on 2026-11-24: every decode kernel is held to a CPU emulation of its arithmetic and to
`ares_quant::dequantize` (`tests/mmvq.rs`).

One thing the device tests caught is worth knowing: a "synchronous" upload
from ordinary memory returns when the data is in the driver's staging buffer,
not when it is on the device, and a kernel on a stream does not wait for it.
`DeviceBuffer::write` therefore waits for the device, and the hot path only
uses page-locked memory on streams.
