# Experiments: the profile of the kernels (#14)

Where the time of real steps goes by kernel family, on both cards, to lead the pass over all
kernels (#14, #15, #18). Measured 2026-11-28 on the development machine (RTX 4070, device 0,
479 GB/s and 58.6 TFLOPS f16 with f32 sums; RTX 3060, device 1, 341 GB/s and 25.6 TFLOPS; both
capped GeForce cards; peer access through the patched driver), every timed run at a 1-minute
load below 1.5 unless said, nothing else on the cards.

**Models and settings.** Qwen3.8-27B UD-Q4_K_S at the production settings
(`/var/tmp/ares-tools/prod-ab/session.sh`: tensor split 0.625 / 0.375, K and V in `q5`, the MTP
block with the draft vocabulary of 66 343 tokens). Qwen3.8-Flash-Next IQ2_XS on both cards with
a RAM tier of 8 GiB (the trial at load chose the RTX 3060 as a worker that holds experts only,
0.612 of the slots, and the RTX 4070 as the mixer with everything else).

**Tools.** `qwen35_run` and `qwen4exp_run` with `--windows` (steps of draft windows of each row
count, whose first row is accepted; the price of a step by its rows, as the conductor's warm-up
measures it), `--engine` (decode with the model's drafts through the conductor),
`--prefill`, `--breakdown` (Flash-Next's own device time by kind). Kernel times are from
Nsight Systems traces of the same runs (CUDA activity only; a per-kernel table by family and
row count), `few_rows_bench`, `expert_layer_bench`, and Nsight Compute, which works for
non-root users on this machine now (`RmProfilingAdminOnly` is 0). Traces of Flash-Next are
not usable for its kernel times: its kernels that wait for experts, the host and the other card
spin, and they spin longer under tracing; its numbers below are the engine's own breakdown and
the benches.

## Qwen3.8-27B

### The price of a step by its rows

Draft windows at the position given, ms per step (median of 32 or 24 steps), and the time
against one row. Before is `0f1df3b8` (main with the tools), after is the branch with the
few-row kernels and the conversion of tiles changed (below); alternating runs, measured layouts
(what the server runs) unless said.

| Cells | | 1 row | 2 | 4 | 8 | 9 | 12 | 16 | 16 against 1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 000 | before | 24.74, 24.75 | 25.56 | 26.81, 26.90 | 30.56, 30.53 | 37.13, 35.54 | 40.60, 39.21 | 57.50, 56.49 | 2.30 |
| | after | 24.74, 24.77 | 25.58 | 26.98, 26.85 | 30.44, 30.54 | 35.11, 36.72 | 37.13, 38.85 | 42.57, 44.61 | 1.76 |
| 1 000, every matrix in the device layout | before | 24.80, 24.78 | 25.62 | 26.95, 26.92 | 30.15, 30.15 | 33.96, 33.99 | 37.61, 37.72 | 54.83, 54.84 | 2.21 |
| | after | 24.77, 24.78 | 25.57 | 26.91, 26.88 | 30.16, 30.09 | 33.96, 33.89 | 35.89, 35.87 | 41.26, 41.25 | 1.67 |
| 32 768 | before | 28.72 | | 33.82 | 38.54 | 49.93 | | 63.79 | 2.22 |
| | after | 28.69 | | 33.78 | 38.36 | 49.54 | | 58.84 | 2.05 |

The 9 to 12 rows of the measured layouts move with the load's layout choice (the RTX 3060 kept
28 to 58 matrices in the file's layout in these loads). At 8 192 cells before: 1 row 27.44,
4 rows 30.80, 8 rows 35.28, 16 rows 56.19 ms (2.05 times).

### Where a step goes, by kernel family (ms per step)

Window steps at 1 000 cells, a trace of each (the windows' closing step of one row taken out).
Before:

| Family | RTX 4070: 1 row | 8 | 9 | 16 | RTX 3060: 1 row | 8 | 9 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| few-row kernels (device layout) | 19.12 | 19.30 | 20.38 | 23.60 | 15.93 | 16.70 | 18.79 | 25.09 |
| decode kernels (file layout; two launches of 8 above 8 rows) | 2.04 | 2.78 | 4.76 | 5.53 | 2.49 | 4.11 | 6.62 | 8.22 |
| merges of the tensor split (the link and the wait for the other card) | 2.34 | 6.73 | 9.24 | 17.41 | 3.24 | 5.54 | 6.09 | 8.76 |
| attention (with the conversion of tiles) | 0.80 | 0.79 | 1.10 | 3.57 | 1.02 | 1.05 | 1.47 | 4.86 |
| delta rule and convolution (window and commit) | 0.55 | 1.28 | 1.33 | 1.63 | 0.63 | 1.45 | 1.53 | 1.95 |
| norms, SwiGLU, activations, embedding, sampler | 0.40 | 0.42 | 0.52 | 0.55 | 0.45 | 0.47 | 0.61 | 0.66 |
| idle between kernels | 1.35 | 1.33 | 1.41 | 1.40 | 2.84 | 3.33 | 3.64 | 4.18 |

After (the same run with the branch):

| Family | RTX 4070: 1 row | 8 | 9 | 16 | RTX 3060: 1 row | 8 | 9 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| few-row kernels (device layout) | 19.17 | 19.34 | 20.07 | 20.70 | 17.10 | 17.81 | 20.03 | 20.95 |
| decode kernels (file layout) | 2.00 | 2.72 | 4.68 | 5.43 | 1.38 | 2.66 | 4.04 | 5.33 |
| merges | 2.29 | 6.36 | 8.24 | 13.16 | 3.33 | 5.61 | 6.24 | 9.48 |
| attention | 0.48 | 0.81 | 1.13 | 1.40 | 0.63 | 1.08 | 1.51 | 1.94 |
| delta rule and convolution | 0.56 | 1.28 | 1.34 | 1.64 | 0.63 | 1.48 | 1.54 | 1.97 |
| idle between kernels | 1.36 | 1.33 | 1.39 | 1.40 | 2.74 | 3.17 | 3.45 | 4.02 |

(The two loads put different matrices in the file's layout on the RTX 3060, 47 and 72: hence the
few-row and the file-layout rows of one row.)

**Against what the cards allow.** At one row the weights of a step run at 92 % (RTX 4070: 8.4 GB
in the device layout in 19.1 ms, 441 GB/s) and 86 % (RTX 3060: 4.7 GB in 15.9 ms, 295 GB/s) of
the memory bandwidth; the RTX 3060's shapes of 2304 to 6656 rows reach 290 to 315 GB/s alone, the
small matrices of tens and hundreds of rows (96 of them a card) far less. The decode kernels of
the file's layout (the codebook i-quants IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, which have no device
layout, and the IQ4_XS, Q4_K and Q5_K shapes the chooser leaves there) run at 77 % on the RTX 3060
(IQ3_S, 262 GB/s). At 16 rows the f16 tensor work of the few-row kernels is 11.2 ms on the RTX
3060 at its rate with f32 sums, below the 15.9 ms of its weights; they were bound by neither,
but by the loads of their activations (below), and are at 20.95 ms now, 1.23 times one row
(RTX 4070 1.08 times). The merges carry 16 rows x 5 120 values as f16 129 times a step, 21 MB
over the RTX 3060's link of 3.3 GB/s: 6.4 ms, which is what they take; the RTX 4070's merges
are that wait. The idle time between kernels is 1.3 ms a step on the RTX 4070 and 2.7 to 4.2 ms on
the RTX 3060 (about 1 100 launches a step).

**Decode with drafts** (`--engine 8`, three requests of 300 tokens, German, English, code, the
most probable token; alternating, two rounds): 84.8 / 89.6 / 84.1 and 84.6 / 88.7 / 84.4
tokens/s before, 82.6 / 87.2 / 84.5 and 85.2 / 88.1 / 83.5 after (the same: windows of up to 8
rows do not run the changed kernel); 2.9 to 3.1 tokens a step, a step 35 ms at the median. A step
is the verify window (mostly 5 or 6 rows: 28.6 to 30.1 ms by the prices the runs measured) and the MTP block's chain on the RTX 4070 (3.6 to
3.8 links of 1.7 ms: 6.3 ms), during which the RTX 3060 idles (about 10 ms a step in the trace).
A link reads the MTP block (335 MiB) and the draft head (278 MiB): 1.28 ms at the bandwidth, 1.7
measured.

**Prompts** (`--prefill`, steps of 512 rows, the second pass): 8 192 tokens 1 265 to 1 269
tokens/s, 32 768 tokens 1 138 to 1 144 tokens/s. A step of 512 rows at 32 k cells, ms:

| Family | RTX 4070 | RTX 3060 |
| --- | ---: | ---: |
| tile kernels of the device layout (f16 sums, `r64h`) | 190.8 | 258.7 |
| tile kernels of the file's layout (the i-quants on int8, Q5_K) | 27.0 | 33.4 |
| prompt attention (`p128i`, `p128`) | 47.8 | 62.9 |
| delta rule, convolution, norms, SwiGLU | 20.6 | 31.0 |
| merges and the partials on the wire | 9.6 | 22.1 |
| conversion of tiles | 7.4 | 9.7 |
| idle | 142.0 | 26.1 |

The RTX 3060 bounds the prompt: at 0.375 of the work with 0.44 of the tensor rate it takes 1.4
times the RTX 4070's time, which waits 116 to 128 ms of every step. At 8 192 tokens the prompt
attention is 16.2 and 20.2 ms of the step. Its weights run at 31 to 35 TFLOPS of the RTX 3060's
51.2 with f16 sums (`r64h`); the RTX 4070's at 52 to 60 of 117.

### What bounded the few-row kernels at 9 to 16 rows

The few-row kernels compute a tile of 16 weight rows on `mma` with the activations as the
other operand, which each lane loaded itself: 4 bytes of 8 rows a load, 16 such loads a step of
16 rows. Each asks L1 for 8 lines, and that, not the weights or the tensor cores, bounded the
kernel: on the RTX 3060 Q4_K 17 408 x 5 120 took 152 us for one row and 231 to 247 us for 16
(the four launches); the same kernel with every lane of a load reading one row took 160 to 169 us
(RTX 4070: 107.6 against 113, one row 106.6). Summing in f16 instead of f32 (the tensor cores'
double rate on these cards) changed nothing (231 us). The fix: `kernels.md`, "Few rows on the
device layout".

### The tile conversion at a window's edge

A window that crosses the edge of a tile converts the tile at the start of its step in every
attention layer (`q7` to `q4`): R - 1 of every 128 steps of windows of R rows. The kernel ran one
block a tile's K or V (4 to 6 blocks for a tile of the 27B on a device) and took 423 us on the
RTX 4070 and 588 us on the RTX 3060 a layer; it was 2.4 and 3.3 ms of the step averages above at
16 rows, where 6 of 21 steps crossed an edge. Now 31 and 51 us (`kernels.md`, the attention
kernels).

## Qwen3.8-Flash-Next IQ2_XS

### Device time by kind (`--breakdown`, ms per step, 1 000 cells, tiers warm, load 2.5 to 2.7 from the engine's own CPU workers)

| Step | step, ms | RTX 4070: dense | mixers | recurrent | attention | experts | RTX 3060: experts | exchange |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 row | 21.8 | 6.39 | 9.80 | 0.75 | 0.48 | 4.49 | 9.32 | 11.53 |
| window of 2 rows | 31.3 | 6.62 | 17.28 | 0.69 | 0.58 | 7.70 | 16.47 | 14.98 |
| window of 4 rows | 45.1 | 7.41 | 24.52 | 0.78 | 0.73 | 10.33 | 22.25 | 20.37 |
| window of 8 rows | 66.2 | 9.39 | 36.53 | 0.95 | 0.78 | 15.67 | 30.42 | 30.28 |
| wide step of 16 rows (a prompt read in steps of 16) | 92 | 17.07 | 49.38 | 1.47 | 1.02 | 20.64 | 43.89 | 46.48 |

A window of 8 rows costs 3.04 times one row; windows of more than 8 rows are refused (the decode
kernels take 1 to 8 rows), so 16 rows are a wide step on the many-row kernels. The RTX 4070's
"mixers" include its wait for the RTX 3060's experts (the read of a block's input merges them),
and the RTX 3060's "exchange" its wait for the RTX 4070: the RTX 3060's experts are the critical
path, 9.3 ms of a 22 ms step at one row and 30.4 of 66 at eight. The host takes 18 to 19 ms of a
step to enqueue it (`host per step: enqueue`), next to the 22 ms the devices take.

### The decode kernels of the experts against the bandwidth

`expert_layer_bench` on the real experts of one layer (every pick in a slot; microseconds a layer
for the experts, GB/s of the bytes the layer reads):

| Layer | rows | experts | RTX 4070: us | GB/s | RTX 3060: us | GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blk.0, IQ2_S / Q2_0 | 1 | 10 | 71.4 | 206 | 141.0 | 111 |
| | 2 | 20 | 117.1 | 238 | 246.7 | 120 |
| | 4 | 34 | 193.9 | 248 | 417.9 | 120 |
| | 8 | 64 | 359.9 | 255 | 785.1 | 121 |
| | 8, the same 10 experts | 10 | 127.5 | 125 | 247.9 | 67 |
| blk.1, IQ2_XXS / Q2_0 | 1 / 8 | 10 / 74 | 66.8 / 430.2 | 200 / 218 | 138.9 / 940.1 | 102 / 102 |
| blk.8, IQ1_M / Q2_0 | 1 / 8 | 10 / 68 | 71.4 / 416.5 | 177 / 188 | 137.3 / 874.9 | 96 / 92 |

An expert costs 5.6 to 7.1 us on the RTX 4070 and 12.2 to 14.1 us on the RTX 3060, where its 1.5
MB would take 3.1 and 4.4 us at the bandwidth. On the RTX 3060 (a trace of the bench) the gate and
up kernel of IQ2_S takes 63 us for ten experts (10.5 MB, 167 GB/s) and the down kernel of Q2_0
77.5 us (4.6 MB, 59 GB/s); at 8 rows 324 and 467 us. Nsight Compute on the down kernel (one row,
RTX 3060): 128 registers a thread, so a third of the warps a multiprocessor can hold, 8.2 million
warp instructions for 16.4 million weights, of which 62 % are integer work (LOP3 26 %, IMAD 14 %,
SHF 13 %, IADD3 9 %) and the dot products 3.7 %; DRAM at 17 % of its peak. The gate and up
kernel: 64 registers, 55 % of the warps, DRAM 48 %, the multiprocessors' issue at 65 %. Iterating
over the rows of an expert's mask with a branch instead of the predicated loop over all eight was
tried and was 30 % slower (a switch per row, and the loads of the rows no longer together).

## The levers, ranked

The gain is per step at the settings above, on the card that bounds the step; "16 rows" is the
cost of a step of 16 rows against one row, which #14 wants near 1.05 to 1.2.

1. **Flash-Next: the decode kernels of the experts** (#14). The RTX 3060's experts are the
   critical path of every decode step and window: at the bandwidth an expert costs 4.4 us there,
   today 12 to 14. Bringing the RTX 3060 to the RTX 4070's 255 GB/s would take its experts from
   9.3 to about 4.5 ms at one row (the step from 22 to about 17 ms, 25 % more tokens per second)
   and from 30 to about 15 ms in a window of 8 rows. The down kernels are integer-bound (the
   2-bit codes of Q2_0 unpacked for `dp4a`, 128 registers); the gate and up kernels at half the
   bandwidth. The way: a rearranged form of an expert in its slot (the device layout's idea: a
   warp's lanes load a tile of rows in one load) and the decode in registers for the tensor
   cores, as the dense few-row kernels do, or fewer instructions per code in the `dp4a` kernels.
2. **27B: the merges of the tensor split at 9 to 16 rows**: 3.3 to 9.5 ms on the RTX 3060,
   the RTX 3060's link (PCIe 3.0 x4) carrying the f16 partials. Not a kernel: fewer bytes on the
   wire (an 8-bit partial with a scale per row would halve it, at a loss the owner decides on)
   or overlapping a half block's merge with the next one's work. 6 ms of the 16-row step.
3. **27B: attention for wide windows** (#14). Windows of 9 to 16 rows run as two items of the
   decode shape of 48 vectors and read every cell twice: at 32 768 cells a step of 9 rows costs
   11.4 ms more than one of 8 (38.4 to 49.5 ms), of which the second pass over K and V is about
   3.3 ms (RTX 4070) and 4.8 ms (RTX 3060). A decode shape of 96 vectors (six warps sharing the
   vectors, as the 48 has three) would read them once.
4. **27B: the codebook i-quants in a device layout** (#18, the first lever). IQ3_S, IQ3_XXS,
   IQ2_S and IQ2_XS run on the decode kernels of the file's layout: 77 % of the RTX 3060's
   bandwidth at one row, and two launches of 8 rows above 8 (1.4 to 2.5 ms at one row, 5.3 to 8.2
   at 16, on the RTX 3060). In the device layout they would take the few-row kernels: about 0.4
   ms of a decode step and 3 to 4 ms of a 16-row step on the RTX 3060, and the prompt's tile
   kernels of the file's layout (27 / 33 ms of a 512-row step).
5. **Prompts: the RTX 3060's tile kernels and prompt attention** (#15). The RTX 3060 bounds the
   prompt (the RTX 4070 waits a third of every step); its prompt attention is 63 ms of a 512-row
   step at 32 k cells against the RTX 4070's 48, the weights 31 to 35 of 51.2 TFLOPS.
6. **Gaps between launches**: 1.3 ms (RTX 4070) and 2.7 to 4.2 ms (RTX 3060) of every step of
   the 27B; Flash-Next's host takes 18 ms to enqueue a step of 22 ms. CUDA graphs or a step that
   overlaps the host's work with the device's (#18).
7. **27B: the recurrent layers in windows**: the delta rule and the convolution grow from 0.6 to
   2.0 ms between one row and 16 (window and commit), the convolution's window kernel 4 times.
8. **The MTP chain**: 1.7 ms a link where its bytes take 1.28 ms at the RTX 4070's bandwidth (75
   %), 6.3 ms of a 35 ms step with drafts, while the RTX 3060 idles.

Done on the branch (`kernel/few-rows-pass`): the few-row kernels at 9 to 16 rows and the tile
conversion (the tables above): the 27B's step of 16 rows from 2.21 to 1.67 times one row with
every matrix in the device layout and from 2.30 to 1.76 with the measured layouts at 1 000
cells, 2.22 to 2.05 at 32 768; one row and windows of up to 9 rows unchanged.
