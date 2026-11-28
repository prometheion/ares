# Experiments: where a decode step goes with most experts off the card

The profile of issue #13 (2026-11-27): Flash-Next UD-Q4_K_XL (104 GiB) on the test machine's
RTX 4060 Ti 16 GB with most experts in RAM and on the SSD, split into the device's own work, the
misses from the SSD and from RAM, the CPU's share and the read-ahead. The ranking of the levers
of #16 at the end is made from it.

## The setup

- *Machine.* The test machine: RTX 4060 Ti 16 GB on PCIe 4.0 x4 (6.4 GB/s for a copy), Core
  Ultra 5 235HX (6 performance and 8 efficiency cores, one thread each), 62 GiB of RAM, a
  Samsung 990 1 TB under btrfs. Nothing else ran on it (`strata.service` stopped by the owner);
  the load averages of 6 to 14 are the runs' own CPU workers.
- *Builds.* `main` at 83408e52 (A) and this branch (B): fc9e2a26, the same code with the wait
  counters below, and b5e9ecff, which adds `--cpu-workers` to the tool.
- *Runs.* `qwen4exp_run --devices 0 --ram-gib R --cells 8192 --rows 512 --text-file
  text-mixed.txt --decode-at 4000 --decode-steps 256 --breakdown --serve model`: the first 4000
  tokens of a text of German and English prose and code are read in steps of 512 rows (which fill
  the tiers through transit), then 256 decode steps of one row, greedy and untimed, then 256 more
  with timed events. The model's pages are dropped from the page cache before each run.
  `--ram-gib 39.5` is the tier of the server's `--ram 40G`. The tiers: 3120 VRAM slots (12.7 % of
  the 24 576 experts, 65 a layer), 4800 RAM slots at 16 GiB (19.5 %) and 13 056 at 39.5 GiB
  (53 %).
- *Counters.* The store's (`StoreStats`: the tool's `store`, `drive` and `steps` lines), the
  device time by kind (`--breakdown`), and a line this branch adds: each layer's wait for its
  misses (from its route record on the host to the last miss in a slot or delivered) by what came
  last: the CPU's sums of experts in RAM, an upload from RAM, a read of the file (uploaded, or
  computed by the CPU once its bytes landed), an expert already on its way.
  `expert_layer_bench` gives the device's own price of a layer's experts with all of them in
  slots, and `expert_bench` what the drive gives.

## Where a step goes

Steps 1 to 256 after the context, the engine as it serves (B, `--serve model`); means over the
runs named. The device's time outside the experts is from the timed steps, where it is the same
(it does not depend on what is missed).

| Part of a step | `--ram-gib 39.5` (B1, B2, B3) | `--ram-gib 16` (B1, B2) |
| --- | ---: | ---: |
| Step: median / mean | 67.5 / 72.1 ms (13.9 tokens/s) | 158.6 / 181.0 ms (5.5 tokens/s) |
| Device: dense products (of them the predictions' two routers a layer) | 18.0 (1.7) ms | 18.0 (1.7) ms |
| Device: stream mixers | 7.8 ms | 7.8 ms |
| Device: recurrent layers, attention, exchange, the rest | 3.0 ms | 3.0 ms |
| Device: router and the experts VRAM holds | about 4.5 ms | about 4.5 ms |
| Waits for the CPU's sums of experts in RAM | 11.9 ms (37.8 layers, 0.31 ms each) | 3.8 ms (16.4 layers, 0.23 ms each) |
| Waits for reads of the file | 25.2 ms (6.9 layers, 3.63 ms each) | 142.5 ms (28.5 layers, 5.01 ms each) |
| Waits for uploads from RAM, for experts on their way | 0.05 ms | 0 |
| Sum of the parts | 70.4 ms | 179.6 ms |
| Picks in a VRAM slot | 57.0 % | 56.8 % |
| Misses a step: on the CPU / uploaded | 206.2 / 0.1 | 207.3 / 0.1 |
| Misses from RAM / from the file / on their way | 197.6 / 8.6 / 0.1 | 144.1 / 61.8 / 1.5 |
| Bytes from the file a step | 27.3 MB | 195 MB |
| An expert's reads (start to bytes in RAM, mean); a matrix at the reader (median) | 3.62 ms; 2.73 ms | 5.09 ms; 4.40 ms |

- *The parts add up to the step.* The device's own work (about 33 ms) and the layers' waits
  (37 and 146 ms) make the mean step within 3 %: the waits hardly overlap anything. The device
  computes while the host serves a layer's misses only the gate and up of the experts VRAM holds,
  about 0.1 ms a layer.
- *The device's own work* is 33 ms a step, the floor of decode on this card at 30 tokens/s. The
  dense products (4.6 GiB at the card's 288 GB/s) are at the card's bandwidth; the stream mixers
  (Q8_0, 644 MiB) take 7.8 ms where their bytes take 2.3 (#18). The experts in VRAM cost 158 to
  189 µs a layer for ten in slots (`expert_layer_bench`: 27 µs of it the router, 228 to 239 GB/s),
  and a step has 5.7 a layer in VRAM.
- *At 40 GiB the misses are the CPU's.* The cost model gives 99.95 % of them to the CPU (an upload
  of 3 MB takes 0.5 ms on the x4 link; the CPU computes a layer's 4.3 misses in 0.33 ms, about
  39 GB/s from RAM), and the layers wait for it 0.31 ms each; `--serve cpu` gives the same
  (65.4 ms). The 8.6 reads of the file a step take a third of the step.
- *At 16 GiB the step waits for the drive*: 62 reads a step, 79 % of the time.
- *The record's way to the host* costs nothing that shows: the host following a route record
  per layer adds 9 µs a layer (`expert_layer_bench`, 0.4 ms a step); a layer's first CPU job is
  made 9 µs after its record, and the sums are on their way 6 to 10 µs after the job.
- *Uploads.* None: with every miss on the CPU the link only carries the promotions between steps
  (7 to 10 a step). `--serve stream` is refused for this file on this card (a layer would need 88
  slots, the card holds 71).

## The drive, and the cores the reads wait for

The drive alone (`expert_bench`, the file's experts at random, three reads each, 1.5 GiB):

| Experts in flight | 1 | 2 | 4 | 64 |
| --- | ---: | ---: | ---: | ---: |
| GB/s | 1.53 | 1.99 | 1.98 | 1.80 |
| An expert, median | 1.99 ms | 3.05 ms | 6.14 ms | 106.5 ms |

(On 2026-11-22 it gave 2.5 GB/s at two in flight; the disk is fuller now.) Beside 13 spinning
threads it gives the same (1.91 ms, 2.00 GB/s). Decode reads one to two experts a layer, so it
should wait 2 to 3 ms a layer, and it waits 3.6 and 5.0 ms. The difference is the cores: the
store's CPU workers are pinned to every core but one (12 workers in the server, 13 here) and spin
between a step's jobs, the host thread spins on the last one, and the I/O thread, woken by each
completion, has to wait for a core. With fewer workers that the scheduler places
(`--cpu-workers`, b5e9ecff), alternating with the engine's placement:

| `--ram-gib`, workers | Step median / mean | Steps 257-512 median | An expert's reads; a matrix | Waits: file / CPU |
| --- | ---: | ---: | ---: | ---: |
| 39.5, the engine's 13 pinned (B1, B2, B3) | 67.5 / 72.1 ms | 93.6 ms | 3.62; 2.73 ms | 25.2 / 11.9 ms |
| 39.5, 13 placed by the scheduler | 68.0 / 73.3 | 93.5 | 3.60; 2.85 | 25.4 / 12.3 |
| 39.5, 12 | 61.9 / 65.8 | 82.8 | 2.78; 1.76 | 17.2 / 13.2 |
| 39.5, 10 | 61.9 / 65.6 | 82.8 | 2.50; 1.60 | 15.4 / 14.3 |
| 39.5, 6 | 64.9 / 68.5 | 82.9 | 2.47; 1.58 | 17.9 / 15.0 |
| 16, the engine's 13 pinned (B1, B2) | 158.6 / 181.0 | 251.2 | 5.09; 4.40 | 142.5 / 3.8 |
| 16, 12 | 125.2 / 145.9 | 210.9 | 3.85; 3.62 | 107.6 / 3.8 |
| 16, 10 | 121.2 / 140.9 | 210.7 | 3.89; 2.95 | 100.7 / 5.8 |

With two cores free (12 workers) an expert's reads take 2.8 ms and with three 2.5 ms, near what
the drive takes for one alone (2.0 ms), and the step is 8 to 12 % faster at 40 GiB and 16 to 24 %
at 16 GiB; twelve workers keep the CPU's share (13.2 against 11.9 ms; 14.3 with ten).

What the threads waited for a core, from `/proc/<pid>/task/*/schedstat` over 20 s of the decode
at 39.5 GiB (alternating; the I/O threads are the reader of the model file and the one of the
n-gram table's rows):

| Run | `ares-io` (2 threads): waited for a core / ran | Waits per wake-up | The host's threads waited | The workers waited, each |
| --- | ---: | ---: | ---: | ---: |
| The engine's 13 pinned workers (two runs) | 1909 to 2113 / 42 ms | 200 to 212 µs (9 564 to 9 954 wake-ups) | 452 to 654 ms | 19 to 1093 ms |
| 10 workers placed by the scheduler (two runs) | 44 to 136 / 52 to 77 ms | 4 to 14 µs | 125 to 170 ms | 186 to 427 ms |

So in the engine's placement the I/O thread waits a tenth of the time for a core, 0.2 ms each
time it wakes for a completion, where it waits 4 to 14 µs with three cores free (the four runs'
steps: 69.9 and 64.0 ms at the median with the engine's placement, 59.1 and 67.7 with ten
workers; the step times of single runs overlap, the I/O thread's waits do not).

## The read-ahead

- At 40 GiB the next layers' routers name 645 experts a step, 52 % of them picked; 40 of them in
  no tier, of which 11 % are picked, so 0.1 to 0.3 are read ahead (half used). At 16 GiB, 151 in
  no tier, 23 % picked: 2.5 read ahead (1.5 used, 1.0 wasted), 149 left. Reading all 151 would
  read 2.4 times the bytes the step misses, on a step the drive bounds.
- A prefetch waits 170 to 250 ms at the reader (it is kept behind every demand read), so what is
  read ahead lands long after its layer.
- Without the predictions (`--predict 0`, 16 GiB) the step is the same (155.1 ms against 155.9
  to 164.0) and the dense part 1.7 ms shorter (16.3 against 18.0 ms): the two routers a layer
  cost 1.7 ms a step and buy nothing here.

## The share in VRAM follows the text

Steps 257 to 512 (the timed ones) found 45 % of their picks in VRAM against 57 % in steps 1 to
256, missed 261 experts a step against 206, read 19.0 from the file against 8.6 (40 GiB; 100
against 62 at 16 GiB), and took 93.6 ms against 67.5 (251 against 159). A decode of 1024 steps
with the share printed per 64 steps (`--every 64`, 150718de, 40 GiB) shows no steady fall: the
windows find 57 to 77 % in VRAM and take 46 to 67 ms at the median, a window where the text
changes finds 22 % (374 misses and 29 reads of the file a step, 113 ms), and the next windows
are back at 64 to 77 % (the run's mean: 65.9 % in VRAM, 55.3 ms at the median, 16.1 tokens/s;
the 1024 timed steps after it 55.8 %, 70.4 ms). VRAM holds 12.7 % of the experts and gets 45 to
77 % of the picks; between steps 7 to 10 experts a step are promoted, and the link carries
nothing else (every miss is the CPU's). How fast the tiers catch up after a change of the text
decides the slow windows; that is not measured beyond these windows. At 16 GiB (512 steps, the
RAM tier holding a fifth of the experts) the fall lasts: 58 to 70 % in steps 1 to 192, then 36 to
47 % for the 320 steps after (94 to 112 reads of the file a step, 239 to 277 ms).

## UD-Q5_K_XL, and a context of 30 000 tokens

The same runs (B, steps 1 to 256 after the context) on the 147 GiB UD-Q5_K_XL (2400 VRAM slots, 9.8
%; 3744 RAM slots at 16 GiB, 10 032 at 39.5) and on UD-Q4_K_XL after 30 000 tokens of context:

| Run | Step median / mean | In VRAM | Misses from RAM / the file (MB a step) | Waits: CPU / file (layers, ms each) |
| --- | ---: | ---: | ---: | ---: |
| UD-Q5_K_XL, 39.5 GiB | 93.8 / 101.8 ms | 54.3 % | 199.6 / 19.1 (77) | 11.6 / 54.7 ms (13.4, 4.1) |
| UD-Q5_K_XL, 16 GiB | 201.8 / 220.0 ms | 54.3 % | 142.2 / 75.3 (303) | 3.8 / 181.1 ms (32.0, 5.7) |
| UD-Q4_K_XL at 30 000, 39.5 GiB | 70.9 / 73.3 ms | 49.4 % | 233.5 / 9.6 (30) | 13.7 / 28.6 ms (7.7, 3.7) |
| UD-Q4_K_XL at 30 000, 16 GiB | 205.6 / 211.9 ms | 49.6 % | 164.2 / 77.0 (242) | 3.6 / 177.5 ms (33.9, 5.2) |

The larger file reads 2.2 times as many experts from the drive at 40 GiB (2.8 times the bytes),
and its step waits for the file half the time. At 30 000 tokens the attention takes 1.14 ms a step
against 0.92 at 4000, and the step is what its misses make it: a long context costs decode little
on this model.

## Two small cards (the RTX 4070 and the RTX 3060, #24)

The same tool on the development machine (both cards on every layer, the RTX 3060 the mixer;
Swift IQ2_XS, the Flash-Next file this machine has, `--ram-gib 8`, `--every 64`; b5e9ecff with
150718de; i5-13400F, six performance cores with two threads each and four efficiency cores;
three other agents built and ran on the machine and its drive, load average 25 to 48):

| Run | Step median / mean | In VRAM | Misses a step: CPU / uploaded; from the file | Waits: file (a layer) / CPU | n-gram rows a step |
| --- | ---: | ---: | ---: | ---: | ---: |
| The engine's workers (load 34) | 62.0 / 81.0 ms | 96.7 % | 15.1 / 0.6; 2.6 | 32.2 (13.1) / 2.3 ms | 25.9 ms |
| 6 workers (load 48) | 100.7 / 119.3 ms | 96.9 % | 10.9 / 3.9; 2.4 | 62.0 (27.8) / 1.8 ms | 33.7 ms |

- The cards hold nearly every pick of this file (13 584 slots, 55 % of its experts), and the step
  waits for the shared drive: 2.5 reads a step at 13 to 28 ms a layer, and the rows of the n-gram
  table, read from the mapped file, 26 to 34 ms a step. Timed: the mixer card's stream mixers
  31.5 ms and its exchange with the worker 3.3 ms, the worker's experts 32 ms and exchange 55 ms
  (it waits for the mixer).
- A first run at load 58 stopped ("a step stood still for 20 s while rows of the n-gram table
  were read"): the tool's limit, at the drive's worst.
- This CPU has a second thread per core and the workers leave them free, so the I/O thread has
  a place; fewer workers did not help here (the two runs differ by the load more than by
  anything else). UD-Q4_K_XL, the file of the profile above, is not on this machine (the disk
  has 81 GB free).

## Main with the crash fixes (29c277e0)

The same runs with `main` at 29c277e0 (#56 and the sweeps merged) against 83408e52, alternating
(2026-11-27 17:19 to 17:33, load 4 to 10): at 39.5 GiB 69.2 / 73.5 ms against 69.3 / 74.0 (median
/ mean; 56.6 % in VRAM, an expert's reads 3.68 against 3.71 ms); at 16 GiB 154.2 / 177.4 against
160.4 / 200.8. The profile holds for the main of now.

## The counters cost nothing

The wait counters (B against A, alternating, untimed medians): at 40 GiB 68.14 and 68.20 ms
against 68.79 and 67.01 (means 72.56 and 72.57 against 72.30 and 72.28); at 16 GiB 155.9 and
161.4 against 142.4 and 164.0 (the first A run's fill held 59.8 % of the picks in VRAM, the others
57 %).

## Through the server

The server sessions of the comparison with Strata (`--ram 40G`, the bench's German, English and
code questions, 300 tokens each, asked twice; see [`against-strata.md`](against-strata.md), 2026-11-27) count the
same per request in `/metrics`. Per generated token (the request's short prompt included):

| UD-Q4_K_XL, 40G | Tokens/s | Waits a token: all / CPU / file | Reads of the file a token | An expert's reads | Picks in VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP drafts, first asks | 13.0 to 17.3 | 41 to 62 / 8 to 14 / 25 to 43 ms | 12 to 24 | 3.6 to 4.4 ms | 28 to 48 % |
| MTP drafts, asked again | 19.4 to 29.7 | 27 to 35 / 6.5 to 10.6 / 13 to 22 ms | 6 to 13 | 3.8 to 4.7 ms | 44 to 61 % |
| No drafts, first asks | 11.7 to 15.4 | 39 to 61 / 10 to 15 / 22 to 41 ms | 9 to 20 | 3.5 to 3.9 ms | 42 to 60 % |
| No drafts, asked again | 17.1 to 19.2 | 25 to 32 / 10 to 12 / 11 to 18 ms | 5 to 8 | 3.7 to 4.1 ms | 63 to 69 % |
| `--ram 16G`, MTP drafts, first asks | 5.2 to 7.4 | 135 to 195 / 2 to 3 / 115 to 175 ms | 71 to 110 | 5.6 to 8.0 ms | 21 to 44 % |
| `--ram 16G`, MTP drafts, asked again | 7.1 to 11.1 | 93 to 141 / 2 to 4 / 73 to 124 ms | 60 to 77 | 6.3 to 7.5 ms | 42 to 56 % |
| `--ram 16G`, no drafts, first asks | 5.1 to 7.2 | 130 to 186 / 3.5 to 5.4 / 108 to 164 ms | 57 to 89 | 4.5 to 5.0 ms | 42 to 57 % |

- A token waits for misses 60 to 80 % of its time on a first ask, more than the tool's 52 %:
  a new question finds 28 to 60 % of its picks in VRAM, where the tool's steps after a context of
  the same text found 57 %. The reads of the file are again the larger part.
- The server's reads take as long as the tool's (3.5 to 4.7 ms an expert against 2.0 for the
  drive alone): it pins 12 workers and spins the conductor and the device thread, so its I/O
  thread waits for a core as the tool's does.
- With drafts a token costs about the picks it costs without (447 to 495 a token on first asks
  against 488 to 492; 54 to 77 % of the drafts accepted), because each row of a window picks
  experts of its own and every distinct miss is paid. So drafts pay little on first asks (13.0 to
  17.3 against 11.7 to 15.4 tokens/s) and more when the tiers hold the text (19.4 to 29.7 against
  17.1 to 19.2).
- At `--ram 16G` the server waits for the file 80 to 90 % of a token, as the tool does, and its
  reads take 4.5 to 8 ms an expert.

## Levers 1 and 5 (#96, 2026-11-28)

What the branch changes (see "Where the CPU workers run" and "Predicted only where a read may
come of it" in [`experts.md`](../architecture/experts.md)):

- *Lever 1, a core for the I/O threads.* On a CPU with one thread per core the CPU workers
  leave one core more free, an efficiency core where there are any, and the I/O threads of the
  store's reader and of the n-gram table's rows are bound to the cores without a worker. On the
  test machine the tool's workers go from 13 to 12 (cores 1 to 12) and its I/O threads to cores
  0 and 13; the server's from 12 to 11 (cores 2 to 12), its I/O threads to 0, 1 and 13. A CPU
  with a second thread per core keeps its workers: on the development machine the tool's nine
  stay on CPUs 2, 4, 6, 8, 10 and 12 to 15, and the I/O threads (and the ring's `iou-wrk`
  threads, which take their affinity) are bound to 0, 1, 3, 5, 7, 9 and 11.
- *Lever 5, no predictions where no place qualifies.* A step of the decode kernels applies the
  next layers' routers only as far ahead as a place of the read-ahead's odds for its kind of
  step qualifies, and as far as the parts can every eighth step of a kind that would predict
  less, so the counts go on.

*The text is unchanged.* The tool's fingerprint on the two local cards (`--fingerprint`, `--ram-gib
8 --cells 4096 --spread tensor --core-share 0.76 --shared worker --serve stream`, the text of the
profile) is `67fbce287c08c750` for `main` (372763fb) and for the branch (159207c7), the files
equal; the branch's steps named 1653 experts a step against 2236.

*The two local cards* (RTX 4070 and RTX 3060, Swift IQ2_XS, `--ram-gib 8 --cells 8192 --rows 512
--decode-at 4000 --decode-steps 256 --breakdown --serve model --every 64`; A is `main`
372763fb, B the branch; in pairs under one hold of the GPU lock, 2026-11-27 23:35 to 2026-11-28
03:41, other agents on the machine between the pairs):

| Run (load at its start) | Steps 1-256: median / mean | In VRAM | Waits: file (layers) / CPU | An expert's reads | Predicted a step | Dense on the 4070, timed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 (8.5) | 23.46 / 24.77 ms | 96.5 % | 5.0 (2.4) / 1.0 ms | 2.09 ms | 638 | 7.40 ms |
| B1 (5.8) | 23.67 / 24.69 | 96.5 % | 5.0 (2.4) / 1.0 | 1.99 | 80 | 6.41 |
| B2 (10.4) | 25.61 / 27.55 | 95.4 % | 6.5 (2.8) / 2.0 | 2.28 | 81 | 6.42 |
| A2 (10.4) | 23.83 / 25.16 | 96.6 % | 5.5 (2.6) / 1.0 | 2.05 | 638 | 7.40 |
| A3 (11.6) | 24.08 / 25.98 | 96.7 % | 6.1 (2.5) / 1.1 | 2.34 | 636 | 7.41 |
| B3 (7.6) | 23.75 / 24.93 | 96.4 % | 5.1 (2.5) / 1.1 | 1.94 | 81 | 6.41 |
| B4 (3.7) | 23.23 / 24.46 | 96.8 % | 4.8 (2.4) / 0.9 | 1.93 | 81 | 6.42 |
| A4 (3.7) | 24.29 / 25.31 | 96.5 % | 5.3 (2.6) / 1.2 | 1.98 | 638 | 7.40 |

- *No loss on a CPU with a second thread per core.* A's steps take 23.5 to 24.3 ms at the
  median, B's 23.2 to 23.8 and once 25.6. That run (B2) found fewer of its picks in VRAM and read
  the file in more layers: with `--serve model` the CPU's sums round differently from run to
  run, so the runs write different continuations (five among the eight, A3's the same as
  B4's) and pick different experts. Over 8 s of decode (A4 against B3 and B4) the I/O threads waited 12.9 ms for a core
  in A (2.3 µs a wake-up) and none in B; the workers waited 1.9 against 0.6 and 1.7 ms.
- *Lever 5* takes the dense part of the timed steps on the RTX 4070 from 7.40 to 6.41 ms: the
  cards hold 96 % of the picks, two to three experts a step are in no tier, and no place of the
  odds qualifies (nothing is read ahead in A either), so seven steps in eight predict nothing.
- The steps 257 to 512 are not compared: they vary by more than the builds (A1 and B2 took
  35.7 and 41.1 ms at the median with 89 % of their picks in VRAM, the other six 23.7 to 24.7 ms
  with 97 %).

*How to leave the core, on the test machine* (2026-11-27 18:35 to 18:51; one build with four
placements behind a switch of the experiment that is not in the code; `--ram-gib 39.5`, in the
order main's, the rule, workers that yield, the rule without binding the I/O threads, then again;
load 1 to 12 at the starts, the runs' own workers; the I/O threads' waits for a core over 20 s
of decode):

| Placement | Steps 1-256: median / mean | Steps 257-512, median | An expert's reads; a matrix | Waits: file / CPU | The I/O threads waited; a wake-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| main's: 13 workers, pinned | 67.6 / 72.0, 67.9 / 72.5 ms | 91.4, 92.7 ms | 3.48, 3.56; 2.55, 2.72 ms | 25.0, 25.7 / 11.8, 11.9 ms | 2696, 2614 ms; 232, 229 µs |
| the rule: 12 workers, the I/O threads on cores 0 and 13 | 58.2 / 61.1, 60.3 / 63.7 | 75.3, 76.4 | 2.24, 2.28; 1.42, 1.46 | 14.7, 16.6 / 11.5, 12.2 | 32, 19 ms; 1.9, 1.1 µs |
| 13 workers that yield every 256 spins | 61.8 / 65.3, 63.6 / 71.0 | 81.3, 81.7 | 2.64, 3.48; 1.87, 1.84 | 18.7, 24.1 / 11.6, 11.8 | 1107, 377 ms; 69, 44 µs |
| the rule, the I/O threads not bound | 58.2 / 60.9, 59.9 / 63.5 | 74.9, 75.4 | 2.26, 2.29; 1.42, 1.47 | 14.5, 16.6 / 11.5, 12.0 | 40, 45 ms; 2.4, 2.6 µs |

The rule is the fastest. Workers that yield keep about half of the loss: a woken I/O thread still
waits for the worker on its core to reach its yield (44 to 69 µs), and the host thread, which
spins too, does not yield. Binding the I/O threads changes nothing here, where the free core is
there either way; it keeps them off the workers' cores whatever the scheduler does.

*Main before and after, on the test machine* (2026-11-28 14:50 to 15:24; A is `main` 372763fb,
the build of the profile with the counters, B is #96 (5718d615) alone, C the `main` of that day
(06e48088, with #97 and #103 after #96); in the order A, B, C and then C, B, A at each size of
the tier; the runs of "Where a step goes", 256 + 256 steps; load 10 to 14 at the starts, the runs'
own workers; the I/O threads' waits for a core over 20 s of decode):

| Run | Steps 1-256: median / mean | Steps 257-512, median | In VRAM | An expert's reads; a matrix | Waits: file (layers) / CPU | The I/O threads, a wake-up | Dense | Predicted a step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 GiB, A | 160.6 / 178.5, 154.3 / 176.7 ms | 246.9, 251.9 ms | 56.8, 58.1 % | 5.01, 5.11; 4.41, 4.53 ms | 139.9 (28.3), 137.8 (27.3) / 3.8, 4.2 ms | 270, 268 µs | 18.0, 18.0 ms | 645, 641 |
| 16 GiB, B | 121.0 / 140.9, 111.6 / 128.4 | 194.4, 203.1 | 56.7, 59.8 % | 3.67, 3.50; 3.04, 3.42 | 102.3 (28.5), 89.6 (26.5) / 3.9, 4.2 | 2.5, 2.5 µs | 17.2, 17.2 | 468, 467 |
| 16 GiB, C | 111.7 / 126.6, 119.3 / 141.1 | 205.8, 191.0 | 60.3, 56.8 % | 3.42, 3.68; 3.61, 3.09 | 87.8 (26.7), 102.8 (28.5) / 4.3, 3.8 | 2.1, 2.5 µs | 17.2, 17.2 | 468, 468 |
| 39.5 GiB, A | 67.0 / 71.9, 66.6 / 72.0 | 94.0, 90.5 | 56.7, 56.7 % | 3.43, 3.48; 2.66, 2.53 | 24.9 (7.2), 25.1 (7.2) / 12.1, 11.9 | 206, 231 µs | 18.0, 18.0 | 645, 645 |
| 39.5 GiB, B | 60.1 / 62.6, 57.5 / 60.1 | 76.0, 76.5 | 57.4, 59.7 % | 2.27, 2.22; 1.46, 1.42 | 16.1 (7.0), 14.2 (6.3) / 12.1, 11.5 | 2.1, 1.6 µs | 16.5, 16.6 | 115, 80 |
| 39.5 GiB, C | 58.3 / 60.9, 57.1 / 59.6 | 75.2, 78.1 | 59.2, 61.6 % | 2.22, 2.35; 1.43, 1.55 | 14.6 (6.4), 13.5 (5.6) / 11.8, 11.6 | 2.4, 3.2 µs | 17.2, 17.1 | 82, 80 |

- *The step.* At 40 GiB #96 takes a step in 58.8 ms against 66.8 at the median (-12 %; the means
  61.4 against 72.0, -15 %), the later steps in 76.3 against 92.2 (-17 %); at 16 GiB in 116
  against 157 ms (-26 %; the means 135 against 178, -24 %), the later steps in 199 against 249
  (-20 %). That is what the profile measured with fewer workers that the scheduler places
  (-6 to -11 and -35 to -42 ms), about.
- *The reads.* An expert's reads take 2.2 to 2.3 ms at 40 GiB (3.4 to 3.5 before) and 3.5 to
  3.7 ms at 16 GiB (5.0 to 5.1): the I/O threads wait 2 to 3 µs a wake-up for a core, where
  they waited 0.2 to 0.27 ms. The CPU's share takes what it took (11.5 to 12.1 ms of waits at 40
  GiB): the core the workers leave costs them nothing that shows.
- *Lever 5.* At 40 GiB no place qualifies (0.1 to 0.2 experts a step were read ahead before),
  the steps predict 80 to 115 experts where they predicted 645, and the dense part takes 16.5 to
  16.6 ms against 18.0 (-1.4 to -1.5 ms of the 1.7 the routers cost). At 16 GiB places one layer
  ahead qualify (2.4 to 3.7 experts read ahead a step, in both builds): the steps predict one
  layer ahead every step and two every eighth, 468 experts against 645, and the dense part takes
  17.2 against 18.0 ms.
- *The main of that day* decodes as #96 does. Its dense part at 40 GiB is 0.6 ms longer than
  #96's (17.1 against 16.5) with as few predictions; that comes from a change after #96 and is not
  looked into here.

*Through the server* (`--ram 40G`, MTP drafts, the bench's decode questions as in "Through the
server" above, 300 tokens each and asked twice; A, B and C as above, in the order A, B, C and
then C, B, A, 2026-11-28 15:24 to 15:41, load 7 to 14; per generated token from the requests'
`/metrics`):

| Run | Tokens/s, first asks: German / English / code | Asked again | Waits a token, first asks | Reads of the file a token, first asks | An expert's demand read |
| --- | ---: | ---: | ---: | ---: | ---: |
| A, 1 | 12.9 / 16.6 / 13.1 | 21.1 / 24.1 / 20.2 | 61 / 43 / 60 ms | 25 / 13 / 22 | 3.7 to 4.3 ms |
| A, 2 | 13.3 / 17.0 / 14.1 | 21.0 / 26.4 / 21.9 | 56 / 40 / 56 | 23 / 11 / 24 | 3.7 to 4.8 |
| B, 1 | 15.0 / 19.7 / 15.0 | 24.0 / 37.9 / 25.9 | 51 / 35 / 53 | 27 / 13 / 26 | 2.7 to 3.7 |
| B, 2 | 14.8 / 19.2 / 15.3 | 29.5 / 27.3 / 23.6 | 49 / 35 / 51 | 23 / 14 / 25 | 2.8 to 3.8 |
| C, 1 | 15.4 / 17.5 / 14.5 | 22.6 / 27.0 / 22.5 | 51 / 37 / 54 | 25 / 13 / 25 | 2.6 to 3.2 |
| C, 2 | 14.9 / 19.5 / 14.8 | 28.0 / 30.5 / 24.5 | 52 / 36 / 53 | 26 / 14 / 27 | 2.8 to 4.0 |

- On first asks #96 writes 14.9, 19.5 and 15.2 tokens/s against 13.1, 16.8 and 13.6 (+14, +16
  and +11 %): a token waits 49 to 53 ms for misses against 56 to 61 on German and code, 35 against
  40 to 43 on English, with as many reads of the file, and an expert's read takes 2.7 to 3.8 ms
  against 3.7 to 4.8. Asked again, the drafts decide more than the reads (65 to 98 % of them
  accepted) and the runs spread: 23.6 to 37.9 tokens/s against 20.2 to 26.4.
- The main of that day holds 1.2 GiB less of the card after loading (14 604 MiB against 15 784)
  and writes what #96 writes.

## The levers of #16, ranked by this profile

What each would save of the measured steps above (one row a step, the engine as it serves:
72.1 ms at 40 GiB, 181 ms at 16 GiB), the largest first. "Measured" means the saving was seen in
a run; the others are bounds from the parts of the step.

| # | Lever | 40 GiB | 16 GiB | Grounds |
| --- | --- | ---: | ---: | --- |
| 1 | A core for the I/O thread (the CPU workers leave the reader a core) | done (#96): -8.0 ms at the median (-12 %), -10.6 on the mean | done (#96): -41 ms at the median (-26 %), -43 on the mean | measured on #96 against main, with lever 5's -1.4 and -0.8 ms (see "Levers 1 and 5"); before, with 12 and 10 workers the scheduler places: -6.3 and -35 ms on the mean step, -11 and -40 ms on the later steps |
| 2 | The stream mixers at the card's bandwidth (#18) | up to -5.5 ms (8 %) | up to -5.5 ms (3 %) | 7.8 ms for 644 MiB that take 2.3 at 288 GB/s |
| 3 | The CPU's share closer to the memory's bandwidth | -2 to -4 ms (3 to 6 %) | -1 ms | 4.3 experts a layer in 0.31 ms, 39 GB/s from RAM |
| 4 | One read per expert (a file of Ares's own) | -1.5 to -3 ms (2 to 4 %) | -6 to -11 ms (3 to 6 %) | an expert alone takes 2.0 ms in three reads at depth 1; what one read of 3 MB takes is not measured |
| 5 | No predictions where no place qualifies | done (#96): -1.4 to -1.5 ms of the dense part | done (#96): -0.8 ms | measured on #96: at 40 GiB no place qualifies and seven steps in eight predict nothing; at 16 GiB places one layer ahead qualify and the second layer is predicted every eighth step; before, `--predict 0` took 1.7 ms off the dense part |
| 6 | The resident experts' down product while the misses come ("reordered inference", LES, ProMoE) | -1.5 ms (2 %) | -1.5 ms (1 %) | the device computes about 0.1 ms of experts a layer, a third of it the down product; a layer waits 0.31 to 5 ms |
| 7 | VRAM catches up faster after the text changes | not known | not known | the share in VRAM varies from 22 to 77 % by 64 steps and a slow window takes twice the step; at 40 GiB it recovers within 64 to 128 steps, at 16 GiB it stayed at 36 to 47 % for 320 steps; the server's first asks find 28 to 60 % |
| 8 | Reading ahead on the predictions | none | none | of the experts no tier holds, 11 % (40 GiB) and 23 % (16 GiB) of the predicted are picked; the drive bounds the 16 GiB step; a prefetch waits 170 to 250 ms |

- *1 comes first.* It is the largest measured saving and the smallest change, and it serves both
  tiers: at 16 GiB the drive waits for the I/O thread, at 40 GiB the reads are a third of the
  step. The workers spin between the jobs of a step (20 ms before they sleep), so on a CPU without
  a second thread per core every core is busy through the step. Strata's rule on this machine is
  9 workers (the performance cores but one and half of the efficiency cores), the host on core 0
  and its I/O threads on the four cores left. #96 leaves one efficiency core more and binds the
  I/O threads to the cores without a worker (12 workers in the tool, 11 in the server); that took
  the waits of the reads where fewer workers took them.
- *7 is not yet a lever.* The tiers do follow the text; what a change would buy is the speed at
  which they catch up after the text changes. The measurement first: the share per 64 steps after
  a change of the text, with the promotions between steps bounded by the link's time as now and
  without the bound (the link carries nothing else in these steps).
- *6 and 8 are small here.* LES's 1.94 times for reading ahead and reordering is on a phone,
  whose compute per layer is long beside a read; here a read is 20 to 50 times a layer's work on
  the device, and the experts no tier holds are the ones a prediction gets wrong.
