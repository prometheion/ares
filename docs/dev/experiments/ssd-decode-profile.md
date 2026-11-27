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
decides the slow windows; that is not measured beyond these windows.

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
| `--ram 16G`, no drafts, first asks | 5.1 to 7.2 | 130 to 186 / 4 to 5 / 108 to 164 ms | 57 to 89 | 4.5 to 5.0 ms | 42 to 57 % |

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
- At `--ram 16G` the server waits for the file 75 to 90 % of a token, as the tool does, and its
  reads take 4.5 to 8 ms an expert.

## The levers of #16, ranked by this profile

What each would save of the measured steps above (one row a step, the engine as it serves:
72.1 ms at 40 GiB, 181 ms at 16 GiB), the largest first. "Measured" means the saving was seen in
a run; the others are bounds from the parts of the step.

| # | Lever | 40 GiB | 16 GiB | Grounds |
| --- | --- | ---: | ---: | --- |
| 1 | A core for the I/O thread (the CPU workers leave the reader a core) | -6 to -11 ms (9 to 12 %) | -35 to -42 ms (16 to 23 %) | measured with 12 and 10 workers the scheduler places: -6.3 and -35 ms on the mean step, -11 and -40 ms on the later steps; the file waits -8 to -10 and -35 to -42 ms, the CPU's waits +1.3 ms with 12 |
| 2 | The stream mixers at the card's bandwidth (#18) | up to -5.5 ms (8 %) | up to -5.5 ms (3 %) | 7.8 ms for 644 MiB that take 2.3 at 288 GB/s |
| 3 | The CPU's share closer to the memory's bandwidth | -2 to -4 ms (3 to 6 %) | -1 ms | 4.3 experts a layer in 0.31 ms, 39 GB/s from RAM |
| 4 | One read per expert (a file of Ares's own) | -1.5 to -3 ms (2 to 4 %) | -6 to -11 ms (3 to 6 %) | an expert alone takes 2.0 ms in three reads at depth 1; what one read of 3 MB takes is not measured |
| 5 | No predictions where no place qualifies | -1.7 ms (2.4 %) | -1.7 ms (1 %) | measured: `--predict 0` takes 1.7 ms off the dense part and reads ahead nothing less that is used |
| 6 | The resident experts' down product while the misses come ("reordered inference", LES, ProMoE) | -1.5 ms (2 %) | -1.5 ms (1 %) | the device computes about 0.1 ms of experts a layer, a third of it the down product; a layer waits 0.31 to 5 ms |
| 7 | VRAM catches up faster after the text changes | not known | not known | the share in VRAM varies from 22 to 77 % by 64 steps of a decode and a slow window takes twice the step; the server's first asks find 28 to 60 %; the tiers do recover within 64 to 128 steps |
| 8 | Reading ahead on the predictions | none | none | of the experts no tier holds, 11 % (40 GiB) and 23 % (16 GiB) of the predicted are picked; the drive bounds the 16 GiB step; a prefetch waits 170 to 250 ms |

- *1 comes first.* It is the largest measured saving and the smallest change, and it serves both
  tiers: at 16 GiB the drive waits for the I/O thread, at 40 GiB the reads are a third of the
  step. The workers spin between the jobs of a step (20 ms before they sleep), so on a CPU without
  a second thread per core every core is busy through the step. Strata's rule on this machine is
  9 workers (the performance cores but one and half of the efficiency cores), the host on core 0
  and its I/O threads on the four cores left.
- *7 is not yet a lever.* The tiers do follow the text; what a change would buy is the speed at
  which they catch up after the text changes. The measurement first: the share per 64 steps after
  a change of the text, with the promotions between steps bounded by the link's time as now and
  without the bound (the link carries nothing else in these steps).
- *6 and 8 are small here.* LES's 1.94 times for reading ahead and reordering is on a phone,
  whose compute per layer is long beside a read; here a read is 20 to 50 times a layer's work on
  the device, and the experts no tier holds are the ones a prediction gets wrong.
