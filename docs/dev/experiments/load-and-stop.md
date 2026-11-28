# Loading again, and stopping while a model loads

Measurements for GitHub issues #66 (a model loaded again in the same process uploads its weights
far slower than a first load) and #68 (a stop of Flash-Next while its tiers fill takes 35 s and
keeps its memory).

## The machine (2026-11-27)

The development machine: RTX 4070 + RTX 3060, 31 GiB of RAM, the models on btrfs (`compress=zstd:1`,
the weights themselves stored uncompressed) on one NVMe drive, `read_ahead_kb` 256. Other agents
built and ran on it throughout: load average 7 to 45, the drive's IO pressure ("full", avg10) 2 to
52 %. Transparent huge pages are `always` with `defrag` `always` (a fault of a huge page reclaims
and compacts on the spot), swappiness 150, multi-generational LRU: the page cache of a model file
that no process holds is taken back within seconds when other processes grow, with many GiB free.

## Copies from a mapping without read-around (host, 2026-11-27)

`/var/tmp/ares-reload/py/mapread.py` and `partial.py`: 100 to 521 MiB of the 27B's file, evicted
from the page cache first (`POSIX_FADV_DONTNEED`), copied in pieces of 16 MiB through a mapping or
read with `pread`; IO pressure 22 to 37 % "full".

| Range | Mapping | Mapping, `MADV_RANDOM` (no read-around) | `pread` |
| --- | --- | --- | --- |
| cold, 512 MiB | 566 MiB/s | | 662 MiB/s |
| cold, 102 MiB | 372 MiB/s | 2 MiB/s | 392 MiB/s |
| the token embedding table (521 MiB), 30 % of it evicted in pieces of 1 MiB | 432 MiB/s | | 1812 MiB/s |

The rule of the kernel (`mm/filemap.c` of Linux's master, read 2026-11-27): every fault of a file
mapping that finds no page counts a miss in the open file's `f_ra.mmap_miss`
(`do_sync_mmap_readahead`), every fault that finds one takes one off (`do_async_mmap_readahead`,
`filemap_map_pages`), and past `MMAP_LOTSAMISS` (100) a fault reads no more around it, only its own
page: the regime of the middle column. `filemap_map_pages` takes nothing off for a folio marked
`workingset`, a page read again soon after the kernel took it back: "If there are too many folios
that are recently evicted in a file, they will probably continue to be evicted. In such situation,
read-ahead is only a waste of IO." An ordinary read is read ahead by its own size and counts no
misses.

## The 27B loaded again in the same process (GitHub issue #66)

`/var/tmp/ares-reload/repro66.sh`: Qwen3.8-27B-UD-Q4_K_S on both cards (`--ram 8G --context 32768
--cache`), a first load, one request, `POST /models/unload`, `POST /models/load`, one request,
SIGTERM. The release build of main (5e494226); the model file was cold at the first load.

| Run | First load: weights of device 0 / 1, load | Page cache of the file at the reload | Reload: weights of device 0 / 1, load |
| --- | --- | --- | --- |
| r66-main-1 (16:29, load 18 to 31, IO full 38 to 47 %) | 17.6 / 16.8 s, 25.3 s | 4.2 of 14.3 GiB | 25.9 / 25.0 s, 61.3 s |
| r66-main-gdb (17:46, under gdb, load 44 to 52, IO full 0 to 12 %) | (stacks every 3 s) | 3.0 GiB, gone within 15 s | still loading after 6 min, killed |

In both the first load read the file at 500 to 950 MiB/s. In the reload, device 0 first copied the
token embedding table (521 MiB, `HostTable::pinned`, `read_matrix` from the mapping) at 5 to
50 MiB/s, 26 s in r66-main-1; in r66-main-gdb the weights went on at 16 to 20 MiB/s on both devices
(the stacks: `read_matrix` and the gather of `Source::Columns` copying from the mapping, and the
driver's `cuMemcpyHtoD` from it), the drive idle (IO pressure 0 %) and the file's pages in the page
cache staying at 150 to 330 MiB: every fault read one page, and the kernel took the pages back as
fast as they came. That pace is the 659 s of PR #64 for the 8.47 GiB of device 0. Between the two
loads the kernel had taken back most of the file's pages while the first engine still ran (the
process's RSS fell from 14.7 to 5.5 GiB in five seconds during the measurement of a step's price,
with GiBs free). The pages of the reload were then refaults of those, marked `workingset`: its hits
took no misses off, past a hundred misses its faults read one page each, and they stayed so for the
whole load. A first load, of pages the kernel had not just taken back, keeps its read-around.

### Before and after (2026-11-27, 18:14 to 22:33)

`repro66.sh` again, main (5e494226, A) and the fix (5bce7ae4, B) in turn, the release builds. From
22:15 on the file was evicted before each run (cold first load), and `py/pressure.py` allocated
memory after the first request, while the first engine held the file mapped, until the kernel had
taken the file's pages back down to 1 GiB (`-h`) or 6 GiB (`-k`), as the builds of other agents did
by themselves before. Times are the weights of device 0 / device 1 and the whole load, from the
log; "IO full" is the drive's pressure (avg10) at the reload.

| Run | First load | Reload | Page cache at the reload, IO full | Read during the reload |
| --- | --- | --- | --- | --- |
| A1 (18:16) | 30.3 / 29.7 s, 39.0 s | 632.5 / 626.6 s, 755.4 s | 63 MiB, 7 % | 50 to 57 MiB/s |
| B1 (18:29) | 6.5 / 7.0 s, 14.3 s | 1.6 / 2.9 s, 8.7 s | 14.6 GiB (nothing taken back) | |
| B2-p (18:50) | 10.2 / 9.9 s, 17.3 s | 9.1 / 8.9 s, 15.2 s | 362 MiB | |
| A2-p (18:51) | 8.2 / 8.0 s, 15.4 s | 8.3 / 8.1 s, 14.4 s | 0 MiB | |
| A4-h (22:15) | 16.2 / 15.6 s, 27.9 s | 12.0 / 11.7 s, 18.6 s | 0 MiB | |
| B4-h (22:16) | 20.2 / 19.7 s, 27.5 s | 28.1 / 27.3 s, 34.8 s | 361 MiB, 73 % | the drive's |
| A4-h2 (22:17) | 21.0 / 20.6 s, 30.6 s | not loaded after 205 s (stopped) | 5.7 GiB, falling to 2.7 GiB, 16 to 39 % | 7 to 55 MiB/s |
| B4-h2 (22:22) | 15.5 / 15.1 s, 22.7 s | 45.0 / 43.8 s, 52.7 s | 363 MiB, 19 to 52 % | 290 to 330 MiB/s |
| B5-k (22:28) | 13.0 / 12.7 s, 20.4 s | 10.6 / 10.3 s, 16.8 s | 3.9 GiB | |
| A5-k (22:29) | 9.4 / 9.2 s, 16.6 s | 16.2 / 15.7 s, 24.2 s | 3.2 GiB | |
| B5-k2 (22:30) | 26.7 / 25.7 s, 36.8 s | 11.3 / 11.1 s, 17.6 s | 137 MiB | |
| A5-k2 (22:31) | 9.5 / 9.3 s, 16.8 s | 13.8 / 13.1 s, 20.1 s | 2.5 GiB | |

With r66-main-1 (61.3 s) and r66-main-gdb (not loaded after 6 minutes), main read one page a fault
in four of its eight reloads, at 7 to 57 MiB/s with a drive that was mostly idle; the pressure
before the reload made it happen once in five tries, the other agents' builds three times in
three. The fix never did in six reloads: its slowest, B4-h2, read at 290 to 330 MiB/s, what the
drive gave while others read from it (IO pressure up to 52 %), and its reloads took 0.5 to 2.3
times its first load in the same process (the 2.3 is that drive), against 0.7 to 19 times and more
for main. The first loads of
both read the cold file at 500 to 950 MiB/s when the drive was free; the fix's first loads are
not slower (14.3 to 36.8 s against 15.4 to 39.0 s, both with the drive shared).

## A stop while Flash-Next's tiers fill (GitHub issue #68)

`/var/tmp/ares-reload/repro68.sh`: Flash-Next IQ2_XS on both cards at `--ram 8G --cache` with a
usage profile of its experts there (so a load fills the tiers from it after the weights, 17 to 39 s
of reads), SIGTERM some seconds after the process started, or 20 s into a reload that a request
asked for after `POST /models/unload`. Main (5e494226, A) and the fix (e9483dd3, B) in turn,
2026-11-27 23:08 to 23:15, load average 39 falling to 6, the drive's IO pressure 20 to 74 % "full".
"Stop" is from the signal to the end of the process; every run gave back all the devices' memory
(11.25 and 11.33 GiB) and the RAM (the process at 0.13 to 0.39 GiB after the unload).

| Case | Main | Fix |
| --- | --- | --- |
| first load, SIGTERM at 5 s (the weights) | 23.4 s: the weights, the whole fill (17 s), the unload | 5.0 s: the weights (4.3 s), no fill, the unload |
| first load, SIGTERM at 20 s (the fill) | 10.3 s (the fill ran 19 s) | 0.7 s (cut after 9 s) |
| first load, SIGTERM at 20 s, earlier (18:30) | 13.2 s | 6.0 s (the signal came during the weights) |
| reload asked for by a request, SIGTERM 20 s in | 35.5 s: the fill ran 37 s, then the request was ended | 15.8 s: the weights under IO pressure 74 %, the request answered 503 at the signal, no fill |
| the same, again | | 0.8 s (the fill cut after 10 s) |

The unload itself took 0.6 to 0.9 s in every run. The 35 s of the issue is the fill of a reload
running to its end under a busy drive, and a request that waited for the reload running into the
end of the drain; the fix ends the fill at the signal and answers the waiting request at once.
PR #64's case, 23:21 to 23:26: four long prompts at once failed the engine ("the rows of slot 1
start at cell 1070 and its sequence is at cell 6057"), the model was taken down and a request
loaded it again, and SIGTERM came 20 s into that reload, during its fill. Main stopped in 12.7 s
(the fill ran its 17 s, then served the request that waited, 200) and, under gdb, in 14.4 s; the fix
in 0.7 s (the fill cut after 7 s, the request answered 503 at the signal). All the memory came back
each time.

What PR #64 saw besides (an unload of 35.5 s after which neither VRAM nor RAM came back) did not
come back in any of nine stops of main (18:30 to 23:26; three of them with a request running, two
after an engine failure, two under gdb): every unload took 0.6 to 0.9 s and gave back all. What held the memory
then is not known; a device thread that does not end within `STOP_PATIENCE` (10 s) is left behind
by `GpuExecutor::shutdown` without a word, with all it holds, which would fit both the time and
the memory (`crates/runtime/src/gpu.rs`).
