# Experiments: server

Dated measurement logs, moved here verbatim on 2026-11-24 from the architecture document (now [`../architecture/`](../architecture/overview.md)) and from [`OPEN.md`](../../OPEN.md). They are not kept current: each names its date, build and machine.

## Priorities of requests, measured (2026-11-22)

Measured 2026-11-22 on the test machine (RTX 4060 Ti 16 GB) with Qwen3.5-0.8B
Q4_K_M, through `/v1/completions` (greedy, `ignore_eos`), runs of priority 0
and of priority 5 alternated (0, 5 then 5, 0) four times each, machine load
4 to 14 from other agents' builds (`/var/tmp/ares-prio/bench_prio.py`):

| The engine is busy with | A new request of priority 0 | of priority 5 (the same as the others) |
| --- | --- | --- |
| both places (`--parallel 2`) taken by generations of 3000 tokens; a short prompt | first token 17 to 24 ms (median 21): one of them gives way | 373 to 9899 ms (median 382): it waits for a place |
| a generation and a prompt of 12 000 words (`--parallel 2`); a prompt of 16 000 words | 1102 to 1168 ms (median 1164) | 1983 to 11 124 ms (median 7340) |
| the same with `--parallel 4` (a place is free) | 1068 to 1233 ms (median 1105) | 3439 to 3806 ms (median 3543) |

Qwen3.8-Flash-Next IQ2_XS (`--parallel 2 --ram 24G`, two runs of each, load 4
to 6): next to two generations of 150 tokens a short request of priority 0 has
its first token after 479 and 966 ms, one of priority 5 after 4102 and 5149 ms;
next to a generation and a prompt of 500 words, a prompt of 700 words after
1467 and 3096 ms against 4146 and 6063 ms.

With four places nothing has to give way: the gain is the order of the prompts
in a piece (the longer prompt of priority 0 goes before the shorter one) and
the debt it does not wait for. What it costs the request of priority 5 that was
there first: its first token came 3372 ms after the new prompt arrived against
1435 ms when the new one had its priority. The long first tokens of priority 5
with two places are the time until one of the generations ends (one ran
3000 tokens in under a second where the greedy text loops and the n-gram
drafts take it).

## The restart in place, measured (2026-11-26 and 27)

On the local cards (RTX 4070 + RTX 3060), `fix/server-sweep` (debug builds of bc6f3eac and
653e45e1), a step held by `ARES_TEST_HOLD_STEPS_WHILE` after a first request was served
(`/var/tmp/ares-fixserver/repro/restart-gpu.sh` and `restart-gpu-fn.sh`). The free memory is what
the new image's load found; "right after the exec" is taken a moment after the line that says it
restarts (the old image's memory is going then).

| | the 0.8B, `--watchdog 5` (load 6 to 10) | Flash-Next IQ2_XS, `--ram 8G --watchdog 60` (load 40) |
| --- | --- | --- |
| VRAM of the process before | 2470 MiB (device 0) | 11 662 and 11 696 MiB |
| `VmRSS` and threads before | 1.16 GB, 26 | 9.65 GB (the tier of 8 GiB in page-locked RAM), 65 |
| right after the exec | no VRAM of the process, device 0 at 18 MiB; 181 MB, 21 threads | the new image starting: 20 kB, 1 thread, the VRAM still being given back |
| free when the new image loaded | 11.52 of 11.68 GiB (device 0), as at the first load | 11.52 and 11.57 GiB, as at the first load |
| the 503 of the held request | 7 s after it came | 60 s of the watchdog and its look |
| ERROR line to exec | 0.5 ms | 6 ms |
| exec to the new image's first line | | 1.5 s (the old image's memory torn down) |
| exec to listening again | 1.8 s | 70 s (the load, the tier filled without a profile) |
| afterwards | 200, same PID and port; at its stop 2.26 of 2.26 GiB released | 200, same PID and port; at its stop 11.25 of 11.25 GiB released |

`VmLck` and `VmPin` stay 0 throughout: the driver's page-locked memory does not show in them, so
`VmRSS` is the measure. On the host (`tests/binary.rs`, the `ares-host` model, `--watchdog 2`): the
stuck step was found 3 s after it was held, the exec came 40 ms after the ERROR line, and the new
image listened 140 ms later. The first Flash-Next run had `--watchdog 10`: Flash-Next's own first
step after the load, the step prices with experts from the drive, outlasted it, and the server
(still waiting for its one model) exited with 75, as it does since 39274398.
