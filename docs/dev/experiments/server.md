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

## Prompt progress and the log on the step's path (2026-11-27)

`feat/progress-logs` against main (8acd73a4, the same code as 83408e52), on the local cards
(RTX 4070 + RTX 3060): Qwen3.8-27B-UD-Q4_K_S with `--ram 6G --context 16384 --parallel 1`,
the binaries started in turn (A, B, A, B), each with a decode of 256 tokens without drafts
(`draft: false`, greedy, `ignore_eos`) twice and a prompt of 8497 tokens read cold (a new first
line each time, one token out) twice, through `/v1/chat/completions`; the load average 14 to 23
from other agents' builds (`/var/tmp/ares-progress/scripts/runab.sh`, `ab.py`):

| Side (load) | Decode, tokens/s | 8.5k prompt, s by the client |
| --- | --- | --- |
| A main (23) | 40.45, 40.40 | 7.085, 7.125 |
| B branch (14) | 40.23, 40.01 | 7.173, 7.157 |
| A main (15) | 40.06, 40.12 | 7.086, 7.093 |
| B branch (16) | 40.44, 40.30 | 7.078, 7.103 |

Decode 40.26 against 40.24 tokens/s, the prompt 7.10 against 7.13 s (B faster in the second
pair): nothing beyond the noise. What changed on the path: a step whose one part is a prompt is
waited for in slices of a second (the same spin, a park with a time limit), the step's events
carry one progress event per prompt, and a window counts its drafts for its request. A first
round at load 30 to 47 (decode with drafts, whose choice follows the measured prices) was too
noisy to read: prompt A 7.02, 7.09, 7.19, 7.16 s against B 7.47, 7.77, 7.13, 7.11 s. Two sides
of the second round were lost to the drive (the 27B's pages had left the page cache, and its
load read 47 MB/s beside other agents' runs) and are not in the table.

The progress through a pass, on the same cards (2026-11-26, Flash-Next IQ2_XS with `--ram 8G`
and its MTP block of q2, from a cold profile, load 3 to 8): a streamed chat of 8883 tokens read
as one pass got `processed` 0, 555, 1203, 2056, 2796, 3490, 4256, 5017, 5804, 6385, 7053, 7773,
8513, 8882 and 8883, one every 1.1 s, and its log the lines at 3.3, 6.6, 9.9 and 13.2 s (623 to
645 tokens per second); the prompt took 14.4 s. Before the fix of the first look (the pass's
first micro-batch waited for the drive for more than a second, and the conductor stopped
looking) the same request got only 0 and 8883, after 20.4 s.
