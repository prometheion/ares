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
