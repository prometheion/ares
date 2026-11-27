# Experiments: host memory

Dated measurement logs, moved here verbatim on 2026-11-24 from the architecture document (now [`../architecture/`](../architecture/overview.md)) and from [`OPEN.md`](../../OPEN.md). They are not kept current: each names its date, build and machine.

## Host memory, measured (2026-11-15)

**Measured** (2026-11-15; RTX 4070 and RTX 3060, 31.2 GiB of RAM, a machine
that other work shared: its available memory moved by several GiB between
runs and the load average was between 4 and 56, which is next to every
number).

*What the tier is worth* (Flash-Next, three texts of German, English and
code, 96 tokens each, each asked twice; tokens per second of generation and
experts read from the file per step, first ask / second ask). The server
before this change with its default, half of what was available, against the
server after it with `--ram` set to what was available less 6.5 GiB (the
measurements had to leave the machine 3 GiB; without a limit the tier takes
what is available less the headroom, about 6 GiB more than here):

| Both cards | Tier | Experts with a slot | German | English | Code | Reads per step |
| --- | --- | --- | --- | --- | --- | --- |
| before, 13.8 GiB available | 6.8 GiB, 4 704 slots | 73.4 % | 14.8 / 26.5 | 11.6 / 9.7 | 5.9 / 19.3 | 17.0 / 6.1, 8.2 / 7.7, 19.9 / 7.2 |
| after, `--ram 12.26G` | 12.2 GiB, 8 832 slots | 90.2 % | 20.6 / 31.0 | 27.8 / 32.1 | 18.5 / 32.3 | 9.8 / 1.9, 3.3 / 2.1, 7.6 / 1.3 |
| before, 20.0 GiB available | 9.9 GiB, 7 104 slots | 83.2 % | 19.2 / 33.4 | 35.9 / 35.8 | 22.6 / 27.0 | 10.8 / 3.7, 4.0 / 3.1, 9.7 / 1.2 |
| after, `--ram 14.80G` | 14.7 GiB, 10 752 slots | 98.1 % | 33.4 / 39.0 | 36.2 / 49.3 | 29.1 / 47.5 | 9.3 / 2.7, 4.4 / 0.7, 6.9 / 0.3 |

On one card the two servers were only measured with tiers of nearly the same
size (9.1 against 9.6 GiB on the RTX 4070, 25 to 27 % of the experts in RAM
next to 20.5 % in VRAM): 1.4 to 9.1 tokens per second and 39 to 93 reads per
step in both, at a load average of 22 to 28, with no difference that the
spread of the runs would show.

*Giving back and taking again* (`crates/server/tests/real_elastic.rs`: the
engine in the test's process follows this machine's memory less 4.5 GiB that
the test hides, so that it gives memory back while the machine itself keeps
its safe margin; no `--ram`; a helper takes 4 GiB in steps of 64 MiB every 50
ms, writing 2.2 to 2.5 GiB per second, holds them 20 s and frees them; a
generation of 1500 tokens runs meanwhile):

| | Both cards | RTX 4070 alone |
| --- | --- | --- |
| Available before the load; the tier after it | 21.1 GiB; 13.9 GiB | 20.3 GiB; 12.7 GiB |
| Headroom after the load | 1 995 MiB, its most: the load's long steps made a release take 24.5 s | 1 995 MiB (17.5 s) |
| Overall tokens per second, German / English / code, first and second ask | 23.1 and 46.2 / 33.5 and 48.3 / 24.3 and 40.8 | 3.0 and 11.2 / 5.3 and 7.8 / 2.2 and 3.7, while the tier gave 4 GiB to other work on the machine (load average 36 to 47) |
| Reads from the file per step | 10.4 and 0.10 / 4.1 and 0.18 / 8.3 and 1.07 | 45 and 25 / 29 and 18 / 76 and 52 |
| The tier while the helper grew | 13.8 to 10.0 GiB; the first piece gone 11 ms after the helper started, whole GiB after 1.3, 2.5 and 3.8 s | 9.0 to 6.3 GiB; first piece after 3.1 s (a step on one card took seconds under that load), whole GiB after 5.8 and 8.3 s |
| Least memory available to the engine's eyes (headroom 1 995 MiB) | 1.88 GiB | 1.60 GiB |
| Resident memory of the process | 15.4 to 11.1 GiB | 9.2 to 6.4 GiB |
| Generation, pieces per second before / while the helper ran | 42.3 / 43.8 | 4.0 / 2.1 |
| The tier back within 256 MiB after the helper freed its memory | 7.2 s (the wait was 6 s); generation 15 pieces per second meanwhile, a piece is locked at every step's end | 18.2 s (wait 12 s) |

*The dense 27B* on both cards, the server before and after with their
defaults (a limit of 9.7 GiB before, of all the machine has after, 31.2 GiB,
of which the checkpoints lock what they use), the three texts with 160
tokens each, two rounds each: 57 to 59 tokens per second of generation for
the first German ask in all four runs, 66 to 80 for the English one, 95 to
132 for code, and 203 to 257 when a text is asked again, with no difference
between the two servers that the rounds would show. Both took a checkpoint
for every new text and held 150 and then 299 MiB of locked memory for them.

The same two tests on small models check what the kernel counts
(`qwen4exp_runtime`: a made-up model's tier from 14.2 to 1.0 MiB and back,
the process's resident memory 13.0 MiB less; `qwen35_prefix`: the 0.8B's
checkpoints from 303 MiB to none while the engine is idle, resident memory
301 MiB less, the next answers equal to those without a cache, none taken
while memory is short, kept again afterwards).

## Page-locked memory between steps, measured (2026-11-24)

The rule that page-locked memory changes only while no device runs a job
(`fix/far-free`; "When page-locked memory changes" in
[`../architecture/host-memory.md`](../architecture/host-memory.md)), against main
be0023db on the RTX 4070 and RTX 3060, alternating main, the branch, main, the
branch; the load average was 4 to 18 from other work on the machine.

*The deadlock gone.* The requests of `real_cache`'s Flash-Next resume test alone
(IQ2_XS, `--ram 8G --context 20480 --parallel 2 --cache`, K and V of the first
18 432 tokens in VRAM and the rest in RAM; a first and a second turn of about
9.8k tokens, two other conversations, the second turn again), which stalled in
two of three runs before: 10 of 10 runs answered all five, the same answers
each time, and every server stopped on SIGTERM in under a second. The test
binary `real_cache` 5 of 5 (4 of 4 tests each, 256 to 292 s). The bench flow of
the K/V merge check (the model's own context, `--ram 8G`, a short request, a
prompt of about 80k tokens beyond VRAM, then an unrelated short request): no
hang in 5 runs of main and 5 of the branch; the long prompt's first token
after 112.8 to 157.0 s on main (median 125.6) and 115.1 to 128.0 s on the
branch (median 124.6), the short request's after 1.5 to 2.4 s and 1.5 to 3.7 s.

*Giving memory back under the rule* (`real_elastic` with a helper that
takes 4 GiB in steps of 64 MiB, the engine on both cards following this
machine less 4.5 GiB, load average 3 to 7): the tier fell from 17.31 to
16.63 GiB, its first piece gone 4.0 s after the helper started (when what the
engine sees fell under its headroom of 437 MiB), the machine's available
memory never under 5.52 GiB, a release seen to take 100 ms; generation ran at
45.7 pieces per second while the helper held its memory (39.7 before), and the
tier was back within 256 MiB 6.0 s after the helper freed it.

*What the frees between steps cost.* Nothing that the runs show:

| Build | Flash-Next decode, tokens/s (German, English, code; first ask and second) | 8k prompt | 27B decode at the owner's settings | 27B 8k |
| --- | --- | --- | --- | --- |
| main | 18.1 72.9 · 32.9 48.6 · 30.8 52.9 | 12.53 s | 62.4 252.4 · 77.8 247.5 · 107.2 247.3 | 6.99 s |
| branch | 22.1 47.1 · 36.5 51.8 · 29.2 66.6 | 12.38 s | 73.8 251.8 · 79.6 252.2 · 108.0 252.1 | 7.04 s |
| main | 22.3 59.5 · 35.2 53.9 · 31.3 71.6 | 12.47 s | 74.9 249.7 · 77.8 250.1 · 103.0 250.0 | 7.07 s |
| branch | 23.1 52.7 · 36.8 58.9 · 31.2 73.2 | 12.47 s | 73.9 250.3 · 78.6 250.6 · 107.8 250.4 | 7.07 s |

Flash-Next (IQ2_XS, both cards, `--ram 8G`, `--cache`, its MTP block): the
second asks swing with the drafts accepted, in both builds; VRAM 11 701 and
11 661 MiB in all four runs, ready in 38.5 to 41.1 s. The 27B (`--split tensor
--shares 0.625,0.375 --context 262144 --parallel 2 --kv q5 --drafters mtp
--draft-vocab`, `--ram 16G`, `--cache`): VRAM 11 457 and 10 095 MiB in all four.
A free that the rule moves to the end of a step costs that step what it cost
the thread that did it before (the end of a step unlocked the tier's pieces on a
device thread, the conductor does it now as it collects the step; the RAM
thread's frees of checkpoint chunks, which ran beside the steps, wait for the
end of the step they meet).

## Long prompts together and again beyond VRAM, measured (2026-11-25)

The defects a user of Flash-Next reported (one RTX 4080, `--ram 40G` and `52G`, coding-agent
requests of 40k to 50k tokens, often two at once), reproduced on the RTX 4070 and RTX 3060
(IQ2_XS with its MTP block, `--ram 8G --context 262144 --parallel 4`: K and V of the first
18 048 tokens in VRAM, the rest in RAM; `fix/flash-crashes`, the script
`/var/tmp/ares-crashfix/repro.py`), main 1a28fb5c against the branch; load average 18 to 33
from other agents' builds.

| | main | branch |
| --- | --- | --- |
| Two chat requests 8 ms apart (one of 17.6k tokens with 13 tools, one of 3.4k without; also 56k and 6k) | the engine failed at once ("a step of 6112 rows in 2 segments"), the model went down | both answer |
| A prompt of 32 034 tokens, then the same again | the second: 17 920 tokens from the cache, all of them replayed, first token after 107 s | 32 033 from the cache, 33 rows replayed, first token after 6.5 s |
| A prompt of 93 425 tokens, then the same again | the second was read again; its tier's growth before a step was refused by the books ("no room for 62.8 MiB more of the tier of experts") and the engine failed | (2026-11-27, a text of 89 814 tokens) 89 813 from the cache, first token after 16.7 s |

The repro on the branch (b251c819), five runs of both cases with new texts each (two
servers of two runs and one of one): 5 of 5 answered; the short request of the pair had its
first token after 14 to 37 s, the long one after 45 to 92 s; the repeat came from the cache
with all but one token every time (32 021 to 32 035), its first token after 1.4 to 5.0 s.
Main failed the pair again in the same run (05:16, "a step of 17586 rows in 2 segments").
Before the place of a checkpoint was kept free of the far pages (an intermediate build), the
repeat of a second text came from the cache all the same but replayed every row (110 s): the
one place held the first text's anchor and the second found no room for its own.

No loss beside it (the same, alternating main, branch, main, branch, each server from a copy
of the same usage profile of the experts; a prompt of 8266 tokens, one of 8271 with 640 from
the cache, a reply of 256 tokens after each; load average 2.5 to 10):

| Run | Prompt 1 | Prompt 2 | Decode 1 | Decode 2 |
| --- | --- | --- | --- | --- |
| main | 14.93 s | 46.44 s | 6.6 tok/s | 6.9 tok/s |
| branch | 14.34 s | 15.89 s | 11.3 tok/s | 14.1 tok/s |
| main | 13.96 s | 15.69 s | 10.7 tok/s | 14.0 tok/s |
| branch | 14.48 s | 16.51 s | 11.1 tok/s | 9.9 tok/s |

The first run of main was disturbed (all four numbers far from the other three runs); the
other three agree within the spread of one build's own runs.

After the place of the step's checkpoints kept its room (2026-11-26, e2fa3e70, rebased on main
0c4fbb60): the branch's own server test had failed twice on b251c819 ("the checkpoint places of
the step: the RAM limit of 8.00 GiB has no room for 112.6 MiB more of checkpoints", at warm-up
and on a request); it passes now from a cold start with the warm-up and from the profile (its
repeat 38 458 of 38 459 tokens from the cache, 58 rows replayed). The repro again, five
servers of one run each with new texts (load 6 to 30 from other agents' builds): 5 of 5; the
pair's long request had its first token after 40 to 48 s and the short one after 12 to 14 s;
the repeat came from the cache with all but one token (32 021 to 32 057), first token after
1.3 to 7.8 s. Main 0c4fbb60 failed the pair at once ("a step of 17586 rows in 2 segments") and
the model went down. A/B as above, main 0c4fbb60 against the branch:

| Run | Load | Prompt 1 | Prompt 2 | Decode 1 | Decode 2 |
| --- | --- | --- | --- | --- | --- |
| main | 6.7 | 14.07 s | 15.75 s | 11.6 tok/s | 13.9 tok/s |
| branch | 4.4 | 14.73 s | 16.55 s | 9.9 tok/s | 13.9 tok/s |
| main | 3.7 | 14.73 s | 16.88 s | 9.5 tok/s | 13.5 tok/s |
| branch | 6.2 | 14.85 s | 16.73 s | 10.8 tok/s | 13.8 tok/s |

Within main's own spread (14.07 to 14.73 s for the first prompt); with the first A/B, main's
first prompt took 13.96 to 14.93 s and the branch's 14.34 to 14.85 s. Load: weights in 9.1 to
9.6 s and ready after 30.1 to 34.3 s in both builds; the plans of VRAM the same (room for
expert slots 6.59 GiB on device 0, 11.22 GiB on device 1). The 8k prompts stay within the
pages in VRAM, so the far K/V's rules are not met there. A short request (55 tokens, 54 from
the cache) after the repeat of 89 814 tokens took 44 s: the experts it missed came from the
drive at a median of 259 ms a read (47 ms an hour before, load 28 from other builds); after the
repeats of 32k it took 2 to 8 s, as on main.
