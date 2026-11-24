# Experiments: host memory

Dated measurement logs, moved here verbatim from `ARCHITECTURE.md` and `OPEN.md` on 2026-11-24. They are not kept current: each names its date, build and machine.

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
