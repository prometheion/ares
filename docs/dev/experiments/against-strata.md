# Experiments: Flash-Next against Strata

Dated measurement logs, moved here verbatim on 2026-11-24 from the architecture document (now [`../architecture/`](../architecture/overview.md)) and from [`OPEN.md`](../../OPEN.md). They are not kept current: each names its date, build and machine.

## UD-Q4_K_XL from the SSD on one 16 GB card (2026-11-22)

**UD-Q4_K_XL from the SSD on one 16 GB card** (2026-11-22, branch `flash/ssd-stream`; the test
machine: RTX 4060 Ti 16 GB on PCIe 4.0 x4, 62 GiB RAM, Samsung 990 1 TB under btrfs; the 104 GiB
file with `--ram 40G` and Strata 0.2.15 (engine 0.1.60) with `--resident-budget-gib 40`, both with
the MTP block of q2; Ares `--context 40960`, started from the profile its warm-up wrote; the bench
of the Strata comparison; every session with the model's pages dropped from the page cache first;
load average 4 to 11 from other agents' work). Tokens per second by the client:

| | Strata 0.1.60 | Ares (verify/stack) |
| --- | --- | --- |
| Decode German 1st / 2nd ask | 9.0 / 10.9 | 12.9 / 21.9 |
| Decode English | 9.7 / 13.0 | 16.5 / 26.8 |
| Decode code | 11.6 / 18.7 | 11.1 / 17.5 |
| First token, 1st / 2nd ask (prompts of 51-69 tokens) | 4.2-7.4 / 0.5-0.8 s | 3.2-4.0 / 3.2-3.3 s |
| Prompt 2k / 8k / 32k | 80 / 305 / 588 | 113-122 / 263-269 / 332-337 |
| Agent turns 1-3, first round (first token) | 24.1 / 23.1 / 5.8 s | 20.2 / 35.7 / 11.4 s |
| Agent turns, second round | 1.5 / 0.4 / 3.1 s | 18.6 / 29.8 / 9.0 s |
| Ready after | 84 s | 35 s |
| MemAvailable while loaded | 12.5 GiB (12.2 GiB of the model's pages cached) | 20.2 GiB |

- *Decode* waits for the misses: 2 to 7 % of the picks come from the file (17 to 25 reads a token
  at 3.3 to 4.6 ms, 27 to 90 MB a token; an expert alone takes 1.5 ms on this drive), 45 % from
  RAM (the CPU and the link), and the layers waited 64 of the 77 ms of a token on German prose.
- *Prompts are bound by the link.* Each wide step uploads every expert it picks that VRAM does
  not hold (the loan leaves 20 to 40 a layer), from RAM or staging: the 32k prompt uploaded 428
  GB in 96 s (4.46 GB/s; the link gives 6.4 for such copies, `upload_bench`) and read 170 GB
  from the drive beside it. Steps of 4096 rows make 8 passes; Strata reads it in 2 chunks.
  The guesses of a lone prompt's next layer as demand reads, their odds, and a loan that leaves a
  layer only a row's picks changed nothing here (13fab11: 95.3 s against 96.4 and 96.7): with the
  MTP block's resident experts the card lends 4096 rows either way, and the drive was not the
  bound. Uploading the guesses RAM holds into free transit slots (4dd63a2) aims at the link's
  idle third; not measured on this machine yet.
- *The second ask* replays the prompt: checkpoints stand on page boundaries (128 cells) and on
  message boundaries 1024 cells apart, and every replayed slice is a wide step that touches most
  experts (a 69-token prompt 3.2 s, 111 rows of an agent turn 18 s). Strata resumes from its
  newest path. See OPEN.md, "Experts from the SSD".
- Strata 0.2.19 (engine 0.1.65; its notes: decode +47 to 59 %, prefill +42 % on this quant on
  RTX 40) is owed a measurement.

## Strata's versions on the test machine (2026-11-22 and 2026-11-23)

Strata moves fast (0.2.19 / engine 0.1.65 on 2026-11-22, 0.2.20 / 0.1.66 on 2026-11-23): 0.1.66
names fused expert kernels for all 13 Unsloth type pairs (MoE decode 1.04 to 1.67 times), INT8
tensor-core dense formats (DFlash2 verify and chunked prefill 1.8 to 3.3 times), faster AVX2 CPU
kernels for cold experts (Q8_0 +55 %, IQ4_XS +33 %, Q4_K +27 %), host K/V offload from 32k on
mid-size cards, a GPU+SSD-only mode, and a fix for NaN outputs on prompts of 24k to 45k. The test
machine's Strata is updated to the newest engine before every comparison; numbers against older
engines below are history.
On 2026-11-23 engine 0.1.66 failed Strata's own test start on the test machine ("verify: layer 0
never rang (operation not supported on global/shared address space)", its CUDA graph capture);
Strata kept 0.1.65, which the comparisons use until a newer engine starts there.

## UD-Q4_K_XL from the SSD against Strata (2026-11-22)

**UD-Q4_K_XL from the SSD against Strata** (2026-11-22, branch `flash/ssd-stream`; the test
machine, 104 GiB file, `--ram 40G` against `--resident-budget-gib 40`; notes
`/var/tmp/ares-flash-ssd/notes.md`). Against Strata 0.2.15 (engine 0.1.60) with drafts: decode
ahead (de 12.9/21.9, en 16.5/26.8, code 11.1/17.5 tokens/s first/second ask against 9.0/10.9,
9.7/13.0, 11.6/18.7), a 2k prompt ahead (17.3 s against 25.4), 8k 12 % behind, 32k 43 %
behind (334 against 588 tokens/s), the second ask of a short prompt 3.2 s against 0.5-0.8.
Strata 0.2.19 (engine 0.1.65, decode +47-59 %, prefill +42 % on this quant by its notes) is
owed a measurement. Found:

## UD-Q4_K_XL and UD-Q5_K_XL against Strata 0.2.22 (2026-11-27)

**Setup.** The test machine (see [`ssd-decode-profile.md`](ssd-decode-profile.md)), nothing else
running on it. Ares: `main` at 83408e52 with the wait counters of `measure/ssd-moe` (fc9e2a26,
otherwise the same code; they cost nothing, see the profile), `--ram 40G`, a fresh `--cache`
directory and no usage profile (the warm-up fills the tiers), its defaults otherwise (the model's
own context of 262 144 cells, K and V beyond VRAM in RAM), the MTP block of q2
(`--drafters mtp-q2_0.gguf`) or `--drafters none`. Strata 0.2.22 (engine and control 0.1.69,
updated with its own updater the same morning) from a copy of the owner's run config with the
pack, the file and `--resident-budget-gib 40`, no picture reader, its MTP drafts (`--spec 4`) or
`"drafting": "off"`, its shipped expert profile and the owner's switches (`STRATA_HC_Q8_TC=0`,
`STRATA_HC_Q8_BF16TC=0`, `STRATA_NV_CHUNK=1`). Strata's packs: UD-Q4_K_XL's of 2026-11-22,
UD-Q5_K_XL's made the same way (`tools/iq_pack.py --compat-bf16`). UD-Q5_K_XL (147.4 GiB)
was downloaded from `unsloth/Qwen3.8-Flash-Next-GGUF` at f5d175de, every part's sha256 checked.
Every session starts with the model's pages dropped from the page cache. The bench of the
earlier comparisons: German, English and code questions of 54 to 72 tokens, 300 tokens greedy,
each asked twice; fresh prompts of about 2k, 8k and 32k tokens, 8 tokens out; a conversation of
three turns with four tools, then the same conversation again. The sessions alternate (Ares,
Strata, Ares, Strata with drafts, then one each without); load average 6.5 to 13 at their starts,
from the engines' own CPU workers.

Tokens per second by the client; prompts in prompt tokens over the time to the first token;
agent turns as the time to the first token:

| UD-Q4_K_XL, 40 GiB | Ares, MTP (2 sessions) | Strata, MTP (2) | Ares, no drafts | Strata, no drafts |
| --- | --- | --- | --- | --- |
| Decode German, 1st / 2nd ask | 13.0-13.3 / 19.4-25.3 | 4.6-5.1 / 7.5-9.7 | 12.1 / 17.1 | 4.9 / 7.1 |
| Decode English | 14.7-17.3 / 23.0-23.3 | 6.6-7.4 / 10.5-11.9 | 15.4 / 19.2 | 5.4 / 6.7 |
| Decode code | 13.0-14.2 / 21.1-29.7 | 6.8-8.5 / 10.9-13.3 | 11.7 / 17.1 | 6.2 / 6.3 |
| Drafts accepted | 1620 of 2290 | 1888 of 3291 | | |
| First token of a question, 1st / 2nd ask | 1.7-3.5 / 1.4-1.9 s | 6.4-8.6 / 0.8-1.4 s | 1.5-3.7 / 1.1-1.6 s | 6.7-8.4 / 0.5-1.3 s |
| Prompt 2k / 8k / 32k (tokens/s) | 175-182 / 486-490 / 836-837 | 86-87 / 255-265 / 516-527 | | |
| Agent turns 1-3, first round | 12.1 / 15.8 / 4.0 s (a session of its own) | 21.4-21.5 / 29.1-31.7 / 5.2-9.8 s; 26.8 / 33.8 / 5.6 s alone | | |
| Agent turns 1-3, second round (cached) | 16.7 / 30.9 / 7.8 s | 3.5-5.0 / 1.1-1.6 / 1.3-2.3 s; 1.5 / 0.35 / 1.7 s alone | | |
| Ready after | 51-53 s | 102-107 s | 48 s | 107 s |
| MemAvailable while loaded | 20.1 GiB | 12.0-12.4 GiB | 20.1 GiB | 12.0 GiB |
| The model's pages cached at the end | 5.7 GiB | 10.8-11.9 GiB | 5.4 GiB | 13.1 GiB |

| UD-Q5_K_XL, 40 GiB | Ares, MTP (2 sessions) | Strata, MTP (2) | Ares, no drafts | Strata, no drafts |
| --- | --- | --- | --- | --- |
| Decode German, 1st / 2nd ask | 9.7-9.8 / 14.4-14.8 | 1.9 / 2.2 | 8.5 / 12.3 | 2.0 / 2.2 |
| Decode English | 11.8-12.1 / 15.0-15.2 | 2.6 / 2.7 | 10.9 / 13.8 | 2.4 / 2.5 |
| Decode code | 7.9-8.5 / 13.5-14.8 | 2.3-2.4 / 2.4-2.6 | 7.8 / 12.1 | 2.1 / 2.1 |
| Drafts accepted | 1491 of 2340 | 1865 of 3230 | | |
| First token of a question, 1st / 2nd ask | 3.8-5.4 / 3.6-4.1 s | 12.0-33.9 / 4.4-7.5 s | 4.0-5.4 / 3.7-3.8 s | 13.4-31.5 / 4.4-6.5 s |
| Prompt 2k / 8k / 32k (tokens/s) | 121 / 355-359 / 763-766 | 53-55 / 157-161 / 341-349 | | |
| Agent turns 1-3, first round | 17.3 / 21.6 / 6.0 s (a session of its own) | 40.1-42.5 / 47.6-48.4 / 33.2-34.2 s | | |
| Agent turns 1-3, second round (cached) | 26.2 / 45.3 / 12.6 s | 9.6-9.9 / 2.9-3.3 / 8.4 s | | |
| Ready after | 51-57 s | 96-99 s | 55 s | 96 s |
| MemAvailable while loaded | 20.0 GiB | 11.6 GiB | | |
| The model's pages cached at the end | 5.2-5.7 GiB | 10.7 GiB | | |

*Again with `main` at 29c277e0* (#56, the crash fixes, and the sweeps merged; the same day 17:32
to 18:04, alternating Ares, Strata, Ares, then UD-Q5_K_XL; load 8.7 to 13.9):

| 40 GiB, MTP drafts | Ares 29c277e0, UD-Q4_K_XL (2 sessions) | Strata 0.1.69, UD-Q4_K_XL | Ares 29c277e0, UD-Q5_K_XL |
| --- | --- | --- | --- |
| Decode German, 1st / 2nd ask | 12.8-13.0 / 19.3-21.8 | 4.8 / 7.1 | 9.1 / 14.3 |
| Decode English | 16.5 / 21.9-22.4 | 6.8 / 9.6 | 11.8 / 15.0 |
| Decode code | 12.3-12.5 / 18.4-20.8 | 7.0 / 10.0 | 8.4 / 13.0 |
| Prompt 2k / 8k / 32k (tokens/s) | 169-173 / 479-481 / 830-834 | 85 / 251 / 511 | 124 / 369 / 764 |
| Agent turns 1-3, first round | 13.1-13.4 / 16.0-16.1 / 6.0-6.5 s | 21.4 / 28.2 / 9.0 s | 18.8 / 21.9 / 8.4 s |
| Agent turns 1-3, second round | 20.1-20.2 / 33.1-33.2 / 10.8-11.1 s | 3.9 / 1.0 / 2.0 s | 28.8 / 46.7 / 14.5 s |
| Ready after | 53-54 s | 108 s | 58 s |

No request failed: the full suite ran in every session, the agent's turns after the 32k prompt
included. Decode and prompts are those of 83408e52 within the spread.

Found:

- *Decode: Ares leads.* On UD-Q4_K_XL 1.5 to 3.4 times Strata's tokens/s with drafts (13.0 to
  17.3 first asks and 19.4 to 29.7 asked again, against 4.6 to 8.5 and 7.5 to 13.3), 1.9 to 2.9
  times without; on UD-Q5_K_XL 4 to 7 times (Strata decodes it at 1.9 to 2.7 tokens/s). Strata
  0.1.69 decodes UD-Q4_K_XL slower here than 0.1.60 did on 2026-11-22 (9.0 to 11.6 on first asks);
  without the owner's switches (`STRATA_HC_Q8_TC`, `STRATA_HC_Q8_BF16TC`, `STRATA_NV_CHUNK`) one
  session gave the same (5.1 to 7.1 and 7.0 to 11.9). With `--ram 16G` Ares decodes UD-Q4_K_XL at
  5.2 to 7.4 and 7.1 to 11.1 tokens/s with drafts, 5.1 to 7.2 and 7.5 to 8.5 without (one session
  each). Where its time goes: [`ssd-decode-profile.md`](ssd-decode-profile.md).
- *Prompts: Ares leads now.* 1.6 to 2.3 times Strata's rate at 2k, 8k and 32k on both files (the
  passes layer by layer of 2026-11-23); before them Ares read prompts 1.4 to 4 times slower
  (UD-Q3_K_XL, 2026-11-23).
- *A short question's first token.* Ares 1.5 to 3.7 s on UD-Q4_K_XL, Strata 6.4 to 8.6 s; asked
  again, Ares 1.1 to 1.9 s against Strata's 0.5 to 1.4 s: Ares computes again a prompt shorter
  than a page of 128 cells (`cached` 3 tokens), Strata reuses all but 7 tokens of it.
- *Cached agent turns stay Ares's weak point.* Asked again, Strata answers turns 2 and 3 in 0.35
  to 2.3 s on UD-Q4_K_XL; Ares takes 7.8 to 30.9 s (12.6 to 45.3 s on UD-Q5_K_XL), longer than
  its own first round (4.0 to 15.8 s). The cache holds the turns' pages but not their checkpoints:
  every turn saved its checkpoints and the cache dropped as many (`ares_checkpoints_saved_total`
  and `_dropped_total`, 1 to 3 a turn), so in the second round turns 1 and 2 restored none and
  replayed every cached row (`ares_replayed_rows_total`: 512 and 6784 rows; turn 3 restored the
  checkpoint the turn before had left and replayed 140 to 164). In the turn of 30.9 s the store
  read 67 GB of experts for that replay, 8200 of them as the guesses of whole layers that a pass
  reads ahead. With `--ram 40G` the tier of experts takes the limit and the prompt cache keeps its
  floor of one checkpoint place, which the newest turn takes; the `--cache` directory did not keep
  the others either. The same on `main` at 29c277e0 (turn 2 asked again: 6837 rows replayed, 33.2
  s).
- *83408e52 fails a request after the 32k prompt* (29c277e0 does not). Both UD-Q4_K_XL sessions, at the agent's first
  turn: "the RAM limit of 39.67 GiB has no room for 61.8 MiB more of the tier of experts: the step
  record and read-back hold 47.4 MiB, the K/V beyond VRAM 127.5 MiB, the tier of experts 39.23
  GiB, the prompt cache 225.3 MiB"; both UD-Q5_K_XL sessions at the agent's third turn: "the RAM
  limit has no room for 649.1 KiB more of K and V beyond VRAM: the tier of experts is at its floor
  and the prompt cache holds the rest". The engine is taken down and loads again on the next
  request (30 to 34 s). The agent rows above are therefore from sessions of their own (warm-up and
  the agent only).
- *Memory.* Strata uses the page cache beyond its budget: 11.6 to 12.4 GiB available while it was
  loaded, 10.7 to 13.1 GiB of the model's pages cached at the end; Ares leaves 20 GiB available
  (its reads are direct; the 5.2 to 5.7 GiB cached are the load's).
- *Cores.* Strata takes 9 workers on this CPU (the performance cores but one and half of the
  efficiency cores) on cores 1 to 9, its host thread on core 0, and its eight I/O threads beside
  the rest on cores 10 to 13; Ares pins 12 workers to every core but the conductor's and the
  device thread's, and its reads wait for a core (the profile's "The drive, and the cores the
  reads wait for").
- *Ready after* 48 to 57 s (Ares) against 96 to 108 s (Strata).
