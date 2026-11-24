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
