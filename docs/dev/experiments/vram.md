# Experiments: VRAM cap, reserve and watch

Dated measurements of what the engine takes of the devices under `--vram`, of the
reserve, and of the watch that gives the expert slots back (GitHub issue #36; how it
works: "VRAM" in [`../architecture/devices.md`](../architecture/devices.md)). They are
not kept current: each names its date, build and machine.

## Under a cap and beside a growing program (2026-11-28)

**Machine.** RTX 4070 (device 0, 12 282 MiB by NVML, 11.68 GiB for CUDA) and RTX 3060
(device 1, 12 288 MiB) on the patched driver 615 with peer access, 31 GiB of RAM; neither
card drives a display (all eight DRM connectors disconnected, NVML's display inactive).
Other agents built and ran on the machine meanwhile: the load average was 12 to 29 of 16
threads, which is next to every speed below. The server is the branch's `ares` (dev
profile, optimized), driven by one request of a long prompt (the repository's documents)
and a greedy decode, and a second short one; `sample.py` read the server's memory on each
card from NVML every 50 ms (the number `nvidia-smi` shows for the process, which
`nvidia-smi --query-compute-apps` sampled every 200 ms beside it and found the same peaks).

**A load beside another program** (`q27-reserve`, 2026-11-28 10:14). A helper held 512 MiB
on device 1 (616 MiB with its context) when the 27B loaded on both cards without a cap: device
1 kept a reserve of 616 MiB ("other processes hold 616.0 MiB there"), a plan could take
10.36 GiB of its 10.97 GiB free, and the context was 165 888 tokens where it is 190 464 without
the helper. The server's peak on device 1 was 10 018 MiB.

**Plans and peaks under a cap.** The peak is the most the process held on the card at any
sample, from the load to the end of the second request.

| Model, options | Cap | A plan may take | Plan | Prompt, decode | Peak device 0 / 1 |
| --- | --- | --- | --- | --- | --- |
| Flash-Next IQ2_XS, both cards, `--ram 8G` | 9 GiB (9216 MiB) | 8.85 / 8.90 GiB (the contexts hold 154 / 104 MiB) | expert slots 3.87 GiB (2880) / 8.38 GiB (6240), K and V of 11 264 tokens in VRAM, the rest of 262 144 in RAM | 21 255 tokens at 496 per s, 300 at 6.3 per s | 8966 / 8952 MiB |
| 27B UD-Q4_K_S, both cards, `--ram 8G` | 10 GiB (10240 MiB) | 9.85 / 9.90 GiB | tensor split 0.583 / 0.417, heads 0.197 / 0.803, context 92 160 tokens, the MTP block | 21 255 tokens at 1036 per s, 400 at 53.9 per s | 9864 / 9202 MiB |

Without a cap the same Flash-Next load holds 11 662 / 11 696 MiB.

**The watch, the first version** (`fn-watch`, the binary before 48521a83). Flash-Next
without a cap, a helper (`helper.py`) on device 1 that asks for 256 MiB every half second
up to 3 GiB, retrying a refusal, holds them 20 s and frees them; a prompt of 1014 tokens and
600 tokens of decode began 4 s after it. The helper's first two asks were refused (the card
was full: 241 MiB free), the next twelve granted: the tier gave 1.0 GiB back on the watch's
thread while the model was idle and 1.9 GiB while it read its prompt, 11 696 to 8752 MiB.
Then it gave nothing for 21 s of decoding while the card had 4 MiB free: the end of a step
gave only when nothing at all touched the store, and the reads of experts ahead from the
file never stop while a model decodes. That is what 48521a83 changed (a read takes no slot
before it has landed). After the helper freed its memory the tier took pieces again from
6.5 s on, one before each step, back to 11 056 MiB in 19 s (the headroom now held the growth
it had seen).

**Where nothing can give way** (`q27-full`, 10:11). The 27B on both cards without a cap, and
the helper asking for 1 GiB on device 1 in steps of 256 MiB: it got 768 MiB, what the card had
free, and 238 asks were refused over two minutes; the watch wrote once "device 1 (NVIDIA
GeForce RTX 3060): other programs need 406.8 MiB more VRAM than is free (462.4 MiB free,
869.3 MiB kept for them), and nothing more of the engine's gives way", and both requests were
served (300 tokens of decode at 115 per s).

**The watch while decoding, and an order that was missing** (`fn-watch2`, `rep2`; 09:35 to
10:31, the binaries of 48521a83 and d56b4505). With the helper starting during a decode of
600 to 900 tokens, the tier gave 64 MiB to 768 MiB at the end of most steps, 11 696 to
7 344 MiB on device 1, and decoding went on at 29 tokens per second (`repro1`, `rep2` round
2). But of five such runs one ended with `CUDA_ERROR_ILLEGAL_ADDRESS` on the devices
(`rep2` round 1, after 23 gives) and two stood still for 30 s inside a step (`fn-watch2`,
`rep2` round 3; the backtraces of round 3 show both device threads waiting for route records
in `Flow::throttle`, the store pumped). The cause: a move of an expert out of a piece that is
given back writes the expert's table entry on the stream of copies out of slots, and the
eviction of that expert from the next piece given at the same end of a step writes its entry
on the stream of uploads; nothing ordered the two, so the older entry could land last and name
a slot that held another expert or had no memory any more. 3e22beb4 orders them
(`SlotDevice::order`) and a host test on slots whose copies out run late fails without it.
The runs after the fix are to be made (the cards were taken by the owner's production).

**Without a cap** (A/B, the binary of 8008eae0 against the branch at 65ac1ebb on the same base,
in turn; cards without a display, nothing else on them; 8204 tokens of prompt and 256 of
greedy decode, then a short request; load average 3 to 7 except the first base run):

| | Flash-Next IQ2_XS, both cards, `--ram 8G` | 27B UD-Q4_K_S, both cards, `--ram 8G` |
| --- | --- | --- |
| Plan, base and new | 4800 slots in 6.45 GiB / 8256 in 11.09 GiB, K and V of 18 048 tokens in VRAM, both | tensor split, context 190 464 tokens, both |
| Peak device 0 / 1, base and new | 11 662 / 11 696 MiB, both | 11 572 / 10 870 MiB, both |
| Prompt, tokens per s | base 509 (the first base run, at load 17 to 19, 387, and its second request stood still in a pass: the fault of the base that main fixed in #100); new 588, 563 | base 1175, 1174; new 1178, 1178 |
| Decode, tokens per s | base 33.4; new 33.3, 35.3 | base 76.8, 75.8; new 69.2, 71.3 |
| Decode step, ms (decode time over the steps it took) | | base 37.7, 38.2; new 37.2, 38.5 |
| Load to served, s | base 13.3; new 23.5, 13.0 | base 19.6, 16.8; new 24.3, 10.5 |
| Unload, ms | base 665; new 851, 843 | base 344, 340; new 340, 359 |

The 27B's greedy text was the same in every run. Its decode step took as long in both
binaries; the tokens per second differ by the drafts that were accepted (168 and 168 of 284 and
300 proposed for the base, 157 and 163 of 309 and 313 for the new one), which the draft policy
chooses for every step from the prices measured at load, and those differ a little from load to
load. Flash-Next's unload takes 0.2 s more: its pools of slots give their memory back piece by
piece (about 280 pieces of 64 MiB on the two cards) where one free each did.
