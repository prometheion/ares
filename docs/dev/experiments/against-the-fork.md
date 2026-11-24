# Experiments: the 27B against the owner's llama.cpp fork

Dated measurement logs, moved here verbatim on 2026-11-24 from the architecture document (now [`../architecture/`](../architecture/overview.md)) and from [`OPEN.md`](../../OPEN.md). They are not kept current: each names its date, build and machine.

## Against the owner's production (2026-11-23, the 27B UD-Q4_K_S on the RTX 4070 + RTX 3060)

The fork's preset (`models.ini`, `-ts 0.625,0.375 -as 1,3`, 262144 cells, 2 slots, q5_1 K/V, MTP
and n-gram drafts) against Ares at the same settings (`--shares 0.625,0.375 --context 262144
--parallel 2 --kv q5 --drafters mtp --draft-vocab`, heads by context 0.300 / 0.700), peer
access, greedy, the bench of `/var/tmp/ares-tools/prod-ab/` (fork 2 rounds, Ares 3):
| | fork | Ares |
| --- | --- | --- |
| Decode, first ask, German / English / code, tokens/s | 61-63 / 67-68 / 76-77 | 71-74 / 78-86 / 103-104 |
| Decode, the same ask again | 62-63 / 67-68 / 76-77 | 251-252 (the n-gram drafter copies its last answer) |
| Prompt 2k / 8k / 32k / 98k, tokens/s | 840 / 994 / 944 / 752 | 1135 / 1030 / 654 / 325 |
| Agent turns, first token, first / second ask | 0.5-7.2 s / 0.1-0.5 s | 0.2-5.9 s / 0.1-0.2 s |
| VRAM at the end | 11.6 / 11.7 GiB | 11.5 / 10.1 GiB |

## Parallel requests against the owner's production (2026-11-23, `bench/parallel`)

The owner (2026-11-23): "How does Ares compare with my fork in parallel calls? That is really the
whole reason I am making the engine." Same settings as the section above (the fork's preset run
as the router's child with its own port and cache; Ares `--parallel 2`, and 4 and 8 at the same
262144 cells), greedy, MTP on, each request its own topic (Ares' n-gram drafter copies from every
sequence it has seen). Client `/var/tmp/ares-tools/prod-ab/parallel.py`, sessions
`psession.sh`, notes and every number `/var/tmp/ares-parallel/notes.md`. Fork and Ares
alternated, two rounds each, machine load 2 to 5 unless named.
| | fork | Ares |
| --- | --- | --- |
| One request alone, de / en / code, tokens/s | 58-65 / 60-64 / 73-76 | 72-74 / 63-75 / 91-103 |
| Two at once, sum while both run, de / en / code | 98 / 104-108 / 116-129 | 105-111 / 93-116 / 106-134 |
| Two at once, the two requests (quiet round) | even (49 + 50, 52 + 55, 57 + 59) | uneven (49 + 67, 52 + 68, 85 + 58) |
| 3 / 4 / 8 requests at once, sum tokens/s (`--parallel 2`) | 86-87 / 100-109 / 95-97 | 115 / 107 / 110 |
| the same with `--parallel 4` / `8` (Ares) | | 117-120 / 139-149 / 143-150; 115-124 / 142-154 / 204-206 |
| 8 at once, slowest first token | 19.3 s | 17.6 s (p2), 11.0 s (p4), 0.51 s (p8) |
| 8k prompt beside a generating request: first token (alone) | 15.2-15.5 s (9.1-9.3) | 20.4 s (7.9) |
| the generating request's rate meanwhile (alone) | 21-23 tokens/s (60-65) | 34-35 (67-72) |
| 32k prompt beside it: first token (alone) | 59.8-59.9 s (34.1) | 124.1-124.4 s (49.0-49.1) |
| 98k prompt beside it: first token (alone) | 208.9-219.1 s (127.8-134.8) | 754-763 s (296) |
| longest gap of the generating request at 8k / 32k / 98k | 0.54-0.66 / 0.51-0.60 / 0.81-1.01 s | 0.37 / 0.47-0.67 / 1.15-1.41 s |
| Two agents (16.6k-token system prompt and tools), cold, first turns | 22.3 / 47.7 s | 19.9 / 19.9 s |
| a turn with a 5.8k tool result at 17k depth, alone / both agents at once | 6.6-7.6 s / 9-17 s | 10.0-10.2 s / 12.1 and 26-29 s |
| small cached turns | 0.23-1.1 s | 0.15-0.25 s |
| session walls, cold pair / warm pair | 70 + 80 s / 38-40 s | 69-71 s / 31-37 s |
| A short request beside two generations (both places busy) | 24.5-25.8 s | priority 0: 0.15-0.22 s (the generation that gives way: a gap of 0.55-0.71 s, ends 1 s later), same priority: 18-21 s |
| the same beside a generation and a 32k prompt | 54.9-57.9 s | priority 0: 0.28-0.41 s, same priority: 38-54 s |
| VRAM peak (p2; p8) | 11.5 / 11.6 GiB | 11.2 / 9.9 GiB; 11.3 / 11.2 GiB |

- Fourteen busy loops beside the server (a machine busy with builds): Ares loses 6 to 12 % alone and
  for two at once, the fork 12 to 24 % (its step grows from 33.6 to 40 ms: its samplers and host
  work). One Ares round during heavy builds (load 13 to 28) lost more (4 at once 83 against 108
  tokens/s); not repeated.

- Seen of the fork: a short request sent 0.5 s before a 32k prompt had its first token after 34 s
  (three of four runs): its prompt went into the same batches as the long one, since concurrent
  prefill starts only once a slot generates. Ares in the same spot: 0.15-0.75 s.
