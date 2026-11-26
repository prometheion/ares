# Experiments: decisions (Jev mode)

Dated measurement logs of `/v1/systemone` and `Engine::decide` ("Decisions" in
[`../architecture/sequence-state.md`](../architecture/sequence-state.md)). They are not kept
current: each names its date, build and machine.

## The answers against the CPU reference and greedy generation (2026-11-25)

Branch `feat/jev`, the development machine (RTX 4070 and RTX 3060), other agents' builds
running (load 13 to 25). One request: an email (a JSON object of `from`, `subject` and a
German `body`) and four questions: a choice of 4 options with descriptions, a noul, a score of
5 levels and a choice of 30 options (letters, then two-letter labels). The tokenizer of Qwen3.5
and Qwen3.8 has 230 two-letter labels after Z that are one token where the reply starts (256
labels). `crates/model/tests/qwen35_decisions.rs`, `crates/server/tests/real_systemone.rs`.

| Model, placement | Checked against | Result |
| --- | --- | --- |
| Qwen3.5-0.8B Q4_K_M, the 4070, exact products | the CPU reference on each question's whole prompt | largest difference of an answer's probability 0.0064; the same choice where the reference's two best are 0.05 apart or more |
| the same | a greedy generation of each question alone (one token) | 4 of 4 replies start with a label, each the chosen one |
| Qwen3.8-27B UD-Q4_K_S, tensor split 0.625 / 0.375 | a greedy generation of each question alone | 4 of 4 start with a label, each the chosen one (the 30-option question chose label R, the option the email is about, at 0.999) |
| Qwen3.8-Flash-Next IQ2_XS, `--ram 8G`, through the route | a chat completion of each question alone (greedy, one token) | 3 of 3 start with a label, each the chosen one |

Times of these runs (not measurements of speed: cold engines, other agents' work on the
machine): the 0.8B answered the four questions (474 tokens, 105 of them shared) in 69 to
165 ms; the 27B in 1920 ms on its first request after the load and 958 ms for the same request
again (port build, before forks stopped taking checkpoints); Flash-Next three questions in
7820 ms on its first request.

## Decisions per second on the 27B at the owner's production settings (2026-11-26)

Branch `feat/jev` at 6f4912e8 (release build), the development machine, both cards, the
owner's production unloaded. The engine as `/var/tmp/ares-tools/prod-ab/session.sh` starts it
for production: Qwen3.8-27B UD-Q4_K_S, tensor split 0.625 / 0.375, a context of 262 144 tokens,
`--parallel 2`, `--kv q5`, MTP drafts with the draft vocabulary, `--ram 16G`, reasoning effort
medium, a prompt cache on SSD (8G here, production has 48G; nothing was read back from it).
Other agents' builds and GPU tests ran beside it (load 6 to 16), one GPU run at a time through
the lock.

The client (`/var/tmp/ares-jev/bench.py`): 8 states, each an email (a JSON object, about 130
tokens of the shared start), and N questions of K options about each (a third of them noul and
score questions for K = 4; all choices for K = 16). Every mode reads states of its own, which no
other mode read before; two rounds, the second in the reverse order of the modes.

- **decide**: one `POST /v1/systemone` per state with its N questions, 2 at a time.
- **batch**: one request with the 8 states in `requests`.
- **gen1**: a chat completion per question, the prompt the decision reads, greedy, thinking
  off, one token: the least that generating an answer can cost. 2 at a time.
- **gen**: the same with the answer written out (at most 24 tokens; the replies are a label
  and its option, a few tokens).

Decisions per second (questions answered), round 1 and round 2:

| N questions x K options | decide | batch | gen1 | gen | decide against gen1 |
| --- | --- | --- | --- | --- | --- |
| 8 x 4 | 9.49, 8.65 | 9.46, 7.07 | 5.01, 5.78 | 5.56, 5.10 | 1.5 to 1.9 times |
| 16 x 16 | 7.34, 6.65 | 7.35, 6.14 | 4.38, 4.06 | 3.74, 3.76 | 1.6 to 1.8 times |
| 8 x 4, `--parallel 8` (not production) | 13.09, 8.64 | 12.44, 8.76 | 4.82, 8.89 | 8.21, 7.01 | 1.0 to 2.7 times |

- The answers are the model's: every gen1 reply started with a label, and with the label the
  decision chose in 63 or 64 of 64 (8 x 4, six rounds on 2026-11-25 and 2026-11-26; one near
  tie parts now and then, see "Rounding" in the topic file) and in 128 of 128 (16 x 16), each
  with the same states for both modes.
- A decision of 8 questions takes 1.3 to 2.1 s, of 16 questions about 4 s: with two slots the
  forks of a decision run two at a time, each restoring the shared start's checkpoint (150 MiB
  of recurrent state) from RAM. A batch is no faster than two decisions at a time, for the
  same reason: the slots bound both. With 8 slots the first round reached 13 decisions per
  second; the second round was slower for all modes (other agents' work on the machine), so
  the 8-slot numbers say little beyond that. Forks that need no slot of their own are the
  lever ("Jev mode" in [`../../OPEN.md`](../../OPEN.md)).
- Before the fix of 2026-11-26 (a tail of the prompt cache that came back from RAM was taken
  for a full page) the chat completions after a batch of decisions took the engine down at
  `--parallel 2`, every time (`ab1.log`, `modes-2.log` of the first runs); the fixed build ran
  every run here without an error.
