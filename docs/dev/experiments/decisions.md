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
