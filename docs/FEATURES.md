# What Ares can do today

Updated 2026-11-16 for `main`. One line per feature; how it works is in
[`ARCHITECTURE.md`](ARCHITECTURE.md), what is missing in [`OPEN.md`](OPEN.md).
Everything here runs on CUDA and was tested on an RTX 4070 with an RTX 3060. On Vulkan
the kernels exist; no model runs there yet.

## Models

| | Qwen3.8 27B (`qwen35`, dense) | Qwen3.8 Flash-Next (`qwen4exp`, MoE) |
| --- | --- | --- |
| Placements | one card, tensor split, layer split | one card, by layers, both cards on every layer |
| Weight types | all 28 GGUF types | all 28, experts included |
| K and V | `f16`, `q8` to `q4`, chosen separately (`--kv`) | `f16` or `q8` (default `q8`); `q7` to `q4` are refused |
| Larger than VRAM | no | experts in VRAM slots, a RAM tier and on the SSD; the CPU computes missed ones |
| Own drafts (MTP) | draft block in the file, with a draft vocabulary | draft block from a file (`--drafters <file>`); exact, no gain yet |
| Prefix cache | yes, with the SSD level | no |

## Speed
- Speculative decoding: the model's own drafts and a lookup in earlier text; window length
  from measured prices. On the 27B the text with drafts is token for token the text without.
- Prompts are read in steps of 512 rows, in pieces between rounds of generation, so running
  requests go on.
- 27B: the token embedding lies in host RAM (380 to 518 MiB of VRAM saved on the first card).
- Flash-Next: experts a prompt misses pass through transit slots; nothing is evicted for them.
- Measured (2026-11-12 to 15): 27B Q4_K_S on two cards reads prompts at about 1350 tokens/s
  and decodes a step in 24.5 ms; Flash-Next IQ2_XS on two cards generates 45 to 49 tokens/s
  without drafts.

## Caches and memory
- Prefix cache (27B): automatic, shared by all requests; checkpoints of the recurrent state.
- Prompt cache on SSD: `--cache <DIR>`, `--cache-size` (default 32G). A conversation resumes
  after its state left the cards, after an unload and after a restart.
- `--ram`: one limit for all page-locked host memory; too small a limit is refused with every
  share named.

## Sampling
- Temperature and dynamic temperature, top-k, top-p, min-p, typical, top-n-sigma, XTC,
  repetition, presence and frequency penalties, DRY, logit bias, mirostat, seed, `n`,
  logprobs. The sampler runs on the device and equals a CPU reference bit for bit.
- Defaults per model for every sampling field a request leaves out (`--temperature`,
  `--top-k`, `--top-p`, `--min-p`, the penalties, DRY, mirostat, `--samplers`, `--seed`):
  options and preset keys named like the request's fields. Without one the engine samples
  greedily.
- Reply length: `--max-tokens` for requests that give no limit, `--max-tokens-limit` as a
  cap (a request above it is cut and the reply says so). `/v1/models` shows what is in effect.

## Output control
- Reasoning: `reasoning_content` in and out, `--reasoning`, `--reasoning-effort`,
  `--reasoning-preserve`.
- Structured output: `grammar` (GBNF), `response_format` with `json_object` or `json_schema`;
  the thinking stays free.
- Tool calls: `tools`, `tool_choice` (`auto`, `none`, `required`, a named function),
  `parallel_tool_calls`, streamed; held to the tools' schemas by a grammar; a call the model
  opens inside its thinking is returned as a call; drafts go on while the rule is free.

## Server
- Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/messages` and
  `/v1/messages/count_tokens`, `/v1/responses` (stateless) and `/v1/responses/input_tokens`,
  `/v1/chat/completions/input_tokens`, `/v1/models`, `/health`, `/metrics`.
- Router mode: `--models-preset <file.ini>` with one section per model and `[*]` for all;
  a model is loaded when a request names it, the least recently used one makes room, idle
  models unload (`idle-unload`), aliases, `POST /models/load` and `/models/unload`,
  `GET /models/sse`.
- Main options (all also preset keys; `ares --help` has the list): `--model`, `--devices`,
  `--split`, `--shares`, `--context`, `--parallel`, `--kv`, `--precision`, `--ram`, `--cache`,
  `--cache-size`, `--drafters`, `--window`, `--draft-vocab`, `--reasoning`.
