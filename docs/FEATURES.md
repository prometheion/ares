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
| K and V | `f16`, `q8` to `q4`, chosen separately (`--kv`) | `f16`, `q8` to `q4`, chosen separately (`--kv`); the draft block's in `q4` |
| Larger than VRAM | no | experts in VRAM slots, a RAM tier and on the SSD; the CPU computes missed ones |
| Own drafts (MTP) | draft block in the file, with a draft vocabulary | draft block from a file (`--drafters <file>`), its experts resident in VRAM; exact |
| DFlash2 drafter (`--drafters <file>`) | chains, one card or tensor split, with the draft vocabulary; exact | chains, one card or both cards on every layer; the mean of the four streams |
| Prefix cache | yes, with the RAM and SSD levels | no |

## Speed
- Speculative decoding: the model's own drafts and a lookup in earlier text; the chain runs
  in a step of its own and the round verifies as many of its drafts as pay, by the
  probability the drafter gave each and the measured price of a step. On the 27B the text
  with drafts is token for token the text without. On Flash-Next a draft row waits for an
  expert from the SSD only while it is worth the wait.
- Prompts are read in steps of 512 rows, in pieces between rounds of generation, so running
  requests go on.
- 27B: the token embedding lies in host RAM (380 to 518 MiB of VRAM saved on the first card).
- Flash-Next: experts a prompt misses pass through transit slots; nothing is evicted for them.
- Flash-Next: the routers of the next two layers, applied on the card, predict a step's
  picks; the store reads ahead from the SSD what no tier holds where such predictions have
  been picked more often than not, behind every demand read and into RAM nobody uses.
- Measured (2026-11-12 to 15): 27B Q4_K_S on two cards reads prompts at about 1350 tokens/s
  and decodes a step in 24.5 ms; Flash-Next IQ2_XS on two cards generates 45 to 49 tokens/s
  without drafts.

## Caches and memory
- Prefix cache (27B): automatic, shared by all requests; checkpoints of the recurrent state.
- Prompt cache in RAM (27B): what leaves the cards is kept in page-locked RAM inside `--ram`,
  the lowest user of it, and comes back from there; elastic, without a flag.
- Prompt cache on SSD: `--cache <DIR>`, `--cache-size` (default 32G), below RAM. Only what
  leaves RAM is written, and what is worth keeping at an unload or a stop; a conversation
  resumes after an unload and after a restart.
- `--ram`: one limit for all page-locked host memory; too small a limit is refused with every
  share named. Below it the engine follows the machine: when other programs need memory it
  gives page-locked memory back between steps (checkpoints first, the expert tier last) and
  takes it again when the memory is free; the reserve it keeps is derived, not fixed.
- Flash-Next: a usage profile of the experts (in the `--cache` directory, else `~/.cache/ares`)
  fills the tiers at the start in place of the warm-up text, and their empty slots while idle.
- Flash-Next: `--expert-prune RULE:RANK` (off by default) lets a step leave out a weak pick whose
  expert only the SSD holds, or take another expert that VRAM or RAM holds in its place
  (`skip`, `router`, `similar`), instead of waiting for the drive; it changes the output.

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
  `/v1/chat/completions/input_tokens`, `/v1/models`, `/health`, `/metrics`, `/props`,
  `/tokenize`, `/detokenize`.
- API keys: `--api-key`, `--api-key-file` (or `ARES_API_KEY`), as `Authorization: Bearer` or
  `x-api-key`, compared in constant time; `/health` stays open.
- CORS for the origins of `--cors-origins`; a server on a local address without keys answers
  only to local names and pages (against DNS rebinding), `--allowed-hosts` adds names.
- Every option is also `ARES_<NAME>` and a line of `--config <FILE>` (command line, then
  environment, then file).
- `/tokenize` (a text or a chat request, with pieces) and `/detokenize`, in llama.cpp's shapes.
- `/props` (`?model=`): context, chat template, request defaults, what the template can do.
- Usage counts the tokens of the thinking (`completion_tokens_details.reasoning_tokens`,
  `output_tokens_details` on `/v1/responses`).
- A stop string that ends a reply is named: `stop_reason: stop_sequence` and `stop_sequence` on
  `/v1/messages`, `stop_reason` (vLLM's field) on the OpenAI routes.
- `--chat-template-file` replaces the model file's chat template.
- A last assistant message is continued (llama.cpp's prefill, `continue_final_message`,
  Anthropic's prefill), in its answer or in its reasoning.
- `--queue <N>`: a full line answers 503 with `Retry-After`; `--timeout` for reading a request
  and writing an answer.
- Router mode: `--models-preset <file.ini>` with one section per model and `[*]` for all;
  a model is loaded when a request names it, the least recently used one makes room, idle
  models unload (`idle-unload`), aliases, `POST /models/load` and `/models/unload`,
  `GET /models/sse`.
- Main options (all also preset keys; `ares --help` has the list): `--model`, `--devices`,
  `--split`, `--shares`, `--context`, `--parallel`, `--kv`, `--precision`, `--ram`, `--cache`,
  `--cache-size`, `--drafters`, `--window`, `--draft-vocab`, `--reasoning`.
