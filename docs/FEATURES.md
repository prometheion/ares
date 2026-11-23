# What Ares can do today

Updated 2026-11-23 for `main`. One line per feature; how it works is in
[`ARCHITECTURE.md`](ARCHITECTURE.md), what is missing in [`OPEN.md`](OPEN.md).
Everything here runs on CUDA and was tested on an RTX 4070 with an RTX 3060. CUDA runs on the
GPU classes it was tested on only: Ampere (8.6, RTX 30) and Ada (8.9, RTX 40); another GPU is
refused (the server skips it) and is to run through Vulkan, where the kernels exist and no model
runs yet.

## Models

| | Qwen3.8 27B (`qwen35`, dense) | Qwen3.8 Flash-Next (`qwen4exp`, MoE) |
| --- | --- | --- |
| Placements | one card, tensor split, layer split | one card, by layers, both cards on every layer (the second card's role, a share of the dense parts or experts only, timed at load) |
| Weight types | all 28 GGUF types | all 28 in every tensor: experts, stream mixers and the n-gram block (exact) too |
| K and V | `f16`, `q8` to `q4`, chosen separately (`--kv`) | `f16`, `q8` to `q4`, chosen separately (`--kv`); the MTP block's in `q4` |
| Larger than VRAM | no | experts in VRAM slots, a RAM tier and on the SSD; the CPU computes missed ones, and what a step picks beyond a card's slots |
| MTP (one driver) | the block in the model file (`--drafters mtp`) or a file of its own (`--drafters <file>`), every placement, chain and picks on the card, a draft vocabulary; exact | the block from a file of its own or a model file that carries it, its experts resident in VRAM, chain and picks on the card, a draft vocabulary; exact |
| DFlash2 drafter (`--drafters <file>`) | chains, one card or tensor split, with the draft vocabulary; exact | chains, one card or both cards on every layer, with the draft vocabulary; the mean of the four streams |
| Prefix cache | yes, with the RAM and SSD levels | yes, with the RAM and SSD levels; its RAM is what the tier of experts leaves |

## Speed
- Speculative decoding: the model's drafts and the n-gram drafter; the chain runs
  in a step of its own and the round verifies as many of its drafts as pay, by the
  probability the drafter gave each and the measured price of a step. On the 27B the text
  with drafts is token for token the text without. On Flash-Next a draft row waits for an
  expert from the SSD only while it is worth the wait.
- Prompts are read in steps of up to `--ubatch-size` rows (512 by default), in pieces between
  rounds of generation, so running requests go on; while another request waits a piece is at
  most half a second. Every generating request keeps `--generation-floor` tokens per second
  (20 by default) while a prompt is read beside it, the prompt gets the rest of the time and at
  least half; on Flash-Next a long prompt beside generation is read layer by layer as when it is
  alone, paused for the rounds. A prompt read alone takes steps of up to `--batch-size` rows: on
  Flash-Next (32 768 by default, as far as they fit) layer by layer, every micro-batch of the
  own rows through a layer and then the layer's experts once for all rows, about 56 KiB a row
  kept on VRAM the expert slots lend for the step; such a step pauses between two
  micro-batches when a request comes, which then runs at once. (With K/V of a tile mean or a
  DFlash drafter: lent steps of up to 8192 rows, their scratch on VRAM the slots lend.)
- Flash-Next: in a prompt step of up to 2048 rows the CPU computes the experts few rows
  picked that are in RAM, beside the uploads, as the cost model splits them: a turn whose
  prompt is cached but for its last rows does not wait for a pass of uploads.
- The path of the prompt kernels is measured per weight type at load, on every card: the most
  exact of those within 5 % of the fastest (f16 where the tensor cores sum in f32 at full
  rate, else f16 summed in f16 or int8).
- The token embedding stays in host RAM (27B: page-locked, 380 to 518 MiB of VRAM saved on the
  first card; Flash-Next: the mapped file): a step stages its rows and the card decodes them.
- Flash-Next: experts a prompt misses pass through transit slots; nothing is evicted for them.
- Flash-Next: the CPU computes a missed expert from its bytes in RAM with kernels for the types of
  the Flash-Next files (AVX2 and AVX-VNNI, portable code on older CPUs), and uploads it on loan
  behind that while the link has room.
- Flash-Next: the routers of the next two layers, applied on the card, predict a step's
  picks; the store reads ahead from the SSD what no tier holds where such predictions have
  been picked more often than not, behind every demand read and into RAM nobody uses.
- Measured (2026-11-12 to 15): 27B Q4_K_S on two cards reads prompts at about 1350 tokens/s
  and decodes a step in 24.5 ms; Flash-Next IQ2_XS on two cards generates 45 to 49 tokens/s
  without drafts.
- Long prompts: the attention of a prompt's rows has a kernel of its own (integer scores over
  quantized K); the 27B at the owner's settings (`q5`, 262144 cells, 2 sequences) reads 32k at
  about 1000 tokens/s and 98k at about 710 (2026-11-23; before 655 and 324).

## Caches and memory
- Prefix cache (both models): automatic, shared by all requests; checkpoints of the state per
  sequence (recurrent state, Flash-Next's indexer keys, n-gram history and feed, MTP block).
- Checkpoints where conversations part (both models): at the end of the system prompt and of
  the tool list (kept longest), at every message boundary (1024 tokens apart at least), every
  8192 tokens of a prompt; a new conversation with the same system prompt, an edited message or
  a cut tool result replays only from the last one before the change.
- Claude Code's billing stamp (`cch=` in its system prompt) pinned on `/v1/messages`, as
  upstream llama.cpp does, so its turns hit the prompt cache.
- Prompt cache in RAM: what leaves the cards is kept in page-locked RAM inside `--ram`,
  the lowest user of it (below Flash-Next's tier of experts), and comes back from there;
  elastic, without a flag; one checkpoint place is always kept, so a conversation resumes
  from its last checkpoint even when the tier takes the rest.
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

## Images
- Image input on both models with `--mmproj <FILE>` (the model's projector): `image_url` parts on
  `/v1/chat/completions` (data URLs, or base64 alone), Anthropic `image` blocks on `/v1/messages`
  (also in tool results), `input_image` on `/v1/responses`; JPEG, PNG, GIF, WebP, BMP. The
  tower runs on a card, its weights in RAM by default (`--mmproj-weights ram|vram`); an image
  becomes at most `--image-max-tokens` rows (4096 by default, as llama.cpp), placed as
  llama.cpp's `mtmd` places them. A conversation with an image resumes from the prompt cache.
  Remote image URLs are not fetched.

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
- Thinking budget: `reasoning_budget_tokens` (and llama.cpp's, vLLM's and Anthropic's names),
  `--reasoning-budget-tokens`, `--reasoning-budget-message`; the thinking ends at the budget and
  the answer follows. A loop in the thinking ends it; one token repeated 256 times ends a reply.
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
- `/tokenize` (a text or a chat request, with pieces) and `/detokenize`, in llama.cpp's shapes;
  a text (there and on `/v1/completions`) reads control tokens such as `<|im_start|>` as llama-server.
- `/props` (`?model=`): context, chat template, request defaults, what the template can do; of a
  model that is not loaded too (read from its preset and file, without a load).
- A watchdog: a step that runs longer than 600 s fails its engine; its requests end with the
  reason and the router unloads it.
- Usage counts the tokens of the thinking (`completion_tokens_details.reasoning_tokens`,
  `output_tokens_details` on `/v1/responses`).
- A stop string that ends a reply is named: `stop_reason: stop_sequence` and `stop_sequence` on
  `/v1/messages`, `stop_reason` (vLLM's field) on the OpenAI routes.
- `--chat-template-file` replaces the model file's chat template.
- A last assistant message is continued (llama.cpp's prefill, `continue_final_message`,
  Anthropic's prefill), in its answer or in its reasoning.
- Claude Code (2.1) as a client of `/v1/messages`: `context_management` with
  `clear_thinking_20251015`, `thinking.display`, system turns between the others, effort `high`.
- `/v1/responses` `truncation: "auto"`: the oldest whole turns of a conversation that does not
  fit the context leave it.
- `/v1/messages`: `context_management` edits `clear_thinking_20251015` and
  `clear_tool_uses_20250919` as Anthropic's service makes them, listed in
  `context_management.applied_edits`; `thinking.display: "omitted"` (an empty thinking block
  whose signature carries the thinking back into the next prompt).
- System and developer messages: the ones first are the system prompt, later ones are turns of
  the user, on every route; `reasoning_effort` `high` is the template's default where the
  template refuses it (Flash-Next), on every route.
- Priorities: `priority` (a lower number first, vLLM's) on the OpenAI routes and the header
  `x-ares-priority` on every route; the line, the places, the pages and the rows of a step go to
  the better priority first, and running requests of a worse one give way (to the prefix cache).
- `--queue <N>`: a full line answers 503 with `Retry-After`, or takes a request of a better
  priority in place of its worst waiting one; `--timeout` for reading a request and writing an
  answer.
- Router mode: `--models-preset <file.ini>` with one section per model and `[*]` for all;
  a model is loaded when a request names it, the least recently used one makes room, idle
  models unload (`idle-unload`), aliases, `POST /models/load` and `/models/unload`,
  `GET /models/sse`.
- Main options (all also preset keys; `ares --help` has the list): `--model`, `--devices`,
  `--split`, `--shares`, `--context`, `--parallel`, `--kv`, `--precision`, `--ram`, `--cache`,
  `--cache-size`, `--drafters`, `--window`, `--draft-vocab`, `--reasoning`, `--mmproj`.
