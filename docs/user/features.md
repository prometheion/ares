# Features

What Ares does today on `main`, one line each. Everything here runs on CUDA and was
tested on an RTX 4070 with an RTX 3060 and on an RTX 4060 Ti (see
[hardware.md](hardware.md)). The long form of this list, as of 2026-11-23, is
[`../dev/features-in-detail.md`](../dev/features-in-detail.md); what is missing is in
[`../OPEN.md`](../OPEN.md).

## Models

| | Qwen3.8 27B (`qwen35`, dense) | Qwen3.8 Flash-Next (`qwen4exp`, mixture of experts) |
| --- | --- | --- |
| Placements | one card, tensor split, layer split | one card, by layers, both cards on every layer (the second card's role measured at load) |
| Weight types | all 28 GGUF types | all 28, in every tensor |
| K and V | `f16`, `q8` to `q4`, K and V apart (`--kv`) | the same; the MTP block's in `q4` |
| Larger than VRAM | no | experts in VRAM slots, a RAM tier and the SSD; the CPU computes missed ones |
| Context | `--context`, or the model's own (262 144) as far as VRAM holds it | `--context` (all sequences together), or the model's own (262 144), with the K and V beyond the first pages in RAM |
| MTP drafts | the block in the model file or in a file of its own; exact | the block from a file of its own or from the model file; exact |
| DFlash2 drafts | chains, one card or tensor split | chains, one card or both cards on every layer |
| Prompt cache | RAM and SSD levels | RAM and SSD levels; its RAM is what the expert tier leaves |
| Images | with `--mmproj` | with `--mmproj` |

## Speed

| Feature | In one line |
| --- | --- |
| Speculative decoding | model drafts and the n-gram drafter; a step verifies as many drafts as pay, by their probability and the measured price of a step |
| Exact drafts | on the 27B the text with drafts is token for token the text without |
| Prompts beside generation | prompts are read in pieces between rounds; each generating request keeps `--generation-floor` tokens/s (20) |
| Long prompts, Flash-Next | read layer by layer, each expert crosses the link once per prompt; a new request cuts in between micro-batches |
| Prompt kernels | the path of each weight type measured at load on every card; the attention of long prompts has a kernel of its own |
| Experts on the CPU | Flash-Next computes a missed expert from RAM on the CPU (AVX2, AVX-VNNI) or uploads it, whichever the cost model says |
| Expert read-ahead | the routers of the next two layers predict picks; the store reads ahead from the SSD what pays |
| Expert pruning | `--expert-prune` (off by default) lets a step skip or replace a weak pick that only the SSD holds; it changes the output |
| Parallel requests | `--parallel` sequences at once (4 by default), each with its own place |
| Priorities | `priority` or `x-ares-priority`; a better priority goes first and a worse running one gives way |

## Caches and memory

| Feature | In one line |
| --- | --- |
| Prefix cache | automatic, shared by all requests, with checkpoints of the recurrent state |
| Checkpoints | at the end of the system prompt and the tools, at every message boundary, every 8192 tokens of a prompt |
| Claude Code's billing stamp | pinned on `/v1/messages` as upstream llama.cpp does, so its turns hit the cache |
| Prompt cache in RAM | inside `--ram`, the lowest user of it; one checkpoint place is always kept |
| Prompt cache on SSD | `--cache <DIR>`, `--cache-size`; a conversation resumes after an unload and a restart |
| One RAM limit | `--ram` covers all page-locked memory; the engine gives memory back when the machine needs it |
| Usage profile | Flash-Next fills its expert tiers from a profile of use at the start |

## API

| Feature | In one line |
| --- | --- |
| Routes | OpenAI chat, completions, Responses (stateless); Anthropic Messages; `/v1/models`, `/health`, `/metrics`, `/props`, `/tokenize`, `/detokenize` ([api.md](api.md)) |
| Sampling | temperature, dynamic temperature, top-k, top-p, min-p, typical, top-n-sigma, XTC, penalties, DRY, logit bias, mirostat, seed, `n`, logprobs; on the device |
| Defaults per model | every sampling field has an option; without one a reply is greedy |
| Thinking | `reasoning_content` in and out, efforts, a budget of tokens, a loop guard |
| Structured output | GBNF `grammar`, `json_schema`, `response_format`; the thinking stays free |
| Tool calls | held to the tools' schemas by a grammar, streamed, also when opened inside the thinking |
| Claude Code | works against `/v1/messages`, with `context_management` and `thinking.display` |
| Reply length | `--max-tokens` and the cap `--max-tokens-limit` |
| Continuing a message | a last assistant message is continued (prefill) |
| Decisions (Jev mode) | `/v1/systemone` as Flide serves it: questions about a state answered from the labels' logits, the state computed once and forked into the questions, several requests in one body |

## Server

| Feature | In one line |
| --- | --- |
| Access | API keys (`--api-key`, `--api-key-file`), CORS (`--cors-origins`), Host and Origin checks |
| Configuration | every option also as `ARES_<NAME>` and in `--config <FILE>` ([options.md](options.md)) |
| Several models | `--models-preset <file.ini>`: loaded on request, aliases, idle unload, load and unload routes |
| The line | `--queue <N>` bounds it (503 with `Retry-After`); `--timeout` for slow clients |
| Watchdog | a step that runs longer than 600 s fails its engine; its requests end with the reason |
| Logs and metrics | a line per request, Prometheus metrics ([quickstart.md](quickstart.md#logs-and-metrics)) |
| Version | `ares --version` names the version, the commit and the day of the build; so do the log's first line and `/props` |
