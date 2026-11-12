# Feature parity checklist: what the other inference engines have, and where Ares stands

Made on 2026-11-11. Read-only survey: nothing was built or run, no repository was changed. The `Ares` column of
R03 to R12, R14, R16, R21 and H02, the counts and the lists were brought up to date on 2026-11-12, when the
sampler's features (penalties, DRY, logit bias, probabilities, `n` and the rest) were built; G01, G03, G06, G07
and G09 were brought up to date the same day, when structured output was built.
One row is one capability that a user or operator of an inference server can observe or rely on.

## How to read it

**Columns.** `Engines`: who has it (short names below). `What it is`: precise enough to build or to reject.
`Ares`: `done`, `partly: <what is missing>` or `not started`, from `docs/ARCHITECTURE.md` (Status of 2026-11-11),
`README.md`, `crates/server/src/{cli,wire,routes,error}.rs` and `crates/runtime/src/conductor.rs`.
`Frame`: `yes`, or `no: <the scope decision that excludes it>`. `Need`: for rows inside the frame, how much users
depend on it (`essential`, `common`, `niche`); this is my judgment, not a measurement.

**Verified or not.** An engine named in a row was checked today in the source listed for it under
[Sources](#sources) (its documentation, or its code or docs on disk). A name with `?` was **not** verified today:
it comes from Ares' own notes, from an ambiguous summary, or is inferred. Web pages were read through a
summarising fetch, so a detail can be wrong even where the feature is verified; the exact option names quoted in
backticks are the ones the fetch returned.

**Short names.** `llama` = llama.cpp upstream `llama-server`; `fork` = the user's llama.cpp fork; `ik` =
ik_llama.cpp; `kobold` = KoboldCpp; `ollama`; `vllm`; `sglang`; `tgi` = Hugging Face TGI (in maintenance mode by
its own README); `trt` = TensorRT-LLM (`trtllm-serve`); `triton`; `nim`; `lmdeploy`; `tabby` = TabbyAPI on
ExLlamaV3 (its sample config lists no ExLlamaV2 backend any more); `mistralrs`; `mlc`; `sonar` = Aphrodite, now
called Sonar; `localai`; `strata`; `avalon`; `tfold` = TensorFold; and three engines outside the brief's list that
have something the others lack: `swap` = llama-swap (the proxy in front of the user's fork), `lmstudio`,
`tokenspeed`.

**Three engines that inherit rows.** `fork`, `avalon` and `ik` are forks of llama.cpp. `fork` is named in a row
only where Ares' documents (`docs/ARCHITECTURE.md`: scope decisions, "What is taken from where", Status, Next
steps, Speculation) or the production preset (its `models.ini`) name
the feature. It presumably has every `llama` row as well (inferred: it is a fork and its preset uses upstream's
keys); its history was not read. `avalon` carries upstream's server unchanged as far as its
`tools/server/README.md` shows (the same text as upstream's), so every `llama` row applies to it; it is named
only for what its own README adds. `ik` is named only for what its README states.

**The frame** (the user's standing scope decisions): server only, one binary, no CLI tools, no GUI; Linux only;
GGUF files with llama.cpp's weight quantization types; CUDA 13 and Vulkan with full parity; one to a few consumer
or workstation GPUs in one machine; no pure CPU inference (the CPU may compute MoE experts that are not on a
GPU); newest model architectures only (Qwen3.8-27B, Qwen3.8-Flash-Next, later Qwen4); no compatibility, legacy
or no-op options; Ares may do a thing differently as long as the result is not worse.

## 1. API surface

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | Chat completions (OpenAI), streamed and not | llama, ik, kobold, ollama, vllm, sglang, tgi, trt, lmdeploy, tabby, mistralrs, mlc, sonar, localai, strata, tfold, lmstudio | `POST /v1/chat/completions`; SSE chunks with a role chunk, deltas, `finish_reason`, `[DONE]` | done (qwen35 and qwen4exp models, text parts only) | yes | essential |
| A02 | Text completions (OpenAI) | llama, kobold, ollama, vllm, sglang, trt, lmdeploy, tabby, tfold, tgi? | `POST /v1/completions` with one string prompt, no chat template | done | yes | common |
| A03 | Prompt as token ids; several prompts in one request | llama, sglang, tfold (ids), vllm? | `prompt` as a list of ids, or a list of prompts answered as several choices | not started (refused by name) | yes | niche |
| A04 | Responses API (OpenAI), stateless | llama, ik, vllm, trt, ollama, strata, tfold, mistralrs, lmstudio, swap | `POST /v1/responses` run as the equivalent chat request; typed SSE events, function tools, reasoning items, `text.format` | not started | yes | common |
| A05 | Stored responses, conversations held by the server | tfold, vllm, sglang, lmstudio | `store`, `previous_response_id`, `GET` and `DELETE /v1/responses/{id}` (vllm also cancel); lmstudio "stateful chats". ollama and strata are stateless on purpose | not started | yes | niche |
| A06 | Responses over WebSocket | tfold | `GET /v1/responses` upgrade, `response.create` events, lanes and a cache per socket | not started | yes | niche |
| A07 | Anthropic Messages API | llama, vllm, strata, tfold, mistralrs, sonar, lmdeploy, lmstudio, swap | `POST /v1/messages`: `system`, content blocks, `tool_use` and `tool_result`, thinking blocks, Anthropic's SSE events, `x-api-key` | not started | yes | common |
| A08 | Token counting | llama, vllm, tfold, mistralrs, swap | `/v1/messages/count_tokens`, `/v1/responses/input_tokens`, `/v1/chat/completions/input_tokens`: render the request, return the count, generate nothing | not started | yes | common |
| A09 | Tokenize and detokenize | llama, vllm, sglang, tfold, tabby, kobold | text or chat messages to ids (optionally the pieces), ids to text | not started | yes | common |
| A10 | Render a request without running it | llama, vllm, trt, tabby, tfold | the prompt text or ids a chat body renders to (`/apply-template`, `/v1/chat/completions/render`, `generate:false`) | not started | yes | niche |
| A11 | Decisions API | llama, ik, vllm, sglang, tfold, lmstudio, localai | typed questions (yes/no, choice, score) answered from the next-token logits of the served model, no text generated: `/v1/systemone`, `/v1/decisions`, `/v1/score` | partly: route, wire format and planner rows exist; `Engine::decide` is not implemented and the route answers 501 | yes | niche |
| A12 | Embeddings | llama, kobold, ollama, vllm, sglang, trt, tabby, localai, tfold, sonar, swap | `/v1/embeddings` (float or base64, `dimensions`) from an embedding model | not started | no: newest model architectures only (needs embedding models) | - |
| A13 | Rerank, score, classify, pooling | llama, vllm, sglang, tfold, sonar, localai, swap | `/v1/rerank` and relatives with reranker or classifier models | not started | no: newest model architectures only (other model kinds) | - |
| A14 | Batch requests | vllm, trt, sglang? | many requests in one call or file (`/v1/chat/completions/batch`, OpenAI files and batches) | not started | yes | niche |
| A15 | Model list | all | `GET /v1/models` with id and context length | done (one model) | yes | essential |
| A16 | Model properties and capability discovery | llama, strata, tfold, tabby, ollama, sglang, vllm, localai | chat template, defaults, modalities, tool and reasoning support, thinking switch: `/props`, a `capabilities` object, `/api/show`, `/get_model_info` | not started | yes | common |
| A17 | Health | all | `GET /health`; 503 while a model loads; some add ready or generate probes (sglang `/ready`, `/health_generate`) | done (liveness only) | yes | essential |
| A18 | Prometheus metrics endpoint | llama, vllm, sglang, tgi, trt, tabby, mistralrs, strata, tfold, sonar, swap | `GET /metrics` in the text format | done (20 counters and gauges) | yes | common |
| A19 | A second, native request dialect | llama (`/completion`), kobold (KoboldAI API, Ollama emulation), ollama (`/api/*`), sglang (`/generate`), tgi (`/generate`), sonar (Kobold API) | an engine-specific generation API beside the OpenAI one | not started | no: no compatibility (the features behind those APIs are rows of their own) | - |
| A20 | Infill | llama, ollama, swap | prefix plus suffix with the model's fill-in-the-middle tokens (`/infill`, `suffix`) | not started (`suffix` is refused) | yes | niche |
| A21 | Speech, transcription, realtime audio | llama, vllm, ollama, kobold, localai, tfold, sonar, swap | `/v1/audio/*`, `/v1/realtime` over WebSocket | not started | no: newest model architectures only (other model kinds) | - |
| A22 | Image, video and music generation | kobold, localai, trt, tfold, mistralrs, swap | `/v1/images/generations` and relatives | not started | no: newest model architectures only (other model kinds) | - |
| A23 | Control of a running completion | llama | `POST /v1/chat/completions/control` with `reasoning_end` ends the thinking of a request in flight | not started | yes | niche |
| A24 | Cancel by id | tabby, vllm, kobold | cancel a request without closing its connection | not started | yes | niche |
| A25 | OpenAPI description | vllm, tgi, mlc, kobold, lmdeploy, tabby, sonar | `/docs` or `openapi.json` | not started | yes | niche |
| A26 | gRPC front end | vllm, sglang, trt, triton? | the same requests over gRPC | not started | yes | niche |
| A27 | Unix socket, several addresses, path prefix | llama, vllm, sglang | `--host x.sock`, `--uds`, `--api-prefix`, `--root-path`, `--reuse-port` | not started | yes | niche |
| A28 | OpenAI error envelope with stable codes | llama, vllm, strata, tfold | `{error:{message,type,param,code}}`; `context_length_exceeded`, `model_not_found`; a request that fails early gets an HTTP status, not an error inside a 200 stream | done | yes | essential |
| A29 | Built-in web UI | llama, kobold, strata, tfold, mistralrs, tabby, localai, swap | chat or admin pages served by the server | not started | no: no GUI | - |

## 2. Request features: sampling

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | Temperature, top-k, top-p, min-p | all | per request; temperature 0 is greedy; `top_k` 0 or -1 is off | done (on the device) | yes | essential |
| R02 | Seed | llama, ollama, vllm, trt, mlc, lmdeploy, strata, tfold, sglang?, tgi? | an explicit seed repeats a draw; none means a fresh one; in trt, tfold and Ares the draw of a position does not depend on batch mates | done (generator keyed by seed and position); the seed drawn for a request is not reported (tfold logs it) | yes | common |
| R03 | Presence and frequency penalty | llama, fork, ollama, vllm, sglang, tgi, trt, mlc, strata, tfold, tabby | OpenAI's rule over the tokens of the reply; under drafts every verified row is penalised for its own path (strata, tfold) | done (llama.cpp's rule: over the last `repeat_last_n` tokens of the prompt and the reply, 64 unless set, at most 1024, not over the reply alone; every verified row is penalised for its own path, and a request with penalties takes lookup drafts and none from the model's draft block) | yes | essential |
| R04 | Repetition penalty with a window | llama, fork, ollama, kobold, vllm, sglang, tgi, trt, lmdeploy, strata, tabby | multiplicative penalty on seen tokens; window `repeat_last_n` or `penalty_range` (tabby adds a decay) | done (`repeat_penalty` / `repetition_penalty` with the window `repeat_last_n`; no decay) | yes | common |
| R05 | DRY | llama, kobold, tabby, sonar | penalty that grows with the length of a repeated sequence; breakers, allowed length, range | done (`dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`: llama.cpp's algorithm and defaults) | yes | niche |
| R06 | XTC | llama, ik, kobold, tabby, sonar | removes the most likely tokens with a given probability | done (`xtc_probability`, `xtc_threshold`) | yes | niche |
| R07 | Further truncation samplers | llama (typical, top-n-sigma), ik (top-n-sigma), kobold (top-a, TFS, typical), tabby (top_a, tfs, typical, skew, smoothing), trt (top-p decay, no-repeat n-gram) | more ways to cut the candidate set | partly: typical (`typical_p`) and top-n-sigma are done; top-a, TFS, skew, smoothing, top-p decay and no-repeat n-gram are not started; `min_keep` above 1 is refused | yes | niche |
| R08 | Mirostat | llama, kobold, tabby, sonar | perplexity-targeting sampler | done (mirostat 1 and 2 with `mirostat_tau`, `mirostat_eta`; its windows have one row, since `mu` moves with every pick) | yes | niche |
| R09 | Dynamic temperature, adaptive-p | llama, ik (adaptive-p), kobold (dynatemp), tabby (both) | temperature or target probability that moves with the distribution | partly: dynamic temperature is done (`dynatemp_range`, `dynatemp_exponent`); adaptive-p is refused (`adaptive_target`, `adaptive_decay`) | yes | niche |
| R10 | Sampler order | llama, kobold, tabby | the request or the server names the order of the filters | done (`samplers` as names or letters, llama.cpp's order when absent; penalties and DRY after a sampler that cuts, `infill` and `adaptive_p` are refused) | yes | niche |
| R11 | Logit bias | llama, strata, sglang, mlc, kobold, tabby, ik, vllm? | additive bias per token id (OpenAI's map); llama also bans with `false` and accepts strings; ik has a bias that expires | done (OpenAI's map and llama.cpp's list; a token id or a text, whose every token gets the bias; `false` bans a token; no bias that expires) | yes | common |
| R12 | Allowed and banned token ids, bad words | vllm, tabby, lmdeploy | restrict the draw to ids, or never draw ids or words | partly: token ids can be banned through `logit_bias` with `false`; no list of allowed ids and no bad words; the sampler takes a mask per row, which only the grammar hook (`TokenFilter`) fills | yes | niche |
| R13 | Banned strings with rewind | kobold, tabby, ik | when a banned string appears, go back and draw again without it | not started | yes | niche |
| R14 | Log probabilities | llama, vllm, sglang, tgi, trt, mlc, lmdeploy, strata, tfold, kobold | per token `logprob`, `bytes`, up to 20 alternatives; streamed too; equal with drafts on or off (tfold) | done (`logprobs` and `top_logprobs` on chat, `logprobs` on completions, `n_probs` with `post_sampling_probs`; per token `logprob`, `bytes` and up to 20 alternatives, streamed in chunks of their own; the model's own softmax or the distribution after the chain; the same with drafts on or off) | yes | common |
| R15 | Log probabilities of the prompt | vllm, tfold, tgi, sglang | `prompt_logprobs`, `echo`, `logprob_start_len`; gives perplexity over HTTP | not started | yes | niche |
| R16 | Several choices | llama, sglang, mlc, tgi, trt, vllm? | `n` above 1, `best_of` | partly: `n` (`n_cmpl`) up to 16, streamed too, as n requests with the seeds `seed`, `seed + 1`, ... that read the prompt once; no `best_of` (refused) | yes | niche |
| R17 | Beam search | vllm, trt | beams with length penalty | not started | yes | niche |
| R18 | Stop strings | all | the reply is cut before the match, also across chunk borders | done | yes | essential |
| R19 | Stop token ids, stop regex, keeping the stop text | vllm, sglang, lmdeploy | `stop_token_ids`, `stop_regex`, `include_stop_str_in_output` | not started | yes | niche |
| R20 | Output limit | all | `max_tokens`, `max_completion_tokens`; -1 or absent means the room left | done | yes | essential |
| R21 | Minimum length, ignoring the end token | llama, vllm, sglang, tfold, lmdeploy, tabby, kobold | `min_tokens`; `ignore_eos` for benchmarks | partly: `ignore_eos` is done; `min_tokens` is not started | yes | niche |
| R22 | Time limit for a request | llama | `t_max_predict_ms` | not started | yes | niche |
| R23 | Sampling defaults of the server | llama, fork, vllm, sglang, trt, strata, tfold, tabby, kobold, ollama | values for fields a request leaves out: flags or preset keys, `generation_config`, model-card sets (strata and tfold keep one set for thinking and one without) | not started (no option) | yes | essential |
| R24 | Forced overrides | tabby, kobold | the server's value wins over the request's | not started | yes | niche |
| R25 | Server default and cap for the output length | llama, fork, kobold, tgi | `n-predict`, `--genlimit`, `--max-total-tokens` | not started | yes | common |
| R26 | Output independent of batch composition | sglang, tfold, strata | the same text alone, next to others, with or without drafts (sglang `--enable-deterministic-inference`; tfold always; strata by switches) | partly: with `--precision exact` and steps of the same size; the fast kernels round differently | yes | common |
| R27 | Guidance (CFG), watermark | kobold (CFG), tgi, vllm, sglang (watermark) | negative prompt; statistical watermark | not started | yes | niche |
| R28 | Logits processors supplied by the user | vllm, sglang, trt, lmdeploy | Python code run on the logits | not started | no: one binary (no plugin code) | - |
| R29 | Token healing | tabby | re-tokenise the seam between prompt and reply | not started | yes | niche |

## 3. Request features: structured output

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| G01 | JSON object and JSON schema | llama, vllm, sglang, tgi, trt, mlc, ollama, tfold, lmdeploy, lmstudio, mistralrs?; strata by validation | `response_format` `json_object` or `json_schema` enforced while decoding | done: `response_format` (`json_object`, `json_schema` with `strict`), llama.cpp's `json_schema` and `schema` of `json_object`, on chat and on completions; the schema is always enforced by the grammar engine (llguidance) on the sampler's mask | yes | essential |
| G02 | Regular expression | sglang, tfold, lmdeploy, vllm?, tgi? | the reply matches a regex | not started | yes | niche |
| G03 | Context-free grammar | llama, kobold, localai (GBNF); sglang, tfold, vllm? (EBNF) | the reply follows a grammar | done: `grammar` in GBNF as llama.cpp reads it (left recursion works), on chat and on completions; syntax errors are 400s with line and column | yes | common |
| G04 | Choice | tfold, vllm? | the reply is one of the given strings | not started | yes | niche |
| G05 | Grammar on a trigger | vllm, sglang | free text until a tag opens, then the grammar of that tag (structural tags); the basis of strict tool calls | not started | yes | common |
| G06 | Grammar with thinking | tfold, sglang? | the grammar starts after the think block closes | done: a grammar, a schema or a `response_format` binds the text after the closing tag of the reasoning (free while it thinks); a template that writes no tag into the prompt lets the reply open the block or answer; thinking off binds the whole reply | yes | common |
| G07 | Grammar under drafts | tfold, vllm?, mlc? | drafts are cut at the first token the grammar refuses; tokens the grammar forces are appended as drafts | partly: a draft the grammar refuses is cut before it is verified, and every row of a window has the mask of its own path; tokens the grammar forces are not offered as drafts | yes | common |
| G08 | Completable within the limit | tfold | `grammar_budget`: never draw a token after which the value cannot close within `max_tokens` | not started | yes | niche |
| G09 | Bounded whitespace in JSON | sglang, tfold | a model cannot fill its reply with blanks inside a value | done for schemas and `json_object` (llama.cpp's white space: one space, or line breaks with an indent of up to 20); a GBNF grammar has the white space its author wrote | yes | niche |

## 4. Request features: tool calling

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | Tool calls | llama, ik, kobold, ollama, vllm, sglang, tgi, trt, lmdeploy, tabby, mistralrs, mlc, sonar, localai, strata, tfold, lmstudio | `tools` rendered by the template; the reply's calls parsed into `tool_calls` (id, name, JSON arguments), streamed as deltas, `finish_reason: tool_calls`; tool results accepted back | not started (the template renders tools and tool results; the engine refuses a request with tools) | yes | essential |
| T02 | Forced tool choice | llama, vllm, tfold, tgi?, mistralrs? | `tool_choice` `required` or a named function enforced by a grammar; `none` hides the tools | not started | yes | common |
| T03 | Parallel tool calls switch | llama, vllm, tfold | `parallel_tool_calls: false` returns at most one call | not started | yes | common |
| T04 | Strict argument schemas | vllm, mistralrs | arguments constrained to the tool's JSON schema (`strict`) | not started | yes | common |
| T05 | Recovery of imperfect calls | strata, tfold, vllm | read drifted envelopes, close markup that the end token cut, type arguments by the schema, repair JSON slips; anything else stays text | not started | yes | common |
| T06 | Calls inside thinking | tfold, tabby, strata | a complete call written before the think block closes is returned as the call | not started | yes | niche |
| T07 | Policy for undeclared tools | strata, tfold | a well-formed call of a tool the request did not offer: text plus a marker, or returned as a call | not started | yes | niche |
| T08 | Deterministic call ids | tfold | ids keyed from the request, so a replay names its calls the same | not started | yes | niche |
| T09 | Tools run by the server | llama, kobold, strata, mistralrs, localai, lmstudio, vllm, sglang | the server spawns MCP servers or built-in tools (files, shell, web search) and runs the loop itself | not started | yes | niche |
| T10 | Anthropic deferred tools | strata | `defer_loading` tools enter the prompt only where referenced, so the cached prefix stays | not started | yes | niche |

## 5. Request features: reasoning

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| K01 | Reasoning separated from the answer | llama, vllm, sglang, trt, strata, tfold, tabby, ollama, sonar, localai? | `reasoning_content` in the message and in stream deltas (Anthropic: thinking blocks; ollama: `thinking`) | not started (everything is `content`) | yes | essential |
| K02 | Thinking switch, effort and template variables per request | llama, vllm, strata, tfold, ollama, sglang?, tabby? | `reasoning_effort`, `chat_template_kwargs` (any template variable), `thinking`, `think` | partly: for the whole server with `--chat-var`; the request fields are refused | yes | essential |
| K03 | Thinking budget | llama, vllm, strata, tfold, tabby | a cap on thinking tokens; at the cap the server writes the closing marker (optionally a message) and the answer goes on | not started | yes | common |
| K04 | Room for the answer | strata, tfold | `max_tokens` counts the answer: thinking gets extra room or is wrapped up early | not started | yes | niche |
| K05 | Loop guard | tfold, strata, vllm, lmdeploy, tabby | a repeating cycle (in thinking or anywhere) closes the block or ends the reply | not started | yes | common |
| K06 | Earlier reasoning in the history | llama, fork, strata, tfold, sglang? | `reasoning_content` of earlier assistant turns is read and rendered (or dropped) the same way on every turn, so the cached prefix stays valid (`reasoning-preserve`, `clear_thinking`) | not started (the field is not read) | yes | common |
| K07 | Reasoning tokens in the usage | tfold | `completion_tokens_details.reasoning_tokens` | not started | yes | niche |

## 6. Request features: streaming, prompt formats, input kinds, context

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | Usage in the stream | llama, vllm, sglang, ollama, strata, tfold | `stream_options.include_usage` | done | yes | common |
| F02 | Keep-alive while the prompt is read | llama, vllm, tabby, strata | SSE comments at an interval | done (fixed interval) | yes | common |
| F03 | Prompt progress in the stream | llama, strata | `return_progress`: total, cached, processed, time | not started | yes | niche |
| F04 | Timings in every answer | llama, strata, tfold, ollama | prompt and generated tokens, times, rates, drafts proposed and accepted | done (under llama.cpp's names) | yes | common |
| F05 | Token ids and a hash of the reply | llama, vllm, tfold | `return_tokens`, `return_token_ids`, `token_sha` | not started | yes | niche |
| F06 | Chat template from the model file | llama, ollama, kobold, strata, tfold, tabby, vllm, sglang | the Jinja template with tools and the generation prompt | done (own Jinja dialect that errors on what it lacks) | yes | essential |
| F07 | Template override | llama, vllm, sglang, tgi, trt, tfold, tabby, kobold | a template file given at start replaces the model's | not started | yes | common |
| F08 | Template or prompt pieces from the request | vllm, trt, ollama, tabby? | `chat_template` in the request (behind a trust flag); ollama `template`, `system`, `raw` | not started | yes | niche |
| F09 | Continuing the last assistant message | llama, vllm, tabby? | the reply continues a given start (`--prefill-assistant`, `continue_final_message`, `/v1/continuations`) | not started | yes | common |
| F10 | Generation prompt and special tokens | vllm, sglang, tfold | `add_generation_prompt`, `add_special_tokens`, `skip_special_tokens` | not started | yes | niche |
| F11 | Image input | llama, fork, kobold, ollama, vllm, sglang, trt, tabby, mistralrs, lmdeploy, strata, tfold, sonar, localai, tgi?, mlc? | `image_url` parts (data URLs) through the model's projector | not started (non-text parts are refused) | yes | common |
| F12 | Image sources and limits | llama, sglang, trt, tfold, tabby, vllm? | remote URLs, local files under an allowed directory, a domain allow list, size and count limits | not started | yes | niche |
| F13 | Image token budget | llama, tfold, kobold | minimum and maximum tokens per image | not started | yes | niche |
| F14 | Audio and video input | llama, vllm, sglang, trt, mistralrs, tfold | `input_audio`, `input_video`, `video_url` parts | not started | no: newest model architectures only (the two test models take images only) | - |
| F15 | File and PDF input | tfold, mistralrs | `input_file` with an inline PDF | not started | yes | niche |
| F16 | An over-long prompt is refused, never cut | strata, tfold, llama? | 400 with `context_length_exceeded` and the counts | done | yes | essential |
| F17 | Output limit fitted to the room left | strata, tfold | a `max_tokens` past the window is shortened (strata's default) or refused (tfold for an explicit value) | done (shortened; the reply ends with `length`) | yes | common |
| F18 | Truncation and context shift | llama, kobold, vllm, tgi, sglang, ollama? | drop the oldest tokens or cut the prompt to N tokens (`--context-shift`, `n_keep`, `truncate_prompt_tokens`, `truncate`, `--allow-auto-truncate`) | not started | yes | common |
| F19 | Compaction by summary | tfold, localai | the server summarises the middle of a conversation that outgrows the window (`--compact-at`) | not started | yes | niche |
| F20 | Context beyond the trained length | llama, kobold, strata, tabby, tgi, lmdeploy | RoPE scaling, YaRN | not started | yes | niche |

## 7. Scheduling and state

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| S01 | Continuous batching | llama, fork, vllm, sglang, tgi, trt, lmdeploy, tabby, mistralrs, mlc, sonar, strata, tfold, ollama? | requests join and leave between steps; prompt rows and generation rows share a step | done (qwen35, qwen4exp) | yes | essential |
| S02 | Limit of concurrent sequences | llama, fork, ollama, vllm, sglang, tgi, trt, lmdeploy, tabby, mlc, strata, tfold | `--parallel`, `max_batch_size`, `--max-num-seqs` | done | yes | essential |
| S03 | Waiting instead of refusing | llama, ollama, tgi, vllm, sglang, tfold, strata | a request that does not fit waits for room | done (pages claimed per step) | yes | essential |
| S04 | Bounded queue and a refusal that says when to retry | ollama, tgi, vllm, sglang, tfold | limits on queued requests or tokens; 503 or 429 with `Retry-After` and a code | not started | yes | common |
| S05 | A prompt next to generation | fork, tfold, vllm, sglang, tgi, trt | a long prompt does not stall running replies; a share setting trades prompt speed for generation speed (`concurrent-prefill`, `--decode-share`, chunked prefill) | done: a prompt is read in pieces of its own between rounds of generation, which ride along with the newest token of every generating sequence; a share setting (`prefill_slowdown`, 2: generation gets half of the time while a prompt waits) is paid back by time; a short prompt is read whole; the setting is in the runtime and the server has no flag for it | yes | essential |
| S06 | Request priorities | vllm, sglang, tfold, triton? | a `priority` field or header; background work yields to foreground work; optional preemption | not started | yes | common |
| S07 | Fairness | tfold, tgi | a request that was passed over too often goes first | done | yes | common |
| S08 | Cache-aware queue order | sglang, vllm, trt | the request with the fewest uncached rows starts first | done | yes | niche |
| S09 | Giving way under memory pressure | sglang, tfold, vllm?, tokenspeed? | a running request is spilled, suspended or restarted so that others can run | partly: the admission object decides it; the conductor has no spill, and a request that gives way stays in the pool's cache | yes | common |
| S10 | Prefix cache across requests | llama, fork, ik, kobold, ollama, vllm, sglang, trt, lmdeploy, tabby, mistralrs, mlc, sonar, strata, tfold | a prompt that starts with computed text reuses its state; automatic, shared between sequences | partly: on for qwen35 in every placement; qwen4exp has no cache | yes | essential |
| S11 | Prefix cache for models with recurrent state | llama, fork, ik, sglang, vllm, trt, tabby, strata, tfold, tokenspeed? | checkpoints of recurrent state at intervals and boundaries; a hit restores one and replays the rest | partly: one checkpoint at a page boundary before the prompt's end, in page-locked RAM; none at message boundaries, no root at the end of the system prompt | yes | essential |
| S12 | Many conversations kept in host RAM | llama, fork, kobold, strata, tfold, tabby, sglang, trt | state that left the GPU is parked in RAM under a budget and restored (`cache-ram`, conversation cache, `sysmem_kv_cache`, `host_cache_size`) | partly: checkpoints live in RAM under `--ram`; K/V pages have no RAM tier | yes | common |
| S13 | Prompt cache on disk | fork, tfold, tabby, sglang, vllm, trt | evicted and idle state written to SSD with a size cap and a fingerprint; survives restarts and model swaps (`cache-disk`, `--prompt-cache-disk`, `disk_cache_dir`) | partly: `ares-store` exists and is tested; it is not connected to the devices | yes | common |
| S14 | Manual save and restore of a conversation | llama, fork, ik, strata, kobold | `POST /slots/{id}?action=save`, `restore` or `erase` with a file name | not started | yes | niche |
| S15 | Cache queries and maintenance | tfold, sglang, vllm, llama | how much of a conversation would resume and from where, preload it, forget everything (`/v1/imprint/*`, `/flush_cache`, `/reset_prefix_cache`) | not started | yes | niche |
| S16 | Retention control | strata, trt, tfold, vllm, llama | pin a prefix, priority and lifetime per token range, number of checkpoints, a request that opts out of caching | not started | yes | niche |
| S17 | Checkpoint placement policy | fork, strata, tabby, trt, sglang | a root at the end of a long system prompt, eviction that spreads checkpoints, intervals while generating and while reading (`checkpoint-root`, `checkpoint-spread`, `cache-save-min`) | partly: the policy is built in `ares-state`; the conductor anchors at a page boundary before the end of the prompt | yes | common |
| S18 | Cache isolation | vllm, trt | `cache_salt` limits sharing to requests with the same salt | not started | yes | niche |
| S19 | Reuse after an edit in the middle | llama, fork, kobold | cached K/V shifted over a removed span instead of recomputed (`cache-reuse`) | not started | yes | niche |
| S20 | Background preparation of the next turn | tfold | what the next turn will need is computed while idle | not started | yes | niche |
| S21 | Cached tokens reported | llama, vllm, sglang, strata, tfold, ollama | `usage.prompt_tokens_details.cached_tokens` | done | yes | common |
| S22 | A disconnect cancels | strata, tfold, llama?, vllm?, sglang?, tabby? | a closed connection frees the sequence at the next step | done | yes | essential |
| S23 | Timeouts | llama, strata, sglang | read and write timeout; a watchdog for a silent engine | not started | yes | niche |
| S24 | Idle unload | llama, ollama, kobold, sglang, vllm, strata, tfold, lmstudio, swap | VRAM is freed after N idle seconds and the model reloads on the next request (`--sleep-idle-seconds`, `keep_alive`, `--idle-unload`, TTL) | not started | yes | common |

## 8. Speed features

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| P01 | Drafts from the model's own head (MTP) | llama, fork, ik, avalon, strata, tfold, vllm, sglang, trt, tabby, sonar | the model's draft block proposes up to N tokens, verified in one window | partly: qwen35 on one device and under a tensor split; not under a layer split; qwen4exp has none | yes | essential |
| P02 | Draft model | llama, kobold, vllm, sglang, trt, tabby, mlc | a second, small model drafts | not started | yes | niche |
| P03 | Lookup drafts from the request's own text | llama, fork, ik, vllm, sglang, tgi, trt, tabby, strata | n-gram or prompt lookup continues an earlier occurrence | done: the lookup drafter, which takes part as soon as a model drafter does and never alone (so nothing speculates for qwen4exp, which has none) | yes | common |
| P04 | Lookup across requests | vllm, sglang, trt, tfold, ik | one corpus shared by all requests (suffix decoding, suffix automaton, recall), with a confidence per token | done: one suffix automaton shared by all requests, with a confidence per token, used by the conductor | yes | common |
| P05 | Lookup corpus on disk | llama, tfold, sglang | drafts from text of earlier runs or a given corpus (lookup cache files, recall store, external corpus) | not started | yes | niche |
| P06 | Block drafters | llama, fork, ik, vllm, sglang, trt, tfold, sonar | DFlash, DSpark, DFlash2, PARD: a small model reads the target's hidden states and proposes a block at once | not started (`--drafters FILE` is refused with "not built yet", the file is not opened; the choice of drafts takes any number of drafters) | yes | common |
| P07 | EAGLE and Medusa heads | llama, vllm, sglang, trt, mlc, tgi, sonar | trained draft heads on the target's features | not started | yes | niche |
| P08 | Draft trees | trt, tfold, sglang?, tokenspeed? | several branches verified in one pass, the accepted path kept | not started | yes | niche |
| P09 | Drafter and draft length chosen by measured gain | fork, vllm, sglang, tabby, tfold | per sequence and step, the draft with the most expected tokens per time (`spec-select`, dynamic speculation, `dynamic_draft`) | done: the drafts of all windows of a step are chosen together by expected tokens per millisecond, from a price of a step by its rows measured at load and kept up to date, and from acceptance measured per drafter, position and sequence; no setting | yes | common |
| P10 | Draft vocabulary | fork, tabby, sglang, tfold | the draft head projects onto a list of frequent tokens, as a view of the output head | done (`--draft-vocab`) | yes | common |
| P11 | Drafts that do not change the output | vllm, tfold, strata, trt? | sampled requests give the same distribution (vllm) or the same tokens (tfold) with drafts | done (the pick of a position is keyed by seed and position); with the fast kernels the text can still differ | yes | essential |
| P12 | Relaxed acceptance | vllm, trt, llama | drafts that are only close are accepted: faster, not exact (FLy, relaxed acceptance while thinking, `p-min`) | not started | yes | niche |
| P13 | Rejected drafts leave no trace in recurrent layers | fork, sglang, tfold, mistralrs? | snapshot on the device, replay at commit or a transition log, without a host round trip (`spec-ckpt-snapshot`) | done (verify window, then commit) | yes | essential |
| P14 | Quantized K and V | llama, fork, ik, kobold, ollama, avalon, strata, tfold, tabby, lmdeploy, vllm, sglang, tgi, trt, sonar | K and V stored below 16 bits, with separate formats for K and V and for the draft block | partly: `f16` and `q8` run; `q7` to `q4` are specified and not wired into a model | yes | essential |
| P15 | Attention that reads quantized K/V in place | fork, tabby?, lmdeploy? | no f16 copy of the cache (`flash-attn-native-quants`) | done (CUDA) | yes | essential |
| P16 | K/V of a running sequence beyond VRAM | llama, sglang, strata?, tabby? | a context larger than VRAM holds, the rest in RAM | not started (designed in the RAM broker) | yes | common |
| P17 | Sparse attention over selected cells | strata, tfold, ik, sglang, trt | for sparse-attention models only the selected cells are read | partly: runs in qwen4exp; kernel choice by measurement and many-row items are open | yes | essential |
| P18 | Every GGUF weight type on the GPU | llama, ik | all 28 types computed on the quantized data | partly: decode kernels for all 28 on CUDA and Vulkan; many-row kernels with a device layout for 12 | yes | essential |
| P19 | Expert tiers: VRAM, RAM, SSD | strata, avalon, ik, sglang, tfold | hot experts in VRAM slots, more in RAM the engine owns, the rest read from the file | partly: runs for qwen4exp behind the runtime and the server, the RAM tier within the one RAM limit; the tier does not give memory back, and experts are not read ahead of their routers | yes | essential |
| P20 | The CPU computes experts that are not on a GPU | llama, ik, kobold, avalon, strata, tabby | missed experts run on the CPU while the GPU runs the cached ones | partly: in the expert store of P19, with its limits | yes | essential |
| P21 | Expert election by heat, prefetch | strata, avalon, ik | the cached set follows use; the next experts are fetched ahead | partly: heat and the split of slots are used by qwen4exp; guesses and the token table are not used by a running model | yes | common |
| P22 | Expert pruning | ik, avalon, strata | low-ranked experts are skipped: speed for quality (SER, tail skip) | partly: the routing bias is built and not used | yes | niche |
| P23 | Expert statistics export | sglang, vllm | record and dump how often each expert is routed; routed experts returned with a reply | not started | yes | niche |
| P24 | Layer split across GPUs | llama, kobold, ollama, strata, avalon, tabby | each GPU holds a range of blocks | partly: qwen35 and qwen4exp run; the model's drafts do not run under a layer split | yes | common |
| P25 | Tensor split across unequal GPUs | fork, llama, ik, tabby; vllm, sglang, trt, tgi, lmdeploy, mlc with equal shards | each GPU holds a slice of every block, shares by speed and memory, sums over peer access or the host | done (qwen35; shares from the bandwidth the devices report) | yes | essential |
| P26 | Automatic placement | llama, ollama, kobold, ik, tabby, mistralrs, strata | devices, split, shares and context chosen from what the GPUs report (`--fit`, autosplit, `tune`) | done | yes | essential |
| P27 | Manual placement | llama, fork, tabby, kobold | devices, split kind and shares can be given; llama and fork also per tensor and per attention head (`-ot`, `attn-split`) | done (`--devices`, `--split`, `--shares`; no per-tensor override) | yes | common |
| P28 | Sampling on the device | fork, strata, tfold, trt, llama, tabby? | only picks or top candidates leave the GPU (`logits-topk`, backend sampling) | done | yes | essential |
| P29 | Speed against exactness | strata, tfold | a switch between the fastest arithmetic and exact arithmetic | done (`--precision`) | yes | common |
| P30 | Per-device tuning by measurement | mistralrs, strata, sglang, tfold | kernel choice per shape measured on the device and cached (`tune`, `--calibrate`, autotune) | partly: a table per device at load on Vulkan; on CUDA shares and kernel boundaries come from reported values and constants | yes | niche |
| P31 | Kernel and backend selection switches | llama, fork, strata, sglang, vllm | flags that choose attention, reduce or kernel variants (`flash-attn`, `allreduce`, `--attention-backend`) | not started | no: no opt-in for improvements (the better path is the only path) | - |
| P32 | Overlapped steps | sglang, vllm, trt | the next step is launched before the last one's results are read | not started | yes | niche |
| P33 | Fast loading | llama, avalon, sglang, strata? | direct I/O into page-locked staging, all devices at once | partly: built; load times are not measured | yes | common |
| P34 | Cold tiers re-encoded | trt | K/V pages re-encoded to fewer bits in the RAM and disk tiers | not started | yes | niche |

## 9. Model management

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| M01 | One GGUF loaded at start, split files included | llama, fork, ik, kobold, strata, avalon | `--model` names any part of a split file | done | yes | essential |
| M02 | Several models, loaded on request | llama, fork, ollama, kobold, tabby, localai, lmstudio, mistralrs, triton, swap | the `model` of a request selects a preset; the model is loaded if needed | not started (planned as a router mode) | yes | essential |
| M03 | Preset file | llama, fork, kobold, tabby, strata, swap | one section per model, a global section, precedence rules, keys that exist only in presets (`load-on-startup`, `stop-timeout`) | not started | yes | essential |
| M04 | Load and unload endpoints, load events | llama, strata, tabby, kobold, ollama, lmstudio, swap, mistralrs? | `POST /models/load` and `/unload`; an SSE stream of load state | not started | yes | common |
| M05 | Limit of loaded models and eviction | llama, ollama, lmstudio, swap | the least recently used idle model is unloaded to make room (`--models-max`, memory margin, auto-evict, swap matrix) | not started | yes | common |
| M06 | Model aliases | llama, sglang, trt, strata, tfold, tabby, localai, swap, vllm? | several ids for one model; tfold gives an id a default priority | not started | yes | common |
| M07 | Model download and cache management | llama, ollama, tabby, lmstudio, localai, mistralrs, strata | pull from a hub by name, list, delete | not started | yes | niche |
| M08 | Configuration file and environment variables | llama, vllm, sglang, trt, tabby, strata, kobold, tfold | every option also as a file key or a variable | not started (flags only, and two variables) | yes | common |
| M09 | Settings changed without a restart | localai, kobold, llama, tfold, swap | runtime settings, config reload, key file reload | not started | yes | niche |
| M10 | LoRA adapters | llama, kobold, vllm, sglang, tgi, trt, tabby, mistralrs, lmdeploy, sonar, nim? | adapters loaded at start, switched per request, loaded while running | not started | no: GGUF model files are the only input the frame names | - |
| M11 | Control vectors | llama, ik, strata | steering vectors added to layers | not started | no: GGUF model files are the only input the frame names | - |
| M12 | Metadata overrides | llama, kobold, sglang, lmdeploy | keys of the model file overridden at load | not started | yes | niche |
| M13 | Weights replaced without a restart | sglang, vllm | the served model's weights reloaded from disk | not started | yes | niche |

## 10. Memory and VRAM

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| V01 | Memory planned before the model runs | llama, tfold | every byte of VRAM is booked; a configuration that does not fit is refused at load with the list of holders | done | yes | essential |
| V02 | Context from free memory | llama, ollama, strata, tfold | the default context is what the VRAM holds | done | yes | common |
| V03 | One pool of state for all sequences | vllm, sglang, llama, tabby | nothing is reserved per slot | done | yes | essential |
| V04 | VRAM reserve for other programs | llama, tabby, strata, tfold, vllm | N MiB per GPU stay free (`--fit-target`, `autosplit_reserve`, `--vram-reserve-mib`) | not started | yes | common |
| V05 | VRAM lent and given back | strata, avalon, fork | buffers follow the work: workspace sized by the live context, compute buffers lent to the expert cache between phases, room given to other programs | partly: by design there are no per-shape buffers (one scratch area per step, booked in the ledger); nothing is given back to other programs | yes | niche |
| V06 | One limit for host RAM | strata, tfold | the most RAM the engine may use | done (`--ram`) | yes | common |
| V07 | Container limits | strata, tfold | cgroup limits are read | not started | yes | niche |
| V08 | Memory report | ik, tfold, strata, ollama | `/memory`, bytes per device in the metrics, `size_vram` | partly: a report at start and pool gauges; no bytes per device | yes | common |
| V09 | Behaviour under host memory pressure | tfold, strata | new work is refused with a code and caches are shed | partly: a broker with priorities exists; only the checkpoints use it | yes | niche |

## 11. Hardware and backends

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| H01 | CUDA | llama, fork, ik, kobold, ollama, vllm, sglang, tgi, trt, lmdeploy, tabby, mistralrs, mlc, sonar, strata, avalon, tfold | every card CUDA 13 supports | partly: Ampere and Ada measured; other classes compiled and untested; Turing has no prompt and no attention path | yes | essential |
| H02 | Vulkan | kobold, mlc, llama? | the same features on any GPU with Vulkan | partly: weight and layer kernels at CUDA's speed, attention kernels slower than CUDA's; no sampler (to be rebuilt against `ares-sampler`), expert or qwen4exp kernels, no model | yes | essential |
| H03 | Several GPUs on Vulkan | llama? | a model split over Vulkan devices | partly: sums go through host pages; no direct route works on the test driver | yes | common |
| H04 | Pure CPU inference, dense layers on the CPU | llama, ik, kobold, ollama, mistralrs, localai | a model or part of a dense model runs on the CPU (`-ngl`) | not started | no: no CPU inference | - |
| H05 | Windows and macOS | llama, kobold, ollama, strata, tfold, mistralrs, mlc, lmstudio | native builds | not started | no: Linux only | - |
| H06 | Vendor backends beyond CUDA and Vulkan | strata (HIP, SYCL, Metal), tfold (Metal, HIP, XPU), avalon (ROCm), mlc (Metal, OpenCL), vllm, sglang, sonar | native AMD, Intel and Apple paths | not started | no: CUDA and Vulkan only | - |
| H07 | Other weight formats | vllm, sglang, tgi, trt, tabby, mlc, mistralrs, tfold, lmdeploy | safetensors, AWQ, GPTQ, FP8, NVFP4, EXL3, compiled model libraries | not started | no: GGUF only | - |
| H08 | Many model families | all | dozens of architectures, old ones kept | not started | no: newest model architectures only | - |
| H09 | Several machines | llama, vllm, sglang, trt, tgi, lmdeploy, localai, tfold, strata, mistralrs | RPC, multi-node, prefill and decode on different servers, routers, P2P | not started | no: one to a few GPUs in one machine | - |
| H10 | Datacenter parallelism | vllm, sglang, trt, lmdeploy, sonar | data, expert, pipeline and context parallelism, expert load balancing | not started | no: one to a few consumer or workstation GPUs | - |
| H11 | CPU thread and NUMA placement | llama, strata, tabby | thread counts, core lists, NUMA for the CPU's share of the work | not started | yes | niche |
| H12 | Unified-memory machines | strata, tfold | DGX Spark: one memory for CPU and GPU, sized from the host's RAM | not started | yes | niche |

## 12. Observability

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| O01 | Latency histograms | vllm, sglang, strata, tfold, tgi?, trt? | time to first token, per token, end to end, queue and prompt time | not started | yes | common |
| O02 | What the server is doing now | llama, strata, tabby, tfold, ollama, vllm | running and waiting requests, tokens processed, free capacity (`/slots`, `/status`, `/v1/status`, `/v1/load`, `/api/ps`, `/load`) | partly: gauges in `/metrics` only | yes | common |
| O03 | Logs: level, JSON, file | llama, tgi, vllm, sglang, tabby | structured logs for collectors | partly: the level through `RUST_LOG`; no JSON or file option | yes | common |
| O04 | Request logging | llama, vllm, sglang, tabby, strata, tfold | prompts, parameters and outputs on request; a finish line with tokens, cache hit, seed | partly: one line per request with id, model and limits | yes | common |
| O05 | Distributed tracing | vllm, sglang, tgi, trt, localai | OpenTelemetry spans per request | not started | yes | niche |
| O06 | Draft statistics | llama, vllm, strata, tfold | proposed and accepted, per request and per draft position | partly: totals and counts per request; not per position | yes | niche |
| O07 | Profiling | vllm, sglang, trt, mlc, tfold | a profile started and stopped over HTTP; GPU time per kernel class | not started | yes | niche |
| O08 | Request ids | vllm, tfold? | an id given by the client is echoed in headers and logs | partly: unique ids in the answer; none accepted, none in headers | yes | niche |
| O09 | Dashboards | tfold, strata, tabby, swap | a live page | not started | no: no GUI | - |
| O10 | Metrics under another engine's names | strata, tfold | vLLM's and llama.cpp's metric names, so existing dashboards work | not started | no: no compatibility | - |

## 13. Operations and security

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| X01 | API keys | llama, vllm, sglang, tgi, tabby, strata, tfold, kobold, localai, lmstudio, swap | bearer keys from a flag, a variable or a file; health stays open | not started | yes | essential |
| X02 | Key roles | tabby, sglang, tfold | admin against inference against read-only; labels; reload of the key file | not started | yes | niche |
| X03 | TLS | llama, vllm, sglang, kobold, nim? | certificate and key, refreshed without a restart | not started | yes | niche |
| X04 | CORS | llama, vllm, tgi, mlc, strata, tabby, ollama | allowed origins, methods, headers | not started | yes | common |
| X05 | Host and Origin checks | strata, tfold, ollama | a local server refuses foreign `Host` names and web origins (DNS rebinding) | not started | yes | common |
| X06 | Rate and concurrency limits | tfold, kobold, tgi, triton | per key or address: requests per second, requests in flight; 429 | not started | yes | niche |
| X07 | Request size limit | tgi, kobold | a body above the limit is refused | done (64 MiB) | yes | common |
| X08 | Graceful shutdown | tgi, tfold, llama, strata? | the first signal lets requests finish, the second ends them; devices are released | done | yes | essential |
| X09 | Container image and service unit | llama, tgi, strata, tfold, mistralrs, lmdeploy, vllm?, sglang?, trt?, localai? | a published image; a systemd unit or guidance | not started | yes | common |
| X10 | No network access by the server | llama, ollama | `--offline`, `OLLAMA_NO_CLOUD` | done (the server never connects out) | yes | common |
| X11 | Guard against non-finite logits | strata | a window with NaN logits ends the request and drops the cached state | not started | yes | niche |

## 14. Quality tools

| ID | Feature | Engines | What it is | Ares | Frame | Need |
| --- | --- | --- | --- | --- | --- | --- |
| Q01 | Exactness that can be checked through HTTP | tfold | per request: a serial reference (`draft:false`), a hash of the reply's tokens, `exact_mode` in the reply | partly: `"draft": false` per request and a `token_hash` of the reply's tokens in the final message and the last event of a stream; `--precision exact` for the whole server; no `exact_mode` in the reply | yes | niche |
| Q02 | Benchmark commands | llama, ik, kobold, trt, mistralrs, strata, tfold, sglang? | `llama-bench`, `sweep-bench`, `trtllm-bench`, `tune` | not started | no: no command line tools | - |
| Q03 | Evaluation commands | llama, trt, sglang? | perplexity, `trtllm-eval` | not started | no: no command line tools (R15 gives perplexity over HTTP) | - |
| Q04 | Quantize and convert tools | llama, ik, tabby, ollama, mlc? | making model files | not started | no: no command line tools | - |
| Q05 | Synthetic acceptance for draft benchmarks | llama, vllm | drafts accepted at given rates, to measure the verify path alone | not started | yes | niche |
| Q06 | Checks of the model file | llama, sglang | tensor values or a checksum checked at load | partly: the file's structure is validated at open; values are not | yes | niche |

## Counts

Counted from the tables above with a script (see the end of this file for the command).

| | Rows |
| --- | --- |
| Total | 223 |
| Inside the frame | 200 |
| - done | 56 |
| - partly | 39 |
| - not started | 105 (essential 6, common 32, niche 67) |
| Outside the frame | 23 (all not started) |

Of the 56 done rows 26 are essential, 24 common and 6 niche; of the 39 partly rows 11 are essential, 14 common
and 14 niche. A row is one capability, so the counts weigh a sampler option like a model router; List 1 orders
them by need instead.

## List 1: inside the frame, not started, most depended-on first

**Essential (6).** Without these a current agent client cannot use the server at all.

- T01 tool calls (parse, stream, results back)
- K01 reasoning separated from the answer
- R23 sampling defaults of the server
- M02 several models loaded on request, and M03 the preset file
- X01 API keys

**Common (32).**

- API: A04 Responses, A07 Anthropic Messages, A08 token counting, A09 tokenize, A16 model properties
- Sampling: R25 server cap for the output length
- Structured output: G05 grammar on a trigger
- Tools: T02 forced tool choice, T03 parallel switch, T04 strict schemas, T05 recovery of imperfect calls
- Reasoning: K03 thinking budget, K05 loop guard, K06 earlier reasoning in the history
- Prompt: F07 template override, F09 continuing the assistant message, F11 image input, F18 truncation
- Scheduling: S04 bounded queue with a retry answer, S06 priorities, S24 idle unload
- Speed: P06 block drafters (DFlash, DSpark), P16 K/V of a running sequence beyond VRAM
- Models: M04 load and unload endpoints, M05 limit and eviction, M06 aliases, M08 configuration file
- Memory: V04 VRAM reserve
- Observability: O01 latency histograms
- Operations: X04 CORS, X05 Host and Origin checks, X09 container image and service unit

**Niche (67).** A03, A05, A06, A10, A14, A20, A23 to A27; R13, R15, R17, R19, R22, R24, R27, R29; G02, G04, G08; T06 to T10; K04, K07; F03, F05, F08, F10, F12, F13, F15, F19, F20; S14 to S16, S18 to S20, S23; P02, P05, P07,
P08, P12, P23, P32, P34; M07, M09, M12, M13; V07; H11, H12; O05, O07; X02, X03, X06, X11; Q05.

The rows that are `partly` are not in this list; the largest of them are Vulkan (H02: no sampler, expert
kernels and no model, against the frame's "full parity"), the mixture-of-experts model without a prefix cache
(S10) and the open parts of its tiers (P19 to P21), the low-bit K/V formats (P14), the RAM and disk tiers of the
cache (S12, S13), the per-request thinking switch (K02) and the prompt next to generation (S05).

## List 2: outside the frame, in many engines (for the user to decide)

| Row | What | Who has it | Excluded by |
| --- | --- | --- | --- |
| H07 | Other weight formats (safetensors, AWQ, GPTQ, FP8, NVFP4, EXL3) | every serving engine except the llama.cpp family | GGUF only |
| H08 | Many model families | all | newest architectures only |
| H04 | Pure CPU inference and dense layers on the CPU | llama.cpp family, ollama, mistralrs, localai | no CPU inference |
| H05 | Windows and macOS | llama.cpp family, ollama, strata, tfold, mistralrs, mlc, lmstudio | Linux only |
| H06 | Native AMD, Intel and Apple backends | strata, tfold, avalon, mlc, vllm, sglang | CUDA and Vulkan only |
| H09, H10 | Several machines; data, expert and pipeline parallelism | vllm, sglang, trt, tgi, lmdeploy, localai, llama (RPC) | one to a few GPUs in one machine |
| M10 | LoRA adapters (static, per request, hot-loaded) | llama, kobold, vllm, sglang, tgi, trt, tabby, mistralrs, lmdeploy, sonar | no scope decision names adapters; the frame names GGUF model files as the only input |
| M11 | Control vectors | llama, ik, strata | same |
| A12, A13 | Embeddings, rerank, classification | nearly all | needs other model architectures |
| A21, A22 | Speech, transcription, realtime audio; image and video generation | llama, vllm, ollama, kobold, localai, tfold | other model kinds |
| F14 | Audio and video input | llama, vllm, sglang, trt, mistralrs | the test models take images only; revisit with Qwen4 |
| A19 | Native request dialects, Ollama and KoboldAI emulation | llama, kobold, ollama, sglang, tgi, sonar | no compatibility |
| A29, O09 | Web UI, dashboards | llama, kobold, strata, tfold, mistralrs, tabby, localai, swap | no GUI |
| Q02 to Q04 | Benchmark, evaluation, quantize and convert commands | llama, ik, trt, sglang, mistralrs, strata, tfold | no command line tools |
| R28 | Logits processors and plugins supplied by the user | vllm, sglang, trt, lmdeploy | one binary |
| P31 | Kernel and backend selection switches | llama, fork, strata, sglang, vllm | no opt-in for improvements |
| O10 | Metrics under vLLM's or llama.cpp's names | strata, tfold | no compatibility |

Three inside-frame rows sit on the border and deserve the same decision: T09 (tools and MCP servers run by the
server: an application layer, not inference), M07 (model download: the frame says models "come as GGUF files")
and A11 (decisions: a second kind of request that five engines added in 2026).

Seen in single engines and outside the frame without doubt: TensorFold's "Living Weights" (facts learned into
the served weights) and its two-Mac speed-up mode; Strata's desktop and terminal apps, `strata connect`
adapters and its MCP server for managing the engine; LocalAI's agents, connectors and model failover; KoboldCpp's
story saves and multiplayer; tunnels (kobold, tgi) and usage statistics sent home (tgi, trt).

## List 3: where engines disagree, and Ares must pick

1. **A field the server does not serve.** Refuse by name (Ares today; tfold for many fields) or ignore it
   (llama and most OpenAI-compatible servers; tfold accepts and ignores a named list on Responses `include`).
2. **The `model` of a request.** Must name a served id (Ares answers 404 `model_not_found`; routers route by it),
   is ignored (strata: one server, one model, because clients send names of their own), or matches an alias
   (llama `--alias`, tfold, strata `aliases`).
3. **Order of the filters.** `min_p` after top-k and top-p (sglang, tfold) or before them (vllm, trt). llama's
   default chain is penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature, and it is
   configurable; tabby has `temperature_last` as an option. Ares: llama's default order, configurable through
   `samplers`.
4. **What a penalty counts.** The tokens of the reply only (tfold; strata since its change), or the last N tokens
   of prompt and reply (llama's `repeat_last_n`; strata's old behaviour behind `STRATA_PENALTY_PROMPT=1`); trt has
   `prompt_ignore_length`. Under drafts: each verified row for its own path (strata, tfold), or penalties refused
   with trees (trt). Ares: the last N tokens of prompt and reply, each verified row for its own path.
5. **What a log probability is.** The model's distribution at temperature 1 before any filter (tfold; llama's
   default) or after the sampler chain (llama `post_sampling_probs`). With thinking: the answer's tokens only
   (tfold) or the thinking's tokens under `reasoning_content` as well (strata). Ares: the model's own distribution
   unless `post_sampling_probs` asks for the one after the chain.
6. **A `max_tokens` past the window.** Shortened to the room left (strata's default; Ares today) or refused with
   400 when the client gave it explicitly (tfold; strata with `fit_max_tokens: false`).
7. **A prompt longer than the context.** Refused (strata, tfold, Ares), cut to N tokens from the left or the
   right (vllm, tgi, ollama `truncate`), or shifted during generation (llama `--context-shift`, kobold, ollama
   `shift`). Shifting cached K/V does not carry recurrent state, so for the target models only refusal or
   recomputation are exact.
8. **How JSON output is enforced.** Constrained decoding (llama, vllm, sglang, tgi, trt, mlc, tfold) or a schema
   in the prompt, validation afterwards and 502 on failure (strata). Grammar dialects differ: GBNF (llama,
   kobold), EBNF through xgrammar (vllm, sglang, tfold), GBNF and JSON schema through llguidance (Ares; Lark and regular expressions are in the engine
   and have no request field).
9. **A grammar and thinking.** The grammar starts after the think block closes (tfold; Ares; llama-server for
   `response_format` and `json_schema`), or thinking has to be off (llama-server for a plain `grammar`, which binds
   from the first token). Ares binds all three after the reasoning; `enable_thinking: false` binds the whole reply.
10. **`tool_choice: required`.** Enforced by a grammar (vllm, llama, tfold's native server), by replacing tokens
    (tfold's Python server), or not supported (ollama's OpenAI route lists `tool_choice` as unsupported).
11. **A call of a tool the request did not declare.** Text plus `strata_rejected_tool_calls` (strata's default),
    or returned as a tool call so the client can answer the model (tfold; strata with
    `undeclared_tool_calls: "call"`).
12. **A malformed call.** Returned as content (tfold), dropped (vllm's `hf` parser), or repaired within stated
    limits (strata's `tool_call_recovery`, on by default).
13. **Name of the reasoning field.** `reasoning_content` (llama, strata, tfold), `thinking` (ollama, Anthropic's
    blocks). Which effort names exist and how an unnamed level maps to the template's levels: nearest level
    (tfold), a map given at start (vllm `--reasoning-effort-map`).
14. **Does `max_tokens` include thinking.** Yes (OpenAI's rule, the default everywhere), or the answer gets its
    own room (strata `answer_room`, tfold `--thinking-reserve`). At a thinking budget: the closing marker alone
    (tfold) or a wrap-up line first (strata; llama `--reasoning-budget-message`).
15. **Earlier reasoning in the prompt.** Kept on every turn (fork's `reasoning-preserve = true`, tfold for
    GLM-5.3) or cleared as the template does by default; it decides whether a new user message invalidates the
    cached prefix.
16. **A full queue.** Wait without a bound (llama, Ares) or answer 503 or 429 with `Retry-After` (ollama, tgi,
    tfold).
17. **Priority.** A number, lower first (vllm; sglang can flip it), or two classes, foreground and background
    (tfold).
18. **Drafts under sampling.** The same tokens as without drafts (tfold; Ares with exact products), the same
    distribution only (vllm's rejection sampler), or acceptance that is deliberately loose (vllm FLy, trt).
19. **Idle unload.** A lifetime per request (ollama `keep_alive`), per model (lmstudio and llama-swap TTL), or one
    setting of the server (llama `--sleep-idle-seconds`, tfold `--idle-unload`).
20. **Responses API state.** `store` defaults to true as at OpenAI (tfold, vllm) or the route is stateless
    (llama, ollama, strata).
21. **Fetching image URLs.** On by default (llama), off unless a flag allows it (tfold `--vision-urls`, tabby
    `disable_fetch_requests`, sglang's allowed domains).
22. **What invalidates state on disk.** A fingerprint of checkpoint, kernel build, OS build and chip (tfold), of
    the model files and the settings that change what the cells mean (strata), or of the model and its flags
    (fork).

## Sources

Read on 2026-11-11. "Local" means files on this machine; everything else was fetched from the web through a
summarising fetch.

| Engine | Source | Notes |
| --- | --- | --- |
| Ares | `docs/ARCHITECTURE.md`, `README.md`, `crates/server/src/{cli,wire,routes,error,lib,main}.rs`, `crates/runtime/src/conductor.rs` (grep), `docs/reference/{engines-survey,serving-engines-code-survey}.md` | local, read only |
| fork | its `models.ini`; Ares' ARCHITECTURE.md | local; the fork's repository was not read |
| llama | `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md` (both halves); the same README in Avalon's tree, `Avalon's tools/server/README.md`, read directly | the two agree |
| avalon | `Avalon's README.md` (commit e09e775, 2026-11-11) | the notes file the survey notes of Avalon did not exist when this was written |
| ik | `https://raw.githubusercontent.com/ikawrakow/ik_llama.cpp/main/README.md` | README only; its server README was not read |
| kobold | `https://github.com/LostRuins/koboldcpp/wiki` (characters 0 to 200000 of the page) | sampler request fields beyond the names in the wiki were not verified |
| ollama | `https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md`, `https://docs.ollama.com/faq`, `https://docs.ollama.com/api/openai-compatibility` | Anthropic compatibility was not checked |
| vllm | `https://docs.vllm.ai/en/latest/serving/online_serving/`, `.../serving/online_serving/openai_compatible_server/`, `.../cli/serve/`, `.../features/`, `.../features/speculative_decoding/`, `.../features/tool_calling/` | "latest" docs, footer date 2026-11-11 |
| sglang | `https://docs.sglang.io/advanced_features/server_arguments.html` (both halves), `.../basic_usage/sampling_params.html`, `.../basic_usage/native_api.html` | the docs index (`llms.txt`) could not be read; HiCache details come from the flag names only |
| tgi | `https://huggingface.co/docs/text-generation-inference/reference/launcher`, `.../conceptual/guidance`, `https://raw.githubusercontent.com/huggingface/text-generation-inference/main/README.md` | image tag 3.3.5; in maintenance mode |
| trt | `https://nvidia.github.io/TensorRT-LLM/` (index, updated 2026-11-03, commit 9e334fb), `commands/trtllm-serve/trtllm-serve.html`, `features/kvcache.html`, `features/speculative-decoding.html`, `features/sampling.html` | |
| triton | `https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html` | one page; it names the rate limiter, the response cache, version policy and warm-up, and does not describe priorities |
| nim | search results over `docs.nvidia.com/nim/large-language-models/` (1.11 and 2.0.x) | the configuration page returned no text; treat every `nim` as weakly verified |
| lmdeploy | `https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html`, `.../api/pipeline.html` | |
| tabby | `https://raw.githubusercontent.com/theroyallab/tabbyAPI/main/config_sample.yml`, `.../sampler_overrides/sample_preset.yml`, `https://theroyallab.github.io/tabbyAPI/` (endpoint list), `https://raw.githubusercontent.com/turboderp-org/exllamav3/master/README.md` | request fields beyond the sampler preset were not verified |
| mistralrs | `https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/README.md` (v0.9.5) | its HTTP reference page was not found; grammar kinds and samplers are from Ares' notes |
| mlc | `https://llm.mlc.ai/docs/deploy/rest.html` | |
| sonar | `https://raw.githubusercontent.com/dphnAI/sonar/main/README.md`, a web search | sampler names (DRY, XTC, Mirostat) from the search summary of the README |
| localai | `https://localai.io/docs/features/` (footer 2026-11-04) | titles and one-line descriptions only |
| strata | `Strata's docs/{DETAILS,AI_REFERENCE,CAPABILITIES}.md` (release 0.2.13, engines 0.1.59) | local, read only; about 100 further documents were not read |
| tfold | TensorFold's `docs/api.md` (whole file), flag names from its README, RUNBOOK and docs; Ares' notes the survey notes of TensorFold | local; version 1.2.7 |
| tokenspeed | Ares' own notes (`engines-survey.md`, `serving-engines-code-survey.md`); `https://raw.githubusercontent.com/lightseekorg/tokenspeed/main/README.md` states none of the serving features | every `tokenspeed?` is from Ares' notes |
| swap | `https://raw.githubusercontent.com/mostlygeek/llama-swap/main/README.md` | |
| lmstudio | `https://lmstudio.ai/docs/developer/core/ttl-and-auto-evict` and the titles of its sidebar | |

**What is inferred and not verified.** Every name with `?`. That `fork` has the rows of `llama` it is not named
in. Ares' state where the documents disagree with each other (P04) or where the code was only searched, not read
(F17 was read in `conductor.rs`; R23: whether sampling metadata of the GGUF is used was not checked). The `Need`
column throughout.

**What this survey did not cover.** Request-level details of KoboldCpp's and TabbyAPI's own APIs; NIM and Triton
beyond one page each; MLC's engine configuration (prefix cache modes); mistral.rs' HTTP reference; SGLang's
OpenAI-route details (structured output back ends, parsers by name); LM Studio beyond one page; the code of any
web-fetched engine.

Counts were made with:
`awk -F'|' 'NF==9 && /^\| [A-Z][0-9][0-9] /{n++; a=$6; f=$7; s=(a ~ /^ done/)?"done":(a ~ /^ partly/)?"partly":"not"; if (f ~ /^ yes/) y[s]++; else o++} END {print n, y["done"], y["partly"], y["not"], o}' docs/reference/feature-checklist.md`
