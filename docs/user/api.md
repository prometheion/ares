# HTTP API

Ares speaks the OpenAI chat, completion and Responses APIs, Anthropic's Messages
API, and llama.cpp's `/props`, `/tokenize` and `/detokenize`. A request is served
in full or refused: a field the server does not serve is a 400 that names it,
never dropped. On the chat and completion routes this holds for the members of
the body; members inside a message, a tool or `stream_options` that the server
does not read are passed over. `null` counts as "not set".

## Routes

| Method | Path | What it is |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok"}`; open without a key |
| `GET` | `/v1/models` | every model: id, aliases, status, the defaults of its requests |
| `POST` | `/v1/chat/completions` | OpenAI chat, whole or streamed |
| `POST` | `/v1/completions` | OpenAI completion of a text prompt, whole or streamed |
| `POST` | `/v1/messages` | Anthropic Messages, whole or streamed |
| `POST` | `/v1/responses` | OpenAI Responses, without state |
| `POST` | `/v1/chat/completions/input_tokens`, `/v1/messages/count_tokens`, `/v1/responses/input_tokens` | count the prompt of a request of that API; nothing is generated |
| any | `/v1/responses/{id}`, `/v1/responses/{id}/{what}` | 404: no response is kept |
| `GET` | `/props` | what a model is (llama.cpp's shape); `?model=<id>` with several models |
| `POST` | `/tokenize`, `/detokenize` | text to tokens and back (llama.cpp's shapes) |
| `GET` | `/metrics` | counters and histograms in the Prometheus text format; `?model=<id>` with several models |
| `POST` | `/models/load`, `/models/unload` | `{"model": "<id>"}`: load or unload a model; answers `{"success": true}` at once |
| `GET` | `/models/sse` | Server-Sent Events: the state of every model, then every change (`{"model", "event": "model_status", "data": {"status", "reason"}}`) |
| `POST` | `/v1/systemone` | the decision route of Jev mode; answers 501 until it is built |

Every generating request names its `model`: an id or an alias. An unknown model
is a 404 `model_not_found` that lists the ids. A model that is not loaded is
loaded, and the request waits; with a preset file the model idle the longest
makes room, and a model that serves requests is never unloaded under them.
`/props`, `/metrics`, `/tokenize` and `/detokenize` may leave the model out when
the server has one.

| Status | When |
| --- | --- |
| 400 | a field the server does not serve, a wrong value, a prompt longer than the context (`context_length_exceeded`), an image without `--mmproj` |
| 401 | a key is missing or wrong (with `WWW-Authenticate: Bearer`, except on `/v1/messages`) |
| 403 | a `Host` or a web page's `Origin` the server does not answer to (see [Keys and access](#keys-and-access)) |
| 404 / 405 | an unknown route or model / a wrong method |
| 408 / 413 | the body did not arrive within `--timeout` / the body is larger than 64 MiB |
| 503 | `overloaded` with `Retry-After: 1` (the `--queue` is full), `model_load_failed`, `model_not_loaded` (`/metrics`, and `/tokenize` and `/detokenize` without a `model`), the server is stopping |

Errors have OpenAI's shape `{"error": {"message", "type", "param", "code"}}`,
and Anthropic's `{"type": "error", "error": {"type", "message"}}` on `/v1/messages`.

## Chat and completions

`/v1/chat/completions` and `/v1/completions` take the same fields, except the
prompt and those marked "chat".

| Field | Meaning | Notes |
| --- | --- | --- |
| `messages` | chat: the conversation | roles `system` (or `developer`), `user`, `assistant`, `tool`; `content` a string or parts `text` and `image_url`; also `name`, `tool_call_id`, `tool_calls`, `reasoning_content` |
| `prompt` | completions: the text | one string (or an array of one string); token ids and several prompts are refused |
| `max_tokens`, `n_predict`; chat: `max_completion_tokens` | the most tokens of the reply | `-1` is no limit of its own; `0` is refused; held to `--max-tokens-limit`; of several, `n_predict` holds, then `max_completion_tokens` |
| `stop` | stop strings | a string or an array; not empty |
| `stream`, `stream_options.include_usage` | Server-Sent Events; a usage chunk at the end | |
| `return_progress` | a stream says how far the prompt has got (llama.cpp's member) | see [Prompt progress](#prompt-progress); changes nothing without `stream` |
| `sse_ping_interval` | the seconds of silence after which a stream sends a ping (llama.cpp's member) | whole seconds; `0` or `-1`: no pings; default 15 |
| `timings_per_token` | the last chunk of every step of a stream has `timings`, the reply's so far (llama.cpp's member) | no hash and no drafts before the end; changes nothing without `stream` |
| `n`, `n_cmpl` | completions of one prompt | 1 to 16; the prompt is read once; of both, `n_cmpl` holds |
| `logprobs`, `top_logprobs`; `n_probs`, `post_sampling_probs` | probabilities of the tokens, in OpenAI's or llama.cpp's form | chat: `logprobs: true` and 0 to 20 alternatives; completions: `logprobs` is the number |
| `priority` | the request's priority | see [Priorities](#priorities) |
| `draft` | `false` turns drafting off for this request | default `true` |
| `continue_final_message`, `add_generation_prompt` | chat: continue the last assistant message, or begin a new turn | a last assistant message is continued by default, unless it calls tools |
| `user`, `metadata`, `store`, `service_tier`, `prompt_cache_key`, `safety_identifier` | bookkeeping | accepted, without effect |

The sampling fields are the fields of the options in
[options.md](options.md#defaults-of-request-fields), with their ranges and
defaults (`temperature`, `top_k`, ..., `samplers`, `seed`), and:

| Field | Notes |
| --- | --- |
| `typical`, `repetition_penalty` | other names of `typical_p` and `repeat_penalty` |
| `top_k`, `seed`, `repeat_last_n` | `top_k` `0` or `-1` is no cut; `seed: -1` draws a seed of its own; a window of `-1` (the whole context) is refused |
| `samplers` | also llama.cpp's string of letters |
| `logit_bias` | an object (OpenAI) or a list of `[token, bias]` pairs (llama.cpp); a token is an id or a text; `false` bans it; at most 1024 |
| `ignore_eos`, `min_keep` | the reply does not end at the end token; a `min_keep` above 1 is refused |

Also refused: `best_of` other than 1, `echo: true`, `suffix`, the legacy
`functions` and `function_call`, adaptive-p (`adaptive_target`,
`adaptive_decay`), and tools and thinking fields on `/v1/completions`, which has
no chat template.

The answer has OpenAI's shape and more: `usage.prompt_tokens_details.cached_tokens`
(the part of the prompt the prompt cache gave), `usage.completion_tokens_details.reasoning_tokens`,
`timings` (llama.cpp's fields: `cache_n`, `prompt_n`, `prompt_ms`, `predicted_n`,
`predicted_ms`, their milliseconds per token and rates,
`draft_n` and `draft_n_accepted`, the request's own drafts, and `token_hash`, a
hash of the reply's token ids; `prompt_ms` runs from the request's start in the
engine, after its wait in line, to its first token, as llama.cpp's does), and `stop_reason` with the stop string that ended
the reply. A request that asked for more tokens than `--max-tokens-limit` gets the
limit, and its response the header `x-ares-max-tokens-limit` with it.

## Messages

`/v1/messages` is served as the chat request that asks the same.

| Field | Notes |
| --- | --- |
| `model`, `max_tokens` | both required (`max_tokens` not on `count_tokens`); `max_tokens` at least 1 |
| `temperature`, `top_p`, `top_k`, `stream` | as on the chat route; other sampling fields are refused |
| `system` | a string or text blocks |
| `messages` | `user`, `assistant` and `system` turns, a string or blocks. User: `text`, `image`, `tool_result` (text and images; `is_error` puts `Error: ` before the text). Assistant: `text`, `thinking`, `redacted_thinking` (kept out of the prompt), `tool_use`. A later `system` turn becomes a user turn for templates that take the system message first only. A last assistant turn is a prefill, which the reply continues |
| `tools` | `name`, `description`, `input_schema`; `type: "custom"`, `strict` and `eager_input_streaming` are accepted |
| `tool_choice` | `auto`, `any`, `tool` with `name`, `none`; `disable_parallel_tool_use` |
| `thinking` | see [Thinking](#thinking) |
| `stop_sequences` | stop strings; `stop_reason` is `stop_sequence` and `stop_sequence` names it |
| `sse_ping_interval` | llama.cpp's member: the seconds of silence before a `ping` event; `0` or `-1`: none |
| `output_config` | `format: {"type": "json_schema", "schema": ...}`; `effort` is the reasoning effort |
| `context_management` | the edits `clear_thinking_20251015` and `clear_tool_uses_20250919`, as Anthropic's service makes them; the response's `context_management.applied_edits` lists those that changed the prompt, with what they cleared; `count_tokens` adds `context_management.original_input_tokens`, the count before the edits |
| `cache_control` | accepted on the body, blocks and tools, without effect: every prefix is cached anyway |
| `metadata`, `service_tier` | accepted, without effect |

Refused: `document` blocks, citations, tools that Anthropic's service runs
(`web_search_...`, `bash_...`) and their blocks, `mcp_servers`, `container`,
`defer_loading: true`, images from a URL, and other edits of `context_management`.
The usage has `input_tokens` (the prompt less what the prompt cache gave),
`cache_read_input_tokens` (from the prompt cache), `cache_creation_input_tokens`
(always 0) and `output_tokens`.

## Responses

`/v1/responses` is served as the chat request that asks the same. It keeps no
state: every response says `"store": false`.

| Field | Notes |
| --- | --- |
| `model`, `temperature`, `top_p`, `stream`, `priority`, `return_progress`, `sse_ping_interval`, `timings_per_token` | as on the chat route; other sampling fields are refused |
| `instructions` | a system message |
| `input` | a string, or items: messages (`user`, `assistant`, `system`, `developer`; parts `input_text`, `output_text`, `input_image`), `function_call`, `function_call_output`, `reasoning` |
| `tools`, `tool_choice`, `parallel_tool_calls` | functions only; `tool_choice` `auto`, `none`, `required`, `{"type": "function", "name"}` |
| `max_output_tokens` | at least 1; a reply it cuts is `incomplete` |
| `reasoning` | `effort`; `summary: "auto"` is accepted, no summary is written |
| `thinking_token_budget` | a budget of thinking tokens (vLLM's field) |
| `text.format` | `text`, `json_object`, `json_schema` |
| `include` | `reasoning.encrypted_content` is accepted and adds nothing |
| `truncation` | `disabled` (default): too long is an error; `auto`: the oldest whole turns leave until it fits |
| `stream_options` | `include_obfuscation: false` only |
| `store`, `background` | `false` only |
| `metadata`, `user`, `safety_identifier`, `service_tier`, `client_metadata`, `prompt_cache_key`, `prompt_cache_retention`, `prompt_cache_options` | accepted, without effect |

The usage has `input_tokens` (the whole prompt), `input_tokens_details.cached_tokens`,
`output_tokens`, `output_tokens_details.reasoning_tokens` for a model that thinks,
and `total_tokens`; the response carries `timings` as on the chat route.

Refused: `previous_response_id`, `conversation`, `item_reference`, `prompt`
templates, `top_logprobs`, `max_tool_calls`, tools that are not functions,
`input_file`, `input_audio`, images by `file_id`, `text.verbosity`,
`reasoning.summary` `concise` and `detailed`, and the chat route's own fields
(`messages`, `max_tokens`, `response_format`, `reasoning_effort`, `top_k`, ...).

## Priorities

A request has a priority, a whole number from -2147483648 to 2147483647; a lower
number goes first; the default is 0. It is the body's `priority` on the chat,
completion and Responses routes, or the header `x-ares-priority` on every route
that generates (for clients that cannot add a field, such as Claude Code with
`ANTHROPIC_CUSTOM_HEADERS`). A body and a header that differ are refused.

| Place | What a better priority gets |
| --- | --- |
| the line | it is taken first |
| a full device | running requests of a worse priority give way; their state goes to the prompt cache and they resume later |
| a step | its rows go first; a prompt that outranks every generating request is read at once |
| a full `--queue` | it takes the place of the worst waiting request of a worse priority (which gets 503), else it is refused itself |

## Thinking

| Control | Chat and completions | Messages | Responses |
| --- | --- | --- | --- |
| on or off | `chat_template_kwargs.enable_thinking`; `reasoning_effort: "none"` is off | `thinking.type`: `enabled` or `adaptive` on, `disabled` off | `reasoning.effort: "none"` is off |
| effort | `reasoning_effort` | `output_config.effort` | `reasoning.effort` |
| budget of tokens | `reasoning_budget_tokens`, `thinking_budget_tokens` or `thinking_token_budget` (`-1`: the model's default), `reasoning_budget_message` | `thinking.budget_tokens` | `thinking_token_budget` |
| format | `reasoning_format`: `auto` or `none` | thinking blocks | a `reasoning` item |

The defaults come from `--reasoning`, `--reasoning-effort`, `--reasoning-format`
and `--reasoning-budget-tokens`. An effort is a 400 when the chat template does not
read one, and a level the template refuses is a 400, except `high`, which then
means the template's default (Flash-Next's template knows `low`, `medium` and
`xhigh`, its default); Messages takes `high` always, as the template's default.
`"none"` is a 400 when the template cannot switch thinking off. A budget is a bound: when it is
spent, the budget's message and the end of the thinking are written and the
answer follows. A budget on a model without thinking is refused. Thinking that
loops is ended the same way, and one token repeated 256 times ends a reply.
Other `chat_template_kwargs` are template variables; `messages`, `tools`,
`add_generation_prompt`, `bos_token` and `eos_token` are the server's.

Send the reasoning back (`reasoning_content`, thinking blocks, reasoning items):
the next prompt then begins with the cached one. On Messages,
`thinking.display: "omitted"` sends thinking blocks with an empty text whose
`signature` carries the thinking; `summarized` and `updates` get it whole.

## Tool calls

The reply is held to the tools by a grammar: the name is one of the request's
tools, the arguments follow the tool's JSON schema (string values are not
checked against `pattern`, `minLength`, `maxLength` or `format`).

| `tool_choice` | The reply |
| --- | --- |
| `auto` (default) | free until the model opens a call; then calls only |
| `required` (Messages `any`) | calls from the first token of the answer, at least one |
| a named function | the same, every call of that function |
| `none` | text; the tools stay in the prompt |

`parallel_tool_calls: false` allows one call. A call the model opens inside its
thinking ends the thinking and is returned as a call. Calls stream as OpenAI
deltas, Anthropic `input_json_delta` or Responses argument deltas;
`finish_reason` is `tool_calls`. Tools for a model whose template writes calls in
a form the server cannot read are a 400 unless `tool_choice` is `none`, and so are
tools next to a structured-output field.

## Structured output

| Field | Routes | Meaning |
| --- | --- | --- |
| `grammar` | chat, completions | a GBNF grammar |
| `json_schema` | chat, completions | a JSON schema; `{}` is any object |
| `response_format` | chat, completions | `text`; `json_object` (with an optional `schema`); `json_schema` with `json_schema.schema` (always enforced, as `strict: true`) |
| `output_config.format` | Messages | `{"type": "json_schema", "schema": ...}` |
| `text.format` | Responses | `text`, `json_object`, `json_schema` with `schema` |

`grammar` with another of them is refused. `json_schema` next to a
`response_format` gives way to the format's schema, as in llama-server (with
`json_object` without a `schema` it holds). The thinking stays free; the rule
binds the answer after it.

## Images

Images need `--mmproj`; without it they are refused.

| Route | Form |
| --- | --- |
| chat | a part `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` (the URL may also be a plain string, or base64 alone) |
| Messages | a block `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`, in user turns and tool results |
| Responses | a part `{"type": "input_image", "image_url": "data:image/png;base64,..."}`; `detail` is accepted |

JPEG, PNG, GIF (its first frame), WebP and BMP are read; an alpha channel is
dropped, and no EXIF orientation or color profile is applied, as in llama.cpp.
An image has at most 16384 pixels a side and 64 megapixels, within the 64 MiB of
a body. Remote URLs are not fetched. One image becomes at most
`--image-max-tokens` tokens; a larger one is shrunk.

## Streaming

| API | Events |
| --- | --- |
| chat, completions | `data:` chunks with deltas (`reasoning_content`, `content`, `tool_calls`), then `data: [DONE]` |
| Messages | `message_start`, `content_block_start`, its deltas and `content_block_stop` per block, `message_delta` with the usage, `message_stop`; `ping` while waiting |
| Responses | `response.created`, `response.in_progress`, the items' events, `response.completed` or `response.incomplete`; every event has a `sequence_number` |

A stream starts after the engine's first step, so a refused request still gets
an HTTP status, or after 10 seconds without it. A failure later is an error event.
A stream that has sent nothing for 15 seconds sends a ping (a comment line `:`,
on Messages a `ping` event); `sse_ping_interval` sets the seconds, `0` or `-1` none,
as llama.cpp's member does (its server's default is 30).

### Prompt progress

`return_progress: true` on a streamed `/v1/chat/completions`, `/v1/completions` or
`/v1/responses` request asks for llama.cpp's `prompt_progress` before the first
text: `{"total", "cache", "processed", "time_ms"}`, all tokens of the prompt, those
the prompt cache gave, those done with the cached ones, and the milliseconds since
the request started in the engine (its wait in line not counted). A client shows
`(processed - cache) / (total - cache)`, as llama.cpp's web UI does.

| API | Where it is |
| --- | --- |
| chat | a chunk with the role and no text (`delta: {"role": "assistant", "content": ""}`) |
| completions | a chunk with `text: ""` |
| Responses | a `response.in_progress` event |

The first comes when the request starts in the engine (`processed` is `cache`),
one after every step that read rows of the prompt, and the last, with
`processed` equal to `total`, just before the first text; `processed` rises with
every one. That is when llama.cpp's server sends them, per batch of the prompt.
What Ares adds: the count of what is done follows the work, not the finished
rows. The rows a prompt that the cache holds runs again to rebuild the model's
state from a checkpoint are work too, so `processed` moves during them; and
Flash-Next, which reads a long prompt layer by layer as one step (a pass), says
how far the pass has got, by its layers and the pieces of each, about once a
second while it runs: `processed` is then `cache` plus that share of the
prompt's tokens, and moves through the pass where a count of finished tokens
would jump from nothing to all at its end. `/v1/messages` has no progress, as in
llama.cpp (Anthropic's API has no such member; it is refused there).

## Other routes

| Route | Fields and answer |
| --- | --- |
| `/v1/models` | `{"object": "list", "data": [...]}`; each model has `id`, `aliases`, `status` (`{"value": ...}`: `unloaded`, `loading`, `loaded`, `unloading` or `failed`, with a `reason` for a failure or a load that waits for room), `defaults` (what a request gets for the fields it leaves out), `max_tokens_limit` if set, and `created` and `context_length` from its last load |
| `/props` | `model`, `build_info` (what the binary is, as `ares --version` prints it), `loaded`, `context_length`, `default_generation_settings` (`params`, `n_ctx`), `chat_template`, `chat_template_caps` (tools, reasoning, effort, `enable_thinking`, thinks by default), `reasoning` (its tags), `modalities` (`vision` true when the model was loaded with `--mmproj`, or its preset names one; `audio` false); a model that is not loaded is described from its file without a load |
| `/tokenize` | `content` (a text, read as a completion prompt) or a chat request (`messages` and the chat fields); `parse_special: false` and `add_special` are refused; answers `{"tokens": [...]}`, with `with_pieces` `{"tokens": [{"id", "piece"}, ...]}` (the text, or its bytes where a token is not whole characters) |
| `/detokenize` | `tokens`; answers `{"content": "..."}`; control tokens write nothing |
| `/metrics` | the histograms `ares_request_queue_seconds`, `ares_request_prompt_seconds`, `ares_time_to_first_token_seconds`, `ares_time_per_output_token_seconds`, `ares_request_seconds`, and counters and gauges of requests, tokens, drafts, the prompt cache in RAM and on the SSD, and Flash-Next's experts |
| `/models/load`, `/models/unload` | an unload takes new requests away and unloads the model when its running requests have ended; the next request loads it again. Also with one `--model` (see [options.md](options.md#the-preset-file) for several) |

## Keys and access

| Check | Rule |
| --- | --- |
| key | with `--api-key` or `--api-key-file`, every route but `/health` wants `Authorization: Bearer <key>` or `x-api-key: <key>` |
| `Host` | a server on a loopback address without keys answers only to `localhost` and to addresses; `--allowed-hosts` adds names |
| `Origin` | without keys, a request a browser sends from a page of another site is refused; local pages, pages of the server's own name or of an `--allowed-hosts` name, and `--cors-origins` pass |
| CORS | answers to an origin of `--cors-origins` carry the CORS headers, and its preflight is answered without a key |

With keys, the key decides: pages are not checked, and names only when
`--allowed-hosts` names hosts. A server that listens beyond this machine without
keys warns at start. [features.md](features.md) lists what Ares can do.
