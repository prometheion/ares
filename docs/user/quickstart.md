# Quickstart

Ares serves GGUF models on one or two NVIDIA cards under Linux, over the OpenAI
and Anthropic HTTP APIs. This page goes from the source to a first answer.

## What you need

| Item | Needed for |
| --- | --- |
| Linux, with an NVIDIA driver for the GPU | building and running |
| An Ampere (RTX 30) or Ada (RTX 40) card | running; other GPUs are refused, see [hardware.md](hardware.md) |
| Rust 1.95 or newer (`rust-toolchain.toml` picks the stable channel) | building |
| CUDA toolkit 13.0 or newer, for `nvcc` | building only, not running |
| `glslc`, the shader compiler of the Vulkan SDK or of the shaderc package | building the Vulkan shaders |

## Build

```sh
cargo build --release
```

The binary is `target/release/ares`. The release build uses fat link-time
optimization and takes a while.

The build environment:

| Variable | Meaning |
| --- | --- |
| `ARES_CUDA_ARCHS` | Compute capabilities to build machine code for, such as `89` for one RTX 40 card. Default: the tested classes, `86,89`. A class that is not tested can be named on purpose; the build warns. Classes below Ampere are an error. |
| `NVCC` | The CUDA compiler. Default: `$CUDA_PATH/bin/nvcc`, `$CUDA_HOME/bin/nvcc`, `nvcc` on `PATH`, `/opt/cuda/bin/nvcc`, `/usr/local/cuda/bin/nvcc`. |
| `GLSLC` | The shader compiler. Default: `glslc` on `PATH`, `$VULKAN_SDK/bin/glslc`, `/usr/bin/glslc`, `/usr/local/bin/glslc`. |
| `ARES_BUILD_COMMIT` | The commit `ares --version` names when the source is no git checkout (a `git archive` snapshot, a copy without `.git`). In a checkout git says it. Default: `unknown`. |
| `SOURCE_DATE_EPOCH` | The date of the build `ares --version` names, in seconds since 1970, for a build that is to come out the same twice. Default: the day of the build. |

## Run a model

Qwen3.8 27B, with everything left to the server:

```sh
target/release/ares --model ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf
```

The server picks the devices, the split and the context from what the GPUs
report, and the log says what it chose. It answers on `127.0.0.1:8000` once
the model is loaded; a model that fails to load ends the start.

Qwen3.8 Flash-Next, a split GGUF file, with its MTP block from a file of its own,
40 GiB of RAM for the experts and the prompt cache, and the prompt cache on the SSD:

```sh
target/release/ares --model ~/models/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf \
  --drafters ~/models/mtp-q2_0.gguf --ram 40G --cache ~/.cache/ares-prompts
```

Name any part of a split model. [models.md](models.md) has the measured setups.

## A first request

A model's id is its file name without `.gguf`: `Qwen3.8-27B-UD-Q4_K_S` above, and
`Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004` for the split file. Every request
names its model. `GET /v1/models` lists the ids.

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-UD-Q4_K_S",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

The reasoning comes back in `message.reasoning_content`, the answer in
`message.content`. Add `"stream": true` for Server-Sent Events. Without sampling
fields the reply is greedy.

## Claude Code

Point Claude Code at the Messages API and give every model name it uses the id
of a model:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_API_KEY=unused
export ANTHROPIC_DEFAULT_OPUS_MODEL=Qwen3.8-27B-UD-Q4_K_S \
  ANTHROPIC_DEFAULT_SONNET_MODEL=Qwen3.8-27B-UD-Q4_K_S \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-UD-Q4_K_S
export API_TIMEOUT_MS=1800000  # a first prompt of 16k tokens can take minutes on Flash-Next
claude --model Qwen3.8-27B-UD-Q4_K_S
```

A preset file gives a model a short id and aliases. With `--api-key`, set
`ANTHROPIC_API_KEY` to one of the keys.

## Logs and metrics

| What | Where |
| --- | --- |
| The log | standard error; `--log-file <FILE>` appends to a file, `--log-format json` writes one JSON object a line |
| Log level | the `RUST_LOG` environment variable, default `info` (for example `RUST_LOG=debug`) |
| At start | the devices, the split, the context and why, the RAM limit, and what each device took |
| Per request | lines for a person, laid out as llama.cpp's server writes its timings: when it is taken (API, model, the parameters it gives), when it starts in the engine (the prompt, what the cache gave, the wait), how far a long prompt has got (from 3 s on, every tenth or 10 s), how fast a long reply comes (`n_gen`, `tg` since its first token and `tg_3s` over the last 3 s, once in 3 s from its 100th token), and at its end the reason, `prompt eval time` (over the rows the engine ran for the prompt, the cached tokens named apart), `eval time`, `total time` and `draft acceptance` (accepted of proposed, mean length and mean chain); the same as fields at `RUST_LOG=ares_server=debug`; the prompt and the reply only with `--log-text` |
| Errors | every refusal and failure a client gets, with its route, status, code and message: a warning, a 500 an error |

`GET /metrics` returns the counters of the loaded model in the Prometheus text
format (`?model=<id>` when the server has several models). Useful ones:

| Metric | Meaning |
| --- | --- |
| `ares_time_to_first_token_seconds`, `ares_time_per_output_token_seconds`, `ares_request_queue_seconds` | histograms: time to the first token, per token after it, and in the line |
| `ares_prompt_tokens_total`, `ares_generated_tokens_total` | prompt tokens computed, tokens generated |
| `ares_cached_cells_total`, `ares_prefix_hits_total` | prompt tokens taken from the prompt cache, and how often |
| `ares_drafts_proposed_total`, `ares_drafts_accepted_total` | draft tokens tried and accepted |
| `ares_expert_hits_total`, `ares_expert_misses_ram_total`, `ares_expert_misses_disk_total` | Flash-Next: picked experts found in VRAM, in RAM, only on the SSD |

Next: [options.md](options.md) (every option, the environment, the config and
preset files), [models.md](models.md), [hardware.md](hardware.md),
[api.md](api.md) (routes and fields) and [features.md](features.md).
