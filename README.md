# Ares

An inference server for large language models on one to a few consumer GPUs,
written in Rust. It reads GGUF models, runs them through CUDA on Linux and
serves them over the OpenAI and Anthropic HTTP APIs.

> **Ares is experimental.** There is no release. Options, the shape of the API,
> the files it keeps on disk and the code itself change without notice. Run it
> beside a server you trust, such as llama.cpp, not in place of one.

## Why Ares

Ares exists for six reasons, and every piece of work is weighed against them:

1. **Control over the features.** The engine does what its owner needs, and no
   feature waits for another project.
2. **Speed**, always.
3. **Parallel requests.** llama.cpp serves several requests at once poorly, and
   vLLM needs a lot of VRAM to do it well.
4. **Efficient use of VRAM and RAM.**
5. **Large mixtures of experts streamed from the SSD:** a model larger than RAM,
   in a 4- or 5-bit quantization, without a large loss of speed.
6. **Room for experiments:** new ways to read experts ahead, to draft, to read
   prompts and to prune experts.

It supports the newest model architectures only and keeps no options for
compatibility: llama-server's option names are not accepted.

The first statement of the goals is
[issue 1](https://github.com/prometheion/ares/issues/1) (2026-11-09). Some of
it has been decided differently since, for example the GPUs CUDA runs on;
[`docs/OPEN.md`](docs/OPEN.md) holds the current decisions.

## What it runs

**Models.** Two architectures, as GGUF files in any of the 28 GGUF weight types:

| Model | Architecture | Kind |
| --- | --- | --- |
| Qwen3.8 27B | `qwen35` | dense, with its MTP block |
| Qwen3.8 Flash-Next | `qwen4exp` | mixture of experts; the experts live in VRAM, in RAM and on the SSD |

Images go through the model's projector (`--mmproj`, the `mmproj` GGUF beside
the model). Other architectures are refused. The small Qwen3.5 0.8B has the
27B's architecture and is used by the tests.

**GPUs.** CUDA on NVIDIA Ampere (compute capability 8.6, RTX 30) and Ada (8.9,
RTX 40), the two classes the engine was tested on. A GPU of another class is
refused when the device is opened, and the server skips it. Blackwell (12.0,
RTX 50) follows once it has been tested. Every other GPU is meant to run
through the Vulkan backend later: its kernels exist, but no model runs on it
yet. Ares was tested with one card and with two. Between cards, data goes
directly from device to device when the driver allows peer access, and through
host memory otherwise.

**System.** Linux only. The engine reads weights with io_uring and direct I/O,
keeps the prompt cache on disk with direct I/O, and loads the driver's
`libcuda.so.1` at run time.

## What it does

A short list. [`docs/FEATURES.md`](docs/FEATURES.md) has one line for every
feature.

- **API routes:** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
  and the Anthropic Messages API (`/v1/messages` and its token count), so Claude
  Code works against Ares; `/v1/models`, `/health`, `/metrics` (Prometheus),
  and llama.cpp's `/tokenize`, `/detokenize` and `/props`.
- **Tools and structured output:** tool calls held to the tools' schemas by a
  grammar, streamed; `response_format` with a JSON schema; GBNF grammars;
  thinking with a budget of tokens.
- **Images** in all three API styles.
- **Prompt cache,** automatic and shared by all requests: checkpoints at the
  end of the system prompt and tools and at every message boundary, kept in
  RAM inside `--ram` and on the SSD with `--cache`. A conversation resumes after
  a restart.
- **Parallel requests and priorities:** `--parallel` sequences at once, a
  `priority` per request (or the header `x-ares-priority`), a bounded line
  with `--queue`.
- **Drafts:** the model's MTP block, an n-gram drafter and DFlash2 drafters.
  How many drafts a step verifies is chosen from the measured price of a step
  and the acceptance. On the 27B the text with drafts is token for token the
  text without.
- **Experts across VRAM, RAM and SSD** (Flash-Next): slots in VRAM, a tier in
  RAM, the rest read from the model file on the SSD. The CPU computes experts
  that are missing on the card. A prompt read alone passes through the model
  layer by layer, so that each expert crosses the link once.
- **One limit for host memory:** `--ram` covers all page-locked memory, and
  the engine gives memory back when other programs need it.
- **Several models** from one preset file, loaded when a request names them.

In work: K and V of a running sequence beyond VRAM, in RAM (Flash-Next), and
Flash-Next models larger than RAM streamed from the SSD.

## Measured

The two tables come from runs of both engines in the same session on the same
machine, alternating, with the setup named. They are measurements of two cards
and one model, not general claims.

**The 27B against llama.cpp at production settings.** Qwen3.8 27B UD-Q4_K_S on
an RTX 4070 and an RTX 3060 (12 GiB each) with peer access, 2026-11-23. The
other engine is a llama.cpp fork with quantized flash attention, tensor split
with peer-to-peer sums and MTP drafts, run with its production preset. Both
engines at the same settings: tensor split 0.625 / 0.375, 262144 cells of K and
V for 2 sequences, K and V in 5 bits (`q5_1` in the fork, `q5` in Ares), MTP and
n-gram drafts, greedy sampling. The fork ran 2 rounds, Ares 3.

| | llama.cpp fork | Ares |
| --- | --- | --- |
| Decode, German / English / code, tokens/s | 61-63 / 67-68 / 76-77 | 71-74 / 78-86 / 103-104 |
| Prompt of 2k / 8k / 32k / 98k tokens, tokens/s | 840 / 994 / 944 / 752 | 1135 / 1030 / 654 / 325 |
| Agent turn, first token, first / second ask | 0.5-7.2 s / 0.1-0.5 s | 0.2-5.9 s / 0.1-0.2 s |
| VRAM at the end, RTX 4070 / RTX 3060 | 11.6 / 11.7 GiB | 11.5 / 10.1 GiB |

Later the same day, the attention of long prompts got a kernel of its own. That
build ran alternating with the build before it (two rounds each, the same
settings): prompts of 2k / 32k / 98k tokens at 1131 / 1000-1002 / 709-710
tokens/s, against 1082-1103 / 653-655 / 324 before. The fork did not run in
that session. Against its numbers above, 98k tokens are still about 6 % behind.

**Parallel requests.** The same cards, model and settings, 2026-11-23, before
the attention kernel above. Every request has its own topic, greedy sampling,
MTP on. The fork ran with its preset's 2 slots, Ares with `--parallel 2` unless
the row says otherwise. Two rounds each, alternating.

| | llama.cpp fork | Ares |
| --- | --- | --- |
| One request alone, German / English / code, tokens/s | 58-65 / 60-64 / 73-76 | 72-74 / 63-75 / 91-103 |
| Two at once, sum while both run, German / English / code, tokens/s | 98 / 104-108 / 116-129 | 105-111 / 93-116 / 106-134 |
| 3 / 4 / 8 at once, sum, tokens/s | 86-87 / 100-109 / 95-97 | 115 / 107 / 110 (the round on a quiet machine; 70 / 83 / 81 during other builds) |
| 8 at once with `--parallel 8`, sum, tokens/s | | 204-206 |
| 8 at once, slowest first token | 19.3 s | 17.6 s; 0.51 s with `--parallel 8` |
| A short request beside two generations, first token | 24.5-25.8 s | 0.15-0.22 s with priority 0; 18-21 s at the same priority |
| Two agents with the same 16.6k tokens of system prompt and tools, cold, first turns | 22.3 / 47.7 s | 19.9 / 19.9 s |
| Small cached turns of an agent | 0.23-1.1 s | 0.15-0.25 s |
| An agent turn with a 5.8k-token tool result at 17k tokens, alone / both agents at once | 6.6-7.6 s / 9-17 s | 10.0-10.2 s / 12.1 and 26-29 s |
| A prompt beside a generating request, first token: 8k / 32k / 98k tokens | 15.2-15.5 / 59.8-59.9 / 209-219 s | 20.4 / 124.1-124.4 / 754-763 s |

The last two rows are where Ares is behind. It gives a long prompt and a
generating request turns, where the fork runs both at the same time; it reads
two prompts one after the other; and before the attention kernel it read deep
in a long context more slowly. The work on this is listed in
[`docs/OPEN.md`](docs/OPEN.md).

Comparisons of Flash-Next with Strata, the engine it is measured against, are
not listed until they have been run in the same session.

## Build

You need:

- Linux, with an NVIDIA driver for the GPU.
- Rust 1.95 or newer (`rust-toolchain.toml` picks the stable channel).
- The CUDA toolkit 13.0 or newer, for `nvcc`. It is needed to build, not to run.
- `glslc`, the shader compiler of the Vulkan SDK or the shaderc package, for
  the Vulkan shaders.

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

## Run

```sh
target/release/ares --model ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf
```

Without more options the server chooses the devices, the split and the context
from what the GPUs report, and the log says what it chose. It listens on
`127.0.0.1:8000`. The model's id is the file name without `.gguf` (here
`Qwen3.8-27B-UD-Q4_K_S`); `/v1/models` lists it.

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-UD-Q4_K_S",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

The settings of the measurements above, for two 12 GiB cards:

```sh
ares --model Qwen3.8-27B-UD-Q4_K_S.gguf \
  --split tensor --shares 0.625,0.375 \
  --context 262144 --parallel 2 --kv q5 \
  --drafters mtp --draft-vocab draft-vocab.txt \
  --ram 16G --cache ~/.cache/ares-prompts --cache-size 64G
```

Flash-Next with its MTP block from a file of its own, and 40 GiB of RAM for the
experts and the prompt cache:

```sh
ares --model Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf \
  --drafters mtp-q2_0.gguf --ram 40G --cache ~/.cache/ares-prompts
```

The main options (`ares --help` lists them all):

| Option | Meaning |
| --- | --- |
| `--model <FILE>` | The GGUF file (any part of a split model). |
| `--models-preset <FILE>` | Several models from an INI file, one section each (below). |
| `--host`, `--port` | Where to listen. Default: `127.0.0.1`, `8000`. |
| `--devices <LIST>` | CUDA devices by the driver's index, such as `0,1`. Default: all. |
| `--split single\|tensor\|layer`, `--shares <LIST>` | How the model is spread over the devices, and each device's share. |
| `--context <TOKENS>` | K/V cells for all sequences together. Default: what the free VRAM holds. |
| `--parallel <N>` | Sequences that run at once. Default: 2. |
| `--kv <FORMAT>` | K and V in `f16`, `q8`, `q7`, `q6`, `q5` or `q4`, or two of them (K,V). Default: `q8`. |
| `--ram <SIZE>` | The most host RAM the engine may use. Default: all the machine has. |
| `--cache <DIR>`, `--cache-size <SIZE>` | The prompt cache on the SSD. Default: none; 32G. |
| `--drafters <LIST>` | `mtp`, `none`, or the file of an MTP block or a DFlash2 drafter. The n-gram drafter takes part with them. |
| `--draft-vocab <FILE>` | Token ids the model drafters may propose, one per line. |
| `--mmproj <FILE>` | The image projector. Without it, images are refused. |
| `--api-key <KEYS>`, `--api-key-file <FILE>` | Keys a request must carry. Default: none. |

Every option is also an environment variable, `ARES_` and its name in capitals
(`ARES_CONTEXT=32768`), and a line of the file of `--config`. The command line
holds over the environment, and the environment over the file. `RUST_LOG` sets
the log level. `ARES_LIBCUDA` names a driver library to load in place of the
system's `libcuda.so.1`, for example a build of the driver with peer access on
consumer cards.

**Several models.** `--models-preset` serves the models of an INI file. A
model is loaded when a request names it, the one used least recently makes
room, and an idle model can be unloaded:

```ini
[*]
ram = 24G
idle-unload = 30m

[qwen-27b]
model = /models/Qwen3.8-27B-UD-Q4_K_S.gguf
aliases = default
load-on-startup = true

[flash-next]
model = /models/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
drafters = /models/mtp-q2_0.gguf
```

**Claude Code.** Point it at the Messages API, and give every model name it
uses the id of a model, here `qwen-27b` of the preset above:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_API_KEY=unused
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen-27b ANTHROPIC_DEFAULT_SONNET_MODEL=qwen-27b \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen-27b
export API_TIMEOUT_MS=1800000  # a first prompt of 16k tokens can take minutes on Flash-Next
claude --model qwen-27b
```

## Tests

```sh
cargo test --workspace
```

Tests that open GPUs are off by default, because opening a device takes VRAM
that a running model may need. Enable them per backend:

```sh
ARES_TEST_CUDA=1 cargo test -p ares-cuda --test gpu -- --test-threads=1
ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test gpu -- --test-threads=1
```

Tests against real models skip with a message unless their model files are
named: `git grep ARES_TEST_` lists the variables (`ARES_TEST_GGUF`,
`ARES_TEST_QWEN35_MODEL`, `ARES_TEST_QWEN38_27B` and others).

## Layout

| Path | Contents |
| --- | --- |
| `crates/core` | Tensor element types, byte units, memory ledgers, the host memory limit |
| `crates/gguf` | GGUF reader |
| `crates/quant` | CPU reference for every quantization type; the K/V formats |
| `crates/cuda` | CUDA device layer and kernels |
| `crates/vulkan` | Vulkan device layer and shaders |
| `crates/state` | Sequence state: pages, prefix sharing, checkpoints |
| `crates/store` | Sequence state on disk (the prompt cache that survives restarts) |
| `crates/io` | Reads of weights straight from the model file (io_uring, direct I/O) |
| `crates/sched` | Step planning, admission, draft policy, the n-gram drafter |
| `crates/weights` | Expert tiers and pruning policy |
| `crates/runtime` | Conductor and device threads: the step loop |
| `crates/model` | Model architectures: CPU reference and device implementation |
| `crates/sampler` | Sampler settings, the arithmetic every backend repeats, the CPU reference of a pick |
| `crates/grammar` | GBNF grammars, JSON schemas and regular expressions turned into the tokens that may come next |
| `crates/tokenizer` | Tokenizer and chat templates |
| `crates/api` | Types between the HTTP server and the runtime |
| `crates/server` | The `ares` binary |

## Documentation

[`docs/README.md`](docs/README.md) says what each document is for:

- [`docs/FEATURES.md`](docs/FEATURES.md): what Ares can do today, one line each.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): how the engine works.
- [`docs/OPEN.md`](docs/OPEN.md): the working list of what is open, measured
  and decided.
- [`docs/reference/`](docs/reference/): studies of the model architectures and
  of other engines.

## License

Apache-2.0; see [`LICENSE`](LICENSE).
