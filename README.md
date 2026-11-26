# Ares

An inference server for large language models on one to a few consumer GPUs,
written in Rust. It reads GGUF models, runs them through CUDA on Linux and
serves them over the OpenAI and Anthropic HTTP APIs.

> **Ares is experimental.** There is no release. Options, the shape of the API,
> the files it keeps on disk and the code itself change without notice. Run it
> beside a server you trust, such as llama.cpp, not in place of one.

## Why Ares

Ares exists for five reasons, and every piece of work is weighed against them:

1. **Speed**, always.
2. **Parallel requests.** llama.cpp serves several requests at once poorly, and
   vLLM needs a lot of VRAM to do it well.
3. **Efficient use of VRAM and RAM.**
4. **Large mixtures of experts streamed from the SSD:** a model larger than RAM,
   in a 4- or 5-bit quantization, without a large loss of speed.
5. **Room for experiments:** new ways to read experts ahead, to draft, to read
   prompts and to prune experts.

It supports the newest model architectures only and keeps no options for
compatibility: llama-server's option names are not accepted.

The first statement of the goals is
[issue 1](https://github.com/prometheion/ares/issues/1) (2026-11-09). Some of
it has been decided differently since, for example the GPUs CUDA runs on;
[`docs/OPEN.md`](docs/OPEN.md) holds the current decisions.

## Status

There is no release. What runs on `main`:

| | |
| --- | --- |
| Models | Qwen3.8 27B (`qwen35`, dense) and Qwen3.8 Flash-Next (`qwen4exp`, a mixture of experts whose experts live in VRAM, RAM and on the SSD), as GGUF files in any of the 28 GGUF weight types; images through the model's projector ([models](docs/user/models.md)) |
| GPUs | CUDA on NVIDIA Ampere (8.6, RTX 30) and Ada (8.9, RTX 40), one card or two; other GPUs are refused and are meant to run through the Vulkan backend later ([hardware](docs/user/hardware.md)) |
| System | Linux only |
| APIs | OpenAI chat, completions and Responses; Anthropic Messages, so Claude Code works against Ares; Prometheus metrics ([API](docs/user/api.md)) |

[`docs/user/features.md`](docs/user/features.md) lists what Ares does, one line
each; [`docs/OPEN.md`](docs/OPEN.md) what is open and in work.

## Getting started

```sh
cargo build --release
target/release/ares --model ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf
```

[`docs/user/quickstart.md`](docs/user/quickstart.md) has what the build needs,
both models, a first request and Claude Code; [`docs/README.md`](docs/README.md)
lists every document, for running Ares and for developing it. The rules for
changing Ares, for people and coding agents, are in [`AGENTS.md`](AGENTS.md).

## Measured

The two tables come from runs of both engines in the same session on the same
machine, alternating, with the setup named. They are measurements of two cards
and one model, not general claims.

**The 27B against llama.cpp at production settings.** Qwen3.8 27B UD-Q4_K_S on
an RTX 4070 and an RTX 3060 (12 GiB each) with peer access, 2026-11-23. The
other engine is the llama.cpp fork Ares grew out of
([Piggidragon/llama.cpp](https://github.com/Piggidragon/llama.cpp)), with quantized flash
attention, tensor split with peer-to-peer sums and MTP drafts, run with its production preset. Both
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

## Thanks

Ares takes what works best from the engines that came before it. Most of it
was learned from reading their code and measuring against them; only ggml's
quantization code is ported (see [`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES)).
With thanks to:

- [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp): GGUF and its
  weight types (`ares-quant` is a port of ggml's quantization code), the shape of
  many kernels, the server's API shapes, and the reference every test is held to.
- [Piggidragon/llama.cpp](https://github.com/Piggidragon/llama.cpp), the llama.cpp
  fork Ares grew out of: quantized flash attention, tensor split with peer-to-peer sums, MTP and
  DSpark drafts, the drafts' price rule and the conversation cache, which Ares
  set out to beat.
- [Strata](https://github.com/Niko1221/Strata): the reference for mixtures of
  experts on consumer cards: experts in VRAM, RAM and on the SSD, the CPU for
  the misses, large prompt chunks on borrowed slots, the usage profile.
- [TokenWing](https://github.com/runmyai/tokenwing) (Run My AI): prompts read
  layer by layer, experts streamed from the SSD, the routers of the next layers
  as a prediction.
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang)
  and [FlashInfer](https://github.com/flashinfer-ai/flashinfer): paged K and V,
  the prefix cache, scheduling and preemption by priority.
- [ktransformers](https://github.com/kvcache-ai/ktransformers),
  [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and the MoE offload
  papers: experts split between GPU and CPU.
- TensorFold, [DFlash](https://huggingface.co/z-lab) and DSpark: drafting with
  trees and block drafters; [Basalt](https://github.com/jesdga95/basalt),
  [Flide](https://github.com/OmniJev/Flide), [beellama.cpp](https://github.com/Anbeeld/beellama.cpp)
  (KVarN) and [TheCodacus' llama.cpp fork](https://github.com/thecodacus/llama.cpp)
  for their ideas.
- [Unsloth](https://huggingface.co/unsloth) for the GGUF files, and the Qwen team
  for the models.

## License

Apache-2.0; see [`LICENSE`](LICENSE).

Code ported from other projects keeps its own license; see
[`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES).
