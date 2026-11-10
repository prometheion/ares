# Ares

A new inference server for large language models that takes the best technique
of each existing engine and leaves out everything kept for compatibility.

- Maximum speed and maximum VRAM efficiency on one to a few consumer GPUs
- GGUF models with llama.cpp's weight quantization, and its own formats for
  the KV cache (`f16`, `q8` down to `q4`)
- CUDA (13.0 or newer) and Vulkan, Linux only, written in Rust
- The newest model architectures only
- One binary, a server; no command line tools, no GUI

The goals are in [issue 1](https://github.com/prometheion/ares/issues/1).
How the engine is built and how far each part is: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## State

Two test models stand in for Qwen4 until it is released. Qwen3.8-27B is
served by the `ares` binary on two GPUs through CUDA, with drafts from the
model's own draft block and sampling on the device, and is ahead of the
llama.cpp fork it is measured against in prompt processing and in
generation. Qwen3.8-Flash-Next, a mixture of experts streamed from SSD,
runs on the same two GPUs and agrees with its CPU reference, but is not
served yet and not yet faster than Strata, the engine it is measured
against. Work is paused until Qwen4. The numbers are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#status). There is no release
before Qwen4.

## Build

Needs Rust (stable), the CUDA toolkit 13.0 or newer for `nvcc`, and `glslc`
for the Vulkan shaders. The toolkit is a build dependency only; at run time
the engine loads the driver's own `libcuda.so.1`.

```sh
cargo build --release
```

```sh
cargo run --release --bin ares -- --model model.gguf
```

Without further options the server chooses the devices, the split and the
context from what the GPUs report and says what it chose; `--help` lists
the options. A build for one machine can name its GPU generations, for
example `ARES_CUDA_ARCHS=86,89`, instead of building the kernels for all
eight.

## Tests

```sh
cargo test --workspace
```

Tests that open GPUs are off by default, because opening a device takes VRAM
that a running model may need. Enable them per backend:

```sh
ARES_TEST_CUDA=1 cargo test -p ares-cuda --test gpu -- --test-threads=1
```

```sh
ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test gpu -- --test-threads=1
```

`ARES_TEST_GGUF=<model.gguf>` adds checks against a real model.

Transfers between GPUs go directly from device to device when the driver
reports peer access (NVLink, or cards and drivers that allow peer-to-peer over
PCIe), and through host memory otherwise. Consumer NVIDIA cards get peer access
with patched kernel modules and the matching driver library; point the engine
at that library with `ARES_LIBCUDA=<path>/libcuda.so.1`.

## Layout

| Path | Contents |
| --- | --- |
| `crates/core` | Tensor element types, byte units, memory ledgers |
| `crates/gguf` | GGUF reader |
| `crates/quant` | CPU reference for every quantization type; the KV formats |
| `crates/cuda` | CUDA device layer and kernels |
| `crates/vulkan` | Vulkan device layer and shaders |
| `crates/state` | Sequence state: pages, prefix sharing, checkpoints |
| `crates/store` | Sequence state on disk (the prompt cache that survives restarts) |
| `crates/io` | Reads of weights straight from the model file (io_uring, direct I/O) |
| `crates/sched` | Step planning, draft policy, n-gram drafter |
| `crates/weights` | Expert tiers and pruning policy |
| `crates/runtime` | Conductor and device threads: the step loop |
| `crates/model` | Model architectures: CPU reference and device implementation |
| `crates/tokenizer` | Tokenizer and chat templates |
| `crates/api` | Types between the HTTP server and the runtime |
| `crates/server` | The `ares` binary |
| `docs/` | Architecture and notes on the engines studied |

## License

Apache-2.0.
