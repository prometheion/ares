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
How the engine is built: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## State

Ares supports two models until Qwen4 replaces them: Qwen3.8-27B, dense, and
Qwen3.8-Flash-Next, a mixture of experts whose experts live in VRAM, RAM and on
the SSD. Both are served by the `ares` binary through CUDA on one or two GPUs;
Vulkan follows when CUDA is complete. What runs today and how fast:
[docs/FEATURES.md](docs/FEATURES.md). What is still open:
[docs/OPEN.md](docs/OPEN.md).

There is no release yet. The first release is the complete engine: every
row of [docs/reference/feature-checklist.md](docs/reference/feature-checklist.md)
that is inside the engine's scope, on CUDA and on Vulkan.

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
the options.

CUDA runs on the GPU classes the engine was tested on: Ampere (8.6, RTX 30)
and Ada (8.9, RTX 40). The build has machine code for those two only, and a
GPU of another class is refused; such GPUs are to run through the Vulkan
backend, which does not run models yet. A developer can build other classes
on purpose with `ARES_CUDA_ARCHS`, for example `ARES_CUDA_ARCHS=86,89,120`;
the build then warns that they are untested.

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
| `crates/sched` | Step planning, admission, draft policy, lookup drafter |
| `crates/weights` | Expert tiers and pruning policy |
| `crates/runtime` | Conductor and device threads: the step loop |
| `crates/model` | Model architectures: CPU reference and device implementation |
| `crates/tokenizer` | Tokenizer and chat templates |
| `crates/api` | Types between the HTTP server and the runtime |
| `crates/server` | The `ares` binary |
| `docs/` | Architecture and notes on the engines studied |

## License

Apache-2.0.
