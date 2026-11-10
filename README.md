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

The backbone exists: the model file reader, the device layers with the weight
kernels, the bookkeeping for sequence state, scheduling and expert placement,
and a runtime that drives the devices with a synthetic model. No real model
runs yet. Architecture-specific work waits for the Qwen4 release.

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
| `crates/api` | Types between the HTTP server and the runtime |
| `crates/server` | The `ares` binary |
| `docs/` | Architecture and notes on the engines studied |

## License

Apache-2.0.
