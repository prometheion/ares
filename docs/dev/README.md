# Documents for developing Ares

For the people and agents who work on the engine. How to run Ares is in
[`../user/`](../user/quickstart.md); what is open, in work and decided is in
[`../OPEN.md`](../OPEN.md). Some documents here name paths on the development
machine (notes, benches and captures under `/var/tmp/`); they are not part of the
repository and are not needed to build, run or test Ares.

| Document | What it is |
| --- | --- |
| [`architecture/`](architecture/overview.md) | How the engine works and why: an overview and one file per topic (below). Kept current at every merge. |
| [`experiments/`](experiments/) | Dated measurement logs and A/B results, by topic. Not kept current. |
| [`studies/`](studies/) | Studies of other engines and of the model architectures, each made on its date. Not kept current. |
| [`history.md`](history.md) | What `OPEN.md` listed as done, verbatim, by date. |
| [`features-in-detail.md`](features-in-detail.md) | The long feature list of 2026-11-23. Not kept current. |

## Architecture

| File | What it holds |
| --- | --- |
| [`overview.md`](architecture/overview.md) | Aims and scope decisions, principles, crates, the shared model layer, a step, threads, what is taken from where, open questions |
| [`sequence-state.md`](architecture/sequence-state.md) | Pages, K/V formats, the reference for attention, what this prepares for Qwen4 |
| [`prompt-cache.md`](architecture/prompt-cache.md) | The prompt cache's levels in RAM and on SSD |
| [`speculation.md`](architecture/speculation.md) | The draft policy, prices, the chain, the n-gram drafter, the MTP driver, drafter kinds, DFlash |
| [`sampling.md`](architecture/sampling.md) | The sampler on the device, request fields, probabilities, cost |
| [`structured-output.md`](architecture/structured-output.md) | Grammars, schemas, tool-call rules and drafts under a rule |
| [`weights.md`](architecture/weights.md) | Dense weights, expert tiers, K/V offload |
| [`devices.md`](architecture/devices.md) | GPU classes, several devices, the 27B on two cards, the second card's role |
| [`kernels.md`](architecture/kernels.md) | Weight kernels for decode and prefill, float matrices, expert, layer and attention kernels, the tensor-core cap |
| [`flash-next.md`](architecture/flash-next.md) | Flash-Next through the runtime: its MTP block, steps that wait for experts, the plan, K and V beyond VRAM, the prefix cache |
| [`experts.md`](architecture/experts.md) | The experts' usage profile, members and loans, reading ahead, pruning |
| [`flash-next-prompts.md`](architecture/flash-next-prompts.md) | Wide steps, large steps on lent VRAM, long prompts layer by layer |
| [`vulkan.md`](architecture/vulkan.md) | The Vulkan backend |
| [`images.md`](architecture/images.md) | The image tower and images in a prompt |
| [`host-memory.md`](architecture/host-memory.md) | The one RAM limit and its users; loading and unloading |
| [`server.md`](architecture/server.md) | The server's code, access, options, properties, stop strings, the line, priorities, the watchdog, logs, router mode |
| [`api.md`](architecture/api.md) | Request defaults, thinking, conversations that continue from the cache, tool calls, the Messages and Responses APIs |

## Experiments

| File | Measurements of |
| --- | --- |
| [`speculation.md`](experiments/speculation.md) | the draft chain, the MTP block of Flash-Next, the n-gram drafter, the MTP driver, the draft policy, DFlash |
| [`two-cards.md`](experiments/two-cards.md) | placements of both models on two cards, the tensor split's scratch, the attention heads by context, the second card's role |
| [`flash-next.md`](experiments/flash-next.md) | Flash-Next's decode through the server, K and V beyond VRAM, files in any type, the usage profile, members and loans, reading ahead, the gap between 17 and 22 ms, pruning |
| [`flash-next-prompts.md`](experiments/flash-next-prompts.md) | wide steps, their float matrices, transit, lent VRAM, passes layer by layer |
| [`against-strata.md`](experiments/against-strata.md) | Flash-Next against Strata on the test machine, from the SSD |
| [`against-the-fork.md`](experiments/against-the-fork.md) | the 27B against the owner's llama.cpp fork at production settings, alone and in parallel |
| [`claude-code.md`](experiments/claude-code.md) | Claude Code against Ares |
| [`images.md`](experiments/images.md) | the image tower and images in a prompt |
| [`host-memory.md`](experiments/host-memory.md) | the expert tier's worth, giving memory back |
| [`server.md`](experiments/server.md) | priorities of requests |

## Studies

| File | Made on | What it is |
| --- | --- | --- |
| [`qwen35-architecture.md`](studies/qwen35-architecture.md) | kept with the code | The specification of `qwen35` (Qwen3.5 and Qwen3.8 dense), which the CPU reference implements, with its proof against llama.cpp |
| [`qwen4exp-architecture.md`](studies/qwen4exp-architecture.md) | 2026-11-10 | The specification of `qwen4exp` (Qwen3.8 Flash-Next) |
| [`kv-real-data.md`](studies/kv-real-data.md) | 2026-11-09 | The K/V formats measured on attention data captured from real models |
| [`feature-checklist.md`](studies/feature-checklist.md) | 2026-11-11 | What the other engines have, one row per capability, and where Ares stood |
| [`engines-survey.md`](studies/engines-survey.md) | 2026-11-09 | Techniques of other engines worth taking, by feature |
| [`serving-engines-code-survey.md`](studies/serving-engines-code-survey.md) | 2026-11-10 | How vLLM, SGLang, FlashInfer and others build their fast parts, read in their code |
| [`moe-offload-survey.md`](studies/moe-offload-survey.md) | 2026-11-10 | How the engines Run My AI credits offload mixtures of experts |
| [`strata-notes.md`](studies/strata-notes.md) | 2026-11-09 | Notes from reading Strata, the reference engine for Flash-Next |
| [`codacus-notes.md`](studies/codacus-notes.md) | 2026-11-09 | Notes from reading TheCodacus' llama.cpp fork |
| [`kvarn-notes.md`](studies/kvarn-notes.md) | 2026-11-09 | Notes on KVarN (beellama.cpp) and TurboQuant K/V formats |
| [`runmyai-notes.md`](studies/runmyai-notes.md) | 2026-11-09 | Notes on Run My AI (TokenWing) from its web page, before its source was open |

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
named, one variable for each file, the same in every crate:

| Variable | Model file |
| --- | --- |
| `ARES_TEST_QWEN35_0_8B` | Qwen3.5-0.8B (`Qwen3.5-0.8B-Q4_K_M.gguf`) |
| `ARES_TEST_QWEN38_27B` | Qwen3.8-27B in Q4_K_S (`Qwen3.8-27B-UD-Q4_K_S.gguf`), on two cards |
| `ARES_TEST_QWEN38_27B_IQ2` | Qwen3.8-27B in IQ2_XXS (`Qwen3.8-27B-UD-IQ2_XXS.gguf`), on one card |
| `ARES_TEST_DRAFT_MODEL` | that IQ2_XXS file with its MTP block (`...-with-draft-block.gguf`) |
| `ARES_TEST_DFLASH_27B` | the 27B's DFlash2 drafter |
| `ARES_TEST_FLASH_NEXT` | Qwen3.8-Flash-Next, the first part of its split file |
| `ARES_TEST_FLASH_NEXT_MTP` | Flash-Next's MTP block (`mtp-q2_0.gguf`) |
| `ARES_TEST_GGUF` | any model file whose tensors the kernel tests of `ares-cuda` and `ares-vulkan` check |

`git grep ARES_TEST_` lists the others (directories of captures, devices,
ports, limits).

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

## Tools beside the code

Some crates keep the programs that made their reference data, with a README
that lists the commands used: `crates/model/tools/capture` and
`crates/model/tools/q4capture` (what llama.cpp computes for each model, block by
block), `crates/quant/tools/kv_real` (the data of `studies/kv-real-data.md`) and
`crates/tokenizer/tools/reference` (llama.cpp's tokenizer and chat templates on
many inputs), and the generators of `crates/quant/src/tables.rs`,
`crates/tokenizer/src/unicode_table.rs` and `crates/quant/tests/golden/`. They
need model files and a llama.cpp tree, a fresh clone built on the CPU (the
commands are in `crates/model/tools/capture/README.md`), which the scripts take
as an argument or as `LLAMA_CPP_DIR`, with no default; they are not part of the
cargo build.

`crates/tokenizer/examples/draft_vocab.rs` makes and checks the draft
vocabularies of `--draft-vocab` from a corpus, with the model's own tokenizer.

`python3 tools/check-doc-links.py` checks the links between the documents, every
mention of a document and every citation of a paragraph in the code
(`<file>.md, "<name>"` or `"<name>" in <file>.md`). Run it after moving or
renaming a paragraph.

## Words the documents use

- **The owner**: the owner of the project, who decides its scope.
- **The fork**: the owner's llama.cpp fork, the production server Ares is
  measured against.
- **The development machine**: an RTX 4070 and an RTX 3060 with 12 GiB each,
  31 GiB of RAM. **The test machine**: an RTX 4060 Ti with 16 GiB, 62 GiB of
  RAM.
- **The 27B** and **Flash-Next**: Qwen3.8 27B and Qwen3.8 Flash-Next, the two
  models Ares runs.

## Where the former ARCHITECTURE.md went

On 2026-11-24 the former `docs/ARCHITECTURE.md` was split into `architecture/` and
`experiments/`, its text moved verbatim (commit 128c474a). A branch that still edits
the old file moves its text into the file below.

| Section of the old file | Now in |
| --- | --- |
| Principles | [`architecture/overview.md`](architecture/overview.md) |
| Crates | [`architecture/overview.md`](architecture/overview.md) |
| &nbsp;&nbsp;The shared model layer | [`architecture/overview.md`](architecture/overview.md) |
| A step | [`architecture/overview.md`](architecture/overview.md) |
| &nbsp;&nbsp;Threads | [`architecture/overview.md`](architecture/overview.md) |
| Sequence state | [`architecture/sequence-state.md`](architecture/sequence-state.md); its "In RAM" and "On SSD" in [`architecture/prompt-cache.md`](architecture/prompt-cache.md) |
| &nbsp;&nbsp;What this prepares for Qwen4 | [`architecture/sequence-state.md`](architecture/sequence-state.md) |
| Speculation | [`architecture/speculation.md`](architecture/speculation.md) |
| &nbsp;&nbsp;Drafters as kinds | [`architecture/speculation.md`](architecture/speculation.md) |
| &nbsp;&nbsp;The DFlash drafter | [`architecture/speculation.md`](architecture/speculation.md) |
| Sampling | [`architecture/sampling.md`](architecture/sampling.md); from "The hook" on in [`architecture/structured-output.md`](architecture/structured-output.md) |
| Weights | [`architecture/weights.md`](architecture/weights.md) |
| Devices | [`architecture/devices.md`](architecture/devices.md); its kernels, from "One table of weight types" to "The tensor-core cap of GeForce cards", in [`architecture/kernels.md`](architecture/kernels.md) |
| &nbsp;&nbsp;Several devices | [`architecture/devices.md`](architecture/devices.md) |
| &nbsp;&nbsp;Qwen3.8-27B on the two cards | [`architecture/devices.md`](architecture/devices.md) |
| &nbsp;&nbsp;Qwen3.8-Flash-Next on the two cards | [`experiments/two-cards.md`](experiments/two-cards.md) |
| &nbsp;&nbsp;What the second card is for | [`architecture/devices.md`](architecture/devices.md) |
| &nbsp;&nbsp;Qwen3.8-Flash-Next through the runtime | [`architecture/flash-next.md`](architecture/flash-next.md); "Wide steps" and from "Where a faster prompt path goes next" to "Long prompts layer by layer" in [`architecture/flash-next-prompts.md`](architecture/flash-next-prompts.md); from "The usage profile of the experts" to "Pruning as a fourth level" in [`architecture/experts.md`](architecture/experts.md) |
| &nbsp;&nbsp;Vulkan | [`architecture/vulkan.md`](architecture/vulkan.md) |
| The image tower | [`architecture/images.md`](architecture/images.md) |
| &nbsp;&nbsp;On the device | [`architecture/images.md`](architecture/images.md) |
| &nbsp;&nbsp;Images in a prompt | [`architecture/images.md`](architecture/images.md) |
| What is taken from where | [`architecture/overview.md`](architecture/overview.md) |
| Status and what is open | [`architecture/overview.md`](architecture/overview.md) |
| Host memory | [`architecture/host-memory.md`](architecture/host-memory.md) |
| Loading and unloading | [`architecture/host-memory.md`](architecture/host-memory.md) |
| Server | [`architecture/server.md`](architecture/server.md); from "The defaults of a model's requests" to "The Responses API" in [`architecture/api.md`](architecture/api.md) |
| Open questions | [`architecture/overview.md`](architecture/overview.md) |

By lines of the old file (at main 528d14d3); a heading in brackets is the one a
measurement log got in its new place:

<details>
<summary>Every range of lines</summary>

| Lines at main 528d14d3 | Now in |
| --- | --- |
| 1-574 | `architecture/overview.md` |
| 575-1019 | `architecture/sequence-state.md` |
| 1020-1326 | `architecture/prompt-cache.md` |
| 1327-1344 | `architecture/sequence-state.md` |
| 1345-1545 | `architecture/speculation.md` |
| 1546-1629 | `experiments/speculation.md` (under "The chain in a step of its own, measured (2026-11-15 and 2026-11-16)") |
| 1630-1722 | `experiments/speculation.md` |
| 1723-1754 | `architecture/speculation.md` |
| 1755-1783 | `experiments/speculation.md` (under "The n-gram drafter, measured (2026-11-10)") |
| 1784-1859 | `architecture/speculation.md` |
| 1860-1877 | `experiments/speculation.md` (under "The MTP driver, measured (2026-11-17)") |
| 1878-1952 | `architecture/speculation.md` |
| 1953-1995 | `experiments/speculation.md` (under "The draft policy on the 27B, measured (2026-11-11)") |
| 1996-2110 | `architecture/speculation.md` |
| 2111-2164 | `experiments/speculation.md` (under "The DFlash drafter, measured (2026-11-16)") |
| 2165-2170 | `architecture/speculation.md` |
| 2171-2534 | `architecture/sampling.md` |
| 2535-2719 | `architecture/structured-output.md` |
| 2720-2891 | `architecture/weights.md` |
| 2892-2927 | `architecture/devices.md` |
| 2928-4064 | `architecture/kernels.md` |
| 4065-4148 | `architecture/devices.md` |
| 4149-4166 | `experiments/two-cards.md` (under "Qwen3.8-27B on the two cards, measured (2026-11-10)") |
| 4167-4284 | `architecture/devices.md` |
| 4285-4387 | `experiments/two-cards.md` (under "The tensor split's step scratch and draft link, measured (2026-11-12)") |
| 4388-4445 | `architecture/devices.md` |
| 4446-4486 | `experiments/two-cards.md` |
| 4487-4518 | `architecture/devices.md` |
| 4519-4591 | `experiments/two-cards.md` |
| 4592-4630 | `architecture/devices.md` |
| 4631-4652 | `experiments/two-cards.md` (under "Flash-Next's second card as an expert tier, measured (2026-11-22)") |
| 4653-4680 | `architecture/devices.md` |
| 4681-4721 | `experiments/two-cards.md` (under "The second card's role timed at load, measured (2026-11-23)") |
| 4722-4731 | `architecture/devices.md` |
| 4732-4862 | `architecture/flash-next.md` |
| 4863-4893 | `experiments/flash-next.md` (under "The MTP block of Flash-Next, measured (2026-11-15)") |
| 4894-4962 | `architecture/flash-next.md` |
| 4963-5091 | `architecture/flash-next-prompts.md` |
| 5092-5163 | `experiments/flash-next.md` |
| 5164-5267 | `architecture/flash-next.md` |
| 5268-5320 | `experiments/flash-next.md` (under "K and V beyond VRAM, measured (2026-11-22)") |
| 5321-5357 | `architecture/flash-next.md` |
| 5358-5383 | `experiments/flash-next.md` (under "Files in any type, measured (2026-11-17)") |
| 5384-5399 | `architecture/flash-next.md` |
| 5400-5432 | `experiments/flash-next.md` (under "UD-Q4_K_XL through the server (2026-11-17)") |
| 5433-5509 | `architecture/experts.md` |
| 5510-5566 | `experiments/flash-next.md` (under "The usage profile, measured (2026-11-16)") |
| 5567-5597 | `architecture/experts.md` |
| 5598-5672 | `experiments/flash-next.md` (under "Members, loans and promotions, measured (2026-11-22)") |
| 5673-5787 | `architecture/experts.md` |
| 5788-5853 | `experiments/flash-next.md` (under "Reading experts ahead, measured (2026-11-16)") |
| 5854-5907 | `experiments/flash-next.md` |
| 5908-5970 | `architecture/experts.md` |
| 5971-6078 | `experiments/flash-next.md` (under "Pruning, measured (2026-11-16)") |
| 6079-6085 | `architecture/experts.md` |
| 6086-6181 | `architecture/flash-next.md` |
| 6182-6363 | `experiments/flash-next-prompts.md` |
| 6364-6449 | `architecture/flash-next-prompts.md` |
| 6450-6482 | `experiments/flash-next-prompts.md` (under "Large prompt steps on lent VRAM, measured (2026-11-22)") |
| 6483-6487 | `architecture/flash-next-prompts.md` |
| 6488-6497 | `experiments/flash-next-prompts.md` |
| 6498-6624 | `architecture/flash-next-prompts.md` |
| 6625-6658 | `experiments/flash-next-prompts.md` (under "Long prompts layer by layer, measured (2026-11-23)") |
| 6659-6696 | `experiments/against-strata.md` |
| 6697-6880 | `architecture/vulkan.md` |
| 6881-7040 | `architecture/images.md` |
| 7041-7081 | `experiments/images.md` (under "The tower on the device, measured (2026-11-16)") |
| 7082-7156 | `architecture/images.md` |
| 7157-7185 | `experiments/images.md` (under "Images in a prompt, checked and measured (2026-11-22)") |
| 7186-7196 | `architecture/images.md` |
| 7197-7230 | `architecture/overview.md` |
| 7231-7440 | `architecture/host-memory.md` |
| 7441-7501 | `experiments/host-memory.md` (under "Host memory, measured (2026-11-15)") |
| 7502-7591 | `architecture/host-memory.md` |
| 7592-7687 | `architecture/server.md` |
| 7688-8477 | `architecture/api.md` |
| 8478-8701 | `architecture/server.md` |
| 8702-8727 | `experiments/server.md` (under "Priorities of requests, measured (2026-11-22)") |
| 8728-8967 | `architecture/server.md` |
| 8968-8975 | `architecture/overview.md` |

</details>
