# The documents of Ares

Ares is experimental, and so are these documents: they describe the engine as
it is on `main` and change with it. The [README](../README.md) of the
repository says what Ares is, how to build it and how to run it.

## The four that are kept current

| Document | What it is for |
| --- | --- |
| [`user/features.md`](user/features.md) | What Ares can do today, one line per feature, with the options and routes by name. Brought up to date at every merge. |
| [`dev/architecture/`](dev/architecture/overview.md) | How the engine works and why: the principles and scope decisions, the crates, a step, sequence state, speculation, sampling, weights, devices and kernels, the image tower, host memory, loading and the server. The place to start before changing code. It is long; find a part by its section. |
| [`OPEN.md`](OPEN.md) | The working list: what is in work, the defects, the measurements owed, the owner's decisions and the questions to the owner. Brought up to date at every merge. |
| [`dev/studies/feature-checklist.md`](dev/studies/feature-checklist.md) | What the other inference engines have, one row per capability, and where Ares stands. Some rows lag behind; `OPEN.md` names them. |

`OPEN.md` is written for the people and agents who work on the engine. It
names paths on the development machine, such as notes, benches and captures
under `/var/tmp/` and the owner's model and llama.cpp directories under `~/`.
They are not part of the repository and are not needed to build, run or test
Ares; they say where the evidence for a line was kept.

## Studies (`reference/`)

Each of these was made on the date it names and is not kept current. They are
the evidence behind decisions in `dev/architecture/`.

| Document | What it is |
| --- | --- |
| [`qwen35-architecture.md`](dev/studies/qwen35-architecture.md) | The specification of the `qwen35` architecture (Qwen3.5 and Qwen3.8 dense), which the CPU reference in `crates/model` implements, with its proof against llama.cpp. |
| [`qwen4exp-architecture.md`](dev/studies/qwen4exp-architecture.md) | The specification of the `qwen4exp` architecture (Qwen3.8 Flash-Next). |
| [`kv-real-data.md`](dev/studies/kv-real-data.md) | The K/V formats measured on attention data captured from real models. |
| [`engines-survey.md`](dev/studies/engines-survey.md) | Techniques of other engines worth taking, by feature. |
| [`serving-engines-code-survey.md`](dev/studies/serving-engines-code-survey.md) | How vLLM, SGLang, FlashInfer and others build their fast parts, read in their code. |
| [`moe-offload-survey.md`](dev/studies/moe-offload-survey.md) | How the engines that the Run My AI project credits offload mixtures of experts. |
| [`strata-notes.md`](dev/studies/strata-notes.md) | Notes from reading Strata, the reference engine for Flash-Next. |
| [`codacus-notes.md`](dev/studies/codacus-notes.md), [`kvarn-notes.md`](dev/studies/kvarn-notes.md), [`runmyai-notes.md`](dev/studies/runmyai-notes.md) | Notes from reading one project each: TheCodacus' llama.cpp fork, KVarN (beellama.cpp), Run My AI. |

## Tools beside the code

Some crates keep the programs that made their reference data, with a README
that lists the commands used: `crates/model/tools/capture` and
`crates/model/tools/q4capture` (what llama.cpp computes for each model, block by
block), `crates/quant/tools/kv_real` (the data of `kv-real-data.md`) and
`crates/tokenizer/tools/reference` (llama.cpp's tokenizer and chat templates on
many inputs). They need a built llama.cpp checkout and model files, and are
not part of the cargo build.

## Words the documents use

- **The owner**: the owner of the project, who decides its scope.
- **The fork**: the owner's llama.cpp fork, the production server Ares is
  measured against.
- **The development machine**: an RTX 4070 and an RTX 3060 with 12 GiB each,
  31 GiB of RAM. **The test machine**: an RTX 4060 Ti with 16 GiB, 62 GiB of
  RAM.
- **The 27B** and **Flash-Next**: Qwen3.8 27B and Qwen3.8 Flash-Next, the two
  models Ares runs.
