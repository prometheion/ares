# Options

`ares --help` prints the same list. There are three ways to start the server:

```sh
ares --model <PATH> [OPTIONS]          # one model
ares --models-preset <FILE> [OPTIONS]  # several models from a preset file
ares --config <FILE> [OPTIONS]         # options from a file (which names the model)
```

How an option is written:

| Form | Example | Note |
| --- | --- | --- |
| `--name value` | `--context 32768` | the value may begin with a dash |
| `--name=value` | `--context=32768` | only for names with two dashes |
| short spelling | `-b 2048`, `-ub 512`, `-rea off` | the short names of llama-server, where Ares has them |
| switch | `--log-text`, `--no-reasoning-preserve` | takes no value |
| size | `16G`, `512M`, `1.5G` | binary units `K`, `M`, `G`, `T`, upper or lower case, `B` or `iB` after them allowed; a number alone is bytes |

An option given twice is an error, except `--chat-var`. An unknown name is
refused; for the names of llama-server that Ares has under another name, or has
no need of, the error names the option of Ares that does the job (`-c` is
`--context`, `-ts` is `--shares`) or says why there is none (`-ngl`: all of a
model is on the devices).

Every option is also an environment variable and a key of the config file (see
[below](#the-environment)). *Server* options hold for the whole server and never
stand in a section of a preset file; all other options are options of a model.

## General

| Option | Meaning | Default |
| --- | --- | --- |
| `--config <FILE>` | A file of options, one on each line (see [The config file](#the-config-file)). Command line only. | `ARES_CONFIG`, else none |
| `-h`, `--help` | Print the help and exit. | |
| `-V`, `--version` | Print what the binary is and exit: `ares <version> (<commit>, <date of the build>)`. The log's first line and `build_info` of `/props` say the same. | |
| `--model <PATH>` | The GGUF model file to serve; any part of a split model. Not with `--models-preset`. | none: `--model` or `--models-preset` is needed |
| `--models-preset <FILE>` | Server option. Serve several models from a preset file (see [The preset file](#the-preset-file)). | none |
| `--host <ADDR>` | Server option. Address to listen on. | `127.0.0.1` |
| `--port <PORT>` | Server option. Port to listen on; `0` picks a free one. | `8000` |
| `--timeout <SECONDS>` | Server option. How long a client may take to send a request, and to take what one write of an answer sent; then the connection is closed and its request ended. A reply the engine takes long for is never cut. At least 1. | `600` |

## Who may use the server

All four are server options.

| Option | Meaning | Default |
| --- | --- | --- |
| `--api-key <KEYS>` | Keys a request must carry one of, separated by commas, as `Authorization: Bearer <key>` or `x-api-key: <key>`; a key holds no white space. `/health` stays open. | none: no key is asked for |
| `--api-key-file <FILE>` | More keys, one on each line; lines that begin with `#` are comments, and a line with a space is an error. A file without a key is an error when `--api-key` names none either. | none |
| `--allowed-hosts <LIST>` | Names in a request's `Host` header that the server answers to besides its local ones (`localhost`, addresses), separated by commas, or `*` for any. Names given here are checked on every address and with keys too, and with keys they turn the check of pages (`--cors-origins`) on as well. | the local names only, when the server listens on a loopback address without keys; else any name |
| `--cors-origins <LIST>` | Origins whose web pages may call the server from a browser and read its answers, such as `https://chat.example.org`, separated by commas, or `*` for any. Without keys, a request from a page of another site is refused unless the page is of this machine (`localhost`, a loopback address), of the server's own name or of an `--allowed-hosts` name; with keys pages are checked only when `--allowed-hosts` names hosts. | none |

## Where the model goes

What is left out is decided from the devices and the model, and the log says how.

| Option | Meaning | Default |
| --- | --- | --- |
| `--devices <LIST>` | CUDA devices to use, by the driver's index, such as `0,1`. A GPU of a class this build does not run is skipped. | all devices the driver reports |
| `--split <KIND>` | `single`: one device holds the model (of several devices, the 27B takes the first that holds it, Flash-Next the one with the most free memory; `--devices` may not name several). `tensor`: every device holds a slice of every block (Flash-Next: exactly two devices on every layer). `layer`: every device holds a range of blocks. | `single` on one device. 27B: `single` when one device holds the model with the context `--context` names, or without it 8192 tokens (the model's own, if less); else `tensor`. Flash-Next: `tensor` on two devices that can write each other's memory (peer access), else `layer` |
| `--shares <LIST>` | Each device's share of the weights under a split, such as `0.625,0.375`. Refused for Flash-Next, whose split of each layer is measured at load. | in proportion to the memory bandwidth each device reports, lowered where a device's memory is too small for its share |

## Sequence state

| Option | Meaning | Default |
| --- | --- | --- |
| `--context <TOKENS>` | Tokens of sequence state for all sequences together. One request holds at most this or the model's own context. A dense model (the 27B) holds a named context in VRAM, and the load fails if the devices cannot hold it; Flash-Next places it as its own: the indexer's keys and the K and V of the first pages in VRAM, the rest in page-locked RAM inside `--ram`. | the model's own context (262144 for Qwen3.8). 27B: as much of it as the free VRAM holds, rounded down to a multiple of 1024. Flash-Next: all of it, the K and V beyond the first pages in page-locked RAM inside `--ram` |
| `--parallel <N>` | Sequences that run at once. Each takes VRAM for its recurrent state; further requests wait. At least 1; Flash-Next at most 8. | `4` |
| `--kv <FORMAT>` | Storage format of K and V: `f16`, `q8`, `q7`, `q6`, `q5` or `q4`, or two names `K,V` such as `q8,f16`. See [models.md](models.md). | `q8` |

## Host memory

| Option | Meaning | Default |
| --- | --- | --- |
| `--ram <SIZE>` | The most RAM the whole engine may use, for all loaded models together. What the engine needs to run comes first; the rest is for Flash-Next's experts that have no slot in VRAM, then for the prompt cache in RAM. Below the limit the engine gives memory back when other programs need it, the prompt cache first. A limit too small to run is refused with every share named. In a model's section of a preset file: that model's part. | all the machine has |

## Prompt cache on the SSD

| Option | Meaning | Default |
| --- | --- | --- |
| `--cache <DIR>` | A directory where the engine keeps the state it computed (K and V, checkpoints of recurrent state), so that a prompt that starts with it resumes from there after the state left RAM, after an unload and after a restart. Texts of 2048 tokens and more are written when they leave RAM; what RAM and the devices hold is written at an unload or a stop. One directory may serve all models. Flash-Next keeps the usage profile of its experts there. | none: nothing is written; the usage profiles go to `$XDG_CACHE_HOME/ares`, else `~/.cache/ares` |
| `--cache-size <SIZE>` | The most the directory may take, at least `1G`; what was used least recently goes first. | `32G` |

## Speed against exactness

| Option | Meaning | Default |
| --- | --- | --- |
| `--precision <KIND>` | `fast`: 8-bit activations where they are faster. `exact`: f16 products everywhere, the same text whatever else runs in a step. Flash-Next takes `fast` only. | `fast` |
| `--drafters <LIST>` | The model drafters, separated by commas: `mtp` (the MTP block in the model file), a file (an MTP block or a DFlash2 drafter, told apart by the architecture the file names), or `none` (no speculation at all). The n-gram drafter takes part with them, never alone. See [models.md](models.md). | 27B: `mtp` if the model file has an MTP block. Flash-Next: none; name `mtp` or a file |
| `-b`, `--batch-size <ROWS>` | Most rows of prompt in one step while a prompt is read alone. Above `--ubatch-size` only for Flash-Next, whose expert slots in VRAM lend the room for the step. At least 8. | 27B: `--ubatch-size`. Flash-Next: 32768 when a prompt is read layer by layer (K and V in `f16` or `q8`, no DFlash drafter), else 8192; as far as the slots can lend the room |
| `-ub`, `--ubatch-size <ROWS>` | Most rows of a step in the memory the model keeps for good, and of a piece of prompt while another request waits. At least 8. | 27B: `512`. Flash-Next: the most rows, a power of two from 8 to 4096, whose step scratch fits in a thirty-second of the smaller device's memory |
| `--window <ROWS>` | Most rows of one sequence's window: its newest token and the drafts verified with it. The rows take VRAM (about 5 MiB a row for the 27B, for every sequence). At least 2; Flash-Next at most 8. Only with a model drafter. | 27B: `8`. Flash-Next: `4` |
| `--draft-vocab <FILE>` | Token ids the model drafters may propose, one per line (lines that begin with `#` are comments; an empty list is an error); a shorter list makes a draft cheaper. Only with a model drafter, and on the 27B not under a layer split. | every token whose row of the output head is on the drafters' device |
| `--expert-prune <RULE:RANK>` | Flash-Next only. What a step does with a pick whose expert only the SSD holds, from the RANK-th pick of a row on (1 is the strongest), instead of waiting for the drive: `skip` leaves it out, `router` takes the router's next best expert that VRAM or RAM holds, `similar` the one most like it there, from a table of similar experts for the model file in the directory of the usage profile (the `qwen4exp_prune` example of `ares-model` writes it; without it the load is refused). `similar:8` touches the 8th to 10th of ten picks; a RANK above the picks of a row is refused. Changes the output. `off` turns it off. | `off` |

## Images

| Option | Meaning | Default |
| --- | --- | --- |
| `--mmproj <FILE>` | The projector the model reads images through: the `mmproj` GGUF of its image tower. Images come in requests as data URLs; remote URLs are not fetched. | none: images are refused |
| `--mmproj-weights <ram\|vram>` | Where the projector's weights stay between images: `ram` (page-locked inside `--ram`, uploaded block by block for every new image) or `vram` (about 0.9 GiB that the context gives up). Only with `--mmproj`. | `ram` |
| `--image-max-tokens <ROWS>` | The most rows (tokens) one image becomes, 8 to 4096; a larger image is shrunk to fit. The VRAM of the tower's scratch follows it (about 340 MiB at 4096). Only with `--mmproj`. | `4096`, as llama.cpp |

## Prompts

| Option | Meaning | Default |
| --- | --- | --- |
| `--chat-var <NAME=VALUE>` | A variable of the model's chat template, such as `add_vision_id=true`. May be given several times; a later value of the same name wins. The value is JSON where it reads as JSON, else text. | none |
| `--chat-template-file <FILE>` | A chat template (Jinja) that takes the place of the model file's, for every conversation. | the model file's |

## The line of requests

| Option | Meaning | Default |
| --- | --- | --- |
| `--queue <N>` | The most requests that may wait for a sequence while `--parallel` run; one more is refused with 503 and `Retry-After`, unless it has a better priority than one waiting that has not started: then that one is refused in its place (see [api.md](api.md#priorities)). | no limit: a request waits until it runs |
| `--generation-floor <TOKENS/S>` | The rate every generating request keeps while a prompt is read beside it; the prompt gets the rest of the devices' time, and at least half. `0` reads prompts at once. | `20` |
| `--watchdog <SECONDS>` | The longest one step may take, the warm-up's too, before the engine is taken for stuck: its requests end with the reason and the model is taken down. `0` watches nothing. | `600` |

## Thinking

These have llama-server's names. A request can change each of them for itself
(see [api.md](api.md#thinking)).

| Option | Meaning | Default |
| --- | --- | --- |
| `-rea`, `--reasoning <on\|off\|auto>` | Whether the model thinks before it answers (the template variable `enable_thinking`); `auto` leaves it to the chat template. | `auto` |
| `--reasoning-format <FORMAT>` | `auto`: the thinking goes to `reasoning_content`, in streamed deltas too. `none`: the thinking stays in `content`, tags and all. | `auto` |
| `--reasoning-effort <LEVEL>` | A level the chat template knows (the variable `reasoning_effort`), such as `low` or `medium`; `default` leaves it to the template. | `default` |
| `--reasoning-preserve`, `--no-reasoning-preserve` | Whether the thinking of earlier turns, as a client sends it back in `reasoning_content`, stays in the prompt. It must, for a conversation to continue from the prompt cache. | on |

## Defaults of request fields

These options give a field of a request its default for this model. The option
is the field's name with dashes (`--top-p` for `top_p`), and takes the values a
request may send, in the forms below (a request's `top_k: -1` and llama.cpp's
string of letters for `samplers` are a request's alone). A request's own value
always holds. The defaults below are the
engine's own: without options a reply is greedy.

| Option | Meaning | Values | Default |
| --- | --- | --- | --- |
| `--temperature <T>` | Temperature; `0` picks the most probable token | from 0 | `0` |
| `--top-k <N>` | Keep the N most probable tokens | a number of tokens, `0` for all | `0` |
| `--top-p <P>` | Keep the most probable tokens that hold P of the probability | above 0, at most 1 | `1` |
| `--min-p <P>` | Drop tokens less than P times as probable as the best | 0 to 1 | `0` |
| `--typical-p <P>` | Locally typical sampling | above 0, at most 1; `1` is off | `1` |
| `--top-n-sigma <N>` | Keep the tokens within N standard deviations of the best logit | a number; `0` or less is off | `0` |
| `--xtc-probability <P>` | XTC: the chance that the most probable tokens are removed | 0 to 1 | `0` (off) |
| `--xtc-threshold <P>` | XTC: the probability from which a token is one of them | 0 to 1 | `0.1` |
| `--dynatemp-range <R>` | A temperature that moves by R around `--temperature` with the entropy | from 0 | `0` (off) |
| `--dynatemp-exponent <E>` | The exponent of that move | from 0 | `1` |
| `--repeat-penalty <N>` | Lowers the logits of the tokens in the last `--repeat-last-n` tokens by this factor | above 0; `1` is off | `1` |
| `--presence-penalty <N>` | Subtracted once from the logits of those tokens | -2 to 2 | `0` |
| `--frequency-penalty <N>` | Subtracted from their logits for every time they stand there | -2 to 2 | `0` |
| `--repeat-last-n <N>` | The tokens the three penalties look at | 0 to 1024; `0` turns them off | `64` |
| `--dry-multiplier <N>` | DRY, the penalty for a token that would repeat a sequence: its strength | from 0 | `0` (off) |
| `--dry-base <N>` | DRY's base, which the penalty grows by with the length of the repeat | from 0 | `1.75` |
| `--dry-allowed-length <N>` | The longest repeat DRY lets pass | up to 1024 | `2` |
| `--dry-penalty-last-n <N>` | The tokens DRY looks back on | up to 1024 | `64` |
| `--dry-sequence-breakers <JSON>` | Strings a repeated sequence does not reach across | a JSON array of strings | `["\n", ":", "\"", "*"]` |
| `--mirostat <0\|1\|2>` | Mirostat in place of the other samplers | `0` (off), `1`, `2` | `0` |
| `--mirostat-tau <N>` | Mirostat's target surprise | above 0 | `5` |
| `--mirostat-eta <N>` | Mirostat's rate | from 0 | `0.1` |
| `--samplers <LIST>` | The samplers that apply and their order | names separated by commas; `penalties` and `dry` first and once each, at most 16 others | `penalties,dry,top_n_sigma,top_k,typ_p,top_p,min_p,xtc,temperature` |
| `--seed <N>` | The seed of a request that names none, which then draws the same for the same prompt; a request asks for a draw of its own with `seed: -1` | a whole number from 0 | a new seed for every request |
| `--reasoning-budget-tokens <N>` | The most tokens the thinking of a reply may take; then the budget's message and the end of the thinking are written, and the answer follows | a number of tokens | none: no budget |
| `--reasoning-budget-message <TEXT>` | What is written before the end of the thinking when its budget is spent | text | none |

## The length of a reply

| Option | Meaning | Default |
| --- | --- | --- |
| `--max-tokens <N>` | The limit of tokens for a request that names none (`max_tokens`, `max_completion_tokens`, `max_output_tokens`, `n_predict`). At least 1. | none: a reply ends when the model ends it or the context is full |
| `--max-tokens-limit <N>` | The most tokens of any reply. A request that asks for more gets this many, and its response says so in the header `x-ares-max-tokens-limit`; a request that names no limit gets `--max-tokens` or, without it, this. A `--max-tokens` above it is an error. | none |

## The log

All three are server options.

| Option | Meaning | Default |
| --- | --- | --- |
| `--log-format <text\|json>` | `text`: a line for people. `json`: one JSON object a line, for collectors. | `text` |
| `--log-file <FILE>` | Append the log to this file. | standard error |
| `--log-text` | The prompt and the reply of every request go into the log as well; without it the log holds neither. | off |

The log level is the `RUST_LOG` environment variable (default `info`).

## The environment

Every option except `--config`, `--help` and `--version` is also an environment variable:
`ARES_`, then the option's name without its leading dashes, with every dash
turned into an underscore, in capitals.

| Command line | Environment |
| --- | --- |
| `--top-p 0.9` | `ARES_TOP_P=0.9` |
| `--models-preset models.ini` | `ARES_MODELS_PRESET=models.ini` |
| `-b 2048` | `ARES_BATCH_SIZE=2048` (short spellings have no variable of their own) |
| `--log-text` | `ARES_LOG_TEXT=true` (a switch takes `true` or `false`) |
| `--no-reasoning-preserve` | `ARES_REASONING_PRESERVE=false` |

- `ARES_CONFIG=<FILE>` names the config file when the command line names none.
- An option that may repeat (`--chat-var`) takes one value from the environment.
- Variables without the `ARES_` prefix are never read: a shell's `CTX` or
  `CONTEXT` changes nothing.
- A value is checked as on the command line, and an error names the variable.

## The config file

`--config <FILE>` (or `ARES_CONFIG`) reads options from a file. It uses the same
reader as the preset file, without sections.

| Line | Meaning |
| --- | --- |
| `name = value` | an option under its name without the dashes (`context = 32768`); white space around both is trimmed |
| `name = true` / `name = false` | a switch (`log-text = true`, `reasoning-preserve = false`) |
| a line that begins with `#` or `;` | a comment; only whole lines are comments |
| an empty line | nothing |

- A key stands once, except `chat-var`.
- Server options and `models-preset` may stand here; `config`, `help` and `version` may not.
- `[section]` lines are an error: several models go in a preset file.
- An unknown key or a wrong value is an error that names the file and the line.
- Paths are taken as written: `~` is not expanded, and a relative path starts at
  the directory the server runs in.

```ini
# ares.conf
model = Qwen3.8-27B-UD-Q4_K_S.gguf
port = 8000
context = 131072
parallel = 2
kv = q5
reasoning-effort = medium
temperature = 0.6
chat-var = add_vision_id=true
```

## Precedence

From the strongest:

| Place | Note |
| --- | --- |
| 1. the command line | |
| 2. the environment (`ARES_...`) | |
| 3. the config file | |
| 4. a model's section of the preset file | only with `--models-preset` |
| 5. the section `[*]` of the preset file | only with `--models-preset` |
| 6. the built-in default | |

With a preset file, a model option from the first three places holds for every
model. `--ram` on the command line (or in the environment or the config file) is
the server's limit and holds over `ram` in `[*]`.

## The preset file

`--models-preset <FILE>` serves the models of an INI file. A model is loaded when
a request names it, unloaded when another model needs its room or when it was
idle for its `idle-unload`, and loaded again by the next request for it. A
request that waits for a load waits; it is not refused because another model is
busy.

```ini
[*]
ram = 24G
idle-unload = 30m

[qwen-27b]
model = Qwen3.8-27B-UD-Q4_K_S.gguf
aliases = default
load-on-startup = true

[flash-next]
model = Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
drafters = mtp-q2_0.gguf
```

| Section | Holds |
| --- | --- |
| `[*]` | keys for every model; at most once; no `model` and no `aliases` |
| `[<id>]` | one model; the name is the id requests use and `/v1/models` lists; it needs a `model` key |

| Key | Meaning | Default |
| --- | --- | --- |
| a model option without the dashes (`model`, `context`, `kv`, `drafters`, `temperature`, ...) | as on the command line; `chat-var` may stand several times | as on the command line |
| `ram` | in `[*]`: the most RAM of the server, for all models together; in a model's section: that model's part of it, and a model without one takes what is left | all the machine has |
| `aliases` | more ids the model answers to, separated by commas; only in a model's section | none |
| `load-on-startup` | `true`: loaded when the server starts, not at its first request | `false` |
| `idle-unload` | how long the model stays loaded without a request: seconds (`600`), a number with `s`, `m` or `h` (`90s`, `30m`, `2h`), or `never` | `never` |

Nothing in the file is ignored. These are errors that name the file and the line:
an unknown key, a server option (`host`, `port`, `api-key`, `log-file`, ...), a
value the option does not take, a key before the first section, a key or a
section that stands twice, a section without `model`, a model file that does not
exist, an id or alias that two sections answer to, a model's `ram` above the
server's, and a `max-tokens` above the `max-tokens-limit` of the same model. A
file without a model section is an error too. Paths are taken as in the config
file.

Presets that share a device should name their `context` and `ram`: a preset
without them gets what is free when it is loaded, which next to other models is
less than alone. [api.md](api.md#other-routes) has the routes that load and
unload models.

## Other environment variables

| Variable | Meaning |
| --- | --- |
| `RUST_LOG` | the log level and filter; default `info` |
| `ARES_CONFIG` | the config file, when `--config` is not given |
| `ARES_LIBCUDA` | the driver library to load instead of the system's `libcuda.so.1`, for example a driver build with peer access on consumer cards (see [hardware.md](hardware.md)) |
| `XDG_CACHE_HOME`, `HOME` | without `--cache`, Flash-Next's expert usage profiles go to `$XDG_CACHE_HOME/ares` (an absolute path only), else `~/.cache/ares`; without either, none is kept |
| `ARES_ATTENTION_FATBIN` | kernel development only: the attention kernels are loaded from this file instead of the built-in ones |

The build reads `ARES_CUDA_ARCHS`, `NVCC` (with `CUDA_PATH`, `CUDA_HOME`) and
`GLSLC` (with `VULKAN_SDK`); see [hardware.md](hardware.md) and
[../dev/README.md](../dev/README.md).
