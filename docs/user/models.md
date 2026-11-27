# Models

Ares runs two model architectures, from GGUF files. The model's id is its file
name without `.gguf`, or the section name in a preset file.

## Architectures

| `general.architecture` | Models | Kind |
| --- | --- | --- |
| `qwen35` | Qwen3.8 27B; Qwen3.5 (the small Qwen3.5 0.8B is used by the tests) | dense, with its MTP block; all of it on the devices |
| `qwen4exp` | Qwen3.8 Flash-Next | mixture of experts; the experts live in VRAM slots, in a RAM tier and in the model file on the SSD |

Any other architecture is refused at load ("this build runs qwen35, qwen4exp"),
and so is a file that names none. A split model is loaded from any of its parts.

## Weight types

All 28 GGUF weight types load, in every tensor:

| Family | Types |
| --- | --- |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` |
| I-quants | `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` |
| Legacy quants | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` |
| Ternary and low-bit | `TQ1_0`, `TQ2_0`, `Q1_0`, `Q2_0` |
| 4-bit floats | `MXFP4`, `NVFP4` |
| Floats | `F32`, `F16`, `BF16` |

On Flash-Next the CPU computes experts that are missing on the card. Its fast
kernels cover the types of the Flash-Next files that were at hand; experts in
other types take a slower path on the CPU.

## K and V formats

`--kv` sets how K and V are stored, for both models.

| Format | Storage |
| --- | --- |
| `f16` | 16-bit floats |
| `q8` | 8-bit (default) |
| `q7`, `q6`, `q5`, `q4` | 7 to 4 bits; a tile of 128 cells is stored relative to its mean, and the first tile and the newest cells stay exact |

Two names set K and V apart: `--kv q8,f16` is K in `q8` and V in `f16`. Fewer bits
leave room for more context. On Flash-Next, `q7` to `q4` change how a long prompt
is read alone (steps of up to 8192 rows instead of passes layer by layer, see
`--batch-size` in [options.md](options.md)).

## Drafters

A drafter proposes the next tokens; the model verifies them in one step. How
many drafts a step verifies is chosen for every step from the measured price of
a step and the acceptance. On the 27B the text with drafts is token for token
the text without.

| `--drafters` | Drafter | Note |
| --- | --- | --- |
| `mtp` | the MTP block in the model file | the default on the 27B when the file has one; refused for a file without one |
| a file of architecture `qwen35-mtp` or `qwen4exp-mtp` | the MTP block of the model, from a file of its own | Flash-Next's block, for example `mtp-q2_0.gguf` |
| a file of architecture `dflash` | a DFlash2 block drafter | not under a layer split |
| `none` | no drafter | the server does not speculate at all |
| (always) | the n-gram drafter | takes part next to a model drafter, never alone |

One MTP block and one DFlash drafter may take part together, separated by a
comma. `--draft-vocab` limits the tokens a model drafter proposes (the ready
lists and how to make one are in [draft-vocab.md](draft-vocab.md)), and
`--window` the rows of a sequence's window; both need a model drafter.

Refused drafter files, by what they are: DSpark drafters (all their variants),
DFlash 1 drafters (without the selector), Spark drafters (gated attention), and
drafters with an embedding table, an output head or a vocabulary of their own.
A request can turn drafting off for itself with `"draft": false`.

## Images

`--mmproj <FILE>` names the model's projector, the `mmproj` GGUF (architecture
`clip`) beside the model file; the projector of another model is refused.
Without it, a request with an image is refused. Images come as data URLs in all
three API styles (see [api.md](api.md#images) and [options.md](options.md#images)).

## What was run

Setups that were run, with what they needed. The numbers are measurements of
one setup on the date given, not general claims.

| Model file | Cards and RAM | Settings | Date | What it needed or gave |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B UD-Q4_K_S | RTX 4070 + RTX 3060, 12 GiB each, with peer access | tensor split 0.625 / 0.375, 262144 cells for 2 sequences, `--kv q5`, MTP and n-gram drafts, greedy sampling | 2026-11-23 | VRAM at the end 11.5 / 10.1 GiB; decode 71-74 / 78-86 / 103-104 tokens/s on German / English / code |
| Qwen3.8 Flash-Next IQ2_XS | the same two 12 GiB cards | `--ram 8G`, MTP block from its file | 2026-11-16 | at `--ram 8G` a quarter of the experts were only on the SSD |
| Qwen3.8 Flash-Next UD-Q3_K_XL | one RTX 4060 Ti 16 GB, 62 GiB of RAM | `--ram 40G`, MTP block from its file | 2026-11-22 | the model's context of 262144 tokens, the K and V beyond 28,544 tokens in RAM |
| Qwen3.8 Flash-Next UD-Q4_K_XL, 104 GiB | the same RTX 4060 Ti 16 GB, 62 GiB of RAM | `--ram 40G` | 2026-11-22 | the file is larger than RAM: the experts that VRAM and RAM do not hold are read from the SSD while it runs |

## Recommended settings

The 27B on two 12 GiB cards, the first row above, with the general draft
vocabulary of this repository, for English prose, code and shell
([draft-vocab.md](draft-vocab.md) says how to make one for another language):

```sh
ares --model ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf --devices 0,1 \
  --split tensor --shares 0.625,0.375 \
  --context 262144 --parallel 2 --kv q5 \
  --drafters mtp --draft-vocab docs/user/draft-vocab/qwen35-general.txt \
  --ram 16G --cache ~/.cache/ares-prompts --cache-size 48G \
  --reasoning-effort medium
```

Flash-Next with its MTP block from a file of its own, and 40 GiB of RAM for the
experts and the prompt cache, as in the third row above:

```sh
ares --model ~/models/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf \
  --drafters ~/models/mtp-q2_0.gguf --ram 40G --cache ~/.cache/ares-prompts
```

Without options the server plans for itself, and the log says what it chose.
[features.md](features.md) lists everything the models can do.
