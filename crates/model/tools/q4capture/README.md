# q4capture: what the llama.cpp fork computes for `qwen4exp`, layer by layer

`q4capture.cpp` links the shared libraries of llama.cpp-native (read only, nothing
is written there), runs a token sequence on the CPU and writes named nodes of the
graph: the two reads of every layer, the output of the recurrent or attention part,
the output of the experts, the residual after the layer, the routed experts with
their weights, the block scores and the blocks the indexer keeps, the final read and
the logits. `q4compare` (an example of `ares-model`) runs the Rust reference on the
same file and tokens and prints the distance per layer. This directory is not a
workspace member. Everything below is CPU only: `CUDA_VISIBLE_DEVICES=""`,
`-ngl 0 -dev none`.

The model is the made-up one of `ares_model::qwen4exp::synthetic` (section 8.3 of
`docs/reference/qwen4exp-architecture.md`): 8 layers (two attention), 3 streams of
32, 12 experts of which a token uses 3, 6 value heads on 2 key heads, 4 query heads
on 2 key/value heads, an indexer budget of 8 cells in blocks of 4, an n-gram layer
with a table of 424 rows. The fork loads it as it is.

```sh
W=/var/tmp/ares-q4capture
T=~/Services/Engine/crates/model/tools/q4capture
L=~/Services/llama.cpp-native    # built, read only
mkdir -p $W/build $W/synth
```

## 1. The model and its tokens

```sh
cargo run --release -p ares-model --example q4synth -- $W/synth f32 1 48
cargo run --release -p ares-model --example q4synth -- $W/synth shipped 1 48
```

`f32` writes every tensor as `f32`; `shipped` uses the unquantized types of the real
file where it has them (BF16 for the mixers, the indexer projections, `ssm_alpha`,
`ssm_beta`, `ple_key`, `ple_value`; F16 for `ple_conv1d`). The arguments after it
are the seed and the number of tokens (`prompt.tokens`).

## 2. The capture program

```sh
g++ -std=c++17 -O1 -o $W/build/q4capture $T/q4capture.cpp \
  -I$L/include -I$L/common -I$L/ggml/include -I$L/vendor \
  -L$L/build/bin -lllama-common -lllama -lggml -lggml-base -lpthread -Wl,-rpath,$L/build/bin
```

The header comment of `q4capture.cpp` lists the environment (`CAP_TOKENS`,
`CAP_OUT`, `CAP_CHUNK`, `CAP_NODES`, `CAP_LIST`), the nodes and the files.

## 3. Runs

```sh
run() {  # run NAME FILE [VAR=value...]
  name=$1 file=$2; shift 2; mkdir -p $W/$name
  env CAP_OUT=$W/$name CAP_TOKENS=$W/synth/prompt.tokens CUDA_VISIBLE_DEVICES="" "$@" \
    nice -n 10 $W/build/q4capture -m $W/synth/$file -ngl 0 -dev none -t 4 -c 1024 \
    -b 512 -ub 512 -fa off -ctk f32 -ctv f32 > $W/$name.log 2>&1
}
run f32-all     synthetic-f32.gguf                  # the sequence in one decode call
run f32-one     synthetic-f32.gguf CAP_CHUNK=1      # token by token
run f32-five    synthetic-f32.gguf CAP_CHUNK=5      # calls of five tokens
run shipped-all synthetic-shipped.gguf
```

`-fa off -ctk f32 -ctv f32` keeps K, V and the indexer keys in `f32`; with `f32`
weights llama.cpp then rounds nothing on the way.

## 4. Comparison

```sh
cargo run --release -p ares-model --example q4compare -- $W/synth/synthetic-f32.gguf $W/f32-all
```

The reference runs with the fork's selection rule (the best `top_k / B` whole
blocks and the tail). A third argument `budget` switches it to Strata's cell budget,
which shows what difference 1 of the specification does.

## 5. What was measured (2026-11-10, fork `prod/native` at `f8f346786`)

All-`f32` model, seed 1, 48 tokens; a row is compared by the RMS of the difference
over the RMS of the reference's row. 37 of the 48 rows are past the dense limit of
11 cells and select.

| Run | Residual after layer 0 / 3 / 7, worst row | Logits, worst row | Routed experts | Blocks kept | Greedy pick |
| --- | --- | --- | --- | --- | --- |
| one call | 5.1e-7 / 2.3e-6 / 1.3e-5 | 1.4e-5 | the same set in every layer and row | the same in both attention layers, all 37 rows | 48 of 48 |
| token by token | 8.0e-6 at worst | 9.3e-6 | the same | the same | 48 of 48 |
| calls of five | 1.3e-5 at worst | 1.4e-5 | the same | the same | 48 of 48 |
| seed 2, 200 tokens, one call | 1.4e-5 at worst | 1.4e-5 | the same | the same, 189 rows | 200 of 200 |
| seed 3, 120 tokens, calls of seven | 1.5e-5 at worst | 1.6e-5 | the same | the same, 109 rows | 120 of 120 |

Per layer the two reads, the output of the recurrent or attention part and the
output of the experts agree in the same range (1e-7 in layer 0 to a few 1e-5 in
layer 7; the median row stays below 6e-6). Block scores agree to 4e-4 of their
value (3e-3 for the smallest scores of the longer runs) after the fork's factor
`1 / sqrt(ID)` is removed. This is rounding: the reference sums in `f64`,
llama.cpp in `f32`.

With the reference on Strata's rule (`budget`) the two differ from the first
attention layer on, in exactly the rows whose tail has fewer than three cells (the
rows where the cell budget takes part of one more block), by up to 0.87 of the
attention output in a row: with a budget of 8 cells the three extra cells are a
quarter of what a row reads. In the real model they are 3 of 2051.

With the `shipped` types the fork is 1e-3 away in layer 0 already and ends with
other experts in some rows and 45 of 48 equal picks. The cause is llama.cpp's CPU
backend, not the architecture: its matrix product converts the activations to the
dot-product type of the weights, which for BF16 weights is BF16 (8 bits of
mantissa) and for F16 weights F16. A temporary patch of the reference that rounded
the activations the same way brought layers 0 and 1 back to 1e-7 .. 3e-6 (later
layers still drift, at 1e-4, which was not chased). The patch is not kept: the
reference does not imitate another implementation's rounding.

## 6. What this does not check

- Strata's selection rule (the cell budget), which the fork does not have, and the
  order of equal scores, which the fork leaves open: both are pinned by the unit
  tests of `pieces::select_cells` and `pieces::route` only.
- Quantized weights: on the CPU the fork rounds activations to 8 bits for them, so
  only the unquantized model is compared. The decoders are checked against ggml in
  `ares-quant`.
- Images, and the MTP draft block (the fork cannot read Strata's MTP file).
- The fork's own generator, `test-llama-archs -a qwen4exp -o DIR`: the prebuilt
  binary in `build/bin` is older than the libraries (2026-10-20 against 2026-11-03)
  and fails with "key not found in model: qwen4exp.ssm.group_count". It was not
  rebuilt; the made-up model above replaces it and has a small `top_k`, more experts
  than a token uses and several value heads per key head, which the fork's own
  model has not.
