# capture: what llama.cpp computes for `qwen35`, block by block

`capture.cpp` links the shared libraries of llama.cpp-native (read only, nothing is
written there), runs a prompt and then generates tokens one at a time, and writes
the output of every block, the output of the final norm and the logits. The Rust
reference (`crates/model`) is compared with it by `examples/compare.rs` and by the
tests in `tests/real_models.rs`. This directory is not a workspace member and has
no Cargo project. Everything below is CPU only: `CUDA_VISIBLE_DEVICES=""`,
`-ngl 0 -dev none`, 6 threads at `nice -n 10`. (llama.cpp still prints that it
found no CUDA device; that is the point.)

```sh
W=/var/tmp/ares-model-capture                  # about 0.8 GB for all runs below
T=$HOME/Services/Engine/crates/model/tools/capture
L=$HOME/Services/llama.cpp-native  # built, read only
M=$HOME/Services/models/test-small/Qwen3.5-0.8B-Q4_K_M.gguf
mkdir -p $W/build
```

## 1. Tokens

Three prompts of `prompts/` (English 231 tokens, German 253, code 241). The ids
come from the tokenizer of llama.cpp. The prebuilt `build/bin/llama-tokenize` of
the fork is older than its libraries (2026-10-20 against 2026-11-03) and crashes
in `common_params_parser_init`, so the tool is compiled from the fork's own
`tools/tokenize/tokenize.cpp` against the current libraries, outside the checkout:

```sh
g++ -std=c++17 -O1 -o $W/build/llama-tokenize $L/tools/tokenize/tokenize.cpp \
  -I$L/include -I$L/common -I$L/ggml/include -I$L/vendor \
  -L$L/build/bin -lllama-common -lllama -lggml -lggml-base -lpthread -Wl,-rpath,$L/build/bin
for p in english german code; do
  CUDA_VISIBLE_DEVICES="" $W/build/llama-tokenize -m $M -f $T/prompts/$p.txt --ids --no-bos \
    --show-count --log-disable > $W/$p.ids.txt
  python3 -I -c "import re,sys; s=open(sys.argv[1]).read(); print(' '.join(re.search(r'\[([0-9, ]+)\]',s).group(1).replace(',','').split()))" \
    $W/$p.ids.txt > $W/$p.tokens
done
```

## 2. The capture program

```sh
g++ -std=c++17 -O1 -o $W/build/capture $T/capture.cpp \
  -I$L/include -I$L/common -I$L/ggml/include -I$L/vendor \
  -L$L/build/bin -lllama-common -lllama -lggml -lggml-base -lpthread -Wl,-rpath,$L/build/bin
```

The header comment of `capture.cpp` lists the environment (`CAP_TOKENS`, `CAP_OUT`,
`CAP_GEN`, `CAP_NODES`, `CAP_LIST`) and the files. The prompt is decoded in one
batch with every row an output (otherwise the last block is narrowed to the rows
that need logits, `crop_before_nextn` in `src/llama-graph.h`); then 40 tokens are
picked greedily from the logits and fed back one at a time.

## 3. Runs

```sh
run() {  # run NAME PROMPT [llama args...]
  name=$1 p=$2; shift 2; mkdir -p $W/$p/$name
  CAP_OUT=$W/$p/$name CAP_TOKENS=$W/$p.tokens CAP_GEN=40 CUDA_VISIBLE_DEVICES="" \
    systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- nice -n 10 \
    $W/build/capture -m $M -ngl 0 -dev none -t 6 -c 4096 -b 512 -ub 512 "$@" > $W/$p/$name.log
}
for p in english german code; do
  run base  $p                          # the defaults: flash attention, f16 K and V
  run t3    $p -t 3                     # 3 threads (the later -t wins): llama.cpp against itself
  run kvf32 $p -ctk f32 -ctv f32        # f32 cache type; flash attention still casts K and V to f16
  run nofa  $p -fa off -ctk f32 -ctv f32 # attention in f32 without flash attention
done
```

The numbers measured with these runs are in
`docs/dev/studies/qwen35-architecture.md`, section "Proof against llama.cpp".

`CAP_NODES` keeps further graph nodes by name, for example
`CAP_NODES=attn_norm-0,linear_attn_out-0,attn_post_norm-0,ffn_out-0`; the names are
the ones `src/models/qwen35.cpp` gives with `cb(...)`. `CAP_LIST=1` prints the
operator of every node of one generation step, which is how it was established that
a step runs `SSM_CONV`, `GATED_DELTA_NET` (fused, sequential per token on the CPU)
and `FLASH_ATTN_EXT`.

## 4. Comparison

```sh
export CARGO_TARGET_DIR=$HOME/Services/Engine/target/agent-model
cargo build --release -p ares-model --example compare
C=$CARGO_TARGET_DIR/release/examples/compare
$C $M $W/english/base [--all-logits]         # reference against llama.cpp
$C --captures $W/english/base $W/english/t3  # llama.cpp against itself
```

The tests that use these files skip with a message unless the variables are set:

```sh
ARES_TEST_QWEN35_0_8B=$M ARES_TEST_QWEN35_CAPTURE=$W \
ARES_TEST_QWEN38_27B=$HOME/Services/models/llama-cpp/Qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_S.gguf \
  cargo test --release -p ares-model --test real_models
```

## 5. The made-up model

`cargo run --release -p ares-model --example synthetic -- $W/synth f32` writes the
model of `tests/common` (eight blocks, three value heads per key head, grouped query
heads, weights in f32) and a prompt of 48 tokens. llama.cpp loads it as it is (the
file has a vocabulary of 128 placeholder tokens):

```sh
CAP_OUT=$W/synth/nofa CAP_TOKENS=$W/synth/prompt.tokens CAP_GEN=40 CUDA_VISIBLE_DEVICES="" \
  nice -n 10 $W/build/capture -m $W/synth/synthetic.gguf -ngl 0 -dev none -t 6 -c 1024 \
  -b 512 -ub 512 -fa off -ctk f32 -ctv f32
$C $W/synth/synthetic.gguf $W/synth/nofa
```

With f32 weights llama.cpp quantizes no activation, and without flash attention
nothing is cast to f16, so what remains between the two is rounding. The constants of
`the_made_up_model_gives_what_llama_cpp_computed_for_it` in `tests/qwen35.rs` are from
this run.

## 6. How the quantized activations were shown to be the cause

This is a measurement that is not part of the reference (the reference does not
imitate llama.cpp), done with a temporary patch of `Reference::matmul` that rounded
every activation row through ggml's `Q8_K` (for `q4_K`, `q5_K`, `q6_K` weights:
blocks of 256, `iscale = -127 / max`, `nearest_int`, `d = 1 / iscale`, from
`quantize_row_q8_K_ref` in `ggml/src/ggml-quants.c`) or `Q8_0` (blocks of 32,
`d = max / 127` stored as f16) before the multiplication. With the patch, the first
matmul of block 0 (`attn_qkv`, input `attn_norm-0` of llama.cpp) agreed with
llama.cpp for the median prompt row to 7e-8 relative where it differed by 4e-3
without it, and the feed-forward part of block 0, fed with `attn_post_norm-0`, to
2.1e-7 (median row) where it differed by 2.7e-2 without it. Rows that did not agree
to rounding were rows where an input element sat at a rounding boundary of the
quantizer: one step of one quantized value changes that row by about 1e-3. The patch
is not kept because the quantizer is a step function and the comparison through many
blocks is then chaotic (a tiny difference upstream flips a rounding step downstream);
the isolated-block comparison of `compare` is the measure that does not accumulate.
