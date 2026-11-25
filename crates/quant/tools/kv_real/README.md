# kv_real: the engine's KV formats on captured attention data

Results and method: `docs/dev/studies/kv-real-data.md`. This directory is not a
workspace member (it has its own `[workspace]`); it builds against
`crates/quant` by path. Commands below are the ones that were used. CPU only:
`CUDA_VISIBLE_DEVICES=""`, `-dev none`, `-ngl 0`, 6 threads at `nice -n 10`.

`L` is a llama.cpp tree that is only read: a fresh clone, built on the CPU as
below (skip the clone and the build when `LLAMA_CPP_DIR` names one already built).
The numbers of the study were taken with the owner's fork
(https://github.com/Piggidragon/llama.cpp) at `f8f346786`; check that out to
repeat them, or take upstream (https://github.com/ggml-org/llama.cpp).

```sh
L=${LLAMA_CPP_DIR:-/var/tmp/llama.cpp}
git clone https://github.com/ggml-org/llama.cpp $L
cmake -S $L -B $L/build -DGGML_CUDA=OFF
cmake --build $L/build -j 6 --target llama llama-common
```

```sh
W=/var/tmp/kvreal-data                       # any directory with ~1.5 GB free
T=$(git rev-parse --show-toplevel)/crates/quant/tools/kv_real
mkdir -p $W/q08 $W/q27b $W/corpus $W/build
```

## 1. The text (3 686 tokens with the Qwen3.5 / Qwen3.8 tokenizer)

```sh
python3 -I $T/corpus/assemble.py $W/corpus/prompt.txt
```

`corpus/segments.txt` (token ranges of the kinds of text, used by `run` for the
per-text tables) is copied next to the data in step 3.

## 2. The capture program (against libllama of `L`, nothing is written there)

```sh
g++ -std=c++17 -O1 -o $W/build/kv-capture $T/capture/capture.cpp \
  -I$L/include -I$L/common -I$L/ggml/include -I$L/vendor \
  -L$L/build/bin -lllama-common -lllama -lggml -lggml-base -lpthread \
  -Wl,-rpath,$L/build/bin
```

## 3. Capture

Per attention layer it writes `Lxx_q.f32` (`[token][n_head][256]`),
`Lxx_k.f32`, `Lxx_v.f32` (`[token][n_kv][256]`), `Lxx_attn.f32` (the model's own
attention output, for `validate`) and `tokens.txt`.

```sh
# Qwen3.5-0.8B, all 6 attention layers (~430 MB, 15 s)
KVCAP_OUT=$W/q08 CUDA_VISIBLE_DEVICES="" nice -n 10 $W/build/kv-capture \
  -m $HOME/Services/models/test-small/Qwen3.5-0.8B-Q4_K_M.gguf \
  -f $W/corpus/prompt.txt -ngl 0 -dev none -t 6 -c 4096 -b 4096 -ub 512

# Qwen3.8-27B, layers 3,15,31,47,63 of 16 (~1.0 GB, 6 min; needs ~20 GiB RAM
# available because the model is mapped)
KVCAP_OUT=$W/q27b KVCAP_LAYERS=3,15,31,47,63 CUDA_VISIBLE_DEVICES="" nice -n 10 \
  $W/build/kv-capture \
  -m $HOME/Services/models/llama-cpp/Qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_S.gguf \
  -f $W/corpus/prompt.txt -ngl 0 -dev none -t 6 -c 4096 -b 4096 -ub 512

cp $T/corpus/segments.txt $W/q08/ ; cp $T/corpus/segments.txt $W/q27b/
```

Wrap either in `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0 --`
to cap memory.

## 4. kvarn's original routines (for `validate`)

```sh
git clone --depth 1 https://github.com/Anbeeld/beellama.cpp.git $W/beellama
git -C $W/beellama rev-parse HEAD   # 1cef3bf6dc53386b26f22aab54460317879d39eb
B=$W/beellama
g++ -std=c++17 -O1 -fno-fast-math -ffp-contract=off -c $B/src/llama-kvarn.cpp \
  -I$B/src -I$B/include -I$B/ggml/include -o $W/build/llama-kvarn.o
g++ -std=c++17 -O1 -ffp-contract=off -o $W/build/kvarn_ref \
  $T/kvarn_ref/kvarn_ref.cpp $T/kvarn_ref/stubs.cpp $W/build/llama-kvarn.o \
  -I$B/src -I$B/include -I$B/ggml/include -L$L/build/bin -lggml-base \
  -Wl,-rpath,$L/build/bin
```

## 5. The tool

Use a target directory of your own (not the workspace's `target/`):

```sh
export CARGO_TARGET_DIR=/var/tmp/kvreal-target RUSTFLAGS="-C target-cpu=native"
cd $T && cargo build --release --offline
K=$CARGO_TARGET_DIR/release/kvreal

$K validate --data $W/q08 --kvarn-ref $W/build/kvarn_ref --work $W/validate
$K validate --data $W/q27b --heads 24 --kv 4 --hd 256      # attention check only
$K stats    --data $W/q08
nice -n 10 $K run --data $W/q08 --stride 2 --threads 6 --out results_q08.tsv
nice -n 10 $K run --data $W/q08 --set extra|sweep|tilemean --stride 2 --out ...
nice -n 10 $K run --data $W/q27b --heads 24 --kv 4 --hd 256 --stride 4 \
  --masks none,policy,kvarn --pairs 44,55,66,88,54 --out results_q27b.tsv
```

`run` takes `--only a,b` (candidates whose name contains any of the strings,
for example `--only 'engine|K4V4|kvarn,engine|K4V4|policy'`), `--masks`,
`--pairs`, `--layers`, `--tmin`. `tables.py FILE main|whatif|side|layers|bins|snr|gt`
and `decomp.py FILE` print markdown tables from a result file.

## 6. The built mean per tile (`--set built`)

Section 13 of the notes. These candidates call the codec of `ares-quant` as it
is built (`encode_tile`/`decode_tile`, `KvPolicy::quantized_cells`), so they
change when `crates/quant` does:

| name | codec | exact cells |
| --- | --- | --- |
| `built\|KbVb\|kvarn` | `encode_tile`/`decode_tile`, tiles of 128 | first 128 + the open group (as `engine+tile-mean(128)\|..\|kvarn`) |
| `built\|KbVb\|policy` | the same | `KvPolicy::new(k, v).quantized_cells(t + 1)` for a query at position `t`: whole tiles between the two windows, the rest f16 |
| `rows\|KbVb\|tile-policy` | `encode_row`/`decode_row` (no mean) | exactly those of `built\|..\|policy` |

Pairs: 8/8, 6/6, 5/5, 4/4, 6/5, 5/4, 8/4, 4/8 (`built|..|kvarn` only for the
first six). The tail `policy` of the earlier sets is the sliding policy with
the defaults of the first measurements (first 128, 64, 32 cells at q4, q5, q6;
newest 256, 128, 64), fixed in `src/main.rs`; it does not follow `KvPolicy::new`,
so the numbers of sections 4 to 9 stay reproducible. `run` also prints `E` lines:
the mean number of exact cells per query for the old sliding policy (`policy`),
the tile policy (`tile-policy`) and kvarn's tail (`kvarn`), per pair.

```sh
$K run --data $W/q08 --set built --stride 2 --threads 6 --out results_built_q08.tsv
$K run --data $W/q27b --heads 24 --kv 4 --hd 256 --set built --stride 4 --threads 6 \
   --out results_built_q27b.tsv
# tables of section 13 (reference rows from the results of the earlier sets):
python3 -I $T/built_tables.py results_q08.tsv results_q08_tilemean.tsv results_q08_extra.tsv \
   -- results_built_q08.tsv [main|scopes|steps|worse|snr|exact|check]
```

`kvreal validate` also compares the built codec with the row codec (8 bits:
bit-identical) and with the what-if of section 6 (SNR, rms difference).

## 7. The stored sequence through the reference attention (`--set head`)

Section 14 of the notes. `head|KbVb|policy` stores the captured K and V of a
head in `ares_quant::KvHead` under `KvPolicy::new(k, v)` (cell by cell, so a
tile is converted from its stored `f16` values) and runs every sampled query
through `KvHead::attend` over cells `0..=t`, the reference the attention
kernels are held to. It is compared with `built|KbVb|policy`, the same codec
and policy on arrays (tiles encoded from the `f32` originals, `f32`
attention). `run` also prints `D` lines: the distance of the head's output
from the arrays' output, query by query (relative L2: mean, 99th percentile,
maximum). Pairs as in section 6.

```sh
$K run --data $W/q08 --set head --stride 2 --threads 6 --out results_head_q08.tsv
$K run --data $W/q27b --heads 24 --kv 4 --hd 256 --set head --stride 4 --threads 6 \
   --out results_head_q27b.tsv
python3 -I $T/built_tables.py results_q08.tsv -- results_head_q08.tsv head
```
