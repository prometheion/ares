# Experiments: images

Dated measurement logs, moved here verbatim from `ARCHITECTURE.md` and `OPEN.md` on 2026-11-24. They are not kept current: each names its date, build and machine.

## The tower on the device, measured (2026-11-16)

**Measured** (2026-11-16, the 27B's projector, `examples/qwen3vl_tower.rs`:
medians of three passes timed by events on the device, the homes taking turns;
images of 256, 1024 and 4096 rows; load average 26 to 36):

| | 256 rows | 1024 rows | 4096 rows |
| --- | --- | --- | --- |
| RTX 4070, weights in VRAM | 27.1 ms | 141.7 ms | 1017 ms |
| RTX 4070, weights in RAM | 38.2 ms | 144.7 ms | 1019 ms |
| RTX 3060, weights in VRAM | 63.1 ms | 310.6 ms | 2306 ms |
| RTX 3060, weights in RAM | 279.8 ms | 342.9 ms | 2330 ms |
| the attention's share (both cards) | 11 % | 29 to 30 % | 60 to 62 % |

- *VRAM held between images*: 885.6 MiB in VRAM, 58.3 MiB (the two slots) in
  RAM, which holds 885.6 MiB of page-locked RAM instead. The scratch of an
  image is 328.6 MiB for 4096 rows (as the driver reports, 330 MiB).
- *The uploads* of the RAM home take 37.2 to 37.8 ms on the RTX 4070 (25 GB/s)
  and 279 ms on the RTX 3060 (3.3 GB/s, its link of four lanes) when they run
  alone. A pass hides 70 % of them at 256 rows on the RTX 4070 and 92 to 94 %
  from 1024 rows on; on the RTX 3060 an image of 256 rows waits for its weights
  (22 % hidden: the pass computes in 63 ms what takes 279 ms to arrive), from
  1024 rows on 88 to 91 % are hidden. What is not hidden there is the first two
  units, which no computation precedes.
- *The attention* reaches 45 to 55 TFLOPS on the RTX 4070 and 19 to 23 on the
  RTX 3060 (counting `4 * patches^2 * 1152` operations a block, the padding of
  the heads not counted). The rest of a pass runs at about 33 TFLOPS on the
  RTX 4070 where the tile kernels alone reach 50 at these sizes: the row kernels
  in between (norms, rotation, GELU, residual, the conversion of the
  attention's output) move about 560 MB a block at 1024 rows, about a third of
  the time outside the attention. Fusing more of them (the residual with the
  norm after it, a bias into the products' epilogue) would take part of that
  away; not built. A GELU cannot go into the epilogue as it is: the f16 form
  needs the largest value of the whole row first.
- Loading the tower takes 0.6 to 2.2 s with the file in the page cache.
- Flash-Next's projector takes the same time (VRAM home: 28.3 and 140.8 ms on
  the RTX 4070, 62.5 and 313.0 ms on the RTX 3060, at 256 and 1024 rows).
- *f16 sums* flushed into `f32` every 64 values (the faster path of the capped
  cards) in every product but `ffn_down` and the patch matrix: 8 to 9 % faster
  at 1024 rows and 4.5 % at 4096 on both cards, and up to twice as far from the
  exact rows (4.0e-4 to 1.18e-2, at most 60 % of the tolerance). Not taken: the
  output would get worse.

## Images in a prompt, checked and measured (2026-11-22)

**Checked against llama.cpp** (2026-11-22, the 27B UD-Q4_K_S and its
`mmproj-F16` on both cards, a tensor split; the fork's `llama-server` on the
same cards with `-ngl 99 -fa on`, both at temperature 0, thinking off): three
images (640 x 480 shapes, 768 x 256 of text, a 1024 x 768 bar chart) give the
fork's prompt token counts exactly (320, 218, 788). With `--kv f16 --precision
exact`, the replies' top five probabilities were compared token by token: the
shapes image gave the same 80 tokens (the largest difference of a candidate's
probability 0.07, against 0.04 for a prompt of text alone, which also gave the
same 80 tokens); the image of text gave the same 19 tokens, where the fork
stands at a three-way near tie (0.35, 0.31, 0.29) and Ares ends the reply.
The two towers' rows differ by their roundings (llama.cpp's are 0.2 to 6 % from
the exact rows on the CPU), which moves the first token's probability from
0.58 to 0.80 on the image of text. With the defaults (`q8` K and V, 8-bit
activations for few rows) the replies leave the fork's after about ten
tokens, as a prompt of text does. A test on the 0.8B with made-up rows
(`tests/qwen35_images.rs`) holds the device to the CPU reference with the rows
at their rotary positions, and checks that the positions change the logits.
Flash-Next IQ2_XS (`mmproj` of its own, `--ram 8G`, both cards) describes the
same images correctly on all three routes.

**Measured** (the same run, defaults, load 25 to 35 from other agents' builds):
an image of 300 rows is read in 452 ms with its 20 tokens of text, the tower
included (its first image; weights in RAM, on the RTX 4070), and the reply
generates at 73 tokens/s with the MTP block's drafts (103 of 157 accepted); an
image of 768 rows in 924 ms. Sent again, the prompt resumes from the cache at
cell 256 of 320 (the cells of the image after it come from the store, the tower
does not run): 150 ms. The tower takes 394.9 MiB of the RTX 4070 (two slots and
the scratch of 4096 rows), and 863 MiB of page-locked RAM.
