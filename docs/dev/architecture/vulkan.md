# Vulkan

`ares-vulkan` gives the same surface (devices, buffers, ordered work,
events, kernels) on Vulkan 1.2 or newer, with the same first kernels as GLSL
compute shaders held to the same tests. It is the generic backend for every
GPU that CUDA 13 does not run on.

- Kernels take buffer addresses and sizes in push constants, exactly as the
  CUDA kernels take pointers. There are no descriptor sets, so a launch needs
  no pipeline and no descriptor write: shapes are data here too.
- A stream records commands into a command buffer and submits them in
  batches. Recording a launch costs under a microsecond; submitting every
  launch on its own would cost about ten.
- A device has one queue, so streams of one device do not run concurrently,
  and separate devices have no events between them: data between devices goes
  through the host.
- Its Q4_K kernel is not a port of the CUDA baseline but already restructured
  (a warp reads one block with consecutive weight and activation loads, a
  workgroup computes four output rows): 17408 x 5120 with one activation row
  takes 0.14 ms on the RTX 4070 (363 GB/s of weights) and 0.26 ms on the RTX
  3060 (194 GB/s). The straight port of the CUDA baseline's layout measures
  69 and 30 GB/s on Vulkan, which points at that layout, not at the backend,
  as what holds the CUDA baseline back.
- The twelve weight types that have a device layout on CUDA have it on
  Vulkan too, rearranged by a shader into the same bytes
  (`ares_quant::layout` is the specification both are tested against byte
  for byte), so one upload format serves both backends. One family of
  kernels on cooperative matrices serves every row count from that copy,
  from the first row, with a table per device from a measurement at load
  and the same API as CUDA's. The other types run their file-layout kernels
  (integer dot products for few rows, cooperative matrices through a decoder
  per type for many). Measured 2026-11-11, a pass over all weights of
  Qwen3.8-27B on the two cards (split 0.625 / 0.375), device layout against
  file layout: one row 21.9 against 21.6 ms, eight rows 24.0 against 40.7,
  sixteen 33.6 against 96.8, sixty-four 68.1 against 102.4. CUDA's device
  layout took 23.4, 25.4, 31.0 and 69.7 ms for the same passes (before its
  few-row kernels). At 1024
  rows a Q4_K matrix runs at 88.5 / 39.9 TOPS on the RTX 4070 / RTX 3060
  and Q6_K at 75.5 / 34.3 *(measured by the agent that built the kernels;
  CUDA: 88.8 / 39.3 and 74.5)*.
- The layer kernels exist with CUDA's semantics and are held to the same
  references.
- Attention exists for every stored K/V format (`f16`, `q8` and the formats
  with a tile mean down to `q4`, separately for K and V), over dense ranges,
  lists of ranges and prefixes that several sequences share, with the kernels
  that append new cells and convert exact tiles. The bytes it writes are the
  CPU reference's (`ares_quant::kv`, which CUDA's are held to as well),
  checked byte for byte; its
  output equals `KvHead::attend` within 1.1e-5 of the largest output element
  on random rows, on the captured rows of Qwen3.8-27B and for every pair of
  formats and both head sizes, on both cards (the reference's tolerance for
  an `f32` kernel is 2e-5; CUDA's tensor-core kernels are held to 8e-3).
  There is no tensor-core instruction in it. An item (some rows of one KV head
  over a list of ranges, as the CUDA plan cuts them) is one workgroup that
  takes a cell per invocation for the scores and a slice of the columns per
  invocation for V, in `f32`, with the rows decoded in registers; it holds up
  to 16 vectors (the query heads of its rows; 8 on a device with less than
  28 KiB of shared memory) where CUDA's hold 64. The planner aims for three
  workgroups for each multiprocessor, as CUDA's does; Vulkan does not report
  their count, so it is measured at load, by how many workgroups of 1024
  invocations the device runs at once (46 and 27 on these cards, the same in
  every load). The device needs what the quantized weight kernels need and
  shuffles in a subgroup.
  Measured 2026-11-12 on the shape of Qwen3.8-27B (head size 256, 4 KV heads
  of 6 query heads), one attention layer, the best of two runs, Vulkan next
  to CUDA on the same card, `ARES_LIBCUDA` from the p2p bench:

  | case (per layer) | RTX 4070 Vulkan us | RTX 4070 CUDA us | ratio | RTX 3060 Vulkan us | RTX 3060 CUDA us | ratio |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | f16/f16 1 rows over 1024 | 27 | 20 | 1.36 | 41 | 47 | 0.87 |
  | f16/f16 8 rows over 1024 | 62 | 34 | 1.84 | 121 | 109 | 1.11 |
  | f16/f16 64 rows over 1024 | 229 | 174 | 1.32 | 611 | 335 | 1.82 |
  | f16/f16 1 rows over 32768 | 358 | 307 | 1.16 | 494 | 526 | 0.94 |
  | f16/f16 8 rows over 32768 | 943 | 415 | 2.27 | 2,336 | 818 | 2.86 |
  | f16/f16 64 rows over 32768 | 7,023 | 3,229 | 2.17 | 16,779 | 6,101 | 2.75 |
  | f16/f16 1 rows over 262144 | 2,568 | 2,267 | 1.13 | 3,504 | 3,957 | 0.89 |
  | f16/f16 8 rows over 262144 | 7,928 | 2,728 | 2.91 | 18,885 | 5,822 | 3.24 |
  | f16/f16 64 rows over 262144 | 58,219 | 24,983 | 2.33 | 137,887 | 47,610 | 2.90 |
  | f16/f16 list of 2051 over 32768 | 38 | 32 | 1.19 | 76 | 92 | 0.83 |
  | q8/q8 1 rows over 1024 | 55 | 28 | 2.01 | 83 | 49 | 1.68 |
  | q8/q8 8 rows over 1024 | 108 | 40 | 2.74 | 226 | 115 | 1.97 |
  | q8/q8 64 rows over 1024 | 361 | 206 | 1.75 | 1,048 | 382 | 2.75 |
  | q8/q8 1 rows over 32768 | 493 | 217 | 2.27 | 888 | 480 | 1.85 |
  | q8/q8 8 rows over 32768 | 1,776 | 382 | 4.65 | 4,175 | 782 | 5.34 |
  | q8/q8 64 rows over 32768 | 45,696 | 3,378 | 13.53 | 32,521 | 6,335 | 5.13 |
  | q8/q8 1 rows over 262144 | 3,463 | 1,376 | 2.52 | 6,688 | 3,552 | 1.88 |
  | q8/q8 8 rows over 262144 | 15,954 | 2,447 | 6.52 | 32,716 | 5,396 | 6.06 |
  | q8/q8 64 rows over 262144 | 103,061 | 25,989 | 3.97 | 260,029 | 49,401 | 5.26 |
  | q8/q8 list of 2051 over 32768 | 71 | 39 | 1.81 | 147 | 72 | 2.06 |
  | q4/q4 1 rows over 1024 | 60 | 46 | 1.29 | 78 | 74 | 1.05 |
  | q4/q4 8 rows over 1024 | 106 | 41 | 2.55 | 200 | 105 | 1.90 |
  | q4/q4 64 rows over 1024 | 375 | 210 | 1.79 | 860 | 391 | 2.20 |
  | q4/q4 1 rows over 32768 | 426 | 196 | 2.17 | 818 | 445 | 1.84 |
  | q4/q4 8 rows over 32768 | 1,655 | 411 | 4.03 | 4,146 | 934 | 4.44 |
  | q4/q4 64 rows over 32768 | 12,787 | 4,208 | 3.04 | 31,231 | 7,988 | 3.91 |
  | q4/q4 1 rows over 262144 | 3,056 | 930 | 3.29 | 6,193 | 2,587 | 2.39 |
  | q4/q4 8 rows over 262144 | 12,978 | 2,701 | 4.80 | 33,379 | 6,082 | 5.49 |
  | q4/q4 64 rows over 262144 | 102,631 | 33,304 | 3.08 | 251,630 | 63,071 | 3.99 |
  | q4/q4 list of 2051 over 32768 | 80 | 61 | 1.30 | 137 | 117 | 1.17 |

  Where Vulkan is behind and what bounds it. With one row of `f16` over
  32768 or more cells it is within 16 % of CUDA on the RTX 4070 (418 GB/s of
  the card's 504 over 262144 cells) and 6 to 11 % faster on the RTX 3060:
  memory bound. With one row of `q8` or `q4` it takes 1.05 to 3.3 times as
  long on both cards: each invocation walks the row of its own cell through
  2-byte loads (the groups of the formats are 2-byte aligned) and unpacks the
  bit planes into `f32` before the products, so the kernel is bound by
  instructions and not by memory (165 and 99 GB/s on the RTX 4070 against
  CUDA's 413 and 321 for one row over 262144 cells). Windows of 8 and 64
  rows take 1.1 to 6.5 times as long, mostly 2 to 5: an item holds 16
  vectors, two rows of six heads, so a window of 8 rows is four items per KV
  head that each read every cell (CUDA: one item of 48 vectors on tensor
  cores), and 64 rows read them 32 times. A kernel for wide windows on the
  cooperative matrices that the weight kernels already use is not built.
  A list of 2051 cells takes 0.83 to 2.1 times as long as CUDA's.
  One entry stands out: `q8`, 64 rows over 32768 cells, on the RTX 4070 takes
  45.7 ms in both runs of the whole table, 13.5 times CUDA's. Alone in a
  process it takes 12.6 ms (3.7 times CUDA's) with a planner target of 78, 121
  or 138 workgroups, as it did in four earlier runs of the whole table, so it
  is not the plan; it depends on what ran before or where the pools lie, and
  the cause is not found.
- The sampler (`SamplerKernels`, `shaders/sample_*.comp`, `lp_*.comp`,
  `draft_*.comp`) has every step, record and entry point of CUDA's:
  corrections and masks, the token map and slices of the vocabulary as
  several devices hold them, the radix descents of top-k, top-p and typical
  sampling, top-n-sigma, min-p, XTC, temperature and its dynamic form,
  mirostat 1 and 2, the draw, the probabilities before and after the chain,
  and the draft kernels (picks into the next records, hidden-state gather,
  shift, keep and commit of the carry). `tests/sampler.rs` is the twin of
  CUDA's: 1312 rows of 41 settings, four shapes of rows and vocabularies
  from one token to Qwen's equal the records of `sample_reference` bit for
  bit on both cards, and so do rows with corrections and masks, with a token
  map, with the vocabulary cut into two or three slices whose exchange is
  added on the host, and in batches of different sizes that share one
  scratch area. A second test holds each function of the arithmetic (the
  weight, both logarithms, the exponential, the power, the Gumbel noise, the
  terms of a token, the division and the square root in double precision, the
  conversions of 64-bit integers) to `ares_sampler::arith` over 20 000
  random inputs from every range the sampler reaches, bit for bit on both
  cards. The code is CUDA's operation for operation, with these differences:
  - No 64-bit atomic (an optional feature on Vulkan, which the first
    mirror already avoided): every chunk of the vocabulary writes its best
    score, or its sums, into the scratch area, and the kernel that follows
    takes the maximum or adds them. The row's state is 128 bytes.
  - Subgroup operations of whatever width the device reports (at least 8)
    and shared memory for one value per subgroup; a maximum or a sum of
    64-bit numbers goes through 32-bit halves or 16-bit limbs, so that the
    64-bit subgroup types are not needed.
  - Division in single and double precision is corrected by fused
    multiply-adds to the correctly rounded quotient (the division of a
    shader is only within a few ulp), and every operation is `precise`, so
    that nothing is fused into a multiply-add.
  - **Double precision.** The reference uses IEEE doubles where a row needs a
    handful of numbers: the entropy and the deviation of a set, the dynamic
    temperature, the limit of mirostat 2, the exponent that mirostat 1
    estimates, the logarithm of the sum that a probability is divided by.
    The kernels repeat them with `double`, which needs the device feature
    `shaderFloat64`; both cards have it, and their addition, subtraction,
    multiplication, division (the corrected form),
    square root, and conversions between 64-bit integers, `f32` and `f64`
    equal Rust's in every bit tested. A device without `shaderFloat64` is
    refused at `SamplerKernels::load` with that reason; no other precision is
    substituted, since a sampler that diverged from the reference in the
    last bit would no longer give a seed the same text on both backends.
    (A soft-float library in 64-bit integers would give the bits on every
    device; it is not built.) Denormal results, and the order of NaNs and
    infinities in arithmetic, are left to the device: the shaders test the
    special logits on their bits, and a logit in the denormal range is not
    in the tests.
  - A device is also refused for less than 32 KiB of shared memory (the
    histograms of the descent) or workgroups of fewer than 256 invocations,
    with the numbers it reports. Nothing asks for a device's name.
- The expert and `qwen4exp` kernels and a model on Vulkan do not exist yet.
- No direct route between the two cards works on this driver: an exported
  allocation is imported by the other card without an error and carries no
  data. Sums go through host pages that both cards import, at the speed of
  CUDA's host route (55 against 47 µs for 10240 values; CUDA's peer route
  takes 18). Direct routes are built and untested for lack of hardware.
- The NVIDIA driver (615.78) has crashed at process exit in both
  directions: with devices still alive, and while destroying a device whose
  shader compiler thread was still working after hundreds of pipelines had
  been created. The tests leave without destroying their devices; the
  engine needs the same care at shutdown.
