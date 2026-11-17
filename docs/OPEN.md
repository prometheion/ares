# What is still open

Updated 2026-11-17, `main` at the shared layer's step 4, DFlash on both models, pruning and the
experts' read-ahead (nothing pushed). What is there is listed in
[`FEATURES.md`](FEATURES.md).
This file is brought up to date at every merge into `main`. It lists work, defects and
measurements that are owed; what a feature must do in detail is in
[`reference/feature-checklist.md`](reference/feature-checklist.md) (223 rows: 70 done,
42 partly, 88 not started, 23 outside the frame; its rows K01, K02, K06 and P14 are out
of date: reasoning is built, and `q7` to `q4` run in the 27B).

Order of work the owner has set: CUDA complete for both models first, then optimising (better
kernels, better prefetching: measured per phase on a quiet machine against the fork and Strata),
Vulkan parity last.

## In work now

| Piece | State |
| --- | --- |
| Shared model layer steps 5 and 6: token rows, one MTP driver, drafter kinds named `ngram`, `mtp`, `dflash` everywhere, MTP from a separate file for every model, the draft vocabulary for every drafter and model | built on `model/mtp-driver` (9 commits), not merged: two server tests of the 27B with drafts end in `CUDA_ERROR_ILLEGAL_ADDRESS`, three host tests expect the old behaviour, measurements and docs unfinished (the agent ran out of context) |
| Flash-Next's float-path tensors in all 28 types (UD-Q4_K_XL has them in Q8_0), every placement for any quantization mix that fits, refusals that name the real limit | built (`moe/q4-load`), to be verified and merged; what is left is below |

## Shared model layer (the owner's rule, 2026-11-16)
Generic for Ares, not like llama.cpp: a few models, shared code where it can be, each model
still at its maximum through its own launch plan. A part becomes a shared block when it owns
state the runtime touches (pages, slot state, checkpoints, windows, commits, replays, trees)
or when it drafts; products, norms, kernel choice per row count, fused launches and exchanges
between cards stay with the model. Today the two models share almost nothing inside
`crates/model`, which is why every parity item below is owed twice. Design study:
`/var/tmp/ares-survey/shared-model-layer.md`. Each step lands green on its own.
- [x] 0. A `blocks` module; the 27B's K/V code moved into it as it is.
- [x] 1. The K/V group as a block, the 27B onto it (bit for bit, same launches).
- [x] 2. The recurrent (delta-rule) block, both models onto it (bit for bit).
- [x] 3. Flash-Next onto the K/V group: K and V in `q7` to `q4`, windows as if alone.
- [x] 4. Slot state and checkpoints as one contract: Flash-Next's prefix cache with RAM and SSD
      (2026-11-16, `blocks/slots.rs`; the 27B moved onto it bit for bit).
- [ ] 5. Token rows (embedding on the host, in the mapped file or on the card; the image rows).
- [ ] 6. One driver for MTP, with the owner's rules (2026-11-16/17): as little per model as can
      be (the model gives its layer and its tap); one set of names for the drafter kinds,
      `ngram`, `mtp`, `dflash`, in types, files, metrics, options and docs; MTP from a
      separate file for every model (`--drafters mtp` for the block in the model file,
      `--drafters <file>` for a separate MTP or DFlash file); the draft vocabulary as one
      head view for every drafter that projects to the vocabulary, on every model. Brings
      Flash-Next's draft vocabulary and picks on the card, the 27B's drafts under a layer split.
- [ ] 7. Routes (`image_url`, Anthropic `image`, Responses `input_image`), limits, errors, `--mmproj`,
        `--mmproj-weights ram|vram` (`ram` by default; the owner, 2026-11-16).
        Embedded images only; fetching remote URLs is backlog. At most 4096 rows per image by
        default (as the fork), with a key to lower it.
  - [ ] 8. Caching: cells named by a digest of the resized pixels, 64-bit names through the index and the SSD level.
- [ ] Audio, video and Gemma 4 are backlog: none of them blocks the first release (the owner,
      2026-11-16).
- [ ] Jev mode: `/v1/systemone` exists and answers 501; `Engine::decide` on the real runtime.
- [ ] Turing.
- [ ] K/V of a running sequence beyond VRAM (P16; an exception, low priority).

## Weight types (`moe/q4-load`, 2026-11-17)
Every tensor of both models takes every weight type the shape allows (the audit of every place
that refused a type is in the branch's notes; ARCHITECTURE.md "Files in any type" and "A layer
with fewer slots in VRAM than a step picks"). Left:
- [ ] Gate and up of a layer's routed experts in two types are refused (the expert kernels read
      both with one decoder): kernels per pair of types or two launches; the owner to decide.
- [ ] A router of a type other than F32 is decoded to F32 at load, four bytes a weight (files
      keep it F32); its kernels (decode, prediction, wide steps) would take the type.
- [ ] Where VRAM holds fewer slots than a step picks, every layer's slots are at its floor, so
      the heat no longer moves slots between layers; the floors in RAM cost `--ram`: UD-Q4_K_XL
      needs 9G on one card and 10G with both cards on every layer (8G by layers).
- [ ] The DFlash drafter's float matrices take the f16 tile path above eight rows without the
      range check of Flash-Next's (`WeightRange`).
- [ ] Shapes, not types: rows whose width a type's block does not divide cannot be stored in it
      (Flash-Next's expert down projections of 640 values, the n-gram table's rows of 160, the
      image tower's `ffn_down` of 4304); the 27B's tensor split cuts in units of 256.

## Optimising (after CUDA is complete, before Vulkan; the owner, 2026-11-16)
Measured per phase on a quiet machine against the fork and Strata, the largest first. Known levers:
- [ ] The fused read of a Q8_0 mixer (UD-Q4_K_XL) takes 43.8 us on the RTX 4070 where BF16 takes
      24.9: it decodes byte by byte (a load of four codes at once would cut it); sixteen rare types
      read through the decoder chosen at run time; the float read is 2 % slower than before the
      types (5400 against 5112 instructions).
- [ ] An expert from the SSD is three reads (its matrices lie apart in the file): 2.8 to 3.6 ms
      where the drive needs 1.4 ms per read. One read per expert (a layout of the SSD level).
- [ ] Long prompts layer by layer with read-ahead; the profile's fill in large ordered reads.
- [ ] Few-row kernels for windows and trees; attention for wide windows; the F32 tile kernel
      (2.5 times BF16); NVFP4 and IQ2_XXS gaps; the image tower's row kernels fused.
- [ ] The sampler's exchange on the second card; a layout choice that is stable between loads;
      the MTP link (1.49 against 1.44 ms possible); DFlash's drafts-only step (4 to 5 ms).

## Vulkan (last)
- [ ] A backend-free crate and the `ares-gpu` facade; expert and Flash-Next kernels; both
      models on Vulkan in every placement; attention for wide windows (2 to 6.5 times slower
      than CUDA today); the delta rule's f16 state rounded per row; the sampler's `-0.0`.

## Measurements owed on a quiet machine
- [ ] Flash-Next IQ2_XS decode step before and after `moe/q4-load` (eight pairs under load 10 to 19:
      means 25.1 and 26.7 ms, within the spread; the read itself is 2 % slower).
- [ ] Flash-Next with the owner's real limit (a RAM tier of 24 to 26 GiB), one card and two.
- [ ] The SSD cache's times to the first token (taken under load so far).
- [ ] Flash-Next with its draft block once speculation pays; target above 56 tokens/s.
- [ ] Before a release: both models against the fork and against Strata, speed, VRAM, quality.

## Questions to the owner
- How long may a request wait behind a long prompt on Flash-Next?
