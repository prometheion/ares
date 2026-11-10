# Notes on Run My AI (TokenWing engine)

From https://runmyai.org/how-it-works as read on 2026-11-09. The engine is
not released yet (the page says the source opens on 2026-11-17 and the flag
reference comes with it), so this is what the page states, nothing checked.
The page gives no tokens per second, no SSD throughput and no latencies.

- **Reads.** Routed experts are read with direct I/O, bypassing the OS file
  cache, with a logged fall-back to cached reads where direct I/O is not
  available. 8 I/O threads by default. A demand read is a single request; a
  prefetch is split into chunks of 512 KiB. Demand reads go before
  prefetches; a prediction that is not selected is cancelled, one that is
  selected is promoted.
- **File layout.** A single file of its own with an index, in which every
  routed expert starts on a 4 KiB boundary (GGUF aligns to 32 bytes only).
- **Placement.** All routed experts on SSD. RAM holds expert cache slots per
  layer, sized at start from free memory. VRAM holds whole MoE layers or a
  cache of the hottest experts, re-decided every 8 tokens (up to 64 experts,
  a swap needs 1.25 times the heat plus 4). Attention, shared experts,
  routers, dense layers, embeddings and the output head are loaded at start.
  The GPU computes the experts in VRAM, the CPU those in RAM, and the results
  are summed.
- **Prediction.** The routers of layers L+1 and L+2 are applied to layer L's
  hidden state plus its shared-expert output; an optional trained correction
  adjusts the prediction. Recall on a model with 512 experts and top 8: 75.7 %
  one layer ahead, 67.2 % two, 61.1 % three; the default looks two ahead.
  Demand reads per token fall from 0.87 to 0.44 with the prediction.
- **Eviction.** A use count that halves every 64 tokens; ties by least recent
  use, then by lower id.
- **Prefill.** Layer by layer in waves of experts, about one read per distinct
  expert. Without that, a 4k prompt in batches of 16 tokens read 954 GiB.
- **Accuracy.** KL divergence against llama.cpp 0.010 to 0.016.
