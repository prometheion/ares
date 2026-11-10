// The generic weights piece of the f16 prefill kernels (see `mm_f16_core.glsl`) for the
// file's layout: two invocations per weight row, each decoding one unit of 32 values of
// the step through the type's decoder (`mmd_<type>.glsl`) ahead of the products, and
// writing 64 bytes of the A tile.

uint64_t wrow;     // the weight row of this invocation (the last row for rows past the end)
uint part;         // which of the row's two units of a step
uint dec[16];      // the unit to come, as f16 pairs

void weights_setup() {
    const uint r = min(wt * MT + (tid >> 1), pc.n_out - 1);
    part = tid & 1u;
    md_init();
    wrow = pc.w + uint64_t(r) * (uint64_t(pc.n_in / BLOCK_VALUES) * BLOCK_BYTES);
}

void weights_load(uint s) {
    md_unit(wrow, 2 * s + part, dec);
}

void weights_store(uint s, uint st) {
    const uint base = ((tid >> 1) * ROW16 + 4 * part) * 4;
    [[unroll]] for (uint q = 0; q < 16; q++) {
        sa[st][base + q] = dec[q];
    }
}
