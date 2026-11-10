// Runs the original CPU quantise / dequantise routines of beellama.cpp
// (src/llama-kvarn.cpp, linked as compiled) on tiles read from a file, so that
// a re-implementation can be compared against them.
//
//   kvarn_ref tiles  IN OUT_DEQ OUT_REC
//     IN: u32 n; then n x { u32 kind (0 K, 1 V), u32 bits, 16384 x f32 tile }
//     OUT_DEQ: n x 16384 f32 (the dequantised tile)
//     OUT_REC: n x { payload, 3 x 128 f16 } the record part of that kind:
//              K: k payload, k_s_col (row scale), k_zp, k_s_row (column scale)
//              V: v payload, v_s_col (column scale), v_s_row (row scale), v_zp
//   kvarn_ref wht IN OUT
//     IN: u32 n; then n x 256 f32 rows; OUT: the same rows through the
//     reference head transform (the function of tests/test-kvarn.cpp)
//
// The function apply_reference_kvarn_wht_head is copied from
// tests/test-kvarn.cpp (it is static there).

#include "llama-kvarn.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void apply_reference_kvarn_wht_head(float * values, int head_width) {
    if (head_width == 64) {
        llama_kvarn_hadamard_64(values);
        return;
    }
    const int slices = head_width / 128;
    for (int slice = 0; slice < slices; ++slice) {
        llama_kvarn_hadamard_128(values + slice * 128);
    }
    if (slices == 1) {
        return;
    }
    const float scale = slices == 2 ? 0.7071067811865475f : 0.5f;
    for (int d = 0; d < 128; ++d) {
        float x[4] = {};
        for (int slice = 0; slice < slices; ++slice) {
            x[slice] = values[slice * 128 + d];
        }
        for (int stride = 1; stride < slices; stride <<= 1) {
            for (int base = 0; base < slices; base += 2 * stride) {
                for (int i = 0; i < stride; ++i) {
                    const float a = x[base + i];
                    const float b = x[base + stride + i];
                    x[base + i] = a + b;
                    x[base + stride + i] = a - b;
                }
            }
        }
        for (int slice = 0; slice < slices; ++slice) {
            values[slice * 128 + d] = x[slice] * scale;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: kvarn_ref tiles IN OUT_DEQ OUT_REC | wht IN OUT\n");
        return 1;
    }
    const std::string mode = argv[1];
    FILE * in = fopen(argv[2], "rb");
    if (!in) { perror(argv[2]); return 1; }
    uint32_t n = 0;
    if (fread(&n, 4, 1, in) != 1) return 1;

    if (mode == "wht") {
        FILE * out = fopen(argv[3], "wb");
        std::vector<float> row(256);
        for (uint32_t i = 0; i < n; ++i) {
            if (fread(row.data(), 4, 256, in) != 256) return 1;
            apply_reference_kvarn_wht_head(row.data(), 256);
            fwrite(row.data(), 4, 256, out);
        }
        fclose(out);
        return 0;
    }

    FILE * out_deq = fopen(argv[3], "wb");
    FILE * out_rec = fopen(argv[4], "wb");
    std::vector<float> tile(128 * 128), deq(128 * 128);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t kind, bits;
        if (fread(&kind, 4, 1, in) != 1 || fread(&bits, 4, 1, in) != 1) return 1;
        if (fread(tile.data(), 4, tile.size(), in) != tile.size()) return 1;
        const auto layout = llama_kvarn_make_layout(128, 128, (int) bits, (int) bits);
        std::vector<uint8_t> record(layout.tile_bytes, 0);
        size_t from, to;
        if (kind == 0) {
            llama_kvarn_quantize_k_tile(tile.data(), 16, (int) bits, layout, record.data());
            llama_kvarn_dequantize_k_tile(record.data(), (int) bits, layout, deq.data());
            from = layout.k_payload_off;
            to = layout.k_s_row_off + 128 * 2;
        } else {
            llama_kvarn_quantize_v_tile(tile.data(), 16, (int) bits, layout, record.data());
            llama_kvarn_dequantize_v_tile(record.data(), (int) bits, layout, deq.data());
            from = layout.v_payload_off;
            to = layout.v_zp_off + 128 * 2;
        }
        fwrite(deq.data(), 4, deq.size(), out_deq);
        fwrite(record.data() + from, 1, to - from, out_rec);
    }
    fclose(out_deq);
    fclose(out_rec);
    return 0;
}
