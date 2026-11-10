// Batch reference for the tokenizer: runs llama.cpp's own vocabulary code on many records
// with one load of the vocabulary, which llama-tokenize (one text per start) cannot do.
//
// Build against a llama.cpp build tree (the paths are those of the checkout and its build):
//
//   g++ -O2 -std=c++17 reftok.cpp -o reftok -I$L/include -I$L/ggml/include \
//       -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin
//
// and run it with CUDA_VISIBLE_DEVICES set to the empty string; only the vocabulary is
// read. It is driven by examples/fuzz.rs of this crate.
//
// Input records on stdin:  u8 mode | u8 flags | u32 len | bytes, little endian
//   mode 0: tokenize (flags bit0 add_special, bit1 parse_special) -> u32 n | i32 ids
//   mode 1: detokenize special=true  (bytes = u32 ids little endian) -> u32 len | bytes
//   mode 2: detokenize special=false
//   mode 3: speed test: tokenize the bytes `flags` times, no output besides u32 total tokens
//   mode 4: speed test: detokenize the ids (special=true) `flags` times, one llama_token_to_piece per token
#include "llama.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

static bool read_exact(void * p, size_t n) { return fread(p, 1, n, stdin) == n; }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: reftok model.gguf < in > out\n"); return 2; }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    freopen(nullptr, "rb", stdin);
    freopen(nullptr, "wb", stdout);
    std::vector<char> buf;
    std::vector<llama_token> toks;
    for (;;) {
        uint8_t hdr[2]; uint32_t len;
        if (!read_exact(hdr, 2)) break;
        if (!read_exact(&len, 4)) return 3;
        buf.resize(len);
        if (len && !read_exact(buf.data(), len)) return 3;
        if (hdr[0] == 0) {
            bool add_special = hdr[1] & 1, parse_special = hdr[1] & 2;
            toks.resize(len + 16);
            int n = llama_tokenize(vocab, buf.data(), (int32_t) len, toks.data(), (int32_t) toks.size(), add_special, parse_special);
            if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, buf.data(), (int32_t) len, toks.data(), (int32_t) toks.size(), add_special, parse_special); }
            uint32_t un = n; fwrite(&un, 4, 1, stdout); fwrite(toks.data(), 4, n, stdout);
        } else if (hdr[0] == 1 || hdr[0] == 2) {
            bool special = hdr[0] == 1;
            std::string out; char piece[1024];
            const int32_t * ids = (const int32_t *) buf.data();
            for (uint32_t i = 0; i < len / 4; i++) {
                int n = llama_token_to_piece(vocab, ids[i], piece, sizeof piece, 0, special);
                if (n < 0) { std::vector<char> big(-n); n = llama_token_to_piece(vocab, ids[i], big.data(), big.size(), 0, special); out.append(big.data(), n); }
                else out.append(piece, n);
            }
            uint32_t ol = out.size(); fwrite(&ol, 4, 1, stdout); fwrite(out.data(), 1, ol, stdout);
        } else if (hdr[0] == 4) {
            int reps = hdr[1];
            const int32_t * ids = (const int32_t *) buf.data();
            uint32_t n = len / 4;
            std::string out; out.reserve(n * 8);
            char piece[1024];
            auto t0 = std::chrono::steady_clock::now();
            uint64_t bytes = 0;
            for (int r = 0; r < reps; r++) {
                out.clear();
                for (uint32_t i = 0; i < n; i++) {
                    int k = llama_token_to_piece(vocab, ids[i], piece, sizeof piece, 0, true);
                    out.append(piece, k);
                }
                bytes += out.size();
            }
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "reftok: detokenize %u tokens x %d: %.3f s, %.2f MB/s, %.2f M tokens/s\n", n, reps, dt, (double) bytes / dt / 1e6, (double) n * reps / dt / 1e6);
            uint32_t ol = out.size(); fwrite(&ol, 4, 1, stdout);
        } else {
            int reps = hdr[1]; uint64_t total = 0;
            toks.resize(len + 16);
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; r++) {
                int n = llama_tokenize(vocab, buf.data(), (int32_t) len, toks.data(), (int32_t) toks.size(), false, true);
                if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, buf.data(), (int32_t) len, toks.data(), (int32_t) toks.size(), false, true); }
                total += n;
            }
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "reftok: %d reps of %u bytes: %.3f s, %.2f MB/s, %llu tokens per rep\n", reps, len, dt, (double) len * reps / dt / 1e6, (unsigned long long)(total / reps));
            uint32_t un = total / reps; fwrite(&un, 4, 1, stdout);
        }
        fflush(stdout);
    }
    llama_model_free(model);
    return 0;
}
