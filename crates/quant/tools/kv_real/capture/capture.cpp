// Captures the Q, K and V tensors that feed attention (after normalisation and
// RoPE) of every attention layer, as f32, per position and head.
//
// Built against the headers and shared libraries of a llama.cpp build tree, which
// is only read. Usage: kv-capture -m model.gguf -f prompt.txt --out DIR [llama args]
// Environment: KVCAP_LAYERS=3,7,11 limits the layers (default: all that appear).
//
// Names in the qwen35 graph (src/models/qwen35.cpp, build_layer_attn):
//   "Qcur-N": GGML_OP_ROPE  [head_dim, n_head,    n_tokens]  query, normed + roped
//   "Kcur-N": GGML_OP_ROPE  [head_dim, n_head_kv, n_tokens]  key, normed + roped
//   "Vcur-N": GGML_OP_RESHAPE [head_dim, n_head_kv, n_tokens] value
// (the same names are also given to the projections before the norm; the op
// tells them apart).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

struct Cap {
    std::string out;
    std::set<int> only;
    // per layer and kind: bytes and shape
    struct Buf {
        std::vector<float> data;
        int64_t ne0 = 0, ne1 = 0, tokens = 0;
    };
    std::map<int, Buf> q, k, v, a;
    std::vector<float> scratch;
};

static bool parse_name(const char * name, const char * prefix, int * il) {
    size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return false;
    char * end = nullptr;
    long v = strtol(name + n + 1, &end, 10);
    if (*end != '\0') return false;
    *il = (int) v;
    return true;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    Cap * cap = (Cap *) user_data;
    int il = -1;
    int kind = 0;  // 1 q, 2 k, 3 v
    if (parse_name(t->name, "Qcur", &il) && t->op == GGML_OP_ROPE) kind = 1;
    else if (parse_name(t->name, "Kcur", &il) && t->op == GGML_OP_ROPE) kind = 2;
    else if (parse_name(t->name, "Vcur", &il) && t->op == GGML_OP_RESHAPE) kind = 3;
    if (kind == 0 && parse_name(t->name, "attn_pregate", &il)) kind = 4;
    if (kind == 0) return ask ? false : true;
    if (!cap->only.empty() && !cap->only.count(il)) return ask ? false : true;
    if (ask) return true;

    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "unexpected type %d for %s\n", (int) t->type, t->name);
        exit(2);
    }
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2];
    Cap::Buf & b = (kind == 1 ? cap->q : kind == 2 ? cap->k : kind == 3 ? cap->v : cap->a)[il];
    if (b.tokens == 0) { b.ne0 = ne0; b.ne1 = ne1; }
    // row by row through the strides: [token][head][dim] in the output
    size_t base = b.data.size();
    b.data.resize(base + (size_t) ne0 * ne1 * ne2);
    std::vector<float> row(ne0);
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            ggml_backend_tensor_get(t, row.data(), i1 * t->nb[1] + i2 * t->nb[2], ne0 * sizeof(float));
            memcpy(&b.data[base + ((size_t) i2 * ne1 + i1) * ne0], row.data(), ne0 * sizeof(float));
        }
    }
    b.tokens += ne2;
    return true;
}

static void dump(const Cap & cap, const char * kind, const std::map<int, Cap::Buf> & m) {
    for (const auto & [il, b] : m) {
        char path[1024];
        snprintf(path, sizeof path, "%s/L%02d_%s.f32", cap.out.c_str(), il, kind);
        FILE * f = fopen(path, "wb");
        if (!f) { perror(path); exit(2); }
        fwrite(b.data.data(), sizeof(float), b.data.size(), f);
        fclose(f);
        printf("layer %2d %s: heads %lld, head_dim %lld, tokens %lld -> %s\n", il, kind, (long long) b.ne1,
               (long long) b.ne0, (long long) b.tokens, path);
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    Cap cap;
    const char * out = getenv("KVCAP_OUT");
    if (!out) { fprintf(stderr, "set KVCAP_OUT\n"); return 1; }
    cap.out = out;
    if (const char * l = getenv("KVCAP_LAYERS")) {
        std::string s = l;
        size_t p = 0;
        while (p < s.size()) {
            size_t e = s.find(',', p);
            if (e == std::string::npos) e = s.size();
            cap.only.insert(atoi(s.substr(p, e - p).c_str()));
            p = e + 1;
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    llama_backend_init();
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cap;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    printf("tokens: %zu (add_bos %d)\n", tokens.size(), (int) add_bos);
    printf("model: layers %d, heads %d, heads_kv %d, n_embd %d\n", llama_model_n_layer(model),
           llama_model_n_head(model), llama_model_n_head_kv(model), llama_model_n_embd(model));

    {
        std::string path = cap.out + "/tokens.txt";
        FILE * f = fopen(path.c_str(), "w");
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string piece = common_token_to_piece(ctx, tokens[i], true);
            for (auto & c : piece) if (c == '\n') c = '\x01';
            fprintf(f, "%zu\t%d\t%s\n", i, tokens[i], piece.c_str());
        }
        fclose(f);
    }

    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    dump(cap, "q", cap.q);
    dump(cap, "k", cap.k);
    dump(cap, "v", cap.v);
    dump(cap, "attn", cap.a);
    llama_backend_free();
    return 0;
}
