// Captures what llama.cpp computes for a qwen35 model on the CPU, so that the Rust
// reference (crates/model) can be compared with it block by block.
//
// Built against the headers and shared libraries of llama.cpp-native, which is only
// read. See README.md for the build and the run.
//
//   capture -m model.gguf -ngl 0 -dev none -t 6 -c 4096 -b 512 -ub 512 [llama args]
//
// Environment:
//   CAP_TOKENS  file with the prompt token ids, separated by white space (required)
//   CAP_OUT     output directory, created by the caller (required)
//   CAP_GEN     number of greedy tokens to generate one at a time (default 32)
//   CAP_NODES   comma separated names of extra graph nodes to keep, for example
//               "attn_norm-0,linear_attn_out-0" (default: none)
//   CAP_LIST    "1" prints the operator of every node of the first decode step
//
// What is written (all little endian, no headers; shapes in the README):
//   prompt.tokens, generated.tokens   token ids, one line, blank separated
//   prefill.<node>.f32                [n_tokens][ne0]            the node over the whole prompt
//   steps.<node>.f32                  [n_steps][ne0]             the node of every generated step
//   prefill.logits_last.f32           [n_vocab]                  logits of the last prompt token
//   prefill.argmax.i32                [n_tokens]                 greedy pick after every prompt token
//   steps.logits.f32                  [n_steps + 1][n_vocab]     row 0 is prefill.logits_last
//
// The nodes that are always kept are the output of every block ("l_out-N") and the
// output of the final norm ("result_norm"). All prompt tokens are flagged as outputs
// so that the last block is not narrowed to the rows that need logits (see
// crop_before_nextn in llama-graph.h) and its output exists for every position.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct Cap {
    std::string out;
    std::string phase = "prefill";
    bool list = false;
    int64_t n_layer = 0;
    std::set<std::string> extra;
    // name -> bytes of ne0 floats per row, appended in call order
    std::map<std::string, std::vector<float>> data;
    std::map<std::string, int64_t> row_len;
};

static bool wanted(const Cap * cap, const char * name) {
    if (!strncmp(name, "l_out-", 6)) return true;
    if (!strcmp(name, "result_norm")) return true;
    return cap->extra.count(name) > 0;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    Cap * cap = (Cap *) user_data;
    const bool keep = wanted(cap, t->name);
    const bool listing = cap->list && cap->phase == "steps";
    if (ask) return keep || listing;
    if (listing) {
        printf("node %-28s op %-18s type %-5s ne [%lld, %lld, %lld, %lld]\n", t->name, ggml_op_name(t->op),
               ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
               (long long) t->ne[3]);
    }
    if (!keep) return true;
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "node %s has type %s, expected f32\n", t->name, ggml_type_name(t->type));
        exit(2);
    }
    const std::string key = cap->phase + "." + t->name;
    auto & buf = cap->data[key];
    const int64_t ne0 = t->ne[0];
    cap->row_len[key] = ne0;
    std::vector<float> row(ne0);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                ggml_backend_tensor_get(t, row.data(), i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3],
                                        ne0 * sizeof(float));
                buf.insert(buf.end(), row.begin(), row.end());
            }
    return true;
}

static void write_file(const std::string & path, const void * p, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { perror(path.c_str()); exit(2); }
    fwrite(p, 1, n, f);
    fclose(f);
}

static std::vector<llama_token> read_tokens(const char * path) {
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    std::vector<llama_token> ids;
    long v;
    while (in >> v) ids.push_back((llama_token) v);
    return ids;
}

static int argmax(const float * x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return best;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * out = getenv("CAP_OUT");
    const char * tokens_path = getenv("CAP_TOKENS");
    if (!out || !tokens_path) { fprintf(stderr, "set CAP_OUT and CAP_TOKENS\n"); return 1; }
    Cap cap;
    cap.out = out;
    cap.list = getenv("CAP_LIST") && !strcmp(getenv("CAP_LIST"), "1");
    const int n_gen = getenv("CAP_GEN") ? atoi(getenv("CAP_GEN")) : 32;
    if (const char * l = getenv("CAP_NODES")) {
        std::stringstream ss(l);
        std::string item;
        while (std::getline(ss, item, ',')) if (!item.empty()) cap.extra.insert(item);
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
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    cap.n_layer = llama_model_n_layer(model);

    std::vector<llama_token> prompt = read_tokens(tokens_path);
    const int n = (int) prompt.size();
    printf("prompt: %d tokens, vocab %d, layers %lld, generating %d\n", n, n_vocab, (long long) cap.n_layer, n_gen);

    // prefill: the whole prompt in one batch, every row an output
    {
        common_batch batch(ctx);
        for (int i = 0; i < n; ++i) batch.add(prompt[i], i, 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            fprintf(stderr, "prefill failed\n");
            return 1;
        }
    }
    std::vector<int32_t> pick(n);
    for (int i = 0; i < n; ++i) pick[i] = argmax(llama_get_logits_ith(ctx, i), n_vocab);
    std::vector<float> logits_steps;
    {
        const float * last = llama_get_logits_ith(ctx, n - 1);
        logits_steps.insert(logits_steps.end(), last, last + n_vocab);
        write_file(cap.out + "/prefill.logits_last.f32", last, (size_t) n_vocab * sizeof(float));
        write_file(cap.out + "/prefill.argmax.i32", pick.data(), pick.size() * sizeof(int32_t));
    }

    // generation: one token per call, greedy
    cap.phase = "steps";
    std::vector<llama_token> generated;
    for (int s = 0; s < n_gen; ++s) {
        const llama_token next = argmax(logits_steps.data() + (size_t) s * n_vocab, n_vocab);
        generated.push_back(next);
        common_batch batch(ctx);
        batch.add(next, n + s, 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            fprintf(stderr, "decode failed at step %d\n", s);
            return 1;
        }
        const float * l = llama_get_logits_ith(ctx, 0);
        logits_steps.insert(logits_steps.end(), l, l + n_vocab);
        cap.list = false;
    }

    for (auto & [key, buf] : cap.data) {
        const std::string path = cap.out + "/" + key + ".f32";
        write_file(path, buf.data(), buf.size() * sizeof(float));
        printf("%-34s rows of %5lld, %8zu rows\n", key.c_str(), (long long) cap.row_len[key],
               buf.size() / (size_t) cap.row_len[key]);
    }
    write_file(cap.out + "/steps.logits.f32", logits_steps.data(), logits_steps.size() * sizeof(float));
    {
        std::string text;
        for (auto t : prompt) text += std::to_string(t) + " ";
        text += "\n";
        write_file(cap.out + "/prompt.tokens", text.data(), text.size());
        text.clear();
        for (auto t : generated) text += std::to_string(t) + " ";
        text += "\n";
        write_file(cap.out + "/generated.tokens", text.data(), text.size());
    }
    printf("generated:");
    for (auto t : generated) printf(" %d", t);
    printf("\n");
    llama_backend_free();
    return 0;
}
