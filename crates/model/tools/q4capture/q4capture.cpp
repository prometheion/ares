// Dumps what llama.cpp computes for a qwen4exp model on the CPU, so that the Rust
// reference (crates/model, qwen4exp) can be compared with it layer by layer.
//
// Built against the headers and shared libraries of llama.cpp-native, which is only
// read. See README.md for the build and the runs.
//
//   q4capture -m model.gguf -ngl 0 -dev none -t 6 -c 1024 -b 512 -ub 512 -fa off -ctk f32 -ctv f32
//
// Environment:
//   CAP_TOKENS  file with the token ids, separated by white space (required)
//   CAP_OUT     output directory, created by the caller (required)
//   CAP_CHUNK   how many tokens go into one decode call (default: all of them at once);
//               1 feeds the sequence token by token
//   CAP_NODES   comma separated names or name prefixes of graph nodes to keep besides the
//               default set; a name is "<what>-<layer>" or "<what>" (default: none)
//   CAP_LIST    "1" prints name, operator, type and shape of every node of the first call
//
// Every kept node is written as one file "<name>.<k>.<type>": k counts the nodes of that
// name inside one decode call in graph order (the two reads of a layer are both called
// "hc_mixed-<layer>": 0 is the read in front of the recurrent or attention part, 1 the one
// in front of the experts), <type> is f32 or i32. A file holds the node's values of every
// decode call, one call after the other, each in memory order (first dimension fastest).
// "index.txt" lists for every file and call the four dimensions. The default set:
//
//   l_last-<layer>          [N, hc, rows]   the residual after the layer
//   hc_mixed-<layer>        [N, rows]       the two reads
//   linear_attn_out-<layer> [N, rows]       output of a recurrent layer
//   attn_output-<layer>     [N, rows]       output of an attention layer
//   ffn_out-<layer>         [N, rows]       output of the mixture of experts
//   ffn_moe_topk-<layer>    [K, rows] i32   routed experts (no particular order)
//   ffn_moe_weights_norm-<layer>            their weights
//   indexer_score-<layer>   [blocks, rows]  block scores (times 1/sqrt(ID); -inf = not visible)
//   indexer_top_k-<layer>   [blocks kept, rows] i32
//   result_norm             [N, rows]       the final read
// and "logits.f32" [rows][V] through the API, "tokens.txt".
//
// All rows are flagged as outputs so that the last layer is not narrowed to the rows that
// need logits (crop_before_nextn in llama-graph.h).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
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
    bool list = false;
    int call = 0;
    std::vector<std::string> extra;
    std::map<std::string, int> seen;                // per decode call: nodes of a name so far
    std::map<std::string, std::vector<char>> data;  // file name -> bytes
    std::string index;
};

static const char * DEFAULT[] = {"l_last-", "hc_mixed-", "linear_attn_out-", "attn_output-", "ffn_out-",
                                 "ffn_moe_topk-", "ffn_moe_weights_norm-", "indexer_score-",
                                 "indexer_top_k-", "result_norm"};

static bool wanted(const Cap * cap, const char * name) {
    for (const char * p : DEFAULT)
        if (!strncmp(name, p, strlen(p))) return true;
    for (const auto & p : cap->extra)
        if (!strncmp(name, p.c_str(), p.size())) return true;
    return false;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    Cap * cap = (Cap *) user_data;
    const bool keep = wanted(cap, t->name);
    const bool listing = cap->list && cap->call == 0;
    if (ask) return keep || listing;
    if (listing) {
        printf("node %-28s op %-18s type %-5s ne [%lld, %lld, %lld, %lld]\n", t->name, ggml_op_name(t->op),
               ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
               (long long) t->ne[3]);
    }
    if (!keep) return true;
    const char * ext = t->type == GGML_TYPE_F32 ? "f32" : t->type == GGML_TYPE_I32 ? "i32" : nullptr;
    if (!ext) {
        fprintf(stderr, "node %s has type %s, expected f32 or i32\n", t->name, ggml_type_name(t->type));
        exit(2);
    }
    const int k = cap->seen[t->name]++;
    const std::string key = std::string(t->name) + "." + std::to_string(k) + "." + ext;
    auto & buf = cap->data[key];
    // row by row through the backend, so that views and strides are honoured
    const int64_t ne0 = t->ne[0];
    std::vector<char> row(ne0 * 4);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                ggml_backend_tensor_get(t, row.data(), i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], ne0 * 4);
                buf.insert(buf.end(), row.begin(), row.end());
            }
    char line[256];
    snprintf(line, sizeof(line), "%s call %d ne %lld %lld %lld %lld\n", key.c_str(), cap->call, (long long) t->ne[0],
             (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    cap->index += line;
    return true;
}

static void write_file(const std::string & path, const void * p, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { perror(path.c_str()); exit(2); }
    fwrite(p, 1, n, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * out = getenv("CAP_OUT");
    const char * tokens_path = getenv("CAP_TOKENS");
    if (!out || !tokens_path) { fprintf(stderr, "set CAP_OUT and CAP_TOKENS\n"); return 1; }
    Cap cap;
    cap.out = out;
    cap.list = getenv("CAP_LIST") && !strcmp(getenv("CAP_LIST"), "1");
    if (const char * l = getenv("CAP_NODES")) {
        std::stringstream ss(l);
        std::string item;
        while (std::getline(ss, item, ',')) if (!item.empty()) cap.extra.push_back(item);
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

    std::vector<llama_token> tokens;
    {
        std::ifstream in(tokens_path);
        if (!in) { fprintf(stderr, "cannot read %s\n", tokens_path); return 1; }
        long v;
        while (in >> v) tokens.push_back((llama_token) v);
    }
    const int n = (int) tokens.size();
    const int chunk = getenv("CAP_CHUNK") ? atoi(getenv("CAP_CHUNK")) : n;
    printf("%d tokens in calls of %d, vocab %d, layers %d\n", n, chunk, n_vocab, (int) llama_model_n_layer(model));

    std::vector<float> logits;
    for (int at = 0; at < n; at += chunk) {
        const int m = std::min(chunk, n - at);
        cap.seen.clear();
        common_batch batch(ctx);
        for (int i = 0; i < m; ++i) batch.add(tokens[at + i], at + i, 0, true);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            fprintf(stderr, "decode failed at token %d\n", at);
            return 1;
        }
        for (int i = 0; i < m; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            logits.insert(logits.end(), l, l + n_vocab);
        }
        cap.call++;
    }

    for (auto & [key, buf] : cap.data) write_file(cap.out + "/" + key, buf.data(), buf.size());
    write_file(cap.out + "/logits.f32", logits.data(), logits.size() * sizeof(float));
    write_file(cap.out + "/index.txt", cap.index.data(), cap.index.size());
    {
        std::string text;
        for (auto t : tokens) text += std::to_string(t) + " ";
        text += "\n";
        write_file(cap.out + "/tokens.txt", text.data(), text.size());
    }
    printf("%zu node files, %d decode calls\n", cap.data.size(), cap.call);
    llama_backend_free();
    return 0;
}
