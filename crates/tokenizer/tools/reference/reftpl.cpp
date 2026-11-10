// Batch reference for chat templates: renders a template given as text with llama.cpp's own
// Jinja code (common/jinja), for any template and without a model. It stops at the template:
// the messages are given to it as they are, without the conventions of the server.
//
// Build against a llama.cpp build tree ($L is the checkout; the library must be built from it):
//
//   g++ -O1 -std=c++17 reftpl.cpp -o reftpl -I$L/common -I$L/include -I$L/ggml/include \
//       -I$L/vendor -I$L/src -L$L/build/bin -lllama-common -lllama -lggml -lggml-base \
//       -Wl,-rpath,$L/build/bin
//
// It is driven by examples/probes.rs of this crate.
//
// stdin: lines of JSON {"template": str, "messages": [...], "tools": [...]?, "kwargs": {...}?,
//                       "enable_thinking": bool?, "add_generation_prompt": bool?}
// stdout: one JSON line per request: {"ok": "<prompt>"} or {"error": "..."}
#include "chat.h"
#include "chat-auto-parser.h"
#include "json.h"
#include <iostream>
#include <string>

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        common_json out = common_json::object();
        try {
            common_json req = common_json::parse(line);
            const std::string src = req.at("template").get<std::string>();
            common_chat_template tmpl(src, "", "");
            autoparser::generation_params gp;
            gp.messages = req.at("messages");
            if (req.contains("tools")) gp.tools = req.at("tools");
            if (req.contains("kwargs")) gp.extra_context = req.at("kwargs");
            gp.enable_thinking = req.contains("enable_thinking") ? req.at("enable_thinking").get<bool>() : true;
            gp.add_generation_prompt = req.contains("add_generation_prompt") ? req.at("add_generation_prompt").get<bool>() : true;
            common_chat_prompt p = common_chat_template_direct_apply(tmpl, gp);
            out["ok"] = p.str();
        } catch (const std::exception & e) {
            out["error"] = std::string(e.what());
        }
        std::string text;
        try { text = out.dump(); } catch (const std::exception & e) { text = "{\"error\": \"output is not valid UTF-8 (" + std::string(e.what()).substr(0, 40) + ")\"}"; }
        std::cout << text << "\n";
        std::cout.flush();
    }
    return 0;
}
