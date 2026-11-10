#!/usr/bin/env python3
"""Write chat_<hash>.jsonl: what llama.cpp renders for chat_requests.jsonl with one template.

    python3 -I tests/golden/generate_chat.py --server http://127.0.0.1:PORT --template TEMPLATE.jinja

The server is llama-server with `--jinja --no-prefill-assistant` (so that a conversation
that ends with the assistant is rendered as it is and not continued), CPU only is fine,
on a port that nothing else uses, and with the template under test: the model's own, or
`--chat-template-file TEMPLATE.jinja` on any model. `--template` is that same file; its
hash names the result, and the tests find the fixture of a model by the hash of the
model's `tokenizer.chat_template`. The hash is FNV-1a (64 bits) of the template's bytes.

Each line of the result is {"request": ..., "prompt": "..."} or {"request": ..., "error": true}.
"""

import argparse
import json
import os
import urllib.error
import urllib.request


def fnv1a(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True)
    parser.add_argument("--template", required=True)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    name = f"chat_{fnv1a(open(args.template, 'rb').read()):016x}.jsonl"
    out_lines = []
    for line in open(os.path.join(here, "chat_requests.jsonl"), encoding="utf8"):
        request = json.loads(line)
        http = urllib.request.Request(
            args.server.rstrip("/") + "/apply-template", data=line.encode("utf8"), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(http, timeout=60) as response:
                result = {"request": request, "prompt": json.load(response)["prompt"]}
        except urllib.error.HTTPError:
            result = {"request": request, "error": True}
        out_lines.append(json.dumps(result, ensure_ascii=False))
    with open(os.path.join(here, name), "w", encoding="utf8") as out:
        out.write("\n".join(out_lines) + "\n")
    print(name, len(out_lines), "requests,", sum('"error": true' in l for l in out_lines), "refused")


if __name__ == "__main__":
    main()
