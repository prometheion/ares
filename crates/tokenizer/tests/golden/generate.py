#!/usr/bin/env python3
"""Write qwen35_ids.txt: the token ids that llama.cpp gives for the files in ../corpus.

    python3 -I tests/golden/generate.py --llama-tokenize PATH/llama-tokenize --model MODEL.gguf

Every Qwen3.5 and Qwen3.8 model has the same vocabulary, so any of their files will
do; the tests of this crate compare with the result for whichever model they are
given. The file has one line per corpus file and mode, "name TAB mode TAB ids", where
mode is `special` (the text of control tokens is turned into the tokens, which is what
chat prompts need) or `plain`, and the ids are separated by spaces.

llama-tokenize must be built from the llama.cpp revision named in
`src/unicode_table.rs` or later, and must work: an old binary that does not match its
libraries crashes before it reads its arguments. Run with CUDA_VISIBLE_DEVICES set to
the empty string if the build has a GPU backend, since only the vocabulary is read.
"""

import argparse
import os
import subprocess
import sys


def tokenize(binary, model, path, parse_special):
    command = [binary, "-m", model, "--ids", "--no-escape", "--log-disable", "-f", path]
    if not parse_special:
        command.append("--no-parse-special")
    out = subprocess.run(command, check=True, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).stdout.strip()
    if not (out.startswith("[") and out.endswith("]")):
        sys.exit(f"{path}: unexpected output {out[:80]!r}")
    inner = out[1:-1].strip()
    return [int(x) for x in inner.split(",")] if inner else []


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llama-tokenize", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    corpus = os.path.join(here, "..", "corpus")
    lines = []
    for name in sorted(os.listdir(corpus)):
        path = os.path.join(corpus, name)
        for mode, parse_special in (("special", True), ("plain", False)):
            ids = tokenize(args.llama_tokenize, args.model, path, parse_special)
            lines.append(f"{name}\t{mode}\t{' '.join(map(str, ids))}")
            print(f"{name} {mode}: {len(ids)} tokens", file=sys.stderr)
    with open(os.path.join(here, "qwen35_ids.txt"), "w") as out:
        out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
