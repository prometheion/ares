#!/usr/bin/env python3
"""Fetches the corpus of the ready draft vocabularies and makes them again.

    python3 -I tools/draft-vocab.py fetch DIR
    python3 -I tools/draft-vocab.py make DIR MODEL.gguf OUT_DIR [--date YYYY-MM-DD]

`fetch` downloads every source below into DIR, which must be empty or not yet
there: Wikipedia articles through the Hugging Face datasets server, and the
other sources as git checkouts at a pinned commit, of which the text files are
copied into DIR/<part>/<source>/. The text is data: nothing in it is run.
Downloaded files are untrusted, so run this with `python3 -I` from the
repository, never from inside DIR. A fetch that was cut off is taken up again
by the same command. About 100 MB of text, and 350 MB of git objects in
DIR/git/ (removable afterwards); a quarter of an hour, most of it waiting for
the datasets server, which allows a few dozen requests a minute.

`make` runs `crates/tokenizer/examples/draft_vocab.rs` for each ready file of
MIXES, with the weights, licences and sources of the parts in its header. The
guide is docs/user/draft-vocab.md.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The Wikipedia dump the parts of prose are taken from: the dataset
# wikimedia/wikipedia on Hugging Face at this revision, read through the
# datasets server, BATCHES batches of 100 articles at offsets spread evenly
# over the dump.
WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKIPEDIA = {
    # part: (config, batches)
    "wiki-en": ("20231101.en", 120),
}
WIKIPEDIA_LICENCE = "CC BY-SA 3.0 and GFDL"

# The sources from git: (part, name, repository, commit, licence, directories
# of the sparse checkout, what to take, the most bytes to take). What to take is
# a list of file name suffixes (".py") and whole names ("Dockerfile"). Files
# above MAX_FILE are left out (generated or minified files). When a source has
# more than its most bytes, the files are taken in the order of the SHA-1 of
# their paths, a fixed order that spreads over the whole tree.
MAX_FILE = 300_000
GIT = [
    ("code", "cpython", "https://github.com/python/cpython",
     "5e58eaad7c58df58dfa8ea549c87996e01dd1077", "PSF-2.0", ["Lib"], [".py"], 12_000_000),
    ("code", "tokio", "https://github.com/tokio-rs/tokio",
     "b53b2001ed2ccfd9b87e32a0c6da1a1eebdc80b8", "MIT", ["tokio", "tokio-util", "tokio-stream"],
     [".rs"], 6_000_000),
    ("code", "ripgrep", "https://github.com/BurntSushi/ripgrep",
     "669cb4c755db20b26c3af3d923c042ae224c7496", "MIT or Unlicense", ["crates"], [".rs"], 3_000_000),
    ("code", "prometheus", "https://github.com/prometheus/prometheus",
     "e2e28adeee2c5883d000fc82aca6008cfbf7e96d", "Apache-2.0",
     ["promql", "tsdb", "scrape", "discovery", "storage", "web/api"], [".go"], 10_000_000),
    ("code", "vscode", "https://github.com/microsoft/vscode",
     "661626d0e72f97806550790a98fb87116c9db825", "MIT", ["src/vs/base", "src/vs/platform"],
     [".ts", ".css", ".html"], 10_000_000),
    ("code", "curl", "https://github.com/curl/curl",
     "3163ec065643ba20709f190111f42c3e1fe95273", "curl (MIT-like)", ["lib", "src"], [".c", ".h"],
     6_000_000),
    ("docs-en", "kubernetes-en", "https://github.com/kubernetes/website",
     "61712ddb619690242534e6e86bb5646c88f16c2e", "CC BY 4.0",
     ["content/en/docs/concepts", "content/en/docs/tasks", "content/en/docs/tutorials",
      "content/en/docs/setup"], [".md"], 12_000_000),
    ("docs-en", "rust-book", "https://github.com/rust-lang/book",
     "c976903459e37407cfa6964530ea3469957b3549", "MIT or Apache-2.0", ["src"], [".md"], 3_000_000),
    ("docs-en", "art-of-command-line-en", "https://github.com/jlevy/the-art-of-command-line",
     "6b50745d2e788add2e8f1ed29010e72659a9a074", "CC BY-SA 4.0", ["."], ["README.md"], 1_000_000),
    ("docs-en", "debian-handbook-en", "https://salsa.debian.org/hertzog/debian-handbook.git",
     "e270a04f621a681c6c1e16b9ee965967186e02ef", "CC BY-SA 3.0 or GPL-2.0-or-later", ["de-DE"],
     [".po"], 6_000_000),
    ("shell", "tldr-en", "https://github.com/tldr-pages/tldr",
     "f22746ca8c573a2f9acee35e51794131ce0267a6", "CC BY 4.0", ["pages"], [".md"], 8_000_000),
    ("shell", "awesome-compose", "https://github.com/docker/awesome-compose",
     "503c0dcef31b07fb77b2d4c5556bfafdf86f4f1a", "CC0-1.0", ["."],
     [".yaml", ".yml", ".conf", ".sh", ".env", ".ini", ".toml", "Dockerfile", ".dockerfile"], 2_000_000),
    ("shell", "kubernetes-examples", "https://github.com/kubernetes/examples",
     "22934a040acbb0c0aa103023ccbd871507bf17f6", "Apache-2.0", ["."],
     [".yaml", ".yml", ".sh", ".conf", "Dockerfile"], 3_000_000),
    ("shell", "ohmyzsh", "https://github.com/ohmyzsh/ohmyzsh",
     "abfce729ab56543b976b08ef34a54b3dc57601a4", "MIT", ["plugins", "lib", "tools"],
     [".zsh", ".sh", ".plugin.zsh"], 3_000_000),
    ("shell", "nvm", "https://github.com/nvm-sh/nvm",
     "4e8758a801eea9479b32a904f31093e410ddaba4", "MIT", ["."], [".sh"], 1_000_000),
    ("shell", "lnav-logs", "https://github.com/tstack/lnav",
     "e43baf0c202cb328c75a16947097b011a5279a11", "BSD-2-Clause", ["test"], ["logfile_"], 2_000_000),
]
# lnav's test logs are taken by the start of their names, and at most this much
# of each, so that one large log does not stand for all.
LOG_HEAD = 40_000

# Sources whose files are turned into plain text: the Debian Administrator's
# Handbook is DocBook in gettext catalogues, and the German catalogues hold the
# English text in `msgid`.
TEXT_OF = {"debian-handbook-en": "msgid"}

# What each part is, for the headers.
PARTS = {
    "wiki-en": "English Wikipedia (wikimedia/wikipedia 20231101.en)",
    "code": "Python, Rust, Go, TypeScript and C",
    "docs-en": "English technical documents in Markdown",
    "shell": "command pages, compose and Kubernetes files, shell scripts, logs",
}

# The ready files: file name, title, coverage, {part: weight}. They are English
# only (the owner, 2026-11-27); docs/user/draft-vocab.md shows how to add a
# language.
MIXES = [
    ("qwen35-en.txt", "English", 0.995, {"wiki-en": 0.6, "docs-en": 0.4}),
    ("qwen35-code.txt", "Code, with its English documents", 0.995,
     {"code": 0.75, "docs-en": 0.25}),
    ("qwen35-general.txt", "English, code and shell", 0.995,
     {"wiki-en": 0.3, "docs-en": 0.2, "code": 0.3, "shell": 0.2}),
]


def get_json(url: str, tries: int = 12):
    """The JSON at `url`. The datasets server allows a few dozen requests a minute; past that it
    answers 429, and the request is made again after a wait."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read())
        except (OSError, ValueError) as e:
            if attempt + 1 == tries:
                raise
            wait = min(10 * 2 ** attempt, 300)
            print(f"  {e}; again in {wait} s", flush=True)
            time.sleep(wait)


def fetch_wikipedia(root: Path, part: str, config: str, batches: int, manifest: dict):
    api = "https://datasets-server.huggingface.co"
    query = urllib.parse.urlencode({"dataset": WIKIPEDIA_DATASET, "config": config})
    size = get_json(f"{api}/size?{query}")["size"]["config"]["num_rows"]
    info = get_json(f"https://huggingface.co/api/datasets/{WIKIPEDIA_DATASET}")
    if info.get("sha") != WIKIPEDIA_REVISION:
        sys.exit(f"{WIKIPEDIA_DATASET} is at revision {info.get('sha')}, not {WIKIPEDIA_REVISION}: "
                 "the datasets server reads the newest revision, so the articles would differ")
    out = root / part / "wikipedia"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for batch in range(batches):
        file = out / f"batch-{batch:04d}.txt"
        if not file.exists():
            offset = batch * (size // batches)
            rows = get_json(f"{api}/rows?{query}&split=train&offset={offset}&length=100")["rows"]
            texts = [r["row"]["text"].strip() for r in rows if not r.get("truncated_cells")]
            text = "\n\n".join(t for t in texts if t) + "\n"
            # Written whole or not at all, so that a fetch cut off goes on where it stopped.
            file.with_suffix(".part").write_text(text, encoding="utf-8")
            file.with_suffix(".part").rename(file)
            time.sleep(2)
        total += file.stat().st_size
        if batch % 20 == 0:
            print(f"  {part}: batch {batch + 1} of {batches}, {total / 1e6:.1f} MB", flush=True)
    manifest[part].append({"source": f"{WIKIPEDIA_DATASET} {config} at {WIKIPEDIA_REVISION[:12]}, "
                                     f"{batches} batches of 100 articles",
                           "licence": WIKIPEDIA_LICENCE, "bytes": total})


def git(*args, cwd=None):
    subprocess.run(["git", "-c", "advice.detachedHead=false", *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL)


def checkout(root: Path, url: str, commit: str, dirs: list[str]) -> Path:
    """A sparse checkout of `dirs` at `commit`, shared by the sources of one repository and
    commit."""
    name = url.rstrip("/").split("/")[-1] + "-" + commit[:12]
    tree = root / "git" / name
    if not tree.exists():
        tree.mkdir(parents=True)
        git("init", "-q", cwd=tree)
        git("remote", "add", "origin", url, cwd=tree)
        git("sparse-checkout", "set", "--no-cone", "/*.md", cwd=tree)
    paths = (tree / ".git" / "info" / "sparse-checkout").read_text().split()
    paths += [("/" + d.strip("/") + "/") if d != "." else "/*" for d in dirs]
    git("sparse-checkout", "set", "--no-cone", *sorted(set(paths)), cwd=tree)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit, cwd=tree)
    git("checkout", "-q", commit, cwd=tree)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, check=True, capture_output=True,
                          text=True).stdout.strip()
    if head != commit:
        sys.exit(f"{url}: checked out {head}, not {commit}")
    return tree


def po_text(data: bytes, field: str) -> bytes:
    """The strings of one field of a gettext catalogue, without DocBook's tags, a paragraph
    each."""
    paragraphs, current, inside = [], None, False

    def end():
        if current:
            text = re.sub(r"<[^>]*>", "", "".join(current))
            for entity, char in (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&amp;", "&")):
                text = text.replace(entity, char)
            if text.strip():
                paragraphs.append(text.strip())

    for line in data.decode("utf-8").splitlines():
        if line.startswith(("msgid ", "msgstr ", "msgctxt ")):
            if inside:
                end()
            inside = line.startswith(field + " ")
            current = [] if inside else None
            line = line.split(" ", 1)[1]
        if inside and line.startswith('"') and line.endswith('"'):
            current.append(line[1:-1].encode().decode("unicode_escape").encode("latin-1").decode())
        elif not line.startswith('"'):
            if inside:
                end()
            inside, current = False, None
    if inside:
        end()
    # The first msgstr is the catalogue's header.
    if field == "msgstr" and paragraphs and paragraphs[0].startswith("Project-Id-Version"):
        paragraphs = paragraphs[1:]
    return ("\n\n".join(paragraphs) + "\n").encode()


def taken(name: str, kinds: list[str]) -> bool:
    return any(name.endswith(k) or name == k or (k.endswith("_") and name.startswith(k))
               for k in kinds)


def fetch_git(root: Path, source, manifest: dict):
    part, name, url, commit, licence, dirs, kinds, most = source
    tree = checkout(root, url, commit, dirs)
    files = []
    for d in dirs:
        base = tree / d
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(tree)
            if any(p.startswith(".") for p in rel.parts) or not path.is_file() or path.is_symlink():
                continue
            # A catalogue holds the text twice, in two languages.
            limit = 4 * MAX_FILE if name in TEXT_OF else MAX_FILE
            if taken(path.name, kinds) and path.stat().st_size <= limit:
                files.append(rel)
    files = sorted(set(files), key=lambda rel: hashlib.sha1(str(rel).encode()).hexdigest())
    out = root / part / name
    if out.exists():
        shutil.rmtree(out)
    total = n = 0
    for rel in files:
        data = (tree / rel).read_bytes()
        if b"\0" in data:
            continue
        if name == "lnav-logs" and len(data) > LOG_HEAD:
            data = data[:data.rfind(b"\n", 0, LOG_HEAD) + 1]
        target = out / rel
        if name in TEXT_OF:
            data = po_text(data, TEXT_OF[name])
            target = target.with_suffix(".txt")
        if total + len(data) > most:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        total += len(data)
        n += 1
    print(f"  {part}/{name}: {n} files, {total / 1e6:.1f} MB", flush=True)
    manifest[part].append({"source": f"{url.removeprefix('https://')} at {commit[:12]}",
                           "licence": licence, "bytes": total})


def fetch(root: Path):
    # A directory this fetch began, and was cut off in, is taken up again.
    begun = root / "fetch-begun"
    if root.exists() and any(root.iterdir()) and not begun.exists():
        sys.exit(f"{root} is not empty: give an empty or a new directory")
    root.mkdir(parents=True, exist_ok=True)
    begun.touch()
    manifest = {part: [] for part in PARTS}
    for part, (config, batches) in WIKIPEDIA.items():
        print(f"{part}: {config}", flush=True)
        fetch_wikipedia(root, part, config, batches, manifest)
    for source in GIT:
        print(f"{source[0]}/{source[1]}: {source[2]}", flush=True)
        fetch_git(root, source, manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    begun.unlink()
    print(f"done; the git objects in {root / 'git'} may be removed")


def make(root: Path, model: str, out_dir: Path, date: str | None):
    manifest = json.loads((root / "manifest.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    example = ["cargo", "run", "--release", "-q", "-p", "ares-tokenizer", "--example", "draft_vocab",
               "--"]
    for file, title, coverage, weights in MIXES:
        args = ["make", model, "--coverage", str(coverage), "--title", title,
                "--out", str(out_dir / file)]
        if date:
            args += ["--date", date]
        for part, weight in weights.items():
            sources = manifest[part]
            licences = sorted({s["licence"] for s in sources})
            what = PARTS[part] + ": " + "; ".join(s["source"] for s in sources)
            args += ["--corpus", str(root / part), "--weight", str(weight), "--name", part,
                     "--licence", ", ".join(licences), "--source", what]
        print(f"{file}: {title}", flush=True)
        subprocess.run(example + args, cwd=REPO, check=True)


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "fetch":
        fetch(Path(args[1]))
    elif len(args) in (4, 6) and args[0] == "make":
        date = None
        if len(args) == 6:
            if args[4] != "--date":
                sys.exit(__doc__)
            date = args[5]
        make(Path(args[1]), args[2], Path(args[3]), date)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
