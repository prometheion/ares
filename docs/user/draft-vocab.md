# Draft vocabularies

`--draft-vocab <FILE>` names the tokens the model drafters (the MTP block and a
DFlash drafter) may propose. This page says what such a list does, which ready
lists there are, and how to make and check your own.

## What the list does

A model drafter ends every draft with the output head: one row of weights for
each token of the vocabulary, 248,320 rows for the Qwen3.5 and Qwen3.8 models.
With a list, the rows of its tokens come first in the head (the same weights in
another order, nothing is copied), and a draft reads those rows alone. A shorter
list makes each draft cheaper. Without a list, a drafter reads every row of the
head on its device. Under a tensor split that is the device's share of the rows,
and the tokens whose rows are on the other device cannot be drafted: for the
27B on two cards at shares 0.625 / 0.375 the tokens 0 to 155,199, which leave
out a sixth of the tokens of German text, the end-of-turn token and the
thinking and tool-call tags.

A token that is not in the list can never be drafted. When the model writes one,
the draft at that place is wrong and the model rejects it. The list does not
decide the text: the model verifies every draft. What a list changes is the
speed.

- **It pays** when the list holds what the model writes. The 27B on two cards
  at the production settings decodes 11 % faster with the general list than
  without one (German prose 21 % faster), and 2.6 % faster than with the
  llama.cpp fork's list it replaces; an MTP link takes 1.4 ms instead of 2.2.
- **It costs** when the text has many tokens outside the list: a language or a
  kind of text the list was not made for. Each such token breaks a chain of
  drafts. The code list makes German prose 35 % slower than the general list
  does, and 21 % slower than no list at all.
- **Smaller is not always better.** The general mix at coverage 0.98 (40,851
  tokens) has links 0.15 ms cheaper than at 0.995, and was 1 % slower on the
  27B: it loses more drafts than the links save.

A list is read at load. Every id must be a token of the model, listed once, and
the list may not be longer than the rows of the head on the drafters' device.
The 27B under a layer split takes no list.

## The ready lists

Qwen3.8 27B, Qwen3.8 Flash-Next and Qwen3.5 0.8B share one tokenizer (the
files of all three have the fingerprint `c2a79d23...`), so one set of lists
serves them all. They are in [`draft-vocab/`](draft-vocab/), each made at
coverage 0.995 of its corpus, whose parts, sizes, weights, licences and pinned
sources its header names:

| File | For | ids | Corpus (weight) | The 27B on two cards, tokens/s |
| --- | --- | --- | --- | --- |
| [`qwen35-general.txt`](draft-vocab/qwen35-general.txt) | a mix of German, English, code and shell | 62,437 | German Wikipedia (0.2) and technical documents (0.15), English Wikipedia (0.15) and technical documents (0.15), code (0.2), shell (0.15) | 89.6 on the mix: German 75.1, English 78.7, code 113.3, shell 91.5 |
| [`qwen35-en-de.txt`](draft-vocab/qwen35-en-de.txt) | English and German prose | 55,841 | Wikipedia and technical documents in both languages | 86.9: German 75.3, English 81.3, code 102.7, shell 88.1 |
| [`qwen35-en.txt`](draft-vocab/qwen35-en.txt) | English prose | 50,617 | English Wikipedia (0.6) and technical documents (0.4) | 82.0: English 81.4; German 61.3, as slow as no list |
| [`qwen35-code.txt`](draft-vocab/qwen35-code.txt) | code | 30,826 | Python, Rust, Go, TypeScript and C (0.75), English technical documents (0.25) | 80.0: code 117.1; German 48.9 |
| no list | | | | 80.9: German 62.1, English 75.1, code 104.9, shell 81.4 |
| the fork's list (`draft-vocab-qwen3.8-ende.txt`) | | 66,343 | not named | 87.3: German 73.6, English 76.3, code 112.4, shell 86.9 |

The numbers are greedy replies of 400 tokens to two prompts of each kind,
alternated over the lists, measured on 2026-11-27 with the settings of the
first row of "What was run" in [models.md](models.md#what-was-run)
([the measurement](../dev/experiments/draft-vocab.md)). Take the general list
when the traffic is mixed, the English and German list for prose, the code list
only for a server that writes code and little else.

## The format

One token id per line. Blank lines and lines that begin with `#` are left out.
The lists of the tool begin with a header of `#` lines:

| Line | What it says |
| --- | --- |
| `# tokenizer:` | the tokenizer the ids belong to: its kind, its number of tokens and its fingerprint, the SHA-256 of its vocabulary and merges in the model file |
| `# read from:` | the model file the tokenizer was read from |
| `# made:` | the date |
| `# rule:` | the coverage or number of ids asked for |
| `# part:` | one line per part of the corpus: name, weight, size, licence and source |
| `# coverage:` | the coverage reached, of the mix and of each part |
| `# ids:` | the number of ids, and how many of them are kept always |
| `# kept always:` | the tokens a drafter must always be able to propose |

The server does not read the header; `draft_vocab check` compares its tokenizer
with a model's.

## Choosing a corpus and a coverage

The list should hold the tokens the model writes for you, so the corpus is text
like the model's answers: in your languages, about your topics, with your kinds
of code and configuration. What you send the model counts only as far as the
model writes it back (code it edits, logs it quotes). If you let the model
think, the thinking is text it writes too.

A part's **weight** is its share of what the model writes. The tool counts each
part by its weight and not by its size, so a large part does not drown a small
one: a part of 10 MB with weight 0.2 counts as much as a part of 1 MB with
weight 0.2.

**Coverage** is the share of the corpus' tokens the list holds. The tool takes
the most frequent tokens until it is reached. The most frequent tokens cover
most text, and each further step of coverage takes many more tokens: on the mix
of the general list, 0.9 takes 14,078 ids, 0.98 40,851, 0.99 52,283, 0.995
62,437 and 0.999 81,163 (the tool prints this curve for every corpus).

- **0.995** is where the ready lists are made, and on the 27B the general list
  at 0.995 was faster than at 0.98. Take it unless you measure otherwise.
- A corpus of encyclopedic text has more rare names than the model's replies,
  so a list covers the replies better than its corpus: the general list covers
  0.994 of its German Wikipedia part and 0.999 of the 27B's German replies.
- Text unlike the corpus loses more, and by topic: the same list covered 0.976
  of two English essays, whose missing tokens were the words of their topics
  (" movable", " antibodies").
- Under the 27B's tensor split, a list longer than the head rows of the
  drafters' device (155,200 at shares 0.625 / 0.375) is refused at load.

## Making your own

The tool is an example of the tokenizer crate. It reads the tokenizer from the
model file, so it tokenizes exactly as the server does:

```sh
cargo run --release -p ares-tokenizer --example draft_vocab -- make \
  ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf --coverage 0.995 \
  --title "German answers and shell" --out my-vocab.txt \
  --corpus ~/corpus/answers-de --weight 0.6 --licence "my own" \
  --corpus ~/corpus/shell --weight 0.4 --licence "MIT" --source "my scripts"
```

- `--corpus PATH` is one part: a file or a directory, read with everything
  under it (hidden entries, files with a NUL byte and files that are not UTF-8
  are left out). The options after it describe that part: `--weight` (1 if not
  given), `--name` (the last part of the path if not given), `--licence` and
  `--source`, which the header names.
- `--coverage C` takes tokens until the coverage is at least `C`; `--ids N`
  takes the `N` most frequent instead.
- The tokens a drafter must always be able to propose go into every list: the
  end-of-text and end-of-turn tokens, the control tokens the chat template
  writes, and the tokens of its thinking and tool-call tags. `draft_vocab info
  MODEL.gguf` prints them, with the tokenizer's fingerprint.
- `--threads N` (4 if not given) counts that many files at once; a few hundred
  MB take seconds.

`tools/draft-vocab.py` makes the ready lists again: `fetch DIR` downloads their
corpus into an empty directory (about 250 MB, from the pinned sources in the
script), and `make DIR MODEL.gguf OUT_DIR` runs the tool for each of them.

## Checking a list on your own text

```sh
cargo run --release -p ares-tokenizer --example draft_vocab -- check \
  ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf docs/user/draft-vocab/qwen35-general.txt \
  --corpus ~/corpus/answers-de --corpus ~/corpus/code
```

It says whether the tokenizer the list's header names is the model's, whether
the list holds the tokens kept always, what it covers of each part (and of their
mix, by the weights given), and the most frequent tokens it lacks (`--missing
N`, 30 if not given). A coverage well below the one in the list's header means
your text is unlike its corpus: make a list from your own text, or add it as a
part.

The best text to check a list on is the model's own replies to your
requests: `--log-text` writes them into the server's log.

The server's log and `/metrics` say how many drafts were proposed and accepted
(`drafts`, `drafts_accepted`; [quickstart.md](quickstart.md#logs-and-metrics)).
Compare a run with the list and one without on the same requests, by tokens
per second: the share of accepted drafts misleads, because with dearer drafts
the server drafts less and accepts more of what it drafts.
