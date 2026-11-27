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

- **It pays** when the list holds what the model writes. With the general
  list the 27B on two cards at the production settings decodes 6.5 % faster
  on English, code and shell than without a list, and 3.7 % faster than with
  the llama.cpp fork's list it replaces; an MTP link takes 1.4 ms instead of
  2.3.
- **It costs** when the text has many tokens outside the list: a language or a
  kind of text the list was not made for. Each such token breaks a chain of
  drafts. The ready lists are English: the general and the English list cover
  0.70 to 0.82 of German text, and the English list made German prose no faster
  than no list at all (see
  [Adding a language](#adding-a-language)).
- **Smaller is not always better.** A general list at coverage 0.98 (40,851
  tokens, measured with German parts) has links 0.15 ms cheaper than at 0.995,
  and was 1 % slower on the 27B: it loses more drafts than the links save.

A list is read at load. Every id must be a token of the model, listed once, and
the list may not be longer than the rows of the head on the drafters' device.
The 27B under a layer split takes no list.

## The ready lists

Qwen3.8 27B, Qwen3.8 Flash-Next and Qwen3.5 0.8B share one tokenizer (the
files of all three have the fingerprint `c2a79d23...`), so one set of lists
serves them all. The lists in [`draft-vocab/`](draft-vocab/) are English: English
prose, code, and the text of tools (shell commands, configuration files,
logs). Each is made at coverage 0.995 of its corpus, whose parts, sizes,
weights, licences and pinned sources its header names:

| File | For | ids | Corpus (weight) |
| --- | --- | --- | --- |
| [`qwen35-general.txt`](draft-vocab/qwen35-general.txt) | English prose, code and shell together | 57,903 | English Wikipedia (0.3), English technical documents (0.2), Python, Rust, Go, TypeScript and C (0.3), command pages, compose and Kubernetes files, shell scripts and logs (0.2) |
| [`qwen35-en.txt`](draft-vocab/qwen35-en.txt) | English prose | 50,617 | English Wikipedia (0.6), English technical documents (0.4) |
| [`qwen35-code.txt`](draft-vocab/qwen35-code.txt) | code | 30,826 | the code above (0.75), English technical documents (0.25) |

Take the general list unless the server writes one kind of text only. What
they gave the 27B on two cards at the production settings, in tokens per
second, greedy replies of 400 tokens to two prompts of each kind:

| List | English prose | code | shell | all |
| --- | --- | --- | --- | --- |
| `qwen35-general.txt` | 80.5 | 112.5 | 95.9 | **96.3** |
| `qwen35-en.txt` | 80.8 | 103.5 | 90.1 | 91.5 |
| no list | 76.4 | 105.2 | 89.5 | 90.4 |
| the fork's list (66,343 ids, not shipped) | 78.5 | 106.1 | 94.1 | 92.9 |

The code list, measured in an earlier run of the same day, was 3 % faster on
code than a general list and 9 % slower on English prose (71.7 against 78.7).

Measured on 2026-11-27 with the settings of the first row of "What was run" in
[models.md](models.md#what-was-run), the lists alternated
([the measurement](../dev/experiments/draft-vocab.md)).

On Flash-Next a list makes a link of its MTP block cheaper (0.8 to 0.9 ms
against 1.4 ms with the whole head); what that gives in tokens per second could
not be told apart from the noise of a busy machine with experts on the SSD
(the same measurement).

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
of the general list, 0.9 takes 11,681 ids, 0.98 37,474, 0.99 48,322, 0.995
57,903 and 0.999 75,209 (the tool prints this curve for every corpus).

- **0.995** is where the ready lists are made, and on the 27B a general list at
  0.995 was faster than at 0.98. Take it unless you measure otherwise.
- A corpus of encyclopedic text has more rare names than the model's replies,
  so a list covers the replies better than its corpus.
- Text unlike the corpus loses more, and by topic: the general list covered
  0.982 of the English essays of the measurement, whose missing tokens were the
  words of their topics (" Gutenberg"), and 0.981 of the shell replies, which
  lacked the words of `zpool status` (" ONLINE").
- Under the 27B's tensor split, a list longer than the head rows of the
  drafters' device (155,200 at shares 0.625 / 0.375) is refused at load.

## Making your own

The tool is an example of the tokenizer crate. It reads the tokenizer from the
model file, so it tokenizes exactly as the server does:

```sh
cargo run --release -p ares-tokenizer --example draft_vocab -- make \
  ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf --coverage 0.995 \
  --title "My answers and scripts" --out my-vocab.txt \
  --corpus ~/corpus/answers --weight 0.6 --licence "my own" \
  --corpus ~/corpus/scripts --weight 0.4 --licence "MIT" --source "my scripts"
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
corpus into an empty directory (about 100 MB of text, from the pinned sources
in the script), and `make DIR MODEL.gguf OUT_DIR` runs the tool for each of
them.

### Adding a language

A model that writes another language needs a list that holds it. The ready
lists leave much of a language out: the general and the English list cover
0.70 to 0.82 of German text. Make a list from the corpus of the ready lists and
a part in your language, weighted by its share of what the model writes.
German, for example, with German Wikipedia articles and German technical
documents of your own, each a directory of UTF-8 text files:

```sh
python3 -I tools/draft-vocab.py fetch ~/corpus/ready
cargo run --release -p ares-tokenizer --example draft_vocab -- make \
  ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf --coverage 0.995 \
  --title "German, English, code and shell" --out de-general.txt \
  --corpus ~/corpus/de-wiki --weight 0.2 --licence "CC BY-SA" --source "German Wikipedia" \
  --corpus ~/corpus/de-docs --weight 0.15 \
  --corpus ~/corpus/ready/wiki-en --weight 0.15 \
  --corpus ~/corpus/ready/docs-en --weight 0.15 \
  --corpus ~/corpus/ready/code --weight 0.2 \
  --corpus ~/corpus/ready/shell --weight 0.15
draft_vocab check ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf de-general.txt --corpus ~/corpus/de-answers
```

(`draft_vocab` stands for the same `cargo run` line.) A list made this way, with
53 MB of German Wikipedia and 2 MB of German technical documents, had 62,437
ids. It made the 27B's German prose 21 % faster than no list and 22 % faster
than the English list, lost 3 % on English prose against the English list, and
was faster than it on code and shell (the first section of the measurement).
Keep the English parts unless the model writes only your language: code,
commands and logs are English text too. A list of one kind of text alone is
fast on it and slow on everything else: the code list made German prose 21 %
slower than no list.

## Checking a list on your own text

```sh
cargo run --release -p ares-tokenizer --example draft_vocab -- check \
  ~/models/Qwen3.8-27B-UD-Q4_K_S.gguf docs/user/draft-vocab/qwen35-general.txt \
  --corpus ~/corpus/answers --corpus ~/corpus/code
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
