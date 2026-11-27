# Experiments: draft vocabularies

Dated measurements of the draft vocabularies of `--draft-vocab`: the ready lists
of [`../../user/draft-vocab/`](../../user/draft-vocab/) and the lists first made
for it, the fork's list they replace, and no list. The guide is [`../../user/draft-vocab.md`](../../user/draft-vocab.md).
Not kept current: each names its date, build and machine.

## The first lists against the fork's and none, measured (2026-11-27)

Measured 2026-11-27 on the development machine (RTX 4070 and RTX 3060, peer
access), the build of the branch (main 83408e52 with the tool, which does not
touch the server), with the 27B UD-Q4_K_S at the owner's production settings:
tensor split 0.625 / 0.375, 262144 cells for 2 sequences, `--kv q5`, MTP and
n-gram drafts, `--ram 16G`, a fresh prompt cache. Each session loads the server
with one list (or none) and asks eight prompts, two each of German prose,
English prose, code (Python, Rust) and shell (a compose file with commands and
their output in English; ZFS diagnosis on Proxmox in German), 400 tokens each,
greedy, without thinking, after a warm-up. The sessions of the lists ran in
turn, five to a hold of the GPU lock, in three rounds whose order rotated; other
agents' builds and runs kept the machine at the loads given. Tokens per second
are the client's (first to last token of the stream), the mean over the
prompts of a kind and over the rounds; the share in brackets is the drafts the
model accepted of those proposed (MTP and n-gram together).

The lists: none (the drafters' device holds the first 155,200 rows of the
head under this split, so the drafts come from those tokens), the fork's
(`draft-vocab-qwen3.8-ende.txt`, 66,343 ids), the four lists first made for
the repository and the general mix at coverage 0.98 (40,851 ids). Of those
four, only `qwen35-en.txt` and `qwen35-code.txt` are shipped: on the owner's
decision of the same day the shipped lists are English only, and the general
list (German, English, code and shell, weights 0.35, 0.3, 0.2, 0.15) and the
English and German list were made with German parts that left the default
corpus. "The general list" in this section is that first one; the English
general list that replaced it is measured in the next section.

| List | ids | German | English | code | shell | all | MTP link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | (155,200) | 62.1 (46 %) | 75.1 (56 %) | 104.9 (72 %) | 81.4 (53 %) | 80.9 (57 %) | 2.2 to 2.3 ms |
| the fork's | 66,343 | 73.6 (51 %) | 76.3 (50 %) | 112.4 (69 %) | 86.9 (52 %) | 87.3 (55 %) | 1.5 ms |
| the general list (with German) | 62,437 | 75.1 (52 %) | 78.7 (50 %) | 113.3 (68 %) | 91.5 (53 %) | **89.6** (55 %) | 1.4 to 1.5 ms |
| the general mix at 0.98 | 40,851 | 73.2 (50 %) | 75.8 (46 %) | 112.2 (63 %) | 94.2 (55 %) | 88.8 (53 %) | 1.3 ms |
| English and German | 55,841 | 75.3 (52 %) | 81.3 (50 %) | 102.7 (60 %) | 88.1 (52 %) | 86.9 (53 %) | 1.4 ms |
| `qwen35-en.txt` | 50,617 | 61.3 (38 %) | 81.4 (56 %) | 103.2 (64 %) | 82.2 (47 %) | 82.0 (51 %) | 1.3 ms |
| `qwen35-code.txt` | 30,826 | 48.9 (26 %) | 71.7 (52 %) | 117.1 (73 %) | 82.2 (45 %) | 80.0 (48 %) | 1.2 ms |

Three sessions each of none, the fork's list and the general list, two of the
others. The MTP link is the price of one link of the draft chain that the
server measures at load (its log's "mtp by drafts", one draft). The sessions in
the order they ran, with the mean of their eight prompts:

| Session (load average) | tokens/s | accepted |
| --- | --- | --- |
| none (4.6) | 80.9 | 56 % |
| the fork's (3.5) | 87.8 | 54 % |
| general (3.1) | 89.5 | 55 % |
| general at 0.98 (3.0) | 89.2 | 53 % |
| code (2.7) | 79.8 | 49 % |
| English and German (4.2) | 86.7 | 54 % |
| English (3.0) | 81.8 | 51 % |
| general (2.8) | 89.9 | 56 % |
| the fork's (2.8) | 87.1 | 55 % |
| none (2.8) | 81.0 | 56 % |
| code (2.7) | 80.2 | 48 % |
| general at 0.98 (2.8) | 88.5 | 54 % |
| English (2.6) | 82.2 | 50 % |
| English and German (2.8) | 87.0 | 53 % |
| general (2.9) | 89.5 | 55 % |
| none (2.3) | 80.8 | 58 % |
| the fork's (2.5) | 86.9 | 56 % |

What it shows:

- **Without a list the 27B drafts German badly.** Under this split the
  drafters' device holds the head rows of tokens 0 to 155,199, and those cover
  0.84 of the German replies (0.88 of the German Wikipedia sample): many German
  pieces have higher ids. They also leave out the end-of-turn token, the
  thinking tags and the tool-call tags (ids from 248,044 on). English and code
  lose almost nothing (0.998 and 1.000), and a link reads 155,200 rows where a
  list's reads 40,000 to 66,000: 2.2 ms against 1.2 to 1.5.
- **The general list is the fastest overall**, 11 % above none and 2.6 % above
  the fork's list, ahead of it on every kind (German +2 %, English +3 %, code +1 %,
  shell +5 %). Its sessions were within 0.5 % of each other.
- **A smaller list does not pay on the 27B.** At coverage 0.98 (40,851 ids) a
  link is 0.15 ms cheaper, but the drafts the model accepts fall more (English
  75.8 against 78.7); shell, with its few distinct tokens, gains.
- **A list made for other text costs.** The code list (30,826 ids, the
  cheapest links) is the fastest on code, 3 % above the general list, and 21 %
  below none on German, where it covers 0.41 of the replies; the English list
  is as slow as none on German. The English and the English and German lists
  are the fastest on English (81.4 and 81.3), and lose 9 % on code.
- **The acceptance alone misleads.** None has the highest share of accepted
  drafts and the lowest speed: with dearer links the policy drafts less, and
  what it drafts is the easy part.

Coverage of the replies themselves (the 33 distinct replies of these sessions,
13,224 tokens, `draft_vocab check`):

| List | German | English | code | shell |
| --- | --- | --- | --- | --- |
| none (tokens 0 to 155,199) | 0.837 | 0.998 | 1.000 | 0.919 |
| the fork's | 0.993 | 0.938 | 0.991 | 0.981 |
| the general list (with German) | 0.999 | 0.976 | 0.996 | 0.988 |
| the general mix at 0.98 | 0.987 | 0.910 | 0.974 | 0.981 |
| English and German | 0.999 | 0.988 | 0.937 | 0.975 |
| `qwen35-en.txt` | 0.746 | 0.986 | 0.936 | 0.848 |
| `qwen35-code.txt` | 0.406 | 0.823 | 0.986 | 0.708 |

The English replies are two (the printing press, the immune system), and what
the lists lack of them are the words of those topics (" movable", "utenberg",
" antibodies"): a small sample, so these figures say more about how a list
falls off on text unlike its corpus than about English in general.

The replies were not the same in every session: greedy text with drafts took
another token at a near tie in some sessions (German: the title "Vom
Dampfpferd zum ICE" against "Von der Dampfpionierzeit zur
Hochgeschwindigkeit"), with any list and without one, and the same reply came
under different lists (one German reply in five sessions: none, the fork's,
the general list and the general mix at 0.98). The lists do not decide the
text; most likely the windows the policy picks do, whose products of other
widths round differently. The speeds compare replies to the same prompts, not
the same replies.

VRAM at the end of a session (`nvidia-smi`) was 11,449 to 11,545 MiB on the
RTX 4070 and 9,987 to 10,201 MiB on the RTX 3060 with every list; the device keeps the embedding rows
of the list's tokens for the draft chain (131 MiB for the general list, 326 MiB
for the 155,200 tokens without a list).

## The English lists, measured (2026-11-27)

Measured the same day, after the shipped lists became English only, on the same
machine, build and settings and with the same client: the 27B at the
production settings, greedy replies of 400 tokens without thinking. Six
prompts: the two English prose prompts and the two code prompts of the section
above, the compose prompt, and the ZFS prompt asked in English. Four lists in
four rounds whose order rotated, eight sessions to a hold of the GPU lock (two
holds, 08:43 to 09:01), at load averages of 2.7 to 10.8. The general list here
is the English one that is shipped: English Wikipedia 0.3, English technical
documents 0.2, code 0.3, shell 0.2 (the first general list's weights without
its German parts, the prose half of the whole), 57,903 ids at coverage 0.995.

| List | ids | English | code | shell | all | MTP link |
| --- | --- | --- | --- | --- | --- | --- |
| none | (155,200) | 76.4 (57 %) | 105.2 (71 %) | 89.5 (58 %) | 90.4 (62 %) | 2.3 ms |
| the fork's | 66,343 | 78.5 (52 %) | 106.1 (72 %) | 94.1 (53 %) | 92.9 (58 %) | 1.5 ms |
| `qwen35-en.txt` | 50,617 | 80.8 (51 %) | 103.5 (62 %) | 90.1 (49 %) | 91.5 (54 %) | 1.4 ms |
| `qwen35-general.txt` | 57,903 | 80.5 (52 %) | 112.5 (66 %) | 95.9 (54 %) | **96.3** (57 %) | 1.4 ms |

| Session (load average) | tokens/s | accepted |
| --- | --- | --- |
| none (3.8) | 89.8 | 62 % |
| the fork's (3.3) | 92.4 | 58 % |
| English (3.0) | 90.2 | 54 % |
| general (10.8) | 95.4 | 58 % |
| the fork's (10.1) | 93.4 | 57 % |
| English (7.5) | 91.9 | 52 % |
| general (5.4) | 97.9 | 57 % |
| none (4.2) | 90.2 | 61 % |
| English (9.7) | 92.7 | 53 % |
| general (5.9) | 96.4 | 58 % |
| none (4.4) | 91.0 | 62 % |
| the fork's (3.5) | 92.1 | 58 % |
| general (2.8) | 95.3 | 57 % |
| none (2.7) | 90.6 | 61 % |
| the fork's (3.1) | 93.6 | 60 % |
| English (3.1) | 91.1 | 54 % |

- **The general list is the fastest**: 6.5 % above none, 3.7 % above the
  fork's list and 5.2 % above the English list. Each of its four sessions
  (95.3 to 97.9) was faster than every session of the other lists (89.8 to
  93.6). Against the fork's list it gains 2.5 % on English prose, 6.0 % on
  code and 1.9 % on shell.
- **The English list** is as fast as the general list on English prose (80.8
  against 80.5) and slower than no list on code (103.5 against 105.2): its
  corpus has no code.
- **Without German, no list loses little coverage here**: the tokens 0 to
  155,199 that the drafters' device holds without a list cover 0.998 of the
  English replies, all of the code and 0.998 of the shell replies. What a list
  gains on this text is the cheaper link (1.4 ms against 2.3) and the
  end-of-turn token.

Coverage of the replies themselves (the 20 distinct replies of these
sessions, 8,015 tokens):

| List | English | code | shell |
| --- | --- | --- | --- |
| none (tokens 0 to 155,199) | 0.998 | 1.000 | 0.998 |
| the fork's | 0.944 | 0.990 | 0.977 |
| `qwen35-en.txt` | 0.986 | 0.935 | 0.942 |
| `qwen35-general.txt` | 0.982 | 0.995 | 0.981 |
| `qwen35-code.txt` | 0.820 | 0.986 | 0.952 |

What the general list lacks of them are the words of their topics
(" Gutenberg") and of the ZFS reply's tables (" ONLINE", " WRITE"). Only this
weighting was measured; equal weights would give 56,272 ids, more prose
59,018. The general list covers 0.70 of the German replies of the section
above and 0.80 of the German Wikipedia sample. VRAM at the end of a session was
11,457 to 11,549 MiB and 9,987 to 10,201 MiB with every list; the device keeps 121.5 MiB of embedding rows
for the general list's tokens.

## Flash-Next with its MTP block, measured (2026-11-27)

Measured 2026-11-27 on the same machine and build: Flash-Next IQ2_XS on both
cards, `--ram 8G`, its MTP block from `mtp-q2_0.gguf`, the same eight prompts
with 200 tokens each; no list, the fork's list and the first general list (with
German), two rounds in turn (none, the fork's, general; general, the fork's,
none):

| Session (load average) | tokens/s, mean of the prompts | accepted | MTP link |
| --- | --- | --- | --- |
| none (32.4) | 21.1 | 28 % | 1.4 ms |
| the fork's (13.5) | 27.9 | 35 % | 0.9 ms |
| general (10.7) | 26.3 | 50 % | 0.8 ms |
| general (6.6) | 14.4 | 22 % | 0.9 ms |
| the fork's (23.4) | 11.8 | 37 % | 0.9 ms |
| none (16.9) | 28.4 | 41 % | 1.4 ms |

The run does not tell the lists apart: sessions of one list differ by a factor
of two, with other agents' builds keeping the machine at loads of 7 to 32, and
at `--ram 8G` a quarter of the experts only on the SSD and the misses computed
on the CPU. What is solid is the price of a link, measured at load: 1.4 ms with
the whole head (the device that holds the output head holds all of it), 0.8 to
0.9 ms with a list of 62,000 or 66,000 tokens, as on 2026-11-17 (1.38 to 1.40
against 0.85 to 0.88, in [`speculation.md`](speculation.md)). Whether that
shows in tokens per second wants a quiet machine, or the test machine, where
the experts fit in RAM.
