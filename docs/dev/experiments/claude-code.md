# Experiments: Claude Code against Ares

Dated measurement logs, moved here verbatim on 2026-11-24 from the architecture document (now [`../architecture/`](../architecture/overview.md)) and from [`OPEN.md`](../../OPEN.md). They are not kept current: each names its date, build and machine.

## Claude Code against Ares (2026-11-22, Claude Code 2.1.301, print mode)

Claude Code in a sandbox of its own (own home, a throwaway project, `ANTHROPIC_BASE_URL` on
Ares), three small tasks: read a file and judge a function, fix it with Edit and run the tests
with Bash. Scripts and logs in `/var/tmp/ares-flashapi` (`notes.md`).

- Flash-Next (two cards split by layers, the driver with peer access being missing; `--ram
  8G`, its MTP block, machine load 8 to 27 from other builds):
  the tasks are done right, in 3 and 4 turns. Claude Code's first prompt is 16.3 thousand
  tokens (the 20 tools of its default set; 3.5 thousand with `--tools Read,Edit,Bash`), read at
  25 to 61 tokens/s: 3.5 to 4.5 minutes to the first token. A turn that continues the cached
  conversation has its first token in 3 to 9 s; replies come at 5 to 9 tokens/s with 50 to 90 %
  of the drafts accepted.

- The 27B (two cards, tensor split, its MTP block, peer access): the same tasks in 2 and 4
  turns, 37 s and 22 s. The first prompt at 1052 tokens/s (15.5 s), a continued turn's first
  token in 0.2 to 0.5 s, replies at 65 to 120 tokens/s.
