#!/usr/bin/env python3
"""Count words in files, ignoring case and punctuation."""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[A-Za-z\u00C0-\u024F]+(?:'[a-z]+)?")


def count_words(paths, *, top=10, min_len=3):
    counts: Counter[str] = Counter()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                for match in WORD.finditer(line):
                    word = match.group(0).lower()
                    if len(word) >= min_len:
                        counts[word] += 1
    return counts.most_common(top)


class Report:
    def __init__(self, rows):
        self.rows = rows

    def __str__(self):
        width = max((len(w) for w, _ in self.rows), default=0)
        return "\n".join(f"{w:<{width}}  {n:>6d}  {'#' * min(n, 40)}" for w, n in self.rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("-n", "--top", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        print(Report(count_words(args.files, top=args.top)))
    except FileNotFoundError as e:
        print(f"error: {e.filename}: not found", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
