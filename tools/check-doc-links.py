#!/usr/bin/env python3
"""Check the links between the documents and the citations of them in the code.

Run from anywhere inside the repository:

    python3 tools/check-doc-links.py

It checks, in every tracked text file:
- each relative Markdown link of a document points to a file that exists, and its
  `#anchor`, if any, to a heading of that file;
- each mention of a document of this repository (`docs/...`, `README.md`, or a path that
  starts with a directory of docs/) names a file that exists, unless it is called the
  former one ("the former `docs/X.md`");
- each citation of a paragraph, written `<file>.md, "<name>"` or `"<name>" in <file>.md`
  (backticks and comment markers between the lines do not matter), names a heading or the
  bold or italic lead of a paragraph of that file.

It prints one line per problem and exits with 1 if there was any.
"""
import os
import re
import subprocess
import sys

TEXT = ('.rs', '.cu', '.cuh', '.h', '.cpp', '.py', '.md', '.toml', '.comp', '.glsl', '.sh')
SKIP = ('crates/tokenizer/tests/corpus/',)
DOC_DIRS = ('user', 'dev', 'architecture', 'experiments', 'studies')
DOC_NAMES = ('OPEN.md', 'history.md', 'ARCHITECTURE.md', 'FEATURES.md')

LINK = re.compile(r'\[[^\]]*\]\(([^)\s]+)\)')
PATH = r'(?P<path>[A-Za-z0-9_./-]+\.md)'
# `<path>.md, "<name>"`, also `<path>.md "<name>"` and `<path>.md, after "<name>"`.
CITE_AFTER = re.compile(PATH + r'`?\)?,?\s+(?:after\s+)?"(?P<name>[^"]{2,160})"')
# `"<name>" in <path>.md`, also `"<name>" under "<other>" in <path>.md`.
CITE_BEFORE = re.compile(r'"(?P<name>[^"]{2,160})"(?:\s+under\s+"[^"]+")?\s+in\s+'
                         r'(?:the\s+)?\[?`?' + PATH)
MENTION = re.compile(r'(?<![\w/.-])(?P<path>[A-Za-z0-9_./-]*[A-Za-z0-9_-]\.md)\b')
COMMENT = re.compile(r'^\s*(?://[/!]?|#|\*|--)\s?')


def root():
    out = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


def norm(text):
    text = text.replace('`', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.rstrip('.:,;').strip()


def slug(heading):
    text = heading.strip().lower().replace('`', '')
    text = re.sub(r'[^\w\- ]', '', text)
    return text.replace(' ', '-')


class Doc:
    def __init__(self, path):
        with open(path) as f:
            self.lines = f.read().split('\n')
        self.anchors = set()
        self.names = []
        fence = False
        counts = {}
        for i, line in enumerate(self.lines):
            if line.startswith('```'):
                fence = not fence
                continue
            if fence:
                continue
            m = re.match(r'^(#+)\s+(.*)$', line)
            if m:
                s = slug(m.group(2))
                n = counts.get(s, 0)
                counts[s] = n + 1
                self.anchors.add(s if n == 0 else '%s-%d' % (s, n))
                self.names.append(norm(m.group(2)))
                continue
            body = re.sub(r'^\s*(?:[-*]\s+|\d+\.\s+)?(?:\[[ x]\]\s+)?(?:\d+\.\s+)?', '', line)
            for lead in (r'^\*\*(.+?)\*\*', r'^\*([^*].*?)\*', r'^_(.+?)_'):
                m = re.match(lead, body)
                if m:
                    # A lead may run on to the next line.
                    if m.group(0).count('*') < 2 and i + 1 < len(self.lines):
                        continue
                    self.names.append(norm(m.group(1)))
                    break
            # A bold lead split over two lines.
            m = re.match(r'^\*\*([^*]+)$', body)
            if m and i + 1 < len(self.lines):
                rest = re.match(r'^\s*([^*]*)\*\*', self.lines[i + 1])
                if rest:
                    self.names.append(norm(m.group(1) + ' ' + rest.group(1)))

    def has(self, name):
        want = norm(name)
        if not want:
            return True
        for have in self.names:
            if have == want or have.startswith(want) or want.startswith(have + ' ('):
                return True
        return False


def main():
    top = root()
    os.chdir(top)
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                           capture_output=True, text=True, check=True).stdout.split()
    files = [f for f in files if f.endswith(TEXT) and not f.startswith(SKIP) and os.path.exists(f)
             and f != 'tools/check-doc-links.py']
    docs = {}

    def doc(path):
        if path not in docs:
            docs[path] = Doc(path)
        return docs[path]

    def resolve(path, src):
        """The file a mention in `src` names, or None. Tries the mention's own directory,
        the repository root and docs/."""
        path = path.split('#')[0]
        here = os.path.dirname(src)
        for base in (here, '', 'docs', 'docs/dev'):
            cand = os.path.normpath(os.path.join(base, path))
            if os.path.isfile(cand):
                return cand
        return None

    def ours(path):
        first = path.split('/')[0]
        return (re.match(r'docs/(user|dev|reference)/|docs/[A-Z]+\.md$', path) or first in DOC_DIRS or path in DOC_NAMES
                or path == 'README.md' or path.startswith('../') and path.endswith('.md')
                and ('docs' in path or 'README' in path))

    problems = []
    for src in files:
        with open(src, errors='replace') as f:
            raw = f.read()
        lines = raw.split('\n')
        is_md = src.endswith('.md')
        # Links of Markdown files.
        if is_md:
            fence = False
            for no, line in enumerate(lines, 1):
                if line.startswith('```'):
                    fence = not fence
                if fence:
                    continue
                for m in LINK.finditer(line):
                    target = m.group(1)
                    if re.match(r'^[a-z]+:', target):
                        continue
                    path, _, anchor = target.partition('#')
                    if path:
                        full = os.path.normpath(os.path.join(os.path.dirname(src), path))
                        if not os.path.exists(full):
                            problems.append('%s:%d: link to a missing file: %s' % (src, no, target))
                            continue
                    else:
                        full = src
                    if anchor and full.endswith('.md') and anchor not in doc(full).anchors:
                        problems.append('%s:%d: no heading for the anchor: %s' % (src, no, target))
        # Flatten comments so that citations across lines are seen whole.
        flat = []
        starts = []
        pos = 0
        for no, line in enumerate(lines, 1):
            text = line if is_md else COMMENT.sub('', line)
            starts.append((pos, no))
            flat.append(text.strip())
            pos += len(flat[-1]) + 1
        flat = ' '.join(flat)

        def line_of(offset):
            no = 1
            for p, n in starts:
                if p > offset:
                    break
                no = n
            return no

        for m in MENTION.finditer(flat):
            path = m.group('path')
            if not ours(path):
                continue
            # "the former `docs/X.md`" names a file that is gone on purpose.
            if flat[max(0, m.start() - 9):m.start()].rstrip('` ').endswith('former'):
                continue
            if resolve(path, src) is None:
                problems.append('%s:%d: names a missing document: %s'
                                % (src, line_of(m.start()), path))
        for pattern in (CITE_AFTER, CITE_BEFORE):
            for m in pattern.finditer(flat):
                path = m.group('path')
                if not ours(path):
                    continue
                full = resolve(path, src)
                if full is None:
                    continue  # reported above
                if not doc(full).has(m.group('name')):
                    problems.append('%s:%d: "%s" is no heading or paragraph lead of %s'
                                    % (src, line_of(m.start()), norm(m.group('name')), full))
    for p in problems:
        print(p)
    if problems:
        print('%d problems' % len(problems))
        return 1
    print('links and citations: clean (%d files)' % len(files))
    return 0


if __name__ == '__main__':
    sys.exit(main())
