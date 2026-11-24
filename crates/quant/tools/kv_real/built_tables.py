#!/usr/bin/env python3
"""Tables of section 13 of docs/dev/studies/kv-real-data.md: the built mean per tile.

usage: built_tables.py OLD_RESULTS... -- BUILT_RESULTS [section]
OLD_RESULTS are result files of `kvreal run` with the default, tilemean and
extra sets (the reference rows), BUILT_RESULTS the file of `--set built`.
"""
import sys
from collections import defaultdict

args = sys.argv[1:]
i = args.index("--")
OLD, NEW = args[:i], args[i + 1]
section = args[i + 2] if len(args) > i + 2 else "all"


D = {}


def load(paths):
    R, S, E = {}, {}, {}
    for p in paths:
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if f[0] == "R":
                R[(f[1], f[2])] = dict(n=int(f[3]), rel=float(f[4]), rel99=float(f[5]), kl=float(f[6]), kl99=float(f[7]), sc=float(f[8]), sc99=float(f[9]))
            elif f[0] == "S":
                S[(f[1], int(f[2]), f[3], f[4])] = float(f[5])
            elif f[0] == "E":
                E[(f[1], f[2])] = (float(f[3]), float(f[4]))
            elif f[0] == "D":
                D[f[1]] = (float(f[2]), float(f[3]), float(f[4]))
    return R, S, E


Ro, So, Eo = load(OLD)
Rn, Sn, En = load([NEW])
R = {**Ro, **Rn}
S = {**So, **Sn}
PAIRS = [(8, 8), (6, 6), (5, 5), (4, 4), (6, 5), (5, 4), (8, 4), (4, 8)]
UNIT = {"rel": 1e3, "rel99": 1e3, "kl": 1e4, "sc": 1e3}


def val(fam, kb, vb, mask, scope, key):
    r = R.get((f"{fam}|K{kb}V{vb}|{mask}", scope))
    return None if r is None else r[key]


def fmt(v, key):
    return "-" if v is None else f"{v * UNIT[key]:.2f}"


ROWS = [
    ("absmax, nothing exact", "absmax", "none"),
    ("engine, old sliding policy (128/256 at q4)", "engine", "policy"),
    ("kvarn quantizer, kvarn's tail", "kvarn", "kvarn"),
    ("engine rows, kvarn's tail", "engine", "kvarn"),
    ("built tiles, kvarn's tail", "built", "kvarn"),
    ("rows, tile policy", "rows", "tile-policy"),
    ("built tiles, tile policy", "built", "policy"),
]


def table(key, scope="all", pairs=PAIRS, rows=ROWS):
    out = ["| candidate | " + " | ".join(f"K{k}/V{v}" for k, v in pairs) + " |", "|" + "---|" * (len(pairs) + 1)]
    for lab, fam, mask in rows:
        out.append(f"| {lab} | " + " | ".join(fmt(val(fam, k, v, mask, scope, key), key) for k, v in pairs) + " |")
    return "\n".join(out)


def layers_table(key, kb=4, vb=4):
    layers = sorted({s[1] for s in R if s[1].startswith("L") and s[0].startswith("built")})
    scopes = layers + ["all"]
    out = ["| candidate | " + " | ".join(scopes) + " |", "|" + "---|" * (len(scopes) + 1)]
    for lab, fam, mask in [("rows, tile policy", "rows", "tile-policy"), ("built tiles, tile policy", "built", "policy")]:
        out.append(f"| {lab} | " + " | ".join(fmt(val(fam, kb, vb, mask, sc, key), key) for sc in scopes) + " |")
    ch = []
    for sc in scopes:
        a, b = val("built", kb, vb, "policy", sc, key), val("rows", kb, vb, "tile-policy", sc, key)
        ch.append(f"{(a / b - 1) * 100:+.0f}%")
    out.append("| change | " + " | ".join(ch) + " |")
    return "\n".join(out)


def bins_table(key, kb=4, vb=4):
    scopes = ["t512-1023", "t1024-2047", "t2048+", "seg:english", "seg:german", "seg:code"]
    out = ["| candidate | " + " | ".join(s.replace("seg:", "") for s in scopes) + " |", "|" + "---|" * (len(scopes) + 1)]
    for lab, fam, mask in [("rows, tile policy", "rows", "tile-policy"), ("built tiles, tile policy", "built", "policy")]:
        out.append(f"| {lab} | " + " | ".join(fmt(val(fam, kb, vb, mask, sc, key), key) for sc in scopes) + " |")
    ch = []
    for sc in scopes:
        a, b = val("built", kb, vb, "policy", sc, key), val("rows", kb, vb, "tile-policy", sc, key)
        ch.append(f"{(a / b - 1) * 100:+.0f}%")
    out.append("| change | " + " | ".join(ch) + " |")
    return "\n".join(out)


def scopes_table(scopes, labels, kb=4, vb=4):
    out = ["| candidate | " + " | ".join(labels) + " |", "|" + "---|" * (len(scopes) + 1)]
    for key, name in (("rel", "output error"), ("kl", "KL")):
        for lab, fam, mask in [("rows, tile policy", "rows", "tile-policy"), ("built tiles, tile policy", "built", "policy")]:
            out.append(f"| {name}: {lab} | " + " | ".join(fmt(val(fam, kb, vb, mask, sc, key), key) for sc in scopes) + " |")
        ch = []
        for sc in scopes:
            a, b = val("built", kb, vb, "policy", sc, key), val("rows", kb, vb, "tile-policy", sc, key)
            ch.append(f"{(a / b - 1) * 100:+.0f}%")
        out.append(f"| {name}: change | " + " | ".join(ch) + " |")
    return "\n".join(out)


def layer_scopes():
    return sorted({s[1] for s in R if s[1].startswith("L") and s[0].startswith("built")}) + ["all"]


def steps_table(key):
    """old sliding policy -> the same codec under the tile policy (windows) -> built tiles (mean)."""
    out = ["| pair | old sliding policy, rows | tile policy, rows | change (wider windows) | tile policy, built tiles | change (mean) | total |", "|---|---|---|---|---|---|---|"]
    for kb, vb in PAIRS:
        o = val("engine", kb, vb, "policy", "all", key)
        r = val("rows", kb, vb, "tile-policy", "all", key)
        b = val("built", kb, vb, "policy", "all", key)
        if o is None:
            out.append(f"| K{kb}/V{vb} | - | {fmt(r, key)} | - | {fmt(b, key)} | {(b / r - 1) * 100:+.0f}% | - |")
        else:
            out.append(f"| K{kb}/V{vb} | {fmt(o, key)} | {fmt(r, key)} | {(r / o - 1) * 100:+.0f}% | {fmt(b, key)} | {(b / r - 1) * 100:+.0f}% | {(b / o - 1) * 100:+.0f}% |")
    return "\n".join(out)


def head_table(model_scopes=("all",)):
    """built|policy (arrays) against head|policy (the stored sequence and its reference attention)."""
    out = ["| pair | output error: arrays / head / change | KL: arrays / head / change | p99 output error: arrays / head | score error: arrays / head | distance of the outputs, query by query: mean / p99 / max |", "|---|---|---|---|---|---|"]
    for kb, vb in PAIRS:
        cells = []
        for key, with_change in (("rel", True), ("kl", True), ("rel99", False), ("sc", False)):
            a, b = val("built", kb, vb, "policy", "all", key), val("head", kb, vb, "policy", "all", key)
            if a is None or b is None:
                cells.append("-")
                continue
            text = f"{a * UNIT[key]:.3f} / {b * UNIT[key]:.3f}"
            if with_change:
                text += f" / {(b / a - 1) * 100:+.2f}%"
            cells.append(text)
        d = D.get(f"head|K{kb}V{vb}|policy")
        dist = "-" if d is None else f"{d[0]:.1e} / {d[1]:.1e} / {d[2]:.1e}"
        out.append(f"| K{kb}/V{vb} | " + " | ".join(cells) + f" | {dist} |")
    return "\n".join(out)


def head_worse():
    """Every scope and measure in which the stored sequence is worse than the arrays by more than 1%."""
    scopes = sorted({s[1] for s in R if s[0].startswith("head|")})
    lines, n = [], 0
    for kb, vb in PAIRS:
        for sc in scopes:
            for key in ("rel", "kl", "rel99", "sc"):
                a, b = val("built", kb, vb, "policy", sc, key), val("head", kb, vb, "policy", sc, key)
                if a is None or b is None:
                    continue
                n += 1
                if abs(b / a - 1) > 0.01:
                    lines.append(f"K{kb}/V{vb} {sc} {key}: arrays {a * UNIT[key]:.3f}, head {b * UNIT[key]:.3f} ({(b / a - 1) * 100:+.2f}%)")
    return n, lines


def worse():
    """Every scope, pair and measure in which built is worse than rows under the tile policy."""
    scopes = sorted({s[1] for s in Rn})
    n = 0
    lines = []
    for kb, vb in PAIRS:
        for sc in scopes:
            for key in ("rel", "rel99", "kl", "sc"):
                a, b = val("built", kb, vb, "policy", sc, key), val("rows", kb, vb, "tile-policy", sc, key)
                if a is None or b is None:
                    continue
                n += 1
                if a > b * (1 + 1e-9):
                    lines.append(f"K{kb}/V{vb} {sc} {key}: built {a * UNIT[key]:.3f} vs rows {b * UNIT[key]:.3f} ({(a / b - 1) * 100:+.1f}%)")
    return n, lines


def snr_table():
    layers = sorted({s[3] for s in S if s[3].startswith("L") and s[0] == "built"})
    out = ["| codec | side | 8 bits | 6 bits | 5 bits | 4 bits |", "|---|---|---|---|---|---|"]
    for fam, lab in [("engine", "rows"), ("built", "built tiles"), ("engine+tile-mean(128)", "what-if tile mean 128")]:
        for side in ("K", "V"):
            vals = []
            for b in (8, 6, 5, 4):
                v = [S[(fam, b, side, L)] for L in layers if (fam, b, side, L) in S]
                vals.append(f"{sum(v) / len(v):.1f}" if v else "-")
            out.append(f"| {lab} | {side} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def exact_table():
    E = {**Eo, **En}
    out = ["| pair | old sliding policy | tile policy | kvarn's tail | of the cells attended (mean) |", "|---|---|---|---|---|"]
    for kb, vb in PAIRS:
        a, c = E.get(("policy", f"K{kb}V{vb}")), E.get(("tile-policy", f"K{kb}V{vb}"))
        k = E.get(("kvarn", f"K{kb}V{vb}"))
        if a:
            out.append(f"| K{kb}/V{vb} | {a[0]:.0f} | {c[0]:.0f} | {k[0]:.0f} | {a[1]:.0f} |")
    return "\n".join(out)


def check():
    """built|kvarn against the what-if engine+tile-mean(128)|..|kvarn, raw values."""
    out = ["| pair | measure | what-if (mean in the original domain) | built (mean in the rotated domain) | difference |", "|---|---|---|---|---|"]
    for kb, vb in [(8, 8), (6, 6), (5, 5), (4, 4), (5, 4)]:
        w = "engine+tile-mean(128)" if (kb, vb) != (8, 8) else "engine"
        for key in ("rel", "kl", "rel99", "sc"):
            a, b = val(w, kb, vb, "kvarn", "all", key), val("built", kb, vb, "kvarn", "all", key)
            if a is None or b is None:
                continue
            out.append(f"| K{kb}/V{vb} | {key} | {a * UNIT[key]:.4f} | {b * UNIT[key]:.4f} | {(b / a - 1) * 100:+.3f}% |")
    return "\n".join(out)


if section in ("all", "main"):
    for key, title in (("rel", "output error, mean (x 1e-3)"), ("kl", "KL, mean (x 1e-4)"), ("rel99", "output error, 99th percentile (x 1e-3)"), ("sc", "score error, mean (x 1e-3 nats)")):
        print(f"\n{title}\n\n{table(key)}")
if section in ("all", "layers"):
    for key, title in (("rel", "output error"), ("kl", "KL")):
        print(f"\nper layer, K4/V4, {title}\n\n{layers_table(key)}")
        print(f"\nby depth and text, K4/V4, {title}\n\n{bins_table(key)}")
if section in ("all", "scopes"):
    print("\n" + scopes_table(layer_scopes(), layer_scopes()))
    print("\n" + scopes_table(["t512-1023", "t1024-2047", "t2048+", "seg:english", "seg:german", "seg:code"], ["t512-1023", "t1024-2047", "t2048+", "english", "german", "code"]))
if section in ("all", "steps"):
    for key, title in (("rel", "output error (x 1e-3)"), ("kl", "KL (x 1e-4)")):
        print(f"\n{title}\n\n{steps_table(key)}")
if section in ("all", "head"):
    print("\n" + head_table())
    n, lines = head_worse()
    print(f"\nthe stored sequence differs from the arrays by more than 1%: {len(lines)} of {n} (pair, scope, measure) combinations")
    for l in lines[:40]:
        print("  " + l)
if section in ("all", "worse"):
    n, lines = worse()
    print(f"\nbuilt worse than rows under the tile policy: {len(lines)} of {n} (pair, scope, measure) combinations")
    for l in lines:
        print("  " + l)
if section in ("all", "snr"):
    print("\n" + snr_table())
if section in ("all", "exact"):
    print("\n" + exact_table())
if section in ("all", "check"):
    print("\n" + check())
