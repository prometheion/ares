#!/usr/bin/env python3
"""Tables for the notes from the results of `kvreal run`."""
import sys
from collections import defaultdict

path = sys.argv[1]
R = {}          # (cand, scope) -> dict
S = {}          # (fam, bits, side, scope) -> db
G = defaultdict(dict)
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if f[0] == "R":
        R[(f[1], f[2])] = dict(n=int(f[3]), rel=float(f[4]), rel99=float(f[5]), kl=float(f[6]), kl99=float(f[7]), sc=float(f[8]), sc99=float(f[9]))
    elif f[0] == "S":
        S[(f[1], int(f[2]), f[3], f[4])] = float(f[5])
    elif f[0] == "G":
        G[f[1]][f[2]] = (int(f[3]), float(f[4]), float(f[5]), float(f[6]))

PAIRS = [(8, 8), (6, 6), (5, 5), (4, 4), (6, 5), (5, 4)]


def name(fam, kb, vb, mask):
    return f"{fam}|K{kb}V{vb}|{mask}"


def get(fam, kb, vb, mask, scope="all", key="rel"):
    r = R.get((name(fam, kb, vb, mask), scope))
    return None if r is None else r[key]


def fmt(v, kind):
    if v is None:
        return "-"
    if kind == "e3":
        return f"{v * 1e3:.2f}"
    if kind == "e4":
        return f"{v * 1e4:.2f}"
    return f"{v:.4g}"


def table(title, rows, key, kind, pairs=PAIRS, scope="all"):
    print(f"\n{title}")
    print("| candidate | " + " | ".join(f"K{k}/V{v}" for k, v in pairs) + " |")
    print("|" + "---|" * (len(pairs) + 1))
    for label, fam, mask in rows:
        vals = [fmt(get(fam, k, v, mask, scope, key), kind) for k, v in pairs]
        print(f"| {label} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "main"
    matrix = []
    for fam, lab in [("absmax-norot", "absmax, no rotation"), ("absmax", "absmax"), ("engine", "engine"), ("kvarn", "kvarn")]:
        for mask, ml in [("none", "nothing exact"), ("policy", "policy defaults"), ("128/128", "first 128 + newest 128"), ("open", "open group only"), ("kvarn", "first 128 + open group"), ("kvarn2", "first 128 + 2 groups")]:
            if (name(fam, 8, 8, mask), "all") in R:
                matrix.append((f"{lab}, {ml}", fam, mask))
    if which == "main":
        table("attention output, relative L2 error, mean (x 1e-3)", matrix, "rel", "e3")
        table("attention output, relative L2 error, 99th percentile (x 1e-3)", matrix, "rel99", "e3")
        table("KL divergence of the attention weights, mean (x 1e-4)", matrix, "kl", "e4")
        table("error of the scaled scores, std over keys, mean (x 1e-3 nats)", matrix, "sc", "e3")

    def rowsets(names, mask):
        return [(lab, fam, mask) for lab, fam in names]

    if which == "whatif":
        names = [("engine (reference)", "engine"), ("kvarn quantizer", "kvarn"),
                 ("engine + static channel scale (first 256 cells)", "engine+scale"),
                 ("engine + static scale^0.5", "engine+scale^0.5"),
                 ("engine + static channel mean", "engine+mean"),
                 ("engine + static mean and scale", "engine+mean+scale"),
                 ("engine + tile normalisation (original domain)", "engine+tiles(orig)"),
                 ("engine + tile normalisation (after rotation, kvarn order)", "engine+tiles(rot)"),
                 ("engine + group offset (midrange)", "engine+group-offset")]
        for mask in ("kvarn", "open"):
            for key, kind, t in [("rel", "e3", "relative L2 (x 1e-3)"), ("kl", "e4", "KL (x 1e-4)")]:
                table(f"what-ifs, exactness '{mask}': {t}", rowsets(names, mask), key, kind)
    if which == "side":
        # K only / V only
        def nm(fam):
            return fam
        names = [("engine", "engine", "engine"), ("kvarn quantizer on K only", "kvarn", "engine"), ("kvarn quantizer on V only", "engine", "kvarn"),
                 ("kvarn on both", "kvarn", "kvarn")]
        for fam in ["engine+mean", "engine+scale^0.5", "engine+tiles(orig)", "engine+group-offset"]:
            names += [(f"{fam} on K only", fam, "engine"), (f"{fam} on V only", "engine", fam), (f"{fam} on both", fam, fam)]
        for key, kind, t in [("rel", "e3", "relative L2 (x 1e-3)"), ("kl", "e4", "KL (x 1e-4)")]:
            print(f"\none side only, exactness 'kvarn': {t}")
            print("| candidate | " + " | ".join(f"K{k}/V{v}" for k, v in PAIRS) + " |")
            print("|" + "---|" * (len(PAIRS) + 1))
            for lab, kf, vf in names:
                fam = kf if kf == vf else f"K:{kf}+V:{vf}"
                vals = [fmt(get(fam, k, v, "kvarn", "all", key), kind) for k, v in PAIRS]
                print(f"| {lab} | " + " | ".join(vals) + " |")
    if which == "layers":
        layers = sorted({s[1] for s in R if s[1].startswith("L")})
        cands = [("absmax", "none"), ("engine", "none"), ("kvarn", "none"), ("engine", "kvarn"), ("kvarn", "kvarn")]
        for kb, vb in [(4, 4), (5, 5), (6, 6)]:
            for key, kind, t in [("rel", "e3", "relative L2 (x 1e-3)"), ("kl", "e4", "KL (x 1e-4)")]:
                print(f"\nper layer, K{kb}/V{vb}: {t}")
                print("| candidate | " + " | ".join(layers) + " | all |")
                print("|" + "---|" * (len(layers) + 2))
                for fam, mask in cands:
                    vals = [fmt(get(fam, kb, vb, mask, L, key), kind) for L in layers + ["all"]]
                    print(f"| {fam}, {mask} | " + " | ".join(vals) + " |")
    if which == "bins":
        scopes = ["t512-1023", "t1024-2047", "t2048+"] + sorted({s[1] for s in R if s[1].startswith("seg:")})
        cands = [("absmax", "none"), ("engine", "none"), ("kvarn", "none"), ("engine", "kvarn"), ("kvarn", "kvarn"), ("engine", "policy"), ("kvarn", "policy")]
        for kb, vb in [(4, 4), (5, 4)]:
            for key, kind, t in [("rel", "e3", "relative L2 (x 1e-3)"), ("kl", "e4", "KL (x 1e-4)")]:
                print(f"\nby depth and text, K{kb}/V{vb}: {t}")
                print("| candidate | " + " | ".join(scopes) + " |")
                print("|" + "---|" * (len(scopes) + 1))
                for fam, mask in cands:
                    vals = [fmt(get(fam, kb, vb, mask, sc, key), kind) for sc in scopes]
                    print(f"| {fam}, {mask} | " + " | ".join(vals) + " |")
    if which == "snr":
        fams = ["absmax-norot", "absmax", "engine", "kvarn", "engine+scale", "engine+scale^0.5", "engine+mean", "engine+mean+scale", "engine+tiles(orig)", "engine+tiles(rot)", "engine+group-offset"]
        for side in ("K", "V"):
            print(f"\nSNR of {side} rows, dB (mean over the layers of the per-layer value; pooled in brackets)")
            print("| codec | 8 | 6 | 5 | 4 |")
            print("|---|---|---|---|---|")
            lay = sorted({s[3] for s in S if s[3].startswith("L")})
            def avg(f, b):
                v = [S[(f, b, side, L)] for L in lay if (f, b, side, L) in S]
                return sum(v) / len(v) if v else float("nan")
            for f in fams:
                print(f"| {f} | " + " | ".join(f"{avg(f, b):.1f} ({S.get((f, b, side, 'all'), float('nan')):.1f})" for b in (8, 6, 5, 4)) + " |")
        layers = sorted({s[3] for s in S if s[3].startswith("L")})
        for side in ("K", "V"):
            for b in (4,):
                print(f"\nSNR of {side} rows at {b} bits per layer, dB")
                print("| codec | " + " | ".join(layers) + " |")
                print("|" + "---|" * (len(layers) + 1))
                for f in ["absmax", "engine", "kvarn", "engine+mean", "engine+scale^0.5"]:
                    print(f"| {f} | " + " | ".join(f"{S.get((f, b, side, L), float('nan')):.1f}" for L in layers) + " |")
    if which == "gt":
        print("| layer | scope | queries | mass on first 128 | mass on newest 128 | entropy (nats) |")
        print("|---|---|---|---|---|---|")
        for L in sorted(G):
            for sc, (n, a, b, e) in G[L].items():
                print(f"| {L} | {sc} | {n} | {a:.3f} | {b:.3f} | {e:.2f} |")
