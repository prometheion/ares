#!/usr/bin/env python3
"""Shares of the gain over absmax/nothing-exact that come from the exact cells and from the quantizer."""
import sys, math
path = sys.argv[1]
src = open(__file__.rsplit('/', 1)[0] + '/tables.py').read().split("if __name__")[0]
exec(src)
for key, label in (("rel", "attention output error"), ("kl", "KL of the attention weights"), ("sc", "score error")):
    print(f"\n{label}")
    print("| pair | absmax, nothing | engine, nothing | kvarn, nothing | absmax, kvarn tail | engine, kvarn tail | kvarn, kvarn tail | engine: exact cells / quantizer | kvarn: exact cells / quantizer |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kb, vb in PAIRS:
        g = lambda f, m: get(f, kb, vb, m, "all", key)
        a0, e0, k0 = g("absmax", "none"), g("engine", "none"), g("kvarn", "none")
        a1, e1, k1 = g("absmax", "kvarn"), g("engine", "kvarn"), g("kvarn", "kvarn")
        def share(x1):
            tot = math.log(a0 / x1)
            ex = math.log(a0 / a1) / tot
            return f"{ex:.0%} / {1 - ex:.0%}"
        f = lambda v: f"{v * (1e4 if key == 'kl' else 1e3):.2f}"
        print(f"| K{kb}/V{vb} | {f(a0)} | {f(e0)} | {f(k0)} | {f(a1)} | {f(e1)} | {f(k1)} | {share(e1)} | {share(k1)} |")
