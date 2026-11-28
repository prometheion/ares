#!/usr/bin/env python3
"""Usage: routes_slots.py TRACE,TRACE... CAPACITY,CAPACITY

Misses with the same total slots spread over layers by need instead of equally."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routes_predict import load, L, E, K
paths = sys.argv[1].split(","); warm = 300
seqs = [load(p) for p in paths]
ids = np.concatenate([s[1] for s in seqs]); T = len(ids); decay = 0.9 ** (1/32)
def sim(caps):
    heat = np.zeros((L,E)); cached = np.zeros((L,E), bool); m = 0; per = np.zeros(L)
    for t in range(T):
        p = ids[t]
        if t >= warm:
            pm = np.zeros((L,E), bool); np.put_along_axis(pm, p, True, axis=1)
            mm = (pm & ~cached); m += mm.sum(); per += mm.sum(axis=1)
        heat *= decay
        for l in range(L):
            heat[l, p[l]] += 1
            for e in p[l]:
                if not cached[l, e]:
                    if cached[l].sum() >= caps[l]:
                        h = np.where(cached[l], heat[l], np.inf); h[p[l]] = np.inf
                        cached[l, np.argmin(h)] = False
                    cached[l, e] = True
    return m / (T - warm), per / (T - warm)
for cap in [int(c) for c in sys.argv[2].split(",")]:
    total = cap * L
    base, per = sim([cap] * L)
    # need: distinct experts per layer over the first 1000 tokens (a profile)
    u = np.array([len(np.unique(ids[:1000, l])) for l in range(L)], float)
    for a in [1.0, 2.0]:
        wgt = u ** a; caps = np.clip(np.round(total * wgt / wgt.sum()), 32, E).astype(int)
        caps = np.minimum(caps, E); m2, per2 = sim(caps)
        print(f"cap {cap}: equal {base:.1f} misses/token; by need^{a} {m2:.1f} ({100*(m2/base-1):+.0f} %), caps layer 0-3 {caps[:4]}, min {caps.min()} max {caps.max()}, sum {caps.sum()} of {total}")
