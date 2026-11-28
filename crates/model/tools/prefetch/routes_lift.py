#!/usr/bin/env python3
"""Usage: routes_lift.py TRACE,TRACE... CAPACITY

Lift-scored co-occurrence (P(e'|e)/P(e')) and per-layer miss recall of the token table."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routes_predict import load, L, E, K
paths = sys.argv[1].split(","); cap = int(sys.argv[2]); warm = 300
seqs = [load(p) for p in paths]
toks = np.concatenate([s[0] for s in seqs]); ids = np.concatenate([s[1] for s in seqs])
w = np.concatenate([s[2] for s in seqs]); gen = np.concatenate([s[3] for s in seqs])
T = len(toks); decay = 0.9 ** (1/32)
heat = np.zeros((L,E)); cached = np.zeros((L,E), bool)
D = [1, 2]
cooc = {d: np.zeros((L,E,E), np.float32) for d in D}
cnt = np.zeros((L,E), np.float32)  # times e picked
tok_tab = {}
names = [f"lift-{d}" for d in D] + ["lift1+tok"]
B_list = [5, 10, 20, 40]
bud = {nm: {B: np.zeros(3) for B in B_list} for nm in names}
lay = np.zeros((L, 3))  # per layer: misses, named by toktab@16, named by toktab+lift1 @16
n_t = 0
def norm(a):
    m = a.max(axis=1, keepdims=True); return np.where(m > 0, a / np.maximum(m, 1e-9), 0)
for t in range(T):
    x = toks[t]; picks = ids[t]; pw = w[t]; nonres = ~cached
    if t >= warm:
        n_t += 1
        pm = np.zeros((L,E), bool); np.put_along_axis(pm, picks, True, axis=1); miss = pm & nonres
        freq = (cnt + 1) / (t + 1)
        sc = {}
        for d in D:
            s = np.zeros((L,E), np.float32)
            for l in range(d, L):
                pe = picks[l-d]
                cond = cooc[d][l][pe] / np.maximum(cnt[l-d][pe][:, None], 1)
                s[l] = pw[l-d] @ (cond / freq[l][None, :])
            sc[f"lift-{d}"] = s
        st = tok_tab.get(x, np.zeros((L,E), np.float32))
        sc["lift1+tok"] = norm(sc["lift-1"]) + norm(st)
        for nm in names:
            s = sc[nm]; sn = s / np.maximum(s.max(axis=1, keepdims=True), 1e-9)
            cand = np.where(nonres & (s > 0), sn, -1).ravel(); order = np.argsort(-cand, kind="stable")
            for B in B_list:
                sel = order[:B]; sel = sel[cand[sel] > 0]
                bud[nm][B] += np.array([len(sel), miss.ravel()[sel].sum(), miss.sum()])
        for l in range(L):
            lay[l, 0] += miss[l].sum()
            for j, s in enumerate([st, sc["lift1+tok"]]):
                if s[l].max() > 0:
                    top = np.argsort(-s[l])[:16]; top = top[s[l][top] > 0]
                    lay[l, 1 + j] += miss[l, top].sum()
    heat *= decay
    for l in range(L):
        heat[l, picks[l]] += 1
        for e in picks[l]:
            if not cached[l, e]:
                if cached[l].sum() >= cap:
                    h = np.where(cached[l], heat[l], np.inf); h[picks[l]] = np.inf
                    cached[l, np.argmin(h)] = False
                cached[l, e] = True
    for d in D:
        for l in range(d, L):
            cooc[d][l][np.ix_(picks[l-d], picks[l])] += 1.0
    for l in range(L):
        cnt[l, picks[l]] += 1
    tt = tok_tab.setdefault(x, np.zeros((L,E), np.float32)); tt *= 0.8
    np.put_along_axis(tt, picks, np.take_along_axis(tt, picks, axis=1) + pw, axis=1)
print(f"capacity {cap}: {n_t} tokens")
for nm in names:
    print(nm.ljust(10), " | ".join(f"B={B}: {r/n_t:4.1f} read, {100*h/max(r,1):4.1f}% used, {100*h/max(m,1):4.1f}% of misses" for B,(r,h,m) in bud[nm].items()))
print("per layer: misses a token / toktab@16 share / lift1+tok@16 share")
for l in range(L):
    if l < 6 or l % 8 == 0:
        print(f"  layer {l:2}: {lay[l,0]/n_t:4.2f}  {100*lay[l,1]/max(lay[l,0],1):5.1f}%  {100*lay[l,2]/max(lay[l,0],1):5.1f}%")
