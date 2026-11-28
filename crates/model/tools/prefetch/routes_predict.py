#!/usr/bin/env python3
"""Offline precision of expert predictors that need only recorded routes.

Usage: routes_predict.py TRACE,TRACE... CAPACITY[,CAPACITY] [WARM-UP]
(the traces of `q4trace`, read in one sequence; CAPACITY experts a layer; docs/dev/studies/prefetch.md)

Reads Q4ROUTE1 traces (ares q4trace), replays a per-layer expert cache (heat with
decay, admission on a miss, the coldest unpicked member leaves) and asks causal
predictors what they would name for (token t, layer l) before the router of l ran.

Predictors (names as printed):
  prev        picks of token t-1 at l                       (known at step start)
  last4       picks of t-1..t-4 at l, weighted               (step start)
  sametok     picks of the last occurrence of token x_t     (step start; x_t is the input)
  toktab      decayed table (x_t, l) -> experts              (step start, or a step earlier
                                                              for a drafted token)
  bigram      table (x_{t-1}, x_t, l) -> experts             (step start)
  cooc-d      table P(e' at l | e at l-d) of the same token, scored by the picks
              of l-d and their weights                        (after the router of l-d)
  cooc1+tok   cooc-1 plus toktab, normalised scores summed
Metrics per predictor and width n (experts named per layer):
  recall      share of all picks named
  miss-rec    share of the cache's misses (picks it did not hold) named
  miss-prec   share of the named experts the cache did not hold that were picked
              (what a read ahead of them would use)
Budget: per token the best `B` named non-resident experts over all layers (by score
rank), the misses they cover and the reads wasted.
"""
import struct
import sys
import numpy as np

L, E, K = 48, 512, 10


def load(path):
    b = open(path, "rb").read()
    assert b[:8] == b"Q4ROUTE1"
    v, l, e, k, T, P, size = struct.unpack_from("<7I", b, 8)
    assert (l, e, k) == (L, E, K)
    toks = np.frombuffer(b, dtype="<u4", count=T, offset=64).astype(np.int64)
    base = 64 + 4 * T
    rec = np.frombuffer(b, dtype=np.uint8, count=T * L * size, offset=base).reshape(T, L, size)
    ids = rec[:, :, : 2 * K].copy().view("<u2").reshape(T, L, K).astype(np.int64)
    w = rec[:, :, 2 * K :].copy().view("<f4").reshape(T, L, K).astype(np.float32)
    gen = np.zeros(T, dtype=bool)
    gen[P:] = True
    return toks, ids, w, gen


def main():
    paths = sys.argv[1].split(",")
    caps = [int(c) for c in sys.argv[2].split(",")]
    warm = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    budgets = [5, 10, 20, 40]
    widths = [10, 16, 32]
    seqs = [load(p) for p in paths]
    toks = np.concatenate([s[0] for s in seqs])
    ids = np.concatenate([s[1] for s in seqs])
    w = np.concatenate([s[2] for s in seqs])
    gen = np.concatenate([s[3] for s in seqs])
    T = len(toks)
    print(f"{T} tokens ({gen.sum()} generated) from {paths}; warm-up {warm}")
    decay = 0.9 ** (1 / 32)
    for cap in caps:
        run(cap, toks, ids, w, gen, warm, decay, widths, budgets)


def topn(score, n):
    idx = np.argpartition(-score, n)[:n]
    return idx[np.argsort(-score[idx], kind="stable")]


def run(cap, toks, ids, w, gen, warm, decay, widths, budgets):
    T = len(toks)
    heat = np.zeros((L, E), dtype=np.float64)
    cached = np.zeros((L, E), dtype=bool)
    # Tables, learned online from the tokens before t.
    D = [1, 2, 3, 4]
    cooc = {d: np.zeros((L, E, E), dtype=np.float32) for d in D}
    tok_tab = {}  # token -> (L, E) decayed counts
    big_tab = {}  # (tok_prev, tok) -> (L, E)
    last_seen = {}  # token -> t
    names = ["prev", "last4", "sametok", "toktab", "bigram"] + [f"cooc-{d}" for d in D] + [
        "cooc1+tok", "cooc2+tok"]
    # tallies[name][width] = [picks, named_picks, misses, named_misses, named_nonres, named_nonres_picked]
    tallies = {g: {nm: {n: np.zeros(6) for n in widths} for nm in names} for g in ("all", "gen")}
    bud = {g: {nm: {B: np.zeros(3) for B in budgets} for nm in names} for g in ("all", "gen")}
    miss_count = {"all": [0, 0], "gen": [0, 0]}  # misses, tokens
    per_layer_miss = np.zeros(L)
    for t in range(T):
        x = toks[t]
        measure = t >= warm
        picks = ids[t]
        pw = w[t]
        nonres = ~cached
        scores = {}
        if measure:
            # step-start predictors
            s_prev = np.zeros((L, E), np.float32)
            if t >= 1:
                np.put_along_axis(s_prev, ids[t - 1], w[t - 1], axis=1)
            s_last4 = np.zeros((L, E), np.float32)
            for j in range(1, 5):
                if t - j >= 0:
                    tmp = np.zeros((L, E), np.float32)
                    np.put_along_axis(tmp, ids[t - j], w[t - j] * (0.7 ** (j - 1)), axis=1)
                    s_last4 += tmp
            s_same = np.zeros((L, E), np.float32)
            if x in last_seen:
                tp = last_seen[x]
                np.put_along_axis(s_same, ids[tp], w[tp], axis=1)
            s_tok = tok_tab.get(x, np.zeros((L, E), np.float32))
            s_big = big_tab.get((toks[t - 1] if t else -1, x), np.zeros((L, E), np.float32))
            scores.update(prev=s_prev, last4=s_last4, sametok=s_same, toktab=s_tok, bigram=s_big)
            for d in D:
                s = np.zeros((L, E), np.float32)
                for l in range(d, L):
                    s[l] = pw[l - d] @ cooc[d][l][picks[l - d]]
                scores[f"cooc-{d}"] = s

            def norm(a):
                m = a.max(axis=1, keepdims=True)
                return np.where(m > 0, a / np.maximum(m, 1e-9), 0)

            scores["cooc1+tok"] = norm(scores["cooc-1"]) + 0.5 * norm(s_tok)
            scores["cooc2+tok"] = norm(scores["cooc-2"]) + 0.5 * norm(s_tok)
            # misses of this token
            pm = np.zeros((L, E), bool)
            np.put_along_axis(pm, picks, True, axis=1)
            miss = pm & nonres
            groups = ["all"] + (["gen"] if gen[t] else [])
            for g in groups:
                miss_count[g][0] += miss.sum()
                miss_count[g][1] += 1
            per_layer_miss += miss.sum(axis=1)
            for nm in names:
                s = scores[nm]
                for n in widths:
                    named = np.zeros((L, E), bool)
                    for l in range(L):
                        if s[l].max() <= 0:
                            continue
                        top = topn(s[l], n)
                        top = top[s[l][top] > 0]
                        named[l, top] = True
                    v = np.array([pm.sum(), (named & pm).sum(), miss.sum(), (named & miss).sum(),
                                  (named & nonres).sum(), (named & nonres & pm).sum()])
                    for g in groups:
                        tallies[g][nm][n] += v
                # budget: the best B non-resident named experts over all layers by
                # (score normalised per layer)
                sn = s / np.maximum(s.max(axis=1, keepdims=True), 1e-9)
                cand = np.where(nonres & (s > 0), sn, -1).ravel()
                order = np.argsort(-cand, kind="stable")
                for B in budgets:
                    sel = order[:B]
                    sel = sel[cand[sel] > 0]
                    hit = miss.ravel()[sel].sum()
                    for g in groups:
                        bud[g][nm][B] += np.array([len(sel), hit, miss.sum()])
        # update cache: heat, admission
        heat *= decay
        for l in range(L):
            heat[l, picks[l]] += 1.0
            for e in picks[l]:
                if not cached[l, e]:
                    if cached[l].sum() >= cap:
                        h = np.where(cached[l], heat[l], np.inf)
                        h[picks[l]] = np.inf
                        cached[l, np.argmin(h)] = False
                    cached[l, e] = True
        # update tables
        for d in D:
            for l in range(d, L):
                c = cooc[d][l]
                c[np.ix_(picks[l - d], picks[l])] += pw[l - d][:, None] * np.ones((1, K), np.float32)
        tt = tok_tab.setdefault(x, np.zeros((L, E), np.float32))
        tt *= 0.8
        np.put_along_axis(tt, picks, np.take_along_axis(tt, picks, axis=1) + pw, axis=1)
        key = (toks[t - 1] if t else -1, x)
        bt = big_tab.setdefault(key, np.zeros((L, E), np.float32))
        bt *= 0.8
        np.put_along_axis(bt, picks, np.take_along_axis(bt, picks, axis=1) + pw, axis=1)
        last_seen[x] = t
    # normalise cooc rows lazily: P(e'|e) ~ counts / count(e); we used raw weighted counts,
    # which favours frequent e'. Good enough for a ranking per token (the sum over picks).
    for g in ("all", "gen"):
        m, n_t = miss_count[g]
        if n_t == 0:
            continue
        print(f"\n=== capacity {cap} of {E} a layer, {g} tokens: {n_t}, misses a token {m / n_t:.1f} "
              f"of {L * K} ({100 * m / n_t / (L * K):.1f} %)")
        print(f"{'predictor':<10} {'n':>3} {'recall':>7} {'miss-rec':>8} {'miss-prec':>9} "
              f"{'named nonres/tok':>16}")
        for nm in names:
            for n in widths:
                a = tallies[g][nm][n]
                print(f"{nm:<10} {n:>3} {100 * a[1] / a[0]:6.1f}% {100 * a[3] / max(a[2], 1):7.1f}% "
                      f"{100 * a[5] / max(a[4], 1):8.1f}% {a[4] / n_t:16.1f}")
        print("budget: best B non-resident named experts a token over all layers -> "
              "reads, used (share), misses covered (share)")
        for nm in names:
            line = []
            for B in budgets:
                r, h, mm = bud[g][nm][B]
                line.append(f"B={B}: {r / n_t:4.1f} read, {100 * h / max(r, 1):4.1f}% used, "
                            f"{100 * h / max(mm, 1):4.1f}% of misses")
            print(f"{nm:<10} " + " | ".join(line))
    print("misses per layer (all measured tokens):", " ".join(f"{v / max(1, miss_count['all'][1]):.1f}"
                                                         for v in per_layer_miss))


if __name__ == "__main__":
    main()
