#!/usr/bin/env python3
"""Is the root exponent alpha enough to reach a target cardinality distribution?

Input: a directory with one sub-directory per sampler configuration, named
<template>_<mode>_p<binding probability>, each holding one bgpgen output JSON
(mode: dfs, a0, a0.5, a1). Written by bench/steering_grid.sh.

Target: the same number of queries in each decade of the cardinality from
[1, 10) to [10^5, 10^6). For a set of configurations the script computes the
best mixture and its yield: the share of counted candidates that can be kept
when every decade gets the same quota. A yield of 100 % means the candidates
already are uniform over the six decades; 6 x the smallest decade share.

    python3 cpp/bench/analyze_steering.py results/steering/swdf tree6
"""
import glob
import json
import math
import os
import sys

DECADES = 6  # [1,10) ... [1e5,1e6)


def shares(path):
    files = glob.glob(os.path.join(path, "*.json"))
    if not files:
        return None
    ys = [r["y"] for r in json.load(open(files[0]))]
    if not ys:
        return None
    counts = [0] * (DECADES + 1)  # last: >= 1e6 or budget hit
    for y in ys:
        b = DECADES if y < 1 else min(DECADES, int(math.log10(y)))
        counts[b] += 1
    return [c / len(ys) for c in counts], len(ys)


def best_mixture(rows):
    """Maximise the smallest decade share over mixtures of the given rows
    (multiplicative weights on the decades, best response on the rows)."""
    lam = [1.0 / DECADES] * DECADES
    mix = [0.0] * len(rows)
    rounds, eta = 20000, 0.05
    for _ in range(rounds):
        best = max(range(len(rows)), key=lambda i: sum(l * s for l, s in zip(lam, rows[i])))
        mix[best] += 1.0 / rounds
        lam = [l * math.exp(-eta * rows[best][b]) for b, l in enumerate(lam)]
        z = sum(lam)
        lam = [l / z for l in lam]
    combined = [sum(m * r[b] for m, r in zip(mix, rows)) for b in range(DECADES)]
    return DECADES * min(combined), mix


def main():
    root, tmpl = sys.argv[1], sys.argv[2]
    configs = {}
    for d in sorted(glob.glob(os.path.join(root, f"{tmpl}_*_p*"))):
        _, mode, p = os.path.basename(d).rsplit("_", 2)
        res = shares(d)
        if res:
            configs[(mode, float(p[1:]))] = res

    modes = ["dfs", "a0", "a0.5", "a1"]
    ps = sorted({p for _, p in configs})
    print(f"### {os.path.basename(root)}, {tmpl}: share of queries per cardinality decade\n")
    print("| sampling | p(bind) | queries | 1–9 | 10–99 | 10²–10³ | 10³–10⁴ | 10⁴–10⁵ | 10⁵–10⁶ | ≥ 10⁶ or budget | yield alone |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for mode in modes:
        for p in ps:
            if (mode, p) not in configs:
                continue
            s, n = configs[(mode, p)]
            cells = " | ".join(f"{100 * x:.0f} %" for x in s)
            print(f"| {mode} | {p:g} | {n} | {cells} | {100 * DECADES * min(s[:DECADES]):.0f} % |")
    print()

    def report(label, keys):
        keys = [k for k in keys if k in configs]
        if not keys:
            return
        y, mix = best_mixture([configs[k][0][:DECADES] for k in keys])
        used = ", ".join(f"{k[0]}/p={k[1]:g}: {100 * m:.0f} %" for k, m in zip(keys, mix) if m > 0.02)
        print(f"| {label} | {len(keys)} | {100 * y:.0f} % | {used} |")

    print(f"### {os.path.basename(root)}, {tmpl}: yield of the best mixture\n")
    print("| knobs that may vary | configurations | yield | mixture |")
    print("|---|---|---|---|")
    report("none (DFS, p = 0.3)", [("dfs", 0.3)])
    report("α only (p = 0.3)", [(m, 0.3) for m in ["a0", "a0.5", "a1"]])
    report("binding only (DFS)", [("dfs", p) for p in ps])
    report("binding only (α = 0)", [("a0", p) for p in ps])
    report("α and binding", [(m, p) for m in ["a0", "a0.5", "a1"] for p in ps])
    report("everything", [(m, p) for m in modes for p in ps])
    print()


if __name__ == "__main__":
    main()
