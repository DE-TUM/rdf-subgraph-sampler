#!/usr/bin/env python3
"""Old sampler against the new defaults: total run time per (graph, template).

    python3 cpp/bench/compare_v2.py results/improvements/*_v2.jsonl
"""
import collections
import json
import statistics as st
import sys

TEMPLATES = ["path-2", "path-3", "path-5", "path-8", "in-star-3", "tree-6", "flower-7",
             "snowflake-8", "triangle-3", "diamond-4", "triangle-tails-5"]

print("| KG | template | old run (ms) | new run (ms) | new / old | pruning switched on | unique old / new |")
print("|---|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    runs = collections.defaultdict(list)
    kg = None
    for line in open(path):
        r = json.loads(line)
        if r["kind"] == "run":
            kg = r["kg"]
            runs[(r["template"], r["mode"])].append(r)
    for t in TEMPLATES:
        old, new = runs[(t, "baseline")], runs[(t, "v2-default")]
        if not old or not new:
            continue
        total = lambda rs: st.median(1000 * (r["sample_s"] + r["weights_s"]) for r in rs)
        uniq = lambda rs: st.median(r["unique"] for r in rs)
        print(f"| {kg} | {t} | {total(old):.1f} | {total(new):.1f} | {total(new) / total(old):.2f} | "
              f"{sum(r['auto_pruned'] for r in new)}/{len(new)} | {uniq(old):.0f} / {uniq(new):.0f} |")
