#!/usr/bin/env python3
"""Tables and figures for the sampling-option experiments.

Reads the JSON-lines files written by bgpgen_eval and prints Markdown tables;
with --figures DIR it also writes two figures (PNG and PDF).

    uv run --with matplotlib --with numpy cpp/bench/analyze_improvements.py \
        results/improvements/*.jsonl --figures figures
"""
import argparse
import json
import statistics as st
from collections import defaultdict

KG_ORDER = ["swdf", "lubm", "wikidata"]
TREE = ["path-2", "path-3", "path-5", "path-8", "in-star-3", "tree-6", "flower-7", "snowflake-8"]
CYCLIC = ["triangle-3", "diamond-4", "triangle-tails-5"]


def load(paths):
    graphs, runs, overlaps = {}, defaultdict(list), []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "graph":
                    graphs[r["kg"]] = r
                elif r["kind"] == "run":
                    runs[(r["kg"], r["template"], r["mode"])].append(r)
                else:
                    overlaps.append(r)
    return graphs, runs, overlaps


def med(rs, fn):
    vals = [fn(r) for r in rs]
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def fmt(v, digits=3):
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}"


def ms_per_1000(r):
    """Sampling time per 1000 unique queries, without the match-weight pass."""
    return 1e6 * r["sample_s"] / r["unique"] if r["unique"] else None


def total_ms_per_1000(r):
    """The same, including the one-off match-weight pass of this run."""
    return 1e6 * (r["sample_s"] + r["weights_s"]) / r["unique"] if r["unique"] else None


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def search_table(runs, kgs, templates, modes, title):
    print(f"### {title}\n")
    header = ["KG", "template", "mode", "unique", "failed attempts", "candidates / query",
              "sampling ms / 1000 queries", "weights (ms, once)", "total ms / 1000 queries"]
    rows = []
    for kg in kgs:
        for t in templates:
            for m in modes:
                rs = runs.get((kg, t, m))
                if not rs:
                    continue
                rows.append([
                    kg, t, m,
                    fmt(med(rs, lambda r: r["unique"])),
                    fmt(med(rs, lambda r: 100 * r["failed"] / r["attempts"] if r["attempts"] else None)) + " %",
                    fmt(med(rs, lambda r: r["expansions"] / r["unique"] if r["unique"] else None)),
                    fmt(med(rs, ms_per_1000)),
                    fmt(med(rs, lambda r: 1000 * r["weights_s"]) if m != "baseline" and m != "early" else None),
                    fmt(med(rs, total_ms_per_1000)),
                ])
    table(header, rows)


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def distribution_table(runs, kgs, templates, modes):
    print("### Distribution of the workload (seed 0)\n")
    header = ["KG", "template", "mode", "queries", "y = 1", "median y", "p90 y", "budget hit",
              "median root out-degree", "p90 root out-degree"]
    rows = []
    for kg in kgs:
        for t in templates:
            for m in modes:
                rs = [r for r in runs.get((kg, t, m), []) if "y" in r]
                if not rs:
                    continue
                r = rs[0]
                ys = [y for y in r["y"] if y >= 0]
                rows.append([
                    kg, t, m, str(len(r["y"])),
                    fmt(100 * sum(y == 1 for y in ys) / len(ys)) + " %" if ys else "–",
                    fmt(quantile(ys, 0.5)), fmt(quantile(ys, 0.9)),
                    fmt(100 * sum(y < 0 for y in r["y"]) / len(r["y"])) + " %",
                    fmt(quantile(r["root_out_degree"], 0.5)),
                    fmt(quantile(r["root_out_degree"], 0.9)),
                ])
    table(header, rows)


def figures(runs, kgs, outdir):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1E293B", "#64748B", "#E2E8F0"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": muted, "ytick.color": muted, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # --- Figure 1: sampling time per 1000 unique queries, by template and mode.
    modes = [("baseline", "Baseline", "#64748B", "o"),
             ("prune", "Pruned DFS", "#2a78d6", "s"),
             ("prune+early", "Pruned DFS, early checks", "#eb6834", "D"),
             ("weighted-a1", "Weighted, α = 1", "#1baf7a", "^")]
    templates = TREE + CYCLIC
    fig, axes = plt.subplots(1, len(kgs), figsize=(3.6 * len(kgs), 3.4), sharey=True, squeeze=False)
    for ax, kg in zip(axes[0], kgs):
        for mode, label, color, marker in modes:
            xs, ys, partial_x, partial_y = [], [], [], []
            for i, t in enumerate(templates):
                rs = runs.get((kg, t, mode))
                if not rs:
                    continue
                v = med(rs, total_ms_per_1000)
                if v is None:
                    continue
                full = med(rs, lambda r: r["unique"]) >= rs[0]["requested"]
                (xs if full else partial_x).append(i)
                (ys if full else partial_y).append(v)
            ax.scatter(xs, ys, s=26, color=color, marker=marker, label=label, zorder=3,
                       edgecolor="white", linewidth=0.6)
            ax.scatter(partial_x, partial_y, s=26, facecolor="white", edgecolor=color,
                       marker=marker, linewidth=1.1, zorder=3)
        ax.axvline(len(TREE) - 0.5, color=grid, linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(templates)))
        ax.set_xticklabels(templates, rotation=55, ha="right")
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(kg, fontsize=10, loc="left")
    axes[0][0].set_ylabel("Time per 1000 unique queries (ms)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.06, "Includes the match-weight pass. Open marker: the run ended with fewer unique queries than requested.",
             ha="center", va="top", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/improvements_sampling_time.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: cardinality distribution (ECDF) per sampling mode.
    dist_modes = [("baseline", "Baseline", "#64748B", "--"),
                  ("weighted-a0", "Weighted, α = 0", "#86b6ef", "-"),
                  ("weighted-a0.5", "Weighted, α = 0.5", "#2a78d6", "-"),
                  ("weighted-a1", "Weighted, α = 1", "#104281", "-")]
    dist_templates = ["path-3", "tree-6"]
    fig, axes = plt.subplots(len(dist_templates), len(kgs), figsize=(3.3 * len(kgs), 4.6),
                             sharey=True, squeeze=False)
    for i, t in enumerate(dist_templates):
        for j, kg in enumerate(kgs):
            ax = axes[i][j]
            for mode, label, color, ls in dist_modes:
                rs = [r for r in runs.get((kg, t, mode), []) if "y" in r]
                if not rs:
                    continue
                ys = sorted(math.log10(y) for y in rs[0]["y"] if y >= 1)
                if not ys:
                    continue
                n = len(rs[0]["y"])  # budget hits stay in the denominator
                ax.step(ys, [(k + 1) / n for k in range(len(ys))], where="post", color=color,
                        linestyle=ls, linewidth=1.6, label=label)
            ax.grid(color=grid, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{kg}, {t}", fontsize=10, loc="left")
            if i == len(dist_templates) - 1:
                ax.set_xlabel("log10 of the cardinality")
        axes[i][0].set_ylabel("Share of queries ≤ x")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/improvements_cardinality_ecdf.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--figures", help="directory for the figures")
    args = ap.parse_args()
    graphs, runs, overlaps = load(args.files)
    kgs = [k for k in KG_ORDER if k in graphs] + [k for k in graphs if k not in KG_ORDER]

    print("### Graphs\n")
    table(["KG", "triples", "predicates", "IRIs", "load (s)"],
          [[k, fmt(graphs[k]["triples"]), fmt(graphs[k]["predicates"]), fmt(graphs[k]["iris"]),
            fmt(graphs[k]["load_s"])] for k in kgs])
    search_table(runs, kgs, TREE, ["baseline", "prune", "weighted-a0", "weighted-a1"],
                 "Tree-shaped templates (median of the seeds)")
    search_table(runs, kgs, CYCLIC,
                 ["baseline", "early", "prune", "prune+early", "weighted-a0", "weighted-a1"],
                 "Templates with a cycle (median of the seeds)")
    distribution_table(runs, kgs, ["path-3", "tree-6", "flower-7"],
                       ["baseline", "prune", "weighted-a0", "weighted-a0.5", "weighted-a1"])

    print("### Root overlap between two path sizes\n")
    table(["KG", "sizes", "seeds", "Jaccard of the root sets", "same root at the same position"],
          [[o["kg"], f"{o['size_a']} and {o['size_b']}",
            "independent" if o["independent"] else "same seed",
            fmt(o["jaccard"]), f"{o['same_position']} of {o['n']}"]
           for o in sorted(overlaps, key=lambda o: (kgs.index(o["kg"]), o["size_a"], o["independent"]))])

    if args.figures:
        figures(runs, kgs, args.figures)


if __name__ == "__main__":
    main()
