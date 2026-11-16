#!/bin/bash
# Grid over the sampling mode (dfs, weighted with alpha 0 / 0.5 / 1) and the
# binding probability, one output directory per configuration. The result is
# read by analyze_steering.py.
#
#   bench/steering_grid.sh BGPGEN GRAPH.nt OUT_DIR tree6|path3
set -euo pipefail
BIN=$1; FILE=$2; OUT=$3; TMPL=$4
TREE='?r B ?a, ?r B ?b, ?a B ?c/B, ?a B ?d/B, ?b B ?e/B, ?b B ?f/B'
for p in 0 0.3 0.7 1.0; do
  for mode in dfs a0 a0.5 a1; do
    out=$OUT/${TMPL}_${mode}_p$p; mkdir -p "$out"
    if [ "$mode" = dfs ]; then samp=(--sampling dfs); else samp=(--sampling weighted --root-alpha "${mode#a}"); fi
    if [ "$TMPL" = tree6 ]; then
      "$BIN" --data "$FILE" --shape complex --template "$TREE" --p-node $p "${samp[@]}" \
             --num 1000 --card-budget 5000000 --output-dir "$out" 2>"$out/log.txt"
    else
      "$BIN" --data "$FILE" --shape path --size 3 --p-start-end $p "${samp[@]}" \
             --num 1000 --card-budget 5000000 --output-dir "$out" 2>"$out/log.txt"
    fi
    echo "$TMPL $mode p=$p done"
  done
done
