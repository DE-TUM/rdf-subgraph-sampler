# bgpgen — fast BGP query sampler with in-memory cardinalities

A self-contained C++20 port of the Python samplers in this repository
(`samplers/*_in_memory.py`). It generates star, path, and complex BGP
queries from an N-Triples graph and computes each query's exact
`COUNT(*)` cardinality in memory — no SPARQL endpoint needed. Output
JSON is schema-compatible with the Python tool, so downstream GNCE
training keeps working unchanged.

The speed comes from the same ideas used in FastShacl: dictionary term
encoding (terms are `uint32` ids, segmented IRIs/bnodes/literals),
parallel mmap parsing, per-predicate CSR adjacency with forward and
reverse halves, and compiling templates and queries to flat integer
structures before any sampling or counting runs.

## Build

```sh
cmake -B build            # Release by default; needs CMake >= 3.20, C++20
cmake --build build -j
build/bgpgen_tests        # doctest suite
```

zlib is optional (enables `.nt.gz` input). `-march=native` is on by
default (`-DBGPGEN_NATIVE=OFF` to disable).

## Usage

```sh
# 1000 star queries each of sizes 2, 4, 8, with exact cardinalities
bgpgen --data graph.nt --shape star --size 2 --size 4 --size 8 --num 1000

# Path queries (chains); a path is a chain template of the complex engine
bgpgen --data graph.nt --shape path --size 3 --num 1000

# Complex shapes via templates ("B" = bound node/predicate, "?x/B" =
# bound with probability --p-node, plain "?x" = variable)
bgpgen --data graph.nt --shape complex \
       --template "?c B ?p1, ?c B ?p2, ?x B ?c, ?y B ?c" --num 500

# Just load and print graph stats
bgpgen --data graph.nt.gz
```

See `bgpgen --help` for all flags (probabilities, dedup method, seed,
output dir, thread count, `--timing`).

## What matches the Python tool

- **Sampling semantics**: star/complex/path loop structure, candidate
  shuffling, reshuffle and consecutive-duplicate caps, probability
  draws, IRI-only traversal for path/complex, literal objects emitted
  verbatim (raw NT tokens).
- **Output schema**: same keys and key order per shape, including the
  star metadata fields; JSON formatting is byte-identical to
  `json.dump(..., indent=2)`.
- **Dedup**: `--dedup wl` is a port of
  `samplers/dedup.py`'s WL-1 canonicalization with
  individualize-and-refine, operating on integer terms; `--dedup hash`
  is the predicate-multiset hash. `--dedup symmetry` (the default) gives
  the same result as `wl` without the WL search: all queries of a run come
  from one template, so two of them are isomorphic exactly when a symmetry
  of the template maps one onto the other. The key is the smallest image
  of the query under these symmetries (interchangeable leaves are sorted,
  the remaining symmetries are tried one by one). Queries where this does
  not hold (a template node bound in one triple and variable in another),
  templates with parallel edges or more than 720 remaining symmetries, and
  the star sampler use the WL key. The WL test data
  (`test_canonicalization.py`) is ported in `tests/test_dedup.cpp`,
  plus an exhaustive check over *every* variable permutation of each
  shape.

Known deviations: RNG sequences differ from Python (same distributions,
different streams), and the star `query_hash` field is a 64-bit hash
instead of MD5 (the field is not used downstream).

## Cardinality (y) without an endpoint

`y` is the exact SPARQL `COUNT(*)` — homomorphism semantics over the
deduplicated triple set, the same number an endpoint returns. Per
query, a small counting plan is compiled: variables that occur once are
never enumerated (their extension count is a CSR span length,
multiplied in as a factor — a star is one pass over hub candidates, not
a join), join variables get binding slots, and patterns run
most-bound-first. Unbound-predicate patterns use the subject-major
index, where a node's row length is its extension count. A step budget
(`--card-budget`, default 50M) replaces the old endpoint timeout:
exhausted means `y = -1`. Counting runs in parallel across queries.

Verified three ways: against a brute-force nested-loop counter on small
graphs (300 randomized queries, `tests/test_cardinality.cpp`), the
`y >= 1` invariant for every sampled query on LUBM (5,083 queries, 0
violations), and 30/30 exact matches against a live SPARQL endpoint on
the 21.4M-triple Wikidata graph.

## Performance

Apple Silicon laptop, all cores. "Python" is the in-memory generator in
this repo (no cardinality — it would need an endpoint at 1–3 s per
query; the C++ numbers *include* exact cardinalities).

| | LUBM (1.0M triples, 163 MB) | Wikidata (21.4M triples, 2.5 GB) |
|---|---|---|
| parse + index | 0.3 s | 5.3 s |
| adjacency index | 0.05 s | 8.9 s |
| 1000 star-4 queries | 10 ms | 41 ms |
| ... + 1000 exact counts | +117 ms | +1.7 s |
| 1000 path-3 queries | 12 ms | 87 ms |
| ... + 1000 exact counts | +4 ms | +116 ms |
| Python: parse + 1000 star-4 | — | 64.4 s |
| Python: parse + 1000 path-3 | — | 103.8 s |

## Sampling options for path and complex shapes

The defaults differ from the first C++ version in four ways: the three
below and the degree check described further down. `--v1-compat` switches
all four back and gives the same output as before for the same seed. (The
default duplicate check, `--dedup symmetry`, gives the same output as `wl`.)

- **Early cycle checks** (`--no-early-checks` to disable). A cycle-closing
  edge is checked directly after its second endpoint is matched, not when
  BFS reaches it. Same matches, far fewer candidates on shapes where other
  branches were extended first.
- **A seed per size** (`--shared-seed` to disable). Each size gets a seed
  derived from `--seed` and the size. With one shared seed, every size
  visits the same start nodes in the same order.
- **Pruning on demand** (`--no-auto-prune` to disable). One pass over the
  data computes, for every template node and graph node, how many matches of
  the template's spanning tree exist below it. Nodes with a count of 0 are
  dead ends. The pass is run only once failed attempts have cost as much
  work as the pass itself; it also ends the run at once when no node can
  root the template. `--prune` runs the pass up front.

`--sampling weighted` replaces the DFS: the root is drawn with probability
proportional to count^alpha (`--root-alpha`, default 1) and every neighbour
proportional to its count, without backtracking. A match below root r then
has probability count(r)^(alpha - 1) / Z: with alpha = 1 every match is
equally likely, with alpha = 0 every feasible root is. Draws that use a
graph node twice or miss a cycle-closing edge are rejected, so this mode
suits tree-shaped templates; for templates with a cycle most draws are
rejected. The counts ignore that different template nodes need different
graph nodes.

`--attempt-budget N` (default 0 = off) makes the DFS give up on a start node
after it has looked at N neighbour candidates. Without it, a start node that
has no match can cost up to 50 candidates per step to the power of the
number of steps, because the search goes back one step at a time. On large
graphs with hub nodes this is where bulk sampling spends its time (Wikidata,
7-edge tree, 20,000 queries: 44 s without a budget, 0.3 s with 100). A start
node that was given up is tried again in the next pass.

By default the search skips graph nodes that have fewer different
neighbours than their template node needs: out-neighbours, in-neighbours,
and both together. No match is lost, because different template nodes
always get different graph nodes. The match counts used for pruning ignore
this, so without the check most failed attempts on trees are start nodes
with too few different neighbours. Start nodes are checked during the
first pass; with pruning the check is part of the counts, so it also rules
out nodes whose neighbours all fail it. The order of random draws changes,
so the output for a seed differs from a run with `--no-degree-check`.

### Options that change which queries are sampled (all off by default)

Each of these changes the workload for a seed. They can be combined.

- `--pred-first`: at every step choose a predicate of the node, then a
  neighbour under it. Without it one of the node's edges is chosen, so a
  predicate is chosen in proportion to its edges.
- `--novelty B` (implies `--pred-first`): choose a predicate with weight
  1/(1 + uses)^B, where `uses` counts the queries emitted so far that have
  the predicate at the same template edge. B = 1 is the tested value.
- `--seed-groups`: visit start nodes with different sets of out-predicates
  first (one node per set, then a second node per set, and so on). Without
  it the order is uniformly random, so a set shared by a million nodes is
  visited a million times as often as a rare one. Costs one sort of the
  start nodes per pass.
- `--skip-known`: decide the constants of the query before the search, and
  let the search go on when a match gives a query that is already known.
  `--max-known K` ends the search on a start node after K known queries;
  K = 1 gives up at the first one, as without the option.
- `--per-seed K|auto` (with `--skip-known`): take up to K new queries from
  one visit of a start node. `auto` takes one in the first pass over the
  start nodes and twice as many in every further pass, so a workload that
  one pass can fill is not affected.
- `--retire-seeds` (with `--skip-known`): a start node that gave no new
  query in a pass is left out of later passes.
- `--card-buckets N --card-max X`: the same number of queries in each of N
  ranges of the result size, equally wide on a log scale between 1 and X.
  Every new query is counted when it is found and dropped if its range is
  full; the search on that start node then ends (the queries near a dropped
  one mostly fall into the same range), unless `--max-known` is set.
  Implies `--skip-known`, and `--pred-first` when steering is on. The count
  uses a budget of 2,000,000 steps; a query over it counts as too large.
  The predicate choice is steered towards the ranges that still lack
  queries (`--no-card-steer` to disable): each range learns an exponent g
  and chooses predicates with weight (triples with the predicate)^g.
- `--sample-threads N`: search with N threads. The start nodes are handed
  out in rounds; inside a round a thread does not see what the others find.
  The workload is the same for the same seed and N, and differs between
  values of N.

For bulk sampling the tested combination is
`--novelty 1 --skip-known --max-known 1 --per-seed auto --retire-seeds`.
Searching on after known queries without `--max-known` made bulk sampling
several times slower.

`bench/` holds the experiment drivers for these options
(`eval_improvements.cpp`, `analyze_improvements.py`, `steering_grid.sh`,
`analyze_steering.py`).

## Layout

```
include/bgpgen/, src/   core library (parser, term table, CSR store,
                        subject index, samplers, template engine, DFS
                        matcher, WL dedup, cardinality counter)
cli/main.cpp            the bgpgen binary
tests/                  doctest suite (load, star, template, dedup,
                        cardinality vs brute force)
third_party/doctest     vendored test framework
```
